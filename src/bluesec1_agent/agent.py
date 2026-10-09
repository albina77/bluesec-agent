import asyncio
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from bluesec1_agent.llm import StructuredStepClient, render_json
from bluesec1_agent.models import FinishInvestigationRequest
from bluesec1_agent.state import EvidenceLedger
from bluesec1_agent.trace import JsonlTraceWriter
from bluesec1_client import (
    RemoteBenchmarkClient,
    RemoteBenchmarkSession,
    TaskResult,
    ToolCallResult,
)

SYSTEM_PROMPT = """
You are a pragmatic senior SOC analyst investigating one security alert.

Return exactly one JSON object matching the supplied NextStep schema on every turn.
Do not emit prose, markdown, XML tags, or extra keys outside that object.

The task observation contains `available_tools`, the authoritative catalog of
enabled tool calls and their argument schemas. Follow that catalog exactly.

Investigation rules:
- Treat the alert as an initial hypothesis, not proof that the activity is malicious.
- Use only tools advertised in the current task observation.
- Execute exactly one tool per step and use its evidence before choosing the next step.
- Start from the entity ids named in the alert, then expand through their relations.
- Prefer following relations over searching again: an entity you already hold tells
  you where to go next.
- Search when you have a concrete string worth looking for and no edge leading to it.
- Do not repeat an identical unhelpful or failed call.
- Use only entity_id values actually returned by the alert or by a tool.
- Submit only response artifacts supported by evidence. Do not invent entity ids.
- Aim to finish in 6-10 total tool calls. Exceed that only when a concrete unresolved
  evidence gap can change the verdict or add a necessary response artifact.
- Keep current_state evidence-focused and the remaining plan short.
- Set task_completed=true exactly when selecting finish_investigation.
- Treat null or missing properties as unknown, never as negative evidence.
- A benign verdict requires positive legitimacy evidence; absence of malicious
  evidence is insufficient.
- Every final artifact or evidence anchor must use an id and property actually observed.
- Before a malicious report, consider the affected host, identity, persistence, file,
  network destination, and executed process. Include only supported, actionable items;
  do not pad the report. For benign reports, cite the strongest positive legitimacy
  properties and relations. State material unknowns without turning them into facts.
- Finish before the investigation step budget is exhausted.
""".strip()


class AgentRunResult(BaseModel):
    """Aggregate results returned by one remote benchmark run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    successful_tasks: int
    failed_tasks: int
    task_results: list[TaskResult] = Field(default_factory=list)


class SimpleSGRAgent:
    """Run a typed one-action-per-turn LLM investigation loop."""

    def __init__(
        self,
        llm: StructuredStepClient,
        *,
        max_steps: int = 14,
        max_task_seconds: float = 240.0,
        max_tasks: int | None = None,
        trace_dir: Path | None = None,
    ) -> None:
        """Store the LLM dependency and per-task step limit.

        Args:
            llm: Structured step generator used for every investigation turn.
            max_steps: Maximum total tool calls allowed for one task.
            max_tasks: Optional number of tasks to consume before closing the run.

        Raises:
            ValueError: If the configured limit is not positive.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if max_task_seconds <= 0:
            raise ValueError("max_task_seconds must be positive")
        if max_tasks is not None and max_tasks < 1:
            raise ValueError("max_tasks must be positive when provided")
        self.llm = llm
        self.max_steps = max_steps
        self.max_task_seconds = max_task_seconds
        self.max_tasks = max_tasks
        self.trace_dir = trace_dir
        self._trace: JsonlTraceWriter | None = None

    async def run(self, client: RemoteBenchmarkClient) -> AgentRunResult:
        """Consume the remote queue and aggregate authoritative task results.

        Args:
            client: Open remote benchmark client that owns the current run.

        Returns:
            Aggregate result for the newly created remote run.

        Raises:
            RuntimeError: If the client is not open.
        """
        remote_run = client.run
        if remote_run is None:
            raise RuntimeError("RemoteBenchmarkClient must be open before running the agent.")

        if self.trace_dir is not None:
            self._trace = JsonlTraceWriter(self.trace_dir, remote_run.run_id)
            self._write_trace("run_started", run_id=remote_run.run_id)

        task_results: list[TaskResult] = []
        while (session := await client.start_task()) is not None:
            logger.info("Starting structured investigation for task {}.", session.task_id)
            self._write_trace("task_started", task_id=session.task_id)
            try:
                async with asyncio.timeout(self.max_task_seconds):
                    task_result = await self._investigate(session)
            except Exception as exc:
                logger.error("Task {} failed in the agent: {}", session.task_id, exc)
                error_message = f"{type(exc).__name__}: {exc}"[:2_000]
                aborted_result = await session.abort_task(error_message=error_message)
                task_result = TaskResult.model_validate(aborted_result)
                self._write_trace(
                    "task_error",
                    task_id=session.task_id,
                    error=error_message,
                )
                if self._is_fatal_provider_error(exc):
                    self._write_trace("task_result", result=task_result.model_dump(mode="json"))
                    raise
            task_results.append(task_result)
            self._write_trace("task_result", result=task_result.model_dump(mode="json"))
            if self.max_tasks is not None and len(task_results) >= self.max_tasks:
                logger.info("Reached configured run limit of {} task(s).", self.max_tasks)
                break

        successful_tasks = sum(
            result.completion_reason in {"terminated", "truncated"} for result in task_results
        )
        result = AgentRunResult(
            run_id=remote_run.run_id,
            successful_tasks=successful_tasks,
            failed_tasks=len(task_results) - successful_tasks,
            task_results=task_results,
        )
        self._write_trace("run_finished", result=result.model_dump(mode="json"))
        return result

    async def _investigate(self, session: RemoteBenchmarkSession) -> TaskResult:
        """Drive one leased task until its terminal investigation tool succeeds.

        Args:
            session: Active remote task session.

        Returns:
            Authoritative terminal task result returned by the server.

        Raises:
            ValueError: If the task observation has an invalid tool catalog.
            RuntimeError: If the agent exhausts its step budget.
        """
        observation = session.task.observation
        tool_specs = self._tool_specs(observation)
        available_tools = {
            specification["name"]
            for specification in tool_specs
            if isinstance(specification.get("name"), str)
        }
        if "finish_investigation" not in available_tools:
            raise ValueError("The runtime did not advertise finish_investigation.")

        public_observation = dict(observation)
        public_observation.pop("available_tools", None)
        public_observation = self._decode_json_value(public_observation)
        ledger = EvidenceLedger()
        ledger.ingest(public_observation)
        calls_made: list[dict[str, Any]] = []
        call_fingerprints: set[str] = set()
        current_state = "Only the initial alert has been reviewed."
        last_feedback: dict[str, Any] | None = None
        started_at = time.monotonic()

        for step_index in range(1, self.max_steps + 1):
            elapsed_seconds = time.monotonic() - started_at
            steps_remaining = self.max_steps - step_index + 1
            seconds_remaining = max(0.0, self.max_task_seconds - elapsed_seconds)
            messages = self._step_messages(
                public_observation=public_observation,
                tool_specs=tool_specs,
                ledger=ledger,
                calls_made=calls_made,
                current_state=current_state,
                last_feedback=last_feedback,
                step_index=step_index,
                steps_remaining=steps_remaining,
                seconds_remaining=seconds_remaining,
            )
            step = await self.llm.next_step(messages)
            current_state = step.current_state
            function = step.function
            validation_errors: list[str] = []
            normalization_notes: list[str] = []
            if isinstance(function, FinishInvestigationRequest):
                function, validation_errors, normalization_notes = ledger.prepare_finish(function)
                if normalization_notes:
                    logger.info("Normalized final report: {}", normalization_notes)
            tool_name = function.tool
            arguments = function.model_dump(
                mode="json",
                by_alias=True,
                exclude={"tool"},
                exclude_none=True,
            )
            tool_call_id = f"step_{step_index}"
            self._write_trace(
                "llm_step",
                task_id=session.task_id,
                step=step_index,
                current_state=step.current_state,
                plan=step.plan_remaining_steps_brief,
                tool_name=tool_name,
                arguments=arguments,
                normalization_notes=normalization_notes,
                usage=getattr(self.llm, "last_usage", None),
            )

            if tool_name not in available_tools:
                feedback: dict[str, Any] = {
                    "status": "error",
                    "message": (
                        f"Tool {tool_name!r} is not advertised for this task. "
                        f"Available tools: {sorted(available_tools)}"
                    ),
                }
                logger.warning("Rejected unavailable tool {} locally.", tool_name)
            elif isinstance(function, FinishInvestigationRequest) and validation_errors:
                feedback = {
                    "tool_name": tool_name,
                    "status": "error",
                    "message": "Final report failed local evidence validation.",
                    "validation_errors": validation_errors,
                }
                logger.warning("Rejected unsupported final report: {}", validation_errors)
            else:
                fingerprint = self._call_fingerprint(tool_name, arguments)
                if tool_name != "finish_investigation" and fingerprint in call_fingerprints:
                    feedback = {
                        "tool_name": tool_name,
                        "status": "error",
                        "message": (
                            "Identical call already executed; choose a different evidence gap."
                        ),
                    }
                    logger.warning("Rejected duplicate tool call {} locally.", tool_name)
                else:
                    call_fingerprints.add(fingerprint)
                    response = await session.call_tool(tool_name, arguments)
                    feedback = self._tool_feedback(response)
                    ledger.ingest(feedback.get("result"))
                    logger.info(
                        "Tool {} returned status {}.",
                        tool_name,
                        feedback.get("status", "unknown"),
                    )
                    self._write_trace(
                        "tool_result",
                        task_id=session.task_id,
                        step=step_index,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        feedback=feedback,
                    )
                    if response.task_result is not None:
                        return TaskResult.model_validate(response.task_result)

            calls_made.append(
                {
                    "step": step_index,
                    "tool": tool_name,
                    "arguments": self._compact_arguments(arguments),
                    "status": feedback.get("status", "unknown"),
                    "reward": feedback.get("reward"),
                }
            )
            last_feedback = feedback
            if feedback.get("status") == "error":
                self._write_trace(
                    "local_rejection",
                    task_id=session.task_id,
                    step=step_index,
                    tool_name=tool_name,
                    feedback=feedback,
                )

        raise RuntimeError(
            f"Agent hit the {self.max_steps}-step limit without finish_investigation."
        )

    @staticmethod
    def _step_messages(
        *,
        public_observation: dict[str, Any],
        tool_specs: list[dict[str, Any]],
        ledger: EvidenceLedger,
        calls_made: list[dict[str, Any]],
        current_state: str,
        last_feedback: dict[str, Any] | None,
        step_index: int,
        steps_remaining: int,
        seconds_remaining: float,
    ) -> list[dict[str, Any]]:
        urgency = ""
        if steps_remaining == 1:
            urgency = "This is the final step. You MUST call finish_investigation now."
        elif steps_remaining <= 3:
            urgency = (
                "Finish now unless exactly one missing fact can materially change the verdict "
                "or required report artifacts."
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": render_json(
                    {
                        "instruction": (
                            "Investigate the alert and submit an evidence-backed report."
                        ),
                        "initial_observation": public_observation,
                        "available_tools": tool_specs,
                        "investigation_state": {
                            "current_state": current_state,
                            "evidence_ledger": ledger.snapshot(),
                            "calls_made": calls_made,
                            "last_tool_feedback": last_feedback,
                        },
                        "budget": {
                            "current_step": step_index,
                            "steps_remaining_including_this_one": steps_remaining,
                            "seconds_remaining_approx": round(seconds_remaining, 1),
                            "instruction": urgency,
                        },
                    }
                ),
            },
        ]

    @staticmethod
    def _call_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
        semantic_arguments = {key: value for key, value in arguments.items() if key != "reasoning"}
        return f"{tool_name}:{json.dumps(semantic_arguments, sort_keys=True, default=str)}"

    @staticmethod
    def _compact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in arguments.items() if key != "reasoning"}

    @staticmethod
    def _is_fatal_provider_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code in {401, 403, 429}:
            return True
        if isinstance(status_code, int) and status_code >= 500:
            return True
        provider_transport_errors = {"APIConnectionError", "APITimeoutError"}
        return any(
            error_type.__name__ in provider_transport_errors
            for error_type in type(error).__mro__
        )

    def _write_trace(self, event: str, **payload: Any) -> None:
        if self._trace is not None:
            self._trace.write(event, **payload)

    @staticmethod
    def _tool_specs(observation: dict[str, Any]) -> list[dict[str, Any]]:
        """Decode the runtime tool catalog from the current observation.

        Args:
            observation: Current public task observation.

        Returns:
            Advertised tool specifications.

        Raises:
            ValueError: If `available_tools` is absent or malformed.
        """
        raw_tool_specs = observation.get("available_tools")
        if not isinstance(raw_tool_specs, str):
            raise ValueError("Observation has no JSON-encoded available_tools field.")
        try:
            tool_specs = json.loads(raw_tool_specs)
        except json.JSONDecodeError as exc:
            raise ValueError("available_tools is not valid JSON.") from exc
        if not isinstance(tool_specs, list) or not all(
            isinstance(specification, dict) for specification in tool_specs
        ):
            raise ValueError("available_tools must encode a list of tool specifications.")
        return [dict(specification) for specification in tool_specs]

    @staticmethod
    def _tool_feedback(response: ToolCallResult) -> dict[str, Any]:
        """Reduce a remote tool response to evidence needed by the LLM.

        Args:
            response: Server tool-call response.

        Returns:
            Compact evidence and lifecycle flags.
        """
        result = response.info.get("tool_result")
        if result is None:
            result = response.observation.get("last_tool_result", response.observation)
        return {
            "tool_name": response.tool_name,
            "status": response.info.get("tool_status", "unknown"),
            "result": SimpleSGRAgent._decode_json_value(result),
            "error": response.info.get("error"),
            "reward": response.reward,
            "terminated": response.terminated,
            "truncated": response.truncated,
        }

    @staticmethod
    def _decode_json_value(value: Any) -> Any:
        """Decode runtime fields that carry structured JSON as strings."""
        if not isinstance(value, str):
            if isinstance(value, dict):
                return {
                    key: SimpleSGRAgent._decode_json_value(child)
                    for key, child in value.items()
                }
            if isinstance(value, list):
                return [SimpleSGRAgent._decode_json_value(child) for child in value]
            return value
        stripped = value.strip()
        if not stripped.startswith(("{", "[")):
            return value
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return value
        return SimpleSGRAgent._decode_json_value(decoded)
