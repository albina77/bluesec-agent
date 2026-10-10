import asyncio
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from bluesec1_agent.llm import StructuredStepClient, render_json
from bluesec1_agent.models import FinishInvestigationRequest, NextStep
from bluesec1_agent.state import EvidenceLedger
from bluesec1_agent.trace import JsonlTraceWriter, RunArtifacts
from bluesec1_client import (
    RemoteBenchmarkClient,
    RemoteBenchmarkSession,
    TaskCapacityExceededError,
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
- Keep two competing hypotheses in current_state: the strongest malicious explanation
  and the strongest benign explanation. Choose the next call that best separates them.
- Treat analyst_attention as observed behavior requiring immediate resolution. Stay on
  that entity's branch; do not wander into unrelated host activity. Broad discovery plus
  VM or sandbox detection is strong evidence of malicious intent unless directly refuted.
- Search when you have a concrete string worth looking for and no edge leading to it.
- Do not repeat an identical unhelpful or failed call.
- Use only entity_id values actually returned by the alert or by a tool.
- Submit only response artifacts supported by evidence. Do not invent entity ids.
- Tool-call efficiency is not penalized while calls stay within the scenario's hidden
  budget. Do not stop merely to hit an arbitrary call count. Continue while a concrete
  evidence gap can change the verdict, establish causality, or add a necessary artifact.
- Stop when both hypotheses have been tested and the verdict, causal chain, actor,
  affected assets, and report artifacts are supported. Do not exhaust the graph after
  those report-critical questions are answered.
- Keep current_state evidence-focused and the remaining plan short.
- Set task_completed=true exactly when selecting finish_investigation.
- Treat null or missing properties as unknown, never as negative evidence.
- A benign verdict requires positive legitimacy evidence; absence of malicious
  evidence is insufficient.
- Signed Microsoft binaries and common administrative tools do not make their commands
  benign. Judge intent from arguments, scripts, relations, actor, and sequence.
- Never cite a null-valued property as evidence and never describe an unknown signature,
  parent, timestamp, reputation, or user as confirmed.
- Every final artifact or evidence anchor must use an id and property actually observed.
- Before a malicious report, consider the affected host, identity, persistence, file,
  network destination, and executed process. Include only supported, actionable items;
  do not pad the report. For benign reports, cite the strongest positive legitimacy
  properties and relations. State material unknowns without turning them into facts.
- A benign report should normally cite a causal relation plus the strongest non-null
  entity properties that establish an expected actor, binary, command, and sequence.
- If the last feedback has status review_required, audit every claim against the
  evidence ledger, correct the candidate, and return finish_investigation again.
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
        max_steps: int = 24,
        max_task_seconds: float = 240.0,
        max_tasks: int | None = None,
        review_final_report: bool = True,
        task_concurrency: int = 1,
        analysis_parallelism: int = 1,
        trace_dir: Path | None = None,
    ) -> None:
        """Store the LLM dependency and per-task step limit.

        Args:
            llm: Structured step generator used for every investigation turn.
            max_steps: Maximum total tool calls allowed for one task.
            max_tasks: Optional number of tasks to consume before closing the run.
            review_final_report: Require one evidence audit before remote submission.
            task_concurrency: Maximum independently investigated tasks in flight.
            analysis_parallelism: Number of parallel hypothesis candidates at key decisions.

        Raises:
            ValueError: If the configured limit is not positive.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if max_task_seconds <= 0:
            raise ValueError("max_task_seconds must be positive")
        if max_tasks is not None and max_tasks < 1:
            raise ValueError("max_tasks must be positive when provided")
        if task_concurrency < 1:
            raise ValueError("task_concurrency must be positive")
        if analysis_parallelism < 1:
            raise ValueError("analysis_parallelism must be positive")
        self.llm = llm
        self.max_steps = max_steps
        self.max_task_seconds = max_task_seconds
        self.max_tasks = max_tasks
        self.review_final_report = review_final_report
        self.task_concurrency = task_concurrency
        self.analysis_parallelism = analysis_parallelism
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
            self._trace = RunArtifacts(
                self.trace_dir,
                remote_run.run_id,
                manifest={
                    "schema_version": 1,
                    "run_id": remote_run.run_id,
                    "model": getattr(self.llm, "model", None),
                    "prompt_version": "evidence-ledger-v1",
                    "profile": "candidate",
                },
            )
            self._write_trace("run_started", run_id=remote_run.run_id)

        task_results: list[TaskResult] = []
        queue_exhausted = False
        while not queue_exhausted:
            remaining = (
                self.task_concurrency
                if self.max_tasks is None
                else min(self.task_concurrency, self.max_tasks - len(task_results))
            )
            if remaining <= 0:
                logger.info("Reached configured run limit of {} task(s).", self.max_tasks)
                break

            sessions: list[RemoteBenchmarkSession] = []
            for _ in range(remaining):
                try:
                    session = await client.start_task()
                except TaskCapacityExceededError as exc:
                    logger.info(
                        "Runtime active-task limit reached at concurrency {}; "
                        "processing the leased batch first.",
                        len(sessions),
                    )
                    if not sessions:
                        raise exc
                    break
                if session is None:
                    queue_exhausted = True
                    break
                sessions.append(session)

            if not sessions:
                break
            batch_results = await asyncio.gather(
                *(self._run_task(session) for session in sessions)
            )
            task_results.extend(batch_results)

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
        if isinstance(self._trace, RunArtifacts):
            self._trace.close("completed", task_count=len(task_results))
        return result

    async def _run_task(self, session: RemoteBenchmarkSession) -> TaskResult:
        """Investigate one leased session and convert task-local failures to aborts."""
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
        self._write_trace("task_result", result=task_result.model_dump(mode="json"))
        return task_result

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
        call_cache: dict[str, dict[str, Any]] = {}
        current_state = "Only the initial alert has been reviewed."
        last_feedback: dict[str, Any] | None = None
        started_at = time.monotonic()
        remote_calls = 0
        max_decisions = self.max_steps + 10
        final_review_complete = not self.review_final_report

        for decision_index in range(1, max_decisions + 1):
            elapsed_seconds = time.monotonic() - started_at
            calls_remaining = self.max_steps - remote_calls
            seconds_remaining = max(0.0, self.max_task_seconds - elapsed_seconds)
            messages = self._step_messages(
                public_observation=public_observation,
                tool_specs=tool_specs,
                ledger=ledger,
                calls_made=calls_made,
                current_state=current_state,
                last_feedback=last_feedback,
                decision_index=decision_index,
                calls_remaining=calls_remaining,
                seconds_remaining=seconds_remaining,
            )
            parallel_roles: tuple[str, ...] = ()
            if decision_index == 1:
                parallel_roles = (
                    "Argue the strongest malicious hypothesis and choose the single tool call "
                    "that can confirm or refute it most efficiently.",
                    "Argue the strongest benign hypothesis using positive evidence and choose "
                    "the single tool call that best distinguishes it from an attack.",
                )
            elif last_feedback and last_feedback.get("status") == "review_required":
                parallel_roles = (
                    "Audit verdict correctness and identify unsupported or contradictory claims.",
                    "Audit artifact and legitimacy-evidence completeness, IDs, fields, and "
                    "causal relations.",
                )
            step = await self._next_step_with_candidates(
                messages,
                roles=parallel_roles,
                task_id=session.task_id,
                decision_index=decision_index,
            )
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
            fingerprint = self._call_fingerprint(tool_name, arguments)
            tool_call_id = f"decision_{decision_index}"
            server_call = False
            self._write_trace(
                "llm_step",
                task_id=session.task_id,
                step=decision_index,
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
            elif isinstance(function, FinishInvestigationRequest) and not final_review_complete:
                final_review_complete = True
                audit_findings = self._report_audit_findings(function)
                feedback = {
                    "tool_name": tool_name,
                    "status": "review_required",
                    "candidate_submission": arguments["submission"],
                    "audit_findings": audit_findings,
                    "message": (
                        "Audit this candidate before submission. Remove claims supported only "
                        "by null or missing values. Check verdict consistency, causal relations, "
                        "actor, command, timeline, and response artifacts. For benign, cite "
                        "positive legitimacy evidence rather than absence of malicious evidence. "
                        "Return a corrected finish_investigation next unless one specific missing "
                        "fact can materially change the verdict."
                    ),
                }
                self._write_trace(
                    "final_review",
                    task_id=session.task_id,
                    step=decision_index,
                    candidate_submission=arguments["submission"],
                )
            else:
                if tool_name != "finish_investigation" and fingerprint in call_cache:
                    cached = call_cache[fingerprint]
                    feedback = {
                        "tool_name": tool_name,
                        "status": "cached",
                        "reward": cached.get("reward"),
                        "message": (
                            "This exact call already ran and its evidence is in the ledger. "
                            "Do not request it again. Follow an unexplored relation, test the "
                            "opposing hypothesis, or finish now."
                        ),
                    }
                    logger.info("Served duplicate {} from local cache.", tool_name)
                    self._write_trace(
                        "cache_hit",
                        task_id=session.task_id,
                        step=decision_index,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                elif tool_name != "finish_investigation" and calls_remaining <= 1:
                    feedback = {
                        "tool_name": tool_name,
                        "status": "error",
                        "message": (
                            "The final server call is reserved for finish_investigation. "
                            "Submit the best supported report now."
                        ),
                    }
                else:
                    remote_calls += 1
                    server_call = True
                    response = await session.call_tool(tool_name, arguments)
                    feedback = self._tool_feedback(response)
                    call_cache[fingerprint] = feedback
                    ledger.ingest(feedback.get("result"))
                    logger.info(
                        "Tool {} returned status {}.",
                        tool_name,
                        feedback.get("status", "unknown"),
                    )
                    self._write_trace(
                        "tool_result",
                        task_id=session.task_id,
                        step=decision_index,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        feedback=feedback,
                    )
                    if response.task_result is not None:
                        return TaskResult.model_validate(response.task_result)

            calls_made.append(
                {
                    "decision": decision_index,
                    "tool": tool_name,
                    "arguments": self._compact_arguments(arguments),
                    "status": feedback.get("status", "unknown"),
                    "reward": feedback.get("reward"),
                    "server_call": server_call,
                }
            )
            last_feedback = feedback
            if feedback.get("status") == "error":
                self._write_trace(
                    "local_rejection",
                    task_id=session.task_id,
                    step=decision_index,
                    tool_name=tool_name,
                    feedback=feedback,
                )

        raise RuntimeError(
            f"Agent hit the {max_decisions}-decision safety limit without "
            "finish_investigation."
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
        decision_index: int,
        calls_remaining: int,
        seconds_remaining: float,
    ) -> list[dict[str, Any]]:
        urgency = ""
        if calls_remaining == 1:
            urgency = "Only the final report call remains. MUST call finish_investigation now."
        elif calls_remaining <= 3:
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
                            "decision_number": decision_index,
                            "server_calls_remaining_including_finish": calls_remaining,
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

    async def _next_step_with_candidates(
        self,
        messages: list[dict[str, Any]],
        *,
        roles: tuple[str, ...],
        task_id: str,
        decision_index: int,
    ) -> NextStep:
        """Generate parallel hypotheses and arbitrate one executable next step."""
        selected_roles = roles[: self.analysis_parallelism]
        if len(selected_roles) < 2:
            return await self.llm.next_step(messages)

        async def candidate(role: str) -> NextStep:
            role_messages = [dict(message) for message in messages]
            role_messages[0] = {
                **role_messages[0],
                "content": f"{role_messages[0]['content']}\n\nSpecialist role: {role}",
            }
            return await self.llm.next_step(role_messages)

        candidates = await asyncio.gather(*(candidate(role) for role in selected_roles))
        candidate_payloads = [item.model_dump(mode="json", by_alias=True) for item in candidates]
        self._write_trace(
            "analyst_candidates",
            task_id=task_id,
            step=decision_index,
            roles=list(selected_roles),
            candidates=candidate_payloads,
        )
        arbitration_messages = [dict(message) for message in messages]
        arbitration_messages.append(
            {
                "role": "user",
                "content": render_json(
                    {
                        "instruction": (
                            "Act as senior arbiter. Compare the specialist candidates against "
                            "the evidence ledger. Return one corrected NextStep. Prefer the call "
                            "with highest expected impact on verdict or report completeness; do "
                            "not combine multiple tool calls."
                        ),
                        "specialist_candidates": candidate_payloads,
                    }
                ),
            }
        )
        return await self.llm.next_step(arbitration_messages)

    @staticmethod
    def _compact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in arguments.items() if key != "reasoning"}

    @staticmethod
    def _report_audit_findings(request: FinishInvestigationRequest) -> list[str]:
        submission = request.submission
        findings: list[str] = []
        if submission.verdict == "benign":
            if not any(
                evidence.anchor == "relation"
                for evidence in submission.legitimacy_evidence
            ):
                findings.append("Candidate has no causal relation anchor.")
            absence_phrases = (
                "no clear malicious",
                "no definitive",
                "no malicious",
                "not observed",
                "absence of",
            )
            if any(phrase in submission.reasoning.lower() for phrase in absence_phrases):
                findings.append(
                    "Candidate relies on absence of malicious evidence; replace this with "
                    "positive legitimacy evidence or reconsider the verdict."
                )
        return findings

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
