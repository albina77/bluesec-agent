import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from bluesec1_agent.agent import SimpleSGRAgent
from bluesec1_agent.models import (
    FinishInvestigationArtifact,
    FinishInvestigationRequest,
    GetEntityRequest,
    MaliciousFinishInvestigationSubmission,
    NextStep,
)
from bluesec1_client import (
    PublicTask,
    RemoteBenchmarkClient,
    RemoteBenchmarkSession,
    TaskResult,
    ToolCallResult,
)


class FakeStepClient:
    """Deterministic structured steps used by the agent unit test."""

    def __init__(self, steps: list[NextStep]) -> None:
        """Store steps in their expected request order.

        Args:
            steps: Prepared LLM steps.
        """
        self.steps = list(steps)
        self.requests: list[list[dict[str, Any]]] = []

    async def next_step(self, messages: list[dict[str, Any]]) -> NextStep:
        """Record one conversation and return its next prepared step.

        Args:
            messages: Current LLM history.

        Returns:
            Next prepared structured step.
        """
        self.requests.append([dict(message) for message in messages])
        return self.steps.pop(0)


class FakeSession:
    """In-memory remote session used by the structured agent test."""

    def __init__(self) -> None:
        """Create a task with a JSON-encoded remote tool catalog."""
        self.task = PublicTask(
            task_id="task-1",
            observation={
                "alert_text": {
                    "trigger_entities": [
                        {"id": "host-1", "type": "host", "properties": {"hostname": "WS-1"}}
                    ]
                },
                "available_tools": json.dumps(
                    [
                        {"name": "get_entity", "parameters": {"type": "object"}},
                        {"name": "finish_investigation", "parameters": {"type": "object"}},
                    ]
                ),
            },
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def task_id(self) -> str:
        """Return the public task identifier.

        Returns:
            Task identifier.
        """
        return self.task.task_id

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolCallResult:
        """Record one call and terminate on the finishing tool.

        Args:
            tool_name: Invoked tool name.
            arguments: Invoked tool arguments.

        Returns:
            Fake server response.
        """
        self.calls.append((tool_name, arguments))
        terminal = tool_name == "finish_investigation"
        task_result = None
        if terminal:
            task_result = TaskResult(
                task_id="task-1",
                completion_reason="terminated",
            )
        return ToolCallResult(
            task_id="task-1",
            tool_name=tool_name,
            reward=1.0,
            terminated=terminal,
            info={
                "tool_status": "ok",
                "tool_result": (
                    [{"entity_id": "process-1", "pid": 42}] if not terminal else "Report accepted"
                ),
            },
            task_result=task_result,
        )

    async def abort_task(self, *, error_message: str | None = None) -> TaskResult:
        """Reject unexpected aborts.

        Args:
            error_message: Agent failure diagnostic.

        Raises:
            AssertionError: Always, because this task must finish normally.
        """
        raise AssertionError(f"Unexpected abort: {error_message}")


class FakeClient:
    """Single-task stand-in for an open RemoteBenchmarkClient."""

    def __init__(self, session: FakeSession) -> None:
        """Store the only session returned before queue exhaustion.

        Args:
            session: Prepared task session.
        """
        self.run = SimpleNamespace(run_id="run-1")
        self.session: FakeSession | None = session

    async def start_task(self) -> RemoteBenchmarkSession | None:
        """Lease the prepared session once.

        Returns:
            Prepared session or `None` after it was leased.
        """
        if self.session is None:
            return None
        session = self.session
        self.session = None
        return cast(RemoteBenchmarkSession, session)


def test_agent_executes_fresh_structured_task_via_remote_session(tmp_path: Path) -> None:
    """Tool evidence should feed the next step before the terminal action."""
    llm = FakeStepClient(
        [
            NextStep(
                current_state="Alert identifies process 42 on host-1.",
                plan_remaining_steps_brief=["Inspect its process ancestry and descendants."],
                task_completed=False,
                function=GetEntityRequest(
                    tool="get_entity",
                    entity_id="entity-1",
                    reasoning="Inspect the alert entity.",
                ),
            ),
            NextStep(
                current_state="The suspicious process is confirmed on host-1.",
                plan_remaining_steps_brief=["Submit the evidence-backed malicious verdict."],
                task_completed=True,
                function=FinishInvestigationRequest(
                    tool="finish_investigation",
                    submission=MaliciousFinishInvestigationSubmission(
                        verdict="malicious",
                        ir_artifacts=[
                            FinishInvestigationArtifact(
                                entity_id="host-1",
                                kind="host_to_isolate",
                            )
                        ],
                        reasoning="The alert process is part of a suspicious process chain.",
                    ),
                ),
            ),
        ]
    )
    session = FakeSession()
    client = FakeClient(session)

    result = asyncio.run(
        SimpleSGRAgent(llm, review_final_report=False, trace_dir=tmp_path).run(
            cast(RemoteBenchmarkClient, client)
        )
    )

    assert result.run_id == "run-1"
    assert result.successful_tasks == 1
    assert result.failed_tasks == 0
    assert [name for name, _ in session.calls] == [
        "get_entity",
        "finish_investigation",
    ]
    assert session.calls[0][1] == {
        "entity_id": "entity-1",
        "reasoning": "Inspect the alert entity.",
    }
    assert session.calls[1][1] == {
        "submission": {
            "verdict": "malicious",
            "ir_artifacts": [{"entity_id": "host-1", "kind": "host_to_isolate"}],
            "reasoning": "The alert process is part of a suspicious process chain.",
        }
    }
    assert "process-1" in str(llm.requests[1][-1]["content"])
    assert "available_tools" not in str(
        json.loads(llm.requests[0][-1]["content"])["initial_observation"]
    )
    assert not any(message.get("role") == "tool" for message in llm.requests[1])
    trace_records = [
        json.loads(line) for line in (tmp_path / "run-1.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in trace_records] == [
        "run_started",
        "task_started",
        "llm_step",
        "tool_result",
        "llm_step",
        "tool_result",
        "task_result",
        "run_finished",
    ]


def test_provider_capacity_and_transport_errors_stop_the_run() -> None:
    class ProviderError(Exception):
        status_code = 429

    class APIConnectionError(Exception):
        pass

    assert SimpleSGRAgent._is_fatal_provider_error(ProviderError())
    assert SimpleSGRAgent._is_fatal_provider_error(APIConnectionError())
    assert not SimpleSGRAgent._is_fatal_provider_error(ValueError("one bad task"))


def test_agent_rejects_invalid_run_task_limit() -> None:
    llm = FakeStepClient([])

    try:
        SimpleSGRAgent(llm, max_tasks=0)
    except ValueError as error:
        assert "max_tasks" in str(error)
    else:
        raise AssertionError("Expected an invalid max_tasks value to be rejected")


def test_tool_feedback_extracts_json_encoded_last_result() -> None:
    response = ToolCallResult(
        task_id="task-1",
        tool_name="get_entity",
        observation={
            "alert_text": "large repeated alert",
            "available_tools": "large repeated tool catalog",
            "last_tool_result": '{"id":"entity-1","properties":{"pid":42}}',
        },
        info={"tool_status": "ok"},
    )

    feedback = SimpleSGRAgent._tool_feedback(response)

    assert feedback["result"] == {"id": "entity-1", "properties": {"pid": 42}}


def test_duplicate_decision_uses_cache_without_spending_server_call() -> None:
    repeated = GetEntityRequest(
        tool="get_entity",
        entity_id="entity-1",
        reasoning="Inspect the same entity.",
    )
    llm = FakeStepClient(
        [
            NextStep(
                current_state="Inspecting entity.",
                plan_remaining_steps_brief=["Inspect entity."],
                task_completed=False,
                function=repeated,
            ),
            NextStep(
                current_state="Accidentally repeated entity.",
                plan_remaining_steps_brief=["Use cached result."],
                task_completed=False,
                function=repeated,
            ),
            NextStep(
                current_state="Enough evidence.",
                plan_remaining_steps_brief=["Finish."],
                task_completed=True,
                function=FinishInvestigationRequest(
                    tool="finish_investigation",
                    submission=MaliciousFinishInvestigationSubmission(
                        verdict="malicious",
                        ir_artifacts=[
                            FinishInvestigationArtifact(
                                entity_id="host-1",
                                kind="host_to_isolate",
                            )
                        ],
                        reasoning="Supported test verdict.",
                    ),
                ),
            ),
        ]
    )
    session = FakeSession()

    asyncio.run(
        SimpleSGRAgent(llm, max_steps=2, review_final_report=False)._investigate(
            cast(RemoteBenchmarkSession, session)
        )
    )

    assert [name for name, _ in session.calls] == ["get_entity", "finish_investigation"]
    third_prompt = json.loads(llm.requests[2][-1]["content"])
    assert third_prompt["budget"]["server_calls_remaining_including_finish"] == 1
    assert third_prompt["investigation_state"]["last_tool_feedback"]["status"] == "cached"


def test_final_report_gets_one_local_review_before_submission() -> None:
    finish = FinishInvestigationRequest(
        tool="finish_investigation",
        submission=MaliciousFinishInvestigationSubmission(
            verdict="malicious",
            ir_artifacts=[
                FinishInvestigationArtifact(
                    entity_id="host-1",
                    kind="host_to_isolate",
                )
            ],
            reasoning="Evidence-backed test verdict.",
        ),
    )
    llm = FakeStepClient(
        [
            NextStep(
                current_state="Candidate report ready.",
                plan_remaining_steps_brief=["Review report."],
                task_completed=True,
                function=finish,
            ),
            NextStep(
                current_state="Report audited against evidence.",
                plan_remaining_steps_brief=["Submit report."],
                task_completed=True,
                function=finish,
            ),
        ]
    )
    session = FakeSession()

    asyncio.run(
        SimpleSGRAgent(llm, max_steps=1)._investigate(
            cast(RemoteBenchmarkSession, session)
        )
    )

    assert [name for name, _ in session.calls] == ["finish_investigation"]
    review_prompt = json.loads(llm.requests[1][-1]["content"])
    assert (
        review_prompt["investigation_state"]["last_tool_feedback"]["status"]
        == "review_required"
    )
