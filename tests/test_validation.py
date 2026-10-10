import asyncio
import json
from typing import cast

import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError
from test_agent import FakeSession, FakeStepClient
from test_llm import FakeOpenAI

from bluesec1_agent.agent import SimpleSGRAgent
from bluesec1_agent.llm import NextStepRequestError, OpenAIStructuredStepClient
from bluesec1_agent.models import (
    FinishInvestigationRequest,
    MaliciousFinishInvestigationSubmission,
    NextStep,
)
from bluesec1_agent.validation import REVIEW_RESPONSE_FORMAT, REVIEW_SCHEMA, ReportReview
from bluesec1_client import RemoteBenchmarkSession

CHECK_NAMES = (
    "observed_facts",
    "alternative_explanation",
    "verdict_consistency",
    "artifact_completeness",
    "response_proportionality",
    "unsupported_claims",
)


def review_payload(decision="approve"):
    check = {"status": "pass", "finding": "Host observed", "evidence_ids": ["host-1"]}
    return {
        **{name: dict(check) for name in CHECK_NAMES},
        "decision": decision,
        "corrections": [] if decision == "approve" else ["Reconcile the actor context"],
        "evidence_request": None,
    }


def finish_step(reason="Evidence-backed report"):
    return NextStep(
        current_state="Report ready",
        plan_remaining_steps_brief=["Submit"],
        task_completed=True,
        function=FinishInvestigationRequest(
            tool="finish_investigation",
            submission=MaliciousFinishInvestigationSubmission(
                verdict="malicious",
                ir_artifacts=[],
                reasoning=reason,
            ),
        ),
    )


class ReviewingClient(FakeStepClient):
    def __init__(self, reviews, steps=None):
        super().__init__(steps or [finish_step(), finish_step("Corrected report")])
        self.reviews = list(reviews)

    async def review_report(self, messages):
        self.review_requests.append(messages)
        return ReportReview.model_validate(self.reviews.pop(0))


@pytest.mark.parametrize("missing", CHECK_NAMES)
def test_every_checkpoint_is_required(missing):
    payload = review_payload()
    del payload[missing]
    with pytest.raises(ValidationError):
        ReportReview.model_validate(payload)


def test_approval_cannot_hide_failed_checkpoint():
    payload = review_payload()
    payload["alternative_explanation"]["status"] = "fail"
    with pytest.raises(ValidationError, match="all checks"):
        ReportReview.model_validate(payload)


def test_strict_schema_closes_objects():
    assert set(CHECK_NAMES) <= set(REVIEW_SCHEMA["required"])

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(REVIEW_SCHEMA)


def test_finishing_mode_does_not_skip_review():
    client = ReviewingClient([review_payload()])
    session = FakeSession()
    asyncio.run(
        SimpleSGRAgent(client, max_steps=1)._investigate(cast(RemoteBenchmarkSession, session))
    )
    assert len(client.review_requests) == 1
    assert [name for name, _ in session.calls] == ["finish_investigation"]


def test_one_correction_then_verification():
    client = ReviewingClient([review_payload("revise_report"), review_payload()])
    session = FakeSession()
    asyncio.run(SimpleSGRAgent(client)._investigate(cast(RemoteBenchmarkSession, session)))
    assert len(client.review_requests) == 2
    assert len(client.requests) == 2
    assert session.calls[0][1]["submission"]["reasoning"] == "Corrected report"


def test_rejected_correction_cannot_escape_via_fallback():
    client = ReviewingClient([review_payload("revise_report")] * 2)
    session = FakeSession()
    with pytest.raises(RuntimeError, match="did not pass"):
        asyncio.run(SimpleSGRAgent(client)._investigate(cast(RemoteBenchmarkSession, session)))
    assert session.calls == []
    assert len(client.review_requests) == 2


def test_unknown_validator_citation_is_rejected():
    payload = review_payload()
    payload["observed_facts"]["evidence_ids"] = ["invented"]
    session = FakeSession()
    with pytest.raises(RuntimeError, match="unobserved"):
        asyncio.run(
            SimpleSGRAgent(ReviewingClient([payload]))._investigate(
                cast(RemoteBenchmarkSession, session)
            )
        )
    assert session.calls == []


@pytest.mark.parametrize(
    "max_calls,expected",
    [(1, ["finish_investigation"]), (2, ["get_entity", "finish_investigation"])],
)
def test_evidence_request_preserves_final_server_call(max_calls, expected):
    payload = review_payload("need_evidence")
    payload["evidence_request"] = {
        "tool": "get_entity",
        "entity_id": "host-1",
        "reasoning": "Check actor context",
    }
    client = ReviewingClient([payload, review_payload()])
    session = FakeSession()
    asyncio.run(
        SimpleSGRAgent(client, max_steps=max_calls)._investigate(
            cast(RemoteBenchmarkSession, session)
        )
    )
    assert [name for name, _ in session.calls] == expected


def test_review_uses_own_schema_and_repairs_invalid_json():
    provider = FakeOpenAI(["invalid", json.dumps(review_payload())])
    client = OpenAIStructuredStepClient(
        base_url="https://example.test",
        api_key="fake",
        model="test",
        timeout_seconds=10,
        max_completion_tokens=1000,
        max_retries=3,
        client=cast(AsyncOpenAI, provider),
    )
    result = asyncio.run(client.review_report([{"role": "system", "content": "Review"}]))
    assert result.decision == "approve"
    assert provider.completions.requests[0]["response_format"] == REVIEW_RESPONSE_FORMAT
    assert len(provider.completions.requests) == 2


def test_review_schema_failure_has_bounded_retries():
    provider = FakeOpenAI(["invalid", "invalid"])
    client = OpenAIStructuredStepClient(
        base_url="https://example.test",
        api_key="fake",
        model="test",
        timeout_seconds=10,
        max_completion_tokens=1000,
        max_retries=3,
        client=cast(AsyncOpenAI, provider),
    )
    with pytest.raises(NextStepRequestError):
        asyncio.run(client.review_report([]))
    assert len(provider.completions.requests) == 2
