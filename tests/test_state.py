from bluesec1_agent.models import (
    BenignFinishInvestigationSubmission,
    FinishInvestigationEntityEvidence,
    FinishInvestigationRelationEvidence,
    FinishInvestigationRequest,
)
from bluesec1_agent.state import EvidenceLedger


def test_evidence_ledger_validates_observed_benign_anchors_and_fields() -> None:
    ledger = EvidenceLedger()
    ledger.ingest(
        {
            "id": "process-1",
            "type": "windows_process",
            "properties": {"image_path": "C:/Windows/System32/taskhostw.exe", "pid": 42},
            "outgoing_relation_ids": ["relation-1"],
        }
    )
    ledger.ingest(
        {
            "id": "relation-1",
            "type": "parent_of",
            "source_id": "process-1",
            "target_id": "process-2",
            "properties": {"timestamp": "2026-01-01T00:00:00Z"},
        }
    )
    request = FinishInvestigationRequest(
        tool="finish_investigation",
        submission=BenignFinishInvestigationSubmission(
            verdict="benign",
            legitimacy_evidence=[
                FinishInvestigationEntityEvidence(
                    anchor="entity",
                    entity_id="process-1",
                    property_fields=["image_path", "pid"],
                ),
                FinishInvestigationRelationEvidence(
                    anchor="relation",
                    relation_id="relation-1",
                    property_fields=["timestamp"],
                ),
            ],
            reasoning="Known Windows process with an expected parent relation.",
        ),
    )

    assert ledger.validate_finish(request) == []
    assert ledger.snapshot()["known_entity_ids"] == ["process-1", "process-2"]
    assert ledger.snapshot()["known_relation_ids"] == ["relation-1"]


def test_evidence_ledger_rejects_unknown_or_duplicate_report_evidence() -> None:
    ledger = EvidenceLedger()
    ledger.ingest({"id": "process-1", "properties": {"pid": 42}})
    request = FinishInvestigationRequest(
        tool="finish_investigation",
        submission=BenignFinishInvestigationSubmission(
            verdict="benign",
            legitimacy_evidence=[
                FinishInvestigationEntityEvidence(
                    anchor="entity",
                    entity_id="process-1",
                    property_fields=["signer"],
                ),
                FinishInvestigationEntityEvidence(
                    anchor="entity",
                    entity_id="process-1",
                    property_fields=["pid"],
                ),
                FinishInvestigationRelationEvidence(
                    anchor="relation",
                    relation_id="missing-relation",
                    property_fields=[],
                ),
            ],
            reasoning="Unsupported test report.",
        ),
    )

    errors = ledger.validate_finish(request)

    assert any("signer" in error for error in errors)
    assert any("missing-relation" in error for error in errors)
    assert any("duplicate legitimacy anchor" in error for error in errors)


def test_evidence_ledger_indexes_json_encoded_runtime_records() -> None:
    ledger = EvidenceLedger()

    ledger.ingest(
        '{"trigger_entities":[{"id":"process-1","properties":{"pid":42}}]}'
    )

    assert ledger.snapshot()["known_entity_ids"] == ["process-1"]


def test_prepare_finish_removes_only_unsupported_benign_fields() -> None:
    ledger = EvidenceLedger()
    ledger.ingest(
        {
            "id": "process-1",
            "properties": {"pid": 42, "signed": True, "parent_image": None},
            "relation_ids": {
                "incoming": ["relation-in"],
                "outgoing": ["relation-out"],
            },
        }
    )
    request = FinishInvestigationRequest(
        tool="finish_investigation",
        submission=BenignFinishInvestigationSubmission(
            verdict="benign",
            legitimacy_evidence=[
                FinishInvestigationEntityEvidence(
                    anchor="entity",
                    entity_id="process-1",
                    property_fields=["pid", "invented_field", "signed", "parent_image"],
                )
            ],
            reasoning="Observed signed process.",
        ),
    )

    normalized, errors, notes = ledger.prepare_finish(request)

    assert errors == []
    assert notes
    evidence = normalized.submission.legitimacy_evidence[0]
    assert evidence.property_fields == ["pid", "signed"]
    snapshot = ledger.snapshot()
    assert snapshot["unexplored_relation_ids"] == ["relation-in", "relation-out"]
    assert "parent_image" not in snapshot["reportable_non_null_fields"]["process-1"]


def test_evidence_ledger_surfaces_discovery_and_sandbox_evasion() -> None:
    ledger = EvidenceLedger()
    ledger.ingest(
        {
            "id": "script-1",
            "type": "powershell_script",
            "properties": {
                "script_text": (
                    "systeminfo; hostname; ipconfig /all; tasklist; "
                    "if ($manufacturer -match 'VMware|VirtualBox|Sandboxie') { exit }"
                )
            },
        }
    )

    signals = ledger.snapshot()["analyst_attention"]

    assert signals[0]["entity_id"] == "script-1"
    assert signals[0]["severity"] == "high"


def test_prepare_finish_requires_relation_for_benign_graph_claim() -> None:
    ledger = EvidenceLedger()
    ledger.ingest({"id": "process-1", "properties": {"signed": True}})
    ledger.ingest(
        {
            "id": "relation-1",
            "type": "parent_of",
            "source_id": "process-2",
            "target_id": "process-1",
        }
    )
    request = FinishInvestigationRequest(
        tool="finish_investigation",
        submission=BenignFinishInvestigationSubmission(
            verdict="benign",
            legitimacy_evidence=[
                FinishInvestigationEntityEvidence(
                    anchor="entity",
                    entity_id="process-1",
                    property_fields=["signed"],
                )
            ],
            reasoning="Expected signed process execution.",
        ),
    )

    _, errors, _ = ledger.prepare_finish(request)

    assert any("causal relation" in error for error in errors)
