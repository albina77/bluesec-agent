from __future__ import annotations

import json
from contextlib import suppress
from copy import deepcopy
from typing import Any

from bluesec1_agent.models import (
    BenignFinishInvestigationSubmission,
    FinishInvestigationRequest,
    MaliciousFinishInvestigationSubmission,
)


class EvidenceLedger:
    """Deduplicated evidence and provenance collected during one investigation."""

    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.relations: dict[str, dict[str, Any]] = {}
        self.entity_ids: set[str] = set()
        self.relation_ids: set[str] = set()
        self.property_fields: dict[str, set[str]] = {}

    def ingest(self, value: Any) -> None:
        """Index entity and relation records found anywhere in a runtime payload."""
        self._visit(value, parent_key=None)

    def snapshot(self) -> dict[str, Any]:
        """Return stable, compact evidence for the next model turn."""
        return {
            "entities": [self.entities[key] for key in sorted(self.entities)],
            "relations": [self.relations[key] for key in sorted(self.relations)],
            "known_entity_ids": sorted(self.entity_ids),
            "known_relation_ids": sorted(self.relation_ids),
        }

    def validate_finish(self, request: FinishInvestigationRequest) -> list[str]:
        """Reject report references that are duplicated or unsupported by evidence."""
        submission = request.submission
        errors: list[str] = []
        if isinstance(submission, MaliciousFinishInvestigationSubmission):
            artifact_ids = [artifact.entity_id for artifact in submission.ir_artifacts]
            errors.extend(self._duplicate_errors(artifact_ids, "IR artifact entity_id"))
            for entity_id in artifact_ids:
                if entity_id not in self.entity_ids:
                    errors.append(f"IR artifact entity_id {entity_id!r} was not observed")
            return errors

        if isinstance(submission, BenignFinishInvestigationSubmission):
            anchors: list[str] = []
            for evidence in submission.legitimacy_evidence:
                if evidence.anchor == "entity":
                    anchor_id = evidence.entity_id
                    anchors.append(f"entity:{anchor_id}")
                    if anchor_id not in self.entity_ids:
                        errors.append(f"legitimacy entity_id {anchor_id!r} was not observed")
                else:
                    anchor_id = evidence.relation_id
                    anchors.append(f"relation:{anchor_id}")
                    if anchor_id not in self.relation_ids:
                        errors.append(f"legitimacy relation_id {anchor_id!r} was not observed")
                known_fields = self.property_fields.get(anchor_id, set())
                unknown_fields = sorted(set(evidence.property_fields) - known_fields)
                if unknown_fields:
                    errors.append(
                        f"fields {unknown_fields!r} were not observed on "
                        f"{evidence.anchor} {anchor_id!r}"
                    )
            errors.extend(self._duplicate_errors(anchors, "legitimacy anchor"))
        return errors

    def prepare_finish(
        self, request: FinishInvestigationRequest
    ) -> tuple[FinishInvestigationRequest, list[str], list[str]]:
        """Remove unsupported benign fields while preserving hard provenance errors."""
        submission = request.submission
        if not isinstance(submission, BenignFinishInvestigationSubmission):
            return request, self.validate_finish(request), []

        errors: list[str] = []
        notes: list[str] = []
        anchors: list[str] = []
        normalized_evidence = []
        for evidence in submission.legitimacy_evidence:
            if evidence.anchor == "entity":
                anchor_id = evidence.entity_id
                anchor_key = f"entity:{anchor_id}"
                if anchor_id not in self.entity_ids:
                    errors.append(f"legitimacy entity_id {anchor_id!r} was not observed")
            else:
                anchor_id = evidence.relation_id
                anchor_key = f"relation:{anchor_id}"
                if anchor_id not in self.relation_ids:
                    errors.append(f"legitimacy relation_id {anchor_id!r} was not observed")
            anchors.append(anchor_key)

            known_fields = self.property_fields.get(anchor_id, set())
            valid_fields = [
                field for field in evidence.property_fields if field in known_fields
            ]
            removed_fields = [
                field for field in evidence.property_fields if field not in known_fields
            ]
            if removed_fields:
                notes.append(
                    f"removed unsupported fields {removed_fields!r} from {anchor_key}"
                )
            if evidence.anchor == "entity" and not valid_fields:
                errors.append(f"no observed property_fields remain for entity {anchor_id!r}")
                normalized_evidence.append(evidence)
            else:
                normalized_evidence.append(
                    evidence.model_copy(update={"property_fields": valid_fields})
                )

        errors.extend(self._duplicate_errors(anchors, "legitimacy anchor"))
        normalized_submission = submission.model_copy(
            update={"legitimacy_evidence": normalized_evidence}
        )
        normalized_request = request.model_copy(update={"submission": normalized_submission})
        return normalized_request, errors, notes

    def _visit(self, value: Any, *, parent_key: str | None) -> None:
        if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            with suppress(json.JSONDecodeError):
                self._visit(json.loads(value), parent_key=parent_key)
            return
        if isinstance(value, list):
            if parent_key and parent_key.endswith("entity_ids"):
                self.entity_ids.update(item for item in value if isinstance(item, str))
            elif parent_key and parent_key.endswith("relation_ids"):
                self.relation_ids.update(item for item in value if isinstance(item, str))
            for item in value:
                self._visit(item, parent_key=parent_key)
            return
        if not isinstance(value, dict):
            return

        record_id = value.get("id")
        if isinstance(record_id, str):
            if "source_id" in value or "target_id" in value:
                self.relation_ids.add(record_id)
                self.relations[record_id] = self._merge(self.relations.get(record_id), value)
            else:
                self.entity_ids.add(record_id)
                self.entities[record_id] = self._merge(self.entities.get(record_id), value)
            self.property_fields.setdefault(record_id, set()).update(self._field_names(value))

        entity_id = value.get("entity_id")
        if isinstance(entity_id, str):
            self.entity_ids.add(entity_id)
            self.property_fields.setdefault(entity_id, set()).update(self._field_names(value))
        relation_id = value.get("relation_id")
        if isinstance(relation_id, str):
            self.relation_ids.add(relation_id)
            self.property_fields.setdefault(relation_id, set()).update(self._field_names(value))

        for key in ("source_id", "target_id", "host_id"):
            referenced_id = value.get(key)
            if isinstance(referenced_id, str):
                self.entity_ids.add(referenced_id)

        for key, child in value.items():
            self._visit(child, parent_key=key)

    @staticmethod
    def _merge(existing: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
        if existing is None:
            return deepcopy(new)
        merged = deepcopy(existing)
        for key, value in new.items():
            if value is not None:
                merged[key] = deepcopy(value)
        return merged

    @staticmethod
    def _field_names(record: dict[str, Any]) -> set[str]:
        names = set(record)
        for container in ("properties", "logon_properties", "access_properties"):
            nested = record.get(container)
            if isinstance(nested, dict):
                names.update(nested)
        return names

    @staticmethod
    def _duplicate_errors(values: list[str], label: str) -> list[str]:
        duplicates = sorted({value for value in values if values.count(value) > 1})
        return [f"duplicate {label}: {value!r}" for value in duplicates]
