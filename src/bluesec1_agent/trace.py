from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlTraceWriter:
    """Append crash-safe participant-visible run events to one JSONL file."""

    def __init__(self, directory: Path, run_id: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        safe_run_id = "".join(
            character for character in run_id if character.isalnum() or character in "-_"
        )
        self.path = directory / f"{safe_run_id or 'run'}.jsonl"

    def write(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


class RunArtifacts(JsonlTraceWriter):
    """Best-effort structured artifacts for comparing local runs.

    The legacy JSONL file is retained for compatibility; the directory contains
    a manifest, event stream, task results, and summary when writes succeed.
    Trace failures never interrupt an investigation or cause a tool retry.
    """

    def __init__(self, directory: Path, run_id: str, *, manifest: dict[str, Any]) -> None:
        super().__init__(directory, run_id)
        self.artifact_dir = directory / ("run-" + self.path.stem)
        self._failed = False
        try:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_json("manifest.json", manifest)
        except OSError:
            self._failed = True

    def _atomic_json(self, name: str, payload: Any) -> None:
        target = self.artifact_dir / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        temporary.replace(target)

    def write(self, event: str, **payload: Any) -> None:
        try:
            super().write(event, **payload)
            if not self._failed:
                record = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **payload}
                with (self.artifact_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                if event == "task_result":
                    with (self.artifact_dir / "task_results.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except OSError:
            self._failed = True

    def close(self, status: str, **summary: Any) -> None:
        if self._failed:
            return
        try:
            self._atomic_json("summary.json", {"status": status, **summary})
        except OSError:
            self._failed = True
