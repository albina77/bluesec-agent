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
