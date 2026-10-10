"""Credential-free regression replay for the bundled organizer fixture."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--scripted", action="store_true", required=True)
    args = parser.parse_args()
    transcript = json.loads((args.fixture / "transcript.json").read_text())
    expected = json.loads((args.fixture / "expected.json").read_text())
    final = transcript[-1]["arguments"].get("submission", {})
    wanted = {(item["entity_id"], item["kind"]) for item in expected["ir_artifacts"]}
    found = {(item["entity_id"], item["kind"]) for item in final.get("ir_artifacts", [])}
    result = {
        "mode": "scripted offline replay",
        "fixture": args.fixture.name,
        "tool_calls": len(transcript),
        "verdict_matches_example": final.get("verdict") == expected["verdict"],
        "missing_artifacts": sorted(wanted - found),
        "unexpected_artifacts": sorted(found - wanted),
        "official_score": None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["verdict_matches_example"] or result["missing_artifacts"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
