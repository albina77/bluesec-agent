"""Credential-free regression replay for the bundled organizer fixture."""

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("compare",))
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path("fixtures/sc0045"))
    parser.add_argument("--scripted", action="store_true")
    args = parser.parse_args()
    if args.command == "compare":
        if len(args.paths) != 2:
            parser.error("compare requires baseline and candidate artifact directories")
        print(json.dumps(compare(args.paths[0], args.paths[1]), indent=2))
        return
    if not args.scripted:
        parser.error("fixture replay requires --scripted")
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


def compare(baseline: Path, candidate: Path) -> dict:
    def load(path: Path) -> dict[str, dict]:
        rows = []
        result_file = path / "task_results.jsonl"
        if result_file.exists():
            for line in result_file.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return {row.get("task_id", str(i)): row for i, row in enumerate(rows)}

    before, after = load(baseline), load(candidate)
    common = sorted(before.keys() & after.keys())
    deltas = []
    for key in common:
        left, right = before[key], after[key]
        left_result = left.get("result", left)
        right_result = right.get("result", right)
        deltas.append({
            "task": key,
            "tool_calls_delta": (
                right_result.get("tool_calls", 0) - left_result.get("tool_calls", 0)
            ),
        })
    return {
        "baseline_tasks": len(before),
        "candidate_tasks": len(after),
        "matched_tasks": len(common),
        "per_task_deltas": deltas,
        "baseline_wall_time_median": statistics.median(
            [r.get("wall_time_seconds", 0) for r in before.values()]
        ) if before else None,
        "candidate_wall_time_median": statistics.median(
            [r.get("wall_time_seconds", 0) for r in after.values()]
        ) if after else None,
        "note": (
            "Compare matched tasks; aggregate score across different task sets is not "
            "evidence of improvement."
        ),
    }


if __name__ == "__main__":
    main()
