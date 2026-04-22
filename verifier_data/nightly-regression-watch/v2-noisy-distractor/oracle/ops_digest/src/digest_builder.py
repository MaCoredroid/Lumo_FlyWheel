from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import normalize_run


def load_runs(fixtures_dir: str | Path) -> list[dict]:
    runs = []
    for path in sorted(Path(fixtures_dir).glob("*.json")):
        runs.append(normalize_run(json.loads(path.read_text())))
    return runs


def select_latest_per_day(runs: list[dict]) -> list[dict]:
    latest = {}
    for run in runs:
        current = latest.get(run["report_date"])
        if current is None or run["completed_at"] > current["completed_at"]:
            latest[run["report_date"]] = run
    return [latest[date] for date in sorted(latest)]


def build_digest_markdown(runs: list[dict]) -> str:
    latest = select_latest_per_day(runs)
    blocking = [run for run in latest if run["is_blocking"]]
    healthy = [run for run in latest if not run["is_blocking"]]

    lines = [
        "# Nightly Regression Watch",
        "",
        "## Action required",
    ]
    if blocking:
        for run in blocking:
            lines.append(f"- {run['report_date']} `{run['run_id']}`: {run['summary']}")
    else:
        lines.append("- none")

    lines.extend(["", "## Healthy nights"])
    if healthy:
        for run in healthy:
            lines.append(f"- {run['report_date']} `{run['run_id']}`: {run['summary']}")
    else:
        lines.append("- none")

    lines.extend(["", "## Notes", "- Generated from the latest completed run for each report_date."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    digest = build_digest_markdown(load_runs(args.fixtures))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
