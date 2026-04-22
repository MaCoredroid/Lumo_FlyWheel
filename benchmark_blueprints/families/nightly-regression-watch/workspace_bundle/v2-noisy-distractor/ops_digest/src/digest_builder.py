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
    by_day = {}
    for run in runs:
        by_day.setdefault(run["report_date"], run)
    return [by_day[date] for date in sorted(by_day)]


def build_digest_markdown(runs: list[dict]) -> str:
    latest = select_latest_per_day(runs)
    lines = [
        "# Nightly Failure Digest",
        "",
        "## Blocking nights",
    ]
    for run in latest:
        prefix = "Action required" if run["is_blocking"] else "Action required"
        lines.append(f"- {prefix}: {run['report_date']} `{run['run_id']}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Flag anything marked fail.")
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
