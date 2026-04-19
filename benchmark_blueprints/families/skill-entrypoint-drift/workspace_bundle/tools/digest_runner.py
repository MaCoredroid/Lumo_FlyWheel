from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.common.report_loader import load_events  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a markdown ops digest from incident events.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSON file containing a list of incident events.",
    )
    parser.add_argument(
        "--summary-length",
        choices=("short", "long"),
        default="short",
        help="Choose whether the summary section is compact or extended.",
    )
    return parser


def render_digest(events: list[dict[str, str]], summary_length: str) -> str:
    total_events = len(events)
    open_events = sum(1 for event in events if event.get("status") == "open")
    severities = [event.get("severity", "unknown") for event in events]
    highest_severity = min(severities, default="unknown", key=_severity_rank)

    lines = [
        "# Ops Digest",
        "",
        "## Summary",
        f"- Total events: {total_events}",
        f"- Open incidents: {open_events}",
        f"- Highest severity: {highest_severity}",
    ]

    if summary_length == "long":
        impacted_services = sorted({event.get("service", "unknown") for event in events})
        lines.append(f"- Impacted services: {', '.join(impacted_services) or 'none'}")

    lines.extend(["", "## Events"])
    if not events:
        lines.append("- No incidents recorded.")
    else:
        for event in events:
            lines.append(
                "- {date} | {severity} | {service} | {status} | {summary}".format(
                    date=event.get("date", "unknown-date"),
                    severity=event.get("severity", "unknown"),
                    service=event.get("service", "unknown-service"),
                    status=event.get("status", "unknown"),
                    summary=event.get("summary", "No summary provided."),
                )
            )

    return "\n".join(lines) + "\n"


def _severity_rank(severity: str) -> tuple[int, str]:
    order = {"sev1": 1, "sev2": 2, "sev3": 3, "sev4": 4}
    return (order.get(severity, 99), severity)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    events = load_events(args.input)
    sys.stdout.write(render_digest(events, args.summary_length))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
