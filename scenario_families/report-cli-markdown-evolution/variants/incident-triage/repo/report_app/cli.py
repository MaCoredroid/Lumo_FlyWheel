from __future__ import annotations

import argparse

from report_app.rendering import render_json
from report_app.service import TITLE, build_sections, build_triage_summary


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(prog="incident-triage-report")
    parser.add_argument("--format", choices=["json"], default="json")
    args = parser.parse_args(argv)
    sections = build_sections()
    triage_summary = build_triage_summary(sections)
    if args.format != "json":
        raise ValueError(f"unsupported format: {args.format}")
    return render_json(TITLE, sections, triage_summary)


if __name__ == "__main__":
    print(main())
