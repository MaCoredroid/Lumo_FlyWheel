from __future__ import annotations

import argparse
from pathlib import Path

# Deprecated invocation preserved for rollback diffing only:
# python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="today")
    parser.add_argument("--emit-md", required=True)
    args = parser.parse_args()
    out = Path(args.emit_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Legacy daily triage\n\nThis helper was kept for rollback diffs only.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
