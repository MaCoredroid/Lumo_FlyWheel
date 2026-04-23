from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="active")
    parser.add_argument("--emit-md", required=True)
    args = parser.parse_args()

    incidents = json.loads(Path("fixtures/open_incidents.json").read_text(encoding="utf-8"))
    blockers = [item for item in incidents if item.get("severity") == "blocker"]
    out = Path(args.emit_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Daily triage", "", f"window={args.window}", ""]
    for item in blockers:
        lines.append(f"- {item['id']}: {item['owner']} -> {item['summary']}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
