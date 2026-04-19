from __future__ import annotations

import json
import sys
from pathlib import Path

from .tool_registry import replay_discovery_session


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m gateway.discovery <discovery-session.jsonl>", file=sys.stderr)
        return 2

    session_path = Path(args[0]).resolve()
    tool_map = replay_discovery_session(session_path)
    print(json.dumps(tool_map, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
