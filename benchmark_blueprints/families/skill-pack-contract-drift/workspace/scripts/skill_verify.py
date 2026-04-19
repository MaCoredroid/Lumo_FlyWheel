from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_report(payload: dict[str, str]) -> str:
    return (
        "# Summary\n\n"
        f"- release: {payload['release']}\n"
        f"- owner: {payload['owner']}\n\n"
        "# Action Items\n\n"
        f"- verify rollout for {payload['release']}\n"
        f"- notify {payload['owner']}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    print(build_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
