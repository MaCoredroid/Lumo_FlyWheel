import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    lines = [
        f"# Handoff for {payload['service']}",
        "",
        f"- Severity: {payload['severity']}",
        f"- Owner: {payload['owner']}",
    ]
    Path(args.output).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
