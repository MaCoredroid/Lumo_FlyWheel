
from __future__ import annotations

import argparse
import json
from pathlib import Path

from codex.config import load_config
from codex.policy import preview_contract


def render_preview(config_path: str | Path = "codex/config.toml") -> dict[str, object]:
    policy = load_config(config_path)
    preview_policy = preview_contract(policy)
    return {
        "schema_version": "sandbox_preview.v1",
        "sandbox": preview_policy["sandbox"],
        "approval_policy": preview_policy["approval_policy"],
        "jobs": [
            {
                "job_id": "policy-smoke",
                "sandbox": preview_policy["sandbox"],
                "approval_policy": preview_policy["approval_policy"],
            },
            {
                "job_id": "preview-render",
                "sandbox": preview_policy["sandbox"],
                "approval_policy": preview_policy["approval_policy"],
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="codex/config.toml")
    args = parser.parse_args()
    print(json.dumps(render_preview(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
