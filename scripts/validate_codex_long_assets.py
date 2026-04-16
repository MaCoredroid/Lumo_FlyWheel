#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumo_flywheel_serving.codex_long_assets import AssetPackError, validate_authored_asset_pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate authored Codex-Long scenario assets.")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--json", action="store_true", help="Emit the validation summary as JSON.")
    args = parser.parse_args()

    try:
        summary = validate_authored_asset_pack(args.repo_root)
    except AssetPackError as exc:
        print(f"asset validation failed: {exc}")
        return 1

    payload = {
        "repo_root": str(summary.repo_root),
        "family_count": summary.family_count,
        "variant_count": summary.variant_count,
        "family_ids": list(summary.family_ids),
        "scenario_types": list(summary.scenario_types),
        "has_freeze_artifacts": summary.has_freeze_artifacts,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"validated {summary.family_count} families / {summary.variant_count} variants; "
            f"freeze_artifacts={summary.has_freeze_artifacts}"
        )
        print("families:")
        for family_id in summary.family_ids:
            print(f"  - {family_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
