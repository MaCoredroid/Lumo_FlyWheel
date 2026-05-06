#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lumo_flywheel_serving.track_b import evaluate_b2_metrics  # noqa: E402


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"metrics must be a JSON object: {path}")
    return payload


def _load_thresholds(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"fixture must be a mapping: {path}")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError(f"fixture missing thresholds mapping: {path}")
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Track B B-2 behavioral metrics.")
    parser.add_argument("--candidate-metrics", required=True, type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metrics = _load_json(args.candidate_metrics)
    result = evaluate_b2_metrics(metrics, _load_thresholds(args.fixture))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
