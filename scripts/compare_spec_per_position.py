#!/usr/bin/env python3
"""Compare canonical spec per-position measurement JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "spec_per_position.v1":
        raise SystemExit(f"not a spec_per_position.v1 measurement: {path}")
    return data


def _label(data: dict[str, Any], fallback: str) -> str:
    return data.get("label") or data.get("mode") or fallback


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{100 * value:5.1f}"


def _divergence(base: dict[str, Any], candidate: dict[str, Any], threshold: float) -> int | None:
    base_pos = base["spec"]["per_position"]
    cand_pos = candidate["spec"]["per_position"]
    for index, (b, c) in enumerate(zip(base_pos, cand_pos), start=1):
        if b is not None and c is not None and b - c >= threshold:
            return index
    return None


def compare(paths: list[str], threshold: float) -> str:
    if len(paths) < 2:
        raise SystemExit("provide at least two measurement JSON files")
    loaded = [_load(path) for path in paths]
    prompt_hash = loaded[0].get("prompt_sha256")
    mismatched = [path for path, data in zip(paths, loaded) if data.get("prompt_sha256") != prompt_hash]
    if mismatched:
        raise SystemExit("prompt hash mismatch; refusing non-identical prompt comparison: " + ", ".join(mismatched))

    lines = []
    lines.append("label        n      avg   acc0  pos1  pos2  pos3  pos4  pos5")
    for idx, data in enumerate(loaded):
        spec = data["spec"]
        label = _label(data, Path(paths[idx]).parent.name)[:10]
        lines.append(
            f"{label:<10} {spec['n_events']:>5.0f} "
            f"{spec['avg']:>7.3f} {_fmt(spec['acc0'])} "
            + " ".join(_fmt(v) for v in spec["per_position"])
        )
    base = loaded[0]
    for idx, data in enumerate(loaded[1:], start=1):
        div = _divergence(base, data, threshold)
        label = _label(data, Path(paths[idx]).parent.name)
        if div is None:
            lines.append(f"{label}: no divergence >= {100 * threshold:.1f} pp vs {_label(base, 'base')}")
        else:
            lines.append(
                f"{label}: first divergence vs {_label(base, 'base')} at pos{div} "
                f"(drop >= {100 * threshold:.1f} pp)"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", nargs="+")
    parser.add_argument("--threshold", type=float, default=0.05, help="absolute probability drop threshold")
    args = parser.parse_args()
    print(compare(args.measurements, args.threshold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
