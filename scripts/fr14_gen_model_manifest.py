#!/usr/bin/env python3
"""Regenerate the fixed32 contract's byte-pinned model block from a real dir.

FR14 (2026-08-16) re-points the fixed32 serving stack from the Qwen3.6-27B-FP8
checkpoint to the post-lm_head-surgery Qwen3.8-27B-NVFP4 checkpoint. The
contract's model block (``MODEL_AUXILIARY_FILES`` / ``MODEL_FILES`` /
``MODEL_FILE_RECORDS`` / ``MODEL_CANONICAL_SHA256``) is byte-exact and has no
partial-swap mode, so it has to be regenerated wholesale. This script is the
reproducible generator for that block: it walks the checkpoint directory with
EXACTLY the semantics ``fr13_fixed32_contract._pinned_model_files`` enforces and
emits both the Python source fragment and a JSON sidecar.

Semantics reproduced verbatim from ``_pinned_model_files`` (contract.py):

* the file set is ``sorted(name for path in model_root.iterdir() if path.is_file())``
  -- ``Path.is_file()`` is False for directories, so ``.cache/huggingface`` and
  any other metadata DIRECTORY is excluded automatically, while dot-FILES
  (``.gitattributes``, ``.lumo_pinned_revision``, ``.lumo_lmhead_surgery.json``)
  ARE members. That is the pre-existing FR13 behaviour -- the served 3.6 dir
  also carries a ``.cache`` directory and pins ``.gitattributes`` -- so nothing
  about the rule changes here, only the resulting names.
* every record is ``{"path", "size", "sha256"}`` built by ``_exact_file_record``
  (regular files only, symlinks refused).
* the canonical digest is ``sha256(canonical_bytes(records))`` where
  ``canonical_bytes`` is compact sorted-key ASCII JSON.

Both FR14 checkpoints deliberately keep their surgery sidecars in the pinned
set -- they are evidence, not noise, and a silent deletion of any of them must
fail the contract:

* arm A (``/models/qwen3.8-27b-nvfp4``, unsloth): ``.lumo_lmhead_surgery.json``
  + ``config.json.pre_lmhead_surgery.bak`` (the FP8 lm_head dequantised to BF16
  so the checkpoint could boot at all -- REDTEAM_20260816.md pass 6) and
  ``.lumo_kv_scheme_surgery.json`` + its ``.bak``.
* arm B (``/models/qwen3.8-27b-nvfp4-radixark``, RadixArk, LIVE):
  ``.lumo_radixark_kv_surgery.json`` + ``config.json.pre_kv_surgery.bak`` +
  ``hf_quant_config.json.pre_kv_surgery.bak``. There is deliberately NO lm_head
  sidecar: arm B serves the NVFP4 head as shipped, through the boot-time loader
  patch, and the absence of a surgery record is itself the evidence for that.

Usage::

    python3 scripts/fr14_gen_model_manifest.py \
        --model-root /models/qwen3.8-27b-nvfp4-radixark \
        --emit-python /tmp/model_block.py \
        --output /tmp/fr14_model_manifest.json

Re-running it against an unchanged checkpoint is a no-op; any difference from
the constants currently in ``fr13_fixed32_contract.py`` means the served bytes
moved and the contract must be re-pinned in the same commit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fr13_fixed32_contract import (  # noqa: E402
    ContractError,
    MODEL_ROOT,
    canonical_bytes,
    sha256_file,
)


def collect_records(model_root: Path) -> list[dict[str, Any]]:
    """Walk ``model_root`` exactly as ``_pinned_model_files`` does."""
    if not model_root.is_dir() or model_root.is_symlink():
        raise ContractError(f"model root is missing or symlinked: {model_root}")
    names = sorted(path.name for path in model_root.iterdir() if path.is_file())
    records: list[dict[str, Any]] = []
    for name in names:
        path = model_root / name
        if not path.is_file() or path.is_symlink():
            raise ContractError(
                f"required regular file is missing or symlinked: {path}"
            )
        records.append(
            {
                "path": name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def render_python_block(records: list[dict[str, Any]], model_root: Path) -> str:
    digest = canonical_digest(records)
    lines: list[str] = []
    lines.append(f'MODEL_ROOT = Path("{model_root}")')
    lines.append("MODEL_AUXILIARY_FILES = (")
    for record in records:
        lines.append(f'    "{record["path"]}",')
    lines.append(")")
    lines.append("MODEL_FILES = tuple(sorted(MODEL_AUXILIARY_FILES))")
    lines.append("MODEL_CANONICAL_SHA256 = (")
    lines.append(f'    "{digest}"')
    lines.append(")")
    lines.append("MODEL_FILE_RECORDS = (")
    for record in records:
        lines.append("    (")
        lines.append(f'        "{record["path"]}",')
        lines.append(f'        {record["size"]},')
        lines.append(f'        "{record["sha256"]}",')
        lines.append("    ),")
    lines.append(")")
    return "\n".join(lines) + "\n"


def canonical_digest(records: list[dict[str, Any]]) -> str:
    import hashlib

    return hashlib.sha256(canonical_bytes(records)).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=MODEL_ROOT,
        help="checkpoint directory to pin (default: the contract's MODEL_ROOT)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the JSON manifest sidecar here",
    )
    parser.add_argument(
        "--emit-python",
        type=Path,
        help="write the ready-to-paste contract source fragment here",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "compare the derived block against the contract's current "
            "constants and exit 2 on any difference"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = collect_records(args.model_root)
    digest = canonical_digest(records)
    payload = {
        "schema": "fr14.model_manifest.v1",
        "model_root": str(args.model_root),
        "file_count": len(records),
        "files": records,
        "canonical_sha256": digest,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="ascii")
    else:
        print(rendered, end="")
    if args.emit_python is not None:
        args.emit_python.write_text(
            render_python_block(records, args.model_root), encoding="ascii"
        )
    if args.check:
        from scripts.fr13_fixed32_contract import (
            MODEL_CANONICAL_SHA256,
            MODEL_FILES,
            expected_model_file_records,
        )

        problems: list[str] = []
        actual_names = tuple(record["path"] for record in records)
        if actual_names != MODEL_FILES:
            problems.append(f"file set differs: {actual_names} != {MODEL_FILES}")
        if records != expected_model_file_records():
            problems.append("per-file (path,size,sha256) records differ")
        if digest != MODEL_CANONICAL_SHA256:
            problems.append(
                f"canonical digest differs: {digest} != {MODEL_CANONICAL_SHA256}"
            )
        if problems:
            for problem in problems:
                print(f"fr14 model manifest check FAILED: {problem}", file=sys.stderr)
            return 2
        print("fr14 model manifest check PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
