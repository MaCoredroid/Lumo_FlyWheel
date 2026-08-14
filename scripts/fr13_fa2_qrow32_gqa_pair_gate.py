#!/usr/bin/env python3
"""Fail-closed identity and dual real-B4 gates for FA2 qrow32 GQA-pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fr13_fa2_qrow32_gate as qrow32_gate
from fr13_patch_fa2_tree_bias import (
    FIXED32_QUERY_GQA_PAIR32_API_DECLARATION,
    FIXED32_QUERY_GQA_PAIR32_API_GATE,
    FIXED32_QUERY_GQA_PAIR32_BATCH_STRIDE_SENTINEL,
    FIXED32_QUERY_GQA_PAIR32_TRANSLATION_UNIT,
)


CANDIDATE_ARM = "gqa_pair"
CANDIDATE_SHA256 = (
    "af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb"
)
CANDIDATE_SIZE = 299_813_360
FA2_HEAD = "29210221863736a08f71a866459e368ad1ac4a95"
SOURCE_FILES = {
    "csrc/flash_attn/flash_api.cpp": (
        "ff33ed53d024ee4cb2f6a69fb168bcb4c07013ecef34d692c74fe8fd0222222c"
    ),
    "csrc/flash_attn/flash_api_torch_lib.cpp": (
        "c575d9f02ba44bf7022c77b80fdf12173da0ecae8a4d7599934c2cc9fa52e121"
    ),
    "csrc/flash_attn/src/flash.h": (
        "e4c7875a72c0bc5f8ed3e0661ef956ca24b38c8f4758ae2a89f5e58b88671c5a"
    ),
    "csrc/flash_attn/src/flash_fwd_fr13_qrow32_gqa_pair_hdim256_bf16_sm80.cu": (
        "0c18535d6eb74bd8aae420fa139bef1f6115eb711b42db36c50fba27dc066884"
    ),
    "csrc/flash_attn/src/flash_fwd_kernel.h": (
        "4f08741030c46d7e1ef1b88a10d4946f625559fedd7658c3b288e0d7a5d58d13"
    ),
    "csrc/flash_attn/src/utils.h": (
        "5887df63c79a3e42fb9ddad93f64fe3c0625dbee4c547af68b6f2108b7beeb5f"
    ),
}
SOURCE_STATUS = (
    " M csrc/flash_attn/flash_api.cpp",
    " M csrc/flash_attn/flash_api_torch_lib.cpp",
    " M csrc/flash_attn/src/flash.h",
    " M csrc/flash_attn/src/flash_fwd_kernel.h",
    " M csrc/flash_attn/src/utils.h",
    "?? csrc/flash_attn/src/flash_fwd_fr13_qrow32_gqa_pair_hdim256_bf16_sm80.cu",
)
SOURCE_CLOSURE_SHA256 = (
    "9c3f9e751da7b783e9d07d8e40d5bc2234b99e719a1048668bd6c82244ed2d81"
)
ARM_SCHEMA = "fr13.fixed32.fa2_qrow32_gqa_pair_b4_live_verification.v1"
DUAL_SCHEMA = "fr13.fixed32.fa2_qrow32_gqa_pair_b4_dual_gate.v1"
B3_ARM_SCHEMA = (
    "fr13.fixed32.fa2_qrow32_gqa_pair_b3_padded_live_verification.v1"
)
HEX = frozenset("0123456789abcdef")

# FR13_FA2_QROW32_B34_PADDED (Mark's ruling 2026-08-13). The width-3 arm's
# numbers, in one place so the verifier and the record cannot disagree.
B3_WIDTH = 3
CANONICAL_WIDTH = 4
ROWS_PER_SLOT = 32
B3_REAL_ROWS = B3_WIDTH * ROWS_PER_SLOT           # 96
CANONICAL_ROWS = CANONICAL_WIDTH * ROWS_PER_SLOT  # 128
NULL_BLOCK_ID = 0


class GateError(ValueError):
    pass


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> Any:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise GateError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise GateError(f"{label} must be a regular non-symlink file")
    return info


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _regular(path, label)
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                GateError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise GateError(f"{label} root must be an object")
    return payload, raw


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.rstrip("\n")
    except subprocess.CalledProcessError as error:
        raise GateError(f"git {' '.join(args)} failed for {repo}") from error


def _require_commit(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise GateError(f"{label} is not a full lowercase commit")
    return value


def validate_candidate(candidate_so: Path, fa2_source: Path) -> dict[str, Any]:
    info = _regular(candidate_so, "GQA-pair candidate SO")
    if info.st_size != CANDIDATE_SIZE or _sha256_file(candidate_so) != CANDIDATE_SHA256:
        raise GateError("GQA-pair candidate SO identity drifted")
    if not fa2_source.is_dir() or fa2_source.is_symlink():
        raise GateError("FA2 source root must be a non-symlink directory")
    head = _git(fa2_source, "rev-parse", "HEAD")
    if head != FA2_HEAD:
        raise GateError("FA2 source HEAD drifted")
    status_lines = tuple(
        line
        for line in _git(
            fa2_source, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line
    )
    if status_lines != SOURCE_STATUS:
        raise GateError("FA2 regenerated source set drifted")
    files: dict[str, str] = {}
    for relative, expected in SOURCE_FILES.items():
        path = fa2_source / relative
        _regular(path, f"FA2 source {relative}")
        actual = _sha256_file(path)
        if actual != expected:
            raise GateError(f"FA2 source hash drifted: {relative}")
        files[relative] = actual
    closure = {"fa2_head": head, "files": files}
    if _sha256_bytes(_canonical_bytes(closure)) != SOURCE_CLOSURE_SHA256:
        raise GateError("FA2 source closure digest drifted")

    translation_unit = (
        fa2_source
        / "csrc/flash_attn/src/flash_fwd_fr13_qrow32_gqa_pair_hdim256_bf16_sm80.cu"
    )
    if translation_unit.read_text(encoding="ascii") != FIXED32_QUERY_GQA_PAIR32_TRANSLATION_UNIT:
        raise GateError("GQA-pair translation unit differs from the generator")
    api = (fa2_source / "csrc/flash_attn/flash_api.cpp").read_text(
        encoding="ascii"
    )
    if (
        api.count(FIXED32_QUERY_GQA_PAIR32_API_DECLARATION.strip()) != 1
        or api.count(FIXED32_QUERY_GQA_PAIR32_API_GATE.strip()) != 1
    ):
        raise GateError("GQA-pair fail-closed API dispatch drifted")
    return {
        "candidate_arm": CANDIDATE_ARM,
        "candidate_so_sha256": CANDIDATE_SHA256,
        "candidate_so_size": CANDIDATE_SIZE,
        "fa2_head": FA2_HEAD,
        "fa2_source_closure_sha256": SOURCE_CLOSURE_SHA256,
        "selector_sentinel": FIXED32_QUERY_GQA_PAIR32_BATCH_STRIDE_SENTINEL,
    }


def _require_summary(
    value: Any, *, label: str, dtype: str, shape: list[int], byte_count: int
) -> None:
    if not isinstance(value, dict) or (
        value.get("dtype") != dtype
        or value.get("shape") != shape
        or value.get("bytes") != byte_count
        or value.get("raw_byte_mismatches") != 0
    ):
        raise GateError(f"{label} raw-byte contract drifted")
    stock = value.get("stock_sha256")
    candidate = value.get("candidate_sha256")
    if (
        not isinstance(stock, str)
        or len(stock) != 64
        or any(char not in HEX for char in stock)
        or candidate != stock
    ):
        raise GateError(f"{label} digest identity drifted")


def verify_arm(args: argparse.Namespace) -> dict[str, Any]:
    identity = validate_candidate(args.candidate_so, args.fa2_source)
    source_commit = _require_commit(args.source_commit, "source commit")
    try:
        base = qrow32_gate.verify_live(args)
    except qrow32_gate.GateError as error:
        raise GateError(str(error)) from error
    result, result_raw = _load_json(args.result, "GQA-pair live result")
    expected = {
        "candidate_arm": CANDIDATE_ARM,
        "candidate_dispatch": "qrow32 GQA-pair exact geometry; no fallback",
        "candidate_so_sha256": CANDIDATE_SHA256,
        "candidate_so_size": CANDIDATE_SIZE,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "fa2_head": FA2_HEAD,
        "fa2_source_closure_sha256": SOURCE_CLOSURE_SHA256,
        "incumbent_dispatch": "stock FA2 exact geometry; no fallback",
        "selector_sentinel": FIXED32_QUERY_GQA_PAIR32_BATCH_STRIDE_SENTINEL,
        "source_commit": source_commit,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise GateError(f"GQA-pair live result {key} drifted")
    for layer in result["layers"]:
        layer_name = layer["layer_name"]
        _require_summary(
            layer.get("output"),
            label=f"{layer_name} output",
            dtype="torch.bfloat16",
            shape=[128, 24, 256],
            byte_count=128 * 24 * 256 * 2,
        )
        _require_summary(
            layer.get("lse"),
            label=f"{layer_name} LSE",
            dtype="torch.float32",
            shape=[24, 128],
            byte_count=24 * 128 * 4,
        )
        for slot in layer["slots"]:
            slot_id = slot["slot"]
            _require_summary(
                slot.get("output"),
                label=f"{layer_name} slot {slot_id} output",
                dtype="torch.bfloat16",
                shape=[32, 24, 256],
                byte_count=32 * 24 * 256 * 2,
            )
            _require_summary(
                slot.get("lse"),
                label=f"{layer_name} slot {slot_id} LSE",
                dtype="torch.float32",
                shape=[24, 32],
                byte_count=24 * 32 * 4,
            )
    logical_topology = {
        "tail6_fixed32": "Tail23",
        "hydra27_fixed32": "Hydra27",
    }[args.fixed32_mode]
    return {
        "schema": ARM_SCHEMA,
        "status": "PASS",
        "fixed32_mode": args.fixed32_mode,
        "logical_topology": logical_topology,
        **identity,
        "source_commit": source_commit,
        "task_ids": list(qrow32_gate.TASK_IDS),
        "subset_sha256": qrow32_gate.EXACT4_SUBSET_SHA256,
        "layer_count": base["layer_count"],
        "slot_coverage": base["slot_coverage"],
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "live_result_sha256": _sha256_bytes(result_raw),
        "fallback_allowed": False,
        "performance_measurement": False,
    }


def _require_identical_summary(value: Any, *, label: str) -> None:
    """A raw-byte summary whose two sides are the SAME bytes, whatever they are.

    Used for the poisoned-vs-clean shadow comparison, where the requirement is
    bit-identity between two candidate runs rather than agreement with a
    pinned stock digest.
    """
    if not isinstance(value, dict) or value.get("raw_byte_mismatches") != 0:
        raise GateError(f"{label} is not bit-identical")
    left = value.get("stock_sha256")
    right = value.get("candidate_sha256")
    if (
        not isinstance(left, str)
        or len(left) != 64
        or any(char not in HEX for char in left)
        or right != left
    ):
        raise GateError(f"{label} digest identity drifted")


def verify_arm_b3(args: argparse.Namespace) -> dict[str, Any]:
    """The width-3 PADDED arm: real rows equal, shadow provably inert.

    Three independent claims, all of which must hold:

      1. output[0:96] and softmax_lse[:, 0:96] are byte-identical between the
         NATIVE width-3 stock call and the PADDED canonical width-4 candidate
         call, on all 16 target layers.
      2. The same real rows are byte-identical between the clean-shadow and
         POISONED-shadow candidate runs. The poisoned run fills the shadow's Q
         rows with NaN and points its block-table row at a page index that
         cannot exist; under seqused_k[3] == 0 the kernel reads neither, so
         any difference means the early exit was not taken.
      3. The shadow half is exactly what the early return writes: zeros in the
         32 O rows, +INF in the 32 LSE entries. The runtime asserts this on
         device and reports the failures; an empty failure list is required.

    The .so identity, the FA2 source closure and the C++ dispatch are checked
    by validate_candidate exactly as for width 4 -- the widening is python-side
    only and this verifier must not pretend otherwise.
    """
    identity = validate_candidate(args.candidate_so, args.fa2_source)
    source_commit = _require_commit(args.source_commit, "source commit")
    result, result_raw = _load_json(args.result, "GQA-pair b3 padded result")
    expected = {
        "schema": "fr13.fixed32.fa2_qrow32_live_paged_exact4_ab.v1",
        "status": "PASS",
        "batch_size": B3_WIDTH,
        "concurrency": B3_WIDTH,
        "physical_rows_per_slot": ROWS_PER_SLOT,
        "total_query_rows": B3_REAL_ROWS,
        "padded_to_canonical_width": True,
        "canonical_width": CANONICAL_WIDTH,
        "canonical_query_rows": CANONICAL_ROWS,
        "shadow_slot": CANONICAL_WIDTH - 1,
        "shadow_seqused_k": 0,
        "shadow_block_table_page": NULL_BLOCK_ID,
        "poisoned_shadow_arm": True,
        "poisoned_shadow_output_raw_byte_mismatches": 0,
        "poisoned_shadow_lse_raw_byte_mismatches": 0,
        "shadow_contract_failures": [],
        "candidate_arm": CANDIDATE_ARM,
        "candidate_so_sha256": CANDIDATE_SHA256,
        "candidate_so_size": CANDIDATE_SIZE,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "fa2_head": FA2_HEAD,
        "fa2_source_closure_sha256": SOURCE_CLOSURE_SHA256,
        "fixed32_mode": args.fixed32_mode,
        "runtime_mode": "FULL",
        "layer_count": 16,
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "selector_sentinel": FIXED32_QUERY_GQA_PAIR32_BATCH_STRIDE_SENTINEL,
        "source_commit": source_commit,
        "fallback_allowed": False,
        "performance_measurement": False,
        "served_return": "stock captured graph output unchanged",
        "task_ids": list(qrow32_gate.TASK_IDS),
        "subset_sha256": qrow32_gate.EXACT4_SUBSET_SHA256,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise GateError(f"GQA-pair b3 padded result {key} drifted")
    operands = result.get("operands")
    if not isinstance(operands, dict) or (
        operands.get("query_shape") != [B3_REAL_ROWS, 24, 256]
        or operands.get("slot_coverage") != list(range(B3_WIDTH))
        or operands.get("query_start_loc")
        != list(range(0, B3_REAL_ROWS + ROWS_PER_SLOT, ROWS_PER_SLOT))
    ):
        raise GateError("GQA-pair b3 padded operand geometry drifted")
    layers = result.get("layers")
    if not isinstance(layers, list) or len(layers) != 16:
        raise GateError("GQA-pair b3 padded layer coverage drifted")
    for layer in layers:
        layer_name = layer["layer_name"]
        _require_summary(
            layer.get("output"),
            label=f"{layer_name} b3 output",
            dtype="torch.bfloat16",
            shape=[B3_REAL_ROWS, 24, 256],
            byte_count=B3_REAL_ROWS * 24 * 256 * 2,
        )
        _require_summary(
            layer.get("lse"),
            label=f"{layer_name} b3 LSE",
            dtype="torch.float32",
            shape=[24, B3_REAL_ROWS],
            byte_count=24 * B3_REAL_ROWS * 4,
        )
        slots = layer.get("slots")
        if not isinstance(slots, list) or len(slots) != B3_WIDTH:
            raise GateError(f"{layer_name} b3 slot coverage drifted")
        for slot in slots:
            slot_id = slot["slot"]
            _require_summary(
                slot.get("output"),
                label=f"{layer_name} b3 slot {slot_id} output",
                dtype="torch.bfloat16",
                shape=[ROWS_PER_SLOT, 24, 256],
                byte_count=ROWS_PER_SLOT * 24 * 256 * 2,
            )
            _require_summary(
                slot.get("lse"),
                label=f"{layer_name} b3 slot {slot_id} LSE",
                dtype="torch.float32",
                shape=[24, ROWS_PER_SLOT],
                byte_count=24 * ROWS_PER_SLOT * 4,
            )
        poisoned = layer.get("poisoned_shadow")
        if not isinstance(poisoned, dict) or (
            poisoned.get("shadow_rows") != [B3_REAL_ROWS, CANONICAL_ROWS]
            or poisoned.get("shadow_seqused_k") != 0
            or poisoned.get("shadow_query_fill") != "nan"
            or not isinstance(poisoned.get("shadow_block_table_page"), int)
            or int(poisoned["shadow_block_table_page"]) == NULL_BLOCK_ID
        ):
            raise GateError(f"{layer_name} poisoned-shadow declaration drifted")
        _require_identical_summary(
            poisoned.get("output"),
            label=f"{layer_name} poisoned-shadow output",
        )
        _require_identical_summary(
            poisoned.get("lse"), label=f"{layer_name} poisoned-shadow LSE"
        )
    logical_topology = {
        "tail6_fixed32": "Tail23",
        "hydra27_fixed32": "Hydra27",
    }[args.fixed32_mode]
    return {
        "schema": B3_ARM_SCHEMA,
        "status": "PASS",
        "fixed32_mode": args.fixed32_mode,
        "logical_topology": logical_topology,
        **identity,
        "source_commit": source_commit,
        "task_ids": list(qrow32_gate.TASK_IDS),
        "subset_sha256": qrow32_gate.EXACT4_SUBSET_SHA256,
        "batch_size": B3_WIDTH,
        "padded_to_canonical_width": True,
        "canonical_width": CANONICAL_WIDTH,
        "shadow_slot": CANONICAL_WIDTH - 1,
        "layer_count": 16,
        "slot_coverage": list(range(B3_WIDTH)),
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "poisoned_shadow_output_raw_byte_mismatches": 0,
        "poisoned_shadow_lse_raw_byte_mismatches": 0,
        "shadow_contract_failures": [],
        "live_result_sha256": _sha256_bytes(result_raw),
        "fallback_allowed": False,
        "performance_measurement": False,
    }


def _verify_dual_b3(
    args: argparse.Namespace, identity: dict[str, Any], source_commit: str
) -> dict[str, Any]:
    """Fold the two width-3 padded arm verifications into the dual gate.

    THE CREDENTIAL'S WIDTH SCOPE IS DECIDED HERE AND NOWHERE ELSE. A dual gate
    built without the b3 verifications qualifies width 4 ONLY, and the pass
    sidecar issued from it authorises width 4 only; padded width-3 serving
    needs the b3 evidence to exist, on BOTH topologies, at THIS commit, from
    the same pinned binary. Supplying one of the two is a hard error rather
    than a silent downgrade -- a half-gated widening is exactly the failure
    this clause exists to prevent.
    """
    tail_path = getattr(args, "tail_b3_verification", None)
    hydra_path = getattr(args, "hydra_b3_verification", None)
    if tail_path is None and hydra_path is None:
        return {"qualified_widths": [CANONICAL_WIDTH]}
    if tail_path is None or hydra_path is None:
        raise GateError(
            "the b3 padded widening needs BOTH topology verifications"
        )
    tail, tail_raw = _load_json(tail_path, "Tail23 b3 verification")
    hydra, hydra_raw = _load_json(hydra_path, "Hydra27 b3 verification")
    for payload, mode, topology in (
        (tail, "tail6_fixed32", "Tail23"),
        (hydra, "hydra27_fixed32", "Hydra27"),
    ):
        live_result_sha256 = payload.get("live_result_sha256")
        if (
            payload.get("schema") != B3_ARM_SCHEMA
            or payload.get("status") != "PASS"
            or payload.get("fixed32_mode") != mode
            or payload.get("logical_topology") != topology
            or payload.get("source_commit") != source_commit
            or payload.get("task_ids") != list(qrow32_gate.TASK_IDS)
            or payload.get("subset_sha256") != qrow32_gate.EXACT4_SUBSET_SHA256
            or payload.get("batch_size") != B3_WIDTH
            or payload.get("padded_to_canonical_width") is not True
            or payload.get("canonical_width") != CANONICAL_WIDTH
            or payload.get("shadow_slot") != CANONICAL_WIDTH - 1
            or payload.get("layer_count") != 16
            or payload.get("slot_coverage") != list(range(B3_WIDTH))
            or payload.get("output_raw_byte_mismatches") != 0
            or payload.get("lse_raw_byte_mismatches") != 0
            or payload.get("poisoned_shadow_output_raw_byte_mismatches") != 0
            or payload.get("poisoned_shadow_lse_raw_byte_mismatches") != 0
            or payload.get("shadow_contract_failures") != []
            or payload.get("fallback_allowed") is not False
            or payload.get("performance_measurement") is not False
            or not isinstance(live_result_sha256, str)
            or len(live_result_sha256) != 64
            or any(char not in HEX for char in live_result_sha256)
        ):
            raise GateError(f"{topology} GQA-pair b3 verification drifted")
        for key, value in identity.items():
            if payload.get(key) != value:
                raise GateError(f"{topology} GQA-pair b3 identity {key} drifted")
    return {
        "qualified_widths": [B3_WIDTH, CANONICAL_WIDTH],
        "tail23_b3_verification_sha256": _sha256_bytes(tail_raw),
        "hydra27_b3_verification_sha256": _sha256_bytes(hydra_raw),
        "b3_padded_to_canonical_width": True,
        "b3_shadow_slot": CANONICAL_WIDTH - 1,
        "b3_output_raw_byte_mismatches": 0,
        "b3_lse_raw_byte_mismatches": 0,
        "b3_poisoned_shadow_output_raw_byte_mismatches": 0,
        "b3_poisoned_shadow_lse_raw_byte_mismatches": 0,
        "b3_shadow_contract_failures": [],
    }


def verify_dual(args: argparse.Namespace) -> dict[str, Any]:
    identity = validate_candidate(args.candidate_so, args.fa2_source)
    source_commit = _require_commit(args.source_commit, "source commit")
    tail, tail_raw = _load_json(args.tail_verification, "Tail23 verification")
    hydra, hydra_raw = _load_json(args.hydra_verification, "Hydra27 verification")
    for payload, mode, topology in (
        (tail, "tail6_fixed32", "Tail23"),
        (hydra, "hydra27_fixed32", "Hydra27"),
    ):
        live_result_sha256 = payload.get("live_result_sha256")
        if (
            payload.get("schema") != ARM_SCHEMA
            or payload.get("status") != "PASS"
            or payload.get("fixed32_mode") != mode
            or payload.get("logical_topology") != topology
            or payload.get("source_commit") != source_commit
            or payload.get("task_ids") != list(qrow32_gate.TASK_IDS)
            or payload.get("subset_sha256") != qrow32_gate.EXACT4_SUBSET_SHA256
            or payload.get("layer_count") != 16
            or payload.get("slot_coverage") != [0, 1, 2, 3]
            or payload.get("output_raw_byte_mismatches") != 0
            or payload.get("lse_raw_byte_mismatches") != 0
            or payload.get("fallback_allowed") is not False
            or payload.get("performance_measurement") is not False
            or not isinstance(live_result_sha256, str)
            or len(live_result_sha256) != 64
            or any(char not in HEX for char in live_result_sha256)
        ):
            raise GateError(f"{topology} GQA-pair arm verification drifted")
        for key, value in identity.items():
            if payload.get(key) != value:
                raise GateError(f"{topology} GQA-pair identity {key} drifted")
    return {
        "schema": DUAL_SCHEMA,
        "status": "PASS",
        **identity,
        "source_commit": source_commit,
        "task_ids": list(qrow32_gate.TASK_IDS),
        "subset_sha256": qrow32_gate.EXACT4_SUBSET_SHA256,
        "qualified_topologies": ["Tail23", "Hydra27"],
        "tail23_verification_sha256": _sha256_bytes(tail_raw),
        "hydra27_verification_sha256": _sha256_bytes(hydra_raw),
        **_verify_dual_b3(args, identity, source_commit),
        "layer_count_per_topology": 16,
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "fallback_allowed": False,
        "performance_measurement": False,
        "timing_eligible": False,
        "production_eligible": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser("validate-candidate")
    candidate.add_argument("--candidate-so", type=Path, required=True)
    candidate.add_argument("--fa2-source", type=Path, required=True)

    arm = commands.add_parser("verify-arm")
    arm.add_argument("--result", type=Path, required=True)
    arm.add_argument("--campaign-arm", type=Path, required=True)
    arm.add_argument("--campaign-provenance", type=Path, required=True)
    arm.add_argument("--candidate-so", type=Path, required=True)
    arm.add_argument("--fa2-source", type=Path, required=True)
    arm.add_argument(
        "--fixed32-mode",
        choices=("tail6_fixed32", "hydra27_fixed32"),
        required=True,
    )
    arm.add_argument("--source-commit", required=True)

    arm_b3 = commands.add_parser("verify-arm-b3")
    arm_b3.add_argument("--result", type=Path, required=True)
    arm_b3.add_argument("--candidate-so", type=Path, required=True)
    arm_b3.add_argument("--fa2-source", type=Path, required=True)
    arm_b3.add_argument(
        "--fixed32-mode",
        choices=("tail6_fixed32", "hydra27_fixed32"),
        required=True,
    )
    arm_b3.add_argument("--source-commit", required=True)

    dual = commands.add_parser("verify-dual")
    dual.add_argument("--tail-verification", type=Path, required=True)
    dual.add_argument("--hydra-verification", type=Path, required=True)
    # OPTIONAL, and both-or-neither. Present => the gate qualifies widths 3
    # and 4 and the credential it feeds may authorise padded width-3 serving;
    # absent => the gate qualifies width 4 only, exactly as the sealed
    # width-4 lineage did.
    dual.add_argument("--tail-b3-verification", type=Path, default=None)
    dual.add_argument("--hydra-b3-verification", type=Path, default=None)
    dual.add_argument("--candidate-so", type=Path, required=True)
    dual.add_argument("--fa2-source", type=Path, required=True)
    dual.add_argument("--source-commit", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-candidate":
        result = validate_candidate(args.candidate_so, args.fa2_source)
    elif args.command == "verify-arm":
        result = verify_arm(args)
    elif args.command == "verify-arm-b3":
        result = verify_arm_b3(args)
    else:
        result = verify_dual(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
