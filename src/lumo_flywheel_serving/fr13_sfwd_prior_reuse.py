from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

import torch
import triton
import triton.language as tl

from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
    _FR13_FIXED32_MODE,
    _FR13_FIXED32_MODES,
    _FR13_FIXED32_PARENT,
    _fr13_fixed32_batch_gdn_byte_diff,
    _fr13_fixed32_batch_gdn_real_event_marker,
)


CANDIDATE = "fixed32_sfwd_prior_reuse_rowgroup32_c64_v1"
ROWS_PER_PROGRAM = 32
BLOCK_C = 64
CHANNELS = 10240
CONV_STATE_LEN = 34
ENABLED_PATH = "/logs/fr13_fixed32_sfwd_prior_reuse_byte_ab.enabled"
REAL_EVENT_PATH = "/logs/fr13_fixed32_sfwd_state_fusion.real_event.arm"
PASS_PATH = "/logs/fr13_fixed32_sfwd_prior_reuse.live_pass.json"
RECORDS_PATH = "/logs/fr13_fixed32_sfwd_prior_reuse.byte_ab.jsonl"
SOURCE_MANIFEST_SCHEMA = "fr13.fixed32.sfwd_prior_reuse.source_manifest.v1"
SOURCE_RELATIVE_PATH = "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse.py"
_STATE = {
    "task_marker": None,
    "batch": None,
    "passed": {},
    "attempts": {},
    "source_binding": None,
}


def fixed32_sfwd_prior_reuse_contract(
    batch_size: int,
    *,
    tree_rows: int,
    conv_width: int,
    conv_state_len: int,
) -> dict[str, object]:
    """Validate the closed rowgroup32/C64 prior-reuse geometry."""
    batch = int(batch_size)
    rows = int(tree_rows)
    width = int(conv_width)
    state_len = int(conv_state_len)
    if batch not in (1, 2, 3, 4):
        raise ValueError(f"FR13 SFWD prior-reuse requires B1-B4, got B={batch}")
    if rows != 32:
        raise ValueError(
            "FR13 SFWD prior-reuse requires exactly 32 physical rows per "
            f"request, got {rows}"
        )
    if width != 4 or state_len != CONV_STATE_LEN:
        raise ValueError(
            "FR13 SFWD prior-reuse requires width/state geometry "
            f"(4, {CONV_STATE_LEN}), got ({width}, {state_len})"
        )
    source_rows = width - 1 + rows + 1
    return {
        "candidate": CANDIDATE,
        "batch_size": batch,
        "physical_rows_per_request": rows,
        "logical_rows": batch * rows,
        "conv_width": width,
        "conv_state_len": state_len,
        "channels": CHANNELS,
        "source_rows_per_request": source_rows,
        "source_rows": batch * source_rows,
        "conv_state_launches_per_layer": 1,
        "conv_rows_per_program": ROWS_PER_PROGRAM,
        "conv_row_groups_per_request": 1,
        "conv_block_c": BLOCK_C,
        "gdn_level_path_programs": (batch, 11 * batch),
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export": True,
        "gdn_flags_export": True,
        "reference_always_served": True,
    }


def fixed32_sfwd_prior_reuse_gate_control(
    *,
    environ=None,
    enabled_path: str | None = None,
    event_path: str | None = None,
) -> tuple[bool, str | None]:
    """Resolve the independent default-off authenticated byte gate."""
    env = os.environ if environ is None else environ
    raw = str(env.get("FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB", ""))
    if raw not in ("", "0", "1"):
        raise RuntimeError(
            "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB must be exactly 0 or 1"
        )
    enabled = enabled_path or str(
        env.get("FR13_FIXED32_SFWD_PRIOR_REUSE_ENABLED_PATH", ENABLED_PATH)
    )
    event = event_path or str(
        env.get("FR13_FIXED32_SFWD_STATE_FUSION_REAL_EVENT_PATH", REAL_EVENT_PATH)
    )
    armed = raw == "1" or os.path.exists(enabled)
    if not armed or not os.path.exists(event):
        return armed, None
    if _FR13_FIXED32_MODE not in _FR13_FIXED32_MODES:
        raise RuntimeError(
            "FR13 SFWD prior-reuse byte gate requires an exact fixed32 runtime"
        )
    return True, _fr13_fixed32_batch_gdn_real_event_marker(event)


def _source_binding(
    *,
    manifest_path: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_source_commit: str | None = None,
    environ=None,
) -> dict[str, str]:
    env = os.environ if environ is None else environ
    path_raw = str(
        manifest_path
        or env.get("FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_PATH", "")
    )
    expected_sha = str(
        expected_manifest_sha256
        or env.get("FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_SHA256", "")
    )
    source_commit = str(
        expected_source_commit
        or env.get("FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_COMMIT", "")
    )
    if (
        not path_raw
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise RuntimeError(
            "FR13 SFWD prior-reuse requires complete source-manifest credentials"
        )
    path = Path(path_raw)
    try:
        info = os.lstat(path)
        raw = path.read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"FR13 SFWD prior-reuse cannot read source manifest: {error}"
        ) from error
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or not raw
    ):
        raise RuntimeError(
            "FR13 SFWD prior-reuse source manifest must be one nonempty regular file"
        )
    observed_sha = hashlib.sha256(raw).hexdigest()
    if observed_sha != expected_sha:
        raise RuntimeError("FR13 SFWD prior-reuse source-manifest SHA-256 drift")
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "FR13 SFWD prior-reuse source manifest is not canonical ASCII JSON"
        ) from error
    files = payload.get("files")
    expected_source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_entry = files.get(SOURCE_RELATIVE_PATH) if isinstance(files, dict) else None
    if (
        payload.get("schema") != SOURCE_MANIFEST_SCHEMA
        or payload.get("candidate") != CANDIDATE
        or payload.get("source_commit") != source_commit
        or not isinstance(source_entry, dict)
        or source_entry.get("sha256") != expected_source_sha
    ):
        raise RuntimeError(
            "FR13 SFWD prior-reuse source manifest is not bound to runtime source"
        )
    return {
        "source_commit": source_commit,
        "source_manifest_sha256": observed_sha,
        "candidate_source_sha256": expected_source_sha,
    }


def _emit(record: dict[str, object]) -> None:
    path = os.environ.get("FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB_PATH", RECORDS_PATH)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = dict(record)
    payload["schema"] = "fr13.fixed32.sfwd_prior_reuse.byte_ab.v1"
    with open(path, "a", encoding="ascii") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _invalidate_pass() -> None:
    path = os.environ.get("FR13_FIXED32_SFWD_PRIOR_REUSE_PASS_PATH", PASS_PATH)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise RuntimeError(
            "FR13 SFWD prior-reuse could not invalidate stale live PASS"
        ) from error


def _pass_emit(
    *,
    task_marker: str,
    batch: int,
    layers: dict[int, str],
    source_binding: dict[str, str],
) -> None:
    if len(layers) != 48 or len(set(layers.values())) != 48:
        return
    if int(batch) != 1:
        raise RuntimeError("FR13 SFWD prior-reuse live PASS is B1-only")
    draft_vocab_k = int(os.environ.get("FR13_DRAFT_VOCAB_K", "0") or 0)
    draft_vocab_root = int(os.environ.get("FR13_DRAFT_VOCAB_ROOT", "0") or 0)
    draft_vocab_blocks = os.environ.get("FR13_DRAFT_VOCAB_BLOCKS", "")
    try:
        draft_vocab_blocks_sha256 = hashlib.sha256(
            Path(draft_vocab_blocks).read_bytes()
        ).hexdigest()
    except OSError as error:
        raise RuntimeError(
            "FR13 SFWD prior-reuse K64/root1 block map is unreadable"
        ) from error
    expected_blocks_sha256 = (
        "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
    )
    if (
        draft_vocab_k != 65536
        or draft_vocab_root != 1
        or draft_vocab_blocks_sha256 != expected_blocks_sha256
    ):
        raise RuntimeError("FR13 SFWD prior-reuse live PASS requires audited K64/root1")
    path = os.environ.get("FR13_FIXED32_SFWD_PRIOR_REUSE_PASS_PATH", PASS_PATH)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "schema": "fr13.fixed32.sfwd_prior_reuse.live_pass.v1",
        "status": "byte_pass_source_only",
        "run_classification": ("one_real_swe_verified_k64_root_b1_byte_diagnostic"),
        "candidate": CANDIDATE,
        **source_binding,
        "task_marker": task_marker,
        "batch": int(batch),
        "draft_vocab_k": draft_vocab_k,
        "draft_vocab_root": draft_vocab_root,
        "draft_vocab_blocks_sha256": draft_vocab_blocks_sha256,
        "layer_count": 48,
        "layers": [
            {
                "layer_key": f"0x{key:x}",
                "layer_prefix_sha256": layers[key],
            }
            for key in sorted(layers)
        ],
        "physical_rows_per_request": 32,
        "conv_rows_per_program": ROWS_PER_PROGRAM,
        "conv_block_c": BLOCK_C,
        "candidate_conv_launches_per_layer": 1,
        "gdn_level_path_programs": [int(batch), 11 * int(batch)],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
        "real_task_authenticated": True,
        "reference_always_served": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
        "production_blocker": (
            "source-only candidate requires later clean timing qualification"
        ),
    }
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="ascii") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def fixed32_sfwd_prior_reuse_byte_gate(
    *,
    task_marker: str,
    layer_prefix: str,
    layer_key: int,
    batch_size: int,
    reference_out: torch.Tensor,
    candidate_out: torch.Tensor,
    reference_source_stage: torch.Tensor,
    candidate_source_stage: torch.Tensor,
    source_manifest_path: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    expected_source_commit: str | None = None,
) -> dict[str, object]:
    """Compare both candidate surfaces while always serving the reference."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FR13 SFWD prior-reuse byte gate is eager-only")
    prefix = "swe_verified:"
    task_id = (
        task_marker[len(prefix) :]
        if isinstance(task_marker, str) and task_marker.startswith(prefix)
        else ""
    )
    if not task_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
        for character in task_id
    ):
        raise RuntimeError(
            "FR13 SFWD prior-reuse requires an authenticated SWE-Verified marker"
        )
    batch = int(batch_size)
    if batch != 1:
        raise RuntimeError("FR13 SFWD prior-reuse byte gate is B1-only")
    fixed32_sfwd_prior_reuse_contract(
        batch,
        tree_rows=32,
        conv_width=4,
        conv_state_len=CONV_STATE_LEN,
    )
    if not isinstance(layer_prefix, str) or not layer_prefix:
        raise RuntimeError("FR13 SFWD prior-reuse requires a layer prefix")
    layer_prefix_sha256 = hashlib.sha256(layer_prefix.encode("utf-8")).hexdigest()
    state = _STATE
    if state["task_marker"] is None:
        state["task_marker"] = task_marker
        state["batch"] = batch
    elif state["task_marker"] != task_marker or int(state["batch"]) != batch:
        raise RuntimeError(
            "FR13 SFWD prior-reuse cannot combine task markers or batch sizes"
        )
    if state["source_binding"] is None:
        state["source_binding"] = _source_binding(
            manifest_path=source_manifest_path,
            expected_manifest_sha256=expected_source_manifest_sha256,
            expected_source_commit=expected_source_commit,
        )
    source_binding = dict(state["source_binding"])
    key = int(layer_key)
    if key not in state["passed"] and len(state["passed"]) >= 48:
        _invalidate_pass()
        raise RuntimeError("FR13 SFWD prior-reuse observed more than 48 layers")
    prior_prefix = state["passed"].get(key)
    if prior_prefix is not None and prior_prefix != layer_prefix_sha256:
        raise RuntimeError(
            "FR13 SFWD prior-reuse layer key was reused by another prefix"
        )
    attempt = int(state["attempts"].get(key, 0)) + 1
    state["attempts"][key] = attempt
    comparisons = [
        _fr13_fixed32_batch_gdn_byte_diff("conv_out", reference_out, candidate_out),
        _fr13_fixed32_batch_gdn_byte_diff(
            "commit_source_stage",
            reference_source_stage,
            candidate_source_stage,
        ),
    ]
    first_nonzero = next(
        (item for item in comparisons if not bool(item["byte_equal"])), None
    )
    passed = first_nonzero is None
    record = {
        "status": "pass" if passed else "mismatch_reference_served",
        "candidate": CANDIDATE,
        **source_binding,
        "task_marker": task_marker,
        "batch": batch,
        "layer_key": f"0x{key:x}",
        "layer_prefix_sha256": layer_prefix_sha256,
        "attempt": attempt,
        "physical_rows_per_request": 32,
        "conv_rows_per_program": ROWS_PER_PROGRAM,
        "conv_block_c": BLOCK_C,
        "candidate_conv_launches_per_layer": 1,
        "gdn_level_path_programs": [batch, 11 * batch],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "comparisons": comparisons,
        "first_nonzero": first_nonzero,
        "zero_diff": passed,
        "real_task_authenticated": True,
        "reference_always_served": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }
    _emit(record)
    if passed:
        state["passed"][key] = layer_prefix_sha256
        _pass_emit(
            task_marker=task_marker,
            batch=batch,
            layers=dict(state["passed"]),
            source_binding=source_binding,
        )
    else:
        state["passed"].pop(key, None)
        _invalidate_pass()
    return record


@triton.jit
def _fr13_fixed32_sfwd_prior_reuse_kernel(
    x,
    conv_state,
    spec_state_indices,
    source_flat,
    conv_weights,
    bias,
    out,
    source_stage,
    x_stride_row,
    conv_stride_row,
    conv_stride_c,
    conv_stride_l,
    ssi_stride_b,
    ssi_stride_s,
    weight_stride_c,
    weight_stride_w,
    B: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fuse fixed32 convolution and commit-source staging with prior reuse."""
    pid_row_group = tl.program_id(0)
    pid_c = tl.program_id(1)
    row_groups = N // ROWS_PER_PROGRAM
    pid_b = pid_row_group // row_groups
    pid_n_group = pid_row_group - pid_b * row_groups
    pid_n_base = pid_n_group * ROWS_PER_PROGRAM
    offs_n = pid_n_base + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]

    bank_row = tl.load(spec_state_indices + pid_b * ssi_stride_b + 0 * ssi_stride_s).to(
        tl.int64
    )
    stage_base = pid_b.to(tl.int64) * SOURCE_ROWS
    prior_0 = tl.load(
        conv_state
        + bank_row * conv_stride_row
        + offs_c * conv_stride_c
        + 0 * conv_stride_l
    )
    prior_1 = tl.load(
        conv_state
        + bank_row * conv_stride_row
        + offs_c * conv_stride_c
        + 1 * conv_stride_l
    )
    prior_2 = tl.load(
        conv_state
        + bank_row * conv_stride_row
        + offs_c * conv_stride_c
        + 2 * conv_stride_l
    )
    acc = tl.zeros((ROWS_PER_PROGRAM, BLOCK_C), dtype=tl.float32)
    if HAS_BIAS:
        acc = tl.load(bias + offs_c).to(tl.float32)
    for tap in tl.static_range(0, WIDTH - 1):
        source_row = tl.load(source_flat + offs_n * WIDTH + tap).to(tl.int64)
        from_prior = source_row < (WIDTH - 1)
        prior_value = tl.where(
            source_row == 0,
            prior_0,
            tl.where(source_row == 1, prior_1, prior_2),
        )
        x_node = source_row - (WIDTH - 1)
        x_value = tl.load(
            x + (pid_b.to(tl.int64) * N + x_node) * x_stride_row + offs_c,
            mask=(~from_prior) & (x_node >= 0) & (x_node < N),
            other=0.0,
        )
        value = tl.where(from_prior, prior_value, x_value).to(tl.bfloat16)
        weight = tl.load(
            conv_weights + offs_c * weight_stride_c + tap * weight_stride_w
        ).to(tl.bfloat16)
        product = (value * weight).to(tl.bfloat16).to(tl.float32)
        acc = acc + product

    current_x = tl.load(x + (pid_b.to(tl.int64) * N + offs_n) * x_stride_row + offs_c)
    current_weight = tl.load(
        conv_weights + offs_c * weight_stride_c + (WIDTH - 1) * weight_stride_w
    ).to(tl.bfloat16)
    current_product = (current_x * current_weight).to(tl.bfloat16).to(tl.float32)
    acc = acc + current_product

    activated = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out + (pid_b * N + offs_n) * C + offs_c, activated)
    tl.store(
        source_stage + (stage_base + (WIDTH - 1) + offs_n) * C + offs_c,
        current_x,
    )
    source_edge_writer = pid_n_base == 0
    tl.store(
        source_stage + stage_base * C + offs_c,
        prior_0,
        mask=source_edge_writer,
    )
    tl.store(
        source_stage + (stage_base + 1) * C + offs_c,
        prior_1,
        mask=source_edge_writer,
    )
    tl.store(
        source_stage + (stage_base + 2) * C + offs_c,
        prior_2,
        mask=source_edge_writer,
    )
    tl.store(
        source_stage + (stage_base + SOURCE_ROWS - 1) * C + offs_c,
        0.0,
        mask=source_edge_writer,
    )


def _source_flat_expected(width: int = 4) -> tuple[int, ...]:
    rows: list[int] = []
    for node in range(len(_FR13_FIXED32_PARENT)):
        path = []
        cursor = node
        while cursor >= 0:
            path.append(cursor)
            cursor = _FR13_FIXED32_PARENT[cursor]
        path.reverse()
        source = list(range(width - 1)) + [width - 1 + path_node for path_node in path]
        rows.extend(source[-width:])
    return tuple(rows)


def launch_fixed32_sfwd_prior_reuse(
    *,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    source_flat: torch.Tensor,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    source_stage: torch.Tensor,
    batch_size: int,
    tree_rows: int,
) -> dict[str, object]:
    """Launch the default-off rowgroup32/C64 prior-reuse candidate."""
    if _FR13_FIXED32_MODE not in _FR13_FIXED32_MODES:
        raise RuntimeError("FR13 SFWD prior-reuse requires an exact fixed32 mode")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FR13 SFWD prior-reuse is eager byte-gate-only")
    batch = int(batch_size)
    rows = int(tree_rows)
    if conv_state.ndim != 3:
        raise ValueError("FR13 SFWD prior-reuse conv_state must be [bank,C,L]")
    channels = int(conv_state.shape[1])
    state_len = int(conv_state.shape[2])
    width = int(conv_weights.shape[1]) if conv_weights.ndim == 2 else -1
    contract = fixed32_sfwd_prior_reuse_contract(
        batch,
        tree_rows=rows,
        conv_width=width,
        conv_state_len=state_len,
    )
    required_rows = batch * rows
    source_rows_per_batch = int(contract["source_rows_per_request"])
    required_source_rows = batch * source_rows_per_batch
    tensors = (
        x,
        conv_state,
        spec_state_indices,
        source_flat,
        conv_weights,
        out,
        source_stage,
    )
    if any(not torch.is_tensor(tensor) for tensor in tensors):
        raise TypeError("FR13 SFWD prior-reuse operands must all be tensors")
    device = x.device
    if device.type != "cuda" or any(tensor.device != device for tensor in tensors):
        raise ValueError("FR13 SFWD prior-reuse operands must share one CUDA device")
    if x.dtype != torch.bfloat16 or any(
        tensor.dtype != torch.bfloat16
        for tensor in (conv_state, conv_weights, out, source_stage)
    ):
        raise ValueError("FR13 SFWD prior-reuse requires exact BF16 operands")
    if bias is not None and (
        not torch.is_tensor(bias)
        or bias.device != device
        or bias.dtype not in (torch.bfloat16, torch.float32)
        or bias.ndim != 1
        or int(bias.numel()) != channels
    ):
        raise ValueError("FR13 SFWD prior-reuse bias must be BF16/FP32 [C] or None")
    geometry_failures = []
    if x.ndim != 2:
        geometry_failures.append("x_ndim")
    if tuple(int(value) for value in x.shape) != (required_rows, channels):
        geometry_failures.append("x_shape")
    if out.shape != x.shape:
        geometry_failures.append("out_shape")
    if conv_weights.shape != (channels, width):
        geometry_failures.append("conv_weights_shape")
    if spec_state_indices.ndim != 2:
        geometry_failures.append("spec_state_indices_ndim")
    if channels != CHANNELS:
        geometry_failures.append("channels")
    if spec_state_indices.ndim < 1 or int(spec_state_indices.shape[0]) < batch:
        geometry_failures.append("spec_state_indices_batch")
    if spec_state_indices.ndim < 2 or int(spec_state_indices.shape[1]) < 1:
        geometry_failures.append("spec_state_indices_width")
    if spec_state_indices.dtype != torch.int32:
        geometry_failures.append("spec_state_indices_dtype")
    if source_flat.ndim != 1:
        geometry_failures.append("source_flat_ndim")
    if source_flat.numel() != rows * width:
        geometry_failures.append("source_flat_numel")
    if source_flat.dtype not in (torch.int32, torch.int64):
        geometry_failures.append("source_flat_dtype")
    if source_stage.ndim != 2:
        geometry_failures.append("source_stage_ndim")
    if source_stage.ndim < 1 or int(source_stage.shape[0]) < required_source_rows:
        geometry_failures.append("source_stage_rows")
    if source_stage.ndim < 2 or int(source_stage.shape[1]) != channels:
        geometry_failures.append("source_stage_channels")
    if x.ndim == 2 and int(x.stride(1)) != 1:
        geometry_failures.append("x_channel_stride")
    if x.ndim == 2 and int(x.stride(0)) < channels:
        geometry_failures.append("x_row_stride")
    if not out.is_contiguous():
        geometry_failures.append("out_contiguous")
    if not source_flat.is_contiguous():
        geometry_failures.append("source_flat_contiguous")
    if not source_stage.is_contiguous():
        geometry_failures.append("source_stage_contiguous")
    if geometry_failures:
        observed = {
            "batch": batch,
            "tree_rows": rows,
            "channels": channels,
            "required_rows": required_rows,
            "required_source_rows": required_source_rows,
            "x": (tuple(x.shape), tuple(x.stride()), str(x.dtype)),
            "out": (tuple(out.shape), tuple(out.stride()), str(out.dtype)),
            "conv_state": (
                tuple(conv_state.shape),
                tuple(conv_state.stride()),
                str(conv_state.dtype),
            ),
            "conv_weights": (
                tuple(conv_weights.shape),
                tuple(conv_weights.stride()),
                str(conv_weights.dtype),
            ),
            "spec_state_indices": (
                tuple(spec_state_indices.shape),
                tuple(spec_state_indices.stride()),
                str(spec_state_indices.dtype),
            ),
            "source_flat": (
                tuple(source_flat.shape),
                tuple(source_flat.stride()),
                str(source_flat.dtype),
            ),
            "source_stage": (
                tuple(source_stage.shape),
                tuple(source_stage.stride()),
                str(source_stage.dtype),
            ),
        }
        raise ValueError(
            "FR13 SFWD prior-reuse operand geometry/layout drift: "
            f"failed={geometry_failures!r}; observed={observed!r}"
        )
    actual_source_flat = tuple(
        int(value) for value in source_flat.detach().cpu().tolist()
    )
    if actual_source_flat != _source_flat_expected(width):
        raise RuntimeError("FR13 SFWD prior-reuse source descriptor drift")

    grid = (batch, triton.cdiv(channels, BLOCK_C))
    bias_arg = bias if bias is not None else x
    _fr13_fixed32_sfwd_prior_reuse_kernel[grid](
        x,
        conv_state,
        spec_state_indices,
        source_flat,
        conv_weights,
        bias_arg,
        out,
        source_stage,
        int(x.stride(0)),
        int(conv_state.stride(0)),
        int(conv_state.stride(1)),
        int(conv_state.stride(2)),
        int(spec_state_indices.stride(0)),
        int(spec_state_indices.stride(1)),
        int(conv_weights.stride(0)),
        int(conv_weights.stride(1)),
        B=batch,
        N=rows,
        C=channels,
        WIDTH=width,
        STATE_LEN=state_len,
        SOURCE_ROWS=source_rows_per_batch,
        HAS_BIAS=bias is not None,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        BLOCK_C=BLOCK_C,
        num_warps=8,
    )
    contract["conv_programs_per_request"] = triton.cdiv(channels, BLOCK_C)
    return contract
