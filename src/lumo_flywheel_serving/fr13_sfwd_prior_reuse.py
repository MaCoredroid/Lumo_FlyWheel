from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

import torch
import triton

from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
    _FR13_FIXED32_MODE,
    _FR13_FIXED32_MODES,
    _fr13_fixed32_batch_gdn_byte_diff,
    _fr13_fixed32_batch_gdn_real_event_marker,
)
from lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless import (
    CHANNELS,
    CONV_WIDTH,
    FIXED32_CHANNEL_SERIAL_FRONTIER5_LIVE_X_SUM,
    FIXED32_CHANNEL_SERIAL_FRONTIER5_ORDER,
    FIXED32_CHANNEL_SERIAL_FRONTIER5_PEAK_LIVE_X,
    FIXED32_CHANNEL_SERIAL_LOADS_PER_CHANNEL,
    FIXED32_PARENT,
    FIXED32_ROWS,
    SOURCE_ROWS,
    X_ROW_STRIDE,
    _fr13_fixed32_sfwd_channel_serial_kernel,
    fixed32_specialized_layout_contract,
)


CANDIDATE = (
    "fixed32_sfwd_channel_serial_r32_b1c128w2_bxc256w4_"
    "u32x2_frontier5_loadonce_v3"
)
ROWS_PER_PROGRAM = 32
BLOCK_C = 128
NUM_WARPS = 2
MULTIBATCH_BLOCK_C = 256
MULTIBATCH_NUM_WARPS = 4
CONV_STATE_LEN = 34
ENABLED_PATH = "/logs/fr13_fixed32_sfwd_prior_reuse_byte_ab.enabled"
REAL_EVENT_PATH = "/logs/fr13_fixed32_sfwd_state_fusion.real_event.arm"
PASS_PATH = "/logs/fr13_fixed32_sfwd_prior_reuse.live_pass.json"
RECORDS_PATH = "/logs/fr13_fixed32_sfwd_prior_reuse.byte_ab.jsonl"
SOURCE_MANIFEST_SCHEMA = "fr13.fixed32.sfwd_prior_reuse.source_manifest.v1"
SOURCE_RELATIVE_PATH = "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse.py"
KERNEL_SOURCE_RELATIVE_PATH = (
    "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py"
)
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
    """Validate the closed rowgroup32 adaptive prior-reuse geometry."""
    batch = int(batch_size)
    rows = int(tree_rows)
    width = int(conv_width)
    state_len = int(conv_state_len)
    if batch not in (1, 2, 3, 4):
        raise ValueError(f"FR13 SFWD prior-reuse requires B1-B4, got B={batch}")
    if rows != FIXED32_ROWS:
        raise ValueError(
            "FR13 SFWD prior-reuse requires exactly 32 physical rows per "
            f"request, got {rows}"
        )
    if width != CONV_WIDTH or state_len != CONV_STATE_LEN:
        raise ValueError(
            "FR13 SFWD prior-reuse requires width/state geometry "
            f"({CONV_WIDTH}, {CONV_STATE_LEN}), got ({width}, {state_len})"
        )
    source_rows = width - 1 + rows + 1
    block_c = BLOCK_C if batch == 1 else MULTIBATCH_BLOCK_C
    num_warps = NUM_WARPS if batch == 1 else MULTIBATCH_NUM_WARPS
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
        "conv_block_c": block_c,
        "conv_num_warps": num_warps,
        "conv_node_order": FIXED32_CHANNEL_SERIAL_FRONTIER5_ORDER,
        "conv_peak_live_x": FIXED32_CHANNEL_SERIAL_FRONTIER5_PEAK_LIVE_X,
        "conv_live_x_sum": FIXED32_CHANNEL_SERIAL_FRONTIER5_LIVE_X_SUM,
        "x_global_loads_per_channel": (
            FIXED32_CHANNEL_SERIAL_LOADS_PER_CHANNEL
        ),
        "x_reload_count": 0,
        "topology_host_validation": "exact_parent_each_launch",
        "source_descriptor_device_validation": False,
        "source_descriptor_launcher_argument": False,
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
    source_path = Path(__file__)
    kernel_source_path = source_path.with_name(
        "fr13_sfwd_prior_reuse_descriptorless.py"
    )
    expected_source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    expected_kernel_source_sha = hashlib.sha256(
        kernel_source_path.read_bytes()
    ).hexdigest()
    source_entry = files.get(SOURCE_RELATIVE_PATH) if isinstance(files, dict) else None
    kernel_source_entry = (
        files.get(KERNEL_SOURCE_RELATIVE_PATH) if isinstance(files, dict) else None
    )
    if (
        payload.get("schema") != SOURCE_MANIFEST_SCHEMA
        or payload.get("candidate") != CANDIDATE
        or payload.get("source_commit") != source_commit
        or not isinstance(source_entry, dict)
        or source_entry.get("sha256") != expected_source_sha
        or not isinstance(kernel_source_entry, dict)
        or kernel_source_entry.get("sha256") != expected_kernel_source_sha
    ):
        raise RuntimeError(
            "FR13 SFWD prior-reuse source manifest is not bound to runtime source"
        )
    return {
        "source_commit": source_commit,
        "source_manifest_sha256": observed_sha,
        "candidate_source_sha256": expected_source_sha,
        "candidate_kernel_source_sha256": expected_kernel_source_sha,
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
        "conv_num_warps": NUM_WARPS,
        "conv_node_order": list(FIXED32_CHANNEL_SERIAL_FRONTIER5_ORDER),
        "conv_peak_live_x": FIXED32_CHANNEL_SERIAL_FRONTIER5_PEAK_LIVE_X,
        "conv_live_x_sum": FIXED32_CHANNEL_SERIAL_FRONTIER5_LIVE_X_SUM,
        "x_global_loads_per_channel": (
            FIXED32_CHANNEL_SERIAL_LOADS_PER_CHANNEL
        ),
        "x_reload_count": 0,
        "topology_host_validation": "exact_parent_each_launch",
        "source_descriptor_device_validation": False,
        "source_descriptor_launcher_argument": False,
        "x_shape": [FIXED32_ROWS, CHANNELS],
        "x_stride": [X_ROW_STRIDE, 1],
        "out_stride": [CHANNELS, 1],
        "source_stage_shape": [SOURCE_ROWS, CHANNELS],
        "source_stage_stride": [CHANNELS, 1],
        "conv_weights_stride": [CONV_WIDTH, 1],
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
        "conv_num_warps": NUM_WARPS,
        "conv_node_order": list(FIXED32_CHANNEL_SERIAL_FRONTIER5_ORDER),
        "conv_peak_live_x": FIXED32_CHANNEL_SERIAL_FRONTIER5_PEAK_LIVE_X,
        "conv_live_x_sum": FIXED32_CHANNEL_SERIAL_FRONTIER5_LIVE_X_SUM,
        "x_global_loads_per_channel": (
            FIXED32_CHANNEL_SERIAL_LOADS_PER_CHANNEL
        ),
        "x_reload_count": 0,
        "topology_host_validation": "exact_parent_each_launch",
        "source_descriptor_device_validation": False,
        "source_descriptor_launcher_argument": False,
        "x_shape": [FIXED32_ROWS, CHANNELS],
        "x_stride": [X_ROW_STRIDE, 1],
        "out_stride": [CHANNELS, 1],
        "source_stage_shape": [SOURCE_ROWS, CHANNELS],
        "source_stage_stride": [CHANNELS, 1],
        "conv_weights_stride": [CONV_WIDTH, 1],
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


def _validate_fixed32_tree_parent(tree_parent: object) -> tuple[int, ...]:
    if not isinstance(tree_parent, (list, tuple)) or any(
        type(value) is not int for value in tree_parent
    ):
        raise ValueError(
            "FR13 SFWD prior-reuse tree parent must be a host int list/tuple"
        )
    actual = tuple(tree_parent)
    if actual != FIXED32_PARENT:
        raise RuntimeError("FR13 SFWD prior-reuse host parent vector drifted")
    return actual


def launch_fixed32_sfwd_prior_reuse(
    *,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    tree_parent: object,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    source_stage: torch.Tensor,
    batch_size: int,
    tree_rows: int,
) -> dict[str, object]:
    """Launch the default-off rowgroup32 adaptive prior-reuse candidate."""
    if _FR13_FIXED32_MODE not in _FR13_FIXED32_MODES:
        raise RuntimeError("FR13 SFWD prior-reuse requires an exact fixed32 mode")
    _validate_fixed32_tree_parent(tree_parent)
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
    if len(
        {
            int(tensor.untyped_storage().data_ptr())
            for tensor in (x, out, source_stage)
        }
    ) != 3:
        geometry_failures.append("x_out_source_storage_alias")
    if x.ndim != 2:
        geometry_failures.append("x_ndim")
    if tuple(int(value) for value in x.shape) != (required_rows, channels):
        geometry_failures.append("x_shape")
    if spec_state_indices.ndim != 2:
        geometry_failures.append("spec_state_indices_ndim")
    if channels != CHANNELS:
        geometry_failures.append("channels")
    if spec_state_indices.ndim < 1 or int(spec_state_indices.shape[0]) < batch:
        geometry_failures.append("spec_state_indices_batch")
    if spec_state_indices.ndim < 2 or int(spec_state_indices.shape[1]) != rows:
        geometry_failures.append("spec_state_indices_width")
    if spec_state_indices.dtype != torch.int32:
        geometry_failures.append("spec_state_indices_dtype")
    if not spec_state_indices.is_contiguous():
        geometry_failures.append("spec_state_indices_contiguous")
    if int(conv_weights.data_ptr()) % 4 != 0:
        geometry_failures.append("conv_weights_u32_alignment")
    if conv_state.ndim == 3 and int(conv_state.stride(1)) != 1:
        geometry_failures.append("conv_state_channel_stride")
    if conv_state.ndim == 3 and int(conv_state.stride(2)) != channels:
        geometry_failures.append("conv_state_state_stride")
    if (
        conv_state.ndim == 3
        and int(conv_state.stride(0)) < channels * state_len
    ):
        geometry_failures.append("conv_state_row_stride")
    if source_stage.ndim != 2:
        geometry_failures.append("source_stage_ndim")
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
    layout_contract = fixed32_specialized_layout_contract(
        batch,
        x_shape=tuple(x.shape),
        x_stride=tuple(x.stride()),
        out_shape=tuple(out.shape),
        out_stride=tuple(out.stride()),
        source_stage_shape=tuple(source_stage.shape),
        source_stage_stride=tuple(source_stage.stride()),
        conv_weights_shape=tuple(conv_weights.shape),
        conv_weights_stride=tuple(conv_weights.stride()),
    )

    block_c = int(contract["conv_block_c"])
    num_warps = int(contract["conv_num_warps"])
    grid = (batch, triton.cdiv(channels, block_c))
    bias_arg = bias if bias is not None else x
    _fr13_fixed32_sfwd_channel_serial_kernel[grid](
        x,
        conv_state,
        spec_state_indices,
        conv_weights,
        bias_arg,
        out,
        source_stage,
        CONV_STRIDE_ROW=int(conv_state.stride(0)),
        B=batch,
        N=rows,
        C=channels,
        WIDTH=width,
        STATE_LEN=state_len,
        SOURCE_ROWS=source_rows_per_batch,
        HAS_BIAS=bias is not None,
        X_STRIDE_ROW=X_ROW_STRIDE,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )
    contract["conv_programs_per_request"] = triton.cdiv(channels, block_c)
    contract["layouts"] = layout_contract["layouts"]
    contract["maximum_offsets"] = layout_contract["maximum_offsets"]
    return contract
