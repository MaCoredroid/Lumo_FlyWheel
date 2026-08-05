"""Default-off fixed32 tree-conv/post-conv-prep fusion candidate.

The eager launcher keeps a direct SSI value check.  CUDA graph capture is only
accepted through an opaque binding created from the persistent physical32
pregather and sticky-committer preseed leases.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path

import torch
import triton

from lumo_flywheel_serving.fr13_sfwd_conv_postprep_fusion_kernel import (
    _fr13_fixed32_sfwd_conv_postprep_fusion_kernel,
    _fr13_fixed32_sfwd_conv_postprep_nodegroup8_direct_kernel,
)
from lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless import (
    FIXED32_CHANNEL_SERIAL_ACTIVATION_WINDOW,
    FIXED32_CHANNEL_SERIAL_DEFERRED_STAGE_PEAK_LIVE_X,
    FIXED32_CHANNEL_SERIAL_FRONTIER5_ORDER,
    FIXED32_CHANNEL_SERIAL_FRONTIER5_PEAK_LIVE_X,
    FIXED32_CHANNEL_SERIAL_PEAK_LIVE_ACC,
    FIXED32_PARENT,
)


CANDIDATE = "fixed32_sfwd_conv_postprep_frontier5_direct_v1"
EMBEDDED_GATE_CANDIDATE = "fixed32_sfwd_conv_postprep_embedded_gate_cta_v1"
DIRECT_NODEGROUP8_CANDIDATE = (
    "fixed32_sfwd_conv_postprep_nodegroup8_direct_v1"
)
DIRECT_NODEGROUP8_EMBEDDED_GATE_CANDIDATE = (
    "fixed32_sfwd_conv_postprep_nodegroup8_direct_embedded_gate_v1"
)
CAPTURE_CACHE_SCHEMA = "fr13.fixed32.sfwd_conv_postprep.capture_cache.v1"
FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))
QUALIFICATION_PROFILE = "k64_root"
DRAFT_VOCAB_K = 65536
DRAFT_VOCAB_ROOT = 1
LAYERS = 48
ROWS = 32
CHANNELS = 10240
CONV_WIDTH = 4
CONV_STATE_LEN = 34
SOURCE_ROWS = 36
X_ROW_STRIDE = 16384
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_K_DIM = 128
HEAD_V_DIM = 128
Q_DIM = NUM_K_HEADS * HEAD_K_DIM
V_DIM = NUM_V_HEADS * HEAD_V_DIM
GATE_BLOCK = 64
SOFTPLUS_THRESHOLD = 20.0
BF16_BYTES = 2
FP32_BYTES = 4
BYTE_GATE_ENABLED_PATH = (
    "/logs/fr13_fixed32_sfwd_conv_postprep_byte_ab.enabled"
)
BYTE_GATE_REAL_EVENT_PATH = (
    "/logs/fr13_fixed32_sfwd_state_fusion.real_event.arm"
)
BYTE_GATE_RECORDS_PATH = (
    "/logs/fr13_fixed32_sfwd_conv_postprep.byte_ab.jsonl"
)
BYTE_GATE_PASS_PATH = (
    "/logs/fr13_fixed32_sfwd_conv_postprep.live_pass.json"
)
SOURCE_MANIFEST_SCHEMA = "fr13.fixed32.sfwd_conv_postprep.source_manifest.v1"
SOURCE_RELATIVE_PATH = (
    "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py"
)
KERNEL_SOURCE_RELATIVE_PATH = (
    "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion_kernel.py"
)
SOURCE_FILES = (
    "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
    "scripts/fr13_b1_composed_stack_gate.py",
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
    "scripts/fr13_cutlass_streamk_pass.py",
    "scripts/fr13_cutlass_wave_binary.py",
    "scripts/fr13_generate_sfwd_conv_postprep_fusion_kernel.py",
    "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "scripts/fr13_patch_cutlass_fixed32_wave.py",
    "scripts/fr13_run_b1_cutlass_streamk_live_gate.sh",
    "scripts/fr13_run_b1_kernel_live_gate.sh",
    "scripts/fr13_run_b1_sfwd_conv_postprep_gate.sh",
    "scripts/fr13_run_b1_target_sfwd_conv_postprep_live_gate.sh",
    "scripts/fr13_run_b4_sfwd_embedded_gate_live_gate.sh",
    "scripts/fr13_sfwd_conv_postprep_gate.py",
    "scripts/run_swe_bench_q36_a.py",
    SOURCE_RELATIVE_PATH,
    KERNEL_SOURCE_RELATIVE_PATH,
    "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py",
    "src/lumo_flywheel_serving/inference_proxy.py",
)
BYTE_SURFACES = (
    "query_spec",
    "key_spec",
    "value_spec",
    "value_tree",
    "g",
    "beta",
    "commit_source_stage",
)
TASK_MARKER = "swe_verified:astropy__astropy-12907"
EXACT4_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
EXACT4_TASK_MARKERS = tuple(
    f"swe_verified:{task_id}" for task_id in EXACT4_TASK_IDS
)
QROW16_FA2_SHA256 = (
    "1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86"
)
QROW16_LIVE_PASS_SHA256 = (
    "36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77"
)
_BYTE_GATE_STATE: dict[str, object] = {
    "task_marker": None,
    "batch_size": None,
    "embedded_gate_cta": None,
    "direct_nodegroup8": None,
    "source_binding": None,
    "passed": {},
    "attempts": {},
    "failed": False,
}


def _read_single_link_regular(path: Path, *, label: str, limit: int) -> bytes:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"FR13 SFWD conv/post-prep {label} is missing") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size <= 0
        or info.st_size > limit
    ):
        raise RuntimeError(
            f"FR13 SFWD conv/post-prep {label} must be one bounded regular file"
        )
    return path.read_bytes()


def _real_event_markers(path: Path, *, batch_size: int) -> str | tuple[str, ...]:
    raw = _read_single_link_regular(path, label="real-event marker", limit=1024)
    if stat.S_IMODE(path.lstat().st_mode) != 0o444:
        raise RuntimeError(
            "FR13 SFWD conv/post-prep real-event marker mode must be 0444"
        )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError(
            "FR13 SFWD conv/post-prep real-event marker must be ASCII"
        ) from error
    if int(batch_size) == 1:
        expected = TASK_MARKER + "\n"
        result: str | tuple[str, ...] = TASK_MARKER
    elif int(batch_size) == 4:
        expected = "\n".join(EXACT4_TASK_MARKERS) + "\n"
        result = EXACT4_TASK_MARKERS
    else:
        raise RuntimeError(
            "FR13 SFWD conv/post-prep marker requires exact B1 or B4"
        )
    if text != expected:
        raise RuntimeError(
            "FR13 SFWD conv/post-prep requires the canonical authenticated "
            "task marker set"
        )
    return result


def fixed32_sfwd_conv_postprep_gate_control(
    *,
    fixed32_mode: str | None,
    environ: Mapping[str, str] | None = None,
    enabled_path: str | None = None,
    event_path: str | None = None,
) -> tuple[bool, str | tuple[str, ...] | None]:
    """Resolve the default-off authenticated B1/B4 shadow gate."""
    env = os.environ if environ is None else environ
    raw = str(env.get("FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB", "0"))
    if raw not in ("0", "1"):
        raise RuntimeError(
            "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB must be exactly 0 or 1"
        )
    direct_raw = str(env.get("FR13_FIXED32_SFWD_NODEGROUP8_DIRECT", "0"))
    if direct_raw not in ("0", "1"):
        raise RuntimeError(
            "FR13_FIXED32_SFWD_NODEGROUP8_DIRECT must be exactly 0 or 1"
        )
    direct_nodegroup8 = direct_raw == "1"
    if direct_nodegroup8 and raw != "1":
        raise RuntimeError(
            "FR13 direct nodegroup8 requires the conv/post-prep byte gate"
        )
    enabled = Path(
        enabled_path
        or env.get(
            "FR13_FIXED32_SFWD_CONV_POSTPREP_ENABLED_PATH",
            BYTE_GATE_ENABLED_PATH,
        )
    )
    event = Path(
        event_path
        or env.get(
            "FR13_FIXED32_SFWD_CONV_POSTPREP_REAL_EVENT_PATH",
            BYTE_GATE_REAL_EVENT_PATH,
        )
    )
    if raw == "0":
        return False, None
    embed_raw = str(env.get("FR13_FIXED32_SFWD_EMBED_GATE_CTA", "0"))
    if embed_raw not in ("0", "1"):
        raise RuntimeError(
            "FR13_FIXED32_SFWD_EMBED_GATE_CTA must be exactly 0 or 1"
        )
    batch_text = str(env.get("MAX_NUM_SEQS", "1"))
    if batch_text not in (("1", "4") if embed_raw == "1" else ("1",)):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep gate batch is not admitted by its schedule"
        )
    if direct_nodegroup8 and (
        fixed32_mode != "hydra27_fixed32"
        or batch_text != "1"
        or str(env.get("FR13_DRAFT_VOCAB_ROOT", "")) != "1"
        or str(env.get("FR13_DRAFT_VOCAB_K", "")) != "65536"
    ):
        raise RuntimeError(
            "FR13 direct nodegroup8 requires Hydra27 physical32 K64/root1 B1"
        )
    enabled_raw = _read_single_link_regular(
        enabled, label="byte-gate arm", limit=16
    )
    if (
        enabled_raw != b"1\n"
        or stat.S_IMODE(enabled.lstat().st_mode) != 0o400
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep byte-gate arm is not exact"
        )
    if not event.exists():
        return True, None
    if fixed32_mode not in FIXED32_MODES:
        raise RuntimeError(
            "FR13 SFWD conv/post-prep byte gate requires an exact fixed32 mode"
        )
    return True, _real_event_markers(event, batch_size=int(batch_text))


def _source_binding(
    *,
    manifest_path: str,
    expected_manifest_sha256: str,
    expected_source_commit: str,
    direct_nodegroup8: bool = False,
) -> dict[str, str]:
    if (
        not manifest_path
        or re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None
        or re.fullmatch(r"[0-9a-f]{40}", expected_source_commit) is None
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep requires complete source credentials"
        )
    raw = _read_single_link_regular(
        Path(manifest_path), label="source manifest", limit=1048576
    )
    if stat.S_IMODE(Path(manifest_path).lstat().st_mode) != 0o400:
        raise RuntimeError(
            "FR13 SFWD conv/post-prep source manifest mode must be 0400"
        )
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "FR13 SFWD conv/post-prep source-manifest SHA-256 drifted"
        )
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "FR13 SFWD conv/post-prep source manifest must be ASCII JSON"
        ) from error
    files = payload.get("files") if isinstance(payload, dict) else None
    source_path = Path(__file__).resolve()
    source_root = source_path.parents[2]
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    kernel_sha256 = hashlib.sha256(
        (source_root / KERNEL_SOURCE_RELATIVE_PATH).read_bytes()
    ).hexdigest()
    source_entry = files.get(SOURCE_RELATIVE_PATH) if isinstance(files, dict) else None
    kernel_entry = (
        files.get(KERNEL_SOURCE_RELATIVE_PATH) if isinstance(files, dict) else None
    )
    source_drift = {}
    if isinstance(files, dict):
        if tuple(sorted(files)) != tuple(sorted(SOURCE_FILES)):
            source_drift["files"] = tuple(sorted(files))
        for relative in SOURCE_FILES:
            path = source_root / relative
            try:
                source_raw = path.read_bytes()
            except OSError as error:
                raise RuntimeError(
                    "FR13 SFWD conv/post-prep source closure is unreadable: "
                    f"{relative}"
                ) from error
            actual = {
                "bytes": len(source_raw),
                "sha256": hashlib.sha256(source_raw).hexdigest(),
            }
            if files.get(relative) != actual:
                source_drift[relative] = (files.get(relative), actual)
    if (
        payload.get("schema") != SOURCE_MANIFEST_SCHEMA
        or payload.get("candidate")
        != (DIRECT_NODEGROUP8_CANDIDATE if direct_nodegroup8 else CANDIDATE)
        or payload.get("source_commit") != expected_source_commit
        or not isinstance(source_entry, dict)
        or source_entry.get("sha256") != source_sha256
        or not isinstance(kernel_entry, dict)
        or kernel_entry.get("sha256") != kernel_sha256
        or source_drift
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep source manifest is not runtime-bound"
        )
    return {
        "source_commit": expected_source_commit,
        "source_manifest_sha256": observed_sha256,
        "candidate_source_sha256": source_sha256,
        "candidate_kernel_source_sha256": kernel_sha256,
    }


def _byte_diff(
    name: str, reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, object]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {
            "name": name,
            "byte_equal": False,
            "shape_or_dtype_mismatch": True,
            "reference_shape": tuple(int(value) for value in reference.shape),
            "candidate_shape": tuple(int(value) for value in candidate.shape),
            "reference_dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
            "differing_bytes": None,
            "first_nonzero_byte": None,
        }
    reference_bytes = reference.contiguous().reshape(-1).view(torch.uint8)
    candidate_bytes = candidate.contiguous().reshape(-1).view(torch.uint8)
    difference = reference_bytes != candidate_bytes
    differing_bytes = int(difference.sum().item())
    first_nonzero = (
        None
        if differing_bytes == 0
        else int(torch.nonzero(difference, as_tuple=False)[0, 0].item())
    )
    return {
        "name": name,
        "byte_equal": differing_bytes == 0,
        "shape_or_dtype_mismatch": False,
        "shape": tuple(int(value) for value in reference.shape),
        "dtype": str(reference.dtype),
        "bytes": int(reference_bytes.numel()),
        "differing_bytes": differing_bytes,
        "first_nonzero_byte": first_nonzero,
        "reference_byte": (
            None if first_nonzero is None else int(reference_bytes[first_nonzero].item())
        ),
        "candidate_byte": (
            None if first_nonzero is None else int(candidate_bytes[first_nonzero].item())
        ),
    }


def _emit_byte_record(
    record: dict[str, object], *, embedded_gate_cta: bool
) -> None:
    path = Path(
        os.environ.get(
            "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB_PATH",
            BYTE_GATE_RECORDS_PATH,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload["schema"] = (
        "fr13.fixed32.sfwd_conv_postprep.embedded_gate.byte_ab.v1"
        if embedded_gate_cta
        else "fr13.fixed32.sfwd_conv_postprep.byte_ab.v1"
    )
    with path.open("a", encoding="ascii") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _invalidate_live_pass() -> None:
    path = Path(
        os.environ.get(
            "FR13_FIXED32_SFWD_CONV_POSTPREP_PASS_PATH", BYTE_GATE_PASS_PATH
        )
    )
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _emit_live_pass(
    *,
    fixed32_mode: str,
    task_markers: tuple[str, ...],
    batch_size: int,
    embedded_gate_cta: bool,
    direct_nodegroup8: bool,
    layers: dict[int, str],
    source_binding: dict[str, str],
) -> None:
    if len(layers) != LAYERS or len(set(layers.values())) != LAYERS:
        return
    draft_vocab_blocks = Path(os.environ.get("FR13_DRAFT_VOCAB_BLOCKS", ""))
    try:
        blocks_sha256 = hashlib.sha256(draft_vocab_blocks.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError(
            "FR13 SFWD conv/post-prep K64 block map is unreadable"
        ) from error
    if (
        os.environ.get("FR13_DRAFT_VOCAB_ROOT") != "1"
        or os.environ.get("FR13_DRAFT_VOCAB_K") != "65536"
        or blocks_sha256
        != "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep PASS requires exact K64/root1"
        )
    batch = int(batch_size)
    if direct_nodegroup8 and (
        fixed32_mode != "hydra27_fixed32"
        or batch != 1
        or task_markers != (TASK_MARKER,)
    ):
        raise RuntimeError(
            "direct nodegroup8 PASS requires authenticated Hydra27 B1"
        )
    if embedded_gate_cta:
        if fixed32_mode != "hydra27_fixed32" or (
            (batch == 1 and task_markers != (TASK_MARKER,))
            or (batch == 4 and task_markers != EXACT4_TASK_MARKERS)
            or batch not in (1, 4)
        ):
            raise RuntimeError(
                "embedded gate CTA PASS requires authenticated Hydra27 B1 or B4"
            )
    elif batch != 1 or task_markers != (TASK_MARKER,):
        raise RuntimeError(
            "standalone gate CTA PASS requires the canonical B1 task"
        )
    qrow_sha256 = os.environ.get("FR13_FA2_QROW16_SO_SHA256", "")
    qrow_pass_sha256 = os.environ.get("FR13_FA2_QROW16_LIVE_PASS_SHA256", "")
    qrow16 = (
        os.environ.get("FR13_FA2_QROW16_PRODUCTION") == "1"
        and qrow_sha256 == QROW16_FA2_SHA256
        and qrow_pass_sha256 == QROW16_LIVE_PASS_SHA256
    )
    if batch == 1 and not qrow16:
        raise RuntimeError(
            "FR13 SFWD conv/post-prep B1 PASS requires pinned Qrow16"
        )
    if batch == 4 and (
        os.environ.get("FR13_FA2_QROW16_PRODUCTION", "0") != "0"
        or os.environ.get("FR13_FA2_QROW32_B1_LIVE_AB_ARM", "")
        or os.environ.get("FR13_FA2_QROW32_B1_PRODUCTION_ARM", "")
        or os.environ.get("FR13_FIXED32_CUTLASS_WAVE", "stock") != "stock"
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep B4 PASS requires the stock attention arm"
        )
    path = Path(
        os.environ.get(
            "FR13_FIXED32_SFWD_CONV_POSTPREP_PASS_PATH", BYTE_GATE_PASS_PATH
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema": (
            "fr13.fixed32.sfwd_conv_postprep.embedded_gate."
            f"{'b1' if batch == 1 else 'exact4_b4'}_live_pass.v1"
            if embedded_gate_cta
            else "fr13.fixed32.sfwd_conv_postprep.live_pass.v1"
        ),
        "status": "byte_pass_source_only",
        "candidate": (
            DIRECT_NODEGROUP8_EMBEDDED_GATE_CANDIDATE
            if direct_nodegroup8 and embedded_gate_cta
            else DIRECT_NODEGROUP8_CANDIDATE
            if direct_nodegroup8
            else EMBEDDED_GATE_CANDIDATE
            if embedded_gate_cta
            else CANDIDATE
        ),
        **source_binding,
        "fixed32_mode": fixed32_mode,
        "batch_size": batch,
        "task_count": len(task_markers),
        "layer_count": LAYERS,
        "layers": [
            {
                "layer_key": f"0x{key:x}",
                "layer_prefix_sha256": layers[key],
            }
            for key in sorted(layers)
        ],
        "physical_rows_per_request": ROWS,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks_sha256": blocks_sha256,
        "compared_byte_surfaces": list(BYTE_SURFACES),
        "real_task_authenticated": True,
        "reference_always_served": True,
        "candidate_returned": False,
        "reference_decision": "serve_incumbent",
        "candidate_decision": "shadow_only",
        "decision_exact": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
        "comparisons": LAYERS * len(BYTE_SURFACES),
        "mismatches": 0,
        "differing_bytes": 0,
        "errors": 0,
    }
    if embedded_gate_cta:
        payload.update(
            {
                "task_ids": list(EXACT4_TASK_IDS[:batch]),
                "task_markers": list(task_markers),
                "embedded_gate_cta": True,
                "gate_scheduling": "append_to_first_channel_programs",
                "channel_programs_per_request": (
                    160 if direct_nodegroup8 else 40
                ),
                "standalone_gate_programs_per_request": 0,
                "programs_per_request": 160 if direct_nodegroup8 else 40,
                "qrow16_production": qrow16,
                "stock_attention": batch == 4,
            }
        )
        if batch == 1:
            payload.update(
                {
                    "qrow16_fa2_sha256": qrow_sha256,
                    "qrow16_live_pass_sha256": qrow_pass_sha256,
                }
            )
    else:
        payload.update(
            {
                "task_marker": TASK_MARKER,
                "qrow16_production": True,
                "qrow16_fa2_sha256": qrow_sha256,
                "qrow16_live_pass_sha256": qrow_pass_sha256,
            }
        )
        if direct_nodegroup8:
            payload.update(
                {
                    "channel_programs_per_request": 160,
                    "standalone_gate_programs_per_request": 4,
                    "programs_per_request": 164,
                }
            )
    if direct_nodegroup8:
        payload["direct_nodegroup8"] = True
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="ascii") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def fixed32_sfwd_conv_postprep_byte_gate(
    *,
    fixed32_mode: str,
    task_marker: str | tuple[str, ...],
    layer_prefix: str,
    layer_key: int,
    batch_size: int,
    reference_query: torch.Tensor,
    candidate_query: torch.Tensor,
    reference_key: torch.Tensor,
    candidate_key: torch.Tensor,
    reference_value_spec: torch.Tensor,
    candidate_value_spec: torch.Tensor,
    reference_value_tree: torch.Tensor,
    candidate_value_tree: torch.Tensor,
    reference_g: torch.Tensor,
    candidate_g: torch.Tensor,
    reference_beta: torch.Tensor,
    candidate_beta: torch.Tensor,
    reference_source_stage: torch.Tensor,
    candidate_source_stage: torch.Tensor,
    source_manifest_path: str,
    expected_source_manifest_sha256: str,
    expected_source_commit: str,
    embedded_gate_cta: bool = False,
    direct_nodegroup8: bool = False,
) -> dict[str, object]:
    """Compare the complete consumer boundary while serving only the reference."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FR13 SFWD conv/post-prep byte gate is eager-only")
    if type(embedded_gate_cta) is not bool:
        raise TypeError("embedded_gate_cta must be bool")
    if type(direct_nodegroup8) is not bool:
        raise TypeError("direct_nodegroup8 must be bool")
    batch = int(batch_size)
    if fixed32_mode not in FIXED32_MODES or batch not in (1, 4):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep byte gate requires fixed32 B1 or B4"
        )
    markers = (
        (task_marker,)
        if isinstance(task_marker, str)
        else tuple(task_marker)
        if isinstance(task_marker, tuple)
        else ()
    )
    expected_markers = (
        (TASK_MARKER,) if batch == 1 else EXACT4_TASK_MARKERS
    )
    standalone_marker_invalid = (
        not embedded_gate_cta
        and (batch != 1 or markers != (TASK_MARKER,))
    )
    embedded_marker_invalid = (
        embedded_gate_cta
        and (
            fixed32_mode != "hydra27_fixed32"
            or markers != expected_markers
        )
    )
    direct_marker_invalid = direct_nodegroup8 and (
        fixed32_mode != "hydra27_fixed32"
        or batch != 1
        or markers != (TASK_MARKER,)
    )
    if standalone_marker_invalid or embedded_marker_invalid or direct_marker_invalid:
        raise RuntimeError(
            "FR13 SFWD conv/post-prep byte gate requires its canonical real "
            "task marker set"
        )
    if not isinstance(layer_prefix, str) or not layer_prefix:
        raise RuntimeError("FR13 SFWD conv/post-prep requires a layer prefix")
    state = _BYTE_GATE_STATE
    if state["task_marker"] is None:
        state["task_marker"] = markers
        state["batch_size"] = batch
        state["embedded_gate_cta"] = embedded_gate_cta
        state["direct_nodegroup8"] = direct_nodegroup8
        _invalidate_live_pass()
    elif (
        tuple(state["task_marker"]) != markers
        or int(state.get("batch_size", -1)) != batch
        or bool(state.get("embedded_gate_cta")) != embedded_gate_cta
        or bool(state.get("direct_nodegroup8")) != direct_nodegroup8
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep cannot combine marker, batch, or schedule"
        )
    if state["source_binding"] is None:
        state["source_binding"] = _source_binding(
            manifest_path=source_manifest_path,
            expected_manifest_sha256=expected_source_manifest_sha256,
            expected_source_commit=expected_source_commit,
            direct_nodegroup8=direct_nodegroup8,
        )
    source_binding = dict(state["source_binding"])
    key = int(layer_key)
    prefix_sha256 = hashlib.sha256(layer_prefix.encode("utf-8")).hexdigest()
    passed_layers = state["passed"]
    attempts = state["attempts"]
    if key not in passed_layers and len(passed_layers) >= LAYERS:
        _invalidate_live_pass()
        raise RuntimeError(
            "FR13 SFWD conv/post-prep observed more than 48 layer identities"
        )
    previous_prefix = passed_layers.get(key)
    if previous_prefix is not None and previous_prefix != prefix_sha256:
        raise RuntimeError(
            "FR13 SFWD conv/post-prep layer key was reused by another prefix"
        )
    attempt = int(attempts.get(key, 0)) + 1
    attempts[key] = attempt
    comparisons = [
        _byte_diff("query_spec", reference_query, candidate_query),
        _byte_diff("key_spec", reference_key, candidate_key),
        _byte_diff("value_spec", reference_value_spec, candidate_value_spec),
        _byte_diff("value_tree", reference_value_tree, candidate_value_tree),
        _byte_diff("g", reference_g, candidate_g),
        _byte_diff("beta", reference_beta, candidate_beta),
        _byte_diff(
            "commit_source_stage",
            reference_source_stage,
            candidate_source_stage,
        ),
    ]
    first_nonzero = next(
        (item for item in comparisons if item["byte_equal"] is not True), None
    )
    passed = first_nonzero is None
    differing_bytes = sum(
        int(item["differing_bytes"] or 0) for item in comparisons
    )
    record = {
        "status": "pass" if passed else "mismatch_reference_served",
        "candidate": (
            DIRECT_NODEGROUP8_EMBEDDED_GATE_CANDIDATE
            if direct_nodegroup8 and embedded_gate_cta
            else DIRECT_NODEGROUP8_CANDIDATE
            if direct_nodegroup8
            else EMBEDDED_GATE_CANDIDATE
            if embedded_gate_cta
            else CANDIDATE
        ),
        **source_binding,
        "fixed32_mode": fixed32_mode,
        "batch_size": batch,
        "layer_key": f"0x{key:x}",
        "layer_prefix_sha256": prefix_sha256,
        "attempt": attempt,
        "physical_rows_per_request": ROWS,
        "compared_byte_surfaces": list(BYTE_SURFACES),
        "comparisons": comparisons,
        "mismatches": 0 if passed else 1,
        "differing_bytes": differing_bytes,
        "first_nonzero": first_nonzero,
        "zero_diff": passed,
        "real_task_authenticated": True,
        "reference_always_served": True,
        "candidate_returned": False,
        "reference_decision": "serve_incumbent",
        "candidate_decision": "shadow_only",
        "decision_exact": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }
    if embedded_gate_cta:
        record.update(
            {
                "task_ids": list(EXACT4_TASK_IDS[:batch]),
                "task_markers": list(markers),
                "embedded_gate_cta": True,
                "gate_scheduling": "append_to_first_channel_programs",
                "programs_per_request": 160 if direct_nodegroup8 else 40,
                "qrow16_production": batch == 1,
                "stock_attention": batch == 4,
            }
        )
        if batch == 1:
            record.update(
                {
                    "qrow16_fa2_sha256": QROW16_FA2_SHA256,
                    "qrow16_live_pass_sha256": QROW16_LIVE_PASS_SHA256,
                }
            )
    else:
        record.update(
            {
                "task_marker": TASK_MARKER,
                "qrow16_production": True,
                "qrow16_fa2_sha256": QROW16_FA2_SHA256,
                "qrow16_live_pass_sha256": QROW16_LIVE_PASS_SHA256,
            }
        )
        if direct_nodegroup8:
            record.update(
                {
                    "channel_programs_per_request": 160,
                    "standalone_gate_programs_per_request": 4,
                    "programs_per_request": 164,
                }
            )
    if direct_nodegroup8:
        record["direct_nodegroup8"] = True
    _emit_byte_record(record, embedded_gate_cta=embedded_gate_cta)
    if not passed:
        state["failed"] = True
    if passed and state["failed"] is not True:
        passed_layers[key] = prefix_sha256
        _emit_live_pass(
            fixed32_mode=fixed32_mode,
            task_markers=markers,
            batch_size=batch,
            embedded_gate_cta=embedded_gate_cta,
            direct_nodegroup8=direct_nodegroup8,
            layers=dict(passed_layers),
            source_binding=source_binding,
        )
    else:
        passed_layers.pop(key, None)
        _invalidate_live_pass()
    return record


def _exact_host_parent(tree_parent: object) -> tuple[int, ...]:
    if not isinstance(tree_parent, (list, tuple)) or any(
        type(value) is not int for value in tree_parent
    ):
        raise ValueError(
            "fixed32 conv/post-prep fusion tree_parent must be a host int "
            "list/tuple"
        )
    actual = tuple(tree_parent)
    if actual != FIXED32_PARENT:
        raise RuntimeError(
            "fixed32 conv/post-prep fusion physical32 parent vector drifted"
        )
    return actual


def fixed32_sfwd_conv_postprep_fusion_contract(
    batch_size: int,
    *,
    fixed32_mode: str,
    tree_parent: object,
    qualification_profile: str,
    draft_vocab_k: int,
    draft_vocab_root: int,
    embed_gate_cta: bool = False,
    direct_nodegroup8: bool = False,
    tree_rows: int = ROWS,
    conv_width: int = CONV_WIDTH,
    conv_state_len: int = CONV_STATE_LEN,
) -> dict[str, object]:
    """Close the only per-layer producer/consumer fusion geometry."""
    batch = int(batch_size)
    rows = int(tree_rows)
    width = int(conv_width)
    state_len = int(conv_state_len)
    if type(embed_gate_cta) is not bool:
        raise TypeError("embed_gate_cta must be bool")
    if type(direct_nodegroup8) is not bool:
        raise TypeError("direct_nodegroup8 must be bool")
    if direct_nodegroup8:
        incumbent = fixed32_sfwd_conv_postprep_fusion_contract(
            batch,
            fixed32_mode=fixed32_mode,
            tree_parent=tree_parent,
            qualification_profile=qualification_profile,
            draft_vocab_k=draft_vocab_k,
            draft_vocab_root=draft_vocab_root,
            embed_gate_cta=embed_gate_cta,
            direct_nodegroup8=False,
            tree_rows=rows,
            conv_width=width,
            conv_state_len=state_len,
        )
        channel_programs = 4 * CHANNELS // int(incumbent["block_c"])
        gating_programs = int(incumbent["gating_programs_per_request"])
        return {
            **incumbent,
            "candidate": (
                DIRECT_NODEGROUP8_EMBEDDED_GATE_CANDIDATE
                if embed_gate_cta
                else DIRECT_NODEGROUP8_CANDIDATE
            ),
            "full_graph_qualified": False,
            "channel_programs_per_request": channel_programs,
            "standalone_gating_programs_per_request": (
                0 if embed_gate_cta else gating_programs
            ),
            "embedded_gating_channel_programs_per_request": (
                gating_programs if embed_gate_cta else 0
            ),
            "programs_per_request": channel_programs + (
                0 if embed_gate_cta else gating_programs
            ),
            "conv_node_order": tuple(range(ROWS)),
            "conv_peak_live_x": 10,
            "conv_peak_live_x_with_stage": 10,
            "node_groups_per_request": 4,
            "nodes_per_channel_program": 8,
            "node_group_rows": ((0, 8), (8, 16), (16, 24), (24, 32)),
            "node_group_unique_x_loads": (8, 14, 18, 14),
            "node_group_peak_live_x": (4, 9, 10, 7),
            "x_loads_per_channel_across_groups": 54,
            "source_edge_writer_group": 0,
            "producer_shape": "direct_scalar_unrolled_fixed_nodegroups8",
            "has_gather": False,
            "algorithmic_shared_bytes": 0,
            "has_reduction": False,
            "has_barrier": False,
            "candidate_codegen_registers_per_thread": (
                48 if embed_gate_cta else 46
            ),
            "source_register_ceiling_per_thread": 48,
            "source_register_ceiling_basis": (
                "direct_nodegroup8_sm121a_standalone46_embedded48"
            ),
            "offline_codegen_stack_bytes": 0,
            "offline_codegen_local_bytes": 0,
            "offline_codegen_shared_bytes": 0,
            "codegen_registers_verified": True,
            "timing_claim": False,
        }
    _exact_host_parent(tree_parent)
    if batch not in (1, 2, 3, 4):
        raise ValueError("fixed32 conv/post-prep fusion requires B1-B4")
    if fixed32_mode not in FIXED32_MODES:
        raise RuntimeError("fixed32 conv/post-prep fusion requires an exact mode")
    if embed_gate_cta and (
        fixed32_mode != "hydra27_fixed32" or batch not in (1, 4)
    ):
        raise RuntimeError(
            "embedded gate CTA requires exact Hydra27 physical32 B1 or B4"
        )
    if (
        str(qualification_profile) != QUALIFICATION_PROFILE
        or int(draft_vocab_k) != DRAFT_VOCAB_K
        or int(draft_vocab_root) != DRAFT_VOCAB_ROOT
    ):
        raise RuntimeError(
            "fixed32 conv/post-prep fusion requires physical32 K64/root1"
        )
    if (rows, width, state_len) != (ROWS, CONV_WIDTH, CONV_STATE_LEN):
        raise ValueError(
            "fixed32 conv/post-prep fusion requires rows/width/state "
            f"({ROWS}, {CONV_WIDTH}, {CONV_STATE_LEN})"
        )
    block_c = 256
    num_warps = 4
    channel_programs = CHANNELS // block_c
    gate_rows_per_program = 2 * block_c // GATE_BLOCK
    gating_programs = ROWS // gate_rows_per_program
    return {
        "candidate": EMBEDDED_GATE_CANDIDATE if embed_gate_cta else CANDIDATE,
        "source_only": True,
        "default_off": True,
        "production_eligible": False,
        "full_graph_qualified": True,
        "capture_ssi_guard": (
            "persistent_pregather_selfcheck_plus_in_kernel_bounds_clamp_"
            "and_sticky_xchg"
        ),
        "capture_host_syncs_per_layer": 0,
        "fixed32_mode": fixed32_mode,
        "qualification_profile": QUALIFICATION_PROFILE,
        "draft_vocab_k": DRAFT_VOCAB_K,
        "draft_vocab_root": DRAFT_VOCAB_ROOT,
        "batch_size": batch,
        "layers": LAYERS,
        "physical_rows_per_request": ROWS,
        "logical_rows": batch * ROWS,
        "channels": CHANNELS,
        "conv_width": CONV_WIDTH,
        "conv_state_len": CONV_STATE_LEN,
        "source_rows_per_request": SOURCE_ROWS,
        "num_k_heads": NUM_K_HEADS,
        "num_v_heads": NUM_V_HEADS,
        "head_k_dim": HEAD_K_DIM,
        "head_v_dim": HEAD_V_DIM,
        "query_channels": Q_DIM,
        "value_channels": V_DIM,
        "block_c": block_c,
        "num_warps": num_warps,
        "channel_programs_per_request": channel_programs,
        "gate_rows_per_program": gate_rows_per_program,
        "gating_programs_per_request": gating_programs,
        "standalone_gating_programs_per_request": (
            0 if embed_gate_cta else gating_programs
        ),
        "embedded_gating_channel_programs_per_request": (
            gating_programs if embed_gate_cta else 0
        ),
        "gate_scheduling": (
            "append_to_first_channel_programs"
            if embed_gate_cta
            else "standalone_after_channel_programs"
        ),
        "programs_per_request": channel_programs + (
            0 if embed_gate_cta else gating_programs
        ),
        "launches_per_layer": 1,
        "launches_for_all_layers": LAYERS,
        "cross_layer_fusion": False,
        "layer_recurrence_order": (
            "conv_postprep_then_gdn_then_residual_then_next_layer"
        ),
        "conv_product_dtype": "bfloat16",
        "conv_accumulator_dtype": "float32",
        "conv_add_order": ("bias", "tap0", "tap1", "tap2", "tap3"),
        "post_activation_boundary_dtype": "bfloat16",
        "query_key_l2norm": "deferred_to_existing_gdn_kernel",
        "postprep_dead_normalized_query_key_outputs": "not_materialized",
        "conv_tap_default": False,
        "commit_source_stage": True,
        "distinct_recurrence_output_storages": True,
        "conv_node_order": FIXED32_CHANNEL_SERIAL_FRONTIER5_ORDER,
        "conv_peak_live_x": FIXED32_CHANNEL_SERIAL_FRONTIER5_PEAK_LIVE_X,
        "conv_activation_window": FIXED32_CHANNEL_SERIAL_ACTIVATION_WINDOW,
        "conv_peak_live_acc": FIXED32_CHANNEL_SERIAL_PEAK_LIVE_ACC,
        "conv_peak_live_x_with_stage": (
            FIXED32_CHANNEL_SERIAL_DEFERRED_STAGE_PEAK_LIVE_X
        ),
        "algorithmic_shared_bytes": 0,
        "has_reduction": False,
        "has_barrier": False,
        "incumbent_codegen_registers_per_thread": 48,
        "candidate_codegen_registers_per_thread": 56,
        "source_register_ceiling_per_thread": 64,
        "source_register_ceiling_basis": (
            "frontier5_conv_branch_plus_direct_store_addresses_or_disjoint_"
            "scalar_gating_branch"
        ),
        "offline_codegen_target": "sm_121a",
        "offline_codegen_stack_bytes": 0,
        "offline_codegen_local_bytes": 0,
        "offline_codegen_shared_bytes": 0,
        "offline_codegen_profiles": ("B1", "B4", "B1_tap", "B4_tap"),
        "codegen_registers_verified": True,
        "timing_claim": False,
    }


def fixed32_sfwd_conv_postprep_static_ledger(
    batch_size: int,
    *,
    layers: int = LAYERS,
    store_conv_tap: bool = False,
    embed_gate_cta: bool = False,
) -> dict[str, object]:
    """Return deterministic logical global-byte and launch accounting."""
    batch = int(batch_size)
    layer_count = int(layers)
    if batch not in (1, 2, 3, 4):
        raise ValueError("fixed32 conv/post-prep ledger requires B1-B4")
    if layer_count != LAYERS:
        raise ValueError("fixed32 conv/post-prep ledger requires exactly 48 layers")
    if type(store_conv_tap) is not bool:
        raise TypeError("store_conv_tap must be bool")
    if type(embed_gate_cta) is not bool:
        raise TypeError("embed_gate_cta must be bool")
    rows = batch * ROWS
    conv_surface = rows * CHANNELS * BF16_BYTES
    recurrence_surface = conv_surface
    value_tree_surface = rows * V_DIM * BF16_BYTES
    gate_surfaces = rows * NUM_V_HEADS * FP32_BYTES * 2
    source_stage_surface = batch * SOURCE_ROWS * CHANNELS * BF16_BYTES
    dead_normalized_qk_surface = rows * (2 * Q_DIM) * BF16_BYTES
    mandatory_direct_writes = (
        recurrence_surface + value_tree_surface + gate_surfaces
        + source_stage_surface
    )
    tap_write = conv_surface if store_conv_tap else 0
    removed_conv_write = 0 if store_conv_tap else conv_surface
    removed_conv_reads = conv_surface * 2
    removed_dead_qk_write = dead_normalized_qk_surface
    logical_traffic_removed = (
        removed_conv_write + removed_conv_reads + removed_dead_qk_write
    )
    block_c = 256
    channel_programs = CHANNELS // block_c
    gate_rows_per_program = 2 * block_c // GATE_BLOCK
    packed_gate_programs = ROWS // gate_rows_per_program
    baseline_programs_per_request = channel_programs + ROWS
    packed_programs_per_request = channel_programs + (
        0 if embed_gate_cta else packed_gate_programs
    )
    gate_row_varying_reads = 2 * ROWS * NUM_V_HEADS * BF16_BYTES
    gate_row_varying_writes = 2 * ROWS * NUM_V_HEADS * FP32_BYTES
    gate_invariant_bytes_per_program = NUM_V_HEADS * (
        FP32_BYTES + BF16_BYTES
    )
    baseline_gate_requested_bytes = (
        gate_row_varying_reads
        + gate_row_varying_writes
        + ROWS * gate_invariant_bytes_per_program
    )
    packed_gate_requested_bytes = (
        gate_row_varying_reads
        + gate_row_varying_writes
        + packed_gate_programs * gate_invariant_bytes_per_program
    )
    return {
        "schema": "fr13.fixed32.sfwd_conv_postprep.static_ledger.v1",
        "candidate": EMBEDDED_GATE_CANDIDATE if embed_gate_cta else CANDIDATE,
        "batch_size": batch,
        "layers": layer_count,
        "rows_per_layer": rows,
        "store_conv_tap": store_conv_tap,
        "per_layer_bytes": {
            "incumbent_conv_intermediate_write": conv_surface,
            "incumbent_rearrange_read": conv_surface,
            "incumbent_fused_postprep_read": conv_surface,
            "incumbent_dead_normalized_qk_write": dead_normalized_qk_surface,
            "direct_recurrence_qkv_write": recurrence_surface,
            "direct_value_tree_write": value_tree_surface,
            "direct_g_beta_write": gate_surfaces,
            "unchanged_commit_source_stage_write": source_stage_surface,
            "optional_conv_tap_write": tap_write,
            "mandatory_direct_writes": mandatory_direct_writes,
            "logical_traffic_removed": logical_traffic_removed,
        },
        "all_layer_bytes": {
            "conv_intermediate_write_removed": removed_conv_write * layer_count,
            "conv_intermediate_reads_removed": removed_conv_reads * layer_count,
            "dead_normalized_qk_write_removed": (
                removed_dead_qk_write * layer_count
            ),
            "logical_traffic_removed": logical_traffic_removed * layer_count,
        },
        "gate_packing": {
            "gate_rows_per_program": gate_rows_per_program,
            "baseline_gate_programs_per_request": ROWS,
            "candidate_gate_programs_per_request": packed_gate_programs,
            "candidate_standalone_gate_programs_per_request": (
                0 if embed_gate_cta else packed_gate_programs
            ),
            "candidate_embedded_gate_programs_per_request": (
                packed_gate_programs if embed_gate_cta else 0
            ),
            "baseline_programs_per_request": baseline_programs_per_request,
            "candidate_programs_per_request": packed_programs_per_request,
            "baseline_programs_per_layer": batch
            * baseline_programs_per_request,
            "candidate_programs_per_layer": batch
            * packed_programs_per_request,
            "baseline_programs_all_layers": batch
            * baseline_programs_per_request
            * layer_count,
            "candidate_programs_all_layers": batch
            * packed_programs_per_request
            * layer_count,
            "requested_global_bytes_per_request_layer": {
                "baseline": baseline_gate_requested_bytes,
                "candidate": packed_gate_requested_bytes,
                "delta": packed_gate_requested_bytes
                - baseline_gate_requested_bytes,
            },
            "requested_global_bytes_all_layers": {
                "baseline": batch
                * baseline_gate_requested_bytes
                * layer_count,
                "candidate": batch
                * packed_gate_requested_bytes
                * layer_count,
                "delta": batch
                * (packed_gate_requested_bytes - baseline_gate_requested_bytes)
                * layer_count,
            },
            "traffic_classification": (
                "exact_fixed32_requested_bytes_not_measured_dram_bytes"
            ),
        },
        "launches": {
            "incumbent_per_layer": 5,
            "candidate_per_layer": 1,
            "incumbent_all_layers": 5 * layer_count,
            "candidate_all_layers": layer_count,
            "removed_all_layers": 4 * layer_count,
            "current_fusion_baseline_all_layers": layer_count,
            "gate_packed_candidate_all_layers": layer_count,
        },
        "excludes": (
            "input_weight_reads",
            "cache_effects",
            "compiler_effects",
            "timing",
        ),
    }


def _storage_identity(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def _storage_bound_failure(tensor: torch.Tensor) -> bool:
    if any(int(stride) < 0 for stride in tensor.stride()):
        return True
    if tensor.numel() == 0:
        return False
    last = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        last += (int(size) - 1) * int(stride)
    capacity = int(tensor.untyped_storage().nbytes()) // int(tensor.element_size())
    return last < 0 or last >= capacity


def _tensor_binding_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        int(tensor.data_ptr()),
        int(tensor.untyped_storage().data_ptr()),
        int(tensor.storage_offset()),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


_CAPTURE_BINDING_AUTH = object()


class _Fixed32SfwdConvPostprepCaptureBinding:
    __slots__ = (
        "auth",
        "batch_size",
        "capacity",
        "layer_index",
        "layer_name",
        "operands",
        "signatures",
        "pregather_state",
        "committer_route",
        "committer_state",
    )

    def __init__(
        self,
        *,
        batch_size: int,
        capacity: int,
        layer_index: int,
        layer_name: str,
        operands: tuple[torch.Tensor, ...],
        pregather_state: dict,
        committer_route: dict,
        committer_state: dict,
    ) -> None:
        self.auth = _CAPTURE_BINDING_AUTH
        self.batch_size = int(batch_size)
        self.capacity = int(capacity)
        self.layer_index = int(layer_index)
        self.layer_name = str(layer_name)
        self.operands = operands
        self.signatures = tuple(_tensor_binding_signature(value) for value in operands)
        self.pregather_state = pregather_state
        self.committer_route = committer_route
        self.committer_state = committer_state


def _capture_preseed_contract(
    *,
    layer_order: tuple[str, ...],
    pregather_state: dict,
    committer_route: dict,
) -> tuple[
    int,
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "fixed32 conv/post-prep capture binding preseed ran during capture"
        )
    capacity = int(pregather_state.get("max_batch_size", 0))
    batches = tuple(range(1, capacity + 1))
    ssi_sources = tuple(pregather_state.get("ssi_sources", ()))
    conv_banks = tuple(pregather_state.get("banks", ()))
    source_stages = tuple(pregather_state.get("source_stagings", ()))
    pregather_contract = pregather_state.get("contract", {})
    states_by_batch = committer_route.get("states_by_batch", {})
    if (
        capacity not in (1, 2, 3, 4)
        or len(layer_order) != LAYERS
        or len(set(layer_order)) != LAYERS
        or tuple(pregather_state.get("layer_order", ())) != layer_order
        or tuple(pregather_state.get("preseeded_batches", ())) != batches
        or set(pregather_state.get("live_selfchecked_batches", ())) != set(batches)
        or len(ssi_sources) != LAYERS
        or len(conv_banks) != LAYERS
        or len(source_stages) != LAYERS
        or pregather_contract.get("commit_route") != "fixed32_direct_source_col0"
        or pregather_contract.get("commit_row_guard_route")
        != "fixed32_triton_alias3_ownerpath_warp32_physical32_v4"
        or int(pregather_contract.get("commit_row_guard_physical_rows", -1))
        != ROWS
        or committer_route.get("mode") != pregather_state.get("mode")
        or int(committer_route.get("capacity", 0)) != capacity
        or committer_route.get("spec_state_indices")
        is not pregather_state.get("commit_spec_state_indices")
        or tuple(sorted(states_by_batch)) != batches
    ):
        raise RuntimeError(
            "fixed32 conv/post-prep capture binding persistent preseed drifted"
        )
    for batch in batches:
        state = states_by_batch[batch]
        sticky_ok = state.get("sticky_guard_ok")
        if (
            state.get("direct_metadata") is not True
            or state.get("sticky_guard") is not True
            or state.get("contract", {}).get("sticky_guard_route")
            != "fixed32_ownerpath_warp32_sticky_scalar_v5"
            or not torch.is_tensor(sticky_ok)
            or sticky_ok.dtype != torch.int32
            or tuple(sticky_ok.shape) != ()
            or not sticky_ok.is_contiguous()
            or int(sticky_ok.data_ptr())
            != int(state.get("sticky_guard_ok_data_ptr", -1))
        ):
            raise RuntimeError(
                "fixed32 conv/post-prep capture binding requires the persistent "
                f"sticky guard for B={batch}"
            )
    return capacity, ssi_sources, conv_banks, source_stages


def preseed_fixed32_sfwd_conv_postprep_capture_bindings(
    *,
    layer_order: object,
    layer_objects: object,
    pregather_state: object,
    committer_route: object,
) -> dict[str, object]:
    """Allocate every capture output and bind it to persistent fixed32 leases."""
    if not isinstance(layer_order, (list, tuple)):
        raise TypeError("fixed32 conv/post-prep layer order must be a sequence")
    order = tuple(str(value) for value in layer_order)
    if not isinstance(layer_objects, (list, tuple)) or len(layer_objects) != LAYERS:
        raise TypeError("fixed32 conv/post-prep requires 48 ordered layer objects")
    if not isinstance(pregather_state, dict) or not isinstance(committer_route, dict):
        raise TypeError("fixed32 conv/post-prep persistent states must be dicts")
    capacity, ssi_sources, conv_banks, source_stages = _capture_preseed_contract(
        layer_order=order,
        pregather_state=pregather_state,
        committer_route=committer_route,
    )
    states_by_batch = committer_route["states_by_batch"]
    expected_batches = tuple(range(1, capacity + 1))
    existing_caches = tuple(
        getattr(layer, "_fr13_fixed32_sfwd_conv_postprep_outputs", None)
        for layer in layer_objects
    )
    existing_graph_caches = tuple(
        isinstance(cache, dict)
        and cache.get("schema") == CAPTURE_CACHE_SCHEMA
        for cache in existing_caches
    )
    if any(existing_graph_caches):
        if not all(existing_graph_caches):
            raise RuntimeError(
                "fixed32 conv/post-prep graph output caches are partially preseeded"
            )
        for layer_index, (layer_name, cache) in enumerate(
            zip(order, existing_caches, strict=True)
        ):
            by_batch = cache.get("by_batch")
            if (
                str(getattr(layer_objects[layer_index], "prefix", ""))
                != layer_name
                or int(cache.get("capacity", 0)) != capacity
                or cache.get("pregather_state") is not pregather_state
                or cache.get("committer_route") is not committer_route
                or not isinstance(cache.get("base"), dict)
                or set(cache["base"]) != {
                    "query",
                    "key_tensor",
                    "value_spec",
                    "value_tree",
                    "g",
                    "beta",
                }
                or not isinstance(by_batch, dict)
                or tuple(sorted(by_batch)) != expected_batches
            ):
                raise RuntimeError(
                    "fixed32 conv/post-prep graph output cache lease drifted: "
                    f"layer={layer_name!r} index={layer_index}"
                )
            for batch in expected_batches:
                entry = by_batch[batch]
                if not isinstance(entry, dict):
                    raise RuntimeError(
                        "fixed32 conv/post-prep graph output entry is malformed"
                    )
                _validate_capture_binding(
                    entry.get("capture_binding"),
                    batch_size=batch,
                    spec_state_indices=entry.get("spec_state_indices"),
                    conv_state=entry.get("conv_state"),
                    query=entry.get("query"),
                    key=entry.get("key_tensor"),
                    value_spec=entry.get("value_spec"),
                    value_tree=entry.get("value_tree"),
                    g=entry.get("g"),
                    beta=entry.get("beta"),
                    source_stage=entry.get("source_stage"),
                )
        return {
            "ready": True,
            "schema": CAPTURE_CACHE_SCHEMA,
            "capacity": capacity,
            "layers": LAYERS,
            "batches": expected_batches,
            "base_output_tensors": LAYERS * 6,
            "bound_output_views": LAYERS * capacity * 6,
            "capture_host_syncs_per_layer": 0,
            "ssi_value_proof": "persistent_pregather_boot_selfcheck",
            "runtime_guard": "persistent_sticky_committer_scalar",
        }
    pending: list[tuple[object, dict[str, object]]] = []
    output_tensors = 0
    output_views = 0
    for layer_index, (layer_name, layer) in enumerate(
        zip(order, layer_objects, strict=True)
    ):
        source_stage = source_stages[layer_index]
        spec_state_indices = ssi_sources[layer_index]
        conv_state = conv_banks[layer_index]
        if (
            str(getattr(layer, "prefix", "")) != layer_name
            or not torch.is_tensor(source_stage)
            or source_stage.device.type != "cuda"
            or source_stage.dtype != torch.bfloat16
            or source_stage.ndim != 2
            or int(source_stage.shape[0]) < capacity * SOURCE_ROWS
            or int(source_stage.shape[1]) != CHANNELS
            or tuple(int(value) for value in source_stage.stride())
            != (CHANNELS, 1)
            or not torch.is_tensor(spec_state_indices)
            or spec_state_indices.dtype != torch.int32
            or spec_state_indices.device != source_stage.device
            or tuple(int(value) for value in spec_state_indices.shape)
            != (capacity, ROWS)
            or tuple(int(value) for value in spec_state_indices.stride())
            != (ROWS, 1)
            or not torch.is_tensor(conv_state)
            or conv_state.device != source_stage.device
            or conv_state.dtype != torch.bfloat16
        ):
            raise RuntimeError(
                "fixed32 conv/post-prep capture preseed layer geometry drifted: "
                f"layer={layer_name!r} index={layer_index}"
            )
        capacity_rows = capacity * ROWS
        base = {
            "query": torch.empty(
                (1, capacity_rows, NUM_K_HEADS, HEAD_K_DIM),
                dtype=torch.bfloat16,
                device=source_stage.device,
            ),
            "key_tensor": torch.empty(
                (1, capacity_rows, NUM_K_HEADS, HEAD_K_DIM),
                dtype=torch.bfloat16,
                device=source_stage.device,
            ),
            "value_spec": torch.empty(
                (1, capacity_rows, NUM_V_HEADS, HEAD_V_DIM),
                dtype=torch.bfloat16,
                device=source_stage.device,
            ),
            "value_tree": torch.empty(
                (capacity_rows, NUM_V_HEADS, HEAD_V_DIM),
                dtype=torch.bfloat16,
                device=source_stage.device,
            ),
            "g": torch.empty(
                (capacity_rows, NUM_V_HEADS),
                dtype=torch.float32,
                device=source_stage.device,
            ),
            "beta": torch.empty(
                (capacity_rows, NUM_V_HEADS),
                dtype=torch.float32,
                device=source_stage.device,
            ),
        }
        output_tensors += len(base)
        by_batch: dict[int, dict[str, object]] = {}
        for batch in range(1, capacity + 1):
            rows = batch * ROWS
            query = base["query"].as_strided(
                (1, rows, NUM_K_HEADS, HEAD_K_DIM),
                (rows * Q_DIM, Q_DIM, HEAD_K_DIM, 1),
            )
            key = base["key_tensor"].as_strided(
                (1, rows, NUM_K_HEADS, HEAD_K_DIM),
                (rows * Q_DIM, Q_DIM, HEAD_K_DIM, 1),
            )
            value_spec = base["value_spec"].as_strided(
                (1, rows, NUM_V_HEADS, HEAD_V_DIM),
                (rows * V_DIM, V_DIM, HEAD_V_DIM, 1),
            )
            value_tree = base["value_tree"][:rows]
            g = base["g"][:rows]
            beta = base["beta"][:rows]
            operands = (
                spec_state_indices,
                conv_state,
                query,
                key,
                value_spec,
                value_tree,
                g,
                beta,
                source_stage,
                states_by_batch[batch]["sticky_guard_ok"],
            )
            binding = _Fixed32SfwdConvPostprepCaptureBinding(
                batch_size=batch,
                capacity=capacity,
                layer_index=layer_index,
                layer_name=layer_name,
                operands=operands,
                pregather_state=pregather_state,
                committer_route=committer_route,
                committer_state=states_by_batch[batch],
            )
            by_batch[batch] = {
                "query": query,
                "key_tensor": key,
                "value_spec": value_spec,
                "value_tree": value_tree,
                "g": g,
                "beta": beta,
                "source_stage": source_stage,
                "spec_state_indices": spec_state_indices,
                "conv_state": conv_state,
                "capture_binding": binding,
            }
            output_views += 6
        pending.append(
            (
                layer,
                {
                    "schema": CAPTURE_CACHE_SCHEMA,
                    "capacity": capacity,
                    "base": base,
                    "by_batch": by_batch,
                    "pregather_state": pregather_state,
                    "committer_route": committer_route,
                },
            )
        )
    for layer, cache in pending:
        layer._fr13_fixed32_sfwd_conv_postprep_outputs = cache
    return {
        "ready": True,
        "schema": CAPTURE_CACHE_SCHEMA,
        "capacity": capacity,
        "layers": len(pending),
        "batches": expected_batches,
        "base_output_tensors": output_tensors,
        "bound_output_views": output_views,
        "capture_host_syncs_per_layer": 0,
        "ssi_value_proof": "persistent_pregather_boot_selfcheck",
        "runtime_guard": "persistent_sticky_committer_scalar",
    }


def _validate_capture_binding(
    binding: object,
    *,
    batch_size: int,
    spec_state_indices: torch.Tensor,
    conv_state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value_spec: torch.Tensor,
    value_tree: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    source_stage: torch.Tensor,
) -> None:
    if (
        type(binding) is not _Fixed32SfwdConvPostprepCaptureBinding
        or binding.auth is not _CAPTURE_BINDING_AUTH
        or binding.batch_size != int(batch_size)
    ):
        raise RuntimeError("fixed32 conv/post-prep capture binding is invalid")
    operands = (
        spec_state_indices,
        conv_state,
        query,
        key,
        value_spec,
        value_tree,
        g,
        beta,
        source_stage,
        binding.committer_state.get("sticky_guard_ok"),
    )
    if any(
        current is not bound
        for current, bound in zip(operands, binding.operands, strict=True)
    ) or tuple(_tensor_binding_signature(value) for value in operands) != (
        binding.signatures
    ):
        raise RuntimeError(
            "fixed32 conv/post-prep capture operand object/data_ptr drifted"
        )
    pregather_state = binding.pregather_state
    committer_route = binding.committer_route
    committer_state = binding.committer_state
    from lumo_flywheel_serving import fr10_gdn_tree_kernel as fixed32_kernel

    current_pregather = getattr(
        fixed32_kernel, "_FR13_FIXED32_CONV_PREGATHER", {}
    ).get("state")
    current_committer = getattr(
        fixed32_kernel, "_FR13_FIXED32_COMMITTER_FAST_ROUTE", {}
    ).get("state")
    layer_index = binding.layer_index
    if (
        current_pregather is not pregather_state
        or current_committer is not committer_route
        or pregather_state["ssi_sources"][layer_index] is not spec_state_indices
        or pregather_state["banks"][layer_index] is not conv_state
        or pregather_state["source_stagings"][layer_index] is not source_stage
        or committer_route["states_by_batch"].get(binding.batch_size)
        is not committer_state
        or committer_state.get("direct_metadata") is not True
        or committer_state.get("sticky_guard") is not True
        or int(committer_state["sticky_guard_ok"].data_ptr())
        != int(committer_state.get("sticky_guard_ok_data_ptr", -1))
    ):
        raise RuntimeError(
            "fixed32 conv/post-prep persistent capture lease changed"
        )


def fixed32_sfwd_conv_postprep_layout_contract(
    *,
    batch_size: int,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value_spec: torch.Tensor,
    value_tree: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    source_stage: torch.Tensor,
    conv_tap: torch.Tensor | None,
    expected_device_type: str = "cuda",
    state_indices_prevalidated: bool = False,
) -> dict[str, object]:
    """Fail closed over every tensor layout, storage bound, and output alias."""
    batch = int(batch_size)
    if batch not in (1, 2, 3, 4):
        raise ValueError("fixed32 conv/post-prep layout requires B1-B4")
    required_rows = batch * ROWS
    required_source_rows = batch * SOURCE_ROWS
    named: dict[str, torch.Tensor | None] = {
        "x": x,
        "conv_state": conv_state,
        "spec_state_indices": spec_state_indices,
        "conv_weights": conv_weights,
        "bias": bias,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "query": query,
        "key": key,
        "value_spec": value_spec,
        "value_tree": value_tree,
        "g": g,
        "beta": beta,
        "source_stage": source_stage,
        "conv_tap": conv_tap,
    }
    if any(
        tensor is not None and not torch.is_tensor(tensor)
        for tensor in named.values()
    ):
        raise TypeError("fixed32 conv/post-prep operands must be tensors or None")
    if not torch.is_tensor(x):
        raise TypeError("fixed32 conv/post-prep x must be a tensor")
    device = x.device
    present = {name: tensor for name, tensor in named.items() if tensor is not None}
    failures: list[str] = []
    if expected_device_type and device.type != expected_device_type:
        failures.append("device_type")
    if any(tensor.device != device for tensor in present.values()):
        failures.append("device_mismatch")

    exact_specs: Mapping[str, tuple[tuple[int, ...], tuple[int, ...], torch.dtype]] = {
        "x": ((required_rows, CHANNELS), (X_ROW_STRIDE, 1), torch.bfloat16),
        "conv_weights": ((CHANNELS, CONV_WIDTH), (CONV_WIDTH, 1), torch.bfloat16),
        "a": ((required_rows, NUM_V_HEADS), (NUM_V_HEADS, 1), torch.bfloat16),
        "b": ((required_rows, NUM_V_HEADS), (NUM_V_HEADS, 1), torch.bfloat16),
        "A_log": ((NUM_V_HEADS,), (1,), torch.float32),
        "dt_bias": ((NUM_V_HEADS,), (1,), torch.bfloat16),
        "query": (
            (1, required_rows, NUM_K_HEADS, HEAD_K_DIM),
            (required_rows * Q_DIM, Q_DIM, HEAD_K_DIM, 1),
            torch.bfloat16,
        ),
        "key": (
            (1, required_rows, NUM_K_HEADS, HEAD_K_DIM),
            (required_rows * Q_DIM, Q_DIM, HEAD_K_DIM, 1),
            torch.bfloat16,
        ),
        "value_spec": (
            (1, required_rows, NUM_V_HEADS, HEAD_V_DIM),
            (required_rows * V_DIM, V_DIM, HEAD_V_DIM, 1),
            torch.bfloat16,
        ),
        "value_tree": (
            (required_rows, NUM_V_HEADS, HEAD_V_DIM),
            (V_DIM, HEAD_V_DIM, 1),
            torch.bfloat16,
        ),
        "g": (
            (required_rows, NUM_V_HEADS),
            (NUM_V_HEADS, 1),
            torch.float32,
        ),
        "beta": (
            (required_rows, NUM_V_HEADS),
            (NUM_V_HEADS, 1),
            torch.float32,
        ),
    }
    if conv_tap is not None:
        exact_specs = dict(exact_specs)
        exact_specs["conv_tap"] = (
            (required_rows, CHANNELS),
            (CHANNELS, 1),
            torch.bfloat16,
        )
    for name, (shape, stride, dtype) in exact_specs.items():
        tensor = present[name]
        if tuple(int(value) for value in tensor.shape) != shape:
            failures.append(f"{name}_shape")
        if tuple(int(value) for value in tensor.stride()) != stride:
            failures.append(f"{name}_stride")
        if tensor.dtype != dtype:
            failures.append(f"{name}_dtype")

    if (
        spec_state_indices.ndim != 2
        or int(spec_state_indices.shape[0]) < batch
        or int(spec_state_indices.shape[1]) != ROWS
    ):
        failures.append("spec_state_indices_shape")
    if tuple(int(value) for value in spec_state_indices.stride()) != (ROWS, 1):
        failures.append("spec_state_indices_stride")
    if spec_state_indices.dtype != torch.int32:
        failures.append("spec_state_indices_dtype")
    if (
        source_stage.ndim != 2
        or int(source_stage.shape[0]) < required_source_rows
        or int(source_stage.shape[1]) != CHANNELS
    ):
        failures.append("source_stage_shape")
    if tuple(int(value) for value in source_stage.stride()) != (CHANNELS, 1):
        failures.append("source_stage_stride")
    if source_stage.dtype != torch.bfloat16:
        failures.append("source_stage_dtype")

    if conv_state.ndim != 3:
        failures.append("conv_state_ndim")
    else:
        if int(conv_state.shape[0]) <= 0:
            failures.append("conv_state_bank_rows")
        if tuple(int(value) for value in conv_state.shape[1:]) != (
            CHANNELS,
            CONV_STATE_LEN,
        ):
            failures.append("conv_state_shape")
        conv_stride = tuple(int(value) for value in conv_state.stride())
        if (
            conv_stride[1:] != (1, CHANNELS)
            or conv_stride[0] < CHANNELS * CONV_STATE_LEN
        ):
            failures.append("conv_state_stride")
        if conv_state.dtype != torch.bfloat16:
            failures.append("conv_state_dtype")
    state_index_bounds: tuple[int, int] | None = None
    if (
        not state_indices_prevalidated
        and spec_state_indices.ndim == 2
        and int(spec_state_indices.shape[0]) >= batch
        and int(spec_state_indices.shape[1]) == ROWS
        and spec_state_indices.dtype == torch.int32
        and spec_state_indices.device == device
        and conv_state.ndim == 3
        and int(conv_state.size(0)) > 0
    ):
        index_min_tensor, index_max_tensor = torch.aminmax(
            spec_state_indices[:batch, :ROWS]
        )
        state_index_bounds = (
            int(index_min_tensor.item()),
            int(index_max_tensor.item()),
        )
        if (
            state_index_bounds[0] < 0
            or state_index_bounds[1] >= int(conv_state.size(0))
        ):
            failures.append("spec_state_indices_values")
    if bias is not None and (
        tuple(int(value) for value in bias.shape) != (CHANNELS,)
        or tuple(int(value) for value in bias.stride()) != (1,)
        or bias.dtype not in (torch.bfloat16, torch.float32)
    ):
        failures.append("bias_layout")
    if int(conv_weights.data_ptr()) % 4 != 0:
        failures.append("conv_weights_u32_alignment")
    for name, tensor in present.items():
        if _storage_bound_failure(tensor):
            failures.append(f"{name}_storage_bounds")

    writable_names = [
        "query",
        "key",
        "value_spec",
        "value_tree",
        "g",
        "beta",
        "source_stage",
    ]
    if conv_tap is not None:
        writable_names.append("conv_tap")
    readable_names = [
        "x",
        "conv_state",
        "spec_state_indices",
        "conv_weights",
        "a",
        "b",
        "A_log",
        "dt_bias",
    ]
    if bias is not None:
        readable_names.append("bias")
    writable_storage = [_storage_identity(present[name]) for name in writable_names]
    if len(set(writable_storage)) != len(writable_storage):
        failures.append("writable_storage_alias")
    readable_storage = {_storage_identity(present[name]) for name in readable_names}
    if any(identity in readable_storage for identity in writable_storage):
        failures.append("input_output_storage_alias")
    if failures:
        observed = {
            name: {
                "shape": tuple(int(value) for value in tensor.shape),
                "stride": tuple(int(value) for value in tensor.stride()),
                "dtype": str(tensor.dtype),
                "storage_offset": int(tensor.storage_offset()),
                "storage_bytes": int(tensor.untyped_storage().nbytes()),
            }
            for name, tensor in present.items()
        }
        raise ValueError(
            "fixed32 conv/post-prep operand contract drift: "
            f"failed={tuple(failures)!r}; observed={observed!r}"
        )
    return {
        "batch_size": batch,
        "device": str(device),
        "conv_state_stride_row": int(conv_state.stride(0)),
        "state_index_bounds": state_index_bounds,
        "state_indices_prevalidated": bool(state_indices_prevalidated),
        "conv_tap": conv_tap is not None,
        "writable_storages": len(writable_storage),
        "input_aliases_allowed": True,
        "input_output_aliases_allowed": False,
    }


def launch_fixed32_sfwd_conv_postprep_fusion(
    *,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value_spec: torch.Tensor,
    value_tree: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    source_stage: torch.Tensor,
    conv_tap: torch.Tensor | None,
    batch_size: int,
    fixed32_mode: str,
    tree_parent: object,
    qualification_profile: str,
    draft_vocab_k: int,
    draft_vocab_root: int,
    embed_gate_cta: bool = False,
    direct_nodegroup8: bool = False,
    source_only_qualification: bool = False,
    capture_binding: object | None = None,
) -> dict[str, object]:
    """Launch one default-off per-layer candidate kernel."""
    if source_only_qualification is not True:
        raise RuntimeError(
            "fixed32 conv/post-prep fusion is source-only and default-off"
        )
    capturing = torch.cuda.is_current_stream_capturing()
    if capture_binding is None and capturing:
        raise RuntimeError(
            "fixed32 conv/post-prep fusion capture requires a persistent binding"
        )
    contract = fixed32_sfwd_conv_postprep_fusion_contract(
        batch_size,
        fixed32_mode=fixed32_mode,
        tree_parent=tree_parent,
        qualification_profile=qualification_profile,
        draft_vocab_k=draft_vocab_k,
        draft_vocab_root=draft_vocab_root,
        embed_gate_cta=embed_gate_cta,
        direct_nodegroup8=direct_nodegroup8,
    )
    if capture_binding is not None:
        _validate_capture_binding(
            capture_binding,
            batch_size=batch_size,
            spec_state_indices=spec_state_indices,
            conv_state=conv_state,
            query=query,
            key=key,
            value_spec=value_spec,
            value_tree=value_tree,
            g=g,
            beta=beta,
            source_stage=source_stage,
        )
    layout = fixed32_sfwd_conv_postprep_layout_contract(
        batch_size=batch_size,
        x=x,
        conv_state=conv_state,
        spec_state_indices=spec_state_indices,
        conv_weights=conv_weights,
        bias=bias,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        query=query,
        key=key,
        value_spec=value_spec,
        value_tree=value_tree,
        g=g,
        beta=beta,
        source_stage=source_stage,
        conv_tap=conv_tap,
        state_indices_prevalidated=capture_binding is not None,
    )
    block_c = int(contract["block_c"])
    num_warps = int(contract["num_warps"])
    channel_tasks = CHANNELS // block_c
    if direct_nodegroup8:
        channel_tasks *= 4
    gate_rows_per_task = 2 * block_c // GATE_BLOCK
    gate_tasks = triton.cdiv(ROWS, gate_rows_per_task)
    grid = (
        int(batch_size),
        channel_tasks if embed_gate_cta else channel_tasks + gate_tasks,
    )
    bias_arg = bias if bias is not None else x
    conv_tap_arg = conv_tap if conv_tap is not None else query
    sticky_guard_arg = (
        capture_binding.committer_state["sticky_guard_ok"]
        if capture_binding is not None
        else spec_state_indices
    )
    if direct_nodegroup8:
        _fr13_fixed32_sfwd_conv_postprep_nodegroup8_direct_kernel[grid](
            x,
            conv_state,
            spec_state_indices,
            sticky_guard_arg,
            conv_weights,
            bias_arg,
            a,
            b,
            A_log,
            dt_bias,
            query,
            key,
            value_spec,
            value_tree,
            g,
            beta,
            source_stage,
            conv_tap_arg,
            CONV_STRIDE_ROW=int(conv_state.stride(0)),
            BANK_ROWS=int(conv_state.size(0)),
            B=int(batch_size),
            N=ROWS,
            C=CHANNELS,
            WIDTH=CONV_WIDTH,
            STATE_LEN=CONV_STATE_LEN,
            SOURCE_ROWS=SOURCE_ROWS,
            H=NUM_K_HEADS,
            HV=NUM_V_HEADS,
            K=HEAD_K_DIM,
            V=HEAD_V_DIM,
            HAS_BIAS=bias is not None,
            STORE_CONV_TAP=conv_tap is not None,
            CAPTURE_GUARD=capture_binding is not None,
            X_STRIDE_ROW=X_ROW_STRIDE,
            BLOCK_C=block_c,
            GATE_BLOCK=GATE_BLOCK,
            SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
            EMBED_GATE_CTA=embed_gate_cta,
            num_warps=num_warps,
        )
    else:
        _fr13_fixed32_sfwd_conv_postprep_fusion_kernel[grid](
            x,
            conv_state,
            spec_state_indices,
            sticky_guard_arg,
            conv_weights,
            bias_arg,
            a,
            b,
            A_log,
            dt_bias,
            query,
            key,
            value_spec,
            value_tree,
            g,
            beta,
            source_stage,
            conv_tap_arg,
            CONV_STRIDE_ROW=int(conv_state.stride(0)),
            BANK_ROWS=int(conv_state.size(0)),
            B=int(batch_size),
            N=ROWS,
            C=CHANNELS,
            WIDTH=CONV_WIDTH,
            STATE_LEN=CONV_STATE_LEN,
            SOURCE_ROWS=SOURCE_ROWS,
            H=NUM_K_HEADS,
            HV=NUM_V_HEADS,
            K=HEAD_K_DIM,
            V=HEAD_V_DIM,
            HAS_BIAS=bias is not None,
            STORE_CONV_TAP=conv_tap is not None,
            CAPTURE_GUARD=capture_binding is not None,
            X_STRIDE_ROW=X_ROW_STRIDE,
            BLOCK_C=block_c,
            GATE_BLOCK=GATE_BLOCK,
            SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
            EMBED_GATE_CTA=embed_gate_cta,
            num_warps=num_warps,
        )
    return {
        **contract,
        "layout": layout,
        "conv_tap": conv_tap is not None,
        "capture_bound": capture_binding is not None,
    }
