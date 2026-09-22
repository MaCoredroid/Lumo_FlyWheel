# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import ast
import hashlib
import json
import os
import time
import torch
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from vllm import envs
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops import (
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from lumo_flywheel_serving.fr10_gdn_tree_kernel import fixed32_batch_gdn_selector, fixed32_sfwd_state_fusion_byte_gate, fixed32_sfwd_state_fusion_gate_control, gather_committed_path_conv_prior, launch_fixed32_sfwd_state_fusion, launch_tree_gdn_prepared, launch_tree_gdn_prepared_fixed32_batch, launch_tree_state_linear_remap, subtree_get
from lumo_flywheel_serving.fr13_sfwd_state_fusion_production import fixed32_sfwd_state_fusion_production_control, fixed32_sfwd_state_fusion_production_engagement
from lumo_flywheel_serving.fr13_replay_conv_remap import replay_conv_state_linear_remap
from lumo_flywheel_serving.fr13_ex2_silu import triton_ex2_silu_bf16
from lumo_flywheel_serving.fr13_tree_conv_fused import build_tree_conv_state_src_indices, conv_wb_staging_get, freeze_conv_wb_staging_sources, fused_tree_conv_source, fused_tree_conv_sources_batched, fused_tree_conv_state_rows, fused_tree_conv_taps_acc, gather_committed_path_conv_prior_prepared, launch_conv_state_writeback, launch_conv_state_writeback_batched, prepare_committed_path_conv_rows, prepare_replay_conv_remap_rows, replay_conv_state_linear_remap_prepared

_FR10_DECODE_MODE = os.environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp")
_FR13_FIXED32_MODE = ''
_FR13_FIXED32_VALID_MASK = 0
_FR13_FIXED32_GDN_PATH_BV_CANDIDATE = None
_FR13_FIXED32_GDN_PATH_BV_PRODUCTION = None
_FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH = None
_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION = False
_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH = None
_FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION = False
_FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH = None
_FR13_FIXED32_TAW_NATIVE_PRECOMPUTE = False
_FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB = False
_FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB = False
_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION = False
_FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB = False
_FR13_FIXED32_SFWD_EMBED_GATE_CTA = False
_FR13_FIXED32_SFWD_NODEGROUP8_DIRECT = False
_FR13_FIXED32_SFWD_CONV_POSTPREP_GRAPH = False
_FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_PATH = '/logs/fr13_fixed32_sfwd_prior_reuse.source_manifest.json'
_FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_SHA256 = ''
_FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_COMMIT = ''
_FR13_FIXED32_EAGER_KERNEL_DIAGNOSTIC = False
_FR13_DRAFT_HEAD_U8_WORKER_ENV_REQUIRED = False
_FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_REQUIRED = False

# FR13_FIXED32_OBSERVED_RUNTIME: event-scoped, host-only accounting at the
# successful runtime call sites. This deliberately records tensor geometry and
# enqueue deltas without reading device values or synchronizing the stream.
_FR13_NSIGHT_ATTESTATION_KEYS = (
    "NVTX_INJECTION64_PATH",
    "NSYSDK_INJECTION64_PATH",
    "NSYS_PROFILING_SESSION_ID",
    "LUMO_NSYS_SESSION_NAME",
)


def _fr13_fixed32_emit_nsight_attestation(
    path="/logs/fr13_fixed32_enginecore_nsight_attestation.json",
):
    """Publish THIS process's REAL Nsight environment for the profiler.

    WHY THIS EXISTS. /proc/<pid>/environ is not a usable source for this process.
    set_process_title() calls setproctitle("VLLM::EngineCore"), which reclaims
    the contiguous argv+environ block and NUL-fills it -- so an outside reader
    sees a zeroed or partially overwritten environment for EXACTLY the process
    whose argv the attestation requires to be "VLLM::EngineCore". The old check
    selected precisely the process whose environ it had destroyed. Variables nsys
    injects with setenv() after exec never appear in that block at all.
    os.environ inside the process is authoritative; nothing outside it is.

    The 2026-08-08 run passed the old check by byte-layout luck -- which
    variables survive the overwrite depends on the block's contents, and the FR14
    stack's differ. AN ATTESTATION SOURCE MUST BE IMMUNE TO THE PROCESS IT
    ATTESTS.

    BEST EFFORT BY DESIGN -- never raises. A missing or malformed artifact makes
    the PROFILER refuse, which is fail-closed where the evidence is consumed; it
    must never take down a serve nobody is profiling. Gated on
    FR13_FIXED32_NVTX_PROFILE so an unprofiled run leaves no sidecar behind,
    matching this tree's rule that an off lever leaves no artifact.
    """
    _json = __import__("json")
    _os = __import__("os")
    try:
        if _os.environ.get("FR13_FIXED32_NVTX_PROFILE") != "1":
            return None
        record = {
            "schema": "fr13.fixed32.enginecore_nsight_attestation.v1",
            "pid": int(_os.getpid()),
            "environ": {
                key: _os.environ[key]
                for key in _FR13_NSIGHT_ATTESTATION_KEYS
                if key in _os.environ
            },
        }
        tmp = path + ".tmp." + str(_os.getpid())
        with open(tmp, "w", encoding="ascii") as handle:
            handle.write(
                _json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
            )
        _os.replace(tmp, path)
        return record
    except Exception:
        return None


_FR13_FIXED32_NSIGHT_ATTESTATION = _fr13_fixed32_emit_nsight_attestation()


_FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS = (
    "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB",
    "FR13_DRAFT_HEAD_M1_R64_U8_QUALITY_GATE",
    "FR13_DRAFT_HEAD_M1_R64_U8_TAW_QUALITY_GATE",
    "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION",
    "FR13_DRAFT_HEAD_M1_R64_U8_SO",
    "FR13_DRAFT_HEAD_M1_R64_U8_SO_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_BUILD_ATTESTATION_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_PATCH_SOURCE_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_RUNNER_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_SUBSET_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_VOCAB_BLOCKS_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_FA2_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_TAW_SOURCE_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_COMMIT",
    "FR13_DRAFT_HEAD_M1_R64_U8_INSTANCE_ID",
    "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_JSON",
    "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_PASS_SIDECAR",
    "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_PASS_SIDECAR_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_INTERNAL_PRODUCTION_ATTESTED",
    "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_ENGAGEMENT_JSON",
    "FR13_DRAFT_VOCAB_BLOCKS",
    "FR13_DRAFT_VOCAB_K",
    "FR13_DRAFT_VOCAB_ROOT",
)

_FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_KEYS = (
    "FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB",
    "FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE",
    "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION",
    "FR13_DRAFT_HEAD_M4_R64_U8_SO",
    "FR13_DRAFT_HEAD_M4_R64_U8_SO_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_BUILD_ATTESTATION_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_PATCH_SOURCE_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_RUNNER_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_SUBSET_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_VOCAB_BLOCKS_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_FA2_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_TAW_SOURCE_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_COMMIT",
    "FR13_DRAFT_HEAD_M4_R64_U8_TASK_IDS",
    "FR13_DRAFT_HEAD_M4_R64_U8_LIVE_JSON",
    "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION_PASS_SIDECAR",
    "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION_PASS_SIDECAR_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_INTERNAL_PRODUCTION_ATTESTED",
    "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION_ENGAGEMENT_JSON",
    "FR13_DRAFT_VOCAB_BLOCKS",
    "FR13_DRAFT_VOCAB_K",
    "FR13_DRAFT_VOCAB_ROOT",
)


def _fr13_draft_head_u8_worker_env_bridge(
    sidecar_path="/logs/fr13_draft_head_m1_r64_u8.worker_env.json",
):
    """Restore the exact qualification env inside the curated EngineCore."""
    _hashlib = __import__("hashlib")
    _json = __import__("json")
    _os = __import__("os")
    _pathlib = __import__("pathlib")
    _stat = __import__("stat")
    try:
        required = _FR13_DRAFT_HEAD_U8_WORKER_ENV_REQUIRED is True
    except NameError:
        required = False
    path = _pathlib.Path(sidecar_path)
    if not path.exists() and not path.is_symlink():
        if required:
            raise RuntimeError("FR13 draft-head U8 worker env sidecar is missing")
        return None
    if not required:
        raise RuntimeError("FR13 draft-head U8 worker env sidecar leaked while off")
    info = path.lstat()
    if (
        not _stat.S_ISREG(info.st_mode)
        or _stat.S_IMODE(info.st_mode) != 0o400
        or not 1 <= info.st_size <= 16384
    ):
        raise RuntimeError("FR13 draft-head U8 worker env sidecar is not canonical")
    raw = path.read_bytes()
    try:
        record = _json.loads(raw)
    except Exception as exc:
        raise RuntimeError(
            "FR13 draft-head U8 worker env sidecar is not JSON"
        ) from exc
    if not isinstance(record, dict) or set(record) != {
        "payload",
        "payload_sha256",
        "schema",
    }:
        raise RuntimeError("FR13 draft-head U8 worker env record drifted")
    payload = record.get("payload")
    if (
        record.get("schema")
        != "fr13.fixed32.dfwd_k64_m1_r64_u8_worker_env.v1"
        or not isinstance(payload, dict)
        or tuple(sorted(payload))
        != tuple(sorted(_FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS))
        or any(not isinstance(value, str) for value in payload.values())
    ):
        raise RuntimeError("FR13 draft-head U8 worker env payload drifted")
    canonical_payload = _json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    payload_sha256 = _hashlib.sha256(canonical_payload).hexdigest()
    live_enabled = payload["FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB"] == "1"
    quality_enabled = (
        payload["FR13_DRAFT_HEAD_M1_R64_U8_QUALITY_GATE"] == "1"
    )
    taw_quality_enabled = (
        payload["FR13_DRAFT_HEAD_M1_R64_U8_TAW_QUALITY_GATE"] == "1"
    )
    production_enabled = (
        payload["FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION"] == "1"
    )
    production_sha = payload[
        "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_PASS_SIDECAR_SHA256"
    ]
    if (
        record.get("payload_sha256") != payload_sha256
        or (live_enabled == production_enabled)
        or payload["FR13_DRAFT_HEAD_M1_R64_U8_QUALITY_GATE"] not in ("0", "1")
        or payload["FR13_DRAFT_HEAD_M1_R64_U8_TAW_QUALITY_GATE"]
        not in ("0", "1")
        or (quality_enabled and not live_enabled)
        or (taw_quality_enabled and not quality_enabled)
        or payload["FR13_DRAFT_VOCAB_K"] != "65536"
        or payload["FR13_DRAFT_VOCAB_ROOT"] != "1"
        or payload["FR13_DRAFT_VOCAB_BLOCKS"]
        != "/workspace/scripts/fr13_dvk_subset_blocks.json"
        or (
            live_enabled
            and (
                payload[
                    "FR13_DRAFT_HEAD_M1_R64_U8_"
                    "INTERNAL_PRODUCTION_ATTESTED"
                ]
                or payload[
                    "FR13_DRAFT_HEAD_M1_R64_U8_"
                    "PRODUCTION_PASS_SIDECAR"
                ]
                or production_sha
            )
        )
        or (
            production_enabled
            and (
                payload[
                    "FR13_DRAFT_HEAD_M1_R64_U8_"
                    "INTERNAL_PRODUCTION_ATTESTED"
                ]
                != "1"
                or payload[
                    "FR13_DRAFT_HEAD_M1_R64_U8_"
                    "PRODUCTION_PASS_SIDECAR"
                ]
                != "/logs/fr13_dfwd_k64_m1_r64_u8.production_credential.json"
                or len(production_sha) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in production_sha
                )
            )
        )
    ):
        raise RuntimeError(
            "FR13 draft-head U8 worker env digest/K64-root drifted"
        )
    conflicts = {
        key: (_os.environ[key], value)
        for key, value in payload.items()
        if key in _os.environ and _os.environ[key] != value
    }
    if conflicts:
        raise RuntimeError(
            "FR13 draft-head U8 inherited worker env conflicts with sidecar: "
            + repr(conflicts)
        )
    _os.environ.update(payload)
    return {
        "schema": "fr13.fixed32.dfwd_k64_m1_r64_u8_worker_env_bridge.v1",
        "sidecar": str(path),
        "sidecar_sha256": _hashlib.sha256(raw).hexdigest(),
        "payload_sha256": payload_sha256,
        "hydrated_keys": list(_FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS),
    }


_FR13_DRAFT_HEAD_U8_WORKER_ENV_BRIDGE = (
    _fr13_draft_head_u8_worker_env_bridge()
)


def _fr13_draft_head_m4_u8_worker_env_bridge(
    sidecar_path="/logs/fr13_draft_head_m4_r64_u8.worker_env.json",
):
    """Restore the exact B4 live qualification env inside EngineCore."""
    _hashlib = __import__("hashlib")
    _json = __import__("json")
    _os = __import__("os")
    _pathlib = __import__("pathlib")
    _stat = __import__("stat")
    try:
        required = _FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_REQUIRED is True
    except NameError:
        required = False
    path = _pathlib.Path(sidecar_path)
    if not path.exists() and not path.is_symlink():
        if required:
            raise RuntimeError("FR13 draft-head M4 U8 worker env sidecar is missing")
        return None
    if not required:
        raise RuntimeError("FR13 draft-head M4 U8 worker env sidecar leaked while off")
    info = path.lstat()
    if (
        not _stat.S_ISREG(info.st_mode)
        or _stat.S_IMODE(info.st_mode) != 0o400
        or not 1 <= info.st_size <= 16384
    ):
        raise RuntimeError("FR13 draft-head M4 U8 worker env sidecar is not canonical")
    raw = path.read_bytes()
    try:
        record = _json.loads(raw)
    except Exception as exc:
        raise RuntimeError("FR13 draft-head M4 U8 worker env sidecar is not JSON") from exc
    payload = record.get("payload") if isinstance(record, dict) else None
    canonical_payload = _json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii") if isinstance(payload, dict) else b""
    payload_sha256 = _hashlib.sha256(canonical_payload).hexdigest()
    task_ids = (
        "astropy__astropy-12907,astropy__astropy-13033,"
        "astropy__astropy-13236,astropy__astropy-13398"
    )
    live_enabled = payload.get("FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB") == "1"
    quality_enabled = (
        payload.get("FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE") == "1"
    )
    production_enabled = (
        payload.get("FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION") == "1"
    )
    production_sha = payload.get(
        "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION_PASS_SIDECAR_SHA256", ""
    )
    hash_keys = tuple(
        key for key in _FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_KEYS
        if key.endswith("SHA256")
        and not key.endswith("PRODUCTION_PASS_SIDECAR_SHA256")
    )
    if (
        not isinstance(record, dict)
        or set(record) != {"payload", "payload_sha256", "schema"}
        or record.get("schema")
        != "fr13.fixed32.dfwd_k64_m4_r64_u8_worker_env.v1"
        or not isinstance(payload, dict)
        or tuple(sorted(payload))
        != tuple(sorted(_FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_KEYS))
        or any(not isinstance(value, str) for value in payload.values())
        or record.get("payload_sha256") != payload_sha256
        or (live_enabled == production_enabled)
        or payload["FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE"] not in ("0", "1")
        or (quality_enabled and not live_enabled)
        or payload["FR13_DRAFT_HEAD_M4_R64_U8_SO"]
        != "/tmp/fr13_bf16_k64_m4_r64_u8.abi3.so"
        or payload["FR13_DRAFT_HEAD_M4_R64_U8_TASK_IDS"] != task_ids
        or payload["FR13_DRAFT_HEAD_M4_R64_U8_LIVE_JSON"]
        != "/logs/fr13_dfwd_k64_m4_r64_u8.live.json"
        or payload["FR13_DRAFT_VOCAB_K"] != "65536"
        or payload["FR13_DRAFT_VOCAB_ROOT"] != "1"
        or payload["FR13_DRAFT_VOCAB_BLOCKS"]
        != "/workspace/scripts/fr13_dvk_subset_blocks.json"
        or (
            live_enabled
            and (
                payload["FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION_PASS_SIDECAR"]
                or production_sha
                or payload[
                    "FR13_DRAFT_HEAD_M4_R64_U8_INTERNAL_PRODUCTION_ATTESTED"
                ]
            )
        )
        or (
            production_enabled
            and (
                payload["FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE"] != "0"
                or payload[
                    "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION_PASS_SIDECAR"
                ]
                != "/logs/fr13_dfwd_k64_m4_r64_u8.production_credential.json"
                or len(production_sha) != 64
                or any(
                    value not in "0123456789abcdef"
                    for value in production_sha
                )
                or payload[
                    "FR13_DRAFT_HEAD_M4_R64_U8_INTERNAL_PRODUCTION_ATTESTED"
                ]
                != "1"
            )
        )
        or any(
            len(payload[key]) != 64
            or any(ch not in "0123456789abcdef" for ch in payload[key])
            for key in hash_keys
        )
        or len(payload["FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_COMMIT"]) != 40
        or any(
            ch not in "0123456789abcdef"
            for ch in payload["FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_COMMIT"]
        )
    ):
        raise RuntimeError("FR13 draft-head M4 U8 worker env payload drifted")
    conflicts = {
        key: (_os.environ[key], value)
        for key, value in payload.items()
        if key in _os.environ and _os.environ[key] != value
    }
    if conflicts:
        raise RuntimeError(
            "FR13 draft-head M4 U8 inherited worker env conflicts with sidecar: "
            + repr(conflicts)
        )
    _os.environ.update(payload)
    return {
        "schema": "fr13.fixed32.dfwd_k64_m4_r64_u8_worker_env_bridge.v1",
        "sidecar": str(path),
        "sidecar_sha256": _hashlib.sha256(raw).hexdigest(),
        "payload_sha256": payload_sha256,
        "hydrated_keys": list(_FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_KEYS),
    }


_FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_BRIDGE = (
    _fr13_draft_head_m4_u8_worker_env_bridge()
)
_FR13_FIXED32_OBSERVED_CURRENT = None
_FR13_FIXED32_CAPTURE_CONTEXT = None
_FR13_FIXED32_CAPTURE_MANIFESTS = {}
_FR13_FIXED32_CAPTURE_FROZEN = False
_FR13_FIXED32_GRAPH_REPLAY_EVIDENCE = []
_FR13_FIXED32_TAW_FULL_GRAPH_PASSES = set()
_FR13_FIXED32_PROFILE_CAPTURE_SCOPE = None
_FR13_FIXED32_PROFILE_MEMORY_SCOPE = False
_FR13_FIXED32_SFWD_CONV_POSTPREP_PROFILE_PRESEED = None
_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT = None
_FR13_FIXED32_DRAFTER_GRAPH_MANIFESTS = {}
_FR13_FIXED32_DRAFTER_GRAPH_BY_BATCH = {}
_FR13_FIXED32_DRAFTER_GRAPH_LIFECYCLE = {}
_FR13_DFWD_UNIFIED_BM8_PRODUCTION_PENDING = {}
_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT = None
_FR13_FIXED32_DRAFTER_REPLAY_EVIDENCE = []
_FR13_DRAFT_HEAD_M32_LIVE_STATE = None
_FR13_DRAFT_HEAD_U8_LIVE_STATE = None
_FR13_DRAFT_HEAD_M4_U8_LIVE_STATE = None
_FR13_FIXED32_ACCEPTED_OUTPUT_CURRENT = None
_FR13_FIXED32_BOOT_WARM_EVIDENCE = None
_FR13_FIXED32_TOPOLOGY_NEEDLE_EMITTED = False
try:
    _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
except NameError:
    _FR13_FIXED32_GDN_PATH_BV_CANDIDATE = None
try:
    _FR13_FIXED32_GDN_PATH_BV_PRODUCTION
except NameError:
    _FR13_FIXED32_GDN_PATH_BV_PRODUCTION = None
try:
    _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH
except NameError:
    _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH = None
try:
    _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION
except NameError:
    _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION = False
try:
    _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH
except NameError:
    _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH = None
try:
    _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION
except NameError:
    _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION = False
try:
    _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH
except NameError:
    _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH = None
try:
    _FR13_FIXED32_TAW_NATIVE_PRECOMPUTE
except NameError:
    _FR13_FIXED32_TAW_NATIVE_PRECOMPUTE = False
try:
    _FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB
except NameError:
    _FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB = False
try:
    _FR13_FIXED32_EAGER_KERNEL_DIAGNOSTIC
except NameError:
    _FR13_FIXED32_EAGER_KERNEL_DIAGNOSTIC = False
try:
    # _fr13_fixed32_validate_forward_work branches on this. Production always
    # gets it from _fr13_fixed32_runtime_bindings, but a bare exec of this
    # source (the patcher self-test) otherwise dies with a NameError.
    _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION
except NameError:
    _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION = False
_FR13_FIXED32_DRAFTER_TREE_LAYER = "mtp.layers.0.self_attn.attn"
_FR13_FIXED32_TARGET_TREE_LAYERS = frozenset(
    "language_model.model.layers.%d.self_attn.attn" % layer
    for layer in range(3, 64, 4)
)
# THE WALK-DERIVED-PIN CLASS, sixth member, and the first one found outside
# fr13_device_multidraft_kernel.py. Round 22's fourth boot died here:
#
#   RuntimeError: FR13 fixed32 GDN schedule work drift:
#     {'path_counts': (1, 11), 'max_lengths': (5, 11), 'launches': 2,
#      'programs': 12, 'padded_slots': 126, 'critical': 16,
#      'export_or_mask': 16915}
#
# Three of the seven fields vary by profile and were all pinned at hydra27's:
# max_lengths[1] is the tail spine depth (7 against tail10's 11), padded_slots
# is sum(paths * length) so 1*5 + 11*7 = 82 against 1*5 + 11*11 = 126, and
# critical is the walk cap, 12 against 16. MEASURED against the topology
# authority and against the corpse, which observed exactly (5, 11) / 126 / 16.
# The other four -- path_counts, launches, programs, export_or_mask -- are
# equal under both profiles and are carried per profile anyway, so a third
# profile cannot inherit a value nobody checked.
#
# WHY A MIRROR AND NOT A DERIVATION. This whole region is the string
# _FR13_FIXED32_OBSERVED_RUNTIME_SOURCE, planted verbatim into the container's
# gdn_linear_attn.py, so it CANNOT import fr13_fixed32_topology or the serving
# package. fr10_gdn_tree_kernel._FR13_FIXED32_SCHEDULE_BY_PROFILE has been
# profile-keyed since round 18 and carries these exact seven fields; this table
# is its mirror, and tests/test_fr14_gdn_schedule_contract_parity.py is what
# keeps the two from drifting.
_FR13_FIXED32_GDN_SCHEDULE_BY_PROFILE = {
    "hydra27_fixed32": {
        "path_counts": (1, 11),
        "max_lengths": (5, 7),
        "launches": 2,
        "programs": 12,
        "padded_slots": 82,
        "critical": 12,
        "export_or_mask": 16915,
    },
    "hydra31_fixed32": {
        "path_counts": (1, 11),
        "max_lengths": (5, 11),
        "launches": 2,
        "programs": 12,
        "padded_slots": 126,
        "critical": 16,
        "export_or_mask": 16915,
    },
}
# NINTH member, found by the census BEFORE boot seven rather than by boot seven.
# _fr14_main was `8 if gated else 6` -- hydra27's Arctic main-tail lengths. The
# authority states 10/12 for hydra31, and rescue_carry_slots 4 against 0. Same
# mirror, same lint.
_FR13_FIXED32_ARCTIC_TAIL_BY_PROFILE = {
    "hydra27_fixed32": {
        "main_tail_length": 6,
        "gated_main_tail_length": 8,
        "arctic_requested_tokens": 12,
        "gated_arctic_requested_tokens": 14,
        "rescue_carry_slots": 4,
        # MEASURED by boot eleven, then derived: sum of the profile's
        # branch-chain lengths, ((1, 4), (2, 6)) -> 10.
        "rescue_path_columns": 10,
    },
    "hydra31_fixed32": {
        "main_tail_length": 10,
        "gated_main_tail_length": 12,
        "arctic_requested_tokens": 16,
        "gated_arctic_requested_tokens": 18,
        "rescue_carry_slots": 0,
        # tail10 shortens rank 2 from 6 to 2 -- exactly the four columns it
        # respends as spine -- so ((1, 4), (2, 2)) -> 6. This is the number the
        # engine served in boot eleven's corpse.
        "rescue_path_columns": 6,
    },
}
_FR13_FIXED32_GDN_TREE_PROFILE_BY_MODE = {
    # An unset mode is the non-fixed32 route, which has always been hydra27's.
    "": "hydra27_fixed32",
    "tail6_fixed32": "hydra27_fixed32",
    "hydra27_fixed32": "hydra27_fixed32",
    "hydra31_fixed32": "hydra31_fixed32",
}
_FR13_FIXED32_GDN_MODE = (
    __import__("os").environ.get("FR13_FIXED32_MODE", "").strip()
)
if _FR13_FIXED32_GDN_MODE not in _FR13_FIXED32_GDN_TREE_PROFILE_BY_MODE:
    # Guessing a schedule depth is how the silent sites get fed.
    raise RuntimeError(
        "FR13 fixed32 GDN schedule has no profile for mode "
        + repr(_FR13_FIXED32_GDN_MODE)
    )
_FR13_FIXED32_GDN_SCHEDULE_EXPECTED = _FR13_FIXED32_GDN_SCHEDULE_BY_PROFILE[
    _FR13_FIXED32_GDN_TREE_PROFILE_BY_MODE[_FR13_FIXED32_GDN_MODE]
]
def _fr13_fixed32_drift_detail(observed, expected):
    """Name what DIFFERS, both sides, labelled which is which.

    Three refusals in this blob have now cost a boot each to read: two dumped a
    whole ~40-field dict, one printed the observed value and left the
    expectation to be found in the source. A tuple of two dicts is not
    two-sided if the reader cannot tell which is which.
    """
    if isinstance(observed, dict) and isinstance(expected, dict):
        names = sorted(
            name
            for name in set(observed) | set(expected)
            if observed.get(name) != expected.get(name)
        )
        parts = []
        for name in names:
            seen = observed.get(name)
            want = expected.get(name)
            # RECURSE. A nested dict printed whole is the boot-six complaint
            # again one level down: the generation-1 flush audit differed in
            # three gdn fields and printed the entire gdn section for both
            # sides.
            if isinstance(seen, dict) and isinstance(want, dict):
                parts.append(
                    str(name)
                    + "{"
                    + _fr13_fixed32_drift_detail(seen, want)
                    + "}"
                )
            else:
                parts.append(
                    str(name)
                    + ": observed "
                    + repr(seen)
                    + " against audited "
                    + repr(want)
                )
        return "; ".join(parts)
    if (
        isinstance(observed, (tuple, list))
        and isinstance(expected, (tuple, list))
        and len(observed) == len(expected)
    ):
        return "; ".join(
            "["
            + str(index)
            + "]: observed "
            + repr(seen)
            + " against audited "
            + repr(want)
            for index, (seen, want) in enumerate(zip(observed, expected))
            if seen != want
        )
    return "observed " + repr(observed) + " against audited " + repr(expected)


_FR13_FIXED32_ARCTIC_TAIL_EXPECTED = _FR13_FIXED32_ARCTIC_TAIL_BY_PROFILE[
    _FR13_FIXED32_GDN_TREE_PROFILE_BY_MODE[_FR13_FIXED32_GDN_MODE]
]


def _fr13_draft_head_m32_live_register(
    compares, mismatches, geometry, candidate
):
    global _FR13_DRAFT_HEAD_M32_LIVE_STATE
    _os = __import__("os")
    if _os.environ.get("FR13_DRAFT_HEAD_M32_LIVE_AB", "0") != "1":
        raise RuntimeError("FR13 draft-head M32 live state registered while off")
    if (
        _FR13_DRAFT_HEAD_M32_LIVE_STATE is not None
        or tuple(compares.shape) != (3,)
        or tuple(mismatches.shape) != (3,)
        or str(compares.dtype) != "torch.int64"
        or str(mismatches.dtype) != "torch.int64"
        or compares.device.type != "cuda"
        or mismatches.device != compares.device
        or not isinstance(geometry, dict)
        or not isinstance(candidate, dict)
    ):
        raise RuntimeError("FR13 draft-head M32 live registration drifted")
    _FR13_DRAFT_HEAD_M32_LIVE_STATE = {
        "compares": compares,
        "mismatches": mismatches,
        "geometry": geometry,
        "candidate": candidate,
        "source_commit": _os.environ.get(
            "FR13_DRAFT_HEAD_M32_SOURCE_COMMIT", ""
        ),
        "candidate_source_sha256": _os.environ.get(
            "FR13_DRAFT_HEAD_M32_QUALIFIED_SOURCE_SHA256", ""
        ),
        "instance_id": _os.environ.get(
            "FR13_DRAFT_HEAD_M32_INSTANCE_ID", ""
        ),
    }


def _fr13_draft_head_m32_live_finalize(events, flush_binding):
    _os = __import__("os")
    if _os.environ.get("FR13_DRAFT_HEAD_M32_LIVE_AB", "0") != "1":
        if _FR13_DRAFT_HEAD_M32_LIVE_STATE is not None:
            raise RuntimeError("FR13 draft-head M32 live state leaked while off")
        return
    state = _FR13_DRAFT_HEAD_M32_LIVE_STATE
    event_rows = list(events)
    event_count = len(event_rows)
    draft_events = sum(int(row.get("batch_size", -1)) for row in event_rows)
    hex_chars = frozenset("0123456789abcdef")
    if (
        not isinstance(state, dict)
        or not isinstance(flush_binding, dict)
        or set(flush_binding) != {
            "action",
            "boundary_snapshot_sha256",
            "complete_work_census_events",
            "events_sha256",
            "generation",
            "nonce",
            "producer_pid",
        }
        or flush_binding.get("action") != "final"
        or type(flush_binding.get("generation")) is not int
        or int(flush_binding["generation"]) < 1
        or type(flush_binding.get("producer_pid")) is not int
        or int(flush_binding["producer_pid"]) < 1
        or any(
            not isinstance(flush_binding.get(key), str)
            or len(flush_binding[key]) != 64
            or any(value not in hex_chars for value in flush_binding[key])
            for key in (
                "nonce",
                "events_sha256",
                "boundary_snapshot_sha256",
            )
        )
        or event_count < 1
        or draft_events != event_count
        or any(int(row.get("batch_size", -1)) != 1 for row in event_rows)
        or int(flush_binding.get("complete_work_census_events", -1))
        != event_count
    ):
        raise RuntimeError("FR13 draft-head M32 live finalization drifted")
    compares = tuple(int(value) for value in state["compares"].tolist())
    mismatches = tuple(
        int(value) for value in state["mismatches"].tolist()
    )
    total = compares[0]
    bad = mismatches[0]
    exact = (
        compares[1:] == (0, 0)
        and mismatches[1:] == (0, 0)
        and total == draft_events * 5
        and bad == 0
    )
    instance_id = state["instance_id"]
    record = {
        "schema": "fr13.fixed32.draft_head_m32_live_ab.v1",
        "status": "PASS" if exact else "FAIL",
        "suite": "SWE-Verified",
        "instance_id": instance_id,
        "task_marker": "swe_verified:" + instance_id,
        "concurrency": 1,
        "batch_size": 1,
        "source_commit": state["source_commit"],
        "candidate_source_sha256": state["candidate_source_sha256"],
        "geometry": state["geometry"],
        "candidate": state["candidate"],
        "completed_events": draft_events,
        "complete_work_census_events": event_count,
        "work_census_last_event_index": event_count - 1,
        "events_sha256": flush_binding["events_sha256"],
        "flush_generation": flush_binding["generation"],
        "flush_nonce": flush_binding["nonce"],
        "producer_pid": flush_binding["producer_pid"],
        "boundary_snapshot_sha256": flush_binding[
            "boundary_snapshot_sha256"
        ],
        "full_logit_comparisons": total,
        "raw_bf16_mismatches": bad,
        "served_return": "reference BF16 logits unchanged",
        "performance_measurement": False,
        "finalized_by_fixed32_flush": True,
        "flush_action": "final",
    }
    path = __import__("pathlib").Path(
        _os.environ.get(
            "FR13_DRAFT_HEAD_M32_LIVE_JSON",
            "/logs/fr13_draft_head_m32.live.json",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(_os.getpid()))
    with open(temporary, "w", encoding="ascii") as handle:
        handle.write(
            __import__("json").dumps(
                record,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        _os.fsync(handle.fileno())
    _os.replace(temporary, path)
    if not exact:
        raise RuntimeError(
            "FR13 draft-head M32 final comparison/event census mismatch: "
            + repr((compares, mismatches, event_count, draft_events))
        )


def _fr13_draft_head_u8_live_register(
    compares, mismatches, nonfinite, geometry, candidate, identities
):
    global _FR13_DRAFT_HEAD_U8_LIVE_STATE
    _os = __import__("os")
    if _os.environ.get("FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB", "0") != "1":
        raise RuntimeError("FR13 draft-head U8 live state registered while off")
    if (
        _FR13_DRAFT_HEAD_U8_LIVE_STATE is not None
        or tuple(compares.shape) != (5,)
        or tuple(mismatches.shape) != (5,)
        or tuple(nonfinite.shape) != (5,)
        or str(compares.dtype) != "torch.int64"
        or str(mismatches.dtype) != "torch.int64"
        or str(nonfinite.dtype) != "torch.int64"
        or compares.device.type != "cuda"
        or mismatches.device != compares.device
        or nonfinite.device != compares.device
        or not isinstance(geometry, dict)
        or not isinstance(candidate, dict)
        or not isinstance(identities, dict)
        or not isinstance(_FR13_DRAFT_HEAD_U8_WORKER_ENV_BRIDGE, dict)
        or _FR13_DRAFT_HEAD_U8_WORKER_ENV_BRIDGE.get("schema")
        != "fr13.fixed32.dfwd_k64_m1_r64_u8_worker_env_bridge.v1"
        or _FR13_DRAFT_HEAD_U8_WORKER_ENV_BRIDGE.get("sidecar")
        != "/logs/fr13_draft_head_m1_r64_u8.worker_env.json"
        or _FR13_DRAFT_HEAD_U8_WORKER_ENV_BRIDGE.get("hydrated_keys")
        != list(_FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS)
    ):
        raise RuntimeError("FR13 draft-head U8 live registration drifted")
    _FR13_DRAFT_HEAD_U8_LIVE_STATE = {
        "compares": compares,
        "mismatches": mismatches,
        "nonfinite": nonfinite,
        "geometry": geometry,
        "candidate": candidate,
        "identities": identities,
        "worker_env_bridge": _FR13_DRAFT_HEAD_U8_WORKER_ENV_BRIDGE,
        "root_forward_steps": [],
        "captured_depths": [],
    }


def _fr13_draft_head_u8_live_depth(batch_size):
    """Bind each shadow call to root or an actual MTP-forward depth."""
    state = _FR13_DRAFT_HEAD_U8_LIVE_STATE
    proposal = _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    context = _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT
    batch = int(batch_size)
    if (
        not isinstance(state, dict)
        or not isinstance(proposal, dict)
        or batch != 1
        or int(proposal.get("batch_size", -1)) != 1
        or proposal.get("mode") != _FR13_FIXED32_MODE
        or proposal.get("measured") not in (True, False)
    ):
        raise RuntimeError("FR13 draft-head U8 has no exact B1 proposal")
    if context is None:
        if (
            proposal.get("mtp_execution_basis") != "unbound"
            or int(proposal.get("mtp_forward_calls", -1)) != 0
            or int(proposal.get("mtp_forward_rows", -1)) != 0
        ):
            raise RuntimeError("FR13 draft-head U8 root lifecycle drifted")
        if proposal["measured"] is True:
            step = int(proposal.get("forward_step_index", -1))
            expected_step = len(state["root_forward_steps"])
            if step != expected_step:
                raise RuntimeError(
                    "FR13 draft-head U8 measured root/event sequence drifted: "
                    + repr((step, expected_step))
                )
            state["root_forward_steps"].append(step)
        return 0
    if not isinstance(context, dict):
        raise RuntimeError("FR13 draft-head U8 capture context is malformed")
    calls = int(context.get("draft_head_u8_calls", -1))
    rows = int(context.get("draft_head_u8_rows", -1))
    depth = int(context.get("mtp_forward_calls", -1))
    if (
        context.get("capturing") is not True
        or int(context.get("batch_size", -1)) != 1
        or context.get("mode") != _FR13_FIXED32_MODE
        or calls < 0
        or rows < 0
        or depth != calls + 1
        or int(context.get("mtp_forward_rows", -1)) != rows + 1
        or depth not in (1, 2, 3, 4)
        or state["captured_depths"] != list(range(1, depth))
    ):
        raise RuntimeError(
            "FR13 draft-head U8 MTP depth lifecycle drifted: "
            + repr((depth, calls, rows, context))
        )
    context["draft_head_u8_calls"] = calls + 1
    context["draft_head_u8_rows"] = rows + 1
    state["captured_depths"].append(depth)
    return depth


def _fr13_draft_head_u8_live_finalize(events, flush_binding):
    _os = __import__("os")
    enabled = _os.environ.get("FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB", "0")
    quality_enabled = (
        _os.environ.get("FR13_DRAFT_HEAD_M1_R64_U8_QUALITY_GATE", "0") == "1"
    )
    taw_quality_enabled = (
        _os.environ.get(
            "FR13_DRAFT_HEAD_M1_R64_U8_TAW_QUALITY_GATE", "0"
        )
        == "1"
    )
    if enabled != "1":
        if _FR13_DRAFT_HEAD_U8_LIVE_STATE is not None:
            raise RuntimeError("FR13 draft-head U8 live state leaked while off")
        return
    state = _FR13_DRAFT_HEAD_U8_LIVE_STATE
    event_rows = list(events)
    event_count = len(event_rows)
    expected_steps = [
        int(row.get("forward_step_index", -1)) for row in event_rows
    ]
    hex_chars = frozenset("0123456789abcdef")
    if (
        not isinstance(state, dict)
        or not isinstance(flush_binding, dict)
        or set(flush_binding) != {
            "action",
            "boundary_snapshot_sha256",
            "complete_work_census_events",
            "events_sha256",
            "generation",
            "nonce",
            "producer_pid",
        }
        or flush_binding.get("action") != "final"
        or type(flush_binding.get("generation")) is not int
        or int(flush_binding["generation"]) < 1
        or type(flush_binding.get("producer_pid")) is not int
        or int(flush_binding["producer_pid"]) < 1
        or any(
            not isinstance(flush_binding.get(key), str)
            or len(flush_binding[key]) != 64
            or any(value not in hex_chars for value in flush_binding[key])
            for key in (
                "nonce",
                "events_sha256",
                "boundary_snapshot_sha256",
            )
        )
        or event_count < 1
        or int(flush_binding.get("complete_work_census_events", -1))
        != event_count
        or any(int(row.get("batch_size", -1)) != 1 for row in event_rows)
        or expected_steps != list(range(event_count))
    ):
        raise RuntimeError("FR13 draft-head U8 live finalization drifted")
    compares = tuple(int(value) for value in state["compares"].tolist())
    mismatches = tuple(int(value) for value in state["mismatches"].tolist())
    nonfinite = tuple(int(value) for value in state["nonfinite"].tolist())
    complete = (
        compares == (event_count,) * 5
        and state["captured_depths"] == [1, 2, 3, 4]
        and state["root_forward_steps"] == expected_steps
    )
    qualified = complete and (
        nonfinite == (0,) * 5 if quality_enabled else mismatches == (0,) * 5
    )
    labels = ("root", "mtp_depth_1", "mtp_depth_2", "mtp_depth_3", "mtp_depth_4")
    per_depth_compares = dict(zip(labels, compares))
    per_depth_mismatches = dict(zip(labels, mismatches))
    per_depth_nonfinite = dict(zip(labels, nonfinite))
    total_compares = sum(compares)
    identities = state["identities"]
    candidate = dict(state["candidate"])
    candidate["shadow_only"] = not quality_enabled
    taw_exact_acceptance = None
    if taw_quality_enabled:
        taw_module = __import__("sys").modules.get(
            "_fr13_device_multidraft_kernel"
        )
        if taw_module is None:
            raise RuntimeError(
                "FR13 draft-head U8 TAW quality lost its device module"
            )
        taw_exact_acceptance = (
            taw_module.fr13_fixed32_taw_candidate_acceptance_census(
                mode="hydra27_fixed32",
                batch_size=1,
                completed_events=event_count,
                events_sha256=flush_binding["events_sha256"],
                candidate_binding={
                    "operation": candidate["operation"],
                    "candidate_so_sha256": identities[
                        "candidate_so_sha256"
                    ],
                    "candidate_source_sha256": identities[
                        "candidate_source_sha256"
                    ],
                    "task_ids": [identities["instance_id"]],
                },
            )
        )
    record = {
        "schema": (
            "fr13.fixed32.dfwd_k64_m1_r64_u8_quality.v2"
            if quality_enabled
            else "fr13.fixed32.dfwd_k64_m1_r64_u8_shadow.v1"
        ),
        "status": "PASS" if qualified else "FAIL",
        "suite": "SWE-Verified",
        "instance_id": identities["instance_id"],
        "task_marker": "swe_verified:" + identities["instance_id"],
        "concurrency": 1,
        "batch_size": 1,
        "source_commit": identities["source_commit"],
        "identities": identities,
        "worker_env_bridge": state["worker_env_bridge"],
        "topology": {
            "mode": "hydra27_fixed32",
            "batch_size": 1,
            "physical_rows": 32,
            "logical_drafts": 27,
            "draft_vocab_k": 65536,
            "draft_vocab_root": 1,
            "execution_basis": "FULL_AND_PIECEWISE_graph_replay",
        },
        "geometry": state["geometry"],
        "candidate": candidate,
        "completed_events": event_count,
        "complete_work_census_events": event_count,
        "work_census_last_event_index": event_count - 1,
        "events_sha256": flush_binding["events_sha256"],
        "flush_generation": flush_binding["generation"],
        "flush_nonce": flush_binding["nonce"],
        "producer_pid": flush_binding["producer_pid"],
        "boundary_snapshot_sha256": flush_binding[
            "boundary_snapshot_sha256"
        ],
        "root_forward_steps": state["root_forward_steps"],
        "captured_mtp_depths": state["captured_depths"],
        "per_depth_full_logit_comparisons": per_depth_compares,
        "per_depth_raw_bf16_mismatches": per_depth_mismatches,
        "per_depth_nonfinite_logits": per_depth_nonfinite,
        "comparison_scope": (
            "all 65536 logits in the fixed K64/root1 draft head; "
            "not the full model vocabulary"
        ),
        "full_logit_comparisons": total_compares,
        "compared_elements": total_compares * 65536,
        "compared_bytes": total_compares * 65536 * 2,
        "raw_bf16_mismatches": sum(mismatches),
        "nonfinite_logits": sum(nonfinite),
        "qualification_policy": (
            "lossless_deterministic_proposal_v1"
            if quality_enabled
            else "raw_bf16_shadow_v1"
        ),
        "proposal_distribution": {
            "candidate_logits_consumed": quality_enabled,
            "draft_probs": None,
            "proposal_token_selector": "argmax_topk",
            "q_mix_definition": "target_overlap_normalized_over_draft_token_ids",
            "rejection_sampler": "fr13_fixed32_deterministic_multidraft",
        },
        "taw_exact_acceptance": taw_exact_acceptance,
        "reference_always_served": not quality_enabled,
        "candidate_returned": quality_enabled,
        "served_return": (
            "candidate BF16 logits"
            if quality_enabled
            else "incumbent BF16 logits object unchanged"
        ),
        "performance_measurement": False,
        "timing_eligible": False,
        "finalized_by_fixed32_flush": True,
        "flush_action": "final",
    }
    path = __import__("pathlib").Path(
        _os.environ.get(
            "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_JSON",
            "/logs/fr13_dfwd_k64_m1_r64_u8.live.json",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(_os.getpid()))
    with open(temporary, "w", encoding="ascii") as handle:
        handle.write(
            __import__("json").dumps(
                record,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        _os.fsync(handle.fileno())
    _os.replace(temporary, path)
    if not qualified:
        raise RuntimeError(
            "FR13 draft-head U8 final depth/event qualification mismatch: "
            + repr(
                (
                    compares,
                    mismatches,
                    nonfinite,
                    state["captured_depths"],
                    state["root_forward_steps"],
                    expected_steps,
                )
            )
        )


def _fr13_draft_head_m4_u8_live_register(
    compares, mismatches, nonfinite, geometry, candidate, identities
):
    global _FR13_DRAFT_HEAD_M4_U8_LIVE_STATE
    _os = __import__("os")
    bridge = _FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_BRIDGE
    if (
        _os.environ.get("FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB", "0") != "1"
        or _FR13_DRAFT_HEAD_M4_U8_LIVE_STATE is not None
        or tuple(compares.shape) != (5,)
        or tuple(mismatches.shape) != (5,)
        or tuple(nonfinite.shape) != (5,)
        or str(compares.dtype) != "torch.int64"
        or str(mismatches.dtype) != "torch.int64"
        or str(nonfinite.dtype) != "torch.int64"
        or compares.device.type != "cuda"
        or mismatches.device != compares.device
        or nonfinite.device != compares.device
        or not isinstance(geometry, dict)
        or not isinstance(candidate, dict)
        or not isinstance(identities, dict)
        or not isinstance(bridge, dict)
        or bridge.get("schema")
        != "fr13.fixed32.dfwd_k64_m4_r64_u8_worker_env_bridge.v1"
        or bridge.get("sidecar")
        != "/logs/fr13_draft_head_m4_r64_u8.worker_env.json"
        or bridge.get("hydrated_keys")
        != list(_FR13_DRAFT_HEAD_M4_U8_WORKER_ENV_KEYS)
    ):
        raise RuntimeError("FR13 draft-head M4 U8 live registration drifted")
    _FR13_DRAFT_HEAD_M4_U8_LIVE_STATE = {
        "compares": compares,
        "mismatches": mismatches,
        "nonfinite": nonfinite,
        "geometry": geometry,
        "candidate": candidate,
        "identities": identities,
        "worker_env_bridge": bridge,
        "root_forward_steps": [],
        "captured_depths": [],
    }


def _fr13_draft_head_m4_u8_live_depth(batch_size):
    """Bind each B4 shadow call to root or one actual MTP depth."""
    state = _FR13_DRAFT_HEAD_M4_U8_LIVE_STATE
    proposal = _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    context = _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT
    batch = int(batch_size)
    if (
        not isinstance(state, dict)
        or not isinstance(proposal, dict)
        or batch != 4
        or int(proposal.get("batch_size", -1)) != 4
        or proposal.get("mode") != _FR13_FIXED32_MODE
        or proposal.get("measured") not in (True, False)
    ):
        raise RuntimeError("FR13 draft-head M4 U8 has no exact B4 proposal")
    if context is None:
        if (
            proposal.get("mtp_execution_basis") != "unbound"
            or int(proposal.get("mtp_forward_calls", -1)) != 0
            or int(proposal.get("mtp_forward_rows", -1)) != 0
        ):
            raise RuntimeError("FR13 draft-head M4 U8 root lifecycle drifted")
        if proposal["measured"] is True:
            step = int(proposal.get("forward_step_index", -1))
            expected_step = len(state["root_forward_steps"])
            if step != expected_step:
                raise RuntimeError(
                    "FR13 draft-head M4 U8 measured root/event sequence drifted: "
                    + repr((step, expected_step))
                )
            state["root_forward_steps"].append(step)
        return 0
    if not isinstance(context, dict):
        raise RuntimeError("FR13 draft-head M4 U8 capture context is malformed")
    calls = int(context.get("draft_head_u8_calls", -1))
    rows = int(context.get("draft_head_u8_rows", -1))
    depth = int(context.get("mtp_forward_calls", -1))
    if (
        context.get("capturing") is not True
        or int(context.get("batch_size", -1)) != 4
        or context.get("mode") != _FR13_FIXED32_MODE
        or calls < 0
        or rows < 0
        or depth != calls + 1
        or int(context.get("mtp_forward_rows", -1)) != rows + 4
        or depth not in (1, 2, 3, 4)
        or state["captured_depths"] != list(range(1, depth))
    ):
        raise RuntimeError(
            "FR13 draft-head M4 U8 MTP depth lifecycle drifted: "
            + repr((depth, calls, rows, context))
        )
    context["draft_head_u8_calls"] = calls + 1
    context["draft_head_u8_rows"] = rows + 4
    state["captured_depths"].append(depth)
    return depth


def _fr13_draft_head_m4_u8_live_finalize(events, flush_binding):
    _os = __import__("os")
    enabled = _os.environ.get("FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB", "0")
    quality_enabled = (
        _os.environ.get("FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE", "0") == "1"
    )
    if enabled != "1":
        if _FR13_DRAFT_HEAD_M4_U8_LIVE_STATE is not None:
            raise RuntimeError("FR13 draft-head M4 U8 live state leaked while off")
        return
    state = _FR13_DRAFT_HEAD_M4_U8_LIVE_STATE
    event_rows = list(events)
    event_count = len(event_rows)
    expected_steps = [
        int(row.get("forward_step_index", -1)) for row in event_rows
    ]
    hex_chars = frozenset("0123456789abcdef")
    if (
        not isinstance(state, dict)
        or not isinstance(flush_binding, dict)
        or set(flush_binding) != {
            "action", "boundary_snapshot_sha256",
            "complete_work_census_events", "events_sha256", "generation",
            "nonce", "producer_pid",
        }
        or flush_binding.get("action") != "final"
        or type(flush_binding.get("generation")) is not int
        or int(flush_binding["generation"]) < 1
        or type(flush_binding.get("producer_pid")) is not int
        or int(flush_binding["producer_pid"]) < 1
        or any(
            not isinstance(flush_binding.get(key), str)
            or len(flush_binding[key]) != 64
            or any(value not in hex_chars for value in flush_binding[key])
            for key in ("nonce", "events_sha256", "boundary_snapshot_sha256")
        )
        or event_count < 1
        or int(flush_binding.get("complete_work_census_events", -1))
        != event_count
        or any(int(row.get("batch_size", -1)) != 4 for row in event_rows)
        or expected_steps != list(range(event_count))
    ):
        raise RuntimeError("FR13 draft-head M4 U8 live finalization drifted")
    compares = tuple(int(value) for value in state["compares"].tolist())
    mismatches = tuple(int(value) for value in state["mismatches"].tolist())
    nonfinite = tuple(int(value) for value in state["nonfinite"].tolist())
    complete = (
        compares == (event_count,) * 5
        and state["captured_depths"] == [1, 2, 3, 4]
        and state["root_forward_steps"] == expected_steps
    )
    qualified = complete and (
        nonfinite == (0,) * 5 if quality_enabled else mismatches == (0,) * 5
    )
    labels = ("root", "mtp_depth_1", "mtp_depth_2", "mtp_depth_3", "mtp_depth_4")
    total_compares = sum(compares)
    identities = state["identities"]
    candidate = dict(state["candidate"])
    candidate["shadow_only"] = not quality_enabled
    taw_exact_acceptance = None
    if quality_enabled:
        taw_module = __import__("sys").modules.get(
            "_fr13_device_multidraft_kernel"
        )
        if taw_module is None:
            raise RuntimeError(
                "FR13 draft-head M4 U8 quality lost its TAW module"
            )
        taw_exact_acceptance = (
            taw_module.fr13_fixed32_taw_candidate_acceptance_census(
                mode="hydra27_fixed32",
                batch_size=4,
                completed_events=event_count,
                events_sha256=flush_binding["events_sha256"],
                candidate_binding={
                    "operation": candidate["operation"],
                    "candidate_so_sha256": identities[
                        "candidate_so_sha256"
                    ],
                    "candidate_source_sha256": identities[
                        "candidate_source_sha256"
                    ],
                    "task_ids": list(identities["task_ids"]),
                },
            )
        )
    record = {
        "schema": (
            "fr13.fixed32.dfwd_k64_m4_r64_u8_quality.v2"
            if quality_enabled
            else "fr13.fixed32.dfwd_k64_m4_r64_u8_shadow.v1"
        ),
        "status": "PASS" if qualified else "FAIL",
        "suite": "SWE-Verified",
        "task_ids": identities["task_ids"],
        "task_markers": ["swe_verified:" + value for value in identities["task_ids"]],
        "concurrency": 4,
        "batch_size": 4,
        "source_commit": identities["source_commit"],
        "identities": identities,
        "worker_env_bridge": state["worker_env_bridge"],
        "topology": {
            "mode": "hydra27_fixed32",
            "batch_size": 4,
            "physical_rows_per_request": 32,
            "total_physical_rows": 128,
            "logical_drafts_per_request": 27,
            "draft_vocab_k": 65536,
            "draft_vocab_root": 1,
            "execution_basis": "FULL_AND_PIECEWISE_graph_replay",
        },
        "geometry": state["geometry"],
        "candidate": candidate,
        "completed_events": event_count,
        "complete_work_census_events": event_count,
        "work_census_last_event_index": event_count - 1,
        "events_sha256": flush_binding["events_sha256"],
        "flush_generation": flush_binding["generation"],
        "flush_nonce": flush_binding["nonce"],
        "producer_pid": flush_binding["producer_pid"],
        "boundary_snapshot_sha256": flush_binding["boundary_snapshot_sha256"],
        "root_forward_steps": state["root_forward_steps"],
        "captured_mtp_depths": state["captured_depths"],
        "per_depth_full_logit_comparisons": dict(zip(labels, compares)),
        "per_depth_raw_bf16_mismatches": dict(zip(labels, mismatches)),
        "per_depth_nonfinite_logits": dict(zip(labels, nonfinite)),
        "comparison_scope": (
            "all four rows and all 65536 logits at each of five fixed "
            "K64/root1 draft-head sites"
        ),
        "full_logit_comparisons": total_compares,
        "compared_elements": total_compares * 4 * 65536,
        "compared_bytes": total_compares * 4 * 65536 * 2,
        "raw_bf16_mismatches": sum(mismatches),
        "nonfinite_logits": sum(nonfinite),
        "qualification_policy": (
            "lossless_deterministic_proposal_taw_exact_v1"
            if quality_enabled
            else "raw_bf16_shadow_v1"
        ),
        "proposal_distribution": {
            "candidate_logits_consumed": quality_enabled,
            "draft_probs": None,
            "proposal_token_selector": "argmax_topk",
            "q_mix_definition": "target_overlap_normalized_over_draft_token_ids",
            "rejection_sampler": "fr13_fixed32_deterministic_multidraft",
        },
        "taw_exact_acceptance": taw_exact_acceptance,
        "reference_always_served": not quality_enabled,
        "candidate_returned": quality_enabled,
        "served_return": (
            "candidate BF16 logits"
            if quality_enabled
            else "incumbent BF16 logits object unchanged"
        ),
        "performance_measurement": False,
        "timing_eligible": False,
        "finalized_by_fixed32_flush": True,
        "flush_action": "final",
    }
    path = __import__("pathlib").Path(
        _os.environ.get(
            "FR13_DRAFT_HEAD_M4_R64_U8_LIVE_JSON",
            "/logs/fr13_dfwd_k64_m4_r64_u8.live.json",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(_os.getpid()))
    with open(temporary, "w", encoding="ascii") as handle:
        handle.write(
            __import__("json").dumps(
                record, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            ) + "\n"
        )
        handle.flush()
        _os.fsync(handle.fileno())
    _os.replace(temporary, path)
    if not qualified:
        raise RuntimeError(
            "FR13 draft-head M4 U8 final depth/event quality mismatch: "
            + repr((compares, mismatches, nonfinite, state["captured_depths"],
                    state["root_forward_steps"], expected_steps))
        )
def _fr13_fixed32_unmeasured_full_row_map_valid(batch_rows, compact_batch):
    """Admit equal full/compact rows only for isolated eager diagnostics."""
    return batch_rows > compact_batch or (
        _FR13_FIXED32_EAGER_KERNEL_DIAGNOSTIC
        and batch_rows == compact_batch
    )


def _fr13_fixed32_topology_needle():
    mask = int(_FR13_FIXED32_VALID_MASK)
    expected_masks = {
        "tail6_fixed32": 0x7A9CE7FF,
        "hydra27_fixed32": 0x7ABDFFFF,
        "hydra31_fixed32": 0x7FFFFFFF,
    }
    if expected_masks.get(_FR13_FIXED32_MODE) != mask:
        raise RuntimeError(
            "FR13 fixed32 topology needle identity drift: "
            + repr((_FR13_FIXED32_MODE, mask))
        )
    return (
        "[FR13_FIXED32] topology engaged: mode="
        + _FR13_FIXED32_MODE
        + " active_drafts="
        + str(mask.bit_count())
        + " valid_mask="
        + f"{mask:#010x}"
    )


def _fr13_fixed32_target_kv_layer_names(layer_names):
    # Hybrid grouping also includes MTP, whose KV uses the restored flat map.
    captured = tuple(layer_names)
    expected_group = _FR13_FIXED32_TARGET_TREE_LAYERS | {
        _FR13_FIXED32_DRAFTER_TREE_LAYER
    }
    if len(captured) != 17 or frozenset(captured) != expected_group:
        raise RuntimeError(
            "FR13 fixed32 KV16 full-attention layer ownership drift"
        )
    target_names = tuple(
        name
        for name in captured
        if name in _FR13_FIXED32_TARGET_TREE_LAYERS
    )
    if (
        len(target_names) != 16
        or frozenset(target_names) != _FR13_FIXED32_TARGET_TREE_LAYERS
    ):
        raise RuntimeError(
            "FR13 fixed32 KV16 target layer selection drift"
        )
    return target_names


_FR13_FIXED32_NONPURE_DISPATCH = {
    "guarded_steps": 0,
    "piecewise_steps": 0,
    "none_steps": 0,
    "forbidden_full_steps": 0,
}
_FR13_FIXED32_NONPURE_COMMIT_REPLAYS_BY_BATCH = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
}


def _fr13_fixed32_observed_new_state(
    mode,
    batch_size,
    forward_step_index,
    execution_basis,
    request_ids=(),
):
    return {
        "mode": mode,
        "batch_size": int(batch_size),
        "forward_step_index": int(forward_step_index),
        "request_ids": tuple(str(value) for value in request_ids),
        "batch_purity": None,
        "execution_basis": execution_basis,
        "forward_graph_id": None,
        "forward_graph_signature": None,
        "forward_graph_replays": 0,
        "tree_layers": set(),
        "tree_calls": 0,
        "tree_q_rows": 0,
        "tree_bias_shape": None,
        "gdn_layers": set(),
        "gdn_calls": set(),
        "gdn_scan_calls": 0,
        "gdn_launches": 0,
        "gdn_path_programs": 0,
        "gdn_padded_slots": 0,
        "gdn_nodes": 0,
        "gdn_critical_path": None,
        "gdn_grid_z": None,
        "gdn_max_path_lengths": None,
        "gdn_export_or_mask": None,
        "gdn_parent_sha256": None,
        "gdn_ancestry_sha256": None,
        "conv_consume_layers": set(),
        "conv_consume_calls": 0,
        "conv_consume_hits": 0,
        "conv_consume_fallbacks": 0,
        "conv_freshness_matches": 0,
        "conv_stage_calls": 0,
        "conv_stage_replays": 0,
        "conv_stage_before_all_consumes": False,
        "conv_stage_layer": None,
        "conv_stage_layers": 0,
        "conv_stage_row_elems": 0,
        "conv_stage_block": 0,
        "conv_stage_programs": 0,
        "conv_stage_ssi_pointer_entries": 0,
        "conv_stage_ssi_groups": 0,
        "conv_stage_source": None,
        "conv_stage_instance": None,
        "conv_source_layers": {},
        # The SFWD conv/post-prep fusion subsumes the pregather stage and the
        # per-layer consume into one kernel, so it reports its own work class
        # rather than incrementing the unfused conv counters.
        "sfwd_conv_postprep_layers": set(),
        "sfwd_conv_postprep_calls": 0,
        "conv_commit": None,
        "kernel_shape": None,
        "conv_pregather": None,
        "sfwd_conv_postprep": None,
        "committer": None,
        "preforward_pack": None,
        "output_publish": None,
        "accepted_path_pack": None,
        "request_key_pack": None,
        "drafter": None,
        "drafter_runtime": None,
        "gdn_comparator": None,
    }


def _fr13_fixed32_manifest_entry(entry, label):
    if not isinstance(entry, tuple) or len(entry) != 2:
        raise RuntimeError("FR13 fixed32 " + str(label) + " manifest is missing")
    signature, canonical = entry
    if (
        not isinstance(signature, str)
        or len(signature) != 64
        or not isinstance(canonical, str)
    ):
        raise RuntimeError(
            "FR13 fixed32 " + str(label) + " manifest has bad types"
        )
    actual_signature = __import__("hashlib").sha256(
        canonical.encode("ascii")
    ).hexdigest()
    if actual_signature != signature:
        raise RuntimeError(
            "FR13 fixed32 "
            + str(label)
            + " manifest signature/canonical mismatch"
        )
    return signature, canonical


def _fr13_fixed32_observed_current(stage):
    if (
        not _FR13_FIXED32_MODE
        or globals().get("_FR13_FIXED32_EAGER_KERNEL_DIAGNOSTIC", False)
    ):
        return None
    event = globals().get("_FR13_FIXED32_OBSERVED_CURRENT")
    if not isinstance(event, dict):
        raise RuntimeError(
            "FR13 fixed32 observed runtime has no open event at " + str(stage)
        )
    return event


def _fr13_fixed32_observed_event_active():
    if (
        not _FR13_FIXED32_MODE
        or globals().get("_FR13_FIXED32_EAGER_KERNEL_DIAGNOSTIC", False)
    ):
        return False
    event = globals().get("_FR13_FIXED32_OBSERVED_CURRENT")
    if event is None:
        return False
    if (
        not isinstance(event, dict)
        or event.get("mode") != _FR13_FIXED32_MODE
        or int(event.get("forward_step_index", -1)) < 0
        or int(globals().get("_FR13_FIXED32_CURRENT_FORWARD_STEP", -1))
        != int(event.get("forward_step_index", -1))
    ):
        raise RuntimeError("FR13 fixed32 active-event marker is incoherent")
    return True


def _fr13_fixed32_profile_capture_scope_begin(
    runtime_mode,
    num_tokens,
    num_reqs,
    uniform,
    has_lora,
    num_active_loras,
):
    global _FR13_FIXED32_PROFILE_CAPTURE_SCOPE
    if not _FR13_FIXED32_MODE:
        return
    if _FR13_FIXED32_PROFILE_CAPTURE_SCOPE is not None:
        raise RuntimeError("FR13 fixed32 profile capture scopes overlapped")
    if (
        _FR13_FIXED32_CAPTURE_FROZEN
        or _FR13_FIXED32_PROFILE_MEMORY_SCOPE is not True
        or _FR13_FIXED32_OBSERVED_CURRENT is not None
        or globals().get("_FR13_FIXED32_PENDING_EVENT") is not None
        or _FR13_FIXED32_CAPTURE_CONTEXT is not None
        or _FR13_FIXED32_CAPTURE_MANIFESTS
        or _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT is not None
        or _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT is not None
    ):
        raise RuntimeError(
            "FR13 fixed32 profile capture began outside pristine bootstrap"
        )
    descriptor = _fr13_fixed32_graph_descriptor(
        runtime_mode,
        num_tokens,
        num_reqs,
        uniform,
        has_lora,
        num_active_loras,
    )
    _FR13_FIXED32_PROFILE_CAPTURE_SCOPE = {
        "descriptor": descriptor,
        "graph_id": None,
        "completed": False,
    }


def _fr13_fixed32_profile_capture_scope_end():
    global _FR13_FIXED32_PROFILE_CAPTURE_SCOPE
    if not _FR13_FIXED32_MODE:
        return
    scope = _FR13_FIXED32_PROFILE_CAPTURE_SCOPE
    # Clear before validating so an exceptional/unbalanced profile attempt
    # cannot leave a stale bypass marker behind.
    _FR13_FIXED32_PROFILE_CAPTURE_SCOPE = None
    graph_id = scope.get("graph_id") if isinstance(scope, dict) else None
    completed = scope.get("completed") if isinstance(scope, dict) else None
    capture_state_closed = (
        graph_id is None and completed is False
    ) or (
        type(graph_id) is int and graph_id > 0 and completed is True
    )
    if (
        not isinstance(scope, dict)
        or set(scope) != {"descriptor", "graph_id", "completed"}
        or not isinstance(scope.get("descriptor"), dict)
        or not capture_state_closed
        or _FR13_FIXED32_CAPTURE_CONTEXT is not None
        or _FR13_FIXED32_CAPTURE_MANIFESTS
        or _FR13_FIXED32_OBSERVED_CURRENT is not None
        or globals().get("_FR13_FIXED32_PENDING_EVENT") is not None
        or _FR13_FIXED32_CAPTURE_FROZEN
        or _FR13_FIXED32_PROFILE_MEMORY_SCOPE is not True
        or _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT is not None
        or _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT is not None
    ):
        raise RuntimeError(
            "FR13 fixed32 profile capture scope did not close cleanly: "
            + repr(scope)
        )


def _fr13_fixed32_sfwd_conv_postprep_profile_capture_active():
    """Return whether SFWD is running in a coherent throwaway FULL graph."""
    scope = globals().get("_FR13_FIXED32_PROFILE_CAPTURE_SCOPE")
    if scope is None:
        return False
    graph_id = scope.get("graph_id") if isinstance(scope, dict) else None
    descriptor = scope.get("descriptor") if isinstance(scope, dict) else None
    if (
        not isinstance(scope, dict)
        or set(scope) != {"descriptor", "graph_id", "completed"}
        or not isinstance(descriptor, dict)
        or descriptor.get("runtime_mode") != "FULL"
        or type(descriptor.get("num_tokens")) is not int
        or type(descriptor.get("num_reqs")) is not int
        or descriptor["num_tokens"] != 32 * descriptor["num_reqs"]
        or descriptor["num_reqs"] not in (1, 2, 3, 4)
        or descriptor.get("uniform") is not True
        or descriptor.get("has_lora") is not False
        or type(descriptor.get("num_active_loras")) is not int
        or descriptor["num_active_loras"] != 0
        or (
            graph_id is not None
            and (type(graph_id) is not int or graph_id <= 0)
        )
        or scope.get("completed") is not False
        or _FR13_FIXED32_PROFILE_MEMORY_SCOPE is not True
        or _FR13_FIXED32_CAPTURE_CONTEXT is not None
        or _FR13_FIXED32_CAPTURE_MANIFESTS
        or _FR13_FIXED32_OBSERVED_CURRENT is not None
        or globals().get("_FR13_FIXED32_PENDING_EVENT") is not None
        or _FR13_FIXED32_CAPTURE_FROZEN
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep profile capture scope drifted"
        )
    return True


def _fr13_fixed32_clear_sfwd_conv_postprep_profile_capture():
    evidence = globals().get(
        "_FR13_FIXED32_SFWD_CONV_POSTPREP_PROFILE_PRESEED"
    )
    if evidence is None:
        return None
    layers = globals().get("_FR13_REPLAY_LAYERS")
    stacks = globals().get("_FR13_EAGER_PACK_STACKS")
    if not isinstance(layers, dict) or not isinstance(stacks, dict):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep profile cleanup registries are missing"
        )
    order = tuple(str(value) for value in stacks.get("layer_order", ()))
    if (
        len(order) != 48
        or len(set(order)) != 48
        or any(name not in layers for name in order)
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep profile cleanup layer order drifted"
        )
    from lumo_flywheel_serving.fr13_sfwd_conv_postprep_fusion import (
        clear_fixed32_sfwd_conv_postprep_profile_capture_bindings as _clear,
    )

    cleared = _clear(layer_objects=tuple(layers[name] for name in order))
    if (
        not isinstance(cleared, dict)
        or int(cleared.get("cleared", -1)) != 48
        or int(cleared.get("layers", -1)) != 48
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep profile bindings did not clear"
        )
    globals()["_FR13_FIXED32_SFWD_CONV_POSTPREP_PROFILE_PRESEED"] = None
    return dict(cleared)


def _fr13_fixed32_profile_memory_scope_begin():
    global _FR13_FIXED32_PROFILE_MEMORY_SCOPE
    if not _FR13_FIXED32_MODE:
        return
    if (
        _FR13_FIXED32_PROFILE_MEMORY_SCOPE is not False
        or _FR13_FIXED32_PROFILE_CAPTURE_SCOPE is not None
        or _FR13_FIXED32_CAPTURE_CONTEXT is not None
        or _FR13_FIXED32_CAPTURE_MANIFESTS
        or _FR13_FIXED32_OBSERVED_CURRENT is not None
        or globals().get("_FR13_FIXED32_PENDING_EVENT") is not None
        or _FR13_FIXED32_CAPTURE_FROZEN
        or globals().get("_FR13_FIXED32_PRESEEDED_BATCHES")
        or _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT is not None
        or _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT is not None
    ):
        raise RuntimeError(
            "FR13 fixed32 profile-memory scope began outside pristine bootstrap"
        )
    _FR13_FIXED32_PROFILE_MEMORY_SCOPE = True


def _fr13_fixed32_profile_memory_scope_end():
    global _FR13_FIXED32_PROFILE_MEMORY_SCOPE
    if not _FR13_FIXED32_MODE:
        return
    _fr13_fixed32_clear_sfwd_conv_postprep_profile_capture()
    active = _FR13_FIXED32_PROFILE_MEMORY_SCOPE
    _FR13_FIXED32_PROFILE_MEMORY_SCOPE = False
    if (
        active is not True
        or _FR13_FIXED32_PROFILE_CAPTURE_SCOPE is not None
        or _FR13_FIXED32_CAPTURE_CONTEXT is not None
        or _FR13_FIXED32_CAPTURE_MANIFESTS
        or _FR13_FIXED32_OBSERVED_CURRENT is not None
        or globals().get("_FR13_FIXED32_PENDING_EVENT") is not None
        or _FR13_FIXED32_CAPTURE_FROZEN
        or globals().get("_FR13_FIXED32_PRESEEDED_BATCHES")
        or _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT is not None
        or _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT is not None
    ):
        raise RuntimeError(
            "FR13 fixed32 profile-memory scope did not close cleanly"
        )
    for layer in (
        globals().get("_FR13_REPLAY_LAYERS", {}) or {}
    ).values():
        layer._fr13_replay_ssm_state = None
        layer._fr13_replay_conv_state = None


def _fr13_fixed32_final_full_preseed_postcheck_required(runtime_mode):
    """Return whether this capture must prove the persistent preseed bundle."""
    if not _FR13_FIXED32_MODE:
        return False
    mode = str(runtime_mode).upper()
    if mode == "PIECEWISE":
        return False
    if mode != "FULL":
        raise RuntimeError(
            "FR13 fixed32 cudagraph preseed mode is invalid: "
            + _fr13_fixed32_drift_detail(mode, "FULL")
        )
    if _FR13_FIXED32_PROFILE_MEMORY_SCOPE is True:
        # Throwaway memory-profile graphs must not publish persistent cache
        # pointers, even though their FULL descriptor has capture geometry.
        return False
    if (
        _FR13_FIXED32_PROFILE_MEMORY_SCOPE is not False
        or _FR13_FIXED32_PROFILE_CAPTURE_SCOPE is not None
    ):
        raise RuntimeError(
            "FR13 fixed32 final preseed resolved inside invalid profile scope"
        )
    return True


def _fr13_fixed32_final_full_preseed_needed(
    runtime_mode,
    num_tokens,
    num_reqs,
    uniform,
    has_lora,
    num_active_loras,
):
    """Return whether final FULL capture still needs an actual-cache producer."""
    if not _fr13_fixed32_final_full_preseed_postcheck_required(runtime_mode):
        return False
    _fr13_fixed32_graph_descriptor(
        str(runtime_mode).upper(),
        num_tokens,
        num_reqs,
        uniform,
        has_lora,
        num_active_loras,
    )
    done = globals().get("_FR13_FIXED32_PRESEEDED_BATCHES")
    capacity = int(globals().get("_FR13_FIXED32_PRESEED_CAP", 0))
    if not done and capacity == 0:
        return True
    if (
        capacity not in (1, 2, 3, 4)
        or set(done or ()) != set(range(1, capacity + 1))
        or int(num_reqs) > capacity
    ):
        raise RuntimeError(
            "FR13 fixed32 final preseed lifecycle drifted: "
            + repr((done, capacity, num_reqs))
        )
    return False


def _fr13_fixed32_warm_device_postprocess_tail(
    taw_module,
    device,
    capacity,
    vocab_size,
):
    """Warm device copies, conv commit, committer replay, and flag clear."""
    slot_paths = globals().get("_LUMO_FA_FIXED32_SLOT_PATHS")
    slot_lens = globals().get("_LUMO_FA_FIXED32_SLOT_LENS")
    spec_paths = globals().get("_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR")
    spec_lens = globals().get("_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR")
    stacks = globals().get("_FR13_EAGER_PACK_STACKS")
    flags = None if not isinstance(stacks, dict) else stacks.get("flags")
    if (
        not torch.is_tensor(slot_paths)
        or not torch.is_tensor(slot_lens)
        or not torch.is_tensor(spec_paths)
        or not torch.is_tensor(spec_lens)
        or not torch.is_tensor(flags)
        or int(slot_paths.shape[0]) < capacity
        or int(slot_lens.shape[0]) < capacity
        or int(spec_paths.shape[0]) < capacity
        or int(spec_lens.shape[0]) < capacity
        or int(slot_paths.shape[1]) != 16
        or int(spec_paths.shape[1]) != 16
        or int(flags.ndim) != 2
        or int(flags.shape[1]) < 1
        or slot_paths.dtype != torch.int32
        or slot_lens.dtype != torch.int32
        or spec_paths.dtype != torch.int32
        or spec_lens.dtype != torch.int32
    ):
        raise RuntimeError(
            "FR13 fixed32 boot warm is missing persistent postprocess buffers"
        )
    saved_slot_paths = slot_paths[:capacity].clone()
    saved_slot_lens = slot_lens[:capacity].clone()
    saved_spec_paths = spec_paths[:capacity].clone()
    saved_spec_lens = spec_lens[:capacity].clone()
    saved_flags = flags.clone()
    batches = tuple(range(1, capacity + 1))
    output_copy_pairs = 0
    slot_copy_pairs = 0
    spec_copy_pairs = 0
    flags_zero_fills = 0
    committer = None
    try:
        for batch in batches:
            (
                device_output,
                _device_output_lens,
                device_paths,
                device_lens,
                device_last_rows,
            ) = taw_module.fr13_fixed32_taw_warm_products(
                device,
                mode=_FR13_FIXED32_MODE,
                valid_mask=int(_FR13_FIXED32_VALID_MASK),
                max_batch_size=capacity,
                vocab_size=int(vocab_size),
                batch_size=batch,
            )
            if (
                device_output.dtype != torch.int64
                or device_paths.dtype != torch.int64
                or device_lens.dtype != torch.int64
                or device_last_rows.dtype != torch.int64
            ):
                raise RuntimeError(
                    "FR13 fixed32 boot warm TAW product dtype drift"
                )
            output_destination = torch.empty(
                tuple(device_output.shape),
                dtype=torch.int32,
                device=device_output.device,
            )
            accepted_destination = torch.empty(
                tuple(device_last_rows.shape),
                dtype=torch.int32,
                device=device_last_rows.device,
            )
            output_destination.copy_(device_output)
            accepted_destination.copy_(device_last_rows)
            output_copy_pairs += 1
            for row in range(batch):
                slot_paths[row].copy_(device_paths[row])
                slot_lens[row].copy_(device_lens[row])
                slot_copy_pairs += 1
            spec_paths[:batch].copy_(device_paths)
            spec_lens[:batch].copy_(device_lens)
            spec_copy_pairs += 1
        from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
            warm_fixed32_committer_graphs_all_batches as _warm_committer,
        )
        committer = _warm_committer()
        flags[:, 0].zero_()
        flags_zero_fills += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        try:
            slot_paths[:capacity].copy_(saved_slot_paths)
            slot_lens[:capacity].copy_(saved_slot_lens)
            spec_paths[:capacity].copy_(saved_spec_paths)
            spec_lens[:capacity].copy_(saved_spec_lens)
            flags.copy_(saved_flags)
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    persistent_copy_state_restored = (
        torch.equal(slot_paths[:capacity], saved_slot_paths)
        and torch.equal(slot_lens[:capacity], saved_slot_lens)
        and torch.equal(spec_paths[:capacity], saved_spec_paths)
        and torch.equal(spec_lens[:capacity], saved_spec_lens)
    )
    flags_state_restored = torch.equal(flags, saved_flags)
    if not persistent_copy_state_restored or not flags_state_restored:
        raise RuntimeError(
            "FR13 fixed32 postprocess-tail warm did not restore buffers"
        )
    return {
        "ready": True,
        "classification": "unmeasured_boot",
        "hardware_scope": "device_postprocess_kernels",
        "wrapper_bookkeeping_warmed": False,
        "copy_source_dtype": "torch.int64",
        "copy_destination_dtype": "torch.int32",
        "batches": batches,
        "output_copy_pairs": output_copy_pairs,
        "slot_copy_pairs": slot_copy_pairs,
        "spec_copy_pairs": spec_copy_pairs,
        "flags_zero_fills": flags_zero_fills,
        "persistent_copy_state_restored": persistent_copy_state_restored,
        "flags_state_restored": flags_state_restored,
    }, committer


def _fr13_fixed32_warm_final_full_postprocess(vocab_size):
    """Warm final TAW/committer paths before any measured fixed32 event."""
    vocab = int(vocab_size)
    existing = globals().get("_FR13_FIXED32_BOOT_WARM_EVIDENCE")
    if isinstance(existing, dict) and existing.get("ready") is True:
        if int(existing.get("vocab_size", -1)) != vocab:
            raise RuntimeError(
                "FR13 fixed32 boot-warm vocabulary changed after readiness"
            )
        return dict(existing)
    observed_absent = _FR13_FIXED32_OBSERVED_CURRENT is None
    pending_absent = globals().get("_FR13_FIXED32_PENDING_EVENT") is None
    if (
        vocab <= 0
        or not observed_absent
        or not pending_absent
        or not _fr13_fixed32_boot_preseed_allowed()
    ):
        raise RuntimeError(
            "FR13 fixed32 postprocess warm entered measured/capture state"
        )
    capacity = int(globals().get("_FR13_FIXED32_PRESEED_CAP", 0))
    done = globals().get("_FR13_FIXED32_PRESEEDED_BATCHES")
    if (
        capacity not in (1, 2, 3, 4)
        or set(done or ()) != set(range(1, capacity + 1))
    ):
        raise RuntimeError(
            "FR13 fixed32 postprocess warm requires complete preseed"
        )
    stacks = globals().get("_FR13_EAGER_PACK_STACKS")
    if not isinstance(stacks, dict) or not torch.is_tensor(
        stacks.get("spec_idx")
    ):
        raise RuntimeError(
            "FR13 fixed32 postprocess warm is missing persistent device state"
        )
    taw_module = __import__("sys").modules.get(
        "_fr13_device_multidraft_kernel"
    )
    if taw_module is None:
        raise RuntimeError(
            "FR13 fixed32 postprocess warm is missing the TAW module"
        )
    device = stacks["spec_idx"].device
    with torch.inference_mode():
        taw = taw_module.fr13_fixed32_taw_warm_execute(
            device,
            mode=_FR13_FIXED32_MODE,
            valid_mask=int(_FR13_FIXED32_VALID_MASK),
            max_batch_size=capacity,
            vocab_size=vocab,
        )
        cfwd_direct = taw_module.fr13_fixed32_cfwd_logit_direct_warm_execute(
            device,
            mode=_FR13_FIXED32_MODE,
            valid_mask=int(_FR13_FIXED32_VALID_MASK),
            max_batch_size=capacity,
            vocab_size=vocab,
        )
        if cfwd_direct.get("requested") is True and (
            cfwd_direct.get("ready") is not True
            or cfwd_direct.get("classification") != "unmeasured_boot"
        ):
            raise RuntimeError("FR13 CFWD logit-direct boot warm did not complete")
        tail, committer = _fr13_fixed32_warm_device_postprocess_tail(
            taw_module,
            device,
            capacity,
            vocab,
        )
    expected_batches = tuple(range(1, capacity + 1))
    if (
        taw.get("ready") is not True
        or taw.get("classification") != "unmeasured_boot"
        or tuple(taw.get("batches", ())) != expected_batches
        or int(taw.get("executions", -1)) != capacity
        or taw.get("cache_lease_current") is not True
        or taw.get("rng_state_restored") is not True
        or taw.get("staging_state_restored") is not True
        or taw.get("measured_state_restored") is not True
        or tail.get("ready") is not True
        or tail.get("classification") != "unmeasured_boot"
        or tail.get("hardware_scope") != "device_postprocess_kernels"
        or tail.get("wrapper_bookkeeping_warmed") is not False
        or tail.get("copy_source_dtype") != "torch.int64"
        or tail.get("copy_destination_dtype") != "torch.int32"
        or tuple(tail.get("batches", ())) != expected_batches
        or int(tail.get("output_copy_pairs", -1)) != capacity
        or int(tail.get("slot_copy_pairs", -1))
        != capacity * (capacity + 1) // 2
        or int(tail.get("spec_copy_pairs", -1)) != capacity
        or int(tail.get("flags_zero_fills", -1)) != 1
        or tail.get("persistent_copy_state_restored") is not True
        or tail.get("flags_state_restored") is not True
        or committer.get("ready") is not True
        or committer.get("classification") != "unmeasured_boot"
        or tuple(committer.get("batches", ())) != expected_batches
        or int(committer.get("replays", -1)) != capacity
        or committer.get("route_lease_current") is not True
        or committer.get("bank_state_restored") is not True
        or committer.get("input_state_restored") is not True
        or committer.get("measured_state_restored") is not True
        or int(committer.get("conv_commit_direct_launches", -1))
        != capacity
        or int(committer.get("conv_commit_gather_launches", -1)) != 0
        or int(committer.get("conv_commit_scatter_launches", -1)) != 0
        or committer.get("conv_bank_state_restored") is not True
        or committer.get("conv_staging_state_restored") is not True
        or committer.get("alias_destination_contract")
        != "exact_alias_only_16x3"
        or committer.get("scratch_overwrite_proven") is not True
        or _FR13_FIXED32_OBSERVED_CURRENT is not None
        or globals().get("_FR13_FIXED32_PENDING_EVENT") is not None
    ):
        raise RuntimeError(
            "FR13 fixed32 postprocess boot warm did not restore state"
        )
    evidence = {
        "ready": True,
        "classification": "unmeasured_boot",
        "mode": _FR13_FIXED32_MODE,
        "capacity": capacity,
        "vocab_size": vocab,
        "batches": expected_batches,
        "taw_executions": capacity,
        "hardware_scope": "device_postprocess_kernels",
        "wrapper_bookkeeping_warmed": False,
        "copy_source_dtype": "torch.int64",
        "copy_destination_dtype": "torch.int32",
        "output_copy_pairs": capacity,
        "slot_copy_pairs": capacity * (capacity + 1) // 2,
        "spec_copy_pairs": capacity,
        "flags_zero_fills": 1,
        "persistent_copy_state_restored": True,
        "flags_state_restored": True,
        "conv_commit_direct_launches": capacity,
        "conv_commit_gather_launches": 0,
        "conv_commit_scatter_launches": 0,
        "committer_replays": capacity,
        "observed_event_absent": observed_absent,
        "pending_event_absent": pending_absent,
    }
    globals()["_FR13_FIXED32_BOOT_WARM_EVIDENCE"] = evidence
    return dict(evidence)


def _fr13_fixed32_preseed_sfwd_conv_postprep_capture():
    """Bind graph outputs to the exact pregather/sticky preseed leases."""
    if not globals().get("_FR13_FIXED32_SFWD_CONV_POSTPREP_GRAPH", False):
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 SFWD conv/post-prep output preseed ran during capture"
        )
    import lumo_flywheel_serving.fr10_gdn_tree_kernel as _fr13_f32_kernel
    from lumo_flywheel_serving.fr13_sfwd_conv_postprep_fusion import (
        preseed_fixed32_sfwd_conv_postprep_capture_bindings as _fr13_sfwd_preseed,
    )

    pregather_state = getattr(
        _fr13_f32_kernel, "_FR13_FIXED32_CONV_PREGATHER", {}
    ).get("state")
    committer_route = getattr(
        _fr13_f32_kernel, "_FR13_FIXED32_COMMITTER_FAST_ROUTE", {}
    ).get("state")
    stacks = globals().get("_FR13_EAGER_PACK_STACKS")
    layers = globals().get("_FR13_REPLAY_LAYERS")
    if (
        not isinstance(pregather_state, dict)
        or not isinstance(committer_route, dict)
        or not isinstance(stacks, dict)
        or not isinstance(layers, dict)
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep graph preseed registries are missing"
        )
    order = tuple(str(value) for value in stacks.get("layer_order", ()))
    if (
        len(order) != 48
        or len(set(order)) != 48
        or any(name not in layers for name in order)
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep graph preseed layer order drifted"
        )
    evidence = _fr13_sfwd_preseed(
        layer_order=order,
        layer_objects=tuple(layers[name] for name in order),
        pregather_state=pregather_state,
        committer_route=committer_route,
    )
    globals()["_FR13_FIXED32_SFWD_CONV_POSTPREP_PRESEED"] = evidence
    return dict(evidence)


def _fr13_fixed32_preseed_sfwd_conv_postprep_profile_capture(num_reqs):
    """Seal all eager profile output leases before the first FULL capture."""
    if not globals().get("_FR13_FIXED32_SFWD_CONV_POSTPREP_GRAPH", False):
        return None
    if not _fr13_fixed32_sfwd_conv_postprep_profile_capture_active():
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 SFWD conv/post-prep profile output preseed ran during capture"
        )
    stacks = globals().get("_FR13_EAGER_PACK_STACKS")
    layers = globals().get("_FR13_REPLAY_LAYERS")
    if not isinstance(stacks, dict) or not isinstance(layers, dict):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep profile preseed registries are missing"
        )
    order = tuple(str(value) for value in stacks.get("layer_order", ()))
    if (
        len(order) != 48
        or len(set(order)) != 48
        or any(name not in layers for name in order)
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep profile preseed layer order drifted"
        )
    from lumo_flywheel_serving.fr13_sfwd_conv_postprep_fusion import (
        preseed_fixed32_sfwd_conv_postprep_profile_capture_bindings as _preseed,
    )

    evidence = _preseed(
        layer_order=order,
        layer_objects=tuple(layers[name] for name in order),
        batch_size=int(num_reqs),
    )
    if (
        evidence is None
        or not isinstance(evidence, dict)
        or evidence.get("ready") is not True
        or evidence.get("profile_capture") is not True
        or int(evidence.get("layers", -1)) != 48
        or int(num_reqs) not in tuple(evidence.get("batches", ()))
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep profile output preseed is incomplete"
        )
    globals()["_FR13_FIXED32_SFWD_CONV_POSTPREP_PROFILE_PRESEED"] = evidence
    return dict(evidence)


def _fr13_fixed32_sfwd_conv_postprep_profile_producer_needed(
    runtime_mode,
    num_tokens,
    num_reqs,
    uniform,
    has_lora,
    num_active_loras,
):
    """Return whether the throwaway FULL profile still needs an eager producer.

    ``profile_cudagraph_memory`` reaches ``_warmup_and_capture`` with the stock
    warmup loop only, and that loop runs ``for_cudagraph_capture=False``
    metadata. The GDN builder then reports ``num_spec_decodes == 0``, the
    fixed32 tree-conv route never executes, and no layer publishes the eager
    conv/post-prep operands the profile preseed seals -- the FULL capture then
    fails with "capture lacks preseeded output bindings". Run one eager
    capture-shaped forward inside the armed profile scope before sealing.
    """
    if not globals().get("_FR13_FIXED32_SFWD_CONV_POSTPREP_GRAPH", False):
        return False
    if str(runtime_mode).upper() != "FULL":
        return False
    if not _fr13_fixed32_sfwd_conv_postprep_profile_capture_active():
        return False
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 SFWD conv/post-prep profile producer ran during capture"
        )
    descriptor = _fr13_fixed32_graph_descriptor(
        str(runtime_mode).upper(),
        num_tokens,
        num_reqs,
        uniform,
        has_lora,
        num_active_loras,
    )
    scope = globals().get("_FR13_FIXED32_PROFILE_CAPTURE_SCOPE")
    if (
        not isinstance(scope, dict)
        or scope.get("descriptor") != descriptor
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep profile producer descriptor drifted"
        )
    stacks = globals().get("_FR13_EAGER_PACK_STACKS")
    layers = globals().get("_FR13_REPLAY_LAYERS")
    if not isinstance(stacks, dict) or not isinstance(layers, dict):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep profile producer registries are missing"
        )
    order = tuple(str(value) for value in stacks.get("layer_order", ()))
    if (
        len(order) != 48
        or len(set(order)) != 48
        or any(name not in layers for name in order)
    ):
        raise RuntimeError(
            "FR13 SFWD conv/post-prep profile producer layer order drifted"
        )
    from lumo_flywheel_serving.fr13_sfwd_conv_postprep_fusion import (
        fixed32_sfwd_conv_postprep_profile_producer_pending as _pending,
    )

    return bool(
        _pending(
            layer_objects=tuple(layers[name] for name in order),
            batch_size=int(num_reqs),
        )
    )


def _fr13_fixed32_assert_final_full_preseed_ready(num_reqs):
    """Fail before CUDA capture unless the eager producer published its lease."""
    def _same_tensor_view(left, right):
        # SD-layout conv cache access transposes the persistent cache on every
        # forward, producing a new Tensor wrapper for the same exact view.
        return bool(
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.device == right.device
            and left.dtype == right.dtype
            and left.layout == right.layout
            and tuple(int(value) for value in left.shape)
            == tuple(int(value) for value in right.shape)
            and tuple(int(value) for value in left.stride())
            == tuple(int(value) for value in right.stride())
            and int(left.storage_offset()) == int(right.storage_offset())
            and int(left.data_ptr()) == int(right.data_ptr())
            and int(left.untyped_storage().data_ptr())
            == int(right.untyped_storage().data_ptr())
        )

    batch = int(num_reqs)
    capacity = int(globals().get("_FR13_FIXED32_PRESEED_CAP", 0))
    done = globals().get("_FR13_FIXED32_PRESEEDED_BATCHES")
    import lumo_flywheel_serving.fr10_gdn_tree_kernel as _fr13_f32_kernel
    from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
        audit_fixed32_conv_commit_lease as _fr13_f32_ca,
        fixed32_committer_counters as _fr13_f32_cc,
        fixed32_committer_warmup_counters as _fr13_f32_cwc,
        fixed32_conv_col0_pregather_counters as _fr13_f32_pc,
    )
    commit_audit = _fr13_f32_ca()
    pregather = _fr13_f32_pc()
    committer = _fr13_f32_cc()
    committer_warm = _fr13_f32_cwc()
    boot_warm = globals().get("_FR13_FIXED32_BOOT_WARM_EVIDENCE")
    if not isinstance(boot_warm, dict):
        boot_warm = {}
    if globals().get("_FR13_FIXED32_SFWD_CONV_POSTPREP_GRAPH", False):
        sfwd_preseed = globals().get(
            "_FR13_FIXED32_SFWD_CONV_POSTPREP_PRESEED"
        )
        from lumo_flywheel_serving.fr13_sfwd_conv_postprep_fusion import (
            fixed32_sfwd_conv_postprep_capture_runtime_guard as _fr13_sfwd_guard,
        )

        sfwd_expected = {
            "ready": True,
            "schema": "fr13.fixed32.sfwd_conv_postprep.capture_cache.v1",
            "capacity": capacity,
            "layers": 48,
            "batches": tuple(range(1, capacity + 1)),
            "base_output_tensors": 48 * 6,
            "bound_output_views": 48 * capacity * 6,
            "capture_host_syncs_per_layer": 0,
            "ssi_value_proof": "persistent_pregather_boot_selfcheck",
            # The committer's sticky guard is an opt-in arm no fixed32 serving
            # launcher sets, so the sentinel the captured fusion poisons is
            # fusion-owned there. Derive the name instead of pinning it.
            "runtime_guard": _fr13_sfwd_guard(
                committer_route=getattr(
                    _fr13_f32_kernel, "_FR13_FIXED32_COMMITTER_FAST_ROUTE", {}
                ).get("state"),
            ),
        }
        if sfwd_preseed != sfwd_expected:
            raise RuntimeError(
                "FR13 SFWD conv/post-prep graph output preseed is incomplete: "
                + _fr13_fixed32_drift_detail(sfwd_preseed, sfwd_expected)
            )
    warm_vocab = int(boot_warm.get("vocab_size", 0))
    pregather_state = getattr(
        _fr13_f32_kernel, "_FR13_FIXED32_CONV_PREGATHER", {}
    ).get("state")
    committer_state = getattr(
        _fr13_f32_kernel, "_FR13_FIXED32_COMMITTER_FAST_ROUTE", {}
    ).get("state")
    stacks = globals().get("_FR13_EAGER_PACK_STACKS")
    layers = globals().get("_FR13_REPLAY_LAYERS")
    order = (
        ()
        if not isinstance(stacks, dict)
        else tuple(str(value) for value in stacks.get("layer_order", ()))
    )
    current_ssm = (
        ()
        if not isinstance(layers, dict) or len(order) != 48
        else tuple(
            getattr(layers.get(name), "_fr13_replay_ssm_state", None)
            for name in order
        )
    )
    current_conv = (
        ()
        if not isinstance(layers, dict) or len(order) != 48
        else tuple(
            getattr(layers.get(name), "_fr13_replay_conv_state", None)
            for name in order
        )
    )
    pregather_banks = (
        ()
        if not isinstance(pregather_state, dict)
        else tuple(pregather_state.get("banks", ()))
    )
    pregather_ssm_banks = (
        ()
        if not isinstance(pregather_state, dict)
        else tuple(pregather_state.get("ssm_banks", ()))
    )
    committer_banks = (
        ()
        if not isinstance(committer_state, dict)
        else tuple(committer_state.get("banks", ()))
    )
    staging = (
        None
        if not isinstance(pregather_state, dict)
        else pregather_state.get("staging")
    )
    current_commit_ssi = (
        None if not isinstance(stacks, dict) else stacks.get("spec_idx")
    )
    current_accepted_paths = globals().get(
        "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR"
    )
    current_accepted_lens = globals().get(
        "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR"
    )
    taw_batches = []
    taw_warm = {}
    taw_module = __import__("sys").modules.get(
        "_fr13_device_multidraft_kernel"
    )
    if (
        taw_module is not None
        and isinstance(pregather_state, dict)
        and capacity in (1, 2, 3, 4)
    ):
        for taw_batch in range(1, capacity + 1):
            taw_module.fr13_fixed32_taw_preseeded_counts(
                pregather_state["anchor"].device,
                mode=_FR13_FIXED32_MODE,
                valid_mask=int(_FR13_FIXED32_VALID_MASK),
                batch_size=taw_batch,
            )
            taw_batches.append(taw_batch)
        if warm_vocab > 0:
            taw_warm = taw_module.fr13_fixed32_taw_warmup_counters(
                pregather_state["anchor"].device,
                mode=_FR13_FIXED32_MODE,
                valid_mask=int(_FR13_FIXED32_VALID_MASK),
                max_batch_size=capacity,
                vocab_size=warm_vocab,
            )
    actual = {
        "batch": batch,
        "capacity": capacity,
        "done": tuple(sorted(int(value) for value in (done or ()))),
        "mode": (
            None
            if not isinstance(pregather_state, dict)
            else pregather_state.get("mode")
        ),
        "preseeded": pregather.get("preseeded"),
        "pointer_entries": pregather.get("pointer_entries"),
        "max_batch_size": pregather.get("max_batch_size"),
        "preseeded_batches": tuple(
            pregather.get("preseeded_batches", ())
        ),
        "layer_order": order,
        "pregather_layer_order": (
            ()
            if not isinstance(pregather_state, dict)
            else tuple(pregather_state.get("layer_order", ()))
        ),
        "current_ssm_complete": (
            len(current_ssm) == 48
            and all(value is not None for value in current_ssm)
        ),
        "current_conv_complete": (
            len(current_conv) == 48
            and all(value is not None for value in current_conv)
        ),
        "pregather_bank_aliases": (
            len(pregather_banks) == 48
            and len(current_conv) == 48
            and all(
                _same_tensor_view(bank, current)
                for bank, current in zip(
                    pregather_banks, current_conv, strict=True
                )
            )
        ),
        "pregather_ssm_bank_aliases": (
            len(pregather_ssm_banks) == 48
            and len(current_ssm) == 48
            and all(
                bank is current
                for bank, current in zip(
                    pregather_ssm_banks, current_ssm, strict=True
                )
            )
        ),
        "committer_bank_aliases": (
            len(committer_banks) == 48
            and len(current_ssm) == 48
            and all(
                bank is current
                for bank, current in zip(
                    committer_banks, current_ssm, strict=True
                )
            )
        ),
        "staging_shape": (
            None
            if not torch.is_tensor(staging)
            else tuple(int(value) for value in staging.shape)
        ),
        "commit_ssi_alias": (
            isinstance(pregather_state, dict)
            and pregather_state.get("commit_spec_state_indices")
            is current_commit_ssi
        ),
        "commit_paths_alias": (
            isinstance(pregather_state, dict)
            and pregather_state.get("accepted_paths")
            is current_accepted_paths
        ),
        "commit_lens_alias": (
            isinstance(pregather_state, dict)
            and pregather_state.get("accepted_lens")
            is current_accepted_lens
        ),
        "commit_route": (
            None
            if not isinstance(pregather_state, dict)
            else pregather_state.get("contract", {}).get("commit_route")
        ),
        "commit_bank_overlap_policy": (
            None
            if not isinstance(pregather_state, dict)
            else pregather_state.get("contract", {}).get(
                "commit_bank_overlap_policy"
            )
        ),
        "commit_bank_partial_overlap": (
            None
            if not isinstance(pregather_state, dict)
            else pregather_state.get("contract", {}).get(
                "commit_bank_partial_overlap"
            )
        ),
        "commit_bank_alias_groups": (
            None
            if not isinstance(pregather_state, dict)
            else pregather_state.get("contract", {}).get(
                "commit_bank_alias_groups"
            )
        ),
        "commit_bank_alias_width": (
            None
            if not isinstance(pregather_state, dict)
            else pregather_state.get("contract", {}).get(
                "commit_bank_alias_width"
            )
        ),
        "commit_bank_destination_guard": (
            None
            if not isinstance(pregather_state, dict)
            else pregather_state.get("contract", {}).get(
                "commit_bank_destination_guard"
            )
        ),
        "commit_null_row_rejected": (
            None
            if not isinstance(pregather_state, dict)
            else pregather_state.get("contract", {}).get(
                "commit_null_row_rejected"
            )
        ),
        "commit_lease_audited": commit_audit.get("lease_audited"),
        "committer_captures": committer.get("captures"),
        "committer_preseeded_graphs": committer.get("preseeded_graphs"),
        "committer_preseeded_batches": tuple(
            committer.get("preseeded_batches", ())
        ),
        "committer_required_capacity": committer.get(
            "required_capacity"
        ),
        "committer_all_batches_ready": committer.get(
            "all_batches_ready"
        ),
        "taw_batches": tuple(taw_batches),
        "boot_warm_ready": boot_warm.get("ready"),
        "boot_warm_classification": boot_warm.get("classification"),
        "boot_warm_mode": boot_warm.get("mode"),
        "boot_warm_capacity": boot_warm.get("capacity"),
        "boot_warm_batches": tuple(boot_warm.get("batches", ())),
        "boot_warm_taw_executions": boot_warm.get("taw_executions"),
        "boot_warm_hardware_scope": boot_warm.get("hardware_scope"),
        "boot_warm_wrapper_bookkeeping_warmed": boot_warm.get(
            "wrapper_bookkeeping_warmed"
        ),
        "boot_warm_copy_source_dtype": boot_warm.get(
            "copy_source_dtype"
        ),
        "boot_warm_copy_destination_dtype": boot_warm.get(
            "copy_destination_dtype"
        ),
        "boot_warm_output_copy_pairs": boot_warm.get(
            "output_copy_pairs"
        ),
        "boot_warm_slot_copy_pairs": boot_warm.get("slot_copy_pairs"),
        "boot_warm_spec_copy_pairs": boot_warm.get("spec_copy_pairs"),
        "boot_warm_flags_zero_fills": boot_warm.get(
            "flags_zero_fills"
        ),
        "boot_warm_copy_state_restored": boot_warm.get(
            "persistent_copy_state_restored"
        ),
        "boot_warm_flags_state_restored": boot_warm.get(
            "flags_state_restored"
        ),
        "boot_warm_conv_direct": boot_warm.get(
            "conv_commit_direct_launches"
        ),
        "boot_warm_conv_gathers": boot_warm.get(
            "conv_commit_gather_launches"
        ),
        "boot_warm_conv_scatters": boot_warm.get(
            "conv_commit_scatter_launches"
        ),
        "boot_warm_committer_replays": boot_warm.get(
            "committer_replays"
        ),
        "boot_warm_observed_absent": boot_warm.get(
            "observed_event_absent"
        ),
        "boot_warm_pending_absent": boot_warm.get(
            "pending_event_absent"
        ),
        "current_observed_absent": (
            globals().get("_FR13_FIXED32_OBSERVED_CURRENT") is None
        ),
        "current_pending_absent": (
            globals().get("_FR13_FIXED32_PENDING_EVENT") is None
        ),
        "taw_warm_ready": taw_warm.get("ready"),
        "taw_warm_classification": taw_warm.get("classification"),
        "taw_warm_batches": tuple(taw_warm.get("batches", ())),
        "taw_warm_executions": taw_warm.get("executions"),
        "taw_warm_cache_lease_current": taw_warm.get(
            "cache_lease_current"
        ),
        "taw_warm_rng_restored": taw_warm.get("rng_state_restored"),
        "taw_warm_staging_restored": taw_warm.get(
            "staging_state_restored"
        ),
        "taw_warm_measured_restored": taw_warm.get(
            "measured_state_restored"
        ),
        "committer_warm_ready": committer_warm.get("ready"),
        "committer_warm_classification": committer_warm.get(
            "classification"
        ),
        "committer_warm_batches": tuple(
            committer_warm.get("batches", ())
        ),
        "committer_warm_replays": committer_warm.get("replays"),
        "committer_warm_conv_direct": committer_warm.get(
            "conv_commit_direct_launches"
        ),
        "committer_warm_conv_gathers": committer_warm.get(
            "conv_commit_gather_launches"
        ),
        "committer_warm_conv_scatters": committer_warm.get(
            "conv_commit_scatter_launches"
        ),
        "committer_warm_route_lease_current": committer_warm.get(
            "route_lease_current"
        ),
        "committer_warm_bank_restored": committer_warm.get(
            "bank_state_restored"
        ),
        "committer_warm_conv_bank_restored": committer_warm.get(
            "conv_bank_state_restored"
        ),
        "committer_warm_conv_staging_restored": committer_warm.get(
            "conv_staging_state_restored"
        ),
        "committer_warm_alias_contract": committer_warm.get(
            "alias_destination_contract"
        ),
        "committer_warm_input_restored": committer_warm.get(
            "input_state_restored"
        ),
        "committer_warm_measured_restored": committer_warm.get(
            "measured_state_restored"
        ),
        "committer_warm_scratch_overwrite_proven": committer_warm.get(
            "scratch_overwrite_proven"
        ),
    }
    row_elems = (
        -1
        if not isinstance(pregather_state, dict)
        else int(pregather_state.get("row_elems", -1))
    )
    expected = {
        "batch": batch,
        "capacity": capacity,
        "done": tuple(range(1, capacity + 1)),
        "mode": _FR13_FIXED32_MODE,
        "preseeded": True,
        "pointer_entries": 48,
        "max_batch_size": capacity,
        "preseeded_batches": tuple(range(1, capacity + 1)),
        "layer_order": order,
        "pregather_layer_order": order,
        "current_ssm_complete": True,
        "current_conv_complete": True,
        "pregather_bank_aliases": True,
        "pregather_ssm_bank_aliases": True,
        "committer_bank_aliases": True,
        "staging_shape": (48, capacity, row_elems),
        "commit_ssi_alias": True,
        "commit_paths_alias": True,
        "commit_lens_alias": True,
        "commit_route": "fixed32_direct_source_col0",
        "commit_bank_overlap_policy": "exact_alias_only_16x3",
        "commit_bank_partial_overlap": False,
        "commit_bank_alias_groups": 16,
        "commit_bank_alias_width": 3,
        "commit_bank_destination_guard": "alias_row_unique",
        "commit_null_row_rejected": True,
        "commit_lease_audited": True,
        "committer_captures": capacity,
        "committer_preseeded_graphs": capacity,
        "committer_preseeded_batches": tuple(
            range(1, capacity + 1)
        ),
        "committer_required_capacity": capacity,
        "committer_all_batches_ready": True,
        "taw_batches": tuple(range(1, capacity + 1)),
        "boot_warm_ready": True,
        "boot_warm_classification": "unmeasured_boot",
        "boot_warm_mode": _FR13_FIXED32_MODE,
        "boot_warm_capacity": capacity,
        "boot_warm_batches": tuple(range(1, capacity + 1)),
        "boot_warm_taw_executions": capacity,
        "boot_warm_hardware_scope": "device_postprocess_kernels",
        "boot_warm_wrapper_bookkeeping_warmed": False,
        "boot_warm_copy_source_dtype": "torch.int64",
        "boot_warm_copy_destination_dtype": "torch.int32",
        "boot_warm_output_copy_pairs": capacity,
        "boot_warm_slot_copy_pairs": capacity * (capacity + 1) // 2,
        "boot_warm_spec_copy_pairs": capacity,
        "boot_warm_flags_zero_fills": 1,
        "boot_warm_copy_state_restored": True,
        "boot_warm_flags_state_restored": True,
        "boot_warm_conv_direct": capacity,
        "boot_warm_conv_gathers": 0,
        "boot_warm_conv_scatters": 0,
        "boot_warm_committer_replays": capacity,
        "boot_warm_observed_absent": True,
        "boot_warm_pending_absent": True,
        "current_observed_absent": True,
        "current_pending_absent": True,
        "taw_warm_ready": True,
        "taw_warm_classification": "unmeasured_boot",
        "taw_warm_batches": tuple(range(1, capacity + 1)),
        "taw_warm_executions": capacity,
        "taw_warm_cache_lease_current": True,
        "taw_warm_rng_restored": True,
        "taw_warm_staging_restored": True,
        "taw_warm_measured_restored": True,
        "committer_warm_ready": True,
        "committer_warm_classification": "unmeasured_boot",
        "committer_warm_batches": tuple(range(1, capacity + 1)),
        "committer_warm_replays": capacity,
        "committer_warm_conv_direct": capacity,
        "committer_warm_conv_gathers": 0,
        "committer_warm_conv_scatters": 0,
        "committer_warm_route_lease_current": True,
        "committer_warm_bank_restored": True,
        "committer_warm_conv_bank_restored": True,
        "committer_warm_conv_staging_restored": True,
        "committer_warm_alias_contract": "exact_alias_only_16x3",
        "committer_warm_input_restored": True,
        "committer_warm_measured_restored": True,
        "committer_warm_scratch_overwrite_proven": True,
    }
    if (
        capacity not in (1, 2, 3, 4)
        or not 1 <= batch <= capacity
        or actual != expected
    ):
        raise RuntimeError(
            "FR13 fixed32 final FULL eager preseed did not publish lease: "
            + repr((actual, expected))
        )


def _fr13_fixed32_boot_preseed_allowed():
    return bool(
        _FR13_FIXED32_MODE
        and _FR13_FIXED32_PROFILE_MEMORY_SCOPE is False
        and _FR13_FIXED32_PROFILE_CAPTURE_SCOPE is None
        and _FR13_FIXED32_CAPTURE_CONTEXT is None
        and not _FR13_FIXED32_CAPTURE_MANIFESTS
        and _FR13_FIXED32_OBSERVED_CURRENT is None
        and globals().get("_FR13_FIXED32_PENDING_EVENT") is None
        and not _FR13_FIXED32_CAPTURE_FROZEN
    )


def _fr13_fixed32_boot_preseed_inputs():
    if (
        not _fr13_fixed32_boot_preseed_allowed()
        or globals().get("_FR13_FIXED32_PRESEEDED_BATCHES")
    ):
        return None
    stacks = globals().get("_FR13_EAGER_PACK_STACKS")
    layers = globals().get("_FR13_REPLAY_LAYERS")
    if stacks is None or layers is None:
        return None
    if not isinstance(stacks, dict) or not isinstance(layers, dict):
        raise RuntimeError("FR13 fixed32 preseed registries are malformed")
    raw_order = stacks.get("layer_order")
    if not isinstance(raw_order, (list, tuple)):
        return None
    order = tuple(str(name) for name in raw_order)
    if (
        len(order) != 48
        or len(set(order)) != 48
        or any(not name for name in order)
    ):
        raise RuntimeError("FR13 fixed32 preseed layer order is malformed")
    if any(name not in layers for name in order):
        return None
    layer_objects = tuple(layers[name] for name in order)
    banks = tuple(
        getattr(layer, "_fr13_replay_ssm_state", None)
        for layer in layer_objects
    )
    conv_banks = tuple(
        getattr(layer, "_fr13_replay_conv_state", None)
        for layer in layer_objects
    )
    if any(bank is None for bank in banks + conv_banks):
        return None
    return stacks, layers, order, layer_objects, banks, conv_banks


def _fr13_fixed32_observed_nonpure_dispatch(cudagraph_mode_name):
    if (
        not _FR13_FIXED32_MODE
        or globals().get("_FR13_FIXED32_EAGER_KERNEL_DIAGNOSTIC", False)
    ):
        return
    counters = _FR13_FIXED32_NONPURE_DISPATCH
    name = str(cudagraph_mode_name).upper()
    counters["guarded_steps"] += 1
    if name == "PIECEWISE":
        counters["piecewise_steps"] += 1
        return
    if name == "NONE":
        counters["none_steps"] += 1
        return
    counters["forbidden_full_steps"] += 1
    raise RuntimeError(
        "FR13 fixed32 nonpure dispatch resolved forbidden mode " + repr(name)
    )


def _fr13_fixed32_nonpure_dispatch_counters():
    counters = dict(_FR13_FIXED32_NONPURE_DISPATCH)
    if (
        set(counters)
        != {
            "guarded_steps",
            "piecewise_steps",
            "none_steps",
            "forbidden_full_steps",
        }
        or any(type(value) is not int or value < 0 for value in counters.values())
        or counters["guarded_steps"]
        != (
            counters["piecewise_steps"]
            + counters["none_steps"]
            + counters["forbidden_full_steps"]
        )
    ):
        raise RuntimeError(
            "FR13 fixed32 nonpure dispatch counters are incoherent: "
            + repr(counters)
        )
    return counters


def _fr13_fixed32_nonpure_commit_replays_by_batch():
    counters = dict(_FR13_FIXED32_NONPURE_COMMIT_REPLAYS_BY_BATCH)
    if (
        set(counters) != {1, 2, 3, 4}
        or any(type(value) is not int or value < 0 for value in counters.values())
    ):
        raise RuntimeError(
            "FR13 fixed32 nonpure commit replay counters are incoherent: "
            + repr(counters)
        )
    return counters


def _fr13_fixed32_taw_full_graph_begin(taw_module, mode, batch_size):
    key = (str(mode), int(batch_size))
    full_graph_passed = key in _FR13_FIXED32_TAW_FULL_GRAPH_PASSES
    if not full_graph_passed:
        entry = taw_module._fr13_fixed32_taw_native_live_entry(
            mode=key[0], batch_size=key[1]
        )
        if entry is None:
            # Untreated width: the gate declines, the engine serves the stock
            # reference and nothing is captured or recorded for this step.
            return {"status": "no_gate", "batch_size": key[1]}
        if entry.get("native_ab_live_gate_pending") is True:
            raise RuntimeError(
                "FR13 fixed32 TAW full-graph gate was already pending"
            )
        # An uncaptured root check may write diagnostic evidence, but it must
        # not satisfy the first measured full-graph replay for this batch.
        entry["native_ab_live_pass_emitted"] = False
    gate_report = taw_module.fr13_fixed32_taw_native_live_gate_begin(
        mode=key[0], batch_size=key[1]
    )
    expected = "passed" if full_graph_passed else "armed"
    if gate_report.get("status") != expected:
        raise RuntimeError(
            "FR13 fixed32 TAW native full-graph gate did not "
            f"{expected}: " + repr(gate_report)
        )
    return gate_report


def _fr13_fixed32_taw_full_graph_on_replay(taw_module, mode, batch_size):
    key = (str(mode), int(batch_size))
    gate_report = taw_module.fr13_fixed32_taw_native_live_gate_on_replay(
        mode=key[0], batch_size=key[1]
    )
    if gate_report.get("status") == "no_gate":
        # Declined at begin for an untreated width; nothing was armed, so there
        # is no replay evidence to read and no pass to record.
        return gate_report
    if gate_report.get("status") != "passed":
        raise RuntimeError(
            "FR13 fixed32 TAW native live gate did not pass on the first "
            "measured full-graph replay: " + repr(gate_report)
        )
    _FR13_FIXED32_TAW_FULL_GRAPH_PASSES.add(key)
    return gate_report


def _fr13_fixed32_observed_begin(
    mode,
    batch_size,
    forward_step_index,
    request_ids,
    batch_rows,
    spec_rows,
    physical_draft_counts,
):
    global _FR13_FIXED32_CAPTURE_FROZEN
    global _FR13_FIXED32_OBSERVED_CURRENT
    global _FR13_FIXED32_TOPOLOGY_NEEDLE_EMITTED
    if (
        not _FR13_FIXED32_MODE
        or globals().get("_FR13_FIXED32_EAGER_KERNEL_DIAGNOSTIC", False)
    ):
        return
    batch = int(batch_size)
    forward = int(forward_step_index)
    req_ids = tuple(str(value) for value in request_ids)
    physical_counts = tuple(int(value) for value in physical_draft_counts)
    if (
        mode != _FR13_FIXED32_MODE
        or batch not in (1, 2, 3, 4)
        or forward < 0
        or len(req_ids) != batch
        or any(not value for value in req_ids)
        or len(set(req_ids)) != batch
        or int(batch_rows) != batch
        or int(spec_rows) != batch
        or len(physical_counts) != batch
        or any(value != 31 for value in physical_counts)
    ):
        raise RuntimeError(
            "FR13 fixed32 observed begin identity drift: "
            + repr(
                (
                    mode,
                    batch,
                    forward,
                    req_ids,
                    batch_rows,
                    spec_rows,
                    physical_counts,
                )
            )
        )
    if _FR13_FIXED32_OBSERVED_CURRENT is not None:
        raise RuntimeError("FR13 fixed32 observed events overlapped")
    if globals().get("_FR13_FIXED32_PENDING_EVENT") is not None:
        raise RuntimeError(
            "FR13 fixed32 began a forward before prior KV completion"
        )
    if _FR13_FIXED32_CAPTURE_CONTEXT is not None:
        raise RuntimeError("FR13 fixed32 measured event began during capture")
    if _FR13_FIXED32_PROFILE_CAPTURE_SCOPE is not None:
        raise RuntimeError(
            "FR13 fixed32 measured event began during profile capture scope"
        )
    if _FR13_FIXED32_PROFILE_MEMORY_SCOPE is not False:
        raise RuntimeError(
            "FR13 fixed32 measured event began during profile-memory scope"
        )
    capacity = int(globals().get("_FR13_FIXED32_PRESEED_CAP", 0))
    captured_batches = []
    for _signature, canonical in _FR13_FIXED32_CAPTURE_MANIFESTS.values():
        try:
            manifest = __import__("json").loads(canonical)
            if (
                manifest.get("mode") == _FR13_FIXED32_MODE
                and manifest.get("descriptor", {}).get("runtime_mode") == "FULL"
                and int(manifest.get("physical_rows_per_request", -1)) == 32
            ):
                captured_batches.append(int(manifest["batch_size"]))
        except Exception as error:
            raise RuntimeError(
                "FR13 fixed32 stored capture manifest is malformed"
            ) from error
    if (
        capacity not in (1, 2, 3, 4)
        or sorted(captured_batches) != list(range(1, capacity + 1))
    ):
        raise RuntimeError(
            "FR13 fixed32 full-graph capture set is incomplete before freeze: "
            + repr((captured_batches, capacity))
        )
    from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
        fixed32_conv_col0_pregather_counters as _fr13_f32_pregather_counters,
    )
    pregather = _fr13_f32_pregather_counters()
    capture_by_batch = pregather.get("graph_capture_stages_by_batch", {})
    actual_by_batch = pregather.get("actual_stages_by_batch", {})
    (
        _fr13_f32_begin_capture_stages,
        _fr13_f32_begin_capture_by_batch,
    ) = _fr13_fixed32_pregather_capture_expectation(capacity)
    if (
        pregather.get("preseeded") is not True
        or int(pregather.get("pointer_entries", -1)) != 48
        or int(pregather.get("max_batch_size", -1)) != capacity
        or tuple(pregather.get("preseeded_batches", ()))
        != tuple(range(1, capacity + 1))
        or int(pregather.get("graph_capture_stages", -1))
        != _fr13_f32_begin_capture_stages
        or {
            batch: int(
                capture_by_batch.get(
                    batch, capture_by_batch.get(str(batch), -1)
                )
            )
            for batch in (1, 2, 3, 4)
        }
        != _fr13_f32_begin_capture_by_batch
        or int(pregather.get("actual_stages", -1)) != 0
        or any(
            int(actual_by_batch.get(batch, actual_by_batch.get(str(batch), -1)))
            != 0
            for batch in (1, 2, 3, 4)
        )
        or int(pregather.get("profile_capture_stages", -1)) != 0
        or int(pregather.get("aux_capture_stages", -1)) != 0
    ):
        raise RuntimeError(
            "FR13 fixed32 pregather capture set is incomplete before freeze: "
            + repr(pregather)
        )
    if (
        _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT is not None
        or _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT is not None
    ):
        raise RuntimeError(
            "FR13 fixed32 measured forward began during drafter work"
        )
    if not _FR13_FIXED32_TOPOLOGY_NEEDLE_EMITTED:
        print(_fr13_fixed32_topology_needle(), flush=True)
        _FR13_FIXED32_TOPOLOGY_NEEDLE_EMITTED = True
    if _FR13_FIXED32_TAW_NATIVE_PRECOMPUTE:
        taw_module = __import__("sys").modules.get(
            "_fr13_device_multidraft_kernel"
        )
        if taw_module is None:
            raise RuntimeError(
                "FR13 fixed32 TAW native live gate is missing its module"
            )
        _fr13_fixed32_taw_full_graph_begin(taw_module, mode, batch)
    _FR13_FIXED32_CAPTURE_FROZEN = True
    globals()["_FR13_FIXED32_CURRENT_FORWARD_STEP"] = forward
    _FR13_FIXED32_OBSERVED_CURRENT = _fr13_fixed32_observed_new_state(
        mode, batch, forward, "unbound", req_ids
    )
    _FR13_FIXED32_OBSERVED_CURRENT["batch_purity"] = {
        "batch_rows": batch,
        "spec_rows": batch,
        "physical_draft_counts": list(physical_counts),
        "mixed_pseudo_rows": 0,
        "all_physical_31": True,
    }
    from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
        fixed32_conv_zero_tail_live_prepare_replay,
    )
    fixed32_conv_zero_tail_live_prepare_replay(
        mode=mode, batch_size=batch, enabled=True
    )
    taw_module = __import__("sys").modules.get(
        "_fr13_device_multidraft_kernel"
    )
    if taw_module is None:
        raise RuntimeError("FR13 fixed32 observed begin is missing the TAW module")
    taw_module.fr13_fixed32_cfwd_logit_direct_live_prepare_replay(
        mode=mode, batch_size=batch, enabled=True
    )


def _fr13_fixed32_observed_work_target(stage, capturing, batch_size):
    batch = int(batch_size)
    if bool(capturing):
        context = globals().get("_FR13_FIXED32_CAPTURE_CONTEXT")
        if context is None:
            if globals().get("_FR13_FIXED32_OBSERVED_CURRENT") is not None:
                raise RuntimeError(
                    "FR13 fixed32 capture/recompile began during measured event"
                )
            # Auxiliary PIECEWISE captures at boot are not acceptance graphs.
            return None, False
        capture_batch = int(context["descriptor"]["num_reqs"])
        if batch != capture_batch:
            raise RuntimeError(
                "FR13 fixed32 capture call batch drift at "
                + str(stage)
                + ": "
                + repr((batch, capture_batch))
            )
        return context["work"], True
    event = globals().get("_FR13_FIXED32_OBSERVED_CURRENT")
    if event is None:
        if globals().get("_FR13_FIXED32_CAPTURE_FROZEN", False):
            raise RuntimeError(
                "FR13 fixed32 unscoped eager work after capture freeze at "
                + str(stage)
            )
        # Profile/warmup execution before the first measured event.
        return None, False
    if batch != int(event["batch_size"]):
        raise RuntimeError(
            "FR13 fixed32 eager work batch drift at "
            + str(stage)
            + ": "
            + repr((batch, event["batch_size"]))
        )
    if event["execution_basis"] not in ("unbound", "eager_direct"):
        raise RuntimeError(
            "FR13 fixed32 eager work mixed with graph replay at " + str(stage)
        )
    event["execution_basis"] = "eager_direct"
    return event, False


def _fr13_fixed32_observed_tree_attn(
    layer_name, q_rows, bias_shape, capturing=False
):
    rows = int(q_rows)
    name = str(layer_name)
    shape = tuple(int(value) for value in bias_shape)
    proposal = globals().get("_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT")
    drafter_capture = globals().get(
        "_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT"
    )
    if proposal is not None or drafter_capture is not None:
        target_capture = globals().get("_FR13_FIXED32_CAPTURE_CONTEXT")
        target_event = globals().get("_FR13_FIXED32_OBSERVED_CURRENT")
        if (
            not isinstance(proposal, dict)
            or not isinstance(drafter_capture, dict)
            or target_capture is not None
            or target_event is not None
            or capturing is not True
            or _FR13_FIXED32_PROFILE_CAPTURE_SCOPE is not None
            or _FR13_FIXED32_PROFILE_MEMORY_SCOPE is not False
        ):
            raise RuntimeError(
                "FR13 fixed32 tree-attention ownership scope drift: "
                + repr(
                    (
                        type(proposal).__name__,
                        type(drafter_capture).__name__,
                        target_capture is not None,
                        target_event is not None,
                        capturing,
                    )
                )
            )
        batch = drafter_capture.get("batch_size")
        tree_calls = drafter_capture.get("tree_attn_calls")
        tree_rows = drafter_capture.get("tree_attn_rows")
        mtp_calls = drafter_capture.get("mtp_forward_calls")
        mtp_rows = drafter_capture.get("mtp_forward_rows")
        prior_name = drafter_capture.get("tree_attn_layer")
        prior_shape = drafter_capture.get("tree_attn_bias_shape")
        if (
            type(batch) is not int
            or batch not in (1, 2, 3, 4)
            or drafter_capture.get("capturing") is not True
            or type(drafter_capture.get("graph_id")) is not int
            or int(drafter_capture["graph_id"]) <= 0
            or drafter_capture.get("mode") != proposal.get("mode")
            or drafter_capture.get("request_ids")
            != proposal.get("request_ids")
            or batch != proposal.get("batch_size")
            or proposal.get("mode") != _FR13_FIXED32_MODE
            or proposal.get("mtp_execution_basis") != "unbound"
            # FR14_GATE_SPLIT_GRAPH: a split capture records `lo` then `hi`
            # inside ONE proposal, so graph_captures is the 1-BASED INDEX of the
            # segment being recorded, not a constant. Tying it to the context's
            # own segment is stricter than a membership test: it makes "the hi
            # capture ran without the lo capture" unrepresentable, and it is
            # byte-identical to the old literal for the ungated context, whose
            # segment is 0. This is the site that refused Arm G.
            or proposal.get("graph_captures")
            != int(drafter_capture.get("segment", 0)) + 1
            or proposal.get("graph_replays") != 0
            or proposal.get("mtp_forward_calls") != 0
            or proposal.get("mtp_forward_rows") != 0
            or type(tree_calls) is not int
            or type(tree_rows) is not int
            or type(mtp_calls) is not int
            or type(mtp_rows) is not int
            # each segment counts its OWN forwards from zero, so the bound is
            # this segment's pass count (4 ungated, 2 per half) -- again
            # identical to the old literal when passes == 4
            or tree_calls not in tuple(
                range(int(drafter_capture.get("passes", 4)))
            )
            or tree_rows != tree_calls * batch
            or mtp_calls != tree_calls
            or mtp_rows != tree_rows
            or name != _FR13_FIXED32_DRAFTER_TREE_LAYER
            or rows != batch
            or shape != (1, 1)
            or prior_name not in (None, name)
            or prior_shape not in (None, shape)
        ):
            raise RuntimeError(
                "FR13 fixed32 drafter tree-attention work drift: "
                + repr(
                    (
                        name,
                        rows,
                        shape,
                        proposal,
                        drafter_capture,
                    )
                )
            )
        drafter_capture["tree_attn_calls"] = tree_calls + 1
        drafter_capture["tree_attn_rows"] = tree_rows + rows
        drafter_capture["tree_attn_layer"] = name
        drafter_capture["tree_attn_bias_shape"] = shape
        return
    if rows <= 0 or rows % 32 != 0:
        raise RuntimeError(
            "FR13 fixed32 tree-attention rows are not fixed32: " + str(rows)
        )
    event, _capture = _fr13_fixed32_observed_work_target(
        "tree attention", capturing, rows // 32
    )
    if event is None:
        return
    expected_rows = int(event["batch_size"]) * 32
    if (
        name not in _FR13_FIXED32_TARGET_TREE_LAYERS
        or rows != expected_rows
        or shape != (32, 32)
    ):
        raise RuntimeError(
            "FR13 fixed32 tree-attention work drift: "
            + repr((name, rows, expected_rows, shape))
        )
    if name in event["tree_layers"]:
        raise RuntimeError(
            "FR13 fixed32 tree-attention layer executed twice: " + name
        )
    event["tree_layers"].add(name)
    event["tree_calls"] += 1
    event["tree_q_rows"] += rows
    prior_shape = event["tree_bias_shape"]
    if prior_shape is not None and prior_shape != shape:
        raise RuntimeError("FR13 fixed32 tree-attention bias shape changed")
    event["tree_bias_shape"] = shape


def _fr13_fixed32_observed_gdn(
    layer_name,
    batch_index,
    batch_size,
    n_actual,
    n_pad,
    q_rows,
    k_rows,
    v_rows,
    out_rows,
    strict_mask_shape,
    visible_mask_shape,
    runtime_state,
    capturing=False,
):
    event, _capture = _fr13_fixed32_observed_work_target(
        "GDN scan", capturing, batch_size
    )
    if event is None:
        return
    name = str(layer_name)
    batch = int(event["batch_size"])
    index = int(batch_index)
    shapes = (
        int(n_actual),
        int(n_pad),
        int(q_rows),
        int(k_rows),
        int(v_rows),
        int(out_rows),
        tuple(int(value) for value in strict_mask_shape),
        tuple(int(value) for value in visible_mask_shape),
    )
    if (
        not name
        or index < 0
        or index >= batch
        or shapes != (32, 32, 32, 32, 32, 32, (32, 32), (32, 32))
    ):
        raise RuntimeError("FR13 fixed32 GDN physical geometry drift: " + repr(shapes))
    call_key = (name, index)
    if call_key in event["gdn_calls"]:
        raise RuntimeError(
            "FR13 fixed32 GDN layer/request executed twice: " + repr(call_key)
        )
    if not isinstance(runtime_state, dict):
        raise RuntimeError("FR13 fixed32 GDN runtime state is missing")
    if (
        _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION
        and batch == _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH
    ):
        executed_gdn = runtime_state.get("executed_gdn")
        expected_grid = (batch,)
        if (
            not isinstance(executed_gdn, dict)
            or executed_gdn.get("route")
            != "fixed32_single_launch_gqa_group3"
            or executed_gdn.get("candidate")
            != "fixed32_gdn_single_launch_gqa_group3_v1"
            or int(executed_gdn.get("physical_launches", -1)) != 1
            or int(executed_gdn.get("physical_programs", -1)) != batch
            or tuple(executed_gdn.get("physical_grid_z", ())) != expected_grid
            or int(
                executed_gdn.get("physical_recurrence_critical_path", -1)
            )
            != 32
            or int(executed_gdn.get("state_export_writes", -1)) != 0
            or int(executed_gdn.get("state_parent_reads", -1)) != 0
        ):
            raise RuntimeError(
                "FR13 GDN GQA-group3 production did not replace the captured "
                "incumbent launch: " + repr(executed_gdn)
            )
    if (
        _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION
        and batch == _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH
    ):
        # THE ENGAGEMENT NEEDLE. A production arm that silently fell back to the
        # incumbent would still serve correct bytes and still look healthy -- it
        # would just be a lie about what ran, and every timing number taken
        # against it would be a measurement of the incumbent. So the arm is
        # required to prove, per decode call, that it actually replaced the
        # captured launch.
        #
        # The two zeros are the load-bearing part and are what distinguishes this
        # from a kernel that merely runs fast. The deployed reference issues TWO
        # launches per request-layer and hands state from the first to the second
        # through an export write and a parent read; the fold issues ONE launch
        # for the whole batch and keeps that state in registers. So a genuine
        # single_launch serve has exactly zero export writes and zero parent
        # reads. A fallback to the reference cannot produce those zeros, and a
        # partial fold cannot either.
        #
        # physical_programs and physical_grid_z are compared against `batch`, not
        # against 1. That is deliberate: the b1-shaped literal is the recurring
        # defect in this arm's history, and at batch 1 the two agree, so a
        # hardcoded 1 here would pass every B1 test and fail only in production at
        # width 4 -- exactly the class of bug this campaign already paid for once.
        executed_gdn = runtime_state.get("executed_gdn")
        expected_grid = (batch,)
        if (
            not isinstance(executed_gdn, dict)
            or executed_gdn.get("route")
            != "fixed32_single_launch_tree"
            or executed_gdn.get("candidate")
            != "fixed32_gdn_single_launch_tree_v2"
            or int(executed_gdn.get("physical_launches", -1)) != 1
            or int(executed_gdn.get("physical_programs", -1)) != batch
            or tuple(executed_gdn.get("physical_grid_z", ())) != expected_grid
            or int(
                executed_gdn.get("physical_recurrence_critical_path", -1)
            )
            != 32
            or int(executed_gdn.get("state_export_writes", -1)) != 0
            or int(executed_gdn.get("state_parent_reads", -1)) != 0
        ):
            raise RuntimeError(
                "FR13 GDN single-launch production did not replace the captured "
                "incumbent launch: " + repr(executed_gdn)
            )
    contract = runtime_state.get("fixed32_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("FR13 fixed32 GDN schedule contract is missing")
    normalized_contract = {
        "path_counts": tuple(int(value) for value in contract.get("path_counts", ())),
        "max_lengths": tuple(int(value) for value in contract.get("max_lengths", ())),
        "launches": int(contract.get("launches", -1)),
        "programs": int(contract.get("programs", -1)),
        "padded_slots": int(contract.get("padded_slots", -1)),
        "critical": int(contract.get("critical", -1)),
        "export_or_mask": int(contract.get("export_or_mask", -1)),
    }
    expected_contract = dict(_FR13_FIXED32_GDN_SCHEDULE_EXPECTED)
    if normalized_contract != expected_contract:
        # TWO-SIDED, and only the fields that differ. The one-sided form cost
        # round 22 a whole boot to diagnose: it printed the observed dict and
        # left the expectation to be read out of the patcher's source.
        _drifted = sorted(
            _name
            for _name in expected_contract
            if normalized_contract.get(_name) != expected_contract[_name]
        )
        raise RuntimeError(
            "FR13 fixed32 GDN schedule work drift for mode "
            + repr(_FR13_FIXED32_GDN_MODE)
            + ": "
            + "; ".join(
                _name
                + ": observed "
                + repr(normalized_contract.get(_name))
                + " against audited "
                + repr(expected_contract[_name])
                for _name in _drifted
            )
        )
    # SEVENTH member of the walk-derived-pin class, one statement after the
    # sixth. `critical` is the walk cap -- 12 at hydra27/tail6, 16 at hydra31 --
    # and it is the ONLY mode-varying field here: n_levels is the launch count
    # (2 under both profiles) and parent/emask/export rows are PHYSICAL_ROWS,
    # which no profile moves. It derives from the same planted table the
    # schedule contract above uses.
    _observed_state = {
        "schedule": runtime_state.get("schedule"),
        "route_armed": runtime_state.get("route_armed"),
        "n_levels": int(runtime_state.get("n_levels", -1)),
        "critical": int(runtime_state.get("critical", -1)),
        "parent_nodes": int(runtime_state.get("parent_nodes", -1)),
        "emask_rows": int(runtime_state.get("emask_rows", -1)),
        "export_rows": int(runtime_state.get("export_rows", -1)),
    }
    _expected_state = {
        "schedule": "fixed32",
        "route_armed": True,
        "n_levels": _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["launches"],
        "critical": _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["critical"],
        "parent_nodes": 32,
        "emask_rows": 32,
        "export_rows": 32,
    }
    if _observed_state != _expected_state:
        # TWO-SIDED, differing fields only. The one-sided form dumped the whole
        # runtime_state, which is how a stale `critical` came to be read as a
        # single-launch route that never armed: the None fields it printed are
        # not in this comparison at all, and are the ordinary state of a
        # default-off lever.
        _state_drift = sorted(
            _name
            for _name in _expected_state
            if _observed_state[_name] != _expected_state[_name]
        )
        raise RuntimeError(
            "FR13 fixed32 GDN runtime schedule state drift for mode "
            + repr(_FR13_FIXED32_GDN_MODE)
            + ": "
            + "; ".join(
                _name
                + ": observed "
                + repr(_observed_state[_name])
                + " against audited "
                + repr(_expected_state[_name])
                for _name in _state_drift
            )
        )
    parent_digest = contract.get("parent_sha256")
    ancestry_digest = contract.get("ancestry_sha256")
    if (
        not isinstance(parent_digest, str)
        or len(parent_digest) != 64
        or not isinstance(ancestry_digest, str)
        or len(ancestry_digest) != 64
    ):
        raise RuntimeError("FR13 fixed32 GDN topology digests are missing")
    for key, value in (
        ("gdn_parent_sha256", parent_digest),
        ("gdn_ancestry_sha256", ancestry_digest),
    ):
        prior = event[key]
        if prior is not None and prior != value:
            raise RuntimeError("FR13 fixed32 GDN topology digest changed")
        event[key] = value
    event["gdn_calls"].add(call_key)
    event["gdn_layers"].add(name)
    event["gdn_scan_calls"] += 1
    event["gdn_launches"] += normalized_contract["launches"]
    event["gdn_path_programs"] += normalized_contract["programs"]
    event["gdn_padded_slots"] += normalized_contract["padded_slots"]
    event["gdn_nodes"] += int(n_actual)
    event["gdn_critical_path"] = normalized_contract["critical"]
    event["gdn_grid_z"] = normalized_contract["path_counts"]
    event["gdn_max_path_lengths"] = normalized_contract["max_lengths"]
    prior_export = event["gdn_export_or_mask"]
    if (
        prior_export is not None
        and int(prior_export) != normalized_contract["export_or_mask"]
    ):
        raise RuntimeError("FR13 fixed32 GDN export mask changed")
    event["gdn_export_or_mask"] = normalized_contract["export_or_mask"]


def _fr13_fixed32_conv_runtime_contract(state, batch_size):
    batch = int(batch_size)
    if not isinstance(state, dict):
        raise RuntimeError("FR13 fixed32 conv pregather state is missing")
    capacity = state.get("max_batch_size")
    banks = state.get("banks")
    ssm_banks = state.get("ssm_banks")
    layer_order = state.get("layer_order")
    offsets = state.get("off16")
    bank_alias_classes = state.get("bank_alias_classes")
    bank_alias_ids = state.get("bank_alias_ids")
    bank_alias_ranks = state.get("bank_alias_ranks")
    bank_alias_ids_device = state.get("bank_alias_ids_device")
    bank_alias_peer_layers = state.get("bank_alias_peer_layers")
    bank_alias_peer_layers_device = state.get(
        "bank_alias_peer_layers_device"
    )
    sources = state.get("ssi_sources")
    ssi_ptrs = state.get("ssi_ptrs")
    ssi_strides = state.get("ssi_strides")
    commit_spec_state_indices = state.get("commit_spec_state_indices")
    accepted_paths = state.get("accepted_paths")
    accepted_lens = state.get("accepted_lens")
    direct_sources = state.get("source_stagings")
    source_offsets = state.get("source_off16")
    direct_state_src = state.get("state_src")
    zero_tail_count_enable = state.get("treeconv_zero_tail_count_enable")
    zero_tail_compared_events = state.get(
        "treeconv_zero_tail_compared_events"
    )
    zero_tail_differing_bytes = state.get(
        "treeconv_zero_tail_differing_bytes"
    )
    source_rows = state.get("source_rows_per_batch")
    conv_c = state.get("conv_c")
    conv_l = state.get("conv_l")
    staging = state.get("staging")
    row_guard_flags_by_batch = state.get("row_guard_flags_by_batch")
    row_elems = state.get("row_elems")
    block = state.get("block")
    contract = state.get("contract")
    if (
        type(capacity) is not int
        or capacity not in (1, 2, 3, 4)
        or not 1 <= batch <= capacity
        or state.get("mode") != _FR13_FIXED32_MODE
        or tuple(state.get("preseeded_batches", ()))
        != tuple(range(1, capacity + 1))
        or not isinstance(banks, tuple)
        or len(banks) != 48
        or not isinstance(ssm_banks, tuple)
        or len(ssm_banks) != 48
        or not isinstance(layer_order, tuple)
        or len(layer_order) != 48
        or len(set(layer_order)) != 48
        or any(not isinstance(name, str) or not name for name in layer_order)
        or not isinstance(contract, dict)
        or contract.get("route") != "in_graph_preconsume"
        or contract.get("staging_bank_nonalias") is not True
        or int(contract.get("block", -1)) != 1024
        or int(contract.get("layers", -1)) != 48
        or int(contract.get("pointer_entries", -1)) != 48
        or int(contract.get("ssi_pointer_entries", -1)) != 48
        or int(contract.get("ssi_groups", -1)) != 3
        or contract.get("commit_bank_overlap_policy")
        != "exact_alias_only_16x3"
        or contract.get("commit_bank_partial_overlap") is not False
        or int(contract.get("commit_bank_alias_groups", -1)) != 16
        or int(contract.get("commit_bank_alias_width", -1)) != 3
        or contract.get("commit_bank_destination_guard") != "alias_row_unique"
        or contract.get("commit_null_row_rejected") is not True
        or contract.get("commit_route") != "fixed32_direct_source_col0"
        or int(contract.get("commit_launches_per_event", -1)) != 1
        or int(contract.get("commit_direct_launches_per_event", -1)) != 1
        or contract.get("commit_row_guard_route")
        != "fixed32_triton_alias3_ownerpath_warp32_physical32_v4"
        or int(contract.get("commit_row_guard_kernel_launches_per_event", -1))
        != 1
        or int(contract.get("commit_row_guard_programs_per_request", -1)) != 48
        or int(contract.get("commit_row_guard_physical_rows", -1)) != 32
        or int(contract.get("commit_row_guard_path_capacity", -1)) != 16
        or int(contract.get("commit_row_guard_alias_width", -1)) != 3
        or int(contract.get("commit_row_guard_compare_capacity", -1)) != 16
        or int(
            contract.get(
                "commit_row_guard_path_validation_programs_per_request", -1
            )
        )
        != 1
        or int(
            contract.get(
                "commit_row_guard_path_vector_loads_per_request", -1
            )
        )
        != 1
        or int(
            contract.get(
                "commit_row_guard_alias_validation_programs_per_event", -1
            )
        )
        != 1
        or int(
            contract.get(
                "commit_row_guard_alias_vector_loads_per_event", -1
            )
        )
        != 2
        or int(
            contract.get(
                "commit_row_guard_selected_row_loads_per_program", -1
            )
        )
        != 0
        or contract.get("commit_row_guard_peer_topology_proof")
        != "preseed_lease_audit"
        or int(contract.get("commit_row_guard_torch_index_transforms", -1)) != 0
        or int(
            contract.get("commit_row_guard_async_scalar_reductions", -1)
        )
        != 1
        or int(contract.get("commit_row_guard_async_assertions", -1)) != 1
        or int(contract.get("commit_full_node_writebacks", -1)) != 0
        or int(contract.get("commit_conv_remaps", -1)) != 0
        or type(row_elems) is not int
        or row_elems <= 0
        or type(block) is not int
        or block != 1024
        or not torch.is_tensor(offsets)
        or tuple(int(value) for value in offsets.shape) != (48,)
        or str(offsets.dtype) != "torch.int64"
        or not offsets.is_contiguous()
        or not isinstance(bank_alias_classes, tuple)
        or len(bank_alias_classes) != 16
        or any(
            not isinstance(indices, tuple) or len(indices) != 3
            for indices in bank_alias_classes
        )
        or not isinstance(bank_alias_ids, tuple)
        or len(bank_alias_ids) != 48
        or not isinstance(bank_alias_ranks, tuple)
        or len(bank_alias_ranks) != 48
        or bank_alias_peer_layers
        != tuple(
            tuple(int(peer) for peer in bank_alias_classes[alias_id])
            for alias_id in bank_alias_ids
        )
        or not torch.is_tensor(bank_alias_ids_device)
        or tuple(int(value) for value in bank_alias_ids_device.shape) != (48,)
        or str(bank_alias_ids_device.dtype) != "torch.int64"
        or not bank_alias_ids_device.is_contiguous()
        or not torch.is_tensor(bank_alias_peer_layers_device)
        or tuple(int(value) for value in bank_alias_peer_layers_device.shape)
        != (48, 3)
        or str(bank_alias_peer_layers_device.dtype) != "torch.int32"
        or not bank_alias_peer_layers_device.is_contiguous()
        or not torch.is_tensor(staging)
        or not isinstance(sources, tuple)
        or len(sources) != 48
        or any(not torch.is_tensor(source) for source in sources)
        or any(
            source.ndim != 2
            or int(source.shape[0]) < capacity
            or int(source.shape[1]) < 1
            or str(source.dtype) != "torch.int32"
            or source.device != staging.device
            or any(int(value) <= 0 for value in source.stride())
            for source in sources
        )
        or len({int(source.data_ptr()) for source in sources}) != 3
        or not torch.is_tensor(ssi_ptrs)
        or tuple(int(value) for value in ssi_ptrs.shape) != (48,)
        or str(ssi_ptrs.dtype) != "torch.int64"
        or not ssi_ptrs.is_contiguous()
        or not torch.is_tensor(ssi_strides)
        or tuple(int(value) for value in ssi_strides.shape) != (48,)
        or str(ssi_strides.dtype) != "torch.int64"
        or not ssi_strides.is_contiguous()
        or tuple(int(value) for value in staging.shape)
        != (48, capacity, row_elems)
        or tuple(int(value) for value in staging.stride())
        != (capacity * row_elems, row_elems, 1)
        or not staging.is_contiguous()
        or any(not torch.is_tensor(bank) for bank in banks)
        or any(not torch.is_tensor(bank) for bank in ssm_banks)
        or any(
            bank.device != staging.device or bank.dtype != staging.dtype
            for bank in banks
        )
        or any(bank.device != staging.device for bank in ssm_banks)
        or any(
            int(bank.untyped_storage().data_ptr())
            != int(ssm_bank.untyped_storage().data_ptr())
            for bank, ssm_bank in zip(banks, ssm_banks, strict=True)
        )
        or not torch.is_tensor(commit_spec_state_indices)
        or commit_spec_state_indices.device != staging.device
        or str(commit_spec_state_indices.dtype) != "torch.int32"
        or commit_spec_state_indices.ndim != 3
        or int(commit_spec_state_indices.shape[0]) != 48
        or int(commit_spec_state_indices.shape[1]) < capacity
        or int(commit_spec_state_indices.shape[2]) != 32
        or not commit_spec_state_indices.is_contiguous()
        or not torch.is_tensor(accepted_paths)
        or accepted_paths.device != staging.device
        or str(accepted_paths.dtype) != "torch.int32"
        or accepted_paths.ndim != 2
        or int(accepted_paths.shape[0]) < capacity
        or int(accepted_paths.shape[1]) != 16
        or not accepted_paths.is_contiguous()
        or not torch.is_tensor(accepted_lens)
        or accepted_lens.device != staging.device
        or str(accepted_lens.dtype) != "torch.int32"
        or accepted_lens.ndim != 1
        or int(accepted_lens.shape[0]) < capacity
        or not accepted_lens.is_contiguous()
        or not isinstance(row_guard_flags_by_batch, dict)
        or tuple(sorted(row_guard_flags_by_batch))
        != tuple(range(1, capacity + 1))
        or any(
            not torch.is_tensor(row_guard_flags_by_batch[guard_batch])
            or row_guard_flags_by_batch[guard_batch].device != staging.device
            or str(row_guard_flags_by_batch[guard_batch].dtype) != "torch.bool"
            or tuple(
                int(value)
                for value in row_guard_flags_by_batch[guard_batch].shape
            )
            != (48 * guard_batch,)
            or not row_guard_flags_by_batch[guard_batch].is_contiguous()
            for guard_batch in range(1, capacity + 1)
        )
        or type(source_rows) is not int
        or source_rows != 36
        or type(conv_c) is not int
        or conv_c != 10240
        or type(conv_l) is not int
        or conv_l != 34
        or not isinstance(direct_sources, tuple)
        or len(direct_sources) != 48
        or any(not torch.is_tensor(source) for source in direct_sources)
        or len({int(source.data_ptr()) for source in direct_sources}) != 48
        or any(
            source.device != staging.device
            or source.dtype != staging.dtype
            or source.ndim != 2
            or int(source.shape[0]) < capacity * int(source_rows)
            or int(source.shape[1]) != int(conv_c)
            or tuple(int(value) for value in source.stride())
            != (int(conv_c), 1)
            or not source.is_contiguous()
            for source in direct_sources
        )
        or not torch.is_tensor(source_offsets)
        or source_offsets.device != staging.device
        or str(source_offsets.dtype) != "torch.int64"
        or tuple(int(value) for value in source_offsets.shape) != (48,)
        or not source_offsets.is_contiguous()
        or not torch.is_tensor(direct_state_src)
        or direct_state_src.device != staging.device
        or str(direct_state_src.dtype) != "torch.int64"
        or tuple(int(value) for value in direct_state_src.shape)
        != (32 * int(conv_l),)
        or not direct_state_src.is_contiguous()
        or any(
            not torch.is_tensor(counter)
            or counter.device != staging.device
            or str(counter.dtype) != "torch.int64"
            or tuple(int(value) for value in counter.shape) != ()
            for counter in (
                zero_tail_count_enable,
                zero_tail_compared_events,
                zero_tail_differing_bytes,
            )
        )
        or offsets.device != staging.device
        or bank_alias_ids_device.device != staging.device
        or bank_alias_peer_layers_device.device != staging.device
        or ssi_ptrs.device != staging.device
        or ssi_strides.device != staging.device
        or int(banks[0].shape[1]) * int(banks[0].shape[2]) != row_elems
    ):
        raise RuntimeError("FR13 fixed32 conv pregather runtime contract drift")
    source_identity = (
        tuple(id(bank) for bank in banks),
        tuple(id(bank) for bank in ssm_banks),
        id(offsets),
        id(bank_alias_ids_device),
        id(bank_alias_peer_layers_device),
        tuple(id(source) for source in sources),
        id(ssi_ptrs),
        id(ssi_strides),
        id(commit_spec_state_indices),
        id(accepted_paths),
        id(accepted_lens),
        tuple(id(source) for source in direct_sources),
        id(source_offsets),
        id(direct_state_src),
        id(zero_tail_count_enable),
        id(zero_tail_compared_events),
        id(zero_tail_differing_bytes),
        id(staging),
        tuple(
            id(row_guard_flags_by_batch[guard_batch])
            for guard_batch in range(1, capacity + 1)
        ),
    )
    source_data_ptrs = (
        tuple(int(bank.data_ptr()) for bank in banks),
        tuple(int(bank.data_ptr()) for bank in ssm_banks),
        tuple(
            int(bank.untyped_storage().data_ptr()) for bank in ssm_banks
        ),
        int(offsets.data_ptr()),
        int(bank_alias_ids_device.data_ptr()),
        int(bank_alias_peer_layers_device.data_ptr()),
        tuple(int(source.data_ptr()) for source in sources),
        int(ssi_ptrs.data_ptr()),
        int(ssi_strides.data_ptr()),
        int(commit_spec_state_indices.data_ptr()),
        int(accepted_paths.data_ptr()),
        int(accepted_lens.data_ptr()),
        tuple(int(source.data_ptr()) for source in direct_sources),
        int(source_offsets.data_ptr()),
        int(direct_state_src.data_ptr()),
        int(zero_tail_count_enable.data_ptr()),
        int(zero_tail_compared_events.data_ptr()),
        int(zero_tail_differing_bytes.data_ptr()),
        int(staging.data_ptr()),
        tuple(
            int(row_guard_flags_by_batch[guard_batch].data_ptr())
            for guard_batch in range(1, capacity + 1)
        ),
    )
    if (
        state.get("source_identity") != source_identity
        or state.get("source_data_ptrs") != source_data_ptrs
    ):
        raise RuntimeError(
            "FR13 fixed32 conv pregather persistent source identity drift"
        )
    bank_data_keys = [int(bank.data_ptr()) for bank in banks]
    bank_storage_keys = [
        int(bank.untyped_storage().data_ptr()) for bank in banks
    ]
    ssm_data_keys = [int(bank.data_ptr()) for bank in ssm_banks]
    ssm_storage_keys = [
        int(bank.untyped_storage().data_ptr()) for bank in ssm_banks
    ]
    ssi_data_keys = [int(source.data_ptr()) for source in sources]
    ssi_storage_keys = [
        int(source.untyped_storage().data_ptr()) for source in sources
    ]
    bank_data_groups = [
        list(dict.fromkeys(bank_data_keys)).index(key)
        for key in bank_data_keys
    ]
    bank_storage_groups = [
        list(dict.fromkeys(bank_storage_keys)).index(key)
        for key in bank_storage_keys
    ]
    ssm_data_groups = [
        list(dict.fromkeys(ssm_data_keys)).index(key) for key in ssm_data_keys
    ]
    ssm_storage_groups = [
        list(dict.fromkeys(ssm_storage_keys)).index(key)
        for key in ssm_storage_keys
    ]
    ssi_data_groups = [
        list(dict.fromkeys(ssi_data_keys)).index(key)
        for key in ssi_data_keys
    ]
    ssi_storage_groups = [
        list(dict.fromkeys(ssi_storage_keys)).index(key)
        for key in ssi_storage_keys
    ]
    staging_storage_key = int(staging.untyped_storage().data_ptr())
    live_alias_classes = tuple(
        tuple(
            index
            for index, pointer in enumerate(bank_data_keys)
            if pointer == unique_pointer
        )
        for unique_pointer in dict.fromkeys(bank_data_keys)
    )
    live_alias_ids = tuple(bank_data_groups)
    live_alias_ranks = tuple(
        live_alias_classes[alias_id].index(index)
        for index, alias_id in enumerate(live_alias_ids)
    )
    if (
        live_alias_classes != bank_alias_classes
        or live_alias_ids != bank_alias_ids
        or live_alias_ranks != bank_alias_ranks
        or len(live_alias_classes) != 16
        or any(len(indices) != 3 for indices in live_alias_classes)
        or ssm_data_groups != bank_data_groups
        or ssm_storage_keys != bank_storage_keys
        or tuple(state.get("bank_data_ptrs", ())) != tuple(bank_data_keys)
        or tuple(state.get("ssm_bank_data_ptrs", ())) != tuple(ssm_data_keys)
        or tuple(state.get("ssm_bank_storage_ptrs", ()))
        != tuple(ssm_storage_keys)
        or len(set(ssi_data_groups)) != 3
        or staging_storage_key in set(bank_storage_keys)
    ):
        raise RuntimeError(
            "FR13 fixed32 conv pregather alias contract drift"
        )
    layout_payload = {
        "capacity": capacity,
        "row_elems": row_elems,
        "block": block,
        "device_type": str(staging.device.type),
        "layer_order": list(layer_order),
        "bank_shapes": [
            [int(value) for value in bank.shape] for bank in banks
        ],
        "bank_strides": [
            [int(value) for value in bank.stride()] for bank in banks
        ],
        "bank_dtypes": [str(bank.dtype) for bank in banks],
        "bank_storage_offsets": [
            int(bank.storage_offset()) for bank in banks
        ],
        "bank_data_alias_groups": bank_data_groups,
        "bank_storage_alias_groups": bank_storage_groups,
        "bank_alias_classes": [list(indices) for indices in bank_alias_classes],
        "bank_alias_ids": list(bank_alias_ids),
        "bank_alias_ranks": list(bank_alias_ranks),
        "ssm_bank_shapes": [
            [int(value) for value in bank.shape] for bank in ssm_banks
        ],
        "ssm_bank_strides": [
            [int(value) for value in bank.stride()] for bank in ssm_banks
        ],
        "ssm_bank_dtypes": [str(bank.dtype) for bank in ssm_banks],
        "ssm_bank_storage_offsets": [
            int(bank.storage_offset()) for bank in ssm_banks
        ],
        "ssm_bank_data_alias_groups": ssm_data_groups,
        "ssm_bank_storage_alias_groups": ssm_storage_groups,
        "bank_alias_ids_shape": [
            int(value) for value in bank_alias_ids_device.shape
        ],
        "bank_alias_ids_stride": [
            int(value) for value in bank_alias_ids_device.stride()
        ],
        "bank_alias_ids_dtype": str(bank_alias_ids_device.dtype),
        "bank_alias_peer_layers": [
            [int(peer) for peer in peers]
            for peers in bank_alias_peer_layers
        ],
        "bank_alias_peer_layers_shape": [
            int(value) for value in bank_alias_peer_layers_device.shape
        ],
        "bank_alias_peer_layers_stride": [
            int(value) for value in bank_alias_peer_layers_device.stride()
        ],
        "bank_alias_peer_layers_dtype": str(
            bank_alias_peer_layers_device.dtype
        ),
        "commit_bank_overlap_policy": contract[
            "commit_bank_overlap_policy"
        ],
        "commit_bank_partial_overlap": contract[
            "commit_bank_partial_overlap"
        ],
        "commit_bank_alias_groups": contract["commit_bank_alias_groups"],
        "commit_bank_alias_width": contract["commit_bank_alias_width"],
        "commit_bank_destination_guard": contract[
            "commit_bank_destination_guard"
        ],
        "commit_null_row_rejected": contract["commit_null_row_rejected"],
        "commit_row_guard_route": contract["commit_row_guard_route"],
        "commit_row_guard_kernel_launches_per_event": contract[
            "commit_row_guard_kernel_launches_per_event"
        ],
        "commit_row_guard_programs_per_request": contract[
            "commit_row_guard_programs_per_request"
        ],
        "commit_row_guard_physical_rows": contract[
            "commit_row_guard_physical_rows"
        ],
        "commit_row_guard_path_capacity": contract[
            "commit_row_guard_path_capacity"
        ],
        "commit_row_guard_alias_width": contract[
            "commit_row_guard_alias_width"
        ],
        "commit_row_guard_compare_capacity": contract[
            "commit_row_guard_compare_capacity"
        ],
        "commit_row_guard_path_validation_programs_per_request": contract[
            "commit_row_guard_path_validation_programs_per_request"
        ],
        "commit_row_guard_path_vector_loads_per_request": contract[
            "commit_row_guard_path_vector_loads_per_request"
        ],
        "commit_row_guard_alias_validation_programs_per_event": contract[
            "commit_row_guard_alias_validation_programs_per_event"
        ],
        "commit_row_guard_alias_vector_loads_per_event": contract[
            "commit_row_guard_alias_vector_loads_per_event"
        ],
        "commit_row_guard_selected_row_loads_per_program": contract[
            "commit_row_guard_selected_row_loads_per_program"
        ],
        "commit_row_guard_peer_topology_proof": contract[
            "commit_row_guard_peer_topology_proof"
        ],
        "commit_row_guard_torch_index_transforms": contract[
            "commit_row_guard_torch_index_transforms"
        ],
        "commit_row_guard_async_scalar_reductions": contract[
            "commit_row_guard_async_scalar_reductions"
        ],
        "commit_row_guard_async_assertions": contract[
            "commit_row_guard_async_assertions"
        ],
        "row_guard_flag_shapes": {
            str(guard_batch): [
                int(value)
                for value in row_guard_flags_by_batch[guard_batch].shape
            ]
            for guard_batch in range(1, capacity + 1)
        },
        "row_guard_flag_strides": {
            str(guard_batch): [
                int(value)
                for value in row_guard_flags_by_batch[guard_batch].stride()
            ]
            for guard_batch in range(1, capacity + 1)
        },
        "row_guard_flag_dtype": "torch.bool",
        "ssi_pointer_entries": 48,
        "ssi_groups": 3,
        "ssi_source_shapes": [
            [int(value) for value in source.shape] for source in sources
        ],
        "ssi_source_strides": [
            [int(value) for value in source.stride()] for source in sources
        ],
        "ssi_source_dtypes": [str(source.dtype) for source in sources],
        "ssi_source_storage_offsets": [
            int(source.storage_offset()) for source in sources
        ],
        "ssi_data_alias_groups": ssi_data_groups,
        "ssi_storage_alias_groups": ssi_storage_groups,
        "direct_source_shapes": [
            [int(value) for value in source.shape]
            for source in direct_sources
        ],
        "direct_source_strides": [
            [int(value) for value in source.stride()]
            for source in direct_sources
        ],
        "direct_source_rows_per_batch": source_rows,
        "direct_source_pointer_entries": 48,
        "direct_state_src_shape": [32, conv_l],
        "direct_commit_grid": [
            48,
            batch,
            (conv_c + block - 1) // block,
        ],
        "ordered_source_consume_mapping": [
            {
                "index": index,
                "layer": layer_order[index],
                "bank_data_alias_group": bank_data_groups[index],
                "bank_storage_alias_group": bank_storage_groups[index],
                "ssm_bank_data_alias_group": ssm_data_groups[index],
                "ssm_bank_storage_alias_group": ssm_storage_groups[index],
                "bank_alias_id": bank_alias_ids[index],
                "bank_alias_rank": bank_alias_ranks[index],
                "ssi_data_alias_group": ssi_data_groups[index],
                "ssi_storage_alias_group": ssi_storage_groups[index],
            }
            for index in range(48)
        ],
        "ssi_ptr_shape": [int(value) for value in ssi_ptrs.shape],
        "ssi_ptr_stride": [int(value) for value in ssi_ptrs.stride()],
        "ssi_ptr_dtype": str(ssi_ptrs.dtype),
        "ssi_stride_shape": [int(value) for value in ssi_strides.shape],
        "ssi_stride_stride": [int(value) for value in ssi_strides.stride()],
        "ssi_stride_dtype": str(ssi_strides.dtype),
        "offset_shape": [int(value) for value in offsets.shape],
        "offset_stride": [int(value) for value in offsets.stride()],
        "offset_dtype": str(offsets.dtype),
        "staging_shape": [int(value) for value in staging.shape],
        "staging_stride": [int(value) for value in staging.stride()],
        "staging_dtype": str(staging.dtype),
        "staging_storage_offset": int(staging.storage_offset()),
        "staging_bank_nonalias": True,
        "grid": [
            48,
            batch,
            (row_elems + block - 1) // block,
        ],
    }
    instance_payload = {
        "source_identity": source_identity,
        "source_data_ptrs": source_data_ptrs,
    }
    layout_canonical = __import__("json").dumps(
        layout_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    instance_canonical = __import__("json").dumps(
        instance_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "layers": 48,
        "ssi_pointer_entries": 48,
        "ssi_groups": 3,
        "row_elems": row_elems,
        "block": block,
        "programs": 48 * batch * ((row_elems + block - 1) // block),
        "layout_sha256": __import__("hashlib").sha256(
            layout_canonical.encode("ascii")
        ).hexdigest(),
        "instance_sha256": __import__("hashlib").sha256(
            instance_canonical.encode("ascii")
        ).hexdigest(),
    }


def _fr13_fixed32_observed_conv_stage(
    layer_name,
    layer_index,
    batch_size,
    runtime_state,
    counters_before,
    counters_after,
    capturing=False,
):
    event, is_capture = _fr13_fixed32_observed_work_target(
        "conv pregather graph stage", capturing, batch_size
    )
    if event is None:
        raise RuntimeError(
            "FR13 fixed32 conv graph stage ran outside final FULL capture"
        )
    batch = int(batch_size)
    index = int(layer_index)
    name = str(layer_name)
    if (
        is_capture is not True
        or bool(capturing) is not True
        or not name
        or index != 0
        or batch != int(event["batch_size"])
        or int(event["conv_stage_calls"]) != 0
        or int(event["conv_consume_calls"]) != 0
        or int(event["gdn_scan_calls"]) != 0
        or event["gdn_calls"]
        or event["conv_source_layers"] != {0: name}
    ):
        raise RuntimeError(
            "FR13 fixed32 conv graph stage ordering drift: "
            + repr(
                (
                    name,
                    index,
                    batch,
                    is_capture,
                    capturing,
                    event["conv_stage_calls"],
                    event["conv_consume_calls"],
                    event["gdn_scan_calls"],
                )
            )
        )
    if not isinstance(counters_before, dict) or not isinstance(
        counters_after, dict
    ):
        raise RuntimeError("FR13 fixed32 conv graph stage counters are missing")
    capture_delta = _fr13_fixed32_counter_delta(
        counters_after, counters_before, "graph_capture_stages"
    )
    capture_batch_delta = _fr13_fixed32_batch_counter_delta(
        counters_after,
        counters_before,
        "graph_capture_stages_by_batch",
        batch,
    )
    other_capture_deltas = {
        other: _fr13_fixed32_batch_counter_delta(
            counters_after,
            counters_before,
            "graph_capture_stages_by_batch",
            other,
        )
        for other in (1, 2, 3, 4)
        if other != batch
    }
    if (
        counters_before.get("preseeded") is not True
        or counters_after.get("preseeded") is not True
        or capture_delta != 1
        or capture_batch_delta != 1
        or any(value != 0 for value in other_capture_deltas.values())
        or int(counters_after.get("actual_stages", -1)) != 0
        or any(
            int(value) != 0
            for value in counters_after.get("actual_stages_by_batch", {}).values()
        )
        or int(counters_after.get("profile_capture_stages", -1)) != 0
        or int(counters_after.get("aux_capture_stages", -1)) != 0
    ):
        raise RuntimeError(
            "FR13 fixed32 conv graph stage counter drift: "
            + repr(
                (
                    capture_delta,
                    capture_batch_delta,
                    other_capture_deltas,
                    counters_after,
                )
            )
        )
    normalized = _fr13_fixed32_conv_runtime_contract(runtime_state, batch)
    event["conv_stage_calls"] = 1
    event["conv_stage_before_all_consumes"] = True
    event["conv_stage_layer"] = name
    event["conv_stage_layers"] = int(normalized["layers"])
    event["conv_stage_row_elems"] = int(normalized["row_elems"])
    event["conv_stage_block"] = int(normalized["block"])
    event["conv_stage_programs"] = int(normalized["programs"])
    event["conv_stage_ssi_pointer_entries"] = int(
        normalized["ssi_pointer_entries"]
    )
    event["conv_stage_ssi_groups"] = int(normalized["ssi_groups"])
    event["conv_stage_source"] = str(normalized["layout_sha256"])
    event["conv_stage_instance"] = str(normalized["instance_sha256"])


def _fr13_fixed32_observed_conv_source(
    layer_name, layer_index, batch_size, capturing=False
):
    event, is_capture = _fr13_fixed32_observed_work_target(
        "conv pregather live SSI source", capturing, batch_size
    )
    if event is None:
        return
    name = str(layer_name)
    index = int(layer_index)
    sources = event["conv_source_layers"]
    if (
        is_capture is not True
        or bool(capturing) is not True
        or not name
        or not 0 <= index < 48
        or index in sources
        or name in sources.values()
        or int(event["conv_consume_calls"]) != len(sources)
        or (
            index == 0
            and (
                sources
                or int(event["conv_stage_calls"]) != 0
                or int(event["gdn_scan_calls"]) != 0
            )
        )
        or (
            index != 0
            and (
                int(event["conv_stage_calls"]) != 1
                or event["conv_stage_before_all_consumes"] is not True
            )
        )
    ):
        raise RuntimeError(
            "FR13 fixed32 live SSI source validation ordering drift: "
            + repr((name, index, sources, event["conv_stage_calls"]))
        )
    sources[index] = name


def _fr13_fixed32_observed_sfwd_conv_postprep(
    layer_name, batch_size, capturing=False
):
    """Record one fused conv/post-prep call into the forward work census.

    The fusion replaces the pregather stage kernel and the 48 per-layer
    consumes with a single kernel per layer, so none of the conv_* counters
    move. Without its own class the census would have to drop the conv
    expectation entirely and would stop proving that the conv work reached the
    captured graph at all.
    """
    event, _capture = _fr13_fixed32_observed_work_target(
        "sfwd conv/post-prep fusion", capturing, batch_size
    )
    if event is None:
        return
    name = str(layer_name)
    batch = int(batch_size)
    if (
        not name
        or batch != int(event["batch_size"])
        or name in event["sfwd_conv_postprep_layers"]
        or int(event["conv_stage_calls"]) != 0
        or int(event["conv_consume_calls"]) != 0
    ):
        raise RuntimeError(
            "FR13 fixed32 SFWD conv/post-prep census drift: "
            + repr(
                {
                    "layer": name,
                    "batch": batch,
                    "event_batch": event["batch_size"],
                    "already_seen": name in event["sfwd_conv_postprep_layers"],
                    "conv_stage_calls": event["conv_stage_calls"],
                    "conv_consume_calls": event["conv_consume_calls"],
                }
            )
        )
    event["sfwd_conv_postprep_layers"].add(name)
    event["sfwd_conv_postprep_calls"] += 1


def _fr13_fixed32_observed_conv_consume(
    layer_name, layer_index, batch_size, hit, capturing=False
):
    event, _capture = _fr13_fixed32_observed_work_target(
        "conv pregather consume", capturing, batch_size
    )
    if event is None:
        return
    name = str(layer_name)
    index = int(layer_index)
    batch = int(batch_size)
    if not name or batch != int(event["batch_size"]):
        raise RuntimeError(
            "FR13 fixed32 conv consume identity drift: "
            + repr((name, batch))
        )
    if (
        int(event["conv_stage_calls"]) != 1
        or event["conv_stage_before_all_consumes"] is not True
        or not isinstance(event["conv_stage_layer"], str)
        or not event["conv_stage_layer"]
        or event["conv_source_layers"].get(index) != name
    ):
        raise RuntimeError(
            "FR13 fixed32 conv pregather consumed before graph stage: " + name
        )
    if name in event["conv_consume_layers"]:
        raise RuntimeError(
            "FR13 fixed32 conv pregather consumed twice for layer " + name
        )
    event["conv_consume_layers"].add(name)
    event["conv_consume_calls"] += 1
    if bool(hit):
        event["conv_consume_hits"] += 1
        event["conv_freshness_matches"] += 1
    else:
        event["conv_consume_fallbacks"] += 1
        raise RuntimeError(
            "FR13 fixed32 conv pregather missed for layer " + name
        )


def _fr13_fixed32_graph_descriptor(
    runtime_mode,
    num_tokens,
    num_reqs,
    uniform,
    has_lora,
    num_active_loras,
):
    descriptor = {
        "runtime_mode": str(runtime_mode),
        "num_tokens": int(num_tokens),
        "num_reqs": int(num_reqs) if num_reqs is not None else None,
        "uniform": bool(uniform),
        "has_lora": bool(has_lora),
        "num_active_loras": int(num_active_loras),
    }
    batch = descriptor["num_reqs"]
    if (
        descriptor["runtime_mode"] != "FULL"
        or batch not in (1, 2, 3, 4)
        or descriptor["num_tokens"] != batch * 32
        or descriptor["uniform"] is not True
        or descriptor["has_lora"] is not False
        or descriptor["num_active_loras"] != 0
    ):
        raise RuntimeError(
            "FR13 fixed32 full-graph descriptor drift: " + repr(descriptor)
        )
    return descriptor


def _fr13_fixed32_kernel_shape():
    """House-enumerated kernel shape this arm ran under.

    Closed set: an arm is one of exactly these two, and every evidence
    artifact records which one so the difference is never silent.
    """
    return (
        "sfwd_fused_conv_postprep"
        if _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION
        else "unfused_conv_pregather"
    )


def _fr13_fixed32_pregather_capture_expectation(capacity):
    """Expected pregather capture-stage counts for this arm's kernel shape.

    The SFWD fusion subsumes the pregather stage kernel, so nothing launches
    at capture and both counters stay at zero.
    """
    if _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION:
        return 0, {batch: 0 for batch in (1, 2, 3, 4)}
    return capacity, {
        batch: 1 if batch <= capacity else 0 for batch in (1, 2, 3, 4)
    }


def _fr13_fixed32_validate_forward_work(work, label):
    batch = int(work["batch_size"])
    expected_gdn_calls = 48 * batch
    stage_row_elems = int(work["conv_stage_row_elems"])
    stage_block = int(work["conv_stage_block"])
    expected_stage_programs = (
        -1
        if stage_row_elems <= 0 or stage_block <= 0
        else 48
        * batch
        * ((stage_row_elems + stage_block - 1) // stage_block)
    )
    actual = {
        "tree_calls": int(work["tree_calls"]),
        "tree_layers": len(work["tree_layers"]),
        "tree_layer_set": frozenset(work["tree_layers"]),
        "tree_q_rows": int(work["tree_q_rows"]),
        "tree_bias_shape": work["tree_bias_shape"],
        "gdn_calls": int(work["gdn_scan_calls"]),
        "gdn_pairs": len(work["gdn_calls"]),
        "gdn_layers": len(work["gdn_layers"]),
        "gdn_launches": int(work["gdn_launches"]),
        "gdn_path_programs": int(work["gdn_path_programs"]),
        "gdn_padded_slots": int(work["gdn_padded_slots"]),
        "gdn_nodes": int(work["gdn_nodes"]),
        "gdn_critical_path": work["gdn_critical_path"],
        "gdn_grid_z": work["gdn_grid_z"],
        "gdn_max_path_lengths": work["gdn_max_path_lengths"],
        "gdn_export_or_mask": work["gdn_export_or_mask"],
        "conv_calls": int(work["conv_consume_calls"]),
        "conv_layers": len(work["conv_consume_layers"]),
        "conv_hits": int(work["conv_consume_hits"]),
        "conv_fallbacks": int(work["conv_consume_fallbacks"]),
        "conv_freshness": int(work["conv_freshness_matches"]),
        "conv_stage_calls": int(work["conv_stage_calls"]),
        "conv_stage_replays": int(work["conv_stage_replays"]),
        "conv_stage_before_all_consumes": work[
            "conv_stage_before_all_consumes"
        ],
        "conv_stage_layers": int(work["conv_stage_layers"]),
        "conv_stage_programs": int(work["conv_stage_programs"]),
        "conv_stage_ssi_pointer_entries": int(
            work["conv_stage_ssi_pointer_entries"]
        ),
        "conv_stage_ssi_groups": int(work["conv_stage_ssi_groups"]),
        "conv_stage_row_elems_positive": stage_row_elems > 0,
        "conv_stage_block": stage_block,
        "conv_stage_layer_present": (
            isinstance(work["conv_stage_layer"], str)
            and bool(work["conv_stage_layer"])
        ),
        "conv_stage_source_sha256": (
            isinstance(work["conv_stage_source"], str)
            and len(work["conv_stage_source"]) == 64
        ),
        "conv_stage_instance_sha256": (
            isinstance(work["conv_stage_instance"], str)
            and len(work["conv_stage_instance"]) == 64
        ),
        "conv_source_validations": len(work["conv_source_layers"]),
        "conv_source_indices": tuple(sorted(work["conv_source_layers"])),
        "conv_source_names": (
            set(work["conv_source_layers"].values())
            == set(work["conv_consume_layers"])
        ),
        "sfwd_conv_postprep_calls": int(work["sfwd_conv_postprep_calls"]),
        "sfwd_conv_postprep_layers": len(work["sfwd_conv_postprep_layers"]),
    }
    # The SFWD conv/post-prep fusion replaces the pregather stage kernel and
    # the 48 per-layer consumes with one kernel per layer, so every conv_*
    # counter legitimately stays at its initial value and the fused class
    # carries the proof instead. Values below are the ones a real fused B1
    # FULL capture produced (2026-08-08 boot screen at f4591891c); the fused
    # class is what keeps this an assertion that the work reached the graph
    # rather than a hole in the census.
    if _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION:
        conv_expected = {
            "conv_calls": 0,
            "conv_layers": 0,
            "conv_hits": 0,
            "conv_fallbacks": 0,
            "conv_freshness": 0,
            "conv_stage_calls": 0,
            "conv_stage_replays": 0,
            "conv_stage_before_all_consumes": False,
            "conv_stage_layers": 0,
            "conv_stage_programs": 0,
            "conv_stage_ssi_pointer_entries": 0,
            "conv_stage_ssi_groups": 0,
            "conv_stage_row_elems_positive": False,
            "conv_stage_block": 0,
            "conv_stage_layer_present": False,
            "conv_stage_source_sha256": False,
            "conv_stage_instance_sha256": False,
            "conv_source_validations": 0,
            "conv_source_indices": (),
            "conv_source_names": True,
            "sfwd_conv_postprep_calls": 48,
            "sfwd_conv_postprep_layers": 48,
        }
    else:
        conv_expected = {
            "conv_calls": 48,
            "conv_layers": 48,
            "conv_hits": 48,
            "conv_fallbacks": 0,
            "conv_freshness": 48,
            "conv_stage_calls": 1,
            "conv_stage_replays": 0 if str(label) == "captured" else 1,
            "conv_stage_before_all_consumes": True,
            "conv_stage_layers": 48,
            "conv_stage_programs": expected_stage_programs,
            "conv_stage_ssi_pointer_entries": 48,
            "conv_stage_ssi_groups": 3,
            "conv_stage_row_elems_positive": True,
            "conv_stage_block": 1024,
            "conv_stage_layer_present": True,
            "conv_stage_source_sha256": True,
            "conv_stage_instance_sha256": True,
            "conv_source_validations": 48,
            "conv_source_indices": tuple(range(48)),
            "conv_source_names": True,
            "sfwd_conv_postprep_calls": 0,
            "sfwd_conv_postprep_layers": 0,
        }
    expected = {
        "tree_calls": 16,
        "tree_layers": 16,
        "tree_layer_set": _FR13_FIXED32_TARGET_TREE_LAYERS,
        "tree_q_rows": 16 * batch * 32,
        "tree_bias_shape": (32, 32),
        # EIGHTH member of the class, and a THIRD KIND: an AGGREGATE. These are
        # per-layer schedule quantities times the GDN layer count, so the pin
        # was a PRODUCT -- 48 * 82 = 3936 -- and a product equals none of the
        # per-mode quantities a value census hunts for. hydra31 computes
        # 48 * 126 = 6048. Every GDN field below is now derived from the same
        # planted schedule authority the contracts read, invariant ones
        # included, so a ninth profile cannot inherit a stale aggregate from
        # any of them.
        "gdn_calls": expected_gdn_calls,
        "gdn_pairs": expected_gdn_calls,
        "gdn_layers": 48,
        "gdn_launches": (
            expected_gdn_calls * _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["launches"]
        ),
        "gdn_path_programs": (
            expected_gdn_calls * _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["programs"]
        ),
        "gdn_padded_slots": (
            expected_gdn_calls
            * _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["padded_slots"]
        ),
        "gdn_nodes": expected_gdn_calls * 32,
        "gdn_critical_path": _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["critical"],
        "gdn_grid_z": _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["path_counts"],
        "gdn_max_path_lengths": (
            _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["max_lengths"]
        ),
        "gdn_export_or_mask": (
            _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["export_or_mask"]
        ),
        **conv_expected,
    }
    if actual != expected:
        # TWO-SIDED, differing entries only. The whole-dict form buried three
        # differing fields in about eighty and cost a boot to read.
        _work_drift = sorted(
            _name
            for _name in set(actual) | set(expected)
            if actual.get(_name) != expected.get(_name)
        )
        raise RuntimeError(
            "FR13 fixed32 "
            + str(label)
            + " forward work is incomplete for mode "
            + repr(_FR13_FIXED32_GDN_MODE)
            + ": "
            + "; ".join(
                _name
                + ": observed "
                + repr(actual.get(_name))
                + " against audited "
                + repr(expected.get(_name))
                for _name in _work_drift
            )
        )


def _fr13_fixed32_forward_graph_registry(measured_by_batch=None):
    """Publish arm-comparable structure from live instance-bound manifests."""
    if not _FR13_FIXED32_MODE:
        return []
    capacity = int(globals().get("_FR13_FIXED32_PRESEED_CAP", 0))
    if capacity not in (1, 2, 3, 4):
        raise RuntimeError("FR13 fixed32 forward registry has invalid capacity")
    if measured_by_batch is None:
        measured_by_batch = {
            batch: sum(
                int(row.get("batch_size", -1)) == batch
                and row.get("event_complete") is True
                and int(row.get("matching_replays", -1)) == 1
                for row in _FR13_FIXED32_GRAPH_REPLAY_EVIDENCE
            )
            for batch in (1, 2, 3, 4)
        }
    normalized_measured = {
        batch: int(
            measured_by_batch.get(
                batch, measured_by_batch.get(str(batch), -1)
            )
        )
        for batch in (1, 2, 3, 4)
    }
    runtime_sys = __import__("sys")
    if "/workspace/scripts" not in runtime_sys.path:
        runtime_sys.path.insert(0, "/workspace/scripts")
    from fr13_fixed32_work_census import (
        forward_graph_structural_manifest,
        forward_graph_structural_signature,
    )
    manifests = {}
    for graph_id, entry in _FR13_FIXED32_CAPTURE_MANIFESTS.items():
        _instance_signature, canonical = _fr13_fixed32_manifest_entry(
            entry, "forward graph " + str(graph_id)
        )
        manifest = __import__("json").loads(canonical)
        descriptor = manifest.get("descriptor")
        tree = manifest.get("tree_attn")
        gdn = manifest.get("gdn")
        conv = manifest.get("conv_pregather")
        sfwd = manifest.get("sfwd_conv_postprep")
        registry_shape = _fr13_fixed32_kernel_shape()
        registry_fused = registry_shape == "sfwd_fused_conv_postprep"
        if manifest.get("kernel_shape") != registry_shape:
            raise RuntimeError(
                "FR13 fixed32 forward manifest kernel shape drifted: "
                + _fr13_fixed32_drift_detail(
                    manifest.get("kernel_shape"), registry_shape
                )
            )
        batch = int(manifest.get("batch_size", -1))
        if (
            manifest.get("schema")
            != "fr13-fixed32-forward-graph-manifest-v2"
            or manifest.get("mode") != _FR13_FIXED32_MODE
            or batch not in range(1, capacity + 1)
            or batch in manifests
            or not isinstance(descriptor, dict)
            or descriptor.get("runtime_mode") != "FULL"
            or int(descriptor.get("num_reqs", -1)) != batch
            or int(descriptor.get("num_tokens", -1)) != 32 * batch
            or descriptor.get("uniform") is not True
            or descriptor.get("has_lora") is not False
            or int(descriptor.get("num_active_loras", -1)) != 0
            or not isinstance(tree, dict)
            or not isinstance(gdn, dict)
            or not isinstance(sfwd if registry_fused else conv, dict)
            or (conv is not None and sfwd is not None)
        ):
            raise RuntimeError(
                "FR13 fixed32 live forward manifest registry drift: "
                + repr((graph_id, manifest))
            )
        tree_calls = int(tree.get("calls", -1))
        scan_calls = int(gdn.get("scan_calls", -1))
        # The fused route has no staging kernel, so it publishes no row/block
        # divisors and no staging layout digests.
        row_elems = -1 if registry_fused else int(conv.get("row_elems", -1))
        block = -1 if registry_fused else int(conv.get("block", -1))
        conv_layout_sha256 = None if registry_fused else conv.get("layout_sha256")
        conv_instance_sha256 = (
            None if registry_fused else conv.get("instance_sha256")
        )
        if (
            tree_calls <= 0
            or scan_calls <= 0
            or (
                not registry_fused
                and (
                    row_elems <= 0
                    or block <= 0
                    or not isinstance(conv_layout_sha256, str)
                    or len(conv_layout_sha256) != 64
                    or not isinstance(conv_instance_sha256, str)
                    or len(conv_instance_sha256) != 64
                )
            )
        ):
            raise RuntimeError(
                "FR13 fixed32 forward structural divisors are invalid"
            )
        structural = {
            "schema": (
                "fr13-fixed32-forward-graph-structural-manifest-sfwd-fused-v1"
                if registry_fused
                else "fr13-fixed32-forward-graph-structural-manifest-v1"
            ),
            "batch_size": batch,
            "descriptor_geometry": {
                "physical_drafts": (
                    int(descriptor["num_tokens"]) // batch - 1
                ),
                "verify_rows_per_request": (
                    int(descriptor["num_tokens"]) // batch
                ),
                "verify_rows": int(descriptor["num_tokens"]),
                "model_layers": (
                    len(tree.get("layers", ())) + len(gdn.get("layers", ()))
                ),
            },
            "tree_attention": {
                "layers": len(tree.get("layers", ())),
                "calls_per_event": tree_calls,
                "q_rows_per_call": int(tree.get("q_rows", -1)) // tree_calls,
                "bias_shape": list(tree.get("bias_shape", ())),
                "physical_parent_sha256": gdn.get("parent_sha256"),
            },
            "gdn": {
                "layers": len(gdn.get("layers", ())),
                "scan_calls": scan_calls,
                "launches_per_scan": (
                    int(gdn.get("launches", -1)) // scan_calls
                ),
                "path_programs_per_scan": (
                    int(gdn.get("path_programs", -1)) // scan_calls
                ),
                "padded_slots_per_scan": (
                    int(gdn.get("padded_slots", -1)) // scan_calls
                ),
                "nodes_per_scan": int(gdn.get("nodes", -1)) // scan_calls,
                "critical_path": int(gdn.get("critical_path", -1)),
                "grid_z": list(gdn.get("grid_z", ())),
                "max_path_lengths": list(
                    gdn.get("max_path_lengths", ())
                ),
                "export_or_mask": int(gdn.get("export_or_mask", -1)),
            },
            **({
                "sfwd_conv_postprep": {
                    "route": sfwd.get("route"),
                    "layers": int(sfwd.get("layers", -1)),
                    "requests": int(sfwd.get("requests", -1)),
                    "calls": int(sfwd.get("calls", -1)),
                    "calls_per_layer": int(sfwd.get("calls_per_layer", -1)),
                    "capture_guard": sfwd.get("capture_guard"),
                    "stage_calls": int(sfwd.get("stage_calls", -1)),
                    "consume_calls": int(sfwd.get("consume_calls", -1)),
                    "source_validations": int(
                        sfwd.get("source_validations", -1)
                    ),
                    "freshness_matches": int(
                        sfwd.get("freshness_matches", -1)
                    ),
                    "staged_rows": int(sfwd.get("staged_rows", -1)),
                }
            } if registry_fused else {}),
            **({} if registry_fused else {"conv_pregather": {
                "route": conv.get("route"),
                "stage_calls": int(conv.get("stage_calls", -1)),
                "stage_before_all_consumes": conv.get(
                    "stage_before_all_consumes"
                ),
                "layers": int(conv.get("layers", -1)),
                "requests": int(conv.get("requests", -1)),
                "row_elems": row_elems,
                "block": block,
                "grid": [
                    int(conv.get("layers", -1)),
                    batch,
                    (row_elems + block - 1) // block,
                ],
                "programs": int(conv.get("programs", -1)),
                "ssi_pointer_entries": int(
                    conv.get("ssi_pointer_entries", -1)
                ),
                "ssi_groups": int(conv.get("ssi_groups", -1)),
                "source_validations": len(
                    conv.get("source_validations", ())
                ),
                "staged_rows": int(conv.get("staged_rows", -1)),
                "consume_calls": int(conv.get("consume_calls", -1)),
                "consume_hits": int(conv.get("consume_hits", -1)),
                "consume_fallbacks": int(
                    conv.get("consume_fallbacks", -1)
                ),
                "freshness_matches": int(
                    conv.get("freshness_matches", -1)
                ),
            }}),
        }
        # THE SERVED MODE, not the era default. This audit is on the live
        # generation-1 flush path, so it is the table every serve is measured
        # against; without the mode it states hydra27's for every profile.
        expected = forward_graph_structural_manifest(
            batch,
            kernel_shape=registry_shape,
            mode=_FR13_FIXED32_GDN_MODE or None,
        )
        if structural != expected:
            raise RuntimeError(
                "FR13 fixed32 live forward structure drift: "
                + _fr13_fixed32_drift_detail(structural, expected)
            )
        structural_canonical = __import__("json").dumps(
            structural,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        live_structural_signature = __import__("hashlib").sha256(
            structural_canonical.encode("ascii")
        ).hexdigest()
        _audited_structural_signature = forward_graph_structural_signature(
            batch,
            kernel_shape=registry_shape,
            mode=_FR13_FIXED32_GDN_MODE or None,
        )
        if live_structural_signature != _audited_structural_signature:
            # THE MOST ONE-SIDED REFUSAL IN THE CAMPAIGN: it named neither hash
            # nor a single field, so a corpse could not say WHAT differed. Both
            # digests, and the structural fields that fed each, or it testifies
            # to nothing.
            raise RuntimeError(
                "FR13 fixed32 live forward structural signature drift for mode "
                + repr(_FR13_FIXED32_GDN_MODE)
                + " batch "
                + repr(batch)
                + " kernel_shape "
                + repr(registry_shape)
                + ": observed "
                + repr(live_structural_signature)
                + " against audited "
                + repr(_audited_structural_signature)
                + "; structure "
                + (
                    _fr13_fixed32_drift_detail(structural, expected)
                    or "identical -- the signature inputs differ outside the "
                    "compared structure"
                )
            )
        manifests[batch] = {
            "conv": conv,
            "sfwd": sfwd,
            "kernel_shape": registry_shape,
            "structural_signature": live_structural_signature,
            "conv_layout_sha256": conv_layout_sha256,
        }
    if sorted(manifests) != list(range(1, capacity + 1)):
        raise RuntimeError(
            "FR13 fixed32 live forward registry is not contiguous: "
            + _fr13_fixed32_drift_detail(
                sorted(manifests), list(range(1, capacity + 1))
            )
        )
    def _fr13_f32_registry_row(batch):
        entry = manifests[batch]
        row = {
            "batch_size": batch,
            "graph_signature": entry["structural_signature"],
            "kernel_shape": entry["kernel_shape"],
            "conv_layout_sha256": entry["conv_layout_sha256"],
            "captures": 1,
            "capture_origin": "final_full",
            "measured_replays": normalized_measured[batch],
        }
        if entry["kernel_shape"] == "sfwd_fused_conv_postprep":
            fused = entry["sfwd"]
            # The fused route reports its own class; the subsumed unfused
            # counters are carried as the zeros they are, never omitted.
            row.update(
                fused_calls=int(fused["calls"]),
                fused_layers=int(fused["layers"]),
                stage_calls=int(fused["stage_calls"]),
                stage_before_all_consumes=False,
                layers=int(fused["layers"]),
                requests=int(fused["requests"]),
                row_elems=0,
                programs=0,
                ssi_pointer_entries=0,
                ssi_groups=0,
                source_validations=int(fused["source_validations"]),
                staged_rows=int(fused["staged_rows"]),
                consume_calls=int(fused["consume_calls"]),
                consume_hits=0,
                consume_fallbacks=0,
                freshness_matches=int(fused["freshness_matches"]),
            )
            return row
        row.update(fused_calls=0, fused_layers=0)
        return row
    return [
        {
            **_fr13_f32_registry_row(batch),
            "stage_calls": int(manifests[batch]["conv"]["stage_calls"]),
            "stage_before_all_consumes": manifests[batch]["conv"][
                "stage_before_all_consumes"
            ],
            "layers": int(manifests[batch]["conv"]["layers"]),
            "requests": int(manifests[batch]["conv"]["requests"]),
            "row_elems": int(manifests[batch]["conv"]["row_elems"]),
            "programs": int(manifests[batch]["conv"]["programs"]),
            "ssi_pointer_entries": int(
                manifests[batch]["conv"]["ssi_pointer_entries"]
            ),
            "ssi_groups": int(manifests[batch]["conv"]["ssi_groups"]),
            "source_validations": len(
                manifests[batch]["conv"]["source_validations"]
            ),
            "staged_rows": int(manifests[batch]["conv"]["staged_rows"]),
            "consume_calls": int(manifests[batch]["conv"]["consume_calls"]),
            "consume_hits": int(manifests[batch]["conv"]["consume_hits"]),
            "consume_fallbacks": int(
                manifests[batch]["conv"]["consume_fallbacks"]
            ),
            "freshness_matches": int(
                manifests[batch]["conv"]["freshness_matches"]
            ),
        }
        if manifests[batch]["kernel_shape"] == "unfused_conv_pregather"
        else _fr13_f32_registry_row(batch)
        for batch in range(1, capacity + 1)
    ]


def _fr13_fixed32_capture_begin(
    graph_id,
    runtime_mode,
    num_tokens,
    num_reqs,
    uniform,
    has_lora,
    num_active_loras,
):
    global _FR13_FIXED32_CAPTURE_CONTEXT
    if not _FR13_FIXED32_MODE:
        return
    if (
        _FR13_FIXED32_CAPTURE_FROZEN
        or _FR13_FIXED32_OBSERVED_CURRENT is not None
        or globals().get("_FR13_FIXED32_PENDING_EVENT") is not None
        or _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT is not None
        or _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT is not None
    ):
        raise RuntimeError(
            "FR13 fixed32 full-graph capture/recompile after measurement began"
        )
    if _FR13_FIXED32_CAPTURE_CONTEXT is not None:
        raise RuntimeError("FR13 fixed32 full-graph captures overlapped")
    identity = int(graph_id)
    if identity <= 0 or identity in _FR13_FIXED32_CAPTURE_MANIFESTS:
        raise RuntimeError(
            "FR13 fixed32 full-graph identity was reused: " + str(identity)
        )
    descriptor = _fr13_fixed32_graph_descriptor(
        runtime_mode,
        num_tokens,
        num_reqs,
        uniform,
        has_lora,
        num_active_loras,
    )
    batch = int(descriptor["num_reqs"])
    profile_scope = globals().get("_FR13_FIXED32_PROFILE_CAPTURE_SCOPE")
    if profile_scope is not None:
        if (
            not isinstance(profile_scope, dict)
            or set(profile_scope) != {"descriptor", "graph_id", "completed"}
            or profile_scope.get("descriptor") != descriptor
            or profile_scope.get("graph_id") is not None
            or profile_scope.get("completed") is not False
            or _FR13_FIXED32_CAPTURE_CONTEXT is not None
            or _FR13_FIXED32_CAPTURE_MANIFESTS
            or _FR13_FIXED32_OBSERVED_CURRENT is not None
            or globals().get("_FR13_FIXED32_PENDING_EVENT") is not None
            or _FR13_FIXED32_CAPTURE_FROZEN
        ):
            raise RuntimeError(
                "FR13 fixed32 profile capture begin scope drift: "
                + repr(profile_scope)
            )
        profile_scope["graph_id"] = identity
        return
    if _FR13_FIXED32_PROFILE_MEMORY_SCOPE is not False:
        raise RuntimeError(
            "FR13 fixed32 final capture began inside profile-memory scope"
        )
    _FR13_FIXED32_CAPTURE_CONTEXT = {
        "graph_id": identity,
        "descriptor": descriptor,
        "work": _fr13_fixed32_observed_new_state(
            _FR13_FIXED32_MODE, batch, -1, "capture_manifest"
        ),
    }
    taw_module = __import__("sys").modules.get(
        "_fr13_device_multidraft_kernel"
    )
    if taw_module is None:
        raise RuntimeError("FR13 fixed32 capture begin is missing the TAW module")
    taw_module.fr13_fixed32_cfwd_logit_direct_capture_begin(
        identity, mode=_FR13_FIXED32_MODE, batch_size=batch
    )
    if (
        _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is not None
        and (
            _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            not in ("single_launch", "gqa_group3", "gqa_group3_bv16")
            or batch == _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH
        )
    ):
        tree_kernel = __import__(
            "lumo_flywheel_serving.fr10_gdn_tree_kernel",
            fromlist=(
                "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE",
                "fixed32_gdn_bv_live_capture_begin",
            ),
        )
        if getattr(
            tree_kernel, "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE", None
        ) != _FR13_FIXED32_GDN_PATH_BV_CANDIDATE:
            raise RuntimeError(
                "FR13 fixed32 GDN BV selector drift between observer and "
                "tree kernel"
            )
        tree_kernel.fixed32_gdn_bv_live_capture_begin(identity, batch)
    if _FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB and batch == 4:
        tree_kernel = __import__(
            "lumo_flywheel_serving.fr10_gdn_tree_kernel",
            fromlist=(
                "_fr13_fixed32_batch_gdn_graph_byte_ab_control",
                "fixed32_batch_gdn_graph_live_capture_begin",
            ),
        )
        if not tree_kernel._fr13_fixed32_batch_gdn_graph_byte_ab_control():
            raise RuntimeError(
                "FR13 fixed32 B4 graph GDN selector drift between observer "
                "and tree kernel"
            )
        tree_kernel.fixed32_batch_gdn_graph_live_capture_begin(identity, batch)
    tree_kernel = __import__(
        "lumo_flywheel_serving.fr10_gdn_tree_kernel",
        fromlist=(
            "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION",
            "fixed32_batch_gdn_bv8_production_capture_begin",
            "fixed32_batch_gdn_bv64_production_capture_begin",
        ),
    )
    production_bv = getattr(
        tree_kernel, "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION", None
    )
    if production_bv == 8:
        tree_kernel.fixed32_batch_gdn_bv8_production_capture_begin(
            identity, batch
        )
    elif production_bv is not None:
        tree_kernel.fixed32_batch_gdn_bv64_production_capture_begin(
            identity, batch
        )


def _fr13_fixed32_gdn_bv_expected_records(candidate, batch_size, scan_calls):
    """Per-launch record count the GDN BV live gate must see.

    The work census counts SERVED per-request GDN scans, which is 48 * batch.
    The folded candidates (single_launch, gqa_group3, gqa_group3_bv16) issue
    one physical launch per GDN layer for the whole batch, so they emit 48
    capture records regardless of width, and the live-gate validators demand
    exactly 48 for them. Every other candidate is per-request and keeps the
    raw census. At batch 1 both branches agree, so the sealed B1 arm is
    unchanged. Fail-closed on any census that is not a whole multiple of the
    batch.
    """
    batch = int(batch_size)
    calls = int(scan_calls)
    if candidate not in ("single_launch", "gqa_group3", "gqa_group3_bv16"):
        return calls
    if batch < 1 or calls % batch != 0:
        raise RuntimeError(
            "FR13 fixed32 GDN BV live-gate scan census is not a whole "
            "multiple of the batch: " + repr((calls, batch))
        )
    return calls // batch


def _fr13_fixed32_capture_end(
    graph_id,
    runtime_mode,
    num_tokens,
    num_reqs,
    uniform,
    has_lora,
    num_active_loras,
):
    global _FR13_FIXED32_CAPTURE_CONTEXT
    if not _FR13_FIXED32_MODE:
        return None
    profile_scope = globals().get("_FR13_FIXED32_PROFILE_CAPTURE_SCOPE")
    if profile_scope is not None:
        descriptor = _fr13_fixed32_graph_descriptor(
            runtime_mode,
            num_tokens,
            num_reqs,
            uniform,
            has_lora,
            num_active_loras,
        )
        identity = int(graph_id)
        if (
            not isinstance(profile_scope, dict)
            or set(profile_scope) != {"descriptor", "graph_id", "completed"}
            or profile_scope.get("descriptor") != descriptor
            or profile_scope.get("graph_id") != identity
            or profile_scope.get("completed") is not False
            or _FR13_FIXED32_CAPTURE_CONTEXT is not None
            or _FR13_FIXED32_CAPTURE_MANIFESTS
            or _FR13_FIXED32_OBSERVED_CURRENT is not None
            or globals().get("_FR13_FIXED32_PENDING_EVENT") is not None
            or _FR13_FIXED32_CAPTURE_FROZEN
        ):
            raise RuntimeError(
                "FR13 fixed32 profile capture end scope drift: "
                + repr((identity, descriptor, profile_scope))
            )
        profile_scope["completed"] = True
        return None
    context = _FR13_FIXED32_CAPTURE_CONTEXT
    if _FR13_FIXED32_PROFILE_MEMORY_SCOPE is not False:
        raise RuntimeError(
            "FR13 fixed32 final capture ended inside profile-memory scope"
        )
    if not isinstance(context, dict):
        raise RuntimeError("FR13 fixed32 full-graph capture has no context")
    descriptor = _fr13_fixed32_graph_descriptor(
        runtime_mode,
        num_tokens,
        num_reqs,
        uniform,
        has_lora,
        num_active_loras,
    )
    identity = int(graph_id)
    if (
        identity != int(context["graph_id"])
        or descriptor != context["descriptor"]
    ):
        raise RuntimeError(
            "FR13 fixed32 full-graph capture identity changed: "
            + repr((identity, descriptor, context))
        )
    work = context["work"]
    _fr13_fixed32_validate_forward_work(work, "captured")
    payload = {
        "schema": "fr13-fixed32-forward-graph-manifest-v2",
        "mode": _FR13_FIXED32_MODE,
        "batch_size": int(work["batch_size"]),
        "physical_rows_per_request": 32,
        "descriptor": descriptor,
        "tree_attn": {
            "layers": sorted(work["tree_layers"]),
            "calls": int(work["tree_calls"]),
            "q_rows": int(work["tree_q_rows"]),
            "bias_shape": list(work["tree_bias_shape"]),
        },
        "gdn": {
            "layers": sorted(work["gdn_layers"]),
            "call_pairs": [
                [name, index] for name, index in sorted(work["gdn_calls"])
            ],
            "scan_calls": int(work["gdn_scan_calls"]),
            "launches": int(work["gdn_launches"]),
            "path_programs": int(work["gdn_path_programs"]),
            "padded_slots": int(work["gdn_padded_slots"]),
            "nodes": int(work["gdn_nodes"]),
            "critical_path": int(work["gdn_critical_path"]),
            "grid_z": list(work["gdn_grid_z"]),
            "max_path_lengths": list(work["gdn_max_path_lengths"]),
            "export_or_mask": int(work["gdn_export_or_mask"]),
            "parent_sha256": work["gdn_parent_sha256"],
            "ancestry_sha256": work["gdn_ancestry_sha256"],
        },
        # The fused arm publishes its own section instead: under fusion every
        # conv_* counter is zero and the layout/instance digests are absent,
        # so a conv_pregather section here would be a fabricated one.
        **(
            {
                "sfwd_conv_postprep": {
                    "route": "fused_conv_postprep_single_kernel",
                    "layers": len(work["sfwd_conv_postprep_layers"]),
                    "requests": int(work["batch_size"]),
                    "calls": int(work["sfwd_conv_postprep_calls"]),
                    "calls_per_layer": 1,
                    "capture_guard": True,
                    "fused_layers": sorted(
                        work["sfwd_conv_postprep_layers"]
                    ),
                    "stage_calls": int(work["conv_stage_calls"]),
                    "consume_calls": int(work["conv_consume_calls"]),
                    "source_validations": len(work["conv_source_layers"]),
                    "freshness_matches": int(work["conv_freshness_matches"]),
                    "staged_rows": 0,
                }
            }
            if _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION
            else {}
        ),
        **({} if _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION else {"conv_pregather": {
            "route": "in_graph_preconsume",
            "stage_calls": int(work["conv_stage_calls"]),
            "stage_before_all_consumes": work[
                "conv_stage_before_all_consumes"
            ],
            "stage_layer": work["conv_stage_layer"],
            "layers": int(work["conv_stage_layers"]),
            "requests": int(work["batch_size"]),
            "row_elems": int(work["conv_stage_row_elems"]),
            "block": int(work["conv_stage_block"]),
            "programs": int(work["conv_stage_programs"]),
            "ssi_pointer_entries": int(
                work["conv_stage_ssi_pointer_entries"]
            ),
            "ssi_groups": int(work["conv_stage_ssi_groups"]),
            "staged_rows": int(work["conv_stage_layers"])
            * int(work["batch_size"]),
            "layout_sha256": work["conv_stage_source"],
            "instance_sha256": work["conv_stage_instance"],
            "source_validations": [
                [index, work["conv_source_layers"][index]]
                for index in sorted(work["conv_source_layers"])
            ],
            "consume_layers": sorted(work["conv_consume_layers"]),
            "consume_calls": int(work["conv_consume_calls"]),
            "consume_hits": int(work["conv_consume_hits"]),
            "consume_fallbacks": int(work["conv_consume_fallbacks"]),
            "freshness_matches": int(work["conv_freshness_matches"]),
        }}),
        "kernel_shape": _fr13_fixed32_kernel_shape(),
    }
    canonical = __import__("json").dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    signature = __import__("hashlib").sha256(
        canonical.encode("ascii")
    ).hexdigest()
    if (
        _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is not None
        and (
            _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            not in ("single_launch", "gqa_group3", "gqa_group3_bv16")
            or int(work["batch_size"])
            == _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH
        )
    ):
        tree_kernel = __import__(
            "lumo_flywheel_serving.fr10_gdn_tree_kernel",
            fromlist=(
                "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE",
                "fixed32_gdn_bv_live_capture_end",
            ),
        )
        if getattr(
            tree_kernel, "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE", None
        ) != _FR13_FIXED32_GDN_PATH_BV_CANDIDATE:
            raise RuntimeError(
                "FR13 fixed32 GDN BV selector drift between observer and "
                "tree kernel"
            )
        tree_kernel.fixed32_gdn_bv_live_capture_end(
            identity,
            signature,
            int(work["batch_size"]),
            _fr13_fixed32_gdn_bv_expected_records(
                _FR13_FIXED32_GDN_PATH_BV_CANDIDATE,
                work["batch_size"],
                work["gdn_scan_calls"],
            ),
        )
    if _FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB and int(work["batch_size"]) == 4:
        tree_kernel = __import__(
            "lumo_flywheel_serving.fr10_gdn_tree_kernel",
            fromlist=(
                "_fr13_fixed32_batch_gdn_graph_byte_ab_control",
                "fixed32_batch_gdn_graph_live_capture_end",
            ),
        )
        if not tree_kernel._fr13_fixed32_batch_gdn_graph_byte_ab_control():
            raise RuntimeError(
                "FR13 fixed32 B4 graph GDN selector drift between observer "
                "and tree kernel"
            )
        tree_kernel.fixed32_batch_gdn_graph_live_capture_end(
            identity,
            4,
            signature,
            48,
        )
    tree_kernel = __import__(
        "lumo_flywheel_serving.fr10_gdn_tree_kernel",
        fromlist=(
            "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION",
            "fixed32_batch_gdn_bv8_production_capture_end",
            "fixed32_batch_gdn_bv64_production_capture_end",
        ),
    )
    production_bv = getattr(
        tree_kernel, "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION", None
    )
    if production_bv == 8:
        tree_kernel.fixed32_batch_gdn_bv8_production_capture_end(
            identity,
            int(work["batch_size"]),
            signature,
            int(work["gdn_scan_calls"]),
        )
    elif production_bv is not None:
        tree_kernel.fixed32_batch_gdn_bv64_production_capture_end(
            identity,
            int(work["batch_size"]),
            signature,
            int(work["gdn_scan_calls"]),
        )
    taw_module = __import__("sys").modules.get(
        "_fr13_device_multidraft_kernel"
    )
    if taw_module is None:
        raise RuntimeError("FR13 fixed32 capture end is missing the TAW module")
    taw_module.fr13_fixed32_cfwd_logit_direct_capture_end(
        identity,
        mode=_FR13_FIXED32_MODE,
        batch_size=int(work["batch_size"]),
    )
    _FR13_FIXED32_CAPTURE_MANIFESTS[identity] = (signature, canonical)
    _FR13_FIXED32_CAPTURE_CONTEXT = None
    return signature


def _fr13_fixed32_observed_graph_replay(
    graph_id,
    graph_signature,
    runtime_mode,
    num_tokens,
    num_reqs,
    uniform,
    has_lora,
    num_active_loras,
):
    event = _fr13_fixed32_observed_current("full cudagraph replay")
    if event is None:
        return
    descriptor = _fr13_fixed32_graph_descriptor(
        runtime_mode,
        num_tokens,
        num_reqs,
        uniform,
        has_lora,
        num_active_loras,
    )
    identity = int(graph_id)
    entry = _FR13_FIXED32_CAPTURE_MANIFESTS.get(identity)
    expected_signature, canonical = _fr13_fixed32_manifest_entry(
        entry,
        "forward graph " + str(identity),
    )
    if (
        not isinstance(graph_signature, str)
        or graph_signature != expected_signature
        or event["execution_basis"] != "unbound"
        or int(event["forward_graph_replays"]) != 0
        or int(descriptor["num_reqs"]) != int(event["batch_size"])
    ):
        raise RuntimeError(
            "FR13 fixed32 replay identity/signature/batch drift: "
            + repr(
                (
                    identity,
                    graph_signature,
                    expected_signature,
                    descriptor,
                    event["execution_basis"],
                    event["forward_graph_replays"],
                    event["batch_size"],
                )
            )
        )
    manifest = __import__("json").loads(canonical)
    if (
        manifest.get("schema") != "fr13-fixed32-forward-graph-manifest-v2"
        or manifest.get("mode") != event["mode"]
        or int(manifest.get("batch_size", -1)) != int(event["batch_size"])
        or manifest.get("descriptor") != descriptor
        or int(manifest.get("physical_rows_per_request", -1)) != 32
    ):
        raise RuntimeError("FR13 fixed32 replay manifest semantic drift")
    # The graph begins with the fixed pregather kernel on every replay. Bind
    # that captured stage to the still-live persistent operands and prove no
    # profile, auxiliary, or host-side fixed stage entered the route.
    tree_kernel = __import__(
        "lumo_flywheel_serving.fr10_gdn_tree_kernel",
        fromlist=(
            "_FR13_FIXED32_CONV_PREGATHER",
            "fixed32_conv_col0_pregather_counters",
        ),
    )
    pregather_state = getattr(
        tree_kernel, "_FR13_FIXED32_CONV_PREGATHER", {}
    ).get("state")
    replay_shape = _fr13_fixed32_kernel_shape()
    replay_fused = replay_shape == "sfwd_fused_conv_postprep"
    conv = manifest.get("conv_pregather")
    sfwd_section = manifest.get("sfwd_conv_postprep")
    if manifest.get("kernel_shape") != replay_shape:
        raise RuntimeError(
            "FR13 fixed32 replay manifest kernel shape drifted: "
            + _fr13_fixed32_drift_detail(
                manifest.get("kernel_shape"), replay_shape
            )
        )
    if not isinstance(sfwd_section if replay_fused else conv, dict):
        raise RuntimeError("FR13 fixed32 replay conv manifest is missing")
    pregather_counters = tree_kernel.fixed32_conv_col0_pregather_counters()
    batch_key = int(event["batch_size"])
    captures_by_batch = pregather_counters.get(
        "graph_capture_stages_by_batch"
    )
    if replay_fused:
        # The fused route stages nothing, so there is no runtime staging
        # contract to compare against and every stage counter must be zero.
        if (
            sfwd_section.get("route") != "fused_conv_postprep_single_kernel"
            or int(sfwd_section.get("layers", -1)) != 48
            or int(sfwd_section.get("calls", -1)) != 48
            or int(sfwd_section.get("requests", -1)) != batch_key
            or len(sfwd_section.get("fused_layers", ())) != 48
            or int(sfwd_section.get("stage_calls", -1)) != 0
            or int(sfwd_section.get("consume_calls", -1)) != 0
            or int(sfwd_section.get("source_validations", -1)) != 0
            or int(sfwd_section.get("freshness_matches", -1)) != 0
            or not isinstance(captures_by_batch, dict)
            or int(
                captures_by_batch.get(
                    batch_key, captures_by_batch.get(str(batch_key), -1)
                )
            )
            != 0
            or int(pregather_counters.get("actual_stages", -1)) != 0
            or any(
                int(value) != 0
                for value in pregather_counters.get(
                    "actual_stages_by_batch", {}
                ).values()
            )
            or int(pregather_counters.get("profile_capture_stages", -1)) != 0
            or int(pregather_counters.get("aux_capture_stages", -1)) != 0
        ):
            raise RuntimeError(
                "FR13 fixed32 graph replay fused conv provenance drift: "
                + repr((sfwd_section, pregather_counters))
            )
        live_conv = None
    else:
        live_conv = _fr13_fixed32_conv_runtime_contract(
            pregather_state, event["batch_size"]
        )
    if not replay_fused and (
        conv.get("route") != "in_graph_preconsume"
        or int(conv.get("stage_calls", -1)) != 1
        or conv.get("stage_before_all_consumes") is not True
        or int(conv.get("layers", -1)) != 48
        or int(conv.get("requests", -1)) != batch_key
        or int(conv.get("row_elems", -1)) != int(live_conv["row_elems"])
        or int(conv.get("block", -1)) != int(live_conv["block"])
        or int(conv.get("programs", -1)) != int(live_conv["programs"])
        or int(conv.get("ssi_pointer_entries", -1)) != 48
        or int(conv.get("ssi_groups", -1)) != 3
        or conv.get("layout_sha256") != live_conv["layout_sha256"]
        or conv.get("instance_sha256") != live_conv["instance_sha256"]
        or len(conv.get("source_validations", ())) != 48
        or [int(row[0]) for row in conv.get("source_validations", ())]
        != list(range(48))
        or not isinstance(captures_by_batch, dict)
        or int(
            captures_by_batch.get(
                batch_key, captures_by_batch.get(str(batch_key), -1)
            )
        )
        != 1
        or int(pregather_counters.get("actual_stages", -1)) != 0
        or any(
            int(value) != 0
            for value in pregather_counters.get(
                "actual_stages_by_batch", {}
            ).values()
        )
        or int(pregather_counters.get("profile_capture_stages", -1)) != 0
        or int(pregather_counters.get("aux_capture_stages", -1)) != 0
    ):
        raise RuntimeError(
            "FR13 fixed32 graph replay conv stage provenance drift: "
            + repr(
                (
                    conv,
                    live_conv,
                    pregather_counters,
                )
            )
        )
    tree = manifest["tree_attn"]
    gdn = manifest["gdn"]
    runtime_sys = __import__("sys")
    if "/workspace/scripts" not in runtime_sys.path:
        runtime_sys.path.insert(0, "/workspace/scripts")
    from fr13_fixed32_work_census import forward_graph_structural_signature

    # THE SECOND CALL SITE, and the one that would have killed boot ten: it
    # also hashed hydra27's manifest for every profile. Found by enumerating
    # every call into the authority that omits the mode, not by another boot.
    census_graph_signature = forward_graph_structural_signature(
        int(event["batch_size"]),
        mode=_FR13_FIXED32_GDN_MODE or None,
    )
    if (
        _FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB
        and int(event["batch_size"]) == 4
    ):
        if not tree_kernel._fr13_fixed32_batch_gdn_graph_byte_ab_control():
            raise RuntimeError(
                "FR13 fixed32 B4 graph GDN selector drift between observer "
                "and tree kernel"
            )
        gate_report = tree_kernel.fixed32_batch_gdn_graph_live_gate_on_replay(
            identity,
            expected_signature,
            4,
            48,
        )
        if (
            gate_report.get("status") != "passed"
            or gate_report.get("graph_id") != identity
            or gate_report.get("graph_signature") != expected_signature
            or gate_report.get("batch_size") != 4
            or gate_report.get("records") != 48
        ):
            raise RuntimeError(
                "FR13 fixed32 B4 graph GDN byte gate did not pass on the "
                "authenticated full-graph replay: " + repr(gate_report)
            )
    if (
        _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is not None
        and (
            _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            not in ("single_launch", "gqa_group3", "gqa_group3_bv16")
            or int(event["batch_size"])
            == _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH
        )
    ):
        if getattr(
            tree_kernel, "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE", None
        ) != _FR13_FIXED32_GDN_PATH_BV_CANDIDATE:
            raise RuntimeError(
                "FR13 fixed32 GDN BV selector drift between observer and "
                "tree kernel"
            )
        gate_report = tree_kernel.fixed32_gdn_bv_live_gate_on_replay(
            identity,
            expected_signature,
            census_graph_signature,
            int(event["batch_size"]),
            _fr13_fixed32_gdn_bv_expected_records(
                _FR13_FIXED32_GDN_PATH_BV_CANDIDATE,
                event["batch_size"],
                gdn["scan_calls"],
            ),
            len(globals().get("_FR13_FIXED32_CENSUS_EVENTS", ())),
            int(event["forward_step_index"]),
            tuple(
                __import__("hashlib").sha256(
                    str(request_id).encode("utf-8")
                ).hexdigest()
                for request_id in event["request_ids"]
            ),
        )
        comparison_status = gate_report.get("comparison_status")
        comparator = gate_report.get("comparator")
        if (
            gate_report.get("status") != "armed"
            or comparison_status
            not in {
                "compared_distinct_request_tuple",
                "already_compared_request_tuple",
            }
            or (
                comparison_status == "compared_distinct_request_tuple"
                and not isinstance(comparator, dict)
            )
            or (
                comparison_status == "already_compared_request_tuple"
                and comparator is not None
            )
            or event.get("gdn_comparator") is not None
        ):
            raise RuntimeError(
                "FR13 fixed32 GDN comparator did not return to its armed "
                "state: " + repr(gate_report)
            )
        event["gdn_comparator"] = (
            dict(comparator) if isinstance(comparator, dict) else None
        )
    production_bv = getattr(
        tree_kernel, "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION", None
    )
    if production_bv == 8:
        production_published = getattr(
            tree_kernel,
            "_FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_PUBLISHED",
            False,
        )
        production_replay = (
            tree_kernel.fixed32_batch_gdn_bv8_production_replay_engaged
        )
        production_label = "batched BV8"
    else:
        production_published = getattr(
            tree_kernel,
            "_FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_PUBLISHED",
            False,
        )
        production_replay = (
            tree_kernel.fixed32_batch_gdn_bv64_production_replay_engaged
        )
        production_label = "BV64"
    if production_bv is not None and not production_published:
        production_report = production_replay(
            identity,
            int(event["batch_size"]),
            expected_signature,
            int(gdn["scan_calls"]),
        )
        expected_status = (
            "ENGAGED"
            if int(event["batch_size"]) == 4
            else (
                "batched_lower_batch"
                if production_bv == 8 and int(event["batch_size"]) >= 2
                else "legacy_lower_batch"
            )
        )
        if production_report.get("status") != expected_status:
            raise RuntimeError(
                f"FR13 fixed32 {production_label} production replay did not "
                "engage its "
                "batch-qualified graph: " + repr(production_report)
            )
    if _FR13_FIXED32_TAW_NATIVE_PRECOMPUTE:
        taw_module = __import__("sys").modules.get(
            "_fr13_device_multidraft_kernel"
        )
        if taw_module is None:
            raise RuntimeError(
                "FR13 fixed32 TAW native replay gate is missing its module"
            )
        _fr13_fixed32_taw_full_graph_on_replay(
            taw_module, event["mode"], int(event["batch_size"])
        )
    event["tree_layers"] = set(tree["layers"])
    event["tree_calls"] = int(tree["calls"])
    event["tree_q_rows"] = int(tree["q_rows"])
    event["tree_bias_shape"] = tuple(tree["bias_shape"])
    event["gdn_layers"] = set(gdn["layers"])
    event["gdn_calls"] = {
        (str(name), int(index)) for name, index in gdn["call_pairs"]
    }
    event["gdn_scan_calls"] = int(gdn["scan_calls"])
    event["gdn_launches"] = int(gdn["launches"])
    event["gdn_path_programs"] = int(gdn["path_programs"])
    event["gdn_padded_slots"] = int(gdn["padded_slots"])
    event["gdn_nodes"] = int(gdn["nodes"])
    event["gdn_critical_path"] = int(gdn["critical_path"])
    event["gdn_grid_z"] = tuple(gdn["grid_z"])
    event["gdn_max_path_lengths"] = tuple(gdn["max_path_lengths"])
    event["gdn_export_or_mask"] = int(gdn["export_or_mask"])
    event["gdn_parent_sha256"] = gdn["parent_sha256"]
    event["gdn_ancestry_sha256"] = gdn["ancestry_sha256"]
    if replay_fused:
        # Restore the fused class from the manifest and leave every subsumed
        # conv_* counter at the zero the fused arm actually produced. The
        # previous unconditional restore fabricated 48 consumes and a staging
        # digest that no kernel in this arm ever performed.
        event["sfwd_conv_postprep_calls"] = int(sfwd_section["calls"])
        event["sfwd_conv_postprep_layers"] = set(
            sfwd_section["fused_layers"]
        )
        event["conv_stage_calls"] = int(sfwd_section["stage_calls"])
        event["conv_stage_replays"] = 0
        event["conv_stage_before_all_consumes"] = False
        event["conv_stage_layer"] = None
        event["conv_stage_layers"] = 0
        event["conv_stage_row_elems"] = 0
        event["conv_stage_block"] = 0
        event["conv_stage_programs"] = 0
        event["conv_stage_ssi_pointer_entries"] = 0
        event["conv_stage_ssi_groups"] = 0
        event["conv_stage_source"] = None
        event["conv_stage_instance"] = None
        event["conv_source_layers"] = {}
        event["conv_consume_layers"] = set()
        event["conv_consume_calls"] = int(sfwd_section["consume_calls"])
        event["conv_consume_hits"] = 0
        event["conv_consume_fallbacks"] = 0
        event["conv_freshness_matches"] = int(
            sfwd_section["freshness_matches"]
        )
    else:
        event["sfwd_conv_postprep_calls"] = 0
        event["sfwd_conv_postprep_layers"] = set()
        event["conv_stage_calls"] = int(conv["stage_calls"])
        event["conv_stage_replays"] = 1
        event["conv_stage_before_all_consumes"] = conv[
            "stage_before_all_consumes"
        ]
        event["conv_stage_layer"] = conv["stage_layer"]
        event["conv_stage_layers"] = int(conv["layers"])
        event["conv_stage_row_elems"] = int(conv["row_elems"])
        event["conv_stage_block"] = int(conv["block"])
        event["conv_stage_programs"] = int(conv["programs"])
        event["conv_stage_ssi_pointer_entries"] = int(
            conv["ssi_pointer_entries"]
        )
        event["conv_stage_ssi_groups"] = int(conv["ssi_groups"])
        event["conv_stage_source"] = conv["layout_sha256"]
        event["conv_stage_instance"] = conv["instance_sha256"]
        event["conv_source_layers"] = {
            int(index): str(name)
            for index, name in conv["source_validations"]
        }
        event["conv_consume_layers"] = set(conv["consume_layers"])
        event["conv_consume_calls"] = int(conv["consume_calls"])
        event["conv_consume_hits"] = int(conv["consume_hits"])
        event["conv_consume_fallbacks"] = int(conv["consume_fallbacks"])
        event["conv_freshness_matches"] = int(conv["freshness_matches"])
    event["execution_basis"] = "cudagraph_full_replay"
    event["forward_graph_id"] = identity
    event["forward_graph_signature"] = expected_signature
    event["forward_graph_replays"] = 1
    _fr13_fixed32_validate_forward_work(event, "replayed")
    _FR13_FIXED32_GRAPH_REPLAY_EVIDENCE.append(
        {
            "mode": event["mode"],
            "batch_size": int(event["batch_size"]),
            "forward_step_index": int(event["forward_step_index"]),
            "graph_id": identity,
            "graph_signature": expected_signature,
            "matching_replays": 1,
            "conv_pregather_stage_replays": 0 if replay_fused else 1,
            "kernel_shape": replay_shape,
            "event_index": None,
            "event_complete": False,
        }
    )
    tree_kernel.fixed32_conv_zero_tail_live_prepare_replay(
        mode=event["mode"],
        batch_size=int(event["batch_size"]),
        enabled=False,
    )


def _fr13_fixed32_drafter_proposal_begin(
    mode,
    request_ids,
    sampled_rows,
    metadata_num_reqs,
    metadata_batch_size,
):
    global _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    if not _FR13_FIXED32_MODE:
        return
    req_ids = tuple(str(value) for value in request_ids)
    batch = len(req_ids)
    if (
        mode != _FR13_FIXED32_MODE
        or batch not in (1, 2, 3, 4)
        or int(sampled_rows) != batch
        or int(metadata_num_reqs) != batch
        or int(metadata_batch_size) != batch
        or any(not value for value in req_ids)
        or len(set(req_ids)) != batch
        or _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT is not None
        or _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT is not None
        or _FR13_FIXED32_OBSERVED_CURRENT is not None
        or _FR13_FIXED32_CAPTURE_CONTEXT is not None
        or _FR13_FIXED32_PROFILE_CAPTURE_SCOPE is not None
        or _FR13_FIXED32_PROFILE_MEMORY_SCOPE is not False
    ):
        raise RuntimeError(
            "FR13 fixed32 drafter proposal begin drift: "
            + repr(
                (
                    mode,
                    req_ids,
                    sampled_rows,
                    metadata_num_reqs,
                    metadata_batch_size,
                )
            )
        )
    pending = globals().get("_FR13_FIXED32_PENDING_EVENT")
    if pending is not None and not isinstance(pending, dict):
        raise RuntimeError("FR13 fixed32 pending event is malformed")
    measured = isinstance(pending, dict)
    if measured:
        if (
            pending.get("target_kv_complete") is not True
            or pending.get("drafter_kv_complete") is not None
            or pending.get("kv_complete") is not None
        ):
            raise RuntimeError(
                "FR13 fixed32 proposal began outside the split KV lifecycle"
            )
        expected_identity = (
            pending.get("mode"),
            int(pending.get("batch_size", -1)),
            tuple(pending.get("request_ids", ())),
        )
        actual_identity = (mode, batch, req_ids)
        if expected_identity != actual_identity:
            raise RuntimeError(
                "FR13 fixed32 verify/proposer request identity drift: "
                + repr((expected_identity, actual_identity))
            )
    proposal = {
        "mode": mode,
        "batch_size": batch,
        "request_ids": req_ids,
        "measured": measured,
        "forward_step_index": (
            int(pending.get("forward_step_index", -1))
            if measured
            else -1
        ),
        "mtp_execution_basis": "unbound",
        "mtp_forward_calls": 0,
        "mtp_forward_rows": 0,
        "graph_id": None,
        "graph_signature": None,
        "graph_replays": 0,
        "graph_captures": 0,
        "arctic": None,
        "publish": None,
    }
    if measured:
        evidence = {
            "mode": mode,
            "batch_size": batch,
            "request_ids": req_ids,
            "forward_step_index": int(proposal["forward_step_index"]),
            "proposal_begins": 1,
            "proposal_ends": 0,
            "graph_id": None,
            "graph_signature": None,
            "graph_captures": 0,
            "matching_replays": 0,
            "event_index": None,
            "event_complete": False,
        }
        _FR13_FIXED32_DRAFTER_REPLAY_EVIDENCE.append(evidence)
        proposal["replay_evidence"] = evidence
    _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT = proposal
    return measured


def _fr13_dfwd_unified_bm8_production_begin(graph_id, batch_size):
    """Arm BM8 only for the attested final fixed32 B1 drafter capture."""
    _os = __import__("os")
    if _os.environ.get("FR13_DFWD_UNIFIED_BM8_PRODUCTION", "0") != "1":
        return
    if _os.environ.get("FR13_DFWD_UNIFIED_BM8_LIVE_AB", "0") != "0":
        raise RuntimeError("FR13 DFWD unified BM8 live A/B leaked into production")
    if (
        _os.environ.get(
            "FR13_DFWD_UNIFIED_BM8_INTERNAL_PRODUCTION_ATTESTED"
        )
        != "1"
    ):
        raise RuntimeError("FR13 DFWD unified BM8 production is not attested")
    if _os.environ.get("FR13_DFWD_UNIFIED_BM8_INTERNAL") is not None:
        raise RuntimeError("FR13 DFWD unified BM8 internal selector leaked")
    context = _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT
    proposal = _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    identity = int(graph_id)
    batch = int(batch_size)
    if (
        batch != 1
        or not isinstance(context, dict)
        or int(context.get("graph_id", 0)) != identity
        or int(context.get("batch_size", -1)) != 1
        or context.get("mode") != _FR13_FIXED32_MODE
        or not isinstance(proposal, dict)
        or proposal.get("measured") is not True
        or int(proposal.get("batch_size", -1)) != 1
        or proposal.get("mode") != _FR13_FIXED32_MODE
        or _FR13_FIXED32_CAPTURE_FROZEN is not True
    ):
        raise RuntimeError("FR13 DFWD unified BM8 production is not final B1")

    full_b1 = []
    for target_graph_id, entry in _FR13_FIXED32_CAPTURE_MANIFESTS.items():
        signature, canonical = _fr13_fixed32_manifest_entry(
            entry, "BM8 target graph " + str(target_graph_id)
        )
        manifest = __import__("json").loads(canonical)
        descriptor = manifest.get("descriptor")
        if (
            manifest.get("schema") == "fr13-fixed32-forward-graph-manifest-v2"
            and manifest.get("mode") == _FR13_FIXED32_MODE
            and int(manifest.get("batch_size", -1)) == 1
            and int(manifest.get("physical_rows_per_request", -1)) == 32
            and isinstance(descriptor, dict)
            and descriptor.get("runtime_mode") == "FULL"
            and int(descriptor.get("num_tokens", -1)) == 32
            and int(descriptor.get("num_reqs", -1)) == 1
            and descriptor.get("uniform") is True
            and descriptor.get("has_lora") is False
            and int(descriptor.get("num_active_loras", -1)) == 0
        ):
            full_b1.append((int(target_graph_id), signature))
    if len(full_b1) != 1:
        raise RuntimeError(
            "FR13 DFWD unified BM8 requires one final FULL fixed32 B1 graph"
        )

    sidecar_sha256 = _os.environ.get(
        "FR13_DFWD_UNIFIED_BM8_PRODUCTION_PASS_SIDECAR_SHA256", ""
    )
    source_sha256 = _os.environ.get(
        "FR13_DFWD_UNIFIED_BM8_QUALIFIED_SOURCE_SHA256", ""
    )
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (sidecar_sha256, source_sha256)
    ):
        raise RuntimeError("FR13 DFWD unified BM8 attestation digest drifted")
    unified = __import__(
        "vllm.v1.attention.ops.triton_unified_attention",
        fromlist=("_FR13_DFWD_UNIFIED_BM8_DISPATCHES",),
    )
    source_path = __import__("pathlib").Path(unified.__file__).resolve()
    actual_source_sha256 = __import__("hashlib").sha256(
        source_path.read_bytes()
    ).hexdigest()
    if actual_source_sha256 != source_sha256:
        raise RuntimeError("FR13 DFWD unified BM8 qualified source drifted")
    dispatches = getattr(unified, "_FR13_DFWD_UNIFIED_BM8_DISPATCHES", None)
    if type(dispatches) is not int or dispatches < 0:
        raise RuntimeError("FR13 DFWD unified BM8 dispatch counter drifted")
    context["bm8_production"] = {
        "dispatches_before": dispatches,
        "guarded_calls": 0,
        "qualified_source_sha256": source_sha256,
        "pass_sidecar_sha256": sidecar_sha256,
        "target_graph_id": full_b1[0][0],
        "target_graph_signature": full_b1[0][1],
    }


def _fr13_dfwd_unified_bm8_production_end(
    graph_id, batch_size, graph_signature
):
    _os = __import__("os")
    if _os.environ.get("FR13_DFWD_UNIFIED_BM8_PRODUCTION", "0") != "1":
        return
    context = _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT
    identity = int(graph_id)
    batch = int(batch_size)
    production = (
        context.get("bm8_production") if isinstance(context, dict) else None
    )
    try:
        if (
            batch != 1
            or not isinstance(context, dict)
            or int(context.get("graph_id", 0)) != identity
            or not isinstance(production, dict)
            or _os.environ.get("FR13_DFWD_UNIFIED_BM8_INTERNAL") is not None
        ):
            raise RuntimeError(
                "FR13 DFWD unified BM8 production capture scope drifted"
            )
        unified = __import__(
            "vllm.v1.attention.ops.triton_unified_attention",
            fromlist=("_FR13_DFWD_UNIFIED_BM8_DISPATCHES",),
        )
        dispatches = getattr(
            unified, "_FR13_DFWD_UNIFIED_BM8_DISPATCHES", None
        )
        dispatches_before = production.get("dispatches_before")
        if (
            type(dispatches) is not int
            or type(dispatches_before) is not int
            or dispatches - dispatches_before != 4
            or int(production.get("guarded_calls", -1)) != 4
        ):
            raise RuntimeError(
                "FR13 DFWD unified BM8 production did not capture four calls"
            )
        record = {
            "schema": "fr13.fixed32.dfwd_unified_bm8_production_capture.v1",
            "status": "CAPTURED_PENDING_REPLAY",
            "runtime_mode": "FULL",
            "batch_size": 1,
            "physical_rows_per_request": 32,
            "candidate": {
                "kernel": "kernel_unified_attention_2d",
                "block_m": 8,
                "block_q": 1,
                "calls": 4,
            },
            "dispatch": "BM8 exact B1 geometry; no fallback",
            "drafter_graph_id": identity,
            "drafter_graph_signature": str(graph_signature),
            "target_graph_id": production["target_graph_id"],
            "target_graph_signature": production["target_graph_signature"],
            "qualified_source_sha256": production[
                "qualified_source_sha256"
            ],
            "pass_sidecar_sha256": production["pass_sidecar_sha256"],
        }
        if identity in _FR13_DFWD_UNIFIED_BM8_PRODUCTION_PENDING:
            raise RuntimeError(
                "FR13 DFWD unified BM8 production capture duplicated"
            )
        _FR13_DFWD_UNIFIED_BM8_PRODUCTION_PENDING[identity] = record
    finally:
        _os.environ.pop("FR13_DFWD_UNIFIED_BM8_INTERNAL", None)


def _fr13_dfwd_unified_bm8_production_replay_installed(
    graph_id, batch_size, graph_signature
):
    _os = __import__("os")
    if _os.environ.get("FR13_DFWD_UNIFIED_BM8_PRODUCTION", "0") != "1":
        return
    identity = int(graph_id)
    batch = int(batch_size)
    record = _FR13_DFWD_UNIFIED_BM8_PRODUCTION_PENDING.get(identity)
    lifecycle = _FR13_FIXED32_DRAFTER_GRAPH_LIFECYCLE.get(identity)
    proposal = _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    if (
        batch != 1
        or not isinstance(record, dict)
        or record.get("status") != "CAPTURED_PENDING_REPLAY"
        or record.get("drafter_graph_signature") != str(graph_signature)
        or not isinstance(lifecycle, dict)
        or int(lifecycle.get("captures", -1)) != 1
        or int(lifecycle.get("measured_replays", -1)) != 1
        or int(lifecycle.get("unmeasured_replays", -1)) != 0
        or not isinstance(proposal, dict)
        or proposal.get("measured") is not True
        or int(proposal.get("graph_id", 0)) != identity
        or proposal.get("graph_signature") != graph_signature
        or int(proposal.get("graph_replays", -1)) != 1
    ):
        raise RuntimeError(
            "FR13 DFWD unified BM8 production replay/install drifted"
        )
    path = __import__("pathlib").Path(
        _os.environ.get(
            "FR13_DFWD_UNIFIED_BM8_PRODUCTION_CAPTURE_JSON",
            "/logs/fr13_dfwd_unified_bm8.production_capture.json",
        )
    )
    temporary = path.with_name(path.name + ".tmp")
    published = dict(record)
    published["status"] = "ENGAGED"
    temporary.write_text(
        __import__("json").dumps(
            published, ensure_ascii=True, indent=2, sort_keys=True
        )
        + "\n",
        encoding="ascii",
    )
    temporary.replace(path)
    del _FR13_DFWD_UNIFIED_BM8_PRODUCTION_PENDING[identity]


def _fr13_fixed32_drafter_graph_capture_begin(
    graph_id, batch_size, passes=4, segment=0
):
    # FR14_GATE_SPLIT_GRAPH: `passes` is the number of post-root MTP forwards
    # this graph records. It is 4 for the single shipped graph and 2 for each
    # half of the split capture; the registry is keyed by (batch, passes) so the
    # two halves coexist. Every existing call site omits it and is unchanged.
    global _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT
    if not _FR13_FIXED32_MODE:
        return
    proposal = _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    batch = int(batch_size)
    identity = int(graph_id)
    passes = int(passes)
    segment = int(segment)
    # `segment` orders the halves of a split capture: 0 = `lo`, 1 = `hi`. It is
    # part of both the registry key and the manifest payload, so the two halves
    # are distinguishable artifacts rather than two graphs sharing one signature.
    if passes not in (2, 4) or segment not in (0, 1) or (
        passes == 4 and segment != 0
    ):
        raise RuntimeError(
            "FR13 fixed32 drafter graph shape must be 4x1 or 2x2: "
            + repr((passes, segment))
        )
    if (
        not isinstance(proposal, dict)
        or batch != int(proposal["batch_size"])
        or proposal.get("mode") != _FR13_FIXED32_MODE
        or proposal.get("mtp_execution_basis") != "unbound"
        or int(proposal.get("mtp_forward_calls", -1)) != 0
        or int(proposal.get("mtp_forward_rows", -1)) != 0
        or int(proposal.get("graph_replays", -1)) != 0
        # a split capture records `lo` then `hi` inside ONE proposal, so the
        # second capture legitimately sees graph_captures == 1
        or int(proposal.get("graph_captures", -1)) not in (
            (0,) if passes == 4 else (0, 1)
        )
        or identity <= 0
        or _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT is not None
        or _FR13_FIXED32_OBSERVED_CURRENT is not None
        or _FR13_FIXED32_CAPTURE_CONTEXT is not None
        or _FR13_FIXED32_PROFILE_CAPTURE_SCOPE is not None
        or _FR13_FIXED32_PROFILE_MEMORY_SCOPE is not False
        or identity in _FR13_FIXED32_DRAFTER_GRAPH_MANIFESTS
        or (batch, passes, segment) in _FR13_FIXED32_DRAFTER_GRAPH_BY_BATCH
    ):
        raise RuntimeError(
            "FR13 fixed32 lazy/duplicate drafter graph capture: "
            + repr((identity, batch))
        )
    _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT = {
        "graph_id": identity,
        "mode": proposal["mode"],
        "batch_size": batch,
        "passes": passes,
        "segment": segment,
        "request_ids": proposal["request_ids"],
        "capturing": True,
        "mtp_forward_calls": 0,
        "mtp_forward_rows": 0,
        "draft_head_fp8_calls": 0,
        "draft_head_fp8_rows": 0,
        "draft_head_u8_calls": 0,
        "draft_head_u8_rows": 0,
        "tree_attn_calls": 0,
        "tree_attn_rows": 0,
        "tree_attn_layer": None,
        "tree_attn_bias_shape": None,
    }
    proposal["graph_captures"] = int(proposal.get("graph_captures", 0)) + 1
    _fr13_dfwd_unified_bm8_production_begin(identity, batch)


def _fr13_fixed32_drafter_mtp_forward(batch_size, capturing):
    batch = int(batch_size)
    if bool(capturing):
        context = _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT
        if (
            not isinstance(context, dict)
            or batch != int(context["batch_size"])
            or context.get("capturing") is not True
            or int(context.get("tree_attn_calls", -1))
            != int(context.get("mtp_forward_calls", -1)) + 1
            or int(context.get("tree_attn_rows", -1))
            != int(context.get("mtp_forward_rows", -1)) + batch
            or context.get("tree_attn_layer")
            != _FR13_FIXED32_DRAFTER_TREE_LAYER
            or context.get("tree_attn_bias_shape") != (1, 1)
        ):
            raise RuntimeError(
                "FR13 fixed32 unscoped drafter capture MTP call"
            )
        context["mtp_forward_calls"] += 1
        context["mtp_forward_rows"] += batch
        return
    proposal = _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    if (
        not isinstance(proposal, dict)
        or batch != int(proposal["batch_size"])
        or proposal["mtp_execution_basis"] not in ("unbound", "eager_direct")
    ):
        raise RuntimeError("FR13 fixed32 eager drafter MTP call drift")
    proposal["mtp_execution_basis"] = "eager_direct"
    proposal["mtp_forward_calls"] += 1
    proposal["mtp_forward_rows"] += batch


def _fr13_fixed32_drafter_fp8_head_selection(batch_size):
    """Classify an FP8 head call from the authenticated graph scope."""
    batch = int(batch_size)
    context = _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT
    if context is None:
        return False
    if not isinstance(context, dict):
        raise RuntimeError(
            "FR13 fixed32 drafter FP8 head has invalid capture context"
        )
    head_calls = int(context.get("draft_head_fp8_calls", -1))
    head_rows = int(context.get("draft_head_fp8_rows", -1))
    if (
        batch != int(context.get("batch_size", -1))
        or context.get("capturing") is not True
        or head_calls < 0
        or head_rows < 0
        or int(context.get("mtp_forward_calls", -1)) != head_calls + 1
        or int(context.get("mtp_forward_rows", -1)) != head_rows + batch
    ):
        raise RuntimeError(
            "FR13 fixed32 drafter FP8 head left capture lifecycle: "
            + repr((batch, context))
        )
    context["draft_head_fp8_calls"] = head_calls + 1
    context["draft_head_fp8_rows"] = head_rows + batch
    return True


def _fr13_fixed32_drafter_u8_head_selection(batch_size):
    """Classify one credentialed U8 serving call in the fixed B1 graph."""
    batch = int(batch_size)
    proposal = _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    context = _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT
    if (
        not isinstance(proposal, dict)
        or batch != 1
        or int(proposal.get("batch_size", -1)) != 1
        or proposal.get("mode") != _FR13_FIXED32_MODE
        or proposal.get("measured") not in (True, False)
    ):
        raise RuntimeError("FR13 draft-head U8 production has no exact B1 proposal")
    if context is None:
        if (
            proposal.get("mtp_execution_basis") != "unbound"
            or int(proposal.get("mtp_forward_calls", -1)) != 0
            or int(proposal.get("mtp_forward_rows", -1)) != 0
        ):
            raise RuntimeError("FR13 draft-head U8 production root lifecycle drifted")
        return False
    if not isinstance(context, dict):
        raise RuntimeError("FR13 draft-head U8 production capture is malformed")
    calls = int(context.get("draft_head_u8_calls", -1))
    rows = int(context.get("draft_head_u8_rows", -1))
    if (
        context.get("capturing") is not True
        or int(context.get("batch_size", -1)) != 1
        or context.get("mode") != _FR13_FIXED32_MODE
        or calls < 0
        or rows < 0
        or int(context.get("mtp_forward_calls", -1)) != calls + 1
        or int(context.get("mtp_forward_rows", -1)) != rows + 1
        or calls not in (0, 1, 2, 3)
    ):
        raise RuntimeError(
            "FR13 draft-head U8 production left capture lifecycle: "
            + repr((batch, context))
        )
    context["draft_head_u8_calls"] = calls + 1
    context["draft_head_u8_rows"] = rows + 1
    return True


def _fr13_fixed32_drafter_graph_capture_end(
    graph_id, batch_size, passes=4, segment=0
):
    global _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT
    if not _FR13_FIXED32_MODE:
        return None
    context = _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT
    identity = int(graph_id)
    batch = int(batch_size)
    passes = int(passes)
    segment = int(segment)
    u8_enabled = any(
        __import__("os").environ.get(key, "0") == "1"
        for key in (
            "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB",
            "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION",
            "FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB",
        )
    )
    if (
        not isinstance(context, dict)
        or identity != int(context["graph_id"])
        or batch != int(context["batch_size"])
        or int(context.get("passes", -1)) != passes
        or int(context.get("segment", -1)) != segment
        or int(context["mtp_forward_calls"]) != passes
        or int(context["mtp_forward_rows"]) != passes * batch
        or int(context.get("tree_attn_calls", -1)) != passes
        or int(context.get("tree_attn_rows", -1)) != passes * batch
        or context.get("tree_attn_layer")
        != _FR13_FIXED32_DRAFTER_TREE_LAYER
        or context.get("tree_attn_bias_shape") != (1, 1)
        or int(context.get("draft_head_u8_calls", -1))
        != (passes if u8_enabled else 0)
        or int(context.get("draft_head_u8_rows", -1))
        != (passes * batch if u8_enabled else 0)
        or context.get("capturing") is not True
    ):
        raise RuntimeError(
            "FR13 fixed32 drafter graph capture work drift: "
            + repr((identity, batch, context))
        )
    payload = {
        "schema": "fr13-fixed32-drafter-graph-manifest-v2",
        "batch_size": batch,
        "mtp_forward_calls": passes,
        "mtp_forward_rows": passes * batch,
        "tree_attn_calls": passes,
        "tree_attn_rows": passes * batch,
        "tree_attn_layer": _FR13_FIXED32_DRAFTER_TREE_LAYER,
        "tree_attn_bias_shape": [1, 1],
    }
    if passes != 4:
        # FR14_GATE_SPLIT_GRAPH: a half-graph is a DIFFERENT artifact from the
        # shipped 4-pass one and must never be able to present its signature.
        payload["schema"] = "fr13-fixed32-drafter-graph-manifest-v3-split"
        payload["split_passes"] = passes
        payload["split_segment"] = segment
    canonical = __import__("json").dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    signature = __import__("hashlib").sha256(
        canonical.encode("ascii")
    ).hexdigest()
    _FR13_FIXED32_DRAFTER_GRAPH_MANIFESTS[identity] = (
        signature,
        canonical,
    )
    _FR13_FIXED32_DRAFTER_GRAPH_BY_BATCH[(batch, passes, segment)] = identity
    proposal = _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    if not isinstance(proposal, dict):
        raise RuntimeError(
            "FR13 fixed32 drafter proposal disappeared at capture end"
        )
    _FR13_FIXED32_DRAFTER_GRAPH_LIFECYCLE[identity] = {
        "batch_size": batch,
        "passes": passes,
        "segment": segment,
        "graph_signature": signature,
        "captures": 1,
        "mtp_forward_calls": int(context["mtp_forward_calls"]),
        "mtp_forward_rows": int(context["mtp_forward_rows"]),
        "draft_head_fp8_calls": int(
            context.get("draft_head_fp8_calls", -1)
        ),
        "draft_head_fp8_rows": int(
            context.get("draft_head_fp8_rows", -1)
        ),
        "draft_head_u8_calls": int(
            context.get("draft_head_u8_calls", -1)
        ),
        "draft_head_u8_rows": int(
            context.get("draft_head_u8_rows", -1)
        ),
        "capture_origin": (
            "measured" if proposal["measured"] else "unmeasured"
        ),
        "measured_replays": 0,
        "unmeasured_replays": 0,
    }
    _fr13_dfwd_unified_bm8_production_end(identity, batch, signature)
    _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT = None
    proposal["captured_graph_id"] = identity
    proposal["captured_graph_signature"] = signature
    return signature


def _fr13_fixed32_drafter_graph_replay(
    graph_id, graph_signature, batch_size, passes=4, segment=0
):
    proposal = _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    identity = int(graph_id)
    batch = int(batch_size)
    passes = int(passes)
    segment = int(segment)
    if not isinstance(proposal, dict):
        raise RuntimeError("FR13 fixed32 drafter replay has no proposal")
    expected_signature, canonical = _fr13_fixed32_manifest_entry(
        _FR13_FIXED32_DRAFTER_GRAPH_MANIFESTS.get(identity),
        "drafter graph " + str(identity),
    )
    manifest = __import__("json").loads(canonical)
    # FR14_GATE_SPLIT_GRAPH: an ungated step replays `lo` then `hi`, so the
    # second call legitimately arrives with the basis already bound and a
    # non-zero running count. The totals are still pinned -- proposal_end
    # requires exactly (4 calls, 2 replays) split / (4, 1) single / (2, 1) gated
    # -- so accumulation cannot hide a missing or extra forward.
    _fr13_prior_replays = int(proposal["graph_replays"])
    _fr13_prior_calls = int(proposal["mtp_forward_calls"])
    if (
        graph_signature != expected_signature
        or _FR13_FIXED32_DRAFTER_GRAPH_BY_BATCH.get((batch, passes, segment))
        != identity
        or manifest.get("schema")
        != (
            "fr13-fixed32-drafter-graph-manifest-v2"
            if passes == 4
            else "fr13-fixed32-drafter-graph-manifest-v3-split"
        )
        or int(manifest.get("batch_size", -1)) != batch
        or int(manifest.get("mtp_forward_calls", -1)) != passes
        or int(manifest.get("mtp_forward_rows", -1)) != passes * batch
        or int(manifest.get("tree_attn_calls", -1)) != passes
        or int(manifest.get("tree_attn_rows", -1)) != passes * batch
        or manifest.get("tree_attn_layer")
        != _FR13_FIXED32_DRAFTER_TREE_LAYER
        or manifest.get("tree_attn_bias_shape") != [1, 1]
        or int(manifest.get("split_segment", 0)) != segment
        or batch != int(proposal["batch_size"])
        or proposal["mtp_execution_basis"] != (
            "unbound" if _fr13_prior_replays == 0 else "cudagraph_replay"
        )
        or _fr13_prior_calls != _fr13_prior_replays * passes
        # segment ordering IS the replay-count invariant: `lo` must be replayed
        # before `hi`, and each exactly once
        or _fr13_prior_replays != segment
    ):
        raise RuntimeError(
            "FR13 fixed32 drafter graph replay drift: "
            + repr(
                (
                    identity,
                    graph_signature,
                    expected_signature,
                    batch,
                    proposal,
                )
            )
        )
    proposal["mtp_execution_basis"] = "cudagraph_replay"
    proposal["mtp_forward_calls"] = _fr13_prior_calls + passes
    proposal["mtp_forward_rows"] = (_fr13_prior_calls + passes) * batch
    proposal["graph_id"] = identity
    proposal["graph_signature"] = expected_signature
    proposal["graph_replays"] = _fr13_prior_replays + 1
    lifecycle = _FR13_FIXED32_DRAFTER_GRAPH_LIFECYCLE.get(identity)
    if (
        not isinstance(lifecycle, dict)
        or int(lifecycle.get("captures", -1)) != 1
        or int(lifecycle.get("batch_size", -1)) != batch
        or int(lifecycle.get("passes", -1)) != passes
        or int(lifecycle.get("segment", -1)) != segment
        or lifecycle.get("graph_signature") != expected_signature
    ):
        raise RuntimeError(
            "FR13 fixed32 drafter graph lifecycle is missing"
        )
    lifecycle[
        "measured_replays" if proposal["measured"] else "unmeasured_replays"
    ] += 1
    if proposal["measured"]:
        evidence = proposal.get("replay_evidence")
        if (
            not isinstance(evidence, dict)
            or int(evidence.get("proposal_begins", -1)) != 1
            or int(evidence.get("matching_replays", -1))
            != _fr13_prior_replays
        ):
            raise RuntimeError(
                "FR13 fixed32 drafter replay evidence is missing"
            )
        evidence["graph_id"] = identity
        evidence["graph_signature"] = expected_signature
        evidence["graph_captures"] = int(proposal["graph_captures"])
        evidence["matching_replays"] = _fr13_prior_replays + 1


def _fr13_fixed32_drafter_graph_registry():
    rows = []
    for _key in sorted(_FR13_FIXED32_DRAFTER_GRAPH_BY_BATCH):
        batch, _key_passes, _key_segment = _key
        graph_id = _FR13_FIXED32_DRAFTER_GRAPH_BY_BATCH[_key]
        signature, _canonical = _fr13_fixed32_manifest_entry(
            _FR13_FIXED32_DRAFTER_GRAPH_MANIFESTS.get(graph_id),
            "drafter graph " + str(graph_id),
        )
        lifecycle = _FR13_FIXED32_DRAFTER_GRAPH_LIFECYCLE.get(graph_id)
        if (
            not isinstance(lifecycle, dict)
            or int(lifecycle.get("batch_size", -1)) != int(batch)
            or int(lifecycle.get("passes", -1)) != int(_key_passes)
            or int(lifecycle.get("segment", -1)) != int(_key_segment)
            or lifecycle.get("graph_signature") != signature
            or int(lifecycle.get("captures", -1)) != 1
            or lifecycle.get("capture_origin")
            not in ("measured", "unmeasured")
        ):
            raise RuntimeError(
                "FR13 fixed32 drafter graph registry drifted"
            )
        rows.append(
            {
                "batch_size": int(batch),
                "passes": int(_key_passes),
                "segment": int(_key_segment),
                "graph_signature": signature,
                "captures": 1,
                "capture_origin": lifecycle["capture_origin"],
                "measured_replays": int(lifecycle["measured_replays"]),
                "unmeasured_replays": int(
                    lifecycle["unmeasured_replays"]
                ),
            }
        )
    return rows


def _fr13_fixed32_drafter_observed_arctic(work):
    proposal = _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    if not isinstance(proposal, dict) or not isinstance(work, dict):
        raise RuntimeError("FR13 fixed32 Arctic observation is unscoped")
    batch = int(proposal["batch_size"])
    actual = {
        "batch_size": int(work.get("batch_size", -1)),
        "main_lookup_calls": int(work.get("main_lookup_calls", -1)),
        "main_lookup_tokens": int(work.get("main_lookup_tokens", -1)),
        "rank1_lookup_calls": int(work.get("rank1_lookup_calls", -1)),
        "rank1_lookup_tokens": int(work.get("rank1_lookup_tokens", -1)),
        "rank2_lookup_calls": int(work.get("rank2_lookup_calls", -1)),
        "rank2_lookup_tokens": int(work.get("rank2_lookup_tokens", -1)),
        "rescue_carry_slots": int(work.get("rescue_carry_slots", -1)),
        "arctic_lookup_calls": int(work.get("arctic_lookup_calls", -1)),
        "arctic_requested_tokens": int(
            work.get("arctic_requested_tokens", -1)
        ),
        "main_tail_columns": int(work.get("main_tail_columns", -1)),
        "rescue_path_columns": int(
            work.get("rescue_path_columns", -1)
        ),
        "merge_fill_calls": int(work.get("merge_fill_calls", -1)),
        "merge_fill_columns": int(work.get("merge_fill_columns", -1)),
        "merge_fill_rows": int(work.get("merge_fill_rows", -1)),
    }
    # FR14_GATE_SPLIT_GRAPH: a gated step hands off at draft position 3, so the
    # main Arctic chain is 8 tokens instead of 6. The rescue chains, the call
    # count and the 31-column pack are identical in both shapes -- only the main
    # chain's length moves, and proposal_end cross-checks it against the pass
    # count so an 8 can never appear on a step that ran four forwards.
    _fr14_gated = bool(work.get("gated", False))
    # NINTH PIN. 8/6 are hydra27's; the authority states 12/10 for hydra31.
    _fr14_main = (
        _FR13_FIXED32_ARCTIC_TAIL_EXPECTED["gated_main_tail_length"]
        if _fr14_gated
        else _FR13_FIXED32_ARCTIC_TAIL_EXPECTED["main_tail_length"]
    )
    _fr14_requested = (
        _FR13_FIXED32_ARCTIC_TAIL_EXPECTED["gated_arctic_requested_tokens"]
        if _fr14_gated
        else _FR13_FIXED32_ARCTIC_TAIL_EXPECTED["arctic_requested_tokens"]
    )
    _fr14_rescue_columns = _FR13_FIXED32_ARCTIC_TAIL_EXPECTED[
        "rescue_path_columns"
    ]
    expected = {
        "batch_size": batch,
        "main_lookup_calls": batch,
        "main_lookup_tokens": _fr14_main * batch,
        "rank1_lookup_calls": batch,
        "rank1_lookup_tokens": 4 * batch,
        "rank2_lookup_calls": batch,
        "rank2_lookup_tokens": 2 * batch,
        "rescue_carry_slots": (
            _FR13_FIXED32_ARCTIC_TAIL_EXPECTED["rescue_carry_slots"] * batch
        ),
        "arctic_lookup_calls": 3 * batch,
        "arctic_requested_tokens": _fr14_requested * batch,
        "main_tail_columns": _fr14_main,
        # RESOLVED by boot eleven, which reached this guard and reported the
        # engine's own geometry: rescue 6, merge-fill 16. The authority now
        # states the rule (sum of branch-chain lengths) and this reads it.
        "rescue_path_columns": _fr14_rescue_columns,
        "merge_fill_calls": 1,
        # main tail + rescue columns. The stale form held the rescue at
        # hydra27's 10 while tracking the main tail, which is exactly how it
        # produced 20 where the engine produced 16: main + rescue is CONSERVED
        # at 16 across both profiles (6 + 10 and 10 + 6).
        "merge_fill_columns": _fr14_main + _fr14_rescue_columns,
        "merge_fill_rows": (_fr14_main + _fr14_rescue_columns) * batch,
    }
    if proposal["arctic"] is not None or actual != expected:
        _arctic_drift = sorted(
            _name
            for _name in set(actual) | set(expected)
            if actual.get(_name) != expected.get(_name)
        )
        raise RuntimeError(
            "FR13 fixed32 direct Arctic/fill work drift for mode "
            + repr(_FR13_FIXED32_GDN_MODE)
            + ": "
            + "; ".join(
                _name
                + ": observed "
                + repr(actual.get(_name))
                + " against audited "
                + repr(expected.get(_name))
                for _name in _arctic_drift
            )
            + ("; arctic proposal already published" if proposal["arctic"] is not None else "")
        )
    proposal["arctic"] = actual


def _fr13_fixed32_drafter_observed_publish(
    packed_shape, packed_dtype, packed_device_type, tree_paths
):
    proposal = _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    if not isinstance(proposal, dict):
        raise RuntimeError("FR13 fixed32 drafter publish is unscoped")
    batch = int(proposal["batch_size"])
    shape = tuple(int(value) for value in packed_shape)
    paths = tuple(tuple(int(value) for value in path) for path in tree_paths)
    index = {path: node + 1 for node, path in enumerate(paths)}
    try:
        physical_parent = (-1,) + tuple(
            0 if len(path) == 1 else int(index[path[:-1]])
            for path in paths
        )
    except KeyError as error:
        raise RuntimeError(
            "FR13 fixed32 drafter parent is missing"
        ) from error
    expected_parent = (
        -1, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 7, 8, 9, 9,
        9, 12, 13, 14, 14, 14, 17, 18, 19, 23, 24, 25, 26, 28,
        29, 30,
    )
    if (
        proposal["publish"] is not None
        or proposal["arctic"] is None
        or shape != (batch, 31)
        or str(packed_dtype) not in ("torch.int64", "int64")
        or str(packed_device_type) != "cuda"
        or len(paths) != 31
        or physical_parent != expected_parent
    ):
        raise RuntimeError(
            "FR13 fixed32 final drafter publish/parent drift: "
            + repr(
                (
                    shape,
                    packed_dtype,
                    packed_device_type,
                    physical_parent,
                )
            )
        )
    proposal["publish"] = {
        "pack_columns": 31,
        "packed_rows": 31 * batch,
        "physical_parent": physical_parent,
        "physical_parent_sha256": __import__("hashlib").sha256(
            __import__("json").dumps(
                list(physical_parent),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
        "publish_shape": shape,
    }


def _fr13_fixed32_drafter_proposal_end(
    mode,
    request_ids,
    output_shape,
    output_dtype,
    output_device_type,
    outer_handoff_completed,
):
    global _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    proposal = _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT
    req_ids = tuple(str(value) for value in request_ids)
    if not isinstance(proposal, dict):
        raise RuntimeError("FR13 fixed32 drafter proposal end is unscoped")
    batch = int(proposal["batch_size"])
    if (
        mode != proposal["mode"]
        or req_ids != proposal["request_ids"]
        or tuple(int(value) for value in output_shape) != (batch, 31)
        or str(output_dtype) not in ("torch.int64", "int64")
        or str(output_device_type) != "cuda"
        or outer_handoff_completed is not True
        or proposal["mtp_execution_basis"] != "cudagraph_replay"
        # FR14_GATE_SPLIT_GRAPH: exactly three shapes are legal per step and
        # nothing else -- (4 forwards, 1 replay) single shipped graph,
        # (4, 2) split ungated, (2, 1) split gated. A missing or extra forward
        # cannot present as any of them.
        or (
            int(proposal["mtp_forward_calls"]),
            int(proposal["graph_replays"]),
        ) not in ((4, 1), (4, 2), (2, 1))
        or int(proposal["mtp_forward_rows"])
        != int(proposal["mtp_forward_calls"]) * batch
        or int(proposal["graph_captures"]) not in (0, 1, 2)
        or not isinstance(proposal["graph_id"], int)
        or not isinstance(proposal["graph_signature"], str)
        or not isinstance(proposal["arctic"], dict)
        or not isinstance(proposal["publish"], dict)
        # THE well-formedness interlock: a 2-forward step MUST have handed off
        # to a GATED Arctic chain, and a 4-forward step MUST NOT have. Either
        # mismatch is a malformed tree, so it is fatal here, before the verifier.
        #
        # PIN ELEVEN, the same 8/6 as pin nine in a second function. The chain
        # lengths are hydra27's; the authority states 12/10 for hydra31.
        or int(proposal["arctic"].get("main_tail_columns", -1))
        != (
            _FR13_FIXED32_ARCTIC_TAIL_EXPECTED["gated_main_tail_length"]
            if int(proposal["mtp_forward_calls"]) == 2
            else _FR13_FIXED32_ARCTIC_TAIL_EXPECTED["main_tail_length"]
        )
    ):
        raise RuntimeError(
            "FR13 fixed32 completed drafter proposal work drift: "
            + repr(proposal)
        )
    if proposal["measured"]:
        pending = globals().get("_FR13_FIXED32_PENDING_EVENT")
        observed = (
            pending.get("observed_work")
            if isinstance(pending, dict)
            else None
        )
        if (
            not isinstance(pending, dict)
            or not isinstance(observed, dict)
            or tuple(pending.get("request_ids", ())) != req_ids
            or observed.get("drafter") is not None
        ):
            raise RuntimeError(
                "FR13 fixed32 drafter could not bind measured event"
            )
        arctic = proposal["arctic"]
        publish = proposal["publish"]
        _fr14_calls = int(proposal["mtp_forward_calls"])
        observed["drafter"] = {
            "mtp_forward_calls": _fr14_calls,
            "mtp_forward_rows": _fr14_calls * batch,
            "arctic_lookup_calls": int(arctic["arctic_lookup_calls"]),
            "arctic_requested_tokens": int(
                arctic["arctic_requested_tokens"]
            ),
            # a gated step hands off at draft position 3, so Arctic's main
            # chain is 8 long instead of 6; the pack is 31 columns either way
            "main_tail_length": int(arctic.get("main_tail_columns", 6)),
            "rescue_chains": [[1, 4], [2, 2]],
            "carry_fill_slots": int(arctic["rescue_carry_slots"]),
            "pack_columns": 31,
            "packed_rows": 31 * batch,
        }
        request_ids_sha256 = __import__("hashlib").sha256(
            __import__("json").dumps(
                list(req_ids),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        request_id_sha256s = [
            __import__("hashlib").sha256(
                request_id.encode("utf-8")
            ).hexdigest()
            for request_id in req_ids
        ]
        observed["drafter_runtime"] = {
            "association": "same_runner_step",
            "forward_step_index": int(proposal["forward_step_index"]),
            "batch_size": batch,
            "request_ids_sha256": request_ids_sha256,
            "request_id_sha256s": request_id_sha256s,
            "proposal_begins": 1,
            "proposal_ends": 1,
            "graph_id": int(proposal["graph_id"]),
            "graph_signature": proposal["graph_signature"],
            "graph_captures": int(proposal["graph_captures"]),
            # FR14_GATE_SPLIT_GRAPH: the runtime-evidence half must report what
            # the step ACTUALLY did, not the single-graph shape. An armed but
            # ungated step is 4 forwards over 2 replays; a gated step is 2 over
            # 1. These were the last three literals that still said "4 / 1", and
            # they are why the round-2 boot refused on its first ungated step.
            "graph_replays": int(proposal["graph_replays"]),
            "mtp_observation": "capture_manifest_bound_replay",
            "mtp_forward_calls": _fr14_calls,
            "mtp_forward_rows": _fr14_calls * batch,
            "arctic_ledger": [
                {
                    "kind": "main",
                    "calls": int(arctic["main_lookup_calls"]),
                    "tokens": int(arctic["main_lookup_tokens"]),
                },
                {
                    "kind": "rank1",
                    "calls": int(arctic["rank1_lookup_calls"]),
                    "tokens": int(arctic["rank1_lookup_tokens"]),
                },
                {
                    "kind": "rank2",
                    "calls": int(arctic["rank2_lookup_calls"]),
                    "tokens": int(arctic["rank2_lookup_tokens"]),
                },
            ],
            "arctic_lookup_calls": int(arctic["arctic_lookup_calls"]),
            "arctic_requested_tokens": int(
                arctic["arctic_requested_tokens"]
            ),
            "merge_fill_calls": int(arctic["merge_fill_calls"]),
            "merge_fill_columns": int(arctic["merge_fill_columns"]),
            "merge_fill_rows": int(arctic["merge_fill_rows"]),
            "rescue_carry_slots": int(arctic["rescue_carry_slots"]),
            "publish_shape": list(publish["publish_shape"]),
            "physical_parent_sha256": publish[
                "physical_parent_sha256"
            ],
            "outer_handoff_calls": 1,
        }
        evidence = proposal.get("replay_evidence")
        if (
            not isinstance(evidence, dict)
            or int(evidence.get("proposal_begins", -1)) != 1
            or int(evidence.get("proposal_ends", -1)) != 0
            # one matching replay per segment actually replayed -- tied to the
            # proposal's own count, the same way the 11th-site fix ties the
            # capture counter to its segment
            or int(evidence.get("matching_replays", -1))
            != int(proposal["graph_replays"])
            or int(evidence.get("graph_captures", -1))
            != int(proposal["graph_captures"])
        ):
            raise RuntimeError(
                "FR13 fixed32 drafter proposal evidence drifted"
            )
        evidence["proposal_ends"] = 1
        _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT = None
        _fr13_fixed32_complete_pending_event()
        return
    _FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT = None


def _fr13_fixed32_counter_delta(after, before, key):
    return int(after.get(key, -1)) - int(before.get(key, -1))


def _fr13_fixed32_batch_counter_delta(after, before, key, batch):
    after_map = after.get(key)
    before_map = before.get(key)
    if not isinstance(after_map, dict) or not isinstance(before_map, dict):
        raise RuntimeError("FR13 fixed32 counter map is missing: " + str(key))
    return int(after_map.get(batch, after_map.get(str(batch), -1))) - int(
        before_map.get(batch, before_map.get(str(batch), -1))
    )


def _fr13_fixed32_observed_commit(
    mode,
    batch_size,
    layer_count,
    accepted_paths_shape,
    accepted_lens_shape,
    committer_before,
    committer_after,
    committer_contract,
    conv_commit_before,
    conv_commit_after,
    conv_commit_contract,
    pregather_before,
    pregather_after,
    pregather_contract,
):
    event = _fr13_fixed32_observed_current("fixed commit")
    if event is None:
        return
    batch = int(batch_size)
    path_shape = tuple(int(value) for value in accepted_paths_shape)
    lens_shape = tuple(int(value) for value in accepted_lens_shape)
    if (
        mode != event["mode"]
        or batch != int(event["batch_size"])
        or int(layer_count) != 48
        or len(path_shape) != 2
        or path_shape[0] < batch
        or path_shape[1] != 16
        or len(lens_shape) != 1
        or lens_shape[0] < batch
    ):
        raise RuntimeError(
            "FR13 fixed32 commit input geometry drift: "
            + repr((mode, batch, layer_count, path_shape, lens_shape))
        )
    if not all(
        isinstance(value, dict)
        for value in (
            committer_before,
            committer_after,
            committer_contract,
            conv_commit_before,
            conv_commit_after,
            conv_commit_contract,
            pregather_before,
            pregather_after,
            pregather_contract,
        )
    ):
        raise RuntimeError("FR13 fixed32 commit runtime contracts are missing")
    replay_delta = _fr13_fixed32_counter_delta(
        committer_after, committer_before, "actual_replays_enqueued"
    )
    replay_batch_delta = _fr13_fixed32_batch_counter_delta(
        committer_after,
        committer_before,
        "actual_replays_by_batch",
        batch,
    )
    capture_delta = _fr13_fixed32_counter_delta(
        committer_after, committer_before, "captures"
    )
    other_replay_deltas = {
        other: _fr13_fixed32_batch_counter_delta(
            committer_after,
            committer_before,
            "actual_replays_by_batch",
            other,
        )
        for other in (1, 2, 3, 4)
        if other != batch
    }
    committer_graph_dead = int(
        replay_delta != 1
        or replay_batch_delta != 1
        or capture_delta != 0
        or any(value != 0 for value in other_replay_deltas.values())
        or committer_after.get("fast_route_ready") is not True
        or committer_after.get("all_batches_ready") is not True
    )
    if committer_graph_dead:
        raise RuntimeError(
            "FR13 fixed32 committer enqueue delta drift: "
            + repr(
                (
                    replay_delta,
                    replay_batch_delta,
                    capture_delta,
                    other_replay_deltas,
                    committer_after,
                )
            )
        )
    layer_batch = committer_contract.get("layer_batch", False)
    metadata_copy_fusion = committer_contract.get(
        "metadata_copy_fusion", False
    )
    metadata_published_delta = _fr13_fixed32_batch_counter_delta(
        committer_after,
        committer_before,
        "metadata_fusion_published_by_batch",
        batch,
    )
    metadata_consumed_delta = _fr13_fixed32_batch_counter_delta(
        committer_after,
        committer_before,
        "metadata_fusion_consumed_by_batch",
        batch,
    )
    metadata_fallback_delta = _fr13_fixed32_batch_counter_delta(
        committer_after,
        committer_before,
        "metadata_fusion_fallbacks_by_batch",
        batch,
    )
    normalized_committer = {
        "batch": int(committer_contract.get("batch", -1)),
        "path_cap": int(committer_contract.get("path_cap", -1)),
        "neutralizations": int(
            committer_contract.get("neutralizations", -1)
        ),
        "ring_gathers": int(committer_contract.get("ring_gathers", -1)),
        "fused_calls": int(committer_contract.get("fused_calls", -1)),
        "graph_replays_per_event": int(
            committer_contract.get("graph_replays_per_event", -1)
        ),
        "preseed_capacity": int(
            committer_contract.get("preseed_capacity", -1)
        ),
    }
    expected_fused_calls = 1 if layer_batch is True else 48
    expected_neutralizations = 0 if layer_batch is True else 5
    expected_ring_gathers = 0 if layer_batch is True else 4
    committer_fallback = int(
        type(layer_batch) is not bool
        or type(metadata_copy_fusion) is not bool
        or (
            metadata_copy_fusion is True
            and (
                layer_batch is not True
                or metadata_published_delta != 1
                or metadata_consumed_delta != 1
                or metadata_fallback_delta != 0
                or int(
                    committer_contract.get(
                        "metadata_copy_launches_per_event", -1
                    )
                ) != 0
                or int(
                    committer_contract.get(
                        "metadata_copy_elements_per_request", -1
                    )
                ) != 17
                or committer_contract.get("metadata_validation_lease")
                != "conv_direct_exact_pointer_batch_stream_one_shot"
                or committer_contract.get("metadata_guarded_fallback")
                is not True
                or committer_contract.get(
                    "duplicate_committer_metadata_guard"
                ) is not False
            )
        )
        or (
            metadata_copy_fusion is False
            and (
                metadata_published_delta != 0
                or metadata_consumed_delta != 0
                or metadata_fallback_delta != 0
            )
        )
        or normalized_committer["batch"] != batch
        or normalized_committer["path_cap"] != 16
        or normalized_committer["neutralizations"] != expected_neutralizations
        or normalized_committer["ring_gathers"] != expected_ring_gathers
        or normalized_committer["fused_calls"] != expected_fused_calls
        or normalized_committer["graph_replays_per_event"] != 1
        or normalized_committer["preseed_capacity"] < batch
        or (
            layer_batch is True
            and (
                int(committer_contract.get("native_reference_fused_calls", -1))
                != 48
                or committer_contract.get("state_only_output_elided") is not True
                or committer_contract.get("active_length_recurrence") is not True
                or committer_contract.get(
                    "pre_replay_dynamic_bound_guard"
                ) is not True
                or int(
                    committer_contract.get("hot_scan_bound_clamps", -1)
                ) != 0
                or int(committer_contract.get("physical_node_domain", -1)) != 32
                or int(committer_contract.get("accepted_steps_max", -1)) != 12
                or committer_contract.get("final_state_store_once") is not True
                or committer_contract.get("direct_ring_loads") is not True
                or int(committer_contract.get("direct_ring_inputs", -1)) != 4
                or int(committer_contract.get("candidate_staging_launches", -1))
                != 0
                or committer_contract.get("gate_coefficients_hoisted") is not True
                or committer_contract.get(
                    "event_independent_gate_precompute"
                ) is not True
                or int(
                    committer_contract.get(
                        "gate_precompute_launches_per_process", -1
                    )
                ) != 1
                or int(committer_contract.get("gate_exp_per_event", -1)) != 0
                or committer_contract.get("full_value_tile") is not True
                or int(committer_contract.get("value_tile", -1)) != 128
                or int(committer_contract.get("kernel_warps", -1)) != 8
                or int(
                    committer_contract.get(
                        "programs_per_layer_request_value_head", -1
                    )
                ) != 1
                or int(
                    committer_contract.get(
                        "duplicate_value_tile_k_loads_per_step", -1
                    )
                ) != 0
                or int(
                    committer_contract.get(
                        "state_elements_per_thread_before_compiler_effects", -1
                    )
                ) != 64
                or committer_contract.get(
                    "physical_alias_row_uniqueness_guard"
                ) != "validate_fixed32_conv_commit_rows"
                or committer_contract.get("byte_gate")
                != "real_swe_all_reachable_accepted_lengths_0_11"
                or committer_contract.get("byte_gate_raw_compare")
                != "torch_equal_uint8"
                or committer_contract.get("unseen_length_route")
                != "shadow_then_reference"
                or int(committer_contract.get("accepted_length_max", -1)) != 11
                or int(
                    committer_contract.get("accepted_length_full_mask", -1)
                ) != 0x0FFF
            )
        )
    )
    if committer_fallback:
        raise RuntimeError(
            "FR13 fixed32 committer graph contract drift: "
            + repr(normalized_committer)
        )
    direct_delta = _fr13_fixed32_counter_delta(
        conv_commit_after, conv_commit_before, "direct_launches"
    )
    gather_delta = _fr13_fixed32_counter_delta(
        conv_commit_after, conv_commit_before, "gather_launches"
    )
    scatter_delta = _fr13_fixed32_counter_delta(
        conv_commit_after, conv_commit_before, "scatter_launches"
    )
    direct_batch_delta = _fr13_fixed32_batch_counter_delta(
        conv_commit_after,
        conv_commit_before,
        "direct_launches_by_batch",
        batch,
    )
    gather_batch_delta = _fr13_fixed32_batch_counter_delta(
        conv_commit_after,
        conv_commit_before,
        "gather_launches_by_batch",
        batch,
    )
    scatter_batch_delta = _fr13_fixed32_batch_counter_delta(
        conv_commit_after,
        conv_commit_before,
        "scatter_launches_by_batch",
        batch,
    )
    other_commit_deltas = {
        other: (
            _fr13_fixed32_batch_counter_delta(
                conv_commit_after,
                conv_commit_before,
                "direct_launches_by_batch",
                other,
            ),
            _fr13_fixed32_batch_counter_delta(
                conv_commit_after,
                conv_commit_before,
                "gather_launches_by_batch",
                other,
            ),
            _fr13_fixed32_batch_counter_delta(
                conv_commit_after,
                conv_commit_before,
                "scatter_launches_by_batch",
                other,
            ),
        )
        for other in (1, 2, 3, 4)
        if other != batch
    }
    conv_row_elems = int(conv_commit_contract.get("row_elems", -1))
    conv_channels = int(conv_commit_contract.get("channels", -1))
    conv_state_length = int(conv_commit_contract.get("state_length", -1))
    conv_source_rows = int(
        conv_commit_contract.get("source_rows_per_batch", -1)
    )
    conv_block = int(conv_commit_contract.get("block", -1))
    if (
        conv_commit_before.get("preseeded") is not True
        or conv_commit_after.get("preseeded") is not True
        or conv_commit_after.get("route") != "fixed32_direct_source_col0"
        or conv_commit_contract.get("route")
        != "fixed32_direct_source_col0"
        or int(conv_commit_contract.get("layers", -1)) != 48
        or conv_row_elems <= 0
        or conv_channels != 10240
        or conv_state_length != 34
        or conv_source_rows != 36
        or conv_block <= 0
        or conv_commit_contract.get("staging_reused") is not False
        or conv_commit_contract.get("source_staging_reused") is not True
        or int(conv_commit_contract.get("source_pointer_entries", -1)) != 48
        or conv_commit_contract.get("row_guard_route")
        != "fixed32_triton_alias3_ownerpath_warp32_physical32_v4"
        or int(conv_commit_contract.get("row_guard_kernel_launches", -1)) != 1
        or int(conv_commit_contract.get("row_guard_programs_per_request", -1))
        != 48
        or int(conv_commit_contract.get("row_guard_physical_rows", -1)) != 32
        or int(conv_commit_contract.get("row_guard_path_capacity", -1)) != 16
        or int(conv_commit_contract.get("row_guard_alias_width", -1)) != 3
        or int(conv_commit_contract.get("row_guard_compare_capacity", -1)) != 16
        or int(
            conv_commit_contract.get(
                "row_guard_path_validation_programs_per_request", -1
            )
        )
        != 1
        or int(
            conv_commit_contract.get(
                "row_guard_path_vector_loads_per_request", -1
            )
        )
        != 1
        or int(
            conv_commit_contract.get(
                "row_guard_alias_validation_programs_per_event", -1
            )
        )
        != 1
        or int(
            conv_commit_contract.get(
                "row_guard_alias_vector_loads_per_event", -1
            )
        )
        != 2
        or int(
            conv_commit_contract.get(
                "row_guard_selected_row_loads_per_program", -1
            )
        )
        != 0
        or conv_commit_contract.get("row_guard_peer_topology_proof")
        != "preseed_lease_audit"
        or int(conv_commit_contract.get("row_guard_torch_index_transforms", -1))
        != 0
        or int(
            conv_commit_contract.get("row_guard_async_scalar_reductions", -1)
        )
        != 1
        or int(conv_commit_contract.get("row_guard_async_assertions", -1)) != 1
        or int(conv_commit_contract.get("full_node_writebacks", -1)) != 0
        or int(conv_commit_contract.get("conv_remaps", -1)) != 0
        or conv_commit_contract.get("commit_bank_overlap_policy")
        != "exact_alias_only_16x3"
        or conv_commit_contract.get("commit_bank_partial_overlap") is not False
        or int(conv_commit_contract.get("commit_bank_alias_groups", -1)) != 16
        or int(conv_commit_contract.get("commit_bank_alias_width", -1)) != 3
        or conv_commit_contract.get("commit_bank_destination_guard")
        != "alias_row_unique"
        or conv_commit_contract.get("commit_null_row_rejected") is not True
        or conv_commit_contract.get("ssi_bound") is not True
        or conv_commit_contract.get("paths_bound") is not True
        or direct_delta != 1
        or gather_delta != 0
        or scatter_delta != 0
        or direct_batch_delta != 1
        or gather_batch_delta != 0
        or scatter_batch_delta != 0
        or any(
            delta != (0, 0, 0) for delta in other_commit_deltas.values()
        )
    ):
        raise RuntimeError(
            "FR13 fixed32 direct conv commit drift: "
            + repr(
                (
                    direct_delta,
                    gather_delta,
                    scatter_delta,
                    direct_batch_delta,
                    gather_batch_delta,
                    scatter_batch_delta,
                    other_commit_deltas,
                    conv_commit_contract,
                )
            )
        )
    stage_delta = _fr13_fixed32_counter_delta(
        pregather_after, pregather_before, "actual_stages"
    )
    stage_batch_delta = _fr13_fixed32_batch_counter_delta(
        pregather_after,
        pregather_before,
        "actual_stages_by_batch",
        batch,
    )
    other_stage_deltas = {
        other: _fr13_fixed32_batch_counter_delta(
            pregather_after,
            pregather_before,
            "actual_stages_by_batch",
            other,
        )
        for other in (1, 2, 3, 4)
        if other != batch
    }
    row_elems = int(pregather_contract.get("row_elems", -1))
    block = int(pregather_contract.get("block", -1))
    pregather_layers = int(pregather_contract.get("layers", -1))
    graph_capture_delta = _fr13_fixed32_counter_delta(
        pregather_after, pregather_before, "graph_capture_stages"
    )
    if (
        pregather_before.get("preseeded") is not True
        or pregather_after.get("preseeded") is not True
        or int(pregather_after.get("pointer_entries", -1)) != 48
        or stage_delta != 0
        or stage_batch_delta != 0
        or any(value != 0 for value in other_stage_deltas.values())
        or graph_capture_delta != 0
        or int(pregather_after.get("profile_capture_stages", -1)) != 0
        or int(pregather_after.get("aux_capture_stages", -1)) != 0
        or any(
            int(value) != 0
            for value in pregather_after.get(
                "actual_stages_by_batch", {}
            ).values()
        )
        or (
            # The fused arm stages nothing at commit: the pregather contract
            # still describes the (unused) staging route, but the event must
            # carry zeros and the fused class instead.
            (
                int(event["conv_stage_calls"]) != 0
                or int(event["conv_stage_replays"]) != 0
                or event["conv_stage_before_all_consumes"] is not False
                or int(event["conv_stage_layers"]) != 0
                or int(event["sfwd_conv_postprep_calls"]) != 48
                or len(event["sfwd_conv_postprep_layers"]) != 48
            )
            if _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION
            else (
                pregather_contract.get("route") != "in_graph_preconsume"
                or pregather_layers != 48
                or row_elems <= 0
                or block <= 0
                or int(pregather_contract.get("graph_capture_stages", -1)) <= 0
                or int(event["conv_stage_calls"]) != 1
                or int(event["conv_stage_replays"]) != 1
                or event["conv_stage_before_all_consumes"] is not True
                or int(event["conv_stage_layers"]) != pregather_layers
                or int(event["conv_stage_row_elems"]) != row_elems
                or int(event["conv_stage_block"]) != block
            )
        )
    ):
        raise RuntimeError(
            "FR13 fixed32 conv pregather stage drift: "
            + repr(
                (
                    stage_delta,
                    stage_batch_delta,
                    other_stage_deltas,
                    pregather_contract,
                    pregather_after,
                )
            )
        )
    conv_rows = int(layer_count) * batch
    conv_programs = (
        int(layer_count)
        * batch
        * ((conv_channels + conv_block - 1) // conv_block)
    )
    event["conv_commit"] = {
        "route": "fixed32_direct_source_col0",
        "layers": int(layer_count),
        "requests": batch,
        "row_elems": conv_row_elems,
        "channels": conv_channels,
        "state_length": conv_state_length,
        "source_rows_per_batch": conv_source_rows,
        "block": conv_block,
        "direct_launches": direct_delta,
        "gather_launches": gather_delta,
        "scatter_launches": scatter_delta,
        "direct_programs": conv_programs,
        "committed_rows": conv_rows,
        "source_staging_reused": True,
        "source_pointer_entries": 48,
        "row_guard_route": conv_commit_contract["row_guard_route"],
        "row_guard_kernel_launches": conv_commit_contract[
            "row_guard_kernel_launches"
        ],
        "row_guard_programs": (
            conv_commit_contract["row_guard_programs_per_request"] * batch
        ),
        "row_guard_physical_rows": conv_commit_contract[
            "row_guard_physical_rows"
        ],
        "row_guard_path_capacity": conv_commit_contract[
            "row_guard_path_capacity"
        ],
        "row_guard_alias_width": conv_commit_contract[
            "row_guard_alias_width"
        ],
        "row_guard_compare_capacity": conv_commit_contract[
            "row_guard_compare_capacity"
        ],
        "row_guard_path_validation_programs": (
            conv_commit_contract[
                "row_guard_path_validation_programs_per_request"
            ]
            * batch
        ),
        "row_guard_path_vector_loads": (
            conv_commit_contract["row_guard_path_vector_loads_per_request"]
            * batch
        ),
        "row_guard_alias_validation_programs": conv_commit_contract[
            "row_guard_alias_validation_programs_per_event"
        ],
        "row_guard_alias_vector_loads": conv_commit_contract[
            "row_guard_alias_vector_loads_per_event"
        ],
        "row_guard_selected_row_loads": (
            conv_commit_contract["row_guard_selected_row_loads_per_program"]
            * conv_commit_contract["row_guard_programs_per_request"]
            * batch
        ),
        "row_guard_peer_topology_proof": conv_commit_contract[
            "row_guard_peer_topology_proof"
        ],
        "row_guard_torch_index_transforms": conv_commit_contract[
            "row_guard_torch_index_transforms"
        ],
        "row_guard_async_scalar_reductions": conv_commit_contract[
            "row_guard_async_scalar_reductions"
        ],
        "row_guard_async_assertions": conv_commit_contract[
            "row_guard_async_assertions"
        ],
        "full_node_writebacks": 0,
        "conv_remaps": 0,
        "host_syncs": 0,
        "skips": 0,
        "fallback": 0,
    }
    programs = pregather_layers * batch * ((row_elems + block - 1) // block)
    event["kernel_shape"] = _fr13_fixed32_kernel_shape()
    if _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION:
        # The fused kernel subsumes the pregather stage kernel. row_elems,
        # programs and staged_rows are the staging route's grid geometry read
        # off the preseeded contract, not launches: no staging kernel ran, so
        # publishing them here would record work no kernel in this arm
        # performed, and the route name would name a route it never took.
        # The fused arm publishes its own class instead, and every subsumed
        # counter is carried as the zero it is.
        event["conv_pregather"] = None
        event["sfwd_conv_postprep"] = {
            "route": "fused_conv_postprep_single_kernel",
            "layers": len(event["sfwd_conv_postprep_layers"]),
            "requests": batch,
            "calls": int(event["sfwd_conv_postprep_calls"]),
            "calls_per_layer": 1,
            "stage_calls": int(event["conv_stage_replays"]),
            "staged_rows": 0,
            "consume_calls": int(event["conv_consume_calls"]),
            "consume_hits": int(event["conv_consume_hits"]),
            "consume_fallbacks": int(event["conv_consume_fallbacks"]),
            "freshness_matches": int(event["conv_freshness_matches"]),
        }
    else:
        event["sfwd_conv_postprep"] = None
        event["conv_pregather"] = {
            "route": "in_graph_preconsume",
            "layout_sha256": event["conv_stage_source"],
            "stage_calls": int(event["conv_stage_replays"]),
            "stage_before_all_consumes": event[
                "conv_stage_before_all_consumes"
            ],
            "layers": pregather_layers,
            "requests": batch,
            "row_elems": row_elems,
            "programs": programs,
            "staged_rows": pregather_layers * batch,
            "consume_calls": int(event["conv_consume_calls"]),
            "consume_hits": int(event["conv_consume_hits"]),
            "consume_fallbacks": int(event["conv_consume_fallbacks"]),
            "freshness_matches": int(event["conv_freshness_matches"]),
        }
    path_cap = normalized_committer["path_cap"]
    fused_calls = normalized_committer["fused_calls"]
    ring_gathers = normalized_committer["ring_gathers"]
    event["committer"] = {
        "route": "fixed16_device_fill_graph",
        "layers": int(layer_count),
        "requests": batch,
        "path_capacity": path_cap,
        "layout_slots": path_cap * batch,
        "ring_gather_ops": ring_gathers,
        "ring_layer_path_rows": (
            ring_gathers * int(layer_count) * path_cap * batch
        ),
        "neutralize_ops": normalized_committer["neutralizations"],
        "fused_layer_calls": fused_calls,
        "graph_replays": replay_delta,
        "graph_captures": capture_delta,
        "host_lens_readbacks": 0,
        "host_flag_readbacks": 0,
        "pointer_table_rebuilds": 0,
        "overflow": int(path_shape[1] > path_cap or 12 > path_cap),
        "fallback": committer_fallback,
        "graph_dead": committer_graph_dead,
    }


def _fr13_fixed32_observed_preforward_pack(
    batch_size,
    tensor_shapes,
    tensor_dtypes,
    tensor_device_types,
    zero_calls,
    copy_calls,
):
    event = _fr13_fixed32_observed_current("pre-forward request-key pack")
    if event is None:
        return
    batch = int(batch_size)
    shapes = tuple(
        tuple(int(value) for value in shape) for shape in tensor_shapes
    )
    if (
        batch != int(event["batch_size"])
        or shapes
        != ((batch, 16), (batch,), (batch, 16), (batch,))
        or tuple(str(value) for value in tensor_dtypes)
        != ("torch.int32",) * 4
        or tuple(str(value) for value in tensor_device_types)
        != ("cuda",) * 4
        or int(zero_calls) != 2
        or int(copy_calls) != 2
        or event["preforward_pack"] is not None
    ):
        raise RuntimeError(
            "FR13 fixed32 direct pre-forward pack drift: "
            + repr(
                (
                    batch,
                    shapes,
                    tensor_dtypes,
                    tensor_device_types,
                    zero_calls,
                    copy_calls,
                )
            )
        )
    event["preforward_pack"] = {
        "zero_calls": 2,
        "copy_calls": 2,
    }


def _fr13_fixed32_observed_publish_pack(
    batch_size,
    tensor_shapes,
    tensor_dtypes,
    tensor_device_types,
    post_output_copy_calls,
    post_path_pack_copy_calls,
    post_rowmap_copy_calls,
    taw_loop_iterations,
):
    event = _fr13_fixed32_observed_current("device publish/pack")
    if event is None:
        return
    batch = int(batch_size)
    actual_shapes = tuple(
        tuple(int(value) for value in shape) for shape in tensor_shapes
    )
    expected_shapes = (
        (batch, 32),
        (batch, 32),
        (batch,),
        (batch,),
        (batch, 16),
        (batch, 16),
        (batch,),
        (batch,),
        (batch, 16),
        (batch,),
    )
    dtypes = tuple(str(value) for value in tensor_dtypes)
    devices = tuple(str(value) for value in tensor_device_types)
    if (
        batch != int(event["batch_size"])
        or actual_shapes != expected_shapes
        or dtypes
        != (
            "torch.int64",
            "torch.int32",
            "torch.int64",
            "torch.int32",
            "torch.int64",
            "torch.int32",
            "torch.int64",
            "torch.int32",
            "torch.int32",
            "torch.int32",
        )
        or devices != ("cuda",) * len(expected_shapes)
        or int(post_output_copy_calls) != 2
        or int(post_path_pack_copy_calls) != 2
        or int(post_rowmap_copy_calls) != 2
        # PIN TEN. A BARE SCALAR comparison, which is why every container-literal
        # scan walked past it: there is no dict here to hold two values. It is
        # the walk cap, 12 at hydra27/tail6 and 16 at hydra31.
        or int(taw_loop_iterations)
        != _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["critical"]
        or event["preforward_pack"]
        != {"zero_calls": 2, "copy_calls": 2}
        or any(
            event[name] is not None
            for name in (
                "output_publish",
                "accepted_path_pack",
                "request_key_pack",
            )
        )
    ):
        raise RuntimeError(
            "FR13 fixed32 direct publish/pack geometry drift: "
            + repr(
                (
                    batch,
                    actual_shapes,
                    dtypes,
                    devices,
                    post_output_copy_calls,
                    post_path_pack_copy_calls,
                    post_rowmap_copy_calls,
                    taw_loop_iterations,
                )
            )
        )
    event["output_publish"] = {
        "route": "device_fixed32",
        "capacity": 32,
        "requests": batch,
        "launches": 2,
        "slots_written": 32 * batch,
        "accepted_rows_written": batch,
        "host_materializations": 0,
        "host_scalar_writes": 0,
        "dtoh": 0,
        "h2d": 0,
        "fallback": 0,
    }
    event["accepted_path_pack"] = {
        "route": "device_fixed16",
        "capacity": 16,
        "requests": batch,
        "pack_launches": 2,
        "slots_written": 16 * batch,
        "source_walk_slots": 12 * batch,
        "lens_written": batch,
        "host_path_items": 0,
        "overflow": 0,
        "fallback": 0,
    }
    event["request_key_pack"] = {
        "route": "device_rowmap",
        "sampler_rows": batch,
        "spec_rows": batch,
        "map_passes": 2,
        "path_slots_gathered": 32 * batch,
        "lens_gathered": 2 * batch,
        "zero_launches": 2,
        "gather_launches": 4,
        "host_dict_inserts": 0,
        "host_hash_lookups": 0,
        "missing": 0,
        "fallback": 0,
    }


def _fr13_fixed32_observed_take(mode, batch_size, forward_step_index):
    global _FR13_FIXED32_OBSERVED_CURRENT
    event = _fr13_fixed32_observed_current("commit seal")
    if event is None:
        return None
    batch = int(batch_size)
    forward = int(forward_step_index)
    if (
        mode != event["mode"]
        or batch != int(event["batch_size"])
        or forward != int(event["forward_step_index"])
    ):
        raise RuntimeError("FR13 fixed32 observed seal identity drift")
    _fr13_fixed32_validate_forward_work(event, "event")
    take_shape = _fr13_fixed32_kernel_shape()
    if event["kernel_shape"] != take_shape:
        raise RuntimeError(
            "FR13 fixed32 event kernel shape drifted: "
            + _fr13_fixed32_drift_detail(event["kernel_shape"], take_shape)
        )
    # One resolver owns the shape-to-section mapping for every reader, so the
    # seal and the record builder cannot drift apart. It raises its own
    # self-naming error when the sections and the shape disagree.
    _fr13_fixed32_observed_conv_work(event, "commit seal")
    shape_independent_sections = {
        name: isinstance(event[name], dict)
        for name in (
            "conv_commit",
            "committer",
            "preforward_pack",
            "output_publish",
            "accepted_path_pack",
            "request_key_pack",
            "batch_purity",
        )
    }
    if (
        event["execution_basis"] != "cudagraph_full_replay"
        or int(event["forward_graph_replays"]) != 1
        or not isinstance(event["forward_graph_id"], int)
        or not isinstance(event["forward_graph_signature"], str)
        or len(event["forward_graph_signature"]) != 64
        or not all(shape_independent_sections.values())
    ):
        raise RuntimeError(
            "FR13 fixed32 event lacks one exact full-graph replay: "
            + repr(
                {
                    "execution_basis": event["execution_basis"],
                    "graph_replays": event["forward_graph_replays"],
                    "graph_id": event["forward_graph_id"],
                    "graph_signature": event["forward_graph_signature"],
                    # Name the sections too: the replay fields above are
                    # usually the ones that pass, so a message carrying only
                    # them reads as a met requirement reported unmet.
                    "kernel_shape": take_shape,
                    "sections_absent": sorted(
                        name
                        for name, seen in shape_independent_sections.items()
                        if not seen
                    ),
                }
            )
        )
    observed = {
        "mode": event["mode"],
        "batch_size": batch,
        "forward_step_index": forward,
        "request_ids": tuple(event["request_ids"]),
        "batch_purity": dict(event["batch_purity"]),
        "execution_provenance": {
            "observed_route": event["execution_basis"],
            "graph_id": int(event["forward_graph_id"]),
            "graph_signature": event["forward_graph_signature"],
            "matching_replays": int(event["forward_graph_replays"]),
            "observed_fields": [
                "full_graph_identity",
                "full_graph_signature",
                "full_graph_batch_descriptor",
                "full_graph_replay_enqueue",
                "conv_pregather_freshness_token",
                "committer_replay_delta",
                "conv_commit_direct_launch_delta",
                "conv_pregather_stage_delta",
                "taw_payload",
                "preforward_pack_call_boundaries",
                "post_taw_publish_call_boundaries",
                "kv_fixed16_geometry",
            ],
            "contract_derived_fields": [
                "tree_attn_inner_calls",
                "gdn_inner_launches",
                "gdn_export_or_mask",
                "committer_graph_inner_ops",
                "kv_inner_apply_calls",
                "conv_commit_inner_launch_programs",
            ],
        },
        "tree_attn": {
            "calls": int(event["tree_calls"]),
            "q_rows": int(event["tree_q_rows"]),
            "bias_shape": list(event["tree_bias_shape"]),
            "physical_parent_digest": event["gdn_parent_sha256"],
            "bias_digest": event["gdn_ancestry_sha256"],
        },
        "gdn": {
            "scan_calls": int(event["gdn_scan_calls"]),
            "launches": int(event["gdn_launches"]),
            "path_programs": int(event["gdn_path_programs"]),
            "padded_slots": int(event["gdn_padded_slots"]),
            "nodes": int(event["gdn_nodes"]),
            "critical_path": int(event["gdn_critical_path"]),
            "grid_z": list(event["gdn_grid_z"]),
            "max_path_lengths": list(event["gdn_max_path_lengths"]),
            "export_or_mask": int(event["gdn_export_or_mask"]),
        },
        "conv_commit": dict(event["conv_commit"]),
        # Exactly one conv work section is published, and the shape names it.
        "kernel_shape": event["kernel_shape"],
        "conv_pregather": (
            None
            if event["conv_pregather"] is None
            else dict(event["conv_pregather"])
        ),
        "sfwd_conv_postprep": (
            None
            if event["sfwd_conv_postprep"] is None
            else dict(event["sfwd_conv_postprep"])
        ),
        "committer": dict(event["committer"]),
        "output_publish": dict(event["output_publish"]),
        "accepted_path_pack": dict(event["accepted_path_pack"]),
        "request_key_pack": dict(event["request_key_pack"]),
        "drafter": None,
        "drafter_runtime": None,
        "gdn_comparator": (
            dict(event["gdn_comparator"])
            if isinstance(event.get("gdn_comparator"), dict)
            else None
        ),
    }
    _FR13_FIXED32_OBSERVED_CURRENT = None
    return observed


def _fr13_fixed32_observed_kv(
    observed,
    batch_size,
    group_count,
    cache_count,
    cache_plane_counts,
    drafter_cache_count,
    drafter_cache_plane_counts,
    accepted_paths_shape,
    accepted_lens_shape,
):
    if not isinstance(observed, dict):
        raise RuntimeError("FR13 fixed32 KV has no observed event")
    batch = int(batch_size)
    planes = tuple(int(value) for value in cache_plane_counts)
    drafter_planes = tuple(
        int(value) for value in drafter_cache_plane_counts
    )
    path_shape = tuple(int(value) for value in accepted_paths_shape)
    lens_shape = tuple(int(value) for value in accepted_lens_shape)
    caches = int(cache_count)
    if (
        batch != int(observed.get("batch_size", -1))
        or int(group_count) != 1
        or caches != 16
        or planes != (2,) * caches
        or int(drafter_cache_count) != 1
        or drafter_planes != (2,)
        or path_shape != (batch, 16)
        or lens_shape != (batch,)
        or "kv_remap" in observed
    ):
        raise RuntimeError(
            "FR13 fixed32 KV observed geometry drift: "
            + repr(
                (
                    batch,
                    group_count,
                    caches,
                    planes,
                    drafter_cache_count,
                    drafter_planes,
                    path_shape,
                    lens_shape,
                )
            )
        )
    total_caches = caches + int(drafter_cache_count)
    pair_rows = total_caches * 16 * batch
    observed["kv_remap"] = {
        "route": "syncfree_target16_postsample_drafter1_postforward",
        "path_capacity": 16,
        "pair_slots": 32 * batch,
        "target_pair_slots": 16 * batch,
        "drafter_pair_slots": 16 * batch,
        "kv_groups": int(group_count),
        "target_cache_tensors": caches,
        "drafter_cache_tensors": int(drafter_cache_count),
        "kv_cache_tensors": total_caches,
        "kv_planes": 2,
        "target_prepare_calls": 1,
        "drafter_prepare_calls": 1,
        "prepare_calls": 2,
        "target_apply_cache_calls": caches,
        "drafter_apply_cache_calls": int(drafter_cache_count),
        "apply_cache_calls": total_caches,
        "src_pair_rows": pair_rows,
        "dst_pair_rows": pair_rows,
        "identity_safe_writes": pair_rows,
        "host_syncs": 0,
        "skips": 0,
        "fallback": 0,
    }


def _fr13_fixed32_observed_conv_work(observed, label):
    """Resolve the conv work section an observed record published.

    Every reader of a record's conv work goes through here, so none of them
    has to name a section literally. Exactly one of the two canonical
    sections is published and it is the one the record's own kernel_shape
    names: a record carrying both took both routes, one carrying neither
    proves no shape, and both fail closed naming what was found.
    """
    shape = observed.get("kernel_shape")
    if shape not in ("unfused_conv_pregather", "sfwd_fused_conv_postprep"):
        raise RuntimeError(
            "FR13 fixed32 observed kernel shape is not canonical at "
            + str(label)
            + ": "
            + repr(shape)
        )
    fused = shape == "sfwd_fused_conv_postprep"
    present = {
        "conv_pregather": isinstance(observed.get("conv_pregather"), dict),
        "sfwd_conv_postprep": isinstance(
            observed.get("sfwd_conv_postprep"), dict
        ),
    }
    expected = {
        "conv_pregather": not fused,
        "sfwd_conv_postprep": fused,
    }
    if present != expected:
        raise RuntimeError(
            "FR13 fixed32 observed conv work section does not match its "
            "kernel shape at "
            + str(label)
            + ": "
            + repr(
                {
                    "kernel_shape": shape,
                    "expected": sorted(
                        name for name, want in expected.items() if want
                    ),
                    "present": sorted(
                        name for name, seen in present.items() if seen
                    ),
                }
            )
        )
    name = "sfwd_conv_postprep" if fused else "conv_pregather"
    route = (
        "fused_conv_postprep_single_kernel"
        if fused
        else "in_graph_preconsume"
    )
    return shape, name, route


def _fr13_fixed32_failure_counts(observed, taw):
    if not isinstance(observed, dict) or not isinstance(taw, dict):
        raise RuntimeError("FR13 fixed32 failure evidence is missing")
    _conv_shape, conv_section, conv_route = _fr13_fixed32_observed_conv_work(
        observed, "failure counts"
    )

    def count(section, field):
        value = observed.get(section, {}).get(field)
        if type(value) is not int or value < 0:
            raise RuntimeError(
                "FR13 fixed32 failure source is not a nonnegative int: "
                + str(section)
                + "."
                + str(field)
            )
        return value

    purity = observed.get("batch_purity")
    if not isinstance(purity, dict) or set(purity) != {
        "batch_rows",
        "spec_rows",
        "physical_draft_counts",
        "mixed_pseudo_rows",
        "all_physical_31",
    }:
        raise RuntimeError("FR13 fixed32 batch-purity evidence is malformed")
    batch = int(observed.get("batch_size", -1))
    batch_rows = purity.get("batch_rows")
    spec_rows = purity.get("spec_rows")
    mixed_rows = purity.get("mixed_pseudo_rows")
    physical_counts = purity.get("physical_draft_counts")
    if (
        type(batch_rows) is not int
        or type(spec_rows) is not int
        or type(mixed_rows) is not int
        or not isinstance(physical_counts, list)
        or any(type(value) is not int or value < 0 for value in physical_counts)
        or type(purity.get("all_physical_31")) is not bool
        or batch_rows != batch
        or not 0 <= spec_rows <= batch_rows
        or len(physical_counts) != batch_rows
        or mixed_rows != batch_rows - spec_rows
        or purity["all_physical_31"]
        is not all(value == 31 for value in physical_counts)
    ):
        raise RuntimeError("FR13 fixed32 batch-purity evidence is incoherent")
    topology_cache_hit = taw.get("topology_cache_hit")
    cache_misses = taw.get("cache_misses")
    if (
        type(topology_cache_hit) is not bool
        or type(cache_misses) is not int
        or cache_misses < 0
    ):
        raise RuntimeError("FR13 fixed32 TAW cache evidence is malformed")
    route_mismatches = sum(
        observed.get(section, {}).get("route") != route
        for section, route in (
            ("output_publish", "device_fixed32"),
            ("accepted_path_pack", "device_fixed16"),
            ("request_key_pack", "device_rowmap"),
            (
                "kv_remap",
                "syncfree_target16_postsample_drafter1_postforward",
            ),
            ("conv_commit", "fixed32_direct_source_col0"),
            # Resolved from the record's shape, never named literally.
            (conv_section, conv_route),
            ("committer", "fixed16_device_fill_graph"),
        )
    )
    taw_iterations = taw.get("loop_iterations")
    output_capacity = observed.get("output_publish", {}).get("capacity")
    accepted_capacity = observed.get("accepted_path_pack", {}).get("capacity")
    if (
        type(taw_iterations) is not int
        or taw_iterations < 0
        or type(output_capacity) is not int
        or output_capacity < 0
        or type(accepted_capacity) is not int
        or accepted_capacity < 0
    ):
        raise RuntimeError("FR13 fixed32 capacity failure evidence is malformed")
    return {
        "fallback": sum(
            count(section, field)
            for section, field in (
                ("output_publish", "fallback"),
                ("accepted_path_pack", "fallback"),
                ("request_key_pack", "fallback"),
                ("kv_remap", "fallback"),
                ("conv_commit", "fallback"),
                # Both shapes publish consume_fallbacks; only the section
                # carrying it differs.
                (conv_section, "consume_fallbacks"),
                ("committer", "fallback"),
            )
        ) + route_mismatches,
        "overflow": (
            count("accepted_path_pack", "overflow")
            + count("committer", "overflow")
            + int(taw_iterations > accepted_capacity)
            + int(taw_iterations > output_capacity)
        ),
        "graph_dead": (
            count("committer", "graph_dead")
            + int(
                observed["committer"].get("route")
                != "fixed16_device_fill_graph"
            )
            + int(observed["committer"].get("graph_replays") != 1)
            + int(observed["committer"].get("graph_captures") != 0)
        ),
        "mixed_pseudo": (
            batch_rows
            - spec_rows
            + sum(value != 31 for value in physical_counts)
        ),
        "taw_cache_miss": cache_misses + int(topology_cache_hit is not True),
    }


def _fr13_fixed32_observed_build_record(
    observed, taw_payload, event_index, producer_pid
):
    if not isinstance(observed, dict):
        raise RuntimeError("FR13 fixed32 census observation is missing")
    identity = (
        observed.get("mode"),
        int(observed.get("batch_size", -1)),
        int(observed.get("forward_step_index", -1)),
    )
    index = int(event_index)
    pid = int(producer_pid)
    if (
        identity[0] != _FR13_FIXED32_MODE
        or identity[1] not in (1, 2, 3, 4)
        or identity[2] < 0
        or index < 0
        or pid <= 0
    ):
        raise RuntimeError(
            "FR13 fixed32 observed event identity is invalid: "
            + repr((identity, index, pid))
        )
    observed_request_ids = tuple(observed.get("request_ids", ()))
    request_ids_sha256 = __import__("hashlib").sha256(
        __import__("json").dumps(
            list(observed_request_ids),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    request_id_sha256s = [
        __import__("hashlib").sha256(
            request_id.encode("utf-8")
        ).hexdigest()
        for request_id in observed_request_ids
    ]
    if (
        len(observed_request_ids) != identity[1]
        or observed.get("drafter_runtime", {}).get("request_ids_sha256")
        != request_ids_sha256
        or observed.get("drafter_runtime", {}).get("request_id_sha256s")
        != request_id_sha256s
    ):
        raise RuntimeError(
            "FR13 fixed32 request identity digest did not bind event"
        )
    if (
        not isinstance(taw_payload, dict)
        or set(taw_payload)
        != {"mode", "valid_mask", "batch_size", "taw"}
        or taw_payload.get("mode") != identity[0]
        or type(taw_payload.get("valid_mask")) is not int
        or taw_payload.get("valid_mask")
        != int(globals().get("_FR13_FIXED32_VALID_MASK", -1))
        or type(taw_payload.get("batch_size")) is not int
        or taw_payload.get("batch_size") != identity[1]
        or not isinstance(taw_payload.get("taw"), dict)
    ):
        raise RuntimeError(
            "FR13 fixed32 TAW wrapper identity/schema mismatch"
        )
    actual_taw = taw_payload.get("taw")
    try:
        __import__("json").dumps(
            actual_taw,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "FR13 fixed32 TAW payload is not canonical JSON"
        ) from exc
    # The conv work section is shape-dependent, so it is resolved rather than
    # listed: the rest of these are published by every shape.
    _fr13_fixed32_observed_conv_work(observed, "observed record")
    for section in (
        "drafter",
        "drafter_runtime",
        "output_publish",
        "accepted_path_pack",
        "request_key_pack",
        "conv_commit",
        "committer",
        "kv_remap",
    ):
        actual = observed.get(section)
        if not isinstance(actual, dict):
            raise RuntimeError(
                "FR13 fixed32 observed section is missing at "
                + section
            )
    tree_observed = observed.get("tree_attn")
    gdn_observed = observed.get("gdn")
    batch_purity = observed.get("batch_purity")
    if (
        not isinstance(tree_observed, dict)
        or not isinstance(gdn_observed, dict)
        or not isinstance(batch_purity, dict)
    ):
        raise RuntimeError("FR13 fixed32 tree/GDN observations are missing")
    failures = _fr13_fixed32_failure_counts(observed, actual_taw)
    if any(value != 0 for value in failures.values()):
        raise RuntimeError(
            "FR13 fixed32 observed failure evidence is nonzero: "
            + repr(failures)
        )
    valid_mask = int(taw_payload["valid_mask"])
    event_id = str(identity[0]) + ":" + str(pid) + ":" + str(index)
    record = {
        "schema": "fr13-fixed32-work-census-v12",
        "event_id": event_id,
        "event_index": index,
        "forward_step_index": identity[2],
        "producer_pid": pid,
        "event_complete": True,
        "mode": identity[0],
        "batch_size": identity[1],
        "physical_drafts": 31,
        "verify_rows": 32 * identity[1],
        "active_nodes": valid_mask.bit_count(),
        "valid_mask": valid_mask,
        "batch_purity": dict(batch_purity),
        "drafter": dict(observed["drafter"]),
        "drafter_runtime": dict(observed["drafter_runtime"]),
        "tree_attn": dict(tree_observed),
        "gdn": dict(gdn_observed),
        "taw": dict(actual_taw),
        "output_publish": dict(observed["output_publish"]),
        "accepted_path_pack": dict(observed["accepted_path_pack"]),
        "request_key_pack": dict(observed["request_key_pack"]),
        "kv_remap": dict(observed["kv_remap"]),
        "conv_commit": dict(observed["conv_commit"]),
        "kernel_shape": observed["kernel_shape"],
        "conv_pregather": (
            None
            if observed["conv_pregather"] is None
            else dict(observed["conv_pregather"])
        ),
        "sfwd_conv_postprep": (
            None
            if observed["sfwd_conv_postprep"] is None
            else dict(observed["sfwd_conv_postprep"])
        ),
        "committer": dict(observed["committer"]),
        "failures": failures,
    }
    provenance = observed.get("execution_provenance")
    evidence = (
        _FR13_FIXED32_GRAPH_REPLAY_EVIDENCE[-1]
        if _FR13_FIXED32_GRAPH_REPLAY_EVIDENCE
        else None
    )
    if (
        not isinstance(provenance, dict)
        or not isinstance(evidence, dict)
        or evidence.get("event_complete") is not False
        or evidence.get("mode") != record.get("mode")
        or int(evidence.get("batch_size", -1))
        != int(record.get("batch_size", -1))
        or int(evidence.get("forward_step_index", -1))
        != int(record.get("forward_step_index", -1))
        or evidence.get("graph_id") != provenance.get("graph_id")
        or evidence.get("graph_signature")
        != provenance.get("graph_signature")
        or int(evidence.get("matching_replays", -1)) != 1
        or int(provenance.get("matching_replays", -1)) != 1
    ):
        raise RuntimeError(
            "FR13 fixed32 replay evidence did not bind completed event: "
            + repr((evidence, provenance))
        )
    evidence["event_index"] = int(record["event_index"])
    evidence["event_complete"] = True
    comparator = observed.get("gdn_comparator")
    if comparator is not None:
        if (
            not isinstance(comparator, dict)
            or comparator.get("runtime_capture_manifest_sha256")
            != provenance.get("graph_signature")
            or comparator.get("census_event_id") != event_id
            or comparator.get("census_event_index") != index
            or comparator.get("census_forward_step_index") != identity[2]
            or comparator.get("request_id_sha256s") != request_id_sha256s
        ):
            raise RuntimeError(
                "FR13 fixed32 GDN comparator did not bind completed event"
            )
        record["gdn_comparator"] = dict(comparator)
    drafter_evidence = (
        _FR13_FIXED32_DRAFTER_REPLAY_EVIDENCE[-1]
        if _FR13_FIXED32_DRAFTER_REPLAY_EVIDENCE
        else None
    )
    runtime = observed.get("drafter_runtime")
    if (
        not isinstance(drafter_evidence, dict)
        or drafter_evidence.get("event_complete") is not False
        or drafter_evidence.get("mode") != record.get("mode")
        or int(drafter_evidence.get("batch_size", -1))
        != int(record.get("batch_size", -1))
        or int(drafter_evidence.get("forward_step_index", -1))
        != int(record.get("forward_step_index", -1))
        or not isinstance(runtime, dict)
        or tuple(drafter_evidence.get("request_ids", ()))
        != tuple(observed.get("request_ids", ()))
        or drafter_evidence.get("graph_id") != runtime.get("graph_id")
        or drafter_evidence.get("graph_signature")
        != runtime.get("graph_signature")
        or int(drafter_evidence.get("proposal_begins", -1)) != 1
        or int(drafter_evidence.get("proposal_ends", -1)) != 1
        # FR14_GATE_SPLIT_GRAPH (13th site, found by the paired-contract sweep
        # rather than by a boot): the DRAFTER evidence chain counts one matching
        # replay per segment replayed, so an armed ungated step carries 2. Tie it
        # to the runtime-evidence half this record is being sealed against --
        # the two must agree, which is strictly stronger than either literal.
        # NOTE the TARGET forward-graph chain a few checks up is unrelated and
        # still legitimately replays exactly once per step.
        or int(drafter_evidence.get("matching_replays", -1))
        != int(runtime.get("graph_replays", -1))
    ):
        raise RuntimeError(
            "FR13 fixed32 drafter evidence did not bind completed event: "
            + repr((drafter_evidence, runtime))
        )
    drafter_evidence["event_index"] = int(record["event_index"])
    drafter_evidence["event_complete"] = True
    return record


def _fr13_fixed32_complete_pending_event():
    global _FR13_FIXED32_PENDING_EVENT
    global _FR13_FIXED32_COMPLETE_EVENTS
    pending = globals().get("_FR13_FIXED32_PENDING_EVENT")
    if not isinstance(pending, dict):
        raise RuntimeError("FR13 fixed32 completion has no pending event")
    observed = pending.get("observed_work")
    events = globals().get("_FR13_FIXED32_CENSUS_EVENTS")
    if not isinstance(observed, dict) or not isinstance(events, list):
        raise RuntimeError("FR13 fixed32 completion state is malformed")
    event_index = len(events)
    if (
        pending.get("target_kv_complete") is not True
        or pending.get("drafter_kv_complete") is not True
        or pending.get("kv_complete") is not True
        or int(pending.get("event_index", -1)) != event_index
        or tuple(pending.get("request_ids", ()))
        != tuple(observed.get("request_ids", ()))
        or not isinstance(observed.get("drafter_runtime"), dict)
    ):
        raise RuntimeError(
            "FR13 fixed32 event sealed before KV/drafter completion"
        )
    runtime_os = __import__("os")
    bound = _fr13_fixed32_observed_build_record(
        observed,
        pending.get("taw"),
        event_index,
        runtime_os.getpid(),
    )
    events.append(bound)
    _FR13_FIXED32_PENDING_EVENT = None
    _FR13_FIXED32_COMPLETE_EVENTS = len(events)

# FR13 DEPRECATION: FR13_APC_HIT_RECURRENT_SUFFIX is force-OFF at gdn
# import (this EngineCore process hosts EVERY gate, TP=1) so no stray
# export can re-engage it. The dead retired-cache family that was
# force-OFF here (and its leaking per-block checkpoint store) was removed.
for _fr13_dep_k in ("FR13_APC_HIT_RECURRENT_SUFFIX",):
    os.environ[_fr13_dep_k] = "0"

# FR13_EAGER_PACK (FIX-2): read ONCE at module scope (flag plan: env is
# read once per boot; init-time allocations are flag-conditional but
# fixed for the life of the process).
_FR13_EAGER_PACK = True  # FR13_EAGER_PACK baked from PATCH-TIME env (worker-env drops it)
_FR13_FLAGS_INKERNEL = False  # scan-kernel flag stores, PATCH-TIME env (default OFF); regate = queue 2c
_FR13_CONV_WB_BATCHED = False  # FR13_CONV_WB_BATCHED (B2c) baked from PATCH-TIME env: ONE batched conv writeback across requests replaces the per-b launch loop (committer host-gap slice)
_FR13_FIXED32_CONV_SOURCE_BATCH = False  # default-OFF: build all B fixed32 conv sources directly in persistent staging
_FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION = fixed32_sfwd_state_fusion_production_control()
# FR13_TREE_CONV_FUSED (FIX-3): read ONCE at module scope; default OFF
# until the byte A/B + live gate pass. ON fuses the tree causal-conv
# emulation's per-node state write-back loop / per-col tap loop /
# remap + committed-prior row math into vectorized torch ops over
# init-time static index tensors (bit-exact-preserving: same
# per-element ops, same order; tree-only — the native
# causal_conv1d_update path is untouched). OFF executes the legacy
# emulation verbatim (the A/B instrument).
_FR13_TREE_CONV_FUSED = True  # FR13_TREE_CONV_FUSED baked ON
_FR13_TREE_CONV_FUSED_CHECKED = False
_FR13_TREE_CONV_FUSED_NEEDLE_DONE = False
# Prepared-row persistent buffers + group-first prep ownership:
# exported by the metadata-builder init (rebuilt + re-exported per
# builder re-init, the FIX-2 cu130 group-union/5x-re-init license;
# CUDA capture happens after the LAST init so the graph binds the
# final addresses).
_FR13_TCF_PREP = None  # per-GROUP dict: group_key -> buffers
_FR13_TCF_LAYER_GROUP = {}
_FR13_TCF_PREP_OWNERS = frozenset()
_FR13_TCF_GROUP_OWNERS = {}
_FR12_SUBKERNEL_CAPTURE_ACTIVE = {}
_FR12_SUBKERNEL_CAPTURE_SAVED_BY_PREFIX = {}


def _fr13_tree_conv_fused_check():
    """FR13_TREE_CONV_FUSED engagement preconditions (class 9).

    Fail-loud, never a silent fall-through to legacy: the fused
    tree-conv path requires the committed-path prior read + the
    page-safe replay remap route + native bf16 taps, and excludes
    every diagnostic capture env (oracle/capture work runs with
    FR13_TREE_CONV_FUSED=0).
    """
    global _FR13_TREE_CONV_FUSED_CHECKED
    if _FR13_TREE_CONV_FUSED_CHECKED:
        return
    if False:  # FR13_CONV_COMMITTED_PATH baked ON; dep-guard never fires
        raise RuntimeError(
            "FR13_TREE_CONV_FUSED=1 requires FR13_CONV_COMMITTED_PATH=1"
        )
    if False:  # FR13_REPLAY_ROUTE baked ON; dep-guard never fires
        raise RuntimeError(
            "FR13_TREE_CONV_FUSED=1 requires FR13_REPLAY_ROUTE=1"
        )
    _fr13_tcf_tap_env = os.environ.get("FR12_TREE_CONV_NATIVE_BF16_TAPS")
    if _fr13_tcf_tap_env is None:
        _fr13_tcf_tap_env = os.environ.get(
            "FR11_TREE_CONV_NATIVE_BF16_TAPS", "1"
        )
    if _fr13_tcf_tap_env == "0":
        raise RuntimeError(
            "FR13_TREE_CONV_FUSED=1 requires native bf16 taps "
            "(FR12/FR11_TREE_CONV_NATIVE_BF16_TAPS != 0)"
        )
    if os.environ.get("FR13_TCF_DIAG_OVERRIDE", "0") == "1":
        # FIX-3 live-gate localization license (2026-06-12): allow
        # the diagnostic capture envs to coexist with the fused path
        # for EAGER dual-arm capture diffs ONLY. Never a gate/serving
        # config; the exclusion below stays the default.
        logger.warning(
            "FR13_TREE_CONV_FUSED: FR13_TCF_DIAG_OVERRIDE=1 — "
            "diagnostic capture envs permitted with the fused path "
            "(localization boots only, NEVER a gate)"
        )
        _FR13_TREE_CONV_FUSED_CHECKED = True
        return
    for _fr13_tcf_env in (
        "FR12_NATIVE_SPINE_ORACLE",
        "FR12_TREE_CONV_NATIVE_PRIOR_READ",
        "FR12_TREE_CONV_STATE_FULL_CAPTURE",
        "FR13_CONV_REPLAY_NODES",
        "FR12_SUBKERNEL_CAPTURE",
        "FR13_CONV_NODEBANK",
    ):
        if os.environ.get(_fr13_tcf_env, "").strip() not in ("", "0"):
            raise RuntimeError(
                "FR13_TREE_CONV_FUSED=1 excludes diagnostic env "
                + _fr13_tcf_env
                + " (run capture/oracle work with FR13_TREE_CONV_FUSED=0)"
            )
    _FR13_TREE_CONV_FUSED_CHECKED = True


def _fr13_tree_conv_fused_needle(
    fused, tree_n, width, state_len, prepared_rows, static_tables,
    zero_row_cached,
):
    """Engagement needle (class 9): fires once, in BOTH flag states."""
    global _FR13_TREE_CONV_FUSED_NEEDLE_DONE
    if _FR13_TREE_CONV_FUSED_NEEDLE_DONE:
        return
    _FR13_TREE_CONV_FUSED_NEEDLE_DONE = True
    _fr13_tcf_msg = (
        "FR13_TREE_CONV_FUSED conv emulation engaged: fused=%d "
        "tree_n=%d width=%d state_len=%d prepared_rows=%d "
        "static_tables=%d zero_row_cached=%d" % (
            1 if fused else 0,
            int(tree_n),
            int(width),
            int(state_len),
            1 if prepared_rows else 0,
            1 if static_tables else 0,
            1 if zero_row_cached else 0,
        )
    )
    try:
        logger.info(_fr13_tcf_msg)
    except Exception:
        print(_fr13_tcf_msg, flush=True)


_FR13_TCF_SC_STATS = {}


def _fr13_tcf_sc_compare(stage, fused, legacy, prefix):
    """FIX-3 dual-path selfcheck compare (localization, eager-only).

    Bitwise (int-view) equality of the fused value vs the legacy
    recompute on the SAME inputs in the SAME forward — immune to
    boot-to-boot substrate noise. Log-only (full-run statistics);
    grep needles: 'FR13_TCF_SELFCHECK MISMATCH' and the periodic
    'FR13_TCF_SELFCHECK stats' lines.
    """
    st = _FR13_TCF_SC_STATS.setdefault(
        stage, {"checks": 0, "mismatch": 0, "logged": 0}
    )
    st["checks"] += 1
    ok = (
        fused.shape == legacy.shape and fused.dtype == legacy.dtype
    )
    if ok:
        _f, _l = fused, legacy
        if _f.dtype == torch.bfloat16:
            _f, _l = _f.view(torch.int16), _l.view(torch.int16)
        elif _f.dtype == torch.float32:
            _f, _l = _f.view(torch.int32), _l.view(torch.int32)
        ok = bool(torch.equal(_f, _l))
    if not ok:
        st["mismatch"] += 1
        if st["logged"] < 5:
            st["logged"] += 1
            try:
                if fused.shape == legacy.shape and fused.is_floating_point():
                    _d = (
                        fused.to(torch.float32) - legacy.to(torch.float32)
                    ).abs()
                    detail = "max_abs=%g ndiff=%d/%d" % (
                        float(_d.max()), int((_d > 0).sum()), _d.numel(),
                    )
                elif fused.shape == legacy.shape:
                    _ne = fused != legacy
                    detail = "int ndiff=%d/%d" % (
                        int(_ne.sum()), _ne.numel(),
                    )
                else:
                    detail = "shape %s vs %s" % (
                        tuple(fused.shape), tuple(legacy.shape),
                    )
            except Exception as _sc_exc:
                detail = "detail-error %s" % (_sc_exc,)
            logger.warning(
                "FR13_TCF_SELFCHECK MISMATCH stage=%s layer=%s "
                "check#%d %s",
                stage, prefix, st["checks"], detail,
            )
    if st["checks"] % 500 == 0:
        logger.info(
            "FR13_TCF_SELFCHECK stats stage=%s checks=%d mismatch=%d",
            stage, st["checks"], st["mismatch"],
        )

# FR13_REPLAY_BOUNDARY_LOG: producer-write..consumer-read interval
# instrument (playbook row 8). Default OFF; eager-only (fail-loud on
# CUDA-graph capture); JSONL via a module-global FH; shared event
# counter incremented by the committer (tap A) so every record between
# commit k and commit k+1 carries event=k.
_FR13_BOUNDARY_EVENT = 0
_FR13_BOUNDARY_LAST_WRITTEN_BY_REQ = {}
_FR13_BOUNDARY_FH = None
_FR13_BOUNDARY_HEADER_DONE = False


# ---- FR13_OBS uniform observability registry (campaign 2026-07-04) ----
# Plain-int counters, ALWAYS ON (NOT gated on FR13_SERVE_LOG/EXACT_SEED):
# a single dict increment, no I/O / torch / sync, exception-safe = free.
# They carry the FULL count behind every FIRST-N / env-gated log line so a
# reader can never mistake a throttle window or an env-silent arm for
# 'clean' (FR13_GATE_BLINDSPOT). Other patched modules import
# gdn_linear_attn and call _fr13_obs_bump(key).
_FR13_OBS = {}
_FR13_OBS_LAST_SUMMARY_T = 0.0
_FR13_OBS_START_T = time.time()
_FR13_OBS_ATEXIT_DONE = False


def _fr13_obs_bump(key, n=1):
    # Unconditional single-int increment; never raises, does NO I/O.
    try:
        _FR13_OBS[key] = _FR13_OBS.get(key, 0) + n
    except Exception:
        pass


def _fr13_obs_final_dump():
    # atexit best-effort dump: /logs/fr13_obs_final.json (override via
    # FR13_OBS_FINAL_PATH) + one FR13_OBS_FINAL eng line if the writer was
    # ever bound. Never raises.
    try:
        _obs_payload = {
            "obs": dict(_FR13_OBS),
            "wall_s": round(
                time.time()
                - float(globals().get("_FR13_OBS_START_T", time.time())),
                3,
            ),
            "ts": round(time.time(), 6),
            "pid": os.getpid(),
        }
    except Exception:
        return
    try:
        _obs_fp = os.environ.get(
            "FR13_OBS_FINAL_PATH", "/logs/fr13_obs_final.json"
        )
        # PER-PID: the EngineCore worker holds the real counters, but a
        # fixed path lets the empty-obs API/front process overwrite it
        # (dumps came back keys=[]). Suffix the pid so the worker's
        # non-empty dump persists; skip writing an empty obs entirely.
        if not _FR13_OBS:
            return
        _obs_root, _obs_ext = os.path.splitext(_obs_fp)
        _obs_fp = _obs_root + "." + str(os.getpid()) + (_obs_ext or ".json")
        _obs_par = os.path.dirname(_obs_fp)
        if _obs_par:
            os.makedirs(_obs_par, exist_ok=True)
        with open(_obs_fp, "w") as _obs_wfh:
            _obs_wfh.write(json.dumps(_obs_payload, default=str))
    except Exception:
        pass


if not globals().get("_FR13_OBS_ATEXIT_DONE"):
    globals()["_FR13_OBS_ATEXIT_DONE"] = True
    try:
        import atexit as _fr13_obs_atexit
        _fr13_obs_atexit.register(_fr13_obs_final_dump)
    except Exception:
        pass






def _fr13_boundary_on():
    return os.environ.get("FR13_REPLAY_BOUNDARY_LOG", "0") == "1"


def _fr13_boundary_layer_match(prefix):
    pats = os.environ.get(
        "FR13_REPLAY_BOUNDARY_LAYERS", "layers.0.linear_attn"
    )
    return any(
        _p.strip() and _p.strip() in str(prefix)
        for _p in pats.split(",")
    )


def _fr13_boundary_emit(record):
    global _FR13_BOUNDARY_FH, _FR13_BOUNDARY_HEADER_DONE
    if not _fr13_boundary_on():
        return
    # Class-9 fail-loud: this instrument is EAGER-ONLY (the final
    # re-gate runs captured with it OFF).
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13_REPLAY_BOUNDARY_LOG is eager-only: emit called "
            "during CUDA-graph capture"
        )
    if _FR13_BOUNDARY_FH is None:
        _fr13_bnd_path = os.environ.get(
            "FR13_REPLAY_BOUNDARY_PATH", "/logs/fr13_replay_boundary.jsonl"
        )
        _fr13_bnd_parent = os.path.dirname(_fr13_bnd_path)
        if _fr13_bnd_parent:
            os.makedirs(_fr13_bnd_parent, exist_ok=True)
        _FR13_BOUNDARY_FH = open(_fr13_bnd_path, "a", buffering=1)
    if not _FR13_BOUNDARY_HEADER_DONE:
        _FR13_BOUNDARY_HEADER_DONE = True
        _FR13_BOUNDARY_FH.write(json.dumps({
            "tap": "header",
            "regime": "eager",
            "ts": round(time.time(), 6),
            "pid": os.getpid(),
            "flags": {
                _k: os.environ.get(_k)
                for _k in (
                    "FR13_REPLAY_ROUTE",
                    "FR13_REPLAY_BOUNDARY_LOG",
                    "FR13_REPLAY_BOUNDARY_LAYERS",
                    "FR13_TREE_REQKEY",
                    "FR13_CONV_COMMITTED_PATH",
                    "FR13_TREE_BONUS_SELF",
                    "FR10_ENABLE_TREE_GDN",
                    "VLLM_BATCH_INVARIANT",
                    "FR13_BI_TREE_ATTN",
                    "FR10_METRICS",
                )
            },
        }) + "\n")
    _fr13_bnd_rec = dict(record)
    _fr13_bnd_rec["regime"] = "eager"
    _fr13_bnd_rec["ts"] = round(time.time(), 6)
    _fr13_bnd_rec.setdefault(
        "event", int(globals().get("_FR13_BOUNDARY_EVENT", 0))
    )
    _FR13_BOUNDARY_FH.write(json.dumps(_fr13_bnd_rec, default=str) + "\n")


def _fr13_boundary_row_digest(t, max_bytes=4096):
    # first-8-bytes + sha256-of-first-4096-bytes of a state row/segment
    # (uint8 reinterpretation; works for bf16 banks where .numpy() on
    # the source dtype would fail).
    _fr13_bnd_flat = t.detach().reshape(-1)
    _fr13_bnd_es = _fr13_bnd_flat.element_size()
    _fr13_bnd_n = min(
        int(_fr13_bnd_flat.numel()), max(1, max_bytes // _fr13_bnd_es)
    )
    _fr13_bnd_raw = (
        _fr13_bnd_flat[:_fr13_bnd_n].cpu().contiguous()
        .view(torch.uint8).numpy().tobytes()
    )
    return (
        list(_fr13_bnd_raw[:8]),
        hashlib.sha256(_fr13_bnd_raw).hexdigest(),
        len(_fr13_bnd_raw),
    )


def _fr13_boundary_window_digest(bank, rows):
    # whole-window digests for interval bisection (taps A/B/B0)
    _fr13_bnd_out = []
    for _fr13_bnd_r in rows:
        _fr13_bnd_f8, _fr13_bnd_sha, _fr13_bnd_nb = (
            _fr13_boundary_row_digest(bank[int(_fr13_bnd_r)])
        )
        _fr13_bnd_out.append({
            "row": int(_fr13_bnd_r),
            "first8": _fr13_bnd_f8,
            "sha4096": _fr13_bnd_sha,
        })
    return _fr13_bnd_out


def _fr12_subkernel_capture_debug(event, **fields):
    debug_path = os.environ.get("FR12_SUBKERNEL_CAPTURE_DEBUG_LOG")
    if not debug_path:
        return
    try:
        parent = os.path.dirname(debug_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        row = {"event": event}
        row.update(fields)
        with open(debug_path, "a", buffering=1) as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except Exception as exc:
        logger.warning("FR12 subkernel capture debug failed: %s", exc)


def _fr12_subkernel_capture_get(self, num_tokens=None, create=False):
    path = os.environ.get("FR12_SUBKERNEL_CAPTURE")
    if not path:
        return None
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            if create:
                _fr12_subkernel_capture_debug("skip_cuda_capture")
            return None
    except Exception as exc:
        if create:
            _fr12_subkernel_capture_debug("skip_cuda_capture_check_error", error=str(exc))
        return None
    prefix = str(getattr(self, "prefix", ""))
    want_prefix = os.environ.get(
        "FR12_SUBKERNEL_CAPTURE_LAYER_PREFIX",
        "language_model.model.layers.0.linear_attn",
    )
    if want_prefix and want_prefix != "*":
        _fr12_wanted = {
            _x.strip() for _x in want_prefix.split(",") if _x.strip()
        }
        if prefix not in _fr12_wanted:
            if create:
                _fr12_subkernel_capture_debug(
                    "skip_prefix_mismatch",
                    prefix=prefix,
                    want_prefix=want_prefix,
                    num_tokens=None if num_tokens is None else int(num_tokens),
                )
            return None
    active = _FR12_SUBKERNEL_CAPTURE_ACTIVE.get(prefix)
    if active is not None:
        return active
    if not create:
        return None
    if num_tokens is not None:
        desired = os.environ.get("FR12_SUBKERNEL_CAPTURE_NUM_TOKENS")
        if desired:
            desired_counts = {
                int(_x.strip()) for _x in desired.split(",") if _x.strip()
            }
            if int(num_tokens) not in desired_counts:
                _fr12_subkernel_capture_debug(
                    "skip_num_tokens_mismatch",
                    prefix=prefix,
                    num_tokens=int(num_tokens),
                    desired=sorted(desired_counts),
                )
                return None
    seen = int(globals().get("_FR12_SUBKERNEL_CAPTURE_SEEN", 0))
    skip = int(os.environ.get("FR12_SUBKERNEL_CAPTURE_SKIP", "0"))
    limit = int(os.environ.get("FR12_SUBKERNEL_CAPTURE_LIMIT", "1"))
    saved = int(globals().get("_FR12_SUBKERNEL_CAPTURE_SAVED", 0))
    prefix_saved = int(_FR12_SUBKERNEL_CAPTURE_SAVED_BY_PREFIX.get(prefix, 0))
    per_prefix = "," in want_prefix or os.environ.get(
        "FR12_SUBKERNEL_CAPTURE_LIMIT_PER_PREFIX", "0"
    ) == "1"
    globals()["_FR12_SUBKERNEL_CAPTURE_SEEN"] = seen + 1
    if seen < skip or (prefix_saved if per_prefix else saved) >= limit:
        _fr12_subkernel_capture_debug(
            "skip_limit",
            prefix=prefix,
            seen=int(seen),
            skip=int(skip),
            saved=int(saved),
            prefix_saved=int(prefix_saved),
            limit=int(limit),
            per_prefix=bool(per_prefix),
        )
        return None
    root, ext = os.path.splitext(path)
    call_path = root + ".call" + str(saved) + (ext or ".pt")
    payload = {
        "schema": "fr12.gdn_l0_subkernel_capture.v1",
        "path": path,
        "call_path": call_path,
        "capture_call_index": int(seen),
        "capture_saved_index": int(saved),
        "layer_prefix": prefix,
        "num_tokens": None if num_tokens is None else int(num_tokens),
        "stages": {},
        "meta": {},
    }
    _FR12_SUBKERNEL_CAPTURE_ACTIVE[prefix] = payload
    _fr12_subkernel_capture_debug(
        "capture_created",
        prefix=prefix,
        num_tokens=None if num_tokens is None else int(num_tokens),
        call_path=call_path,
        seen=int(seen),
        saved=int(saved),
    )
    return payload


def _fr12_subkernel_capture_flush(payload, final=False):
    if payload is None:
        return
    try:
        out = str(payload["call_path"])
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(payload, out)
        if int(payload.get("capture_saved_index", 0)) == 0:
            torch.save(payload, str(payload["path"]))
        if final:
            globals()["_FR12_SUBKERNEL_CAPTURE_SAVED"] = (
                int(globals().get("_FR12_SUBKERNEL_CAPTURE_SAVED", 0)) + 1
            )
            _FR12_SUBKERNEL_CAPTURE_SAVED_BY_PREFIX[
                str(payload.get("layer_prefix", ""))
            ] = int(
                _FR12_SUBKERNEL_CAPTURE_SAVED_BY_PREFIX.get(
                    str(payload.get("layer_prefix", "")), 0
                )
            ) + 1
            _FR12_SUBKERNEL_CAPTURE_ACTIVE.pop(
                str(payload.get("layer_prefix", "")), None
            )
    except Exception as exc:
        logger.warning("FR12 subkernel capture flush failed: %s", exc)


def _fr12_subkernel_capture_tensor(self, name, tensor, create=False, extra=None):
    if tensor is None:
        return
    try:
        if extra and extra.get("num_actual_tokens") is not None:
            num_tokens = int(extra.get("num_actual_tokens"))
        else:
            num_tokens = int(tensor.shape[0]) if tensor.ndim > 0 else None
        payload = _fr12_subkernel_capture_get(
            self, num_tokens=num_tokens, create=create
        )
        if payload is None:
            return
        item = {
            "shape": [int(_x) for _x in tensor.shape],
            "dtype": str(tensor.dtype),
            "tensor": tensor.detach().to(torch.float32).cpu(),
        }
        if extra:
            item["extra"] = extra
        payload["stages"][name] = item
        _fr12_subkernel_capture_flush(payload)
    except Exception as exc:
        logger.warning("FR12 subkernel capture stage %s failed: %s", name, exc)


logger = init_logger(__name__)

# FR13 APC worker-env PROOF + bridge (runs at gdn_linear_attn module import IN THE
# EngineCore worker pid 176). VERIFIED 2026-06-27 (in-container spawn+setproctitle
# probe): the mp/spawn worker inherits the FULL parent os.environ — every FR13_APC_*
# master (SNAP_FIX, HIT_SUFFIX_CAP, ...) and PYTHONPATH are LIVE in the worker's
# os.environ, which is exactly the dict every APC gate reads (e.g. patcher 5516/6735/
# 11412). The "vars dropped in the worker" premise was a /proc/PID/environ ARTIFACT:
# set_process_title -> setproctitle("VLLM::EngineCore") overwrites the contiguous
# argv+environ memory block (8384 leading NUL bytes; 0 readable vars), so reading
# /proc is unreliable AFTER boot, while os.environ is intact. The gates therefore
# already engage; this block is the POSITIVE worker-side engagement proof + a
# belt-and-suspenders inject for any hypothetical future curated-env path.
#
# Two-part contract:
#  (1) INJECT (self-gated): only when SNAP_FIX is ABSENT from the live dict (a future
#      curated path) do we read the pid-1 sidecar and fill the missing FR13_APC_* —
#      never overwriting an explicit worker value, no sidecar (locked non-APC path)
#      -> open() raises -> inject skipped. On the proven full-inherit path this branch
#      is correctly a no-op (SNAP_FIX present).
#  (2) MARKER (UNCONDITIONAL): ALWAYS write /logs/fr13_apc_bridge_loaded.flag recording
#      what the worker's LIVE os.environ holds for the gate-bearing keys. This is the
#      real engagement proof (the gates read THIS dict), independent of whether the
#      inject branch fired. variant.sh greps this marker for SNAP_FIX/CAP before
#      recording any APC verdict, so a swallowed/absent bridge can never masquerade as
#      "engaged". Locked non-APC path: SNAP_FIX defaults ABSENT in pid-1's child env
#      too, so the marker simply records ABSENT/64 and the variant APC-assert (only run
#      for APC arms) is skipped -> byte-identical behaviour, no inject, no os.environ
#      mutation.
import os as _fr13apc_os
_fr13apc_pid = str(_fr13apc_os.getpid())
try:
    if "FR13_APC_SNAP_FIX" not in _fr13apc_os.environ:
        _fr13apc_fp = _fr13apc_os.environ.get("FR13_APC_ENV_FLAG_FILE", "/logs/fr13_apc_env.flag")
        with open(_fr13apc_fp) as _fr13apc_fh:
            for _fr13apc_line in _fr13apc_fh:
                _fr13apc_line = _fr13apc_line.strip()
                if "=" in _fr13apc_line:
                    _fr13apc_k, _fr13apc_v = _fr13apc_line.split("=", 1)
                    if _fr13apc_k.startswith("FR13_APC_") and _fr13apc_k not in _fr13apc_os.environ:
                        _fr13apc_os.environ[_fr13apc_k] = _fr13apc_v
except Exception as _fr13apc_inject_exc:
    # fail-loud: a swallowed inject error must NOT look like a clean skip
    try:
        with open(_fr13apc_os.environ.get("FR13_APC_BRIDGE_ERR_FILE", "/logs/fr13_apc_bridge_error.flag"), "w") as _fr13apc_e:
            _fr13apc_e.write("pid=" + _fr13apc_pid + " inject_exc=" + repr(_fr13apc_inject_exc))
    except Exception:
        pass
# UNCONDITIONAL engagement proof: record the LIVE os.environ values the gates will read.
try:
    _fr13apc_snap = _fr13apc_os.environ.get("FR13_APC_SNAP_FIX", "ABSENT")
    _fr13apc_cap = _fr13apc_os.environ.get("FR13_APC_HIT_SUFFIX_CAP", "ABSENT")
    _fr13apc_za = _fr13apc_os.environ.get("FR13_APC_SNAP_FIX_ZEROACCEPT", "ABSENT")
    _fr13apc_conv = _fr13apc_os.environ.get("FR13_APC_CONV_FIX", "ABSENT")
    _fr13apc_hrs = _fr13apc_os.environ.get("FR13_APC_HIT_RECURRENT_SUFFIX", "ABSENT")
    print("FR13_APC_ENV_BRIDGE_LOADED worker pid=" + _fr13apc_pid + " SNAP_FIX=" + _fr13apc_snap + " ZEROACCEPT=" + _fr13apc_za + " HIT_SUFFIX_CAP=" + _fr13apc_cap + " CONV_FIX=" + _fr13apc_conv + " HIT_RECURRENT_SUFFIX=" + _fr13apc_hrs, flush=True)
    # /logs is bind-mounted; vLLM swallows bare worker stdout so the file is the reliable
    # host-checkable proof of what the worker's live os.environ holds AT gdn import.
    with open(_fr13apc_os.environ.get("FR13_APC_BRIDGE_MARKER_FILE", "/logs/fr13_apc_bridge_loaded.flag"), "w") as _fr13apc_m:
        _fr13apc_m.write("pid=" + _fr13apc_pid + " SNAP_FIX=" + _fr13apc_snap + " HIT_SUFFIX_CAP=" + _fr13apc_cap + " ZEROACCEPT=" + _fr13apc_za + " CONV_FIX=" + _fr13apc_conv + " HIT_RECURRENT_SUFFIX=" + _fr13apc_hrs)
except Exception as _fr13apc_marker_exc:
    try:
        with open(_fr13apc_os.environ.get("FR13_APC_BRIDGE_ERR_FILE", "/logs/fr13_apc_bridge_error.flag"), "w") as _fr13apc_e:
            _fr13apc_e.write("pid=" + _fr13apc_pid + " marker_exc=" + repr(_fr13apc_marker_exc))
    except Exception:
        pass

_FR13_GDN_SUBOP_MAB_FLAG = None


def _fr13_gdn_subop_mab_enabled():
    """Resolve the FR13_GDN_SUBOP_MAB master switch for the EngineCore worker.

    The GDN forward runs in the mp/spawn EngineCore worker (VLLM::EngineCore),
    which is launched with a CURATED env that drops even registered VLLM_-
    prefixed vars; the bare FR13_GDN_SUBOP_MAB never reaches /proc/<worker>/
    environ (measured 2026-06-14: 14/66 FR13_* survive, master dropped).  The
    patcher (fr10_phase4_patch_vllm_tree_gdn.py) DOES run in pid 1 where the var
    is present, so at patch time it writes a sidecar flag file into /logs (host
    + worker visible bind mount).  Resolve from env first (cheap, future ray-
    executor path), else the sidecar.  Default OFF (no env, no sidecar) =>
    byte-identical locked path.  Result is cached after first resolution.
    """
    global _FR13_GDN_SUBOP_MAB_FLAG
    if _FR13_GDN_SUBOP_MAB_FLAG is not None:
        return _FR13_GDN_SUBOP_MAB_FLAG
    val = os.environ.get("FR13_GDN_SUBOP_MAB")
    if val is not None and val != "":
        # An EXPLICIT env value is authoritative (incl. "0" => OFF wins over any
        # stale sidecar; no leak).  Only ABSENCE consults the sidecar.
        _FR13_GDN_SUBOP_MAB_FLAG = val == "1"
        return _FR13_GDN_SUBOP_MAB_FLAG
    # env absent/empty -> consult the patch-time sidecar (worker path)
    try:
        flag_path = os.environ.get(
            "FR13_GDN_SUBOP_MAB_FLAG_FILE", "/logs/fr13_gdn_subop_mab.flag"
        )
        with open(flag_path, "r") as _fh:
            _FR13_GDN_SUBOP_MAB_FLAG = _fh.read().strip() == "1"
    except Exception:
        _FR13_GDN_SUBOP_MAB_FLAG = False
    return _FR13_GDN_SUBOP_MAB_FLAG


_FR13_SUBOP_STAGE_SEEN = set()
_FR13_SUBOP_STAGE_COUNTS = {}


def _fr13_subop_stage(tag, msg, once=True, level="error"):
    """FR13_GDN_SUBOP_MAB class-9 stage marker (NEVER silently vacuous).

    Emits a single grep-able `FR13_SUBOP_STAGE=<tag>` line at ERROR level so the
    FAILING STAGE of a chase is unmistakable in the worker log flood.  `once`
    de-dups per (prefix-less) tag so a per-forward stage logs exactly one line.
    Always increments a monotone counter so the reducer can assert a non-zero
    record count vs a non-zero engaged count (vacuous-boot discriminator).

    Default-OFF safe: callers guard every invocation behind
    _fr13_gdn_subop_mab_enabled(); this fn itself does no env work and never
    touches the live forward.
    """
    _FR13_SUBOP_STAGE_COUNTS[tag] = int(_FR13_SUBOP_STAGE_COUNTS.get(tag, 0)) + 1
    if once:
        if tag in _FR13_SUBOP_STAGE_SEEN:
            return
        _FR13_SUBOP_STAGE_SEEN.add(tag)
    try:
        emit = getattr(logger, level, None) or logger.error
        emit("FR13_SUBOP_STAGE=%s %s", tag, msg)
    except Exception:
        pass


def _fr13_subop_worker_env_gate():
    """STAGE (i): on first resolution, log whether the master reached THIS proc
    (the mp/spawn EngineCore worker) via env or the pid-1 sidecar bridge, and
    self-check /proc/self/environ.  One ERROR line names the channel so a future
    'env did not reach the worker' regression (failures #2/#3) is loud.

    Observe-only: reads env + the sidecar flag path; writes nothing.
    """
    if not _fr13_gdn_subop_mab_enabled():
        return
    env_present = os.environ.get("FR13_GDN_SUBOP_MAB") in ("1", "0") and bool(
        os.environ.get("FR13_GDN_SUBOP_MAB")
    )
    flag_path = os.environ.get(
        "FR13_GDN_SUBOP_MAB_FLAG_FILE", "/logs/fr13_gdn_subop_mab.flag"
    )
    sidecar_present = False
    try:
        with open(flag_path, "r") as _fh:
            sidecar_present = _fh.read().strip() == "1"
    except Exception:
        sidecar_present = False
    proc_present = False
    try:
        with open("/proc/self/environ", "rb") as _pf:
            proc_present = b"FR13_GDN_SUBOP_MAB=" in _pf.read()
    except Exception:
        proc_present = None
    channel = (
        "env" if env_present else ("sidecar" if sidecar_present else "NONE")
    )
    _fr13_subop_stage(
        "worker-env",
        (
            "engaged channel=" + str(channel)
            + " env_master=" + repr(os.environ.get("FR13_GDN_SUBOP_MAB"))
            + " sidecar=" + str(sidecar_present)
            + " proc_self_has_master=" + str(proc_present)
            + " pid=" + str(os.getpid())
        ),
    )


def _fr13_gdn_subop_mab(
    self,
    pre_conv_spec,
    mixed_qkv_spec,
    a,
    b,
    query_spec,
    key_spec,
    value_spec,
    conv_state_snapshot,
    conv_weights,
    ssm_state,
    spec_state_indices_tensor,
    num_accepted_tokens,
    spec_query_start_loc,
    attn_metadata,
    num_actual_tokens,
):
    """OBSERVE-ONLY L0-GDN sub-op M=10/M=5/M=1 A/B on captured inputs.

    Re-runs conv1d_update + the fused_sigmoid_gating scan on the deep-spine
    carrier row at three row-counts, reusing the SAME captured pre_conv input +
    h0 recurrent state + conv prior-window snapshot, varying ONLY which rows are
    present:
      M10 = the FULL present tree (all tree_n nodes, the served FLAT layout); the
            deep-spine row's conv window / recurrent accumulation is co-resident
            with the BRANCH rows.
      M5  = the deep node's ANCESTRY (path0 / spine slice) ONLY; the deep node
            sees its TRUE causal context, no branch rows present.
      M1  = the deep node ALONE (decode geometry, 1-token query).

    IMPORTANT (receptive-field entanglement — see notes): conv1d (depthwise
    causal, width W) and the GDN scan (recurrent) are STATEFUL/CONTEXTUAL ops.
    Native causal_conv1d_update / fused_sigmoid_gating consume the QUERY rows in
    FLAT order, so:
      * M10 gives the deep row a window contaminated by co-resident BRANCH rows
        (flat positions deep_row-W+1 .. deep_row may be branch nodes).
      * M5 gives the deep row its correct spine ancestry window.
      * M1 deep-alone CANNOT reproduce the deep row's query-side context (its
        ancestors are not present as query rows; only the prior-window remains)
        -> M1 is RECEPTIVE-FIELD-REDUCED, reported with a caveat, NOT the clean
        co-residency control.
    The clean co-residency verdict is M10-vs-M5 (branch rows present vs absent,
    deep node otherwise identical context).  The FIRST sub-op whose M10-vs-M5
    deep-row RAW != 0 is the carrier's birthplace.  M*-vs-M1 are recorded too;
    a large M5-vs-M1 (with small M10-vs-M5) localizes the carrier to the
    STATEFUL receptive field rather than a batch-tile occupancy effect.

    Writes one JSONL record per captured verify event.  Default OFF; observe-only
    (every arm runs on detached clones; no mutation of the live forward, no
    splice, no copy-recurrent, no dense-route).
    """
    if not _fr13_gdn_subop_mab_enabled():
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
    except Exception:
        return
    if conv_state_snapshot is None:
        # The conv-state snapshot is taken at the conv site under the same flag;
        # its absence means the stash did not run (class-9 disengagement) — fail
        # loud rather than read the in-place-mutated live conv_state.
        logger.warning(
            "FR13_GDN_SUBOP_MAB: conv_state snapshot missing (stash disengaged)"
        )
        return
    prefix = str(getattr(self, "prefix", ""))
    want = os.environ.get(
        "FR13_GDN_SUBOP_MAB_LAYER",
        "language_model.model.layers.0.linear_attn",
    )
    if want and want != "*":
        wanted = {x.strip() for x in want.split(",") if x.strip()}
        if prefix not in wanted:
            return
    skip = int(os.environ.get("FR13_GDN_SUBOP_MAB_SKIP", "0"))
    limit = int(os.environ.get("FR13_GDN_SUBOP_MAB_LIMIT", "1"))
    counts = globals().setdefault("_FR13_GDN_SUBOP_MAB_COUNTS", {})
    seen = int(counts.get(prefix, 0))
    counts[prefix] = seen + 1
    if not (skip <= seen < (skip + limit)):
        return
    dump = os.environ.get(
        "FR13_GDN_SUBOP_MAB_DUMP",
        "output/fr13_gdn_subop_mab/fr13_gdn_subop_mab.jsonl",
    )
    # ----- class-9 ENGAGEMENT ASSERTS (HOISTED above the swallowing try so a
    # disengagement is a LOUD `FR13_SUBOP_STAGE=engage-fail` ERROR, never a
    # warning lost in the flood — the call WAS reached, so a vacuous skip here is
    # a real defect, not a legit pure-decode skip).  Each guard records the
    # specific reason then returns WITHOUT writing a record.
    def _engage_fail(reason):
        _fr13_subop_stage("engage-fail", reason, once=False)
    if pre_conv_spec is None:
        _engage_fail("pre_conv_spec not captured (conv-site stash disengaged)")
        return
    path0 = getattr(attn_metadata, "fr10_tree_path0_nodes", None)
    if path0 is None:
        _engage_fail("tree DISENGAGED (no path0)")
        return
    tree_parent = getattr(attn_metadata, "fr10_tree_parent", None)
    if tree_parent is None:
        _engage_fail("tree DISENGAGED (no parent)")
        return
    num_spec = int(attn_metadata.num_spec_decodes)
    if num_spec < 1:
        _engage_fail("num_spec_decodes < 1 (= " + str(num_spec) + ")")
        return
    tree_n = int(tree_parent.numel())
    # tok/draft engagement: the served tree must match expected geometry
    # (cat9 => 10 nodes incl root).  Env override allows other shapes; otherwise
    # a silent shape change is class-9 and must be loud.
    expect_tree_n = os.environ.get("FR13_GDN_SUBOP_MAB_EXPECT_TREE_N", "10")
    if expect_tree_n and expect_tree_n != "*":
        if tree_n != int(expect_tree_n):
            _engage_fail(
                "tree_n=" + str(tree_n) + " != expected " + str(expect_tree_n)
                + " (silent tree-shape change; class 9)"
            )
            return
    dev = mixed_qkv_spec.device
    path0_list = [int(x) for x in path0.detach().cpu().reshape(-1).tolist()]
    # Only nodes that actually exist in the served tree (< tree_n).
    spine_rows = [r for r in path0_list if 0 <= r < tree_n]
    if len(spine_rows) < 2:
        _engage_fail("degenerate spine " + str(spine_rows))
        return
    deep_row = spine_rows[-1]
    # STAGE (iv-pre): the event is fully engaged; the next failure point is the
    # arm execution (native kernels) which stays inside the swallowing try.
    _fr13_subop_stage(
        "engaged",
        "layer=" + str(prefix) + " tree_n=" + str(tree_n)
        + " deep_row=" + str(deep_row) + " num_spec=" + str(num_spec),
        once=False,
    )
    try:
        # The FULL-tree arm = batch-0's tree-row block [0:tree_n).
        start = 0
        end = tree_n
        # ----- captured inputs for batch 0 (clone => observe-only) -------------
        pre_conv0 = pre_conv_spec[start:end].detach().clone()
        a0 = a[start:end].detach().clone()
        b0 = b[start:end].detach().clone()
        q0 = query_spec[0, start:end].detach().clone()
        k0 = key_spec[0, start:end].detach().clone()
        v0 = value_spec[start:end].detach().clone()
        # conv1d_update reads conv_state at conv_state_indices=ssi[:,0]; the
        # prior-window is the committed-path bank row.  Clone the WHOLE conv_state
        # so each arm's in-place causal_conv1d_update write never perturbs the
        # next arm or the live state (observe-only).
        ssi0 = spec_state_indices_tensor[0:1].detach().clone()
        nacc = (
            None
            if num_accepted_tokens is None
            else num_accepted_tokens[0:1].detach().clone().to(num_accepted_tokens.dtype)
        )
        width = int(conv_weights.shape[-1])
        # ---- bounds-guard scaffolding (a CUDA device-side assert is a GPU-
        # runtime fault that CPU wiring tests cannot catch; raise a CLEAN Python
        # error here BEFORE any native-kernel launch so the engine never hits an
        # OOB global read inside the triton kernel) -----------------------------
        max_path_len = int(ssi0.shape[-1])
        n_conv_banks = int(conv_state_snapshot.shape[0])
        n_ssm_banks = int(ssm_state.shape[0])

        def _guard_rows(rows):
            # every selected tree-row must exist in the captured batch-0 block
            # [0:tree_n) AND be a valid COLUMN of the [1, max_path_len] index
            # table (index_select(1, idx) OOB is itself a CUDA assert).
            for r in rows:
                if not (0 <= int(r) < tree_n):
                    raise RuntimeError(
                        "FR13_GDN_SUBOP_MAB: row "
                        + str(int(r))
                        + " out of [0,tree_n=" + str(tree_n) + ")"
                    )
                if not (0 <= int(r) < max_path_len):
                    raise RuntimeError(
                        "FR13_GDN_SUBOP_MAB: row "
                        + str(int(r))
                        + " >= max_path_len=" + str(max_path_len)
                        + " (index_select(1,.) OOB)"
                    )

        # The deep node's prior recurrent / conv state lives in the committed
        # (path0[0]) bank.  Validate it is an in-range cache bank BEFORE passing
        # it to either native kernel.
        prior_conv_bank = int(ssi0[0, 0].item())
        if not (0 <= prior_conv_bank < n_conv_banks):
            raise RuntimeError(
                "FR13_GDN_SUBOP_MAB: prior conv bank "
                + str(prior_conv_bank)
                + " out of [0," + str(n_conv_banks) + ")"
            )
        if not (0 <= prior_conv_bank < n_ssm_banks):
            raise RuntimeError(
                "FR13_GDN_SUBOP_MAB: prior ssm bank "
                + str(prior_conv_bank)
                + " out of [0," + str(n_ssm_banks) + ")"
            )

        def _conv_arm(rows, served_geom):
            m = len(rows)
            _guard_rows(rows)
            idx = torch.tensor(rows, dtype=torch.long, device=dev)
            x = pre_conv0.index_select(0, idx).contiguous()
            cs = conv_state_snapshot.detach().clone()
            qsl = torch.tensor([0, m], dtype=spec_query_start_loc.dtype, device=dev)
            # conv_state_indices = the single VALID committed bank (already
            # guarded in-range above); the depthwise causal conv has no cross-
            # bank reduction, so num_accepted_tokens (the sliding-window offset)
            # is the only spec-decode knob.
            #
            # KERNEL-VALID REDUCED-ROW GEOMETRY (FR13_SUBOP_MAB device-assert fix,
            # mirrors the scan-arm served_geom split):
            #
            #  * SERVED (M10) arm — keep the spec geometry so the deep row sees the
            #    SAME branch-co-resident sliding-window the live forward consumed
            #    (this is the faithful reference; geometry is the live one, in-range
            #    by construction).  num_accepted_tokens = the served nacc.
            #
            #  * REDUCED (M5/M1) arms — pass num_accepted_tokens=None.  With the
            #    spec path DISABLED the wrapper sets state_len = width-1 (NOT the
            #    spec formula width-1+(seqlen-1)) and the kernel uses
            #    conv_state_token_offset = 0, so it reads the deep node's committed
            #    prior window at the STANDARD width-1 location of the (in-range,
            #    guarded) committed bank and writes only width-1 columns.  This
            #    footprint is INDEPENDENT of nacc / max_path_len / the physical
            #    conv_state column count -> it provably cannot drive an OOB
            #    global read/store inside _causal_conv1d_update_kernel (the spec
            #    path's conv_state_token_offset = nacc-1 + the width-1+(m-1) store
            #    span on the reduced arm was the residual device-side assert
            #    beyond the host-side _guard_rows).  The deep node's true prior
            #    conv state already lives in that committed bank's width-1 window,
            #    so the reduced conv arm is the correct receptive-field-reduced
            #    control (matches M1 = prior-window-only by construction).
            if served_geom:
                if nacc is None:
                    conv_nacc = None
                else:
                    conv_nacc = nacc.clamp(min=1, max=m)
            else:
                conv_nacc = None
            out = causal_conv1d_update(
                x,
                cs,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=ssi0[:, 0],
                num_accepted_tokens=conv_nacc,
                query_start_loc=qsl,
                max_query_len=m,
                validate_data=False,
            )
            return out.detach()

        # For the scan we re-use the LIVE post-conv mixed_qkv_spec rows (the same
        # conv output the served forward consumed) so the scan A/B is isolated
        # from the conv A/B: each arm slices the SAME a/b/q/k/v and re-runs the
        # recurrent core on a CLONED h0 (inplace_final_state must not mutate the
        # served state).
        #
        # FR13_SUBOP_AB_CRASHED_PIVOT_CHAIN3 FIX: the previous arm passed
        # ssm_state_indices=ssi0.index_select(1, idx) + num_accepted_tokens=nacc.
        # The scan kernel reads its INITIAL state at column i_t=nacc-1 and STORES
        # per-timestep at columns 0..m-1 of that reduced table, indexing the LIVE
        # multi-bank ssm_state cache by whatever bank id sits in those columns.
        # For the reduced M5/M1 arms the deep node's column (ssi0[0, deep_row])
        # is a BRANCH/LEAF bank that need not be a valid recurrent prior-state
        # bank -> h0 + state_idx*stride is an OOB global read -> CUDA device-side
        # assert.  Fix (verdict nextAction): for the reduced arms pass the deep
        # node's prior state as a 1-row initial_state at bos=0 with a VALID bank
        # and DISABLE the spec-decode i_t=nacc-1 column read (num_accepted_tokens
        # =None -> initial read at column 0).  We build a [1,m] index table all
        # pointing at the committed prior bank so every read/store hits one
        # in-range bank; the deep node's true prior state already lives there.
        # The FULL M10 arm keeps the served geometry (validated in-range).
        def _scan_arm(rows, served_geom):
            m = len(rows)
            _guard_rows(rows)
            idx = torch.tensor(rows, dtype=torch.long, device=dev)
            qsl = torch.tensor([0, m], dtype=spec_query_start_loc.dtype, device=dev)
            ss = ssm_state.detach().clone()
            if served_geom:
                # M10 = the FULL present tree in the SERVED flat layout; reuse the
                # served per-node bank table and spec-decode offset verbatim (this
                # is the reference arm; geometry guaranteed in-range by max_path_len
                # >= tree_n and per-node valid banks).
                scan_indices = ssi0.index_select(1, idx)
                # served forward passes raw num_accepted (1 <= nacc <= tree_n);
                # m == tree_n here so the i_t=nacc-1 init column is guaranteed
                # in-range (asserted by the final guard below) -> keep it raw for
                # a faithful served reference.
                scan_nacc = None if nacc is None else nacc
            else:
                # REDUCED arms: clean 1-row-initial-state-at-bos=0-valid-bank.
                # All m columns point at the single committed prior bank; the
                # initial read (num_accepted_tokens=None -> column 0) loads the
                # deep node's prior state from that valid bank.  inplace stores
                # land in the cloned ss at that one bank (harmless; observe-only).
                scan_indices = torch.full(
                    (1, m),
                    prior_conv_bank,
                    dtype=ssi0.dtype,
                    device=dev,
                )
                scan_nacc = None
            # FINAL bank-range guard: every value the kernel will dereference must
            # be a valid ssm cache bank (PAD_SLOT_ID=-1 is skipped by the kernel
            # and allowed; any other negative or >= n_ssm_banks is an OOB read).
            _bank_vals = scan_indices.reshape(-1).tolist()
            for _bv in _bank_vals:
                _bvi = int(_bv)
                if _bvi == -1:
                    continue
                if not (0 <= _bvi < n_ssm_banks):
                    raise RuntimeError(
                        "FR13_GDN_SUBOP_MAB: scan ssm bank "
                        + str(_bvi)
                        + " out of [0," + str(n_ssm_banks) + ")"
                        + " (served_geom=" + str(bool(served_geom)) + ")"
                    )
            if scan_nacc is not None:
                # the kernel reads the initial state at column i_t=nacc-1; that
                # column must exist in the [1,m] table.
                _ni = int(scan_nacc.reshape(-1).max().item()) - 1
                if not (0 <= _ni < m):
                    raise RuntimeError(
                        "FR13_GDN_SUBOP_MAB: nacc-1=" + str(_ni)
                        + " out of [0,m=" + str(m) + ") init-state column"
                    )
            out, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a0.index_select(0, idx).contiguous(),
                b=b0.index_select(0, idx).contiguous(),
                dt_bias=self.dt_bias,
                q=q0.index_select(0, idx).unsqueeze(0).contiguous(),
                k=k0.index_select(0, idx).unsqueeze(0).contiguous(),
                v=v0.index_select(0, idx).unsqueeze(0).contiguous(),
                initial_state=ss,
                inplace_final_state=True,
                cu_seqlens=qsl,
                ssm_state_indices=scan_indices,
                num_accepted_tokens=scan_nacc,
                use_qk_l2norm_in_kernel=True,
            )
            return out.squeeze(0).detach()

        # M10 = the FULL present tree (all nodes 0..tree_n-1, in flat order).
        full_rows = list(range(tree_n))
        # M5 = the spine-slice (path0_nodes); M1 = deep-spine row alone.
        m1_rows = [deep_row]

        def _row_max_abs(big, small, big_row, small_row):
            return float(
                (
                    big[big_row].to(torch.float32) - small[small_row].to(torch.float32)
                )
                .abs()
                .max()
                .item()
            )

        def _row_mean_abs(big, small, big_row, small_row):
            return float(
                (
                    big[big_row].to(torch.float32) - small[small_row].to(torch.float32)
                )
                .abs()
                .mean()
                .item()
            )

        m5_deep = spine_rows.index(deep_row)

        def _triplet(t10, t5, t1):
            # t10/t5 are [M,...] (deep at row deep_row / m5_deep); t1 is [1,...].
            return {
                "m10_deep_vs_m5_deep_max_abs": _row_max_abs(t10, t5, deep_row, m5_deep),
                "m10_deep_vs_m5_deep_mean_abs": _row_mean_abs(
                    t10, t5, deep_row, m5_deep
                ),
                "m5_deep_vs_m1_max_abs": _row_max_abs(t5, t1, m5_deep, 0),
                "m5_deep_vs_m1_mean_abs": _row_mean_abs(t5, t1, m5_deep, 0),
                "m10_deep_vs_m1_max_abs": _row_max_abs(t10, t1, deep_row, 0),
                "m10_deep_vs_m1_mean_abs": _row_mean_abs(t10, t1, deep_row, 0),
            }

        subops = {}
        # ----- pre_conv (control sub-op): row-independent at the in_proj output,
        # so M10-deep == M5-deep == M1 BIT-EXACTLY by construction.  A non-zero
        # here means the captured input itself is M-dependent (would INVALIDATE
        # the downstream A/B) -> the harness flags it. -------------------------
        idx_spine = torch.tensor(spine_rows, dtype=torch.long, device=dev)
        idx_m1 = torch.tensor(m1_rows, dtype=torch.long, device=dev)
        pc_m10 = pre_conv0
        pc_m5 = pre_conv0.index_select(0, idx_spine)
        pc_m1 = pre_conv0.index_select(0, idx_m1)
        subops["pre_conv"] = _triplet(pc_m10, pc_m5, pc_m1)
        # ----- conv1d_out ------------------------------------------------------
        # Three native-kernel arms (causal_conv1d_update) at M10/M5/M1 = the
        # ROW-OCCUPANCY axis.  Per FR13_CONV_FIX_DESIGN the tree-conv emulation
        # is row-occupancy M-INVARIANT by construction (no cross-row reduction),
        # so conv1d_out M10-vs-M5 is EXPECTED ~0; a non-zero would overturn that.
        conv_m10 = _conv_arm(full_rows, served_geom=True)
        conv_m5 = _conv_arm(spine_rows, served_geom=False)
        conv_m1 = _conv_arm(m1_rows, served_geom=False)
        subops["conv1d_out"] = _triplet(conv_m10, conv_m5, conv_m1)
        # SECOND axis (kernel-identity realization seam, NOT row-occupancy): the
        # SERVED fused tree-conv output (bf16-tap MAC + triton_ex2_silu, already
        # held in mixed_qkv_spec [num_spec*tree_n, dim]) for the deep-spine row
        # vs the native causal_conv1d_update arms.  This is the ~9.77e-4 carrier
        # FR13_CONV_FIX_DESIGN names (our-kernel vs native realization seam).
        try:
            served_conv_deep = mixed_qkv_spec[start:end][deep_row]
            subops["conv1d_out"]["served_fused_vs_native_decode_max_abs"] = float(
                (served_conv_deep.to(torch.float32) - conv_m1[0].to(torch.float32))
                .abs()
                .max()
                .item()
            )
            subops["conv1d_out"]["served_fused_vs_native_m5_max_abs"] = float(
                (
                    served_conv_deep.to(torch.float32)
                    - conv_m5[m5_deep].to(torch.float32)
                )
                .abs()
                .max()
                .item()
            )
        except Exception:
            subops["conv1d_out"]["served_fused_vs_native_decode_max_abs"] = None
        # ----- scan_out (recurrent core) --------------------------------------
        # M10 = served flat geometry (reference, served bank table); M5/M1 =
        # reduced arms (clean 1-row-initial-state-at-bos=0-valid-bank, no live
        # multi-bank index into branch/leaf banks -> no OOB CUDA assert).
        scan_m10 = _scan_arm(full_rows, served_geom=True)
        scan_m5 = _scan_arm(spine_rows, served_geom=False)
        scan_m1 = _scan_arm(m1_rows, served_geom=False)
        subops["scan_out"] = _triplet(scan_m10, scan_m5, scan_m1)
        # Clean co-residency verdict: first sub-op whose deep-row M10-vs-M5 RAW
        # crosses threshold = the carrier's birthplace (branch rows present vs
        # absent, deep node's context otherwise identical).
        thresh = float(os.environ.get("FR13_GDN_SUBOP_MAB_THRESHOLD", "0.0"))
        order = ["pre_conv", "conv1d_out", "scan_out"]
        first = None
        for name in order:
            if subops[name]["m10_deep_vs_m5_deep_max_abs"] > thresh:
                first = name
                break
        first_m1 = None
        for name in order:
            if subops[name]["m5_deep_vs_m1_max_abs"] > thresh:
                first_m1 = name
                break
        pre_conv_m_invariant = (
            subops["pre_conv"]["m10_deep_vs_m5_deep_max_abs"] == 0.0
            and subops["pre_conv"]["m10_deep_vs_m1_max_abs"] == 0.0
        )
        rec = {
            "schema": "fr13.gdn_subop_mab.v2",
            "layer_prefix": prefix,
            "capture_event_index": int(seen),
            "tree_n": int(tree_n),
            "num_spec_decodes": int(num_spec),
            "num_actual_tokens": int(num_actual_tokens),
            "full_rows": [int(x) for x in full_rows],
            "spine_rows": [int(x) for x in spine_rows],
            "deep_row": int(deep_row),
            "m5_deep_index": int(m5_deep),
            "conv_width": int(width),
            "num_accepted_tokens": (
                None if nacc is None else [int(x) for x in nacc.detach().cpu().tolist()]
            ),
            "subops": subops,
            # CLEAN co-residency verdict (branch rows present vs absent):
            "first_coresidency_subop_m10_vs_m5": first,
            # M1 (decode-geometry, receptive-field-reduced) crossover:
            "first_subop_m5_vs_m1": first_m1,
            "pre_conv_m_invariant": bool(pre_conv_m_invariant),
            "threshold": thresh,
            "m1_receptive_field_caveat": (
                "M1 = deep node alone: its query-side ancestry is NOT present, "
                "so conv/scan see only the prior-window. M5-vs-M1 != 0 localizes "
                "the carrier to the STATEFUL receptive field; the clean co-"
                "residency control is M10-vs-M5."
            ),
        }
        out_path = dump
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "a", buffering=1) as fh:
            fh.write(json.dumps(rec) + chr(10))
        _fr13_subop_stage(
            "record-written",
            "layer=" + str(prefix) + " event=" + str(seen)
            + " path=" + str(out_path),
            once=False,
        )
        logger.warning(
            "FR13_GDN_SUBOP_MAB layer=%s event=%s deep_row=%s | M10-vs-M5 "
            "pre_conv=%.3e conv1d_out=%.3e scan_out=%.3e | first(coresid)=%s | "
            "M5-vs-M1 conv=%.3e scan=%.3e first(m1)=%s | pre_conv_M_inv=%s",
            prefix,
            seen,
            deep_row,
            subops["pre_conv"]["m10_deep_vs_m5_deep_max_abs"],
            subops["conv1d_out"]["m10_deep_vs_m5_deep_max_abs"],
            subops["scan_out"]["m10_deep_vs_m5_deep_max_abs"],
            first,
            subops["conv1d_out"]["m5_deep_vs_m1_max_abs"],
            subops["scan_out"]["m5_deep_vs_m1_max_abs"],
            first_m1,
            pre_conv_m_invariant,
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        # ARM-execution failure (e.g. a per-arm bank/row bounds-guard raise, a
        # native-kernel fault).  LOUD + distinct tag so it is never confused with
        # the success warning at the M10-vs-M5 emit; the event produced NO record.
        _fr13_subop_stage(
            "arm-fail",
            "layer=" + str(getattr(self, "prefix", "")) + " exc=" + repr(exc),
            once=False,
        )


# ============================================================================
# FR13_CONV_SUBOP_MAB: fused-build conv-subop M-invariance A/B (default OFF).
# Settles whether the SHIP fused causal-conv (fused_tree_conv_taps_acc + silu)
# is the M-dependent carrier of the cat6-vs-cat8 SPINE accept-rate gap.  Unlike
# the garble-era GDN_SUBOP_MAB (which re-ran the NATIVE causal_conv1d_update and
# never engaged on the fused ship build), this re-runs the SAME imported FUSED op
# on the SPINE-only sub-window and compares raw int-view (threshold 0.0).
# OBSERVE-ONLY: fresh tensors only, never mutates conv_state/_fr10_tree_conv_out.
# ============================================================================
_FR13_CONV_SUBOP_MAB_FLAG = None
_FR13_CONV_SUBOP_MAB_SEEN = {}


def _fr13_conv_subop_mab_enabled():
    """Resolve the FR13_CONV_SUBOP_MAB master switch (env-first, sidecar fallback).

    Mirrors _fr13_gdn_subop_mab_enabled: the EngineCore worker inherits the full
    os.environ (VERIFIED 2026-06-27; the earlier "curated env drops FR13_*" was a
    /proc/PID/environ setproctitle artifact), so the env read is authoritative.
    The pid-1 patch-time sidecar (/logs/fr13_conv_subop_mab.flag) is kept as
    belt-and-suspenders for any hypothetical future curated path.  Default OFF
    (no env, no sidecar) => byte-identical locked path.  Cached after first call.
    """
    global _FR13_CONV_SUBOP_MAB_FLAG
    if _FR13_CONV_SUBOP_MAB_FLAG is not None:
        return _FR13_CONV_SUBOP_MAB_FLAG
    val = os.environ.get("FR13_CONV_SUBOP_MAB")
    if val is not None and val != "":
        _FR13_CONV_SUBOP_MAB_FLAG = val == "1"
        return _FR13_CONV_SUBOP_MAB_FLAG
    try:
        flag_path = os.environ.get(
            "FR13_CONV_SUBOP_MAB_FLAG_FILE", "/logs/fr13_conv_subop_mab.flag"
        )
        with open(flag_path, "r") as _fh:
            _FR13_CONV_SUBOP_MAB_FLAG = _fh.read().strip() == "1"
    except Exception:
        _FR13_CONV_SUBOP_MAB_FLAG = False
    return _FR13_CONV_SUBOP_MAB_FLAG


def _fr13_conv_subop_mab(
    *, acc, window, spine, conv_weights, bias, activation, dtype,
    layer_prefix, tree_n,
):
    """OBSERVE-ONLY fused-conv M-invariance A/B (default-OFF; ship path untouched).

    Re-runs the SAME imported fused op (fused_tree_conv_taps_acc + triton silu) on
    the SPINE-only sub-window (M_reduced) and compares, raw int-view (threshold
    0.0 -- NOT atol), vs the full-M spine rows.  `acc` is the PRE-splice full-M
    taps output (fp32 [tree_n, dim]); silu is recomputed fresh here => NO native-
    spine-splice confound (the :3356 index_copy_ overwrites _fr10_out, not _acc),
    NO live mutation (fresh tensors only).
    first_conv_subop (first nonzero wins): conv_taps -> conv_out_silu -> none(=>scan/FA2).
    """
    if not _fr13_conv_subop_mab_enabled():
        return
    # EAGER-ONLY: this A/B .item()-syncs + launches kernels, which is illegal
    # during CUDA graph capture (poisons the capture -> EngineCore init fails).
    # Guard BEFORE any CUDA op (query is capture-safe) so the capture is never
    # touched; warn once (never silently vacuous).  Boot the localizer with
    # ENFORCE_EAGER=1 so the tree-verify forwards run eager and this fires.
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        if not _FR13_CONV_SUBOP_MAB_SEEN.get("__capture_warned__"):
            _FR13_CONV_SUBOP_MAB_SEEN["__capture_warned__"] = 1
            logger.warning(
                "FR13_CONV_SUBOP_MAB is EAGER-ONLY (syncs) but hit a capturing "
                "stream; boot with ENFORCE_EAGER=1. Skipping A/B under capture."
            )
        return
    try:
        prefix = str(layer_prefix)
        limit = int(os.environ.get("FR13_CONV_SUBOP_MAB_LIMIT", "16"))
        seen = int(_FR13_CONV_SUBOP_MAB_SEEN.get(prefix, 0))
        if seen >= limit:
            return
        if spine is None or int(spine.numel()) < 2:
            return
        sp = spine.to(torch.long)
        # reduced arm: SAME imported fused taps on the spine-only sub-window
        red_win = window.index_select(0, sp).contiguous()
        red_acc = fused_tree_conv_taps_acc(
            window=red_win, conv_weights=conv_weights, bias=bias
        )
        # full arm: live full-M taps (=acc); silu recomputed fresh (pre-splice)
        if activation in (True, "silu", "swish"):
            full_out_all = triton_ex2_silu_bf16(acc, out_dtype=dtype)
            red_out = triton_ex2_silu_bf16(red_acc, out_dtype=dtype)
        else:
            full_out_all = acc.to(dtype=dtype)
            red_out = red_acc.to(dtype=dtype)
        full_acc = acc.index_select(0, sp).contiguous()
        full_out = full_out_all.index_select(0, sp).contiguous()

        def _iv(t):
            return (
                t.view(torch.int32)
                if t.dtype == torch.float32
                else t.view(torch.int16)
            )

        taps_mm = int((_iv(full_acc) != _iv(red_acc)).sum().item())
        out_mm = int((_iv(full_out) != _iv(red_out)).sum().item())
        deep_mm = int((_iv(full_out[-1:]) != _iv(red_out[-1:])).sum().item())
        first = (
            "conv_taps" if taps_mm > 0
            else ("conv_out_silu" if out_mm > 0 else "none")
        )
        rec = {
            "layer_prefix": prefix,
            "tree_n": int(tree_n),
            "m_reduced": int(sp.numel()),
            "deep_row": int(sp[-1].item()),
            "taps_mismatch": taps_mm,
            "out_mismatch": out_mm,
            "deep_row_mismatch": deep_mm,
            "first_conv_subop": first,
        }
        dump = (
            os.environ.get("FR13_CONV_SUBOP_MAB_DUMP", "")
            or "/logs/fr13_conv_subop_mab.jsonl"
        )
        parent = os.path.dirname(dump)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(dump, "a", buffering=1) as _fh:
            _fh.write(json.dumps(rec) + chr(10))
        _FR13_CONV_SUBOP_MAB_SEEN[prefix] = seen + 1
        logger.warning(
            "FR13_CONV_SUBOP_MAB layer=%s tree_n=%s m_red=%s deep=%s | "
            "taps_mm=%s out_mm=%s deep_mm=%s -> first=%s",
            prefix, tree_n, int(sp.numel()), int(sp[-1].item()),
            taps_mm, out_mm, deep_mm, first,
        )
    except Exception as _exc:  # pragma: no cover - diagnostic only
        logger.warning("FR13_CONV_SUBOP_MAB failed: %s", repr(_exc))



def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # FlashInfer returns (output, state) when output_final_state=True,
    # or just output when output_final_state=False.
    # Unsqueeze back to 4D (1, L, H, D) to match fla output format
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        additional_config = get_current_vllm_config().additional_config
        assert isinstance(additional_config, dict)
        backend_cfg = additional_config.get("gdn_prefill_backend", "auto")
        backend = str(backend_cfg).strip().lower()

        supports_flashinfer = (
            current_platform.is_cuda() and current_platform.is_device_capability(90)
        )

        if backend == "flashinfer":
            use_flashinfer = supports_flashinfer
            if not use_flashinfer:
                logger.warning_once(
                    "GDN prefill backend 'flashinfer' is selected but "
                    "cannot use this kernel on the current platform. "
                    "Falling back to Triton/FLA."
                )
        elif backend == "triton":
            use_flashinfer = False
        else:
            use_flashinfer = supports_flashinfer

        if use_flashinfer:
            logger.info_once("Using FlashInfer GDN prefill kernel")
            logger.info_once(
                "FlashInfer GDN prefill kernel is JIT-compiled; first run may "
                "take a while to compile. Set `--gdn-prefill-backend triton` to "
                "avoid JIT compile time.",
            )
        else:
            logger.info_once("Using Triton/FLA GDN prefill kernel")

        self._forward_method = (
            self.forward_cuda if use_flashinfer else self.forward_native
        )

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        return fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )


@PluggableLayer.register("gated_delta_net_attention")
def _lumo_fb_proj_meta(self):
    """Return the GDNAttentionMetadata for this layer, or None.

    Gate-OFF (LUMO_FB_KERNEL_ROWS unset/!="1") returns None immediately so
    the stock projection runs unchanged (byte-identical default path).
    """
    if os.environ.get("LUMO_FB_KERNEL_ROWS") != "1":
        return None
    try:
        _meta = get_forward_context().attn_metadata
        if isinstance(_meta, dict):
            _meta = _meta.get(self.prefix)
        return _meta
    except Exception:
        return None


def _lumo_fb_proj_spans(meta):
    """If meta is a pure-spec-decode batch with >1 sequence, return
    (spans, row_len, pad_rows); else None. spans = per-sequence (start, end)
    row ranges from spec_query_start_loc. The pad group is a FIXED
    pad_rows * row_len, tree_n-independent (M-invariant for cuBLASLt).
    """
    if meta is None:
        return None
    try:
        _nspec = int(getattr(meta, "num_spec_decodes", 0))
        _qsl = getattr(meta, "spec_query_start_loc", None)
        if (
            _nspec > 1
            and int(getattr(meta, "num_prefills", 0)) == 0
            and int(getattr(meta, "num_decodes", 0)) == 0
            and _qsl is not None
        ):
            _qsl_cpu = _qsl[: _nspec + 1].detach().cpu().tolist()
            _spans = [
                (int(_qsl_cpu[_i]), int(_qsl_cpu[_i + 1]))
                for _i in range(_nspec)
            ]
            _row_len = max((_e - _s for _s, _e in _spans), default=0)
            _pad_rows = max(
                _nspec, int(os.environ.get("LUMO_FB_PROJ_PAD_ROWS", "16"))
            )
            if _row_len > 0 and _pad_rows > _nspec:
                return _spans, _row_len, _pad_rows
    except Exception:
        return None
    return None


def _lumo_fb_proj_padded(self, proj, src, spans, row_len, pad_rows):
    """Lossless-by-construction padded projection.

    Build a fixed (pad_rows * row_len, hidden) zero buffer, copy each
    sequence's real rows into its slot, run ONE proj() at the pinned shape,
    then gather the real rows back in original order. The zero pad-rows are
    discarded; each real-row output == proj(src[row]) bit-for-bit because a
    GEMM row is W @ src[row], independent of the other (zero) rows.
    """
    _padded = src.new_zeros((pad_rows * row_len, src.shape[-1]))
    for _i, (_s, _e) in enumerate(spans):
        _ps = _i * row_len
        _padded[_ps:_ps + (_e - _s)] = src[_s:_e]
    _out_padded, _ = proj(_padded)
    _parts = []
    for _i, (_s, _e) in enumerate(spans):
        _ps = _i * row_len
        _parts.append(_out_padded[_ps:_ps + (_e - _s)])
    return torch.cat(_parts, dim=0)


def _lumo_fb_proj_bmm(self, proj, src):
    """Per-row-M=1 batched bmm for the bf16 in_proj_ba GEMM.

    ROOT CAUSE (GB10 microbench, fr13_inproj_ba_mkey_microbench): the bf16 cuBLASLt
    kernel for in_proj_ba SWITCHES at M>=9 (M<=8 == M=1 bit-for-bit; M>=9 differs by
    ~1 bf16 ULP, 9.8e-4). The tree verify runs M=tree_n(>=9)/xconcurrency => the WRONG
    kernel => ~1 ULP seed that amplifies ~492x into the near-neighbor garble. native
    (M<=8, incl B=8) is on the CLEAN side => 0% garble.

    FIX: run each row as its own M=1 GEMM via a single batched bmm => every row uses the
    M=1 kernel => bit-EXACT to native (microbench vs-native=0.0), M-invariant for ANY M.
    Compute-only: the weight is a broadcast VIEW (no copy/HBM tax), ONE launch. Keeps
    branches. bias added post-matmul to match nn.Linear.
    """
    _W = proj.weight  # [out, in], bf16
    _out = torch.bmm(
        src.unsqueeze(1),
        _W.t().unsqueeze(0).expand(src.shape[0], -1, -1),
    ).squeeze(1)
    if getattr(proj, "bias", None) is not None:
        _out = _out + proj.bias
    return _out



class GatedDeltaNetAttention(PluggableLayer, MambaBase):
    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        create_in_proj_qkvz: bool = True,
        gqa_interleaved_layout=False,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix
        self.config = config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.speculative_config = vllm_config.speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )
        self.gqa_interleaved_layout = gqa_interleaved_layout
        self._forward_method = (
            self.forward_xpu if current_platform.is_xpu() else self.forward_cuda
        )

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,
        # we need to create qkvz_proj adaptively here.
        # When create_in_proj_qkvz is False (e.g. LoRA enabled in Qwen3.5),
        # in_proj_qkv and in_proj_z are created separately instead.
        if create_in_proj_qkvz:
            self.in_proj_qkvz = self.create_qkvz_proj(
                hidden_size=self.hidden_size,
                key_dim=self.key_dim,
                value_dim=self.value_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkvz",
            )
        else:
            # LoRA case (Qwen3.5 only): keep q/k/v and z as separate modules
            # so that LoRA adapters can be applied independently.
            self.in_proj_qkv = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[self.key_dim, self.key_dim, self.value_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkv",
            )
            self.in_proj_z = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=self.value_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_z",
            )
        # ba_proj doesn't support blockwise fp8 quantization.
        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint
        # layouts, so we use a factory method to create the projection.
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [
                query_key_settings,
                query_key_settings,
                value_settings,
            ],
            self.tp_size,
            self.tp_rank,
        )

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        assert output_gate_type in ["silu", "swish", "sigmoid"], (
            f"unsupported {output_gate_type=}"
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            activation=output_gate_type,
            device=current_platform.current_device(),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), qkvz weights are
        # stored as a single fused tensor with interleaved GQA layout, so we
        # use one output shard to preserve the interleaving across TP ranks.
        # When gqa_interleaved_layout=False (Qwen3.5), the checkpoint has
        # separate q, k, v, z weights, so we use 4 independent output sizes.
        output_sizes = (
            [sum((key_dim, key_dim, value_dim, value_dim))]
            if self.gqa_interleaved_layout
            else [key_dim, key_dim, value_dim, value_dim]
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), in_proj_ba is stored
        # as a single fused weight [b_g0, a_g0, b_g1, a_g1, ...] interleaved
        # by key-head group; a single output shard preserves this across TP.
        # When gqa_interleaved_layout=False (Qwen3.5), in_proj_b and in_proj_a
        # are separate checkpoint weights, so we use 2 independent output sizes.
        output_sizes = (
            [num_v_heads * 2] if self.gqa_interleaved_layout else [num_v_heads] * 2
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query.contiguous(), key.contiguous(), value.contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        self._forward_method(hidden_states, output)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if os.environ.get("FR12_SUBKERNEL_CAPTURE_INPUT", "0") == "1":
            _fr12_subkernel_capture_tensor(
                self,
                "input_hidden",
                hidden_states[:num_tokens],
                create=True,
                extra={"num_tokens": int(num_tokens)},
            )
        if hasattr(self, "in_proj_qkv"):
            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            # in_proj_ba M-keying fix: per-row-M=1 bmm is bit-EXACT to native and
            # kills the M>=9 cuBLASLt kernel-switch ~1 ULP garble seed. Flag BAKED
            # at patch time (worker env drops FR13_*). Else LUMO_FB pad. Else stock.
            # bmm meta fetched DIRECTLY (NOT via _lumo_fb_proj_meta, which
            # gates on LUMO_FB_KERNEL_ROWS at worker forward-time = dropped from
            # the EngineCore worker env => was silently None). No env dep here.
            _bmm_meta = None
            try:
                _bmm_meta = get_forward_context().attn_metadata
                if isinstance(_bmm_meta, dict):
                    _bmm_meta = _bmm_meta.get(self.prefix)
            except Exception:
                _bmm_meta = None
            if (False
                    and _bmm_meta is not None
                    and int(getattr(_bmm_meta, "num_prefills", 0)) == 0
                    and int(getattr(_bmm_meta, "num_decodes", 0)) == 0
                    and int(getattr(_bmm_meta, "num_spec_decodes", 0)) >= 1):
                try:
                    ba = _lumo_fb_proj_bmm(self, self.in_proj_ba, hidden_states)
                    if not globals().get("_FR13_INPROJ_BA_BMM_LOGGED"):
                        globals()["_FR13_INPROJ_BA_BMM_LOGGED"] = True
                        logger.error(
                            "FR13_INPROJ_BA_BMM ENGAGED: per-row-M=1 bmm rows=%d",
                            int(hidden_states.shape[0]),
                        )
                except Exception:
                    ba, _ = self.in_proj_ba(hidden_states)
            else:
                _lumo_fb_spans = _lumo_fb_proj_spans(_lumo_fb_proj_meta(self))
                if _lumo_fb_spans is not None:
                    try:
                        ba = _lumo_fb_proj_padded(
                            self, self.in_proj_ba, hidden_states, *_lumo_fb_spans
                        )
                    except Exception:
                        ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba
                )
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
                )
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.prefix),
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        if os.environ.get("FR12_SUBKERNEL_CAPTURE_Z", "0") == "1":
            _fr12_subkernel_capture_tensor(
                self,
                "gate_z",
                z.reshape(z_shape_og)[:num_tokens],
                create=False,
                extra={"num_tokens": int(num_tokens)},
            )
        _fr12_subkernel_capture_tensor(
            self,
            "gate_out",
            core_attn_out[:num_tokens],
            create=False,
            extra={"num_tokens": int(num_tokens)},
        )
        # LUMO_FB out-proj batch-invariance (consistency; out_proj is fp8
        # so usually a no-op). Gate-OFF => verbatim stock out_proj below.
        _lumo_fb_out_spans = _lumo_fb_proj_spans(_lumo_fb_proj_meta(self))
        _lumo_fb_out_done = False
        if _lumo_fb_out_spans is not None:
            try:
                _lumo_fb_o = _lumo_fb_proj_padded(
                    self, self.out_proj, core_attn_out, *_lumo_fb_out_spans
                )
                output[:_lumo_fb_o.shape[0]] = _lumo_fb_o
                _lumo_fb_out_done = True
            except Exception:
                _lumo_fb_out_done = False
        if not _lumo_fb_out_done:
            output[:num_tokens], _ = self.out_proj(core_attn_out)
        _fr12_subkernel_capture_tensor(
            self,
            "o_proj_out",
            output[:num_tokens],
            create=False,
            extra={"num_tokens": int(num_tokens)},
        )
        _fr12_payload = _fr12_subkernel_capture_get(self, create=False)
        if _fr12_payload is not None:
            _fr12_subkernel_capture_flush(_fr12_payload, final=True)

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        assert not hasattr(self, "in_proj_qkv"), "lora isn't supported on XPU."

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        # ============================================================
        # Part 2: Core Attention
        # ============================================================
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        z = torch.empty_like(core_attn_out)
        if attn_metadata_raw is not None:
            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw[self.prefix]

            # TODO: xpu does not support this param yet
            spec_sequence_masks = attn_metadata.spec_sequence_masks  # type: ignore[attr-defined]
            assert spec_sequence_masks is None

            conv_weights = self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            )

            conv_state = self.kv_cache[0]
            ssm_state = self.kv_cache[1]

            torch.ops._xpu_C.gdn_attention(
                core_attn_out,
                z,
                projected_states_qkvz,
                projected_states_ba,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                conv_state=conv_state,
                ssm_state=ssm_state,
                conv_weights=conv_weights,
                conv_bias=self.conv1d.bias,
                activation=self.activation,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_prefills=attn_metadata.num_prefills,  # type: ignore[attr-defined]
                num_decodes=attn_metadata.num_decodes,  # type: ignore[attr-defined]
                has_initial_state=attn_metadata.has_initial_state,  # type: ignore[attr-defined]
                non_spec_query_start_loc=attn_metadata.non_spec_query_start_loc,  # type: ignore[attr-defined]
                non_spec_state_indices_tensor=attn_metadata.non_spec_state_indices_tensor,  # type: ignore[attr-defined]
                num_actual_tokens=attn_metadata.num_actual_tokens,  # type: ignore[attr-defined]
                tp_size=self.tp_size,
                reorder_input=not self.gqa_interleaved_layout,
            )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _warmup_prefill_kernels(self, mixed_qkv: torch.Tensor) -> None:
        """Warm up GDN prefill kernels during V1 profiling.

        During V1 profile runs, ``_forward_core`` returns early because
        ``attn_metadata`` is ``None``, so the autotuned kernels used by
        ``chunk_gated_delta_rule`` (e.g. ``solve_tril``,
        ``chunk_scaled_dot_kkt``) are never invoked.  After profiling,
        vLLM allocates KV cache using most of the remaining GPU memory.
        When the first real inference triggers the autotuner it OOMs
        because there is not enough memory left for benchmarking.

        This method runs minimal forward passes through
        ``chunk_gated_delta_rule`` with small dummy tensors to force
        autotuning while GPU memory is still plentiful.  The autotuner
        results are cached globally, so only the first layer incurs
        actual benchmarking cost.

        All kernels including ``chunk_fwd_kernel_o`` now use a fixed
        ``BT = chunk_size`` (64).  A single warmup pass with T = 64
        is sufficient to populate the autotuner cache.

        The decode path uses ``fused_sigmoid_gating_delta_rule_update``
        which has fixed kernel parameters (no autotuning), so only the
        prefill (chunked) path needs warming up.
        """
        if hasattr(self, "_prefill_kernels_warmed_up"):
            return
        self._prefill_kernels_warmed_up = True

        device = mixed_qkv.device
        dtype = mixed_qkv.dtype
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        _, state_dtype = self.get_state_dtype()

        # All kernels use BT = chunk_size, so a single pass with T = chunk_size
        # is sufficient to populate every autotuner cache. Mirror the real
        # prefill path here: build q/k/v/g/beta via fused_post_conv_prep and
        # then run chunk_gated_delta_rule with in-kernel L2 norm disabled.
        T = FLA_CHUNK_SIZE
        dummy_mixed_qkv = torch.randn(
            T, mixed_qkv.shape[-1], device=device, dtype=dtype
        )
        dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=dummy_mixed_qkv,
            a=dummy_a,
            b=dummy_b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=num_k_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)
        state = torch.zeros(
            1,
            num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=device,
            dtype=state_dtype,
        )
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

        try:
            self.chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
            )
        except Exception:
            logger.warning(
                "GDN prefill kernel warmup (T=%d) failed for "
                "layer %s. First inference may OOM due to "
                "autotuner.",
                T,
                self.prefix,
                exc_info=True,
            )
        else:
            logger.debug(
                "GDN prefill kernel warmup (T=%d) completed for layer %s",
                T,
                self.prefix,
            )
        finally:
            del dummy_mixed_qkv, q, k, v, dummy_a, dummy_b, g, beta, state, cu_seqlens

        torch.accelerator.empty_cache()

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            # V1 profile run — warm up prefill kernels so that
            # autotuning completes before KV cache allocation.
            self._warmup_prefill_kernels(mixed_qkv)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if (
            self.enable_packed_recurrent_decode
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        if (
            _fr13_boundary_on()
            and spec_sequence_masks is not None
            and attn_metadata.num_spec_decodes > 0
            and spec_state_indices_tensor is not None
            and _fr13_boundary_layer_match(self.prefix)
        ):
            # FR13_REPLAY_BOUNDARY tap B0 (forward entry, BEFORE the conv
            # branch and the scan): whole-window digest. Segments the
            # producer-write..consumer-read interval:
            #   A.post(k) -> B0(k+1): drafter/runner/prepare;
            #   B0(k+1) -> B(k+1): layer-local conv branch;
            #   B(k)   -> A.pre(k): scan + layers 1..63 + sampler head.
            _fr13_b0_lens = globals().get("_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR")
            _fr13_b0_req_ids = (
                globals().get("_LUMO_FA_SAMPLER_ROW_REQ_IDS") or []
            )
            for _fr13_b0_i in range(int(attn_metadata.num_spec_decodes)):
                _fr13_b0_window = [
                    int(_fr13_b0_x)
                    for _fr13_b0_x in spec_state_indices_tensor[_fr13_b0_i]
                    .detach().cpu().tolist()
                ]
                _fr13_boundary_emit({
                    "tap": "B0",
                    "layer": str(self.prefix),
                    "slot": int(_fr13_b0_i),
                    "req_id": (
                        str(_fr13_b0_req_ids[_fr13_b0_i])
                        if _fr13_b0_i < len(_fr13_b0_req_ids)
                        else None
                    ),
                    "lens_now": (
                        None
                        if _fr13_b0_lens is None
                        else int(
                            _fr13_b0_lens[_fr13_b0_i].detach().cpu().item()
                        )
                    ),
                    "window_now": _fr13_b0_window,
                    "window_digest": _fr13_boundary_window_digest(
                        ssm_state, _fr13_b0_window
                    ),
                })

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            try:
                from vllm.v1.sample import rejection_sampler as _fr10_rs_mode
                _fr10_active_decode_mode = getattr(
                    _fr10_rs_mode, "_FR10_DECODE_MODE", _FR10_DECODE_MODE
                )
            except Exception:
                _fr10_active_decode_mode = _FR10_DECODE_MODE
            use_fr10_tree_conv = (
                os.environ.get("FR10_ENABLE_TREE_GDN") == "1"
                and _fr10_active_decode_mode == "tree_mtp"
                and getattr(attn_metadata, "fr10_tree_parent", None) is not None
                and attn_metadata.num_spec_decodes > 0
            )
            _fr13_conv_postprep_profile_capture = (
                _fr13_fixed32_sfwd_conv_postprep_profile_capture_active()
            )
            _fr13_conv_postprep_active = bool(
                _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION
            )
            _fr13_conv_postprep_candidate = _fr13_conv_postprep_active
            _fr13_conv_postprep_gate_enabled = False
            _fr13_conv_postprep_task_marker = None
            _fr13_conv_postprep_query = None
            _fr13_conv_postprep_key = None
            _fr13_conv_postprep_value_spec = None
            _fr13_conv_postprep_value_tree = None
            _fr13_conv_postprep_g = None
            _fr13_conv_postprep_beta = None
            if (
                _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION
                or _FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB
            ) and not use_fr10_tree_conv:
                raise RuntimeError(
                    "FR13 SFWD conv/post-prep selector requires the "
                    "physical32 tree-conv route"
                )
            _fr10_tree_conv_expected = (
                _fr10_active_decode_mode == "tree_mtp"
                and getattr(attn_metadata, "fr10_tree_parent", None) is not None
                and attn_metadata.num_spec_decodes > 0
            )
            _fr10_conv_diag = getattr(attn_metadata, "fr10_tree_conv_diag", None)
            if os.environ.get("FR10_METRICS", "0") == "1" and _fr10_conv_diag is not None:
                _fr10_conv_diag[14].add_(float(attn_metadata.num_spec_decodes))
                if use_fr10_tree_conv:
                    _fr10_conv_diag[15].add_(float(attn_metadata.num_spec_decodes))
                else:
                    _fr10_conv_diag[16].add_(float(attn_metadata.num_spec_decodes))
                _fr10_conv_diag[20].add_(float(attn_metadata.num_prefills))
                _fr10_conv_diag[21].add_(float(attn_metadata.num_decodes))
            _fr12_subkernel_capture_enabled = bool(os.environ.get("FR12_SUBKERNEL_CAPTURE"))
            _fr13_gdn_subop_mab_on = _fr13_gdn_subop_mab_enabled()
            _fr12_pre_conv_spec = None
            if (
                _fr12_subkernel_capture_enabled
                or _fr13_gdn_subop_mab_on
            ):
                _fr12_pre_conv_spec = mixed_qkv_spec.detach().clone()
            if _fr13_gdn_subop_mab_on:
                # FR13_GDN_SUBOP_MAB: stash the PRE-conv input AND a snapshot of
                # conv_state TAKEN BEFORE the served causal_conv1d_update mutates
                # it in-place (decoherence-free: the conv A/B re-runs read the
                # same prior-window the served conv consumed).  The scan-site A/B
                # (after q/k/v are split) reads these back.  Default-OFF;
                # observe-only (clone => no mutation of the live forward).
                self._fr13_gdn_subop_mab_pre_conv = _fr12_pre_conv_spec
                self._fr13_gdn_subop_mab_conv_state = (
                    conv_state.detach().clone()
                )
            if _fr12_subkernel_capture_enabled:
                try:
                    _fr12_pre_extra = {
                        "num_spec_decodes": int(attn_metadata.num_spec_decodes),
                        "num_actual_tokens": int(num_actual_tokens),
                        "tree_conv_active": bool(use_fr10_tree_conv),
                        "tree_conv_expected": bool(_fr10_tree_conv_expected),
                    }
                    if getattr(attn_metadata, "fr10_tree_parent", None) is not None:
                        _fr12_pre_extra["tree_parent"] = [
                            int(_x)
                            for _x in attn_metadata.fr10_tree_parent.detach().cpu().tolist()
                        ]
                    if spec_token_indx is not None:
                        _fr12_pre_extra["spec_token_indx"] = [
                            int(_x) for _x in spec_token_indx.detach().cpu().tolist()
                        ]
                    if spec_query_start_loc is not None:
                        _fr12_pre_extra["spec_query_start_loc"] = [
                            int(_x) for _x in spec_query_start_loc.detach().cpu().tolist()
                        ]
                    _fr12_subkernel_capture_tensor(
                        self,
                        "pre_conv",
                        mixed_qkv_spec,
                        create=True,
                        extra=_fr12_pre_extra,
                    )
                except Exception as _fr12_pre_cap_exc:
                    logger.warning("FR12 pre-conv capture failed: %s", _fr12_pre_cap_exc)
            if use_fr10_tree_conv:
                if _FR13_TREE_CONV_FUSED:
                    # FR13_TREE_CONV_FUSED (FIX-3) engagement preconditions
                    # (class 9, fail-loud — never a silent fall-through to
                    # legacy): committed-path prior read + replay route +
                    # native bf16 taps required; diagnostic capture envs
                    # excluded; dtype uniformity is the no-op-cast license
                    # for the fused tap mul.
                    _fr13_tree_conv_fused_check()
                    # NOTE (FR13 garble fp32-seed fix, PARKED 2026-07-10): a global upcast of
                    # mixed_qkv_spec to fp32 here CASCADES — mixed_qkv_spec is the post-conv scan
                    # input AND the FR13_REPLAY_ROUTE staging-ring source (bf16 byte-copy contract).
                    # A correct fp32-conv needs conv-ONLY fp32 (fp32 tap compute + a separate fp32
                    # conv-output for the bank write + fp32 bank + bf16 downcast for scan/ring), not a
                    # blanket upcast. See FR13_GARBLE_DRIFT_BINDING_PROVEN.md. Reverted to strict assert.
                    if not (
                        mixed_qkv_spec.dtype
                        == conv_state.dtype
                        == conv_weights.dtype
                    ):
                        raise RuntimeError(
                            "FR13_TREE_CONV_FUSED dtype uniformity violated: "
                            + str(mixed_qkv_spec.dtype)
                            + "/"
                            + str(conv_state.dtype)
                            + "/"
                            + str(conv_weights.dtype)
                        )
                _fr13_conv_committed_path = (
                    True  # FR13_CONV_COMMITTED_PATH baked ON
                )
                _fr13_committed_read_cols = None
                _fr13_committed_bank_rows = None
                _fr13_committed_prior_bank = None
                try:
                    _fr10_accepted_paths_tensor = globals().get(
                        "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR"
                    )
                    _fr10_accepted_lens_tensor = globals().get(
                        "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR"
                    )
                    if _fr10_accepted_paths_tensor is None:
                        raise RuntimeError("missing_accepted_path_device_tensor")
                    if _fr10_accepted_lens_tensor is None:
                        raise RuntimeError("missing_accepted_lens_device_tensor")
                    if os.environ.get("FR10_METRICS", "0") == "1":
                        try:
                            import time as _fr10_len_time

                            _fr10_len_log = os.environ.get(
                                "FR10_TREE_GDN_COMMIT_HANDOFF_LOG"
                            )
                            _fr10_len_count = int(
                                globals().get(
                                    "_FR10_TREE_LENGTH_ALIGNMENT_LOG_COUNT", 0
                                )
                            )
                            _fr10_len_limit = int(
                                os.environ.get(
                                    "FR10_TREE_GDN_COMMIT_HANDOFF_LIMIT", "32"
                                )
                            )
                            if _fr10_len_log and _fr10_len_count < _fr10_len_limit:
                                _fr10_len_rows = []
                                for _fr10_len_b in range(
                                    int(attn_metadata.num_spec_decodes)
                                ):
                                    _fr10_meta_len = None
                                    if num_accepted_tokens is not None:
                                        _fr10_meta_len = int(
                                            num_accepted_tokens[_fr10_len_b]
                                            .detach()
                                            .cpu()
                                            .item()
                                        )
                                    _fr10_path_len = int(
                                        _fr10_accepted_lens_tensor[_fr10_len_b]
                                        .detach()
                                        .cpu()
                                        .item()
                                    )
                                    _fr10_len_rows.append(
                                        {
                                            "batch_index": int(_fr10_len_b),
                                            "metadata_num_accepted_tokens": _fr10_meta_len,
                                            "accepted_tree_len": int(_fr10_path_len),
                                            "metadata_read_col": (
                                                None
                                                if _fr10_meta_len is None
                                                else max(0, int(_fr10_meta_len) - 1)
                                            ),
                                            "accepted_len_read_col": max(
                                                0, int(_fr10_path_len) - 1
                                            ),
                                        }
                                    )
                                with open(_fr10_len_log, "a", buffering=1) as _fr10_fh:
                                    _fr10_fh.write(
                                        json.dumps(
                                            {
                                                "schema": "fr10.length_alignment.v1",
                                                "event": "tree_length_alignment",
                                                "ts": round(_fr10_len_time.time(), 4),
                                                "layer_prefix": str(self.prefix),
                                                "rows": _fr10_len_rows,
                                            }
                                        )
                                        + chr(10)
                                    )
                                globals()[
                                    "_FR10_TREE_LENGTH_ALIGNMENT_LOG_COUNT"
                                ] = _fr10_len_count + 1
                        except Exception as _fr10_len_exc:
                            if os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1":
                                raise RuntimeError(
                                    "FR10 tree length-alignment logging failed: "
                                    + type(_fr10_len_exc).__name__
                                    + ":"
                                    + str(_fr10_len_exc)
                                ) from _fr10_len_exc
                    if (
                        _fr13_conv_committed_path
                        and (
                            _FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION
                            is not None
                            or _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION
                        )
                    ):
                        # The qualified fused producer reads accepted col-0
                        # directly and publishes the commit-source stage.
                        pass
                    elif _fr13_conv_committed_path:
                        # FR13_CONV_COMMITTED_PATH (default ON): snapshot the
                        # committed-path prior conv window BEFORE the in-place
                        # remap below mutates the bank. The window is read
                        # from the accepted path's LEAF NODE column
                        # (accepted_paths[b, len-1], node-indexed layout), so
                        # it is built from the COMMITTED token path only —
                        # branch-winner-valid; spine winners are byte-identical
                        # to the legacy post-remap linear-column read (the
                        # leaf's source row is never a remap destination).
                        # FR13_CONV_COMMITTED_PATH=0 restores the legacy read.
                        # Under FR13_REPLAY_ROUTE=1 this snapshot MUST still
                        # run: it is conv-bank-only (node-indexed, pre-remap)
                        # and the conv carrier is untouched by the replay
                        # route (the remap below becomes the page-safe
                        # conv-only torch remap; the ssm half is dead).
                        if _FR13_TREE_CONV_FUSED:
                            # FR13_TREE_CONV_FUSED (FIX-3): once-per-group
                            # prepared row math + per-layer bank snapshot.
                            # The row math reads ONLY accepted paths/lens +
                            # spec indices (nothing in the interval mutates
                            # them), so the group-first layer's captured
                            # write into the init-time persistent buffers is
                            # exact for every later layer in its group; the
                            # bank index_select stays IN-LAYER and PRE-remap
                            # (per-layer snapshot semantics preserved
                            # exactly).
                            _fr13_tcf_prep_all = globals().get("_FR13_TCF_PREP")
                            _fr13_tcf_layer_group = globals().get(
                                "_FR13_TCF_LAYER_GROUP"
                            )
                            _fr13_tcf_group_owners = globals().get(
                                "_FR13_TCF_GROUP_OWNERS"
                            )
                            if (
                                not _fr13_tcf_prep_all
                                or not _fr13_tcf_layer_group
                                or not _fr13_tcf_group_owners
                            ):
                                raise RuntimeError(
                                    "FR13_TREE_CONV_FUSED engaged without "
                                    "builder-init prep buffers/owners "
                                    "(fail-loud, class 9)"
                                )
                            # PER-GROUP prep (selfcheck-proven root cause):
                            # spec_state_indices is kv-cache-group-local, so
                            # every layer must read the prep slot of ITS OWN
                            # group, written by that group's owner layer.
                            _fr13_tcf_gk = _fr13_tcf_layer_group.get(
                                str(self.prefix)
                            )
                            if _fr13_tcf_gk is None:
                                raise RuntimeError(
                                    "FR13_TREE_CONV_FUSED layer "
                                    + str(self.prefix)
                                    + " missing from the builder layer-group "
                                    "map (fail-loud, class 9)"
                                )
                            _fr13_tcf_prep = _fr13_tcf_prep_all[_fr13_tcf_gk]
                            _fr13_tcf_b = int(attn_metadata.num_spec_decodes)
                            _fr13_tcf_path_cols = min(
                                int(_fr10_accepted_paths_tensor.size(-1)),
                                int(spec_state_indices_tensor.size(-1)),
                            )
                            _fr13_tcf_rows_n = _fr13_tcf_b * _fr13_tcf_path_cols
                            if (
                                _fr13_tcf_group_owners[_fr13_tcf_gk]
                                == str(self.prefix)
                            ):
                                (
                                    _fr13_tcf_new_cols,
                                    _fr13_tcf_new_rows,
                                ) = prepare_committed_path_conv_rows(
                                    spec_state_indices=spec_state_indices_tensor,
                                    accepted_paths=_fr10_accepted_paths_tensor,
                                    num_accepted_tokens=_fr10_accepted_lens_tensor,
                                    num_spec_decodes=_fr13_tcf_b,
                                )
                                _fr13_tcf_prep["read_cols"][
                                    :_fr13_tcf_b
                                ].copy_(_fr13_tcf_new_cols)
                                _fr13_tcf_prep["bank_rows"][
                                    :_fr13_tcf_b
                                ].copy_(_fr13_tcf_new_rows)
                                if not _FR13_FIXED32_MODE:
                                    (
                                        _fr13_tcf_new_src,
                                        _fr13_tcf_new_dst,
                                    ) = prepare_replay_conv_remap_rows(
                                        spec_state_indices=spec_state_indices_tensor,
                                        accepted_paths=_fr10_accepted_paths_tensor,
                                        num_accepted_tokens=_fr10_accepted_lens_tensor,
                                        num_spec_decodes=_fr13_tcf_b,
                                        max_path_len=int(
                                            spec_state_indices_tensor.size(-1)
                                        ),
                                    )
                                    if (
                                        int(_fr13_tcf_new_src.numel())
                                        != _fr13_tcf_rows_n
                                    ):
                                        raise RuntimeError(
                                            "FR13_TREE_CONV_FUSED prepared remap "
                                            "rows disagree with the static "
                                            "path-cols geometry (fail-loud)"
                                        )
                                    _fr13_tcf_prep["src_rows"][
                                        :_fr13_tcf_rows_n
                                    ].copy_(_fr13_tcf_new_src)
                                    _fr13_tcf_prep["dst_rows"][
                                        :_fr13_tcf_rows_n
                                    ].copy_(_fr13_tcf_new_dst)
                            _fr13_committed_read_cols = _fr13_tcf_prep[
                                "read_cols"
                            ][:_fr13_tcf_b]
                            _fr13_committed_bank_rows = _fr13_tcf_prep[
                                "bank_rows"
                            ][:_fr13_tcf_b]
                            # FR13_CONV_PREGATHER consume — SERVED-PATH site
                            # (2026-07-24 rewire): the original consume sat in
                            # the deleted NPR diagnostic branch (2026-07-25) =
                            # DEAD twin, so the lever was VACUOUS on the
                            # deployed stack. Fixed32 captures one 48xB stage at
                            # final-FULL graph entry, before layer-0 consumes;
                            # every replay therefore gathers the CURRENT live
                            # ssi col0 rows. The legacy non-fixed route retains
                            # its prior-step request/page/sequence token guard.
                            # Both routes are pure copies; legacy fallback is
                            # the per-layer gather below.
                            _fr13_cpg_cbank = None
                            _fr13_cpg_capturing2 = bool(
                                torch.cuda.is_available()
                                and torch.cuda.is_current_stream_capturing()
                            )
                            _fr13_cpg_capture_ctx2 = globals().get(
                                "_FR13_FIXED32_CAPTURE_CONTEXT"
                            )
                            _fr13_cpg_accept_capture2 = False
                            if (
                                _FR13_FIXED32_MODE
                                and _fr13_cpg_capturing2
                                and _fr13_cpg_capture_ctx2 is not None
                            ):
                                _fr13_cpg_capture_desc2 = (
                                    _fr13_cpg_capture_ctx2.get("descriptor")
                                    if isinstance(_fr13_cpg_capture_ctx2, dict)
                                    else None
                                )
                                if (
                                    not isinstance(
                                        _fr13_cpg_capture_desc2, dict
                                    )
                                    or _fr13_cpg_capture_desc2.get(
                                        "runtime_mode"
                                    )
                                    != "FULL"
                                    or int(
                                        _fr13_cpg_capture_desc2.get(
                                            "num_reqs", -1
                                        )
                                    )
                                    != int(_fr13_tcf_b)
                                    or int(
                                        _fr13_cpg_capture_desc2.get(
                                            "num_tokens", -1
                                        )
                                    )
                                    != int(_fr13_tcf_b) * 32
                                ):
                                    raise RuntimeError(
                                        "FR13 fixed32 capture pregather "
                                        "context drifted"
                                    )
                                _fr13_cpg_accept_capture2 = True
                            if (
                                globals().get("_FR13_CONV_PREGATHER_ON", False)
                                and os.environ.get(
                                    "FR13_TREE_RUNROW_INIT", "1") == "1"
                            ):
                                # staged windows are COL0; only the
                                # RUNROW_INIT=1 route reads col0 here (the
                                # same env the prepare fn reads per call).
                                import lumo_flywheel_serving.fr10_gdn_tree_kernel as _fr13_cpg_kernel2
                                from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
                                    conv_col0_staged as _fr13_cpg_staged2,
                                    fixed32_conv_col0_pregather_counters
                                    as _fr13_cpg_counters2,
                                    launch_fixed32_conv_col0_pregather
                                    as _fr13_cpg_graph_stage2,
                                    selfcheck_fixed32_conv_col0_ssi_sources
                                    as _fr13_cpg_ssi_selfcheck2,
                                    validate_fixed32_conv_col0_ssi_source
                                    as _fr13_cpg_validate_ssi2,
                                )
                                _fr13_cpg_idx2 = globals().get(
                                    "_FR13_CPG_LAYER_IDX", {}).get(str(self.prefix))
                                _fr13_cpg_pairs2 = globals().get(
                                    "_LUMO_FA_SPEC_ROW_CONV_COL0", None)
                                _fr13_cpg_seq2 = globals().get(
                                    "_LUMO_FA_STEP_SEQ", None)
                                if (
                                    _FR13_FIXED32_MODE
                                    and not _fr13_cpg_capturing2
                                ):
                                    _fr13_cpg_ssi_selfcheck2(
                                        num_spec_decodes=int(_fr13_tcf_b),
                                    )
                                if (
                                    _fr13_cpg_idx2 is not None
                                    and (
                                        _fr13_cpg_accept_capture2
                                        or _fr13_cpg_seq2 is not None
                                    )
                                    and (
                                        _FR13_FIXED32_MODE
                                        or _fr13_cpg_pairs2 is not None
                                    )
                                ):
                                    if _fr13_cpg_accept_capture2:
                                        # The final FULL graph begins with one
                                        # 48xB pregather launch at fixed layer
                                        # index 0. Replay therefore refreshes
                                        # staging from the current persistent
                                        # ssi/banks before any layer consumes it,
                                        # including the first real event and
                                        # request turnover.
                                        _fr13_cpg_state2 = getattr(
                                            _fr13_cpg_kernel2,
                                            "_FR13_FIXED32_CONV_PREGATHER",
                                            {},
                                        ).get("state")
                                        _fr13_cpg_cap2 = (
                                            None
                                            if not isinstance(
                                                _fr13_cpg_state2, dict
                                            )
                                            else _fr13_cpg_state2.get(
                                                "max_batch_size"
                                            )
                                        )
                                        _fr13_cpg_staging2 = (
                                            None
                                            if not isinstance(
                                                _fr13_cpg_state2, dict
                                            )
                                            else _fr13_cpg_state2.get("staging")
                                        )
                                        if (
                                            type(_fr13_cpg_cap2) is not int
                                            or _fr13_cpg_cap2 not in (1, 2, 3, 4)
                                            or _fr13_cpg_cap2 < int(_fr13_tcf_b)
                                            or _fr13_cpg_state2.get("mode")
                                            != _FR13_FIXED32_MODE
                                            or tuple(
                                                _fr13_cpg_state2.get(
                                                    "preseeded_batches", ()
                                                )
                                            )
                                            != tuple(
                                                range(1, _fr13_cpg_cap2 + 1)
                                            )
                                            or not torch.is_tensor(
                                                _fr13_cpg_staging2
                                            )
                                            or _fr13_cpg_staging2.ndim != 3
                                            or tuple(
                                                int(_fr13_cpg_dim2)
                                                for _fr13_cpg_dim2
                                                in _fr13_cpg_staging2.shape[:2]
                                            )
                                            != (48, _fr13_cpg_cap2)
                                            or int(
                                                _fr13_cpg_staging2.shape[2]
                                            )
                                            != int(
                                                conv_state.shape[1]
                                                * conv_state.shape[2]
                                            )
                                            or _fr13_cpg_staging2.device
                                            != conv_state.device
                                            or _fr13_cpg_staging2.dtype
                                            != conv_state.dtype
                                            or not 0
                                            <= int(_fr13_cpg_idx2)
                                            < 48
                                        ):
                                            raise RuntimeError(
                                                "FR13 fixed32 capture pregather "
                                                "lease contract drifted"
                                            )
                                        _fr13_cpg_ref_bank2 = (
                                            _fr13_cpg_state2["banks"][
                                                int(_fr13_cpg_idx2)
                                            ]
                                        )
                                        if (
                                            int(_fr13_cpg_ref_bank2.data_ptr())
                                            != int(conv_state.data_ptr())
                                            or tuple(
                                                int(_fr13_cpg_dim2)
                                                for _fr13_cpg_dim2
                                                in _fr13_cpg_ref_bank2.shape
                                            )
                                            != tuple(
                                                int(_fr13_cpg_dim2)
                                                for _fr13_cpg_dim2
                                                in conv_state.shape
                                            )
                                            or tuple(
                                                int(_fr13_cpg_dim2)
                                                for _fr13_cpg_dim2
                                                in _fr13_cpg_ref_bank2.stride()
                                            )
                                            != tuple(
                                                int(_fr13_cpg_dim2)
                                                for _fr13_cpg_dim2
                                                in conv_state.stride()
                                            )
                                            or int(
                                                _fr13_cpg_ref_bank2.storage_offset()
                                            )
                                            != int(conv_state.storage_offset())
                                            or _fr13_cpg_ref_bank2.dtype
                                            != conv_state.dtype
                                            or _fr13_cpg_ref_bank2.device
                                            != conv_state.device
                                        ):
                                            raise RuntimeError(
                                                "FR13 fixed32 capture pregather "
                                                "layer/bank alias drifted"
                                            )
                                        _fr13_cpg_validate_ssi2(
                                            layer_name=str(self.prefix),
                                            layer_index=int(_fr13_cpg_idx2),
                                            spec_state_indices=(
                                                spec_state_indices_tensor
                                            ),
                                            num_spec_decodes=int(_fr13_tcf_b),
                                        )
                                        if (
                                            _fr13_cpg_capturing2
                                            or _fr13_fixed32_observed_event_active()
                                        ):
                                            _fr13_fixed32_observed_conv_source(
                                                str(self.prefix),
                                                int(_fr13_cpg_idx2),
                                                int(_fr13_tcf_b),
                                                _fr13_cpg_capturing2,
                                            )
                                        if int(_fr13_cpg_idx2) == 0:
                                            _fr13_cpg_before2 = (
                                                _fr13_cpg_counters2()
                                            )
                                            _fr13_cpg_graph_stage2(
                                                num_spec_decodes=int(
                                                    _fr13_tcf_b
                                                ),
                                                req_ids_token=None,
                                                graph_capture=True,
                                            )
                                            _fr13_cpg_after2 = (
                                                _fr13_cpg_counters2()
                                            )
                                            if (
                                                _fr13_cpg_capturing2
                                                or _fr13_fixed32_observed_event_active()
                                            ):
                                                _fr13_fixed32_observed_conv_stage(
                                                    str(self.prefix),
                                                    int(_fr13_cpg_idx2),
                                                    int(_fr13_tcf_b),
                                                    _fr13_cpg_state2,
                                                    _fr13_cpg_before2,
                                                    _fr13_cpg_after2,
                                                    _fr13_cpg_capturing2,
                                                )
                                        _fr13_cpg_f2 = _fr13_cpg_staging2[
                                            int(_fr13_cpg_idx2),
                                            : int(_fr13_tcf_b),
                                        ]
                                    else:
                                        _fr13_cpg_f2 = _fr13_cpg_staged2(
                                            (
                                                (
                                                    _FR13_FIXED32_MODE
                                                    if _FR13_FIXED32_MODE
                                                    else _fr13_cpg_pairs2
                                                ),
                                                int(_fr13_cpg_seq2) - 1,
                                            ),
                                            int(_fr13_cpg_idx2),
                                        )
                                    if (
                                        _fr13_cpg_f2 is not None
                                        and int(_fr13_cpg_f2.shape[0])
                                        == int(_fr13_tcf_b)
                                    ):
                                        _fr13_cpg_cbank = _fr13_cpg_f2.view(
                                            int(_fr13_tcf_b),
                                            *conv_state.shape[1:],
                                        )
                            if _fr13_cpg_cbank is not None:
                                _fr13_committed_prior_bank = _fr13_cpg_cbank
                            else:
                                _fr13_committed_prior_bank = (
                                    gather_committed_path_conv_prior_prepared(
                                        conv_state=conv_state,
                                        bank_rows=_fr13_committed_bank_rows,
                                    )
                                )
                            if (
                                _FR13_FIXED32_MODE
                                and (
                                    _fr13_cpg_capturing2
                                    or _fr13_fixed32_observed_event_active()
                                )
                            ):
                                _fr13_fixed32_observed_conv_consume(
                                    str(self.prefix),
                                    int(_fr13_cpg_idx2),
                                    int(_fr13_tcf_b),
                                    _fr13_cpg_cbank is not None,
                                    _fr13_cpg_capturing2,
                                )
                            if os.environ.get("FR13_TCF_SELFCHECK", "0") == "1":
                                if torch.cuda.is_current_stream_capturing():
                                    raise RuntimeError(
                                        "FR13_TCF_SELFCHECK=1 is eager-only"
                                    )
                                (
                                    _fr13_sc_cols,
                                    _fr13_sc_rows,
                                    _fr13_sc_bank,
                                ) = gather_committed_path_conv_prior(
                                    conv_state=conv_state,
                                    spec_state_indices=spec_state_indices_tensor,
                                    accepted_paths=_fr10_accepted_paths_tensor,
                                    num_accepted_tokens=_fr10_accepted_lens_tensor,
                                    num_spec_decodes=int(
                                        attn_metadata.num_spec_decodes
                                    ),
                                )
                                _fr13_tcf_sc_compare(
                                    "committed_read_cols",
                                    _fr13_committed_read_cols,
                                    _fr13_sc_cols,
                                    str(self.prefix),
                                )
                                _fr13_tcf_sc_compare(
                                    "committed_bank_rows",
                                    _fr13_committed_bank_rows,
                                    _fr13_sc_rows,
                                    str(self.prefix),
                                )
                                _fr13_tcf_sc_compare(
                                    "committed_prior_bank",
                                    _fr13_committed_prior_bank,
                                    _fr13_sc_bank,
                                    str(self.prefix),
                                )
                        else:
                            (
                                _fr13_committed_read_cols,
                                _fr13_committed_bank_rows,
                                _fr13_committed_prior_bank,
                            ) = gather_committed_path_conv_prior(
                                conv_state=conv_state,
                                spec_state_indices=spec_state_indices_tensor,
                                accepted_paths=_fr10_accepted_paths_tensor,
                                num_accepted_tokens=_fr10_accepted_lens_tensor,
                                num_spec_decodes=int(attn_metadata.num_spec_decodes),
                            )
                    # FR13_REPLAY_ROUTE: the committer replay already
                    # published accepted ssm states to LINEAR bank columns,
                    # so the ssm half of the remap is dead (and would corrupt
                    # the bank: node-column states are never staged under the
                    # flag). The conv half MUST stay: it is the FR13
                    # conv-prior-window carrier and the conv spec branch
                    # still publishes node columns.
                    #
                    # FR13_REPLAY_PAGE_SAFE_CONV_REMAP (boundary-trace root
                    # cause, 2026-06-10): conv (kv[0]) and ssm (kv[1]) are
                    # as_strided views over the SAME mamba page with
                    # stride(0) == num_element_per_page, and the Triton remap
                    # copies state.stride(0) elements per row -- a "conv-only"
                    # launch therefore copies the WHOLE page and drags
                    # never-written node-column ssm bytes over the replay's
                    # just-published linear-column ssm states (live byte
                    # prediction B.window[c] == A.post.window[node path[c]]
                    # matched 581/581 on both probed layers). Under the flag
                    # the conv half runs through the page-safe torch remap
                    # (identical permutation, copies ONLY the conv view's
                    # logical elements; frozen kernel untouched). Flag OFF
                    # keeps the legacy whole-page launch verbatim: the legacy
                    # all-rows ssm publish refreshes every window column each
                    # event, making the page-wide copy semantically identical
                    # to the intended ssm remap there.
                    if _FR13_FIXED32_MODE:
                        # The committed col0 prior was snapshotted above and the
                        # fixed32 full-node writeback below replaces every one
                        # of these remap destinations before post-accept.
                        pass
                    elif True:  # FR13_REPLAY_ROUTE baked ON
                        if _FR13_TREE_CONV_FUSED:
                            # FR13_TREE_CONV_FUSED (FIX-3): identical
                            # permutation, identical materialize-before-
                            # scatter order, conv-view-only (page-safe); the
                            # shared row math was computed once per kv-cache
                            # group above. The frozen library fn is the
                            # byte-verbatim OFF arm.
                            _fr13_sc_remap_on = (
                                os.environ.get("FR13_TCF_SELFCHECK", "0")
                                == "1"
                            )
                            if _fr13_sc_remap_on:
                                # FIX-3 selfcheck: pre-remap snapshot of the
                                # source rows; post-remap the dst rows must
                                # equal pre[src] (materialize-before-scatter
                                # permutation = the T4-anchored library
                                # semantics).
                                _fr13_sc_srcr = _fr13_tcf_prep["src_rows"][
                                    :_fr13_tcf_rows_n
                                ].to(torch.long)
                                _fr13_sc_dstr = _fr13_tcf_prep["dst_rows"][
                                    :_fr13_tcf_rows_n
                                ].to(torch.long)
                                _fr13_sc_pre_rows = conv_state.index_select(
                                    0, _fr13_sc_srcr
                                ).clone()
                            replay_conv_state_linear_remap_prepared(
                                conv_state=conv_state,
                                src_rows=_fr13_tcf_prep["src_rows"][
                                    :_fr13_tcf_rows_n
                                ],
                                dst_rows=_fr13_tcf_prep["dst_rows"][
                                    :_fr13_tcf_rows_n
                                ],
                            )
                            if _fr13_sc_remap_on:
                                _fr13_tcf_sc_compare(
                                    "remap_dst_permutation",
                                    conv_state.index_select(
                                        0, _fr13_sc_dstr
                                    ),
                                    _fr13_sc_pre_rows,
                                    str(self.prefix),
                                )
                        else:
                            replay_conv_state_linear_remap(
                                conv_state=conv_state,
                                spec_state_indices=spec_state_indices_tensor,
                                accepted_paths=_fr10_accepted_paths_tensor,
                                num_accepted_tokens=_fr10_accepted_lens_tensor,
                                num_spec_decodes=int(attn_metadata.num_spec_decodes),
                                max_path_len=int(spec_state_indices_tensor.size(-1)),
                            )
                    else:
                        launch_tree_state_linear_remap(
                            ssm_state=ssm_state,
                            conv_state=conv_state,
                            spec_state_indices=spec_state_indices_tensor,
                            accepted_paths=_fr10_accepted_paths_tensor,
                            num_accepted_tokens=_fr10_accepted_lens_tensor,
                            num_spec_decodes=int(attn_metadata.num_spec_decodes),
                            max_path_len=int(spec_state_indices_tensor.size(-1)),
                        )
                except Exception as _fr10_seed_conv_exc:
                    if (
                        _fr10_tree_conv_expected
                        and os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1"
                    ):
                        raise RuntimeError(
                            "FR10 tree state linear remap failed: "
                            + type(_fr10_seed_conv_exc).__name__
                            + ":"
                            + str(_fr10_seed_conv_exc)
                        ) from _fr10_seed_conv_exc
                if (
                    _FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION is not None
                    or _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION
                ):
                    _fr10_prior_read_mode = "sfwd_state_fusion_direct_col0"
                    _fr10_conv_read_cols = None
                    _fr10_prior_conv_bank_rows = None
                    _fr10_prior_conv_state_bank = None
                elif (
                    _fr13_conv_committed_path
                    and _fr13_committed_prior_bank is not None
                ):
                    # FR13_CONV_COMMITTED_PATH: committed-path window gathered
                    # node-indexed BEFORE the remap (see comment above the
                    # launch_tree_state_linear_remap call). read_cols here are
                    # NODE columns, not linear accepted positions.
                    _fr10_prior_read_mode = "committed_path_node"
                    _fr10_conv_read_cols = _fr13_committed_read_cols
                    _fr10_prior_conv_bank_rows = _fr13_committed_bank_rows
                    _fr10_prior_conv_state_bank = _fr13_committed_prior_bank
                else:
                    _fr10_prior_read_mode = "compact_head"
                    if _fr10_accepted_lens_tensor is None:
                        _fr10_conv_read_cols = torch.zeros(
                            (int(attn_metadata.num_spec_decodes), 1),
                            dtype=torch.long,
                            device=spec_state_indices_tensor.device,
                        )
                    else:
                        _fr10_conv_read_cols = torch.clamp(
                            (
                                _fr10_accepted_lens_tensor[
                                    : attn_metadata.num_spec_decodes
                                ]
                                - 1
                            ).to(torch.long),
                            min=0,
                            max=int(spec_state_indices_tensor.size(-1)) - 1,
                        ).view(-1, 1)
                    _fr10_prior_conv_bank_rows = spec_state_indices_tensor[
                        : attn_metadata.num_spec_decodes
                    ].gather(1, _fr10_conv_read_cols)
                    _fr10_prior_conv_state_bank = torch.index_select(
                        conv_state,
                        0,
                        _fr10_prior_conv_bank_rows.reshape(-1).to(torch.long),
                    )
                try:
                    _fr10_tree_src = None
                    _fr10_spec_env = os.environ.get("SPEC_CONFIG")
                    if _fr10_spec_env:
                        _fr10_tree_src = json.loads(_fr10_spec_env).get(
                            "speculative_token_tree"
                        )
                    if _fr10_tree_src is None and self.speculative_config is not None:
                        _fr10_tree_src = self.speculative_config.speculative_token_tree
                    _fr10_choices = sorted(
                        ast.literal_eval(_fr10_tree_src), key=lambda _p: (len(_p), _p)
                    )
                    _fr10_index = {_p: _i + 1 for _i, _p in enumerate(_fr10_choices)}
                    _fr10_parent = [-1]
                    for _fr10_choice in _fr10_choices:
                        _fr10_parent.append(
                            0
                            if len(_fr10_choice) == 1
                            else _fr10_index[_fr10_choice[:-1]]
                        )
                    _fr10_conv_parent = list(_fr10_parent)
                    _fr10_path0_node_tensor = getattr(
                        attn_metadata, "fr10_tree_path0_nodes", None
                    )
                    assert _fr10_path0_node_tensor is not None
                    _fr10_path0_nodes_py = [0] + [
                        _fr10_index[_fr10_choice]
                        for _fr10_choice in _fr10_choices
                        if all(int(_fr10_part) == 0 for _fr10_part in _fr10_choice)
                    ]
                    _fr10_tree_n = len(_fr10_parent)
                    _fr10_branch_nodes_py = [
                        _fr10_i
                        for _fr10_i in range(_fr10_tree_n)
                        if _fr10_i not in set(_fr10_path0_nodes_py)
                    ]
                    _fr10_width = int(conv_weights.shape[1])
                    _fr10_tree_source_indices = getattr(
                        attn_metadata, "fr10_tree_conv_source_indices", None
                    )[_fr10_width]
                    _fr10_flat_source_indices = getattr(
                        attn_metadata, "fr10_flat_conv_source_indices", None
                    )[_fr10_width]
                    _fr10_path0_source_indices = getattr(
                        attn_metadata, "fr10_path0_conv_source_indices", None
                    )[_fr10_width]
                    _fr10_path_node_tensors = getattr(
                        attn_metadata, "fr10_tree_path_node_tensors", None
                    )
                    assert _fr10_path_node_tensors is not None
                    _fr10_source_flat = _fr10_tree_source_indices.reshape(-1)
                    _fr10_flat_source_flat = _fr10_flat_source_indices.reshape(-1)
                    _fr10_path0_source_flat = _fr10_path0_source_indices.reshape(-1)
                    if _FR13_EAGER_PACK:
                        # FR13_EAGER_PACK (FIX-2 2h): hoist the two constant
                        # aranges (2 const kernels x 48 layers = 96/draft).
                        # Cached on the layer keyed by the FULL value domain
                        # (width, conv columns, device); populated on the
                        # first NON-capturing forward (eager warmup precedes
                        # capture). A capture-time miss builds the identical
                        # tensors like legacy WITHOUT retaining them, so no
                        # capture-pool allocation outlives its graph.
                        _fr13_ep_ar_key = (
                            int(_fr10_width),
                            int(conv_state.size(2)),
                            str(mixed_qkv_spec.device),
                        )
                        _fr13_ep_ar = getattr(
                            self, "_fr13_eager_pack_conv_arange", None
                        )
                        if _fr13_ep_ar is not None and _fr13_ep_ar[0] == _fr13_ep_ar_key:
                            _fr10_prior_col_base = _fr13_ep_ar[1]
                        else:
                            _fr10_prior_col_base = torch.arange(
                                _fr10_width - 1,
                                dtype=torch.long,
                                device=mixed_qkv_spec.device,
                            )
                            if not torch.cuda.is_current_stream_capturing():
                                self._fr13_eager_pack_conv_arange = (
                                    _fr13_ep_ar_key,
                                    _fr10_prior_col_base,
                                )
                    else:
                        _fr10_prior_col_base = torch.arange(
                            _fr10_width - 1,
                            dtype=torch.long,
                            device=mixed_qkv_spec.device,
                        )
                    if _FR13_TREE_CONV_FUSED:
                        # FR13_TREE_CONV_FUSED (FIX-3): value-static state
                        # write-back gather table + shared zero source row,
                        # layer-keyed by the FULL value domain (the FIX-2
                        # arange-cache pattern, class 6). Populated on the
                        # first NON-capturing forward (eager warmup precedes
                        # capture); a capture-time miss builds identical
                        # tensors WITHOUT retaining them, so no capture-pool
                        # allocation outlives its graph.
                        _fr13_tcf_key = (
                            tuple(_fr10_conv_parent),
                            int(_fr10_width),
                            int(conv_state.size(2)),
                            int(mixed_qkv_spec.size(1)),
                            str(mixed_qkv_spec.dtype),
                            str(mixed_qkv_spec.device),
                        )
                        _fr13_tcf_cached = getattr(
                            self, "_fr13_tree_conv_fused_static", None
                        )
                        if (
                            _fr13_tcf_cached is not None
                            and _fr13_tcf_cached[0] == _fr13_tcf_key
                        ):
                            _fr13_tcf_state_src = _fr13_tcf_cached[1]
                            _fr13_tcf_zero_row = _fr13_tcf_cached[2]
                        else:
                            _fr13_tcf_state_src = build_tree_conv_state_src_indices(
                                parent=_fr10_conv_parent,
                                width=_fr10_width,
                                state_len=int(conv_state.size(2)),
                                device=mixed_qkv_spec.device,
                            )
                            _fr13_tcf_zero_row = torch.zeros(
                                (1, int(mixed_qkv_spec.size(1))),
                                dtype=mixed_qkv_spec.dtype,
                                device=mixed_qkv_spec.device,
                            )
                            if not torch.cuda.is_current_stream_capturing():
                                self._fr13_tree_conv_fused_static = (
                                    _fr13_tcf_key,
                                    _fr13_tcf_state_src,
                                    _fr13_tcf_zero_row,
                                )
                    if not _FR13_TREE_CONV_FUSED_NEEDLE_DONE:
                        # FR13_TREE_CONV_FUSED needle (class 9): one-shot,
                        # BOTH states, first spec-decode forward (eager
                        # warmup precedes capture).
                        _fr13_tree_conv_fused_needle(
                            _FR13_TREE_CONV_FUSED,
                            _fr10_tree_n,
                            _fr10_width,
                            int(conv_state.size(2)),
                            bool(
                                _FR13_TREE_CONV_FUSED
                                and globals().get("_FR13_TCF_PREP")
                            ),
                            bool(
                                _FR13_TREE_CONV_FUSED
                                and getattr(
                                    self, "_fr13_tree_conv_fused_static", None
                                )
                                is not None
                            ),
                            bool(
                                _FR13_TREE_CONV_FUSED
                                and getattr(
                                    self, "_fr13_tree_conv_fused_static", None
                                )
                                is not None
                            ),
                        )
                    _fr12_bf16_tap_env = os.environ.get(
                        "FR12_TREE_CONV_NATIVE_BF16_TAPS"
                    )
                    if _fr12_bf16_tap_env is None:
                        _fr12_bf16_tap_env = os.environ.get(
                            "FR11_TREE_CONV_NATIVE_BF16_TAPS", "1"
                        )
                    _fr11_native_bf16_taps = _fr12_bf16_tap_env != "0"
                    if not _FR13_EAGER_PACK:
                        # FR13_EAGER_PACK (FIX-2 2h): dead cast deleted under
                        # the flag — census-verified NO consumers (1 cast x48
                        # layers/draft of pure waste); legacy keeps it for the
                        # byte-verbatim OFF arm.
                        _fr10_weight_f = conv_weights.to(torch.float32)
                    def _fr11_conv_tap_product(_fr11_x, _fr11_w):
                        if _fr11_native_bf16_taps:
                            _fr11_dtype = mixed_qkv_spec.dtype
                            _fr11_w_cast = _fr11_w.to(_fr11_dtype)
                            if _fr11_x.ndim == 2:
                                _fr11_w_cast = _fr11_w_cast.unsqueeze(0)
                            return (
                                _fr11_x.to(_fr11_dtype) * _fr11_w_cast
                            ).to(_fr11_dtype).to(torch.float32)
                        _fr11_w_f = _fr11_w.to(torch.float32)
                        if _fr11_x.ndim == 2:
                            _fr11_w_f = _fr11_w_f.unsqueeze(0)
                        return _fr11_x.to(torch.float32) * _fr11_w_f
                    _fr10_tree_conv_out = (
                        None
                        if _fr13_conv_postprep_active
                        else torch.empty_like(mixed_qkv_spec)
                    )
                    _fr10_conv_diag = getattr(
                        attn_metadata, "fr10_tree_conv_diag", None
                    )
                    _fr10_log_conv_diag = (
                        os.environ.get("FR10_METRICS", "0") == "1"
                        and _fr10_conv_diag is not None
                    )
                    if (
                        _FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION is None
                        and not _fr13_conv_postprep_active
                    ):
                        assert _fr10_prior_conv_state_bank is not None
                    _fr12_native_spine_oracle_enabled = (
                        os.environ.get("FR12_NATIVE_SPINE_ORACLE", "0") == "1"
                    )
                    _fr12_native_spine_conv_enabled = (
                        _fr12_native_spine_oracle_enabled
                        and os.environ.get("FR12_TREE_CONV_NATIVE_SPINE", "0") == "1"
                    )
                    _fr12_native_spine_conv_out = None
                    if _fr12_native_spine_conv_enabled:
                        _fr12_path0_len = int(_fr10_path0_node_tensor.numel())
                        _fr12_native_spine_x = (
                            mixed_qkv_spec.view(
                                int(attn_metadata.num_spec_decodes),
                                _fr10_tree_n,
                                mixed_qkv_spec.size(1),
                            )
                            .index_select(1, _fr10_path0_node_tensor)
                            .reshape(
                                int(attn_metadata.num_spec_decodes)
                                * _fr12_path0_len,
                                mixed_qkv_spec.size(1),
                            )
                            .contiguous()
                        )
                        _fr12_native_spine_qsl = (
                            torch.arange(
                                int(attn_metadata.num_spec_decodes) + 1,
                                dtype=spec_query_start_loc.dtype,
                                device=spec_query_start_loc.device,
                            )
                            * _fr12_path0_len
                        )
                        _fr12_native_spine_conv_out = causal_conv1d_update(
                            _fr12_native_spine_x,
                            conv_state,
                            conv_weights,
                            self.conv1d.bias,
                            self.activation,
                            conv_state_indices=spec_state_indices_tensor[:, 0][
                                : attn_metadata.num_spec_decodes
                            ],
                            num_accepted_tokens=num_accepted_tokens,
                            query_start_loc=_fr12_native_spine_qsl,
                            max_query_len=_fr12_path0_len,
                            validate_data=False,
                        ).view(
                            int(attn_metadata.num_spec_decodes),
                            _fr12_path0_len,
                            mixed_qkv_spec.size(1),
                        )
                    _fr13_wbb_stage = None
                    _fr13_wbb_srows = 0
                    _fr13_f32_source_batch = None
                    _fr13_f32_prior_windows = None
                    if _FR13_FIXED32_CONV_SOURCE_BATCH:
                        if not _FR13_FIXED32_MODE or not _FR13_CONV_WB_BATCHED:
                            raise RuntimeError(
                                "FR13_FIXED32_CONV_SOURCE_BATCH requires the "
                                "fixed32 batched-writeback staging route"
                            )
                        _fr13_f32_source_b = int(
                            attn_metadata.num_spec_decodes
                        )
                        _fr13_wbb_srows = (
                            int(_fr10_prior_col_base.numel())
                            + int(_fr10_tree_n)
                            + 1
                        )
                        _fr13_wbb_stage = conv_wb_staging_get(
                            str(self.prefix),
                            _fr13_f32_source_b * _fr13_wbb_srows,
                            int(mixed_qkv_spec.size(1)),
                            mixed_qkv_spec.dtype,
                            mixed_qkv_spec.device,
                        )
                        (
                            _fr13_f32_source_batch,
                            _fr13_f32_prior_windows,
                        ) = fused_tree_conv_sources_batched(
                            prior_bank=_fr10_prior_conv_state_bank,
                            prior_cols=_fr10_prior_col_base,
                            x=mixed_qkv_spec,
                            zero_row=_fr13_tcf_zero_row,
                            staging=_fr13_wbb_stage,
                            batch=_fr13_f32_source_b,
                            tree_n=_fr10_tree_n,
                        )
                    _fr13_sfwd_gate_enabled = False
                    _fr13_sfwd_task_markers = None
                    _fr13_sfwd_task_marker = None
                    _fr13_sfwd_candidate_kind = None
                    _fr13_sfwd_candidate_out = None
                    _fr13_sfwd_candidate_stage = None
                    _fr13_sfwd_production = (
                        _FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION
                    )
                    if _FR13_FIXED32_MODE and _fr13_sfwd_production is None:
                        if _FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB:
                            (
                                _fr13_sfwd_gate_enabled,
                                _fr13_sfwd_task_marker,
                            ) = fixed32_sfwd_prior_reuse_gate_control()
                            _fr13_sfwd_candidate_kind = "prior_reuse"
                        else:
                            (
                                _fr13_sfwd_gate_enabled,
                                _fr13_sfwd_task_markers,
                            ) = fixed32_sfwd_state_fusion_gate_control()
                            _fr13_sfwd_candidate_kind = "rowgroup8"
                    if _FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB:
                        (
                            _fr13_conv_postprep_gate_enabled,
                            _fr13_conv_postprep_task_marker,
                        ) = fixed32_sfwd_conv_postprep_gate_control(
                            fixed32_mode=_FR13_FIXED32_MODE,
                        )
                        _fr13_conv_postprep_candidate = bool(
                            _fr13_conv_postprep_task_marker is not None
                        )
                    if _fr13_conv_postprep_candidate:
                        _fr13_conv_postprep_b = int(
                            attn_metadata.num_spec_decodes
                        )
                        if (
                            not _FR13_FIXED32_MODE
                            or not 1 <= _fr13_conv_postprep_b <= 4
                            or not _FR13_CONV_WB_BATCHED
                            or not _FR13_TREE_CONV_FUSED
                            or _FR13_FIXED32_CONV_SOURCE_BATCH
                            or os.environ.get("FR13_RING_EXPORT", "1") != "1"
                            or not _FR13_FLAGS_INKERNEL
                            or os.environ.get("FR13_TREE_RUNROW_INIT", "1")
                            != "1"
                            or self.activation not in (True, "silu", "swish")
                            or _fr12_native_spine_conv_out is not None
                            or _fr12_subkernel_capture_enabled
                            or _fr13_gdn_subop_mab_on
                            or _fr13_sfwd_gate_enabled
                            or _fr13_sfwd_production is not None
                            or int(_fr10_tree_n) != 32
                            or int(conv_state.size(2)) != 34
                            or int(_fr10_width) != 4
                        ):
                            raise RuntimeError(
                                "FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION "
                                "dependency/geometry contract drifted"
                            )
                        _fr13_wbb_srows = 36
                        _fr13_wbb_stage = conv_wb_staging_get(
                            str(self.prefix),
                            _fr13_conv_postprep_b * _fr13_wbb_srows,
                            int(mixed_qkv_spec.size(1)),
                            mixed_qkv_spec.dtype,
                            mixed_qkv_spec.device,
                        )
                        _fr13_conv_postprep_rows = (
                            _fr13_conv_postprep_b * 32
                        )
                        _fr13_conv_postprep_cache_key = (
                            _fr13_conv_postprep_b,
                            str(mixed_qkv_spec.dtype),
                            str(mixed_qkv_spec.device),
                        )
                        _fr13_conv_postprep_cache = getattr(
                            self,
                            "_fr13_fixed32_sfwd_conv_postprep_outputs",
                            None,
                        )
                        _fr13_conv_postprep_capturing = bool(
                            torch.cuda.is_available()
                            and torch.cuda.is_current_stream_capturing()
                        )
                        _fr13_conv_postprep_graph_cache = bool(
                            isinstance(_fr13_conv_postprep_cache, dict)
                            and _fr13_conv_postprep_cache.get("schema")
                            == "fr13.fixed32.sfwd_conv_postprep.capture_cache.v1"
                        )
                        _fr13_conv_postprep_profile_graph_cache = bool(
                            _fr13_conv_postprep_graph_cache
                            and _fr13_conv_postprep_cache.get(
                                "profile_capture"
                            )
                            is True
                        )
                        _fr13_conv_postprep_binding = None
                        _fr13_conv_postprep_ssi = spec_state_indices_tensor
                        _fr13_conv_postprep_conv_state = conv_state
                        _fr13_conv_postprep_source_stage = _fr13_wbb_stage
                        if (
                            _FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB
                            and not _fr13_conv_postprep_graph_cache
                        ):
                            _fr13_conv_postprep_source_stage = torch.empty_like(
                                _fr13_wbb_stage[
                                    : _fr13_conv_postprep_b * _fr13_wbb_srows
                                ]
                            )
                        if _fr13_conv_postprep_graph_cache:
                            _fr13_conv_postprep_entry = (
                                _fr13_conv_postprep_cache.get("by_batch", {})
                            ).get(_fr13_conv_postprep_b)
                            if not isinstance(_fr13_conv_postprep_entry, dict):
                                raise RuntimeError(
                                    "FR13 SFWD conv/post-prep graph output cache "
                                    "does not cover the capture batch"
                                )
                            _fr13_conv_postprep_query = (
                                _fr13_conv_postprep_entry["query"]
                            )
                            _fr13_conv_postprep_key = (
                                _fr13_conv_postprep_entry["key_tensor"]
                            )
                            _fr13_conv_postprep_value_spec = (
                                _fr13_conv_postprep_entry["value_spec"]
                            )
                            _fr13_conv_postprep_value_tree = (
                                _fr13_conv_postprep_entry["value_tree"]
                            )
                            _fr13_conv_postprep_g = (
                                _fr13_conv_postprep_entry["g"]
                            )
                            _fr13_conv_postprep_beta = (
                                _fr13_conv_postprep_entry["beta"]
                            )
                            _fr13_conv_postprep_ssi = (
                                _fr13_conv_postprep_entry["spec_state_indices"]
                            )
                            _fr13_conv_postprep_conv_state = (
                                _fr13_conv_postprep_entry["conv_state"]
                            )
                            _fr13_conv_postprep_source_stage = (
                                _fr13_conv_postprep_entry["source_stage"]
                            )
                            _fr13_conv_postprep_binding = (
                                _fr13_conv_postprep_entry["capture_binding"]
                            )
                            if (
                                (
                                    not _fr13_conv_postprep_profile_graph_cache
                                    and _fr13_wbb_stage
                                    is not _fr13_conv_postprep_source_stage
                                )
                                or int(conv_state.data_ptr())
                                != int(_fr13_conv_postprep_conv_state.data_ptr())
                                or tuple(int(value) for value in conv_state.shape)
                                != tuple(
                                    int(value)
                                    for value in _fr13_conv_postprep_conv_state.shape
                                )
                                or tuple(int(value) for value in conv_state.stride())
                                != tuple(
                                    int(value)
                                    for value in _fr13_conv_postprep_conv_state.stride()
                                )
                                or int(conv_state.storage_offset())
                                != int(
                                    _fr13_conv_postprep_conv_state.storage_offset()
                                )
                            ):
                                # Built only on the failure path: this runs per
                                # layer on every graph-cache forward, so the
                                # timing arm must not pay for the observation.
                                raise RuntimeError(
                                    "FR13 SFWD conv/post-prep graph input lease "
                                    "object/data_ptr drifted: "
                                    + repr(
                                        {
                                            "layer": str(self.prefix),
                                            "batch": _fr13_conv_postprep_b,
                                            "profile_cache": bool(
                                                _fr13_conv_postprep_profile_graph_cache
                                            ),
                                            "staging_identity": bool(
                                                _fr13_wbb_stage
                                                is _fr13_conv_postprep_source_stage
                                            ),
                                            "conv_state_observed": (
                                                int(conv_state.data_ptr()),
                                                tuple(
                                                    int(value)
                                                    for value in conv_state.shape
                                                ),
                                                tuple(
                                                    int(value)
                                                    for value in conv_state.stride()
                                                ),
                                                int(conv_state.storage_offset()),
                                            ),
                                            "conv_state_bound": (
                                                int(
                                                    _fr13_conv_postprep_conv_state.data_ptr()
                                                ),
                                                tuple(
                                                    int(value)
                                                    for value in _fr13_conv_postprep_conv_state.shape
                                                ),
                                                tuple(
                                                    int(value)
                                                    for value in _fr13_conv_postprep_conv_state.stride()
                                                ),
                                                int(
                                                    _fr13_conv_postprep_conv_state.storage_offset()
                                                ),
                                            ),
                                        }
                                    )
                                )
                        elif _fr13_conv_postprep_capturing:
                            raise RuntimeError(
                                "FR13 SFWD conv/post-prep capture lacks preseeded "
                                "output bindings"
                            )
                        elif _fr13_conv_postprep_cache is None:
                            _fr13_conv_postprep_query = torch.empty(
                                (1, _fr13_conv_postprep_rows, 16, 128),
                                dtype=mixed_qkv_spec.dtype,
                                device=mixed_qkv_spec.device,
                            )
                            _fr13_conv_postprep_key = torch.empty_like(
                                _fr13_conv_postprep_query
                            )
                            _fr13_conv_postprep_value_spec = torch.empty(
                                (1, _fr13_conv_postprep_rows, 48, 128),
                                dtype=mixed_qkv_spec.dtype,
                                device=mixed_qkv_spec.device,
                            )
                            _fr13_conv_postprep_value_tree = torch.empty(
                                (_fr13_conv_postprep_rows, 48, 128),
                                dtype=mixed_qkv_spec.dtype,
                                device=mixed_qkv_spec.device,
                            )
                            _fr13_conv_postprep_g = torch.empty(
                                (_fr13_conv_postprep_rows, 48),
                                dtype=torch.float32,
                                device=mixed_qkv_spec.device,
                            )
                            _fr13_conv_postprep_beta = torch.empty_like(
                                _fr13_conv_postprep_g
                            )
                            _fr13_conv_postprep_cache = {
                                "key": _fr13_conv_postprep_cache_key,
                                "query": _fr13_conv_postprep_query,
                                "key_tensor": _fr13_conv_postprep_key,
                                "value_spec": _fr13_conv_postprep_value_spec,
                                "value_tree": _fr13_conv_postprep_value_tree,
                                "g": _fr13_conv_postprep_g,
                                "beta": _fr13_conv_postprep_beta,
                            }
                            if _fr13_conv_postprep_profile_capture:
                                _fr13_conv_postprep_cache[
                                    "profile_capture_pending"
                                ] = {
                                    "batch_size": _fr13_conv_postprep_b,
                                    "spec_state_indices": (
                                        spec_state_indices_tensor
                                    ),
                                    "conv_state": conv_state,
                                    "source_stage": (
                                        _fr13_conv_postprep_source_stage
                                    ),
                                }
                            self._fr13_fixed32_sfwd_conv_postprep_outputs = (
                                _fr13_conv_postprep_cache
                            )
                        else:
                            if (
                                _fr13_conv_postprep_cache.get("key")
                                != _fr13_conv_postprep_cache_key
                            ):
                                raise RuntimeError(
                                    "FR13 SFWD conv/post-prep persistent output "
                                    "geometry changed after initialization"
                                )
                            _fr13_conv_postprep_query = (
                                _fr13_conv_postprep_cache["query"]
                            )
                            _fr13_conv_postprep_key = (
                                _fr13_conv_postprep_cache["key_tensor"]
                            )
                            _fr13_conv_postprep_value_spec = (
                                _fr13_conv_postprep_cache["value_spec"]
                            )
                            _fr13_conv_postprep_value_tree = (
                                _fr13_conv_postprep_cache["value_tree"]
                            )
                            _fr13_conv_postprep_g = (
                                _fr13_conv_postprep_cache["g"]
                            )
                            _fr13_conv_postprep_beta = (
                                _fr13_conv_postprep_cache["beta"]
                            )
                            if _fr13_conv_postprep_profile_capture:
                                _fr13_conv_postprep_cache[
                                    "profile_capture_pending"
                                ] = {
                                    "batch_size": _fr13_conv_postprep_b,
                                    "spec_state_indices": (
                                        spec_state_indices_tensor
                                    ),
                                    "conv_state": conv_state,
                                    "source_stage": (
                                        _fr13_conv_postprep_source_stage
                                    ),
                                }
                        launch_fixed32_sfwd_conv_postprep_fusion(
                            x=mixed_qkv_spec,
                            conv_state=_fr13_conv_postprep_conv_state,
                            spec_state_indices=_fr13_conv_postprep_ssi,
                            conv_weights=conv_weights,
                            bias=self.conv1d.bias,
                            a=a,
                            b=b,
                            A_log=self.A_log,
                            dt_bias=self.dt_bias,
                            query=_fr13_conv_postprep_query,
                            key=_fr13_conv_postprep_key,
                            value_spec=_fr13_conv_postprep_value_spec,
                            value_tree=_fr13_conv_postprep_value_tree,
                            g=_fr13_conv_postprep_g,
                            beta=_fr13_conv_postprep_beta,
                            source_stage=_fr13_conv_postprep_source_stage,
                            conv_tap=None,
                            batch_size=_fr13_conv_postprep_b,
                            fixed32_mode=_FR13_FIXED32_MODE,
                            tree_parent=_fr10_parent,
                            qualification_profile="k64_root",
                            draft_vocab_k=65536,
                            draft_vocab_root=1,
                            embed_gate_cta=(
                                _FR13_FIXED32_SFWD_EMBED_GATE_CTA
                            ),
                            direct_nodegroup8=(
                                _FR13_FIXED32_SFWD_NODEGROUP8_DIRECT
                            ),
                            source_only_qualification=True,
                            capture_binding=_fr13_conv_postprep_binding,
                        )
                        if _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION and (
                            _fr13_conv_postprep_capturing
                            or _fr13_fixed32_observed_event_active()
                        ):
                            # One fused call per layer replaces the pregather
                            # stage plus this layer's conv consume; the census
                            # counts it as its own work class.
                            _fr13_fixed32_observed_sfwd_conv_postprep(
                                str(self.prefix),
                                int(_fr13_conv_postprep_b),
                                _fr13_conv_postprep_capturing,
                            )
                        if (
                            _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION
                            and not getattr(
                                self,
                                "_fr13_sfwd_conv_postprep_production_engaged",
                                False,
                            )
                        ):
                            print(
                                "[FR13_SFWD_CONV_POSTPREP] production engaged "
                                f"layer={self.prefix} B={_fr13_conv_postprep_b} "
                                f"rows={_fr13_conv_postprep_rows}",
                                flush=True,
                            )
                            self._fr13_sfwd_conv_postprep_production_engaged = True
                    elif _fr13_sfwd_production is not None:
                        if (
                            not _FR13_FIXED32_MODE
                            or not _FR13_CONV_WB_BATCHED
                            or not _FR13_TREE_CONV_FUSED
                            or _FR13_FIXED32_CONV_SOURCE_BATCH
                            or os.environ.get("FR13_RING_EXPORT", "1") != "1"
                            or not _FR13_FLAGS_INKERNEL
                            or os.environ.get("FR13_TREE_RUNROW_INIT", "1")
                            != "1"
                            or self.activation not in (True, "silu", "swish")
                            or _fr12_native_spine_conv_out is not None
                            or int(attn_metadata.num_spec_decodes) != 1
                            or int(_fr10_tree_n) != 32
                            or int(conv_state.size(2)) != 34
                            or int(_fr10_width) != 4
                        ):
                            raise RuntimeError(
                                "FR13_FIXED32_SFWD_STATE_FUSION production "
                                "dependency/geometry contract drifted"
                            )
                        from lumo_flywheel_serving.fr13_tree_conv_fused import (
                            conv_wb_staging_get as _fr13_sfwd_stage_get,
                        )
                        _fr13_wbb_srows = (
                            int(_fr10_width) - 1 + int(_fr10_tree_n) + 1
                        )
                        _fr13_wbb_stage = _fr13_sfwd_stage_get(
                            str(self.prefix),
                            _fr13_wbb_srows,
                            int(mixed_qkv_spec.size(1)),
                            mixed_qkv_spec.dtype,
                            mixed_qkv_spec.device,
                        )
                        launch_fixed32_sfwd_state_fusion(
                            x=mixed_qkv_spec,
                            conv_state=conv_state,
                            spec_state_indices=spec_state_indices_tensor,
                            source_flat=_fr10_source_flat,
                            conv_weights=conv_weights,
                            bias=self.conv1d.bias,
                            out=_fr10_tree_conv_out,
                            source_stage=_fr13_wbb_stage[:_fr13_wbb_srows],
                            batch_size=1,
                            tree_rows=int(_fr10_tree_n),
                        )
                        fixed32_sfwd_state_fusion_production_engagement(
                            credential=_fr13_sfwd_production,
                            layer_key=int(conv_weights.data_ptr()),
                            batch_size=1,
                        )
                    elif (
                        _fr13_sfwd_gate_enabled
                        and (
                            (
                                _fr13_sfwd_candidate_kind == "prior_reuse"
                                and _fr13_sfwd_task_marker is not None
                                and int(attn_metadata.num_spec_decodes) == 1
                            )
                            or (
                                _fr13_sfwd_candidate_kind == "rowgroup8"
                                and _fr13_sfwd_task_markers is not None
                                and int(attn_metadata.num_spec_decodes) == 4
                            )
                        )
                    ):
                        # Source-only/default-off qualification: run the fused
                        # conv/state producer first on the authenticated real
                        # event, then execute and serve the incumbent below.
                        # No candidate bytes become model inputs in this arm.
                        if (
                            not _FR13_CONV_WB_BATCHED
                            or not _FR13_TREE_CONV_FUSED
                            or os.environ.get("FR13_RING_EXPORT", "1") != "1"
                            or not _FR13_FLAGS_INKERNEL
                            or os.environ.get("FR13_TREE_RUNROW_INIT", "1")
                            != "1"
                            or self.activation not in (True, "silu", "swish")
                            or _fr12_native_spine_conv_out is not None
                            or int(_fr10_tree_n) != 32
                            or int(conv_state.size(2)) != 34
                            or int(_fr10_width) != 4
                        ):
                            raise RuntimeError(
                                "FR13_FIXED32_SFWD_STATE_FUSION dependency/"
                                "geometry contract drifted"
                            )
                        from lumo_flywheel_serving.fr13_tree_conv_fused import (
                            conv_wb_staging_get as _fr13_sfwd_stage_get,
                        )
                        _fr13_sfwd_b = int(
                            attn_metadata.num_spec_decodes
                        )
                        _fr13_sfwd_srows = (
                            int(_fr10_width) - 1 + int(_fr10_tree_n) + 1
                        )
                        _fr13_sfwd_reference_stage = (
                            _fr13_sfwd_stage_get(
                                str(self.prefix),
                                _fr13_sfwd_b * _fr13_sfwd_srows,
                                int(mixed_qkv_spec.size(1)),
                                mixed_qkv_spec.dtype,
                                mixed_qkv_spec.device,
                            )
                        )
                        _fr13_sfwd_candidate_out = torch.empty(
                            tuple(mixed_qkv_spec.shape),
                            dtype=mixed_qkv_spec.dtype,
                            device=mixed_qkv_spec.device,
                        )
                        _fr13_sfwd_candidate_stage = torch.empty_like(
                            _fr13_sfwd_reference_stage[
                                : _fr13_sfwd_b * _fr13_sfwd_srows
                            ]
                        )
                        if _fr13_sfwd_candidate_kind == "prior_reuse":
                            launch_fixed32_sfwd_prior_reuse(
                                x=mixed_qkv_spec,
                                conv_state=conv_state,
                                spec_state_indices=spec_state_indices_tensor,
                                tree_parent=_fr10_parent,
                                conv_weights=conv_weights,
                                bias=self.conv1d.bias,
                                out=_fr13_sfwd_candidate_out,
                                source_stage=_fr13_sfwd_candidate_stage,
                                batch_size=_fr13_sfwd_b,
                                tree_rows=int(_fr10_tree_n),
                            )
                        else:
                            launch_fixed32_sfwd_state_fusion(
                                x=mixed_qkv_spec,
                                conv_state=conv_state,
                                spec_state_indices=spec_state_indices_tensor,
                                source_flat=_fr10_source_flat,
                                conv_weights=conv_weights,
                                bias=self.conv1d.bias,
                                out=_fr13_sfwd_candidate_out,
                                source_stage=_fr13_sfwd_candidate_stage,
                                batch_size=_fr13_sfwd_b,
                                tree_rows=int(_fr10_tree_n),
                            )
                    for _fr10_b in range(
                        0
                        if (
                            _fr13_sfwd_production is not None
                            or _fr13_conv_postprep_active
                        )
                        else attn_metadata.num_spec_decodes
                    ):
                        _fr10_start = _fr10_b * _fr10_tree_n
                        _fr10_end = _fr10_start + _fr10_tree_n
                        _fr10_x = mixed_qkv_spec[_fr10_start:_fr10_end]
                        _fr10_prior_cols = _fr10_prior_col_base
                        if _FR13_FIXED32_CONV_SOURCE_BATCH:
                            _fr10_prior_window = _fr13_f32_prior_windows[
                                _fr10_b
                            ]
                            _fr10_source = _fr13_f32_source_batch[_fr10_b]
                        else:
                            _fr10_prior_window = _fr10_prior_conv_state_bank[
                                _fr10_b
                            ].index_select(1, _fr10_prior_cols)
                        if not _FR13_FIXED32_CONV_SOURCE_BATCH:
                            if _FR13_TREE_CONV_FUSED:
                                # FR13_TREE_CONV_FUSED (FIX-3): ONE shared
                                # source with the zero row appended for the
                                # write-back gather. Window/flat-source
                                # indices never reference the appended row
                                # (all < width-1 + tree_n; CPU-executed
                                # invariance check in the byte A/B), so every
                                # downstream window index_select output is
                                # byte-identical to the legacy two-operand
                                # cat.
                                _fr10_source = fused_tree_conv_source(
                                    prior_window=_fr10_prior_window,
                                    x=_fr10_x,
                                    zero_row=_fr13_tcf_zero_row,
                                )
                            else:
                                _fr10_source = torch.cat(
                                    (_fr10_prior_window.transpose(0, 1), _fr10_x),
                                    dim=0,
                                )
                        if _FR13_CONV_WB_BATCHED:
                            # B2c: stage this request's shared source for the
                            # ONE batched writeback after the loop. copy_
                            # preserves bytes; downstream window reads keep
                            # using _fr10_source unchanged.
                            if _FR13_FIXED32_CONV_SOURCE_BATCH:
                                if (
                                    _fr13_wbb_stage is None
                                    or _fr10_source.data_ptr()
                                    != _fr13_wbb_stage[
                                        _fr10_b * _fr13_wbb_srows
                                    ].data_ptr()
                                ):
                                    raise RuntimeError(
                                        "FR13 fixed32 batched source staging "
                                        "identity drift"
                                    )
                            elif _fr13_wbb_stage is None:
                                from lumo_flywheel_serving.fr13_tree_conv_fused import (
                                    conv_wb_staging_get as _fr13_wbb_get,
                                )
                                _fr13_wbb_srows = int(_fr10_source.size(0))
                                # capacity-keyed get: preseeded at builder
                                # init with the row-cap bound; this never
                                # reallocs (fail-loud under capture if the
                                # preseed is missing/undersized).
                                _fr13_wbb_stage = _fr13_wbb_get(
                                    str(self.prefix),
                                    int(attn_metadata.num_spec_decodes)
                                    * _fr13_wbb_srows,
                                    int(_fr10_source.size(1)),
                                    _fr10_source.dtype,
                                    _fr10_source.device,
                                )
                            if not _FR13_FIXED32_CONV_SOURCE_BATCH:
                                _fr13_wbb_stage[
                                    _fr10_b * _fr13_wbb_srows:
                                    (_fr10_b + 1) * _fr13_wbb_srows
                                ].copy_(_fr10_source)
                        _fr10_window = _fr10_source.index_select(
                            0, _fr10_source_flat
                        ).view(_fr10_tree_n, _fr10_width, _fr10_x.size(1))
                        if not _FR13_EAGER_PACK:
                            # FR13_EAGER_PACK (FIX-2 2h): _fr10_path0_x is NOT
                            # dead (FR10_METRICS=1 consumers) — census caution.
                            # Under the flag it moves INSIDE the two gated
                            # diagnostic branches below (identical inputs,
                            # identical value); it is never deleted.
                            _fr10_path0_x = _fr10_x.index_select(
                                0, _fr10_path0_node_tensor
                            )
                        if _fr10_b == 0:
                            try:
                                _fr12_payload = _fr12_subkernel_capture_get(
                                    self, create=False
                                )
                            except Exception as _fr12_tree_detail_exc:
                                logger.warning(
                                    "FR12 tree conv detail capture failed: %s",
                                    _fr12_tree_detail_exc,
                                )
                        if _FR13_TREE_CONV_FUSED:
                            # FR13_TREE_CONV_FUSED (FIX-3): ONE bf16
                            # elementwise mul + ONE fp32 cast + bias
                            # broadcast + EXPLICIT ordered adds — identical
                            # per-element ops and the legacy operand/add
                            # order (reduction ops are BANNED in the fused
                            # arm: unspecified order breaks bit-exactness).
                            # The silu below is the SAME triton kernel
                            # object/launch as legacy (shared lines).
                            _fr10_acc = fused_tree_conv_taps_acc(
                                window=_fr10_window,
                                conv_weights=conv_weights,
                                bias=self.conv1d.bias,
                            )
                        else:
                            if self.conv1d.bias is None:
                                _fr10_acc = torch.zeros_like(_fr10_x, dtype=torch.float32)
                            else:
                                _fr10_acc = self.conv1d.bias.to(torch.float32).unsqueeze(
                                    0
                                ).expand_as(_fr10_x.float()).clone()
                            for _fr10_col in range(_fr10_width):
                                _fr10_acc = _fr10_acc + _fr11_conv_tap_product(
                                    _fr10_window[:, _fr10_col, :],
                                    conv_weights[:, _fr10_col],
                                )
                        if self.activation in (True, "silu", "swish"):
                            _fr10_out = triton_ex2_silu_bf16(
                                _fr10_acc, out_dtype=mixed_qkv_spec.dtype
                            )
                        else:
                            _fr10_out = _fr10_acc.to(dtype=mixed_qkv_spec.dtype)
                        if _fr12_native_spine_conv_out is not None:
                            _fr10_out.index_copy_(
                                0,
                                _fr10_path0_node_tensor,
                                _fr12_native_spine_conv_out[_fr10_b],
                            )
                        _fr10_tree_conv_out[_fr10_start:_fr10_end] = _fr10_out
                        if _fr10_b == 0 and _fr13_conv_subop_mab_enabled():
                            # FR13_CONV_SUBOP_MAB (default OFF): observe-only
                            # fused-conv M-invariance A/B on the spine request.
                            # _fr10_acc is the PRE-splice full-M taps output.
                            _fr13_conv_subop_mab(
                                acc=_fr10_acc,
                                window=_fr10_window,
                                spine=_fr10_path0_node_tensor,
                                conv_weights=conv_weights,
                                bias=self.conv1d.bias,
                                activation=self.activation,
                                dtype=mixed_qkv_spec.dtype,
                                layer_prefix=str(getattr(self, "prefix", "")),
                                tree_n=_fr10_tree_n,
                            )
                        if _FR13_TREE_CONV_FUSED:
                            # FR13_TREE_CONV_FUSED (FIX-3): the per-node
                            # state write-back loop (7 device nodes x tree_n
                            # — THE tree_n node multiplier, ~44-52% of the
                            # census extra nodes) as ONE static-index gather
                            # over the shared source (class 3
                            # gather-then-scatter; the index_copy_ below is
                            # unchanged). Pure data movement, zero
                            # arithmetic: new_state[i][:, j] = (prior ++
                            # x[path_i] ++ zeros)[path_len_i + j] — the
                            # closed-form composition of the loop's index
                            # math, CPU-proven byte-identical per topology
                            # in tests/test_fr13_tree_conv_fused_byte_ab.py.
                            # FR13_CONV_WB_FUSED (B2a): when armed AND no
                            # diagnostic consumer needs the materialized
                            # rows, skip the gather+contiguous here and let
                            # the single fused kernel at the scatter site do
                            # gather->page-write in one launch (byte-copy
                            # contract; nsys-measured ~8ms/draft pair).
                            _fr13_conv_wb_fused_now = (
                                os.environ.get("FR13_CONV_WB_FUSED", "1")
                                == "1"
                                and os.environ.get(
                                    "FR13_TCF_SELFCHECK", "0"
                                )
                                != "1"
                                and not os.environ.get(
                                    "FR10_TREE_GDN_COMMIT_HANDOFF_LOG"
                                )
                                and not os.environ.get(
                                    "FR10_TREE_GDN_SRC_NATIVE_PAYLOAD"
                                )
                                and not _fr10_log_conv_diag
                            )
                            if _fr13_conv_wb_fused_now:
                                _fr10_new_state = None
                            else:
                                _fr10_new_state = fused_tree_conv_state_rows(
                                    source_z=_fr10_source,
                                    state_src=_fr13_tcf_state_src,
                                    tree_n=_fr10_tree_n,
                                    state_len=int(conv_state.size(2)),
                                ).to(dtype=conv_state.dtype)
                            if os.environ.get("FR13_TCF_SELFCHECK", "0") == "1":
                                # FIX-3 dual-path selfcheck: recompute the
                                # ENTIRE legacy conv section on the same
                                # inputs; bitwise compare window/acc/out/
                                # new_state (covers source + taps + silu).
                                _fr13_sc_source = torch.cat(
                                    (_fr10_prior_window.transpose(0, 1), _fr10_x),
                                    dim=0,
                                )
                                _fr13_sc_window = _fr13_sc_source.index_select(
                                    0, _fr10_source_flat
                                ).view(_fr10_tree_n, _fr10_width, _fr10_x.size(1))
                                _fr13_tcf_sc_compare(
                                    "conv_window", _fr10_window,
                                    _fr13_sc_window, str(self.prefix),
                                )
                                if self.conv1d.bias is None:
                                    _fr13_sc_acc = torch.zeros_like(
                                        _fr10_x, dtype=torch.float32
                                    )
                                else:
                                    _fr13_sc_acc = self.conv1d.bias.to(
                                        torch.float32
                                    ).unsqueeze(0).expand_as(
                                        _fr10_x.float()
                                    ).clone()
                                for _fr13_sc_col in range(_fr10_width):
                                    _fr13_sc_acc = (
                                        _fr13_sc_acc
                                        + _fr11_conv_tap_product(
                                            _fr13_sc_window[:, _fr13_sc_col, :],
                                            conv_weights[:, _fr13_sc_col],
                                        )
                                    )
                                _fr13_tcf_sc_compare(
                                    "conv_acc", _fr10_acc, _fr13_sc_acc,
                                    str(self.prefix),
                                )
                                if self.activation in (True, "silu", "swish"):
                                    _fr13_sc_out = triton_ex2_silu_bf16(
                                        _fr13_sc_acc,
                                        out_dtype=mixed_qkv_spec.dtype,
                                    )
                                else:
                                    _fr13_sc_out = _fr13_sc_acc.to(
                                        dtype=mixed_qkv_spec.dtype
                                    )
                                _fr13_tcf_sc_compare(
                                    "conv_out", _fr10_out, _fr13_sc_out,
                                    str(self.prefix),
                                )
                                _fr13_sc_rows_l = []
                                for _fr13_sc_i in range(_fr10_tree_n):
                                    _fr13_sc_path = _fr10_path_node_tensors[
                                        _fr13_sc_i
                                    ]
                                    _fr13_sc_nx = _fr10_x.index_select(
                                        0, _fr13_sc_path
                                    )
                                    _fr13_sc_src2 = torch.cat(
                                        (
                                            _fr10_prior_window.transpose(0, 1),
                                            _fr13_sc_nx,
                                        ),
                                        dim=0,
                                    )
                                    _fr13_sc_src2 = torch.cat(
                                        (
                                            _fr13_sc_src2,
                                            _fr10_x.new_zeros(
                                                (
                                                    int(conv_state.size(2)),
                                                    int(_fr10_x.size(1)),
                                                )
                                            ),
                                        ),
                                        dim=0,
                                    )
                                    _fr13_sc_sidx = (
                                        _fr13_sc_path.numel()
                                        + torch.arange(
                                            conv_state.size(2),
                                            dtype=torch.long,
                                            device=mixed_qkv_spec.device,
                                        )
                                    )
                                    _fr13_sc_rows_l.append(
                                        _fr13_sc_src2.index_select(
                                            0, _fr13_sc_sidx
                                        ).transpose(0, 1)
                                    )
                                _fr13_sc_new_state = torch.stack(
                                    _fr13_sc_rows_l, dim=0
                                ).to(dtype=conv_state.dtype)
                                _fr13_tcf_sc_compare(
                                    "conv_new_state", _fr10_new_state,
                                    _fr13_sc_new_state, str(self.prefix),
                                )
                        else:
                            _fr10_node_state_rows = []
                            for _fr10_node_i in range(_fr10_tree_n):
                                _fr10_node_path = _fr10_path_node_tensors[_fr10_node_i]
                                _fr10_node_x = _fr10_x.index_select(0, _fr10_node_path)
                                _fr10_node_state_source = torch.cat(
                                    (_fr10_prior_window.transpose(0, 1), _fr10_node_x),
                                    dim=0,
                                )
                                _fr10_node_state_source = torch.cat(
                                    (
                                        _fr10_node_state_source,
                                        _fr10_x.new_zeros(
                                            (
                                                int(conv_state.size(2)),
                                                int(_fr10_x.size(1)),
                                            )
                                        ),
                                    ),
                                    dim=0,
                                )
                                _fr10_node_store_idx = (
                                    _fr10_node_path.numel()
                                    + torch.arange(
                                        conv_state.size(2),
                                        dtype=torch.long,
                                        device=mixed_qkv_spec.device,
                                    )
                                )
                                _fr10_node_state_rows.append(
                                    _fr10_node_state_source.index_select(
                                        0, _fr10_node_store_idx
                                    ).transpose(0, 1)
                                )
                            _fr10_new_state = torch.stack(
                                _fr10_node_state_rows, dim=0
                            ).to(dtype=conv_state.dtype)
                        _fr10_conv_handoff_active = (
                            os.environ.get("FR10_TREE_GDN_COMMIT_HANDOFF_LOG")
                            or os.environ.get("FR10_TREE_GDN_SRC_NATIVE_PAYLOAD")
                        )
                        if _fr10_conv_handoff_active:
                            _fr10_conv_handoff_prefix = os.environ.get(
                                "FR10_TREE_GDN_COMMIT_HANDOFF_LAYER_PREFIX", ""
                            )
                            if (
                                _fr10_conv_handoff_prefix
                                and _fr10_conv_handoff_prefix != str(self.prefix)
                            ):
                                _fr10_conv_handoff_active = False
                        if _fr10_conv_handoff_active:
                            try:
                                globals().setdefault(
                                    "_FR10_COMMIT_HANDOFF_CURR_CONV_BY_B", {}
                                )[int(_fr10_b)] = {
                                    "prior": _fr10_prior_conv_state_bank[
                                        _fr10_b
                                    ].detach().clone(),
                                    "rows": _fr10_new_state.detach().clone(),
                                }
                            except Exception:
                                pass
                        if _fr10_new_state is None:
                            # FR13_CONV_WB_FUSED: one fused gather->page-write
                            # launch replaces gather + transpose-contiguous +
                            # index_copy_ (byte-copy contract; int32 dst rows
                            # consumed directly, no .to(long) cast kernel).
                            if _FR13_CONV_WB_BATCHED:
                                # B2c: source staged above; ONE batched launch
                                # fires after the loop (same bytes, disjoint
                                # dsts; no same-step reader before it).
                                pass
                            else:
                                launch_conv_state_writeback(
                                    source_z=_fr10_source,
                                    state_src=_fr13_tcf_state_src,
                                    dst_rows=spec_state_indices_tensor[
                                        _fr10_b, :_fr10_tree_n
                                    ],
                                    conv_state=conv_state,
                                    tree_n=_fr10_tree_n,
                                    state_len=int(conv_state.size(2)),
                                )
                        else:
                            conv_state.index_copy_(
                                0,
                                spec_state_indices_tensor[
                                    _fr10_b, :_fr10_tree_n
                                ].to(torch.long),
                                _fr10_new_state,
                            )
                        if _fr10_log_conv_diag:
                            if _FR13_EAGER_PACK:
                                # FR13_EAGER_PACK 2h gated-move: recompute from
                                # the SAME unmutated inputs as the legacy site
                                # (identical value; FR10_METRICS=1 only).
                                _fr10_path0_x = _fr10_x.index_select(
                                    0, _fr10_path0_node_tensor
                                )
                            _fr10_path0_source = torch.cat(
                                (_fr10_prior_window.transpose(0, 1), _fr10_path0_x),
                                dim=0,
                            )
                            _fr10_path0_window = _fr10_path0_source.index_select(
                                0, _fr10_path0_source_flat
                            ).view(
                                _fr10_path0_node_tensor.numel(),
                                _fr10_width,
                                _fr10_path0_x.size(1),
                            )
                            if self.conv1d.bias is None:
                                _fr10_path0_acc = torch.zeros_like(
                                    _fr10_path0_x, dtype=torch.float32
                                )
                            else:
                                _fr10_path0_acc = self.conv1d.bias.to(
                                    torch.float32
                                ).unsqueeze(0).expand_as(_fr10_path0_x.float()).clone()
                            for _fr10_col in range(_fr10_width):
                                _fr10_path0_acc = (
                                    _fr10_path0_acc
                                    + _fr11_conv_tap_product(
                                        _fr10_path0_window[:, _fr10_col, :],
                                        conv_weights[:, _fr10_col],
                                    )
                                )
                            if self.activation in (True, "silu", "swish"):
                                _fr10_path0_ref = triton_ex2_silu_bf16(
                                    _fr10_path0_acc, out_dtype=mixed_qkv_spec.dtype
                                )
                            else:
                                _fr10_path0_ref = _fr10_path0_acc.to(
                                    dtype=mixed_qkv_spec.dtype
                                )
                            _fr10_tree_path0 = _fr10_out.index_select(
                                0, _fr10_path0_node_tensor
                            )
                            _fr10_flat_window = _fr10_source.index_select(
                                0, _fr10_flat_source_flat
                            ).view(_fr10_tree_n, _fr10_width, _fr10_x.size(1))
                            if self.conv1d.bias is None:
                                _fr10_flat_acc = torch.zeros_like(
                                    _fr10_x, dtype=torch.float32
                                )
                            else:
                                _fr10_flat_acc = self.conv1d.bias.to(
                                    torch.float32
                                ).unsqueeze(0).expand_as(_fr10_x.float()).clone()
                            for _fr10_col in range(_fr10_width):
                                _fr10_flat_acc = (
                                    _fr10_flat_acc
                                    + _fr11_conv_tap_product(
                                        _fr10_flat_window[:, _fr10_col, :],
                                        conv_weights[:, _fr10_col],
                                    )
                                )
                            if self.activation in (True, "silu", "swish"):
                                _fr10_native_flat_path0 = triton_ex2_silu_bf16(
                                    _fr10_flat_acc, out_dtype=mixed_qkv_spec.dtype
                                ).index_select(0, _fr10_path0_node_tensor)
                            else:
                                _fr10_native_flat_path0 = _fr10_flat_acc.to(
                                    dtype=mixed_qkv_spec.dtype
                                ).index_select(0, _fr10_path0_node_tensor)
                            _fr10_serial_out = torch.empty_like(_fr10_out)
                            _fr10_serial_state_rows = []
                            _fr10_replay_pert_x = _fr10_x.clone()
                            for _fr10_branch_node in _fr10_branch_nodes_py:
                                _fr10_replay_pert_x[_fr10_branch_node].add_(10000.0)
                            _fr10_pert_source = torch.cat(
                                (_fr10_prior_window.transpose(0, 1), _fr10_replay_pert_x),
                                dim=0,
                            )
                            _fr10_pert_window = _fr10_pert_source.index_select(
                                0, _fr10_source_flat
                            ).view(_fr10_tree_n, _fr10_width, _fr10_x.size(1))
                            _fr10_flat_pert_window = _fr10_pert_source.index_select(
                                0, _fr10_flat_source_flat
                            ).view(_fr10_tree_n, _fr10_width, _fr10_x.size(1))
                            if self.conv1d.bias is None:
                                _fr10_pert_acc = torch.zeros_like(
                                    _fr10_x, dtype=torch.float32
                                )
                                _fr10_flat_pert_acc = torch.zeros_like(
                                    _fr10_x, dtype=torch.float32
                                )
                            else:
                                _fr10_pert_acc = self.conv1d.bias.to(
                                    torch.float32
                                ).unsqueeze(0).expand_as(_fr10_x.float()).clone()
                                _fr10_flat_pert_acc = self.conv1d.bias.to(
                                    torch.float32
                                ).unsqueeze(0).expand_as(_fr10_x.float()).clone()
                            for _fr10_col in range(_fr10_width):
                                _fr10_pert_acc = (
                                    _fr10_pert_acc
                                    + _fr11_conv_tap_product(
                                        _fr10_pert_window[:, _fr10_col, :],
                                        conv_weights[:, _fr10_col],
                                    )
                                )
                                _fr10_flat_pert_acc = (
                                    _fr10_flat_pert_acc
                                    + _fr11_conv_tap_product(
                                        _fr10_flat_pert_window[:, _fr10_col, :],
                                        conv_weights[:, _fr10_col],
                                    )
                                )
                            if self.activation in (True, "silu", "swish"):
                                _fr10_pert_path0 = triton_ex2_silu_bf16(
                                    _fr10_pert_acc, out_dtype=mixed_qkv_spec.dtype
                                ).index_select(0, _fr10_path0_node_tensor)
                                _fr10_flat_pert_path0 = triton_ex2_silu_bf16(
                                    _fr10_flat_pert_acc,
                                    out_dtype=mixed_qkv_spec.dtype,
                                ).index_select(0, _fr10_path0_node_tensor)
                            else:
                                _fr10_pert_path0 = _fr10_pert_acc.to(
                                    dtype=mixed_qkv_spec.dtype
                                ).index_select(0, _fr10_path0_node_tensor)
                                _fr10_flat_pert_path0 = _fr10_flat_pert_acc.to(
                                    dtype=mixed_qkv_spec.dtype
                                ).index_select(0, _fr10_path0_node_tensor)
                            for _fr10_node_i in range(_fr10_tree_n):
                                _fr10_node_path = _fr10_path_node_tensors[_fr10_node_i]
                                _fr10_node_x = _fr10_x.index_select(0, _fr10_node_path)
                                _fr10_serial_source = torch.cat(
                                    (_fr10_prior_window.transpose(0, 1), _fr10_node_x),
                                    dim=0,
                                )
                                _fr10_serial_window_idx = (
                                    _fr10_node_path.numel()
                                    - 1
                                    + torch.arange(
                                        _fr10_width,
                                        dtype=torch.long,
                                        device=mixed_qkv_spec.device,
                                    )
                                )
                                _fr10_serial_window = _fr10_serial_source.index_select(
                                    0, _fr10_serial_window_idx
                                )
                                if self.conv1d.bias is None:
                                    _fr10_serial_acc = torch.zeros_like(
                                        _fr10_x[_fr10_node_i], dtype=torch.float32
                                    )
                                else:
                                    _fr10_serial_acc = self.conv1d.bias.to(
                                        torch.float32
                                    ).clone()
                                for _fr10_col in range(_fr10_width):
                                    _fr10_serial_acc = (
                                        _fr10_serial_acc
                                        + _fr11_conv_tap_product(
                                            _fr10_serial_window[_fr10_col],
                                            conv_weights[:, _fr10_col],
                                        )
                                    )
                                if self.activation in (True, "silu", "swish"):
                                    _fr10_serial_out[
                                        _fr10_node_i
                                    ] = triton_ex2_silu_bf16(
                                        _fr10_serial_acc, out_dtype=mixed_qkv_spec.dtype
                                    )
                                else:
                                    _fr10_serial_out[_fr10_node_i] = _fr10_serial_acc.to(
                                        dtype=mixed_qkv_spec.dtype
                                    )
                                _fr10_serial_state_source = torch.cat(
                                    (_fr10_prior_window.transpose(0, 1), _fr10_node_x),
                                    dim=0,
                                )
                                _fr10_serial_state_source = torch.cat(
                                    (
                                        _fr10_serial_state_source,
                                        _fr10_x.new_zeros(
                                            (
                                                int(conv_state.size(2)),
                                                int(_fr10_x.size(1)),
                                            )
                                        ),
                                    ),
                                    dim=0,
                                )
                                _fr10_serial_state_idx = (
                                    _fr10_node_path.numel()
                                    + torch.arange(
                                        conv_state.size(2),
                                        dtype=torch.long,
                                        device=mixed_qkv_spec.device,
                                    )
                                )
                                _fr10_serial_state_rows.append(
                                    _fr10_serial_state_source.index_select(
                                        0, _fr10_serial_state_idx
                                    ).transpose(0, 1)
                                )
                            _fr10_serial_state = torch.stack(
                                _fr10_serial_state_rows, dim=0
                            ).to(dtype=conv_state.dtype)
                            _fr10_tree_delta = (
                                _fr10_tree_path0.float() - _fr10_path0_ref.float()
                            ).abs()
                            _fr10_native_delta = (
                                _fr10_native_flat_path0.float()
                                - _fr10_path0_ref.float()
                            ).abs()
                            _fr10_tree_max = _fr10_tree_delta.max()
                            _fr10_native_max = _fr10_native_delta.max()
                            _fr10_serial_out_max = (
                                _fr10_out.float() - _fr10_serial_out.float()
                            ).abs().max()
                            _fr10_serial_state_max = (
                                _fr10_new_state.float() - _fr10_serial_state.float()
                            ).abs().max()
                            _fr10_sibling_path0_max = (
                                _fr10_pert_path0.float() - _fr10_tree_path0.float()
                            ).abs().max()
                            _fr10_flat_sibling_path0_max = (
                                _fr10_flat_pert_path0.float()
                                - _fr10_native_flat_path0.float()
                            ).abs().max()
                            _fr10_conv_diag[0].copy_(
                                torch.maximum(_fr10_conv_diag[0], _fr10_tree_max)
                            )
                            _fr10_conv_diag[1].copy_(
                                torch.maximum(_fr10_conv_diag[1], _fr10_native_max)
                            )
                            _fr10_conv_diag[2].add_(
                                (_fr10_tree_max != 0).to(dtype=torch.float32)
                            )
                            _fr10_conv_diag[3].add_(
                                (_fr10_native_max != 0).to(dtype=torch.float32)
                            )
                            _fr10_conv_diag[4].add_(1.0)
                            _fr10_conv_diag[5].fill_(float(_fr10_tree_n))
                            _fr10_conv_diag[6].copy_(
                                torch.maximum(_fr10_conv_diag[6], _fr10_serial_out_max)
                            )
                            _fr10_conv_diag[7].copy_(
                                torch.maximum(_fr10_conv_diag[7], _fr10_serial_state_max)
                            )
                            _fr10_conv_diag[8].copy_(
                                torch.maximum(_fr10_conv_diag[8], _fr10_sibling_path0_max)
                            )
                            _fr10_conv_diag[9].copy_(
                                torch.maximum(
                                    _fr10_conv_diag[9], _fr10_flat_sibling_path0_max
                                )
                            )
                            _fr10_conv_diag[10].add_(
                                (_fr10_sibling_path0_max != 0).to(dtype=torch.float32)
                            )
                            _fr10_conv_diag[11].add_(
                                (_fr10_flat_sibling_path0_max != 0).to(dtype=torch.float32)
                            )
                    if _fr13_sfwd_candidate_out is not None:
                        if (
                            _fr13_wbb_stage is None
                            or _fr13_sfwd_candidate_stage is None
                            or _fr13_wbb_srows != _fr13_sfwd_srows
                        ):
                            raise RuntimeError(
                                "FR13_FIXED32_SFWD_STATE_FUSION incumbent "
                                "commit-source stage was not produced"
                            )
                        if _fr13_sfwd_candidate_kind == "prior_reuse":
                            _fr13_sfwd_record = (
                                fixed32_sfwd_prior_reuse_byte_gate(
                                    task_marker=_fr13_sfwd_task_marker,
                                    layer_prefix=str(self.prefix),
                                    layer_key=int(conv_weights.data_ptr()),
                                    batch_size=int(
                                        attn_metadata.num_spec_decodes
                                    ),
                                    reference_out=_fr10_tree_conv_out,
                                    candidate_out=_fr13_sfwd_candidate_out,
                                    reference_source_stage=_fr13_wbb_stage[
                                        : int(attn_metadata.num_spec_decodes)
                                        * _fr13_wbb_srows
                                    ],
                                    candidate_source_stage=(
                                        _fr13_sfwd_candidate_stage
                                    ),
                                    source_manifest_path=_FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_PATH,
                                    expected_source_manifest_sha256=_FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_SHA256,
                                    expected_source_commit=_FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_COMMIT,
                                )
                            )
                        else:
                            _fr13_sfwd_record = (
                                fixed32_sfwd_state_fusion_byte_gate(
                                    task_markers=_fr13_sfwd_task_markers,
                                    layer_key=int(conv_weights.data_ptr()),
                                    batch_size=int(
                                        attn_metadata.num_spec_decodes
                                    ),
                                    reference_out=_fr10_tree_conv_out,
                                    candidate_out=_fr13_sfwd_candidate_out,
                                    reference_source_stage=_fr13_wbb_stage[
                                        : int(attn_metadata.num_spec_decodes)
                                        * _fr13_wbb_srows
                                    ],
                                    candidate_source_stage=(
                                        _fr13_sfwd_candidate_stage
                                    ),
                                )
                            )
                        if not bool(_fr13_sfwd_record["zero_diff"]):
                            logger.warning_once(
                                "FR13 fixed32 SFWD state-fusion candidate "
                                "mismatched; incumbent bytes remain served"
                            )
                    # FR13_CONV_WB_BATCHED (B2c): ONE batched writeback for all
                    # requests (replaces B per-request launches; same bytes,
                    # disjoint dsts). Fires only on the fused arm; the pool
                    # route is now the only route (the nodebank arm, which
                    # added a batched col0 pool write here, was deleted
                    # 2026-07-25). No same-step
                    # reader of these dsts runs before this point (taps/scan
                    # read the staged source; committer + pregather run
                    # post-forward; the linear remap consumed PREV-step
                    # deposits at forward start).
                    if (
                        _FR13_CONV_WB_BATCHED
                        and not _FR13_FIXED32_MODE
                        and _fr13_wbb_stage is not None
                        and _fr13_conv_wb_fused_now
                        and int(attn_metadata.num_spec_decodes) > 0
                    ):
                        _fr13_wbb_b = int(attn_metadata.num_spec_decodes)
                        launch_conv_state_writeback_batched(
                            source_z=_fr13_wbb_stage,
                            state_src=_fr13_tcf_state_src,
                            dst_rows=spec_state_indices_tensor[
                                :_fr13_wbb_b, :_fr10_tree_n
                            ].reshape(-1),
                            conv_state=conv_state,
                            tree_n=_fr10_tree_n,
                            state_len=int(conv_state.size(2)),
                            batch=_fr13_wbb_b,
                            src_rows_per_b=_fr13_wbb_srows,
                        )
                    if not _fr13_conv_postprep_active:
                        mixed_qkv_spec = _fr10_tree_conv_out
                except Exception as _fr10_tree_conv_exc:
                    if (
                        _fr13_conv_postprep_candidate
                        or (
                            _fr10_tree_conv_expected
                            and os.environ.get(
                                "FR10_ALLOW_LINEAR_FALLBACK", "0"
                            )
                            != "1"
                        )
                    ):
                        raise RuntimeError(
                            "FR10 tree causal-conv disengaged: "
                            + type(_fr10_tree_conv_exc).__name__
                            + ":"
                            + str(_fr10_tree_conv_exc)
                        ) from _fr10_tree_conv_exc
                    if os.environ.get("FR10_METRICS", "0") == "1":
                        logger.warning_once(
                            "FR10 tree causal-conv fallback to native flat order: %s",
                            _fr10_tree_conv_exc,
                        )
                    mixed_qkv_spec = causal_conv1d_update(
                        mixed_qkv_spec,
                        conv_state,
                        conv_weights,
                        self.conv1d.bias,
                        self.activation,
                        conv_state_indices=spec_state_indices_tensor[:, 0][
                            : attn_metadata.num_spec_decodes
                        ],
                        num_accepted_tokens=num_accepted_tokens,
                        query_start_loc=spec_query_start_loc,
                        max_query_len=spec_state_indices_tensor.size(-1),
                        validate_data=False,
                    )
            else:
                if (
                    _fr10_tree_conv_expected
                    and os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1"
                ):
                    raise RuntimeError(
                        "FR10 tree causal-conv disengaged: eligible_tree_spec_row_flat_fallback"
                    )
                mixed_qkv_spec = causal_conv1d_update(
                    mixed_qkv_spec,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                        : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                    ],
                    num_accepted_tokens=num_accepted_tokens,
                    query_start_loc=spec_query_start_loc,
                    max_query_len=spec_state_indices_tensor.size(-1),
                    validate_data=False,
                )
            try:
                _fr12_payload = None
                if _fr12_subkernel_capture_enabled:
                    _fr12_conv_extra = {
                        "num_spec_decodes": int(attn_metadata.num_spec_decodes),
                        "num_actual_tokens": int(num_actual_tokens),
                        "tree_conv_active": bool(use_fr10_tree_conv),
                        "tree_conv_expected": bool(_fr10_tree_conv_expected),
                    }
                    if getattr(attn_metadata, "fr10_tree_parent", None) is not None:
                        _fr12_conv_extra["tree_parent"] = [
                            int(_x)
                            for _x in attn_metadata.fr10_tree_parent.detach().cpu().tolist()
                        ]
                    if spec_token_indx is not None:
                        _fr12_conv_extra["spec_token_indx"] = [
                            int(_x) for _x in spec_token_indx.detach().cpu().tolist()
                        ]
                    _fr12_subkernel_capture_tensor(
                        self,
                        "conv1d_out",
                        mixed_qkv_spec,
                        create=True,
                        extra=_fr12_conv_extra,
                    )
                    _fr12_payload = _fr12_subkernel_capture_get(self, create=False)
            except Exception as _fr12_conv_cap_exc:
                logger.warning("FR12 conv capture failed: %s", _fr12_conv_cap_exc)

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            _fr13_prefill_pre_conv_capture = mixed_qkv_non_spec.detach().clone()
            # FR13_APC_CONV_RESTORE_CAPTURE (default 0 -> inert / byte-identical).
            # On a cache-HIT prefill row, conv_state[non_spec_state_indices_tensor]
            # currently holds the RESTORED conv window seed (the K-1 prior-window
            # snapshot read back by physical block_id) -- this is the conv twin of
            # the SSM initial_state restored seed. causal_conv1d_fn below mutates
            # conv_state IN PLACE at those same cache_indices, so the seed MUST be
            # snapshotted HERE, before the kernel call, or it is lost. Default OFF
            # -> no read, no clone -> off-path byte-identical.
            _fr13_prefill_conv_restore_capture = None
            if (
                os.environ.get("FR13_APC_CONV_RESTORE_CAPTURE", "0") == "1"
                and non_spec_state_indices_tensor is not None
            ):
                _fr13_prefill_conv_restore_capture = (
                    conv_state.index_select(
                        0,
                        non_spec_state_indices_tensor.to(torch.long),
                    )
                    .detach()
                    .cpu()
                    .clone()
                )
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            _fr13_prefill_conv_out_capture = mixed_qkv_non_spec.detach().clone()
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        if (
            _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION
            and spec_sequence_masks is not None
        ):
            if any(
                tensor is None
                for tensor in (
                    _fr13_conv_postprep_query,
                    _fr13_conv_postprep_key,
                    _fr13_conv_postprep_value_spec,
                )
            ):
                raise RuntimeError(
                    "FR13 SFWD conv/post-prep recurrence outputs were not produced"
                )
            query_spec = _fr13_conv_postprep_query
            key_spec = _fr13_conv_postprep_key
            value_spec = _fr13_conv_postprep_value_spec
        else:
            query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(
                mixed_qkv_spec
            )
        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = fused_post_conv_prep(
                conv_output=mixed_qkv_non_spec,
                a=a_non_spec,
                b=b_non_spec,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_k_heads=self.num_k_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )
            query_non_spec = query_non_spec.unsqueeze(0)
            key_non_spec = key_non_spec.unsqueeze(0)
            value_non_spec = value_non_spec.unsqueeze(0)
            g_non_spec = g_non_spec.unsqueeze(0)
            beta_non_spec = beta_non_spec.unsqueeze(0)
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            g_non_spec = None
            beta_non_spec = None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            try:
                from vllm.v1.sample import rejection_sampler as _fr10_rs_mode
                _fr10_active_decode_mode = getattr(
                    _fr10_rs_mode, "_FR10_DECODE_MODE", _FR10_DECODE_MODE
                )
            except Exception:
                _fr10_active_decode_mode = _FR10_DECODE_MODE
            use_fr10_tree = (
                os.environ.get("FR10_ENABLE_TREE_GDN") == "1"
                and _fr10_active_decode_mode == "tree_mtp"
                and getattr(attn_metadata, "fr10_tree_parent", None) is not None
                and attn_metadata.num_spec_decodes > 0
            )
            _fr10_tree_scan_expected = (
                _fr10_active_decode_mode == "tree_mtp"
                and getattr(attn_metadata, "fr10_tree_parent", None) is not None
                and attn_metadata.num_spec_decodes > 0
            )
            _fr10_scan_branch_diag = getattr(
                attn_metadata, "fr10_tree_conv_diag", None
            )
            if (
                os.environ.get("FR10_METRICS", "0") == "1"
                and _fr10_scan_branch_diag is not None
            ):
                _fr10_scan_branch_diag[17].add_(float(attn_metadata.num_spec_decodes))
                if use_fr10_tree:
                    _fr10_scan_branch_diag[18].add_(float(attn_metadata.num_spec_decodes))
                else:
                    _fr10_scan_branch_diag[19].add_(float(attn_metadata.num_spec_decodes))
                _fr10_scan_branch_diag[22].add_(float(attn_metadata.num_spec_decodes))
            # STAGE (ii) — CALL-SITE ENGAGEMENT (class-9, failure-#4 hole).  The
            # conv-site stash runs on enabled() ALONE, but the A/B CALL below is
            # gated on use_fr10_tree.  If the flag is ON yet use_fr10_tree is
            # False on a spec-verify forward, the call is SILENTLY skipped (stash
            # accumulates, zero records, zero log).  Emit ONE loud ERROR naming
            # the failing precondition so the chase is never vacuous.  Observe-
            # only; no effect on the live forward; no-op when the flag is OFF.
            if _fr13_gdn_subop_mab_enabled() and not use_fr10_tree:
                _fr13_subop_worker_env_gate()
                _fr13_subop_stage(
                    "callsite-skip",
                    (
                        "flag ON but use_fr10_tree False on a spec-verify forward "
                        "(stash accumulates, A/B call skipped). reasons:"
                        " FR10_ENABLE_TREE_GDN="
                        + repr(os.environ.get("FR10_ENABLE_TREE_GDN"))
                        + " decode_mode=" + repr(_fr10_active_decode_mode)
                        + " tree_parent_set="
                        + str(getattr(attn_metadata, "fr10_tree_parent", None) is not None)
                        + " num_spec_decodes=" + str(attn_metadata.num_spec_decodes)
                    ),
                )
                # release the orphaned stash so it cannot leak into a later event
                self._fr13_gdn_subop_mab_pre_conv = None
                self._fr13_gdn_subop_mab_conv_state = None
            elif _fr13_gdn_subop_mab_enabled() and use_fr10_tree:
                _fr13_subop_worker_env_gate()
            if use_fr10_tree:
                assert spec_query_start_loc is not None
                assert spec_state_indices_tensor is not None
                assert attn_metadata.fr10_tree_parent is not None
                assert attn_metadata.fr10_tree_strict_mask is not None
                assert attn_metadata.fr10_tree_visible_mask is not None
                _fr10_accepted_lens_tensor = globals().get(
                    "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR"
                )
                if _fr10_accepted_lens_tensor is None:
                    raise RuntimeError("missing_accepted_lens_device_tensor")
                if os.environ.get("FR10_METRICS", "0") == "1":
                    logger.warning_once(
                        "FR10 tree GDN verifier branch active for layer %s", self.prefix
                    )
                if _fr13_gdn_subop_mab_enabled():
                    # FR13_GDN_SUBOP_MAB (default OFF, observe-only): re-run the
                    # L0-GDN sub-op sequence at M=10/M=5/M=1 on the SAME captured
                    # pre_conv + h0 + conv prior-window to localize WHERE the
                    # deep-spine row first becomes M-dependent.  No mutation of
                    # the live forward.
                    _fr13_gdn_subop_mab(
                        self,
                        getattr(self, "_fr13_gdn_subop_mab_pre_conv", None),
                        mixed_qkv_spec,
                        a,
                        b,
                        query_spec,
                        key_spec,
                        value_spec,
                        getattr(self, "_fr13_gdn_subop_mab_conv_state", None),
                        conv_weights,
                        ssm_state,
                        spec_state_indices_tensor,
                        num_accepted_tokens,
                        spec_query_start_loc,
                        attn_metadata,
                        num_actual_tokens,
                    )
                    self._fr13_gdn_subop_mab_pre_conv = None
                    self._fr13_gdn_subop_mab_conv_state = None
                if _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION:
                    if any(
                        tensor is None
                        for tensor in (
                            _fr13_conv_postprep_value_tree,
                            _fr13_conv_postprep_g,
                            _fr13_conv_postprep_beta,
                        )
                    ):
                        raise RuntimeError(
                            "FR13 SFWD conv/post-prep scan inputs were not produced"
                        )
                    value_tree = _fr13_conv_postprep_value_tree
                    g_tree = _fr13_conv_postprep_g
                    beta_tree = _fr13_conv_postprep_beta
                else:
                    _, _, value_tree, g_tree, beta_tree = fused_post_conv_prep(
                        conv_output=mixed_qkv_spec,
                        a=a,
                        b=b,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        num_k_heads=self.num_k_heads // self.tp_size,
                        head_k_dim=self.head_k_dim,
                        head_v_dim=self.head_v_dim,
                        apply_l2norm=True,
                        output_g_exp=False,
                    )
                    if (
                        _FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB
                        and _fr13_conv_postprep_candidate
                    ):
                        _fr13_conv_postprep_record = (
                            fixed32_sfwd_conv_postprep_byte_gate(
                                fixed32_mode=_FR13_FIXED32_MODE,
                                task_marker=_fr13_conv_postprep_task_marker,
                                layer_prefix=str(self.prefix),
                                layer_key=int(conv_weights.data_ptr()),
                                batch_size=int(attn_metadata.num_spec_decodes),
                                reference_query=query_spec,
                                candidate_query=_fr13_conv_postprep_query,
                                reference_key=key_spec,
                                candidate_key=_fr13_conv_postprep_key,
                                reference_value_spec=value_spec,
                                candidate_value_spec=_fr13_conv_postprep_value_spec,
                                reference_value_tree=value_tree,
                                candidate_value_tree=_fr13_conv_postprep_value_tree,
                                reference_g=g_tree,
                                candidate_g=_fr13_conv_postprep_g,
                                reference_beta=beta_tree,
                                candidate_beta=_fr13_conv_postprep_beta,
                                reference_source_stage=_fr13_wbb_stage[
                                    : int(attn_metadata.num_spec_decodes)
                                    * _fr13_wbb_srows
                                ],
                                candidate_source_stage=(
                                    _fr13_conv_postprep_source_stage
                                ),
                                source_manifest_path=os.environ.get(
                                    "FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_PATH",
                                    "",
                                ),
                                expected_source_manifest_sha256=os.environ.get(
                                    "FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256",
                                    "",
                                ),
                                expected_source_commit=os.environ.get(
                                    "FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT",
                                    "",
                                ),
                                embedded_gate_cta=(
                                    _FR13_FIXED32_SFWD_EMBED_GATE_CTA
                                ),
                                direct_nodegroup8=(
                                    _FR13_FIXED32_SFWD_NODEGROUP8_DIRECT
                                ),
                            )
                        )
                        if not bool(_fr13_conv_postprep_record["zero_diff"]):
                            logger.warning_once(
                                "FR13 SFWD conv/post-prep candidate mismatched; "
                                "incumbent tensors remain served"
                            )
                tree_n = int(attn_metadata.fr10_tree_parent.numel())
                tree_n_pad = int(attn_metadata.fr10_tree_visible_mask.size(0))
                _fr12_native_spine_oracle_enabled = (
                    os.environ.get("FR12_NATIVE_SPINE_ORACLE", "0") == "1"
                )
                _fr12_native_spine_scan_enabled = (
                    _fr12_native_spine_oracle_enabled
                    and os.environ.get("FR12_TREE_SCAN_NATIVE_SPINE", "0") == "1"
                )
                _fr12_scan_path0_node_tensor = None
                if _fr12_native_spine_scan_enabled:
                    _fr12_scan_path0_node_tensor = getattr(
                        attn_metadata, "fr10_tree_path0_nodes", None
                    )
                    if _fr12_scan_path0_node_tensor is None:
                        raise RuntimeError(
                            "FR12 native-spine scan requested without path0 nodes"
                        )
                core_attn_out_spec = torch.empty(
                    (1, query_spec.size(1), value_tree.size(1), value_tree.size(2)),
                    dtype=query_spec.dtype,
                    device=query_spec.device,
                )
                _fr13_replay_route_on = (
                    # FR13_REPLAY_ROUTE: baked ON by default (serving unchanged);
                    # env-gated so a G1 kernel-capture can set FR13_REPLAY_ROUTE=0
                    # to keep the per-node tree_state scratch alive for
                    # FR10_TREE_GDN_CAPTURE_PAYLOAD. tree-path only (use_fr10_tree),
                    # so naive_mtp/native is unaffected.
                    os.environ.get("FR13_REPLAY_ROUTE", "1") == "1"
                )
                if _fr13_replay_route_on:
                    # FR13_REPLAY_ROUTE: the per-node scratch (tree_state) no
                    # longer exists, so every diagnostic that embeds or
                    # splices it must refuse loudly instead of silently
                    # reading garbage (fail-loud policy).
                    if (
                        os.environ.get("FR10_TREE_GDN_CAPTURE_PAYLOAD")
                        or os.environ.get("FR10_TREE_GDN_COMMIT_HANDOFF_LOG")
                        or os.environ.get("FR10_TREE_GDN_SRC_NATIVE_PAYLOAD")
                        or os.environ.get("FR12_TREE_SCAN_NATIVE_SPINE", "0") == "1"
                    ):
                        raise RuntimeError(
                            "FR13_REPLAY_ROUTE is incompatible with tree_state "
                            "diagnostics (FR10_TREE_GDN_CAPTURE_PAYLOAD/"
                            "FR10_TREE_GDN_COMMIT_HANDOFF_LOG/"
                            "FR10_TREE_GDN_SRC_NATIVE_PAYLOAD/"
                            "FR12_TREE_SCAN_NATIVE_SPINE); capture with the "
                            "flag OFF"
                        )
                    # FR13_REPLAY_ROUTE: skip the 201.3MB/layer per-step
                    # scratch alloc (the capture-blocking allocation); the
                    # scan runs with STORE_NODE_STATES=False and the durable
                    # accepted states are produced by the committer replay.
                    tree_state_all = None
                for fr10_b in range(attn_metadata.num_spec_decodes):
                    # Full CUDA graph capture cannot tolerate GPU->CPU syncs.
                    # In pure tree-spec decode vLLM lays each spec decode out as
                    # one fixed tree block, so offsets are static from tree_n.
                    start = fr10_b * tree_n
                    end = start + tree_n
                    tree_state = None  # STATELESS-TREE: replay-only, no per-node scratch
                    if os.environ.get("FR12_SUBKERNEL_CAPTURE"):
                        try:
                            _fr12_capture_h0_col = torch.clamp(
                                _fr10_accepted_lens_tensor[
                                    : attn_metadata.num_spec_decodes
                                ].to(torch.long)
                                - 1,
                                min=0,
                                max=int(spec_state_indices_tensor.size(-1)) - 1,
                            ).view(-1, 1)
                            _fr12_capture_h0_rows = spec_state_indices_tensor[
                                : attn_metadata.num_spec_decodes
                            ].gather(1, _fr12_capture_h0_col).reshape(-1).to(torch.long)
                            _fr12_subkernel_capture_tensor(
                                self,
                                "h0_state_in",
                                torch.index_select(ssm_state, 0, _fr12_capture_h0_rows),
                                create=False,
                                extra={
                                    "num_actual_tokens": int(num_actual_tokens),
                                    "h0_rows": [
                                        int(_x)
                                        for _x in _fr12_capture_h0_rows.detach()
                                        .cpu()
                                        .tolist()
                                    ],
                                    "h0_cols": [
                                        int(_x)
                                        for _x in _fr12_capture_h0_col.reshape(-1)
                                        .detach()
                                        .cpu()
                                        .tolist()
                                    ],
                                },
                            )
                        except Exception as _fr12_h0_cap_exc:
                            logger.warning("FR12 h0 capture failed: %s", _fr12_h0_cap_exc)
                    _fr10_capture_scan_payload = (
                        os.environ.get("FR10_TREE_GDN_CAPTURE_PAYLOAD")
                        and not globals().get("_FR10_TREE_GDN_CAPTURE_DONE", False)
                        and fr10_b == 0
                    )
                    if _fr10_capture_scan_payload:
                        _fr10_capture_prefix = os.environ.get(
                            "FR10_TREE_GDN_CAPTURE_PAYLOAD_LAYER_PREFIX", ""
                        )
                        if (
                            _fr10_capture_prefix
                            and _fr10_capture_prefix != str(self.prefix)
                        ):
                            _fr10_capture_scan_payload = False
                        _fr10_capture_counts = os.environ.get(
                            "FR10_TREE_GDN_CAPTURE_PAYLOAD_NUM_TOKENS", ""
                        )
                        if _fr10_capture_scan_payload and _fr10_capture_counts:
                            _fr10_wanted_counts = {
                                int(_x.strip())
                                for _x in _fr10_capture_counts.split(",")
                                if _x.strip()
                            }
                            if int(num_actual_tokens) not in _fr10_wanted_counts:
                                _fr10_capture_scan_payload = False
                    _fr10_commit_handoff_active = os.environ.get(
                        "FR10_TREE_GDN_COMMIT_HANDOFF_LOG"
                    ) or os.environ.get("FR10_TREE_GDN_SRC_NATIVE_PAYLOAD")
                    if _fr10_commit_handoff_active:
                        _fr10_commit_handoff_prefix = os.environ.get(
                            "FR10_TREE_GDN_COMMIT_HANDOFF_LAYER_PREFIX", ""
                        )
                        if (
                            _fr10_commit_handoff_prefix
                            and _fr10_commit_handoff_prefix != str(self.prefix)
                        ):
                            _fr10_commit_handoff_active = False
                    _fr10_read_col = 0
                    if _fr10_capture_scan_payload or _fr10_commit_handoff_active:
                        try:
                            _fr10_read_col = max(
                                0,
                                min(
                                    int(
                                        _fr10_accepted_lens_tensor[
                                            fr10_b
                                        ].detach().cpu().item()
                                    )
                                    - 1,
                                    int(spec_state_indices_tensor.size(-1)) - 1,
                                )
                            )
                        except Exception:
                            _fr10_read_col = 0
                    _fr10_capture_state_index = None
                    _fr10_capture_h0 = None
                    if _fr10_capture_scan_payload:
                        _fr10_capture_state_index = int(
                            spec_state_indices_tensor[fr10_b, _fr10_read_col]
                            .detach()
                            .cpu()
                            .item()
                        )
                        _fr10_capture_h0 = (
                            ssm_state[_fr10_capture_state_index].detach().cpu().clone()
                        )
                    if _fr10_commit_handoff_active:
                        try:
                            _fr10_commit_state_index = int(
                                spec_state_indices_tensor[fr10_b, _fr10_read_col]
                                .detach()
                                .cpu()
                                .item()
                            )
                            _fr10_commit_h0 = (
                                ssm_state[_fr10_commit_state_index].detach().clone()
                            )
                        except Exception:
                            _fr10_commit_handoff_active = False
                    try:
                        _fr10_prev_read = None
                        if _fr10_capture_scan_payload or _fr10_commit_handoff_active:
                            _fr10_prev_read = globals().setdefault(
                                "_FR10_TREE_READ_PREV", {}
                            ).get((str(self.prefix), int(fr10_b)))
                        _fr10_rows = globals().get(
                            "_LUMO_FA_LAST_ACCEPTED_TREE_ROWS", []
                        )
                        _fr10_lens = globals().get(
                            "_LUMO_FA_LAST_ACCEPTED_TREE_LENS", []
                        )
                        _fr10_node_paths = globals().get(
                            "_LUMO_FA_LAST_ACCEPTED_TREE_NODE_PATHS", []
                        )
                        _fr10_has_accept = (
                            _fr10_lens is not None
                            and fr10_b < len(_fr10_lens)
                            and int(_fr10_lens[fr10_b]) > 0
                        )
                        if (
                            _fr10_prev_read is not None
                            and _fr10_has_accept
                            and fr10_b < len(_fr10_rows)
                        ):
                            _fr10_seed_row = max(
                                0,
                                min(
                                    int(_fr10_rows[fr10_b]),
                                    int(_fr10_prev_read["tree_n"]) - 1,
                                ),
                            )
                            _fr10_seed_path = (
                                [int(_x) for _x in _fr10_node_paths[fr10_b]]
                                if fr10_b < len(_fr10_node_paths)
                                else [int(_fr10_seed_row)]
                            )
                            _fr10_seed_path_len = min(
                                len(_fr10_seed_path),
                                int(_fr10_lens[fr10_b]),
                                int(spec_state_indices_tensor.size(-1)),
                                int(_fr10_prev_read["tree_n"]),
                            )
                            if _fr10_seed_path_len <= 0:
                                raise RuntimeError("accepted path is empty")
                            _fr10_seed_state_index = int(
                                spec_state_indices_tensor[
                                    fr10_b, _fr10_read_col
                                ]
                                .detach()
                                .cpu()
                                .item()
                            )
                            if _fr10_commit_handoff_active:
                                _fr10_commit_h0 = (
                                    ssm_state[_fr10_commit_state_index].detach().clone()
                                )
                            if _fr10_capture_scan_payload:
                                _fr10_capture_h0 = (
                                    ssm_state[_fr10_capture_state_index]
                                    .detach()
                                    .cpu()
                                    .clone()
                                )
                            try:
                                import json as _fr10_seed_json
                                import time as _fr10_seed_time

                                _fr10_seed_log = os.environ.get(
                                    "FR10_TREE_GDN_COMMIT_HANDOFF_LOG"
                                )
                                _fr10_seed_count = int(
                                    globals().get(
                                        "_FR10_TREE_READ_HANDOFF_LOG_COUNT", 0
                                    )
                                )
                                _fr10_seed_limit = int(
                                    os.environ.get(
                                        "FR10_TREE_GDN_COMMIT_HANDOFF_LIMIT", "32"
                                    )
                                )
                                if _fr10_seed_log and _fr10_seed_count < _fr10_seed_limit:
                                    _fr10_seed_accepted_bank_row = None
                                    _fr10_prev_spec_indices = _fr10_prev_read.get(
                                        "spec_state_indices"
                                    )
                                    if _fr10_prev_spec_indices is not None:
                                        _fr10_seed_accepted_bank_row = int(
                                            _fr10_prev_spec_indices[int(_fr10_seed_row)]
                                            .detach()
                                            .cpu()
                                            .item()
                                        )
                                    _fr10_seed_next_ssm = ssm_state[
                                        _fr10_seed_state_index
                                    ].detach().clone()
                                    _fr10_seed_expected_ssm = _fr10_prev_read[
                                        "tree_state"
                                    ][_fr10_seed_row]
                                    _fr10_seed_curr_conv = globals().get(
                                        "_FR10_COMMIT_HANDOFF_CURR_CONV_BY_B", {}
                                    ).get(int(fr10_b), {})
                                    _fr10_seed_next_conv = _fr10_seed_curr_conv.get(
                                        "prior"
                                    )
                                    _fr10_seed_expected_conv = _fr10_prev_read[
                                        "conv_rows"
                                    ][_fr10_seed_row]
                                    _fr10_seed_ssm_max = float(
                                        (
                                            _fr10_seed_next_ssm.float()
                                            - _fr10_seed_expected_ssm.float()
                                        )
                                        .abs()
                                        .max()
                                        .detach()
                                        .cpu()
                                        .item()
                                    )
                                    _fr10_seed_conv_max = None
                                    if _fr10_seed_next_conv is not None:
                                        _fr10_seed_conv_max = float(
                                            (
                                                _fr10_seed_next_conv.float()
                                                - _fr10_seed_expected_conv.float()
                                            )
                                            .abs()
                                            .max()
                                            .detach()
                                            .cpu()
                                            .item()
                                        )
                                    with open(_fr10_seed_log, "a", buffering=1) as _fr10_fh:
                                        _fr10_fh.write(
                                            _fr10_seed_json.dumps(
                                                {
                                                    "schema": "fr10.commit_native_handoff.v1",
                                                    "event": "commit_native_handoff",
                                                    "ts": round(_fr10_seed_time.time(), 4),
                                                    "layer_prefix": str(self.prefix),
                                                    "batch_index": int(fr10_b),
                                                    "prev_accepted_len": int(_fr10_lens[fr10_b]),
                                                    "accepted_node_row": int(_fr10_seed_row),
                                                    "accepted_node_path": [
                                                        int(_x)
                                                        for _x in _fr10_seed_path[
                                                            :_fr10_seed_path_len
                                                        ]
                                                    ],
                                                    "accepted_linear_read_col": int(
                                                        _fr10_seed_path_len - 1
                                                    ),
                                                    "accepted_spec_state_bank_row": (
                                                        None
                                                        if _fr10_seed_accepted_bank_row is None
                                                        else int(_fr10_seed_accepted_bank_row)
                                                    ),
                                                    "accepted_bank_row": (
                                                        None
                                                        if _fr10_seed_accepted_bank_row is None
                                                        else int(_fr10_seed_accepted_bank_row)
                                                    ),
                                                    "next_read_bank_row": int(
                                                        _fr10_seed_state_index
                                                    ),
                                                    "address_coincide": bool(
                                                        _fr10_seed_accepted_bank_row is not None
                                                        and int(_fr10_seed_accepted_bank_row)
                                                        == int(_fr10_seed_state_index)
                                                    ),
                                                    "linear_column_coincide": bool(
                                                        _fr10_seed_path_len
                                                        == int(_fr10_lens[fr10_b])
                                                    ),
                                                    "cache_state_coincide": bool(
                                                        _fr10_seed_ssm_max == 0.0
                                                        and _fr10_seed_conv_max == 0.0
                                                    ),
                                                    "ssm_bank_vs_cached_tree_state_max_abs": _fr10_seed_ssm_max,
                                                    "conv_cache_vs_cache_max_abs": _fr10_seed_conv_max,
                                                }
                                            )
                                            + chr(10)
                                        )
                                    _fr10_src_native_path = os.environ.get(
                                        "FR10_TREE_GDN_SRC_NATIVE_PAYLOAD"
                                    )
                                    if (
                                        _fr10_src_native_path
                                        and not globals().get(
                                            "_FR10_TREE_GDN_SRC_NATIVE_PAYLOAD_DONE",
                                            False,
                                        )
                                        and _fr10_prev_read.get("query_spec") is not None
                                        and _fr10_prev_read.get("h0_cpu") is not None
                                    ):
                                        _fr10_token_ids = globals().get(
                                            "_LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS", []
                                        )
                                        torch.save(
                                            {
                                                "schema": "fr10.src_native_handoff_payload.v1",
                                                "layer_prefix": str(self.prefix),
                                                "batch_index": int(fr10_b),
                                                "accepted_len": int(_fr10_lens[fr10_b]),
                                                "accepted_node_id": int(_fr10_seed_row),
                                                "accepted_node_path": [
                                                    int(_x)
                                                    for _x in _fr10_seed_path[
                                                        :_fr10_seed_path_len
                                                    ]
                                                ],
                                                "accepted_token_ids": (
                                                    [
                                                        int(_x)
                                                        for _x in _fr10_token_ids[fr10_b]
                                                    ]
                                                    if fr10_b < len(_fr10_token_ids)
                                                    else []
                                                ),
                                                "tree_parent": list(
                                                    _fr10_prev_read["tree_parent"]
                                                ),
                                                "output_scale": float(
                                                    _fr10_prev_read["output_scale"]
                                                ),
                                                "query_spec": _fr10_prev_read["query_spec"],
                                                "key_spec": _fr10_prev_read["key_spec"],
                                                "value_spec": _fr10_prev_read["value_spec"],
                                                "value_tree": _fr10_prev_read["value_tree"],
                                                "a": _fr10_prev_read["a"],
                                                "b": _fr10_prev_read["b"],
                                                "g_tree": _fr10_prev_read["g_tree"],
                                                "beta_tree": _fr10_prev_read["beta_tree"],
                                                "A_log": _fr10_prev_read["A_log"],
                                                "dt_bias": _fr10_prev_read["dt_bias"],
                                                "prev_h0": _fr10_prev_read["h0_cpu"],
                                                "serving_tree_state": _fr10_prev_read[
                                                    "tree_state_cpu"
                                                ],
                                                "next_read_ssm_state": _fr10_seed_next_ssm.detach()
                                                .cpu()
                                                .clone(),
                                                "prev_conv_prior": _fr10_prev_read[
                                                    "conv_prior_cpu"
                                                ],
                                                "serving_conv_rows": _fr10_prev_read[
                                                    "conv_rows_cpu"
                                                ],
                                                "next_read_conv_state": (
                                                    None
                                                    if _fr10_seed_next_conv is None
                                                    else _fr10_seed_next_conv.detach()
                                                    .cpu()
                                                    .clone()
                                                ),
                                                "accepted_spec_state_bank_row": (
                                                    None
                                                    if _fr10_seed_accepted_bank_row is None
                                                    else int(_fr10_seed_accepted_bank_row)
                                                ),
                                                "accepted_bank_row": (
                                                    None
                                                    if _fr10_seed_accepted_bank_row is None
                                                    else int(_fr10_seed_accepted_bank_row)
                                                ),
                                                "next_read_bank_row": int(
                                                    _fr10_seed_state_index
                                                ),
                                            },
                                            _fr10_src_native_path,
                                        )
                                        globals()[
                                            "_FR10_TREE_GDN_SRC_NATIVE_PAYLOAD_DONE"
                                        ] = True
                                    globals()[
                                        "_FR10_TREE_READ_HANDOFF_LOG_COUNT"
                                    ] = _fr10_seed_count + 1
                            except Exception as _fr10_seed_log_exc:
                                if os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1":
                                    raise RuntimeError(
                                        "FR10 tree SSM handoff logging failed: "
                                        + type(_fr10_seed_log_exc).__name__
                                        + ":"
                                        + str(_fr10_seed_log_exc)
                            ) from _fr10_seed_log_exc
                    except Exception as _fr10_seed_exc:
                        if os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1":
                            raise RuntimeError(
                                "FR10 tree SSM handoff oracle failed: "
                                + type(_fr10_seed_exc).__name__
                                + ":"
                                + str(_fr10_seed_exc)
                            ) from _fr10_seed_exc
                    _fr10_event_h0 = None
                    if _fr10_capture_scan_payload or _fr10_commit_handoff_active:
                        try:
                            _fr10_event_h0 = (
                                ssm_state[
                                    int(
                                        spec_state_indices_tensor[
                                            fr10_b, _fr10_read_col
                                        ].detach().cpu().item()
                                    )
                                ]
                                .detach()
                                .clone()
                            )
                        except Exception:
                            _fr10_event_h0 = None
                    if _fr13_boundary_on() and _fr13_boundary_layer_match(
                        self.prefix
                    ):
                        # FR13_REPLAY_BOUNDARY tap B (CONSUMER): the h0 column/
                        # row/bytes the scan is about to read, as-read. Joined
                        # by the reducer against tap A (event N) on req_id =
                        # the playbook row-8 boundary instrument. Eager-only
                        # diagnostics (syncs); emit() fail-louds on capture.
                        _fr13_bnd_lens_now = int(
                            _fr10_accepted_lens_tensor[fr10_b]
                            .detach().cpu().item()
                        )
                        _fr13_bnd_read_col = max(
                            0,
                            min(
                                _fr13_bnd_lens_now - 1,
                                int(spec_state_indices_tensor.size(-1)) - 1,
                            ),
                        )
                        _fr13_bnd_read_row = int(
                            spec_state_indices_tensor[
                                fr10_b, _fr13_bnd_read_col
                            ].detach().cpu().item()
                        )
                        (
                            _fr13_bnd_first8,
                            _fr13_bnd_sha,
                            _fr13_bnd_nb,
                        ) = _fr13_boundary_row_digest(
                            ssm_state[_fr13_bnd_read_row]
                        )
                        _fr13_bnd_req_ids = (
                            globals().get("_LUMO_FA_SAMPLER_ROW_REQ_IDS") or []
                        )
                        _fr13_bnd_window_now = [
                            int(_x)
                            for _x in spec_state_indices_tensor[fr10_b]
                            .detach().cpu().tolist()
                        ]
                        _fr13_boundary_emit({
                            "tap": "B",
                            "layer": str(self.prefix),
                            "slot": int(fr10_b),
                            "req_id": (
                                str(_fr13_bnd_req_ids[fr10_b])
                                if fr10_b < len(_fr13_bnd_req_ids)
                                else None
                            ),
                            "lens_now": _fr13_bnd_lens_now,
                            "read_col": int(_fr13_bnd_read_col),
                            "read_row": int(_fr13_bnd_read_row),
                            "first8": _fr13_bnd_first8,
                            "sha4096": _fr13_bnd_sha,
                            "num_spec_decodes": int(
                                attn_metadata.num_spec_decodes
                            ),
                            "window_now": _fr13_bnd_window_now,
                            "window_digest": _fr13_boundary_window_digest(
                                ssm_state, _fr13_bnd_window_now
                            ),
                            "replay_route": bool(_fr13_replay_route_on),
                        })
                        # FR13_CHASE_DIAG instrument (ii): IN-PROCESS byte
                        # verdict — producer's as-written digests (tap A,
                        # event N, _FR13_BOUNDARY_LAST_WRITTEN_BY_REQ) vs
                        # this consumer's as-read h0 row (event N+1). The
                        # playbook class-8 boundary instrument, joined here
                        # instead of by an offline reducer; ROWBUG class
                        # carried from the published path so the decision
                        # rule (A: integer mismatch + state byte-equal => H1;
                        # B: state as-read != as-written => class 4/7/8
                        # handoff fix FIRST) reads off one jsonl.
                        _fr13_ch_req = (
                            str(_fr13_bnd_req_ids[fr10_b])
                            if fr10_b < len(_fr13_bnd_req_ids)
                            else None
                        )
                        _fr13_ch_prev = (
                            globals().get(
                                "_FR13_BOUNDARY_LAST_WRITTEN_BY_REQ", {}
                            ).get(_fr13_ch_req)
                            if _fr13_ch_req is not None
                            else None
                        )
                        if _fr13_ch_prev is None:
                            _fr13_ch_verdict = "NO_PRIOR_WRITE"
                            _fr13_ch_w_sha = None
                        else:
                            _fr13_ch_w_sha = _fr13_ch_prev.get(
                                "sha_by_row", {}
                            ).get(str(_fr13_bnd_read_row))
                            if _fr13_ch_w_sha is None:
                                _fr13_ch_verdict = "READ_ROW_NOT_WRITTEN"
                            elif _fr13_ch_w_sha == _fr13_bnd_sha:
                                _fr13_ch_verdict = "BYTE_EQUAL"
                            else:
                                _fr13_ch_verdict = "BYTE_DIFF"
                        _fr13_boundary_emit({
                            "tap": "B_JOIN",
                            "layer": str(self.prefix),
                            "slot": int(fr10_b),
                            "req_id": _fr13_ch_req,
                            "verdict": _fr13_ch_verdict,
                            "read_row": int(_fr13_bnd_read_row),
                            "read_col": int(_fr13_bnd_read_col),
                            "lens_now": _fr13_bnd_lens_now,
                            "sha_as_read": _fr13_bnd_sha,
                            "sha_as_written": _fr13_ch_w_sha,
                            "prev_event": (
                                None if _fr13_ch_prev is None
                                else _fr13_ch_prev.get("event")
                            ),
                            "prev_accepted_len": (
                                None if _fr13_ch_prev is None
                                else _fr13_ch_prev.get("accepted_len")
                            ),
                            "prev_accepted_path": (
                                None if _fr13_ch_prev is None
                                else _fr13_ch_prev.get("accepted_path")
                            ),
                            "prev_rowbug": (
                                None if _fr13_ch_prev is None
                                else _fr13_ch_prev.get("rowbug")
                            ),
                        })
                    if _fr13_replay_route_on:
                        # FR13_REPLAY_ROUTE activation ring + scan-time
                        # snapshot. PERSISTENT preallocated staging only --
                        # never dict-pinned per-step buffers (the gate-4
                        # root cause #2, FR13_ACCEPT_ONLY_GATE4_FAIL_BIND.md).
                        # All staging buffers are allocated at GDN
                        # METADATA-BUILDER INIT (gdn_attn.py __init__, the
                        # persistent accepted-paths/lens pattern): a lazy
                        # first-flagged-forward allocation here would sit
                        # INSIDE the CUDA-graph-captured region =
                        # stale-pointer aliasing (gate-4 root cause #2), so
                        # this path only WRITES; it never allocates and
                        # creates no per-step Python objects.
                        # The ring stores EXACTLY what the scan consumes at
                        # consumed precision: k pre-l2norm, v, raw_a, raw_b
                        # byte-copies (~16.2KiB/node vs the 3.146MB state
                        # row). A_log/dt_bias are persistent params; q is not
                        # needed (q-side ops never touch state).
                        if getattr(self, "_fr13_replay_ring_k", None) is None:
                            raise RuntimeError(
                                "FR13_REPLAY_ROUTE: replay staging buffers "
                                "missing for layer " + str(self.prefix)
                                + "; they must be allocated at GDN metadata-"
                                "builder init (capture-unsafe lazy allocation "
                                "in the forward is banned)"
                            )
                        if (
                            self._fr13_replay_ring_k.size(1) != tree_n_pad
                            or self._fr13_replay_ring_k.dtype != key_spec.dtype
                            or self._fr13_replay_ring_v.dtype != value_tree.dtype
                            or self._fr13_replay_ring_a.dtype != a.dtype
                            or self._fr13_replay_ring_b.dtype != b.dtype
                        ):
                            raise RuntimeError(
                                "FR13_REPLAY_ROUTE: staging ring shape/dtype "
                                "mismatch for layer " + str(self.prefix)
                                + "; the init-time allocation must match the "
                                "consumed activations exactly (byte-copy "
                                "contract)"
                            )
                        if fr10_b == 0:
                            # SNAPSHOT prev accepted lens + spec indices AT
                            # SCAN TIME: the committer refills
                            # _LUMO_FA_ACCEPTED_TREE_LENS_TENSOR with the NEW
                            # lens BEFORE its publish/replay block, so the
                            # replay's h0 base column (prev_len-1) is not
                            # derivable at the launch site (the verify-rider
                            # Option-1 gap).
                            self._fr13_replay_prev_lens[
                                : attn_metadata.num_spec_decodes
                            ].copy_(
                                _fr10_accepted_lens_tensor[
                                    : attn_metadata.num_spec_decodes
                                ]
                            )
                            self._fr13_replay_spec_idx[
                                : attn_metadata.num_spec_decodes
                            ].copy_(
                                spec_state_indices_tensor[
                                    : attn_metadata.num_spec_decodes
                                ]
                            )
                            # Capture-safe staging handshake (replaces the
                            # banned per-step meta dict): the preallocated
                            # int32 flag tensor is written by CAPTURED device
                            # ops ([0]=fresh, [1]=staged spec-decode rows), so
                            # a CUDA-graph REPLAY re-arms freshness even
                            # though this Python never re-runs; the fill
                            # values are per-captured-graph constants
                            # (UNIFORM_BATCH capture). The bank ref is a fixed
                            # attribute write of an existing persistent
                            # tensor -- no per-step object creation.
                            if not _FR13_FLAGS_INKERNEL:
                                self._fr13_replay_flags[0].fill_(1)
                                self._fr13_replay_flags[1].fill_(
                                    attn_metadata.num_spec_decodes
                                )
                            self._fr13_replay_ssm_state = ssm_state
                            # STATELESS-TREE: stage the LIVE conv cache view too
                            # (same fixed-attribute pattern), so the post-accept
                            # conv committer can copy this-step's committed leaf
                            # window -> col 0 and burn the ephemeral cols.
                            self._fr13_replay_conv_state = conv_state
                            if _FR13_FIXED32_MODE:
                                _fr13_f32_done = globals().setdefault(
                                    "_FR13_FIXED32_PRESEEDED_BATCHES", set()
                                )
                                _fr13_f32_B = int(
                                    attn_metadata.num_spec_decodes
                                )
                                _fr13_f32_inputs = (
                                    _fr13_fixed32_boot_preseed_inputs()
                                )
                                if _fr13_f32_inputs is not None:
                                        (
                                            _fr13_f32_stacks,
                                            _fr13_f32_layers,
                                            _fr13_f32_order,
                                            _fr13_f32_layer_objects,
                                            _fr13_f32_banks,
                                            _fr13_f32_conv_banks,
                                        ) = _fr13_f32_inputs
                                        _fr13_f32_stacks[
                                            "fixed32_order"
                                        ] = _fr13_f32_order
                                        _fr13_f32_stacks[
                                            "fixed32_layers"
                                        ] = _fr13_f32_layer_objects
                                        _fr13_f32_stacks[
                                            "fixed32_banks"
                                        ] = _fr13_f32_banks
                                        _fr13_f32_stacks[
                                            "fixed32_conv_banks"
                                        ] = _fr13_f32_conv_banks
                                        _fr13_f32_stacks[
                                            "fixed32_bank_anchor"
                                        ] = _fr13_f32_banks[0]
                                        _fr13_f32_stacks[
                                            "fixed32_bank_shape"
                                        ] = tuple(
                                            int(_fr13_f32_dim)
                                            for _fr13_f32_dim
                                            in _fr13_f32_banks[0].shape
                                        )
                                        _fr13_f32_stacks[
                                            "fixed32_bank_stride"
                                        ] = int(
                                            _fr13_f32_banks[0].stride(0)
                                        )
                                        _fr13_f32_stacks[
                                            "fixed32_output_scale"
                                        ] = float(
                                            _fr13_f32_stacks["output_scale"]
                                        )
                                        _fr13_f32_stacks[
                                            "fixed32_qk_l2norm"
                                        ] = True
                                        import importlib.util as _fr13_f32_ilu
                                        import sys as _fr13_f32_sys
                                        _fr13_f32_dm = _fr13_f32_sys.modules.get(
                                            "_fr13_device_multidraft_kernel"
                                        )
                                        if _fr13_f32_dm is None:
                                            _fr13_f32_path = os.environ.get(
                                                "FR13_DEVICE_MULTIDRAFT_KERNEL",
                                                "/workspace/scripts/"
                                                "fr13_device_multidraft_kernel.py",
                                            )
                                            _fr13_f32_spec = (
                                                _fr13_f32_ilu.spec_from_file_location(
                                                    "_fr13_device_multidraft_kernel",
                                                    _fr13_f32_path,
                                                )
                                            )
                                            if (
                                                _fr13_f32_spec is None
                                                or _fr13_f32_spec.loader is None
                                            ):
                                                raise RuntimeError(
                                                    "FR13 fixed32 cannot load "
                                                    "device committer module"
                                                )
                                            _fr13_f32_dm = (
                                                _fr13_f32_ilu.module_from_spec(
                                                    _fr13_f32_spec
                                                )
                                            )
                                            _fr13_f32_spec.loader.exec_module(
                                                _fr13_f32_dm
                                            )
                                            _fr13_f32_sys.modules[
                                                "_fr13_device_multidraft_kernel"
                                            ] = _fr13_f32_dm
                                        _fr13_f32_dm.fr13_fixed32_taw_preseed(
                                            spec_state_indices_tensor.device,
                                            mode=_FR13_FIXED32_MODE,
                                        )
                                        from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
                                            preseed_fixed32_committer_graphs_all_batches
                                            as _fr13_f32_preseed,
                                            preseed_fixed32_conv_col0_pregather
                                            as _fr13_f32_pregather,
                                            selfcheck_fixed32_conv_col0_ssi_sources
                                            as _fr13_f32_pregather_selfcheck,
                                        )
                                        _fr13_f32_cap = min(
                                            4,
                                            int(
                                                _fr13_f32_stacks[
                                                    "fixed32_server_cap"
                                                ]
                                            ),
                                        )
                                        if (
                                            _fr13_f32_cap < 1
                                            or int(
                                                _fr13_f32_stacks[
                                                    "ring_k"
                                                ].shape[1]
                                            ) < _fr13_f32_cap
                                            or int(
                                                _fr10_accepted_paths_tensor.shape[0]
                                            ) < _fr13_f32_cap
                                        ):
                                            raise RuntimeError(
                                                "FR13 fixed32 preseed capacity "
                                                "does not cover server occupancy"
                                            )
                                        _fr13_f32_preseed(
                                            banks_list=_fr13_f32_banks,
                                            spec_state_indices=(
                                                _fr13_f32_stacks["spec_idx"]
                                            ),
                                            accepted_paths=(
                                                _fr10_accepted_paths_tensor[
                                                    :_fr13_f32_cap, :16
                                                ].contiguous()
                                            ),
                                            accepted_lens=(
                                                _fr10_accepted_lens_tensor[
                                                    :_fr13_f32_cap
                                                ]
                                            ),
                                            k_rings=_fr13_f32_stacks["ring_k"],
                                            k_norm_rings=_fr13_f32_stacks[
                                                "ring_k_norm"
                                            ],
                                            gate_rings=_fr13_f32_stacks[
                                                "ring_gate"
                                            ],
                                            v_rings=_fr13_f32_stacks["ring_v"],
                                            a_rings=_fr13_f32_stacks["ring_a"],
                                            b_rings=_fr13_f32_stacks["ring_b"],
                                            A_logs=_fr13_f32_stacks["A_log"],
                                            dt_biases=_fr13_f32_stacks[
                                                "dt_bias"
                                            ],
                                            num_layers=48,
                                            max_batch_size=_fr13_f32_cap,
                                            output_scale=(
                                                _fr13_f32_stacks[
                                                    "fixed32_output_scale"
                                                ]
                                            ),
                                            use_qk_l2norm_in_kernel=(
                                                _fr13_f32_stacks[
                                                    "fixed32_qk_l2norm"
                                                ]
                                            ),
                                            burn_node_bank=False,
                                            root_node=0,
                                            max_path=16,
                                        )
                                        _fr13_f32_pregather(
                                            conv_banks=_fr13_f32_conv_banks,
                                            ssm_banks=_fr13_f32_banks,
                                            layer_order=_fr13_f32_order,
                                            max_batch_size=_fr13_f32_cap,
                                            commit_spec_state_indices=(
                                                _fr13_f32_stacks["spec_idx"]
                                            ),
                                            accepted_paths=(
                                                _fr10_accepted_paths_tensor
                                            ),
                                            accepted_lens=(
                                                _fr10_accepted_lens_tensor
                                            ),
                                            commit_source_stagings=(
                                                freeze_conv_wb_staging_sources(
                                                    _fr13_f32_order,
                                                    _fr13_f32_cap
                                                    * _fr13_wbb_srows,
                                                    int(conv_state.size(1)),
                                                    conv_state.dtype,
                                                    conv_state.device,
                                                )
                                            ),
                                            commit_state_src=(
                                                _fr13_tcf_state_src
                                            ),
                                            source_rows_per_batch=(
                                                _fr13_wbb_srows
                                            ),
                                        )
                                        if _fr13_f32_B != _fr13_f32_cap:
                                            raise RuntimeError(
                                                "FR13 fixed32 initial pregather "
                                                "producer must cover server "
                                                "capacity: "
                                                f"B={_fr13_f32_B} "
                                                f"capacity={_fr13_f32_cap}"
                                            )
                                        for _fr13_f32_live_B in range(
                                            1, _fr13_f32_cap + 1
                                        ):
                                            _fr13_f32_pregather_selfcheck(
                                                num_spec_decodes=(
                                                    _fr13_f32_live_B
                                                ),
                                            )
                                        _fr13_f32_done.update(
                                            range(1, _fr13_f32_cap + 1)
                                        )
                                        globals()[
                                            "_FR13_FIXED32_PRESEED_CAP"
                                        ] = _fr13_f32_cap
                        # FR13_RING_EXPORT: when ON, the scan kernel stages the
                        # ring in-kernel from its own inputs (byte-copy contract
                        # preserved; see _tree_gdn_kernel RING_EXPORT), removing
                        # these 4 per-layer aten copy launches -- measured as
                        # part of the +20ms/draft tree-only elementwise soup in
                        # the 2026-07-22 nsys differential. Default OFF =
                        # byte-identical staged bytes via the original copies.
                        _fr13_ring_export = (
                            os.environ.get("FR13_RING_EXPORT", "1") == "1"
                        )
                        if not _fr13_ring_export:
                            self._fr13_replay_ring_k[fr10_b, :tree_n].copy_(
                                key_spec[0, start:end]
                            )
                            self._fr13_replay_ring_v[fr10_b, :tree_n].copy_(
                                value_tree[start:end]
                            )
                            self._fr13_replay_ring_a[fr10_b, :tree_n].copy_(
                                a[start:end]
                            )
                            self._fr13_replay_ring_b[fr10_b, :tree_n].copy_(
                                b[start:end]
                            )
                    _fr13_fixed32_batch_gdn_selector = (
                        fixed32_batch_gdn_selector(
                            int(attn_metadata.num_spec_decodes)
                        )
                        if _FR13_FIXED32_MODE
                        else None
                    )
                    _fr13_fixed32_batch_gdn = bool(
                        _fr13_fixed32_batch_gdn_selector is not None
                    )
                    if _fr13_fixed32_batch_gdn:
                        if fr10_b == 0:
                            _fr13_gdn_batch = int(
                                attn_metadata.num_spec_decodes
                            )
                            _fr13_gdn_rows = _fr13_gdn_batch * tree_n
                            launch_tree_gdn_prepared_fixed32_batch(
                                staging_flags=(
                                    self._fr13_replay_flags
                                    if _FR13_FLAGS_INKERNEL else None
                                ),
                                staging_rows=_fr13_gdn_batch,
                                batch_size=_fr13_gdn_batch,
                                q=query_spec[
                                    0, :_fr13_gdn_rows
                                ].contiguous(),
                                k=key_spec[
                                    0, :_fr13_gdn_rows
                                ].contiguous(),
                                v=value_tree[:_fr13_gdn_rows].contiguous(),
                                g=g_tree[:_fr13_gdn_rows].contiguous(),
                                beta=beta_tree[:_fr13_gdn_rows].contiguous(),
                                raw_a=a[:_fr13_gdn_rows].contiguous(),
                                raw_b=b[:_fr13_gdn_rows].contiguous(),
                                A_log=self.A_log,
                                dt_bias=self.dt_bias,
                                h0=ssm_state,
                                h0_indices=spec_state_indices_tensor,
                                h0_num_accepted_tokens=(
                                    _fr10_accepted_lens_tensor
                                ),
                                h0_use_accepted_column=(
                                    os.environ.get(
                                        "FR13_TREE_RUNROW_INIT", "1"
                                    ) != "1"
                                ),
                                n_actual=tree_n,
                                n_pad=tree_n_pad,
                                strict_mask=(
                                    attn_metadata.fr10_tree_strict_mask
                                ),
                                visible_mask=(
                                    attn_metadata.fr10_tree_visible_mask
                                ),
                                out=core_attn_out_spec[
                                    0, :_fr13_gdn_rows
                                ],
                                output_scale=self.head_k_dim**-0.5,
                                use_qk_l2norm_in_kernel=True,
                                ring_k=(
                                    self._fr13_replay_ring_k[
                                        :_fr13_gdn_batch, :tree_n
                                    ].flatten(0, 1)
                                    if _fr13_replay_route_on
                                    and os.environ.get(
                                        "FR13_RING_EXPORT", "1"
                                    ) == "1"
                                    else None
                                ),
                                ring_k_norm=(
                                    self._fr13_replay_ring_k_norm[
                                        :_fr13_gdn_batch, :tree_n
                                    ].flatten(0, 1)
                                    if getattr(
                                        self,
                                        "_fr13_replay_ring_k_norm",
                                        None,
                                    )
                                    is not None
                                    else None
                                ),
                                ring_gate=(
                                    self._fr13_replay_ring_gate[
                                        :_fr13_gdn_batch, :tree_n
                                    ].flatten(0, 1)
                                    if getattr(
                                        self,
                                        "_fr13_replay_ring_gate",
                                        None,
                                    )
                                    is not None
                                    else None
                                ),
                                ring_v=(
                                    self._fr13_replay_ring_v[
                                        :_fr13_gdn_batch, :tree_n
                                    ].flatten(0, 1)
                                    if _fr13_replay_route_on
                                    and os.environ.get(
                                        "FR13_RING_EXPORT", "1"
                                    ) == "1"
                                    else None
                                ),
                                ring_a=(
                                    self._fr13_replay_ring_a[
                                        :_fr13_gdn_batch, :tree_n
                                    ].flatten(0, 1)
                                    if _fr13_replay_route_on
                                    and os.environ.get(
                                        "FR13_RING_EXPORT", "1"
                                    ) == "1"
                                    else None
                                ),
                                ring_b=(
                                    self._fr13_replay_ring_b[
                                        :_fr13_gdn_batch, :tree_n
                                    ].flatten(0, 1)
                                    if _fr13_replay_route_on
                                    and os.environ.get(
                                        "FR13_RING_EXPORT", "1"
                                    ) == "1"
                                    else None
                                ),
                                invocation_counter=(
                                    attn_metadata.fr10_tree_invocation_counter
                                    if os.environ.get(
                                        "FR10_METRICS", "0"
                                    ) == "1"
                                    else None
                                ),
                            )
                        tree_out = core_attn_out_spec[0, start:end]
                    else:
                        tree_out, _ = launch_tree_gdn_prepared(
                            staging_flags=(
                                self._fr13_replay_flags
                                if _FR13_FLAGS_INKERNEL else None
                            ),
                            staging_rows=int(attn_metadata.num_spec_decodes),
                            q=query_spec[0, start:end].contiguous(),
                            k=key_spec[0, start:end].contiguous(),
                            v=value_tree[start:end].contiguous(),
                            g=g_tree[start:end].contiguous(),
                            beta=beta_tree[start:end].contiguous(),
                            raw_a=a[start:end].contiguous(),
                            raw_b=b[start:end].contiguous(),
                            A_log=self.A_log,
                            dt_bias=self.dt_bias,
                            h0=ssm_state,
                            h0_indices=spec_state_indices_tensor,
                            h0_num_accepted_tokens=_fr10_accepted_lens_tensor,
                            h0_is_bank=True,
                            h0_index_row=fr10_b * spec_state_indices_tensor.size(-1),
                            h0_batch_index=fr10_b,
                            # STATELESS-TREE: FR13_TREE_RUNROW_INIT -> seed the scan h0
                            # from col 0 (running row) instead of accepted col nacc-1.
                            h0_use_accepted_column=(
                                os.environ.get("FR13_TREE_RUNROW_INIT", "1") != "1"
                            ),
                            n_actual=tree_n,
                            n_pad=tree_n_pad,
                            strict_mask=attn_metadata.fr10_tree_strict_mask,
                            visible_mask=attn_metadata.fr10_tree_visible_mask,
                            out=core_attn_out_spec[0, start:end],
                            state=tree_state,
                            output_scale=self.head_k_dim**-0.5,
                            use_qk_l2norm_in_kernel=True,
                            ring_k=(
                                self._fr13_replay_ring_k[fr10_b]
                                if _fr13_replay_route_on
                                and os.environ.get("FR13_RING_EXPORT", "1") == "1"
                                else None
                            ),
                            ring_k_norm=(
                                self._fr13_replay_ring_k_norm[fr10_b]
                                if getattr(
                                    self,
                                    "_fr13_replay_ring_k_norm",
                                    None,
                                )
                                is not None
                                else None
                            ),
                            ring_gate=(
                                self._fr13_replay_ring_gate[fr10_b]
                                if getattr(
                                    self,
                                    "_fr13_replay_ring_gate",
                                    None,
                                )
                                is not None
                                else None
                            ),
                            ring_v=(
                                self._fr13_replay_ring_v[fr10_b]
                                if _fr13_replay_route_on
                                and os.environ.get("FR13_RING_EXPORT", "1") == "1"
                                else None
                            ),
                            ring_a=(
                                self._fr13_replay_ring_a[fr10_b]
                                if _fr13_replay_route_on
                                and os.environ.get("FR13_RING_EXPORT", "1") == "1"
                                else None
                            ),
                            ring_b=(
                                self._fr13_replay_ring_b[fr10_b]
                                if _fr13_replay_route_on
                                and os.environ.get("FR13_RING_EXPORT", "1") == "1"
                                else None
                            ),
                            invocation_counter=(
                                attn_metadata.fr10_tree_invocation_counter
                                if os.environ.get("FR10_METRICS", "0") == "1"
                                else None
                            ),
                        )
                    if (
                        _FR13_FIXED32_MODE
                        and (
                            bool(
                                torch.cuda.is_available()
                                and torch.cuda.is_current_stream_capturing()
                            )
                            or _fr13_fixed32_observed_event_active()
                        )
                    ):
                        _fr13_f32_scan_state = subtree_get(
                            int(tree_n),
                            int(value_tree.shape[1]),
                            int(value_tree.shape[2]),
                            int(query_spec.shape[-1]),
                            query_spec.device,
                        )
                        _fr13_fixed32_observed_gdn(
                            str(self.prefix),
                            int(fr10_b),
                            int(attn_metadata.num_spec_decodes),
                            int(tree_n),
                            int(tree_n_pad),
                            int(query_spec[0, start:end].shape[0]),
                            int(key_spec[0, start:end].shape[0]),
                            int(value_tree[start:end].shape[0]),
                            int(core_attn_out_spec[0, start:end].shape[0]),
                            tuple(
                                int(value)
                                for value in attn_metadata.fr10_tree_strict_mask.shape
                            ),
                            tuple(
                                int(value)
                                for value in attn_metadata.fr10_tree_visible_mask.shape
                            ),
                            {
                                "schedule": _fr13_f32_scan_state.get("schedule"),
                                "route_armed": _fr13_f32_scan_state.get(
                                    "route_armed"
                                ),
                                "n_levels": int(
                                    _fr13_f32_scan_state.get("n_levels", -1)
                                ),
                                "critical": int(
                                    _fr13_f32_scan_state.get("critical", -1)
                                ),
                                "parent_nodes": len(
                                    _fr13_f32_scan_state.get("parent", ())
                                ),
                                "emask_rows": int(
                                    _fr13_f32_scan_state["emask"].shape[0]
                                ),
                                "export_rows": int(
                                    _fr13_f32_scan_state["export"].shape[0]
                                ),
                                "fixed32_contract": _fr13_f32_scan_state.get(
                                    "fixed32_contract"
                                ),
                                "fixed32_single_launch_contract": (
                                    _fr13_f32_scan_state.get(
                                        "fixed32_single_launch_contract"
                                    )
                                ),
                                "executed_gdn": _fr13_f32_scan_state.get(
                                    "last_executed_gdn"
                                ),
                            },
                            bool(
                                torch.cuda.is_available()
                                and torch.cuda.is_current_stream_capturing()
                            ),
                        )
                    if _fr12_native_spine_scan_enabled:
                        assert _fr12_scan_path0_node_tensor is not None
                        _fr12_scan_path0_len = int(
                            _fr12_scan_path0_node_tensor.numel()
                        )
                        _fr12_scan_qsl = (
                            torch.arange(
                                2,
                                dtype=spec_query_start_loc.dtype,
                                device=spec_query_start_loc.device,
                            )
                            * _fr12_scan_path0_len
                        )
                        _fr12_scan_indices = spec_state_indices_tensor[
                            fr10_b : fr10_b + 1
                        ].index_select(1, _fr12_scan_path0_node_tensor)
                        _fr12_scan_num_accepted = num_accepted_tokens[
                            fr10_b : fr10_b + 1
                        ]
                        _fr12_native_scan_out, _ = (
                            fused_sigmoid_gating_delta_rule_update(
                                A_log=self.A_log,
                                a=a[start:end]
                                .index_select(0, _fr12_scan_path0_node_tensor)
                                .contiguous(),
                                b=b[start:end]
                                .index_select(0, _fr12_scan_path0_node_tensor)
                                .contiguous(),
                                dt_bias=self.dt_bias,
                                q=query_spec[0, start:end]
                                .index_select(0, _fr12_scan_path0_node_tensor)
                                .unsqueeze(0)
                                .contiguous(),
                                k=key_spec[0, start:end]
                                .index_select(0, _fr12_scan_path0_node_tensor)
                                .unsqueeze(0)
                                .contiguous(),
                                v=value_spec[0, start:end]
                                .index_select(0, _fr12_scan_path0_node_tensor)
                                .unsqueeze(0)
                                .contiguous(),
                                initial_state=ssm_state,
                                inplace_final_state=True,
                                cu_seqlens=_fr12_scan_qsl,
                                ssm_state_indices=_fr12_scan_indices,
                                num_accepted_tokens=_fr12_scan_num_accepted,
                                use_qk_l2norm_in_kernel=True,
                            )
                        )
                        tree_out.index_copy_(
                            0,
                            _fr12_scan_path0_node_tensor,
                            _fr12_native_scan_out.squeeze(0).to(dtype=tree_out.dtype),
                        )
                        tree_state.index_copy_(
                            0,
                            _fr12_scan_path0_node_tensor,
                            ssm_state.index_select(
                                0, _fr12_scan_indices.squeeze(0).to(torch.long)
                            ).to(dtype=tree_state.dtype),
                        )
                    _fr10_root_h0_log = os.environ.get("FR10_TREE_GDN_ROOT_H0_LOG")
                    _fr10_root_h0_log_prefix = os.environ.get(
                        "FR10_TREE_GDN_ROOT_H0_LOG_LAYER_PREFIX",
                        "language_model.model.layers.0.linear_attn",
                    )
                    _fr10_root_h0_log_limit = int(
                        os.environ.get("FR10_TREE_GDN_ROOT_H0_LOG_LIMIT", "20")
                    )
                    _fr10_root_h0_log_count = int(
                        globals().get("_FR10_TREE_GDN_ROOT_H0_LOG_COUNT", 0)
                    )
                    _fr10_root_h0_in_cuda_capture = bool(
                        torch.cuda.is_available()
                        and torch.cuda.is_current_stream_capturing()
                    )
                    if (
                        _fr10_root_h0_log
                        and fr10_b == 0
                        and str(self.prefix) == _fr10_root_h0_log_prefix
                        and _fr10_root_h0_log_count < _fr10_root_h0_log_limit
                        and not _fr10_root_h0_in_cuda_capture
                    ):
                        try:
                            _fr10_live_col = max(
                                0,
                                min(
                                    int(
                                        _fr10_accepted_lens_tensor[fr10_b]
                                        .detach()
                                        .cpu()
                                        .item()
                                    )
                                    - 1,
                                    int(spec_state_indices_tensor.size(-1)) - 1,
                                ),
                            )
                            _fr10_col0_row = int(
                                spec_state_indices_tensor[fr10_b, 0]
                                .detach()
                                .cpu()
                                .item()
                            )
                            _fr10_live_row = int(
                                spec_state_indices_tensor[fr10_b, _fr10_live_col]
                                .detach()
                                .cpu()
                                .item()
                            )
                            _fr10_root_row = {
                                "event_index": int(_fr10_root_h0_log_count),
                                "schema": "fr10.tree_root_h0_probe_row.v2",
                                "layer_prefix": str(self.prefix),
                                "batch_index": int(fr10_b),
                                "tree_n": int(tree_n),
                                "query_spec_rows": int(query_spec.size(1)),
                                "start": int(start),
                                "end": int(end),
                                "accepted_len": int(_fr10_live_col + 1),
                                "col0": 0,
                                "live_col": int(_fr10_live_col),
                                "col0_bank_row": int(_fr10_col0_row),
                                "live_bank_row": int(_fr10_live_row),
                                "h0_col0": ssm_state[_fr10_col0_row]
                                .detach()
                                .cpu()
                                .clone(),
                                "h0_live": ssm_state[_fr10_live_row]
                                .detach()
                                .cpu()
                                .clone(),
                                "q_root": query_spec[0, start].detach().cpu().clone(),
                                "k_root": key_spec[0, start].detach().cpu().clone(),
                                "value_spec_root": value_spec[start]
                                .detach()
                                .cpu()
                                .clone(),
                                "value_tree_root": value_tree[start]
                                .detach()
                                .cpu()
                                .clone(),
                                "a_root": a[start].detach().cpu().clone(),
                                "b_root": b[start].detach().cpu().clone(),
                                "g_tree_root": g_tree[start].detach().cpu().clone(),
                                "beta_tree_root": beta_tree[start]
                                .detach()
                                .cpu()
                                .clone(),
                                "A_log": self.A_log.detach().cpu().clone(),
                                "dt_bias": self.dt_bias.detach().cpu().clone(),
                                "serving_tree_out_root": tree_out[0]
                                .detach()
                                .cpu()
                                .clone(),
                                "serving_tree_state_root": tree_state[0]
                                .detach()
                                .cpu()
                                .clone(),
                                "output_scale": float(self.head_k_dim**-0.5),
                            }
                            _fr10_root_rows = list(
                                globals().get("_FR10_TREE_GDN_ROOT_H0_LOG_ROWS", [])
                            )
                            _fr10_root_rows.append(_fr10_root_row)
                            globals()["_FR10_TREE_GDN_ROOT_H0_LOG_ROWS"] = (
                                _fr10_root_rows
                            )
                            torch.save(
                                {
                                    "schema": "fr10.tree_root_h0_probe.v2",
                                    "layer_prefix": _fr10_root_h0_log_prefix,
                                    "limit": int(_fr10_root_h0_log_limit),
                                    "rows": _fr10_root_rows,
                                },
                                _fr10_root_h0_log,
                            )
                            globals()["_FR10_TREE_GDN_ROOT_H0_LOG_COUNT"] = (
                                _fr10_root_h0_log_count + 1
                            )
                        except Exception as _fr10_root_h0_exc:
                            if os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1":
                                raise RuntimeError(
                                    "FR10 tree root h0 probe failed: "
                                    + type(_fr10_root_h0_exc).__name__
                                    + ":"
                                    + str(_fr10_root_h0_exc)
                                ) from _fr10_root_h0_exc
                    if not _FR13_EAGER_PACK:
                        # FR13_EAGER_PACK (FIX-2 2h): VERIFIED-DEAD self-copy.
                        # launch_tree_gdn_prepared writes into out=
                        # core_attn_out_spec[0, start:end] and returns that
                        # same view as tree_out, so this setitem copies the
                        # view onto itself (48 kernels/draft of identity).
                        core_attn_out_spec[0, start:end] = tree_out[:tree_n]
                    if _fr10_capture_scan_payload:
                        try:
                            _fr10_payload_path = os.environ.get(
                                "FR10_TREE_GDN_CAPTURE_PAYLOAD"
                            )
                            torch.save(
                                {
                                    "schema": "fr10.tree_gdn_scan_capture.v1",
                                    "layer_prefix": str(self.prefix),
                                    "batch_index": int(fr10_b),
                                    "tree_parent": [
                                        int(_x)
                                        for _x in attn_metadata.fr10_tree_parent.detach()
                                        .cpu()
                                        .tolist()
                                    ],
                                    "n_actual": int(tree_n),
                                    "n_pad": int(tree_n_pad),
                                    "state_index": int(_fr10_capture_state_index),
                                    "output_scale": float(self.head_k_dim**-0.5),
                                    "query_spec": query_spec[0, start:end]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                    "key_spec": key_spec[0, start:end]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                    "value_spec": value_spec[0, start:end]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                    "a": a[start:end].detach().cpu().clone(),
                                    "b": b[start:end].detach().cpu().clone(),
                                    "A_log": self.A_log.detach().cpu().clone(),
                                    "dt_bias": self.dt_bias.detach().cpu().clone(),
                                    "h0": _fr10_capture_h0,
                                    "value_tree": value_tree[start:end]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                    "g_tree": g_tree[start:end].detach().cpu().clone(),
                                    "beta_tree": beta_tree[start:end]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                    "serving_out": tree_out[:tree_n]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                    "serving_state": tree_state[:tree_n]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                },
                                _fr10_payload_path,
                            )
                            globals()["_FR10_TREE_GDN_CAPTURE_DONE"] = True
                        except Exception as _fr10_capture_exc:
                            logger.warning_once(
                                "FR10 tree GDN scan capture failed: %s",
                                _fr10_capture_exc,
                            )
                    # Persist every verified tree-node state. The sampled
                    # committer publishes the accepted path after this forward;
                    # the next decode remaps those node rows into stock linear
                    # columns with launch_tree_state_linear_remap before the
                    # recurrent consumers read them.
                    #
                    # FR13_REPLAY_ROUTE replaces this all-rows publish: the
                    # committer launches launch_tree_gdn_replay, which
                    # re-executes root+accepted-path from the activation ring
                    # and writes the bank LINEAR columns directly (including
                    # the zero-accept row-0 refresh), so nothing is published
                    # here and the next-step ssm remap is skipped.
                    if _fr10_commit_handoff_active:
                        try:
                            _fr10_curr_conv = globals().get(
                                "_FR10_COMMIT_HANDOFF_CURR_CONV_BY_B", {}
                            ).get(int(fr10_b), {})
                            _fr10_curr_conv_rows = _fr10_curr_conv.get("rows")
                            if _fr10_curr_conv_rows is not None:
                                globals().setdefault(
                                    "_FR10_TREE_READ_PREV", {}
                                )[(str(self.prefix), int(fr10_b))] = {
                                    "tree_n": int(tree_n),
                                    "tree_state": tree_state[:tree_n].detach().clone(),
                                    "conv_rows": _fr10_curr_conv_rows[
                                        :tree_n
                                    ].detach().clone(),
                                    "spec_state_indices": spec_state_indices_tensor[
                                        fr10_b, :tree_n
                                    ].detach().clone(),
                                    "tree_parent": [
                                        int(_x)
                                        for _x in attn_metadata.fr10_tree_parent.detach()
                                        .cpu()
                                        .tolist()
                                    ],
                                    "output_scale": float(self.head_k_dim**-0.5),
                                    "query_spec": query_spec[
                                        0, start:end
                                    ].detach().cpu().clone(),
                                    "key_spec": key_spec[
                                        0, start:end
                                    ].detach().cpu().clone(),
                                    "value_spec": value_spec[
                                        0, start:end
                                    ].detach().cpu().clone(),
                                    "value_tree": value_tree[
                                        start:end
                                    ].detach().cpu().clone(),
                                    "a": a[start:end].detach().cpu().clone(),
                                    "b": b[start:end].detach().cpu().clone(),
                                    "g_tree": g_tree[start:end].detach().cpu().clone(),
                                    "beta_tree": beta_tree[
                                        start:end
                                    ].detach().cpu().clone(),
                                    "A_log": self.A_log.detach().cpu().clone(),
                                    "dt_bias": self.dt_bias.detach().cpu().clone(),
                                    "h0_cpu": (
                                        None
                                        if _fr10_event_h0 is None
                                        else _fr10_event_h0.detach().cpu().clone()
                                    ),
                                    "tree_state_cpu": tree_state[
                                        :tree_n
                                    ].detach().cpu().clone(),
                                    "conv_prior_cpu": _fr10_curr_conv.get(
                                        "prior"
                                    ).detach().cpu().clone(),
                                    "conv_rows_cpu": _fr10_curr_conv_rows[
                                        :tree_n
                                    ].detach().cpu().clone(),
                                }
                        except Exception:
                            pass
                    _fr10_scan_diag = getattr(
                        attn_metadata, "fr10_tree_conv_diag", None
                    )
                    if (
                        os.environ.get("FR10_METRICS", "0") == "1"
                        and _fr10_scan_diag is not None
                        # FR13_REPLAY_ROUTE: diag[12]/[13] measure the staged
                        # all-rows publish, which does not exist under the
                        # replay route; skipping keeps them at zeros-init, so
                        # treat scan_state_staging as VACUOUS when the flag is
                        # on (do not gate on it; use the replay byte A/B and
                        # the durable accepted-state diff gate instead).
                        and not _fr13_replay_route_on
                    ):
                        _fr10_staged_state = ssm_state.index_select(
                            0, spec_state_indices_tensor[fr10_b, :tree_n].to(torch.long)
                        )
                        _fr10_staged_delta = (
                            _fr10_staged_state.float()
                            - tree_state[:tree_n].to(dtype=ssm_state.dtype).float()
                        ).abs().max()
                        _fr10_scan_diag[12].copy_(
                            torch.maximum(_fr10_scan_diag[12], _fr10_staged_delta)
                        )
                        _fr10_scan_diag[13].add_(
                            (_fr10_staged_delta != 0).to(dtype=torch.float32)
                        )
                last_recurrent_state = tree_state_all
            else:
                if (
                    _fr10_tree_scan_expected
                    and os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1"
                ):
                    raise RuntimeError(
                        "FR10 tree scan disengaged: eligible_tree_spec_row_flat_fallback"
                    )
                if os.environ.get("FR12_SUBKERNEL_CAPTURE"):
                    try:
                        if num_accepted_tokens is None:
                            _fr12_native_h0_col = torch.zeros(
                                (int(attn_metadata.num_spec_decodes), 1),
                                dtype=torch.long,
                                device=spec_state_indices_tensor.device,
                            )
                        else:
                            _fr12_native_h0_col = torch.clamp(
                                num_accepted_tokens[
                                    : attn_metadata.num_spec_decodes
                                ].to(torch.long)
                                - 1,
                                min=0,
                                max=int(spec_state_indices_tensor.size(-1)) - 1,
                            ).view(-1, 1)
                        _fr12_native_h0_rows = spec_state_indices_tensor[
                            : attn_metadata.num_spec_decodes
                        ].gather(1, _fr12_native_h0_col).reshape(-1).to(torch.long)
                        _fr12_subkernel_capture_tensor(
                            self,
                            "h0_state_in",
                            torch.index_select(ssm_state, 0, _fr12_native_h0_rows),
                            create=False,
                            extra={
                                "num_actual_tokens": int(num_actual_tokens),
                                "h0_rows": [
                                    int(_x)
                                    for _x in _fr12_native_h0_rows.detach()
                                    .cpu()
                                    .tolist()
                                ],
                                "h0_cols": [
                                    int(_x)
                                    for _x in _fr12_native_h0_col.reshape(-1)
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ],
                            },
                        )
                    except Exception as _fr12_native_h0_cap_exc:
                        logger.warning(
                            "FR12 native h0 capture failed: %s",
                            _fr12_native_h0_cap_exc,
                        )
                core_attn_out_spec, last_recurrent_state = (
                    fused_sigmoid_gating_delta_rule_update(
                        A_log=self.A_log,
                        a=a,
                        b=b,
                        dt_bias=self.dt_bias,
                        q=query_spec,
                        k=key_spec,
                        v=value_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                            : attn_metadata.num_spec_decodes
                            + 1  # type: ignore[attr-defined]
                        ],
                        ssm_state_indices=spec_state_indices_tensor,
                        num_accepted_tokens=num_accepted_tokens,
                        use_qk_l2norm_in_kernel=True,
                    )
                )
            # FR13_DECODE_GDN_CAPTURE (default-OFF, eager-only, throttled): per-
            # decode-step full SSM recurrent end-state + conv rows + core_out for
            # all co-resident tree rows. env unset -> skipped -> byte-identical.
            try:
                _fr13_dc_path = ''
                if (
                    _fr13_dc_path
                    and not (
                        torch.cuda.is_available()
                        and torch.cuda.is_current_stream_capturing()
                    )
                    and spec_state_indices_tensor is not None
                    and int(getattr(attn_metadata, "num_spec_decodes", 0)) > 0
                ):
                    _fr13_dc_prefix = str(self.prefix)
                    _fr13_dc_want = os.environ.get(
                        "FR13_DECODE_GDN_CAPTURE_LAYER_PREFIX", "*"
                    )
                    _fr13_dc_wanted = {
                        _x.strip() for _x in _fr13_dc_want.split(",") if _x.strip()
                    }
                    _fr13_dc_ok = (
                        "*" in _fr13_dc_wanted
                        or not _fr13_dc_wanted
                        or _fr13_dc_prefix in _fr13_dc_wanted
                    )
                    if _fr13_dc_ok:
                        _fr13_dc_steps = globals().setdefault(
                            "_FR13_DECODE_GDN_STEP_BY_PREFIX", {}
                        )
                        _fr13_dc_step = int(_fr13_dc_steps.get(_fr13_dc_prefix, 0))
                        _fr13_dc_steps[_fr13_dc_prefix] = _fr13_dc_step + 1
                        _fr13_dc_lo = int(
                            os.environ.get("FR13_DECODE_GDN_CAPTURE_STEP_LO", "0")
                        )
                        _fr13_dc_hi = int(
                            os.environ.get(
                                "FR13_DECODE_GDN_CAPTURE_STEP_HI", "1000000000000"
                            )
                        )
                        _fr13_dc_saved_by = globals().setdefault(
                            "_FR13_DECODE_GDN_SAVED_BY_PREFIX", {}
                        )
                        _fr13_dc_lim = int(
                            os.environ.get(
                                "FR13_DECODE_GDN_CAPTURE_LIMIT_PER_PREFIX", "8"
                            )
                        )
                        if (
                            _fr13_dc_lo <= _fr13_dc_step <= _fr13_dc_hi
                            and int(_fr13_dc_saved_by.get(_fr13_dc_prefix, 0))
                            < _fr13_dc_lim
                        ):
                            _fr13_dc_rows = torch.unique(
                                spec_state_indices_tensor[
                                    : int(attn_metadata.num_spec_decodes)
                                ]
                                .reshape(-1)
                                .to(torch.long)
                            )
                            _fr13_dc_root, _fr13_dc_ext = os.path.splitext(
                                _fr13_dc_path
                            )
                            _fr13_dc_call = (
                                _fr13_dc_root
                                + "."
                                + _fr13_dc_prefix.replace(".", "_")
                                + ".step"
                                + str(_fr13_dc_step)
                                + (_fr13_dc_ext or ".pt")
                            )
                            _fr13_dc_payload = {
                                "schema": "fr13.decode_gdn_capture.v1",
                                "layer_prefix": _fr13_dc_prefix,
                                "decode_step": int(_fr13_dc_step),
                                "num_spec_decodes": int(
                                    attn_metadata.num_spec_decodes
                                ),
                                "num_accepted_tokens": (
                                    None
                                    if num_accepted_tokens is None
                                    else num_accepted_tokens.detach().cpu().clone()
                                ),
                                "spec_state_indices": spec_state_indices_tensor.detach()
                                .cpu()
                                .clone(),
                                "coresident_rows": _fr13_dc_rows.detach().cpu().clone(),
                                "last_recurrent_state": last_recurrent_state.detach()
                                .to(torch.float32)
                                .cpu()
                                .clone(),
                                "conv_state_rows": torch.index_select(
                                    conv_state, 0, _fr13_dc_rows
                                )
                                .detach()
                                .to(torch.float32)
                                .cpu()
                                .clone(),
                                "core_out_spec": core_attn_out_spec.detach()
                                .to(torch.float32)
                                .cpu()
                                .clone(),
                            }
                            _fr13_dc_par = os.path.dirname(_fr13_dc_call)
                            if _fr13_dc_par:
                                os.makedirs(_fr13_dc_par, exist_ok=True)
                            torch.save(_fr13_dc_payload, _fr13_dc_call)
                            _fr13_dc_saved_by[_fr13_dc_prefix] = (
                                int(_fr13_dc_saved_by.get(_fr13_dc_prefix, 0)) + 1
                            )
            except Exception as _fr13_dc_exc:
                try:
                    logger.warning(
                        "FR13 decode GDN capture failed: %s", _fr13_dc_exc
                    )
                except Exception:
                    pass
            if os.environ.get("FR12_SUBKERNEL_CAPTURE"):
                try:
                    _fr12_scan_extra = {
                        "num_spec_decodes": int(attn_metadata.num_spec_decodes),
                        "num_actual_tokens": int(num_actual_tokens),
                        "tree_scan_active": bool(use_fr10_tree),
                        "tree_scan_expected": bool(_fr10_tree_scan_expected),
                    }
                    if getattr(attn_metadata, "fr10_tree_parent", None) is not None:
                        _fr12_scan_extra["tree_parent"] = [
                            int(_x)
                            for _x in attn_metadata.fr10_tree_parent.detach().cpu().tolist()
                        ]
                    if spec_token_indx is not None:
                        _fr12_scan_extra["spec_token_indx"] = [
                            int(_x) for _x in spec_token_indx.detach().cpu().tolist()
                        ]
                    _fr12_subkernel_capture_tensor(
                        self,
                        "gdn_scan_out",
                        core_attn_out_spec.squeeze(0),
                        create=False,
                        extra=_fr12_scan_extra,
                    )
                except Exception as _fr12_scan_cap_exc:
                    logger.warning("FR12 scan capture failed: %s", _fr12_scan_cap_exc)
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            assert non_spec_state_indices_tensor is not None
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()  # type: ignore[index]
            assert has_initial_state is not None
            initial_state[~has_initial_state, ...] = 0  # type: ignore[operator]
            # FR13_APC_HIT_RECURRENT_SUFFIX (default 0 = inert / byte-identical).
            # When enabled AND there is at least one APC cache-hit prefill row
            # (has_initial_state True), recompute the first <=64-token suffix
            # chunk of each cache-hit row with the bit-exact sequential rank-1
            # native kernel (fused_sigmoid_gating_delta_rule_update) seeded by
            # the restored boundary state, eliminating the chunked-prefill
            # restart-fold artifact that drives the cache-ON clear-margin
            # residual. Fresh rows (has_initial_state False) and, when the flag
            # is off, ALL rows take the native chunk path UNCHANGED below.
            _fr13_apc_active = bool(
                os.environ.get("FR13_APC_HIT_RECURRENT_SUFFIX", "0") == "1"
                and has_initial_state is not None
                and bool(has_initial_state.any())
            )
            if _fr13_apc_active:
                # ---- gated bounded recurrent-suffix recompute ----
                _fr13_qsl = non_spec_query_start_loc
                _fr13_Nns = int(_fr13_qsl.numel()) - 1
                _fr13_dev = query_non_spec.device
                # query/key are [1, T, H, K]; value is [1, T, HV, V]; a/b are
                # [T, HV] RAW gating; squeeze the leading batch dim for gather.
                _fr13_q = query_non_spec.squeeze(0)
                _fr13_k = key_non_spec.squeeze(0)
                _fr13_v = value_non_spec.squeeze(0)
                _fr13_a = a_non_spec
                _fr13_b = b_non_spec
                # initial_state is [Nns, HV, V, K] fp32 (ssm_state slice; HV is
                # the value-head count, the same index used by the kernel h0
                # read p_h0 = h0 + bos*HV*V*K). Row order == cu_seqlens segments.
                # 1) first-chunk gather over ONLY cache-hit rows, L_r=min(FC,len)
                # FR13_APC_HIT_SUFFIX_CAP overrides the recurrent recompute cap
                # (default 64). Set it large (e.g. 1000000) to recompute the WHOLE
                # cache-hit suffix RECURRENTLY -- the UNCAPPED recurrent reprefill
                # fix the value-vs-oracle probe indicates: every hit-row token is
                # rolled by the bit-exact sequential rank-1 kernel, so NO chunked
                # WY touches the cached boundary state (tail re-chunk below is
                # skipped because suffix_len <= FC). Bigger cap = more recurrent
                # compute on the (rare) re-prefill step; pure-decode steps are
                # untouched. Read per-call so an A/B can flip it without a rebuild.
                _fr13_FC = int(os.environ.get("FR13_APC_HIT_SUFFIX_CAP") or "64")
                if _fr13_FC < 1:
                    _fr13_FC = 64
                _fr13_gather = []
                _fr13_sub_cu = [0]
                _fr13_hit_rows = []
                _fr13_tail_rows = []      # hit rows with suffix_len > 64
                _fr13_suffix_end = []     # per hit row, local end offset s_{r+1}
                for _fr13_r in range(_fr13_Nns):
                    if not bool(has_initial_state[_fr13_r]):
                        continue
                    _fr13_s = int(_fr13_qsl[_fr13_r])
                    _fr13_e = int(_fr13_qsl[_fr13_r + 1])
                    _fr13_L = min(_fr13_FC, _fr13_e - _fr13_s)
                    _fr13_gather.extend(range(_fr13_s, _fr13_s + _fr13_L))
                    _fr13_sub_cu.append(_fr13_sub_cu[-1] + _fr13_L)
                    _fr13_hit_rows.append(_fr13_r)
                    _fr13_suffix_end.append(_fr13_e)
                    if (_fr13_e - _fr13_s) > _fr13_FC:
                        _fr13_tail_rows.append(_fr13_r)
                _fr13_gi = torch.as_tensor(_fr13_gather, dtype=torch.long, device=_fr13_dev)
                _fr13_cu = torch.as_tensor(_fr13_sub_cu, dtype=torch.int32, device=_fr13_dev)
                _fr13_hit_idx = torch.as_tensor(_fr13_hit_rows, dtype=torch.long, device=_fr13_dev)
                # h0 for hit rows, contiguous [Nhit, HV, V, K] fp32.
                _fr13_h0 = initial_state[_fr13_hit_idx].contiguous()
                # 2) bit-exact sequential rank-1 recompute of the first chunks.
                #    query/key are ALREADY l2-normed (fused_post_conv_prep
                #    apply_l2norm=True) -> use_qk_l2norm_in_kernel=False (Option
                #    B); a/b are RAW so the kernel derives g/beta. scale defaults
                #    to K**-0.5 on both arms. inplace_final_state=False yields a
                #    per-token state trajectory ht[bos+i_t] in fp32.
                _fr13_o_fc, _fr13_ht = fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=_fr13_a[_fr13_gi].contiguous(),
                    b=_fr13_b[_fr13_gi].contiguous(),
                    dt_bias=self.dt_bias,
                    q=_fr13_q[_fr13_gi].unsqueeze(0).contiguous(),
                    k=_fr13_k[_fr13_gi].unsqueeze(0).contiguous(),
                    v=_fr13_v[_fr13_gi].unsqueeze(0).contiguous(),
                    initial_state=_fr13_h0,
                    inplace_final_state=False,
                    cu_seqlens=_fr13_cu,
                    ssm_state_indices=None,
                    num_accepted_tokens=None,
                    use_qk_l2norm_in_kernel=False,
                )
                # _fr13_o_fc: [sumL, HV, V] bf16 ; _fr13_ht: [sumL, HV, V, K] fp32
                # post-first-chunk recurrent state per hit row = ht[sub_cu[i+1]-1]
                _fr13_post = [
                    _fr13_ht[int(_fr13_sub_cu[_fr13_i + 1]) - 1]
                    for _fr13_i in range(len(_fr13_hit_rows))
                ]
                # 3) start from the native chunk over the WHOLE non-spec batch so
                #    fresh rows + the tails of hit rows are filled exactly as
                #    native; then OVERWRITE hit-row first-chunk outputs with the
                #    serial o, and overwrite hit-row final state from the serial
                #    path (post for suffix<=64, tail re-chunk for suffix>64).
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = self.chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=non_spec_query_start_loc,
                    chunk_indices=attn_metadata.chunk_indices,
                    chunk_offsets=attn_metadata.chunk_offsets,
                    use_qk_l2norm_in_kernel=False,
                )
                # scatter serial first-chunk outputs (bf16) into the flat output.
                core_attn_out_non_spec[0, _fr13_gi] = _fr13_o_fc.to(
                    core_attn_out_non_spec.dtype
                )
                # tail re-chunk for hit rows whose suffix_len > 64, seeded by the
                # per-row post-first-chunk fp32 state. Build a reduced varlen
                # batch over the tails [s_r+64, s_{r+1}); pass cu_seqlens only and
                # let FLA recompute chunk_indices/chunk_offsets internally.
                _fr13_final_by_row = {}
                if len(_fr13_tail_rows) > 0:
                    _fr13_tail_gather = []
                    _fr13_tail_cu = [0]
                    _fr13_tail_h0 = []
                    _fr13_tail_order = []
                    for _fr13_i, _fr13_r in enumerate(_fr13_hit_rows):
                        _fr13_e = _fr13_suffix_end[_fr13_i]
                        _fr13_s = int(_fr13_qsl[_fr13_r])
                        if (_fr13_e - _fr13_s) <= _fr13_FC:
                            # whole suffix in first chunk -> post IS the final.
                            _fr13_final_by_row[_fr13_r] = _fr13_post[_fr13_i]
                            continue
                        _fr13_tstart = _fr13_s + _fr13_FC
                        _fr13_tail_gather.extend(range(_fr13_tstart, _fr13_e))
                        _fr13_tail_cu.append(
                            _fr13_tail_cu[-1] + (_fr13_e - _fr13_tstart)
                        )
                        _fr13_tail_h0.append(_fr13_post[_fr13_i])
                        _fr13_tail_order.append(_fr13_r)
                    if len(_fr13_tail_order) > 0:
                        _fr13_tgi = torch.as_tensor(
                            _fr13_tail_gather, dtype=torch.long, device=_fr13_dev
                        )
                        _fr13_tcu = torch.as_tensor(
                            _fr13_tail_cu, dtype=torch.int32, device=_fr13_dev
                        )
                        # stack per-row post states -> [Ntail, HV, V, K] fp32.
                        _fr13_th0 = torch.stack(_fr13_tail_h0, dim=0).contiguous()
                        (
                            _fr13_tail_out,
                            _fr13_tail_final,
                        ) = self.chunk_gated_delta_rule(
                            q=_fr13_q[_fr13_tgi].unsqueeze(0).contiguous(),
                            k=_fr13_k[_fr13_tgi].unsqueeze(0).contiguous(),
                            v=_fr13_v[_fr13_tgi].unsqueeze(0).contiguous(),
                            g=g_non_spec.squeeze(0)[_fr13_tgi].unsqueeze(0).contiguous(),
                            beta=beta_non_spec.squeeze(0)[_fr13_tgi].unsqueeze(0).contiguous(),
                            initial_state=_fr13_th0,
                            output_final_state=True,
                            cu_seqlens=_fr13_tcu,
                            chunk_indices=None,
                            chunk_offsets=None,
                            use_qk_l2norm_in_kernel=False,
                        )
                        # scatter tail outputs (bf16) into the flat output.
                        core_attn_out_non_spec[0, _fr13_tgi] = (
                            _fr13_tail_out.squeeze(0).to(core_attn_out_non_spec.dtype)
                        )
                        for _fr13_j, _fr13_r in enumerate(_fr13_tail_order):
                            _fr13_final_by_row[_fr13_r] = _fr13_tail_final[_fr13_j]
                else:
                    # no tails -> every hit row's final state is its post state.
                    for _fr13_i, _fr13_r in enumerate(_fr13_hit_rows):
                        _fr13_final_by_row[_fr13_r] = _fr13_post[_fr13_i]
                # overwrite last_recurrent_state for hit rows (fresh rows keep
                # the native chunk final). last_recurrent_state is [Nns, HV, V, K]
                # row-aligned to non_spec_state_indices_tensor / cu_seqlens.
                for _fr13_r, _fr13_st in _fr13_final_by_row.items():
                    last_recurrent_state[_fr13_r] = _fr13_st.to(
                        last_recurrent_state.dtype
                    )
            else:
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = self.chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=non_spec_query_start_loc,
                    chunk_indices=attn_metadata.chunk_indices,
                    chunk_offsets=attn_metadata.chunk_offsets,
                    use_qk_l2norm_in_kernel=False,
                )
            try:
                _fr13_prefill_capture_path = os.environ.get("FR13_PREFILL_GDN_CAPTURE")
                if _fr13_prefill_capture_path:
                    _fr13_prefix = str(self.prefix)
                    _fr13_want_prefix = os.environ.get("FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX", "")
                    if _fr13_want_prefix:
                        _fr13_wanted = {
                            _x.strip()
                            for _x in _fr13_want_prefix.split(",")
                            if _x.strip()
                        }
                    else:
                        _fr13_wanted = set()
                    _fr13_prefix_ok = (
                        not _fr13_wanted
                        or "*" in _fr13_wanted
                        or _fr13_prefix in _fr13_wanted
                    )
                    _fr13_saved_by_prefix = globals().setdefault(
                        "_FR13_PREFILL_GDN_CAPTURE_SAVED_BY_PREFIX", {}
                    )
                    _fr13_limit = int(os.environ.get("FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX", "1"))
                    if (
                        _fr13_prefix_ok
                        and int(_fr13_saved_by_prefix.get(_fr13_prefix, 0)) < _fr13_limit
                        and not (
                            torch.cuda.is_available()
                            and torch.cuda.is_current_stream_capturing()
                        )
                    ):
                        _fr13_root, _fr13_ext = os.path.splitext(_fr13_prefill_capture_path)
                        _fr13_saved_total = int(
                            globals().get("_FR13_PREFILL_GDN_CAPTURE_SAVED", 0)
                        )
                        _fr13_call_path = (
                            _fr13_root
                            + ".call"
                            + str(_fr13_saved_total)
                            + (_fr13_ext or ".pt")
                        )
                        _fr13_prefill_payload = {
                            "schema": "fr13.prefill_gdn_capture.v1",
                            "path": _fr13_prefill_capture_path,
                            "call_path": _fr13_call_path,
                            "capture_saved_index": int(_fr13_saved_total),
                            "layer_prefix": _fr13_prefix,
                            "layer_name": str(self.prefix),
                            "num_actual_tokens": int(num_actual_tokens),
                            "num_prefills": int(attn_metadata.num_prefills),
                            "num_decodes": int(attn_metadata.num_decodes),
                            "num_spec_decodes": int(attn_metadata.num_spec_decodes),
                            "state_indices": (
                                None
                                if non_spec_state_indices_tensor is None
                                else non_spec_state_indices_tensor.detach().cpu().clone()
                            ),
                            "query_start_loc": (
                                None
                                if non_spec_query_start_loc is None
                                else non_spec_query_start_loc.detach().cpu().clone()
                            ),
                            "has_initial_state": (
                                None
                                if has_initial_state is None
                                else has_initial_state.detach().cpu().clone()
                            ),
                            "chunk_indices": (
                                None
                                if getattr(attn_metadata, "chunk_indices", None) is None
                                else attn_metadata.chunk_indices.detach().cpu().clone()
                            ),
                            "chunk_offsets": (
                                None
                                if getattr(attn_metadata, "chunk_offsets", None) is None
                                else attn_metadata.chunk_offsets.detach().cpu().clone()
                            ),
                            "pre_conv": _fr13_prefill_pre_conv_capture.detach().cpu().clone(),
                            "conv_out": _fr13_prefill_conv_out_capture.detach().cpu().clone(),
                            # FR13_APC_CONV_RESTORE_CAPTURE: per-row RESTORED conv
                            # window seed (conv twin of "initial_state"), already on
                            # CPU and only set when the flag is ON (else None). The
                            # row order matches state_indices / has_initial_state /
                            # query_start_loc segments. "block_ids" duplicates the
                            # physical cache rows (== state_indices) so the diff can
                            # tell cached-full-block vs partial-tail by row, and
                            # "seg_starts" gives each row's aligned base position.
                            "conv_restore": _fr13_prefill_conv_restore_capture,
                            "block_ids": (
                                None
                                if non_spec_state_indices_tensor is None
                                else non_spec_state_indices_tensor.detach().cpu().clone()
                            ),
                            "seg_starts": (
                                None
                                if non_spec_query_start_loc is None
                                else non_spec_query_start_loc.detach().cpu().clone()
                            ),
                            "a_non_spec": a_non_spec.detach().cpu().clone(),
                            "b_non_spec": b_non_spec.detach().cpu().clone(),
                            "query": query_non_spec.squeeze(0).detach().cpu().clone(),
                            "key": key_non_spec.squeeze(0).detach().cpu().clone(),
                            "value": value_non_spec.squeeze(0).detach().cpu().clone(),
                            "g": g_non_spec.squeeze(0).detach().cpu().clone(),
                            "beta": beta_non_spec.squeeze(0).detach().cpu().clone(),
                            "initial_state": initial_state.detach().cpu().clone(),
                            "core_out": core_attn_out_non_spec.squeeze(0).detach().cpu().clone(),
                            "final_state": last_recurrent_state.detach().cpu().clone(),
                            "A_log": self.A_log.detach().cpu().clone(),
                            "dt_bias": self.dt_bias.detach().cpu().clone(),
                        }
                        _fr13_parent = os.path.dirname(_fr13_call_path)
                        if _fr13_parent:
                            os.makedirs(_fr13_parent, exist_ok=True)
                        torch.save(_fr13_prefill_payload, _fr13_call_path)
                        if _fr13_saved_total == 0:
                            torch.save(_fr13_prefill_payload, _fr13_prefill_capture_path)
                        _fr13_saved_by_prefix[_fr13_prefix] = (
                            int(_fr13_saved_by_prefix.get(_fr13_prefix, 0)) + 1
                        )
                        globals()["_FR13_PREFILL_GDN_CAPTURE_SAVED"] = (
                            _fr13_saved_total + 1
                        )
            except Exception as _fr13_prefill_capture_exc:
                logger.warning(
                    "FR13 prefill GDN capture failed: %s",
                    _fr13_prefill_capture_exc,
                )
            # Init cache
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype
            )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """
        Core attention computation with a packed non-spec decode fast path.
        """
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            validate_data=False,
        )
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            use_qk_l2norm_in_kernel=True,
        )
        return


def gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """
    Custom op for the core attention computation.
    Only handles the convolution + recurrent attention part.
    Input/output projections are handled outside this op.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
    )


def gdn_attention_core_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="gdn_attention_core",
    op_func=gdn_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=gdn_attention_core_fake,
)


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
