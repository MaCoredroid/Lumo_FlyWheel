#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PATCH_VERSION = "0.19.0"
DEFAULT_TARGET_ROOT = Path("/usr/local/lib/python3.12/dist-packages")
GPU_MODEL_RUNNER = Path("vllm/v1/worker/gpu_model_runner.py")
GPU_WORKER = Path("vllm/v1/worker/gpu_worker.py")
HELPER_MODULE_PATH = Path("vllm/v1/worker/fr9_isolated_forward_probe.py")

HELPER_MODULE = r'''# SPDX-License-Identifier: Apache-2.0
"""FR9 diagnostic isolated one-step forward probe.

This module is inert unless LUMO_FR9_ISOLATED_FORWARD_PROBE=1 is set. It is
diagnostic-only and exists to prove the P0 primitive: copy public recurrent/KV
state to a scratch branch, run a single target token with num_reqs=1, return
logits/state hashes, and leave public cache blocks unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker import mamba_utils

logger = init_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ProbeConfig:
    enabled: bool
    output_dir: Path
    probe_request_ids: frozenset[str]
    max_probes: int
    repeats: int
    strict: bool


class FR9IsolatedForwardProbe:
    def __init__(self, config: ProbeConfig) -> None:
        self.config = config
        self._probes_written = 0
        self._seen: set[tuple[str, int]] = set()

    @classmethod
    def disabled(cls) -> "FR9IsolatedForwardProbe":
        return cls(
            ProbeConfig(
                enabled=False,
                output_dir=Path("/tmp/lumo-fr9-isolated-forward"),
                probe_request_ids=frozenset(),
                max_probes=0,
                repeats=2,
                strict=False,
            )
        )

    @classmethod
    def from_env(cls) -> "FR9IsolatedForwardProbe":
        enabled = _env_flag("LUMO_FR9_ISOLATED_FORWARD_PROBE", "VLLM_LUMO_FR9_ISOLATED_FORWARD_PROBE")
        if not enabled:
            return cls.disabled()
        return cls(
            ProbeConfig(
                enabled=True,
                output_dir=Path(_env_value("LUMO_FR9_ISOLATED_FORWARD_DIR", "VLLM_LUMO_FR9_ISOLATED_FORWARD_DIR", default="/tmp/lumo-fr9-isolated-forward")),
                probe_request_ids=frozenset(_split_env("LUMO_FR9_ISOLATED_FORWARD_REQ_IDS", "VLLM_LUMO_FR9_ISOLATED_FORWARD_REQ_IDS")),
                max_probes=max(1, _env_int("LUMO_FR9_ISOLATED_FORWARD_MAX_PROBES", "VLLM_LUMO_FR9_ISOLATED_FORWARD_MAX_PROBES", default=8)),
                repeats=max(2, _env_int("LUMO_FR9_ISOLATED_FORWARD_REPEATS", "VLLM_LUMO_FR9_ISOLATED_FORWARD_REPEATS", default=2)),
                strict=_env_flag("LUMO_FR9_ISOLATED_FORWARD_STRICT", "VLLM_LUMO_FR9_ISOLATED_FORWARD_STRICT"),
            )
        )

    def maybe_run(self, *, runner: Any, scheduler_output: SchedulerOutput, logits: torch.Tensor | None, sampler_output: Any) -> None:
        if not self.config.enabled or logits is None:
            return
        if self._probes_written >= self.config.max_probes:
            return
        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            req_ids = list(runner.input_batch.req_ids[: runner.input_batch.num_reqs])
            sampled = getattr(sampler_output, "sampled_token_ids", None)
            if sampled is None:
                return
            for req_index, req_id in enumerate(req_ids):
                if self._probes_written >= self.config.max_probes:
                    break
                if self.config.probe_request_ids and req_id not in self.config.probe_request_ids:
                    continue
                req_state = runner.requests.get(req_id)
                if req_state is None:
                    continue
                generated_index = len(req_state.output_token_ids)
                key = (req_id, generated_index)
                if key in self._seen:
                    continue
                self._seen.add(key)
                token_id = _sampled_token_at(sampled, req_index)
                if token_id is None:
                    continue
                payload = run_runner_probe(
                    runner,
                    req_id=req_id,
                    target_token_id=token_id,
                    repeats=self.config.repeats,
                    source="auto_after_sample",
                )
                path = self.config.output_dir / f"probe_req_{_safe_name(req_id)}_tok_{generated_index:06d}.json"
                _write_json(path, payload)
                self._probes_written += 1
        except Exception:
            logger.exception("FR9 isolated forward probe failed")
            if self.config.strict:
                raise


def run_worker_probe(worker: Any, payload_json: str = "{}") -> dict[str, Any]:
    if not _env_flag("LUMO_FR9_ISOLATED_FORWARD_PROBE", "VLLM_LUMO_FR9_ISOLATED_FORWARD_PROBE"):
        return {"ok": False, "reason": "probe_disabled"}
    try:
        payload = json.loads(payload_json or "{}")
    except json.JSONDecodeError as exc:
        return {"ok": False, "reason": f"invalid_json:{exc}"}
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return {"ok": False, "reason": "model_runner_missing"}
    req_id = payload.get("req_id")
    req_ids = list(runner.input_batch.req_ids[: runner.input_batch.num_reqs])
    if req_id is None:
        if not req_ids:
            return {"ok": False, "reason": "no_active_requests"}
        req_id = req_ids[0]
    req_state = runner.requests.get(req_id)
    if req_state is None:
        return {"ok": False, "reason": "request_missing", "req_id": req_id}
    if "target_token_id" in payload:
        target_token_id = int(payload["target_token_id"])
    elif req_state.output_token_ids:
        target_token_id = int(req_state.output_token_ids[-1])
    else:
        return {"ok": False, "reason": "target_token_missing", "req_id": req_id}
    return run_runner_probe(
        runner,
        req_id=str(req_id),
        target_token_id=target_token_id,
        repeats=int(payload.get("repeats", 2)),
        source="collective_rpc",
    )


def run_runner_probe(
    runner: Any,
    *,
    req_id: str,
    target_token_id: int,
    repeats: int,
    source: str,
) -> dict[str, Any]:
    if getattr(runner, "execute_model_state", None) is not None:
        return {"ok": False, "reason": "runner_busy_execute_model_state"}
    source_req = runner.requests.get(req_id)
    if source_req is None:
        return {"ok": False, "reason": "request_missing", "req_id": req_id}
    if source_req.num_tokens < 2:
        return {"ok": False, "reason": "request_too_short", "req_id": req_id}

    public_hash_before = _hash_public_current_blocks(runner, source_req)
    results = []
    for repeat_index in range(max(2, repeats)):
        results.append(
            _run_isolated_once(
                runner,
                source_req=source_req,
                target_token_id=int(target_token_id),
                repeat_index=repeat_index,
            )
        )
    public_hash_after = _hash_public_current_blocks(runner, source_req)

    logits_hashes = [r.get("logits_sha256") for r in results]
    state_hashes = [r.get("state_sha256") for r in results]
    ok_runs = all(r.get("ok") for r in results)
    return {
        "ok": bool(ok_runs),
        "source": source,
        "req_id": req_id,
        "target_token_id": int(target_token_id),
        "num_public_reqs_at_probe": int(runner.input_batch.num_reqs),
        "isolated_num_reqs": 1,
        "repeats": results,
        "bit_reproducible_logits": ok_runs and len(set(logits_hashes)) == 1,
        "bit_reproducible_state": ok_runs and len(set(state_hashes)) == 1,
        "public_cache_unchanged": public_hash_before == public_hash_after,
        "public_cache_before_sha256": public_hash_before,
        "public_cache_after_sha256": public_hash_after,
        "timestamp_unix": time.time(),
    }


def _run_isolated_once(
    runner: Any,
    *,
    source_req: CachedRequestState,
    target_token_id: int,
    repeat_index: int,
) -> dict[str, Any]:
    hidden_id = f"{source_req.req_id}::fr9_iso::{repeat_index}"
    saved = _SavedRunnerState.capture(runner)
    try:
        num_computed = source_req.num_tokens - 1
        hidden_req, scratch = _make_hidden_request_and_scratch(
            runner,
            source_req=source_req,
            hidden_id=hidden_id,
            num_computed=num_computed,
        )
        _copy_public_blocks_to_scratch(runner, scratch)

        iso_batch = _make_input_batch_like(runner, max_num_reqs=1)
        iso_batch.add_request(hidden_req)
        iso_batch.refresh_metadata()

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[hidden_id],
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[None],
                num_computed_tokens=[num_computed],
                num_output_tokens=[len(hidden_req.output_token_ids)],
            ),
            num_scheduled_tokens={hidden_id: 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0 for _ in runner.kv_cache_config.kv_cache_groups],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

        runner.input_batch = iso_batch
        runner.requests = {hidden_id: hidden_req}
        runner.mamba_state_idx = {}

        if runner.cache_config.mamba_cache_mode == "align":
            mamba_utils.preprocess_mamba(
                scheduler_output,
                runner.kv_cache_config,
                runner.cache_config,
                runner.mamba_state_idx,
                runner.input_batch,
                runner.requests,
                runner.compilation_config.static_forward_context,
                runner.model.get_mamba_state_copy_func(),
                runner._get_mamba_copy_bufs(),
            )
            runner.num_accepted_tokens.np[:1] = runner.input_batch.num_accepted_tokens_cpu[:1]
            runner.num_accepted_tokens.copy_to_gpu(1)

        num_scheduled_tokens_np = np.array([1], dtype=np.int32)
        logits_indices, _spec_decode_metadata = runner._prepare_inputs(
            scheduler_output,
            num_scheduled_tokens_np,
        )
        slot_mappings_by_group, slot_mappings = runner._get_slot_mappings(
            num_tokens_padded=1,
            num_reqs_padded=1,
            num_tokens_unpadded=1,
            ubatch_slices=None,
        )
        attn_metadata, _ = runner._build_attention_metadata(
            num_tokens=1,
            num_tokens_padded=None,
            num_reqs=1,
            num_reqs_padded=None,
            max_query_len=1,
            ubatch_slices=None,
            logits_indices=logits_indices,
            use_spec_decode=False,
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
            cascade_attn_prefix_lens=None,
            slot_mappings=slot_mappings_by_group,
        )
        input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, _ = runner._preprocess(
            scheduler_output,
            1,
            None,
        )
        with set_forward_context(
            attn_metadata,
            runner.vllm_config,
            num_tokens=1,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=slot_mappings,
            skip_compiled=True,
        ):
            model_output = runner._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        hidden_states = model_output[0] if runner.use_aux_hidden_state_outputs else model_output
        sample_hidden_states = hidden_states[logits_indices]
        logits = runner.model.compute_logits(sample_hidden_states)
        torch.cuda.synchronize()
        return {
            "ok": True,
            "repeat_index": repeat_index,
            "hidden_req_id": hidden_id,
            "target_token_id": int(target_token_id),
            "num_computed": int(num_computed),
            "logits_sha256": _tensor_sha256(logits),
            "state_sha256": _hash_scratch_blocks(runner, scratch),
            "scratch": scratch,
        }
    except Exception as exc:
        logger.exception("FR9 isolated one-step forward failed")
        return {
            "ok": False,
            "repeat_index": repeat_index,
            "reason": f"{type(exc).__name__}:{exc}",
        }
    finally:
        saved.restore(runner)


@dataclass
class _SavedRunnerState:
    input_batch: Any
    requests: dict[str, CachedRequestState]
    mamba_state_idx: dict[str, int]
    execute_model_state: Any
    kv_connector_output: Any

    @classmethod
    def capture(cls, runner: Any) -> "_SavedRunnerState":
        return cls(
            input_batch=runner.input_batch,
            requests=runner.requests,
            mamba_state_idx=dict(runner.mamba_state_idx),
            execute_model_state=runner.execute_model_state,
            kv_connector_output=runner.kv_connector_output,
        )

    def restore(self, runner: Any) -> None:
        runner.input_batch = self.input_batch
        runner.requests = self.requests
        runner.mamba_state_idx = self.mamba_state_idx
        runner.execute_model_state = self.execute_model_state
        runner.kv_connector_output = self.kv_connector_output


def _make_hidden_request_and_scratch(
    runner: Any,
    *,
    source_req: CachedRequestState,
    hidden_id: str,
    num_computed: int,
) -> tuple[CachedRequestState, list[dict[str, int]]]:
    output_token_ids = list(source_req.output_token_ids)
    block_ids = [list(group_ids) for group_ids in source_req.block_ids]
    scratch: list[dict[str, int]] = []
    for gid, kv_group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec = kv_group.kv_cache_spec
        block_size = getattr(spec, "block_size", None)
        if block_size is None or isinstance(spec, EncoderOnlyAttentionSpec):
            continue
        cur_idx = int(num_computed // block_size)
        ids = block_ids[gid]
        if cur_idx >= len(ids):
            raise RuntimeError(f"current block index {cur_idx} out of range for group {gid}")
        public_block_id = int(ids[cur_idx])
        scratch_idx = len(ids) - 1
        untracked_scratch = False
        if scratch_idx > cur_idx:
            scratch_block_id = int(ids[scratch_idx])
            if public_block_id == scratch_block_id:
                raise RuntimeError(f"scratch block aliases public block for group {gid}")
        else:
            if not _env_flag(
                "LUMO_FR9_ISOLATED_FORWARD_UNTRACKED_SCRATCH",
                "VLLM_LUMO_FR9_ISOLATED_FORWARD_UNTRACKED_SCRATCH",
            ):
                raise RuntimeError(
                    f"no tracked scratch block for group {gid}; len={len(ids)} "
                    f"current={cur_idx}. Runner-local untracked scratch is disabled "
                    "because it is diagnostic-only and can violate vLLM block-pool "
                    "ownership."
                )
            scratch_block_id = _choose_untracked_scratch_block(
                runner,
                group_id=gid,
                public_block_id=public_block_id,
            )
            untracked_scratch = True
        ids[cur_idx] = scratch_block_id
        scratch.append(
            {
                "group_id": int(gid),
                "current_index": int(cur_idx),
                "public_block_id": public_block_id,
                "scratch_block_id": scratch_block_id,
                "untracked_scratch": bool(untracked_scratch),
            }
        )
    if not scratch:
        raise RuntimeError("no scratch-capable KV/Mamba groups found")
    return (
        CachedRequestState(
            req_id=hidden_id,
            prompt_token_ids=copy.copy(source_req.prompt_token_ids),
            mm_features=list(source_req.mm_features),
            sampling_params=source_req.sampling_params,
            generator=None,
            block_ids=tuple(block_ids),
            num_computed_tokens=int(num_computed),
            output_token_ids=output_token_ids,
            mrope_positions=source_req.mrope_positions,
            mrope_position_delta=source_req.mrope_position_delta,
            xdrope_positions=source_req.xdrope_positions,
            lora_request=source_req.lora_request,
            prompt_embeds=source_req.prompt_embeds,
            pooling_params=source_req.pooling_params,
        ),
        scratch,
    )


def _choose_untracked_scratch_block(
    runner: Any,
    *,
    group_id: int,
    public_block_id: int,
) -> int:
    used: set[int] = set()
    for req_state in runner.requests.values():
        try:
            used.update(int(block_id) for block_id in req_state.block_ids[group_id])
        except Exception:
            continue
    used.add(int(public_block_id))
    num_blocks = _num_cache_blocks_for_group(runner, group_id)
    for block_id in range(num_blocks - 1, 0, -1):
        if block_id not in used:
            return int(block_id)
    raise RuntimeError(
        f"no runner-local untracked scratch block for group {group_id}; "
        f"num_blocks={num_blocks} active_used={len(used)}"
    )


def _num_cache_blocks_for_group(runner: Any, group_id: int) -> int:
    forward_context = runner.compilation_config.static_forward_context
    for layer_name in runner.kv_cache_config.kv_cache_groups[group_id].layer_names:
        attention = forward_context.get(layer_name)
        kv_caches = getattr(attention, "kv_cache", None)
        if kv_caches is None:
            continue
        for state in _iter_kv_cache_tensors(kv_caches):
            return _num_blocks_in_cache_tensor(state)
    raise RuntimeError(f"no cache tensor found for group {group_id}")


def _make_input_batch_like(runner: Any, *, max_num_reqs: int) -> InputBatch:
    block_sizes = [
        getattr(group.kv_cache_spec, "block_size", runner.cache_config.block_size)
        for group in runner.kv_cache_config.kv_cache_groups
    ]
    kernel_block_sizes = getattr(runner, "_kernel_block_sizes", block_sizes)
    return InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=max(runner.max_model_len, runner.max_encoder_len),
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=block_sizes,
        kernel_block_sizes=kernel_block_sizes,
        is_spec_decode=False,
        logitsprocs=runner.input_batch.logitsprocs,
        logitsprocs_need_output_token_ids=runner.input_batch.logitsprocs_need_output_token_ids,
        is_pooling_model=runner.is_pooling_model,
        cp_kv_cache_interleave_size=runner.parallel_config.cp_kv_cache_interleave_size,
    )


def _copy_public_blocks_to_scratch(runner: Any, scratch: list[dict[str, int]]) -> None:
    forward_context = runner.compilation_config.static_forward_context
    for item in scratch:
        gid = item["group_id"]
        public_block_id = item["public_block_id"]
        scratch_block_id = item["scratch_block_id"]
        layer_names = runner.kv_cache_config.kv_cache_groups[gid].layer_names
        for layer_name in layer_names:
            attention = forward_context.get(layer_name)
            kv_caches = getattr(attention, "kv_cache", None)
            if kv_caches is None:
                continue
            for state in _iter_kv_cache_tensors(kv_caches):
                _cache_block_view(state, scratch_block_id).copy_(
                    _cache_block_view(state, public_block_id),
                    non_blocking=False,
                )
    torch.cuda.synchronize()


def _hash_public_current_blocks(runner: Any, req_state: CachedRequestState) -> str:
    parts = []
    for gid, kv_group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec = kv_group.kv_cache_spec
        block_size = getattr(spec, "block_size", None)
        if block_size is None or isinstance(spec, EncoderOnlyAttentionSpec):
            continue
        num_computed = max(0, req_state.num_tokens - 1)
        idx = int(num_computed // block_size)
        if idx >= len(req_state.block_ids[gid]):
            continue
        parts.append({"group_id": gid, "scratch_block_id": int(req_state.block_ids[gid][idx])})
    return _hash_scratch_blocks(runner, parts)


def _hash_scratch_blocks(runner: Any, scratch: list[dict[str, int]]) -> str:
    h = hashlib.sha256()
    forward_context = runner.compilation_config.static_forward_context
    for item in sorted(scratch, key=lambda x: x["group_id"]):
        gid = item["group_id"]
        block_id = item["scratch_block_id"]
        h.update(f"group:{gid}:block:{block_id}".encode())
        for layer_name in runner.kv_cache_config.kv_cache_groups[gid].layer_names:
            attention = forward_context.get(layer_name)
            kv_caches = getattr(attention, "kv_cache", None)
            if kv_caches is None:
                continue
            h.update(layer_name.encode())
            for state_index, state in enumerate(_iter_kv_cache_tensors(kv_caches)):
                h.update(f"state:{state_index}:".encode())
                h.update(_tensor_bytes_for_hash(_cache_block_view(state, block_id)))
    return h.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes_for_hash(tensor)).hexdigest()


def _iter_kv_cache_tensors(kv_caches: Any):
    if torch.is_tensor(kv_caches):
        yield kv_caches
        return
    for state in kv_caches:
        yield state


def _num_blocks_in_cache_tensor(state: torch.Tensor) -> int:
    if state.dim() >= 2 and int(state.shape[0]) in (1, 2):
        return int(state.shape[1])
    return int(state.shape[0])


def _cache_block_view(state: torch.Tensor, block_id: int) -> torch.Tensor:
    block_id = int(block_id)
    if block_id < int(state.shape[0]):
        return state[block_id]
    if state.dim() >= 2 and block_id < int(state.shape[1]):
        return state[:, block_id]
    raise IndexError(
        f"block_id {block_id} out of bounds for cache tensor shape "
        f"{tuple(int(x) for x in state.shape)}"
    )


def _tensor_bytes_for_hash(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().contiguous().to("cpu")
    try:
        return cpu.numpy().tobytes()
    except TypeError:
        if cpu.dtype is torch.bfloat16:
            return cpu.view(torch.uint16).numpy().tobytes()
        return cpu.view(torch.uint8).numpy().tobytes()


def _sampled_token_at(sampled: Any, req_index: int) -> int | None:
    try:
        row = sampled[req_index]
        if isinstance(row, torch.Tensor):
            if row.numel() == 0:
                return None
            return int(row.flatten()[0].item())
        if isinstance(row, (list, tuple)):
            return int(row[0]) if row else None
        return int(row)
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def _env_flag(*names: str) -> bool:
    return _env_value(*names).strip().lower() in _TRUE_VALUES


def _env_int(*names: str, default: int) -> int:
    value = _env_value(*names, default=str(default)).strip()
    try:
        return int(value)
    except ValueError:
        return default


def _split_env(*names: str) -> list[str]:
    raw = _env_value(*names)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _safe_name(value: str) -> str:
    return _SAFE_NAME_RE.sub("_", value)[:160]
'''


@dataclass(frozen=True)
class Replacement:
    label: str
    target: Path
    before: str
    after: str


REPLACEMENTS = (
    Replacement(
        label="GPUModelRunner import FR9 probe",
        target=GPU_MODEL_RUNNER,
        before="from vllm.v1.worker.p2b_debug_export import P2BDebugExporter\n",
        after=(
            "from vllm.v1.worker.p2b_debug_export import P2BDebugExporter\n"
            "from vllm.v1.worker.fr9_isolated_forward_probe import FR9IsolatedForwardProbe\n"
        ),
    ),
    Replacement(
        label="GPUModelRunner construct FR9 probe",
        target=GPU_MODEL_RUNNER,
        before="        self.p2b_debug_exporter = P2BDebugExporter.from_env()\n",
        after=(
            "        self.p2b_debug_exporter = P2BDebugExporter.from_env()\n"
            "        self.fr9_isolated_forward_probe = FR9IsolatedForwardProbe.from_env()\n"
        ),
    ),
    Replacement(
        label="GPUModelRunner sample_tokens FR9 hook",
        target=GPU_MODEL_RUNNER,
        before="        self.p2b_debug_exporter.export_state_snapshots(runner=self)\n",
        after=(
            "        self.p2b_debug_exporter.export_state_snapshots(runner=self)\n"
            "        self.fr9_isolated_forward_probe.maybe_run(\n"
            "            runner=self,\n"
            "            scheduler_output=scheduler_output,\n"
            "            logits=logits,\n"
            "            sampler_output=sampler_output,\n"
            "        )\n"
        ),
    ),
    Replacement(
        label="Worker collective_rpc FR9 probe",
        target=GPU_WORKER,
        before="    def sleep(self, level: int = 1) -> None:\n",
        after=(
            "    def lumo_fr9_isolated_forward_probe(self, payload_json: str = \"{}\") -> dict[str, Any]:\n"
            "        from vllm.v1.worker.fr9_isolated_forward_probe import run_worker_probe\n"
            "\n"
            "        return run_worker_probe(self, payload_json)\n"
            "\n"
            "    def sleep(self, level: int = 1) -> None:\n"
        ),
    ),
)


def apply_patch_to_root(
    target_root: Path,
    *,
    dry_run: bool = False,
    skip_version_check: bool = False,
) -> list[str]:
    if not skip_version_check:
        _check_vllm_version()

    applied: list[str] = []
    updated: dict[Path, str] = {}
    for replacement in REPLACEMENTS:
        target = target_root / replacement.target
        if not target.is_file():
            raise FileNotFoundError(f"missing vLLM target file: {target}")
        text = updated.get(target)
        if text is None:
            text = target.read_text(encoding="utf-8")
        if replacement.after in text:
            updated[target] = text
            continue
        if replacement.before not in text:
            raise RuntimeError(f"{replacement.label} anchor not found in {target}")
        updated[target] = text.replace(replacement.before, replacement.after, 1)
        applied.append(replacement.label)

    if not dry_run:
        for target, text in updated.items():
            target.write_text(text, encoding="utf-8")
        helper_module = target_root / HELPER_MODULE_PATH
        helper_module.write_text(HELPER_MODULE, encoding="utf-8")
    return applied


def _check_vllm_version() -> None:
    try:
        actual = version("vllm")
    except PackageNotFoundError as exc:
        raise RuntimeError("vLLM is not installed in the target Python environment") from exc
    public_actual = actual.split("+", 1)[0]
    if public_actual != PATCH_VERSION:
        raise RuntimeError(
            f"FR9 isolated forward patch targets vLLM {PATCH_VERSION}, found {actual}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the FR9 isolated forward diagnostic probe.")
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-version-check", action="store_true")
    args = parser.parse_args(argv)

    applied = apply_patch_to_root(
        args.target_root,
        dry_run=args.dry_run,
        skip_version_check=args.skip_version_check,
    )
    action = "would apply" if args.dry_run else "applied"
    print(f"{action} {len(applied)} FR9 isolated forward probe edits")
    for label in applied:
        print(f"- {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
