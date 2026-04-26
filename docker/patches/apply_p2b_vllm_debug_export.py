#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PATCH_VERSION = "0.19.0"
DEFAULT_TARGET_ROOT = Path("/usr/local/lib/python3.12/dist-packages")
GPU_MODEL_RUNNER = Path("vllm/v1/worker/gpu_model_runner.py")
HELPER_MODULE_PATH = Path("vllm/v1/worker/p2b_debug_export.py")
DEBUG_ENV_VARS = (
    "LUMO_P2B_VLLM_DEBUG_EXPORT",
    "LUMO_P2B_DEBUG_EXPORT_DIR",
    "LUMO_P2B_DEBUG_PROBE_REQUEST_IDS",
    "LUMO_P2B_DEBUG_STATE_TOKENS",
    "LUMO_P2B_DEBUG_STRICT",
)

HELPER_MODULE = r'''# SPDX-License-Identifier: Apache-2.0
"""Repo-owned P2b vLLM debug export hooks.

This module is intentionally inert unless LUMO_P2B_VLLM_DEBUG_EXPORT is set
and LUMO_P2B_DEBUG_PROBE_REQUEST_IDS names one or more request ids.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


_TRUE_VALUES = {"1", "true", "yes", "on"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class P2BDebugConfig:
    enabled: bool
    output_dir: Path
    probe_request_ids: frozenset[str]
    state_tokens: frozenset[int]
    strict: bool = False


class P2BDebugExporter:
    def __init__(self, config: P2BDebugConfig) -> None:
        self.config = config
        self._exported_logits_count: dict[str, int] = {}
        self._exported_state: set[tuple[str, int]] = set()

    @classmethod
    def disabled(cls) -> "P2BDebugExporter":
        return cls(
            P2BDebugConfig(
                enabled=False,
                output_dir=Path("/tmp/lumo-p2b-vllm-debug"),
                probe_request_ids=frozenset(),
                state_tokens=frozenset({1, 1024}),
            )
        )

    @classmethod
    def from_env(cls) -> "P2BDebugExporter":
        probe_request_ids = frozenset(
            value
            for value in _split_env("LUMO_P2B_DEBUG_PROBE_REQUEST_IDS", "VLLM_LUMO_P2B_DEBUG_PROBE_REQUEST_IDS")
            if value
        )
        enabled = (
            _env_value("LUMO_P2B_VLLM_DEBUG_EXPORT", "VLLM_LUMO_P2B_DEBUG_EXPORT").strip().lower()
            in _TRUE_VALUES
        ) or bool(probe_request_ids)
        if not enabled:
            return cls.disabled()

        if not probe_request_ids:
            logger.warning(
                "LUMO_P2B_VLLM_DEBUG_EXPORT is set but "
                "LUMO_P2B_DEBUG_PROBE_REQUEST_IDS is empty; exporter is disabled."
            )
            return cls.disabled()

        state_tokens = frozenset(
            int(value)
            for value in _split_env(
                "LUMO_P2B_DEBUG_STATE_TOKENS",
                "VLLM_LUMO_P2B_DEBUG_STATE_TOKENS",
                default="1,1024",
            )
        )
        output_dir = Path(
            _env_value(
                "LUMO_P2B_DEBUG_EXPORT_DIR",
                "VLLM_LUMO_P2B_DEBUG_EXPORT_DIR",
                default="/tmp/lumo-p2b-vllm-debug",
            )
        )
        strict = _env_value("LUMO_P2B_DEBUG_STRICT", "VLLM_LUMO_P2B_DEBUG_STRICT").strip().lower() in _TRUE_VALUES
        return cls(
            P2BDebugConfig(
                enabled=True,
                output_dir=output_dir,
                probe_request_ids=probe_request_ids,
                state_tokens=state_tokens,
                strict=strict,
            )
        )

    def export_logits(
        self,
        *,
        logits: torch.Tensor | None,
        req_ids: Iterable[str],
        output_token_ids: list[list[int]],
    ) -> None:
        if not self.config.enabled or logits is None:
            return
        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            for req_index, req_id in enumerate(req_ids):
                if not self._matches_probe_request(req_id):
                    continue
                if req_index >= logits.shape[0]:
                    continue
                generated_token_index = self._next_generated_token_index(
                    req_id=req_id,
                    req_index=req_index,
                    output_token_ids=output_token_ids,
                )
                payload = {
                    "kind": "full_vocab_logits_before_sampling",
                    "request_id": req_id,
                    "generated_token_index": generated_token_index,
                    "logits": logits[req_index].detach().to("cpu", dtype=torch.float32),
                }
                self._write_pt(
                    f"logits_req_{_safe_name(req_id)}_tok_{generated_token_index:06d}.pt",
                    payload,
                )
        except Exception:
            logger.exception("P2b debug logits export failed")
            if self.config.strict:
                raise

    def export_state_snapshots(self, *, runner: Any) -> None:
        if not self.config.enabled:
            return
        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            copy_bufs = runner._get_mamba_copy_bufs()
            kv_cache_config = runner.kv_cache_config
            forward_context = runner.compilation_config.static_forward_context
            req_ids = runner.input_batch.req_ids[: runner.input_batch.num_reqs]
            for req_id in req_ids:
                if not self._matches_probe_request(req_id):
                    continue
                req_state = runner.requests.get(req_id)
                if req_state is None:
                    continue
                generated_token_index = len(req_state.output_token_ids)
                if generated_token_index not in self.config.state_tokens:
                    continue
                exported_key = (req_id, generated_token_index)
                if exported_key in self._exported_state:
                    continue

                state_block_idx = runner.mamba_state_idx.get(req_id)
                if state_block_idx is None or state_block_idx < 0:
                    continue
                snapshots: dict[str, list[dict[str, Any]]] = {}
                for mamba_group_id in copy_bufs.mamba_group_ids:
                    block_ids = req_state.block_ids[mamba_group_id]
                    if state_block_idx >= len(block_ids):
                        continue
                    block_id = int(block_ids[state_block_idx])
                    layer_names = kv_cache_config.kv_cache_groups[
                        mamba_group_id
                    ].layer_names
                    for layer_name in layer_names:
                        attention = forward_context.get(layer_name)
                        kv_caches = getattr(attention, "kv_cache", None)
                        if not kv_caches:
                            continue
                        layer_payload: list[dict[str, Any]] = []
                        for state_index, state in enumerate(kv_caches):
                            if block_id >= state.shape[0]:
                                continue
                            layer_payload.append(
                                {
                                    "state_index": state_index,
                                    "block_id": block_id,
                                    "tensor": state[block_id].detach().to("cpu"),
                                }
                            )
                        if layer_payload:
                            snapshots[layer_name] = layer_payload
                if not snapshots:
                    continue

                payload = {
                    "kind": "qwen35_mamba_deltanet_recurrent_state",
                    "request_id": req_id,
                    "generated_token_index": generated_token_index,
                    "state_tokens": sorted(self.config.state_tokens),
                    "layers": snapshots,
                }
                self._write_pt(
                    f"state_req_{_safe_name(req_id)}_tok_{generated_token_index:06d}.pt",
                    payload,
                )
                self._exported_state.add(exported_key)
        except Exception:
            logger.exception("P2b debug recurrent-state export failed")
            if self.config.strict:
                raise

    def _write_pt(self, filename: str, payload: dict[str, Any]) -> None:
        final_path = self.config.output_dir / filename
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(final_path)

    def _matches_probe_request(self, req_id: str) -> bool:
        return "*" in self.config.probe_request_ids or req_id in self.config.probe_request_ids

    def _next_generated_token_index(
        self,
        *,
        req_id: str,
        req_index: int,
        output_token_ids: list[list[int]],
    ) -> int:
        if req_index < len(output_token_ids):
            generated_token_index = len(output_token_ids[req_index]) + 1
        else:
            generated_token_index = self._exported_logits_count.get(req_id, 0) + 1
        self._exported_logits_count[req_id] = generated_token_index
        return generated_token_index


def _env_value(name: str, alias: str | None = None, *, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    if alias is not None and alias in os.environ:
        return os.environ[alias]
    return default


def _split_env(name: str, alias: str | None = None, *, default: str = "") -> list[str]:
    raw = _env_value(name, alias, default=default)
    return [value.strip() for value in raw.split(",") if value.strip()]


def _safe_name(value: str) -> str:
    return _SAFE_NAME_RE.sub("_", value)[:160]
'''


@dataclass(frozen=True)
class Replacement:
    label: str
    before: str
    after: str


REPLACEMENTS = (
    Replacement(
        label="GPUModelRunner import P2BDebugExporter",
        before="from vllm.v1.worker import mamba_utils\n",
        after=(
            "from vllm.v1.worker import mamba_utils\n"
            "from vllm.v1.worker.p2b_debug_export import P2BDebugExporter\n"
        ),
    ),
    Replacement(
        label="GPUModelRunner.__init__ construct disabled-by-default exporter",
        before=(
            "        # Sampler\n"
            "        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)\n"
        ),
        after=(
            "        # Sampler\n"
            "        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)\n"
            "        self.p2b_debug_exporter = P2BDebugExporter.from_env()\n"
        ),
    ),
    Replacement(
        label="GPUModelRunner._sample logits hook before sampling",
        before=(
            "        # Update output token ids with tokens sampled in last step\n"
            "        # if async scheduling and required by current sampling params.\n"
            "        self.input_batch.update_async_output_token_ids()\n"
            "        if spec_decode_metadata is None:\n"
        ),
        after=(
            "        # Update output token ids with tokens sampled in last step\n"
            "        # if async scheduling and required by current sampling params.\n"
            "        self.input_batch.update_async_output_token_ids()\n"
            "        self.p2b_debug_exporter.export_logits(\n"
            "            logits=logits,\n"
            "            req_ids=self.input_batch.req_ids[: self.input_batch.num_reqs],\n"
            "            output_token_ids=sampling_metadata.output_token_ids,\n"
            "        )\n"
            "        if spec_decode_metadata is None:\n"
        ),
    ),
    Replacement(
        label="GPUModelRunner.sample_tokens state hook after generated-token update",
        before=(
            "        self._update_states_after_model_execute(\n"
            "            sampler_output.sampled_token_ids, scheduler_output\n"
            "        )\n"
        ),
        after=(
            "        self._update_states_after_model_execute(\n"
            "            sampler_output.sampled_token_ids, scheduler_output\n"
            "        )\n"
            "        self.p2b_debug_exporter.export_state_snapshots(runner=self)\n"
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

    gpu_model_runner = target_root / GPU_MODEL_RUNNER
    if not gpu_model_runner.is_file():
        raise FileNotFoundError(f"missing vLLM target file: {gpu_model_runner}")

    text = gpu_model_runner.read_text(encoding="utf-8")
    applied: list[str] = []
    for replacement in REPLACEMENTS:
        if replacement.after in text:
            continue
        if replacement.before not in text:
            raise RuntimeError(
                f"{replacement.label} anchor not found in {gpu_model_runner}"
            )
        text = text.replace(replacement.before, replacement.after, 1)
        applied.append(replacement.label)

    if not dry_run:
        gpu_model_runner.write_text(text, encoding="utf-8")
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
            f"P2b debug export patch targets vLLM {PATCH_VERSION}, found {actual}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the repo-owned P2b vLLM debug export scaffold.")
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
    print(f"{action} {len(applied)} P2b vLLM debug export edits")
    for label in applied:
        print(f"- {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
