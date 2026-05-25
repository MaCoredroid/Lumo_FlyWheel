#!/usr/bin/env python3
"""Relaunch driver for the Track B vLLM container -- config E variant.

Config E = Qwen 3.6 27B (dense, FP8) NATIVE MTP head, no suffix decoding.
Two deltas vs the all-off relaunch (/tmp/relaunch_qwen36_off.py):

1. Bundle: /tmp/lumo-track-b-bundle-qwen36-mtp/bundle.yaml has
   `spec_decode: {method: qwen3_5_mtp, num_speculative_tokens: 1}`.
   model_server passes that verbatim as --speculative-config, so vLLM loads
   the in-checkpoint MTP head (the `mtp.*` tensors; mtp_num_hidden_layers=1)
   via its Qwen3_5MTP class. method=qwen3_5_mtp is the ONLY vLLM MTP path
   that reads `mtp_num_hidden_layers` (every other reads num_nextn_predict_layers,
   which this checkpoint lacks) -- so it is the correct method for dense
   Qwen3.6-27B FP8 (model_type=qwen3_5_text; "3.6" is branding, family=qwen3_5).

2. Prelaunch shell: SAME as all-off -- KEEP prefix only (GPU memory hygiene,
   PR#39562 KV-allocator stop-gap, PR#39055 qwen3 reasoning-parser tool-call
   recovery, forced tool_choice parser patch). NO T1-T4 suffix patches: MTP
   does not use SuffixDecodingProposer at all.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path("/home/mark/shared/lumoFlyWheel")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from scripts.run_track_b_loop import _track_b_runtime_prelaunch_shell
from lumo_flywheel_serving.model_server import ModelServer

_MARKER = "applied forced tool_choice parser patch')\nPY\n"


def _config_e_prelaunch_shell() -> str:
    full = _track_b_runtime_prelaunch_shell()
    idx = full.find(_MARKER)
    if idx < 0:
        raise RuntimeError("forced tool_choice truncation marker not found")
    truncated = full[: idx + len(_MARKER)]
    if "T1_SESSION_SCOPING" in truncated or "T2_T4_COMPOSITE" in truncated:
        raise RuntimeError("truncation left T-technique blocks in the shell")
    for keep in ("drop_caches", "PR39562", "PR39055", "forced tool_choice"):
        if keep not in truncated:
            raise RuntimeError(f"truncation dropped a KEEP block: {keep}")
    return truncated


def main() -> int:
    server = ModelServer(
        registry_path=REPO / "model_registry.yaml",
        port=9950,
        container_name="lumo-vllm-track-b-suffix",
        logs_root=Path("/tmp/lumo-l0c-fp8-cutlass-run30-logs"),
        triton_cache_root=Path("/tmp/lumo-l0c-fp8-cutlass-run30-triton"),
        state_root=Path("/tmp/lumo-l0c-fp8-cutlass-run30-state"),
        proxy_port=8088,
        ready_timeout_s=900,
        prelaunch_shell=_config_e_prelaunch_shell(),
    )
    server.load_tuned_config("/tmp/lumo-track-b-bundle-qwen36-mtp/bundle.yaml")
    server.start("qwen3.6-27b")
    print("READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
