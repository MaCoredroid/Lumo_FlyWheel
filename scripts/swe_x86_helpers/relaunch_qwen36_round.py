#!/usr/bin/env python3
"""Parameterized relaunch for the Round-5 B=4 sweep: one script for all rounds.

  --config D            full T1+T2+T3+T4 suffix stack (the original config D)
  --config E --mtp N    Qwen3.6 native MTP head, num_speculative_tokens=N (no suffix)

Both variants get the per-agent spec-decode STEP TRACE patch: a source-edit of
Scheduler.make_spec_decoding_stats (called per-request per-step with request_id)
that appends {ts,rid,draft,acc} rows to /logs/per_req_spec_trace.jsonl (a
bind-mounted host path), so per-agent acceptance + per-step timing stay clean at
ANY batch size (B>1) -- the global /metrics deltas can't separate concurrent
streams, this can (rid carries the proxy's session-prefixed request id).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

REPO = Path("/home/mark/shared/lumoFlyWheel")
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
from scripts.run_track_b_loop import _track_b_runtime_prelaunch_shell
from lumo_flywheel_serving.model_server import ModelServer

_KEEP_MARKER = "applied forced tool_choice parser patch')\nPY\n"

# Source-edit (NOT monkeypatch -- prelaunch patches the file before vLLM imports it)
# of Scheduler.make_spec_decoding_stats to emit per-request per-step rows.
_SPEC_TRACE_BLOCK = r'''
python3 - <<'LUMOSPECTRACE'
from pathlib import Path
p = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py')
text = p.read_text()
sentinel = '# LUMO_PER_AGENT_SPEC_TRACE'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] per-agent spec trace already present')
else:
    anchor = '    ) -> SpecDecodingStats | None:\n        if not self.log_stats or not num_draft_tokens:'
    if anchor not in text:
        raise RuntimeError('make_spec_decoding_stats anchor not found for per-agent spec trace')
    inject = ('    ) -> SpecDecodingStats | None:\n'
              '        ' + sentinel + '\n'
              '        try:\n'
              '            import json as _lj, time as _lt, os as _lo\n'
              '            global _LUMO_SPEC_FH\n'
              '            try:\n'
              '                _LUMO_SPEC_FH\n'
              '            except NameError:\n'
              '                _LUMO_SPEC_FH = open(_lo.environ.get("LUMO_PER_REQ_SPEC_TRACE", "/logs/per_req_spec_trace.jsonl"), "a", buffering=1)\n'
              '            _LUMO_SPEC_FH.write(_lj.dumps({"ts": round(_lt.time(), 4), "rid": request_id, "draft": num_draft_tokens, "acc": num_accepted_tokens}) + "\n")\n'
              '        except Exception:\n'
              '            pass\n'
              '        if not self.log_stats or not num_draft_tokens:')
    text = text.replace(anchor, inject, 1)
    p.write_text(text)
    print('[TRACK-B-PRELAUNCH] applied per-agent spec-decode step trace patch')
LUMOSPECTRACE
'''


def _prelaunch_for(config: str) -> str:
    full = _track_b_runtime_prelaunch_shell()
    if config == "D":
        base = full  # full T1+T2+T3+T4 stack
        if "T2_T4_COMPOSITE" not in base:
            raise RuntimeError("config D expects the full prelaunch shell")
    else:  # E -- KEEP prefix only (no suffix T-patches; MTP doesn't use them)
        idx = full.find(_KEEP_MARKER)
        if idx < 0:
            raise RuntimeError("forced tool_choice marker not found")
        base = full[: idx + len(_KEEP_MARKER)]
        if "T1_SESSION_SCOPING" in base:
            raise RuntimeError("config E shell must not contain T1")
    return base + _SPEC_TRACE_BLOCK


def _mtp_bundle(n: int) -> str:
    src = Path("/tmp/lumo-track-b-bundle-qwen36-off/bundle.yaml").read_text()
    src = src.replace("bundle_id: 712fd011-4b16-4051-9e8c-875405b70f5b",
                      f"bundle_id: e0000000-mtp{n}-4000-9000-config-e-qwen36")
    src = src.replace("  spec_decode: {}",
                      f"  spec_decode:\n    method: qwen3_5_mtp\n    num_speculative_tokens: {n}")
    out = Path(f"/tmp/lumo-track-b-bundle-qwen36-mtp{n}"); out.mkdir(exist_ok=True)
    (out / "bundle.yaml").write_text(src)
    return str(out / "bundle.yaml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["D", "E"], required=True)
    ap.add_argument("--mtp", type=int, default=1, help="num_speculative_tokens for config E")
    args = ap.parse_args()
    bundle = ("/tmp/lumo-track-b-bundle-qwen36/bundle.yaml" if args.config == "D"
              else _mtp_bundle(args.mtp))
    server = ModelServer(
        registry_path=REPO / "model_registry.yaml", port=9950,
        container_name="lumo-vllm-track-b-suffix",
        logs_root=Path("/tmp/lumo-l0c-fp8-cutlass-run30-logs"),
        triton_cache_root=Path("/tmp/lumo-l0c-fp8-cutlass-run30-triton"),
        state_root=Path("/tmp/lumo-l0c-fp8-cutlass-run30-state"),
        proxy_port=8088, ready_timeout_s=900,
        prelaunch_shell=_prelaunch_for(args.config),
    )
    server.load_tuned_config(bundle)
    server.start("qwen3.6-27b")
    print(f"READY config={args.config} mtp={args.mtp if args.config=='E' else '-'} bundle={bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
