# CNB-55 v4a run1 — config record

**STOPPED BY MARK on 2026-05-24** at **5/11 complete** (stop called during family 6,
release-note-to-plan-translation, which was killed mid-run and is NOT a valid result).
Moving to next step. This file records the exact config used so the run is reproducible.

## What ran
- **Benchmark**: CNB-55 **v4a active set** (11 families; reference runner = `codex exec`).
- **Completed (5)**: dead-flag-reachability-audit, fanout-fullstack-release-blocker,
  incident-evidence-synthesis, multi-tool-transaction-repair, policy-aware-request-resolution.
  All `status=WROTE_FILES` (codex rc=1 but produced artifacts; grading not yet run).
- **Not run (6)**: release-note-to-plan-translation (killed mid-run), responses-sdk-adapter-cutover,
  responsive-checkout-visual-regression, security-audit-hotfix-remediation,
  sqlalchemy-2-session-modernization, transcript-merge-regression.
- **Grading**: NOT performed (deferred off-box):
  `scripts/run_v4a_graders_on_validation.py --validation-root output/cnb_v4a_run1 --out-dir <off-box>`
  (needs verifiers/ + verifier_data/, no GPU).

## vLLM serving surface — config D (all-on)
- Container `lumo-vllm-track-b-suffix`, image `lumo-flywheel-vllm:26.01-py3-v0.19.0`, DGX (GB10 aarch64).
- Relaunch: `/tmp/relaunch_qwen36_AD.py` (bundle `/tmp/lumo-track-b-bundle-qwen36/bundle.yaml`,
  full T1+T2+T3+T4 prelaunch shell).
- **Spec-decode**: method=suffix, num_speculative_tokens=12, rejection_sample=probabilistic,
  max_tree_depth=32, max_cached_requests=1000, min_token_prob=0.05.
- **Prelaunch techniques applied**: T1 session scoping, T2+T4 composite drafting, T3 composite
  drafting. **CAVEAT**: T3 oracle FastAPI middleware install FAILED this boot
  ("Cannot add middleware after an application has started") and reverted to the un-instrumented
  path (defensive try/except by design; deterministic on vLLM v0.19.0 → consistent with the
  original 05-19 config-D reference runs).
- vLLM args: max_num_seqs=4, gpu_memory_utilization=0.9, max_model_len=131072, fp8,
  enable_prefix_caching=true, enable_chunked_prefill=true, served_model_name=**qwen3.6-27b**,
  reasoning_parser=qwen3, tool_call_parser=qwen3_xml.

## Proxy (codex-bench-proxy, DGX :8022 -> vLLM :9950)
- NONSTREAM_BYPASS=1, AUTO_CONTINUE=1, RETRY_UPSTREAM_400=1, MAX_OUTPUT_TOKENS=80000,
  FORCE_TEMPERATURE=1.0, FORCE_TOP_P=0.95.

## Runner (codex-on-x86)
- **Concurrency: B=1, strictly serial.**
- Host: alienware (x86), `docker codex-runner:v1` (codex-cli 0.128.0), `--network=host`,
  reaching the DGX proxy via reverse SSH tunnel (alienware:8022 -> DGX:8022, -R 9950 too).
- Runner: `~/cnb_v4a/run_cnb_v4a_x86.py` (committed copy alongside this run).
  Per family: copy `workspace_bundle/v1-clean-baseline` -> workspace, prompt from AGENTS.md
  (+ .scenario_variant), `codex exec` with model=**qwen3.6-27b**, reasoning_effort=high,
  wire_api=responses, stream_idle_timeout_ms=600000, per-task timeout 1800s.
- Output: alienware `~/cnb_v4a/output/run1` -> repo `output/cnb_v4a_run1` (raw per-task
  workspace + codex_stdout/stderr + runner_metadata.json + driver.log + summary.json).

## Infra notes
- Reused the SWE `swe_infra` tmux session (tunnel keeper + capture streamer + vLLM step-trace).
- Per-step throughput trace: `output/cnb_v4a_run1/dgx_steptrace.jsonl`.
