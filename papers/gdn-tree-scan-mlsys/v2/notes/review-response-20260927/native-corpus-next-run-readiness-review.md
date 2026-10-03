# Native corpus: aligned A next-run readiness review

2026-09-29. **No concrete launcher/readiness/schema incompatibility found in the current local source package that requires a code repair before preparing aligned_nonpacked/A.** This is a bounded source/CPU disposition, not permission to launch or evidence of current machine readiness. The parent retains the prerequisite that root instrumentation first passes and its owned process finishes cleanup.

Current identities:

| Input | SHA-256 |
|---|---|
| `tools/run_q1_native_corpus_v1.sh` | `1464009d5115c1a55addc5b3fe31c972a999366ee4add4d16d209f4f2d141243` |
| `tools/q1_native_corpus_v1.py` | `d4502b7f47ab1ac6ed8cd295aa5565726fad486c4a1ff1bda6a13fe24e4ea770` |
| `fullmodel/native-corpus-v1/CORPUS.json` | `f0320a57bdec8fb9aaf878313972ea391f936fe48a8e4ad46ddacb3375e40e44` |
| `tools/q1_readiness_consumer_v1.py` | `2230ba1dea4551b30d5efad88516a7aa6ad3487c61880b9bd0984d6ca2cc2eb9` |
| `tools/memory_recovery_v2_8.py` | `b85c13a77c9a6060d2f9b6bd221997851159f716efff8aa0c53751edfed22616` |

The manifest rebuilds identically against current sources, including the later accepted corpus reducer. The actual job builder produces 84 unique calibration cases × R2 = 168 requests, `smoke=false`, packed flag `0`, and full state/KV archival. All three calibration prefix files match their frozen sizes and SHA-256 values. The existing arm renderer validates with no problems, and local `bash -n` passes. No launch, model, tensor, Docker, GPU, query, cleanup, SSH, or gate operation was performed.

## Exact next inputs

The existing manifest fixes these values; its identifier date is not a claim about the future launch time:

- `NATIVE_ARM=aligned_nonpacked`, `NATIVE_PROCESS=A`.
- `RUN_ID=q1-native-corpus-calibration-20260929T045821Z-aligned_nonpacked-A`.
- `NATIVE_CORPUS=/home/mark/lumotree-review-20260927/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/fullmodel/native-corpus-v1/CORPUS.json`.
- `NATIVE_CORPUS_SHA=f0320a57bdec8fb9aaf878313972ea391f936fe48a8e4ad46ddacb3375e40e44`.
- `CORPUS_GATE`: the parent's exact per-run gate path; if omitted, the wrapper uses monitor `GATE-Q1-NATIVE-CORPUS.json`. Actual invocation needs no positional argument; `--dry-run` only prints and exits before validating readiness.

Wrapper lines 39–83 require `approved=true`, `gate=Q1-NATIVE-CORPUS`, that exact `approved_run_id`, and `launch_limit=1`. `reviewed_scope` must contain arm, process, repeats 2, fresh_processes 1, all 84 manifest case IDs in order, expected_requests 168, save_kv_bytes true, and block calibration. `reviewed_hashes` comprises all 16 manifest source hashes plus image_id, corpus, token_fixtures, shortest-prefix token IDs, and fork_binary. The exact 21-value map and required scope are retained in the companion audit; it deliberately is not an approved gate. `recovery_binding` must bind the current v2.8 runner SHA and an existing authority directory.

The fixed native configuration remains the accepted smoke configuration: utilization 0.6, B1, model length 131072, seed 0, spec-off, primary nonpacked recurrence, patched FA2 through its native causal interface, and align cache mode. Native request sampling remains greedy. The candidate's separate request-seed and 64→1024 hydration repair does not require changing this native job or renderer.

## Readiness and namespace requirements

The shared consumer accepts either an admitting terminal receipt for **this run ID and exact final gate hash**, or the explicit parent disposition `readiness_treatment_disposition=no_reclaim_boot_approved` with an idle authority and no matching run receipt. A matching failed receipt takes precedence and refuses; the disposition is not a bypass. Root-attempt readiness evidence cannot be reused as this native run's receipt. The wrapper calls only the verifier; it performs no implicit recovery/query. The current query/global producer's fixed 82.26 GiB threshold is stricter than the native 0.6 setting but is not a schema mismatch or an instruction to change that threshold.

Before the wrapper can proceed, its run output directory and the gate-directory file `native-corpus-binding-<RUN_ID>.json` must be absent, the exact container name must be unused, compute and running `lumotree-` containers must be absent, port 9950 must be free, and the pinned image/digest/fork must pass current host checks. The exclusive binding file is written before contention checks: a prelaunch refusal can reserve that run namespace. Preserve such evidence and handle it explicitly; do not assume an automatic retry is available. This review did not inspect live remote namespaces or machine state.

The gate itself does not parse a root-result acceptance object. The parent's decision to issue it after accepting root instrumentation is therefore a required sequencing step. Retain sole-parent serial execution; the point-in-time contention check is not a cross-run GPU lock. The accepted unknown-creation latch and exact-CID cleanup remain present and unchanged. No additional operational framework is requested.

Only aligned A is selected by this invocation. The other three declared native arm/process runs and full 672-observation reduction remain separate stages; this source review yields no candidate, continuous-cycle, MTP, timing, or workload qualification.

Companion: `native-corpus-next-run-readiness-audit.json`, SHA-256 `82f83359cd57d2836156eb11684d72c3a12982a3bf95c459f0b2513b2d519158`. It contains the exact prerequisite map and bounded CPU checks, not an operational authorization.
