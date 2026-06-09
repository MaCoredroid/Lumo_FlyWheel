# FR13 canonical measurement harness — STOP hand-rolling e2e/drift comparisons

**Why this exists:** the measurement tools already exist but were (1) uncommitted (vanish on context reset), (2) un-wired (no launcher/doc invoked them), (3) un-enforced (pairing checked but never *raised*). So each codex session re-hand-rolled `python3 - <<'PY'` capture+compare glue and re-hit the **prompt-pairing bug** (native arm on a different prompt → lcp=0 artifact, the same bug that burned 3-4 WY boots). The fix is to make ONE entry point canonical, committed, and pairing-enforced.

## The canonical tools (COMMIT these; do NOT re-implement inline)
- **Capture (one arm):** `scripts/fr10_quick_decode_tps_probe.py` — records per-request `prompt_token_ids` + `token_ids` + TPS. Run once per arm (native, tree) into a run dir; writes `<arm>_request_metrics.jsonl` + the greedy probe JSON.
- **Argmax flip-layer reducer:** `scripts/fr13_argmax_lcp_localize.py --tree-run <dir> --native-run <dir> --out <json>` — has `_prompt_identity()` (compares `tree_tokens==native_tokens`, emits `byte_identical`) + `_first_layer_delta()` (first nonzero per-layer hidden = the flip layer). **MUST be committed + invoked, not re-written inline.**
- **Bag-TV / accept reducer:** `scripts/fr13_compare_deliverable.py --left <native.json> --right <tree.json> --out <json>` — TV over served token bags + first-diff.

## The single orchestrator contract (`scripts/fr13_e2e_measure.py` — delivered)
ONE command, owns capture→pairing→reduce so nothing is hand-rolled:
1. **Native arm:** boot forked-FA2 server (or reuse a pinned native capture per [[reference_capture_once_native_pin_prompt]]); run the probe greedy (temp0/top_p1) → `native_run/native_request_metrics.jsonl` (+ hidden capture for the ladder).
2. **Tree arm:** SAME prompts/config, `tree_mtp` → `tree_run/tree_request_metrics.jsonl` (+ hidden capture).
3. **HARD PAIRING ASSERT (the bug-killer):** load both `*_request_metrics.jsonl`; if any pair's `prompt_token_ids` differ byte-for-byte → **`raise SystemExit` with the first-diff index/tokens.** No comparison runs on mismatched prompts. (Today `_prompt_identity` only *reports* this — make the orchestrator FAIL on `byte_identical==false`, and additionally make `_prompt_identity` raise.)
4. **Reduce:** call `fr13_argmax_lcp_localize` (flip layer, spine+branch) + `fr13_compare_deliverable` (bag-TV, accept/event) + the TPS from the probe.
5. **Emit ONE JSON** + print a ladder-log-ready row (commit hash + config + per-depth argmax match + bag-TV + accept/event + TPS), bind to `FR13_LADDER_LOG.md`.
6. **Memory hygiene:** `recover_host_memory` (or docker rm + sync + drop_caches) between arms / on exit — forked-FA2 exit wedges ~90GB ([[reference_modelserver_host_memory_recovery]]).

Implemented as `scripts/fr13_e2e_measure.py`. It does not boot servers itself:
use already-launched native/tree endpoints for capture, or pass `--skip-capture
--native-run ... --tree-run ...` for CPU-only reduction of existing artifacts.

## Discipline (the actual root cause)
- **Commit measurement tools the moment they work** (the num_warps gate, the localizer) — an untracked `.py` is lost on reset and re-hand-rolled. Verify in HEAD + pushed each tick ([[feedback_monitor_verify_work_committed]]).
- **Never bind a comparison whose pairing assert didn't pass** — a `byte_identical:false` / lcp=0 is a measurement artifact, not a result.
- **Use the orchestrator, not inline `python3 -`.** If the orchestrator lacks something, extend IT (and commit), don't improvise around it.
