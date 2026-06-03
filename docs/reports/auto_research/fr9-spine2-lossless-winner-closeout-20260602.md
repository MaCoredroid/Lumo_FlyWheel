# FR9 Spine-2 Lossless Winner Closeout

**Date:** 2026-06-02
**Branch:** `fr9-spine2-lossless-winner`
**Verdict:** CLOSED_NON_SHIP for public independent `spines>1` on this stack.
The lossy best-of-spines public winner is deleted/fail-closed, but selector-off
two-spine mode is not a verified lossless public mode on vLLM 0.19 with
GDN/linear-attention. The controlled greedy probe showed that merely
co-scheduling hidden spine 1 perturbs spine 0's public greedy tokens. The branch
now fails closed for independent `spines>1`; `spines=1` remains the only
lossless path and the only speed-candidate path.

## What changed

- Deleted the launchable longest-accepted hidden-winner public commit path in
  `scripts/swe_x86_helpers/relaunch_qwen36_round.py`.
- Added explicit public commit policy validation:
  `LUMO_IR_PUBLIC_COMMIT_POLICY=lossless` is the default and only accepted
  public policy.
- Fail-closed before model launch for:
  `best_of_spines`, `unsafe_best_of_spines`, `deterministic_best`, unknown
  policy, hidden-publication request before selector implementation, selector
  enabled before implementation, disabled independent-row winner commit, and
  independent `spines>1` on the current GDN/vLLM stack.
- In the injected winner patch, public commit row is selected by
  `spine_id == 0`, not tensor row order.
- Hidden rows remain diagnostic only. The trace records `candidate_winner_*`,
  but `winner_spine` remains 0 and recurrent state copies from spine 0 to
  hidden sibling rows. This diagnostic path is no longer admitted as a public
  lossless mode because the hidden sibling can perturb spine 0's recurrent
  logits before winner commit.
- Kept the Qwen parser/protocol guards from `argrepair7`.
- Added explicit policy fields to winner trace, independent winner summary, and
  agentic summary annotation.
- Added `scripts/fr9_gate_b_greedy_probe.py` to collect exact temp=0 token IDs
  from the live vLLM endpoint for fixed b1 and b4 prompt shapes and compare
  artifacts without hand-editing trace files.

## Evidence Collected

Focused tests:

```text
python3 -m pytest -q \
  tests/test_run_codex_experiment_spines.py \
  tests/test_independent_winner_commit_patch.py \
  tests/test_independent_winner_trace.py \
  tests/test_lossless_selector_gate_c_stub_design.py

37 passed in 0.43s
```

Compile/static checks:

```text
python3 -m py_compile \
  scripts/swe_x86_helpers/relaunch_qwen36_round.py \
  scripts/verify_independent_winner_trace.py \
  scripts/run_codex_experiment.py \
  scripts/fr9_gate_b_greedy_probe.py
git diff --check
```

Both passed.

Gate A fail-closed probes:

- `LUMO_IR_PUBLIC_COMMIT_POLICY=best_of_spines` failed before model launch.
- `LUMO_IR_PUBLIC_COMMIT_POLICY=longest_prefix` failed before model launch.
- `LUMO_IR_ALLOW_STOCHASTIC_HIDDEN_WINNER=1` failed before model launch.
- `LUMO_IR_LOSSLESS_SELECTOR_ENABLED=1` failed before model launch because the
  selector is not implemented.
- `--config Fb --row-mode independent --spines 2` now fails before ModelServer
  launch because independent `spines>1` is not verified lossless on the
  vLLM 0.19 GDN/linear-attention stack.

Earlier live relaunch before the final GDN fail-closed gate:

```text
LUMO_GPU_MEMORY_UTILIZATION=0.88 \
python3 scripts/swe_x86_helpers/relaunch_qwen36_round.py \
  --config Fb --mtp 5 --row-mode independent --spines 2

READY config=Fb mtp=5 row_mode=independent spines=2 policy=lossless
```

The first 0.90-memory relaunch failed honestly before serving:

```text
Free memory on device cuda:0 (105.35/117.51 GiB) on startup is less than
desired GPU memory utilization (0.9, 105.76 GiB).
```

Live temp=0.6 probe appended 11 new winner events:

```json
{
  "rows": 11,
  "policies": {"lossless": 11},
  "lossless_public_stream_events": 11,
  "selector_enabled_events": 0,
  "winner_spines": {"0": 11},
  "winner_nonzero_spine_events": 0,
  "copy_missing_sum": 0,
  "superset_violations": 0
}
```

Live temp=0 probe appended 10 new winner events:

```json
{
  "rows": 10,
  "policies": {"lossless": 10},
  "lossless_public_stream_events": 10,
  "selector_enabled_events": 0,
  "winner_spines": {"0": 10},
  "winner_nonzero_spine_events": 0,
  "copy_missing_sum": 0,
  "superset_violations": 0
}
```

Gate C design stub, not a selector proof:

- The old hollow placeholder was renamed to
  `tests/test_lossless_selector_gate_c_stub_design.py`.
- It now characterizes the analytic max-order-statistic negative control from
  `/tmp/fr9_lossless_research.md`: the best-of-spines order statistic must fail
  versus target `p` and match the biased analytic distribution
  `p'(z)=p(z)*(2*CDF(z)-p(z))`.
- It still does not exercise a production multi-draft selector because no
  selector-on path is implemented on this branch.

Controlled Gate B probe after the batch-invariance directive:

```text
LUMO_BATCH_INVARIANT_VLLM=1 LUMO_GPU_MEMORY_UTILIZATION=0.88 \
python3 scripts/swe_x86_helpers/relaunch_qwen36_round.py \
  --config Fb --mtp 5 --row-mode independent --spines 1

READY config=Fb mtp=5 row_mode=independent spines=1 policy=lossless
```

The relaunch log for this arm reported `--attention-backend FLASH_ATTN` and
`max_num_seqs=4`. Collection:

```text
python3 scripts/fr9_gate_b_greedy_probe.py collect \
  --arm s1_batchinv_flashattn \
  --out /tmp/fr9_gate_b_s1_batchinv_flashattn.json \
  --wait-health 30

{"arm": "s1_batchinv_flashattn", "out": "/tmp/fr9_gate_b_s1_batchinv_flashattn.json", "records": 8, "reset_prefix_cache_error": null}

python3 scripts/fr9_gate_b_greedy_probe.py compare-batches \
  /tmp/fr9_gate_b_s1_batchinv_flashattn.json

"exact_match": true, "matched_records": 4
```

Then:

```text
LUMO_BATCH_INVARIANT_VLLM=1 LUMO_GPU_MEMORY_UTILIZATION=0.88 \
python3 scripts/swe_x86_helpers/relaunch_qwen36_round.py \
  --config Fb --mtp 5 --row-mode independent --spines 2

READY config=Fb mtp=5 row_mode=independent spines=2 policy=lossless
```

The relaunch log for this arm reported `--attention-backend FLASH_ATTN`,
`max_num_seqs=8`, and `LUMO_IR_PUBLIC_COMMIT_POLICY=lossless`. Collection:

```text
python3 scripts/fr9_gate_b_greedy_probe.py collect \
  --arm s2_batchinv_flashattn \
  --out /tmp/fr9_gate_b_s2_batchinv_flashattn.json \
  --wait-health 30

{"arm": "s2_batchinv_flashattn", "out": "/tmp/fr9_gate_b_s2_batchinv_flashattn.json", "records": 8, "reset_prefix_cache_error": null}

python3 scripts/fr9_gate_b_greedy_probe.py compare-batches \
  /tmp/fr9_gate_b_s2_batchinv_flashattn.json

"exact_match": true, "matched_records": 4
```

Recent spine-2 winner trace for that collection window:

```json
{
  "rows": 40,
  "policies": {"lossless": 40},
  "winner_spines": {"0": 40, "1": 0},
  "selector_enabled": 0,
  "non_lossless": 0,
  "suppressed": 2,
  "copy_missing": 0
}
```

Cross-arm exact comparison still failed:

```text
python3 scripts/fr9_gate_b_greedy_probe.py compare \
  /tmp/fr9_gate_b_s1_batchinv_flashattn.json \
  /tmp/fr9_gate_b_s2_batchinv_flashattn.json

"exact_match": false, "matched_records": 8
```

Mismatches were present at matched batch shapes (`b1` and `b4`) for two prompts:

```text
prompt 0: "Q: Count from one to five. A:"
  first_diff_index=18
  s1 token_ids=[271, 248068, 271, 248069, 271, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 13, 248044]
  s2 token_ids=[271, 248068, 271, 248069, 271, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 248044]

prompt 2: "Q: Write three lowercase letters in alphabetical order. A:"
  first_diff_index=2
  s1 token_ids=[271, 248068, 198, 8160, 579, 264, 7047, 1817, 25, 271, 16, 13, 220, 2972, 2014, 53983, 2570, 5396, 64700, 198, 256, 471, 2972, 14162]
  s2 token_ids=[271, 248068, 271, 248069, 271, 13290, 248044]
```

This result must not be described as the earlier b4-only batch-shape mismatch:
the same-arm b1-vs-b4 controls passed for both arms. It also must not be
described as hidden-winner publication: recent trace shows selector-off
spine-2 committed spine 0 for all 40 recent winner events. The honest current
status is that Gate B exact equality is still not passed under the controlled
spines1-versus-spines2 probe, and the remaining mismatch needs separate
investigation.

Root-cause finding from the follow-up briefs:

- `/tmp/fr9_gdn_state_isolation.md` identifies the remaining perturbation as a
  GDN/linear-attention recurrent-state isolation problem, not attention/GEMM
  batch noise. `LUMO_BATCH_INVARIANT_VLLM=1` plus FLASH_ATTN controls the
  attention/GEMM side, but does not make the GDN/Mamba recurrent kernels
  batch-invariant.
- vLLM issue #42960 explicitly says raw batch-invariant mode is unsupported for
  `GDN_ATTN` and hard-aborts with `RuntimeError: VLLM batch_invariant mode is
  not supported for GDN_ATTN`: <https://github.com/vllm-project/vllm/issues/42960>.
- The observed greedy flip is therefore evidence that co-scheduling hidden
  spine 1 changes spine 0's logits before any public winner selection occurs.
  Copying or suppressing the hidden winner after verification cannot repair a
  perturbed public recurrent state/logit stream.

Temp=0 clone-suppression attempt:

- I briefly implemented a deterministic-request bypass that would avoid creating
  hidden clones at `temperature <= 0`. That would make `spines=2` greedy output
  match `spines=1` only by reducing the served path to a single spine by
  construction.
- That behavior is not evidence that the two-spine path is intrinsically
  greedy-lossless, and it says nothing about production `temperature=0.6`, where
  the hidden clone still runs and can perturb spine 0.
- The bypass was removed from the final code in favor of fail-closing
  independent `spines>1` until GDN state isolation or a lossless selector plus
  public recurrent-state recompute exists.

Gate D methodology and attempted sampling:

- `/tmp/fr9_gate_d_methodology.md` says the 512-sample temp=0.6 comparison is
  badly underpowered for a losslessness claim. The 12 generated tokens in a
  completion are autoregressively dependent and cannot be counted as 12 iid
  next-token draws.
- I collected one underpowered s2 artifact at
  `/tmp/fr9_gate_d_s2_temp06_selector_off_clone_active.json`
  (`samples=512`, `temperature=0.6`, clone active) before the methodology brief
  arrived. I stopped the matching s1 run after reading the brief because it
  would not support a Gate D pass or fail.
- No Gate D conclusion is claimed from this sampling. A valid Gate D would first
  measure the `spines=1` self-distance across relaunches, then compare
  `spines=2` against that floor as a TV-upper-bound equivalence test; the cheaper
  rigorous route is direct post-temperature top-k logprob comparison for fixed
  prompts.

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| A policy fail-closed | Passed | Unit tests plus launch-time fail-closed probes; independent `spines>1` now rejected on GDN/vLLM 0.19 |
| B greedy equality | Failed for real two-spine path | Same-arm b1-vs-b4 passed under `LUMO_BATCH_INVARIANT_VLLM=1` + FLASH_ATTN, and s2 trace committed spine 0 only; controlled s1-vs-s2 exact token comparison still failed on two prompts. A temp=0 clone bypass would pass only by reducing to `spines=1`, so it is not used as proof |
| C synthetic selector convergence | Not passed | Only a clearly marked Gate C stub/design-power check exists; no selector-on implementation is exercised |
| D target-model sampling equivalence | Not passed / not proven | One 512-sample s2 artifact was collected but is underpowered and has no s1 self-distance baseline; no valid TV-upper-bound equivalence result exists |
| E SWE admission | Not run | No 16-task SWE run collected |

## Honesty Notes

- I do not claim full losslessness proof. The actual clone-active two-spine path
  failed controlled greedy equality, and no valid Gate D equivalence result
  exists.
- I do not classify the earlier b4-only divergence as a spine-2 distribution
  bug; after batch-invariance controls, both arms were internally b1-vs-b4
  stable. The remaining mismatch is at matched batch shapes and with
  spine0-only public commits in the recent s2 trace.
- I do not present temp=0 clone suppression as a losslessness proof. That
  workaround would only turn `spines=2` greedy into `spines=1` by construction
  and does not address production `temperature=0.6`.
- I do not claim a speed win and did not run kernel profiling.
- Selector-on remains intentionally fail-closed on this branch. Independent
  `spines>1` is also fail-closed on this GDN/vLLM stack until recurrent-state
  isolation or a real distribution-preserving multi-draft selector with public
  recurrent-state recompute exists.
- Any future non-spine0 public commit must also implement GDN/linear-attention
  public recurrent-state recompute from the prior committed spine0 state; copying
  a sibling spine state across a divergent prefix remains forbidden.
