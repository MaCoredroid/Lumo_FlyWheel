# FR13 Step 3 Post-Handoff Bind

Date: 2026-06-11 UTC

Commit under test: `7ff88162c54af53b75d8d14aed80c4044b9cef63`

Included fix: `37a98fbd` (`Fix FR13 replay B4 async draft handoff`)

## Verdict

FAIL. The fixed replay-on tree arm no longer crashes and produced a valid
CUDA-captured 16-record B=4 arm, but the committed 3-arm reducer returned a
valid gate failure.

Reducer exit code: `2`

Reducer artifact: `output/fr13_step3_b4_gate/fr13_corruption_gate.json`

## Commands

```bash
bash output/fr13_step3_b4_gate/run_arm.sh tree tree_mtp TREE_ATTN 1313 tree tree \
  > output/fr13_step3_b4_gate/tree_arm_runner.log 2>&1

NUM_SPECULATIVE_TOKENS=5 \
  bash output/fr13_step3_b4_gate/run_arm.sh native naive_mtp FLASH_ATTN 1313 native native \
  > output/fr13_step3_b4_gate/native_arm_runner.log 2>&1

NUM_SPECULATIVE_TOKENS=5 \
  bash output/fr13_step3_b4_gate/run_arm.sh native_noise naive_mtp FLASH_ATTN 2313 native_noise native \
  > output/fr13_step3_b4_gate/native_noise_arm_runner.log 2>&1

bash output/fr13_step3_b4_gate/run_reduce.sh
```

The old K=9 native/native-noise artifacts were archived before rerun under:
`output/fr13_step3_b4_gate/archive_k9_pre_e5_20260611/`

## Arm Evidence

| arm | backend/mode | seed | records | active spec width | draft_tokens/draft | FULL capture | accept/event | warm TPS |
|---|---|---:|---:|---:|---:|---|---:|---:|
| tree | `TREE_ATTN/tree_mtp` | 1313 | 16 | 9 | 9.0 | yes | 2.132045 | 7.566623 |
| native | `FLASH_ATTN/naive_mtp` | 1313 | 16 | 5 | 5.0 | yes | 2.783088 | 14.314460 |
| native_noise | `FLASH_ATTN/naive_mtp` | 2313 | 16 | 5 | 5.0 | yes | 2.937023 | 14.396713 |

K=5 native proof:

- `docker_full.log` for both native arms reports `SpeculativeConfig(... num_spec_tokens=5)`.
- `logs/fr10_mtp_draft_trace.jsonl` width set is `[5]` for both native arms.
- Probe summaries give `spec_draft_tokens/spec_drafts == 5.0` for both native arms.
- Prompt/request key checks matched across all three arms: 16 rows, prompt token counts
  `[681, 1080, 829, 1614]`, equal record-key sets, and equal prompt token ids.

Replay-on tree proof:

- Tree arm booted `TREE_ATTN/tree_mtp` with `FR13_REPLAY_ROUTE=1`.
- FULL CUDA graph capture completed (`decode, FULL` 4/4 and `Graph capturing finished`).
- Tree probe produced 16 records and 647 `tree_path_lcp.jsonl` rows.
- Tree crash scan found no `vectorized_gather`, `device-side assert`, `CUDA error`,
  `Traceback`, `ERROR`, `AssertionError`, or `FATAL` signatures in the checked tree logs.

## Reducer Result

`scripts/fr13_corruption_gate.py` returned:

- `valid: true`
- `verdict: FAIL`
- `prompt_identity.tree_vs_native: true`
- self-noise mask positions: `364`
- native self bag-TV: `0.19677734375`

Failing metrics:

- Real-loss rate: `0.28378378378378377` > `0.05`
- Depth collapse: prompt `0`, run length `6`, run end position `59`
- Bag-TV: `0.2438904313016529` > reducer budget `0.19677734375`
- Accept/event: tree `2.132045088566828` vs native `2.7830882352941178`, delta `-0.6510431467272899`

The bag-TV failure is robust to the documented `0.113` floor: `0.2438904313016529`
is above both `0.113` and the reducer's native-self budget `0.19677734375`.

## Caveats

- The committed reducer still prints `bag_tv_floor: 0.0593`; in this run the
  effective reducer budget was the larger native-self value, `0.19677734375`.
  This does not affect the verdict because tree bag-TV exceeded both the
  documented `0.113` floor and the reducer budget.
- The launcher still includes the 9-node `speculative_token_tree` string in the
  native K=5 SPEC_CONFIG. vLLM's active config, MTP traces, and draft ratios all
  prove the native comparator used K=5, not K=9.
- Optional per-event superset trace reduction was `null`; the summary-level
  accept/event gate failed independently.

