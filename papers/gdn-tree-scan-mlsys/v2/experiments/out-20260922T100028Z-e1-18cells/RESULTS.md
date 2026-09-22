# E1 results — frozen 18-cell campaign `out-20260922T100028Z-e1-18cells` (compact; written 2026-09-22T12:17:25Z)

Scope: measured tokens per wall-second of the frozen stack on ONE hardware/model configuration (GB10, Qwen3.6-27B-FP8, pinned vLLM image), three arms (native chain-5 = baseline; native chain-11 = longer-chain control; cat10 tree on the qualified stock-TREE_ATTN policy-B route), two actual batch conditions (B1: 8 sequential prompts; B4: two cohorts of 4), three paired boot blocks, one fresh boot per cell, synchronous scheduling, eager, cache off, temperature 0, 128-token budget. Native arms are the PATCHED-RUNNER NATIVE BASELINE (stock MTP method, naive_mtp decode mode, tree GDN off, no tree descriptor). Rates are the frozen primary estimand: Σ API-bound emitted tokens over Σ wall of UNIQUE retained pure same-cohort physical intervals after the identified warm-up; every valid interval retained (no interval exceeded the 1.5 s diagnostic cap in any cell). Aggregate: `e1/e1_aggregate.py` sha 98beec9efac26f56 (unchanged; default frozen rules: paired whole-boot blocks, B1/B4 separately, 10 000 bootstrap resamples of the 3 blocks, seed 20260921, 95 % percentile, precision = (U−L)/(2·mean native-5 rate), target 0.10, no replacement). `aggregate.json` sha dbd4542ee9899da3; campaign snapshot SHA256SUMS d1ab57c53464ef81.

## Outcome
- 18 / 18 cells sealed VALID (F1 terminal seals; no INVALID, no INSUFFICIENT_SUPPORT, no retry, no re-run); all four native routes qualified by their in-boot untimed preflights (live + final PASS; margin-gated greedy rule 4/4 in every preflight, low-margin prefixes also equal); tree routes qualified before the campaign (bounded, closed).
- Coverage floors met in every cell (B1: every one of the 8 prompts ≥ 24 retained intervals; B4: both cohorts ≥ 25 complete four-request intervals). Precision: all six contrasts within the 0.10 target (no precision miss). Runtime: 10:00:28Z → 12:16:26Z, serial.
- Deviation T1 (recorded in `e1/E1_ASEXECUTED_DEVIATIONS.md`, not corrected retroactively): tree-arm engine seed = vLLM default 0 (launcher v7 carries no `--seed`), native arms 20260921; API request seeds 20260921 everywhere; greedy workload; reviewer's source chain: the duplicate-source selector uses its own generator — no corrupted sampling/rate evidence indicated. Cell 16 (tree B1): the p021 timed request stopped at EOS after 101 tokens (retained; floor unaffected).

## Aggregate (frozen rule)
**B1** — per-arm block rates (tok/s, blocks 1/2/3) and mean: native-5 13.682 / 13.662 / 13.667 → mean 13.670; native-11 11.440 / 11.440 / 11.435 → mean 11.439; tree 12.597 / 12.607 / 12.880 → mean 12.695
- B1 tree minus native5: paired block differences -1.085, -1.055, -0.787; mean -0.976; bootstrap 95 % percentile interval [-1.085, -0.787]; precision (U−L)/(2·R̄_native-5=13.670) = 0.0109 vs target 0.10 → within target
- B1 tree minus native11: paired block differences +1.157, +1.167, +1.445; mean +1.256; bootstrap 95 % percentile interval [+1.157, +1.445]; precision (U−L)/(2·R̄_native-5=13.670) = 0.0105 vs target 0.10 → within target
- B1 native11 minus native5: paired block differences -2.242, -2.222, -2.232; mean -2.232; bootstrap 95 % percentile interval [-2.242, -2.222]; precision (U−L)/(2·R̄_native-5=13.670) = 0.0008 vs target 0.10 → within target
**B4** — per-arm block rates (tok/s, blocks 1/2/3) and mean: native-5 57.331 / 52.519 / 56.275 → mean 55.375; native-11 43.471 / 42.280 / 40.701 → mean 42.151; tree 47.651 / 46.689 / 47.512 → mean 47.284
- B4 tree minus native5: paired block differences -9.680, -5.830, -8.763; mean -8.091; bootstrap 95 % percentile interval [-9.680, -5.830]; precision (U−L)/(2·R̄_native-5=55.375) = 0.0348 vs target 0.10 → within target
- B4 tree minus native11: paired block differences +4.179, +4.410, +6.811; mean +5.133; bootstrap 95 % percentile interval [+4.179, +6.811]; precision (U−L)/(2·R̄_native-5=55.375) = 0.0238 vs target 0.10 → within target
- B4 native11 minus native5: paired block differences -13.860, -10.240, -15.574; mean -13.225; bootstrap 95 % percentile interval [-15.574, -10.240]; precision (U−L)/(2·R̄_native-5=55.375) = 0.0482 vs target 0.10 → within target

Reading: in both batch conditions the cat10 tree is slower than the native chain-5 baseline and faster than the native chain-11 control; the three-block intervals are coarse by construction but their half-widths are 0.001–0.05 of the baseline rate.

## Per-cell support (sealed results)
| Cell | Block | Arm | Batch | Status | Rate tok/s | Retained intervals | Wall s | API-bound tokens | Min intervals per prompt/cohort | Prespecified exclusions | Intervals > 1.5 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | native-5 | B1 | VALID | 13.682 | 312 | 72.72 | 995 | 29 | 117 | 0 |
| 2 | 1 | native-5 | B4 | VALID | 57.331 | 56 | 13.40 | 768 | 28 | 66 | 0 |
| 3 | 1 | native-11 | B4 | VALID | 43.471 | 52 | 18.20 | 791 | 26 | 61 | 0 |
| 4 | 1 | native-11 | B1 | VALID | 11.440 | 249 | 86.19 | 986 | 24 | 114 | 0 |
| 5 | 1 | tree | B1 | VALID | 12.597 | 314 | 78.98 | 995 | 34 | 32 | 0 |
| 6 | 1 | tree | B4 | VALID | 47.651 | 51 | 14.35 | 684 | 25 | 54 | 0 |
| 7 | 2 | tree | B1 | VALID | 12.607 | 314 | 78.92 | 995 | 34 | 32 | 0 |
| 8 | 2 | tree | B4 | VALID | 46.689 | 61 | 17.22 | 804 | 30 | 36 | 0 |
| 9 | 2 | native-5 | B4 | VALID | 52.519 | 64 | 15.33 | 805 | 29 | 42 | 0 |
| 10 | 2 | native-5 | B1 | VALID | 13.662 | 312 | 72.83 | 995 | 29 | 28 | 0 |
| 11 | 2 | native-11 | B1 | VALID | 11.440 | 249 | 86.19 | 986 | 24 | 28 | 0 |
| 12 | 2 | native-11 | B4 | VALID | 42.280 | 52 | 18.16 | 768 | 25 | 39 | 0 |
| 13 | 3 | native-11 | B1 | VALID | 11.435 | 249 | 86.22 | 986 | 24 | 28 | 0 |
| 14 | 3 | native-11 | B4 | VALID | 40.701 | 52 | 18.18 | 740 | 26 | 39 | 0 |
| 15 | 3 | tree | B4 | VALID | 47.512 | 62 | 17.43 | 828 | 30 | 30 | 0 |
| 16 | 3 | tree | B1 | VALID | 12.880 | 298 | 74.92 | 965 | 29 | 30 | 0 |
| 17 | 3 | native-5 | B1 | VALID | 13.667 | 312 | 72.80 | 995 | 29 | 28 | 0 |
| 18 | 3 | native-5 | B4 | VALID | 56.275 | 60 | 14.36 | 808 | 27 | 43 | 0 |
