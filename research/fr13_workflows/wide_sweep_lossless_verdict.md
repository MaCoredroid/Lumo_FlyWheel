# FR13_RESHAPE_WIDE lossless gate verdict — BOTH wide arms PASS (temp-0.6, B=1)

Per-token clear-margin argmax flip-rate vs the no-spec RECURRENT decode oracle (seed1313,
topk20, thresh1.0 nat), 40 sampled SWE turns/arm. Tree arm LOSSLESS iff its flip-rate is
NOT above the depth-matched native floor (Wilson CI / two-proportion z). US vs native-E5/E3
vs no-spec oracle — never a proxy.

| arm | flips/positions | rate | Wilson 95% CI | engaged | det |
|---|---|---|---|---|---|
| nativeE3 (d3 floor) | 1102/10180 | 10.83% | [10.24, 11.44]% | True | True |
| cat555 (d3 wide, 15-node) | 994/9256 | 10.74% | [10.12, 11.39]% | True | True |
| nativeE5 (d5 floor, banked) | 630/4883 | 12.90% | [11.99, 13.87]% | True | True |
| cat55221 (d5 wide, 15-node) | 1168/10658 | 10.96% | [10.38, 11.57]% | True | True |

## Verdict: LOSSLESS at both depths
- **cat555 (depth-3 wide): LOSSLESS** — 10.74% vs E3 floor 10.83%, z(arm−floor) = −0.19
  (statistically indistinguishable from native; CIs fully overlap).
- **cat55221 (depth-5 wide): LOSSLESS** — 10.96% vs E5 floor 12.90%, z = −3.52
  (significantly BELOW the native floor — the wide superset commits at least as cleanly as native).

The general width-N caterpillar drafter (FR13_RESHAPE_WIDE, top-5) is **lossless within the native
floor** at both depths — confirming the drafter-agnostic committer holds for the wider trees
(leaves are pure topk reads, never fed forward; rejection-sampled against the true target).
engaged=True + det=True on all arms => valid, non-vacuous.

## Combined with the speed verdict (wide_sweep_verdict.md)
- SPEED: wider-not-deeper does NOT help (cat6 6-node +15.7% gpu / +27% deploy WINS; cat55221
  15-node ties E5 −0.4%; cat555 −3.3% vs E3). Sweet spot = moderate width.
- LOSSLESS: both wide arms PASS.
Net: the wide trees are CORRECT and LOSSLESS, but offer no speed advantage over the moderate
cat6 — so the deployed cat6 stays the best shape. The sweep answered the "does wider help"
question (no) without any losslessness cost.
