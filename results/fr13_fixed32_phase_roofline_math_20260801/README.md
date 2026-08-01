# FR13 fixed32 full-vocabulary phase roofline

Status: **evidence reconstruction; no new GPU measurement**.

This artifact separates three things that must not be conflated:

1. mandatory logical weight bytes and optimistic roofline bounds;
2. completed real SWE-Verified timing from the older `K=65536`, `root=1`
   workload;
3. source-only or byte-qualified kernel candidates with no accepted timing.

All latency numbers below are per physical speculative-decode step. B4 means
four requests share that step; the model weights are read once per shared
forward, while row compute and state work scale with the physical row count.

## Acceptance arithmetic

The required `K=0`, `root=0` step has 42,025,179,008 mandatory weight bytes.
At the GB10 unified-memory planning ceiling of 273 GB/s, the optimistic
weight-only floor is 153.938385 ms and the 1.15x cap is 177.029142 ms. The
entire implementation therefore has only 23.090758 ms above the weight floor
for activation/state traffic, attention, launches, synchronization, host work,
and bandwidth inefficiency.

The 273 GB/s number is an architectural roofline, not a measured achieved
bandwidth. The real B1 attribution observed only 212.118 GB/s when target FP8
projection weight bytes are divided by their measured time, 209.216 GB/s for
the verifier full head, and 127.937 GB/s for the old 64K drafter GEMV heads.
Those observations are shape- and source-specific priors, not replacement
floors.

## Phase roofline

| Phase | Mandatory weight bytes | 273 GB/s floor | B1 compute sanity | B4 compute sanity | Defensible bound class |
| --- | ---: | ---: | ---: | ---: | --- |
| SFWD target model | 24,382,399,488 | 89.312819 ms | 17.280 ms | 69.120 ms | weight bandwidth plus projection-wave occupancy; state/attention subpaths are memory and launch bound |
| DFWD, five MTP forwards plus five full heads | 15,099,982,720 | 55.311292 ms | 0.139888 ms logical | 0.559551 ms logical | weight-stream memory bound; short attention and selection remain launch sensitive |
| CFWD commit and sampling | 0 | 0 ms | not quantified | not quantified | state-memory, softmax, reduction, and launch bound; no model-weight floor |
| Other, verifier full head plus wall residual | 2,542,796,800 | 9.314274 ms | 0.650956 ms head | 2.603824 ms head | verifier head is weight bound; host/synchronization residual is unmodeled |
| Total | 42,025,179,008 | 153.938385 ms | phase floors are not additive with compute | phase floors are not additive with compute | 23.090758 ms total nonweight allowance |

The SFWD 17.28/69.12 ms values are the existing analytic 0.54 ms per physical
row model, not measured FLOP throughput. The more explicit projection-only
calculation is 23,823,646,720 FP8 weight bytes and 256 GEMMs. At M=32/M=128,
the weight-only arithmetic intensity is 64/256 flop/B. At the separate 125
TFLOP/s dense-BF16 planning ceiling, projection compute floors are
12.197707/48.790828 ms and only 17.472/69.888 TFLOP/s is required to finish at
the 87.266105 ms projection weight floor. Even the B4 projection is therefore
on the memory side of this conservative compute sanity check.

DFWD is more strongly memory dominated. The five full-vocabulary BF16 heads
alone stream 12,713,984,000 bytes for a 46.571370 ms weight floor; the five MTP
forwards add 2,385,998,720 bytes and 8.739922 ms. Logical B1/B4 row compute is
only 0.139888/0.559551 ms at 125 TFLOP/s. A source-only M32 padded-head
candidate raises the phase compute sanity to 3.292956/3.407484 ms, still far
below the 55.311292 ms weight floor.

The weighted phases are not compute-bound at B1 or B4 under the committed
planning assumptions. That does not make the complete step a pure weight-read
problem: CFWD has no weight floor, and SFWD attention, GDN, and conv paths move
substantial state and pay many device launch boundaries.

## Real timing anchor and remaining gap

There is no committed real SWE-Verified exact4 full-vocabulary B1 or B4 wall
timing in the inspected evidence. The nearest valid phase prior is the real
exact4 Hydra27 B1 K64 arm:

| Phase | Measured K64 prior | Ideal full-vocab delta | Planning value | Full-vocab weight floor | Excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFWD | 159.619263 ms | 0 | 159.619263 ms | 89.312819 ms | 70.306445 ms |
| DFWD | 36.813368 ms | 34.280369 ms | 71.093737 ms | 55.311292 ms | 15.782445 ms |
| CFWD | 20.677391 ms | 0 | 20.677391 ms | 0 | 20.677391 ms |
| Other | 15.669768 ms | 0 | 15.669768 ms | 9.314274 ms | 6.355494 ms |
| Total | 232.779790 ms | 34.280369 ms | 267.060159 ms | 153.938385 ms | 113.121775 ms |

The 34.280369 ms delta is only the added full-head bytes at ideal 273 GB/s.
Thus 267.060159 ms is an optimistic transfer estimate, not a full-vocabulary
measurement. It remains 90.031017 ms above the cap. Equivalently, excess above
the weight floor must fall from 113.121775 ms to at most 23.090758 ms, removing
79.59% of that excess.

Even if DFWD, CFWD, and Other reached their lower bounds, SFWD would have to
fall by at least 47.215687 ms. The historical ideal Stream-K recovery model is
only 10.923627 ms, so it cannot close the SFWD requirement alone; at least
36.292060 ms of further SFWD reduction remains under that optimistic model.

## Small-phase classification

| Kernel group | B1 evidence | B4 evidence | Classification |
| --- | --- | --- | --- |
| target FP8 projections | 112.312954 ms stale real-workload attribution, 256 launches | no timing, 256 launches | weight bandwidth plus output-tile/wave occupancy |
| tree FA2 attention | 24.708601 ms old-stock attribution, 16 launches | no timing, 16 launches | short-sequence memory and launch bound; KV bytes are context dependent |
| tree GDN path | 14.019520 ms stale attribution, 96 launches | no timing, 96 batch-folded launches | state memory plus launch bound |
| conv state motion | 15.014089 ms stale attribution, 144 incumbent launches | no timing; one-launch-per-layer candidate is source-only | state movement plus launch bound |
| MTP projections | 8.514285 ms stale attribution, 20 launches | no timing, 20 launches | weight-stream memory bound |
| five drafter heads | full-vocab time unmeasured | full-vocab time unmeasured | dominant DFWD weight stream; stock and padded-head efficiency must be measured |
| DFWD attention/selection | 10.773280 ms stale listed attribution | no timing; launch structure expected invariant | short work plus launch bound |
| CFWD state/TAW | 20.677391 ms whole-phase K64 exact4 prior | no timing | launch, state traffic, softmax, and reduction bound |
| verifier head | 12.153933 ms stale full-head attribution | no timing | weight-stream memory bound |

The GDN cross-level source audit exposes 2,415,919,104 logical FP32 handoff
bytes at B1 and 9,663,676,416 at B4. Their 273 GB/s HBM equivalents are
8.849521 and 35.398082 ms. These are not hard DRAM floors because cache
residency was not measured, but the B4 value alone exceeds the complete
23.090758 ms nonweight allowance if it reaches unified memory. B4 therefore
cannot be assumed to inherit B1 timing merely because its launch count is
fixed.

The only completed real B4 exact4 evidence in this lineage is a K64/root1 GDN
byte gate. It served the reference, collected no timing, and is not comparable
to the full-vocabulary goal. B4 phase and wall timing remain unmeasured.

## Decision

The path to the cap requires all of the following, in this order of impact:

1. move SFWD projections materially toward the 89.312819 ms target-model
   weight floor, then remeasure the entire SFWD phase;
2. keep B4 GDN/attention/conv state traffic resident or remove round trips,
   because launch invariance alone does not control the 4x row/state work;
3. measure and optimize the five full-vocabulary drafter heads, whose immutable
   weight floor is already 46.571370 ms;
4. trim CFWD launch/reduction work, while recognizing that deleting the entire
   old 20.677391 ms CFWD phase still would not close the 90.031017 ms gap.

Qualification still requires matched real SWE-Verified exact4 full-wall runs
at B1 and B4. One-task diagnostics and synthetic/probe timings are not
acceptance evidence.

`roofline.json` contains the same accounting with evidence class, source
commit, and artifact path for every non-derived claim.
