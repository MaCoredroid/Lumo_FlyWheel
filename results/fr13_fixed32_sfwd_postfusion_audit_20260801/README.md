# FR13 fixed32 SFWD post-state-fusion kernel audit

Status: **SOURCE-BOUND CPU AUDIT; NO NEW GPU MEASUREMENT**

This audit is bound to source commit `5838534519f4f9e91e7fa9979e3dd868884d6cd6`.
It used the existing real SWE-Verified B1 Nsight attribution only. It did not
run a probe, synthetic workload, container, or GPU command, and it makes no B1
or B4 acceptance claim.

## Ranking after current candidates

The historical B1 SFWD ranking was:

| Group | Real Nsight ms/event | Launches/event | Current disposition |
| --- | ---: | ---: | --- |
| target FP8 projections | 112.312954 | 256 | excluded: StreamK work |
| tree FA2 attention | 24.708601 | 16 | already replaced by the attested Qrow16 production path; no current symbol attribution |
| conv state motion | 15.014089 | 144 | addressed by the new default-off one-launch-per-layer state-fusion candidate; candidate time is unmeasured |
| tree GDN path scan | 14.019520 | 96 | source structure remains two launches per layer |

The `24.708601 ms/event` attention number belongs to the old stock FA2 symbol,
not Qrow16. The valid Qrow16 exact4 arm observed a `6.246844 ms/event` whole-wall
improvement, but free-running trajectory differences prevent assigning that
delta to the attention kernel. Reusing the old attention total as a current
optimization budget would therefore be unsound.

A strict current-duration ranking is impossible without a new Nsight capture.
The ranking here is deliberately narrower: source-unchanged groups that are not
already claimed by an active kernel candidate.

The largest still-unclaimed launch/memory group is the 48-layer tree GDN path
scan. The real B1 Nsight capture attributed `14.019520105 ms/event` to
`_tree_gdn_path_kernel`. The exact fixed32 schedule remains 48 layers times two
physical launches, with request batch folded into the grid for B2-B4.

## Exact call sites

- B1 kernel: `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:7124`,
  `_tree_gdn_path_kernel`.
- B2-B4 kernel: the same file at line 7349,
  `_tree_gdn_path_kernel_fixed32_batch`.
- B1 two-level launch loop: lines 11534-11605.
- B2-B4 two-level launch loop: lines 12262-12328.
- Patched model dispatch: `scripts/fr10_phase4_patch_vllm_tree_gdn.py:11328`
  for the batch-folded route and line 11421 for B1.
- The new conv/state candidate is separate: kernel line 4206, entrypoint line
  5292, and patched-model call line 9457.

The GDN first launch computes the five-node root path and writes five FP32
handoff states. The second launch runs eleven dependent paths and reads those
parent states. Ring K/V/A/B export and freshness flags are already inside the
path kernels. With 48 value heads and 128x128 FP32 state per head, the explicit
cross-level handoff is 2,415,919,104 bytes/event at B1 and 9,663,676,416 bytes/event
at B4 before cache effects.

## Conservative saving

- B1 claimable saving: `0.000000 ms/event`. The stale full-group budget is
  `14.019520105 ms/event`; it is a ceiling for prioritization, not a forecast.
- B4 claimable saving: `0.000000 ms/event`. B4 has no real Nsight duration. A
  loose 4x B1 work-scaling reference is `56.078080422 ms/event`; it is not a
  bound or measurement.

No positive point estimate is defensible. A production candidate would have to
be rejected on regression, so zero is the only conservative accepted-candidate
claim before matched real timing.

## Implementation decision

No kernel was added. The second level consumes FP32 state produced by the first.
Triton has no safe cross-program grid barrier for folding these levels into one
ordinary launch. Spin barriers risk residency deadlock; recomputing the five-node
root chain in eleven paths adds 50 node updates per layer and creates duplicate
ring/state writers; moving the work into one program destroys the deployed
head/value parallelism. None is a narrow, semantics-preserving fusion.

The next GDN change should begin only with a design that preserves the exact
ordered node update, single-writer ring/flags contract, and B1-B4 batch-folded
geometry. It then needs real SWE-Verified byte equality before timing.
