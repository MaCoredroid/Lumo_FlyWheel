# Mechanism → implementation → fresh evidence → ablation

Status: system-design rewrite, 2026-09-22. User approved this framing and targeted missing experiments. This map does not promote historical numbers or treat source availability as a timing result. Detailed source/loaded-route evidence is recorded in [the independent route audit](../optimization-route-audit-2026-09-22.md).

| Mechanism | Implementation | Fresh measured-route status | Evidence and remaining question |
| --- | --- | --- | --- |
| Shared descriptor and selected continuation | Tree parents/masks/slots; GDN/conv/KV publication and pending token | Selected synchronous flat-map Cat10 route | Fresh E2 bounded checks; broader full-model/cache/graph/stochastic claims remain outside scope |
| Sequential scan and replay | `fr10_gdn_tree_kernel.py`, shared node update and authoritative running-row publication | Generic Cat10 scan; its per-request scan is already monolithic | Fresh E7a and E2; replay arithmetic is capacity-bounded with masked slots, not simply accepted-length proportional |
| Draft-logit reuse | Emitted `eagle.py` single-logits branch | Active, baked ON | Existing env=0 cannot ablate it. E8 creates a real isolated switch and checks same-input candidate equivalence first |
| Device probability arithmetic | Loaded device multidraft committer | Active, with Python request/node loops and scalar readbacks | Fresh selected-path checks; no proof of wholly device-resident control and no fresh device/host latency ablation |
| Packed eager transport | Stacked operand rings and shared metadata | Active buffers; all-layer replay is not engaged in canonical greedy replay | Loaded replay still loops per layer; buffer engagement is not launch-batching evidence. No isolated fresh benefit claimed |
| Vectorized tree convolution | Static source indices, ordered tap arithmetic, batched row preparation | Active in fresh E1 | Preserves intended operation ordering; independent compiled/live checks remain distinct from source reasoning |
| Attention layout and KV remap | FA2 spine-first witness; stock TREE_ATTN flat-map continuation | Fresh E1 uses flat slots and KV remap, not the earlier FA2 layout | Fresh E2 rejects an incompatible combined policy. Do not use a corrupt state/layout path as a serving-speed baseline |
| Attention grouping/padding | Forked-FA2 optimization family | Outside fresh stock TREE_ATTN | Separate route qualification and matched timing required |
| Fixed-shape folded scan | Fixed32 batched/path-specific candidate | Outside generic Cat10 | Requires compatible topology/geometry; cannot infer a gain from toggling an inactive flag |
| Batched acceptance walk (TAW) | Generic/fixed-shape stochastic candidates | Excluded by the fresh all-greedy predicates | Stochastic route qualification needed; do not alter temperature merely to activate it and call that a same-route ablation |

## Minimum fresh experiment

**E8: draft-logit reuse, B1, same Cat10/TREE_ATTN route.** Run a real source-controlled ON/OFF switch in a new experiment snapshot. Forward explicit engine seed, record loaded source and actual branch engagement, and execute the existing in-process comparison only after checking its coverage. Qualification must reconcile spine token choice, top-k candidates/proposal information and downstream selected paths; a failed same-input check ends the experiment before timing. A check of only final argmax is insufficient to claim whole logits/state equality.

If qualification passes, compare ON and OFF in three paired independent-boot blocks at B1, with common engine/API seed, pinned weights/image, same prompt list, warmup, budget, backend, scheduler and measurement support. All original cells and failures are retained; no favorable retries or prompt replacement. Report coarse paired uncertainty and complete output-stream agreement separately from rate. This is an incremental optimization comparison within the tree configuration, not a replacement for E1 or a general advantage over native MTP. No B4 or multi-factor campaign is implied by this initial test.

Other optimization families remain implemented designs with explicitly missing attribution evidence. The fixed-shape/FA2/stochastic route changes require their own qualification before any composition study.

## Execution update, 22 September21:52UTC

Both actual qualification arms passed independent review:515paired head comparisons per arm,103qualified proposals,8complete32-ID API streams each, zero timed support. All8streams match across these two boots; this does not prove universal model-state equivalence. Qualification source manifest `bc2feab3`; final pass receipt `815b348b`. The exact timing manifest `8bbb5b48` passed source/CPU/Linux launcher review and was handed to the existing sole Claude GPU worker for6fixed cells. No timing estimate exists yet. The original launcher failure occurred before Docker/model/API work and remains preserved separately.
