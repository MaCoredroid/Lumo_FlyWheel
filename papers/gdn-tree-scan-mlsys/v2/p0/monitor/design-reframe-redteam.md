# Design-reframe red-team — 2026-09-22

**Disposition: PASS for the current design and scoped-evidence claims. No unresolved material manuscript finding.** This is a read-only source/claim review, with only this report written. No inference, GPU work, model boot, campaign change, or source edit was performed. The separately proposed E8 attribution qualification is necessary only for a new component-benefit claim, not for the claims closed here.

## Reviewed identity

Canonical paper directory: `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/papers/gdn-tree-scan-mlsys/v2`.

| File | Final reviewed SHA-256 |
|---|---|
| `main.tex` | `0d7bab3835839559797213a42c892c11e10002d93c23eb3d9897babf9e2cc642` |
| `abstract.tex` | `887f36b63b62f07784a3b9a2f767e9960ca31ea6b4dbb076d031b92ee202180a` |
| `ref.bib` | `64ca0fc74b5185f2f61358945a2f1250de76a7ec7c2ad793504dff0fc22693c0` |
| `notes/optimization-route-audit-2026-09-22.md` | `a667aeb25254714116ba77e1a674ab7d866f5cfc76c03679a70de7903320cc22` |

Initial reframe `077274fd…` was followed by narrow owner edits during review. I re-read the changed passages at the final identity above. The pre-reframe backup is `p0/monitor/design-reframe-before-20260922T204805Z/main.tex`, SHA `09d5fbd626233c5f363aaa94f3fce6dd9e06164d01b419c63d1d1fa8e2285d92`.

For source citations below, `R` is the remote repository `/home/mark/lumo-paper-v2-20260921`; `C` is `R/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T100028Z-e1-18cells/cell_05_b1_tree_B1_a1`; `L=C/loaded_backend`. The current remote dependency hashes still match the frozen E1 `campaign_snapshot/campaign_identity.txt`; the current local kernel differs as a whole, so I independently read the remote campaign-bound source and checked the four relevant function bodies against the local copy.

| Source read | SHA-256 |
|---|---|
| `R/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` | `98f7c8511c1fa991fd36526e69a8b753a5205716bf5b0a53c058177c965cdb20` |
| `R/src/lumo_flywheel_serving/fr13_tree_conv_fused.py` (campaign identity) | `564c7dc7ba61ccd9579f2fb7a4af9a2cdaaa1753dfb98434f2011457d90d9a6e` |
| `R/scripts/fr13_device_multidraft_kernel.py` | `648ff0e33e8a7cdd0fbe8ebb3d2befaa27002a3347f94f9bdc0668365ed2e8e9` |
| `L/eagle.py` | `aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7` |
| `L/gdn_linear_attn.py` | `723974c703d1ad1545399b78526e5d50ae7f7f362e434bad384a0a3327595b28` |
| `L/rejection_sampler.py` | `5be4ad43409da82b63bfd0177926467e9a2b59da553ee6dd04d3130a5d77b12f` |
| `L/gpu_model_runner.py` | `b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40` |
| `L/tree_attn.py` | `a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97` |

## Findings resolved during this pass

1. **Replay work versus selected-path semantics.** The initial `main.tex:178` said work was proportional to accepted path length. The actual kernel unrolls the configured capacity (`fr10_gdn_tree_kernel.py:11829`), performs the shared update (`11881–11896`), then masks inactive-state updates/stores (`11897–11914`). The all-layer sibling has the same pattern (`12932–12998`). Final line 178 now distinguishes a capacity-bounded masked loop from accepted-node semantics; Algorithm 2's caption at 147 explicitly describes logical rather than padded compiled loops. **Closed, no experiment.** The carried-state `O(B_v d_k)` claim remains correct and explicitly excludes operand-ring/serving storage.

2. **Temporary node states versus exported states.** New line 142 correctly describes the cached table, whereas the retained fresh-pilot paragraph originally said the route materialized no per-node state. The kernel allocates `h_cache` at `10618`, while `launch_tree_gdn_prepared` explicitly avoids per-node HBM state export (`16671–16675`). Final `main.tex:290` says no per-node states are exported for the served-state comparison. **Closed, no experiment.** This is the sole Results wording change.

3. **Shared convolution preparation scope.** Preparation is group-owner gated in `L/gdn_linear_attn.py:11185–11208`; `C/docker_logs.txt:209–213` records three state groups. Final `main.tex:197` says indices are prepared once **per state group**, rather than implying one global preparation. **Closed, no experiment.**

4. **Previously identified device-control wording.** `main.tex:194,215` now accurately says device probability arithmetic while retaining Python request/node loops and scalar readbacks. The module has a request loop at `1092`, a node loop at `1116`, and scalar materialization around `1106,1138,1152–1159`. The startup banner at `C/docker_logs.txt:283` incorrectly says no per-node Python loop; the paper does not inherit that banner claim. **Closed in manuscript.** This is not evidence of measured greedy softmax-transfer savings or a fully device-resident walk.

## Claim–source/evidence disposition

| Final passage | Evidence and interpretation |
|---|---|
| Architecture and shared update, `main.tex:87–169` | Parent/path recurrence follows the actual `_gdn_node_step` (`10409–10491`), cached parent state (`10618–10663`), scan call (`10807–10822`) and replay call (`11881–11896`). Pending-token versus materialized-prefix semantics are explicit. Shared source is not equated with compiled bit identity. Existing selected-route state/API checks remain the finite runtime evidence. |
| Workspace/recomputation, `172–187` | `h_cache` has logical `[N_SPAN,BLOCK_V,DIM_K]` fp32 shape. The full-table footprint and values-per-lane estimate are labeled analytic; no allocated-register, spill-free, measured-memory-saving or latency theorem is asserted. Replay has one carried tile plus candidate-scaled stored operands; path-scan and fixed32 implementations exist separately. |
| Active logit reuse, `192,214` | `L/eagle.py:1486–1489` bakes ON unless local-argmax reduction is used; root/loop branches reuse computed logits (`4719–4748,5273–5288`). Actual log `C/docker_logs.txt:280` reports `single_logits=True`, local argmax false. Baked availability is not an OFF/ON ablation or isolated measured gain. |
| Active convolution and packed buffers, `197–199,216–217` | Helper builds static window/state source indices (`fr13_tree_conv_fused.py:91–180`), gathers sources (`182–285`), and uses explicit ordered tap additions/casts (`286–336`). Loaded call sites are `12576–12586,12657–12666`; actual engagement is at log `281`. Shared metadata is group-specific. All-layer replay requires separate arming (`L/rejection_sampler.py:3219–3233`); the unarmed branch iterates registered layers and calls individual replay (`3510–3615`). Stacked rings do not prove replay-launch batching. |
| Fixed32 scan/TAW and FA2 candidates, `202–221` | Implemented source candidates remain distinct from Cat10's generic scan (`L/gdn_linear_attn.py:14765`). Generic TAW/depthsync exclude `all_greedy` (`device_multidraft:1034–1058`); fixed32 and FA2 candidates have other geometry/backend guards, as independently documented in the route audit. Their existence is not credited as activation or speedup in E1. |
| Physical mapping, `226–245` | FA2 spine-first witness is explicitly scoped, while fresh stock TREE_ATTN uses flat slots plus remap. The adverse policy-A first-attention witness and corrected B1/B4 evidence are preserved. No flag-only spine-first ablation is implied to be legal under stock TREE_ATTN. |
| Novelty and external comparisons, `50–85,361–365` | Related Work is byte-identical to the previously verified section. No first hybrid-tree, new recurrence, new reconstruction/replay concept, external-system speed ranking, or refutation of Bole/TreeWY is added. Fresh compact measurements remain local method-family realizations. |
| Final timing and conclusion, `324–372` | E1 is the completed as-executed instrumented configuration comparison. Negative tree-versus-MTP5 results, positive longer-chain comparison, dependent stream-divergence counts, coarse three-block precision, distinct backends and T1 engine-seed deviation remain explicit. Component attribution, broad quality and full-model equivalence are not inferred. |

## Results, historical-number and figure integrity

- Extracted the entire Results section (up to the next section) from the backup and final draft. It is unchanged except the exported-state clarification above. Its numeric token sequence is identical. Original Results-body SHA was `dcdbdcd586a19ca9ed50686209d2052b05b48d0e6d954eb08b89da3a6ff2c224`; final is `983332f2bc31c6024f3d1523e9726b07f33c39f65a13be4d82f5bfd5bd5ab12d`.
- Thus the E7a failed numerical/padding/timing/coverage checks and flawed-freeze disclosure, E7b stage/horizon limits and omitted-remap caveat, adverse policy-A and duplicate-sibling witnesses, bounded B4/compaction/API evidence, and all E1 adverse outcomes survive the reframe. Their independently checked underlying values are documented in the prior final E1 and E2/E7a reports; no numeric reanalysis was needed for unchanged data.
- Historical pre-v2 campaign numbers remain excluded. Qualitative archived mechanisms and citations remain. New arithmetic/storage dimensions are analytic, not resurrected historical performance. The fresh synthetic stress paragraph and fresh pilot/confirmation measurements retain their precision and reference boundaries.
- Loaded figure content is consistent: pipeline mapping is backend-specific; scan/replay separates temporary table and accepted publication; state contract distinguishes pending token; spine diagram is explicitly illustrative; the only empirical plotted rates are fresh E1. No old numerical figure has been reintroduced.
- All 27 cited keys resolve and all 27 bibliography entries are cited. No missing `ref`/`eqref` target or missing loaded figure/input was found. PDF layout inspection remains the parent's separate build/visual-QA responsibility.

| Loaded figure | SHA-256 |
|---|---|
| `figures/pipeline.tikz` | `d66369de7d4ae08ec4fab7467f9ca0b5c9a5d6e759109e51f87db8f2cb1b2892` |
| `figures/state-contract.tikz` | `50d1d7f5d30f4a1ed79ee5634299028d352765a68089bef2c340431d03b46898` |
| `figures/scan-replay.tikz` | `affc55cd6972d929de38dc8ebdd86c6dcea77fb0beb73c96551010bfaffadcfe` |
| `figures/spine-layout.tikz` | `390a6d64b34627e93a2d63cae64d971abe54569be978fa5477984e3228d5579c` |
| `figures/e1-timing.pdf` | `9fc36e22047056751bb540da8ce27983f1683baf4ec69ebad787c72cd06bdf61` |

## Finite experiment decision

No additional experiment is indispensable to the current implementation/design description plus its preserved fresh, scoped results. This verdict does not establish or evade a positive optimization claim: the paper explicitly reports the depth-matched loss and says the optimization candidates have no attributed gain. A stronger claim that a particular optimization reduces work/latency empirically requires an actually engaged, compatible ON/OFF qualification and fresh attribution evidence. The planned single-logits E8 is one bounded route to that claim upgrade; its CPU/source qualification and any later GPU execution must be reviewed separately. A failed invariant should stop timing and remain reportable. No fixed32/TAW/FA2 port or broad new campaign is required merely to describe those implemented candidates accurately.
