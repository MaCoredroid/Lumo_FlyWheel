# Latest-design and supersession audit — 2026-09-22

**Final delta disposition:** the parent has corrected the substantive method-supersession findings and E7a hierarchy in `main.tex` SHA `f093628b147118c87df3203fa76e711e9fc8c71e7733595af918367ed8bf43ac`; the recheck below closes them. E8's completed-campaign result still needs integration after independent raw review, plus two small scope-wording edits. No further experiment is required by this audit. The initial findings and checked identities are preserved below for chronology.

This is a bounded read-only source/document audit. Only this report was written; no serving source, manuscript, frozen experiment, GPU task, or automation was changed. Line references below address the reviewed source identities listed at the end; the parent is editing the manuscript concurrently. No applicable `AGENTS.md` was present in the repository or ancestor directories checked.

## What is authoritative now

Paths are repository-relative unless prefixed with `V`, `D`, or `L`:

- `V = papers/gdn-tree-scan-mlsys/v2`.
- `D = V/p0/monitor/e1-source-20260922T1150Z`.
- `L = V/experiments/out-20260922T100028Z-e1-18cells/cell_05_b1_tree_B1_a1/loaded_backend`.
- The local Pages worktree is `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel-gh-pages`.

1. **Repository serving inventory:** local HEAD `84daa5519ead3caed9cf9a9922e528e64cf73cb1` adds the paper to serving base `origin/main=984f613d7885d9bf7898866827552b68a3199d0c`. Independently ran `git diff --name-only origin/main..HEAD -- scripts src`: empty. The newer integration branch `origin/vllm-upstream/plan-v3=855f6cf03697bc66265e87f2d477074976f76ca2` also has an empty `scripts/src` diff against origin/main. Its September updates do not replace the serving method used by E1/E8.
2. **Qualified and measured E1 route:** actual loaded modules, Docker configuration, qualification receipts, and the immutable campaign identity outrank current source comments or old “canonical HEAD” documents. The route is Cat10 (root plus nine drafts, depth five), stock-image `TREE_ATTN`, flat candidate slots, accepted-path KV remap, synchronous eager execution, prefix cache off, temperature zero, actual B1/B4. The native comparators use the same patched runner with tree work disabled. It is a fixed qualified configuration comparison, not a search for the globally fastest tree or native deployment.
3. **Current source is not byte-identical to E1:** compared both complete files. The frozen patcher differs from current HEAD only by emitting `_FR13_FIXED32_MODE` unconditionally (`D/scripts/fr10_phase4_patch_vllm_tree_gdn.py:26279–26289`); the current conditional definition otherwise leaves the plain tree route with an undefined name. The frozen GDN kernel additionally has `@triton.jit` on `_tree_gdn_replay_all_layers_kernel` (`D/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:12804`). Do not describe HEAD alone as the exact tested implementation. The latter annotation does not establish activation of the separate all-layer replay route.
4. **Latest compatible component evidence:** E8 applies a hash-checked source shim to E1's preserved loaded `eagle.py`, making the previously baked single-logits switch real. Qualification-v2 has passed both actual arms; each arm has 103 complete proposals and 515 paired same-input full-logit/argmax/top-two checks, plus complete candidate and API binding. It is untimed evidence, not a speed result. The six-cell B1 timing campaign is separately frozen and was still running at this audit cutoff. Its prospective primary statistic is mean paired relative change; it must not be conflated with E1's comparator or estimator.

The current design chain is therefore **repository inventory → exact repaired experiment dependencies → emitted/loaded modules → actual configuration and engagement → qualification → each scoped measurement**. A newer document date or an old page's “promoted default” label cannot bypass this chain.

## Concrete manuscript corrections

| Location in reviewed `V/main.tex` | Finding and required disposition |
|---|---|
| 57 | “FlashAttention-2's work partitioning supplies the attention substrate examined here” identifies the wrong active substrate. `L/tree_attn.py:30,1152–1172` imports and calls Triton `unified_attention`, with a tree bias and flat block table. Keep FA2 as related work if useful; identify stock `TREE_ATTN` as the measured backend. |
| 111, 120 | The descriptor section leads with unused heuristic-suffix variants and the forked-FA2 route. Replace those descriptions with the actual native-MTP principal chain plus runner-up candidates, the shared ancestry descriptor, and the current flat-slot attention address contract. A drafter-independent interface is still a valid design boundary; it does not require recounting inactive proposal sources. |
| 192, 214 | The duplicate head is described as work done “solely to recover branch candidates.” In `L/eagle.py:568–572`, `_greedy_sample` recomputes logits to obtain the **principal spine argmax**; both branches already compute primary logits for branch candidates. The reused branch takes the argmax from those existing logits (`4747–4785`, `5320–5329`). Say that one full head result supplies both spine selection and runner-up candidates, avoiding the legacy duplicate head for spine selection. The old “no isolated head-work ablation” status must be replaced by E1's baked-ON boundary and E8's actual qualification/final timing status. |
| 227–245 and `figures/spine-layout.tikz` | Remove the archived remap bug, missing-launcher-flags story, and FA2 spine-first repair/witness from the main method. They are already scoped as history, so this is chiefly compliance with the requested current-method focus; no historical finding needs to be erased from artifacts. A spine-first layout is specifically incompatible with the present `TREE_ATTN` read/write contract under the tested settings. Replace with flat verification slots, copying selected node KV into linear continuation positions, and shared accepted-path publication across recurrent/conv/KV state. Retain the fresh E2 incompatible-policy witness and its precise backend boundary. |
| 247–251 | Remove the old local WY prototype bug and archived layer-localization narrative. Retain the generic distinction between verifier output, committed state, later-layer propagation, and future continuation, plus the explicitly named native rounding contracts at 253. **Do not remove fresh E7a's triangular/Neumann comparison merely because it belongs to the same algebra family as an older prototype.** It is a distinct, freshly executed characterization. |
| 262–273, 365, 372, 400–418 | The experiment inventory and “remaining component ablations” discussion omit E8 and still foreground a speculative composition campaign. Add the completed bounded E8 qualification and, only after closure, all six timing cells with the frozen primary/diagnostic estimators. Keep untested composition claims absent; a broad new E3 campaign is not required to report this one compatible component ablation. |
| 35, 191, 368; `abstract.tex:2`; pipeline label | “Device acceptance” is shorthand that can imply more than the carefully corrected explanation at 194. Prefer “device probability arithmetic” in the summary/diagram, or explicitly qualify host request/node control and scalar readbacks. This is a consistency correction, not a newly discovered execution change. |
| Current-method uses of `lumo2026receipts`; `ref.bib:124–129` | The cited receipt snapshot is `55f5585…`, while `lumo2026archive` at `ref.bib:234–239` points to `984f613d…`. Current inventory descriptions should use the latter or the actual frozen artifact identity; old receipt citations should remain only for explicitly historical statements. Neither public Git pin alone identifies the two repaired E1 dependency files. |

The parent's proposed removal of the old FA2 and WY repair narratives is safe. Their necessary methodological content survives in the current address contract, explicit finite-precision reference, and fresh qualification/failure evidence. No mechanism in the current selected route requires retaining the old layout figure or repair chronology in the paper body.

## E7a: final confirmation must lead

The reviewed paper **does disclose** the failed confirmation and original-checker defects; the issue is emphasis and temporal hierarchy, not concealed data. At `main.tex:290`, the earlier eight-prefix pilot is introduced with all-eight native/deployed-store byte agreement. The subsequent confirmation has 31/32 prefixes passing the frozen provenance gate (`292`), two deployed/native bfloat16 mismatches, other numerical/control failures, and failed timing checks (`294–298`). At `300` the narrative then returns to pilot-only local timing ranges.

Recommended order: open E7a with **31 of 32 provenance-passing confirmation prefixes, frozen numerical/control/timing criteria not satisfied, no compact candidate promoted**. Then explain the eight-prefix pilot's role in prefix selection and the flawed frozen analysis. Preserve the p087 denominator and retrospective inspector correction; do not replace the frozen verdict with the repaired check. The pilot's detailed passing output claims and timing ranges can move to the artifact or be compactly marked developmental characterization. This prevents an earlier successful subset from appearing to supersede the later failure. The fresh synthetic tiny-gate control tests a separate numerical condition and is not invalidated merely by being earlier.

The six E7b diagnostic boots at `304–310` used no attention-KV remap, as the paper already states. Their layer-local substitutions remain valid scoped observations, but they are **not the current qualified continuation route**. Prefer a compact statement of that local numerical question; move harness-repair chronology into the artifact. Preserve the finite negative finding and the lack of full-model equivalence. The corrected selected-route checks at `314–322` are the current serving-contract evidence.

## Current mechanisms that must remain accurately mapped

| Mechanism | Current E1/E8 status and primary source pointers |
|---|---|
| Branch-local GDN scan plus accepted-path replay | Active. Cat10 uses the ordinary per-request monolithic scan, not the fixed32 folded candidate (`L/gdn_linear_attn.py:14632–14648,14765–14848`; `D/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:17356–17362,17763–17765`). Replay adds separate selected-path work. `figures/scan-replay.tikz` and `state-contract.tikz` describe the present lifetime correctly. |
| Single-logits drafter | Active, baked ON in E1 (`L/eagle.py:1486–1495`); E8's explicit source ON/OFF and head census now provide the compatible attribution experiment. E8's same-input correctness gate is stronger than the legacy token-only selfcheck; complete cross-boot stream equality is reported descriptively rather than imposed as a retention rule. |
| Device multidraft | Active, but request/node loops and scalar reads remain (`D/scripts/fr13_device_multidraft_kernel.py:1075–1152`). Greedy uses canonical multidraft (`L/rejection_sampler.py:3762–3775`); do not revive zero-host-loop claims from old engagement text. |
| Vectorized tree convolution | Active: baked flag and runtime branch at `L/gdn_linear_attn.py:8514,11877–11904`; first tree B1/B4 logs both report `fused=1 tree_n=10 width=4 state_len=12`. Current paper 197–199 correctly includes it. No fresh isolated fusion gain is established. |
| EAGER_PACK | Active stacked operand storage/preparation (`L/gdn_linear_attn.py:8501,11797,11913–11918`; boot logs report union of 48 layers). The canonical committer's separate all-layer replay guard at `L/rejection_sampler.py:3219–3232` is unarmed in E1. Per-layer replay remains at `3517–3522,3559,3615–3630`. Current paper correctly avoids inferring all-layer launch batching from allocation. |
| Flat attention slots and sync-free selected-path remap | Active policy B: `FR13_ATTN_KV_REMAP=1`, `FR13_SLOT_REORDER=0`, `FR13_KV_REMAP_SYNCFREE=1`. This should replace spine-first layout as the main attention implementation. Current E2 helper and actual B1/B4 checks are scoped evidence; live conv/KV snapshots and general full-model equivalence remain unestablished. |
| Fixed32 fold, TAW/depthsync, forked-FA2 grouping, drafter/whole-region graphs, truncated-head/split-K variants | Implemented inventory on other guarded routes, not newly measured E1/E8 components. Fixed32 needs actual32/pad32; the TAW/depthsync branches require non-greedy entry, while the present route is all-greedy. FA2 grouping requires its separate fixed32 B4/dual-gate route (`scripts/fr13_launch_forked_fa2_tree_server.sh:1699–1700,1745–1759`). These are not superseded out of existence, but cannot be credited as active current-route optimizations. A short guarded inventory is acceptable; none requires a port or new experiment merely to remove obsolete main-text claims. |

The older “burn every scratch byte” design is not necessary to state the current contract. `FR13_BURN_REDUNDANCY_VERDICT.md:7,12` identifies why the running-row route does not consume retired node columns. Current `main.tex:108,144` correctly distinguishes retirement from mandatory zeroing. Keep that correction; do not restore the older site's stronger cache-by-construction equivalence language.

## Local Pages and newer integration work: boundaries

The Pages worktree is commit `11b0eeb0d345b03d35fe6d3ebdf4cba09b4c2d15`. Its dated volumes document previous configurations; they are not an authoritative description of the current measured route.

- `gdn-tree-scan.html:1254–1256,1286–1290` foregrounds FA2, served-realization equivalence, and a “distribution-preserving within the native recurrent-oracle floor” definition. Those stronger claims must not be imported into current v2: present evidence is bounded greedy/continuation qualification, with observed stream divergence and unestablished stochastic/full-model equivalence.
- `stateless-tree.html:1288–1291,1321` makes burn-to-zero and cache-divergence-impossible claims. Current v2 appropriately replaces them with a state-lifetime contract and finite coverage.
- `spine-reorder.html:1288,1321` concerns the archived FA2 layout repair. It does not authorize `SLOT_REORDER=1` in stock `TREE_ATTN`.
- `every-lever.html:1288–1289` reports drafter graph, reduced head, batched replay, and other old optimization outcomes. Its own `1360,1388` discuss sampling and instrumentation reversals. Keep these historical quantitative claims out of v2 and do not mistake their source availability for E1/E8 engagement.
- The newest volume, `only-quantization.html:1265–1268,1313,1457,1531`, concerns Qwen3.8/NVFP4 and a separate B1 split-K route, explicitly with replication owed and no served B4 result. It does not supersede the pinned FP8 system comparison or require an NVFP4 rerun to make this paper current. If mentioned, it is an unqualified/unmatched extension outside this paper's measurement claim.
- On `origin/vllm-upstream/plan-v3`, `docs/vllm-upstream/PLAN_V3.md:16–20,51–56` explicitly treats tree work as a record and separates upstream defect work. `COMMENT_53142_REPLY_DRAFT.md:1–15` describes an external adaptation of the restore-fidelity fixture, a bug caught in #55507 and external fix `adc7d30`; the posted scope says the adapted arms/fix were **not run by us**. #54076's September configuration reproduction is likewise newer evidence for a different upstream cache/geometry surface. Neither update changes the scripts/src serving design or qualifies prefix caching in E1/E8. Do not claim the external fix was integrated or count these developments as new measured tree results.

Some root documents also retain explicitly dated “canonical” labels: `FR13_PIPELINE_LOCK.md:3–29`, `FR13_KERNEL_STATUS.md:4–38`, and `docs/unified-tree-spec-decode.md:3,40` describe earlier locked FA2/kernel or drafter variants. Their chronology is useful provenance. Actual loaded source/configuration must control the current paper description.

## E1 and E8 answer different questions

Keep E1's complete 18-cell comparison, native5/native11 distinction, all initial cells, coarse uncertainty, observed continuation divergence, and engine-seed deviation. Current `main.tex:259,328–355` correctly restricts it to fixed qualified eager configurations. E8 compares only single-logits ON/OFF **within the same tree route at B1**, with a common engine/API seed and separate six-cell design. It cannot retrospectively repair E1's seed mismatch or prove equivalent continuation quality. Conversely, E1 cannot supply E8's OFF control. Report the complete closed E8 result whether small, negative, or imprecise; no precision-driven extension or favorable-repeat policy is needed.

No additional GPU experiment is necessary for the editorial/source corrections in this report. Completing and independently reducing the already frozen E8 campaign is the only outstanding experiment needed for the new head-reuse attribution claim. Broader ports, stochastic routes, cache-hit routes, external authors' systems, and composed optimization sweeps are unnecessary unless the manuscript elects to assert those stronger claims.

## Checked identities

```text
c842d7ccef2f5a5b389cd02dd1d55825747d92be7866a5c14901dcc06e502722  V/main.tex
887f36b63b62f07784a3b9a2f767e9960ca31ea6b4dbb076d031b92ee202180a  V/abstract.tex
64ca0fc74b5185f2f61358945a2f1250de76a7ec7c2ad793504dff0fc22693c0  V/ref.bib
d66369de7d4ae08ec4fab7467f9ca0b5c9a5d6e759109e51f87db8f2cb1b2892  V/figures/pipeline.tikz
50d1d7f5d30f4a1ed79ee5634299028d352765a68089bef2c340431d03b46898  V/figures/state-contract.tikz
affc55cd6972d929de38dc8ebdd86c6dcea77fb0beb73c96551010bfaffadcfe  V/figures/scan-replay.tikz
390a6d64b34627e93a2d63cae64d971abe54569be978fa5477984e3228d5579c  V/figures/spine-layout.tikz
e40b56950ca6e4c974295978eced4d5a499a6d2e7125b17eaf4201acec9995ed  V/notes/design/optimization-evidence-map.md
c382ef22a5a8d4e84209e6723f8eea21c3819543c6e942c62951385b5c1c003e  scripts/fr10_phase4_patch_vllm_tree_gdn.py
d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8  src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
a7f8f0943d1a13ac4742e824ea4e126c6ff341342aba677db5169e712c7e7281  D/scripts/fr10_phase4_patch_vllm_tree_gdn.py
98f7c8511c1fa991fd36526e69a8b753a5205716bf5b0a53c058177c965cdb20  D/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7  L/eagle.py
a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97  L/tree_attn.py
e8e02700cad9337bfa61f1446b3f1b090404f144c8acfe193fbad197291cfca3  V/p0/monitor/e8-qualification-results-redteam.md
0d94d655bd7ced238f19a0c6bd0a5e23deadf57346b0135d5933db1bdc201c54  Pages/gdn-tree-scan.html
708b425ef6888472c996f9dc89c8590248610b1ba9689398652068cf4f80c0c9  Pages/stateless-tree.html
e5d0f539f9e7baf92ed2702c93ef422bd7cdb92e89045ff663a9ab5f5f4d608f  Pages/spine-reorder.html
b34da0912c4a0264a9eebfd986428cc4f77a91ae9a7c683d48e990004255dc01  Pages/every-lever.html
dbfe085a657a977263aec1d9dcb22e4494f328681d2dfcc7cdbec2d31ff5d3d4  Pages/only-quantization.html
```

E8 immutable identities read through the independently reviewed actual receipt: qualification-v2 manifest `bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030`; actual final qualification receipt `815b348bdb95c455122e7f1232449d4e12dd339edbd6c3c8e2b9a980c6342513`; timing manifest `8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1`. No partial timing result was used in this audit.

## Final delta recheck after the parent's current-method revision

Reviewed updated `main.tex` SHA `f093628b147118c87df3203fa76e711e9fc8c71e7733595af918367ed8bf43ac`, `ref.bib` SHA `88ceccadbd0376d694f6037181ebf0dacd71f0936558d6c88f856666ccc05fc8`, and unchanged abstract SHA `887f36b63b62f07784a3b9a2f767e9960ca31ea6b4dbb076d031b92ee202180a`. Only text/source reads were used.

**Closed:** wrong examined FA2 substrate (`57`); heuristic suffix source in the current pipeline; archived launcher/remap/WY repair narratives and included spine-layout figure; duplicate-head purpose (`192,214`); old current-method receipt citations; pilot passing output and timing ranges. The current flat-map mechanism is stated directly at `120,227–229`. E7a now opens with 31/32 provenance and failed promotion criteria (`264`), explains the pilot only as freeze provenance (`274`), and preserves all later failures and checker corrections (`276–282`). Current implemented but inactive FA2/fixed-shape candidates remain explicitly guarded inventory, without gain claims. The fixed qualified E1 scope is correctly preserved (`241,308–337`).

**Remaining integration items, sent to the parent:**

1. E8 is acknowledged in optimization row `214`, but its complete result, evidence table, discussion/conclusion, and validation scope must be integrated together after the independent raw reduction. The parent reports all six cells closed around 22:40Z; this audit did not read the final timing archive or infer an effect from partial data. This is a pending result integration, not a reason to reopen or supersede E1.
2. Align summary shorthand with the correct body limitation: `abstract.tex:2`, `main.tex:35,42,350`, and the pipeline label still use “device-side/device acceptance.” Prefer “device probability arithmetic,” or explicitly acknowledge the retained host control. Section `194` already states the exact boundary correctly.
3. At `main.tex:292`, the E7b diagnostic paragraph ends “qualification must first repair and check that route,” while `296–304` subsequently reports the corrected selected-route qualification. Name this as the **earlier, unremapped diagnostic route** and point to the subsequent correction. This avoids an obsolete pending status without upgrading the diagnostic boots into correct-attention continuation evidence.

No other material current-method supersession mismatch was found in this bounded delta recheck. All three remaining items are manuscript integration/wording; none requires new GPU work or broader qualification.
