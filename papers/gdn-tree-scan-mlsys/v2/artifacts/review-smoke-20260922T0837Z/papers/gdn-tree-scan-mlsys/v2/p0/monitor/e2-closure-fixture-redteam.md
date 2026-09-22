# E2 closure fixture red-team

Read-only bounded review, 2026-09-22. Source of record: `mark@100.103.10.122:/home/mark/lumo-paper-v2-20260921`. No GPU, inference, container launch, tmux interaction, or source edits. Only this local report was written. Paths below are relative to that remote repository.

**Disposition: useful partial closure; not yet whole-model E2 qualification.** The new tests catch missing convolution publication and missing KV copies. They do not yet exercise the selected production KV variant or the actual pending-token/next-consumer composition. Three concrete fixture gaps remain; no broader experiment sweep is needed.

## Executed evidence

The requested test hashes matched exactly and remained stable across the checks. The patcher and kernel hashes also remained stable.

- An ordinary host invocation of the two pytest files stopped during collection: the KV test imports the full kernel module, and host Python has no `triton`. This is an environment limitation, not an observed kernel failure.
- A memory-only module containing the **exact AST-extracted** `launch_attn_kv_linear_remap` and `_launch_attn_kv_linear_remap_impl`, with timing observers replaced by no-ops, allowed both files to run on CPU: **13 passed in 0.07 s**. This validates the extracted Torch helper tests, not full-module/container loading.
- Negative control: replace the extracted convolution committer with a no-op; `test_col0_receives_committed_leaf_window_for_every_case` rejects request 1's branch window.
- Negative control: skip KV copies but return the expected engagement count 5; `test_b4_cohort_branch_spine_zero_and_prefix_exact_slots(7)` rejects whole-bank inequality.
- Direct storage identity check: convolution and SSM tensors from `_commit_and_gather` have **different storage pointers**.

| Claim/surface | What these files establish | Remaining original-surface gap |
|---|---|---|
| Conv publication → column-0 gather | Actual extracted committer and actual extracted gather compose correctly for one B4 cohort: zero, branch, depth-5 spine, prefix; disjoint permuted physical rows; non-destination conv rows unchanged. | Direct calls bypass the replay-route call site; the subsequent consumer is a test helper. |
| Shared-page conv/SSM safety | Conv logical rows outside destinations remain unchanged. | The SSM assertion uses an unrelated allocation, so cannot detect corruption of co-resident SSM/page padding. |
| KV accepted-node copies | Legacy synchronous helper performs expected snapshot-source copies on permuted slots, B4 and B1 depth-5 spine, with full-bank equality and meaningful negative control. | Current runner selects sync-free code, and canonical slot reorder requires `dst_pi` plus mapping restoration. Neither is called. |
| Pending token / next consumer | Test-only window construction appends a supplied random bonus once; KV tests inspect post-copy markers. | No production pending-token selection/consumer or next-forward/drafter mapping is executed. |

## Required bounded fixes

### 1. Test the selected KV implementation and its mapping lifecycle

`tests/test_fr13_attn_kv_remap_pytest.py:1–2,19,72–74` calls and labels `launch_attn_kv_linear_remap` as the loaded route. The current patcher instead selects `launch_attn_kv_linear_remap_syncfree` through an unconditional `elif True` at `scripts/fr10_phase4_patch_vllm_tree_gdn.py:41913–41927` when the overlap branch is inactive. The legacy branch at 41929–41937 is unreachable in that case.

For the intended canonical configuration, `scripts/fr13_required_tree_flags.sh:26–31` and `scripts/fr13_canonical_env.sh:24–27` select remap=1, slot-reorder=1, sync-free=1. The remap call receives `_fr13_sr_pi_t` as `dst_pi` (patcher 41770–41780). The new tests pass no `dst_pi` and do not exercise slot-map permutation/restoration. The sync-free implementation has separate fixed-shape masking and copy logic (kernel 10200–10275, 10325–10344); legacy helper success alone does not execute it.

**Minimum:** retain the current independent markers/oracle, but run the selected sync-free function with the actual cat10 `pi`, permuted verify mapping and destination correction. Reuse the same B1/B4 cases. Bind the selected emitted call and restoration before the next reader/drafter to this composition. Assert committed K/V through the restored next-consumer mapping, including untouched other-request slots; use an actual multi-block layout for the existing page-permutation check. Its current `kv.shape=(2,1,256,H,D)` (`_marker_kv`, lines 25–33) permutes offsets inside one block and does not test block-index arithmetic.

This corrects the earlier `e2-minimal-qualification-review.md` suggestion that slot reorder can simply be left off: it is part of the **currently selected canonical contract**. A flat-map alternative may be a different expressly selected and qualified route, but it is not the selected production route's evidence. Do not add optimization A/B arms or import unrelated graph/APC defaults into this narrowed eager/cache-off E1 plan. If overlap is selected later, bind the prepared-slot/apply branch instead; do not demand it while disabled.

### 2. Use real shared-page storage for the convolution safety assertion

`tests/test_fr13_conv_col0_publish_read_composition.py:46,65–67,101–108` allocates conv and SSM separately; SSM is never supplied to the operation. Thus “an SSM bank sharing the test's memory model” in lines 10–11 is unsupported.

**Minimum:** reuse `_page_shared_bank` from `tests/test_fr13_replay_conv_remap_page_safe.py:32–63`, whose conv and SSM are strided views into one raw page with padding. Run the same publication/gather cohort and check all bytes outside the intended conv destinations. Preserve the existing no-publication negative control. This is a fixture repair, not a demand for another model run.

### 3. Bind the next production convolution consumer and pending-token identity

`test_pending_bonus_token_consumed_once_by_next_consumer` (lines 111–122) invokes `_node_window_row` imported from another **test file** (line 119). Both sides explicitly append the supplied bonus once. This checks the window identity but cannot detect a production consumer that skips/doubles the token, reads a stale prior, or uses a different pending token.

`_commit_and_gather` only clears `FR13_TREE_RUNROW_INIT` to select its default (line 62); it does not execute the `REPLAY_ROUTE=1` entry/call-site. The extracted helper composition is valid evidence for those helpers, but its docstring must not imply a served route was exercised.

**Minimum:** extend this same fixture with the production prepared/gather/window consumer used by the selected route, or an extracted emitted next-forward fragment, using the published column-0 prior. Bind the next root to the committer's pending correction/bonus identity, assert the expected state/window after exactly one consumption, and retain a skip/double-consumption negative control. Reuse the already planned fused-conv B4 byte gate for the GPU numerical portion; do not treat the test-only `fused_tree_conv_layer` driver as proof of loaded call-site engagement without its wiring/boundary evidence.

## Qualification boundary / stop rule

The two new files can close their helper scope after these bounded repairs. They do not release E1 by themselves. Reuse the planned untimed final-configuration B1/B4 preflight to establish actual remap engagement, fresh request/path alignment, publication before overwrite, restored next-reader mapping, pending-token identity, and state linkage. No extra performance cells are required. The helper's mixed-span no-op test (KV lines 137–144) establishes no writes in that case; it does **not** establish correctness of serving mixed prefill/decode through that no-op. Match or explicitly reject that transition in the existing preflight.

No additional architecture, topology, stochastic sampler, cache-hit/eviction, or quality sweep is requested. Once the selected fixtures and existing served-boundary gates pass with stable loaded hashes/flags, stop this fixture review. Whole-model E2 and E1 measurements remain distinct from prior layer-62 fixed-input diagnostics.

## Snapshot identities

| File | SHA-256 |
|---|---|
| `tests/test_fr13_conv_col0_publish_read_composition.py` | `8ea2cbff7b2f17bbb215dc5b52f4698012bfa4144ee0102472a75dc003156650` |
| `tests/test_fr13_attn_kv_remap_pytest.py` | `da90dda56703aa9e2fe72b287691372f299cdd9a8e5e2fd8e796bdcd862f1a05` |
| `scripts/fr10_phase4_patch_vllm_tree_gdn.py` | `a7f8f0943d1a13ac4742e824ea4e126c6ff341342aba677db5169e712c7e7281` |
| `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` | `98f7c8511c1fa991fd36526e69a8b753a5205716bf5b0a53c058177c965cdb20` |
| `src/lumo_flywheel_serving/fr13_tree_conv_fused.py` | `564c7dc7ba61ccd9579f2fb7a4af9a2cdaaa1753dfb98434f2011457d90d9a6e` |
| `scripts/fr13_required_tree_flags.sh` | `74de72c709d5eba3a00146056f2af38c4d8b443b6925c5a8e4feb7fbe8e95e21` |
| `scripts/fr13_canonical_env.sh` | `b81e8d2f6d3a63e68e365d69ef56a4e62f1fc184c42eb9664852a06b2b1662fa` |
| `tests/test_fr13_replay_conv_remap_page_safe.py` | `18902b821c7d3e21aefdeaac44aabff7c39fd8be605e0254f34c83ba6e8c9f1f` |
| `tests/test_fr13_conv_committed_path.py` | `bf7ce331752a499810ee8f7f9299de685c1fb1c3a7c0ee3a5580435d145fb514` |
| `papers/gdn-tree-scan-mlsys/v2/p0/monitor/e2-minimal-qualification-review.md` | `e21ebddeeaec7f9766642f18fb4f735c923ef95f8134326cbc67797f06e55d91` |

## Reproduce the CPU extraction check

Run from the remote repository root with `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -B`. This intentionally avoids importing Triton and does not represent a full-module or pinned-container execution.

```python
import ast,hashlib,json,os,sys,tempfile,types
from pathlib import Path
import torch,pytest
files=["tests/test_fr13_conv_col0_publish_read_composition.py","tests/test_fr13_attn_kv_remap_pytest.py","scripts/fr10_phase4_patch_vllm_tree_gdn.py","src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"]
before={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
# Host has torch but not triton. Install a memory-only module with exact source
# of the two CPU torch functions; timing observers are inert in this harness.
name="lumo_flywheel_serving.fr10_gdn_tree_kernel"
shim=types.ModuleType(name)
shim.__dict__.update(torch=torch,_fr13_span_begin=lambda *a:None,_fr13_span_end=lambda *a:None)
tree=ast.parse(Path(files[3]).read_text())
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {"launch_attn_kv_linear_remap","_launch_attn_kv_linear_remap_impl"}]
assert len(selected)==2
exec(compile(ast.Module(body=selected,type_ignores=[]),files[3],"exec"),shim.__dict__)
sys.modules[name]=shim
with tempfile.TemporaryDirectory(prefix="e2-redteam-") as td:
 rc=pytest.main(["-q","-p","no:cacheprovider","--basetemp",td+"/pytest",*files[:2]])
print("CPU_EXTRACTED_HELPERS_EXIT",rc)
conv=sys.modules["test_fr13_conv_col0_publish_read_composition"]
kv=sys.modules["test_fr13_attn_kv_remap_pytest"]
results={}
orig=conv._load_conv_committer
conv._load_conv_committer=lambda: (lambda *a,**k:None)
with pytest.MonkeyPatch.context() as mp:
 try:conv.test_col0_receives_committed_leaf_window_for_every_case(mp); results["conv_skip_publication"]="NOT_REJECTED"
 except AssertionError as e:results["conv_skip_publication"]="rejected: "+str(e)
conv._load_conv_committer=orig
origkv=kv.launch_attn_kv_linear_remap
kv.launch_attn_kv_linear_remap=lambda **kw:5
try:kv.test_b4_cohort_branch_spine_zero_and_prefix_exact_slots(7);results["kv_skip_copy_keep_engagement"]="NOT_REJECTED"
except AssertionError:results["kv_skip_copy_keep_engagement"]="rejected by whole-bank equality"
kv.launch_attn_kv_linear_remap=origkv
with pytest.MonkeyPatch.context() as mp:
 data=conv._commit_and_gather(mp)
 results["conv_and_ssm_share_storage"] = data[3].untyped_storage().data_ptr()==data[5].untyped_storage().data_ptr()
after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
print("CONTROLS",json.dumps(results,sort_keys=True))
print("HASHES",json.dumps(after,sort_keys=True))
print("STABLE",before==after)
```


## Addendum: fused-conv failure diagnosis

Bounded read-only follow-up, 2026-09-22. No GPU execution or source changes by this reviewer. Existing GPU logs are under `experiments/e2/baseline_subset_20260922/` (relative to paper v2).

**Finding: the observed full-pipeline failure is a state-input/oracle semantic mismatch, not evidence of a floating-point tolerance failure. It affects the selected cat10 test cases, so their qualification cannot be declared passed yet, but these logs do not establish a defect in the current production fused arithmetic.**

### Exact affected cases and evidence

- `fused_conv_head_check_080300.txt:4–21,22–39`: the unmodified HEAD test collected 286 cases and reported **16 failed, 269 passed, 1 skipped**, both with the worktree kernel and HEAD kernel. Failures comprise six T4 cases (B1/B2 × branchy/cat9/chain5), nine T5 cases (seeds 0/1/2 × those three topologies), and cat9 B2 T7. T4 fails on **read-column integer equality**, before convolution arithmetic (`fused_conv_head_check_080151.txt:5–9`; test line 503 in the original snapshot).
- The test's `cat9` topology is nine draft nodes plus root, i.e. the selected **cat10**, with parent list `[-1,0,1,1,2,2,4,4,6,6]`. Thus selected B1 cat10 seeds 0/1/2 fail in the log; this cannot be dismissed as an unrelated topology.
- `e2_closure_b4_fixtures_080006.txt:7–25,259–269` additionally records the new cat10 B4 full-pipeline failure; it fails bitwise output equality.
- Most decisively, `fused_conv_runrow0_check_080600.txt:4–7` records the same HEAD suite at `RUNROW_INIT=0`: **285 passed, 1 skipped**; at explicit `RUNROW_INIT=1`: the original **16 failed, 269 passed, 1 skipped**.

### Cause and independent CPU isolation

In `src/lumo_flywheel_serving/fr13_tree_conv_fused.py`, the **test-only** `legacy_gather_committed_path_conv_prior_reference` (552–583) unconditionally chooses the old accepted-leaf column. `legacy_tree_conv_layer_reference` calls it at 615–625. By contrast, the actual prepared helper chooses column 0 under the served `RUNROW_INIT=1` default (401–407). The actual non-prepared production gather in `fr10_gdn_tree_kernel.py:10347–10405` also honors that same column-0 default. They are not comparing the same prior state.

A CPU-only in-memory isolation ran cat10 B1/seed0 and B4/seed7 with width4/state_len12/dim24. Both arms used the same CPU SiLU; CUDA synchronization was replaced with a no-op. This isolates data movement and is **not a GPU arithmetic qualification**. The production source was unchanged:

| Mode | B1 reference/prepared columns | B4 reference/prepared columns | Output + whole-bank bit equality |
|---|---|---|---|
| Original reference, RUNROW=1 | `[9]` / `[0]` | `[9,0,8,0]` / `[0,0,0,0]` | Both fail |
| Original reference, RUNROW=0 | `[9]` / `[9]` | `[9,0,8,0]` / same | Both pass |
| Only reference changed in memory to independent column-0 selection, RUNROW=1 | `[0]` / `[0]` | all 0 / all 0 | Both pass |

The independent replacement only constructs `cols=zeros((B,1))`, gathers those rows from `spec_state_indices`, and index-selects the prior bank; it does not call the prepared function. A separate exact AST extraction of the **actual production gather** agreed in columns, rows and bank values with the prepared helper under RUNROW=1 on disjoint permuted rows, a valid depth-5 path and zero acceptance. This corroborates a stale reference rather than different production fused/unfused state semantics. Edge-value inputs include NaNs, so the isolation used integer-view equality, not a numerical maximum or relaxed tolerance.

### Minimum disposition

1. Preserve the old logs and legacy reference as historical evidence. Explicit RUNROW=0 is appropriate for a test labeled **legacy semantics**; it must not alone close the current default route.
2. Add/bind an independent **current column-0 reference** to the selected cat10 T4 and full-pipeline B1/B4 comparisons, preserving integer-view output, row and whole-bank/page checks. Reuse the original selected cases; no unrelated topology repair or tolerance relaxation is needed. Keep a wrong/stale-prior negative control.
3. While correcting the B4 fixture, align its accepted-path metadata with its served-case description. At the reviewed snapshot, `_run_pipeline_pair` stores the root-inclusive spine `[0,1,2,4,6,8]` with length6 and sets its “one-node sibling” from `branch[0]=0` (original lines 577–590). Actual committer metadata excludes the already consumed root and has maximum accepted-draft length5. These values are usable as an arbitrary-index legacy equivalence probe, but not as the claimed served B4 acceptance cases. Use the valid root-excluding paths in the existing selected-route composition.
4. Owner should rerun only the corrected selected cat10 gate in the pinned image, then use the already planned served-boundary preflight. Stop after these pass with the intended loaded source/flags. This reviewer has not executed or declared those repaired GPU gates passed.

**No justified unfused environment fallback:** the current patcher emits `if True: # FR13_TREE_CONV_FUSED baked ON` (11596) and `_FR13_TREE_CONV_FUSED = True` (11901). Setting `FR13_TREE_CONV_FUSED=0` alone does not select an unfused route. Restoring a real fallback would require a source change plus its own matched-state qualification; the available evidence instead supports the smaller reference/fixture repair. Do not turn off RUNROW_INIT to make the selected-route gate green; production stateless-tree requires running-row commit/init together.

### Additional provenance

- Original working test used by the B4 failure log: `tests/test_fr13_tree_conv_fused_byte_ab.py` SHA-256 `bde4600e7b4156f43acf36d075f05312f20b58d21ad612d28f65c53f1ca7f83d`.
- Unmodified HEAD test from `git show HEAD:tests/test_fr13_tree_conv_fused_byte_ab.py`: `7022f3a7a6a1317862200d05239998091e5217b2c6ec08a95e238515160c701f`.
- HEAD kernel from `git show HEAD:src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`: `d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8`.
- Test during CPU isolation: `5db8f05dbbb92196e750e5f544fb0061de362b8583befab95e2ab5d87af2598e`; it and fused helper `564c7dc7ba61ccd9579f2fb7a4af9a2cdaaa1753dfb98434f2011457d90d9a6e` were stable across execution. The owner continued editing the test afterward (later observed `4bd658444332e63ed9c5a75874772e1a578bc5c6bc5ab736564247613fe6e0fb`); that newer revision was not re-executed or certified here.
- `e2_closure_b4_fixtures_080006.txt`: `2238ec079218d82b3cd61cec94edd96c464c3a44bf387f023b0c36de5d944fef`.
- `fused_conv_head_check_080151.txt`: `5e9c051c65fe0f4a8e6b744e2c560bc010d280a17dd5ed6f5a443090a4f03b7d`.
- `fused_conv_head_check_080300.txt`: `b15459ad3599d6af9ee9df3e21a945ef64849c609cfa931241ce6c79dbf332d2`.
- `fused_conv_runrow0_check_080600.txt`: `7c00e4949c9f332e7d42331d19f287c8221b7bda0c7b5c0773ca150d575362ee`.


## Settled-source recheck: bounded closure status

2026-09-22 follow-up after the owner reported the selected fixtures settled. **Two original fixture gaps remain; the fused-conv reference mismatch is resolved for the documented helper gate.** Actual B1/B4 served capture gates remain pending and were not inspected or executed here.

### Exact sources reviewed and executed

These full hashes were stable before/after the CPU execution:

| Source | SHA-256 |
|---|---|
| `tests/test_fr13_conv_col0_publish_read_composition.py` | `9f3101d827450b6b589fecdc16fd34b1ebfdb11b429be69ce3a37eda480e6f1b` |
| `tests/test_fr13_attn_kv_remap_pytest.py` | `6b169a286c3dac8c7cc0191a27c0e9e80dd068deb379c4029fe6240b137848a6` |
| `tests/test_fr13_tree_conv_fused_byte_ab.py` | `987fd8b20ca2298a2bba0073e82d87efbb0fc653183679206e35f8da7caabcdf` |
| `tests/test_fr13_eager_pack_replay_byte_ab.py` | `6ee62d581f18c8a703e3614ff9df682ee6679b859afd5433120c312af4b4c45c` |
| Kernel module | `98f7c8511c1fa991fd36526e69a8b753a5205716bf5b0a53c058177c965cdb20` |

Unchanged supporting source: fused module `564c7dc7ba61ccd9579f2fb7a4af9a2cdaaa1753dfb98434f2011457d90d9a6e`; patcher `a7f8f0943d1a13ac4742e824ea4e126c6ff341342aba677db5169e712c7e7281`.

As before, host Python lacks Triton. I extracted the six exact Torch KV wrapper/implementation/prepare/apply functions into an in-memory module, with `_FR13_FIXED32_MODE=None` and inert timing observers. CPU execution of both revised fixture files, T4b B1/B4, and T4 cat9 B1/B2 returned **17 passed in 0.22 s**. No GPU function ran.

### Closed or properly narrowed

- **Fused-conv stale reference:** the independent active-row reference and autouse binding (fused test 114–143) preserve the remaining reference operations and use column 0 under RUNROW=1. T4 calls this independent reference directly. The old failing logs remain preserved. This closes the identified mismatched-input explanation; it does not relax numerical equality.
- **Root-excluding metadata:** T4b now uses valid root-excluding accepted paths at 716–740. Added B4 rows in the numerical pipeline use zero/depth5/one-node accepted paths at 625–635. Original row 0 remains root-inclusive and is now explicitly labeled legacy arbitrary-index arithmetic at 591–597; do not count that row as a valid served acceptance case. The separate composition fixture uses valid B4 zero/branch/spine/prefix metadata.
- **Pending-token window consumption at helper level:** convolution composition now invokes `fused_tree_conv_layer`, which calls the shared prepared-row/window/state helpers, plus the reference driver with independent column-0 selection (conv test 106–138). An in-memory wrapper that invokes this consumer twice is rejected by the expected next-root-window assertion. This closes the old test-helper-only window-construction gap at the arithmetic/composition level.
- **B4 packed replay:** the revised test adds `(48,4,1616)` (54), disjoint permuted per-request row sets (100–110), distinct acceptance buffers (120–128), and retains integer-view equality of every layer's entire bank (186–190). This is a meaningful four-request packing comparison. Its retained first path `[2,6,7,9]` is an arbitrary-index stress input, not a valid cat10 parent chain; valid zero/spine/branch rows are also present. This numerical transport test does not claim served request scheduling/identity qualification.

**Naming limit:** `fused_tree_conv_layer` and `legacy_tree_conv_layer_reference` are test pipeline drivers over production/shared primitives (the fused module labels this region “Full per-layer drivers” and its reference section “TEST-ONLY”). They are not the loaded emitted forward entry point. Revise the fixture's “PRODUCTION consumer/served path” wording accordingly. Selection of the actual correction/bonus token, emitted call-site ordering and loaded next-forward state identity still belong to the already planned served capture gate. No additional boot is requested here.

### Remaining 1: KV reorder fixture still does not match the source-selected contract

The revised test now calls the correct sync-free helper, but its permutation and accepted-node handling are wrong for the selected cat10 source:

- Fixture line 30: `CAT10_PI=[0,1,3,5,7,9,2,4,6,8]`.
- Runner construction at patcher 38805–38820, applied to the selected sorted cat10 choices, produces **`[0,1,2,4,6,8,3,5,7,9]`**.
- Fixture 109–116 transforms the accepted paths with `CAT10_PI[node]`. The runner 41840–41848 supplies the fresh published accepted-node IDs unchanged. Source auto-threading means accepted node `node` resides at **`sm_perm[qsl+node]`**, not `sm_flat[qsl+node]`. The test currently changes both its inputs and oracle to a different identity contract.
- Its “restore” test (120–131) uses reorder OFF and flips the map arbitrarily after preparation. It usefully checks slot-tensor snapshot independence, but does not execute the selected reorder/restore lifecycle.
- KV storage still has one block of 256 slots (39); the planned multi-block address check is not present.

**Reproduction:** keep the fixture's inputs/permutation but remove its `ap_perm=pi[ap]` rewrite and call the same sync-free function with unchanged `ap`. Its own consumer oracle fails: `req 0 depth 2: flat slot 139 does not hold node 2's K/V`. This is a fixture contract failure, not a production kernel failure. Conversely, using the source-derived permutation, unchanged accepted IDs, expected source `before[...,sm_perm[qsl+node],...]`, and restoration `sm_perm[pi]` passes committed-source and flat-map-restoration checks in memory.

**Small repair:** derive/validate `pi` from the same selected tree construction, retain published node IDs, establish markers at verify-time source slots, and assert next-reader committed values through the restored flat mapping. Reuse the same B1/B4 cases and give the existing 256 slots multiple blocks. Keep a wrong-source/remap-omission negative control. Only this canonical remap1/reorder1/syncfree1 variant is required; overlap/off variants do not substitute for it.

### Remaining 2: shared-page sentinels do not detect the original whole-page copy failure

The convolution fixture now uses real shared storage, which fixes the earlier unrelated-allocation issue. However every non-conv remainder in every page contains the **same positive infinity**, and `_sentinel_ok` only tests `isinf` (conv test 60–62; fused test 446–462). Copying a source page's co-resident bytes to a destination page is invisible under this construction.

I replaced only `_load_conv_committer` in memory with a faulty committer that gathers **entire physical pages**, then copies them to the column-0 destination pages. It overwrites the simulated SSM/padding along with conv. **`test_commit_publishes_committed_window_into_column0_on_shared_page` still passes.** This is the same page-stride copy class the original finding required the fixture to detect.

The bounded faulty operation was:

```python
conv = layers[0]._fr13_replay_conv_state
ssi = layers[0]._fr13_replay_spec_idx
page = conv.stride(0)
pages = torch.as_strided(conv, (conv.shape[0], page), (page, 1))
leaf = paths.gather(1, (lens.long()-1).clamp(min=0).view(-1,1)).long()
leaf = torch.where(lens.view(-1,1)>0, leaf, torch.zeros_like(leaf))
src = ssi.long().gather(1, leaf).reshape(-1)
dst = ssi[:,0].long()
pages.index_copy_(0, dst, pages.index_select(0, src))
```

**Small repair:** initialize the non-conv regions with distinct per-page raw markers, snapshot them, and assert their exact bits remain unchanged after commit and next consumption. The deliberate whole-page-copy control must then fail. No model execution or larger allocation is necessary.

### Saved pinned-image evidence and its limits

The saved logs exist and their recorded source-hash prefixes match the full settled hashes above:

- `fused_conv_active_row_ref_081049.txt` identifies GPU container `e2-fh6-081049`, test `987fd8b20ca2298a`, kernel `98f7c8511c1fa991`, fused module `564c7dc7ba61ccd9`; reports **288 passed, 1 skipped**, followed by **15 CPU checks passed**. SHA-256: `e0f45cc858a75973ab35defba778cee031c92a5dcab19ec8a63f8e75f125ec80`.
- `e2_closure_fixtures_v3_080954.txt` identifies CPU container `e2-fx3-080954`, conv `9f3101d827450b6b`, KV `6b169a286c3dac8c`, kernel/fused module; reports **13 passed**. SHA-256: `d27f530545b50253484ab1087df8e5531afbe55153afd3903f2d172e7de21934`.
- `e2_closure_b4_fixtures_080006.txt` records eager test `6ee62d581f18c8a7`; combined run has **8 passed, 11 failed** and the failure list consists exclusively of fused-conv cases. The B4 eager-pack case is therefore part of the passing remainder, consistent with the worker status. Its existing SHA-256 remains `2238ec079218d82b3cd61cec94edd96c464c3a44bf387f023b0c36de5d944fef`.

These are saved test outputs labeled pinned-image by the worker; this CPU review did not start/inspect those containers or independently reconstruct their image digest. Most importantly, their pass counts cannot close the two demonstrated fixture-contract blind spots above. Once those two small controls are repaired and pass, the bounded CPU fixture findings can close; actual loaded B1/B4 publication/next-consumer gates remain separately pending.


## Backend-contract ruling: TREE_ATTN requires the flat slot policy

Read-only source/CPU follow-up on completed policy-A B1 boot `experiments/out-20260922T081202Z-e2-j1prime-p072-kvpolicyA-b1/e7b_001_p072_arm1_none-B_fs_ieee-all/`. No new inference, GPU use, or source mutation.

**Verdict:** the runner's spine-first slot reorder is incompatible with this boot's **unified-attention TREE_ATTN path**. Freeze **`FR13_ATTN_KV_REMAP=1, FR13_SLOT_REORDER=0, FR13_KV_REMAP_SYNCFREE=1`** for this route. This conclusion follows from address/mask contracts, independently of observed logits. The earlier canonical-registry recommendation in this report is superseded for this specifically identified backend path; registry flags cannot substitute for the consumer's contract.

### Source and runtime binding

The completed container `e7a-capture-081202` has been removed, so a fresh literal copy of its filesystem is unavailable. I inspected the preserved plain-route emitted sources at:

`experiments/out-20260922T000315Z-e7a-patcher-dryrun/B_fixed_plain/files/`

Their dry-run image identity matches policy A exactly: image ID `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`, digest `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`. The preserved patcher hash is the same `a7f8f0943d1a13ac4742e824ea4e126c6ff341342aba677db5169e712c7e7281` recorded for policy A.

Policy-A `docker_inspect.json` records remap/reorder/syncfree all 1 and a command invoking **only** `fr10_phase4_patch_vllm_tree_gdn.py` before `--attention-backend TREE_ATTN`; it does not invoke `fr13_patch_fa2_tree_bias.py`. `docker_logs.txt:45` confirms TREE_ATTN selection; line 283 confirms slot reorder with `pi=[0,1,2,4,6,8,3,5,7,9]`; line 335 confirms sync-free remap engagement (`foreign_first=-1`). This is a source-lineage reconstruction with runtime corroboration, not a claimed hash of an uncaptured policy-A backend file. Preserve the loaded backend/runner/attention-op files and their hashes in the next already planned capture.

Correction to worker reply 21: `tree_attn.py` **is** patched and preserved, as the dry-run artifact and policy-A patcher output show. Its patches do not implement the slot-reorder consumer changes.

### Exact contract

Line numbers here refer to the preserved emitted files, except where the patcher is explicitly named.

1. **Writes honor the slot map.** `vllm/v1/attention/backends/tree_attn.py:1066–1090` implements `do_kv_cache_update` with `ops.reshape_and_cache_flash(..., slot_mapping, ...)`. Runner reorder changes that map to `sm_perm[j]=sm_flat[pi_inv[j]]` (emitted `gpu_model_runner.py:3516`; patcher 38853–38855).

2. **Reads use flat sequence/block-table positions.** `tree_attn.py:1152–1167` calls `unified_attention` with the cache, `block_table`, and `decode_meta.tree_attn_bias`; it passes no slot map or permutation. In `vllm/v1/attention/ops/triton_unified_attention.py:812–835`, a logical `seq_offset` reads block `block_table[seq_offset // BLOCK_SIZE]` and offset `seq_offset % BLOCK_SIZE`. The segmented path uses the same addressing at 1221–1244. Thus after permuted writes, physical suffix column `c` contains node **`pi[c]`**, not node `c`.

3. **The mask still names original node columns.** `tree_attn.py:856–865,971–1005` builds the root/self/ancestor mask from choices sorted by `(depth, path)`. Unified attention adds the mask at `query_pos, key_rel_pos=seq_offset-context_len` (949–960; segmented 1358–1369). No `bias[:,pi]` adjustment exists in this path. `causal=True` is also unchanged.

4. **Post-accept remap is too late to repair a verify-time mismatch.** It runs after verifier attention and acceptance. Slot-map restoration before the drafter likewise cannot undo the current verify output.

The fork-FA2 patch supplies the missing companion behavior: `scripts/fr13_patch_fa2_tree_bias.py:9389–9465,9637` introduces key-axis bias permutation and a causal flag for the FA2 tree-bias call. Its fallback `unified_attention` still uses the original `decode_meta.tree_attn_bias`. The correct distinction is the actual loaded attention call path, not merely a generic “tree” configuration or even the class name TREE_ATTN.

### CPU address witness

I extracted the two exact mask-building functions from the preserved `tree_attn.py`, used policy A's captured tree choices, and reproduced the runner's permutation. With B1 and B4, disjoint permuted physical block tables, block size4 and a multi-block suffix, each verifier node wrote its own unique marker through the chosen slot map. Reads then followed the actual kernel's block-table/modulo rule and exact extracted node mask:

| Policy | B1 wrong query rows | B4 wrong query rows | First wrong read |
|---|---:|---:|---|
| Reorder OFF | 0 | 0 | None |
| Reorder ON | 7 | 28 | Query row3 expects nodes `[0,1,3]`, reads `[0,1,4]` |

Rows 0–2 are unchanged because those permutation entries are fixed. This matches the reported localization but does not use output differences as the proof. On the same CPU multi-block layout, the exact sync-free remap function with `dst_pi=None`, unchanged node IDs and valid B4 branch/spine/zero/sibling paths put all committed tokens in their expected linear slots.

### Minimum repair and stop rule

- Freeze/forward the flat TREE_ATTN policy above. Make launch/qualification gates backend-specific: require remap and sync-free; require reorder OFF on the unchanged unified-attention path. Fail closed on an incompatible combination. Do not globally turn off the fork-FA2 policy.
- Use one compact CPU address/mask marker control like the witness above: flat policy must preserve verifier node identity; forcing reorder ON without its reader/mask adaptation must fail. Retain the flat-map, multi-block post-accept remap/next-consumer fixture. No extra ablation campaign is required.
- Continue only the already planned corrected B1 and actual-B4 served publication/next-consumer checks on this frozen policy. Policy A remains a preserved failed compatibility diagnostic, not whole-model E2 qualification. Its local layer-62 arithmetic/h0 checks remain local evidence.
- Capture loaded source hashes, effective policy, remap engagement, and the expected absence of slot reorder. Do not patch/rewrite the completed policy-A record or relabel earlier remap-OFF captures as flat-policy qualification.

### Hashes

Preserved emitted sources:

- `tree_attn.py`: `a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97`.
- `triton_unified_attention.py`: `5dd83ac4fbd0a08054a5b7188f4937cdf33fce854b96d0a38e616a9e4c7789c0`.
- `gpu_model_runner.py`: `9487b658ab8eef802737370e6923aaeae865156625d07fdcd49d3fb2772e9cf4`.
- FA2 companion patcher: `8f61b9cd6a079d0605daedb545b8a2e95225714693c775d396059bab2ce626b2`.

Policy-A evidence:

- `docker_inspect.json`: `f29f43eb2389896847d2da61bdf6a77a175dc48d4bbea31154e58feb3e0a8d3e`.
- `docker_logs.txt`: `0f197d65a995d57faef90ec2f8031d234b349d543976960753b00fb42285faf1`.
- `image_identity.txt`: `6bce4d8f72ec7a4ad4a501c6ab7fe821e9b5eae1f02fa96347dc4f37d3a74bde`.

The dry-run `image_identity.txt` SHA-256 is `7155995dd9a703f1328854a50b93cfe6a51f6946edf85a58f000483eaa99ca15`; its formatting differs from policy A but the image ID/digest match exactly.

