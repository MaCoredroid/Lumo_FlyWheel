# E2 fixture final recheck — 2026-09-22 08:30 UTC

Bounded read-only/CPU review of the two original fixture gaps. Remote checkout: `mark@100.103.10.122:/home/mark/lumo-paper-v2-20260921`. No GPU, inference, tmux, manuscript, or canonical source changes. Temporary pytest storage was outside the checkout; bytecode and pytest cache writes were disabled.

**Decision: both original fixture gaps are closed at helper/composition scope.** No new material implementation failure found. This is not qualification of the loaded serving entry, its call ordering, or the backend contract; that remains the independent served-source/boot gate.

## Exact source identities

| File, relative to checkout | SHA-256 |
|---|---|
| `tests/test_fr13_conv_col0_publish_read_composition.py` | `0b0593d0b01448954c9ef34a60326155da14f3c267493059a388159b4009262f` |
| `tests/test_fr13_attn_kv_remap_pytest.py` — actual bytes exercised | `3887f66404be0e20c43adf6a40e605af89a8d3aaf28797e37558488681a9a638` |
| `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` | `98f7c8511c1fa991fd36526e69a8b753a5205716bf5b0a53c058177c965cdb20` |
| `tests/test_fr13_conv_committed_path.py` | `bf7ce331752a499810ee8f7f9299de685c1fb1c3a7c0ee3a5580435d145fb514` |
| `tests/test_fr13_tree_conv_fused_byte_ab.py` | `987fd8b20ca2298a2bba0073e82d87efbb0fc653183679206e35f8da7caabcdf` |
| `src/lumo_flywheel_serving/fr13_tree_conv_fused.py` | `564c7dc7ba61ccd9579f2fb7a4af9a2cdaaa1753dfb98434f2011457d90d9a6e` |
| `scripts/fr10_phase4_patch_vllm_tree_gdn.py` | `a7f8f0943d1a13ac4742e824ea4e126c6ff341342aba677db5169e712c7e7281` |

The assigned KV SHA `7b538afa55183ce43096af34cc5b96d9bcfc9b43ac39f39be567a6aee3c414b1` was inspected initially; its policy-B test name/docstring changed during the review. The table identifies the actual executed bytes. Kernel/conv/KV hashes were printed before and after the CPU run and matched.

## Original gap closure and independent negatives

1. **pi and accepted-node identity:** KV lines 31–56 derive `pi=[0,1,2,4,6,8,3,5,7,9]` from the parent/choice vectors; lines 92–100 construct `sm_perm` using the inverse. The policy-A fixture at lines 138–175 passes accepted IDs unchanged and independently checks the source at `sm_perm[qsl+node]` against the flat destination `sm_flat[qsl+m+1]`. This agrees with the inspected production helper at kernel lines 9843–9852 and 10253–10256. Replacing pi with the old incorrect value is rejected. Independently wrapping the exact production sync-free helper to perform the old `pi[accepted_paths]` rewrite is rejected at request 0, depth 3: destination slot 1 gets marker 66 instead of source marker 108. **Closed.**

2. **Whole-page overwrite visibility:** conv lines 52 and 64–82 stamp and compare the raw bits of every page's remainder separately. The first marker `r+1` is distinct and exactly representable for all 64 pages, even though larger marker values may round in bf16. Conv line 105 checks all remainder bytes after publication. The new built-in negative at lines 161–171 detects a full-page copy. Independently replacing the emitted committer with a whole-page-copy implementation, which still publishes every correct conv window, causes the main composition fixture to fail specifically with “the conv commit wrote outside the conv slice of a shared page (SSM stand-in bytes changed).” **Closed.**

The two fixture modules passed **17/17 cases in 0.22 seconds**, using the exact six production Torch helper definitions extracted by AST into an in-memory import module. Only timer observers were stubbed; `_FR13_FIXED32_MODE=None` selects this task's non-fixed32 route. This was CPU execution; no Triton/CUDA inference was performed.

## Multi-block address scope

The committed KV fixture still uses shape `[2,1,256,2,3]` (lines 67–75) and its consumer checker indexes block zero (lines 103–119). Thus that fixture alone cannot catch block-index mistakes. A bounded independent check reused the same three B4 cohorts and 256 physical slots, reshaped to two multi-block cache layouts, then compared **the entire cache** against an independently constructed accepted-source-to-linear-destination oracle. Both direct sync-free and prepare/apply paths passed:

| Seed | Cache shape | Foreign copies | Cross-block copies | Full-cache oracle |
|---|---|---:|---:|---|
| 7 | `[2,16,16,2,3]` | 5 | 5 | pass, direct and split |
| 7 | `[2,8,32,2,3]` | 5 | 4 | pass, direct and split |
| 11 | `[2,16,16,2,3]` | 5 | 5 | pass, direct and split |
| 11 | `[2,8,32,2,3]` | 5 | 5 | pass, direct and split |
| 23 | `[2,16,16,2,3]` | 5 | 4 | pass, direct and split |
| 23 | `[2,8,32,2,3]` | 5 | 4 | pass, direct and split |

This checks kernel lines 10338–10343 (`slot // block_size`, `slot % block_size`) across actual nonzero block indices. **No new experiment axis or GPU run is needed for this arithmetic.** Minimal durable regression improvement: parameterize the existing flat-policy fixture with one `block_size=16` case and make its oracle use block/offset indexing. This review's passing check closes the immediate address-arithmetic question; the persisted single-block test limitation should not be described as a found implementation defect or a served-backend proof.

## Selected policy-B binding

Current `experiments/e1/E1_FREEZE.md` SHA `0378b22dcdd7699c088740c56bd363d07280693eb465bdc4a737add8bdb7b70b`, line 18 selects **remap=1 / reorder=0 / syncfree=1**, conditional on the independent TREE_ATTN source check. The updated flat-policy fixture, lines 123–138, exercises the sync-free helper with `dst_pi=None` and the flat source mapping, consistent with that policy.

`experiments/e7a/serve_drivers.v11.sh` SHA `f23d2adc8995bacfb6472ef18a4d93528adebd16cdac61a5066506b59b847e7c`, line 218 requires the three explicit `E7B_KV_*` values and forwards them. `experiments/e7a/e7a_capture_launch.v6.sh` SHA `e2135df5bd15765449feb0bef47a4cf8a1444fba43f49dfced9af4607f18cd3d`, lines 351–353 requires/forwards the corresponding container envs without defaults. This is source forwarding evidence, not a claim that an already executed boot used these settings.

The repository-wide required/canonical files still select reorder=1 (required file line 27; canonical file line 25). The task's explicit override is therefore necessary and is present in the reviewed design/forwarding chain. Actual boot env, baked flags, engagement, and backend behavior remain the separate serving gate. Wording such as “qualified route” in the test docstring should be read as the selected policy at helper scope until that gate passes.

## Reproduction command

Run from the remote checkout as:

```sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -B - <<'PY'
import ast, hashlib, json, os, sys, tempfile, types
from pathlib import Path
import torch, pytest
torch.set_num_threads(1)
kpath=Path("src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py")
convpath=Path("tests/test_fr13_conv_col0_publish_read_composition.py")
kvpath=Path("tests/test_fr13_attn_kv_remap_pytest.py")
for p in [kpath,convpath,kvpath]:
 print("SOURCE",p,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
names={"launch_attn_kv_linear_remap","_launch_attn_kv_linear_remap_impl","launch_attn_kv_linear_remap_syncfree","_launch_attn_kv_linear_remap_syncfree_impl","prepare_kv_remap_slots","apply_kv_remap_slots"}
tree=ast.parse(kpath.read_text())
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(selected)==6
name="lumo_flywheel_serving.fr10_gdn_tree_kernel"
shim=types.ModuleType(name)
shim.__dict__.update(torch=torch,_FR13_FIXED32_MODE=None,_fr13_span_begin=lambda *a:None,_fr13_span_end=lambda *a:None)
exec(compile(ast.Module(body=selected,type_ignores=[]),str(kpath),"exec"),shim.__dict__)
sys.modules[name]=shim
with tempfile.TemporaryDirectory(prefix="e2-fixture-final-review-") as td:
 rc=pytest.main(["-q","-p","no:cacheprovider","--basetemp",td+"/pytest",str(convpath),str(kvpath)])
assert rc==0,rc
kv=sys.modules["test_fr13_attn_kv_remap_pytest"]
conv=sys.modules["test_fr13_conv_col0_publish_read_composition"]
old_pi=kv.CAT10_PI
kv.CAT10_PI=[0,1,3,5,7,9,2,4,6,8]
try:
 try:kv.test_runner_pi_matches_the_deployed_caterpillar()
 except AssertionError:print("NEGATIVE wrong original pi REJECTED")
 else:raise RuntimeError("wrong original pi accepted")
finally:kv.CAT10_PI=old_pi
original=kv.launch_attn_kv_linear_remap_syncfree
def wrong_rewrite(**kw):
 kw["accepted_paths"]=torch.tensor(kv.CAT10_PI,dtype=torch.long)[kw["accepted_paths"].long()].to(torch.int32)
 return original(**kw)
kv.launch_attn_kv_linear_remap_syncfree=wrong_rewrite
try:
 try:kv.test_policy_a_reorder_on_syncfree_with_runner_pi_unchanged_node_ids_and_restore(7)
 except AssertionError as e:print("NEGATIVE old accepted-ID rewrite REJECTED",str(e))
 else:raise RuntimeError("wrong accepted-ID rewrite accepted")
finally:kv.launch_attn_kv_linear_remap_syncfree=original
def fullpage_commit(layers, paths, lens, B, enabled):
 for layer in layers:
  cv=layer._fr13_replay_conv_state; ssi=layer._fr13_replay_spec_idx
  size=conv.DIM*conv.STATE_LEN+conv.PAD
  pages=torch.empty(0,dtype=cv.dtype).set_(cv.untyped_storage(),0,(64,size),(size,1))
  for b in range(B):
   leaf=int(paths[b,int(lens[b])-1]) if int(lens[b]) else 0
   src=int(ssi[b,leaf]); dst=int(ssi[b,0])
   pages[dst]=pages[src].clone()
with pytest.MonkeyPatch.context() as mp:
 mp.setattr(conv,"_load_conv_committer",lambda:fullpage_commit)
 try:conv.test_commit_publishes_committed_window_into_column0_on_shared_page(mp)
 except AssertionError as e:print("NEGATIVE old whole-page-copy publisher REJECTED",str(e))
 else:raise RuntimeError("whole-page-copy publisher accepted")
for seed in [7,11,23]:
 sm,qsl,ap,acc=kv._cohort(kv.CASES,seed)
 for bs in [16,32]:
  original_caches=kv._marker_kv()
  caches=[x.reshape(2,256//bs,bs,2,3).clone() for x in original_caches]
  before=[x.clone() for x in caches]
  kwargs=dict(slot_mapping=sm,query_start_loc=qsl,accepted_paths=ap,num_accepted_tokens=acc,num_spec_decodes=len(kv.CASES),dst_pi=None)
  original(kv_caches=caches,**kwargs)
  expected=[x.clone() for x in before]
  cross=0;foreign=0
  for b,path in enumerate(kv.CASES):
   base=int(qsl[b])
   for m,node in enumerate(path):
    src=int(sm[base+node]);dst=int(sm[base+m+1])
    cross += int(src//bs != dst//bs);foreign += int(src!=dst)
    for out,old in zip(expected,before):out[:,dst//bs,dst%bs]=old[:,src//bs,src%bs]
  assert all(torch.equal(a,b) for a,b in zip(caches,expected))
  prep=shim.prepare_kv_remap_slots(**kwargs)
  split=[x.clone() for x in before]
  shim.apply_kv_remap_slots(kv_caches=split,dst_slot=prep[0],src_slot=prep[1],active_flat=prep[2])
  assert all(torch.equal(a,b) for a,b in zip(split,expected))
  print("MULTIBLOCK",json.dumps(dict(seed=seed,shape=list(caches[0].shape),foreign=foreign,cross_block=cross,direct_and_split="PASS",complete_cache_oracle="PASS")))
print("FINAL_SOURCE_IDENTITIES")
for p in [kpath,convpath,kvpath]:print(p,hashlib.sha256(p.read_bytes()).hexdigest())
PY
```

Observed summary: `17 passed`; all three deliberately corrupted variants rejected; all six multi-block direct/split full-cache comparisons passed. No remaining original fixture gap.

