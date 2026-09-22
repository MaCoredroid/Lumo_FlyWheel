# E1 campaign launch red-team — bounded draft review

Review date: 2026-09-22, approximately 09:15–09:21 UTC. Reviewer: independent Codex subagent `paper_redteam_round2`.

**Disposition: do not launch the timed campaign from this reviewed draft. The patched-runner native strategy has a defensible guarded stock fallback; the remaining blockers are campaign integration defects.** All findings were sent promptly to the parent. The worker was actively editing these files. Findings below identify exact reviewed versions, not an allegation that the subsequently revised implementation or any executed E1 campaign has these defects. Hold further review until the owner supplies settled hashes and CPU stub results.

Scope: read-only remote source inspection and CPU-only counterexamples; no GPU, inference, running container, tmux, or worker-source modification. The one Docker operation used a uniquely named **stopped** container to extract two pinned stock Python files; that exact created container was removed. This report is the only local output. Remote paths below are under `mark@100.103.10.122:/home/mark/lumo-paper-v2-20260921/`, abbreviated WT; E = `WT/papers/gdn-tree-scan-mlsys/v2/experiments/e1`.

## Identities of reviewed sources

| Source | SHA-256 |
|---|---|
| E/E1_FREEZE.md | 624f61e48e977ff35928cdc1d79fb727853a6dc5aac8b5e86cf41c07dbd6cda9 |
| E/E1_DESIGN.md | 1b2bd5858da377dc333614a6e0aa169fd4c0e5b9276c0da1f3782a7c6a68282d |
| E/e1_native_launch.v1.sh (superseded by v2 below) | 5f10b6fd3e488dba06379fa0a70ba5dd4d2cb4e5adcd85a9f55951323be7710c |
| E/e1_native_launch.v2.sh | 5d86340e683a92c64caabee51bbdd2514cae8e9a1f96c6299dfd14500dba2658 |
| E/e1_cell_driver.v1.sh — principal reviewed revision | ba8a71fe41f80cdd8761ed08d32db0ddf59442aa787cf45ee3e61eaaf3841c9e |
| E/e1_cell_summary.py — principal reviewed revision | 10d6d2c0588c15358a9012d69af5ca744820d27b170088b3738ccf781de329b9 |
| E/e1_cell_verify.py — principal reviewed revision | 91b85c7c89ad5638fce982ff69652fc6466d509d583f11af8a08e13a90c04e71 |
| E/e1_workload.py | 54d026b5724fe4829d28fa47da41195eb88a5ef35f3d25ed145cce327227e041 |
| E/e1_native_preflight.py | 3b35e13e6d28b742b22ee6186e4a121768457794dcc49ad7a7cd70573f0be7e2 |
| E/e1_run_cells.v1.sh | aac1ca32fba82921c5c36a6690f961255fff7a37f050d8af520833964a94f270 |
| E/e1_join.py | cbee2ae70947f1adb4073bd151a7148fcb05cbc02d1c5dff37b381ffadf1e766 |
| E/e1_recorder.py | 1cf3f53552e008c44f7b7e5e6187d8052bcfd4c75bd6f81e2e07d1f70fbf65a0 |
| E/e1_event_recorder_shim.py | 06344718f035949ea7f1cea49e4677e9acc2a71a460bbbc7b94e01a1bbbbc1df |
| E/e1_api_tokens_from_capture.v2.py | 6d36ee15d8b5b6289c24ebc437f621dfd32adc010aae9ff1495fb34af0121f9f |
| E/test_e1_cell_driver_stub.sh | f8170b52c3fb4d2799a3fa83891fee4042417a4c00ef83ac716544a41b76a242 |
| WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py | a7f8f0943d1a13ac4742e824ea4e126c6ff341342aba677db5169e712c7e7281 |
| WT/src/lumo_flywheel_serving/fr10_decode_modes.py | 9111854366d4856559d9b42c23d5ca6c84800638c08f98d9f15bbeb09d93499b |

A final identity-only read found newer driver `e2fe232dcddaf0e689baa4a61f2ff8c829f9693a32bbea0d38b0250bfccd119b`, summary `1ab035d134631e0e35614ebb8393a9a108610cac82865d6c70b803350fc7cc0b`, verify `8401d423db628c2fcf19440e7147fedd619c2534c076940a71c70affcc65090a`, generator `819a910daa39ae16736a65ed8d3872a52103ec379a3c676da48db6a1424d0b6b`, and cells `691e59ed3c982ef9aa432adf00b97922ff231d025ebae762ebdee8b701f239ee`. **These were not re-reviewed and are not declared defective or closed by this report.** Runner and native preflight hashes remained as above.

## Guarded native baseline: source result and boundary

The statement “every FR13 feature OFF” is inaccurate: the patcher bakes some replay/conv paths on, and globally forcing retired RUNROW_INIT=0 is not a viable bypass. The valid argument is the branch guard, with the baseline labeled **patched-runner native**.

The native launch v2 corrects the two already reported v1 blockers: lines 110–112 explicitly disable prefix caching by default; lines 238–239 forward `FR10_DECODE_MODE_DEFAULT=naive_mtp` and `FR10_ENABLE_TREE_GDN=0` into the container. Lines 102–104 create native MTP-5 or MTP-11 without a tree descriptor. Lines 247–259 apply the common patcher/shim and select eager/synchronous operation when the driver passes the frozen settings. The source guard uses **naive_mtp**, not “native_mtp.”

The loaded B1 source used for independent comparison is:
`WT/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T082815Z-e2-b1-policyB-sync-p072/e7b_001_p072_arm1_none-B_fs_ieee-all/loaded_backend/`.

| Loaded source | SHA-256 |
|---|---|
| gpu_model_runner.py | b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40 |
| rejection_sampler.py | 2680d038b9e626f659b17e6910dd89fde3ea9581eb21ea468ccd80f572ed1a56 |
| gdn_linear_attn.py | 63bc4503d560f54cf9923fc03e69267368f3951a92b1b5734a8ef871c40a0b99 |

Concrete guards:

- Decode modes source 20–24 permits naive_mtp/non_mtp/tree_mtp; patcher 18594/18618 otherwise defaults requests to tree_mtp. Runner 2889 propagates the homogeneous mode; runner 3971 needs tree mode plus the descriptor before attaching tree metadata.
- GDN 10907–10911 (tree conv) and 13525–13529 (tree recurrence) require tree GDN enabled, tree mode, and parent metadata. Native conv fallback 13351–13364 retains native indices and accepted-token count. Native recurrent update fallback 15355–15373 does likewise.
- Sampler 172 enters the tree path only with parent metadata; 463–475 makes a missing tree-parent condition fail in tree mode. Native greedy fallback is at 3872–3889.
- Patcher 26036–26044 has the Mamba copy tree branch gated by tree default mode and tree-GDN enablement; explicitly setting both native controls therefore matters. Native drafter optimizations around 27004 depend on tree-choice shape metadata.

Pinned image: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`. Extracted stock rejection sampler SHA `d54ae03b5a5335ea133a5644bc9ea6331db68544b77bfd4f9970bbaaf7309313`; stock GDN source SHA `c6c1a8ae5c9039f7a17c004761486013de58756a2988e25af78e4b6aa5316019`.

CPU AST results:

```text
NATIVE_REJECTION_BODY_AST_EQUALS_STOCK True stock_n 21 native_n 21
NATIVE_CALL 13351 EXACT_STOCK_AST_MATCH True
NATIVE_CALL 15355 EXACT_STOCK_AST_MATCH True
REMOVED_ONLY e1-native-source-review-d2bb1ea800f9
```

The sampler comparison removes only the two top-level tree-parent conditional branches from the patched rejection_sample body and compares the remainder to stock. The operator-call comparisons preserve every argument. Thus no unavoidable numerical change was found on these critical native branches. This is conditional static evidence, not a claim that a native server has already executed this route. Actual loaded native sources/configuration, absence of tree engagement, recorder output binding, and the pre-timing native qualification gate remain necessary. The stale launcher comments “all features OFF” / “only loader shim” should be corrected to the guarded-route description.

## Concrete blockers in the reviewed draft

### C1 — cell and campaign dispatch Python does not compile

Driver lines 95 and 107, runner line 24 interpolate escaped double quotes inside f-string expressions, e.g. `c[\\"index\\"]`. Extracting the exact shell-quoted Python bodies with shlex and calling Python compile() gives:

```text
driver:95 SyntaxError: unexpected character after line continuation character
driver:107 SyntaxError: unexpected character after line continuation character
```

Runner line 24 contains the same expression. This fails before a boot. Minimal correction: valid Python formatting, plus exercise the real snapshot/re-exec CPU stub. A shell-only bash -n check does not compile embedded Python.

### C2 — tree timing still enables heavy, asymmetric capture

Driver lines 120 and 123 set `FR10_METRICS=1`, `LUMO_MTP_DRAFT_TRACE_FILE=/logs/fr10_mtp_draft_trace.jsonl`, and path-LCP logging. Tree launcher v7 forwards these variables at 322/420/422. This contradicts E1_FREEZE line 18.

This is not merely a noisy log: loaded rejection_sampler 150–159 executes `target_logits.detach().to(torch.float32).clone()` with FR10_METRICS, and patcher 32543–32570 triggers the draft trace on either FR10_METRICS or the explicit trace path and executes `out.detach().cpu().tolist()`. Native v2 leaves this machinery disabled. These additional tree-only GPU allocations and CPU transfers belong outside the timing campaign.

Minimal correction: timing uses FR10_METRICS=0 and empty draft/LCP/debug/capture sinks; retain the common E1 recorder and SFWD timer. Verify the **actual container env** and the CPU stub against these values. Current verifier line 22 and stub line 20/42 omit the metrics/draft trace variables, so their “no capture” checks do not cover this defect.

### C3 — summary supplies API IDs to an engine-ID phase manifest

Workload lines 55–56 put response IDs in `man.phases`; API mapper produces engine-request keys. Summary line 23 forwards `man.phases` without translation, while its API→engine map is only constructed later at 31–37. Joiner 87–93 correctly demands phases cover actual recorded engine IDs.

Independent CPU minimal reproducer used a sealed, valid two-forward single-request fixture and direct token IDs [10,11]:

```text
current_api_id_manifest rc 2
invalid manifest: recorded requests in no phase: ['cmpl-abc-0-engine']
mapped_engine_id_manifest rc 0
n_usable 1 rate 10.0
```

Minimal correction: construct a unique, complete API→engine mapping first, translate **all** warmup/preflight/timed phase memberships with it, then invoke the joiner. Reject ambiguous or missing mapping. Do not weaken the joiner's complete-output requirement.

The earlier summary version's `retained`/request_ids schema mismatch was already fixed in the reviewed 10d6d2c revision and is not an open finding here.

### C4 — “untimed preflight” actually examines and conditionally retains timed data after shutdown

Runner lines 39–49 execute the complete driver, verify it, then run native_preflight and retain that same boot's timing window on PASS. Driver line 141 invokes the workload's warmup→timed sequence unconditionally. Native preflight lines 38–56 explicitly examine all eight **timed** requests; line 60 requires final recorder seal. This does not implement the accepted before-timing segment.

Minimal correction within the same first native boot: execute a distinct untimed preflight phase with fixed pass/fail criteria, resolve PASS before submitting timed requests, perform the frozen shape-matched warmup/reset boundary, then submit timed requests. Preserve and reconcile every output from every phase, and require final seal after the complete run. A final post-run audit may verify the earlier gate but cannot substitute for it or reuse its measurements favorably. No extra boots or axes are requested.

### C5 — campaign snapshots record drift but do not prevent post-start changes or replacement

Runner 20–21 copies/hashes files but 24/26 continue reading the caller's live CELLS/Q paths; 39 invokes the live E driver each iteration, whose line 97 snapshots then-current helpers. Driver line 96 accepts an existing cell directory with mkdir -p and subsequent writes overwrite existing files. FROM/TO lets the same index be run again without a refusal. This is weaker than the frozen policy and “immutable” header.

Minimal correction: execute one campaign snapshot, freeze/hash all runtime sources/configuration before first retained data (including patcher and route dependencies mounted through /workspace), validate identities at each launch, and refuse any existing cell payload or changed campaign identity. Keep failed/ineligible cells listed and stop or resume only unattempted cells under the same frozen identity. Do not silently overwrite or relaunch a measured cell. Pinned default env should be verified against the frozen cells, including exact cat10 SPEC_CONFIG; an externally inherited TREE/SPEC_CONFIG must not bypass the freeze.

### C6 — legitimate zero retained support is classified as instrumentation failure

Joiner line 142 returns rc3 when a fully reconciled, valid run has no usable intervals. Summary line 26 treats every nonzero return as INVALID before reading the report. Runner then stops at line 52. Frozen EOS/budget outcomes that fail structural floors must be INSUFFICIENT_SUPPORT, not INVALID; complete valid short runs can produce zero intervals without an instrumentation defect.

Minimal correction: inspect the complete join result. Zero support with complete API reconciliation and no invalid/refused state maps to INSUFFICIENT_SUPPORT; instrumentation loss, refused token evidence, malformed phase mapping, and missing output remain INVALID. This preserves the planned table without replacing/lengthening requests.

## Supported pieces and explicitly unfinished work

The order generator uses the prescribed three arm permutations and alternating B1/B4 order; it gives 18 cells with each arm×batch represented once per block. Native5 and cat10 are the depth-matched comparison; native11 is explicitly the longer-chain control. The new request client directly requests token IDs, fixes temperature zero/seed, validates frozen prefix hashes, and implements the specified B1/B4 warmup/request counts. Full API reconciliation is enforced by the unchanged joiner once the phase map is correct. Actual occupancy and complete request sets—not MAX_NUM_SEQS alone—drive retained intervals. The stable joiner retains all valid slow same-cohort intervals in the primary rate; 1.5 s is diagnostic only.

The summary's current floor form checks all eight B1 prompts or both exact B4 cohort sets, not only aggregate N. At final freeze, bind those memberships to the eight frozen prompt IDs and maintain per-prompt/cohort accounting and exclusions. The reviewed order metadata still named old native/tree launcher versions while the driver chose v2/v7; synchronize the frozen metadata with the executed identities before first timing.

A final campaign bootstrap/aggregate implementation was not present in the reviewed files. Treat that as unfinished, not a defect in an executed analysis. It must use the frozen paired whole-boot block unit, three blocks separately by B, the native5 mean-rate denominator for half-width, and no selective unpaired replacement. Three-block uncertainty remains coarse as already disclosed. The report does not request a larger campaign.

## Reproduction commands

All remote Python below ran via `ssh mark@100.103.10.122 'python3 - <<...'`. No launcher was executed. Exact static stock comparison program:

```python
import subprocess,tarfile,io,hashlib,uuid,ast
from pathlib import Path
name="e1-native-source-review-"+uuid.uuid4().hex[:12]
cid=subprocess.check_output(["docker","create","--name",name,"--network","none","--entrypoint","/bin/true","vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"],text=True).strip()
loaded=Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T082815Z-e2-b1-policyB-sync-p072/e7b_001_p072_arm1_none-B_fs_ieee-all/loaded_backend")
try:
 for rel in ["vllm/v1/sample/rejection_sampler.py","vllm/model_executor/layers/mamba/gdn_linear_attn.py"]:
  out=subprocess.check_output(["docker","cp",cid+":/usr/local/lib/python3.12/dist-packages/"+rel,"-"])
  with tarfile.open(fileobj=io.BytesIO(out)) as t:bs=next(t.extractfile(m).read() for m in t if m.isfile())
  current=(loaded/Path(rel).name).read_bytes()
  print("SOURCE",rel,"STOCK",hashlib.sha256(bs).hexdigest(),"PATCHED",hashlib.sha256(current).hexdigest())
  ta,tb=ast.parse(bs.decode()),ast.parse(current.decode())
  if "rejection" in rel:
   a=next(n for n in ta.body if isinstance(n,ast.FunctionDef) and n.name=="rejection_sample")
   b=next(n for n in tb.body if isinstance(n,ast.FunctionDef) and n.name=="rejection_sample")
   bodyb=[n for n in b.body if not(isinstance(n,ast.If) and "tree_parent_indices" in ast.unparse(n.test))]
   print("NATIVE_REJECTION_BODY_AST_EQUALS_STOCK",ast.dump(ast.Module(body=a.body,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=bodyb,type_ignores=[]),include_attributes=False),"stock_n",len(a.body),"native_n",len(bodyb))
  else:
   for fn in ["causal_conv1d_update","fused_sigmoid_gating_delta_rule_update"]:
    aa=[n for n in ast.walk(ta) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==fn]
    bb=[n for n in ast.walk(tb) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==fn]
    print("CALL",fn,"STOCK_CALLS",len(aa),"PATCHED_CALLS",len(bb))
    for n in bb:
     if 13340<=n.lineno<=13364 or 15354<=n.lineno<=15373:
      print("NATIVE_CALL",n.lineno,"EXACT_STOCK_AST_MATCH",any(ast.dump(n,include_attributes=False)==ast.dump(m,include_attributes=False) for m in aa))
finally:subprocess.run(["docker","rm",cid],check=True,capture_output=True);print("REMOVED_ONLY",name)

```

Exact phase-ID counterexample (temporary files automatically removed):

```python
import ast,json,subprocess,tempfile,sys
from pathlib import Path
E=Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/e1")
ta=ast.parse((E/"test_e1_recorder_join_cpu.py").read_text()); ns={"json":json,"PROBE":{"event":"recorder_probe","t":0.0}}
for n in ta.body:
 if isinstance(n,ast.FunctionDef) and n.name in ["W","FE","PS","CR","row"]:
  exec(compile(ast.Module(body=[n],type_ignores=[]),"<testhelpers>","exec"),ns)
with tempfile.TemporaryDirectory(prefix="e1-campaign-redteam-") as d:
 p=Path(d); rid="cmpl-abc-0-engine"; seq=[]
 for k in range(2): seq += [ns["FE"](k+1),ns["PS"](k+1,k+1,0.1*k,[rid]),ns["CR"](k+1,k+1,[ns["row"](0,rid,k,[10+k])])]
 ns["W"](str(p/"events.jsonl"),seq); (p/"api.json").write_text(json.dumps({rid:[10,11]}))
 for label,id_ in [("current_api_id_manifest","cmpl-abc"),("mapped_engine_id_manifest",rid)]:
  (p/"man.json").write_text(json.dumps({"phases":{"warmup":[],"timed":[id_]}}))
  r=subprocess.run([sys.executable,str(E/"e1_join.py"),str(p/"events.jsonl"),"--api-tokens",str(p/"api.json"),"--manifest",str(p/"man.json"),"--expect-reqs","1","--json",str(p/"out.json")],text=True,capture_output=True)
  j=json.loads((p/"out.json").read_text())
  print(label,"rc",r.returncode,"invalid",j.get("invalid"),"n_usable",j.get("n_usable"),"rate",j.get("tokens_per_wall_second"))

```

