# E1 campaign final red-team — reply-24 settled repair

2026-09-22. Read-only review of the reply-24 settled sources, with independent CPU reruns and two additional counterexamples. No GPU, inference, launcher/container start, tmux, or canonical source edits. Test scratch directories created by this review were removed.

**Verdict: remaining material launch blockers F1 and F2 below.** C1/C2/C3/C4/C6's direct repairs work on this version; C5 is substantially repaired but still allows configuration/reference drift. A failed final native gate can leave a measured VALID cell eligible for aggregation. Fixes are bounded orchestration/configuration changes, not additional experiments.

## Reviewed identities

Remote E = `/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/e1` on `mark@100.103.10.122`. Full hashes independently read from current bytes and matched reply-24's prefixes:

| E file | SHA-256 |
|---|---|
| e1_run_cells.v2.sh | c79655d2ba57563bb4671c2ae80e5f3b9f4abf478b8f3a19250470b65d2fc63b |
| e1_cell_driver.v1.sh | 8d45c4439e78f2741a6b2ca2aee2c8a64028c51b6357d39d751797e5c09e3b9a |
| e1_workload.py | bcc5af170fd16599223ad7286f0367c82f978faf640ac4acebe0e9c36d001594 |
| e1_cell_verify.py | dec15e5bf3111fe476f3be0064ce74002199bd0ea90ffe8eb342590eace64c1d |
| e1_cell_summary.py | ae3016f05246f4719c9ccf78222a92b1959446a5f4dcfcce79b43b9fe7923dc6 |
| e1_native_preflight.py | 6c671e0ef7f3834c98acd462c5d8c74ac3be096f3385db89fa5071230a8c5463 |
| e1_native_launch.v2.sh | 5d86340e683a92c64caabee51bbdd2514cae8e9a1f96c6299dfd14500dba2658 |
| e1_aggregate.py | 8cead3a82210ec30fb63cf38a0daf1b84d48e9994b9cfcd2d897bdc0ea3eb045 |
| e1_make_cells.py | 12bfe978c211549b3548342440dab0d115b9fcb8443b624f2842127dd97af61c |
| e1_cells.v1.json | d5aa5b6eb3da19bec9955f57162b2508c85b4d6413eedf93cfdd1f3157a75363 |
| e1_qualification_manifest.v1.json | ef79dd675b4318fe50a2a747bef8824d023226401328e75bff813c4d21dd13b8 |
| E1_FREEZE.md | dbd1605eee3ee06c39bc3267a5dd1253a68dbc86f19eea804692d91ebd066a40 |
| e1_recorder.py | 1cf3f53552e008c44f7b7e5e6187d8052bcfd4c75bd6f81e2e07d1f70fbf65a0 |
| e1_event_recorder_shim.py | 06344718f035949ea7f1cea49e4677e9acc2a71a460bbbc7b94e01a1bbbbc1df |
| e1_join.py | cbee2ae70947f1adb4073bd151a7148fcb05cbc02d1c5dff37b381ffadf1e766 |
| e1_api_tokens_from_capture.v2.py | 6d36ee15d8b5b6289c24ebc437f621dfd32adc010aae9ff1495fb34af0121f9f |

## F1 — final native gate failure leaves a VALID measured result

Runner 95 executes the summary, which writes VALID/INSUFFICIENT_SUPPORT to cell_result.json; native final audit occurs afterward at 97. The failure branch 105 exits 12 without invalidating that file. Aggregate 25 trusts a VALID file without checking native final eligibility. Resume counts that cell as already measured at runner 78/84 and skips it.

Independent CPU control used the exact runner with only the scratch WT path changed and API-compatible stubs for the driver/verifier/summary. The final native audit was deliberately made to fail. The **real aggregate** then read the resulting campaign:

```text
runner_rc 12
cell_result {'status': 'VALID', 'primary': {'tokens_per_wall_second': 123.0}}
qualification_exists False
aggregate_rc 0 first_cell VALID rate 123.0
```

This remains wrong even if a particular deterministic final check usually duplicates an earlier one: audit failure, missing artifact, process interruption, or incomplete finalization must not publish a measured cell. Minimal repair: finalize all required eligibility gates before publishing a terminal measured result, or persist a fail-closed finalization record and make resume/aggregate require it. Explicit final failure must mark the attempted cell INVALID; interrupted finalization must stay unmeasured. Preserve the failed attempt.

## F2 — two live inputs still bypass the campaign freeze

1. **GPU memory utilization / maximum model length:** cells' frozen_settings say 0.6 / 16384, but driver 134 uses ambient `GPU_UTIL` / `MAX_MODEL_LEN`. The verifier does not check either. A resume in a different shell can change them while every snapshot/route hash still matches.

   Independent CPU execution of the exact driver COMMON assignment with ambient values 0.77 / 8192 produced those nonfrozen values. The native verifier was run on its owner positive fixture with those explicit nonfrozen Cmd arguments appended and returned **VERIFY PASS (rc 0)**.

   Minimal repair: populate these values from the frozen cells configuration, and verify the actual container command against those frozen values. The same run must not silently change between attempts or arms.

2. **Native greedy reference:** preflight 26/54 reads the live default `experiments/out-20260921T231853Z-e7a-step2-native-score/prefix_scores.json`. Runner 29–33 neither copies nor hashes that file. The scores determine both the expected first token and which prefixes enter the ≥0.5 margin gate. The frozen gate therefore depends on a mutable unbound file outside the snapshot.

   Minimal repair: include the exact scores JSON in the campaign snapshot/hash set and pass that path explicitly to live and final native preflight. No new scoring or inference is needed.

## F3 — qualification manifest metadata corrections (parent-identified, independently confirmed)

Q's tree/B1 evidence string still says T9=2.36e-4 / T10=1; tree/B4 says T9≤4.79e-4 / T10=1. These are reversed. T9 is served bf16 ULP16 (1), T10 is served-output absolute error. The dated freeze now uses the correct labels.

Both Q reports_sha256 maps omit the actual final closure report `e2-b4-final-redteam.md` (105d5eb78029e70a…) and instead retain the earlier failed gate review. Preserve historical reports if useful, but add the final closure report's exact hash and clearly distinguish superseded reports. No new experiment is needed.

## Direct closure of the prior findings and CPU checks

- C1: heredoc dispatch compiles and the actual campaign snapshot/re-exec route runs under stubs.
- C2: driver 139–146 sets FR10_METRICS=0 and empty draft/LCP/sampler sinks; verifier 18 checks actual container env. Common recorder/timer remain. Native guarded stock fallback source proof from the previous report is unchanged; no pristine-upstream baseline is required.
- C3: summary 25–37 uniquely translates every API phase ID, with preflight + warmup assigned to the joiner's warmup exclusion. Missing/ambiguous IDs fail.
- C4: driver 162–170 now runs a distinct preflight client segment, requires live PASS, then submits fixed warmup/timed requests. Final complete API reconciliation includes preflight/warmup/timed output. F1 concerns the final eligibility publication boundary.
- C5: campaign snapshot, source identity checks, explicit SPEC_CONFIG, no overwrite and preserved attempts are implemented. F2 identifies the remaining specific inputs.
- C6: summary 43–48 maps valid complete rc3 zero-support joins to INSUFFICIENT_SUPPORT. Invalid/refused joins remain INVALID.
- The 18-cell balanced order, B1/B4 cohorts, direct token-ID request, fixed warmup and structural floors remain as prescribed. Slow valid intervals remain primary; elapsed-time clipping is diagnostic.
- Aggregate 29–42 uses whole paired boot-block rate differences, three blocks separately for B1/B4, 10,000 resamples, and the native5 mean-rate denominator. Incomplete paired contrasts are not estimable; insufficient-support cells provide no primary rate. No statistical-unit or contrast error found. The coarse three-block CI remains disclosed.
- The frozen native margin rule and first-four-cell live preflights were declared before native timing. Actual native serving qualification is still pending each first planned boot.

Independent reruns used temporary output paths, not canonical tests_out:

| CPU control | SHA-256 | Reproduced |
|---|---|---|
| test_e1_campaign_stub.sh | eb38e72dba64b02286de414808e6d29393a7fbe58dbdac1f551775243703af03 | 12/12, rc0 |
| test_e1_native_preflight_cpu.py | 8ffc9d43ada719f6c1fd281b27faae9849774a7711e19c32bf32dbc85d8bcf7a | 15/15, rc0 |
| test_e1_cell_summary_real_b1.sh | 73352db9285d24cb653083859857afc3f28a54a0b72e8376f2175f5b07ee6188 | 4/4, rc0 |
| test_e1_aggregate_cpu.py | 94dc8868b5cadd577db8f197dcde5a8cb712dffdb5a4ca355040693ecd0fbc09 | 5/5, rc0 |

These positive/negative owner controls pass; F1 and F2 are additional independent counterexamples they do not cover. No further benchmark axis is requested.

## Exact independent counterexamples

Executed remotely with `python3`; all created paths are temporary and automatically removed.

### F1: final audit failure through the real runner and aggregate

```python
from pathlib import Path
import tempfile,subprocess,json,os,shutil
E=Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/e1")
with tempfile.TemporaryDirectory(prefix="e1-final-negative-") as td:
 W=Path(td)/"wt"; E2=W/"papers/gdn-tree-scan-mlsys/v2/experiments/e1"; E2.mkdir(parents=True)
 (E2.parent/"e7a").mkdir(); (W/"scripts").mkdir(); (W/"src/lumo_flywheel_serving").mkdir(parents=True)
 files=["e1_run_cells.v2.sh","e1_cell_driver.v1.sh","e1_native_launch.v2.sh","e1_workload.py","e1_cell_verify.py","e1_cell_summary.py","e1_native_preflight.py","e1_recorder.py","e1_event_recorder_shim.py","e1_join.py","e1_api_tokens_from_capture.v2.py","E1_FREEZE.md"]
 for n in files: shutil.copy2(E/n,E2/n)
 runner=(E2/"e1_run_cells.v2.sh").read_text().replace("WT=/home/mark/lumo-paper-v2-20260921; EXP=",f"WT={W}; EXP=")
 (E2/"e1_run_cells.v2.sh").write_text(runner)
 # Native preflight final audit deliberately fails; driver and summary are API-compatible CPU stubs.
 (E2/"e1_cell_driver.v1.sh").write_text("""#!/usr/bin/env bash
set -eu
C="$3/$E1_RUN_NAME"
mkdir -p "$C/script_snapshot"
cp "$E1_SOURCE_DIR/e1_cell_verify.py" "$E1_SOURCE_DIR/e1_cell_summary.py" "$E1_SOURCE_DIR/e1_native_preflight.py" "$C/script_snapshot/"
echo preflight_verdict=PASS > "$C/driver_trace.txt"
""")
 (E2/"e1_cell_verify.py").write_text("import sys; sys.exit(0)\n")
 (E2/"e1_cell_summary.py").write_text("import json,sys; json.dump({'status':'VALID','primary':{'tokens_per_wall_second':123.0}},open(sys.argv[1]+'/cell_result.json','w'))\n")
 (E2/"e1_native_preflight.py").write_text("import sys; print('FAIL final audit injected CPU control'); sys.exit(1)\n")
 for f in ["scripts/gpu_oom_guard.sh","scripts/fr10_phase4_patch_vllm_tree_gdn.py","src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py","src/lumo_flywheel_serving/fr10_decode_modes.py"]: (W/f).write_text("# CPU stub\n")
 (E2.parent/"e7a/e7a_capture_launch.v7.sh").write_text("# CPU stub\n")
 shutil.copy2(E/"e1_cells.v1.json",E2/"cells.json")
 (E2/"qual.json").write_text(json.dumps({"qualified":{"native-5/B1:preflight":{}}}))
 for n in ["pool.json","frozen.json"]: (E2/n).write_text("{}")
 B=Path(td)/"bin"; B.mkdir()
 for n,body in [("docker","exit 0"),("pgrep","exit 1")]:
  (B/n).write_text("#!/bin/sh\n"+body+"\n"); (B/n).chmod(0o755)
 subprocess.run(["git","init","-q",str(W)],check=True)
 subprocess.run(["git","-C",str(W),"add","."],check=True)
 subprocess.run(["git","-C",str(W),"-c","user.name=CPU","-c","user.email=cpu@invalid","commit","-qm","CPU"],check=True)
 R=Path(td)/"campaign"
 env=dict(os.environ,PATH=str(B)+":"+os.environ["PATH"],E1_QUAL_MANIFEST=str(E2/"qual.json"),E1_POOL=str(E2/"pool.json"),E1_FROZEN=str(E2/"frozen.json"))
 p=subprocess.run(["bash",str(E2/"e1_run_cells.v2.sh"),str(E2/"cells.json"),str(R),"1","1"],env=env,text=True,capture_output=True)
 c=R/"cell_01_b1_native-5_B1_a1"/"cell_result.json"
 print("runner_rc",p.returncode,"cell_result",json.loads(c.read_text()),"qualification_exists",(R/"qualification.json").exists())
 a=subprocess.run(["python3",str(E/"e1_aggregate.py"),str(R)],text=True,capture_output=True)
 j=json.loads((R/"aggregate.json").read_text())
 print("aggregate_rc",a.returncode,"first_cell",j["cells"][0]["status"],"rate",j["cells"][0]["rate"])

```

### F2: nonfrozen Cmd accepted by the verifier

```python
from pathlib import Path
import ast,tempfile,json,subprocess,os
E=Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/e1")
t=ast.parse((E/"test_e1_native_preflight_cpu.py").read_text())
nodes=[]
for n in t.body:
 if isinstance(n,ast.With): break
 nodes.append(n)
ns={"__file__":str(E/"test_e1_native_preflight_cpu.py")}
exec(compile(ast.Module(body=nodes,type_ignores=[]),"<fixture>","exec"),ns)
with tempfile.TemporaryDirectory(prefix="e1-env-negative-") as td:
 ns["make"](td)
 p=Path(td)/"docker_inspect.json"; j=json.loads(p.read_text())
 j[0]["Config"]["Cmd"]+=["--gpu-memory-utilization","0.77","--max-model-len","8192"]; p.write_text(json.dumps(j))
 rc,o=ns["ver"](td)
 print("nonfrozen_config_verifier_rc",rc,"tail",o.splitlines()[-1])
s=(E/"e1_cell_driver.v1.sh").read_text(); common=s[s.index("COMMON=("):s.index("\ncase ",s.index("COMMON=("))]
prefix="WT=x; C=x; PORT=9950; NSEQ=1; RUN=x; SPEC_CONFIG_FROZEN=x; TREE_FROZEN=x; NSPEC=5; RUNREL=x\n"
r=subprocess.run(["bash","-c",prefix+common+"\nprintf '%s\\n' \"\u0024{COMMON[@]}\""],env=dict(os.environ,GPU_UTIL="0.77",MAX_MODEL_LEN="8192"),capture_output=True,text=True)
print("actual_driver_COMMON", [x for x in r.stdout.splitlines() if x.startswith(("GPU_UTIL=","MAX_MODEL_LEN="))])
print("frozen",json.loads((E/"e1_cells.v1.json").read_text())["frozen_settings"]["gpu_util"],json.loads((E/"e1_cells.v1.json").read_text())["frozen_settings"]["max_model_len"])

```
 


## Reply-25 recheck — 2026-09-22, after 09:50 UTC

**F1, F2, and Q metadata are CLOSED on the exact versions below.** The previous counterexamples remain preserved above. The additional parent-requested source-identity check found two concrete runtime dependencies still omitted from the guard (last paragraph); add those before first data. No numerical/helper qualification is reopened and no new experiment is requested.

| Updated source | Independently verified SHA-256 |
|---|---|
| e1_run_cells.v2.sh | 08e3aa21b08cdd5b80b801c91eb54de91e8ab5f15923be2b802189df9bd4e082 |
| e1_cell_driver.v1.sh | 25aa6c2195957630f9554961a799426806529d07fa779d2b7042ebeaee31cfc4 |
| e1_cell_summary.py | 04a119321a2aa95aec826c56f229800ea4f2bf2e5126b8bfdce4875939759dff |
| e1_cell_verify.py | e4c73148de56d75bcc92b6224f714958cd3e29e29d4f829ec09bc495958fb485 |
| e1_native_preflight.py | 6dd79084a8a7dcec0a5a84432bd768ddc64cd73c1a7da96d6da480a732ba7562 |
| e1_aggregate.py | 98beec9efac26f560783d9d77af3e2154edbd6b24ebadfe141b0cafce6e48a68 |
| e1_qualification_manifest.v1.json | c60a4a4c661e84a6bc8ab6baade49a1ef7f5d0cf298f8606c937894a387216da |
| E1_FREEZE.md | 11f5dd00b8d8b7073575f10cd9a00a4282760746be05f16bb9c7d188fa027a4e |

F1: summary 88–92 writes only cell_result.preliminary.json. Runner 117–136 publishes the terminal result only after the required gates; final-audit failure seals INVALID, removes primary, and preserves the preliminary result. Aggregate 23–29 refuses missing/unsealed results; resume 94–105 counts only eligible sealed results. Explicit failed or unsealed/aborted finalization cannot reproduce the old measured VALID false acceptance. A partial terminal JSON interrupted during writing may halt a reader rather than classify the cell; it does not yield eligible timing data.

F2 allocation: driver 125–137 reads the values from frozen_settings into COMMON; runner 117 passes the same values to verifier, which tests the actual Cmd at 19–20. F2 reference: runner 26/34 snapshots the scores before SHA256SUMS; driver 114–115 includes the scores in the cell snapshot; driver 167 and runner 123 explicitly pass frozen paths to live/final checks. Preflight 26 requires --scores and 28 records its hash.

Q: T9 now explicitly identifies served bf16 ULP16=1, T10 the served-output absolute errors. Both tree entries carry the final B4 closure report SHA `105d5eb78029e70a864cbefa3cb71a2886f94733d9a107055be59ef77d654788`; the local report bytes independently match that value. B4 batch order is p072,p095,p017,p085. These metadata corrections are closed.

Targeted CPU reruns, using temporary output paths and the real settled scripts:

| Control | SHA-256 | Result |
|---|---|---|
| test_e1_runner_terminal_gate.sh | 81cd0db047f7b930c8cf54788d1d8822bf522581f507cf7d7eff5a6fbca4fa33 | 9/9 PASS |
| test_e1_campaign_stub.sh | 1c72d8d1bb7e55a7feae6bafc9b5b44dfe70a24d755e5f4c1bd92e3d4009c2ba | 13/13 PASS |
| test_e1_native_preflight_cpu.py | e73cdffb20361a8b856508843542be68e7180a52e7bd658818a17303909670f6 | 17/17 PASS |
| test_e1_cell_summary_real_b1.sh | f0462d3567ab8b8f48600b92a2b43e5b0a51cf3225650702c640014ff926b731 | 5/5 PASS |
| test_e1_aggregate_cpu.py | 6c5aaf74c71c2f389dc7aaf88b9c12fcdeb0a7f0e943b17e2d7e7531ebebfcde | 7/7 PASS |

All **51 controls pass**, all commands returned zero. The terminal-gate rerun covers the exact previous failed-final scenario (preliminary VALID123 retained separately, sealed INVALID, aggregate rate None, resume rc46), successful explicit retry in a new attempt, and no repeat of a measured cell. Allocation drift and missing terminal seals now fail. The CPU fixtures use stubbed launchers; no container was started or GPU inference performed. All scratch directories created by this review were removed.

### Additional parent-requested source identity check: two actual imports remain unguarded

Runner 37 lists only patcher, fr10_gdn_tree_kernel.py, fr10_decode_modes.py, and the OOM guard. Both launchers mount the live worktree at /workspace. The current patcher actually imports/loads these additional files, yet neither is in that identity list:

- `scripts/fr13_device_multidraft_kernel.py`: SHA `648ff0e33e8a7cdd0fbe8ebb3d2befaa27002a3347f94f9bdc0668365ed2e8e9`. Patcher 20531/28055/28309 directly names the /workspace file for dynamic loading.
- `src/lumo_flywheel_serving/fr13_tree_conv_fused.py`: SHA `564c7dc7ba61ccd9579f2fb7a4af9a2cdaaa1753dfb98434f2011457d90d9a6e`. Patcher 11730/11871 imports it for the served fused-conv route.

Minimal remaining delta: add these two paths to campaign_identity before first data, so the existing before-each-cell hash loop also guards them. This is an exact dependency omission within the requested freeze scope, not a new runtime qualification requirement. F1/F2/Q have no remaining material finding.
