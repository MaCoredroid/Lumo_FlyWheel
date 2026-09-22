# First timed tree cells: independent raw audit

Audit started 2026-09-22 10:35 UTC. Read-only source/config/JSON audit; no GPU work, inference, old test suites, campaign writes, or extra boots. Scope is first planned tree/B1 cell 5 and tree/B4 cell 6 in the authorized 18-cell campaign. Native first-cell qualification is documented separately.

Remote root: `/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T100028Z-e1-18cells`.

## Finding T1: distinguish request seed from actual engine seed

Actual tree cell 5 `docker_inspect.json` contains no `--seed` argument and no seed environment variable. `docker_logs.txt` EngineCore initialization explicitly reports `seed=0`. The four native cells use `--seed '20260921'`. Every inspected tree API body does pass `seed=20260921`, matching the frozen workload. The tree launcher's lack of engine seed handling therefore makes an unqualified statement that both engine settings share seed 20260921 inaccurate. Preserve the actual configuration and distinguish these seed scopes in the measurement record; do not silently relabel engine seed 0 as 20260921.

Bounded source follow-up does **not** establish corrupt sampling or require replacement boots. Loaded `gpu_model_runner.py:1152–1159` creates a request generator only for `SamplingType.RANDOM_SEED`; the greedy route has no such request generator. Loaded `rejection_sampler.py:3762–3775` sends `all_greedy=True` and that generator map to the device committer via lines 2751–2761. Actual dependency `scripts/fr13_device_multidraft_kernel.py:1034–1058` excludes TAW/depthsync when greedy; lines 1098–1108 create a fresh explicit `torch.Generator` per request/call and only manually seed it when a request generator is present. Greedy child selection passes that explicit generator at lines 1149–1150. The global engine seed is therefore not read at this observed selector. Duplicate siblings are still selected through the already qualified point-mass sampler; temperature zero should not be described as avoiding every internal RNG operation.

Identities: loaded runner `b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40`; loaded sampler `5be4ad43409da82b63bfd0177926467e9a2b59da553ee6dd04d3130a5d77b12f`; device dependency `648ff0e33e8a7cdd0fbe8ebb3d2befaa27002a3347f94f9bdc0668365ed2e8e9`. This is an exact configuration/reporting discrepancy, not a demonstrated numerical or rate error. Parent was notified before tree measurement closed.

## Cell 5: raw support PASS, seed scope caveat above

Cell `cell_05_b1_tree_B1_a1` sealed VALID at 10:40:52 UTC. The independent reconstruction matched saved join and primary measurement exactly: **995 emitted API-bound tokens / 78.98466038424522 s = 12.597382772294214 tokens/s**, across **314 unique physical intervals**. All eight frozen prompts have 34–46 qualifying intervals. No valid interval exceeds the 1.5 s diagnostic cutoff. Warmup is excluded; all warmup and timed output streams, including ordinary prefill/mixed outputs, reconcile to direct API token IDs. The only clipping is the last output row of each affected request.

Verified actual cat10/depth5 nine-draft config, tree MTP/GDN, `TREE_ATTN`, eager, synchronous scheduling, prefix cache off, actual max sequences 1, memory utilization 0.6, maximum model length 16384, policy B remap=1/reorder=0/syncfree=1. `FR13_TREE_RUNROW_INIT` is absent and uses reviewed default 1; this is not a failed config gate. Metrics and heavy capture sinks are off. Five loaded patched modules match the previously reviewed hashes. Both immutable snapshot manifests verify. Recorder probe, strictly increasing event/forward counters, request identities, output lengths, full-phase reconciliation, physical occupancy and same-request successor support, structural floor, run-close counts, and terminal seal pass. Tree eligibility comes from frozen E2 qualification, not native live preflight.

Raw reconstruction result:

```json
{
  "cell": 5,
  "arm": "tree",
  "batch": 1,
  "status": "VALID",
  "seal_utc": "2026-09-22T10:40:52Z",
  "gates": {
    "cell_verify.json": 16
  },
  "events": 1082,
  "forwards": 360,
  "physical_steps": 346,
  "output_records": 360,
  "nonpure_outputs": 14,
  "phase_requests": {
    "warmup": 2,
    "timed": 8
  },
  "phase_api_tokens": {
    "timed": 1024,
    "warmup": 64
  },
  "api_tokens": 1088,
  "ledger_tokens": 1102,
  "clipped_tokens": 14,
  "clipped_requests": 8,
  "support": {
    "intervals": 314,
    "tokens": 995,
    "wall_s": 78.98466038424522,
    "rate": 12.597382772294214,
    "over_1p5_s": 0
  },
  "excluded": {
    "preflight/warmup": 24,
    "intervening forward": 7,
    "terminal": 1
  },
  "excluded_walls": {
    "preflight/warmup": 13.129355503246188,
    "intervening forward": 15.115963538177311
  },
  "per_unit": {
    "p072": {
      "intervals": 39,
      "tokens": 125,
      "wall_s": 9.712802954949439
    },
    "p017": {
      "intervals": 40,
      "tokens": 123,
      "wall_s": 9.992829789407551
    },
    "p015": {
      "intervals": 40,
      "tokens": 123,
      "wall_s": 9.965609173290431
    },
    "p021": {
      "intervals": 38,
      "tokens": 124,
      "wall_s": 9.483131636865437
    },
    "p085": {
      "intervals": 46,
      "tokens": 126,
      "wall_s": 11.674322916194797
    },
    "p095": {
      "intervals": 42,
      "tokens": 123,
      "wall_s": 10.641424522735178
    },
    "p058": {
      "intervals": 35,
      "tokens": 126,
      "wall_s": 8.903224444948137
    },
    "p083": {
      "intervals": 34,
      "tokens": 125,
      "wall_s": 8.611314945854247
    }
  },
  "finish_counts": {
    "length": 10
  },
  "hashes": {
    "logs/e1_events.jsonl": "395c3e5b97b2b292bdc2e297f921bad588fa854c5005e0d6ffe71d71eac86b71",
    "e1_manifest.json": "183714818e4421e393bfa995a4827f573b123bf42865626b430d2a07adfc738b",
    "docker_inspect.json": "98fbcdb5b367dbe8b171d8b9b98daab74bd94c55a2ef26852417d9f4898e842c",
    "cell_result.json": "4ee281136878a174de81289b815b807ebcdd4f69f1087fa897d6c94136e61004",
    "e1_join.json": "8b7b9174b3082abb0392e46714003b774e7ed87bb7b4f59f27c8ec55661f05ee"
  },
  "loaded_hashes": {
    "gpu_model_runner.py": "b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40",
    "eagle.py": "aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7",
    "tree_attn.py": "a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97",
    "gdn_linear_attn.py": "723974c703d1ad1545399b78526e5d50ae7f7f362e434bad384a0a3327595b28",
    "rejection_sampler.py": "5be4ad43409da82b63bfd0177926467e9a2b59da553ee6dd04d3130a5d77b12f"
  },
  "seed_scope": {
    "request_seed": 20260921,
    "engine_seed": 0,
    "tree_engine_seed_deviation_recorded": true
  }
}
```

## Independent reproducer

Command used (code below is sent on stdin; no remote files written):

```sh
ssh mark@100.103.10.122 'python3 - 5' <<'PY'
# code below
PY
```

For cell 6, replace argument 5 with 6. The code explicitly records the engine seed deviation; its raw-support PASS does not assert identical engine seeds.

```python
from pathlib import Path
import json,hashlib,math,sys,re
from collections import Counter,defaultdict
R=Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T100028Z-e1-18cells")
S=R/"campaign_snapshot"; idx=int(sys.argv[1])
cell=next(c for c in json.loads((S/"e1_cells.json").read_text())["cells"] if c["index"]==idx)
N=cell["max_num_seqs"]; K=cell["num_speculative_tokens"]; is_native=cell["arm"]!="tree"
C=R/("cell_%02d_b%d_%s_%s_a1"%(idx,cell["block"],cell["arm"],cell["batch"]))
result=json.loads((C/"cell_result.json").read_text())
assert result["terminal_seal"]["sealed"] and result["status"] in ("VALID","INSUFFICIENT_SUPPORT"),result
assert result["terminal_seal"]["campaign_snapshot_sha256sums"]==hashlib.sha256((S/"SHA256SUMS").read_bytes()).hexdigest()
assert result["terminal_seal"]["gates"]["verify"]=="PASS"
if is_native: assert result["terminal_seal"]["gates"]["preflight_live"]=="PASS" and result["terminal_seal"]["gates"]["preflight_final"]=="PASS"
for D in [S,C/"script_snapshot"]:
 for line in (D/"SHA256SUMS").read_text().splitlines():
  h,n=line.split(None,1); assert hashlib.sha256((D/n.lstrip("*")).read_bytes()).hexdigest()==h,(D,n)
cfg=json.loads((C/"docker_inspect.json").read_text())[0]; env=dict(x.split("=",1) for x in cfg["Config"]["Env"] if "=" in x); cmd=" ".join(cfg["Config"]["Cmd"])
assert env["SPEC_CONFIG"]==cell["spec_config"]
if is_native: assert env["FR10_DECODE_MODE_DEFAULT"]=="naive_mtp" and env["FR10_ENABLE_TREE_GDN"]=="0"
else:
 assert env["FR10_DECODE_MODE_DEFAULT"]=="tree_mtp" and env["FR10_ENABLE_TREE_GDN"]=="1"
 for key,value in {"FR13_ATTN_KV_REMAP":"1","FR13_SLOT_REORDER":"0","FR13_KV_REMAP_SYNCFREE":"1","FR13_REPLAY_ROUTE":"1","FR13_TREE_RUNROW_INIT":"1"}.items(): assert env.get(key, "1" if key=="FR13_TREE_RUNROW_INIT" else None)==value,(key,env.get(key))
 assert env.get("FR13_EAGER_PACK","1")=="1" and env.get("E7A_CAPTURE_SHIM","0")=="0" and not env.get("E7B_SUBSTITUTE")
for flag in ["--no-async-scheduling","--no-enable-prefix-caching","--enforce-eager",("--attention-backend 'FLASH_ATTN'" if is_native else "--attention-backend 'TREE_ATTN'"),f"--max-num-seqs '{N}'","--gpu-memory-utilization '0.6'","--max-model-len '16384'"]: assert flag in cmd,flag
if is_native: assert "--seed '20260921'" in cmd
else: assert "--seed " not in cmd # observed engine-default 0, tracked configuration deviation
assert cfg["Config"]["Image"].endswith("3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776")
assert any(m["Destination"]=="/models" and not m["RW"] for m in cfg["Mounts"])
assert env.get("FR10_METRICS","0")!="1" and env.get("FR13_SFWD_GPU_TIMER")=="1" and env.get("E1_RECORD")=="/logs/e1_events.jsonl"
for key in ["LUMO_MTP_DRAFT_TRACE_FILE","LUMO_TREE_PATH_LCP_LOG","LUMO_TREE_SAMPLER_DEBUG_LOG","FR13_FINAL_LOGIT_CAPTURE","FR10_LAYER_HIDDEN_CAPTURE","FR10_TREE_GDN_CAPTURE_PAYLOAD"]: assert not env.get(key)
log=(C/"docker_logs.txt").read_text(errors="replace")
if is_native: assert "ENGAGED" not in log
else: assert "FR13_ATTN_KV_REMAP ENGAGED" in log and "FR13_SLOT_REORDER ENGAGED" not in log
gates={}
for name in (["native_preflight_live.json","native_preflight_final.json","cell_verify.json"] if is_native else ["cell_verify.json"]):
 j=json.loads((C/name).read_text()); assert j["all_pass"] and all(c["pass"] for c in j["checks"])
 gates[name]=len(j["checks"])
 if "preflight" in name: assert j["reference_scores"]["sha256"]==hashlib.sha256((S/"prefix_scores.json").read_bytes()).hexdigest()
trace=(C/"driver_trace.txt").read_text()
if is_native: assert trace.index("preflight_verdict=PASS")<trace.index("workload_start_utc")
ev=[json.loads(x) for x in (C/"logs/e1_events.jsonl").read_text().splitlines() if x.strip()]
assert [x["n"] for x in ev]==list(range(1,len(ev)+1))
assert ev[0]["event"]=="recorder_probe" and ev[-1]["event"]=="run_close"
assert ev[-1]["n_events"]==len(ev) and not ev[-1].get("sink_failed")
assert not (C/"logs/e1_recorder_FAILED.flag").exists()
assert not any(x["event"]=="error" or x.get("errors",0)>0 or x.get("sink_failures",0)>0 for x in ev)
fe={}; ps={}; outs={}; breaks=set()
for e in ev:
 k=e["event"]
 if k=="forward_entry": assert e["seq"] not in fe; fe[e["seq"]]=e
 if k=="physical_step": assert e["physical_step_id"] not in ps; ps[e["physical_step_id"]]=e
 if k=="output_rows": assert e["seq"] not in outs; outs[e["seq"]]=e
 if k=="chain_break": breaks.add(e["seq"])
assert sorted(fe)==list(range(min(fe),max(fe)+1))
assert ev[-1]["n_forwards"]==len(fe) and ev[-1]["n_physical_steps"]==len(ps) and ev[-1]["n_output_records"]==len(outs)
byreq=defaultdict(list)
for seq,o in sorted(outs.items()):
 assert seq in fe and o["kind"] in ["pure","nonpure"]
 if o["kind"]=="pure": assert ps[o["physical_step_id"]]["seq"]==seq
 for row in o["rows"]:
  rid=row["request_id"]; assert row["step_idx"]==len(byreq[rid])
  assert row["n_emitted"]==len(row["emitted_ids"])
  byreq[rid].append((seq,row["emitted_ids"]))
for pid,p in ps.items():
 o=outs[p["seq"]]; rids=[x["request_id"] for x in o["rows"]]
 assert o["kind"]=="pure" and len(rids)==len(set(rids))==p["num_reqs"]
 assert set(rids)==set(p["request_ids"]) and 1<=p["num_reqs"]<=N
man=json.loads((C/"e1_manifest.json").read_text())
frozen=json.loads((S/"frozen_prefixes.json").read_text()); expected=[x["id"] for x in frozen["pilot"]]
prefhash={x["id"]:x["prefix_sha256"] for x in frozen["pilot"]}
pool={x["id"]:x for x in json.loads((S/"prefix_pool.json").read_text())["prefixes"]}
phases={}; rid2meta={}; api={}; bounds={}; trunc={}
for m in man["requests"]:
 rid=[r for r in byreq if r==m["response_id"] or r.startswith(m["response_id"]+"-")]
 assert len(rid)==1; rid=rid[0]
 cap=C/"cohort"/("req_"+m["slot"]+"_"+m["prefix_id"])/"capture_request.json"
 c=json.loads(cap.read_text()); toks=c["response_logprobs_tokens"]
 assert all(isinstance(x,str) and x.startswith("token_id:") for x in toks)
 ids=[int(x[9:]) for x in toks]
 assert len(ids)==m["n_api_tokens"] and c["response_id"]==m["response_id"]
 req=c["request"]; assert req["return_tokens_as_token_ids"] is True and req["temperature"]==0 and req["seed"]==20260921
 assert c["max_tokens"]==(128 if m["phase"]=="timed" else 32)
 assert m["finish_reason"] in ["length","stop"]
 if m["finish_reason"]=="length": assert len(ids)==m["max_tokens"]
 text=pool[m["prefix_id"]]["text"]
 assert hashlib.sha256(text.encode()).hexdigest()==prefhash[m["prefix_id"]]==c["prompt_sha256"]==c["prefix_sha256"]
 body={"model":req["model"],"prompt":text,"max_tokens":req["max_tokens"],"temperature":req["temperature"],"seed":req["seed"],"logprobs":req["logprobs"],"echo":req["echo"],"return_tokens_as_token_ids":req["return_tokens_as_token_ids"]}
 assert hashlib.sha256(json.dumps(body).encode()).hexdigest()==c["request_body_sha256"]
 stream=[t for _,rowids in byreq[rid] for t in rowids]; counts={seq:len(rowids) for seq,rowids in byreq[rid]}
 if stream!=ids:
  lastseq,lastids=byreq[rid][-1]; headlen=len(stream)-len(lastids)
  assert len(ids)<len(stream) and stream[:len(ids)]==ids and len(ids)>=headlen
  counts[lastseq]=len(ids)-headlen; trunc[rid]=len(stream)-len(ids)
 api[rid]=ids; bounds[rid]=counts; phases[rid]=m["phase"]; rid2meta[rid]=m
assert set(phases)==set(byreq)
assert [m["prefix_id"] for m in man["requests"] if m["phase"]=="timed"]==expected
expected_phase_counts={"warmup":(2 if N==1 else 4),"timed":8}
if is_native: expected_phase_counts["preflight"]=8
assert dict(Counter(phases.values()))==expected_phase_counts
cohorts={frozenset(expected[:4]):"A(1-4)",frozenset(expected[4:]):"B(5-8)"}
ret=[]; excluded=Counter(); per=defaultdict(lambda:{"intervals":0,"tokens":0,"wall_s":0.0}); excwalls=defaultdict(float)
for pid,p in sorted(ps.items()):
 nxt=ps.get(pid+1); rids=p["request_ids"]; reason=None
 if nxt is None: reason="terminal"
 elif any(phases[rid]!="timed" for rid in rids): reason="preflight/warmup"
 elif nxt["seq"]!=p["seq"]+1: reason="intervening forward"
 elif any(p["seq"]<=b<=nxt["seq"] for b in breaks): reason="chain break"
 elif set(nxt["request_ids"])!=set(rids): reason="cohort change"
 elif len(rids)!=N: reason="occupancy ramp/drain"
 elif any(x.get("discarded") for x in outs[p["seq"]]["rows"]): reason="discarded"
 if reason:
  excluded[reason]+=1
  if nxt: excwalls[reason]+=nxt["t_start"]-p["t_start"]
  continue
 wall=nxt["t_start"]-p["t_start"]; assert wall>0
 tok=sum(bounds[rid][p["seq"]] for rid in rids); ret.append((pid,p["seq"],tok,wall))
 name=rid2meta[rids[0]]["prefix_id"] if N==1 else cohorts[frozenset(rid2meta[r]["prefix_id"] for r in rids)]
 unit=per[name]; unit["intervals"]+=1; unit["tokens"]+=tok; unit["wall_s"]+=wall
T=sum(x[2] for x in ret); W=sum(x[3] for x in ret); rate=T/W if W else None
j=json.loads((C/"e1_join.json").read_text())
assert j["n_usable"]==len(ret) and j["sum_emitted_tokens_api_bound_pure_support"]==T
assert math.isclose(j["sum_wall_s_unique_physical_steps"],W,rel_tol=1e-12)
assert j["tokens_per_wall_second"]==rate or math.isclose(j["tokens_per_wall_second"],rate,rel_tol=1e-12)
assert set(x["physical_step_id"] for x in j["steps"])==set(x[0] for x in ret)
if result["status"]=="VALID":
 assert set(per)==(set(expected) if N==1 else set(cohorts.values())) and all(x["intervals"]>=1 for x in per.values())
 assert result["primary"]["sum_emitted_tokens_api_bound_pure_support"]==T
 assert math.isclose(result["primary"]["tokens_per_wall_second"],rate,rel_tol=1e-12)
qual=dict(json.loads((S/"e1_qualification_manifest.json").read_text())["qualified"])
if (R/"qualification.json").exists(): qual.update(json.loads((R/"qualification.json").read_text())["qualified"])
assert cell["arm"]+"/"+cell["batch"] in qual
summary={"cell":idx,"arm":cell["arm"],"batch":N,"status":result["status"],"seal_utc":result["terminal_seal"]["utc"],"gates":gates,
"events":len(ev),"forwards":len(fe),"physical_steps":len(ps),"output_records":len(outs),"nonpure_outputs":sum(o["kind"]=="nonpure" for o in outs.values()),
"phase_requests":dict(Counter(phases.values())),"phase_api_tokens":{ph:sum(len(api[r]) for r in api if phases[r]==ph) for ph in set(phases.values())},
"api_tokens":sum(map(len,api.values())),"ledger_tokens":sum(len(ids) for seqs in byreq.values() for _,ids in seqs),"clipped_tokens":sum(trunc.values()),"clipped_requests":len(trunc),
"support":{"intervals":len(ret),"tokens":T,"wall_s":W,"rate":rate,"over_1p5_s":sum(x[3]>1.5 for x in ret)},
"excluded":dict(excluded),"excluded_walls":dict(excwalls),"per_unit":dict(per),"finish_counts":dict(Counter(m["finish_reason"] for m in man["requests"]))}
hashes={}
for n in ["logs/e1_events.jsonl","e1_manifest.json","docker_inspect.json","cell_result.json","e1_join.json","native_preflight_live.json","native_preflight_final.json"]:
 p=C/n
 if p.exists(): hashes[n]=hashlib.sha256(p.read_bytes()).hexdigest()
summary["hashes"]=hashes
summary["loaded_hashes"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (C/"loaded_backend").glob("*.py")}
summary["seed_scope"]={"request_seed":20260921,"engine_seed":20260921 if is_native else 0,"tree_engine_seed_deviation_recorded":not is_native}
print(json.dumps(summary,indent=1))
print("PASS independent raw support variant (tree engine seed deviation recorded)",idx)

```

## T1 disposition, 2026-09-22 10:44–10:45 UTC

Parent independently reproduced the loaded-source chain and confirmed the freeze's original wording says settings are identical across arms, including seed 20260921. The discrepancy is preserved as an **as-executed configuration deviation**, not a silent reinterpretation of the freeze. All 18 cells continue unchanged. Direct API/interval validity stands; a runner VALID result and this raw-support PASS do not imply perfect frozen-configuration compliance. Final comparison must state request seed 20260921, native engine seed 20260921, and tree engine seed 0. The source establishes no engine-global seed input at the observed greedy selector; it does **not** establish whole-engine seed invariance. The worker is recording a separate addendum/STATUS entry. No extra boot is justified solely by this bounded finding for the honestly scoped as-executed comparison.

## Cell 6

Cell `cell_06_b1_tree_B4_a1` sealed VALID at 10:47:25 UTC. Independent raw-support PASS: **684 emitted API-bound tokens / 14.354477006942034 s = 47.65063886822262 tokens/s**, across **51 unique full-B4 intervals**. Cohort A (first four frozen prefixes) supplies 26 intervals, 339 tokens, 7.229520715773106 s; cohort B supplies 25 intervals, 345 tokens, 7.124956291168928 s. Both structural floors pass. Thirty-three occupancy-ramp/drain intervals and six cohort-change intervals are excluded by the frozen support definition; none is relabeled full B4. The 13.641744510270655 s gap contains an intervening nonpure forward and is excluded for that reason, not because it exceeds the diagnostic cutoff. No retained valid interval exceeds 1.5 s.

All 12 requests (4 warmup, 8 timed) use direct API IDs, the exact frozen prompt/request bodies, temperature 0 and request seed 20260921; all finish at the declared length. The ledger's 1,172 IDs reconcile to 1,152 API IDs, with 20 clipped IDs across 9 requests, entirely within final output rows. The full warmup and timed streams reconcile before selecting physical support. Probe/end counts and terminal seal pass; saved rate and exact physical-step support equal the independent reconstruction.

Actual configuration matches cell 5 with maximum sequences 4. The same five loaded modules match; immutable campaign and per-cell snapshots verify. Policy B remap/reorder/syncfree, synchronous eager execution, disabled prefix cache, and absence of heavy capture are verified. The same documented engine-seed deviation applies: actual startup is seed 0, API seed is 20260921. No new material raw-measurement issue was found.

```json
{
  "cell": 6,
  "arm": "tree",
  "batch": 4,
  "status": "VALID",
  "seal_utc": "2026-09-22T10:47:25Z",
  "gates": {
    "cell_verify.json": 16
  },
  "events": 356,
  "forwards": 118,
  "physical_steps": 105,
  "output_records": 118,
  "nonpure_outputs": 13,
  "phase_requests": {
    "warmup": 4,
    "timed": 8
  },
  "phase_api_tokens": {
    "timed": 1024,
    "warmup": 128
  },
  "api_tokens": 1152,
  "ledger_tokens": 1172,
  "clipped_tokens": 20,
  "clipped_requests": 9,
  "support": {
    "intervals": 51,
    "tokens": 684,
    "wall_s": 14.354477006942034,
    "rate": 47.65063886822262,
    "over_1p5_s": 0
  },
  "excluded": {
    "preflight/warmup": 13,
    "cohort change": 6,
    "occupancy ramp/drain": 33,
    "intervening forward": 1,
    "terminal": 1
  },
  "excluded_walls": {
    "preflight/warmup": 12.286753243766725,
    "cohort change": 1.5898460363969207,
    "occupancy ramp/drain": 8.678084421902895,
    "intervening forward": 13.641744510270655
  },
  "per_unit": {
    "A(1-4)": {
      "intervals": 26,
      "tokens": 339,
      "wall_s": 7.229520715773106
    },
    "B(5-8)": {
      "intervals": 25,
      "tokens": 345,
      "wall_s": 7.124956291168928
    }
  },
  "finish_counts": {
    "length": 12
  },
  "hashes": {
    "logs/e1_events.jsonl": "7b14d94c9f3f15c43aaa629bca7de21b111f8600f421cf0c7fc13d197abaa1e3",
    "e1_manifest.json": "37a0e9c6802d163104792c92d526af2b79a80ef56f4e367caa8ff5f645eaf253",
    "docker_inspect.json": "c902265545110b4cb2342563ef8be854e292ed646524ae65d7a2ab684608aab5",
    "cell_result.json": "f3849d486615b96d1b88d652e1ef0aa135e65ce9883aaad7472ba34d84411cb4",
    "e1_join.json": "b87429a24d7f24e44191272a57b6f5979bd5bf04a4dc931e7c0b50f2faaf1bdb"
  },
  "loaded_hashes": {
    "gpu_model_runner.py": "b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40",
    "eagle.py": "aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7",
    "tree_attn.py": "a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97",
    "gdn_linear_attn.py": "723974c703d1ad1545399b78526e5d50ae7f7f362e434bad384a0a3327595b28",
    "rejection_sampler.py": "5be4ad43409da82b63bfd0177926467e9a2b59da553ee6dd04d3130a5d77b12f"
  },
  "seed_scope": {
    "request_seed": 20260921,
    "engine_seed": 0,
    "tree_engine_seed_deviation_recorded": true
  }
}
```

Completed saved startup logs independently confirm engine seed 0 for both tree cells. Their SHA256 identities are cell 5 `8b7678970e0af0bd51a83bce59c88be82532c8e7efe837e3bd3ab536f2ad49bf`, cell 6 `1131efb004788720c7188884d9fe4d9b3f6488ec5e362f9183fb65467980dea1`. Frozen tree launcher `e7a_capture_launch.v7.sh` is `a5c398ffff43d1d8ff4678e8df3750987cd520e7710d044cb4cbeb3c1bee7f72`; native launcher `e1_native_launch.v2.sh` is `5d86340e683a92c64caabee51bbdd2514cae8e9a1f96c6299dfd14500dba2658`.

Bounded verdict at 10:48 UTC: both first tree cells have valid, independently reconstructed API-bound physical measurements. The seed configuration deviation is preserved for as-executed interpretation; neither runner VALID nor raw-support PASS certifies exact compliance with the original identical-seed wording. No other actionable material issue remains in this assigned two-cell surface. Later blocks, final paired aggregation and final artifact refresh remain separate work.
