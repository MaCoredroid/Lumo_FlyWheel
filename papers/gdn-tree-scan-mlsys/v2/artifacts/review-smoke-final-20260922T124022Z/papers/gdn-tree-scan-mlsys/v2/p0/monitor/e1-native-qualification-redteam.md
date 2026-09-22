# Native E1 variants — independent raw qualification review

Read-only continuation of the first-cell review, 2026-09-22. Campaign:
`mark@100.103.10.122:/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T100028Z-e1-18cells`.

Scope: cells 2 native-5/B4, 3 native-11/B4, 4 native-11/B1, each first planned boot for that native route/batch combination. The existing `e1-first-native-cell-redteam.md` remains preserved (SHA `ce669ef61421a0d45fc7ac0ac996ea167bb5255aea4bf8b8ab57f5b29ecd1e2c`). No GPU/inference, old test suites, CPU stress, source/campaign mutation or tmux interaction. Checks run only on completed raw artifacts. The displayed rates are **retained-interval rates**, not complete API elapsed-time rates or campaign-level comparisons.

For each completed cell the independent program checks actual Cmd/env/mounts; snapshot hashes and native reference identity; live/final preflight and terminal gates; all raw request-body/prompt hashes; complete direct API token-ID streams against every pure/nonpure ledger row; terminal clipping restricted to the last output row; exact forward/request/physical identities; the phase boundary; unique retained start-to-start intervals; B1/B4 coverage floors; and the independent token/wall sum against the saved joiner/result.

## Cell 2 — native-5/B4 — PASS (reviewed approximately 10:18 UTC)

No material finding. Cell `cell_02_b1_native-5_B4_a1` is sealed VALID at 10:16:31Z. Actual native-5, B4, FLASH_ATTN, naive_mtp/tree-GDN-off, eager, synchronous, prefix-cache-off, frozen 0.6/16384 allocation and seed all match. No heavy capture/trace flag or tree ENGAGED line. Live and final preflight each 9/9 PASS; post-boot verifier 16/16 PASS. All five loaded source hashes match the already independently native-fallback-verified cell 1.

All **20 requests** (8 preflight, 4 warmup, 8 timed) finish at their frozen token budgets and reconcile **1,408 direct API IDs**. The 30 surplus ledger tokens across 16 requests are exclusively last-row clipping. No output/event loss or earlier-row truncation is accepted.

The primary calculation exactly reproduces **768 supported tokens / 13.395950520411134 s = 57.33075818918667 tokens/s**, over **56 unique actual four-request intervals**. Both frozen cohorts contribute 28 intervals (A:426 tokens/6.594811412s; B:342 tokens/6.801139109s). The physical interval ID set equals the saved joiner's set. All four requests are present at both boundaries of each retained interval.

There are 66 excluded physical steps: 35 preflight/warmup, 6 cohort changes, 23 ramp/drain occupancy, 1 intervening nonpure forward and 1 terminal. The intervening-forward interval spans 12.299369148s and is excluded by the explicit forward-sequence rule, not an elapsed-time threshold. None of the retained intervals exceeds 1.5s. Only 768 of the 1,024 timed API tokens lie on retained full-B4 support; this is the prescribed scope, not dropped token evidence.

Exact raw reconstruction output, including evidence hashes:

```json
{
 "cell": 2,
 "arm": "native-5",
 "batch": 4,
 "status": "VALID",
 "seal_utc": "2026-09-22T10:16:31Z",
 "gates": {
  "native_preflight_live.json": 9,
  "native_preflight_final.json": 9,
  "cell_verify.json": 16
 },
 "events": 440,
 "forwards": 146,
 "physical_steps": 122,
 "output_records": 146,
 "nonpure_outputs": 24,
 "phase_requests": {
  "preflight": 8,
  "warmup": 4,
  "timed": 8
 },
 "phase_api_tokens": {
  "preflight": 256,
  "timed": 1024,
  "warmup": 128
 },
 "api_tokens": 1408,
 "ledger_tokens": 1438,
 "clipped_tokens": 30,
 "clipped_requests": 16,
 "support": {
  "intervals": 56,
  "tokens": 768,
  "wall_s": 13.395950520411134,
  "rate": 57.33075818918667,
  "over_1p5_s": 0
 },
 "excluded": {
  "preflight/warmup": 35,
  "cohort change": 6,
  "occupancy ramp/drain": 23,
  "intervening forward": 1,
  "terminal": 1
 },
 "excluded_walls": {
  "preflight/warmup": 25.930584077723324,
  "cohort change": 1.380979673936963,
  "occupancy ramp/drain": 5.204786996357143,
  "intervening forward": 12.299369147978723
 },
 "per_unit": {
  "A(1-4)": {
   "intervals": 28,
   "tokens": 426,
   "wall_s": 6.594811411574483
  },
  "B(5-8)": {
   "intervals": 28,
   "tokens": 342,
   "wall_s": 6.801139108836651
  }
 },
 "finish_counts": {
  "length": 20
 },
 "hashes": {
  "logs/e1_events.jsonl": "e33a318a7e897b8910ffbd65782fb81751de029ec7f864ec9b6f0ed07f0a8634",
  "e1_manifest.json": "4c362fed3afdc8cb7a84df4485239c8266a47e2ba657cd9a74df1c164a987128",
  "docker_inspect.json": "08dee788d0cf1e3f914e4c9b921ae16890503a1f68cd4307fe519d1f01434a78",
  "cell_result.json": "45952e2e06318589117ce24a8b3d7f2392d172072eaea942b87797c075955594",
  "e1_join.json": "8cf21088d06b055c22a890388c01345ffdb1e886b0fd8e9858284dce255efa5a",
  "native_preflight_live.json": "c2f43230df1eccb084cdd371a0795457ab055dc0718d7f180d921e03b06e4e9b",
  "native_preflight_final.json": "e4a09046123b32dd959d5488cd8f0431089139a74ef1f2331fe8c71a24364e29"
 },
 "loaded_hashes": {
  "gpu_model_runner.py": "b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40",
  "eagle.py": "aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7",
  "tree_attn.py": "a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97",
  "gdn_linear_attn.py": "723974c703d1ad1545399b78526e5d50ae7f7f362e434bad384a0a3327595b28",
  "rejection_sampler.py": "5be4ad43409da82b63bfd0177926467e9a2b59da553ee6dd04d3130a5d77b12f"
 }
}
```

## Independent reproduction program

Run remotely as `python3 - 2`, `python3 - 3`, or `python3 - 4` with the following program on stdin, only after that cell is terminal. The program does not call/import the production joiner. It uses lightweight JSON reads and assertions and writes nothing to the campaign.

```python
from pathlib import Path
import json,hashlib,math,sys,re
from collections import Counter,defaultdict
R=Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T100028Z-e1-18cells")
S=R/"campaign_snapshot"; idx=int(sys.argv[1])
cell=next(c for c in json.loads((S/"e1_cells.json").read_text())["cells"] if c["index"]==idx)
N=cell["max_num_seqs"]; K=cell["num_speculative_tokens"]
C=R/("cell_%02d_b%d_%s_%s_a1"%(idx,cell["block"],cell["arm"],cell["batch"]))
result=json.loads((C/"cell_result.json").read_text())
assert result["terminal_seal"]["sealed"] and result["status"] in ("VALID","INSUFFICIENT_SUPPORT"),result
assert result["terminal_seal"]["campaign_snapshot_sha256sums"]==hashlib.sha256((S/"SHA256SUMS").read_bytes()).hexdigest()
assert result["terminal_seal"]["gates"]["preflight_live"]=="PASS" and result["terminal_seal"]["gates"]["preflight_final"]=="PASS"
for D in [S,C/"script_snapshot"]:
 for line in (D/"SHA256SUMS").read_text().splitlines():
  h,n=line.split(None,1); assert hashlib.sha256((D/n.lstrip("*")).read_bytes()).hexdigest()==h,(D,n)
cfg=json.loads((C/"docker_inspect.json").read_text())[0]; env=dict(x.split("=",1) for x in cfg["Config"]["Env"] if "=" in x); cmd=" ".join(cfg["Config"]["Cmd"])
assert json.loads(env["SPEC_CONFIG"])=={"method":"qwen3_5_mtp","num_speculative_tokens":K}
assert env["FR10_DECODE_MODE_DEFAULT"]=="naive_mtp" and env["FR10_ENABLE_TREE_GDN"]=="0"
for flag in ["--no-async-scheduling","--no-enable-prefix-caching","--enforce-eager","--attention-backend 'FLASH_ATTN'",f"--max-num-seqs '{N}'","--gpu-memory-utilization '0.6'","--max-model-len '16384'","--seed '20260921'"]: assert flag in cmd,flag
assert cfg["Config"]["Image"].endswith("3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776")
assert any(m["Destination"]=="/models" and not m["RW"] for m in cfg["Mounts"])
assert env.get("FR10_METRICS","0")!="1" and env.get("FR13_SFWD_GPU_TIMER")=="1" and env.get("E1_RECORD")=="/logs/e1_events.jsonl"
for key in ["LUMO_MTP_DRAFT_TRACE_FILE","LUMO_TREE_PATH_LCP_LOG","LUMO_TREE_SAMPLER_DEBUG_LOG","FR13_FINAL_LOGIT_CAPTURE","FR10_LAYER_HIDDEN_CAPTURE","FR10_TREE_GDN_CAPTURE_PAYLOAD"]: assert not env.get(key)
assert "ENGAGED" not in (C/"docker_logs.txt").read_text(errors="replace")
gates={}
for name in ["native_preflight_live.json","native_preflight_final.json","cell_verify.json"]:
 j=json.loads((C/name).read_text()); assert j["all_pass"] and all(c["pass"] for c in j["checks"])
 gates[name]=len(j["checks"])
 if "preflight" in name: assert j["reference_scores"]["sha256"]==hashlib.sha256((S/"prefix_scores.json").read_bytes()).hexdigest()
trace=(C/"driver_trace.txt").read_text()
assert trace.index("preflight_verdict=PASS")<trace.index("workload_start_utc")
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
assert dict(Counter(phases.values()))=={"preflight":8,"warmup":(2 if N==1 else 4),"timed":8}
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
qual=json.loads((R/"qualification.json").read_text())["qualified"]
assert cell["arm"]+"/"+cell["batch"] in qual
summary={"cell":idx,"arm":cell["arm"],"batch":N,"status":result["status"],"seal_utc":result["terminal_seal"]["utc"],"gates":gates,
"events":len(ev),"forwards":len(fe),"physical_steps":len(ps),"output_records":len(outs),"nonpure_outputs":sum(o["kind"]=="nonpure" for o in outs.values()),
"phase_requests":dict(Counter(phases.values())),"phase_api_tokens":{ph:sum(len(api[r]) for r in api if phases[r]==ph) for ph in set(phases.values())},
"api_tokens":sum(map(len,api.values())),"ledger_tokens":sum(len(ids) for seqs in byreq.values() for _,ids in seqs),"clipped_tokens":sum(trunc.values()),"clipped_requests":len(trunc),
"support":{"intervals":len(ret),"tokens":T,"wall_s":W,"rate":rate,"over_1p5_s":sum(x[3]>1.5 for x in ret)},
"excluded":dict(excluded),"excluded_walls":dict(excwalls),"per_unit":dict(per),"finish_counts":dict(Counter(m["finish_reason"] for m in man["requests"]))}
hashes={}
for n in ["logs/e1_events.jsonl","e1_manifest.json","docker_inspect.json","cell_result.json","e1_join.json","native_preflight_live.json","native_preflight_final.json"]:
 p=C/n; hashes[n]=hashlib.sha256(p.read_bytes()).hexdigest()
summary["hashes"]=hashes
summary["loaded_hashes"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (C/"loaded_backend").glob("*.py")}
print(json.dumps(summary,indent=1))
print("PASS independent raw native variant",idx)

```


## Cell 3 — native-11/B4 — PASS (reviewed approximately 10:25 UTC)

No material finding. Cell `cell_03_b1_native-11_B4_a1` is sealed VALID at 10:24:14Z. Actual SPEC_CONFIG is native MTP-11 without a tree descriptor, with the frozen B4/sync/eager/cache-off/settings; the identical five loaded source hashes preserve the reviewed guarded native fallback. Live/final preflight each 9/9 and post-boot verifier 16/16 pass. The live PASS precedes the main workload, and the final terminal seal is bound to the frozen campaign.

All 20 request-body and prompt identities match, and all 1,408 direct API IDs reconcile against 1,472 ledger IDs. The 64 excess ledger IDs are allowed final-row budget clipping in 14 requests; no earlier token mismatch or missing row is accepted. All 20 requests finish by length.

Independent reconstruction gives **791 supported tokens / 18.19595991075039 s = 43.47118832311055 tokens/s**, over **52 unique full-B4 intervals**. Cohorts A and B each contribute 26 intervals, with 417 and 374 supported tokens respectively. Actual request sets agree at both boundaries; both coverage units pass. No retained interval exceeds 1.5s.

The 61 excluded physical steps comprise 35 preflight/warmup, 6 cohort changes, 18 ramp/drain occupancy, 1 intervening forward and 1 terminal. The 13.352838296s intervening-forward interval is excluded for the recorded sequence break; it is not a slow-interval trim. The primary token set is 791 of 1,024 timed API IDs, exactly the prescribed full-B4 support. No between-arm conclusion is drawn from this single block.

Exact reconstruction and raw hashes:

```json
{
 "cell": 3,
 "arm": "native-11",
 "batch": 4,
 "status": "VALID",
 "seal_utc": "2026-09-22T10:24:14Z",
 "gates": {
  "native_preflight_live.json": 9,
  "native_preflight_final.json": 9,
  "cell_verify.json": 16
 },
 "events": 407,
 "forwards": 135,
 "physical_steps": 113,
 "output_records": 135,
 "nonpure_outputs": 22,
 "phase_requests": {
  "preflight": 8,
  "warmup": 4,
  "timed": 8
 },
 "phase_api_tokens": {
  "timed": 1024,
  "preflight": 256,
  "warmup": 128
 },
 "api_tokens": 1408,
 "ledger_tokens": 1472,
 "clipped_tokens": 64,
 "clipped_requests": 14,
 "support": {
  "intervals": 52,
  "tokens": 791,
  "wall_s": 18.19595991075039,
  "rate": 43.47118832311055,
  "over_1p5_s": 0
 },
 "excluded": {
  "preflight/warmup": 35,
  "cohort change": 6,
  "occupancy ramp/drain": 18,
  "intervening forward": 1,
  "terminal": 1
 },
 "excluded_walls": {
  "preflight/warmup": 32.94759762939066,
  "cohort change": 2.0016485303640366,
  "occupancy ramp/drain": 6.006635854020715,
  "intervening forward": 13.352838296443224
 },
 "per_unit": {
  "A(1-4)": {
   "intervals": 26,
   "tokens": 417,
   "wall_s": 8.967796535231173
  },
  "B(5-8)": {
   "intervals": 26,
   "tokens": 374,
   "wall_s": 9.228163375519216
  }
 },
 "finish_counts": {
  "length": 20
 },
 "hashes": {
  "logs/e1_events.jsonl": "e1a4322e2fdddf29342c36f07c6d7025acb88d6e4f6e806af07a73776deecd97",
  "e1_manifest.json": "2ebb208f59da85ec6182295ad23a9b85a2d6607ea4667848d29495b0a4b36e3c",
  "docker_inspect.json": "cba28ee3dd221d4e39f3d0466a242791ee8d6470c0c8dc64143877572a8561ab",
  "cell_result.json": "bdc663bd53e7227c4d4c499c87145473119844e1c5033ac901a03b4d4c9399c9",
  "e1_join.json": "20d87dbe726553a02f6f1a7f4f5b5fd77633f443d0b519ab4e0b935b30709289",
  "native_preflight_live.json": "74ae6f123022f7a47bc8c64315cc8b5328117d87182cf9b1a37908a2637031ae",
  "native_preflight_final.json": "db139eece0c801cc8ac824bfb7690e550a0a374b551e5650afd9fe5d2cb4004d"
 },
 "loaded_hashes": {
  "gpu_model_runner.py": "b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40",
  "eagle.py": "aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7",
  "tree_attn.py": "a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97",
  "gdn_linear_attn.py": "723974c703d1ad1545399b78526e5d50ae7f7f362e434bad384a0a3327595b28",
  "rejection_sampler.py": "5be4ad43409da82b63bfd0177926467e9a2b59da553ee6dd04d3130a5d77b12f"
 }
}
```


## Cell 4 — native-11/B1 — PASS (reviewed approximately 10:34 UTC)

No material finding. Cell `cell_04_b1_native-11_B1_a1` is sealed VALID at 10:33:21Z. Actual native MTP-11/B1 configuration, frozen settings, source hashes, direct-ID requests, live/final preflight and terminal eligibility all pass. As with the other native variants, live/final preflight each passes 9 checks and the actual post-boot verifier passes 16.

All 18 request-body and prompt hashes match the frozen workload. All 1,344 direct API IDs reconcile against 1,409 ledger IDs, with exactly 65 final-row clipped tokens across 14 requests. Every request finishes by the prescribed length. No instrumentation loss, wrong IDs or earlier-row clipping was accepted.

The raw reconstruction exactly reproduces **986 supported tokens / 86.189124035649 s = 11.439958475412476 tokens/s**, over **249 unique B1 intervals**. All eight frozen prompts contribute 24–38 intervals; the coverage floor passes. Exclusions are 106 preflight/warmup, 7 intervening-forward and 1 terminal physical step. No retained interval exceeds 1.5s.

Exact raw reconstruction and hashes:

```json
{
 "cell": 4,
 "arm": "native-11",
 "batch": 1,
 "status": "VALID",
 "seal_utc": "2026-09-22T10:33:21Z",
 "gates": {
  "native_preflight_live.json": 9,
  "native_preflight_final.json": 9,
  "cell_verify.json": 16
 },
 "events": 1169,
 "forwards": 389,
 "physical_steps": 363,
 "output_records": 389,
 "nonpure_outputs": 26,
 "phase_requests": {
  "preflight": 8,
  "warmup": 2,
  "timed": 8
 },
 "phase_api_tokens": {
  "warmup": 64,
  "preflight": 256,
  "timed": 1024
 },
 "api_tokens": 1344,
 "ledger_tokens": 1409,
 "clipped_tokens": 65,
 "clipped_requests": 14,
 "support": {
  "intervals": 249,
  "tokens": 986,
  "wall_s": 86.189124035649,
  "rate": 11.439958475412476,
  "over_1p5_s": 0
 },
 "excluded": {
  "preflight/warmup": 106,
  "intervening forward": 7,
  "terminal": 1
 },
 "excluded_walls": {
  "preflight/warmup": 55.91165150702,
  "intervening forward": 16.23633097857237
 },
 "per_unit": {
  "p072": {
   "intervals": 26,
   "tokens": 118,
   "wall_s": 8.957621013745666
  },
  "p017": {
   "intervals": 28,
   "tokens": 122,
   "wall_s": 9.656124436296523
  },
  "p015": {
   "intervals": 35,
   "tokens": 122,
   "wall_s": 12.034819783642888
  },
  "p021": {
   "intervals": 24,
   "tokens": 124,
   "wall_s": 8.247500425204635
  },
  "p085": {
   "intervals": 38,
   "tokens": 125,
   "wall_s": 13.227761984802783
  },
  "p095": {
   "intervals": 37,
   "tokens": 126,
   "wall_s": 12.834117038175464
  },
  "p058": {
   "intervals": 35,
   "tokens": 124,
   "wall_s": 12.164388533681631
  },
  "p083": {
   "intervals": 26,
   "tokens": 125,
   "wall_s": 9.066790820099413
  }
 },
 "finish_counts": {
  "length": 18
 },
 "hashes": {
  "logs/e1_events.jsonl": "3b1304b644c7204f53179f4a3633a18de240e7e2adf0b843df048885ca911cca",
  "e1_manifest.json": "550352cc637b331ec9bcc40517c6e160a6dbb622baa5f598e1bdd31015506497",
  "docker_inspect.json": "56493c92a9ff896042f976c63163cf015e10a63dfe6b2924dc1f1f84eb686e9b",
  "cell_result.json": "79ada4f5fbc542ebb01f1c20aefff56303d7817826336d9ed0fe1e9398e1c1b3",
  "e1_join.json": "098cd7d35fe5e80c207d16d53d8ac5efebe26972d0d60b3d5e666cf81194e287",
  "native_preflight_live.json": "4aae1472bb261d2e2db423fc47ad6cdd2a42ce2c7487774238221acd8e80e063",
  "native_preflight_final.json": "a2f43981aafc9de63f08dc0f34c7a6c79268c5133679dba781363f5bb8191fc8"
 },
 "loaded_hashes": {
  "gpu_model_runner.py": "b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40",
  "eagle.py": "aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7",
  "tree_attn.py": "a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97",
  "gdn_linear_attn.py": "723974c703d1ad1545399b78526e5d50ae7f7f362e434bad384a0a3327595b28",
  "rejection_sampler.py": "5be4ad43409da82b63bfd0177926467e9a2b59da553ee6dd04d3130a5d77b12f"
 }
}
```

## Native subset closure

The four first native arm×batch boots now have bounded independent raw PASS:

| First native cell | Supported tokens | Unique intervals | Supported seconds | Retained tokens/s |
|---|---:|---:|---:|---:|
| 1 — native-5/B1 (separate preserved report) | 995 | 312 | 72.721983117 | 13.682245139 |
| 2 — native-5/B4 | 768 | 56 | 13.395950520 | 57.330758189 |
| 3 — native-11/B4 | 791 | 52 | 18.195959911 | 43.471188323 |
| 4 — native-11/B1 | 986 | 249 | 86.189124036 | 11.439958475 |

This closes the requested first-native live-route and measurement checks for the authorized campaign. It is not a completed 18-cell analysis or a confidence-interval claim. No new boot, parameter change, data replacement, or additional experiment is requested. The ongoing campaign can continue under its frozen gates.
