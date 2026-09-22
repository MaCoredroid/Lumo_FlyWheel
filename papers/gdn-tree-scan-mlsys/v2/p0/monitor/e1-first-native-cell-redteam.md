# First native E1 cell — independent raw review

Reviewed 2026-09-22 through approximately 10:11 UTC. **Bounded PASS: no material finding in cell 1 (native-5, B1).** This validates this cell's frozen configuration, native preflight/terminal eligibility, exact API/event support and reported rate. It does not qualify native-11/B4 or establish a campaign-level comparison.

Remote campaign:
`mark@100.103.10.122:/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T100028Z-e1-18cells`.

Cell subdirectory: `cell_01_b1_native-5_B1_a1`. Below, C denotes this directory and S the campaign's `campaign_snapshot`.

Only lightweight read-only file/JSON/AST analysis was performed. No GPU/inference, container launch, CPU stress work, historical suite rerun, campaign/source mutation, or tmux interaction. The independent audit took ~0.024 s of command wall time, the source AST comparison ~0.72 s.

## Actual freeze and route

Campaign snapshot 19/19 SHA256SUMS entries and cell script snapshot 15/15 matched independently. Frozen runner SHA `910e2dac38a8661361da7f011f8a6a48563b84f6799eb5b1257fc558fa765d72`; driver SHA `25aa6c2195957630f9554961a799426806529d07fa779d2b7042ebeaee31cfc4`.

The final two source-identity omissions from the previous review are **closed in this actual snapshot**: campaign_identity.txt explicitly includes the device multidraft module `648ff0e33e8a7cdd0fbe8ebb3d2befaa27002a3347f94f9bdc0668365ed2e8e9` and fused-conv module `564c7dc7ba61ccd9579f2fb7a4af9a2cdaaa1753dfb98434f2011457d90d9a6e`, alongside patcher/kernel/decode modes/OOM guard and HEAD.

Actual C/docker_inspect.json identifies:
- Pinned image `3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`, FP8 model `/models/qwen3.6-27b-fp8`, read-only /models.
- SPEC_CONFIG native qwen3_5_mtp, 5 speculative tokens, no tree descriptor; `naive_mtp`, tree GDN disabled.
- Actual B1, FLASH_ATTN, eager, explicit prefix-cache off and synchronous scheduling.
- GPU utilization 0.6, model length 16384, seed 20260921.
- Common E1 recorder and SFWD timer; no FR10 metrics, draft/LCP/sampler trace or heavy capture environment.
- All five recorder shim anchors present. Actual post-boot verifier 16/16 PASS. No ENGAGED line in the saved server log.

Loaded source hashes:

| C/loaded_backend file | SHA-256 |
|---|---|
| gpu_model_runner.py | b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40 |
| rejection_sampler.py | 5be4ad43409da82b63bfd0177926467e9a2b59da553ee6dd04d3130a5d77b12f |
| gdn_linear_attn.py | 723974c703d1ad1545399b78526e5d50ae7f7f362e434bad384a0a3327595b28 |
| eagle.py | aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7 |
| tree_attn.py | a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97 |

Actual rejection_sample after excluding the two tree-parent branches is AST-identical to the previously independently stock-verified native body. Actual native conv calls at 13319/13341 and recurrent update at 15336 match the previously stock-verified call ASTs, including arguments. Differences in entire GDN/sampler file hashes from the capture boot do not alter these verified native fallbacks. The baseline remains labeled **patched-runner native**.

## Gate and phase chronology

C/driver_trace.txt records:
- 10:00:28Z: cell start.
- 10:05:57.415991874Z: untimed preflight starts after health (320 s).
- 10:07:20Z: live preflight PASS.
- 10:07:20.720239164Z: fixed warmup/main workload starts.
- 10:08:53.589455001Z: workload ends.
- 10:08:57Z: owned container stopped; driver rc0.
- 10:08:59Z: terminal eligibility seal and native-5/B1 qualification published.

Live and final preflight each report **9/9 PASS**, with the same frozen reference SHA `2940202fa60523823340e3ea89809800bd9b9b1f75226b3306d3f5ce6b35e870`. Campaign and cell copies independently match. Four prespecified high-margin prefixes match the reference first token; all four low-margin prefixes also match, reported without changing their gate status. There are 8 distinct preflight requests, 2 warmup requests, and 8 timed requests. All finish by the prescribed length (32/32/128 tokens respectively).

The durable result is sealed VALID after verify, summary, live and final preflight PASS. Its campaign SHA256SUMS digest `d1ab57c53464ef814c7eed3b87ebd21a670cf8278541b8adc4864e5cf6dc2ecd` independently agrees with S/SHA256SUMS. The campaign qualification entry points to this cell's live/final reports and durable result.

## Independent API / physical-support calculation

The independent code below reconstructs support from raw C/logs/e1_events.jsonl and each C/cohort/req_*/capture_request.json, then compares its result to the joiner and durable result. It does not call the joiner or import its implementation.

- Event counters 1..1367 and all seal totals agree: 455 forwards, 429 physical steps, 455 output records, including 26 nonpure outputs.
- No recorder error, sink failure, missing/duplicate output record, request-row identity mismatch or orphan pure output was found.
- All 18 request prompt hashes and reconstructed request-body hashes match the frozen pool/prefix manifest. Requests directly ask for token IDs, temperature zero and the frozen seed/budgets.
- All **1,344 API token IDs** reconcile with the complete ledger. The ledger has 1,372 IDs; its 28 surplus tokens are exclusively allowed final-row budget clipping across 12 requests. No earlier row or equal-count wrong-ID acceptance is used.
- Timing uses only the eight timed requests. Their API output totals 1,024 IDs; **995 IDs** are supported by the retained physical intervals. The remaining 29 API tokens are outside this prespecified timing support.
- **312 unique intervals**, **995 supported tokens**, **72.72198311705142 s**, rate **13.682245138976391 tokens/s**. This exactly matches the saved joiner/result and physical-step ID set.
- All intervals are actual B1 and consecutive pure same-request steps. Excluded steps: 109 preflight/warmup, 7 with an intervening forward, 1 terminal step. None of the retained intervals exceeds 1.5 s; no elapsed-time trimming changes the rate.

| Timed prefix | Retained intervals | Supported tokens | Supported wall (s) |
|---|---:|---:|---:|
| p072 | 29 | 125 | 6.735802257 |
| p017 | 36 | 123 | 8.331856121 |
| p015 | 39 | 121 | 9.032349216 |
| p021 | 39 | 126 | 9.022395360 |
| p085 | 46 | 126 | 10.774224680 |
| p095 | 46 | 123 | 10.767720748 |
| p058 | 43 | 126 | 10.074119512 |
| p083 | 34 | 125 | 7.983515223 |

Every frozen prompt contributes at least one interval, so the coverage floor passes without replacement or extension. This is the frozen **retained-interval rate**, not end-to-end API throughput. It is one boot/block cell; no campaign comparison or confidence interval is inferred.

## Primary raw identities

| Path relative to C | SHA-256 |
|---|---|
| logs/e1_events.jsonl | 35e0507d47763c927cc6e3c724b69c9a6cfcfb3ad2f2cbeadd7bf675995b4022 |
| e1_manifest.json | bcaaadddc2ad1aa7131be78d30c31ea7ec4cab8b48aa65d292a1933838eada5d |
| docker_inspect.json | 5d0f809c42c5311b421c44d4400def8413d8fcb856ed51ab87786be3eff4a6e1 |
| cell_result.json | 954a4a968cd2c12515e556438c4ea6de26117677ecb651f65a6689544e249854 |
| e1_join.json | bca904fb8118420664d9173e1c6dc38d4e5c783827f47d824d3c668ee8942174 |
| native_preflight_live.json | 64a852fecf2a15b0cbdf9afd923dcb0f6bba9cf64670ad7a8fd50229c4869996 |
| native_preflight_final.json | 270f5c5c67feea3a76bbfc8e4d12aff9176aef441e13cf9dd4aeeed1006849a6 |

## Executed independent raw reproducer

Ran with remote `python3` over SSH, no outputs written to the campaign. Output ends `PASS independent raw identity/support/rate`; the script prints per-capture hashes as well as aggregate support.

```python
from pathlib import Path
import json,hashlib,math
from collections import Counter,defaultdict
R=Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T100028Z-e1-18cells")
C=R/"cell_01_b1_native-5_B1_a1"
result=json.loads((C/"cell_result.json").read_text())
print("terminal_status",result.get("status"),"seal",result.get("terminal_seal"))
assert result["terminal_seal"]["sealed"] and result["status"] in ("VALID","INSUFFICIENT_SUPPORT")
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
 assert seq in fe
 assert o["kind"] in ["pure","nonpure"]
 if o["kind"]=="pure": assert ps[o["physical_step_id"]]["seq"]==seq
 for row in o["rows"]:
  rid=row["request_id"]; assert row["step_idx"]==len(byreq[rid])
  assert row["n_emitted"]==len(row["emitted_ids"])
  byreq[rid].append((seq,row["emitted_ids"]))
for pid,p in ps.items():
 o=outs[p["seq"]]; rids=[x["request_id"] for x in o["rows"]]
 assert o["kind"]=="pure" and len(rids)==len(set(rids))==p["num_reqs"]==1
 assert set(rids)==set(p["request_ids"])
man=json.loads((C/"e1_manifest.json").read_text())
frozen=json.loads((R/"campaign_snapshot/frozen_prefixes.json").read_text())
expected=[x["id"] for x in frozen["pilot"]]
phases={}; rid2meta={}; api={}; bounds={}; trunc={}; hashes={}
for m in man["requests"]:
 rid=[r for r in byreq if r==m["response_id"] or r.startswith(m["response_id"]+"-")]
 assert len(rid)==1; rid=rid[0]
 cap=C/"cohort"/("req_"+m["slot"]+"_"+m["prefix_id"])/"capture_request.json"
 c=json.loads(cap.read_text()); toks=c["response_logprobs_tokens"]
 assert all(isinstance(x,str) and x.startswith("token_id:") for x in toks)
 ids=[int(x[9:]) for x in toks]
 assert len(ids)==m["n_api_tokens"] and c["response_id"]==m["response_id"]
 assert c["request"]["return_tokens_as_token_ids"] is True
 assert c["request"]["temperature"]==0 and c["request"]["seed"]==20260921
 assert c["max_tokens"]==(128 if m["phase"]=="timed" else 32)
 assert m["finish_reason"] in ["length","stop"]
 if m["finish_reason"]=="length": assert len(ids)==m["max_tokens"]
 assert m["prefix_id"] in expected
 stream=[t for _,rowids in byreq[rid] for t in rowids]
 counts={seq:len(rowids) for seq,rowids in byreq[rid]}
 if stream!=ids:
  lastseq,lastids=byreq[rid][-1]; headlen=len(stream)-len(lastids)
  assert len(ids)<len(stream) and stream[:len(ids)]==ids and len(ids)>=headlen
  counts[lastseq]=len(ids)-headlen; trunc[rid]=len(stream)-len(ids)
 api[rid]=ids; bounds[rid]=counts; phases[rid]=m["phase"]; rid2meta[rid]=m
 hashes[str(cap.relative_to(C))]=hashlib.sha256(cap.read_bytes()).hexdigest()
assert set(phases)==set(byreq)
assert [m["prefix_id"] for m in man["requests"] if m["phase"]=="timed"]==expected
ret=[]; excluded=Counter(); per=defaultdict(lambda:{"intervals":0,"tokens":0,"wall_s":0.0})
for pid,p in sorted(ps.items()):
 nxt=ps.get(pid+1); rid=p["request_ids"][0]
 reason=None
 if nxt is None: reason="terminal"
 elif phases[rid]!="timed": reason="preflight/warmup"
 elif nxt["seq"]!=p["seq"]+1: reason="intervening forward"
 elif any(p["seq"]<=b<=nxt["seq"] for b in breaks): reason="chain break"
 elif nxt["request_ids"]!=p["request_ids"]: reason="cohort change"
 elif any(x.get("discarded") for x in outs[p["seq"]]["rows"]): reason="discarded"
 if reason: excluded[reason]+=1; continue
 wall=nxt["t_start"]-p["t_start"]; assert wall>0
 tok=bounds[rid][p["seq"]]; ret.append((pid,p["seq"],tok,wall))
 unit=per[rid2meta[rid]["prefix_id"]]; unit["intervals"]+=1; unit["tokens"]+=tok; unit["wall_s"]+=wall
T=sum(x[2] for x in ret); W=sum(x[3] for x in ret); rate=T/W if W else None
print("raw_counts",{"events":len(ev),"forwards":len(fe),"physical_steps":len(ps),"output_records":len(outs),"nonpure_outputs":sum(o["kind"]=="nonpure" for o in outs.values()),"api_requests":len(api),"phase_requests":dict(Counter(phases.values())),"api_tokens":sum(map(len,api.values())),"truncated_tail_tokens":trunc})
print("independent_support",{"n":len(ret),"tokens":T,"wall_s":W,"rate":rate,"over_1p5_s":sum(x[3]>1.5 for x in ret),"excluded":dict(excluded),"per_prompt":dict(per),"interval_ids":[x[0] for x in ret]})
j=json.loads((C/"e1_join.json").read_text())
assert j["n_usable"]==len(ret)
assert j["sum_emitted_tokens_api_bound_pure_support"]==T
assert math.isclose(j["sum_wall_s_unique_physical_steps"],W,rel_tol=1e-12)
assert (j["tokens_per_wall_second"]==rate or math.isclose(j["tokens_per_wall_second"],rate,rel_tol=1e-12))
assert set(x["physical_step_id"] for x in j["steps"])==set(x[0] for x in ret)
if result["status"]=="VALID":
 assert set(per)==set(expected) and all(x["intervals"]>=1 for x in per.values())
 assert result["primary"]["sum_emitted_tokens_api_bound_pure_support"]==T
 assert math.isclose(result["primary"]["tokens_per_wall_second"],rate,rel_tol=1e-12)
for n in ["logs/e1_events.jsonl","e1_manifest.json","docker_inspect.json","cell_result.json","e1_join.json","native_preflight_live.json","native_preflight_final.json"]:
 p=C/n
 if p.exists(): hashes[n]=hashlib.sha256(p.read_bytes()).hexdigest()
print("hashes",hashes)
print("PASS independent raw identity/support/rate")

```

The additional request/prompt SHA validation:

```python
from pathlib import Path
import json,hashlib
R=Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T100028Z-e1-18cells")
C=R/"cell_01_b1_native-5_B1_a1"; S=R/"campaign_snapshot"
pool={x["id"]:x for x in json.loads((S/"prefix_pool.json").read_text())["prefixes"]}
frozen={x["id"]:x["prefix_sha256"] for x in json.loads((S/"frozen_prefixes.json").read_text())["pilot"]}
n=0
for p in (C/"cohort").glob("*/capture_request.json"):
 c=json.loads(p.read_text()); text=pool[c["prefix_id"]]["text"]
 h=hashlib.sha256(text.encode()).hexdigest()
 assert h==frozen[c["prefix_id"]]==c["prompt_sha256"]==c["prefix_sha256"]
 r=c["request"]
 body={"model":r["model"],"prompt":text,"max_tokens":r["max_tokens"],"temperature":r["temperature"],"seed":r["seed"],"logprobs":r["logprobs"],"echo":r["echo"],"return_tokens_as_token_ids":r["return_tokens_as_token_ids"]}
 assert hashlib.sha256(json.dumps(body).encode()).hexdigest()==c["request_body_sha256"]; n+=1
print("request_prompt_and_body_hashes_PASS",n)

```

