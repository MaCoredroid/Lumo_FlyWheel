# Policy-B B1 independent raw-evidence review

2026-09-22. Read-only remote source inspection and CPU reductions; no inference, GPU work, container launch, source edits or new experiment cells.

**Disposition: the bounded synchronous B1 policy-B diagnostic passes.** No additional B1 boot is needed to reconstruct accepted-plus-bonus greedy outputs: existing logits, drafts, states and ledger suffice. B4 and the planned E2→E1 qualification remain separate. This report does not claim sequential full-model equivalence or timing qualification.

Remote base: `/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/`. Each alias below ends in `e7b_001_p072_arm1_none-B_fs_ieee-all/`.

- **B:** `out-20260922T082815Z-e2-b1-policyB-sync-p072/`.
- **old:** `out-20260922T071945Z-e7b-r16-p072-commit-batch/`.
- **A:** `out-20260922T081202Z-e2-j1prime-p072-kvpolicyA-b1/`.

## What the evidence establishes

| Claim | Independent check | Disposition / limit |
|---|---|---|
| Selected flat TREE_ATTN policy actually loaded | Actual Docker environment is KV remap / slot reorder / syncfree = **1 / 0 / 1**; command has TREE_ATTN, max-num-seqs 1, no-async-scheduling, enforce-eager. Live loaded-source hashes below. Log line 333 confirms remap engaged. | Correct selected policy; not inferred from requested settings alone. |
| First-forward permutation defect repaired | Same prompt hash, identical 3×10 position tensor and input hidden tensor. B vs old: all 64 hidden/residual pairs, final norm and logits bitwise equal. First hidden and logits files themselves have identical SHA-256. B vs A: layers 0–2 equal; first divergence at layer 3; logits max absolute delta 6.125, differing argmax only row 5. | Supports repair of the demonstrated first-forward mismatch. Captured input_ids fields are None, so direct input-ID tensor equality is **not** asserted. Later KV-on and KV-off continuations need not coincide. |
| Authoritative recurrent-state publication consumed next | 12 layer-62 operand/state pairs; actual h0 row and published row 21, h0 column 0; RUNROW_INIT true and rows-consistent-before true for all. Mode none, no substitution, replay-before = published-after bitwise. All 11 available published-after→next-h0 transitions bitwise equal. | Exact selected-layer publication/read linkage; does not cover uncaptured recurrent layers, conv or full-model sequential reference by itself. |
| Greedy output on accepted path, including bonus | Independently walk full-vocabulary logits argmax along actual parent/child topology and draft tokens at each of 12 steps. Every derived path and accepted length equals the captured published metadata, and every derived token list equals the E1 output row. All 12 position tensors track the prior emitted length with depth offsets [0,1,2,2,3,3,4,4,5,5]. | Covers all 31 post-prefill API tokens. Initial prefill token 6813 is bound by API/ledger and is highest in returned top-logprobs, but full-vocabulary prefill logits were not captured. |
| API and terminal accounting | 34 structural outputs = 1 prefill + 33 tree outputs. API emits their exact first 32, with finish=length. Only the last two tokens of the last row are truncated. | All 32 API IDs bound to the ledger; only 31 independently reconstructed from captured full logits. Terminal accepted metadata describes the structural walk, not an entirely API-emitted path. |
| E1 recorder lifecycle and reduction | 41 contiguous events; 13 forwards, 12 physical tree steps, 13 output rows, one prefill chain break, zero errors/sink failures. Matching final atexit seal. Independently reran snapshotted joiner: rc=0. Recomputed 11 intervals and 29 supported API tokens directly from raw events. | Instrumentation pilot passes; not a warmed timing cell or a performance result. |

The loaded stock TREE_ATTN contract remains the reason for flat slots: `tree_attn.py:1087` uses slot mapping for cache writes, while `1165–1167` passes node-order bias and block tables to unified attention. The unified reader addresses flat physical offsets (`triton_unified_attention.py:815–835`; segmented path `1224–1244`). The selected command does not invoke the fork-FA2 companion patcher. Exact source-contract reasoning and distinct-row controls are preserved in `e2-closure-fixture-redteam.md`; this run adds the actual loaded-source and first-forward witness.

Request window: 08:34:04.362469770Z–08:34:21.950083006Z. Request ID `cmpl-95488b508431b321-0-9d01285a`; prefix p072 SHA-256 `4022f519838d33bcad42a55b3c34dbfeddd575bd56488a4f3509a1382f417f03`; 256 prompt tokens, temperature 0, seed 20260921, max_tokens 32. Capture provenance has 51 passing checks; operand gate has 8. This review additionally reconstructs the greedy walk instead of treating either gate as sufficient evidence.

## Per-step path / output ledger

Step indices are zero-based captures. All paths below match captured state metadata and all token lists match E1 output rows.

| Step | Accepted node path | Full structural output IDs (accepted + bonus) | Published state → next h0 |
|---|---|---|---|
| 0 | 1 | 12333, 36349 | Bitwise equal |
| 1 | 1,2,5 | 853, 6971, 63, 198 | Bitwise equal |
| 2 | 1 | 220, 11069 | Bitwise equal |
| 3 | root only | 220 | Bitwise equal |
| 4 | 1,2,5 | 16, 15, 14, 16 | Bitwise equal |
| 5 | 1,2,4 | 15, 5642, 13, 271 | Bitwise equal |
| 6 | 1 | 550, 9019 | Bitwise equal |
| 7 | 1,2 | 38540, 271, 12 | Bitwise equal |
| 8 | root only | 25946 | Bitwise equal |
| 9 | root only | 26679 | Bitwise equal |
| 10 | 1,2,4,7 | 357, 13239, 430, 279, 491 | Bitwise equal |
| 11 | 1,2,5 | 32439, 23, 469, 2629 | No successor (terminal) |

Step 11 API receives only `32439, 23`; `469, 2629` are structural tail output only. It has no captured successor. Positions at each root are [256,258,262,264,265,269,273,275,278,279,280,285]. The 11 completed intervals cover tree outputs from steps 0–10 (29 tokens). The initial prefill token and final two API-bound tokens lie outside that complete-interval support.

E1 manifest assigns this single request to timed and has no warmup requests. This is an instrumentation check, not adequate timing practice for E1. The primary interval sum is 13.273978726007044 seconds and includes the first 6.904658170416951-second interval; it is not silently removed by the 1.5-second diagnostic threshold. The terminal interval is excluded because it lacks a successor. Snapshot `e1_join.py:43–63` validates the seal; `85–93` validates phase membership; `99` excludes missing-successor intervals; `128` allows only terminal-row API truncation. These checks and the independent sum agree.

The API’s longest common prefix is 29 tokens with old KV-off (both length 32), and 17 with A (A ended at 18, B at 32). This is **not** evidence of full continuation equivalence. The targeted first-forward repair is the defensible cross-run result.

## Minimum next step and stop conditions

Proceed with the already planned B4 selected-route capture and its row-aware gates, then the separately specified sequential E2 route. No additional ablation or repeated B1 launch follows from this review. Stop promotion if actual selected flags/source differ, any published state fails its next-h0 link, a published path/output differs from the full-logit greedy walk, API tokens cannot be bound with terminal-only truncation, or the ledger fails its complete-close checks. These are the existing concrete gates, not added experiment axes.

## Reproduction and immutable hashes

At the remote repository root, set B to its exact run directory above. A read-only joiner rerun is:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -B "$B/script_snapshot/e1_join.py" "$B/logs/e1_events.jsonl" --api-tokens "$B/e1_api_tokens.json" --manifest "$B/e1_manifest.json" --expect-reqs 1
```

The independent greedy reduction uses, for each i=0..11, `logs/final_logits.call{i}.pt["logits"].argmax(-1)`, draft trace record i `["draft"][0]`, operands `["parents"]`, and the corresponding state `["path"]` / `["accepted_len"]`. Starting node 0, append its argmax token; follow the first child j whose parent is the current node and whose draft[j-1] equals that token; stop when no child matches. Compare the entire token list to pure E1 `output_rows[i].rows[0].emitted_ids`. Full source of the executed independent reduction follows the hash ledger; it only reads files and prints JSON.

Loaded source hashes were read directly inside the running B container `e7a-capture-082815`, without importing serving modules. Image ID `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`; repository image digest `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`.

```text
a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97  loaded/vllm/v1/attention/backends/tree_attn.py
5dd83ac4fbd0a08054a5b7188f4937cdf33fce854b96d0a38e616a9e4c7789c0  loaded/vllm/v1/attention/ops/triton_unified_attention.py
b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40  loaded/vllm/v1/worker/gpu_model_runner.py
63bc4503d560f54cf9923fc03e69267368f3951a92b1b5734a8ef871c40a0b99  loaded/vllm/model_executor/layers/mamba/gdn_linear_attn.py
2680d038b9e626f659b17e6910dd89fde3ea9581eb21ea468ccd80f572ed1a56  loaded/vllm/v1/sample/rejection_sampler.py
9c3a6d94b2e1c6ccb07e4aedbbc29d812b9cbf4d102789b5f8a262b04b99b5ca  B/script_snapshot/e7b_runtime.py
cbee2ae70947f1adb4073bd151a7148fcb05cbc02d1c5dff37b381ffadf1e766  B/script_snapshot/e1_join.py
1cf3f53552e008c44f7b7e5e6187d8052bcfd4c75bd6f81e2e07d1f70fbf65a0  B/script_snapshot/e1_recorder.py
06344718f035949ea7f1cea49e4677e9acc2a71a460bbbc7b94e01a1bbbbc1df  B/script_snapshot/e1_event_recorder_shim.py
f23d2adc8995bacfb6472ef18a4d93528adebd16cdac61a5066506b59b847e7c  B/script_snapshot/serve_drivers.v11.sh
6e71a24c5b7ea82932b219b7502a5c19d3b367e37105451a306b7afb3c95ed04  B/script_snapshot/inspect_capture.py
f4eaa94164db87824575726a7a4004ab0ad9dd42c4db29565727c0f62f64bf99  B/script_snapshot/e7b_operands_gate.py
bc06fab1d853aa07b2c6a0716f8df22a73b82d48b4493f66bea045dd741455f1  A/capture_request.json
6c4c74dc5914c4d744a272a09fab329813806725f449381d95a580ebf4eb5cf2  A/logs/final_logits.call0.pt
00e7b51e5464091ef30679fb8f709fff1f591fe671343110cea9c2e164d93adf  A/logs/layer_hidden.call0.pt
992a90de17b7287f86b6b84e8e5013629a61b8e89bc2b4ef0b0413f430576f6c  B/capture_provenance.json
12dae64baa74df003f7f2b238b7b6d2ff220a19100c741ae9613204f47289317  B/capture_request.json
2d610f47fa56870bbdf83e887da00e2ebb3ebbb2cfe427dc18307a00fb23545b  B/docker_inspect.json
26de50139d88c8c2c0560d9f874a8fe71271d9a3a9164b72ff62f3bb0a7b7a66  B/docker_logs.txt
e92e9c2013930b0112185c9f240df10fb7ba6f1699fd647f12b127210d763077  B/e1_api_tokens.json
e09423e5887c336a29fdf4d4884157367cb3b0a06d7c50568fdb094c45b7eac7  B/e1_join.json
d94d2d54d2a5a6e05cad41970fceb54effbb12972eb2d6acd8e98b28411d127f  B/e1_manifest.json
b962e6903f9d8602ac6abb9d84e68cdb18a85d8a5857280745bbccbd3f147f4c  B/logs/e1_events.jsonl
f6e3a93bb59e00fcaeebe5af5049028dc564454441b830c21777c24077a0f126  B/logs/final_logits.call0.pt
b049bc21a7ae2210c6f95c89383c2c86ee76eb725cc1406e025f812b431385d4  B/logs/fr10_mtp_draft_trace.jsonl
147c240d332a7eefc0af7ec6e9cc5448654c834b7f9527966105986d3ffb0bc0  B/logs/fr10_tree_depth_positions.jsonl
5a4ba5e6472cd473b220a24e2f3549f95650877f1ea96f325c6cf6cd386e3f1c  B/logs/layer_hidden.call0.pt
76e2b85ef5934141b3f349cc6bcc1a2269d03814411357a122eedd21704179d9  B/logs/per_req_spec_trace.jsonl
199a647e24f8e26476b5ea3c19ed698e962c02b194f287bcfaa9ddabaa6eac04  B/logs/tree_sampler_debug.jsonl.commit
e264a989f5c051c8286c5049bd8dd8286c8f66c6e55b8129c0e42b916f9f87bb  B/operands_gate.json
7e1276f4e14ba1b5e0c3d71d21411af5c13ac11dd1fde95b339c7ef3005c64e5  old/capture_request.json
f6e3a93bb59e00fcaeebe5af5049028dc564454441b830c21777c24077a0f126  old/logs/final_logits.call0.pt
5a4ba5e6472cd473b220a24e2f3549f95650877f1ea96f325c6cf6cd386e3f1c  old/logs/layer_hidden.call0.pt
350ade23ad3c492f76052430729e96e34898b4b4b8cbedfc10850d67a08a2cc9  B/logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step0.pt
d5329883af2896e5e3bdbacfd7155abe67e4fb4ce558c1aa90a03f55ef24addd  B/logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step1.pt
c651f71b9b864874bd1a287939650b9be325f10d98317cba1387ce36e869b1eb  B/logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step10.pt
6b95359e3296b116a36fd9837884e9ebd6d6a19f05b6f41ab40653d749c40032  B/logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step11.pt
3398be624b755ff7a541a4238ee556b1149252c9490571dfc3cd71958b384425  B/logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step2.pt
46dfb20a132b83bd9b375cc4d06ad94e96f13e0ee986139fec1f9f52442e7756  B/logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step3.pt
3a1771c2965fdb272e87d50bf315cdbe3efc6625591c796e2f1813fc6e94692b  B/logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step4.pt
eee53873abc6de97d0cbaf7f1853c7b3a66b3ad8b7e751b013d72336bd395f49  B/logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step5.pt
9cb859b3c822a81f08216afa0b0198e87c93dcc972233af4539f879ab3dd86e8  B/logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step6.pt
d45057d4fd48b269a9b718ff673e06a565eee99a418974130dce0907cc51dc74  B/logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step7.pt
a0844217a9d07f144c968bfdbc3bd1253605e5c9fa70e1b81a281b8bcfe7b649  B/logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step8.pt
f5f52888ec87da6f3caac48ecc813c2f69c485fc3cc9a212d44dbafd73aa243d  B/logs/e7b_operands/language_model_model_layers_62_linear_attn.row0.step9.pt
487bd214a6fae7a99681f60577033cd6775591f92415033e71f5c146a94b0006  B/logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step0.pt
96543ce2aa63e52cc03d322889be4bdc147ef7f777a7c895e5cd2ed980c8abc5  B/logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step1.pt
8f9f92068ff8427f8dc03ccdce38b4a1669b18068096bbdf47db8bc30962ec0d  B/logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step10.pt
d89592c9cdd9698aa4a59b2badef1800039c6c0085193b768f4c3e717fdef174  B/logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step11.pt
0ff6540aee3de4a3b809620a91d10676495c7a2dfab343c3449b2e19a8829a6c  B/logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step2.pt
c49a06f5334ac97f6ec666550de1cb0041a64a366705b8adc7beb73c2b9a9dc8  B/logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step3.pt
3dcf565ed203fb318b03133ba2fc798f75e56ef1ae6bb242623b8d67b0a2f5f2  B/logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step4.pt
aa9616ca23122edfc40797b7e0cc9b863256de69d6d9b424c1100425f99717f3  B/logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step5.pt
0246a85364ae5b218f479026b7d9386580b617769df7208e4c035cbe515d7774  B/logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step6.pt
f0f59cfb58059c582364730b3516612e059bbf04c81254b897b1d21433d779cc  B/logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step7.pt
3ebf783d3e7f3ead70306068b333f4554cf15b2aa4725fac937814419cb5514f  B/logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step8.pt
94418193e62aa59cb0e22a59381f334f39d2f3ff68e23ed4191a780579edbae9  B/logs/e7b_state/language_model_model_layers_62_linear_attn.row0.step9.pt
f6e3a93bb59e00fcaeebe5af5049028dc564454441b830c21777c24077a0f126  B/logs/final_logits.call0.pt
2a1507da49460dbdf7ad5bd8b3f835d0652a63a543896a6378fca898cd104429  B/logs/final_logits.call1.pt
4820b958292bb81d22419ef2fb1a5628f747101daab90231ab415b5ad22728e5  B/logs/final_logits.call10.pt
b7747f3987aec379598c6c1f2c25b3c43ead7dc5e7bc8d51121cb67d1f512e64  B/logs/final_logits.call11.pt
6c7c7c5ccf06df39435959e68bb600d41d196ded62b09adaee150f03105ff4a4  B/logs/final_logits.call2.pt
3852868c03eafb5d2122db7dba48a1e8bdf808100d39777b325fdf62ef6d4007  B/logs/final_logits.call3.pt
82af848e50fc093ea995126ed23c6b31bb50fb652a895d98d7fb7163c6a32f79  B/logs/final_logits.call4.pt
f0dad9d9f2667f67ebb8031f1d36dad904f05435ff52ed5f9779d09388e5b493  B/logs/final_logits.call5.pt
cb85d96fc7d61344c6bae86e4c7d6e97592d8518dd249fb6f32f11087f5d221f  B/logs/final_logits.call6.pt
1ddf2355a14acfdd8bee12703016243369b4116ffda64033b681e154db38e620  B/logs/final_logits.call7.pt
152c6466584bf351d1d098afc3dbc55c127802009d6af49ce3c5378cf908fe24  B/logs/final_logits.call8.pt
58362de1efa3fba4e842d7a407c9d8c0d3553dcba8e04458ef23b39e651b5baf  B/logs/final_logits.call9.pt
```

```python
from pathlib import Path
import torch,json,hashlib,collections
root=Path("papers/gdn-tree-scan-mlsys/v2/experiments")
bn="e7b_001_p072_arm1_none-B_fs_ieee-all"
runs={"B":root/"out-20260922T082815Z-e2-b1-policyB-sync-p072"/bn,"old":root/"out-20260922T071945Z-e7b-r16-p072-commit-batch"/bn,"A":root/"out-20260922T081202Z-e2-j1prime-p072-kvpolicyA-b1"/bn}
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def pt(p):return torch.load(p,map_location="cpu",weights_only=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
b=runs["B"];request=read(b/"capture_request.json");prov=read(b/"capture_provenance.json")
reqs={k:read(r/"capture_request.json") for k,r in runs.items()}
hs={k:pt(r/"logs/layer_hidden.call0.pt") for k,r in runs.items()}
ls={k:pt(r/"logs/final_logits.call0.pt") for k,r in runs.items()}
out={"first_forward":{}}
for label in ["old","A"]:
 h=hs[label];cur=hs["B"];diffl=[]
 for i,(x,y) in enumerate(zip(cur["layers"],h["layers"])):
  if not all(bits(x[t],y[t]) for t in ["hidden","residual"]):diffl.append(i)
 out["first_forward"][label]=dict(prompt_same=request["prompt_sha256"]==reqs[label]["prompt_sha256"],positions_bits_equal=bits(cur["positions"],h["positions"]),input_ids=[cur["input_ids"],h["input_ids"]],input_hidden_bits_equal=bits(cur["input_hidden"],h["input_hidden"]),differing_hidden_residual_layers=diffl,final_norm_bits_equal=bits(cur["final_norm_hidden"],h["final_norm_hidden"]),logits_bits_equal=bits(ls["B"]["logits"],ls[label]["logits"]),logit_absmax=float((ls["B"]["logits"]-ls[label]["logits"]).abs().max()),argmax_different_rows=(ls["B"]["logits"].argmax(-1)!=ls[label]["logits"].argmax(-1)).nonzero().view(-1).tolist())
# Operands and state identity + 11 complete successor links
stem="language_model_model_layers_62_linear_attn.row0.step"
ops=[pt(b/f"logs/e7b_operands/{stem}{i}.pt") for i in range(12)]
states=[pt(b/f"logs/e7b_state/{stem}{i}.pt") for i in range(12)]
parents=ops[0]["parents"];events=lines(b/"logs/e1_events.jsonl");outputs=[e for e in events if e["event"]=="output_rows"];pure=[e for e in outputs if e["kind"]=="pure"]
drafts=lines(b/"logs/fr10_mtp_draft_trace.jsonl")
firstidx=prov["first_verified_draft_record"]["idx"];start=next(i for i,d in enumerate(drafts) if d["idx"]==firstidx and d["ts"]==prov["first_verified_draft_record"]["ts"])
ledger=[]
for i,(o,s) in enumerate(zip(ops,states)):
 raw=pt(b/f"logs/final_logits.call{i}.pt")["logits"];am=raw.argmax(-1).tolist();draft=drafts[start+i]["draft"][0]
 node=0;path=[];walk=[]
 for depth in range(11):
  tok=am[node];walk.append(tok)
  matches=[j for j in range(1,len(parents)) if parents[j]==node and draft[j-1]==tok]
  if not matches:break
  node=min(matches);path.append(node)
 else:raise AssertionError("walk did not terminate")
 row=pure[i]["rows"][0]
 ledger.append(dict(step=i,path=s["path"],accepted_len=s["accepted_len"],h0_col=o["h0_col"],h0_row=o["h0_row"],authoritative_row=s["authoritative_row"],identity_ok=o["layer"]==s["layer"] and o["row"]==s["row"]==0 and o["step"]==s["step"]==i and o["h0_row"]==s["authoritative_row"],no_substitution=s["mode"]=="none" and not s["applied"] and bits(s["replay_before"],s["published_after"]),next_h0_bits_equal=bits(s["published_after"],ops[i+1]["h0"]) if i+1<len(ops) else None,greedy_path=path,greedy_ids=walk,published_path_equals_walk=s["path"]==path and s["accepted_len"]==len(path),ledger_ids=row["emitted_ids"],ledger_equals_walk=row["emitted_ids"]==walk))
out["states_and_greedy"]=ledger
api=[int(t.split(":")[1]) for t in request["response_logprobs_tokens"]];prefix=outputs[0]["rows"][0]["emitted_ids"];walk_ids=prefix+[t for x in ledger for t in x["greedy_ids"]]
out["api_binding"]=dict(api_tokens=len(api),finish=request["finish_reason"],request=request["request"],prefill_ids=prefix,prefill_equals_api_prefix=api[:len(prefix)]==prefix,prefill_is_top_returned_logprob=max(request["response_logprobs_first"],key=request["response_logprobs_first"].get)==f"token_id:{api[0]}",structural_tokens=len(walk_ids),all_api_match_greedy_walk_prefix=walk_ids[:len(api)]==api,terminal_structural_excess=len(walk_ids)-len(api))
out["ledger"]=dict(counts=dict(collections.Counter(x["event"] for x in events)),counters_contiguous=[x["n"] for x in events]==list(range(1,len(events)+1)),seal=events[-1],api_map_exact=read(b/"e1_api_tokens.json")=={pure[0]["rows"][0]["request_id"]:api},manifest=read(b/"e1_manifest.json"),join={k:v for k,v in read(b/"e1_join.json").items() if k!="steps"})
out["api_vs_controls"]={}
for label in ["old","A"]:
 other=[int(t.split(":")[1]) for t in reqs[label]["response_logprobs_tokens"]];common=next((i for i,(x,y) in enumerate(zip(api,other)) if x!=y),min(len(api),len(other)))
 out["api_vs_controls"][label]=dict(common_prefix=common,other_n=len(other),this_n=len(api))
out["hashes"]={k:{rel:sha(r/rel) for rel in ["capture_request.json","logs/layer_hidden.call0.pt","logs/final_logits.call0.pt"]} for k,r in runs.items()}
for rel in ["capture_provenance.json","operands_gate.json","e1_api_tokens.json","e1_manifest.json","e1_join.json","logs/e1_events.jsonl","logs/tree_sampler_debug.jsonl.commit","logs/fr10_mtp_draft_trace.jsonl","logs/fr10_tree_depth_positions.jsonl","logs/per_req_spec_trace.jsonl","docker_inspect.json","docker_logs.txt"]:
 out["hashes"]["B"][rel]=sha(b/rel)
records=[]
for p in sorted((b/"logs").rglob("*.pt")):
 if "e7b_operands" in str(p) or "e7b_state" in str(p) or p.name.startswith("final_logits.call"):records.append([str(p.relative_to(b)),sha(p)])
out["state_operands_logits_hash_ledger"]=records
print(json.dumps(out,sort_keys=True))
```

