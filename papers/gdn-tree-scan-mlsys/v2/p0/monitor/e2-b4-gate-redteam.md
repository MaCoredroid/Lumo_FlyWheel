# E2 actual-B4 gate red-team — 2026-09-22

Read-only/CPU review of one stable source set in `mark@100.103.10.122:/home/mark/lumo-paper-v2-20260921`. No GPU, inference, tmux, or canonical source edits. **The capture boot need not wait. Offline qualification and E1 timing must wait for the concrete gate fixes below.**

## Reviewed identities

Paths below are relative to `papers/gdn-tree-scan-mlsys/v2/experiments/`.

| File | SHA-256 |
|---|---|
| `e2/e7b_b4_capture_gate.py` | `056c2cb8691999c0e11a5b2947cef90fc4c8553ef4874b555ac4e01034ddf7d6` |
| `e2/test_e7b_b4_capture_gate_cpu.py` | `2175c8bc969f40429401ca2c2929cd8149e43f06033286d586fa74247fe6bbbd` |
| `e2/e7b_operands_gate.py` | `f4eaa94164db87824575726a7a4004ab0ad9dd42c4db29565727c0f62f64bf99` |
| `e2/e7b_loop.v3.sh` | `99c7d3117f5aab60656235f94d8f09391f57318bb7ae92cb306bbb5345256d94` |
| `e7a/serve_drivers.v12.sh` | `a1a3490de13e8fb8494c0bb32ac31b79076aa40ac699344c26464368cae48102` |
| `e1/e1_api_tokens_from_capture.py` | `757fc74baeedfab7421f589d3cdadeb6ddacd1a3570d7130c9bf521bde18c1ae` |

Gate/test hashes matched before and after independent negatives. The operands-gate hash matched before/after its reproduction. All four gate/test/operands/loop identities were checked again at the end; none changed.

Driver v12 lines 228–229 correctly select `10*COHORT=40` capture rows; line 141 snapshots the new gate, and loop v3 lines 112–114 executes that snapshot. These fixes address the earlier ten-row filter omission. The issues below concern what the captured data can establish.

## Material findings and minimal fixes

### B4-1 — capture identity and substantive hidden/operand coverage are not checked

Gate lines 27–35 sort filenames and equate counts. They ignore `capture_call_index` and `capture_saved_index`, and never compare logit versus hidden capture identities. Lines 39–45 merely require operand files to exist; operand contents are never loaded. State checks omit the layer field despite the header claiming it is checked. Hidden captures need only a `rows` list; the owner's positive fixture has `layers=[]` and no input-hidden values.

Independent CPU corruptions all return **0, C1–C5 PASS**:

- Set both call-1 payloads' `capture_call_index/capture_saved_index` to 999.
- Replace `layer_hidden.call1.pt` with a byte copy of call 0.
- Change row-3/step-1 operand metadata to `row=999, step=999, layer="wrong"`.
- Change the row-3/step-1 state's layer to `"wrong"`.

Minimal fix: require unique, consecutive saved indices matching filenames, matching logit/hidden call identities, and complete required hidden tensors/row coverage. Load both operand and state payloads and verify filename versus payload layer/row/step. Establish the mapping to physical steps using all relevant recorded forwards and the source's capture counters; count equality alone is not identity. Existing source captures include call/saved indices, hidden positions/tensors, operand metadata, and a draft trace. The real stack's hidden `input_ids` field can be `None`; do not require a new capture merely to populate it. Use the existing bytes before considering additional instrumentation. If source counters cannot establish a unique mapping on the actual run, report that run unbound instead of assigning by ordinal.

The preceding operands gate also ignores layer/row metadata (its `load_steps`, lines 15–19, keys only by the embedded step), so it does not repair all these false passes. E1's full API join does not inspect hidden/operand capture identity.

### B4-2 — the accepted draft nodes are not bound to the greedy walk

Gate lines 59–67 prove `argmax(logits[P[i]]) == E[i]`, but never read candidate draft token identities from any source. Thus they do not prove that the next accepted node contains the token predicted by its parent. A CPU case with **every supplied captured input token set to 0**, while every accepted-path emitted/argmax token is above 1000, still returns **C1–C5 PASS**. The owner's synthetic positive has no draft trace at all and also passes. The input-ID mutation illustrates the missing check; it is not a requirement that the real stack populate this optional hidden field.

Minimal fix: reuse the already reviewed B1 greedy-walk logic per ten-row request block, using **`logs/fr10_mtp_draft_trace.jsonl`**, whose records contain ordered `idx`/`ts` and `draft[row]`. Bind that row and record to the actual request/physical step; then start at root 0, take full-vocabulary argmax, follow the first matching child according to operand parents and candidate draft IDs, and stop when no child matches. Compare the resulting entire accepted path/length and accepted-plus-bonus output to the publication and ledger. The existing B1 reproducer is in `p0/monitor/e2-policyB-b1-redteam.md`, lines 166–185; its initial draft index is provenance-bound. For B4, establish the corresponding starting/matching identity from actual row/request/step evidence rather than merely assuming ordinal correspondence. This reuses captured drafts and logits and requests no new capture. The existing operands gate's valid-topology check only checks edges, not draft-token equality.

### B4-3 — “terminal partial prefix” means last B4 appearance, not terminal request output

Gate lines 47–49 derive `last_step_of` only from the N-active subset, and line 62 allows truncation at that subset's last entry. Reproduction: after the owner's shortened final B4 output for `req1`, append a legitimate later B1 physical step/output for the same request and append its token to the API response. Gate still passes and labels the earlier B4 output `terminal_partial_prefix`.

Minimal fix: determine each request's final output from the **complete ledger**, including later reduced-cohort and nonpure outputs, and reconcile it with the API sequence and finish reason. Permit the prefix exception only at an actual final request output with supported terminal clipping. Retain full root-inclusive checking for nonterminal B4 steps. The E1 API join can validate the complete sequence but does not validate this E2 path-length exception.

### B4-4 — preceding operands gate still compares each row against every request's trace

`e7b_operands_gate.py` lines 62–64 flatten every trace record to `acc`, then compare each row's step k against global `acc[k]` and require that row's state count to cover the whole global trace. Loop v3 lines 106–108 now applies this B1 reducer to rows 0,1,2,3.

A correct synthetic three-step/four-request case with all accepts equal to 2 fails G3a on **all four rows**: each has 3 states, while the global trace has 12 records; there are no accepted-length mismatches. This is an actual B4 false rejection, not a request for more coverage. Source trace producer at repository patcher lines 18776–18778 already emits `rid`; actual B1 trace example:
`out-20260922T082815Z-e2-b1-policyB-sync-p072/e7b_001_p072_arm1_none-B_fs_ieee-all/logs/per_req_spec_trace.jsonl`.

Minimal fix: join state row/step to the request identity at its physical step, then compare to that request's corresponding trace event. Do not weaken counts or assume persistent batch-row identity after requests finish. State handoff checks must stay within the same request when slots compact; scoped B4 segments can reuse the complete ledger to prove their request sets.

### B4-5 — API identity reconciliation can fail without failing the loop

Capture gate C5 (lines 69–71) checks only N files with some token strings. The actual API-to-engine request mapping lives in `e1_api_tokens_from_capture.py`, but loop v3 lines 118–124 only print “api-token map REFUSED” if mapping fails and then continue to `E7B LOOP DONE`. Missing recorder files can also skip the join branch at line 117. The current positive synthetic fixture itself uses API IDs `cmpl-0…3` and engine IDs `req0…3`, which cannot map under the helper's explicit rule, although the new capture gate labels it PASS.

Minimal fix: for the selected recorder-on B4 qualification route, require the event file and fail the loop if API mapping or full-sequence join fails. Require exactly four unique request identities mapped to the intended four API responses. Keep this mandatory evidence distinct from the joiner's optional rate precision result. This reuses the existing mapper/joiner, not a new recorder control campaign.

## CPU verification results

The owner's nine controls all pass. Independent added cases demonstrate the gaps:

| Case | Observed result | Required result |
|---|---|---|
| Owner positive fixture | PASS | PASS once populated with real required metadata |
| Wrong call/saved identity | PASS | reject |
| Duplicate hidden payload | PASS | reject |
| Wrong operand row/step/layer | PASS | reject |
| Wrong state layer | PASS | reject |
| All accepted draft input IDs wrong | PASS | reject |
| Shortened last-B4 output followed by B1 continuation | PASS, labeled terminal | reject terminal exception |
| Correct 3-step × 4-request trace | operands gate FAIL on all four rows | pass |

These are minimal CPU negatives and source-binding fixes for the already planned B4 route. No graph, cache, stochastic, model, extra serving axis, or additional GPU campaign is requested. Once repaired, run the corrected offline gates on the existing B4 capture; only evidence actually missing from those captures could justify another capture.

## Exact reproduction

Each block was run through SSH from the remote checkout, with `CUDA_VISIBLE_DEVICES=`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, and `python3 -B -`. Temporary fixtures live only under `tempfile.TemporaryDirectory`, outside the checkout.

### Capture identity and terminal negatives

```python
import ast,sys,os,json,tempfile,subprocess,hashlib,shutil
from pathlib import Path
import torch
root=Path("papers/gdn-tree-scan-mlsys/v2/experiments/e2")
test=root/"test_e7b_b4_capture_gate_cpu.py"; gate=root/"e7b_b4_capture_gate.py"
for p in [test,gate]:print("SOURCE",p.name,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
ns={"__file__":str(test.resolve()),"__name__":"review_fixture"}
nodes=[]
for x in ast.parse(test.read_text()).body:
 if isinstance(x,(ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef)): 
  if isinstance(x,ast.Assign) and any(isinstance(t,ast.Attribute) or isinstance(t,ast.Subscript) for t in x.targets):continue
  nodes.append(x)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(test),"exec"),ns)
with tempfile.TemporaryDirectory(prefix="e2b4-redteam-") as td:
 def run_case(name, mutate):
  d=Path(td)/name; ns["make"](str(d)); mutate(d)
  rc,c=ns["run"](str(d));print("CASE",name,"RC",rc,json.dumps(c),flush=True)
 run_case("positive",lambda d:None)
 def corrupt_capture_identity(d):
  for family in ["final_logits","layer_hidden"]:
   p=d/"logs"/f"{family}.call1.pt"; z=torch.load(p,weights_only=False)
   z["capture_call_index"]=999;z["capture_saved_index"]=999;torch.save(z,p)
 run_case("wrong_capture_identity",corrupt_capture_identity)
 def duplicate_hidden(d):shutil.copyfile(d/"logs/layer_hidden.call0.pt",d/"logs/layer_hidden.call1.pt")
 run_case("duplicate_hidden_payload",duplicate_hidden)
 def corrupt_operand_identity(d):
  p=d/"logs/e7b_operands"/(ns["L"]+".row3.step1.pt")
  z=torch.load(p,weights_only=False);z.update(row=999,step=999,layer="wrong");torch.save(z,p)
 run_case("wrong_operand_identity",corrupt_operand_identity)
 def corrupt_state_layer(d):
  p=d/"logs/e7b_state"/(ns["L"]+".row3.step1.pt")
  z=torch.load(p,weights_only=False);z["layer"]="wrong";torch.save(z,p)
 run_case("wrong_state_layer",corrupt_state_layer)
 def nonterminal_partial(d):
  p=d/"logs/e1_events.jsonl"; ev=[json.loads(l) for l in p.read_text().splitlines()];seal=ev.pop()
  ev.extend([{"event":"physical_step","seq":4,"physical_step_id":103,"request_ids":["req1"],"num_reqs":1},
             {"event":"output_rows","seq":4,"physical_step_id":103,"kind":"pure","num_reqs":1,"rows":[{"request_id":"req1","emitted_ids":[42]}]},seal])
  p.write_text("".join(json.dumps(e)+"\n" for e in ev))
  p=next((d/"cohort").glob("req_1_*/capture_request.json"));z=json.loads(p.read_text());z["response_logprobs_tokens"].append("token_id:42");p.write_text(json.dumps(z))
 run_case("partial_at_last_B4_but_not_terminal",nonterminal_partial)
for p in [test,gate]:print("FINAL_SOURCE",p.name,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
```

### Accepted draft identity negative

```python
import ast,hashlib,json,os,sys,tempfile
from pathlib import Path
import torch
test=Path("papers/gdn-tree-scan-mlsys/v2/experiments/e2/test_e7b_b4_capture_gate_cpu.py")
ns={"__file__":str(test.resolve()),"__name__":"review_fixture"}
nodes=[x for x in ast.parse(test.read_text()).body if isinstance(x,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.Assign)) and not(isinstance(x,ast.Assign) and any(isinstance(t,(ast.Attribute,ast.Subscript)) for t in x.targets))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(test),"exec"),ns)
with tempfile.TemporaryDirectory(prefix="e2b4-greedy-review-") as td:
 ns["make"](td)
 for p in (Path(td)/"logs").glob("layer_hidden.call*.pt"):
  z=torch.load(p,weights_only=False);z["input_ids"]=torch.zeros(40,dtype=torch.long);torch.save(z,p)
 rc,c=ns["run"](td);print("ACCEPTED_DRAFT_INPUT_IDS_ALL_ZERO_BUT_ROOT_ARGMAX_ABOVE_1000",rc,json.dumps(c))
```

### Correct B4 trace false rejection

```python
import ast,hashlib,json,os,sys,tempfile,subprocess
from pathlib import Path
import torch
root=Path("papers/gdn-tree-scan-mlsys/v2/experiments/e2")
gate=root/"e7b_operands_gate.py"
print("SOURCE",gate.name,hashlib.sha256(gate.read_bytes()).hexdigest())
with tempfile.TemporaryDirectory(prefix="e2b4-trace-review-") as td:
 d=Path(td);layer="language_model_model_layers_62_linear_attn"
 for k in ["e7b_operands","e7b_state"]:(d/"logs"/k).mkdir(parents=True)
 parents=[-1,0,1,1,2,2,4,4,6,6]
 (d/"capture_expected.json").write_text(json.dumps({"tree_parents_expected":parents}))
 trace=[]
 for step in range(3):
  for row in range(4):
   op={"step":step,"row":row,"layer":layer,"h0":torch.zeros(2,2),"h0_row":row}
   st={"step":step,"row":row,"layer":layer,"published_after":torch.zeros(2,2),"authoritative_row":row,"accepted_len":2,"path":[1,2],"rows_consistent_before":True}
   torch.save(op,d/"logs/e7b_operands"/f"{layer}.row{row}.step{step}.pt")
   torch.save(st,d/"logs/e7b_state"/f"{layer}.row{row}.step{step}.pt")
   trace.append({"rid":f"req{row}","acc":2,"draft":9})
 (d/"logs/per_req_spec_trace.jsonl").write_text("".join(json.dumps(x)+"\n" for x in trace))
 out=d/"gate.json"
 p=subprocess.run([sys.executable,str(gate),td,"--rows","0,1,2,3","--json",str(out)],capture_output=True,text=True)
 j=json.loads(out.read_text());print("B4_CORRECT_PER_REQUEST_TRACE_RC",p.returncode)
 for row,sub in j["per_row"].items():
  print("ROW",row,json.dumps([c for c in sub["checks"] if not c["pass"]]))
print("FINAL_SOURCE",gate.name,hashlib.sha256(gate.read_bytes()).hexdigest())
```

Owner controls command: `python3 -B papers/gdn-tree-scan-mlsys/v2/experiments/e2/test_e7b_b4_capture_gate_cpu.py <temporary-dir>/results.json`, under the same CPU environment. Observed exit 0, all nine checks PASS.
