from pathlib import Path
import contextlib, datetime as dt, hashlib, importlib.util, io, json, sys, tempfile, types
import torch
torch.set_num_threads(1)
v=Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2")
ins=v/"experiments/e7a/inspect_capture.py"
spec=importlib.util.spec_from_file_location("review08_inspector",ins); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
res={"utc":dt.datetime.now(dt.timezone.utc).isoformat(),"scope":"Synthetic CPU provenance negative controls only; zero model inference", "inspector_sha256":hashlib.sha256(ins.read_bytes()).hexdigest(),"cases":{}}
parents=[-1,0,1,1,2,2,4,4,6,6]; n=len(parents)
with tempfile.TemporaryDirectory(prefix="lumo-review08-") as td:
 run=Path(td);(run/"logs").mkdir()
 pool=run/"pool.json";txt="synthetic reviewer fixture";hs=hashlib.sha256(txt.encode()).hexdigest()
 pool.write_text(json.dumps({"prefixes":[{"id":"fixture","text":txt,"prefix_sha256":hs,"tokens":256}]}))
 exp={"prefix_id":"fixture","layer_prefix":"language_model.model.layers.62.linear_attn","tree_parents_expected":parents,"pool_path":str(pool),"pool_sha256":hashlib.sha256(pool.read_bytes()).hexdigest()}
 (run/"capture_expected.json").write_text(json.dumps(exp))
 p={"schema":"fr10.tree_gdn_scan_capture.v1","layer_prefix":exp["layer_prefix"],"batch_index":0,"tree_parent":parents,"n_actual":n,"n_pad":16,"state_index":0,"output_scale":128**-.5,"serving_state":None}
 for k,shape,dtype in [("query_spec",(n,16,128),torch.bfloat16),("key_spec",(n,16,128),torch.bfloat16),("value_tree",(n,48,128),torch.bfloat16),("a",(n,48),torch.bfloat16),("b",(n,48),torch.bfloat16),("A_log",(48,),torch.float32),("dt_bias",(48,),torch.bfloat16),("h0",(48,128,128),torch.float32),("serving_out",(n,48,128),torch.bfloat16)]:
  p[k]=torch.ones(shape,dtype=dtype)
 pay=run/"logs/tree_gdn_capture_payload.pt";torch.save(p,pay)
 now=dt.datetime.now(dt.timezone.utc);stamp=now.isoformat()
 (run/"driver_trace.txt").write_text(f"request_start_utc={stamp}\nrequest_end_utc={stamp}\npayload_mtime_utc={stamp} size={pay.stat().st_size}\n")
 depth=[len(mod._anc(parents,i)) for i in range(n)]
 dp={"event":"tree_depth_positions","tree_n":n,"num_scheduled_tokens":[n],"base_contract":"state=num_computed_tokens_cpu-1,mrope=num_computed_tokens_cpu","flat_first_tree":list(range(n)),"depth_first_tree":depth}
 lcp={"req_index":0,"node_count":n-1,"draft_token_ids":list(range(n-1)),"emitted_tokens":list(range(32)),"ts":now.timestamp()}
 for name,obj in [("fr10_tree_depth_positions.jsonl",dp),("tree_path_lcp.jsonl",lcp)]:
  (run/"logs"/name).write_text(json.dumps(obj)+"\n");(run/"logs"/(name+".lines_before_request")).write_text("0")
 (run/"e7a_capture_shim.json").write_text(json.dumps({"edits":[{"anchor":"guard"},{"anchor":"serving_state"}],"sha256_before":"a","sha256_after":"b"}))
 for case,resp,available in [("positive_exact","complete text",True),("tokenizer_unavailable","complete text",False),("negative_empty_response","",True),("negative_truncated_response","complete",True)]:
  req={"prefix_id":"fixture","prefix_sha256":hs,"prompt_sha256":hs,"response_id":"fixture-response","usage":{"prompt_tokens":256,"completion_tokens":32},"response_text":resp}
  (run/"capture_request.json").write_text(json.dumps(req))
  class Tok:
   @staticmethod
   def from_pretrained(*args,**kwargs):
    if not available: raise ImportError("Reviewer forced unavailable tokenizer")
    return Tok()
   def decode(self,*args,**kwargs): return "complete text"
  fake=types.ModuleType("transformers");fake.AutoTokenizer=Tok;sys.modules["transformers"]=fake
  sys.argv=[str(ins),str(run)]
  with contextlib.redirect_stdout(io.StringIO()): rc=mod.main()
  out=json.loads((run/"capture_provenance.json").read_text())
  res["cases"][case]={"rc":rc,"all_pass":out["all_pass"],"unverified":out.get("unverified"),"failed":[k for k,x in out["checks"].items() if x["pass"] is False],"token_binding":out["checks"].get("emitted_tokens_decode_to_response_text")}
out=v/"p0/monitor/2026-09-22-review-08-inspector-controls.json";out.write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
