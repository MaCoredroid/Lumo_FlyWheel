# B4 API token-string inversion — independent CPU review

**Decision: the 128 recorded API token strings are uniquely recoverable as token IDs under the exact pinned formatter. No additional serving boot is required solely for this format omission.** These are explicitly reconstructed API identities, not directly recorded `token_id:` strings. Preserve the original capture JSON unchanged and attach the hash-bound derivation; future E1/cohort requests must ask for direct token IDs.

Run examined:

`/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T084319Z-e2-b4-policyB-sync-cohort4/e7b_001_p072_arm1_none-B_fs_ieee-all`

The four request records have 32 tokens each, 72 distinct strings in total, all finish_reason `length`. **No E1 ledger, draft trace, model logits, or desired IDs were read to select inverse candidates.** The independent output sequences also decode exactly to each captured response text; that is a consistency check, not how IDs were selected.

## Exact formatter and decoder

The image is `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`; the run records image ID `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`.

Static source was read via uniquely named **stopped** `docker create` objects, `docker cp`, then removal of only the created object. None was started. The pinned tokenizers package was copied to temporary host CPU storage and imported in a fresh subprocess. Temporary objects and files were removed. No GPU or model process was launched.

Image package root: `/usr/local/lib/python3.12/dist-packages/`.

| Relative source | SHA-256 | Relevant behavior |
|---|---|---|
| `vllm/_version.py` | `b85e9e94c79b0c19ebf7c3dbf1a432dd1c70c803bf0cfed80bdc32cd457ec78b` | vLLM 0.19.2rc1.dev134+gfe9c3d6c5 |
| `vllm/entrypoints/openai/completion/serving.py` | `ffb08a4e5113ed2bf2f23b5389f4f01f1b3393a7b3a5028e26e4c91ce3831bc5` | lines 585–620: absent direct-ID option delegates token representation to `_get_decoded_token` |
| `vllm/entrypoints/openai/engine/serving.py` | `2132293166071cebafa04479532467d07f9c970c2912542fa3fe66d94ed4a3fa` | lines 723–741: use non-None `logprob.decoded_token`, otherwise singleton decode |
| `vllm/tokenizers/detokenizer_utils.py` | `9f2216dde3b41fba2f494cb21ede2619b9a628eaf7229fa30202bedc13792974` | lines 83–104: independently decode each candidate ID |
| `vllm/v1/engine/logprobs.py` | `62db8c773edaa39208b5a27ba876a1706801d66276925933cda539675383a1ea` | lines 92–105 then 249–346: singleton decode plus exact UTF-8 correction for strings ending U+FFFD, using up to four preceding sampled IDs |
| `vllm/tokenizers/hf.py` | `f462103da34122052402a996c127512d1e1c5d0aa25fc130cf97d4082a92f6f8` | tokenizer factory/caching, lines 73–125 |
| `transformers/__init__.py` | `2a80bb92259b69439d6d180e7a42f5d49677fab14246f7139797eec1979574da` | transformers 5.6.0 |
| `transformers/models/qwen2/tokenization_qwen2.py` | `fac4e6576bfe2369731be147a4e530f262bdf32f2ac50436f96f0d8bdd2fc628` | Qwen2Tokenizer inherits TokenizersBackend |
| `transformers/tokenization_utils_tokenizers.py` | `354caf3d9cef9589c688261631b265d61b1beef5449cece5d13957e48394d046` | lines 1018–1042: backend decode, default skip_special_tokens=False, optional whitespace cleanup |
| `tokenizers-0.22.2.dist-info/METADATA` | `15a5ddaf489f592b77e0a934c0eeb46b51130a94064ca129232afdb0a3868efc` | pinned tokenizers 0.22.2 |
| `tokenizers/tokenizers.abi3.so` | `fa049ce975669d8a90fb48960f412e626fa54cf596c2f75d6820949f4888e910` | exact native decoder used in final CPU enumeration |

Model root: `/models/qwen3.6-27b-fp8/`, mounted read-only by the capture container.

| File | SHA-256 |
|---|---|
| `tokenizer.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `tokenizer_config.json` | `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b` |
| `vocab.json` | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |
| `merges.txt` | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `config.json` | `f78c412bfdec65a88c8aa2a031d39c2fda32e3377ae48a77f971bc40a4f095df` |

The decoder is ByteLevel; cleanup is false. `vocab.json` exactly equals `tokenizer.json.model.vocab`. The 26 added-token identities already in tokenizer.json match the config. **The config adds seven additional audio/TTS tokens at IDs 248070–248076**, so a raw tokenizer.json-only load is incomplete. Final enumeration installs all 33 configured added tokens and asserts every configured ID. Domain: **248077 tokenizer IDs plus 243 padded model IDs, all model output IDs 0–248319**.

## Exhaustive, noncircular result

For each position, enumerate the whole domain through singleton decode followed by the **AST-extracted exact pinned** `_verify_tokens`/`_correct_decoded_token` methods. Context is only the prior four IDs already uniquely recovered from preceding API strings. There are 951 replacement-ending candidate strings to which the contextual repair may apply. Require exactly one formatter-matching candidate before advancing to the next position. This inductive procedure found **no missing or ambiguous position**.

Final SHA-256 of compact UTF-8 JSON `[[id, singleton_decoded_string], ...]` over all 248320 IDs:

`0423a31175fa2895310e53dd325b3b0b0b74f558a41db8f1bcb045d1ccd83a40`

Final SHA-256 of all 128 ordered `{position,string,context_ids,candidates}` records (requests sorted by directory):

`b4b9cad2bb0c36a6a8f634fdcab8980adeff0ce95aeb4ba0c8763051c1d7b08a`

| Request | Original capture_request.json SHA-256 | Candidate/context proof SHA-256 |
|---|---|---|
| req_0_p072 | `ebe977a2189c4d3c6a6b0eca59598c76f7b77ddcca8a132546372e81a4ee5980` | `2f4486281aef005527412affe376264059975f0d309aa1d6bc803d9dc2b9b42b` |
| req_1_p017 | `38f65eaf163434a5b0b0420cfb88d923275e449ddc5df1c0c4b174c30d2b4afe` | `335970f23fdb9aedd23f3dfef6aec996035f5dcaac93b04768e8b7ce64095dc4` |
| req_2_p085 | `a970cd4a6612eb466700697f644b1db997647f4ac6b901525ad1080d7ca9e57e` | `27dea2cb5efc8d3570fa55f2cbc2642a026149c59328cab57b44b5e2166dcfe0` |
| req_3_p095 | `24f741a9a232290868b48155db928e64c37d08df2ae8f22b2b67ac546a13d8cf` | `d94698b3e62f9ebeecd816cd135a69746548e9b0e8616323518543e7ad53c583` |

Recovered IDs, keyed by the **API response identity**, without consulting engine IDs:

```json
{
  "cmpl-a33355be5168a83b": [6813,12333,36349,853,6971,6813,12333,36349,648,6971,6813,12333,36349,788,6971,6813,12333,36349,2135,6971,6813,12333,36349,736,6971,6813,12333,36349,1824,6971,6813,12333],
  "cmpl-be7c5fc6026dc3ca": [264,2972,14892,22258,332,21853,421,11693,279,220,16,21,13,19,20,85123,198,429,279,220,17,24,13,16,16,15686,13,561,491,795,4774,279],
  "cmpl-822fbbe30317f121": [62987,5303,506,62987,854,13,561,8057,2099,369,23185,1518,279,8057,8213,26,279,62987,33877,513,23185,1518,62987,8213,13,1061,27028,2107,3274,32524,318,7838],
  "cmpl-bba5676034b3072d": [8288,2686,1892,424,369,264,2972,588,61384,332,220,19,18295,23014,318,1719,220,21,19,45668,1994,23985,220,16,19,19,20112,33,8,421,8311,310]
}
```

The earlier provisional singleton-only map used host tokenizers 0.22.1. Its `dbb728…` hash is not the final proof. The intermediate pinned `602729…` map omitted the seven config-only special tokens. The final `0423a311…` map above supersedes both. The unique observed candidates never changed.

## Scope and integration

This resolves only the current cohort client's API-format omission. It does not by itself establish request-to-physical-step capture identity, greedy correctness, recurrent handoff, or E1 rate validity. Feed a separately labeled, hash-bound recovered API-ID sidecar through the existing full request/ledger join; do not replace original strings in the raw capture or present this run as having requested direct IDs. Future clients should explicitly set `return_tokens_as_token_ids=true` (and can also retain direct `token_ids` when requested).

The raw capture record includes `response_raw_sha256`, but this review's input is the retained `capture_request.json`; no raw HTTP-response body file was present in the four request directories. The exact-string preservation and extraction come from those immutable recorded fields. Do not assert a newly recovered raw HTTP response body.

## Exact final CPU reproduction

Run this Python script on the remote host. It extracts only static source and the tokenizer library from a stopped inspection container, executes the exhaustive formatter inversion on CPU, prints hashes/results, and removes its own object. No vLLM/model modules are imported or served.

```python
child = r"""
import ast,json,hashlib,sys,types
from pathlib import Path
import tokenizers
td=Path(sys.argv[1]);base=Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T084319Z-e2-b4-policyB-sync-cohort4/e7b_001_p072_arm1_none-B_fs_ieee-all")
tp=Path("/models/qwen3.6-27b-fp8/tokenizer.json"); cfg=json.loads(Path("/models/qwen3.6-27b-fp8/config.json").read_text())
tok=tokenizers.Tokenizer.from_file(str(tp))
tc=json.loads(Path("/models/qwen3.6-27b-fp8/tokenizer_config.json").read_text())
for token_id,record in sorted(tc["added_tokens_decoder"].items(),key=lambda x:int(x[0])):
 tok.add_special_tokens([tokenizers.AddedToken(**record)])
 assert tok.token_to_id(record["content"])==int(token_id)
actual_vocab=tok.get_vocab(with_added_tokens=True); upper=max(max(actual_vocab.values())+1,cfg.get("vocab_size",0),cfg.get("text_config",{}).get("vocab_size",0))
class ExactBackend:
 def decode(self,ids):return tok.decode(ids,skip_special_tokens=False)
obj=types.SimpleNamespace(tokenizer=ExactBackend())
source=(td/"logprobs_source.py").read_text()
cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=="LogprobsProcessor")
selected=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ["_correct_decoded_token","_verify_tokens"]]
ns={};exec(compile(ast.Module(body=ast.parse("from __future__ import annotations").body+selected,type_ignores=[]),"pinned/logprobs.py","exec"),ns)
for n in ["_correct_decoded_token","_verify_tokens"]:setattr(obj,n,types.MethodType(ns[n],obj))
ids=list(range(upper)); decoded=[obj.tokenizer.decode([i]) for i in ids]
base_map_hash=hashlib.sha256(json.dumps(list(zip(ids,decoded)),ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
print("PINNED_TOKENIZERS",tokenizers.__version__,"MODULE",tokenizers.__file__)
print("ID_DOMAIN",json.dumps({"model_upper_exclusive":upper,"tokenizer_vocab_count":len(actual_vocab),"tokenizer_max_id":max(actual_vocab.values()),"replacement_ending_candidates":sum(s.endswith(chr(65533)) for s in decoded),"base_full_map_sha256":base_map_hash}),flush=True)
records=[]; requests=[]
for p in sorted(base.glob("cohort/req_*/capture_request.json")):
 d=json.loads(p.read_text()); recovered=[]; details=[]
 for pos,observed in enumerate(d["response_logprobs_tokens"]):
  context=recovered[-4:]
  formatted=obj._verify_tokens(decoded.copy(),ids,context)
  cands=[i for i,s in enumerate(formatted) if s==observed]
  details.append({"position":pos,"string":observed,"context_ids":context,"candidates":cands})
  if len(cands)!=1:raise RuntimeError(json.dumps({"request":p.parent.name,"position":pos,"candidates":cands}))
  recovered.append(cands[0])
 result={"request":p.parent.name,"capture_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"response_id":d["response_id"],"ids":recovered,"observed_positions":len(details),"all_unique":True,"decode_concat_equals_text":obj.tokenizer.decode(recovered)==d["text"],"candidates_and_context_sha256":hashlib.sha256(json.dumps(details,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()}
 records+=details;requests.append(result);print("REQUEST",json.dumps(result,ensure_ascii=False),flush=True)
print("ALL_POSITIONS",len(records),"ALL_CONTEXT_CANDIDATE_RECORDS_SHA256",hashlib.sha256(json.dumps(records,ensure_ascii=False,separators=(",",":")).encode()).hexdigest())
print("NO_LEDGER_OR_DRAFT_TRACE_READ")
"""
import subprocess,tarfile,io,hashlib,uuid,tempfile,os
from pathlib import Path
name="e2-api-exact-cpu-"+uuid.uuid4().hex[:12];image="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
cid=subprocess.check_output(["docker","create","--name",name,"--network","none","--entrypoint","/bin/true",image],text=True).strip()
def cp(rel):
 p=subprocess.run(["docker","cp",cid+":/usr/local/lib/python3.12/dist-packages/"+rel,"-"],capture_output=True)
 if p.returncode:print("ABSENT",rel);return {}
 with tarfile.open(fileobj=io.BytesIO(p.stdout)) as t:return {m.name:t.extractfile(m).read() for m in t if m.isfile()}
try:
 with tempfile.TemporaryDirectory(prefix="e2-api-pinned-cpu-") as td:
  root=Path(td)
  for n,b in cp("tokenizers").items():
   p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
   if n.endswith(".so"):print("PINNED_TOKENIZERS_ELF",n,hashlib.sha256(b).hexdigest(),flush=True)
  for rel in ["vllm/v1/engine/logprobs.py","transformers/tokenization_utils_tokenizers.py","transformers/models/qwen2/tokenization_qwen2.py"]:
   files=cp(rel)
   for n,b in files.items():
    print("SOURCE",rel,hashlib.sha256(b).hexdigest(),flush=True)
    if rel.endswith("logprobs.py"):(root/"logprobs_source.py").write_bytes(b)
    else:
     ls=b.decode().splitlines()
     inds={j for i,l in enumerate(ls) if any(x in l for x in ["def _decode", "def decode(", "TokenizersBackend", "clean_up_tokenization_spaces", "skip_special_tokens"]) for j in range(max(0,i-2),min(len(ls),i+10))}
     for i in sorted(inds):print(f"{i+1}: {ls[i]}")
  subprocess.run(["python3","-B","-c",child,td],check=True,env={**os.environ,"PYTHONPATH":td,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1","TOKENIZERS_PARALLELISM":"false"})
finally:subprocess.run(["docker","rm",cid],check=True,capture_output=True);print("REMOVED_ONLY",name)
```

Observed final output: tokenizers 0.22.2; domain 248320; tokenizer vocabulary 248077; 951 replacement-ending candidate strings; four requests with 32/32 unique positions and exact response-text reconstruction; all-candidate/context digest `b4b9cad2…`.

