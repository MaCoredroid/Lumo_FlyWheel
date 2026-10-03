#!/usr/bin/env python3
"""Execute actual helper_hashes AST, no production imports/Torch/GPU."""
import ast,hashlib,json,os,pathlib
V2=pathlib.Path(__file__).resolve().parents[3]
R=V2/'experiments/review-response-20260927/tools/q1_component_runner_v2_2.py'
T=ast.parse(R.read_text());names={'HELPER_FILES','REPO_HELPER_FILES'}
selected=[n for n in T.body if (isinstance(n,ast.FunctionDef) and n.name=='helper_hashes') or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
ns={'os':os,'HERE':str(R.parent),'ROOT':str(V2.parents[2]),'sha256_file':sha}
exec(compile(ast.Module(body=selected,type_ignores=[]),str(R),'exec'),ns)
m=ns['helper_hashes']()
assert 'q1_component_runner_v2_2.py' not in m
assert m['q1_component_runner_v2_1.py']=='b7241e8b925a9b5d9e921673d84f84b701f9b1404ed71d727bc5ade37d15cb65'
print(json.dumps({'runner_source_sha256':sha(R),'actual_runner_sha256':sha(R),'actual_helper_map':m,'expected_helper_key':'q1_component_runner_v2_2.py','failure_reproduced':True,'no_torch_or_candidate_execution':True},indent=2))
