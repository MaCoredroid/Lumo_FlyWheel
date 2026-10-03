# E3 identity adapter v2 — independent repair review

27 September 2026. **PASS for the bounded CPU validator preparation. Original findings F1–F4 are closed at the hashes below; no new material adapter-logic blocker was found.** This is not WP/WC approval, launch permission, live collector qualification, or an experimental result. The actual collectors and launcher/evaluator integration remain explicitly unimplemented/unreviewed.

Only this report and local temporary synthetic reproducer files were written. No remote writes, GPU, server, model API, workload execution, evaluator task, or manuscript edit occurred. The original review and superseded snapshot remain unchanged.

## Checked identities

Paths below are relative to `experiments/review-response-20260927/workload-plan/tools/identity-adapter/`.

| File | SHA-256 |
|---|---|
| `MANIFEST.json` | `bee950319dbb59244f2cec4b12081c9fda16eec8352a7f538032f453608f4e95` |
| `e3_preflight.py` | `1bc3be933b08b0eb99432a2b70d0a91aed863b798bf6d3f764775b4dd01e7a06` |
| `HANDOFF.md` | `e11b87652bea80ed18e8c188c9dc90287241f86c97dbbfad128349cc90c47254` |
| `REPAIR-REVIEW.md` | `dd91e2ed04c8fa559e2895cc711e968d93f7380f42e0517e9744b598e04c43b7` |
| `test_e3_preflight.py` | `4293670794e47eec563f568bb3f7261d60111c4e4d427e94811c9079a0ed9cb0` |
| `known-locks.json` | `8ace7c4c3270f45c39cc4e247373c58579716a947b822792f585ef3dd03d493f` |
| `verify_bundle.py` | `fb3f226298004c715e840a6053e9e507a384dbec1927deefbc589f0d6c8f851b` |

## Executed checks

Used `/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B`:

- `verify_bundle.py`: **29/29 sealed payloads verified**, all original source anchors verified; independently pinned the delivered manifest and adapter-source hashes before review.
- `-m unittest discover -s . -p test_e3_preflight.py -q`: **102 tests passed** (0.726 seconds).
- Independently reran **27 synthetic controls**: four intended positive identities passed (launch, agent, evaluator, and failed agent with empty patch); **23 invalid identities refused**, including all eight original counterexample classes and additional swapped-valid/binding/type/chronology cases below.
- Reexecuted the exact DRAFT CLI from `VALIDATION.json`: exit **2**, `REFUSED`, `runtime freeze status differs from bound identity`.
- Rehashed the superseded v1 manifest (`b9a13706…`) and all its 11 listed payload files: all intact (12 original files including its manifest). The initial report remains SHA-256 `796632efabc2103297153cdc533c048cda77c6edccea1c5b3de7d510db479dec`.

## Closure of original findings

**F1, terminal/predecessor identity — closed.** `e3_preflight.py:145–161,317–323,390–395` parses nonempty terminal records; binds frozen order, phase, configuration, task, retry, boot and clock; rejects nonterminal state and contradictory chronology; and binds completion to the exact prediction and runtime subject. Empty predecessor/completion records, a different valid prior order row, a new prediction under the same attempt, wrong phase, Boolean retry, reversed times and post-gate recording all refuse. Failed terminal state plus an empty patch remains admissible, correctly avoiding success filtering.

**F2, owned stopped evaluator container — closed within validator scope.** `:373–404` requires the actual container identifier, exact image/TestSpec/task/run/prediction subject, ownership labels, `created` and stopped state, never-started sentinel, matching creation time, no applied patch/tests, and a dedicated approved-collector observation. A different syntactically valid container ID with the original observation fails its subject binding; running/wrong/unowned containers and integer-zero-as-false refuse. The pure adapter still cannot create or enforce that boundary; the future wrapper must consume the exact inspected container.

**F3, source members and collector authority — closed.** `:57–79,120–143,244–257,362–371` parses nonempty purpose-specific manifests, hashes every source member, checks byte counts and duplicate paths, restricts collector source hashes to approved roles, and compares the complete runtime-observed member maps. The original changed-member, unapproved-collector and empty-collector counterexamples now refuse. Changed runtime member hashes, Boolean member byte counts and missing completion collector/raw evidence also refuse.

**F4, chronology and stale evidence — closed against the declared trust boundary.** `:81–143` uses the frozen execution window, clock, planned per-attempt boot, stage interval, maximum age and bounded skew. `1900` observations, wrong boot or stage-window bindings, and invalid terminal times refuse. It does not pretend that a receipt author's timestamp is an independently authenticated wall clock. The handoff explicitly requires the reviewed real launcher/collectors to obtain and enforce actual clock/state observations, preserve earlier stage timestamps and retain immutable gate receipts.

## Remaining integration requirements, not reopened validator findings

Before any WP/WC or workload launch, the parent still needs actual approved freeze values/source manifests/collector implementations and reviewed enforcing call sites. Those must (1) obtain real observations, (2) stop dependent execution on refusal at prelaunch, pre-agent and owned stopped-container/pre-evaluator boundaries, (3) retain earlier gate receipts and chronology, and (4) serialize/reserve attempts without race or silent retry. Honest execution of an approved collector is a trust assumption here; a self-authored coherent packet is not live evidence. The source explicitly returns `qualification_experiments_completed: 0`.

The nonpilot image locks, retry extension and any differing-template equivalence remain pending as documented. No additional adapter code fix is required by this bounded re-review; no inference or new numerical experiment is requested.

## Independent control results

| Case | Stage | Result |
|---|---|---|
| `synthetic_positive_launch` | launch | IDENTITY_GATE_PASS |
| `synthetic_positive_agent` | agent | IDENTITY_GATE_PASS |
| `synthetic_positive_evaluator` | evaluator | IDENTITY_GATE_PASS |
| `failed_completion_empty_patch_retained` | evaluator | IDENTITY_GATE_PASS |
| `empty_agent_completion` | evaluator | REFUSED: empty agent_completion receipt |
| `other_attempt_running_completion` | evaluator | REFUSED: terminal schema differs from bound identity |
| `empty_predecessor` | launch | REFUSED: empty attempt_closure receipt |
| `running_wrong_unowned_container` | evaluator | REFUSED: actual evaluator container ID must be a SHA256 |
| `1900_observation` | launch | REFUSED: observation timestamp outside frozen attempt/boot/stage window |
| `unapproved_collector` | launch | REFUSED: collector source/role is not approved in freeze |
| `empty_collector` | launch | REFUSED: collector source/role is not approved in freeze |
| `changed_source_under_same_manifest` | launch | REFUSED: source member /synthetic/runtime-AR.py differs from bound identity |
| `other_valid_prior_order` | launch | REFUSED: terminal order_row differs from bound identity |
| `same_attempt_new_prediction_stale_completion` | evaluator | REFUSED: terminal prediction binding differs from bound identity |
| `terminal_retry_boolean` | evaluator | REFUSED: terminal retry_index differs from bound identity |
| `terminal_wrong_phase` | evaluator | REFUSED: terminal phase differs from bound identity |
| `terminal_reversed_times` | evaluator | REFUSED: terminal chronology outside frozen attempt ordering |
| `terminal_recorded_after_gate` | evaluator | REFUSED: terminal chronology outside frozen attempt ordering |
| `other_valid_container_id` | evaluator | REFUSED: observation subject binding differs from bound identity |
| `container_observation_wrong_subject` | evaluator | REFUSED: observation subject binding differs from bound identity |
| `container_running_integer_zero` | evaluator | REFUSED: evaluator container must be stopped differs from bound identity |
| `source_bytes_boolean` | launch | REFUSED: source member byte count must be positive |
| `runtime_loaded_member_hash_changed` | agent | REFUSED: actual runtime source members differs from bound identity |
| `timeline_observation_wrong_boot` | launch | REFUSED: observation frozen boot differs from bound identity |
| `observation_wrong_stage_window` | launch | REFUSED: observation stage window differs from bound identity |
| `partial_completion_missing_collector` | evaluator | REFUSED: collector source/role is not approved in freeze |
| `completion_raw_evidence_missing` | evaluator | REFUSED: observation raw evidence missing/duplicate |

## Reproducer

The exact local script `/tmp/e3_adapter_v2_independent_review.py` is included below for persistence; its output was `/tmp/e3_adapter_v2_independent_review.json`. It imports the current validator and test fixture without changing either. Synthetic fixtures intentionally stand in for nonexistent approved live collectors; they cannot qualify a real campaign.

Script SHA-256 `896d59251619bb5e94db0913929adc5ddd3a03f0e953afe3e6f29b44e68d1809`.

```python
import sys,json,hashlib,pathlib,copy,subprocess
BASE=pathlib.Path('/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/tools/identity-adapter')
sys.path.insert(0,str(BASE))
import e3_preflight as e
from test_e3_preflight import AdapterTests as T
assert e.sha((BASE/'MANIFEST.json').read_bytes())=='bee950319dbb59244f2cec4b12081c9fda16eec8352a7f538032f453608f4e95'
assert e.sha((BASE/'e3_preflight.py').read_bytes())=='1bc3be933b08b0eb99432a2b70d0a91aed863b798bf6d3f764775b4dd01e7a06'
rows=[]
def check(name,stage,fn,positive=False):
    t=T();t.setUp()
    try:
        if stage=='agent':t.agent()
        if stage=='evaluator':t.evaluator()
        fn(t)
        try:r=t.check(stage);out={'status':r['status']}
        except e.Refusal as x:out={'status':'REFUSED','reason':str(x)}
        except Exception as x:out={'status':'EXCEPTION','type':type(x).__name__,'reason':str(x)}
        assert out['status']==('IDENTITY_GATE_PASS' if positive else 'REFUSED'),(name,out)
        rows.append({'name':name,'stage':stage,**out})
    finally:t.doCleanups()
def evhash(t,field,data):t.packet['evaluation'][field]=t.add('mutated-'+field,data)
def change_completion(t,fn):
    ev=t.packet['evaluation'];ev['agent_completion_receipt_sha256']=t.edit_evidence(ev['agent_completion_receipt_sha256'],fn,'changed-completion.json')
def renew_eval(t):
    ev=t.packet['evaluation'];ev['container_receipt_sha256']=t.observation('evaluator_container',{k:v for k,v in ev.items() if k!='container_receipt_sha256'})
def prior_empty(t):
    t.next_attempt();t.packet['prior_attempts'][0]['receipt_sha256']=t.add('empty-prior',b'')
def changed_member(t):
    member=t.source_member();pathlib.Path(t.evidence[member['sha256']]).write_bytes(b'changed source after manifest seal')
def wrong_prior(t):
    t.next_attempt();row=t.packet['prior_attempts'][0]
    row['receipt_sha256']=t.edit_evidence(row['receipt_sha256'],lambda x:x.update(order_row=t.freeze['order'][1],attempt_id=e.attempt_id(t.fsha,t.freeze['order'][1])),'other-valid-order-row.json')
def reversed_terminal(t):change_completion(t,lambda x:x.update(started_at_utc='2026-09-27T00:00:19+00:00'))
def late_terminal(t):change_completion(t,lambda x:x.update(recorded_at_utc='2026-09-27T00:00:21+00:00'))
def same_id_other_prediction(t):
    old=e.load(t.evidence[t.packet['evaluation']['prediction_sha256']]);old[0]['model_patch']='different patch'
    evhash(t,'prediction_sha256',e.canonical(old));renew_eval(t)
def eval_wrong_subject(t):
    ev=t.packet['evaluation'];h=ev['container_receipt_sha256'];ev['container_receipt_sha256']=t.edit_evidence(h,lambda x:x.update(subject_sha256='f'*64))
def bool_terminal(t):change_completion(t,lambda x:x.update(retry_index=False))
def bool_bytes(t):
    h=t.freeze['configurations'][t.packet['order_row']['configuration_id']]['runtime_source_manifest_sha256']
    h=t.edit_evidence(h,lambda x:x['files'][0].update(bytes=True),'bad-size-manifest.json');t.change_config('runtime_source_manifest_sha256',h)
def failed_empty_positive(t):change_completion(t,lambda x:x.update(state='FAILED'));renew_eval(t)
for s in ('launch','agent','evaluator'):check('synthetic_positive_'+s,s,lambda t:None,True)
check('failed_completion_empty_patch_retained','evaluator',failed_empty_positive,True)
check('empty_agent_completion','evaluator',lambda t:evhash(t,'agent_completion_receipt_sha256',b''))
check('other_attempt_running_completion','evaluator',lambda t:evhash(t,'agent_completion_receipt_sha256',e.canonical({'attempt_id':'another-attempt','state':'RUNNING'})))
check('empty_predecessor','launch',prior_empty)
check('running_wrong_unowned_container','evaluator',lambda t:t.packet['evaluation'].update(container_running=True,container_status='running',container_id='wrong-container',ownership_attempt_id='other-attempt'))
check('1900_observation','launch',lambda t:t.edit_observation(lambda x:x.update(observed_at_utc='1900-01-01T00:00:00+00:00')))
check('unapproved_collector','launch',lambda t:t.edit_observation(lambda x:x.update(collector_sha256=t.add('unapproved.py',b'print("new collector")\n'))))
check('empty_collector','launch',lambda t:t.edit_observation(lambda x:x.update(collector_sha256=t.add('empty.py',b''))))
check('changed_source_under_same_manifest','launch',changed_member)
check('other_valid_prior_order','launch',wrong_prior)
check('same_attempt_new_prediction_stale_completion','evaluator',same_id_other_prediction)
check('terminal_retry_boolean','evaluator',bool_terminal)
check('terminal_wrong_phase','evaluator',lambda t:change_completion(t,lambda x:x.update(phase='confirmation')))
check('terminal_reversed_times','evaluator',reversed_terminal)
check('terminal_recorded_after_gate','evaluator',late_terminal)
check('other_valid_container_id','evaluator',lambda t:t.packet['evaluation'].update(container_id='c'*64))
check('container_observation_wrong_subject','evaluator',eval_wrong_subject)
check('container_running_integer_zero','evaluator',lambda t:t.packet['evaluation'].update(container_running=0))
check('source_bytes_boolean','launch',bool_bytes)
check('runtime_loaded_member_hash_changed','agent',lambda t:t.packet['runtime']['observed_source_files']['runtime_source'][0].update(sha256='d'*64))
check('timeline_observation_wrong_boot','launch',lambda t:t.packet.update(timeline_receipt_sha256=t.edit_evidence(t.packet['timeline_receipt_sha256'],lambda x:x.update(boot_id='other-approved-boot'))))
check('observation_wrong_stage_window','launch',lambda t:t.edit_observation(lambda x:x.update(window={'boot_started_at_utc':'2026-09-27T00:00:10+00:00','agent_gate_at_utc':'2026-09-27T00:00:12+00:00'})))
check('partial_completion_missing_collector','evaluator',lambda t:change_completion(t,lambda x:x.pop('collector_sha256')))
check('completion_raw_evidence_missing','evaluator',lambda t:change_completion(t,lambda x:x.update(raw_evidence_sha256=[])))
validation=e.load(BASE/'VALIDATION.json');p=subprocess.run(validation['draft_cli_command'],text=True,capture_output=True)
assert p.returncode==2 and json.loads(p.stdout)['status']=='REFUSED'
manifest=e.load(BASE/'MANIFEST.json');old=BASE/'superseded/v1-b9a13706d23d';oldm=e.load(old/'MANIFEST.json')
for row in oldm['files']:
 path=old/row['path'];assert e.sha(path.read_bytes())==row['sha256'] and path.stat().st_size==row['bytes']
assert e.sha((old/'MANIFEST.json').read_bytes())=='b9a13706d23dd5a54cccd5c4f4ed65ece05516d0c23f4bf15e3ac6e240fabf3e'
original=BASE.parents[5]/'notes/review-response-20260927/e3-identity-adapter-review.md'
# The initial review remains immutable; resolve paper root explicitly.
original=pathlib.Path('/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/papers/gdn-tree-scan-mlsys/v2/notes/review-response-20260927/e3-identity-adapter-review.md')
assert e.sha(original.read_bytes())=='796632efabc2103297153cdc533c048cda77c6edccea1c5b3de7d510db479dec'
out={'reviewed':{name:e.sha((BASE/name).read_bytes()) for name in ['MANIFEST.json','e3_preflight.py','HANDOFF.md','REPAIR-REVIEW.md','test_e3_preflight.py','known-locks.json','verify_bundle.py']},'controls':rows,'draft_cli':{'exit':p.returncode,'result':json.loads(p.stdout)},'superseded_payload_files_verified':len(oldm['files']),'initial_review_unchanged':True,'reproducer_sha256':e.sha(pathlib.Path(__file__).read_bytes())}
pathlib.Path('/tmp/e3_adapter_v2_independent_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
```
