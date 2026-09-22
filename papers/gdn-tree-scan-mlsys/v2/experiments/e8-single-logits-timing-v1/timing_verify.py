#!/usr/bin/env python3
"""Read-only receipt/source/config checker and exact-support cell reducer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'qualification-source'))
from e8_verify import check, sha, stage as qstage, run as qrun


def stage(root):
    root = Path(root)
    m = json.loads((root/'timing_manifest.json').read_text())
    check(m['schema'] == 'e8.timing.freeze.v1' and len(m['cells']) == 6, 'timing freeze')
    for rel, digest in m['files'].items():
        p = root/rel
        check(p.resolve().is_relative_to(root.resolve()) and p.is_file() and sha(p) == digest, 'timing source '+rel)
    check(sha(root/'qualification-source/manifest.json') == m['qualification_manifest_sha256'], 'qualified source identity')
    qstage(root/'qualification-source')
    return m


def qualification(root, qr):
    root, qr = Path(root), Path(qr)
    m = stage(root)
    check(not (qr/'FAILED.json').exists(), 'qualification has failure marker')
    final = json.loads((qr/'QUALIFICATION_PASS.json').read_text())
    check(final.get('status') == 'QUALIFICATION_PASS' and final.get('arms') == ['on','off'], 'two-arm qualification missing')
    check(final.get('stage_manifest_sha256') == m['qualification_manifest_sha256'] == sha(qr/'stage/manifest.json'), 'qualification source differs')
    for arm in ('on','off'):
        receipt_path = qr/('qualification_'+arm+'.json')
        check(sha(receipt_path) == final['receipts'][arm], 'qualification receipt hash '+arm)
        receipt = json.loads(receipt_path.read_text())
        raw = qr/('qualification_'+arm)
        fresh = qrun(qr/'stage', raw, arm)
        check(receipt == fresh, 'qualification raw/source evidence no longer matches '+arm)
    return {'root': str(qr.resolve()), 'pass_sha256': sha(qr/'QUALIFICATION_PASS.json'),
            'source_manifest_sha256': m['qualification_manifest_sha256']}


def validate_head(head, arm):
    check(head.get('schema') == 'e8.head.v1' and head.get('closed') is True and not head.get('failures') and head.get('qualify') is False and head.get('arm') == arm, 'clean head receipt')
    n = head.get('proposals', 0)
    check(type(n) is int and n > 0 and type(head.get('owner_pid')) is int and head['owner_pid'] > 0, 'head coverage/owner')
    check(head.get('primary_head_calls') == 5*n and head.get('legacy_head_calls') == (0 if arm == 'on' else 5*n), 'clean 5-versus10 head census')
    for k in ('qualified_proposals','checked_heads','root_heads','loop_heads','dispatch_ops_checked'):
        check(head.get(k, 0) == 0, 'qualification instrumentation active in timing: '+k)


def config(root, r, arm):
    """Validate actual clean-mode controls and exact qualified module identities."""
    import ast
    root, r = Path(root), Path(r)
    m = stage(root); qm = qstage(root/'qualification-source')
    d = json.loads((r/'docker_inspect.json').read_text())[0]
    env = dict(x.split('=',1) for x in d['Config']['Env'] if '=' in x)
    need = {'E8_ARM': arm, 'E8_QUALIFY': '0', 'FR13_FIX1_SELFCHECK':'0', 'FR13_DRAFTER_SINGLE_LOGITS': '1' if arm=='on' else '0',
            'FR10_ENABLE_TREE_GDN':'1','FR10_ALLOW_LINEAR_FALLBACK':'0','FR10_DECODE_MODE_DEFAULT':'tree_mtp',
            'FR13_TREE_RUNROW_INIT':'1','FR13_FORCE_SPINE_COMMIT':'0','FR13_COMMIT_ARGMAX_GATE':'0',
            'FR13_REPLAY_ROUTE':'1','FR13_EAGER_PACK':'1','FR13_TREE_CONV_FUSED':'1',
            'FR13_ATTN_KV_REMAP':'1','FR13_SLOT_REORDER':'0','FR13_KV_REMAP_SYNCFREE':'1',
            'FR10_METRICS':'0','E7A_CAPTURE_SHIM':'0','E7B_SHIM_PATH':'', 'E1_RECORD':'/logs/e1_events.jsonl',
            'FR13_SFWD_GPU_TIMER':'1','E1_SHIM_PATH':'/e8/frozen/e1/e1_event_recorder_shim.py','E1_RUNTIME_DIR':'/e8/frozen/e1'}
    for k,v in need.items(): check(env.get(k)==v, 'clean actual env '+k)
    for k in ('FR13_FIXED32_MODE','FR13_TAW','FR13_DM_DEPTHSYNC','FR13_STEP_GRAPH','FR13_SAMPLED_REPLAY_BATCHED','FR13_COMMITTER_NATIVE_BATCHED','FR13_COMMITTER_GRAPH','FR13_REPLAY_MULTISTREAM'):
        check(env.get(k,'') in ('','0'), 'unexpected route '+k)
    check(env.get('FR13_DEVICE_MULTIDRAFT','1')=='1', 'committer changed')
    for k in ('FR10_TREE_GDN_CAPTURE_PAYLOAD','FR10_LAYER_HIDDEN_CAPTURE','FR13_DECODE_GDN_CAPTURE','LUMO_MTP_DRAFT_TRACE_FILE','LUMO_TREE_SAMPLER_DEBUG_LOG','LUMO_TREE_PATH_LCP_LOG'):
        check(not env.get(k), 'heavy capture '+k)
    check(not list((r/'logs').glob('*.arm')), 'unexpected arm marker')
    spec=json.loads(env['SPEC_CONFIG']); tree=[(0,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0),(0,1),(0,0,1),(0,0,0,1),(0,0,0,0,1)]
    check(set(spec)=={'method','num_speculative_tokens','speculative_token_tree'} and spec['method']=='qwen3_5_mtp' and spec['num_speculative_tokens']==9 and ast.literal_eval(spec['speculative_token_tree'])==tree, 'Cat10 topology')
    check(d['Config']['Image']==m['image'], 'image')
    cmd=' '.join(d['Config']['Cmd'])
    for token in ["--seed '20260921'","--max-num-seqs '1'","--attention-backend 'TREE_ATTN'",'--enforce-eager','--no-enable-prefix-caching','--no-async-scheduling',"--gpu-memory-utilization '0.6'","--max-model-len '16384'",'vllm serve '+m['model_path'],'--served-model-name '+m['model_name']]:
        check(token in cmd, 'actual command '+token)
    for name,h in qm['unchanged_loaded_modules'].items(): check(sha(r/'loaded_backend'/name)==h, 'unchanged module '+name)
    sh=json.loads((r/'logs/e8_shim.json').read_text())
    check(sh['qualify'] is False and sh['arm']==arm and sh['output_sha256']==qm['eagle_variants'][arm+'_clean']==sha(r/'loaded_backend/eagle.py'), 'actual clean arm')
    model=json.loads((r/'logs/served_model.json').read_text())
    check(model['path']==m['model_path'] and model['checkpoint_identity']==m['checkpoint_identity'], 'weights identity')
    return m


def reduce_cell(root, r, arm):
    root,r=Path(root),Path(r);m=config(root,r,arm)
    from ownership import verify as ownership_verify
    ownership=ownership_verify(r)
    head=json.loads((r/'logs/e8_head_gate.json').read_text());validate_head(head,arm)
    check(json.loads((r/'logs/e8_head_owner.json').read_text()).get('pid')==head['owner_pid'], 'owner binding')
    check(not (r/'logs/fr13_fix1_selfcheck.json').exists(), 'selfcheck output in clean timing')
    check(not (r/'FAILED.json').exists(), 'cell failure marker')
    frozen=json.loads((root/'qualification-source/frozen/e1/frozen_prefixes.json').read_text())['pilot']
    hashes={p['id']:p['prefix_sha256'] for p in frozen};requests=[json.loads(p.read_text()) for p in sorted((r/'cohort').glob('*/capture_request.json'))]
    check(len(requests)==10, 'exact warmup+timed request count')
    amap=json.loads((r/'api_tokens.json').read_text());timed={};warm=[];streams={};slots={}
    for q in requests:
        check(not q.get('error') and q.get('response_id'), 'API failure')
        ph=q['phase'];budget=32 if ph=='warmup' else 128 if ph=='timed' else -1
        check(q['slot'] not in slots, 'duplicate workload slot');slots[q['slot']]=q
        check(q['request'].get('model')==m['model_name'], 'API model')
        check(q['max_tokens']==q['request'].get('max_tokens')==budget and budget>0, 'phase/token budget')
        check(q['seed']==q['request'].get('seed')==20260921 and q['request'].get('temperature')==0 and q['request'].get('return_tokens_as_token_ids') is True, 'API sampling contract')
        check(q['prefix_id'] in hashes and q['prefix_sha256']==q['prompt_sha256']==hashes[q['prefix_id']], 'prefix hash')
        rid=[rid for rid in amap if rid==q['response_id'] or rid.startswith(q['response_id']+'-')]
        check(len(rid)==1 and q['response_logprobs_tokens'] and all(str(x).startswith('token_id:') for x in q['response_logprobs_tokens']), 'direct IDs')
        check([int(x.split(':',1)[1]) for x in q['response_logprobs_tokens']]==amap[rid[0]], 'API map differs from captured direct IDs')
        if ph=='warmup':
            check(q['prefix_id']==frozen[0]['id'], 'warmup prefix');warm.append(rid[0])
        else:
            check(q['prefix_id'] not in timed, 'duplicate timed prefix');timed[q['prefix_id']]=rid[0];streams[q['prefix_id']]=amap[rid[0]]
    check(len(set(warm))==2 and len(set(timed.values()))==8 and not set(warm)&set(timed.values()) and set(timed)==set(hashes), 'workload coverage')
    order=['w0','w1']+['t'+str(i) for i in range(8)]
    check(set(slots)==set(order), 'exact ten workload slots')
    for i,key in enumerate(order):
        q=slots[key]
        check(q['prefix_id']==frozen[0 if i<2 else i-2]['id'] and q['phase']==('warmup' if i<2 else 'timed'), 'frozen workload order')
        check(q['t_end_perf']>=q['t_start_perf'], 'request clock order')
        if i:check(q['t_start_perf']>=slots[order[i-1]]['t_end_perf'], 'B1 sequential phase boundaries')
    j=json.loads((r/'join.json').read_text())
    check(not j.get('invalid') and not j.get('refused') and len(j.get('token_evidence',{}))==10, 'complete API/event join')
    check(set(j['phases']['warmup'])==set(warm) and set(j['phases']['timed'])==set(timed.values()), 'actual phase support')
    events=[json.loads(x) for x in (r/'logs/e1_events.jsonl').read_text().splitlines() if x.strip()]
    used={rid:0 for rid in amap};bound={}
    for e in events:
        if e.get('event')!='output_rows':continue
        for row in e['rows']:
            rid=row['request_id'];n=min(len(row['emitted_ids']), max(0,len(amap[rid])-used[rid]));bound[(e['seq'],rid)]=n;used[rid]+=len(row['emitted_ids'])
    per={pid:{'intervals':0,'wall_s':0.0,'api_bound_tokens':0} for pid in timed};inverse={rid:pid for pid,rid in timed.items()}
    for s in j['steps']:
        check(len(s['rows'])==1 and s['rows'][0]['request_id'] in inverse, 'B1 timed support')
        rid=s['rows'][0]['request_id'];p=per[inverse[rid]];p['intervals']+=1;p['wall_s']+=s['wall_s'];p['api_bound_tokens']+=bound[(s['seq'],rid)]
    tokens=sum(p['api_bound_tokens'] for p in per.values());wall=sum(p['wall_s'] for p in per.values());n=sum(p['intervals'] for p in per.values())
    check(tokens==j['sum_emitted_tokens_api_bound_pure_support'] and abs(wall-j['sum_wall_s_unique_physical_steps'])<1e-9, 'support numerator/denominator')
    floor=n>=8 and all(p['intervals']>=1 for p in per.values())
    ex={}
    for e in j['excluded'].values():
        p=ex.setdefault(e['reason'],{'count':0,'wall_s':0.0});p['count']+=1;p['wall_s']+=e.get('wall_s') or 0
    return {'status':'VALID' if floor else 'INSUFFICIENT_SUPPORT','arm':arm,'rate':tokens/wall if floor and wall>0 else None,
            'api_bound_tokens':tokens,'unique_wall_s':wall,'n_intervals':n,'per_prefix':per,'exclusions':ex,
            'over_cap_diagnostic':j['over_cap_diagnostic'],'streams':streams,'head_census':head,'ownership':ownership}


def evidence_files(r):
    return {str(p.relative_to(r)):sha(p) for p in sorted(r.rglob('*')) if p.is_file() and p.name!='cell_result.json'}


def seal_cell(root, r, cell, result, qualification_sha):
    root,r=Path(root),Path(r)
    check(not (r/'FAILED.json').exists() and not (r/'cell_result.json').exists(),'failed/already sealed cell')
    check(result['arm']==cell['arm'] and result['status'] in ('VALID','INSUFFICIENT_SUPPORT'),'terminal result identity')
    record={'schema':'e8.timing.cell.v1','terminal_seal':True,'cell':cell,'timing_manifest_sha256':sha(root/'timing_manifest.json'),
            'qualification_pass_sha256':qualification_sha,'result':result,'evidence_sha256':evidence_files(r)}
    with (r/'cell_result.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    return record


def sealed_cell(root, r, cell, qualification_sha):
    root,r=Path(root),Path(r)
    check(not (r/'FAILED.json').exists(),'cell marked failed')
    d=json.loads((r/'cell_result.json').read_text())
    check(d.get('schema')=='e8.timing.cell.v1' and d.get('terminal_seal') is True and d.get('cell')==cell,'missing/wrong terminal cell seal')
    check(d['timing_manifest_sha256']==sha(root/'timing_manifest.json') and d['qualification_pass_sha256']==qualification_sha,'terminal source/qualification binding')
    check(d['evidence_sha256']==evidence_files(r),'raw evidence changed after seal')
    check(d['result']==reduce_cell(root,r,cell['arm']),'sealed result differs from raw reconstruction')
    return d['result']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--qualification');a=ap.parse_args()
    result=qualification(a.stage,a.qualification) if a.qualification else {'status':'STAGE_PASS','files':len(stage(a.stage)['files'])}
    print(json.dumps(result))


if __name__=='__main__':main()
