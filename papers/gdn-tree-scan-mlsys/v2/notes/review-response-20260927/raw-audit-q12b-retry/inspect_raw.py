#!/usr/bin/env python3
"""Read-only stdlib audit of the single approved retry. No tensor arithmetic, Docker, GPU, or writes. JSON on stdout."""
import collections, datetime, hashlib, json, math, pathlib
ROOT = pathlib.Path('/home/mark/lumotree-review-20260927/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/runs/q1.2b/q12b-calibration-retry-20260928T010045Z')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(p.read_text())
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=True).encode()).hexdigest()
def check(v, msg):
    if not v:raise RuntimeError(msg)
def utc(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
F=('rms_C0_C2','maxabs_C0_C2','rms_C1_C2','maxabs_C1_C2','rms_C2','maxabs_C2')
lb=load(ROOT/'LAUNCH-BINDING.json');gate=load(ROOT/'GATE-Q1.2b.snapshot.json');fm=load(ROOT/'fixture_manifest.snapshot.json');obs=load(ROOT/'expected_observations.snapshot.json');pol=load(ROOT/'policy.snapshot.json')
check(lb['run_id']==ROOT.name==gate['approved_run_id'],'run identity');check(gate['approved'] is True and gate['gate']=='Q1.2b','approval');check(lb['scope']=={'repeats':2,'block':'calibration','negatives':1,'processes':['A','B']},'scope')
for key,rel in lb['snapshots'].items():check(sha(ROOT/rel)==lb['expect'][key],f'snapshot {key}')
for key,h in gate['reviewed_hashes'].items():check(lb['expect'][key]==h,f'gate binding {key}')
fids={f['fixture_id'] for f in fm['fixtures'] if f['block']=='calibration'};cases={c['case_id']:c for c in obs['cases'] if c['fixture_id'] in fids};L=fm['geometry']['layers_per_fixture'];neg_expected={f'NEG|{fid}|L{l}|{n}' for fid in fids for n in ('N1_sibling_substitution','N3_ring_swap','N4_stale_metadata','N5_offpath_sentinels') for l in (range(2) if n=='N3_ring_swap' else range(L))}
check(len(cases)==5760 and len(neg_expected)==292,'frozen denominator')
k,u,eta=(pol['constants'][x] for x in ('kappa','u32','eta32'))
def verdict(r):
    check(all(type(r.get(f))==list and len(r[f])==48 for f in F),'metric array shape')
    counts=collections.Counter();rmsfails=maxfails=0
    for h in range(48):
        vals=[r[f][h] for f in F];check(all(isinstance(v,(float,int)) and not isinstance(v,bool) for v in vals),'metric scalar')
        a,b,c,d,e,f=vals
        if r['reference_valid'] is not True or not all(math.isfinite(v) for v in vals[2:]):v='UNCOVERED'
        elif r['metrics_valid'] is not True or not math.isfinite(a) or not math.isfinite(b):v='FAIL'
        else:
            rf=a>k*c+u*e+eta;mf=b>k*d+u*f+eta;rmsfails+=rf;maxfails+=mf;v='FAIL' if rf or mf else 'PASS'
        counts[v]+=1
    result='FAIL' if counts['FAIL'] else 'UNCOVERED' if counts['UNCOVERED'] else 'PASS'
    return result,counts,rmsfails,maxfails
out={'run_id':ROOT.name,'scope':'independent metadata/hash/declared-array audit; no tensor numerical recomputation','binding_sha256':sha(ROOT/'LAUNCH-BINDING.json'),'gate_sha256':sha(ROOT/'GATE-Q1.2b.snapshot.json'),'cases_per_process':len(cases),'head_observations_per_process':sum(c['heads']*len(c['surfaces'])*len(c['depths']) for c in cases.values()),'negative_records_per_process':len(neg_expected),'processes':{}}
meta={};tensor_refs={}
for tag in ('A','B'):
    root=ROOT/f'proc{tag}'
    if not (root/'result.json').exists():out['processes'][tag]={'status':'PENDING'};continue
    result=load(root/'result.json');att=result['attestation'];cid=(ROOT/f'proc{tag}.cid').read_text().strip()
    check(result['characterization_complete'] is True and result['integrity_ok'] is True,'runner completion');check(att['hostname_in_container']==cid[:12] and att['container_name_env']==f'lumotree-review-q12b-{tag}','container identity');check(att['helper_hashes']==lb['helpers'],'helper identities')
    check(sha(root/'raw_inventory.jsonl')==result['raw_inventory']['sha256'],'inventory sha')
    entries=[json.loads(x) for x in (root/'raw_inventory.jsonl').read_text().splitlines()];check(len(entries)==result['raw_inventory']['records']==17572,'inventory census')
    maps={x:{} for x in ('ref','eligibility','metrics','negative')};times=collections.defaultdict(lambda:collections.defaultdict(list));casecounts=collections.defaultdict(collections.Counter);headcounts=collections.defaultdict(collections.Counter);negative=collections.defaultdict(collections.Counter);fail_examples=[]
    for e in entries:
        kind,identity=e['kind'],e['case_id'];check(kind in maps and identity not in maps[kind],'duplicate/kind')
        path=root/e['path'];raw=path.read_bytes();check(hashlib.sha256(raw).hexdigest()==e['file_sha256'] and len(raw)==e['bytes'],'record hash/size');r=json.loads(raw);rsha=r.pop('record_sha256');check(canon(r)==rsha==e['record_sha256'] and r['case_id']==identity,'record identity/seal')
        for tk in ('candidate_tensor','native_tensor'):
            if tk in r:
                t=r[tk];check(r[tk.replace('_tensor','_sha256')]==t['sha256'],'tensor record link');p=ROOT/'tensors'/(t['sha256']+'.bin');check(p.stat().st_size==t['bytes'],'tensor presence/length');tensor_refs[t['sha256']]=t
        times[e['fixture_id']][kind].append(utc(e['utc']))
        if kind in ('metrics','negative'):
            v,counts,rf,mf=verdict(r)
            if kind=='metrics':
                c=cases[identity];check(r['c2_sha256']==c['reference_sha256'] and r['surface']==c['surfaces'][0] and r['depth']==c['depths'][0],'metric manifest link')
                key=c['kind'];casecounts[key][v]+=1;headcounts[key].update(counts)
                if v!='PASS' and len(fail_examples)<6:fail_examples.append({'case_id':identity,'verdict':v,'heads':dict(counts),'rms_failed_heads':rf,'max_failed_heads':mf})
            else:negative[r['negative']][v]+=1
        keep=('native_sha256','candidate_sha256','c2_sha256','execution','native_repeat_sha256s','ref_record_sha256','reference_valid','evidence_sha256','eligible','negative')
        maps[kind][identity]={x:r[x] for x in keep if x in r}
    for kind in ('ref','eligibility','metrics'):check(set(maps[kind])==set(cases),f'exact case product {kind}')
    check(set(maps['negative'])==neg_expected,'exact negative product')
    for identity,c in cases.items():
        r,m=maps['ref'][identity],maps['metrics'][identity];check(r['native_sha256']==m['native_sha256'] and r['c2_sha256']==m['c2_sha256'],'ref/final binding');check(r['native_repeat_sha256s']==[r['native_sha256']]*2,'native repeats')
        if c['kind']=='state':check(m['execution']['accepted_nodes']==c['accepted_nodes'] and m['execution']['path_id']==identity.rsplit('|',1)[1] and type(m['execution']['replays_enqueued'])==int and m['execution']['replays_enqueued']==1,'publication path')
    n5=0
    for identity,r in maps['negative'].items():
        _,fid,li,nt=identity.split('|');target=f'{fid}|{li}|state|n14';check(r['native_sha256']==maps['ref'][target]['native_sha256'] and r['c2_sha256']==cases[target]['reference_sha256'],'negative reference authority')
        if nt=='N5_offpath_sentinels':check(r['candidate_sha256']==maps['metrics'][target]['candidate_sha256'],'N5 clean/poison equality');n5+=1
    fixtures=[]
    for f in result['fixtures']:
        fid=f['fixture_id'];phase=f['phase'];check(utc(phase['R_sealed_utc'])<=utc(phase['C_first_scan_utc']),'reference-first phase');check(max(times[fid]['ref']+times[fid]['eligibility'])<=min(times[fid]['metrics']),'reference-first inventory')
        reps=f['repeat_hashes'];native=f['native_repeats'];check([r['repeat'] for r in reps]==[0,1] and [r['repeat'] for r in native]==[0,1],'repeat product');check(native[0]['state_sha256']==native[1]['state_sha256'] and native[0]['out_sha256']==native[1]['out_sha256'],'native determinism')
        check(reps[0]['scan_node_sha256']==reps[1]['scan_node_sha256'] and reps[0]['publish']==reps[1]['publish'],'candidate determinism')
        pathids={c['case_id'].rsplit('|',1)[1] for c in cases.values() if c['fixture_id']==fid and c['kind']=='state'}
        for rep in reps:
            check(set(rep['publish'])==pathids,'publication path coverage')
            for pid,pub in rep['publish'].items():
                check(type(pub['replays_enqueued'])==int and pub['replays_enqueued']==1 and all(pub[x] is True for x in ('pointer_identity_unchanged','control_rows_intact','staging_neutral_tail')),'publication evidence')
                for l,h in enumerate(pub['state_sha256']):check(h==maps['metrics'][f'{fid}|L{l}|state|{pid}']['candidate_sha256'],'publication raw hash')
        fixtures.append({'fixture_id':fid,'phase':phase,'eligibility':f['eligibility'],'route_stamp_after_first_scan':f['route_stamp_after_first_scan'],'negative_census':f['negatives'],'state_bitwise_native':f['findings']['published_equals_native_chain_bitwise_all']})
    out['processes'][tag]={'status':'COMPLETED_RAW_RECORD_CENSUS_VERIFIED','result_sha256':sha(root/'result.json'),'inventory_sha256':sha(root/'raw_inventory.jsonl'),'record_counts':{x:len(y) for x,y in maps.items()},'container_id':cid,'pid':att['pid'],'utc':att['utc'],'case_verdicts_by_surface':dict(casecounts),'head_verdicts_by_surface':dict(headcounts),'negative_verdicts':dict(negative),'n5_exact_clean_hash_matches':n5,'first_failure_examples':fail_examples,'fixtures':fixtures,'tensor_store_reported':result['tensor_store'],'boot_sequence':att['boot_sequence'],'loaded_sources':att['helper_hashes']}
    meta[tag]=maps
if set(meta)=={'A','B'}:
    check(out['processes']['A']['container_id']!=out['processes']['B']['container_id'],'fresh AB containers')
    for kind in ('metrics','ref','negative'):
        for cid in meta['A'][kind]:
            for key in ('native_sha256','candidate_sha256','c2_sha256'):
                check(meta['A'][kind][cid].get(key)==meta['B'][kind][cid].get(key),f'AB tensor identity {kind} {cid} {key}')
    out['AB_raw_hash_identity']='exact for every retained ordinary/reference/negative tensor identity'
out['tensor_refs_unique_stat_verified']=len(tensor_refs);out['tensor_ref_bytes_unique']=sum(t['bytes'] for t in tensor_refs.values());out['tensor_bytes_rehashed_here']=False
for filename in ('summary.v2.json','RUN-RECEIPT.json'):
    if (ROOT/filename).exists():
        d=load(ROOT/filename)
        if filename=='RUN-RECEIPT.json':d={k:v for k,v in d.items() if k!='files_recursive_excluding_tensor_store'}
        out[filename]={'sha256':sha(ROOT/filename),'data':d}
print(json.dumps(out,indent=2,allow_nan=False))
