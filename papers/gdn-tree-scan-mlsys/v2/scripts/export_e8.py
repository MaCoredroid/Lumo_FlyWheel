#!/usr/bin/env python3
"""Export only the closed, exact E8 campaign plus every attempted E8 source/run.

CPU/local packaging only. No SSH, Docker, model import or inference. Check all
terminal/source guards before reading complete payloads or opening an archive.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ORIGINAL = 'experiments/e8-single-logits'
QUAL_STAGE = 'experiments/e8-single-logits-qualification-v2'
TIME_STAGE = 'experiments/e8-single-logits-timing-v1'
FAILED = 'experiments/out-20260922T2119Z-e8-single-logits'
QUAL_RUN = 'experiments/out-20260922T213142Z-e8-single-logits-v2'
TIME_RUN = 'experiments/out-20260922T215254Z-e8-timing'
ORIGINAL_SHA = '45da91477bf238e99f70c2201bc722059427ff9cced399754171a4371c41efcf'
QUAL_SHA = 'bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030'
TIME_SHA = '8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1'
GATE_SHA = '815b348bdb95c455122e7f1232449d4e12dd339edbd6c3c8e2b9a980c6342513'
MEMBER = 'E8-EVIDENCE-MANIFEST.json'
EXCLUDED = {'triton_cache', '__pycache__', '.cache'}
REVIEWS = {
    'p0/monitor/e8-harness-redteam.md': '46e4160b3614918e7b96dd79f8d060d5aa436b7d223b96473df1b82958b1f74b',
    'p0/monitor/e8-qualification-v2-redteam.md': 'f4f7ad5108eeb3e4539952a92f1ce3320d23f7b687bb366d3e023b711808c676',
    'p0/monitor/e8-timing-harness-redteam.md': 'fff590de052d939d9f7ff484de95f0d9f61f013a95b60fca99c4216d0a7446fc',
    'p0/monitor/e8-qualification-results-redteam.md': 'e8e02700cad9337bfa61f1446b3f1b090404f144c8acfe193fbad197291cfca3',
    'p0/monitor/e8-weight-preflight-20260922.json': 'c68aa9d5f111239219dbfc4b9e6f9862e5e8239c758cde47ed4bfb28469a4442',
    'p0/model-weight-hashes.json': 'baa9c888989c486d1a3346547a99d4b20067e99d83b4fb80b1e378ffb445f1ae',
    'notes/optimization-route-audit-2026-09-22.md': 'a667aeb25254714116ba77e1a674ab7d866f5cfc76c03679a70de7903320cc22',
}
E1_BASE = 'experiments/out-20260922T100028Z-e1-18cells'
E1_CELL = E1_BASE+'/cell_05_b1_tree_B1_a1'


def require(ok, reason):
    if not ok: raise ValueError('E8 EXPORT REFUSED: '+reason)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def load(path): return json.loads(Path(path).read_text())


def safe(root, relative):
    root=Path(root).resolve();rel=Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts,'unsafe member '+str(relative))
    p=root/rel
    require(p.resolve().is_relative_to(root),'escaping member '+str(relative))
    require(not any((root/Path(*rel.parts[:i])).is_symlink() for i in range(1,len(rel.parts)+1)),'symlink member '+str(relative))
    return p


def tree_files(root):
    require(root.is_dir(),'missing evidence directory '+str(root))
    files=set()
    for p in root.rglob('*'):
        rel=p.relative_to(root)
        if EXCLUDED.intersection(rel.parts):continue
        require(not p.is_symlink(),'symlink evidence '+str(p))
        if p.is_file():files.add(p)
    return files


def freeze(root, filename, expected):
    require((root/filename).is_file() and digest(root/filename)==expected,'unexpected/missing frozen version '+str(root))
    m=load(root/filename)
    for rel,h in m['files'].items():
        p=safe(root,rel);require(p.is_file() and digest(p)==h,'frozen payload changed '+str(p))
    return m


def timing_closure(root, m):
    require((root/'CAMPAIGN_COMPLETE.json').is_file(),'timing campaign not terminal; copy all closed local evidence first')
    require(not (root/'FAILED.json').exists(),'timing campaign failed; no complete-campaign export')
    end=load(root/'CAMPAIGN_COMPLETE.json')
    require(end.get('status')=='COMPLETE' and end.get('timing_manifest_sha256')==TIME_SHA,'campaign completion identity')
    require((root/'aggregate.json').is_file() and digest(root/'aggregate.json')==end.get('aggregate_sha256'),'missing/changed final aggregate')
    expected={c['name'] for c in m['cells']}
    require({p.name for p in root.glob('cell_*') if p.is_dir()}==expected and len(expected)==6,'missing or additional timing cells')
    out=[]
    for c in m['cells']:
        r=root/c['name'];p=r/'cell_result.json'
        require(p.is_file() and not (r/'FAILED.json').exists(),'missing/failed terminal cell '+c['name'])
        d=load(p)
        require(d.get('terminal_seal') is True and d.get('cell')==c,'preliminary/wrong terminal cell '+c['name'])
        require(d.get('timing_manifest_sha256')==TIME_SHA and d.get('qualification_pass_sha256')==GATE_SHA,'cell source/gate identity '+c['name'])
        require(d['result']['status'] in ('VALID','INSUFFICIENT_SUPPORT'),'unexpected result status')
        actual={str(x.relative_to(r)):digest(x) for x in tree_files(r) if x!=p}
        require(actual==d['evidence_sha256'],'changed/incomplete cell raw evidence '+c['name'])
        out.append({'name':c['name'],'status':d['result']['status'],'terminal_sha256':digest(p)})
    return out


def run(argv):
    r=subprocess.run([str(x) for x in argv],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    require(r.returncode in (0,3),'CPU replay failed '+repr(argv)+'\n'+r.stdout[-2000:])
    return r


def replay(paper):
    """Write temporary CPU outputs only; compare original JSON including rates."""
    tr=paper/TIME_RUN;stage=tr/'stage';qr=paper/QUAL_RUN
    r=run([sys.executable,'-B',stage/'timing_verify.py','--stage',stage,'--qualification',tr/'qualification_evidence'])
    require(r.returncode==0,'qualification admission failed')
    joined=[]
    with tempfile.TemporaryDirectory(prefix='e8-export-replay-') as tmp:
        tmp=Path(tmp)
        runs=[(qr/('qualification_'+a),qr/'stage/frozen/e1/e1_join.py',3) for a in ('on','off')]
        runs += [(tr/c['name'],stage/'qualification-source/frozen/e1/e1_join.py',None) for c in load(stage/'timing_manifest.json')['cells']]
        for i,(raw,join,expected_rc) in enumerate(runs):
            out=tmp/(str(i)+'.json')
            result=run([sys.executable,'-B',join,raw/'logs/e1_events.jsonl','--api-tokens',raw/'api_tokens.json','--manifest',raw/'join_manifest.json','--expect-reqs','1','--json',out])
            require(expected_rc is None or result.returncode==expected_rc,'wrong qualification join rc')
            require(load(out)==load(raw/'join.json'),'raw join does not reproduce '+str(raw))
            require(result.returncode==int((raw/'join.rc').read_text()),'archived join exit-code mismatch')
            joined.append({'root':str(raw.relative_to(paper)),'rc':result.returncode,'join_sha256':digest(raw/'join.json')})
        out=tmp/'aggregate.json'
        result=run([sys.executable,'-B',stage/'aggregate.py','--stage',stage,'--run',tr,'--output',out])
        require(result.returncode==0,'aggregate replay failed')
        old,new=load(tr/'aggregate.json'),load(out)
        # Only this output metadata field changes under extraction. Preserve the
        # archived provenance string; do not rewrite any input or numeric value.
        require(old['qualification']['root'].endswith('/'+TIME_RUN+'/qualification_evidence'),'unexpected original absolute qualification provenance')
        require(new['qualification']['root']==str((tr/'qualification_evidence').resolve()),'wrong relocated qualification root')
        new['qualification']['root']=old['qualification']['root']
        require(old==new,'frozen aggregate does not reproduce after explicit path-only normalization')
    return {'status':'PASS','joins':joined,'aggregate_sha256':digest(tr/'aggregate.json'),
            'normalization':'comparison only: qualification.root absolute prefix; archived bytes are unchanged'}


def validate(paper):
    # Cheap closure first: do not package a live/partial local transfer.
    tr=paper/TIME_RUN
    require((tr/'CAMPAIGN_COMPLETE.json').is_file(),'timing campaign not complete locally')
    tm=freeze(paper/TIME_STAGE,'timing_manifest.json',TIME_SHA)
    for rel,h in [(ORIGINAL,ORIGINAL_SHA),(FAILED+'/stage',ORIGINAL_SHA),(QUAL_STAGE,QUAL_SHA),(QUAL_RUN+'/stage',QUAL_SHA),
                  (TIME_STAGE+'/qualification-source',QUAL_SHA),(TIME_RUN+'/stage/qualification-source',QUAL_SHA),
                  (TIME_RUN+'/qualification_evidence/stage',QUAL_SHA)]:freeze(paper/rel,'manifest.json',h)
    freeze(tr/'stage','timing_manifest.json',TIME_SHA)
    bad=load(paper/FAILED/'FAILED.json')
    require(bad.get('status')=='FAILED_NO_TIMING_NO_RETRY' and ',: command not found' in bad.get('error',''),'original infrastructure failure missing/changed')
    require(not (paper/FAILED/'QUALIFICATION_PASS.json').exists(),'failed original unexpectedly qualified')
    require(not list((paper/FAILED).rglob('capture_request.json')) and not list((paper/FAILED).rglob('e1_events.jsonl')),'original failure contains unexpected inference evidence')
    for qr in (paper/QUAL_RUN,tr/'qualification_evidence'):
        require(digest(qr/'QUALIFICATION_PASS.json')==GATE_SHA and not (qr/'FAILED.json').exists(),'qualification verdict changed')
        q=load(qr/'QUALIFICATION_PASS.json');require(q['arms']==['on','off'] and q['stage_manifest_sha256']==QUAL_SHA,'qualification identity')
        for arm in q['arms']:
            receipt=qr/('qualification_'+arm+'.json');require(digest(receipt)==q['receipts'][arm],'qualification receipt changed')
            actual={str(p.relative_to(qr/('qualification_'+arm))):digest(p) for p in tree_files(qr/('qualification_'+arm))}
            require(actual==load(receipt)['evidence_sha256'],'qualification raw payload changed')
    cells=timing_closure(tr,tm)
    for rel,h in REVIEWS.items():require(digest(safe(paper,rel))==h,'review/provenance version changed '+rel)
    return {'cells':cells,'reproduction':replay(paper)}


def collect(paper, sources):
    files={safe(paper,rel) for rel in REVIEWS}
    files.add(sources)
    for rel,h in load(sources).items():
        p=safe(paper,rel);require(p.is_file() and digest(p)==h,'additional reviewed source changed '+rel);files.add(p)
    for rel in (ORIGINAL,QUAL_STAGE,TIME_STAGE,FAILED,QUAL_RUN,TIME_RUN):files.update(tree_files(paper/rel))
    # Prior E1 bytes used ONLY by shipped CPU regression/source-generation tools.
    for rel in ('scripts/export_e8.py','notes/e8-packaging-plan.md','p0/monitor/e8_parent_qualification_check.py',
                'p0/monitor/e8-on-parent-check.json','p0/monitor/e8-off-parent-check.json'):
        p=safe(paper,rel);require(p.is_file(),'missing replay/report dependency '+rel);files.add(p)
    files.update(tree_files(paper/E1_CELL/'cohort'))
    for rel in ('logs/e1_events.jsonl','e1_join.json','e1_api_tokens.json','loaded_backend/eagle.py'):
        p=safe(paper,E1_CELL+'/'+rel);require(p.is_file(),'missing E1 CPU fixture '+rel);files.add(p)
    require(digest(paper/E1_CELL/'loaded_backend/eagle.py')=='aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7','E8 shim input source differs')
    q=load(paper/QUAL_STAGE/'manifest.json')
    # The generator globs all five original loaded modules; retain the four
    # unchanged modules as well as eagle.py so regeneration preserves their pins.
    for name,h in q['unchanged_loaded_modules'].items():
        original=safe(paper,E1_CELL+'/loaded_backend/'+name)
        require(original.is_file() and digest(original)==h,'original loaded module differs '+name)
        files.add(original)
    for rel,h in q['files'].items():
        if rel.startswith('frozen/repository/'):
            mapped='p0/monitor/e1-source-20260922T1150Z/'+rel.removeprefix('frozen/repository/')
        elif rel.startswith('frozen/e1/'):
            mapped=E1_BASE+'/campaign_snapshot/'+rel.removeprefix('frozen/e1/')
        else:continue
        p=safe(paper,mapped);require(digest(p)==h,'original prepare input differs '+mapped);files.add(p)
    launch=safe(paper,E1_BASE+'/campaign_snapshot/e7a_capture_launch.v7.sh')
    require(digest(launch)=='a5c398ffff43d1d8ff4678e8df3750987cd520e7710d044cb4cbeb3c1bee7f72','base launcher dependency differs');files.add(launch)
    return sorted(files)


def verify_extracted(paper):
    m=load(paper/MEMBER)
    require(m['schema']=='e8.evidence.export.v1','export schema')
    for row in m['files']:
        p=safe(paper,row['path']);require(p.is_file() and p.stat().st_size==row['size'] and digest(p)==row['sha256'],'extracted member mismatch '+row['path'])
    return {'status':'EXTRACTED_MEMBERS_AND_CPU_REPLAY_PASS','files':len(m['files']),'validation':validate(paper)}


def selfcheck():
    cases=[]
    def refused(name,fn):
        try:fn()
        except (ValueError,FileNotFoundError):cases.append(name)
        else:raise AssertionError(name)
    with tempfile.TemporaryDirectory(prefix='e8-export-controls-') as t:
        p=Path(t)
        refused('missing_terminal',lambda:timing_closure(p,{'cells':[]}))
        refused('absolute_member',lambda:safe(p,'/tmp/other'))
        refused('escaping_member',lambda:safe(p,'../other'))
        (p/'link').symlink_to(p/'file');refused('symlink_member',lambda:safe(p,'link'))
        (p/'manifest.json').write_text('{"files":{}}');refused('wrong_version',lambda:freeze(p,'manifest.json','0'*64))
        good=digest(p/'manifest.json');freeze(p,'manifest.json',good);cases.append('positive_frozen_manifest')
        (p/'manifest.json').write_text(json.dumps({'files':{'missing':'0'*64}}));h=digest(p/'manifest.json')
        refused('missing_pinned_payload',lambda:freeze(p,'manifest.json',h))
        r=p/'campaign';r.mkdir();cells=[]
        for i in range(6):
            c={'index':i+1,'name':'cell_'+str(i+1),'arm':'off' if i%2==0 else 'on','block':i//2+1};cells.append(c)
            d=r/c['name'];d.mkdir();(d/'raw.json').write_text('{}')
            rec={'terminal_seal':True,'cell':c,'timing_manifest_sha256':TIME_SHA,'qualification_pass_sha256':GATE_SHA,
                 'result':{'status':'INSUFFICIENT_SUPPORT' if i==5 else 'VALID'},'evidence_sha256':{'raw.json':digest(d/'raw.json')}}
            (d/'cell_result.json').write_text(json.dumps(rec))
        (r/'aggregate.json').write_text('{}');(r/'CAMPAIGN_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','timing_manifest_sha256':TIME_SHA,'aggregate_sha256':digest(r/'aggregate.json')}))
        m={'cells':cells};require(len(timing_closure(r,m))==6,'positive closure');cases.append('six_terminal_cells_including_insufficient_preserved')
        d=r/cells[0]['name'];saved=(d/'cell_result.json').read_bytes();(d/'cell_result.json').unlink()
        refused('missing_cell_seal',lambda:timing_closure(r,m));(d/'cell_result.json').write_bytes(saved)
        rec=load(d/'cell_result.json');rec['terminal_seal']=False;(d/'cell_result.json').write_text(json.dumps(rec))
        refused('preliminary_cell',lambda:timing_closure(r,m));(d/'cell_result.json').write_bytes(saved)
        (d/'raw.json').write_text('changed');refused('changed_raw',lambda:timing_closure(r,m));(d/'raw.json').write_text('{}')
        (r/'FAILED.json').write_text('{}');refused('failed_campaign',lambda:timing_closure(r,m));(r/'FAILED.json').unlink()
        (r/'cell_extra').mkdir();refused('extra_replacement_cell',lambda:timing_closure(r,m));(r/'cell_extra').rmdir()
        (r/'aggregate.json').write_text('changed');refused('changed_aggregate',lambda:timing_closure(r,m))
    return {'status':'PASS','count':len(cases),'cases':cases,'no_archive_or_experiment_created':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--paper',type=Path);ap.add_argument('--sources',type=Path);ap.add_argument('--output',type=Path)
    ap.add_argument('--timing-review',help='v2-relative final actual-timing review, required in the extra sources hash map')
    ap.add_argument('--check-only',action='store_true');ap.add_argument('--verify-extracted',action='store_true');ap.add_argument('--selfcheck',action='store_true');a=ap.parse_args()
    if a.selfcheck:print(json.dumps(selfcheck(),indent=2));return
    require(a.paper is not None,'--paper required');paper=a.paper.resolve()
    if a.verify_extracted:print(json.dumps(verify_extracted(paper),indent=2));return
    require(a.sources is not None,'--sources required: final extra report/source hashes')
    sources=a.sources.resolve();require(sources.is_relative_to(paper) and sources.is_file(),'sources map must exist under paper')
    require(a.timing_review is not None and a.timing_review in load(sources),'final actual-timing review must be explicitly hash-bound in --sources')
    require(safe(paper,a.timing_review).is_file(),'missing final actual-timing review')
    validation=validate(paper);files=collect(paper,sources)
    rows=[{'path':str(p.relative_to(paper)),'size':p.stat().st_size,'sha256':digest(p)} for p in files]
    m={'schema':'e8.evidence.export.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_root':str(paper),
       'scope':'Private E8 B1 single-logits evidence; preserves original pre-container failure, both qualified arms and all six immutable timing cells. Packaging does not extend scientific claims.',
       'roots':{'original_source':ORIGINAL,'failed_infrastructure_attempt':FAILED,'qualification_source':QUAL_STAGE,'qualification_run':QUAL_RUN,'timing_source':TIME_STAGE,'timing_campaign':TIME_RUN},
       'pins':{'original':ORIGINAL_SHA,'qualification':QUAL_SHA,'timing':TIME_SHA,'qualification_pass':GATE_SHA},'validation':validation,'files':rows,
       'final_timing_review':a.timing_review,'additional_sources_map':str(sources.relative_to(paper)),
       'path_mapping':'Extract archive into an empty directory treated as v2. Member paths are v2-relative. Original absolute provenance strings are preserved; CPU aggregate comparison normalizes only qualification.root.',
       'external_runtime_dependencies':['Exact pinned vLLM Docker image','Model weights named/hashed in p0/model-weight-hashes.json; not redistributed','Linux/GPU only for fresh serving; Python3.10+ standard library suffices for raw joins and aggregate'],
       'excluded':['model weights','Docker image',*sorted(EXCLUDED)],'prior_E1_fixture_scope':'CPU regression and source-generation inputs only; not E8 measurement data'}
    if a.check_only:print(json.dumps({'status':'READY_NO_ARCHIVE_CREATED','files':len(rows),'validation':validation},indent=2));return
    require(a.output is not None,'--output required unless --check-only')
    output=a.output.resolve();side=Path(str(output)+'.manifest.json');partial=Path(str(output)+'.partial')
    require(not any(p.exists() for p in (output,side,partial)),'preserve existing exports/partial export')
    require(not any(output.is_relative_to(paper/r) for r in (ORIGINAL,QUAL_STAGE,TIME_STAGE,FAILED,QUAL_RUN,TIME_RUN)),'output cannot be inside evidence/source')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(partial,'w:gz',compresslevel=1) as archive:
        for p,row in zip(files,rows):
            data=p.read_bytes();require(hashlib.sha256(data).hexdigest()==row['sha256'],'file changed during export '+str(p))
            member=tarfile.TarInfo(row['path']);member.size=len(data);member.mode=0o644;archive.addfile(member,io.BytesIO(data))
        data=(json.dumps(m,indent=2)+'\n').encode();member=tarfile.TarInfo(MEMBER);member.size=len(data);archive.addfile(member,io.BytesIO(data))
    with tarfile.open(partial,'r:gz') as archive:
        require(archive.getnames()==[x['path'] for x in rows]+[MEMBER],'archive inventory differs')
        for row in rows:require(hashlib.sha256(archive.extractfile(row['path']).read()).hexdigest()==row['sha256'],'written archive member differs')
    partial.rename(output)
    side.write_text(json.dumps(m|{'archive_sha256':digest(output),'archive_bytes':output.stat().st_size},indent=2)+'\n')
    print(json.dumps({'status':'EXPORTED','files':len(rows),'archive_sha256':digest(output),'archive_bytes':output.stat().st_size}))


if __name__=='__main__':main()
