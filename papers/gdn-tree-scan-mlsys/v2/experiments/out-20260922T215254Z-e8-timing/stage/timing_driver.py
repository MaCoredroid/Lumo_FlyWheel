#!/usr/bin/env python3
"""Plan by default; six immutable cells only after the two-arm qualification gate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

from timing_verify import check, sha, stage, qualification, config, reduce_cell, seal_cell
from aggregate import aggregate
from e8_qualify import call, MODULES
from ownership import workload

HERE = Path(__file__).resolve().parent


def write_new(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def environment(repo, root, r, m, arm, container, port):
    env = {k:v for k,v in os.environ.items() if not k.startswith(('FR','LUMO_','VLLM_','E1_','E7','E8_','CUDA_'))}
    env.update(REPO=str(repo), CONTAINER=container, PORT=str(port), LOG_DIR=str(r/'logs'),
               IMAGE=m['image'], SERVED_MODEL_PATH=m['model_path'], SERVED_MODEL_NAME=m['model_name'],
               MAX_NUM_SEQS='1', GPU_UTIL='0.6', MAX_MODEL_LEN='16384', SEED='20260921',
               E8_RUN_ID=root.name, E8_ARM=arm, E8_QUALIFY='0', E8_TIMING_QUALIFICATION=str(root/'qualification_evidence'),
               ENFORCE_EAGER='1', VLLM_SYNC_SCHED='1', ATTENTION_BACKEND='TREE_ATTN', BATCH_INVARIANT='0', CUDAGRAPH_MODE='',
               FR10_ENABLE_TREE_GDN='1', FR10_ALLOW_LINEAR_FALLBACK='0', FR10_DECODE_MODE_DEFAULT='tree_mtp',
               FR13_DRAFTER_SINGLE_LOGITS='1' if arm=='on' else '0', FR13_REPLAY_ROUTE='1', FR13_EAGER_PACK='1',
               FR13_TREE_CONV_FUSED='1', FR13_TREE_RUNROW_INIT='1', FR13_ENABLE_APC='0',
               FR13_ATTN_KV_REMAP='1', FR13_SLOT_REORDER='0', FR13_KV_REMAP_SYNCFREE='1',
               FR13_FORCE_SPINE_COMMIT='0', FR13_COMMIT_ARGMAX_GATE='0', FR10_METRICS='0', E7A_CAPTURE_SHIM='0',
               FR10_TREE_DEPTH_POSITION_LOG='', E7B_SHIM_PATH='',
               E1_SHIM_PATH='/e8/frozen/e1/e1_event_recorder_shim.py', E1_RUNTIME_DIR='/e8/frozen/e1',
               E1_RECORD='/logs/e1_events.jsonl', FR13_SFWD_GPU_TIMER='1',
               TREE='[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]',
               NUM_SPECULATIVE_TOKENS='9')
    env['SPEC_CONFIG']=json.dumps({'method':'qwen3_5_mtp','num_speculative_tokens':9,'speculative_token_tree':env['TREE']},separators=(',',':'))
    return env


def idle():
    check(not call(['docker','ps','-q']).stdout.strip(), 'another container is active')
    check(not call(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).stdout.strip(), 'another GPU process is active')


def run_cell(repo, root, m, cell, port):
    stage(HERE)
    qualification(HERE, root/'qualification_evidence')
    idle()
    arm=cell['arm'];r=root/cell['name'];r.mkdir(exist_ok=False)
    (r/'logs').mkdir();(r/'loaded_backend').mkdir()
    container='e8-sl-'+root.name[-30:]+'-'+str(cell['index'])+'-'+arm
    check(subprocess.run(['docker','inspect',container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0,'container already exists')
    env=environment(repo,root,r,m,arm,container,port)
    write_new(r/'launch_env.json',{k:env[k] for k in env if k not in os.environ or env[k]!=os.environ.get(k)})
    try:
        try:
            call(['bash',HERE/'timing_launch.v1.sh'],env=env,out=r/'launcher_stdout.txt',timeout=m['deadlines_seconds']['launch'])
            call(['docker','inspect',container],out=r/'docker_inspect.json')
            deadline=time.monotonic()+m['deadlines_seconds']['health']
            while True:
                check(time.monotonic()<deadline,'health timeout')
                d=json.loads(call(['docker','inspect',container]).stdout)[0]
                check(d['State']['Running'],'server exited before measurement')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as h:
                        if h.status==200:break
                except Exception:pass
                time.sleep(5)
            for name,path in MODULES.items():
                call(['docker','cp',container+':/usr/local/lib/python3.12/dist-packages/'+path,r/'loaded_backend'/name])
            # Actual source/config guard before any workload; not just env intent.
            config(HERE,r,arm)
            inspected=json.loads((r/'docker_inspect.json').read_text())[0]
            ports=inspected['NetworkSettings']['Ports'].get('9950/tcp') or []
            check(any(x['HostPort']==str(port) for x in ports),'health port does not belong to owned container')
            src=HERE/'qualification-source/frozen/e1'
            workload([sys.executable,src/'e1_workload.py',f'http://127.0.0.1:{port}',src/'prefix_pool.json',src/'frozen_prefixes.json',r,
                  '--batch','1','--phase','main','--warmup-tokens','32','--timed-tokens','128','--seed','20260921'],
                 container,inspected['Id'],root.name,r,timeout=m['deadlines_seconds']['workload'])
        finally:
            d=subprocess.run(['docker','inspect',container],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
            if d.returncode==0:
                owned=json.loads(d.stdout)[0]
                check(owned['Config'].get('Labels',{}).get('lumo.e8.run')==root.name,'refusing cleanup of unowned container')
                call(['docker','stop','-t','60',container],out=r/'stop.txt')
                call(['docker','logs',container],out=r/'docker_logs.txt')
                call(['docker','inspect',container],out=r/'docker_inspect_final.json')
                call(['docker','rm',container],out=r/'remove.txt')
        src=HERE/'qualification-source/frozen/e1'
        call([sys.executable,src/'e1_api_tokens_from_capture.v2.py',r,r/'logs/e1_events.jsonl',r/'api_tokens.json','--report',r/'api_map_report.json'],out=r/'api_map.txt')
        api=json.loads((r/'api_tokens.json').read_text());phases={'warmup':[],'timed':[]}
        for p in sorted((r/'cohort').glob('*/capture_request.json')):
            q=json.loads(p.read_text());rid=[x for x in api if x==q['response_id'] or x.startswith(q['response_id']+'-')]
            check(len(rid)==1 and q['phase'] in phases,'request-to-engine phase identity')
            phases[q['phase']].append(rid[0])
        write_new(r/'join_manifest.json',{'phases':phases,'source':'all ten captured API requests mapped to engine IDs'})
        j=call([sys.executable,src/'e1_join.py',r/'logs/e1_events.jsonl','--api-tokens',r/'api_tokens.json','--manifest',r/'join_manifest.json','--expect-reqs','1','--json',r/'join.json'],out=r/'join.txt',allowed=(0,3))
        (r/'join.rc').write_text(str(j.returncode)+'\n')
        result=reduce_cell(HERE,r,arm)
        stage(HERE);qualification(HERE,root/'qualification_evidence')
        # No terminal seal until workload, cleanup, exact join, census, source and
        # qualification guards all finish. Insufficient support is retained.
        seal_cell(HERE,r,cell,result,sha(root/'qualification_evidence/QUALIFICATION_PASS.json'))
    except BaseException as exc:
        write_new(r/'FAILED.json',{'status':'FAILED_NO_RETRY','error':repr(exc),'cell':cell})
        raise


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',required=True);ap.add_argument('--run',required=True);ap.add_argument('--qualification')
    ap.add_argument('--port',type=int,default=9969);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--_snapshot-child',action='store_true',help=argparse.SUPPRESS)
    a=ap.parse_args();m=stage(HERE);root=Path(a.run).resolve();repo=Path(a.repo).resolve()
    check(all(c.isalnum() or c in '-_' for c in root.name),'unsafe run name')
    if not a._snapshot_child:
        check(not root.exists(),'run directory exists; no retry/resume/overwrite')
        check(not root.is_relative_to(HERE),'run must be outside stage')
        print(json.dumps({'status':'PLAN','cells':m['cells'],'qualification_required':m['qualification_manifest_sha256'],'timing_manifest_sha256':sha(HERE/'timing_manifest.json')}),flush=True)
        if not a.execute:return
        check(sys.platform=='linux' and a.qualification,'Linux plus completed qualification required')
        qr=Path(a.qualification).resolve();qualification(HERE,qr);idle()
        root.mkdir(parents=True,exist_ok=False)
        shutil.copytree(HERE,root/'stage',ignore=shutil.ignore_patterns('__pycache__','tests_out'))
        shutil.copytree(qr,root/'qualification_evidence',ignore=shutil.ignore_patterns('__pycache__','tests_out'))
        stage(root/'stage');qualification(root/'stage',root/'qualification_evidence')
        # The campaign itself executes only the immutable copied driver/modules.
        os.execv(sys.executable,[sys.executable,'-B',str(root/'stage/timing_driver.py'),'--repo',str(repo),'--run',str(root),'--port',str(a.port),'--execute','--_snapshot-child'])
    check(a.execute and sys.platform=='linux' and HERE==root/'stage','invalid internal snapshot invocation')
    check(not any((root/c['name']).exists() for c in m['cells']),'existing cell refuses continuation')
    q=qualification(HERE,root/'qualification_evidence')
    write_new(root/'CAMPAIGN_STARTED.json',{'schema':'e8.timing.start.v1','timing_manifest_sha256':sha(HERE/'timing_manifest.json'),'qualification':q,'cells':m['cells']})
    try:
        for cell in m['cells']:run_cell(repo,root,m,cell,a.port)
        result=aggregate(HERE,root)
        write_new(root/'aggregate.json',result)
        write_new(root/'CAMPAIGN_COMPLETE.json',{'status':'COMPLETE','aggregate_sha256':sha(root/'aggregate.json'),'timing_manifest_sha256':sha(HERE/'timing_manifest.json')})
    except BaseException as exc:
        write_new(root/'FAILED.json',{'status':'FAILED_NO_RETRY_OR_REPLACEMENT','error':repr(exc)})
        raise


if __name__=='__main__':main()
