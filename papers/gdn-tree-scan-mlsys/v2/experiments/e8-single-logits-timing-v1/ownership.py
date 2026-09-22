#!/usr/bin/env python3
"""Sample host GPU PIDs against the single labeled container during workload.

Read-only telemetry, not continuous isolation proof. Any failed/ambiguous query,
foreign container, or GPU PID outside docker-top host PIDs invalidates the cell.
"""
import json
from pathlib import Path
import subprocess
import time
from timing_verify import check


def assess(raw, container_id, run_id):
    check(all(x['returncode']==0 for x in raw.values()), 'ownership query failed')
    d=json.loads(raw['inspect']['stdout'])[0]
    check(d['Id']==container_id and d['State']['Running'] and d['Config'].get('Labels',{}).get('lumo.e8.run')==run_id,'owned container identity/state')
    ids=set(raw['containers']['stdout'].split())
    check(ids=={container_id}, 'foreign/missing active container')
    gpu=set(int(x.strip()) for x in raw['gpu']['stdout'].splitlines() if x.strip())
    lines=raw['top']['stdout'].splitlines()
    check(lines and lines[0].strip()=='PID','docker-top host PID header')
    owned=set(int(x.strip()) for x in lines[1:] if x.strip())
    check(gpu and owned and gpu<=owned,'GPU PID missing or outside owned container host PID set')
    return {'gpu_host_pids':sorted(gpu),'owned_host_pids':sorted(owned),'container_id':container_id}


def sample(container, container_id, run_id, path):
    start=time.monotonic();raw={};entry={'start_monotonic':start,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    commands={'inspect':['docker','inspect',container], 'containers':['docker','ps','--no-trunc','-q'],
              'gpu':['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], 'top':['docker','top',container,'-eo','pid']}
    try:
        for key,args in commands.items():
            try:
                r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=10)
                raw[key]={'argv':args,'returncode':r.returncode,'stdout':r.stdout}
            except subprocess.TimeoutExpired as exc:
                raw[key]={'argv':args,'returncode':-1,'stdout':str(exc.stdout),'timeout':True}
        entry.update(status='PASS',identity=assess(raw,container_id,run_id))
    except BaseException as exc:
        entry.update(status='INVALID',error=repr(exc))
        raise
    finally:
        entry.update(raw=raw,end_monotonic=time.monotonic())
        with Path(path).open('a') as f:f.write(json.dumps(entry)+'\n');f.flush()
    return entry


def workload(argv, container, container_id, run_id, run, timeout):
    run=Path(run);telemetry=run/'ownership_samples.jsonl'
    check(not telemetry.exists(),'ownership telemetry exists')
    sample(container,container_id,run_id,telemetry)
    start=time.monotonic()
    with (run/'workload.txt').open('x') as out:
        proc=subprocess.Popen([str(x) for x in argv],stdout=out,stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                check(time.monotonic()-start<timeout,'workload deadline exceeded')
                sample(container,container_id,run_id,telemetry)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
            check(proc.returncode==0,'workload client failed')
            end=time.monotonic()
            check(end-start<=timeout,'workload deadline exceeded')
            sample(container,container_id,run_id,telemetry)
        finally:
            if proc.poll() is None:proc.kill();proc.wait(timeout=10)
    receipt={'status':'PASS','container_id':container_id,'run_id':run_id,'workload_start_monotonic':start,'workload_end_monotonic':end,
             'sampling_interval_seconds':5,'visibility_limit':'occasional host process samples; cannot exclude contention entirely between samples'}
    with (run/'ownership_receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')


def verify(run):
    run=Path(run);d=json.loads((run/'ownership_receipt.json').read_text());rows=[json.loads(x) for x in (run/'ownership_samples.jsonl').read_text().splitlines()]
    check(d['status']=='PASS' and len(rows)>=2 and all(x.get('status')=='PASS' for x in rows),'ownership samples/seal missing or invalid')
    for i,x in enumerate(rows):
        check(x['identity']==assess(x['raw'],d['container_id'],d['run_id']),'ownership raw reconstruction')
        check(x['end_monotonic']>=x['start_monotonic'],'sample time order')
        if i:check(x['start_monotonic']>=rows[i-1]['end_monotonic'],'sample order')
    check(rows[0]['end_monotonic']<=d['workload_start_monotonic']<=d['workload_end_monotonic']<=rows[-1]['start_monotonic'],'ownership samples do not bracket workload')
    actual=json.loads((run/'docker_inspect.json').read_text())[0]
    check(actual['Id']==d['container_id'] and actual['Config'].get('Labels',{}).get('lumo.e8.run')==d['run_id'],'telemetry/container binding')
    return {'samples':len(rows),'gpu_host_pids':sorted({p for x in rows for p in x['identity']['gpu_host_pids']}),
            'sampling_interval_seconds':5,'max_gap_seconds':max((b['start_monotonic']-a['end_monotonic'] for a,b in zip(rows,rows[1:])),default=0),
            'visibility_limit':d['visibility_limit']}
