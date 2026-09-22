#!/usr/bin/env python3
"""Linux shell execution smoke with strict subprocess stubs; never starts Docker.

Runs the exact generated outer launcher, including real stage verification and
model-directory identity plumbing, then its captured bash -lc body. The absent
live qualification gate is explicitly stubbed AFTER real source validation;
this test cannot authorize timing or establish numerical correctness. Only memory
recovery/check heredocs and in-container programs are stubbed. All other Python
invocations are refused unless explicitly allowlisted below.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REAL_PYTHON = sys.executable


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(item):
    with open(os.environ['SMOKE_TRACE'], 'a') as f:
        f.write(json.dumps(item) + '\n')


def stub(role, args):
    if role == 'python3':
        if args == ['-']:
            source = sys.stdin.read()
            allow = json.loads(Path(os.environ['SMOKE_HEREDOCS']).read_text())
            kind = allow.get(hashlib.sha256(source.encode()).hexdigest())
            assert kind in ('memory_recovery', 'memory_check', 'tree_count'), source
            log({'kind': kind, 'stubbed': kind != 'tree_count'})
            if kind == 'tree_count':
                return subprocess.run([REAL_PYTHON, '-'], input=source, text=True).returncode
            return 0
        if len(args)==5 and args[1]=='--stage' and args[3]=='--qualification' and args[0]==str(Path(args[2])/'timing_verify.py'):
            # Shell-plumbing test only: real source validation, explicit fake live
            # receipt admission. Separate CPU controls test the REAL refusal gate.
            log({'kind':'stubbed_live_qualification_gate','argv':args})
            return subprocess.run([REAL_PYTHON,'-B',args[0],'--stage',args[2]]).returncode
        if len(args) == 3 and args[1] == '--stage' and args[0] == str(Path(args[2])/'e8_verify.py'):
            log({'kind': 'real_stage_verification', 'argv': args})
            return subprocess.run([REAL_PYTHON, '-B', *args]).returncode
        expected = [
            ['/workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py'],
            ['/e8/e8_shim.py', '--arm', os.environ['E8_ARM'], '--report', '/logs/e8_shim.json'],
            ['/e8/frozen/e1/e1_event_recorder_shim.py', '--report', '/logs/e1_event_recorder_shim.json'],
        ]
        assert args in expected, ('unapproved Python call', args)
        log({'kind': 'inner_python', 'argv': args})
        return 0
    if role == 'docker':
        log({'kind': 'docker', 'argv': args})
        if args[:1] == ['inspect']:
            assert len(args) == 2 and args[1].startswith('e8-sl-smoke-'), args
            return 1
        assert args[:1] == ['run'], args
        assert args[-2] == '-lc' and args[-4] == 'bash', args[-4:]
        env = dict(os.environ)
        i = 1
        envs, mounts = {}, []
        singles = {'-d'}
        pairs = {'--label', '--name', '--gpus', '--ipc', '--ulimit', '-p', '-v', '-e', '--entrypoint'}
        while i < len(args)-3:
            a = args[i]
            if a in singles or a.startswith('--ipc='):
                i += 1
                continue
            assert a in pairs, ('unexpected Docker argument', a)
            value = args[i+1]
            if a == '-e':
                key, val = value.split('=', 1)
                envs[key] = val
            if a == '-v':
                mounts.append(value)
            i += 2
        assert i == len(args)-3 and args[i] == os.environ['SMOKE_IMAGE'], args[i:]
        env.update(envs)
        env['PATH'] = os.environ['SMOKE_BIN'] + ':/usr/bin:/bin'
        env['BASH_ENV'] = os.environ['SMOKE_BASH_ENV']
        log({'kind': 'container_contract', 'env': envs, 'mounts': mounts, 'image': args[i], 'body': args[-1]})
        r = subprocess.run(['bash', '-lc', args[-1]], env=env)
        assert r.returncode == 0, r.returncode
        print('stub-container-id')
        return 0
    if role == 'vllm':
        log({'kind': 'vllm', 'argv': args})
        return 0
    raise AssertionError(('unapproved stub', role, args))


def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('--output', type=Path, default=HERE/'tests_out/launcher_execution.json')
    a = ap.parse_args()
    assert sys.platform == 'linux', 'Execute on Linux: launcher requires GNU find and Bash 4+.'
    launcher = HERE/'timing_launch.v1.sh'
    source = launcher.read_text()
    # Exact heredoc bytes are pinned by the stage manifest; unexpected Python
    # stdin is refused rather than accidentally running memory recovery.
    chunks = source.split("<<'PY'\n")[1:]
    codes = [x.split('\nPY\n', 1)[0]+'\n' for x in chunks]
    assert len(codes) == 3
    kinds = ['tree_count', 'memory_recovery', 'memory_check']
    assert 'recover_host_memory()' in codes[1] and 'MemFree>=100GiB' in codes[2]
    records = {}
    with tempfile.TemporaryDirectory(prefix='e8-launcher-stub-') as temp:
        t = Path(temp)
        bindir = t/'bin'; bindir.mkdir()
        for name in ('python3', 'docker', 'vllm'):
            p = bindir/name
            p.write_text('#!/bin/sh\nexec '+shlex.quote(REAL_PYTHON)+' -B '+shlex.quote(str(Path(__file__).resolve()))+' --stub '+name+' "$@"\n')
            p.chmod(0o755)
        benv = t/'bash_env'; benv.write_text('export PATH='+shlex.quote(str(bindir)+':/usr/bin:/bin')+'\n')
        allow = t/'heredocs.json'; allow.write_text(json.dumps({hashlib.sha256(c.encode()).hexdigest(): k for c,k in zip(codes,kinds)}))
        model = t/'mock_model'; model.mkdir(); (model/'fake.safetensors').write_bytes(b'smoke-only')
        repo = t/'mock_repo'; repo.mkdir()
        image = 'vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776'
        env = {'PATH': str(bindir)+':/usr/bin:/bin', 'HOME': str(t), 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
               'SMOKE_BIN': str(bindir), 'SMOKE_BASH_ENV': str(benv), 'SMOKE_HEREDOCS': str(allow), 'SMOKE_IMAGE': image,
               'REPO': str(repo), 'SERVED_MODEL_PATH': str(model), 'SERVED_MODEL_NAME': 'qwen3.6-27b',
               'SEED': '20260921', 'E8_QUALIFY': '0', 'E8_TIMING_QUALIFICATION': str(t/'absent_live_qualification'), 'E8_RUN_ID': 'smoke', 'PORT': '9968', 'IMAGE': image,
               'MAX_NUM_SEQS': '1', 'GPU_UTIL': '0.6', 'MAX_MODEL_LEN': '16384', 'ENFORCE_EAGER': '1',
               'VLLM_SYNC_SCHED': '1', 'ATTENTION_BACKEND': 'TREE_ATTN', 'FR13_ENABLE_APC': '0',
               'FR13_TREE_RUNROW_INIT': '1', 'FR13_ATTN_KV_REMAP': '1', 'FR13_SLOT_REORDER': '0', 'FR13_KV_REMAP_SYNCFREE': '1',
               'E1_SHIM_PATH': '/e8/frozen/e1/e1_event_recorder_shim.py', 'E1_RUNTIME_DIR': '/e8/frozen/e1',
               'E1_RECORD': '/logs/e1_events.jsonl', 'FR13_SFWD_GPU_TIMER': '1', 'FR10_TREE_DEPTH_POSITION_LOG': ''}
        for arm in ('on', 'off'):
            trace = t/(arm+'.jsonl')
            env.update(E8_ARM=arm, CONTAINER='e8-sl-smoke-'+arm, LOG_DIR=str(t/('logs_'+arm)), SMOKE_TRACE=str(trace), FR13_DRAFTER_SINGLE_LOGITS='1' if arm=='on' else '0')
            r = subprocess.run(['bash', str(launcher)], env=env, text=True, capture_output=True, timeout=60)
            events = [json.loads(x) for x in trace.read_text().splitlines()]
            assert r.returncode == 0, r.stderr+r.stdout
            assert [x['kind'] for x in events] == ['stubbed_live_qualification_gate', 'real_stage_verification', 'tree_count', 'docker', 'memory_recovery', 'memory_check', 'docker', 'container_contract', 'inner_python', 'inner_python', 'inner_python', 'vllm']
            ce = next(x for x in events if x['kind']=='container_contract')
            for key,val in {'E8_ARM':arm,'E8_QUALIFY':'0','FR13_FIX1_SELFCHECK':'0','FR13_DRAFTER_SINGLE_LOGITS':'1' if arm=='on' else '0','FR13_TREE_RUNROW_INIT':'1','FR13_ATTN_KV_REMAP':'1','FR13_SLOT_REORDER':'0','FR13_KV_REMAP_SYNCFREE':'1','E1_RECORD':'/logs/e1_events.jsonl'}.items():
                assert ce['env'][key] == val, (key,ce['env'].get(key))
            spec = json.loads(ce['env']['SPEC_CONFIG']); assert spec['num_speculative_tokens']==9
            va = events[-1]['argv']; assert va[0]=='serve' and va[1]==str(model)
            for key,val in {'--seed':'20260921','--max-num-seqs':'1','--gpu-memory-utilization':'0.6','--max-model-len':'16384','--attention-backend':'TREE_ATTN'}.items():
                assert va[va.index(key)+1] == val, (key,va)
            assert all(x in va for x in ('--no-async-scheduling','--no-enable-prefix-caching','--enforce-eager'))
            assert json.loads(va[va.index('--speculative-config')+1])==spec
            assert len([x for x in ce['mounts'] if '/frozen/repository/' in x])==6
            records[arm] = {'status':'PASS','events':events,'stdout':r.stdout,'stderr':r.stderr}
    report={'status':'PASS','platform':sys.platform,'launcher_sha256':digest(launcher),'manifest_sha256':digest(HERE/'timing_manifest.json'),'no_real_docker_or_model_execution':True,'records':records}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'PASS','cases':list(records),'live_qualification_gate_stubbed':True,'report':str(a.output),'sha256':digest(a.output)}))


if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[1]=='--stub':
        sys.exit(stub(sys.argv[2],sys.argv[3:]))
    main()
