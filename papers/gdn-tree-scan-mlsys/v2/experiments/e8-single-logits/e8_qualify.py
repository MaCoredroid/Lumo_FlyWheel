#!/usr/bin/env python3
"""Plan by default. --execute runs two bounded qualification boots, never timing."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
from e8_verify import check, run as verify_run, sha, stage

HERE = Path(__file__).resolve().parent
MODULES = {
    "tree_attn.py": "vllm/v1/attention/backends/tree_attn.py",
    "gpu_model_runner.py": "vllm/v1/worker/gpu_model_runner.py",
    "rejection_sampler.py": "vllm/v1/sample/rejection_sampler.py",
    "gdn_linear_attn.py": "vllm/model_executor/layers/mamba/gdn_linear_attn.py",
    "eagle.py": "vllm/v1/spec_decode/eagle.py",
}


def call(argv, *, env=None, out=None, allowed=(0,), timeout=180):
    try:
        r = subprocess.run([str(x) for x in argv], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if out:
            Path(out).write_text((exc.stdout.decode(errors='replace') if isinstance(exc.stdout, bytes) else (exc.stdout or '')) + '\nE8 DEADLINE EXCEEDED\n')
        raise
    if out:
        Path(out).write_text(r.stdout)
    check(r.returncode in allowed, "command failed: " + repr(argv) + "\n" + r.stdout[-2000:])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=9968)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    m = stage(HERE)
    repo, root = Path(a.repo).resolve(), Path(a.run).resolve()
    check(not root.exists(), "run directory exists; no reuse/retry")
    check(not root.is_relative_to(HERE), "run must be outside staging source")
    print(json.dumps({"status": "PLAN", "qualification_order": m["qualification_order"], "prompts_per_arm": 8,
                      "max_tokens": 32, "batch": 1, "timing_enabled": False, "root": str(root), "manifest_sha256": sha(HERE/'manifest.json')}))
    if not a.execute:
        return
    check(sys.platform == 'linux', "GPU qualification only on the reviewed Linux host")
    check(not call(['docker', 'ps', '-q']).stdout.strip(), "another container is active")
    check(not call(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).stdout.strip(), "another GPU compute process is active")
    root.mkdir(parents=True, exist_ok=False)
    snap = root / 'stage'
    shutil.copytree(HERE, snap, ignore=shutil.ignore_patterns('__pycache__', 'tests_out'))
    stage(snap)
    run_id = root.name
    check(all(c.isalnum() or c in '-_' for c in run_id), "unsafe run name")
    receipts = []
    try:
        for idx, arm in enumerate(m['qualification_order']):
            r = root / ('qualification_' + arm)
            (r / 'logs').mkdir(parents=True)
            (r / 'loaded_backend').mkdir()
            container = 'e8-sl-' + run_id[-38:] + '-' + arm
            check(subprocess.run(['docker', 'inspect', container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0, "container already exists")
            env = {k: v for k, v in os.environ.items() if not k.startswith(('FR', 'LUMO_', 'VLLM_', 'E1_', 'E7', 'E8_', 'CUDA_'))}
            # Every inherited launcher option is overwritten or cleared here.
            env.update(REPO=str(repo), CONTAINER=container, PORT=str(a.port), LOG_DIR=str(r/'logs'),
                       IMAGE=m['image'], SERVED_MODEL_PATH=m['model_path'], SERVED_MODEL_NAME=m['model_name'],
                       MAX_NUM_SEQS='1', GPU_UTIL='0.6', MAX_MODEL_LEN='16384', SEED='20260921',
                       E8_RUN_ID=run_id, E8_ARM=arm, E8_QUALIFY='1', ENFORCE_EAGER='1', VLLM_SYNC_SCHED='1',
                       ATTENTION_BACKEND='TREE_ATTN', BATCH_INVARIANT='0', CUDAGRAPH_MODE='',
                       FR10_ENABLE_TREE_GDN='1', FR10_ALLOW_LINEAR_FALLBACK='0', FR10_DECODE_MODE_DEFAULT='tree_mtp',
                       FR13_DRAFTER_SINGLE_LOGITS='1' if arm == 'on' else '0', FR13_REPLAY_ROUTE='1',
                       FR13_EAGER_PACK='1', FR13_TREE_CONV_FUSED='1', FR13_TREE_RUNROW_INIT='1', FR13_ENABLE_APC='0',
                       FR13_ATTN_KV_REMAP='1', FR13_SLOT_REORDER='0', FR13_KV_REMAP_SYNCFREE='1',
                       FR13_FORCE_SPINE_COMMIT='0', FR13_COMMIT_ARGMAX_GATE='0', FR10_METRICS='0', E7A_CAPTURE_SHIM='0',
                       FR10_TREE_DEPTH_POSITION_LOG='', E7B_SHIM_PATH='',
                       E1_SHIM_PATH='/e8/frozen/e1/e1_event_recorder_shim.py', E1_RUNTIME_DIR='/e8/frozen/e1',
                       E1_RECORD='/logs/e1_events.jsonl', FR13_SFWD_GPU_TIMER='1',
                       TREE='[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]',
                       NUM_SPECULATIVE_TOKENS='9')
            env['SPEC_CONFIG'] = json.dumps({'method': 'qwen3_5_mtp', 'num_speculative_tokens': 9, 'speculative_token_tree': env['TREE']}, separators=(',', ':'))
            (r/'launch_env.json').write_text(json.dumps({k: env[k] for k in env if k not in os.environ or env[k] != os.environ.get(k)}, indent=2)+'\n')
            try:
                call(['bash', snap/'e8_launch.v1.sh'], env=env, out=r/'launcher_stdout.txt', timeout=300)
                call(['docker', 'inspect', container], out=r/'docker_inspect.json')
                deadline = time.monotonic() + 1500
                while True:
                    check(time.monotonic() < deadline, 'health timeout')
                    inspect = json.loads(call(['docker', 'inspect', container]).stdout)[0]
                    check(inspect['State']['Running'], 'server exited before qualification')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health', timeout=2) as h:
                            if h.status == 200:
                                break
                    except Exception:
                        pass
                    time.sleep(5)
                for name, path in MODULES.items():
                    call(['docker', 'cp', container+':/usr/local/lib/python3.12/dist-packages/'+path, r/'loaded_backend'/name])
                for name, expected in m['unchanged_loaded_modules'].items():
                    check(sha(r/'loaded_backend'/name) == expected, 'loaded state/commit source before requests: '+name)
                check(sha(r/'loaded_backend/eagle.py') == m['eagle_variants'][arm+'_qualification'], 'loaded arm before requests')
                call([sys.executable, snap/'frozen/e1/e1_workload.py', f'http://127.0.0.1:{a.port}', snap/'frozen/e1/prefix_pool.json',
                      snap/'frozen/e1/frozen_prefixes.json', r, '--batch', '1', '--phase', 'preflight', '--warmup-tokens', '32', '--seed', '20260921'], out=r/'workload.txt', timeout=600)
            finally:
                insp = subprocess.run(['docker', 'inspect', container], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                if insp.returncode == 0:
                    owned = json.loads(insp.stdout)[0]
                    check(owned['Config'].get('Labels', {}).get('lumo.e8.run') == run_id, 'refusing cleanup of unowned container')
                    call(['docker', 'stop', '-t', '60', container], out=r/'stop.txt')
                    call(['docker', 'logs', container], out=r/'docker_logs.txt')
                    call(['docker', 'inspect', container], out=r/'docker_inspect_final.json')
                    call(['docker', 'rm', container], out=r/'remove.txt')
            call([sys.executable, snap/'frozen/e1/e1_api_tokens_from_capture.v2.py', r, r/'logs/e1_events.jsonl', r/'api_tokens.json', '--report', r/'api_map_report.json'], out=r/'api_map.txt')
            api = json.loads((r/'api_tokens.json').read_text())
            # The copied joiner understands warmup/timed, using ENGINE IDs.
            # Map all qualified preflight requests to its untimed exclusion;
            # never relabel diagnostic traffic as timing support.
            jm = {'phases': {'warmup': sorted(api), 'timed': []}, 'qualification_only': True,
                  'source_phase': 'preflight', 'api_to_engine_rule': 'copied exact-or-unique-prefix API mapper'}
            (r/'join_manifest.json').write_text(json.dumps(jm, indent=2)+'\n')
            j = call([sys.executable, snap/'frozen/e1/e1_join.py', r/'logs/e1_events.jsonl', '--api-tokens', r/'api_tokens.json', '--manifest', r/'join_manifest.json', '--expect-reqs', '1', '--json', r/'join.json'], out=r/'join.txt', allowed=(3,))
            (r/'join.rc').write_text(str(j.returncode)+'\n')
            receipt = verify_run(snap, r, arm)
            (root/('qualification_'+arm+'.json')).write_text(json.dumps(receipt, indent=2)+'\n')
            receipts.append(receipt)
        final = {'status': 'QUALIFICATION_PASS', 'timing_authorized_by_this_driver': False,
                 'stage_manifest_sha256': sha(snap/'manifest.json'), 'arms': [x['arm'] for x in receipts],
                 'receipts': {arm: sha(root/('qualification_'+arm+'.json')) for arm in m['qualification_order']}}
        (root/'QUALIFICATION_PASS.json').write_text(json.dumps(final, indent=2)+'\n')
    except Exception as exc:
        (root/'FAILED.json').write_text(json.dumps({'status': 'FAILED_NO_TIMING_NO_RETRY', 'error': repr(exc)}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
