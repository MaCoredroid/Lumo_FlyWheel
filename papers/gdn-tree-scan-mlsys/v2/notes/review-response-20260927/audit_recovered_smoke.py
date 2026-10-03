"""Independent CPU-only reader for the recovered native root-only R2 smoke."""
import collections
import hashlib
import json
import math
import pathlib
import re
import sys

import numpy as np

p = pathlib.Path(sys.argv[1]).resolve()
sha = lambda data: hashlib.sha256(data).hexdigest()
load = lambda name: json.loads((p / name).read_text())
receipt = load('RUN-RECEIPT.json')
for name, rec in receipt['files_recursive_excluding_object_store'].items():
    raw = (p / name).read_bytes()
    assert len(raw) == rec['bytes'] and sha(raw) == rec['sha256'], name
nonobjects = {str(f.relative_to(p)) for f in p.rglob('*') if f.is_file() and not str(f.relative_to(p)).startswith('q1_ref/objects/')}
assert nonobjects == set(receipt['files_recursive_excluding_object_store']) | {'RUN-RECEIPT.json'}
binding, gate = load('LAUNCH-BINDING.json'), load('GATE-Q1-NATIVE-SMOKE.snapshot.json')
assert {k: v for k, v in binding['expect'].items() if k != 'gate'} == gate['reviewed_hashes']
assert binding['expect']['gate'] == sha((p / 'GATE-Q1-NATIVE-SMOKE.snapshot.json').read_bytes())
assert binding['scope'] == gate['reviewed_scope']
assert binding['run_id'] == gate['approved_run_id'] == p.name
job = load('q1_ref_hooks_job.json')
assert binding['job_sha256'] == sha((p / 'q1_ref_hooks_job.json').read_bytes())
assert binding['engine_config_sha256'] == sha((p / 'engine-config.json').read_bytes())
arm = load('engine-config.json')['arms']['aligned_nonpacked']
argv = load('engine_launch_argv.json')
assert sha(arm['rendered_command_NOT_EXECUTED'].encode()) == binding['rendered_command_sha256']
assert argv['argv'] == arm['docker_argv_without_script'] and argv['script'] == arm['in_container_script']
assert job['repeats'] == 2 and job['expected_requests'] == 2 and len(job['cases']) == 1
assert job['cases'][0]['case_id'] == 'calibration-short_available__c0__root-only'
assert job['cases'][0]['accepted_len'] == 0
assert job['expect_packed_flag'] == '0' and job['process'] == 'A'
boot = load('q1_ref/boot_attestation.aligned_nonpacked.pA.json')
assert boot['problems'] == [] and boot['fatal'] is False
assert boot['layers'] == {'gdn': 48, 'attention': 16}
assert boot['packed_attribute_all_layers_match'] is True and boot['packed_attribute_missing_or_untyped'] == 0
assert boot['attention_impl']['all_layers_fa2'] is True and boot['attention_impl']['layers'] == 16 and boot['attention_impl']['bad'] == []
assert boot['fa2']['installed_so_sha256'] == binding['expect']['fork_binary']
assert boot['fa2']['fa_version_selected'] == 2
assert boot['cache']['mamba_cache_mode'] == 'align' and boot['cache']['mamba_ssm_cache_dtype'] == 'float32'
assert boot['cache']['speculative_config_is_none'] is True

docs = [json.loads(f.read_text()) for f in sorted((p / 'q1_ref/cases').glob('*.json'))]
assert len(docs) == 2 and [d['repeat'] for d in docs] == [0, 1]
assert len({d['obs_id'] for d in docs}) == 2 and len({d['req_id'] for d in docs}) == 2
refs = {}
def add(h, n, dtype):
    assert re.fullmatch('[0-9a-f]{64}', h)
    assert h not in refs or refs[h] == (n, dtype), h
    refs[h] = (n, dtype)

for d in docs:
    body = d.copy()
    record_sha = body.pop('record_sha256')
    assert sha(json.dumps(body, sort_keys=True, default=str).encode()) == record_sha
    assert d['run_id'] == p.name and d['process'] == 'A' and d['arm'] == 'aligned_nonpacked'
    assert d['job_sha256'] == binding['job_sha256']
    assert d['valid'] is True and d['problems'] == [] and d['request_still_active_at_seal'] is False
    assert d['control']['chain_tokens'] == [11352, 25559] and d['control']['chain_positions'] == [13487, 13488]
    assert d['control']['prefix_len'] == 13487
    trace = [t for t in d['consumed_trace'] if 'token' in t]
    assert [(t['token'], t['position'], t['num_computed_before']) for t in trace] == [(11352, 13487, 13487), (25559, 13488, 13488)]
    assert all(t['token_source'] == 'runner.input_ids.gpu (embed path; input_ids arg None)' and t['positions_shape'] == [3, 1] for t in trace)
    assert d['forced_count'] == 3
    for tag, extent in [('o0', 13487), ('o1', 13488)]:
        o = d[tag]
        assert o['obs_id'] == d['obs_id'] and o['tag'] == tag.upper() and o['materialized_tokens'] == extent
        gnames = {int(re.search(r'layers\.(\d+)\.', n)[1]) for n in o['gdn']}
        anames = {int(re.search(r'layers\.(\d+)\.', n)[1]) for n in o['attention']}
        assert gnames == {i for i in range(64) if i % 4 != 3}
        assert anames == {i for i in range(64) if i % 4 == 3}
        assert collections.Counter(g['group'] for g in o['gdn'].values()) == {0: 16, 1: 16, 2: 16}
        conv, ssm, attn = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        for name in sorted(o['gdn']):
            g = o['gdn'][name]
            assert g['state_row'] > 0
            assert g['conv']['shape'] == [3, 10240] and g['conv']['dtype'] == 'torch.bfloat16'
            assert g['ssm']['shape'] == [48, 128, 128] and g['ssm']['dtype'] == 'torch.float32'
            add(g['conv']['sha256'], 3 * 10240 * 2, 'bf16')
            add(g['ssm']['sha256'], 48 * 128 * 128 * 4, 'f32')
            conv.update(bytes.fromhex(g['conv']['sha256']))
            ssm.update(bytes.fromhex(g['ssm']['sha256']))
        for name in sorted(o['attention']):
            a = o['attention'][name]
            assert a['group'] == 3 and a['materialized_tokens'] == extent
            assert a['shape'][0] == 2 and a['shape'][2:] == [64, 4, 256] and a['kernel_block_size'] == 64 and a['dtype'] == 'torch.bfloat16'
            assert len(a['full_blocks']) == extent // 64 == 210
            blocks = a['full_blocks'] + [a['tail']]
            ids = [blk['physical_block'] for blk in blocks]
            assert len(set(ids)) == 211 and all(0 < b < a['shape'][1] for b in ids)
            logical = hashlib.sha256()
            for index, blk in enumerate(blocks):
                assert blk['logical_index'] == index
                ntok = 64 if index < 210 else extent % 64
                if index == 210:
                    assert blk['valid_tokens'] == ntok
                per_object = ntok * 4 * 256 * 2
                assert blk['bytes'] == 2 * per_object
                for kv in ['k', 'v']:
                    add(blk[kv], per_object, 'bf16')
                    logical.update(bytes.fromhex(blk[kv]))
            assert logical.hexdigest() == a['logical_digest']
            attn.update(bytes.fromhex(a['logical_digest']))
        assert conv.hexdigest() == o['conv_all_sha256'] and ssm.hexdigest() == o['ssm_all_sha256'] and attn.hexdigest() == o['attn_kv_all_sha256']
        assert sha((conv.hexdigest() + ssm.hexdigest() + attn.hexdigest()).encode()) == o['logical_digest']
    o2 = d['o2']
    assert o2['obs_id'] == d['obs_id'] and o2['num_computed_before'] == 13488 and o2['vocab'] == 248320
    assert o2['logits_dtype'] == 'torch.bfloat16' and o2['raw']['bytes'] == 248320 * 4
    add(o2['raw']['sha256'], 248320 * 4, 'f32')

objects = p / 'q1_ref/objects'
assert set(f.name for f in objects.iterdir() if f.is_file()) == {h + '.bin' for h in refs}, 'object sync/membership incomplete'
assert len(refs) == receipt['object_store']['objects'] == 6977
assert sum(n for n, dtype in refs.values()) == receipt['object_store']['bytes'] == 1195911168
counts = collections.Counter()
for h, (n, dtype) in refs.items():
    raw = (objects / (h + '.bin')).read_bytes()
    assert len(raw) == n and sha(raw) == h, h
    if dtype == 'bf16':
        values = np.frombuffer(raw, dtype='<u2')
        assert not np.any((values & 0x7f80) == 0x7f80), h
    else:
        assert np.isfinite(np.frombuffer(raw, dtype='<f4')).all(), h
    counts[dtype] += 1

logit_summaries = []
for d in docs:
    o = d['o2']
    values = np.frombuffer((objects / (o['raw']['sha256'] + '.bin')).read_bytes(), dtype='<f4')
    arg = int(np.argmax(values))
    top = float(values[arg])
    second = float(np.max(np.concatenate((values[:arg], values[arg+1:]))))
    ties = int(np.sum(values == top))
    assert o['argmax_smallest_id'] == arg and o['top1'] == top and o['top2'] == second
    assert o['margin'] == top - second and o['exact_tie_count'] == ties and o['all_finite'] is True
    logit_summaries.append(dict(repeat=d['repeat'], vocab=len(values), argmax=arg, top1=top, top2=second, margin=top-second, ties=ties, sha256=o['raw']['sha256']))

for tag in ['o0', 'o1']:
    assert docs[0][tag]['logical_digest'] == docs[1][tag]['logical_digest']
assert docs[0]['o2']['raw']['sha256'] == docs[1]['o2']['raw']['sha256']
driver = [load(str(f.relative_to(p))) for f in sorted((p / 'driver').glob('driver.*.json'))]
assert len(driver) == 2
for index, row in enumerate(driver):
    r = row['receipt']
    assert r['repeat'] == index and r['seal_authenticated'] is True and r['seal_problems'] == []
    assert r['usage']['prompt_tokens'] == r['prompt_tokens_sent'] == 13487 and r['http_error'] is None
    assert r['raw_objects_authenticated'] == {'authenticated':6977, 'bytes':1195911168, 'problems':0}
inspect = load('engine_inspect.json')[0]
assert inspect['Image'] == binding['expect']['image_id'] and inspect['RestartCount'] == 0
assert inspect['State']['Status'] == 'exited' and inspect['State']['Running'] is False and inspect['State']['ExitCode'] == 0
assert receipt['engine']['cleanup_state'] == 'stopped_and_removed' and receipt['status'] == 'COMPLETED_driver_rc=0_cleanup_rc=0'
print(json.dumps(dict(status='PASS_NATIVE_ROOT_ONLY_INSTRUMENTATION_SMOKE', nonobject_files=38, case_seals=2, unique_objects=len(refs), object_bytes=sum(x[0] for x in refs.values()), finite_objects_by_dtype=dict(counts), logit_summaries=logit_summaries, o0_digest=docs[0]['o0']['logical_digest'], o1_digest=docs[0]['o1']['logical_digest'], note='One case, two repeats, one native process. No candidate or broader qualification.'), indent=2))
