#!/usr/bin/env python3
"""Recover named historical cohorts from immutable Git blobs; run no inference.

Earlier routes are explicitly superseded. Task verdicts and synthetic serving
coverage survive; invalidated speed proxies and credentials are not promoted.
"""
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[2]
OUT = PAPER / 'results/history-recovery-20260930'
JULY = '51b7dafdf'
ROOT = 'output/fr13_kvremap_tail6/'
ARMS = {'kvremap_tail6_kvr1': 'Tree tail6',
        'native5_control_kvr1': 'Native MTP-5',
        'native11_control_kvr1': 'Native MTP-11'}
CALIBRATION_COMMITS = {
    'dspark_bench_bs1.jsonl': '6530b1f17d84714e8aaf98b1ea5431a403f1e3ea',
    'dspark_bench_bs8.jsonl': '6530b1f17d84714e8aaf98b1ea5431a403f1e3ea',
    'bench_bs1.jsonl': 'd060db3c0c7d17fcee858a46c5b3c61ba80e6ce2',
    'n2_bench_bs1.jsonl': 'ae9cfc6d6f8b22cc06fdccf684f6aa0a58f8e671'}
bindings = {}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


def read_blob(rev, path):
    commit = git('rev-parse', rev + '^{commit}').decode().strip()
    data = git('show', commit + ':' + path)
    if data.startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise ValueError('Need materialized LFS object: ' + path)
    key = commit + ':' + path
    bindings[key] = {'commit': commit, 'path': path,
                     'git_blob': git('rev-parse', key).decode().strip(),
                     'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return data


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def recover_qualifications():
    def keep(rev, path, label):
        raw = read_blob(rev, path)
        dst = OUT / 'raw/qualifications' / label / Path(path).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(raw)
        return ([json.loads(line) for line in raw.decode().splitlines()]
                if path.endswith('.jsonl') else json.loads(raw))

    b1 = 'results/fr13_fixed32_qrow16_num_splits0_live_pass_20260731T173608Z/'
    gate = keep('d857dad0a', b1 + 'fr13_fa2_qrow16_live_paged_ab.json', 'july31-b1')
    flush = keep('d857dad0a', b1 + 'lifecycle/fixed32_final_flush.json', 'july31-b1')
    if (gate['served_return'] != 'stock captured graph output unchanged' or
            any(gate[k]['raw_byte_mismatches'] != 0 for k in ['output', 'lse'])):
        raise ValueError('Historical B1 shadow component disagrees')
    b4 = 'results/fr13_fixed32_b4_graph_batch_bv8_exact4_lifecycle_rejected_20260731T225552Z/'
    layers = keep('92d705c31', b4 + 'kernel/fr13_fixed32_batch_gdn_byte_ab.jsonl', 'july31-b4')
    verdict = keep('92d705c31', b4 + 'verdict.json', 'july31-b4')
    failed = keep('92d705c31', b4 + 'flush/failed_generation5_ack.json', 'july31-b4')
    surfaces = [c for r in layers for c in r['comparisons']]
    graph_surfaces = [c for r in layers for c in r['graph_comparisons']]
    if len({r['layer_key'] for r in layers}) != 48:
        raise ValueError('Unexpected layer coverage')
    if not all(r['reference_restored_and_served'] for r in layers):
        raise ValueError('Unexpected B4 serving route')
    if not all(c['byte_equal'] and c['differing_bytes'] == 0 and
               not c['shape_or_dtype_mismatch'] for c in surfaces + graph_surfaces):
        raise ValueError('Historical B4 component disagrees')
    if verdict['claims']['campaign_lifecycle_valid'] or failed['status'] == 'ok':
        raise ValueError('Rejection boundary changed')
    v7 = 'results/fr13_fixed32_taw_source_v7_b1_b4_bound_20260805/'
    credentials = [keep('da5cb4b72', v7 + name, 'aug05-v7-invalidated')
                   for name in ['b1_credential.json', 'b4_byte_gate.json']]
    for record in credentials:
        if record['candidate_returned'] or record['probability_mismatches'] or record['product_mismatches']:
            raise ValueError('Unexpected historical source-v7 record')
    supersession = read_blob('b48dfe124', 'results/fr13_taw_widegate_shape_pin_20260815/README.md')
    if b'invalidates every existing v7 PASS bundle' not in supersession:
        raise ValueError('Missing explicit supersession')
    (OUT / 'raw/qualifications/aug05-v7-invalidated/SUPERSESSION.md').write_bytes(supersession)
    current_path = ('papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/cqc10/'
                    'hydra27_fixed32_promoab_Cqc10/fixed32_final_flush.json')
    current = keep('470a589d4', current_path, 'aug24-nvfp4-cqc10')
    june = 'FR13_REPLAY_GPU_GATES_BIND.md'
    raw = read_blob('f4d971c1c', june)
    (OUT / 'raw/qualifications/JUNE-REPLAY-REPORTED.md').write_bytes(raw)
    return {
        'july31_b1_shadow_attention': {
            'era': 'SUPERSEDED_FP8_QROW16_NUM_SPLITS0', 'scope': 'ONE_ATTENTION_LAYER_SHADOW',
            'sequence_length': gate['operands']['seq_lens'][0],
            'output_bytes': gate['output']['bytes'], 'lse_bytes': gate['lse']['bytes'],
            'byte_mismatches': 0, 'candidate_served': False,
            'final_work_census': flush['ack']['counters'], 'current_qualification_transfer': False},
        'july31_b4_shadow_gdn': {
            'era': 'SUPERSEDED_FP8_TAIL6_BV8_BATCHING', 'scope': '48_LAYER_SHADOW_COMPONENT',
            'layer_records': len(layers), 'candidate_surface_comparisons': len(surfaces),
            'graph_baseline_comparisons': len(graph_surfaces), 'byte_mismatches': 0,
            'candidate_served': False, 'campaign_status': verdict['status'],
            'failed_snapshot_is_terminal_census': False, 'current_qualification_transfer': False},
        'aug05_v7_credentials': {
            'era': 'SUPERSEDED_FP8_HYDRA27_K64_SOURCE_V7', 'scope': 'SHADOW_TAW_COMPONENT',
            'b4_batches_recorded': credentials[1]['qualified_batches'],
            'probability_mismatches_recorded': 0, 'product_mismatches_recorded': 0,
            'exhaustive_case_denominator': None, 'candidate_served': False,
            'status': 'EXPLICITLY_INVALIDATED_AUG15', 'current_qualification_transfer': False},
        'june_replay': {
            'era': 'SUPERSEDED_SHARED_STEP_SCAN_REPLAY', 'scope': 'CONTEMPORANEOUS_REPORT_ONLY',
            'reported_byte_cases': 126, 'primary_raw_recovered': False,
            'status': 'INITIAL_LIVE_FAIL_THEN_REPORTED_WIRING_FIX', 'current_qualification_transfer': False},
        'aug24_cqc10_flush': {
            'era': 'CURRENT_DEPLOYED_NVFP4_HYDRA27_NATIVE_COMMITTER',
            'scope': 'RECORDED_TERMINAL_WORK_INTERVALS_NOT_STATE_ORACLE',
            'generation': current['ack']['generation'], 'status': current['ack']['status'],
            'final_work_census': current['ack']['counters'], 'fullmodel_equivalence': False}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = git('ls-tree', '-r', '--name-only', JULY, ROOT).decode().splitlines()
    records, arms = [], []
    task_sets = []
    config_flags = ['FR13_ENABLE_APC', 'FR13_REPLAY_ROUTE',
                    'FR13_ATTN_KV_REMAP', 'FR13_SLOT_REORDER',
                    'FR13_TREE_RUNROW_INIT', 'FR13_DRAFT_SOURCE']
    for arm, label in ARMS.items():
        prefix = ROOT + arm + '/'
        arm_rows = []
        for path in paths:
            if not path.startswith(prefix) or not path.endswith('/eval_report.json'):
                continue
            raw = read_blob(JULY, path)
            evaluation = json.loads(raw)
            task = evaluation['instance_id']
            metadata_path = path.rsplit('/eval/', 1)[0] + '/runner_metadata.json'
            metadata_raw = read_blob(JULY, metadata_path)
            metadata = json.loads(metadata_raw)
            if evaluation != metadata['eval_report'] or task != metadata['instance_id']:
                raise ValueError('Evaluator/runner identity mismatch: ' + task)
            if metadata['dataset_name'] != 'princeton-nlp/SWE-bench_Verified':
                raise ValueError('Unexpected dataset')
            if evaluation.get('dataset_name', metadata['dataset_name']) != metadata['dataset_name']:
                raise ValueError('Evaluator/runner dataset mismatch')
            if evaluation['passed'] != (evaluation['verdict'] == 'resolved'):
                raise ValueError('Inconsistent verdict')
            dst = OUT / 'raw/july22' / arm / task
            dst.mkdir(parents=True, exist_ok=True)
            (dst / 'eval_report.json').write_bytes(raw)
            (dst / 'runner_metadata.json').write_bytes(metadata_raw)
            row = {'arm': arm, 'label': label, 'instance_id': task,
                   'verdict': evaluation['verdict'], 'passed': evaluation['passed'],
                   'failure_mode': evaluation['failure_mode'],
                   'agent_elapsed_s': metadata['agent']['elapsed_s'],
                   'timed_out': metadata['agent']['timed_out'],
                   'patch_bytes': metadata['patch_bytes'],
                   'recorded_model_id': evaluation.get('model_id'),
                   'eval_source_path': path, 'runner_source_path': metadata_path}
            arm_rows.append(row)
        if len(arm_rows) != 16 or len({r['instance_id'] for r in arm_rows}) != 16:
            raise ValueError('Expected sixteen unique task attempts per arm')
        summary = json.loads(read_blob(JULY, prefix + 'swe_out/verified/campaign_summary.json'))
        counts = dict(Counter(r['verdict'] for r in arm_rows))
        if counts != summary['verdict_counts'] or summary['instances_total'] != len(arm_rows):
            raise ValueError('Campaign summary differs from primary reports')
        boot = read_blob(JULY, prefix + 'boot_log_snapshot.txt').decode()
        model = re.search(r"config: model='([^']+)'", boot).group(1)
        quant = re.search(r'quantization=([^,]+)', boot).group(1)
        max_seqs = int(re.search(r"'max_num_seqs': (\d+)", boot).group(1))
        if not model.endswith('qwen3.6-27b-fp8') or quant != 'fp8' or max_seqs != 4:
            raise ValueError('Unexpected historical serving identity')
        env = dict(line.split('=', 1) for line in
                   read_blob(JULY, prefix + 'container_env.txt').decode().splitlines()
                   if '=' in line)
        source = read_blob(JULY, prefix + 'git_head.txt').decode().strip()
        span = (datetime.fromisoformat(summary['ended_at'].replace('Z', '+00:00')) -
                datetime.fromisoformat(summary['started_at'].replace('Z', '+00:00'))).total_seconds()
        arms.append({'arm': arm, 'label': label, 'source_commit_recorded': source,
                     'model_path_recorded': model, 'quantization': quant,
                     'max_num_seqs': max_seqs,
                     'flags': {k: env.get(k) for k in config_flags},
                     'verdict_counts': counts,
                     'failure_mode_counts': dict(Counter(r['failure_mode'] for r in arm_rows)),
                     'attempts': len(arm_rows), 'timeouts': sum(r['timed_out'] for r in arm_rows),
                     'empty_patches': sum(r['patch_bytes'] == 0 for r in arm_rows),
                     'summed_agent_elapsed_s': sum(r['agent_elapsed_s'] for r in arm_rows),
                     'campaign_summary_span_s': span})
        task_sets.append({r['instance_id'] for r in arm_rows})
        records.extend(arm_rows)
    if not all(s == task_sets[0] for s in task_sets):
        raise ValueError('Historical arms did not run the same task IDs')
    by_task = {t: {r['arm']: r for r in records if r['instance_id'] == t}
               for t in sorted(task_sets[0])}
    paired = {}
    tree = next(iter(ARMS))
    for control in list(ARMS)[1:]:
        paired[control] = dict(Counter(
            ('both_resolved' if a['passed'] and b['passed'] else
             'tree_only_resolved' if a['passed'] else
             'control_only_resolved' if b['passed'] else 'neither_resolved')
            for t, rs in by_task.items() for a, b in [(rs[tree], rs[control])]))
    # Retain the metric erratum; never revive the source report's decode proxies.
    erratum = PAPER / 'p0/P0-MEASUREMENT-ERRATUM.md'
    closeout = read_blob(JULY, 'FR13_POSTSNAPFIX3_CLOSEOUT.md')
    (OUT / 'raw/july22/FR13_POSTSNAPFIX3_CLOSEOUT.md').write_bytes(closeout)
    calibration = []
    for name in ['dspark_bench_bs1.jsonl', 'dspark_bench_bs8.jsonl',
                 'bench_bs1.jsonl', 'n2_bench_bs1.jsonl']:
        path = 'results/fr14_nvfp4_port_20260816/sglang_calibration/' + name
        rev = CALIBRATION_COMMITS[name]
        raw = read_blob(rev, path)
        values = [json.loads(line) for line in raw.decode().splitlines() if line]
        if len(values) != 1:
            raise ValueError('Unexpected calibration record count')
        data = values[0]
        rate = data['total_output_tokens'] / data['duration']
        if abs(rate - data['output_throughput']) > 1e-10:
            raise ValueError('Synthetic output-rate reduction mismatch')
        # Explicit safe projection: omit authentication and network settings.
        fields = ['dataset_name', 'max_concurrency', 'random_input_len', 'random_output_len',
                  'completed', 'total_output_tokens', 'duration', 'output_throughput',
                  'concurrency', 'accept_length']
        server_fields = ['model_path', 'dtype', 'kv_cache_dtype', 'attention_backend',
                         'speculative_algorithm', 'speculative_num_steps',
                         'speculative_eagle_topk', 'speculative_num_draft_tokens',
                         'speculative_draft_model_path', 'random_seed']
        projection = {k: data.get(k) for k in fields}
        projection['server_info'] = {k: data['server_info'].get(k) for k in server_fields}
        projection['source_commit'] = git('rev-parse', rev).decode().strip()
        projection['source_path'] = path
        projection['output_rate_recomputed'] = rate
        projection['scope'] = 'SYNTHETIC_SERVING_NOT_AGENT_OR_CURRENT_TREE_COMPARISON'
        write_json(OUT / 'raw/calibration' / (name + '.projection.json'), projection)
        calibration.append(projection)
    qualifications = recover_qualifications()
    audit = {'schema': 'lumotree-git-history-recovery-v1',
             'new_gpu_runs': 0,
             'july22': {'era': 'SUPERSEDED_FP8_TAIL6_NATIVE_MTP_JULY22',
                        'archive_commit': git('rev-parse', JULY).decode().strip(),
                        'common_task_count': len(task_sets[0]), 'attempts': len(records),
                        'arms': arms, 'paired_outcomes': paired, 'records': records,
                        'allowed_claim': 'Recorded task outcomes and agent durations on this named historical cohort',
                        'excluded_claims': ['Current NVFP4 qualification', 'Quality preservation',
                                            'Current native-versus-tree speedup',
                                            'Historical invalidated decode/host-cost proxies'],
                        'measurement_erratum_sha256': hashlib.sha256(erratum.read_bytes()).hexdigest()},
             'synthetic_serving': calibration,
             'qualification_era_register': qualifications,
             'git_input_bindings': sorted(bindings.values(), key=lambda r: (r['commit'], r['path']))}
    write_json(OUT / 'audit.json', audit)
    with (OUT / 'july22-attempts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    lines = [r'\begin{table}[t]',
             r'\caption{Superseded July 22 FP8 cohort: one attempt per arm on the same sixteen Astropy tasks. Counts come from all 48 saved evaluator reports. Summed agent time is accumulated task time under a four-sequence server, not elapsed campaign time or isolated decoding time.}',
             r'\label{tab:historical-fp8}', r'\centering\small',
             r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
             r'Historical arm & Attempts & Resolved & Agent min \\', r'\midrule']
    for a in arms:
        lines.append(f"{a['label']} & {a['attempts']} & {a['verdict_counts']['resolved']} & {a['summed_agent_elapsed_s']/60:.2f} " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    (OUT / 'july22-table.tex').write_text('\n'.join(lines))
    print(json.dumps({'historical_attempts': len(records), 'common_tasks': len(task_sets[0]),
                      'resolved': {a['label']: a['verdict_counts']['resolved'] for a in arms},
                      'git_blobs_bound': len(bindings), 'synthetic_records': len(calibration),
                      'paired_outcomes': paired, 'new_gpu_runs': 0}, indent=2))


if __name__ == '__main__':
    main()
