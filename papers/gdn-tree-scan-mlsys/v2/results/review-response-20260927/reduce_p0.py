#!/usr/bin/env python3
"""Read-only P0 reduction; emits only into this directory. No inference/network.

Replays the previous source-bound rate reducers, then adds task-aligned raw
acceptance and phase counters. Never substitutes accepted tokens for API tokens.
"""
from __future__ import annotations
import csv
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
OLD = HERE.parent / 'agent-workload'
sys.path.insert(0, str(OLD))
from shared_rate_reduce import Audit, rates, close, MODEL

PAIR = ('astropy__astropy-12907', 'astropy__astropy-13033')
SHARED = ('Sr12', 'Cqc16', 'Cqc15', 'SGLang_earlier', 'SGLang_later')
MARKERS = {'Sr12': '/Sr12/', 'Cqc16': '/Cqc16/', 'Cqc15': '/Cqc15/',
           'Cqc10': '/cqc10/', 'SGLang_earlier': '/sglang-step2-20260924/',
           'SGLang_later': '/sglang16_evidence/'}
PHASES = {
    'target_forward_gpu': ('fr13_decode_forward_gpu_seconds_total', 'fr13_decode_forward_gpu_steps_total'),
    'drafter_gpu': ('fr13_drafter_gpu_seconds_total', 'fr13_drafter_gpu_spans_total'),
    'committer_dispatch_gpu': ('fr13_committer_gpu_seconds_total', 'fr13_committer_gpu_spans_total'),
    'admitted_step_wall': ('fr13_decode_step_wall_seconds_total', 'fr13_decode_step_wall_steps_total'),
}
ENV_KEYS = ('LUMO_PROXY_FORCE_TEMPERATURE', 'LUMO_PROXY_FORCE_TOP_P',
            'LUMO_PROXY_FORCE_TOP_K', 'LUMO_PROXY_FORCE_MIN_P',
            'LUMO_PROXY_FORCE_PRESENCE_PENALTY', 'LUMO_PROXY_MAX_OUTPUT_TOKENS',
            'LUMO_PROXY_AUTO_CONTINUE')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def write_csv(name, rows):
    with (HERE/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(snapshot, name, *, optional=False, position=None):
    found = []
    for (key, label_tuple), value in snapshot.items():
        if key != name:
            continue
        labels = dict(label_tuple)
        if 'model_name' in labels and labels['model_name'] != MODEL:
            raise ValueError('wrong model for ' + name)
        if 'engine' in labels and labels['engine'] != '0':
            raise ValueError('wrong engine for ' + name)
        if position is not None and labels.get('position') != str(position):
            continue
        if not math.isfinite(value) or value < 0:
            raise ValueError('invalid counter ' + name)
        found.append(value)
    if not found and optional:
        return None
    if len(found) != 1:
        raise ValueError(f'expected one metric series {name}: {len(found)}')
    return found[0]


def delta(pre, post, name, **kw):
    a, b = metric(pre, name, **kw), metric(post, name, **kw)
    if a is None and b is None:
        return None
    if a is None or b is None:
        raise ValueError('one-sided counter presence: ' + name)
    if b < a:
        raise ValueError('counter reset: ' + name)
    return b-a


def validate_acceptance(drafts, draft_tokens, accepted, positions):
    if len(positions) != 31:
        raise ValueError('expected all 31 physical acceptance positions')
    for x in (drafts, draft_tokens, accepted, *positions):
        if not math.isfinite(x) or x < 0 or x != round(x):
            raise ValueError('invalid nonnegative integer count')
    if drafts <= 0 or draft_tokens != 31*drafts:
        raise ValueError('draft population is not exactly 31 physical slots/event')
    if positions[0] > drafts or any(a < b for a, b in zip(positions, positions[1:])):
        raise ValueError('acceptance survival counts are not monotone within event count')
    if sum(positions) != accepted:
        raise ValueError('position sum differs from accepted-token scalar')
    exact = [drafts-positions[0]] + [a-b for a,b in zip(positions, positions[1:])] + [positions[-1]]
    if sum(exact) != drafts or sum(i*x for i,x in enumerate(exact)) != accepted:
        raise ValueError('acceptance histogram conservation failed')
    return exact


def tree_metrics(audit, root, meta):
    pre = audit.snapshot(root/'vllm_metrics_pre.txt')
    post = audit.snapshot(root/'vllm_metrics_post.txt')
    values = {k: delta(pre, post, 'vllm:spec_decode_num_'+k+'_total')
              for k in ('drafts', 'draft_tokens', 'accepted_tokens')}
    positions = []
    for snapshot in (pre, post):
        labels = [dict(l).get('position') for (k,l) in snapshot
                  if k == 'vllm:spec_decode_num_accepted_tokens_per_pos_total']
        if set(labels) != {str(i) for i in range(31)} or len(labels) != 31:
            raise ValueError('wrong raw acceptance position labels')
    for i in range(31):
        positions.append(delta(pre,post,'vllm:spec_decode_num_accepted_tokens_per_pos_total',position=i))
    exact = validate_acceptance(values['drafts'], values['draft_tokens'], values['accepted_tokens'], positions)
    boundary = meta.get('fixed32_task_boundary', {})
    if boundary.get('mode') != 'hydra27_fixed32':
        raise ValueError('task boundary does not bind hydra27_fixed32')
    for when in ('pre','post'):
        ack = boundary[when]
        if ack['status'] != 'ok':
            raise ValueError('failed phase-counter flush')
        for key in ('sfwd_pending','dfwd_pending','cfwd_pending'):
            if ack['counters'][key] != 0:
                raise ValueError('pending phase samples at task boundary')
    phases = {}
    for phase,(seconds_key,count_key) in PHASES.items():
        s,n = (delta(pre,post,'vllm:'+key, optional=True) for key in (seconds_key,count_key))
        if s is None and n is None:
            phases[phase] = {'available':False}
        elif s is None or n is None or n <= 0:
            raise ValueError('incomplete phase numerator/denominator')
        else:
            phases[phase] = {'available':True, 'seconds':s, 'spans':n,
                             'ms_per_own_span':1000*s/n,
                             'span_count_equals_spec_events':n == values['drafts']}
    wall = {k:delta(pre,post,'vllm:fr13_decode_step_wall_'+k+'_total',optional=True)
            for k in ('attempts','rejected','drafts')}
    expected = boundary['forward_step_interval']['expected_complete_events']
    close(expected, phases['target_forward_gpu']['spans'], 'flushed boundary vs forward counter')
    close(expected, delta(pre,post,'vllm:fr13_fixed32_complete_work_census_events_total'), 'census population')
    return {**values, 'physical_draft_slots_per_event':31,
            'logical_candidate_count_from_bound_mode':27,
            'output_side_accepted_per_spec_event': values['accepted_tokens']/values['drafts'],
            'output_side_accepted_per_physical_slot': values['accepted_tokens']/values['draft_tokens'],
            'survival_count_by_zero_based_output_position':positions,
            'exact_output_side_accepted_count_histogram':exact,
            'committed_path_or_branch_histogram':None,
            'phases':phases,'wall_population':wall,
            'boundary_pending_samples_zero':True}


def aggregate(rows):
    fields = ('output_tokens','completed_requests','request_latency_sum_s','ttft_sum_s','agent_elapsed_s')
    result = {f:sum(t[f] for t in rows) for f in fields}
    result.update(rates(result))
    result.update(task_count=len(rows), resolved=sum(r['eval_verdict']=='resolved' for r in rows),
                  empty_patches=sum(r['patch_bytes']==0 for r in rows),
                  completed=sum(r['completed'] for r in rows))
    if rows[0]['engine'] == 'patched vLLM':
        counters = {k:sum(r['speculative'][k] for r in rows) for k in ('drafts','draft_tokens','accepted_tokens')}
        positions = [sum(r['speculative']['survival_count_by_zero_based_output_position'][i] for r in rows) for i in range(31)]
        counters['exact_output_side_accepted_count_histogram'] = validate_acceptance(counters['drafts'],counters['draft_tokens'],counters['accepted_tokens'],positions)
        counters['survival_count_by_zero_based_output_position'] = positions
        counters['output_side_accepted_per_spec_event'] = counters['accepted_tokens']/counters['drafts']
        counters['output_side_accepted_per_physical_slot'] = counters['accepted_tokens']/counters['draft_tokens']
        counters['phases'] = {}
        for phase in PHASES:
            spans = [r['speculative']['phases'][phase] for r in rows]
            if all(s['available'] for s in spans):
                seconds, n = sum(s['seconds'] for s in spans), sum(s['spans'] for s in spans)
                counters['phases'][phase] = dict(seconds=seconds,spans=n,ms_per_own_span=1000*seconds/n,
                    span_count_equals_spec_events=n==counters['drafts'])
        result['speculative'] = counters
    return result


def main():
    audit = Audit(REPO)
    audit.read(HERE.parents[1]/'paper.config.yaml')
    replay = subprocess.check_output([sys.executable,str(OLD/'patch_producing_rate_reduce.py')])
    saved = audit.read(OLD/'patch-producing-rate-audit.json').encode()
    if replay != saved:
        raise ValueError('existing source-bound reducer replay differs')
    old = json.loads(replay)
    comp = json.loads(audit.read(OLD/'competitive-rate-audit.json'))
    # Bind every input, including existing secondary projections, before use.
    for path,digest in {**old['input_sha256'],**old['configuration_and_behavior_projections']}.items():
        if hashlib.sha256(audit.read(REPO/path).encode()).hexdigest() != digest:
            raise ValueError('input hash mismatch '+path)
    config_path = OLD/'raw/shared-tasks-20260924/audit-projections/nvfp4-b1-cohort-config-behavior-20260924.json'
    configs = json.loads(audit.read(config_path))
    projections = {r['tag']:r for r in configs['runs']}
    behavior = {(r['tag'],t['task']):t for r in configs['runs'] for t in r['tasks']}
    cq_behavior = json.loads(audit.read(OLD/'cqc10-behavior-audit.json'))
    behavior.update({('Cqc10',t['task']):t for t in cq_behavior['tasks']})
    sg_settings = json.loads(audit.read(OLD/'sglang-settings-projection.json'))
    sg_earlier = json.loads(audit.read(OLD/'sglang-step2-config-projection.json'))
    audit.read(OLD/'sglang-step2-behavior-projection.json')
    source_review = {}
    # Current source documents the interpretation; recorded git objects bind the
    # historical patcher review without asserting every loaded byte was attested.
    for rel in ('src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py',
                'scripts/fr13_fixed32_contract.py',
                'scripts/fr10_phase4_patch_vllm_tree_gdn.py'):
        audit.read(REPO/rel)
    semantics_root=HERE/'source-semantics/current-deployment-image'
    semantics_manifest=json.loads(audit.read(semantics_root/'image-identity.json'))
    for entry in semantics_manifest['files']:
        source=semantics_root/entry['path']
        if hashlib.sha256(audit.read(source).encode()).hexdigest()!=entry['sha256']:
            raise ValueError('extracted source hash mismatch')
    for tag,p in projections.items():
        commit=p['git_head']['value']; rel='scripts/fr10_phase4_patch_vllm_tree_gdn.py'
        blob=subprocess.check_output(['git','show',commit+':'+rel],cwd=REPO)
        tree=ast.parse(blob); lines=blob.decode().splitlines()
        functions=[]
        for f in tree.body:
            if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)):
                text='\n'.join(lines[f.lineno-1:f.end_lineno])
                if 'SCHEDULER_PATH.write_text' in text:
                    functions.append({'function':f.name,'lines':[f.lineno,f.end_lineno],
                        'sha256':hashlib.sha256(text.encode()).hexdigest()})
        source_review[tag]={'git_commit':commit,'path':rel,'blob_sha256':hashlib.sha256(blob).hexdigest(),
            'scheduler_mutating_functions':functions,
            'limit':'Historical git object, not an attestation of the installed scheduler after every launch patch.'}
    supplemental=[]
    early_summary_path=REPO/'results/fr14_nvfp4_port_20260816/ablation_a_step2.json'
    early_summary=json.loads(audit.read(early_summary_path))
    for short,outcome in (('13236','resolved in contemporaneous campaign summary'),('13398','stopped in flight at GPU budget')):
        supplemental.append({'arm':'SGLang_earlier','task':'astropy__astropy-'+short,
            'status':outcome,'source':str(early_summary_path.relative_to(REPO)),
            'task_bracket_pooled_rate':None,'reason':'No complete per-task pre/post metrics in the recovered shared-pair bundle; old console average not substituted.'})
    for run,short in (('run1','13033'),('r1','13236')):
        trace=REPO/'results/fr14_nvfp4_port_20260816/sglang16_evidence'/run/'swe_out/verified/per_task'/('astropy__astropy-'+short)/'qwen_trace.jsonl'
        audit.read(trace)
        supplemental.append({'arm':'SGLang_later','task':'astropy__astropy-'+short,'run':run,
            'status':'interrupted trace; missing terminal metadata/evaluation/post bracket',
            'source':str(trace.relative_to(REPO)),'task_bracket_pooled_rate':None,
            'reason':'Distinct partial attempt retained; never counted as a completed verdict or silently combined with later retry.'})
    bracket_path=HERE.parents[1]/'notes/rate-candidates-2026-09-24/nvfp4-b1-cohort-brackets-20260924.json'
    bracket_projection=json.loads(audit.read(bracket_path))
    projected_tasks={(r['tag'],t['task']):t for r in bracket_projection for a in r['arms'] for t in a['tasks']}
    for tag,short in (('Cqc16','13236'),('Cqc15','13398')):
        tid='astropy__astropy-'+short; b=behavior[tag,tid];p=projected_tasks[tag,tid]
        if p['runner_metadata.json'] is not None or p['eval/eval_report.json'] is not None:
            raise ValueError('supplemental tree status changed; re-audit its classification')
        supplemental.append({'arm':tag,'task':tid,
            'status':'degeneration heuristic fired; no terminal agent metadata/evaluation' if tag=='Cqc16' else 'incomplete evaluation; one request running at final bracket',
            'source':str(config_path.relative_to(REPO)),
            'metric_projection_source':str(bracket_path.relative_to(REPO)),
            'metric_boundary_projection':p['metrics'],
            'original_source_bindings':p['inputs'],
            'trace':b['trace'],'recorded_trace_diagnostic':{k:b[k] for k in ('visible_chars','thinking_chars','tool_calls','malformed_tool_calls','degeneration_flag')},
            'task_bracket_pooled_rate':None,
            'reason':'Engine bracket is idle/population-aligned but task has no terminal metadata or verdict; not a completed-agent outcome.' if tag=='Cqc16' else 'Final running-request gauge is nonzero; token and completion/TTFT populations differ. No completed-request rate can be inferred from global generation delta.'})
    lineage=[]
    for (tag,tid),p in projected_tasks.items():
        if tag not in ('Cp1','Cqc12','Cqc12_unharvested'):continue
        lineage.append({'arm':tag,'task':tid,'source':str(bracket_path.relative_to(REPO)),
            'original_source_bindings':p['inputs'],'runner_metadata_projection':p['runner_metadata.json'],
            'evaluation_projection':p['eval/eval_report.json'],'pooled_decode_rate':None,
            'scope':'Earlier development lineage retained as provenance only; not a current-build performance claim or a pooled campaign denominator.'})
    audit.read(OLD.parents[1]/'notes/sglang-swe-recovered-evidence-2026-09-23.md')
    cqroot = OLD/'raw/cqc10/hydra27_fixed32_promoab_Cqc10'
    cq_att = json.loads(audit.read(cqroot/'logs/fr13_fixed32_runtime_attestation.json'))
    cq_image=json.loads(audit.read(cqroot/'fixed32_container_identity.json'))
    if semantics_manifest['image_id']!=cq_image['image_id'] or semantics_manifest['configured_image']!=cq_image['configured_image']:
        raise ValueError('metric source image does not match recorded current-route base image')
    engine_configs = {}
    for name in ('Sr12','Cqc16','Cqc15','Cqc10'):
        if name == 'Cqc10':
            commit = audit.read(cqroot/'git_head.txt').strip()
            att = cq_att
            env = sg_settings['cqc10_proxy_env_allowlist']
            source = str((cqroot/'logs/fr13_fixed32_runtime_attestation.json').relative_to(REPO))
        else:
            p = projections[name]
            commit = p['git_head']['value']; att=p['runtime_attestation']['data']
            env = p['config']['offload_proxy_env.txt'];source=str(config_path.relative_to(REPO))
        engine_configs[name] = {'engine':'patched vLLM', 'vllm_version':att['vllm']['version'],
            'repository_commit_recorded':commit,'runtime_attestation_canonical_sha256':att['overall_canonical_sha256'],
            'attention':'patched FA2 GQA-pair split-K4',
            'fa2_binary_sha256':att['forked_fa2']['destination']['sha256'],
            'decode_route':'Hydra27 logical drafts / 31 physical drafts plus root / fixed32',
            'sampling_requested':{k:env[k] for k in ENV_KEYS if k in env},
            'source':source, 'model_display_name':MODEL,
            'weight_tokenizer_identity_status':'not established by display name or runtime attestation; see gap register',
            'agent_task_concurrency':1,'max_num_seqs':1}
    for name in ('SGLang_earlier','SGLang_later'):
        settings=sg_settings['runs'][0]['launcher_argv_settings']
        env=sg_earlier['proxy_settings'] if name=='SGLang_earlier' else sg_settings['runs'][0]['proxy_env_allowlist']
        engine_configs[name]={'engine':'SGLang','attention':'FlashInfer',
            'model_path_recorded':settings['model-path'],'model_display_name':MODEL,
            'speculative_algorithm':'EAGLE','speculative_steps':3,'speculative_topk':1,'speculative_draft_tokens':4,
            'sampling_requested':{k:env[k] for k in ENV_KEYS if k in env},
            'agent_task_concurrency':1,
            'source':str((OLD/('sglang-step2-config-projection.json' if name=='SGLang_earlier' else 'sglang-settings-projection.json')).relative_to(REPO)),
            'historical_loaded_binary_identity':'not attested per historical task; retained-image source is contextual',
            'weight_tokenizer_identity_status':'not established; as-shipped model path differs from conversion lineage'}
    engine_configs['Cqc10']['historical_container_identity']=cq_image
    attempts=[]
    # Sr12 shared-pair is a SUBSET, not another pair of attempts.
    for name in (*SHARED,'Cqc10'):
        source_arm = comp['arms']['Sr12_full_four_tasks' if name=='Sr12' else name]
        for oldrow in source_arm['tasks']:
            paths=[REPO/p for p in comp['input_sha256'] if MARKERS[name] in p and p.endswith(oldrow['task']+'/runner_metadata.json')]
            if len(paths)!=1:raise ValueError('ambiguous metadata')
            root=paths[0].parent
            meta=json.loads(audit.read(paths[0])); engine='sglang' if name.startswith('SGLang') else 'tree'
            row=audit.task(root,engine)
            for key in ('output_tokens','completed_requests','request_latency_sum_s','ttft_sum_s','agent_elapsed_s','pooled_post_first_token_tokens_per_s'):
                close(row[key],oldrow[key],'prior reduction '+key)
            agent=meta['agent']
            row.update(arm=name,engine=engine_configs[name]['engine'],
                attempt_id=name+':'+row['task'],source_metadata=str(paths[0].relative_to(REPO)),
                started_at=meta.get('started_at'),ended_at=meta.get('ended_at'),
                patch_bytes=meta['patch_bytes'],exit_code=agent['exit_code'],timed_out=agent['timed_out'],
                budget_capped=agent.get('budget_capped',False),agent_wall_budget_s=agent.get('campaign_budget_s'),
                completed=agent['exit_code']==0 and not agent['timed_out'] and not agent.get('budget_capped',False) and bool(meta.get('ended_at')),
                response_cap=int(engine_configs[name]['sampling_requested']['LUMO_PROXY_MAX_OUTPUT_TOKENS']))
            b=behavior.get((name,row['task']))
            row['recorded_trace_diagnostic']=None if b is None else {k:b[k] for k in ('flag','degeneration_flag','malformed_tool_calls','tool_calls') if k in b}
            row['patch_producing_pair_eligible'] = old['eligibility'][name]['eligible'] if row['task'] in PAIR and name in SHARED else None
            row['exclusion_reason'] = 'whole shared-task pair excluded: earlier SGLang task 13033 has empty patch and capped thinking-only response' if name=='SGLang_earlier' else None
            if engine=='tree':row['speculative']=tree_metrics(audit,root,meta)
            else:
                pre,post=audit.snapshot(root/'vllm_metrics_pre.txt'),audit.snapshot(root/'vllm_metrics_post.txt')
                row['speculative']={'verify_calls':delta(pre,post,'sglang:spec_verify_calls_total',optional=True),
                    'accepted_token_total':None,'acceptance_histogram':None,
                    'reason':'Saved spec_accept_length/rate are gauges of a recent interval, not additive task totals; no averaging or differencing.'}
            if engine=='tree':
                row['speculative']['accepted_plus_events_minus_api_output_tokens']=row['speculative']['accepted_tokens']+row['speculative']['drafts']-row['output_tokens']
            attempts.append(row)
    if len(attempts)!=22 or len({r['attempt_id'] for r in attempts})!=22:raise ValueError('unique attempt accounting')
    groups={name:aggregate([r for r in attempts if r['arm']==name and (r['task'] in PAIR if name in SHARED else True)]) for name in (*SHARED,'Cqc10')}
    groups['Sr12_full_four_tasks']=aggregate([r for r in attempts if r['arm']=='Sr12'])
    for name,total in groups.items():
        close(total['pooled_post_first_token_tokens_per_s'],comp['arms'][name]['totals']['pooled_post_first_token_tokens_per_s'],'aggregate rate')
    report={'schema':'lumotree-review-response-p0-v1','new_inference_launched':False,'python_executable':sys.executable,
        'scope':'All 22 unique attempts in recovered current-route comparison evidence; not every attempt in repository history. Cohort summaries overlap explicitly.',
        'formula':old['formula'],'rate_population':'All completed model requests inside quiescent per-task brackets; includes hidden successful compaction if present.',
        'timing_scope':'Sum of request latencies minus TTFT; overlapping requests count separately. This is pooled decode rate, not machine wall throughput.',
        'selection':old['scope'],'unique_attempt_count':22,'attempts':attempts,'cohorts':groups,
        'configuration':engine_configs,'historical_patcher_review':source_review,
        'metric_source_extraction':semantics_manifest,
        'metric_source_limit':'Extracted immutable base image ID/digest matches historical Cqc10. Historical patcher functions were examined separately; this does not retroactively attest every installed scheduler byte after runtime patching.',
        'additional_archived_attempts_outside_complete_attempt_table':supplemental,
        'development_lineage_provenance_only':lineage,
        'input_sha256':dict(sorted(audit.hashes.items())),
        'implementation_sha256':sha(Path(__file__))}
    write_json('p0-evidence.json',report)
    write_json('additional-attempt-ledger.json',{'current_comparison_supplement':supplemental,
        'development_lineage_provenance_only':lineage,
        'scope':'Adverse, incomplete and earlier records remain visible; absent terminal metrics/verdicts are not invented.'})
    scalar_fields=('arm','engine','task','started_at','ended_at','eval_verdict','completed','patch_bytes','timed_out','budget_capped','response_cap','output_tokens','completed_requests','request_latency_sum_s','ttft_sum_s','agent_elapsed_s','pooled_post_first_token_tokens_per_s','patch_producing_pair_eligible','exclusion_reason')
    write_csv('all-attempts.csv',[{k:r[k] for k in scalar_fields} for r in attempts])
    write_csv('cohort-summary.csv',[{'cohort':k,**{f:v[f] for f in ('task_count','resolved','empty_patches','agent_elapsed_s','output_tokens','completed_requests','post_first_token_latency_sum_s','pooled_post_first_token_tokens_per_s')}} for k,v in groups.items()])
    depth=[];phase_rows=[]
    for r in attempts:
        if r['engine']!='patched vLLM':continue
        for i,n in enumerate(r['speculative']['exact_output_side_accepted_count_histogram']):
            depth.append({'arm':r['arm'],'task':r['task'],'output_side_accepted_count':i,'event_count':int(n)})
        for p,x in r['speculative']['phases'].items():
            if x['available']:phase_rows.append({'arm':r['arm'],'task':r['task'],'phase':p,**{k:v for k,v in x.items() if k!='available'}})
    write_csv('output-acceptance-histograms.csv',depth);write_csv('phase-counters.csv',phase_rows)
    lines=[r'% Generated by reduce_p0.py. Exploratory cohorts; Sr12 subset overlaps full-four cohort.',r'\begin{tabular}{lrrrr}',r'\toprule',r'Build/cohort & Tasks & Resolved & Agent min & Pooled tok/s \\',r'\midrule']
    labels={'Sr12':'Aug. 19 tree (bdca0bd)', 'Cqc16':'Aug. 19 tree (78a29d3)', 'Cqc15':'Aug. 23 tree (a450f6c)', 'SGLang_earlier':'Earlier SGLang$^{*}$','SGLang_later':'Later SGLang','Cqc10':'Aug. 24 tree (10 tasks)','Sr12_full_four_tasks':'Aug. 19 tree (4 tasks)'}
    for k,x in groups.items():lines.append(f"{labels[k]} & {x['task_count']} & {x['resolved']} & {x['agent_elapsed_s']/60:.2f} & {x['pooled_post_first_token_tokens_per_s']:.2f} " + r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'% * Earlier SGLang remains visible as an all-attempt diagnostic: one empty patch; not an eligible patch-producing pair.']
    (HERE/'all-attempts-table.tex').write_text('\n'.join(lines)+'\n')
    pair_lines=[r'% Generated shared-task rows; keeps failed/empty-patch attempts.',
        r'\begin{tabular}{llrrrl}',r'\toprule',
        r'Build & Task & Agent s & Output tokens & Patch bytes & Verdict \\',r'\midrule']
    for r in attempts:
        if r['arm'] in SHARED and r['task'] in PAIR:
            pair_lines.append(f"{labels[r['arm']]} & {r['task'].rsplit('-',1)[-1]} & {r['agent_elapsed_s']:.3f} & {r['output_tokens']:.0f} & {r['patch_bytes']} & {r['eval_verdict']} " + r'\\')
    pair_lines += [r'\bottomrule',r'\end{tabular}']
    (HERE/'paired-task-outcomes.tex').write_text('\n'.join(pair_lines)+'\n')
    payload=[]
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='MANIFEST.json':
            payload.append({'path':str(path.relative_to(HERE)),'bytes':path.stat().st_size,'sha256':sha(path)})
    write_json('MANIFEST.json',{'schema':'lumotree-review-response-p0-delivery-v1',
        'files':payload,'source_input_count':len(audit.hashes),'evidence':'p0-evidence.json',
        'reproduction':'Run reduce_p0.py with Python from paper.config.yaml; then unittest discover.'})
    print(json.dumps({'unique_attempts':22,'cohorts':{k:round(v['pooled_post_first_token_tokens_per_s'],4) for k,v in groups.items()},'input_files_bound':len(audit.hashes)},indent=2))


if __name__=='__main__':
    main()
