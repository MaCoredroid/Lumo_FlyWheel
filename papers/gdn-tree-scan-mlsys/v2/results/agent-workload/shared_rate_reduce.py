#!/usr/bin/env python3
"""Recompute common request-rate estimators from saved SWE task brackets.

CPU only; no network, inference, or edits to original records. Both systems use
the same algebra, pooling completed request tokens and latencies before division.
These are descriptive rates on different Astropy subsets and serving stacks.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
TREE_IDS = ('13977', '14096', '14182', '14309', '14365',
            '14369', '14508', '14539', '14598', '14995')
SG_TASKS = (('run1', '12907'), ('r1', '13033'))
MODEL = 'qwen3.8-27b-nvfp4-radixark'


def close(a, b, label):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(f'{label}: {a} != {b}')


def rates(row):
    n, r = row['output_tokens'], row['completed_requests']
    e, f = row['request_latency_sum_s'], row['ttft_sum_s']
    if not (n > r > 0 and e > f >= 0 and row['agent_elapsed_s'] > 0):
        raise ValueError('invalid token/request/time support')
    return {
        'pooled_post_first_token_tokens_per_s': (n-r)/(e-f),
        'pooled_request_tokens_per_s_including_ttft': n/e,
        'output_tokens_per_agent_elapsed_second': n/row['agent_elapsed_s'],
        'post_first_token_count': n-r,
        'post_first_token_latency_sum_s': e-f,
    }


class Audit:
    def __init__(self, repo):
        self.repo, self.hashes = repo, {}

    def read(self, path):
        data = path.read_bytes()
        self.hashes[str(path.relative_to(self.repo))] = hashlib.sha256(data).hexdigest()
        return data.decode('utf-8')

    def snapshot(self, path):
        values = {}
        for line in self.read(path).splitlines():
            if not line or line.startswith('#'):
                continue
            m = re.fullmatch(r'([^\s{]+)(\{[^}]*\})?\s+(\S+)(?:\s+\S+)?', line)
            if not m:
                raise ValueError(f'unparsed metric line in {path}')
            name, labels, raw = m.groups()
            if name.endswith(('_bucket', '_created')):
                continue
            labels = tuple(sorted(re.findall(r'(\w+)="([^"]*)"', labels or '')))
            key = (name, labels)
            if key in values:
                raise ValueError(f'duplicate metric {key}')
            values[key] = float(raw)
        return values

    def task(self, root, arm):
        prefix = 'vllm:' if arm == 'tree' else 'sglang:'
        before = self.snapshot(root/'vllm_metrics_pre.txt')
        after = self.snapshot(root/'vllm_metrics_post.txt')
        used = {}

        def total(snapshot, name, streaming=None):
            found = []
            for (metric, labels), value in snapshot.items():
                if metric != prefix+name:
                    continue
                labels = dict(labels)
                if labels.get('model_name') != MODEL:
                    raise ValueError(f'unexpected model for {metric}: {labels}')
                if arm == 'tree' and labels.get('engine') != '0':
                    raise ValueError('unexpected vLLM engine')
                if arm == 'sglang' and labels.get('engine_type') != 'unified':
                    raise ValueError('unexpected SGLang engine')
                if 'is_streaming' in labels and labels['is_streaming'] not in ('true', 'false'):
                    raise ValueError('unexpected streaming label')
                if streaming is not None and labels.get('is_streaming') != streaming:
                    continue
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'invalid metric {metric}')
                found.append(value)
            if not found:
                raise ValueError(f'missing {prefix+name}, streaming={streaming}')
            return sum(found)

        def delta(name, streaming=None):
            value = total(after, name, streaming)-total(before, name, streaming)
            if value < 0:
                raise ValueError(f'counter reset: {name}')
            used[name + (f'|streaming={streaming}' if streaming else '')] = value
            return value

        # Empty queue/running gauges at both boundaries exclude in-flight carryover.
        for name in (('num_requests_running', 'num_requests_waiting') if arm == 'tree'
                     else ('num_running_reqs', 'num_queue_reqs')):
            close(total(before, name), 0, 'pre boundary '+name)
            close(total(after, name), 0, 'post boundary '+name)

        count_name = 'request_success_total' if arm == 'tree' else 'num_requests_total'
        hist_name = 'request_generation_tokens' if arm == 'tree' else 'generation_tokens_histogram'
        n, r = delta('generation_tokens_total'), delta(count_name)
        for value in (n, r):
            close(value, round(value), 'integer token/request count')
        close(delta(hist_name+'_sum'), n, 'completed token histogram vs generation counter')
        close(delta(hist_name+'_count'), r, 'completed token histogram count')
        close(delta('e2e_request_latency_seconds_count'), r, 'E2E request count')
        close(delta('time_to_first_token_seconds_count'), r, 'TTFT request count')
        e, f = delta('e2e_request_latency_seconds_sum'), delta('time_to_first_token_seconds_sum')

        if arm == 'sglang':
            # A non-streaming first-response clock is not a first-token clock.
            for name in ('generation_tokens_total', 'num_requests_total',
                         'e2e_request_latency_seconds_count', 'time_to_first_token_seconds_count',
                         'e2e_request_latency_seconds_sum', 'time_to_first_token_seconds_sum'):
                close(delta(name, 'false'), 0, 'nonstreaming contribution '+name)
            # Streaming series are lazily created after the pre-task warmup.
            # All requests minus zero nonstreaming requests is the streaming
            # population; do not require an uninstantiated pre-task series.
        else:
            close(delta('request_decode_time_seconds_count'), r, 'decode request count')
            delta('request_decode_time_seconds_sum')  # Independent clock, not substituted.

        meta = json.loads(self.read(root/'runner_metadata.json'))
        evaluation = json.loads(self.read(root/'eval/eval_report.json'))
        if meta['instance_id'] != root.name or evaluation['instance_id'] != root.name:
            raise ValueError('task identity mismatch')
        if meta['agent']['timed_out'] or meta['agent'].get('budget_capped', False):
            raise ValueError('incomplete task is not a completed-task bracket')
        row = {
            'task': root.name, 'eval_verdict': evaluation['verdict'],
            'output_tokens': n, 'completed_requests': r,
            'request_latency_sum_s': e, 'ttft_sum_s': f,
            'agent_elapsed_s': meta['agent']['elapsed_s'],
            'raw_metric_deltas': used,
            'boundary_and_population_checks': 'PASS',
        }
        row.update(rates(row))
        return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=HERE.parents[4])
    args = ap.parse_args()
    repo = args.repo.resolve()
    audit = Audit(repo)
    tree = repo/'papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10/swe_out/verified/per_task'
    sg = repo/'results/fr14_nvfp4_port_20260816/sglang16_evidence'
    sets = {'tree': [tree/('astropy__astropy-'+t) for t in TREE_IDS],
            'sglang': [sg/run/'swe_out/verified/per_task'/('astropy__astropy-'+t)
                       for run, t in SG_TASKS]}
    arms = {}
    fields = ('output_tokens', 'completed_requests', 'request_latency_sum_s',
              'ttft_sum_s', 'agent_elapsed_s')
    for arm, paths in sets.items():
        tasks = [audit.task(path, arm) for path in paths]
        totals = {k:sum(t[k] for t in tasks) for k in fields}
        totals.update(rates(totals))
        arms[arm] = {'task_count':len(tasks), 'resolved':sum(t['eval_verdict']=='resolved' for t in tasks),
                     'tasks':tasks, 'totals':totals}
    differences = {k:100*(arms['tree']['totals'][k]/arms['sglang']['totals'][k]-1)
                   for k in ('pooled_post_first_token_tokens_per_s',
                             'pooled_request_tokens_per_s_including_ttft',
                             'output_tokens_per_agent_elapsed_second')}
    report = {
        'schema':'shared-swe-request-rate-v1',
        'scope':'Descriptive common-estimator comparison on different SWE-bench Verified Astropy subsets; different deployed serving stacks; no causal speedup claim.',
        'formulas':{
            'pooled_post_first_token_tokens_per_s':'(sum output tokens - completed request count) / (sum E2E request latency - sum TTFT)',
            'pooled_request_tokens_per_s_including_ttft':'sum output tokens / sum E2E request latency',
            'output_tokens_per_agent_elapsed_second':'sum output tokens / sum agent elapsed time',
        },
        'timing_boundary':'Server-recorded request timing; E2E includes TTFT. Summed request latency may count overlapping requests separately; it is not a machine wall-clock throughput denominator.',
        'population':'All completed model requests inside completed task brackets, including any agent-internal compaction. SGLang nonstreaming delta is zero in both selected brackets.',
        'arms':arms,
        'tree_relative_to_sglang_percent_descriptive':differences,
        'input_sha256':dict(sorted(audit.hashes.items())),
        'reducer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_inference_launched':False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
