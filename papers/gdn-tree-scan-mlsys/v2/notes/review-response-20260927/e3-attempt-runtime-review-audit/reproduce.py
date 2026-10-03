"""Independent CPU-only review probes; all transport/evaluator activity is injected."""
from pathlib import Path
import copy
import hashlib
import json
import sys
from unittest.mock import patch

BUNDLE = Path(__file__).resolve().parents[3] / 'experiments/review-response-20260927/workload-plan/tools/attempt-runtime'
sys.path.insert(0, str(BUNDLE))
import test_runtime as t
import evaluator as ev
import agent as ag
from dependencies import c, e, g

PINS = {
    'MANIFEST.json': '2f15cfc427e081001d934b868c82b63973f449bea63f504a2a6a6683a60170f1',
    'runtime.py': '1dd34ef9d30dec24c1b594c148928499e10bb655222f8c3c67b342cda4745d45',
    'evaluator.py': '71ee0a8fda59d3bf1f57912a1de359b4931639dcafd11e8fff7f218abdb84882',
}
for name, expected in PINS.items():
    assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == expected


def fresh(fn):
    x = t.RuntimeTests()
    x.setUp()
    try:
        return fn(x)
    finally:
        x.doCleanups()


def actual_gate(x, manual_transition=None):
    # Replace only physical inputs with the already accepted small raw fixtures.
    # Unlike RuntimeTests, retain the real GateHarness and Collector methods.
    x.fx.runtime_fixture()
    assert x.f.fsha == x.r.freeze_sha
    x.r.packet = copy.deepcopy(x.fx.packet)
    x.r.packet['stage'] = 'launch'
    x.r.packet['timeline'] = {
        key: x.r.packet['timeline'][key]
        for key in ('clock_id', 'boot_id', 'attempt_opened_at_utc', 'launch_gate_at_utc')
    }
    x.fx.host.now = '2026-09-27T00:00:12+00:00'
    x.r.packet['timeline_receipt_sha256'] = x.fx.collector.envelope(
        x.f.freeze, x.r.packet, 'attempt_timeline', x.r.packet['timeline'],
        [x.fx.store.json({'fixture': 'launch timeline'})],
    )
    x.r.packet['evidence_files'].update(x.fx.store.files)
    x.r.gates = g.GateHarness(x.r.campaign, x.f.known, x.f.freeze, x.f.fsha,
                            x.fx.collector, x.r.journal)
    with x.r.own():
        launch = x.r.launch_gate()['status']
        x.fx.host.now = '2026-09-27T00:00:14+00:00'
        if manual_transition:
            x.r.packet['stage'] = 'agent'
        if manual_transition == 'stage_and_observed_boot':
            x.r.packet['timeline']['boot_started_at_utc'] = c.normalized_utc(
                x.fx.host.containers[x.fx.server]['State']['StartedAt'])
        try:
            terminal = x.r.run_agent(x.fx.server, x.fx.agent, x.fx.manifests,
                                    '/proof.json', x.fx.model_names, x.prediction)
            result = {'terminal_observed': bool(terminal), 'closure_status': x.r.close()['status']}
        except Exception as exc:
            result = {'exception_type': type(exc).__name__, 'reason': str(exc)}
        return dict(result, launch=launch, manual_fixture_transition=manual_transition,
                    injected_start_calls=sum(call[0] == 'start' for call in x.transport.calls))


def unbound_spec(x):
    with x.r.own():
        x.run_agent()
        official, spec, prediction, module = x.evaluator_fixture()
        spec.repo = 'wrong/repository'
        spec.version = 'wrong-version'
        spec.eval_script = 'UNBOUND_EVAL_SCRIPT_SENTINEL'
        spec.FAIL_TO_PASS = ['unbound_test']
        spec.PASS_TO_PASS = []
        seen = {}
        original = module.run_instance

        def record_spec(actual_spec, *args, **kwargs):
            seen.update(instance_id=actual_spec.instance_id, repo=actual_spec.repo,
                        version=actual_spec.version, eval_script=actual_spec.eval_script,
                        FAIL_TO_PASS=actual_spec.FAIL_TO_PASS)
            return original(actual_spec, *args, **kwargs)

        module.run_instance = record_spec
        with patch.object(ev, 'verify_harness', return_value=None), patch.object(
                ev, 'selected_record', side_effect=AssertionError('selected row must be bound')) as selected:
            result = official.run(x.r, spec, prediction)
            verdict = x.r.close()
        return {'forwarded_spec': seen, 'selected_record_calls': selected.call_count,
                'injected_evaluator_return': result, 'closure_status': verdict['status'],
                'closure_resolved': e.load(x.r.directory / 'attempt-closure.json')['resolved'],
                'scope': 'official body/source check are fixture doubles; no task tests execute'}


def recovery(x):
    with x.r.own():
        x.run_agent('/missing.json')
    before = len(x.transport.calls)
    with x.r.own(recovery=True):
        result = x.r.recover('/missing.json')
    actions = [v[0] for v in x.transport.calls[before:] if v[0] in ('start', 'wait', 'kill')]
    assert not actions
    return {'status': result['status'], 'recovery_mutating_calls': actions}


def patch_distinction(x):
    with x.r.own():
        x.run_agent()
        observed = x.observed_patch_file(b'')
        destination = x.root / 'empty-observed-prediction.json'
        ag.prediction_from_observed_patch(x.r, observed, destination)
        empty_value = e.load(destination)[0]['model_patch']
        missing_destination = x.root / 'missing-not-created.json'
        try:
            ag.prediction_from_observed_patch(x.r, x.root / 'missing.diff', missing_destination)
        except FileNotFoundError:
            missing_refused = True
        else:
            missing_refused = False
        assert empty_value == '' and missing_refused and not missing_destination.exists()
        return {'observed_empty_patch': empty_value, 'missing_patch_refused': missing_refused,
                'missing_prediction_created': missing_destination.exists()}


results = {
    'scope': 'INJECTED_CPU_ONLY_NO_DOCKER_SSH_MODEL_EVALUATOR_TASK',
    'pins_verified': PINS,
    'real_gate_documented_sequence': fresh(actual_gate),
    'real_gate_stage_only_diagnostic': fresh(lambda x: actual_gate(x, 'stage_only')),
    'real_gate_manual_transition_positive_control': fresh(lambda x: actual_gate(x, 'stage_and_observed_boot')),
    'unbound_testspec': fresh(unbound_spec),
    'recovery_control': fresh(recovery),
    'empty_missing_patch_control': fresh(patch_distinction),
}
assert results['real_gate_documented_sequence']['injected_start_calls'] == 0
assert 'actual agent hook phase' in results['real_gate_documented_sequence']['reason']
assert results['real_gate_manual_transition_positive_control']['terminal_observed'] is True
assert results['unbound_testspec']['closure_resolved'] is True
print(json.dumps(results, indent=2, sort_keys=True))
