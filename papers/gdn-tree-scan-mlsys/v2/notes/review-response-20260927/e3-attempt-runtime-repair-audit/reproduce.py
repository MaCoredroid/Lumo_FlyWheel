"""Focused independent F1/F2 repair review. No real Docker, model or task execution."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast
import dataclasses
import hashlib
import inspect
import io
import json
import sys
import unittest

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[2] / 'experiments/review-response-20260927/workload-plan/tools/attempt-runtime'
sys.path.insert(0, str(BUNDLE))
import test_runtime as t
import evaluator as ev
from dependencies import c, e, g

PINS = {
    'MANIFEST.json': 'be15b0fcefcf9bda4f70168607829a8be8d849cc358935dfdfe0af903b3eea6d',
    'runtime.py': '09f35b0faa72839c9a654249f29e20d39b0adae9011082bdb5c0e2cc8778e2e9',
    'evaluator.py': '9b52383d2b5f0e0794cb9dee1194fdf8fe74c846cfc6a4b92a7387f220ac2e71',
    'unit-tests-repair-v2.txt': '173cffd9d60c08e0ae103b9b779a97154aec700c3887bd70f68542503596168d',
}
for name, expected in PINS.items():
    assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == expected

names = sorted(n for n in unittest.defaultTestLoader.getTestCaseNames(t.RuntimeTests)
               if n.startswith(('test_f1_', 'test_f2_')))
stream = io.StringIO()
suite = unittest.TestSuite(t.RuntimeTests(name) for name in names)
run = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
(HERE / 'focused-tests.txt').write_text(stream.getvalue())
assert run.wasSuccessful()


def fresh(fn):
    x = t.RuntimeTests()
    x.setUp()
    try:
        return fn(x)
    finally:
        x.doCleanups()


def boot_and_gate(x):
    assert type(x.r.gates) is g.GateHarness and type(x.r.collector) is c.Collector
    with x.r.own():
        x.r.launch_gate()
        launch = (x.r.directory / 'launch-packet.json').read_bytes()
        assert x.r.packet['stage'] == 'launch'
        assert 'boot_started_at_utc' not in x.r.packet['timeline']
        with patch.object(x.r.collector, 'runtime', wraps=x.r.collector.runtime) as collector:
            terminal = x.r.run_agent(x.fx.server, x.fx.agent, x.fx.manifests,
                                    '/proof.json', x.fx.model_names, x.prediction)
        assert terminal and collector.call_count == 1
        transition = e.load(x.r.directory / 'agent-stage-transition.json')
        assert transition['launch_packet_sha256'] == e.digest(c.decode(launch))
        assert transition['boot_started_at_utc'] == c.normalized_utc(
            x.fx.host.containers[x.fx.server]['State']['StartedAt'])
        assert (x.r.directory / 'launch-packet.json').read_bytes() == launch
        raw = [e.load(e.evidence(x.r.packet, h, 'transition raw'))
               for h in transition['raw_evidence_sha256']]
        docker = next(v['observation'] for v in raw if 'observation' in v)
        assert docker['Id'] == x.fx.server
        verdict = x.r.close()
        assert verdict['status'] == 'CLOSABLE'
        return {'result': 'PASS', 'real_collector_calls': collector.call_count,
                'launch_bytes_unchanged': True, 'raw_boot_identity_verified': True,
                'boot_started_at_utc': transition['boot_started_at_utc'],
                'injected_start_calls': sum(v[0] == 'start' for v in x.transport.calls)}


def former_substitution(x):
    with x.r.own():
        x.run_agent()
        official, _, pred, module = x.evaluator_fixture()
        unbound = SimpleNamespace(instance_id=x.packet['order_row']['instance_id'],
                                 repo='wrong/repository', version='wrong-version',
                                 eval_script='UNBOUND_EVAL_SCRIPT_SENTINEL',
                                 FAIL_TO_PASS=['unbound_test'], PASS_TO_PASS=[])
        with patch.object(module, 'run_instance') as body:
            try:
                official.run(x.r, unbound, pred)
            except e.Refusal as exc:
                reason = str(exc)
            else:
                raise AssertionError('caller TestSpec was accepted')
        assert body.call_count == 0 and x.r.actors['evaluator'] is None
        assert not (x.r.directory / 'evaluator-invocation.json').exists()
        return {'result': 'REFUSED', 'reason': reason,
                'evaluator_calls': 0, 'invocation_created': False}


def constructed_spec(x):
    with x.r.own():
        x.run_agent()
        official, _, pred, module = x.evaluator_fixture()
        original = module.run_instance
        seen = {}
        generated = {}
        factory_globals = x.factory_module.__dict__
        for name in ('make_repo_script_list', 'make_env_script_list', 'make_eval_script_list'):
            original_generator = factory_globals[name]

            def observe_generator(*args, _name=name, _original=original_generator):
                generated[_name] = args
                return _original(*args)

            factory_globals[name] = observe_generator

        def observe_body(spec, *args, **kwargs):
            standard = spec.standard
            fields = dataclasses.asdict(standard)
            projection = {'fields': fields, 'eval_script': standard.eval_script,
                          'install_repo_script': standard.install_repo_script,
                          'setup_env_script': standard.setup_env_script}
            expected_fields = {f.name for f in dataclasses.fields(standard)}
            assert set(fields) == expected_fields
            row = x.dataset_row
            assert fields['instance_id'] == row['instance_id']
            assert fields['repo'] == row['repo'] and fields['version'] == row['version']
            assert fields['FAIL_TO_PASS'] == json.loads(row['FAIL_TO_PASS'])
            assert fields['PASS_TO_PASS'] == json.loads(row['PASS_TO_PASS'])
            assert fields['arch'] == x.r.packet['evaluator_image']['architecture'] == 'x86_64'
            assert fields['namespace'] is None
            assert all(fields[key] == 'latest' for key in
                       ('base_image_tag', 'env_image_tag', 'instance_image_tag'))
            assert fields['docker_specs'] == factory_globals['MAP_REPO_VERSION_TO_SPECS'][row['repo']][row['version']].get('docker_specs', {})
            assert fields['language'] == factory_globals['MAP_REPO_TO_EXT'][row['repo']]
            assert generated['make_env_script_list'][0] is row
            assert generated['make_eval_script_list'][0] is row
            assert generated['make_eval_script_list'][-2:] == (row['base_commit'], row['test_patch'])
            assert generated['make_repo_script_list'][3] == row['base_commit']
            lock = x.r.known['pilot_evaluator_images'][row['instance_id']]
            assert spec.instance_image_key == lock['pinned_ref'] and spec.is_remote_image is True
            invocation = e.load(x.r.directory / 'evaluator-invocation.json')
            binding = invocation['selected_test_spec']
            assert binding['test_spec_sha256'] == e.digest(projection)
            assert binding['selected_row_sha256'] == e.digest(row)
            assert binding['dataset_sha256'] == x.r.known['dataset']['sha256']
            assert binding['constructor_sha256'] == e.sha(Path(x.factory_module.__file__).read_bytes())
            assert binding['harness_source_manifest_sha256'] == e.digest(x.r.known['evaluator_harness']['source_files'])
            for value in ('EVALUATOR_ONLY_TEST', 'EVALUATOR_ONLY_GOLD', 'fixture_fail', 'fixture_pass'):
                assert value not in e.canonical(invocation).decode()
                assert value not in e.canonical(ev.safe_agent_input(row)).decode()
            seen.update(full_projection_hash_matches=True, selected_row_hash_matches=True,
                        dataset_constructor_harness_hashes_match=True,
                        dataclass_fields=sorted(expected_fields),
                        generated_script_keys=['eval_script', 'install_repo_script', 'setup_env_script'],
                        default_options_preserved=True, immutable_image_override_verified=True,
                        test_content_absent_from_invocation_and_agent_input=True)
            return original(spec, *args, **kwargs)

        module.run_instance = observe_body
        with patch.object(ev, 'verify_harness', return_value=None), patch.object(
                ev, 'selected_record', wraps=ev.selected_record) as selected:
            official.run(x.r, None, pred)
        assert selected.call_count == 1
        assert x.r.close()['status'] == 'CLOSABLE'
        return dict(seen, result='PASS', selected_row_reads=selected.call_count,
                    evaluator_body='injected double; no official task tests')


def wrong_metadata(x, key):
    with x.r.own():
        x.run_agent()
        official, _, pred, module = x.evaluator_fixture()
        x.dataset_row[key] = 'wrong-metadata'
        with patch.object(ev, 'verify_harness', return_value=None), patch.object(
                module, 'run_instance') as body:
            try:
                official.run(x.r, None, pred)
            except e.Refusal as exc:
                reason = str(exc)
            else:
                raise AssertionError('wrong selected-row metadata accepted')
        assert body.call_count == 0
        return {'result': 'REFUSED', 'reason': reason, 'evaluator_calls': 0}


def test_names(path):
    return {n.name for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')}

original_names = test_names(BUNDLE / 'superseded/submitted-v1-2f15cfc427e0/test_runtime.py')
current_names = test_names(BUNDLE / 'test_runtime.py')
assert original_names <= current_names
results = {
    'scope': 'FOCUSED_INJECTED_CPU_ONLY', 'pins_verified': PINS,
    'focused_tests_run': run.testsRun, 'focused_tests_passed': run.wasSuccessful(),
    'author_full_suite_log_verified': {'reported_tests': 91, 'independently_rerun': False},
    'original_test_names_retained': len(original_names), 'new_test_names': len(current_names - original_names),
    'F1_original_sequence': fresh(boot_and_gate),
    'F2_original_substitution': fresh(former_substitution),
    'F2_internal_construction': fresh(constructed_spec),
    'additional_metadata_refusals': {key: fresh(lambda x, key=key: wrong_metadata(x, key))
                                   for key in ('instance_id', 'base_commit', 'version', 'environment_setup_commit')},
}
print(json.dumps(results, indent=2, sort_keys=True))
