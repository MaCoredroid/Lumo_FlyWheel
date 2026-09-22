#!/usr/bin/env python3
"""Small CPU-only source/contract negatives. Does not call Docker or import Torch."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from e8_shim import EXPECTED, render
from e8_verify import head_report, stage

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]
SOURCE = V2/'experiments/out-20260922T100028Z-e1-18cells/cell_05_b1_tree_B1_a1/loaded_backend/eagle.py'


def main():
    results = {}
    raw = SOURCE.read_bytes()
    variants = {}
    for arm in ('on', 'off'):
        for q in (True, False):
            out, rep = render(raw, arm, q)
            tree = ast.parse(out)
            constants = {x.targets[0].id: ast.literal_eval(x.value) for x in tree.body if isinstance(x, ast.Assign) and len(x.targets) == 1 and isinstance(x.targets[0], ast.Name) and x.targets[0].id in ('_E8_ARM', '_E8_QUALIFY')}
            assert constants == {'_E8_ARM': arm, '_E8_QUALIFY': q}
            assert out.count(b'_e8_primary(self,') == 4
            assert out.count(b'_e8_legacy(self,') == 1
            assert out.count(b'_e8_finish(self,') == 1
            assert rep['input_sha256'] == EXPECTED
            variants[(arm, q)] = out
            results[f'render_{arm}_{q}'] = 'PASS'
    assert len(set(variants.values())) == 4
    results['four_distinct_emitted_sources'] = 'PASS'
    for name, args in [('wrong_hash', (raw+b'\n', 'on', True)), ('wrong_arm', (raw, 'invalid', True)), ('non_boolean', (raw, 'on', 1))]:
        try:
            render(*args)
        except ValueError:
            results[name] = 'REFUSED_AS_REQUIRED'
        else:
            raise AssertionError(name)
    good = {'schema': 'e8.head.v1', 'arm': 'on', 'qualify': True, 'closed': True, 'failures': [], 'owner_pid': 123,
            'proposals': 8, 'qualified_proposals': 8, 'primary_head_calls': 40, 'legacy_head_calls': 40,
            'checked_heads': 40, 'root_heads': 8, 'loop_heads': 32, 'dispatch_ops_checked': 800}
    head_report(good, 'on'); results['positive_head_census'] = 'PASS'
    for key, value in [('closed', False), ('failures', ['bad']), ('qualified_proposals', 7), ('primary_head_calls', 41),
                       ('legacy_head_calls', 39), ('checked_heads', 39), ('root_heads', 7), ('loop_heads', 31),
                       ('dispatch_ops_checked', 0), ('arm', 'off'), ('qualify', False), ('proposals', 0), ('owner_pid', None)]:
        d = copy.deepcopy(good); d[key] = value
        try:
            head_report(d, 'on')
        except ValueError:
            results['head_negative_'+key] = 'REFUSED_AS_REQUIRED'
        else:
            raise AssertionError(key)
    m = stage(HERE); results['stage_hashes'] = 'PASS'
    launcher = (HERE/'e8_launch.v1.sh').read_text()
    assert not any(line.startswith('+') for line in launcher.splitlines())
    assert 'FR13_FIX1_SELFCHECK="$E8_QUALIFY"' in launcher
    assert '-e E8_ARM="$E8_ARM"' in launcher and '--seed \'$SEED\'' in launcher
    assert '/e8/e8_shim.py --arm' in launcher and '--qualify --report /logs/e8_shim.json' in launcher
    assert 'docker rm -f' not in launcher
    subprocess.run(['bash', '-n', str(HERE/'e8_launch.v1.sh')], check=True)
    results['launcher_syntax_forwarding_isolation'] = 'PASS'
    for p in HERE.glob('*.py'):
        compile(p.read_text(), str(p), 'exec')
    results['all_python_compile'] = 'PASS'
    with tempfile.TemporaryDirectory(prefix='e8-source-control-') as tmp:
        p = Path(tmp)
        # Importing a serving helper without executing any head must not create
        # or overwrite another process's report at atexit.
        import os, sys
        report = p/'head.json'; report.write_text('sentinel')
        env = dict(os.environ, E8_HEAD_REPORT=str(report), PYTHONPATH=str(HERE))
        subprocess.run([sys.executable, '-B', '-c', 'import e8_head_gate'], env=env, check=True)
        assert report.read_text() == 'sentinel' and not (p/'e8_head_owner.json').exists()
        results['import_without_head_preserves_receipt'] = 'PASS'
        (p/'manifest.json').write_text(json.dumps({'schema': 'e8.freeze.v1', 'batch': 1, 'files': {'missing.py': '0'*64}}))
        try:
            stage(p)
        except ValueError:
            results['missing_manifest_dependency'] = 'REFUSED_AS_REQUIRED'
        else:
            raise AssertionError('missing dependency')
    out = HERE/'tests_out'; out.mkdir(exist_ok=True)
    (out/'cpu_controls.json').write_text(json.dumps({'status': 'PASS', 'count': len(results), 'results': results}, indent=2)+'\n')
    print(json.dumps({'status': 'PASS', 'count': len(results), 'results': results}, indent=2))


if __name__ == '__main__':
    main()
