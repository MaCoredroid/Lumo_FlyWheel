# E3 identity adapter: independent CPU red-team review

27 September 2026. **Verdict: adapter logic needs repair before integration.** The intentionally incomplete drafts correctly refuse. The supplied 51 tests and bundle checks pass, but eight independent invalid fixtures return `IDENTITY_GATE_PASS`. Four bounded binding gaps are described below. This review approves neither WP nor WC and launches no workload, evaluator task, server, GPU, model API, or remote operation.

Reviewed directory: `experiments/review-response-20260927/workload-plan/tools/identity-adapter/`, relative to paper v2. Only this independent report was written; tests used local temporary synthetic files. The owner has been sent exact reproductions and will preserve the reviewed seal before repair. Findings apply to the hashes below, not an unreviewed repair in progress.

## Reviewed identities and passing controls

| File | SHA-256 |
|---|---|
| `MANIFEST.json` | `b9a13706d23dd5a54cccd5c4f4ed65ece05516d0c23f4bf15e3ac6e240fabf3e` |
| `HANDOFF.md` | `a7a58a8759371fae3b544a5b59338ede1c9cbdec1d94c28a8eb989854fa5aee8` |
| `known-locks.json` | `8ace7c4c3270f45c39cc4e247373c58579716a947b822792f585ef3dd03d493f` |
| `e3_preflight.py` | `d4232dc19cbf9a15d4ba2db15cfc67bb427e8454858ef248e23f59a230b84a10` |
| `test_e3_preflight.py` | `5e9df4476d3eba2ff7cb7b35e366ed79a7adb94ad258f8f93e088b1423b0020a` |

Executed from the adapter directory with `/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B`:

- `test_e3_preflight.py`: **51/51 PASS**.
- `verify_bundle.py`: **11 adapter files verified**, original anchors verified, runtime freeze pending, zero experiments.
- The exact draft CLI command saved in `VALIDATION.json`: exit **2**, `REFUSED`, reason `runtime freeze status differs from bound identity`.
- Independently rehashed all **13 original source bindings** from `known-locks.json`; they remain byte-identical, including the accepted workload/evaluator/model manifests.

Existing tests correctly reject altered dataset bytes/snapshot, wrong phase/order/task, missing prior ledger rows, changed model file maps, floating evaluator tags, image/template mismatch, wrong observation attempt or subject, empty raw-observation lists/files, explicit retries, stale report flags, duplicate JSON keys, and non-finite JSON. The missing checks below are different from those passing controls.

## Concrete blockers

### F1 — Closure and completion bodies are not bound to the claimed attempt

`e3_preflight.py:217–221` checks each predecessor's ID and `CLOSED_WITH_RECORD` only in the caller's packet, then merely hashes the referenced file. It never parses that closure file. Similarly, `:279` only hashes the agent-completion file before permitting evaluator stage.

Reproduced: an **empty predecessor receipt** admits the next launch; an **empty agent-completion receipt** admits evaluation; a completion JSON explicitly containing `{"attempt_id":"another-attempt","state":"RUNNING"}` also admits evaluation. The frozen configuration and current attempt are otherwise valid synthetic fixtures. These are accidental stale/wrong-file cases the adapter can reject without attempting hardware attestation.

**Minimal repair:** define and parse predecessor terminal-record and agent-terminal-record schemas. Bind freeze, phase, order row/configuration, task, attempt, retry and permitted terminal status; reject `RUNNING`, empty records and identity mismatches. Bind the agent terminal record to the exact final prediction hash subsequently evaluated. Preserve failed/timeout/empty-patch starts under explicitly declared terminal handling; do not make test success or nonempty patches conditions for retaining the all-start ledger. Add swapped-valid-record, stale-completion/different-prediction and empty-body controls.

### F2 — Evaluator stage does not validate an owned, never-started container

`e3_preflight.py:269–280` compares the supplied TestSpec key and image ID, but requires no evaluator-container ID, owned-attempt label, creation/start state, or evaluator-stage observation envelope. The existing `evaluator_inspection` envelope at `:232–241` binds the generic image identity, not the final evaluator container and evaluation subject.

Reproduced: evaluator stage passes with `container_running=true`, `container_status="running"`, `container_id="wrong-container"`, and `ownership_attempt_id="other-attempt"`. Those fields are ignored; a packet with none of them passes too. Thus the claimed stopped-container boundary is not checked by the adapter.

**Minimal repair:** require a fresh evaluator-container observation bound to the evaluation subject: actual container ID, immutable image, actual TestSpec/task and run ID, prediction hash, owner/attempt labels, and created/not-running/not-previously-started state before patch application/tests. Parse the relevant recorded inspection fields. The future wrapper must obtain this inspection from the same owned container that it returns to `run_instance`; on refusal, it must not start that container. Add correct-image/wrong-owner and stopped-but-reused container negatives. The adapter need not itself run Docker.

### F3 — Source-manifest members and approved collector identity are not enforced

`e3_preflight.py:158–159` hashes the shared/route/runtime manifest files but does not validate their member maps or member bytes. `observation():61` accepts any hash-verifiable collector file supplied in `evidence_files`; it does not require that collector to be approved by the configuration's runtime-source manifest, and even accepts an empty collector file. Runtime observations contain no required actual-loaded source map checked against these manifests.

Reproduced: after sealing a runtime manifest naming a source file and its original hash, changing that member's bytes still passes launch. Replacing an observation collector with an **unreviewed different collector**, while keeping the approved freeze unchanged, passes. A **zero-byte collector** passes as well.

**Minimal repair:** define nonempty manifest schemas and validate expected member identity. For local sources, hash members directly; for sources observed inside a server, compare the observed loaded-source map to the frozen expected map through the approved collector. Restrict observation collectors to explicitly approved source members/roles and require nonempty matching bytes. Do not silently resolve arbitrary new collectors from an untrusted packet. Add deleted/changed member and unapproved/empty collector negatives. This is source-integrity enforcement, not a claim that a reviewed collector cannot lie.

### F4 — Timestamp validation checks format, not freshness or chronology

`observation():58–60` only requires a parseable UTC timestamp. A correctly bound observation dated **1900-01-01T00:00:00+00:00** passes. There is no binding to attempt reservation, boot creation, gate acquisition or pre-agent/pre-evaluator chronology. Wrong-attempt detection alone does not reject stale observations from a reused attempt/boot identity.

**Minimal repair:** bind observation times to declared attempt/boot/start and stage chronology from the trusted launcher clock, with a fixed predeclared allowable skew. Require acquisition after the relevant resource/attempt boundary and before the dependent transition. Include future and out-of-order negatives as well as old timestamps. Do not invent a generic “within N seconds of now” threshold during a run; the clock/skew policy belongs in the reviewed freeze. Preserving startup observations across agent/evaluator gates remains valid when their boot and chronology are correctly bound.

## Compact reproducer against the reviewed seal

Run the following with `python3 -B -` from the original adapter snapshot. It imports only the standard-library validator and its synthetic fixture. Every listed case printed `IDENTITY_GATE_PASS` in this review.

```python
import json
from pathlib import Path
import e3_preflight as e
from test_e3_preflight import AdapterTests
assert e.sha(Path(e.__file__).read_bytes()) == (
    'd4232dc19cbf9a15d4ba2db15cfc67bb427e8454858ef248e23f59a230b84a10')

def add(t, name, body):
    p = t.root / name
    p.write_bytes(body)
    h = e.sha(body)
    t.evidence[h] = str(p)
    return h

def completion(t, body):
    t.evaluator()
    t.packet['evaluation']['agent_completion_receipt_sha256'] = add(t, 'terminal', body)

def prior(t):
    old, row = t.freeze['order'][:2]
    t.packet.update(attempt_ordinal=2, order_row=row)
    t.packet['prior_attempts'] = [{
        'attempt_id': e.attempt_id(t.fsha, old), 'state': 'CLOSED_WITH_RECORD',
        'receipt_sha256': add(t, 'prior', b'')}]
    c = t.freeze['configurations'][row['configuration_id']]
    t.packet['launch'].update({k: c[k] for k in
        ['image_id', 'argv', 'model_path', 'tokenizer_path',
         'effective_template_sha256', 'content_format']})
    t.rebind()

def container(t):
    t.evaluator()
    t.packet['evaluation'].update(container_running=True, container_status='running',
        container_id='wrong-container', ownership_attempt_id='other-attempt')

def collector(t, body):
    h = add(t, 'collector.py', body)
    t.edit_observation(lambda o: o.update(collector_sha256=h))

def source(t):
    p = t.root / 'worker.py'
    p.write_text('ORIGINAL SOURCE\n')
    manifest = {'schema': 'runtime-source-manifest-v1',
                'files': [{'path': str(p), 'sha256': e.sha(p.read_bytes())}]}
    h = add(t, 'source-manifest.json', e.canonical(manifest))
    t.change_config('runtime_source_manifest_sha256', h)
    p.write_text('CHANGED EXECUTED SOURCE\n')

cases = [
    ('empty completion', lambda t: completion(t, b''), 'evaluator'),
    ('wrong RUNNING completion', lambda t: completion(t,
        b'{"attempt_id":"another-attempt","state":"RUNNING"}'), 'evaluator'),
    ('empty predecessor', prior, 'launch'),
    ('running wrong-owned evaluator', container, 'evaluator'),
    ('old timestamp', lambda t: t.edit_observation(lambda o: o.update(
        observed_at_utc='1900-01-01T00:00:00+00:00')), 'launch'),
    ('unreviewed collector', lambda t: collector(t, b'UNREVIEWED\n'), 'launch'),
    ('empty collector', lambda t: collector(t, b''), 'launch'),
    ('changed source member', source, 'launch'),
]
for name, mutate, stage in cases:
    t = AdapterTests('test_synthetic_launch_positive')
    t.setUp()
    try:
        mutate(t)
        print(name, t.check(stage)['status'])
    finally:
        t.doCleanups()
```

## Explicitly pending integration, distinct from these bugs

The supplied adapter is a pure validator. No actual prelaunch, pre-agent or pre-evaluator hook is implemented by this directory. `HANDOFF.md` correctly says the campaign wrapper/worker still must replace the **consumed** dataset/image/argv values, preserve historical fixed32 provenance gates, avoid the evaluator's unpinned dataset reload, use a real immutable TestSpec image key, and intercept the actual `build_container` boundary before `run_instance` starts it. Checking a sidecar while historical helpers still use `latest` would not implement those requirements.

I do not count that disclosed unfinished integration as a newly discovered adapter bug. Likewise, draft refusal, unbound nonpilot evaluator images, undecided runtime/common policy, unsupported retries and absent real runtime receipts are intentional blockers. Repair F1–F4 first, then review the actual wrapper and execution-level CPU stubs at all three transitions. None of the synthetic positive tests constitutes seed-delivery, qualification, task-accounting or workload evidence.
