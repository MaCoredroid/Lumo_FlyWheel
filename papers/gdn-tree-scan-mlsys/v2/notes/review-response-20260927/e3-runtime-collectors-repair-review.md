# E3 runtime collectors v2 repair review — 2026-09-27

**Disposition: approve the repaired, sealed CPU preparation within its stated scope. F1–F4 are addressed; no blocking finding remains in these deltas.** This approves source and injected-fixture behavior, not a runtime freeze, actual instrumentation, launcher insertion or workload execution. No Docker subprocess, remote operation, server/model request, GPU work, evaluator task or workload ran during this review. Only this review note was written; temporary local fixtures were cleaned up.

## Seal and reproduction

Reviewed `REPAIR-REVIEW.md`, `HANDOFF.md`, `collectors.py`, `contracts.py`, `integration.py`, the tests, DRAFT extension/mapping, source proposal and bindings. The independently computed hashes match the supplied pins:

| File | SHA-256 |
|---|---|
| `MANIFEST.json` | `aead7cdbea04e10bf591afa5056e4083a362fc990cbedeecfa76f4c8939ef0ee` |
| `collectors.py` | `32200854bd57d80d6bb35b336fb568b0ac8dcbc3ff1f376cd395d2697bd657b9` |
| `integration.py` | `017eab3ea748341b0a826b2a7a469886e160ea53f0f41f044e4b90869a695f8f` |
| `contracts.py` | `7f1022f1a8825e1c2dfba312dc59aa0c59b513a37e393deeed68fb8cc58e5a9c` |

From `experiments/review-response-20260927/workload-plan/tools/runtime-collectors`, reran:

```sh
/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -m unittest discover -s . -p test_collectors.py -q
/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B verify_bundle.py
```

Results: **81 tests passed**; **25 sealed payload files**, **29 accepted adapter files**, source anchors and known locks verified. An AST comparison confirms all 51 original test names remain, with 30 additions. Separately verified every original payload against the preserved v1 manifest: **10/10 byte hashes and sizes match**, and that manifest still hashes to `554b49adbfa6bcf230e21d8345b359bf49e8b3811c3a4357bd0d555b4171e56c`. The accepted adapter manifest remains `bee950319dbb59244f2cec4b12081c9fda16eec8352a7f538032f453608f4e95`. These are CPU verification counts; experimental observations remain zero.

## Independent adversarial reruns

Used fresh `test_collectors.CollectorTests` instances with `setUp()` and `doCleanups()` for each row. Runtime cases call `runtime_fixture()` first; `run_runtime()` uses only the injected host. Wrapper checks use a fresh `AttemptJournal` reservation and `GateHarness` with the fixture's bound freeze. No implementation or test source was changed.

| Case and reproduction | Observed result |
|---|---|
| F1: preserve raw producer times `00:00:14`; set `host.now` to `00:02:14`; call `run_runtime()` | Refuses: raw producer exceeds frozen maximum age (120 seconds versus 30). |
| F2: change agent label `lumotree.configuration` to `wrong-configuration`; collect | Refuses before admission: owned configuration differs. |
| F3: set delivered `effective_seed=7`; call `refresh_probes()` then collect against fixture expectation 26092700 | Refuses: delivered seed differs from independent frozen expectation. |
| F4: exited agent with code 137, existing reserved directory, `/missing-prediction.json`; call `agent_terminal` | Refuses evaluation and retains reservation plus `agent-terminal-REFUSED.json`: actual exit 137, prediction `MISSING`, no invented prediction or successful terminal. The journal's referenced content-addressed failure bytes were independently verified. |
| Collect at `00:00:14`, delay `GateHarness.check(...,'agent',...)` until `00:00:44` | Passes at the exact 30-second boundary; all six contract records retain the original production time, actual check time and age 30. |
| Same collection, delay wrapper check until `00:00:45` | Refuses at 31 seconds; `agent-REFUSED.json` exists and `agent-PASS.json` does not. |
| Bind the shipped `seed-expectations.DRAFT.json` as the mapping | Refuses draft approval status. |
| Independently replace fixture mapping with expected seed 7 for every row, rebind the freeze/attempt, update delivered seed and run `before_agent_request` | Combined wrapper gate passes; expected and actual are both 7, with a changed freeze identity and scope `pre_agent_probe`. This is a fixture, not a campaign seed choice. |
| Remove `collector_contracts` source approval from the fixture manifest | Refuses unapproved source/role. |

## Closure of the four findings

**F1 — addressed.** Every raw probe hash is in the runtime subject. `contracts.validate_runtime` reopens the bound raw bytes and checks producer identity/request, timestamps and frozen freshness. Both collection and the wrapper enforce it; the wrapper uses its current clock for agent admission, so a delay cannot preserve a stale pass. The suite additionally covers one stale producer, future time beyond skew and the exact boundary. Evaluator-stage revalidation intentionally uses the retained admission interval; it does not require fresh warmup probes after a long task.

**F2 — addressed.** Admission checks campaign, attempt, phase, configuration, boot and task. Actual agent container ID/image and ownership enter the bound subject. Terminal collection requires the same ID/image and verifies the startup subject; the suite rejects copied-label replacement, changed image and packet-only rebinding. Separate-host observation remains explicit.

**F3 — addressed for named pre-agent probes.** The additive contract requires the parent freeze to hash an independent mapping covering the full ordered phase and each complete scheduled row. Request IDs and expected integer seeds are explicit; delivered values must match. Mapping/row hashes and actual/expected values enter the runtime subject and contract verdict. The unchanged accepted adapter alone does not enforce this extension: the reviewed wrapper must also approve and call `contracts.py`. Source and handoff make this dependency explicit.

**F4 — addressed for collection-failure retention.** Missing, unreadable, malformed/empty/stale prediction and unobservable container paths retain an attempt-bound refusal with available evidence and explicit absence. Present empty-string patches remain valid prediction records; an empty or absent file is not converted into one. A prior terminal receipt cannot be overwritten by retry. The future wrapper must retain the reservation and lock through closure; a collection refusal does not itself close the row.

## Bounded acceptance

The preparation consistently distinguishes observed runtime fields from intended configuration and fails when required telemetry is missing. The repaired source is suitable for the next reviewed implementation step. The actual instrumentation producer, measured-request seed derivation/delivery/accounting, real enforcing call sites, runtime clock/transport qualification and final ordered attempt closures remain future obligations, as disclosed. They are not additional requirements for accepting this CPU preparation, and no corresponding runtime or scientific outcome is established here. Failed or missing-prediction attempts must remain in the all-start ledger when that closure path is implemented.
