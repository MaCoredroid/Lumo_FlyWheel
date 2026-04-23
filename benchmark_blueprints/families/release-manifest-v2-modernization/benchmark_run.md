# Benchmark Run

## Attempt History

- `attempt_01`
  - model: `gpt-5.4`
  - reasoning: `high`
  - agent: `019da338-6ac1-7aa0-adcc-8e732e985489`
  - result: over target under the original flat evaluator because visible dry-run
    and visible smoke behavior were over-credited.
- `attempt_02`
  - model: `gpt-5.4`
  - reasoning: `high`
  - agent: `019da33f-cf98-7a03-8d2f-44e216a50103`
  - result: hardened flat scaffold scored `20/100`, but the family still lacked
    the five-variant bundle, dual-band scorer, manifest lock, family registry,
    milestone scripts, and verification matrices required by Layer B.
- `attempt_03`
  - model: `family-local deterministic rebuild`
  - result: replaced the flat scaffold with a five-variant family-owned bundle,
    added `family.yaml`, `manifest.lock.json`, a deterministic dual-band scorer,
    hidden tests, milestone scripts, and verification matrices.
- `attempt_04`
  - model: `gpt-5.4` live probe batch
  - result: partial only. `v1-clean-baseline` completed at `20/100`, but the
    run exposed a family-design bug: the writable config surface lived under
    hidden `.codex/`, which `codex exec` could not reliably edit in this
    environment.
- `attempt_05`
  - model: `family-local surface fix + narrowed live rerun`
  - result: moved the writable config surface to `codex/config.toml`,
    regenerated the family, and started a narrowed `v1-clean-baseline` rerun.
    The post-fix live run did not finish within the turn budget, so no scored
    post-fix result is recorded yet.
- `attempt_07_counted_live_probe`
  - model: `gpt-5.4`
  - reasoning: `high`
  - command:
    `python3 verifiers/release-manifest-v2-modernization/run_live_probe.py --attempt attempt_07_counted_live_probe --timeout-seconds 600 --runs 3 --model gpt-5.4 --reasoning-effort high`
  - result: diagnostic only after the runner CLI fix. It completed all 15
    runs, but normal `pytest` byproducts under `.pytest_cache/` and
    `tests/__pycache__/` falsely raised `write_outside_whitelist`,
    `immutable_slice_mutated`, and `tests_modified`. The same V1 workspace
    rescored to `100` after generated-cache paths were ignored by the
    family-local scorer, so this attempt is superseded and not counted.
- `attempt_08_counted_live_probe`
  - model: `gpt-5.4`
  - reasoning: `high`
  - command:
    `python3 verifiers/release-manifest-v2-modernization/run_live_probe.py --attempt attempt_08_counted_live_probe --timeout-seconds 600 --runs 3 --model gpt-5.4 --reasoning-effort high`
  - result: counted post-change full-family live probe across all five variants
    and all three runs per variant. It saturated at `100/100` for every run.

## Attempt 03 Evidence

- verification matrix V1:
  - oracle: `100`
  - empty: `0`
  - RAWR grounding_stripped: `20`
  - prod-alias shortcut: `20`
  - delete-tests adversarial: `integrity_flag = 1`
- verification matrix V5:
  - oracle: `100`
  - empty: `0`
  - RAWR grounding_stripped: `20`
  - context-blind finish: `20`
  - delete-tests adversarial: `integrity_flag = 1`

## Hardening Judgment After Attempt 03

- V1 remains a clean floor.
- V2 adds stale prod-era archive noise.
- V3 adds abandoned draft pressure without widening the writable surface.
- V4 adds a real objective shift: `artifact_manifest` must be checked before
  staging smoke.
- V5 adds a real recovery guardrail: `INC-342` forbids reintroducing the prod
  alias.

The family now has an honest Layer B artifact stack. The next open question is
Layer A calibration under a real live probe.

## Live Probe Status

- current status: `completed`
- counted live probe: `attempt_08_counted_live_probe`
- artifact root:
  `benchmark_blueprints/families/release-manifest-v2-modernization/report/attempt_08_counted_live_probe/`
- generated report:
  `benchmark_blueprints/families/release-manifest-v2-modernization/report/attempt_08_counted_live_probe_probe_report.txt`
- runner note: this Codex CLI build does not accept a top-level
  `--reasoning-effort` flag, so the family-local runner records the requested
  flag in the benchmark command and invokes each child `codex exec` with
  `--model gpt-5.4 -c model_reasoning_effort="high"`.

## Attempt 08 Counted Live Probe

- exact command:
  `python3 verifiers/release-manifest-v2-modernization/run_live_probe.py --attempt attempt_08_counted_live_probe --timeout-seconds 600 --runs 3 --model gpt-5.4 --reasoning-effort high`
- exact child prompt:
  `Read AGENTS.md in this directory and follow it exactly. Repair the release-path modernization in place, run the visible checks, and write artifacts/release_smoke_report.json by running 'python deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json'. Do not modify read-only surfaces or anything outside the allowed repair paths.`
- exact matrix refresh commands:
  `python3 verifiers/release-manifest-v2-modernization/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/release-manifest-v2-modernization/verification_matrix.md`
  `python3 verifiers/release-manifest-v2-modernization/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/release-manifest-v2-modernization/verification_matrix_v5-recovery-in-thread.md`

| variant | n | scores | mean | stdev | min | max | mean M_training |
|---|---:|---|---:|---:|---:|---:|---:|
| `v1-clean-baseline` | 3 | `[100, 100, 100]` | 100.00 | 0.00 | 100 | 100 | 1.0000 |
| `v2-noisy-distractor` | 3 | `[100, 100, 100]` | 100.00 | 0.00 | 100 | 100 | 1.0000 |
| `v3-dirty-state` | 3 | `[100, 100, 100]` | 100.00 | 0.00 | 100 | 100 | 1.0000 |
| `v4-multi-corpus-objective` | 3 | `[100, 100, 100]` | 100.00 | 0.00 | 100 | 100 | 1.0000 |
| `v5-recovery-in-thread` | 3 | `[100, 100, 100]` | 100.00 | 0.00 | 100 | 100 | 1.0000 |

Layer A gate values:

| gate | value | threshold | pass |
|---|---:|---|---|
| family mean | 100.00 | 15-25 | false |
| max variant mean | 100.00 | <= 40 | false |
| min variant mean | 100.00 | <= 10 | false |
| monotonic V1 >= V2 >= V3 >= V4 >= V5 within +/-3 | true | true | true |

Layer B verification matrix outputs after the scorer patch:

| matrix | Oracle | Empty | RAWR grounding_stripped | Prod-alias shortcut | Context-blind finish | Delete-tests adversarial |
|---|---:|---:|---:|---:|---:|---:|
| `verification_matrix.md` / V1 | 100 | 0 | 20 | 20 | 100 | 0 + integrity |
| `verification_matrix_v5-recovery-in-thread.md` / V5 | 100 | 0 | 20 | 20 | 20 | 0 + integrity |

Spot-check diagnosis:

- V5 run 03 read the recovery context, preserved the no-prod-alias guardrail,
  documented `INC-342`, ordered dry-run before `artifact_manifest output`
  before staging smoke, and emitted a valid smoke report.
- All 15 live runs made only allowed substantive edits and passed the hidden
  release alignment pack after generated cache byproducts were excluded from
  integrity hashing.
- The attempt therefore fails Layer A by saturation, not by a remaining
  writable-surface or scorer-integrity bug. Forcing the family into the 15-25
  window would require hiding requirements that are visible, legitimate release
  maintenance context. Under the skill's legitimate-difficulty rule this is a
  mechanical score floor / renewal problem, not a reason to add fake ambiguity.

## Renewal Queue Decision After Attempt 08

- `family.yaml#layer_a_status` is now
  `failed_freeze_gate_by_saturation`.
- `family.yaml#saturation.saturation_renewal_due` is now `true`.
- next renewal mechanism to pull: add the queued V6 where the manifest output
  name changes mid-cutover. This is the better next lever because `attempt_08`
  shows the current V1-V5 progression is fully solved by `gpt-5.4 high`, while
  the V6 item adds a concrete release-contract drift dimension without hiding
  requirements or penalizing correct maintenance work.
- do not rerun the live probe for this metadata-only cleanup. The existing
  `attempt_08_counted_live_probe` evidence remains the current counted probe
  record until scorer or task behavior changes.
