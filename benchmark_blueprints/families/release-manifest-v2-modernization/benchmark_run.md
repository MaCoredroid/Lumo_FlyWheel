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

- current status: `pending`
- exact state:
  - one pre-fix live run (`attempt_04` / `v1-clean-baseline`) completed at
    `20/100` and identified the hidden-path blocker
  - the blocker was removed by moving the writable config surface to
    `codex/config.toml`
  - the post-fix live rerun was started but not completed before this turn
    ended
