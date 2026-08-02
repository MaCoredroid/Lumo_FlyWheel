# Verification

- Final gate validator: pass.
- Real task boundary: exactly one authenticated SWE-Verified task, resolved,
  with pre- and post-task metric snapshots present.
- Coverage: 25,056 records, 522 per layer, all 48 layers, and both byte
  surfaces in every record.
- Equality: zero differing bytes and zero shape or dtype mismatches.
- Serving: reference returned for every record; production and timing disabled.
- K64 route: one gather-shim engagement, one root-gather engagement, zero
  full-vocabulary or linear fallback, and the pinned block-map hash matched.
- Identity: source, runtime, and external launch/end manifests were byte
  identical.
- Lifecycle: launcher exited zero; teardown left zero Docker containers and
  zero GPU compute processes.
- Reduced-package policy: no raw task, model, request, response, patch, process,
  environment, container, or comparison-log data is included.
