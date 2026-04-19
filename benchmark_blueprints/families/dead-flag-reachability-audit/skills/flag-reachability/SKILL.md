# Flag Reachability

Use this skill when the task asks whether a feature flag or rollout control is live, partial, or dead.

## Objective

Distinguish config and parser visibility from actual behavioral reachability.

## Procedure

1. List the named controls exactly as given.
2. For each control, inspect defaults, env parsing, runtime branching, tests, and docs separately.
3. Track aliasing explicitly. A live alias is not the same as a live standalone flag.
4. Record one false-positive path per control when available.
5. Keep cleanup recommendations narrower than the evidence.

## Guardrails

- Grep counts are not reachability proof.
- Tests that exercise stubs or smoke paths do not automatically prove production liveness.
- Docs can be stale in either direction.

