# B1 M128 target plus SFWD exact4 boot failure

This bundle records the 2026-08-05 real SWE-Verified exact4 timing attempt from
source commit `a8a904ed6c27a6338d43151038c155ebb76e3656`.

The reference arm completed all four canonical tasks and measured 234.9222 ms
per step (1.9633x the 119.6580 ms mandatory-weight floor), 24.6158 full-step
wall TPS, 160.8235 ms verifier/SFWD, 37.5204 ms drafter, 20.5215 ms committer,
16.0567 ms other overhead, and 4.7828 accepted drafts per event.

The target plus SFWD candidate produced no performance sample. It failed during
FULL CUDA-graph profile capture before health with:

```
RuntimeError: FR13 SFWD conv/post-prep capture lacks preseeded output bindings
```

The following profile-scope error is secondary cleanup after that exception.
The container exited 1 and Docker reported `OOMKilled=false`. Therefore the
candidate arm is a source boot failure, not a speed result and not an OOM.

The launch/end runtime and external manifests are byte-identical. The retained
full Docker log is the authoritative traceback. `manifest.json` classifies the
reference and candidate separately so the valid reference cannot be mistaken
for a candidate measurement.
