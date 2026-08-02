# Fixed32 GDN single-launch v2 live qualification readiness

This is reduced static-readiness evidence for the default-off
`fixed32_gdn_single_launch_tree_v2` candidate. It is not a byte PASS, serving
credential, timing result, acceptance result, or production authorization.
No Docker container, GPU kernel, SWE-Verified task, timing arm, capture-only
probe, speed probe, or synthetic probe was launched while preparing it.

## Bound qualification

- implementation commit: `10538aea58caa52ce815a978140c2f3ce3664bfc`
- kernel source SHA-256: `ca5ff6496c7cf3221996e6aa5971d36207e305e51f5c4a308f71d15165ab659a`
- sm_121a audit checksum-manifest SHA-256: `bf5dafec70ee932b9d3aa7043c1f7e78014ecf783222b3fbfcf76cf9816e260d`
- physical rows per request: `32`
- draft vocabulary: `K=65536`, root enabled
- block map SHA-256: `85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff`
- graph comparison: 48 layers, exact raw bytes for `out`, `ring_k`,
  `ring_v`, `ring_a`, `ring_b`, `flags`, and `counter`
- B1 diagnostic: the first canonical exact4 SWE-Verified task, batch and
  concurrency `1`
- decisive B4 gate: canonical exact4 SWE-Verified tasks, batch and concurrency
  `4`

The gate always serves the incumbent two-launch result. Candidate state and
all authoritative surfaces are restored after every comparison, and candidate
bytes cannot be served while either qualification selector is armed. Production
remains default-off. The emitted live verdict is byte-only and is explicitly
ineligible for timing, floor acceptance, or production enablement.

## Launch commands

Provision the ignored pinned runtime inputs as physical regular files, use a
new `RUNROOT` below `output/`, and point `FORKED_FA2_SO` at the exact stock file
with SHA-256
`f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`.

```bash
RUNROOT=output/<new-b1-runroot> TAG=<unique-tag> FORKED_FA2_SO=/absolute/path/to/stock-fa2.so bash scripts/fr13_run_b1_gdn_single_launch_live_gate.sh
RUNROOT=output/<new-b4-runroot> TAG=<unique-tag> FORKED_FA2_SO=/absolute/path/to/stock-fa2.so bash scripts/fr13_run_b4_gdn_single_launch_live_gate.sh
```

`PYTHON_BIN=/absolute/path/to/python` may be supplied when the worktree does
not have a physical `.venv/bin/python`. The runners fail closed on missing or
symlinked pinned inputs, a dirty tracked worktree, any existing Docker
container, non-canonical task/block-map/FA2 bytes, probe selectors, source
drift, incomplete authenticated task lifecycles, or a missing exact live PASS.

## Static verification

- focused and related tests: `102 passed`
- runtime-manifest isolated self-test: pass
- full pinned-input runtime-manifest preflight: 87 files, canonical SHA-256
  `8b0ae02a3dda01b63b353ad8d3f1ed72ed25aa360f06846f57ea86d4dfe8905a`
- changed verifier/tests Ruff check: pass
- changed Python AST parsing and shell syntax: pass
- wrong-FA2 runner preflight: rejected before runroot creation

The ignored `.lumo.local.env`, pinned SWE-Verified cache blob/ref, and prompt
file used for the manifest preflight were removed afterward. Their contents and
the raw runtime manifest are not included here.
