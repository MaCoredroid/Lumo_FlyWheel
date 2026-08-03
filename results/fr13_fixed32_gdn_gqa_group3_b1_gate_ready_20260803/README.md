# Fixed32 GDN GQA-group3 B1 real-gate readiness

Status: **source-bound one-real-task B1 byte gate ready; default off; GPU gate
not run**.

The dedicated entrypoint
`scripts/fr13_run_b1_gdn_gqa_group3_live_gate.sh` bakes the exact
`hydra27_fixed32`, B1, `gqa_group3` tuple. The shared runner rejects any
candidate/entrypoint mismatch before creating a run directory. It also pins
physical32, K64/root1, BV8, the canonical one-task SWE-Verified subset, the
stock FA2 binary identity, and clears the GQA3 production arm.

The graph comparator runs the two-launch incumbent, snapshots and restores all
mutable state, runs GQA3 on the same captured operands, compares output plus
ring/state/counter surfaces byte-for-byte, restores the incumbent state in a
`finally` path, and serves only the incumbent result. The reducer joins every
comparator event to authenticated proxy/engine request ledgers, task
completion, graph census, launch/end manifests, the exact Git commit, and the
combined incumbent-plus-GQA3 source hash before issuing a credential.

## Gate command

Run only from a clean checkout of the committed branch after all existing GPU
and Docker work has finished. `STOCK_FA2_SO` must name the absolute regular
file with SHA-256
`f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`
and size `299183936` bytes.

```bash
TAG="fr13_gdn_gqa3_b1_byte_$(date -u +%Y%m%dT%H%M%SZ)"
RUNROOT="output/$TAG"
PYTHON_BIN=.venv/bin/python \
RUNROOT="$RUNROOT" \
TAG="$TAG" \
FORKED_FA2_SO="$STOCK_FA2_SO" \
bash scripts/fr13_run_b1_gdn_gqa_group3_live_gate.sh
```

On PASS, the source-bound credential is written below the new run root at:

```text
hydra27_fixed32_hydra27_gdn_gqa_group3_b1_${TAG}/hydra27_gdn_gqa_group3_b1_credential.json
```

No GPU kernel, service, Docker container, real SWE-Verified task, raw-byte
comparison, timing campaign, TPS measurement, acceptance measurement, or
hardware-floor test was run for this readiness artifact. Raw logs, prompts,
responses, patches, credentials, environment data, process identifiers,
container identifiers, and host identifiers are not published.
