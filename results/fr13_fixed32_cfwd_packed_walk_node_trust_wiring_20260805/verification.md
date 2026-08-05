# Verification

The focused suite covered the source candidate, CPU oracle, exact selector,
default-off behavior, source identities, runner, existing CFWD decision path,
live-gate contract, and comparator wiring:

```bash
PYTHONPATH=$PWD pytest -q \
  tests/test_fr13_fixed32_cfwd_packed_walk_node_trust.py \
  tests/test_fr13_fixed32_cfwd_packed_walk_node_trust_wiring.py \
  tests/test_fr13_fixed32_cfwd_logit_direct_decision.py \
  tests/test_fr13_fixed32_cfwd_logit_direct_runners.py \
  tests/test_fr13_fixed32_cfwd_logit_direct_live_gate.py -x
```

Result: 78 passed.

A broader launcher/provenance selection reached 215 passes and one skip. It
then stopped in `test_fixed32_runtime_manifest_includes_qwen_settings` because
this isolated worktree does not contain the required local Hugging Face
SWE-Verified cache blob. The failure occurred while opening that external cache
path, before any assertion related to this wiring.

`bash -n` passed for the generic launcher and new real-task runner. `py_compile`
passed for the runtime overlay, wrapper, runtime manifest, wiring test, and
artifact verifier. `verify_wiring_artifact.py` validated every pinned source
hash and the selector/runner/comparator contract. `SHA256SUMS` verification also
passed.

No GPU API, Docker, service, real task, request, response, timing, or acceptance
path was used. Runtime correctness and speed remain unmeasured.
