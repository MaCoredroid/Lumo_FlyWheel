# Witness-to-O1 binding closure

**PASS for this exact offline-auditor repair.** Both false passes reported in `q1-candidate-raw-audit-v2-source-review.md` now refuse. The historical source, controls, and failure report remain preserved. This is a source/CPU result, not runtime, launch, or full-model qualification approval.

Auditor SHA256: `6e7db114b7f0bcfb5282f8880175e146e344d2213faf85e5aa1b19c3258c78c8` (16,124 bytes). Immutable snapshot and controls are under `p0/monitor/review-response-20260927/candidate-raw-audit-v2-binding-closure-review/`.

The sole semantic addition, `witness_o1_binding()`, runs after the existing native-state and witness validators. It retains the cache identity check, joins materialized logical positions with the captured interval, requires matching physical block IDs, and compares K and V independently using their distinct raw layouts. Both byte sources are reauthenticated. The complete logical intersection must be covered for every target attention layer; the reported interval and byte count exclude unmaterialized or uncaptured regions. Existing plan, phase, geometry, finite-state, and identity preconditions remain in the calling path. No hook or numerical-rule change is needed for this repair.

Independently executed against the frozen source:

- Original full-geometry per-record positive control passes. The former conflicting-root-byte and conflicting-physical-block cases now fail with their corresponding byte/block mismatch errors. The previous six missing/corrupt/identity/counter negatives still refuse.
- Parent's adapted per-record closure script passes; SHA256 `c3b533a5a8b16970f63469fb1a5a206d4a942a0b8b39c063921ca6ae38300fd5`.
- Parent's three-method byte-layout suite passes; SHA256 `f8f51fefaa85a14222d9be3cd68138df9cfd7a368a645b25aa8000366def3827`. Distinct nonzero K/V values cross a full block and valid partial tail; block drift, tail data, swapped planes, and extent errors refuse. A partial captured interval checks only its materialized overlap.
- Five further independent negatives refuse: mismatch in the last layer's tail V plane, missing materialized block, zero intersection, corrupt witness object, and corrupt O1 object. A self-consistent changed byte outside O1's materialized extent remains outside this helper's claim and reports `outside_intersection_checked=false`. The preceding witness audit still owns publication/untouched-byte checks over its broader captured window.

The three executed script logs, original reviewer reproduction, `BINDING-CONTROLS.json`, and `JOIN-EDGE-CONTROLS.json` are preserved beside the snapshot. All tests used standard-library synthetic raw objects; no Torch, GPU, cache, container, gate, launcher, or implementation operation occurred. The full 84-case job/runtime bridge was not exercised or approved by these unit controls. No remaining blocker was found in this bounded delta.
