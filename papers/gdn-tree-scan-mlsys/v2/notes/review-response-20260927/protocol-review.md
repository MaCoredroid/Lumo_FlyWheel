# Independent Q1 protocol review

27 September 2026. Reviewer role only; no GPU launches, remote writes, or runtime edits. This is separate from the executor's protocol and the parent-owned `GATE-Q1.json`.

## Current verdict

**REVISIONS REQUIRED — first protocol draft reviewed at 20:36–20:40 UTC.** No omnibus Q1 launch approval. Matrix corrections and identity preparation are useful, but the numerical policy still contains an unjustified fp32-to-bf16 perturbation and the proposed GPU/full-model adapters do not yet exist. CPU preparation can continue; a separately bounded native-only characterization can be reviewed once its executable runner, frozen fixtures and exact outputs exist. See the hash-bound review below. The parent gate remains false.

The canonical current-deployment image is immutable ID `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`, configured digest `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`, recorded vLLM `0.19.2rc1.dev134+gfe9c3d6c5`. The earlier fr9iso image used in author component smoke cannot qualify this deployment. This binding comes from the live campaign handoff, not an inferred Docker tag.

## Static checks completed while the protocol is drafted

The following are concrete constraints from the current source, beyond the initial qualification red-team inventory. Paths are repository-relative.

1. **Preserve storage identity in the paired diagnostic.** `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py::_fr13_fixed32_committer_fast_state` (13966) requires the exact persistent 48-bank tuple and all preseeded operand tensors. Direct-metadata mode also pins accepted-path storage. `preseed_fixed32_committer_graphs_all_batches` (15480) records these identities. The diagnostic must snapshot/restore tensor contents in place, rather than substitute clones or remove the guards. Independent reference tensors can be separate, but the candidate must execute the real entry point with its real bound storage.
2. **The conv lease is part of the contract.** `_fr13_fixed32_committer_replay` (16127) consumes a preceding conv validation lease keyed to pointers, shapes, strides, dtype, device and stream. `_fr13_fixed32_committer_metadata_lease_key` (13220) and consume-once helpers (13273–13301) make the coupling explicit. A standalone graph-body invocation does not exercise that composition. Run a missing/mismatched/repeated lease negative if that mode is active.
3. **Native replay is not the final observation boundary.** The generated `_fr13_fixed32_device_commit_route` in `scripts/fr10_phase4_patch_vllm_tree_gdn.py` (around 19979) consumes the five TAW products and publishes the running-state path. Target KV remap around 41600 stages a request-keyed, forward-indexed MTP payload containing the physical source permutation. Drafter consumption/slot restoration follows. Check the boundary after this complete sequence and then execute the next target forward. State snapshots taken immediately after recurrent replay miss later ownership and KV errors.
4. **Retain source and execution census.** The KV stage requires exactly sixteen target caches and one drafter cache, rejects a repeated completion event, and reconciles actual replay count before setting `target_kv_complete`. A standalone fake cache tuple or manually fabricated event record can test a helper but cannot be presented as a live-current-route event.
5. **Restore numerical and lifecycle state separately.** Snapshot/restore helpers in the existing layer-batch shadow gate (15384) are good patterns for tensor restoration, but that candidate is disabled in the current deployment. Warmup, capture, diagnostic replay, and actual continuation can also mutate acceptance counters, pending-event records and consumed leases. Declare which are restored or kept in a diagnostic namespace; never erase a real failure or count a shadow as an agent event.
6. **Use truthful workload identity.** `scripts/fr13_patch_fa2_tree_bias.py` admits explicitly named Tier-B workloads, including `random1024_calibration`; `tests/test_fr14_fa2_tierb_qualification.py:3347` tests that exact no-subset route. Do not label recorded agent-prefix diagnostics `random1024_calibration` merely to pass launch pins. A new diagnostic workload needs explicit provenance and a scoped adapter/launch policy, with existing source guards preserved and changes recorded.

The current pure topology authority was executed locally with bytecode writes disabled to check the matrix denominator. It yields **28** materialized root-to-node cases including root-only. Draft accepted-length counts are `1:3, 2:5, 3:5, 4:4, 5:4, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1`. Root-inclusive inactive physical IDs are **18, 23, 25, 27**; the deepest valid endpoint is 31. The executor's matrix must distinguish root-inclusive physical node IDs from draft-only row IDs, whose indices differ by one.

## Conditions for protocol approval

Approval requires reviewing the concrete protocol, machine-readable case matrix, adapters and source manifest together. Each item must link to an executable implementation or an explicitly staged, independently gated step:

- Current immutable image; loaded native recurrence and patched modules/binary hashes; actual Hydra27/two-level/FA2/native-replay route checks. Reject Cat10 or a single-launch candidate as a substitute.
- Candidate-blind calibration. Case split and scales/floors are fixed before candidate measurements; candidate errors cannot set the thresholds applied to themselves. Reused FA2 tolerances remain component-scoped.
- Exact structural/copied-byte checks; independent sequential reference construction; named actual native reference; nonfinite and untouched-row checks; powered negative controls with required nonzero failure.
- Explicit root/materialized/pending token semantics; every active accepted path or a justified declared coverage denominator; stale/padded/short-after-long and repeated-state cases.
- True next-forward logits using the model and published recurrent/conv/KV/drafter boundary, including graph and cache lifecycle. Random projection or saved scan output is not this test.
- Fail-closed handling of required missing capture, skipped GPU case, incomplete layer set, mismatched image, interrupted job and source drift.
- Correctness and diagnostic timing clearly separated. Q1 protocol approval alone is not Q1 result acceptance or permission for M1/WP/WC.

The reviewer will append a hash-bound verdict once the worker material is available. Only the parent changes gate files.

## Matrix draft review, first version (20:26–20:29 UTC)

**REVISIONS REQUIRED; protocol/adapter approval remains pending.** Reviewed remotely without mutation:

```text
849767bf7dfc9c6c0a1a4c37951beab120b20ece040fb65cd95952a10b550061 q1_case_matrix.json
5879e2859f83c19ad0edadc36b4f30edfaa7dbf5d46f805d5419515fd34ea88c tools/build_q1_case_matrix.py
```

The authority-derived F1 path set is correct: 28 cases, proper ancestor chains, expected inactive rows, and longest spine. Frozen F2 sequences have valid references and include long/short/root/off-spine transitions. These are useful declarations; no executable qualification harness is present in this reviewed snapshot.

Required corrections:

1. **Invalid pending-case reference.** F3-P1 hardcodes `F1-n09-0000`, which is absent from F1. The four-draft spine case is `F1-n14-0000`, root-inclusive chain `[0,1,4,9,14]`. Use `spine_by_depth[4]["case_id"]`; validate every cross-reference, including F3, before writing the file. The generator currently validates F2 references only.
2. **Neutral-tail off-by-one.** F4-B2 says staging positions `>= accepted_len` are neutral. In the actual fixed16 staging, position zero is the root and positions `0..accepted_len` are valid (`_fr13_fixed32_committer_graph_body` uses `path_offsets <= accepted_lens`). The neutral tail starts at **position > accepted_len**, equivalently `>= accepted_len + 1`. Keep that distinct from accepted-path metadata, whose active entries are positions `< accepted_len` and exclude the root.
3. **Do not assert an unimplemented redundant guard.** F4-B1 expects refusal when a padding node is forced into the commit path. Current `_fr13_fixed32_conv_commit_row_guard_kernel` (7264) validates owners/row ranges, accepted length and node range, but does not validate Hydra27 logical validity or ancestry. The correct negative is that the independent topology oracle rejects the bad result and that the active sampler never selects masked nodes. If commit refusal is desired, it is a new defense-in-depth requirement; absence cannot be reported as a regression of an existing guard.
4. **148 currently counts declarations, not executable cases.** Four input-class labels multiply F1/F2/F3, but the synthetic class contains three named stress strata without explicit expansion, input hashes, seeds or fixture IDs. F4/F5/N are unexpanded rule descriptions, and some reuse F1/F2 evidence. Before qualification, materialize unique case IDs with exact input fixture, path, sequence, mutation, comparison surfaces and expected verdict. State whether a case is one independent execution or a coverage assertion on another execution. Do not count each coverage assertion as a completed experiment.
5. **Bind the calibration/evaluation split before candidate evaluation.** `calibration-prefixes` currently names only “3 disjoint prefixes”; evaluation names only length ranges. Choose actual recorded prefixes and tokenization/checkpoint identity, freeze their content hashes and deterministic transformations, and validate no overlap at the chosen source/trajectory granularity. Hash-bound fixture discovery can be a preparatory stage, but this draft is not yet a frozen input split or a numerical policy.

These findings were sent to the parent as soon as identified. Full protocol, numerical gate order, source manifest and actual next-forward implementation remain unreviewed because they were not yet present.

## Protocol v1 and matrix v3 review (20:36–20:40 UTC)

**REVISIONS REQUIRED.** Read-only review of the following remote files:

```text
34f1e791ded4c07274b0715b5edda5094e62c3e4a64cd71385202311013f63aa PROTOCOL-Q1.md
df2a4c7b01a6ae09ef06c397f0bb0d2f084f40421b39e6d65f01284429173eed q1_case_matrix.json
b8fb930437a85e5ca884c05b818bfebbe84185ba13447a1153c3dc86735d5d2f tools/build_q1_case_matrix.py
c16ff62bfcff6086cda3941b5474f38325c316690295d3f88da71241c6c81fc7 SOURCE-RUNTIME-MANIFEST.json
```

The earlier F3 reference, neutral-tail off-by-one and invented commit-padding guard are corrected. Matrix v3 explicitly distinguishes scenario declarations from an executable denominator and marks the latter pending. The protocol now follows the actual publication boundary through target KV, deferred drafter consumption, slot restoration and proposal-end sealing. These changes address the initial review; they do not implement the test.

### Required revisions before numerical qualification

1. **Replace the logit tolerance perturbation (protocol line 73).** The deployed recurrent cache is fp32. Rounding it to bf16 is not an exercised storage precision, so its next-forward sensitivity is not a defensible allowance for this implementation. Candidate-blind calibration can still normalize away genuine divergence if its reference perturbation is too large. Use actual repeated native runs and explicitly supported native execution/precision variants; preserve a same-shape exact control. If a prospective tolerance cannot yet be justified, characterize raw logit errors without granting a numerical pass. A future bound based on measured state uncertainty must constrain a declared perturbation by independently established native error and include layer/surface-specific effects; it cannot choose an arbitrary lower precision. State the multiplier, absolute caps and near-zero floors before opening candidate results, and demonstrate detection of wrong-state negatives. Small-margin top-1 changes are observations, not automatic exemptions from the logit gate. Only define temperature-based distributions at positive temperature; temperature zero uses greedy decisions and margins.
2. **Separate mathematical truth from native rounding (line 53).** A C2 described as an independent float64 recurrence cannot silently compute sigmoid beta in fp32. Promote the supplied rounded operands to float64 before arithmetic in the mathematical oracle. If emulating native gate/conv rounding is also needed, implement and name a separate rounding-emulation comparator with every cast and operation order explicit. In particular, “bf16 taps then SiLU in float64” does not define the rounding points in the native conv update. Test both with small hand-computed cases, near-zero q/k, the softplus branch boundary and bf16 rounding-boundary inputs. Fix these reference definitions before candidate calibration; a later change requires a new protocol hash and rerun.
3. **Make calibration a separate gated execution (lines 69–85).** The heading promises a policy frozen before candidate evaluation, but the stage currently combines native calibration and C0 evaluation and freezes the policy merely before a candidate verdict. Split native-only characterization from candidate qualification. First review the actual runner, fixtures, native import/dispatch and a machine-readable output schema. Run only native/reference characterization, then freeze the exact policy JSON and hash through independent review. Only then run C0. Specify the formulas or fixed values for `roundoff_floor`, denominator floor and `cap_head`; “from calibration” is insufficient. Keep held-out trajectories disjoint at the source/trajectory level, not merely different slices of one prefix. Baseline-only calibration does not authorize a candidate-dependent tolerance revision.
4. **Resolve the cloned-bank contradiction (line 85).** R-SNAP forbids replacing a preseeded candidate bank or ring. Independent C1/C2 storage can be separately allocated, and a component candidate can be preseeded once against its own test storage. State exactly which tensor is cloned and when; restore candidate contents with `copy_` into the same persistent buffers thereafter. Exercise the real replay entry point and its conv lease. Calling the graph body directly or rebinding a bank after preseed is not current-route qualification.
5. **Keep native-from-candidate-rings narrowly scoped (line 85).** C1 replay from the same captured rings can establish replay/publication correctness. It cannot catch the candidate gathering the wrong branch's operands into those rings. The component reference must independently gather q/k/v/gates from the frozen raw operands and authority-derived path and compare the rings against those sources. The full-model native reference must independently process the common token history; it cannot consume candidate intermediate rings or candidate-derived metadata as truth. This distinction is necessary to detect the intended wrong-sibling and wrong-path negatives.
6. **Choose an executable full-model reference and align its observations (lines 47, 109–113).** The pending C1 mechanism and hook placement determine whether O1/O2 are meaningful. First require root-row next-forward logits from the same pending token, position and logical history. All 32 branch-row logits need 32 appropriately aligned sequential reference histories or an explicitly justified subset, not a comparison to one native token's logits. Compare same-input drafter scores and categorical top-k membership with a declared tie/margin policy. Do not treat token IDs as bounded numerical tensors. Keep logical token/position histories exact; cross-engine physical KV addresses and request-key representations may differ, so compare the ownership-preserving mapping rather than require raw address equality.

### Executable scope and feasible next steps

At this snapshot the remote campaign contains the protocol, matrix generator, source/runtime manifest generator, status helper and route-identity notes. It does **not** contain `q1_component_runner.py`, `q1_boundary_hooks.py`, `q1_reduce.py`, the execution-manifest generator, prefix/fixture manifests or the new oracle/fixture tests. Sections 8–10 openly list pending pieces. They can be reviewed as design, but their declarations cannot receive a launch/result approval.

The next reviewable unit should be small:

1. Implement the pure float64 oracle, optional named rounding emulator and deterministic synthetic fixture generator; CPU-test against independent hand cases. Freeze source hashes, seeds, geometry and raw fixture hashes. Implement a case-ID/coverage self-test that fails on missing required records.
2. Implement a **native-only** component characterization command in the immutable production image. Bind the loaded native function/module path and checksum; verify geometry and dtype; emit repeated-native exact equality and native-vs-oracle raw per-layer/head metrics. No C0 measurements, model boot, logits claim or speed claim in this step. An explicit scoped gate can approve this command after source review without approving full Q1.
3. Independently review its results and freeze the numerical policy. Then review and run the actual candidate scan/ring/publication component command, including conv-lease engagement and powered negative controls. Report its narrow credential.
4. Implement/review the actual engine hooks and independent native full-model reference, with the captured-prefix split and execution manifest. Only this stage can close compositional next-forward, graph/cache lifecycle, pending-token and drafter/KV qualification.

The identity manifest currently records the immutable production image, correct binary size/hash and honest source drift. Route parity still requires more than printing the same schedule name: record the loaded modules and active branch/geometry/committer contracts. A full-model diagnostic workload must retain its own identity and cannot borrow the SWE or random1024 credential. Source or numerical policy changes after review require a new hash-bound review, not silent reuse of this verdict.

## Bounded Q1.2a native-only characterization conditions

This proposed stage does not depend on the full-model U5/U8/U9 choices. It is **not yet launch-approved**: the executable runner and fixtures must first exist and pass review. It makes no candidate, timing, next-forward, model-quality or complete-Q1 claim.

1. Freeze an actual runner, deterministic raw fixture generator, CPU oracle tests and fixture manifest with hashes. Specify seeds, generation equations/ranges, actual stored dtypes, state scale strata, head mapping and active lengths including the root. Bind the loaded native import's file/hash/signature in the immutable production image; do not infer its casts from the source comments about an older native image.
2. Geometry is 16 K heads, 48 V heads, dimensions 128, fp32 recurrent cache. Source operands retain their real declared storage types. Each C2 operand is the exact value stored in the frozen raw input, promoted once to float64. For value head `h`, use q/k head `floor(h/3)`. Compute in float64:

   ```text
   qhat = q / sqrt(sum(q*q) + 1e-6)
   khat = k / sqrt(sum(k*k) + 1e-6)
   x = raw_a + dt_bias
   sp = logaddexp(0, x) if x <= 20 else x
   g = -exp(A_log) * sp
   beta = sigmoid(raw_b)
   Sdecay = exp(g) * S
   residual = beta * (v - Sdecay @ khat)
   Snext = Sdecay + outer(residual, khat)
   out = (Snext @ qhat) / sqrt(128)
   ```

   State orientation is V-by-K. No intermediate fp32/bf16 recast is part of this mathematical oracle. Any emulation of native casts is a separate named comparator. The GDN-only characterization can mark convolution explicitly out of scope until its exact tap, accumulator, output and SiLU rounding points are implemented. It must not pretend this omission is conv qualification.
3. Run only native+C2 calibration fixtures, at least eight native repeats in each of two fresh processes. Emit exact same-shape repeat equality, per-head/per-layer max-abs and RMS error, ULP/relative error diagnostics, raw tensors, commands and loaded native source identity. A differing native call shape is a separate numerical condition, not an exact-repeat replicate. There are no C0 measurements or candidate verdicts. Nonfinite data on declared finite-valid fixtures and missing records fail the characterization integrity check.
4. Native-only temporary banks can be independently allocated. Later candidate component tests must preseed once against their own persistent banks and restore contents with `copy_`, exercise actual guard/lease/replay entry points, and never substitute clones into preseeded bindings. No native-only result establishes engagement of Lumo's graph or publication path.
5. Characterization requires **no numeric pass tolerance**. It records the native precision budget. Only after these results are inspected may the independently reviewed, candidate-blind policy be frozen for the later candidate stage.

One concrete, deliberately strict recipe to propose **before** reading candidate outputs is:

```text
u32 = 2^-24               # binary32 unit roundoff
eta32 = 2^-149            # smallest positive binary32 subnormal
floor_rms(case) = u32 * RMS(C2(case)) + eta32
floor_max(case, head) = u32 * maxabs(C2(case, head)) + eta32
RMS(C0-C2) <= 1.10 * RMS(C1-C2) + floor_rms(case)
cap[stratum, layer, head] = max over frozen native calibration cases of
    (1.10 * maxabs(C1-C2) + floor_max(case, head))
maxabs(C0-C2) <= frozen cap[stratum, layer, head]
```

These are a proposed deployment acceptance criterion, **not** a proved error bound. The floor is tied to the actual comparison surface's storage precision: use binary32 for the recurrent cache, and separately declare a genuinely bf16 output surface rather than converting the cache to bf16. Native-only calibration freezes absolute caps and the admissible input/reference scale range per stratum. Evaluation outside those ranges is UNCOVERED, not an excuse to inflate a cap. Raw error is always reported. If native error is pathological, repeated native calls are nondeterministic, or the caps admit designated wrong-state controls, investigate the reference or reduce the claim; do not widen tolerances against candidate observations. A later policy change requires a new hash and held-out evaluation.

This recipe supplies reproducible floors without inventing a universal `1e-3`/`1e-2` allowance. It does not supply a model-logit gate; that separate gate requires the actual native full-model reference and its supported precision behavior. The parent may approve this bounded native-only command once its implementation meets the above conditions while leaving every candidate/full-model gate false.
