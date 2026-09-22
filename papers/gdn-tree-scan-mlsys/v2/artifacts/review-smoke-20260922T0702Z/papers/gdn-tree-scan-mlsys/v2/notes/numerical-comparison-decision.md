# Numerical comparison and experiment decision

Updated 2026-09-21 after the user's request to retain the discussion in the paper and finalize a bounded experiment list. No new model experiment was executed. The user supports spending on bounded experiments; this update records the selected design, not a result.

## What the repository establishes

1. The old local WY state-write path reused an excessively rounded output basis. Its archive reports roughly 1.66e-3 state error, 3.32 maximum logit discrepancy, and 56% rejection before the state-path repair. The repaired isolated state error must not be conflated with completed end-to-end validation. Source: [failure A8](../../../../FR13_REPLAY_CHASEDOWN_BANK.md), lines 126-133.
2. The June 9 correction explicitly keeps WY open at the acceptance/reference-floor bar and describes the spine's 6/6 argmax result. It rejects treating the absence of byte equality as a general mathematical failure. Source: [corrected verdict](../../../../docs/archive/wy/FR13_WY_VS_SEQUENTIAL_VERDICT.md), line 3.
3. The later diffuse-drift investigations also examine our sequential tree implementation. They describe accumulation across model layers, reference-kernel differences, and state/output distinctions. They are not tests of the TreeWY authors' code. Sources: [diffuse GDN](../../../../FR13_DIFFUSE_GDN_EXPLAINED.md), lines 11-39; [realization agreement](../../../../FR13_REALIZATION_AGREEMENT.md), lines 20-35 and 85-107. Historical causal interpretations should be checked against later repairs before reuse as current-system claims.
4. The August 28 reconciliation reaffirms that WY was parked, notes the finite tested spine agreement, and walks back a practical correctness-lead claim. This later assessment constrains interpretation of the earlier failure reports. Source: [pass 248](../../../../results/fr14_nvfp4_port_20260816/REDTEAM_20260816.md), lines 7363-7385.
5. Current source shares the sequential node-update body between scan and replay, while explicitly requiring compiled GPU agreement checks. Source: [kernel](../../../../src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py), lines 10425-10435 and 11768-11785.

The evidence manifest hashes these documentary/code sources as H6. This is not a fresh reproduction of their original tensor captures.

## Literature and our interpretation

[Bole v1, Sections IV-V](https://arxiv.org/html/2608.01651v1#S4) uses an ancestor-masked correction system, finite-Neumann evaluation, value tiling, compact accepted-state reconstruction, and hardware-calibrated verification budgeting. [TreeWY v1](https://arxiv.org/html/2608.20961v1) supplies the triangular-solve/reconstruction comparison.

Our interpretation: real-arithmetic equivalence alone does not settle numerical agreement with a particular native speculative-update or recurrent-decode path. The actual precision, rounding boundaries, and reference must be recorded. Conversely, an error in our prototype, or mismatch against our fork, does not prove a universal defect in the published method. Local arithmetic, full-model propagation, actual device sampling, and durable-state continuation need separate measurements. Faster commitment is a hypothesis worth testing, not an existing result.

## Selected scope and reasons

- P0: provenance audit, including effective temperature and actual compiled routes. Reuse archived accounting; no inference needed for this audit.
- E7a: three local mechanisms on identical captured operands: sequential replay, corrected triangular-solve reconstruction, and finite-Neumann reconstruction. Use distinct high-precision and native-GPU references. Isolate solver and commit arithmetic. Report local latency and memory only for actual device implementations.
- E2/E7b: one shared stage-isolation/continuation harness plus existing correctness fixtures. Compare verification-only, commit-only, and combined changes on fixed prefixes; then check actual sampling, accepted-prefix state, and next-step behavior. Use eight pilot prefixes and a separate 32-prefix confirmation set; these sizes do not imply quality-equivalence power.
- E1: native MTP-5, depth-matched native chain, and frozen sequential tree; 18 initial timing cells. Add at most one reconstruction candidate after correctness qualification, retaining the sequential tree control: 24 cells.
- E3: conditional four-arm scan/acceptance-walk composition, initially 16 paired timing cells, only if these unfinished optimizations are to enter the final implementation.
- Defer E4 full serving, broad E5 task quality, E6 NVFP4, and E7c authors-system port. Exact sampler fixtures remain in E2.

Ordering is P0 -> E7a -> E2/E7b -> optional E3 -> E1. A candidate that fails the declared agreement criteria is documented and excluded from deployment claims; the existing sequential system still receives the matched performance evaluation. The diagnostic pilot establishes measured runtime and variance before scaling the confirmation matrix. Implementation effort is recorded separately from GPU execution time.
