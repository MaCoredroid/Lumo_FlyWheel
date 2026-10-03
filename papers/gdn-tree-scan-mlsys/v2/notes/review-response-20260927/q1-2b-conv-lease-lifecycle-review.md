# Q1.2b pre-fixture conv-lease failure — bounded lifecycle review

**Disposition:** the failed first calibration attempt is an initialization failure, with no candidate/reference fixture result. Repair the component harness to establish the existing production operand lease before calling warm. Do not skip warm, weaken the kernel guard, edit private lease state, change numerical policy, or reuse the failed run directory. No new execution is authorized by this note.

Run `q12b-calibration-20260928T002602Z` has terminal status `PROCESS_A_FAILED_rc=1`; A exit 1, B absent, reducer absent, tensor-store objects/bytes both zero. The local log reaches fixed32 committer graph preseed, then fails in runner `Backend.__init__` line 193 at kernel line 15793: `FR13 fixed32 postprocess boot warm has no clean conv lease`. The check is before fixture reference/candidate phases. Result-audit preparation was paused in favor of this parent-requested diagnosis.

## Exact source/evidence identities

Repository-relative paths:

| Path | SHA-256 |
|---|---|
| `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` | `d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8` |
| `scripts/fr10_phase4_patch_vllm_tree_gdn.py` | `c382ef22a5a8d4e84209e6723f8eea21c3819543c6e942c62951385b5c1c003e` |
| `src/lumo_flywheel_serving/fr13_tree_conv_fused.py` | `564c7dc7ba61ccd9579f2fb7a4af9a2cdaaa1753dfb98434f2011457d90d9a6e` |
| `tests/test_fr13_fixed32_conv_commit_cuda.py` | `d22c8ff27034771e1c292b40225a5e15ac987e687893df1ecf2bdc2aa09d466a` |
| `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_component_runner_v2.py` | `cbb754e8d306758f5d11ab65fca4d533614839980d32ff16f82d76fc7a885544` |
| Failed run `procA.log` | `e62006fccc25d7775a365493ec89e3dd1431d7538c57eb152c5cda65388a311f` |
| Failed run `RUN-RECEIPT.json` | `1e7d2b3be65ed8193c380cbb79b3b81e85fd670d5dae2d01ab13a538c5647782` |

The first kernel hash is exactly the gate-bound production kernel. Patcher/test/conv-helper files above were read as implementation references, not run or newly qualified.

## Why adding independent conv buffers is insufficient

The public `preseed_fixed32_conv_col0_pregather` API (kernel 7972–7997) calls `_validate_fixed32_conv_pregather_preseed` (7737–7968). That validator requires:

- Exactly 48 conv and companion SSM bank entries; exactly three registered builder SSI groups covering 48 unique layer names, on one CUDA device and covering capacity B=1.
- **16 exact conv-pointer aliases of width three**, each spanning the three SSI groups; the SSM pointer alias classes must be identical. Each conv/SSM pair must share the same underlying storage pointer. Partial overlaps between distinct conv spans are forbidden. Thus the current 48 independently allocated SSM banks cannot satisfy this contract.
- Conv banks share dtype, shape and stride; row layout is dense within a page, all strides positive and pointers 16-byte aligned. The SSM **FP32** state shape/dtype must remain the reviewed `(rows,48,128,128)` contract; a toy BF16 SSM test view is not a substitute.
- The commit index tensor must be contiguous int32 **`(48,B,32)`**, while the failed harness allocated `(48,1,2)`. Accepted paths are contiguous int32 `(B,16)` and lengths `(B,)` (8108–8140).
- There must be 48 distinct, contiguous, aligned source stagings; an int64 contiguous state-source index vector of length `32 * conv_l`; values must cover the declared source-row range (8017–8060).

The existing deployed-shape example is `tests/test_fr13_fixed32_conv_commit_cuda.py:25–68,90–195`: sixteen shared page buffers, each conv/SSM view repeated across three groups; deployed conv channels/length `10240/34`, source rows `36`; the state-source map comes from `build_tree_conv_state_src_indices(parent, width=4, state_len=34)` in `fr13_tree_conv_fused.py:110`. Its helper uses a small BF16 SSM shape for a conv-only test, so reuse its **layout construction pattern**, not its toy recurrent dtype/shape or test monkeypatches. Production call wiring is in the patcher at 16867–16992.

## Smallest faithful harness repair

1. Construct and retain the production-compatible shared conv/FP32-SSM page views and three SSI-group tensors. Use the same persistent 48-entry tuples and commit metadata tensors for preseed, lease audit, warm, and measured GDN publication. Do not rebuild equivalent tuples/views between calls: warm checks `conv_state['ssm_banks'] is route['banks']` and commit-index object identity (15759–15767).
2. Assign distinct active run rows to the three alias ranks before fixture initialization/publication. Revise the harness's `RUN_ROW`, restore, output-readback and control-row expectations consistently. The failed code uses row 1 for every layer; after aliasing, that would overwrite three fixtures' states in one physical row. Control sentinels must be defined per physical storage/row rather than assigning conflicting per-layer values to an aliased row. Keep a reserved scratch row and sufficient warm rows; warm requires at least rows 1–3 for B1, and no active/control collision.
3. Public initialization sequence: register three groups with `register_fixed32_conv_col0_ssi_group` (7683–7735); preseed the existing GDN committer graphs; call `preseed_fixed32_conv_col0_pregather` with the same SSM tuple/commit metadata and persistent source staging; run `audit_fixed32_conv_commit_lease` (9060 onward); then invoke the unchanged `warm_fixed32_committer_graphs_all_batches`. Production also calls `selfcheck_fixed32_conv_col0_ssi_sources` for the served capacity after preseed (patcher 16981–16989); reuse that existing check if the harness exposes those SSI sources.
4. Preserve the actual warm receipt. It must report `ready`, `classification=unmeasured_boot`, `route_lease_current`, `bank_state_restored`, `conv_bank_state_restored`, `conv_staging_state_restored`, `input_state_restored`, and `measured_state_restored`. Warm executes the guarded conv commit and recurrent replay on isolated alias-rank rows, then restores inputs, affected rows, scratch metadata, callbacks and measured counters (15888–16057); evidence is assembled at 16072–16122. Check measured GDN replay and conv commit counters are still zero afterward. Do not relabel warm activity as fixture candidate evidence.
5. Record this as **conv lifecycle setup**, not conv numerical qualification. The former `no_conv=true` attestation becomes literally inaccurate if the public warm executes a conv commit; replace it with an explicit boundary such as “conv warm/setup executed; no conv qualification cases.” The scientific fixture denominator remains the same GDN state/output cases, and publication equality remains the existing paired policy plus causal controls.

If importing `fr13_tree_conv_fused.py` to build the exact source map, include its source bytes in the new reviewed dependency map. Otherwise bind an independently checked materialization to the same parent topology; do not silently expand an unbound runtime import. Version and reseal the harness/launcher/manifest changes; preserve all original failure receipts and the prior freeze.

## Focused pre-execution controls

Before another parent decision, the worker can use CPU layout/alias tests and source checks to demonstrate: 16 groups × 3 aliases with matching conv/SSM storage ownership; three distinct logical run rows per alias; `(48,1,32)` metadata; no overlapping conv/SSM regions within each page; immutable object reuse; correct public API order; and refusal for an absent group, wrong alias or wrong companion binding. Existing negative examples are `tests/test_fr13_fixed32_conv_commit_cuda.py:587–672` (read only here). A CPU stub must enforce these real lease requirements rather than returning unconditional warm success. This avoids repeating the blind spot of the initial stub harness; it is not a substitute for subsequent actual initialization evidence.

No kernel/policy modification, GPU run, remote operation or source edit was performed for this diagnosis. The frozen first-run evidence remains untouched. Fresh initialization and calibration execution, if any, remain separate parent approvals.
