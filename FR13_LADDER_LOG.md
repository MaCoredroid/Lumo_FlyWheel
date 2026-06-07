# FR-13 ladder log — per-commit strict drift gates (verify-path + regular-decode)

Each entry binds a commit to its STRICT top-down ladder results. Two gates per commit:
1. **VERIFY-PATH** — tree verify (tree-bias) byte-exact to native-FA2-on-path oracle: input + every full_attn layer + final logits = 0.0, spine AND branch.
2. **REGULAR-DECODE (verifier-only)** — plain decode (no tree/spec/bias) with the forked FA2 == ORIGINAL pristine stock FA2: 0.0 at every layer.

A `FR10_ALLOW_LINEAR_FALLBACK` run is DIAGNOSTIC ONLY (GDN may go linear) → NEVER bound here. Both gates must be STRICT + 0.0 to record a PASS. (codex_fr14 appends; this first entry recorded by the monitor from verified artifacts.)

---

## Commit `2900203b` (fr13 handle suffix tree bias offsets) — forked FA2 tree-bias kernel

### Kernel smoke (direct CUDA, isolated)
- `suffix_offset_tree_bias_smoke`: **equal=True, max_abs=0.0** vs native-FA2-on-path oracle (whole tree, one FA2 call). PASS.

### Gate 2 — REGULAR-DECODE (verifier-only) — **PASS, byte-exact**
- Artifacts: `output/fr13_regular_decode_gate_latest.json` (`fr13.ladder_table.v1`), `output/fr13_regular_decode_full_attn_gate_latest.json` (`fr13.regular_full_attn_pair_table.v1`).
- Config: strict — plain decode, no tree/spec, no bias; forked `_vllm_fa2_C` vs **pristine stock** `_vllm_fa2_C`; B=1 eager.
- Result: input `torch_equal=true, max_abs=0.0`; **all full_attn layers passed=True**; final logits `torch_equal=true, max_abs=0.0`. **`passed=True`.**
- ⟹ the fork does NOT touch regular decode; the E5 lossless baseline (which runs the same forked FA2 at no-bias) is provably unmoved; the `apply_tree_bias` insertion does not perturb FA2 codegen/MMA.

### Gate 1 — VERIFY-PATH — **PENDING (rerun in progress)**
- The strict verify run captured full_attn stage outputs + spine logits, but the FA2 route bypassed the TREE_ATTN op-capture hook, so the low-level tree-op (q/k/v) for **branch**-path replay was not emitted. codex is patching the FA2 route to call `_fr13_tree_attn_op_capture(...)` before the forked FA2, then rerunning the strict verify gate (no `FR10_ALLOW_LINEAR_FALLBACK`). Spine full-attn byte-exact pending confirmation; branch-path oracle pending the rerun. NOT yet a PASS.

---

## Commit `fe21cb73` (FR13 WIP: route TREE_ATTN through forked FA2 + path oracle) — Gate 1 rerun result

### Gate 1 — VERIFY-PATH — **NOT byte-exact 0.0 (single-ULP fragment-grouping floor); strict config, NOT a PASS-at-0.0**
- **Strict config (verified):** `--attention-backend TREE_ATTN`, `FR13_FA2_TREE_BIAS=1`, `FR10_ALLOW_LINEAR_FALLBACK` **UNSET**, GDN tree branch active, all 16 `tree_attn_op.call*.pt` captured. Run dir `output/fr13_verify_strict_tree_20260607T091935Z` (forked FA2 computing, splice OFF — no reward-hack).
- **Bias is true `-inf`** (capture `tree_attn_bias` `uniq_nonzero=[-inf]`, `has_neginf=True`) ⟹ masked (non-ancestor) keys contribute `exp2(-inf)=0` **exactly**; `x+0.0` is exact in IEEE fp. So the residual is **not** a masking/bias bug.
- **Residual = a single-ULP MMA fragment reduction-grouping floor on scattered no-copy keys.** In the shared no-copy tree KV the ancestor keys sit at scattered slots (`TREE_PARENT=[-1,0,1,1,2,2,4,4,6,6]`; branch nodes 3,5,7,9 interleaved between spine nodes), so the path keys land in different accumulation lanes than the **packed** oracle's contiguous slots → fp32 non-associative partial sums differ by one ULP. Measured (call15 corrected oracle `fa2_path_refs_v2`):
  - **stacked-spine packed oracle (`tree_vs_fa2_spine`): `max_abs 0.00390625`, `nonzero: 1`** (one element, one bf16 ULP).
  - branch row 6 (path `[0,1,2,4,6]`): 1 nonzero, 0.0039. call10 branch row 9 (path `[0,1,2,4,6,9]`): 0.00098 (1 ULP).
  - per-row oracle row-0 `0.125/512-nonzero` is a **single-query-harness artifact** (row 0 is **exact** inside the stacked-spine oracle, which has only 1 total nonzero on row 6) — to confirm via the new-direction workflow.
- **Interpretation (red-team, monitor):** this is the **same** reduction-grouping irreducibility `FR13_FA2_TREE_BIAS_FORK_RESEARCH.md` pinned on Triton-vs-CUTLASS — and it is now shown **intrinsic to no-copy shared-KV itself**, independent of kernel. Literal **byte-exact 0.0 on scattered rows is unreachable without per-row contiguous repacking** (which is a design change / possible copy). It IS within the E5 self-noise floor (~0.059) ⟹ **distributionally/argmax lossless** (the theorem-backed branch gate, `[[reference_gdn_tree_branch_oracle_losslessness]]`).
- **Verdict:** **NOT a 0.0 PASS** under the literal byte-exact-0.0 gate. The single-ULP floor is the expected no-copy regime, not a bug. **Open gate-definition decision (user):** accept the 1-ULP grouping floor and proceed to the e2e deliverable gate (lossless within E5 floor + superset accept/event ≥ E5) vs insist on literal 0.0 via per-row gather. New-direction workflow launched to verify the floor is irreducible + confirm the row-0 artifact + produce the e2e measurement plan. NOT bound as a PASS.
- Gate 2 (regular-decode) unchanged from `2900203b`/`d2f1ba18`: byte-exact 0.0, still valid (no regular-decode code changed).
