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
