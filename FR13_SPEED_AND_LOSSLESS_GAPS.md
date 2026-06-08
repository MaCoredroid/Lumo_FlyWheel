# FR-13 — two gaps the e2e must clear separately: (A) accept/event≥native ≠ drift0/lossless; (B) accept/event≥native ≠ speed≥E5

User raised 2026-06-08. Both are real; neither is implied by accept/event alone. Recording so we don't certify a win on the wrong metric.

## GAP A — "accept/event ≥ native" does NOT imply "drift = 0 / lossless" (the implication is one-way)
- **drift=0 ⟹ accept/event ≥ native** (superset-by-math: identical verify logits ⟹ identical spine accept decisions; branches add ≥0). TRUE.
- **accept/event ≥ native ⟹ drift=0** is FALSE. The rejection-sampling committer ALWAYS emits tokens distributed as the **tree's OWN verify logits** (that is the sampler's invariant — output ~ p_target regardless of q_draft). `accept/event` only measures how often the drafter's proposals AGREE with the tree's p_target.
- **Failure mode the user is right to fear:** a *self-consistently drifted* tree (drafter and verifier both see the same drifted state) can AGREE with itself MORE than native does → **higher accept/event while emitting the WRONG (drifted) distribution = LOSSY but fast.** A good accept/event number can MASK drift.
- **Consequence:** losslessness MUST be certified by **(1) verify-logit drift = 0** (top-down ladder, spine+branch) **AND (2) output bag-TV ≤ E5 self-noise floor (~0.059)**, measured INDEPENDENTLY. NEVER infer losslessness from accept/event ≥ native. The two gates stay strict and separate.
- Current state is the *opposite* sign (accept/event 1.11 < native 3.21): drift currently HURTS agreement (drafter proposed for native-like logits, tree verify is drifted → rejects). Driving drift→0 should raise accept/event back toward native — but when it does, we still prove drift=0 + bag-TV separately, not declare victory on the accept number.

## GAP B — "accept/event ≥ native" does NOT imply "speed ≥ E5"
TPS = (tokens emitted per forward) / (time per forward) = **(accept/event + 1) / forward_time**. Speed≥E5 needs `(acc_tree+1)/t_tree ≥ (acc_native+1)/t_native`. If our forward is slower (`t_tree > t_native`), accept/event must be proportionally HIGHER than native just to break even. So drift=0 (which only gets accept/event UP to ~native) is **necessary but not sufficient for speed** — we also need `t_tree ≲ t_native`.

Decompose the measured 5.4× TPS gap (tree 8.78 vs native 47.3 per-req; 2.67 vs 15.6 warm):
- ~2.0× from accept/event (native emits 4.21 tok/fwd, tree 2.11) — closes when drift→0.
- ~2.7× residual = **forward-cost gap** (tree ~240ms vs native ~89ms/forward). This is the part drift-fixing does NOT touch.

### Forward-cost gap, component 1 — MEASUREMENT CONTAMINATION (flaggable by reading the docker log; FIXABLE)
The tree speed arm (`output/fr13_argmax_e2e_20260608T055851Z/docker_after_tree_probe.log`) ran with:
- **`FR10_METRICS=1`** (should be 0 for a speed run — `feedback_flag_gate_metrics_reuse_infra`, FR13_FLAGS.md §D).
- **FR12 diagnostic capture hooks firing 1680×**: `"FR12 pre-conv/conv/h0/scan capture failed: Cannot copy between CPU and CUDA tensors during CUDA graph capture"` (`gdn_linear_attn.py:1079/2317/2534/3375`). These CPU↔CUDA copies add per-forward overhead and break CUDA-graph capture of the GDN ops → eager. The env flags `FR12_*_CAPTURE=` are EMPTY, yet the code fires → the capture hook is NOT fully env-gated (runs regardless). 
- Native arm: clean capture (8s / 0.40 GiB) vs tree (12s / **2.08 GiB** of graph noise). 
**⟹ the tree 2.67 TPS is a pessimistic, contaminated number.** The clean speed measurement (FR10_METRICS=0 + ALL FR12/FR13 capture code OFF/compiled-out) is unknown and must be re-run. This likely closes a large chunk of the 2.7×.

### Forward-cost gap, components 2-4 — INHERENT (under investigation by read-only agent a92797…)
2. **9-node tree vs 5-chain**: ~1.8× more query positions through attention+GDN per forward (bandwidth-bound ⟹ marginal, but nonzero). If branches add 0 accepts (they did: pos5-8=0), it is pure overhead — a topology cost.
3. **GDN tree kernel vs FLA chunked scan**: if our kernel ancestor-replays the recurrent state per node (~O(nodes×depth)) instead of a single chunked scan, that is real extra compute. (agent reading `fr10_phase4_patch_vllm_tree_gdn.py`.)
4. **CUDA-graph FULL-capture of the tree path**: known vLLM-0.19 risk that GDN/tree backends don't FULL-capture (`project_round_f_tree_delta_outcome`). If, even hooks-off, the tree forward can't FULL-capture while native can, the eager launch overhead is an inherent speed floor. MUST verify with a clean run.

## Action: the e2e re-run after the drift fix MUST also be the CLEAN speed run
When codex re-runs the e2e (post prefill-patch, drift→0): set **FR10_METRICS=0** and disable ALL FR12/FR13 capture code (not just empty env — ensure the hooks don't fire; `gdn_linear_attn.py` capture must be compiled-out/guarded), confirm the tree FULL-captures (no "capture failed" warnings, capture footprint ~native's), THEN measure accept/event AND TPS. Report BOTH gates: (A) drift=0 spine+branch + bag-TV≤floor (lossless), (B) accept/event≥3.21 AND TPS≥native (speed). A win needs all of A+B, not accept/event alone.
