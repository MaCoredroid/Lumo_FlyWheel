# E2 / E7b coverage map (existing fixtures vs. required combinations)

Drafted 2026-09-21 from test headers and names in `tests/` at HEAD `984f613d` (CPU reading only; nothing executed here). "Existing" means a test file exists and names the case; it does NOT mean it was run on the current GPU route in this worktree. Every row must be executed post-boot on the actual graph route before it counts (review-experiments.md: "Source availability alone is not a passing execution record").

| Required combination (review-experiments.md E2) | Existing fixture(s) | Executed where | Gap |
| --- | --- | --- | --- |
| Sibling wins / sibling rejected / deep path wins | `test_fr13_replay_reference_bitexact.py::test_replay_chain_matches_scan_chain_all_paths_deployed_tree`; `test_fr13_conv_committed_path.py::test_branch_winner_window_matches_committed_token_replay` | CPU fp32 proofs | GPU route + attention KV for branch winners |
| Zero drafts accepted / all accepted | `…bitexact::test_zero_accept_replays_root_into_column_zero`, `::test_replay_chain_every_accepted_len_on_spine`; `conv_committed_path::test_zero_accept_reads_root_node_window`; E7a device harness: every accepted prefix incl. zero-accept (GPU, kernel level) | CPU + E7a GPU kernel | full-model step with sampler |
| Recurrent state at native materialization boundary | E7a stage-3 commits vs native spec-update kernel (GPU, kernel level; 12 operand sets) | E7a | model-level continuation (next forward) |
| Convolution history at boundary | `test_fr13_conv_committed_path.py` (15 tests: committed-path window, forced-spine diag); `test_fr13_fixed32_conv_commit_*` (cuda/wiring/zero_tail/batched_slots/row_guard) | CPU + some CUDA (fixed32 route) | plain route on current image; pending-token continuation |
| Attention KV remap | `test_fr13_attn_kv_remap.py` (cat9 spine remap, CPU); `test_fr13_fixed32_kv_remap.py`; `test_fr13_replay_conv_remap_page_safe.py` | CPU | branch-winner KV on GPU; page reuse/eviction |
| Next forward incl. pending correction token | none found by name | — | NEW harness needed (E7b continuation horizons 1/8/32) |
| Cold state / prefix-cache hit / eviction / recycled rows | none in tests/ (APC machinery removed per paper; `FR13_ENABLE_APC` launcher flag exists) | — | NEW (or declare out of scope for the stateless route) |
| Actual B1 / B4 | `test_fr13_b4_gdn_bv{64,8}_production.py`, `test_fr13_fixed32_gdn_batch_graph_gate.py`, `test_fr13_eager_pack_replay_byte_ab.py` (GPU-gated, fixed32/B4 routes); E7a B4 synthetic batch invariance (kernel level) | GPU-gated | plain route B4 on current image |
| Graph route intended for E1 | `test_fr13_fixed32_taw_fullgraph_route.py`, `test_fr13_fixed32_gdn_batch_graph_gate.py` (fixed32 mode only) | GPU-gated | plain-route CUDA-graph capture check at boot (cudagraph_mode recorded) |
| Fixed-spine numerical behavior (shared prefix + candidates) | H2 archive (`FR13_SLOT_REORDER_ARTIFACTS`); `scripts/fr13_slot_reorder_s0_test.py` | archived | fresh rerun on current route |
| Analytic sampler cases + biased negative control | `test_fr10_tree_rejection_sampler.py` (7 tests incl. `test_biased_multidraft_negative_control_fails_convergence`) | CPU (numpy) | trace into the deployed GPU sampler (`test_fr13_tier_b_sampler_pin.py` pins tier-B sampler text only) |
| B1 cumulative-sum nondeterminism limitation | documented (August closeout) | — | preserve; do not claim universal byte-exact sampler |

Plan to close gaps (bounded, after the fresh-capture boot is qualified): one teacher-forced continuation harness driving the served model at fixed prefixes with (a) verifier-only, (b) commit-only, (c) combined substitutions of the compact route against the sequential route, capturing per-layer residuals and final logits at horizons 1/8/32; reuse the E7a device kernels for the substitution arithmetic; run the existing GPU-gated tests on the current image as the baseline subset first.
