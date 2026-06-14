# FR13 — FA2-fork is M_DEPENDENT on the spine row (in-process A/B); carrier LOCALIZED + fixable (FR13_FA2_QPAD)

Date 2026-06-14. Workflow `wob0t2y8v` (in-process FA2 M=10-vs-M=5 A/B), Verdict **holds=True = M_DEPENDENT**.
Raw: `research/fr13_workflows/fa2_minvariance_ab_wob0t2y8v.raw.json`. This is the DEFINITIVE, decoherence-free
localization the free-running ladder couldn't be.

## The controlled A/B (no stream decoherence)
In ONE cat9 boot at the p3 deep-accept carrier event, re-call OUR forked-FA2 (apply_tree_bias post-QK,
contiguous-KV oracle) TWICE on the SAME captured K/V: M=10 (full tree) vs M=5 (spine-slice = spine rows
[0,1,2,4,6] + the 5x5 spine sub-bias, restricted spine-ancestor KV). The ONLY varied factor is M
(query-row occupancy). Compare the deep-spine row (flat row 6 = node5) attn_out, RAW max_abs per full-attn layer.

## Result: M_DEPENDENT (RAW != 0)
- Carrier event: 15/16 full-attn layers RAW=0.0 bit-exact; **L31 = 3.90625e-3 = exactly 1 bf16-ULP** (single
  channel, mean_abs 6.36e-7). NOT a bit-identity.
- 14-event sweep: 26/224 cells nonzero (~12%), recurs across **14 of 16 full-attn layers**, every value an
  EXACT power-of-2 (bf16 quanta). Divergence MONOTONE in spine depth (depth-0/1/2 bit-identical) =>
  pure QUERY-OCCUPANCY (kBlockM=64 MMA fragment tile + Is_even_MN=false predication + tree_bias lane
  offsets q/k_offset=max_seqlen_q-rows) — NOT a KV-slicing artifact (red-team neutralized).
- => the forked-FA2 query-tile is THE M-dependent carrier (co-residency batch-variance); GDN scan PROVEN
  M-invariant; the ~1-ULP/full-attn-layer compounds over 16 layers + the deep stack to the argmax flip.

## FIX: FR13_FA2_QPAD (pad query to fixed N_PAD_Q, flag-gated default-OFF, keeps FULL ACCEPT)
Pad the query (+ the MxM tree_bias) to a fixed N_PAD_Q in the fork's tree-bias decode dispatch so the
spine row's kBlockM tile / Is_even_MN / tree_bias offsets are M-invariant; padded rows = -inf-masked filler
(contribute 0, sliced [:M]). Lossless-by-construction (value-preserving), targeted (only the tree-verify
forked-FA2 call, NOT global BI -> avoids the cat9+BI=34 + +13GB blowup), our kernel still computes.
GATE SEQUENCE: (1) re-run THIS MAB A/B with QPAD -> carrier L31 RAW -> 0.0 + the 26/224 sweep cells -> 0;
(2) e2e per-token argmax-vs-clean-decode flip count -> does 22 -> ~3? (the DECISIVE gate; my red-team
concern = N_PAD_Q makes verify M-invariant but the gate is verify-vs-DECODE, so gate 2 is load-bearing).
A/B instrument banked: scripts/fr13_fa2_mab_replay.py + the FR13_FA2_MAB hook (default-OFF). Pairs with
[[reference_diffuse_gdn_accumulation_explained]], [[feedback_no_reroute_reward_hacking]], [[project_fr13_fa2_fork_nocopy_floor]].

## BUILD STATUS (branch fr13-fa2-qpad, CPU-built, GPU-gates pending)
FR13_FA2_QPAD implemented in scripts/fr13_patch_fa2_tree_bias.py (`_fr13_fa2_qpad_prepare` +
`_fr13_fa2_qpad_should_apply`), injected at module scope into the installed
`vllm/vllm_flash_attn/flash_attn_interface.py` -- the SINGLE forked-FA2 tree-bias dispatch that BOTH the
deployed tree-decode (`tree_attn.py` -> `flash_attn_varlen_func(..., tree_bias=...)`) AND the MAB replay use,
so GATE-1 and GATE-2 both exercise it identically. Flag-gated `FR13_FA2_QPAD` (default OFF) + `FR13_FA2_QPAD_N`
(default 64); wired into scripts/fr13_launch_forked_fa2_tree_server.sh (default OFF).
CONSTRUCTION: pad query to N_PAD_Q rows (real rows [0:M], zeros [M:N]) AND extend the suffix-key extent by
`pad` (seqused_k += pad on the paged decode; appended zero KV rows on the contiguous MAB path) so the kernel's
causal offset (max_seqlen_k - max_seqlen_q == context_len) and the bias column origin are UNCHANGED -> each
real row's score tile is bit-identical to the unpadded call, only the TILE OCCUPANCY (the M-dependent carrier)
is now constant. The padded bias [N,N] is real top-left MxM + padded diagonal 0 + everything else -inf, so
real rows never see padded/garbage keys and padded query rows (sliced [:M]) never affect real rows.
DEFAULT-PATH PROOF: pre-QPAD-patched vs post-QPAD-patched flash_attn_interface.py diff = 190 additions, 0
deletions/modifications; every behavioral add is under `if _fr13_fa2_qpad_should_apply(...)` (False unless
FR13_FA2_QPAD=1 AND tree_bias present AND fa_version==2) or the inert `_fr13_qpad_unpad=None` init + its
`if _fr13_qpad_unpad is not None` return guard -> flag-off path byte-identical to the locked cat9 default.
Regular (non-spec) decode has no tree_bias so it never QPADs even with the flag on (verifier-only preserved).
LOSSLESS-BY-CONSTRUCTION VERIFIED on CPU: scripts/fr13_fa2_qpad_lossless_oracle.py (kernel-faithful
per-column-dot + -inf-exact softmax) -> real-row outputs BIT-IDENTICAL (0.0) to the unpadded forked-FA2 across
M=5/9/10, B=1/3, fp32/bf16, contiguous + paged-with-garbage-padded-KV. NOTE the per-column dot is essential:
a width-dependent `q@k.T` batched GEMM injects a ~6e-8 artifact the CUTLASS MMA does NOT have (each S[i,j] is
computed over the fixed head dim, key-count-invariant).
PENDING (GPU, serialized, NOT run here): GATE-1 = MAB A/B re-run with FR13_FA2_QPAD=1 + FR13_FA2_MAB=1 ->
carrier L31 RAW -> 0.0 + the 26/224 sweep cells -> 0; GATE-2 (DECISIVE, verify-vs-DECODE) = e2e per-token
argmax flip count 22 -> ~3 + accept/event >= native + within_boot_det [T,T,T,T]. RED-TEAM (load-bearing):
if GATE-2 does NOT drop 22->~3 (or worsens), the QPAD-to-64 does not match the M=1 decode tile geometry ->
REPORT it, do NOT claim success; the correction would be to match the decode's M=1 tile, not an arbitrary N.
