# FR10 — cross-step state residual (fresh session brief, 2026-06-04)

You are the worker on branch `fr10-gdn-tree-kernel`. Claude (monitor/red-team) runs a /loop and
steers + verifies. This brief is the ANCHOR — read it + `FR10_STATUS.md` + `FR10_DRAFTER_TOPOLOGY_FINDING.md`
before acting. **DO NOT re-litigate anything in "SETTLED" below — those are proven, with gates.**

## GOAL
Lossless GDN/STree caterpillar token-tree verifier for Qwen3.6-27B that beats native MTP-5
(E5 ~39.9 decode TPS). B=4, temp=0.6, top_p=0.95, gpu-mem 0.88, fp8, mtp depth 5. Caterpillar tree
`[(0,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0),(0,1),(0,0,1),(0,0,0,1),(0,0,0,0,1)]`.
Native baseline on the same-8 coding prompts = **3.076 accepted/event** (the target to match).

## SETTLED — do NOT re-investigate or re-run these gates
1. **Draft is byte-exact native.** GATE-D1 draft-token parity = `match`, 32/32 at every depth 0..4
   (`cce6f92a` caterpillar native-spine + read-only top-2 leaf, slots spine `[0,1,3,5,7]` / leaves
   `[2,4,6,8]`). The read-only drafter gate (GATE-D1 spine IDs == native; GATE-D2 per-step logits ==
   native; leaf is a pure side-read) must STAY green — never change drafter math.
2. **Within-event verify is PROVEN clean (conv + scan + kernel output), incl node-0.** Scan-output
   captured-replay: node0 out 0.0/4.5e-13, all nodes ≤3.8e-6 (bf16 eps), `serving_vs_replay` bit-exact,
   powered flat negative control fails 0.5/4.96. Conv byte-exact (`conv_all_nodes_vs_serial_max=0.0`).
   **DO NOT re-run scan/conv/kernel gates — they are settled.**
3. Mixed-batch tree-branch wiring fixed (`fc48f1a6`, `num_spec_decodes>0`). Tree-branch nonvacuity
   green (10608/10608 spec rows in tree branch, 0 flat fallback).
4. The preprocess state-copy redirect (`a130fdd4`) is **INERT** in serving: redirect-active fraction
   0/0 because `prev_state_idx == curr_state_idx` (no block migration). Don't rely on it.

## THE RESIDUAL (the whole remaining task)
Tree path0 accept ≈ **0.557/event** (depth-0 frac 0.267) vs native **3.076** (depth-0 0.68).
**Node-0 — the caterpillar root, shared by EVERY path — is rejected ~67% of events.** Its draft is
native (GATE-D1) and its conv+scan output are byte-exact, so the ONLY remaining input that can make
node-0's target logits wrong is the **starting recurrent state** fed into the step. => the residual is
the **cross-step state hand-off**: the `initial_state` the NEXT step reads ≠ native.

Mechanism (since no copy fires): the GDN layer commits ALL tree-node states via
`ssm_state.index_copy_(0, spec_state_indices_tensor[fr10_b, :tree_n], tree_state[:tree_n])`
(patcher ~L970) — node j → bank row `spec_state_indices_tensor[fr10_b, j]` — and the request RETAINS
its block. So the next step's `initial_state` = whatever row the request reads, which must be the
**accepted node's** state row.

## PRIME SUSPECT — addressing off-by-one
`accepted_row = int(best_path[best_lcp-1]) + 1 if best_lcp>0 else 0` (patcher ~L1319), and the SSM
copy passes `accept_token_bias + 1` (~L2591), conv passes `block_ids[src_block_idx + accept_token_bias]`
(~L2614). The stacked `+1`s are where a faithful-copy-of-the-WRONG-row hides (that's why `d5ca8980`'s
`dest==src` copy gate stays green while acceptance is broken). The accept is **bimodal** (mostly-0 with
a full-spine spike) = a CONDITIONAL addressing bug: the read row aligns for some acceptance patterns,
not others.

## TASK (in order)
1. **Instrument, per request:** `{accepted node id = best_path[best_lcp-1], the spec_state_indices row
   the accepted state was written to, the bank block/row the request READS as initial_state at step
   N+1}`. Assert all three coincide. This alone likely exposes the off-by-one.
2. **Build the one missing gate — `src==native` (the decisive one):** capture the state the next step
   ACTUALLY READS as `initial_state` (ssm_state AND conv_state), run a native decode over ONLY the
   accepted token ids as the oracle, byte-exact compare, with a powered negative control (a deliberately
   wrong row FAILS). Regime = `deterministic_captured_replay_byte_exact` (fixed captured inputs, NOT
   live end-to-end). The existing `d5ca8980` gate only checks `dest==src` (copy fidelity) — insufficient.
3. **Fix** the addressing so the next step reads the accepted node's state row (audit the `+1`s vs the
   `spec_state_indices` layout). Keep conv_state committed with the accepted row too.
4. **Confirm:** re-measure path0 accept on the same-8 prompts; target ≈ native 3.076, node-0 frac → ~0.68,
   survival gradual (no node-0 cliff). Then the superset gate (tree accept ≥ native, strict-win CI>0).

## CONSTRAINTS
- Memory recovery before EVERY server reboot: `sync; drop_caches; compact_memory; swapoff -a; swapon -a;
  drop_caches` via `LUMO_SUDO_PASSWORD` — direct-docker launch wedges ~100GiB on GB10. ONE server at a
  time. oom_score_adj-protect this stack.
- Iterate OFFLINE (replay/parity on captured tensors) where possible; boot ONCE to confirm. Stop the
  boot-per-bug cycle.
- Commit + push every step on `fr10-gdn-tree-kernel`; record numbers in committed docs (output/ is gitignored).
- All math/parity through committed tests, never hand-rolled one-offs. Deliverable kernel = CUDA-only +
  CUDA-graph-capturable (CPU recurrent is oracle only). Lossless is a hard constraint; speed beats E5.

## DEBUGGING PROTOCOL (standing, user 2026-06-05 — applies to worker AND monitor)
READ THE ACTUAL SOURCE FIRST, both vLLM and our own code, before guessing from behavior:
1. For any vLLM seam you patch/depend on, read the LIVE source (extract from the image:
   `docker run --rm --entrypoint cat <image> <path>` — no container needed). Container being
   down is not an excuse.
2. For our own code, read the function that actually runs (the production dispatch, the gate's
   computation — not its name; the kernel's indexing) before concluding anything from a metric.
3. When a metric drives a decision, read what it COMPUTES, not what it's named (`*_vs_native`
   must run native). Prefer a gate with a powered negative control over an inline number.
4. When a fix has a cheap byte-exact oracle (e.g. the offline src_native gate), run it to CONFIRM
   the fix BEFORE the expensive end-to-end measurement. Gate first, acceptance second.
Evidence this is non-negotiable: the cross-step bug (stock mamba linear `accept_token_bias`) cost
~15 ticks of behavioral state-fixing on the wrong buffer; one read of `mamba_utils.py` found it.
