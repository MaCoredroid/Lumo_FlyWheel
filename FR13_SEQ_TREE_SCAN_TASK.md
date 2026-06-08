# FR13 — PIVOT to the sequential rank-1 GDN tree-scan (user-confirmed 2026-06-08, option a: pivot + measure TPS empirically)

## DECISION (see FR13_WY_VS_SEQUENTIAL_VERDICT.md, source-verified)
The WY batched UT-solve kernel is **abandoned as the deliverable** (kept ONLY as an fp32 oracle). Native's VERIFY oracle is the **SEQUENTIAL rank-1 recurrence** (`gdn_linear_attn.py:1117 if spec_sequence_masks is not None: -> fused_sigmoid_gating_delta_rule_update`; chunked-WY `:1142` is prefill-only). WY-batched can NEVER be bit-exact to a sequential oracle (different summation tree, fp non-associativity — same impossibility class as Triton->FA2). The bit-exact-able path is a **sequential rank-1 TREE-SCAN** = native's own kernel re-indexed by tree-ancestry. On the pure spine it collapses to native's loop IDENTICALLY -> bit-exact by construction.

## THE TASK
Make our **`use_wy=False`** path in `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` (the sequential ancestor-replay path, the nested `for i`/`for j` ~L278-300) **op-for-op IDENTICAL** to native `fused_sigmoid_gating.py:152-168`, then add per-node ancestry masking + **register-resident branch checkpointing** (NO HBM state re-streaming).

### The exact native op sequence to mirror (fp32 accumulate throughout), per token in topological order:
1. `b_g = -exp(A_log) * softplus(a + dt_bias)` (the gate; raw-g form)
2. in-kernel `l2norm(q,k)` with `+1e-6` eps; `q *= scale`
3. `b_h *= exp(b_g)` (state decay)
4. `b_v -= tl.sum(b_h * b_k[None,:], 1)` (delta read)
5. `b_v *= b_beta` where `b_beta = sigmoid(b)` in **PURE fp32 — NO bf16 cast** (CORRECTED 2026-06-08 by workflow w4abw0spa: the native SEQUENTIAL kernel `fused_sigmoid_gating.py:150` has NO bf16 anywhere; the beta-bf16 was a WY-path-only artifact (`_tree_gdn_wy_kernel` FLA_BF16_BOUNDARIES). **DO NOT add bf16 to `_tree_gdn_kernel` — it would INJECT drift, not remove it.**)
6. `b_h += b_v[:,None] * b_k[None,:]` (rank-1 write)
7. `b_o = tl.sum(b_h * b_q[None,:], 1)` (readout)
Source: `fused_sigmoid_gating.py:134-168` (h0 load L134, the `for i_t in range(0,T)` loop L136, ops L158-167). Read the LIVE file first; mirror EXACTLY (cast boundaries, accumulation dtype, op order).

### Tree extension (lossless, theorem-backed — reference_gdn_tree_branch_oracle_losslessness)
- Walk tree nodes in TOPOLOGICAL order; each node updates `b_h` from its PARENT's state (ancestry), not the linear predecessor. Spine (MTP-5 chain) = native's linear loop exactly.
- Off-spine branches = native sequential run on the node's path-to-root. Keep per-node state **register-resident**; checkpoint to registers/SRAM (bit-exact fp32 copy), write ONLY the committed accepted-path state to HBM.
- **SPEED CONSTRAINT (the +35.8% replay tax was HBM state-traffic, NOT sequential-ness):** do NOT re-load h0 per node or write per-node intermediate state to HBM. This is the load-bearing speed design.

## GATES (strict, per commit; bind to FR13_LADDER_LOG.md)
1. **Per-layer ladder spine AND branch -> 0.0** (B=1 eager first; pinned-prompt paired harness, reuse the existing paired-run infra + `fr13_layer0_subop_localize.py` + native-on-path oracle). first_nonzero must clear layer-0, then propagate to ALL 64 layers + final logits = within self-noise. Branch = TRUE native-on-branch-path oracle (per-depth argmax), NOT the tree self-target proxy.
2. **Gate-2 (regular decode == pristine):** plain decode with the kernel == stock, 0.0 every layer (the OFF/default path must be untouched).
3. **CUDA-graph capture** (hooks OFF) before the final measurement; re-confirm 0.0 at B=4 (co-residency changes things).
4. **e2e vs E5** (`output/fr10_native_mtp5_same8_20260604T210257Z`, accept/event 3.076, floor ~0.059): lossless within E5 self-noise floor + accept/event >= 3.076.
5. **TPS measured EMPIRICALLY (the user's de-risk):** B=4 decode TPS of the sequential tree-scan vs (a) native MTP-5, (b) the WY kernel. Must be >= native and must NOT show the +35.8% replay tax. Metrics OFF, clean CUDA-graph. Report the E5-vs-SEQ table; do NOT self-declare PASS.

## DISCIPLINE (standing)
ONE GPU (no concurrent --gpus; relaunch WITHOUT --rm; recover_host_memory between arms — forked exit wedges ~90GB, sudo pw in .lumo.local.env). NO reward-hack: build OUR sequential kernel; do NOT splice native fused_sigmoid_gating/causal_conv1d_update as "our kernel" (oracle ONLY, gate runs splice-OFF). NO copy/dense/reroute. Commit+push+bind EVERY step (in HEAD AND pushed). Report at the deliverable (lossless within floor + accept/event >= 3.076 + TPS >= native) or a genuine wall. The monitor runs parallel CPU workflows ahead of you.
