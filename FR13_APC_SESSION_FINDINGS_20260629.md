# FR13 APC — 2026-06-29 findings: char-8 is a tool-call flake; the cache "drift" is a kernel-realization defect; redesign = SGLang's design

## TL;DR
1. **char-8 ≠ cache derail.** The "char-8" malformed-JSON tool-call runaway (`BadRequestError: Unterminated string … column 9 (char 8)`) is a **server-side re-parse flake** of a *prior* turn's tool-call `arguments` JSON, documented in our own proxy as a flaky decode and retried. It fires mostly at **cold cache**, occurs in **cache-OFF** runs too, and is **largely cache-independent**. The reconciliation workflow's "cache∧spec conjunction" verdict was **overturned by its own red-team**.
2. **The GDN align "drift" is sparse/bounded, not a gross wrong-state.** The headline `77.96` is inflated by capture **position/turn misalignment** + outlier channels; the per-element **mean drift ≈ fp (0.0069)**. Root cause (verified on the **real deployed vLLM 0.19.2**): a **kernel-realization mismatch** — prefill writes the cache with the CHUNKED kernel, decode/spec with the RECURRENT kernel; a hit can restore a recurrent-realization slot and continue it chunked. Accumulates over boundaries. This **confirms** the launcher's prior GPU-verified baked finding.
3. **`mamba_block_size=8192` is a band-aid** — it cuts the boundary *count* (less accumulated realization error) but costs up to `block_size` tokens of mamba **tail re-prefill per cache hit** (~8× more than 1024) = a real TTFT tax. **1024 + spec + cache fails the task.** The real fix is a lossless single-realization cache.
4. **Version de-risk:** the deployment runs vLLM **`0.19.2rc1.dev134` (py3.12)**. A prior root-cause agent audited the **wrong version (v0.22.0)** → its file paths/line numbers are fiction, though its structural conclusions hold on the real code.

---

## 1. char-8 reconciliation — OVERTURNED
- Dataset: **1917 live rollouts, ~73–75 with char-8** (base rate ~3.8%).
- First-pass verdict (single-agent, the workflow fan-out failed): "char-8 = (engaged cache) ∧ (spec), Fisher p=0.004." **Red-team dismantled it:**
  - "Engaged cache" was a **warmup-contaminated** label — `cached_max` is the per-*boot* max including the cache-priming warmup probe, **not** the cache state at the breaking request. rg_ON_r1 ("82.4% engaged") showed **0.0% prefix-cache hit at the moment of the 400**; its trace is **698 bytes (turn-1 death)**.
  - The engaged/vacuous split is **circular** (the cmax=0 cases segregated out as "cache-OFF" *are* the char-8 deaths).
  - char-8 occurs **cache-OFF** too (rg_OFF_r1 emits malformed-XML tool-call warnings and still resolves).
- **Real mechanism (held, host-verified):** char-8 is a re-parse of a malformed tool-call argument from a prior turn. **`src/lumo_flywheel_serving/inference_proxy.py:2013-2017`** documents the `Unterminated string` 400 as *"a flaky decode of the request, not a real client error,"* retried via `LUMO_PROXY_RETRY_UPSTREAM_400`. Source: vLLM chat renderer `json.loads(arguments)`.
- **Cache-at-derail join (decisive):** of 64 char-8 traces with determinable cache state, **~50 were COLD/no-cache at the breaking request, 14 warm**; 41/75 died ~turn-1. Smoking gun: deep `fr13_bigdenom` tree runs show `cached_max=40768` (the "engaged" label) but **`cached_at_last=0`** (cold at the actual char-8 request).
- **Conclusion:** char-8 cannot be attributed to cache/spec/spine/tree; it's a tool-call-format flaky-decode artifact. This **weakens the blocksize-finding framing** (8192 "resolves" / 1024 "fails" were scored via char-8).

## 2. GDN align "drift" — root cause (verified on REAL vLLM 0.19.2)
Real file `model_executor/layers/mamba/gdn_linear_attn.py`:
- **Stock align CONTINUES** (not a restart): prefill does `initial_state = ssm_state[non_spec_state_indices_tensor]` (~L984), zeros only fresh rows (~L986), `self.chunk_gated_delta_rule(initial_state=…, output_final_state=True)` (~L988), write-back `ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)` (~L1004).
- **The mismatch:** prefill writes the cache via the **CHUNKED** kernel (L1004); **decode** via `fused_sigmoid_gating_delta_rule_update` (~L1008); **spec** writes recurrent IN-PLACE to the node-bank `spec_state_indices_tensor` (~L957-973). A cache hit can restore a slot last written by the recurrent kernel and continue it chunked → realizations differ ~0.0078 >> bf16 ULP, accumulating over boundaries (~30 @1024 vs ~3 @8192).
- **Extra realizations / terms:** a **3rd realization** (FlashInfer `fi_chunk_gated_delta_rule` ~L70 vs fla chunked ~L199); the write-back `.to(ssm_state.dtype)` **truncates to bf16** under default `auto` (deployment pins fp32).
- **Magnitude is sparse:** per-element mean ≈ 0.0069 (at fp); a thin tail (layer-0 + large-magnitude channels) diverges 10–80% **relative**. Drift curve: block 1024 state_max=77.96, 2048=49.70 (decreasing with block; 8192 reduce pending; 4096 boot failed on the launcher bug below).
- This **matches the launcher's baked finding** (`fr13_launch_forked_fa2_tree_server.sh:273-281`): *"the cached GDN checkpoint is the RECURRENT-decode kernel's realization, not the CHUNKED-prefill realization cache-OFF holds."*

## 3. The 8192 band-aid + speed
- On a cache hit the mamba state is reused only to the last `block_size` boundary; the **tail (up to block_size tokens) re-prefills** (re-run GDN chunked scan). **8192 ⇒ up to 8192 (avg ~4096) tokens re-chunked per hit; 1024 ⇒ ≤1024 (avg ~512).** KV-cache TTFT win + decode-spec TPS are unaffected; the mamba-TTFT win is the tax.
- 1024 didn't solve spec+cache, so fine block sizes have a real correctness problem (the accumulated realization mismatch) — block size is a **lose-lose dial** (small=more drift, large=more TTFT).

## 4. The lossless redesign = SGLang's design (greenlit; design in progress)
Goal: **a cache hit should be bit-identical to no-cache** — same kernel, same realization, same positions. Three moves:
- **R1 single realization:** cache only ever holds the CHUNKED-prefill realization; on a hit, finish the tail through the *same* chunked kernel. Pin ONE chunked kernel (fla vs FlashInfer).
- **R2 fine checkpoints:** checkpoint at 64-aligned (FLA chunk) positions, not a coarse block — tiny tail, kills the band-aid + TTFT tax.
- **R3 fp32 end-to-end** (already enforced).
- **Crux (decode-checkpoint):** decode/spec produce recurrent-realization state (can't chunk a 1-token step) → the decoded region has no chunked checkpoint. Options: (a) re-chunk the decoded tail on hit, (b) periodically re-chunk during decode, (c) store recurrent checkpoints but re-chunk on restore. **Open — the design agent is settling it on the real code.**
- **This is exactly the SGLang approach**, already the launcher's documented direction (`:273-281`): *"SGLang MambaRadixCache / Execution-State-Capsules (arXiv 2606.20537) / Sparse-Prefix-Caching (2605.05219): cache chunked-realization checkpoints ONLY at 64-aligned positions (chunked-only chain, exact base) + restore the remainder THROUGH the chunked kernel."* A prior in-progress attempt exists: `FR13_APC_EXACT_SEED` (worktree `wf_4f4d8bf1`). The superseded `FR13_APC_HIT_RECURRENT_SUFFIX` failed because it re-prefilled the tail with the *recurrent* kernel (≠ chunked, can't be bit-exact) — un-baked 2026-06-27.
- Mostly **stock-vLLM territory** (the mamba cache write contract + restore path) → potentially upstreamable; the spec node-bank piece stays in the FR13 patch.
- **Caveat:** SGLang characterization is the codebase's cited research, not an independent read of SGLang source — verify against SGLang's actual `MambaRadixCache` before final implementation.

## 5. Bugs fixed + infra (this session)
- **Launcher continuation bug (FIXED):** the experimental-flag cleanup inserted a comment line inside the `\`-continued `docker run` command → `docker run requires at least 1 argument`, breaking **every** boot (e2e gate, TTFT sweep, drift 4096). `bash -n` passed (syntactically valid); only runtime failed. Fixed by pulling the comment out of the continuation.
- **Experimental flags removed** from the launcher (shadow / stale-injection / conv-pre-redirect / cacherow-dump / leaf-crosscheck / conv-snap-fix / pre-snap-fix / stale-hit-detect). Kept: `MAMBA_BLOCK_SIZE=8192` + proven-baked `CONV_FIX`/`CONV_SNAPSHOT`/`SNAP_FIX`/`ZEROACCEPT`.
- **PRE_SNAP_FIX confirmed VACUOUS** (drift 77.96→77.96 unchanged) — the preprocess SSM redirect did nothing.
- New scripts: `fr13_apc_ttft_sweep.sh` + `fr13_apc_ttft_probe.py` (TTFT-speedup per block, cold-miss vs warm-hit), `fr13_apc_drift_curve_extend.sh`.

## 6. Open / next
- **Lossless redesign** (greenlit): re-ground to real 0.19.2 (in progress), settle the decode-checkpoint decision, propose the precise diff, validate cache-ON vs cache-OFF state diff → fp. Build on `EXACT_SEED`.
- **TTFT sweep** to put a number on the 8192 tax.
- **Clean big-N** spec+cache cache-ON vs cache-OFF on 12907 with `LUMO_PROXY_RETRY_UPSTREAM_400=1` (treat char-8 as the retryable flake it is) — the live ship gate, no longer confounded by char-8.
- **Verify SGLang `MambaRadixCache` source** before final implementation.
