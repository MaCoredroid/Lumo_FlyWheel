# FR-11 — Resolve no-copy GDN tree verify: BUG-FIXABLE vs PRECISION-FLOOR

**Branch:** `fr11-gdn-nocopy-probe` · **Driver:** codex gpt-5.5-high in tmux `codex_fr11` · **Red-team + commit/push:** Claude (Opus) on a 10-min loop.

## Why we're here (read these first)
- `FR10_NOCOPY_RESOLUTION.md` — the resolution pass. No-copy was CLOSED then REOPENED: the no-go was under-investigated. Verdict downgraded to **NEEDS-ONE-GPU-PROBE**.
- `FR10_PAPER_NOGO_RESEARCH.md` — literature: **no theoretical no-go**. STree is itself no-copy and refutes "shared state degrades path0"; the scary 0.038 is orthogonal. So no-copy is open, not dead.

## What the prior pass established (don't re-derive — verify)
1. **A real source seam:** FR10 tree conv does **fp32** tap products (`src/lumo_flywheel_serving/fr10_tree_conv.py:59-61,68,72`; live patch `scripts/fr10_phase4_patch_vllm_tree_gdn.py:576,608-611`), native `causal_conv1d_update` does **bf16** tap products then fp32-accumulate (`/tmp/vllm_live_019/.../causal_conv1d.py:442`). Window/index byte-correct; only the dtype differs.
2. **The GDN scan is exonerated** (7.45e-9). The 0.0156 layer-0 drift originates **upstream of the scan** (conv / o_proj / state-handoff), NOT the scan algebra.
3. **The closeout's "0.0156 diffuse, no locus" was a tensor confusion** (conv-output space mag~1 vs residual-hidden space mag~1.9); faithfully propagated, the conv seam reaches only ~1.2e-4 at GDN output. So its causal sufficiency for the live loss is **unproven**.

## The two decisive measurements — DO THESE (both boot-free GPU: replay captured tensors, NO model load)

### Probe β — event-0 state-handoff byte-compare (RUN FIRST; could flip to BUG-FIXABLE)
At decode-event-0, dump the spine's initial recurrent state `h0[ssm_state_indices[i_n, num_accepted_tokens-1]]` and the conv prior-state window (`conv_state_token_offset = num_accepted_tokens-1`, native `causal_conv1d.py:859-860`) loaded by the **native MTP-5** forward, and **byte-compare** to what the **FR10 tree spine** loads.
- **Match** → #39273/#40738 wrong-state class excluded for the seed (prior evidence: `ssm_next_vs_native 2.86e-6`, `conv_next_vs_native 0.0`).
- **Differ by ≥1 column** → **#40738-class wrong-`initial_state` bug → BUG-FIXABLE.** Fix = port #40738's `conv_state_token_offset`/`num_accepted_tokens` threading, then confirm accept recovers toward 3.076.

### Probe α — conv-seam → residual propagation (decides precision-floor vs locus-elsewhere)
Replay one captured bf16 `mixed_qkv_spec`/`conv_weights`/`bias` through (a) native `causal_conv1d_update` vs (b) fp32-product tree conv, then push **both through the real `o_proj` + RMSNormGated + residual** to the layer-0 residual-hidden output. Write `output/fr10_nocopy_resolve/gpu_conv_seam_replay.py`.
- `|residual(a)−residual(b)| ≈ 0.0156` → conv-dtype seam is **causally sufficient** → PRECISION-FLOOR; test the one-line conv fix for acceptance recovery.
- `≈ 1.2e-4–5e-4` → conv seam is **NOT** the source → locus elsewhere; hunt the next upstream seam.

### Conv tap-dtype fix (cheap, correct-direction regardless)
Drop the `.to(torch.float32)` upcast on the conv taps so the tree conv matches native's bf16 product (→ matches native bit-for-bit, 0.0). Flag-gate it. Measure accept/event vs native MTP-5 (E5 ≈ **3.076**), B=4, temp0.6, top_p0.95, mtp5, **metrics OFF for speed**.

## Hard rules (both codex and Claude)
- **Read live vLLM source, not behavior.** Canonical: `/tmp/vllm-0.22-src/vllm-0.22.0/vllm` (0.22) and `/tmp/vllm_live_019` (the live-container seams cited above). Read the patched seam AND its downstream consumer before patching/guessing.
- **Fail loud.** Assert tree-engagement before recording any number (FR10_REQUIRE_TREE / has_tree_parent_indices). No silent fallback to linear.
- **Don't retry measured dead-ends:** M-RoPE explicit broadcast (→1.20); TREE_ATTN backend (1.49 < 1.77); big-tree kernel speed opt (can't beat 135µs FLA flat); byte-exact-vs-MTP-5-baseline (wrong bar — baseline itself diverges 6e-5 from non-MTP).
- **Commit + push every step** to `fr11-gdn-nocopy-probe`. Record numbers in committed docs (`output/` is gitignored).
- **GPU/host hygiene:** GB10 unified mem; reset prefix cache + `torch.cuda.empty_cache()` between runs; ModelServer's built-in host-memory recovery runs at vLLM start/stop (don't bypass via docker restart).

## Definition of done
Probe β + Probe α both run with real numbers → verdict **BUG-FIXABLE** (with the fix landed + accept/event measured toward 3.076) or **PRECISION-FLOOR** (proven, no-copy closed on evidence, pivot to copy-recurrent multi-spine). Either way: committed doc with the numbers.
