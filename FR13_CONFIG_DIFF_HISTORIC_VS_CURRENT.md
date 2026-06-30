# FR13 — Config diff: historic cache-OFF (23.88/18.80) vs current build + spec-accept

Source: config-diff workflow `wf_c4ac40c4-07c` (4 agents, read-only). Settles why the historic cache-OFF
decode numbers (cat6root 23.88, E5/spine 18.80) differ from the current cache-ON (cat6root ~16.8), and
whether spec-decode acceptance regressed.

## TL;DR — the "~30% decode tax" is mostly artifact, not a real cache penalty
1. **Basis confound (biggest).** Historic **23.88** is `derived_tps` = `gen_tok / Σ request_decode_time`
   (token-weighted, prefill-independent). On a **per-request** basis the SAME historic cat6root run is
   **18.51**, not 23.88. So a big chunk of "23.88 → 16.8" compares two *different bases*. The physical gap
   is closer to **~18.5 → 16.8 (≈9%)**, not ~30%. **Any clean comparison must use the same basis on both
   sides.** (The current-build 3-way does — same `reduce()` on every arm — so it sidesteps this entirely.)
2. **Spec acceptance did NOT regress.** On the matched task (12907), current cat6root accept/event
   **3.52** vs historic **3.22** = **+9.5%**; cat10 ties (3.53 vs 3.56). The historic 4-task aggregate
   accept (3.82) is a task-mix/denominator artifact, not a per-kernel edge. So acceptance is **not** the
   cause of the decode gap.

## Config-diff table (axes that differ are bold)
| Axis | Historic (cat6root, 23.88) | Current (cat6_ON) | Differs? |
|---|---|---|---|
| Graph mode | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE | no |
| enforce_eager | False | False | no |
| **Mamba/APC block size** | **816** (auto attn-align floor) | **1024** (64-aligned, for bit-exact APC) | **YES** |
| **Prefix cache (FR13_ENABLE_APC)** | **OFF** | **ON** | **YES (the headline)** |
| **FR13_APC_EXACT_SEED** | **OFF** (pre-EXACT_SEED) | **ON** | **YES** |
| Tree shape / num_spec | cat6root 6-node / 6 | cat6root 6-node / 6 | no |
| Sampling temp | 0.6 | 0.6 | no |
| **Sampling top_p** | **1.0** | **0.95** (codex client) | **yes (trajectory only)** |
| KV cache dtype | bf16 (auto; fp8 = weights only) | bf16 | no |
| **SSM/Mamba cache dtype** | **bf16** | **float32** (`--mamba-ssm-cache-dtype float32`) | **YES** |
| BATCH_INVARIANT | 0 | 0 | no |
| vLLM version | 0.19.2rc1.dev134+gfe9c3d6c5 | same image sha | no |
| **GPU_UTIL** | **0.82** | **0.65** (cat6_ON, leak) | **yes (negligible at B=1)** |
| max_num_seqs | 1 | 1 | no |
| FIX-1/2/3 | 1/1/1 | 1/1/1 | no |

## Ranked decode-delta hypothesis (after removing the basis confound)
1. **APC ON vs OFF** — the headline manipulation + the only entirely-new code path (snapshot/restore,
   conv-snapshot, per-step committer bookkeeping/host-sync). The cost we *want* to measure, but here fully
   confounded with every drifted axis below.
2. **SSM cache dtype bf16 → float32** — **HIGH, and the most actionable unconfounded knob.** float32 GDN
   conv/recurrent state ~doubles state-cache bandwidth on a memory-bound recurrent decode. fp32 was chosen
   for bit-exact APC; **if a cheaper SSM precision is lossless-enough, this is directly recoverable.**
3. Block size 816 → 1024 — medium (page size / layout; chosen for alignment not speed).
4. GPU_UTIL 0.82 → 0.65 — low at B=1 (memory-headroom knob, not a per-token-speed knob).
5. EXACT_SEED ON — low-med, bundled with APC, small added sync.
6. top_p 1.0 → 0.95 — negligible per-forward; perturbs *trajectory* (feeds the denominator confound).

## Structural acceptance ceiling (independent finding)
Tree positions **≥5 accept exactly ZERO** (cat6root pos5=0; cat10 pos5–9=0). Only the first 5 spec
positions ever commit → realized accept/event is capped near **~3.5** regardless of tree depth. This is why
**cat10 (wider/deeper) does not beat cat6root** — its extra nodes are dead weight at B=1.

## Implication for the clean 3-way
- The current-build 3-way (`scripts/fr13_apc_3way_gate.sh`, same basis every arm) **controls for**: basis,
  block size, top_p, vLLM version, FIX-1/2/3, graph mode.
- It does **NOT** by itself separate the **cache code path** from the **fp32-SSM** cost, because the
  deployed cache-ON arm *requires* fp32 SSM for bit-exactness while cache-OFF naturally runs bf16. So
  cat6_ON(fp32+cache) vs cat6_OFF(bf16+no-cache) measures the **deployed package** penalty — which is the
  right deployment number — but to *attribute* it, a follow-up cat6_OFF-fp32 (or cat6_ON-bf16, lossy) arm
  isolates the SSM-dtype component. That's the real lever for "preserve both speedups."
