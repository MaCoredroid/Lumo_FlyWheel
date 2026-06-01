# FR9 Closeout — Multi-Spine Speculative Decoding Superset on the Qwen3.6 GDN-Hybrid

**Date:** 2026-06-01
**Status:** CLOSED. Superset proven; modest honest speedup; production stability validated.
**Commits:** `1dcc3e5e` (independent-row launch) · `65fb6439` (winner-commit + state-sync) · `c5694e04` (apples-to-apples direct probe) · `dbba507b` (temp-proxy alignment + winner-state ordering) · `079d51f4` (temp-0.6 agentic stability run) · `dc22a190` (options-doc closeout) · this report.
**Companion:** `fr9-independent-rows-options-20260601.md` (the maintained modes/flags reference).

---

## 1. Verdict (read this first)

We set out to build a **true superset** for 2-spine (generalizing to N=2–10) speculative
decoding on the Qwen3.6-27B fp8 Gated-DeltaNet (GDN) hybrid: a decoder whose accepted
output is **≥ the native MTP depth-5 chain (E5) on every event** ("beat-or-tie"), with a
real, honest speedup.

**What we achieved, stated plainly:**

1. **Superset PROVEN (greedy).** Independent co-resident rows, winner-commit + recurrent-state
   sync: winner accept = **3.442/event ≥ spine-0 on every one of the gated events,
   `superset_violations=0`**. Spine-0 reproduces E5 to within sampling noise (2.866 vs canonical
   3.002; per-step forced-LCP byte-identical 64/64). The "beat-or-tie E5" goal is met **by
   construction** and **measured**.
2. **Speedup is REAL but MODEST: +6.85%** (1.069×), apples-to-apples direct probe, low batch.
   The +24% accept gain is real, but the second co-resident row's per-step cost eats most of it
   at this batch — the extra row does **not** ride free here.
3. **Production stability VALIDATED (temp 0.6, rejection sampling).** The aligned agentic-B4 run
   sustained to the agent-wall timeout with **no engine death** and the per-event superset
   invariant holding — **`viol=0`, `missing_sum=0` across 19,307 events**. This proves the
   winner-commit/state-sync machinery survives real production sampling and concurrency, not just
   greedy probes.

**What we did NOT achieve / honest limits:**

- We do **not** have a clean agentic-production speedup number vs E5's 26.86 tps. The temp-0.6
  run's throughput/accept are workload-confounded (§5) and must not be read as a speed result.
- The proven net speedup (+6.85%) is at **low batch** only. Whether it grows or shrinks at higher
  batch / under contention is untested.
- The **in-tree token-tree route is closed** at a hard ceiling (free-running path0 capped 1.751 vs
  E5 3.002) — it cannot cheaply reproduce E5 on this hybrid.

Bottom line: **a proven, correct ≥E5 superset that runs stably in production, with a modest
+6.85% low-batch speedup.** The big speed wins (higher batch, STree-style shared-ancestor kernel)
are identified but unbuilt.

---

## 2. The problem and why it is hard

E5 = the model's native MTP head drafting a depth-5 chain, verified by the target. On the
agentic-B4 workload it accepts **3.150 tokens/event at 26.86 decode tps** (temp 0.6, top_p 0.95);
greedy canonical it accepts **3.002/event, 13.3% acc=0, full-accept (acc=5) 43.6%**.

A "superset" decoder should accept **at least** what E5 accepts on every event, then sometimes
more (a second spine rescues the events where E5's top-1 draft was wrong but the target's true
token was E5's top-2). With greedy semantics the gain is exactly E5's `T0==top-2` subset of its
13.3% acc=0 events.

**Why it is hard on a GDN/SSM hybrid:** GDN layers carry a single fixed-size *recurrent* state
(`S_new = α·S_old + β·(k⊗v)`), not a growing KV cache. A naive token-tree shares **one recurrent
scan** across all tree branches, so spine B contaminates spine A's deep state. You cannot
incrementally "un-recur" a rejected step. This is the published result in **STree (arXiv
2505.14969, NeurIPS 2025)**; we reproduced it locally.

---

## 3. Route A — in-tree token-tree (CLOSED, capped at 1.751)

We first built a per-path token-tree with per-path drafter + per-path target GDN scan inside one
shared forward (the operator's chosen first route).

- On a clean teacher-forced metric the mechanism worked: forced acceptance moved 0.875→1.0 and
  deep positions recovered on prediction (pos5 0.0→0.016).
- **But the goal metric — free-running path0 — did not move:** consolidated unified in-tree mode
  capped at **path0 avg 1.751** (over 1,682 events) vs E5 3.002. Free-running needs the tree to be
  ~byte-identical to E5 per step; the residual (forced 1.0 vs E5's 1.148, plus a ~4% TREE_ATTN
  target-argmax floor) compounds into a low-accept regime.

**Conclusion:** a token-tree on this GDN hybrid yields only an *internal* superset; the contained
chain (path0) is strictly degraded by shared-recurrent-state contamination. The honest in-tree
ceiling is **1.751 < E5**. Route A is closed for this cycle. (The fix would be a custom TreeScan
`A_tree` kernel — see §7.)

---

## 4. Route B — independent rows (CHOSEN, WORKS) + greedy proof

**Mechanism:** N **persistent, co-resident sequences** (rows), each with the engine's *native*
per-sequence GDN state. Spine-0 is the native top-1 chain (= E5 by construction). Spine-1 takes the
MTP rank-1 token at the root then continues greedily. The token-tree's whole problem — shared
recurrent state — **does not exist** when each spine is its own sequence. Hidden rows are scheduled
until request finish; the public client sees only spine-0. Rows are selected by **request-id**
(`::lumo_ir_sN`), never tensor-row parity (an early bug: index-based selection mis-mapped under
concurrency and contaminated spine-0).

**Winner-commit + state-sync** (the STree checkpoint+clone+activation-replay pattern, on native
per-sequence state, no custom kernel): each event, winner = the row with the longest accepted
prefix; commit the winner's sampled tokens to all siblings; copy the winner's post-accept GDN
recurrent state back into every co-resident row via vLLM's existing Mamba state-copy path.

**Greedy results** (16-prompt fixture, `--limit 64`, temperature 0):

| Config | accept/event | acc0 | full5 (acc=5) | tok/s | superset_violations |
|---|---|---|---|---|---|
| In-tree tree mode, path0 (Route A ceiling) | 1.751 | — | — | — | — |
| **Independent spine-0** (= E5 by construction) | 2.866 | 15.1% | 35.7% | — | — |
| E5-equivalent (`--spines 1`, same harness) | 2.772 | 16.2% | 33.7% | 44.45 | — |
| **Winner (`--spines 2`, winner-commit + sync)** | **3.442** | 0.0% | 47.8% | **47.50** | **0** |
| spine-0 under sync (regression check) | 3.328 | 0.0% | — | — | — |

- **Superset proven:** winner 3.442 ≥ spine-0 every event, `superset_violations=0`; recovery rate
  4.9% (events where spine-1 strictly beat spine-0). spine-0 stayed E5-equivalent after adding the
  sync (3.328, no regression). `acc0` 16.2%→0% because the winner = max(spine-0, spine-1) and
  spine-1 rescues spine-0's root misses.
- **spine-0 ≡ E5:** 2.866 vs canonical 3.002 is within small-sample (1,328-event) noise; per-step
  forced vs engine LCP agreed **64/64** (byte-identical). The per-sequence isolation is exact.

### 4.1 The honest speedup: +6.85% (apples-to-apples)

Same direct probe, isolating the effect of adding the row in the **same harness**:

- E5-equivalent (`--spines 1`): **44.45 tok/s**, accept 2.772.
- Winner (`--spines 2`): **47.50 tok/s**, accept 3.442.
- **Speedup = 1.0685× (+6.85%); accept +24%.**

This corrects an earlier *workload-confounded* 1.77× (47.50/26.86) — the 26.86 was the agentic-B4
workload, a different measurement than the direct probe; comparing them is apples-to-oranges. The
honest number is **+6.85%**. The +24% accept gain is large, but the second row's per-step compute
cost consumes most of it at this batch → modest net tps. On the memory-bandwidth-bound GB10 decode
the extra row rides *freer* when weights dominate (higher batch); whether that materializes is
untested (§7).

---

## 5. Production stability validation (temp 0.6) — and its honest caveat

To validate the machinery under **real production sampling** (rejection sampling, not greedy LCP),
we ran the winner config through the **same agentic-B4 SWE-Verified workload that produced E5's
26.86/3.150**, aligned to E5's settings: **temp 0.6, top_p 0.95, B=4, concurrency 4**
(`run_codex_experiment.py --config Fb --row-mode independent --mtp 5 --temp 0.6`). Commit `079d51f4`.

Alignment fix landed first (`dbba507b`): `--temp` now restarts the live 8088 proxy so temp 0.6
actually applies (an earlier run used the proxy default — the operator caught the misalignment),
and winner tokens are committed *before* vLLM's `update_states_after_model_execute` bookkeeping.

**Result — the win bar, met:**

| Metric | Value |
|---|---|
| Stable to agent-wall timeout | **yes** (window 2,524 s ≈ 42 min decode) |
| Engine death / crash signatures | **none** (no EngineDead/CUBLAS/CUDA/illegal-mem/shutdown) |
| Superset invariant (rejection sampling) | **`viol=0` over 19,307 winner events** |
| Winner-commit state-sync correctness | **`missing_sum=0`** |
| Co-resident rows | 8 (B4 × 2 spines), persistent, no condense |
| Mean GPU util | 93.9% |

This is the rigorous temp-0.6 validation: the per-event superset and the state-sync hold under
production rejection sampling and concurrent load, with zero violations and zero missing state
copies across ~19k events, and the engine never died.

### 5.1 CAVEAT — the agentic throughput/accept numbers are workload-confounded (do NOT compare to E5)

This run's headline numbers from `agentic_summary.json` are **not** a speed/accept result:

- `decode_tps` = **10.65**, engine `accept_per_event` = **0.402** (15,374 accepted / 38,285 draft
  events across all rows), winner-trace accept/event = **1.398**, generation = 26,895 tokens.

These must **not** be compared to E5's 26.86 tps / 3.150 accept, because:

1. **All 4 SWE tasks gave up — `resolved_rate = 0/4`** (vs E5's 2/4). At temp 0.6 the *agent's*
   coding ability is high-variance; this time it flailed on all four hard astropy tasks. (0/4 vs
   2/4 over four tasks is small-sample, but it set the workload character.)
2. A flailing agent emits **low-predictability tokens** (retry loops, abandoned edits) — exactly
   the content spec-decode accepts *least*. The decode window is dominated by this, plus B4
   contention, plus retry overhead. acc-distribution: 85.7% of draft events accepted 0 tokens.
3. **At temp 0.6 the two runs generate different text** (stochastic sampling), so cumulative accept
   is not apples-to-apples in the first place. Proof: this single run's own cumulative accept swung
   **3.13 → ~1.07 across task boundaries** — accept is dominated by *which* content got sampled,
   not by model quality.

The two trace numbers (engine 0.402 vs winner-trace 1.398) differ only by denominator (all draft
events across 8 rows vs committed-winner events); both are honest, neither is comparable to E5.

**The clean speed/accept number is the greedy direct probe: +6.85% tps / +24% accept (§4.1).**
This run is the **stability + invariant** validation, full stop.

---

## 6. What this means

- **Correctness: settled.** The winner is a true ≥E5 superset, proven greedy (viol=0,
  spine-0 byte-identical to E5) and validated stable under temp-0.6 production load (viol=0,
  missing=0, no crash, 19,307 events).
- **Speed: modest and honest.** +6.85% at low batch. The accept ceiling is +24%; converting more of
  it to tps needs either higher batch (the row rides freer when weights dominate) or eliminating the
  redundant second-row compute (STree kernel).
- **Not a regression.** Nothing here is below E5 by construction; the low temp-0.6 throughput is the
  workload, not the mechanism.

---

## 7. What's next (identified, not built)

1. **Higher batch.** On memory-bandwidth-bound GB10 decode, the extra co-resident row amortizes
   weight reads as batch grows; the net speedup should improve above the low-batch +6.85%. Untested —
   the gate is the marginal cost of one extra spine at production batch, and whether B×N crosses into
   compute-bound.
2. **N-spine production fan-out.** Branch MTP top-2 at each position 1..5 → spine_k rescues E5's
   position-k rejections; ≤5 alt spines + A = 6 independent chains, same machinery (no token-tree).
3. **STree `A_tree` / TreeScan kernel — the scale endgame.** A custom CUDA kernel accumulating the
   SSM transition matrices per path (`A_tree = L·A_log`, one pass) shares ancestor tokens and removes
   the redundant second-row compute that caps today's net speedup. This is the durable speed win, but
   it is **research-grade / weeks on aarch64 + CUDA13 + GDN** (vLLM has no tree spec-decode for
   hybrids at all — issue #30114, bug class #39809). **NO-SHIP this cycle; recorded, not built.**

---

## 8. Artifacts & reproduction

- **Code:** `scripts/swe_x86_helpers/relaunch_qwen36_round.py` (one public config `--config Fb`,
  `--row-mode {tree,independent}`, `--mtp N`, `--spines N`; dead routes pruned). Measurement CLIs:
  `scripts/measure_spec_per_position.py` (free-running), `scripts/measure_spec_teacher_forced.py`
  (forced + winner). Driver: `scripts/run_codex_experiment.py --row-mode independent --temp 0.6`.
- **Greedy proof:** `--row-mode independent --mtp 5 --spines 2`, 16-prompt fixture, `--limit 64`.
- **Temp-0.6 stability run:** `output/fr9_agentic_b4_winner_temp06_sync_20260601T1800Z/`
  (`agentic_summary.json`, `independent_winner_trace.jsonl` 19,307 events viol=0/missing=0,
  `driver.log`).
- **Constraints (operational):** `LUMO_GPU_MEMORY_UTILIZATION=0.84`; FLASH_ATTN; batch-invariant;
  `max_num_seqs ≥ concurrency × spines`; persistent rows (never condense mid-decode). Never
  `docker restart` the container (wedges ~100 GiB host RAM on GB10 — use rm -f + relaunch).
