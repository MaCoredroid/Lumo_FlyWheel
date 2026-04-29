# L0c canary architecture decisions

Short, operator-facing record of the L0c-DeltaNet canary's architectural choices, captured during the canary v3 → v4 → v5 sequence on real GB10 hardware. These are the live settings — change them only if the underlying evidence shifts.

## Decision 1 — parity fixture uses 1 probe (down from 16)

**Setting:** `--probe-count 1` for `regenerate_deltanet_parity_fixture.py`. Total fixture build cost is 1 probe × 3 reproducibility runs ≈ 15 min after vLLM cold start. (`build_parity_fixture.py` enforces `--reproducibility-runs=3` for production fixtures, so 3 runs is the floor regardless of probe count.)

**Why we used to use more probes:** The fixture format inherits from a 64-probe HLD-original baseline, later truncated to 16 in the heavy family. The implicit assumption was "more probes = more chances to catch a divergence."

**Why 1 is sufficient on this kernel:**

- **The chunked-delta DeltaNet kernel is data-oblivious in branching.** Tile shape (K=128, V=128, BT=64) is fixed regardless of prompt content. Input data only varies the numerical values flowing through `tl.dot`, not the control flow exercised. Once one probe long enough to trigger the full recurrence runs, all kernel branches are exercised.
- **Empirically, parity verdicts are binary on this gate.** From canary v3:
  - iter-2 (`cache_modifier=".cg"` on w/k loads): bit-identical across all 16 probes.
  - iter-3 (`evict_first` on h0 loads): failed at probe 0 with overshoot 0.309.
  - iter-1 (`evict_first` on h0 loads, different position): only diverged at probe 8 because of a vLLM disconnect, not parity.
  - **Discrimination was never "passes 12/16 vs 14/16."** It was "pass everything" or "fail at probe 0."
- **Per-probe wallclock dominates.** ~5 min/probe on GB10, with the dominant cost being the LUMO_P2B state-checkpoint GPU→CPU memcpy on the unified-memory architecture (task #28). 16 probes = 80 min of agent apply-and-test for no extra discrimination.

**Operational impact:** Iteration wallclock drops from ~3 hr (cold start + 16-probe agent run + 5 min cold restart + 16-probe controller verification re-run) to ~30 min (cold start + 1-probe agent run + 1-probe controller verification).

**Caveat — when to re-add probes:** if a future kernel mutation surfaces a bug that triggers only at a specific token-length boundary, attention-head saturation pattern, or numerical precision corner, 1 probe will miss it. Re-add probes (or add a single probe that exercises that specific shape) only after a known regression pattern requires shape-diversity coverage. Premature multi-probe is overhead.

---

## Decision 2 — keep Design A: fresh agent per iteration (do NOT switch to a persistent agent that self-corrects)

**Setting:** orchestrator spawns a brand-new Claude subprocess per iteration. Agent writes one mutation, calls `apply-and-test`, exits. Next iteration = fresh agent. Persistence is externalized to `mutations_rejected.tsv` + `candidates/<NNN>/parity_check.json` + `candidates/<NNN>/BLOCKED.md` — the next iter's agent reads those files via its prompt's *Reading prior-iteration history* section.

**Alternative considered:** one persistent agent across all attempts in the same session — agent reads its own parity verdict, hypothesizes what went wrong, retries. Reasoning chain stays intact across iters.

**Why we kept Design A** (web research + canary v3 evidence):

- **Karpathy's own `karpathy/autoresearch` (March 2026, 630 lines)** uses Design A: fresh attempt per call, persistence via git commits + `run.log`. Cerebras's post-mortem on autoresearch defends the choice explicitly: *"one experiment per call was a deliberate architectural choice — prevents context overflow, clean error recovery, state separation through git history."*
- **Pure persistent-agent has a documented failure mode:** *"drift within hours"* with infrequent verifier check-ins (Cerebras + Kingy/autoresearch retros).
- **AutoKernel** (RightNow AI — closest analog to our Triton/CUDA pipeline) uses persistent agent BUT with explicit abandonment conditions (5 consecutive reverts → reset target). That's not "pure Design B" — it's a hybrid with reset triggers built in. The reset trigger is the load-bearing piece; the persistence is convenience.
- **Best-of-N at N=8 with strong reasoners** beats serial refinement on kernel optimization (Kevin/KernelAgent: 36% → 72% at N=8 with strong models). With Claude Opus 4.7 xhigh and `--total-attempt-cap 8`, we are in the regime where best-of-N (= Design A) is the empirically winning choice.
- **Crash isolation matters when wallclock is expensive.** A persistent agent that crashes 6 attempts in costs 6 × 30 min of work. A fresh agent that crashes costs the current attempt only.

**What we already get from "Design A + structured failure history" (Karpathy's "context engineering" thesis):**

- The agent prompt at `auto_research.py:5728` (`L0C_ITERATION_BRIEF_TEMPLATE`) tells each fresh agent to read `mutations_rejected.tsv`, `results.tsv`, and `candidates/<NNN>/parity_check.json` before proposing.
- The controller writes BLOCKED.md with its canonical verdict (per `auto_research.py:6796` agent-loop hardening commit `6fb4908`).
- The agent prompt explicitly warns about autotune-cache divergence between agent's apply-and-test and controller's authoritative re-run.

**When we'd revisit:** if cap=8 attempts converges much slower than expected (e.g., < 25% acceptance rate after many runs across multiple kernels), reconsider hybrid: keep fresh-agent outer loop but add a synthesized `digest.md` of patterns observed across rounds (not just per-round). Reset triggers from AutoKernel: `consecutive_compile_fails ≥ 2` (already set as `L0C_COMPILE_FAILURES_THRESHOLD`).

---

## Sources

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [Cerebras: how to stop your autoresearch loop from cheating](https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating)
- [MarkTechPost: AutoKernel](https://www.marktechpost.com/2026/04/06/rightnow-ai-releases-autokernel-an-open-source-framework-that-applies-an-autonomous-agent-loop-to-gpu-kernel-optimization-for-arbitrary-pytorch-models/)
- [Karpathy on context engineering (X)](https://x.com/karpathy/status/1937902205765607626)
- canary v3 evidence: `output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-deltanet-20260429T053345Z/`
