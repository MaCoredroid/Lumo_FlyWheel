# FR10 Worker Kickoff (codex)

You are the **worker** on the FR10 GDN tree-verifier track. A Claude session is
your **online researcher + red-team**, monitoring you every 10 minutes via tmux
and steering. Work the tasks below. **Do not give up.** When blocked, write the
blocker to `FR10_STATUS.md` (see Comms) — do not silently stall or fake a result.

Repo: `/home/mark/shared/lumoFlyWheel` (on branch `main`, just pulled).

## 0. Read first
- Spec: `docs/reports/auto_research/fr10-gdn-stree-verifier-latest-stack-spec-20260603.md` (read it in full).
- Accepted lossless baseline closeout: `docs/reports/auto_research/fr9-b4-temp06-options-closeout-20260601.md`.
- Mission: a **lossless** single-request GDN/STree token-tree verifier for
  Qwen3.6-27B's hybrid attention + Gated DeltaNet path. **Lossless first, speed
  second — but since we are rolling a NEW kernel, build for both at once.**

## TASK P1 — GDN tree-algebra parity proof  (PRIORITY, do this FIRST; CPU-only, gating)
Spec §7 Phase 1. This is the gate: **no CUDA/kernel work until this passes.**

Build a pure-PyTorch (CPU, host `.venv`) proof that a packed/trunk-sharing GDN
tree evaluator equals serial per-path evaluation. Ground it on the REAL Qwen3.6
delta-rule, not a guess:
- Serial per-path reference already exists — REUSE it:
  `/tmp/vllm-0.22-src/vllm-0.22.0/vllm/model_executor/layers/mamba/ops/cpu/recurrent_gated_delta_rule.py`
  (`recurrent_gated_delta_rule()`, `gdn_gating()`, `l2norm()`,
  and `chunk_gated_delta_rule()` as a second independent oracle).
- Real config dims: read the Qwen3.6 GDN layer
  `/tmp/vllm-0.22-src/vllm-0.22.0/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
  (`QwenGatedDeltaNetAttention`) + the model `hf_config` so head counts,
  key/value head_dim, conv, gating, `use_qk_l2norm_in_kernel`, scale all match.

Build three evaluators over a tree descriptor (node_id, parent_id, depth,
sibling_index, token_id, ancestor offsets, per-node position):
1. serial per-path (replay prefix+ancestors(node) for each node) — the oracle,
2. packed-tree (one pass, shared prefix state),
3. trunk-sharing (compute trunk factors once, extend each branch from parent state).

**Pass conditions (all must hold; these are the gates):**
- Every tree-node post-token recurrent state AND output/logit == serial per-path
  reference within a stated tolerance, for every GDN layer and every node.
- Appending a sibling leaf does NOT change ANY trunk node's state/logit.
- Accepted-path final state == serial native decode for that same token path.
- Random small-tree generator: spine depth 1–6, branch width 2–3, nodes {2,3,6,8,14}.
- dtype sweep (fp32 master; bf16/fp16 with documented tolerance).

**Negative controls (RED-TEAM REQUIRES these to FAIL loudly):**
- A sibling that mutates a shared mutable row / reused recurrent state MUST make
  the parity test fail. A trunk-contaminating packing MUST be caught.
- A "longest-accepted hidden winner" selector MUST fail the distribution gate
  (see `tests/test_lossless_selector_gate_c_stub_design.py` for the shape).

Deliverables: reference + tests under `tests/` + script(s) under `scripts/`, and
a written proof `docs/reports/auto_research/fr10-gdn-tree-algebra-proof-20260603.md`
with the derivation (why appending a leaf does not mutate trunk — the within-chunk
delta solve is causal/lower-triangular; gate products are ancestor-path products).

## TASK P0 — Freeze E3/E5 + spines=1 baseline  (start in parallel; long, low-touch)
Spec §7 Phase 0. These are CORRECTNESS reference streams, NOT speed targets.

Measurement protocol MUST match prior E3/E5 exactly:
- B=4 (concurrency=4), temp=0.6, mtp=5, spines=1, gpu_memory_utilization=0.88.
- Dataset: SWE-bench Verified 16-instance subset
  `docs/reports/auto_research/swe-bench-concprobe16-verified-instances-20260522.json`.
- x86 SWE box, agent wall 1800s, eval timeout 1800s (same as accepted arm).
- Accepted baseline tag for reference/comparison:
  `fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z`.

FIRST audit whether the existing accepted-baseline artifacts already satisfy the
P0 reference-stream requirement (greedy token streams, sampled outcomes,
per-event accept counters, engine-step latency, CUDA-graph/capture status). Only
(re)run what is genuinely missing — do not burn a multi-hour campaign to
reproduce artifacts we already have. Record exact stack versions (CUDA, driver,
PyTorch, Triton, vLLM, FlashAttention, model revision) in the run artifact.
Evidence rule: zero-byte / contaminated request-metrics tags are NOT accepted.

## Environment notes
- GB10 (DGX Spark) unified memory — decode is **memory-bandwidth bound**; this is
  the bottleneck the whole kernel must respect. Host venv is CPU-only; GPU is only
  available inside the NVIDIA vLLM container. P1 is CPU-only and runs in `.venv`.
- vLLM relaunch / host-memory recovery goes through ModelServer (sync+drop_caches+
  swap cycle). Do not `docker restart` to bring vLLM up — it wedges ~100GiB.

## Comms protocol
- Keep a live `FR10_STATUS.md` at repo root: current phase, what passed, what is
  blocked, exact commands run, and any question for the researcher. Update it
  whenever state changes. The Claude researcher polls it every ~10 min and steers
  you via tmux. Surface blockers early — that is how you get unblocked, not by
  guessing. Commit deliverables; never fabricate a passing gate.

Start with TASK P1. Confirm you've read the spec, then begin.
