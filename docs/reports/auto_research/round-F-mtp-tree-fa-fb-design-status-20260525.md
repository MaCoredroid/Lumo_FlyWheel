# Round F — MTP top-k branching tree: F_a (tree-attn) vs F_b (batched-paths)

**Generated:** 2026-05-25
**Status:** Design + implementation record. F_a reached `propose_tree` (TREE_ATTN
active) but **crashed on a vLLM 0.19.0 M-RoPE bug** (`propose_tree` references
`self.positions`, which M-RoPE models don't allocate). A narrow text-only
M-RoPE `propose_tree` patch is now in the prelaunch; **pending: canary (draft=6 +
`inv`) → no target-verify crash → B-1/B-2/B-3 lossless gate → only then the
16-task run.** F_b not yet built. **No measurement results yet.**
**Last updated:** 2026-05-25 (post-crash; supersedes the earlier "serving/pending
canary" status).
**Builds on:** `round5-rd-spec-mtp-suffix-harness-codesign-20260520.md` (§10.1 open
question: MTP tree topology) and `round5-b4-sweep-runbook-20260525.md` (the D vs
E-mtp{1,2,3,6} linear sweep).

---

## 1. Goal

Round F tests the one MTP lever the Round-5 B=4 sweep did **not**: instead of a
single linear draft chain (configs E1/E2/E3/E6), submit **many more drafting
candidates drawn from the MTP head's top-k probability distribution**, arranged
as a branching tree, and verify them. Same everything else as the sweep: native
Qwen3.6 MTP head, `num_speculative_tokens=3`, **B=4, temp 0.6, top_p 0.95**, the
same 16 SWE-Bench Verified astropy instances, same Codex harness, same realized
KV regime. Lossless by construction (drafter-side; rejection sampling preserves
the target distribution) — so the gate is B-1/B-2/B-3 distribution-equivalence +
a "did the tree actually run" canary, **not** the SWE-Bench G2 quality gate.

The depth sweep plateaued (E3 > E6), so *depth* is exhausted. **Branching adds
first-token diversity rather than depth** — the untested hypothesis.

---

## 2. Two ways to verify multiple candidates

Both verify the *same* candidate set; they differ in **how** the target model
sees it, and therefore in attention backend, scaling, and engineering cost.

### F_a — tree-attn (one packed tree, tree mask, one verification pass)
The MTP head proposes a branching tree; the tree is flattened into one sequence
and verified in a **single target forward** using a **tree attention mask** (each
node attends only to its ancestors + prompt). This is vLLM's native
`propose_tree` path + `TreeAttentionBackend`. The shared prompt prefix is
processed once; shared internal nodes are computed once.

### F_b — batched-paths (K linear paths as separate sequences, FlashAttention)
Enumerate the tree's root-to-leaf paths and verify each as its own sequence in a
batch, on ordinary causal (FlashAttention) attention — no tree mask. The shared
prompt prefix is deduplicated via prefix caching, but shared *internal* path
tokens are recomputed per path.

### Why this distinction is the whole point

- **Both capture the main spec-decode win.** In decode we are bandwidth-bound on
  the 27B FP8 weight read; if F_b batches its K paths into one forward, it
  amortizes that weight read just like the tree. "One pass vs many" is *not* the
  real axis.
- **Scaling with candidate count is the real axis.** Tree-attn node count grows
  **additively** and shares ancestors; batched-paths **path** count grows
  **multiplicatively** (`b^d` full-length sequences) and recomputes shared
  ancestors. For "way more candidates" (the stated goal), tree-attn is the only
  one that scales; batched-paths is viable only for a handful of paths.
- **Backend / correctness / effort:**

  | | F_a tree-attn | F_b batched-paths |
  |---|---|---|
  | Attention backend | `TREE_ATTN` (target off FlashAttn → confound vs E3; GB10 kernel perf unproven) | FlashAttn (validated; clean topology-only comparison) |
  | Verification | vLLM-native `propose_tree` + proven rejection sampler (lossless) | **custom** cross-path acceptance (real lossless-bug surface) |
  | Status | **built & serving** (~30-line selector patch) | **not built** (~1–2 wk; not a vLLM-native mode) |
  | Scales w/ candidates | additive | multiplicative (intractable when wide) |

### Decision
Run **F_a first**; it gates F_b's value:
- F_a wins TPS vs E3 → F_b only matters to check if FlashAttn does even better.
- Branching never helps (F_a mean `acc` ≈ E3) → both variants moot; don't build F_b.
- F_a `acc` good but TPS dragged by the GB10 `TREE_ATTN` kernel → *that* is when
  F_b (FlashAttn) is worth the build. A cheaper `TREE_ATTN`+linear control would
  isolate kernel cost from topology before committing to F_b.

---

## 3. Mechanism findings (vLLM 0.19.0, deployed container)

1. **`propose_tree` does genuine top-k branching.** `eagle.py:propose_tree`
   samples `torch.topk(logits, num_children)` at the root and each level — the
   MTP head shares the EAGLE proposer (`use_eagle()` true for `method="mtp"`).
2. **Activation requires `TREE_ATTN` on decoder self-attn — config alone does not
   do it.** vLLM's attention selector has no tree logic, and `VLLM_ATTENTION_BACKEND`
   is **not consumed** in 0.19.0 (verified: env set in-process, model still
   auto-selected FLASH_ATTN). So `speculative_token_tree` set + no backend force
   ⇒ **silent linear fallback** (canary 1 confirmed: `draft=3`, not 6).
3. **Both target and draft need the tree backend.** `TreeAttentionMetadataBuilder`
   serves the draft (`build_for_drafting`) *and* the target verify
   (`build` → the tree attention mask). A draft-only override would mis-compute
   acceptance. ⇒ force `TREE_ATTN` for all decoder self-attn (target stays
   decoder-only; mm/vit encoders use separate backends).
4. **Only regular/uniform trees are supported** (`child_drafts_per_level[L] =
   num_drafts[L] // num_drafts[L-1]` must be exact ≥1).
5. **Realized KV is `auto`/bf16, not fp8.** Bundles say `kv_cache_dtype: fp8_e5m2`,
   but `ModelServer._initial_kv_cache_dtype()` rewrites it to `auto` for fp8
   checkpoints (+ a runtime fallback). The live engine log confirms
   `dtype=torch.bfloat16, kv_cache_dtype=auto`. So **D/E/F all run realized-auto
   KV** — F vs E3 is apples-to-apples on KV, and `TREE_ATTN`'s lack of fp8 support
   is moot (it supports bf16; Qwen3.6 head_dim=256 is in its supported set).
   *Always report configured vs realized KV.*
6. **Config D does not use tree attention.** Arctic's `SuffixDecodingDraft`
   carries `token_ids` **and** `parents` (a real tree), but vLLM's
   `suffix_decoding.py` wrapper returns only `token_ids` and **discards
   `parents`** → linear verification, FLASH_ATTN. (Latent capability: a future
   hybrid could feed Arctic's `parents` into the same `TREE_ATTN` verify path —
   separate config, not F.)
7. **vLLM 0.19.0 tree spec is not M-RoPE-safe (the blocker).** `propose_tree`
   references `self.positions` unconditionally (eagle.py:971/1056/1075), but
   M-RoPE models allocate `self.mrope_positions` instead (only the non-mrope
   branch creates `self.positions`). The linear `propose` path is mrope-aware
   (`_get_positions`/`_set_positions`); `propose_tree` is not. **Qwen3.6-27B is
   M-RoPE** (multimodal Qwen3_5) → `propose_tree` crashes with
   `AttributeError: 'EagleProposer' object has no attribute 'positions'`. F_a's
   `TREE_ATTN` force was correct but exposed an unimplemented combination. No
   public vLLM issue/PR for tree-spec + M-RoPE; the documented Qwen3.6 spec
   config is **linear** MTP. **Fix:** a narrow text-only `propose_tree` patch
   (M-RoPE text-only has identical position IDs across the 3 dims, so 1D tree
   slot math is sound) reusing `_get_positions`/`_set_positions` and feeding the
   draft model 3D positions. Target-verify mrope-correctness is *not* guaranteed
   by this patch — B-1/B-2/B-3 must gate the run.

---

## 4. Implementation (this round)

- **Config F is first-class** (like D's self-contained suffix stack):
  `relaunch_qwen36_round.py --config F --mtp N` → E-style MTP prelaunch + the
  tree-attn selector source-edit + an MTP bundle carrying `speculative_token_tree`.
  Routed through `run_codex_experiment.py` `apply_config` (`--config F`).
- **Default tree** (`_default_tree(n)`): top-2 at root, each extended as a linear
  chain to depth n. For n=3: `[(0,),(1,),(0,0),(1,0),(0,0,0),(1,0,0)]` — two
  parallel depth-3 chains seeded by the MTP head's top-2 first tokens. **6-node
  budget** vs E3's 3-node chain. Small on purpose (test the hypothesis before
  tree-attn/verifier overhead dominates).
- **Tree-attn force = prelaunch source-edit** of `vllm/v1/attention/selector.py`
  (same pattern as the scheduler/parser patches): when `speculative_token_tree`
  is *branching* (`len(tree) > max depth`), force `TREE_ATTN` for decoder
  self-attn. Linear chains (config E) are left untouched.
- **Telemetry extension** (`make_spec_decoding_stats` source-edit): per-agent
  trace now logs `{ts, rid, draft, acc, inv}` where `inv =
  num_invalid_spec_tokens[rid]`. Derives per step: nodes proposed (`draft`),
  accepted path length (`acc`), rejected (`inv`), wasted (`draft-acc`),
  accepted/node (`acc/draft`), and per-step latency (consecutive `ts` deltas).
  "Does branching help" is answered by comparing mean `acc` across F_a/F_b/E3.
  Per-*specific*-branch attribution ("did top-2 ever beat top-1") would need a
  deeper rejection-sampler hook — not added unless wanted.

---

## 5. Measurement design

- **Comparator: existing E3** (`q36a_E3_b4`, n=3 linear). No fresh F0 control —
  the E1/E2/E3/E6 sweep already covers linear MTP at matched conditions.
- **Report by node budget** (F=6 vs E3=3), not "depth 3 vs depth 3."
- **Confound to keep visible:** F = `TREE_ATTN` + tree vs E3 = FLASH_ATTN +
  linear, so the delta combines **topology + backend swap**. For a ship decision
  that is the correct end-to-end comparison (the backend is intrinsic to F). To
  *attribute* a delta, add a one-off `TREE_ATTN` + linear (n=3) control.
- **Each round:** same 16 astropy instances, B=4, temp 0.6, top_p 0.95,
  realized-auto KV, 1800s codex wall + 1800s eval. Incremental commit+push per
  task; raw artifacts only on the DGX (no analysis there).

---

## 6. Current status (2026-05-25)

| Item | State |
|---|---|
| Mechanism investigation | Complete (vLLM 0.19.0 source, deployed container) |
| Config F (pluggable) | Implemented (`relaunch_qwen36_round.py`, `run_codex_experiment.py`) |
| Tree-attn selector patch | Implemented; **vLLM accepted `TREE_ATTN`** (`cuda.py:274`), graph capture clean, served healthy |
| Canary 1 (env-only approach) | FAILED as expected — `draft=3` (silent linear fallback); proved config-only insufficient |
| Telemetry extension (`inv`) | Implemented + compiles |
| Canary 2 (TREE_ATTN forced) | **Crashed**: `propose_tree` reached, but `AttributeError: EagleProposer has no attribute 'positions'` — vLLM 0.19.0 M-RoPE bug (see finding 7) |
| propose_tree M-RoPE patch | Implemented (4-edit prelaunch source-edit, validated end-to-end; reuses `_get_positions`/`_set_positions`) |
| F_a canary (`draft=6` + `inv`) | Pending — relaunch with patch, then verify |
| Target-verify correctness | **Unverified** — B-1/B-2/B-3 required (if target tree positions are also mrope-wrong, byte-exact greedy match catches it) |
| F_a 16-task run (`q36a_F_a_b4`) | Not started (gated on canary + B-1/B-2/B-3) |
| F_b (batched-paths) | Not started; ~1–2 wk build; sidesteps the mrope bug (linear path); gated on F_a result |
| Container state | DOWN (config-F crashed; deliberately not crash-looped) |
| Monitoring loop | cron `dc2c36a0` (10 min) watching the F_a run |

**Operational note:** config-F relaunches on GB10 commonly hit a transient
GPU-mem OOM on first `docker run` and self-heal via ModelServer's
`gpu_mem_util` backoff + host-memory recovery (adds a few minutes; not a fault).

---

## 7. Files

- `scripts/swe_x86_helpers/relaunch_qwen36_round.py` — config F, `_default_tree`,
  `_TREE_ATTN_BLOCK` (selector patch), extended `_SPEC_TRACE_BLOCK` (`inv`).
- `scripts/run_codex_experiment.py` — `apply_config` routes `--config F`.
- Comparator: `output/q36a_E3_b4/` (n=3 linear).
- Subset: `docs/reports/auto_research/swe-bench-concprobe16-verified-instances-20260522.json`.
