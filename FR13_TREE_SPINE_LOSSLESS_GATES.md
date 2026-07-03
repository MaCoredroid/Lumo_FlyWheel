# FR13 — Gate ladder to confirm the tree-GDN-kernel carrier + validate the fix, on BOTH tree & spine

**Date:** 2026-07-02. Workflow wtlxfpoil (6 agents) + adversarial verdict (sufficient=False → made runnable).
Carrier established (wkrqdt1gl): cache-seeded ssm_state consumed by `_tree_gdn_kernel`/`_gdn_node_step`
(fp32-carry, fr10_gdn_tree_kernel.py:693) = different rounding than native chunked-FLA (patch:4189-4207);
spine breaks too (mode-gated dispatch). Fix = align rounding OR recompute kernel (FR13_SCAN_ALIGN_MODE=recompute).

## The ladder (ordered; every gate runs BOTH spine=chain5 AND tree=cat8)
**G1 — KERNEL-CONFIRM (deterministic, seed-fixed, cheapest, RUN FIRST).** Capture ONE decode-step tree payload,
diff `gdn_scan_out` per node across 3 arms on the SAME seed: (A) default h_cache `_tree_gdn_kernel`, (B) recompute
kernel (SCAN_ALIGN_MODE=recompute), (C) native FLA replay per root→node path. Split spine-nodes vs branch-nodes.
**PASS:** default-vs-FLA > ~1e-3 on spine AND branch nodes (kernel is the carrier) AND recompute-vs-FLA == 0.0
bit-exact on all nodes (recompute is the fix) → confirms `_gdn_node_step` locus + **exonerates FA2/TREE_ATTN by
elimination** (byte-identical across all 3 arms). REUSE `scripts/fr12_branch_path_oracle_probe.py` (already does
A-vs-C with spine/branch split, probe:236,360). No temp 0.6, no SWE agent, one capture + offline diff.

**G2 — DEPLOYMENT LOSSLESS (temp 0.6, real prompts, no teacher-force).** Sig-0 (ADDED per verdict): temp-0.6
same-prompt q-vs-p TV, default-tree vs native — the one *sufficient* distributional gate every prior conclusion
skipped. Sig-1: carrier_locator_grid cells N / D0,D(spine) / B0,B(tree) same-prompt malformed + accept. Sig-2:
SWE qwen-code resolve on 13453. PASS: D and B each match N. (Sig-1/2 are non-inferiority = necessary-not-sufficient;
Sig-0 TV is the confirmer.)

**G3 — FIX-VALIDATION.** 7 boots cache-ON: {chain5,cat8}×{body,recompute}+native, real SWE. Three legs BOTH
topologies: KERNEL (G1 flips==0 under recompute), BEHAVIOR (Fisher resolve recompute-vs-native p>0.05, accept≥3.2),
SPEED (recompute TPS ≥ 0.90×default). Reuse fr13_apc_e2e_lossless_gate.sh + fr13_measure.py; build thin driver.

## Adversarial verdict (sufficient=False as written) — fixes REQUIRED before first run
1. **BLOCKER / build item 0:** G1 cannot boot — `patch:4346 _fr13_replay_route_on = True` is BAKED, and
   patch:4353-4366 RAISES the moment FR10_TREE_GDN_CAPTURE_PAYLOAD is set. **Must un-bake REPLAY_ROUTE (env-gate
   patch:4346) first.** This is the true blocker, not the probe extension.
2. **Oracle identity:** G1's reference is `fr12_branch_path_oracle_probe._native_path_scan` (probe:246, a fresh
   FLA call on captured q/k/v) — NOT `FR12_NATIVE_SPINE_ORACLE` (a different in-serving substitution that also trips
   the REPLAY_ROUTE raise). Use the probe's FLA-replay.
3. **G1 spine arm is redundant as a second boot:** the probe splits spine-vs-branch nodes WITHIN one cat8 payload
   (spine_nodes = the root→leaf chain), so ONE cat8 capture already proves "spine nodes route to _gdn_node_step and
   break." chain5 becomes a cheap config check (does MODE-gating reach the kernel), not a second capture.
4. **G2/G3 behavior floors only fail-to-reject** — a lossy kernel can resolve 13453 by luck at temp 0.6. Hence
   Sig-0 (the q-vs-p TV gate) is mandatory for the deployment lossless *claim*.

## Focused build order (reuse-first, GPU-free first)
0. **Un-bake REPLAY_ROUTE** (patch:4346 → env-gated) — GPU-free source edit, the blocker.
1. **Extend fr12_branch_path_oracle_probe.py** with arm-B (recompute payload diff) — GPU-free, ~1 arg + 1 branch.
2. **Run G1** on one cat8 capture (default vs recompute vs FLA) — the cheapest decisive gate: confirms the locus
   AND the recompute fix's bit-exactness before any temp-0.6/SWE compute. If G1 fails, G2/G3 are moot.
3. Only then G2 (add the q-vs-p TV Sig-0 + a vs-native reducer + a recompute grid cell) and G3 (lift the
   variant:304-305 hard-block on FR13_SCAN_ALIGN=1 + thin driver).

---

## INVERSION (workflow wk8akwphe, user-driven) — the tree-kernel drift CANCELS; the fix is EXACT_SEED, and it needs VERIFYING not building

The user's two questions (APC = prefill-only; native prefill/decode are also two modes) forced the right frame. Result: **do NOT implement (a) recompute OR (b) tree-produce-cache.**

**Why both are wrong:** cache-OFF tree's cross-turn durable seed is **also FLA-produced** — with APC off, context_lens=0 (patch:793), so regenerated tokens re-prefill COLD through FLA `chunk_gated_delta_rule` whose final overwrites `ssm_state[non_spec]` (patch:6123), which the tree decode reads as h0 (patch:5048). So **both arms seed from FLA and both run the same tree decode.** The tree kernel's per-node/handoff drift (`-0.0→+0.0` flip, kernel:782-787) is **real but IDENTICAL in cache-ON and cache-OFF → it CANCELS in the cache-ON==cache-OFF comparison.** So the tree kernel is **not** the cache-carrier. A tree-produced checkpoint (b) would *create* an FLA-vs-tree mismatch; recompute (a) targets native-exactness we don't need.

**The actual invariant (matches the user's reasoning):** cache-ON == cache-OFF iff the **restored seed == cold FLA prefill** at the boundary. That is exactly what **EXACT_SEED already does**: FLA checkpoint at a 64-aligned boundary + FLA-continued <64 remainder = bit-exact to cold FLA prefill (Sparse-Prefix-Caching Remark 1). **Minimal change = NONE.** Keep `FR13_APC_EXACT_SEED=1`, `HRS=0`, `--block-size 1024`, fp32 ssm cache.

**THE TENSION (unresolved, decisive):** the banked cat8-ON run that was 0/8-empty **had EXACT_SEED=1**. So either (i) EXACT_SEED isn't actually delivering bit-exactness (restore imperfection real), OR (ii) the cat8-ON empty-stalls were the **nudge confound** (qwen-code stalls on forked arms regardless of cache), NOT the cache — in which case the whole cache-carrier hunt partly chased the nudge.

**THE DECISIVE UNRUN GATE (the one thing to do):** cache-ON(EXACT_SEED+1024) vs cache-OFF(same config, APC off), **SAME prompt, temp 0.6, piecewise, N≥3/arm, Fisher-tested, non-vacuity-guarded** (assert real cache hits ES_WRITE>2000 + ES_RESTORE>0 AND spec engaged). PASS = resolve + malformed rate match cache-OFF (NOT byte-identical). It is **designed but NEVER RUN** — the exact failure mode as the historical lossless-gate miss. L0 state-diff PASSED; L1 live proxy was running, unfinished.

---

## ES_OBSERVE RESOLUTION (2026-07-03) — THE TENSION resolved toward (ii): the empties are the agent give-up, NOT ES lossiness

Ran ONE nudge-less qwen-code SWE arm, cat8 cache-ON + `FR13_APC_EXACT_SEED=1` + **`FR13_SERVE_LOG=1`** (so ES engagement is finally observable — it was silenced in the banked 0/8 run), on the 4 forked-stall tasks (13453 13579 13033 13236). Run `output/fr13_es_observe/run_20260702T235258Z`, arm `m_cat8on_obs`. Then read the ES restore code + the L0 success evidence. Verdict on the two horns of THE TENSION:

**(i) "EXACT_SEED isn't delivering bit-exactness" — REFUTED, two independent ways:**
- **Code/state proof (existing):** `FR13_APC_EXACT_SEED_SUCCESS.md` — L0 eager state-diff **PASSED** at `block_size=1024`: 47/48 GDN layers reach fp-level drift (mean ≈ 0.0005); the layer-0 30.11 is a **cross-position measurement artifact** (the *no-cache* cross-position baseline is **38.44 — larger** than the cache's 30.11, i.e. restore introduces *less* difference than the no-cache reference itself). Restore is bit-exact by Sparse-Prefix-Caching Remark 1 (64-aligned FLA checkpoint + FLA-continued <64 remainder).
- **Live engagement (this run):** `ES_GATE bs=832` (**832 = 13×64, 64-ALIGNED**, so the `%64==0` capture gate at patch:6547-6549 FIRED — the align-override-to-816 gap is NOT present here), `ES_WRITE=2795`, `ES_SEED_APPLIED=1595`, `ES_CKPT0_CAPTURE=96`. ES is **ENGAGED + ALIGNED**, not silently disabled. Banked to `.../m_cat8on_obs/es_engage_verdict.txt`.
  - *Config note:* this arm ran at the align-default `bs=832` (I did not force `APC_BLOCK_SIZE=1024`); 832 is 64-aligned so ES fired and is bit-exact by the same Remark-1 argument. The **banked 0/8 cat8-ON run used the validated `--block-size 1024`** and was *also* 0/8 empty — so the empties are independent of {832, 1024}.

**(ii) "the empties are the agent give-up (nudge confound)" — CONFIRMED.** All 4 tasks: `patch.diff = 0 bytes`, and every trace shows the **identical give-up shape — exactly ONE `read_file` tool call, then stop** (`byName.read_file.count=1`, `files.totalLinesAdded=0/totalLinesRemoved=0`, `decisions.auto_accept=1`, 1–2 turns, coherent-but-minimal text like 13579's "I'll start by reading the task description in AGENTS.md"). This is **not** derailed/corrupted output (no char-8, no degenerate loop, no malformed markup) — it is the **known Qwen3.6 interleaved-thinking ~100% attempt-1 give-up** (tracked separately as task #13), amplified because nudge-less qwen-code has no AUTO_CONTINUE net on the forked arm.

**CONCLUSION — the "tree+cache leaning-LOSSY" verdict FLIPS to leaning-LOSSLESS-with-agent-confound.** The cat8-ON 0/8 empties reproduce **with ES proven bit-exact (@1024) AND observed firing+aligned (@832)**, as single-`read_file` agent give-ups — so the carrier of the empties is the **agent (interleaved-thinking attempt-1 give-up), not the cache restore.** Combined with the INVERSION (the tree-kernel drift cancels cache-ON vs cache-OFF because cache-OFF also seeds from cold FLA), the cache-on-tree/spine path has **no remaining identified lossy mechanism.** 

**THE ONE RESIDUAL (still never run, per the historical miss):** the temp-0.6 q-vs-p TV / L1 piecewise-Fisher cache-ON-vs-OFF gate would close this to *certainty* (it would show subtle distributional lossiness if any exists, agent-free). It remains designed-not-run. Everything the existing data + code can establish points to **lossless**; the give-up empties are the agent, and the honest close requires fixing task #13 (interleaved-thinking give-up) so the arm can actually attempt tasks — not a cache fix.

---

## CORRECTION (2026-07-03, user-caught) — the ES_OBSERVE "flip to leaning-LOSSLESS" was WRONG. Cache-ON IS the carrier.

The section above concluded the cat8-ON empties were the nudge-free agent giving up, not the cache. **That is refuted by the control I failed to consult: the SAME nudge-free qwen-code, cat8, with cache OFF, on the SAME tasks.**

**The clean same-agent cache ON-vs-OFF contrast** (agent = nudge-free qwen-code throughout; kernel = cat8 forked tree throughout; the ONLY variable is `FR13_APC_EXACT_SEED`/APC):

| task | cat8 cache-**OFF** (`EXACT_SEED=0`) | cat8 cache-**ON** (`EXACT_SEED=1`, `ES_WRITE=2795`) |
|---|---|---|
| 13453 | **12 turns, 29 read_file, 408B patch** (deep engage) | **2 turns, 1 read_file, 0B** (instant give-up) |
| 13579 | **22 turns, 1433B patch** (deep engage) | **2 turns, 1 read_file, 0B** (instant give-up) |
| native (control) | — | all attempt (13453=408B, 13579=1046B, 13033=980B, …) |

Runs: cache-OFF = `output/fr13_carrier_swe/run_20260702T212430Z/m_cat8off`; cache-ON = `output/fr13_tree_cache_matrix/run_20260702T092119Z/m_cat8on` + `output/fr13_es_observe/run_20260702T235258Z/m_cat8on_obs`.

**Where my ES_OBSERVE reasoning went wrong:** the cache-OFF *solve* I had cited earlier (377B on 13453, 072605Z) was **codex WITH the AUTO_CONTINUE nudge** — a different agent — so I dismissed the OFF-solve as a nudge artifact and concluded "nudge-free agent just gives up." But the **`fr13_carrier_swe` runs ARE nudge-free qwen-code with cache OFF, and they engage 12–22 turns and produce real patches.** Holding the nudge-free agent fixed, turning the cache ON collapses 12–22 turns of engagement into a 1-read give-up. **The carrier is cache-ON on the tree kernel, not the agent.**

**Reconciling with the L0 bit-exact PASS:** L0 (`FR13_APC_EXACT_SEED_SUCCESS.md`) proves the **prefill SSM checkpoint restore** is bit-exact @1024. It does **not** test the **tree-decode branching verify** — the num_spec sibling-fork committer path (greedy-LCP over the co-resident h_cache scan) that the spine-5 lossless proof never covered (see `project_fr13_statediff_no_drain`, `project_fr13_tree_cache_lossy`). So "prefill restore bit-exact" and "cache-ON degrades cat8 tree decode enough to flip the agent to give-up" are **both true and not contradictory**: the lossy channel is downstream of the bit-exact restore, in the branching verify. The INVERSION (tree-kernel seed drift cancels cache-on-vs-off) covered the recurrent SEED only, NOT the decode trajectory — and the empirical trajectory does NOT cancel (accept 3.39→2.3-3.0; attempt→give-up).

**CORRECTED VERDICT: leaning-LOSSY stands — now with the strongest evidence yet** (clean same-nudge-free-agent cat8 cache-OFF-engages vs cache-ON-gives-up on 13453/13579). Residual caveats: N=2 both-arm tasks; the two arms are different boots (a same-boot cache-ON-vs-OFF L1 Fisher / the temp-0.6 q-vs-p TV gate would make it airtight — still the one unrun gate). But the agent-give-up explanation is dead: the same nudge-free agent solves cat8 with cache OFF.
