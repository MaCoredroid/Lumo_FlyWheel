# FR-13 handoff → codex_fr15 (codex_fr14 hung mid-task)

You are **codex_fr15**, taking over the FR-13 tree-verify work. Read this, then execute.

## Goal
Byte-exact **lossless tree verify**: forked vLLM FA2 (CUTLASS) carrying a tree additive-bias + our tree-GDN kernel, **verify-path only**. Then e2e: accept/event > E5 (3.076), lossless within E5 floor.

## Read first (full context)
- `FR13_FA2_TREE_BIAS_FORK.md` (spec), `FR13_FA2_TREE_BIAS_FORK_RESEARCH.md`
- `FR13_GATEA_DEEP_DIVERGENCE.md` (the FULL diagnosis — read carefully)
- `FR13_FLAGS.md` (flags: wire-our-code vs drift-measure), `FR13_LADDER_LOG.md` (per-commit gate bindings)
- Deep history (optional): prior session transcript `/home/mark/.codex/sessions/2026/06/07/rollout-2026-06-07T06-05-21-019ea0af-93ea-7ff2-9825-40f61de99e65.jsonl`

## Current state
- FA2 fork works; full_attn byte-exact in **decode** (gate-2 PASS, committed).
- GDN conv silu **fixed** (ex2.approx replica via Triton `tl.exp2`, committed; `src/lumo_flywheel_serving/fr13_ex2_silu.py`).
- **REMAINING FRONT:** the forked-FA2/TREE_ATTN **PREFILL** diverges from native FA2/FLASH_ATTN at full_attn **L7 `attn_out_raw`**, which contaminates the GDN prefill state **`h0_state_in`=0.0007** carried into the verify event. The Python backend routes prefill to `flash_attn_varlen_func` correctly; the divergence is the forked-`.so`/TREE_ATTN prefill not byte-exact for the no-bias path (gate-2 only tested decode, not prefill).

## Immediate task (the fix)
1. **Write the prefill patch** in `scripts/fr13_patch_fa2_tree_bias.py`: route the `TreeAttentionImpl` **prefill** branch through `flash_attn_varlen_func` with **NO tree bias**, mirroring the existing **DECODE** patch in that same file — it already imports + calls `flash_attn_varlen_func`; **reuse that exact import + call**. **Do NOT re-discover the import via `docker run --rm` inspection — that is what hung codex_fr14.** `py_compile` to validate.
2. **Offline replay** (no GPU): confirm prefill **L7 `attn_out_raw` + `h0` = 0.0** vs native, reusing the prefill reducers already in `scripts/` (`fr13_prefill_*`, `prefill_full_attn_replay`/`prefill_gdn_state_replay`).
3. **ONE live test:** prefill 0.0 **AND** gate-1 (verify drift) + gate-2 (regular decode) **not regressed**.
4. **Commit + push each step.** Bind both gates to the commit in `FR13_LADDER_LOG.md`.

## Discipline (non-negotiable)
- **ONE GPU at a time** (no concurrent `docker run --gpus`; relaunch crashed captures WITHOUT `--rm`; kill leftover containers/health-loops; `empty_cache`).
- **Memory hygiene:** each forked-FA2 server exit wedges ~90 GB on GB10 → recover via `docker rm` + ModelServer `sync`+`drop_caches` (sudo pw in `.lumo.local.env` `LUMO_SUDO_PASSWORD`); use `--gpu-memory-utilization 0.4` for B=1 diagnostics; verify `nvidia-smi`+`free` before each launch.
- **Offline-iterate BEFORE any live server run** (no GPU thrash).
- **NO reward-hack** (no copy/reroute/splice; our kernel computes, verified vs native; the prefill→`flash_attn_varlen_func` is correct native-path scoping, NOT a reroute of the verify).
- **Commit + push EVERY real step** to `main`; verify it's in HEAD.
- **Grind autonomously** to drift=0 (spine+branch, input→every layer→logits) then the e2e (accept/event > E5) — **do not stop to ask**. After this prefill front, continue any remaining GDN sub-op fronts the same way.

## Then
After the prefill fix lands (h0=0.0) and the verify-path full ladder is 0.0 spine+branch + gate-2 holds → e2e at B=4 vs E5 (`output/fr10_native_mtp5_same8_20260604T210257Z`): bag-TV within floor + accept/event ≥ 3.076.

**Start now: read the docs above, then write the prefill patch (step 1).**
