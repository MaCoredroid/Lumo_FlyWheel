# FR13 APC — Unified Engagement Gate (design + draft)

**Status:** design + draft only. NO GPU/docker launched. Code study against branch
`fr13-apc-ssm-shadow`, HEAD `99ee7fc0`.

## The problem (recurring, expensive)

An APC fix FLAG can be set to `1` yet NEVER ACTUALLY ENGAGE, and the run's verdict is
trusted as if the fix were active = a VACUOUS gate. There are **THREE places** a fix can
be vacuous, each independently documented this campaign:

| # | Vacuity mode | Where it lives | Documented instance |
|---|--------------|----------------|---------------------|
| (1) | `e` — flag not passed via docker `-e` | launcher `fr13_launch_forked_fa2_tree_server.sh` (the `-e FR13_APC_*` block) | EXACT_SEED: launcher lacked the `-e FR13_APC_EXACT_SEED` passthrough → `.Config.Env` had no EXACT_SEED → bridge `EXACT_SEED=ABSENT` |
| (2) | `bridge` — flag dropped by worker env curation / not bridged by the sidecar | `_fr13_write_apc_env_sidecar` keys list (L17178) + the env-bridge marker print (L1352) | SHADOW/EXACT_SEED: `present` filter dropped EXACT_SEED (never in `os.environ` of pid-1 because (1)); env.flag had only 4 keys → bridge `EXACT_SEED=ABSENT` in the worker |
| (3) | `predicate` — the code-path PREDICATE never matches so the fix body never runs | the fix's engagement site in the GDN forward / scheduler | CONV_SNAP_FIX: redirect required `func=='get_conv_copy_spec'` which never matched → `FR13_CONV_REDIRECT_FIRED=0` even with the flag on. EXACT_SEED cross-turn: ckpt keyed by `req_id` (changes per turn) → `ES_REDIRECT_USED=0`, all `ES_DRAIN_NOCKPT` |

Existing per-fix checks are PARTIAL — they only cover **part (a)** (flag=1 in the worker
via the bridge marker), never **part (b)** (the fix's code body actually executed). The
documented vacuity instances were all part-(b) failures that a flag-only check passes.

## The deliverable

ONE manifest-driven gate that, for EVERY active APC fix flag, asserts BEFORE any
SWE/lossless verdict is trusted:

- **(a)** the flag is `1` in the WORKER (bridge marker / sidecar), AND
- **(b)** the fix's ENGAGEMENT COUNTER incremented `>0` (the code body actually executed
  on a cache-hit / `num_accepted>1` event).

If either is `0`/absent → **FAIL LOUD, verdict UNTRUSTED** (exit non-zero before any
solve-rate / lossless number is recorded). This is the
[[feedback_fail_loud_assert_engagement]] principle generalized from the per-fix
`FR13_CONV_REDIRECT_FIRED` / `ES_REDIRECT_USED` / bridge-marker checks into one gate.

## Per-fix manifest (the load-bearing table)

For each APC fix flag: the flag name, its bridge-marker field, its engagement counter
(global that increments when the fix body runs), the code site, the bridge field that
proves it reached the worker, and which vacuity mode it is exposed to.

| Flag | Bridge-marker field (L1352) | Engagement counter | Code site (counter / fix body) | Bridge field proving worker arrival | Vacuity exposure |
|------|----------------------------|--------------------|--------------------------------|-------------------------------------|------------------|
| `FR13_APC_SNAP_FIX` | `SNAP_FIX` ✓ | `gdn._FR13_SNAP_FIX_FIRED` ✓ (exists) | bump: patcher L12284-12286; body: redirect `src_ptrs_np[ent]=state[leaf]` L12279 | `SNAP_FIX=` in marker | predicate (leaf-miss → no fire), bridge |
| `FR13_APC_CONV_FIX` / `CONV_SNAP_FIX` | `CONV_FIX` ✓ (note: marker prints `CONV_FIX` not `CONV_SNAP_FIX`) | `gdn._FR13_CONV_REDIRECT_FIRED` ✓ (exists) | bump: patcher L12288-12289 (gated by `_fr13_fx_conv_on`); body: same redirect | `CONV_FIX=` in marker | **predicate (the documented `func==` mismatch)**, bridge |
| `FR13_APC_SNAP_FIX_ZEROACCEPT` | `ZEROACCEPT` ✓ | **MISSING** (no `_FR13_ZEROACCEPT_FIRED`; publish happens inside `_fr13_publish_apc_ssm_leaf` zero-accept branch L6937+ but no counter) | publish site `_fr13_publish_apc_ssm_leaf` ~L6829-6960 | `ZEROACCEPT=` in marker | predicate, bridge — **needs adding** |
| `FR13_APC_HIT_SUFFIX_CAP` | `HIT_SUFFIX_CAP` ✓ | **MISSING** (cap is a value not a fire-site; inert while HRS=0) | n/a until HRS path | `HIT_SUFFIX_CAP=` in marker | bridge (the documented cap-defaulted-to-64), predicate — **needs adding when HRS on** |
| `FR13_APC_HIT_RECURRENT_SUFFIX` | `HIT_RECURRENT_SUFFIX` ✓ | **MISSING** (`_fr13_post` recurrent-exact state ~L5609; no fire-counter) | HRS recompute site ~L5609-5623 | `HIT_RECURRENT_SUFFIX=` in marker | predicate, bridge — **needs adding** |
| `FR13_APC_SHADOW` | `SHADOW` ✓ | **MISSING** (the scheduler-zero engagement at L6399-6408 increments NOTHING; the `_FR13_APC_SHADOW_STATE["written"]` counter is the shadow-LOG record count, a DIFFERENT diagnostic, only when `SHADOW_VALUE`/log path fires) | scheduler-zero: `_patch_scheduler_apc_shadow` L6399-6408 | `SHADOW=` in marker | predicate (`num_new_local_computed_tokens>0` must hold = a real hit), bridge — **needs adding** |
| `FR13_APC_EXACT_SEED` | **MISSING from marker print (L1352)** | **MISSING (no impl on this branch; `ES_REDIRECT_USED` lives only on worktree)** | n/a on this branch | **NO bridge field; NOT in sidecar keys (L17178); NO `-e` in launcher** | **ALL THREE (e + bridge + predicate)** — most exposed |

## What is already present vs MISSING (the fix list this design produces)

**Counters that EXIST (reusable):**
- `gdn._FR13_SNAP_FIX_FIRED` (patcher L12284) — increments on every applied SSM redirect.
- `gdn._FR13_CONV_REDIRECT_FIRED` (patcher L12288) — increments on every applied conv redirect.

**Counters MISSING (must be ADDED, each a `gdn._FR13_<X>_FIRED` int bump at the fix body):**
- `_FR13_SHADOW_FIRED` — bump inside `_patch_scheduler_apc_shadow` right where it zeros
  `num_new_local_computed_tokens` (L6406-6407). This is the ONLY proof the re-prefill
  zeroing actually ran on a real hit. (Scheduler runs in pid-1, NOT the worker — so this
  counter must be exposed from the scheduler process, see "Counter exposure" below.)
- `_FR13_ZEROACCEPT_FIRED` — bump in `_fr13_publish_apc_ssm_leaf` zero-accept branch.
- `_FR13_HRS_FIRED` — bump in the HRS recurrent-suffix recompute site (~L5609).
- `_FR13_EXACT_SEED_USED` (== the worktree `ES_REDIRECT_USED`) — when EXACT_SEED is merged.

**Bridge gaps (must be ADDED):**
- Add `EXACT_SEED` to the marker print (L1352) + the sidecar keys list (L17178).
- Add `-e FR13_APC_EXACT_SEED` to the launcher (the documented (1) fix).

## Counter exposure: two process domains

The engagement counters live in TWO different processes, and the harness reads them
differently:

1. **GDN-forward counters** (`_FR13_SNAP_FIX_FIRED`, `_FR13_CONV_REDIRECT_FIRED`,
   `_FR13_ZEROACCEPT_FIRED`, `_FR13_HRS_FIRED`, `_FR13_EXACT_SEED_USED`) run in the
   **EngineCore WORKER** (the GDN custom op). Exposure: write them to a `/logs` JSONL/flag
   alongside the bridge marker, OR `logger.warning` them (→ docker logs). The existing
   `FR13_CONV_REDIRECT_FIRED=%d` logger.warning (L12291) and the bridge marker file are the
   template. **Recommendation: extend the bridge marker file write to ALSO snapshot the
   live counter values at an `atexit`/periodic hook** so the harness reads ONE file.
2. **Scheduler counter** (`_FR13_SHADOW_FIRED`) runs in **pid-1** (the scheduler), not the
   worker. It must be written to its own `/logs/fr13_apc_engagement.flag` from the scheduler
   process (a small `atexit` or a periodic flush keyed off the same env).

To make the gate uniform, write ALL engagement counters to a single host-readable file
`/logs/fr13_apc_engagement.flag` (one `KEY=N` per line), written by both the worker
(GDN-forward counters) and pid-1 (scheduler SHADOW counter). The gate greps that file the
same way it greps the bridge marker.

## The manifest (machine-readable)

`scripts/fr13_apc_engagement_manifest.json` — per fix: the flag env, the required value, the
bridge-marker field, and the engagement-counter key. The gate iterates it. A NULL counter key
means "counter MISSING — must be added before this fix can be gated" and the gate FAILS LOUD on
it (so a flag with no counter can never be silently trusted). See the committed JSON.

## The gate (draft)

`scripts/fr13_apc_engagement_gate.sh` — sourced/called by the harness AFTER boot + the first
cache-hit turn, BEFORE recording any verdict. For each manifest entry whose flag is REQUIRED `1`:
1. assert the bridge-marker file (`/logs/fr13_apc_bridge_loaded.flag`) shows `FIELD=1` (part a);
2. assert the engagement file (`/logs/fr13_apc_engagement.flag`) shows `COUNTER>0` (part b);
3. if the manifest counter key is `null` (MISSING) → FAIL LOUD ("fix X has no engagement
   counter; add one before trusting any verdict");
4. any bridge-error flag present → hard fail.
Exit non-zero on any failure; the harness must not record the SWE/lossless number.

## Vacuity-mode coverage matrix (does the gate catch each of the 3 modes?)

| Mode | Caught by | How |
|------|-----------|-----|
| (1) `e` not passed | part (a) | flag never reaches worker → bridge marker shows `FIELD=ABSENT`/`0` → FAIL |
| (2) bridge drop | part (a) | sidecar `present` filter drops it → env.flag missing key → marker `ABSENT` → FAIL |
| (3) predicate never matches | part (b) | flag=1 in worker (passes a) but counter stays 0 → FAIL |

The KEY insight the campaign learned the hard way: modes (1)+(2) both surface as
`FIELD=ABSENT` in the bridge marker (part a), but mode (3) ONLY surfaces in the engagement
COUNTER (part b). A flag-only gate passes mode (3) vacuously — which is exactly how the
CONV `func==` mismatch and the EXACT_SEED `req_id` keying both "looked validated".

## NON-VACUITY of the SHADOW counter itself (red-team)

`_FR13_SHADOW_FIRED` must bump at the LINE THAT ZEROES the hit (L6406-6407), NOT merely
where the env is read — otherwise the counter increments even on the `num_new_local_computed_tokens==0`
no-op path and is itself vacuous. The predicate it proves is "a real prefix-cache hit was
suppressed", which requires `num_new_local_computed_tokens > 0` at entry (already the `if`
guard at L6403). Bump inside that `if`, after the zeroing, so SHADOW_FIRED>0 ⟺ at least one
real hit was re-prefilled.

---

## FAMILY ADDENDUM — SSM REPLAY COMMIT PATH (write-side) — 2026-06-28, code re-verified

Scope: `FR13_REPLAY_ROUTE` / `_tree_gdn_replay_kernel` / `launch_tree_gdn_replay` /
`_fr13_replay_durable_ab` / `FR13_APC_BLOCK_ALIGN_45477` / `FR13_APC_HIT_RECURRENT_SUFFIX`.
These are the WRITE side (the committer + scheduler), distinct from the RESTORE-side redirects
(SNAP_FIX/CONV) the original table above covers. They were ABSENT from the original manifest;
I added rows 8-10 (REPLAY_ROUTE, REPLAY_DURABLE_AB, BLOCK_ALIGN_45477).

**Scorecard (verified against HEAD 99ee7fc0):**

| flag | bridge field | engagement counter | code site (verified) | vacuity mode | gap |
|------|--------------|--------------------|----------------------|--------------|-----|
| `FR13_REPLAY_ROUTE` | **none** (baked `:=1`, `-e :435`, not in marker) | **MISSING** | `launch_tree_gdn_replay` `fr10_gdn_tree_kernel.py:1073` (kernel `:908`); callers patcher `:8629`,`:9281` | predicate (silent fall-back to non-replay commit) | ADD `_FR13_REPLAY_LAUNCH_FIRED` — the SSM commit path has NO counter at all |
| `FR13_REPLAY_DURABLE_AB` | none (own sidecar flag, not APC sidecar) | `_FR13_RDAB_RECORDS` (`:6826` def, `:7072` bump) **EXISTS** | `_fr13_replay_durable_ab` `:7086`; caller `:8665` | predicate (`_fr13_rdab_layer_match` + enable guard) | surface `_FR13_RDAB_RECORDS` in the dump; rdab JSONL existence is a proxy |
| `FR13_APC_BLOCK_ALIGN_45477` | **MISSING from marker** (IS in sidecar keys `:17181` + `-e :416`) | **MISSING** | inject `:6361-6373` (block-END align in `_patch_scheduler_mamba_block_align_45477`) | bridge (marker omits) + predicate | ADD `_FR13_APC_BLOCK_ALIGN_FIRED` (scheduler-pid1) + add to marker print `:1352` |
| `FR13_APC_HIT_RECURRENT_SUFFIX` | `HIT_RECURRENT_SUFFIX` (in marker `:1350`) | **MISSING** | flag-gated body `:5548-5720` (`_fr13_apc_active` at `:5548`; `if _fr13_apc_active:` at `:5553`) | **e** (no `-e` line in launcher — only baked+exported `:275-277`) + **bridge** (NOT in sidecar `keys[]` `:17178-17190`) + **predicate** (no counter) | ADD `-e FR13_APC_HIT_RECURRENT_SUFFIX` + sidecar key + `_FR13_HRS_FIRED` |

**Two corrections to the original table's line numbers (verified live):**
- HRS counter site is the flag-gated block at `:5548-5720` (the `if _fr13_apc_active:` at `:5553`),
  NOT "~L5609 `_fr13_post`". `_fr13_post` (`:5626`) is an intermediate state list inside the body,
  not the engagement boundary; bump `_FR13_HRS_FIRED` once when `_fr13_apc_active` is true and a
  cache-hit row was found (`len(_fr13_hit_rows) > 0`), so the counter proves a real hit-row
  recompute, not merely "flag on".
- HRS additionally has an **`-e` HOLE**: the original doc listed HRS only under bridge/predicate,
  but it is also mode-**e** exposed (no `-e FR13_APC_HIT_RECURRENT_SUFFIX` in the launcher docker
  block, and it is not in the sidecar key list). So HRS is exposed to ALL THREE modes, like
  EXACT_SEED. This is the same class of bug as the EXACT_SEED launcher `-e` miss.

**Where the write-side counters must be dumped from:**
- worker-process counters (`_FR13_REPLAY_LAUNCH_FIRED`, `_FR13_RDAB_RECORDS`, `_FR13_HRS_FIRED`) →
  the worker's `/logs/fr13_apc_engagement.flag` (same file/flush hook as SNAP_FIX/CONV).
- scheduler-pid1 counter (`_FR13_APC_BLOCK_ALIGN_FIRED`) → written from pid-1, same domain split
  as `_FR13_SHADOW_FIRED`. The gate greps the merged file.

**Precondition discipline (so a 0 counter is honest):**
- write-side fixes (REPLAY_ROUTE, REPLAY_DURABLE_AB) → precondition `num_spec_decodes>0` (a
  spec-decode commit happened), NOT a cache hit. A pure-decode-only / no-spec arm legitimately
  has `_FR13_REPLAY_LAUNCH_FIRED==0`; the gate must not flag that as vacuous when the arm is
  no-spec. Encode `precondition` per row (added to the JSON).
- BLOCK_ALIGN_45477 → precondition: the unaligned-resume branch
  (`num_computed_tokens_after_sched < last_cache_position`) on an APC arm. On an arm whose hits
  all land block-aligned, it legitimately never fires.

**Why this family is the highest-value to instrument:** the no-spec+cache 2/2 proof
([[project_fr13_apc_spec_specific_carrier]]) shows the SWE break is spec-decode-specific
(`num_accepted>1` commit), i.e. on the WRITE side this family owns. Yet this family had the
LEAST engagement instrumentation (replay kernel + HRS = zero counters). A vacuous WRITE-side fix
is exactly the failure that produced the "looked validated" EXACT_SEED arc.
