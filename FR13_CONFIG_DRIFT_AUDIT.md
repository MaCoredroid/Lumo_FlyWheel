All facts confirmed. The audit's central discovery: a registry the task's layer list didn't include — `scripts/fr13_canonical_env.sh` (created 2026-07-26, commit d4f31b8bc) — already exists and silently resolved most scout-flagged drifts by promoting them to canonical default-ON. But it also **introduced a new instance of the very disease it was built to cure**, and the two genuine live regressions remain. Full deliverables below.

---

# Config-Drift Audit — lumoFlyWheel FR13 campaign path

Scope traced end-to-end: `fr13_b4_campaign_driver.sh` → `fr13_bigdenom_swe_serve{,_variant}.sh` → `fr13_launch_forked_fa2_tree_server.sh` / `fr13_launch_locked.sh` → container; and the agent leg `offload_codex_proxy.sh` → `relaunch_proxy_remote.sh` → `inference_proxy.py`; plus `run_swe_bench_q36_a.py` docker templates. **Pivotal file not in the task's layer list:** `scripts/fr13_canonical_env.sh` — sourced by the driver at line 23, it is now the de-facto registry and changes the verdict on most scout findings.

## Deliverable 1 — FULL drift table (worst-first)

Verdict legend: **LIVE-DRIFT** = wrong/contested value actually reaches the container/agent on the default campaign path · **HARDENING-GAP** = correct value reaches the path today, but only via a soft default with no fail-loud assert / not centralized (silent-regress risk) · **LATENT-SPLIT** = divergent value only on a non-default fallback branch · **DOC-DRIFT** = stale comment/doc, no runtime effect · **SAFE** = confirmed correct + protected.

| # | env | layer(s) where it lives | current value reaching **default** campaign path | canonical | verdict |
|---|-----|------------------------|-----------------------------------|-----------|---------|
| 1 | **FR13_PARENT_GATHER**, **FR13_CONV_PREGATHER** | `fr13_canonical_env.sh:35-36` defaults **=1**; `fr13_required_tree_flags.sh:35-36` says **REVERTED 2026-07-24, re-gate before re-bake**; launcher `:665/:684` default 0 (inherited 1 wins); patcher `fr10_phase4_...py:551` runs reverted path when ==1 | **=1 live** (empirically: `bar18/bar19/tawcg/…/container_env.txt` all =1) | **CONTESTED** — two registries contradict; `regate_queue.sh` does not exist so the documented re-gate precondition is **unmet** | **LIVE-DRIFT (worst)** |
| 2 | **LUMO_SWE_STALL_KILL_S** | read only in `run_swe_bench_q36_a.py:1015`; **set by no layer** (grep empty across all scripts); `canonical_env.sh:56-60` agent section names other agent envs but **omits this one** | unset → `0.0` → **trace-growth stall watchdog OFF** | documented "600s stall-watchdog" backstop (driver `:26`, runner `:118`) | **LIVE-DRIFT** — dead code; under `WALL=0` gate arms the only hang backstop is agent-side idle |
| 3 | **FR13_ENABLE_APC** | `canonical_env.sh:41` **=1** (fixed); launcher `:230` default 0; **not** in `FR13_REQUIRED_TREE_FLAGS`; not in serve_variant NEEDS `:407-425` | =1 (cache = deliverable) | =1 | **HARDENING-GAP** — correct default, **no fail-loud assert**; direct serve_variant (no driver) → 0 |
| 4 | **FR13_SUBTREE_PARALLEL**, **FR13_FLAGS_INKERNEL**, **FR13_DRAFTER_GRAPH**, **FR13_DM_DEPTHSYNC** | `canonical_env.sh:33,34,37,38` default **=1** (fixed from seq-only); launcher defaults 0; none in the assertion array | =1 (validated: #60 subtree byte-exact +4.7%; dscg/R4 PASS) | =1 | **HARDENING-GAP** — soft default only, no engagement assert; standalone `launch_locked` (doesn't source canonical_env) → 0 |
| 5 | **LUMO_FB_KERNEL_ROWS / LUMO_FB_PROJ_PAD_ROWS** | launcher `:171/:172` empty/16 (OFF); forced `=1/=16` by `launch_locked:48-49` (cat9) + `serve_variant:355` (reshape); asserted `serve_variant:411`; **absent from both registries** | 1/16 on every tree arm (fail-loud if missing) | 1/16 | **HARDENING-GAP** — protected by scattered copy+assert, not registry; reshape-arm worker check `:450/456` weaker than cat9 `serve.sh:203-204` (asserts pid-1 env only, not worker /proc) |
| 6 | **LUMO_PROXY_MAX_OUTPUT_TOKENS** | `relaunch_proxy_remote.sh:46` **=32768** (offload/default); `relaunch_proxy.sh:19` =16384 (legacy) | 32768 (Qwen3.6 thinking) | 32768 | **LATENT-SPLIT** — default path 32768 ✓; OFFLOAD_AGENT=0 fallback truncates to 16384 |
| 7 | **QWEN_STREAM_IDLE_TIMEOUT_MS** | `run_swe_bench_q36_a.py:275` **=600000** (instance_image=default); `:115` =240000 (legacy template) | 600000 (§79) | 600000 | **LATENT-SPLIT** — default ✓; `SWE_AGENT_ENV=legacy` → 240000 |
| 8 | **LUMO_PROXY_SSE_HEARTBEAT_S** | `canonical_env.sh:49` **=15** + `offload_codex_proxy.sh:100` **=15** (double-covered, fixed 4df608b75); `relaunch_proxy.sh` (legacy) never sets it; helper comment `:77-84` stale | **=15 (ON)** on default offload path | 15 | **FIXED** (default path); LATENT on OFFLOAD_AGENT=0 (inherits 15 from canonical via driver); DOC-DRIFT in helper comment |
| 9 | **FR13_APC_CONV_SNAPSHOT** | launcher self-contradicts: `:327 :=1` vs `-e :604 :-0`; patcher default `"1"` | resolves **=1** under APC (`:=1` runs first); `:-0` unreachable when live | 1 | **HARDENING-GAP** — internal default mismatch; footgun if ordering refactored; not in registry/NEEDS |
| 10 | **FR13_APC_BURN_NODE_BANK** | forked `:678` default **0**; `launch_locked:26` forces =1 with stale comment "baked default-on"; assert relaxed (`patcher:7741-7770`) to allow commit=1/init=1/burn=0 | **0** (red-team + live: redundant for served path) | 0 | **DOC-DRIFT** — locked `:26`=1 + comment vestigial; forked burn=0 byte-identical to locked burn=1 for served output |
| 11 | **LUMO_PROXY_AUTO_CONTINUE_MESSAGE** | `relaunch_proxy.sh:8` (soft old text) vs `relaunch_proxy_remote.sh:37` (forceful) | inert — nudge `AUTO_CONTINUE=0` banned + **fail-loud asserted** (`offload_codex_proxy.sh:124`, `serve.sh:328`) | forceful variant | **DOC-DRIFT** — dead unless nudge on; default path uses forceful variant anyway |
| 12 | **FR13_CONV_NODEBANK / FR13_SPEC_BLOCKS_CAP / FR13_CONV_WB_BATCHED** | launcher defaults 0; set =1 only in bar/stack seq files; `required_tree_flags:39-41` = QUEUED comments; coupling (CAP⇒NODEBANK) patcher-enforced fail-loud | **OFF** (queued, re-gate pending) | OFF-by-design | **SAFE** — OFF is the intended state; omission loses a lever, not correctness (opposite risk profile to #1/#2) |

**Confirmed SAFE / no-drift (not enumerated as rows):** the 8-flag `FR13_REQUIRED_TREE_FLAGS` block (ATTN_KV_REMAP, SLOT_REORDER, RING_EXPORT, CONV_WB_FUSED, COMMITTER_BATCHED, KV_REMAP_SYNCFREE, INPUTPREP_GUARD, DRAFT_VOCAB_K) — single-source + fail-loud, 0 seq copies; `LUMO_PROXY_FORCE_TEMPERATURE=0.6`, `LUMO_PROXY_AUTO_CONTINUE=0`, `FR13_SCAN_ALIGN=0`, `FR13_APC_EXACT_SEED=0` — all fail-loud asserted; `GPU_OOM_GUARD`, `DOCKER_MEM_CAP`, `HEALTH_TIMEOUT_S`, `OFFLOAD_LINK_DOWN_MAX_S`, `PYTORCH_CUDA_ALLOC_CONF` — baked chokepoint defaults; `FR10_ENABLE_TREE_GDN` + native-bf16 taps — hardcoded literals (drift-immune); ~150 diagnostic timers — empty/0 = safe-off. **DEAD (correctly):** `FR13_APC_HIT_RECURRENT_SUFFIX` force-off `patcher:899`; `es_ckpt` family asserted-off; `FR13_PIPELINE_LOCK.md` stale (0/14 current flags) but non-load-bearing (launch_locked sources the live registry).

**Two structural (non-env) drift vectors confirmed:** (a) `canonical_env.sh:14-15` header **falsely** claims it is sourced by `fr13_launch_locked.sh` — it is not (`launch_locked:37` sources only `required_tree_flags.sh`); standalone locked launch misses every promoted default (#3, #4). (b) Multi-arm `SEQUENCE_FILE` is `source`d (driver `:99`) so arm-A `export`s leak into arm-B — handled manually today (`seq_if_pair` explicit 0s, `seq_msr` unset), fragile. (c) Driver `run_variant` `:71-76` unconditionally sets `FR13_*_GPU_TIMER=1`, so seq headers claiming "CLEAN zero instruments" are literally false (timers are observer-safe per bv1, but the label violates two-kinds-of-runs discipline).

---

## Deliverable 2 — Centralization design (ONE registry, extended to all 3 layers)

**Do not create a new file.** `scripts/fr13_canonical_env.sh` already *is* the intended ONE registry — the fix is to (i) close its three structural holes, (ii) reconcile it with `fr13_required_tree_flags.sh` instead of contradicting it, and (iii) extend the fail-loud assertion pattern to the proxy and agent layers it currently only *comments about*.

### 2.1 File layout — four explicit tiers in `fr13_canonical_env.sh`

```
scripts/fr13_canonical_env.sh          # THE registry (host-shell exports)
scripts/fr13_required_tree_flags.sh    # ASSERTION spec — the FR13_REQUIRED_TREE_FLAGS
                                        # array = the subset of Tier-A that must be
                                        # fail-loud verified in the live container.
```

Restructure `canonical_env.sh` into labelled tiers so BAKED vs EXPERIMENT is unambiguous:

- **Tier A — BAKED + ASSERTED (tree/serving).** Proven fixes that must be ON on every tree arm. Values here MUST equal the `FR13_REQUIRED_TREE_FLAGS` entries (a preflight diff, below, enforces equality). Today: the 8 required flags + `MAMBA_BLOCK_SIZE/APC_BLOCK_SIZE/MAMBA_SSM_CACHE_DTYPE`. **Add** `FR13_ENABLE_APC`, `LUMO_FB_KERNEL_ROWS=1`, `LUMO_FB_PROJ_PAD_ROWS=16` (promote from scattered copies).
- **Tier B — BAKED default-ON, SOFT (tree/serving).** Validated wins, default-ON, but not yet worth a hard boot-abort. Today: `FR13_SUBTREE_PARALLEL`, `FR13_FLAGS_INKERNEL`, `FR13_DRAFTER_GRAPH`, `FR13_DM_DEPTHSYNC`. Each carries a one-line gate-evidence + commit hash. Promotion B→A = adding its `KEY=VALUE` to the assertion array.
- **Tier C — PROTECTIONS by non-tree layer.** Proxy: `LUMO_PROXY_SSE_HEARTBEAT_S`, `LUMO_PROXY_QWEN_SAMPLING`, `LUMO_PROXY_MAX_OUTPUT_TOKENS`, `LUMO_PROXY_FORCE_TEMPERATURE` (canonical 0.6), `LUMO_PROXY_AUTO_CONTINUE=0`. Agent: `QWEN_STREAM_IDLE_TIMEOUT_MS`, `QWEN_CODE_MAX_OUTPUT_TOKENS`, **`LUMO_SWE_STALL_KILL_S`** — promoted from comment-only to real `export`s (see 2.3).
- **Tier D — EXPERIMENT deltas.** **Never in the registry.** Only in `output/fr13_msr/seq_*.sh`. Today: `FR13_CONV_NODEBANK`, `FR13_SPEC_BLOCKS_CAP`, `FR13_CONV_WB_BATCHED`, `FR13_HC_INTERNAL`, `FR13_TAW`, per-arm `FR13_DRAFT_VOCAB_K` sweeps. Launcher default-OFF is their canonical safe state.

### 2.2 Who sources it (single entry invariant)

- `fr13_b4_campaign_driver.sh:23` — already sources it (all campaign arms). ✓
- **`fr13_launch_locked.sh` — MUST source it** (line ~37, before `required_tree_flags.sh`). Closes the standalone-locked hole (#3/#4) and makes the header's existing claim true.
- Any new driver: source it first, before `SEQUENCE_FILE`. Precedence is correct today (canonical uses `${VAR:-default}`, seq uses hard `export`, seq sourced after → experiment deltas win, baked defaults hold when a seq omits them).

### 2.3 How the assertion pattern extends to proxy + agent

The `required_tree_flags.sh` pattern = build a `NEEDS=("KEY=VALUE" …)` array, dump the **live** env of the running unit, `grep -q "^KEY=VALUE$"` each, `exit 3` on miss. Replicate per layer against the *right* live surface:

- **Tree (exists):** assert against `container_env.txt` (pid-1) AND `worker_environ_needle.txt` (EngineCore /proc). **Gap to close:** cat9 (`serve.sh:203-204`) checks the worker; the reshape/variant path (`serve_variant:450`) only prints. Make the reshape path assert the worker value too (known hazard: `-e` vars get dropped from the curated worker env).
- **Proxy (partial):** `offload_codex_proxy.sh:120-125` already reads the *remote* proxy `/proc/<pid>/environ` and fail-loud asserts `FORCE_TEMPERATURE=0.6` + `AUTO_CONTINUE=0`. **Extend the same block** to assert `LUMO_PROXY_SSE_HEARTBEAT_S` is a positive number (not 0/unset) and `LUMO_PROXY_MAX_OUTPUT_TOKENS=32768`. This is the exact surface that would have caught regression (2) at boot.
- **Agent:** the agent runs in a per-instance docker image; envs are injected as `-e` in `run_swe_bench_q36_a.py` docker-command strings via shell expansion `${VAR:-default}`. So the registry can own the *values* by exporting them in the host shell that runs the runner — the `${VAR:-…}` picks them up. Add a **runner preflight** in `run_swe_bench_q36_a.py` (before dispatch): if `AGENT_WALL_S<=0` (no-wall gate mode) assert `_stall_kill_s() > 0`, else `raise` — so the documented hang backstop can never be silently absent again.

### 2.4 How EXPERIMENT stays separate from BAKED (structural, not by discipline)

Add a **seq-file lint** run by the driver right after it sources `SEQUENCE_FILE`: for every variable name exported by `canonical_env.sh` (Tiers A–C), grep the seq file; if a seq exports a registry var **without** a `# OVERRIDE-JUSTIFIED: <reason>` marker on the same line, `exit`. This makes the hand-copied-seq DRIFT VECTOR structurally impossible for baked flags while still allowing a deliberate, annotated A/B override (e.g. `seq_if_pair` legs). It also catches the reverse of regression (2): a *stale* seq re-pinning a value the registry has since changed.

---

## Deliverable 3 — Minimal migration steps (concrete, ordered)

**Step 0 — OWNER DECISION REQUIRED (do not auto-pick): resolve #1.** `canonical_env.sh:35-36` re-bakes `FR13_PARENT_GATHER=1`/`FR13_CONV_PREGATHER=1` while `required_tree_flags.sh:35-36` says they are precautionarily reverted and `regate_queue.sh` (the documented re-gate) doesn't exist. These directly contradict and both flags are **live =1** in current bar18/bar19/tawcg containers, actively running the reverted code path. Two mutually exclusive corrections:
   - (a) They ARE re-baked → delete the "REVERTED / re-gate before re-bake" comments in `required_tree_flags.sh:35-36`, and record the re-gate evidence + commit hash in `canonical_env.sh:35-36` (replace the bare "lean stack" comment).
   - (b) They are NOT → change `canonical_env.sh:35-36` to `${…:-0}` and confirm `container_env.txt` drops to 0.
   Consistent with the still-standing revert and the loop-escalation history (root-caused to host-driver degradation, reverts held precautionary), **(b) is the conservative default**; but this is a behavior change on a lossless-gated path, so it is an explicit call, flagged per the STOP-and-report rule for table-row factual changes.

**Step 1 — wire the dead watchdog (#2).** In `canonical_env.sh` agent section (currently comment-only `:56-60`), promote to real exports: `export LUMO_SWE_STALL_KILL_S="${LUMO_SWE_STALL_KILL_S:-600}"` (matches the documented 600s backstop), plus `export QWEN_STREAM_IDLE_TIMEOUT_MS="${…:-600000}"` and `export QWEN_CODE_MAX_OUTPUT_TOKENS="${…:-32768}"`. The runner's `os.environ.get`/`${VAR:-…}` picks them up with zero runner edits. Add the runner preflight from 2.3.

**Step 2 — close the split-defaults (#6, #7).** `relaunch_proxy.sh:19` 16384→32768; `run_swe_bench_q36_a.py:115` legacy template 240000→600000. Removes the two LATENT-SPLIT footguns.

**Step 3 — promote scattered baked flags into Tier A (#3, #5).** Add `FR13_ENABLE_APC`, `LUMO_FB_KERNEL_ROWS=1`, `LUMO_FB_PROJ_PAD_ROWS=16` to `FR13_REQUIRED_TREE_FLAGS` (so they gain the fail-loud engagement assert), and delete the redundant hardcoded copies at `launch_locked:48-49` / `serve_variant:355` (they now flow from the registry, matching how ATTN_KV_REMAP was de-scattered). Fix the reshape-arm worker assertion (2.3) so LUMO_FB is checked in the worker /proc, not just pid-1.

**Step 4 — make the header true (#3/#4 standalone hole).** Add `source "$HERE/fr13_canonical_env.sh"` to `fr13_launch_locked.sh` (before its `required_tree_flags.sh` source). Now standalone locked launch gets the promoted defaults; the header claim at `canonical_env.sh:14-15` becomes accurate.

**Step 5 — extend proxy assertion (belt for regression-2 class).** In `offload_codex_proxy.sh` (alongside the existing `:120-125` temp/nudge asserts), add fail-loud checks that the remote proxy env carries `LUMO_PROXY_SSE_HEARTBEAT_S>0` and `LUMO_PROXY_MAX_OUTPUT_TOKENS=32768`.

**Step 6 — cosmetic reconciliations (no behavior change).** `canonical_env.sh`/launcher: change `FR13_APC_CONV_SNAPSHOT` `-e` default `:604` `:-0`→`:-1` to match the block `:=1` (#9); update/remove the stale `FR13_APC_BURN_NODE_BANK` comment at `launch_locked:26` (#10); fix the stale SSE comment block at `offload_codex_proxy.sh:77-84` (#8); align/retire `AUTO_CONTINUE_MESSAGE` string (#11).

**Step 7 — install the seq-file lint (2.4)** in the driver, and add the `SEQUENCE_FILE` cross-arm reset guard (auto-`unset` all registry vars between `run_*` calls in a multi-arm seq) to kill drift vectors (b)/(c).

**Priority:** Steps 0–2 are the actual lost/contested protections (do first); 3–5 are the anti-regression hardening that makes the next drift fail loud at boot; 6–7 are hygiene. Every step is a single-file edit with the file:line cited above; none changes a proven-baked value except Step 0, which is gated on the owner decision.