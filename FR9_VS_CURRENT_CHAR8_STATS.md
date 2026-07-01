# FR9 (8/16) vs CURRENT: char-8 luck vs statistically-more analysis

**Question:** Is fr9's 8/16 just LUCK (char-8 roughly as frequent, fr9 happened not to hit it on
the critical apply_patch turns of solved tasks) — OR does the CURRENT pipeline STATISTICALLY
generate MORE char-8 (especially on apply_patch, especially with cache ON)?

**Read-only.** No GPU/inference/docker touched. All numbers re-verified against the raw files.

---

## VERDICT (crisp)

**fr9 is NOT proven lucky, and the current pipeline is NOT proven to generate more char-8 —
because fr9 has ZERO measurable char-8 (wrong log stream captured + the char-8-prone
apply_patch-diff path was never exercised).** What IS solid:

1. **The solve-rate gap is beyond chance** (Fisher two-sided p = 0.0155, OR = 15) — fr9 8/16 vs
   current 1/16 is a *real* regression, not luck. But that regression is driven by the **giant
   bundle of current-only config changes**, and char-8 is only *one candidate* cause.
2. **char-8 is NOT cache-correlated.** Within the current run, cache-OFF has a *higher* char-8
   rate than cache-ON (0.429 vs 0.295 /turn; ratio 1.45× the WRONG way). Every cross-arm test is
   non-significant (tasks-with-char8 p=0.685; rate-ratio p=0.35). This *refutes* "cache causes
   char-8" and is consistent with char-8 being a **Qwen tool-call JSON-truncation flake**.
3. **fr9-vs-current char-8 rate is UNMEASURABLE**, not "lower." fr9 = 0/0 (undefined) on the
   apply_patch-diff denominator and 0/1426 on the command-execution proxy — but the correct vLLM
   server log was never saved and the long-diff-in-JSON apply_patch path was **never used** in fr9
   (0 `*** Begin Patch` blocks; edits went via `sed`/heredoc/python).

**Most-likely axis for the extra char-8 in the current pipeline:** NOT cache. The leading
candidates are **(a) the spec-tree / tree-attention decode path (cat8/chain5 tree_mtp + forked-FA2
TREE_ATTN)** and **(b) the offload-tunnel + thinking-cap proxy path** (LUMO_PROXY_THINK_BUDGET=500,
reasoning-parser qwen3, tailscale hop) that reshapes the tool-call token stream. Both are new and
both plausibly increase mid-tool-call truncation. **Confidence: LOW that we can separate them from
the data in hand.**

**DECIDER:** the queued **e5_OFF cache-OFF spine-5 arm** (same spine-5 topology as fr9, APC OFF) is
the arm that isolates cache from the rest. Only that arm (plus captured vLLM/docker logs on BOTH
cache states) can turn "direction" into "attribution."

---

## (1) Fisher-exact on the SOLVE-RATE — is 8/16 vs 1/16 luck?

| run | resolved | not | rate |
|-----|----------|-----|------|
| fr9 (spine-5, APC OFF, on-GB10) | 8 | 8 | 0.500 |
| current m_e5_ON (chain5, APC ON, offload) | 1 | 15 | 0.063 |

- **Fisher two-sided p = 0.0155**, one-sided (fr9 better) **p = 0.0078**, **odds ratio = 15.0**.
- Sensitivity, pooling both current arms (4/31): **two-sided p = 0.0115** — still significant.

**=> The solve-rate collapse is beyond chance (p<0.02). fr9 is genuinely better; this is a real
regression, not a coin-flip.** (What causes it is a separate question — see attribution.)

## (2) char-8-per-apply_patch RATE — fr9 vs current

| run | char-8 events | apply_patch-diff calls | char-8 / apply_patch | char-8 / turn |
|-----|---------------|------------------------|----------------------|----------------|
| **fr9** | **0** (verified: genuine `char 8)` / `Unterminated string` = 0; the 16 "JSONDecodeError" hits are astropy `ecsv.py` *source code*, not vLLM) | **0** (`*** Begin Patch` in tool args = 0; edits via sed/heredoc/python) | **UNDEFINED (0/0)** | 0 / 1426 cmd-exec = **0.000** |
| **current (ON+OFF)** | 28 (ON 13 + OFF 15; docker cross-check ON = 26 = 2×13) | 4 completed (ON 1 + OFF 3; truncated attempts uncountable cross-arm) | **not per-event determinable** | 28 / 79 = **0.354** |

- fr9 captured only per-task `codex_trace/stderr` + spec traces; **no vLLM server log / docker /
  proxy dumps** — the exact stream where char-8 (a vLLM `_postprocess_messages` `json.loads`
  failure, `chat_utils.py:1595`) surfaces. So fr9's 0 is **UNMEASURABLE-AS-ZERO**, not a proven
  absence.
- **The char-8-prone path was structurally absent in fr9:** no apply_patch tool carried a diff, so
  the long-JSON-argument truncation trigger was never pulled.

**=> You cannot say current's per-apply_patch char-8 rate is "higher than fr9" — fr9's denominator
is 0/0. You can only say current DOES exhibit char-8 (0.35/turn) and fr9 shows NONE detectable on a
path that was never exercised.**

## (3) Within current: cache-ON vs cache-OFF char-8 rate

| arm | cache | char-8 events | turns | char-8/turn | char-8/task | tasks_w/char8 | terminal char-8 |
|-----|-------|---------------|-------|-------------|-------------|---------------|------------------|
| m_e5_ON | **ON** | 13 | 44 | **0.295** | 0.81 | 11/16 | 4 |
| m_cat8_OFF | **OFF** | 15 | 35 | **0.429** | 1.00 | 12/15 | 9 |

- **Direction: cache-OFF is HIGHER** — ratio OFF/ON = **1.45×**, the OPPOSITE of "cache causes char-8."
- tasks-with-char8 (11/16 vs 12/15): **Fisher two-sided p = 0.685 (NS)**.
- rate-ratio test (given 28 events, expect 15.6 in ON, observed 13): **two-sided p = 0.35 (NS)**.

**=> No statistical support for cache-correlation; if anything the sign is against it. Consistent
with char-8 = Qwen tool-call JSON flake, cache-independent.** (Caveat: ON and OFF also differ in
spec topology — chain5 vs cat8 tree — so this is "ON-spine vs OFF-tree," not a clean cache A/B.)

## (4) Reconcile with the known finding

The prior finding (char-8 fires on cache-OFF too; cache-independent PRESENCE) is **fully upheld**:
cache-OFF here shows 15 events / 12 tasks — char-8 is clearly present without the cache. The
"more FREQUENT/FATAL with the current pipeline" half is **partially supported but confounded**:
current shows terminal char-8 on 13/31 tasks and 0.35/turn, whereas fr9 shows none — BUT fr9's
zero is on an unexercised path with the wrong logs, so the frequency delta is real *relative to a
missing baseline*, not a clean +Δ attributable to cache. **char-8 presence is cache-independent;
char-8 fatality is currently high, but the culprit inside "current" is not the cache.**

## (5) Attribution — which axis drives the extra char-8?

| axis | fr9 | current | char-8 plausibility |
|------|-----|---------|---------------------|
| **prefix-cache / APC** | OFF | ON (EXACT_SEED) | **LOW** — OFF arm has *more* char-8; direction refutes it |
| **thinking-cap** | none | THINK_BUDGET=500, qwen3 parser | **MED-HIGH** — reasoning-parser reshapes tool-call token stream; new truncation surface |
| **spec-tree / TREE_ATTN** | flat spine-5, default attn | tree_mtp cat8/chain5 + forked-FA2 TREE_ATTN | **MED-HIGH** — new decode path; accept/verify mismatch can truncate tool-call emission |
| **offload / tunnel** | on-GB10, local proxy | OFFLOAD_CODEX over tailscale to :9950 | **MED** — extra hop/proxy can clip streamed args |
| **vLLM version** | Jun-2 pre-FR10/13 | 0.19.2rc1.dev134 + FR13 patches | **MED** — `_postprocess_messages` path is exactly where char-8 raises |
| harness concurrency | 4 | 1 | LOW for char-8 (serial should reduce, not add) |
| gpu_mem_util | 0.88 | 0.6 | LOW |
| temperature | 0.6 | 0.6 (forced) | none (same) |

**Ranking:** cache = LEAST likely (data actively against). The extra char-8 is most plausibly the
**new tool-call token-stream path** — i.e. **thinking-cap/reasoning-parser + spec-tree decode +
offload proxy + newer vLLM**, all introduced together. **These four cannot be separated with the
current data.**

---

## Data-availability honesty & the DECIDER

- **fr9:** no vLLM server log, no docker_full.log, no proxy dumps — only codex traces/stderr +
  spec traces. char-8 is UNMEASURABLE (0 detectable, not proven-absent), and its apply_patch-diff
  path was never used. **fr9 supplies NO char-8 data point.**
- **current:** docker_full.log + proxy dumps exist **ONLY for cache-ON (m_e5_ON)**; cache-OFF
  (m_cat8_OFF) has **empty proxy dumps and no docker_full.log** → per-event apply_patch attribution
  is **impossible cross-arm**. The 2nd cache-ON arm (m_cat8_ON) **does not exist yet**.
- The current ON/OFF contrast is confounded (ON=chain5 spine, OFF=cat8 tree) — it is not a clean
  cache A/B.

**DECIDER (blocking):** run the queued **e5_OFF cache-OFF spine-5 arm** — same spine-5 topology as
ON, APC OFF — **and capture docker_full.log + proxy dumps on BOTH cache states**. That is the only
configuration that isolates the cache axis from spec-tree/offload/version and makes
char-8-per-apply_patch attributable per-event. Until then: **regression is real (p=0.016), char-8
is real and cache-INDEPENDENT, and the extra char-8's driver is un-attributable within
{thinking-cap, spec-tree, offload, vLLM-version}.**

---

## Config diff (fr9 vs current) — axes that changed

| axis | fr9 | current | same? |
|------|-----|---------|-------|
| prefix-cache / APC | OFF (no EXACT_SEED; FR13-era feature absent) | ON, Mamba 'align', FR13_APC_EXACT_SEED=1 (SNAP_FIX=1, HIT_SUFFIX_CAP=64, ZEROACCEPT=1, CONV_FIX=1, HRS=0) | no |
| thinking cap | none | LUMO_PROXY_THINK_BUDGET=500, qwen3 reasoning-parser | no |
| block_size | not captured (pre-sweep) | 1024 (m_e5_ON) | no |
| mamba cache mode | n/a (APC off) | mamba-block-size 1024, ssm float32, mode forced 'align' | no |
| spec / tree topology | native spine-5 flat chain, n_spec=5, prop=verify=5 | tree_mtp: chain5 n=5 (e5) / cat8 branch n=8; TREE_ATTN | no |
| harness placement | on-GB10, local proxy, concurrency=4 | OFFLOAD_CODEX=1, proxy+codex on alienware, tailscale, serial | no |
| vLLM build | Jun-2 pre-FR10/13 | 0.19.2rc1.dev134+gfe9c3d6c5 + FR13 forked-FA2 + APC bridge | no |
| concurrency | 4 | 1 | no |
| gpu_mem_util | 0.88 | 0.6 | no |
| temperature | 0.6 | 0.6 (proxy-forced) | **yes** |
| attention backend | default (no fork/TREE_ATTN) | TREE_ATTN + forked FA2 (FR13_FA2_TREE_BIAS/PREFILL_NATIVE/EXP2) | no |
| model | qwen3.6-27b-fp8 | qwen3.6-27b-fp8 (same) | **yes** |

*Only temperature and model are unchanged; ~11 axes co-vary — the core reason cache cannot be
isolated without the e5_OFF spine-5 decider arm.*
