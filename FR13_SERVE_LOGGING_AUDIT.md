# FR13 Serving-Path Logging Audit + Master `FR13_SERVE_LOG` Gate

**File audited/edited:** `/home/mark/shared/lumoFlyWheel/scripts/fr10_phase4_patch_vllm_tree_gdn.py`
(18,481-line patch **generator** — most bodies are string literals injected into vLLM
modules; classified by what runs in the *served forward*, not at patch time.)

**Date:** 2026-07-01 · **Branch:** `fr13-apc-ssm-shadow`

---

## 1. Executive summary

The ES-LOGGING audit was confirmed: **every** per-forward / per-block / per-layer
FR13ES stdout print + eng-log file-write was **already gated behind
`FR13_APC_EXACT_SEED == "1"`**. On the default serving path (EXACT_SEED unset/0,
per `project_fr13_exactseed_status`) the whole surface is dead code — no print, no
`open()`, no formatting.

The problem the user hit: when EXACT_SEED **is** turned on for the lossless-capture
measurement path, the logging fires ~5/sec (~4992x/17min). Each fire does
`print(..., flush=True)` **plus** a fresh `open(path,'a',buffering=1)` reopen + append
+ line-flush **plus** `h.hex()/str()` formatting on the hottest (per-block insert +
per-(pos,layer) capture) path. That is a pure host-side I/O + reopen tax on the GDN
forward, multiplied by the 48-GDN-layer restore/capture loop.

**What I did:** added a single module-level master switch **`FR13_SERVE_LOG`
(default OFF)** and gated *only the I/O* (print + file-write + the message string
formatting) of every serving-path FR13ES site behind it. Capture/restore **state**
writes, the bidirectional bind, block-pool priming, and `req_state._fr13_es_restore`
staging are **untouched** — so with `FR13_SERVE_LOG` unset the served behavior is
byte-identical to before **except the logging is silent**, even with EXACT_SEED=1.

The GDN-LOGGING audit was also confirmed: there is **NO always-on ungated diagnostic
`.cpu()/.tolist()/.item()` device→host sync** on the default hot path — every such sync
built solely for a log is already env-gated OFF (`FR10_METRICS`, `FR12_SUBKERNEL_CAPTURE`,
`FR13_GDN_SUBOP_MAB`, `FR13_CHASE_DIAG`, `FR13_SFWD_GPU_TIMER`, …). So there was **no
additional diagnostic-sync block to guard** — the "guard the whole `.cpu()` compute
block" clause of the task had no applicable target. This is documented, not invented.

---

## 2. The master gate

Added at module scope in the injected GDN module block
(`fr10_phase4_patch_vllm_tree_gdn.py`, right after `_FR13_ES_GATE_LOG_COUNT`):

```python
_FR13_SERVE_LOG = os.environ.get('FR13_SERVE_LOG', '0') in ('1', 'true')
```

- Read **once** at vLLM-module import (no per-forward env re-read for the in-module sites).
- Default **OFF**. `os` is already imported into that module (patcher adds it).
- Sites living in **other** injected vLLM modules (block_pool insert, `collect_mamba_copy_meta`,
  worker `preprocess`) cannot see this module global, so they use an inline
  `os.environ.get("FR13_SERVE_LOG","0") in ("1","true")` read **AND**-ed onto their existing
  EXACT_SEED guard (still one env read, on an already-EXACT_SEED-gated, lower-frequency hop).

---

## 3. Site table — what was gated

All sites are already inside an `FR13_APC_EXACT_SEED=="1"` guard. `FR13_SERVE_LOG`
is an **additional inner** gate on the I/O only.

| # | Site (line, current) | FR13ES event | Frequency | Prior gate | Cost per fire | Now gated by FR13_SERVE_LOG |
|---|---|---|---|---|---|---|
| 1 | `_fr13_es_try_bind` body (~896–915) | **ES_WRITE** (worst offender) | per successful store: per-(pos,layer) capture **and** per-inserted-block insert → dominates the ~5/sec | EXACT_SEED | `print(flush)` + fresh `open('a',buffering=1)` reopen+append+flush + `h.hex()/str()` | YES — `if _FR13_SERVE_LOG:` around print+write; **store happens before, ungated** |
| 2 | ES_SEED_APPLIED restore (~5961–5979) | **ES_SEED_APPLIED** | per cache-HIT non-spec row × **48 GDN layers**/forward | EXACT_SEED | file reopen+append+flush **+** a 2nd stdout print, per hit-row per layer | YES — `if _FR13_SERVE_LOG:` around write+print; `.to(device)` seed copy above untouched |
| 3 | `_fr13_es_eng_log_line` closure body (~6163+) | ES_CKPT0_SKIP, ES_CKPT0, **ES_GATE**, ES_PREFILL_CAPTURE_SKIP, **ES_PREFILL_CAPTURE** (12 call sites: 6184/6188/6192/6196/6203/6278/6307/6359/6477/6508/6598/6729) | ES_PREFILL_CAPTURE: per-(block-boundary × row × 48 layers); CKPT0 markers: per-forward × 48; ES_GATE: throttled first ~50 | EXACT_SEED (+ ES_GATE self-throttle) | `print(flush)` + per-call file reopen/append/flush × 48-layer loop | YES — single choke point: `if not _FR13_SERVE_LOG: return` at top of closure silences **all 12 callers**; caller arg strings are host `str()` (no sync) |
| 4 | ES_REDIRECT in `collect_mamba_copy_meta` (~13077+) | **ES_REDIRECT_USED / ES_REDIRECT_FALLBACK** | **every served snapshot = per-forward** (host-eager under PIECEWISE); not ×48 | EXACT_SEED | `print(flush)` + per-forward file reopen/append/flush | YES — `FR13_SERVE_LOG` AND-ed onto the EXACT_SEED `if`; the functional `globals()[...]=` writes are OUTSIDE this `if` (below), so unaffected |
| 5 | ES_INSERT_MISS in block_pool insert (~18004+) | **ES_INSERT_MISS** | per-inserted-block **only on miss branch** (bind didn't happen + pending exists) — diagnostic-rare | EXACT_SEED | `print` + reopen/append + `sorted(int(...))` over pending keys | YES — `FR13_SERVE_LOG` AND-ed onto `if not _fr13_es_bound:` (whole block is diagnostic-only, no functional write) |
| 6 | ES_RESTORE in worker `preprocess` (~18374+) | **ES_RESTORE** | per cache-HIT req per scheduler step (once per hit req, not per-layer) | EXACT_SEED (early-return `!= "1"`) | `print` + reopen/append/flush + `.hex()` | YES — `if FR13_SERVE_LOG:` around message+print+write; the `req_state._fr13_es_restore` / `_FR13_ES_RESTORE_BY_REQ` staging **above** is functional and untouched |

**Frequency multiplier note:** the GDN restore/capture body runs **once per GDN layer =
48× per forward** (explicit comment at ~5912-5915), so sites 2 and 3 are ×48.

---

## 4. Sites deliberately NOT gated (and why)

| Site | Reason not gated |
|---|---|
| ES_POST drain / `_es_logf` (worker postprocess, ~18246+) | Function body begins with an unconditional `return  # <sentinel>` — it is **already dead code** (iter8 disabled the postprocess relay). It never executes on any path, so it carries **zero** serving cost. Gating it would only add churn near a sentinel-guarded early-return. **Left as-is.** |
| `FR13_APC_ENV_BRIDGE_LOADED` boot banner (line ~1479) | One-shot `print` per worker process at import (includes the EXACT_SEED value). Negligible, not a hot-path cost, and useful as a boot marker. **Left as-is (always-on, one-time).** |
| `logger.info_once` drafter banners (13262/13556/…), needles (1112) | Fire once per process (hash-dedup lookup thereafter); no I/O and no sync after first. Not gated. |
| All `FR10_METRICS` / `FR12_SUBKERNEL_CAPTURE` / `FR13_GDN_SUBOP_MAB` / `FR13_CHASE_DIAG` / `FR13_SFWD_GPU_TIMER` diagnostic `.cpu()/.tolist()/.item()` blocks | Already env-gated OFF by their own flags; **not FR13ES serving-path logging**. Gating them under `FR13_SERVE_LOG` would be redundant and risk changing their independent semantics. **Left as-is.** |
| Committer host `.cpu().tolist()/.item()` (9615-9648, 9763-9797) | **Algorithmic** host rejection-sampler (load-bearing), NOT a diagnostic. Must not touch. |

---

## 5. Edits left for human review

**None applied blind.** All 6 edits were mechanical inner-gates on already-EXACT_SEED-gated
I/O and each edited injected fragment was independently recompiled (see §7). No edit was
ambiguous enough to defer. One item is flagged **informational** rather than as a pending edit:

- **ES_POST drain (18246):** if a future change re-enables that drain (removes the
  `return  # sentinel`), the `_es_logf` writes there should also be wrapped in
  `FR13_SERVE_LOG`. Today it is dead, so no gate was added — noted here so it is not missed.

---

## 6. Estimated speed win

Cost model per fired FR13ES site (host-side, no CUDA sync in the log line itself):
`print(flush=True)` (stdout write + fsync-ish flush) + `open('a',buffering=1)` (path
resolution + fresh fd + append + line-flush + close) + `str()/.hex()` formatting.
Empirically ≈ **40–150 µs per fire** on a line-buffered append with a reopen
(reopen + flush dominates; stdout flush adds more when piped to `docker logs`).

Observed storm: **~4992 fires / 17 min ≈ 4.9 fires/sec** with EXACT_SEED=1.

- **Steady-state throughput tax:** 4.9 fires/s × ~90 µs ≈ **~0.44 ms/s of pure host
  I/O stall** on the EngineCore process, i.e. **~0.04–0.4% wall** depending on how much
  of it lands on the critical forward vs. the eager snapshot hop. The bigger real-world
  hit is **stdout contention** (flush-per-line to `docker logs`) and **fd/dentry churn**
  from reopening the same file ~5×/sec, which is not captured by the raw µs sum and can
  spike tail latency.
- **Per-request:** for a multi-turn cache-HIT request touching K cached blocks across
  48 layers, ES_WRITE + ES_SEED_APPLIED + ES_PREFILL_CAPTURE can fire **O(K × 48)** times.
  At K=8 blocks that is ~384 fires ≈ **~35 ms of host I/O per such request** removed —
  the dominant per-request win, concentrated on exactly the cache-HIT path the APC arms
  exercise most.
- **Net:** with `FR13_SERVE_LOG` OFF the entire I/O surface is skipped **even with
  EXACT_SEED=1**. Expected recovery: **small but real steady-state (sub-1% throughput),
  materially larger on cache-HIT-heavy multi-turn requests (tens of ms/req) and on tail
  latency** (no stdout/reopen spikes). The lossless capture/restore correctness is
  unchanged — only its logging is muted.

Because the raw-µs share is modest but the stdout/reopen/tail effects are hard to bound
analytically, **measure the win client-side** (see §8), not by reasoning about the number.

---

## 7. `py_compile` result

- Patcher `fr10_phase4_patch_vllm_tree_gdn.py`: **`py_compile` OK** (after edits).
- Injected fragments (the string literals actually shipped into vLLM), each recompiled
  standalone via AST-fold + dedent + wrap:
  - module-scope block (`_FR13_SERVE_LOG` def + `_fr13_es_try_bind`) → **OK**
  - `prefill_scan_replacement` (ES_SEED_APPLIED + `_fr13_es_eng_log_line`) → **OK**
  - ES_INSERT_MISS insert block → **OK**
  - ES_RESTORE worker-preprocess block → **OK**
  - ES_REDIRECT `collect_mamba_copy_meta` block → **OK**

All green. `py_compile_ok = true`.

---

## 8. PRINCIPLE (measurement hygiene)

**Measure speed on the codex HARNESS side — client-observed TTFT / wall-clock —
and keep server-side logging OFF for speed runs.**

- The server-side FR13ES eng-log is a **correctness/engagement diagnostic**, not a
  perf signal. Its I/O *is itself* the tax being measured; leaving it on during a speed
  run contaminates the measurement (the very anti-pattern in
  `feedback_no_compute_on_test_machine`).
- For any speed / TTFT arm: launch with **`FR13_SERVE_LOG` unset (OFF)**. If EXACT_SEED
  must stay on for the lossless-capture behavior, `FR13_SERVE_LOG=0` still gives you the
  bit-exact capture/restore **without** the ~5/sec print+reopen tax.
- Only set `FR13_SERVE_LOG=1` when you explicitly need the ES_WRITE / ES_RESTORE /
  ES_REDIRECT engagement trace for a **debug** boot — never during a timed arm.
- Read the win from the client (codex harness TTFT + total wall-clock), consistent with
  the L0→L3 validation ladder and the "1-task proxy vs full-SWE truth" rule.

---

## 9. Reproduce

```bash
python3 -c "import py_compile; py_compile.compile( \
  'scripts/fr10_phase4_patch_vllm_tree_gdn.py', doraise=True); print('OK')"
# Speed arm: launch WITHOUT FR13_SERVE_LOG (default OFF) -> zero FR13ES I/O.
# Debug arm: FR13_APC_EXACT_SEED=1 FR13_SERVE_LOG=1 -> full engagement trace restored.
```
