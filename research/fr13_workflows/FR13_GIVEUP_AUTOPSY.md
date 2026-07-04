# FR13 Give-Up Autopsy — astropy__astropy-13453

**Task:** `astropy__astropy-13453` (astropy io/ascii HTML writer ignores `formats=`).
**Agent:** qwen-code v0.19.4, model qwen3.6-27b, permission_mode yolo, temp **forced 0.6** by proxy, seed=None, max_tokens=65536, auto-continue configured ON. Offloaded to alienware.
**Harness/task/seed policy identical across all 4 arms.** Branch `fr13-mainpick` @ 08b629ef.

| Arm | Config | Wall | Patch | Verdict | Degrade turn |
|-----|--------|------|-------|---------|--------------|
| m_tree_nocache | cat8, cache OFF | 1719s | **551B** | **RESOLVED** | — (reference) |
| m_tree_cache_base | cat8 + EXACT_SEED cache | 843s | 0 | give-up | turn 6 / 6 |
| m_tree_recompute_np | base + SCAN_ALIGN=recompute + N_PAD=1 | 751s | 0 | give-up | turn 2 / 2 |
| m_tree_patha | base + BLOCK_REFOLD=1 | 613s | 0 | give-up | turn 3 / 3 |

All three give-ups: `empty_patch_retry.cause=agent_gave_up`, eval `patch_apply_failed / empty_patch`, exit 0, no timeout, no network drop, `totalErrors=0`, all HTTP 200.

> Evidence caveats baked into this doc: **proxy_pair_dumps/ is EMPTY in every arm** (no response bodies) — completion text is reconstructed from `codex_trace.jsonl`. **ES logs carry NO timestamps** — any per-marker wall-clock in the source reports is inferred; only `proxy_request_dumps/` epoch-ms filenames are authoritative for turn timing. **offload_request_metrics.jsonl is STALE** (dated 02:53/03:17, a prior run) — not used. **Engine logs never contain generated tokens** — drift text lives only in the trace.

---

## 1. Give-up mechanism — per arm

**Shared control-flow signature (all 3 give-ups, verified):** healthy exploration for 1–5 turns with valid XML tool calls, then the **last substantive turn emits NO tool_use, `stop_reason=null`**, begins on-task then drifts off-task, produces no patch. The configured auto-continue nudge (`"...VERY NEXT action MUST be an apply_patch"`, MAX_RETRIES=3) **fired zero times** — almost certainly because the terminal completion ends at natural EOS mid-generation (`stop_reason=null`), so the harness books `agent_gave_up` at teardown instead of injecting the mid-session nudge. That is a real rescue-path gap, shared, not a differentiator.

**Content signature DIFFERS per arm** (this is the key cross-exam correction — it is *not* one uniform "garble"):

- **m_tree_cache_base — turn 6 (@08:25:29.5Z, ~34.9k-tok prompt, its 5th seeded restore, 1830 out-tok).** Drift is in the **THINK channel**: one on-task sentence about `core.py`/`_set_col_formats`, then a **coherent code-switch** into fluent Node/PostgreSQL JavaScript (`getRecords`/`PostgreSQLDriver`/SQL builders) then fluent **CJK Chinese** prose (`def get_current_time`, `MyClass`). No tool call, natural EOS. **Not token garble — a fluent off-task topic-hijack.**
- **m_tree_recompute_np — turn 2 (@08:41:01.2Z, its only substantive turn, 505 out-tok).** Drift in the **THINK channel**: on-task ("...understand how the HTML,") hijacks at the token **"HTML"** into a **coherent English web-dev essay** (HTML/CSS/JS, `document.write`, PHP `fwrite`, XSS, "In summary…"). No tool call, clean EOS well under max.
- **m_tree_patha — turn 3 (@08:51:38.1Z, 662 out-tok).** Drift in the **ANSWER channel** (post-`</think>`): thinking preamble is on-task ("Let me look at the HTML writer implementation…"), then ~660 answer tokens that the trace **did not capture at all** — only `text='\n\n'` survives, raw stream unrecoverable (stdout 0B, metrics 0B, pair dumps empty). Subtype (garble vs broken-XML vs looping) **cannot be byte-confirmed**.

**Correlation with cache-hit onset — there is NONE, because no cache HIT ever occurred.** All three ES logs show `row0_hit=False ×50` (`bs=832`, `row0_plen=681` = the warmup fingerprint that never matches the live SWE prefix); `ES_REDIRECT_FALLBACK` fired on every gate (384/96/192). The exact-seed **REDIRECT never engaged in any arm.** What DID fire is the seed **INJECT/RESTORE** path (`ES_RESTORE seeded=True`), and its onset does **not** line up with the degrade turn: cache_base restores on turns 2–6 but degrades on turn 6 (**+4**); patha restores turn 2 (healthy) & 3 (**+1**); recompute_np restores only on turn 2 (**+0**, its lone turn). No consistent offset ⇒ degrade is not a discrete first-restore event.

**Reference arm (m_tree_nocache) passes the same exploration cleanly** — 26 calls, read the same `html.py`/`core.py`/`_set_col_formats`, reached the correct 2-line edit, stopped on a well-formed 708-char summary. **But via a different route:** its turn 1 spawned the **Explore subagent** (tool:agent, ~300 tok), giving the main agent short-context turns; all three cache arms did `read_file AGENTS.md` (69 tok) at turn 1 and explored in one **monolithic long context**. That route split happens at **cold prefill on byte-identical input** (see §5).

---

## 2. What today's null results EXCLUDE

Both fixes tried today target the **SSM seed**, which independent measurement already proved is **bit-exact under cache** (`ES ssm min_dist=0.0` on all 48 GDN layers, native+cache and cat8+cache alike, commit 0d12cdbf). So both were expected inert, and both **engaged and still gave up**:

- **recompute+NP** (SCAN_ALIGN=recompute, N_PAD=1 — targets the FLA chunk-restart fp-accum order): 751s / 0B, garbled turn 2.
- **Path A** (BLOCK_REFOLD=1): `REFOLD_APPLIED=432`, but `REFOLD_RESTORE_OTHER=30` (fell back to the co-resident leaf = wrong source, all at pos=23296) and `REFOLD_SKIP=816`: 613s / 0B, garbled turn 3.

**Excluded as the carrier:** (a) SSM recurrent-state restore / seed corruption (bit-exact; both SSM-side fixes inert). (b) The FLA chunk-restart `core_out` perturbation (~1e-2, layer-0 `max_abs=3.7637`) — **identical in the RESOLVING native+cache arm**, so it cannot discriminate give-up. (c) Discrete cache-HIT KV corruption — **no cache HIT ever fired**. (d) **char-8 / cat8 malformed-tool-XML** — REFUTED by the engine log: `qwen3xml_tool_parser "not well-formed"` counts are **nocache(RESOLVED)=9** (all recovered into 200 OK), **cache_base=0, recompute_np=0, patha=0**. The garble class appears *only* in the arm that resolved. Report 4's "char-8 class" headline for patha is an unverifiable inference from a token-count mismatch and is over-claimed.

**Not excluded** (both fixes leave them untouched): full-attn RoPE/position path under restore; tree topology/verify sensitivity; trajectory-route selection; temp-0.6 sampling variance.

---

## 3. Surviving carrier hypotheses — RANKED

**H1 — Config-deterministic trajectory divergence → long-context off-task drift.** *(strongest surviving; combines cross-exam suspects 1+2)*
Mechanism: cache/carrier-ON deterministically flips the **turn-1 route** from the resolved arm's Explore-subagent delegation to a monolithic direct-exploration in the main context; the long single-context decode then drifts onto a generic **web-dev n-gram attractor** keyed by the "HTML write" task tokens, and never returns to emit a tool call.
Evidence: turn-1 request bytes **identical** across all 4 arms yet route splits at **cold prefill, before any restore**; 2/3 terminals are verbatim-confirmed coherent hijacks (English web-dev essay; Node/PostgreSQL JS→Chinese) — fluent, not garble; 0 engine XML warnings; drift lands on the "HTML" boundary in recompute_np. Determinism-per-config makes it reproducible, not a fluke.
Weakness: does not by itself prove **cache is the root** vs "cache merely selects a different-but-valid trajectory that happens to drift."

**H2 — Low-power sampling / agent artifact, weakly coupled to cache losslessness.** *(medium)*
Mechanism: temp-0.6, n=1-per-config; give-up is agent-behavior variance amplified by whichever token stream the config selects, not a serving defect.
Evidence: no cache HIT anywhere to blame; first-token TV is known decoupled from give-up (native+cache TV **0.881** RESOLVES vs cat8 0.496); prior cap-matrix 6/6 resolved and a 2×2 "cache-fails" collapsed to temp-0.6 noise; give-up premise flagged SHAKY.
Weakness: determinism-per-config (cache_base≡patha turn-1 thinking byte-identical) argues the outcome is *reproducible per config*, not pure coin-flip — so "noise" must mean "config-selected deterministic bad trajectory," which is really H1.

**H3 — Full-attn RoPE/position/KV mis-wiring on the seeded-restore boundary.** *(weakest survivor; Report 5 champion)*
Mechanism: `_fr10_mrope_base` (patcher:11828) is an uncorrected `num_computed_tokens_cpu`; the depth-position remap (11831–11844) rewrites only spec/tree rows ⇒ on a restore-boundary turn the tree's full-attn positions can offset, corrupting full-attn KV for a whole turn and steering the accepted path toward a degenerate branch.
Evidence: diagnosed 2026-06-28, **never fixed**; the divergence ladder measured only the 48 GDN layers and **skipped all 16 full_attn layers [3,7,…,63]**, so this path is **unmeasured under cache**; both SSM-side fixes leave it untouched; tree `bs=832` vs native `bs=1024` gives a smaller, N_PAD-misaligned restart boundary.
Weakness: its named trigger is the **cache-hit boundary, which never occurred** (`row0_hit=False ×50`). Only the seed-inject/restore boundary fired. Argued from an evidence gap, not positive evidence.

**REFUTED:** char-8/cat8 malformed-XML (§2); SSM seed corruption (§2); discrete cache-HIT KV corruption (§2).

---

## 4. DECISIVE NEXT TESTS — cheapest first

**T0 — Token-level trace diff (NO GPU, existing artifacts).** Diff generated tokens of `m_tree_nocache` (resolved) vs each give-up trace: (a) confirm the turn-1 route split (subagent vs `read_file AGENTS.md`); (b) find the **first off-task token** and whether it lands on the "HTML"/web-dev boundary; (c) correlate with a drop in accept-length at that step. *Discriminates:* whether divergence originates at the turn-1 route (H1/H2) or mid-decode (H1 drift); free, do first.

**T1 — B=1 same-seed repeat of m_tree_cache_base on 13453 (playbook first gate; 1 GPU run, SAME boot / in-process).** Per MEMORY, use the in-process same-boot gate, not cross-boot byte-identity. *Discriminates:* if it flips to RESOLVE ⇒ give-up is sampling/config-selection variance (H2) and the "tree-cache defect" framing collapses; if it reproduces the garble ⇒ deterministic per config (supports H1). **Mandatory before any deeper attribution.**

**T2 — Paired native (non-tree MTP-5) + cache-ON control on 13453 (1 GPU run).** The "decisive control" named in the Track A commit and never run. *Discriminates:* native+cache RESOLVES ⇒ differentiator is the **tree route/topology**, not cache-per-se (supports H1); native+cache also GIVES UP ⇒ **cache/carrier** is the carrier independent of tree.

**T3 — Route-forcing (1 GPU run).** Run cache-ON with the Explore-subagent route **forced** (or cache-OFF with subagent **disabled**). *Discriminates:* give-up tracks ROUTE ⇒ H1/H2 trajectory-shape confirmed; give-up persists on the subagent route under cache ⇒ carrier corrupts the decode regardless of route (supports H3).

**T4 — `FR13_APC_EXACT_SEED=0`, else identical (1 GPU run).** Disable the seed-inject/restore path only. *Discriminates:* turn-2/turn-6 coherence returns ⇒ the inject/restore path is implicated; unchanged ⇒ it is not (points back to tree-config, H1).

**T5 — Full-attn capture (1 GPU run, existing instrument).** Point `_fr12_full_attn_capture_tensor` (patcher:17533) at the first post-restore decode turn; compare tree+cache-ON vs cache-OFF `q_after_rope` at layer 3, dump `_fr10_mrope_base` vs the true absolute prefix position. *Discriminates:* H3 directly. Only worth spending GPU on if T2/T3 point at cache-per-se rather than tree-route.

**Infra fix to enable byte-classification:** re-enable proxy_pair response capture before the next give-up run so patha's ~660 lost answer tokens are dumpable and the drift subtype can be byte-classified against cat discriminators.

---

## 5. Confound ledger

- **n=1 per DISTINCT config; no config repeated.** The playbook first gate (B=1 same-seed repeat) was **never run today**. Give-up premise flagged SHAKY (6/6 cap-matrix arms resolved; a 2×2 "cache-fails" earlier collapsed to temp-0.6 noise; char-8 was n=1/single-task).
- **No cache HIT ever fired** (`row0_hit=False ×50`, redirect fallback every gate) ⇒ "cache corruption" is unproven; only the seed-inject/restore path fired, and its onset does not align with the degrade turn.
- **The decisive control was never run:** native (non-tree) + cache-ON on 13453.
- **Single task** (astropy-13453 only) — no generalization across tasks.
- **Raw streams partially unrecoverable:** proxy_pair_dumps empty in all arms; patha's 662 terminal tokens fully lost (stdout/metrics 0B). cache_base/recompute_np drift recovered only from the trace's thinking channel.
- **ES logs have no timestamps** (reported ES wall-clocks are inferred); `offload_request_metrics.jsonl` is STALE (prior run) and must not be used.
- **Agent-side confounds RULED OUT:** turn-1 request bodies **byte-identical** across all 4 arms (system sha8=3792e2b8 / user sha8=1ec17829, identical 60-tool list, identical agents list, temp=None@proxy→0.6, max_tokens=65536, seed=None); same qwen-code 0.19.4, model, permission_mode. No version/prompt/tool/temperature differentiator.
- **Determinism-per-config:** turn-1 thinking byte-identical between cache_base and patha (recompute_np differs, still same action) ⇒ sampler effectively deterministic per serving-config despite seed=None ⇒ divergences are engine-config-driven and reproducible per config, but determinism **cannot separate** "carrier corrupts the stream" from "carrier deterministically selects a different valid trajectory that drifts."
- **Auto-continue rescue gap** (shared, not a differentiator): the nudge designed to rescue a tool-less give-up fired 0× in all three arms (terminal `stop_reason=null`), converting a recoverable tool-less stop into a hard empty-patch give-up.
