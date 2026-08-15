# DESIGN: exact token reconciliation for fixed32 campaigns

**Status:** item 3 (audit ordering) is LANDED in this commit. Items 1, 2, 4 and
the loud-failure guard are SPECIFIED HERE and NOT BUILT. This document is the
spec for the agent that builds them.

**Why it matters:** this is the phase-4 prerequisite. Sealing requires
audit-clean arms, and today no arm can be audit-clean at n=16.

---

## 1. The problem, stated exactly

`validate_fixed32_qwen_campaign_metrics` (`scripts/fr13_fixed32_contract.py`,
~2903-2907) closes this identity with no slack:

```python
if (deltas["prompt_tokens"] != result_prompt_total
        or deltas["generation_tokens"] != result_generation_total):
    raise ContractError("fixed32 qwen campaign aggregate and vLLM token usage do not reconcile")
```

* `deltas[...]` — vLLM's own Prometheus counters. Authoritative.
* `result_*_total` — the sum of **qwen-code's self-reported** `result.usage`.

**Root cause: a third-party agent's self-report is being used as an exact
ledger.** It is not one. It under-credits its own hidden requests.

### The under-crediting, proven at request level

Independent diagnosis (read-only, request-level) established that **no
unattributed traffic exists**: 639/639 engine requests belong to a
ledger-attributed chat completion, zero retries, zero preemptions, max-tokens
algebra closing to the token. The gap is entirely qwen-code's accounting. Three
mechanisms:

1. **Rejected-compaction calls.** Single-request proof — control arm
   `astropy__astropy-14369` has exactly one uncredited **1,550-token** request: a
   361 s compaction at ~69.4k prompt whose summary was billed by the engine and
   then discarded by the agent. Its other three hidden requests sum
   `144 + 443 + 125 = 712`, which is that task's exact hidden credit.
2. **Retried-and-discarded first turns.** Control arm `astropy__astropy-13398`
   shows five pre-first-turn requests, ~119k prompt, coinciding with six
   tool-parser "not well-formed" warnings.
3. **Delegated sub-agent turns that self-report `0/0`.** Already acknowledged in
   this repo — see `fr13_fixed32_contract.py:1579-1600`
   (`_fixed32_qwen_unobservable_compaction_boundaries`), which documents that
   delegated conversations report `{"input_tokens": 0, "output_tokens": 0}` on
   every assistant record.

Concentrated in **3 of 32 task-instances**. The 4-task gates reconcile by
**luck, not structure**: P(clean) ≈ 0.7 at n=4, ≈ 0.2 at n=16. Phase 4 will hit
this.

### Observed impact

2026-08-15 width-4 screen, control arm: vLLM `25,867,251` prompt /
`253,618` generation vs agent-reported `25,677,471` / `247,964` — a gap of
**189,780 prompt (0.73%) and 5,654 generation (2.2%)**. Both arms died at
teardown; ~10 GPU-hours produced no verdict.

---

## 2. What must be built

### Item 1 — put usage on the ingress ledger

`src/lumo_flywheel_serving/inference_proxy.py`. The proxy terminates every
completion, already stamps `task_key_id`, and **already extracts** per-request
`prompt_tokens`/`completion_tokens` (see `_build_request_metrics_row`, ~3609).
Record them on the `request_complete` / `logical_complete` ledger rows.

* Add `prompt_tokens` / `completion_tokens` to `_FIXED32_LEDGER_KEYS` and to
  `Fixed32DigestLedger.append(...)`, defaulting to `None`.
* They live in the row dict, so they **join `record_sha256` automatically** —
  token counts become tamper-evident. That is the point; do not special-case
  them out of the digest.
* Legal only on `request_complete` (engine) and `logical_complete` (proxy);
  `None` everywhere else. Enforce in the per-event validator alongside the
  existing `exact_digests(...)` calls.
* Values: non-negative `int` or `None`. Reject `bool` explicitly (it is an `int`
  in Python and this codebase already guards that pattern elsewhere).
* **Source the values from the proxy's existing `capture_state` usage, NOT from
  `vllm_request_metrics.jsonl`** — see Item 4 for why that file cannot be
  trusted.

### Item 2 — reconcile against the ledger sum

`scripts/fr13_fixed32_contract.py`.

* Reconcile vLLM campaign counters against the **ledger sum** — exact,
  per-request, task-attributed.
* **Demote** `qwen_trace` usage to structural/turn checks. It stays useful for
  turn counts, compaction structure and the max-tokens algebra (all of which
  already close). It stops being the token ledger.
* **Compatibility path, mandatory.** Historical ledgers have no token fields.
  When the rows lack them, fall back to today's qwen-trace comparison so
  existing artifacts still validate. A clean B1/4-task run must stay
  **bit-identical** in outcome. Decide the branch on *field presence in the
  ledger*, never on task count.

### Item 4 — the per-request meter is dead; fix or delete it

`vllm_request_metrics.jsonl` is **0 bytes in every arm of every run** — both
screen arms *and* the successful 4-task gate. The pending metadata records it:
`vllm_request_metrics_bytes = 0`, warning `"proxy capture file not present;
verbose request metrics unavailable"`.

Cause, `inference_proxy.py` ~5562:

```python
if not row.get("request_id"):
    return
capture.record(row)
except Exception:
    pass   # "Capture must never break inference traffic."
```

`request_id` comes from `capture_state["response_id"]`, set on only some
response paths. When absent, every row is dropped — silently, inside a bare
`except: pass`. The fail-safe protecting inference traffic also hides the total
absence of the meter.

This is why Item 1 sources from `capture_state` directly. Either make this file
work (set `response_id` on all paths) or delete it and its `runner_metadata`
warning field. **Do not build reconciliation on top of it.**

### Loud failure — a meter that records nothing must fail the audit

The deepest lesson here: this meter recorded nothing for months and no gate
noticed, because "absent" and "empty" were indistinguishable from "fine".

Whatever meter Item 1/2 lands on must **fail closed on emptiness**. If a
campaign completed N engine requests and the ledger carries token fields for
zero of them, that is a failed audit, not a silent fallback. The compatibility
path in Item 2 must be reachable only for ledgers written *before* the fields
existed — distinguish "old schema" from "new schema, no data", and treat the
second as failure.

---

## 3. Evidence to build against

Both live under the arm's `logs/`, and **persistence through a failed teardown is
proven** — the diagnosis was performed on the two failed screen arms themselves:

| file | rows (control arm) | carries |
|---|---|---|
| `logs/fr13_fixed32_engine_ingress.jsonl` | 1,284 | `engine_request_id_sha256` → `task_key_id` |
| `logs/per_req_spec_trace.jsonl` | 49,279 | `rid` (real vLLM request id) + per-step accepted-token counts |

Join: `sha256(rid)` → `engine_request_id_sha256`.

Note the diagnosis derived **generation** tokens from `per_req_spec_trace`
(Σ accepted + n_steps, calibrated to 225 / 253,618) precisely because the ledger
rows do **not** carry usage today. Item 1 is what removes that indirection.

Reference runroot:
`output/fr13_gdn_single_launch_width4_screen_20260815T014426Z/pass_00/`

---

## 4. Tests required

* Regression from the **real 14369 numbers**: hidden requests `144`, `443`,
  `125`, `1550`; task hidden credit `712`. Assert the ledger-sum path reconciles
  where the qwen-trace path does not.
* A **delegated sub-agent `0/0`** case — the mechanism `fr13_fixed32_contract.py`
  already documents at 1579-1600.
* **B1 / 4-task behaviour bit-unchanged** where it already reconciles.
* Old-schema ledger (no token fields) → compatibility path taken.
* New-schema ledger with zero populated rows → **audit fails loudly**.

---

## 5. What landed in this commit (item 3)

`scripts/fr13_floor_gate.py`: `fixed32_runner_metadata(task_dir, *,
allow_unpromoted)`.

Runner metadata is written as `runner_metadata.pending.json` and promoted by the
end-of-campaign finalizer. The traffic audit required the **promoted** file — but
the finalizer promotes nothing until it has reconciled token accounting, and
raises on the first task when that fails. So a campaign with wrong accounting
leaves every task unpromoted, and **the audit that exists to diagnose exactly
that failure could not run at all.** It demanded the artifact whose absence is
the symptom.

Both screen arms: 16/16 pending, 0 promoted, audit dead on the alphabetically
first task. The "12907 is corrupt" reading was an artifact of alphabetical
order — nothing was wrong with 12907.

The fix: the audit — and only the audit — may read the pending form.
`allow_unpromoted_metadata` defaults to `False` at every other call site, so
promoted-metadata gates are bit-unchanged. The audit payload now carries
`metadata_promotion.unpromoted_metadata_task_ids` and
`all_task_metadata_promoted`, so an audit run on unpromoted metadata can never
be mistaken for one run on published metadata.

9 tests. They **fail at the parent commit and pass here**, which is the property
that makes them regression tests rather than decoration.
