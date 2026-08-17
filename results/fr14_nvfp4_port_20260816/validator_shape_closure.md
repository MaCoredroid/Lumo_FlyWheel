# FR14 — provenance-validator shape closure against the real 3.8 corpus

The fixed32 provenance validator independently counts completed model requests
from the qwen-code trace, so the harness can cross-check the engine's own
counter. It was built against Qwen3.6 traces. Qwen3.8 killed three arms in three
serves on shapes it did not model.

Rather than patch the reported shape each time, the validator was run directly
over **every** `qwen_trace.jsonl` on disk (14 traces, 5 runroots) to get a
reproducible map, and the full shape space of the corpus was enumerated.

## Result

| | before | after |
|---|---:|---:|
| traces that parse | 12 / 14 | **12 / 14** |
| traces that RAISE | 2 (arm kills) | 0 |
| traces REFUSED on purpose | 0 | **2** (dead-engine probes, see C2) |

Every one of the 12 real task traces keeps **exactly** the count it had
(10, 18, 11, 20, 56, 76, 10, 18, 63, 16, 22, 92). Nothing was fixed by weakening
the counter.

## Corpus shape enumeration

Across all 14 traces, so the next surprise is a known unknown:

| dimension | observed |
|---|---|
| event types | `system` 14, `assistant` 1144, `user` 445, `result` 14 — nothing else |
| result subtypes | `success` 14 — nothing else |
| assistant `stop_reason` | `None` 734, `"tool_use"` 410 — nothing else |
| content block types | `thinking` 366, `text` 369, `tool_use` 444, `tool_result` 444 |
| `message.id != event.uuid` | 0 |
| duplicate `tool_use` ids | 0 |
| `parent_tool_use_id` non-null | 113 (one sub-agent, depth 1, all in 13398) |
| tool names | `run_shell_command` 198, `read_file` 115, `grep_search` 68, `edit` 25, `write_file` 9, `glob` 8, `todo_write` 8, `web_fetch` 8, `list_directory` 2, `tool_search` 2, `agent` 1 |

Content blocks never mix: a logical turn serialises as `thinking` → `text` →
`tool_use`, and the validator's contiguity grouping re-assembles it.

## Closed

### 1. `web_fetch` schema rejection — was arm kill #1 and #2

`fr14_b1_stock_20260817T020534Z` / 13236, trace line 159:
`web_fetch {"url": ".../table.py"}` with no `prompt`. qwen-code validates a call
against its JSON schema **before** executing it, so this never reached
`executeDirectFetch`: it fetched nothing and issued no `runSideQuery`. The trace
proves it — `is_error: true`, `"params must have required property 'prompt'"`.

The validator was treating a malformed invocation as unaccountable traffic when
it is the opposite: **zero requests owed**. Now counted as zero.

Fail-closed half kept: a malformed invocation whose result is the *success*
display describes traffic that cannot have happened, and still raises. The same
trace still reports `hidden_web_fetch=4` for its four valid calls.

### 2. `result.permission_denials` non-empty — was arm kill #3

`fr14_b1_stock_20260817T031507Z` / 13236, result record: under the no-net agent
settings qwen-code enforces the `web_fetch` deny rule against equivalent shell
commands, so the model's `curl https://…` came back "denied by permission rules"
and was recorded there.

The validator demanded `permission_denials == []` — so it failed on its own
safety feature. A denial hides no model request: the assistant's `tool_use` is
already counted in its group, the denial arrives as an ordinary paired
`tool_result`, and unlike `web_fetch` the denied tool never runs and never calls
the model.

Fail-closed half kept and made specific: the field must be a list; each entry
must carry a non-empty `tool_name` and `tool_use_id`; **and the `tool_use_id`
must name a call this trace actually contains** — a denial referring to an
unknown call means the trace is not a complete record of its own session. (That
join was claimed in a comment before it was implemented; the audit caught it.)

### 3. Dead-engine overcount — silent, and the worst direction

Both `fr14_b1_probe_*` traces contain one assistant record that is qwen-code
narrating its own failure:

* `[API Error: EngineCore encountered an issue. See stack trace (above) …]`
* `[API Error: Connection error. (cause: fetch failed)]` — never left the client

with all-zero usage, closed by `subtype:"success"`, `is_error:false`,
`num_turns:1`. Nothing distinguishes it from a served turn, so the validator
returned `completed_logical_model_requests = 1` for a request the engine served
**zero** of. **It manufactured evidence of traffic that never happened** — the
one direction a fail-closed counter must never fail in, because an invented
request silently absolves a real gap elsewhere in the ledger.

The trace cannot tell "EngineCore died after serving" from "the fetch never
left the client", so the honest answer is refusal, not a guess. Now refused.

The local compression-failure terminal is *also* an `[API Error: …]` banner, but
it is a fully modelled shape with its own accounting, matched exactly by
`_QWEN_COMPACTION_FAILURE_TEXT_RE`; it is explicitly excluded from the refusal.
The refusal also keys on the banner **prefix**, so an agent merely discussing an
API error in its final answer still counts.

## OPEN — not fixed here, with evidence

### C1. Sub-agent-scoped compaction is structurally invisible: undercount by 1

`fr14_b1_stock_20260816T204931Z` / **13398** (arm A's banked stock arm):

| source | value |
|---|---|
| engine `request_success_total` delta | **77** |
| `max_tokens` histogram | `le_20000: 1`, `le_50000: 77` |
| `max_tokens_sum` | `2510368` = 76 × 32768 + 1 × 20000 (exact) |
| validator, no-metrics path | **76** — wrong |

The 77th is a compaction at `max_tokens=20000` that happened **inside the
sub-agent conversation**. `_fixed32_qwen_hidden_compaction_requests` infers
compactions from input-token drops between consecutive *top-level* groups, and
this trace's top-level `input_tokens` is strictly monotonic
(23794 → 65367, zero drops). Every nested record reports
`{"input_tokens": 0, "output_tokens": 0}`, so the drop is unobservable by
construction.

Two consequences:

* the **no-metrics path** returns 76 and reports success — and that path is live
  in `run_swe_bench_q36_a.py`, `fr13_fixed32_contract.py`,
  `fr13_depth_acceptance.py` and `fr13_floor_gate.py`;
* the **metrics path** gets 77 but by the wrong mechanism, booking a
  *successful* compaction as a *failed* one.

The tolerance that absorbs it, `unobservable_compaction_boundaries`, is **33**
for this trace — a very wide silent allowance that grows with sub-agent length.

**Not fixed here on purpose.** The correct fix changes the compaction algebra
itself (teaching it to attribute drops inside a sub-agent subtree), and doing
that under a live campaign risks breaking the metrics path for every arm. It
wants its own commit, its own reconciliation against 13398's exact numbers, and
its own before/after over the whole corpus. Recorded with the numbers needed to
write it.

### Latent shapes — the next arm kills, ordered by likelihood

Each is a real code path with zero coverage from 3.8 data:

1. **`stop_reason` other than `None`/`"tool_use"`** → instant refusal. The
   validator's own comments say `length` completions are legal and were seen at
   scale in FR13; all 14 traces have `length = 0`. **Highest-value untested
   class.**
2. `agent` with `run_in_background` or `isolation` → unsupported-invocation
   raise. One `agent` call in 14 traces is not coverage.
3. `agent` or `web_fetch` with an unknown input field → hard refusal (the new
   schema-rejection path is checked *after* the unknown-field check).
4. A sub-agent turn that is not a tool call → "agent response group has no tool
   call". All 34 nested groups today are pure `tool_use`; the top-level
   conversation emits `thinking`+`text` every turn, so this looks like a
   serializer quirk that could change in any build.
5. `num_turns` counting delegated turns → every sub-agent trace dies.
6. Batched multi-`tool_result` user messages → the subtree walk refuses.
7. Non-string `tool_result.content` (e.g. a `computer_use__*` screenshot) →
   "tool result content is empty".
8. Three or more `result` records / any nested error boundary — the entire
   `_validate_fixed32_qwen_nested_error_boundary` path (~80 lines, 6 distinct
   errors) is **completely unexercised** by 3.8 data.

### Separately observed

All four `fr14_b1_stock_20260816T*` traces raise
`fixed32 qwen pre metric vllm:prompt_tokens_total labels differ` on the
metrics-armed path: those arms served `model_name="qwen3.8-27b-nvfp4"` while the
contract now pins `qwen3.8-27b-nvfp4-radixark`. Expected after the arm-B swap,
but it means **no stock-arm trace has ever been through the metrics
cross-check**.

## The meter is unchanged

The engine's max-token histogram and the ingress ledger remain the meters. The
trace is only ever allowed to **name** the requests they already counted. Every
change above either teaches the trace to name a request correctly, or makes it
refuse rather than guess.
