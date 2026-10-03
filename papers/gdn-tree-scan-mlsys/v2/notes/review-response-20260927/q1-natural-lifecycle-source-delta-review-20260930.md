# Natural lifecycle source delta review

Disposition: no concrete source blocker found in the separately generated schedule, forcing and runtime. This is preparation-only source review, not a runnable collector or lifecycle qualification. No tests, runtime imports, model/GPU work, gates or implementation changes were performed.

## Reviewed bytes

| File | SHA-256 |
| --- | --- |
| `tools/q1_natural_lifecycle_source_recipe_v1.py` | `91b0bad6ce1144ab8419da2177a1308cdece2c092605eed2caae481aa22d112e` |
| `tools/q1_natural_lifecycle_schedule_v1.py` | `90c3468be83f4bcd8f1ddde4ec836c91e52c415ce31b82d3d3c5209669a06845` |
| `tools/q1_natural_lifecycle_forcing_v1.py` | `0111ea806823f549386f61a1aafa4ac3c5efc335a4d32942db5f67f13504c665` |
| `tools/q1_natural_lifecycle_runtime_v1.py` | `1ac73fd62b0a090978844ea87003e66984d6e7fb5f3b1a66de1307527b0ccabc` |
| `identity/natural-lifecycle-source-preparation-20260930/PREPARATION.json` | `bbe95ade361816bf8ffb3f3607ce70e2624b8b901d6042209fc3a319d60a0af7` |

Paths above are relative to `experiments/review-response-20260927`. The three current predecessor files independently match their exact pins in the recipe and receipt: schedule `c7c1e53bc2cb2be78f9ef143f6eacd804c2c50765b23f9798fca142c1755afb1`, forcing `93db3462d89a643a590482c55796eadc9041023896cdde217e31a1c18e108f22`, runtime `f8735399464c6f71ce400c7d13cba7c0fbdb6a082a71630ba1fe34e153b598d8`. All generated output hashes independently match the preparation receipt.

## Delta and dependency findings

- Schedule lines 49–85 explicitly change initialization to `natural_no_import`, make every `hydrate_joint_o0` false and observe the initial natural boundary exactly once. Token/path/position construction, prior-O2 ownership, interior two-phase MTP observation and terminal-only `after_z_first`, page units and `len(tokens)+1` request budget remain unchanged. The historical fixture `hydrate_before_cycle` requirement is deliberately retained as input identity, not executed as an import.
- Static AST comparison found every forcing function/method and every Cursor method identical to the pinned predecessor. Only the forcing schedule import changes. Runtime sampled/drafts/logits/acceptance methods are also identical, preserving actual token forcing and deferred publication behavior.
- Runtime lines 22–52 retain exact frozen padded-prefix and actual prompt checks but remove common-source authentication/transaction fields. Lines 80–101 keep the live mapper and inter-forward map continuity; initial target/MTP snapshots are reads, followed by a one-time observed-boundary latch. No `Import.Transaction`, `Joint.import_common_o0`, `_sequence_tx` or `_sequence_source` remains in this runtime. Carry requires exactly the preceding observations/seals.
- Read the inherited GraphCallbacks and PublicationCallbacks connections. They operate through the actual sequence/frame/owner and `c.o0`, with no requirement for an import receipt or transaction. Graphs still imports the predecessor Forcing module only for its unchanged canonical digest function; it does not instantiate an old Sequence or compare against the old schedule. Its imported Joint module supplies accepted helper aliases, not a collector initialization call.
- The runtime seal change is the separate natural-lifecycle schema; existing receipt cardinality/order, retained O2 digests, actual target-root joins, publication readbacks and deferred event seal checks are preserved.

## Connection boundary

The existing `q1_candidate_sequence_dispatch_v1.py` still imports the common-import runtime, and the accepted sequence outer collector still calls the old source-reference signature. A natural outer/dispatch successor must explicitly select this new runtime and its `begin_sequence_case(..., prefix)` signature; the submitted files do not claim that connection exists. Do not reuse a common-import raw reader that requires import/source receipts for the new schema.

No population, repeat count or L3/L4/L5 criterion is admitted by these files. Fixed calibration inputs and the single-request cursor remain available preparation inputs. Actual APC lookup/admission, natural initialization, retirement/reuse and stale-consumer refusal still require the planned worker/scheduler and raw-device evidence; the new `state_import_performed=False` field is a source-mode statement, not evidence that those lifecycle events occurred.
