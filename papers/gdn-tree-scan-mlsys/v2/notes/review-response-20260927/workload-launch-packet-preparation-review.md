# Launch-packet preparation: bounded independent source review

Disposition: **preparation remains incomplete; no runtime admission or launch approval**. One concrete model-file schema mismatch must be repaired before both launch and agent gates can accept the actual collector output. A collector-role declaration is also missing. These are source/integration observations, not failed experiments. No Docker, SSH, HTTP, GPU, model reads, agent, evaluator, or cleanup operations were performed.

## Reviewed identity and bounded control

Paths below are relative to `experiments/review-response-20260927/workload-plan`. Exact bytes are preserved under `p0/monitor/review-response-20260927/launch-packet-preparation-independent-v1/` in `SNAPSHOT.json` (14 files) and `SUPPLEMENT.json` (two supporting reader files). This report describes that snapshot; it does not silently adopt later source changes.

| Source | SHA-256 |
|---|---|
| `tools/identity-adapter/e3_preflight_v5.py` | `c3e8a7e18089a846c2dedd3b3bedc7453376d9b04e4370ee90da99c0e911b356` |
| `tools/identity-adapter/known-locks-v3.json` | `c0c9d1ed924fc7452357a95d42b279f037112f5e3f4b4137636b82001f1fd86a` |
| `tools/runtime-collectors/collectors_v3.py` | `f145c7443d9fe2cdaae58541c3d000843e83d7c7c13094251284174c0fdfd13e` |
| `tools/attempt-runtime/sole_executor_v3_10.py` | `82b56406c23565897dd4ee8d663c555bba449389c72e65b52dba2f02a7d15b68` |
| `runtime-package-v1/runtime-freeze.DRAFT.json` | `c2d62f8e659aa481208569191be433755593f087363dd5963d183d2dc1ef09a6` |
| `runtime-package-v1/source-manifests/collectors.json` | `0017493b62cdc0ffd8df19fba03294899527ce6eef4bf2b39ac13242c410a19f` |

The independent standard-library control extracted the exact `Collector.files` function AST and supplied a mock snapshot, without importing an engine or reading a model. All 22 original file rows in the two known model views use `{sha256,size}`; the collector returns `{sha256,bytes}`. Identical hash/length data compare unequal until the key is renamed. The retained result is `size-bytes-schema-control.json` in the snapshot directory.

## Concrete gaps and minimal changes

**G1 — identity-preserving model schema normalization is required.** `collectors_v3.py:102–109` emits `bytes`; `runtime():205` uses that map for actual loaded files. `e3_preflight_v5.py:390,399` exact-compare against the known-lock model views, which use `size`. This is a deterministic false refusal with correctly observed files, and copying expected values into an observation would be the wrong repair. A versioned known-lock successor can rename only `model_views[*].files[*].size` to `bytes`, retaining every path, SHA-256 and integer length. Keep `dataset.size` unchanged; it has a separate explicit consumer at validator line368. Reject conflicting duplicate size/bytes fields if normalization accepts both. Update the lock/source bindings through the ordinary source freeze; no scientific or numerical rule changes are needed.

**G2 — `terminal_collection_failure` is omitted from the collector role map.** `populate_four_source_manifests_v1.py:101–102` and generated `source-manifests/collectors.json` declare successful agent completion and runtime roles, but omit this failure role. The actual collector emits a `lumotree-e3-terminal-collection-failure-v1` receipt with `REFUSED_NO_EVALUATION` when terminal observation fails (`collectors_v3.py`, `agent_terminal`). Add the role to that same exact collector member and regenerate the declaration. This is **not a demonstrated current failure-closure refusal**: `closure_v3_3.py:120–131` authenticates the exact collector source bytes and failure identity/evidence directly, without a role-specific `approved_collector` call. Its existing refusal evidence must remain preserved; no success/completion may be synthesized on this path. No other concrete role omission was identified in the scoped map.

**G3 — a fresh launch-packet builder is absent from the current connected path.** `sole_executor_v3_10.py` consumes `--packet`; `runtime_v3_3.py:106–107` checks the supplied packet. `assemble_four_attempt_draft_v1.py` prepares declared configuration and `populate_four_source_manifests_v1.py` prepares source manifests/evidence, not fresh launch observations. Test launch-packet constructors are schema fixtures, not reusable live evidence. This is a known unfinished preparation boundary, not a validator bug.

## Minimal packet recipe supported by the current validator

| Packet family | Exact content/boundary | Existing reusable source |
|---|---|---|
| Identity/order | Schema `lumotree-e3-attempt-receipt-v2`, stage `launch`, exact phase/freeze SHA/ordinal/row, retry index0, derived attempt ID. | `e3_preflight_v5.py:337–364`; derive from the final parent freeze, not a draft. |
| Prior attempts | Exactly the preceding ordinals, retained actual terminal closure receipts, `CLOSED_WITH_RECORD`, nonoverlap before this attempt opens. | Caller `prior_evidence_map`/`verify_closed_prefix`; do not fabricate a closed prefix. |
| Dataset/task | Actual readable dataset bytes with known hash and size; exact selected instance ID, repository, base commit, version and environment setup commit. | Validator365–370 and known task bindings. Do not read or project gold patch/reference fields into agent input. |
| Evaluator image | Immutable reference and image ID, actual architecture, actual Docker argument, historically bound image Git/tree proof, harness version/source-manifest digest/dependencies. | `Collector.image():90–96` for fresh image identity; retained image-preflight proof for Git fields; `evaluator_v2.verify_harness():77–81` for actual installed harness source. |
| Launch/model | Exact configured image ID/argv/model/tokenizer/template hash/content format plus freshly observed model-file hashes and lengths under the actual resolved host mounts. | `Collector.files():102–110`; `source_members()` when appropriate; source-bound host→container mount mapping. A configured container pathname is not itself a verified host pathname. |
| Timing | Frozen clock authority and boot ID, retained actual opening clock, closing `launch_gate_at_utc` after observations; exact age/skew/window rules. | Validator89–147; collector `finalize_stage():143–151`. No backdating of historical observations. |
| Observations | `evaluator_inspection`, `model_mount`, and `attempt_timeline` envelopes: exact subject digest/attempt/freeze/configuration/clock/boot/window, current timestamp, approved producer and retained nonempty raw-evidence hashes. | `Collector.envelope()`/`_envelope_at_clock()` and content-addressed `EvidenceStore`. All raw objects must be observed or explicitly historical; a declaration is not an observation. |
| Evidence closure | Map every source/receipt hash to actual retained bytes, including all four configurations' source manifests and qualification receipts, required common receipts and transitive manifest members. | Validator48–87,265–340. `validate_configs` runs over the entire frozen configuration set even for the first arm. |

The envelope verifier binds bytes and subject identity but does not independently infer the semantics of every raw receipt. The builder must derive facts from its read-only readers; it must not copy expected lock values and label them observed. Build observation windows after the closing clock is known, and validate them before returning. Let the caller refuse an aged or changed packet rather than refresh timestamps around old observations. Later runtime, agent-completion and evaluator-container evidence must remain absent until those events actually happen.

## Historical image proof versus fresh identity

Image HEAD, mode-only changes, and tracked-blob identity are immutable, source-bound historical Git proofs for an exact image ID/registry digest. They can be reused with that explicit provenance; obtaining them does not require starting another container. A fresh read-only Docker image inspection is different evidence: on the chosen agent/evaluator Docker client it establishes that the requested immutable reference resolves to the expected ID and architecture **now**. Retain both identities and their separate timestamps; do not describe the historical Git inspection as freshly repeated.

For the two-host configuration, serving-image inspection belongs to the DGX client; task/evaluator-image inspection belongs to the x86 agent client. Existing `two_host_v1.RoutedReadHost:386–424` provides this routing for a frozen server-image set; its file and clock reads remain on the DGX. The builder must use the exact selected client identities and distinguish their evidence. Harness source/dependency proof belongs to the actual official evaluator/coordinator environment, not to the task image merely because that image is present. `verify_harness` is reusable source-byte verification; the packet also requires version and dependency evidence.

## Intentionally pending draft authority

The reviewed package is `DRAFT_NOT_FROZEN`, pilot phase. Every time-policy field and all four boot IDs are null; all route qualifications are unfilled. Required common receipts still missing are `harness_tools`, `measurement_policy`, `metric_closure`, `qualification`, `reset_policy`, `retry_policy`, and `tuning_opportunity`. These are explicit pending parent decisions/evidence, not values for a builder to invent. Source/configuration closure does not establish serving-route qualification. No new experiment is requested by this audit, and no gate was opened.
