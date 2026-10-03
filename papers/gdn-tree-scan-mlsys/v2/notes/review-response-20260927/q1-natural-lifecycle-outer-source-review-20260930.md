# Natural lifecycle outer/raw connection: bounded source review

Disposition: two concrete joins need repair before this collector is treated as source-ready. Review covers only the new natural binding/hooks/dispatch/raw/phase successor, with selected predecessor predicates for the removed import. No tests, synthetic fixtures, runtime imports, GPU actions, gates or implementation edits. Both preparation receipts' predecessor/output hashes match the inspected files.

## F1 — Planned request keys are not bound to actual worker IDs

`q1_natural_lifecycle_binding_v1.py:97–108` derives deterministic HTTP request IDs, including the identical key required for same-ID replacement. `q1_natural_lifecycle_hooks_v1.py:99–123` adopts any B1 actual worker `rid`; it checks prompt/sampling and subsequent continuity but never relates that ID to `control.http_request_id`. `q1_natural_lifecycle_raw_v1.py:98–107,207–210` authenticates the planned entry and self-consistent actual event owner separately, without joining them.

Consequently, the complete recorded observations of a planned same-ID pair can have distinct actual worker request IDs without this collector/raw auditor refusing them. The final lifecycle reducer may still refuse, but the outer's advertised request-plan binding is incomplete at this connection.

Minimal correction: derive the exact expected actual request ID using the pinned completion ingress and internal-ID mode, then enforce the same relation in the live bind and raw auditor. The retained source `workload-plan/inspections/cpu-completion-salt-20260929T112500Z/entrypoints/openai/completion/serving.py:132` adds `cmpl-`; its engine serving `_base_request_id` uses `X-Request-Id` when present. Bind the actual internal randomization/prompt suffix rules too; bare header equality is not automatically correct. The planned salt can remain HTTP-side evidence because CachedRequestState does not expose it, but the future driver/API receipt and final reducer must join it to this worker observation.

## F2 — Removing the import audit also removes natural allocation proof

The old `q1_candidate_joint_import_audit_v1.py` mixes import predicates with valid live-allocation checks. The new raw auditor removes its whole call at lines 161–170, while its otherwise unchanged `live_map` is only a subset of those checks.

Specifically, the removed lines 50–79 require positive owner/context/config/request-object/batch IDs, four unique group IDs, exact bank registry and order, tuple/member identities, layer-to-bank transpose/SSM relationships and the 16 classes ×3 GDN topology. Removed lines 85–92 require the MTP layer/builder owner IDs, complete allocation bounds and target/MTP same-device disjoint spans. New `live_map` reads per-layer signatures and maps but does not require `_fixed32_order`, either bank tuple, `_layer_bank_relationship`, the owner fields above or MTP layer/builder IDs. Omission of those fields therefore is no longer refused by this auditor. Individual raw-state geometry and hashes do not restore those joins.

Minimal correction: extract and invoke the non-import identity/bank/alias/MTP-separation checks on the first natural live map, joined to the actual initial target/MTP raw snapshots and first owner. Keep unchanged per-cycle `Maps.extension` and state/publication checks. Do not reintroduce a transaction, common-source digest or import receipt as a prerequisite for natural state. Legitimate shared target allocation views must remain accepted; only the accepted actual span/alias conditions should be retained.

## Checks retained and scope

The dispatch delta is only its runtime import. The phase reader changes its initial MTP anchor name from imported to natural, preserving all phase/token/position/graph/raw predicates. The natural raw reader still authenticates full state/tensor payloads, first natural snapshot equality, carried logical prefix digests, per-cycle physical maps, prepared tokens/staged drafts, publication-owner/census/deferred-seal joins, full target heads, actual MTP phases and publication readbacks. The outer has independent entry order, complete/partial record preservation and inherited fail-stop. No hidden call to the common-state importer remains in the new runtime connection.

The plan's pairing rules are intentions only: they do not prove an actual cache hit, retirement, physical reuse or stale refusal. The returned flags correctly leave these false. The unfinished API/driver/device-event/terminal-reducer connection is not counted as a new defect or as a qualified population.

## Source hashes

- `tools/q1_natural_lifecycle_binding_v1.py`: `d519a44c45318e5998d88a58f4b2693d9c1e14b45697897692061348a47a61b8`
- `tools/q1_natural_lifecycle_hooks_v1.py`: `5d7820530350dca58a8fd8b83019b1a59bb75fc91f1d9f573bf1fbb0da577dfa`
- `tools/q1_natural_lifecycle_dispatch_v1.py`: `39c75cfef7e578ec39082290e4d7a390cd5bbe38d5c5674bf5fa00395e7d4f13`
- `tools/q1_natural_lifecycle_raw_v1.py`: `7c84353f316708a1237df691180abc291525aee216244f295075b42f3957d69c`
- `tools/q1_natural_lifecycle_phase_raw_v1.py`: `2b46288e39ed141b3f538d6fd73ab94c4c6fe8b4afdc37126e1bac0183271dea`
- `identity/natural-lifecycle-outer-preparation-20260930/PREPARATION.json`: `cde7a3ee0cf679e8049ff1696ea45aa5584c5393ccafae115b15f0fea256217b`
- `identity/natural-lifecycle-outer-preparation-20260930/RAW-PREPARATION.json`: `c7b0839a23a8be21293a5173d17acc93aaf1b41290378be2756dab85d9303505`

## Repair closure — source delta only

Both findings above are closed in the inspected successor bytes. Preserved original files remain under `identity/natural-lifecycle-outer-pre-review-20260930/`. No tests, synthetic records, runtime imports or device operations were executed for this closure.

F1: binding now requires an explicit request-ID mode, rejects randomized-mode L5, derives `cmpl-{http_request_id}-0` with exactly eight lowercase hexadecimal suffix characters only in randomized mode, and joins that ID at every live bind and again in the raw auditor. Enabled hooks also require the actual `vllm.envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` boolean to match the declared mode. This matches the captured current-image completion route (`entrypoints/openai/completion/serving.py:132,167`) and input processor (`v1/engine/input_processor.py:224–232`). All six retained HTTP source files match their manifest; manifest SHA `5e1b7e5befcfe600d3cf237ad35f62f931381c9cfc4a0ab15d8d5a1cfe41b3fd`. The future driver must use the same single-prompt ingress and retain the transmitted salt/header/body evidence; no such execution is claimed here.

F2: new `q1_natural_lifecycle_allocation_raw_v1.audit` restores positive actual owner/group IDs, exact registry/order/relation/tuple-to-view/transpose checks, the 16×3 GDN alias population, actual MTP layer/drafter/builder identities, allocation bounds, shared device and target/MTP span separation. The natural raw reader invokes it for every cycle against already authenticated target/MTP snapshots and actual publication owner, while retaining the logical-map and carry joins. It imports signature helpers only; it does not require or synthesize an import transaction/source receipt.

No further concrete blocker found in these corrections. Source readiness remains limited to the submitted outer/raw connection; no executable F5 plan, lifecycle verdict, scientific gate or launch is approved.

Repair hashes:

- `q1_natural_lifecycle_binding_v1.py`: `8a1486dd93616795c16e948bdf6a5f5ad11cb47904f16f06ff1e42eee7899af0`
- `q1_natural_lifecycle_hooks_v1.py`: `2a75404f18241db69c4a96b0ecbafa171f41c1d27614bf38e5d833e2aec90556`
- `q1_natural_lifecycle_raw_v1.py`: `251b9e2e7db1778a4dd58ba8c8655d1b76c555b6b584f332d2a6eeadcaf762f9`
- `q1_natural_lifecycle_allocation_raw_v1.py`: `fb33b1241201e5bfa3a64af781ae1561b1315bf5dca1bbcdee3beec4ad84143b`
