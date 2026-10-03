# Candidate hydration v2: bounded closure review

Reviewed the immutable parent snapshot `p0/monitor/review-response-20260927/candidate-runtime-draft-20260928T0418Z`, captured 2026-09-28 04:18:15 UTC. All seven files match their snapshot SHA-256 and byte lengths. This is an early source review, **not final freeze acceptance or launch approval**. Only the importer and its hook call sites were assessed; no launcher/driver review, Torch import, GPU/container execution, SSH, or source/gate changes occurred.

Source hashes:

- `tools/q1_candidate_hydration_v2.py`: `0e0b94102ca07d3da42513c4188e2020fe3bd8d74a99386f1f7f72ee9c7d0462`
- `tools/q1_candidate_hooks_v1.py`: `5fe75b4a7c5ef0cd950eb242f97ea17832e1f4fbcd66417d6c699fe0110a6fcb`

## Closed portions

The hook constructs all candidate destinations and calls `plan_all` before `apply_all` (`hooks:242–258`). Complete native/candidate layer sets are checked; every contiguous destination byte span is compared for overlap (`hydration:55–89`). All source objects are authenticated before copying (`124–156`). After **all** copies, every imported region is read back from live tensors, checked against its object, and folded in native sorted layer/block order into the bound complete O0 digest (`172–191`). The hook independently snapshots post-import O0 and compares its digest (`293–295`). This closes the earlier stale per-layer receipt problem.

The new complementary guards execute before and after copying, and `guard_scope` truthfully limits them to selected padding, null/adjacent unselected GDN rows, KV tail suffixes and adjacent unselected blocks (`90–108`, `159–161`, `192–193`). Farther regions and other tensors are explicitly excluded. That is a bounded guard scope, not a claim that the entire complementary allocation is hashed.

## Required repairs within the original three findings

**F1 — Hard fail-stop is still incomplete.** In `apply_all:170`, `copied` increments only after `copy_` returns. A first copy that partly writes and then raises reaches lines 198–199 with `copied == 0`, producing `process_unusable=False`. The hook's handler at lines 288–292 then returns to the current forward. Mark mutation as attempted immediately before the first device write; any exception from that point must be unusable, even if no copy finished.

After a successful import, an exception in the second O0 snapshot (`hooks:294`) is caught by the generic handler at lines 303–305, which seals INVALID and returns without marking the process unusable. Track import/mutation entry across the full S0 transaction through final snapshot/digest and fail-stop for every exception after that point. Set the in-memory unusable latch **before** attempting receipt/marker I/O; the current ImportFailure branch seals first at line 290, so a seal-write exception can prevent the latch from being set. A best-effort failed receipt must not be a prerequisite for stopping execution. The next pre-forward `_guard` is useful, but the failing current pre-forward must also raise.

**F2 — Runner identity and exact production alias bindings remain incomplete.** `_identity_snapshot` (`hooks:117–128`) now rereads tensors from the actual runner registry and records their Python IDs, pointers, strides and offsets. It still omits shape, dtype, device, shared storage base and the bank-container identities promised by `hydration:14–15`. It records only element IDs/pointers from optional `_FR13_EAGER_PACK_STACKS` entries; absent registries or empty tuples are accepted, and replacing a bank tuple with a different tuple containing the same tensors is invisible. Bind the actual existing production lease/preseed bank tuples and registry-to-layer relationships, require their presence, and record tuple IDs plus the full tensor/storage signatures before and after the import. Preserve legitimate shared storage.

The same distinction affects `plan_all:63–80`: classes are grouped by **storage base**, whereas the previously reviewed production authority groups exact conv-bank **data pointers** and additionally requires each layer's conv and SSM to share a storage base (`fr10_gdn_tree_kernel.py:7877–7892,7909–7945`, unchanged SHA `d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8`). Equality of the *partitions* of conv/SSM layer names does not establish that per-layer shared-base relationship or exact bank-view geometry. Use the existing production alias/lease validator or reproduce those exact bounded checks; do not treat shared base alone as proof of an exact bank alias. The global disjoint-write check is valuable and should remain.

## Focused CPU control-flow evidence

Executed only selected AST function/class bodies from the two hashed files, with standard-library stubs and byte/view stand-ins; no Torch module was imported. These controls establish Python exception/binding behavior, not CUDA behavior:

- A stub first `copy_` changed its destination and then raised. `apply_all` reported `stage=preconditions`, `process_unusable=False`.
- A successful stub `_hydrate` followed by an O0 `_snapshot` exception caused `on_pre_forward` to return normally with an INVALID sealed case and `self.unusable is None`.
- Replacing `fixed32_banks` with a new tuple containing the same tensor objects left `_identity_snapshot` unchanged.

Minimum closure tests: first-write exception with mutation begun; post-import snapshot failure; receipt-write failure after mutation; same-tensor bank-tuple replacement; and valid production alias grouping versus mismatched conv/SSM base or bank geometry. Keep the final full-region reread and source preauthentication positive controls. The current draft is **not yet closed on fail-stop and identity/alias attestation**; no broader experiment or numerical-policy change is requested.
