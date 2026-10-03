# Natural lifecycle joined reader: bounded source review

Source review only: no tests, synthetic fixtures, runtime imports, GPU operations or gate changes. This report preserves the initial findings and will distinguish any subsequent narrowly reviewed closures.

Initial reader SHA: `319c42a0db94419b817988d962832d284ab519eefaecc356418a464af8d549c3`. Binding reviewed at `da7c5326ac7628e78fe27f5190129f7e9faecd4b09c59854d16ad6fb5544ed23`.

## Findings and repairs observed

1. **Actual incarnation and storage ownership join.** The initial reader authenticated natural records and host/device streams separately, but only joined request strings between them. Device boundaries must also match the authenticated natural first `before.live_map` request-object, runner, batch, PID and actual tensor/storage identities. Same-ID replacement makes the request-string-only join specifically insufficient. SHA `41ddebb6dbf96cdc9ebf9d21a39246cb972250accb2e8cd2420b34dae7e77b25` adds PID/runner/batch/bank snapshot and actual old/new object joins. Remaining requested closure: bind the zero host call/enclosing update owners explicitly, and bind each raw witness view identity to the actual target bank signature or MTP map signature, rather than just accepting the separate bank metadata label.
2. **Stale source forward was not bound to its real source seal.** Initially only `oldstep < newstep` was checked. The same successor now requires the old step equal the authenticated source's first actual deferred seal, binds source run/runner/index, checks actual typed request tuples and B1 geometry, and retains the new-step/one-event-increment seal join. This part is closed by source inspection.
3. **Reset and API response evidence.** Initially reset completion was required only before admission, not before the actual local-prefix lookup supplying the result; retained API responses were not recomputed. Parent successor `2e5aa29996b3078fac8af2be7888a0fa16b2faed6ea73e8a57f09a4fe41c64a0` places reset completion before the exact lookup, requires later resets begin after the preceding planned admission, and checks actual response ID and exact integer prompt/completion usage. This closes the available-data checks by source inspection.
4. **L5 protected consumer coverage.** The generic device authenticator deliberately accepts cache-only v1 or v2 witnesses. The L5 joined reader must specifically require v2 with all eleven mandatory consumer views: products 0–4, actual output tokens and accepted rows, and the four persistent slot/spec path/length tensors. It must retain the same full extra-view inventory before/after and the observer's additional route-tensor coverage condition. Without that callsite requirement, cache-only evidence could yield the broader protected-storage flag while omitting commit outputs/control buffers. Fix requested; not yet inspected in this report revision.

## Sound scope and admission limits

The prospective binding now requires adjacent complete pairs, the composed runner hash, source retirement, L4 zero/first-use boundaries and L5 replacement first-use. It does not invent a fixed F5 input population or numerical threshold. The joined reader authenticates exact case/driver/device file inventories, full event streams and planned reset counts, then independently re-runs natural raw and zero/complement audits. It preserves explicit false whole-state, lifecycle and workload qualification flags, and leaves external container/source/terminal authentication to its caller. These scope limits remain appropriate.

## Final bounded closure

Reviewed final reader SHA `50db66f5bc092e13f858a4ac886f8cee397fca9b1f28a33400abdc3af1b74b93`; binding remains `da7c5326ac7628e78fe27f5190129f7e9faecd4b09c59854d16ad6fb5544ed23`. No remaining blocker was found in the requested repairs. This is source acceptance only; no positive runtime corpus or F5 qualification is asserted.

- New `bind_views` (lines 42–54) checks each target view's actual object ID and eight address/storage/shape/stride/offset/dtype/device fields against the authenticated natural bank list, using the existing signature converter. MTP is separately checked against its actual map signature, including the existing `offset` to `storage_offset` conversion. Both normal device boundaries and stale before/after witnesses call it.
- The zero call's before/after host runner IDs and enclosing update's runner/batch IDs now equal the authenticated natural map (lines 142–152), in addition to the corrected request-object joins.
- The L5 callsite now requires v2, the eleven mandatory current product/output/path/length names, more than eleven total consumer views, and the same full extra-view inventory before/after (lines 243–252). The generic byte reader remains unchanged.
- The previously inspected exact source/new-forward seal, typed B1/request, reset-before-lookup/after-previous-admission, and actual API response/token-count checks remain present.

Only source and AST/schema inspection were used. No tests, unit fixtures, GPU work, runtime imports or gate changes were performed. The earlier findings above are retained as history and are closed by this explicit successor. External terminal/source/container authentication and the existing pending numerical/lifecycle claims remain outside this source verdict.
