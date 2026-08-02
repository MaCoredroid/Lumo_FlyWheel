# Qrow16 attested production selector

Status: source-complete, default-off, build and real measurement pending.

`FR13_FA2_QROW16_PRODUCTION=1` is accepted only for fixed32 B1 and is mutually
exclusive with the live A/B. It requires:

- the exact candidate SO and its SHA-256;
- the prior live-paged PASS JSON and its raw SHA-256;
- zero BF16 output-byte and FP32 LSE-byte mismatches in that PASS;
- the PASS candidate SHA to equal the production SO SHA.

The launcher validates those inputs and issues a mode-0600 canonical sidecar in
the new run's log directory. The sidecar binds the live result digest, candidate
SO digest, instance, output digest, LSE digest, and its own canonical digest.
The container revalidates the sidecar and installed SO before setting the
launcher-private production attestation marker.

Only during final FULL B1 graph capture does `TreeAttentionImpl` set the private
C++ qrow dispatch around an exact 32-row production tree call. The C++ selector
throws if any geometry predicate differs. Throwaway memory-profile graphs,
preseed/eager calls, and unrelated FA2 calls remain stock. The private selector
is removed in `finally`, so there is no selector leakage. Capture finalization
also requires 16 distinct target tree layers and writes
`/logs/fr13_fa2_qrow16_production_capture.json`; missing/bypassed layers abort
capture rather than silently serving stock.

## Activation

```bash
export FORKED_FA2_SO=/absolute/path/to/qrow16/_vllm_fa2_C.abi3.so
export FR13_FA2_QROW16_SO_SHA256=$(sha256sum "$FORKED_FA2_SO" | cut -d' ' -f1)
export FR13_FA2_QROW16_LIVE_PASS_JSON=/absolute/path/to/fr13_fa2_qrow16_live_paged_ab.json
export FR13_FA2_QROW16_LIVE_PASS_SHA256=$(sha256sum "$FR13_FA2_QROW16_LIVE_PASS_JSON" | cut -d' ' -f1)
export FR13_FA2_QROW16_PRODUCTION=1

# Run the normal fixed32 B1 FULL-graph launcher and real SWE-Verified workload.
```

This source artifact contains no GPU correctness or performance claim.
