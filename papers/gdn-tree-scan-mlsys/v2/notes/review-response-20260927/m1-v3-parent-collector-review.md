# M1 v3 collector: bounded parent evidence review

Disposition: HOLD for evidence-completeness enforcement and the independent runtime interface repairs. This review does not change the first-stage design, numerical criteria, methods, paths or sample count. No GPU, Torch/CUDA import, container or real collector run occurred.

Reviewed snapshot `p0/monitor/review-response-20260927/m1-v3-reviewed-20260928T0455Z`: all 21 files match the remote frozen bytes. Freeze SHA256 `abd45b3c72e8eed43a35fd608bef9288cee38d8e1d9e5f3399832746f4259599`; collector `512bdab2a0b0e7283d1f09b7b9d85793a147b242a7853179bd3e5b4e937c4940`; driver `4ad560d96844ad4fd6abe7bfc484f01ac0fbb16925c217c27d46c64801a29d75`.

## Source closures retained

The actual caller now generates one common operand set, hashes q/k/v/a/b/A_log/dt_bias/S0 per layer and delivers it to all four serial methods. The first-verification capture is hooked at executed Lumo scan, Weaver output, or TreeWY remapping, before later steps overwrite outputs. It computes the named C2 per-head RMS/Linf/nonfinite diagnostics for 48 layers and 28 active physical nodes, and stores raw output bytes. Durable state snapshots occur at reset and each logical publication; TreeWY's deferred publication and final flush are represented correctly in the finite-cycle loop. Reference state advancement is a declared sequential C2 chain; errors are diagnostics, without a new tolerance or qualification claim. Computed allocation/traffic quantities remain explicitly unmeasured. Preserve these closures and the previously reviewed author-default/aligned variant distinctions.

## One bounded evidence-completeness correction

The collector records evidence but does not enforce all required capture premises before declaring the stage complete:

- `endpoint_state_diagnostics`, lines227-228, records each layer's exact reset-S0 comparison, but a false value is not rejected or made a stop condition. `on_state`, lines286-288, tests only nonfinites. Exact common S0 is an input premise, not a proposed numerical tolerance.
- `run_method`, lines305-313, permits `first_verification_diagnostics_summary=None` and appends endpoint capture metadata, then calls `validate_v3`/`reduce_runtime`. Those validators do not inspect the new diagnostic/capture fields. Missing output coverage, unavailable cells or missing endpoint captures can therefore remain outside the completeness verdict.
- Source inspection of the v2 validator confirms it has no knowledge of these new collector fields. A standard-library AST control of the actual v3 validator (v2 receipt validation explicitly stubbed as already satisfied) returns no new problems for either missing first diagnostics/zero captures, or a false reset-S0 comparison. This proves the missing v3 guard, not that an actual GPU run has lost captures.

Minimal repair: enforce the already-frozen capture contract for runtime completion. At the initial capture, require all48 reset states have the correct shape/dtype and exactly equal the corresponding S0 before verification; retain evidence and mark that method failed on mismatch. Before accepting a complete method, require first diagnostics and48 authenticated output references with all28 expected node entries per layer, zero unavailable/nonfinite/invalid cells, and exactly four raw endpoint captures (reset plus three publications, including final TreeWY flush), each covering all48 layers. Validate recorded raw tensor reference shape/dtype/length and completeness consistently with the store. Partial captures are valid failure evidence but cannot be complete-stage evidence. Do not add any bound on nonzero finite C2 errors; numerical acceptance remains outside this untimed stage.

Use focused CPU metadata/stub negatives for missing first output, one unavailable node/layer, missing final flush capture and false reset, plus a complete positive fixture. No new GPU experiment or enlarged test matrix is requested.

## Runtime and launcher dependency

The independent reviewer owns the manifest path resolution and constructor-failure closure report. The parent separately reproduced the constructor exception escaping `run_method` at line289 before its try: an actual AST call with a failing factory raised and retained zero files in a temporary directory. Treat this as the same finding, not another issue. Keep the original exception and any raw evidence if subsequent diagnostics fail; do not let evidence serialization erase the primary failure.

The host launcher remains to be implemented by the owned Claude executor; the parent owns approval, not code production. The parent clarified this handoff and authorized CPU preparation. A final freeze with the launcher, actual path-based identities and evidence guards is required before the first untimed stage gate. No launch, memory query or reclaim is authorized by this review.
