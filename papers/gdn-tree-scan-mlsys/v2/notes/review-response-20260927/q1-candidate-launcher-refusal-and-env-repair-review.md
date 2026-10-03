# Independent Q1 candidate launcher refusal / environment repair review

2026-09-28; run `q1-candidate-stage1-20260928T181822Z`. **Refusal diagnosis accepted; repaired executable integration remains pending.** No numerical-policy, serving-setting, boot, or gate change is approved by this note.

## Authenticated failure evidence

An immutable local payload snapshot is retained at `p0/monitor/review-response-20260927/candidate-env-refusal-reviewed-20260928T1830Z/repo/`: 41 files, 1,397,537 bytes, all equal to read-only remote hashes. All 19 files bound by the candidate run receipt match size and SHA-256. The snapshot includes the staged actual launcher and siblings, wrapper, renderer chain, host environment and JSON, original gate snapshot, launch binding, raw refusal/reducer logs, authority and memory receipt.

| Artifact | SHA-256 |
| --- | --- |
| Snapshot index | `9ca04febb43783c4661abe752b020cc61f1fdef83fa712a5f38c51d420cbbad9` |
| Independent refusal audit | `e0a3971d3fef8d9145d14f64a67bd2e19286df1ad58f7653acd5ec88c9616edf` |
| Sealed worker refusal report | `3ddcf32b854446936caae8b990b1bfce82f21c68e8a94325ae150fa4202cf39b` |
| Candidate run receipt | `55dd78190155a49f48d6cda2fc6600d3f7cbb1edf17963ab23ee2f20911a5996` |
| Actual staged diagnostic launcher v3.1 | `f53abe7b03a1f5b92ca35f866fdc5e5303c2ffeba142472c308b2523780a501e` |
| Host env v2.2 | `816b3a5a82502cbf754b13df94a550c303197b48fc64df8cc11295f664f77347` |
| Renderer v2.2 | `bd497f9afa652d6f5ebf5943fa1530dc78bb58cb2a1e73e01a2ba46f219fa558` |
| Launcher stderr | `7cd489615bc24bea2eb441fd00c634ea503c29c0c152e422ad2a196aac033798` |
| Original approved gate snapshot | `812f2c032c657820e3f5321030d9f47c87d95c227026f2c2a334cb1adf2874cf` |

The host-env bytes match both the gate and launch binding and are reproduced exactly by the frozen renderer's pure `host_env_lines` function applied to its sealed JSON. Local CPU execution of the actual wrapper's `set -a; source; set +a` export sequence confirms that all four private names are present with empty values in the child environment.

The staged launcher records caller membership with `[[ -v NAME ]]` at lines 281–288. Its unconditional private-sidecar refusal at lines 1838–1843 tests `set:*`, so empty values must refuse. Renderer lines 108–113 forward the four receipt-derived empty assignments; this explains the exact raw stderr and exit 2. The derived tier-B credential/workload route was entered beforehand, as raw stdout records. `docker run` is later at line 7077 and was not reached.

The candidate receipt records no owned CID, no driver start, zero objects, and no health event; there is no candidate CID file or case output. Together with this source control flow and exact refusal, this supports **zero candidate containers, model boots, and driver requests**. `engine_started_utc` is written immediately before invoking the host launcher; it is an invocation timestamp, not evidence that the engine started. The separate successful memory-readiness operation did create and remove one pinned-image query container, so a blanket “zero containers for the whole attempt” statement would be incorrect.

The no-data reducer traceback is preserved (`f309f764eb46ca69a04de68fcf2fb8a6219a3e231392f87e028112b02c61c710`); no reduction or scientific result exists. A report-only no-driver reducer disposition can be added separately but is not needed to explain this refusal.

## Minimal repair and connected controls

1. In versioned renderer/env v2.3, omit the four `FR13_FA2_QROW32_B{1,4}_PRODUCTION_PASS_SIDECAR{,_SHA256}` assignments. Preserve every other scientific/serving value, diagnostic derived-route identity, model/binary pin, and the actual launcher's credential guard.
2. At the effective child-environment boundary, reject unexpected nonempty inherited values without printing their contents; remove legitimate empty inherited defaults with explicit `unset`. Merely dropping assignments leaves inherited empty values present. Silently unsetting nonempty credentials would erase unexpected caller input rather than preserve the trust boundary.
3. **Check source failure propagation.** Existing wrapper line 175 uses bare `source` under `set -uo pipefail`, with no `-e` or return-code check. A sourced env that only `return 2`s on contamination is insufficient: the wrapper proceeds. Either exit the owned wrapper (its existing EXIT sealer handles prelaunch failure) or use an explicitly checked, versioned wrapper source boundary. The connected negative must confirm no launcher/container call follows a refusal.
4. Do not derive a universal forbidden-key list from every `set:*` occurrence. `FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO` at line 325 is part of the decision to avoid ambient `.lumo.local.env`; it is not one of the unconditional private-sidecar refusals. Lines 2968–3035 separately validate a legitimate FP8 binary selector. Keep it absent for this unchanged NVFP4 candidate, as v2.2 already does, without changing unrelated valid routes.

Required bounded CPU controls for the delivered repair: original v2.2 reproduces the exact refusal; v2.3 clean and inherited-empty environments reach the real launcher's first stubbed `docker run`; each nonempty private name refuses before that boundary; actual renderer/env/wrapper/launcher bytes and final argv are bound. Assert unchanged derived credential/patcher/workload, image, model, FA2 route, GPU_UTIL 0.7, geometry, graph settings, and request/job bindings. Stub external commands without altering the launcher's guard/control-flow source, and prevent any actual Docker/GPU/privileged call. Label any substituted host binary/credential verifier in the harness so coverage is not overstated.

The local machine has Bash 3.2, which cannot execute this Bash associative-array/`[[ -v ]]` launcher. The export-boundary control and pure renderer reproduction above are complete; full connected Bash 5 launcher controls must be reviewed from the worker's versioned CPU-only harness and raw receipts. No full-launcher pass is claimed here. The v2.3 source was not yet delivered at the bounded inventory; final repair acceptance is pending that package.

No remote write, GPU query, cache operation, container, model boot, or calibration was performed by this review. Existing scientific evidence and counters remain unchanged.
