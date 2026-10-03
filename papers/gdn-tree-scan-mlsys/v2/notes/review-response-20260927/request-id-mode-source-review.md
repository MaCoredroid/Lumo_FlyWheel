# Request-ID mode correction — bounded independent review

**Verdict: PASS for this source correction and CPU controls. No new blocking defect found.** No engine import, remote operation, Docker/container, GPU, HTTP/model request, agent, or workload was performed. This is not full runtime admission or gate approval.

## Reviewed source identity

Under `experiments/review-response-20260927/workload-plan/tools/`:

| File | SHA256 |
| --- | --- |
| `runtime-collectors/request_worker_observer_v1.py` | `f4c37b47e39c5a3c250fe3b668204d10a2847e73e92a9a7fca714bbcccd99986` |
| `runtime-collectors/request_evidence_bundle_v1.py` | `6d4d2fbc8537129248a39158dc3c68924e20c8104b901553aa86c3718336f5c3` |
| `attempt-runtime/probe_boundary_v1.py` | `bf5bc8cb1989522abeb1d61a95d7cb1b1aea4e1467d11effdda417790ab2e7f6` |

Source/test copies, raw test logs and dependency/launcher hashes are preserved in `p0/monitor/review-response-20260927/request-id-mode-independent-20260929T002734Z/`. Final hashes matched the implementation files after testing. The parent's earlier bytes remain in `before-request-id-mode-20260929T004125Z/`; earlier source reviews describe their preserved snapshots, not the corrected request-ID behavior.

## Source-grounded conclusion

The retained current-image `input_processor.py` (SHA256 `17c091fef7bbaa4537335367abf8267345a782914f00fe01919c6efbbeff9c3b`) sets `external_req_id = request_id` at line 224. At lines 225–234, `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` preserves that ID; the other branch appends eight characters from `random_uuid()`.

The Lumo workload launcher `lumotree_workload_owned_v2_1.sh` explicitly passes `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` into Docker at lines 5542–5543, revalidates it after local environment loading at 6831–6838, and excludes a competing forwarded copy at 7042–7044. Its host environment also sets the flag at line 512. The AR/CHAIN v3 launchers do not supply this disabling flag. Their runtime observation must still demonstrate the frozen randomized mode; the correction does not infer actual ID behavior merely from flag absence.

The correction requires `external_id_unchanged` for LUMOTREE and `external_id_random_suffix8` for AR/CHAIN_MTP. The original method executes first and exactly once. The wrapper checks its actual returned request fields, preserves its return/exception, and records the mode only after validation. Worker-side mapping reads and the host bundle independently validate the frozen mode and the precise ID relation. The sampling record remains joined through the mapping file's hash. The boundary rejects a missing/wrong mode in the recipe before preparation. Source-specific method binding remains intact. SGLang's unchanged direct-ID join is not relaxed by this vLLM-specific correction.

## CPU controls

Reproduced **30 supplied tests**: 13 observer, 11 bundle and 6 boundary controls, all passing.

**11 independent controls pass.** They extract only `InputProcessor.assign_request_id` from the exact retained image source and execute that method with injected environment/logger/UUID dependencies—without importing vLLM. The real observer's method-source binding remains enabled. Positive paths verify AR and CHAIN randomized joins and Lumo's unchanged-ID join through worker sampling and the real bundle validator. Lumo graph metadata and process ownership are synthetic fixtures, not device evidence.

Negative controls cover actual flag behavior opposite the frozen mode, another request ID, a prepopulated external ID rejected by the original method, mapping-mode tamper, consistently rewritten foreign internal IDs, source-byte mismatch, and a boundary recipe requesting another/missing mode. Original return values and failures are preserved. The first independent fixture run had a relocated helper-directory lookup error; that script/log are retained as `*.attempt1`, and the final run points only that fixture lookup at the reviewed dependency directory. No implementation was edited.

The parent still owns final recipe/source-pin freezing and live runtime admission. This review changes neither request IDs in the engine nor serving settings; it closes the observer's incorrect assumption about existing Lumo behavior. No extra probe, retry, task, or workload authorization is implied.
