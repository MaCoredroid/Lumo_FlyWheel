# Current identity gate v5 — bounded source review

**Disposition: PASS for the submitted identity-source correction. No new blocker found. This is not a runtime freeze, route qualification, or launch authorization.** Review used local files and CPU controls only; no transport, engine, Docker, GPU, or workload ran.

## Scope and evidence

The reviewed adapter is `workload-plan/tools/identity-adapter/e3_preflight_v5.py` (`c3e8a7e18089a846c2dedd3b3bedc7453376d9b04e4370ee90da99c0e911b356`). The reviewed lock is `known-locks-v2.json` (`41e5c6981d952207a8a7c8e4a15482cbb65c209f1d540dfefac49743ed6a8fb2`). The source snapshot, runnable reviewer controls, results, and supplied test log are under `p0/monitor/review-response-20260927/current-identity-v5-source-review/`.

The eight supplied `CurrentIdentityTests` passed. Seventeen additional focused positive/refusal controls passed. The report records 338 assertions including source stability and hash/size checks, across 159 retained source/manifest files; these are CPU checks, not experiments. The full `sole_executor_v3_7` module imported successfully without calling `main`, and resolves its gate and runtime to v5.

## Findings

1. **Identity alignment is correct.** `e3_preflight_v5.py:282–299` selects the exact common `/models/qwen3.8-27b-nvfp4-radixark` view and requires it to exist in the unchanged model-view identities. It pins each arm's image, including SGLang (previously only syntactically checked). All four identities agree with the retained `CONFIGURATIONS-v3.3.json` (`a08a31e3c2ea9bac2c101cc3ac6af69b87f048128e6c71e40cc9367a30fe55cd`). The common explicit-tools template is `4c9168e20b0187b30a94083bdb0940f279d1a35ed6ac5694c1ffdab444c04f53`. SGLang's config prose annotates the model path; its actual identity path and `--model-path`/`--tokenizer-path` forms agree with the gate.

2. **Source binding is checked, not merely recorded.** `load_known` (`e3_preflight_v5.py:172–184`) checks the lock hash, then every retained source hash and byte count within the campaign. All 16 source bindings matched locally. The original 13 remain unchanged; only current configuration, current template, and current prompt-parity evidence were added. A byte-modified current configuration refuses. The old lock also refuses the new expected lock hash.

3. **The pin chain reaches the real caller.** `collectors_v3.py:10–18` pins and imports v5. `closure_v3_3.py:26–27,541` pins the new lock; `runtime_v3_3.py:23–27` uses that same name/hash. `dependencies_v3_3.py:8–18` verifies the current identity/collector/closure manifests before importing them; inherited v3.2 seals still verify. `pin_observation_successors_v1.py:14` includes both v5 and the new lock in the identity seal. All current and inherited manifest hashes and member hashes/sizes checked successfully. The identity manifest is `9a505188a5049fcee3365200947e0a7de89b9aa5f20dc13d1d3ab2b654eb1cc8`. The CLI has explicit required lock-path and expected-hash arguments, with no stale default (`e3_preflight_v5.py:468–483`).

4. **The correction does not weaken scientific admission.** Re-keyed synthetic configurations with current identities pass the configuration layer. Old SGLang view, old template, crossed images in every arm, missing model mapping, omitted runtime source manifest, absent qualification (both `None` and a valid-looking but unavailable hash), `auto` precision, and mismatched model argv refuse. Source-manifest authentication, route qualification evidence, precision checks, runtime freeze, schedule, and observation gates remain unchanged (`e3_preflight_v5.py:296–338` and later validation). The supplied fixture uses vLLM-style argv for all arms, so the independent positive control separately checked SGLang's actual model/tokenizer flag forms. Neither positive fixture is real route evidence.

## Boundaries

`CONFIGURATIONS-v3.3.json` remains `launch_authorized: false`. This review verifies the current identity selector and source chain; it does not manufacture missing runtime/qualification receipts or approve any boot. The old adapter/lock remain separate files, and the preserved predecessor caller/collector/closure bytes remain under `before-current-identity-20260929T012358Z`. The unrelated worker-recipe v1.3 change was not needed for this bounded review and is not newly approved here.

## Active downstream hashes

- `collectors_v3.py`: `f145c7443d9fe2cdaae58541c3d000843e83d7c7c13094251284174c0fdfd13e`
- `closure_v3_3.py`: `791a983b6cc0f7600b5554cad10fa3f9ef42d650a8de2619c5911a1ed4372d23`
- `runtime_v3_3.py`: `3c2da10b0c2325b51522f99d2a42c8d3783920ecebbbbc1c137caa85771a498b`
- `dependencies_v3_3.py`: `e6fc21f41a2c39c82ab28cba7c7c0118e5f2e0074e9e808430405d9a64e4a31c`
- `pin_observation_successors_v1.py`: `b28e7e1acb9d00a38b6a57e5ef4fbe9ace13af702e4b0a8b0c0e878b5addcca4`
