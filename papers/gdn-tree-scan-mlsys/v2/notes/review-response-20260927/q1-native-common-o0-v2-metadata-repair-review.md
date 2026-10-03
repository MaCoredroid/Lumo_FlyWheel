# Common-O0 v2 deployed-metadata repair review

2026-09-29. **PASS for this bounded source/CPU repair.** No unresolved material finding in the reviewed delta. This is repair readiness, not a GPU result, numerical qualification, full-Q1/MTP qualification or launch authority. The failed v1 attempt and original natural-prefill failure remain adverse records; its packed controls remain separate. No implementation or gate was edited, and no GPU/model/container/memory operation was performed.

## Repair and actual deployed contract

The SHA-pinned deployed `GDNAttentionMetadata` class has `non_spec_query_start_loc` and `num_actual_tokens`, but **no** `non_spec_query_start_loc_cpu`. Exact source: `experiments/review-response-20260927/workload-plan/inspections/cpu-native-state-map-20260929T121600Z/gdn_attn.py:39–74`, SHA `3c60cde9b0ef4bb6e4c105491afa50d96cad368399d36bfbed81e47a6e1b4b68`. The CPU variable exists only inside its builder (e.g. lines 210, 303–343), and is not a returned dataclass field (lines 424–450). Thus v1's attribute access was a real harness compatibility defect, not a numerical failure.

`tools/q1_native_common_o0_v2.py:79–86` retains independent checks of actual `runner.seq_lens`, GPU query interval and `runner.query_start_loc.np == [0,1]`, recording the CPU values in `prepared.query_cpu`. Lines 116–144 retain all six exact decode/prefill/spec counts and the actual per-layer GPU interval `[0,1]`, require `num_actual_tokens == 1`, and truthfully name the CPU source `runner.query_start_loc.np`. It no longer invents a field on the deployed metadata object. The same `live_destinations` checks still run before copying and at post-copy identity validation.

`tools/q1_native_common_o0_audit_v2.py:41–42,54–59` independently requires the recorded prepared CPU/GPU intervals, sequence extent, counts, actual-token count and CPU source label. No numerical envelope, full-vocabulary argmax, imported-O0 equality, raw object authentication, state-map/alias checks, target-only boundary or fixed denominator was relaxed.

## Connected controls and unchanged scope

The v2 CPU tests extract only the real dataclass AST from the exact SHA-pinned source, then instantiate it with supported fields; they do not import the vLLM module or synthesize the absent CPU field. The added positive test explicitly confirms that field is absent. Added negatives independently corrupt `num_actual_tokens` and the runner CPU interval; the existing GPU interval, count, live-cache identity, pre-mutation refusal, position-axis and fail-stop controls remain.

I independently executed the exact synced test file on the remote host using `/usr/bin/python3 -B .../tools/tests/test_q1_native_common_o0_v2.py -v`, with `CUDA_VISIBLE_DEVICES=''`, `OPENBLAS_NUM_THREADS=2`, `OMP_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1`. **7/7 tests pass, exit 0, 2.129 seconds.** All nine v2 source files and the test file matched local hashes before and after execution. The retained full log contains inherited unclosed-file ResourceWarnings; no test error/failure occurred. Tests use CPU tensors and temporary synthetic object files, not model data. This run is independent of the parent's seven-test receipt.

Pure CPU construction also confirms:

- v1/v2 corpus values differ only in schema and source hashes; same 84 cases, A/B × R2, 336 observations, fixed all-case A/r0 source selection, unchanged criteria/non-claims.
- Rendered serving argv, environment, model/image, resources, mounts, seed, precision, graph and cache settings are identical; only successor patcher/script identity changes. The JOB remains correctly canonically sealed.
- The five-anchor runner patch changes only the imported hook module from v1 to v2. Config, hook, driver, audit, corpus and launcher all name the connected v2 source chain. The corpus hash inventory includes the actual new dependencies; legacy helper/source-manifest versions remain intentionally unchanged.
- All nine frozen v1 source files still match `fullmodel/native-common-o0-v1/CORPUS.json` SHA `1a31e846218df03300b560f2e0d470edce79960e2d971a46a871ff94a34ac7ae`. No replacement of v1 evidence occurred.
- Python compilation and `bash -n` pass. No shell launcher, server, Docker or CUDA query was executed. Final v2 corpus/fresh gate preparation remains a parent action after this review.

## Reviewed hashes

File names are under `experiments/review-response-20260927/tools/`.

| File | SHA-256 |
| --- | --- |
| `q1_native_common_o0_audit_v2.py` | `eaafef1ea14fc544c2a5dbe2bd05c7ef0903e0336e61a9642282d73b98d58bff` |
| `q1_native_common_o0_corpus_v2.py` | `bc4e04b0b6cabcc0a4f10ab63e9fd3859ce2721a22dac0a9505bf441362c1c37` |
| `q1_native_common_o0_v2.py` | `a080f7623b83fc2f3a144bbc86b1d233d1f0d58a76bcd3b4f31e46cf0a6f3f63` |
| `q1_patch_reference_common_o0_v2.py` | `81e562ca23f313dccd9da750c11ceabe574d3a7f00ae51c88e7567fa456a57f1` |
| `q1_reference_common_o0_job_v2.py` | `2298ca183fee296c33d9ff3f7e0e19bec5813d6c137ef4f841f160b02b8c66cb` |
| `q1_reference_driver_common_o0_v2.py` | `ae3f16b18af0f76d0181cba82bb6f22b5316f8b510273c377aed83e0f85ea109` |
| `q1_reference_hooks_common_o0_v2.py` | `afc097aced90221fd79d0ed77c8c7c3ee1cff8775c0522e0544b65df0204ec40` |
| `q1_spec_off_common_o0_config_v2.py` | `03352ed3e4202282fb117c5fe5ff58c05b99771b98142c600937240f4b27772d` |
| `run_q1_native_common_o0_v2.sh` | `1fb4d1ed9504fb2dfbd7031d93683a9a0b96aa757a4bb711789fdf759185395b` |
| `tests/test_q1_native_common_o0_v2.py` | `360b089bc26af37a417aa856ae314b788da5e528329252036c0641df42febe84` |

## Evidence

- `p0/monitor/review-response-20260927/common-o0-v2-repair-review/STATIC-CHECKS.json`: `5a28c5d51e8462bbce394c59bb3d578448a35eb8a6801862b3a05f3728c246a2`
- `p0/monitor/review-response-20260927/common-o0-v2-repair-review/INDEPENDENT-CPU-TESTS.json`: `1f1c309ec6365bc7d2edf4434a33a7030e6e7e92f38f57bf65bfbaa3d210a153`
- `p0/monitor/review-response-20260927/common-o0-v2-repair-review/V1-PRESERVATION.json`: `bf4d3c7b5ae6146da2e2a2c6b0e81cd16fac4d369c207bc48f1eb0b40a1fc82a`
