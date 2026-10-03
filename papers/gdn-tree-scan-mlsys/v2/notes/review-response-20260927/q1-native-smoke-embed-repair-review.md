# Native-smoke embed-path repair: independent bounded review

Reviewed 2026-09-28T02:57:24.336526+00:00. **PASS for the bounded source/CPU preparation repair; no material blocker found.** This is not launch authorization, a successful live smoke, numerical qualification, or permission to expand the experiment. Parent owns the next gate. The prior attempted run `q1-native-smoke-20260928T022909Z` remains failed: first request INVALID, no valid O0/O1/O2 observations.

## Exact reviewed source

All campaign-relative paths below are under `experiments/review-response-20260927/`.

| Executable | SHA256 |
|---|---|
| `q1_reference_hooks_v2_1.py` | `1ac939405de01a4c22e5d73decb9994cfb3a3214fd838ed2a734b39110b2b8cf` |
| `q1_patch_reference_runner_v2_1.py` | `ee61c5d779cc39dce9757bae944f51d991bf2b15ef89b04bf69e9a58c91b315a` |
| `q1_spec_off_engine_config_v2_1.py` | `cd7fbc1322c9c4206990f2223728c2b49ae551c1a644a99fbafd2576380deb14` |
| `run_q1_native_smoke_v2_4.sh` | `df5e0c1982f12060a4277bcd79d9762c371ed009c8a5ed47d3cfc7062903a721` |

Final freeze `FREEZE-Q1-FULLMODEL-DESIGN-v2.5.json`: file SHA256 `81b94eb336080ec5fa3aa6043b136b8b1667454fc0dcc740eda16aa15568bdc5`; recorded canonical SHA256 `db4fcf1485095c061f1a08587b7a80ef0f2e6df8664cb4c54fae04ae8ca05f88`. I independently verified all **61 campaign members + 3 repository members** against their recorded SHA256 and byte sizes, with zero mismatches. This count excludes the freeze file itself and the remote FA2 binary; the parent separately verifies remote parity/binary. The four executable hashes equal the preserved preliminary snapshot. v2.5 adds tests/provenance while preserving v2.4 and the executable bytes.

Pinned stock runner: `identity/native_source/vllm__v1__worker__gpu_model_runner.py`, SHA256 `904d7d357ba076c93cefee5f7f030e2be577dc12599aebdf56395b894e2077b0`. Independently rendered patched runner: SHA256 `3bc57571d11944ec9af43f1db49a51d1685e1a75b625815f265699deba65b73f`; all five anchors occur once and the emitted Python compiles without importing vLLM.

## Source and behavior findings

1. **The fallback observes the actual embedding input on this route.** Stock runner lines 3214–3235 enter the multimodal-capable model path even for text. Lines 3226–3229 call `model.embed_input_ids(self.input_ids.gpu[:num_scheduled_tokens], ...)`; `_prepare_mm_inputs` lines 3182–3191 returns `input_ids=None` where raw IDs are not required. The new pre-forward hook reads that same prepared GPU buffer only when its argument is None (`q1_reference_hooks_v2_1.py:495–504`). `_preprocess` is invoked at stock lines 3986–3996; the hook is inserted immediately before `_model_forward` at 4035. There is no intervening token-buffer mutation. This observes the real prepared token IDs; it does not reconstruct them from the fixture or substitute the expected token. The Qwen source's `embed_input_ids` at `identity/native_source/vllm__model_executor__models__qwen3_5.py:628–642` embeds IDs and returns those embeddings when multimodal inputs are empty. The unchanged driver sends an integer-token prompt, not arbitrary embeddings or media (`q1_reference_driver_v2.py:88–108`).

2. **Request offset and position checks remain meaningful.** The hook uses `query_start_loc.np[req_index]` for both ID and position, retains ordinary non-None argument precedence, and rejects missing buffers/positions and wrong token/position before taking O0/O1. For two-dimensional mRoPE positions it reads row 0 and records the complete shape. This is unchanged from v2 apart from the shape/source diagnostics. The stock runner passes `mrope_positions.gpu[:, :num_input_tokens]` at 3276–3277; completion positions are prepared by `_calc_mrope_positions` at 2471–2517 using the request's position delta/context. This review establishes correct indexing of the existing scalar-position contract. It does **not** claim independently checking all mRoPE axes, arbitrary multimodal positions, or embedding-value equivalence. Those are outside the two text-token native-smoke requests.

3. **Token forcing and numerical criteria are unchanged.** `on_sampled` (556–575) still writes the prescribed next chain token into the actual sampler output, then the terminal token. The next prepared buffer is observed after stock `_prepare_input_ids` handles normal/async input propagation (1600–1705). The independent CPU seam probe executed the actual extracted `on_pre_forward` and `on_sampled` methods through prefill → root → z → terminal and checked observed IDs plus O0/O1 boundaries. AST comparison shows only `_attest_boot` and `on_pre_forward` differ; state extraction, logits/argmax, grammar guard, forcing, object authentication and final mandatory-observation checks are identical. No kernel, numerical tolerance, fixture, model, cache/sampling/graph setting, or request count was changed.

4. **Version forwarding and refusal remain bound.** Patcher imports `q1_reference_hooks_v2_1`; config invokes the new patcher at its real mounted path and derives the new hook hash. Both arms' env, serve argv and Docker argv (apart from the explicitly separate patch script string) match v2 exactly. No `--allow-hash-mismatch` appears on the actual launch path. Launcher lines 60–70 hash-bind the new `hooks`, `patcher`, `config` and launcher, then render v2.1 config at line 132; inherited cleanup remains v2.3. Existing gates naming old keys/hashes do not satisfy the new equality checks. I did not reopen unchanged cleanup or full-model protocol reviews.

## Bounded independent CPU evidence

Reproducer (stdlib only; no Torch/vLLM imports):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 notes/review-response-20260927/native-smoke-embed-draft-snapshot/independent_review.py
```

**16 checks passed**: prepared-buffer offset/non-None precedence; missing-buffer/missing-position refusal; wrong token/scalar position/mRoPE row-0 position refusal, each sealed once; valid mRoPE-shaped indexing; actual forced root/z/terminal progression; old/new method AST comparison; exact stock-source anchors/emitted compile and repatch/missing-anchor refusal; identical configuration settings; actual shell syntax/dry-run; frozen member inventory. State storage/boot collaborators are CPU fakes, so these controls do not establish live state capture validity.

The author’s focused Torch CPU controls are separately source-bound evidence, not an independent rerun: **10 + 4 pass**, logs with respectively 16 and 10 start/end/current matching source/evidence hashes. I checked both test sources, including the new wrong-position control, and verified the log-bound bytes. Local system and bundled Python have neither Torch nor pytest; no packages were installed. No GPU imports, Docker/model/API calls, remote operations, gate writes, or campaign changes were performed.

| Review artifact | SHA256 |
|---|---|
| `native-smoke-embed-draft-snapshot/independent_review.py` | `18719b360094740e4a2f539c450fa62af83d5c7e677cf5af8c61052cbd3108f7` |
| `native-smoke-embed-draft-snapshot/independent-review.json` | `fdcec3c52c680bfcf7efa26a82d8fd1c784f65305984e0126829dc28d586c394` |
| `native-smoke-embed-draft-snapshot/author-log-bindings.json` | `16ac1e7cbf692c53c59f74323d210c0407d1e4922e66647ab8f9e03a730b0838` |
| `tools/test_log.q1_fullmodel.v2_1_repair.attempt1.txt` | `4f48c0a6a754c3eb12416a445e6220b5b4600d585ace75061bab5ac6b7be77d1` |
| `tools/test_log.q1_fullmodel.v2_1_repair_ext.attempt1.txt` | `b6879d90f8188bc6b8ddfef740ae6a241f626ada36a511d72194cb3c5d8b2692` |

The live native smoke must still produce two independently authenticated VALID observations and normal terminal cleanup under a new parent gate. No such result is inferred from this preparation review.
