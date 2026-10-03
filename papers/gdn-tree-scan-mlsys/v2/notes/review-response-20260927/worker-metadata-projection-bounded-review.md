# Worker metadata and runtime projection: bounded independent review

Reviewed 2026-09-28. Source snapshot: `p0/monitor/review-response-20260927/worker-metadata-reviewed-20260928T210538Z/`. No engine imports, containers, remote operations, model requests, GPU work, or workload attempts were performed. The four-attempt scope remains **0/4; WP closed**. This reviews initialization instrumentation and a draft projection, not runtime admission.

## Verdict

The vLLM initialization observation is sound for the named allocated cache views: it reads only host dtype/shape metadata, excludes the pinned profiling call, emits only after successful outer initialization, preserves the original return, and releases its temporary references. One concrete labeling/coverage repair is needed for speculative arms: the map contains both target and draft layers but the receipt labels all of it `target`.

The SGLang hook is similarly limited to dtype/shape and its packed-state description matches the retained pool source. It fails closed on quantized, post-capture, or alternative pool classes. The actual factory choice for this EAGLE MTP deployment and complete rank-field definition are not established by the retained source set; parent is recovering the missing factory/ParallelState files. That is pending source applicability, not evidence of an unsafe fallback.

The projection correctly rejects explicit dtype mismatches and `auto`. It is not yet an admission validator: its worker source/role/coverage inputs and optional dispatch need binding by the forthcoming connected caller. Returning matched expected dtype spellings after comparing every observed row is not itself fabrication; accepting a self-authored receipt without source authentication would be.

## F1 — Partition the shared vLLM cache map before claiming target coverage

`worker_metadata_v1.py:51–79` walks every cache group and hard-codes `role='target'`. Pinned `gpu_model_runner.py:6882–6911` builds cache specs from all registered `AttentionLayerBase` instances, including draft layers. The retained effective Eagle source at `identity/generated_source/probe-20260928T035847Z/logs/generated/eagle.patched.py:7536–7554` identifies `_draft_attn_layer_names` from layers added while loading the drafter; its `:7889–7932` selects those layers from the shared cache groups. `qwen3_5_mtp.py:91–100` constructs full-attention MTP layers. The current checkpoint's text configuration has 64 target layers: 16 full-attention and 48 GDN; its MTP layer is additional.

CPU counterexample `test_counterexample_vllm_joint_map_labeled_target` supplies this shared-map structure and shows the draft attention row emitted under role `target`. Smallest repair: source-bound partition using the actual drafter layer-name set, retain separate target/draft descriptions, require the target layer map against the pinned model architecture, and keep the draft compute policy separate if reporting it. Do not silently discard the extra allocated map or use the patched Lumo proposer hash as the stock CHAIN proposer hash. Parent has accepted this repair direction.

No additional defect is claimed for the trailing encoder-only group: pinned `may_add_encoder_only_layers_to_kv_cache_config():6860–6880` also places those names in `runner_only_attn_layers`, which the observer skips. Likewise, shared-cache aliases are added back into cache groups by the upstream sharing helper; the current Qwen sources do not request such sharing.

## Draft projection: smallest outstanding admission bindings

These are integration requirements exposed by the draft, not a reopening of the accepted collector/adapter contracts.

1. **Authenticate the worker receipt and role.** `runtime_projection_v1.py:71–104` checks attempt/boot/PID and numeric dtype equality but ignores `engine`, `role`, `producer_sha256`, `source_binding`, and the metadata-only scope fields. CPU counterexamples demonstrate that a record missing all producer/source fields, or relabeled as another engine/draft, still passes. Before connecting it, bind exact worker producer/dependency and effective engine-source hashes, expected engine/role, and actual worker PID/container/boot observation to the retained raw receipt. A coordinator's own process/source observation is not this worker observation. The existing all-request/clock/container ownership obligations remain in force.

2. **Freeze the target layer map independently.** The exact-set comparison at `:91` works if `expected_layers` comes from the pinned model/route. A one-attention/one-GDN record passes if the caller derives `expected_layers` from that same incomplete record. Keep the 64 target-layer mapping independent of the observation and include only the intended worker/rank's partition. The matching full map, not merely three nonempty groups, proves coverage. The receipt describes logical persistent cache views; it does not prove scratch-buffer precision, tensor arithmetic precision, or unique/peak allocated bytes.

3. **Require dispatch where the route claim requires it.** `check_route():107–121` allows `dispatch=None`, including a Lumo route-shaped configuration. Keep this useful configuration comparison distinct from an observed executed path. The connected Lumo admission branch must require the source-bound completed dispatch for its exact probe; a successful parameter comparison alone cannot attest the custom verifier ran. No request or kernel-timing instrumentation was exercised in this review.

The compute field is correctly labeled `INSTANTIATED_MODEL_COMPUTE_POLICY`, read from `runner.model_config.dtype` (vLLM) or `runner.dtype` (SGLang), rather than advertised as a proof of all kernel arithmetic. KV/convolution/recurrent dtypes are read from live cache view objects and compared separately.

## SGLang source applicability and lifecycle

The retained `model_runner.py:807–828` assigns the result's request/KV pools and performs post-pool wiring before returning; `:833–835` places this before decode graph capture. `memory_pool.py:480–481,540–574,771–782` supports layer-axis-zero packed convolution/temporal state and the separate speculative scratch fields. `:3555–3644` supports the hybrid full-attention mapping and shared Mamba pool identity checked by the hook. `:1781,1917–1932` supports refusing `post_capture_active`: its buffer views may represent unbacked address ranges before later materialization. `:1851–1854,1901–1910` also shows allocation can include an upstream copy-kernel warmup; the metadata receipt is not a statement that no startup execution occurred.

The exact plain `MHATokenToKVPool` checks are conservative. The same retained source permits caller-selected page-major or unified backing, and the `model_runner.py:812–819` choice is delegated to the missing `sglang/srt/model_executor/pool_configurator.py` and its imports. `runner.ps` is source-supported at `model_runner.py:310`; exact `tp_rank/tp_size/pp_rank/pp_size` construction should be bound to `sglang/srt/distributed/parallel_state_wrapper.py`. The current fake draft-MHA positive test does not resolve actual EAGLE draft factory choice. Do not relax these checks merely to pass; inspect the source-selected class and metadata layout first.

An allocation receipt may exist before later graph/model startup fails. That accurately records a completed allocation, not a ready server. The actual successful boot, final owned worker population and probe-completed-before-baseline checks remain necessary. The hook adds no forward-path work, new warmup, tensor-value reads, CUDA API call, synchronization, or timing measurement.

## CPU evidence and exact source bindings

Command: configured Python with `-B`, executing snapshot `reviewer_cpu.py`. `test-log-attempt1.txt` retains **28 passing controls**: 10 delivered vLLM, 9 delivered SGLang, and 9 independent. Five independent controls deliberately reproduce the limitations above. The remaining controls exercise mismatch/partial-map refusals, draft rejection by target projection, and an AST-extracted copy of the pinned outer initialization method with fake dependencies; the latter verifies profiling suppression and immediate release of all fake tensor references. Subprocess and socket entry points were blocked during controls. No vLLM, SGLang or Torch module was imported.

| Reviewed file | SHA256 |
| --- | --- |
| `worker_metadata_v1.py` | `6c4625893a95e0545645a5d898ad7b9c75ef9c8d7c61dbe7abccfba0991704d7` |
| `runtime_projection_v1.py` | `34d9932b00b71fd80118a5fda90b1c689be8671564bda1e646d0db6e7eb974dd` |
| `sglang_worker_metadata_v1.py` | `f1f5a327c9c47ff1deaa9f1a71c264e94da1237d26040ec006b9ba35bd3da05c` |
| Pinned vLLM runner | `904d7d357ba076c93cefee5f7f030e2be577dc12599aebdf56395b894e2077b0` |
| Pinned GDN state-slot source | `c6c1a8ae5c9039f7a17c004761486013de58756a2988e25af78e4b6aa5316019` |
| Retained effective Eagle proposer | `8b22dd4fb15044813c8da28afa43b90f178709f7ed3111b3925b28abea717d62` |
| SGLang model runner | `d2924c79228e58bb917926063f81a9b16dfaafc177652f7862e8430cd3e54701` |
| SGLang memory pool | `25ff2309585909a2589f1bf7ca094453b13785c054ab30b416a76167fca1069f` |

`source-snapshot.json` and `source-reference-bindings.json` bind complete paths, sizes, other test/model-source hashes, and the preserved copies. The final review manifest binds this note, source copies, independent controls, and their output. Any repair is a later source version and is not implicitly approved by this review.
