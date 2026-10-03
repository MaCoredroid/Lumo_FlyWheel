# Worker partition repair and SGLang factory follow-up

2026-09-28. Bounded local source/CPU review only. No installation, container/remote operation, engine import, model request, GPU work or workload/evaluator execution. WP remains closed; workload **0/4**. Parent separately owns native M1 work.

## vLLM F1 repair closed

`worker_metadata_v1.py` SHA256 `97587d1be791c6621afd834a9014d57a3f2f14167e976fbde195268146492dbb` now partitions the shared cache map using actual `runner.drafter._draft_attn_layer_names` and validates the drafter class-source hash. `allocated_tensors` contains target views; `draft_allocated_tensors` separately retains draft views and their map-source binding. AR emits an empty draft map. This closes the concrete mislabeling counterexample in `worker-metadata-projection-bounded-review.md`; it does not supply the still-required independent target architecture map or authenticated connected projection.

The pinned runner selects `EagleProposer` at `gpu_model_runner.py:560–561`. The retained effective Lumo proposer derives the map at `eagle.patched.py:7536–7554` and binds its layers to shared groups at `:7889–7932`. Its class and base reside in that same source file. Preserve a separate source binding for the stock CHAIN proposer; the Lumo patched-file hash cannot stand in for it.

**25 passing CPU controls** are retained: 11 delivered vLLM, 9 unchanged SGLang, and 5 independent. New controls verify exact union/disjointness of target/draft rows, missing/wrong drafter source refusal, missing/duplicate/unallocated draft-name refusal, the AR empty partition, and the pinned SGLang hybrid constructor's target/draft layer mappings using an AST-extracted method with fake factories. No vLLM/SGLang/Torch import or allocation occurs; subprocess/network entry points are blocked. Previous profiling, failure, lifetime, and persistence controls still pass.

Snapshot: `p0/monitor/review-response-20260927/worker-partition-factory-reviewed-20260928T212236Z/`. Test output: `test-log-attempt1.txt`. Original reviewed snapshot and note remain unchanged.

## SGLang: source facts now resolved

The extraction receipt is `workload-plan/inspections/codex-sglang-source-20260928T210817Z/RECEIPT.json`. All three file sizes and SHA256 hashes were independently checked. It records image `sha256:0076dffa60b76b7bf033c04d05e0cc69d46f2b8cd60aa2468827782afe9bc38f`, owned container `a44648493b93833312fbdfae144817440c60bc16d570bec569061e6c2cf374e5`, never started/no GPU request, and removal of that exact CID. This review did not repeat those operations.

| Source | SHA256 |
| --- | --- |
| `srt/model_executor/pool_configurator.py` | `afb83cba19b4d11c49c2886c71f1796730c7598f72dc3f2d9bf32314b04a992d` |
| `srt/mem_cache/kv_cache_configurator.py` | `68abcdbf4fde6bf6fe7fa0e512a7934f2f2c57ca65e4415b6005c6c710997fa7` |
| `srt/distributed/parallel_state_wrapper.py` | `d057a8cfac66219345bd83fd752207094f7cdae42a803e14598e060f8ee0c3c1` |

- `ParallelState` explicitly defines integer `tp_rank`, `tp_size`, `pp_rank`, `pp_size`; the hook's four rank attributes are correct for this source. Their live values are still to be observed.
- `KVCacheConfigurator._init_pools():375–408` takes the unified target-pool path only when `enable_unified_memory`, null disaggregation and no preexisting request pool hold. `:416–438` otherwise shares the target request pool with the draft; its new-Mamba clone exception is specifically Inkling MTP, not Qwen.
- `_build_token_to_kv_pool():884–887` chooses page-major versus plain MHA backing using `enable_page_major_kv_layout`. `:954–976` selects hybrid-linear whenever `mambaish_config` is truthy, else the plain MHA builder. The declared four-arm SGLang argv does not enable unified/page-major layout, and retained `server_args.py:933–950` defaults both false. BF16 KV bypasses FP4/MXFP8 branches. This is a declared branch expectation, not an observation of live objects.
- `_build_hybrid_linear_kv_pool():1390–1434` constructs `HybridLinearKVPool`; target full-attention IDs come from the hybrid model's IDs within the PP layer range, while a draft takes **`[0]`**. It passes through `req_to_token_pool.mamba_pool` and the chosen MHA class. This means an actual hybrid draft would share the target Mamba pool; it must not be presented as independently allocated draft recurrent state. The current observer safely refuses hybrid draft layout.
- `_build_mha_kv_pool():1455–1487` constructs plain MHA backing when selected and BF16 is used. That is compatible with the current observer's draft branch. `model_config.py:675–683` changes the Qwen draft architecture to `Qwen3_5ForCausalLMMTP` with one next-N layer.

## One remaining predicate needed for a conclusive draft-factory verdict

`KVCacheConfigurator.__post_init__():233–234` delegates to `mambaish_config(self.model_config)` / `hybrid_gdn_config(...)`, imported from **`/sgl-workspace/sglang/python/sglang/srt/configs/hybrid_arch.py`**. That file was not among the extracted three or the earlier retained copies found in this bounded review. Whether `Qwen3_5ForCausalLMMTP` is excluded from or retained by that predicate decides the actual draft branch. The fake draft-MHA tests cannot settle it. Parent has been given this exact missing path; do not infer its result from the display name or remove the conservative class check.

Post-capture applicability is likewise a declared/runtime boundary: the factory passes `self.post_capture_kv_active` into the backing and the observer requires it false. `model_runner.py:594–596` obtains it through `is_post_capture_kv_active(...)`; retained `server_args.py:4839–4888` shows related planning constraints, including environment and graph configuration. No live value has been observed. An active post-capture pool must continue to refuse until there is a source-bound final-materialization observation, rather than reading reserved-view metadata as materialized serving storage.

The source choice above does not reopen the previous review's metadata-only safety assessment, or approve installation/live admission. Existing worker source/role/PID, independent target coverage, route dispatch and successful boot/probe/baseline bindings remain outstanding integration work owned by parent.
