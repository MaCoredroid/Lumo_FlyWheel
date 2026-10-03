# Four-arm worker source recipes v1.2: bounded review

**PASS for the prospective v1.2 source recipes; no blocking defect found.** This closes only the addition of Lumo's distinct source recipe and its compatibility with the existing metadata/request observers. All policies remain DRAFT. No runtime, model, GPU, workload or gate action is approved, and no scientific result is inferred. The CPU source-preparation operation/result is under separate review; this reviewer authenticated the retained source artifacts needed by the recipe without repeating that operational audit.

## Source and consumer checks

The independent snapshot contains 49 files, including the 34 Lumo source members, recipe/generator artifacts, selected accepted baseline sources and actual consumers. Every member of the Lumo manifest matches its retained bytes, and all review inputs still matched after the controls. The generator authenticates manifest `05e7c1c0e478813a20a466df2b5849a40604c314e4fe5bb1f9a5af4dca26998a`, checks the separate source-preparation receipt, and requires the prospective profile hash `d2f9e0b7aa22d563003b1295c011e6e74efdb3c07de4eae211cb84640ee91f86`. It does not use the AR/CHAIN source dictionary for Lumo.

The prior AR, CHAIN_MTP and SGLANG_EAGLE recipe objects are exactly unchanged after accounting for their shared layer-source digest now including Lumo provenance. Their accepted v1.1 logic is not reopened. Lumo's runner, GDN, graph wrapper and Eagle proposer hashes differ from AR/CHAIN as expected. Input processor, KV interface and parallel state legitimately retain identical bytes; this is explicit source equality, not accidental reuse of a different patched file. All module paths match their actual manifest paths, and worker/request source-binding dictionaries match the corresponding bootstrap module entries.

**Allocation metadata remains compatible.** In the new `v1/worker/gpu_model_runner.py`, `initialize_kv_cache` (10189–10246), `initialize_kv_cache_tensors` (10106–10159), and `_reshape_kv_cache_tensors` (9968–10070) are AST-identical to the accepted reference source. The outer initialization still defaults `is_profiling=False` and calls the named inner allocation routine. The resulting named AttentionSpec tensor and MambaSpec two-state views remain what `worker_metadata_v1_2.py:62–107, 139–205` observes. Patched GDN still consumes cache position 0 as convolution and position 1 as recurrent state (10814–10818 and 15890–15895). This establishes source compatibility, not live allocation coverage or device-state correctness.

**Target and draft names remain separate.** Lumo's `qwen3_5.py` target constructors are AST-identical to the accepted model constructors, yielding the same `language_model.model.layers.{i}` prefixes. Patched `qwen3_next.py:672–679` retains the `.self_attn.attn` suffix; the target therefore still has exactly 16 attention and 48 convolution/recurrent layer names. The recipe independently binds the patched Eagle source (`8b22dd4f…`), whose `_draft_attn_layer_names` remains the newly-added-layer set difference at lines 7551–7553. The metadata collector hashes the actual drafter class source and partitions those names away from target cache records. Lumo does not inherit CHAIN_MTP's different Eagle source hash.

**The request observer wraps the actual patched call.** The model-runner source replaces `GPUModelRunner.execute_model` with `_fr13_sg_execute_model_locked` at 11040; this wrapper is in the same bound source, accepts `self, *a, **k`, and forwards them to `_fr13_sg_orig_execute_model` (10946–10979). The existing observer binds the installed method, so it preserves that lock/sample-lifecycle wrapper rather than bypassing it. The original execute method still calls `self._update_states(scheduler_output)` at 5580, and the named request remains in `self.requests` after update (1229). The unchanged input processor is assigned the correct Lumo `external_id_unchanged` mode. Actual request-mode validation rejects the AR-style suffix mode for Lumo.

**Graph evidence fields match the patched source.** The patched `CUDAGraphWrapper.__call__` selects `concrete_cudagraph_entries` by `batch_descriptor`, stores `_fr13_fixed32_graph_signature` at capture completion (343–353), replays `entry.cudagraph` and returns `entry.output` (404–421). Patched GDN retains `_FR13_FIXED32_CAPTURE_MANIFESTS` and stores `(signature, canonical)` at 5501; its manifest schema, batch/row fields and descriptor agree with the observer at `request_worker_observer_v1.py:186–222`. The observer records completed host dispatch only, preserving its explicit lack of device-completion or timing inference. This source review does not establish that any such replay occurred.

## Controls and disposition

The reviewer completed 136 bounded CPU/source assertions, including hash checks; these are not experiments. Isolated generation reproduced both v1.2 artifacts byte-for-byte. Controls substituted actual AR runner, GDN, graph and Eagle bytes at the corresponding Lumo source reads, preserving the expected Lumo hashes: all four refused before outputs. Manifest byte drift also refused. The actual observer method binder accepted the real outer-wrapper definition compiled alone with its source filename, rejected an AR source hash on that Lumo file, and rejected a foreign source path. The wrapper body was never executed. Draft observer policy refusal and the correct request-ID mode were checked using the actual unchanged observer module. No Torch/model import, runtime activation, SSH, Docker or GPU action occurred.

The source layer is ready to be incorporated into a separately reviewed parent freeze. Actual boot must still reproduce the prospective-profile source hashes and establish owned worker allocation, route and request evidence. No launch authority or qualification follows from this PASS.

| Artifact / binding | SHA-256 |
| --- | --- |
| `build_worker_source_recipes_v1_2.py` | `766330757f49ddf2881a53268f7fa0839c40fde4b8b941c77f7f03882fc26326` |
| `RECIPES.DRAFT.json` | `18b78e397391a6e82c9842614d96a79dde6d77bcef200497cefa7c6c3bd5a7d1` |
| `LAYER-MAP-SOURCE.json` | `76198a59eb06bc5ad8ca049575c2cc48f6137f8570706f17951e0424a4b4241f` |
| `MANIFEST.json` | `05e7c1c0e478813a20a466df2b5849a40604c314e4fe5bb1f9a5af4dca26998a` |
| Lumo `cuda_graph` | `c6fab35aeafd5b3ccba4320cfdcbb10a0829e8b46cccb039736c16e9546ddf61` |
| Lumo `gdn_state_slots` | `23df7748f02a742753e586487e3e905e0ffb23815a9c5aaf87f5cf2c4a2ff91a` |
| Lumo `input_processor` | `17c091fef7bbaa4537335367abf8267345a782914f00fe01919c6efbbeff9c3b` |
| Lumo `kv_cache_interface` | `0db7ff3efa74e29a53a4a921d4ac4446fef3999f78747186796ee68f58cc8a79` |
| Lumo `model_runner` | `3d9df101ebb1b2d5e8bc0ca844451b47a935f4adf48f7c6ab4ef81438ce21b79` |
| Lumo `parallel_state` | `7a6a6ea7cc7cd8aa4353cbab84fad365db6f9ea1fbd1db593b7b023a7ff04a76` |
| Lumo `request_observer` | `f4c37b47e39c5a3c250fe3b668204d10a2847e73e92a9a7fca714bbcccd99986` |
| Lumo `worker_metadata` | `bf08b96a9f11adb38513f1d4376a6b32d438c464f9f073ae03a73a11eaa43875` |
| Lumo draft Eagle | `8b22dd4fb15044813c8da28afa43b90f178709f7ed3111b3925b28abea717d62` |

Reviewer artifacts: `p0/monitor/review-response-20260927/worker-source-recipes-v12-source-review/{SNAPSHOT.json,CONTROLS.json,REVIEW-SEAL.json}`. Reproducer: `p0/monitor/review-response-20260927/worker-source-recipes-v12-review-controls.py`. The seal binds the note, both ledgers and reproducer. The earlier three-arm review remains `notes/review-response-20260927/worker-source-recipes-source-review.md`.
