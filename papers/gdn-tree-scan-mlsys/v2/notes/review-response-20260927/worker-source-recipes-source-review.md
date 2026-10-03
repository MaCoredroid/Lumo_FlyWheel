# Worker source recipes: independent source review

**Disposition: v1.1 passes the bounded prospective source review.** The original v1 contained one SGLang architecture-provenance gap; the versioned v1.1 closes it. No incorrect current module paths or target layer names were found. All three recipes remain DRAFT, and LUMOTREE remains pending its own distinct post-patch sources. This review grants no runtime, GPU, workload or qualification authority.

The reviewed generator path is `workload-plan/tools/attempt-runtime/`, rather than the campaign-level `tools/attempt-runtime/`. Sources were copied into a hash-bound reviewer snapshot before controls, and compared again afterward. No original or successor file was edited by the reviewer.

## Original finding and closure

**F1 — v1's SGLang map was not derived from the source actually selecting layer kinds.** `build_worker_source_recipes_v1.py:69–85` called its rule “by layer_types” and constructed SGLang IDs directly from the checkpoint `text_config.layer_types`. The retained SGLang model instead selects `config.layers_block_type[idx]` (`models/qwen3_5.py:1507–1516`), and the retained pool uses an explicit `full_attention_layer_id_mapping` (`mem_cache/memory_pool.py:3667–3669`). The two originally cited source-extraction receipts did not contain the config implementation connecting those fields. This was a missing derivation, not evidence that this checkpoint's expected 16 attention / 48 recurrent layers were wrong.

The parent separately extracted `configs/qwen3_5.py` and its `configs/qwen3_next.py` base from the same immutable SGLang image. The retained receipts report never-started containers, no GPU request and exact cleanup. The independent reviewer authenticated both local source files against those receipts:

- `qwen3_5_config.py`: `53f92b2be9d880716a1850cb4484ea2fa46408ab0b05be847ce0fa88edf71890`.
- `qwen3_next_config.py`: `071ce509469c3d3320a7c7dcdb58702a53045fa9856d1c858501e2740b9c5df2`.

`Qwen3_5TextConfig` inherits `Qwen3NextConfig` (lines 15–29). The base's `layers_block_type`, `linear_layer_ids` and `full_attention_layer_ids` properties at lines 260–284 derive the pattern from `num_hidden_layers` and `full_attention_interval`. The pinned checkpoint explicitly has 64 layers and interval 4. Therefore the current map is exactly attention indices 3, 7, …, 63 and all remaining indices recurrent/convolution.

The v1.1 generator at lines 70–107 pins both new source hashes and extraction receipts, verifies the subclass relation, executes only the exact-source enum/getter AST, and requires its resulting layer kinds to equal the pinned checkpoint list. It uses that derived list for SGLang at line 114 and records the derivation in the layer-source artifact. This closes F1. Independent injected CPU controls changed the interval to 3, swapped an attention/recurrent position while preserving counts, and removed the interval; each refused before creating output artifacts. These semantic controls intentionally bypassed only the initial config-byte hash to reach the derivation; an ordinary changed-config digest was separately rejected by the actual `read` guard. The implementation's normal hash checks were not changed.

## Accepted bindings and consumers

**Exact-image source identities.** All 23 retained vLLM source files match the pinned post-patch `MANIFEST.json` (`183597730e406d18c44e786021bbeebb54506ade33d822e835a98108b19bb9e7`) and its retained RESULT receipt. Its recorded image is `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`; source preparation, not a model boot or qualification, is recorded. All nine SGLang source entries from the two original extraction receipts also match their bytes and paths, with image `sha256:0076dffa60b76b7bf033c04d05e0cc69d46f2b8cd60aa2468827782afe9bc38f`. Every engine module selected in the recipes matches an actual manifest/receipt absolute path and SHA, and each `/opt/lumotree-observer/code/` helper matches the actual local helper bytes. The source extraction operations themselves were parent-performed; this review only read their retained artifacts.

**vLLM names are correct for this checkpoint.** The pinned checkpoint architecture is `Qwen3_5ForConditionalGeneration`. Actual retained `model_executor/models/qwen3_5.py:617–619` creates `language_model`; lines 483–485 add `.model`; lines 229–237 construct `.layers.{i}`; decoder lines 136–149 add `.linear_attn` or `.self_attn`. Inherited `qwen3_next.py:254–261` adds `.attn` to full-attention modules. Thus AR and CHAIN_MTP correctly declare 48 names ending `.linear_attn` for both recurrent and convolution, and 16 ending `.self_attn.attn`. Although `vllm_naming_source` names `qwen3_next.py`, the containing source-manifest binding also authenticates the necessary `qwen3_5.py` prefix/override construction. No layer-name change is required.

**Target and draft are separated.** AR has no draft-source hash; CHAIN_MTP binds `v1/spec_decode/eagle.py` (`2c569fd2b60e81b897360071fe433b259575c74b4c7a2a7832560abe2c7cc4b3`). The actual runner selects `EagleProposer` on its eagle-capable route at lines 560–561. The retained proposer computes `_draft_attn_layer_names` as newly added attention layers at lines 1287–1305. `worker_metadata_v1_2.py:74–105` hashes the actual drafter class source, requires a complete unique draft-name subset of caches, and partitions those names away from target metadata. The target expectation remains the same independently declared 64-layer map. This is a prospective source binding, not an observation that the intended MTP class or its allocations ran.

SGLang correctly uses decimal layer IDs, matching `sglang_worker_metadata_v1_1.py:49–65, 79–93`. Its target record describes packed Mamba state with explicit layer indices. Draft records remain separate and must identify their role; shared target Mamba state is recorded as a link rather than counted as a draft allocation. Bootstrap selects the matching v1.1 common metadata helper for SGLang and v1.2 for vLLM. Worker/request source-binding dictionaries are exact subsets of the bootstrap module dictionary, and the producer hashes match those selected helpers.

**Dynamic fields are supplied by the real consumer.** `probe_boundary_v1.py:55–67` obtains the pre-agent request's seed expectation, inserts attempt/boot identity, supplies request/metadata/installation paths, and sets `/opt/lumotree-observer/empty-pycache`. Recipes correctly omit these dynamic fields. `worker_bootstrap_v2.py:44–57, 84–100` validates actual imported source paths/hashes and source-loader/cache state. Its engine-specific imports and producer selection at 144–169 agree with the recipes. No false installed-package path or absent recipe key was found.

**DRAFT stays blocked.** `PB.policies` marks the instantiated bootstrap object FROZEN only as part of consuming a parent-frozen outer spec, but deliberately preserves the nested request and worker policy statuses. The actual `Observer.__init__` at line 80 rejects the DRAFT request policy before using an observation directory or installing the final finders. `runtime_projection_v1_2.py:83` rejects the DRAFT worker policy. Independent controls instantiated all three via the actual `PB.policies` AST and confirmed both statuses remain DRAFT; actual observer and projection validators refused them. Parent freeze must explicitly review/freeze the nested policies; the generator does not silently promote them. No runtime activation was attempted here.

## Reproducibility and remaining scope

The original controls authenticated and snapshotted 50 files and completed 150 checks, including source hashes, module/producer matching, layer populations, draft refusal and isolated byte-identical generation. The successor added seven snapshot files and 28 focused checks, including exact-image config provenance and mismatch refusals. Counts include hash checks, not just independent semantic cases. Neither control script imported Torch or activated a runtime.

The v1.1 recipe differs from v1 only in generator identity and the expanded layer-source digest; module paths, module hashes, current target maps, request modes, draft binding and all policy statuses remain unchanged. Both artifact pairs regenerate byte-for-byte in reviewer-owned temporary output directories. LUMOTREE cannot inherit either AR or CHAIN_MTP bindings; its separate preparation remains pending as explicitly represented.

| Artifact | SHA-256 |
| --- | --- |
| v1 `RECIPES.DRAFT.json` | `bfd46e9c09ad5988a63b1ab1af82ab36df11dd6bf3cde4f2eff1201be0c0ca01` |
| v1 `LAYER-MAP-SOURCE.json` | `31109e326e1ab0384d03dfb2f6591e9a97fd466369c0e1c2102230b42354b95b` |
| v1.1 `RECIPES.DRAFT.json` | `78aa08adb45ae2b5db848f6cd059bee374b287930dd068f7818039a3e3160ce5` |
| v1.1 `LAYER-MAP-SOURCE.json` | `d36975899c22adf31849d3b3dbf5819729701d74f1b86b4ae679b2ef1e633563` |
| `build_worker_source_recipes_v1.py` | `1b01a33971d941ed1ce521c692caf1729e6fae397fbaf7294a11e89d3f809258` |
| `build_worker_source_recipes_v1_1.py` | `86611ed46e2f36d1527cc3a70a5045ee52936190c0cca125964f17192e8a34e8` |

Reviewer artifacts are under `p0/monitor/review-response-20260927/worker-source-recipes-source-review/` (`SNAPSHOT.json`, `CONTROLS.json`, and `successor-v1.1/`). Reproducers are `p0/monitor/review-response-20260927/worker-source-recipes-review-controls.py` and `worker-source-recipes-v11-review-controls.py`. Their exact hashes, both snapshots and this note are recorded in `REVIEW-SEAL.json`.

No further source correction is required for these three v1.1 recipes within this review. Actual post-patch source equality, allocation coverage, request routing, owned process evidence and parent gate approval still must be established by the future frozen runtime workflow; no experiment or qualification count advances from this note.
