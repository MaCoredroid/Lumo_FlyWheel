# Full-model reference: pinned FA2 API clarification

Read-only source check, 2026-09-27 local time. This closes the specific FA2 API uncertainty in `q1-fullmodel-v2-preliminary-repair-review.md`; it does not approve a launch or review later implementation changes.

**The preliminary hook calls the wrong module.** Its `_fa2_facts` asks `vllm.vllm_flash_attn.flash_attn_interface` for `get_flash_attn_version` (`q1_reference_hooks_v2.py:158-162`, preliminary SHA `0319ed7e82004f8030736be76e0578c85aa1c91423fb568fbde00c614c35c374`). The exact extracted interface contains zero occurrences of that name and no dynamic module `__getattr__`. Its `DEFAULT_FA_VERSION=2` is a function default, not the selector. Therefore that hook records `fa_version_selected=None` and its lines 182-183 reject every admitted case, even when the correct fork is installed.

The supported selector is:

```python
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
get_flash_attn_version(requires_alibi=False, head_size=256)
```

The actual per-layer dispatch field is more authoritative than a fresh global selector call. For every one of the 16 active attention layers, require and record:

```python
backend = layer.get_attn_backend()  # returns layer.attn_backend
impl = layer.impl
assert backend.get_name() == "FLASH_ATTN"
assert type(impl).__module__ == "vllm.v1.attention.backends.flash_attn"
assert type(impl).__name__ == "FlashAttentionImpl"
assert type(impl.vllm_flash_attn_version) is int
assert impl.vllm_flash_attn_version == 2
```

Record `impl.num_heads`, `impl.num_kv_heads`, `impl.head_size`, `impl.attn_type` and the backend/implementation source identities as well. Retain installed/loaded fork SHA and interface SHA checks. The interface imports `_vllm_fa2_C`; its loaded extension's `__file__` provides a concrete binary path to attest. Do not substitute `DEFAULT_FA_VERSION`, a loaded fork symbol, or a global selector return for these per-layer checks.

## Exact source chain

The files were read via SSH from the existing extracted source directory, without importing vLLM or running Docker/model/GPU commands:

`/home/mark/lumotree-review-20260927/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/identity/native_source/`

| Extracted file | SHA-256 | Relevant lines |
| --- | --- | --- |
| `vllm__vllm_flash_attn__flash_attn_interface.py` | `9caca9584061cfd60c9ebd404d9a8881a62fe7894242efefcba5b88fc086d5b6` | 13-17 load FA2; 49 default constant; 204 explicit `fa_version` parameter; 285-325 FA2 branch calls `torch.ops._vllm_fa2_C.varlen_fwd`. No version-selector getter. |
| `vllm__v1__attention__backends__fa_utils.py` | `379bdccb69048f7bf6746816d146b383adceadef5e03bcd1ab1aa216ebbdfb27` | 56-58 selector signature; 74-83 architecture default; 88-93 current-config override; 129-141 head-size fallback; 143-153 support validation/return. |
| `vllm__v1__attention__backends__flash_attn.py` | `c949e39a0b0b3dfa316017c422b1cb1368cd3a4a1328a1c691f5d7f44ed571e7` | 103-104 backend name; 126-127 implementation class; 134-164 KV layout; 607-633 geometry and stored selected version; 793-815 direct varlen call passes the stored field at 809; cascade also passes it at 837. |
| `vllm__model_executor__layers__attention__attention.py` | `1e5894a5ddc0dcbd343ba4b3e4c6d674d87604232dabede2aed1383a5ac4e713` | 299-310 backend selection; 344-358 `self.impl` and backend enum; 583-584 backend accessor; 768-780 actual attention operator calls `self.impl.forward`. |

The selector's source default for compute-capability major 12 is FA2, subject to config override and support checks; this is a source-level inference, not a live observation from this review. The explicit override field is `vllm_config.attention_config.flash_attn_version`. Setting it to 2 through the engine's supported configuration makes intent explicit, while the per-layer check verifies resolution. No environment-variable spelling or CLI syntax is inferred here.

The pinned backend confirms the logical KV shape used by the repair: `(2, num_blocks, block_size, num_kv_heads, head_size)`. NHD/HND change storage stride order, not those logical axes; the already reviewed stock runner restores the backend's logical view after allocation. Preserve the group-specific block mapping and valid-tail extraction.

The repo's minimal `_patch_flash_attn_interface` only modifies dispatch to select `varlen_fwd_tree_bias` when a tree bias is supplied and otherwise retain `varlen_fwd`; it does not add the missing getter. Keep that minimal installation and the ordinary native causal geometry. No numerical policy, experiment scope, worker source or gate was changed by this check. Re-review the repaired implementation only when its final smoke snapshot is frozen.
