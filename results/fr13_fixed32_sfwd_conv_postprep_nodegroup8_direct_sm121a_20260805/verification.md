# Verification

Source branch: `codex/sfwd-direct-nodegroups8-next-20260805`

Source commit: `b73d78f681d0cea8487b97a75eaf2ac44d3bc8ec`

## CPU/source tests

```text
CUDA_VISIBLE_DEVICES='' pytest -q \
  tests/test_fr13_fixed32_sfwd_conv_postprep_nodegroup8_direct.py \
  tests/test_fr13_fixed32_sfwd_conv_postprep_fusion.py \
  tests/test_fr13_fixed32_sfwd_conv_postprep_wiring.py \
  tests/test_fr13_fixed32_sfwd_prior_reuse_descriptorless.py
64 passed in 1.34s
```

The new tests cover B1/B4 CPU byte equality against the serial exact-product
oracle, topology-derived source rows, all 32 output and stage writers, group-0
edge ownership, standalone/embedded program geometry, generator idempotence,
selector default-off behavior, and the incumbent kernel function SHA-256.

A broader related run produced 71 passes and one unrelated failure. The failing
test pins a historical SHA-256 for `fr10_gdn_tree_kernel.py`; that file already
differs on base commit `2c2802374` and this branch does not modify it.

## Offline compile

```text
CUDA_VISIBLE_DEVICES='' \
TRITON_CACHE_DIR=/home/mark/scratch/fr13_sfwd_nodegroup8_direct_codegen_cache_20260805 \
/home/mark/shared/fr13_toolchain_torch211_cu130/bin/python \
  offline_codegen_audit.py --repo <repo> --output <artifact>
```

The audit compiled incumbent and nodegroup8 functions from the exact source
commit for B1/B4 standalone and embedded profiles. It performed no device API
call and makes no timing or acceptance claim. The audit checked
`MemAvailable >= 20 GiB` before every build; all resource results have zero
stack, local, and shared bytes and zero calls.
