# FR13 fixed32 FA2 suffix-only candidate

Status: CPU-only production-toolchain build complete. The build ran only after
the active exact4 campaign released the measurement slot. No GPU test or
real-task run has been performed. See `build_manifest.json` for the result.

## Provenance

- Application source commit: `fc855e594d5246a2a66a6a3615b289b7a95f54c8`
- Base artifact: `53256ac84d2a7e22c037ebac3261e0f01f50cce0`
- Provenance-only commit: `c8a908596795bd3330a28a62a3dbcade02ad364e`
- Pinned FA2 upstream: `29210221863736a08f71a866459e368ad1ac4a95`
- Production build image: `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`
- Image repo digest: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`

The rejected split-K candidate changed floating reduction association. This
candidate contains no split-K selection, output-stride mutation, or API-path
change. It adds a CTA-uniform return only when a key tile cannot overlap the
tree-bias suffix. A returned tile could not have loaded bias or mutated a score
in the prior helper. Overlap tiles execute the prior helper body byte for byte.

This return skips only the scalar tree-bias walk after QK. It does not skip K/V
loads, QK, softmax, or PV.

## Source audit

A clean source-only application to the pinned FA2 upstream produced:

| File | SHA256 | Relation to verified split4 build |
| --- | --- | --- |
| `flash_api.cpp` | `20c6c6e22121c1c621c569ef02fac0fe1a736ff131d469a4d6de21ab5b2dde71` | different; stock split selection restored |
| `flash_api_torch_lib.cpp` | `c575d9f02ba44bf7022c77b80fdf12173da0ecae8a4d7599934c2cc9fa52e121` | identical |
| `src/flash.h` | `e4c7875a72c0bc5f8ed3e0661ef956ca24b38c8f4758ae2a89f5e58b88671c5a` | identical |
| `src/flash_fwd_kernel.h` | `934e8c6c2e72c667f3cb0a8dc53b11c16a4eba8e3ac2b5811c882eff399ac3de` | identical |

The prior production split4 build has 55 target objects: 52 CUDA forward
objects and three C++ objects. Because the generated kernel header is identical,
all 52 CUDA objects remain valid. `flash_api_sparse.cpp.o` and
`flash_api_torch_lib.cpp.o` also remain valid. Only `flash_api.cpp.o` must be
forcibly rebuilt, then the shared object relinked. The configured graph is:

- `build.no-reconfigure.ninja`: `e01ef0b3551e9dc46fdc54d4f241241f9ef5332cc81b911bcbfc23c9ea1c9121`
- `CMakeFiles/rules.ninja`: `48682694f3e80b4bd4fee4e60fc81ecfb26c69b1c401f3fa33875f453a95c488`

## Fixed32 work arithmetic

For each request with context `C`, fixed32 uses 32 query rows, a 32x32 bias,
24 query heads, 16 target full-attention layers, and N=64 key tiles. The number
of early-return tiles is `floor(C / 64)` per query-head-layer CTA. Tail6 and
Hydra23 use the same physical 32-row geometry.

At the 14,568-token fixture, 227 of 229 target key tiles return early:

| Route | B1 early/total walks | B4 early/total walks |
| --- | ---: | ---: |
| Target verifier | 87,168 / 87,936 | 348,672 / 351,744 |
| Four MTP drafter calls | 21,792 / 21,888 | 87,168 / 87,552 |
| Combined | 108,960 / 109,824 | 435,840 / 439,296 |

## Released build procedure

Do not run this block until the active real-task campaign releases CPU and
container work. It hardlink-clones the prior verified build, then breaks every
path that Ninja can mutate. The prior split binary and canonical baseline remain
immutable comparison artifacts.

```bash
set -euo pipefail

REPO=/home/mark/shared/lumoFlyWheel-fa2-suffix-only
SPLIT_ROOT=/home/mark/shared/lumoFlyWheel-fa2-split4/output/fr13_fa2_split4_50acc5b6_build/vllm-source
OUT=$REPO/output/fr13_fa2_suffix_fc855e59_build
CAND_ROOT=$OUT/vllm-source
CLEAN_FA2=$OUT/fa2-source-clean
SPLIT_FA2=$SPLIT_ROOT/build/lumo_cutlass_research/_deps/vllm-flash-attn-src
CAND_BUILD=$CAND_ROOT/build/lumo_cutlass_research
CAND_FA2=$CAND_BUILD/_deps/vllm-flash-attn-src
IMAGE=sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc

test ! -e "$OUT"
mkdir -p "$OUT"
git -C "$SPLIT_FA2" worktree add --detach "$CLEAN_FA2" \
  29210221863736a08f71a866459e368ad1ac4a95
/home/mark/shared/lumoFlyWheel-publish-20260730/.venv/bin/python \
  "$REPO/scripts/fr13_patch_fa2_tree_bias.py" \
  --fa2-src "$CLEAN_FA2" --skip-python --tree-bias-tile-earlyout

cp -al "$SPLIT_ROOT" "$CAND_ROOT"
cp -a --remove-destination \
  "$CLEAN_FA2/csrc/flash_attn/flash_api.cpp" \
  "$CAND_FA2/csrc/flash_attn/flash_api.cpp"
cp -a --remove-destination \
  "$SPLIT_ROOT/build/lumo_cutlass_research/.ninja_log" \
  "$CAND_BUILD/.ninja_log"
cp -a --remove-destination \
  "$SPLIT_ROOT/build/lumo_cutlass_research/.ninja_deps" \
  "$CAND_BUILD/.ninja_deps"

API_OBJ=$CAND_BUILD/vllm-flash-attn/CMakeFiles/_vllm_fa2_C.dir/csrc/flash_attn/flash_api.cpp.o
CAND_SO=$CAND_BUILD/vllm-flash-attn/_vllm_fa2_C.abi3.so
rm -f "$API_OBJ" "$CAND_SO"

test "$(sha256sum "$CAND_FA2/csrc/flash_attn/flash_api.cpp" | cut -d' ' -f1)" = \
  20c6c6e22121c1c621c569ef02fac0fe1a736ff131d469a4d6de21ab5b2dde71
test "$(sha256sum "$CAND_FA2/csrc/flash_attn/src/flash_fwd_kernel.h" | cut -d' ' -f1)" = \
  934e8c6c2e72c667f3cb0a8dc53b11c16a4eba8e3ac2b5811c882eff399ac3de
! rg -n 'FR13_FA2_TREE_SPLITKV|fr13_tree_splitkv|o_batch_stride = max_seqlen_q' \
  "$CAND_FA2/csrc/flash_attn/flash_api.cpp"

docker run --rm \
  -v "$CAND_ROOT:/opt/vllm-source" \
  -w /opt/vllm-source \
  --entrypoint bash "$IMAGE" -lc \
  'set -euo pipefail; ln -s /usr/bin/env /usr/local/bin/ccache; ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/aarch64-linux-gnu/libcuda.so; ninja -C build/lumo_cutlass_research -f build.no-reconfigure.ninja -n _vllm_fa2_C'
```

The dry run must list exactly two steps: rebuild `flash_api.cpp.o`, then link
`_vllm_fa2_C.abi3.so`. Any CUDA compilation or other C++ compilation means a
source/hash/mtime invariant drifted and the incremental build must stop.

After that gate, replace `-n` in the final container command with `-j8` to
perform the two-step build. No `--gpus` flag is needed.

## Post-link gates

1. Record candidate SHA256, size, ELF header, `.nv_fatbin` sections, and build
   image/toolchain versions.
2. Recheck the split4 binary remains exactly
   `744819aaa230b3c9f2610f27c9d8c603461917163d364d99d65ffdcd158bea8d`
   at 299,181,128 bytes.
3. Recheck both canonical baseline copies remain exactly
   `97fa2519739b3f976debb8377f8829cf3a167b410d1770bb42db390f8c5c0ae1`
   at 301,219,928 bytes.
4. Compare candidate versus baseline and split4 with `nm -D`: strong exports,
   all defined exports, undefined symbols, and dynamic dependencies must have
   zero set delta. Required symbols include `PyInit__vllm_fa2_C`,
   `mha_varlen_fwd_tree_bias`, and `set_params_tree_bias`.
5. Load the candidate with `torch.ops.load_library` in the pinned build image
   without a GPU and assert the `varlen_fwd_tree_bias` schema is registered.
6. Hash every reused object before and after link. All 54 reused object hashes
   must be unchanged; `flash_api.cpp.o` must be new.
7. Only after these gates may the binary be bound into an isolated application
   branch for CUDA byte-equivalence and canonical real SWE-Verified timing.
