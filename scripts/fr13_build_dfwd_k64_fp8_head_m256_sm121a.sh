#!/usr/bin/env bash
# Offline compiler-only build of the exact K64 block-FP8 M256 head candidate.
set -euo pipefail

: "${VLLM_ROOT:?set VLLM_ROOT to a disposable vLLM fe9c3d6c5 worktree}"
: "${CUTLASS_ROOT:?set CUTLASS_ROOT to pinned CUTLASS source}"
: "${BASE_OBJECT_ROOT:?set BASE_OBJECT_ROOT to the pinned full-extension objects}"
: "${TORCH_ROOT:?set TORCH_ROOT to the pinned torch package root}"
: "${PYTHON_BIN:?set PYTHON_BIN to the pinned torch Python}"
: "${BUILD_ROOT:?set BUILD_ROOT to a new output directory}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
VLLM_ROOT=$(realpath "$VLLM_ROOT")
CUTLASS_ROOT=$(realpath "$CUTLASS_ROOT")
BASE_OBJECT_ROOT=$(realpath "$BASE_OBJECT_ROOT")
TORCH_ROOT=$(realpath "$TORCH_ROOT")
PYTHON_BIN=$(realpath "$PYTHON_BIN")
BUILD_ROOT=$(realpath -m "$BUILD_ROOT")

VLLM_COMMIT=fe9c3d6c5f66c873d196800384ed6880687b9e52
CUTLASS_COMMIT=da5e086dab31d63815acafdac9a9c5893b1c69e2
BLOCKWISE_SOURCE="$VLLM_ROOT/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu"
BLOCKWISE_DISPATCH="${BLOCKWISE_SOURCE%.cu}_dispatch.cuh"
PATCHER="$REPO/scripts/fr13_patch_cutlass_fixed32_wave.py"
OUTPUT="$BUILD_ROOT/_C_stable_libtorch.k64_head_m256.sm121a.abi3.so"
OBJECT="$BUILD_ROOT/scaled_mm_blockwise_sm120_fp8.k64_head_m256.cu.o"

[[ ! -e "$BUILD_ROOT" && ! -L "$BUILD_ROOT" ]] \
  || { echo "BUILD_ROOT must be a new path" >&2; exit 2; }
[[ -d "$VLLM_ROOT/.git" || -f "$VLLM_ROOT/.git" ]] \
  || { echo "VLLM_ROOT must be a git worktree" >&2; exit 2; }
[[ "$(git -C "$VLLM_ROOT" rev-parse HEAD)" == "$VLLM_COMMIT" ]] \
  || { echo "vLLM source commit drifted" >&2; exit 2; }
[[ -d "$CUTLASS_ROOT/.git" || -f "$CUTLASS_ROOT/.git" ]] \
  || { echo "CUTLASS_ROOT must be a git worktree" >&2; exit 2; }
[[ "$(git -C "$CUTLASS_ROOT" rev-parse HEAD)" == "$CUTLASS_COMMIT" ]] \
  || { echo "CUTLASS source commit drifted" >&2; exit 2; }
[[ -f "$BLOCKWISE_SOURCE" && ! -L "$BLOCKWISE_SOURCE" \
   && "$(sha256sum "$BLOCKWISE_SOURCE" | awk '{print $1}')" \
      == "194d4b5f529dfb690eeb6d864919ae7f9b859097568a513b3f3cf78051a93499" ]] \
  || { echo "vLLM blockwise source drifted" >&2; exit 2; }
[[ -f "$PATCHER" && ! -L "$PATCHER" && -x "$PYTHON_BIN" ]] \
  || { echo "patcher or pinned Python is unavailable" >&2; exit 2; }
[[ -d "$TORCH_ROOT/lib" && -d "$TORCH_ROOT/include" ]] \
  || { echo "pinned torch package is unavailable" >&2; exit 2; }

"$PYTHON_BIN" "$PATCHER" "$VLLM_ROOT" --cutlass-root "$CUTLASS_ROOT"
[[ "$(sha256sum "$BLOCKWISE_DISPATCH" | awk '{print $1}')" \
    == "c8efe4917d5f207e23ca3c5b44feb6a2e704594dd35bc0a9d212a5fa431105ea" ]] \
  || { echo "patched K64 M256 dispatch drifted" >&2; exit 2; }

# Every object except the blockwise SM120 translation unit is authenticated and
# reused from the full pinned build. This keeps the experiment to one kernel TU.
PINNED_BASE_OBJECTS=(
  "f0b65f50166e45a1edf80185659ff4179f7c24ffa853b1e62ea73838ca44985a csrc/libtorch_stable/torch_bindings.cpp.o"
  "ab50346f5cffe17f9cbadc29e6035ba00baed40a0a40f71ce4206de4f5058730 csrc/cutlass_extensions/common.cpp.o"
  "b47e569e3ab7064309400e034794ff80cd8801e8723e944ead5e75cf66e45b40 csrc/cuda_utils_kernels.cu.o"
  "dc3f21a73d38b673fbfec495e15007121b283a9726bb568621633f7c3c3c3104 csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu.o"
  "b671249d9b57eecf7c57e31cdba8a95d31de07de70ca475e50241b974c82b4c6 csrc/libtorch_stable/quantization/fp4/nvfp4_quant_entry.cu.o"
  "55e451b8d64f68e0c92399ae18ced8e41dc8634f2399ceae49559fda67675bc0 csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_entry.cu.o"
  "1e9dc95d90d2b6df17a469c584ae6e6175c4fee9fa2393f420a6f560ba4580a4 csrc/libtorch_stable/permute_cols.cu.o"
  "01a847bc48df3f0f4cbf7cdc698c7edb1b29140220ec82c2fa67bf43c8883c8d csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu.o"
  "798592154beeb96dc819a11a4605a91d403ae2e4981f6595e9dd56ff9371f348 csrc/libtorch_stable/quantization/w8a8/int8/per_token_group_quant.cu.o"
  "693cb73e8ef0e0d74393cd1823326df3b376d667f941d98b820d19e5380f8074 csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu.o"
  "f3a6f10348ded463c76ed09b7106f52bd453e1530d56242fba71079e7e78c623 csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu.o"
  "29ace1d8d26f686d5d6247f910e88f98bcf96dc97ad7ef35567f32005a939c7e csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x.cu.o"
  "5400ff26cf31732889f02d2bb41b3c391d3012bbb7b980f4b5543aba20889177 csrc/libtorch_stable/quantization/w8a8/cutlass/moe/moe_data.cu.o"
  "db08aeda1a6d5dc01d1fedab904496c8eecd5bbe5538a389f6d7b9c0bd36ae48 csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu.o"
  "ef5bb81f73bb93a47b5cc3ad3f61d5612778206550485ba10298aa49a7e1866f csrc/libtorch_stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu.o"
  "71680701fb7d08b749efa5b49c1c19908ba4eba358afed13f555060a7e70b6d0 csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu.o"
  "5e2e0edb0e16a5f6a6700d9253b951e84b9318bdc3ec31ab69cbec9b0545540f csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu.o"
  "2590d8af40568b5e18336a5d707132470d4798248eecdecb2e829340b9a6edf0 csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu.o"
)
for entry in "${PINNED_BASE_OBJECTS[@]}"; do
  expected=${entry%% *}
  relative=${entry#* }
  object="$BASE_OBJECT_ROOT/$relative"
  [[ -f "$object" && ! -L "$object" \
     && "$(sha256sum "$object" | awk '{print $1}')" == "$expected" ]] \
    || { echo "base object drifted: $relative" >&2; exit 2; }
done

mkdir -p "$BUILD_ROOT/cuda-cache"
export CUDA_CACHE_PATH="$BUILD_ROOT/cuda-cache"
COMMON_DEFS=(
  -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1
  -DPy_LIMITED_API=3
  -DTORCH_EXTENSION_NAME=_C_stable_libtorch
  -DTORCH_TARGET_VERSION=0x020A000000000000ULL
  -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_CUDA -DUSE_DISTRIBUTED
  -DUSE_NVSHMEM -DUSE_RPC -DUSE_TENSORPIPE -D_C_stable_libtorch_EXPORTS
)
COMMON_INCLUDES=(
  -I"$VLLM_ROOT/csrc"
  -I"$CUTLASS_ROOT/include"
  -I"$CUTLASS_ROOT/tools/util/include"
  -isystem /usr/include/python3.12
  -isystem "$TORCH_ROOT/include"
  -isystem "$TORCH_ROOT/include/torch/csrc/api/include"
  -isystem /usr/local/cuda/include
)
CUDA_FLAGS=(
  -forward-unknown-to-host-compiler
  -DONNX_NAMESPACE=onnx_c2
  -Xcudafe
  --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl
  --expt-relaxed-constexpr --expt-extended-lambda
  -O2 -g -DNDEBUG -std=c++17 -Xcompiler=-fPIC
  -DENABLE_FP8 --threads=1 --compress-mode=size
  -DENABLE_CUTLASS_MLA=1 -DENABLE_SCALED_MM_SM120=1
  -DENABLE_SCALED_MM_C2X=1 -DENABLE_NVFP4_SM120=1
  -DENABLE_CUTLASS_MOE_SM120=1
  -gencode arch=compute_121a,code=sm_121a
  --resource-usage
)

/usr/bin/time -v /usr/local/cuda/bin/nvcc \
  "${COMMON_DEFS[@]}" "${COMMON_INCLUDES[@]}" "${CUDA_FLAGS[@]}" \
  -c "$BLOCKWISE_SOURCE" -o "$OBJECT" 2>&1 | tee "$BUILD_ROOT/compile.log"

BASE="$BASE_OBJECT_ROOT"
TORCH_LIB="$TORCH_ROOT/lib"
/usr/bin/c++ -fPIC -I/usr/local/cuda/include -I/usr/local/cuda/include/cccl \
  -O2 -g -DNDEBUG -shared -o "$OUTPUT" \
  "$BASE/csrc/libtorch_stable/torch_bindings.cpp.o" \
  "$BASE/csrc/cutlass_extensions/common.cpp.o" \
  "$BASE/csrc/cuda_utils_kernels.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/fp4/nvfp4_quant_entry.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_entry.cu.o" \
  "$BASE/csrc/libtorch_stable/permute_cols.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/w8a8/int8/per_token_group_quant.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu.o" \
  "$OBJECT" \
  "$BASE/csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/w8a8/cutlass/moe/moe_data.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu.o" \
  "$BASE/csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu.o" \
  -L/usr/local/cuda/targets/sbsa-linux/lib/stubs \
  -L/usr/local/cuda/targets/sbsa-linux/lib \
  -Wl,-rpath,"$TORCH_LIB":/usr/local/cuda/lib64: \
  "$TORCH_LIB/libtorch.so" /usr/local/cuda/lib64/libcudart.so \
  /usr/lib/aarch64-linux-gnu/libcuda.so \
  -Wl,--no-as-needed,"$TORCH_LIB/libtorch_cpu.so" -Wl,--as-needed \
  -Wl,--no-as-needed,"$TORCH_LIB/libtorch_cuda.so" -Wl,--as-needed \
  "$TORCH_LIB/libc10_cuda.so" "$TORCH_LIB/libc10.so" \
  /usr/local/cuda/lib64/libcudart.so \
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl

chmod 0555 "$OUTPUT"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" - "$OUTPUT" <<'PY'
import sys
import torch

torch.ops.load_library(sys.argv[1])
assert not torch.cuda.is_initialized()
assert hasattr(torch.ops._C, "cutlass_scaled_mm")
assert hasattr(torch.ops._C, "per_token_group_fp8_quant")
print("load_library=PASS cuda_initialized=False")
PY

/usr/local/cuda/bin/cuobjdump --dump-resource-usage "$OUTPUT" \
  > "$BUILD_ROOT/resource_usage.txt"
sha256sum "$OBJECT" "$OUTPUT"
