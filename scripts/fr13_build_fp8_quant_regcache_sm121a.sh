#!/usr/bin/env bash
# Offline compiler-only build of the pinned SM121a stable-libtorch extension.
set -euo pipefail

: "${VLLM_ROOT:?set VLLM_ROOT to patched vLLM fe9c3d6c5 source}"
: "${BUILD_ROOT:?set BUILD_ROOT to a new writable build directory}"

VLLM_ROOT=$(realpath "$VLLM_ROOT")
BUILD_ROOT=$(realpath -m "$BUILD_ROOT")
TARGET_SOURCE="$VLLM_ROOT/csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu"
OUTPUT="$BUILD_ROOT/_C_stable_libtorch.fp8_quant_regcache.sm121a.abi3.so"

[[ -d "$VLLM_ROOT" && -f "$TARGET_SOURCE" && ! -L "$TARGET_SOURCE" ]] \
  || { echo "patched vLLM source is unavailable" >&2; exit 2; }
[[ "$(git -C "$VLLM_ROOT" rev-parse HEAD)" == "fe9c3d6c5f66c873d196800384ed6880687b9e52" ]] \
  || { echo "vLLM source commit drifted" >&2; exit 2; }
grep -Fq 'FR13_FIXED32_B1_FP8_QUANT_REGCACHE: candidate kernel' "$TARGET_SOURCE" \
  || { echo "FP8 quant regcache patch is absent" >&2; exit 2; }
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] \
  || { echo "refusing to replace build output: $OUTPUT" >&2; exit 2; }

mkdir -p "$BUILD_ROOT/objects" "$BUILD_ROOT/cuda-cache"
export CUDA_CACHE_PATH="$BUILD_ROOT/cuda-cache"

COMMON_DEFS=(
  -DPy_LIMITED_API=3
  -DTORCH_EXTENSION_NAME=_C_stable_libtorch
  -DTORCH_TARGET_VERSION=0x020A000000000000ULL
  -DUSE_C10D_GLOO
  -DUSE_C10D_NCCL
  -DUSE_CUDA
  -DUSE_DISTRIBUTED
  -DUSE_NVSHMEM
  -DUSE_RPC
  -DUSE_TENSORPIPE
  -D_C_stable_libtorch_EXPORTS
)
COMMON_INCLUDES=(
  -I"$VLLM_ROOT/csrc"
  -isystem /usr/include/python3.12
  -isystem /usr/local/lib/python3.12/dist-packages/torch/include
  -isystem /usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include
  -isystem /usr/local/cuda/include
)
CUDA_FLAGS=(
  -forward-unknown-to-host-compiler
  -DONNX_NAMESPACE=onnx_c2
  -Xcudafe
  --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl
  --expt-relaxed-constexpr
  --expt-extended-lambda
  -O2
  -g
  -DNDEBUG
  -std=c++17
  -Xcompiler=-fPIC
  -DENABLE_FP8
  --compress-mode=size
  -gencode
  arch=compute_121a,code=sm_121a
)

/usr/bin/c++ "${COMMON_DEFS[@]}" "${COMMON_INCLUDES[@]}" \
  -O2 -g -DNDEBUG -std=gnu++17 -fPIC \
  -c "$VLLM_ROOT/csrc/libtorch_stable/torch_bindings.cpp" \
  -o "$BUILD_ROOT/objects/torch_bindings.cpp.o"
/usr/local/cuda/bin/nvcc "${COMMON_DEFS[@]}" "${COMMON_INCLUDES[@]}" \
  "${CUDA_FLAGS[@]}" \
  -c "$VLLM_ROOT/csrc/libtorch_stable/permute_cols.cu" \
  -o "$BUILD_ROOT/objects/permute_cols.cu.o"
/usr/local/cuda/bin/nvcc "${COMMON_DEFS[@]}" "${COMMON_INCLUDES[@]}" \
  "${CUDA_FLAGS[@]}" -c "$TARGET_SOURCE" \
  -o "$BUILD_ROOT/objects/per_token_group_quant_fp8.cu.o"
/usr/local/cuda/bin/nvcc "${COMMON_DEFS[@]}" "${COMMON_INCLUDES[@]}" \
  "${CUDA_FLAGS[@]}" \
  -c "$VLLM_ROOT/csrc/libtorch_stable/quantization/w8a8/int8/per_token_group_quant.cu" \
  -o "$BUILD_ROOT/objects/per_token_group_quant_int8.cu.o"

/usr/bin/c++ -fPIC -O2 -g -DNDEBUG -shared -o "$OUTPUT" \
  "$BUILD_ROOT/objects/torch_bindings.cpp.o" \
  "$BUILD_ROOT/objects/permute_cols.cu.o" \
  "$BUILD_ROOT/objects/per_token_group_quant_fp8.cu.o" \
  "$BUILD_ROOT/objects/per_token_group_quant_int8.cu.o" \
  -Wl,-rpath,/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64: \
  -L/usr/local/cuda/targets/sbsa-linux/lib/stubs \
  -L/usr/local/cuda/targets/sbsa-linux/lib \
  /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch.so \
  /usr/local/cuda/lib64/libcudart.so \
  /usr/lib/aarch64-linux-gnu/libcuda.so \
  -Wl,--no-as-needed,/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so \
  -Wl,--as-needed \
  -Wl,--no-as-needed,/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so \
  -Wl,--as-needed \
  /usr/local/lib/python3.12/dist-packages/torch/lib/libc10_cuda.so \
  /usr/local/lib/python3.12/dist-packages/torch/lib/libc10.so \
  /usr/local/cuda/lib64/libcudart.so \
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl
chmod 0555 "$OUTPUT"
sha256sum "$OUTPUT"
