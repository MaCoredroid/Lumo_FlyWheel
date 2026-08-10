#!/usr/bin/env bash
# Build the B1 GQA-pair FA2 candidate (sequences=1, 2 query heads per CTA).
#
# The recipe is the 2026-08-05 Codex build rollout as recovered for the B4
# GQA-pair ILKV rebuild; only the translation unit, the object/library names and
# the private launcher symbol differ. In particular the 55 reference objects are
# reused BYTE-FOR-BYTE from the qualified qrow16 build, so the stock kernels the
# byte A/B compares against stay bit-identical to the original build and any ABI
# or byte difference remains attributable to the GQA pairing rather than to this
# toolchain. Only flash_api.cpp (the dispatch shim) and the new TU are compiled.
#
# CPU only: no GPU is visible to any step, and every container asserts that.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

case "${FR13_BUILD_B1_GQA_PAIR:-0}" in
  1) ;;
  0)
    echo "B1 GQA-pair build is disabled; set FR13_BUILD_B1_GQA_PAIR=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_BUILD_B1_GQA_PAIR must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

# ------------------------------------------------------------------- pins
IMAGE='vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776'
FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
# Pinned by the codegen, ahead of the build: scripts/fr13_qrow32_b1_pass_sidecar.py
# GQA_PAIR_SOURCE_CLOSURE_SHA256.
SOURCE_CLOSURE_SHA256=172b5e7131841ce45650bb8eea35f0b427ca660ce8f145bd39b55b00a336ebf4

Q=${Q:-/home/mark/shared/lumoFlyWheel-qrow16-thin/output/fr13_fa2_qrow16_num_splits0_build_20260731}
FA2_ORIGIN=${FA2_ORIGIN:-$Q/vllm-source/build/lumo_cutlass_research/_deps/vllm-flash-attn-src}
CUTLASS=${CUTLASS:-$FA2_ORIGIN/csrc/cutlass}
REF=${REF:-$Q/vllm-source/build/lumo_cutlass_research/vllm-flash-attn}
BUILD=${BUILD:-/home/mark/fr13_fa2_qrow32_gqa_pair_b1_sm121a_$(date -u +%Y%m%d)}

SOURCE="$BUILD/source"
TU=flash_fwd_fr13_qrow32_gqa_pair_b1_hdim256_bf16_sm80.cu
OBJ=qrow32_gqa_pair_b1
SO="$BUILD/_vllm_fa2_qrow32_gqa_pair_b1_sm121a.abi3.so"
LAUNCHER_SYMBOL=fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1
PYTHON_BIN=${PYTHON_BIN:-$REPO/.venv/bin/python}

# -------------------------------------------------------------- preflight
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ ! -e "$BUILD" && ! -L "$BUILD" ]] \
  || { echo "BUILD must be new: $BUILD" >&2; exit 2; }
for required in "$FA2_ORIGIN" "$CUTLASS" "$REF"; do
  [[ -d "$required" && ! -L "$required" ]] \
    || { echo "build input missing or unsafe: $required" >&2; exit 2; }
done
unset required
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "$(git rev-parse '@{upstream}')" ]] \
  || { echo "source commit must be pushed before building" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "pinned image absent: $IMAGE" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the build" >&2; exit 2; }

SOURCE_COMMIT=$(git rev-parse HEAD)
mkdir -p "$BUILD"
# Host-side provenance. It is a separate file because compile_env.txt is written
# by the build container as root and is not appendable from the host.
printf 'repo_commit=%s\nsource_closure_sha256=%s\nrunner_sha256=%s\nstarted=%s\n' \
  "$SOURCE_COMMIT" "$SOURCE_CLOSURE_SHA256" \
  "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" \
  "$(date -u +%FT%TZ)" > "$BUILD/build_provenance.txt"

# ------------------------------------------- 1. regenerate the FA2 source
cp -a "$FA2_ORIGIN" "$SOURCE"
git -C "$SOURCE" checkout -- .
git -C "$SOURCE" clean -fdq
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$FA2_HEAD" ]] \
  || { echo "FA2 source head drifted" >&2; exit 2; }
[[ -z "$(git -C "$SOURCE" status --porcelain=v1 --untracked-files=all)" ]] \
  || { echo "FA2 source did not reset to pristine" >&2; exit 2; }

"$PYTHON_BIN" "$REPO/scripts/fr13_patch_fa2_tree_bias.py" \
  --fa2-src "$SOURCE" --skip-python \
  --tree-bias-tile-earlyout --fixed32-query-gqa-pair32-b1 \
  2>&1 | tee "$BUILD/patcher.log"

# Fail closed on the exact modified-source set, per-file hashes and closure
# digest the gate will later re-derive.
"$PYTHON_BIN" "$REPO/scripts/fr13_qrow32_b1_pass_sidecar.py" validate-source \
  --source-root "$SOURCE" --arm gqa_pair \
  | tee "$BUILD/source_closure.json"
[[ "$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["canonical_sha256"])' \
      "$BUILD/source_closure.json")" == "$SOURCE_CLOSURE_SHA256" ]] \
  || { echo "regenerated FA2 source closure is not the pinned closure" >&2; exit 3; }

DRUN=(docker run --rm --network none --cpus=1 --cpu-shares=2
  --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES= --env CUDA_CACHE_DISABLE=1
  --entrypoint /bin/bash)

FLAGS='
DEFINES="-DFLASHATTENTION_DISABLE_BACKWARD -DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_PYBIND -DPy_LIMITED_API=3 -DTORCH_EXTENSION_NAME=_vllm_fa2_C -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_NVSHMEM -DUSE_RPC -DUSE_TENSORPIPE -D_vllm_fa2_C_EXPORTS"
INCLUDES="-I/src/csrc -I/src/csrc/flash_attn -I/src/csrc/flash_attn/src -I/src/csrc/common -I/cutlass/include -isystem /usr/include/python3.12 -isystem /usr/local/lib/python3.12/dist-packages/torch/include -isystem /usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include"
CUDA_FLAGS="-DONNX_NAMESPACE=onnx_c2 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda -O3 -g -DNDEBUG -std=c++17 -Xcompiler=-fPIC -DENABLE_FP8 --use_fast_math -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1 -gencode arch=compute_121a,code=sm_121a"
if compgen -G "/dev/nvidia*" >/dev/null; then echo unexpected_gpu_device >&2; exit 91; fi
'

echo "== Step B: compile the B1 GQA-pair TU =="
"${DRUN[@]}" -v "$SOURCE:/src:ro" -v "$CUTLASS:/cutlass:ro" -v "$BUILD:/build" "$IMAGE" -lc "
set -euo pipefail
$FLAGS
printf 'image=%s\nsource_commit=%s\nnetwork=none\ncpus=1\nNVIDIA_VISIBLE_DEVICES=%s\nCUDA_VISIBLE_DEVICES=%s\nCUDA_CACHE_DISABLE=%s\n' \
  'sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776' \
  '$FA2_HEAD' \
  \"\$NVIDIA_VISIBLE_DEVICES\" \"\$CUDA_VISIBLE_DEVICES\" \"\$CUDA_CACHE_DISABLE\" > /build/compile_env.txt
nice -n 19 ionice -c 3 nvcc \$DEFINES \$INCLUDES \$CUDA_FLAGS -c \
  /src/csrc/flash_attn/src/$TU \
  -o /build/$OBJ.raw.sm121a.o 2>&1 | tee /build/compile.log >/dev/null
echo TU_COMPILED
"

echo "== Step C: compile flash_api.cpp (host compiler, -O2) =="
"${DRUN[@]}" -v "$SOURCE:/src:ro" -v "$CUTLASS:/cutlass:ro" -v "$BUILD:/build" "$IMAGE" -lc "
set -euo pipefail
$FLAGS
nice -n 19 ionice -c 3 /usr/bin/c++ \$DEFINES -I/usr/local/cuda/include -I/usr/local/cuda/include/cccl \$INCLUDES \
  -O2 -g -DNDEBUG -std=c++17 -fPIC -c /src/csrc/flash_attn/flash_api.cpp \
  -o /build/flash_api_pair_b1.o 2>&1 | tee /build/compile_flash_api.log >/dev/null
echo FLASH_API_COMPILED
"

echo "== pin the Thrust arch-mangled ABI namespace (host objcopy) =="
OLD='_ZN6thrust24THRUST_300001_SM_1210_NS6system6detail10sequential3seqE'
NEW='_ZN6thrust23THRUST_300001_SM_800_NS6system6detail10sequential3seqE'
objcopy --redefine-sym "$OLD=$NEW" \
  "$BUILD/$OBJ.raw.sm121a.o" "$BUILD/$OBJ.pinnedabi.sm121a.o"
# Compare the SYMBOL NAME field only. objcopy --redefine-sym renames symbols, not
# section names, so the inert .rodata.<mangled> section keeps the sm_121a string.
if nm -a "$BUILD/$OBJ.pinnedabi.sm121a.o" | awk -v s="$OLD" '$3 == s {found=1} END {exit !found}'; then
  echo old_thrust_namespace_survived >&2; exit 93; fi
if ! nm -a "$BUILD/$OBJ.pinnedabi.sm121a.o" | awk -v s="$NEW" '$3 == s {found=1} END {exit !found}'; then
  echo pinned_thrust_namespace_missing >&2; exit 93; fi
nm -a "$BUILD/$OBJ.pinnedabi.sm121a.o" | grep -F "$NEW" > "$BUILD/pinnedabi_thrust_symbol.txt"
echo ABI_PINNED

echo "== resource contract (1 cubin, STACK:0, no spills, no LDL/STL/CALL) =="
# Register count is RECORDED, not asserted: the B4 sibling's REG:243 was measured
# for a different translation unit and this one has never been compiled, so
# pinning it here would encode an unmeasured assumption. The 96 KiB dynamic
# shared allocation is already enforced at compile time by the TU's
# static_assert(smem_size == 96 * 1024).
"${DRUN[@]}" -v "$BUILD:/build" "$IMAGE" -lc "
set -euo pipefail
cuobjdump --list-elf /build/$OBJ.raw.sm121a.o | tee /build/cuobjdump_list_elf.txt
cuobjdump --dump-resource-usage /build/$OBJ.raw.sm121a.o | tee /build/cuobjdump_resource_usage.txt
"
[[ "$(grep -c 'ELF file' "$BUILD/cuobjdump_list_elf.txt")" -eq 1 ]] \
  || { echo "expected exactly one cubin" >&2; exit 92; }
grep -qE 'STACK:0' "$BUILD/cuobjdump_resource_usage.txt" \
  || { echo "kernel uses stack" >&2; exit 92; }
grep -qE 'REG:[0-9]+' "$BUILD/cuobjdump_resource_usage.txt" \
  || { echo "no register report" >&2; exit 92; }

"${DRUN[@]}" -v "$BUILD:/build" -v /usr/local/cuda/bin/nvdisasm:/usr/local/bin/nvdisasm:ro \
  --env NVDISASM_PATH=/usr/local/bin/nvdisasm "$IMAGE" -lc "
set -euo pipefail
cuobjdump --dump-sass /build/$OBJ.raw.sm121a.o > /build/$OBJ.sm121a.sass
if grep -E '[[:space:]](LDL|STL|CALL)(\.|[[:space:]])' /build/$OBJ.sm121a.sass > /build/forbidden_sass.txt; then
  cat /build/forbidden_sass.txt >&2; exit 92; else : > /build/forbidden_sass.txt; fi
echo SASS_CLEAN"

echo "== Link: flash_api_pair_b1.o, 55 sorted reference objects, pinned TU =="
"${DRUN[@]}" -v "$BUILD:/out" -v "$REF:/ref:ro" "$IMAGE" -lc "
set -euo pipefail
if compgen -G '/dev/nvidia*' >/dev/null; then echo unexpected_gpu_device >&2; exit 91; fi
mapfile -d '' objects < <(find /ref/CMakeFiles/_vllm_fa2_C.dir -type f -name '*.o' ! -path '*/flash_api.cpp.o' -print0 | sort -z)
printf 'reference_objects_without_flash_api=%s\n' \"\${#objects[@]}\" | tee /out/link_manifest.txt
test \"\${#objects[@]}\" -eq 55
nice -n 19 ionice -c 3 /usr/bin/c++ -fPIC -I/usr/local/cuda/include -I/usr/local/cuda/include/cccl -O2 -g -DNDEBUG -shared \
  -o /out/$(basename "$SO") \
  /out/flash_api_pair_b1.o \"\${objects[@]}\" /out/$OBJ.pinnedabi.sm121a.o \
  -L/usr/local/cuda/targets/sbsa-linux/lib/stubs -L/usr/local/cuda/targets/sbsa-linux/lib \
  -Wl,-rpath,/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64: \
  /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch.so \
  /usr/local/cuda/lib64/libcudart.so /usr/local/cuda/targets/sbsa-linux/lib/stubs/libcuda.so \
  -Wl,--no-as-needed,/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so -Wl,--as-needed \
  -Wl,--no-as-needed,/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so -Wl,--as-needed \
  /usr/local/lib/python3.12/dist-packages/torch/lib/libc10_cuda.so \
  /usr/local/lib/python3.12/dist-packages/torch/lib/libc10.so \
  /usr/local/cuda/lib64/libcudart.so -lcudadevrt -lcudart_static -lrt -lpthread -ldl \
  2>&1 | tee /out/link.log
echo LINKED"

echo "== ABI audit vs the qualified qrow16 reference =="
CAND="$SO"; REFSO="$REF/_vllm_fa2_C.abi3.so"
for kind in stock candidate; do
  if [ "$kind" = stock ]; then so="$REFSO"; else so="$CAND"; fi
  readelf -W --dyn-syms "$so" | awk '$7 != "UND" && ($5 == "GLOBAL" || $5 == "WEAK" || $5 == "UNIQUE") {print $4, $5, $6, $8}' | sort -u > "$BUILD/${kind}_defined_dynamic.txt"
  readelf -W --dyn-syms "$so" | awk '$7 == "UND" && $5 != "LOCAL" {print $4, $5, $6, $8}' | sort -u > "$BUILD/${kind}_undefined_dynamic.txt"
  readelf -W -d "$so" | awk '$2 == "(NEEDED)" {print $5}' | sort -u > "$BUILD/${kind}_dt_needed.txt"
  readelf -W -d "$so" | awk '$2 == "(RPATH)" || $2 == "(RUNPATH)" {print $2, $5}' | sort -u > "$BUILD/${kind}_runtime_path.txt"
done
unset kind so
for f in defined_dynamic undefined_dynamic dt_needed runtime_path; do
  diff -u "$BUILD/stock_$f.txt" "$BUILD/candidate_$f.txt" > "$BUILD/$f.diff" || true
  b=$(stat -c %s "$BUILD/$f.diff"); printf '%s.diff=%s bytes\n' "$f" "$b"
  test "$b" -eq 0 || { echo "ABI DRIFT in $f" >&2; exit 94; }
done
unset f b
# The private launcher must stay LOCAL: a dynamic export would let anything but
# the in-binary sentinel gate reach the candidate kernel.
if readelf -W --dyn-syms "$CAND" | c++filt | grep -qF "$LAUNCHER_SYMBOL"; then
  echo private_launcher_leaked >&2; exit 94; fi
readelf -Ws "$CAND" | c++filt | grep -F "$LAUNCHER_SYMBOL" \
  > "$BUILD/candidate_private_launcher_symbol.txt"
grep -q ' LOCAL ' "$BUILD/candidate_private_launcher_symbol.txt" \
  || { echo "private launcher is not LOCAL" >&2; exit 94; }
echo ABI_AUDIT_PASS

echo "== offline torch load (no GPU) =="
"${DRUN[@]}" -v "$BUILD:/build:ro" "$IMAGE" -lc "
python3 - <<'PY'
import json, torch
torch.ops.load_library('/build/$(basename "$SO")')
ns = torch.ops._vllm_fa2_C
ok = hasattr(ns, 'varlen_fwd') and hasattr(ns, 'varlen_fwd_tree_bias')
print(json.dumps({'library_loaded': True, 'registered_required_ops': bool(ok), 'torch': torch.__version__}, sort_keys=True))
raise SystemExit(0 if ok else 95)
PY" | tee "$BUILD/offline_torch_load.txt"

echo "== RESULT =="
sha256sum "$SO" | tee "$BUILD/candidate_so_sha256.txt"
stat -c '%s bytes' "$SO"
grep -E 'REG:|STACK:' "$BUILD/cuobjdump_resource_usage.txt" || true
echo BUILD_COMPLETE
echo
echo "NEXT: pin candidate_so_sha256 + size into"
echo "  scripts/fr13_qrow32_b1_pass_sidecar.py    GQA_PAIR_CANDIDATE_SHA256 / _SIZE"
echo "  scripts/fr13_fixed32_contract.py          QROW32_B1_GQA_PAIR_FA2_SHA256 / _SIZE"
echo "  scripts/fr13_patch_fa2_tree_bias.py       _FR13_FA2_QROW32_B1_GQA_PAIR_CANDIDATE_SHA256 / _SIZE"
echo "then add the gqa_pair arm case to scripts/fr13_run_b1_k64_qrow32_split2_live_gate.sh."
