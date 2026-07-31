# Qrow16 post-Hydra build and byte gate

Status: source-audited and ready to compile after the active Hydra campaign
releases the machine. This procedure does not authorize deployment or timing.

## Pinned inputs

- FA2 upstream: `29210221863736a08f71a866459e368ad1ac4a95`
- Exact-safe base SO: `f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`,
  299,183,936 bytes
- Exact-safe launch header: `d9e9f4b92cb731d7955b514449e59b8e411bf7a0c929aafb454f2402d41fe976`
- Qrow16 launch header after idempotent source application:
  `88bfcc5b1c4bbe9b95e8747b0efd58f0938b67ebfcf64f7c7a517489f09961e2`
- Unchanged suffix-early-out kernel header:
  `934e8c6c2e72c667f3cb0a8dc53b11c16a4eba8e3ac2b5811c882eff399ac3de`
- Production image:
  `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`

The qrow patch must be applied with `--tree-bias-tile-earlyout` as well as
`--fixed32-query-tile16`. Omitting the first flag intentionally restores the
non-early-out helper and would no longer be an A/B against the exact-safe base.

## Production build

Run only after Hydra and any other measurement process has exited:

```bash
set -euo pipefail

REPO=/home/mark/shared/lumoFlyWheel-b1-gate-bundle
BASE_ROOT=/home/mark/shared/lumoFlyWheel-fa2-suffix-only/output/fr13_fa2_suffix_fc855e59_build/vllm-source
OUT=$REPO/output/fr13_fa2_qrow16_build
CAND_ROOT=$OUT/vllm-source
CAND_BUILD=$CAND_ROOT/build/lumo_cutlass_research
CAND_FA2=$CAND_BUILD/_deps/vllm-flash-attn-src
CAND_SO=$CAND_BUILD/vllm-flash-attn/_vllm_fa2_C.abi3.so
IMAGE=sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc

test ! -e "$OUT"
test "$(sha256sum "$BASE_ROOT/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so" | cut -d' ' -f1)" = \
  f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
mkdir -p "$OUT"
cp -a --reflink=auto "$BASE_ROOT" "$CAND_ROOT"

python3 "$REPO/scripts/fr13_patch_fa2_tree_bias.py" \
  --fa2-src "$CAND_FA2" --skip-python \
  --tree-bias-tile-earlyout --fixed32-query-tile16
test "$(sha256sum "$CAND_FA2/csrc/flash_attn/src/flash_fwd_launch_template.h" | cut -d' ' -f1)" = \
  88bfcc5b1c4bbe9b95e8747b0efd58f0938b67ebfcf64f7c7a517489f09961e2
test "$(sha256sum "$CAND_FA2/csrc/flash_attn/src/flash_fwd_kernel.h" | cut -d' ' -f1)" = \
  934e8c6c2e72c667f3cb0a8dc53b11c16a4eba8e3ac2b5811c882eff399ac3de
rm -f "$CAND_SO"

docker run --rm \
  -v "$CAND_ROOT:/opt/vllm-source" \
  -w /opt/vllm-source \
  --entrypoint bash "$IMAGE" -lc \
  'set -euo pipefail; ln -s /usr/bin/env /usr/local/bin/ccache; ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/aarch64-linux-gnu/libcuda.so; ninja -C build/lumo_cutlass_research -f build.no-reconfigure.ninja -n _vllm_fa2_C' \
  | tee "$OUT/ninja_dry_run.txt"

test "$(rg -c 'Building CUDA object' "$OUT/ninja_dry_run.txt")" -eq 48
test "$(rg -c 'Linking CXX shared library.*_vllm_fa2_C' "$OUT/ninja_dry_run.txt")" -eq 1

docker run --rm \
  -v "$CAND_ROOT:/opt/vllm-source" \
  -w /opt/vllm-source \
  --entrypoint bash "$IMAGE" -lc \
  'set -euo pipefail; ln -s /usr/bin/env /usr/local/bin/ccache; ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/aarch64-linux-gnu/libcuda.so; ninja -C build/lumo_cutlass_research -f build.no-reconfigure.ninja -j8 _vllm_fa2_C'

sha256sum "$CAND_SO" | tee "$OUT/candidate_so.sha256"
stat -c '%s' "$CAND_SO" | tee "$OUT/candidate_so.size"
nm -aC "$CAND_SO" > "$OUT/nm_all_demangled.txt"
readelf -d "$CAND_SO" > "$OUT/readelf_dynamic.txt"
cuobjdump --dump-elf-symbols "$CAND_SO" > "$OUT/cuobjdump_elf_symbols.txt"
```

Before GPU use, require zero candidate-vs-exact-safe deltas in defined dynamic
exports, undefined dynamic symbols, and `DT_NEEDED`. The CUDA symbol dump must
contain the `256x16x64, 1 warp, Split=false` main kernel and no qrow16 combine
specialization. The final Ninja dry run must report no work.

## Real same-boot live-paged A/B

Run the bundle launcher with the exact candidate SO hash and exactly one real
SWE-Verified task at B1. Do not use a capture file, dense/repacked KV, a probe,
or a second process:

```bash
set -euo pipefail

export FORKED_FA2_SO=/absolute/path/to/qrow16/_vllm_fa2_C.abi3.so
export FR13_FA2_QROW16_SO_SHA256=$(sha256sum "$FORKED_FA2_SO" | cut -d' ' -f1)
export FR13_FA2_QROW16_LIVE_PAGED_AB=1
export FR13_FA2_QROW16_LIVE_PAGED_AB_INSTANCE_ID=astropy__astropy-12907

# Launch the fixed32 B1 server, then send only that normal SWE-Verified task.
# Require /logs/fr13_fa2_qrow16_live_paged_ab.json to report PASS and zero
# output/LSE raw-byte mismatches.
```

The stock FULL graph produces and serves the request output. Immediately after
its first real observed replay, the same EngineCore process recalls stock and
qrow16 using retained live paged operands and compares BF16 output plus FP32
LSE raw bytes. The candidate dispatch throws on any non-production geometry,
so a stock-vs-stock false pass is not possible.

`scripts/fr13_fa2_qrow16_byte_ab.py` is compile preflight only. Only after the
live-paged gate passes may the candidate run canonical real SWE-Verified exact4
B1, followed by B4 and exact16 acceptance.
