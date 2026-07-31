# Qrow16 post-Hydra build and byte gate

Status: source-audited and ready to compile after the active Hydra campaign
releases the machine. This procedure does not authorize deployment or timing.

## Pinned inputs

- FA2 upstream: `29210221863736a08f71a866459e368ad1ac4a95`
- Exact-safe base SO: `f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`,
  299,183,936 bytes
- Exact-safe launch header: `d9e9f4b92cb731d7955b514449e59b8e411bf7a0c929aafb454f2402d41fe976`
- Qrow16 launch header after idempotent source application:
  `40a8f29600aa0c237404ca208a78c5ea3e1ccf10c61046b80fbc4852d9dbd225`
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

REPO=/home/mark/shared/lumoFlyWheel-fa2-qrow16
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
  40a8f29600aa0c237404ca208a78c5ea3e1ccf10c61046b80fbc4852d9dbd225
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

## Real same-boot byte A/B

The capture must come from one real SWE-Verified fixed32 B1 event on the pinned
exact-safe SO. It must store compacted paged `key_cache`/`value_cache` (with its
`block_table` remapped), `query`, `seq_lens`, `max_seq_len`, fp32 `tree_bias`,
`scale`, optional `softcap`, and provenance. Provenance must name the
SWE-Verified instance, concurrency 1, physical node count 32, and the pinned
source SO hash. This is a correctness gate, not a performance probe.

The gate calls the candidate at B1, then duplicates that exact request to B2
and B4. `params.b != 1` forces both duplicate calls through the stock geometry
inside the same loaded binary and CUDA process. The first request's output and
fp32 LSE, plus all duplicate replicas, must be byte-identical:

```bash
set -euo pipefail

REPO=/home/mark/shared/lumoFlyWheel-fa2-qrow16
OUT=$REPO/output/fr13_fa2_qrow16_build
CAND_SO=$OUT/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so
CAPTURE=/absolute/path/to/real_swe_verified_fixed32_b1_paged_capture.pt
RESULT=$OUT/qrow16_same_boot_byte_ab.json
IMAGE=sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc

docker run --rm --gpus all \
  -v "$REPO:/workspace:ro" \
  -v "$CAND_SO:/tmp/qrow16.so:ro" \
  -v "$CAPTURE:/tmp/capture.pt:ro" \
  -v "$OUT:/results" \
  --entrypoint bash "$IMAGE" -lc \
  'set -euo pipefail; cp /tmp/qrow16.so /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so; python3 /workspace/scripts/fr13_patch_fa2_tree_bias.py --skip-source; python3 /workspace/scripts/fr13_fa2_qrow16_byte_ab.py --capture /tmp/capture.pt --out /results/qrow16_same_boot_byte_ab.json'

python3 - "$RESULT" <<'PY'
import json, pathlib, sys
row = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert row["passed"] is True, row
assert all(v["output_byte_equal"] and v["lse_byte_equal"] and v["stock_replicas_byte_equal"] for v in row["comparisons"].values())
PY
```

Only after the byte gate passes may the candidate run the canonical real
SWE-Verified exact4 B1 campaign, followed by B4 and exact16 acceptance.
