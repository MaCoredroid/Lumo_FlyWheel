# Fixed32 SFWD row-group-8 K64/root1 gate handoff

Status: **source, K64/root1 gate wiring, provenance checks, and focused tests
pass; the real B1 byte gate has not run from this branch**.

The candidate fuses exact fixed32 selective-forward convolution with both
state-motion directions. It owns eight physical tree rows per program, halves
the row-group-4 CTA count, and preserves the incumbent's ordered BF16 product
and FP32 accumulation contract. The real gate runs the candidate only in
shadow and always serves incumbent bytes.

## Bound source and workload

- Source and gate commit:
  `0a73d82ab9fbf6d5d5828e7b4f14b4689c4a7b64`
- Kernel source SHA-256:
  `6b1087f091e27f22a1fe1f033538dcaf5aa626962d9b9d285774d0677ab49f67`
- Gate runner SHA-256:
  `57dfaa7c0d91c90bd4324db76cb70d8233c19503814b545b62aa92e49f1f0458`
- Launcher SHA-256:
  `09e972776dec5c32819f5fd9efcc9941feaa3287b1a9f0ae1d722ea76836aacb`
- SWE-Verified subset SHA-256:
  `cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb`
- K64 block-map SHA-256:
  `85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff`
- Workload: Hydra27 fixed32, physical rows 32, B1/concurrency 1,
  draft-vocabulary K=65536, root reduction enabled
- Mandatory weight bytes: `32,666,638,208`
- Weight-only hardware floor: `119.658015414 ms/step`
- One-sided U95 cap at 1.15x: `137.6067177261 ms/step`

The gate runner checks the source-owned block map before launch. The kernel
re-hashes the mounted map before emitting a live PASS, and the reducer requires
the same K64/root1 fields and SHA. It also requires exactly one gather-shim
engagement, exactly one root-gather engagement, no draft-vocabulary disable or
full-head fallback, and byte-identical launch/end runtime and external
manifests. A full-vocabulary, contiguous-subset, fallback, or mid-run drifted
arm cannot satisfy this handoff's PASS contract.

## Offline kernel evidence

The prior codegen artifact is
`../fr13_fixed32_sfwd_state_fusion_rowgroup8_codegen_20260802/`. At the exact
deployed B1/B4 specialization, row-group 8 uses 160/640 CTAs versus 320/1280
for row-group 4. It compiles with 88 registers, 1,024 ELF shared bytes, zero
stack or local memory, zero spills, and no device calls. Its larger static body
and register footprint make this a candidate requiring a live comparison, not
a latency result.

## Verification completed

```text
git diff --check: pass
bash -n gate runner and launcher: pass
python -m py_compile kernel and focused test: pass
focused pytest: 31 passed in 1.01s
```

No synthetic workload, probe task, timing sample, acceptance sample, or full
TPS measurement was used to prepare this handoff.

## Real B1 byte gate

Run only after the current GPU owner has ended and teardown is independently
clean:

```bash
cd /home/mark/lumoFlyWheel-sfwd-state-fusion-rowgroup8
SOURCE_COMMIT=0a73d82ab9fbf6d5d5828e7b4f14b4689c4a7b64
git merge-base --is-ancestor "$SOURCE_COMMIT" HEAD
test "$(sha256sum scripts/fr13_run_b1_sfwd_state_fusion_gate.sh | awk '{print $1}')" = 57dfaa7c0d91c90bd4324db76cb70d8233c19503814b545b62aa92e49f1f0458
test "$(sha256sum src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py | awk '{print $1}')" = 6b1087f091e27f22a1fe1f033538dcaf5aa626962d9b9d285774d0677ab49f67
FA2=/home/mark/lumoFlyWheel-b1-wide256-k64-root-profile/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so
test "$(sha256sum "$FA2" | awk '{print $1}')" = f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TAG="sfwd_rowgroup8_k64_root_${STAMP}"
RUNROOT="output/fr13_b1_sfwd_rowgroup8_k64_root_${STAMP}"
RUNROOT="$RUNROOT" TAG="$TAG" FORKED_FA2_SO="$FA2" \
  bash scripts/fr13_run_b1_sfwd_state_fusion_gate.sh
```

The required result is one authenticated real SWE-Verified B1 task, all 48
layers present, and exact bytes for both convolution output and the complete
commit-source stage. A PASS remains a one-task correctness diagnostic. It does
not authorize production and cannot support acceptance, timing, TPS, or
hardware-floor claims. Those require the standing exact4 or exact16 real-task
campaign after explicit production qualification.
