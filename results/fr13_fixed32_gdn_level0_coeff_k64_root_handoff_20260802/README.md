# Fixed32 GDN level-0 coefficient staging: K64/root1 B1 handoff

Status: source/static audit complete; real GPU byte gate not run here.

This handoff reuses the reviewed zero-spill SM121 build at
`../fr13_fixed32_gdn_level0_coeff_sm121_build_review_20260801/`. It does not
retune the drafter or vocabulary: the live gate is pinned to root 1, K=65536,
the existing block map, physical rows 32, BV=8, and Hydra27. Stock is always
served and the candidate runs only as a shadow comparator.

The one-task run is a real SWE-Verified B1 correctness diagnostic. It is not
acceptance, floor, throughput, or latency evidence. A PASS only authorizes the
next exact4 timing step; it does not enable production by default.

## Identity and geometry

- Candidate source commit: `f5ccbdfdd1b7244cb551bca69d1ff099b9ab2c70`
- Kernel source SHA-256: `16fde18ebf4ace9893d2f8890294c894c71222b85d7c9cdc4bc7789cf5afff4e`
- Offline build manifest SHA-256: `f6e4b5553aec439535a0a36b3b2ec45b33c41c5c7aa2a7cbe968d8ec34aea0e7`
- B1 candidate level-0 CUBIN SHA-256: `599bb5b1a61193922979f06696c3c62bcd1166de310c6399a0549958aa1940b1`
- B1 candidate level-1 CUBIN SHA-256: `4f4144e674712d064790796b3398d8b191d351e257d90abcf9596f9f3a7657b5`
- Source shape: N=32, KH=16, VH=48, K=128, V=128, BV=8, warps=8
- Path decomposition: one level-0 path with max length 5, then eleven level-1
  paths with max length 7
- Launches: two per layer and 96 per 48-layer event
- Production selector: default off; the diagnostic selector is exclusive and
  source-bound

The CUBINs are offline codegen evidence generated from the exact source hash.
Triton JITs the live kernel from that source; these CUBIN files are not loaded
by the live gate. The live PASS records the runtime source hash and the host
validator independently checks it. Therefore this handoff does not overstate
the offline CUBIN hashes as live installed-binary attestation.

## Static instruction inventory

Counting every addressed SASS instruction in each complete function and
weighting one level-0 path plus eleven level-1 paths gives:

| Batch | Stock L0 | Stock L1 | Candidate L0 | Candidate L1 | Weighted stock | Weighted candidate | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 584 | 488 | 856 | 400 | 5,952 | 5,256 | -696 (-11.69%) |
| B4 | 600 | 504 | 880 | 416 | 6,144 | 5,456 | -688 (-11.20%) |

At 768 CTAs per B1 path and 3,072 CTAs per B4 path, the corresponding static
CTA-slot deltas are -534,528 for B1 and -2,113,536 for B4. Excluding padding
NOPs yields -746 weighted instructions (-12.74%) for B1 and -688 (-11.50%)
for B4. These are code-inventory proxies, not dynamic executed-instruction or
latency measurements: predicates, branches, scheduling, memory traffic, and
occupancy are not resolved offline. All eight audited specializations have
zero stack, zero local memory, and zero LDL/STL spill instructions.

## Exact run command

Run only when the GPU campaign slot is explicitly free:

```bash
cd /home/mark/lumoFlyWheel-gdn-level0-coeff
SOURCE=src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
FA2=/home/mark/lumoFlyWheel-b1-wide256-k64-root-profile/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = 16fde18ebf4ace9893d2f8890294c894c71222b85d7c9cdc4bc7789cf5afff4e
test "$(sha256sum "$FA2" | awk '{print $1}')" = f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="output/fr13_b1_gdn_level0_coeff_k64_root_gate_${STAMP}"
TAG="gdn_level0_coeff_k64_root_${STAMP}"
RUNROOT="$RUNROOT" \
TAG="$TAG" \
TOPOLOGY=hydra27_fixed32 \
FORKED_FA2_SO="$FA2" \
bash scripts/fr13_run_b1_gdn_level0_coeff_live_gate.sh
ARMDIR="$RUNROOT/hydra27_fixed32_k64_gdn_level0_coeff_gate_${TAG}"
python3 - "$ARMDIR/gate_summary.json" \
  "$ARMDIR/swe_out/verified/per_task/astropy__astropy-12907/fixed32_gdn_level0_coeff_real_task_arm.json" <<'PY'
import json
import sys
from pathlib import Path

gate_path, arm_path = map(Path, sys.argv[1:])
assert all(path.is_file() and not path.is_symlink() for path in (gate_path, arm_path))
gate = json.loads(gate_path.read_text(encoding="ascii"))
arm = json.loads(arm_path.read_text(encoding="ascii"))
assert gate["status"] == "pass"
assert gate["run_classification"] == "one_real_swe_verified_k64_b1_byte_diagnostic"
assert gate["acceptance_valid"] is False and gate["timing_eligible"] is False
assert gate["reference_served"] is True and gate["candidate_shadow_only"] is True
assert gate["draft_vocab_root"] == 1 and gate["draft_vocab_k"] == 65536
assert arm["schema"] == "fr13-fixed32-gdn-level0-coeff-real-task-arm-v1"
assert arm["state"] == "ended"
assert arm["instance_id"] == "astropy__astropy-12907"
assert arm["marker"] == "swe_verified:astropy__astropy-12907"
assert arm["gate_eligible"] is False and arm["floor_acceptance_eligible"] is False
print("authenticated GDN coefficient B1 byte diagnostic: PASS")
PY
```

Expected classification:
`one_real_swe_verified_k64_b1_byte_diagnostic`, task
`astropy__astropy-12907`, batch/concurrency 1, acceptance invalid, timing
ineligible, and floor-acceptance ineligible.
