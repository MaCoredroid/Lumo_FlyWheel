# FR13 fixed32 unified-attention BM8 B1 live-gate readiness

This artifact records CPU-only readiness for a deterministic real
SWE-Verified B1 raw-byte gate. It makes no GPU parity, speed, hardware-floor,
quality, or acceptance claim.

## Audit result

The BM8 source was already present in the integrated fixed32 tree. Two gaps
made its prior diagnostic insufficient:

1. `proposal["measured"]` proved fixed32 census ownership but did not prove that
   the replay occurred inside the pinned SWE task. A pre-task measured replay
   could therefore consume the one-shot gate.
2. The PASS record described BM8 geometry but did not bind the emitted Triton
   source, Eagle replay hook, patcher, and repository commit by exact identity.

The repaired gate waits for an atomic mode-0400 marker published only after the
pinned task's pre-flush boundary. It compares four private stock-BM16 outputs
with four private candidate-BM8 outputs, requires four candidate dispatches,
and continues to serve the captured stock graph output. The marker is rotated
after the task's post-flush boundary.

At prelaunch, the container writes a mode-0400 identity JSON containing the
repository commit and SHA-256 hashes of the patcher, emitted unified-attention
source, and patched Eagle replay hook. The live gate re-hashes all three files
and embeds the exact identity in its PASS record. The host validator rejects a
missing task marker, source drift, fewer than four calls, false stock-vs-stock
dispatch, any byte mismatch, or a non-PASS record.

## Source

- Branch: `agent/fixed32-bm8-livegate-ready`
- Integrated base with corrected floor: `2675cdf3bdde282f533b29c8954bd90dd7a2a76b`
- Source checkpoint: `a52de453c17ee75c922bed13d6f911acf3820e36`
- Original BM8 integration: `a477684375fce02ecbe0b0563600042943609900`
- Selector: `FR13_DFWD_UNIFIED_BM8_LIVE_AB=1`
- Production selector: none
- Default behavior: stock BM16, because the diagnostic selector defaults to 0

## Next gate

Run only after the GPU is free, using the canonical stock FA2 object with
SHA-256 `f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`:

```bash
RUNROOT=output/fr13_bm8_b1_live_gate_<UTC> \
TAG=bm8_b1_live_<UTC> \
FORKED_FA2_SO=<canonical-stock-fa2-so> \
FR13_GATE_BM8=1 \
FR13_GATE_QROW16=0 \
FR13_GATE_TAW_NATIVE=0 \
FR13_GATE_DRAFT_HEAD_PAD=0 \
FR13_GATE_GDN_BV=0 \
bash scripts/fr13_run_b1_kernel_live_gate.sh
```

This is a one-task B1 diagnostic and is not formal exact4/exact16 acceptance or
timing evidence. A successful command must terminate with a validator PASS for
`astropy__astropy-12907`; task resolution alone is insufficient.

## Verification

- Python compilation: pass
- Shell syntax: pass
- Ruff on the new validator, orchestrator, and BM8 tests: pass
- Focused CPU tests: 119 passed, 1 environment skip
- GPU or Docker launched: no
- Correct floor retained: 119.658015414 ms/event
- One-sided 1.15x cap retained: 137.606717726 ms/event
