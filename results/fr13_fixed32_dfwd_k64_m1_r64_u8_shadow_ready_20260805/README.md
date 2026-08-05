# DFWD K64 M1 R64-U8 Shadow Qualification Readiness

Status: `SHADOW_READY_UNMEASURED`

This artifact records a default-off, incumbent-served qualification path for
the linked DFWD K64 M1 R64-U8 candidate. No GPU, Docker, or SWE-Verified task
was run from this worktree. It makes no correctness, performance, timing, or
production claim.

The prepared real run is Hydra27 fixed32 at B1 on
`astropy__astropy-12907`, using the fixed K64/root1 draft head and
`FULL_AND_PIECEWISE` CUDA graphs. For every authenticated measured event, the
shadow compares exact raw BF16 bits for all 65,536 configured draft-head logits
at the root and actual MTP depths 1, 2, 3, and 4. This is exhaustive within the
fixed K64/root1 head. It is not an exhaustive comparison of the full model
vocabulary.

The candidate result is never returned. The incumbent BF16 logits object is
served unchanged. Qualification fails closed on missing or conflicting worker
environment state, identity drift, missing depths, event-count drift, any raw
BF16 mismatch, incomplete terminal evidence, or an unresolved real task.

Source state:

- Rebased main boundary: `1f7485ade5ec6bfacf51dde7afa514531effcbcd`
- U8 wiring commit: `67aa481eb9563daa035490185bcf649945b5ad6e`
- Source tip including independent M32 repair: `674f574a0346b4f7b2bc96a30a4ad403841c41d4`
- Branch: `codex/dfwd-u8-shadow-qualification-20260805`

Prepared runner:

```bash
RUNROOT=output/<new-run-directory> \
TAG=<unique-tag> \
bash scripts/fr13_run_b1_dfwd_k64_m1_r64_u8_live_gate.sh
```

The runner requires a clean tracked worktree and no existing Docker containers.
The resulting live and terminal sidecars must pass
`scripts/fr13_dfwd_k64_m1_r64_u8_gate.py` before any correctness statement is
made.
