# Fixed32 full-vocabulary BF16 small-M real-B1 diagnostic

Status: `PREPARED_NOT_RUN`.

This artifact prepares one authenticated `SWE-Verified` B1 task using the
canonical `astropy__astropy-12907` subset. It shadows the first measured real
decode event at the root and four MTP head positions with padded GEMMs for
`M={2,4,8,16}`. The stock `UnquantizedEmbeddingMethod` BF16 result remains the
served result for every head and every event.

The output is diagnostic evidence only. It is not a synthetic probe, timing
arm, throughput result, B1/B4 acceptance result, or hardware-floor claim.
Numerical mismatch is a valid completed result for an individual M; missing
root-plus-four comparisons, provenance drift, or task-census drift fails the
run.

Run `prepared_command.sh` only after the active SFWD container releases the
GPU. The runner itself also requires zero existing Docker containers.

Expected machine-readable outputs below the new runroot:

- `hydra27_fixed32_*/logs/fr13_draft_head_msweep.live.json`
- `hydra27_fixed32_*/draft_head_msweep_validation.json`
- `launcher_meta.txt`

