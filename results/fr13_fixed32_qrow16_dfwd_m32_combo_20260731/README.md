# Fixed32 qrow16 plus DFWD M32 gate bundle

This branch is a clean source-only integration on `47de87295`. Both candidates
remain default off and nondeployable. It combines the final FA2 qrow16 source
and real-capture output/LSE gate with the exact-gated DFWD BF16 M32 source and
full-logit byte gate.

The standing-rule-valid B1 task is `astropy__astropy-12907`: the prior Tail
exact run completed 515 events in about 180.8 seconds of agent wall, making it
the shortest exact4 task. No synthetic or probe traffic is permitted.

For the diagnostic task, set `FR13_DRAFT_HEAD_M32=0` and
`FR13_DRAFT_HEAD_M32_BYTE_AB=1`. That gate evaluates both draft-head paths on
each real input but always returns the reference logits. Keep the served FA2
path on the pinned exact-safe binary while capturing the real paged B1
operands. After the qrow16 SO passes the documented CPU/ABI build gates, run
`scripts/fr13_fa2_qrow16_byte_ab.py` on that capture; it compares candidate B1
output and LSE with stock B2/B4 fallbacks inside one CUDA process.

This task is diagnostic only. Neither its wall time nor its acceptance is a
formal candidate result. Candidate-only real B1 timing starts only after both
byte gates are zero-diff and their artifacts are committed.
