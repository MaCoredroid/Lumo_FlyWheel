# SFWD state-fusion exact4 B4 preflight

This bundle prepares the source-only `fixed32_sfwd_state_fusion_v1` byte gate
for canonical real SWE-Verified exact4 traffic at server capacity and
concurrency 4.

The gate is deliberately eager and candidate-shadow-only. It compares
`conv_out` and `commit_source_stage` for 48 unique layers at four physical
requests of 32 rows each, while the incumbent reference remains served. It
makes no acceptance, timing, floor, or production claim.

`prepared_command.sh` is ready for the live GPU host after setting the absolute
pinned `FORKED_FA2_SO` path. No live task, Docker, or GPU command was run while
preparing this bundle.

Production stays default-off. A B4 qualification is not a serving credential.
After an authenticated B1 byte PASS exists, use
`scripts/fr13_sfwd_state_fusion_b4_pass.py bind-prerequisites` to bind the B1
PASS and B4 qualification. The resulting artifact remains byte-only and does
not enable candidate serving.
