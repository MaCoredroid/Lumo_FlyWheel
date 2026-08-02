# FR13 fixed32 SFWD C64 prior-reuse integration

Status: **READY_NOT_EXECUTED** for one real SWE-Verified K64/root1 B1
reference-served byte gate.

The source checkpoint is
`259d050211a8a721feba34d1c4eeab768bd61535`. It adds the distinct,
default-off candidate `fixed32_sfwd_prior_reuse_rowgroup32_c64_v1` without
changing the previously qualified rowgroup8 kernel file. That file remains
byte-identical at SHA-256
`c3036ae4775553e3aeb2131e8b3609c852a22ab86493f7d9843d4aeaed825a70`.

## Integrated gate

The candidate fuses one 32-row request into one row group with `BLOCK_C=64`.
It loads prior state columns 0-2 once, selects the unchanged source row for
taps 0-2, specializes the final current-row tap, and directly stages the
commit source. The gate:

- is B1, eager, K64/root1, and real SWE-Verified only;
- compares `conv_out` and `commit_source_stage` for all 48 layers;
- binds layer pointer and prefix hash identities;
- binds source commit, launch/end manifest, and runtime module hash;
- always returns the incumbent tensors;
- rejects all other candidate, timing, production, and fallback routes; and
- cannot emit a production- or floor-eligible PASS.

## Offline SM121 result

CUDA visibility was explicitly empty. The exact committed kernel was compiled
for B1 and B4 with separate fresh caches. No GPU kernel, Docker container,
service, task, request, probe, or timing run was launched.

| Metric | B1 | B4 |
|---|---:|---:|
| CTAs per launch | 160 | 640 |
| CTAs per request | 160 | 160 |
| Registers per thread | 62 | 62 |
| Allocated registers per CTA | 16384 | 16384 |
| Static / encoded SASS | 993 / 1008 | 993 / 1008 |
| LDG / STG / LDS / STS | 64 / 20 / 0 / 0 | 64 / 20 / 0 / 0 |
| Launch shared bytes | 0 | 0 |
| Stack / local / calls | 0 / 0 / 0 | 0 / 0 / 0 |
| Cubin bytes | 69088 | 69088 |

B1 and B4 cubin, PTX, SASS, and resource reports are byte-identical. A second
fresh-cache build reproduced both variants byte for byte. The cubin SHA-256 is
`8ed5af3a8efaf7eff45a048a9689c09f5d5440223a63dcd2220218573c453adc`.

These are static compile properties, not latency, occupancy, byte-correctness,
TPS, or hardware-floor evidence. The next required action is the source-bound
one-task B1 gate via `scripts/fr13_run_b1_sfwd_prior_reuse_gate.sh`; only a
clean byte PASS can authorize later real-task timing.

The package is reduced: it contains summaries, checksums, and audit scripts,
but no cubin, PTX, SASS, IR, task/model content, request logs, environment
dump, credentials, or secrets.
