# Static scheduler corrected real B1 gate

Status: PASS for one-task real SWE-Verified B1 byte correctness only.

This is not timing or floor-acceptance evidence. The diagnostic served the
stock result, used one task, and has `acceptance_valid=false` and
`comparator_timing_eligible=false`.

## Result

- Task: `astropy__astropy-12907`, resolved in 395.1 seconds.
- Full vocabulary: root 0, K 0.
- Physical comparator rows: 32.
- Comparisons: 320 contiguous calls, invocations 0 through 319.
- Compared output: 311,951,360 bytes.
- Equality: 320/320 byte-exact, zero differing bytes.
- Coverage: all five corrected projection shapes, including 20 calls to
  packed-QKV `(M,N,K)=(32,14336,5120)`.
- Formal gate: `status=pass`, no errors.
- Independent qualification validator: `status=QUALIFIED`; this denotes the
  production credential's correctness check, not performance acceptance.

| `(M,N,K)` | Calls | Mismatches |
| --- | ---: | ---: |
| `(32,5120,6144)` | 80 | 0 |
| `(32,5120,17408)` | 80 | 0 |
| `(32,14336,5120)` | 20 | 0 |
| `(32,16384,5120)` | 60 | 0 |
| `(32,34816,5120)` | 80 | 0 |

## Permission replay

The original top-level launcher returned 1 only after the task completed. The
host reducer could not read the nonsecret binary attestation because the
container's atomic writer created the bind-mounted file as `root:root` mode
`0600`. The file's SHA-256 was
`ac4342fc57300553eb91d1f30c1f36d25867fa69dd5e16a9424f09d2611e047d`.

Ownership alone was changed to `mark:mark`; mode stayed `0600` and the content
SHA-256 did not change. Replaying the unchanged reducer at source commit
`f0f91bd53` returned 0 and wrote the committed raw PASS JSON. This repair did
not alter the comparator, task, binary attestation, or gate inputs.

The narrow lifecycle fix makes only this nonsecret attestation explicitly
mode `0644` at atomic publication. A focused unit test pins that behavior so
future host reducers do not require an ownership repair.

## Contents

- `cutlass_streamk_byte_gate.raw.json`: formal corrected PASS.
- `comparisons.raw.jsonl`: all 320 raw comparison records.
- `comparator_summary.json`: structured reduction of the raw JSONL.
- `campaign_summary.raw.json` and `swe_orchestrator.raw.log`: task resolution.
- `binary_attestation.raw.json`, `real_task_arm.raw.json`, and
  `b1_diagnostic.raw.json`: binary, task, and diagnostic identities.
- `permission_replay.json`: rc1 cause and content-preserving replay record.
- `qualification_validation.json`: offline correctness validation.
- `run_identity.json`: pinned run and artifact hashes.

No container environment, process environment, credentials, secrets, or full
container logs are included.
