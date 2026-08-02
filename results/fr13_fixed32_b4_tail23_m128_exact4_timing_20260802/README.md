# FR13 fixed32 B4 Tail23 persistent-M128 exact4 timing

Status: **completed real SWE-Verified exact4 diagnostic; negative full-TPS
screen; not formal floor acceptance**.

## Scope

- Real SWE-Verified canonical four-task set, B4/concurrency 4.
- Tail23 logical topology in a fixed 32-row physical tree.
- Draft vocabulary K64/root1 in both arms.
- Stock CUTLASS versus the byte-qualified persistent-M128 candidate.
- Same all-parent committer selector and equal normalized physical-work
  signature in both arms.
- Full-step timing includes SFWD, DFWD, CFWD, and other wall time.

## Result

| Metric | Stock | Persistent M128 | Candidate change |
| --- | ---: | ---: | ---: |
| Full step wall | 325.0571 ms | 318.4117 ms | -2.044% |
| Full-step wall TPS | 69.8274 | 63.9685 | -8.391% |
| Accepted drafts/event | 7.7671 | 7.2388 | -6.801% |
| SFWD GPU/step | 219.7473 ms | 214.6336 ms | -2.327% |
| DFWD GPU/step | 42.2387 ms | 41.6468 ms | -1.401% |
| CFWD GPU/step | 44.4621 ms | 42.1106 ms | -5.289% |
| Other wall/step | 18.6090 ms | 20.0207 ms | +7.587% |
| GPU components/step | 306.4481 ms | 298.3910 ms | -2.629% |
| Wall/floor ratio | 2.7166x | 2.6610x | -0.0555x |

The candidate reduces kernel-side SFWD and full-step wall, but it loses the
decision metric because its independent task arm accepts fewer drafts. The
full-TPS ratio is `0.916095`, so this exact4 screen does not justify promotion
or an exact16 statistical floor campaign.

The candidate remains `180.8050 ms/step` above the `137.6067 ms` one-sided
1.15x cap and would need another `56.783%` wall reduction from this point.

## Reducer recovery

All four real tasks and both server arms completed before reduction. The first
reducer attempt failed because the root-running container wrote the non-secret
CUTLASS binary attestation as mode `0600`, making it unreadable to the host
user after teardown. No performance task was rerun.

The completed attestation was changed to read-only mode, the original binary
and production-sidecar validators passed, and the timing script's exact
embedded reducer was replayed against the original completed arm artifacts.
It validated exact4 provenance, deployment timing fields, the K64/root1 floor,
production binding, and equal normalized work signatures before emitting the
summary.

Future launches now chmod this attestation `0444` inside the container before
host reduction. Fix commit: `e0b4b290a`.

## Claims

This is a valid exact4 diagnostic comparison and a clean full-step breakdown.
It is not formal hardware-floor acceptance, does not carry a one-sided U95,
does not enable the candidate by default, and contains no raw prompts,
responses, generated task patches, traces, logs, process/container IDs,
environment dumps, or secrets.
