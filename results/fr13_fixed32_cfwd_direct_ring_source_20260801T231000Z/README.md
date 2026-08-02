# Fixed32 CFWD direct-ring layer-batch candidate

Status: **default off; source/static ready; live GPU compile, byte coverage,
same-process timing handoff, and real SWE-Verified timing pending**.

## Candidate

This branch keeps the incumbent fixed32 committer as the reference and changes
only the unqualified 48-layer state-update candidate. The candidate now:

- launches one recurrence kernel across all 48 layers;
- omits the unused output projection and output tensor;
- runs only `accepted_drafts + 1` recurrence steps instead of all 16 storage
  slots;
- writes each authoritative final state row once, after the recurrence;
- reads K, V, a, and b directly from the four live activation rings, with no
  candidate staging gather or staging-row allocation;
- hoists `exp(A_log)` and the dt-bias load outside the recurrence loop.

The mathematical recurrence remains ordered FP32. The fixed path storage stays
16 columns, but accepted draft lengths are topology-bounded to `0..11`; columns
12 through 15 are unreachable padding. The byte-coverage mask is therefore
`0x0fff`, not `0xffff`, and runtime guards fail closed above 11.

## Static work removed

These are logical-work counts, not DRAM or latency measurements.

- Candidate staging traffic removed: 24.28125 MiB/event at B1 and
  97.125 MiB/event at B4.
- At the measured Hydra27 mean acceptance of 4.753885 drafts/event, active
  recurrence executes 5.753885 of 16 storage iterations, a 64.0382% reduction
  in recurrence iterations.
- Hoisting the final state store removes 15 of 16 stores in the fixed-capacity
  case: 2.109375 GiB/event at B1 and 8.4375 GiB/event at B4 in logical store
  bytes. At the measured mean acceptance, the avoided logical store traffic is
  approximately 717.813 MB/event at B1 and 2.871 GB/event at B4.

No speedup is claimed from these counts. Cache behavior, register pressure,
occupancy, and the longer fused kernel must be measured on the target GPU.

## Qualification lifecycle

A dedicated B1/sequential real-task arm and task bracket can collect missing
accepted-length byte coverage on an authenticated SWE-Verified task. The arm
is atomically published with mode 0400, is removed before the post-boundary
snapshot, requires monotonic attempts/coverage, and records newly covered and
remaining depths.

This run is explicitly classified as qualification only:

- `performance_measurement=false`
- `timing_eligible=false`
- `floor_acceptance_eligible=false`
- `process_local_qualification_only=true`
- `durable_production_pass=false`
- `same_process_timing_handoff_implemented=false`

Coverage cannot authorize a rebooted process. B4 also needs a campaign-scoped
exact4/16 lifecycle because its tasks overlap; a single task-ID marker is not a
valid B4 credential. Both lifecycles must hand qualification to timing in the
same server process before this candidate can produce valid performance data.

## Required next gates

1. Compile on the target GPU and require acceptable register, stack, and local
   memory use; reject spills or resource regressions that erase the saved work.
2. Complete raw-byte state equality for every reachable accepted length
   `0..11`, separately at every used occupancy.
3. Add and validate the same-process B1 and B4 qualification-to-timing handoff.
4. Run only the standing real SWE-Verified task sets: B1 one-task diagnostics,
   exact4/16 formal campaigns, and B4 exact4/16 campaigns.
5. Report full-step wall TPS and SFWD, DFWD, CFWD, and other milliseconds per
   event against the 119.658015414 ms weight-read floor and 137.606717726 ms
   one-sided 1.15x cap.

