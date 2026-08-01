# B4 Tail23 Qwen compaction contract abort

The real SWE-Verified exact-four Tail23 all-parent B4 K64 campaign completed
all four tasks from source `080c417ed627e155c98e715327e0fdeb48d542ab`.
Its post-task campaign finalizer rejected the run before publishing any formal
task metadata or campaign proof.

The authenticated ingress ledgers contain 123 completed model requests. The
four structurally valid Qwen traces account for 119 normal requests. One trace
ends with the already-pinned exact synthetic compression-failure terminal, and
the campaign endpoint metrics prove that the remaining four requests were
20,000-token compaction calls. All four completed successfully at the engine;
none produced a trace-visible successful compaction response.

The old contract rejected every failed-only compaction case even when the
exact synthetic terminal and metric algebra independently proved it. Code
commit `2177cbba1148e58f705a5bc9dc02860c6aa87156` permits that case only when the
exact terminal recognizer is true. Ordinary count gaps, terminal near-misses,
and metric tampering still fail closed. The same rule now covers both B1
single-task and B4 campaign reconciliation.

The completed run is not a formal production pass. Exact task-auth snapshots
needed to publish final provenance were held only in process memory and were
intentionally omitted from pending metadata, so they cannot be reconstructed
without inventing evidence. The exact-four gate must be rerun from the fixed
source. No M128 gate, timing arm, TPS result, acceptance result, or
hardware-floor result was issued from this run.

This directory contains reduced aggregate metadata only. It contains no task
identifiers, prompts, responses, patches, trace contents or hashes, raw logs,
credentials, process identities, or container identities.
