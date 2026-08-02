# Packed x-gather SFWD B1 gate readiness

Status: **READY_NOT_EXECUTED**.

Source commit `eb1a69de3dc180bd29b4488e834c60a3db7bca88` binds the
packed x-gather kernel into the existing default-off, reference-served real
SWE-Verified K64/root1 B1 byte gate.

The kernel loads the full current `32x64` channel tile once per CTA, stages it
for cross-warp gather, and reuses it for the three historical convolution
taps. The incumbent descriptorless kernel logically reads 114 `x` elements
per channel: 32 current rows plus 82 historical tree rows. The three tap
counts are 23, 28, and 31. The x-gather kernel
reads only the 32 current rows from global memory.

Including prior state, weights, output, commit-source staging, and the state
index, the analytical global traffic is 24,196 bytes per incumbent CTA versus
13,700 bytes per x-gather CTA. That is 10,496 bytes, or
43.3790709208134%, less logical global traffic before cache effects. At 160
CTAs, this is 1,679,360 bytes per request-layer and 80,609,280 bytes per
48-layer forward. The candidate adds 14,592 logical shared-memory bytes per
CTA plus synchronization.

The selected row32/C64/W16 B1 specialization compiles offline for SM121a at
55 registers, 4,096 launch-shared bytes, 408 static and 424 encoded SASS
instructions, and zero stack, local memory, spills, or calls. Those facts do
not establish correctness or speed.

The fresh live gate must compare exact bytes for both `conv_out` and
`commit_source_stage` on all 48 layers while serving only the reference. No
GPU, Docker container, real task, synthetic probe, timing, or acceptance run
was launched while producing this package.

This reduced package excludes raw logs, model/task content, requests,
responses, patches, environment values, process/container identifiers, and
secrets.
