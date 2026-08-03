# Fixed32 FA2 B1 qrow32 split2 SM121a checkpoint

Status: **admit to real SWE-Verified A/B; not production-qualified**.

The split2 kernel restores a 48-CTA main launch on a 48-SM target while
retaining qrow32's single aggregate K/V scan per query head. `blockIdx.y`
selects one of two disjoint K-block intervals; their union is the original
interval and they do not overlap. The launcher then calls FA2's existing
split-K combine kernel with its required independent four-warp traits.

Fresh CUDA 13.0 SM121a codegen from FA2 commit `2921022186` reports:

| kernel | launch | registers | static/dynamic shared | stack/local/spills |
| --- | --- | ---: | --- | --- |
| qrow16 baseline | 48 CTAs x 1 warp | 212 | 1 KiB / 80 KiB | 0/0/0 |
| qrow32 no-split | 24 CTAs x 2 warps | 246 | 1 KiB / 80 KiB | 0/0/0 |
| qrow32 split2 main | 48 CTAs x 2 warps | 250 | 1 KiB / 80 KiB | 0/0/0 |
| stock FA2 combine | 192 CTAs x 4 warps | 44 | 1,072 B / 0 | 0/0/0 |

The split2 main has four registers of headroom under its explicit 254 cap.
Its static body has 3,656 instructions and preserves `HMMA=512`, `FFMA=132`,
`FMUL=264`, `LDGSTS=176`, `LDSM=288`, and `STG=38`. The combine body has
1,096 instructions; its seven static `CALL` sites are inherited from FA2's
existing exp/log reduction implementation, not a new combine path.

`opcode_launch_proxy.tsv` records the requested static-site x launched-warp
comparison. That conservative proxy overstates split work because it cannot
model the halved K-loop trip count or thread predicates. For the core K/V
pipeline, exact interval partitioning gives the meaningful adjusted proxy:
split2 `LDGSTS=8,448`, `HMMA=24,576`, and `LDSM=13,824`. Thus split2 keeps the
same qrow32 K/V traffic proxy as no-split and cuts qrow16 `LDGSTS` by 47.62%,
while adding duplicated Q reads, FP32 partial output traffic, and combine
launch work.

For exact B1 dimensions, split2's additional traffic over no-split is
3,551,232 bytes per layer: one extra BF16 Q read plus two FP32 partial writes
and reads (including LSE). Main K/V payload is `24,576 * N` bytes per layer,
so at the established `N=5120` point the overhead is 2.82% of qrow32 K/V
payload. Qrow16's two full query-tile scans consume `49,152 * N` K/V bytes.

Both qrow32 arms remain hidden, distinctly tagged, fail-closed, and
default-off. This host-only checkpoint used no GPU, synthetic probe, real
task, task data, or timing. The admission decision is only to run lossless
byte qualification and full-step timing on the standing real SWE-Verified
4-task set (or established 16-task set).
