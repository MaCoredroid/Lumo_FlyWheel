# Fixed32 DFWD K64 M1 shuffle R32 build

Status: pinned host build and static codegen audit pass; default off, not
byte-qualified, not timed, and not acceptance valid. No GPU, synthetic probe,
Docker workload, or real task was run.

The R32 candidate maps 32 output rows to each 512-thread CTA and launches 2,048
CTAs for the fixed K64 head. Its R16 parent maps 16 rows to 256-thread CTAs and
launches 4,096 CTAs. Total threads and all per-row arithmetic remain unchanged;
R32 only halves block scheduling work.

The `sm_121a` kernel uses 18 registers/thread and zero stack, local, or shared
memory. SASS contains no barrier, local load/store, spill, atomic, or call. The
four width-16 shuffles and four FP32 adds implement the fixed `8+4+2+1`
reduction. The dependent loop has one static FP32 FMA instruction and the
epilogue has the second, followed by BF16 conversion and one output store.

The immutable candidate binary is `fr13_bf16_gemvx_k64_m1_shuffle_r32.abi3.so`
with SHA256
`c389bf5e01b942cfe73b2e4fc05db7b158f16b61205c9f3e9988cbd8a82474dd`
and 113,648 bytes. It requires no GLIBC symbol newer than 2.32.

Source commit: `0f627cfac51d4a85408ba8ad3e3040d33b6f43b6` on
`agent/fixed32-dfwd-k64-m1-shuffle-r32-20260802`.

Next is a default-off real SWE-Verified B1 K64/root diagnostic covering root and
all four MTP head positions. Only after a clean real diagnostic may it enter
matched full-step timing. B1 remains diagnostic; formal acceptance requires the
standing exact4 B4 or exact16 campaign.
