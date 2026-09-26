Second GB10 here (cc 12.1, 48 SMs, driver 590.48.01) — but torch 2.13 / CUDA 13.2 and
Triton ptxas 13.1, not your 13.0, which may explain the gaps.

At `b40edf4`: 34 passed, 15 keys set-equal, `sm120` byte-identical, row-0 hashes match
across M ∈ {1, 8, 32, 2048} on all 15 shapes, and the per-M tile survives `torch.compile`
and cudagraph replay.

Both effects reproduce: tuned/default 0.72 / 0.75 / 0.93 at M=1/32/256, and
1.03 / 1.26 / 1.20 at M=512/1024/2048 (worst shape 1.72x).

Two questions:

1. Against `sm120` — what a GB10 runs today, not the default tile — I measure 0.89x at
M ≤ 256 but 1.16x at M=1024 and 2048. Is deepening the large-M regression versus the
status quo intended?

2. I could not reproduce the `BLOCK_K` argument: changing only `BLOCK_SIZE_K` (32/64/128)
left all 2.24e9 output elements bitwise identical, while split-K halves and cuBLAS do
change them. Does `BLOCK_K` move the bits on your box? If not, could M ≥ 512 take the
default tile?

Probe: {{BRANCH_LINK}}

Numbers gathered with AI assistance; I ran and reviewed them.
