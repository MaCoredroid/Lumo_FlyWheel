# Fixed32 GDN single-launch v2 SM121a resource audit

Verdict: **CODEGEN_VIABILITY_PASS_ZERO_SPILL**.

This is a reduced offline compile/resource pass, not a GPU qualification or a
performance claim. The exact B1 and B4 live specializations compile twice from
fresh, isolated caches to deterministic `sm_121a` cubins. Both use 97
registers/thread and have zero stack, local memory, LDL, STL, calls, global
scratch, and tensor memory. Those results make a live byte-qualification
reasonable; they do not authorize production.

## Exact specialization

The signature and every constexpr in `compiler_audit.json` are derived from
the production launchers and source at commit
`82bdc7e03838593706dfb9e96648dcac7234a615`:

- q/k/v/raw-a/raw-b/out/rings: BF16 pointers;
- g/beta/A-log/dt-bias/h0: FP32 pointers;
- h0 indices: int64; descriptors, accepted counts, counter, and flags: int32;
- N=32, KH=16, VH=48, DK=128, DV=128, BV=8, 8 warps;
- output scale `128**-0.5`, QK normalization on, raw gating on;
- h0 bank on, bank stride 786,432 elements, index row 0, request stride 32,
  accepted-count stride 1, accepted-column selection off;
- counter, ring export, and flags export on; scan alignment off;
- root steps 5, branch max length 7, max group members 3, groups 5;
- flag rows 1 for B1 and 4 for B4;
- `num_stages` is unset in source and resolves to Triton 3.6.0's default 3.

The fixed32 index-row stride is 32 because the live physical tree uses 31
speculative rows plus the root column. The B1 program id is always request 0;
B4 uses that stride to address requests 0 through 3.

## Resources

| Metric | B1 | B4 |
| --- | ---: | ---: |
| Grid | 48 x 16 x 1 | 48 x 16 x 4 |
| CTAs per request | 768 | 768 |
| CTAs in the launch | 768 | 3,072 |
| Registers/thread | 97 | 97 |
| Registers/CTA (256 threads) | 24,832 | 24,832 |
| Stack bytes | 0 | 0 |
| Local bytes | 0 | 0 |
| Shared bytes, cuobjdump | 1,024 | 1,024 |
| Shared bytes, Triton metadata | 128 | 128 |
| LDL / STL / call instructions | 0 / 0 / 0 | 0 / 0 / 0 |
| SASS instructions | 7,232 | 7,232 |
| Primary text bytes | 115,712 | 115,712 |
| Supplemental capmerc text bytes | 15,118 | 15,118 |
| Cubin bytes | 567,680 | 567,680 |

`cuobjdump` and `nvdisasm` independently count the same 7,232 primary SASS
instructions. The primary text size is exactly 7,232 x 16 bytes. B1 and B4
have identical resources and body sizes but distinct cubin/compile hashes
because `FLAGS_ROWS` is an exact compile-time value.

The source contract's 4,096 nominal FP32 values per CTA is arithmetic, not the
compiler allocation. The compiled allocation is 97 32-bit registers/thread,
or 24,832 allocated registers over 256 threads. No device occupancy was
queried or inferred.

## Current path comparison

The current two-launch path was compiled in the same two fresh-cache passes
with the same live datatypes, constants, counter/flag behavior, and SM121a
toolchain. Per request it launches 768 level-0 CTAs and 8,448 level-1 CTAs,
9,216 total. Thus its B1 launches contain 768 and 8,448 CTAs; its B4 launches
contain 3,072 and 33,792 CTAs. The single-launch candidate contains 768 CTAs
per request in one launch, a 12x static CTA-count reduction.

The exact current B1 level-0/level-1 bodies use 80/80 registers and contain
552/488 instructions. B4 uses 64/80 registers and 592/504 instructions. All
four are also zero-spill. The candidate body is much larger: 7,232
instructions and 115,712 primary text bytes. Summing separate static kernel
bodies is not a dynamic instruction or latency model, so no speed conclusion
is drawn from either the lower CTA count or the larger body.

The candidate also removes the current path's five FP32 state exports and
eleven parent reads per request/layer. At the exact geometry, one full state is
3,145,728 bytes, so the removed handoff traffic is 15,728,640 export bytes and
34,603,008 parent-read bytes per request/layer. These are descriptor-derived
byte counts, not bandwidth or latency measurements.

## Level-0 coefficient context

The reviewed coefficient artifact at commit
`78d2b64aeebea8ba2fea679b2c9e826cd274d4d0` keeps two launches and the same CTA
grids. Its reduced B1 candidate resources are 77/64 registers for levels 0/1;
B4 is 79/64. All have zero stack/local/LDL/STL/calls. Their primary bodies are
856/400 instructions for B1 and 880/416 for B4.

That comparison is contextual, not exact-live equivalence. Its builder uses
`COUNT_INVOCATION=False`, and its B4 build records
`H0_INDEX_BATCH_STRIDE=1` instead of the live stride 32. Its manifest also
queries a PATH `ptxas` separately rather than identifying Triton's internal
producer. The figures above were independently reduced from its retained
cubins with the current CUDA inspectors; no old binary was copied here.

## Toolchain and boundary

Triton 3.6.0 selected its packaged Blackwell assembler
`backends/nvidia/bin/ptxas-blackwell`, CUDA 12.9 V12.9.86. That is the actual
producer. System CUDA 13.0 `cuobjdump` and `nvdisasm` were inspectors. The audit
harness did not invoke system `nvcc`, and `nvcc` was not a compiler producer.

No GPU kernel, service, synthetic probe, real task, byte gate, CUDA graph,
timing, or acceptance campaign ran. Remaining risk is the 32-node serial
recurrence, 97-register allocation, and very large SASS body, plus unverified
raw-byte parity, graph capture/replay, instruction-cache behavior, occupancy,
and real workload performance. A live B1/B4 qualification is still required.
