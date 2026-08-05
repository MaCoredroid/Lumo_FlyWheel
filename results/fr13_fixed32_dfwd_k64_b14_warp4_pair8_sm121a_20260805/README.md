# Fixed32 DFWD K64 B1/B4 warp4-pair8 candidate

Status: **default off, runtime unwired, static codegen pass**. This artifact
contains no GPU execution, real-task correctness, timing, acceptance, or
hardware-floor claim. The historical 36.8134 ms/event DFWD figure is context
for prioritization only and was not measured with this candidate.

## Kernel change

The integrated R64-U8 B1/B4 heads use 1,024-thread CTAs and scalar BF16 global
loads. This candidate uses 2,048 256-thread CTAs. Each of eight warps computes
four adjacent vocabulary rows and loads eight BF16 values at a time with
`LDG.E.128`. B1 reuses one packed hidden octet across four vocabulary rows. B4
also loads each packed weight octet once and reuses it across four requests.

The full `65536 x 5120` head remains intact. The candidate emits proposal
logits only and has no target logits, target probabilities, RNG, acceptance,
or rejection inputs. Its width-32 reduction can change drafter quality, which
is allowed only while target-authoritative rejection sampling remains the
lossless serving authority.

## Static result

| Metric | B1 | B4 |
| --- | ---: | ---: |
| Registers/thread | 40 | 80 |
| Stack/spill/local/shared bytes | 0 | 0 |
| Barriers/atomics/calls | 0 | 0 |
| Packed global loads in steady body | 5 | 8 |
| Output stores | 4 | 16 |
| Modeled load-issue reduction vs R64-U8 | 92.1875% | 95.0% |
| Modeled warp-request-byte reduction | 37.5% | 60.0% |

The byte model counts logical warp requests before cache effects; it is not a
DRAM measurement. Both arms still require 671,088,640 bytes of BF16 weights per
head call. Two independent CUDA 13.0 `sm_121a` builds produced byte-identical
cubins and disassembly. A Torch 2.11.0+cu130 linked extension registered both
ops with `CUDA_VISIBLE_DEVICES` empty.

## Next gate

Wire one authenticated selector without changing the incumbent route, then run
real SWE-Verified B1 and B4 proposal/target-authority gates. Only candidate-
served exact4/exact16 runs may establish throughput or hardware-floor progress.

Source checkpoint: `26d18c3a3220a071c34fd8c4fb4967e6e6ba64b1`.
