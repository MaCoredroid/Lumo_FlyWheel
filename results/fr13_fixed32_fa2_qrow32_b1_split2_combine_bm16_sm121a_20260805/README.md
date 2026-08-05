# Fixed32 FA2 B1 split2 BM16 combine candidate

Status: **offline codegen pass; admit to a real SWE-Verified byte and timing
A/B only**.

The current qrow32 B1 split2 arm launches FA2's generic four-warp combine at
`kBlockM=4`. Exact B1 has `1 * 24 * 32 = 768` logical output rows, so the
combine grid is 192 CTAs per layer. This candidate instantiates the unchanged
FA2 combine body at `kBlockM=16`, giving exactly 48 CTAs per layer on the
48-SM target.

The specialization is byte-equivalence defensible, but not yet byte
qualified. Both arms use two context splits, a two-lane LSE reduction, and
split accumulation order `[0, 1]` for every `(row, column)` output. The
logical output coverage is identical and bijective. The attention main
kernel is unchanged; its extracted SASS is byte-identical between the BM4
control and BM16 candidate.

| combine arm | CTAs/layer | threads/CTA | registers | static shared | stack/local | static SASS instructions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pinned BM4 control | 192 | 128 | 44 | 1,072 B | 0/0 | 1,080 |
| BM16 candidate | 48 | 128 | 76 | 1,168 B | 0/0 | 1,056 |

Across 16 tree-attention layers, CTA instances fall from 3,072 to 768. The
static CTA-instruction proxy falls 75.56%. This is not a timing result: BM16
also reduces the available combine warps from 16 to four per SM if BM4's four
CTAs are concurrently resident. A real task may therefore show either a win
or a regression depending on latency hiding.

The combine's mandatory scratch reads and final writes are unchanged at
1,975,296 bytes per layer, or 31,604,736 bytes per 16-layer full step. At the
campaign's 273 GB/s weight-read floor bandwidth that traffic alone is about
0.116 ms/full step. This change removes CTA scheduling and replicated control
work; it does not remove a kernel launch, arithmetic, or scratch traffic.

The control was rebuilt with CUDA 13.0.88, FA2 commit `2921022186`, CUTLASS
commit `62750a2b`, and CUDA-enabled PyTorch 2.10.0 headers. Its full SASS dump
matches the pinned v3 control exactly. No GPU, Docker, synthetic probe, real
task, task payload, linked shared object, or timing was used here.

Do not merge this source over the pinned v3 generator or repin production
identity until it passes the canonical real B1 byte comparator and full-step
timing. If it wins B1, B4 is unaffected because this is the exact B1 split2
translation unit.
