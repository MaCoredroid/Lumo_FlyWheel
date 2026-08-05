# Fixed32 FA2 qrow32 exact-B4 candidate, SM121a

Status: **CPU/codegen admission pass; real exact4 byte-shadow gate pending**.

The B1 qrow32 no-split binary cannot cover exact4 B4. Its launcher requires
`b == 1`, `total_q == 32`, one static sequence, and doubled K/V batch strides;
its pinned library contains only the B1 no-split and split2 launchers. The
generic B4 path instead requires `b == 4`, `total_q == 128`, four static
sequences, and one `1024 * 4 * 256` K/V batch stride.

This checkpoint adds an inner exact-B4 guard to the BM32 launcher and pins the
previously open `qrow32` selector arm to a B4-specific library. A hidden,
single-purpose C++ adapter lets the already qualified exact-B4 API gate call
the BM32 launcher without changing the public extension ABI. The candidate is
not stored in Git:

| field | value |
| --- | --- |
| candidate SHA-256 | `77f3fb22c19d0eb2ac0ec28230cf9401221425692a505efde62aa838760d81ce` |
| candidate size | `299876120` bytes |
| FA2 head | `29210221863736a08f71a866459e368ad1ac4a95` |
| source closure | `3e3c18565e738f20d0a5ab5fe50d018f3d8cbd5cb94082dcd55ca730a790163c` |

The SM121a device image is unchanged by the new host guard: its SASS digest is
`a814ccdb99b9a63b915ac762c5cc02dc536a98f0d9c1484a5853daaa74024cc5`
before and after the guard. Ptxas reports 252 registers, zero stack and spills,
one barrier; cuobjdump reports 1,024 bytes static shared and zero local memory.
The 3,992 SASS instructions contain no `LDL`, `STL`, or `CALL`.

For physical32 B4, qrow16 launches `2 * 4 * 24 = 192` CTAs per layer. BM32
qrow32 launches `6 * 4 * 4 = 96` CTAs per layer. Both use one attention launch
per layer and neither uses a combine kernel. Across the 16 target tree layers,
this is 3,072 to 1,536 CTAs, a reduction of 1,536 CTAs or 50%, with launch
count unchanged at 16.

The existing old generic qrow32 library is not admitted: it was built from a
different translation unit and has defined/undefined dynamic-symbol drift.
The new finalized candidate exactly matches the pinned GQA API reference's
defined symbols, undefined symbols, and `DT_NEEDED` lists, and both launchers
remain local symbols.

No GPU, Docker container, task payload, synthetic timing, raw SASS, object, or
shared library is included. The required next step is the canonical real
SWE-Verified exact4 B4 raw-byte A/B on Hydra27 K64/root1 (and Tail23 where the
standing gate requires it). The current host Torch installation cannot load
even the stock or pinned GQA reference because its CUDAStream symbol providers
differ, so host loading is not accepted as a substitute for that pinned-image
live gate. No performance or production claim is made.
