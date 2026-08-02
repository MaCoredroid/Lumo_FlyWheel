# Verification

- Source revision: `0cdab29fcbc90351e915bb687994785cbd2fcdd9`.
- Checked-in codegen-helper hash: bound and verified before both builds.
- Focused state-fusion plus int32-descriptor suite: 18 passed.
- Python bytecode compilation: passed.
- Live padded B4 address maximum and unsafe-stride rejection: passed.
- Fresh isolated-cache builds: two.
- Compiled specializations per build: B1 and B4.
- Duplicate generated output trees: byte identical.
- B1/B4 cubin, PTX, SASS, resource, and non-launch metrics: identical.
- Target and producer: SM121a, `ptxas-blackwell` 12.9.86.
- Reported/allocated registers per thread: 56/56.
- Allocated registers per CTA: 14,336.
- Static/encoded SASS instructions: 912/928.
- LDG/STG/LDS/STS: 64/20/0/0.
- Stack, local memory, LDL, STL, shared memory, and calls: zero.
- Source tree whitespace/error check: passed.
- CUDA visibility during codegen: explicitly empty.
- Live correctness, timing, SWE task, service, and production selection: not run.
