# Verification

- Source revision: `8a3e77792ecdfd7df798ab24715aefbca11d3c93`.
- Focused prior-reuse plus int32-descriptor suite: 16 passed.
- Python bytecode compilation: passed.
- Fresh isolated-cache builds: two.
- Compiled specializations per build: B1 and B4.
- Duplicate generated output trees: byte identical.
- B1/B4 cubin, PTX, SASS, resource, and non-launch metrics: identical.
- Target and producer: SM121a, `ptxas-blackwell` 12.9.86.
- Stack, local memory, LDL, STL, shared memory, and calls: zero.
- Source tree whitespace/error check: passed.
- CUDA visibility during codegen: explicitly empty.
- Live correctness, timing, SWE task, service, and production selection: not run.
