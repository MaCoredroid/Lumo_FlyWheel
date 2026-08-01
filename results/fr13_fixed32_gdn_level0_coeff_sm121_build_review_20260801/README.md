# Fixed32 GDN coefficient staging: reviewed SM121 build

This is an offline, source-bound build of the production specialization after
the independent review corrected the `COUNT_INVOCATION` mismatch. Production,
the live gate, and this build all use `COUNT_INVOCATION=False`.

The first exact-production build exposed `STACK:16` and five `LDL`/`STL`
instructions in B1 candidate level 0. Moving the independent coefficient
staging before the level-0 recurrence removed that spill without adding a
launch or changing the scratch layout. The final resource summary is:

| Specialization | Registers | Stack | Local | Spill instructions |
| --- | ---: | ---: | ---: | ---: |
| B1 stock level 0 | 64 | 0 | 0 | 0 |
| B1 stock level 1 | 80 | 0 | 0 | 0 |
| B1 candidate level 0 | 77 | 0 | 0 | 0 |
| B1 candidate level 1 | 64 | 0 | 0 | 0 |
| B4 stock level 0 | 77 | 0 | 0 | 0 |
| B4 stock level 1 | 80 | 0 | 0 | 0 |
| B4 candidate level 0 | 79 | 0 | 0 | 0 |
| B4 candidate level 1 | 64 | 0 | 0 | 0 |

- Kernel SHA-256: `16fde18ebf4ace9893d2f8890294c894c71222b85d7c9cdc4bc7789cf5afff4e`
- Builder SHA-256: `74e37ea630786f5a6c08d37e05bb8bfad4be2fcd305daccaab82a67c92d8da97`
- Target: `sm_121`
- Triton: `3.6.0`
- CUDA offline tools: `13.0`

`build_manifest.json` binds every CUBIN, PTX, TTGIR, SASS, and resource report.
`SHA256SUMS` covers the complete directory. No GPU was used. This artifact is
not a live byte PASS, a B4 production qualification, or timing evidence.
