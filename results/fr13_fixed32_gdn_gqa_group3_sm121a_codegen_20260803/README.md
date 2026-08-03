# Fixed32 GDN GQA-group3 SM121a codegen

Status: **offline resource gate PASS; candidate remains credential-gated and
requires real SWE-Verified byte and timing gates**.

Source revision `936dd110c01d34f8c1c5c64676dde5739d0d2fa3` compiles the
exact physical32, K64/root1, BV8 GDN GQA-group3 B1 and B4 production
specializations for `sm_121a`. The audit also compiles the incumbent at the
same 16-key-head, 48-value-head geometry. Both the base production closure and
the established K-norm/gate/decay committer-stack closure are covered.

## Resource result

| Profile | Kernel | Registers/thread | Register bytes/CTA | Stack/local | LDL/STL/calls | Grid B1 / B4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | Incumbent | 80 | 81,920 | 0 / 0 | 0 / 0 / 0 | 768 / 3,072 |
| Base | GQA3 | 120 | 122,880 | 0 / 0 | 0 / 0 / 0 | 256 / 1,024 |
| Committer stack | Incumbent | 80 | 81,920 | 0 / 0 | 0 / 0 / 0 | 768 / 3,072 |
| Committer stack | GQA3 | 126 | 129,024 | 0 / 0 | 0 / 0 / 0 | 256 / 1,024 |

Every build uses 256 threads, resolves the unset Triton stage request to three
stages, reports 16 launch shared bytes and 1,024 ELF reserved shared bytes,
and emits no device calls. B1 and B4 have identical resource counts within a
profile. Two independent fresh-cache builds are byte-identical across all
eight cubins, PTX files, SASS listings, compiler IR files, and summaries.

## Spill fix

The inherited `maxnreg=80` cap was invalid for the three simultaneously live
value-head states when K-norm/gate/decay export is enabled. It compiled with a
16-byte stack frame plus four `LDL` and three `STL` instruction sites at both
B1 and B4. GQA3 now uses an explicit 128-register cap for that profile; ptxas
settles at 126 registers with zero stack, local memory, `LDL`, or `STL`.
The incumbent retains its established 80-register cap.

## Work interpretation

GQA3 increases per-CTA registers by 50% in the base profile and 57.5% in the
committer-stack profile. The reduction premise therefore cannot rest on CTA
count alone. Multiplying static instruction sites by the exact grid still
decreases the base SASS proxy by 36.9%, LDG sites by 39.7%, and STG sites by
18.2%. In the committer profile the corresponding reductions are 37.7%,
39.7%, and 19.6%. These are static codegen proxies, not latency or throughput
measurements.

## Decision

The candidate passes the offline compile, launch-image, and spill gates. It is
not performance-promoted by this artifact. No GPU kernel was launched, no
serving process or task ran, and no acceptance, TPS, full-step wall time, or
hardware-floor measurement was collected. The remaining authority is the
credential-bound real SWE-Verified B1/B4 byte gate followed by the standing
four-task timing campaign and 16-task confirmation if the U95 cap clears.

Only sanitized summaries and reproduction code are checked in. Cubin, PTX,
SASS, compiler IR/cache, raw logs, task/model/request/response/patch content,
credentials, environment dumps, process IDs, and container IDs are excluded.
