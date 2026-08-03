# Fixed32 tree-conv zero-tail SM121a codegen

Status: **default-off candidate passes offline source, route, and SM121a
codegen gates; real-task byte and timing qualification remain required**.

This audit covers the fixed32 tree-conv commit path for the deployment context
of 32 physical rows, K64 drafter vocabulary, and root reduction on. K64 and
root reduction do not alter this tree-conv kernel specialization; they are
recorded here to bind the intended end-to-end route.

The candidate source revision is
`0112ac7c49188baa6ab44bb9d9a832423520d8b7`. Its pre-change incumbent is
`b7fc9c594de58d5c38f7ad2da31a262d3cab7669`.

## Route

The fixed32 patch guards the retained generic batched writeback kernel out of
this route. Accepted-leaf commit uses one direct launch per event with:

- 48 layers, BF16, 10,240 channels, state length 34, source rows 36
- 32 physical rows per request and `BLOCK_C=1024`
- B1: 480 CTAs/event; B4: 1,920 CTAs/event
- zero full-node writebacks and zero fixed32 conv remaps

Logical Tail/Hydra active-node counts therefore do not change this physical32
launch geometry. B4 still scales by four because four requests have distinct
destination state.

The candidate replaces source loads for state columns 3 through 33 with BF16
zero while retaining all 34 destination stores. It is fail-closed: arming is
accepted only for the exact fixed32/BF16/10,240/34/36 contract and when every
physical-row tail source index points to the known zero row. The selector is
strict `0|1` and defaults to `0`.

## Codegen result

Two isolated, empty-cache builds compiled both direct kernels for B1 and B4 to
actual `.target sm_121a`. All candidate variants have zero stack, local
memory, LDL, STL, and calls.

| Kernel | Batch | Incumbent LDG/STG | Candidate LDG/STG | Registers inc/cand |
|---|---:|---:|---:|---:|
| direct | B1 | 311 / 272 | 32 / 272 | 64 / 64 |
| direct | B4 | 311 / 272 | 32 / 272 | 64 / 64 |
| metadata | B1 | 312 / 274 | 33 / 274 | 64 / 64 |
| metadata | B4 | 312 / 274 | 33 / 274 | 72 / 64 |

The source-read model falls from 34 to 3 columns per committed row, a
31/34 or 91.176% reduction. Full destination stores are unchanged.

| Batch | Incumbent global bytes/event | Candidate global bytes/event | Saved |
|---|---:|---:|---:|
| B1 | 66,846,720 | 36,372,480 | 30,474,240 |
| B4 | 267,386,880 | 145,489,920 | 121,896,960 |

At the campaign's 273 GB/s reference bandwidth, those saved bytes correspond
only to traffic lower bounds of 0.111627 ms/event at B1 and 0.446509 ms/event
at B4. They are not latency predictions or measured hardware-floor progress.

Selector-off disassembled machine instructions and resource reports are
byte-identical to the pre-change incumbent in all four specializations. PTX
and cubin container hashes differ because the new constexpr source signature
and debug metadata are present, so the verifier does not mislabel container
identity as machine-code identity.

For context only, the retained generic kernel compiles to 48 registers with
zero spills, 307 LDG, and 272 STG, but would require 15,360 B1 or 61,440 B4
CTAs/event and model 2.139 GB or 8.556 GB/event. It is not selected by the
fixed32 route.

## Decision

Keep the optimization default off until the standing real SWE-Verified B1/B4
byte gate passes on the fixed32 K64/root1 route, followed by full-step timing
and breakdown measurement on that same task set. This offline result neither
qualifies acceptance nor establishes TPS or the 1.15x hardware-floor target.

The checked package contains reproduction scripts and reduced summaries only.
It excludes cubin, PTX, SASS, compiler caches, raw logs, tasks, prompts,
responses, patches, credentials, environment dumps, process IDs, and
container IDs. No GPU kernel, service, task, request, timing run, or acceptance
run was launched by this audit.
