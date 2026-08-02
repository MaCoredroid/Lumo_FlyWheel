# Fixed32 SFWD frontier-5 source audit

Status: **SOURCE_AUDIT_PASS_CODEGEN_AND_REAL_GATES_REQUIRED**.

Source commit `ac8d848b63278a9c956ebbb31b9b7836372816f1` replaces the
late-tap0 reload schedule with a load-once permutation of all 32 fixed-tree
nodes. The selected order has a peak live-current-row frontier of 5 and a
live-frontier sum of 116, versus 11 and 230 for natural node order. It issues
exactly 32 current-row loads per channel and no reloads.

An exact memoized search visited 2,257 partial schedules and found no
load-once schedule with peak frontier 4. The selected schedule attains peak 5,
so the peak frontier is optimal for this fixed topology and load-once model.
The source tests recover every tap operand from the kernel AST and require the
original ordered BF16 product rounding, left-to-right FP32 accumulation, SiLU
expression, output row, and source-stage row for every node.

The output execution order is permuted, so the launcher now rejects shared
storage among input, output, and source-stage tensors. The current production
wiring allocates those surfaces independently.

This is a source-only result. No compiler, Docker container, GPU kernel, task,
request, timing, throughput, acceptance, or hardware-floor run was used.
Offline SM121a codegen, a real SWE-Verified byte gate, and then real-task
timing are still required.

This package excludes raw task/model content, requests, responses, patches,
logs, environment values, process/container identifiers, binaries, PTX, SASS,
credentials, and secrets.
