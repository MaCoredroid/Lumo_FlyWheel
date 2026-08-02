# Descriptorless fixed-base SFWD real B1 byte pass

Status: **PASS**.

Source commit `d023e7bcc4990f59f33218fb41dcc3b3de8f8abe` passed the
required reference-served real SWE-Verified K64/root1 B1 byte gate. The task
resolved cleanly.

The candidate ran once per layer using one row32/C64 launch. Its exact live
layout was padded `x` stride `[16384,1]`, dense output and source-stage stride
`[10240,1]`, and convolution-weight stride `[4,1]`. The kernel carried no
source-descriptor pointer.

Across 21,504 layer invocations and all 48 layers, the gate compared both
`conv_out` and `commit_source_stage`: 43,008 surface instances and
29,947,330,560 bytes. Every byte matched, with no shape or dtype mismatch. The
reference tensors remained the only served outputs and commit sources.

Source, runtime, and external manifests were identical at launch and end.
Docker and the GPU were clean after teardown.

This establishes B1 byte correctness only. It does not provide candidate
timing, B4 evidence, production qualification, or hardware-floor acceptance.
The next valid performance step is a matched real B1 stock/candidate timing
pair, followed by canonical exact4 B4 qualification.

This reduced package contains only the validated summary, source manifest,
derived verification, and checksums. It excludes raw logs, model/task content,
requests, responses, patches, environment values, process/container
identifiers, and secrets.
