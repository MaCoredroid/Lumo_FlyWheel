# Fixed32 FA2 qrow32 B1 interleaved-K/V SM121a candidate

Status: **offline build and ABI audit pass; real SWE-Verified byte parity and
timing remain pending**.

The attempt-5 live diagnostic showed the actual B1 key/value views use block
stride 2,097,152 elements. Their row/head/element strides remain
1,024/256/1, and the value base is 1,048,576 elements after the key base in
the backing `[block, K/V, token, head, dim]` allocation.

This revision changes the B1 no-split and split2 static page resolver plus
their launcher guards to that exact layout. The shared header assertion is
layout-specific: B1 (`kStaticSequences == 1`) requires the doubled stride,
while the existing B4 and qrow16 layouts remain at their prior exact stride.
Tests construct the real interleaved storage views, verify address mapping,
and reject the old compact-page B1 geometry.

The pinned SM121a library is
`07e02c0a53185c48d745fb221e7c807f97bfe40f61354e4242e9271e743e13c1`
(300,140,712 bytes). It is stored locally with the build attestation and is
not committed to Git. The canonical source closure is
`a4a6d96cad9b34b73ddc4fb2fcda230c033b30246509c1a24208b2f2955d2bcc`.

The build used the pinned vLLM image with network disabled and no GPU device.
Dynamic defined symbols, undefined symbols, and `DT_NEEDED`/`RUNPATH` match
the qualified qrow16 reference exactly. Both new host launchers are local,
and all three compiled kernels are spill-free.

No speed, byte-parity, or hardware-floor claim is made here. Admission still
requires the real B1 SWE-Verified byte gate, followed by the standing exact4
full-step timing campaign.
