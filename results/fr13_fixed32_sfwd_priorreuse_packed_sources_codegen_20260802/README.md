# Packed-source fixed32 SFWD codegen

Status: **OFFLINE_CODEGEN_PASS_RUNTIME_UNQUALIFIED**.

Source commit `c6300f58ac2361e6ad63a2c463bdb137c93ffb21` adds a separate
packed-source SFWD kernel. It replaces three repeated topology division and
modulo chains with one compile-time table of eight 64-bit words. Each word
contains four source-delta entries; the decoder reconstructs all three
historical convolution rows without a runtime descriptor pointer.

The source-level decoder matches the established descriptorless mapping for
all 32 physical nodes. The existing descriptorless fixed-base kernel remains
unchanged and is still the only launch-integrated candidate on its branch.

## Offline result

Both candidate schedules keep 160 CTAs per request and compile to identical B1
and B4 binaries. Separate fresh caches reproduce those binaries exactly.

| Schedule | Threads | Registers | Static / encoded SASS | LDG / STG | Cubin |
|---|---:|---:|---:|---:|---:|
| incumbent row32/C64/W8 | 256 | 40 | 718 / 736 | 37 / 20 | 51184 B |
| packed row32/C64/W8 | 256 | 40 | 677 / 688 | 40 / 20 | 49432 B |
| packed row32/C64/W16 | 512 | 44 | 405 / 416 | 24 / 12 | 33824 B |

At the apples-to-apples W8 schedule, packing removes 41 static and 48 encoded
instructions while holding the 40-register allocation and zero-spill contract.
It also exposes three additional static LDG instructions, so the smaller body
is not sufficient to claim a runtime win.

W16 halves the elements handled per thread and increases the register request
to 44. Its shorter static body cannot be compared as total execution work
without accounting for twice as many warps. Real byte equivalence and matched
B1/B4 timing must choose the schedule.

## Reproduction

Run `offline_codegen_audit.py` four times with `CUDA_VISIBLE_DEVICES=`: B1/B4
for W8 and W16, each in two fresh cache/output trees. Use source revision
`c6300f58a`, row group 32, block C64, state length 34, and the exact canonical
source path in this checkout. Then run:

```bash
/home/mark/fr13_streamk_build/venv/bin/python \
  verify_codegen_outputs.py \
  --w8-primary /tmp/fr13_packed_w8_primary \
  --w8-rebuild /tmp/fr13_packed_w8_rebuild \
  --w16-primary /tmp/fr13_packed_w16_primary \
  --w16-rebuild /tmp/fr13_packed_w16_rebuild
```

This reduced package excludes cubin, PTX, SASS, IR, raw logs, task/model
content, requests, responses, patches, environment values, process/container
identifiers, and secrets.
