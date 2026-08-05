# Fixed32 B1 FA2 truthful-reference v5 SM121a build

Status: **offline binary pass; real SWE-Verified byte qualification pending**.

The v4 combined B1 binary contained the qrow16 CUDA object, but its host API
did not dispatch the private qrow16 sentinel. That made the intended reference
arm untruthful: it could fall through to stock FA2. V5 adds the missing hidden
host dispatch for sentinel `1179791667` on the exact B1 geometry. The existing
qrow32 no-split and split2 sentinels remain `1179791668` and `1179791669`.

The qrow16 reference gate requires batch one, 32 physical query rows, BF16,
24 query heads, four KV heads, head dimension 256, `num_splits=0`, and forced
split dispatch. It also checks the real interleaved B1 K/V page stride of
`2 * 1024 * 4 * 256`; the retained qrow16 CUDA object consumes runtime
strides. Incorrect geometry fails closed before launch.

## Build boundary

- Repository commit: `0cb1664cba90c20b2f4e2b5ebae2544d876a0c6d`.
- FA2 source commit: `29210221863736a08f71a866459e368ad1ac4a95`.
- V5 source was copied from immutable v4 source.
- A full tree comparison found exactly one changed file:
  `csrc/flash_attn/flash_api.cpp`.
- Only the host `flash_api.cpp` object was recompiled.
- The qrow16, qrow32 no-split, and qrow32 split2 CUDA objects were retained
  byte-for-byte.
- Relinking used the pinned vLLM image with the network disabled and no GPU
  device exposed.

The candidate shared object and intermediate objects are intentionally not
committed. Their SHA-256 identities and sizes are in `manifest.json`.

## Offline qualification

The public defined and undefined dynamic symbol sets, `DT_NEEDED`, and
`RUNPATH` match the pinned qrow16 reference shared object. All three private
launchers have `LOCAL` binding. A GPU-disabled `torch.ops.load_library` call
succeeded and registered `varlen_fwd_tree_bias`. The source-closure validator,
candidate identity validator, Python compile checks, shell syntax checks, and
focused repository tests passed.

## Admission boundary

This artifact contains no GPU run, task run, output comparison, LSE
comparison, timing, speedup, or hardware-floor result. It makes no byte-parity
or performance claim. Real SWE-Verified retained-operand qualification remains
required before the qrow32 arm can be admitted or timed.
