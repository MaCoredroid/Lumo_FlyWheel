# Fixed32 DFWD K64 M4 R64-U8 linked build

Status: **reproducible canonical linked build, default off and runtime
unwired**. No GPU kernel execution, SWE-Verified task, byte gate, timing,
acceptance, or production admission was performed.

The reviewed source was linked twice from empty build directories with
`CUDA_VISIBLE_DEVICES=''`, PyTorch `2.11.0+cu130`, CUDA 13.0, and target
`sm_121a`. The builder pins
`--frandom-seed=fr13_bf16_k64_m4_r64_u8`. Both embedded SM121a cubins are
byte-identical and pass the reviewed static codegen checker.

The two raw host ELFs differ in exactly six bytes, all inside the non-loadable
`.strtab`: nvcc writes its PID-bearing `tmpxft_*` filename into two local
symbols. Their GNU build IDs and every other byte are equal. Applying GNU
binutils 2.42 `strip --strip-unneeded` removes that local symbol table and
produces byte-identical loadable libraries. Both canonical libraries were
loaded independently under the pinned Torch ABI and registered
`fr13_bf16_k64_head::gemvx_m4_shuffle_r64_u8_out` without a GPU.

## Build result

- Source SHA-256: `a52361be1c9052a46509cc230ea320c4beb6d15f261327edc835d8da3ae00d9e`
- Canonical shared-object SHA-256: `6cb24782495ff1c1457ebbf9cbcfcd6ca7b372378d3b435f80054688432a365f`
- Canonical shared-object bytes: `134320`
- Linked cubin SHA-256: `952395db481f7c1b8d8c631789d961f41c6ec8d3cbb85c80b9ee5f1b2371a4e1`
- Registers/thread: `56`
- Stack/local/shared bytes: `0/0/0`
- Registered op: `fr13_bf16_k64_head::gemvx_m4_shuffle_r64_u8_out`

The shared object is omitted from Git, matching the B1 linked-build precedent.
The retained gate input is
`/home/mark/shared/fr13_dfwd_m4_u8_linked_build_bb8a4a8a2_20260805/canonical-primary-bin/fr13_bf16_k64_m4_r64_u8.abi3.so`.

## Next default-off gate

The next change must use a separate `FR13_DRAFT_HEAD_M4_R64_U8_*` family and
remain mutually exclusive with every existing draft-head mode. Wire it through
these existing runtime boundaries:

1. In `scripts/fr10_phase4_patch_vllm_tree_gdn.py`, admit only exact B4,
   Hydra27 fixed32, physical32, K64/root1, contiguous BF16 input
   `[4,5120]`, weight `[65536,5120]`, and output `[4,65536]`. Load the pinned
   canonical SO, verify its source/build/hash closure, and preallocate one
   `[4,65536]` candidate output.
2. At `_fr13_dvk_logits`, run stock first and the M4 op second at all five head
   sites: root and MTP depths 1 through 4. Compare raw BF16 words with
   `view(torch.int16)` and always return the stock logits object.
3. Extend the fixed32 GDN lifecycle/flush record with per-depth comparison and
   mismatch counters. Every complete event must contribute exactly one
   comparison at each of the five depths, with zero mismatches and no fallback.
4. In `scripts/fr13_launch_forked_fa2_tree_server.sh`, add fail-closed caller
   guards, worker-env bridge keys, immutable source/SO/build/subset/block-map/FA2
   identities, and a read-only mount at
   `/tmp/fr13_bf16_k64_m4_r64_u8.abi3.so`.
5. Add `scripts/fr13_run_b4_dfwd_k64_m4_r64_u8_live_gate.sh` and a strict
   reducer. The runner must use canonical SWE-Verified exact4
   `config/fr13_fixed32/subset_b4_four.json` at `BSIZE=4`, `CONC=4`,
   `FULL_AND_PIECEWISE`, K64/root1, and stock FA2. For `E` complete events the
   reducer must require `5E` full-logit comparisons, `20E*65536` compared BF16
   elements, zero mismatches at every depth, all four exact task IDs completed,
   incumbent served, and no timing or acceptance claim.

Only after that real exact4 five-site byte gate passes may a separate
source-bound production selector be added and timed. The diagnostic selector
itself is not timing eligible.
