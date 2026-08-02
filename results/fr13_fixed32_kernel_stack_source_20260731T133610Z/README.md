# FR13 fixed32 exact-kernel stack integration

This artifact binds the CPU/static integration of the current exact-math
kernel candidates. It makes no GPU byte-parity, speed, SWE-Verified quality,
or acceptance claim.

## Source lineage

- Live-gate base: `4220616f9684d76075b040a96aeb636ae55cb349`
- Qrow16 attested production selector: `bb399adb30f210e57084bb87c6d4d99fffd4f020`
- One-launch committer GDN layer batch: `c11323290`
- Two-launch B2-B4 GDN candidate: `f75c023c9`
- B2-B4 real-event byte gate: `d642d7067fbf1a83cf205f281cf7fde33f903137`
- B2-B4 gate evidence artifact: `e00501666`
- Wider path-BV live gate: `9b1d1f19790fcfc6441bcc24742b3a00dee39017`
- Default-off B2-B4 selector integration: `7185e3a78`

## Dispatch contract

- Qrow16 production, padded draft heads, native TAW precompute, one-launch
  committer GDN, B2-B4 batched GDN, and wider path-BV candidates are all
  default-off.
- B1 never enters the B2-B4 batched-GDN API.
- `FR13_FIXED32_BATCH_GDN_BYTE_AB=1` is the eager diagnostic selector. It
  compares all touched bytes on a real SWE-Verified event, restores state, and
  always serves the legacy per-request result, including after a layer passes.
- The diagnostic writes a production prerequisite only after 48 unique layer
  keys pass under one task marker and one batch size.
- `FR13_FIXED32_BATCH_GDN_PRODUCTION=1` is accepted only with that PASS record.
  Diagnostic and production modes are mutually exclusive.
- Wider path-BV diagnostics are mutually exclusive with B2-B4 batched-GDN
  diagnostic/production routes. BV16/32/64/128 checks all restore and serve BV8.

## Verification

```text
python3 -m py_compile \
  src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  scripts/fr10_phase4_patch_vllm_tree_gdn.py \
  scripts/fr13_patch_fa2_tree_bias.py \
  scripts/fr13_qrow16_pass_sidecar.py
bash -n scripts/fr13_launch_forked_fa2_tree_server.sh
pytest -q <10 focused gate modules>
45 passed, 1 skipped in 3.25s
pytest -q tests/test_fr13_draft_head_pad_rows.py \
  tests/test_fr13_fixed32_taw_exact_commit_kernel.py
11 passed in 1.03s
```

The skip is the pre-existing CUDA/Triton test on this no-GPU integration pass.

