# FR13 spec-page reservation surgery — design (2026-07-24)

Goal: cut tail6's per-request mamba spec-page reservation from 22 pages to
~13 (max accepted path 12 + col0), reclaiming ~9 pages/request of cached-
history retention (measured root cause of the 71%-vs-85% hit gap and the
0.33-vs-0.13 prefill_frac residual vs native at 0.70).

Measured basis: vLLM reserves `num_speculative_blocks = num_speculative_tokens`
mamba pages per running request (abstract.py); consumer sweep shows the tree
uses them as (a) conv NODE deposits (all 21 cols, tiny payloads on padded
pages), (b) post-remap LINEAR cols 0..nacc-1 read by stock consumers
(nacc <= 12 on tail6), (c) col0 persistent. The waste is (a): kilobyte conv
windows pinning full pages.

## Decomposition (each piece independently gateable)

Piece 1+2 (one flag, FR13_CONV_NODEBANK — they must land together):
- Own bank `[B, N_TREE, C, W]` per layer-group (small: 4 x 21 x conv-window),
  allocated once, module-held.
- Conv WRITEBACK retarget: launch_conv_state_writeback dst = bank rows
  instead of spec-slot pages (kernel dst param swap; payloads identical).
- Conv NODE-SOURCE retarget: every consumer that READS node cols pre-remap
  reads the bank: (i) the linear remap's conv gather (new two-tensor variant
  of _linear_remap_rows_gather_kernel: src=bank, dst=pool linear cols);
  (ii) CONV_COMMITTED_PATH prior-bank gather; (iii) FULL_CAPTURE diag.
- NPR/pregather (col0 read) untouched: col0 stays in-pool.
- Byte gate: offline — same inputs, bank path vs pool path, compare the pool
  linear cols + prior windows bit-for-bit. Pool pages for node cols become
  WRITE-NEVER under the flag (assertable).

Piece 3 (FR13_SPEC_BLOCKS_CAP, only after 1+2 gate PASS):
- Patch MambaSpec construction: num_speculative_blocks =
  min(num_spec_tokens, MAX_PATH(=12) + 1) when the nodebank flag is baked.
- ssi/block-table columns shrink to 13; assert no consumer addresses col>=13
  (grep + runtime assert on ssi width).
- Gate: boot + pool-size line (expect ~+9 pages/request capacity), hit-rate
  probe vs baseline, accept band, 4-task screen.

## Non-goals / hazards
- SSM node states: already in our banks; untouched.
- Linear cols stay in-pool (stock reader contract preserved — the docstring
  contract that killed the earlier "cap to 2" idea).
- The conv prior-window seam history demands the write-never assert + the
  stale-token discipline from the pregather.
