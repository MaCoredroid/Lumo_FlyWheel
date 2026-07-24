# FR13 spec-page reservation surgery — design (2026-07-24)

**STATUS (2026-07-24 post-reboot): BUILT, gates queued (regate_queue.sh 2e/2f).**
- Piece 1+2 = FR13_CONV_NODEBANK (patch-time baked const): writeback -> bank
  mirror + pool col0-only; remap src -> bank via
  `replay_conv_state_linear_remap_from_bank` (invalid lanes substitute
  current dst bytes — bank-space has no identity self-copy); stateless
  committer leaf read -> bank (bank-width clamp, not spec_cols); committed-
  path prior consumer is col0-only under RUNROW_INIT=1 => untouched
  (RUNROW_INIT!=1 + nodebank = patch-time raise, ditto TCF_SELFCHECK and
  FULL_CAPTURE). NEW HAZARD FOUND + CLOSED at design time: the bank is
  ORDINAL-keyed while the remap consumes PREV-step deposits — composition
  change re-points via `ordinal_perm`, a persistent device buffer host-
  refreshed each step in _prepare_inputs (outside capture). Offline route
  byte gate `output/fr13_msr/gate_conv_nodebank_byte.py`: CPU PASS 36/36
  (dtypes x seeds x ordinal-perms, invalid lanes, nacc=0).
- Piece 3 = FR13_SPEC_BLOCKS_CAP=12 (patch-time): abstract.py construction
  min() + gdn_attn page-col width caps (_fr13_page_cols; token-count uses of
  num_spec stay uncapped) + _fr13_tcf_cols follows. Requires nodebank
  (patcher main raises). Write-never on vacated pool node cols is enforced
  STRUCTURALLY by the cap (cols cease to exist); pre-cap the route byte gate
  covers the contract, so no runtime write-guard instrument is added
  (observer-effect discipline).

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
