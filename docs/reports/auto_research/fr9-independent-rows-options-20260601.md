# FR9 Fb Modes

`scripts/swe_x86_helpers/relaunch_qwen36_round.py` keeps one public config,
`--config Fb`, with two row modes:

- `--row-mode tree`: consolidated in-tree `speculative_token_tree` path. `--mtp`
  is logical depth; `--spines` is the number of regular root-to-leaf spines.
  This mode emits `tree_path_lcp_max.jsonl` and is measured with
  `measure_spec_per_position.py --mode tree`.
- `--row-mode independent`: native linear MTP, no `speculative_token_tree`, no
  TreeAttention, no custom per-path GDN scan. The launcher creates hidden
  co-resident request rows with `LUMO_INDEPENDENT_ROWS=1` and
  `LUMO_IR_SPINES=N`; spine 0 is the native top-1 chain and later rows use
  rank-s root tokens before continuing through the normal linear MTP drafter.
  The public client sees only spine 0 output while hidden rows stay scheduled
  until request finish. This mode is measured with
  `measure_spec_per_position.py --mode independent --spine 0`.

Operational defaults for independent rows:

- Use `LUMO_GPU_MEMORY_UTILIZATION=0.84`.
- Keep `VLLM_BATCH_INVARIANT=1` through the launcher and FLASH_ATTN.
- Set `max_num_seqs >= user_concurrency * spines`; the launcher defaults to
  `4 * --spines` when `LUMO_VLLM_MAX_NUM_SEQS` is unset.
- `--spines` is supported for 1 through 10; `--spines 1` is the E5-equivalent
  native chain.

Current FR9 checkpoint:

- Unified in-tree `--row-mode tree --mtp 5 --spines 2` free-running path0:
  `avg=1.7514863258026159` over 1682 events.
- Independent rows `--row-mode independent --mtp 5 --spines 2`, spine 0:
  `avg=2.8659638554216866`, `acc0=0.15060240963855423`,
  `full5=0.3569277108433735` over 1328 events from the 16-prompt fixture
  (`--limit 64` covers the full fixture).
- Winner-commit checkpoint after adding recurrent-state sync:
  `avg=3.441666666666667`, `acc0=0.0`, `full5=0.4777777777777778`,
  `superset_violations=0`, `recovery_rate=0.049305555555555554`, with
  pre-commit spine 0 at `avg=3.327777777777778`. The same run produced
  `47.50160804227232` completion tokens/sec versus the E5 reference
  `26.86`.
- Independent rows are persistent. The current winner-commit path chooses the
  longest accepted row each event, commits that row's sampled tokens to every
  sibling request, and copies the winner's post-accept GDN recurrent state back
  into all co-resident rows through vLLM's existing Mamba state-copy path.

Retired routes remain out of the public surface: `LUMO_FB_*` inject/collapse,
tree-delta, internal rows, kernel rows, and force-capture flags are not part of
the maintained launcher interface.
