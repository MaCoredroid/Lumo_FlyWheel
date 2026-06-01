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
  pre-commit spine 0 under sync at `avg=3.327777777777778`, `acc0=0.0`.
  The same direct-probe run produced `47.50160804227232` completion tokens/sec.
- Apples-to-apples direct-probe E5-equivalent checkpoint
  (`--row-mode independent --spines 1`, same 16-prompt fixture, `--limit 64`,
  temperature 0): `avg=2.7720739219712525`, `acc0=0.16221765913757702`,
  `full5=0.33744010951403147`, `44.454399711901914` completion tokens/sec.
  Direct-probe winner speedup is therefore `1.0685468333869903x`.
- Independent rows are persistent. The current winner-commit path chooses the
  longest accepted row each event, commits that row's sampled tokens to every
  sibling request, and copies the winner's post-accept GDN recurrent state back
  into all co-resident rows through vLLM's existing Mamba state-copy path.

Retired routes remain out of the public surface: `LUMO_FB_*` inject/collapse,
tree-delta, internal rows, kernel rows, and force-capture flags are not part of
the maintained launcher interface.

## Closeout

FR9's ship-worthy result is independent rows plus winner commit and recurrent
state sync. The temp-0.6 agentic-B4 winner run met the operator stability bar:
`fr9_agentic_b4_winner_temp06_sync_20260601T1800Z` ran to the agent-wall
timeout without engine death, CUBLAS/CUDA/illegal-memory, or shutdown
signatures. The rejection-sampling superset invariant held under production
load with co-resident rows: `viol=0`, `missing_sum=0` over 19,307 winner
events.

Keep the three FR9 results separate:

- Greedy superset proof: the deterministic 16-prompt gate proves the invariant.
  Winner commit produced `avg=3.441666666666667` versus E5 `avg=3.002` from
  the prior gate, with `superset_violations=0` and spine 0 remaining
  E5-equivalent (`LCP 64/64`).
- Greedy apples-to-apples direct probe: E5-equivalent direct probe measured
  `44.454399711901914` tok/s and `avg=2.7720739219712525`; winner direct probe
  measured `47.50160804227232` tok/s and `avg=3.441666666666667`. The clean
  speed result is `1.0685468333869903x` (`+6.85%` tok/s), with higher
  deterministic accept.
- Temp-0.6 agentic-B4 stability: commit `079d51f4` validates that winner commit
  and recurrent-state sync survive production rejection sampling. The bonus
  numbers from this run, `decode_tps=10.65`, steptrace accept/event `0.402`,
  and winner-trace acc/event `1.398`, are workload-confounded. All four SWE
  agents gave up (`resolved_rate=0/4`), so the decode window is dominated by
  agent flailing, retry behavior, low-predictability tokens, and B4 contention.
  These numbers must not be compared to the E5 agentic baseline
  `26.86` tok/s / `3.150` accept/event.

The in-tree route is closed for FR9. Unified tree mode capped at free-running
path0 `1.751` versus E5 `3.002`, consistent with shared-GDN token-tree scan
contamination. STree-style `A_tree` kernel/shared-ancestor reuse remains the
scale endgame, but it is a no-ship item for this cycle and was not built.
