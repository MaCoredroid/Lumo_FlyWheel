# Why option 1 (snapshot-side fix) failed — the cohort-coverage wall (2026-06-20)

Branch `fr13-prefix-cache`. This writeup explains, structurally, why every variant of the
**snapshot-side** APC SSM fix could not make the GDN tree-spec serve lossless — and why that
failure mode points directly at option 3 (commit-site write-through). One sentence: **most of the
align's snapshot copies never pass through the code path we overrode, so the fix covers ~4% of
the copies that actually matter.**

---

## The two places state lives, and the one place the snapshot looks

- **Node bank** (`spec_state_indices[b, accepted_len-1]` == `layer._fr13_replay_spec_idx[b][-1]`):
  where the GDN tree committer writes the **accepted-leaf recurrent state** every decode step, and
  where the next-step regular decode reads it back. This is the *real* working state.
- **Block pool** (`state[block_ids[cur_block_idx + num_accepted - 1]]`): an APC-addressed row,
  chosen by block-boundary / bias arithmetic. **The only consumer of this row is the APC align
  snapshot.** The tree committer never writes here; the decode never reads here.

vLLM's `align` cache (`get_temporal_copy_spec`) snapshots/restores the **block-pool** row. For the
tree committer those two row-spaces disagree at `num_accepted > 1` → the snapshot saves a stale row
→ on the next cache hit the GDN recurrent state is restored from garbage → runaway garble → the SWE
agent emits an empty patch and gives up. (Cache-OFF runs the full wall and produces a real patch —
that is the black-box ground truth we gate on.)

## What option 1 tried

Option 1 = fix it **at the read side**: make the snapshot read the committed-leaf node-bank row
instead of the stale block-pool row. Four concrete attempts, each fixed the previous one's bug and
exposed the next:

1. **Module-global override** of `get_temporal_copy_spec` reading `_FR13_CUR_SSM_LEAF_ROW`. *Never
   landed* — the align batches **all** reqs' bias-chokepoint calls and *then* all the copies, so the
   single global is clobbered by the last (found=False) bias before the override reads it
   (`FR13_OV_DIAG`: `bare_leaf=None` for every override call even with the leaf set right before it).
2. **Direct copy_spec pointer substitution** in `collect_mamba_copy_meta` (the substitution executes
   before the Tap-C read, mechanism proven). Fired in **preprocess** but not the **postprocess**
   snapshot. Black-box: agent still gave up empty.
3. **Leaf off-by-one** corrected: `_FR13_APC_SSM_LEAF_BY_REQ` held `+1` node ids (81 vs committed
   80); switched the source to `_FR13_BOUNDARY_LAST_WRITTEN_BY_REQ[req].rows[-1]` (the actual written
   bank rows). Still didn't move the black-box.
4. **Steady-state per-copy diagnostic** (the decisive one): over a periodic sample, the committed
   leaf was available for **2 of 50** temporal copies; for the other **48/50 the leaf was `None`**,
   so the substitution fell through to the stock stale row. When it *did* fire it was numerically
   correct (leaf=80 in `last_written` vs stock 87) — but it fired ~4% of the time.

## The structural reason — "most don't hit the path we work on"

The snapshot-side override can only act on a copy **if that copy's request is in the committed-leaf
map** at the moment the align processes it. But the align does **not** iterate the same population
the committer does:

- **The committer** runs once per **committed req per decode step** — a small, well-defined set, and
  it's exactly the set whose leaf we know.
- **The align snapshot** iterates a **block-boundary cohort**: every req whose block-alignment /
  state-index transition condition trips at this step (`preprocess`: `prev_state_idx != curr_state_idx`;
  `postprocess`: `aligned_new_computed_tokens >= num_tokens_running_state`). That cohort is dominated
  by copies whose req **is not in the committed-leaf map at that instant** — different reqs, different
  phase, or boundary-only bookkeeping copies with no live commit. Hence `leaf=None` for ~96% of them.

So the override sits on a path that **most of the poison-carrying copies simply do not take.** No
amount of fixing *the override's* correctness (keying, off-by-one, race, pointer-vs-tap order — all
fixed) changes the fact that the override is only *reached* for a small minority of the copies that
need it. Patching the read side is patching the wrong end of the pipe: you can make the faucet
perfect, but most of the water is coming through the other 24 faucets you don't control.

This is also why every white-box "stale_read→0" signal was misleading: the tap fires on the copies
we *do* reach, looks clean there, and is blind to the 48/50 we never touch. The black-box (agent
gave_up_empty) was the only honest gate, and it stayed red.

## Why this points straight at option 3 (commit-site write-through)

Flip the side. Instead of teaching the **read** (align) to find the committed state, make the
**write** (commit) deposit the committed state into the exact row the **unmodified** align will read.

- The commit runs **per committed req, every decode step** — the complete population, the one place
  we always have the leaf. There is **no cohort to miss**: if a req committed, its block-pool row was
  synced at commit, so whenever the align later snapshots that row it reads correct state.
- The align stays **stock** — no race, no keying, no per-copy coverage gamble. We're not trying to
  intercept 50 heterogeneous copies; we're maintaining one invariant ("the block-pool row mirrors the
  committed leaf") at the single site that sees every commit.
- This is also what **SGLang** effectively does: it snapshots the **real working-state slot at the
  commit/cache point, per request**, never a block-addressed/reconstructed row
  ([[sglang_mamba_radix_cache_design]]). Option 3 is the within-vLLM-align analog of that principle.

The one thing option 3 must prove (and option 1 never had to): that `block_ids[cur_block_idx +
num_accepted-1]` — the row the align *will* read — is **knowable/stable at commit time**, and that
the block-pool row is read **only** by the align (so overwriting it can't corrupt the decode). Those
are exactly the questions the commit-writethrough design workflow (wgzvsr2h2) is pinning before we
implement. If block_ids isn't stable commit→align, the fallback is to write the leaf into whatever
slot the align resolves, but the *coverage* property (all committed reqs, at the source) holds
either way.

## One-line takeaways
- Option 1 failed on **coverage, not correctness**: the override is reached for ~4% of the copies
  that carry the poison; the align's block-boundary cohort ≠ the committer's per-req population.
- White-box stale_read was blind to the 96% we never reached; only the black-box (empty give-up) was
  honest.
- Option 3 moves the fix to the **write side**, which sees **every** commit → cohort-complete by
  construction, align stays stock → mirrors SGLang's commit-time snapshot of the real working slot.

Cross-refs: [[apc_ssm_carrier_deep_findings]], [[apc_ssm_drilldown_design]],
[[sglang_mamba_radix_cache_design]], [[project_fr13_conv_priorwindow_root]].
