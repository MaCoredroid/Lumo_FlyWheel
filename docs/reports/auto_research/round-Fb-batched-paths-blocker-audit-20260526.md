# Round F_b Batched-Paths Audit: Core Path-Clone Blocker

**Generated:** 2026-05-26  
**Scope:** Qwen3.6-27B FP8 on vLLM 0.19.0, container
`lumo-vllm-track-b-suffix`, port `9950`.  
**Status:** F_b is not shipped. The launcher/source-edit route is insufficient;
the missing piece is a vLLM core primitive for temporary path clones plus
winner-state collapse.

## Objective Restatement

Ship **F_b**: enumerate K MTP root branches as K independent linear draft paths,
verify those paths as separate FlashAttention/GDN histories sharing the prompt
prefix, run lossless rejection per path, then commit the longest accepted path.
It must:

- avoid `TREE_ATTN` and the F_a packed-tree/GDN topology problem;
- pass greedy byte-exact OFF correctness;
- beat E3 in `scripts/spec_speed_probe.py` (`decode_tps > ~17.67`,
  `acc/ev >= 2.235`);
- expose a canary proving K batched paths actually ran;
- commit and push verified changes.

## Current Evidence

- `scripts/swe_x86_helpers/relaunch_qwen36_round.py` currently exposes
  `choices=["D", "E", "F"]`; there is no `Fb` config.
- The live container is not an F_b run. Its log shows a `speculative_token_tree`
  and `Using AttentionBackendEnum.TREE_ATTN backend`.
- Existing probe artifacts under `output/spec_speed_probe/` are E3/F_a
  diagnostics only; there is no F_b probe.
- `git status --short --branch` was clean at audit time (`main...origin/main`).

## Why the Launcher Patch Route Fails

vLLM 0.19.0 has a one-request/one-spec-stream contract:

- `Request` owns one `spec_token_ids` list and one `num_tokens_with_spec`.
- The scheduler schedules one speculative token stream per request.
- `CachedRequestState` owns one `output_token_ids` history per request.
- `SpecDecodeMetadata` and `rejection_sampler.py` return one accepted sequence per
  scheduled request.
- GDN/Mamba state is keyed by request/block state index and updated in place.

F_b needs K temporary histories for one logical request. Flattening K paths into
one request recreates the F_a sibling-interleaving bug. Serially trying paths
inside one request cannot preserve independent GDN recurrent state. The existing
`parallel_drafting` hook is also not F_b: it inserts masked draft-model slots for
models with `pard_token`/`ptd_token_id`; native Qwen MTP does not provide the
required independent target histories or collapse semantics.

## Exact Missing Primitive

Implement an internal **path-clone verify primitive**:

1. For one logical request, create K temporary path rows that share prefix cache
   blocks but have independent suffix KV/GDN state.
2. Verify each candidate linear path as a normal sequence on FlashAttention plus
   the standard GDN path, never `TREE_ATTN`.
3. Run the flat-chain rejection sampler per path.
4. Select the winning accepted path under the lossless rule.
5. Copy/collapse the winner's committed tokens, KV blocks, and GDN/Mamba running
   state back into the original request.
6. Free the losing temporary path blocks/states.
7. Emit telemetry proving K paths ran and which path won.

Without step 5, a path can look correct for logits while leaving the live request
with stale or corrupted recurrent state.

## Completion Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Read goal and referenced files | Completed before edits; plan was stated | Done |
| Add `Fb` config | No `Fb` in launcher choices | Missing |
| K linear paths on FlashAttention | Live runtime is tree config using `TREE_ATTN` | Missing |
| GDN-safe independent histories | No path-clone/collapse primitive exists | Missing |
| Canary for K batched paths | No F_b canary exists | Missing |
| GATE1 byte-exact OFF | Not run; no implementation | Missing |
| GATE2 speed/acceptance | Not run; no implementation | Missing |
| Commit/push shipped implementation | No implementation commit | Missing |

## Recommended Next Work

Do not add another `speculative_token_tree` shape or launcher-only F block. Build
the path-clone primitive first, preferably in a proper editable vLLM fork rather
than an expanding prelaunch heredoc. Once the primitive exists, wire `Fb` through
`relaunch_qwen36_round.py` and gate it with the existing canary, OFF comparison,
and `spec_speed_probe.py`.
