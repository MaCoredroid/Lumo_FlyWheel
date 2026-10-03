# M1-Q Lumo lifetime helper: bounded source closure

2026-09-28. Bounded source/CPU review accepts the repaired helper and its collector/launcher integration at these SHA-256 identities:

| File under campaign `tools/m1/` | SHA-256 |
| --- | --- |
| `m1_q_executor_lifetime_v1.py` | `c2a40888373a9de0a002dd49a8c86dc819c514f6948a6d50f362021f5ebe5d59` |
| `m1_q_c0_collect_v1_1.py` | `736ee7b49be82ebed4cbb37e10002a766461ff73cfda4b183446059bb50b8a08` |
| `run_m1_q_c0_owned_v1_1.py` | `7c226d00905855c57aac633b36dc3a1022f63ec98d6d707decb962e40f5b5087` |

The helper reuses only a fully constructed Lumo Backend and runner module. Each cycle constructs a fresh facade; the base constructor supplies fresh logs/results/lifecycle/counter bookkeeping, while the existing collector supplies a fresh capture wrapper. The unchanged base reset and scientific methods still execute. Pending metadata is zeroed in existing buffers; static SSI and boot-fixed pointer identities are checked. Author methods instantiate the unchanged base type and receive no Lumo Backend. Boot provenance is explicitly once per process; counters remain cumulative and flat integer counter deltas are observed from before/after snapshots.

Three concrete draft defects were reproduced and repaired before this closure:

1. The new imported helper was omitted from mandatory source bindings. Both collector and launcher now require its hash in the gate's source map.
2. After an unsuccessful second reset, `complete_lumo_cycle()` returned the first reset's record. Each factory dispatch now registers a new token and `reset_not_completed` status before construction, and an incomplete reset returns that record without a prior counter baseline.
3. `prev_lens` pointer identity was compared only within each reset. It is now frozen at initial Backend adoption and checked across reuse as well as after the base reset.

The initial three reproductions and positive controls remain in `p0/monitor/review-response-20260927/m1q-lifetime-helper-reviewed-20260928T222450Z/`. That draft helper was `3b97e85ba657f1b50de79e7d5ad76d6e9b67024facdbb22e39816863f1dfb3ae`; its six controls include three deliberately reproduced defects, not a clean verdict.

The repaired snapshot is `p0/monitor/review-response-20260927/m1q-lifetime-helper-repair-reviewed-20260928T222630Z/`. Nineteen independent controls pass: nine helper controls plus ten carried-forward collector failure/domain controls. The helper controls use only fake scalar/tensor/backend objects with deterministic opaque contents. They establish one Backend for two distinct facades, isolation of bookkeeping, no old facade retained by the holder, in-place pending reset, untouched author base type, refusal of inter-cycle and during-reset pointer replacement, constructor failure without an adopted Backend, and base-reset failure retaining the Backend but marking the current reset incomplete. Collector controls retain the earlier protection for escaped cycles, index/final-receipt failures, nonfinite stopping, input mutation, and exact four-method/two-repeat paths.

Reproduce from the repaired snapshot:

```sh
/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -m unittest -v reviewer_controls reviewer_collector_controls
```

No Torch, CUDA, candidate, actual executor, launcher, numerical fixture, or model code was invoked by these controls. The helper module itself is stdlib-only; base objects are reviewer fixtures. Original executed v1 collector and launcher were verified unchanged at `2bb6103e88d84b51c6e022c5c9ad424e151c94c85f8d3e34a70bd94028a37b92` and `34de4761a587ac7bd1cd3ca0e58a26458e5267269c5239d28cb0bdf3059e68b7`. The accepted executor, runner, adapters and kernel were not edited by this review.

No further blocker was found within the requested source/lifetime scope. The prospective repair amendment, final gate/source binding, resource admission and repaired execution remain parent-owned. Actual pointer/reset/counter evidence must still be retained when run. This source closure does not authorize a launch, clear the preserved failed run, or establish numerical qualification.
