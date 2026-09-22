# Bounded E2 closure decision and E1 support recommendation

2026-09-22. Read-only source/artifact review; no GPU, inference, new campaign or canonical-source changes.

**Recommendation:** once the repaired B4 identity/greedy/operand/API gates and the existing frozen numerical checks pass on its captured data, close the selected **sequential tree route’s readiness gate for scoped E1 measurement**. No separate full-model sequential-oracle campaign is necessary for that descriptive stack comparison. Native-5/native-11 still require their specified untimed per-arm/per-batch preflight inside the first planned boots. Complete the evidence manifest before retaining data from each route.

“Sequential route” names the production replay implementation. It must not be read as a newly established universal equivalence to native autoregressive decoding. My earlier B1 report’s “sequential E2 remains separate” was a scope boundary, not a request for another unbounded experiment stage.

## Finite closure of the four original surfaces

| Original surface | Existing closure evidence | Exact remaining requirement |
|---|---|---|
| Conv column-0 publication, next read, pending root once | Repaired shared-page B4 composition + independent whole-page-copy negative; selected cat10 B1/B4 fused byte A/B; loaded publisher and consumer ordering below. | No new conv model capture by default. Name this **tested helper composition plus source/engagement binding**, not a direct live conv-bank byte comparison. |
| Plain full-attention KV, including branch remap and next consumer | Selected flat-policy sync-free fixtures, independent wrong-path negative and multi-block full-cache oracle; actual B1 policy 1/0/1, source ordering and first-forward repair; B4 loaded same runner/backend. | Finish B4 request/step binding on existing bytes. Preserve actual branch paths and the remap engagement record. Do not require a numerical foreign-count log from a helper that returns the -1 sentinel. |
| B4 pack/fused-conv/recurrent isolation | Existing pinned-image B4 pack and fused fixtures; B1 state publication/read links and numerical fidelity; B4 actual 40-row captures now exist. | Repaired B4 gates must bind all four requests and complete-ledger tails, including transitions into retained support, without equating persistent row numbers with request IDs. A passing old count-only gate is insufficient. |
| Loaded greedy policy | Re-executed point-mass and duplicate-sibling gates; independent B1 full-logit accepted-plus-bonus walks; B4 drafts/logits/ledger already captured. | Complete the corrected B4 per-request greedy walk and final-only API truncation check. Native preflights remain route-specific and untimed; no new no-speculation performance arm. |

This recommendation **explicitly narrows the earlier suggested live conv/KV-byte recording requirement** to composed evidence. The existing captures do not contain a complete live conv/KV before/after/consumer byte oracle. The repaired helper tests and loaded call sites can establish the contract used by the scoped stack; they cannot be relabeled as direct runtime byte equality or full-model native equivalence. Record that evidence distinction in the qualification addendum instead of silently treating absent tensors as a pass.

## Loaded call-site check

Source references below are to B1’s `loaded_backend/`, under
`experiments/out-20260922T082815Z-e2-b1-policyB-sync-p072/e7b_001_p072_arm1_none-B_fs_ieee-all/`.

- **Conv publication is on the executed committer route.** `rejection_sampler.py:2969–3009` converts accepted draft IDs to root-excluding GDN node IDs and fills the path/lens tensors. The replay block raises on absent layers/request identity (`3085–3106`), constructs request-keyed replay rows (`3111–3132`), requires RUNROW_COMMIT and RUNROW_INIT together and true (`3138–3165`), and calls `_fr13_conv_commit_to_col0` at `3167–3173`. The helper at `1288–1353` copies the accepted leaf’s **conv view** to the request’s column 0; zero acceptance selects root column 0, missing staged banks raise, and index_select snapshots before index_copy. These are the same publisher semantics exercised by the shared-page composition and whole-page-copy negative.
- **The next fused consumer snapshots the prior before remapping.** `gdn_linear_attn.py:8524` bakes fused conv true; `11027–11028` enables committed-path reading; `11146–11172` requires the group preparation buffers. `11531–11538` uses a valid staged col0 prior or the prepared per-layer gather; `11632–11665` performs the conv-view-only remap afterward. The direct prior is handed to the fused source at `12582–12600`. In the current helper `fr13_tree_conv_fused.py:401–419`, RUNROW_INIT=1 selects column 0 and snapshots it; `182–199` concatenates prior, current tree input, and the unused appended zero row. The existing next-consumer test (`test_fr13_conv_col0_publish_read_composition.py:126–157`) proves root input is appended once after the committed path, including zero/branch/full-spine cases. This is an implementation/composition proof, not measurement of every live layer’s conv bytes.
- **KV uses fresh committed paths before the next consumer.** `gpu_model_runner.py:6670–6674` samples first; `7042–7086` selects remap=1 and reads this step’s request-keyed paths/freshness, rather than the next-forward path tensor. `7124–7140` requires positive committed path support; an all-zero-accept step requires no draft-slot copies. `7205–7214` invokes the same sync-free helper tested offline. The method then performs state bookkeeping (`7242–7244`) and later invokes the drafter (`7388` / `7459`); `7692–7730` restores slots before drafting when reorder is enabled. For policy B the mapping is already flat. Cache writes/read addresses match the independently checked TREE_ATTN contract. Sync scheduling and the ordinary same-stream branch give the required ordering; this review does not qualify the optional overlap stream.
- **Actual engagement:** B1 `docker_logs.txt:282` records fused conv; `333` records KV remap. The remap’s `foreign_first=-1` is the selected sync-free return sentinel (`gpu_model_runner.py:7215`), not a measured negative copy count or proof of zero foreign paths. B1 accepted paths include noncontiguous nodes (e.g. [1,2,5]), so branch-copy cases are present. B4 logs likewise record fused conv at line 281 and remap at 333. Its runner and TREE_ATTN hashes equal B1; GDN/sampler differences are capture setup paths and the absent one-shot E7a shim, with serving arithmetic unchanged.

B4’s raw ledger is now sealed: 19 forwards, 14 physical tree steps, eight B4 forwards (seq 6–13), then B3/B2/B1 tails; five initial nonpure forwards. **This is an inventory, not a passing verdict on B4 identity/numerical gates.** The full-B4 segment has seven complete B4→B4 intervals, not eight. Do not erase the initial mixed phase or the reduced-cohort tail when reconciling identity.

## J3 and the release manifest

The old plan literally requires J1+J3 and separately prescribes B1 boots on p017/p085. Both prefixes already exist in the B4 cohort, alongside p072/p095. Actual request records show p017=256 prompt tokens and p085=4096 (the old plan mislabeled p017’s length); p095 also has 4096. Explicitly supersede the extra J3 boot prescription by reusing these existing captures after their request/row identity is established.

Retain the named frozen numerical reading T9/T10/T6, via CPU reduction on the bound per-request records, and report any misses. Do not run a B1-only ordinal chain reducer across B4 row turnover. B1’s saved `fixture_none_fidelity.json` passes all 12 steps; SHA-256 `23d620db161af1baed4fff613659d1f89910127cec5949a951ea6f787c4b29e6`.

The final manifest should identify: exact scope/flags; fixture hashes and pinned-image results; loaded-source hashes; B1/B4 gate artifacts and their boundaries; CPU numerical results; native preflight result per route; exclusions/deferred claims. Remaining **data work is existing-artifact reduction plus the already planned native preflights**, not another tree campaign. Graphs, cache hit/eviction, stochastic policy, compact promotion, universal model equivalence and a new full-model sequential oracle are deferred scope, not hidden mandatory cells.

## E1 support-floor advice before confirmation

Replace the unobserved N1=100/N4=40 placeholders with a structural coverage floor:

- **B1:** every one of the eight frozen prompts contributes at least one valid retained interval; hence **N1 ≥ 8 per cell**.
- **B4:** each of the two frozen cohorts contributes at least one complete interval with the same four active request IDs; hence **N4 ≥ 2 per cell**.
- All requested outputs, including warmup/ramp/drain/terminal outputs, still require complete API reconciliation and a valid close seal. Missing instrumentation invalidates the cell; ordinary EOS with zero eligible support makes the cell insufficient for the specified estimand.

These are proposed minima for defining the stated workload/occupancy estimand, **not** a sample-size claim or assurance of precision. Freeze them before any retained confirmation data. The independent 0.10 paired-block half-width rule remains unchanged, with only three blocks and its explicitly coarse uncertainty. Larger accepted chunks in native-11 naturally yield fewer physical intervals; the floor should not penalize this by imposing an arbitrary per-step count derived from tree behavior.

Keep max_tokens=128, EOS, prefixes, cohort allocation and arm order unchanged. If a prompt/cohort has zero eligible support, report that fact and the full cell as insufficient support; do not replace it or silently extend output. Report, for each prompt/cohort: termination, API tokens, supported tokens, retained intervals/wall, and excluded warmup/ramp/drain/terminal counts. This prevents a positive cell total from concealing missing workload strata. No additional pilot is recommended by default.

## Exact source/artifact identities

Common remote root is `/home/mark/lumo-paper-v2-20260921/`. B4 run is `experiments/out-20260922T084319Z-e2-b4-policyB-sync-cohort4/e7b_001_p072_arm1_none-B_fs_ieee-all/`; experiments is relative to the paper v2 directory.

| Artifact | SHA-256 |
|---|---|
| B1/B4 loaded gpu_model_runner.py | b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40 |
| B1/B4 loaded tree_attn.py | a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97 |
| B1 loaded gdn_linear_attn.py | 63bc4503d560f54cf9923fc03e69267368f3951a92b1b5734a8ef871c40a0b99 |
| B4 loaded gdn_linear_attn.py | 5175f7f42a1fa2157e8be6423a69c71e800d415bbb2d45a3aeb065394c6ce9ed |
| B1 loaded rejection_sampler.py | 2680d038b9e626f659b17e6910dd89fde3ea9581eb21ea468ccd80f572ed1a56 |
| B4 loaded rejection_sampler.py | 110371ca5ca623d04d5e47ac2f3227592880daf4544e54913d3d6016c68b2bb0 |
| B4 docker_logs.txt | daa444abfe4c33a598f12f011250f9aa81c1970f96368c1a6dc2f1f4a200c4fe |
| B4 logs/e1_events.jsonl | cf410506441426904020ac756f60daa23f9d42ba5fca236b37499b16f26ae40b |
| B4 docker_inspect.json | 5b489f3b456750910ddd4af7595aa7f9450d8aa552a044d2dde2151948e2e793 |

Fixture and kernel identities/independent controls are preserved in `e2-fixture-final-recheck.md`; complete B1 raw hashes/reproducer are in `e2-policyB-b1-redteam.md`. No prior result is overwritten.

