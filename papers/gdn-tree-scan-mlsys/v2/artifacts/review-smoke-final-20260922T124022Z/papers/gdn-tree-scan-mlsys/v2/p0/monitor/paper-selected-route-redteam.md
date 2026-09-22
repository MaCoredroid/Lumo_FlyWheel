# Selected-route manuscript delta review — 2026-09-22

Read-only review of local and remote `main.tex` **c23eeeed858d1975921b2a3035a72e1538174f71bb152ba6b705919d686c0bf0**, and `abstract.tex` **2952d72ce3c00fa1f425b4ef63abbb8324dd2f5180261402c24576ace10f7e19**. Scope: new selected-route claims, changed evidence status, limits, and E1 plan. No source edits, GPU work, new campaign, or broader literature review.

**Disposition: the selected-route numerical and causal claims are supported. No additional tree experiment is required under the agreed finite E1 scope.** Before final integration, clarify the three manuscript items below and update the companion's stale route status. The planned native-arm preflights and E1 timing remain pending.

## Required bounded corrections

1. **Abstract status is stale/ambiguous (abstract.tex:2).** “Current model-level continuation ... remain[s] unqualified” now obscures the bounded B1/B4 qualification reported at main.tex:378 and the executed status at 181/196/386/389. The full-model sequential claim remains unqualified, but the observed selected route is qualified for scoped measurement. Suggested replacement: “Stage-isolated diagnostics and source-bound B1/B4 checks provide bounded continuation evidence; full-model sequential equivalence remains unestablished and matched current-stack timing is pending.” This does not promote the compact candidates or erase failed E7a criteria.

2. **Name the planned native baseline implementation (main.tex:411).** The paper currently says three arms and strong native chains without explaining that the forthcoming native arms use the patched runner. Current E1_FREEZE.md:44–48 explicitly specifies stock MTP behavior in the patched runner, `naive_mtp`, tree flag off/no tree descriptor, with native per-arm/per-batch preflights not yet executed. Add one sentence: “The planned chain baselines use stock MTP behavior within the instrumented patched runner; each native arm and batch condition must pass its untimed source/configuration, API/ledger, occupancy, termination, and prescribed greedy checks before its timing window is retained.” Do not describe the baseline as an unmodified runner or claim all FR13 features are globally disabled.

3. **Match the repaired stopping guard precisely (main.tex:380).** “Absence of premature stopping at a node with a matching child” is accurate for all 32 observed blocks, but the cited repaired gate specifically rejects a **nonterminal** stop and allows terminal API clipping. State the boundary explicitly, e.g. “...the terminal bonus, and rejection of nonterminal stopping while a matching child remains, with terminal API clipping handled separately.” The final gate (`e7b_b4_capture_gate.py` 0f0aef4f…, lines 152–160) has that exception. No first/last/longest sibling selection promise should be introduced.

**Companion release status:** E1_FREEZE.md **0fbc25b72b6f48db924cb557acaa1f1eaeadc47023e303d876328951537d239b**, line 38 still says B1 remap is under investigation and the B4 boot is “NOT yet done.” Add a dated superseding closure disposition with the actual final reports/source hashes and explicit native-preflight pending status. E2_QUALIFICATION_PLAN.md **93351d9ccaad82383abd2d048f0b0876576a3554010c09e82d28fe36cda4004d** likewise retains historical pending sections; its existing J3 supersession addendum correctly reuses B4 p017/p085/p095 captures and disclaims B1 long-context qualification. Preserve chronology and identify the current release decision. This is a manifest/status fix, not a request for another model boot.

## Claim checks that pass

| Manuscript lines | Checked claim | Disposition and scope |
|---|---|---|
| 372 | TREE_ATTN write/read permutation mismatch; node 3 address witness; first difference at layer 3; max logits delta 6.125; one argmax row | Matches loaded-source/address review and `results/e2-policyA-first-forward-audit.json` (changed row 5). Same input hidden/positions is stated correctly; direct input-ID tensor equality is not asserted. |
| 374 | B1 first-forward equality, 12 publications/11 bytewise links, 31 post-prefill API tokens reconstructed, 34 structural/32 API, 11 intervals | Matches independent B1 report. The initial prefill token's weaker evidence is disclosed; later KV-off continuation is correctly denied oracle status. |
| 376; 196; 413 | Helper composition, source/guard binding, multi-block controls; no live conv/KV byte oracle or full-model proof | Matches finite closure decision and independent helper recheck. These limits prevent helper success from being relabeled as complete runtime cache equivalence. |
| 378 | 53 layer computations; 49 bytewise handoffs; 32 blocks in eight full B4 forwards; 85 structural/83 API-visible captured tokens; 128 total API tokens; seven B4 intervals | All exact. “Full B4 forward” means a complete 40-row capture; p072's last such forward still extends two structural tokens beyond its API cutoff. No eight-interval or 85-API-token claim is made. |
| 378 | Exhaustive exact formatter inverse over 248,320 model IDs; no ledger/draft IDs used; all 128 unique | Matches agent2's independent proof. Original strings and derived IDs are distinguished. Do not later shorten this to directly captured API IDs. |
| 380 | Equal-overlap duplicate-token source selection remains random at temperature 0; [1,3] is legal even though first-child walk gives [1,2,4] | Matches sampler source and p085 raw witness. No distribution-preservation proof is implied. Add the stop-boundary qualification above. |
| 181/196/386/389/420 | Bounded E2 executed, E1 timing pending, no full-model sequential or general sampler-law guarantee | Supported. The contradictory broad abstract wording is the required reconciliation. |
| 411 | 18 initial cells; per-prompt/cohort structural coverage; separate 10% paired-block precision target; three-block uncertainty coarse | Matches frozen scope. Coverage floors are not power guarantees; physical steps are not independent boot replicates. No extra pilot or arbitrary larger per-step floor is needed. |

The B4 numerical statement is **not bitwise output agreement**: the maximum output absolute error is 4.789560742476662e-4, significant bf16 ULP maximum 1, maximum replay-state error 9.5367431640625e-7, and the minimum rounded-oracle/served-output bitwise fraction is 0.9998534917831421. All 53 steps meet the declared numerical classes; the 49 state-copy links are separately bytewise. The current main text preserves this distinction.

## Finite remaining work

For the claims reviewed here: finish the explicit text/status/manifest corrections and preserve the exact analysis companion. The existing tree evidence closes the agreed bounded readiness gate. The already-planned untimed native preflight for each native arm × batch condition remains required before retaining that route's E1 timing, including the reset warmup/timed boundary and final ledger re-audit. E1 confirmation measurements themselves remain outstanding. Broader stochastic deployment, graph/cache-hit behavior, compact-candidate promotion, and a full-model sequential oracle are deferred claims, not silently required extra cells.

## Evidence identities

Paths are relative to `papers/gdn-tree-scan-mlsys/v2/`. These reports retain exact raw artifact/source hashes and reproduction instructions.

| Evidence | SHA-256 |
|---|---|
| `p0/monitor/e2-policyB-b1-redteam.md` | `dd7de6d504fdd4daf8d091b36cc8e02b9ebe7c89aee7e61e81fa518b51e1be8f` |
| `p0/monitor/e2-b4-final-redteam.md` | `105d5eb78029e70a864cbefa3cb71a2886f94733d9a107055be59ef77d654788` |
| `p0/monitor/e2-b4-api-inversion-redteam.md` | `f3b37263801f0faeaf4ecd4faeaeb6382b0e014fac54501109e4367e1e0a388a` |
| `p0/monitor/e2-fixture-final-recheck.md` | `3d3c7343664a777cbc5e9df886b4026ea604751e4264cdf1a4fd4bfc28397f8f` |
| `p0/monitor/e2-closure-decision.md` | `422b53087cc46fce0d131479a8839444b2bb473b7db6fa905c334e975c70c2af` |
| `results/e2-policyA-first-forward-audit.json` | `ec12ccf0ca25d0c7315dd8509a3290dfc1762f8258c65b46202ab6f4c870db1e` |

## Closure recheck — 2026-09-22, after the 09:30 UTC manuscript update

**All three manuscript clarifications are closed.** Read-only comparison confirms local and remote files have identical SHA-256 values:

- `main.tex`: `1857d76012642eb284ac8e1c4361a7af6bf4fa5d21697751d79d19edaabfe606`.
- `abstract.tex`: `9a29ea33ffb9a3fdb84ee3daf980861535f0ab9414165e0516c6b67170d7b0b0`.

The abstract now names the bounded selected-route evidence and keeps full-model sequential equivalence unestablished and timing pending. Main line 411 explicitly labels stock MTP paths in the same patched runner/recorder, native mode without a tree descriptor, and pending B1/B4 live preflights. Main line 380 now rejects nonterminal premature stops while allowing API-budget clipping only at the request's final output boundary. These statements match the reviewed evidence and intended protocol; no additional experiment is required to close these wording findings.

The separate companion E1_FREEZE/E2 status item remains **pending under the worker's ownership**, without rechecking moving companion files in this pass. Native preflight and E1 numerical integration remain future work. This recheck covers only the three requested main/abstract changes.
