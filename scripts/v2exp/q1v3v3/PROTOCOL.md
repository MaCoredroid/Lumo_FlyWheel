# q1v3 v3 — full-model multi-cycle continuation correctness for LumoTree (frozen protocol v3)

Status: **frozen before any v3 GPU collection**. The natural-path harvest is a separate, earlier run used only to construct cases. `FREEZE.json` binds this file, `cases.json`, the request fixtures and every executed source by sha256. `run_all.sh` refuses to start if any of them differs.

Earlier versions are kept unedited:
- **v1:** invalid. A harness forcing defect meant the candidate's tree forwards did not consume the forced tokens.
- **v2:** INCONCLUSIVE under R6. The tolerance, 4 × the worst native envelope, was too loose to detect the injected faults.
- **v2 post-hoc analysis:** disclosed separately. It motivated the per-cell rule below.

## 1. Claim under test

After a cycle, LumoTree (route `hydra27_fixed32`, deployed serve-only vehicle) does four things on the target model:
- verifies a tree;
- commits the accepted path (GDN replay, convolution-history gather, attention-KV remap);
- publishes the pending token;
- continues.

The claim is that the state it continues from, and its next-forward logits, stay within the variation between native vLLM implementations at the same point. The native implementations are:
- native A, the reference;
- B, a separate process with the same configuration;
- V, A run at the candidate's KV capacity;
- P, the stock packed GDN decode kernel.

The claim must hold over 7 consecutive cycles, on paths the deployed system actually selects, including when the prefix state comes from the prefix cache.

## 2. Cases (`cases.json`, built by `cases.py build` from the harvest trace)

**Prefixes.** Six fresh prompts, `p1`–`p6`, rendered at about 15k to 50k tokens. They are SWE-bench Verified agent requests from four tasks (astropy-14096, 14309, 14508 and 14995) and five agent conversations. None of them appears in v1, v2 or the 43-request replay tuning corpus.

**Natural cycles.** A harvest run of the deployed vehicle recorded each decode step, using record-only hooks at the deployed sampling settings. The record per step:
- the 32 tree inputs (root plus 31 natural MTP and Arctic drafts);
- the TAW outcome (accepted root-inclusive path, emitted tokens).

Cycles k = 0..K−1 (K ≤ 6, fewer only if a stop token would enter the stream) are exactly those natural steps. Cycle K is a root-only cycle on the natural step-K rows that consumes the last natural pending token. Its state (O1_K) and next-token logits therefore observe that token's consumption. A flush step then consumes z_K (logits only) and emits the terminal token.

**Three cases per prefix**, all on the identical stream:

| Case | Common O0 | Prefix cache |
|---|---|---|
| `N` | imported | cold, salted (natural digest) |
| `Nb` | imported | unsalted; primes the cache and repeats `N` |
| `Nr` | none (`o0_mode=none`) | continues from the engine's own state, restored from the prefix cache |

**Negative controls.** On the `N` case of the two shortest prefixes that have a natural cycle k ≥ 1 with an accepted path of length ≥ 2 and an active same-depth sibling. The mutation is applied at that cycle. The five mutations are as in v1/v2:

| Control | What it breaks |
|---|---|
| NC_CONV | conv gather skipped |
| NC_GDN | GDN replay skipped |
| NC_KV | KV remap off by one |
| NC_STALE | stale root input |
| NC_SIB | sibling rows committed, emitted tokens unchanged |

**Candidate repeat.** One clean repeat of the first negative-control base case.

## 3. Observations and metrics

These are unchanged from v1 §4–5:
- O0 / O1_k captures: GDN running rows (conv taps, FP32 SSM), and K/V on continuation positions;
- full-vocabulary logits for every consumed index;
- surfaces `next_kl`, `gdn_rel`, `conv_rel`, `kv_new_rel`, `kv_hist_rel` (max over layers, relative L2), plus the greedy token and the top-2 margin of A.

The v2 harness fixes stay: forced rows are indexed by tree step, and all 32 tree inputs are checked against the forced rows.

## 4. Decision rules (fixed now)

Distances are measured against A, cell by cell (case, cycle). The per-cell native reference for surface s is:

`R_s(cell) = max over {B, V, P} of d_s(arm vs A, cell), floored at F_s`

with `F = 0.02` for the four state surfaces and `F = 0.15` nats for `next_kl`.

**Candidate cell PASS** requires all of the following:
- `d_s(CAND vs A, cell) ≤ 2.0 × R_s(cell)` for every surface;
- the greedy token equals A's, **or** A's top-2 margin is within the cell band `max(0.25, 2 × max native next-token max-abs at the cell)`, **or** the cell is native-unstable (R5).

The constants were calibrated on the v2 data:

| Surface | Clean candidate at most | Injected faults at least |
|---|---|---|
| State surfaces | 1.29 | 2.83 |
| KL, with the 0.15 floor | 1.41 | 5.18 |

The v3 prefixes are fresh, so v3 is an out-of-sample test of the rule.

**Gates:**

| Gate | Requirement |
|---|---|
| R1 integrity | A, B, V, **P** and CAND produce every planned observation, sealed and valid; the natives consumed exactly the forced streams; every candidate tree forward consumed exactly the forced rows. |
| R2 prompt parity | Each prefix renders to the same (length, sha256) in every arm. |
| R3 O0 import | Every import read-back equals A's source (cache-reuse cases import nothing). |
| R5 native categorical stability | ≤ 10% of candidate cells are native-unstable. |
| R6 discrimination | Every negative control is executed and detected at its mutation cell, on the base case's per-cell reference with the same multiplier. NC_CONV, NC_GDN, NC_KV and NC_STALE must exceed the bound on their target surface (conv, GDN, KV-new, next-KL). NC_SIB must exceed it on at least one of KV-new, conv and GDN. |

**Cache-reuse coverage** is reported, not gated. Every `Nr` observation must start with ≥ 90% of the prefix already cached in A, B, V, P and CAND. If this fails, the verdict makes no statement about cache reuse.

**Verdict (`VERDICT.json`):**

| Verdict | Condition |
|---|---|
| **EQUIVALENT** | R1, R2, R3, R5, R6 hold; every candidate cell passes; no candidate structural failure. |
| **NOT_EQUIVALENT** | R1, R2, R3, R6 hold, and some candidate cell fails or a candidate structural check fails. |
| **INCONCLUSIVE** | Otherwise. |

## 5. Not covered (stated up front)

- Sampling and distribution preservation. It is tested separately, by a sampler-level test whose sensitivity to faults is shown first.
- MTP/draft-model state against a native MTP reference.
- Batch > 1.
- Speed.

Logit comparisons concern the next-forward distribution after each commit, not sampled outputs.
