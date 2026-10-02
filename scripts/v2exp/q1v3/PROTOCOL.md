# q1v3 — full-model multi-cycle state-equivalence check for LumoTree (protocol v2; v1 text below unchanged)

## v2 amendment (2026-10-02, frozen before any v2 GPU run)

v1 run q1v3-20261001T223300Z (verdict INCONCLUSIVE, R6) is retained unedited in
`scripts/v2exp/results/q1v3_v1_20261002/`. Its candidate observations are **invalid**: a harness defect, not a
candidate property. The engine's drafter also runs after every prefill chunk, and v1 indexed the forced tree rows by
the raw drafter-call count. Forced rows therefore went into discarded prefill-chunk drafts (cold prefixes, 14-29
chunks) or arrived one cycle early (prefix-cached cases). The tree forwards verified other tokens while the harness
forced the commit products. A GPU diagnostic captured the 32 inputs of every tree forward: 0 of 25 steps carried the
forced rows, and every step equalled the drafter output of the last prefill chunk or the step before.
v1 checked only the root input, so the defect stayed invisible. The v1 negative controls ran on the same broken
forcing and are also invalid.

v2 changes the harness only:
1. `on_drafts` indexes forced rows by the tree step the drafts feed (number of tree steps already consumed).
2. New integrity check: at every candidate tree forward all 32 input tokens must equal the forced rows (flush rows
   for the flush step). A mismatch invalidates the observation (R1), since it is a harness failure, not candidate
   behaviour.
3. The simulator runs chunked prefill with a drafter call per chunk. v1 hooks fail the end-to-end tests under it;
   v2 hooks pass all 15.

Decision rules, metrics, tolerances, cases, arms and negative controls are **unchanged** (sections 2-6). Note
recorded before the v2 run: v1's native envelope (calibration cells of B, V, P vs A) gives state tolerances tau of
1.0-1.6 (relative L2). Two unrelated vectors of equal norm give about sqrt(2) = 1.41, so R6 may fail again on
state-surface negative controls. Any tolerance redesign would be a separate, disclosed protocol version.

# v1 protocol text


Status: **frozen before any GPU run**. `FREEZE.json` binds this file, `cases.v1.json`, the request fixtures and every
executed source by sha256; `run_all.sh` refuses to start if any of them differ. Any change after the first GPU
launch is a new protocol version (v2) with its own freeze; results of v1 attempts are kept, never edited.

## 1. Claim under test

After LumoTree (route `hydra27_fixed32`, deployed Cqc10 serve-only vehicle, Qwen3.8-27B NVFP4) verifies a forced
candidate tree and commits a forced accepted path — recurrent GDN replay, convolution-history gather, attention-KV
remap, pending token — the target-model state it continues from is equivalent, within a tolerance calibrated on
native-vs-native variation, to the state native sequential (spec-off, one token per step) decoding reaches after
consuming the same tokens; and this holds over **4–7 consecutive cycles** carrying the actually published state
(one common-O0 import at cycle 0, no re-hydration between cycles).

Not claimed (see §8): MTP/draft-model equivalence, APC/lifecycle correctness, batch > 1, speed, sampling
distribution, natural (unforced) acceptance.

## 2. Arms (one engine at a time, fresh process each)

| arm | engine | KV capacity | role |
|---|---|---|---|
| **A** | native vLLM 0.19.2rc1.dev134, same image / lm_head patch / forked FA2 .so / flags as `serve_native.sh` | `--kv-cache-memory-bytes 42949672960` (40 GiB) | reference; its cold prefill of each prefix defines the **common O0** |
| **CAND** | deployed LumoTree vehicle (`promoab_tail10_serve_only.sh` → `v2exp_serve_only_variant.sh` → `fr13_launch_forked_fa2_tree_server.sh`), generated q1v3 copy | **not pinnable**: the launcher's exact-pair table requires `KV_CACHE_MEMORY_BYTES=""` at `MAX_NUM_SEQS=1`; capacity follows `GPU_UTIL=0.70` and is recorded | candidate |
| **B** | as A | as A | process repeatability under KV pinning |
| **V** | as A | `--kv-cache-memory-bytes` = CAND's observed KV tensor bytes (boot record) | KV-geometry sensitivity at the candidate's capacity |
| **P** | as A + `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1` (stock packed GDN decode) | as A | equivalent-implementation variant (optional) |

**KV pinning statement.** The candidate cannot be KV-pinned without editing the deployed launcher, so it is not.
The comparison does not depend on candidate pinning: (i) every arm starts each case from the *same imported*
O0 bytes, so prefill-capacity effects are removed; (ii) arm V runs the native reference at the candidate's KV
capacity and arm B at the pinned capacity, and both enter the calibrated envelope (§5), so any numerical effect
of KV capacity is part of the tolerance instead of being attributed to LumoTree; (iii) all block counts are
recorded in `boot.*.json`. Exact bank-shape equality between native (64-token kernel pages) and candidate
(1024-token pages) is impossible by construction and is not attempted.

The hooks only (a) overwrite forced tokens/products in place, (b) read state, (c) import the common O0 once per
case before the first continuation forward, and (d) in negative-control observations only, apply the declared
mutation. No serving flag, kernel, env pin or cache size of the deployed route is changed.

## 3. Case set (`cases.v1.json`, built by `cases.py build`)

* Prefixes: the six Codex Q1 prefixes (calibration short/medium/long 13 487 / 28 941 / 60 083 tokens; held-out
  evaluation short/medium/long 13 373 / 28 930 / 48 688) sent as their frozen chat bodies, plus one held-out
  **block-boundary** prefix from the v2exp corpus (25 535 tokens; 1024-boundary at 25 600).
* 33 positive cases, 147 cycles: per Codex prefix five schedules (C1 spine-bonus, C2 nc-base, C3 branch-leaf,
  C4 root-only, C5 deep-branch), each 4–5 cycles; boundary prefix three 7-cycle schedules where the boundary falls
  inside an accepted path (K1), on the last accepted draft (K2), or on the pending token (K3).
* Coverage: root-only (L=0), spine (L=1,3,5), max depth 11 (full spine, bonus from leaf), off-spine branches
  (rows 2-7-12-17-22; 3-8-13; 4-11; 9-16; 14-20; 14-21; 1-6), padding-adjacent node 13 (its only child, row 18,
  is an inactive padding row), pending kinds bonus / correction / sibling-token (z equals the token of an
  unaccepted child of the last accepted node), page/mamba-block boundary crossings.
* Forced tokens: cycle-0 tree rows of Codex prefixes reuse `token-fixtures.v1.json`; all other forced tokens are
  sha256-seeded draws from the prefix's distinct non-special tokens. After the last cycle a flush step consumes
  the final pending token and emits the terminal token.
* Negative controls (candidate only; stream of `C2-nc-base` on cal-short and cal-medium; mutation at cycle 1,
  path S3 = rows 0-1-4-9; run after all positive observations, least invasive first):

| id | mutation | must fail at cycle 1 |
|---|---|---|
| NC_CONV | running-row conv taps restored to their pre-step value (conv gather skipped) | conv |
| NC_GDN | running-row SSM restored to its pre-step value (replay skipped) | GDN |
| NC_KV | `accepted_paths+1` passed to the KV16 remap only (off-by-one remap) | KV (new positions) |
| NC_STALE | next root *input* overwritten with the previous root; emitted token unchanged | next-token logits |
| NC_SIB | committed rows = sibling path 0-1-4-10; emitted tokens unchanged | KV, conv, GDN |

* One clean candidate repeat of `cal-short__C2-nc-base` (within-process determinism; diagnostic).

## 4. Observations (identical stream indices in every arm)

Stream = consumed tokens r₀, accepted drafts, z₀, …, z_{K−1}; index i ↔ position P+i.

* **O0** (pre-forward of the step consuming r₀): natural digest (first, cold-prefill case of each prefix), then
  import of A's O0 for that prefix (48 GDN running rows: 3 live conv taps + FP32 SSM; 16 attention layers: K/V
  for [0,P)); read-back must equal the source bytes.
* **O1_k** (pre-forward of the step consuming z_k): GDN running rows (conv live taps, SSM) and target K/V for all
  continuation positions [P, P+b_k).
* **Logits**: full-vocabulary BF16 row for every consumed stream index (native: its decode row; candidate: the
  tree row that consumed it). `next_k` = logits at index b_k (after consuming z_k), i.e. the first forward that
  runs on the state published by commit k.
* Structural (exact): candidate root input at every tree step = expected pending token; published
  `_LUMO_FA_ACCEPTED_TREE_PATHS/LENS` = forced rows; natives consumed exactly the forced stream.

## 5. Metrics, tolerances (frozen constants in `reduce.py`)

Per (case, cycle k), candidate vs A and every native arm vs A:
`next_kl` = KL(softmax A ‖ softmax X) of next_k, fp64, full vocabulary; `gdn_rel`, `conv_rel` = max over the 48
layers of ‖X−A‖₂/‖A‖₂; `kv_new_rel` = max over 16 layers × {K,V} of the relative L2 on cycle k's newly committed
positions; `kv_hist_rel` = same on all earlier continuation positions. Reported only: max-abs logit difference,
top-5 overlap, verify-row KL along the accepted path, conv with reversed tap order (layout diagnostic).

* Envelope E_s = max of surface s over all **calibration-prefix** cells of native pairs (A,B), (A,V), (A,P).
* Native tolerance τ⁰_s = max(F_s, 4·E_s), floors F = 0.02 for every surface (used for R4).
* Candidate tolerance τ_s = τ⁰_s, except next-token KL: τ_kl = max(τ⁰_kl, 4·D_kl), where D_kl is the largest
  candidate-vs-A KL over the **cycle-0 verify rows** of calibration cases (rows that consume r₀ and the cycle-0
  accepted drafts: they run on the bitwise-identical imported O0 before any commit, so D measures LumoTree's
  verification-kernel numerics, not commit state). Reported separately: whether every cycle-0 verify row is
  within τ⁰_kl.
* Greedy band = max(0.25, 2·E_maxabs, 2·D_maxabs) (E_maxabs: calibration native envelope of next-token max-abs
  logit difference; D_maxabs: same quantity on the candidate's cycle-0 verify rows). Smallest-ID tie-break.

## 6. Decision rules

* **R1 integrity**: arms A, B, V, CAND produce every planned observation, sealed, valid (complete coverage,
  record digests and object sha256s verify); natives consumed exactly the forced streams.
* **R2 prompt parity**: each prefix renders to the same (length, sha256) in every arm.
* **R3 O0**: every import read-back equals the A source.
* **R4 held-out reference**: every native-pair cell on held-out prefixes (evaluation + boundary) is within τ.
* **R5 native categorical stability**: ≤ 10 % of candidate cells have a native arm whose next-token greedy differs
  from A's while A's margin exceeds the band (such cells are exempt from the greedy criterion).
* **R6 discrimination**: NC_CONV, NC_GDN, NC_KV and NC_STALE are executed and each exceeds the candidate τ on its target surface
  at its mutation cycle; NC_SIB, if executed, must also be detected.
* **Candidate cell PASS**: all five surfaces ≤ τ and (greedy equal, or A's margin ≤ band, or the cell is R5-exempt).

Verdict (`VERDICT.json`):
* **EQUIVALENT** — R1–R6 hold, every candidate cell passes, no candidate structural failure.
* **NOT_EQUIVALENT** — R1, R2, R3, R6 hold and at least one candidate cell fails on a calibration prefix (or on a
  held-out prefix while R4 holds), or a candidate structural check fails.
* **INCONCLUSIVE** — otherwise (missing/invalid data, reference not generalising, comparator not discriminating).

Reported, not gating: A-vs-B repeatability class (EXACT if every A/B cell is bitwise equal, else SPREAD); natural
(prefill) O0 equality per arm; per-cycle growth of candidate distances; verify-only cycle-0 KL (candidate kernel
noise on identical state); candidate repeat; MTP top-1 vs native greedy agreement.

## 7. Integrity and procedure

* One driver (`run_all.sh`): A → CAND → B → V → P → `reduce.py`. A failed A stops the run; other failed stages are
  reported and make the verdict INCONCLUSIVE through R1. A stage may be re-run (same `--run-id`, its partial output
  moved aside); every attempt's logs are kept.
* Raw evidence: per observation `cases/<obs_id>.json` (record digest) + raw tensor objects (sha256 in the record);
  engine logs, patch receipts, boot records, launch-chain generation receipt, client responses.
* No threshold, case, arm or rule may change after seeing data under this version.

## 8. Known limitations (stated up front)

* MTP/draft state is not compared to a native MTP reference (no native MTP head in the spec-off arms; the MTP KV of
  the prefix is the candidate's own prefill). Natural draft trees are recorded; the MTP-top-1 agreement is a
  diagnostic only. The draft path cannot affect the forced commits examined here.
* Attention KV is compared on continuation positions only; prefix positions are the imported O0 (verified).
* Convolution comparison assumes the candidate's live taps are rows 0–2 of its 34-row history in native order (the
  layout Codex's accepted stage-1 hydration used); a reversed-order diagnostic is reported.
* B1, one request at a time, temperature 1.0 forced streams (deployment refuses temperature 0); APC on as deployed,
  but prefix-cache lifecycle correctness is out of scope; boundary cases check the running state only.
* The calibrated tolerance is only as tight as native variation plus floors; R6 guards against a tolerance too
  loose to see real state bugs, but a bug smaller than τ would pass.
