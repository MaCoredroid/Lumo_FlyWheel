# FR13 — Per-Event Superset Gate (design + ready-to-run spec) — 2026-06-15

**Scope:** DESIGN ONLY (CPU, read-only; a big-denom GPU serve + the directional-asymmetry CPU run are
concurrent — nothing here edits code or boots). Pins: repo HEAD `75251bd8`, vLLM source via
`scripts/vllm_src.sh` (`3dbe092e`). Grounding rule: int-view equality NEVER atol; the lossless/greedy
reference is the **deployment-correct recurrent decode oracle** (`scripts/fr13_recurrent_decode_oracle.py`),
NOT a serial-torch ref / chunked-prefill / backend name (bug-class #10/#11).

---

## 0. The problem this gate fixes (the user's framing)

The deliverable speed claim is the **depth-matched SUPERSET**: the cat9 9-node caterpillar tree beats the
E5 5-spine native MTP-5 (`FR13_DIRECTION_AND_NUMBERS.md`). We have been quoting the **aggregate** accept/event
(cat9 ≈3.198 vs native 3.076, "+0.12 edge"). That number is **cross-trajectory** — cat9 and native each free-run
their own stream, so the events are not the same events. Per **bug-class #12 (measurement traps)**: *"non-like-
for-like trajectories after fixes"* + *"raw counters only … source-index traces"*. A positive aggregate edge does
NOT prove a per-event superset, and it cannot tell a **lossless** gain (leaf == greedy) from a **lossy flip**
(leaf != greedy) — exactly the channel the flip analysis tracks. This doc specifies the **per-event** gate that
unifies speed (the saves) and lossless (the flips) on the SAME leaves = the banked **greedy branch rescue**
(+35 genuine branch accepts at greedy, `8add39e6`; `FR13_ACCEPTANCE_LADDER_BIND.md` R3).

---

## 1. The accept trace — what we capture, per event, for BOTH arms on the SAME reference

### 1.1 The apple-to-apple principle (avoid bug-class #12 cross-trajectory)

Free-running both arms diverges after the first fork (one different served token reseeds the drafter and the
trajectories never re-align). The clean count requires **the SAME reference stream teacher-forced through both
verifiers, compared PER POSITION**. The reference is one stream of served positions `s[0..L-1]` (the cat9 served
stream, or — equivalently for the gate — the recurrent-oracle greedy stream). For each reference position we ask:

- **(E5 arm)** does the MTP-5 **spine** draft at depth d match the target greedy (accept) or miss (cut)?
- **(cat9 arm)** does the spine match (accept); OR where the spine is cut, does a **leaf** draft match the
  target greedy (lossless save) / a non-greedy token (lossy flip-save); or do both miss?

Both arms scored on the **same reference positions** ⇒ a clean per-event domination count, no trajectory drift.

### 1.2 cat9 per-event trace — ALREADY EMITTED (committer dump), CODE-READ

`scripts/fr10_phase4_patch_vllm_tree_gdn.py`, flag `FR13_FORK_MARGIN_DUMP=1` (default OFF, READ-ONLY,
eager-only; OFF boot is byte-identical to the locked path — bug-class #10 needle at L6122). Per spec-step per
request it writes (jsonl, L7314-7356):

| field | meaning (the per-event accept trace) |
|---|---|
| `spine_path`, `spine_lcp` (== `path0_lcp`) | the MTP-5 spine path `[0,1,3,5,7]` and **how many spine tokens it accepts** = the E5-equivalent accept on cat9's OWN drafts/targets this event |
| `best_path`, `winner_lcp` (== `best_lcp`) | the cat9 committed path and **how many cat9 accepts** (max over all 5 root-to-leaf paths) |
| `best_leaf` vs `spine_leaf` | which leaf won; `best_leaf != spine_leaf` = a FORK (a leaf out-matched the spine) |
| `lcp_delta_winner_minus_spine` | `winner_lcp - spine_lcp` = **the leaf save count** this event (≥1 on a fork) |
| `is_fork`, `split_pos`, `split_node` | topology branch point (where leaf vs spine diverge) |
| `spine_token_at_split`, `winner_token_at_split` | the spine token the spine WOULD serve vs the leaf token cat9 served at the split |
| `committed_row` | the served tokens (accepted prefix + bonus) |
| `winner_div_margin`, `spine_div_margin`, `split_node_margin` | VERIFY top-2 logprob margins at the deciding nodes (for the fixable-vs-fundamental near-tie split, `fr13_fork_margin_classify.py`) |

Mechanism CODE-READ (committer `_lumo_tree_path_lcp_max_greedy_sample`, L6894-6925): `best_lcp = max over leaves
of lcp`, tie-break = earliest-leaf on EQUAL lcp (L6917-6919, `if lcp > best_lcp` strictly keeps the first) and
the spine is the first-child chain. **Consequence (CODE-READ): a leaf can only win the commit when its
`lcp STRICTLY > spine_lcp`.** So `winner_lcp >= spine_lcp` ALWAYS — i.e. **cat9 can never accept fewer spine
tokens than the spine alone, by construction** (`superset_violation = best_lcp < path0_lcp` is structurally
False at greedy; the only historical violations were the S1 bonus-ROW bug, `4d45be27`, a token-value defect not
an lcp regression — bug-class #5). This is the spine-non-regression guarantee, but it must still be **measured**
live because S2 verify corruption / S3 drafter non-byte-identity can move which tokens enter the spine lcp.

### 1.3 E5 (native MTP-5) accept trace — two routes, code-read

- **(EXISTS) free-run native spec trace:** `output/fr10_native_mtp5_same8_*/logs/per_req_spec_trace.jsonl`,
  rows `{"acc": k, "draft": 5, ...}` = native accept length per event. Loaded by
  `src/lumo_flywheel_serving/fr10_superset_gate.py::load_spec_trace` → `AcceptanceEvent(accepted_len=acc,
  path0_len=acc)`. **This is cross-trajectory** (native's own stream, native's own drafts) — usable only for the
  AGGREGATE reconciliation (§3), NOT the per-event gate. (native draft tokens per depth are also banked:
  `logs/fr10_mtp_draft_trace.jsonl`, `{"event":"mtp_draft","draft":[[d0..d4],...]}`.)
- **(APPLE-TO-APPLE) E5-on-cat9's-event = `spine_lcp` from the cat9 dump.** The cat9 spine path `[0,1,3,5,7]`
  IS the MTP-5 spine fed by the same drafter at the same committed prefix; `spine_lcp` is precisely "how many
  spine tokens the MTP-5 verify accepts THIS event". So **the cat9 `FR13_FORK_MARGIN_DUMP` already carries the
  E5 arm**, per event, on the identical reference — no separate E5 boot is needed for the spine-vs-cat9
  comparison. Caveat (bug-class #11 + S3, `FR13_ACCEPTANCE_LADDER_BIND.md`): cat9's spine drafts are NOT
  byte-identical to a standalone native MTP-5 run (alt co-residency / BI-asymmetry can flip a spine draft), so
  `spine_lcp` is the *within-cat9 spine*, not a third-party native re-run. For the **superset structural gate**
  this is the correct apple (same event, same drafts, spine-vs-tree); for an *absolute* native-quality anchor,
  pair it with the §1.4 recurrent oracle (the deployment ground truth).

### 1.4 The greedy reference / lossless judge — the recurrent decode oracle (CODE-READ)

`scripts/fr13_recurrent_decode_oracle.py rescore`: loads a served stream `--src {prompts, records[].
served_token_ids}`, re-prefills the prompt once (chunked, as in deploy) then **teacher-forces every served
position through the RECURRENT single-token decode path** (`_forward_core_decode_non_spec`, the deployment path;
NOT chunked-prefill — that is the whole reason this oracle exists, docstring L16-27). Per position it records
`oracle_argmax_id` (the **greedy decode token** = the lossless reference), `flip = served != argmax`,
`clear_margin` (flip beyond 1.0 nat). Engagement asserts GDN present + recurrent path fired (class #9), within-
proc determinism rep1==rep2 (class #8). Output schema `fr13.recurrent_decode_oracle.rescore.v1`, per_prompt[].
positions[] with `oracle_argmax_id` and `clear_margin`. **This argmax is the per-position greedy that decides
whether a leaf save is lossless (leaf == oracle argmax) or a lossy flip (leaf != oracle argmax).**

### 1.5 The JOIN (non-vacuous, bug-class #12) — ALREADY BUILT

`scripts/fr13_fork_margin_classify.py` already joins the global monotonic dump `step` counter to per-prompt
served positions: walk dump records in step order, consume each `committed_row`, match the running concat
against `capture.records[pid].served_token_ids`; the first contiguous run that equals the served stream defines
the (position → dump-step) map; **ASSERT every scored position lands on a real dump step or FAIL LOUD** (class
#9). This is the exact join the per-event gate needs — extend its reduce, do not re-invent it.

---

## 2. The gate computation — per-position classification + metric

For each reference position i (joined to its cat9 dump event via §1.5; the greedy token = the recurrent-oracle
`oracle_argmax_id[i]` from §1.4), classify the **{E5 spine} × {cat9}** cell:

**E5 axis (from `spine_lcp` / depth d of position i within its event):**
- `E5 spine-ACCEPT` at depth d ⟺ the spine draft at depth d == target greedy ⟺ d < `spine_lcp`.
- `E5 spine-CUT` at depth d ⟺ d == `spine_lcp` (the spine first misses here).

**cat9 axis at the same depth d:**
- `cat9 spine-ACCEPT` ⟺ d < `winner_lcp` AND the accepted node at d is on the spine path.
- `cat9 leaf-SAVE` ⟺ d is in `[spine_lcp, winner_lcp)` on a fork (`best_leaf != spine_leaf`): cat9 accepted a
  token at depth d that the spine did NOT (the leaf out-matched). Sub-classify by the §1.4 greedy:
  - `leaf-save LOSSLESS` ⟺ the served leaf token == `oracle_argmax_id[i]` (the greedy decode).
  - `leaf-save FLIP (lossy)` ⟺ served leaf token != `oracle_argmax_id[i]` (the same fork the flip analysis
    counts; `clear_margin` flags the >1-nat ones).
- `cat9 both-MISS` ⟺ neither spine nor any leaf accepted at d (cat9 also cuts at d).

### 2.1 Tally (per the user's metric)

| counter | definition |
|---|---|
| `spine_regressions` | events where E5 accepts the spine token at d but cat9 does NOT (E5 spine-accept ∧ cat9 not-accept at that depth). **Structurally 0 at greedy** (§1.2: `winner_lcp >= spine_lcp` always) — measured to confirm S2/S3 didn't break it. |
| `lossless_leaf_saves` | leaf-saves where E5 cut at d AND leaf == greedy (`oracle_argmax_id`) |
| `lossy_leaf_saves` | leaf-saves where E5 cut at d AND leaf != greedy = a flip |
| `net_lossless_leaf_saves` | `lossless_leaf_saves − lossy_leaf_saves − spine_regressions` |

### 2.2 PASS condition

> **cat9 is a LOSSLESS SUPERSET of E5 ⟺ `net_lossless_leaf_saves > 0` AND `spine_regressions == 0`.**

Read: cat9 never serves fewer spine tokens than E5 (no LOSS), and the leaf gains are NET lossless (more greedy-
matching saves than flips). A loose aggregate edge that is **mostly lossy flips** FAILS here — the per-event gate
strips the lossy fraction the aggregate hides. (Operational note for temp>0 / B=4: at deployed 0.6 the lossless
judge is distributional, not argmax — but the gate is run AT GREEDY where greedy branch rescue is REAL and the
ruling holds, `FR13_DIRECTION_AND_NUMBERS.md` L17; superset is shape-specific so compare cat9 vs the **same-shape**
native row, never across shapes — bug-class #12.)

### 2.3 Per-depth report (so a single bad depth is visible, not averaged away)

Emit the tally **per depth d=0..4** as well as totaled. Rationale: `FR13_ACCEPTANCE_LADDER_BIND.md` R2 shows
the deficit is d0-concentrated and the depth-1 sibling is dead (62% of rejects at step-0). The gate must show
WHERE the net saves and the flips land, or it repeats the **scalar-metric blind-spot** (a small per-depth defect
hides in the band — `reference_scalar_metric_per_token_blindspot`).

---

## 3. Reconciliation with the aggregate +0.12 edge — how much is lossless

The aggregate accept/event edge decomposes EXACTLY (per-event sum / N_events):

```
aggregate_edge  =  (cat9 accept/event)  −  (E5 accept/event)
               ≈   Σ_events (winner_lcp − spine_lcp) / N        # the GROSS leaf-saves per event
               =   gross_leaf_saves_per_event
```

because `winner_lcp = spine_lcp + (leaf save count)` every event (§1.2). The **gate then strips the lossy
fraction**:

```
net_lossless_per_event = (lossless_leaf_saves − lossy_leaf_saves − spine_regressions) / N
```

So the aggregate +0.12 is the **GROSS** number; the gate's net is the **LOSSLESS** number. If most of the gross
saves are flips, `net ≤ 0` ⇒ **the loose superset hid a lossy gain** even though the aggregate looked positive.

**Quantify from banked fork data (MEASURED, small-sample, bug-class #12 caveat):**
- cat9+K1 clear-margin flips vs the recurrent oracle = **20 / 466 = 4.29%** (`output/fr13_recurrent_oracle/
  rescore_cat9.json`; `FR13_DIFFUSE_BIGDENOM_TEST_PLAN.md` §1). Native floor ≈ 0.6%.
- Realized branch upside at greedy = **+35 genuine branch accepts** over the 163-event window
  (`FR13_ACCEPTANCE_LADDER_BIND.md` R3, `8add39e6`) ⇒ gross leaf-saves ≈ 35.
- The flips ARE the lossy leaf-saves (a leaf-save that != greedy IS a fork that != oracle). So a first-order
  banked estimate: of ~35 gross saves, the ≤20 clear-margin flips (some of which are these forks, some are
  spine-realization flips per the concurrent directional-asymmetry run) are the lossy candidates; the rest are
  lossless. **The exact lossless/lossy split is NOT yet computed** — it requires running the §2 classifier over
  the JOINED dump+oracle (next section). This is the precise quantity the gate produces; do not assert it.

**Bug-class #9 (vacuous) guard for the reconciliation:** the aggregate 3.198/3.076 was cross-trajectory; the
per-event net is the binding number. Quoting only the aggregate edge as "superset proven" is the vacuous-pass
trap — `evaluate_total_acceptance_gate` (averages) PASSES on a lossy gain; `evaluate_superset_hard_gate`
(per-event `tree>=path0`) catches spine regression but is BLIND to lossless-vs-flip on the saves. The §2 gate is
the missing classifier.

---

## 4. Runnable-now vs new harness

### 4.1 RUNNABLE NOW (no new GPU, no new instrument — all banked / built)
- **cat9 per-event accept trace:** the `FR13_FORK_MARGIN_DUMP` jsonl from any cat9 boot (spine_lcp, winner_lcp,
  best_leaf, committed_row, per-node margins). Banked dumps exist from the acceptance-ladder + gold-gate boots.
- **Greedy reference / lossless judge:** `fr13_recurrent_decode_oracle.py rescore` over the cat9 served stream
  (banked `output/fr13_recurrent_oracle/rescore_cat9.json` for the 4×128 window).
- **The JOIN:** `scripts/fr13_fork_margin_classify.py` (dump↔served-position map, fail-loud non-vacuity).
- **The aggregate / per-event structural superset:** `src/lumo_flywheel_serving/fr10_superset_gate.py`
  (`evaluate_superset_hard_gate` = per-event `tree>=path0` = spine-non-regression; `load_spec_trace` for the
  native aggregate anchor) + `scripts/fr10_superset_gate_report.py`.

### 4.2 NEW REDUCER NEEDED (CPU-only, ~1 file, no GPU)
A small reducer `fr13_perevent_superset_gate.py` that:
1. loads the cat9 `FR13_FORK_MARGIN_DUMP` jsonl + the recurrent-oracle rescore json + the served capture;
2. reuses `fr13_fork_margin_classify.py`'s JOIN to map every served position → its dump event + depth d;
3. for each position emits the §2 cell {E5 spine-accept/cut} × {cat9 spine-accept / leaf-save-lossless /
   leaf-save-flip / both-miss}, using `oracle_argmax_id` as the greedy judge;
4. tallies `spine_regressions / lossless_leaf_saves / lossy_leaf_saves / net`, per depth and total, and emits
   PASS = net>0 ∧ spine_regressions==0;
5. emits the §3 reconciliation: `gross_leaf_saves_per_event` (= aggregate edge) vs `net_lossless_per_event`,
   with the lossless fraction = lossless_leaf_saves / gross.
   (Class #9 asserts: every scored position joins a real dump step; oracle rep1==rep2; flag-state header.)

### 4.3 MINIMAL GPU SPEC (reuse the big-denom streams — no bespoke boot)
The per-event gate needs, on the SAME reference stream:
- **cat9 served stream + per-event `FR13_FORK_MARGIN_DUMP`** (one cat9 boot, `FR13_FORK_MARGIN_DUMP=1`,
  `FR13_FORK_MARGIN_DUMP_PATH=/logs/...`, eager — these are already the big-denom serve's flags via
  `LUMO_PROXY_PAIR_DUMP_DIR` + the dump path; the dump is READ-ONLY so the served stream is the locked path).
- **recurrent-oracle rescore** of that served stream (CPU/GPU offline `vllm.LLM`, no spec) → the greedy judge.
So the minimal GPU run = **the big-denominator cat9 serve already in flight** with `FR13_FORK_MARGIN_DUMP=1`
co-armed, then the offline recurrent-oracle rescore. **No new E5 boot is required for the gate** (the E5 spine
arm = cat9's own `spine_lcp` per §1.2/§1.3); a native free-run is needed ONLY for the absolute-aggregate anchor
in §3, and that is already banked (`fr10_native_mtp5_same8_*`). If the big-denom serve did NOT co-arm the fork
dump, ONE additional cat9 boot with `FR13_FORK_MARGIN_DUMP=1` on the SAME pinned prompts (capture-once,
`reference_capture_once_native_pin_prompt`) closes it. Estimated GPU: 0 new boots if co-armed; else 1 cat9 boot.

---

## 5. MEASURED vs CODE-READ vs INFERRED (honesty ledger)

- **CODE-READ (source, this session):** the cat9 dump fields + their semantics (`fr10_phase4_patch_vllm_tree_
  gdn.py` L7194-7356); the committer `best_lcp = max-over-leaves`, strict tie-break, `winner_lcp >= spine_lcp`
  structural guarantee (L6894-6925); the recurrent-oracle path + argmax semantics (`fr13_recurrent_decode_
  oracle.py` L16-52, L372-415); the existing JOIN (`fr13_fork_margin_classify.py`); the superset-gate evaluators
  (`fr10_superset_gate.py` L250-309); native accept trace format (`per_req_spec_trace.jsonl` `acc`).
- **MEASURED (banked, small-sample, #12-caveated):** cat9+K1 clear-margin flips 20/466 = 4.29%; native ~0.6%;
  +35 branch accepts over 163 events; aggregate cat9≈3.198 vs native 3.076 (cross-trajectory).
- **INFERRED (to be produced by the §4.2 reducer — NOT yet a number):** the lossless-vs-lossy split of the
  gross leaf-saves; `net_lossless_leaf_saves`; per-depth tally; PASS/FAIL. The §3 "≤20 of ~35 saves are the
  lossy candidates" is a first-order banked ESTIMATE, not the computed split.

---

## 6. Bug-class playbook rows quoted (mandatory)

- **#12 Measurement traps:** *"non-like-for-like trajectories after fixes; per-pos counters indexing accepted-
  path-length ('branches added 0' artifact); single-draw floors … raw counters only; capture-once pinned
  prompts; source-index traces; label every estimate."* → this gate is per-event SAME-reference (not cross-
  trajectory aggregate), uses the raw `spine_lcp`/`winner_lcp` source counters, pins prompts, labels every
  estimate as MEASURED/CODE-READ/INFERRED.
- **#9 Silent fallback / vacuous instrument:** *"a run 'passes' while measuring nothing; engagement asserts …
  fail-loud on disengagement."* → the JOIN asserts every scored position lands on a real dump step; the oracle
  asserts the recurrent decode path fired + GDN present + rep1==rep2; flag-state header in the artifact; the
  aggregate-only "superset proven" is explicitly flagged as the vacuous-pass trap (§3).
- (Carried, relevant) **#10/#11:** the lossless judge is the deployment recurrent decode oracle, not a serial-
  torch ref / chunked prefill / backend name; int-view never atol; cat9's spine drafts are NOT byte-identical
  to a standalone native run (S3/BI-asymmetry) — the apple is the within-event spine vs tree.
