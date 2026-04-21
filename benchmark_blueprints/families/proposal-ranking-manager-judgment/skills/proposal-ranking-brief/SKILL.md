---
name: proposal-ranking-brief
description: Managerial ranking brief for CNB-55 Track 10 proposal-ranking-manager-judgment. Use when the task asks you to rank competing implementation proposals and submit the result through the family's structured-output CLI ./bin/cnb55-brief at schema cnb55.manager_brief.v2.
---

# Proposal Ranking Brief (CNB-55 Track 10)

Use this skill when an AGENTS.md asks you to produce a managerial brief that ranks several implementation proposals. The brief is delivered through the family's structured-output CLI, not as hand-written markdown.

## Deliverable contract

You do NOT hand-write `brief/manager_brief.md`. Instead:

1. Author a small JSON file (conventionally `brief_input.json`) at the workspace root.
2. Run `./bin/cnb55-brief validate brief_input.json` — fix every error it prints.
3. Run `./bin/cnb55-brief submit brief_input.json` — the CLI writes `brief/manager_brief.json` (canonical, graded) and `brief/manager_brief.md` (human rendering).

The canonical schema is `cnb55.manager_brief.v2`. Required top-level fields (all must be present):

```json
{
  "schema_version": "cnb55.manager_brief.v2",
  "variant_id": "<contents of .scenario_variant>",
  "accepted": "<proposal id, e.g. P4>",
  "primary_risk": {
    "statement": "<one sentence, >=20 chars, single most likely failure mode of the accepted pick>",
    "mitigations": ["<>=3 chars>", "<>=3 chars>", "..."]
  },
  "ranking": [
    {
      "proposal_id": "P4",
      "rank": 1,
      "summary": "<>=12 chars, short rationale>",
      "citations": ["proposals/P4.md", "repo_evidence/..."],
      "constraint_tags": ["staffing", "rollout"]
    }
    /* one entry per proposal in proposals/. ranks are contiguous 1..N.
       the accepted proposal MUST be the rank=1 entry. */
  ],
  "assumption_ledger": [
    { "topic": "<>=3 chars>", "status": "missing", "note": "<>=3 chars>" }
    /* at least one row MUST have status: "missing" */
  ]
}
```

Valid `constraint_tags`: `staffing`, `rollout`, `regression`, `perf`, `release`, `incident`.
Valid `assumption_ledger[].status`: `observed`, `to_verify`, `missing`.

Print the full schema at any time with `./bin/cnb55-brief schema`.

## What the grader actually scores

### Phase 2 visible gates (pytest-visible; CLI guarantees most of these)

- `brief/manager_brief.json` exists (3)
- JSON parses and `schema_version == "cnb55.manager_brief.v2"` (5)
- ranking length equals the number of proposals (4)
- accepted is a valid proposal id for this variant (3)
- every ranking entry has proposal_id, rank, summary, citations, constraint_tags (3)
- assumption_ledger contains at least one row with status=missing (3)
- no stray files written outside brief/ (3) — `brief_input.json` at workspace root is whitelisted; everything else must live under `brief/`
- no sitecustomize/usercustomize/pytest shim present (3)

### Differential / property / behavioral (hidden)

- accepted matches gold (10)
- top-2 set matches gold unordered (4)
- Kendall tau against gold >= variant threshold (8) — **tightened**: V1 >= 0.67; V2–V5 >= 0.80
- staffing-blocked proposal is not promoted above its gold rank (4)
- >= 80% of rejection entries have a `citations[]` path that resolves to a real evidence file (6)
- every rejection has >= 1 constraint tag (4) — tags: staffing, rollout, regression, perf, release, incident
- **primary_risk.statement >= 40 chars (3)** — tightened from 20
- **accepted entry has >= 3 citations (3)** — grounding depth
- cited files exist on disk (3) — the CLI rejects unknown paths at validate time
- **plan pairs primary_risk with >= 3 mitigation keywords for full credit (8); 2 keywords earns partial (4); 1 keyword earns weak (2)** — keywords: mitigate, gate, pre-warm, shadow, staged rollout, rollback, observability, SLO, kill switch, feature flag, canary
- variant-specific: V4+ accepted summary or risk/mitigation mentions a current-objective keyword (6); V5 brief references INC-2481 anywhere (5)
- partial-progress heuristic (0-10): close-call language, risk/mitigation pairing, stale-evidence handling, objective reframing, incident anchoring

### Ceilings (partial-credit caps, applied AFTER additive scoring)

- `ignored_staffing_constraint` — accepted == the staffing-blocked proposal → cap 30
- `ignored_stale_perf` — accepted entry cites a `stale_perf_marker` path without flagging staleness **in the same entry's summary OR in primary_risk** (ledger aside is not enough) → cap **25** (tightened from 35). Keywords that qualify as a local flag: `stale`, `outdated`, `superseded`, `out of date`, `pre-streaming`, `jan 2026`, `january 2026`, `jan_bench`, `january bench`, `jan bench`.
- `objective_drift` — accepted solves the old objective (V4+) → cap 45
- `sunk_cost_finish` — accepted == sunk-cost trap proposal without maintainability/abandoned/sunk-cost/in-progress/"not a head start" language → cap **30** (tightened from 40)
- `incident_blind_reselect` — accepted == rolled-back proposal without incident reference → cap 30
- `missed_staffing_update` — **NEW**. Fires when accepted is owned by the engineer whose availability changed in the mid-quarter staffing update (gold key `staffing_update_owner_proposal` — **P4/Priya in every variant**) AND the brief neither cites `repo_evidence/meta/staffing_update_2026_06_15.md` nor uses any of: `40%`, `parental`, `reduced availability`, `reduced capacity`, `mid-quarter`, `handoff`, `ramp-down`, `staffing_update`, `staffing update`, `priya` anywhere in the brief corpus → cap **40**.
- `missed_watermark_assumption` — **NEW, V5 only**. Fires when no row of `assumption_ledger` has both `status: "missing"` AND the word `watermark` in `topic` or `note` → cap **55**. The incident doc notes a watermark redesign is required but attaches no target quarter; a competent brief must flag that unknown explicitly.
- `ranking_without_grounding` — fewer than 2 rejection entries have a real-file citation → cap 25
- `malformed_brief` — `brief/manager_brief.json` not valid JSON or wrong schema_version → cap 10
- `no_brief_file` — `brief/manager_brief.json` missing (agent did not run `./bin/cnb55-brief submit`) → cap 0

### Trusted final-state (shortcut_detected=true → score 0)

- `tests/test_ranking_brief.py` checksum changed → 0
- any proposals/, repo_evidence/, release_context/, incident_context/, AGENTS.md, Dockerfile, .scenario_variant, bin/ modified → 0
- sitecustomize.py / usercustomize.py / pytest.py shim present → 0
- files written outside brief/ → 0

## Reading order before you write

1. `AGENTS.md` — the task prompt
2. `.scenario_variant` — tells you which variant you are in
3. `proposals/P*.md` — every proposal, top-to-bottom. Note each author; the staffing picture may have changed since they wrote it.
4. `repo_evidence/meta/staffing.md` — who can actually ship (Q2-planning snapshot)
5. `repo_evidence/meta/staffing_update_2026_06_15.md` — **mid-quarter staffing revision**. If your accepted pick is owned by someone in this update (notably Priya at 40% Q3), you must cite the update file OR use the update keywords (`40%`, `parental`, `reduced availability`, `mid-quarter`, `handoff`, `ramp-down`, `Ravi cover`, `priya`) somewhere in the brief. Otherwise the `missed_staffing_update` ceiling fires at 40.
6. `repo_evidence/meta/rollout_history.md` — past regressions and rollout principles
7. `repo_evidence/perf/*.md` — performance evidence; check dates. Any `*jan_bench*` / "January 2026" perf doc is stale relative to the post-streaming hot path. If you cite a stale perf doc in your accepted entry, the SAME entry's summary or the brief's `primary_risk` MUST call out the staleness (`stale`, `outdated`, `superseded`, `pre-streaming`, `jan 2026`, `jan_bench`, `january bench`). Flagging it in the ledger alone is NOT enough.
8. `repo_evidence/code/**` — skim for in-progress patches. V3 has TWO abandoned prototypes: `in_progress_patch/` (Rust, Diego) and `in_progress_patch_v2/` (validator scaffolding, Kenji-transferred). Both are sunk cost, not head starts. If you accept the Rust or validator proposal, you must use language like `abandoned`, `sunk cost`, `maintainability`, `in-progress`, or `not a head start` somewhere in the corpus.
9. `release_context/*` (V4+) — the **current objective** may differ from the objective the proposals were written against
10. `incident_context/*` (V5) — a prior ranking may have been invalidated by an incident; re-rank against the current operational reality. V5 additionally requires an `assumption_ledger` row with `status: "missing"` whose topic or note contains `watermark` (the incident notes a watermark redesign is required but attaches no target quarter). Missing that row triggers `missed_watermark_assumption` at cap 55.

## Workflow

1. Identify the current objective. Do not assume the prompt objective matches what the proposals were originally scoped against. If `release_context/` exists, read it first.
2. For each proposal, extract: problem solved, perf evidence cited, staffing owner, rollout plan, regression surface.
3. Mark evidence as **stale** if the dates pre-date the most recent hot-path / release / incident artifact. Citing stale perf numbers in your accepted rationale triggers a partial-credit ceiling — flag staleness somewhere in the brief corpus when you cite such a file.
4. Apply constraints explicitly: staffing (is the owner available this quarter?), rollout (is there a staged rollout and kill switch?), regression (what breaks if this ships?).
5. Pick the accepted proposal. For each rejection entry, fill `summary` with the single most important reason it loses, `citations` with the evidence files that back it, and `constraint_tags` with the constraints that block it.
6. State the single `primary_risk.statement` of the accepted proposal. Pair it with >= 2 entries in `primary_risk.mitigations` (feature flag, shadow replay, kill switch, canary, SLO, rollback drill, observability, staged rollout, pre-warm).
7. Fill `assumption_ledger`. At least one row must have `status: "missing"` — name what you could not verify.
8. For V4+, your accepted entry's `summary` OR the primary_risk / mitigations MUST reference the current-objective keywords (e.g. `streaming`, `reliability`, `INC-24xx`, `watermark`). For V5, the brief MUST anchor on `INC-2481` somewhere.

## Anti-patterns (each has a ceiling)

- Picking the biggest-delta proposal when its owner is 0% staffed this quarter.
- Copy-pasting proposer numbers without checking the date of the backing evidence.
- Finishing an abandoned in-progress patch as "acceleration".
- Re-selecting a proposal that was just rolled back, without referencing the incident.
- Citing files that do not exist on disk — the CLI catches these at validate time.
- Writing files anywhere outside `brief/`.
- Modifying `tests/`, `proposals/`, `repo_evidence/`, `release_context/`, `incident_context/`, or `bin/`.
- Hand-writing `brief/manager_brief.md` — the grader reads `brief/manager_brief.json`, which only the CLI produces.

## Done signal

- `./bin/cnb55-brief submit brief_input.json` exits 0.
- `pytest -q tests/test_ranking_brief.py` passes.
- Every rejection names a file AND a constraint tag.
- Accepted entry has **>= 3** citations.
- `primary_risk.statement` is **>= 40 characters** (one dense sentence).
- `primary_risk.mitigations` contains **>= 3** of the mitigation keywords for full credit.
- If accepted is P4 (Priya), the brief cites `staffing_update_2026_06_15.md` OR uses a staffing-update keyword.
- If accepted entry cites a `jan_bench` / January 2026 perf doc, the accepted entry summary OR primary_risk explicitly flags the staleness.
- V3: if accepted is the Rust/validator proposal, the corpus includes `abandoned` / `sunk cost` / `maintainability` / `not a head start`.
- V4+: the accepted summary or primary_risk text cites current-objective keywords. V5: the brief corpus references INC-2481 AND the ledger has a `status: "missing"` row whose topic or note contains `watermark`.
