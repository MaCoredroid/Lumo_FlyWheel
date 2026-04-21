#!/usr/bin/env python3
"""
CNB-55 Track 10 — proposal-ranking-manager-judgment scorer (v2).

Scores a manager-framed brief produced by the family's structured-output CLI
(`./bin/cnb55-brief submit brief_input.json`), which writes the canonical
`brief/manager_brief.json` that this scorer reads.

Drops all markdown regex extraction. Citation checks, constraint-tag checks,
mitigation checks, and anti-pattern ceilings all read structured JSON fields.
The human-readable `brief/manager_brief.md` is rendered by the CLI and is not
consulted by the grader.

Result schema (unchanged):

    {
      "pass": bool,
      "score": int,            # 0..100
      "raw_score_pre_ceiling": int,
      "milestones": {...},
      "breakdown": {...},
      "ceilings_applied": [...],
      "errors": [...],
      "shortcut_detected": bool,
      "variant_id": str,
      "wall_clock_seconds": int
    }

Determinism: fixed under CNB55_SEED=42. All dict output keys sorted.

Contract references:
- benchmark_blueprints/families/proposal-ranking-manager-judgment/evaluator_contract.md
- benchmark_blueprints/families/proposal-ranking-manager-judgment/task_spec.md
- benchmark_blueprints/families/proposal-ranking-manager-judgment/bin/cnb55-brief
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEED = int(os.environ.get("CNB55_SEED", "42"))

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA_ROOT = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

SCHEMA_VERSION = "cnb55.manager_brief.v2"

CONSTRAINT_TAGS = {
    "staffing", "rollout", "regression", "perf", "release", "incident",
}

# Kept in sync with bin/cnb55-brief.
MITIGATION_KEYWORDS = (
    "feature flag", "kill switch", "shadow", "canary",
    "staged rollout", "rollback", "observability", "slo",
    "pre-warm", "prewarm", "mitigate", "gate", "gated",
)

VALID_PROPOSAL_IDS_BY_VARIANT = {
    "v1-clean-baseline": {"P1", "P2", "P3", "P4"},
    "v2-noisy-distractor": {"P1", "P2", "P3", "P4", "P5"},
    "v3-dirty-state": {"P1", "P2", "P3", "P4", "P5"},
    "v4-multi-corpus-objective": {"P1", "P2", "P3", "P4", "P5"},
    "v5-recovery-in-thread": {"P1", "P2", "P3", "P4", "P5"},
}

TAU_THRESHOLD_BY_VARIANT = {
    # Tightened for attempt_02b hardening. N=4 variants admit discrete tau-b
    # values {1.0, 0.67, 0.33, 0.0, ...}; N=5 admit {1.0, 0.80, 0.60, ...}.
    # V1 (N=4): 0.67 → at most one inversion vs gold.
    # V2–V5 (N=5): 0.80 → at most one inversion vs gold.
    "v1-clean-baseline": 0.67,
    "v2-noisy-distractor": 0.80,
    "v3-dirty-state": 0.80,
    "v4-multi-corpus-objective": 0.80,
    "v5-recovery-in-thread": 0.80,
}

# Keywords used to detect an in-place staleness flag on an accepted entry
# that cites a stale_perf_marker path.
STALE_PERF_FLAG_KEYWORDS = (
    "stale", "outdated", "superseded", "out of date", "out-of-date",
    "pre-streaming", "pre streaming", "jan 2026", "january 2026", "jan_bench",
    "january bench", "jan bench",
)

# Keywords that acknowledge the mid-quarter staffing update for Priya.
STAFFING_UPDATE_KEYWORDS = (
    "40%", "parental", "reduced availability", "reduced capacity",
    "mid-quarter", "mid quarter", "handoff", "ramp-down", "ramp down",
    "staffing_update", "staffing update", "priya",
)

# Relative file trees that must NOT be modified by the agent.
READONLY_TREES = (
    "proposals",
    "repo_evidence",
    "release_context",
    "incident_context",
    "tests",
    "AGENTS.md",
    "Dockerfile",
    ".scenario_variant",
)

SHIM_FILES = (
    "sitecustomize.py",
    "usercustomize.py",
    "pytest.py",
)


# --------------------------------------------------------------------- utils


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def sha256_of_tree(root: Path, rel: str) -> str:
    """Stable content-addressable hash of a file or directory tree."""
    target = root / rel
    if not target.exists():
        return "MISSING"
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_of_file(target).encode())
        return h.hexdigest()
    for p in sorted(target.rglob("*")):
        rel_p = p.relative_to(target).as_posix()
        if p.is_file():
            h.update(b"F:" + rel_p.encode() + b"\x00")
            h.update(sha256_of_file(p).encode() + b"\x00")
        elif p.is_dir():
            h.update(b"D:" + rel_p.encode() + b"\x00")
    return h.hexdigest()


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall tau-b between two rankings. Returns value in [-1, 1]."""
    common = [x for x in a if x in b]
    if len(common) < 2:
        return 0.0
    rank_a = {x: i for i, x in enumerate(a) if x in common}
    rank_b = {x: i for i, x in enumerate(b) if x in common}
    n = len(common)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = common[i], common[j]
            da = rank_a[x] - rank_a[y]
            db = rank_b[x] - rank_b[y]
            prod = da * db
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def evidence_file_set() -> set[str]:
    ev: set[str] = set()
    for rel in ("proposals", "repo_evidence", "release_context", "incident_context"):
        base = AGENT_WS / rel
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file():
                    ev.add(p.relative_to(AGENT_WS).as_posix())
    return ev


# --------------------------------------------------------------------- state


@dataclass
class ScorerState:
    errors: list[str] = field(default_factory=list)
    shortcut: bool = False
    milestones: dict[str, bool] = field(default_factory=dict)
    breakdown: dict[str, int] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    raw_score: int = 0
    ceiling_cap: int = 100
    brief: dict[str, Any] | None = None  # canonical brief/manager_brief.json

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add(self, key: str, points: int) -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.raw_score += points

    def apply_ceiling(self, name: str, cap: int) -> None:
        self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def final(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))


# --------------------------------------------------------------------- stages


def load_gold(variant_id: str) -> dict[str, Any]:
    path = VERIFIER_DATA_ROOT / variant_id / "gold_ranking.json"
    return json.loads(path.read_text())


def trusted_final_state(state: ScorerState, gold: dict[str, Any]) -> None:
    """Pass/fail gates from evaluator_contract.md §Trusted Final-State Checks."""

    # (1) tests/test_ranking_brief.py checksum unchanged
    expected_sha = gold.get("test_ranking_brief_sha256")
    if expected_sha:
        try:
            actual = sha256_of_file(AGENT_WS / "tests" / "test_ranking_brief.py")
        except Exception:
            actual = None
        if actual and actual != expected_sha:
            state.apply_ceiling("tests_modified", 0)
            state.shortcut = True
            state.add_error("tests/test_ranking_brief.py was modified")

    # (2) No shim files
    for shim in SHIM_FILES:
        if (AGENT_WS / shim).exists():
            state.apply_ceiling(f"shim:{shim}", 0)
            state.shortcut = True
            state.add_error(f"disallowed shim file present: {shim}")

    # (3) Readonly trees unchanged
    expected_tree_hashes = gold.get("readonly_tree_hashes", {})
    for rel, expected in expected_tree_hashes.items():
        actual = sha256_of_tree(AGENT_WS, rel)
        if expected and actual != expected:
            state.apply_ceiling(f"readonly_tree:{rel}", 0)
            state.shortcut = True
            state.add_error(f"readonly tree mutated: {rel}")

    # (4) No files outside brief/ were written.
    #
    # Allowed writes:
    #   * anything under brief/ (agent output directory)
    #   * brief_input.json at workspace root — the CLI input file that the
    #     SKILL instructs agents to author; ./bin/cnb55-brief reads it and
    #     renders brief/manager_brief.{json,md}. It is a legitimate on-path
    #     artifact and must not be conflated with stray writes.
    allowed_write_prefixes = ("brief/",)
    allowed_write_files = {"brief_input.json"}
    manifest_path = VERIFIER_DATA_ROOT / VARIANT_ID / "workspace_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        known = set(manifest.get("files", []))
        for p in AGENT_WS.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(AGENT_WS).as_posix()
            if rel in known:
                continue
            if any(rel.startswith(pfx) for pfx in allowed_write_prefixes):
                continue
            if rel in allowed_write_files:
                continue
            state.apply_ceiling("wrote_outside_brief", 0)
            state.shortcut = True
            state.add_error(f"wrote file outside brief/: {rel}")
            break

    # (5) Network egress marker
    if (AGENT_WS / ".network_egress_detected").exists():
        state.apply_ceiling("network_egress", 0)
        state.shortcut = True
        state.add_error("network egress marker present")


def read_brief(state: ScorerState) -> None:
    """Read canonical brief/manager_brief.json (produced by cnb55-brief submit)."""
    brief_path = AGENT_WS / "brief" / "manager_brief.json"
    if not brief_path.exists() or brief_path.stat().st_size == 0:
        state.apply_ceiling("no_brief_file", 0)
        state.add_error(
            "brief/manager_brief.json missing or empty — "
            "agent must run `./bin/cnb55-brief submit brief_input.json`"
        )
        return
    try:
        doc = json.loads(brief_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        state.apply_ceiling("malformed_brief", 10)
        state.add_error(f"brief/manager_brief.json not valid JSON: {e}")
        return
    if not isinstance(doc, dict):
        state.apply_ceiling("malformed_brief", 10)
        state.add_error("brief/manager_brief.json top-level must be an object")
        return
    if doc.get("schema_version") != SCHEMA_VERSION:
        state.apply_ceiling("malformed_brief", 10)
        state.add_error(
            f"brief schema_version must be {SCHEMA_VERSION}, got {doc.get('schema_version')!r}"
        )
        return
    state.brief = doc
    state.add("phase2.brief_exists", 3)
    state.milestones["brief_exists"] = True


def phase2_structural(state: ScorerState, gold: dict[str, Any]) -> None:
    """Phase 2 visible pytest-style structural gates (post-CLI, most are guaranteed)."""
    if state.brief is None:
        return
    brief = state.brief

    # Brief parses (CLI guarantees it's JSON; award the 5 parse points).
    state.add("phase2.brief_parses", 5)
    state.milestones["brief_parses"] = True

    # Ranking length equals number of proposals for this variant.
    ranking = brief.get("ranking")
    expected_len = len(gold["ranking"])
    if isinstance(ranking, list) and len(ranking) == expected_len:
        state.add("phase2.ranking_length", 4)
        state.milestones["ranking_length"] = True
    else:
        state.add_error(
            f"ranking length mismatch: expected {expected_len}, got "
            f"{'not a list' if not isinstance(ranking, list) else len(ranking)}"
        )

    # Accepted is valid for this variant.
    accepted = brief.get("accepted")
    valid = VALID_PROPOSAL_IDS_BY_VARIANT.get(VARIANT_ID, set())
    if isinstance(accepted, str) and accepted in valid:
        state.add("phase2.accepted_valid", 3)
        state.milestones["accepted_valid"] = True
    else:
        state.add_error(f"accepted={accepted!r} not a valid proposal id for {VARIANT_ID}")

    # Entry fields (CLI guarantees fields exist + summary is >=12 chars).
    if isinstance(ranking, list) and ranking and all(
        isinstance(e, dict)
        and {"proposal_id", "rank", "summary", "citations", "constraint_tags"} <= set(e.keys())
        for e in ranking
    ):
        state.add("phase2.entry_fields", 3)
        state.milestones["entry_fields"] = True
    else:
        state.add_error("one or more ranking entries missing required fields")

    # Assumption ledger has a 'missing' row (CLI guarantees, but verify).
    al = brief.get("assumption_ledger")
    has_missing = (
        isinstance(al, list)
        and any(isinstance(r, dict) and r.get("status") == "missing" for r in al)
    )
    if has_missing:
        state.add("phase2.assumption_ledger", 3)
        state.milestones["assumption_ledger"] = True
    else:
        state.add_error("assumption_ledger missing or no row has status=missing")

    # variant_id matches the .scenario_variant on disk.
    sv_path = AGENT_WS / ".scenario_variant"
    if sv_path.exists():
        expected_variant = sv_path.read_text().strip()
        if brief.get("variant_id") == expected_variant:
            state.milestones["variant_id_matches"] = True
        else:
            state.add_error(
                f"variant_id={brief.get('variant_id')!r} does not match "
                f".scenario_variant={expected_variant!r}"
            )

    # Phase 2 trust bonus (positive analogue of shortcut ceilings).
    # Attempt_02b hardening: these are baseline "did not shoot yourself in
    # the foot" bonuses; shrinking them narrows the easy-points floor so
    # real ranking quality drives the score. Kept at 3 each so the oracle
    # reaches the §10.1 freeze-gate target of >= 90.
    if not any(c == "wrote_outside_brief" for c in state.ceilings_applied):
        state.add("phase2.no_stray_files", 3)
        state.milestones["no_stray_files"] = True
    if not any(c.startswith("shim:") for c in state.ceilings_applied):
        state.add("phase2.no_shim", 3)
        state.milestones["no_shim"] = True


def score_ranking_alignment(state: ScorerState, gold: dict[str, Any]) -> None:
    """Differential checks against the gold ranking."""
    if state.brief is None:
        return
    ranking = state.brief.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return

    ordered_ids = [
        e["proposal_id"] for e in ranking
        if isinstance(e, dict) and isinstance(e.get("proposal_id"), str)
    ]
    gold_order = [e["proposal_id"] for e in gold["ranking"]]

    # Accepted match
    if state.brief.get("accepted") == gold["accepted"]:
        state.add("behavioral.accepted_match", 10)
        state.milestones["accepted_match"] = True

    # Top-2 set (unordered)
    top2 = set(ordered_ids[:2])
    gold_top2 = set(gold_order[:2])
    if top2 == gold_top2:
        state.add("behavioral.top2_match", 4)
        state.milestones["top2_match"] = True

    # Kendall tau against gold
    tau = kendall_tau(ordered_ids, gold_order)
    state.breakdown["differential.kendall_tau_raw_x100"] = int(round(tau * 100))
    tau_threshold = TAU_THRESHOLD_BY_VARIANT[VARIANT_ID]
    if tau >= tau_threshold:
        state.add("differential.kendall_tau", 8)
        state.milestones["kendall_tau"] = True

    # Staffing-blocked proposal not promoted above its gold rank
    blocked = gold.get("staffing_blocked_proposal")
    if blocked and blocked in ordered_ids and blocked in gold_order:
        submitted_rank = ordered_ids.index(blocked)
        gold_rank = gold_order.index(blocked)
        if submitted_rank >= gold_rank:
            state.add("differential.staffing_respected", 4)
            state.milestones["staffing_respected"] = True
        else:
            state.add_error(
                f"staffing-blocked proposal {blocked} promoted above gold rank"
            )


def score_grounding(state: ScorerState, gold: dict[str, Any]) -> None:
    """Structured citation + constraint-tag grounding checks."""
    if state.brief is None:
        return
    ranking = state.brief.get("ranking") or []
    accepted = state.brief.get("accepted")

    rejection_entries = [
        e for e in ranking
        if isinstance(e, dict) and e.get("proposal_id") != accepted
    ]

    evidence_files = evidence_file_set()

    # Property: each rejection cites >=1 existing evidence file.
    cites = 0
    bad_cites: list[str] = []
    for e in rejection_entries:
        cits = e.get("citations") or []
        if not isinstance(cits, list):
            continue
        e_good = [c for c in cits if isinstance(c, str) and c.lstrip("./") in evidence_files]
        e_bad = [c for c in cits if isinstance(c, str) and c.lstrip("./") not in evidence_files]
        bad_cites.extend(e_bad)
        if e_good:
            cites += 1
    fraction = cites / max(1, len(rejection_entries))
    if fraction >= 0.8:
        state.add("property.rejection_cites_evidence", 6)
        state.milestones["rejection_cites_evidence"] = True
    elif fraction >= 0.5:
        state.add("property.rejection_cites_evidence_partial", 3)

    if cites < 2:
        state.apply_ceiling("ranking_without_grounding", 25)

    # Regression: cited files exist on disk.
    if not bad_cites:
        state.add("regression.citations_valid", 3)
        state.milestones["citations_valid"] = True
    else:
        state.add_error(f"brief cites non-existent files: {bad_cites}")

    # Property: each rejection has >= 1 constraint tag.
    tagged = 0
    for e in rejection_entries:
        tags = e.get("constraint_tags") or []
        if isinstance(tags, list) and any(t in CONSTRAINT_TAGS for t in tags):
            tagged += 1
    if rejection_entries and tagged == len(rejection_entries):
        state.add("property.constraint_tagged", 4)
        state.milestones["constraint_tagged"] = True
    elif tagged >= max(1, len(rejection_entries) // 2):
        state.add("property.constraint_tagged_partial", 2)

    # Property: primary_risk statement non-trivial.
    # Attempt_02b hardening: raised minimum length from 20 to 40 chars so a
    # one-clause placeholder no longer earns the 3 points.
    pr = state.brief.get("primary_risk") or {}
    statement = pr.get("statement", "") if isinstance(pr, dict) else ""
    if isinstance(statement, str) and len(statement.strip()) >= 40:
        state.add("property.primary_risk_nontrivial", 3)
        state.milestones["primary_risk_nontrivial"] = True

    # Property: accepted entry has >= 3 citations (grounding depth).
    # Rationale: a defensible accept in this family should lean on at least
    # (a) the proposal file, (b) perf/staffing evidence, and (c) a rollout
    # or release artefact. Entries with fewer citations reflect thin
    # grounding and lose the 3 points.
    accepted = state.brief.get("accepted")
    accepted_entry = next(
        (e for e in (state.brief.get("ranking") or [])
         if isinstance(e, dict) and e.get("proposal_id") == accepted),
        None,
    )
    if accepted_entry is not None:
        cits = accepted_entry.get("citations") or []
        if isinstance(cits, list) and len([c for c in cits if isinstance(c, str)]) >= 3:
            state.add("property.accepted_citations_depth", 3)
            state.milestones["accepted_citations_depth"] = True


def score_risk_mitigation(state: ScorerState, gold: dict[str, Any]) -> None:
    """Structured primary_risk.mitigations keyword check.

    Attempt_02b hardening: require >= 3 distinct mitigation keywords (was
    >= 2) for the full 8 points. Two keywords earns partial 4. One keyword
    earns partial 2. The set of keywords is printed by
    `./bin/cnb55-brief schema` and documented in the SKILL.
    """
    if state.brief is None:
        return
    pr = state.brief.get("primary_risk") or {}
    mitigations = pr.get("mitigations", []) if isinstance(pr, dict) else []
    if not isinstance(mitigations, list):
        return
    combined = " ".join(m for m in mitigations if isinstance(m, str)).lower()
    hits = sum(1 for kw in MITIGATION_KEYWORDS if kw in combined)
    if hits >= 3:
        state.add("plan.risk_mitigation_paired", 8)
        state.milestones["risk_mitigation_paired"] = True
    elif hits == 2:
        state.add("plan.risk_mitigation_partial", 4)
    elif hits == 1:
        state.add("plan.risk_mitigation_weak", 2)


def score_variant_ceilings(state: ScorerState, gold: dict[str, Any]) -> None:
    """Variant-dependent ceilings + objective/incident bonuses.

    Attempt_02b hardening:
      * `ignored_stale_perf` cap 35 → 25, and the staleness flag must
        appear in the SAME entry's summary or in primary_risk (not just
        anywhere in the brief corpus — a ledger aside is not enough).
      * `sunk_cost_finish` cap 40 → 30.
      * New `missed_staffing_update` ceiling (cap 40): fires when the
        accepted proposal's author is affected by the mid-quarter
        staffing update AND the brief neither cites the update file nor
        uses any staffing-update keyword anywhere in the corpus.
      * New `missed_watermark_assumption` ceiling (cap 55): V5 only.
        Fires when no assumption_ledger row is both status="missing" AND
        has "watermark" in topic or note. The incident is real; a
        competent brief must explicitly flag the unknown watermark
        redesign timeline as an open assumption.
    """
    if state.brief is None:
        return
    brief = state.brief
    accepted = brief.get("accepted")
    ranking = brief.get("ranking") or []

    # Find accepted entry's summary + citations (used by ceiling checks).
    accepted_entry = next(
        (e for e in ranking
         if isinstance(e, dict) and e.get("proposal_id") == accepted),
        {},
    )
    accepted_summary = (accepted_entry.get("summary") or "").lower()
    accepted_citations = accepted_entry.get("citations") or []

    # Brief-wide text corpus for keyword scans (concat structured fields).
    corpus_parts: list[str] = []
    for e in ranking:
        if not isinstance(e, dict):
            continue
        corpus_parts.append(str(e.get("summary", "")))
    pr = brief.get("primary_risk") or {}
    pr_text_parts: list[str] = []
    if isinstance(pr, dict):
        pr_text_parts.append(str(pr.get("statement", "")))
        for m in pr.get("mitigations", []) or []:
            pr_text_parts.append(str(m))
        corpus_parts.extend(pr_text_parts)
    for row in brief.get("assumption_ledger", []) or []:
        if isinstance(row, dict):
            corpus_parts.append(str(row.get("topic", "")))
            corpus_parts.append(str(row.get("note", "")))
    brief_corpus = "\n".join(corpus_parts).lower()
    pr_text = "\n".join(pr_text_parts).lower()

    # --- ceilings driven by accepted proposal ---

    # Ignored staffing constraint: accepted == the staffing-blocked proposal
    blocked = gold.get("staffing_blocked_proposal")
    if blocked and accepted == blocked:
        state.apply_ceiling("ignored_staffing_constraint", 30)

    # Objective drift (V4+): accepted is the old-objective pick
    old_objective_preferred = gold.get("old_objective_preferred")
    if old_objective_preferred and accepted in old_objective_preferred:
        state.apply_ceiling("objective_drift", 45)

    # Incident-blind reselect (V5): accepted == rolled-back proposal without incident_ref
    rolled_back = gold.get("rolled_back_proposal")
    incident_ref = gold.get("incident_ref")
    if rolled_back and accepted == rolled_back:
        mentions_incident = bool(incident_ref) and incident_ref in " ".join(corpus_parts)
        if not mentions_incident:
            state.apply_ceiling("incident_blind_reselect", 30)

    # Sunk-cost finish (V3+): accepted == in-progress patch target w/o abandoned-language
    sunk_cost_trap = gold.get("sunk_cost_trap_proposal")
    if sunk_cost_trap and accepted == sunk_cost_trap:
        if not re.search(
            r"(maintainability|abandoned|sunk[- ]?cost|in[- ]?progress|not a head start)",
            brief_corpus,
        ):
            state.apply_ceiling("sunk_cost_finish", 30)

    # V2+ stale perf ceiling: accepted entry cites a stale_perf_marker path
    # without the brief flagging staleness in the SAME entry or in
    # primary_risk. A ledger row elsewhere is not enough — the flag must
    # live with the citation it qualifies.
    stale_perf_markers = gold.get("stale_perf_markers") or []
    if stale_perf_markers:
        cites_stale = any(
            isinstance(c, str) and any(marker in c.lower() for marker in stale_perf_markers)
            for c in accepted_citations
        )
        local_flag_text = (accepted_summary + "\n" + pr_text)
        flags_local = any(w in local_flag_text for w in STALE_PERF_FLAG_KEYWORDS)
        if cites_stale and not flags_local:
            state.apply_ceiling("ignored_stale_perf", 25)

    # V1+ missed staffing update ceiling.
    # Gold field `staffing_update_owner_proposal` names the proposal whose
    # author was affected. `staffing_update_file` is the canonical path
    # under repo_evidence/. The ceiling fires only when accepted == that
    # proposal AND the brief neither cites the file nor uses any keyword
    # from STAFFING_UPDATE_KEYWORDS anywhere in the corpus.
    su_prop = gold.get("staffing_update_owner_proposal")
    su_file = gold.get("staffing_update_file")
    if su_prop and accepted == su_prop:
        cites_update = False
        if su_file:
            # Check across ALL citations in all entries (brief-wide).
            for e in ranking:
                if not isinstance(e, dict):
                    continue
                for c in (e.get("citations") or []):
                    if isinstance(c, str) and su_file in c:
                        cites_update = True
                        break
                if cites_update:
                    break
        mentions_update = any(kw in brief_corpus for kw in STAFFING_UPDATE_KEYWORDS)
        if not (cites_update or mentions_update):
            state.apply_ceiling("missed_staffing_update", 40)

    # V5 missed_watermark_assumption ceiling.
    if gold.get("requires_watermark_assumption"):
        ledger = brief.get("assumption_ledger") or []
        has_wm_missing = False
        for row in ledger:
            if not isinstance(row, dict):
                continue
            if row.get("status") != "missing":
                continue
            topic = str(row.get("topic", "")).lower()
            note = str(row.get("note", "")).lower()
            if "watermark" in topic or "watermark" in note:
                has_wm_missing = True
                break
        if not has_wm_missing:
            state.apply_ceiling("missed_watermark_assumption", 55)

    # --- positive variant bonuses ---

    # V4+ current-objective acknowledgement.
    current_objective = gold.get("current_objective")
    if current_objective:
        keywords = current_objective.get("keywords", []) or []
        # Accept a hit in either the accepted_summary OR the primary_risk /
        # mitigations (agents often put objective rationale in primary_risk).
        haystack = accepted_summary + "\n" + brief_corpus
        if any(kw.lower() in haystack for kw in keywords):
            state.add("objective.acknowledges_current", 6)
            state.milestones["objective_acknowledged"] = True

    # V5 incident anchoring.
    if incident_ref and incident_ref in " ".join(corpus_parts):
        state.add("incident.anchored", 5)
        state.milestones["incident_anchored"] = True


def score_partial_progress_heuristic(state: ScorerState, gold: dict[str, Any]) -> None:
    """
    Deterministic stand-in for the LLM-judge partial_progress rubric (0-10).
    The judge rubric at verifier_data/_rubrics/partial_progress.md remains
    authoritative; this is the floor.

    Dimensions (each 0 or 2, max 10):
      D1 close-call / trade-off framing, D2 risk-mitigation pairing,
      D3 stale-evidence handling, D4 objective reframing, D5 incident anchoring.
    """
    if state.brief is None:
        return
    ranking = state.brief.get("ranking") or []
    pr = state.brief.get("primary_risk") or {}
    corpus_parts: list[str] = []
    for e in ranking:
        if isinstance(e, dict):
            corpus_parts.append(str(e.get("summary", "")))
    if isinstance(pr, dict):
        corpus_parts.append(str(pr.get("statement", "")))
        for m in pr.get("mitigations", []) or []:
            corpus_parts.append(str(m))
    for row in state.brief.get("assumption_ledger", []) or []:
        if isinstance(row, dict):
            corpus_parts.append(str(row.get("note", "")))
    text = "\n".join(corpus_parts).lower()

    score = 0

    # D1 close-call / trade-off / vs framing.
    if re.search(r"(close[- ]?call|trade[- ]?off|\bvs\b|versus)", text):
        score += 2

    # D2 risk-mitigation pairing (already awarded as milestone).
    if state.milestones.get("risk_mitigation_paired"):
        score += 2

    # D3 contradictory-evidence handling (V2+).
    if gold.get("stale_perf_markers"):
        if any(w in text for w in ("stale", "outdated", "superseded", "out of date")):
            score += 2
    else:
        score += 2  # free for V1

    # D4 objective reframing (V4+).
    if gold.get("current_objective"):
        if state.milestones.get("objective_acknowledged"):
            score += 2
    else:
        score += 2

    # D5 incident anchoring (V5).
    if gold.get("incident_ref"):
        if state.milestones.get("incident_anchored"):
            score += 2
    else:
        score += 2

    state.add("partial_progress.heuristic", score)
    state.milestones["partial_progress_heuristic"] = score >= 6


# --------------------------------------------------------------------- main


def main() -> int:
    start = time.time()
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    state = ScorerState()
    try:
        gold = load_gold(VARIANT_ID)
    except Exception as e:
        state.add_error(f"gold_ranking.json load failure: {e}")
        RESULT_FILE.write_text(
            json.dumps(
                {
                    "pass": False,
                    "score": 0,
                    "milestones": {},
                    "breakdown": {},
                    "ceilings_applied": [],
                    "errors": state.errors,
                    "shortcut_detected": False,
                    "wall_clock_seconds": int(time.time() - start),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    # Stage 1: trusted final-state (pass/fail gates).
    trusted_final_state(state, gold)

    # Stage 2: read canonical JSON brief.
    read_brief(state)

    # Stage 3: structural phase-2 + additive scoring.
    if state.brief is not None:
        phase2_structural(state, gold)
        score_ranking_alignment(state, gold)
        score_grounding(state, gold)
        score_risk_mitigation(state, gold)
        score_variant_ceilings(state, gold)
        score_partial_progress_heuristic(state, gold)

    final_score = state.final()

    pass_bar = gold.get("pass_bar", 60)
    passed = (
        not state.shortcut
        and state.milestones.get("brief_exists", False)
        and state.milestones.get("brief_parses", False)
        and final_score >= pass_bar
    )

    result = {
        "pass": bool(passed),
        "score": int(final_score),
        "raw_score_pre_ceiling": int(state.raw_score),
        "milestones": dict(sorted(state.milestones.items())),
        "breakdown": dict(sorted(state.breakdown.items())),
        "ceilings_applied": sorted(state.ceilings_applied),
        "errors": state.errors,
        "shortcut_detected": bool(state.shortcut),
        "variant_id": VARIANT_ID,
        "wall_clock_seconds": int(time.time() - start),
    }

    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True))
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
