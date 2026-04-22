#!/usr/bin/env python3
"""
§5 verification matrix runner for the proposal-ranking-manager-judgment family.

Synthesizes 6 trajectory types against V1 and prints a 6-row × 5-metric table
(per HLD-Family-Test-Requirements §5). This is the scorer-side slice of the
matrix: P_benchmark and M_training come from the scorer directly; G, R, and
S_TTC are computed from the HLD §7 formulas assuming best-case values for
turn-level quantities the scorer does not see (C=T=0, L derived from the
agent we're simulating, H from the scorer output).

Trajectories:
  1. Oracle                       — full pass
  2. Empty                        — no brief written
  3. RAWR/grounding_stripped      — right pick, fabricated citations
  4. Pick-P3 (staffing-blocked)   — ignored_staffing_constraint ceiling
  5. Top1-wrong                   — swap accepted with a rejected
  6. Delete-tests (adversarial)   — integrity trip, H=1

Usage:
  python3 scripts/run_verification_matrix.py [--variant v1-clean-baseline]
          [--out benchmark_blueprints/.../verification_matrix.md]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAMILY = REPO / "benchmark_blueprints/families/proposal-ranking-manager-judgment"
WS_BUNDLE = FAMILY / "workspace_bundle"
VER_DATA = REPO / "verifier_data/proposal-ranking-manager-judgment"
SCORER = REPO / "verifiers/proposal-ranking-manager-judgment/score_ranking.py"


# ---------------- trajectory synthesizers ----------------


def make_oracle(ws: Path, variant: str) -> None:
    """Copy canonical oracle brief into the workspace."""
    src = VER_DATA / variant / "oracle" / "manager_brief.json"
    dst = ws / "brief" / "manager_brief.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    # Mirror md for completeness.
    md_src = VER_DATA / variant / "oracle" / "manager_brief.md"
    if md_src.exists():
        shutil.copy(md_src, ws / "brief" / "manager_brief.md")


def make_empty(ws: Path, variant: str) -> None:
    """Do nothing — workspace has no brief file."""
    # Intentionally empty.
    return


def make_grounding_stripped(ws: Path, variant: str) -> None:
    """Oracle ranking/accepted, but wipe citations[] in every rejection entry
    (RAWR grounding_stripped). Should trip ranking_without_grounding ceiling."""
    src = VER_DATA / variant / "oracle" / "manager_brief.json"
    brief = json.loads(src.read_text())
    for entry in brief.get("ranking", []):
        entry["citations"] = []
    dst = ws / "brief" / "manager_brief.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(brief, indent=2, sort_keys=True))


def make_pick_p3(ws: Path, variant: str) -> None:
    """Pick P3 (staffing-blocked in V1–V3). Keeps the oracle's citations so
    only the ceiling fires, not the grounding one."""
    src = VER_DATA / variant / "oracle" / "manager_brief.json"
    brief = json.loads(src.read_text())
    # Flip accepted to P3; rebuild the ranking so P3 is first.
    old_accepted = brief.get("accepted")
    brief["accepted"] = "P3"
    ranking = brief.get("ranking", [])
    # Move P3 to front if present, else synthesize.
    new_ranking = [e for e in ranking if e.get("proposal_id") == "P3"]
    new_ranking += [e for e in ranking if e.get("proposal_id") != "P3"]
    brief["ranking"] = new_ranking
    # Leave primary_risk alone — the oracle's value is a legitimate risk
    # statement that doesn't reference the specific accepted proposal.
    dst = ws / "brief" / "manager_brief.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(brief, indent=2, sort_keys=True))


def make_top1_wrong(ws: Path, variant: str) -> None:
    """Swap accepted with the 2nd-ranked proposal. Doesn't hit a ceiling, but
    loses accepted_match + kendall_tau points. Proxy for 'close-call miss'."""
    src = VER_DATA / variant / "oracle" / "manager_brief.json"
    brief = json.loads(src.read_text())
    ranking = brief.get("ranking", [])
    if len(ranking) >= 2:
        r = list(ranking)
        r[0], r[1] = r[1], r[0]
        brief["ranking"] = r
        brief["accepted"] = r[0].get("proposal_id")
    dst = ws / "brief" / "manager_brief.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(brief, indent=2, sort_keys=True))


def make_delete_tests(ws: Path, variant: str) -> None:
    """Oracle brief + tamper with tests/ → trip tests_modified integrity rule."""
    make_oracle(ws, variant)
    tests_dir = ws / "tests"
    if tests_dir.exists():
        # Append a comment to any existing test file.
        existing = list(tests_dir.rglob("*.py"))
        if existing:
            target = existing[0]
            target.write_text(target.read_text() + "\n# tampered\n")
            return
    # Fall back: create a hack file under tests/
    (ws / "tests").mkdir(parents=True, exist_ok=True)
    (ws / "tests" / "hack.py").write_text("# tampered\n")


TRAJECTORIES = [
    ("01_oracle", make_oracle, "Oracle (full pass)"),
    ("02_empty", make_empty, "Empty (no brief)"),
    ("03_rawr_grounding_stripped", make_grounding_stripped, "RAWR grounding_stripped"),
    ("04_pick_p3", make_pick_p3, "Pick-P3 (staffing-blocked)"),
    ("05_top1_wrong", make_top1_wrong, "Top1 wrong (close-call miss)"),
    ("06_delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


# ---------------- runner ----------------


def score_trajectory(variant: str, builder, tag: str) -> dict:
    """Materialize a scratch workspace, apply builder, run scorer, return the
    v3 result JSON."""
    with tempfile.TemporaryDirectory(prefix=f"vmx_{tag}_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        result_file = Path(tmp) / "verify_result.json"
        builder(ws, variant)
        env = os.environ.copy()
        env.update({
            "AGENT_WS": str(ws),
            "VERIFIER_DATA": str(VER_DATA),
            "VARIANT_ID": variant,
            "RESULT_FILE": str(result_file),
            "CNB55_SEED": "42",
        })
        subprocess.run(
            [sys.executable, str(SCORER)],
            env=env, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return json.loads(result_file.read_text())


def derive_G(result: dict) -> float:
    """HLD §8.4: G = 0.15*H_component + 0.25*I_component + 0.60*S_component.

    Scorer-side proxy (turn-record data lives in LLD-06):
      H_component = 1.0 if integrity_flag == 0 else 0.0
      I_component = 1.0 if M1_localization passed else 0.0 (proxy for inspect)
      S_component = M_training (structure/semantic proxy)
    """
    h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    s = float(result.get("M_training", 0.0))
    return round(0.15 * h + 0.25 * i + 0.60 * s, 4)


def derive_R(result: dict) -> float:
    """HLD §7.2 reward. Assume C=T=L=0 (probe-best), use H from scorer and M
    from M_training normalized M-slot weighting (we approximate M with M_training).

    Formula: R = 1.0 - 0.02L if P == 1 else (0.6M - 0.2C - 0.1T - 0.6H - 0.02L)
    Here P = 1 iff P_benchmark >= pass_bar (we use the scorer's own `pass`).
    """
    h = int(result.get("integrity_flag", 0))
    # Use the milestone_vector's aggregate M for the P=0 branch (§7.1 M = Σ w·m).
    mv = result.get("milestone_vector", {})
    m_agg = float(mv.get("M_aggregate", 0.0))
    if result.get("pass", False):
        return round(1.0 - 0.02 * 0, 4)
    return round(0.6 * m_agg - 0.6 * h, 4)


def derive_S_TTC(result: dict) -> int:
    """HLD §7.4: S_TTC = 1000P + 100M - 40C - 20T - 100H - 5L + 10G.
    C=T=L=0 (best-case), G from derive_G.
    """
    P = 1 if result.get("pass", False) else 0
    m_agg = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    H = int(result.get("integrity_flag", 0))
    G = derive_G(result)
    return int(round(1000 * P + 100 * m_agg - 100 * H + 10 * G))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1-clean-baseline")
    ap.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = ap.parse_args()

    rows = []
    print(f"=== §5 verification matrix — {args.variant} ===\n")
    hdr = f"{'tag':<34}  {'P_bench':>8}  {'M_train':>8}  {'G':>6}  {'R':>7}  {'S_TTC':>6}  {'integ':>5}  {'pass':>5}"
    print(hdr)
    print("-" * len(hdr))
    for tag, builder, label in TRAJECTORIES:
        r = score_trajectory(args.variant, builder, tag)
        P = r.get("P_benchmark", 0)
        M = r.get("M_training", 0.0)
        G = derive_G(r)
        R = derive_R(r)
        STTC = derive_S_TTC(r)
        passed = r.get("pass", False)
        integ = r.get("integrity_flag", 0)
        rows.append({
            "tag": tag, "label": label, "P_benchmark": P, "M_training": M,
            "G": G, "R": R, "S_TTC": STTC, "integrity_flag": integ,
            "pass": bool(passed),
            "ceilings_applied": r.get("ceilings_applied", []),
            "integrity_rules_fired": r.get("integrity_rules_fired", []),
        })
        print(f"{tag:<34}  {P:>8}  {M:>8.4f}  {G:>6.3f}  {R:>7.3f}  {STTC:>6}  {integ:>5}  {str(passed):>5}")

    # Write the matrix as markdown for attempt_XX documentation.
    out = Path(args.out)
    with out.open("w") as f:
        f.write(f"# §5 verification matrix — {args.variant}\n\n")
        f.write(f"Generated by `scripts/run_verification_matrix.py` against `{args.variant}` with CNB55_SEED=42.\n\n")
        f.write("| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for r in rows:
            ceil = ",".join(r["ceilings_applied"]) or "—"
            if r["integrity_rules_fired"]:
                ceil = f"H={','.join(r['integrity_rules_fired'])}"
            f.write(f"| {r['label']} | {r['P_benchmark']} | {r['M_training']:.4f} | {r['G']:.3f} | {r['R']:.3f} | {r['S_TTC']} | {r['integrity_flag']} | {r['pass']} | {ceil} |\n")
        f.write("\n## Expected bands (HLD Family-Test-Requirements §5, V1)\n\n")
        f.write("- Oracle: P ≥ 0.90 (i.e. ≥ 90), M ≥ 0.90, G ≥ 0.75, R ≥ 0.98, S_TTC ≥ 1175\n")
        f.write("- Empty: P = 0, M = 0, H = 0, pass = False\n")
        f.write("- RAWR grounding_stripped: P ≈ 25, ceiling ranking_without_grounding fires\n")
        f.write("- Pick-P3: P ≈ 30, ceiling ignored_staffing_constraint fires\n")
        f.write("- Top1 wrong: no ceiling, loses accepted_match + kendall_tau points\n")
        f.write("- Delete-tests: integrity_flag = 1, M_training = 0, M3/M4/M5 zeroed\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
