#!/usr/bin/env python3
"""Run the per-task deterministic grader on every cell that lacks grader_result.json.

For each (point, task, attempt) cell:
  - AGENT_WS = <run_dir>/workspace/
  - RESULT_FILE = <run_dir>/grader_result.json
  - VARIANT_ID = "v1-clean-baseline"
  - Invoke verifiers/<task>/verify.sh
  - Tolerate errors (record in summary, continue)

Prints a running progress bar and final per-point pass-rate summary.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/sessions/charming-hopeful-johnson/mnt/Lumo_FlyWheel")
ROOT_V4A_V2 = REPO / "output" / "track_b_e2e_v4a_v2"
ROOT_ABLATION = REPO / "output" / "track_b_e2e_v4a_v2_ablation"
VERIFIERS = REPO / "verifiers"
SUMMARY = REPO / "output" / "grade_all_cells_summary.json"

POINTS = {
    "D": [
        ROOT_V4A_V2 / "round_0",
        ROOT_V4A_V2 / "round_0_phase1_task1_2_PRESERVED",
        ROOT_V4A_V2 / "round_0_phase2_task3_4_PRESERVED",
        ROOT_V4A_V2 / "round_0_phase3a_PRESERVED",
    ],
    "A": [ROOT_ABLATION / "round_1"],
    "B": [ROOT_ABLATION / "round_2"],
    "C": [ROOT_ABLATION / "round_3"],
    "OFF": [ROOT_ABLATION / "round_4"],
}


SCORER_MAP = {
    "responses-sdk-adapter-cutover": "score_responses_cutover.py",
    "release-note-to-plan-translation": "score_ranking.py",
    "policy-aware-request-resolution": "score_ranking.py",
    "responsive-checkout-visual-regression": "score_visual_regression.py",
    "dead-flag-reachability-audit": "score_reachability.py",
    "incident-evidence-synthesis": "score_incident_packet.py",
    "multi-tool-transaction-repair": "score_transaction_repair.py",
    "fanout-fullstack-release-blocker": "score_release_blocker.py",
    "transcript-merge-regression": "score_transcript_merge.py",
    "security-audit-hotfix-remediation": "score_hotfix.py",
    "sqlalchemy-2-session-modernization": "score_sqlalchemy_session_modernization.py",
}


def grade_cell(run_dir: Path, task_short: str) -> dict:
    """Run verifier on a single cell; return result summary."""
    workspace = run_dir / "workspace"
    result_file = run_dir / "grader_result.json"
    scorer_name = SCORER_MAP.get(task_short)
    if scorer_name is None:
        return {"status": "no_scorer_mapping", "task": task_short}
    scorer_path = VERIFIERS / task_short / scorer_name
    if not scorer_path.is_file():
        return {"status": "scorer_missing", "scorer_path": str(scorer_path)}
    cmd = ["python3", str(scorer_path)]
    if not workspace.is_dir():
        return {"status": "no_workspace", "workspace_path": str(workspace)}
    # Skip if already graded (idempotent)
    if result_file.is_file():
        try:
            r = json.loads(result_file.read_text())
            return {
                "status": "already_graded",
                "P_benchmark": r.get("P_benchmark"),
                "M_aggregate": r.get("M_aggregate") or r.get("milestone_vector", {}).get("M_aggregate"),
            }
        except Exception:
            pass
    env = os.environ.copy()
    env["AGENT_WS"] = str(workspace)
    env["RESULT_FILE"] = str(result_file)
    env["VARIANT_ID"] = "v1-clean-baseline"
    env["CNB55_SEED"] = "42"
    env["PATH"] = f"{os.path.expanduser('~/.local/bin')}:{env.get('PATH', '')}"
    # Bundle root for hash comparison (per-family)
    env.setdefault("WORKSPACE_BUNDLE_ROOT", str(REPO / "benchmark_blueprints" / "tracks"))
    # Verifier data root (gold references per family)
    env["VERIFIER_DATA"] = str(REPO / "verifier_data" / task_short)
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO),
        )
        rc = proc.returncode
        # The verifier writes the json to RESULT_FILE; read it back
        if result_file.is_file():
            try:
                r = json.loads(result_file.read_text())
                return {
                    "status": "graded",
                    "rc": rc,
                    "P_benchmark": r.get("P_benchmark"),
                    "M_aggregate": r.get("M_aggregate") or r.get("milestone_vector", {}).get("M_aggregate"),
                    "integrity_flag": r.get("integrity_flag"),
                    "ceilings_applied": r.get("ceilings_applied", []),
                }
            except Exception as e:
                return {"status": "result_unreadable", "error": str(e), "rc": rc}
        else:
            return {
                "status": "no_result_written",
                "rc": rc,
                "stdout_tail": proc.stdout[-300:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-300:] if proc.stderr else "",
            }
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "exec_error", "error": str(e)}


def main():
    # Build the list of cells to grade
    cells_to_grade = []
    for point, roots in POINTS.items():
        for parent in roots:
            if not parent.is_dir(): continue
            for task_dir in parent.iterdir():
                if not task_dir.is_dir() or "__" not in task_dir.name: continue
                task_short = task_dir.name.split("__")[0]
                for run_dir in sorted(task_dir.iterdir()):
                    if not run_dir.is_dir() or not run_dir.name.startswith("run_"): continue
                    if (run_dir / "grader_result.json").is_file():
                        continue  # already graded
                    if point == "D" and "fanout" in task_dir.name and parent.name != "round_0":
                        continue  # dedup
                    cells_to_grade.append({
                        "point": point,
                        "task_short": task_short,
                        "run_dir": str(run_dir),
                        "attempt": run_dir.name,
                    })

    print(f"Cells to grade: {len(cells_to_grade)}")
    print()

    results = []
    t0 = time.time()
    for i, cell in enumerate(cells_to_grade, 1):
        run_dir = Path(cell["run_dir"])
        r = grade_cell(run_dir, cell["task_short"])
        cell.update(r)
        results.append(cell)
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(cells_to_grade) - i) / rate if rate > 0 else 0
        status = r["status"]
        p = r.get("P_benchmark")
        passed = "PASS" if (p is not None and p >= 65) else ("FAIL" if p is not None else status)
        print(f"  [{i:3d}/{len(cells_to_grade)}] {cell['point']:<3} {cell['task_short'][:36]:<36} {cell['attempt']:<7} -> {passed} (P={p}) [{rate:.1f} cells/s, ETA {eta/60:.1f}min]")

    # Summary
    from collections import Counter
    status_counts = Counter(r["status"] for r in results)
    print()
    print("="*70)
    print(f"Status counts: {dict(status_counts)}")

    # Per-point pass rate
    by_point = {}
    for r in results:
        pt = r["point"]
        p = r.get("P_benchmark")
        if p is None: continue
        by_point.setdefault(pt, []).append(p)
    print()
    print(f"{'Point':<5} {'n':>4} {'n_pass':>7} {'pass_rate':>10} {'median_P':>10}")
    for pt in ["OFF", "A", "B", "C", "D"]:
        if pt not in by_point: continue
        ps = by_point[pt]
        n_pass = sum(1 for p in ps if p >= 65)
        import statistics
        print(f"{pt:<5} {len(ps):>4} {n_pass:>7} {n_pass/len(ps)*100:>9.1f}% {statistics.median(ps):>10.1f}")

    SUMMARY.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote: {SUMMARY}")


if __name__ == "__main__":
    main()
