#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY = REPO / "benchmark_blueprints" / "families" / "security-audit-hotfix-remediation"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / "security-audit-hotfix-remediation"
SCORER = REPO / "verifiers" / "security-audit-hotfix-remediation" / "score_hotfix.py"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_oracle(ws: Path, variant: str) -> None:
    oracle = VERIFIER_DATA / variant / "oracle"
    shutil.copy(oracle / "storage.py", ws / "repo" / "export_api" / "storage.py")
    shutil.copy(oracle / "triage_note.json", ws / "audit" / "triage_note.json")
    shutil.copy(oracle / "deploy_hotfix.json", ws / "repo" / "ops" / "deploy_hotfix.json")
    shutil.copy(oracle / "test_hotfix_regressions.py", ws / "repo" / "tests" / "test_hotfix_regressions.py")


def make_empty(ws: Path, variant: str) -> None:
    return


def make_grounding_stripped(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    triage_path = ws / "audit" / "triage_note.json"
    triage = load_json(triage_path)
    for item in triage["items"]:
        item["evidence_paths"] = ["audit/triage_matrix.csv", "audit/triage_matrix.csv"]
    triage_path.write_text(json.dumps(triage, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_pick_ceiling(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    naive = """from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = unquote(requested_path).replace('\\\\', '/')
    if '..' in normalized or normalized.startswith('/'):
        raise ExportPathViolation('blocked suspicious export path')
    candidate = (tenant_root / normalized.lstrip('/')).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation('candidate escaped the tenant root')
    return candidate
"""
    (ws / "repo" / "export_api" / "storage.py").write_text(naive, encoding="utf-8")


def make_top1_wrong(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    too_strict = """from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = unquote(requested_path).replace('\\\\', '/')
    if '/' in normalized or '\\\\' in normalized:
        raise ExportPathViolation('nested exports disabled')
    candidate = (tenant_root / normalized).resolve(strict=False)
    if candidate.parent != tenant_root.resolve():
        raise ExportPathViolation('candidate escaped the tenant root')
    return candidate
"""
    (ws / "repo" / "export_api" / "storage.py").write_text(too_strict, encoding="utf-8")


def make_delete_tests(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    target = ws / "repo" / "tests" / "test_routes.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")


TRAJECTORIES = [
    ("01_oracle", copy_oracle, "Oracle"),
    ("02_empty", make_empty, "Empty"),
    ("03_grounding_stripped", make_grounding_stripped, "RAWR grounding_stripped"),
    ("04_pick_ceiling", make_pick_ceiling, "Pick-ceiling blocklist fix"),
    ("05_top1_wrong", make_top1_wrong, "Top1-wrong nested exports disabled"),
    ("06_delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


def score_trajectory(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix="hotfix-matrix-") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        result_file = Path(tmp) / "verify_result.json"
        builder(ws, variant)
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "VARIANT_ID": variant,
                "RESULT_FILE": str(result_file),
            }
        )
        subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
        return load_json(result_file)


def derive_g(result: dict) -> float:
    h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    s = float(result.get("M_training", 0.0))
    return round(0.15 * h + 0.25 * i + 0.60 * s, 4)


def derive_r(result: dict) -> float:
    h = int(result.get("integrity_flag", 0))
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    if result.get("pass", False):
        return 1.0
    return round(0.6 * m - 0.6 * h, 4)


def derive_sttc(result: dict) -> int:
    p = 1 if result.get("pass", False) else 0
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = int(result.get("integrity_flag", 0))
    g = derive_g(result)
    return int(round(1000 * p + 100 * m - 100 * h + 10 * g))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    for _, builder, label in TRAJECTORIES:
        result = score_trajectory(args.variant, builder)
        rows.append(
            {
                "label": label,
                "P_benchmark": result["P_benchmark"],
                "M_training": result["M_training"],
                "G": derive_g(result),
                "R": derive_r(result),
                "S_TTC": derive_sttc(result),
                "integrity": result["integrity_flag"],
                "pass": result["pass"],
                "ceilings": result["integrity_rules_fired"] or result["ceilings_applied"],
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write(f"# §5 verification matrix — {args.variant}\n\n")
        handle.write("| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            ceilings = ",".join(row["ceilings"]) if row["ceilings"] else "—"
            handle.write(
                f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | "
                f"{row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity']} | "
                f"{row['pass']} | {ceilings} |\n"
            )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
