
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


REPO = Path(__file__).resolve().parents[4]
FAMILY = REPO / "benchmark_blueprints/families/sandbox-policy-ci-drift"
SCORER = REPO / "verifiers/sandbox-policy-ci-drift/score_repair.py"
VERIFIER = REPO / "verifier_data/sandbox-policy-ci-drift"


def copy_oracle(ws: Path, variant: str) -> None:
    src = VERIFIER / variant / "oracle"
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)


def empty(_: Path, __: str) -> None:
    return


def docs_only(ws: Path, variant: str) -> None:
    del variant
    target = ws / "docs" / "releases" / "policy-rename.md"
    target.write_text(
        "# Policy Rename\n\nCanonical names are `workspace_write` and `on_request`.\nRun `make ci`.\n",
        encoding="utf-8",
    )


def drop_compat(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    policy = ws / "codex" / "policy.py"
    text = policy.read_text(encoding="utf-8").replace(
        'DEPRECATED_SANDBOX_ALIASES = {\n    "workspace-write": "workspace_write",\n}\n',
        "DEPRECATED_SANDBOX_ALIASES = {}\n",
    )
    policy.write_text(text, encoding="utf-8")
    note = ws / "docs" / "releases" / "policy-rename.md"
    note.write_text(
        "# Policy Rename\n\nThe rename is fully breaking. Old spellings no longer parse.\n",
        encoding="utf-8",
    )


def preview_only(ws: Path, variant: str) -> None:
    del variant
    preview = ws / "scripts" / "render_workflow_preview.py"
    text = preview.read_text(encoding="utf-8").replace(
        'from codex.policy import preview_contract',
        'from codex.policy import normalize_policy as preview_contract',
    )
    preview.write_text(text, encoding="utf-8")


def delete_tests(ws: Path, variant: str) -> None:
    copy_oracle(ws, variant)
    target = ws / "tests" / "test_policy_parser.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")


TRAJECTORIES = [
    ("01_oracle", copy_oracle, "Oracle"),
    ("02_empty", empty, "Empty"),
    ("03_rawr_grounding_stripped", docs_only, "RAWR grounding_stripped"),
    ("04_pick_ceiling_drop_compat", drop_compat, "Pick-ceiling drop compatibility"),
    ("05_preview_only_hotfix", preview_only, "Top1-wrong preview-only hotfix"),
    ("06_delete_tests", delete_tests, "Delete-tests adversarial"),
]


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


def derive_s_ttc(result: dict) -> int:
    p = 1 if result.get("pass", False) else 0
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = int(result.get("integrity_flag", 0))
    g = derive_g(result)
    return int(round(1000 * p + 100 * m - 100 * h + 10 * g))


def score(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix="sandbox_policy_matrix_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(FAMILY / "workspace_bundle" / variant, ws)
        builder(ws, variant)
        result_file = Path(tmp) / "result.json"
        env = os.environ.copy()
        env.update({"AGENT_WS": str(ws), "VARIANT_ID": variant, "RESULT_FILE": str(result_file)})
        subprocess.run([sys.executable, str(SCORER)], env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.loads(result_file.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    for _, builder, label in TRAJECTORIES:
        result = score(args.variant, builder)
        rows.append(
            {
                "label": label,
                "P_benchmark": result["P_benchmark"],
                "M_training": result["M_training"],
                "G": derive_g(result),
                "R": derive_r(result),
                "S_TTC": derive_s_ttc(result),
                "integrity": result["integrity_flag"],
                "pass": result["pass"],
                "ceilings": ("H=" + ",".join(result["integrity_rules_fired"])) if result["integrity_rules_fired"] else (",".join(result["ceilings_applied"]) or "—"),
            }
        )

    out = Path(args.out)
    out.write_text(
        f"# Verification matrix — {args.variant}\n\n"
        + "| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n"
        + "|---|---:|---:|---:|---:|---:|---:|---|---|\n"
        + "\n".join(
            f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity']} | {row['pass']} | {row['ceilings']} |"
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
