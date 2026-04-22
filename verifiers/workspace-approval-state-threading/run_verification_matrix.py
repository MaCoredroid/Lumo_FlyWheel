#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "benchmark_blueprints" / "families" / "workspace-approval-state-threading" / "workspace_bundle"
VERIFIER_DATA = REPO_ROOT / "verifier_data" / "workspace-approval-state-threading"
SCORER = REPO_ROOT / "verifiers" / "workspace-approval-state-threading" / "score_workspace_approval.py"


def run_scorer(workspace: Path, variant_id: str) -> dict:
    result_file = workspace / "_verify_result.json"
    env = os.environ | {
        "AGENT_WS": str(workspace),
        "VERIFIER_DATA": str(VERIFIER_DATA),
        "RESULT_FILE": str(result_file),
        "VARIANT_ID": variant_id,
    }
    subprocess.run([os.environ.get("PYTHON", "python3"), str(SCORER)], check=True, env=env, cwd=REPO_ROOT)
    return json.loads(result_file.read_text())


def replace_with_oracle(workspace: Path, variant_id: str) -> None:
    oracle = VERIFIER_DATA / variant_id / "oracle"
    for src in oracle.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(oracle)
        dst = workspace / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def mutate_rawr(workspace: Path) -> None:
    preview = workspace / "artifacts" / "preview" / "workspace_admin_capture.json"
    data = json.loads(preview.read_text())
    data["columns"] = ["workspace", "risk_level"]
    data["filtered_row"]["approval_state"] = "MISSING"
    preview.write_text(json.dumps(data, indent=2) + "\n")
    note = workspace / "artifacts" / "rollout" / "approval_state_rollout_note.json"
    if note.exists():
        note.unlink()


def mutate_alias_shortcut(workspace: Path) -> None:
    serializer = workspace / "backend" / "api" / "serializers.py"
    serializer.write_text(
        "from __future__ import annotations\n\n\n"
        "def serialize_workspace(row: dict) -> dict:\n"
        "    return {\n"
        "        'workspace_id': row['workspace_id'],\n"
        "        'workspace_name': row['workspace_name'],\n"
        "        'risk_level': row['risk_level'],\n"
        "        'approval_state': row['risk_level'],\n"
        "    }\n"
    )
    note = workspace / "artifacts" / "rollout" / "approval_state_rollout_note.json"
    if note.exists():
        note.write_text(
            json.dumps(
                {
                    "schema_version": "cnb55.rollout_note.v1",
                    "legacy_row_fallback": "placeholder",
                    "consistency_surfaces": ["frontend"],
                    "screenshot_name": "workspace-admin-approval-state.png",
                    "variant_notes": "shortcut",
                },
                indent=2,
            )
            + "\n"
        )


def mutate_frontend_only(workspace: Path) -> None:
    table = workspace / "frontend" / "src" / "components" / "WorkspaceTable.tsx"
    table.write_text(
        "export const TABLE_COLUMNS = ['workspace', 'risk_level', 'approval_state'];\n"
        "export const APPROVAL_STATE_COLUMN_LABEL = 'Approval state';\n"
        "export function renderApprovalStateBadge(row: { risk_level: string }) {\n"
        "  return row.risk_level;\n"
        "}\n"
    )
    preview = workspace / "artifacts" / "preview" / "workspace_admin_capture.json"
    preview.write_text(
        json.dumps(
            {
                "screenshot_name": "workspace-admin-approval-state.png",
                "columns": ["workspace", "risk_level", "approval_state"],
                "filtered_row": {"workspace_id": "ws-legacy-03", "approval_state": "manual_review", "source": "legacy_fallback"},
                "badge_values": ["approved", "blocked", "manual_review"],
            },
            indent=2,
        )
        + "\n"
    )


def mutate_delete_tests(workspace: Path) -> None:
    (workspace / "seed_data" / "mixed_workspaces.json").write_text("[]\n")


def compute_proxies(result: dict) -> tuple[float, float, int]:
    g = round(0.25 + result["M_training"] * 0.75, 3)
    r = round(result["P_benchmark"] / 100.0, 3)
    s_ttc = int(1200 * result["M_training"])
    return g, r, s_ttc


def make_workspace_copy(variant_id: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{variant_id}-matrix-"))
    shutil.copytree(WORKSPACE_ROOT / variant_id, temp_dir / "workspace")
    return temp_dir / "workspace"


def main() -> None:
    variant_id = os.environ.get("VARIANT_ID", "v1-clean-baseline")
    rows = []
    scenarios = [
        ("Oracle (full pass)", lambda ws: replace_with_oracle(ws, variant_id)),
        ("Empty (untouched bundle)", lambda ws: None),
        ("RAWR grounding_stripped", lambda ws: (replace_with_oracle(ws, variant_id), mutate_rawr(ws))),
        ("Alias shortcut", lambda ws: (replace_with_oracle(ws, variant_id), mutate_alias_shortcut(ws))),
        ("Frontend-only fabrication", lambda ws: mutate_frontend_only(ws)),
        ("Delete-tests adversarial", lambda ws: (replace_with_oracle(ws, variant_id), mutate_delete_tests(ws))),
    ]

    for label, mutator in scenarios:
        workspace = make_workspace_copy(variant_id)
        mutator(workspace)
        result = run_scorer(workspace, variant_id)
        g, r, s_ttc = compute_proxies(result)
        rows.append(
            {
                "trajectory": label,
                "P_benchmark": result["P_benchmark"],
                "M_training": result["M_training"],
                "G": g,
                "R": r,
                "S_TTC": s_ttc,
                "integrity": result["integrity_flag"],
                "pass": result["pass"],
                "ceilings": ", ".join(result["ceilings_applied"]) or "—",
            }
        )
        shutil.rmtree(workspace.parent)

    lines = [
        f"# verification_matrix - {variant_id}",
        "",
        "| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['trajectory']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | "
            f"{row['R']:.3f} | {row['S_TTC']} | {row['integrity']} | {row['pass']} | {row['ceilings']} |"
        )
    output_name = "verification_matrix.md" if variant_id == "v1-clean-baseline" else f"verification_matrix_{variant_id}.md"
    (REPO_ROOT / "benchmark_blueprints" / "families" / "workspace-approval-state-threading" / output_name).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
