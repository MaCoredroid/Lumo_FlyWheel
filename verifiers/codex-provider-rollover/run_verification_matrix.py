#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILY_ROOT = REPO_ROOT / "benchmark_blueprints" / "families" / "codex-provider-rollover"
WORKSPACE_ROOT = FAMILY_ROOT / "workspace_bundle"
VERIFIER_ROOT = REPO_ROOT / "verifier_data" / "codex-provider-rollover"
SCORER = REPO_ROOT / "verifiers" / "codex-provider-rollover" / "score_provider_rollover.py"


def run_scorer(workspace: Path, variant_id: str) -> dict:
    result_file = workspace / "_verify_result.json"
    env = os.environ | {
        "AGENT_WS": str(workspace),
        "VERIFIER_DATA": str(VERIFIER_ROOT),
        "RESULT_FILE": str(result_file),
        "VARIANT_ID": variant_id,
    }
    subprocess.run([os.environ.get("PYTHON", "python3"), str(SCORER)], check=True, env=env, cwd=REPO_ROOT)
    return json.loads(result_file.read_text())


def copy_oracle(workspace: Path, variant_id: str) -> None:
    oracle = VERIFIER_ROOT / variant_id / "oracle"
    for src in oracle.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(oracle)
        dst = workspace / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def mutate_rawr(workspace: Path, variant_id: str) -> None:
    copy_oracle(workspace, variant_id)
    provider_doc = workspace / "serving_maintenance" / "docs" / "provider_rollover.md"
    smoke_doc = workspace / "serving_maintenance" / "docs" / "smoke.md"
    provider_doc.write_text(
        "# Provider rollover\n\n"
        "Use the maintained provider, but keep this note generic.\n"
        "Do not preserve the maintenance-profile-specific grounding here.\n"
    )
    smoke_doc.write_text(
        "# Smoke procedure\n\n"
        "Run a second turn after the first one succeeds.\n"
    )


def mutate_pick_ceiling(workspace: Path, variant_id: str) -> None:
    copy_oracle(workspace, variant_id)
    config = workspace / "serving_maintenance" / ".codex" / "config.toml"
    config.write_text(config.read_text().replace('provider = "responses_proxy"', 'provider = "legacy_vllm"', 1))


def mutate_top1_wrong(workspace: Path, variant_id: str) -> None:
    copy_oracle(workspace, variant_id)
    config = workspace / "serving_maintenance" / ".codex" / "config.toml"
    config.write_text(
        "# generated from template\n"
        "provider = \"responses_proxy\"\n"
        "model = \"gpt-5.4\"\n"
        "\n"
        "[model_providers.responses_proxy]\n"
        "base_url = \"http://127.0.0.1:11434/v1/responses\"\n"
        "env_key = \"OPENAI_API_KEY\"\n"
        "wire_api = \"responses\"\n"
        "store = true\n"
    )


def mutate_delete_tests(workspace: Path, variant_id: str) -> None:
    copy_oracle(workspace, variant_id)
    test_file = workspace / "tests" / "test_docs_sync.py"
    test_file.write_text("print('deleted')\n")


def make_workspace_copy(variant_id: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{variant_id}-matrix-"))
    shutil.copytree(WORKSPACE_ROOT / variant_id, temp_dir / "workspace")
    return temp_dir / "workspace"


def compute_proxies(result: dict) -> tuple[float, float, int]:
    g = round(0.25 + result["M_training"] * 0.75, 3)
    r = round(result["P_benchmark"] / 100.0, 3)
    s_ttc = int(1200 * result["M_training"])
    return g, r, s_ttc


def build_rows(variant_id: str) -> list[dict[str, object]]:
    rows = []
    scenarios = [
        ("Oracle (full pass)", lambda ws: copy_oracle(ws, variant_id)),
        ("Empty (untouched bundle)", lambda ws: None),
        ("RAWR grounding_stripped", lambda ws: mutate_rawr(ws, variant_id)),
        ("Pick-ceiling", lambda ws: mutate_pick_ceiling(ws, variant_id)),
        ("Top1-wrong", lambda ws: mutate_top1_wrong(ws, variant_id)),
        ("Delete-tests adversarial", lambda ws: mutate_delete_tests(ws, variant_id)),
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
    return rows


def render_matrix(variant_id: str, rows: list[dict[str, object]]) -> str:
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
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out")
    args = parser.parse_args()

    rows = build_rows(args.variant)
    output = render_matrix(args.variant, rows)
    if args.out:
        out_path = Path(args.out)
    else:
        filename = "verification_matrix.md" if args.variant == "v1-clean-baseline" else f"verification_matrix_{args.variant}.md"
        out_path = FAMILY_ROOT / filename
    out_path.write_text(output)


if __name__ == "__main__":
    main()
