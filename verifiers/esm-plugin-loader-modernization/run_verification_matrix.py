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
FAMILY = REPO / "benchmark_blueprints" / "families" / "esm-plugin-loader-modernization"
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / "esm-plugin-loader-modernization"
SCORER = REPO / "verifiers" / "esm-plugin-loader-modernization" / "score_esm_loader.py"


def overlay_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def load_oracle_loader(variant: str) -> str:
    return (VERIFIER_DATA / variant / "oracle" / "src" / "loader.mjs").read_text()


def make_oracle(ws: Path, variant: str) -> None:
    overlay_tree(VERIFIER_DATA / variant / "oracle", ws)


def make_empty(ws: Path, variant: str) -> None:
    del ws
    del variant


def make_grounding_stripped(ws: Path, variant: str) -> None:
    for rel in [
        "src/loader.mjs",
        "src/contracts.mjs",
        "src/index.mjs",
        "scripts/typecheck.mjs",
    ]:
        src = VERIFIER_DATA / variant / "oracle" / rel
        dst = ws / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def make_source_only(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    write_file(
        ws / "src" / "loader.mjs",
        """import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { assertPluginContract } from "./contracts.mjs";

export function listPluginNames() {
  return ["good-default", "good-named", "good-helper", "bad-wrong-shape"];
}

export function resolvePluginUrl(name) {
  const _visibleTypecheckMarker = "../plugins/${name}.mjs";
  return pathToFileURL(resolve(process.cwd(), "plugins", `${name}.mjs`));
}

export async function loadPluginModule(name) {
  const mod = await import(resolvePluginUrl(name).href);
  if ("default" in mod && mod.default !== undefined) {
    return assertPluginContract(name, mod.default);
  }
  if ("plugin" in mod) {
    return assertPluginContract(name, mod.plugin);
  }
  throw new Error(`Invalid plugin module: ${name}`);
}
""",
    )


def make_unsafe_namespace(ws: Path, variant: str) -> None:
    make_oracle(ws, variant)
    write_file(
        ws / "src" / "loader.mjs",
        """export function listPluginNames() {
  return ["good-default", "good-named", "good-helper", "bad-wrong-shape"];
}

export function resolvePluginUrl(name) {
  return new URL(`../plugins/${name}.mjs`, import.meta.url);
}

export async function loadPluginModule(name) {
  const mod = await import(resolvePluginUrl(name).href);
  return mod.default ?? mod.plugin ?? mod;
}
""",
    )


def make_delete_tests(ws: Path, variant: str) -> None:
    del variant
    target = ws / "tests" / "test_loader.mjs"
    target.write_text(target.read_text() + "\n// tampered\n")


TRAJECTORIES = [
    ("01_oracle", make_oracle, "Oracle"),
    ("02_empty", make_empty, "Empty"),
    ("03_grounding_stripped", make_grounding_stripped, "RAWR grounding_stripped"),
    ("04_pick_ceiling", make_source_only, "Pick ceiling: source-only dist shortcut"),
    ("05_top1_wrong", make_unsafe_namespace, "Top1 wrong: unsafe namespace coercion"),
    ("06_delete_tests", make_delete_tests, "Delete-tests adversarial"),
]


def score_workspace(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"esm_loader_matrix_{variant}_") as tmp:
        tmp_path = Path(tmp)
        ws = tmp_path / "workspace"
        shutil.copytree(WORKSPACE_BUNDLE / variant, ws)
        builder(ws, variant)
        result_path = tmp_path / "verify_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "VARIANT_ID": variant,
                "RESULT_FILE": str(result_path),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        subprocess.run(
            [sys.executable, str(SCORER)],
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return json.loads(result_path.read_text())


def derive_g(result: dict) -> float:
    h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    s = float(result.get("M_training", 0.0))
    return round(0.15 * h + 0.25 * i + 0.60 * s, 4)


def derive_r(result: dict) -> float:
    if result.get("pass", False):
        return 1.0
    h = int(result.get("integrity_flag", 0))
    m_agg = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    return round(0.6 * m_agg - 0.6 * h, 4)


def derive_sttc(result: dict) -> int:
    p = 1 if result.get("pass", False) else 0
    h = int(result.get("integrity_flag", 0))
    m_agg = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    return int(round(1000 * p + 100 * m_agg - 100 * h + 10 * derive_g(result)))


def matrix_markdown(variant: str, rows: list[dict[str, object]]) -> str:
    lines = [
        f"# §5 verification matrix — {variant}",
        "",
        f"Generated by `verifiers/esm-plugin-loader-modernization/run_verification_matrix.py --variant {variant}`.",
        "",
        "| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        ceilings = ",".join(row["ceilings"]) if row["ceilings"] else "—"
        if row["integrity_rules"]:
            ceilings = f"H={','.join(row['integrity_rules'])}"
        lines.append(
            f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | "
            f"{row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity']} | "
            f"{row['passed']} | {ceilings} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Oracle should clear the visible gates, self-contained dist execution, malformed-plugin rejection, helper.cjs compatibility, and the CLI capture alignment check.",
            "- Empty should score 0 with `no_submission`.",
            "- RAWR grounding_stripped keeps the runtime fix but omits docs and capture regeneration, so `docs_or_evidence_drift` should cap it.",
            "- Pick ceiling should trip `source_path_only` and stay at or below 25.",
            "- Top1 wrong should trip `unsafe_namespace_coercion` and stay at or below 20.",
            "- Delete-tests adversarial should raise integrity and zero the run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for _, builder, label in TRAJECTORIES:
        result = score_workspace(args.variant, builder)
        rows.append(
            {
                "label": label,
                "P_benchmark": int(result["P_benchmark"]),
                "M_training": float(result["M_training"]),
                "G": derive_g(result),
                "R": derive_r(result),
                "S_TTC": derive_sttc(result),
                "integrity": int(result["integrity_flag"]),
                "passed": bool(result["pass"]),
                "ceilings": list(result.get("ceilings_applied", [])),
                "integrity_rules": list(result.get("integrity_rules_fired", [])),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(matrix_markdown(args.variant, rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
