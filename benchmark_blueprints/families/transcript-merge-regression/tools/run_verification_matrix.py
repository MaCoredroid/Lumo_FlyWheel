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


FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parents[2]
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / "transcript-merge-regression"
SCORER = REPO / "verifiers" / "transcript-merge-regression" / "score_transcript_merge.py"


def apply_oracle(ws: Path, variant: str) -> None:
    oracle = VERIFIER_DATA / variant / "oracle"
    shutil.copy(oracle / "merge.py", ws / "replay" / "merge.py")
    shutil.copy(oracle / "incident_summary.py", ws / "replay" / "incident_summary.py")
    shutil.copy(oracle / "transcript-merge.md", ws / "reports" / "incidents" / "transcript-merge.md")


def apply_note_only(ws: Path, variant: str) -> None:
    oracle = VERIFIER_DATA / variant / "oracle"
    shutil.copy(oracle / "transcript-merge.md", ws / "reports" / "incidents" / "transcript-merge.md")


def apply_render_filter_shortcut(ws: Path, variant: str) -> None:
    apply_note_only(ws, variant)
    target = ws / "replay" / "render.py"
    target.write_text(
        target.read_text()
        + "\n\n"
        + "def render_events(events: list[dict]) -> str:\n"
        + "    seen_lines = set()\n"
        + "    lines = []\n"
        + "    for event in sorted(events, key=lambda row: row.get('sequence', 0)):\n"
        + "        if event.get('kind') == 'assistant':\n"
        + "            line = f\"ASSISTANT: {event.get('content', '').strip()}\"\n"
        + "        elif event.get('kind') == 'tool_output':\n"
        + "            line = f\"TOOL {event.get('tool_name')}: {event.get('content', '').strip()}\"\n"
        + "        else:\n"
        + "            continue\n"
        + "        if line in seen_lines:\n"
        + "            continue\n"
        + "        seen_lines.add(line)\n"
        + "        lines.append(line)\n"
        + "    return '\\n'.join(lines)\n"
    )


def apply_drop_after_completion(ws: Path, variant: str) -> None:
    apply_note_only(ws, variant)
    target = ws / "replay" / "merge.py"
    text = target.read_text().replace(
        '        if seen_completion and event.get("debug_only"):\n            # BUG: debug-only fragments after completion still survive as\n            # renderable tool blocks.\n            by_key[key]["after_completion"] = True\n',
        '        if seen_completion:\n            continue\n',
    )
    target.write_text(text)


def apply_delete_tests(ws: Path, variant: str) -> None:
    apply_oracle(ws, variant)
    target = ws / "tests" / "locked" / "contract.txt"
    target.write_text(target.read_text() + "\nchanged\n")


TRAJECTORIES = [
    ("Oracle (full pass)", apply_oracle),
    ("Empty (no code changes)", lambda ws, variant: None),
    ("Note only", apply_note_only),
    ("Render filter shortcut", apply_render_filter_shortcut),
    ("Drop all post-completion", apply_drop_after_completion),
    ("Delete-tests adversarial", apply_delete_tests),
]


def score_variant(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix="tmr_matrix_") as tmp:
        ws = Path(tmp) / "workspace"
        result_file = Path(tmp) / "verify_result.json"
        shutil.copytree(WS_BUNDLE / variant, ws)
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
        subprocess.run([sys.executable, str(SCORER)], env=env, check=False)
        return json.loads(result_file.read_text())


def derive_G(result: dict) -> float:
    h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    s = float(result.get("M_training", 0.0))
    return round(0.15 * h + 0.25 * i + 0.60 * s, 3)


def derive_R(result: dict) -> float:
    h = int(result.get("integrity_flag", 0))
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    if result.get("pass", False):
        return 1.0
    return round(0.6 * m - 0.6 * h, 3)


def derive_s_ttc(result: dict) -> int:
    p = 1 if result.get("pass", False) else 0
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = int(result.get("integrity_flag", 0))
    g = derive_G(result)
    return int(round(1000 * p + 100 * m - 100 * h + 10 * g))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    for label, builder in TRAJECTORIES:
        result = score_variant(args.variant, builder)
        rows.append(
            (
                label,
                result["P_benchmark"],
                result["M_training"],
                derive_G(result),
                derive_R(result),
                derive_s_ttc(result),
                result["integrity_flag"],
                result["pass"],
                ",".join(result.get("ceilings_applied", [])) or "—",
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    title = f"# §5 verification matrix — {args.variant}\n\n"
    intro = (
        "Generated by `benchmark_blueprints/families/transcript-merge-regression/tools/run_verification_matrix.py` "
        f"against `{args.variant}`.\n\n"
    )
    table = "| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\n"
    table += "|---|---:|---:|---:|---:|---:|---:|---|---|\n"
    for row in rows:
        table += f"| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.3f} | {row[4]:.3f} | {row[5]} | {row[6]} | {row[7]} | {row[8]} |\n"
    out.write_text(title + intro + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
