#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY_ID = "objective-driven-repo-improvement"
FAMILY = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / FAMILY_ID
SCORER = REPO / "verifiers" / FAMILY_ID / "score_objective_delta.py"

DEFAULT_VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

PROMPT = (
    "Read AGENTS.md in this directory and follow it exactly. "
    "Write brief_input.json at the workspace root and run ./bin/cnb55-brief submit brief_input.json "
    "to produce brief/manager_brief.json. Do not modify anything outside brief/ or brief_input.json."
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def monotonicity(values: list[int], tolerance: int = 3) -> bool:
    return all(values[i] + tolerance >= values[i + 1] for i in range(len(values) - 1))


def summarise(results: list[dict]) -> dict:
    scores = [int(item["score"]) for item in results]
    return {
        "family_mean": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "max_variant_score": max(scores) if scores else 0,
        "min_variant_score": min(scores) if scores else 0,
        "monotonicity_tolerance_3": monotonicity(scores, tolerance=3),
        "oracle_floor": None,
        "freeze_gate_window": [15, 25],
    }


def acceptance(summary: dict) -> str:
    mean_ok = 15 <= summary["family_mean"] <= 25
    max_ok = summary["max_variant_score"] <= 40
    min_ok = summary["min_variant_score"] <= 10
    mono_ok = summary["monotonicity_tolerance_3"]
    if mean_ok and max_ok and min_ok and mono_ok:
        return "Layer A freeze gate passed"
    return "Layer A freeze gate not yet passed"


def write_markdown(attempt_dir: Path, command: str, results: list[dict], summary: dict) -> None:
    lines = [
        f"# {attempt_dir.name} live probe",
        "",
        f"- command: `{command}`",
        f"- family mean: `{summary['family_mean']}`",
        f"- max variant score: `{summary['max_variant_score']}`",
        f"- min variant score: `{summary['min_variant_score']}`",
        f"- monotonicity within ±3: `{summary['monotonicity_tolerance_3']}`",
        f"- judgment: `{acceptance(summary)}`",
        "",
        "| variant | codex_exit | seconds | score | M_training | pass | integrity | ceilings | errors | accepted |",
        "|---|---:|---:|---:|---:|---|---:|---|---|---|",
    ]
    for item in results:
        ceilings = ",".join(item["ceilings"]) or "—"
        errors = ",".join(item["errors"]) or "—"
        accepted = item.get("accepted") or "—"
        lines.append(
            f"| {item['variant']} | {item['codex_exit']} | {item['seconds']} | {item['score']} | "
            f"{item['M_training']:.2f} | {item['pass']} | {item['integrity_flag']} | "
            f"{ceilings} | {errors} | {accepted} |"
        )
    (attempt_dir / "summary.md").write_text("\n".join(lines) + "\n")


def run_variant(variant: str, timeout_seconds: int, attempt_dir: Path) -> dict:
    logs_dir = attempt_dir / "codex_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{variant}.log"
    work_root = Path(tempfile.mkdtemp(prefix=f"{FAMILY_ID}_{variant}_"))
    ws = work_root / "workspace"
    shutil.copytree(WORKSPACE_BUNDLE / variant, ws)
    result_path = work_root / "verify_result.json"

    start = time.time()
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            [
                "timeout",
                str(timeout_seconds),
                "codex",
                "exec",
                "--cd",
                str(ws),
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                "--color",
                "never",
                "--ephemeral",
                PROMPT,
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = int(time.time() - start)

    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(ws),
            "VERIFIER_DATA": str(VERIFIER_DATA),
            "RESULT_FILE": str(result_path),
            "VARIANT_ID": variant,
            "CNB55_SEED": "42",
        }
    )
    subprocess.run(["python3", str(SCORER)], env=env, check=True)
    result = load_json(result_path)

    accepted = None
    brief_path = ws / "brief" / "manager_brief.json"
    if brief_path.exists():
        try:
            accepted = load_json(brief_path).get("accepted")
        except Exception:
            accepted = None

    return {
        "variant": variant,
        "codex_exit": proc.returncode,
        "seconds": elapsed,
        "score": int(result["score"]),
        "M_training": float(result["M_training"]),
        "pass": bool(result["pass"]),
        "integrity_flag": int(result["integrity_flag"]),
        "ceilings": list(result.get("ceilings_applied", [])),
        "errors": list(result.get("errors", [])),
        "accepted": accepted,
        "result_path": str(result_path),
        "log_path": str(log_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True, help="attempt label, e.g. attempt_04")
    ap.add_argument("--timeout-seconds", type=int, default=900)
    ap.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    args = ap.parse_args()

    attempt_dir = FAMILY / "live_probe" / args.attempt
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    results = [run_variant(variant, args.timeout_seconds, attempt_dir) for variant in args.variants]
    summary = summarise(results)
    command = (
        f"python3 verifiers/{FAMILY_ID}/run_live_probe.py --attempt {args.attempt} "
        f"--timeout-seconds {args.timeout_seconds} --variants {' '.join(args.variants)}"
    )
    payload = {
        "family_id": FAMILY_ID,
        "attempt": args.attempt,
        "command": command,
        "prompt": PROMPT,
        "results": results,
        "summary": summary,
        "acceptance_judgment": acceptance(summary),
    }
    (attempt_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(attempt_dir, command, results, summary)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
