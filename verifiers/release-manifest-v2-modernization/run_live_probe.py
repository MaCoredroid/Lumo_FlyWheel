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
FAMILY_ID = "release-manifest-v2-modernization"
FAMILY = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / FAMILY_ID
SCORER = REPO / "verifiers" / FAMILY_ID / "score_release_modernization.py"

DEFAULT_VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

PROMPT = (
    "Read AGENTS.md in this directory and follow it exactly. "
    "Repair the release-path modernization in place, run the visible checks, "
    "and write artifacts/release_smoke_report.json by running "
    "`python deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json`. "
    "Do not modify read-only surfaces or anything outside the allowed repair paths."
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
        f"- monotonicity within +/-3: `{summary['monotonicity_tolerance_3']}`",
        f"- judgment: `{acceptance(summary)}`",
        "",
        "| variant | codex_exit | seconds | score | M_training | pass | integrity | ceilings | errors |",
        "|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for item in results:
        ceilings = ",".join(item["ceilings"]) or "—"
        errors = ",".join(item["errors"]) or "—"
        lines.append(
            f"| {item['variant']} | {item['codex_exit']} | {item['seconds']} | {item['score']} | "
            f"{item['M_training']:.2f} | {item['pass']} | {item['integrity_flag']} | {ceilings} | {errors} |"
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
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    subprocess.run([os.environ.get("PYTHON", "python3"), str(SCORER)], env=env, check=True)
    result = load_json(result_path)

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
        "result_path": str(result_path),
        "log_path": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    args = parser.parse_args()

    attempt_dir = FAMILY / "report" / args.attempt
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
