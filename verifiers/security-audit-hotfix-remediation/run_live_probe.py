#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY = REPO / "benchmark_blueprints/families/security-audit-hotfix-remediation"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data/security-audit-hotfix-remediation"
SCORER = REPO / "verifiers/security-audit-hotfix-remediation/score_hotfix.py"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

PROMPT = (
    "Read AGENTS.md and follow it exactly. Inspect the audit artifacts, repo code, "
    "request samples, and any variant-specific corpora. Implement the smallest safe "
    "hotfix in repo/export_api, author audit/triage_note.json and repo/ops/deploy_hotfix.json, "
    "optionally add repo/tests/test_hotfix_regressions.py, validate the JSON artifacts "
    "with python3 bin/cnb55-hotfix-artifacts validate audit/triage_note.json repo/ops/deploy_hotfix.json, "
    "and run make test. Do not modify immutable evidence or visible tests."
)


def run_one(variant: str, model: str, reasoning: str, timeout: int, run_index: int, probe_run_id: str) -> dict:
    start = time.time()
    with tempfile.TemporaryDirectory(prefix=f"security_hotfix_probe_{variant}_") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, ws)
        cmd = [
            "codex",
            "-a",
            "never",
            "exec",
            "--skip-git-repo-check",
            "--json",
            "-m",
            model,
            "-c",
            f'reasoning_effort="{reasoning}"',
            "-c",
            f'model_reasoning_effort="{reasoning}"',
            "-s",
            "workspace-write",
            "-C",
            str(ws),
            PROMPT,
        ]
        timed_out = False
        solve_stdout = ""
        solve_stderr = ""
        solve_returncode = 0
        try:
            solve = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
            solve_stdout = solve.stdout
            solve_stderr = solve.stderr
            solve_returncode = solve.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            solve_stdout = exc.stdout or ""
            solve_stderr = exc.stderr or ""
            solve_returncode = 124

        result_path = Path(tmp) / "verify_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "RESULT_FILE": str(result_path),
                "VARIANT_ID": variant,
            }
        )
        subprocess.run([sys.executable, str(SCORER)], check=True, env=env)
        scored = json.loads(result_path.read_text(encoding="utf-8"))
        if timed_out:
            scored.setdefault("errors", []).append(f"codex exec timed out after {timeout} seconds")
        scored.update(
            {
                "probe_run_id": probe_run_id,
                "variant": variant,
                "run_index": run_index,
                "model": model,
                "reasoning": reasoning,
                "command": " ".join(cmd),
                "prompt": PROMPT,
                "codex_exit": solve_returncode,
                "codex_timed_out": timed_out,
                "codex_seconds": round(time.time() - start, 2),
                "solve_stdout_tail": solve_stdout[-4000:],
                "solve_stderr_tail": solve_stderr[-4000:],
            }
        )
        return scored


def summarize(records: list[dict]) -> dict:
    by_variant: dict[str, list[dict]] = {variant: [] for variant in VARIANTS}
    for record in records:
        by_variant[record["variant"]].append(record)

    variants: dict[str, dict] = {}
    for variant in VARIANTS:
        rows = by_variant[variant]
        scores = [row["P_benchmark"] for row in rows]
        raw_scores = [row["raw_score_pre_ceiling"] for row in rows]
        m_scores = [row["M_training"] for row in rows]
        ceiling_hits: dict[str, int] = {}
        for row in rows:
            for ceiling in row.get("ceilings_applied", []):
                ceiling_hits[ceiling] = ceiling_hits.get(ceiling, 0) + 1
        variants[variant] = {
            "n": len(rows),
            "mean": statistics.mean(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "scores": scores,
            "raw_scores": raw_scores,
            "mean_M_training": statistics.mean(m_scores),
            "stdev_M_training": statistics.stdev(m_scores) if len(m_scores) > 1 else 0.0,
            "ceiling_hits": ceiling_hits,
            "shortcut_runs": sum(1 for row in rows if row.get("shortcut_detected")),
        }

    means = [variants[v]["mean"] for v in VARIANTS]
    family_mean = statistics.mean(means)
    max_variant_mean = max(means)
    min_variant_mean = min(means)
    monotonic_breaks = []
    for idx in range(len(VARIANTS) - 1):
        a = VARIANTS[idx]
        b = VARIANTS[idx + 1]
        if variants[a]["mean"] + 3.0 < variants[b]["mean"]:
            monotonic_breaks.append(f"{a} ({variants[a]['mean']:.2f}) < {b} ({variants[b]['mean']:.2f}) beyond +/-3")
    gate = {
        "family_mean": family_mean,
        "family_mean_ok": 15.0 <= family_mean <= 25.0,
        "max_variant_mean": max_variant_mean,
        "max_variant_ok": max_variant_mean <= 40.0,
        "min_variant_mean": min_variant_mean,
        "hard_variant_ok": min_variant_mean <= 10.0,
        "monotonic_ok": len(monotonic_breaks) == 0,
        "monotonic_breaks": monotonic_breaks,
    }
    gate["all_pass"] = all([gate["family_mean_ok"], gate["max_variant_ok"], gate["hard_variant_ok"], gate["monotonic_ok"]])
    gate["max_observed_stdev_M_training"] = max(variants[v]["stdev_M_training"] for v in VARIANTS)
    return {"variants": variants, "gate": gate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning", default="high")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--jsonl-out", default=str(FAMILY / "report" / "probe_runs.jsonl"))
    parser.add_argument("--summary-out", default=str(FAMILY / "report" / "probe_summary_latest.json"))
    parser.add_argument("--probe-run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()

    variants = [args.variant] if args.variant else VARIANTS
    jsonl_out = Path(args.jsonl_out)
    summary_out = Path(args.summary_out)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_out.exists():
        jsonl_out.unlink()

    records: list[dict] = []
    with jsonl_out.open("a", encoding="utf-8") as handle:
        for variant in variants:
            for run_index in range(1, args.repeats + 1):
                print(f"[probe] {variant} run {run_index}/{args.repeats}", flush=True)
                record = run_one(variant, args.model, args.reasoning, args.timeout, run_index, args.probe_run_id)
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                print(
                    f"  score={record['P_benchmark']} raw={record['raw_score_pre_ceiling']} "
                    f"pass={record['pass']} ceilings={record['ceilings_applied']}",
                    flush=True,
                )

    summary = summarize(records)
    summary["probe_run_id"] = args.probe_run_id
    summary["model"] = args.model
    summary["reasoning"] = args.reasoning
    summary["repeats"] = args.repeats
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
