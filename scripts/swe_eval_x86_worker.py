#!/usr/bin/env python3
"""Standalone native-x86_64 SWE-Bench eval worker (runs on the alienware box).

Driven over SSH by the DGX Spark orchestrator: receives one (instance_id,
patch.diff) pair, runs the official swebench harness natively on x86_64
(no arch shim, no emulation — every instance's conda env builds because the
old python/dep pins exist as x86_64 packages), and writes the same artifact
set the local arm64 path produces:

    <output-dir>/predictions.jsonl
    <output-dir>/eval_report.json
    <output-dir>/normalized_eval.json
    <output-dir>/eval.log
    <output-dir>/logs/run_evaluation/<run_id>/<model>/<id>/...

Exit codes mirror codex-bench-eval-swe: 0 resolved, 1 failed, 2 crash.

Disk discipline: cache_level defaults to 'env' (reuse env images across
same-repo instances). The orchestrator-side loop prunes images when the
docker partition gets low; this worker also accepts --cache-level to force
'base' + clean for tight-disk situations.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any

EXIT_RESOLVED, EXIT_FAILED, EXIT_CRASH = 0, 1, 2


def _classify(payload: dict[str, Any]) -> tuple[str, bool, str | None]:
    if not payload:
        return "crash", False, "infra_error"
    if payload.get("resolved"):
        return "resolved", True, "tests_passed"
    if payload.get("patch_successfully_applied") is False:
        return "failed", False, "patch_apply_failed"
    return "failed", False, "tests_failed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--patch-path", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model-name", default="qwen3.6-27b-fp8::codex-cli-0.128.0::q36-a")
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--cache-level", default="env", choices=("none", "base", "env", "instance"))
    ap.add_argument("--clean", default="false")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    elog = out / "eval.log"

    def log(*lines: str) -> None:
        with elog.open("a", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln.rstrip("\n") + "\n")

    arch = platform.machine().lower()  # expect x86_64 on alienware
    instance_id = args.instance_id
    predictions_path = out / "predictions.jsonl"
    run_id = args.run_id or f"x86eval-{out.name}-{int(time.time())}"

    if not args.patch_path.is_file():
        log(f"patch_path missing: {args.patch_path}")
        (out / "eval_report.json").write_text(json.dumps({
            "instance_id": instance_id, "verdict": "crash", "passed": False,
            "failure_mode": "infra_error", "error": "patch_missing",
            "arch": arch, "eval_host": platform.node(),
        }, indent=2))
        return EXIT_CRASH

    patch_text = args.patch_path.read_text()
    predictions_path.write_text(json.dumps({
        "instance_id": instance_id,
        "model_name_or_path": args.model_name,
        "model_patch": patch_text,
    }) + "\n")

    if not patch_text.strip():
        log("empty patch; skipping harness")
        (out / "eval_report.json").write_text(json.dumps({
            "instance_id": instance_id, "verdict": "failed", "passed": False,
            "failure_mode": "patch_apply_failed", "error": "empty_patch",
            "arch": arch, "eval_host": platform.node(),
            "eval_wall_clock_seconds": 0.0,
        }, indent=2))
        (out / "normalized_eval.json").write_text(json.dumps({
            "track": "swe_bench", "instance_id": instance_id, "outcome": "failed",
            "failure_mode": "patch_apply_failed", "arch": arch,
        }, indent=2))
        return EXIT_FAILED

    from swebench.harness import run_evaluation as _re

    clean = str(args.clean).lower() in {"1", "true", "yes"}
    cwd_save = Path.cwd()
    os.chdir(out)
    h_out, h_err = io.StringIO(), io.StringIO()
    start = time.monotonic()
    exit_code = 0
    err: str | None = None
    try:
        with contextlib.redirect_stdout(h_out), contextlib.redirect_stderr(h_err):
            _re.main(
                dataset_name=args.dataset_name,
                split=args.split,
                instance_ids=[instance_id],
                predictions_path=str(predictions_path),
                max_workers=1,
                force_rebuild=False,
                cache_level=args.cache_level,
                clean=clean,
                open_file_limit=4096,
                run_id=run_id,
                timeout=args.timeout_s,
                namespace="swebench",
                rewrite_reports=False,
                modal=False,
                instance_image_tag="latest",
                env_image_tag="latest",
                report_dir=str(out),
            )
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:  # noqa: BLE001
        exit_code = -1
        err = f"{type(exc).__name__}: {exc}"
        log(traceback.format_exc())
    finally:
        os.chdir(cwd_save)
        elapsed = time.monotonic() - start

    log(f"harness elapsed_s={elapsed:.3f} exit_code={exit_code} arch={arch}",
        "-- stdout --", h_out.getvalue(), "-- stderr --", h_err.getvalue())

    report = None
    try:
        cand = (out / "logs" / "run_evaluation" / run_id /
                args.model_name.replace("/", "__") / instance_id / "report.json")
        if cand.is_file():
            report = json.loads(cand.read_text())
        else:
            summ = next(out.glob(f"*.{run_id}.json"), None)
            if summ is not None:
                sp = json.loads(summ.read_text())
                if instance_id in sp.get("resolved_ids", []):
                    report = {"resolved": True}
                elif instance_id in sp.get("unresolved_ids", []):
                    report = {"resolved": False, "patch_successfully_applied": True}
                elif instance_id in sp.get("empty_patch_ids", []):
                    report = {"resolved": False, "patch_successfully_applied": False}
    except Exception as exc:  # noqa: BLE001
        log(f"failed to read report: {exc}", traceback.format_exc())

    if report is None:
        verdict, passed, fmode = "crash", False, "infra_error"
    else:
        payload = report[instance_id] if (isinstance(report, dict) and instance_id in report) else report
        verdict, passed, fmode = _classify(payload if isinstance(payload, dict) else {})

    (out / "eval_report.json").write_text(json.dumps({
        "track": "swe_bench", "instance_id": instance_id,
        "model_id": args.model_name, "dataset_name": args.dataset_name,
        "verdict": verdict, "passed": passed, "failure_mode": fmode,
        "harness_exit_code": exit_code,
        "eval_wall_clock_seconds": round(elapsed, 3),
        "error": err, "arch": arch, "eval_host": platform.node(),
        "run_id": run_id, "namespace": "swebench", "cache_level": args.cache_level,
    }, indent=2))
    (out / "normalized_eval.json").write_text(json.dumps({
        "track": "swe_bench", "instance_id": instance_id, "outcome": verdict,
        "failure_mode": fmode, "dataset_name": args.dataset_name,
        "model_id": args.model_name, "eval_wall_clock_seconds": round(elapsed, 3),
        "arch": arch, "eval_host": platform.node(),
    }, indent=2))

    print(f"VERDICT {instance_id} {verdict} {fmode} elapsed_s={elapsed:.1f} arch={arch}", flush=True)
    return EXIT_RESOLVED if verdict == "resolved" else (EXIT_FAILED if verdict == "failed" else EXIT_CRASH)


if __name__ == "__main__":
    raise SystemExit(main())
