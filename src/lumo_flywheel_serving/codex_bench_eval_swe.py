"""codex-bench-eval-swe CLI.

Implements the LLD-05 §4 SWE-bench evaluator entry point. Drives the
upstream `swebench` harness on a single (instance_id, patch) pair under
an attempt-scoped output directory and emits the four-artifact set
(predictions.jsonl, eval.log, eval_report.json, normalized_eval.json).

Exit-code contract (LLD-05 §4.2):
  0 -> resolved (patch passes the instance's gold tests)
  1 -> failed   (patch evaluated, did not resolve)
  2 -> crash    (anything else: missing patch, harness exception, etc.)

ARM64 note: upstream `swebench.harness.test_spec.test_spec.make_test_spec`
defaults to `arch="x86_64"` and the CLI surface does not expose the knob.
On aarch64 hosts (DGX Spark) we monkey-patch the symbol so the harness
pulls the prebuilt `swebench/sweb.eval.arm64.<id>:latest` images.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("codex-bench-eval-swe")

EXIT_RESOLVED = 0
EXIT_FAILED = 1
EXIT_CRASH = 2

DEFAULT_MODEL_NAME = "codex-bench-eval-swe::unknown"
DEFAULT_TIMEOUT_S = 1800


def _resolve_auto_namespace(instance_id: str, arch: str) -> str | None:
    """Probe Docker Hub for the prebuilt eval image; fall back to local build.

    At 2026-05, ~45%% of SWE-Bench Verified arm64 instances lack a published
    `swebench/sweb.eval.arm64.<id>:latest` manifest, while the local build path
    works for every instance (slower on first hit, cached thereafter). Auto
    mode prefers the fast path when the manifest exists.
    """
    image = f"swebench/sweb.eval.{arch}.{instance_id.lower().replace('__', '_1776_')}:latest"
    try:
        rc = subprocess.run(
            ["docker", "manifest", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).returncode
    except Exception:  # noqa: BLE001
        rc = 1
    return "swebench" if rc == 0 else None


def _apply_arm64_shim() -> str:
    """Patch swebench.make_test_spec so the harness picks arm64 images.

    Returns the arch string actually wired ("arm64" on aarch64, "x86_64"
    elsewhere). Idempotent.
    """
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"aarch64", "arm64"} else "x86_64"
    from swebench.harness.test_spec import test_spec as _ts
    from swebench.harness import run_evaluation as _re
    from swebench.harness import docker_build as _db

    orig = getattr(_ts, "_lumo_orig_make_test_spec", None) or _ts.make_test_spec
    _ts._lumo_orig_make_test_spec = orig

    def patched(instance, namespace=None, base_image_tag="latest",
                env_image_tag="latest", instance_image_tag="latest",
                arch=arch):
        return orig(
            instance,
            namespace=namespace,
            base_image_tag=base_image_tag,
            env_image_tag=env_image_tag,
            instance_image_tag=instance_image_tag,
            arch=arch,
        )

    _ts.make_test_spec = patched
    _re.make_test_spec = patched
    _db.make_test_spec = patched
    return arch


def _write_eval_log(output_dir: Path, lines: list[str]) -> None:
    log_path = output_dir / "eval.log"
    with log_path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def _emit_report(
    output_dir: Path,
    *,
    instance_id: str,
    dataset_name: str,
    model_name_or_path: str,
    patch_path: Path,
    predictions_path: Path,
    verdict: str,
    passed: bool,
    failure_mode: str | None,
    harness_exit_code: int,
    eval_wall_clock_seconds: float,
    error: str | None,
    extra: dict[str, Any] | None = None,
) -> None:
    report = {
        "track": "swe_bench",
        "instance_id": instance_id,
        "model_id": model_name_or_path,
        "dataset_name": dataset_name,
        "patch_path": str(patch_path),
        "prediction_path": str(predictions_path),
        "verdict": verdict,
        "passed": passed,
        "failure_mode": failure_mode,
        "harness_exit_code": harness_exit_code,
        "eval_wall_clock_seconds": round(eval_wall_clock_seconds, 3),
        "error": error,
    }
    if extra:
        report.update(extra)
    (output_dir / "eval_report.json").write_text(json.dumps(report, indent=2))

    normalized = {
        "track": "swe_bench",
        "instance_id": instance_id,
        "outcome": verdict,
        "failure_mode": failure_mode,
        "dataset_name": dataset_name,
        "model_id": model_name_or_path,
        "eval_wall_clock_seconds": round(eval_wall_clock_seconds, 3),
        "arch": platform.machine().lower(),
    }
    (output_dir / "normalized_eval.json").write_text(json.dumps(normalized, indent=2))


def _build_predictions(
    *,
    instance_id: str,
    patch_text: str,
    model_name_or_path: str,
    predictions_path: Path,
) -> None:
    record = {
        "instance_id": instance_id,
        "model_name_or_path": model_name_or_path,
        "model_patch": patch_text,
    }
    predictions_path.write_text(json.dumps(record) + "\n")


def _classify_failure(report_payload: dict[str, Any]) -> tuple[str, bool, str | None]:
    """Map swebench per-instance report -> (verdict, passed, failure_mode).

    failure_mode enum (LLD-05 §4.4): tests_passed | tests_failed |
    patch_apply_failed | infra_error.
    """
    if not report_payload:
        return "crash", False, "infra_error"

    if report_payload.get("resolved"):
        return "resolved", True, "tests_passed"

    if report_payload.get("patch_successfully_applied") is False:
        return "failed", False, "patch_apply_failed"

    return "failed", False, "tests_failed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="codex-bench-eval-swe",
        description="LLD-05 §4 SWE-bench evaluator entry point.",
    )
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--patch-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--namespace",
        default="auto",
        help=(
            "Image namespace policy. "
            "'auto' (default) probes Docker Hub for the prebuilt arm64 image and "
            "falls back to local build if missing — recommended for ARM64. "
            "'swebench' forces the prebuilt path (crashes for arm64 instances "
            "without a published image, e.g. ~45%% of Verified at 2026-05). "
            "'none' or 'local' forces local build for every instance."
        ),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="String to record in predictions.jsonl.model_name_or_path")
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--cache-level", default="env",
                        choices=("none", "base", "env", "instance"))
    parser.add_argument("--run-id", default=None,
                        help="Optional swebench run_id (default: derived from output dir).")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_path: Path = args.patch_path
    instance_id: str = args.instance_id
    dataset_name: str = args.dataset_name
    predictions_path = output_dir / "predictions.jsonl"
    run_id = args.run_id or f"cbe-{output_dir.name}-{int(time.time())}"

    # Step 1: patch validation
    if not patch_path.exists() or not patch_path.is_file():
        _write_eval_log(output_dir, [
            f"patch_path does not exist or is not a file: {patch_path}",
        ])
        _emit_report(
            output_dir,
            instance_id=instance_id,
            dataset_name=dataset_name,
            model_name_or_path=args.model_name,
            patch_path=patch_path,
            predictions_path=predictions_path,
            verdict="crash",
            passed=False,
            failure_mode="infra_error",
            harness_exit_code=-1,
            eval_wall_clock_seconds=0.0,
            error="patch_missing",
        )
        return EXIT_CRASH

    try:
        patch_text = patch_path.read_text()
    except Exception as exc:  # noqa: BLE001
        _write_eval_log(output_dir, [
            f"failed to read patch_path={patch_path}: {exc}",
            traceback.format_exc(),
        ])
        _emit_report(
            output_dir,
            instance_id=instance_id,
            dataset_name=dataset_name,
            model_name_or_path=args.model_name,
            patch_path=patch_path,
            predictions_path=predictions_path,
            verdict="crash",
            passed=False,
            failure_mode="infra_error",
            harness_exit_code=-1,
            eval_wall_clock_seconds=0.0,
            error=f"patch_read_error: {exc}",
        )
        return EXIT_CRASH

    # Step 2: synthesize upstream predictions.jsonl
    _build_predictions(
        instance_id=instance_id,
        patch_text=patch_text,
        model_name_or_path=args.model_name,
        predictions_path=predictions_path,
    )

    # Empty-patch fast path: harness would mark "empty_patch", we map to failed.
    if not patch_text.strip():
        _write_eval_log(output_dir, ["patch is empty; skipping harness invocation"])
        _emit_report(
            output_dir,
            instance_id=instance_id,
            dataset_name=dataset_name,
            model_name_or_path=args.model_name,
            patch_path=patch_path,
            predictions_path=predictions_path,
            verdict="failed",
            passed=False,
            failure_mode="patch_apply_failed",
            harness_exit_code=0,
            eval_wall_clock_seconds=0.0,
            error="empty_patch",
        )
        return EXIT_FAILED

    # Step 3: invoke the harness in-process with arch shim
    arch = _apply_arm64_shim()

    # Resolve --namespace policy. The upstream CLI converts "none" -> Python None
    # via its optional_str type, but our in-process main(namespace=...) takes the
    # raw value, so we must do the conversion here.
    namespace_arg = args.namespace.strip().lower()
    if namespace_arg == "auto":
        effective_namespace = _resolve_auto_namespace(instance_id, arch)
    elif namespace_arg in {"none", "local"}:
        effective_namespace = None
    else:
        effective_namespace = args.namespace

    _write_eval_log(output_dir, [
        f"arch={arch} dataset={dataset_name} split={args.split} run_id={run_id}",
        f"requested_namespace={args.namespace} effective_namespace={effective_namespace} "
        f"timeout_s={args.timeout_s} cache_level={args.cache_level}",
    ])

    try:
        from swebench.harness import run_evaluation as _re
    except Exception as exc:  # noqa: BLE001
        _write_eval_log(output_dir, [
            f"failed to import swebench.harness.run_evaluation: {exc}",
            traceback.format_exc(),
        ])
        _emit_report(
            output_dir,
            instance_id=instance_id,
            dataset_name=dataset_name,
            model_name_or_path=args.model_name,
            patch_path=patch_path,
            predictions_path=predictions_path,
            verdict="crash",
            passed=False,
            failure_mode="infra_error",
            harness_exit_code=-1,
            eval_wall_clock_seconds=0.0,
            error=f"swebench_import_error: {exc}",
        )
        return EXIT_CRASH

    # Harness writes per-instance logs to ./logs/run_evaluation/<run_id>/<model>/<id>/
    # and a summary JSON to <report_dir>/<model>.<run_id>.json. We chdir into
    # output_dir so all artifacts cluster under the attempt-scoped dir.
    cwd_save = Path.cwd()
    os.chdir(output_dir)
    harness_stdout = io.StringIO()
    harness_stderr = io.StringIO()
    start = time.monotonic()
    harness_exit_code = 0
    harness_error: str | None = None
    try:
        with contextlib.redirect_stdout(harness_stdout), contextlib.redirect_stderr(harness_stderr):
            _re.main(
                dataset_name=dataset_name,
                split=args.split,
                instance_ids=[instance_id],
                predictions_path=str(predictions_path),
                max_workers=1,
                force_rebuild=False,
                cache_level=args.cache_level,
                clean=False,
                open_file_limit=4096,
                run_id=run_id,
                timeout=args.timeout_s,
                namespace=effective_namespace,
                rewrite_reports=False,
                modal=False,
                instance_image_tag="latest",
                env_image_tag="latest",
                report_dir=str(output_dir),
            )
    except SystemExit as exc:
        harness_exit_code = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:  # noqa: BLE001
        harness_exit_code = -1
        harness_error = f"{type(exc).__name__}: {exc}"
        _write_eval_log(output_dir, [traceback.format_exc()])
    finally:
        os.chdir(cwd_save)
        elapsed = time.monotonic() - start

    _write_eval_log(output_dir, [
        f"harness elapsed_s={elapsed:.3f} exit_code={harness_exit_code}",
        "-- harness stdout --",
        harness_stdout.getvalue(),
        "-- harness stderr --",
        harness_stderr.getvalue(),
    ])

    # Step 4: read harness per-instance report and classify
    instance_report = None
    try:
        instance_report_dir = (
            output_dir / "logs" / "run_evaluation" / run_id /
            args.model_name.replace("/", "__") / instance_id
        )
        candidate = instance_report_dir / "report.json"
        if candidate.is_file():
            instance_report = json.loads(candidate.read_text())
        else:
            # Fallback: glob the summary file written by the harness
            summary = next(
                output_dir.glob(f"*.{run_id}.json"),
                None,
            )
            if summary is not None:
                summary_payload = json.loads(summary.read_text())
                if instance_id in summary_payload.get("resolved_ids", []):
                    instance_report = {"resolved": True}
                elif instance_id in summary_payload.get("unresolved_ids", []):
                    instance_report = {"resolved": False, "patch_successfully_applied": True}
                elif instance_id in summary_payload.get("empty_patch_ids", []):
                    instance_report = {"resolved": False, "patch_successfully_applied": False}
                else:
                    instance_report = None
    except Exception as exc:  # noqa: BLE001
        _write_eval_log(output_dir, [
            f"failed to read harness report: {exc}",
            traceback.format_exc(),
        ])

    if instance_report is None:
        verdict, passed, failure_mode = "crash", False, "infra_error"
    else:
        # Per-instance reports from swebench wrap the verdict inside the
        # instance_id key (eg {"astropy__...": {"resolved": true, ...}})
        if isinstance(instance_report, dict) and instance_id in instance_report:
            payload = instance_report[instance_id]
        else:
            payload = instance_report
        verdict, passed, failure_mode = _classify_failure(payload if isinstance(payload, dict) else {})

    _emit_report(
        output_dir,
        instance_id=instance_id,
        dataset_name=dataset_name,
        model_name_or_path=args.model_name,
        patch_path=patch_path,
        predictions_path=predictions_path,
        verdict=verdict,
        passed=passed,
        failure_mode=failure_mode,
        harness_exit_code=harness_exit_code,
        eval_wall_clock_seconds=elapsed,
        error=harness_error,
        extra={
            "arch": arch,
            "run_id": run_id,
            "namespace": effective_namespace if effective_namespace is not None else "none",
            "requested_namespace": args.namespace,
            "cache_level": args.cache_level,
        },
    )

    if verdict == "resolved":
        return EXIT_RESOLVED
    if verdict == "failed":
        return EXIT_FAILED
    return EXIT_CRASH


if __name__ == "__main__":
    raise SystemExit(main())
