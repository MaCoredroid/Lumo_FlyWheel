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


def _native_arch() -> str:
    machine = platform.machine().lower()
    return "arm64" if machine in {"aarch64", "arm64"} else "x86_64"


def _apply_arch_shim(arch: str) -> str:
    """Patch swebench.make_test_spec so the harness uses the given arch.

    The upstream CLI exposes no arch knob and hard-defaults to x86_64. We
    force the chosen arch ("arm64" for native aarch64 runs, or "x86_64" for
    the QEMU-emulation fallback on instances whose env can't build natively
    on aarch64). Idempotent; re-applying with a new arch re-patches.
    """
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


def _x86_emulation_available() -> bool:
    """True if qemu-x86_64 is registered in binfmt_misc (Docker can run amd64)."""
    try:
        data = Path("/proc/sys/fs/binfmt_misc/qemu-x86_64").read_text()
        return "enabled" in data
    except Exception:  # noqa: BLE001
        return False


def _arm64_env_build_failed(output_dir: Path, run_id: str) -> bool:
    """Detect the conda/arm64 env-build failure signature in harness logs.

    These are the instances whose Python/dep pins have no linux-aarch64
    conda build (python<=3.6, setuptools==38.2.4, etc.). Signature is a
    BuildImageError / PackagesNotFoundError in the build_images log tree.
    """
    log_root = output_dir / "logs"
    if not log_root.is_dir():
        return False
    signatures = (
        "PackagesNotFoundError",
        "BuildImageError",
        "ResolvePackageNotFound",
        "No module named 'pkg_resources'",
    )
    try:
        for build_log in log_root.rglob("build_image.log"):
            text = build_log.read_text(errors="replace")
            if any(sig in text for sig in signatures):
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


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

    # Step 3: invoke the harness, with an arm64 -> x86_64-emulation fallback.
    try:
        from swebench.harness import run_evaluation as _re  # noqa: F401
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

    def _run_attempt(arch: str, run_id: str) -> dict[str, Any]:
        """One harness invocation for a given arch. Returns a result dict."""
        from swebench.harness import run_evaluation as _re

        _apply_arch_shim(arch)
        namespace_arg = args.namespace.strip().lower()
        if namespace_arg in {"none", "local"}:
            eff_ns: str | None = None
        elif namespace_arg == "auto":
            eff_ns = _resolve_auto_namespace(instance_id, arch)
        else:
            eff_ns = args.namespace
        # x86_64 emulation path always uses the prebuilt swebench namespace
        # (building old x86 envs locally under emulation is far slower and
        # unnecessary — the team publishes x86_64 images for every instance).
        if arch == "x86_64":
            eff_ns = "swebench"

        _write_eval_log(output_dir, [
            f"[attempt arch={arch}] dataset={dataset_name} split={args.split} run_id={run_id}",
            f"[attempt arch={arch}] effective_namespace={eff_ns} "
            f"timeout_s={args.timeout_s} cache_level={args.cache_level}",
        ])

        cwd_save = Path.cwd()
        os.chdir(output_dir)
        h_stdout, h_stderr = io.StringIO(), io.StringIO()
        start = time.monotonic()
        exit_code = 0
        err: str | None = None
        try:
            with contextlib.redirect_stdout(h_stdout), contextlib.redirect_stderr(h_stderr):
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
                    namespace=eff_ns,
                    rewrite_reports=False,
                    modal=False,
                    instance_image_tag="latest",
                    env_image_tag="latest",
                    report_dir=str(output_dir),
                )
        except SystemExit as exc:
            exit_code = int(exc.code) if isinstance(exc.code, int) else 1
        except Exception as exc:  # noqa: BLE001
            exit_code = -1
            err = f"{type(exc).__name__}: {exc}"
            _write_eval_log(output_dir, [traceback.format_exc()])
        finally:
            os.chdir(cwd_save)
            elapsed = time.monotonic() - start

        _write_eval_log(output_dir, [
            f"[attempt arch={arch}] harness elapsed_s={elapsed:.3f} exit_code={exit_code}",
            "-- harness stdout --", h_stdout.getvalue(),
            "-- harness stderr --", h_stderr.getvalue(),
        ])

        # Read + classify the per-instance report.
        report = None
        try:
            cand = (output_dir / "logs" / "run_evaluation" / run_id /
                    args.model_name.replace("/", "__") / instance_id / "report.json")
            if cand.is_file():
                report = json.loads(cand.read_text())
            else:
                summary = next(output_dir.glob(f"*.{run_id}.json"), None)
                if summary is not None:
                    sp = json.loads(summary.read_text())
                    if instance_id in sp.get("resolved_ids", []):
                        report = {"resolved": True}
                    elif instance_id in sp.get("unresolved_ids", []):
                        report = {"resolved": False, "patch_successfully_applied": True}
                    elif instance_id in sp.get("empty_patch_ids", []):
                        report = {"resolved": False, "patch_successfully_applied": False}
        except Exception as exc:  # noqa: BLE001
            _write_eval_log(output_dir, [f"failed to read harness report: {exc}",
                                         traceback.format_exc()])

        if report is None:
            verdict, passed, failure_mode = "crash", False, "infra_error"
        else:
            payload = report[instance_id] if (isinstance(report, dict) and instance_id in report) else report
            verdict, passed, failure_mode = _classify_failure(payload if isinstance(payload, dict) else {})

        return {
            "arch": arch, "run_id": run_id, "namespace": eff_ns,
            "verdict": verdict, "passed": passed, "failure_mode": failure_mode,
            "exit_code": exit_code, "elapsed": elapsed, "error": err,
        }

    native = _native_arch()
    result = _run_attempt(native, run_id)

    # Fallback: if the native (arm64) attempt crashed because the conda env
    # can't build on aarch64 (old python / dep pins), retry the SAME instance
    # under x86_64 QEMU emulation using the prebuilt swebench images. This
    # recovers the ~20-35% of SWE-Bench Verified whose env predates aarch64
    # conda support. See docs swe-bench-campaign-progress §"ARM64 carve-out".
    used_x86_fallback = False
    if (
        native == "arm64"
        and result["verdict"] == "crash"
        and _arm64_env_build_failed(output_dir, result["run_id"])
        and _x86_emulation_available()
    ):
        _write_eval_log(output_dir, [
            "[fallback] arm64 env build failed; retrying under x86_64 QEMU emulation",
        ])
        x86_run_id = f"{run_id}-x86emu"
        x86_result = _run_attempt("x86_64", x86_run_id)
        used_x86_fallback = True
        # Prefer the x86 result unless it also crashed (then keep showing it).
        result = x86_result

    _emit_report(
        output_dir,
        instance_id=instance_id,
        dataset_name=dataset_name,
        model_name_or_path=args.model_name,
        patch_path=patch_path,
        predictions_path=predictions_path,
        verdict=result["verdict"],
        passed=result["passed"],
        failure_mode=result["failure_mode"],
        harness_exit_code=result["exit_code"],
        eval_wall_clock_seconds=result["elapsed"],
        error=result["error"],
        extra={
            "arch": result["arch"],
            "run_id": result["run_id"],
            "namespace": result["namespace"] if result["namespace"] is not None else "none",
            "requested_namespace": args.namespace,
            "cache_level": args.cache_level,
            "x86_emulation_fallback": used_x86_fallback,
        },
    )

    if result["verdict"] == "resolved":
        return EXIT_RESOLVED
    if result["verdict"] == "failed":
        return EXIT_FAILED
    return EXIT_CRASH


if __name__ == "__main__":
    raise SystemExit(main())
