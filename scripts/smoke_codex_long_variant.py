#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from lumo_flywheel_serving.yaml_utils import load_yaml_file

GRADER_DOCKERFILE_LABEL = "org.lumo.codex_long_grader_dockerfile_sha"


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )


def _docker_image_exists(image_ref: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image_ref],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _docker_image_label(image_ref: str, label: str) -> str | None:
    result = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            image_ref,
            "--format",
            f"{{{{ index .Config.Labels {json.dumps(label)} }}}}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    if not value or value == "<no value>":
        return None
    return value


def _grader_dockerfile_sha(repo_root: Path) -> str:
    dockerfile = repo_root / "docker" / "Dockerfile.codex-long-grader"
    return hashlib.sha256(dockerfile.read_bytes()).hexdigest()


def _ensure_grader_image(repo_root: Path, grader_image: str) -> None:
    expected_sha = _grader_dockerfile_sha(repo_root)
    if _docker_image_exists(grader_image) and _docker_image_label(grader_image, GRADER_DOCKERFILE_LABEL) == expected_sha:
        return
    _run(
        [
            "docker",
            "build",
            "--build-arg",
            f"GRADER_DOCKERFILE_SHA={expected_sha}",
            "-t",
            grader_image,
            "-f",
            str(repo_root / "docker" / "Dockerfile.codex-long-grader"),
            str(repo_root),
        ]
    )


def _prepare_build_context(variant_dir: Path, repo_override: Path | None) -> tuple[Path, bool]:
    if repo_override is None:
        return variant_dir, False

    temp_dir = Path(tempfile.mkdtemp(prefix="codex-long-build-context-"))
    dockerfile_text = (variant_dir / "Dockerfile").read_text(encoding="utf-8")
    smoke_block = (
        "RUN set -eux; \\\n"
        "    set +e; \\\n"
        "    "
    )
    if smoke_block in dockerfile_text:
        dockerfile_text = dockerfile_text.split("RUN set -eux; \\\n", 1)[0].rstrip() + "\n\nRUN true\n"
    (temp_dir / "Dockerfile").write_text(dockerfile_text, encoding="utf-8")
    shutil.copytree(repo_override, temp_dir / "repo")
    return temp_dir, True


def _functional_run(
    image_ref: str,
    functional_dir: Path,
    check_id: str,
    command: str,
    timeout_seconds: int,
) -> None:
    container_name = f"codex-long-functional-{hashlib.sha256(f'{image_ref}:{check_id}'.encode('utf-8')).hexdigest()[:12]}"
    shell_command = f"{command} > /functional/{check_id}_output.log 2>&1; echo $? > /functional/{check_id}_exit_code"
    try:
        _run(
            [
                "docker",
                "run",
                "--name",
                container_name,
                "--rm",
                "--network",
                "none",
                "-v",
                f"{functional_dir}:/functional",
                image_ref,
                "sh",
                "-lc",
                shell_command,
            ],
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        _run(["docker", "rm", "-f", container_name], check=False)
        (functional_dir / f"{check_id}_exit_code").write_text("124\n", encoding="utf-8")
        output_log = functional_dir / f"{check_id}_output.log"
        existing = output_log.read_text(encoding="utf-8") if output_log.exists() else ""
        output_log.write_text(
            existing + f"\nfunctional check timed out after {timeout_seconds}s\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and smoke a Codex-Long variant through Phase 2/3 grading.")
    parser.add_argument("--family", required=True, help="Family id under scenario_families/")
    parser.add_argument("--variant", required=True, help="Variant id under scenario_families/<family>/variants/")
    parser.add_argument(
        "--repo-override",
        type=Path,
        help="Optional replacement repo tree. This lets later red-team rounds run a modified candidate fix through the same smoke path.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--grader-image", default="codex-long-grader-local")
    parser.add_argument("--expect", choices=("pass", "fail"), default="fail")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    family_dir = repo_root / "scenario_families" / args.family
    variant_dir = family_dir / "variants" / args.variant
    if not variant_dir.exists():
        raise SystemExit(f"unknown variant path: {variant_dir}")

    family_spec = load_yaml_file(family_dir / "family.yaml") or {}
    grading = family_spec.get("grading_invariant", {})
    functional_checks = grading.get("functional_checks", [])
    if not functional_checks:
        raise SystemExit(f"family '{args.family}' does not define functional_checks")

    build_context, ephemeral_context = _prepare_build_context(variant_dir, args.repo_override.resolve() if args.repo_override else None)
    image_ref = f"codex-long-smoke-{args.family}-{args.variant}".replace("/", "-")
    extract_container = f"codex-long-extract-{args.family}-{args.variant}".replace("/", "-")
    temp_root = Path(tempfile.mkdtemp(prefix="codex-long-smoke-"))
    functional_dir = temp_root / "functional"
    agent_root = temp_root / "agent_root"
    results_dir = temp_root / "results"
    functional_dir.mkdir(parents=True, exist_ok=True)
    agent_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        _run(["docker", "build", "-t", image_ref, "-f", str(build_context / "Dockerfile"), str(build_context)])

        _run(
            [
                "docker",
                "run",
                "--rm",
                image_ref,
                "sh",
                "-lc",
                "test -f /workspace/AGENTS.md && ! test -e /workspace/verifiers && ! test -e /workspace/verifier_data && ! test -e /workspace/oracle",
            ]
        )

        for check in functional_checks:
            timeout_seconds = int(check["timeout_seconds"])
            if timeout_seconds <= 0:
                raise SystemExit(
                    f"functional check '{check['id']}' for family '{args.family}' must declare timeout_seconds > 0"
                )
            _functional_run(
                image_ref,
                functional_dir,
                str(check["id"]),
                str(check["command"]),
                timeout_seconds,
            )

        _run(["docker", "create", "--name", extract_container, image_ref, "true"])
        try:
            _run(["docker", "cp", f"{extract_container}:.", str(agent_root)])
        finally:
            _run(["docker", "rm", "-f", extract_container], check=False)

        _ensure_grader_image(repo_root, args.grader_image)
        verify = _run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-v",
                f"{agent_root}:/agent:ro",
                "-v",
                f"{functional_dir}:/functional:ro",
                "-v",
                f"{repo_root / 'verifiers' / args.family}:/verifier:ro",
                "-v",
                f"{repo_root / 'verifier_data' / args.family}:/verifier_data:ro",
                "-v",
                f"{results_dir}:/results",
                args.grader_image,
                "/verifier/verify.sh",
            ],
            capture=True,
        )
        result_path = results_dir / "verify_result.json"
        if not result_path.exists():
            raise SystemExit("Phase 3 did not produce verify_result.json")
        verify_result = json.loads(result_path.read_text(encoding="utf-8"))
        expected_pass = args.expect == "pass"
        if bool(verify_result.get("pass")) != expected_pass:
            print(verify.stdout)
            print(verify.stderr)
            raise SystemExit(
                f"expected Phase 3 pass={expected_pass} for {args.family}/{args.variant}, "
                f"got {verify_result.get('pass')}"
            )

        payload = {
            "family": args.family,
            "variant": args.variant,
            "image_ref": image_ref,
            "expect": args.expect,
            "verify_result": verify_result,
            "functional_dir": str(functional_dir),
            "results_dir": str(results_dir),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"smoke complete for {args.family}/{args.variant}: "
                f"pass={verify_result.get('pass')} shortcut_detected={verify_result.get('shortcut_detected')}"
            )
    finally:
        if not args.keep_artifacts:
            _run(["docker", "rmi", "-f", image_ref], check=False)
            shutil.rmtree(temp_root, ignore_errors=True)
            if ephemeral_context:
                shutil.rmtree(build_context, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
