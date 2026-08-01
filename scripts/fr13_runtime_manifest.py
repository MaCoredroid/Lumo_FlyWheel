#!/usr/bin/env python3
"""Emit the fail-closed source and data manifest for FR13 fixed-32 campaigns."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class ManifestError(RuntimeError):
    """A manifest input failed a fail-closed validation."""


@dataclass(frozen=True)
class ProfileSpec:
    host_script_source: tuple[str, ...]
    python_package_source: tuple[str, ...]
    runtime_data_and_config: tuple[str, ...]
    required_absence: tuple[str, ...]
    verdict_tools: tuple[str, ...]
    package_dir: str
    package_name: str
    package_file_count: int


FIXED32_HOST_SCRIPT_SOURCE = (
    "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
    "scripts/fr10_quick_decode_tps_probe.py",
    "scripts/fr13_arctic_suffix_adapter.py",
    "scripts/fr13_b4_gdn_bv64_pass.py",
    "scripts/fr13_b4_gdn_bv8_pass.py",
    "scripts/fr13_b4_campaign_driver.sh",
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
    "scripts/fr13_canonical_env.sh",
    "scripts/fr13_cutlass_streamk_pass.py",
    "scripts/fr13_cutlass_wave_binary.py",
    "scripts/fr13_device_multidraft_kernel.py",
    "scripts/fr13_fixed32_flush_protocol.py",
    "scripts/fr13_fixed32_contract.py",
    "scripts/fr13_fixed32_topology.py",
    "scripts/fr13_hardware_floor_ledger.py",
    "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "scripts/fr13_measure.py",
    "scripts/fr13_merged_drafter.py",
    "scripts/fr13_merged_fill.py",
    "scripts/fr13_patch_cutlass_fixed32_wave.py",
    "scripts/fr13_patch_fa2_tree_bias.py",
    "scripts/fr13_required_tree_flags.sh",
    "scripts/fr13_run_b4_gdn_bv8_timing.sh",
    "scripts/fr13_run_b4_gdn_wide_live_gate.sh",
    "scripts/fr13_sg_warmup_capture_inject.py",
    "scripts/gpu_oom_guard.sh",
    "scripts/run_swe_bench_q36_a.py",
    "scripts/sample_dcgm_during_task.py",
    "scripts/swe_x86_helpers/offload_codex_proxy.sh",
    "scripts/swe_x86_helpers/relaunch_proxy_remote.sh",
)

FIXED32_PYTHON_PACKAGE_SOURCE = (
    "src/lumo_flywheel_serving/__init__.py",
    "src/lumo_flywheel_serving/auto_research.py",
    "src/lumo_flywheel_serving/cutlass_overlay_runtime.py",
    "src/lumo_flywheel_serving/data_pool.py",
    "src/lumo_flywheel_serving/fr10_decode_modes.py",
    "src/lumo_flywheel_serving/fr10_equivalence_gate.py",
    "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py",
    "src/lumo_flywheel_serving/fr10_tree_rejection_sampler.py",
    "src/lumo_flywheel_serving/fr13_ex2_silu.py",
    "src/lumo_flywheel_serving/fr13_fa2_spine_reorder.py",
    "src/lumo_flywheel_serving/fr13_replay_conv_remap.py",
    "src/lumo_flywheel_serving/fr13_tree_conv_fused.py",
    "src/lumo_flywheel_serving/inference_proxy.py",
    "src/lumo_flywheel_serving/kernel_activation.py",
    "src/lumo_flywheel_serving/measurement_harness.py",
    "src/lumo_flywheel_serving/metrics.py",
    "src/lumo_flywheel_serving/model_server.py",
    "src/lumo_flywheel_serving/parity_fixture.py",
    "src/lumo_flywheel_serving/parity_probe.py",
    "src/lumo_flywheel_serving/registry.py",
    "src/lumo_flywheel_serving/round_driver.py",
    "src/lumo_flywheel_serving/task_orchestrator.py",
    "src/lumo_flywheel_serving/tuned_config.py",
    "src/lumo_flywheel_serving/workload_p1.py",
    "src/lumo_flywheel_serving/yaml_utils.py",
)

SWE_VERIFIED_CACHE_ROOT = (
    ".cache/huggingface/hub/datasets--princeton-nlp--SWE-bench_Verified"
)
FIXED32_RUNTIME_DATA_AND_CONFIG = (
    ".lumo.local.env",
    f"{SWE_VERIFIED_CACHE_ROOT}/blobs/"
    "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd",
    f"{SWE_VERIFIED_CACHE_ROOT}/refs/main",
    "docker/chat_templates/qwen3-openai-codex.jinja",
    "model_registry.yaml",
    "output/fr13_acceptance_ladder/prompts_swe4.json",
    "config/fr13_fixed32/qwen_system_settings.json",
    "config/fr13_fixed32/subset_b4_four.json",
    "config/fr13_fixed32/subset_b4_sixteen.json",
    "scripts/fr13_dvk_subset_blocks.json",
)

FIXED32_REQUIRED_ABSENCE = ("output/fr13_prewarm/corpus_active.jsonl",)

FIXED32_VERDICT_TOOLS = (
    "scripts/fr13_depth_acceptance.py",
    "scripts/fr13_fixed32_work_census.py",
    "scripts/fr13_floor_gate.py",
    "scripts/fr13_runtime_manifest.py",
)

PROFILES = {
    "fixed32": ProfileSpec(
        host_script_source=FIXED32_HOST_SCRIPT_SOURCE,
        python_package_source=FIXED32_PYTHON_PACKAGE_SOURCE,
        runtime_data_and_config=FIXED32_RUNTIME_DATA_AND_CONFIG,
        required_absence=FIXED32_REQUIRED_ABSENCE,
        verdict_tools=FIXED32_VERDICT_TOOLS,
        package_dir="src/lumo_flywheel_serving",
        package_name="lumo_flywheel_serving",
        package_file_count=25,
    ),
}

SCHEMA = "fr13-runtime-manifest-v1"
CANONICAL_FORMAT = "utf8-json-sort-keys-compact-v1"
READ_CHUNK_BYTES = 1024 * 1024
OPEN_BASE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
OPEN_DIRECTORY_FLAGS = OPEN_BASE_FLAGS | getattr(os, "O_DIRECTORY", 0)
OPEN_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _normalize_repo_relative(raw_path: str, *, label: str) -> str:
    if not raw_path or "\x00" in raw_path or "\\" in raw_path:
        raise ManifestError(f"{label} must be a non-empty POSIX repo-relative path")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or str(path) != raw_path:
        raise ManifestError(
            f"{label} must be a canonical repo-relative path: {raw_path!r}"
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestError(
            f"{label} must not contain empty, dot, or parent components: {raw_path!r}"
        )
    return raw_path


def _validate_profile_spec(spec: ProfileSpec) -> None:
    if len(spec.python_package_source) != spec.package_file_count:
        raise ManifestError(
            "internal profile error: Python package closure must contain exactly "
            f"{spec.package_file_count} files"
        )
    sections = {
        "host_script_source": spec.host_script_source,
        "python_package_source": spec.python_package_source,
        "runtime_data_and_config": spec.runtime_data_and_config,
        "verdict_tools": spec.verdict_tools,
    }
    seen: dict[str, str] = {}
    for section, paths in sections.items():
        if len(paths) != len(set(paths)):
            raise ManifestError(f"internal profile error: duplicate path in {section}")
        for raw_path in paths:
            path = _normalize_repo_relative(raw_path, label=f"internal {section} path")
            if path in seen:
                raise ManifestError(
                    "internal profile error: path appears in multiple closures: "
                    f"{path!r} in {seen[path]} and {section}"
                )
            seen[path] = section
    if len(spec.required_absence) != len(set(spec.required_absence)):
        raise ManifestError("internal profile error: duplicate required-absence path")
    for raw_path in spec.required_absence:
        path = _normalize_repo_relative(
            raw_path, label="internal required-absence path"
        )
        if path in seen:
            raise ManifestError(
                "internal profile error: required-present and required-absent "
                f"path overlap: {path!r}"
            )


def _open_file_no_symlinks(repo: Path, relative_path: str) -> int:
    parts = PurePosixPath(relative_path).parts
    directory_fd = os.open(repo, OPEN_DIRECTORY_FLAGS)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                OPEN_DIRECTORY_FLAGS | OPEN_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            parts[-1],
            OPEN_BASE_FLAGS | OPEN_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except FileNotFoundError as error:
        raise ManifestError(f"missing required file: {relative_path}") from error
    except OSError as error:
        raise ManifestError(
            f"cannot securely open required file {relative_path}: {error.strerror}"
        ) from error
    finally:
        os.close(directory_fd)


def _file_identity(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _read_required_file(
    repo: Path,
    relative_path: str,
    *,
    capture: bool,
) -> tuple[dict[str, object], bytes | None]:
    file_fd = _open_file_no_symlinks(repo, relative_path)
    captured = bytearray() if capture else None
    digest = hashlib.sha256()
    total_size = 0
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ManifestError(f"required path is not a regular file: {relative_path}")
        while True:
            chunk = os.read(file_fd, READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total_size += len(chunk)
            if captured is not None:
                captured.extend(chunk)
        after = os.fstat(file_fd)
    finally:
        os.close(file_fd)
    if _file_identity(before) != _file_identity(after) or total_size != before.st_size:
        raise ManifestError(f"required file changed while hashing: {relative_path}")

    check_fd = _open_file_no_symlinks(repo, relative_path)
    try:
        current = os.fstat(check_fd)
    finally:
        os.close(check_fd)
    if _file_identity(before) != _file_identity(current):
        raise ManifestError(f"required file changed while hashing: {relative_path}")

    record: dict[str, object] = {
        "path": relative_path,
        "sha256": digest.hexdigest(),
        "size": total_size,
    }
    return record, bytes(captured) if captured is not None else None


def _require_absent(repo: Path, relative_path: str) -> None:
    parts = PurePosixPath(relative_path).parts
    directory_fd = os.open(repo, OPEN_DIRECTORY_FLAGS)
    try:
        for part in parts[:-1]:
            try:
                next_fd = os.open(
                    part,
                    OPEN_DIRECTORY_FLAGS | OPEN_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                return
            except OSError as error:
                raise ManifestError(
                    "cannot establish required absence for "
                    f"{relative_path}: {error.strerror}"
                ) from error
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise ManifestError(
            f"required-absence path unexpectedly exists: {relative_path}"
        )
    finally:
        os.close(directory_fd)


def _package_module_name(relative_path: str, package_dir: str) -> str:
    path = PurePosixPath(relative_path)
    if str(path.parent) != package_dir or path.suffix != ".py":
        raise ManifestError(
            "internal profile error: package closure path is outside the flat "
            f"package directory: {relative_path}"
        )
    return path.stem


def _local_import_modules(
    source: bytes, *, filename: str, package_name: str
) -> set[str]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        raise ManifestError(
            f"cannot parse required package source {filename}: "
            f"{error.msg} at line {error.lineno}"
        ) from error

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                prefix = f"{package_name}."
                if alias.name.startswith(prefix):
                    modules.add(alias.name[len(prefix) :].split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level > 1:
                modules.add("..")
            elif node.level == 1:
                if node.module:
                    modules.add(node.module.split(".", 1)[0])
                else:
                    modules.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif node.module and node.module.startswith(f"{package_name}."):
                modules.add(node.module[len(package_name) + 1 :].split(".", 1)[0])
    return modules


def _validate_package_import_closure(
    spec: ProfileSpec,
    package_sources: dict[str, bytes],
) -> None:
    expected_modules = {
        _package_module_name(path, spec.package_dir)
        for path in spec.python_package_source
    }
    expected_modules.discard("__init__")
    imported_modules: set[str] = set()
    for relative_path, source in package_sources.items():
        imported_modules.update(
            _local_import_modules(
                source,
                filename=relative_path,
                package_name=spec.package_name,
            )
        )
    unexpected = sorted(imported_modules - expected_modules)
    if unexpected:
        unexpected_paths = [
            f"{spec.package_dir}/{module}.py" if module != ".." else "<parent package>"
            for module in unexpected
        ]
        raise ManifestError(
            "unexpected local package dependency files outside the reviewed "
            f"{spec.package_file_count}-file closure: {unexpected_paths}"
        )


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def build_manifest(
    repo: Path,
    *,
    profile: str,
    sequence: str,
    spec_override: ProfileSpec | None = None,
) -> dict[str, object]:
    repo = repo.expanduser().resolve(strict=True)
    if not repo.is_dir():
        raise ManifestError(f"repo is not a directory: {repo}")
    try:
        spec = spec_override if spec_override is not None else PROFILES[profile]
    except KeyError as error:
        raise ManifestError(f"unsupported profile: {profile}") from error
    _validate_profile_spec(spec)

    sequence = _normalize_repo_relative(sequence, label="--sequence")
    sequence_path = PurePosixPath(sequence)
    if sequence_path.parts[0] != "scripts" or sequence_path.suffix != ".sh":
        raise ManifestError("--sequence must name a .sh file under scripts/")

    all_static_paths = {
        *spec.host_script_source,
        *spec.python_package_source,
        *spec.runtime_data_and_config,
        *spec.verdict_tools,
        *spec.required_absence,
    }
    if sequence in all_static_paths:
        raise ManifestError(
            f"--sequence unexpectedly overlaps a static profile path: {sequence}"
        )

    closure_paths = {
        "host_script_source": (*spec.host_script_source, sequence),
        "python_package_source": spec.python_package_source,
        "runtime_data_and_config": spec.runtime_data_and_config,
        "verdict_tools": spec.verdict_tools,
    }
    closures: dict[str, list[dict[str, object]]] = {}
    package_sources: dict[str, bytes] = {}
    for section, paths in closure_paths.items():
        records: list[dict[str, object]] = []
        for relative_path in sorted(paths):
            record, source = _read_required_file(
                repo,
                relative_path,
                capture=section == "python_package_source",
            )
            records.append(record)
            if source is not None:
                package_sources[relative_path] = source
        closures[section] = records

    _validate_package_import_closure(spec, package_sources)
    for relative_path in sorted(spec.required_absence):
        _require_absent(repo, relative_path)

    file_count = sum(len(records) for records in closures.values())
    total_size = sum(
        int(record["size"]) for records in closures.values() for record in records
    )
    payload: dict[str, object] = {
        "canonical_format": CANONICAL_FORMAT,
        "closures": closures,
        "profile": profile,
        "required_absence": [
            {"path": path, "required_state": "absent"}
            for path in sorted(spec.required_absence)
        ],
        "schema": SCHEMA,
        "sequence": sequence,
        "summary": {
            "file_count": file_count,
            "python_package_file_count": len(closures["python_package_source"]),
            "total_size": total_size,
        },
    }
    return {
        **payload,
        "overall_canonical_sha256": hashlib.sha256(
            _canonical_bytes(payload)
        ).hexdigest(),
    }


def _render_manifest(manifest: dict[str, object]) -> bytes:
    return (
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_output(
    output: str,
    content: bytes,
    *,
    forbidden_paths: frozenset[Path] = frozenset(),
) -> None:
    if output == "-":
        sys.stdout.buffer.write(content)
        return
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.parent.resolve(strict=True) / output_path.name
    if output_path in forbidden_paths:
        raise ManifestError(
            f"refusing output path reserved by the manifest: {output_path}"
        )
    if output_path.is_symlink():
        raise ManifestError(f"refusing symlink output path: {output_path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _forbidden_output_paths(
    repo: Path,
    spec: ProfileSpec,
    sequence: str,
) -> frozenset[Path]:
    relative_paths = {
        *spec.host_script_source,
        *spec.python_package_source,
        *spec.runtime_data_and_config,
        *spec.required_absence,
        *spec.verdict_tools,
        sequence,
    }
    return frozenset(
        (repo / relative_path).parent.resolve(strict=False)
        / PurePosixPath(relative_path).name
        for relative_path in relative_paths
    )


def _fixture_write(repo: Path, relative_path: str, content: bytes) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def self_test() -> None:
    fixture_spec = ProfileSpec(
        host_script_source=("scripts/driver.sh",),
        python_package_source=(
            "src/fixture_pkg/__init__.py",
            "src/fixture_pkg/helper.py",
        ),
        runtime_data_and_config=(".secret.env", "config/runtime.json"),
        required_absence=("output/fallback/corpus_active.jsonl",),
        verdict_tools=("scripts/verdict.py",),
        package_dir="src/fixture_pkg",
        package_name="fixture_pkg",
        package_file_count=2,
    )
    with tempfile.TemporaryDirectory(prefix="fr13-runtime-manifest-test-") as raw:
        repo = Path(raw)
        fixture_files = {
            "scripts/driver.sh": b"#!/usr/bin/env bash\nset -eu\n",
            "scripts/fixed32_seq.sh": b"run_variant tail fixed32 31 1\n",
            "scripts/verdict.py": b"print('verdict')\n",
            "src/fixture_pkg/__init__.py": b"from .helper import VALUE\n",
            "src/fixture_pkg/helper.py": b"VALUE = 32\n",
            ".secret.env": b"API_TOKEN=do-not-emit-this-value\n",
            "config/runtime.json": b'{"rows":32}\n',
        }
        for relative_path, content in fixture_files.items():
            _fixture_write(repo, relative_path, content)

        first = build_manifest(
            repo,
            profile="fixed32",
            sequence="scripts/fixed32_seq.sh",
            spec_override=fixture_spec,
        )
        second = build_manifest(
            repo,
            profile="fixed32",
            sequence="scripts/fixed32_seq.sh",
            spec_override=fixture_spec,
        )
        if first != second:
            raise AssertionError("manifest is not deterministic")
        digest = first["overall_canonical_sha256"]
        payload = {
            key: value
            for key, value in first.items()
            if key != "overall_canonical_sha256"
        }
        if digest != hashlib.sha256(_canonical_bytes(payload)).hexdigest():
            raise AssertionError("overall canonical digest does not verify")
        rendered = _render_manifest(first)
        if b"do-not-emit-this-value" in rendered:
            raise AssertionError("manifest leaked a raw secret value")
        output = repo / "manifest.json"
        _write_output(str(output), rendered)
        if output.read_bytes() != rendered:
            raise AssertionError("atomic output content mismatch")
        forbidden = _forbidden_output_paths(
            repo, fixture_spec, "scripts/fixed32_seq.sh"
        )
        try:
            _write_output(
                str(repo / "config/runtime.json"),
                rendered,
                forbidden_paths=forbidden,
            )
        except ManifestError as error:
            if "reserved by the manifest" not in str(error):
                raise
        else:
            raise AssertionError("attested input was accepted as an output path")

        absent = repo / fixture_spec.required_absence[0]
        absent.parent.mkdir(parents=True, exist_ok=True)
        absent.write_text("unexpected\n", encoding="utf-8")
        try:
            build_manifest(
                repo,
                profile="fixed32",
                sequence="scripts/fixed32_seq.sh",
                spec_override=fixture_spec,
            )
        except ManifestError as error:
            if "unexpectedly exists" not in str(error):
                raise
        else:
            raise AssertionError("required-absence violation was accepted")
        absent.unlink()

        (repo / "config/runtime.json").unlink()
        try:
            build_manifest(
                repo,
                profile="fixed32",
                sequence="scripts/fixed32_seq.sh",
                spec_override=fixture_spec,
            )
        except ManifestError as error:
            if "missing required file" not in str(error):
                raise
        else:
            raise AssertionError("missing required file was accepted")
        _fixture_write(repo, "config/runtime.json", b'{"rows":32}\n')

        _fixture_write(repo, "src/fixture_pkg/extra.py", b"EXTRA = True\n")
        _fixture_write(
            repo,
            "src/fixture_pkg/helper.py",
            b"from .extra import EXTRA\nVALUE = 32 if EXTRA else 0\n",
        )
        try:
            build_manifest(
                repo,
                profile="fixed32",
                sequence="scripts/fixed32_seq.sh",
                spec_override=fixture_spec,
            )
        except ManifestError as error:
            if "unexpected local package dependency" not in str(error):
                raise
        else:
            raise AssertionError("unexpected package dependency was accepted")

    print("PASS fr13_runtime_manifest self-test")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository root (default: .)")
    parser.add_argument("--profile", choices=sorted(PROFILES))
    parser.add_argument(
        "--sequence",
        help="required repo-relative fixed32 campaign sequence",
    )
    parser.add_argument("--output", help="output JSON path, or - for stdout")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run isolated deterministic/fail-closed fixture tests",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.self_test:
        if (
            args.profile is not None
            or args.sequence is not None
            or args.output is not None
        ):
            parser.error("--self-test cannot be combined with manifest output options")
        self_test()
        return 0
    missing = [
        option
        for option, value in (
            ("--profile", args.profile),
            ("--sequence", args.sequence),
            ("--output", args.output),
        )
        if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    try:
        repo = Path(args.repo).expanduser().resolve(strict=True)
        spec = PROFILES[args.profile]
        manifest = build_manifest(
            repo,
            profile=args.profile,
            sequence=args.sequence,
        )
        _write_output(
            args.output,
            _render_manifest(manifest),
            forbidden_paths=_forbidden_output_paths(
                repo,
                spec,
                args.sequence,
            ),
        )
    except (ManifestError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
