#!/usr/bin/env python3
"""Issue, verify, and install the source-bound native key-group CFWD binary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lumo_flywheel_serving import (  # noqa: E402
    fr13_cfwd_native_fullvalue_cuda as candidate,
)

import fr13_patch_vllm_cfwd_native_fullvalue_cuda as patcher  # noqa: E402


ELF_MAGIC = b"\x7fELF"
OPERATOR_NEEDLE = b"fr13_fixed32_cfwd_native_fullvalue"
MAX_BINDING_BYTES = 64 * 1024
PATCHED_PATHS = (
    Path("CMakeLists.txt"),
    Path("csrc/ops.h"),
    Path("csrc/torch_bindings.cpp"),
    patcher.CUDA_DESTINATION,
)


class BinaryBindingError(RuntimeError):
    """The candidate binary cannot be tied to the exact source/build contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path, *, max_bytes: int | None = None) -> bytes:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise BinaryBindingError(f"not a regular non-symlink file: {path}")
    if metadata.st_nlink != 1:
        raise BinaryBindingError(f"file must have exactly one hard link: {path}")
    if metadata.st_size <= 0 or (
        max_bytes is not None and metadata.st_size > max_bytes
    ):
        raise BinaryBindingError(f"file size is outside the contract: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        observed = os.fstat(descriptor)
        if (
            observed.st_dev != metadata.st_dev
            or observed.st_ino != metadata.st_ino
            or observed.st_size != metadata.st_size
        ):
            raise BinaryBindingError(f"file changed while opening: {path}")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise BinaryBindingError(f"short read: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise BinaryBindingError(f"file grew while reading: {path}")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _git(source_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ("git", "-C", str(source_root), *arguments),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise BinaryBindingError(f"vLLM git validation failed: {detail}") from error


def _expected_patched_sources(source_root: Path, cuda_payload: bytes) -> dict[Path, bytes]:
    original: dict[Path, str] = {}
    for relative, expected_sha256 in patcher.PINNED_SHA256.items():
        payload = _git(
            source_root,
            "show",
            f"{candidate.VLLM_COMMIT}:{relative.as_posix()}",
        )
        if _sha256_bytes(payload) != expected_sha256:
            raise BinaryBindingError(f"pinned vLLM base bytes drift: {relative}")
        original[relative] = payload.decode("utf-8")
    return {
        Path("CMakeLists.txt"): patcher._replace_once(
            original[Path("CMakeLists.txt")],
            patcher.CMAKE_ANCHOR,
            patcher.CMAKE_REPLACEMENT,
            "CMake",
        ).encode("utf-8"),
        Path("csrc/ops.h"): patcher._replace_once(
            original[Path("csrc/ops.h")],
            patcher.OPS_ANCHOR,
            patcher.OPS_REPLACEMENT,
            "ops declaration",
        ).encode("utf-8"),
        Path("csrc/torch_bindings.cpp"): patcher._replace_once(
            original[Path("csrc/torch_bindings.cpp")],
            patcher.BINDINGS_ANCHOR,
            patcher.BINDINGS_REPLACEMENT,
            "torch binding",
        ).encode("utf-8"),
        patcher.CUDA_DESTINATION: cuda_payload,
    }


def verify_patched_source(
    source_root: Path, repo_root: Path
) -> dict[str, str]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise BinaryBindingError("vLLM source root must be a real directory")
    source_root = source_root.resolve(strict=True)
    repo_root = repo_root.resolve(strict=True)
    head = _git(source_root, "rev-parse", "HEAD").decode("ascii").strip()
    if head != candidate.VLLM_COMMIT:
        raise BinaryBindingError(f"vLLM HEAD drift: {head}")
    top = Path(
        _git(source_root, "rev-parse", "--show-toplevel")
        .decode("utf-8")
        .strip()
    ).resolve(strict=True)
    if top != source_root:
        raise BinaryBindingError("vLLM source root is not its git toplevel")

    cuda_path = repo_root / candidate.CUDA_SOURCE_PATH
    patcher_path = repo_root / candidate.PATCHER_SOURCE_PATH
    cuda_payload = _read_regular(cuda_path)
    patcher_payload = _read_regular(patcher_path)
    if _sha256_bytes(cuda_payload) != candidate.CUDA_SOURCE_SHA256:
        raise BinaryBindingError("candidate CUDA source SHA-256 drift")
    if _sha256_bytes(patcher_payload) != candidate.PATCHER_SOURCE_SHA256:
        raise BinaryBindingError("candidate patcher source SHA-256 drift")

    expected = _expected_patched_sources(source_root, cuda_payload)
    observed_hashes: dict[str, str] = {}
    for relative, expected_payload in expected.items():
        observed = _read_regular(source_root / relative)
        if observed != expected_payload:
            raise BinaryBindingError(f"patched vLLM source bytes drift: {relative}")
        observed_hashes[relative.as_posix()] = _sha256_bytes(observed)

    dirty = _git(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).decode("utf-8")
    expected_status = {
        " M CMakeLists.txt",
        " M csrc/ops.h",
        " M csrc/torch_bindings.cpp",
        f"?? {patcher.CUDA_DESTINATION.as_posix()}",
    }
    if set(dirty.splitlines()) != expected_status:
        raise BinaryBindingError("patched vLLM git status does not match the contract")
    return observed_hashes


def _build_output_path(build_dir: Path, token: str) -> Path:
    decoded = token.replace("$ ", " ").replace("$:", ":")
    path = Path(decoded)
    resolved = (path if path.is_absolute() else build_dir / path).resolve(
        strict=False
    )
    try:
        resolved.relative_to(build_dir)
    except ValueError as error:
        raise BinaryBindingError("build.ninja output escapes the build directory") from error
    return resolved


def verify_build_graph(
    build_dir: Path, source_root: Path, binary_path: Path
) -> dict[str, object]:
    if build_dir.is_symlink() or not build_dir.is_dir():
        raise BinaryBindingError("build directory must be a real directory")
    build_dir = build_dir.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    binary_path = binary_path.resolve(strict=True)
    try:
        binary_path.relative_to(build_dir)
    except ValueError as error:
        raise BinaryBindingError(
            "candidate _C.abi3.so is outside its build directory"
        ) from error
    cache_path = build_dir / "CMakeCache.txt"
    ninja_path = build_dir / "build.ninja"
    cache = _read_regular(cache_path, max_bytes=16 * 1024 * 1024)
    ninja = _read_regular(ninja_path, max_bytes=256 * 1024 * 1024)
    source_declaration = f"CMAKE_HOME_DIRECTORY:INTERNAL={source_root}\n".encode()
    if source_declaration not in cache:
        raise BinaryBindingError("CMake build is not tied to the pinned source root")
    try:
        ninja_text = ninja.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BinaryBindingError("build.ninja is not UTF-8") from error
    logical_lines: list[str] = []
    pending = ""
    for physical in ninja_text.splitlines():
        stripped = physical.rstrip()
        if stripped.endswith("$"):
            pending += stripped[:-1]
            continue
        logical_lines.append(pending + physical.lstrip() if pending else physical)
        pending = ""
    if pending:
        raise BinaryBindingError("build.ninja ends inside a continuation")

    dependencies: dict[str, tuple[str, ...]] = {}
    for line in logical_lines:
        if not line.startswith("build "):
            continue
        declaration = line.removeprefix("build ")
        outputs_text, separator, rule_and_inputs = declaration.partition(": ")
        if not separator:
            raise BinaryBindingError("malformed build.ninja build edge")
        outputs = tuple(outputs_text.split())
        tokens = rule_and_inputs.split()
        if not outputs or not tokens:
            raise BinaryBindingError("empty build.ninja build edge")
        inputs = tuple(token for token in tokens[1:] if token not in {"|", "||"})
        for output in outputs:
            if output in dependencies:
                raise BinaryBindingError(f"duplicate build.ninja output: {output}")
            dependencies[output] = inputs

    full_targets = tuple(
        output
        for output in dependencies
        if Path(output.replace("$ ", " ")).name == "_C.abi3.so"
    )
    if not full_targets:
        raise BinaryBindingError("full vLLM _C.abi3.so target is absent from build.ninja")
    matched_full_targets = tuple(
        target
        for target in full_targets
        if _build_output_path(build_dir, target) == binary_path
    )
    if not matched_full_targets:
        raise BinaryBindingError(
            "candidate binary is not the full _C.abi3.so build output"
        )
    reachable = set(matched_full_targets)
    frontier = list(matched_full_targets)
    while frontier:
        output = frontier.pop()
        for dependency in dependencies.get(output, ()):
            if dependency not in reachable:
                reachable.add(dependency)
                frontier.append(dependency)
    cuda_suffix = patcher.CUDA_DESTINATION.as_posix()
    cuda_absolute = str(source_root / patcher.CUDA_DESTINATION)
    def is_candidate_source(dependency: str) -> bool:
        decoded = dependency.replace("$ ", " ").replace("$:", ":")
        return decoded in {cuda_suffix, cuda_absolute} or decoded.endswith(
            f"/{cuda_suffix}"
        )

    candidate_object_outputs = tuple(
        sorted(
            output
            for output in reachable
            if output in dependencies
            and _build_output_path(build_dir, output).suffix == ".o"
            and any(
                is_candidate_source(dependency)
                for dependency in dependencies.get(output, ())
            )
        )
    )
    if len(candidate_object_outputs) != 1:
        raise BinaryBindingError(
            "full vLLM _C.abi3.so target does not reach exactly one candidate object"
        )
    candidate_object_path = _build_output_path(
        build_dir, candidate_object_outputs[0]
    )
    candidate_object_output = candidate_object_path.relative_to(
        build_dir
    ).as_posix()
    return {
        "generator": "ninja",
        "candidate_source_in_build_graph": True,
        "candidate_object_outputs": [candidate_object_output],
        "full_vllm_extension_target": "_C.abi3.so",
        "cmake_cache_sha256": _sha256_bytes(cache),
        "build_ninja_sha256": _sha256_bytes(ninja),
    }


def verify_binary(path: Path) -> dict[str, object]:
    payload = _read_regular(path)
    if not payload.startswith(ELF_MAGIC):
        raise BinaryBindingError("candidate _C.abi3.so is not ELF")
    if OPERATOR_NEEDLE not in payload:
        raise BinaryBindingError("candidate operator is absent from _C.abi3.so")
    return {"sha256": _sha256_bytes(payload), "bytes": len(payload)}


def _force_rebuild_candidate_target(
    *,
    build_dir: Path,
    source_root: Path,
    binary_path: Path,
    candidate_object_outputs: tuple[str, ...],
) -> dict[str, object]:
    build_dir = build_dir.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    binary_path = binary_path.resolve(strict=True)
    try:
        target = binary_path.relative_to(build_dir).as_posix()
    except ValueError as error:
        raise BinaryBindingError(
            "candidate _C.abi3.so is outside its build directory"
        ) from error
    if binary_path.name != "_C.abi3.so":
        raise BinaryBindingError("candidate binary is not the full _C.abi3.so")
    if len(candidate_object_outputs) != 1:
        raise BinaryBindingError("forced rebuild requires exactly one candidate object")
    candidate_object_path = _build_output_path(
        build_dir, candidate_object_outputs[0]
    )

    cache = _read_regular(
        build_dir / "CMakeCache.txt", max_bytes=16 * 1024 * 1024
    ).decode("utf-8")
    cmake_commands = {
        value
        for line in cache.splitlines()
        if line.startswith("CMAKE_COMMAND:")
        for _, separator, value in (line.partition("="),)
        if separator
    }
    if len(cmake_commands) != 1:
        raise BinaryBindingError("CMakeCache does not bind one CMake command")
    cmake_path = Path(cmake_commands.pop())
    if not cmake_path.is_absolute():
        raise BinaryBindingError("CMake command path is not absolute")
    try:
        cmake_path = cmake_path.resolve(strict=True)
    except OSError as error:
        raise BinaryBindingError("CMake command is unavailable") from error
    if not cmake_path.is_file() or not os.access(cmake_path, os.X_OK):
        raise BinaryBindingError("CMake command is not executable")

    source_path = source_root / patcher.CUDA_DESTINATION
    source_before = source_path.lstat()
    binary_before = binary_path.lstat()
    if (
        source_path.is_symlink()
        or not stat.S_ISREG(source_before.st_mode)
        or source_before.st_nlink != 1
        or binary_path.is_symlink()
        or not stat.S_ISREG(binary_before.st_mode)
        or binary_before.st_nlink != 1
    ):
        raise BinaryBindingError("forced rebuild inputs are not regular files")
    touched_ns = max(
        time.time_ns(), source_before.st_mtime_ns + 1, binary_before.st_mtime_ns + 1
    )
    os.utime(source_path, ns=(source_before.st_atime_ns, touched_ns), follow_symlinks=False)
    source_touched = source_path.lstat()
    if source_touched.st_mtime_ns <= binary_before.st_mtime_ns:
        raise BinaryBindingError("candidate source could not be made newer than binary")

    command = (
        str(cmake_path),
        "--build",
        str(build_dir),
        "--target",
        target,
    )
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace")[-4096:].strip()
        raise BinaryBindingError(f"forced candidate rebuild failed: {detail}") from error
    binary_after = binary_path.lstat()
    candidate_object = candidate_object_path.lstat()
    if (
        binary_path.is_symlink()
        or not stat.S_ISREG(binary_after.st_mode)
        or binary_after.st_nlink != 1
        or candidate_object_path.is_symlink()
        or not stat.S_ISREG(candidate_object.st_mode)
        or candidate_object.st_nlink != 1
        or candidate_object.st_mtime_ns < source_touched.st_mtime_ns
        or binary_after.st_mtime_ns < candidate_object.st_mtime_ns
        or binary_after.st_mtime_ns < source_touched.st_mtime_ns
    ):
        raise BinaryBindingError(
            "full _C.abi3.so was not relinked after the forced source rebuild"
        )
    return {
        "candidate_source_forced_rebuild": True,
        "candidate_source_mtime_ns": source_touched.st_mtime_ns,
        "candidate_objects": [
            {
                "path": candidate_object_outputs[0],
                "sha256": _sha256_bytes(_read_regular(candidate_object_path)),
                "bytes": candidate_object.st_size,
                "mtime_ns": candidate_object.st_mtime_ns,
            }
        ],
        "full_extension_mtime_ns": binary_after.st_mtime_ns,
    }


def _strict_binding_payload(path: Path) -> Mapping[str, object]:
    if stat.S_IMODE(path.lstat().st_mode) != 0o400:
        raise BinaryBindingError("binding mode must be exactly 0400")
    payload = _read_regular(path, max_bytes=MAX_BINDING_BYTES)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise BinaryBindingError(f"duplicate binding key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise BinaryBindingError(f"non-finite binding constant: {value}")

    try:
        parsed = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BinaryBindingError("binding is not ASCII JSON") from error
    if not isinstance(parsed, Mapping):
        raise BinaryBindingError("binding is not a JSON object")
    return parsed


def verify_binding(
    binding_path: Path, binary_path: Path, repo_root: Path
) -> dict[str, object]:
    binding = candidate.validate_binary_binding(
        _strict_binding_payload(binding_path)
    )
    binary = verify_binary(binary_path)
    if binding["binary"] != binary:
        raise BinaryBindingError("candidate binary differs from its binding")
    repo_root = repo_root.resolve(strict=True)
    source_pairs = {
        candidate.CUDA_SOURCE_PATH: candidate.CUDA_SOURCE_SHA256,
        candidate.PATCHER_SOURCE_PATH: candidate.PATCHER_SOURCE_SHA256,
    }
    for relative, expected in source_pairs.items():
        if _sha256_bytes(_read_regular(repo_root / relative)) != expected:
            raise BinaryBindingError(f"bound repository source drift: {relative}")
    return dict(binding)


def _atomic_write(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def issue_binding(
    *,
    repo_root: Path,
    source_root: Path,
    build_dir: Path,
    binary_path: Path,
    output: Path,
) -> dict[str, object]:
    patched_vllm = verify_patched_source(source_root, repo_root)
    build_before = verify_build_graph(build_dir, source_root, binary_path)
    rebuild = _force_rebuild_candidate_target(
        build_dir=build_dir,
        source_root=source_root,
        binary_path=binary_path,
        candidate_object_outputs=tuple(build_before["candidate_object_outputs"]),
    )
    if verify_patched_source(source_root, repo_root) != patched_vllm:
        raise BinaryBindingError("patched vLLM source changed during forced rebuild")
    build = verify_build_graph(build_dir, source_root, binary_path)
    if build != build_before:
        raise BinaryBindingError("CMake/Ninja graph changed during forced rebuild")
    build.update(rebuild)
    binary = verify_binary(binary_path)
    payload: dict[str, object] = {
        "schema": candidate.BINARY_BINDING_SCHEMA,
        "candidate": candidate.CANDIDATE,
        "vllm_base_commit": candidate.VLLM_COMMIT,
        "operator": candidate.OPERATOR,
        "architecture": "sm_121a",
        "source_sha256": {
            candidate.CUDA_SOURCE_PATH: candidate.CUDA_SOURCE_SHA256,
            candidate.PATCHER_SOURCE_PATH: candidate.PATCHER_SOURCE_SHA256,
        },
        "patched_vllm_sha256": patched_vllm,
        "build": build,
        "binary": binary,
        "default_on": False,
        "production_authorized": False,
        "timing_eligible": False,
    }
    encoded = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    _atomic_write(output, encoded, 0o400)
    verify_binding(output, binary_path, repo_root)
    return payload


def install_candidate(
    *,
    repo_root: Path,
    binary_path: Path,
    binding_path: Path,
    destination: Path,
    binding_destination: Path,
    arm_path: Path,
) -> dict[str, object]:
    binding = verify_binding(binding_path, binary_path, repo_root)
    destination_info = destination.lstat()
    if destination.is_symlink() or not stat.S_ISREG(destination_info.st_mode):
        raise BinaryBindingError("installed vLLM _C destination must already be regular")
    _atomic_write(destination, _read_regular(binary_path), 0o555)
    installed = verify_binary(destination)
    if installed != binding["binary"]:
        raise BinaryBindingError("installed candidate binary identity drift")
    binding_payload = _read_regular(binding_path, max_bytes=MAX_BINDING_BYTES)
    _atomic_write(binding_destination, binding_payload, 0o400)
    candidate.load_binary_binding(binding_destination)
    _atomic_write(arm_path, b"diagnostic\n", 0o400)
    return {
        "schema": "fr13.fixed32.cfwd_native_keygroup_install.v1",
        "candidate": candidate.CANDIDATE,
        "destination": "vllm/_C.abi3.so",
        "binary": installed,
        "binding_sha256": _sha256_bytes(binding_payload),
        "default_on": False,
        "production_authorized": False,
        "timing_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    issue = commands.add_parser("issue")
    issue.add_argument("--repo", type=Path, required=True)
    issue.add_argument("--source-root", type=Path, required=True)
    issue.add_argument("--build-dir", type=Path, required=True)
    issue.add_argument("--candidate-so", type=Path, required=True)
    issue.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--repo", type=Path, required=True)
    verify.add_argument("--candidate-so", type=Path, required=True)
    verify.add_argument("--binding", type=Path, required=True)

    install = commands.add_parser("install")
    install.add_argument("--repo", type=Path, required=True)
    install.add_argument("--candidate-so", type=Path, required=True)
    install.add_argument("--binding", type=Path, required=True)
    install.add_argument("--destination", type=Path, required=True)
    install.add_argument("--binding-destination", type=Path, required=True)
    install.add_argument("--arm", type=Path, required=True)

    arguments = parser.parse_args()
    if arguments.command == "issue":
        payload = issue_binding(
            repo_root=arguments.repo,
            source_root=arguments.source_root,
            build_dir=arguments.build_dir,
            binary_path=arguments.candidate_so,
            output=arguments.output,
        )
    elif arguments.command == "verify":
        payload = verify_binding(
            arguments.binding, arguments.candidate_so, arguments.repo
        )
    else:
        payload = install_candidate(
            repo_root=arguments.repo,
            binary_path=arguments.candidate_so,
            binding_path=arguments.binding,
            destination=arguments.destination,
            binding_destination=arguments.binding_destination,
            arm_path=arguments.arm,
        )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
