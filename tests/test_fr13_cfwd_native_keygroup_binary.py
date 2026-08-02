from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import fr13_cfwd_native_keygroup_binary as binary_gate  # noqa: E402
from lumo_flywheel_serving import (  # noqa: E402
    fr13_cfwd_native_fullvalue_cuda as candidate,
)


def _fake_binary(path: Path, suffix: bytes = b"") -> None:
    path.write_bytes(
        binary_gate.ELF_MAGIC
        + b"\x00" * 32
        + binary_gate.OPERATOR_NEEDLE
        + suffix
    )


def _binding(binary_path: Path) -> dict[str, object]:
    identity = binary_gate.verify_binary(binary_path)
    return {
        "schema": candidate.BINARY_BINDING_SCHEMA,
        "candidate": candidate.CANDIDATE,
        "vllm_base_commit": candidate.VLLM_COMMIT,
        "operator": candidate.OPERATOR,
        "architecture": "sm_121a",
        "source_sha256": {
            candidate.CUDA_SOURCE_PATH: candidate.CUDA_SOURCE_SHA256,
            candidate.PATCHER_SOURCE_PATH: candidate.PATCHER_SOURCE_SHA256,
        },
        "patched_vllm_sha256": candidate.PATCHED_VLLM_SHA256,
        "build": {
            "generator": "ninja",
            "candidate_source_in_build_graph": True,
            "candidate_source_forced_rebuild": True,
            "candidate_source_mtime_ns": 1,
            "full_vllm_extension_target": "_C.abi3.so",
            "full_extension_mtime_ns": 2,
            "cmake_cache_sha256": "b" * 64,
            "build_ninja_sha256": "c" * 64,
        },
        "binary": identity,
        "default_on": False,
        "production_authorized": False,
        "timing_eligible": False,
    }


def _write_binding(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    path.chmod(0o400)


def test_issue_binds_exact_patched_source_build_graph_and_full_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_so = tmp_path / "_C.abi3.so"
    output = tmp_path / "binding.json"
    _fake_binary(candidate_so)
    monkeypatch.setattr(
        binary_gate,
        "verify_patched_source",
        lambda _source, _repo: dict(candidate.PATCHED_VLLM_SHA256),
    )
    monkeypatch.setattr(
        binary_gate,
        "verify_build_graph",
        lambda _build, _source, _binary: {
            "generator": "ninja",
            "candidate_source_in_build_graph": True,
            "full_vllm_extension_target": "_C.abi3.so",
            "cmake_cache_sha256": "b" * 64,
            "build_ninja_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        binary_gate,
        "_force_rebuild_candidate_target",
        lambda **_kwargs: {
            "candidate_source_forced_rebuild": True,
            "candidate_source_mtime_ns": 1,
            "full_extension_mtime_ns": 2,
        },
    )

    issued = binary_gate.issue_binding(
        repo_root=ROOT,
        source_root=tmp_path / "source",
        build_dir=tmp_path / "build",
        binary_path=candidate_so,
        output=output,
    )

    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    assert issued["patched_vllm_sha256"] == candidate.PATCHED_VLLM_SHA256
    assert issued["build"]["full_vllm_extension_target"] == "_C.abi3.so"
    assert issued["build"]["candidate_source_forced_rebuild"] is True
    assert issued["binary"] == binary_gate.verify_binary(candidate_so)
    assert binary_gate.verify_binding(output, candidate_so, ROOT)[
        "timing_eligible"
    ] is False


def test_binding_verification_rejects_binary_or_repository_source_drift(
    tmp_path: Path,
) -> None:
    candidate_so = tmp_path / "_C.abi3.so"
    binding_path = tmp_path / "binding.json"
    _fake_binary(candidate_so)
    _write_binding(binding_path, _binding(candidate_so))
    binary_gate.verify_binding(binding_path, candidate_so, ROOT)

    _fake_binary(candidate_so, b"drift")
    with pytest.raises(binary_gate.BinaryBindingError, match="differs"):
        binary_gate.verify_binding(binding_path, candidate_so, ROOT)


def test_install_replaces_full_vllm_extension_and_materializes_private_gate(
    tmp_path: Path,
) -> None:
    candidate_so = tmp_path / "candidate._C.abi3.so"
    binding_path = tmp_path / "binding.json"
    destination = tmp_path / "site-packages/vllm/_C.abi3.so"
    binding_destination = tmp_path / "logs/binding.json"
    arm_path = tmp_path / "logs/arm"
    _fake_binary(candidate_so)
    _write_binding(binding_path, _binding(candidate_so))
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stock")

    installed = binary_gate.install_candidate(
        repo_root=ROOT,
        binary_path=candidate_so,
        binding_path=binding_path,
        destination=destination,
        binding_destination=binding_destination,
        arm_path=arm_path,
    )

    assert installed["destination"] == "vllm/_C.abi3.so"
    assert destination.read_bytes() == candidate_so.read_bytes()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert stat.S_IMODE(binding_destination.stat().st_mode) == 0o400
    assert stat.S_IMODE(arm_path.stat().st_mode) == 0o400
    assert arm_path.read_text(encoding="ascii") == "diagnostic\n"


def test_binary_verification_rejects_non_elf_or_missing_operator(
    tmp_path: Path,
) -> None:
    binary_path = tmp_path / "candidate.so"
    binary_path.write_bytes(b"not-elf")
    with pytest.raises(binary_gate.BinaryBindingError, match="not ELF"):
        binary_gate.verify_binary(binary_path)

    binary_path.write_bytes(binary_gate.ELF_MAGIC + b"without-op")
    with pytest.raises(binary_gate.BinaryBindingError, match="operator is absent"):
        binary_gate.verify_binary(binary_path)


def test_build_graph_requires_candidate_in_full_extension_target(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "vllm"
    build_dir = tmp_path / "build"
    source_root.mkdir()
    build_dir.mkdir()
    candidate_so = build_dir / "vllm/_C.abi3.so"
    candidate_so.parent.mkdir()
    candidate_so.write_bytes(b"fixture")
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={source_root.resolve()}\n",
        encoding="utf-8",
    )
    (build_dir / "build.ninja").write_text(
        "build vllm/_C.abi3.so: link "
        f"{source_root.resolve() / binary_gate.patcher.CUDA_DESTINATION}\n",
        encoding="utf-8",
    )
    result = binary_gate.verify_build_graph(
        build_dir, source_root, candidate_so
    )
    assert result["candidate_source_in_build_graph"] is True
    assert result["full_vllm_extension_target"] == "_C.abi3.so"

    (build_dir / "build.ninja").write_text(
        "build vllm/_C_stable_libtorch.abi3.so: link other.cu\n",
        encoding="utf-8",
    )
    with pytest.raises(binary_gate.BinaryBindingError, match="_C.abi3.so target"):
        binary_gate.verify_build_graph(build_dir, source_root, candidate_so)


def test_build_graph_rejects_candidate_reachable_only_from_stable_extension(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "vllm"
    build_dir = tmp_path / "build"
    source_root.mkdir()
    build_dir.mkdir()
    candidate_so = build_dir / "vllm/_C.abi3.so"
    candidate_so.parent.mkdir()
    candidate_so.write_bytes(b"fixture")
    candidate_source = source_root / binary_gate.patcher.CUDA_DESTINATION
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={source_root.resolve()}\n",
        encoding="utf-8",
    )
    (build_dir / "build.ninja").write_text(
        "build candidate.o: cuda "
        f"{candidate_source}\n"
        "build other.o: cuda other.cu\n"
        "build vllm/_C_stable_libtorch.abi3.so: link candidate.o\n"
        "build vllm/_C.abi3.so: link other.o\n",
        encoding="utf-8",
    )

    with pytest.raises(
        binary_gate.BinaryBindingError,
        match="full vLLM .* does not reach candidate CUDA source",
    ):
        binary_gate.verify_build_graph(build_dir, source_root, candidate_so)


def test_build_graph_rejects_arbitrary_binary_outside_canonical_output(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "vllm"
    build_dir = tmp_path / "build"
    source_root.mkdir()
    build_dir.mkdir()
    canonical_so = build_dir / "vllm/_C.abi3.so"
    canonical_so.parent.mkdir()
    canonical_so.write_bytes(b"canonical")
    arbitrary_so = tmp_path / "arbitrary._C.abi3.so"
    arbitrary_so.write_bytes(b"arbitrary")
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={source_root.resolve()}\n",
        encoding="utf-8",
    )
    (build_dir / "build.ninja").write_text(
        "build vllm/_C.abi3.so: link "
        f"{source_root / binary_gate.patcher.CUDA_DESTINATION}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        binary_gate.BinaryBindingError,
        match="outside its build directory",
    ):
        binary_gate.verify_build_graph(
            build_dir, source_root, arbitrary_so
        )


@pytest.mark.parametrize("relink", (True, False))
def test_forced_rebuild_requires_full_extension_newer_than_touched_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relink: bool,
) -> None:
    source_root = tmp_path / "vllm"
    build_dir = tmp_path / "build"
    source_path = source_root / binary_gate.patcher.CUDA_DESTINATION
    binary_path = build_dir / "vllm/_C.abi3.so"
    source_path.parent.mkdir(parents=True)
    binary_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"source")
    binary_path.write_bytes(b"binary")
    cmake = Path("/bin/true").resolve(strict=True)
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_COMMAND:INTERNAL={cmake}\n",
        encoding="utf-8",
    )

    def run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess:
        assert command == (
            str(cmake),
            "--build",
            str(build_dir.resolve()),
            "--target",
            "vllm/_C.abi3.so",
        )
        assert kwargs["check"] is True
        if relink:
            source_mtime_ns = source_path.stat().st_mtime_ns
            os.utime(
                binary_path,
                ns=(binary_path.stat().st_atime_ns, source_mtime_ns + 1),
            )
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(binary_gate.subprocess, "run", run)
    if relink:
        report = binary_gate._force_rebuild_candidate_target(
            build_dir=build_dir,
            source_root=source_root,
            binary_path=binary_path,
        )
        assert report["candidate_source_forced_rebuild"] is True
        assert report["full_extension_mtime_ns"] >= report[
            "candidate_source_mtime_ns"
        ]
    else:
        with pytest.raises(
            binary_gate.BinaryBindingError,
            match="was not relinked",
        ):
            binary_gate._force_rebuild_candidate_target(
                build_dir=build_dir,
                source_root=source_root,
                binary_path=binary_path,
            )
