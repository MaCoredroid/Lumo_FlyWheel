from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "fr13_build_bf16_gemvx_k64_m4_shuffle_r64_u8.py"
SOURCE = REPO / "csrc" / "fr13_bf16_gemvx_k64_m4_shuffle_r64_u8.cu"


def _load_builder():
    spec = importlib.util.spec_from_file_location("fr13_m4_r64_u8_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assignment(source: str, name: str) -> object:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"missing {name}")


def test_builder_pins_source_and_deployed_toolchain() -> None:
    source = SCRIPT.read_text(encoding="ascii")
    assert _assignment(source, "SOURCE_SHA256") == (
        "a52361be1c9052a46509cc230ea320c4beb6d15f261327edc835d8da3ae00d9e"
    )
    assert _assignment(source, "EXPECTED_TORCH") == "2.11.0+cu130"
    assert _assignment(source, "EXPECTED_CUDA") == "13.0"
    assert _assignment(source, "EXPECTED_ARCH") == "12.1a"
    assert "fr13_bf16_k64_m4_r64_u8_sm121a" in source
    assert "gemvx_m4_shuffle_r64_u8_out" in source
    assert '"--frandom-seed=fr13_bf16_k64_m4_r64_u8"' in source
    assert "Path(\n        load(" not in source


def test_builder_resolves_the_library_registered_by_non_python_load(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    exact = build_dir / f"{builder.EXTENSION_NAME}.so"
    versioned = build_dir / f"{builder.EXTENSION_NAME}_v2.so"
    exact.touch()
    versioned.touch()

    assert builder.newly_loaded_library(build_dir, set(), {str(exact)}) == exact
    assert (
        builder.newly_loaded_library(
            build_dir, {str(exact)}, {str(exact), str(versioned)}
        )
        == versioned
    )
    with pytest.raises(RuntimeError):
        builder.newly_loaded_library(build_dir, set(), set())
    with pytest.raises(RuntimeError):
        builder.newly_loaded_library(build_dir, set(), {str(exact), str(versioned)})


def test_builder_copies_a_non_python_load_result_and_attests_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _load_builder()
    source = tmp_path / "candidate.cu"
    source.write_bytes(b"fixed32 B4 candidate")
    include = tmp_path / "include"
    include.mkdir()
    (include / "cusparse.h").touch()

    loaded: set[str] = set()
    operation = types.SimpleNamespace(gemvx_m4_shuffle_r64_u8_out=object())
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = builder.EXPECTED_TORCH
    fake_torch.ops = types.SimpleNamespace(
        loaded_libraries=loaded, fr13_bf16_k64_head=operation
    )
    fake_cpp_extension = types.ModuleType("torch.utils.cpp_extension")

    def fake_load(**kwargs: object) -> None:
        built = Path(str(kwargs["build_directory"])) / f"{kwargs['name']}.so"
        built.write_bytes(b"linked library")
        loaded.add(str(built.resolve()))

    fake_cpp_extension.load = fake_load
    fake_utils = types.ModuleType("torch.utils")
    fake_utils.cpp_extension = fake_cpp_extension
    fake_torch.utils = fake_utils
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", fake_cpp_extension)
    monkeypatch.setattr(builder, "REPO", tmp_path)
    monkeypatch.setattr(builder, "SOURCE", source)
    monkeypatch.setattr(
        builder, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(builder, "CUDA_PACKAGE_INCLUDE", include)
    monkeypatch.setattr(
        builder.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout="release 13.0"),
    )
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "restore-me")

    output = tmp_path / "out" / "candidate.so"
    attestation = tmp_path / "out" / "build.json"
    payload = builder.build(output, tmp_path / "build", attestation)

    assert output.read_bytes() == b"linked library"
    assert json.loads(attestation.read_text(encoding="ascii")) == payload
    assert payload["status"] == "BUILT_UNQUALIFIED"
    assert type(payload["gpu_runtime_used"]) is bool


def test_builder_attests_exact_b4_reused_weight_contract() -> None:
    source = SCRIPT.read_text(encoding="ascii")
    for contract in (
        '"batch_scope": "B4_exact"',
        '"grid": [1024, 1, 1]',
        '"block": [16, 64, 1]',
        '"input": "BF16[4,5120] contiguous"',
        '"weight": "BF16[65536,5120] contiguous"',
        '"output": "BF16[4,65536] contiguous"',
        '"independent_accumulators": 4',
        '"weight_reuse_batch": 4',
    ):
        assert contract in source
    assert SOURCE.name in source
    assert '"gpu_runtime_used": False' in source
    assert '"byte_equality_claim": False' in source
