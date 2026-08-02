from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr13_derive_qwen_agent_bundle_cap256 as derivation  # noqa: E402
import fr13_floor_gate as floor_gate  # noqa: E402
import fr13_runtime_manifest as runtime_manifest  # noqa: E402
import run_swe_bench_q36_a as runner  # noqa: E402


def _write_miniature_bundle(root: Path, chunk: bytes) -> None:
    package = root / derivation.PACKAGE_RELATIVE_PATH
    cap_chunk = root / derivation.CAP_CHUNK_RELATIVE_PATH
    package.parent.mkdir(parents=True)
    cap_chunk.parent.mkdir(parents=True)
    package.write_text('{"version":"0.19.4"}\n', encoding="ascii")
    cap_chunk.write_bytes(chunk)
    cap_chunk.chmod(0o644)


def _configure_miniature_contract(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    chunk: bytes,
) -> tuple[derivation.BundleScan, derivation.BundleScan]:
    derived_chunk = chunk.replace(
        derivation.SOURCE_NEEDLE,
        derivation.DERIVED_NEEDLE,
    )
    monkeypatch.setattr(derivation, "CAP_CHUNK_BYTES", len(chunk))
    monkeypatch.setattr(
        derivation,
        "SOURCE_CAP_CHUNK_SHA256",
        hashlib.sha256(chunk).hexdigest(),
    )
    monkeypatch.setattr(
        derivation,
        "DERIVED_CAP_CHUNK_SHA256",
        hashlib.sha256(derived_chunk).hexdigest(),
    )
    monkeypatch.setattr(
        derivation,
        "REQUIRED_ENTRYPOINTS",
        [
            derivation.PACKAGE_RELATIVE_PATH,
            derivation.CAP_CHUNK_RELATIVE_PATH,
        ],
    )
    source_scan = derivation.scan_bundle(source)
    monkeypatch.setattr(
        derivation,
        "SOURCE_EXPECTED",
        copy.deepcopy(source_scan.observation),
    )
    simulated = derivation._simulate_derived(source_scan)
    monkeypatch.setattr(
        derivation,
        "DERIVED_EXPECTED",
        copy.deepcopy(simulated.observation),
    )
    return source_scan, simulated


def test_fixed32_cap256_contract_matches_runner_and_floor_gate() -> None:
    expected = derivation.DERIVED_EXPECTED["bundle_tree"]

    assert derivation.SOURCE_CAP == 100
    assert derivation.DERIVED_CAP == 256
    assert derivation.SOURCE_MANIFEST_SHA256 == (
        "2643d1d64c03887654794d9bd00a88fb"
        "f9ced7362e034557cf196b8a37e744bc"
    )
    assert derivation.DERIVED_MANIFEST_SHA256 == (
        "594cac41e2d5ed505e0646f318b263ff"
        "70e200bcffe97326fe1c042fdc220516"
    )
    assert runner._FIXED32_QWEN_BUNDLE_TREE_EXPECTED == expected
    assert floor_gate.FIXED32_QWEN_BUNDLE_TREE == expected
    assert runner._FIXED32_QWEN_BUNDLE_REMOTE_PATH.endswith(
        derivation.DERIVED_MANIFEST_SHA256
    )
    assert derivation.CAP_CHUNK_RELATIVE_PATH in expected["entrypoints"]
    assert (
        "scripts/fr13_derive_qwen_agent_bundle_cap256.py"
        in runtime_manifest.FIXED32_HOST_SCRIPT_SOURCE
    )


def test_cap_patch_changes_only_the_three_digit_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        b"var TOOL_CALL_LOOP_THRESHOLD = 5;\n"
        + derivation.SOURCE_NEEDLE
        + b"\nvar STAGNATION_THRESHOLD = 8;\n"
    )
    expected = source.replace(
        derivation.SOURCE_NEEDLE,
        derivation.DERIVED_NEEDLE,
    )
    monkeypatch.setattr(derivation, "CAP_CHUNK_BYTES", len(source))
    monkeypatch.setattr(
        derivation,
        "SOURCE_CAP_CHUNK_SHA256",
        hashlib.sha256(source).hexdigest(),
    )
    monkeypatch.setattr(
        derivation,
        "DERIVED_CAP_CHUNK_SHA256",
        hashlib.sha256(expected).hexdigest(),
    )

    actual = derivation._derive_chunk(source)

    assert actual == expected
    assert b"TOOL_CALL_LOOP_THRESHOLD = 5" in actual
    assert b"STAGNATION_THRESHOLD = 8" in actual
    assert sum(before != after for before, after in zip(source, actual, strict=True)) == 3


def test_derivation_copies_and_publishes_exactly_one_manifest_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "derived"
    chunk = b"prefix\n" + derivation.SOURCE_NEEDLE + b"\nsuffix\n"
    _write_miniature_bundle(source, chunk)
    source_scan, simulated = _configure_miniature_contract(
        monkeypatch,
        source,
        chunk,
    )

    report = derivation.derive_bundle(
        source=source,
        output=output,
        dry_run=False,
    )

    assert report["status"] == "created"
    assert (source / derivation.CAP_CHUNK_RELATIVE_PATH).read_bytes() == chunk
    assert (output / derivation.CAP_CHUNK_RELATIVE_PATH).read_bytes() == (
        chunk.replace(derivation.SOURCE_NEEDLE, derivation.DERIVED_NEEDLE)
    )
    derived_scan = derivation.scan_bundle(output)
    assert derived_scan.manifest == simulated.manifest
    source_entries = {
        entry["path"]: entry for entry in source_scan.manifest["entries"]
    }
    derived_entries = {
        entry["path"]: entry for entry in derived_scan.manifest["entries"]
    }
    assert {
        path
        for path in source_entries
        if source_entries[path] != derived_entries[path]
    } == {derivation.CAP_CHUNK_RELATIVE_PATH}


def test_derivation_rejects_unrelated_source_tree_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "derived"
    chunk = b"prefix\n" + derivation.SOURCE_NEEDLE + b"\nsuffix\n"
    _write_miniature_bundle(source, chunk)
    _configure_miniature_contract(monkeypatch, source, chunk)
    (source / "unrelated.txt").write_text("drift\n", encoding="ascii")

    with pytest.raises(
        derivation.BundleDerivationError,
        match="source bundle differs from its canonical tree",
    ):
        derivation.derive_bundle(
            source=source,
            output=output,
            dry_run=False,
        )

    assert not output.exists()
