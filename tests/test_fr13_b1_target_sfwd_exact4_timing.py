from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
RUNNER = SCRIPTS / "fr13_run_b1_target_sfwd_exact4_timing.sh"
SHARED_ENV = SCRIPTS / "fr13_fixed32_sfwd_fusion_env.sh"
LAUNCHER = SCRIPTS / "fr13_launch_forked_fa2_tree_server.sh"
MANIFEST = SCRIPTS / "fr13_runtime_manifest.py"
HISTORICAL_PASS = (
    "results/fr13_b1_m128_cooperative_target_sfwd_real_gate_"
    "a8a904ed6_20260805/target_combined_pass.json"
)


def _load_pass_module(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        name, SCRIPTS / "fr13_cutlass_streamk_pass.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _historical_pass() -> dict[str, object]:
    raw = subprocess.check_output(
        ["git", "show", f"HEAD:{HISTORICAL_PASS}"], cwd=REPO
    )
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


def test_historical_cooperative_target_source_is_pinned_to_git_blobs() -> None:
    module = _load_pass_module("fr13_exact4_historical_source_test")
    payload = _historical_pass()

    identity = module.validate_historical_qualification_source_binding(
        payload["source_commit"],
        payload["source_identity"],
        SCRIPTS / "fr13_patch_cutlass_fixed32_wave.py",
        payload["candidate"],
    )

    assert payload["source_commit"] == module.HISTORICAL_QUALIFICATION_SOURCE_COMMIT
    assert payload["candidate"] == module.HISTORICAL_QUALIFICATION_SELECTOR
    assert (
        module._canonical_identity_sha256(identity)
        == module.HISTORICAL_QUALIFICATION_SOURCE_IDENTITY_SHA256
    )
    assert identity["files"][str(module.PATCH_SOURCE)]["sha256"] == (
        module.SOURCE_CONTRACTS[module.HISTORICAL_QUALIFICATION_SELECTOR][
            "patch_source_sha256"
        ]
    )


def test_historical_cooperative_target_rejects_identity_or_selector_drift() -> None:
    module = _load_pass_module("fr13_exact4_historical_drift_test")
    payload = _historical_pass()
    identity = json.loads(json.dumps(payload["source_identity"]))
    first = next(iter(identity["files"].values()))
    first["sha256"] = "0" * 64

    with pytest.raises(module.QualificationError, match="source identity mismatch"):
        module.validate_historical_qualification_source_binding(
            payload["source_commit"],
            identity,
            SCRIPTS / "fr13_patch_cutlass_fixed32_wave.py",
            payload["candidate"],
        )
    with pytest.raises(module.QualificationError, match="restricted"):
        module.validate_historical_qualification_source_binding(
            payload["source_commit"],
            payload["source_identity"],
            SCRIPTS / "fr13_patch_cutlass_fixed32_wave.py",
            "identity_onen_n5120_fullgrid_b1",
        )


def _break_git(module, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "_git_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module._GitUnavailableError("git is unavailable for source binding")
        ),
    )


def _mounted_runtime_identity(module, source_commit: str) -> dict[str, object]:
    files: dict[str, object] = {}
    for relative in module.RUNTIME_SOURCE_BINDING_PATHS:
        working = REPO / relative
        files[relative] = {
            "bytes": working.stat().st_size,
            "sha256": module.sha256_file(working),
        }
    return {
        "schema": module.RUNTIME_SOURCE_BINDING_SCHEMA,
        "source_commit": source_commit,
        "files": files,
    }


def test_historical_binding_survives_a_container_without_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_pass_module("fr13_exact4_historical_no_git_test")
    payload = _historical_pass()
    patch_source = SCRIPTS / "fr13_patch_cutlass_fixed32_wave.py"
    expected = module.validate_historical_qualification_source_binding(
        payload["source_commit"],
        payload["source_identity"],
        patch_source,
        payload["candidate"],
    )
    _break_git(module, monkeypatch)

    # The strict entry point still fails closed without git ...
    with pytest.raises(
        module.QualificationError, match="git is unavailable for source binding"
    ):
        module.validate_historical_qualification_source_binding(
            payload["source_commit"],
            payload["source_identity"],
            patch_source,
            payload["candidate"],
        )
    # ... while in-container verification falls back to the pinned identity.
    assert (
        module._validate_historical_qualification_binding(
            payload["source_commit"],
            payload["source_identity"],
            patch_source,
            payload["candidate"],
        )
        == expected
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("digest", "source identity mismatch"),
        ("selector", "restricted"),
        ("commit", "restricted"),
    ),
)
def test_no_git_historical_binding_rejects_pinned_identity_drift(
    monkeypatch: pytest.MonkeyPatch, mutation: str, match: str
) -> None:
    module = _load_pass_module(f"fr13_exact4_historical_no_git_{mutation}_test")
    payload = _historical_pass()
    patch_source = SCRIPTS / "fr13_patch_cutlass_fixed32_wave.py"
    source_commit = payload["source_commit"]
    identity = payload["source_identity"]
    candidate = payload["candidate"]
    if mutation == "digest":
        identity = json.loads(json.dumps(identity))
        next(iter(identity["files"].values()))["sha256"] = "0" * 64
    elif mutation == "selector":
        candidate = "identity_onen_n5120_fullgrid_b1"
    else:
        source_commit = "d" * 40
    _break_git(module, monkeypatch)

    with pytest.raises(module.QualificationError, match=match):
        module._validate_historical_qualification_binding(
            source_commit, identity, patch_source, candidate
        )


def test_no_git_runtime_identity_binding_accepts_the_mounted_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_pass_module("fr13_exact4_runtime_no_git_test")
    patch_source = SCRIPTS / "fr13_patch_cutlass_fixed32_wave.py"
    identity = _mounted_runtime_identity(module, "b" * 40)
    _break_git(module, monkeypatch)

    with pytest.raises(
        module.QualificationError, match="git is unavailable for source binding"
    ):
        module.validate_runtime_source_commit_identity("b" * 40, patch_source)
    assert (
        module._validate_runtime_source_identity_binding(
            "b" * 40, identity, patch_source
        )
        == identity
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("commit", "runtime source-binding commit is invalid"),
        ("schema", "runtime source identity schema mismatch"),
        ("bound_commit", "runtime source identity commit mismatch"),
        ("manifest", "runtime source identity file manifest mismatch"),
        ("record", "runtime source identity record is invalid"),
        ("bytes", "runtime source-binding file size mismatch"),
        ("sha256", "runtime source-binding file SHA-256 mismatch"),
    ),
)
def test_no_git_runtime_identity_binding_rejects_mounted_drift(
    monkeypatch: pytest.MonkeyPatch, mutation: str, match: str
) -> None:
    module = _load_pass_module(f"fr13_exact4_runtime_no_git_{mutation}_test")
    patch_source = SCRIPTS / "fr13_patch_cutlass_fixed32_wave.py"
    source_commit = "b" * 40
    identity = _mounted_runtime_identity(module, source_commit)
    last = module.RUNTIME_SOURCE_BINDING_PATHS[-1]
    if mutation == "commit":
        source_commit = "not-a-commit"
        identity["source_commit"] = source_commit
    elif mutation == "schema":
        identity["schema"] = module.SOURCE_BINDING_SCHEMA
    elif mutation == "bound_commit":
        identity["source_commit"] = "c" * 40
    elif mutation == "manifest":
        del identity["files"][last]
    elif mutation == "record":
        identity["files"][last] = {"sha256": "0" * 64}
    elif mutation == "bytes":
        identity["files"][last]["bytes"] += 1
    else:
        identity["files"][last]["sha256"] = "0" * 64
    _break_git(module, monkeypatch)

    with pytest.raises(module.QualificationError, match=match):
        module._validate_runtime_source_identity_binding(
            source_commit, identity, patch_source
        )


def test_dual_identity_fields_are_all_or_nothing() -> None:
    module = _load_pass_module("fr13_exact4_dual_identity_fields_test")

    assert module._historical_qualification_requested({}) is False
    with pytest.raises(module.QualificationError, match="incomplete"):
        module._historical_qualification_requested(
            {"qualification_source_mode": module.HISTORICAL_QUALIFICATION_MODE}
        )
    with pytest.raises(module.QualificationError, match="mode mismatch"):
        module._historical_qualification_requested(
            {
                "qualification_source_mode": "current",
                "runtime_source_commit": "a" * 40,
                "runtime_source_identity": {},
            }
        )


def test_exact4_runner_pins_workload_runtime_and_real_credentials() -> None:
    # The candidate serve environment lives in the file the runner sources, so
    # that scripts/fr13_run_b1_sfwd_fusion_boot_diag.sh cannot boot a different
    # shape than the arm it screens. The runner's effective definition is the
    # pair of files; pin against both.
    runner = RUNNER.read_text(encoding="ascii") + SHARED_ENV.read_text(
        encoding="ascii"
    )

    assert "TASK_SET=exact4" in runner
    assert "exact16" not in runner
    assert "subset_b4_four.json" in runner
    assert "EXPECTED_TASKS=4" in runner
    for task_id in (
        "astropy__astropy-12907",
        "astropy__astropy-13033",
        "astropy__astropy-13236",
        "astropy__astropy-13398",
    ):
        assert task_id in runner
    for pin in (
        "MAX_NUM_SEQS_OVR=1",
        "SWE_CONCURRENCY=1",
        "CUDAGRAPH_MODE=FULL_AND_PIECEWISE",
        "FR13_DRAFT_VOCAB_ROOT=1",
        "FR13_DRAFT_VOCAB_K=65536",
        "physical_rows=32",
        "FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_SOURCE_COMMIT=",
        "--runtime-source-commit \"$SOURCE_COMMIT\"",
        "SFWD_PROFILE_PRESEED_COMMIT=ff067115c547a39bad706c10f91552896a87d264",
    ):
        assert pin in runner
    assert HISTORICAL_PASS in runner
    assert "fresh standalone SFWD gate summary" in runner
    assert "no_fallback" in runner
    assert "production engaged layer=" in runner
    assert "linear fallback engaged" in runner
    assert "capture lacks preseeded output bindings" in runner
    subprocess.run(["bash", "-n", RUNNER], cwd=REPO, check=True)


def test_launcher_allows_only_the_exact_historical_b1_tuple() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert (
        '"$FR13_FIXED32_CUTLASS_WAVE" == "identity_wide256_fullgrid_b1"'
        in launcher
    )
    assert (
        '"$FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_SOURCE_COMMIT" '
        '== "a8a904ed6c27a6338d43151038c155ebb76e3656"'
        in launcher
    )
    assert (
        '--runtime-source-commit "$_fr13_cutlass_runtime_source_commit"'
        in launcher
    )
    assert (
        "CUTLASS B1 historical qualification is restricted to the pinned "
        "cooperative target" in launcher
    )


def test_runtime_manifest_closes_the_exact4_runner() -> None:
    manifest = MANIFEST.read_text(encoding="ascii")
    pass_source = (SCRIPTS / "fr13_cutlass_streamk_pass.py").read_text(
        encoding="ascii"
    )

    assert '"scripts/fr13_run_b1_target_sfwd_exact4_timing.sh"' in manifest
    assert '"scripts/fr13_run_b1_target_sfwd_exact4_timing.sh"' in pass_source
    assert '"scripts/fr13_runtime_manifest.py"' in pass_source
