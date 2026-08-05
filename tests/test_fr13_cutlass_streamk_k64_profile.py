from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
BLOCK_MAP = SCRIPTS / "fr13_dvk_subset_blocks.json"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "fr13_cutlass_streamk_pass.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _k64_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load("fr13_cutlass_streamk_k64_profile_test")
    candidate_bytes = b"wide256 cap320 candidate\n"
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(candidate_bytes)
    patch_bytes = b"cap320 patch source\n"
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
    patch_source = tmp_path / "patch.py"
    patch_source.write_bytes(patch_bytes)
    monkeypatch.setattr(module.binary, "WIDE256_CANDIDATE_SIZE", len(candidate_bytes))
    monkeypatch.setattr(module.binary, "WIDE256_CANDIDATE_SHA256", candidate_sha256)
    monkeypatch.setattr(module, "PATCH_SOURCE_SHA256", patch_sha256)
    profile = module.QUALIFICATION_PROFILES["k64_root"]
    live = {
        "schema": module.K64_ROOT_LIVE_SCHEMA,
        "status": "pass",
        "run_classification": profile["run_classification"],
        "acceptance_valid": False,
        "task_count": 1,
        "task_ids": list(module.EXPECTED_TASK_IDS),
        "task_marker": module.EXPECTED_TASK_MARKER,
        "qualification_profile": "k64_root",
        "draft_vocab_root": 1,
        "draft_vocab_k": 65_536,
        "draft_vocab_blocks": module.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
        "draft_vocab_blocks_sha256": module.DRAFT_VOCAB_BLOCKS_SHA256,
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "comparator_timing_eligible": False,
        "batch_size": 1,
        "concurrency": 1,
        "fixed_rows": 32,
        "candidate": "streamk_force_wide256",
        "diagnostic_selector": "streamk_force_wide256_byte_ab",
        "served_result": "stock",
        "production_enabled": False,
        "comparison_call_limit": module.MAX_COMPARISONS,
        "comparisons": 257,
        "observed_m_values": [32],
        "observed_projection_nk": [
            list(shape) for shape in module.EXPECTED_PROJECTION_NK
        ],
        "mismatching_comparisons": 0,
        "differing_bytes": 0,
        "candidate_family": "streamk_force_wide256",
        "candidate_sha256": candidate_sha256,
        "candidate_bytes": len(candidate_bytes),
        "patch_source_sha256": patch_sha256,
        "vllm_base_commit": module.VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": module.PATCHED_DISPATCH_SHA256,
        "source_commit": "c" * 40,
        "binary_attestation_sha256": "d" * 64,
        "real_task_arm_sha256": "e" * 64,
        "container_env_sha256": "f" * 64,
        "errors": [],
    }
    live_path = tmp_path / "live.json"
    live_path.write_text(json.dumps(live, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live_path.read_bytes()).hexdigest()
    return module, candidate, patch_source, live_path, live_sha256


def _source_binding_repo(
    tmp_path: Path, module, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, Path]:
    repo = tmp_path / "source-repo"
    repo.mkdir()
    for index, relative in enumerate(module.SOURCE_BINDING_PATHS):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source binding file {index}: {relative}\n".encode("ascii"))
    block_map = repo / module.DRAFT_VOCAB_BLOCKS_SOURCE
    block_map.write_bytes(BLOCK_MAP.read_bytes())
    patch_source = repo / module.PATCH_SOURCE
    monkeypatch.setattr(
        module,
        "PATCH_SOURCE_SHA256",
        hashlib.sha256(patch_source.read_bytes()).hexdigest(),
    )
    subprocess.run(
        ["git", "init", "-q", "-b", "main"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "add", "--", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=FR13 test",
            "-c",
            "user.email=fr13-test@example.invalid",
            "commit",
            "-qm",
            "source binding fixture",
        ],
        cwd=repo,
        check=True,
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, source_commit, patch_source


def test_k64_root_live_pass_issues_and_verifies_distinct_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256 = _k64_fixture(
        tmp_path, monkeypatch
    )
    sidecar = tmp_path / "sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        candidate_selector="streamk_force_wide256",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    verified = module.verify_sidecar(
        sidecar,
        sidecar_sha256,
        candidate,
        patch_source,
        candidate_selector="streamk_force_wide256",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )

    assert verified == issued
    assert issued["schema"] == module.K64_ROOT_SIDECAR_SCHEMA
    assert issued["qualification_profile"] == "k64_root"
    assert issued["qualified_draft_vocab_root"] == 1
    assert issued["qualified_draft_vocab_k"] == 65_536
    assert issued["qualified_comparison_call_limit"] == 320
    assert (
        issued["qualified_draft_vocab_blocks_sha256"]
        == module.DRAFT_VOCAB_BLOCKS_SHA256
    )


def test_k64_root_accepts_static_persistent_stocktile_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, _ = _k64_fixture(tmp_path, monkeypatch)
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    monkeypatch.setattr(
        module.binary,
        "STATIC_PERSISTENT_B1_CANDIDATE_SIZE",
        len(candidate.read_bytes()),
    )
    monkeypatch.setattr(
        module.binary,
        "STATIC_PERSISTENT_B1_CANDIDATE_SHA256",
        candidate_sha256,
    )
    payload = json.loads(live.read_text(encoding="ascii"))
    payload.update(
        {
            "schema": module.STATIC_PERSISTENT_K64_ROOT_LIVE_SCHEMA,
            "candidate": "static_persistent_stocktile",
            "candidate_family": "static_persistent_stocktile",
            "diagnostic_selector": "static_persistent_stocktile_byte_ab",
        }
    )
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()
    sidecar = tmp_path / "static-sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        candidate_selector="static_persistent_stocktile",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )

    assert issued["candidate_selector"] == "static_persistent_stocktile"
    assert issued["diagnostic_selector"] == "static_persistent_stocktile_byte_ab"
    assert issued["qualification_profile"] == "k64_root"
    assert issued["qualified_comparison_call_limit"] == 320


def test_k64_root_accepts_divisor_static_stocktile_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, _ = _k64_fixture(tmp_path, monkeypatch)
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    monkeypatch.setattr(
        module.binary,
        "DIVISOR_STATIC_B1_CANDIDATE_SIZE",
        len(candidate.read_bytes()),
    )
    monkeypatch.setattr(
        module.binary,
        "DIVISOR_STATIC_B1_CANDIDATE_SHA256",
        candidate_sha256,
    )
    payload = json.loads(live.read_text(encoding="ascii"))
    payload.update(
        {
            "schema": module.DIVISOR_STATIC_K64_ROOT_LIVE_SCHEMA,
            "candidate": "divisor_static_stocktile",
            "candidate_family": "divisor_static_stocktile",
            "diagnostic_selector": "divisor_static_stocktile_byte_ab",
        }
    )
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()
    sidecar = tmp_path / "divisor-static-sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        candidate_selector="divisor_static_stocktile",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )

    assert issued["candidate_selector"] == "divisor_static_stocktile"
    assert issued["diagnostic_selector"] == "divisor_static_stocktile_byte_ab"
    assert issued["qualification_profile"] == "k64_root"
    assert issued["qualified_comparison_call_limit"] == 320


@pytest.mark.parametrize(
    (
        "candidate_selector",
        "diagnostic_selector",
        "live_schema",
        "size_attr",
        "sha_attr",
    ),
    (
        (
            "identity_onen_b1",
            "identity_onen_b1_byte_ab",
            "IDENTITY_ONEN_B1_K64_ROOT_LIVE_SCHEMA",
            "IDENTITY_ONEN_B1_CANDIDATE_SIZE",
            "IDENTITY_ONEN_B1_CANDIDATE_SHA256",
        ),
        (
            "identity_onen_n5120_single_b1",
            "identity_onen_n5120_single_b1_byte_ab",
            "IDENTITY_ONEN_N5120_SINGLE_B1_K64_ROOT_LIVE_SCHEMA",
            "IDENTITY_ONEN_N5120_SINGLE_B1_CANDIDATE_SIZE",
            "IDENTITY_ONEN_N5120_SINGLE_B1_CANDIDATE_SHA256",
        ),
        (
            "identity_onen_n5120_fullgrid_b1",
            "identity_onen_n5120_fullgrid_b1_byte_ab",
            "IDENTITY_ONEN_N5120_FULLGRID_B1_K64_ROOT_LIVE_SCHEMA",
            "IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SIZE",
            "IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SHA256",
        ),
        (
            "identity_wide256_fullgrid_b1",
            "identity_wide256_fullgrid_b1_byte_ab",
            "IDENTITY_WIDE256_FULLGRID_B1_K64_ROOT_LIVE_SCHEMA",
            "IDENTITY_WIDE256_FULLGRID_B1_CANDIDATE_SIZE",
            "IDENTITY_WIDE256_FULLGRID_B1_CANDIDATE_SHA256",
        ),
    ),
)
def test_k64_root_accepts_source_bound_onen_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_selector: str,
    diagnostic_selector: str,
    live_schema: str,
    size_attr: str,
    sha_attr: str,
) -> None:
    module, candidate, _, live, _ = _k64_fixture(tmp_path, monkeypatch)
    repo, source_commit, patch_source = _source_binding_repo(
        tmp_path, module, monkeypatch
    )
    if candidate_selector in module.SOURCE_CONTRACTS:
        contract = dict(module.SOURCE_CONTRACTS[candidate_selector])
        contract["patch_source_sha256"] = hashlib.sha256(
            patch_source.read_bytes()
        ).hexdigest()
        monkeypatch.setitem(module.SOURCE_CONTRACTS, candidate_selector, contract)
    source_contract = module._source_contract(candidate_selector)
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    monkeypatch.setattr(
        module.binary,
        size_attr,
        len(candidate.read_bytes()),
    )
    monkeypatch.setattr(
        module.binary,
        sha_attr,
        candidate_sha256,
    )
    payload = json.loads(live.read_text(encoding="ascii"))
    payload.update(
        {
            "schema": getattr(module, live_schema),
            "candidate": candidate_selector,
            "candidate_family": candidate_selector,
            "diagnostic_selector": diagnostic_selector,
            "patch_source_sha256": source_contract["patch_source_sha256"],
            "patched_dispatch_sha256": source_contract[
                "patched_dispatch_sha256"
            ],
            "source_commit": source_commit,
            "source_identity": module.validate_source_commit_binding(
                source_commit, patch_source, candidate_selector
            ),
        }
    )
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()
    sidecar = tmp_path / "onen-b1-sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        candidate_selector=candidate_selector,
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    monkeypatch.setattr(
        module,
        "_git_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module._GitUnavailableError("git is unavailable for source binding")
        ),
    )
    verified = module.verify_sidecar(
        sidecar,
        sidecar_sha256,
        candidate,
        patch_source,
        candidate_selector=candidate_selector,
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )
    binding = None
    if candidate_selector in module.binary.INSTALLABLE_SELECTORS:
        destination = tmp_path / "installed.so"
        destination.write_bytes(b"stock\n")
        attestation = tmp_path / "attestation.json"
        monkeypatch.setitem(sys.modules, "fr13_cutlass_streamk_pass", module)
        monkeypatch.setattr(module.binary, "CONTAINER_SOURCE", candidate)
        monkeypatch.setattr(module.binary, "CONTAINER_DESTINATION", destination)
        module.binary.install_candidate(
            candidate,
            destination,
            attestation,
            candidate_selector,
            qualification_profile="k64_root",
            production_sidecar=sidecar,
            expected_production_sidecar_sha256=sidecar_sha256,
            patch_source=patch_source,
        )
        binding = module.validate_production_attestation(
            attestation,
            sidecar_sha256,
            qualification_profile="k64_root",
            draft_vocab_blocks=repo / module.DRAFT_VOCAB_BLOCKS_SOURCE,
            patch_source=patch_source,
        )

    assert verified == issued
    assert issued["candidate_selector"] == candidate_selector
    assert issued["diagnostic_selector"] == diagnostic_selector
    assert issued["qualification_profile"] == "k64_root"
    assert issued["qualified_comparison_call_limit"] == 320
    assert issued["qualification_source_identity"]["source_commit"] == source_commit
    if binding is not None:
        assert binding["qualification_source_identity"] == issued[
            "qualification_source_identity"
        ]


@pytest.mark.parametrize(
    ("failure", "match"),
    (
        ("tampered", "SHA-256 mismatch"),
        ("missing", "does not exist"),
        ("symlink", "not a regular non-symlink file"),
        ("wrong_commit", "identity commit mismatch"),
    ),
)
def test_no_git_runtime_source_binding_rejects_mounted_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    match: str,
) -> None:
    module = _load(f"fr13_cutlass_streamk_no_git_{failure}_test")
    repo, source_commit, patch_source = _source_binding_repo(
        tmp_path, module, monkeypatch
    )
    candidate_selector = "identity_onen_n5120_fullgrid_b1"
    contract = dict(module.SOURCE_CONTRACTS[candidate_selector])
    contract["patch_source_sha256"] = hashlib.sha256(
        patch_source.read_bytes()
    ).hexdigest()
    monkeypatch.setitem(module.SOURCE_CONTRACTS, candidate_selector, contract)
    source_identity = module.validate_source_commit_binding(
        source_commit, patch_source, candidate_selector
    )
    target = repo / module.SOURCE_BINDING_PATHS[-1]
    if failure == "tampered":
        original = target.read_bytes()
        target.write_bytes(bytes((original[0] ^ 1,)) + original[1:])
    elif failure == "missing":
        target.unlink()
    elif failure == "symlink":
        target.unlink()
        target.symlink_to(patch_source)
    else:
        source_commit = "d" * 40
    monkeypatch.setattr(
        module,
        "_git_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module._GitUnavailableError("git is unavailable for source binding")
        ),
    )

    with pytest.raises(module.QualificationError, match=match):
        module._validate_runtime_source_commit_binding(
            source_commit,
            source_identity,
            patch_source,
            candidate_selector,
        )


def test_strict_source_binding_still_requires_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("fr13_cutlass_streamk_strict_no_git_test")
    _, source_commit, patch_source = _source_binding_repo(
        tmp_path, module, monkeypatch
    )
    monkeypatch.setattr(
        module,
        "_git_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module._GitUnavailableError("git is unavailable for source binding")
        ),
    )

    with pytest.raises(
        module.QualificationError, match="git is unavailable for source binding"
    ):
        module.validate_source_commit_binding(source_commit, patch_source)


@pytest.mark.parametrize(
    "candidate_selector",
    (
        "identity_onen_b1",
        "identity_onen_n5120_single_b1",
        "identity_onen_n5120_fullgrid_b1",
        "identity_wide256_fullgrid_b1",
    ),
)
def test_source_bound_onen_sidecar_rejects_full_vocab_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_selector: str,
) -> None:
    module, candidate, patch_source, _, _ = _k64_fixture(tmp_path, monkeypatch)
    sidecar = tmp_path / "full-vocab-sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": module.SIDECAR_SCHEMA,
                "candidate_selector": candidate_selector,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )

    with pytest.raises(
        module.QualificationError,
        match=rf"{candidate_selector} qualification requires the k64_root profile",
    ):
        module.verify_sidecar(
            sidecar,
            hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            candidate,
            patch_source,
            candidate_selector=candidate_selector,
            qualification_profile="full_vocab",
        )


def test_n5120_single_source_contract_remains_pinned_to_original_patch() -> None:
    module = _load("fr13_cutlass_streamk_n5120_source_contract_test")
    contract = module._source_contract("identity_onen_n5120_single_b1")

    assert contract["patch_source_sha256"] == (
        "eadff808ef7db8de342d8c51e046cda9cc78bc4e308d1c1d08d5b33f7af1d2b0"
    )
    assert contract["patched_dispatch_sha256"] == (
        "5e856f587480d2d04d9127b25e12d40ef82b8d07a2301389ab757523ce206d2d"
    )
    assert module.sha256_file(REPO / module.PATCH_SOURCE) != contract[
        "patch_source_sha256"
    ]


def test_n5120_fullgrid_source_contract_remains_pinned_to_qualified_patch() -> None:
    module = _load("fr13_cutlass_streamk_n5120_fullgrid_source_contract_test")
    contract = module._source_contract("identity_onen_n5120_fullgrid_b1")

    assert contract["patch_source_sha256"] == (
        "623582b257a13f7551c81aaf8e87f7542ddb4d6564636f5e177ec0807126a341"
    )
    assert contract["patched_dispatch_sha256"] == (
        "710da7d3a8e24c83f9f095222d5297d96f610c6310f3a8537ed1b925a25ece56"
    )
    assert module.sha256_file(REPO / module.PATCH_SOURCE) != contract[
        "patch_source_sha256"
    ]


def test_wide256_fullgrid_source_contract_matches_integrated_patcher() -> None:
    module = _load("fr13_cutlass_streamk_wide256_fullgrid_source_contract_test")
    contract = module._source_contract("identity_wide256_fullgrid_b1")

    assert contract["patch_source_sha256"] == (
        "ae9591a0c255c54bd8b5fed8576105013fce7f5f0834dbfb51ca1d455441f976"
    )
    assert contract["patched_dispatch_sha256"] == (
        "569aea20321ba5461c4d3c9187aadf5390be363485f9aee538a738ef269ca6f0"
    )
    assert module.sha256_file(REPO / module.PATCH_SOURCE) == contract[
        "patch_source_sha256"
    ]


def test_source_binding_rejects_forged_40hex_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("fr13_cutlass_streamk_forged_commit_test")
    _, _, patch_source = _source_binding_repo(tmp_path, module, monkeypatch)

    with pytest.raises(module.QualificationError, match="commit resolution failed"):
        module.validate_source_commit_binding("f" * 40, patch_source)


def test_source_binding_rejects_changed_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("fr13_cutlass_streamk_dirty_tree_test")
    repo, source_commit, patch_source = _source_binding_repo(
        tmp_path, module, monkeypatch
    )
    launcher = repo / "scripts/fr13_launch_forked_fa2_tree_server.sh"
    launcher.write_bytes(launcher.read_bytes() + b"dirty change\n")

    with pytest.raises(
        module.QualificationError, match="dirty tracked working tree"
    ):
        module.validate_source_commit_binding(source_commit, patch_source)


def test_k64_root_rejects_non_wide_candidate_and_block_map_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256 = _k64_fixture(
        tmp_path, monkeypatch
    )

    with pytest.raises(module.QualificationError, match="restricted to"):
        module.validate_live_result(
            live,
            live_sha256,
            candidate,
            patch_source,
            candidate_selector="streamk_coop128",
            qualification_profile="k64_root",
        )

    drifted = tmp_path / "blocks.json"
    drifted.write_text("{}\n", encoding="ascii")
    with pytest.raises(module.QualificationError, match="block-map SHA-256 mismatch"):
        module.validate_live_result(
            live,
            live_sha256,
            candidate,
            patch_source,
            candidate_selector="streamk_force_wide256",
            qualification_profile="k64_root",
            draft_vocab_blocks=drifted,
        )


def test_k64_root_attestation_preserves_profile_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256 = _k64_fixture(
        tmp_path, monkeypatch
    )
    sidecar = tmp_path / "sidecar.json"
    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        candidate_selector="streamk_force_wide256",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    qualification = dict(issued)
    qualification["sidecar_sha256"] = sidecar_sha256
    identity = {
        "path": str(module.binary.CONTAINER_SOURCE),
        "bytes": len(candidate.read_bytes()),
        "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "regular": True,
        "symlink": False,
    }
    destination = dict(identity)
    destination["path"] = str(module.binary.CONTAINER_DESTINATION)
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "schema": module.ATTESTATION_SCHEMA,
                "selector": "streamk_force_wide256",
                "source": identity,
                "destination": destination,
                "installed_mode": "0555",
                "production_enabled": True,
                "qualification": qualification,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )

    result = module.validate_production_attestation(
        attestation,
        sidecar_sha256,
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )

    assert result["schema"].endswith("k64_root.production_binding.v1")
    assert result["qualification_profile"] == "k64_root"
    assert result["qualified_comparison_call_limit"] == 320


def _astropy13236_fullgrid_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module, candidate, patch_source, live, _ = _k64_fixture(tmp_path, monkeypatch)
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    monkeypatch.setattr(
        module.binary,
        "IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SIZE",
        len(candidate.read_bytes()),
    )
    monkeypatch.setattr(
        module.binary,
        "IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SHA256",
        candidate_sha256,
    )
    contract = dict(module.SOURCE_CONTRACTS["identity_onen_n5120_fullgrid_b1"])
    contract["patch_source_sha256"] = hashlib.sha256(
        patch_source.read_bytes()
    ).hexdigest()
    monkeypatch.setitem(
        module.SOURCE_CONTRACTS,
        "identity_onen_n5120_fullgrid_b1",
        contract,
    )
    source_identity = {"source_commit": "c" * 40}
    monkeypatch.setattr(
        module,
        "validate_source_commit_binding",
        lambda *_args, **_kwargs: source_identity,
    )
    payload = json.loads(live.read_text(encoding="ascii"))
    payload.update(
        {
            "schema": module.IDENTITY_ONEN_N5120_FULLGRID_B1_K64_ROOT_LIVE_SCHEMA,
            "candidate": "identity_onen_n5120_fullgrid_b1",
            "diagnostic_selector": "identity_onen_n5120_fullgrid_b1_byte_ab",
            "task_ids": ["astropy__astropy-13236"],
            "task_marker": "swe_verified:astropy__astropy-13236",
            "diagnostic_task_profile": "astropy13236",
            "patch_source_sha256": contract["patch_source_sha256"],
            "patched_dispatch_sha256": contract["patched_dispatch_sha256"],
            "source_identity": source_identity,
        }
    )
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    return module, candidate, patch_source, live


def test_astropy13236_fullgrid_live_pass_issues_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live = _astropy13236_fullgrid_fixture(
        tmp_path, monkeypatch
    )
    sidecar = tmp_path / "astropy13236-sidecar.json"
    issued = module.issue_sidecar(
        live,
        hashlib.sha256(live.read_bytes()).hexdigest(),
        candidate,
        sidecar,
        patch_source,
        candidate_selector="identity_onen_n5120_fullgrid_b1",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
        diagnostic_task_profile="astropy13236",
    )
    verified = module.verify_sidecar(
        sidecar,
        hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        candidate,
        patch_source,
        candidate_selector="identity_onen_n5120_fullgrid_b1",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
        diagnostic_task_profile="astropy13236",
    )

    assert verified == issued
    assert issued["qualification_task_profile"] == "astropy13236"
    assert issued["qualification_task_ids"] == ["astropy__astropy-13236"]
    assert issued["qualification_task_marker"].endswith("astropy-13236")


@pytest.mark.parametrize(
    ("key", "value", "match"),
    (
        ("diagnostic_task_profile", "astropy12907", "diagnostic_task_profile"),
        ("task_ids", ["astropy__astropy-12907"], "task_ids"),
        (
            "task_marker",
            "swe_verified:astropy__astropy-12907",
            "task_marker",
        ),
    ),
)
def test_astropy13236_fullgrid_rejects_wrong_profile_task_or_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: object,
    match: str,
) -> None:
    module, candidate, patch_source, live = _astropy13236_fullgrid_fixture(
        tmp_path, monkeypatch
    )
    payload = json.loads(live.read_text(encoding="ascii"))
    payload[key] = value
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")

    with pytest.raises(module.QualificationError, match=match):
        module.validate_live_result(
            live,
            hashlib.sha256(live.read_bytes()).hexdigest(),
            candidate,
            patch_source,
            candidate_selector="identity_onen_n5120_fullgrid_b1",
            qualification_profile="k64_root",
            draft_vocab_blocks=BLOCK_MAP,
            diagnostic_task_profile="astropy13236",
        )


def test_astropy13236_profile_rejects_non_n5120_candidate() -> None:
    module = _load("fr13_cutlass_streamk_task_profile_restriction")

    with pytest.raises(module.QualificationError, match="is not allowed"):
        module._diagnostic_task_profile("streamk_force_wide256", "astropy13236")
