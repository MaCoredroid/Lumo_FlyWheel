"""The GDN single-launch production arm: built, credential-bound, default OFF.

BUILT 2026-08-14. This is the promotion MACHINERY for the folded GDN scan
candidate ``fixed32_gdn_single_launch_tree_v2``, and it deliberately ships
switched off. That distinction is the point of this file, so it is worth being
exact about what the arm has and has not earned.

WHAT IT HAS
  legality  the b4 Hydra27 live byte gate,
            output/fr13_gdn_single_launch_b4_gate_20260814T191621Z, returned
            PASS on real SWE-Verified traffic: 1,584 records (33 comparator
            events x 48 GDN layers) compared across all seven surfaces
            (output, ring_k, ring_v, ring_a, ring_b, flags, counter),
            raw_byte_equal true, every event at request-tuple width 4. The B1
            arm was sealed earlier at hydra27:b1.
  price     results/fr13_gdn_scan_b4_probe_20260814, measured directly at b=4
            rather than transferred from b=1: two_launch 861.504 us per
            layer-batch against single_launch 674.336, a saving of 8.984
            ms/step over 48 layers = 2.14x the 4.20 ms sealed MDE.

WHAT IT DOES NOT HAVE
  A sealing campaign. There is no pre-registered, paired, one-sided 95% lower
  bound on the batch-conditioned width-4 improvement, and no ruling on one.
  Byte-legality says the kernel MAY serve; it does not say the tree SHOULD
  serve it by default. The two FA2 arms in the same registry ship as
  ``gqa_pair`` because they cleared exactly that bar; this one ships as ``0``
  because it has not.

WHAT THESE TESTS PROTECT

1. ONE SOURCE OF TRUTH. scripts/fr13_canonical_env.sh owns the shipped value
   and the launcher restates it as a fallback. The two may never disagree --
   the failure mode the mamba-narrowing promotion left open for two days -- so
   these tests parse both files and compare rather than restating a literal in
   a third place.

2. THE ARM IS OFF, AND CANNOT DRIFT ON BY ACCIDENT. A one-token edit to the
   registry is the whole promotion, which is convenient for the eventual flip
   and dangerous in the meantime. So the shipped value is asserted directly,
   and the registry is required to keep carrying its own justification: if
   somebody flips the value they must also delete the paragraph explaining why
   it was zero, which is a deliberate, visible act rather than a typo.

3. THE CREDENTIAL CHAIN IS REAL. The arm is HEAD-bound: the credential must
   name the exact commit being served, so any source change at all invalidates
   it and forces a re-gate. The validator is exercised against the ACTUAL
   phase-1 artifact, not a synthetic fixture, because a validator that only
   ever sees hand-made input tends to encode the fixture rather than the gate.

4. THE TWO FOLDED ARMS CANNOT BE CONFUSED. single_launch and gqa_group3 share
   a schema, a scope, a task list and a reference. What separates them is the
   grouped-source pin: the twin requires ``gqa_group3_source_sha256`` to be a
   digest, this one requires it to be null. Neither credential can satisfy the
   other's clause, so the arms are mutually exclusive by construction and not
   merely by convention.

5. PROMOTION MOVES THE PINS WITH THE ARM. The engagement needle proves the
   folded kernel actually replaced the incumbent launch -- zero state export
   writes, zero parent reads, grid z equal to the batch -- and it compares
   against ``batch`` rather than against 1, because the b1-shaped literal is
   the defect this campaign has already paid for once.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "scripts" / "fr13_canonical_env.sh"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
VALIDATOR = REPO / "scripts" / "fr13_gdn_single_launch_production_credential.py"
MANIFEST = REPO / "scripts" / "fr13_runtime_manifest.py"
TWIN = REPO / "scripts" / "fr13_gdn_gqa_group3_production_credential.py"

CREDENTIAL = (
    REPO
    / "output"
    / "fr13_gdn_single_launch_b4_gate_20260814T191621Z"
    / "hydra27_fixed32_hydra27_gdn_single_launch_b4_20260814T191621Z"
    / "hydra27_gdn_single_launch_b4_credential.json"
)
CREDENTIAL_COMMIT = "ddbfe5b688bd2f3f05b925b20fc0a56e15ad24bf"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _validate(credential: Path, *, commit: str, mode: str, batch: int):
    return subprocess.run(
        [
            str(REPO / ".venv" / "bin" / "python"),
            str(VALIDATOR),
            "--credential",
            str(credential),
            "--source-commit",
            commit,
            "--profile",
            "fixed32",
            "--mode",
            mode,
            "--batch",
            str(batch),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


# ---------------------------------------------------------------- one source of truth


def test_registry_owns_the_value_and_ships_the_arm_off() -> None:
    registry = _read(REGISTRY)
    match = re.search(
        r'export FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_DEFAULT='
        r'"\$\{FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_DEFAULT:-([^}]*)\}"',
        registry,
    )
    assert match is not None, "the registry must own the shipped default"
    assert match.group(1) == "0", (
        "the GDN single-launch arm is byte-legal but NOT sealed; it must ship "
        "off until a sealing campaign and a ruling say otherwise"
    )


def test_launcher_fallback_matches_the_registry_exactly() -> None:
    registry = _read(REGISTRY)
    launcher = _read(LAUNCHER)
    shipped = re.search(
        r'export FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_DEFAULT='
        r'"\$\{FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_DEFAULT:-([^}]*)\}"',
        registry,
    )
    fallback = re.search(
        r'_fr13_gdn_single_launch_production=\$\{'
        r'FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_DEFAULT:-([^}]*)\}',
        launcher,
    )
    assert shipped is not None and fallback is not None
    assert shipped.group(1) == fallback.group(1), (
        "the registry and the launcher fallback disagree about what this "
        "branch ships"
    )


def test_the_registry_cannot_export_the_selector_itself() -> None:
    registry = _read(REGISTRY)
    assert "FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_DEFAULT" in registry
    assert not re.search(
        r"^export FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION=",
        registry,
        re.MULTILINE,
    ), (
        "the registry is sourced before the batch is known and the selector is "
        "credentialed; exporting it here would hand every arm a selector the "
        "launcher must then refuse at boot"
    )


def test_the_registry_keeps_carrying_its_own_justification() -> None:
    registry = _read(REGISTRY)
    # Flipping the value is a one-token edit. Requiring the justification to
    # travel with it makes an accidental flip impossible to do quietly.
    for phrase in (
        "WHY THIS SHIPS AS 0",
        "not sealed",
        "fr13_gdn_single_launch_b4_gate_20260814T191621Z",
        "fr13_gdn_scan_b4_probe_20260814",
    ):
        assert phrase in registry, f"registry lost its rationale: {phrase!r}"


# ---------------------------------------------------------------- scoping the default


def test_named_and_unset_are_distinguished_before_normalisation() -> None:
    launcher = _read(LAUNCHER)
    assert "_FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_NAMED=0" in launcher
    assert "[[ -v FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION ]]" in launcher
    named_at = launcher.index("[[ -v FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION ]]")
    normalised_at = launcher.index(
        "_fr13_gdn_single_launch_production=${FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION:-0}"
    )
    assert named_at < normalised_at, (
        "`${VAR:-0}` cannot tell 'named 0' from 'unset'; the distinction must "
        "be recorded before it is erased"
    )


def test_the_default_only_applies_in_the_credentialed_serving_shape() -> None:
    launcher = _read(LAUNCHER)
    start = launcher.index("_FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_NAMED == 0")
    block = launcher[start : start + 1600]
    # A credential must have been presented -- batch shape alone is every
    # campaign arm in the tree and would hand all of them a credentialed
    # selector the chain must then refuse at boot.
    assert '-n "$_fr13_gdn_single_launch_pass_json"' in block
    assert '"${SWE_CONCURRENCY:-}" == "$MAX_NUM_SEQS"' in block
    assert '"${FR13_FIXED32_B1_DIAGNOSTIC:-0}" == "0"' in block
    # and no sibling GDN selector may be engaged
    assert '"$_fr13_gdn_gqa_group3_production" == "0"' in block
    assert '-z "$_fr13_gdn_path_bv_candidate"' in block
    assert '-z "$_fr13_gdn_path_bv_production"' in block
    assert '-z "$_fr13_gdn_single_launch_expected_batch"' in block


def test_the_two_folded_arms_exclude_each_other_in_both_directions() -> None:
    launcher = _read(LAUNCHER)
    patcher = _read(PATCHER)
    # launcher: each production block refuses the other arm
    assert '&& "$_fr13_gdn_single_launch_production" == "0" \\' in launcher
    assert '&& "$_fr13_gdn_gqa_group3_production" == "0" \\' in launcher
    # patcher: one call site, so the selector tuple is exhaustive
    assert "single_launch_production," in patcher
    assert 'or single_launch_production == "1"' in patcher
    assert 'or gqa_group3_production == "1"' in patcher


def test_the_gate_selector_is_not_the_production_selector() -> None:
    launcher = _read(LAUNCHER)
    # The live byte gate reaches the folded kernel through the DIAGNOSTIC
    # candidate selector and must never be exported as a serving arm.
    assert "-e FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE=" not in launcher


# ---------------------------------------------------------------- the credential chain


@pytest.mark.skipif(
    not CREDENTIAL.is_file(), reason="phase-1 gate artifact is not present"
)
def test_validator_accepts_the_real_phase_1_credential_at_its_own_commit() -> None:
    done = _validate(
        CREDENTIAL, commit=CREDENTIAL_COMMIT, mode="hydra27_fixed32", batch=4
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.skipif(
    not CREDENTIAL.is_file(), reason="phase-1 gate artifact is not present"
)
@pytest.mark.parametrize(
    "commit,mode,batch",
    [
        ("0" * 40, "hydra27_fixed32", 4),          # wrong HEAD
        (CREDENTIAL_COMMIT, "hydra27_fixed32", 1),  # wrong batch
        (CREDENTIAL_COMMIT, "tail6_fixed32", 4),    # wrong mode
    ],
)
def test_validator_is_head_mode_and_batch_bound(commit, mode, batch) -> None:
    done = _validate(CREDENTIAL, commit=commit, mode=mode, batch=batch)
    assert done.returncode != 0, (
        "the credential is strictly bound; it must not validate off-target"
    )


@pytest.mark.skipif(
    not CREDENTIAL.is_file(), reason="phase-1 gate artifact is not present"
)
def test_the_credential_is_a_legality_artifact_not_a_performance_one(tmp_path) -> None:
    payload = json.loads(CREDENTIAL.read_text())
    # The gate serves the REFERENCE and compares the candidate as a shadow, so
    # it cannot speak to performance or acceptance and must not claim to.
    assert payload["reference_served"] is True
    assert payload["production_enabled"] is False
    assert payload["performance_measurement"] is False
    assert payload["acceptance_valid"] is False
    assert payload["floor_acceptance_eligible"] is False
    assert payload["raw_byte_equal"] is True
    assert payload["candidate_physical_launches_per_request_layer"] == 1
    assert payload["reference_physical_launches_per_request_layer"] == 2


@pytest.mark.skipif(
    not CREDENTIAL.is_file(), reason="phase-1 gate artifact is not present"
)
def test_a_sibling_shaped_credential_is_refused(tmp_path) -> None:
    # Same schema, same scope, same tasks, same reference -- the ONLY thing that
    # separates the two folded arms at this layer is the grouped-source pin.
    payload = json.loads(CREDENTIAL.read_text())
    payload["gqa_group3_source_sha256"] = "b" * 64
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(payload))
    done = _validate(
        forged, commit=CREDENTIAL_COMMIT, mode="hydra27_fixed32", batch=4
    )
    assert done.returncode != 0
    assert "grouped-GQA source pin" in done.stderr


@pytest.mark.skipif(
    not CREDENTIAL.is_file(), reason="phase-1 gate artifact is not present"
)
def test_duplicate_json_keys_are_refused(tmp_path) -> None:
    raw = CREDENTIAL.read_text().rstrip()
    assert raw.endswith("}")
    forged = tmp_path / "dupe.json"
    forged.write_text(raw[:-1] + ', "status": "PASS"}')
    done = _validate(
        forged, commit=CREDENTIAL_COMMIT, mode="hydra27_fixed32", batch=4
    )
    assert done.returncode != 0
    assert "duplicate JSON key" in done.stderr


def test_a_symlinked_credential_is_refused(tmp_path) -> None:
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    done = _validate(link, commit="a" * 40, mode="hydra27_fixed32", batch=4)
    assert done.returncode != 0
    assert "singly linked regular non-symlink" in done.stderr


def test_the_twin_validators_are_structural_mirrors() -> None:
    mine = _read(VALIDATOR)
    twin = _read(TWIN)
    for shared in (
        "fr13.fixed32.gdn_single_launch.real_task_credential.v3",
        "fixed32_gdn_two_launch_reference_v1",
        "scripts/fr13_fixed32_floor_timers_seq.sh",
        "_duplicate_checked",
        "_reject_constant",
    ):
        assert shared in mine and shared in twin
    assert 'CANDIDATE = "fixed32_gdn_single_launch_tree_v2"' in mine
    assert 'CANDIDATE = "fixed32_gdn_single_launch_gqa_group3_v1"' in twin


# ---------------------------------------------------------------- pins move with the arm


def test_launcher_verifies_the_credential_twice_and_binds_head_both_times() -> None:
    launcher = _read(LAUNCHER)
    assert launcher.count("fr13_gdn_single_launch_production_credential.py") == 2, (
        "verify the caller's path AND the copy that reaches the container"
    )
    assert (
        launcher.count('--source-commit "$_fr13_gdn_single_launch_source_commit"')
        == 2
    )
    assert "git rev-parse --verify 'HEAD^{commit}'" in launcher


def test_launcher_writes_exports_and_removes_the_production_sidecars() -> None:
    launcher = _read(LAUNCHER)
    for sidecar in (
        "fr13_fixed32_gdn_single_launch.production.arm",
        "fr13_fixed32_gdn_single_launch.production_batch.flag",
        "fr13_fixed32_gdn_single_launch.production_credential.json",
    ):
        # written once, removed in BOTH the fixed32 and non-fixed32 arms, so a
        # stale sidecar can never present itself as this boot's licence
        assert launcher.count(sidecar) >= 3, sidecar
    assert (
        "-e FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION="
        '"$_fr13_gdn_single_launch_production"' in launcher
    ), "the container must receive the RESOLVED value, not the raw caller env"
    assert (
        "-e FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_PASS_PATH="
        "/logs/fr13_fixed32_gdn_single_launch.production_credential.json"
        in launcher
    )


def test_the_caller_env_snapshot_covers_the_new_selector_trio() -> None:
    launcher = _read(LAUNCHER)
    for name in (
        "_FR13_CALLER_GDN_SINGLE_LAUNCH_PRODUCTION",
        "_FR13_CALLER_GDN_SINGLE_LAUNCH_BATCH",
        "_FR13_CALLER_GDN_SINGLE_LAUNCH_PASS",
    ):
        # snapshot + arming test + re-compare + scrub = four mentions
        assert launcher.count(name) >= 4, name


def test_the_engagement_needle_proves_the_fold_and_is_not_b1_shaped() -> None:
    patcher = _read(PATCHER)
    start = patcher.index("_FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION\n        and batch ==")
    block = patcher[start : start + 3000]
    assert '!= "fixed32_single_launch_tree"' in block
    assert '!= "fixed32_gdn_single_launch_tree_v2"' in block
    # the fold's structural signature: the handoff is gone, not merely faster
    assert 'int(executed_gdn.get("state_export_writes", -1)) != 0' in block
    assert 'int(executed_gdn.get("state_parent_reads", -1)) != 0' in block
    # and the width checks are keyed to the batch, never to a literal 1
    assert 'int(executed_gdn.get("physical_programs", -1)) != batch' in block
    assert "expected_grid = (batch,)" in block
    assert (
        "FR13 GDN single-launch production did not replace the captured "
        in patcher
    )


def test_the_patcher_pins_the_production_shape() -> None:
    patcher = _read(PATCHER)
    assert (
        "FR13 GDN single-launch production requires exact credentialed "
        in patcher
    )
    assert "exact_single_launch_production" in patcher
    assert (
        "FR13 GDN single-launch production batch is set without its arm"
        in patcher
    )


def test_the_credential_validator_is_in_the_runtime_manifest() -> None:
    manifest = _read(MANIFEST)
    assert "scripts/fr13_gdn_single_launch_production_credential.py" in manifest, (
        "the validator is part of the served source closure; if it is not in "
        "the manifest it can change without invalidating a credential"
    )
