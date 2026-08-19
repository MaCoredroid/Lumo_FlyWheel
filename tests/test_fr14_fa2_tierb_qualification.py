"""FR14 Tier-B qualification route: the second door, and its lock.

Mark's pass-64 ruling created a route by which an arm that cannot be
byte-identical to the incumbent BY CONSTRUCTION may still serve. The danger in
such a route is not that it exists; it is that it becomes a way around the
byte-exact door for something that should have gone through it. Most of this
file is about that: the Tier-B validator must refuse a non-tier-b arm before it
looks at a single measurement, the byte gate must still refuse a differing
reduction topology, and the production arm set must not have grown.

The rest pins the two properties a credential has to have to be worth
anything: it must bind to the exact kernel it attests, and its verdict must be
RECOMPUTED from the measurements rather than read out of the file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts/fr13_patch_fa2_tree_bias.py"
SIDECAR = REPO / "scripts/fr13_qrow32_b1_pass_sidecar.py"
CONTRACT = REPO / "scripts/fr13_fixed32_contract.py"
REDUCER = REPO / "scripts/fr14_reduce_splitk_tierb_credential.py"
BOUNDS = REPO / "results/fr14_nvfp4_port_20260816/fr14_splitk_tierb_bounds.json"
PROBE_A = REPO / "results/fr14_nvfp4_port_20260816/fr14_splitk_fa2_probe_result.json"
PROBE_B = (
    REPO
    / "results/fr14_nvfp4_port_20260816/fr14_splitk_fa2_probe_result_process_b.json"
)
SPLITK_ARM = "gqa_pair_splitk"
SPLITK_SO_SHA256 = (
    "28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857"
)
FAKE_COMMIT = "0" * 40
FAKE_PATCH_SHA = "1" * 64


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _sidecar():
    return _module(SIDECAR, "fr14_tierb_test_sidecar")


def _contract():
    sys.path.insert(0, str(REPO / "scripts"))
    return _module(CONTRACT, "fr13_fixed32_contract")


def _reducer():
    return _module(REDUCER, "fr14_tierb_test_reducer")


def _selectors():
    """The B1 selector helpers as they are injected into the served vLLM."""
    import os

    patcher = _module(PATCHER, "fr14_tierb_test_patcher")
    namespace: dict = {"os": os}
    exec(  # noqa: S102 - the emitted source is the thing under test
        compile(
            "import os\n" + patcher.FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS,
            "<b1_selectors>",
            "exec",
        ),
        namespace,
    )
    return namespace


def _credential(tmp_path: Path, **overrides):
    """A real credential, reduced from the banked probe runs."""
    reducer = _reducer()
    sidecar = _sidecar()
    contract = _contract()
    bounds = sidecar.load_tierb_bounds(BOUNDS)
    probes = [json.loads(PROBE_A.read_text()), json.loads(PROBE_B.read_text())]
    payload = reducer.build_credential(
        probes,
        arm=SPLITK_ARM,
        source_commit=FAKE_COMMIT,
        patch_source_sha256=FAKE_PATCH_SHA,
        bounds=bounds,
        bounds_sha256=sidecar.TIERB_BOUNDS_SHA256,
        sidecar=sidecar,
        contract=contract,
    )
    for dotted, value in overrides.items():
        node = payload
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    if "credential_sha256" not in overrides:
        payload["credential_sha256"] = sidecar.tierb_credential_digest(payload)
    path = tmp_path / "credential.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return path, payload


def _validate(path: Path, arm: str = SPLITK_ARM):
    return _sidecar().validate_tierb_credential(
        path,
        arm=arm,
        expected_candidate_sha256=SPLITK_SO_SHA256,
        expected_source_commit=FAKE_COMMIT,
        expected_patch_source_sha256=FAKE_PATCH_SHA,
        bounds_path=BOUNDS,
    )


# ------------------------------------------------- nothing else gets easier


def test_tierb_credential_is_refused_for_every_arm_that_is_not_tier_b(
    tmp_path: Path,
) -> None:
    """The refusal that matters most, and it is FIRST.

    A Tier-B credential presented for a byte-gated arm must fail because that
    arm is not tier-b -- before any measurement is examined, and no matter how
    good the numbers in it are.
    """
    path, _ = _credential(tmp_path)
    sidecar = _sidecar()
    for arm in ("nosplit", "split2", "visibility", "gqa_pair"):
        with pytest.raises(ValueError, match="not accepted for arm"):
            _validate(path, arm=arm)
    # And the message has to point at the door that arm should use.
    with pytest.raises(ValueError, match="Tier-A path is unchanged"):
        _validate(path, arm="gqa_pair")
    assert sidecar.TIERB_ARMS == (SPLITK_ARM,)


def test_the_byte_gate_still_refuses_a_differing_reduction_topology() -> None:
    """Tier-B did not weaken Tier-A: the byte gate's rule is untouched."""
    require = _selectors()["_fr13_fa2_qrow32_b1_require_same_reduction"]
    require("gqa_pair", 0)
    require("nosplit", 0)
    with pytest.raises(RuntimeError, match="identical reduction topology"):
        require(SPLITK_ARM, 0)
    with pytest.raises(RuntimeError, match="identical reduction topology"):
        require("split2", 0)


def test_production_arm_set_did_not_grow(monkeypatch) -> None:
    """Tier-B grants live-A/B serving, not promoted-default."""
    namespace = _selectors()
    assert namespace["_FR13_FA2_QROW32_B1_PRODUCTION_ARMS"] == (
        "nosplit", "gqa_pair",
    )
    assert SPLITK_ARM in namespace["_FR13_FA2_QROW32_B1_TIER_B_ARMS"]
    assert SPLITK_ARM not in namespace["_FR13_FA2_QROW32_B1_PRODUCTION_ARMS"]
    monkeypatch.setenv("FR13_FA2_QROW32_B1_PRODUCTION_ARM", SPLITK_ARM)
    with pytest.raises(RuntimeError, match="must be empty or one of"):
        namespace["_fr13_fa2_qrow32_b1_serving_arm"]()
    # The sidecar's mirrored view must agree, or a tier-b arm could be both.
    assert _sidecar()._FR13_PRODUCTION_ARMS_VIEW == (
        namespace["_FR13_FA2_QROW32_B1_PRODUCTION_ARMS"]
    )


def test_contract_keeps_tier_b_out_of_the_production_allowlist() -> None:
    contract = _contract()
    assert contract.QROW32_B1_TIER_B_ARMS == (SPLITK_ARM,)
    resolve = contract._expected_runtime_fa2_identity
    # The live route resolves the split-K identity -- Arm S's refusal (1).
    assert resolve({
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM": SPLITK_ARM,
        "FR13_FA2_QROW32_B1_SO_SHA256": contract.QROW32_B1_SPLITK_FA2_SHA256,
    }) == (
        contract.QROW32_B1_SPLITK_FA2_SIZE,
        contract.QROW32_B1_SPLITK_FA2_SHA256,
    )
    # It no longer falls through to split2's pins, which is what refused.
    assert (
        contract.QROW32_B1_SPLITK_FA2_SHA256
        != contract.QROW32_B1_SPLIT2_FA2_SHA256
    )
    # But the production door is shut.
    with pytest.raises(contract.ContractError, match="must be empty, nosplit"):
        resolve({
            "FR13_FA2_QROW32_B1_PRODUCTION_ARM": SPLITK_ARM,
            "FR13_FA2_QROW32_B1_SO_SHA256": (
                contract.QROW32_B1_SPLITK_FA2_SHA256
            ),
        })
    # And a split-K live arm pointed at another arm's binary is refused.
    with pytest.raises(contract.ContractError, match="not the pinned candidate"):
        resolve({
            "FR13_FA2_QROW32_B1_LIVE_AB_ARM": SPLITK_ARM,
            "FR13_FA2_QROW32_B1_SO_SHA256": (
                contract.QROW32_B1_GQA_PAIR_FA2_SHA256
            ),
        })


# ----------------------------------------------------- the serving resolver


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, (None, None)),
        ({"FR13_FA2_QROW32_B1_PRODUCTION_ARM": "gqa_pair"}, ("gqa_pair", "A")),
        ({"FR13_FA2_QROW32_B1_PRODUCTION_ARM": "nosplit"}, ("nosplit", "A")),
        # Named as a live arm is the SHADOW route it always was; serving the
        # candidate needs a second, explicit flag.
        ({"FR13_FA2_QROW32_B1_LIVE_AB_ARM": SPLITK_ARM}, (None, None)),
        (
            {
                "FR13_FA2_QROW32_B1_LIVE_AB_ARM": SPLITK_ARM,
                "FR13_FA2_QROW32_B1_TIER_B_SERVE": "1",
            },
            (SPLITK_ARM, "B"),
        ),
        # The serve flag does nothing for an arm that is not tier-b.
        (
            {
                "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "gqa_pair",
                "FR13_FA2_QROW32_B1_TIER_B_SERVE": "1",
            },
            (None, None),
        ),
    ],
)
def test_serving_arm_resolution(monkeypatch, env, expected) -> None:
    namespace = _selectors()
    for key in (
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM",
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM",
        "FR13_FA2_QROW32_B1_TIER_B_SERVE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert namespace["_fr13_fa2_qrow32_b1_serving_arm"]() == expected


def test_serve_flag_must_be_exactly_zero_or_one(monkeypatch) -> None:
    namespace = _selectors()
    monkeypatch.setenv("FR13_FA2_QROW32_B1_LIVE_AB_ARM", SPLITK_ARM)
    monkeypatch.setenv("FR13_FA2_QROW32_B1_TIER_B_SERVE", "true")
    with pytest.raises(RuntimeError, match="exactly 0 or 1"):
        namespace["_fr13_fa2_qrow32_b1_serving_arm"]()


# ------------------------------------------------------------- the credential


def test_credential_reduces_and_validates(tmp_path: Path) -> None:
    path, payload = _credential(tmp_path)
    verdict = _validate(path)
    assert verdict["tier"] == "B"
    assert verdict["arm"] == SPLITK_ARM
    assert verdict["grants"].startswith("live-A/B serving only")
    assert payload["bounds_evaluation"]["bounds_passed"] is True
    assert payload["determinism"]["all_cases_bitwise_identical"] is True
    assert payload["determinism"]["cross_process_digests_identical"] is True
    assert payload["determinism"]["processes"] == 2
    # Every pre-registered bound is evaluated, none silently skipped.
    assert [b["id"] for b in verdict["bounds"]["bounds"]] == [
        f"B{i}" for i in range(1, 10)
    ]


def test_validator_recomputes_the_verdict_and_ignores_a_forged_one(
    tmp_path: Path,
) -> None:
    """A credential records measurements; the verdict is not its to declare."""
    # Make a measurement fail the bound, and forge the verdict to say PASS.
    path, payload = _credential(
        tmp_path,
        **{
            "measurements.output_ulp_le_2_fraction": 0.10,
            "bounds_evaluation.bounds_passed": True,
        },
    )
    assert payload["bounds_evaluation"]["bounds_passed"] is True
    with pytest.raises(ValueError, match="does not clear its pre-registered"):
        _validate(path)


def test_validator_refuses_a_body_the_digest_does_not_cover(
    tmp_path: Path,
) -> None:
    path, _ = _credential(
        tmp_path, **{"credential_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="digest does not cover its body"):
        _validate(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("so_sha256", "9" * 64),
        ("source_closure_sha256", "9" * 64),
        ("fa2_head", "9" * 40),
        ("sass_digest_sha256", "9" * 64),
        ("baseline_sass_digest_sha256", "9" * 64),
        ("source_commit", "9" * 40),
        ("patch_source_sha256", "9" * 64),
        ("bounds_sha256", "9" * 64),
        ("so_size", 1),
    ],
)
def test_credential_is_bound_on_every_identity_field(
    tmp_path: Path, field, value
) -> None:
    """A credential that outlives a rebuild authorises numerics nobody measured."""
    path, _ = _credential(tmp_path, **{f"identity.{field}": value})
    with pytest.raises(ValueError, match="does not bind this arm"):
        _validate(path)


def test_credential_must_name_every_binding_field(tmp_path: Path) -> None:
    sidecar = _sidecar()
    path, payload = _credential(tmp_path)
    for field in sidecar.TIERB_BINDING_FIELDS:
        body = json.loads(path.read_text())
        del body["identity"][field]
        body["credential_sha256"] = sidecar.tierb_credential_digest(body)
        stripped = tmp_path / "stripped.json"
        stripped.write_text(json.dumps(body, sort_keys=True), encoding="ascii")
        with pytest.raises(ValueError, match="is not bound on"):
            _validate(stripped)


@pytest.mark.parametrize(
    "flag", ["all_cases_bitwise_identical", "cross_process_digests_identical"]
)
def test_determinism_is_a_hard_gate(tmp_path: Path, flag) -> None:
    path, _ = _credential(tmp_path, **{f"determinism.{flag}": False})
    with pytest.raises(ValueError, match="determinism gate did not pass"):
        _validate(path)


def test_probe_below_the_pre_registered_floor_is_refused(
    tmp_path: Path,
) -> None:
    """A bound is only as strong as the probe that tested it."""
    for field, value in (
        ("seeds", 2),
        ("seq_lens", [20480, 23000]),
        ("determinism_reps", 2),
        ("determinism_processes", 1),
        ("operand_scale", "legacy0p1"),
        ("output_elements", 1000),
        ("exact_seeds", 1),
    ):
        path, _ = _credential(tmp_path, **{f"probe.{field}": value})
        with pytest.raises(ValueError, match="weaker than the pre-registered"):
            _validate(path)


def test_bounds_file_must_be_the_pre_registered_bytes(tmp_path: Path) -> None:
    """The pre-registration is only meaningful if it cannot be edited later."""
    sidecar = _sidecar()
    tampered = tmp_path / "bounds.json"
    body = json.loads(BOUNDS.read_text())
    for spec in body["bounds"]:
        if spec["id"] == "B2":
            spec["bound"] = 0.01
    tampered.write_text(json.dumps(body, sort_keys=True), encoding="ascii")
    with pytest.raises(ValueError, match="not the pre-registered file"):
        sidecar.load_tierb_bounds(tampered)
    assert sidecar.load_tierb_bounds(BOUNDS)["arm"] == SPLITK_ARM


# ---------------------------------------------------------------- the reducer


def test_reducer_requires_two_independent_processes() -> None:
    reducer = _reducer()
    probe = json.loads(PROBE_A.read_text())
    with pytest.raises(ValueError, match="at least two independent probe"):
        reducer.reduce_probes([probe], arm=SPLITK_ARM)


def test_reducer_refuses_duplicate_process_tags() -> None:
    reducer = _reducer()
    probe = json.loads(PROBE_A.read_text())
    with pytest.raises(ValueError, match="duplicate tags"):
        reducer.reduce_probes([probe, json.loads(PROBE_A.read_text())],
                              arm=SPLITK_ARM)


def test_reducer_refuses_when_processes_disagree() -> None:
    """The cross-process claim is checked, not trusted."""
    reducer = _reducer()
    a = json.loads(PROBE_A.read_text())
    b = json.loads(PROBE_B.read_text())
    b["determinism"][0]["output_sha16"] = "deadbeefdeadbeef"
    with pytest.raises(ValueError, match="differ ACROSS processes"):
        reducer.reduce_probes([a, b], arm=SPLITK_ARM)


def test_reducer_refuses_a_probe_of_a_different_binary(tmp_path: Path) -> None:
    reducer, sidecar, contract = _reducer(), _sidecar(), _contract()
    a = json.loads(PROBE_A.read_text())
    b = json.loads(PROBE_B.read_text())
    a["so_sha256"] = b["so_sha256"] = "7" * 64
    with pytest.raises(ValueError, match="did not measure the pinned binary"):
        reducer.build_credential(
            [a, b], arm=SPLITK_ARM, source_commit=FAKE_COMMIT,
            patch_source_sha256=FAKE_PATCH_SHA,
            bounds=sidecar.load_tierb_bounds(BOUNDS),
            bounds_sha256=sidecar.TIERB_BOUNDS_SHA256,
            sidecar=sidecar, contract=contract,
        )


def test_reducer_refuses_a_probe_of_a_different_arm() -> None:
    reducer = _reducer()
    a = json.loads(PROBE_A.read_text())
    b = json.loads(PROBE_B.read_text())
    a["candidate_arm"] = b["candidate_arm"] = "gqa_pair"
    with pytest.raises(ValueError, match="not 'gqa_pair_splitk'"):
        reducer.reduce_probes([a, b], arm=SPLITK_ARM)


def test_reduced_measurements_match_the_banked_probe() -> None:
    """The reduction is arithmetic on the artifacts, not a restatement."""
    reducer = _reducer()
    probes = [json.loads(PROBE_A.read_text()), json.loads(PROBE_B.read_text())]
    _, measured, probe = reducer.reduce_probes(probes, arm=SPLITK_ARM)
    summary = probes[0]["characterization_summary"]["captured"]
    hist = summary["output_ulp_histogram"]
    total = sum(hist.values())
    assert measured["output_ulp_le_2_fraction"] == pytest.approx(
        (hist["0"] + hist["1"] + hist["2"]) / total
    )
    assert measured["lse_max_ulp"] == summary["lse_max_ulp"]
    assert measured["nonfinite_disagreements"] == 0
    assert probe["operand_scale"] == "captured"
    assert probe["output_elements"] == total
    # B6's two halves must come from the same cases, or the comparison is
    # between different experiments.
    exact = [
        r for r in probes[0]["exact_reference"] if r["scale"] == "captured"
    ]
    assert measured["argmax_flips_vs_exact_candidate"] == sum(
        r[SPLITK_ARM]["argmax_flips_vs_exact"] for r in exact
    )
    assert measured["argmax_flips_vs_exact_incumbent"] == sum(
        r["gqa_pair"]["argmax_flips_vs_exact"] for r in exact
    )


# ----------------------------------------------- Arm S refusal (1), closed


LAUNCHERS = (
    REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh",
    REPO / "scripts/fr14_armb_leg3_launch_nomiddleware.sh",
)


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_in_container_qualification_map_names_the_tier_b_arm(launcher) -> None:
    """Arm S died here: the bash pin case admitted the arm, this map did not.

    The map is executed inside the container, where the bash case has already
    passed, so a missing entry does not refuse -- it answers with ANOTHER
    arm's pins and the boot dies at "binary identity is not qualified" with
    nothing pointing at the cause.
    """
    text = launcher.read_text()
    assert '"gqa_pair_splitk": (' in text
    assert "contract.QROW32_B1_SPLITK_FA2_SHA256," in text
    assert "contract.QROW32_B1_SPLITK_FA2_SIZE," in text
    # A tier-b arm that resolved a non-tier-b binary must refuse rather than
    # serve someone else's kernel under this arm's name.
    assert "contract.QROW32_B1_TIER_B_ARMS" in text
    assert "tier-b arm resolved a non-tier-b binary" in text
    # And the fall-through this entry was missing from is GONE: the map now
    # has no pins-valued default at all, so an arm nobody wrote a key for
    # refuses instead of inheriting split2's identity (Arm S, 17th site).
    assert "}.get(b1_pin_arm)" in text
    assert "if expected is None:" in text
    assert "has no pinned binary identity" in text
    assert "QROW32_B1_SPLIT2_FA2_SIZE,\n        ),\n    )" not in text
    # nosplit and split2 are NAMED rather than defaulted.
    assert '"nosplit": (' in text and '"split2": (' in text


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_launcher_embedded_python_still_parses(launcher) -> None:
    """The map lives inside a heredoc, so bash -n cannot see a syntax error."""
    import ast
    import re

    lines = launcher.read_text().splitlines(keepends=True)
    target = next(
        i for i, line in enumerate(lines)
        if "binary identity is not qualified" in line and "B1" in line
    )
    opener = re.compile(r"<<\s*'?([A-Z]+)'?\s*$")
    start = max(i for i in range(target) if opener.search(lines[i]))
    tag = opener.search(lines[start]).group(1)
    end = next(i for i in range(target, len(lines)) if lines[i].strip() == tag)
    ast.parse("".join(lines[start + 1:end]))


def test_earned_credential_is_present_and_validates() -> None:
    """The banked credential must still clear its own bounds on demand.

    Not a re-run of the gate -- the measurements are fixed. This is the check
    that the artifact in the tree is a credential the validator accepts, so a
    later edit to the bounds, the schema or the binding fields cannot leave a
    stale PASS sitting in results/ looking authoritative.
    """
    sidecar = _sidecar()
    path = (
        REPO
        / "results/fr14_nvfp4_port_20260816/fr14_splitk_tierb_credential.json"
    )
    if not path.is_file():
        pytest.skip("credential not yet earned at this HEAD")
    payload = json.loads(path.read_text())
    assert payload["schema"] == sidecar.TIERB_SCHEMA
    assert payload["arm"] == SPLITK_ARM and payload["tier"] == "B"
    assert payload["bounds_evaluation"]["bounds_passed"] is True
    assert payload["probe_strength"]["probe_strength_passed"] is True
    assert payload["determinism"]["cross_process_digests_identical"] is True
    assert payload["determinism"]["processes"] >= 2
    assert (
        sidecar.tierb_credential_digest(payload)
        == payload["credential_sha256"]
    )
    assert payload["identity"]["bounds_sha256"] == sidecar.TIERB_BOUNDS_SHA256
    # It authorises the pinned binary and nothing else.
    contract = _contract()
    assert payload["identity"]["so_sha256"] == (
        contract.QROW32_B1_SPLITK_FA2_SHA256
    )
    assert payload["identity"]["so_size"] == contract.QROW32_B1_SPLITK_FA2_SIZE
    # Re-validating requires the commit it was earned at, which is the
    # property that makes it expire when HEAD moves.
    verdict = sidecar.validate_tierb_credential(
        path,
        arm=SPLITK_ARM,
        expected_candidate_sha256=contract.QROW32_B1_SPLITK_FA2_SHA256,
        expected_source_commit=payload["identity"]["source_commit"],
        expected_patch_source_sha256=payload["identity"]["patch_source_sha256"],
        bounds_path=BOUNDS,
    )
    assert verdict["grants"].startswith("live-A/B serving only")
    with pytest.raises(ValueError, match="does not bind this arm"):
        sidecar.validate_tierb_credential(
            path,
            arm=SPLITK_ARM,
            expected_candidate_sha256=contract.QROW32_B1_SPLITK_FA2_SHA256,
            expected_source_commit="a" * 40,
            expected_patch_source_sha256=payload["identity"][
                "patch_source_sha256"
            ],
            bounds_path=BOUNDS,
        )


# --------------------------------------- the 17th site: identity resolution


def _identities():
    namespace = _selectors()
    return (
        namespace["_fr13_fa2_qrow32_b1_identity"],
        namespace["_FR13_FA2_QROW32_B1_IDENTITIES"],
        namespace["_FR13_FA2_QROW32_B1_ARMS"],
    )


def test_every_registered_arm_has_its_own_pinned_identity() -> None:
    """The invariant the 17th site broke.

    Arm S's fifth boot selected gqa_pair_splitk, reached
    _fr13_fa2_qrow32_b1_identity, and got SPLIT2's pins back -- because the
    resolver branched on two arm names and then returned the incumbent's
    identity to everything else. Nothing served only because the environment's
    declared sha happened not to match what came back. That is an accident, not
    a guard.
    """
    identity, table, arms = _identities()
    assert set(table) == set(arms), (
        "the identity table and the arm registry must cover each other exactly"
    )
    # No two arms may share pins unless they genuinely share a binary.
    by_sha: dict[str, list[str]] = {}
    for arm in sorted(table):
        by_sha.setdefault(identity(arm)["candidate_sha256"], []).append(arm)
    assert by_sha[
        "28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857"
    ] == [SPLITK_ARM], "split-K's binary must belong to split-K alone"
    assert by_sha[
        "3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae"
    ] == ["gqa_pair"]
    # nosplit and split2 genuinely ship in one .so; that is the only sharing.
    assert sorted(
        by_sha["a9d8a6887b8b27b3a83af60bba7945eb66caff174ba710c2ee2aea92b8e7081a"]
    ) == ["nosplit", "split2"]


@pytest.mark.parametrize(
    "unknown",
    ["gqa_pair_splitk2", "splitk", "gqa_pair ", "GQA_PAIR", "", "nosplit2",
     "gqa_pair_splitk_disabled"],
)
def test_an_unknown_arm_refuses_instead_of_inheriting_pins(unknown) -> None:
    """The half that makes an eighteenth site impossible.

    Adding an arm to the registry without adding it here must fail LOUDLY at
    selection, not quietly attest somebody else's artifact.
    """
    identity, _table, _arms = _identities()
    with pytest.raises(RuntimeError, match="has no pinned identity for arm"):
        identity(unknown)


def test_the_splitk_identity_carries_both_sass_digests() -> None:
    """This arm's .so sha is not rebuild-reproducible, so the sha is not enough.

    Two links from one closure produced two .so hashes at an identical size.
    The SASS digests are what attest that the KERNEL is the characterized one,
    and the baseline digest is what keeps "the split-K header edits are inert
    at Split=false" a measurement. The in-container resolver carried neither
    until the 17th site; the launcher's bash pin case had both all along.
    """
    identity, _table, _arms = _identities()
    splitk = identity(SPLITK_ARM)
    assert splitk["sass_digest_sha256"] == (
        "3f24d70dce2ff70ad9209bad5af2a93cc39453df529cb298e4476cbfbfd80b9e"
    )
    assert splitk["baseline_sass_digest_sha256"] == (
        "fa01f98840420b9c0177d06297aacabb0ed5e00c674511fdaa4aa618c3473470"
    )
    # And no other arm claims them.
    for arm in ("nosplit", "split2", "visibility", "gqa_pair"):
        assert "sass_digest_sha256" not in identity(arm)


def test_require_identity_checks_the_sass_digests_container_side(
    monkeypatch,
) -> None:
    """The bash pin case checks these on the host; this is the other half."""
    namespace = _selectors()
    require = namespace["_fr13_fa2_qrow32_b1_require_identity"]
    env = {
        "FR13_FA2_QROW32_B1_SO_SHA256": SPLITK_SO_SHA256,
        "FR13_FA2_QROW32_B1_SO_SIZE": "300123792",
        "FR13_FA2_QROW32_B1_FA2_HEAD": (
            "29210221863736a08f71a866459e368ad1ac4a95"
        ),
        "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256": (
            "4ed00909cef7ea83849f897018ea4f6a14119b8d160927af426938920c170878"
        ),
        "FR13_FA2_QROW32_B1_SOURCE_COMMIT": "b" * 40,
        "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256": "c" * 64,
        "FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST": (
            "3f24d70dce2ff70ad9209bad5af2a93cc39453df529cb298e4476cbfbfd80b9e"
        ),
        "FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST": (
            "fa01f98840420b9c0177d06297aacabb0ed5e00c674511fdaa4aa618c3473470"
        ),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert require(SPLITK_ARM)[0] == SPLITK_SO_SHA256

    # Each digest is load-bearing on its own.
    for key in (
        "FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST",
        "FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST",
    ):
        monkeypatch.setenv(key, "9" * 64)
        with pytest.raises(RuntimeError, match="drifted for this arm"):
            require(SPLITK_ARM)
        monkeypatch.setenv(key, env[key])

    # And the identity itself still refuses a mismatched artifact.
    monkeypatch.setenv("FR13_FA2_QROW32_B1_SO_SHA256", "d" * 64)
    with pytest.raises(RuntimeError, match="pinned identity drifted"):
        require(SPLITK_ARM)


def test_identity_fraud_is_impossible_for_a_selected_but_unpinned_arm(
    monkeypatch,
) -> None:
    """The fraud the 17th refusal prevented, reproduced as a test.

    Serve arm X while attesting arm Y. Previously reachable by selecting any
    arm the resolver had no branch for and declaring the incumbent's identity;
    now the resolver refuses before any comparison happens.
    """
    namespace = _selectors()
    require = namespace["_fr13_fa2_qrow32_b1_require_identity"]
    # The incumbent's identity, declared in full and self-consistent.
    for key, value in (
        ("FR13_FA2_QROW32_B1_SO_SHA256",
         "a9d8a6887b8b27b3a83af60bba7945eb66caff174ba710c2ee2aea92b8e7081a"),
        ("FR13_FA2_QROW32_B1_SO_SIZE", "300154616"),
        ("FR13_FA2_QROW32_B1_FA2_HEAD",
         "29210221863736a08f71a866459e368ad1ac4a95"),
        ("FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256",
         "22b8c2016443a151bf50f62166f7cc3b9ce45137138d948b76fdfded74c395ff"),
        ("FR13_FA2_QROW32_B1_SOURCE_COMMIT", "b" * 40),
        ("FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256", "c" * 64),
    ):
        monkeypatch.setenv(key, value)
    # A registered arm still resolves its own identity, so this is a real
    # environment and not a broken one.
    assert require("nosplit")[0].startswith("a9d8a688")
    # An arm with no pinned identity refuses -- it does NOT inherit the
    # incumbent's and go on to serve.
    with pytest.raises(RuntimeError, match="has no pinned identity for arm"):
        require("gqa_pair_splitk_v2")


# ------------------------- the 18th site: ARMED is not ENGAGED


def test_serving_hook_installs_for_a_tier_b_serve_not_only_production() -> None:
    """THE 18th SITE, and it was mine.

    Round 6 armed a tier-b serve. The launcher spells that as a LIVE arm, so
    its elif chain passed --fixed32-query-tile32-b1-live-ab and never the
    production flag -- and the install of
    _fr13_fa2_qrow32_b1_production_begin, the only thing that retags the
    operand on the SERVED path, is gated on the production flag. The resolver
    had been taught about tier-b arms; the code that installs its CALLER had
    not. 395 seconds of task, a clean eyeball, and the incumbent kernel.
    """
    patcher = PATCHER.read_text()
    marker = "if (\n        fixed32_query_tile32_b1_production\n        or fixed32_query_tile32_b1_tier_b_serve\n    ) and ("
    assert marker in patcher, (
        "the serving call site must install for a tier-b serve as well as for "
        "a production arm"
    )
    # And the capture-end hook, which is what writes the engagement record.
    assert "if fixed32_query_tile32_b1_tier_b_serve:" in patcher
    assert "_patch_cuda_graph_qrow32_b1_production(\n                cuda_graph_path\n            )" in patcher


def test_the_tier_b_serve_flag_is_a_modifier_not_a_selector() -> None:
    """It composes with the live-A/B selector; it does not replace it."""
    import subprocess

    out = subprocess.run(
        [sys.executable, str(PATCHER), "--skip-source", "--skip-python",
         "--fixed32-query-tile32-b1-tier-b-serve"],
        capture_output=True, text=True,
    )
    assert out.returncode != 0
    assert "requires --fixed32-query-tile32-b1-live-ab" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_launcher_passes_the_serve_flag_when_armed(launcher) -> None:
    text = launcher.read_text()
    assert (
        '$(if [[ "${FR13_FA2_QROW32_B1_TIER_B_SERVE:-0}" == "1" ]]; then '
        "printf '%s' '--fixed32-query-tile32-b1-tier-b-serve'; fi)" in text
    ), f"{launcher.name} never passes the serving flag"
    # It must be a SEPARATE expansion, not another arm of the elif chain that
    # already swallowed it once.
    start = text.index("--fixed32-query-tile16-live-ab")
    chain = text[text.rindex("\n", 0, start) + 1: text.index("\n", start)]
    assert "tier-b-serve" not in chain, (
        "the serve flag must not live inside the mutually-exclusive selector "
        "chain -- that is the elif that hid the 18th site"
    )
    # And it must be guarded against .lumo.local.env.
    assert "\n  FR13_FA2_QROW32_B1_TIER_B_SERVE\n" in text


def test_engagement_is_counted_at_the_retag_not_read_from_the_environment() -> None:
    """The record must report what RAN.

    Round 6's artifact said tier_b_serving: true and "candidate output served
    (tier-b)". Both were computed from environment variables, so they asserted
    a serve that never happened and the campaign believed them. This is the
    same defect the draft-vocabulary identity fix already named -- two
    hardcodings agreeing with each other while both disagreed with reality.
    """
    blob = _module(PATCHER, "fr14_serve_patcher").FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS
    assert "_FR13_FA2_QROW32_B1_TIER_B_ENGAGEMENT" in blob
    # counted where the retag happens
    assert '_FR13_FA2_QROW32_B1_TIER_B_ENGAGEMENT["calls"] += 1' in blob
    # the verdict reads the counter, not the environment
    assert (
        '"tier_b_serving": bool(\n            tier_b and '
        '_FR13_FA2_QROW32_B1_TIER_B_ENGAGEMENT["calls"] > 0\n        ),' in blob
    )
    # armed and engaged are reported separately
    assert '"tier_b_serve_armed"' in blob
    assert '"tier_b_engagement"' in blob


def test_armed_but_never_engaged_is_a_refusal() -> None:
    """Round 6's exact state must fail the run, not decorate it."""
    blob = _module(PATCHER, "fr14_serve_patcher2").FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS
    assert "tier-b serve is ARMED but never ENGAGED" in blob
    assert "--fixed32-query-tile32-b1-tier-b-serve" in blob
    # The refusal must come BEFORE the pass/fail branch, so an otherwise-clean
    # shadow comparison cannot report PASS for a serve that did not happen.
    assert blob.index("ARMED but never ENGAGED") < blob.index(
        "FR13 qrow32 B1 tier-b live gate failed"
    )


def test_round6_artifact_is_diagnosable_from_the_new_fields() -> None:
    """The banked round-6 record, read through the new schema.

    It carries tier_b_serving: true from the OLD env-derived field. Under the
    new fields the same run would have reported armed=true, engagement calls=0,
    serving=false -- and refused. This test pins the diagnosis so the artifact
    cannot be re-read as a successful serve later.
    """
    banked = (
        REPO / "output/fr14_promoab_Sr6_20260819T005801Z"
        / "hydra27_fixed32_promoab_S_r6/logs"
        / "fr13_fa2_qrow32_b1_gqa_pair_splitk_live_paged_ab.json"
    )
    if not banked.is_file():
        pytest.skip("round-6 run output not present")
    payload = json.loads(banked.read_text())
    # What it claimed.
    assert payload["tier_b_serving"] is True
    assert payload["served_return"] == "candidate output served (tier-b)"
    # What it could not show, because the field did not exist: any observation
    # of engagement. Its absence IS the finding.
    assert "tier_b_engagement" not in payload
    assert "tier_b_serve_armed" not in payload
    # And the run wrote no production engagement record, which is the
    # independent evidence that the serving hook never ran.
    engagement = banked.parent / "fr13_fa2_qrow32_b1_production_engagement.json"
    assert not engagement.exists(), (
        "a production engagement record would mean the hook DID run and the "
        "diagnosis needs revisiting"
    )
