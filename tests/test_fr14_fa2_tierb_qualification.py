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
import re
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
    import logging
    import os

    import torch

    patcher = _module(PATCHER, "fr14_tierb_test_patcher")
    # The blob runs inside vLLM, where os/torch/logger are already module
    # globals. Providing them here is what lets the SERVED PATH be executed on
    # CPU instead of merely grepped -- which is the difference between finding
    # sites 18-20 in milliseconds and finding them in five GPU boots.
    namespace: dict = {
        "os": os, "torch": torch, "logger": logging.getLogger("fr14_tierb"),
    }
    exec(  # noqa: S102 - the emitted source is the thing under test
        compile(
            "import os\nimport torch\n"
            + patcher.FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS,
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
        # The live-A/B selector is the SHADOW route and only that. It does not
        # serve, with or without any modifier -- which is the pun that cost
        # sites 17 through 20.
        ({"FR13_FA2_QROW32_B1_LIVE_AB_ARM": SPLITK_ARM}, (None, None)),
        # Tier-B serving has its own first-class name.
        ({"FR13_FA2_QROW32_B1_TIER_B_ARM": SPLITK_ARM}, (SPLITK_ARM, "B")),
    ],
)
def test_serving_arm_resolution(monkeypatch, env, expected) -> None:
    namespace = _selectors()
    for key in (
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM",
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM",
        "FR13_FA2_QROW32_B1_TIER_B_ARM",
        "FR13_FA2_QROW32_B1_TIER_B_SERVE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert namespace["_fr13_fa2_qrow32_b1_serving_arm"]() == expected


def test_the_retired_piggyback_spelling_fails_loudly(monkeypatch) -> None:
    """It must not degrade to a shadow run -- round 6 did that for 395 s."""
    namespace = _selectors()
    monkeypatch.delenv("FR13_FA2_QROW32_B1_TIER_B_ARM", raising=False)
    monkeypatch.setenv("FR13_FA2_QROW32_B1_LIVE_AB_ARM", SPLITK_ARM)
    monkeypatch.setenv("FR13_FA2_QROW32_B1_TIER_B_SERVE", "1")
    with pytest.raises(RuntimeError, match="TIER_B_SERVE is retired"):
        namespace["_fr13_fa2_qrow32_b1_serving_arm"]()


@pytest.mark.parametrize(
    "env,expect",
    [
        ({"FR13_FA2_QROW32_B1_TIER_B_ARM": "gqa_pair"},
         "must be empty or one of"),
        ({"FR13_FA2_QROW32_B1_TIER_B_ARM": SPLITK_ARM,
          "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "gqa_pair"},
         "mutually exclusive"),
        # A serve and a single-instance diagnostic cannot be one boot. Letting
        # them coexist is how the identity contradiction at site 20 arose.
        ({"FR13_FA2_QROW32_B1_TIER_B_ARM": SPLITK_ARM,
          "FR13_FA2_QROW32_B1_LIVE_AB_ARM": SPLITK_ARM},
         "mutually exclusive"),
    ],
)
def test_tier_b_selector_refuses_incoherent_modes(monkeypatch, env, expect):
    namespace = _selectors()
    for key in (
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM",
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM",
        "FR13_FA2_QROW32_B1_TIER_B_ARM",
        "FR13_FA2_QROW32_B1_TIER_B_SERVE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match=expect):
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
    # The third twin. This list was a 2-tuple until the universal resolver
    # test counted its own answers and found four where six were expected --
    # the same one-twin-short mistake that let fr14_leg3 drift a whole arm
    # behind in the paired-contract family.
    REPO / "scripts/fr14_leg3_launch_nomiddleware.sh",
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
    # A tier-B SERVE takes the SERVING hook -- the same branch a production arm
    # takes, because it is a serve. It used to reach the chain as a modifier of
    # the live-A/B branch, which is how it inherited the shadow mode's plumbing
    # and none of the serving mode's.
    assert "elif fixed32_query_tile32_b1_tier_b_serve:" in patcher
    assert patcher.index("elif fixed32_query_tile32_b1_tier_b_serve:") < (
        patcher.index("elif fixed32_query_tile32_b1_production:")
    )


def test_the_tier_b_serve_flag_is_a_first_class_selector() -> None:
    """It stands alone, and it is mutually exclusive with the shadow selector.

    THE DESIGN DECISION behind sites 17-20: tier-B serving was spelled as the
    live-A/B selector plus a modifier, so every gate in the tree had to be read
    twice -- once as "shadow", once as "serve" -- and one of the two readings
    was missed four times running. A mode that is neither production nor shadow
    needs its own name.
    """
    import subprocess

    alone = subprocess.run(
        [sys.executable, str(PATCHER), "--skip-source", "--skip-python",
         "--fixed32-query-tile32-b1-tier-b-serve"],
        capture_output=True, text=True,
    )
    assert alone.returncode == 0, alone.stderr
    assert "'fixed32_query_tile32_b1_tier_b_serve': True" in alone.stdout

    together = subprocess.run(
        [sys.executable, str(PATCHER), "--skip-source", "--skip-python",
         "--fixed32-query-tile32-b1-tier-b-serve",
         "--fixed32-query-tile32-b1-live-ab"],
        capture_output=True, text=True,
    )
    assert together.returncode != 0
    assert "mutually exclusive" in together.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_launcher_passes_the_serve_flag_when_armed(launcher) -> None:
    text = launcher.read_text()
    assert (
        '$(if [[ -n "$FR13_FA2_QROW32_B1_TIER_B_ARM" ]]; then '
        "printf '%s' '--fixed32-query-tile32-b1-tier-b-serve'; fi)" in text
    ), f"{launcher.name} never passes the serving flag"
    # The retired spelling must fail loudly rather than degrade to a shadow run.
    assert "FR13_FA2_QROW32_B1_TIER_B_SERVE is retired" in text
    # And the tier-B arm must reach the pin-arm resolution.
    assert "_FR13_FA2_QROW32_B1_PIN_ARM=$FR13_FA2_QROW32_B1_TIER_B_ARM" in text
    # It must be a SEPARATE expansion, not another arm of the elif chain that
    # already swallowed it once.
    start = text.index("--fixed32-query-tile16-live-ab")
    chain = text[text.rindex("\n", 0, start) + 1: text.index("\n", start)]
    assert "tier-b-serve" not in chain, (
        "the serve flag must not live inside the mutually-exclusive selector "
        "chain -- that is the elif that hid the 18th site"
    )
    # And it must be guarded against .lumo.local.env.
    assert "\n  FR13_FA2_QROW32_B1_TIER_B_ARM\n" in text


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


# ===========================================================================
# 5. THE CPU END-TO-END SERVE. Every site 17-20 was found by booting a GPU
#    server for minutes or hours. Every one of them is reachable from here in
#    milliseconds: the served path's precondition chain is pure Python until
#    the kernel launch, so it can be walked on CPU with fake tensors. What
#    this cannot check is the kernel itself -- and the kernel was never the
#    problem.
# ===========================================================================

import contextlib

CANONICAL_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
TARGET_LAYER = "language_model.model.layers.3.self_attn.attn"


def _served_operands():
    """The exact fixed32 B1 geometry the in-binary gate demands, on CPU."""
    import torch

    query = torch.zeros(32, 24, 256, dtype=torch.bfloat16)
    kv = torch.zeros(3, 2, 1024, 4, 256, dtype=torch.bfloat16)
    key_cache, value_cache = kv[:, 0], kv[:, 1]
    tree_bias = torch.zeros(32, 32, dtype=torch.float32)
    return dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        cu_seqlens_q=torch.tensor([0, 32], dtype=torch.int32),
        max_seqlen_q=32,
        seqused_k=torch.tensor([2048], dtype=torch.int32),
        max_seqlen_k=2048,
        causal=False,
        window_size=None,
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        softcap=0.0,
        num_splits=0,
        tree_bias=tree_bias,
    )


class _Layer:
    def __init__(self, name=TARGET_LAYER):
        self.layer_name = name


@contextlib.contextmanager
def _tier_b_env(monkeypatch, credential_path, **overrides):
    """Everything a tier-B SERVE must have, and nothing a tier-A serve needs.

    Deliberately assembled from scratch rather than copied from a tier-A
    fixture: the point of the walk is that the two modes' preconditions are
    different sets, and a fixture that inherited tier-A's would hide exactly
    the gates this is meant to find.
    """
    env = {
        "FR13_FA2_QROW32_B1_TIER_B_ARM": SPLITK_ARM,
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "",
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "",
        "FR13_FA2_QROW32_B1_INTERNAL_ATTESTED": "1",
        "FR13_FA2_QROW32_B1_SO_SHA256": SPLITK_SO_SHA256,
        "FR13_FA2_QROW32_B1_SO_SIZE": "300123792",
        "FR13_FA2_QROW32_B1_FA2_HEAD": (
            "29210221863736a08f71a866459e368ad1ac4a95"
        ),
        "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256": (
            "4ed00909cef7ea83849f897018ea4f6a14119b8d160927af426938920c170878"
        ),
        "FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST": (
            "3f24d70dce2ff70ad9209bad5af2a93cc39453df529cb298e4476cbfbfd80b9e"
        ),
        "FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST": (
            "fa01f98840420b9c0177d06297aacabb0ed5e00c674511fdaa4aa618c3473470"
        ),
        "FR13_FA2_QROW32_B1_EXACT4_TASK_IDS": ",".join(CANONICAL_TASK_IDS),
        "FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256": EXACT4_SUBSET_SHA256,
        "FR13_DRAFT_VOCAB_ROOT": "0",
        "FR13_DRAFT_VOCAB_K": "0",
        "FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE": "full_vocab",
        "ENFORCE_EAGER": "1",
    }
    payload = json.loads(Path(credential_path).read_text())
    env["FR13_FA2_QROW32_B1_SOURCE_COMMIT"] = payload["identity"][
        "source_commit"
    ]
    env["FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256"] = payload["identity"][
        "patch_source_sha256"
    ]
    env["FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL"] = str(credential_path)
    env["FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_SHA256"] = __import__(
        "hashlib"
    ).sha256(Path(credential_path).read_bytes()).hexdigest()
    env.update(overrides)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    yield env


def _install_fake_vllm(monkeypatch):
    """The served path consults vLLM's gdn module for capture scope.

    Stubbed rather than skipped: the point of the CPU walk is to execute the
    real precondition chain, and a `pytest.skip` when vllm is absent would
    have left sites 18-20 exactly as undetectable as they were.
    """
    import sys
    import types

    gdn = types.ModuleType("gdn_linear_attn")
    gdn._FR13_FIXED32_PROFILE_CAPTURE_SCOPE = None
    gdn._FR13_FIXED32_CAPTURE_CONTEXT = None
    gdn._FR13_FIXED32_OBSERVED_CURRENT = None
    gdn._fr13_fixed32_observed_event_active = lambda: False
    mods = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType(
            "vllm.model_executor.layers"),
        "vllm.model_executor.layers.mamba": types.ModuleType(
            "vllm.model_executor.layers.mamba"),
    }
    mods["vllm.model_executor.layers.mamba"].gdn_linear_attn = gdn
    mods["vllm.model_executor.layers.mamba.gdn_linear_attn"] = gdn
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)
    return gdn


def _fresh_selectors():
    """A pristine namespace: the engagement counter must start at zero."""
    namespace = _selectors()
    namespace["_FR13_FA2_QROW32_B1_TIER_B_ENGAGEMENT"]["calls"] = 0
    namespace["_FR13_FA2_QROW32_B1_TIER_B_ENGAGEMENT"]["layers"].clear()
    namespace["_FR13_FA2_QROW32_B1_TIER_B_ENGAGEMENT"]["graph_ids"].clear()
    namespace["_FR13_FA2_QROW32_B1_TIER_B_STATE"].clear()
    return namespace


def test_cpu_end_to_end_tier_b_serve_reaches_the_retag(monkeypatch, tmp_path):
    """Walk the whole served-path chain and assert the operand IS retagged.

    Sites 18, 19 and 20 all live on this chain, and all three would have
    failed this test in milliseconds. Site 18 (hook not installed) is the one
    exception -- it is an INSTALL-time defect, covered separately -- which is
    why that test exists too.
    """
    _install_fake_vllm(monkeypatch)
    namespace = _fresh_selectors()
    path, _payload = _credential(tmp_path)
    begin = namespace["_fr13_fa2_qrow32_b1_production_begin"]
    with _tier_b_env(monkeypatch, path):
        selection = begin(layer=_Layer(), **_served_operands())

    assert selection is not None, "the served path returned no selection"
    assert selection["tier"] == "B"
    assert selection["arm"] == SPLITK_ARM
    assert selection["candidate_served"] is True
    # THE POINT: the operand carries the split-K sentinel, so the kernel that
    # runs is the candidate and not the incumbent.
    assert int(selection["tree_bias"].stride(0)) == 1179791671
    assert selection["num_splits"] == 4
    assert selection["sentinel"] == 1179791671
    # And the credential was validated on the way through, not assumed.
    assert selection["tier_b_credential"] is not None
    assert selection["tier_b_credential"]["credential_sha256"]
    # AND THE CONTRACT, in the same loop. Round 10 died at
    # _expected_runtime_fa2_identity while every other link on this chain had
    # already passed, so the walk now executes it here rather than trusting
    # that a resolver two files away agrees.
    contract = _contract()
    assert contract._expected_runtime_fa2_identity({
        "FR13_FA2_QROW32_B1_TIER_B_ARM": SPLITK_ARM,
        "FR13_FA2_QROW32_B1_SO_SHA256": SPLITK_SO_SHA256,
    }) == (300123792, SPLITK_SO_SHA256)
    # Engagement is OBSERVED.
    engagement = namespace["_FR13_FA2_QROW32_B1_TIER_B_ENGAGEMENT"]
    assert engagement["calls"] == 1
    assert engagement["layers"] == {TARGET_LAYER}


def test_cpu_end_to_end_tier_a_serve_is_unchanged(monkeypatch, tmp_path):
    """The tier-A path must be exactly what it was; tier-B added a door."""
    _install_fake_vllm(monkeypatch)
    namespace = _fresh_selectors()
    begin = namespace["_fr13_fa2_qrow32_b1_production_begin"]
    for key, value in {
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "gqa_pair",
        "FR13_FA2_QROW32_B1_TIER_B_ARM": "",
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "",
        "FR13_FA2_QROW32_B1_INTERNAL_ATTESTED": "1",
        "FR13_FA2_QROW32_B1_SO_SHA256": (
            "3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae"
        ),
        "FR13_FA2_QROW32_B1_SO_SIZE": "299815552",
        "FR13_FA2_QROW32_B1_FA2_HEAD": (
            "29210221863736a08f71a866459e368ad1ac4a95"
        ),
        "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256": (
            "172b5e7131841ce45650bb8eea35f0b427ca660ce8f145bd39b55b00a336ebf4"
        ),
        "FR13_FA2_QROW32_B1_SOURCE_COMMIT": "b" * 40,
        "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256": "c" * 64,
        "FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR_SHA256": "d" * 64,
        "FR13_FA2_QROW32_B1_EXACT4_TASK_IDS": ",".join(CANONICAL_TASK_IDS),
        "FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256": EXACT4_SUBSET_SHA256,
        "FR13_DRAFT_VOCAB_ROOT": "0",
        "FR13_DRAFT_VOCAB_K": "0",
        "FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE": "full_vocab",
        "ENFORCE_EAGER": "1",
    }.items():
        monkeypatch.setenv(key, value)
    selection = begin(layer=_Layer(), **_served_operands())
    assert selection["tier"] == "A"
    assert selection["arm"] == "gqa_pair"
    assert int(selection["tree_bias"].stride(0)) == 1179791670
    assert selection["tier_b_credential"] is None
    # tier-A must not touch the tier-B counter
    assert namespace["_FR13_FA2_QROW32_B1_TIER_B_ENGAGEMENT"]["calls"] == 0


@pytest.mark.parametrize(
    "drop,expect",
    [
        # Site 19: the attestation the tier-A block exports and tier-B did not.
        ("FR13_FA2_QROW32_B1_INTERNAL_ATTESTED", "launcher attestation"),
        # Site 20: the canonical exact4 identity a serve must carry.
        ("FR13_FA2_QROW32_B1_EXACT4_TASK_IDS", "exact4 identity drifted"),
        ("FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256", "exact4 identity drifted"),
        # The credential itself.
        ("FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL", "TIER_B_CREDENTIAL"),
        # Binary identity.
        ("FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST", "drifted for this arm"),
    ],
)
def test_each_served_path_precondition_is_load_bearing(
    monkeypatch, tmp_path, drop, expect
):
    """Every gate on the chain, removed one at a time, must refuse.

    This is the walk expressed as a test: it does not assert that the chain
    has some particular length, it asserts that each link actually holds
    weight. A precondition nothing tests is a precondition that will be
    quietly dropped by the next mode that comes along -- which is the whole
    history of sites 17 through 20.
    """
    _install_fake_vllm(monkeypatch)
    namespace = _fresh_selectors()
    path, _payload = _credential(tmp_path)
    begin = namespace["_fr13_fa2_qrow32_b1_production_begin"]
    with _tier_b_env(monkeypatch, path):
        monkeypatch.setenv(drop, "")
        with pytest.raises(RuntimeError, match=expect):
            begin(layer=_Layer(), **_served_operands())


def test_serving_a_tier_b_arm_never_takes_the_byte_gate_branch(
    monkeypatch, tmp_path
):
    """require_same_reduction must not run for tier B, and must for tier A."""
    _install_fake_vllm(monkeypatch)
    namespace = _fresh_selectors()
    path, _payload = _credential(tmp_path)
    calls = []
    real = namespace["_fr13_fa2_qrow32_b1_require_same_reduction"]

    def _spy(arm, reference_num_splits):
        calls.append(arm)
        return real(arm, reference_num_splits)

    namespace["_fr13_fa2_qrow32_b1_require_same_reduction"] = _spy
    begin = namespace["_fr13_fa2_qrow32_b1_production_begin"]
    with _tier_b_env(monkeypatch, path):
        begin(layer=_Layer(), **_served_operands())
    assert calls == [], (
        "the raw-byte reduction check ran on a tier-B serve; it would refuse "
        "num_splits=4 by construction and is not the tier-B contract"
    )


def test_a_non_target_layer_is_refused_on_the_served_path(
    monkeypatch, tmp_path
):
    _install_fake_vllm(monkeypatch)
    namespace = _fresh_selectors()
    path, _payload = _credential(tmp_path)
    begin = namespace["_fr13_fa2_qrow32_b1_production_begin"]
    with _tier_b_env(monkeypatch, path):
        with pytest.raises(RuntimeError, match="layer identity drifted"):
            begin(layer=_Layer("language_model.model.layers.4.self_attn.attn"),
                  **_served_operands())


def test_drifted_geometry_is_refused_on_the_served_path(monkeypatch, tmp_path):
    import torch

    _install_fake_vllm(monkeypatch)
    namespace = _fresh_selectors()
    path, _payload = _credential(tmp_path)
    begin = namespace["_fr13_fa2_qrow32_b1_production_begin"]
    operands = _served_operands()
    operands["query"] = torch.zeros(16, 24, 256, dtype=torch.bfloat16)
    with _tier_b_env(monkeypatch, path):
        with pytest.raises(RuntimeError, match="geometry drifted"):
            begin(layer=_Layer(), **operands)


def test_engagement_record_is_tier_aware_and_reports_what_ran(
    monkeypatch, tmp_path
):
    """The emitter must not KeyError on tier B, and must not hardcode identity.

    Both were live defects: pass_sidecar_sha256 was a bare os.environ[...]
    that only a tier-A boot could satisfy, and draft_vocab_root/k were the
    literals 1 / 65536 -- the same hardcoding the draft-vocabulary red-team
    caught once already.
    """
    _install_fake_vllm(monkeypatch)
    namespace = _fresh_selectors()
    path, _payload = _credential(tmp_path)
    record_fn = namespace["_fr13_fa2_qrow32_b1_production_record"]
    with _tier_b_env(monkeypatch, path):
        record = record_fn(
            arm=SPLITK_ARM, runtime_mode="FULL", graph_id=7,
            graph_signature="sig", layers=[TARGET_LAYER], calls=16, tier="B",
        )
    assert record["tier"] == "B"
    assert record["pass_sidecar_sha256"] is None
    assert record["tier_b_credential_sha256"]
    assert record["arm"] == SPLITK_ARM
    assert record["num_splits"] == 4
    assert record["selector_sentinel"] == 1179791671
    # AS SERVED: this boot is full_vocab, so the record must not claim K64.
    assert record["draft_vocab_root"] == 0
    assert record["draft_vocab_k"] == 0
    assert record["qualification_profile"] == "full_vocab"
    # And the candidate identity is the split-K binary, not the incumbent's.
    assert record["candidate_so_sha256"] == SPLITK_SO_SHA256
    assert record["candidate_so_size"] == 300123792


# --------------------------- site 23: the resolver, and the credential path


def _launcher_python_resolver(launcher, env):
    """Execute the launcher's OWN in-container pin-arm resolver."""
    import re
    import types

    lines = launcher.read_text().splitlines(keepends=True)
    target = next(
        i for i, line in enumerate(lines)
        if "binary identity is not qualified" in line and "B1" in line
    )
    opener = re.compile(r"<<\s*'?([A-Z]+)'?\s*$")
    start = max(i for i in range(target) if opener.search(lines[i]))
    tag = opener.search(lines[start]).group(1)
    end = next(i for i in range(target, len(lines)) if lines[i].strip() == tag)
    body = "".join(lines[start + 1:end])
    fragment = body[
        body.index("    _b1_pin_selectors = ("): body.index("    expected = {")
    ]

    class _Exit(Exception):
        pass

    namespace = {"os": types.SimpleNamespace(environ=dict(env)),
                 "SystemExit": _Exit}
    source = "if True:\n" + "\n".join(
        "    " + line for line in fragment.splitlines()
    )
    try:
        exec(compile(source, "<py_resolver>", "exec"), namespace)  # noqa: S102
    except _Exit as exc:
        raise RuntimeError(str(exc)) from None
    return namespace["b1_pin_arm"]


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_python_resolver_knows_the_tier_b_spelling(launcher) -> None:
    """SITE 23, which closed the loop on site 17.

    Site 17 replaced a `.get(arm, split2_pins)` with a table -- and the table
    kept an "" key. This resolver never learned TIER_B_ARM, so a tier-B boot
    resolved "" and the "" key handed back split2's pins. Naming a default does
    not remove it; the removal has to happen at the resolver.
    """
    assert _launcher_python_resolver(
        launcher, {"FR13_FA2_QROW32_B1_TIER_B_ARM": SPLITK_ARM}
    ) == SPLITK_ARM
    # And "" can no longer be produced at all: the no-selector case is spelled.
    assert _launcher_python_resolver(launcher, {}) == "nosplit"
    assert '"": (' not in launcher.read_text(), (
        "the empty key is back; it is the default this fix removed"
    )


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_python_resolver_is_total_against_the_environment(launcher) -> None:
    """The property that catches the 24th selector nobody has written yet.

    The resolver is made total against the ENVIRONMENT, not against a list of
    keys somebody remembered to update: any FR13_FA2_QROW32_B1_*_ARM that is
    set and unknown to it is a refusal. Sites 2.1, 17 and 23 were all one
    variable this resolver had not been told about.
    """
    with pytest.raises(RuntimeError, match="does not know these selectors"):
        _launcher_python_resolver(
            launcher, {"FR13_FA2_QROW32_B1_SOMETHING_NEW_ARM": "whatever"}
        )
    # A known non-pin selector must NOT trip it, or the sweep gets widened
    # until it passes and stops meaning anything.
    assert _launcher_python_resolver(
        launcher, {"FR13_FA2_QROW32_B1_TIMING_ARM": "gqa_pair"}
    ) == "nosplit"
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        _launcher_python_resolver(launcher, {
            "FR13_FA2_QROW32_B1_TIER_B_ARM": SPLITK_ARM,
            "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "gqa_pair",
        })


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_credential_path_is_split_host_and_container(launcher) -> None:
    """Settled by measurement, not by argument.

    Measured 2026-08-19 with the launcher's own mounts: the repo mounts at
    /workspace, so a host path is NOT readable in the container -- /home exists
    there but is the image's own and the file is absent. One variable cannot
    serve both consumers.
    """
    text = launcher.read_text()
    # The host-side check reads the _HOST spelling...
    assert 'sha256sum "$FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_HOST"' in text
    # ...the container-side path is DERIVED, never supplied, so the two cannot
    # be given out of sync.
    assert (
        "FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL=/logs/"
        "fr13_fa2_qrow32_b1_tier_b_credential.json" in text
    )
    # The staged copy is re-digested, so a copy that changed the file refuses.
    assert "tier-b credential changed while being staged" in text
    # And the container consumer reads the container spelling.
    assert '--credential "\\$FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL"' in text
    # Both spellings are guarded against .lumo.local.env.
    assert "\n  FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_HOST\n" in text
    assert "\n  FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL\n" in text


def _bash_pin_arm_for(launcher, env):
    """Execute the launcher's OWN bash pin-arm resolver lines, unmodified."""
    import subprocess

    text = launcher.read_text()
    start = text.index(
        '  _FR13_FA2_QROW32_B1_PIN_ARM=$FR13_FA2_QROW32_B1_LIVE_AB_ARM'
    )
    end = text.index('  case "$_FR13_FA2_QROW32_B1_PIN_ARM" in', start)
    fragment = text[start:end]
    preamble = "set -u\n" + "".join(
        '{}="{}"\n'.format(name, env.get(name, ""))
        for name in (
            "FR13_FA2_QROW32_B1_LIVE_AB_ARM",
            "FR13_FA2_QROW32_B1_TIER_B_ARM",
            "FR13_FA2_QROW32_B1_PRODUCTION_ARM",
        )
    )
    script = (
        preamble
        + fragment
        + '\nprintf "%s" "$_FR13_FA2_QROW32_B1_PIN_ARM"\n'
    )
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    return out.stdout

# ===========================================================================
# THE UNIVERSAL RESOLVER TEST.
#
# Four resolvers have now answered the same question wrongly across ten
# rounds -- sites 2.1, 17, 23 and 24 -- and each fix taught ONE of them. The
# round-9 twin detector compared two; this supersedes it by enumerating ALL of
# them, and by enumerating them from the SOURCE rather than from a list, so
# the enumeration cannot rot the way every hand-maintained list in this
# campaign has.
#
# Discovery: grep every read of the two legacy selectors across the launchers,
# the patcher blobs, the contract and the sidecar. Every hit is either an
# executable arm->identity resolver (exercised below against the canonical
# tier-B environment, and required to answer split-K) or an adjudicated
# non-resolver with a written reason. A new hit that is neither fails.
# ===========================================================================

LEGACY_SELECTOR_RE = re.compile(
    r"FR13_FA2_QROW32_B1_(?:LIVE_AB|PRODUCTION)_ARM"
)
RESOLVER_SURFACE = (
    REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh",
    REPO / "scripts/fr14_armb_leg3_launch_nomiddleware.sh",
    REPO / "scripts/fr14_leg3_launch_nomiddleware.sh",
    REPO / "scripts/fr13_patch_fa2_tree_bias.py",
    REPO / "scripts/fr13_fixed32_contract.py",
    REPO / "scripts/fr13_qrow32_b1_pass_sidecar.py",
)

# Every line that reads a legacy selector and is NOT an arm->identity resolver,
# with the reason. Keyed on a substring of the line, because line numbers rot
# faster than code does. If a line here stops existing the registry is stale;
# if a line appears that is not here and not a resolver, the test fails.
# Reads that ARE arm->identity resolvers and ARE executed by the universal
# test below. Listing them separately keeps the adjudication registry honest:
# a resolver must be exercised, not excused.
EXERCISED_RESOLVERS = (
    ("_FR13_FA2_QROW32_B1_PIN_ARM=$FR13_FA2_QROW32_B1_LIVE_AB_ARM",
     "bash pin-arm resolver (executed, all three twins)"),
    ("_FR13_FA2_QROW32_B1_PIN_ARM=$FR13_FA2_QROW32_B1_PRODUCTION_ARM",
     "bash pin-arm resolver, production leg (executed)"),
    ('qrow32_b1_live = env.get("FR13_FA2_QROW32_B1_LIVE_AB_ARM"',
     "contract _expected_runtime_fa2_identity (executed) -- site 24"),
    ('qrow32_b1_production = env.get("FR13_FA2_QROW32_B1_PRODUCTION_ARM"',
     "contract _expected_runtime_fa2_identity, production leg (executed)"),
    ('"FR13_FA2_QROW32_B1_LIVE_AB_ARM",',
     "python in-container resolver tuple (executed, all three twins)"),
    ('"FR13_FA2_QROW32_B1_PRODUCTION_ARM",',
     "python in-container resolver tuple, production leg (executed)"),
)

ADJUDICATED_NON_RESOLVERS = (
    ("_FR13_M32_GUARD", "guard-name registry: protects the var, does not resolve it"),
    ("_FR13_CALLER_M32_GUARD", "caller-guard equality checks, not resolution"),
    ("=${FR13_FA2_QROW32_B1_LIVE_AB_ARM:-}", "set -u default"),
    ("=${FR13_FA2_QROW32_B1_PRODUCTION_ARM:-}", "set -u default"),
    ("-v FR13_FA2_QROW32_B1_PRODUCTION_ARM", "was-it-named probe, not resolution"),
    ("_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED", "was-it-named flag"),
    ("SELECTOR_COUNT", "mutual-exclusion counter"),
    ("case \"$FR13_FA2_QROW32_B1_LIVE_AB_ARM\"", "arm allowlist"),
    ("case \"$FR13_FA2_QROW32_B1_PRODUCTION_ARM\"", "arm allowlist"),
    ("must be empty", "allowlist error text"),
    ("-e FR13_FA2_QROW32_B1_LIVE_AB_ARM", "container transport"),
    ("-e FR13_FA2_QROW32_B1_PRODUCTION_ARM", "container transport"),
    ("export FR13_FA2_QROW32_B1_LIVE_AB_ARM", "child-interpreter transport"),
    ("PRODUCTION_ARM_DEFAULT", "the promoted default's own name"),
    ("_FR13_FA2_QROW32_B1_PRODUCTION_ARMS", "the production arm SET, not a resolver"),
    ("FR13_FA2_QROW32_B1_LIVE_AB_ARM\"", "env read inside an adjudicated resolver"),
    ("FR13_FA2_QROW32_B1_PRODUCTION_ARM\"", "env read inside an adjudicated resolver"),
    ("FR13_FA2_QROW32_B1_LIVE_AB_ARM',", "env read inside an adjudicated resolver"),
    ("FR13_FA2_QROW32_B1_PRODUCTION_ARM',", "env read inside an adjudicated resolver"),
    ("-n \"$FR13_FA2_QROW32_B1_PRODUCTION_ARM\"", "mode predicate"),
    ("-n \"${FR13_FA2_QROW32_B1_LIVE_AB_ARM:-}\"", "mode predicate"),
    ("-n \"${FR13_FA2_QROW32_B1_PRODUCTION_ARM:-}\"", "mode predicate"),
    ("-n \"$FR13_FA2_QROW32_B1_LIVE_AB_ARM\"", "mode predicate"),
    ("-z \"$FR13_FA2_QROW32_B1_LIVE_AB_ARM\"", "mode predicate"),
    ("-z \"$FR13_FA2_QROW32_B1_PRODUCTION_ARM\"", "mode predicate"),
    ("-z \"${FR13_FA2_QROW32_B1_LIVE_AB_ARM:-}\"", "mode predicate"),
    ("-z \"${FR13_FA2_QROW32_B1_PRODUCTION_ARM:-}\"", "mode predicate"),
    ("!= \"gqa_pair\"", "mode predicate"),
    ("== \"gqa_pair\"", "mode predicate"),
    ("#", "comment"),
    ('== "nosplit"', "mode predicate"),
    ('-n "\\${FR13_FA2_QROW32_B1_PRODUCTION_ARM}"',
     "tier-A attestation block predicate, inside a heredoc"),
    ('-n "\\${FR13_FA2_QROW32_B1_TIER_B_ARM}"',
     "tier-B attestation block predicate, inside a heredoc"),
)

# Bare variable names on their own line: the guard-name registry.
BARE_NAME_LINES = (
    "FR13_FA2_QROW32_B1_LIVE_AB_ARM",
    "FR13_FA2_QROW32_B1_PRODUCTION_ARM",
)

# The canonical tier-B environment: exactly what a round-11 boot will set.
CANONICAL_TIER_B_ENV = {
    "FR13_FA2_QROW32_B1_TIER_B_ARM": SPLITK_ARM,
    "FR13_FA2_QROW32_B1_SO_SHA256": SPLITK_SO_SHA256,
    "FR13_FA2_QROW32_B1_SO_SIZE": "300123792",
}
SPLITK_SIZE = 300123792
SPLITK_SASS = "3f24d70dce2ff70ad9209bad5af2a93cc39453df529cb298e4476cbfbfd80b9e"
SPLITK_BASELINE_SASS = (
    "fa01f98840420b9c0177d06297aacabb0ed5e00c674511fdaa4aa618c3473470"
)


def _legacy_selector_reads():
    """Every line in the surface that reads a legacy selector."""
    hits = []
    for path in RESOLVER_SURFACE:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if LEGACY_SELECTOR_RE.search(line):
                hits.append((path, number, line))
    return hits


def test_every_legacy_selector_read_is_a_resolver_or_adjudicated():
    """The enumeration cannot rot, because it is not a list.

    A new read of LIVE_AB_ARM or PRODUCTION_ARM anywhere in the resolver
    surface is either exercised by the universal test below or written down
    here with a reason. Sites 2.1, 17, 23 and 24 were each a read nobody had
    enumerated.
    """
    unreviewed = []
    exercised = 0
    for path, number, line in _legacy_selector_reads():
        stripped = line.strip()
        if stripped in BARE_NAME_LINES:
            continue
        if any(needle in line for needle, _why in EXERCISED_RESOLVERS):
            exercised += 1
            continue
        if any(needle in line for needle, _why in ADJUDICATED_NON_RESOLVERS):
            continue
        if stripped.startswith("#") or stripped.startswith("*"):
            continue
        unreviewed.append(f"{path.name}:{number}: {stripped[:100]}")
    # A registry that adjudicates everything and exercises nothing would pass
    # while testing nothing, so the count is asserted too.
    assert exercised >= 8, (
        f"only {exercised} resolver reads were exercised; the universal test "
        "must actually run them"
    )
    assert not unreviewed, (
        "unadjudicated reads of a legacy B1 selector -- each is either an "
        "arm->identity resolver that must answer the canonical tier-B "
        "environment, or a non-resolver needing a written reason:\n  "
        + "\n  ".join(unreviewed)
    )


def test_universal_resolver_agreement_on_the_canonical_tier_b_environment():
    """EVERY arm->identity resolver, fed the same environment, must agree.

    This is the test that ends the class. Ten rounds, four resolvers, one
    question: which binary does this boot authorise? They are executed here
    side by side and required to say split-K.
    """
    answers = {}

    # 1-3. the bash pin-arm resolver, in all three launcher twins
    for launcher in LAUNCHERS:
        answers[f"bash:{launcher.name}"] = _bash_pin_arm_for(
            launcher, CANONICAL_TIER_B_ENV
        )
    # 4-6. the in-container python resolver, in all three twins
    for launcher in LAUNCHERS:
        answers[f"python:{launcher.name}"] = _launcher_python_resolver(
            launcher, CANONICAL_TIER_B_ENV
        )
    for name, arm in answers.items():
        assert arm == SPLITK_ARM, f"{name} resolved {arm!r}, not the tier-B arm"

    # 7. the runtime contract -- SITE 24
    contract = _contract()
    size, sha = contract._expected_runtime_fa2_identity(
        dict(CANONICAL_TIER_B_ENV)
    )
    assert (size, sha) == (SPLITK_SIZE, SPLITK_SO_SHA256), (
        "the runtime contract resolved the wrong binary for a tier-B boot"
    )

    # 8. the injected blob's identity table, which also carries the SASS pins
    selectors = _selectors()
    identity = selectors["_fr13_fa2_qrow32_b1_identity"](SPLITK_ARM)
    assert identity["candidate_sha256"] == SPLITK_SO_SHA256
    assert identity["candidate_size"] == SPLITK_SIZE
    assert identity["sass_digest_sha256"] == SPLITK_SASS
    assert identity["baseline_sass_digest_sha256"] == SPLITK_BASELINE_SASS

    # 9. the sidecar's candidate contract
    sidecar = _sidecar()
    candidate = sidecar._candidate_contract(SPLITK_ARM)
    assert candidate["sha256"] == SPLITK_SO_SHA256
    assert candidate["size"] == SPLITK_SIZE

    # 10. and the serving resolver itself, which decides the tier
    import os as _os

    saved = {
        k: _os.environ.get(k)
        for k in ("FR13_FA2_QROW32_B1_TIER_B_ARM",
                  "FR13_FA2_QROW32_B1_LIVE_AB_ARM",
                  "FR13_FA2_QROW32_B1_PRODUCTION_ARM")
    }
    try:
        for key in saved:
            _os.environ.pop(key, None)
        _os.environ["FR13_FA2_QROW32_B1_TIER_B_ARM"] = SPLITK_ARM
        assert selectors["_fr13_fa2_qrow32_b1_serving_arm"]() == (
            SPLITK_ARM, "B"
        )
    finally:
        for key, value in saved.items():
            if value is None:
                _os.environ.pop(key, None)
            else:
                _os.environ[key] = value

    # Six executed launcher resolvers (3 bash + 3 python) plus the contract,
    # the injected identity table, the sidecar contract and the serving
    # resolver: nine independent answers to one question.
    assert len(answers) == 6, sorted(answers)


def test_the_retired_pun_no_longer_out_resolves_the_real_spelling():
    """Site 24's signature: the pun worked and the real spelling did not."""
    contract = _contract()
    new_spelling = contract._expected_runtime_fa2_identity({
        "FR13_FA2_QROW32_B1_TIER_B_ARM": SPLITK_ARM,
        "FR13_FA2_QROW32_B1_SO_SHA256": SPLITK_SO_SHA256,
    })
    assert new_spelling == (SPLITK_SIZE, SPLITK_SO_SHA256)
    # The pun still resolves the same binary (the live-A/B shadow arm is a real
    # mode), but it is no longer the ONLY spelling that does.
    pun = contract._expected_runtime_fa2_identity({
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM": SPLITK_ARM,
        "FR13_FA2_QROW32_B1_SO_SHA256": SPLITK_SO_SHA256,
    })
    assert pun == new_spelling


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_the_production_default_does_not_fire_under_a_tier_b_boot(launcher):
    """SITE 25, found by the enumeration before it fired.

    A Hydra27 B1 launch that names no production arm DEFAULTS
    FR13_FA2_QROW32_B1_PRODUCTION_ARM to gqa_pair. Its guard listed every other
    selector and not TIER_B_ARM, so a tier-B boot would have had the promoted
    arm armed underneath it -- caught downstream by the mutual-exclusion checks
    added at sites 23/24, but caught as a crash rather than never happening.
    """
    text = launcher.read_text()
    block_start = text.index("_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED == 0")
    block = text[block_start: text.index("]]; then", block_start)]
    assert '-z "$FR13_FA2_QROW32_B1_TIER_B_ARM"' in block, (
        "the promoted default's guard does not exclude a tier-B boot"
    )


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_the_contract_resolver_can_see_the_selector_it_learned(launcher):
    """A resolver that knows a selector it cannot see is still stranded."""
    text = launcher.read_text()
    export_at = text.index("export FR13_FA2_QROW32_B1_LIVE_AB_ARM")
    exported = text[export_at: text.index("PYTHONPATH", export_at)]
    assert "FR13_FA2_QROW32_B1_TIER_B_ARM" in exported, (
        "the child interpreter that runs _expected_runtime_fa2_identity never "
        "receives the tier-B selector"
    )
