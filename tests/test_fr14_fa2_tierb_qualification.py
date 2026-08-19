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
import ast
import json
import re
import subprocess
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


def test_earned_credential_is_present_and_validates(tmp_path: Path) -> None:
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
    # SCOPE, post-promotion: the commit is RECORDED, not matched. A credential
    # earned at an older HEAD stays valid as long as everything that
    # determines the numerics is unchanged -- because under a promoted default
    # with a hard refusal, commit-binding breaks the boot on every commit and
    # protects nothing a digest does not already protect.
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
    # But the PATCHER is still bound: it decides dispatch, so it can change
    # what the kernel computes.
    with pytest.raises(ValueError, match="does not bind this arm"):
        sidecar.validate_tierb_credential(
            path,
            arm=SPLITK_ARM,
            expected_candidate_sha256=contract.QROW32_B1_SPLITK_FA2_SHA256,
            expected_source_commit=payload["identity"]["source_commit"],
            expected_patch_source_sha256="a" * 64,
            bounds_path=BOUNDS,
        )
    # And a malformed commit is still refused: "recorded" is not "absent".
    body = json.loads(path.read_text())
    body["identity"]["source_commit"] = "not-a-commit"
    body["credential_sha256"] = sidecar.tierb_credential_digest(body)
    # tmp_path, NOT path.parent: an earlier revision wrote this beside the
    # earned credential in results/, leaving an untracked credential-shaped
    # file with source_commit "not-a-commit" in the deliverable directory --
    # a test artifact that reads as a provenance leak to anyone auditing it.
    broken = tmp_path / "broken_commit.json"
    broken.write_text(json.dumps(body, sort_keys=True), encoding="ascii")
    with pytest.raises(ValueError, match="not a lowercase commit"):
        sidecar.validate_tierb_credential(
            broken,
            arm=SPLITK_ARM,
            expected_candidate_sha256=contract.QROW32_B1_SPLITK_FA2_SHA256,
            expected_source_commit=payload["identity"]["source_commit"],
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
        # SITE 17: the serve gate now delegates to the workload accessor, so
        # dropping either pin is caught as "the declared workload's identity
        # is not what arrived" rather than as a bespoke exact4 comparison.
        ("FR13_FA2_QROW32_B1_EXACT4_TASK_IDS", "declares workload 'exact4'"),
        ("FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256", "declares workload 'exact4'"),
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


# ===========================================================================
# THE CROSS-FILE SYMBOL DETECTOR (sites 18 and 25).
#
# Both were producer/consumer installer splits: a fragment CALLING a symbol
# injected under one condition, the blob DEFINING it injected under another.
# Both were found by booting a GPU server and reading a traceback. Both are
# static facts about the patcher's output.
# ===========================================================================

SYMBOL_SWEEP = REPO / "scripts/fr14_patch_symbol_resolution_sweep.py"


def _symbol_sweep():
    return _module(SYMBOL_SWEEP, "fr14_symbol_sweep")


def test_every_injected_symbol_resolves_in_every_arm_mode():
    """The test that ends the patching family.

    Run for all three modes, because a detector that only knew the mode under
    repair would have passed on the day site 18 shipped.
    """
    sweep = _symbol_sweep()
    report = sweep.sweep()
    assert {row["mode"] for row in report["modes"]} == {
        "tier_a_production", "tier_b_serve", "live_ab_shadow"
    }
    for row in report["modes"]:
        assert not row["dangling"], (
            f"{row['mode']}: injected symbols with no definition:\n  "
            + "\n  ".join(
                f"{d['symbol']} in {d['referenced_in']} ({d['reason']})"
                for d in row["dangling"]
            )
        )
        # A sweep that resolved nothing would also report nothing dangling.
        assert row["symbols_resolved"] >= 60, row["symbols_resolved"]
        # And the cross-file edge is the one that actually bit twice.
        assert len(row["cross_file_edges"]) == 1, row["cross_file_edges"]
        edge = row["cross_file_edges"][0]
        assert edge["referenced_in"].endswith("cuda_graph.py")
        assert edge["defined_in"].endswith("tree_attn.py")


def test_symbol_detector_catches_site_25(tmp_path):
    """Revert site 25 and the detector must find it -- and find all three.

    Site 25 was the helpers blob's install condition omitting tier-B while
    cuda_graph.py's import of one of its symbols was installed for tier-B. The
    same disjunction gates production_begin, production_end and capture_end, so
    each would have bitten in turn had they been fixed one at a time. All three
    must appear.
    """
    sweep = _symbol_sweep()
    src = PATCHER.read_text()
    current = (
        "    if (\n"
        "        fixed32_query_tile32_b1_live_ab\n"
        "        or fixed32_query_tile32_b1_production\n"
        "        or fixed32_query_tile32_b1_tier_b_serve\n"
        "    ):"
    )
    reverted = (
        "    if fixed32_query_tile32_b1_live_ab or "
        "fixed32_query_tile32_b1_production:"
    )
    assert current in src, "the site-25 install condition moved"
    mutant = tmp_path / "patcher_site25.py"
    mutant.write_text(src.replace(current, reverted, 1))
    sweep.PATCHER = mutant

    report = sweep.sweep()
    by_mode = {row["mode"]: row for row in report["modes"]}
    # tier-A and the shadow are unaffected: the bug was tier-B-only.
    assert not by_mode["tier_a_production"]["dangling"]
    assert not by_mode["live_ab_shadow"]["dangling"]
    dangling = {d["symbol"] for d in by_mode["tier_b_serve"]["dangling"]}
    assert dangling == {
        "_fr13_fa2_qrow32_b1_production_begin",
        "_fr13_fa2_qrow32_b1_production_end",
        "_fr13_fa2_qrow32_b1_production_capture_end",
    }, dangling
    # and the cross-file one is diagnosed as a cross-file failure, not a
    # missing local name -- so the reader is pointed at the right installer.
    cross = next(
        d for d in by_mode["tier_b_serve"]["dangling"]
        if d["symbol"].endswith("capture_end")
    )
    assert "does not define it" in cross["reason"]
    assert cross["referenced_in"].endswith("cuda_graph.py")


def test_symbol_detector_catches_a_reverted_site_18(tmp_path):
    """The other installer, reverted: the serving call site without tier-B."""
    sweep = _symbol_sweep()
    src = PATCHER.read_text()
    current = (
        "    if (\n"
        "        fixed32_query_tile32_b1_production\n"
        "        or fixed32_query_tile32_b1_tier_b_serve\n"
        "    ) and ("
    )
    assert current in src, "the site-18 install condition moved"
    mutant = tmp_path / "patcher_site18.py"
    mutant.write_text(
        src.replace(current, "    if fixed32_query_tile32_b1_production and (", 1)
    )
    sweep.PATCHER = mutant
    report = sweep.sweep()
    by_mode = {row["mode"]: row for row in report["modes"]}
    # The consumer in cuda_graph.py is still installed for tier-B; its producer
    # is still installed; so nothing DANGLES -- but the call site is gone, and
    # that is a different detector's job (the CPU end-to-end walk). Recorded
    # here so the reach of this one is written down rather than assumed.
    assert not by_mode["tier_b_serve"]["dangling"]


def test_the_tier_b_serve_is_mutually_exclusive_with_the_shadow_at_the_patcher():
    """Found by the symbol inventory, not by a boot.

    tier_b_serve was absent from _patch_tree_attn's private-selector dict, so
    it could be combined with b1_live_ab: both wrappers installed into one
    decode call, while cuda_graph.py got only the live-replay hook -- leaving
    production_capture_end defined, its state populated, and its verification
    never run.
    """
    patcher = _module(PATCHER, "fr14_symbol_patcher")
    sweep = _symbol_sweep()
    import tempfile as _tempfile

    root = Path(_tempfile.mkdtemp()) / "site-packages"
    sweep.build_engine_tree(root)
    with pytest.raises(ValueError, match="mutually exclusive"):
        patcher.patch_installed_vllm(
            root,
            fixed32_query_tile32_b1_tier_b_serve=True,
            fixed32_query_tile32_b1_live_ab=True,
        )


def test_patched_modules_import_resolve(tmp_path):
    """TASK 3: the ImportError class dies in tests.

    Site 25 presented as an ImportError at vLLM module import, after the
    launcher, the credentials and the gate had all passed. The patched modules
    are compiled and their module-level bodies executed here against stub
    packages, so an injected import that cannot resolve fails in
    milliseconds instead of minutes.
    """
    import py_compile

    sweep = _symbol_sweep()
    patcher = _module(PATCHER, "fr14_symbol_patcher_import")
    for mode, params in sweep.MODES.items():
        root = tmp_path / mode / "site-packages"
        sweep.build_engine_tree(root)
        patcher.patch_installed_vllm(root, **params)
        for relative in sweep.STUBS:
            path = root / relative
            # py_compile is what the patcher itself runs; assert it here too so
            # a syntactically broken injection cannot reach a boot.
            py_compile.compile(str(path), doraise=True)

        # And the cross-file import specifically: the symbol cuda_graph.py
        # imports must exist as a top-level def in the module it names.
        cuda_graph = (root / "vllm/compilation/cuda_graph.py").read_text()
        tree_attn = ast.parse(
            (root / "vllm/v1/attention/backends/tree_attn.py").read_text()
        )
        defined = {
            node.name for node in tree_attn.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for line in cuda_graph.splitlines():
            stripped = line.strip().rstrip(",")
            if stripped.startswith("_fr13_fa2_qrow32_b1_") and stripped.endswith(
                ("_replay", "_capture_end")
            ):
                assert stripped in defined, (
                    f"{mode}: cuda_graph.py imports {stripped} but "
                    "tree_attn.py does not define it"
                )


# ===========================================================================
# THE PROMOTED DEFAULT (Mark, FR14 pass 100).
#
# Split-K is the production default. It is armed as a TIER-B serve, so the
# byte-exact Tier-A door is untouched and _FR13_FA2_QROW32_B1_PRODUCTION_ARMS
# is unchanged -- what promotion changed is that the tier-B route is armed by
# default rather than by name.
#
# The default block is executed here, not grepped: the bash is extracted from
# each launcher and run under four environments.
# ===========================================================================

SPLITK_DEFAULT_SO_SHA256 = SPLITK_SO_SHA256


def _run_default_block(launcher, env, so_path, credential_path):
    """Execute the launcher's OWN promoted-default block."""
    import subprocess

    text = launcher.read_text()
    lit_start = text.index(
        "# ---------------------------------------------------------------- split-K"
    )
    lit_end = text.index("\n", text.index("_FR13_SPLITK_DEFAULT_CREDENTIAL=", lit_start)) + 1
    literals = text[lit_start:lit_end]
    blk_start = text.index(
        "  # ===================================================================\n"
        "  # SPLIT-K IS THE PROMOTED DEFAULT"
    )
    blk_end = text.index("  fi\n", blk_start) + len("  fi\n")
    block = text[blk_start:blk_end]

    preamble = "set -u\nREPO=.\n"
    preamble += f'_FR13_SPLITK_DEFAULT_SO="{so_path}"\n'
    preamble += f'_FR13_SPLITK_DEFAULT_CREDENTIAL="{credential_path}"\n'
    for name in (
        "FR13_FA2_QROW32_B1_TIER_B_ARM",
        "FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_HOST",
        "FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_SHA256",
        "FORKED_FA2_SO",
        "FR13_FA2_QROW32_B1_SO_SHA256",
        "FR13_FA2_QROW32_B1_SO_SIZE",
        "FR13_FA2_QROW32_B1_FA2_HEAD",
        "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256",
        "FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST",
        "FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST",
    ):
        preamble += '{}="{}"\n'.format(name, env.get(name, ""))
    script = (
        preamble
        + literals.replace("${_FR13_SPLITK_DEFAULT_SO:-", "${__unused_so:-")
                 .replace("${_FR13_SPLITK_DEFAULT_CREDENTIAL:-", "${__unused_cred:-")
        + block
        + '\nprintf "ARM=%s\\nSO=%s\\n" '
          '"$FR13_FA2_QROW32_B1_TIER_B_ARM" "$FORKED_FA2_SO"\n'
    )
    # the literal block re-assigns from its own defaults; keep the test's
    # injected paths by re-asserting them after the literals
    script = script.replace(
        "\n  # ==============",
        f'\n_FR13_SPLITK_DEFAULT_SO="{so_path}"\n'
        f'_FR13_SPLITK_DEFAULT_CREDENTIAL="{credential_path}"\n  # ==============',
        1,
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


@pytest.fixture
def staged_default(tmp_path):
    """A staged binary whose digest is the pinned one, and a credential."""
    import hashlib

    so = tmp_path / "splitk.so"
    # Search for bytes hashing to the pin is impossible; the test instead
    # asserts the REFUSAL path on a wrong digest and the ACCEPT path by
    # pointing the pin at this file's real digest through the env override.
    so.write_bytes(b"not the real kernel, but a real file")
    credential = tmp_path / "credential.json"
    credential.write_text(_credential_sealed_against(_patcher_digest()))
    digest = hashlib.sha256(so.read_bytes()).hexdigest()
    return so, credential, digest


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_default_boot_arms_split_k(launcher, staged_default):
    """(a) A plain launch arms split-K."""
    so, credential, digest = staged_default
    env = {"FR13_FA2_QROW32_B1_SO_SHA256": digest}
    # The block compares the staged file against the LITERAL pin, so point the
    # literal at this file's digest for the accept path.
    text = launcher.read_text()
    assert f"_FR13_SPLITK_DEFAULT_SO_SHA256={SPLITK_DEFAULT_SO_SHA256}" in text, (
        "the promoted default's binary pin is not the characterized kernel"
    )
    out = _run_default_block(
        launcher, env, so, credential
    )
    # With the real pin and a stub file the digest cannot match, so this must
    # REFUSE -- which is itself behaviour (b). The accept path is exercised by
    # overriding the pin below.
    assert out.returncode == 2
    assert "staged binary missing or not the pinned kernel" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_default_boot_refuses_a_missing_binary(launcher, staged_default):
    """(b) Missing or wrong binary REFUSES -- it does not fall back.

    The incumbent gqa_pair default degrades to the incumbent on a stale
    credential, on the principle that a promotion must never refuse a boot.
    That principle is inverted here deliberately: a promoted default that
    silently serves something else is an unlabelled A/B, and round 6 spent a
    whole arm measuring the incumbent while every artifact said split-K.
    """
    _so, credential, _digest = staged_default
    out = _run_default_block(
        launcher, {}, "/nonexistent/splitk.so", credential
    )
    assert out.returncode == 2
    assert "staged binary missing or not the pinned kernel" in out.stderr
    assert "must not silently serve the incumbent" in out.stderr
    # and it must NOT have quietly armed gqa_pair
    assert "gqa_pair\n" not in out.stdout


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_default_boot_refuses_a_missing_credential(launcher, staged_default):
    """The credential half of the same refusal."""
    so, _credential, digest = staged_default
    text = launcher.read_text()
    patched = text.replace(
        f"_FR13_SPLITK_DEFAULT_SO_SHA256={SPLITK_DEFAULT_SO_SHA256}",
        f"_FR13_SPLITK_DEFAULT_SO_SHA256={digest}",
        1,
    )
    import tempfile

    scratch = Path(tempfile.mkdtemp()) / launcher.name
    scratch.write_text(patched)
    out = _run_default_block(scratch, {}, so, "/nonexistent/credential.json")
    assert out.returncode == 2
    assert "tier-b credential missing" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_default_boot_accepts_a_staged_pair(launcher, staged_default):
    """(a), properly: with binary and credential present, split-K is armed."""
    so, credential, digest = staged_default
    text = launcher.read_text().replace(
        f"_FR13_SPLITK_DEFAULT_SO_SHA256={SPLITK_DEFAULT_SO_SHA256}",
        f"_FR13_SPLITK_DEFAULT_SO_SHA256={digest}",
        1,
    )
    import tempfile

    scratch = Path(tempfile.mkdtemp()) / launcher.name
    scratch.write_text(text)
    out = _run_default_block(scratch, {}, so, credential)
    assert out.returncode == 0, out.stderr
    assert f"ARM={SPLITK_ARM}" in out.stdout
    assert "serving the PROMOTED DEFAULT" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_explicit_opt_out_is_preserved(launcher, staged_default):
    """(c) Naming an arm explicitly must still work, for A/Bs.

    Round 20's H31i topology A/B runs on stock FA2 by explicit opt-out, so
    this is load-bearing right now and not a hypothetical.
    """
    so, credential, _digest = staged_default
    out = _run_default_block(
        launcher, {"FR13_FA2_QROW32_B1_TIER_B_ARM": "gqa_pair_splitk"},
        so, credential,
    )
    # already named -> the block is a no-op and does not re-check the binary
    assert out.returncode == 0, out.stderr
    assert f"ARM={SPLITK_ARM}" in out.stdout
    assert "serving the PROMOTED DEFAULT" not in out.stderr
    # and the production opt-out door is still shut for split-K
    text = launcher.read_text()
    assert '""|nosplit|gqa_pair) ;;' in text, (
        "the production allowlist must still refuse split-K -- promotion armed "
        "the tier-B route, it did not widen the byte-gated one"
    )


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_the_default_is_gated_on_hydra27(launcher):
    """(d) hydra31 must NOT arm it -- excluded topologically."""
    # Checked STRUCTURALLY: the block must lie between the mode-gated `if` and
    # its closing `fi`. A first version of this test sliced backwards for the
    # guard text and landed inside a COMMENT that quoted it -- the same
    # text-keying mistake this campaign has now made three times.
    lines = launcher.read_text().splitlines()
    opener = next(
        i for i, line in enumerate(lines)
        if "_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED == 0" in line
        and line.startswith("if ((")
    )
    then = next(
        i for i in range(opener, len(lines))
        if lines[i].rstrip().endswith("]]; then")
    )
    closing = next(i for i in range(then, len(lines)) if lines[i] == "fi")
    block = next(
        i for i, line in enumerate(lines)
        if "# SPLIT-K IS THE PROMOTED DEFAULT" in line
    )
    assert then < block < closing, (
        "the promoted split-K default is not nested inside the mode-gated "
        "block; hydra31 would arm it before its own qualification"
    )
    guard = "\n".join(lines[opener:then + 1])
    assert '"${FR13_FIXED32_MODE:-}" == "hydra27_fixed32"' in guard
    assert "hydra31" not in guard


def test_promotion_did_not_widen_the_byte_gated_door():
    """The invariant the whole tier-B architecture rests on."""
    namespace = _selectors()
    assert namespace["_FR13_FA2_QROW32_B1_PRODUCTION_ARMS"] == (
        "nosplit", "gqa_pair",
    )
    contract = _contract()
    with pytest.raises(contract.ContractError, match="must be empty, nosplit"):
        contract._expected_runtime_fa2_identity({
            "FR13_FA2_QROW32_B1_PRODUCTION_ARM": SPLITK_ARM,
            "FR13_FA2_QROW32_B1_SO_SHA256": SPLITK_SO_SHA256,
        })


# ===========================================================================
# THE FULL-BOOT WALK (F1/F2, pass 106).
#
# The walk above answered "does the block arm?".  The question was "does the
# boot SURVIVE the arming?" -- and the answer was no: arming set
# SELECTOR_COUNT=1, which opened a selector gate 950 lines later that demanded
# a SOURCE_COMMIT and a PATCH_SOURCE_SHA256 the block never set, so every
# plain hydra27 B1 launch exited 2 and the promoted default had never served.
# A four-environment walk that stops at the block's own `fi` cannot see that,
# because the defect is not IN the block -- it is in what the block hands to
# the rest of the boot.
#
# So this walk composes the three regions the arming actually flows through
# and executes them together:
#
#   region 1  the split-K default literals
#   region 2  the mode-gated promoted-default region (BOTH defaults, so F2's
#             arbitration is exercised, not just split-K's block)
#   region 3  SELECTOR_COUNT accumulation + the mutual-exclusion refusal
#   region 4  the B1 selector gate, including the reconciled commit clause
#
# Regions are sliced by anchors that exist in all three twins and are
# executed verbatim -- no paraphrase of the launcher's logic lives here, which
# is the only way a walk can fail when the launcher changes underneath it.
# ===========================================================================


def _patcher_digest():
    """sha256 of the patcher, the one provenance clause the gate still binds."""
    import hashlib

    return hashlib.sha256(PATCHER.read_bytes()).hexdigest()


def _credential_sealed_against(patch_digest):
    """The staged credential, re-sealed against a given patcher digest.

    SITE 16 (a): the launcher now MINTS the patcher digest from the
    credential's sealed identity instead of by hashing the patcher, so the
    selector gate is sealed-vs-disk rather than disk-vs-disk. That makes these
    walks depend on the staged credential agreeing with the patcher in the
    tree -- which it does not, and must not be assumed to, between an edit
    that touches the patcher and the runner's re-seal. So the walk seals its
    own copy. test_the_staged_credential_is_sealed_against_this_patcher is
    what watches the REAL one.
    """
    body = json.loads(
        (REPO / "results/fr14_nvfp4_port_20260816"
         / "fr14_splitk_tierb_credential.json").read_text()
    )
    body["identity"]["patch_source_sha256"] = patch_digest
    return json.dumps(body, indent=2, sort_keys=True)


def _boot_regions(text):
    """Slice the regions a promoted-default boot flows through.

    Region 0 is the REAL _fr13_assert_draft_vocab_profile definition, lifted
    from the launcher rather than stubbed. Site 12 (pass 113) was a defect in
    whether that helper is CALLED at the B1 selector -- the forks hard-coded
    K64/root1 instead, making full_vocab impossible -- and a walk that stubs
    the helper to `return 0` cannot see it. What the walk stubs, it cannot
    test.
    """
    helper_start = text.index("_fr13_assert_draft_vocab_profile() {")
    helper_end = text.index("\n}\n", helper_start) + len("\n}\n")
    region0 = text[helper_start:helper_end]

    # SITE 15's region. The credential pointer auto-imports whenever the file
    # exists, ~500 lines before the default block, and the walk could not see
    # it because the walk started AFTER it. What a walk omits, it cannot test
    # -- the same lesson site 12 taught about what a walk stubs.
    region0 += "_FR13_B1_POINTER_IMPORTED=()\n"
    if "_fr13_b1_load_credential_pointer() {" in text:
        loader_start = text.index("_fr13_b1_load_credential_pointer() {")
        loader_end = text.index("\n}\n", loader_start) + len("\n}\n")
        region0 += text[loader_start:loader_end]
    else:
        region0 += "_fr13_b1_load_credential_pointer() { return 0; }\n"
    withdraw_start = text.index("_fr13_b1_withdraw_pointer_imports() {")
    withdraw_end = text.index("\n}\n", withdraw_start) + len("\n}\n")
    region0 += text[withdraw_start:withdraw_end]

    lit_start = text.index(
        "# ---------------------------------------------------------------- split-K"
    )
    lit_end = text.index("\n", text.index("_FR13_SPLITK_DEFAULT_CREDENTIAL=", lit_start)) + 1
    region1 = text[lit_start:lit_end]

    lines = text.splitlines(keepends=True)
    opener = next(
        i for i, line in enumerate(lines)
        if line.startswith("if ((")
        and "_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED == 0" in line
    )
    closing = next(i for i in range(opener, len(lines)) if lines[i] == "fi\n")
    region2 = "".join(lines[opener:closing + 1])

    acc = text.index("_FR13_FA2_QROW32_B1_SELECTOR_COUNT=0\n")
    excl = text.index(
        'echo "FR13 qrow32 B1 live A/B and production arms are mutually exclusive" >&2\n'
        "  exit 2\nfi\n", acc
    )
    region3 = text[acc:excl] + (
        'echo "FR13 qrow32 B1 live A/B and production arms are mutually exclusive" >&2\n'
        "  exit 2\nfi\n"
    )

    gate = text.index(
        "if (( _FR13_FA2_QROW32_B1_SELECTOR_COUNT > 0 )); then\n"
        "  _FR13_FA2_QROW32_B1_CANDIDATE_MODE=1\n"
    )
    tail = (
        '    [[ "$FR13_FA2_QROW32_B1_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {\n'
        '      echo "FR13 qrow32 B1 tier-b selector requires a well-formed source commit" >&2\n'
        "      exit 2\n"
        "    }\n"
        "  fi\n"
    )
    region4 = text[gate:text.index(tail, gate) + len(tail)] + "fi\n"
    return region0, region1, region2, region3, region4


_BOOT_STUBS = """set -u
REPO=.
FR13_FIXED32_MODE=hydra27_fixed32
MAX_NUM_SEQS=1
SWE_CONCURRENCY=1
FR13_FIXED32_B1_DIAGNOSTIC=0
FR13_FA2_QROW16_LIVE_PAGED_AB=0
FR13_FA2_QROW16_PRODUCTION=0
FR13_FA2_QROW32_LIVE_PAGED_AB=0
"""

# The two draft-vocabulary identities the profile helper admits. full_vocab
# is the shape split-K has actually served in -- round 12's promotion evidence
# and measurement 1 both run K0 -- and it is the shape site 12 made impossible
# in the forks, so it is a boot case here, not a footnote.
_VOCAB_PROFILES = {
    "k64_root": {
        "FR13_DRAFT_VOCAB_ROOT": "1",
        "FR13_DRAFT_VOCAB_K": "65536",
        "FR13_DRAFT_VOCAB_BLOCKS": "/workspace/scripts/fr13_dvk_subset_blocks.json",
        "FR13_NEEDS_ALLOW": "",
    },
    "full_vocab": {
        "FR13_DRAFT_VOCAB_ROOT": "0",
        "FR13_DRAFT_VOCAB_K": "0",
        "FR13_DRAFT_VOCAB_BLOCKS": "",
        "FR13_NEEDS_ALLOW": "FR13_DRAFT_VOCAB_K=0",
    },
}

_BOOT_VARS = (
    "_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED",
    "FR13_FA2_QROW32_B1_LIVE_AB_ARM",
    "FR13_FA2_QROW32_B1_PRODUCTION_ARM",
    "FR13_FA2_QROW32_B1_PRODUCTION_ARM_DEFAULT",
    "FR13_FA2_QROW32_B1_TIER_B_ARM",
    "FR13_FA2_QROW32_B1_TIER_B_SERVE",
    "FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_HOST",
    "FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_SHA256",
    "FR13_FA2_QROW32_B1_TIMING_ARM",
    "FR13_FA2_QROW32_B4_TIMING_ARM",
    "FR13_FA2_QROW32_B4_PRODUCTION_ARM",
    "FR13_FA2_QROW32_B1_SOURCE_COMMIT",
    "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256",
    "FR13_FA2_QROW32_B1_GQA_PAIR_GATE_JSON",
    "FR13_FA2_QROW32_B1_GQA_PAIR_LIVE_RESULT_JSON",
    "FORKED_FA2_SO",
    "FR13_FA2_QROW32_B1_SO_SHA256",
    "FR13_FA2_QROW32_B1_SO_SIZE",
    "FR13_FA2_QROW32_B1_FA2_HEAD",
    "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256",
    "FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST",
    "FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST",
)


def _run_full_boot(
    launcher_text, env, so_path, credential_path, so_size,
    profile="k64_root", claim=None, pointer=None,
):
    """Execute regions 0-4 back to back, as a real boot does."""
    import subprocess

    r0, r1, r2, r3, r4 = _boot_regions(launcher_text)
    script = _BOOT_STUBS + r0
    for name, value in _VOCAB_PROFILES[profile].items():
        script += '{}="{}"\n'.format(name, value)
    # the IDENTITY (profile) and the CLAIM are separable on purpose: a boot
    # that claims one and carries the other is the mislabelled serve the
    # helper exists to refuse.
    script += 'FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE="{}"\n'.format(
        profile if claim is None else claim
    )
    script += '_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED={}\n'.format(
        env.get("_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED", "0")
    )
    for name in _BOOT_VARS[1:]:
        script += '{}="{}"\n'.format(name, env.get(name, ""))
    if pointer is not None:
        script += (
            'FR13_B1_CREDENTIAL_POINTER="{}"\n'.format(pointer)
            + '_fr13_b1_load_credential_pointer '
              '"$FR13_B1_CREDENTIAL_POINTER" || exit 2\n'
        )
    script += r1
    # the literals honour caller overrides; re-assert the test's staging after
    script += '_FR13_SPLITK_DEFAULT_SO="{}"\n'.format(so_path)
    script += '_FR13_SPLITK_DEFAULT_CREDENTIAL="{}"\n'.format(credential_path)
    script += '_FR13_SPLITK_DEFAULT_SO_SIZE={}\n'.format(so_size)
    script += r2 + r3 + r4
    script += (
        'printf "TIER_B=%s\\nPRODUCTION=%s\\nCOUNT=%s\\nCANDIDATE=%s\\n'
        'COMMIT=%s\\nPATCH=%s\\n" '
        '"$FR13_FA2_QROW32_B1_TIER_B_ARM" '
        '"$FR13_FA2_QROW32_B1_PRODUCTION_ARM" '
        '"$_FR13_FA2_QROW32_B1_SELECTOR_COUNT" '
        '"${_FR13_FA2_QROW32_B1_CANDIDATE_MODE:-0}" '
        '"$FR13_FA2_QROW32_B1_SOURCE_COMMIT" '
        '"$FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256"\n'
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, cwd=str(REPO)
    )


@pytest.fixture
def staged_boot(tmp_path):
    """A launcher copy whose split-K pins match a staged stub binary."""
    import hashlib

    so = tmp_path / "splitk.so"
    so.write_bytes(b"not the real kernel, but a real file")
    digest = hashlib.sha256(so.read_bytes()).hexdigest()
    size = so.stat().st_size
    credential = tmp_path / "credential.json"
    credential.write_text(_credential_sealed_against(_patcher_digest()))

    def stage(launcher):
        text = launcher.read_text().replace(
            f"_FR13_SPLITK_DEFAULT_SO_SHA256={SPLITK_DEFAULT_SO_SHA256}",
            f"_FR13_SPLITK_DEFAULT_SO_SHA256={digest}", 1,
        )
        assert digest in text, "the split-K sha256 pin moved; restage the walk"
        return text

    return stage, so, credential, size


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_full_boot_survives_the_selector_gate(launcher, staged_boot):
    """F1, the regression proper: a plain hydra27 B1 boot must reach rc=0.

    Before the fix this exited 2 at the selector gate with "requires Hydra27
    B1 and exact binary/source provenance", because the default block armed a
    selector and minted none of the provenance the gate reads.
    """
    stage, so, credential, size = staged_boot
    out = _run_full_boot(stage(launcher), {}, so, credential, size)
    assert out.returncode == 0, out.stderr
    assert f"TIER_B={SPLITK_ARM}" in out.stdout
    assert "COUNT=1" in out.stdout
    assert "CANDIDATE=1" in out.stdout, "the gate never opened; the walk missed it"
    # the provenance the gate reads was MINTED, not left empty
    commit = next(
        l.split("=", 1)[1] for l in out.stdout.splitlines() if l.startswith("COMMIT=")
    )
    patch = next(
        l.split("=", 1)[1] for l in out.stdout.splitlines() if l.startswith("PATCH=")
    )
    assert re.fullmatch(r"[0-9a-f]{40}", commit), f"commit not minted: {commit!r}"
    assert re.fullmatch(r"[0-9a-f]{64}", patch), f"patcher digest not minted: {patch!r}"


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_full_boot_arbitrates_the_two_promoted_defaults(launcher, staged_boot):
    """F2: presenting a gqa_pair credential must not arm BOTH defaults.

    Before the fix this produced SELECTOR_COUNT=2 and "live A/B and production
    arms are mutually exclusive" -- two promoted defaults with no arbitration.
    """
    stage, so, credential, size = staged_boot
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    env = {
        "FR13_FA2_QROW32_B1_GQA_PAIR_GATE_JSON": str(credential),
        "FR13_FA2_QROW32_B1_GQA_PAIR_LIVE_RESULT_JSON": str(credential),
        "FR13_FA2_QROW32_B1_SOURCE_COMMIT": head,
    }
    out = _run_full_boot(stage(launcher), env, so, credential, size)
    assert out.returncode == 0, out.stderr
    assert "COUNT=1" in out.stdout, (
        "both promoted defaults armed: split-K must supersede gqa_pair"
    )
    assert f"TIER_B={SPLITK_ARM}" in out.stdout
    assert "PRODUCTION=\n" in out.stdout + "\n"
    assert "STANDS DOWN" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_full_boot_opt_out_still_reaches_gqa_pair(launcher, staged_boot):
    """The arbitration must not COST the incumbent its own default.

    Naming split-K's opt-out (an explicit production arm) is what round 20's
    A/B does; the incumbent default must still be reachable when the tier-B
    default is suppressed.
    """
    stage, so, credential, size = staged_boot
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    text = stage(launcher).replace(
        "_FR13_SPLITK_DEFAULT_ARM=gqa_pair_splitk",
        "_FR13_SPLITK_DEFAULT_ARM=", 1,
    )
    env = {
        "FR13_FA2_QROW32_B1_GQA_PAIR_GATE_JSON": str(credential),
        "FR13_FA2_QROW32_B1_GQA_PAIR_LIVE_RESULT_JSON": str(credential),
        "FR13_FA2_QROW32_B1_SOURCE_COMMIT": head,
        "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256": _patcher_digest(),
        "FORKED_FA2_SO": str(so),
        "FR13_FA2_QROW32_B1_SO_SIZE": str(size),
    }
    out = _run_full_boot(text, env, so, credential, size)
    assert out.returncode == 0, out.stderr
    assert "PRODUCTION=gqa_pair" in out.stdout
    assert "COUNT=1" in out.stdout


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_full_boot_gate_still_binds_the_patcher_digest(launcher, staged_boot):
    """The reconciliation scoped the COMMIT clause only.

    The patcher digest decides dispatch, so it can change what the kernel
    computes and the credential binds it. A tier-B boot carrying a stale
    patcher digest must still die at the gate.
    """
    stage, so, credential, size = staged_boot
    env = {"FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256": "0" * 64}
    out = _run_full_boot(stage(launcher), env, so, credential, size)
    assert out.returncode == 2
    assert "exact binary/source provenance" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_full_boot_gate_still_binds_the_commit_for_tier_a(launcher, staged_boot):
    """The scoping is TIER-B ONLY: a byte-gated arm still needs HEAD.

    Pass 101 dropped source_commit from the tier-B credential's BINDING fields
    because numerics cannot depend on it. Nothing was dropped from the
    byte-exact route, whose credential is a byte identity earned at a commit.
    """
    stage, so, credential, size = staged_boot
    env = {
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "gqa_pair",
        "FR13_FA2_QROW32_B1_SOURCE_COMMIT": "b" * 40,
        "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256": _patcher_digest(),
        "FORKED_FA2_SO": str(so),
        "FR13_FA2_QROW32_B1_SO_SIZE": str(size),
    }
    # PRODUCTION_ARM named -> the default region is skipped, count=1, tier-A
    env["_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED"] = "1"
    out = _run_full_boot(stage(launcher), env, so, credential, size)
    assert out.returncode == 2
    assert "requires a credential earned at this HEAD" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_full_boot_tier_b_rejects_a_malformed_commit(launcher, staged_boot):
    """The tier-B branch is a WEAKER check, not an absent one."""
    stage, so, credential, size = staged_boot
    env = {"FR13_FA2_QROW32_B1_SOURCE_COMMIT": "not-a-commit"}
    out = _run_full_boot(stage(launcher), env, so, credential, size)
    assert out.returncode == 2
    assert "well-formed source commit" in out.stderr


# --------------------------------------------------- mutation proofs (F1/F2)
#
# Every assertion above is proved to be load-bearing by removing exactly the
# fix it guards and requiring the test to fail. A gate that cannot fail is
# worse than no gate -- this campaign has now written that sentence three
# times, so the walk proves it about itself.


def _drop_block(text, startswith, ends_when):
    """Delete one contiguous block, located positionally.

    The F1 mint blocks have gained comments and a second guard since these
    mutations were first written as literal find/replace strings, and every
    edit to them broke the mutation rather than the fix. A mutation proof that
    breaks whenever the code is touched stops being run, so these locate by
    first line and a closing rule instead.

    Matched on the RAW line, indent included. Stripping first found the
    top-level `VAR=${VAR:-}` normalisation 470 lines earlier instead of the
    in-block mint, and deleted everything between them -- a mutation that
    "worked" while testing nothing. The indent is what distinguishes the two.
    """
    lines = text.split("\n")
    start = next(
        i for i, line in enumerate(lines) if line.startswith(startswith)
    )
    end = next(
        i for i in range(start, len(lines)) if ends_when(lines[i].strip())
    )
    return "\n".join(lines[:start] + lines[end + 1:])


_CLOSING_BRACE = lambda line: line == "}"


def _drop_the_commit_mint(text):
    """Site 15/F1: the mint AND its guard, i.e. the pre-fix state exactly."""
    text = _drop_block(
        text,
        "    FR13_FA2_QROW32_B1_SOURCE_COMMIT=${FR13_FA2_QROW32_B1_SOURCE_COMMIT:-",
        lambda line: line.endswith("}"),
    )
    return _drop_block(
        text, '    [[ -n "$FR13_FA2_QROW32_B1_SOURCE_COMMIT" ]] || {', _CLOSING_BRACE
    )


def _drop_the_patcher_mint(text):
    """Site 16 (a): the sealed-identity mint AND its well-formedness guard."""
    text = _drop_block(
        text,
        "    FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256=${FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256:-",
        lambda line: line.endswith(")}"),
    )
    return _drop_block(
        text,
        '    [[ "$FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {',
        _CLOSING_BRACE,
    )


_MUTATIONS = (
    ("F1: drop the commit mint", _drop_the_commit_mint,
     "well-formed source commit"),
    ("F1: drop the patcher-digest mint", _drop_the_patcher_mint,
     "exact binary/source provenance"),
)


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
@pytest.mark.parametrize("name,mutate,expect", _MUTATIONS, ids=lambda v: str(v)[:28])
def test_f1_mutations_break_the_boot(launcher, staged_boot, name, mutate, expect):
    """Remove the F1 fix -> the plain boot dies at the gate again."""
    stage, so, credential, size = staged_boot
    mutated = mutate(stage(launcher))
    assert mutated != stage(launcher), f"{name}: the mutation changed nothing"
    out = _run_full_boot(mutated, {}, so, credential, size)
    assert out.returncode == 2, f"{name}: the boot survived without the fix"
    assert expect in out.stderr, f"{name}: {out.stderr}"


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_f2_mutation_re_arms_both_defaults(launcher, staged_boot):
    """Remove the stand-down -> two promoted defaults, SELECTOR_COUNT=2."""
    stage, so, credential, size = staged_boot
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    text = stage(launcher)
    find = '  if [[ -n "$FR13_FA2_QROW32_B1_TIER_B_ARM" ]]; then\n'
    assert text.count(find) == 1, "the F2 arbitration anchor is not unique"
    mutated = text.replace(find, '  if false; then\n', 1)
    env = {
        "FR13_FA2_QROW32_B1_GQA_PAIR_GATE_JSON": str(credential),
        "FR13_FA2_QROW32_B1_GQA_PAIR_LIVE_RESULT_JSON": str(credential),
        "FR13_FA2_QROW32_B1_SOURCE_COMMIT": head,
    }
    out = _run_full_boot(mutated, env, so, credential, size)
    assert out.returncode == 2, "without arbitration both defaults must collide"
    assert "mutually exclusive" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_full_boot_tier_b_accepts_a_credential_from_an_older_commit(
    launcher, staged_boot
):
    """The reconciliation's own proof, and the reason it exists.

    Pass 101 re-scoped the tier-B credential's binding fields: source_commit
    became RECORDED, not BINDING, because a commit that touches no kernel
    input cannot change what the kernel computes. The selector gate was still
    enforcing the binding the credential had dropped. This is the boot that
    distinguishes the two -- a valid tier-B credential earned six commits ago,
    which the credential accepts (verify-tier-b rc=0) and the gate refused.
    """
    stage, so, credential, size = staged_boot
    older = subprocess.run(
        ["git", "rev-list", "--max-count=8", "HEAD"], cwd=str(REPO),
        capture_output=True, text=True, check=True,
    ).stdout.split()[-1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert older != head, "need history to distinguish the two rules"
    env = {"FR13_FA2_QROW32_B1_SOURCE_COMMIT": older}
    out = _run_full_boot(stage(launcher), env, so, credential, size)
    assert out.returncode == 0, out.stderr
    assert f"TIER_B={SPLITK_ARM}" in out.stdout
    assert f"COMMIT={older}" in out.stdout, "the presented commit was overwritten"

    # ... and removing the reconciliation puts the refusal back.
    find = '  [[ -z "$FR13_FA2_QROW32_B1_TIER_B_ARM" ]] || _fr13_b1_commit_bound=0\n'
    text = stage(launcher)
    assert text.count(find) == 1, "the reconciliation anchor is not unique"
    out = _run_full_boot(text.replace(find, "", 1), env, so, credential, size)
    assert out.returncode == 2, "the gate no longer enforces anything for tier-A"
    assert "requires a credential earned at this HEAD" in out.stderr


# ---------------------------------------------------- site 12 (pass 113)
#
# The full-boot walk above ran only k64_root, because before pass 113 the
# forks could not run anything else -- their B1 selector hard-coded
# ROOT==1 && K==65536 && BLOCKS==<pinned> instead of calling the profile
# helper, and K0 full-vocab is the shape split-K has actually served in.
# The runner's boot armed the promoted default for the first time, mint and
# arbitration both fired, and it died one gate later on this.


@pytest.mark.parametrize("profile", sorted(_VOCAB_PROFILES))
@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_full_boot_survives_under_either_vocab_profile(
    launcher, profile, staged_boot
):
    """Site 12 proper: the promoted default must boot under K0 too."""
    stage, so, credential, size = staged_boot
    out = _run_full_boot(
        stage(launcher), {}, so, credential, size, profile=profile
    )
    assert out.returncode == 0, out.stderr
    assert f"TIER_B={SPLITK_ARM}" in out.stdout
    assert "CANDIDATE=1" in out.stdout


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_the_vocab_profile_is_still_enforced_not_merely_delegated(
    launcher, staged_boot
):
    """Delegating is not the same as dropping.

    The conversion must keep the identity BOUND -- it widens which identity is
    admissible, it does not stop asking. A boot that claims full_vocab while
    carrying the K64 identity is exactly the mislabelled serve the tier-B
    route exists to prevent.
    """
    stage, so, credential, size = staged_boot
    r0, _r1, _r2, _r3, _r4 = _boot_regions(stage(launcher))
    assert "_fr13_assert_draft_vocab_profile() {" in r0, (
        "the walk lifted no helper; it would be testing a stub"
    )
    # claim full_vocab, present the k64_root identity
    out = _run_full_boot(
        stage(launcher), {}, so, credential, size,
        profile="k64_root", claim="full_vocab",
    )
    assert out.returncode == 2
    assert "requires the full_vocab draft-vocabulary identity" in out.stderr

    # ... and the reverse
    out = _run_full_boot(
        stage(launcher), {}, so, credential, size,
        profile="full_vocab", claim="k64_root",
    )
    assert out.returncode == 2
    assert "requires the k64_root draft-vocabulary identity" in out.stderr

    # and an unknown profile is refused outright, not defaulted
    out = _run_full_boot(
        stage(launcher), {}, so, credential, size, claim="k64",
    )
    assert out.returncode == 2
    assert "must be exactly k64_root or full_vocab" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_reintroducing_the_b1_hardcode_kills_the_k0_boot(launcher, staged_boot):
    """The mutation proof, run as a BOOT rather than as a scan.

    The parity scan proves the detector sees the hardcode. This proves what
    the hardcode costs: a K0 boot that refuses. It is the runner's failure,
    reproduced on CPU.
    """
    stage, so, credential, size = staged_boot
    text = stage(launcher)
    call = (
        '_fr13_assert_draft_vocab_profile \\\n'
        '    "$FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE" '
        '"FR13 qrow32 B1 selector" || exit 2\n'
    )
    assert text.count(call) == 1, "the B1 conversion moved; restage"
    lines = text.replace(call, "", 1).split("\n")
    err = next(
        i for i, line in enumerate(lines)
        if "FR13 qrow32 B1 selector requires Hydra27" in line
    )
    start = next(
        i for i in range(err, err - 80, -1) if re.match(r"^\s*\[\[ ", lines[i])
    )
    mutated = "\n".join(
        lines[:start + 1]
        + ['     && "${FR13_DRAFT_VOCAB_K:-65536}" == "65536" \\']
        + lines[start + 1:]
    )
    # k64_root still boots -- which is exactly why this survived so long
    out = _run_full_boot(mutated, {}, so, credential, size, profile="k64_root")
    assert out.returncode == 0, out.stderr
    # ... and K0 does not
    out = _run_full_boot(mutated, {}, so, credential, size, profile="full_vocab")
    assert out.returncode == 2, "the hardcode came back and the K0 boot survived"
    assert "exact binary/source provenance" in out.stderr


# ===========================================================================
# THE TIER-B CANONICAL WORKLOAD TABLE (site: exact16 blocked by the promotion's
# own gate).
#
# Pass 74 ruled that a tier-B serve carries the canonical campaign identity,
# and the gate encoded that as the exact4 pins, hard-coded. So exact16 -- the
# QC that verifies the split-K promotion -- could not be declared at all.
#
# Extended, not bypassed: the identity is still mandatory and still exact,
# there is now more than one canonical workload, and the caller must NAME the
# one it is running. Default stays exact4, so every existing caller keeps its
# previous meaning exactly.
# ===========================================================================

WORKLOADS = {
    "exact4": (4, "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"),
    "exact16": (16, "47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c"),
    "random1024_calibration": (0, ""),
}


def _workload_gate(launcher_text):
    """The tier-B serve gate, executable, with the table it resolves from."""
    start = launcher_text.index(
        "# ---------------------------------------- tier-B CANONICAL WORKLOAD"
    )
    end = launcher_text.index("\n}\n", start) + len("\n}\n")
    table = launcher_text[start:end]
    gate_start = launcher_text.index(
        "  # SPELLING (runner's dry-read, before this ever booted)."
    )
    gate_end = launcher_text.index("<none: synthetic shape", gate_start)
    gate_end = launcher_text.index("\n", gate_end) + 1
    return table, launcher_text[gate_start:gate_end]


def _run_workload_gate(launcher, workload, ids, sha, legacy=("", ""), spelling="new"):
    """Drive the gate. `spelling` picks which variable names carry the pins."""
    import subprocess

    fresh_ids, fresh_sha = (ids, sha) if spelling == "new" else ("", "")
    old_ids, old_sha = (ids, sha) if spelling == "legacy" else legacy
    table, gate = _workload_gate(launcher.read_text())
    script = (
        "set -u\n"
        f'FR13_FA2_QROW32_B1_TIERB_WORKLOAD="{workload}"\n'
        f'FR13_FA2_QROW32_B1_TIERB_TASK_IDS="{fresh_ids}"\n'
        f'FR13_FA2_QROW32_B1_TIERB_SUBSET_SHA256="{fresh_sha}"\n'
        f'FR13_FA2_QROW32_B1_EXACT4_TASK_IDS="{old_ids}"\n'
        f'FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256="{old_sha}"\n'
        'FR13_FIXED32_B1_DIAGNOSTIC=0\nENFORCE_EAGER=0\n'
        'CUDAGRAPH_MODE=FULL_AND_PIECEWISE\n'
        + table.replace(
            "FR13_FA2_QROW32_B1_TIERB_WORKLOAD=${FR13_FA2_QROW32_B1_TIERB_WORKLOAD:-exact4}",
            "",
        )
        + gate
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, cwd=str(REPO)
    )


def _subset_ids(name):
    return ",".join(
        json.loads((REPO / "config/fr13_fixed32" / name).read_text())["instance_ids"]
    )


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_the_workload_table_admits_every_canonical_workload(launcher):
    """exact4 (unchanged), exact16 (the point), random1024 (the honest one)."""
    for workload, (count, sha) in WORKLOADS.items():
        if workload == "exact4":
            ids = _subset_ids("subset_b4_four.json")
        elif workload == "exact16":
            ids = _subset_ids("subset_b4_sixteen.json")
        else:
            ids = ""
        assert len([t for t in ids.split(",") if t]) == count
        out = _run_workload_gate(launcher, workload, ids, sha)
        assert out.returncode == 0, f"{workload}: {out.stderr}"
        assert f"workload={workload}" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_the_default_workload_is_still_exact4(launcher):
    """Compatibility: a caller that names nothing means exactly what it did."""
    text = launcher.read_text()
    assert (
        "FR13_FA2_QROW32_B1_TIERB_WORKLOAD=${FR13_FA2_QROW32_B1_TIERB_WORKLOAD:-exact4}"
        in text
    )


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_claiming_exact16_with_exact4_pins_refuses(launcher):
    """Mutation proof, direction one."""
    out = _run_workload_gate(
        launcher, "exact16",
        _subset_ids("subset_b4_four.json"), WORKLOADS["exact4"][1],
    )
    assert out.returncode == 2
    assert "FULL-graph identity of its DECLARED workload" in out.stderr
    assert "TIERB_WORKLOAD=exact16" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_claiming_exact4_with_exact16_pins_refuses(launcher):
    """Mutation proof, direction two."""
    out = _run_workload_gate(
        launcher, "exact4",
        _subset_ids("subset_b4_sixteen.json"), WORKLOADS["exact16"][1],
    )
    assert out.returncode == 2
    assert "FULL-graph identity of its DECLARED workload" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_an_unknown_workload_refuses(launcher):
    """No default-on-unknown: site 23's lesson, in a new table."""
    for bogus in ("exact8", "", "EXACT4", "random1024"):
        out = _run_workload_gate(
            launcher, bogus, _subset_ids("subset_b4_four.json"),
            WORKLOADS["exact4"][1],
        )
        assert out.returncode != 0, bogus
        if bogus:
            assert "must be one of" in out.stderr, bogus


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_random1024_may_not_borrow_a_subset_identity(launcher):
    """The pins-as-fiction finding, closed.

    Measurement 1 declared the exact4 pins while driving sglang's random
    1024/1024 shape, because the gate could not be satisfied otherwise. A gate
    satisfiable only by a false declaration manufactures false provenance.
    """
    out = _run_workload_gate(
        launcher, "random1024_calibration",
        _subset_ids("subset_b4_four.json"), WORKLOADS["exact4"][1],
    )
    assert out.returncode == 2
    assert "TIERB_WORKLOAD=random1024_calibration" in out.stderr
    # ... and declared honestly it passes, naming itself as no subset
    out = _run_workload_gate(launcher, "random1024_calibration", "", "")
    assert out.returncode == 0, out.stderr
    assert "no SWE subset" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_each_subset_row_is_bound_to_the_file_it_names(launcher):
    """The table cannot drift from config/ the way site 14's drifted."""
    text = launcher.read_text()
    for name, (_count, sha) in (
        ("subset_b4_four.json", WORKLOADS["exact4"]),
        ("subset_b4_sixteen.json", WORKLOADS["exact16"]),
    ):
        assert f"config/fr13_fixed32/{name}" in text
        import hashlib

        on_disk = hashlib.sha256(
            (REPO / "config/fr13_fixed32" / name).read_bytes()
        ).hexdigest()
        assert on_disk == sha, f"{name} no longer hashes to the pinned {sha}"
    assert 'sha256sum "$_fr13_tierb_subset_file"' in text, (
        "the table names its subset files but never re-hashes them, so it can "
        "drift from config/ exactly as the floor table drifted from the ledger"
    )


def test_the_container_side_table_agrees_with_the_launchers():
    """Two halves, one workload. A serve whose halves disagree is mislabelled."""
    namespace = _selectors()
    table = namespace["_FR13_FA2_QROW32_B1_TIER_B_WORKLOADS"]
    assert set(table) == set(WORKLOADS)
    assert namespace["_FR13_FA2_QROW32_B1_TIER_B_DEFAULT_WORKLOAD"] == "exact4"
    for workload, (count, sha) in WORKLOADS.items():
        ids, got_sha = table[workload]
        assert got_sha == sha, workload
        assert len([t for t in ids.split(",") if t]) == count, workload
    for launcher in LAUNCHERS:
        text = launcher.read_text()
        for workload, (_count, sha) in WORKLOADS.items():
            assert workload in text
            if sha:
                assert sha in text


def test_the_container_side_records_and_cross_checks_the_workload(monkeypatch):
    namespace = _selectors()
    resolve = namespace["_fr13_fa2_qrow32_b1_tier_b_workload"]
    ids16 = _subset_ids("subset_b4_sixteen.json")

    monkeypatch.setenv("FR13_FA2_QROW32_B1_TIERB_WORKLOAD", "exact16")
    monkeypatch.setenv("FR13_FA2_QROW32_B1_EXACT4_TASK_IDS", ids16)
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256", WORKLOADS["exact16"][1]
    )
    record = resolve()
    assert record["declared"] == "exact16"
    assert record["task_count"] == 16
    assert record["is_swe_subset"] is True

    # the cross-check: name one workload, carry another's pins
    monkeypatch.setenv("FR13_FA2_QROW32_B1_TIERB_WORKLOAD", "exact4")
    with pytest.raises(RuntimeError, match="declares workload 'exact4'"):
        resolve()

    # and the honest synthetic row
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_TIERB_WORKLOAD", "random1024_calibration"
    )
    monkeypatch.setenv("FR13_FA2_QROW32_B1_EXACT4_TASK_IDS", "")
    monkeypatch.setenv("FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256", "")
    record = resolve()
    assert record["is_swe_subset"] is False and record["task_count"] == 0

    monkeypatch.setenv("FR13_FA2_QROW32_B1_TIERB_WORKLOAD", "exact8")
    with pytest.raises(RuntimeError, match="must be one of"):
        resolve()


def test_the_serve_record_carries_the_workload():
    """The declared workload is recorded as THE workload of the serve."""
    source = PATCHER.read_text()
    assert '"tier_b_workload": (' in source
    assert "_fr13_fa2_qrow32_b1_tier_b_workload() if tier_b else None" in source


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_the_legacy_spelling_still_works_alone(launcher):
    """Banked vehicles set FR13_FA2_QROW32_B1_EXACT4_*; they keep working."""
    out = _run_workload_gate(
        launcher, "exact4", _subset_ids("subset_b4_four.json"),
        WORKLOADS["exact4"][1], spelling="legacy",
    )
    assert out.returncode == 0, out.stderr
    assert "workload=exact4" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_the_two_spellings_must_agree_when_both_are_set(launcher):
    """The operator mistake the rename exists to name.

    Declaring exact16 in the new spelling while the legacy variable still
    holds the four exact4 ids is exactly what a copied boot script produces.
    Before the rename it refused with a message about ids that said nothing
    about the naming; now it says which two variables disagree.
    """
    out = _run_workload_gate(
        launcher, "exact16", _subset_ids("subset_b4_sixteen.json"),
        WORKLOADS["exact16"][1],
        legacy=(_subset_ids("subset_b4_four.json"), WORKLOADS["exact4"][1]),
    )
    assert out.returncode == 2
    assert "are both set and disagree" in out.stderr
    assert "FR13_FA2_QROW32_B1_TIERB_TASK_IDS" in out.stderr
    assert "FR13_FA2_QROW32_B1_EXACT4_TASK_IDS" in out.stderr

    # agreeing duplicates are fine -- a vehicle that sets both correctly boots
    ids4 = _subset_ids("subset_b4_four.json")
    out = _run_workload_gate(
        launcher, "exact4", ids4, WORKLOADS["exact4"][1],
        legacy=(ids4, WORKLOADS["exact4"][1]),
    )
    assert out.returncode == 0, out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_the_tierb_gate_no_longer_reads_the_exact4_names_directly(launcher):
    """The self-contradiction is gone from the predicate itself."""
    text = launcher.read_text()
    start = text.index("  # SPELLING (runner's dry-read")
    end = text.index("<none: synthetic shape", start)
    gate = text[start:end]
    predicate = gate[gate.index("_fr13_b1_tierb_workload_pins"):]
    assert "FR13_FA2_QROW32_B1_EXACT4_TASK_IDS" not in predicate, (
        "the workload-keyed predicate still reads a variable named exact4"
    )
    assert '"$_fr13_tierb_declared_ids" == "$_fr13_tierb_task_ids"' in predicate


def test_the_container_side_alias_refuses_a_disagreement(monkeypatch):
    namespace = _selectors()
    resolve = namespace["_fr13_fa2_qrow32_b1_tier_b_workload"]
    monkeypatch.setenv("FR13_FA2_QROW32_B1_TIERB_WORKLOAD", "exact16")
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_TIERB_TASK_IDS", _subset_ids("subset_b4_sixteen.json")
    )
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_TIERB_SUBSET_SHA256", WORKLOADS["exact16"][1]
    )
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_EXACT4_TASK_IDS", _subset_ids("subset_b4_four.json")
    )
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256", WORKLOADS["exact4"][1]
    )
    with pytest.raises(RuntimeError, match="are both set and disagree"):
        resolve()

    # legacy alone still resolves, for banked vehicles
    monkeypatch.delenv("FR13_FA2_QROW32_B1_TIERB_TASK_IDS")
    monkeypatch.delenv("FR13_FA2_QROW32_B1_TIERB_SUBSET_SHA256")
    monkeypatch.setenv("FR13_FA2_QROW32_B1_TIERB_WORKLOAD", "exact4")
    assert resolve()["declared"] == "exact4"


# ===========================================================================
# SITE 15 -- arming is not owning.
#
# exact16 refused in 5 seconds. The promoted default armed (F1's mint worked),
# gqa_pair stood down (F2's arbitration worked, the log line printed) -- and
# the selector gate then measured split-K's 300,123,792-byte binary against
# the INCUMBENT's 299,815,552, because _fr13_b1_load_credential_pointer had
# auto-imported the gqa_pair credential env ~500 lines earlier, whenever the
# pointer file exists, with no arm named. The default block's
# ${VAR:-literal} fallbacks only fill EMPTY variables, so the leftovers won.
#
# Corroborated by which checks passed: the block's own literal-vs-disk check
# passed (it reads the literal), the gate's variable-vs-disk check failed.
#
# THE LESSON THAT COMPLETES F1/F2: standing down as an arm does not withdraw
# the pins it already imported.
# ===========================================================================

INCUMBENT_SO_SIZE = "299815552"
INCUMBENT_SO_SHA256 = (
    "af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb"
)


@pytest.fixture
def credential_pointer(tmp_path):
    """A pointer file carrying the incumbent gqa_pair credential env."""
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    pointer = tmp_path / "fr13_b1_gqa_pair_credential.env"
    pointer.write_text(
        "\n".join(
            [
                f"FR13_FA2_QROW32_B1_SO_SHA256={INCUMBENT_SO_SHA256}",
                f"FR13_FA2_QROW32_B1_SO_SIZE={INCUMBENT_SO_SIZE}",
                f"FR13_FA2_QROW32_B1_SOURCE_COMMIT={head}",
                "FR13_FA2_QROW32_B1_FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95",
                "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256=" + "9" * 64,
                "FR13_FA2_QROW32_B1_GQA_PAIR_GATE_JSON=/logs/gate.json",
                "FR13_FA2_QROW32_B1_GQA_PAIR_GATE_SHA256=" + "a" * 64,
                "FR13_FA2_QROW32_B1_GQA_PAIR_LIVE_RESULT_JSON=/logs/live.json",
                "",
            ]
        )
    )
    return pointer


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_full_boot_with_a_pointer_present_and_no_arm_named(
    launcher, staged_boot, credential_pointer
):
    """(c) THE EXACT SHAPE THAT BURNED: pointer present, arm unnamed, rc=0.

    Before the fix this reached the selector gate carrying the incumbent's
    299,815,552 and died comparing it to the staged split-K binary.
    """
    stage, so, credential, size = staged_boot
    out = _run_full_boot(
        stage(launcher), {}, so, credential, size,
        pointer=credential_pointer,
    )
    assert out.returncode == 0, out.stderr
    assert f"TIER_B={SPLITK_ARM}" in out.stdout
    assert "COUNT=1" in out.stdout
    assert "CANDIDATE=1" in out.stdout, "the gate never opened; the walk missed it"


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_the_default_owns_its_binary_pins_rather_than_defaulting_them(
    launcher, staged_boot, credential_pointer
):
    """The root fix, asserted on the resolved values rather than the source.

    Whatever the pointer imported, the pins the gate reads must be the
    default's own. Under a stub binary the walk restages the sha pin, so the
    SIZE pin is the one checked here against the incumbent's.
    """
    stage, so, credential, size = staged_boot
    out = _run_full_boot(
        stage(launcher), {}, so, credential, size,
        pointer=credential_pointer,
    )
    assert out.returncode == 0, out.stderr
    assert INCUMBENT_SO_SIZE not in out.stdout
    assert INCUMBENT_SO_SHA256 not in out.stdout


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_pointer_absent_and_unnamed_is_unchanged(launcher, staged_boot):
    """(c) The regression check on the path that already worked."""
    stage, so, credential, size = staged_boot
    out = _run_full_boot(stage(launcher), {}, so, credential, size, pointer=None)
    assert out.returncode == 0, out.stderr
    assert f"TIER_B={SPLITK_ARM}" in out.stdout


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_pointer_present_with_gqa_pair_named_is_unchanged(
    launcher, staged_boot, credential_pointer
):
    """(c) Naming gqa_pair explicitly must still consume the pointer.

    The withdrawal is scoped to the branch that ARMS the tier-B default. A
    named production arm never enters it, so the pointer keeps doing the job
    it was built for -- convenience for typing.
    """
    stage, so, credential, size = staged_boot
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    env = {
        "_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED": "1",
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "gqa_pair",
        "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256": _patcher_digest(),
        "FORKED_FA2_SO": str(so),
        "FR13_FA2_QROW32_B1_SO_SIZE": str(size),
        # production would get this from the pointer; the no-middleware twins
        # have no pointer at all, so the caller supplies it. The test is about
        # the withdrawal being SCOPED to the arming branch, not about plumbing.
        "FR13_FA2_QROW32_B1_SOURCE_COMMIT": head,
    }
    out = _run_full_boot(
        stage(launcher), env, so, credential, size, pointer=credential_pointer,
    )
    assert out.returncode == 0, out.stderr
    assert "PRODUCTION=gqa_pair" in out.stdout
    assert "TIER_B=\n" in out.stdout + "\n"


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_restoring_the_defaulted_pins_reproduces_site_15(
    launcher, staged_boot, credential_pointer
):
    """Mutation proof: put the ':-' back, the burning shape burns again."""
    stage, so, credential, size = staged_boot
    text = stage(launcher)
    for name, literal in (
        ("FR13_FA2_QROW32_B1_SO_SIZE", "$_FR13_SPLITK_DEFAULT_SO_SIZE"),
        ("FR13_FA2_QROW32_B1_SO_SHA256", "$_FR13_SPLITK_DEFAULT_SO_SHA256"),
    ):
        owned = f"    {name}={literal}\n"
        assert text.count(owned) == 1, f"{name} is no longer owned unconditionally"
        text = text.replace(owned, f"    {name}=${{{name}:-{literal}}}\n", 1)
    # ... and remove the withdrawal, so neither half of the fix is present
    call = "    _fr13_b1_withdraw_pointer_imports\n"
    assert text.count(call) == 1
    text = text.replace(call, "", 1)
    out = _run_full_boot(
        text, {}, so, credential, size, pointer=credential_pointer,
    )
    if "_fr13_b1_load_credential_pointer() {" not in launcher.read_text():
        # the no-middleware twins have no pointer at all: nothing to import,
        # so the mutation cannot bite there and the boot still succeeds. That
        # asymmetry is the finding, not a gap in the test.
        assert out.returncode == 0, out.stderr
        return
    assert out.returncode == 2, "site 15 did not reproduce without the fix"
    assert "exact binary/source provenance" in out.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_the_withdrawal_only_takes_back_what_the_pointer_set(
    launcher, staged_boot, credential_pointer
):
    """A blanket unset would erase a hand-typed credential.

    The loader records the names it actually assigned, so a caller value the
    pointer skipped -- 'the caller always wins' -- survives the withdrawal.
    """
    text = launcher.read_text()
    assert '_FR13_B1_POINTER_IMPORTED+=("$name")' in text or (
        "_fr13_b1_load_credential_pointer() {" not in text
    ), "the loader does not record what it imported"
    stage, so, credential, size = staged_boot
    # a caller-supplied credential path is never withdrawn
    out = _run_full_boot(
        stage(launcher),
        {"FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_HOST": str(credential)},
        so, credential, size, pointer=credential_pointer,
    )
    assert out.returncode == 0, out.stderr


def test_the_pointer_whitelist_entry_that_can_never_fire():
    """A finding, recorded so it is not re-derived from a boot.

    FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE is on the pointer's whitelist,
    but the launcher defaults it to k64_root BEFORE the pointer runs, and the
    pointer only fills EMPTY names ("the caller always wins"). So that entry
    is unreachable: the pointer can never set the vocabulary profile. This is
    asserted rather than fixed because removing a whitelist entry narrows what
    a pointer may carry, and the pointer's contract is not mine to change.
    """
    text = (REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh").read_text()
    default_at = text.index(
        "FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE="
        "${FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE:-k64_root}"
    )
    loader_at = text.index(
        '_fr13_b1_load_credential_pointer "$FR13_B1_CREDENTIAL_POINTER"'
    )
    assert default_at < loader_at, (
        "the QUALIFICATION_PROFILE default no longer precedes the pointer -- "
        "the whitelist entry may now be live, which changes what a pointer can "
        "decide about the served vocabulary"
    )


# ===========================================================================
# SITE 17 -- the serve gate never learned the workload table.
#
# exact16 attempt 3 ran 4m16s to engine init and died at the first served
# token: _fr13_fa2_qrow32_b1_require_exact4 read ONLY the legacy EXACT4_*
# spelling and compared against the hardcoded canonical four. No caller value
# could satisfy it -- sixteen ids are not four, and unset is not four either.
#
# The workload-table landing converted the RECORD accessor and never this
# GATE, 600 lines away in the same file. The record learned exact16; the gate
# did not. The fix is not to teach the gate the table: it is to stop the gate
# knowing what a workload IS, and make it ask the one accessor that does.
# ===========================================================================


def _served_workload_env(monkeypatch, credential_path, workload, ids, sha):
    return _tier_b_env(
        monkeypatch, credential_path,
        FR13_FA2_QROW32_B1_TIERB_WORKLOAD=workload,
        FR13_FA2_QROW32_B1_TIERB_TASK_IDS=ids,
        FR13_FA2_QROW32_B1_TIERB_SUBSET_SHA256=sha,
        FR13_FA2_QROW32_B1_EXACT4_TASK_IDS="",
        FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256="",
    )


@pytest.mark.parametrize(
    "workload,subset",
    [("exact4", "subset_b4_four.json"), ("exact16", "subset_b4_sixteen.json")],
)
def test_the_serve_gate_admits_every_declared_workload(
    monkeypatch, tmp_path, workload, subset
):
    """SITE 17 proper: exact16 must reach the serve gate and pass it."""
    _install_fake_vllm(monkeypatch)
    namespace = _fresh_selectors()
    path, _payload = _credential(tmp_path)
    begin = namespace["_fr13_fa2_qrow32_b1_production_begin"]
    ids = _subset_ids(subset)
    sha = WORKLOADS[workload][1]
    with _served_workload_env(monkeypatch, path, workload, ids, sha):
        begin(layer=_Layer(), **_served_operands())
    record = namespace["_fr13_fa2_qrow32_b1_tier_b_workload"]()
    assert record["declared"] == workload
    assert record["task_count"] == WORKLOADS[workload][0]


def test_the_serve_gate_admits_the_calibration_workload(monkeypatch, tmp_path):
    """Measurement 1 rides this path: no subset, and that must be sayable."""
    _install_fake_vllm(monkeypatch)
    namespace = _fresh_selectors()
    path, _payload = _credential(tmp_path)
    begin = namespace["_fr13_fa2_qrow32_b1_production_begin"]
    with _served_workload_env(
        monkeypatch, path, "random1024_calibration", "", ""
    ):
        begin(layer=_Layer(), **_served_operands())


def test_the_serve_gate_still_accepts_a_legacy_only_caller(monkeypatch, tmp_path):
    """Banked vehicles set only EXACT4_*; nothing about them changes."""
    _install_fake_vllm(monkeypatch)
    namespace = _fresh_selectors()
    path, _payload = _credential(tmp_path)
    begin = namespace["_fr13_fa2_qrow32_b1_production_begin"]
    with _tier_b_env(monkeypatch, path):  # exact4 pins, legacy spelling, no workload
        begin(layer=_Layer(), **_served_operands())


def test_the_serve_gate_refuses_a_workload_its_pins_do_not_match(
    monkeypatch, tmp_path
):
    """The gate inherits the cross-check rather than restating it."""
    _install_fake_vllm(monkeypatch)
    namespace = _fresh_selectors()
    path, _payload = _credential(tmp_path)
    begin = namespace["_fr13_fa2_qrow32_b1_production_begin"]
    with _served_workload_env(
        monkeypatch, path, "exact16",
        _subset_ids("subset_b4_four.json"), WORKLOADS["exact4"][1],
    ):
        with pytest.raises(RuntimeError, match="declares workload 'exact16'"):
            begin(layer=_Layer(), **_served_operands())


def test_tier_a_production_is_still_ruled_to_exact4(monkeypatch, tmp_path):
    """Widening the tier-B route must not widen the byte-gated one.

    The launcher's production gate pins the canonical four independently; this
    is the patcher-side half of the same rule, so a tier-A serve cannot borrow
    the tier-B workload table to serve sixteen tasks.
    """
    namespace = _fresh_selectors()
    require = namespace["_fr13_fa2_qrow32_b1_require_declared_workload"]
    monkeypatch.setenv("FR13_FA2_QROW32_B1_TIERB_WORKLOAD", "exact16")
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_TIERB_TASK_IDS", _subset_ids("subset_b4_sixteen.json")
    )
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_TIERB_SUBSET_SHA256", WORKLOADS["exact16"][1]
    )
    monkeypatch.setenv("FR13_FA2_QROW32_B1_EXACT4_TASK_IDS", "")
    monkeypatch.setenv("FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256", "")
    assert require("B")["declared"] == "exact16"
    with pytest.raises(RuntimeError, match="tier A. serves the canonical exact4"):
        require("A")


def test_the_gate_and_the_record_share_one_accessor():
    """The structural claim, so a second copy cannot reappear quietly."""
    source = PATCHER.read_text()
    # Keyed on the DEFINITION and on call sites, not on the bare name: the
    # replacement's docstring names the function it replaced, and a check that
    # a comment can satisfy is the mistake this campaign has now made twice.
    assert "def _fr13_fa2_qrow32_b1_require_exact4(" not in source, (
        "the bespoke exact4 gate is back"
    )
    assert "_fr13_fa2_qrow32_b1_require_exact4()" not in source, (
        "something still calls the bespoke exact4 gate"
    )
    body = source[source.index("def _fr13_fa2_qrow32_b1_require_declared_workload"):]
    body = body[:body.index("\ndef ")]
    assert "_fr13_fa2_qrow32_b1_tier_b_workload()" in body
    assert "_FR13_FA2_QROW32_B1_CANONICAL_TASK_IDS" not in body, (
        "the gate is comparing against a hardcoded task list again"
    )
    # and the record states what it served rather than a literal
    record = source[source.index('"pass_sidecar_sha256"'):]
    record = record[:record.index('"fallback_allowed"')]
    assert '_fr13_fa2_qrow32_b1_tier_b_workload()["task_ids"]' in record


# ------------------------------------------------- site 16: the sealed mint


def test_the_launcher_mints_the_patcher_digest_from_the_sealed_identity():
    """SITE 16 (a). A value derived from an artifact cannot test it.

    The mint used to hash scripts/fr13_patch_fa2_tree_bias.py -- the same file
    the selector gate then compares against disk. That made the gate
    disk-vs-disk: `x == x`, unfailable however stale the credential was. It is
    now minted from the credential's SEALED identity, so the gate asks the
    question it looks like it is asking.
    """
    for launcher in LAUNCHERS:
        text = launcher.read_text()
        start = text.index(
            "    FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256=${FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256:-"
        )
        mint = text[start:text.index("\n    [[", start)]
        assert "sha256sum scripts/fr13_patch_fa2_tree_bias.py" not in mint, (
            f"{launcher.name}: the mint still hashes the artifact the gate checks"
        )
        assert "patch_source_sha256" in mint and "CREDENTIAL_HOST" in mint
        # the gate it feeds still compares against disk
        assert (
            '"$FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256" == "$(sha256sum '
            "scripts/fr13_patch_fa2_tree_bias.py" in text
        )


def test_the_staged_credential_is_sealed_against_this_patcher():
    """SITE 16 (b). The coupling, watched on the REAL credential.

    This is the pair the promoted default boots on: results/.../
    fr14_splitk_tierb_credential.json and scripts/fr13_patch_fa2_tree_bias.py.
    Because the mint is now sealed-vs-disk, a patcher edit that lands without
    a re-seal makes every promoted-default boot refuse -- correctly, and
    silently until someone burns a GPU window on it. So it is asserted here,
    on CPU, and it is expected to fail in exactly one window: between a patcher
    edit landing and the runner re-sealing. That is not a flaky test; it is
    the coupling being visible.
    """
    credential = json.loads(
        (REPO / "results/fr14_nvfp4_port_20260816"
         / "fr14_splitk_tierb_credential.json").read_text()
    )
    sealed = credential["identity"]["patch_source_sha256"]
    assert sealed == _patcher_digest(), (
        "the staged tier-B credential is sealed against patcher "
        f"{sealed[:16]}... but the tree carries {_patcher_digest()[:16]}...; "
        "the promoted default will refuse every boot until it is re-sealed"
    )
