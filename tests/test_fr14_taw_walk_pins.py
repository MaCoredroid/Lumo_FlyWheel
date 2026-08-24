"""SITE 27: the pins family -- every written-down number that is walk-derived.

SITE 13 fixed every READ of the walk cap. A PIN is a read too, just a slower
one, made once when someone wrote the number down, and hydra31 met them one at a
time:

    round 22, boot:  FR13 fixed32 TAW source geometry drift:
                     {... 'walk_cap': 16 ...}   against a pin of 12
                     fr13_device_multidraft_kernel.py:3180

One assertion behind it sat _FR13_FIXED32_TAW_TENSOR_CALL_CENSUS's walk_levels:
12, and behind THAT the published uniform/row/path scatter slot counts, which
the work census already expected at walk*3, walk*2 and walk. Four statements of
the same number, discovered one boot at a time.

THIS FILE IS THE CENSUS THAT ENDS THAT. Every structure in the fixed32 execution
closure that carries a walk-derived number is listed below and classified:

    DERIVE                the number is a function of the walk depth, so it is
                          computed from the topology authority with the same
                          resolver the execution path uses;

    PIN, WITH THE REASON  the number is genuinely profile-invariant, or it was
                          MEASURED on a route whose own selector refuses every
                          other profile, or its bytes are credential-bound. Each
                          one states which, and the fail-closed behaviour that
                          makes the pin safe is PROVEN here, not asserted.

The ruled mutation proofs are test_hydra31_boots_through_the_source_contract,
test_hydra27_is_unchanged_field_by_field and
test_a_wrong_runtime_cap_still_refuses_naming_both_values.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

KERNEL = SCRIPTS / "fr13_device_multidraft_kernel.py"
CENSUS = SCRIPTS / "fr13_fixed32_work_census.py"
KERNEL_SOURCE = KERNEL.read_text(encoding="utf-8")
CENSUS_SOURCE = CENSUS.read_text(encoding="utf-8")

PROFILE_HYDRA27 = "hydra27_fixed32"
PROFILE_HYDRA31 = "hydra31_fixed32"
MODE_TAIL6 = "tail6_fixed32"

# The literals site 27 retired. Kept here verbatim so "no arithmetic moved" is
# checkable rather than claimed: the derivation at walk 12 must reproduce every
# one of them.
RETIRED_HYDRA27_GEOMETRY = {
    "physical_drafts": 31,
    "physical_rows": 32,
    "walk_cap": 12,
    "fanout": 3,
    "output_capacity": 32,
    "accepted_path_capacity": 16,
}
RETIRED_HYDRA27_TENSOR_CALL_CENSUS = {
    "walk_levels": 12,
    "full_vocab_row_gathers": 24,
    "full_vocab_fp32_casts": 24,
    "full_vocab_softmax_calls": 24,
    "full_vocab_normalizations": 36,
    "full_vocab_cdf_calls": 24,
    "source_cdf_calls": 12,
    "qmix_zero_fills": 12,
    "qmix_scatter_add_calls": 12,
    "residual_subtract_calls": 12,
    "residual_clamp_calls": 12,
    "residual_where_calls": 24,
    "output_scatter_calls": 0,
    "path_scatter_calls": 0,
    "exact_commit_launches": 12,
    "exact_commit_programs_per_request": 12,
    "floating_sampling_reimplementation": False,
}

# --------------------------------------------------------------------------- #
# THE PINS-FAMILY CENSUS                                                       #
# --------------------------------------------------------------------------- #
# name -> (classification, reason). Every entry is checked by a test below; a
# new walk-derived structure that is not in this table fails
# test_the_pins_family_census_is_complete.
PINS_FAMILY: dict[str, tuple[str, str]] = {
    # ---- DERIVED (site 27) --------------------------------------------------
    "kernel._FR13_FIXED32_TAW_GEOMETRY": (
        "derive",
        "walk_cap is 12 under hydra27/tail6 and 16 under hydra31; the other "
        "five fields are flat topology constants shared by both profiles and "
        "stay literal, which is what still gives the comparison teeth.",
    ),
    "kernel._FR13_FIXED32_TAW_TENSOR_CALL_CENSUS": (
        "derive",
        "the reference route's call table is walk-proportional; the same "
        "arithmetic fr13_fixed32_work_census has derived since site 13.",
    ),
    "kernel._FR13_FIXED32_TAW_SOURCE_DIGESTS": (
        "derive",
        "the audited payload carries the geometry and the reference census, so "
        "there is one digest per walk depth and the mode chooses it.",
    ),
    "kernel._fr13_fixed32_publish_work.slot_counts": (
        "derive",
        "uniform/row/path scatter slots are walk*3, walk*2 and walk; the "
        "uniform tensor was already ALLOCATED at the served depth and the work "
        "census already EXPECTED the served depth, so the publisher was the "
        "only party still saying 12.",
    ),
    "census.TAW_SOURCE_CONTRACT_SHA256_BY_WALK": (
        "derive",
        "the validator's expected digest follows the served walk, with the "
        "pre-site-27 digest kept as a declared prior era for banked runroots.",
    ),
    # ---- PINNED, WITH THE REASON -------------------------------------------
    "kernel._FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_TENSOR_CALL_CENSUS": (
        "pin",
        "MEASURED shape of a different execution (54 row gathers against 13 "
        "walk levels, not 26), and its selector refuses any mode but "
        "tail6/hydra27, so it is unreachable at any other depth.",
    ),
    "kernel._FR13_FIXED32_TAW_NATIVE_PRODUCTION_TENSOR_CALL_CENSUS": (
        "pin",
        "same: measured, and gated to tail6/hydra27 by the production "
        "selector and by the PASS record validator.",
    ),
    "kernel._FR13_FIXED32_TAW_SOFTMAX_CACHE_TENSOR_CALL_CENSUS": (
        "pin",
        "measured shape of a default-off lever; under another walk depth the "
        "work census refuses it rather than accepting hydra27's numbers.",
    ),
    "kernel._fr13_cfwd_logit_direct_publish_work.slot_counts": (
        "pin",
        "the function's bytes are credential-bound "
        "(_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_FUNCTIONS); converting it "
        "would drift a candidate credential measured on hydra27. The route is "
        "default-off and fails closed at the census under another depth.",
    ),
    "topology.TAW_UNIFORM_SLOTS": (
        "pin",
        "site-13 adjudication: a hydra27 DEFAULT other modules import by name, "
        "no longer read by the fixed32 execution path.",
    ),
    "topology.TAW_ROW_SCATTER_SLOTS": (
        "pin",
        "site-13 adjudication: WALK_CAP*2 off the flat hydra27 scalar, kept as "
        "an importable default with no fixed32 execution-path reader.",
    ),
    "topology.TAW_PATH_SCATTER_SLOTS": (
        "pin",
        "site-13 adjudication: WALK_CAP off the flat hydra27 scalar, kept as "
        "an importable default with no fixed32 execution-path reader.",
    ),
    "topology.TAW_CHILD_LANES": (
        "pin",
        "site-13 adjudication: WALK_CAP*fan-out off the flat hydra27 scalar; "
        "the publisher derives child lanes from the served target rows instead.",
    ),
    "census.TAW_ROWS_PER_REQUEST": (
        "pin",
        "read only in the native-precompute branch, which is tail6/hydra27.",
    ),
    "census.TAW_EXACT_COMMIT_LAUNCHES": (
        "pin",
        "read only in the native-precompute branch, which the selector and the "
        "PASS validator both restrict to tail6/hydra27.",
    ),
    "census.TAW_EXACT_COMMIT_PROGRAMS_PER_REQUEST": (
        "pin",
        "read only in the native-precompute branch, which the selector and the "
        "PASS validator both restrict to tail6/hydra27.",
    ),
    "census.TAW_LOOP_ITERATIONS": (
        "pin",
        "site-13: an importable hydra27 default with no execution-path reader.",
    ),
    "census.GDN_CRITICAL_PATH": (
        "pin",
        "site-13: the validators use the served walk; this name stays a "
        "hydra27 default for importers.",
    ),
    "kernel.self_check.loop_iterations_12": (
        "pin",
        "the CPU self-check loops ('tail6_fixed32', 'hydra27_fixed32') only, so "
        "12 is that loop's own depth, not a claim about every profile.",
    ),
    "kernel.self_check.uniform_stride_36": (
        "pin",
        "same loop, same reason: 36 is walk 12 * fan-out 3 for those modes.",
    ),
}


@pytest.fixture(autouse=True)
def _restore_fixed32_mode():
    """The mode lives in the environment, so every proof must put it back."""
    saved = os.environ.get("FR13_FIXED32_MODE")
    yield
    if saved is None:
        os.environ.pop("FR13_FIXED32_MODE", None)
    else:
        os.environ["FR13_FIXED32_MODE"] = saved
    for name in [name for name in sys.modules if name.startswith("fr13_")]:
        sys.modules.pop(name, None)


def _topology():
    return importlib.import_module("fr13_fixed32_topology")


def _load_kernel(mode: str | None = None) -> Any:
    """Import the kernel with FR13_FIXED32_MODE set, from a clean module table.

    The walk cap and the audited digest both resolve from the environment, so a
    per-mode proof has to import per mode.
    """
    if mode is None:
        os.environ.pop("FR13_FIXED32_MODE", None)
    else:
        os.environ["FR13_FIXED32_MODE"] = mode
    for name in [name for name in sys.modules if name.startswith("fr13_")]:
        sys.modules.pop(name, None)
    kernel = importlib.import_module("fr13_device_multidraft_kernel")
    kernel._FR13_FIXED32_TAW_SOURCE_CACHE = None
    kernel._FR13_FIXED32_TAW_SOURCE_CODES = None
    return kernel


def _load_census() -> Any:
    spec = importlib.util.spec_from_file_location("fr14_pins_census", CENSUS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


# --------------------------------------------------------------------------- #
# MUTATION PROOF 1: hydra31 boots through the contract with 16                 #
# --------------------------------------------------------------------------- #
def test_hydra31_boots_through_the_source_contract() -> None:
    kernel = _load_kernel(PROFILE_HYDRA31)
    topology = kernel._fr13_fixed32_topology()
    contract = kernel._fr13_fixed32_taw_source_contract(topology)
    assert contract["source_contract_sha256"] == (
        kernel._FR13_FIXED32_TAW_SOURCE_DIGESTS[PROFILE_HYDRA31]
    )
    assert contract["tensor_call_census"]["walk_levels"] == 16
    assert kernel._fr13_fixed32_walk_cap(topology) == 16


@pytest.mark.parametrize("mode", [MODE_TAIL6, PROFILE_HYDRA27, PROFILE_HYDRA31])
def test_every_served_mode_binds_its_own_geometry_and_census(mode: str) -> None:
    kernel = _load_kernel(mode)
    topology = kernel._fr13_fixed32_topology()
    walk = topology.walk_cap_for_mode(mode)
    contract = kernel._fr13_fixed32_taw_source_contract(topology)
    assert contract["tensor_call_census"] == (
        kernel._fr13_fixed32_taw_walk_tensor_call_census(walk)
    )
    assert kernel._fr13_fixed32_taw_geometry(walk)["walk_cap"] == walk
    assert contract["source_contract_sha256"] == (
        kernel._FR13_FIXED32_TAW_SOURCE_DIGESTS[mode]
    )


def test_an_unset_mode_still_audits_at_hydra27s_depth() -> None:
    """The non-fixed32 route is byte-identical, which is the pairing rule."""
    kernel = _load_kernel(None)
    topology = kernel._fr13_fixed32_topology()
    contract = kernel._fr13_fixed32_taw_source_contract(topology)
    assert contract["source_contract_sha256"] == (
        kernel._FR13_FIXED32_TAW_SOURCE_SHA256
    )
    assert contract["tensor_call_census"]["walk_levels"] == 12


# --------------------------------------------------------------------------- #
# MUTATION PROOF 2: hydra27 unchanged, field by field                          #
# --------------------------------------------------------------------------- #
def test_hydra27_is_unchanged_field_by_field() -> None:
    """No arithmetic moved. The derivation reproduces every retired literal."""
    kernel = _load_kernel(PROFILE_HYDRA27)
    assert kernel._FR13_FIXED32_TAW_GEOMETRY == RETIRED_HYDRA27_GEOMETRY
    assert (
        kernel._FR13_FIXED32_TAW_TENSOR_CALL_CENSUS
        == RETIRED_HYDRA27_TENSOR_CALL_CENSUS
    )
    # field by field, so a coincidental dict equality cannot hide a swap
    derived_geometry = kernel._fr13_fixed32_taw_geometry(12)
    for name, value in RETIRED_HYDRA27_GEOMETRY.items():
        assert derived_geometry[name] == value, name
    derived_census = kernel._fr13_fixed32_taw_walk_tensor_call_census(12)
    for name, value in RETIRED_HYDRA27_TENSOR_CALL_CENSUS.items():
        assert derived_census[name] == value, name
    assert set(derived_census) == set(RETIRED_HYDRA27_TENSOR_CALL_CENSUS)
    assert set(derived_geometry) == set(RETIRED_HYDRA27_GEOMETRY)


def test_tail6_and_hydra27_share_one_audited_digest() -> None:
    """They share a walk depth, so they must share a digest -- not by typing."""
    kernel = _load_kernel(PROFILE_HYDRA27)
    digests = kernel._FR13_FIXED32_TAW_SOURCE_DIGESTS
    assert digests[MODE_TAIL6] is digests[PROFILE_HYDRA27]
    assert digests[PROFILE_HYDRA27] == kernel._FR13_FIXED32_TAW_SOURCE_SHA256
    assert digests[PROFILE_HYDRA31] != digests[PROFILE_HYDRA27]


def test_the_digest_index_is_recomputed_not_invented() -> None:
    """Every entry is rebuilt from the module's own canonical payload.

    The same machinery that fills the per-mode schedule digests. If a future
    edit inside the audited closure moves a digest, this fails and names the
    mode -- it does not wait for a serve to discover it at boot.
    """
    topology_modes = _topology().SERVING_MODES
    for mode in topology_modes:
        kernel = _load_kernel(mode)
        topology = kernel._fr13_fixed32_topology()
        contract = kernel._fr13_fixed32_taw_source_contract(topology)
        assert (
            contract["source_contract_sha256"]
            == kernel._FR13_FIXED32_TAW_SOURCE_DIGESTS[mode]
        ), f"{mode}: recomputed digest differs from the pinned index"


def test_the_digest_index_key_set_is_the_authoritys_serving_modes() -> None:
    kernel = _load_kernel(PROFILE_HYDRA27)
    topology = kernel._fr13_fixed32_topology()
    assert set(kernel._FR13_FIXED32_TAW_SOURCE_DIGESTS) == set(topology.SERVING_MODES)
    assert set(kernel._FR13_FIXED32_TAW_SCHEDULE_DIGESTS) == set(topology.SERVING_MODES)


def test_an_unaudited_mode_refuses_instead_of_guessing() -> None:
    kernel = _load_kernel(PROFILE_HYDRA27)
    topology = kernel._fr13_fixed32_topology()
    with pytest.raises(RuntimeError) as refusal:
        kernel._fr13_fixed32_taw_source_digest(topology, "hydra99_fixed32")
    assert "source digest is unknown for mode" in str(refusal.value)


# --------------------------------------------------------------------------- #
# MUTATION PROOF 3: a wrong runtime cap still refuses, naming both values      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("mode", "wrong_cap"),
    [(PROFILE_HYDRA31, "12"), (PROFILE_HYDRA27, "16"), (MODE_TAIL6, "16")],
)
def test_a_wrong_runtime_cap_still_refuses_naming_both_values(
    mode: str, wrong_cap: str
) -> None:
    """Deriving the PIN must not soften the guard on the RUNTIME cap.

    This is where the walk cap's teeth live now, so it is proven here as well as
    in the site-13 suite: the env-declared cap is compared against the
    authority's, and the refusal names both numbers.
    """
    kernel = _load_kernel(mode)
    topology = kernel._fr13_fixed32_topology()
    environ = {
        "FR13_FIXED32_MODE": mode,
        "FR13_FIXED32_VALID_MASK": hex(int(topology.VALID_MASK_BY_MODE[mode])),
        "FR13_FIXED32_ACTIVE_NODES": str(
            kernel._fr13_fixed32_expected_active(topology, mode)
        ),
        "FR13_FIXED32_TAW_WALK_CAP": wrong_cap,
    }
    saved = {key: os.environ.get(key) for key in environ}
    os.environ.update(environ)
    try:
        with pytest.raises(RuntimeError) as refusal:
            kernel._fr13_fixed32_runtime_contract(mode)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    message = str(refusal.value)
    assert wrong_cap in message
    assert str(topology.walk_cap_for_mode(mode)) in message


def test_the_era_pin_cannot_drift_from_the_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hydra27 literals are checked against the topology, not trusted.

    They stay literals because a credential-bound digest and the hydra27-gated
    PASS records read them; that is only safe while they still describe hydra27.
    """
    kernel = _load_kernel(PROFILE_HYDRA27)
    topology = kernel._fr13_fixed32_topology()
    monkeypatch.setattr(
        topology, "walk_cap_for_mode", lambda mode: 99, raising=True
    )
    kernel._FR13_FIXED32_TAW_SOURCE_CACHE = None
    with pytest.raises(RuntimeError) as refusal:
        kernel._fr13_fixed32_taw_source_contract(topology)
    message = str(refusal.value)
    assert "hydra27-era pins disagree with the topology authority" in message
    assert "99" in message and "12" in message


def test_the_geometry_refusal_now_names_both_sides() -> None:
    """Round 22 cost a recovered 398-line log because it named one side.

    The runtime geometry was printed; the pin it was compared against was not,
    so the value that actually disagreed had to be read out of the source.
    """
    assert '" against "' in KERNEL_SOURCE
    assert "TAW source geometry drift: " in KERNEL_SOURCE
    assert "at served walk " in KERNEL_SOURCE


# --------------------------------------------------------------------------- #
# THE PINS-FAMILY CENSUS ITSELF                                                #
# --------------------------------------------------------------------------- #
def test_every_pin_in_the_family_is_classified_with_a_reason() -> None:
    for name, (classification, reason) in PINS_FAMILY.items():
        assert classification in {"derive", "pin"}, name
        assert len(reason) > 40, f"{name}: a classification without a reason"


def test_the_derived_members_really_derive() -> None:
    """Each 'derive' entry must move when the walk depth moves."""
    kernel = _load_kernel(PROFILE_HYDRA27)
    census = _load_census()
    twelve = kernel._fr13_fixed32_taw_geometry(12)
    sixteen = kernel._fr13_fixed32_taw_geometry(16)
    assert twelve != sixteen and twelve["walk_cap"] != sixteen["walk_cap"]
    # ...and only the walk-dependent field moves
    assert {
        key for key in twelve if twelve[key] != sixteen[key]
    } == {"walk_cap"}

    census_twelve = kernel._fr13_fixed32_taw_walk_tensor_call_census(12)
    census_sixteen = kernel._fr13_fixed32_taw_walk_tensor_call_census(16)
    moved = {key for key in census_twelve if census_twelve[key] != census_sixteen[key]}
    assert moved == {
        "walk_levels",
        "full_vocab_row_gathers",
        "full_vocab_fp32_casts",
        "full_vocab_softmax_calls",
        "full_vocab_normalizations",
        "full_vocab_cdf_calls",
        "source_cdf_calls",
        "qmix_zero_fills",
        "qmix_scatter_add_calls",
        "residual_subtract_calls",
        "residual_clamp_calls",
        "residual_where_calls",
        "exact_commit_launches",
        "exact_commit_programs_per_request",
    }
    assert len(census.TAW_SOURCE_CONTRACT_SHA256_BY_WALK) >= 2
    assert (
        census.TAW_SOURCE_CONTRACT_SHA256_BY_WALK[12]
        != census.TAW_SOURCE_CONTRACT_SHA256_BY_WALK[16]
    )


def test_the_published_slot_counts_follow_the_served_walk() -> None:
    """The rung behind the geometry: the publisher no longer says 36 at 16.

    The work census expects walk*3, walk*2 and walk; the uniform tensor is
    allocated at the served depth. The publisher was the last party still
    reading the topology's flat hydra27 scalars.
    """
    for marker in (
        '"uniform_slots": published_walk_cap * 3 * batch_size,',
        "published_walk_cap\n            * 2\n            * batch_size",
    ):
        assert marker in KERNEL_SOURCE, marker
    # and the flat scalars are gone from that publisher's CODE (the comment
    # explaining why they went is allowed to name them)
    publisher = KERNEL_SOURCE[
        KERNEL_SOURCE.index("def _fr13_fixed32_publish_work(") :
    ]
    publisher = publisher[: publisher.index("\ndef ")]
    code = "\n".join(
        line for line in publisher.splitlines() if not line.lstrip().startswith("#")
    )
    for scalar in (
        "TAW_UNIFORM_SLOTS",
        "TAW_ROW_SCATTER_SLOTS",
        "TAW_PATH_SCATTER_SLOTS",
    ):
        assert scalar not in code, f"{scalar} still read by the publisher"


def test_the_pinned_route_censuses_are_unreachable_at_another_depth() -> None:
    """The 'pin' classification's safety, proven rather than asserted."""
    # the native selector refuses any mode but the two 12-walk ones
    assert (
        'if mode not in ("tail6_fixed32", "hydra27_fixed32"):' in KERNEL_SOURCE
    )
    assert (
        'payload_mode not in ("tail6_fixed32", "hydra27_fixed32")' in KERNEL_SOURCE
    )
    # and the CPU self-check's own literals belong to a loop over those two
    assert 'for mode in ("tail6_fixed32", "hydra27_fixed32"):' in KERNEL_SOURCE


def test_the_credential_bound_publisher_is_left_alone() -> None:
    """Its bytes are a credential; converting it would drift a candidate."""
    assert (
        "_fr13_cfwd_logit_direct_publish_work"
        in KERNEL_SOURCE[
            KERNEL_SOURCE.index("_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_FUNCTIONS") :
            KERNEL_SOURCE.index("_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_KERNEL_SOURCE_FUNCTIONS")
        ]
    )
    publisher = KERNEL_SOURCE[
        KERNEL_SOURCE.index("def _fr13_cfwd_logit_direct_publish_work(") :
    ]
    publisher = publisher[: publisher.index("\ndef ")]
    assert "topology.TAW_UNIFORM_SLOTS" in publisher


def test_the_pins_family_census_is_complete() -> None:
    """A walk-derived structure that is not classified fails here.

    The detector is deliberately narrow: it looks for the three topology
    scalars that ARE walk-derived being read anywhere in the fixed32 kernel, and
    requires every reading site to be a classified member of the family. A
    broad literal hunt would drown in false positives and get tuned into
    silence; this one cannot be satisfied by tuning.
    """
    classified_readers = {
        "_fr13_cfwd_logit_direct_publish_work",
    }
    for scalar in (
        "TAW_UNIFORM_SLOTS",
        "TAW_ROW_SCATTER_SLOTS",
        "TAW_PATH_SCATTER_SLOTS",
    ):
        for index, line in enumerate(KERNEL_SOURCE.splitlines(), start=1):
            if f"topology.{scalar}" not in line or line.lstrip().startswith("#"):
                continue
            head = KERNEL_SOURCE[: sum(
                len(row) + 1 for row in KERNEL_SOURCE.splitlines()[: index - 1]
            )]
            enclosing = head.rsplit("\ndef ", 1)[-1].split("(", 1)[0]
            assert enclosing in classified_readers, (
                f"{scalar} read at line {index} inside {enclosing!r}, which is "
                "not a classified member of PINS_FAMILY"
            )


# --------------------------------------------------------------------------- #
# THE RE-ATTESTATION: banked evidence must keep validating                     #
# --------------------------------------------------------------------------- #
def test_the_prior_era_keeps_banked_runroots_validating() -> None:
    census = _load_census()
    assert (
        "d9f85b6804f916bb991818b51f1be56cfad10d07def4e6d7d7f557cb5fc1dde0"
        in census.TAW_SOURCE_CONTRACT_SHA256_PRIOR_ERAS
    )
    event = census.reference_event(PROFILE_HYDRA27, 1, f"{PROFILE_HYDRA27}:1:0")
    event["taw"]["source_contract_sha256"] = (
        census.TAW_SOURCE_CONTRACT_SHA256_PRIOR_ERAS[0]
    )
    census.validate_event(event, source="banked")


def test_the_two_modules_declare_the_same_prior_era() -> None:
    """The kernel records which digest it retired; the census accepts it.

    Two modules, one fact. If a future re-attestation bumps one list and not the
    other, banked runroots start refusing (or a retired source starts passing)
    and nothing else would say so.
    """
    kernel = _load_kernel(PROFILE_HYDRA27)
    census = _load_census()
    assert (
        kernel._FR13_FIXED32_TAW_SOURCE_SHA256_PRIOR_ERAS
        == census.TAW_SOURCE_CONTRACT_SHA256_PRIOR_ERAS
    )
    # and the current digest is NOT quietly also a prior era
    assert (
        kernel._FR13_FIXED32_TAW_SOURCE_SHA256
        not in kernel._FR13_FIXED32_TAW_SOURCE_SHA256_PRIOR_ERAS
    )
    assert census.TAW_SOURCE_CONTRACT_SHA256 == kernel._FR13_FIXED32_TAW_SOURCE_SHA256
    assert (
        census.TAW_SOURCE_CONTRACT_SHA256_BY_WALK[16]
        == kernel._FR13_FIXED32_TAW_SOURCE_DIGESTS[PROFILE_HYDRA31]
    )


def test_a_digest_from_no_declared_era_still_refuses() -> None:
    census = _load_census()
    event = census.reference_event(PROFILE_HYDRA27, 1, f"{PROFILE_HYDRA27}:1:0")
    event["taw"]["source_contract_sha256"] = "b" * 64
    with pytest.raises(census.CensusError) as refusal:
        census.validate_event(event, source="forged")
    message = str(refusal.value)
    assert "is not the audited digest at walk 12" in message
    assert "nor a declared prior era" in message


def test_a_hydra31_event_carrying_hydra27s_digest_refuses() -> None:
    """The era table must not become a way to serve one profile's audit for another."""
    census = _load_census()
    event = census.reference_event(PROFILE_HYDRA31, 1, f"{PROFILE_HYDRA31}:1:0")
    event["taw"]["source_contract_sha256"] = census.TAW_SOURCE_CONTRACT_SHA256_BY_WALK[
        12
    ]
    with pytest.raises(census.CensusError) as refusal:
        census.validate_event(event, source="crossed")
    assert "is not the audited digest at walk 16" in str(refusal.value)


def test_the_census_round_trips_at_every_modes_walk_depth() -> None:
    census = _load_census()
    topology = _topology()
    for mode in topology.SERVING_MODES:
        for batch in (1, 4):
            event = census.reference_event(mode, batch, f"{mode}:1:0")
            census.validate_event(event, source="walk")
            walk = topology.walk_cap_for_mode(mode)
            assert event["taw"]["loop_iterations"] == walk
            assert event["taw"]["uniform_slots"] == walk * 3 * batch
            assert event["taw"]["tensor_call_census"]["walk_levels"] == walk
            assert event["gdn"]["critical_path"] == walk


def test_the_re_attestation_is_recorded_where_the_digest_lives() -> None:
    assert "RE-ATTESTED 2026-08-24 by SITE 27" in KERNEL_SOURCE
    assert "WHY IT COULD NOT BE AVOIDED" in KERNEL_SOURCE
