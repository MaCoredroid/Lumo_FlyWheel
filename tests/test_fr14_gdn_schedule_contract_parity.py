"""The walk-derived-pin class, SIXTH member, and the census that spans the class.

ROUND 22, FOURTH BOOT died in a file the census was never pointed at:

    RuntimeError: FR13 fixed32 GDN schedule work drift:
      {'path_counts': (1, 11), 'max_lengths': (5, 11), 'launches': 2,
       'programs': 12, 'padded_slots': 126, 'critical': 16,
       'export_or_mask': 16915}

against a pin of (5, 7) / 82 / 12 planted by
scripts/fr10_phase4_patch_vllm_tree_gdn.py.

TWO REASONS THE EARLIER CENSUS COULD NOT SEE IT, and both are about REACH:

  1. IT WAS IN ANOTHER FILE. The census ran over fr13_device_multidraft_kernel
     because that is where the first five members lived. The class's extent is
     the SERVE EXECUTION CLOSURE -- 207 files -- because patchers plant
     contracts into the container and gates assert them.

  2. IT WAS INSIDE A STRING. The pin lives in the 316KB raw string
     _FR13_FIXED32_OBSERVED_RUNTIME_SOURCE (patcher lines 660-8743), planted
     verbatim into the container's gdn_linear_attn.py. An AST scan of the
     patcher cannot see it AT ALL: `ast.parse` reports zero assignments named
     `expected_contract` in a file where grep finds one. A census that walks
     only real code is blind to every planted contract in the repo.

So the scan below parses each closure file AND re-parses any large string
constant that is itself Python, and it classifies what it finds rather than
filtering it down to what it can already explain.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

PATCHER = SCRIPTS / "fr10_phase4_patch_vllm_tree_gdn.py"
GDN_KERNEL = REPO / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
PATCHER_SOURCE = PATCHER.read_text(encoding="utf-8")
GDN_KERNEL_SOURCE = GDN_KERNEL.read_text(encoding="utf-8")

CONTRACT_FIELDS = (
    "path_counts",
    "max_lengths",
    "launches",
    "programs",
    "padded_slots",
    "critical",
    "export_or_mask",
)


def _topology():
    import importlib

    return importlib.import_module("fr13_fixed32_topology")


def _planted_blob() -> str:
    """The raw string the patcher plants into gdn_linear_attn.py."""
    tree = ast.parse(PATCHER_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
            for target in node.targets
        ):
            return node.value.value
    raise AssertionError("the planted runtime blob is gone")


def _blob_schedule_table(mode: str) -> dict:
    """Execute the blob's own table under a served mode, as the engine would."""
    blob = _planted_blob()
    tree = ast.parse(blob)
    lines = blob.split("\n")
    kept = []
    for node in tree.body:
        names = [
            target.id
            for target in getattr(node, "targets", [])
            if isinstance(target, ast.Name)
        ]
        touches = any(name.startswith("_FR13_FIXED32_GDN_SCHEDULE") for name in names)
        touches = touches or "_FR13_FIXED32_GDN_TREE_PROFILE_BY_MODE" in names
        touches = touches or "_FR13_FIXED32_GDN_MODE" in names
        touches = touches or (
            isinstance(node, ast.If) and "_FR13_FIXED32_GDN_MODE" in ast.dump(node)
        )
        if touches:
            kept.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    assert kept, "the blob carries no GDN schedule table"
    saved = os.environ.get("FR13_FIXED32_MODE")
    os.environ["FR13_FIXED32_MODE"] = mode
    namespace: dict = {}
    try:
        exec("\n".join(kept), namespace)  # noqa: S102 - the blob is our own source
    finally:
        if saved is None:
            os.environ.pop("FR13_FIXED32_MODE", None)
        else:
            os.environ["FR13_FIXED32_MODE"] = saved
    return namespace


def _serving_table() -> dict:
    """fr10_gdn_tree_kernel's profile-keyed schedule, lifted without importing."""
    tree = ast.parse(GDN_KERNEL_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "_FR13_FIXED32_SCHEDULE_BY_PROFILE"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("the serving package's schedule table is gone")


# --------------------------------------------------------------------------- #
# THE PARITY LINT: a mirror is only safe while something compares it           #
# --------------------------------------------------------------------------- #
def test_the_planted_table_mirrors_the_serving_package_exactly() -> None:
    """The blob CANNOT import, so it carries a copy. This is what keeps it honest.

    fr10_gdn_tree_kernel._FR13_FIXED32_SCHEDULE_BY_PROFILE has been
    profile-keyed since round 18 and carries these seven fields; the patcher's
    copy was never keyed, which is the whole defect.
    """
    serving = _serving_table()
    for profile in ("hydra27_fixed32", "hydra31_fixed32"):
        blob = _blob_schedule_table(profile)["_FR13_FIXED32_GDN_SCHEDULE_BY_PROFILE"]
        assert set(blob) == {"hydra27_fixed32", "hydra31_fixed32"}
        for name in CONTRACT_FIELDS:
            assert blob[profile][name] == serving[profile][name], (
                f"{profile}.{name}: planted {blob[profile][name]!r} against "
                f"serving {serving[profile][name]!r}"
            )
        assert set(blob[profile]) == set(CONTRACT_FIELDS)


def test_the_table_agrees_with_the_topology_authority() -> None:
    """Both copies must still describe the tree the authority declares.

    MEASURED, not assumed: padded_slots is sum(paths * length) over the levels,
    which is 1*5 + 11*7 = 82 at hydra27 and 1*5 + 11*11 = 126 at hydra31, and
    `critical` is the walk cap. Nothing here is a formula someone remembered.
    """
    topology = _topology()
    serving = _serving_table()
    for mode in topology.SERVING_MODES:
        profile_name = topology.TREE_PROFILE_BY_MODE[mode]
        profile = topology.PROFILES[profile_name]
        entry = serving[profile_name]
        assert entry["path_counts"] == tuple(profile["gdn_level_path_counts"])
        assert entry["max_lengths"] == tuple(profile["gdn_level_max_lengths"])
        assert entry["launches"] == int(profile["gdn_launches"])
        assert entry["programs"] == int(profile["gdn_path_programs"])
        assert entry["padded_slots"] == int(profile["gdn_padded_slots"])
        assert entry["critical"] == int(topology.walk_cap_for_mode(mode))
        # the derivation of padded_slots, checked rather than trusted
        assert entry["padded_slots"] == sum(
            paths * length
            for paths, length in zip(
                entry["path_counts"], entry["max_lengths"], strict=True
            )
        )


@pytest.mark.parametrize(
    ("mode", "padded", "critical", "max_lengths"),
    [
        ("hydra27_fixed32", 82, 12, (5, 7)),
        ("tail6_fixed32", 82, 12, (5, 7)),
        ("", 82, 12, (5, 7)),
        ("hydra31_fixed32", 126, 16, (5, 11)),
    ],
)
def test_the_planted_expectation_follows_the_served_mode(
    mode: str, padded: int, critical: int, max_lengths: tuple
) -> None:
    """MUTATION PROOF. hydra31 expects what the corpse observed; the rest are
    byte-identical to the retired pin, unset mode included."""
    expected = _blob_schedule_table(mode)["_FR13_FIXED32_GDN_SCHEDULE_EXPECTED"]
    assert expected["padded_slots"] == padded
    assert expected["critical"] == critical
    assert expected["max_lengths"] == max_lengths
    assert expected["launches"] == 2
    assert expected["programs"] == 12
    assert expected["path_counts"] == (1, 11)
    assert expected["export_or_mask"] == 16915


def test_the_corpse_values_are_exactly_what_hydra31_now_expects() -> None:
    """The observed dict from round 22's fourth boot, verbatim."""
    observed = {
        "path_counts": (1, 11),
        "max_lengths": (5, 11),
        "launches": 2,
        "programs": 12,
        "padded_slots": 126,
        "critical": 16,
        "export_or_mask": 16915,
    }
    expected = _blob_schedule_table("hydra31_fixed32")[
        "_FR13_FIXED32_GDN_SCHEDULE_EXPECTED"
    ]
    assert dict(expected) == observed


def test_an_unknown_mode_refuses_instead_of_defaulting() -> None:
    with pytest.raises(RuntimeError) as refusal:
        _blob_schedule_table("nonsense_mode")
    assert "GDN schedule has no profile for mode" in str(refusal.value)


def test_the_gdn_refusal_is_two_sided() -> None:
    """It printed observed only, which cost a whole boot to diagnose."""
    blob = _planted_blob()
    assert "GDN schedule work drift for mode " in blob
    assert '": observed "' in blob
    assert '" against audited "' in blob
    assert "for _name in _drifted" in blob
    # and the retired one-sided form is gone
    assert '"FR13 fixed32 GDN schedule work drift: "' not in blob


def test_the_logical_schedule_stamps_derive_from_the_authority() -> None:
    """The statelessness audit's two sites: informational, and still evidence.

    A stamp that states another profile's schedule is a provenance lie whether
    or not a guard reads it today.
    """
    for literal in (
        '"logical_padded_slots": 82,',
        '"logical_padded_slots": 82 * batch,',
        '"logical_critical_path": 12,',
    ):
        assert literal not in GDN_KERNEL_SOURCE, literal
    assert (
        '"logical_launches": _FR13_FIXED32_SCHEDULE_EXPECTED["launches"],'
        in GDN_KERNEL_SOURCE
    )
    assert GDN_KERNEL_SOURCE.count(
        '_FR13_FIXED32_SCHEDULE_EXPECTED["padded_slots"]'
    ) >= 2
    assert GDN_KERNEL_SOURCE.count(
        '_FR13_FIXED32_SCHEDULE_EXPECTED["critical"]'
    ) >= 2


# --------------------------------------------------------------------------- #
# THE CENSUS, over the serve execution closure, blobs included                 #
# --------------------------------------------------------------------------- #
#: (file, scope) -> classification + reason. Every Tier-1 hit must be here.
#: "derive"    the number follows the served profile
#: "pin"       genuinely invariant, route-gated, credential-bound, or a fixture
#: "authority" the table that DEFINES the per-profile values
#: "coincide"  the values are not walk quantities at all; recorded so the next
#:             reader does not re-investigate them
CLOSURE_CENSUS: dict[str, dict[str, str]] = {
    "scripts/fr10_phase4_patch_vllm_tree_gdn.py": {
        "<module>": "coincide: the patch-time tree tables are NODE IDS",
        "BLOB:<module>": "derive: the planted GDN schedule table (this landing)",
        "BLOB:_fr13_fixed32_drafter_observed_publish": (
            "pin: drafter publish shapes are physical (31/32), the hits are ids"
        ),
        "BLOB:_fr13_fixed32_validate_forward_work": (
            "pin: forward-work shapes gated to the hydra27 capture route"
        ),
        "_fr13_fixed32_observed_runtime_self_test": (
            "pin: the self-test builds hydra27 fixtures for its own namespace"
        ),
    },
    "scripts/fr13_cfwd_logit_direct_decision_kernel.py": {
        "<module>": "pin: credential-bound CFWD candidate, hydra27-qualified",
    },
    "scripts/fr13_cfwd_logit_direct_packed_runtime_overlay.py": {
        "fr13_fixed32_cfwd_logit_direct_warm_execute": (
            "pin: 13/17 are the CFWD candidate's qualified scope rows"
        ),
    },
    "scripts/fr13_cfwd_packed_walk_node_trust_kernel.py": {
        "packed_walk_node_trust_contract": (
            "pin: default-off packed-walk lever, byte-qualified on hydra27"
        ),
    },
    "scripts/fr13_device_multidraft_kernel.py": {
        "_fr13_fixed32_topology": (
            "pin: site-13's deliberate hydra27-identity assertions on the "
            "authority itself"
        ),
        "_fr13_fixed32_test_accept_leaf_depth_pad": (
            "pin: self-test fixture built for the two 12-walk modes its own loop\n"
            "iterates; the values are pad depths, not a served contract"
        ),
        "_fr13_fixed32_test_mode_switch_batches": (
            "pin: self-test loops tail6/hydra27 only"
        ),
    },
    "scripts/fr13_device_multidraft_offline_gate.py": {
        "stress_cases": "pin: offline stress fixtures, not a served contract",
    },
    "scripts/fr13_fixed32_topology.py": {
        "<module>": "authority: the per-profile tables every derivation reads",
        "validate_contract": "authority: hydra27's own self-validation",
        "validate_tail10_contract": "authority: hydra31's own self-validation",
    },
    "scripts/fr13_fixed32_work_census.py": {
        "<module>": (
            "pin: site-27 classified these as hydra27 defaults with no "
            "execution-path reader; the validators derive from the served walk"
        ),
    },
    "scripts/fr13_floor_gate.py": {
        "<module>": "coincide: window/threshold constants, not walk quantities",
    },
    "scripts/fr13_patch_fa2_tree_bias.py": {
        "<module>": "coincide: the FA2 bias table is NODE IDS",
    },
    "scripts/fr13_sfwd_state_fusion_pass.py": {
        "validate_engagement": "pin: SFWD lever, byte-qualified on hydra27",
    },
    "src/lumo_flywheel_serving/auto_research.py": {
        "_t_critical_95_two_sided": (
            "coincide: a Student-t table; 7/12/15 are degrees of freedom"
        ),
    },
    "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py": {
        "<module>": "authority: _FR13_FIXED32_SCHEDULE_BY_PROFILE, round 18",
        "fixed32_batch_gdn_launch_contract": (
            "derive: the logical schedule stamps, fixed in this landing"
        ),
        "preseed_fixed32_committer_graph": (
            "derive: the logical schedule stamps, fixed in this landing"
        ),
    },
    "src/lumo_flywheel_serving/fr13_gdn_gqa_group3.py": {
        "launch_fixed32_gdn_gqa_group3_source_candidate": (
            "pin: GQA-group3 candidate, default-off and hydra27-qualified"
        ),
    },
    "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py": {
        "fixed32_sfwd_conv_postprep_fusion_contract": (
            "pin: SFWD conv/post-prep lever, byte-qualified on hydra27"
        ),
    },
    "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py": {
        "<module>": "coincide: descriptor tables are NODE IDS",
    },
    "src/lumo_flywheel_serving/fr13_sfwd_state_fusion_production.py": {
        "fixed32_sfwd_state_fusion_production_engagement": (
            "pin: SFWD state-fusion production candidate, hydra27-qualified"
        ),
    },
    "tests/test_fr13_fixed32_gdn_gqa_group3.py": {
        "test_compile_time_physical32_schedule_matches_validated_descriptors": (
            "pin: the test states hydra27's descriptors on purpose"
        ),
    },
    "tests/test_fr14_contract_profile_signatures.py": {
        "<module>": "pin: the site-13 suite's own hydra27 fixtures",
        "_build_pregather_record": (
            "pin: a test fixture record, hydra27 by construction"
        ),
        "test_the_ladder_is_self_proving": (
            "pin: the ladder fixture states its own known sequence"
        ),
        "test_the_ladder_turns_a_known_sequence_into_a_known_ladder": (
            "pin: the ladder fixture states its own known sequence"
        ),
        "test_the_preseed_completes_end_to_end_for_every_serving_mode": (
            "pin: states 13/17 and 11/21 per mode, which is the point"
        ),
        "test_the_qualified_scope_schedule_is_not_either_modes_own": (
            "pin: states the union against both modes, which is the point"
        ),
    },
}


def _candidate_values() -> tuple[set[int], set[int]]:
    topology = _topology()
    varying: dict[str, list[int]] = {}
    for mode in topology.SERVING_MODES:
        profile = topology.PROFILES[topology.TREE_PROFILE_BY_MODE[mode]]
        walk = int(topology.walk_cap_for_mode(mode))
        counts = tuple(profile["gdn_level_path_counts"])
        lengths = tuple(profile["gdn_level_max_lengths"])
        for name, value in (
            ("walk", walk),
            ("walk*2", 2 * walk),
            ("walk*3", 3 * walk),
            ("walk+1", walk + 1),
            ("max_depth", int(profile["max_physical_depth"])),
            ("active", int(profile["active_drafts"])),
            ("gdn_padded", int(profile["gdn_padded_slots"])),
            ("gdn_len0", lengths[0]),
            ("gdn_len1", lengths[1]),
            ("gdn_count1", counts[1]),
            ("gdn_launches", int(profile["gdn_launches"])),
            ("gdn_programs", int(profile["gdn_path_programs"])),
        ):
            varying.setdefault(name, []).append(value)
    values: set[int] = set()
    for per_mode in varying.values():
        if len(set(per_mode)) > 1:
            values |= set(per_mode)
    invariant = {
        int(topology.PHYSICAL_DRAFTS),
        int(topology.PHYSICAL_ROWS),
        int(topology.SAMPLER_MAX_FANOUT),
        int(topology.COMMIT_PATH_CAP),
        int(topology.OUTPUT_PUBLISH_CAPACITY),
        int(topology.GDN_LAYERS),
        int(topology.MODEL_LAYERS),
        int(topology.TREE_ATTENTION_LAYERS),
    }
    return values - invariant, values & invariant


def _scan(text: str, values: set[int]) -> set[str]:
    """Scopes carrying >=2 DISTINCT mode-varying values in one container.

    Two coincidences inside one literal is the discriminator: a single 12 is a
    loop bound, a dict holding 82 and 12 is a hydra27 GDN schedule. Large
    string constants that parse as Python are re-scanned, because a planted
    contract is exactly as real as a compiled one.
    """
    scopes: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return scopes
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(line, node.name)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Dict, ast.Tuple, ast.List)):
            found = {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, int)
                and not isinstance(child.value, bool)
                and child.value in values
            }
            if len(found) >= 2:
                scopes.add(owner.get(node.lineno, "<module>"))
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) > 20_000
        ):
            scopes |= {"BLOB:" + name for name in _scan(node.value, values)}
    return scopes


def test_the_census_universe_is_the_serve_execution_closure() -> None:
    """The universe claim, stated so it can fail."""
    import fr14_mode_table_parity as parity

    closure = parity.serve_execution_closure()
    assert len(closure) >= 200, f"closure collapsed to {len(closure)} files"
    # it must reach BOTH roots the class has been found in
    assert any(rel.startswith("src/lumo_flywheel_serving/") for rel in closure)
    assert any(rel.startswith("scripts/") for rel in closure)
    for required in (
        "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
        "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py",
        "scripts/fr13_device_multidraft_kernel.py",
    ):
        assert required in closure, f"{required} is outside the census universe"


def test_a_planted_contract_is_invisible_to_a_plain_ast_scan() -> None:
    """The second reason the earlier census could not see this.

    Proof, not assertion: the patcher's AST contains no assignment named
    `expected_contract`, while the planted blob does.
    """
    tree = ast.parse(PATCHER_SOURCE)
    real = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "expected_contract"
            for target in node.targets
        )
    ]
    assert real == []
    assert "expected_contract" in _planted_blob()


def test_every_mode_varying_literal_in_the_closure_is_classified() -> None:
    """THE CENSUS. Reach = the closure; blobs included; nothing filtered away."""
    import fr14_mode_table_parity as parity

    unambiguous, _ambiguous = _candidate_values()
    unclassified: dict[str, list[str]] = {}
    for rel in parity.serve_execution_closure():
        if not rel.endswith(".py"):
            continue
        path = REPO / rel
        if not path.is_file():
            continue
        scopes = _scan(path.read_text(errors="replace"), unambiguous)
        if not scopes:
            continue
        known = CLOSURE_CENSUS.get(rel, {})
        missing = sorted(scope for scope in scopes if scope not in known)
        if missing:
            unclassified[rel] = missing
    assert not unclassified, (
        "unclassified mode-varying literals in the serve execution closure: "
        + repr(unclassified)
    )


def test_the_census_reaches_the_two_statelessness_audit_sites() -> None:
    """Named explicitly because they were reported from outside the census."""
    unambiguous, _ambiguous = _candidate_values()
    scopes = _scan(GDN_KERNEL_SOURCE, unambiguous)
    assert "fixed32_batch_gdn_launch_contract" in scopes or (
        "preseed_fixed32_committer_graph" in scopes
    ), "the logical-schedule stamps fell out of the census"
    assert "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py" in CLOSURE_CENSUS


def test_every_classification_carries_a_reason() -> None:
    for rel, scopes in CLOSURE_CENSUS.items():
        for scope, reason in scopes.items():
            kind = reason.split(":", 1)[0]
            assert kind in {"derive", "pin", "authority", "coincide"}, (rel, scope)
            assert len(reason) > 25, f"{rel}:{scope} has no reason"


def test_the_ambiguous_values_are_reported_not_asserted() -> None:
    """A detector that cannot tell two readings apart must say so."""
    unambiguous, ambiguous = _candidate_values()
    topology = _topology()
    assert int(topology.PHYSICAL_DRAFTS) in ambiguous
    assert int(topology.PHYSICAL_ROWS) in ambiguous
    assert not (unambiguous & ambiguous)
    # the GDN quantities this landing turns on must be in the asserted half
    for value in (7, 11, 12, 82, 126):
        assert value in unambiguous, value
