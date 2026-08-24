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


# --------------------------------------------------------------------------- #
# THE METHOD REVIEW, due after the sixth distinct refusal                      #
# --------------------------------------------------------------------------- #
# BOOT FIVE died one statement after boot four, on the SEVENTH member of the
# value class -- and my own census had dropped it. The rule was "a container
# holding TWO OR MORE distinct mode-varying values", chosen so that node-id
# tables would not drown the report. The guard that killed boot five holds
# exactly ONE:
#
#     {"schedule": "fixed32", "route_armed": True, "n_levels": 2,
#      "critical": 12, "parent_nodes": 32, "emask_rows": 32, "export_rows": 32}
#
# 12 is the walk cap; 32 is PHYSICAL_ROWS, which is profile-invariant and so is
# not a candidate at all; 2 and True are not candidates. One distinct value,
# below my threshold, silently dropped. THAT is the recall hole, and it is a
# more precise finding than "the census cannot see predicates": a contract
# carrying a single mode-varying number among invariants is the commonest
# shape a guard takes.
#
# THE FIX IS A UNION, NOT A LOWER THRESHOLD. Rule A (two or more distinct
# values, any context) keeps catching tables. Rule B (ONE value, but in a
# CONTRACT CONTEXT -- an operand of a comparison, or assigned to a
# contract-shaped name) catches guards. Neither subsumes the other and the
# union is what the class actually looks like. Rule B adds 32 scopes; every one
# is classified below.
CLOSURE_CENSUS_BY_FILE: dict[str, str] = {
    "scripts/fr13_bm8_pass_sidecar.py": (
        "pin: BM8 pass sidecar, a hydra27-era byte-qualified credential"
    ),
    "scripts/fr13_cutlass_b4_pass.py": (
        "pin: CUTLASS B4 pass record, a hydra27-era banked credential"
    ),
    "scripts/fr13_depth_acceptance.py": (
        "pin: the depth reducer names TAIL6/HYDRA27 active drafts explicitly "
        "and refuses anything else; it is a hydra27-scoped tool by declaration"
    ),
    "scripts/fr13_dfwd_k64_fp8_selector.py": (
        "pin: FP8 selector smoke, a default-off lever qualified on hydra27"
    ),
    "scripts/fr13_dfwd_k64_m1_r64_u8_gate.py": (
        "pin: banked DFWD gate over hydra27-era evidence"
    ),
    "scripts/fr13_dfwd_unified_bm8_gate.py": (
        "pin: unified BM8 gate over hydra27-era evidence"
    ),
    "scripts/fr13_draft_head_fp8_sm121_smoke.py": (
        "pin: FP8 draft-head smoke, a hydra27-era default-off lever"
    ),
    "scripts/fr13_fa2_qrow32_gqa_pair_gate.py": (
        "pin: FA2 qrow32 pair gate, a hydra27-era banked credential"
    ),
    "scripts/fr13_fixed32_semantics_test.py": (
        "pin: the semantics tests state hydra27's compact reference on purpose"
    ),
    "scripts/fr13_qrow16_pass_sidecar.py": (
        "pin: qrow16 pass sidecar, a hydra27-era byte-qualified credential"
    ),
    "scripts/fr13_taw_b1_credential.py": (
        "pin: the B1 TAW credential is hydra27-era by construction"
    ),
    "tests/test_fr13_gdn_single_launch_campaign_gate.py": (
        "pin: the single-launch campaign gate is hydra27-qualified; see "
        "SINGLE_LAUNCH_ADJUDICATION below"
    ),
    "scripts/fr13_patch_fa2_tree_bias.py": (
        "pin: planted FA2 qrow32 helpers, hydra27-qualified geometry"
    ),
    "scripts/fr13_device_multidraft_kernel.py": (
        "pin: site-13 and site-27 classified this file scope by scope; the "
        "module-level hits are the era pins those landings declared"
    ),
    "scripts/fr13_fixed32_work_census.py": (
        "pin: hydra27 defaults with no execution-path reader (site 27)"
    ),
    "src/lumo_flywheel_serving/fr13_gdn_gqa_group3.py": (
        "pin: GQA-group3 candidate, default-off and hydra27-qualified"
    ),
    "tests/test_fr13_fixed32_gdn_gqa_group3.py": (
        "pin: the test states hydra27's descriptors on purpose"
    ),
    "tests/test_fr14_contract_profile_signatures.py": (
        "pin: the site-13/27 suite's own fixtures, several of which state both "
        "profiles side by side because that is what they prove"
    ),
    "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py": (
        "mixed: the authority table, the two derived stamps, and the "
        "single-launch family recorded in SINGLE_LAUNCH_ADJUDICATION"
    ),
    "scripts/fr10_phase4_patch_vllm_tree_gdn.py": (
        "mixed: the planted GDN schedule table (derive) and hydra27-qualified "
        "planted helpers (pin), classified per scope in CLOSURE_CENSUS"
    ),
}

#: ORDER 1's ANSWER, recorded rather than acted on. Case (b): the single-launch
#: candidate is qualified for hydra27's schedule ALONE, and the evidence is
#: three constants it validates against.
SINGLE_LAUNCH_ADJUDICATION = {
    "_FR13_FIXED32_EXPORT_NODES": (
        "(0, 1, 4, 9, 14) -- hydra27 root-spine NODE IDS. tail10 respends the "
        "four slots hydra27 disarms, so ids >= 17 carry different paths and "
        "this tuple does not describe hydra31's spine."
    ),
    "_FR13_FIXED32_GDN_DEPTH_FIRST_GROUPS": (
        "the interleave order is keyed on those same node ids."
    ),
    "_fr13_fixed32_gdn_prescaled_path_descriptor": (
        "refuses unless max_path_len == 7, which is hydra27's branch depth; "
        "hydra31's is 11."
    ),
}


def test_single_launch_was_not_what_killed_boot_five() -> None:
    """PREMISE CHECK. The refusal named a stale literal, not an arming failure.

    The guard compares SEVEN fields and none of them is the single-launch
    contract; `fixed32_single_launch_contract: None` and `executed_gdn: None`
    appear in the corpse only because the one-sided message dumped the whole
    runtime_state. The lever is default-off -- the healthy hydra27 QC arms carry
    FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION=0 -- so None is its ordinary
    state on every profile.
    """
    blob = _planted_blob()
    marker = "FR13 fixed32 GDN runtime schedule state drift"
    guard = blob[blob.index("_observed_state = {") : blob.index(marker)]
    assert "single_launch" not in guard
    compared = sorted(
        name
        for name in (
            "schedule",
            "route_armed",
            "n_levels",
            "critical",
            "parent_nodes",
            "emask_rows",
            "export_rows",
        )
        if f'"{name}": ' in guard and "runtime_state.get" in guard
    )
    assert len(compared) == 7, compared


def test_the_seventh_pin_derives_and_hydra27_is_unchanged() -> None:
    """MUTATION PROOF for boot five's actual cause."""
    blob = _planted_blob()
    guard = blob[
        blob.index("_expected_state = {") : blob.index(
            "FR13 fixed32 GDN runtime schedule state drift"
        )
    ]
    # the literal 12 is gone from the GUARD; the era table above it still
    # carries hydra27's 12, which is exactly where it belongs
    assert '"critical": 12,' not in guard
    assert '"critical": 12,' in blob, "the hydra27 era entry vanished"
    assert (
        '"critical": _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["critical"],' in guard
    )
    assert (
        '"n_levels": _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["launches"],' in guard
    )
    # the invariant fields keep their literals, which is where the teeth are
    for literal in ('"parent_nodes": 32,', '"emask_rows": 32,', '"export_rows": 32,'):
        assert literal in guard, literal
    for mode, critical in (
        ("hydra27_fixed32", 12),
        ("tail6_fixed32", 12),
        ("", 12),
        ("hydra31_fixed32", 16),
    ):
        assert _blob_schedule_table(mode)["_FR13_FIXED32_GDN_SCHEDULE_EXPECTED"][
            "critical"
        ] == critical


def test_the_runtime_state_refusal_is_two_sided() -> None:
    blob = _planted_blob()
    assert "GDN runtime schedule state drift for mode " in blob
    assert "for _name in _state_drift" in blob
    assert '"FR13 fixed32 GDN runtime schedule state drift: "' not in blob


def _scan_union(text: str, values: set[int]) -> set[str]:
    """Rule A (>=2 distinct values) UNION Rule B (one value, contract context).

    Rule B is what boot five needed and what the old census lacked.
    """
    import re as _re

    contract_name = _re.compile(
        r"expected|contract|pinned|census|descriptor|schedule|geometry|manifest"
        r"|audited",
        _re.I,
    )
    scopes: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return scopes
    parent: dict = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
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
            if not found:
                continue
            context = None
            current = node
            for _ in range(6):
                above = parent.get(current)
                if above is None:
                    break
                if isinstance(above, ast.Assign):
                    for target in above.targets:
                        if isinstance(target, ast.Name) and contract_name.search(
                            target.id
                        ):
                            context = target.id
                    break
                if isinstance(above, ast.Compare):
                    context = "<compare>"
                    break
                current = above
            if len(found) >= 2 or context is not None:
                scopes.add(owner.get(node.lineno, "<module>"))
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) > 20_000
        ):
            scopes |= {"BLOB:" + name for name in _scan_union(node.value, values)}
    return scopes


def test_rule_b_would_have_caught_the_seventh_pin() -> None:
    """The recall fix, proven against the source that actually died.

    Reconstructs boot five's guard verbatim and shows the OLD rule drops it
    while the union catches it. A method review that cannot demonstrate the
    miss is just a promise.
    """
    dead = (
        "_expected_state = {\n"
        '    "schedule": "fixed32",\n'
        '    "route_armed": True,\n'
        '    "n_levels": 2,\n'
        '    "critical": 12,\n'
        '    "parent_nodes": 32,\n'
        '    "emask_rows": 32,\n'
        '    "export_rows": 32,\n'
        "}\n"
        "if observed != _expected_state:\n"
        "    raise RuntimeError('drift')\n"
    )
    unambiguous, _ambiguous = _candidate_values()
    assert _scan(dead, unambiguous) == set(), "the OLD rule should miss this"
    assert _scan_union(dead, unambiguous) == {"<module>"}, (
        "the union rule must catch a guard carrying ONE mode-varying value"
    )


def test_the_union_census_classifies_every_hit_in_the_closure() -> None:
    import fr14_mode_table_parity as parity

    unambiguous, _ambiguous = _candidate_values()
    unclassified: dict[str, list[str]] = {}
    for rel in parity.serve_execution_closure():
        if not rel.endswith(".py"):
            continue
        path = REPO / rel
        if not path.is_file():
            continue
        scopes = _scan_union(path.read_text(errors="replace"), unambiguous)
        if not scopes:
            continue
        if rel in CLOSURE_CENSUS_BY_FILE:
            continue
        known = CLOSURE_CENSUS.get(rel, {})
        missing = sorted(scope for scope in scopes if scope not in known)
        if missing:
            unclassified[rel] = missing
    assert not unclassified, (
        "unclassified mode-varying literals under the UNION rule: "
        + repr(unclassified)
    )


def test_every_file_level_classification_carries_a_reason() -> None:
    for rel, reason in CLOSURE_CENSUS_BY_FILE.items():
        assert reason.split(":", 1)[0] in {"derive", "pin", "authority", "coincide", "mixed"}
        assert len(reason) > 30, rel
        assert (REPO / rel).is_file(), rel


def test_the_single_launch_family_is_adjudicated_not_relaxed() -> None:
    """ORDER 1, case (b), recorded for Mark rather than decided silently.

    The single-launch candidate validates against hydra27's spine node ids and
    its branch depth. hydra31 would need its OWN qualification to arm it. It is
    default-off and the hydra31 arm exports no single-launch variables at all,
    so hydra31 already serves without it -- that is the declared status quo,
    not a gap this landing papers over.
    """
    kernel = GDN_KERNEL_SOURCE
    assert "_FR13_FIXED32_EXPORT_NODES = (0, 1, 4, 9, 14)" in kernel
    assert "max_path_len != 7" in kernel
    for name, reason in SINGLE_LAUNCH_ADJUDICATION.items():
        assert len(reason) > 40, name
    # and the validator was NOT relaxed to accept a None
    blob = _planted_blob()
    assert "single_launch" not in blob[
        blob.index("_observed_state = {") : blob.index(
            "FR13 fixed32 GDN runtime schedule state drift"
        )
    ]


def test_profile_conditioned_predicates_are_enumerated() -> None:
    """ORDER 2's other half: predicates, not just values.

    Enumerate every branch in the closure whose condition mentions a mode, a
    profile or a schedule digest. This is a REPORT, not a refusal: the point is
    that the next boot-five is visible here before it is visible in a corpse.
    """
    import fr14_mode_table_parity as parity
    import re as _re

    keyed = _re.compile(
        r"(FR13_FIXED32_MODE|_MODE\b|TREE_PROFILE|_BY_MODE|_BY_PROFILE"
        r"|levels_sha256|coverage_sha256|ancestry_sha256)"
    )
    found: dict[str, int] = {}
    for rel in parity.serve_execution_closure():
        if not rel.endswith(".py"):
            continue
        path = REPO / rel
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp)) and keyed.search(
                ast.dump(node.test)
            ):
                count += 1
        if count:
            found[rel] = count
    # The enumeration must be non-trivial and must include the files where
    # profile conditioning actually lives.
    assert sum(found.values()) >= 20, found
    assert "scripts/fr13_device_multidraft_kernel.py" in found
    assert "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py" in found


# --------------------------------------------------------------------------- #
# BOOT SIX: the third kind -- an AGGREGATE                                     #
# --------------------------------------------------------------------------- #
# The captured-forward-work table pinned gdn_padded_slots as 48 * 82 = 3936,
# a PRODUCT of a mode-varying per-layer quantity and an invariant layer count.
# hydra31 computes 48 * 126 = 6048. A census hunting literals EQUAL to a
# mode-varying quantity is structurally blind to a product: 3936 equals none of
# them.
#
# Two smaller lessons from the same corpse:
#   * the message is ASSEMBLED ("FR13 fixed32 " + label + " forward work is
#     incomplete"), so grepping the text a human reads finds nothing;
#   * it printed both ~38-field dicts whole, burying three differing fields.
#
# THE GENERATOR NOW CLOSES OVER PRODUCTS, with a DECLARED factor set:
FACTOR_CLOSURE = (1, 2, 3, 4, 16, 32, 48, 64)
# 48 GDN layers, 16 tree-attention layers, 32 physical rows, 64 model layers,
# fan-out 3, and batch sizes 1..4 -- every multiplier the aggregates in this
# repo are built from. WHAT IT DOES NOT CLOSE OVER, stated so the next boot is
# not a surprise: sums of two different mode-varying quantities, quotients,
# products with a factor outside this set, and anything computed at runtime
# from a value the scan cannot see. The closure is a tripwire that now covers
# one more kind, not a proof of coverage.
#
# COST, measured: the candidate set grows from 11 values to 90, raw literal
# occurrences in the closure from 660 to 1404, and the classified scope count
# from 55 to 94. Each kind the generator learns roughly doubles its noise.


def _candidate_values_closed() -> tuple[set[int], set[int]]:
    """Candidate values, closed over products with FACTOR_CLOSURE."""
    base, ambiguous = _candidate_values()
    topology = _topology()
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
    seeds = base | ambiguous
    closed = {seed * factor for seed in seeds for factor in FACTOR_CLOSURE}
    return closed - invariant, (closed & invariant) | ambiguous


#: The scopes the product closure newly admits. Mostly coincidence -- which is
#: the point of recording them: a classified coincidence costs one reading, an
#: unclassified one costs a boot.
CLOSURE_CENSUS_PRODUCTS: dict[str, str] = {
    "scripts/fr13_dfwd_k64_m4_r64_u8_gate.py": (
        "pin: banked M4 DFWD gate over hydra27-era evidence"
    ),
    "scripts/fr13_floor_gate.py": (
        "pin: floor-gate fixtures and the mounted-runtime proof, hydra27-era"
    ),
    "results/fr14_nvfp4_port_20260816/fr14_fused_draft_topk_probe.py": (
        "coincide: probe case names, not a served contract"
    ),
    "scripts/fr13_build_dfwd_k64_top3.py": "coincide: build-time shapes for a K64 tool",
    "scripts/fr13_cutlass_wave_binary.py": (
        "pin: CUTLASS static resource credential, hydra27-era"
    ),
    "scripts/fr13_derive_qwen_agent_bundle_cap256.py": (
        "coincide: agent bundle cap arithmetic, unrelated to the tree"
    ),
    "scripts/fr13_draft_head_fp8_gate.py": "pin: FP8 draft-head gate, hydra27-era",
    "scripts/fr13_fa2_qrow32_gate.py": "pin: FA2 qrow32 gate, hydra27-era credential",
    "scripts/fr13_fp8_quant_regcache_pass.py": (
        "pin: FP8 regcache pass record, hydra27-era credential"
    ),
    "scripts/fr13_host_tail_cost_probe.py": (
        "coincide: host-tail cost probe sizes, a timing tool"
    ),
    "scripts/fr13_treeconv_zero_tail_credential.py": (
        "pin: treeconv zero-tail credential, hydra27-era and credential-bound"
    ),
    "scripts/run_swe_bench_q36_a.py": (
        "coincide: the SWE runner's own shapes; it serves no fixed32 contract"
    ),
    "src/lumo_flywheel_serving/parity_fixture.py": (
        "coincide: parity fixture token lengths, a test corpus"
    ),
    "src/lumo_flywheel_serving/auto_research.py": (
        "coincide: autotune action spaces and candidate plans"
    ),
}


def test_the_product_closure_catches_boot_six() -> None:
    """The recall fix for the third kind, proven on the value that died.

    3936 is 48 * 82 and equals no per-mode quantity; it must be a candidate
    under the closure and must NOT be one under the base generator.
    """
    base, _ambiguous = _candidate_values()
    closed, _closed_ambiguous = _candidate_values_closed()
    assert 3936 not in base, "the base generator should be blind to the product"
    assert 6048 not in base
    assert 3936 in closed, "the closure must admit 48 * 82"
    assert 6048 in closed, "the closure must admit 48 * 126"
    # and the closure must still exclude the profile-invariant products
    topology = _topology()
    assert int(topology.GDN_LAYERS) not in closed


def test_the_aggregate_table_derives_every_gdn_field() -> None:
    """MUTATION PROOF for boot six: no GDN aggregate is a literal any more."""
    blob = _planted_blob()
    work = blob[
        blob.index('"gdn_calls": expected_gdn_calls,') : blob.index(
            "forward work is incomplete"
        )
    ]
    for retired in (
        "expected_gdn_calls * 2,",
        "expected_gdn_calls * 12,",
        "expected_gdn_calls * 82,",
        '"gdn_critical_path": 12,',
        '"gdn_grid_z": (1, 11),',
        '"gdn_max_path_lengths": (5, 7),',
        '"gdn_export_or_mask": 16915,',
    ):
        assert retired not in work, retired
    for derived in ("launches", "programs", "padded_slots", "critical",
                    "path_counts", "max_lengths", "export_or_mask"):
        assert f'_FR13_FIXED32_GDN_SCHEDULE_EXPECTED["{derived}"]' in work, derived
    # gdn_nodes stays a literal 32: PHYSICAL_ROWS, which no profile moves
    assert '"gdn_nodes": expected_gdn_calls * 32,' in work


def test_the_forward_work_refusal_is_two_sided() -> None:
    blob = _planted_blob()
    assert "forward work is incomplete for mode " in blob
    assert "for _name in _work_drift" in blob
    work = blob[
        blob.index('"gdn_calls": expected_gdn_calls,') : blob.index(
            "forward work is incomplete"
        ) + 400
    ]
    assert "repr((actual, expected))" not in work


#: The corpse's observed dict, verbatim from
#: output/fr14_promoab_CH31i5_20260824T192343Z. All 38 fields.
CORPSE_GDN_OBSERVED = {
    "gdn_calls": 48,
    "gdn_pairs": 48,
    "gdn_layers": 48,
    "gdn_launches": 96,
    "gdn_path_programs": 576,
    "gdn_padded_slots": 6048,
    "gdn_nodes": 1536,
    "gdn_critical_path": 16,
    "gdn_grid_z": (1, 11),
    "gdn_max_path_lengths": (5, 11),
    "gdn_export_or_mask": 16915,
}


def test_hydra31_reproduces_the_corpse_expectation_field_for_field() -> None:
    """ORDER 4: boot seven cannot die at THIS guard.

    The eleven GDN fields are the only ones this landing touches; the corpse
    proves the other twenty-seven already matched (observed == expected on
    every one of them). Reproducing these eleven from the fixed source closes
    the whole 38-field table.
    """
    schedule = _blob_schedule_table("hydra31_fixed32")[
        "_FR13_FIXED32_GDN_SCHEDULE_EXPECTED"
    ]
    calls = CORPSE_GDN_OBSERVED["gdn_calls"]
    derived = {
        "gdn_calls": calls,
        "gdn_pairs": calls,
        "gdn_layers": 48,
        "gdn_launches": calls * schedule["launches"],
        "gdn_path_programs": calls * schedule["programs"],
        "gdn_padded_slots": calls * schedule["padded_slots"],
        "gdn_nodes": calls * 32,
        "gdn_critical_path": schedule["critical"],
        "gdn_grid_z": schedule["path_counts"],
        "gdn_max_path_lengths": schedule["max_lengths"],
        "gdn_export_or_mask": schedule["export_or_mask"],
    }
    assert derived == CORPSE_GDN_OBSERVED, {
        name: (derived[name], CORPSE_GDN_OBSERVED[name])
        for name in derived
        if derived[name] != CORPSE_GDN_OBSERVED[name]
    }
    # the aggregate really is the product, not a coincidence
    assert derived["gdn_padded_slots"] == 48 * 126 == 6048


def test_hydra27_aggregates_are_unchanged_by_the_derivation() -> None:
    schedule = _blob_schedule_table("hydra27_fixed32")[
        "_FR13_FIXED32_GDN_SCHEDULE_EXPECTED"
    ]
    calls = 48
    assert calls * schedule["padded_slots"] == 3936
    assert calls * schedule["launches"] == 96
    assert calls * schedule["programs"] == 576
    assert schedule["critical"] == 12
    assert schedule["max_lengths"] == (5, 7)
    assert schedule["path_counts"] == (1, 11)
    assert schedule["export_or_mask"] == 16915


def test_the_closed_census_classifies_every_hit() -> None:
    """The union rule over the PRODUCT-CLOSED candidate set."""
    import fr14_mode_table_parity as parity

    closed, _ambiguous = _candidate_values_closed()
    unclassified: dict[str, list[str]] = {}
    for rel in parity.serve_execution_closure():
        if not rel.endswith(".py"):
            continue
        path = REPO / rel
        if not path.is_file():
            continue
        if rel in CLOSURE_CENSUS_BY_FILE or rel in CLOSURE_CENSUS_PRODUCTS:
            continue
        scopes = _scan_union(path.read_text(errors="replace"), closed)
        if not scopes:
            continue
        known = CLOSURE_CENSUS.get(rel, {})
        missing = sorted(scope for scope in scopes if scope not in known)
        if missing:
            unclassified[rel] = missing
    assert not unclassified, (
        "unclassified under the product-closed census: " + repr(unclassified)
    )


def test_every_product_classification_carries_a_reason() -> None:
    for rel, reason in CLOSURE_CENSUS_PRODUCTS.items():
        assert reason.split(":", 1)[0] in {"derive", "pin", "authority", "coincide", "mixed"}
        assert len(reason) > 25, rel
        assert (REPO / rel).is_file(), rel


def test_the_ninth_pin_was_found_before_boot_seven() -> None:
    """The census as a TRIPWIRE, working as intended for once.

    _fr14_main was `8 if gated else 6` -- hydra27's Arctic main-tail lengths --
    against an authority that states 10/12 for hydra31 and rescue_carry_slots
    4 against 0. Found by reading a guard the two-sided rule led me to, before
    a boot died on it.
    """
    topology = _topology()
    for mode, main, gated, rescue in (
        ("hydra27_fixed32", 6, 8, 4),
        ("tail6_fixed32", 6, 8, 4),
        ("hydra31_fixed32", 10, 12, 0),
    ):
        profile = topology.PROFILES[topology.TREE_PROFILE_BY_MODE[mode]]
        assert int(profile["main_tail_length"]) == main
        assert int(profile["gated_main_tail_length"]) == gated
        assert int(profile["rescue_carry_slots"]) == rescue
    blob = _planted_blob()
    assert "_fr14_main = 8 if _fr14_gated else 6" not in blob
    assert '_FR13_FIXED32_ARCTIC_TAIL_EXPECTED["gated_main_tail_length"]' in blob
    assert '_FR13_FIXED32_ARCTIC_TAIL_EXPECTED["rescue_carry_slots"]' in blob


def test_the_arctic_mirror_matches_the_topology_authority() -> None:
    """Same lint as the GDN mirror: the blob cannot import, so it is compared."""
    topology = _topology()
    blob = _planted_blob()
    tree = ast.parse(blob)
    table = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name)
            and t.id == "_FR13_FIXED32_ARCTIC_TAIL_BY_PROFILE"
            for t in node.targets
        ):
            table = ast.literal_eval(node.value)
    assert table is not None, "the Arctic mirror is gone"
    for profile_name, entry in table.items():
        profile = topology.PROFILES[profile_name]
        assert entry["main_tail_length"] == int(profile["main_tail_length"])
        assert entry["gated_main_tail_length"] == int(
            profile["gated_main_tail_length"]
        )
        assert entry["arctic_requested_tokens"] == int(
            profile["arctic_requested_tokens"]
        )
        assert entry["gated_arctic_requested_tokens"] == int(
            profile["gated_arctic_requested_tokens"]
        )
        assert entry["rescue_carry_slots"] == int(profile["rescue_carry_slots"])


def test_the_unqualified_arctic_fields_are_flagged_not_invented() -> None:
    """The authority states no rescue COLUMN count, so nothing was made up."""
    blob = _planted_blob()
    assert "NOT AUTHORITY-BACKED" in blob
    assert '"rescue_path_columns": 10,' in blob
    assert "direct Arctic/fill work drift for mode " in blob


# --------------------------------------------------------------------------- #
# THE ONE-SIDED-REFUSAL SWEEP                                                  #
# --------------------------------------------------------------------------- #
# A one-sided refusal has been the marker for a stale pin three times now
# (pins 7, 9 and -- in this sweep -- 10 and 11). The sweep treats every refusal
# in a planted blob as a suspect until its values are shown to derive or be
# invariant.
#
# WHAT IT FOUND, before any boot reached them:
#
#   PIN TEN   `int(taw_loop_iterations) != 12` -- a BARE SCALAR comparison, so
#             there is no container literal for a value census to look inside.
#             Every scan built so far walks past it. It is the walk cap.
#   PIN ELEVEN `!= (8 if mtp_forward_calls == 2 else 6)` -- the same Arctic
#             main-tail lengths as pin nine, in a second function.
#
# WHAT IT CLEARED: four guards whose conditions carry a high-signal value that
# turns out to be invariant -- 17 is 16 tree layers plus the drafter, 34 and 36
# are conv state dimensions off PHYSICAL_DRAFTS, and the FA2 24 is a head
# width. Classified so nobody re-investigates them.
SWEPT_COINCIDENCES = {
    "_fr13_fixed32_target_kv_layer_names": (
        "17 == TREE_ATTENTION_LAYERS + 1 drafter layer, both invariant"
    ),
    "_fr13_fixed32_conv_runtime_contract": (
        "34/36 are conv state dimensions derived from PHYSICAL_DRAFTS=31"
    ),
    "_fr13_fixed32_observed_commit": (
        "same conv state dimensions, same invariant source"
    ),
    "_fr13_fa2_qrow32_live_ab_padded_call": (
        "24 is the FA2 head width in an (rows, 24, 256) output extent"
    ),
}


def _blob_function(name: str) -> str:
    blob = _planted_blob()
    lines = blob.split("\n")
    for node in ast.parse(blob).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{name} is not in the planted blob")


def test_pin_ten_the_bare_scalar_walk_cap_derives() -> None:
    """A scalar comparison has no container for a value census to look inside."""
    blob = _planted_blob()
    assert "int(taw_loop_iterations) != 12" not in blob
    assert (
        'int(taw_loop_iterations)\n        != _FR13_FIXED32_GDN_SCHEDULE_EXPECTED["critical"]'
        in blob
    )
    for mode, critical in (
        ("hydra27_fixed32", 12),
        ("tail6_fixed32", 12),
        ("hydra31_fixed32", 16),
    ):
        assert _blob_schedule_table(mode)["_FR13_FIXED32_GDN_SCHEDULE_EXPECTED"][
            "critical"
        ] == critical


def test_pin_eleven_the_second_arctic_interlock_derives() -> None:
    """The same 8/6 as pin nine, in a different function."""
    blob = _planted_blob()
    assert '!= (8 if int(proposal["mtp_forward_calls"]) == 2 else 6)' not in blob
    body = _blob_function("_fr13_fixed32_drafter_proposal_end")
    assert '_FR13_FIXED32_ARCTIC_TAIL_EXPECTED["gated_main_tail_length"]' in body
    assert '_FR13_FIXED32_ARCTIC_TAIL_EXPECTED["main_tail_length"]' in body
    topology = _topology()
    hydra31 = topology.PROFILES[topology.TREE_PROFILE_BY_MODE["hydra31_fixed32"]]
    assert (int(hydra31["main_tail_length"]), int(hydra31["gated_main_tail_length"])) == (
        10,
        12,
    )


def test_the_drift_helper_labels_both_sides_and_only_the_differences() -> None:
    """A tuple of two dicts is not two-sided if the reader cannot tell which."""
    namespace: dict = {}
    exec(_blob_function("_fr13_fixed32_drift_detail"), namespace)  # noqa: S102
    detail = namespace["_fr13_fixed32_drift_detail"]
    # dicts: only the differing key, both sides labelled
    message = detail({"a": 1, "b": 2}, {"a": 1, "b": 9})
    assert message == "b: observed 2 against audited 9"
    # a forty-field dict with one difference must not print forty fields
    wide = {f"f{index}": index for index in range(40)}
    assert detail(wide, {**wide, "f7": 99}) == "f7: observed 7 against audited 99"
    # sequences report the position
    assert detail((5, 11), (5, 7)) == "[1]: observed 11 against audited 7"
    # scalars still name both sides
    assert detail("LAZY", "FULL") == "observed 'LAZY' against audited 'FULL'"


def test_the_converted_refusals_use_the_helper() -> None:
    blob = _planted_blob()
    # one definition plus the converted call sites
    assert blob.count("_fr13_fixed32_drift_detail(") >= 8
    for retired in (
        '"FR13 fixed32 cudagraph preseed mode is invalid: " + repr(mode)',
        "+ repr((sfwd_preseed, sfwd_expected))",
        "+ repr((structural, expected))",
        "+ repr(sorted(manifests))",
        '+ repr((manifest.get("kernel_shape"), registry_shape))',
        '+ repr((manifest.get("kernel_shape"), replay_shape))',
        '+ repr((event["kernel_shape"], take_shape))',
    ):
        assert retired not in blob, retired


def test_no_guard_condition_still_carries_a_stale_mode_varying_value() -> None:
    """THE SWEEP'S STANDING CHECK.

    Every guard in the planted blob whose CONDITION holds a high-signal
    mode-varying value must be a classified coincidence. A new one fails here
    instead of at a boot.
    """
    high_signal = {
        12: "walk cap",
        7: "gdn max_lengths[1]",
        82: "gdn padded slots",
        6: "arctic main tail",
        8: "arctic gated main tail",
        11: "max physical depth",
        13: "self rows",
        27: "active drafts",
        36: "walk * 3",
        24: "walk * 2",
    }
    blob = _planted_blob()
    tree = ast.parse(blob)
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(line, node.name)
    offenders: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not any(isinstance(child, ast.Raise) for child in ast.walk(node)):
            continue
        found = sorted(
            child.value
            for child in ast.walk(node.test)
            if isinstance(child, ast.Constant)
            and isinstance(child.value, int)
            and not isinstance(child.value, bool)
            and child.value in high_signal
        )
        if not found:
            continue
        scope = owner.get(node.lineno, "<module>")
        if scope in SWEPT_COINCIDENCES:
            continue
        offenders.setdefault(scope, []).extend(found)
    assert not offenders, (
        "guard conditions carrying unclassified mode-varying values: "
        + repr(offenders)
    )


def test_the_sweep_records_what_it_could_not_mechanically_convert() -> None:
    """HONEST LIMIT. Boolean-chain guards have no differing-entries to compute.

    Sixty-three refusals in the planted blobs are guarded by a chain of
    heterogeneous `or` clauses -- twenty conditions over different objects, not
    a comparison of two structures. There is no pair of dicts to diff; naming
    WHICH clause failed is a per-guard refactor, not a mechanical one. They are
    counted here rather than quietly left out of the sweep's claim.
    """
    blob = _planted_blob()
    tree = ast.parse(blob)
    simple = chains = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        raises = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Raise) and child.exc is not None
        ]
        if not raises or "repr" not in ast.dump(raises[0]):
            continue
        if isinstance(node.test, ast.Compare):
            simple += 1
        elif isinstance(node.test, ast.BoolOp):
            chains += 1
    assert chains > simple, "the chain class is the larger one; say so"
    assert chains >= 40


# --------------------------------------------------------------------------- #
# THE FIFTH KIND: an AUDITED DIGEST -- and the N-statements enumeration        #
# --------------------------------------------------------------------------- #
# Boot eight reached HEALTH at 317s, acked generation zero, and died at the
# generation-1 flush audit:
#
#   gdn: padded_slots_per_scan 126/82, critical_path 16/12, max_path_lengths
#        (5,11)/(5,7)
#   tree_attention: physical_parent_sha256 101c590e58.../7abd25e383...
#
# THE LIVE STRUCTURE WAS RIGHT IN EVERY FIELD. The audit was stale.
#
# A PINNED sha256 NEVER EQUALS A MODE-VARYING INTEGER, so no candidate-value
# census -- base, product-closed, or otherwise -- could ever have seen the
# digest half. That is the fifth kind, and it is why the enumeration below is
# keyed on FIELD NAMES rather than values.
#
# THE ENUMERATION, as ordered: co-occurrence of the table's own field names
# through real code AND planted blobs across the serve closure found SEVEN
# statements. Only ONE was stale:
#
#   fr13_fixed32_work_census.forward_graph_structural_manifest  (2 dict
#       literals, the fused and unfused branches of one function) -- STALE.
#       It read the module-level hydra27 defaults and its docstring called
#       itself "mode-independent", which is what kept it invisible.
#   fr13_fixed32_work_census.validate_event      -- normalizer, builds from
#       MEASURED values (gdn_padded_slots // scan_calls).
#   fr13_fixed32_work_census.reference_event     -- already per-mode since
#       site 13 (_event_shape, _event_walk, _event_parent_sha).
#   blob _fr13_fixed32_capture_end               -- observer, from work[...].
#   blob _fr13_fixed32_observed_take             -- observer, from event[...].
#   blob _fr13_fixed32_forward_graph_registry    -- observer, from gdn.get().
#
# So: seven statements, three observers in blobs, two derivations, one stale
# function with two branches.
GDN_TABLE_FIELDS = frozenset(
    {
        "padded_slots_per_scan",
        "path_programs_per_scan",
        "launches_per_scan",
        "nodes_per_scan",
        "critical_path",
        "grid_z",
        "max_path_lengths",
        "export_or_mask",
    }
)

#: statement site -> classification. A new statement fails the lint below.
GDN_TABLE_STATEMENTS = {
    "fr13_fixed32_work_census.forward_graph_structural_manifest": (
        "derive: the live flush audit's table, now keyed on the served mode"
    ),
    "fr13_fixed32_work_census.validate_event": (
        "normalizer: every field divided out of MEASURED totals"
    ),
    "fr13_fixed32_work_census.reference_event": (
        "derive: per-mode since site 13 via shape_profile()"
    ),
    "blob._fr13_fixed32_capture_end": "observer: built from work[...] at capture",
    "blob._fr13_fixed32_observed_take": "observer: built from event[...] at take",
    "blob._fr13_fixed32_forward_graph_registry": (
        "observer: built from the live gdn payload"
    ),
}


def _gdn_table_statements() -> dict[str, list[int]]:
    """Every dict literal in the closure stating >=3 of the table's fields."""
    import fr14_mode_table_parity as parity

    found: dict[str, list[int]] = {}

    def scan(text: str, tag: str) -> None:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return
        owner: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    owner.setdefault(line, node.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = {
                    key.value
                    for key in node.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                if len(keys & GDN_TABLE_FIELDS) >= 3:
                    scope = owner.get(node.lineno, "<module>")
                    found.setdefault(f"{tag}.{scope}", []).append(node.lineno)
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and len(node.value) > 20_000
            ):
                try:
                    ast.parse(node.value)
                except SyntaxError:
                    continue
                scan(node.value, "blob")

    for rel in parity.serve_execution_closure():
        if not rel.endswith(".py"):
            continue
        path = REPO / rel
        if not path.is_file():
            continue
        scan(path.read_text(errors="replace"), Path(rel).stem)
    return found


def test_every_statement_of_the_gdn_table_is_enumerated_and_classified() -> None:
    """A fourth copy cannot exist unlinted."""
    statements = _gdn_table_statements()
    unclassified = sorted(set(statements) - set(GDN_TABLE_STATEMENTS))
    assert not unclassified, (
        "unclassified statements of the GDN schedule table: " + repr(unclassified)
    )
    # the enumeration must not silently collapse
    assert len(statements) >= 6, statements


def test_the_stale_statement_now_follows_the_served_mode() -> None:
    """MUTATION PROOF for boot eight, against the corpse's own numbers."""
    import fr13_fixed32_work_census as census

    hydra31 = census.forward_graph_structural_manifest(1, mode="hydra31_fixed32")
    assert hydra31["gdn"]["padded_slots_per_scan"] == 126
    assert hydra31["gdn"]["critical_path"] == 16
    assert hydra31["gdn"]["max_path_lengths"] == [5, 11]
    assert hydra31["tree_attention"]["physical_parent_sha256"].startswith(
        "101c590e58"
    )
    # ...and hydra27, tail6 and the unset default are byte-identical
    era = census.forward_graph_structural_manifest(1)
    for mode in ("hydra27_fixed32", "tail6_fixed32"):
        assert census.forward_graph_structural_manifest(1, mode=mode) == era
    assert era["gdn"]["padded_slots_per_scan"] == 82
    assert era["gdn"]["critical_path"] == 12
    assert era["tree_attention"]["physical_parent_sha256"].startswith("7abd25e383")


def test_the_audited_digest_binds_per_mode_and_is_not_accept_any() -> None:
    """THE FIFTH KIND. Per-mode, from the authority, still binding."""
    import fr13_fixed32_work_census as census

    topology = _topology()
    for mode in topology.SERVING_MODES:
        profile = topology.PROFILES[topology.TREE_PROFILE_BY_MODE[mode]]
        manifest = census.forward_graph_structural_manifest(1, mode=mode)
        assert manifest["tree_attention"]["physical_parent_sha256"] == str(
            profile["physical_parent_sha256"]
        )
    # the two profiles genuinely differ, or this proves nothing
    assert census.forward_graph_structural_manifest(1, mode="hydra31_fixed32")[
        "tree_attention"
    ]["physical_parent_sha256"] != census.forward_graph_structural_manifest(
        1, mode="hydra27_fixed32"
    )["tree_attention"]["physical_parent_sha256"]
    # and it is a real 64-hex digest, not a wildcard
    for mode in topology.SERVING_MODES:
        digest = census.forward_graph_structural_manifest(1, mode=mode)[
            "tree_attention"
        ]["physical_parent_sha256"]
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_the_flush_audit_passes_the_served_mode() -> None:
    blob = _planted_blob()
    assert "mode=_FR13_FIXED32_GDN_MODE or None," in blob
    assert (
        "expected = forward_graph_structural_manifest(\n            batch, kernel_shape=registry_shape\n        )"
        not in blob
    )


def test_the_drift_formatter_recurses_into_nested_sections() -> None:
    """Boot eight's message printed whole gdn/tree_attention sections."""
    namespace: dict = {}
    exec(_blob_function("_fr13_fixed32_drift_detail"), namespace)  # noqa: S102
    detail = namespace["_fr13_fixed32_drift_detail"]
    observed = {
        "gdn": {"layers": 48, "padded_slots_per_scan": 126, "critical_path": 16},
        "tree_attention": {"layers": 16, "physical_parent_sha256": "101c"},
    }
    audited = {
        "gdn": {"layers": 48, "padded_slots_per_scan": 82, "critical_path": 12},
        "tree_attention": {"layers": 16, "physical_parent_sha256": "7abd"},
    }
    message = detail(observed, audited)
    # only the differing leaves, and the invariant ones stay out of it
    assert "layers" not in message
    assert "padded_slots_per_scan: observed 126 against audited 82" in message
    assert "critical_path: observed 16 against audited 12" in message
    assert "physical_parent_sha256: observed '101c' against audited '7abd'" in message
    assert message.startswith("gdn{")


# --------------------------------------------------------------------------- #
# BOOT NINE: the composite signature, and closing the flush path wholesale     #
# --------------------------------------------------------------------------- #
# Boot nine reached health at 307s, acked generation zero, PASSED the structure
# audit the previous landing fixed, and died ~50 lines later:
#
#     RuntimeError('FR13 fixed32 live forward structural signature drift')
#
# THE MOST ONE-SIDED REFUSAL IN THE CAMPAIGN. No observed, no audited, not even
# the two hashes. The corpse cannot say what differed -- I went looking for the
# stored signature to CPU-verify against and there is none, because the guard
# recorded nothing. A guard that testifies nothing is worse than no guard: it
# stops the run AND withholds the reason.
#
# THE CAUSE: forward_graph_structural_signature had no `mode` parameter at all,
# so it hashed hydra27's manifest for every profile. Not a stale literal -- a
# MISSING ARGUMENT. That is a fourth tripwire, and it is the sharpest one yet
# for this class: enumerate every call INTO the authority that omits the mode
# the authority accepts. It found both call sites at once, including the one in
# _fr13_fixed32_observed_graph_replay that would have killed boot ten.
FLUSH_PATH_AUTHORITY_CALLS = {
    "forward_graph_structural_manifest": "accepts mode; both call sites pass it",
    "forward_graph_structural_signature": (
        "accepts mode as of this landing; both call sites pass it"
    ),
}


def test_the_signature_function_takes_the_served_mode() -> None:
    """MUTATION PROOF for boot nine's cause."""
    import inspect

    import fr13_fixed32_work_census as census

    assert "mode" in inspect.signature(
        census.forward_graph_structural_signature
    ).parameters
    hydra31 = census.forward_graph_structural_signature(1, mode="hydra31_fixed32")
    era = census.forward_graph_structural_signature(1)
    assert hydra31 != era, "the profiles must genuinely differ"
    for mode in ("hydra27_fixed32", "tail6_fixed32"):
        assert census.forward_graph_structural_signature(1, mode=mode) == era


def test_the_signature_is_the_hash_of_the_manifest_and_nothing_else() -> None:
    """THE CLOSED LOOP that replaces a CPU-vs-corpse comparison.

    The corpse stored no signature, so there is nothing to compare against
    directly. But the manifest is the signature's ONLY input, and that manifest
    was verified field-by-field against boot EIGHT's corpse -- which did print
    both sides. So verifying sha256(manifest) == signature() closes the loop.
    """
    import hashlib
    import json

    import fr13_fixed32_work_census as census

    manifest = census.forward_graph_structural_manifest(1, mode="hydra31_fixed32")
    # the values boot eight observed live, carried into what gets hashed
    assert manifest["gdn"]["padded_slots_per_scan"] == 126
    assert manifest["gdn"]["critical_path"] == 16
    assert manifest["gdn"]["max_path_lengths"] == [5, 11]
    assert manifest["tree_attention"]["physical_parent_sha256"].startswith(
        "101c590e58"
    )
    digest = hashlib.sha256(
        json.dumps(
            manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    ).hexdigest()
    assert digest == census.forward_graph_structural_signature(
        1, mode="hydra31_fixed32"
    )


def test_every_signature_the_flush_path_can_ask_for_resolves() -> None:
    """WHOLESALE, not one boot at a time: all 32 combinations."""
    import fr13_fixed32_work_census as census

    topology = _topology()
    modes = list(topology.SERVING_MODES) + [None]
    shapes = (census.UNFUSED_KERNEL_SHAPE, census.FUSED_KERNEL_SHAPE)
    seen = 0
    for mode in modes:
        for shape in shapes:
            for batch in (1, 2, 3, 4):
                digest = census.forward_graph_structural_signature(
                    batch, kernel_shape=shape, mode=mode
                )
                assert len(digest) == 64
                seen += 1
    assert seen == 32


def test_the_pinned_signature_table_is_per_mode_and_still_binds() -> None:
    import fr13_fixed32_work_census as census

    by_mode = census.FORWARD_GRAPH_STRUCTURAL_SIGNATURES_BY_MODE
    assert set(by_mode) == set(_topology().SERVING_MODES)
    # tail6 and hydra27 share the era table BY REFERENCE, not a retyped copy
    assert by_mode["tail6_fixed32"] is by_mode["hydra27_fixed32"]
    assert by_mode["tail6_fixed32"] is census.FORWARD_GRAPH_STRUCTURAL_SIGNATURES
    assert (
        by_mode["hydra31_fixed32"]
        is census.FORWARD_GRAPH_STRUCTURAL_SIGNATURES_HYDRA31
    )
    # every hydra31 entry is RECOMPUTED here, not trusted
    import hashlib
    import json

    for shape, rows in census.FORWARD_GRAPH_STRUCTURAL_SIGNATURES_HYDRA31.items():
        for batch, pinned in rows.items():
            manifest = census.forward_graph_structural_manifest(
                batch, kernel_shape=shape, mode="hydra31_fixed32"
            )
            assert (
                hashlib.sha256(
                    json.dumps(
                        manifest,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                ).hexdigest()
                == pinned
            ), (shape, batch)
    # NOT ACCEPT-ANY: a wrong pin still refuses, and names both digests.
    saved = census.FORWARD_GRAPH_STRUCTURAL_SIGNATURES_HYDRA31[
        census.UNFUSED_KERNEL_SHAPE
    ][1]
    census.FORWARD_GRAPH_STRUCTURAL_SIGNATURES_HYDRA31[
        census.UNFUSED_KERNEL_SHAPE
    ][1] = "0" * 64
    try:
        with pytest.raises(census.CensusError) as refusal:
            census.forward_graph_structural_signature(1, mode="hydra31_fixed32")
    finally:
        census.FORWARD_GRAPH_STRUCTURAL_SIGNATURES_HYDRA31[
            census.UNFUSED_KERNEL_SHAPE
        ][1] = saved
    message = str(refusal.value)
    assert "drifted from its pinned signature" in message
    assert "mode='hydra31_fixed32'" in message
    assert "computed=" in message and "pinned=" in message


def test_the_signature_refusal_now_testifies() -> None:
    """Both hashes AND the structural fields that fed each."""
    blob = _planted_blob()
    assert (
        '"FR13 fixed32 live forward structural signature drift"' not in blob
    ), "the bare-string refusal survived"
    assert "live forward structural signature drift for mode " in blob
    assert "+ repr(live_structural_signature)" in blob
    assert "+ repr(_audited_structural_signature)" in blob
    assert "_fr13_fixed32_drift_detail(structural, expected)" in blob


def test_no_authority_call_on_the_flush_path_omits_the_mode() -> None:
    """THE FOURTH TRIPWIRE, as a standing check.

    Every call from the planted blob into a work_census function that ACCEPTS a
    mode must pass one. This is what found boot ten's guard before boot ten.
    """
    import inspect

    import fr13_fixed32_work_census as census

    blob = _planted_blob()
    tree = ast.parse(blob)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == (
            "fr13_fixed32_work_census"
        ):
            imported |= {alias.name for alias in node.names}
    assert imported, "the blob no longer imports the census authority"
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id not in imported:
            continue
        target = getattr(census, node.func.id, None)
        if target is None:
            continue
        try:
            accepts = "mode" in inspect.signature(target).parameters
        except (TypeError, ValueError):
            continue
        if accepts and "mode" not in {kw.arg for kw in node.keywords}:
            offenders.append((node.lineno, node.func.id))
    assert not offenders, (
        "authority calls on the flush path that omit the served mode: "
        + repr(offenders)
    )
    for name in imported:
        assert name in FLUSH_PATH_AUTHORITY_CALLS, name


def test_the_downstream_flush_path_carries_no_other_stale_expectation() -> None:
    """ORDER: close that section wholesale, not one boot at a time.

    All three value/digest tripwires over every guard downstream of the audits,
    from the forward-graph registry to the end of the blob. The single value
    hit is the conv state dimension already classified as coincidence, and
    there is no pinned digest literal at all.
    """
    import re as _re

    blob = _planted_blob()
    tree = ast.parse(blob)
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(line, node.name)
    start = next(
        node.lineno
        for node in ast.parse(blob).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_forward_graph_registry"
    )
    high_signal = {12, 7, 82, 6, 8, 11, 13, 17, 27, 36, 24, 126}
    value_hits: dict[str, list[int]] = {}
    digest_hits: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or node.lineno < start:
            continue
        if not any(isinstance(c, ast.Raise) for c in ast.walk(node)):
            continue
        scope = owner.get(node.lineno, "<module>")
        values = sorted(
            c.value
            for c in ast.walk(node.test)
            if isinstance(c, ast.Constant)
            and isinstance(c.value, int)
            and not isinstance(c.value, bool)
            and c.value in high_signal
        )
        if values:
            value_hits.setdefault(scope, []).extend(values)
        if any(
            isinstance(c, ast.Constant)
            and isinstance(c.value, str)
            and _re.fullmatch(r"[0-9a-f]{64}", c.value or "")
            for c in ast.walk(node.test)
        ):
            digest_hits.setdefault(scope, []).append(node.lineno)
    assert set(value_hits) <= {"_fr13_fixed32_observed_commit"}, value_hits
    assert not digest_hits, digest_hits
