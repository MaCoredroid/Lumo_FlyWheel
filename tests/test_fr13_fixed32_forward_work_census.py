from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"

TREE_LAYERS = frozenset(
    "language_model.model.layers.%d.self_attn.attn" % layer
    for layer in range(3, 64, 4)
)
SFWD_LAYERS = frozenset(
    "language_model.model.layers.%d.linear_attn" % layer for layer in range(48)
)


def _census_runtime() -> dict[str, object]:
    """Exec the census validator out of the embedded runtime source."""
    source = PATCHER.read_text(encoding="utf-8")
    runtime = next(
        node.value.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
        and isinstance(node.value, ast.Constant)
    )
    wanted = {
        "_fr13_fixed32_validate_forward_work",
        "_fr13_fixed32_observed_sfwd_conv_postprep",
    }
    definitions = [
        node
        for node in ast.parse(runtime).body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace: dict[str, object] = {
        "_FR13_FIXED32_TARGET_TREE_LAYERS": TREE_LAYERS,
        "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION": False,
    }
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]), "<census>", "exec"
        ),
        namespace,
    )
    return namespace


def _work(*, fused: bool, batch: int = 1) -> dict[str, object]:
    """The census a real B1 FULL capture publishes.

    Fused values are the ones observed in the 2026-08-08 boot screen at
    f4591891c (container log, "captured forward work is incomplete"): every
    conv_* counter sits at its initial value because the fusion replaces the
    pregather stage kernel and the 48 per-layer consumes with one kernel per
    layer.
    """
    work: dict[str, object] = {
        "batch_size": batch,
        "tree_calls": 16,
        "tree_layers": set(TREE_LAYERS),
        "tree_q_rows": 16 * batch * 32,
        "tree_bias_shape": (32, 32),
        "gdn_scan_calls": 48 * batch,
        "gdn_calls": {index: 1 for index in range(48 * batch)},
        "gdn_layers": set(range(48)),
        "gdn_launches": 48 * batch * 2,
        "gdn_path_programs": 48 * batch * 12,
        "gdn_padded_slots": 48 * batch * 82,
        "gdn_nodes": 48 * batch * 32,
        "gdn_critical_path": 12,
        "gdn_grid_z": (1, 11),
        "gdn_max_path_lengths": (5, 7),
        "gdn_export_or_mask": 16915,
    }
    if fused:
        work.update(
            conv_consume_calls=0,
            conv_consume_layers=set(),
            conv_consume_hits=0,
            conv_consume_fallbacks=0,
            conv_freshness_matches=0,
            conv_stage_calls=0,
            conv_stage_replays=0,
            conv_stage_before_all_consumes=False,
            conv_stage_layers=0,
            conv_stage_programs=0,
            conv_stage_ssi_pointer_entries=0,
            conv_stage_ssi_groups=0,
            conv_stage_row_elems=0,
            conv_stage_block=0,
            conv_stage_layer=None,
            conv_stage_source=None,
            conv_stage_instance=None,
            conv_source_layers={},
            sfwd_conv_postprep_calls=48,
            sfwd_conv_postprep_layers=set(SFWD_LAYERS),
        )
    else:
        names = {index: "layer.%d" % index for index in range(48)}
        work.update(
            conv_consume_calls=48,
            conv_consume_layers=set(names.values()),
            conv_consume_hits=48,
            conv_consume_fallbacks=0,
            conv_freshness_matches=48,
            conv_stage_calls=1,
            conv_stage_replays=0,
            conv_stage_before_all_consumes=True,
            conv_stage_layers=48,
            conv_stage_programs=48 * batch,
            conv_stage_ssi_pointer_entries=48,
            conv_stage_ssi_groups=3,
            conv_stage_row_elems=1024,
            conv_stage_block=1024,
            conv_stage_layer="layer.0",
            conv_stage_source="a" * 64,
            conv_stage_instance="b" * 64,
            conv_source_layers=names,
            sfwd_conv_postprep_calls=0,
            sfwd_conv_postprep_layers=set(),
        )
    return work


@pytest.mark.parametrize("batch", (1, 4))
def test_census_accepts_the_unfused_capture(batch: int) -> None:
    runtime = _census_runtime()
    runtime["_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION"] = False
    runtime["_fr13_fixed32_validate_forward_work"](
        _work(fused=False, batch=batch), "captured"
    )


@pytest.mark.parametrize("batch", (1, 4))
def test_census_accepts_the_fused_capture(batch: int) -> None:
    """Regression for the 2026-08-08 boot screen at f4591891c."""
    runtime = _census_runtime()
    runtime["_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION"] = True
    runtime["_fr13_fixed32_validate_forward_work"](
        _work(fused=True, batch=batch), "captured"
    )


def test_fused_census_still_proves_every_layer_reached_the_graph() -> None:
    """Relaxing conv_* must not become a hole in the census.

    The fused class is the replacement proof: a kernel missing from the
    captured graph has to still fail the census.
    """
    runtime = _census_runtime()
    runtime["_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION"] = True
    validate = runtime["_fr13_fixed32_validate_forward_work"]
    for missing in (1, 48):
        work = _work(fused=True)
        kept = sorted(SFWD_LAYERS)[: 48 - missing]
        work["sfwd_conv_postprep_calls"] = 48 - missing
        work["sfwd_conv_postprep_layers"] = set(kept)
        with pytest.raises(RuntimeError, match="forward work is incomplete"):
            validate(work, "captured")


def test_each_shape_rejects_the_other_shapes_conv_counters() -> None:
    """Fusion off must not accept a census with no conv work, and vice versa."""
    runtime = _census_runtime()
    validate = runtime["_fr13_fixed32_validate_forward_work"]
    runtime["_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION"] = False
    with pytest.raises(RuntimeError, match="forward work is incomplete"):
        validate(_work(fused=True), "captured")
    runtime["_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION"] = True
    with pytest.raises(RuntimeError, match="forward work is incomplete"):
        validate(_work(fused=False), "captured")


def test_sfwd_observer_counts_each_layer_once() -> None:
    runtime = _census_runtime()
    observe = runtime["_fr13_fixed32_observed_sfwd_conv_postprep"]
    event = {
        "batch_size": 1,
        "sfwd_conv_postprep_layers": set(),
        "sfwd_conv_postprep_calls": 0,
        "conv_stage_calls": 0,
        "conv_consume_calls": 0,
    }
    runtime["_fr13_fixed32_observed_work_target"] = (
        lambda label, capturing, batch: (event, True)
    )
    for name in sorted(SFWD_LAYERS):
        observe(name, 1, True)
    assert event["sfwd_conv_postprep_calls"] == 48
    assert event["sfwd_conv_postprep_layers"] == set(SFWD_LAYERS)
    # A layer reported twice into one event is a census defect, not a no-op.
    with pytest.raises(RuntimeError, match="SFWD conv/post-prep census drift"):
        observe(sorted(SFWD_LAYERS)[0], 1, True)


def test_sfwd_observer_rejects_mixing_with_the_unfused_conv_path() -> None:
    """The two paths are exclusive; both counting would double-count the work."""
    runtime = _census_runtime()
    observe = runtime["_fr13_fixed32_observed_sfwd_conv_postprep"]
    for conflicting in ("conv_stage_calls", "conv_consume_calls"):
        event = {
            "batch_size": 1,
            "sfwd_conv_postprep_layers": set(),
            "sfwd_conv_postprep_calls": 0,
            "conv_stage_calls": 0,
            "conv_consume_calls": 0,
        }
        event[conflicting] = 1
        runtime["_fr13_fixed32_observed_work_target"] = (
            lambda label, capturing, batch: (event, True)
        )
        with pytest.raises(RuntimeError, match="SFWD conv/post-prep census drift"):
            observe("language_model.model.layers.0.linear_attn", 1, True)


def _flush_pregather_predicate():
    """Wrap the flush block's real conv-pregather predicate as a callable.

    The check lives inside the `fixed_flush` source the patcher injects, in a
    no-argument function that reads whole-module state, so slice out the
    predicate text itself and drive it directly.
    """
    source = PATCHER.read_text(encoding="utf-8")
    fragment = next(
        node.value.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "fixed_flush"
        and isinstance(node.value, ast.Constant)
    )
    lines = fragment.splitlines()
    start = next(
        i for i, line in enumerate(lines) if "_fr13_f32_flush_fused = bool(" in line
    )
    end = next(
        i
        for i, line in enumerate(lines)
        if i > start and "conv pregather counters mismatch" in line
    )
    # Take the block up to and including the raise's closing repr line.
    tail = next(
        i for i, line in enumerate(lines) if i > end and line.strip() == ")"
    )
    body = "\n".join(lines[start : tail + 1])
    body = body.replace(
        'getattr(gdn, "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION", False)', "fused"
    )
    # The raise reports a sibling local from the enclosing flush scope.
    wrapper = (
        "def _predicate(pc, preseed_cap, fused):\n"
        "    expected_pregather_replays_by_batch = {}\n"
        + textwrap.indent(textwrap.dedent(body), "    ")
    )
    namespace: dict[str, object] = {}
    exec(wrapper, namespace)
    return namespace["_predicate"]


def _pregather_counters(*, fused: bool, preseed_cap: int = 1) -> dict[str, object]:
    """Counters a real arm publishes. Fused values are the observed ones."""
    capture = 0 if fused else preseed_cap
    return {
        "preseeded": True,
        "pointer_entries": 48,
        "preseeded_batches": tuple(range(1, preseed_cap + 1)),
        "max_batch_size": preseed_cap,
        "actual_stages": 0,
        "actual_stages_by_batch": {batch: 0 for batch in (1, 2, 3, 4)},
        "graph_capture_stages": capture,
        "graph_capture_stages_by_batch": {
            batch: 0 if fused else (1 if batch <= preseed_cap else 0)
            for batch in (1, 2, 3, 4)
        },
        "profile_capture_stages": 0,
        "aux_capture_stages": 0,
    }


@pytest.mark.parametrize("preseed_cap", (1, 4))
@pytest.mark.parametrize("fused", (False, True))
def test_flush_pregather_accepts_its_own_shape(
    fused: bool, preseed_cap: int
) -> None:
    """Regression for the 2026-08-08 candidate arm at 569e97a28.

    The flush demanded graph_capture_stages == preseed_cap, but the fusion
    subsumes the pregather stage kernel so nothing launches at capture. The
    observed counters were exactly the fused set below.
    """
    predicate = _flush_pregather_predicate()
    predicate(
        _pregather_counters(fused=fused, preseed_cap=preseed_cap),
        preseed_cap,
        fused,
    )


@pytest.mark.parametrize("fused", (False, True))
def test_flush_pregather_rejects_the_other_shape(fused: bool) -> None:
    """Shapes stay mutually exclusive: neither may accept the other's counts."""
    predicate = _flush_pregather_predicate()
    with pytest.raises(RuntimeError, match="conv pregather counters mismatch"):
        predicate(_pregather_counters(fused=not fused), 1, fused)


def test_flush_pregather_still_rejects_live_stage_launches() -> None:
    """Relaxing capture stages must not relax the live-replay stage floor."""
    predicate = _flush_pregather_predicate()
    counters = _pregather_counters(fused=True)
    counters["actual_stages"] = 1
    counters["actual_stages_by_batch"] = {1: 1, 2: 0, 3: 0, 4: 0}
    with pytest.raises(RuntimeError, match="conv pregather counters mismatch"):
        predicate(counters, 1, True)


def test_flush_pregather_message_names_the_shape() -> None:
    predicate = _flush_pregather_predicate()
    with pytest.raises(RuntimeError, match=r"mismatch \(fused\)"):
        predicate(_pregather_counters(fused=False), 1, True)
    with pytest.raises(RuntimeError, match=r"mismatch \(unfused\)"):
        predicate(_pregather_counters(fused=True), 1, False)


def test_sfwd_observer_is_inert_without_an_active_event() -> None:
    runtime = _census_runtime()
    runtime["_fr13_fixed32_observed_work_target"] = (
        lambda label, capturing, batch: (None, False)
    )
    runtime["_fr13_fixed32_observed_sfwd_conv_postprep"](
        "language_model.model.layers.0.linear_attn", 1, False
    )


# --- canonical structural references -------------------------------------
#
# Two pinned references, one per kernel shape. The stock arm must keep the
# reference it has always had; the fused arm carries its own.

import sys as _sys  # noqa: E402

if str(ROOT / "scripts") not in _sys.path:
    _sys.path.insert(0, str(ROOT / "scripts"))
import fr13_fixed32_work_census as census  # noqa: E402


UNFUSED_SIGNATURES = {
    1: "2373bfbd2ac6ab7a6fd67af5570385f2aea2a16a1e80b804bdf12e092f319423",
    2: "508a856a418e5954083e8aaf93efa1e6f89b65562f3c20414418b9dd640e5362",
    3: "f451f42fc2803a8a3a7d7359e39487ba944fc27618a043d5026d766f2e94cba7",
    4: "025bc236c194ee88a512ccb633b0247cfa3e4a15e17975061083b62d7be921cb",
}


@pytest.mark.parametrize("batch", (1, 2, 3, 4))
def test_unfused_structural_signature_is_unchanged(batch: int) -> None:
    """The stock arm's reference must stay byte-identical forever.

    These are the values the module produced before the fused shape existed.
    """
    assert census.forward_graph_structural_signature(batch) == UNFUSED_SIGNATURES[batch]
    assert (
        census.forward_graph_structural_signature(
            batch, kernel_shape=census.UNFUSED_KERNEL_SHAPE
        )
        == UNFUSED_SIGNATURES[batch]
    )
    manifest = census.forward_graph_structural_manifest(batch)
    assert manifest["schema"] == census.STRUCTURAL_MANIFEST_SCHEMA
    assert "conv_pregather" in manifest
    assert "sfwd_conv_postprep" not in manifest


@pytest.mark.parametrize("batch", (1, 2, 3, 4))
def test_fused_reference_is_pinned_and_distinct(batch: int) -> None:
    fused = census.forward_graph_structural_signature(
        batch, kernel_shape=census.FUSED_KERNEL_SHAPE
    )
    assert fused == census.FORWARD_GRAPH_STRUCTURAL_SIGNATURES[
        census.FUSED_KERNEL_SHAPE
    ][batch]
    # An arm matches exactly one reference.
    assert fused != UNFUSED_SIGNATURES[batch]
    manifest = census.forward_graph_structural_manifest(
        batch, kernel_shape=census.FUSED_KERNEL_SHAPE
    )
    assert manifest["schema"] == census.STRUCTURAL_MANIFEST_FUSED_SCHEMA
    assert "conv_pregather" not in manifest
    assert manifest["sfwd_conv_postprep"]["calls"] == 48
    # The fused route must pin the subsumed unfused work at zero, so a run
    # that took both routes cannot match this reference.
    for subsumed in (
        "stage_calls",
        "consume_calls",
        "source_validations",
        "freshness_matches",
        "staged_rows",
    ):
        assert manifest["sfwd_conv_postprep"][subsumed] == 0


@pytest.mark.parametrize("batch", (1, 2, 3, 4))
def test_workload_identity_is_shared_across_shapes(batch: int) -> None:
    """What makes the pair comparable stays strict and identical.

    Kernel structure is what the candidate is allowed to change; the workload
    -- draft counts, verify rows, tree geometry, GDN scan shape -- is not.
    """
    unfused = census.forward_graph_structural_manifest(batch)
    fused = census.forward_graph_structural_manifest(
        batch, kernel_shape=census.FUSED_KERNEL_SHAPE
    )
    for shared in ("batch_size", "descriptor_geometry", "tree_attention", "gdn"):
        assert unfused[shared] == fused[shared], shared


def test_signature_rejects_a_manifest_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canonical references are written down, not derived.

    Editing a manifest without updating its pin must fail loudly rather than
    silently reclassify an arm.
    """
    original = census.forward_graph_structural_manifest

    def tampered(batch_size, *, kernel_shape=census.UNFUSED_KERNEL_SHAPE):
        manifest = dict(original(batch_size, kernel_shape=kernel_shape))
        manifest["batch_size"] = manifest["batch_size"] + 100
        return manifest

    monkeypatch.setattr(census, "forward_graph_structural_manifest", tampered)
    for shape in census.KERNEL_SHAPES:
        with pytest.raises(census.CensusError, match="drifted from its pinned"):
            census.forward_graph_structural_signature(1, kernel_shape=shape)


def test_unknown_kernel_shape_is_rejected() -> None:
    for bad in ("fused", "", "CONV_PREGATHER", None):
        with pytest.raises(ValueError, match="kernel_shape"):
            census.forward_graph_structural_manifest(1, kernel_shape=bad)
        with pytest.raises(ValueError, match="kernel_shape"):
            census.forward_graph_structural_signature(1, kernel_shape=bad)


# --- replay-side restore (the fabrication gap) ---------------------------


def _replay_runtime() -> dict[str, object]:
    """Exec the pieces of the replay restore we can drive directly."""
    source = PATCHER.read_text(encoding="utf-8")
    runtime = next(
        node.value.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
        and isinstance(node.value, ast.Constant)
    )
    wanted = {
        "_fr13_fixed32_kernel_shape",
        "_fr13_fixed32_pregather_capture_expectation",
    }
    definitions = [
        node
        for node in ast.parse(runtime).body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace: dict[str, object] = {
        "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION": False
    }
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), "<rt>", "exec"),
        namespace,
    )
    return namespace


def test_kernel_shape_is_a_closed_enumerated_set() -> None:
    runtime = _replay_runtime()
    shape = runtime["_fr13_fixed32_kernel_shape"]
    assert shape() == "unfused_conv_pregather"
    runtime["_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION"] = True
    assert shape() == "sfwd_fused_conv_postprep"
    # The runtime vocabulary and the census vocabulary must be one set.
    assert set(census.KERNEL_SHAPES) == {
        "unfused_conv_pregather",
        "sfwd_fused_conv_postprep",
    }


@pytest.mark.parametrize("capacity", (1, 4))
def test_pregather_capture_expectation_follows_the_shape(capacity: int) -> None:
    runtime = _replay_runtime()
    expectation = runtime["_fr13_fixed32_pregather_capture_expectation"]
    stages, by_batch = expectation(capacity)
    assert stages == capacity
    assert by_batch == {b: 1 if b <= capacity else 0 for b in (1, 2, 3, 4)}
    runtime["_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION"] = True
    stages, by_batch = expectation(capacity)
    assert stages == 0
    assert by_batch == {b: 0 for b in (1, 2, 3, 4)}


def test_replay_restore_never_fabricates_unfused_conv_work() -> None:
    """The worst failure mode in an evidence chain is invented counters.

    The replay path used to restore conv_stage_calls/consume_calls and a
    staging layout digest from the manifest unconditionally. Under fusion no
    kernel in the arm performs that work, so a fused replay would have carried
    48 consumes and a staging digest that never happened. Assert the fused
    branch restores the fused class and leaves every subsumed counter at zero.
    """
    source = PATCHER.read_text(encoding="utf-8")
    runtime = next(
        node.value.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
        and isinstance(node.value, ast.Constant)
    )
    replay = next(
        ast.get_source_segment(runtime, node)
        for node in ast.parse(runtime).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_observed_graph_replay"
    )
    assert replay is not None
    # Anchor on the restore block (the last fused branch), not the earlier
    # provenance one.
    fused_index = replay.rindex("if replay_fused:")
    unfused_index = replay.index("    else:", fused_index)
    fused_block = replay[fused_index:unfused_index]
    # The fused branch restores the fused class...
    assert 'event["sfwd_conv_postprep_calls"] = int(sfwd_section["calls"])' in fused_block
    assert 'event["sfwd_conv_postprep_layers"] = set(' in fused_block
    # ...and pins every subsumed counter at zero rather than reading the
    # unfused manifest section.
    for zeroed in (
        'event["conv_stage_replays"] = 0',
        'event["conv_stage_before_all_consumes"] = False',
        'event["conv_stage_layer"] = None',
        'event["conv_stage_layers"] = 0',
        'event["conv_stage_row_elems"] = 0',
        'event["conv_stage_block"] = 0',
        'event["conv_stage_programs"] = 0',
        'event["conv_stage_ssi_pointer_entries"] = 0',
        'event["conv_stage_ssi_groups"] = 0',
        'event["conv_stage_source"] = None',
        'event["conv_stage_instance"] = None',
        'event["conv_source_layers"] = {}',
        'event["conv_consume_layers"] = set()',
        'event["conv_consume_hits"] = 0',
        'event["conv_consume_fallbacks"] = 0',
    ):
        assert zeroed in fused_block, zeroed
    # The fused branch must never read the unfused section.
    assert 'conv["' not in fused_block
    # And the unfused branch must be unchanged in what it restores.
    unfused_block = replay[unfused_index:]
    for restored in (
        'event["conv_stage_calls"] = int(conv["stage_calls"])',
        'event["conv_stage_replays"] = 1',
        'event["conv_stage_source"] = conv["layout_sha256"]',
        'event["conv_consume_calls"] = int(conv["consume_calls"])',
    ):
        assert restored in unfused_block, restored


def test_replay_evidence_records_the_kernel_shape() -> None:
    """Constraint 3: the difference is explicit in the evidence, never silent."""
    source = PATCHER.read_text(encoding="utf-8")
    assert '"kernel_shape": replay_shape,' in source
    assert '"kernel_shape": registry_shape,' in source
    assert '"kernel_shape": _fr13_fixed32_kernel_shape(),' in source
    assert '"conv_pregather_stage_replays": 0 if replay_fused else 1,' in source


# --- attested kernel_shape (report schema v13) ---------------------------


def test_report_schema_family_is_v13() -> None:
    for schema in (
        census.REPORT_SCHEMA,
        census.ARM_REPORT_SCHEMA,
        census.SELF_TEST_SCHEMA,
    ):
        assert schema.endswith("-v13")
    # The per-event schema is a separate family and must not have moved.
    assert census.SCHEMA == "fr13-fixed32-work-census-v12"


@pytest.mark.parametrize("batch", (1, 2, 3, 4))
def test_kernel_shape_is_derived_from_the_signature(batch: int) -> None:
    for shape in census.KERNEL_SHAPES:
        signature = census.forward_graph_structural_signature(
            batch, kernel_shape=shape
        )
        assert census.kernel_shape_for_signature(signature) == shape
        assert (
            census.assert_kernel_shape_attested(shape, signature, source="r")
            == shape
        )


def test_a_report_cannot_claim_a_shape_its_signature_disproves() -> None:
    fused = census.forward_graph_structural_signature(
        1, kernel_shape=census.FUSED_KERNEL_SHAPE
    )
    unfused = census.forward_graph_structural_signature(1)
    with pytest.raises(census.CensusError, match="but its graph_signature attests"):
        census.assert_kernel_shape_attested(
            census.UNFUSED_KERNEL_SHAPE, fused, source="r"
        )
    with pytest.raises(census.CensusError, match="but its graph_signature attests"):
        census.assert_kernel_shape_attested(
            census.FUSED_KERNEL_SHAPE, unfused, source="r"
        )


def test_unknown_signatures_and_shapes_fail_closed() -> None:
    for bogus in ("0" * 64, "", "not-a-signature"):
        with pytest.raises(census.CensusError, match="matches no pinned canonical"):
            census.kernel_shape_for_signature(bogus)
    # A declared value outside the closed set can never be attested.
    fused = census.forward_graph_structural_signature(
        1, kernel_shape=census.FUSED_KERNEL_SHAPE
    )
    for bogus in ("fused", "unknown", None, "", "SFWD_FUSED_CONV_POSTPREP"):
        with pytest.raises(census.CensusError, match="but its graph_signature attests"):
            census.assert_kernel_shape_attested(bogus, fused, source="r")


def test_v12_rows_without_the_shape_trio_still_validate() -> None:
    """Constraint 4: recorded evidence predates the field and is untouched."""
    trio = census.FORWARD_GRAPH_REGISTRY_KERNEL_SHAPE_KEYS
    assert trio == {"kernel_shape", "fused_calls", "fused_layers"}
    # The trio is not part of the required key set, so a v12-shaped row is
    # exactly as valid as it was.
    assert not (trio & census.FORWARD_GRAPH_REGISTRY_KEYS)


@pytest.mark.parametrize(
    "gate", ("scripts/fr13_floor_gate.py", "scripts/fr13_depth_acceptance.py")
)
def test_both_gates_attest_the_shape_and_expect_per_shape_rows(gate: str) -> None:
    source = (ROOT / gate).read_text(encoding="utf-8")
    # Attested, not declared.
    assert "assert_kernel_shape_attested(" in source
    assert 'row["kernel_shape"],' in source
    # Per-shape canonical reference.
    assert "kernel_shape=row_shape" in source
    # The fused row pins the subsumed unfused work at zero, so an arm that
    # took both routes satisfies neither shape.
    for zeroed in (
        '"stage_calls": 0 if row_fused else 1',
        '"consume_calls": 0 if row_fused else CONV_PREGATHER_LAYERS',
        '"source_validations": 0 if row_fused else CONV_PREGATHER_LAYERS',
        '"staged_rows": 0 if row_fused else CONV_PREGATHER_LAYERS * batch',
    ):
        assert zeroed in source, (gate, zeroed)
    assert '"fused_calls": CONV_PREGATHER_LAYERS if row_fused else 0' in source
    assert '"fused_layers": CONV_PREGATHER_LAYERS if row_fused else 0' in source


# --- both shapes through the fixtures and both v5 gate twins -------------


import importlib.util as _importlib_util  # noqa: E402
import json as _json  # noqa: E402


def _load_gate(name: str, relative: str):
    spec = _importlib_util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FLOOR_GATE = _load_gate("fr13_forward_census_floor_gate", "scripts/fr13_floor_gate.py")
DEPTH_GATE = _load_gate(
    "fr13_forward_census_depth_acceptance", "scripts/fr13_depth_acceptance.py"
)


def _arm_records(mode: str, *, events: int = 3, kernel_shape: str | None = None):
    records = [
        census.reference_event(
            mode,
            1,
            f"{mode}:shape-fixture:{index}",
            event_index=index,
            forward_step_index=index,
            request_ids=[f"req-{mode}-{index}"],
        )
        for index in range(events)
    ]
    terminal = census.reference_terminal_summary(
        records,
        fixture_synthetic_runtime_proof=True,
        **({} if kernel_shape is None else {"kernel_shape": kernel_shape}),
    )
    return [*records, terminal]


def _campaign_report():
    tail = _arm_records(census.TAIL_MODE)
    hydra = _arm_records(census.HYDRA_MODE)
    return census.validate_campaign(
        [(record, f"tail:{index}") for index, record in enumerate(tail)],
        [(record, f"hydra:{index}") for index, record in enumerate(hydra)],
        required_batches=(1,),
    )


def _fused_row(row: dict) -> dict:
    return {
        **row,
        "graph_signature": census.forward_graph_structural_signature(
            row["batch_size"], kernel_shape=census.FUSED_KERNEL_SHAPE
        ),
        "conv_layout_sha256": None,
        "kernel_shape": census.FUSED_KERNEL_SHAPE,
        "fused_calls": census.CONV_PREGATHER_LAYERS,
        "fused_layers": census.CONV_PREGATHER_LAYERS,
        "stage_calls": 0,
        "stage_before_all_consumes": False,
        "row_elems": 0,
        "programs": 0,
        "ssi_pointer_entries": 0,
        "ssi_groups": 0,
        "source_validations": 0,
        "staged_rows": 0,
        "consume_calls": 0,
        "consume_hits": 0,
        "freshness_matches": 0,
    }


def _fused_report(mutate=None, modes=(census.TAIL_MODE, census.HYDRA_MODE)):
    report = _json.loads(_json.dumps(_campaign_report()))
    for mode in modes:
        rows = [
            _fused_row(row)
            for row in report["forward_graph_registries"][mode]
        ]
        if mutate is not None:
            for row in rows:
                mutate(row)
        report["forward_graph_registries"][mode] = rows
        report["terminal_summaries"][mode]["forward_graph_registry"] = (
            _json.loads(_json.dumps(rows))
        )
    return report


def test_the_census_fixture_registry_records_its_own_shape() -> None:
    """The fixture the gates' self-tests build carries the attested trio."""
    for shape in census.KERNEL_SHAPES:
        events = [
            census.reference_event(
                census.TAIL_MODE,
                1,
                f"tail:shape:{index}",
                event_index=index,
                forward_step_index=index,
                request_ids=[f"req-{index}"],
            )
            for index in range(2)
        ]
        row = census.reference_terminal_summary(
            events,
            fixture_synthetic_runtime_proof=True,
            kernel_shape=shape,
        )["forward_graph_registry"][0]
        assert census.FORWARD_GRAPH_REGISTRY_KERNEL_SHAPE_KEYS <= set(row)
        assert (
            census.assert_kernel_shape_attested(
                row["kernel_shape"], row["graph_signature"], source="fixture"
            )
            == shape
        )
        fused = shape == census.FUSED_KERNEL_SHAPE
        assert row["fused_calls"] == (
            census.CONV_PREGATHER_LAYERS if fused else 0
        )
        assert row["stage_calls"] == (0 if fused else 1)
        assert row["staged_rows"] == (
            0 if fused else census.CONV_PREGATHER_LAYERS
        )
        # The fused route publishes no staging layout digest.
        assert (row["conv_layout_sha256"] is None) is fused


def test_the_unfused_campaign_fixture_satisfies_both_v5_twins() -> None:
    """The fixture route feeds the v13 gates; this is what self-test runs."""
    report = _campaign_report()
    floor = FLOOR_GATE.validate_work_census_v5_report(report, required_batch=1)
    depth = DEPTH_GATE.validate_work_census_v5_report(report, required_batch=1)
    # Depth acceptance sha256-compares its summary against the floor gate's.
    assert floor == depth
    lifecycle = floor["forward_graph_pregather_lifecycle"]
    assert lifecycle["kernel_shape"] == census.UNFUSED_KERNEL_SHAPE
    assert lifecycle["stage_precedes_all_layer_consumes"]
    assert lifecycle["per_batch"]["1"]["stage_calls_per_capture"] == 1


def test_the_fused_registry_satisfies_both_v5_twins() -> None:
    report = _fused_report()
    floor = FLOOR_GATE.validate_work_census_v5_report(report, required_batch=1)
    depth = DEPTH_GATE.validate_work_census_v5_report(report, required_batch=1)
    assert floor == depth
    lifecycle = floor["forward_graph_pregather_lifecycle"]
    assert lifecycle["kernel_shape"] == census.FUSED_KERNEL_SHAPE
    # Nothing staged, so nothing is claimed about staging order or digests.
    assert not lifecycle["stage_precedes_all_layer_consumes"]
    assert not lifecycle["conv_layout_signatures_unique_within_each_arm"]
    assert not lifecycle["conv_layout_signatures_equal_across_arms_per_batch"]
    per_batch = lifecycle["per_batch"]["1"]
    assert per_batch["fused_calls"] == census.CONV_PREGATHER_LAYERS
    assert per_batch["fused_layers"] == census.CONV_PREGATHER_LAYERS
    assert per_batch["conv_layout_sha256"] is None
    for zeroed in (
        "stage_calls_per_capture",
        "row_elems",
        "programs",
        "ssi_pointer_entries",
        "ssi_groups",
        "source_validations",
        "staged_rows",
        "consume_calls",
        "consume_hits",
        "freshness_matches",
    ):
        assert per_batch[zeroed] == 0, zeroed
    assert per_batch["stage_before_all_consumes"] is False


@pytest.mark.parametrize(
    ("mutate", "needle"),
    (
        (
            lambda row: row.__setitem__(
                "kernel_shape", census.UNFUSED_KERNEL_SHAPE
            ),
            "but its graph_signature attests",
        ),
        (
            lambda row: row.__setitem__("conv_layout_sha256", "0" * 64),
            "identity is invalid",
        ),
        (
            lambda row: row.__setitem__("stage_calls", 1),
            "does not prove one ordered final-FULL pregather capture",
        ),
        (
            lambda row: row.__setitem__(
                "consume_calls", census.CONV_PREGATHER_LAYERS
            ),
            "does not prove one ordered final-FULL pregather capture",
        ),
        (
            lambda row: row.__setitem__(
                "fused_calls", census.CONV_PREGATHER_LAYERS - 1
            ),
            "does not prove one ordered final-FULL pregather capture",
        ),
    ),
)
def test_both_v5_twins_reject_a_fused_row_that_also_staged(
    mutate, needle: str
) -> None:
    """An arm that took both routes satisfies neither shape.

    The shape attestation is the census module's own, so it raises the
    census error in both gates; every other rejection is the gate's.
    """
    report = _fused_report(mutate)
    with pytest.raises(
        (FLOOR_GATE.GateError, census.CensusError), match=needle
    ):
        FLOOR_GATE.validate_work_census_v5_report(report, required_batch=1)
    with pytest.raises((ValueError, census.CensusError), match=needle):
        DEPTH_GATE.validate_work_census_v5_report(report, required_batch=1)


def test_both_v5_twins_reject_arms_that_ran_different_shapes() -> None:
    report = _fused_report(modes=(census.TAIL_MODE,))
    needle = "forward graph/layout signatures differ"
    with pytest.raises(FLOOR_GATE.GateError, match=needle):
        FLOOR_GATE.validate_work_census_v5_report(report, required_batch=1)
    with pytest.raises(ValueError, match=needle):
        DEPTH_GATE.validate_work_census_v5_report(report, required_batch=1)


def test_a_fused_terminal_needs_events_that_never_staged() -> None:
    """The per-event schema is a separate family, so this fails closed."""
    tail = _arm_records(
        census.TAIL_MODE, kernel_shape=census.FUSED_KERNEL_SHAPE
    )
    hydra = _arm_records(
        census.HYDRA_MODE, kernel_shape=census.FUSED_KERNEL_SHAPE
    )
    with pytest.raises(
        census.CensusError,
        match="publish no staging layout digest",
    ):
        census.validate_campaign(
            [(record, f"tail:{index}") for index, record in enumerate(tail)],
            [(record, f"hydra:{index}") for index, record in enumerate(hydra)],
            required_batches=(1,),
        )


# --- the per-event census family learns the fused shape ------------------


def _event_writer_sections() -> dict[str, set[str]]:
    """Keys the runtime writer publishes for each conv work section."""
    source = PATCHER.read_text(encoding="utf-8")
    runtime = next(
        node.value.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
        and isinstance(node.value, ast.Constant)
    )
    sections: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(runtime)):
        if not isinstance(node, ast.Assign) or not isinstance(
            node.value, ast.Dict
        ):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "event"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value
                in ("conv_pregather", "sfwd_conv_postprep")
            ):
                sections[target.slice.value] = {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant)
                }
    return sections


def test_the_event_writer_and_validator_agree_on_both_sections() -> None:
    sections = _event_writer_sections()
    assert sections["conv_pregather"] == set(census.CONV_PREGATHER_KEYS)
    assert sections["sfwd_conv_postprep"] == set(
        census.SFWD_CONV_POSTPREP_KEYS
    )


def test_the_fused_event_section_cannot_carry_staging_geometry() -> None:
    """row_elems/programs/layout are staging grid, not fused observations."""
    for forbidden in ("row_elems", "programs", "layout_sha256"):
        assert forbidden not in census.SFWD_CONV_POSTPREP_KEYS
    # staged_rows survives only so the fused arm states the zero explicitly.
    assert "staged_rows" in census.SFWD_CONV_POSTPREP_KEYS


def test_the_writer_records_the_shape_and_never_both_sections() -> None:
    source = PATCHER.read_text(encoding="utf-8")
    assert 'event["kernel_shape"] = _fr13_fixed32_kernel_shape()' in source
    assert 'event["conv_pregather"] = None' in source
    assert 'event["sfwd_conv_postprep"] = None' in source
    assert '"route": "fused_conv_postprep_single_kernel",' in source


def _shape_event(shape: str, **kwargs):
    return census.reference_event(
        census.TAIL_MODE,
        1,
        "tail:event-shape",
        event_index=0,
        forward_step_index=0,
        request_ids=["req-0"],
        kernel_shape=shape,
        **kwargs,
    )


@pytest.mark.parametrize("shape", census.KERNEL_SHAPES)
def test_events_validate_under_both_shapes(shape: str) -> None:
    event = _shape_event(shape)
    validated = census.validate_event(event, source="s")
    fused = shape == census.FUSED_KERNEL_SHAPE
    assert (validated.conv_layout_sha256 is None) is fused
    conv = validated.normalized_work["conv_pregather"]
    if fused:
        assert conv["kernel_shape"] == census.FUSED_KERNEL_SHAPE
        assert conv["route"] == census.SFWD_CONV_POSTPREP_ROUTE
        assert conv["calls_per_event"] == census.CONV_PREGATHER_LAYERS
        for zeroed in (
            "stage_calls_per_event",
            "row_elems",
            "programs_per_request",
            "staged_rows_per_request",
            "consume_calls_per_event",
            "consume_hits_per_event",
            "freshness_matches_per_event",
        ):
            assert conv[zeroed] == 0, zeroed
    else:
        assert conv["route"] == census.CONV_PREGATHER_ROUTE
        assert conv["stage_calls_per_event"] == 1
        assert conv["row_elems"] == census.CONV_PREGATHER_ROW_ELEMS


def test_v12_events_without_the_shape_keys_still_validate() -> None:
    """Constraint: recorded evidence predates both keys and is untouched."""
    event = {
        key: value
        for key, value in _shape_event(census.UNFUSED_KERNEL_SHAPE).items()
        if key not in census.EVENT_KERNEL_SHAPE_KEYS
    }
    assert not (census.EVENT_KERNEL_SHAPE_KEYS & set(event))
    validated = census.validate_event(event, source="s")
    assert validated.conv_layout_sha256 is not None
    # The optional pair is never in the required key set.
    assert not (census.EVENT_KERNEL_SHAPE_KEYS & census.TOP_LEVEL_KEYS)


@pytest.mark.parametrize(
    ("declared", "section_shape"),
    (
        (census.FUSED_KERNEL_SHAPE, census.UNFUSED_KERNEL_SHAPE),
        (census.UNFUSED_KERNEL_SHAPE, census.FUSED_KERNEL_SHAPE),
    ),
)
def test_an_event_cannot_declare_a_shape_its_section_disproves(
    declared: str, section_shape: str
) -> None:
    event = dict(_shape_event(section_shape))
    event["kernel_shape"] = declared
    with pytest.raises(census.CensusError, match="but it carries sections"):
        census.validate_event(event, source="s")


def test_an_event_that_took_both_routes_proves_neither() -> None:
    event = dict(_shape_event(census.FUSED_KERNEL_SHAPE))
    event["conv_pregather"] = _shape_event(
        census.UNFUSED_KERNEL_SHAPE
    )["conv_pregather"]
    with pytest.raises(census.CensusError, match="but it carries sections"):
        census.validate_event(event, source="s")
    empty = dict(_shape_event(census.FUSED_KERNEL_SHAPE))
    empty["sfwd_conv_postprep"] = None
    with pytest.raises(census.CensusError, match="but it carries sections"):
        census.validate_event(empty, source="s")


@pytest.mark.parametrize(
    "declared", ("fused", "", None, "SFWD_FUSED_CONV_POSTPREP", 0)
)
def test_events_reject_non_canonical_kernel_shapes(declared) -> None:
    event = dict(_shape_event(census.FUSED_KERNEL_SHAPE))
    event["kernel_shape"] = declared
    with pytest.raises(census.CensusError, match="kernel_shape: expected one of"):
        census.validate_event(event, source="s")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("calls", census.CONV_PREGATHER_LAYERS - 1),
        ("calls_per_layer", 2),
        ("stage_calls", 1),
        ("staged_rows", census.CONV_PREGATHER_LAYERS),
        ("consume_calls", census.CONV_PREGATHER_LAYERS),
        ("consume_hits", census.CONV_PREGATHER_LAYERS),
        ("freshness_matches", census.CONV_PREGATHER_LAYERS),
        ("route", census.CONV_PREGATHER_ROUTE),
    ),
)
def test_a_fused_event_that_also_staged_is_rejected(field: str, value) -> None:
    event = dict(_shape_event(census.FUSED_KERNEL_SHAPE))
    event["sfwd_conv_postprep"] = {
        **event["sfwd_conv_postprep"],
        field: value,
    }
    with pytest.raises(census.CensusError):
        census.validate_event(event, source="s")


@pytest.mark.parametrize("shape", census.KERNEL_SHAPES)
def test_a_whole_campaign_validates_under_both_shapes(shape: str) -> None:
    """The chain the runner walks: events, terminal, campaign, both gates."""
    def arm(mode: str):
        events = [
            census.reference_event(
                mode,
                1,
                f"{mode}:campaign:{index}",
                event_index=index,
                forward_step_index=index,
                request_ids=[f"req-{mode}-{index}"],
                kernel_shape=shape,
            )
            for index in range(3)
        ]
        return [
            *events,
            census.reference_terminal_summary(
                events,
                fixture_synthetic_runtime_proof=True,
                kernel_shape=shape,
            ),
        ]

    tail, hydra = arm(census.TAIL_MODE), arm(census.HYDRA_MODE)
    report = census.validate_campaign(
        [(record, f"tail:{index}") for index, record in enumerate(tail)],
        [(record, f"hydra:{index}") for index, record in enumerate(hydra)],
        required_batches=(1,),
    )
    floor = FLOOR_GATE.validate_work_census_v5_report(report, required_batch=1)
    depth = DEPTH_GATE.validate_work_census_v5_report(report, required_batch=1)
    assert floor == depth
    assert (
        floor["forward_graph_pregather_lifecycle"]["kernel_shape"] == shape
    )


def test_a_fused_terminal_still_rejects_events_that_staged() -> None:
    """The contradiction the exact4 candidate arm recorded."""
    events = [
        census.reference_event(
            census.TAIL_MODE,
            1,
            f"tail:mixed:{index}",
            event_index=index,
            forward_step_index=index,
            request_ids=[f"req-{index}"],
        )
        for index in range(2)
    ]
    hydra = [
        census.reference_event(
            census.HYDRA_MODE,
            1,
            f"hydra:mixed:{index}",
            event_index=index,
            forward_step_index=index,
            request_ids=[f"hreq-{index}"],
        )
        for index in range(2)
    ]
    tail_records = [
        *events,
        census.reference_terminal_summary(
            events,
            fixture_synthetic_runtime_proof=True,
            kernel_shape=census.FUSED_KERNEL_SHAPE,
        ),
    ]
    hydra_records = [
        *hydra,
        census.reference_terminal_summary(
            hydra,
            fixture_synthetic_runtime_proof=True,
            kernel_shape=census.FUSED_KERNEL_SHAPE,
        ),
    ]
    with pytest.raises(
        census.CensusError, match="publish no staging layout digest"
    ):
        census.validate_campaign(
            [(r, f"t:{i}") for i, r in enumerate(tail_records)],
            [(r, f"h:{i}") for i, r in enumerate(hydra_records)],
            required_batches=(1,),
        )


# --- replay provenance under both shapes ---------------------------------


def _observed_take_runtime(*, fused: bool):
    """Exec the commit-seal replay check out of the embedded runtime."""
    source = PATCHER.read_text(encoding="utf-8")
    runtime = next(
        node.value.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
        and isinstance(node.value, ast.Constant)
    )
    wanted = {
        "_fr13_fixed32_observed_take",
        "_fr13_fixed32_observed_current",
        "_fr13_fixed32_kernel_shape",
        "_fr13_fixed32_observed_conv_work",
        "_fr13_fixed32_validate_forward_work",
    }
    definitions = [
        node
        for node in ast.parse(runtime).body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace: dict[str, object] = {
        "_FR13_FIXED32_TARGET_TREE_LAYERS": TREE_LAYERS,
        "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION": fused,
        "_FR13_FIXED32_MODE": "hydra27_fixed32",
        "_FR13_FIXED32_OBSERVED_CURRENT": None,
    }
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            "<observed-take>",
            "exec",
        ),
        namespace,
    )
    return namespace


def _sealed_event(*, fused: bool, batch: int = 1) -> dict[str, object]:
    """One completed event, as the writer publishes it for its shape."""
    event = dict(_work(fused=fused, batch=batch))
    # _work() is the capture-time census; a sealed event has replayed once.
    if not fused:
        event["conv_stage_replays"] = 1
    section = {
        "route": "fused_conv_postprep_single_kernel",
        "layers": 48,
        "requests": batch,
        "calls": 48,
        "calls_per_layer": 1,
        "stage_calls": 0,
        "staged_rows": 0,
        "consume_calls": 0,
        "consume_hits": 0,
        "consume_fallbacks": 0,
        "freshness_matches": 0,
    }
    event.update(
        mode="hydra27_fixed32",
        forward_step_index=0,
        request_ids=("req-0",),
        execution_basis="cudagraph_full_replay",
        forward_graph_replays=1,
        forward_graph_id=253756081434992,
        forward_graph_signature="2e" + "0" * 62,
        kernel_shape=(
            "sfwd_fused_conv_postprep" if fused else "unfused_conv_pregather"
        ),
        conv_pregather=None if fused else {"route": "in_graph_preconsume"},
        sfwd_conv_postprep=section if fused else None,
        conv_commit={},
        committer={},
        preforward_pack={},
        output_publish={},
        accepted_path_pack={},
        request_key_pack={},
        kv_remap={},
        batch_purity={},
        drafter={},
        drafter_runtime={},
        taw={},
        gdn_comparator=None,
        gdn_parent_sha256="c" * 64,
        gdn_ancestry_sha256="d" * 64,
        failures={},
    )
    return event


@pytest.mark.parametrize("fused", (False, True))
@pytest.mark.parametrize("batch", (1, 4))
def test_replay_provenance_accepts_one_replay_under_both_shapes(
    fused: bool, batch: int
) -> None:
    """Regression: the fused event has no conv_pregather section to find."""
    runtime = _observed_take_runtime(fused=fused)
    event = _sealed_event(fused=fused, batch=batch)
    runtime["_FR13_FIXED32_OBSERVED_CURRENT"] = event
    observed = runtime["_fr13_fixed32_observed_take"](
        "hydra27_fixed32", batch, 0
    )
    assert observed["kernel_shape"] == event["kernel_shape"]
    assert (observed["conv_pregather"] is None) is fused
    assert (observed["sfwd_conv_postprep"] is None) is not fused


@pytest.mark.parametrize("fused", (False, True))
def test_replay_provenance_rejects_the_other_shapes_section(
    fused: bool,
) -> None:
    """An event publishing the wrong section fails, and the message says so."""
    runtime = _observed_take_runtime(fused=fused)
    event = _sealed_event(fused=fused)
    event["conv_pregather"], event["sfwd_conv_postprep"] = (
        event["sfwd_conv_postprep"],
        event["conv_pregather"],
    )
    runtime["_FR13_FIXED32_OBSERVED_CURRENT"] = event
    with pytest.raises(RuntimeError) as caught:
        runtime["_fr13_fixed32_observed_take"]("hydra27_fixed32", 1, 0)
    message = str(caught.value)
    # The replay fields all pass here, so the message must name the section
    # that did not: reporting only the passing fields is what misled the diag.
    assert "conv work section does not match its kernel shape" in message
    assert "'expected'" in message and "'present'" in message
    assert "kernel_shape" in message


def test_replay_provenance_rejects_an_event_whose_shape_drifted() -> None:
    runtime = _observed_take_runtime(fused=True)
    event = _sealed_event(fused=True)
    event["kernel_shape"] = "unfused_conv_pregather"
    runtime["_FR13_FIXED32_OBSERVED_CURRENT"] = event
    with pytest.raises(RuntimeError, match="event kernel shape drifted"):
        runtime["_fr13_fixed32_observed_take"]("hydra27_fixed32", 1, 0)


# --- every observed-section reader goes through the resolver -------------


_TAW_EVIDENCE = {
    "loop_iterations": 1,
    "topology_cache_hit": True,
    "cache_misses": 0,
}


def _conv_work_runtime(*, fused: bool):
    """Exec the resolver and its two real callers out of the runtime."""
    source = PATCHER.read_text(encoding="utf-8")
    runtime = next(
        node.value.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
        and isinstance(node.value, ast.Constant)
    )
    wanted = {
        "_fr13_fixed32_observed_conv_work",
        "_fr13_fixed32_failure_counts",
    }
    definitions = [
        node
        for node in ast.parse(runtime).body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace: dict[str, object] = {
        "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION": fused,
    }
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            "<conv-work>",
            "exec",
        ),
        namespace,
    )
    return namespace


def _observed_record(*, fused: bool, batch: int = 1) -> dict[str, object]:
    """One observed record as _fr13_fixed32_observed_take returns it."""
    record: dict[str, object] = {
        "mode": "hydra27_fixed32",
        "batch_size": batch,
        "kernel_shape": (
            "sfwd_fused_conv_postprep" if fused else "unfused_conv_pregather"
        ),
        "conv_pregather": (
            None
            if fused
            else {"route": "in_graph_preconsume", "consume_fallbacks": 0}
        ),
        "sfwd_conv_postprep": (
            {
                "route": "fused_conv_postprep_single_kernel",
                "consume_fallbacks": 0,
            }
            if fused
            else None
        ),
        "output_publish": {
            "route": "device_fixed32",
            "fallback": 0,
            "capacity": 64,
        },
        "accepted_path_pack": {
            "route": "device_fixed16",
            "fallback": 0,
            "capacity": 64,
            "overflow": 0,
        },
        "request_key_pack": {"route": "device_rowmap", "fallback": 0},
        "kv_remap": {
            "route": "syncfree_target16_postsample_drafter1_postforward",
            "fallback": 0,
        },
        "conv_commit": {
            "route": "fixed32_direct_source_col0",
            "fallback": 0,
        },
        "committer": {
            "route": "fixed16_device_fill_graph",
            "fallback": 0,
            "overflow": 0,
            "graph_dead": 0,
            "graph_replays": 1,
            "graph_captures": 0,
        },
        "batch_purity": {
            "batch_rows": batch,
            "spec_rows": batch,
            "physical_draft_counts": [31] * batch,
            "mixed_pseudo_rows": 0,
            "all_physical_31": True,
        },
    }
    return record


@pytest.mark.parametrize("fused", (False, True))
def test_the_resolver_names_the_section_for_each_shape(fused: bool) -> None:
    runtime = _conv_work_runtime(fused=fused)
    shape, section, route = runtime["_fr13_fixed32_observed_conv_work"](
        _observed_record(fused=fused), "test"
    )
    assert shape == (
        "sfwd_fused_conv_postprep" if fused else "unfused_conv_pregather"
    )
    assert section == (
        "sfwd_conv_postprep" if fused else "conv_pregather"
    )
    assert route == (
        "fused_conv_postprep_single_kernel"
        if fused
        else "in_graph_preconsume"
    )


@pytest.mark.parametrize("fused", (False, True))
def test_failure_counts_reads_the_published_section_under_both_shapes(
    fused: bool,
) -> None:
    """Regression: the fused record has no conv_pregather to .get() into."""
    runtime = _conv_work_runtime(fused=fused)
    counts = runtime["_fr13_fixed32_failure_counts"](
        _observed_record(fused=fused),
        _TAW_EVIDENCE,
    )
    # No route mismatch and no fallback: the resolver pointed the reader at
    # the section this shape actually published.
    assert counts["fallback"] == 0
    assert counts["overflow"] == 0
    assert counts["graph_dead"] == 0


@pytest.mark.parametrize("fused", (False, True))
def test_readers_reject_a_record_carrying_the_other_shapes_section(
    fused: bool,
) -> None:
    runtime = _conv_work_runtime(fused=fused)
    record = _observed_record(fused=fused)
    record["conv_pregather"], record["sfwd_conv_postprep"] = (
        record["sfwd_conv_postprep"],
        record["conv_pregather"],
    )
    with pytest.raises(
        RuntimeError, match="conv work section does not match its kernel shape"
    ):
        runtime["_fr13_fixed32_observed_conv_work"](record, "test")
    with pytest.raises(
        RuntimeError, match="conv work section does not match its kernel shape"
    ):
        runtime["_fr13_fixed32_failure_counts"](
            record, _TAW_EVIDENCE
        )


def test_readers_reject_a_record_carrying_both_or_neither_section() -> None:
    runtime = _conv_work_runtime(fused=True)
    both = _observed_record(fused=True)
    both["conv_pregather"] = {
        "route": "in_graph_preconsume",
        "consume_fallbacks": 0,
    }
    neither = _observed_record(fused=True)
    neither["sfwd_conv_postprep"] = None
    for record in (both, neither):
        with pytest.raises(
            RuntimeError,
            match="conv work section does not match its kernel shape",
        ):
            runtime["_fr13_fixed32_observed_conv_work"](record, "test")


@pytest.mark.parametrize("shape", ("fused", "", None, "conv_pregather"))
def test_readers_reject_a_non_canonical_observed_kernel_shape(shape) -> None:
    runtime = _conv_work_runtime(fused=True)
    record = _observed_record(fused=True)
    record["kernel_shape"] = shape
    with pytest.raises(
        RuntimeError, match="observed kernel shape is not canonical"
    ):
        runtime["_fr13_fixed32_observed_conv_work"](record, "test")


def test_no_reader_names_a_conv_work_section_literally() -> None:
    """By construction: the fixed section lists carry no conv work name.

    Every literal that remains is a writer, the resolver itself, or a
    manifest read that already branches on shape. A reader that iterates a
    hardcoded section tuple is what broke the last two boots.
    """
    source = PATCHER.read_text(encoding="utf-8")
    runtime = next(
        node.value.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
        and isinstance(node.value, ast.Constant)
    )
    sections = {"conv_pregather", "sfwd_conv_postprep"}
    offenders = []
    for node in ast.walk(ast.parse(runtime)):
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for element in node.elts:
                if (
                    isinstance(element, ast.Constant)
                    and element.value in sections
                ):
                    offenders.append((element.lineno, element.value))
    assert offenders == [], offenders
