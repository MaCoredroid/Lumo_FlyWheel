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
