from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path("scripts").resolve()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

topology = importlib.import_module("fr13_fixed32_topology")
census = importlib.import_module("fr13_fixed32_work_census")

MANIFEST_PATH = Path(
    "results/fr13_fixed32_hydra31_staged_20260731/topology_manifest.json"
)


def test_hydra31_activates_only_the_paid_rank2_suffix() -> None:
    changed = tuple(
        node
        for node, (hydra27, hydra31) in enumerate(
            zip(topology.HYDRA27_VALID, topology.HYDRA31_VALID, strict=True)
        )
        if hydra27 != hydra31
    )

    assert changed == (17, 22, 24, 26)
    assert changed == topology.HYDRA31_ACTIVATED_DRAFT_IDS
    assert topology.HYDRA31_ACTIVATED_PATHS == (
        (2, 0, 0, 0),
        (2, 0, 0, 0, 0),
        (2, 0, 0, 0, 0, 0),
        (2, 0, 0, 0, 0, 0, 0),
    )
    assert sum(topology.HYDRA31_VALID) == topology.HYDRA31_ACTIVE_DRAFTS == 31
    assert topology.HYDRA31_VALID_MASK == 0x7FFFFFFF
    assert topology.HYDRA31_INACTIVE_DRAFT_IDS == ()
    assert topology.staged_active_choices(topology.HYDRA31_STAGED_MODE) == (
        topology.FIXED32_CHOICES
    )


def test_hydra31_keeps_fixed32_launch_and_capacity_geometry() -> None:
    table, counts = topology.staged_sampler_child_table(
        topology.HYDRA31_STAGED_MODE
    )

    assert topology.PHYSICAL_DRAFTS == 31
    assert topology.PHYSICAL_ROWS == 32
    assert (len(table), len(table[0])) == topology.SAMPLER_TABLE_SHAPE == (32, 3)
    assert len(counts) == topology.PHYSICAL_ROWS
    assert sum(counts) == topology.PHYSICAL_DRAFTS

    signature = topology.HYDRA31_STAGED_MANIFEST["fixed_execution_signature"]
    assert signature == topology.FIXED_EXECUTION_SIGNATURE
    assert signature["target_rows"] == 32
    assert signature["tree_attention_rows"] == 32
    assert signature["gdn_rows"] == 32
    assert signature["sampler_parent_scans"] == 31
    assert signature["sampler_walk_iterations"] == 12
    assert signature["sampler_table_shape"] == (32, 3)
    assert signature["output_publish_capacity"] == 32
    assert signature["committer_path_capacity"] == 16
    assert signature["kv_remap_path_capacity"] == 16
    assert signature["arctic_lookup_chains"] == ((1, 4), (2, 2))
    assert signature["arctic_lookup_tokens_per_request"] == 12
    assert signature["rescue_carry_slots_per_request"] == 4


def test_hydra31_is_default_off_in_runtime_and_census_registries() -> None:
    mode = topology.HYDRA31_STAGED_MODE

    assert topology.HYDRA31_STAGED_MANIFEST["default_enabled"] is False
    assert mode not in topology.VALID_BY_MODE
    assert mode not in topology.VALID_MASK_BY_MODE
    assert mode not in census.MODE_SEMANTICS
    with pytest.raises(KeyError):
        census.reference_event(mode, 1, "must-remain-default-off")


@pytest.mark.parametrize("batch_size", (1, 4))
def test_census_physical_work_remains_fixed_at_b1_and_b4(batch_size: int) -> None:
    tail = census.validate_event(
        census.reference_event(census.TAIL_MODE, batch_size, "tail"),
        source=f"tail-b{batch_size}",
    )
    hydra = census.validate_event(
        census.reference_event(census.HYDRA_MODE, batch_size, "hydra"),
        source=f"hydra-b{batch_size}",
    )

    assert tail.normalized_work == hydra.normalized_work
    work = hydra.normalized_work
    assert work["physical_drafts"] == 31
    assert work["verify_rows_per_request"] == 32
    assert work["tree_attn"]["q_rows_per_call_per_request"] == 32
    assert work["gdn"]["nodes_per_scan"] == 32
    assert work["taw"]["table_rows_per_request"] == 32
    assert work["taw"]["loop_iterations"] == 12
    assert work["output_publish"]["capacity"] == 32
    assert work["committer"]["path_capacity"] == 16
    assert work["kv_remap"]["path_capacity"] == 16


def test_published_staged_manifest_matches_the_code_contract() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = json.loads(
        json.dumps(
            topology.HYDRA31_STAGED_MANIFEST,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )

    assert payload == expected
