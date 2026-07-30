from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path("scripts").resolve()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

topology = importlib.import_module("fr13_fixed32_topology")
census = importlib.import_module("fr13_fixed32_work_census")


PINNED_TEXT_CONFIG = {
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}

PHYSICAL_WORK_SECTIONS = (
    "drafter",
    "drafter_runtime",
    "tree_attn",
    "gdn",
    "taw",
    "output_publish",
    "accepted_path_pack",
    "request_key_pack",
    "kv_remap",
    "conv_commit",
    "conv_pregather",
    "committer",
)


def test_pregather_geometry_matches_pinned_qwen_state_shape() -> None:
    channels = (
        2
        * PINNED_TEXT_CONFIG["linear_num_key_heads"]
        * PINNED_TEXT_CONFIG["linear_key_head_dim"]
        + PINNED_TEXT_CONFIG["linear_num_value_heads"]
        * PINNED_TEXT_CONFIG["linear_value_head_dim"]
    )
    state_length = (
        PINNED_TEXT_CONFIG["linear_conv_kernel_dim"]
        - 1
        + topology.PHYSICAL_DRAFTS
    )
    row_elems = channels * state_length

    assert channels == topology.GDN_CONV_CHANNELS == 10240
    assert state_length == topology.GDN_CONV_STATE_LENGTH == 34
    assert row_elems == topology.CONV_PREGATHER_ROW_ELEMS == 348160


@pytest.mark.parametrize(
    ("batch_size", "expected_grid", "expected_programs"),
    (
        (1, [48, 1, 340], 16320),
        (4, [48, 4, 340], 65280),
    ),
)
def test_pregather_grid_is_fixed_for_physical_31(
    batch_size: int,
    expected_grid: list[int],
    expected_programs: int,
) -> None:
    manifest = census.forward_graph_structural_manifest(batch_size)
    conv = manifest["conv_pregather"]

    assert conv["row_elems"] == 348160
    assert conv["grid"] == expected_grid
    assert conv["programs"] == expected_programs

    tail = census.reference_event("tail6_fixed32", batch_size, "tail")
    hydra = census.reference_event("hydra27_fixed32", batch_size, "hydra")
    assert tail["conv_pregather"] == hydra["conv_pregather"]


@pytest.mark.parametrize("batch_size", (1, 4))
def test_logical_mask_does_not_change_physical_work(batch_size: int) -> None:
    tail = census.reference_event("tail6_fixed32", batch_size, "same-work")
    hydra = census.reference_event("hydra27_fixed32", batch_size, "same-work")

    for section in PHYSICAL_WORK_SECTIONS:
        assert tail[section] == hydra[section], section


def test_census_rejects_stale_pregather_row_width() -> None:
    event = copy.deepcopy(
        census.reference_event("tail6_fixed32", 1, "stale-row-width")
    )
    event["conv_pregather"]["row_elems"] = 32768
    event["conv_pregather"]["programs"] = 1536

    with pytest.raises(census.CensusError, match="row_elems"):
        census.validate_event(event, source="stale-row-width")
