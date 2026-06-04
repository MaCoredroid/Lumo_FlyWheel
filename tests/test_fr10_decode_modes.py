from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumo_flywheel_serving.fr10_decode_modes import (
    NAIVE_MTP,
    NON_MTP,
    TREE_MTP,
    decode_mode_from_request,
    evaluate_mixed_mode_contamination_control,
    evaluate_relaunch_equivalence_sampling,
    evaluate_relaunch_equivalence_tokens,
    homogeneous_mode_decision,
    mode_allows_spec_decode,
    mode_uses_tree_gdn,
    normalize_decode_mode,
    require_homogeneous_modes,
    select_path0_spec_tokens,
    tree_path0_choice_indices,
)
from lumo_flywheel_serving.fr10_equivalence_gate import (
    SamplingRecord,
    StateParityRow,
    TokenRecord,
)


def _request(extra_args: dict[str, object] | None) -> SimpleNamespace:
    return SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra_args))


def _token_record(tokens: list[int]) -> TokenRecord:
    return TokenRecord("b1", 0, "prompt", tuple(tokens))


def _sample(prompt_id: int, sample_index: int, token: int) -> SamplingRecord:
    return SamplingRecord(prompt_id, sample_index, f"prompt-{prompt_id}", (token,))


def test_decode_mode_parses_vllm_xargs_aliases_strictly() -> None:
    assert decode_mode_from_request(_request({"fr10_decode_mode": "tree-mtp"})) == TREE_MTP
    assert decode_mode_from_request(_request({"decode_mode": "naive_mtp"})) == NAIVE_MTP
    assert decode_mode_from_request(_request({"mode": "non_mtp"})) == NON_MTP
    assert decode_mode_from_request(_request({})) == TREE_MTP

    with pytest.raises(ValueError, match="invalid FR10 decode mode"):
        normalize_decode_mode("native_tree")
    with pytest.raises(ValueError, match="must be a string"):
        normalize_decode_mode(1)


def test_mode_capability_helpers_are_fail_closed() -> None:
    assert mode_allows_spec_decode(TREE_MTP)
    assert mode_allows_spec_decode(NAIVE_MTP)
    assert not mode_allows_spec_decode(NON_MTP)
    assert mode_uses_tree_gdn(TREE_MTP)
    assert not mode_uses_tree_gdn(NAIVE_MTP)
    assert not mode_uses_tree_gdn(NON_MTP)


def test_tree_server_can_select_naive_mtp_path0_from_flat_tree_tokens() -> None:
    tree = [
        (0,),
        (1,),
        (0, 0),
        (1, 0),
        (0, 0, 0),
        (1, 0, 0),
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
    ]

    assert tree_path0_choice_indices(tree) == (0, 2, 4, 6, 8)
    assert select_path0_spec_tokens([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], tree) == [
        10,
        12,
        14,
        16,
        18,
    ]


def test_homogeneous_mode_decision_defers_mixed_requests() -> None:
    decision = homogeneous_mode_decision([TREE_MTP, TREE_MTP, NON_MTP, TREE_MTP])

    assert decision.mode == TREE_MTP
    assert decision.accepted_indices == (0, 1, 3)
    assert decision.deferred_indices == (2,)
    assert decision.mixed
    assert require_homogeneous_modes([NAIVE_MTP, NAIVE_MTP]) == NAIVE_MTP
    with pytest.raises(RuntimeError, match="mixed decode-mode batch rejected"):
        require_homogeneous_modes([NAIVE_MTP, TREE_MTP])


def test_relaunch_equivalence_token_gate_detects_shared_server_drift() -> None:
    dedicated = {("b1", 0): _token_record([1, 2, 3])}
    shared_ok = {("b1", 0): _token_record([1, 2, 3])}
    shared_bad = {("b1", 0): _token_record([1, 9, 3])}

    assert evaluate_relaunch_equivalence_tokens(shared_ok, dedicated, mode=TREE_MTP).passed
    report = evaluate_relaunch_equivalence_tokens(shared_bad, dedicated, mode=TREE_MTP)
    assert not report.passed
    assert report.metrics["flip_count"] == 1


def test_relaunch_equivalence_sampling_uses_same_regime_floor() -> None:
    dedicated = [_sample(0, 0, 1), _sample(0, 1, 2), _sample(0, 2, 2), _sample(0, 3, 3)]
    shared_same = [_sample(0, 0, 1), _sample(0, 1, 2), _sample(0, 2, 2), _sample(0, 3, 3)]
    floor_other = [_sample(0, 0, 1), _sample(0, 1, 2), _sample(0, 2, 3), _sample(0, 3, 3)]
    shared_bad = [_sample(0, 0, 9), _sample(0, 1, 9), _sample(0, 2, 9), _sample(0, 3, 9)]

    assert evaluate_relaunch_equivalence_sampling(
        shared_same,
        dedicated,
        floor_other,
        dedicated,
        mode=TREE_MTP,
        positions=1,
    ).passed
    report = evaluate_relaunch_equivalence_sampling(
        shared_bad,
        dedicated,
        floor_other,
        dedicated,
        mode=TREE_MTP,
        positions=1,
    )
    assert not report.passed
    assert "exceeds same-regime floor" in report.violations[0]


def test_mixed_mode_contamination_control_is_powered() -> None:
    contaminated = [
        StateParityRow(layer=0, node=0, max_state_abs=0.5618, max_output_abs=0.5618)
    ]
    clean = [StateParityRow(layer=0, node=0, max_state_abs=7e-6, max_output_abs=7e-9)]

    assert evaluate_mixed_mode_contamination_control(contaminated).passed
    report = evaluate_mixed_mode_contamination_control(clean)
    assert not report.passed
    assert "no power" in report.violations[0]

