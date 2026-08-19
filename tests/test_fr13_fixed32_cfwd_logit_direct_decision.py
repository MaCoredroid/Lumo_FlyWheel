from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


KERNEL_PATH = Path("scripts/fr13_cfwd_logit_direct_decision_kernel.py")
SERVED_PATH = Path("scripts/fr13_device_multidraft_cfwd_packed_v3.py")
BASE_PATH = Path("scripts/fr13_device_multidraft_kernel.py")
OVERLAY_PATH = Path("scripts/fr13_cfwd_logit_direct_packed_runtime_overlay.py")
GENERATOR_PATH = Path("scripts/fr13_generate_cfwd_packed_runtime_overlay.py")
SPEC = importlib.util.spec_from_file_location(
    "fr13_cfwd_logit_direct_decision_kernel_test",
    KERNEL_PATH,
)
assert SPEC is not None and SPEC.loader is not None
kernel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kernel)


def _inverse_cdf(weights, uniform):
    cdf = torch.cumsum(weights, dim=-1)
    threshold = uniform.unsqueeze(-1) * cdf[..., -1:]
    return (cdf <= threshold).sum(dim=-1).clamp(max=weights.shape[-1] - 1)


def _dense_probability_reference(logits, kid_tokens, kid_mask, uniforms):
    probability = torch.softmax(logits.to(torch.float64), dim=-1)
    overlaps = torch.gather(probability, -1, kid_tokens) * kid_mask
    overlap_mass = overlaps.sum(-1, keepdim=True)
    source = _inverse_cdf(overlaps, uniforms[..., 0])
    selected = torch.gather(kid_tokens, -1, source.unsqueeze(-1)).squeeze(-1)
    q_weights = overlaps / overlap_mass.clamp(min=1.0e-30)
    same = (kid_tokens == selected.unsqueeze(-1)) & kid_mask
    q_mix_selected = (q_weights * same).sum(-1)
    target_at_token = torch.gather(
        probability, -1, selected.unsqueeze(-1)
    ).squeeze(-1)
    accept_probability = (
        target_at_token / q_mix_selected.clamp(min=1.0e-30)
    ).clamp(max=1.0)
    accepted = (
        kid_mask.any(-1)
        & (overlap_mass.squeeze(-1) > 0)
        & (uniforms[..., 1] < accept_probability)
    )

    q_mix = torch.zeros_like(probability)
    q_mix.scatter_add_(-1, kid_tokens, q_weights * kid_mask)
    residual = (probability - q_mix).clamp(min=0)
    residual_mass = residual.sum(-1, keepdim=True)
    sampling_probability = torch.where(
        residual_mass > 0,
        residual / residual_mass.clamp(min=1.0e-30),
        probability,
    )
    rejected = _inverse_cdf(sampling_probability, uniforms[..., 2])
    return source, selected, rejected, accepted, sampling_probability


@pytest.mark.parametrize("mode", sorted(kernel.FIXED32_MODES))
@pytest.mark.parametrize(
    ("batch_size", "incumbent_bytes", "candidate_bytes", "removed_bytes"),
    (
        (1, 190_709_760, 14_640, 190_695_120),
        (4, 762_839_040, 58_560, 762_780_480),
    ),
)
def test_contract_has_exact_fixed32_work_ledger(
    mode: str,
    batch_size: int,
    incumbent_bytes: int,
    candidate_bytes: int,
    removed_bytes: int,
) -> None:
    contract = kernel.fixed32_cfwd_logit_direct_contract(
        batch_size,
        mode=mode,
    )
    assert contract["physical_rows"] == 32
    assert contract["physical_drafts"] == 31
    assert contract["fixed_work_for_exact_bound_topology"] is True
    assert contract["logical_topology"] in {"Tail23", "Hydra27"}
    assert contract["vocab_size"] == 248_320
    assert contract["vocab_blocks"] == 61
    assert contract["incumbent_probability_producer_tensor_ops"] == 4
    assert contract["candidate_triton_launch_sites"] == 2
    assert contract["producer_dispatch_sites_removed_static"] == 2
    assert contract["physical_kernel_launches_removed"] == "pending_gpu_trace"
    assert contract["incumbent_full_vocab_materialized_bytes"] == incumbent_bytes
    assert contract["candidate_block_stat_materialized_bytes"] == candidate_bytes
    assert contract["full_vocab_materialized_bytes_removed"] == removed_bytes
    assert contract["candidate_block_stat_workspace_bytes"] == (
        batch_size * 15_360
    )
    assert contract["decision_programs_per_request_before"] == 30
    assert contract["decision_programs_per_request_after"] == 30
    assert contract["decision_values_stored_per_request_before"] == 81
    assert contract["decision_values_stored_per_request_after"] == 30
    assert contract["integer_walk_topology_index_loads_per_request_before"] == 24
    assert contract["integer_walk_topology_index_loads_per_request_after"] == 0
    assert contract["compact_decision_workspace_bytes_before"] == (
        batch_size * 529
    )
    assert contract["physical_decision_workspace_bytes_before"] == (
        batch_size * 1_048
    )
    assert contract["packed_physical_decision_workspace_bytes_after"] == (
        batch_size * 504
    )
    assert contract["decision_workspace_bytes_removed"] == batch_size * 544
    assert contract["decision_workspace_zero_seeded_once"] is True
    assert contract["decision_padding_initialization_stores_per_event"] == 0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"batch_size": 2, "mode": "tail6_fixed32"},
        {"batch_size": 1, "mode": "tail23"},
        {
            "batch_size": 1,
            "mode": "hydra27_fixed32",
            "physical_rows": 31,
        },
        {
            "batch_size": 1,
            "mode": "hydra27_fixed32",
            "vocab_size": 248_319,
        },
    ),
)
def test_contract_fails_closed_on_geometry_drift(kwargs) -> None:
    with pytest.raises(ValueError):
        kernel.fixed32_cfwd_logit_direct_contract(**kwargs)


def test_workspace_is_persistent_physical32_for_b1_and_b4() -> None:
    for batch_size in (1, 4):
        spec = kernel.workspace_spec(batch_size)
        assert spec["block_maxima"] == (
            (batch_size * 30, 64),
            torch.float32,
        )
        assert spec["block_sums"] == (
            (batch_size * 30, 64),
            torch.float32,
        )
        assert spec["self_token"] == ((batch_size, 31), torch.long)
        assert spec["event"] == ((batch_size, 32), torch.long)
        assert spec["invalid"] == ((1,), torch.int32)


@pytest.mark.parametrize("mode", sorted(kernel.FIXED32_MODES))
@pytest.mark.parametrize("batch_size", (1, 4))
def test_physical_decision_slots_preserve_compact_reachable_rows(
    mode: str, batch_size: int
) -> None:
    metadata = _valid_metadata(batch_size, mode)
    generator = torch.Generator().manual_seed(1_000 * batch_size + len(mode))
    compact_self = torch.randint(
        0, 1_000, (batch_size, kernel.SELF_ROWS), generator=generator
    )
    compact_target = torch.randint(
        0, 1_000, (batch_size, kernel.TARGET_ROWS), generator=generator
    )
    physical_self = torch.full((batch_size, kernel.PHYSICAL_DRAFTS), -1)
    physical_target = torch.full((batch_size, kernel.PHYSICAL_ROWS), -1)
    self_sources = metadata["self_source_indices"].reshape(batch_size, -1)
    self_nodes = self_sources - (
        torch.arange(batch_size).unsqueeze(1) * kernel.PHYSICAL_DRAFTS
    )
    physical_self.scatter_(1, self_nodes, compact_self)
    parent_slots = metadata["target_parent_slots"].expand(batch_size, -1)
    physical_target.scatter_(1, parent_slots, compact_target)

    assert torch.equal(torch.gather(physical_self, 1, self_nodes), compact_self)
    assert torch.equal(
        torch.gather(physical_target, 1, parent_slots), compact_target
    )


def _valid_metadata(batch_size: int, mode: str) -> dict[str, torch.Tensor]:
    self_sources = [
        request * 31 + node
        for request in range(batch_size)
        for node in kernel.SELF_SOURCE_NODES
    ]
    target_sources = [
        request * 31 + node
        for request in range(batch_size)
        for node in kernel.TARGET_SOURCE_NODES
    ]
    table = [[-1] * 3 for _ in range(32)]
    counts = [0] * 32
    for parent_slot, children in kernel.MODE_CHILDREN[mode].items():
        counts[parent_slot] = len(children)
        table[parent_slot][: len(children)] = children
    return {
        "self_source_indices": torch.tensor(self_sources),
        "target_source_indices": torch.tensor(target_sources),
        "child_table": torch.tensor([table for _ in range(batch_size)]),
        "child_counts": torch.tensor([counts for _ in range(batch_size)]),
        "self_uniform_levels": torch.tensor(kernel.SELF_UNIFORM_LEVELS),
        "target_parent_slots": torch.tensor(kernel.TARGET_PARENT_SLOTS),
        "target_uniform_levels": torch.tensor(kernel.TARGET_UNIFORM_LEVELS),
    }


def _reference_physical_walk(
    *,
    mode: str,
    self_token: torch.Tensor,
    source: torch.Tensor,
    selected_token: torch.Tensor,
    rejected_token: torch.Tensor,
    accepted: torch.Tensor,
    bonus_token: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    batch = int(self_token.shape[0])
    output = torch.full((batch, kernel.PHYSICAL_ROWS), -1, dtype=torch.long)
    output_lens = torch.zeros(batch, dtype=torch.long)
    paths = torch.zeros((batch, 16), dtype=torch.long)
    path_lens = torch.zeros(batch, dtype=torch.long)
    last_row = torch.zeros(batch, dtype=torch.long)
    for request in range(batch):
        current = -1
        alive = True
        for _level in range(kernel.WALK_CAP):
            if not alive:
                continue
            parent_slot = current + 1
            children = kernel.MODE_CHILDREN[mode].get(parent_slot, ())
            if not children:
                output[request, output_lens[request]] = (
                    self_token[request, current]
                    if current >= 0
                    else bonus_token[request]
                )
                output_lens[request] += 1
                alive = False
                continue
            if accepted[request, parent_slot]:
                output[request, output_lens[request]] = selected_token[
                    request, parent_slot
                ]
                accepted_node = children[int(source[request, parent_slot])]
                accepted_row = accepted_node + 1
                paths[request, path_lens[request]] = accepted_row
                path_lens[request] += 1
                last_row[request] = accepted_row
                current = accepted_node
            else:
                output[request, output_lens[request]] = rejected_token[
                    request, parent_slot
                ]
                alive = False
            output_lens[request] += 1
    return output, output_lens, paths, path_lens, last_row


@pytest.mark.parametrize("mode", sorted(kernel.FIXED32_MODES))
@pytest.mark.parametrize("batch_size", (1, 4))
def test_packed_events_cover_every_fixed_parent_and_boundary_token(
    mode: str, batch_size: int
) -> None:
    metadata = _valid_metadata(batch_size, mode)
    source = torch.zeros((batch_size, kernel.PHYSICAL_ROWS), dtype=torch.long)
    selected = torch.zeros_like(source)
    rejected = torch.zeros_like(source)
    accepted = torch.zeros_like(source, dtype=torch.bool)
    for request in range(batch_size):
        for ordinal, (parent, children) in enumerate(
            kernel.MODE_CHILDREN[mode].items()
        ):
            lane = (request + ordinal) % len(children)
            source[request, parent] = lane
            selected[request, parent] = (
                0 if (request + ordinal) % 2 == 0 else kernel.VOCAB_SIZE - 1
            )
            rejected[request, parent] = (
                kernel.VOCAB_SIZE - 1
                if (request + ordinal) % 2 == 0
                else 0
            )
            accepted[request, parent] = (request + ordinal) % 3 != 0

    events = kernel.pack_physical_event_oracle(
        source,
        selected,
        rejected,
        accepted,
        metadata["child_table"],
        metadata["child_counts"],
    )
    for request in range(batch_size):
        for parent in range(kernel.PHYSICAL_ROWS):
            children = kernel.MODE_CHILDREN[mode].get(parent, ())
            packed = int(events[request, parent])
            if not children:
                assert packed == 0
                continue
            assert packed & kernel.PACKED_EVENT_PARENT_MASK
            expected_token = (
                selected[request, parent]
                if accepted[request, parent]
                else rejected[request, parent]
            )
            assert packed & kernel.PACKED_EVENT_TOKEN_MASK == int(expected_token)
            accepted_row = (
                packed >> kernel.PACKED_EVENT_ACCEPTED_ROW_SHIFT
            ) & kernel.PACKED_EVENT_ACCEPTED_ROW_MASK
            expected_row = (
                children[int(source[request, parent])] + 1
                if accepted[request, parent]
                else 0
            )
            assert accepted_row == expected_row

    # Root source zero accepting node zero must encode row one, while a reject
    # keeps row zero. These are the two sentinel-sensitive boundaries.
    source.zero_()
    selected.zero_()
    rejected.fill_(kernel.VOCAB_SIZE - 1)
    accepted.zero_()
    accepted[0, 0] = True
    events = kernel.pack_physical_event_oracle(
        source,
        selected,
        rejected,
        accepted,
        metadata["child_table"],
        metadata["child_counts"],
    )
    assert events[0, 0].item() & kernel.PACKED_EVENT_TOKEN_MASK == 0
    assert (
        events[0, 0].item() >> kernel.PACKED_EVENT_ACCEPTED_ROW_SHIFT
    ) & kernel.PACKED_EVENT_ACCEPTED_ROW_MASK == 1
    reject_parent = next(parent for parent in kernel.MODE_CHILDREN[mode] if parent)
    assert events[0, reject_parent].item() & kernel.PACKED_EVENT_TOKEN_MASK == (
        kernel.VOCAB_SIZE - 1
    )
    assert (
        events[0, reject_parent].item()
        >> kernel.PACKED_EVENT_ACCEPTED_ROW_SHIFT
    ) & kernel.PACKED_EVENT_ACCEPTED_ROW_MASK == 0


@pytest.mark.parametrize("mode", sorted(kernel.FIXED32_MODES))
@pytest.mark.parametrize("batch_size", (1, 4))
def test_randomized_reference_and_packed_walks_are_exact(
    mode: str, batch_size: int
) -> None:
    metadata = _valid_metadata(batch_size, mode)
    for seed in range(64):
        generator = torch.Generator().manual_seed(
            10_000 * batch_size + 100 * len(mode) + seed
        )
        shape = (batch_size, kernel.PHYSICAL_ROWS)
        source = torch.zeros(shape, dtype=torch.long)
        selected = torch.randint(
            0, kernel.VOCAB_SIZE, shape, generator=generator
        )
        rejected = torch.randint(
            0, kernel.VOCAB_SIZE, shape, generator=generator
        )
        accepted = torch.zeros(shape, dtype=torch.bool)
        for parent, children in kernel.MODE_CHILDREN[mode].items():
            source[:, parent] = torch.randint(
                0, len(children), (batch_size,), generator=generator
            )
            accepted[:, parent] = torch.randint(
                0, 2, (batch_size,), generator=generator
            ).bool()
        self_token = torch.randint(
            0,
            kernel.VOCAB_SIZE,
            (batch_size, kernel.PHYSICAL_DRAFTS),
            generator=generator,
        )
        bonus = torch.randint(
            0, kernel.VOCAB_SIZE, (batch_size,), generator=generator
        )
        events = kernel.pack_physical_event_oracle(
            source,
            selected,
            rejected,
            accepted,
            metadata["child_table"],
            metadata["child_counts"],
        )
        expected = _reference_physical_walk(
            mode=mode,
            self_token=self_token,
            source=source,
            selected_token=selected,
            rejected_token=rejected,
            accepted=accepted,
            bonus_token=bonus,
        )
        actual = kernel.packed_physical_walk_oracle(
            self_token, events, bonus
        )
        assert all(
            torch.equal(expected_value, actual_value)
            for expected_value, actual_value in zip(
                expected, actual, strict=True
            )
        )


@pytest.mark.parametrize("mode", sorted(kernel.FIXED32_MODES))
@pytest.mark.parametrize("batch_size", (1, 4))
def test_metadata_binding_is_exact_and_pointer_version_bound(
    mode: str, batch_size: int
) -> None:
    metadata = _valid_metadata(batch_size, mode)
    binding = kernel.prepare_metadata_binding(
        **metadata,
        batch_size=batch_size,
        mode=mode,
    )
    operands = kernel._metadata_operands(**metadata)
    kernel._validate_metadata_binding(
        binding,
        operands=operands,
        batch_size=batch_size,
        mode=mode,
    )

    metadata["target_parent_slots"].add_(0)
    with pytest.raises(ValueError, match="metadata binding drift"):
        kernel._validate_metadata_binding(
            binding,
            operands=kernel._metadata_operands(**metadata),
            batch_size=batch_size,
            mode=mode,
        )


def test_metadata_binding_supports_inference_tensors_and_rejects_replacement() -> None:
    with torch.inference_mode():
        metadata = _valid_metadata(1, "hydra27_fixed32")
        assert all(
            torch.is_inference(value)
            for _name, value in kernel._metadata_operands(**metadata)
        )
        binding = kernel.prepare_metadata_binding(
            **metadata,
            batch_size=1,
            mode="hydra27_fixed32",
        )
        assert all(version == -1 for _name, _pointer, version in binding.identities)
        kernel._validate_metadata_binding(
            binding,
            operands=kernel._metadata_operands(**metadata),
            batch_size=1,
            mode="hydra27_fixed32",
        )

        metadata["target_parent_slots"] = metadata["target_parent_slots"].clone()
        with pytest.raises(ValueError, match="metadata binding drift"):
            kernel._validate_metadata_binding(
                binding,
                operands=kernel._metadata_operands(**metadata),
                batch_size=1,
                mode="hydra27_fixed32",
            )


def test_metadata_binding_rejects_wrong_topology_contents() -> None:
    metadata = _valid_metadata(1, "hydra27_fixed32")
    metadata["child_table"][0, 7, 0] = -1
    with pytest.raises(ValueError, match="exact metadata drift: child_table"):
        kernel.prepare_metadata_binding(
            **metadata,
            batch_size=1,
            mode="hydra27_fixed32",
        )


def test_workspace_alias_check_uses_exact_contiguous_byte_intervals() -> None:
    storage = torch.empty(16, dtype=torch.float32)
    kernel._reject_workspace_aliases((storage[:8],), {"output": storage[8:]})
    with pytest.raises(ValueError, match="writable storage alias"):
        kernel._reject_workspace_aliases((storage[:8],), {"output": storage[7:]})


@pytest.mark.parametrize("rows", (1, 17, 68))
def test_logit_space_algebra_matches_dense_probability_math(rows: int) -> None:
    for seed in range(12):
        generator = torch.Generator().manual_seed(rows * 1000 + seed)
        logits = torch.randn((rows, 97), generator=generator, dtype=torch.float64)
        kid_tokens = torch.randint(0, 97, (rows, 3), generator=generator)
        if seed % 2 == 0:
            kid_tokens[:, 1] = kid_tokens[:, 0]
        if seed % 3 == 0:
            kid_tokens[:, 2] = kid_tokens[:, 0]
        counts = torch.randint(1, 4, (rows, 1), generator=generator)
        kid_mask = torch.arange(3).unsqueeze(0) < counts
        uniforms = torch.rand((rows, 3), generator=generator, dtype=torch.float64)

        expected = _dense_probability_reference(
            logits, kid_tokens, kid_mask, uniforms
        )
        actual = kernel.logit_direct_decision_oracle(
            logits, kid_tokens, kid_mask, uniforms
        )
        for expected_product, actual_product in zip(
            expected[:4], actual[:4], strict=True
        ):
            assert torch.equal(expected_product, actual_product)
        actual_probability = actual[4] / actual[4].sum(-1, keepdim=True)
        torch.testing.assert_close(
            actual_probability,
            expected[4],
            rtol=2.0e-15,
            atol=2.0e-15,
        )


def test_uniform_columns_preserve_source_accept_residual_order() -> None:
    probability = torch.tensor([[0.05, 0.15, 0.25, 0.30, 0.25]])
    logits = probability.log()
    kid_tokens = torch.tensor([[1, 3, 4]])
    kid_mask = torch.ones_like(kid_tokens, dtype=torch.bool)
    base_uniforms = torch.tensor([[0.0, 0.0, 0.0]])
    base = kernel.logit_direct_decision_oracle(
        logits, kid_tokens, kid_mask, base_uniforms
    )
    assert tuple(value.item() for value in base[:4]) == (0, 1, 0, True)

    source_uniforms = base_uniforms.clone()
    source_uniforms[:, 0] = 0.3
    source_changed = kernel.logit_direct_decision_oracle(
        logits, kid_tokens, kid_mask, source_uniforms
    )
    assert tuple(value.item() for value in source_changed[:4]) == (1, 3, 0, True)

    accept_uniforms = base_uniforms.clone()
    accept_uniforms[:, 1] = 0.8
    accept_changed = kernel.logit_direct_decision_oracle(
        logits, kid_tokens, kid_mask, accept_uniforms
    )
    assert tuple(value.item() for value in accept_changed[:4]) == (0, 1, 0, False)

    residual_uniforms = base_uniforms.clone()
    residual_uniforms[:, 2] = 0.2
    residual_changed = kernel.logit_direct_decision_oracle(
        logits, kid_tokens, kid_mask, residual_uniforms
    )
    assert tuple(value.item() for value in residual_changed[:4]) == (
        0,
        1,
        2,
        True,
    )


def test_strict_inverse_cdf_and_zero_residual_fallback_match_served_rule() -> None:
    weights = torch.tensor([[0.25, 0.75, 0.0]], dtype=torch.float64)
    assert kernel._inverse_cdf_oracle(weights, torch.tensor([0.0])).item() == 0
    assert kernel._inverse_cdf_oracle(weights, torch.tensor([0.25])).item() == 1

    logits = weights[:, :2].log()
    kids = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[True, True, False]])
    uniforms = torch.tensor([[0.0, 0.99, 0.25]], dtype=torch.float64)
    actual = kernel.logit_direct_decision_oracle(logits, kids, mask, uniforms)
    assert actual[2].item() == 1
    torch.testing.assert_close(
        actual[4] / actual[4].sum(-1, keepdim=True),
        torch.softmax(logits, dim=-1),
    )


def _kernel_definitions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_fr13_cfwd_logit_")
    }


def _kernel_launches(tree: ast.AST) -> dict[str, ast.Call]:
    launches = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Subscript):
            continue
        value = node.func.value
        if isinstance(value, ast.Name) and value.id.startswith("_fr13_cfwd_logit_"):
            launches[value.id] = node
    return launches


def test_source_has_two_arity_exact_kernel_launches_and_no_dense_ops() -> None:
    source = KERNEL_PATH.read_text(encoding="ascii")
    tree = ast.parse(source)
    definitions = _kernel_definitions(tree)
    launches = _kernel_launches(tree)
    expected_names = {
        "_fr13_cfwd_logit_block_stats_kernel",
        "_fr13_cfwd_logit_direct_decision_kernel",
    }
    assert definitions.keys() == expected_names
    assert launches.keys() == expected_names
    for name in expected_names:
        definition_args = [arg.arg for arg in definitions[name].args.args]
        call = launches[name]
        launch_kwargs = {
            keyword.arg
            for keyword in call.keywords
            if keyword.arg not in {"num_warps", "num_stages", "waves_per_eu"}
        }
        assert len(call.args) + len(launch_kwargs) == len(definition_args)
        assert definition_args[: len(call.args)]
        assert launch_kwargs == set(definition_args[len(call.args) :])

    launch_source = inspect.getsource(kernel.launch_logit_direct_fixed32)
    assert "torch.softmax" not in launch_source
    assert "torch.zeros_like" not in launch_source
    assert "scatter_add" not in launch_source
    assert launch_source.count("_kernel[") == 2
    assert launch_source.count("num_stages=3") == 2


def test_kernel_uses_scan_terminal_masses_and_sticky_domain_guards() -> None:
    source = KERNEL_PATH.read_text(encoding="ascii")
    assert source.count("tl.atomic_max(invalid_out, 1") == 2
    assert "child_packing_valid" in source
    assert "kid_token_valid" in source
    assert "source_cdf" in source
    assert "block_cdf" in source
    assert "local_total" in source
    assert "local_total / tl.maximum(selected_block_mass" in source
    assert "self_token_out + physical_self_offset" in source
    assert source.count("+ physical_target_offset") == 1
    assert "self_token_out + self_row" not in source
    assert "event_out + target_row" not in source


def test_served_integer_walk_reads_physical_slots_without_topology_maps() -> None:
    source = OVERLAY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    kernel_source = definitions[
        "_fr13_fixed32_taw_packed_physical_slot_commit_kernel"
    ]
    assert "self_slot_by_node" not in kernel_source
    assert "target_slot_by_parent" not in kernel_source
    assert "child_table" not in kernel_source
    assert "child_counts" not in kernel_source
    assert "request * PHYSICAL_DRAFTS + safe_current" in kernel_source
    assert "request * PHYSICAL_ROWS + parent_slot" in kernel_source

    walk_source = definitions["_fr13_cfwd_logit_direct_walk_cuda"]
    assert "_fr13_fixed32_taw_packed_physical_slot_commit_kernel" in walk_source
    assert 'entry["native_self_slot_by_node"]' not in walk_source
    assert 'entry["native_target_slot_by_parent"]' not in walk_source

    compare_source = definitions["_fr13_cfwd_logit_direct_compare_kernel"]
    assert "cand_self_token + self_source" in compare_source
    assert "cand_event + physical_target_offset" in compare_source

    base_source = BASE_PATH.read_text(encoding="utf-8")
    taw_kernel_contract = base_source[
        base_source.index("_FR13_FIXED32_TAW_KERNEL_SOURCE_FUNCTIONS = (") :
        base_source.index("_FR13_FIXED32_TAW_GEOMETRY = {")
    ]
    assert "_fr13_fixed32_taw_packed_physical_slot_commit_kernel" not in (
        taw_kernel_contract
    )
    assert "_fr13_fixed32_taw_packed_physical_slot_commit_kernel" in source


def test_cfwd_integration_source_contract_is_separate_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "fr13_cfwd_integration_source_contract_test",
        SERVED_PATH,
    )
    assert spec is not None and spec.loader is not None
    device = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(device)
    device = device._base
    contract = device._fr13_cfwd_logit_direct_integration_source_contract()
    assert contract == {
        "integration_source_schema": (
            "fr13.fixed32.cfwd_logit_direct.integration_source.v2"
        ),
        "integration_source_sha256": (
            "421465c6c04de8c26e3ea724a7d2f0d3f00fe50b4fdc9f57c35e71e71212297b"
        ),
    }
    assert device._FR13_FIXED32_TAW_SOURCE_SHA256 == (
        "6ffe57287e768bfee5e2e72f10de0dfea6fb3e6c0fa50f32b6c099c63fa916a2"
    )
    assert device._FR13_FIXED32_TAW_KERNEL_SOURCE_FUNCTIONS == (
        "_fr13_fixed32_taw_exact_commit_kernel",
        "_fr13_fixed32_taw_all_parent_commit_kernel",
    )
    assert "_fr13_fixed32_taw_packed_physical_slot_commit_kernel" in (
        device._FR13_CFWD_LOGIT_DIRECT_INTEGRATION_KERNEL_SOURCE_FUNCTIONS
    )
    assert "_fr13_cfwd_logit_direct_walk_cuda" in (
        device._FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_FUNCTIONS
    )
    assert "fr13_fixed32_cfwd_logit_direct_commit" in (
        device._FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_FUNCTIONS
    )

    monkeypatch.setattr(
        device,
        "_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_CACHE",
        None,
    )
    monkeypatch.setattr(
        device,
        "_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SHA256",
        "0" * 64,
    )
    with pytest.raises(RuntimeError, match="composed integration source identity drifted"):
        device._fr13_cfwd_logit_direct_integration_source_contract()


def test_packed_cfwd_keeps_taw_b1_b4_source_identity_exact() -> None:
    spec = importlib.util.spec_from_file_location(
        "fr13_cfwd_taw_source_identity_test",
        SERVED_PATH,
    )
    assert spec is not None and spec.loader is not None
    device = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(device)
    device = device._base
    topology = device._fr13_fixed32_topology()
    for batch_size in (1, 4):
        contract = device._fr13_fixed32_taw_source_contract(
            topology, batch_size=batch_size
        )
        assert contract["source_contract_schema"] == (
            "fr13-fixed32-taw-all-parent-v7"
        )
        assert contract["source_contract_sha256"] == (
            "6ffe57287e768bfee5e2e72f10de0dfea6fb3e6c0fa50f32b6c099c63fa916a2"
        )


@pytest.mark.parametrize("mode", sorted(kernel.FIXED32_MODES))
def test_physical_walk_preseeds_unwritten_leaf_slots_and_child_address(
    mode: str,
) -> None:
    workspace = kernel.allocate_workspace(device="cpu", batch_size=1)
    for name in ("self_token", "event"):
        assert torch.count_nonzero(workspace[name]).item() == 0

    self_token = workspace["self_token"][0].tolist()
    events = workspace["event"][0].tolist()
    for node in kernel.SELF_SOURCE_NODES:
        self_token[node] = 10_000 + node
    # Force root -> node 0 -> leaf node 4. Physical target slot 5 is not a
    # producer row and remains safely zero-seeded outside measured replays.
    events[0] = 100 | (1 << 18) | kernel.PACKED_EVENT_PARENT_MASK
    events[1] = 101 | (5 << 18) | kernel.PACKED_EVENT_PARENT_MASK
    metadata = _valid_metadata(1, mode)
    current = -1
    for _level in range(kernel.WALK_CAP):
        parent_slot = current + 1
        children = kernel.MODE_CHILDREN[mode].get(parent_slot, ())
        has_kids = bool(children)
        if not has_kids:
            assert current == 4
            assert self_token[current] == 10_004
            assert events[parent_slot] == 0
            assert metadata["child_table"][0, parent_slot, 0].item() == -1
            break
        accepted_row = (
            events[parent_slot] >> kernel.PACKED_EVENT_ACCEPTED_ROW_SHIFT
        ) & kernel.PACKED_EVENT_ACCEPTED_ROW_MASK
        current = accepted_row - 1
    else:
        pytest.fail("zero-seeded leaf path did not terminate")

    source = OVERLAY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    physical_kernel = next(
        ast.get_source_segment(source, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_taw_packed_physical_slot_commit_kernel"
    )
    assert "mask=leaf & current_valid" not in physical_kernel
    assert physical_kernel.count("mask=has_kids") == 1
    assert physical_kernel.count("mask=is_accepted") == 1
    allocation_source = inspect.getsource(kernel.allocate_workspace)
    assert "torch.zeros" in allocation_source
    assert "torch.empty" not in allocation_source


def test_candidate_is_default_off_and_wired_only_through_shadow_wrapper() -> None:
    served_source = (
        BASE_PATH.read_text(encoding="utf-8")
        + OVERLAY_PATH.read_text(encoding="utf-8")
    )
    assert kernel.CANDIDATE in served_source
    assert KERNEL_PATH.name in served_source
    assert 'os.environ.get("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "0")' in served_source
    assert "return reference" in served_source
    assert kernel.fixed32_cfwd_logit_direct_contract(
        1, mode="tail6_fixed32"
    )["candidate_default_off"] is True


def test_cfwd_overlay_preserves_credential_bound_device_bytes() -> None:
    import hashlib

    assert hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() == (
        "8dbb0bd07d6431ad99e70ca0c2e09dd2b36e72883eedb21143e0813819141d14"
    )


def test_cfwd_overlay_is_reproducible_from_reviewed_candidate() -> None:
    spec = importlib.util.spec_from_file_location(
        "fr13_generate_cfwd_packed_runtime_overlay_test",
        GENERATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert generator.generate() == OVERLAY_PATH.read_text(encoding="ascii")
