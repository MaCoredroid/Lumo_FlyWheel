from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


KERNEL_PATH = Path("scripts/fr13_cfwd_logit_direct_decision_kernel.py")
SERVED_PATH = Path("scripts/fr13_device_multidraft_kernel.py")
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
        (1, 116_686_848, 142_560, 116_544_288),
        (4, 466_747_392, 570_240, 466_177_152),
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
    assert contract["fixed_work_for_any_logical_tree_lte"] == 32
    assert contract["vocab_size"] == 151_936
    assert contract["vocab_blocks"] == 594
    assert contract["incumbent_probability_producer_tensor_ops"] == 4
    assert contract["candidate_triton_launch_sites"] == 2
    assert contract["producer_dispatch_sites_removed_static"] == 2
    assert contract["physical_kernel_launches_removed"] == "pending_gpu_trace"
    assert contract["incumbent_full_vocab_materialized_bytes"] == incumbent_bytes
    assert contract["candidate_block_stat_materialized_bytes"] == candidate_bytes
    assert contract["full_vocab_materialized_bytes_removed"] == removed_bytes
    assert contract["candidate_block_stat_workspace_bytes"] == (
        batch_size * 245_760
    )


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
            "vocab_size": 151_935,
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
            (batch_size * 30, 1024),
            torch.float32,
        )
        assert spec["block_sums"] == (
            (batch_size * 30, 1024),
            torch.float32,
        )
        assert spec["self_token"] == ((batch_size, 13), torch.long)
        assert spec["source"] == ((batch_size, 17), torch.long)
        assert spec["accepted"] == ((batch_size, 17), torch.bool)


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


def test_candidate_is_source_only_and_default_off() -> None:
    served_source = SERVED_PATH.read_text(encoding="utf-8")
    assert kernel.CANDIDATE not in served_source
    assert KERNEL_PATH.name not in served_source
    assert kernel.fixed32_cfwd_logit_direct_contract(
        1, mode="tail6_fixed32"
    )["candidate_default_off"] is True


class _FakeKernelLaunch:
    def __init__(self) -> None:
        self.calls = 0

    def __getitem__(self, _grid):
        def launch(*_args, **_kwargs) -> None:
            self.calls += 1

        return launch


def _qualified_cfwd_launch_args() -> dict[str, object]:
    batch = 1
    flat_rows = batch * kernel.PHYSICAL_DRAFTS
    child_counts = torch.zeros(
        (batch, kernel.PHYSICAL_ROWS), dtype=torch.int64
    )
    child_table = torch.full(
        (batch, kernel.PHYSICAL_ROWS, kernel.FANOUT),
        -1,
        dtype=torch.int64,
    )
    return {
        "self_logits": torch.empty(
            (flat_rows, kernel.VOCAB_SIZE), dtype=torch.float32
        ),
        "target_logits": torch.empty(
            (flat_rows, kernel.VOCAB_SIZE), dtype=torch.float32
        ),
        "self_source_indices": torch.arange(
            kernel.SELF_ROWS, dtype=torch.int64
        ),
        "target_source_indices": torch.arange(
            kernel.TARGET_ROWS, dtype=torch.int64
        ),
        "drafts": torch.zeros(
            (batch, kernel.PHYSICAL_DRAFTS), dtype=torch.int64
        ),
        "child_table": child_table,
        "child_counts": child_counts,
        "self_uniform_levels": torch.zeros(
            (kernel.SELF_ROWS,), dtype=torch.int64
        ),
        "target_parent_slots": torch.zeros(
            (kernel.TARGET_ROWS,), dtype=torch.int64
        ),
        "target_uniform_levels": torch.zeros(
            (kernel.TARGET_ROWS,), dtype=torch.int64
        ),
        "uniforms": torch.full(
            (batch, kernel.WALK_CAP, 3), 0.5, dtype=torch.float32
        ),
        "workspace": kernel.allocate_workspace(device="cpu", batch_size=batch),
        "batch_size": batch,
        "mode": "tail6_fixed32",
    }


def _arm_cpu_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeKernelLaunch, _FakeKernelLaunch]:
    stats = _FakeKernelLaunch()
    decisions = _FakeKernelLaunch()
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _self: True))
    monkeypatch.setattr(kernel, "triton", object())
    monkeypatch.setattr(kernel, "tl", object())
    monkeypatch.setattr(
        kernel, "_fr13_cfwd_logit_block_stats_kernel", stats, raising=False
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_cfwd_logit_direct_decision_kernel",
        decisions,
        raising=False,
    )
    return stats, decisions


def test_launch_preflight_accepts_the_exact_qualified_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _qualified_cfwd_launch_args()
    stats, decisions = _arm_cpu_launch(monkeypatch)

    result = kernel.launch_logit_direct_fixed32(**args)

    assert len(result) == 5
    assert stats.calls == 1
    assert decisions.calls == 1


@pytest.mark.parametrize(
    "case",
    (
        "dtype",
        "shape",
        "stride",
        "source_row",
        "parent_slot",
        "uniform_level",
        "child_count",
        "child_packing",
        "draft_token",
        "uniform_range",
        "workspace_stride",
        "workspace_alias",
        "workspace_keys",
    ),
)
def test_launch_preflight_fails_closed_on_pointer_domain_drift(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    args = _qualified_cfwd_launch_args()
    workspace = args["workspace"]
    assert isinstance(workspace, dict)
    if case == "dtype":
        args["target_parent_slots"] = args["target_parent_slots"].to(torch.int32)
    elif case == "shape":
        args["self_source_indices"] = args["self_source_indices"][:-1]
    elif case == "stride":
        args["uniforms"] = torch.empty(
            (1, kernel.WALK_CAP, 6), dtype=torch.float32
        )[..., ::2]
    elif case == "source_row":
        args["self_source_indices"][0] = kernel.PHYSICAL_DRAFTS
    elif case == "parent_slot":
        args["target_parent_slots"][0] = kernel.PHYSICAL_ROWS
    elif case == "uniform_level":
        args["target_uniform_levels"][0] = kernel.WALK_CAP
    elif case == "child_count":
        args["child_counts"][0, 0] = kernel.FANOUT + 1
    elif case == "child_packing":
        args["child_table"][0, 0, 0] = 0
    elif case == "draft_token":
        args["drafts"][0, 0] = kernel.VOCAB_SIZE
    elif case == "uniform_range":
        args["uniforms"][0, 0, 0] = float("nan")
    elif case == "workspace_stride":
        workspace["block_maxima"] = torch.empty(
            (kernel.SELF_ROWS + kernel.TARGET_ROWS, 2 * kernel.MAX_BLOCKS),
            dtype=torch.float32,
        )[:, ::2]
    elif case == "workspace_alias":
        workspace["self_token"] = (
            args["drafts"]
            .reshape(-1)[: kernel.SELF_ROWS]
            .reshape(1, -1)
        )
    elif case == "workspace_keys":
        workspace["unexpected"] = workspace["accepted"]
    else:  # pragma: no cover - parameter list is closed above
        raise AssertionError(case)
    stats, decisions = _arm_cpu_launch(monkeypatch)

    with pytest.raises((TypeError, ValueError)):
        kernel.launch_logit_direct_fixed32(**args)

    assert stats.calls == 0
    assert decisions.calls == 0


def test_launch_preflight_rejects_cross_device_operands_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _qualified_cfwd_launch_args()
    mismatched = args["drafts"]
    stats, decisions = _arm_cpu_launch(monkeypatch)
    monkeypatch.setattr(
        torch.Tensor,
        "device",
        property(
            lambda self: torch.device(
                "cuda:1" if self is mismatched else "cuda:0"
            )
        ),
    )

    with pytest.raises(ValueError, match="share one device"):
        kernel.launch_logit_direct_fixed32(**args)

    assert stats.calls == 0
    assert decisions.calls == 0
