from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fixed32 fused TAW equivalence requires CUDA and Triton",
)

MODULE_PATH = Path("scripts/fr13_device_multidraft_kernel.py")
SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_taw_fused_cuda",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
taw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(taw)

VOCAB = 248_320
SCENARIOS = (
    "accept_leaf",
    "duplicate_accept",
    "duplicate_reject",
    "zero_overlap",
)


def _entry(topology, mode: str, batch_size: int) -> dict[str, object]:
    valid_mask = int(topology.VALID_MASK_BY_MODE[mode])
    taw._fr13_fixed32_test_set_env(topology, mode)
    taw.fr13_fixed32_taw_preseed(
        torch.device("cuda"),
        mode=mode,
        valid_mask=valid_mask,
    )
    key = taw.fr13_fixed32_taw_cache_key(
        mode,
        valid_mask,
        batch_size,
        torch.device("cuda"),
    )
    return taw._FR13_FIXED32_TAW_CACHE[key]


def _production_fixture(
    topology,
    mode: str,
    entry: dict[str, object],
    scenarios: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(scenarios)
    physical_drafts = int(topology.PHYSICAL_DRAFTS)
    rows = batch_size * physical_drafts
    device = torch.device("cuda")

    drafts = (
        torch.arange(1, physical_drafts + 1, dtype=torch.int64, device=device)
        .repeat(batch_size, 1)
        .contiguous()
    )
    bonus = torch.full(
        (batch_size,),
        VOCAB - 2,
        dtype=torch.int64,
        device=device,
    )
    target = torch.full(
        (rows, VOCAB),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    target[:, 0] = 0.0
    self_logits = torch.full_like(target, float("-inf"))
    self_logits[:, VOCAB - 1] = 0.0
    uniforms = torch.full(
        (batch_size, int(topology.WALK_CAP), 3),
        0.1,
        dtype=torch.float32,
        device=device,
    )

    children = topology.active_child_lists(mode)
    for request in range(batch_size):
        start = request * physical_drafts
        for child_nodes in children.values():
            first_child = int(child_nodes[0])
            token = int(drafts[request, first_child])
            target[start + first_child].fill_(float("-inf"))
            target[start + first_child, token] = 0.0

        root_children = tuple(int(node) for node in children[-1])
        root_row = start + root_children[0]
        scenario = scenarios[request]
        if scenario.startswith("duplicate_"):
            for child in root_children:
                drafts[request, child] = 5
            target[root_row].fill_(float("-inf"))
            target[root_row, 5] = math.log(0.4)
            target[root_row, 123] = math.log(0.6)
            uniforms[request, 0, 0] = 0.5
            uniforms[request, 0, 1] = (
                0.2 if scenario == "duplicate_accept" else 0.8
            )
            uniforms[request, 0, 2] = 0.5
        elif scenario == "zero_overlap":
            target[root_row].fill_(float("-inf"))
            target[root_row, 124] = 0.0
            uniforms[request, 0] = torch.tensor(
                [0.5, 0.5, 0.5],
                dtype=torch.float32,
                device=device,
            )
        elif scenario != "accept_leaf":
            raise AssertionError(f"unknown fused TAW scenario: {scenario}")

    entry["draft_tokens"].copy_(drafts)
    entry["bonus_tokens"].copy_(bonus)
    return target, self_logits, uniforms


def _dense_production_fixture(
    topology,
    entry: dict[str, object],
    *,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    physical_drafts = int(topology.PHYSICAL_DRAFTS)
    rows = batch_size * physical_drafts
    walk_cap = int(topology.WALK_CAP)
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    target = torch.randn(
        (rows, VOCAB),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).mul_(0.5)
    self_logits = torch.randn(
        (rows, VOCAB),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).mul_(0.5)
    drafts = (
        (
            torch.arange(
                physical_drafts,
                dtype=torch.int64,
                device=device,
            )
            * 8_191
            + 257
        )
        .remainder(VOCAB)
        .repeat(batch_size, 1)
        .contiguous()
    )
    bonus = torch.arange(
        batch_size,
        dtype=torch.int64,
        device=device,
    ).add_(VOCAB - batch_size - 1)
    uniforms = (
        torch.tensor(
            [0.217, 0.625, 0.731],
            dtype=torch.float32,
            device=device,
        )
        .view(1, 1, 3)
        .expand(batch_size, walk_cap, 3)
        .clone()
    )
    entry["draft_tokens"].copy_(drafts)
    entry["bonus_tokens"].copy_(bonus)
    return target, self_logits, uniforms


def _run_exact_pair(
    topology,
    entry: dict[str, object],
    target: torch.Tensor,
    self_logits: torch.Tensor,
    uniforms: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    kwargs = {
        "walk_cap": int(topology.WALK_CAP),
    }
    reference = taw._fr13_fixed32_taw_execute_torch(
        topology,
        entry,
        entry["draft_tokens"],
        target,
        self_logits,
        entry["bonus_tokens"],
        uniforms,
        **kwargs,
    )
    expected = tuple(tensor.clone() for tensor in reference[:5])
    fused = taw._fr13_fixed32_taw_execute_fused(
        topology,
        entry,
        entry["draft_tokens"],
        target,
        self_logits,
        entry["bonus_tokens"],
        uniforms,
        **kwargs,
    )
    actual = tuple(tensor.clone() for tensor in fused[:5])
    torch.cuda.synchronize()
    return expected, actual


def _assert_fixture_exercised(
    topology,
    mode: str,
    scenarios: tuple[str, ...],
    expected: tuple[torch.Tensor, ...],
) -> None:
    output, output_lens, paths, path_lens, last_row = expected
    root_children = tuple(int(node) for node in topology.active_child_lists(mode)[-1])
    for request, scenario in enumerate(scenarios):
        if scenario == "accept_leaf":
            assert int(path_lens[request]) > 0
            assert int(output_lens[request]) == int(path_lens[request]) + 1
            assert int(output[request, output_lens[request] - 1]) == VOCAB - 1
            assert int(last_row[request]) == int(
                paths[request, path_lens[request] - 1]
            )
        elif scenario == "duplicate_accept":
            assert int(output[request, 0]) == 5
            assert int(path_lens[request]) > 0
            assert int(paths[request, 0]) == root_children[1] + 1
            assert int(last_row[request]) > 0
        elif scenario == "duplicate_reject":
            assert int(output[request, 0]) == 123
            assert int(output_lens[request]) == 1
            assert int(path_lens[request]) == 0
            assert int(last_row[request]) == 0
        elif scenario == "zero_overlap":
            assert int(output[request, 0]) == 124
            assert int(output_lens[request]) == 1
            assert int(path_lens[request]) == 0
            assert int(last_row[request]) == 0


@pytest.mark.parametrize("mode", ("tail6_fixed32", "hydra27_fixed32"))
@pytest.mark.parametrize("batch_size", (1, 4))
def test_fused_taw_exact_products_at_deployed_vocab(
    mode: str,
    batch_size: int,
) -> None:
    topology = taw._fr13_fixed32_topology()
    entry = _entry(topology, mode, batch_size)
    scenario_groups = (
        tuple((scenario,)) for scenario in SCENARIOS
    ) if batch_size == 1 else (SCENARIOS,)

    for scenarios in scenario_groups:
        target, self_logits, uniforms = _production_fixture(
            topology,
            mode,
            entry,
            scenarios,
        )
        expected, actual = _run_exact_pair(
            topology,
            entry,
            target,
            self_logits,
            uniforms,
        )
        _assert_fixture_exercised(topology, mode, scenarios, expected)
        for name, reference, candidate in zip(
            (
                "output_tokens",
                "output_lens",
                "accepted_path_rows",
                "accepted_lens",
                "last_row",
            ),
            expected,
            actual,
            strict=True,
        ):
            assert torch.equal(candidate, reference), (
                f"{mode}/B{batch_size}/{scenarios}: {name} mismatch\n"
                f"reference={reference}\ncandidate={candidate}"
            )


@pytest.mark.parametrize("mode", ("tail6_fixed32", "hydra27_fixed32"))
@pytest.mark.parametrize("batch_size", (1, 4))
def test_fused_taw_dense_finite_products_at_deployed_vocab(
    mode: str,
    batch_size: int,
) -> None:
    topology = taw._fr13_fixed32_topology()
    entry = _entry(topology, mode, batch_size)
    mode_seed = 6 if mode == "tail6_fixed32" else 27
    target, self_logits, uniforms = _dense_production_fixture(
        topology,
        entry,
        batch_size=batch_size,
        seed=2026073100 + mode_seed * 10 + batch_size,
    )
    expected, actual = _run_exact_pair(
        topology,
        entry,
        target,
        self_logits,
        uniforms,
    )

    assert torch.isfinite(target).all()
    assert torch.isfinite(self_logits).all()
    assert torch.all((uniforms > 0.2) & (uniforms < 0.8))
    assert int(expected[1].min()) == 1
    assert int(expected[1].max()) == 1
    assert int(expected[0][:, 0].min()) >= taw._FR13_FIXED32_TAW_CHUNK_SIZE
    for name, reference, candidate in zip(
        (
            "output_tokens",
            "output_lens",
            "accepted_path_rows",
            "accepted_lens",
            "last_row",
        ),
        expected,
        actual,
        strict=True,
    ):
        assert torch.equal(candidate, reference), (
            f"dense {mode}/B{batch_size}: {name} mismatch\n"
            f"reference={reference}\ncandidate={candidate}"
        )
