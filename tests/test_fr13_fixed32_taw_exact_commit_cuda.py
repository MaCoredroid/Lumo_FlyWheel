from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fixed32 exact commit byte gate requires CUDA and Triton",
)

MODULE_PATH = Path("scripts/fr13_device_multidraft_kernel.py")
SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_taw_exact_commit_cuda",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
taw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(taw)

VOCAB = 248_320
PRODUCT_NAMES = (
    "output_tokens",
    "output_lens",
    "accepted_path_rows",
    "accepted_lens",
    "last_row",
)
THRESHOLD_CASES = (
    "source_lo",
    "source_hi",
    "accept_lo",
    "accept_hi",
    "residual_lo",
    "residual_hi",
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


def _fixture(
    topology,
    mode: str,
    entry: dict[str, object],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    physical_drafts = int(topology.PHYSICAL_DRAFTS)
    rows = batch_size * physical_drafts
    device = torch.device("cuda")
    candidate_tokens = (101, 60_001, 180_001)
    residual_tokens = (20_001, 220_001)

    drafts = (
        torch.arange(
            physical_drafts,
            dtype=torch.int64,
            device=device,
        )
        .add_(1_000)
        .repeat(batch_size, 1)
        .contiguous()
    )
    bonus = torch.arange(
        batch_size,
        dtype=torch.int64,
        device=device,
    ).add_(VOCAB - batch_size - 1)
    target = torch.full(
        (rows, VOCAB),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    target[:, 0] = 0.0
    self_logits = torch.full_like(target, float("-inf"))
    for request in range(batch_size):
        self_logits[
            request * physical_drafts : (request + 1) * physical_drafts,
            VOCAB - request - 1,
        ] = 0.0

    children = topology.active_child_lists(mode)
    for request in range(batch_size):
        start = request * physical_drafts
        for child_nodes in children.values():
            nodes = tuple(int(node) for node in child_nodes)
            row = start + nodes[0]
            target[row].fill_(float("-inf"))
            for lane, child in enumerate(nodes):
                token = candidate_tokens[lane]
                drafts[request, child] = token
                target[row, token] = math.log(0.25)
            target[row, residual_tokens[0]] = math.log(0.15)
            target[row, residual_tokens[1]] = math.log(
                0.10 + 0.25 * (3 - len(nodes))
            )

    uniforms = torch.full(
        (batch_size, int(topology.WALK_CAP), 3),
        0.1,
        dtype=torch.float32,
        device=device,
    )
    entry["draft_tokens"].copy_(drafts)
    entry["bonus_tokens"].copy_(bonus)
    return target, self_logits, uniforms


def _run_pair(
    topology,
    entry: dict[str, object],
    target: torch.Tensor,
    self_logits: torch.Tensor,
    uniforms: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    kwargs = {"walk_cap": int(topology.WALK_CAP)}
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
    candidate = taw._fr13_fixed32_taw_execute_exact_cuda(
        topology,
        entry,
        entry["draft_tokens"],
        target,
        self_logits,
        entry["bonus_tokens"],
        uniforms,
        **kwargs,
    )
    actual = tuple(tensor.clone() for tensor in candidate[:5])
    torch.cuda.synchronize()
    for name, expected_tensor, actual_tensor in zip(
        PRODUCT_NAMES,
        expected,
        actual,
        strict=True,
    ):
        assert torch.equal(actual_tensor, expected_tensor), (
            f"{name} byte mismatch\n"
            f"reference={expected_tensor}\ncandidate={actual_tensor}"
        )
    return expected


def _set_threshold_case(
    topology,
    mode: str,
    entry: dict[str, object],
    target: torch.Tensor,
    uniforms: torch.Tensor,
    request: int,
    case: str,
) -> None:
    root_children = tuple(
        int(node) for node in topology.active_child_lists(mode)[-1]
    )
    first_child = root_children[0]
    row = request * int(topology.PHYSICAL_DRAFTS) + first_child
    target_prob = torch.softmax(target[row].to(torch.float32), dim=-1)
    target_prob = target_prob / target_prob.sum()
    kid_tokens = entry["draft_tokens"][request, list(root_children)]
    overlaps = target_prob[kid_tokens]
    overlap_mass = overlaps.sum()
    source_cdf = torch.cumsum(overlaps, dim=-1)
    source_boundary = source_cdf[0] / source_cdf[-1]
    selected_token = kid_tokens[0]
    same_token = kid_tokens == selected_token
    q_mix_token = (overlaps * same_token).sum() / overlap_mass.clamp(min=1e-30)
    accept_probability = (
        target_prob[selected_token] / q_mix_token.clamp(min=1e-30)
    ).clamp(max=1.0)

    weights = overlaps / overlap_mass
    q_mix = torch.zeros_like(target_prob)
    q_mix.scatter_add_(0, kid_tokens, weights)
    residual = (target_prob - q_mix).clamp(min=0)
    residual_mass = residual.sum()
    residual = torch.where(
        residual_mass > 0,
        residual / residual_mass.clamp(min=1e-30),
        target_prob,
    )
    residual_cdf = torch.cumsum(residual, dim=-1)
    residual_boundary = residual_cdf[20_001] / residual_cdf[-1]

    zero = torch.zeros((), dtype=torch.float32, device=target.device)
    one = torch.ones((), dtype=torch.float32, device=target.device)
    uniforms[request, 0] = torch.tensor(
        [0.1, 0.1, 0.1],
        dtype=torch.float32,
        device=target.device,
    )
    if case == "source_lo":
        uniforms[request, 0, 0] = torch.nextafter(source_boundary, zero)
    elif case == "source_hi":
        uniforms[request, 0, 0] = torch.nextafter(source_boundary, one)
    elif case == "accept_lo":
        uniforms[request, 0, 1] = torch.nextafter(accept_probability, zero)
    elif case == "accept_hi":
        uniforms[request, 0, 1] = torch.nextafter(accept_probability, one)
    elif case == "residual_lo":
        uniforms[request, 0, 1] = one
        uniforms[request, 0, 2] = torch.nextafter(residual_boundary, zero)
    elif case == "residual_hi":
        uniforms[request, 0, 1] = one
        uniforms[request, 0, 2] = torch.nextafter(residual_boundary, one)
    else:
        raise AssertionError(f"unknown threshold case: {case}")


@pytest.mark.parametrize("mode", ("tail6_fixed32", "hydra27_fixed32"))
@pytest.mark.parametrize("batch_size", (1, 4))
def test_exact_commit_products_and_threshold_boundaries(
    mode: str,
    batch_size: int,
) -> None:
    topology = taw._fr13_fixed32_topology()
    entry = _entry(topology, mode, batch_size)
    target, self_logits, uniforms = _fixture(
        topology,
        mode,
        entry,
        batch_size=batch_size,
    )

    ordinary = _run_pair(topology, entry, target, self_logits, uniforms)
    assert torch.all(ordinary[3] > 0)
    assert torch.equal(ordinary[1], ordinary[3] + 1)
    assert torch.equal(ordinary[4], ordinary[2].gather(
        1,
        (ordinary[3] - 1).unsqueeze(1),
    ).squeeze(1))

    observed: dict[str, tuple[torch.Tensor, ...]] = {}
    groups = (
        tuple((case,) for case in THRESHOLD_CASES)
        if batch_size == 1
        else (
            THRESHOLD_CASES[:4],
            THRESHOLD_CASES[4:] + THRESHOLD_CASES[:2],
        )
    )
    for cases in groups:
        uniforms.fill_(0.1)
        for request, case in enumerate(cases):
            _set_threshold_case(
                topology,
                mode,
                entry,
                target,
                uniforms,
                request,
                case,
            )
        expected = _run_pair(topology, entry, target, self_logits, uniforms)
        for request, case in enumerate(cases):
            observed[case] = tuple(tensor[request].clone() for tensor in expected)

    assert not torch.equal(observed["source_lo"][2], observed["source_hi"][2])
    assert int(observed["accept_lo"][3]) > 0
    assert int(observed["accept_hi"][3]) == 0
    assert int(observed["residual_lo"][1]) == 1
    assert int(observed["residual_hi"][1]) == 1
    assert not torch.equal(
        observed["residual_lo"][0],
        observed["residual_hi"][0],
    )
