"""Byte-equivalence + wiring for the default-off batched slot publish.

The transformation under test replaces the fixed32 commit route's per-row slot
publish loop with two ``index_copy_`` calls sourced from the compact rows. The
claim is that it writes IDENTICAL BYTES into both buffer families and therefore
identical committed conv state. These tests prove that on synthetic states for
every legal batch, every slot permutation and BOTH mamba-narrowing regimes, and
pin the launch arithmetic the lever is justified by.

CPU only. No GPU, no container, no serving image.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from lumo_flywheel_serving.fr13_fixed32_commit_slot_scatter import (
    FLAG_ENV,
    SLOT_SCATTER_INDEX_CACHE_LIMIT,
    assert_batched_slots_requires_fixed32,
    batched_slot_index_build_launches,
    batched_slot_launch_census,
    clear_slot_scatter_index_cache,
    publish_committer_paths,
    resolve_batched_slots,
    slot_scatter_index,
    slot_scatter_index_cache_size,
)

PATCHER_PATH = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py")

PATH_COLS = 16
SLOT_ROWS = 4
LAYERS = 3          # a 3-layer stand-in for the served 48; the publish block
                    # is layer-agnostic and 48 buys nothing but runtime.
CONV_ROWS = 32      # fixed32 physical rows
CONV_C = 8
CONV_L = 6
SPEC_COLS = 16


# --------------------------------------------------------------------------
# synthetic state
# --------------------------------------------------------------------------


def _make_buffers(batch: int, *, seed: int):
    """Persistent publish buffers plus this event's device products.

    Slot buffers are pre-poisoned with a recognisable pattern so a test can
    tell "written correctly" from "left alone" from "clobbered".
    """
    gen = torch.Generator().manual_seed(seed)
    device_paths = torch.randint(
        0, SPEC_COLS, (batch, PATH_COLS), generator=gen, dtype=torch.int64
    )
    device_lens = torch.randint(
        0, PATH_COLS + 1, (batch,), generator=gen, dtype=torch.int64
    )
    slot_paths = torch.full((SLOT_ROWS, PATH_COLS), -7, dtype=torch.int32)
    slot_lens = torch.full((SLOT_ROWS,), -7, dtype=torch.int32)
    spec_paths = torch.full((SLOT_ROWS, PATH_COLS), -9, dtype=torch.int32)
    spec_lens = torch.full((SLOT_ROWS,), -9, dtype=torch.int32)
    return (
        slot_paths,
        slot_lens,
        spec_paths,
        spec_lens,
        device_paths,
        device_lens,
    )


def _spec_state_indices(*, narrowed: bool) -> torch.Tensor:
    """(LAYERS, SLOT_ROWS, SPEC_COLS) bank-row table for both regimes.

    OFF  -- every logical column addresses its own physical row.
    ON   -- FR13_MAMBA_SPEC_BLOCKS_CDIV republishes ONE scratch page across
            logical columns 1..num_spec, so col0 is the running row and every
            other column is the SAME aliased row. That is the shape the served
            path has carried since the 2026-08-10 promotion, and it is the one
            a publish change is most likely to break, because a mistaken slot
            row now collides with a shared page instead of a private one.
    """
    table = torch.zeros((LAYERS, SLOT_ROWS, SPEC_COLS), dtype=torch.int32)
    scratch_base = SLOT_ROWS * 2  # rows [0, 8) are the running rows
    for layer in range(LAYERS):
        for row in range(SLOT_ROWS):
            running = row * 2
            table[layer, row, 0] = running
            if narrowed:
                # ONE shared scratch page across every spec column.
                table[layer, row, 1:] = scratch_base + layer
            else:
                for col in range(1, SPEC_COLS):
                    table[layer, row, col] = scratch_base + (
                        (layer * SLOT_ROWS * SPEC_COLS + row * SPEC_COLS + col)
                        % (CONV_ROWS - scratch_base)
                    )
    if int(table.max()) >= CONV_ROWS or int(table.min()) < 0:
        raise AssertionError("synthetic ssi escaped the physical bank")
    return table


def _reference_conv_commit(
    *,
    conv_banks: torch.Tensor,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    batch: int,
) -> torch.Tensor:
    """CPU model of ``_fr13_fixed32_conv_direct_col0_kernel``'s addressing.

    Leaf node = ``accepted_paths[b, clamp(len-1, 0)]`` (zero-accept rows commit
    the root, node column 0); destination = ``spec_state_indices[layer, b, 0]``
    -- col0-only, which is what ``route=fixed32_direct_source_col0`` means.
    The arithmetic is a copy, so the model only has to reproduce WHICH rows move
    to be a faithful detector of a publish regression.
    """
    out = conv_banks.clone()
    for layer in range(int(spec_state_indices.shape[0])):
        for row in range(batch):
            alen = int(accepted_lens[row])
            leaf_pos = max(alen - 1, 0)
            leaf_node = (
                int(accepted_paths[row, leaf_pos]) if alen > 0 else 0
            )
            leaf_node = max(0, min(leaf_node, SPEC_COLS - 1))
            src = int(spec_state_indices[layer, row, leaf_node])
            dst = int(spec_state_indices[layer, row, 0])
            out[layer, dst] = conv_banks[layer, src]
    return out


class _AtenLaunchCounter(TorchDispatchMode):
    """Counts dispatched ATen ops -- the CPU-visible proxy for a kernel launch.

    Views and metadata calls do not dispatch a kernel on any backend, so the
    count is the launch count the serving path pays, not a Python statement
    count.
    """

    def __init__(self) -> None:
        self.ops: list[str] = []

    # aten op names dispatch as "aten.<base>.<overload>"; the view family
    # returns a new TensorImpl over the same storage and launches nothing.
    _VIEW_BASES = frozenset(
        (
            "select", "slice", "view", "detach", "alias", "as_strided",
            "_unsafe_view", "expand", "permute", "t", "transpose",
            "unsqueeze", "squeeze", "reshape", "narrow",
        )
    )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        parts = name.split(".")
        base = parts[1] if len(parts) > 1 else name
        if base not in self._VIEW_BASES:
            self.ops.append(name)
        return func(*args, **(kwargs or {}))


# --------------------------------------------------------------------------
# byte equivalence -- the load-bearing test
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_index_cache():
    clear_slot_scatter_index_cache()
    yield
    clear_slot_scatter_index_cache()


def _all_slot_maps(batch: int):
    return [
        tuple(perm)
        for perm in itertools.permutations(range(SLOT_ROWS), batch)
    ]


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("narrowed", [False, True])
def test_batched_publish_is_byte_identical_to_the_loop(batch, narrowed):
    """Both publish forms leave every buffer byte-identical, both regimes.

    The narrowing regime is carried through explicitly even though the publish
    block never reads the spec table: "this lever is orthogonal to
    FR13_MAMBA_SPEC_BLOCKS_CDIV" is a claim, and a claim in this campaign gets
    a test. The regime IS read by the committed-state test below.
    """
    ssi = _spec_state_indices(narrowed=narrowed)
    aliased_cols = int(
        (ssi[:, :, 1:] == ssi[:, :, 1:2]).all().item()
    )
    assert aliased_cols == int(narrowed)
    for slot_map in _all_slot_maps(batch):
        seq = _make_buffers(batch, seed=1000 + batch * 10 + int(narrowed))
        bat = _make_buffers(batch, seed=1000 + batch * 10 + int(narrowed))

        publish_committer_paths(
            slot_paths=seq[0], slot_lens=seq[1],
            spec_paths=seq[2], spec_lens=seq[3],
            device_paths=seq[4], device_lens=seq[5],
            slot_indices=slot_map, batch=batch, batched=False,
        )
        publish_committer_paths(
            slot_paths=bat[0], slot_lens=bat[1],
            spec_paths=bat[2], spec_lens=bat[3],
            device_paths=bat[4], device_lens=bat[5],
            slot_indices=slot_map, batch=batch, batched=True,
        )

        for name, left, right in (
            ("slot_paths", seq[0], bat[0]),
            ("slot_lens", seq[1], bat[1]),
            ("spec_paths", seq[2], bat[2]),
            ("spec_lens", seq[3], bat[3]),
        ):
            assert left.dtype == right.dtype, name
            assert (
                left.numpy().tobytes() == right.numpy().tobytes()
            ), f"{name} byte drift at batch={batch} slots={slot_map}"

        # untouched slot rows must still carry the poison, in BOTH forms:
        # the sparse-write contract is part of byte identity.
        for row in range(SLOT_ROWS):
            if row in slot_map:
                continue
            assert int(seq[1][row]) == -7
            assert int(bat[1][row]) == -7


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("narrowed", [False, True])
def test_committed_conv_state_is_byte_identical(batch, narrowed):
    """The downstream commit, not just the buffers, is byte-identical.

    Buffers matching is necessary but not sufficient: the point of the publish
    is what the conv committer then reads out of it. This drives the reference
    committer off each publish and compares the resulting conv banks.
    """
    ssi = _spec_state_indices(narrowed=narrowed)
    banks = torch.arange(
        LAYERS * CONV_ROWS * CONV_C * CONV_L, dtype=torch.float32
    ).reshape(LAYERS, CONV_ROWS, CONV_C * CONV_L)

    for slot_map in _all_slot_maps(batch):
        seq = _make_buffers(batch, seed=77 + batch)
        bat = _make_buffers(batch, seed=77 + batch)
        publish_committer_paths(
            slot_paths=seq[0], slot_lens=seq[1],
            spec_paths=seq[2], spec_lens=seq[3],
            device_paths=seq[4], device_lens=seq[5],
            slot_indices=slot_map, batch=batch, batched=False,
        )
        publish_committer_paths(
            slot_paths=bat[0], slot_lens=bat[1],
            spec_paths=bat[2], spec_lens=bat[3],
            device_paths=bat[4], device_lens=bat[5],
            slot_indices=slot_map, batch=batch, batched=True,
        )
        committed_seq = _reference_conv_commit(
            conv_banks=banks, spec_state_indices=ssi,
            accepted_paths=seq[2], accepted_lens=seq[3], batch=batch,
        )
        committed_bat = _reference_conv_commit(
            conv_banks=banks, spec_state_indices=ssi,
            accepted_paths=bat[2], accepted_lens=bat[3], batch=batch,
        )
        assert (
            committed_seq.numpy().tobytes()
            == committed_bat.numpy().tobytes()
        ), f"conv commit byte drift batch={batch} slots={slot_map}"


def test_zero_accept_rows_commit_the_root_in_both_forms():
    """acc_len == 0 must still round-trip: the root-column path is the one a
    sloppy publish silently breaks, because its leaf index is synthesised."""
    batch = 4
    ssi = _spec_state_indices(narrowed=True)
    banks = torch.arange(
        LAYERS * CONV_ROWS * CONV_C * CONV_L, dtype=torch.float32
    ).reshape(LAYERS, CONV_ROWS, CONV_C * CONV_L)
    seq = _make_buffers(batch, seed=5)
    bat = _make_buffers(batch, seed=5)
    seq[5].zero_()
    bat[5].zero_()
    slot_map = (3, 1, 0, 2)
    publish_committer_paths(
        slot_paths=seq[0], slot_lens=seq[1], spec_paths=seq[2],
        spec_lens=seq[3], device_paths=seq[4], device_lens=seq[5],
        slot_indices=slot_map, batch=batch, batched=False,
    )
    publish_committer_paths(
        slot_paths=bat[0], slot_lens=bat[1], spec_paths=bat[2],
        spec_lens=bat[3], device_paths=bat[4], device_lens=bat[5],
        slot_indices=slot_map, batch=batch, batched=True,
    )
    assert (seq[3] == 0).all()
    assert seq[3].numpy().tobytes() == bat[3].numpy().tobytes()
    assert seq[0].numpy().tobytes() == bat[0].numpy().tobytes()
    left = _reference_conv_commit(
        conv_banks=banks, spec_state_indices=ssi, accepted_paths=seq[2],
        accepted_lens=seq[3], batch=batch,
    )
    right = _reference_conv_commit(
        conv_banks=banks, spec_state_indices=ssi, accepted_paths=bat[2],
        accepted_lens=bat[3], batch=batch,
    )
    assert left.numpy().tobytes() == right.numpy().tobytes()


# --------------------------------------------------------------------------
# launch arithmetic -- the reason the lever exists
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_measured_launch_counts_match_the_census(batch):
    """The census function is the claim; this measures the code against it."""
    slot_map = tuple(range(batch))
    for batched in (False, True):
        buffers = _make_buffers(batch, seed=42)
        if batched:  # warm the index cache: steady state is a cache hit
            slot_scatter_index(slot_map, buffers[2].device)
        counter = _AtenLaunchCounter()
        with counter:
            publish_committer_paths(
                slot_paths=buffers[0], slot_lens=buffers[1],
                spec_paths=buffers[2], spec_lens=buffers[3],
                device_paths=buffers[4], device_lens=buffers[5],
                slot_indices=slot_map, batch=batch, batched=batched,
            )
        expected = batched_slot_launch_census(batch=batch, batched=batched)
        assert len(counter.ops) == expected, (
            f"batched={batched} batch={batch}: measured {counter.ops}"
        )


def test_census_is_the_documented_saving():
    assert [
        batched_slot_launch_census(batch=b, batched=False) for b in (1, 2, 3, 4)
    ] == [4, 6, 8, 10]
    assert [
        batched_slot_launch_census(batch=b, batched=True) for b in (1, 2, 3, 4)
    ] == [4, 4, 4, 4]
    # B1 is explicitly a NULL for this lever. Stating it as a test stops a
    # future reader from quoting a B4 saving as a B1 one.
    assert batched_slot_launch_census(
        batch=1, batched=True
    ) == batched_slot_launch_census(batch=1, batched=False)
    assert batched_slot_index_build_launches(cache_hit=True) == 0
    assert batched_slot_index_build_launches(cache_hit=False) == 1


def test_index_cache_amortises_and_revalidates():
    device = torch.device("cpu")
    first = slot_scatter_index((2, 0, 1), device)
    second = slot_scatter_index((2, 0, 1), device)
    assert first is second
    assert slot_scatter_index_cache_size() == 1
    slot_scatter_index((0, 1, 2), device)
    assert slot_scatter_index_cache_size() == 2
    # a corrupted cache entry must raise, not be handed out
    first.fill_(0)
    with pytest.raises(RuntimeError, match="cache drift"):
        slot_scatter_index((2, 0, 1), device)


# --------------------------------------------------------------------------
# guards -- every one of these is a way the lever could ship a lie
# --------------------------------------------------------------------------


def test_flag_is_strict_and_defaults_off():
    assert resolve_batched_slots({}) is False
    assert resolve_batched_slots({FLAG_ENV: "0"}) is False
    assert resolve_batched_slots({FLAG_ENV: "1"}) is True
    assert resolve_batched_slots({FLAG_ENV: " 1 "}) is True
    for bad in ("true", "yes", "2", "", "01", "on"):
        with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
            resolve_batched_slots({FLAG_ENV: bad})


def test_flag_refuses_to_arm_outside_fixed32():
    assert_batched_slots_requires_fixed32(enabled=False, fixed32_mode=None)
    assert_batched_slots_requires_fixed32(
        enabled=True, fixed32_mode="tail6_fixed32"
    )
    assert_batched_slots_requires_fixed32(
        enabled=True, fixed32_mode="hydra27_fixed32"
    )
    for mode in (None, "", "tail6", "hydra27", "fixed32", 0):
        with pytest.raises(RuntimeError, match="requires fixed32"):
            assert_batched_slots_requires_fixed32(
                enabled=True, fixed32_mode=mode
            )


def test_batched_publish_refuses_aliased_storage():
    """The reorder is only observation-free on disjoint storage."""
    batch = 2
    buffers = list(_make_buffers(batch, seed=3))
    buffers[0] = buffers[2]  # slot_paths aliases spec_paths
    with pytest.raises(RuntimeError, match="disjoint"):
        publish_committer_paths(
            slot_paths=buffers[0], slot_lens=buffers[1],
            spec_paths=buffers[2], spec_lens=buffers[3],
            device_paths=buffers[4], device_lens=buffers[5],
            slot_indices=(0, 1), batch=batch, batched=True,
        )


def test_publish_refuses_duplicate_or_out_of_range_slots():
    batch = 2
    for slot_map in ((1, 1), (0, SLOT_ROWS), (0, -1)):
        for batched in (False, True):
            buffers = _make_buffers(batch, seed=4)
            with pytest.raises(RuntimeError, match="slot-map drift"):
                publish_committer_paths(
                    slot_paths=buffers[0], slot_lens=buffers[1],
                    spec_paths=buffers[2], spec_lens=buffers[3],
                    device_paths=buffers[4], device_lens=buffers[5],
                    slot_indices=slot_map, batch=batch, batched=batched,
                )


def test_publish_refuses_geometry_drift():
    buffers = _make_buffers(2, seed=6)
    with pytest.raises(RuntimeError, match="slot map width"):
        publish_committer_paths(
            slot_paths=buffers[0], slot_lens=buffers[1],
            spec_paths=buffers[2], spec_lens=buffers[3],
            device_paths=buffers[4], device_lens=buffers[5],
            slot_indices=(0,), batch=2, batched=True,
        )
    with pytest.raises(ValueError, match=r"batch must be an int in \[1, 4\]"):
        publish_committer_paths(
            slot_paths=buffers[0], slot_lens=buffers[1],
            spec_paths=buffers[2], spec_lens=buffers[3],
            device_paths=buffers[4], device_lens=buffers[5],
            slot_indices=(0, 1), batch=5, batched=True,
        )


def test_slot_index_rejects_duplicates_and_bad_input():
    device = torch.device("cpu")
    with pytest.raises(ValueError, match="distinct"):
        slot_scatter_index((0, 0), device)
    with pytest.raises(ValueError, match="negative"):
        slot_scatter_index((0, -1), device)
    with pytest.raises(ValueError, match="empty"):
        slot_scatter_index((), device)
    with pytest.raises(TypeError):
        slot_scatter_index(0, device)


def test_index_cache_limit_is_bounded():
    assert SLOT_SCATTER_INDEX_CACHE_LIMIT == 512


# --------------------------------------------------------------------------
# flag threading through the patcher
# --------------------------------------------------------------------------


def _patcher_tree() -> ast.Module:
    return ast.parse(PATCHER_PATH.read_text())


def _fixed_route_source(tree: ast.Module) -> str:
    sources = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "def _fr13_fixed32_device_commit_route" in node.value
    ]
    assert len(sources) == 1
    return sources[0]


def test_patcher_declares_the_flag_and_a_strict_resolver():
    text = PATCHER_PATH.read_text()
    assert '"FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS", "0"' in text
    tree = _patcher_tree()
    names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "_fr13_fixed32_conv_commit_batched_slots" in names
    assert (
        "_fr13_assert_fixed32_conv_commit_batched_slots_requires_fixed32"
        in names
    )


def test_route_calls_the_module_and_keeps_no_per_row_loop():
    source = _fixed_route_source(_patcher_tree())
    assert "publish_committer_paths" in source
    assert "slot_paths[slot_row].copy_" not in source
    assert "for compact_row, slot_row in enumerate(slot_indices)" not in source
    assert "_FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS = False" in source
    route = ast.parse(source)
    fn = [
        node
        for node in route.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_device_commit_route"
    ]
    assert len(fn) == 1
    calls = [
        node.func.id
        for node in ast.walk(fn[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("_fr13_f32_publish_paths") == 1
    # publish must still precede the conv commit and the replay.
    assert calls.index("_fr13_f32_publish_paths") < calls.index(
        "_fixed_conv_commit"
    )
    assert calls.index("_fixed_conv_commit") < calls.index("_fixed_replay")


def test_injected_helper_compiles_with_and_without_the_prelude():
    """The injected source must compile in BOTH shapes it is ever seen in.

    Production prepends a prelude that binds the flag as a literal; a bare exec
    (self-test, and any future offline audit) does not. The NameError guard is
    what makes the second shape legal, and a guard nobody compiles is a guard
    nobody has.
    """
    source = _fixed_route_source(_patcher_tree())
    compile(source, "<injected-bare>", "exec")
    prelude = (
        "_FR13_FIXED32_MODE = 'tail6_fixed32'\n"
        "_FR13_FIXED32_VALID_MASK = 0\n"
        "_FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS = True\n"
    )
    compile(prelude + source, "<injected-prelude>", "exec")
    # the guard must sit BEFORE the route that reads the name
    assert source.index("_FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS = False") < (
        source.index("def _fr13_fixed32_device_commit_route")
    )


def test_prelude_bakes_the_flag_and_the_preflight_asserts_it():
    text = PATCHER_PATH.read_text()
    assert (
        '"_FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS = "' in text
        and "{_fr13_fixed32_conv_commit_batched_slots()!r}" in text
    )
    tree = _patcher_tree()
    main = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    assert len(main) == 1
    main_calls = {
        node.func.id
        for node in ast.walk(main[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert (
        "_fr13_assert_fixed32_conv_commit_batched_slots_requires_fixed32"
        in main_calls
    )
