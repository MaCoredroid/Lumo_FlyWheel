"""The padded-B4 width-3 path: shadow inertness, staging lifecycle, scope.

MARK'S RULING 2026-08-13 accepts padded-B4. The sealed .so serves ONE
compile-time grid and ONE canonical geometry (b == 4, total_q == 128), so a
width-3 FULL graph reaches it by PADDING -- one inert shadow request in slot 3
carrying zero key rows -- and never by widening the kernel.

These tests pin the four things that make that safe:

  1. THE SHADOW IS A NO-OP. Modelled on CPU against the kernel's documented
     early-return semantics: at seqused_k == 0 compute_attn_1rowblock_splitkv
     takes the n_block_min >= n_block_max exit (flash_fwd_kernel.h:759), which
     is BEFORE Q is read and BEFORE the block table is formed
     (flash_fwd_kernel.h:872). It writes zeros to its O rows and +INF to its
     LSE entries and returns. So poisoning the shadow's Q rows and its
     block-table row must not move a single real byte.
  2. THE STAGING LIFECYCLE. Allocation happens only from the pre-capture hook;
     the serving path looks up and raises; allocating during capture is
     refused outright.
  3. THE FIVE b-DEPENDENT GEOMETRY CLAUSES, at width 3 and at width 4.
  4. THE SCOPE TOKEN ROUND-TRIP: writers emit the b34 token, readers accept
     the sealed b4 token too.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts/fr13_patch_fa2_tree_bias.py"
GQA_GATE = REPO / "scripts/fr13_fa2_qrow32_gqa_pair_gate.py"
SIDECAR = REPO / "scripts/fr13_qrow32_b4_pass_sidecar.py"
PAIR_REDUCE = REPO / "scripts/fr13_b4_gqa_width4_pair_reduce.py"
LAUNCHER = REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"
INNER_RUNNER = REPO / "scripts/fr13_run_b4_fa2_qrow32_live_gate.sh"
B34_RUNNER = REPO / "scripts/fr13_run_b34_fa2_qrow32_gqa_pair_live_gate.sh"
B4_TIMING_RUNNER = REPO / "scripts/fr13_run_b4_fa2_qrow32_gqa_pair_timing.sh"
FR10_PATCHER = REPO / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"

SEALED_SCOPE = "final_fixed32_b4_full_graph_only"
B34_SCOPE = "final_fixed32_b34_full_graph_only"
TARGET_LAYERS = tuple(
    f"language_model.model.layers.{index}.self_attn.attn"
    for index in range(3, 64, 4)
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1. The shadow is a source-level no-op, modelled on CPU
# --------------------------------------------------------------------------


NEG_INF = float("-inf")
POS_INF = float("inf")


def _paged_attention_request(query, pages, block_row, seqused_k, scale):
    """One request, exactly as the kernel treats it -- early return included.

    Returns (out[rows, heads, dim] bf16, lse[heads, rows] fp32).

    The early return is the whole point. `if seqused_k == 0` here stands for
    the kernel's `n_block_min >= n_block_max` at flash_fwd_kernel.h:759: it
    fires BEFORE Q is loaded and BEFORE the block table is dereferenced, so
    neither operand can influence the result, and the only writes are the
    zero-fill of O and the +INF fill of LSE.
    """
    rows, heads, dim = query.shape
    if int(seqused_k) == 0:
        return (
            torch.zeros((rows, heads, dim), dtype=torch.bfloat16),
            torch.full((heads, rows), POS_INF, dtype=torch.float32),
        )
    page_size = pages.shape[1]
    gathered = torch.cat(
        [pages[int(page)] for page in block_row.tolist()], dim=0
    )[: int(seqused_k)]
    keys = gathered.to(torch.float32)
    q = query.to(torch.float32)
    out = torch.zeros((rows, heads, dim), dtype=torch.float32)
    lse = torch.zeros((heads, rows), dtype=torch.float32)
    for head in range(heads):
        scores = (q[:, head, :] @ keys.t()) * scale
        maximum = scores.max(dim=1, keepdim=True).values
        weights = torch.exp(scores - maximum)
        total = weights.sum(dim=1, keepdim=True)
        out[:, head, :] = (weights / total) @ keys
        lse[head, :] = (maximum.squeeze(1) + torch.log(total.squeeze(1)))
    assert page_size > 0
    return out.to(torch.bfloat16), lse


def _paged_attention_batch(query, pages, block_table, seqused_k, rows, scale):
    outs, lses = [], []
    for slot in range(block_table.shape[0]):
        begin = slot * rows
        out, lse = _paged_attention_request(
            query[begin : begin + rows],
            pages,
            block_table[slot],
            int(seqused_k[slot]),
            scale,
        )
        outs.append(out)
        lses.append(lse)
    return torch.cat(outs, dim=0), torch.cat(lses, dim=1)


def _raw(tensor):
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


@pytest.fixture()
def cpu_shadow_world():
    """A width-3 decode and its padded width-4 presentation, on CPU."""
    torch.manual_seed(20260813)
    rows, heads, dim = 8, 2, 16  # the SHAPE is not the claim; the ALGEBRA is
    width, canonical = 3, 4
    pages = torch.randn((12, 4, dim), dtype=torch.float32).to(torch.bfloat16)
    pages = pages.to(torch.float32)
    block_table = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int32
    )
    seqused_k = torch.tensor([9, 7, 11], dtype=torch.int32)
    query = torch.randn((width * rows, heads, dim)).to(torch.bfloat16)
    return {
        "rows": rows,
        "width": width,
        "canonical": canonical,
        "pages": pages,
        "block_table": block_table,
        "seqused_k": seqused_k,
        "query": query,
        "scale": dim**-0.5,
    }


def _pad(world, *, poison):
    """Build the canonical width-4 operands from the width-3 ones."""
    rows = world["rows"]
    width = world["width"]
    canonical = world["canonical"]
    real_rows = width * rows
    query = torch.zeros(
        (canonical * rows, world["query"].shape[1], world["query"].shape[2]),
        dtype=torch.bfloat16,
    )
    query[:real_rows] = world["query"]
    seqused_k = torch.zeros((canonical,), dtype=torch.int32)
    seqused_k[:width] = world["seqused_k"]  # slot 3 stays 0: the whole trick
    block_table = torch.zeros(
        (canonical, world["block_table"].shape[1]), dtype=torch.int32
    )
    block_table[:width] = world["block_table"]
    if poison:
        query[real_rows:] = float("nan")
        block_table[canonical - 1] = 0x7FFFFFF0
    return query, seqused_k, block_table


def test_padded_shadow_leaves_the_real_rows_byte_identical(cpu_shadow_world):
    """Claim 1: padding to width 4 does not move a real byte."""
    world = cpu_shadow_world
    real_rows = world["width"] * world["rows"]
    native_out, native_lse = _paged_attention_batch(
        world["query"],
        world["pages"],
        world["block_table"],
        world["seqused_k"],
        world["rows"],
        world["scale"],
    )
    query, seqused_k, block_table = _pad(world, poison=False)
    padded_out, padded_lse = _paged_attention_batch(
        query,
        world["pages"],
        block_table,
        seqused_k,
        world["rows"],
        world["scale"],
    )
    assert _raw(padded_out[:real_rows]) == _raw(native_out)
    assert _raw(padded_lse[:, :real_rows]) == _raw(native_lse)


def test_poisoned_shadow_changes_nothing_the_gate_reads(cpu_shadow_world):
    """Claim 1, sharpened: NaN Q rows + an impossible page are unread.

    This is the de-risk's recommended gate clause. It is strictly stronger
    than "matches stock", because it fails if the early exit is not taken even
    when the arithmetic would otherwise coincide.
    """
    world = cpu_shadow_world
    real_rows = world["width"] * world["rows"]
    clean_q, clean_s, clean_bt = _pad(world, poison=False)
    clean_out, clean_lse = _paged_attention_batch(
        clean_q, world["pages"], clean_bt, clean_s, world["rows"], world["scale"]
    )
    poison_q, poison_s, poison_bt = _pad(world, poison=True)
    # The poisoning is real: the operands genuinely differ.
    assert torch.isnan(poison_q[real_rows:].to(torch.float32)).all()
    assert int(poison_bt[-1][0]) == 0x7FFFFFF0
    assert _raw(poison_q) != _raw(clean_q)
    assert _raw(poison_bt) != _raw(clean_bt)
    poison_out, poison_lse = _paged_attention_batch(
        poison_q,
        world["pages"],
        poison_bt,
        poison_s,
        world["rows"],
        world["scale"],
    )
    assert _raw(poison_out[:real_rows]) == _raw(clean_out[:real_rows])
    assert _raw(poison_lse[:, :real_rows]) == _raw(clean_lse[:, :real_rows])


def test_shadow_half_is_exactly_the_early_returns_two_writes(cpu_shadow_world):
    """Claim 1, completed: zeros in O, +INF in LSE, on clean AND poisoned."""
    world = cpu_shadow_world
    real_rows = world["width"] * world["rows"]
    for poison in (False, True):
        query, seqused_k, block_table = _pad(world, poison=poison)
        out, lse = _paged_attention_batch(
            query,
            world["pages"],
            block_table,
            seqused_k,
            world["rows"],
            world["scale"],
        )
        shadow_out = out[real_rows:]
        shadow_lse = lse[:, real_rows:]
        assert shadow_out.numel() > 0 and shadow_lse.numel() > 0
        assert bool((shadow_out.to(torch.float32) == 0.0).all())
        assert bool(torch.isinf(shadow_lse).all())
        assert bool((shadow_lse > 0).all())


def test_a_nonzero_shadow_would_have_been_caught(cpu_shadow_world):
    """The negative control: this test suite can actually fail.

    If the shadow carried key rows, the early exit would not fire, the poison
    WOULD be read, and the shadow half would stop being zeros/+INF. Asserting
    that here is what makes the three tests above evidence rather than
    tautology.
    """
    world = cpu_shadow_world
    real_rows = world["width"] * world["rows"]
    query, seqused_k, block_table = _pad(world, poison=False)
    seqused_k[-1] = 4  # the scope's ORIGINAL proposal: a live shadow
    block_table[-1] = torch.tensor([1, 2, 3], dtype=torch.int32)
    out, lse = _paged_attention_batch(
        query, world["pages"], block_table, seqused_k, world["rows"], world["scale"]
    )
    assert not bool((out[real_rows:].to(torch.float32) == 0.0).all())
    assert not bool(torch.isinf(lse[:, real_rows:]).all())


# --------------------------------------------------------------------------
# The helper block, executed against real torch with a CUDA-shaped shim
# --------------------------------------------------------------------------


class _FakeDevice:
    """A device that claims to be CUDA while allocating on the host."""

    def __init__(self, spec="cuda:0"):
        text = str(spec)
        self.type = "cuda"
        self.index = int(text.split(":")[1]) if ":" in text else None

    def __str__(self):
        return "cuda" if self.index is None else f"cuda:{self.index}"

    def __repr__(self):
        return f"device('{self}')"

    def __eq__(self, other):
        return isinstance(other, _FakeDevice) and self.index == other.index

    def __hash__(self):
        return hash(("cuda", self.index))


class _FakeCuda:
    def __init__(self, state):
        self._state = state

    def is_available(self):
        return True

    def is_current_stream_capturing(self):
        return bool(self._state["capturing"])

    def current_device(self):
        return 0


class _ShimTorch:
    """Delegates to real torch, but pretends host tensors live on CUDA."""

    def __init__(self, state):
        self.cuda = _FakeCuda(state)
        self.device = _FakeDevice

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def _strip(kwargs):
        kwargs.pop("device", None)
        return kwargs

    def zeros(self, *args, **kwargs):
        return torch.zeros(*args, **self._strip(kwargs))

    def full(self, *args, **kwargs):
        return torch.full(*args, **self._strip(kwargs))

    def arange(self, *args, **kwargs):
        return torch.arange(*args, **self._strip(kwargs))


@pytest.fixture()
def b34(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """The B4 production helper block, live, with the real geometry."""
    patcher = _module(PATCHER, "qrow32_b34_helpers")
    state = {"capturing": False}

    def _authorise(*widths):
        """Install a pass sidecar authorising exactly these served widths.

        The runtime reads its authorised widths out of the credential BODY,
        so every test that reaches the selector needs a real one on disk whose
        bytes match the digest env. Rewriting it is how a test switches
        between the sealed width-4 licence and the widened width-3+4 licence.
        """
        sidecar = tmp_path / "fr13_fa2_qrow32_b4_production_pass.json"
        raw = json.dumps(
            {"production_widths": sorted(int(w) for w in widths)},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        sidecar.write_bytes(raw)
        monkeypatch.setenv(
            "FR13_FA2_QROW32_B4_PRODUCTION_PASS_SIDECAR", str(sidecar)
        )
        monkeypatch.setenv(
            "FR13_FA2_QROW32_B4_PRODUCTION_PASS_SIDECAR_SHA256",
            hashlib.sha256(raw).hexdigest(),
        )
        return sidecar

    state["authorise"] = _authorise
    namespace: dict[str, object] = {
        "os": os,
        "torch": _ShimTorch(state),
        "__name__": "fr13_b34_helpers",
    }
    exec(  # noqa: S102 - the helper block is repo source, executed on purpose
        compile(
            patcher.FIXED32_QUERY_TILE32_B4_PRODUCTION_HELPERS,
            "<b34_helpers>",
            "exec",
        ),
        namespace,
    )
    gdn = types.ModuleType("vllm.model_executor.layers.mamba.gdn_linear_attn")
    gdn._FR13_FIXED32_CAPTURE_CONTEXT = None
    gdn._FR13_FIXED32_PROFILE_CAPTURE_SCOPE = None
    gdn._FR13_FIXED32_PROFILE_MEMORY_SCOPE = False
    mamba = types.ModuleType("vllm.model_executor.layers.mamba")
    mamba.gdn_linear_attn = gdn
    forward_context = types.ModuleType("vllm.forward_context")
    holder = {"context": None}
    forward_context.get_forward_context = lambda: holder["context"]
    for name, module in (
        ("vllm", types.ModuleType("vllm")),
        ("vllm.forward_context", forward_context),
        ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
        (
            "vllm.model_executor.layers",
            types.ModuleType("vllm.model_executor.layers"),
        ),
        ("vllm.model_executor.layers.mamba", mamba),
        ("vllm.model_executor.layers.mamba.gdn_linear_attn", gdn),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    arm = namespace["_FR13_FA2_QROW32_B4_ARMS"]["gqa_pair"]
    for name, value in (
        ("FR13_FA2_QROW32_B4_PRODUCTION_ARM", "gqa_pair"),
        ("FR13_FA2_QROW32_B4_INTERNAL_ATTESTED", "1"),
        ("FR13_DRAFT_VOCAB_ROOT", "1"),
        ("FR13_DRAFT_VOCAB_K", "65536"),
        ("FR13_FIXED32_MODE", "hydra27_fixed32"),
        ("FR13_FA2_QROW32_SO_SHA256", arm["candidate_sha256"]),
        ("FR13_FA2_QROW32_SO_SIZE", str(arm["candidate_size"])),
        ("FR13_FA2_QROW32_FA2_HEAD", arm["fa2_head"]),
        ("FR13_FA2_QROW32_SOURCE_CLOSURE_SHA256", arm["source_closure_sha256"]),
        ("FR13_FA2_QROW32_SOURCE_COMMIT", "c" * 40),
        ("FR13_FA2_QROW32_B4_PATCH_SOURCE_SHA256", "d" * 64),
        ("FR13_FA2_QROW32_B4_DUAL_GATE_SHA256", "f" * 64),
        (
            "FR13_FA2_QROW32_B4_EXACT4_TASK_IDS",
            ",".join(namespace["_FR13_FA2_QROW32_B4_CANONICAL_TASK_IDS"]),
        ),
        (
            "FR13_FA2_QROW32_B4_EXACT4_SUBSET_SHA256",
            namespace["_FR13_FA2_QROW32_B4_EXACT4_SUBSET_SHA256"],
        ),
    ):
        monkeypatch.setenv(name, value)
    # The default licence is the WIDENED one, because most of this file
    # exercises the padded width-3 path. The tests that care about the sealed
    # width-4 licence call state["authorise"](4) explicitly.
    _authorise(3, 4)
    monkeypatch.delenv("ENFORCE_EAGER", raising=False)
    return namespace, gdn, state, holder


BLOCK_COLUMNS = 64


def _operands(width, *, block_columns=BLOCK_COLUMNS):
    """The real inbound decode operands at a given served width."""
    rows = 32 * width
    return {
        "query": torch.zeros((rows, 24, 256), dtype=torch.bfloat16),
        # The deployed KV cache interleaves K and V, so the outer stride is
        # TWICE the per-tensor block extent. Slicing a (8,2,1024,4,256) base
        # reproduces that exactly, which is what the predicate pins.
        "key_cache": torch.zeros((8, 2, 1024, 4, 256), dtype=torch.bfloat16)[
            :, 0
        ],
        "value_cache": None,
        "cu_seqlens_q": torch.arange(
            0, rows + 32, 32, dtype=torch.int32
        ),
        "max_seqlen_q": 32,
        "seqused_k": torch.full((width,), 96, dtype=torch.int32),
        "max_seqlen_k": 4096,
        "causal": False,
        "window_size": None,
        "block_table": torch.zeros(
            (width, block_columns), dtype=torch.int32
        ),
        "softcap": 0.0,
        "num_splits": 0,
        "tree_bias": torch.zeros((32, 32), dtype=torch.float32),
    }


def _layer(name=TARGET_LAYERS[0]):
    layer = types.SimpleNamespace()
    layer.layer_name = name
    return layer


def _forward_context(block_table):
    metadata = {
        name: types.SimpleNamespace(block_table=block_table)
        for name in TARGET_LAYERS
    }
    return types.SimpleNamespace(attn_metadata=metadata)


# --------------------------------------------------------------------------
# 2. Staging lifecycle
# --------------------------------------------------------------------------


def test_staging_refuses_to_allocate_during_capture(b34) -> None:
    namespace, _gdn, state, _holder = b34
    state["capturing"] = True
    with pytest.raises(RuntimeError, match="must be allocated before capture"):
        namespace["_fr13_fa2_qrow32_b34_staging"]("cuda:0", BLOCK_COLUMNS)
    assert namespace["_FR13_FA2_QROW32_B34_STAGING"] == {}


def test_the_serving_path_never_allocates_it_only_looks_up(b34) -> None:
    """Lazy allocate-on-first-use is the hazard; the lookup is the fix."""
    namespace, _gdn, _state, _holder = b34
    with pytest.raises(RuntimeError, match="was not allocated before capture"):
        namespace["_fr13_fa2_qrow32_b34_require_staging"]("cuda:0", BLOCK_COLUMNS)
    # It reported enough to diagnose: the key, what exists, and the hook count.
    assert namespace["_FR13_FA2_QROW32_B34_STAGING"] == {}
    assert int(namespace["_FR13_FA2_QROW32_B34_PRECAPTURE"]["calls"]) == 0


def test_precapture_hook_allocates_once_and_the_lookup_then_finds_it(
    b34,
) -> None:
    namespace, _gdn, _state, holder = b34
    block_table = torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    holder["context"] = _forward_context(block_table)
    hook = namespace["_fr13_fa2_qrow32_b34_precapture_staging"]
    staged = hook(11, "FULL", 3)
    assert staged is not None
    again = hook(12, "FULL", 3)
    assert again is staged
    precapture = namespace["_FR13_FA2_QROW32_B34_PRECAPTURE"]
    assert int(precapture["calls"]) == 2
    # Allocated ONCE, not once per graph.
    assert len(precapture["allocations"]) == 1
    assert precapture["allocations"][0]["staging_bytes"] == 2 * 128 * 24 * 256 * 2
    assert precapture["allocations"][0]["block_columns"] == BLOCK_COLUMNS
    assert set(precapture["graphs"]) == {11, 12}
    found = namespace["_fr13_fa2_qrow32_b34_require_staging"](
        _FakeDevice("cuda:0"), BLOCK_COLUMNS
    )
    assert found is staged


def test_precapture_hook_refuses_to_run_inside_capture(b34) -> None:
    namespace, _gdn, state, holder = b34
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    state["capturing"] = True
    with pytest.raises(RuntimeError, match="ran INSIDE capture"):
        namespace["_fr13_fa2_qrow32_b34_precapture_staging"](11, "FULL", 3)


@pytest.mark.parametrize(
    "graph_id,mode,width",
    [
        (11, "PIECEWISE", 3),  # not a FULL graph
        (11, "FULL", 4),       # canonical width needs no staging
        (11, "FULL", 2),       # below the qualified widths
    ],
)
def test_precapture_hook_is_a_noop_off_its_operating_point(
    b34, graph_id, mode, width
) -> None:
    namespace, _gdn, _state, holder = b34
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    assert (
        namespace["_fr13_fa2_qrow32_b34_precapture_staging"](
            graph_id, mode, width
        )
        is None
    )
    assert namespace["_FR13_FA2_QROW32_B34_STAGING"] == {}


def test_precapture_hook_does_nothing_without_the_production_arm(
    b34, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, _gdn, _state, holder = b34
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    monkeypatch.delenv("FR13_FA2_QROW32_B4_PRODUCTION_ARM", raising=False)
    assert (
        namespace["_fr13_fa2_qrow32_b34_precapture_staging"](11, "FULL", 3)
        is None
    )


def test_precapture_hook_skips_the_memory_profile_bootstrap_graph(b34) -> None:
    """3.0 MiB inside the profile scope would bias the KV-cache sizing."""
    namespace, gdn, _state, holder = b34
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    gdn._FR13_FIXED32_PROFILE_CAPTURE_SCOPE = {
        "descriptor": {"num_reqs": 3},
        "graph_id": 5,
        "completed": False,
    }
    gdn._FR13_FIXED32_PROFILE_MEMORY_SCOPE = True
    assert (
        namespace["_fr13_fa2_qrow32_b34_precapture_staging"](5, "FULL", 3)
        is None
    )
    assert namespace["_FR13_FA2_QROW32_B34_STAGING"] == {}


def test_precapture_hook_fails_loud_without_attention_metadata(b34) -> None:
    namespace, _gdn, _state, holder = b34
    holder["context"] = types.SimpleNamespace(attn_metadata=None)
    with pytest.raises(RuntimeError, match="no per-layer attention"):
        namespace["_fr13_fa2_qrow32_b34_precapture_staging"](11, "FULL", 3)


def test_staging_key_resolves_cuda_and_cuda_zero_to_the_same_slot(b34) -> None:
    namespace, _gdn, _state, _holder = b34
    key = namespace["_fr13_fa2_qrow32_b34_staging_key"]
    assert key("cuda", BLOCK_COLUMNS) == key("cuda:0", BLOCK_COLUMNS)
    assert key(_FakeDevice("cuda:0"), BLOCK_COLUMNS) == ("cuda:0", BLOCK_COLUMNS)
    assert key("cuda:1", BLOCK_COLUMNS) != key("cuda:0", BLOCK_COLUMNS)
    with pytest.raises(RuntimeError, match="block columns are not positive"):
        key("cuda:0", 0)


def test_a_different_block_table_width_is_a_different_staging_slot(b34) -> None:
    """A block-table reshape must NOT silently reuse the old staging."""
    namespace, _gdn, _state, holder = b34
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    namespace["_fr13_fa2_qrow32_b34_precapture_staging"](11, "FULL", 3)
    with pytest.raises(RuntimeError, match="was not allocated before capture"):
        namespace["_fr13_fa2_qrow32_b34_require_staging"](
            "cuda:0", BLOCK_COLUMNS * 2
        )


def test_the_staged_shadow_slot_is_zero_keys_and_the_null_page(b34) -> None:
    namespace, _gdn, _state, holder = b34
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    staged = namespace["_fr13_fa2_qrow32_b34_precapture_staging"](11, "FULL", 3)
    assert int(staged["seqused_k"][3]) == 0
    assert bool((staged["block_table"][3] == 0).all())
    assert [int(v) for v in staged["cu_seqlens_q"].tolist()] == [0, 32, 64, 96, 128]
    assert tuple(staged["query"].shape) == (128, 24, 256)
    assert tuple(staged["out"].shape) == (128, 24, 256)
    assert not namespace["_fr13_fa2_qrow32_b34_shadow_mismatches"](staged, 3)


def test_a_drifted_shadow_slot_names_itself(b34) -> None:
    namespace, _gdn, _state, holder = b34
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    staged = namespace["_fr13_fa2_qrow32_b34_precapture_staging"](11, "FULL", 3)
    staged["seqused_k"][3] = 64  # the scope's original, rejected, proposal
    mismatches = namespace["_fr13_fa2_qrow32_b34_shadow_mismatches"](staged, 3)
    assert any(item.startswith("shadow_seqused_k=") for item in mismatches)
    staged["seqused_k"][3] = 0
    staged["block_table"][3] = 7
    mismatches = namespace["_fr13_fa2_qrow32_b34_shadow_mismatches"](staged, 3)
    assert any(item.startswith("shadow_block_table_row=") for item in mismatches)


# --------------------------------------------------------------------------
# 3. The five b-dependent geometry clauses, at width 3 and width 4
# --------------------------------------------------------------------------


FIVE_CLAUSES = (
    "query(dtype,shape,stride)",
    "cu_seqlens_q(dtype,shape)",
    "seqused_k(dtype,shape)",
    "block_table(dtype,shape)",
    "tree_bias(dtype,shape,stride)",
)


@pytest.mark.parametrize("width", [3, 4])
def test_the_five_geometry_clauses_pass_at_the_qualified_widths(
    b34, width
) -> None:
    namespace, _gdn, _state, _holder = b34
    operands = _operands(width)
    operands["value_cache"] = operands["key_cache"]
    assert not namespace["_fr13_fa2_qrow32_b4_geometry_mismatches"](
        batch_size=width, **operands
    )


@pytest.mark.parametrize("width", [1, 2, 5])
def test_an_unqualified_width_is_refused_by_the_width_guard(b34, width) -> None:
    namespace, _gdn, _state, _holder = b34
    operands = _operands(max(width, 1))
    operands["value_cache"] = operands["key_cache"]
    mismatches = namespace["_fr13_fa2_qrow32_b4_geometry_mismatches"](
        batch_size=width, **operands
    )
    assert mismatches == (f"batch_size={width!r}",)


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("clause", FIVE_CLAUSES)
def test_each_b_dependent_clause_fails_alone_when_its_operand_drifts(
    b34, width, clause
) -> None:
    """Every clause must be independently load-bearing at BOTH widths."""
    namespace, _gdn, _state, _holder = b34
    operands = _operands(width)
    operands["value_cache"] = operands["key_cache"]
    wrong = 4 if width == 3 else 3
    if clause == "query(dtype,shape,stride)":
        operands["query"] = torch.zeros((32 * wrong, 24, 256), dtype=torch.bfloat16)
    elif clause == "cu_seqlens_q(dtype,shape)":
        operands["cu_seqlens_q"] = torch.arange(
            0, 32 * wrong + 32, 32, dtype=torch.int32
        )
    elif clause == "seqused_k(dtype,shape)":
        operands["seqused_k"] = torch.full((wrong,), 96, dtype=torch.int32)
    elif clause == "block_table(dtype,shape)":
        operands["block_table"] = torch.zeros(
            (wrong, BLOCK_COLUMNS), dtype=torch.int32
        )
    else:
        operands["tree_bias"] = torch.zeros((wrong, 32, 32), dtype=torch.float32)
    mismatches = namespace["_fr13_fa2_qrow32_b4_geometry_mismatches"](
        batch_size=width, **operands
    )
    assert [item.split("=", 1)[0] for item in mismatches] == [clause]


def test_the_broadcast_tree_bias_tile_is_legal_at_both_widths(b34) -> None:
    namespace, _gdn, _state, _holder = b34
    for width in (3, 4):
        operands = _operands(width)
        operands["value_cache"] = operands["key_cache"]
        operands["tree_bias"] = torch.zeros((32, 32), dtype=torch.float32)
        assert not namespace["_fr13_fa2_qrow32_b4_geometry_mismatches"](
            batch_size=width, **operands
        )


# --------------------------------------------------------------------------
# 4. Guard behaviour at the operating points the runtime visits
# --------------------------------------------------------------------------


def _begin(namespace, width, layer_name=TARGET_LAYERS[0], **overrides):
    operands = _operands(width)
    operands["value_cache"] = operands["key_cache"]
    operands.update(overrides)
    return namespace["_fr13_fa2_qrow32_b4_production_begin"](
        layer=_layer(layer_name), **operands
    )


def _capture(gdn, graph_id, width):
    gdn._FR13_FIXED32_CAPTURE_CONTEXT = {
        "graph_id": graph_id,
        "descriptor": {"num_reqs": width, "runtime_mode": "FULL"},
    }


@pytest.mark.parametrize("width", [1, 2])
def test_widths_below_the_qualified_set_still_bypass(b34, width) -> None:
    """Excluded on ECONOMICS, not safety: stock already fits one wave there."""
    namespace, gdn, state, _holder = b34
    state["capturing"] = True
    _capture(gdn, 100 + width, width)
    selection = _begin(namespace, width)
    assert selection["candidate_served"] is False
    assert selection["bypass_reason"] == "non_b34_capture"
    namespace["_fr13_fa2_qrow32_b4_production_end"](selection, completed=True)
    assert namespace["_FR13_FA2_QROW32_B4_PRODUCTION_GRAPHS"] == {}


def test_a_width_three_capture_without_precapture_staging_fails_loud(
    b34,
) -> None:
    """The whole point of caveat 2, stated as behaviour."""
    namespace, gdn, state, _holder = b34
    state["capturing"] = True
    _capture(gdn, 103, 3)
    with pytest.raises(RuntimeError, match="was not allocated before capture"):
        _begin(namespace, 3)


def test_a_width_three_capture_with_precapture_staging_serves_the_candidate(
    b34,
) -> None:
    namespace, gdn, state, holder = b34
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    staged = namespace["_fr13_fa2_qrow32_b34_precapture_staging"](103, "FULL", 3)
    state["capturing"] = True
    _capture(gdn, 103, 3)
    selection = _begin(namespace, 3)
    assert selection["candidate_served"] is True
    assert selection["batch_size"] == 3
    assert selection["staged"] is staged
    assert int(selection["tree_bias"].stride(0)) == 131092
    assert tuple(selection["tree_bias"].shape) == (4, 32, 32)


def test_the_canonical_width_never_touches_staging(b34) -> None:
    namespace, gdn, state, _holder = b34
    state["capturing"] = True
    _capture(gdn, 104, 4)
    selection = _begin(namespace, 4)
    assert selection["candidate_served"] is True
    assert selection["batch_size"] == 4
    assert selection["staged"] is None
    assert namespace["_FR13_FA2_QROW32_B34_STAGING"] == {}


def test_a_capture_descriptor_that_disagrees_with_the_call_is_fatal(
    b34,
) -> None:
    """The descriptor says 4; the operands say 3. Refuse, do not guess."""
    namespace, gdn, state, _holder = b34
    state["capturing"] = True
    _capture(gdn, 105, 4)
    with pytest.raises(RuntimeError, match="capture width disagrees"):
        _begin(namespace, 3)


def test_an_unknown_capture_batch_is_still_fail_closed(b34) -> None:
    namespace, gdn, state, _holder = b34
    state["capturing"] = True
    _capture(gdn, 106, 7)
    with pytest.raises(RuntimeError, match="capture batch drifted"):
        _begin(namespace, 4)


def test_a_piecewise_or_eager_step_bypasses_rather_than_killing_the_server(
    b34,
) -> None:
    namespace, _gdn, state, _holder = b34
    state["capturing"] = False
    selection = _begin(namespace, 4)
    assert selection["bypass_reason"] == "outside_capture"


def test_a_non_target_layer_is_refused_at_both_widths(b34) -> None:
    namespace, gdn, state, holder = b34
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    namespace["_fr13_fa2_qrow32_b34_precapture_staging"](107, "FULL", 3)
    state["capturing"] = True
    for width in (3, 4):
        _capture(gdn, 107 + width, width)
        with pytest.raises(RuntimeError, match="layer identity drifted"):
            _begin(namespace, width, layer_name="language_model.model.layers.0.self_attn.attn")


def _engage_all_layers(namespace, gdn, state, holder, width):
    state["capturing"] = False
    if width != 4:
        holder["context"] = _forward_context(
            torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
        )
        namespace["_fr13_fa2_qrow32_b34_precapture_staging"](200 + width, "FULL", width)
    state["capturing"] = True
    _capture(gdn, 200 + width, width)
    for layer_name in TARGET_LAYERS:
        selection = _begin(namespace, width, layer_name=layer_name)
        namespace["_fr13_fa2_qrow32_b4_production_end"](selection, completed=True)
    return 200 + width


@pytest.mark.parametrize("width", [3, 4])
def test_capture_end_writes_a_width_aware_engagement_record(
    b34, width, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, gdn, state, holder = b34
    engagement = tmp_path / "engagement.json"
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B4_PRODUCTION_ENGAGEMENT_JSON", str(engagement)
    )
    graph_id = _engage_all_layers(namespace, gdn, state, holder, width)
    namespace["_fr13_fa2_qrow32_b4_production_capture_end"](
        graph_id, "c" * 64, "FULL", width
    )
    expected = engagement if width == 4 else tmp_path / "engagement_b3.json"
    other = tmp_path / "engagement_b3.json" if width == 4 else engagement
    assert expected.exists()
    # The two widths write DIFFERENT files; neither can clobber the other.
    assert not other.exists()
    record = json.loads(expected.read_text(encoding="ascii"))
    assert record["batch_size"] == width
    assert record["concurrency"] == width
    assert record["total_query_rows"] == 32 * width
    assert record["padded_to_canonical_width"] is (width != 4)
    assert record["canonical_width"] == 4
    assert record["layer_count"] == 16
    assert record["candidate_served"] is True
    if width == 3:
        assert record["shadow_slot"] == 3
        assert record["shadow_seqused_k"] == 0
        assert record["shadow_block_table_page"] == 0
        assert record["staging_precapture_allocations"]
    else:
        assert record["shadow_slot"] is None
        assert record["shadow_seqused_k"] is None


def test_capture_end_refuses_a_padded_graph_that_did_not_stage_every_layer(
    b34, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, gdn, state, holder = b34
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B4_PRODUCTION_ENGAGEMENT_JSON",
        str(tmp_path / "engagement.json"),
    )
    graph_id = _engage_all_layers(namespace, gdn, state, holder, 3)
    graphs = namespace["_FR13_FA2_QROW32_B4_PRODUCTION_GRAPHS"]
    graphs[graph_id]["staged_layers"].discard(TARGET_LAYERS[5])
    with pytest.raises(RuntimeError, match="did not stage every target"):
        namespace["_fr13_fa2_qrow32_b4_production_capture_end"](
            graph_id, "c" * 64, "FULL", 3
        )


def test_capture_end_refuses_a_canonical_graph_that_used_staging(
    b34, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, gdn, state, holder = b34
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B4_PRODUCTION_ENGAGEMENT_JSON",
        str(tmp_path / "engagement.json"),
    )
    graph_id = _engage_all_layers(namespace, gdn, state, holder, 4)
    graphs = namespace["_FR13_FA2_QROW32_B4_PRODUCTION_GRAPHS"]
    graphs[graph_id]["staged_layers"].add(TARGET_LAYERS[0])
    with pytest.raises(RuntimeError, match="canonical-width graph used padded"):
        namespace["_fr13_fa2_qrow32_b4_production_capture_end"](
            graph_id, "c" * 64, "FULL", 4
        )


def test_capture_end_refuses_a_padded_graph_with_no_precapture_record(
    b34, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, gdn, state, holder = b34
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B4_PRODUCTION_ENGAGEMENT_JSON",
        str(tmp_path / "engagement.json"),
    )
    graph_id = _engage_all_layers(namespace, gdn, state, holder, 3)
    namespace["_FR13_FA2_QROW32_B34_PRECAPTURE"]["graphs"].clear()
    with pytest.raises(RuntimeError, match="no pre-capture staging record"):
        namespace["_fr13_fa2_qrow32_b4_production_capture_end"](
            graph_id, "c" * 64, "FULL", 3
        )


def test_capture_end_still_catches_a_sentinel_leak_outside_the_qualified_set(
    b34, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, _gdn, _state, _holder = b34
    engagement = tmp_path / "engagement.json"
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B4_PRODUCTION_ENGAGEMENT_JSON", str(engagement)
    )
    capture_end = namespace["_fr13_fa2_qrow32_b4_production_capture_end"]
    capture_end(11, "a" * 64, "FULL", 1)
    assert not engagement.exists()
    namespace["_FR13_FA2_QROW32_B4_PRODUCTION_GRAPHS"][11] = {
        "layers": {TARGET_LAYERS[0]},
        "arm": "gqa_pair",
        "staged_layers": set(),
        "batch_size": 1,
    }
    with pytest.raises(RuntimeError, match="engaged outside FULL B4"):
        capture_end(11, "a" * 64, "FULL", 1)


def test_a_graph_cannot_mix_two_served_widths(b34) -> None:
    namespace, gdn, state, holder = b34
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    namespace["_fr13_fa2_qrow32_b34_precapture_staging"](300, "FULL", 3)
    state["capturing"] = True
    _capture(gdn, 300, 3)
    selection = _begin(namespace, 3, layer_name=TARGET_LAYERS[0])
    namespace["_fr13_fa2_qrow32_b4_production_end"](selection, completed=True)
    _capture(gdn, 300, 4)
    other = _begin(namespace, 4, layer_name=TARGET_LAYERS[1])
    with pytest.raises(RuntimeError, match="capture engagement drifted"):
        namespace["_fr13_fa2_qrow32_b4_production_end"](other, completed=True)


def test_the_engagement_record_refuses_an_inconsistent_padding_claim(
    b34,
) -> None:
    namespace, _gdn, _state, _holder = b34
    with pytest.raises(RuntimeError, match="padding disagrees with width"):
        namespace["_fr13_fa2_qrow32_b4_production_record"](
            arm="gqa_pair", runtime_mode="FULL", graph_id=1,
            graph_signature="a" * 64, layers=list(TARGET_LAYERS), calls=16,
            batch_size=3, padded=False,
        )
    with pytest.raises(RuntimeError, match="width is not authorised"):
        namespace["_fr13_fa2_qrow32_b4_production_record"](
            arm="gqa_pair", runtime_mode="FULL", graph_id=1,
            graph_signature="a" * 64, layers=list(TARGET_LAYERS), calls=16,
            batch_size=2, padded=False,
        )


# --------------------------------------------------------------------------
# 5. Scope token round-trip
# --------------------------------------------------------------------------


def test_the_engagement_record_emits_the_b34_token_at_both_widths(
    b34, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, gdn, state, holder = b34
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B4_PRODUCTION_ENGAGEMENT_JSON",
        str(tmp_path / "engagement.json"),
    )
    for width in (4, 3):
        graph_id = _engage_all_layers(namespace, gdn, state, holder, width)
        namespace["_fr13_fa2_qrow32_b4_production_capture_end"](
            graph_id, "c" * 64, "FULL", width
        )
    for name in ("engagement.json", "engagement_b3.json"):
        record = json.loads((tmp_path / name).read_text(encoding="ascii"))
        assert record["candidate_scope"] == B34_SCOPE
        assert record["candidate_scope_widths"] == [3, 4]


def test_readers_accept_the_sealed_token_and_the_widened_one() -> None:
    """The sealed +29.50 ms/step width-4 pair must stay re-reducible."""
    reduce_module = _module(PAIR_REDUCE, "b4_pair_reduce_scope")
    assert reduce_module.SEALED_B4_CANDIDATE_SCOPE == SEALED_SCOPE
    assert reduce_module.B34_CANDIDATE_SCOPE == B34_SCOPE
    assert set(reduce_module.ACCEPTED_CANDIDATE_SCOPES) == {
        SEALED_SCOPE,
        B34_SCOPE,
    }
    timing = B4_TIMING_RUNNER.read_text(encoding="utf-8")
    assert 'engagement.get("candidate_scope") not in (' in timing
    assert f'"{SEALED_SCOPE}",' in timing
    assert f'"{B34_SCOPE}",' in timing


@pytest.mark.parametrize("scope", [SEALED_SCOPE, B34_SCOPE])
def test_the_pair_reducer_accepts_either_declared_scope(scope) -> None:
    reduce_module = _module(PAIR_REDUCE, "b4_pair_reduce_accept")
    engagement = {
        "status": "ENGAGED",
        "runtime_mode": "FULL",
        "arm": "gqa_pair",
        "layer_count": 16,
        "batch_size": 4,
        "total_query_rows": 128,
        "candidate_served": True,
        "fallback_allowed": False,
        "candidate_scope": scope,
        "task_count": 4,
        "layers": list(TARGET_LAYERS),
    }
    reduce_module.validate_pair_engagement(
        None, engagement, expected_task_count=4
    )


def test_the_pair_reducer_still_refuses_an_invented_scope() -> None:
    reduce_module = _module(PAIR_REDUCE, "b4_pair_reduce_refuse")
    engagement = {
        "status": "ENGAGED",
        "runtime_mode": "FULL",
        "arm": "gqa_pair",
        "layer_count": 16,
        "batch_size": 4,
        "total_query_rows": 128,
        "candidate_served": True,
        "fallback_allowed": False,
        "candidate_scope": "everything_everywhere",
        "task_count": 4,
        "layers": list(TARGET_LAYERS),
    }
    with pytest.raises(reduce_module.PairError):
        reduce_module.validate_pair_engagement(
            None, engagement, expected_task_count=4
        )


def test_the_credential_scope_widened_but_no_binary_pin_moved() -> None:
    sidecar = _module(SIDECAR, "b4_sidecar_scope")
    widened = sidecar.PRODUCTION_SCOPE_BY_WIDTHS[(3, 4)]
    assert "widths 3 and 4" in widened
    assert "width 3 or 4" in sidecar.REQUIRED_RUNTIME_BY_WIDTHS[(3, 4)]
    # The sealed width-4 prose is preserved WORD FOR WORD, because a
    # credential issued from the banked width-4 dual gate must re-verify
    # against it byte for byte.
    assert sidecar.PRODUCTION_SCOPE_BY_WIDTHS[(4,)] == (
        "qrow32 GQA-pair B4 exact tree attention only"
    )
    assert sidecar.REQUIRED_RUNTIME_BY_WIDTHS[(4,)] == (
        "fixed32 K64 ROOT=1 exact4 B4 physical32 FULL graph on Tail23 or "
        "Hydra27"
    )
    # Zero rebuild: every identity pin is exactly what the de-risk banked.
    assert sidecar.CANDIDATE_SHA256 == (
        "af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb"
    )
    assert sidecar.CANDIDATE_SIZE == 299_813_360
    assert sidecar.FA2_HEAD == "29210221863736a08f71a866459e368ad1ac4a95"
    assert sidecar.SOURCE_CLOSURE_SHA256 == (
        "9c3f9e751da7b783e9d07d8e40d5bc2234b99e719a1048668bd6c82244ed2d81"
    )
    assert sidecar.SELECTOR_SENTINEL == 0x20014


# --------------------------------------------------------------------------
# 6. The patcher's pre-capture injection
# --------------------------------------------------------------------------


PRISTINE_CUDA_GRAPH = Path("/tmp/vllm_pristine_019/extracted/vllm/compilation/cuda_graph.py")

FR10_CAPTURE_BEGIN_ANCHOR = (
    "            cudagraph = torch.cuda.CUDAGraph()\n"
    "\n"
    "            with ExitStack() as stack:\n"
)


def _fr10_capture_begin_injection(text: str) -> str:
    """Reproduce the fr10 patcher's capture-begin edit, for ordering tests."""
    assert text.count(FR10_CAPTURE_BEGIN_ANCHOR) == 1
    return text.replace(
        FR10_CAPTURE_BEGIN_ANCHOR,
        "            cudagraph = torch.cuda.CUDAGraph()\n"
        '            if self.runtime_mode.name == "FULL":\n'
        "                from vllm.model_executor.layers.mamba import (\n"
        "                    gdn_linear_attn as _fr13_f32_capture_gdn,\n"
        "                )\n"
        "                _fr13_f32_capture_gdn._fr13_fixed32_capture_begin(\n"
        "                    id(cudagraph),\n"
        "                )\n"
        "\n"
        "            with ExitStack() as stack:\n",
        1,
    )


@pytest.fixture()
def cuda_graph_source(tmp_path: Path) -> Path:
    if not PRISTINE_CUDA_GRAPH.is_file():
        pytest.skip("pristine vLLM cuda_graph.py is not available here")
    target = tmp_path / "cuda_graph.py"
    target.write_text(PRISTINE_CUDA_GRAPH.read_text(encoding="utf-8"))
    return target


def test_the_precapture_injection_refuses_to_run_before_the_fr10_hook(
    cuda_graph_source: Path,
) -> None:
    """Ordering is a precondition, not a coincidence.

    The fr10 capture-begin anchor spans three lines. Injecting between them
    would break it, so the staging injector demands that fr10 has already run.
    """
    patcher = _module(PATCHER, "qrow32_b34_patch_order")
    with pytest.raises(RuntimeError, match="fr10 fixed32 capture-begin hook"):
        patcher._patch_cuda_graph_qrow32_b34_precapture_staging(cuda_graph_source)


def test_the_precapture_hook_lands_before_capture_and_is_idempotent(
    cuda_graph_source: Path,
) -> None:
    patcher = _module(PATCHER, "qrow32_b34_patch_place")
    cuda_graph_source.write_text(
        _fr10_capture_begin_injection(cuda_graph_source.read_text())
    )
    assert patcher._patch_cuda_graph_qrow32_b34_precapture_staging(
        cuda_graph_source
    )
    assert not patcher._patch_cuda_graph_qrow32_b34_precapture_staging(
        cuda_graph_source
    )
    text = cuda_graph_source.read_text()
    construct = text.index("cudagraph = torch.cuda.CUDAGraph()")
    hook = text.index("_fr13_fa2_qrow32_b34_precapture_staging(")
    fr10 = text.index("_fr13_fixed32_capture_begin(")
    exit_stack = text.index("with ExitStack() as stack:")
    graph_ctx = text.index("with torch.cuda.graph(")
    # After the graph object exists, before the fr10 hook, and well before any
    # capture has begun.
    assert construct < hook < fr10 < exit_stack < graph_ctx
    # It is a FULL-graph-only hook and it hands over the identity capture-end
    # will later use.
    segment = text[hook - 400 : hook + 300]
    assert 'self.runtime_mode.name == "FULL"' in segment
    assert "id(cudagraph)" in segment
    assert "entry.batch_descriptor.num_reqs" in segment
    ast.parse(text)


def test_the_b4_production_wiring_installs_the_staging_hook_too() -> None:
    """A production build without the hook would fail closed at width 3."""
    text = PATCHER.read_text(encoding="utf-8")
    assert "_patch_cuda_graph_qrow32_b34_precapture_staging" in text
    assert "# FR13_FA2_QROW32_B34_PRECAPTURE_STAGING" in text
    index = text.index("elif fixed32_query_tile32_b4_production:")
    block = text[index : index + 900]
    assert "_patch_cuda_graph_qrow32_b34_precapture_staging(" in block
    assert "_patch_cuda_graph_qrow32_b4_production(" in block


def test_the_fr10_capture_lifecycle_anchor_this_hook_rides_still_exists() -> None:
    """If fr10's anchor ever moves, this hook's placement claim is stale."""
    fr10 = FR10_PATCHER.read_text(encoding="utf-8")
    assert "def _fr13_fixed32_capture_begin(" in fr10
    assert '"            cudagraph = torch.cuda.CUDAGraph()\\n"' in fr10
    assert "_fr13_f32_capture_gdn._fr13_fixed32_capture_begin(" in fr10


# --------------------------------------------------------------------------
# 7. The b=3 byte-gate arm
# --------------------------------------------------------------------------


def test_the_live_ab_block_defines_its_own_width_constants() -> None:
    """The two helper blocks are installed by mutually exclusive flags.

    A live-AB build has no B4 production block, so a cross-block name would be
    a NameError at the first width-3 replay -- and ast.parse cannot see it.
    """
    patcher = _module(PATCHER, "qrow32_b34_blob_names")
    live = patcher.FIXED32_QUERY_TILE32_LIVE_AB_HELPERS
    production = patcher.FIXED32_QUERY_TILE32_B4_PRODUCTION_HELPERS
    for label, blob in (("live", live), ("production", production)):
        tree = ast.parse(blob)
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined.add((alias.asname or alias.name).split(".")[0])
        used = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        leaked = sorted(
            name
            for name in used - defined
            if name.startswith("_FR13") or name.startswith("_fr13")
        )
        assert leaked == [], f"{label} block leaks {leaked}"
    assert "_FR13_FA2_QROW32_LIVE_AB_WIDTHS = (3, 4)" in live
    assert "_FR13_FA2_QROW32_LIVE_AB_POISON_BLOCK_ID" in live


def test_the_live_gate_states_the_three_padded_claims() -> None:
    patcher = _module(PATCHER, "qrow32_b34_gate_claims")
    live = patcher.FIXED32_QUERY_TILE32_LIVE_AB_HELPERS
    # 1. the padded candidate call exists and is only used off the canonical
    assert "def _fr13_fa2_qrow32_live_ab_padded_call(" in live
    assert "FR13 qrow32 live gate padded the canonical width" in live
    # 2. the poisoned variant poisons BOTH operands the de-risk named
    assert 'staged["query"][real_rows:] = float("nan")' in live
    assert "_FR13_FA2_QROW32_LIVE_AB_POISON_BLOCK_ID" in live
    # 3. the shadow contract is checked as bytes, on device
    assert "def _fr13_fa2_qrow32_live_ab_shadow_mismatches(" in live
    assert "shadow_output_not_zero" in live
    assert "shadow_lse_not_positive_infinity" in live
    # and the verdict actually consumes all of it
    assert "total_poison_output_mismatches == 0" in live
    assert "and not shadow_failures" in live


def test_the_live_gate_attempts_once_per_width_not_once_per_process() -> None:
    patcher = _module(PATCHER, "qrow32_b34_gate_once")
    live = patcher.FIXED32_QUERY_TILE32_LIVE_AB_HELPERS
    assert "_FR13_FA2_QROW32_LIVE_AB_WIDTHS_ATTEMPTED = set()" in live
    assert "if width in _FR13_FA2_QROW32_LIVE_AB_WIDTHS_ATTEMPTED:" in live
    # and the padded arm is default-off
    assert 'os.environ.get(\n        "FR13_FA2_QROW32_LIVE_PAGED_AB_B3", "0"\n    ) != "1"' in live


def _b3_gate_record(source_commit: str) -> dict:
    layers = []
    for name in TARGET_LAYERS:
        summary = {
            "dtype": "torch.bfloat16",
            "shape": [96, 24, 256],
            "bytes": 96 * 24 * 256 * 2,
            "raw_byte_mismatches": 0,
            "stock_sha256": "a" * 64,
            "candidate_sha256": "a" * 64,
        }
        lse = {
            "dtype": "torch.float32",
            "shape": [24, 96],
            "bytes": 24 * 96 * 4,
            "raw_byte_mismatches": 0,
            "stock_sha256": "b" * 64,
            "candidate_sha256": "b" * 64,
        }
        slots = [
            {
                "slot": slot,
                "output": {
                    "dtype": "torch.bfloat16",
                    "shape": [32, 24, 256],
                    "bytes": 32 * 24 * 256 * 2,
                    "raw_byte_mismatches": 0,
                    "stock_sha256": "c" * 64,
                    "candidate_sha256": "c" * 64,
                },
                "lse": {
                    "dtype": "torch.float32",
                    "shape": [24, 32],
                    "bytes": 24 * 32 * 4,
                    "raw_byte_mismatches": 0,
                    "stock_sha256": "d" * 64,
                    "candidate_sha256": "d" * 64,
                },
            }
            for slot in range(3)
        ]
        layers.append(
            {
                "layer_name": name,
                "output": summary,
                "lse": lse,
                "slots": slots,
                "poisoned_shadow": {
                    "output": {
                        "raw_byte_mismatches": 0,
                        "stock_sha256": "e" * 64,
                        "candidate_sha256": "e" * 64,
                    },
                    "lse": {
                        "raw_byte_mismatches": 0,
                        "stock_sha256": "f" * 64,
                        "candidate_sha256": "f" * 64,
                    },
                    "shadow_rows": [96, 128],
                    "shadow_seqused_k": 0,
                    "shadow_block_table_page": 0x7FFFFFF0,
                    "shadow_query_fill": "nan",
                },
            }
        )
    return {
        "schema": "fr13.fixed32.fa2_qrow32_live_paged_exact4_ab.v1",
        "status": "PASS",
        "batch_size": 3,
        "concurrency": 3,
        "physical_rows_per_slot": 32,
        "total_query_rows": 96,
        "padded_to_canonical_width": True,
        "canonical_width": 4,
        "canonical_query_rows": 128,
        "shadow_slot": 3,
        "shadow_seqused_k": 0,
        "shadow_block_table_page": 0,
        "poisoned_shadow_arm": True,
        "poisoned_shadow_output_raw_byte_mismatches": 0,
        "poisoned_shadow_lse_raw_byte_mismatches": 0,
        "shadow_contract_failures": [],
        "candidate_arm": "gqa_pair",
        "candidate_so_sha256": (
            "af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb"
        ),
        "candidate_so_size": 299_813_360,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "fa2_head": "29210221863736a08f71a866459e368ad1ac4a95",
        "fa2_source_closure_sha256": (
            "9c3f9e751da7b783e9d07d8e40d5bc2234b99e719a1048668bd6c82244ed2d81"
        ),
        "fixed32_mode": "hydra27_fixed32",
        "runtime_mode": "FULL",
        "layer_count": 16,
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "selector_sentinel": 0x20014,
        "source_commit": source_commit,
        "fallback_allowed": False,
        "performance_measurement": False,
        "served_return": "stock captured graph output unchanged",
        "task_ids": [
            "astropy__astropy-12907",
            "astropy__astropy-13033",
            "astropy__astropy-13236",
            "astropy__astropy-13398",
        ],
        "subset_sha256": (
            "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
        ),
        "operands": {
            "query_shape": [96, 24, 256],
            "slot_coverage": [0, 1, 2],
            "query_start_loc": [0, 32, 64, 96],
        },
        "layers": layers,
    }


def _run_b3_verifier(tmp_path: Path, record: dict, monkeypatch):
    gate = _module(GQA_GATE, "qrow32_gqa_pair_gate_b3")
    result = tmp_path / "live_b3.json"
    result.write_text(
        json.dumps(record, ensure_ascii=True), encoding="ascii"
    )
    monkeypatch.setattr(
        gate, "validate_candidate", lambda so, source: {"candidate_arm": "gqa_pair"}
    )
    args = types.SimpleNamespace(
        result=result,
        candidate_so=tmp_path / "candidate.so",
        fa2_source=tmp_path,
        fixed32_mode="hydra27_fixed32",
        source_commit="c" * 40,
    )
    return gate, args


def test_the_b3_verifier_accepts_a_clean_padded_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _b3_gate_record("c" * 40)
    gate, args = _run_b3_verifier(tmp_path, record, monkeypatch)
    verdict = gate.verify_arm_b3(args)
    assert verdict["schema"] == gate.B3_ARM_SCHEMA
    assert verdict["status"] == "PASS"
    assert verdict["batch_size"] == 3
    assert verdict["padded_to_canonical_width"] is True
    assert verdict["shadow_slot"] == 3
    assert verdict["slot_coverage"] == [0, 1, 2]
    assert verdict["poisoned_shadow_output_raw_byte_mismatches"] == 0


@pytest.mark.parametrize(
    "mutate,pattern",
    [
        (
            lambda r: r.__setitem__("shadow_contract_failures", ["l3:clean:x"]),
            "shadow_contract_failures drifted",
        ),
        (
            lambda r: r.__setitem__(
                "poisoned_shadow_output_raw_byte_mismatches", 1
            ),
            "poisoned_shadow_output_raw_byte_mismatches drifted",
        ),
        (
            lambda r: r.__setitem__("padded_to_canonical_width", False),
            "padded_to_canonical_width drifted",
        ),
        (
            lambda r: r.__setitem__("shadow_seqused_k", 64),
            "shadow_seqused_k drifted",
        ),
        (
            lambda r: r["layers"][0]["poisoned_shadow"].__setitem__(
                "raw_byte_mismatches_placeholder", 0
            )
            or r["layers"][0]["poisoned_shadow"]["output"].__setitem__(
                "candidate_sha256", "9" * 64
            ),
            "poisoned-shadow output",
        ),
        (
            lambda r: r["layers"][0]["poisoned_shadow"].__setitem__(
                "shadow_block_table_page", 0
            ),
            "poisoned-shadow declaration drifted",
        ),
        (
            lambda r: r["layers"][0]["poisoned_shadow"].__setitem__(
                "shadow_query_fill", "zero"
            ),
            "poisoned-shadow declaration drifted",
        ),
        (
            lambda r: r["layers"].pop(),
            "layer coverage drifted",
        ),
        (
            lambda r: r["layers"][0]["slots"].pop(),
            "slot coverage drifted",
        ),
    ],
)
def test_the_b3_verifier_is_fail_closed_on_every_padded_clause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate, pattern
) -> None:
    record = _b3_gate_record("c" * 40)
    mutate(record)
    gate, args = _run_b3_verifier(tmp_path, record, monkeypatch)
    with pytest.raises(gate.GateError, match=pattern):
        gate.verify_arm_b3(args)


# --------------------------------------------------------------------------
# 8. Runner and launcher wiring
# --------------------------------------------------------------------------


def test_the_b3_gate_flag_is_declared_defaulted_validated_and_forwarded() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    name = "FR13_FA2_QROW32_LIVE_PAGED_AB_B3"
    assert f"\n  {name}\n" in launcher                       # caller-guard list
    assert f"{name}=${{{name}:-0}}" in launcher              # default-off
    assert f'case "${name}" in' in launcher                  # 0/1 validator
    assert f'-e {name}="${name}"' in launcher                # into the container
    # It is only meaningful for the de-risked candidate.
    assert 'b3 padded gate requires the gqa_pair live A/B arm' in launcher


def test_the_inner_runner_keeps_the_sealed_width4_gate_default_off() -> None:
    inner = INNER_RUNNER.read_text(encoding="utf-8")
    assert "LIVE_AB_B3=${FR13_QROW32_LIVE_AB_B3:-0}" in inner
    assert 'FR13_FA2_QROW32_LIVE_PAGED_AB_B3="$LIVE_AB_B3"' in inner
    assert "the b3 padded arm is only de-risked for gqa_pair" in inner
    assert "padded_b3_arm=%s" in inner


def test_the_b34_runner_mirrors_the_sealed_dual_gate_lineage() -> None:
    runner = B34_RUNNER.read_text(encoding="utf-8")
    sealed = (
        REPO / "scripts/fr13_run_b4_fa2_qrow32_gqa_pair_live_gate.sh"
    ).read_text(encoding="utf-8")
    # Default-off, same shape as its parent.
    assert "FR13_RUN_B34_QROW32_GQA_PAIR_LIVE_GATE" in runner
    for clause in (
        "RUNROOT must resolve below",
        "tracked worktree must be clean",
        "source commit must be pushed before the dual gate",
        "all Docker containers must be absent before the dual gate",
        "candidate binary/source identity changed during the dual gate",
        "dual gate source changed during execution",
        "frozen pushed source changed during the dual gate",
    ):
        assert clause in sealed and clause in runner
    # And it adds exactly the b3 arm, per topology, in its own file.
    assert "FR13_QROW32_LIVE_AB_B3=1" in runner
    assert "verify-arm-b3" in runner
    assert "fr13_fa2_qrow32_live_paged_ab_b3.json" in runner
    assert "${label}_b3_verification.json" in runner
    assert runner.count("run_arm tail6_fixed32 tail23") == 1
    assert runner.count("run_arm hydra27_fixed32 hydra27") == 1


def test_the_b34_runner_is_executable_and_parses() -> None:
    assert os.access(B34_RUNNER, os.X_OK)
    import subprocess

    subprocess.run(["bash", "-n", str(B34_RUNNER)], check=True)


# --------------------------------------------------------------------------
# 9. The in-capture path reads no device VALUES
#
# The single defect that made the padded arm unshippable: the serving path
# called the shadow VALUE predicate from inside CUDA graph capture, and that
# predicate does .item()/.tolist() on CUDA staging tensors. Tensor.item() and
# Tensor.tolist() lower to cudaMemcpyAsync D2H + cudaStreamSynchronize;
# cudaStreamSynchronize on a capturing stream returns
# cudaErrorStreamCaptureUnsupported and invalidates the capture. It would have
# fired on the FIRST target layer of the width-3 FULL capture -- i.e. at
# server startup -- so the b3 byte gate and the width-3 timing arm could never
# have run. The 78 original cases missed it because the shim's "CUDA" tensors
# are host tensors, where .item() always succeeds.
#
# These tests close that: the shim's tensors now REFUSE device reads while the
# fake stream is capturing, exactly as the real ones would.
# --------------------------------------------------------------------------


class _CaptureIllegalRead(RuntimeError):
    """What the driver would raise: a D2H sync on a capturing stream."""


@pytest.fixture()
def capture_hostile(monkeypatch: pytest.MonkeyPatch, b34):
    """Make .item()/.tolist() fatal while the fake stream is capturing."""
    namespace, gdn, state, holder = b34
    for name in ("item", "tolist"):
        original = getattr(torch.Tensor, name)

        def guarded(self, *args, _original=original, **kwargs):
            if state["capturing"]:
                raise _CaptureIllegalRead(
                    "operation not permitted when stream is capturing"
                )
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, guarded, raising=True)
    return namespace, gdn, state, holder


def test_the_guard_itself_bites(capture_hostile) -> None:
    """Without this, every test below would pass vacuously."""
    _namespace, _gdn, state, _holder = capture_hostile
    tensor = torch.zeros((2,), dtype=torch.int32)
    assert tensor.tolist() == [0, 0]
    state["capturing"] = True
    with pytest.raises(_CaptureIllegalRead):
        tensor.tolist()
    with pytest.raises(_CaptureIllegalRead):
        tensor[0].item()


def test_a_width_three_capture_survives_a_capture_hostile_stream(
    capture_hostile,
) -> None:
    """THE REGRESSION. This is the b=3 FULL capture, end to end."""
    namespace, gdn, state, holder = capture_hostile
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    # Pre-capture: value reads are legal here, and this is where they happen.
    staged = namespace["_fr13_fa2_qrow32_b34_precapture_staging"](301, "FULL", 3)
    assert staged is not None
    state["capturing"] = True
    _capture(gdn, 301, 3)
    # Every one of the 16 target layers, because the failure was per-layer.
    for layer_name in TARGET_LAYERS:
        selection = _begin(namespace, 3, layer_name=layer_name)
        assert selection["candidate_served"] is True
        assert selection["staged"] is staged


def test_the_canonical_width_also_survives_a_capture_hostile_stream(
    capture_hostile,
) -> None:
    namespace, gdn, state, _holder = capture_hostile
    state["capturing"] = True
    _capture(gdn, 302, 4)
    selection = _begin(namespace, 4)
    assert selection["candidate_served"] is True
    assert selection["staged"] is None


def test_the_precapture_hook_still_proves_the_shadow_by_reading_it(
    capture_hostile,
) -> None:
    """The values are not skipped, they are proven EARLIER. Show the reads."""
    namespace, _gdn, state, holder = capture_hostile
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    staged = namespace["_fr13_fa2_qrow32_b34_precapture_staging"](303, "FULL", 3)
    assert staged["precapture_proof"][3] == {
        "cu_seqlens_q": (0, 32, 64, 96, 128),
        "shadow_seqused_k": 0,
        "shadow_block_table_page": 0,
        "shadow_block_table_row_all_null": True,
    }
    # And the hook genuinely could not have run under capture.
    state["capturing"] = True
    with pytest.raises(RuntimeError, match="inside CUDA capture"):
        namespace["_fr13_fa2_qrow32_b34_shadow_mismatches"](staged, 3)


def test_the_metadata_predicate_requires_the_precapture_proof(b34) -> None:
    """Metadata alone is not enough: the shadow proof must be present."""
    namespace, _gdn, _state, holder = b34
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    staged = namespace["_fr13_fa2_qrow32_b34_precapture_staging"](304, "FULL", 3)
    predicate = namespace["_fr13_fa2_qrow32_b34_staged_metadata_mismatches"]
    assert predicate(staged, 3) == ()
    staged.pop("precapture_proof")
    assert any(
        m.startswith("shadow_precapture_proof=") for m in predicate(staged, 3)
    )


def test_a_padded_capture_without_the_proof_refuses_to_serve(b34) -> None:
    namespace, gdn, state, holder = b34
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    staged = namespace["_fr13_fa2_qrow32_b34_precapture_staging"](305, "FULL", 3)
    staged["precapture_proof"][3]["shadow_seqused_k"] = 4
    state["capturing"] = True
    _capture(gdn, 305, 3)
    with pytest.raises(RuntimeError, match="staged operands drifted"):
        _begin(namespace, 3)


def test_a_drifted_staged_shape_is_caught_without_a_device_read(
    capture_hostile,
) -> None:
    namespace, _gdn, state, holder = capture_hostile
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    staged = namespace["_fr13_fa2_qrow32_b34_precapture_staging"](306, "FULL", 3)
    staged["query"] = torch.zeros((127, 24, 256), dtype=torch.bfloat16)
    state["capturing"] = True
    predicate = namespace["_fr13_fa2_qrow32_b34_staged_metadata_mismatches"]
    assert any(m.startswith("staged_query(") for m in predicate(staged, 3))


def test_no_device_value_read_is_reachable_from_the_serving_path() -> None:
    """A source-level audit, so the rule survives a future edit.

    The serving entry point is _fr13_fa2_qrow32_b4_production_begin. Walk the
    call graph of the helper block from it and assert that nothing it can
    reach calls .item() or .tolist(). The two functions that legitimately do
    -- the pre-capture value predicate and its sole caller, the pre-capture
    hook -- must not be reachable from it.
    """
    patcher = _module(PATCHER, "qrow32_b34_capture_audit")
    tree = ast.parse(patcher.FIXED32_QUERY_TILE32_B4_PRODUCTION_HELPERS)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    calls = {
        name: {
            sub.func.id
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
        }
        for name, node in functions.items()
    }
    reads = {
        name
        for name, node in functions.items()
        if any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in ("item", "tolist")
            for sub in ast.walk(node)
        )
    }
    # Exactly one function is allowed to read device values.
    assert reads == {"_fr13_fa2_qrow32_b34_shadow_mismatches"}, reads
    reachable: set[str] = set()
    frontier = ["_fr13_fa2_qrow32_b4_production_begin"]
    while frontier:
        current = frontier.pop()
        for callee in calls.get(current, ()):
            if callee in functions and callee not in reachable:
                reachable.add(callee)
                frontier.append(callee)
    assert "_fr13_fa2_qrow32_b34_shadow_mismatches" not in reachable
    assert "_fr13_fa2_qrow32_b34_precapture_staging" not in reachable
    assert not (reads & reachable), reads & reachable
    # And the capture-safe twin IS the one the serving path uses.
    assert (
        "_fr13_fa2_qrow32_b34_staged_metadata_mismatches" in reachable
    )


# --------------------------------------------------------------------------
# 10. The production tree-bias tagger and the geometry predicate agree
#
# The production tagger accepted only (32,32) and (4,32,32) while the
# production geometry predicate accepts (batch_size,32,32) at every qualified
# width and the gate-side twin goes out of its way to handle (3,32,32). A
# per-slot width-3 mask therefore passed the predicate and then died in the
# tagger with "tree bias shape drifted", during the width-3 FULL capture.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 32), (3, 32, 32), (4, 32, 32)])
def test_every_shape_the_geometry_predicate_admits_is_taggable(
    b34, shape
) -> None:
    namespace, _gdn, _state, _holder = b34
    width = 3 if len(shape) == 2 or shape[0] == 3 else 4
    bias = torch.zeros(shape, dtype=torch.float32)
    geometry = namespace["_fr13_fa2_qrow32_b4_geometry_mismatches"]
    operands = _operands(width)
    operands["value_cache"] = operands["key_cache"]
    operands["tree_bias"] = bias
    assert geometry(batch_size=width, **operands) == ()
    tagged = namespace["_fr13_fa2_qrow32_b4_candidate_tree_bias"](
        bias, "gqa_pair"
    )
    assert tuple(tagged.shape) == (4, 32, 32)
    assert int(tagged.stride(0)) == 0x20014


def test_the_per_slot_width_three_mask_fills_the_shadow_plane_from_plane_zero(
    b34,
) -> None:
    """Deterministic, not uninitialised -- the fail-closed choice.

    The shadow never reads its plane (seqused_k == 0 exits before the mask is
    touched), but the C++ side checks tree_bias.size(0) == batch_size == 4, so
    a plane must exist and it must be defined.
    """
    namespace, _gdn, _state, _holder = b34
    bias = torch.stack(
        [torch.full((32, 32), float(slot)) for slot in range(3)]
    ).to(torch.float32)
    tagged = namespace["_fr13_fa2_qrow32_b4_candidate_tree_bias"](
        bias, "gqa_pair"
    )
    for slot in range(3):
        assert bool((tagged[slot] == float(slot)).all())
    assert bool((tagged[3] == tagged[0]).all())


def test_the_production_tagger_matches_the_gate_twin_shape_for_shape() -> None:
    """The two blocks are mutually exclusive; they must not disagree."""
    patcher = _module(PATCHER, "qrow32_b34_tagger_parity")
    tree = ast.parse(patcher.FIXED32_QUERY_TILE32_B4_PRODUCTION_HELPERS)
    production = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fa2_qrow32_b4_candidate_tree_bias"
    )
    source = ast.unparse(production)
    # It no longer hard-codes the canonical widths, and it grew the same
    # torch.cat shadow-plane fill the gate-side twin has.
    assert "(32, 32), (4, 32, 32)" not in source
    assert "_FR13_FA2_QROW32_B34_WIDTHS" in source
    assert "torch.cat" in source


@pytest.mark.parametrize("shape", [(2, 32, 32), (5, 32, 32), (4, 16, 32)])
def test_a_mask_the_predicate_rejects_is_still_refused_by_the_tagger(
    b34, shape
) -> None:
    namespace, _gdn, _state, _holder = b34
    with pytest.raises(RuntimeError, match="tree bias shape drifted"):
        namespace["_fr13_fa2_qrow32_b4_candidate_tree_bias"](
            torch.zeros(shape, dtype=torch.float32), "gqa_pair"
        )


# --------------------------------------------------------------------------
# 11. The credential authorises widths; the code does not authorise itself
#
# PRODUCTION_SCOPE was widened to say "widths 3 and 4" while the dual gate it
# is issued from still bound only the two width-4 arm verifications, and the
# selector served width 3 whenever the arm was set. A banked width-4 dual gate
# would therefore have reissued a credential that turned padded width-3
# serving on for every timing run, with the shadow doctrine never byte-checked
# on that machine.
# --------------------------------------------------------------------------


def _dual_gate_payload(source_commit, *, with_b3):
    sidecar = _module(SIDECAR, "b4_sidecar_widths")
    payload = {
        "schema": sidecar.DUAL_GATE_SCHEMA,
        "status": "PASS",
        "candidate_arm": sidecar.ARM,
        "selector_sentinel": sidecar.SELECTOR_SENTINEL,
        "candidate_so_sha256": sidecar.CANDIDATE_SHA256,
        "candidate_so_size": sidecar.CANDIDATE_SIZE,
        "fa2_head": sidecar.FA2_HEAD,
        "fa2_source_closure_sha256": sidecar.SOURCE_CLOSURE_SHA256,
        "task_ids": list(sidecar.EXACT4_TASK_IDS),
        "subset_sha256": sidecar.EXACT4_SUBSET_SHA256,
        "qualified_topologies": list(sidecar.QUALIFIED_TOPOLOGIES),
        "layer_count_per_topology": 16,
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "fallback_allowed": False,
        "performance_measurement": False,
        "timing_eligible": False,
        "production_eligible": False,
        "source_commit": source_commit,
        "tail23_verification_sha256": "1" * 64,
        "hydra27_verification_sha256": "2" * 64,
    }
    if with_b3:
        payload.update(
            {
                "qualified_widths": [3, 4],
                "tail23_b3_verification_sha256": "3" * 64,
                "hydra27_b3_verification_sha256": "4" * 64,
                "b3_padded_to_canonical_width": True,
                "b3_shadow_slot": 3,
                "b3_output_raw_byte_mismatches": 0,
                "b3_lse_raw_byte_mismatches": 0,
                "b3_poisoned_shadow_output_raw_byte_mismatches": 0,
                "b3_poisoned_shadow_lse_raw_byte_mismatches": 0,
                "b3_shadow_contract_failures": [],
            }
        )
    return payload


def test_a_width_four_only_dual_gate_issues_the_sealed_width_four_scope() -> None:
    sidecar = _module(SIDECAR, "b4_sidecar_widths")
    commit = "a" * 40
    summary = sidecar.validate_dual_gate(
        _dual_gate_payload(commit, with_b3=False), source_commit=commit
    )
    assert summary["widths"] == (4,)
    assert summary["b3"] == {}
    texts = sidecar._scope_texts(summary["widths"])
    assert texts["production_scope"] == (
        "qrow32 GQA-pair B4 exact tree attention only"
    )
    assert texts["production_widths"] == [4]


def test_only_a_b3_bearing_dual_gate_widens_the_credential() -> None:
    sidecar = _module(SIDECAR, "b4_sidecar_widths")
    commit = "a" * 40
    summary = sidecar.validate_dual_gate(
        _dual_gate_payload(commit, with_b3=True), source_commit=commit
    )
    assert summary["widths"] == (3, 4)
    assert summary["b3"]["tail23_b3_verification_sha256"] == "3" * 64
    texts = sidecar._scope_texts(summary["widths"])
    assert "widths 3 and 4" in texts["production_scope"]
    assert "poisoned shadow" in texts["credential_basis"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"tail23_b3_verification_sha256": None},
        {"hydra27_b3_verification_sha256": None},
        {"b3_output_raw_byte_mismatches": 1},
        {"b3_lse_raw_byte_mismatches": 1},
        {"b3_poisoned_shadow_output_raw_byte_mismatches": 1},
        {"b3_poisoned_shadow_lse_raw_byte_mismatches": 1},
        {"b3_shadow_contract_failures": ["slot 3 lse not +INF"]},
        {"b3_padded_to_canonical_width": False},
        {"b3_shadow_slot": 2},
        {"qualified_widths": [3]},
        {"qualified_widths": [2, 3, 4]},
    ],
)
def test_a_widening_claim_without_its_evidence_is_refused(mutation) -> None:
    """Declaring the width is not proving it."""
    sidecar = _module(SIDECAR, "b4_sidecar_widths")
    commit = "a" * 40
    payload = _dual_gate_payload(commit, with_b3=True)
    for key, value in mutation.items():
        if value is None:
            payload.pop(key)
        else:
            payload[key] = value
    with pytest.raises(sidecar.SidecarError):
        sidecar.validate_dual_gate(payload, source_commit=commit)


def test_b3_evidence_outside_a_b3_scope_is_refused() -> None:
    """A gate cannot smuggle b3 digests under a width-4 declaration."""
    sidecar = _module(SIDECAR, "b4_sidecar_widths")
    commit = "a" * 40
    payload = _dual_gate_payload(commit, with_b3=True)
    payload["qualified_widths"] = [4]
    with pytest.raises(sidecar.SidecarError, match="outside its declared scope"):
        sidecar.validate_dual_gate(payload, source_commit=commit)


def test_the_b34_runner_feeds_its_b3_verifications_into_verify_dual() -> None:
    """Otherwise the two b3 files are consumed by nothing at all."""
    runner = B34_RUNNER.read_text(encoding="utf-8")
    block = runner[runner.index("verify-dual") :]
    block = block[: block.index("dual_gate_verification.json")]
    assert "--tail-b3-verification" in block
    assert "--hydra-b3-verification" in block
    assert "tail23_b3_verification.json" in block
    assert "hydra27_b3_verification.json" in block


def test_verify_dual_declares_width_four_only_without_the_b3_arms() -> None:
    gate = _module(GQA_GATE, "qrow32_gate_widths")
    args = types.SimpleNamespace(
        tail_b3_verification=None, hydra_b3_verification=None
    )
    assert gate._verify_dual_b3(args, {}, "a" * 40) == {
        "qualified_widths": [4]
    }


def test_verify_dual_refuses_half_the_b3_evidence(tmp_path) -> None:
    gate = _module(GQA_GATE, "qrow32_gate_widths")
    args = types.SimpleNamespace(
        tail_b3_verification=tmp_path / "tail.json",
        hydra_b3_verification=None,
    )
    with pytest.raises(gate.GateError, match="BOTH topology verifications"):
        gate._verify_dual_b3(args, {}, "a" * 40)


@pytest.mark.parametrize("widths", [(3, 4), (4,)])
def test_the_runtime_reads_its_authorised_widths_from_the_credential(
    b34, widths
) -> None:
    namespace, _gdn, state, _holder = b34
    state["authorise"](*widths)
    assert namespace["_fr13_fa2_qrow32_b34_authorised_widths"]() == widths


def test_the_sealed_width_four_credential_bypasses_width_three(b34) -> None:
    """THE REGRESSION: a banked width-4 credential must not serve width 3."""
    namespace, gdn, state, holder = b34
    state["authorise"](4)
    holder["context"] = _forward_context(
        torch.zeros((4, BLOCK_COLUMNS), dtype=torch.int32)
    )
    # It does not even allocate the 3.0 MiB of staging.
    assert namespace["_fr13_fa2_qrow32_b34_precapture_staging"](
        401, "FULL", 3
    ) is None
    assert namespace["_FR13_FA2_QROW32_B34_STAGING"] == {}
    state["capturing"] = True
    _capture(gdn, 401, 3)
    selection = _begin(namespace, 3)
    assert selection["candidate_served"] is False
    assert selection["bypass_reason"] == "non_b34_capture"
    # ...while width 4 is served exactly as before.
    _capture(gdn, 402, 4)
    assert _begin(namespace, 4)["candidate_served"] is True


@pytest.mark.parametrize("widths", [(2, 4), (3,), (4, 5), (1, 3, 4)])
def test_an_unqualified_credential_width_is_fatal(b34, widths) -> None:
    """Widths 1 and 2 are not qualified, and 4 is not optional."""
    namespace, _gdn, state, _holder = b34
    state["authorise"](*widths)
    with pytest.raises(RuntimeError, match="not a qualified scope"):
        namespace["_fr13_fa2_qrow32_b34_authorised_widths"]()


def test_a_tampered_credential_body_is_fatal(b34, tmp_path) -> None:
    """The digest is the binding; the body is only trusted through it."""
    namespace, _gdn, state, _holder = b34
    sidecar = state["authorise"](4)
    sidecar.write_bytes(b'{"production_widths":[3,4]}')
    with pytest.raises(RuntimeError, match="pass sidecar bytes drifted"):
        namespace["_fr13_fa2_qrow32_b34_authorised_widths"]()


# --------------------------------------------------------------------------
# 12. The reducer's treated set comes from the engagement, not from max(width)
#
# `treated = max(widths)` classified width 3 -- now genuinely treated under a
# widened credential -- as a placebo AND then chose it as the
# difference-in-differences control, silently subtracting the effect from
# itself while printing "placebo clean".
# --------------------------------------------------------------------------


def _reduce_module():
    return _module(PAIR_REDUCE, "b4_pair_reduce_widths")


def _by_width(rows):
    return {
        "available": True,
        "by_width": {
            str(width): {
                "steps": steps,
                "mean_ms": ms,
                "fraction": 0.0,
                "sd_ms": 0.0,
            }
            for width, (steps, ms) in rows.items()
        },
    }


def test_the_treated_set_is_read_off_the_engagement_record() -> None:
    module = _reduce_module()
    assert module.treated_widths(
        {"candidate_scope": B34_SCOPE, "candidate_scope_widths": [3, 4]}
    ) == (3, 4)
    assert module.treated_widths(
        {"candidate_scope": SEALED_SCOPE, "candidate_scope_widths": [4]}
    ) == (4,)
    # The banked width-4 pair predates the field and must stay re-reducible.
    assert module.treated_widths({"candidate_scope": SEALED_SCOPE}) == (4,)


def test_an_unresolvable_treated_set_is_refused_rather_than_guessed() -> None:
    module = _reduce_module()
    with pytest.raises(module.PairError, match="must not be guessed"):
        module.treated_widths({"candidate_scope": B34_SCOPE})


def test_a_treated_width_three_is_never_the_did_control() -> None:
    """THE REGRESSION, as arithmetic."""
    module = _reduce_module()
    stock = _by_width({2: (1000, 285.0), 3: (1000, 362.0), 4: (3000, 411.0)})
    cand = _by_width({2: (1000, 302.0), 3: (1000, 340.0), 4: (3000, 382.0)})
    placebo = module.per_width_placebo(stock, cand, (3, 4))
    roles = {r["width"]: r["role"] for r in placebo["rows"]}
    assert roles == {2: "placebo", 3: "treated", 4: "treated"}
    assert placebo["treated_widths"] == [3, 4]
    assert placebo["treated_width"] == 4
    # Width 2 is the only untreated width, so it is the only control.
    assert placebo["difference_in_differences"]["control_width"] == 2


def test_the_same_data_under_the_sealed_scope_still_controls_on_width_three() -> None:
    """The sealed lineage is unchanged: width 3 is a control there."""
    module = _reduce_module()
    stock = _by_width({2: (1000, 285.0), 3: (1000, 362.0), 4: (3000, 411.0)})
    cand = _by_width({2: (1000, 302.0), 3: (1000, 340.0), 4: (3000, 382.0)})
    placebo = module.per_width_placebo(stock, cand, (4,))
    assert placebo["difference_in_differences"]["control_width"] == 3
    assert placebo["treated_widths"] == [4]


def test_a_placebo_leak_is_only_looked_for_at_untreated_widths() -> None:
    module = _reduce_module()
    stock = _by_width({2: (1000, 285.0), 3: (1000, 362.0), 4: (3000, 411.0)})
    cand = _by_width({2: (1000, 302.0), 3: (1000, 340.0), 4: (3000, 382.0)})
    placebo = module.per_width_placebo(stock, cand, (3, 4))
    leaks = [
        r
        for r in placebo["rows"]
        if not r["candidate_engaged"] and r["improvement_ms"] >= 4.2
    ]
    assert leaks == []


def test_the_reducer_carries_the_scope_widths_off_the_engagement_file(
    tmp_path,
) -> None:
    module = _reduce_module()
    arm = tmp_path / "arm"
    (arm / "logs").mkdir(parents=True)
    (arm / "logs" / "fr13_fa2_qrow32_b4_production_engagement.json").write_text(
        json.dumps(
            {
                "candidate_scope": B34_SCOPE,
                "candidate_scope_widths": [3, 4],
            }
        ),
        encoding="ascii",
    )
    engagement = module.read_engagement(arm)
    assert engagement["candidate_scope_widths"] == [3, 4]
    assert module.treated_widths(engagement) == (3, 4)


def _issue_body(sidecar, widths):
    """The credential body a gate qualifying `widths` would produce."""
    summary = {
        "source_commit": "a" * 40,
        "tail23_verification_sha256": "1" * 64,
        "hydra27_verification_sha256": "2" * 64,
        "widths": widths,
        "b3": (
            {
                "tail23_b3_verification_sha256": "3" * 64,
                "hydra27_b3_verification_sha256": "4" * 64,
            }
            if 3 in widths
            else {}
        ),
    }
    return sidecar._body(
        patch={"source_commit": "a" * 40, "patch_source_sha256": "b" * 64},
        dual_gate_sha256="c" * 64,
        dual_gate_canonical_sha256="d" * 64,
        summary=summary,
    )


@pytest.mark.parametrize("widths", [(4,), (3, 4)])
def test_a_credential_verifies_against_the_scope_it_was_issued_with(
    widths, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = _module(SIDECAR, "b4_sidecar_roundtrip")
    body = _issue_body(sidecar, widths)
    assert body["production_widths"] == list(widths)
    payload = dict(body)
    payload["canonical_sha256"] = sidecar._digest(sidecar.canonical_bytes(body))
    raw = sidecar.canonical_bytes(payload) + b"\n"
    path = tmp_path / "pass.json"
    path.write_bytes(raw)
    monkeypatch.setattr(sidecar, "validate_candidate", lambda *a, **k: {})
    monkeypatch.setattr(
        sidecar,
        "validate_patch_source_digest",
        lambda *a, **k: {
            "source_commit": "a" * 40,
            "patch_source_sha256": "b" * 64,
        },
    )
    verified = sidecar.verify_sidecar(
        sidecar_path=path,
        expected_sidecar_sha256=hashlib.sha256(raw).hexdigest(),
        candidate_so=path,
        expected_candidate_sha256=sidecar.CANDIDATE_SHA256,
        arm=sidecar.ARM,
        patch_source=path,
        expected_source_commit="a" * 40,
        expected_patch_source_sha256="b" * 64,
    )
    assert verified["production_widths"] == list(widths)


def test_widened_prose_over_a_width_four_credential_is_refused(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attack this exists to stop: edit the scope text, keep the widths."""
    sidecar = _module(SIDECAR, "b4_sidecar_tamper")
    body = _issue_body(sidecar, (4,))
    body["production_scope"] = sidecar.PRODUCTION_SCOPE_BY_WIDTHS[(3, 4)]
    payload = dict(body)
    payload["canonical_sha256"] = sidecar._digest(sidecar.canonical_bytes(body))
    raw = sidecar.canonical_bytes(payload) + b"\n"
    path = tmp_path / "pass.json"
    path.write_bytes(raw)
    monkeypatch.setattr(sidecar, "validate_candidate", lambda *a, **k: {})
    monkeypatch.setattr(
        sidecar,
        "validate_patch_source_digest",
        lambda *a, **k: {
            "source_commit": "a" * 40,
            "patch_source_sha256": "b" * 64,
        },
    )
    with pytest.raises(sidecar.SidecarError, match="contract drifted"):
        sidecar.verify_sidecar(
            sidecar_path=path,
            expected_sidecar_sha256=hashlib.sha256(raw).hexdigest(),
            candidate_so=path,
            expected_candidate_sha256=sidecar.CANDIDATE_SHA256,
            arm=sidecar.ARM,
            patch_source=path,
            expected_source_commit="a" * 40,
            expected_patch_source_sha256="b" * 64,
        )
