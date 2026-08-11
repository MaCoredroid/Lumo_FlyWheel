"""FR13_CFWD_CAPTURE_REMAINDER: the committer fill folded into the CFWD graph.

The module itself cannot be imported on this host (it needs triton/vllm), so the
structural assertions read the source, and the equivalence test AST-extracts the
real `_fill` body and executes it against a reference restatement of the eager
fill it replaces. That is the load-bearing test: the arm is only legal if the
captured fill lands byte-identical buffers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path("src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py")
GRAPH_FN = "_fr13_native_committer_all_layers_graph"


def _source() -> str:
    return SOURCE.read_text()


def _graph_fn_node() -> ast.FunctionDef:
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == GRAPH_FN:
            return node
    raise AssertionError(f"{GRAPH_FN} not found in {SOURCE}")


def _nested(name: str) -> ast.FunctionDef:
    for node in ast.walk(_graph_fn_node()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"nested {name}() not found in {GRAPH_FN}")


# --------------------------------------------------------------- structure


def test_flag_is_opt_in_and_sidecar_backed() -> None:
    text = _source()
    assert 'os.environ.get("FR13_CFWD_CAPTURE_REMAINDER") == "1"' in text
    # EngineCore worker curation drops bare FR13_* vars, so a sidecar is required.
    assert '"/logs/fr13_cfwd_capture_remainder.arm"' in text
    assert '"/tmp/fr13_cfwd_capture_remainder.arm"' in text
    assert "def _fr13_cfwd_capture_remainder_on() -> bool:" in text


def test_default_off_leaves_the_eager_fill_exactly_as_it_was() -> None:
    """When the arm is off the committer must take the original code path."""
    text = _source()
    assert "if not _remainder:" in text
    for original in (
        "abuf.fill_(-1e4)",
        "ssi_buf.fill_(SCRATCH)",
        "kbuf[:, s0:s0 + _nl] = k_rings[:, b, nodes]",
        "ssi_buf[:, :B, :] = spec_state_indices[:, :B, 0:1].to(torch.int32)",
    ):
        assert original in text, f"eager fill line dropped: {original}"


def test_arm_is_part_of_the_graph_signature() -> None:
    """An armed and an unarmed graph have different bodies and must not alias."""
    text = _source()
    assert "id(banks_list[0]),\n           _remainder)" in text


def test_capture_region_includes_the_fill_only_when_armed() -> None:
    body = _nested("_body")
    src = ast.unparse(body)
    assert "if _remainder:" in src and "_fill()" in src and "_loop()" in src


def test_replay_guards_are_self_diagnosing() -> None:
    text = _source()
    assert "FR13_CFWD_CAPTURE_REMAINDER requires burn OFF" in text
    assert "ring addresses moved" in text
    assert "would read stale memory" in text
    # The guard must raise, not warn: a stale ring commits wrong state silently.
    assert 'st["ring_ptrs"] != _ring_ptrs()' in text


def test_flag_reaches_the_server_and_drops_a_sidecar() -> None:
    launcher = Path("scripts/fr13_launch_forked_fa2_tree_server.sh").read_text()
    assert (
        '-e FR13_CFWD_CAPTURE_REMAINDER="${FR13_CFWD_CAPTURE_REMAINDER:-0}"'
        in launcher
    )
    assert 'echo "1" > "$LOG_DIR/fr13_cfwd_capture_remainder.arm"' in launcher
    # Unarmed boots must delete the sidecar, else a prior ON boot leaks.
    assert (
        'rm -f "$LOG_DIR/fr13_cfwd_capture_remainder.arm"' in launcher
    )


# --------------------------------------------------------------- equivalence


def _reference_eager_fill(
    *, L, B, MAX_B, MAX_PATH, SCRATCH, root_node, acc,
    accepted_paths, spec_state_indices, k_rings, v_rings, a_rings, b_rings,
    num_kh, dim_k, num_vh, dim_v, dtype,
):
    """Restatement of the original eager fill (the code under `if not _remainder`)."""
    MAXT = MAX_B * MAX_PATH
    kbuf = torch.zeros(L, MAXT, num_kh, dim_k, dtype=dtype)
    vbuf = torch.zeros(L, MAXT, num_vh, dim_v, dtype=dtype)
    abuf = torch.full((L, MAXT, num_vh), -1e4, dtype=dtype)
    bbuf = torch.zeros(L, MAXT, num_vh, dtype=dtype)
    ssi = torch.zeros(L, MAX_B, MAX_PATH, dtype=torch.int32)

    abuf.fill_(-1e4)
    bbuf.zero_()
    kbuf.zero_()
    vbuf.zero_()
    ssi.fill_(SCRATCH)
    for b in range(B):
        _nl = 1 + int(acc[b])
        nodes = torch.cat([
            torch.full((1,), int(root_node), dtype=torch.long),
            accepted_paths[b, : int(acc[b])].to(torch.long),
        ])
        s0 = b * MAX_PATH
        kbuf[:, s0:s0 + _nl] = k_rings[:, b, nodes]
        vbuf[:, s0:s0 + _nl] = v_rings[:, b, nodes]
        abuf[:, s0:s0 + _nl] = a_rings[:, b, nodes]
        bbuf[:, s0:s0 + _nl] = b_rings[:, b, nodes]
    ssi[:, :B, :] = spec_state_indices[:, :B, 0:1].to(torch.int32)
    return kbuf, vbuf, abuf, bbuf, ssi


def _run_real_fill(**dims):
    """Execute the REAL `_fill` body extracted from the module source."""
    L, B, MAX_B = dims["L"], dims["B"], dims["MAX_B"]
    MAX_PATH, SCRATCH = dims["MAX_PATH"], dims["SCRATCH"]
    num_kh, dim_k = dims["num_kh"], dims["dim_k"]
    num_vh, dim_v = dims["num_vh"], dims["dim_v"]
    RING, dtype = dims["RING"], dims["dtype"]
    MAXT = MAX_B * MAX_PATH

    kbuf = torch.zeros(L, MAXT, num_kh, dim_k, dtype=dtype)
    vbuf = torch.zeros(L, MAXT, num_vh, dim_v, dtype=dtype)
    abuf = torch.full((L, MAXT, num_vh), -1e4, dtype=dtype)
    bbuf = torch.zeros(L, MAXT, num_vh, dtype=dtype)
    ssi_buf = torch.zeros(L, MAX_B, MAX_PATH, dtype=torch.int32)

    # staging, exactly as the eager prologue of the armed path does it
    acc_stage = torch.full((MAX_B,), -1, dtype=torch.long)
    acc_stage[:B].copy_(dims["acc"][:B])
    path_stage = torch.zeros((MAX_B, MAX_PATH - 1), dtype=torch.long)
    path_stage[:B].copy_(dims["accepted_paths"][:B, : MAX_PATH - 1])
    ssi_stage = torch.full((L, MAX_B, 1), SCRATCH, dtype=torch.int32)
    ssi_stage[:, :B].copy_(dims["spec_state_indices"][:, :B, 0:1])

    st = {
        "node_mat": torch.zeros((MAX_B, MAX_PATH), dtype=torch.long),
        "ar": torch.arange(MAX_PATH),
        "arb": torch.arange(MAX_B),
        "acc_stage": acc_stage,
        "path_stage": path_stage,
        "ssi_stage": ssi_stage,
    }
    namespace = {
        "torch": torch, "st": st, "kbuf": kbuf, "vbuf": vbuf,
        "abuf": abuf, "bbuf": bbuf, "ssi_buf": ssi_buf,
        "k_rings": dims["k_rings"], "v_rings": dims["v_rings"],
        "a_rings": dims["a_rings"], "b_rings": dims["b_rings"],
        "L": L, "MAX_B": MAX_B, "MAX_PATH": MAX_PATH, "RING": RING,
        "root_node": dims["root_node"], "num_kh": num_kh, "dim_k": dim_k,
        "num_vh": num_vh, "dim_v": dim_v,
    }
    exec(compile(ast.unparse(_nested("_fill")), "<fr13-cfwd-fill>", "exec"), namespace)
    namespace["_fill"]()
    return kbuf, vbuf, abuf, bbuf, ssi_buf


@pytest.mark.parametrize(
    "L,B,MAX_B,MAX_PATH,accepted",
    [
        (3, 1, 1, 16, [5]),            # B1 serving shape
        (3, 1, 4, 16, [0]),            # B1 in a B4-capacity graph, root only
        (2, 4, 4, 16, [0, 3, 15, 7]),  # B4, full spread incl. max path
        (2, 2, 4, 8, [1, 6]),          # padding rows b >= B must stay neutral
    ],
)
def test_captured_fill_is_bitwise_identical_to_the_eager_fill(
    L, B, MAX_B, MAX_PATH, accepted
) -> None:
    torch.manual_seed(0xF213)
    num_kh, dim_k, num_vh, dim_v = 2, 4, 2, 4
    RING, SCRATCH, root_node = 32, 99, 0
    dtype = torch.float32

    acc = torch.tensor(accepted, dtype=torch.long)
    accepted_paths = torch.randint(0, RING, (MAX_B, MAX_PATH - 1), dtype=torch.long)
    spec_state_indices = torch.randint(0, 50, (L, MAX_B, MAX_PATH), dtype=torch.int32)
    k_rings = torch.randn(L, MAX_B, RING, num_kh, dim_k, dtype=dtype)
    v_rings = torch.randn(L, MAX_B, RING, num_vh, dim_v, dtype=dtype)
    a_rings = torch.randn(L, MAX_B, RING, num_vh, dtype=dtype)
    b_rings = torch.randn(L, MAX_B, RING, num_vh, dtype=dtype)

    shared = dict(
        L=L, B=B, MAX_B=MAX_B, MAX_PATH=MAX_PATH, SCRATCH=SCRATCH,
        root_node=root_node, acc=acc, accepted_paths=accepted_paths,
        spec_state_indices=spec_state_indices, k_rings=k_rings, v_rings=v_rings,
        a_rings=a_rings, b_rings=b_rings, num_kh=num_kh, dim_k=dim_k,
        num_vh=num_vh, dim_v=dim_v, dtype=dtype,
    )
    expected = _reference_eager_fill(**shared)
    actual = _run_real_fill(RING=RING, **shared)

    for name, want, got in zip(
        ("kbuf", "vbuf", "abuf", "bbuf", "ssi"), expected, actual
    ):
        assert torch.equal(want, got), f"{name} drifted between eager and captured fill"
