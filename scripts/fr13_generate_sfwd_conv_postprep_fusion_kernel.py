#!/usr/bin/env python3
"""Generate the fixed32 SFWD conv/post-prep fusion kernel source.

The arithmetic body is derived mechanically from the qualified frontier-5
load-once SFWD producer.  Keeping generation explicit prevents the 32-node
unroll from drifting while the output stores are redirected to the recurrence
and post-prep surfaces.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "src"
    / "lumo_flywheel_serving"
    / "fr13_sfwd_prior_reuse_descriptorless.py"
)
OUTPUT = (
    ROOT
    / "src"
    / "lumo_flywheel_serving"
    / "fr13_sfwd_conv_postprep_fusion_kernel.py"
)
SOURCE_FUNCTION = "_fr13_fixed32_sfwd_channel_serial_kernel"
OUTPUT_FUNCTION = "_fr13_fixed32_sfwd_conv_postprep_fusion_kernel"
NODEGROUP_ROWS = 8


PREAMBLE = '''"""Generated fixed32 SFWD tree-conv/post-prep fusion kernel.

Regenerate with ``scripts/fr13_generate_sfwd_conv_postprep_fusion_kernel.py``.
Do not edit the unrolled producer body by hand.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _fr13_store_fixed32_conv_outputs(
    query,
    key,
    value_spec,
    value_tree,
    conv_tap,
    activated,
    pid_b,
    node: tl.constexpr,
    offs_c,
    N: tl.constexpr,
    C: tl.constexpr,
    Q_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    STORE_CONV_TAP: tl.constexpr,
):
    """Publish the exact BF16 boundary without a full conv intermediate."""
    row = pid_b * N + node
    activated_bf16 = activated.to(tl.bfloat16)
    q_mask = offs_c < Q_DIM
    k_mask = (offs_c >= Q_DIM) & (offs_c < 2 * Q_DIM)
    v_mask = offs_c >= 2 * Q_DIM
    tl.store(
        query + row * Q_DIM + offs_c,
        activated_bf16,
        mask=q_mask,
    )
    tl.store(
        key + row * Q_DIM + (offs_c - Q_DIM),
        activated_bf16,
        mask=k_mask,
    )
    value_offset = row * V_DIM + (offs_c - 2 * Q_DIM)
    tl.store(value_spec + value_offset, activated_bf16, mask=v_mask)
    tl.store(value_tree + value_offset, activated_bf16, mask=v_mask)
    if STORE_CONV_TAP:
        tl.store(conv_tap + row * C + offs_c, activated_bf16)


'''


SIGNATURE = '''@triton.jit
def _fr13_fixed32_sfwd_conv_postprep_fusion_kernel(
    x,
    conv_state,
    spec_state_indices,
    sticky_guard_ok,
    conv_weights,
    bias,
    a,
    b,
    A_log,
    dt_bias,
    query,
    key,
    value_spec,
    value_tree,
    g,
    beta,
    source_stage,
    conv_tap,
    CONV_STRIDE_ROW: tl.constexpr,
    BANK_ROWS: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STORE_CONV_TAP: tl.constexpr,
    CAPTURE_GUARD: tl.constexpr,
    X_STRIDE_ROW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GATE_BLOCK: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    EMBED_GATE_CTA: tl.constexpr,
):
    """Fuse one fixed32 layer's conv, recurrence outputs, and post-prep."""
    pid_b = tl.program_id(0)
    pid_task = tl.program_id(1)
    channel_tasks: tl.constexpr = C // BLOCK_C
    Q_DIM: tl.constexpr = H * K
    V_DIM: tl.constexpr = HV * V
    GATE_ROWS: tl.constexpr = 2 * BLOCK_C // GATE_BLOCK
    GATE_TASKS: tl.constexpr = N // GATE_ROWS
    if EMBED_GATE_CTA:
        pid_c = pid_task
'''


EMBEDDED_GATING = '''        # The 40-program schedule appends the four unchanged gate tiles to
        # its first four channel programs.
        if pid_task < GATE_TASKS:
            pid_n_base = pid_task * GATE_ROWS
'''


STANDALONE_PROLOGUE = '''    else:
        if pid_task < channel_tasks:
            pid_c = pid_task
'''


STANDALONE_GATING = '''        else:
            pid_n_base = (pid_task - channel_tasks) * GATE_ROWS
'''


GATING_BODY = '''offs_n = pid_n_base + tl.arange(0, GATE_ROWS)[:, None]
offs_h_1d = tl.arange(0, GATE_BLOCK)
h_mask = offs_h_1d < HV
offs_h = offs_h_1d[None, :]
gate_mask = (offs_n < N) & (offs_h < HV)
row = pid_b * N + offs_n
if pid_n_base < N:
    a_value = tl.load(
        a + row * HV + offs_h,
        mask=gate_mask,
        other=0.0,
    ).to(tl.float32)
    b_value = tl.load(
        b + row * HV + offs_h,
        mask=gate_mask,
        other=0.0,
    ).to(tl.float32)
    A_log_value = tl.load(
        A_log + offs_h_1d,
        mask=h_mask,
        other=0.0,
    ).to(tl.float32)[None, :]
    dt_bias_value = tl.load(
        dt_bias + offs_h_1d,
        mask=h_mask,
        other=0.0,
    ).to(tl.float32)[None, :]
    gate_input = a_value + dt_bias_value
    softplus = tl.where(
        gate_input > 0,
        gate_input + tl.log(1.0 + tl.exp(-gate_input)),
        tl.log(1.0 + tl.exp(gate_input)),
    )
    softplus = tl.where(
        gate_input <= SOFTPLUS_THRESHOLD,
        softplus,
        gate_input,
    )
    g_value = -tl.exp(A_log_value) * softplus
    beta_value = tl.sigmoid(b_value)
    tl.store(g + row * HV + offs_h, g_value, mask=gate_mask)
    tl.store(beta + row * HV + offs_h, beta_value, mask=gate_mask)
'''


NODEGROUP_SIGNATURE = '''


@triton.jit
def _fr13_fixed32_sfwd_conv_postprep_nodegroup8_direct_kernel(
    x,
    conv_state,
    spec_state_indices,
    sticky_guard_ok,
    conv_weights,
    bias,
    a,
    b,
    A_log,
    dt_bias,
    query,
    key,
    value_spec,
    value_tree,
    g,
    beta,
    source_stage,
    conv_tap,
    CONV_STRIDE_ROW: tl.constexpr,
    BANK_ROWS: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STORE_CONV_TAP: tl.constexpr,
    CAPTURE_GUARD: tl.constexpr,
    X_STRIDE_ROW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GATE_BLOCK: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    EMBED_GATE_CTA: tl.constexpr,
):
    """Run four direct eight-node channel groups without gather or shared state."""
    pid_b = tl.program_id(0)
    pid_task = tl.program_id(1)
    channel_tiles: tl.constexpr = C // BLOCK_C
    channel_tasks: tl.constexpr = 4 * channel_tiles
    Q_DIM: tl.constexpr = H * K
    V_DIM: tl.constexpr = HV * V
    GATE_ROWS: tl.constexpr = 2 * BLOCK_C // GATE_BLOCK
    GATE_TASKS: tl.constexpr = N // GATE_ROWS
    if pid_task < channel_tasks:
        pid_group = pid_task // channel_tiles
        pid_c = pid_task - pid_group * channel_tiles
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        x_batch = x + pid_b * N * X_STRIDE_ROW
        stage_batch = source_stage + pid_b * SOURCE_ROWS * C

        bank_row_raw = tl.load(spec_state_indices + pid_b * N).to(tl.int64)
        bank_row_ok = (bank_row_raw >= 0) & (bank_row_raw < BANK_ROWS)
        bank_row = tl.maximum(0, tl.minimum(bank_row_raw, BANK_ROWS - 1))
        if CAPTURE_GUARD:
            tl.atomic_xchg(sticky_guard_ok, 0, mask=~bank_row_ok)
        prior_base = conv_state + bank_row * CONV_STRIDE_ROW + offs_c

        weight_channels = conv_weights + offs_c * WIDTH
        weight_pair_01 = tl.load(weight_channels.to(tl.pointer_type(tl.uint32)))
        weight_pair_23 = tl.load(
            (weight_channels + 2).to(tl.pointer_type(tl.uint32))
        )
        weight_0 = weight_pair_01.to(tl.uint16).to(
            tl.bfloat16, bitcast=True
        )
        weight_1 = (weight_pair_01 >> 16).to(tl.uint16).to(
            tl.bfloat16, bitcast=True
        )
        weight_2 = weight_pair_23.to(tl.uint16).to(
            tl.bfloat16, bitcast=True
        )
        weight_3 = (weight_pair_23 >> 16).to(tl.uint16).to(
            tl.bfloat16, bitcast=True
        )

        bias_value = tl.zeros((BLOCK_C,), dtype=tl.float32)
        if HAS_BIAS:
            bias_value = tl.load(bias + offs_c).to(tl.float32)
'''


NODEGROUP_EMBEDDED_GATING = '''    if EMBED_GATE_CTA:
        if pid_task < GATE_TASKS:
            pid_n_base = pid_task * GATE_ROWS
'''


NODEGROUP_STANDALONE_GATING = '''    else:
        if pid_task >= channel_tasks:
            pid_n_base = (pid_task - channel_tasks) * GATE_ROWS
'''


def _source_function() -> str:
    raw = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(raw)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == SOURCE_FUNCTION
    )
    segment = ast.get_source_segment(raw, node)
    if segment is None:
        raise RuntimeError("cannot recover source SFWD kernel")
    return segment


def _producer_body() -> str:
    source = _source_function()
    lines = source.splitlines()
    first_body = next(
        index for index, line in enumerate(lines) if line.startswith("    pid_b =")
    )
    body = lines[first_body:]
    expected_prefix = [
        "    pid_b = tl.program_id(0)",
        "    pid_c = tl.program_id(1)",
        "    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)",
    ]
    if body[:3] != expected_prefix:
        raise RuntimeError("frontier-5 producer prologue drifted")
    body = body[2:]
    body = [line for line in body if "out_batch = out +" not in line]
    joined = "\n".join(body)
    pattern = re.compile(
        r"^    tl\.store\(out_batch \+ (\d+) \* C \+ offs_c, activated_(\d+)\)$",
        flags=re.MULTILINE,
    )

    def replace_store(match: re.Match[str]) -> str:
        node_a, node_b = match.groups()
        if node_a != node_b:
            raise RuntimeError("frontier-5 output node mismatch")
        return "\n".join(
            (
                "    _fr13_store_fixed32_conv_outputs(",
                "        query,",
                "        key,",
                "        value_spec,",
                "        value_tree,",
                "        conv_tap,",
                f"        activated_{node_a},",
                "        pid_b,",
                f"        {node_a},",
                "        offs_c,",
                "        N,",
                "        C,",
                "        Q_DIM,",
                "        V_DIM,",
                "        STORE_CONV_TAP,",
                "    )",
            )
        )

    joined, replacements = pattern.subn(replace_store, joined)
    if replacements != 32:
        raise RuntimeError(
            f"expected 32 frontier-5 output stores, found {replacements}"
        )
    if "out_batch" in joined or "tl.store(out" in joined:
        raise RuntimeError("full conv intermediate store survived generation")
    unsafe_bank_load = """    bank_row = tl.load(spec_state_indices + pid_b * N).to(tl.int64)
    prior_base = conv_state + bank_row * CONV_STRIDE_ROW + offs_c
"""
    guarded_bank_load = """    bank_row_raw = tl.load(spec_state_indices + pid_b * N).to(tl.int64)
    bank_row_ok = (bank_row_raw >= 0) & (bank_row_raw < BANK_ROWS)
    bank_row = tl.maximum(0, tl.minimum(bank_row_raw, BANK_ROWS - 1))
    if CAPTURE_GUARD:
        # Valid replays perform no store. The first invalid replay makes the
        # committer's existing async assertion fail permanently, while the
        # clamped row prevents an out-of-bounds read before that assertion.
        tl.atomic_xchg(sticky_guard_ok, 0, mask=~bank_row_ok)
    prior_base = conv_state + bank_row * CONV_STRIDE_ROW + offs_c
"""
    if joined.count(unsafe_bank_load) != 1:
        raise RuntimeError("frontier-5 bank-row load drifted")
    joined = joined.replace(unsafe_bank_load, guarded_bank_load, 1)
    return joined + "\n"


def _descriptorless_sources() -> tuple[tuple[int, int, int], ...]:
    """Evaluate only the pure topology declarations from the Triton module."""
    raw = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(raw)
    names = {
        "FIXED32_ROWS",
        "fixed32_derived_parent_q",
        "fixed32_descriptorless_sources",
    }
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    rows = namespace["fixed32_descriptorless_sources"]()
    if (
        not isinstance(rows, tuple)
        or len(rows) != 32
        or any(
            not isinstance(triple, tuple)
            or len(triple) != 3
            or any(type(row) is not int for row in triple)
            for triple in rows
        )
    ):
        raise RuntimeError("fixed32 descriptorless source topology drifted")
    for node, triple in enumerate(rows):
        if any(row < 0 or row >= node + 3 for row in triple):
            raise RuntimeError(
                f"fixed32 descriptorless source row {node} escaped its prefix"
            )
    return rows


def _nodegroup_store(node: int, *, indent: str) -> list[str]:
    return [
        f"{indent}_fr13_store_fixed32_conv_outputs(",
        f"{indent}    query,",
        f"{indent}    key,",
        f"{indent}    value_spec,",
        f"{indent}    value_tree,",
        f"{indent}    conv_tap,",
        f"{indent}    activated_{node},",
        f"{indent}    pid_b,",
        f"{indent}    {node},",
        f"{indent}    offs_c,",
        f"{indent}    N,",
        f"{indent}    C,",
        f"{indent}    Q_DIM,",
        f"{indent}    V_DIM,",
        f"{indent}    STORE_CONV_TAP,",
        f"{indent})",
    ]


def _nodegroup_branch(
    group: int,
    sources: tuple[tuple[int, int, int], ...],
) -> str:
    first = group * NODEGROUP_ROWS
    nodes = tuple(range(first, first + NODEGROUP_ROWS))
    keyword = "if" if group == 0 else "elif"
    indent = " " * 8
    body_indent = " " * 12
    lines = [f"{indent}{keyword} pid_group == {group}:"]
    prior_rows = sorted(
        {row for node in nodes for row in sources[node] if row < 3}
        | ({0, 1, 2} if group == 0 else set())
    )
    for row in prior_rows:
        suffix = "" if row == 0 else f" + {row} * C"
        lines.append(
            f"{body_indent}prior_{row} = tl.load(prior_base{suffix})"
        )

    loaded_x: set[int] = set()
    for node in nodes:
        x_rows = tuple(row - 3 for row in sources[node] if row >= 3) + (node,)
        for x_row in x_rows:
            if x_row in loaded_x:
                continue
            loaded_x.add(x_row)
            lines.append(
                f"{body_indent}x_g{group}_{x_row} = tl.load("
            )
            lines.append(
                f"{body_indent}    x_batch + {x_row} * X_STRIDE_ROW + offs_c"
            )
            lines.append(f"{body_indent})")

        operands = tuple(
            f"prior_{row}" if row < 3 else f"x_g{group}_{row - 3}"
            for row in sources[node]
        ) + (f"x_g{group}_{node}",)
        for tap, operand in enumerate(operands):
            lines.append(
                f"{body_indent}product_{tap} = ("
                f"{operand} * weight_{tap}"
                ").to(tl.bfloat16).to(tl.float32)"
            )
            if tap == 0:
                lines.append(f"{body_indent}acc = bias_value + product_0")
            else:
                lines.append(f"{body_indent}acc = acc + product_{tap}")
        lines.append(
            f"{body_indent}activated_{node} = acc / "
            "(1.0 + tl.exp(0.0 - acc))"
        )
        lines.extend(_nodegroup_store(node, indent=body_indent))
        lines.extend(
            (
                f"{body_indent}tl.store(",
                f"{body_indent}    stage_batch + ((WIDTH - 1) + {node}) * C + offs_c,",
                f"{body_indent}    x_g{group}_{node},",
                f"{body_indent})",
            )
        )

    expected_x = {
        row - 3
        for node in nodes
        for row in sources[node]
        if row >= 3
    } | set(nodes)
    if loaded_x != expected_x:
        raise RuntimeError(f"nodegroup {group} x-row coverage drifted")
    if group == 0:
        lines.extend(
            (
                f"{body_indent}tl.store(stage_batch + offs_c, prior_0)",
                f"{body_indent}tl.store(stage_batch + C + offs_c, prior_1)",
                f"{body_indent}tl.store(stage_batch + 2 * C + offs_c, prior_2)",
                f"{body_indent}tl.store(",
                f"{body_indent}    stage_batch + (SOURCE_ROWS - 1) * C + offs_c,",
                f"{body_indent}    0.0,",
                f"{body_indent})",
            )
        )
    return "\n".join(lines) + "\n"


def _nodegroup_kernel() -> str:
    sources = _descriptorless_sources()
    branches = "".join(_nodegroup_branch(group, sources) for group in range(4))
    return (
        NODEGROUP_SIGNATURE
        + branches
        + NODEGROUP_EMBEDDED_GATING
        + _indent_block(GATING_BODY, 12)
        + NODEGROUP_STANDALONE_GATING
        + _indent_block(GATING_BODY, 12)
    )


def _indent_block(source: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else "" for line in source.splitlines()) + "\n"


def generate() -> str:
    producer = _producer_body()
    return (
        PREAMBLE
        + SIGNATURE
        + _indent_block(producer, 4)
        + EMBEDDED_GATING
        + _indent_block(GATING_BODY, 12)
        + STANDALONE_PROLOGUE
        + _indent_block(producer, 8)
        + STANDALONE_GATING
        + _indent_block(GATING_BODY, 12)
        + _nodegroup_kernel()
    )


def main() -> None:
    OUTPUT.write_text(generate(), encoding="utf-8")


if __name__ == "__main__":
    main()
