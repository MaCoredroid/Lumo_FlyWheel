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
):
    """Fuse one fixed32 layer's conv, recurrence outputs, and post-prep."""
    pid_b = tl.program_id(0)
    pid_task = tl.program_id(1)
    channel_tasks: tl.constexpr = C // BLOCK_C
    Q_DIM: tl.constexpr = H * K
    V_DIM: tl.constexpr = HV * V
    if pid_task < channel_tasks:
        pid_c = pid_task
'''


GATING = '''    else:
        GATE_ROWS: tl.constexpr = 2 * BLOCK_C // GATE_BLOCK
        pid_n_base = (pid_task - channel_tasks) * GATE_ROWS
        offs_n = pid_n_base + tl.arange(0, GATE_ROWS)[:, None]
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
    return "\n".join(
        "    " + line if line else "" for line in joined.splitlines()
    ) + "\n"


def generate() -> str:
    return PREAMBLE + SIGNATURE + _producer_body() + GATING


def main() -> None:
    OUTPUT.write_text(generate(), encoding="utf-8")


if __name__ == "__main__":
    main()
