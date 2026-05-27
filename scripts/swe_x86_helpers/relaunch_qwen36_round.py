#!/usr/bin/env python3
"""Parameterized relaunch for the Round-5 B=4 sweep: one script for all rounds.

  --config D            full T1+T2+T3+T4 suffix stack (the original config D)
  --config E --mtp N    Qwen3.6 native MTP head, num_speculative_tokens=N (no suffix)

Both variants get the per-agent spec-decode STEP TRACE patch: a source-edit of
Scheduler.make_spec_decoding_stats (called per-request per-step with request_id)
that appends {ts,rid,draft,acc} rows to /logs/per_req_spec_trace.jsonl (a
bind-mounted host path), so per-agent acceptance + per-step timing stay clean at
ANY batch size (B>1) -- the global /metrics deltas can't separate concurrent
streams, this can (rid carries the proxy's session-prefixed request id).
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

REPO = Path("/home/mark/shared/lumoFlyWheel")
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
from scripts.run_track_b_loop import _track_b_runtime_prelaunch_shell
from lumo_flywheel_serving.model_server import ModelServer

_KEEP_MARKER = "applied forced tool_choice parser patch')\nPY\n"

# Source-edit (NOT monkeypatch -- prelaunch patches the file before vLLM imports
# it) of Scheduler.make_spec_decoding_stats to emit per-request per-step rows.
# The inner python builds the injected source with chr(10) for EVERY newline
# (both line separators and the JSON-line terminator) so no backslash-escape has
# to survive the raw-string -> heredoc -> inner-python -> written-source layers.
_SPEC_TRACE_BLOCK = r'''
python3 - <<'LUMOSPECTRACE'
from pathlib import Path
nl = chr(10)
p = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py')
text = p.read_text()
sentinel = '# LUMO_PER_AGENT_SPEC_TRACE'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] per-agent spec trace already present')
else:
    anchor = '    ) -> SpecDecodingStats | None:' + nl + '        if not self.log_stats or not num_draft_tokens:'
    if anchor not in text:
        raise RuntimeError('make_spec_decoding_stats anchor not found for per-agent spec trace')
    inject = nl.join([
        '    ) -> SpecDecodingStats | None:',
        '        ' + sentinel,
        '        import json as _lj, time as _lt, os as _lo',
        '        if _lo.environ.get("LUMO_FB_ASSERT_WIDTH") == "1":',
        '            _ctrl_depth = None',
        '            try:',
        '                _ctrl_path = _lo.environ.get("LUMO_FB_CONTROL_FILE", "/logs/fb_control.json")',
        '                if _lo.path.exists(_ctrl_path):',
        '                    with open(_ctrl_path) as _cf:',
        '                        _ctrl_payload = _lj.load(_cf)',
        '                    _ctrl_depth = int(_ctrl_payload.get("depth")) if _ctrl_payload.get("depth") is not None else None',
        '                elif _lo.environ.get("LUMO_FB_DEPTH"):',
        '                    _ctrl_depth = int(_lo.environ.get("LUMO_FB_DEPTH"))',
        '            except Exception:',
        '                _ctrl_depth = None',
        '            if _ctrl_depth is not None and int(num_draft_tokens) != int(_ctrl_depth):',
        '                raise RuntimeError(f"LUMO_FB_ASSERT_WIDTH failed: verify_width={num_draft_tokens} active_depth={_ctrl_depth}")',
        '        try:',
        '            global _LUMO_SPEC_FH',
        '            try:',
        '                _LUMO_SPEC_FH',
        '            except NameError:',
        '                _LUMO_SPEC_FH = open(_lo.environ.get("LUMO_PER_REQ_SPEC_TRACE", "/logs/per_req_spec_trace.jsonl"), "a", buffering=1)',
        '            _linv = (num_invalid_spec_tokens.get(request_id, 0) if num_invalid_spec_tokens else 0)',
        '            _LUMO_SPEC_FH.write(_lj.dumps({"ts": round(_lt.time(), 4), "rid": request_id, "draft": num_draft_tokens, "proposal_width": num_draft_tokens, "verify_width": num_draft_tokens, "acc": num_accepted_tokens, "inv": _linv}) + chr(10))',
        '        except Exception:',
        '            pass',
        '        if not self.log_stats or not num_draft_tokens:',
    ])
    text = text.replace(anchor, inject, 1)
    p.write_text(text)
    import py_compile, tempfile
    py_compile.compile(str(p), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied per-agent spec-decode step trace patch')
LUMOSPECTRACE
'''


# F_b kernel-row foundation: let the GDN SSM recurrent update read its initial
# state from one state slot and write the evolved per-token states to another
# slot table. This is the primitive needed for no-copy K-path rows: siblings
# read the shared prefix state, but each row stores its private evolution.
_FB_KERNEL_ROWS_BLOCK = r"""
python3 - <<'LUMOFBKERNELROWS'
from pathlib import Path

fg = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py')
text = fg.read_text()
sentinel = '# LUMO_FB_KERNEL_ROWS_SSM'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b kernel-row SSM patch already present')
else:
    old = '"IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,'
    new = old + '\n        "HAS_INITIAL_STATE_INDICES": lambda args: args["initial_state_indices"] is not None,'
    if old not in text:
        raise RuntimeError('F_b kernel-row SSM heuristic anchor not found')
    text = text.replace(old, new, 1)

    old = '''    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
'''
    new = '''    cu_seqlens,
    ssm_state_indices,
    initial_state_indices,
    num_accepted_tokens,
    scale,
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row SSM kernel arg anchor not found')
    text = text.replace(old, new, 1)

    old = '''    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):'''
    new = '''    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    HAS_INITIAL_STATE_INDICES: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    # LUMO_FB_KERNEL_ROWS_SSM'''
    if old not in text:
        raise RuntimeError('F_b kernel-row SSM constexpr anchor not found')
    text = text.replace(old, new, 1)

    old = '''        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            # Load state index and check for PAD_SLOT_ID (-1)
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                tl.int64
            )
            # Skip if state index is invalid (PAD_SLOT_ID = -1)
            if state_idx < 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K'''
    new = '''        if IS_CONTINUOUS_BATCHING:
            if HAS_INITIAL_STATE_INDICES:
                # F_b kernel rows: read the shared prefix state from a separate
                # read-only slot, while stores below use ssm_state_indices as
                # the private per-row write table.
                state_idx = tl.load(initial_state_indices + i_n).to(tl.int64)
            else:
                if IS_SPEC_DECODING:
                    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                else:
                    i_t = 0
                # Load state index and check for PAD_SLOT_ID (-1)
                state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                    tl.int64
                )
            # Skip if state index is invalid (PAD_SLOT_ID = -1)
            if state_idx < 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K'''
    if old not in text:
        raise RuntimeError('F_b kernel-row SSM read anchor not found')
    text = text.replace(old, new, 1)

    old = '''    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):'''
    new = '''    ssm_state_indices: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):'''
    if old not in text:
        raise RuntimeError('F_b kernel-row SSM wrapper arg anchor not found')
    text = text.replace(old, new, 1)

    old = '''        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
'''
    new = '''        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        initial_state_indices=initial_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row SSM launch arg anchor not found')
    text = text.replace(old, new, 1)

    fg.write_text(text)
    import py_compile
    py_compile.compile(str(fg), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b kernel-row SSM read/write patch')

gl = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py')
text = gl.read_text()
sentinel = '# LUMO_FB_KERNEL_ROWS_GDN_LINEAR'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b kernel-row gdn_linear patch already present')
else:
    old = '''        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
'''
    new = '''        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        # LUMO_FB_KERNEL_ROWS_GDN_LINEAR: optional separate read slot for SSM
        # kernel rows.  When None, upstream read/write semantics are unchanged.
        spec_initial_state_indices_tensor = getattr(
            attn_metadata, "spec_initial_state_indices_tensor", None)
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row gdn_linear metadata anchor not found')
    text = text.replace(old, new, 1)

    old = '''                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
'''
    new = '''                    ssm_state_indices=spec_state_indices_tensor,
                    initial_state_indices=spec_initial_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row gdn_linear fused call anchor not found')
    text = text.replace(old, new, 1)

    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b kernel-row gdn_linear SSM hook')

ga = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py')
text = ga.read_text()
sentinel = '# LUMO_FB_KERNEL_ROWS_GDN_ATTN'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b kernel-row gdn_attn patch already present')
else:
    old = '''from dataclasses import dataclass

import torch
'''
    new = '''from dataclasses import dataclass
import os as _lumo_fb_kernel_os

import torch
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row gdn_attn import anchor not found')
    text = text.replace(old, new, 1)

    old = '''    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
'''
    new = '''    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    # LUMO_FB_KERNEL_ROWS_GDN_ATTN: separate read-only prefix state slot for
    # no-copy path rows. spec_state_indices_tensor remains the private write
    # table used by the recurrent kernels.
    spec_initial_state_indices_tensor: torch.Tensor | None = None
    spec_initial_state_slot_tensor: torch.Tensor | None = None
    spec_write_state_slot_tensor: torch.Tensor | None = None
    non_spec_state_indices_tensor: torch.Tensor | None = (
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row gdn_attn dataclass anchor not found')
    text = text.replace(old, new, 1)

    old = '''        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
'''
    new = '''        self.spec_initial_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_initial_state_slot_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_write_state_slot_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row gdn_attn cudagraph alloc anchor not found')
    text = text.replace(old, new, 1)

    old = '''            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
'''
    new = '''            spec_state_indices_tensor = None
            spec_initial_state_indices_tensor = None
            spec_initial_state_slot_tensor = None
            spec_write_state_slot_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row gdn_attn nonspec init anchor not found')
    text = text.replace(old, new, 1)

    old = '''            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]
'''
    new = '''            # F_b kernel-row convention: block-table column 0 carries the
            # shared read-only prefix state.  Columns 1..num_spec+1 are the
            # private write table used by recurrent kernels.
            spec_initial_state_indices_tensor = None
            spec_initial_state_slot_tensor = None
            spec_write_state_slot_tensor = None
            if _lumo_fb_kernel_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1":
                _fb_write_end = int(self.num_spec + 2)
                if block_table_tensor.size(1) < _fb_write_end:
                    raise RuntimeError(
                        "LUMO_FB_KERNEL_ROWS requires block_table write columns "
                        f"through {_fb_write_end - 1}, got width {block_table_tensor.size(1)}")
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks, 1:_fb_write_end
                ]
                spec_initial_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks, 0
                ].contiguous()
                _fb_n = int(num_spec_decodes)
                spec_initial_state_slot_tensor = None
                spec_write_state_slot_tensor = torch.zeros(
                    (_fb_n,), dtype=torch.int32, device=query_start_loc.device)

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row gdn_attn spec initial anchor not found')
    text = text.replace(old, new, 1)

    old = '''            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
'''
    new = '''            if spec_initial_state_indices_tensor is not None:
                self.spec_initial_state_indices_tensor[:num_spec_decodes].copy_(
                    spec_initial_state_indices_tensor, non_blocking=True
                )
                spec_initial_state_indices_tensor = (
                    self.spec_initial_state_indices_tensor[:batch_size]
                )
                spec_initial_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)
                spec_initial_state_slot_tensor = None
                self.spec_write_state_slot_tensor[:num_spec_decodes].copy_(
                    spec_write_state_slot_tensor, non_blocking=True
                )
                spec_write_state_slot_tensor = (
                    self.spec_write_state_slot_tensor[:batch_size]
                )
                spec_write_state_slot_tensor[num_spec_decodes:].fill_(0)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row gdn_attn cudagraph copy anchor not found')
    text = text.replace(old, new, 1)

    old = '''            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
'''
    new = '''            spec_state_indices_tensor=spec_state_indices_tensor,
            spec_initial_state_indices_tensor=spec_initial_state_indices_tensor,
            spec_initial_state_slot_tensor=spec_initial_state_slot_tensor,
            spec_write_state_slot_tensor=spec_write_state_slot_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row gdn_attn metadata ctor anchor not found')
    text = text.replace(old, new, 1)

    ga.write_text(text)
    import py_compile
    py_compile.compile(str(ga), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b kernel-row gdn_attn metadata hook')

text = gl.read_text()
sentinel = '# LUMO_FB_KERNEL_ROWS_GDN_LINEAR_CONV'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b kernel-row gdn_linear conv patch already present')
else:
    old = '''        spec_initial_state_indices_tensor = getattr(
            attn_metadata, "spec_initial_state_indices_tensor", None)
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
'''
    new = '''        spec_initial_state_indices_tensor = getattr(
            attn_metadata, "spec_initial_state_indices_tensor", None)
        # LUMO_FB_KERNEL_ROWS_GDN_LINEAR_CONV: conv update uses the same
        # block-table convention as SSM: read from shared prefix slot, write to
        # the row-private slot.
        spec_initial_state_slot_tensor = getattr(
            attn_metadata, "spec_initial_state_slot_tensor", None)
        spec_write_state_slot_tensor = getattr(
            attn_metadata, "spec_write_state_slot_tensor", None)
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row gdn_linear conv metadata anchor not found')
    text = text.replace(old, new, 1)

    old = '''                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
'''
    new = '''                conv_state_indices=(
                    spec_state_indices_tensor
                    if spec_initial_state_indices_tensor is not None
                    else spec_state_indices_tensor[:, 0][
                        : attn_metadata.num_spec_decodes
                    ]
                ),
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                block_idx_last_scheduled_token=spec_write_state_slot_tensor,
                initial_state_idx=spec_write_state_slot_tensor,
                initial_state_indices=spec_initial_state_indices_tensor,
                validate_data=False,
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row gdn_linear conv call anchor not found')
    text = text.replace(old, new, 1)

    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b kernel-row gdn_linear conv hook')

cc = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py')
text = cc.read_text()
sentinel = '# LUMO_FB_KERNEL_ROWS_CONV_READ'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b kernel-row causal_conv read patch already present')
else:
    old = '''    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # (batch, dim, seqlen)
'''
    new = '''    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    initial_state_indices_ptr,  # (batch,), physical read state ids
    o_ptr,  # (batch, dim, seqlen)
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row causal_conv kernel arg anchor not found')
    text = text.replace(old, new, 1)

    old = '''    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
'''
    new = '''    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    HAS_INITIAL_STATE_INDICES: tl.constexpr,
    FB_WRITE_COLS: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row causal_conv constexpr anchor not found')
    text = text.replace(old, new, 1)

    old = '''    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
    ).to(tl.int64)
'''
    new = '''    # cache_idx
    # LUMO_FB_KERNEL_ROWS_CONV_READ: no-copy path rows read the shared prefix
    # conv state by physical id, while conv_state_indices_ptr remains the
    # private write table.
    if HAS_INITIAL_STATE_INDICES:
        conv_states_input_coord = tl.load(initial_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        conv_states_input_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
        ).to(tl.int64)
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row causal_conv read anchor not found')
    text = text.replace(old, new, 1)

    old = '''    # Get the state from the initial_state_idx
    # cache_idx
    conv_states_offset = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
    ).to(tl.int64)
    conv_state_ptrs_target = (
        conv_state_ptr
        + (conv_states_offset * stride_conv_state_seq)  # Offset from seq
        + (idx_feats * stride_conv_state_dim)
    )[None, :] + (  # [BLOCK_N,]
        idx_tokens * stride_conv_state_tok
    )[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)
'''
    new = '''    # Get the state from the initial_state_idx
    # cache_idx
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    if HAS_INITIAL_STATE_INDICES:
        # LUMO_FB_KERNEL_ROWS_CONV_FANOUT: SSM stores one state per draft
        # depth, while the conv kernel keeps a single multi-offset state that
        # can serve any later accepted-token offset. Duplicate that conv state
        # into every private write column so promoting any accepted-depth block
        # carries a valid conv+SSM pair.
        for _lumo_fb_i in tl.static_range(0, FB_WRITE_COLS):
            conv_states_offset = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_state_indices
                + _lumo_fb_i
            ).to(tl.int64)
            conv_state_ptrs_target = (
                conv_state_ptr
                + (conv_states_offset * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )[None, :] + (idx_tokens * stride_conv_state_tok)[:, None]
            tl.store(conv_state_ptrs_target, new_conv_state, mask)
    else:
        conv_states_offset = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
        ).to(tl.int64)
        conv_state_ptrs_target = (
            conv_state_ptr
            + (conv_states_offset * stride_conv_state_seq)  # Offset from seq
            + (idx_feats * stride_conv_state_dim)
        )[None, :] + (  # [BLOCK_N,]
            idx_tokens * stride_conv_state_tok
        )[:, None]
        tl.store(conv_state_ptrs_target, new_conv_state, mask)
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row causal_conv fanout anchor not found')
    text = text.replace(old, new, 1)

    old = '''    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
):
'''
    new = '''    initial_state_idx: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    validate_data=False,
):
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row causal_conv wrapper arg anchor not found')
    text = text.replace(old, new, 1)

    old = '''        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
'''
    new = '''        block_idx_last_scheduled_token,
        initial_state_idx,
        initial_state_indices,
        out,
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row causal_conv launch arg anchor not found')
    text = text.replace(old, new, 1)

    old = '''        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
'''
    new = '''        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        HAS_INITIAL_STATE_INDICES=initial_state_indices is not None,
        FB_WRITE_COLS=max_query_len if initial_state_indices is not None else 1,
        NP2_STATELEN=np2_statelen,
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row causal_conv launch meta anchor not found')
    text = text.replace(old, new, 1)

    cc.write_text(text)
    import py_compile
    py_compile.compile(str(cc), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b kernel-row causal_conv read hook')

ma = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/abstract.py')
text = ma.read_text()
sentinel = '# LUMO_FB_KERNEL_ROWS_EXTRA_STATE_BLOCK'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b kernel-row Mamba extra-block patch already present')
else:
    old = '''from abc import abstractmethod
from collections.abc import Iterable

import torch
'''
    new = '''from abc import abstractmethod
from collections.abc import Iterable
import os as _lumo_fb_kernel_os

import torch
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row Mamba abstract import anchor not found')
    text = text.replace(old, new, 1)

    old = '''            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            ),
'''
    new = '''            num_speculative_blocks=(
                (vllm_config.speculative_config.num_speculative_tokens
                 + (1 if _lumo_fb_kernel_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1" else 0))
                if vllm_config.speculative_config
                else 0
            ),  # LUMO_FB_KERNEL_ROWS_EXTRA_STATE_BLOCK
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row Mamba abstract spec-block anchor not found')
    text = text.replace(old, new, 1)

    ma.write_text(text)
    import py_compile
    py_compile.compile(str(ma), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b kernel-row Mamba extra state-block patch')
LUMOFBKERNELROWS
"""


# Force the TreeAttention backend for decoder self-attention when a BRANCHING
# speculative_token_tree is configured. Needed because (1) vLLM 0.19.0 does not
# honor VLLM_ATTENTION_BACKEND for model attention selection, and (2) tree spec
# needs tree attention for BOTH the draft proposal (build_for_drafting) AND the
# target verify step (build -> tree attention mask), so a draft-only override is
# insufficient. Linear chains (len(tree_choices) == max depth) are left on the
# auto-selected backend, so config E behaviour is unchanged. Source-edit (not a
# monkeypatch) so it lands before the engine imports the selector.
_TREE_ATTN_BLOCK = r'''
python3 - <<'LUMOTREEATTN'
from pathlib import Path
nl = chr(10)
p = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/selector.py')
text = p.read_text()
sentinel = '# LUMO_FORCE_TREE_ATTN'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree-attn force patch already present')
else:
    anchor = nl.join([
        '    return _cached_get_attn_backend(',
        '        backend=vllm_config.attention_config.backend,',
        '        attn_selector_config=attn_selector_config,',
        '        num_heads=num_heads,',
        '    )',
    ])
    if anchor not in text:
        raise RuntimeError('selector get_attn_backend anchor not found for tree-attn force')
    inject = nl.join([
        '    ' + sentinel + ': branching speculative_token_tree -> TreeAttention',
        '    # for decoder self-attn (target verify + draft both need the tree mask).',
        '    _lumo_backend = vllm_config.attention_config.backend',
        '    try:',
        '        _lspec = getattr(vllm_config, "speculative_config", None)',
        '        _ltree = getattr(_lspec, "speculative_token_tree", None) if _lspec is not None else None',
        '        if _ltree:',
        '            import ast as _last',
        '            from vllm.v1.attention.backends.registry import AttentionBackendEnum as _LABE',
        '            _ltc = _last.literal_eval(_ltree)',
        '            _ldepth = max(len(_t) for _t in _ltc)',
        '            if len(_ltc) > _ldepth and "encoder" not in str(attn_type).lower():',
        '                _lumo_backend = _LABE.TREE_ATTN',
        '    except Exception:',
        '        pass',
        '    return _cached_get_attn_backend(',
        '        backend=_lumo_backend,',
        '        attn_selector_config=attn_selector_config,',
        '        num_heads=num_heads,',
        '    )',
    ])
    text = text.replace(anchor, inject, 1)
    p.write_text(text)
    import py_compile
    py_compile.compile(str(p), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree-attn force patch (selector.py)')
LUMOTREEATTN
'''


# vLLM 0.19.0's propose_tree references self.positions unconditionally, but
# M-RoPE models (Qwen3.6 is multimodal Qwen3_5) allocate self.mrope_positions
# instead -> AttributeError at decode. The linear propose() path is mrope-aware
# (via _get_positions/_set_positions) but propose_tree is not. This patch makes
# propose_tree mrope-aware for TEXT-ONLY inputs (all 3 M-RoPE dims identical, per
# vLLM's own note): reduce the incoming 3D positions to 1D for tree slot math,
# take device/dtype off input_ids, write back via _set_positions (broadcast 1D
# ->3D for mrope), and feed the draft model via _get_positions (3D for mrope).
# Lossless gate (B-1/B-2/B-3) still required before any SWE run: if the TARGET
# verify's tree positions are also mrope-wrong, byte-exact greedy match catches it.
_MROPE_TREE_BLOCK = r'''
python3 - <<'LUMOMROPETREE'
from pathlib import Path
nl = chr(10)
p = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py')
text = p.read_text()
sentinel = '# LUMO_MROPE_TREE'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] propose_tree M-RoPE patch already present')
else:
    edits = [
        (
            '        assert isinstance(tree_attn_metadata_builder, TreeAttentionMetadataBuilder)' + nl,
            '        assert isinstance(tree_attn_metadata_builder, TreeAttentionMetadataBuilder)' + nl + nl.join([
                '        ' + sentinel + ': text-only M-RoPE -> 1D positions for tree slot math',
                '        if self.uses_mrope and positions.dim() > 1:',
                '            positions = positions[0]',
                '',
            ]),
        ),
        (
            '            0, device=self.positions.device, dtype=self.positions.dtype',
            '            0, device=self.input_ids.device, dtype=torch.int64',
        ),
        (
            '            self.positions[:num_tokens] = tree_positions.view(-1)',
            nl.join([
                '            _lt_tp = tree_positions.view(-1)',
                '            self._set_positions(num_tokens, _lt_tp.unsqueeze(0).expand(3, -1) if self.uses_mrope else _lt_tp)',
            ]),
        ),
        (
            nl.join([
                '                last_hidden_states, hidden_states = self.model(',
                '                    input_ids=self.input_ids[:num_input_tokens],',
                '                    positions=self.positions[:num_input_tokens],',
                '                    hidden_states=self.hidden_states[:num_input_tokens],',
                '                    inputs_embeds=None,',
                '                )',
            ]),
            nl.join([
                '                # LUMO_MROPE_TREE(mm): multimodal MTP draft expects inputs_embeds',
                '                # (matches the compiled signature of the linear propose path).',
                '                if self.supports_mm_inputs:',
                '                    self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(',
                '                        self.input_ids[:num_tokens],',
                '                        multimodal_embeddings=None,',
                '                        is_multimodal=None,',
                '                    )',
                '                    _lt_in_ids = None',
                '                    _lt_in_emb = self.inputs_embeds[:num_input_tokens]',
                '                else:',
                '                    _lt_in_ids = self.input_ids[:num_input_tokens]',
                '                    _lt_in_emb = None',
                '                _lt_ret = self.model(',
                '                    input_ids=_lt_in_ids,',
                '                    positions=self._get_positions(num_input_tokens),',
                '                    hidden_states=self.hidden_states[:num_input_tokens],',
                '                    inputs_embeds=_lt_in_emb,',
                '                )',
                '                # LUMO_MROPE_TREE: MTP returns a single tensor, not a tuple',
                '                if self.model_returns_tuple():',
                '                    last_hidden_states, hidden_states = _lt_ret',
                '                else:',
                '                    last_hidden_states = hidden_states = _lt_ret',
            ]),
        ),
        # Env-gated per-level draft-token logger (LUMO_TREE_DRAFT_DEBUG=1) to
        # localize the depth-2 acceptance cliff: logs each level's proposed
        # token ids + base position, so we can diff the tree's top-1 path
        # against the linear chain / actual output.
        (
            '        return draft_token_ids_list',
            nl.join([
                '        try:',
                '            import os as _do',
                '            if _do.environ.get("LUMO_TREE_DRAFT_DEBUG") == "1":',
                '                import json as _dj, time as _dt',
                '                global _LUMO_TREE_DBG_FH',
                '                try:',
                '                    _LUMO_TREE_DBG_FH',
                '                except NameError:',
                '                    _LUMO_TREE_DBG_FH = open("/logs/tree_draft_debug.jsonl", "a", buffering=1)',
                '                _dbp = int(positions.flatten()[0].item())',
                '                _dlv = [t.tolist() for t in draft_token_ids_list]',
                '                _LUMO_TREE_DBG_FH.write(_dj.dumps({"ts": round(_dt.time(), 4), "base_pos": _dbp, "levels": _dlv}) + chr(10))',
                '        except Exception:',
                '            pass',
                '        return draft_token_ids_list',
            ]),
        ),
    ]
    for anchor, new in edits:
        if text.count(anchor) != 1:
            raise RuntimeError('propose_tree M-RoPE anchor not unique: ' + repr(anchor[:60]))
        text = text.replace(anchor, new, 1)
    p.write_text(text)
    import py_compile
    py_compile.compile(str(p), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied propose_tree M-RoPE patch (eagle.py)')
LUMOMROPETREE
'''


_TREE_REJECTION_BLOCK = r'''
python3 - <<'LUMOTREEREJECT'
from pathlib import Path
nl = chr(10)

rs = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py')
text = rs.read_text()
sentinel = '# LUMO_TREE_REJECTION'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree rejection sampler patch already present')
else:
    helper_anchor = nl + nl + 'def rejection_sample('
    helper = nl.join([
        '',
        '',
        sentinel + ': sample target tokens for tree-walk acceptance',
        'def _lumo_sample_token_ids_from_logits(',
        '    logits: torch.Tensor,',
        '    cu_num_draft_tokens: torch.Tensor,',
        '    sampling_metadata: SamplingMetadata,',
        ') -> torch.Tensor:',
        '    greedy_token_ids = logits.argmax(dim=-1)',
        '    if sampling_metadata.all_greedy:',
        '        return greedy_token_ids.to(torch.int32)',
        '    probs = logits.softmax(dim=-1, dtype=torch.float32)',
        '    q = torch.empty_like(probs)',
        '    q.exponential_()',
        '    sampled_token_ids = probs.div(q).argmax(dim=-1)',
        '    if not sampling_metadata.all_random:',
        '        is_greedy = expand_batch_to_tokens(',
        '            sampling_metadata.temperature,',
        '            cu_num_draft_tokens,',
        '            logits.shape[0],',
        '        ) == GREEDY_TEMPERATURE',
        '        sampled_token_ids = torch.where(',
        '            is_greedy, greedy_token_ids, sampled_token_ids',
        '        )',
        '    return sampled_token_ids.to(torch.int32)',
    ])
    if helper_anchor not in text:
        raise RuntimeError('rejection_sample helper anchor not found')
    text = text.replace(helper_anchor, helper + nl + nl + 'def rejection_sample(', 1)

    prep_anchor = nl.join([
        '        target_logits = apply_sampling_constraints(',
        '            target_logits,',
        '            metadata.cu_num_draft_tokens,',
        '            sampling_metadata,',
        '        )',
        '',
        '        output_token_ids = rejection_sample(',
    ])
    prep_inject = nl.join([
        '        target_logits = apply_sampling_constraints(',
        '            target_logits,',
        '            metadata.cu_num_draft_tokens,',
        '            sampling_metadata,',
        '        )',
        '',
        '        lumo_tree_parent_indices = getattr(metadata, "tree_parent_indices", None)',
        '        lumo_tree_token_ids = None',
        '        if lumo_tree_parent_indices is not None:',
        '            tree_self_logits = logits[metadata.tree_self_logits_indices]',
        '            tree_self_logits = tree_self_logits.to(torch.float32)',
        '            if not self.is_processed_logprobs_mode:',
        '                tree_self_logits = tree_self_logits.clone()',
        '            tree_self_logits = self.apply_logits_processors(',
        '                tree_self_logits, sampling_metadata, metadata',
        '            )',
        '            tree_self_logits = apply_sampling_constraints(',
        '                tree_self_logits,',
        '                metadata.cu_num_draft_tokens,',
        '                sampling_metadata,',
        '            )',
        '            lumo_parent_token_ids = _lumo_sample_token_ids_from_logits(',
        '                target_logits, metadata.cu_num_draft_tokens, sampling_metadata',
        '            )',
        '            lumo_self_token_ids = _lumo_sample_token_ids_from_logits(',
        '                tree_self_logits, metadata.cu_num_draft_tokens, sampling_metadata',
        '            )',
        '            lumo_tree_token_ids = torch.stack(',
        '                [lumo_parent_token_ids, lumo_self_token_ids], dim=0',
        '            ).contiguous()',
        '            try:',
        '                import os as _tdo',
        '                if _tdo.environ.get("LUMO_TREE_DRAFT_DEBUG") == "1":',
        '                    import json as _tdj, time as _tdt',
        '                    global _LUMO_TREE_ACCEPT_DBG_FH, _LUMO_TREE_ACCEPT_DBG_N',
        '                    try:',
        '                        _LUMO_TREE_ACCEPT_DBG_N',
        '                    except NameError:',
        '                        _LUMO_TREE_ACCEPT_DBG_N = 0',
        '                    if _LUMO_TREE_ACCEPT_DBG_N < 64:',
        '                        try:',
        '                            _LUMO_TREE_ACCEPT_DBG_FH',
        '                        except NameError:',
        '                            _LUMO_TREE_ACCEPT_DBG_FH = open("/logs/tree_accept_debug.jsonl", "a", buffering=1)',
        '                        _LUMO_TREE_ACCEPT_DBG_FH.write(_tdj.dumps({',
        '                            "ts": round(_tdt.time(), 4),',
        '                            "num_draft_tokens": metadata.num_draft_tokens,',
        '                            "parents": lumo_tree_parent_indices.detach().cpu().tolist(),',
        '                            "draft": metadata.draft_token_ids.detach().cpu().tolist(),',
        '                            "parent_target": lumo_parent_token_ids.detach().cpu().tolist(),',
        '                            "self_target": lumo_self_token_ids.detach().cpu().tolist(),',
        '                            "target_logits_indices": metadata.target_logits_indices.detach().cpu().tolist(),',
        '                            "self_logits_indices": metadata.tree_self_logits_indices.detach().cpu().tolist(),',
        '                        }) + chr(10))',
        '                        _LUMO_TREE_ACCEPT_DBG_N += 1',
        '            except Exception:',
        '                pass',
        '',
        '        output_token_ids = rejection_sample(',
    ])
    if prep_anchor not in text:
        raise RuntimeError('tree token prep anchor not found')
    text = text.replace(prep_anchor, prep_inject, 1)

    call_anchor = nl.join([
        '            bonus_token_ids,',
        '            sampling_metadata,',
        '        )',
    ])
    call_inject = nl.join([
        '            bonus_token_ids,',
        '            sampling_metadata,',
        '            tree_parent_indices=lumo_tree_parent_indices,',
        '            tree_token_ids=lumo_tree_token_ids,',
        '        )',
    ])
    if call_anchor not in text:
        raise RuntimeError('rejection_sample call anchor not found')
    text = text.replace(call_anchor, call_inject, 1)

    sig_anchor = nl.join([
        '    bonus_token_ids: torch.Tensor,',
        '    sampling_metadata: SamplingMetadata,',
        ') -> torch.Tensor:',
    ])
    sig_inject = nl.join([
        '    bonus_token_ids: torch.Tensor,',
        '    sampling_metadata: SamplingMetadata,',
        '    tree_parent_indices: torch.Tensor | None = None,',
        '    tree_token_ids: torch.Tensor | None = None,',
        ') -> torch.Tensor:',
    ])
    if sig_anchor not in text:
        raise RuntimeError('rejection_sample signature anchor not found')
    text = text.replace(sig_anchor, sig_inject, 1)

    branch_anchor = nl.join([
        '    if sampling_metadata.all_greedy:',
        '        is_greedy = None',
    ])
    branch_inject = nl.join([
        '    if tree_parent_indices is not None and tree_token_ids is not None:',
        '        assert tree_parent_indices.is_contiguous()',
        '        assert tree_token_ids.is_contiguous()',
        '        if sampling_metadata.all_greedy:',
        '            lumo_tree_sample_kernel[(batch_size,)](',
        '                output_token_ids,',
        '                cu_num_draft_tokens,',
        '                draft_token_ids,',
        '                tree_parent_indices,',
        '                tree_token_ids[0],',
        '                tree_token_ids[1],',
        '                max_spec_len,',
        '            )',
        '        else:',
        '            target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)',
        '            uniform_probs = generate_uniform_probs(',
        '                num_tokens,',
        '                num_draft_tokens,',
        '                sampling_metadata.generators,',
        '                device,',
        '            )',
        '            lumo_tree_prob_sample_kernel[(batch_size,)](',
        '                output_token_ids,',
        '                cu_num_draft_tokens,',
        '                draft_token_ids,',
        '                tree_parent_indices,',
        '                tree_token_ids[0],',
        '                tree_token_ids[1],',
        '                target_probs,',
        '                uniform_probs,',
        '                max_spec_len,',
        '                vocab_size,',
        '            )',
        '        return output_token_ids',
        '',
        '    if sampling_metadata.all_greedy:',
        '        is_greedy = None',
    ])
    if branch_anchor not in text:
        raise RuntimeError('tree rejection branch anchor not found')
    text = text.replace(branch_anchor, branch_inject, 1)

    kernel_anchor = nl + nl + '# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.' + nl + '@triton.jit(do_not_specialize=["max_spec_len"])' + nl + 'def rejection_greedy_sample_kernel('
    kernel = nl.join([
        '',
        '',
        '# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.',
        '@triton.jit(do_not_specialize=["max_spec_len"])',
        'def lumo_tree_sample_kernel(',
        '    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]',
        '    cu_num_draft_tokens_ptr,  # [batch_size]',
        '    draft_token_ids_ptr,  # [num_tokens]',
        '    tree_parent_indices_ptr,  # [num_tokens], parent node or -1 for root',
        '    parent_token_ids_ptr,  # [num_tokens], target sample at each node parent',
        '    self_token_ids_ptr,  # [num_tokens], target sample at each node',
        '    max_spec_len,',
        '):',
        '    req_idx = tl.program_id(0)',
        '    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)',
        '    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)',
        '    num_draft_tokens = end_idx - start_idx',
        '',
        '    current_parent = -1',
        '    out_pos = 0',
        '    done = False',
        '    for _step in range(max_spec_len + 1):',
        '        if not done:',
        '            first_child = -1',
        '            matched_child = -1',
        '            target_token_id = -1',
        '            for pos in range(num_draft_tokens):',
        '                parent = tl.load(tree_parent_indices_ptr + start_idx + pos)',
        '                if parent == current_parent:',
        '                    if first_child == -1:',
        '                        first_child = pos',
        '                        target_token_id = tl.load(parent_token_ids_ptr + start_idx + pos)',
        '                    draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)',
        '                    if (matched_child == -1) and (draft_token_id == target_token_id):',
        '                        matched_child = pos',
        '',
        '            if first_child == -1:',
        '                if current_parent >= 0:',
        '                    token_id = tl.load(self_token_ids_ptr + start_idx + current_parent)',
        '                    tl.store(',
        '                        output_token_ids_ptr + req_idx * (max_spec_len + 1) + out_pos,',
        '                        token_id,',
        '                    )',
        '                done = True',
        '            else:',
        '                tl.store(',
        '                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + out_pos,',
        '                    target_token_id,',
        '                )',
        '                out_pos += 1',
        '                if matched_child >= 0:',
        '                    current_parent = matched_child',
        '                else:',
        '                    done = True',
    ])
    prob_kernel = nl.join([
        '',
        '',
        '# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.',
        '@triton.jit(do_not_specialize=["max_spec_len"])',
        'def lumo_tree_prob_sample_kernel(',
        '    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]',
        '    cu_num_draft_tokens_ptr,  # [batch_size]',
        '    draft_token_ids_ptr,  # [num_tokens]',
        '    tree_parent_indices_ptr,  # [num_tokens], parent node or -1 for root',
        '    parent_token_ids_ptr,  # [num_tokens], target sample at each node parent',
        '    self_token_ids_ptr,  # [num_tokens], target sample at each node',
        '    target_probs_ptr,  # [num_tokens, vocab_size]',
        '    uniform_probs_ptr,  # [num_tokens]',
        '    max_spec_len,',
        '    vocab_size,',
        '):',
        '    req_idx = tl.program_id(0)',
        '    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)',
        '    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)',
        '    num_draft_tokens = end_idx - start_idx',
        '',
        '    current_parent = -1',
        '    out_pos = 0',
        '    done = False',
        '    for _step in range(max_spec_len + 1):',
        '        if not done:',
        '            first_child = -1',
        '            accepted_child = -1',
        '            accepted_token_id = -1',
        '            fallback_token_id = -1',
        '            for pos in range(num_draft_tokens):',
        '                parent = tl.load(tree_parent_indices_ptr + start_idx + pos)',
        '                if parent == current_parent:',
        '                    if first_child == -1:',
        '                        first_child = pos',
        '                        fallback_token_id = tl.load(parent_token_ids_ptr + start_idx + pos)',
        '                    draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)',
        '                    target_prob = tl.load(',
        '                        target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id',
        '                    )',
        '                    uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)',
        '                    if (accepted_child == -1) and (target_prob >= uniform_prob):',
        '                        accepted_child = pos',
        '                        accepted_token_id = draft_token_id',
        '',
        '            if first_child == -1:',
        '                if current_parent >= 0:',
        '                    token_id = tl.load(self_token_ids_ptr + start_idx + current_parent)',
        '                    tl.store(',
        '                        output_token_ids_ptr + req_idx * (max_spec_len + 1) + out_pos,',
        '                        token_id,',
        '                    )',
        '                done = True',
        '            elif accepted_child >= 0:',
        '                tl.store(',
        '                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + out_pos,',
        '                    accepted_token_id,',
        '                )',
        '                out_pos += 1',
        '                current_parent = accepted_child',
        '            else:',
        '                tl.store(',
        '                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + out_pos,',
        '                    fallback_token_id,',
        '                )',
        '                done = True',
    ])
    if kernel_anchor not in text:
        raise RuntimeError('tree sample kernel anchor not found')
    text = text.replace(kernel_anchor, kernel + prob_kernel + kernel_anchor, 1)

    rs.write_text(text)
    import py_compile
    py_compile.compile(str(rs), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree-aware rejection sampler patch')

gm = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = gm.read_text()
sentinel = '# LUMO_TREE_METADATA'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree metadata/position patch already present')
else:
    meta_anchor = nl.join([
        '        # TODO: Optimize the CPU -> GPU copy.',
        '        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(',
    ])
    meta_inject = nl.join([
        '        ' + sentinel + ': tree parent map + parent-logit remap.',
        '        lumo_tree_parent_indices = None',
        '        lumo_tree_self_logits_indices = None',
        '        lumo_draft_token_indices = None',
        '        try:',
        '            _lspec = getattr(self.vllm_config, "speculative_config", None)',
        '            _ltree_src = getattr(_lspec, "speculative_token_tree", None) if _lspec is not None else None',
        '            if _ltree_src:',
        '                _choices = __import__("ast").literal_eval(_ltree_src)',
        '                _max_depth = max(len(_t) for _t in _choices)',
        '                if len(_choices) > _max_depth:',
        '                    _path_to_idx = {tuple(_p): _i for _i, _p in enumerate(_choices)}',
        '                    _parents_template = np.array([',
        '                        _path_to_idx.get(tuple(_p[:-1]), -1) for _p in _choices',
        '                    ], dtype=np.int32)',
        '                    _tree_len = int(len(_choices))',
        '                    _parents = []',
        '                    _target = []',
        '                    _self = []',
        '                    _draft = []',
        '                    _sampled_start = 0',
        '                    _ok = True',
        '                    for _n in num_draft_tokens.tolist():',
        '                        _n = int(_n)',
        '                        if _n == 0:',
        '                            _sampled_start += 1',
        '                            continue',
        '                        if _n != _tree_len:',
        '                            _ok = False',
        '                            break',
        '                        for _node_idx, _parent in enumerate(_parents_template.tolist()):',
        '                            _parent_local = 0 if _parent < 0 else int(_parent) + 1',
        '                            _parents.append(int(_parent))',
        '                            _target.append(_sampled_start + _parent_local)',
        '                            _self.append(_sampled_start + _node_idx + 1)',
        '                            _draft.append(_sampled_start + _node_idx + 1)',
        '                        _sampled_start += _n + 1',
        '                    if _ok and len(_target) == int(cu_num_draft_tokens[-1]):',
        '                        target_logits_indices = np.array(_target, dtype=np.int32)',
        '                        lumo_tree_parent_indices = torch.from_numpy(',
        '                            np.array(_parents, dtype=np.int32)',
        '                        ).to(self.device, non_blocking=True)',
        '                        lumo_tree_self_logits_indices = torch.from_numpy(',
        '                            np.array(_self, dtype=np.int32)',
        '                        ).to(self.device, non_blocking=True)',
        '                        lumo_draft_token_indices = torch.from_numpy(',
        '                            np.array(_draft, dtype=np.int32)',
        '                        ).to(self.device, non_blocking=True)',
        '        except Exception:',
        '            lumo_tree_parent_indices = None',
        '            lumo_tree_self_logits_indices = None',
        '            lumo_draft_token_indices = None',
        '',
        '        # TODO: Optimize the CPU -> GPU copy.',
        '        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(',
    ])
    if meta_anchor not in text:
        raise RuntimeError('metadata CPU-copy anchor not found')
    text = text.replace(meta_anchor, meta_inject, 1)

    draft_anchor = nl.join([
        '        # Compute the draft token ids.',
        '        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]',
        '        draft_token_ids = self.input_ids.gpu[logits_indices]',
        '        draft_token_ids = draft_token_ids[target_logits_indices + 1]',
        '',
        '        return SpecDecodeMetadata(',
    ])
    draft_inject = nl.join([
        '        # Compute the draft token ids.',
        '        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]',
        '        draft_token_ids = self.input_ids.gpu[logits_indices]',
        '        if lumo_draft_token_indices is not None:',
        '            draft_token_ids = draft_token_ids[lumo_draft_token_indices]',
        '        else:',
        '            draft_token_ids = draft_token_ids[target_logits_indices + 1]',
        '',
        '        _lumo_meta = SpecDecodeMetadata(',
    ])
    if draft_anchor not in text:
        raise RuntimeError('draft token gather anchor not found')
    text = text.replace(draft_anchor, draft_inject, 1)

    return_anchor = nl.join([
        '            logits_indices=logits_indices,',
        '        )',
        '',
        '    def _prepare_kv_sharing_fast_prefill(',
    ])
    return_inject = nl.join([
        '            logits_indices=logits_indices,',
        '        )',
        '        if lumo_tree_parent_indices is not None:',
        '            _lumo_meta.tree_parent_indices = lumo_tree_parent_indices',
        '            _lumo_meta.tree_self_logits_indices = lumo_tree_self_logits_indices',
        '        return _lumo_meta',
        '',
        '    def _prepare_kv_sharing_fast_prefill(',
    ])
    if return_anchor not in text:
        raise RuntimeError('metadata return anchor not found')
    text = text.replace(return_anchor, return_inject, 1)

    pos_anchor = nl.join([
        '        self.positions[:total_num_scheduled_tokens] = (',
        '            self.num_computed_tokens[req_indices_gpu].to(torch.int64)',
        '            + self.query_pos.gpu[:total_num_scheduled_tokens]',
        '        )',
        '        self.seq_lens[:num_reqs] = (',
    ])
    pos_inject = nl.join([
        '        self.positions[:total_num_scheduled_tokens] = (',
        '            self.num_computed_tokens[req_indices_gpu].to(torch.int64)',
        '            + self.query_pos.gpu[:total_num_scheduled_tokens]',
        '        )',
        '        # LUMO_TREE_POS: tree verify tokens use depth, not flat order.',
        '        try:',
        '            _ltree_offsets = getattr(self, "_lumo_tree_depth_offsets", None)',
        '            if _ltree_offsets is None:',
        '                _lspec = getattr(self.vllm_config, "speculative_config", None)',
        '                _ltree_src = getattr(_lspec, "speculative_token_tree", None) if _lspec is not None else None',
        '                _ltree_offsets = False',
        '                if _ltree_src:',
        '                    _choices = __import__("ast").literal_eval(_ltree_src)',
        '                    _max_depth = max(len(_t) for _t in _choices)',
        '                    if len(_choices) > _max_depth:',
        '                        _ltree_offsets = [0] + [len(_t) for _t in _choices]',
        '                self._lumo_tree_depth_offsets = _ltree_offsets',
        '            if _ltree_offsets and not self.uses_mrope:',
        '                _off_t = torch.tensor(_ltree_offsets, device=self.positions.device, dtype=torch.int64)',
        '                _ptr = 0',
        '                for _n_sched in num_scheduled_tokens[:num_reqs]:',
        '                    _n_sched = int(_n_sched)',
        '                    if _n_sched == len(_ltree_offsets):',
        '                        _base = self.positions[_ptr].clone()',
        '                        self.positions[_ptr:_ptr + _n_sched] = _base + _off_t',
        '                    _ptr += _n_sched',
        '        except Exception:',
        '            pass',
        '        self.seq_lens[:num_reqs] = (',
    ])
    if pos_anchor not in text:
        raise RuntimeError('non-mrope tree position anchor not found')
    text = text.replace(pos_anchor, pos_inject, 1)

    mrope_anchor = nl.join([
        '                MRotaryEmbedding.get_next_input_positions_tensor(',
        '                    out=self.mrope_positions.np,',
        '                    out_offset=dst_start,',
        '                    mrope_position_delta=req.mrope_position_delta,',
        '                    context_len=num_computed_tokens + prompt_part_len,',
        '                    num_new_tokens=completion_part_len,',
        '                )',
        '',
        '                mrope_pos_ptr += completion_part_len',
    ])
    mrope_inject = nl.join([
        '                MRotaryEmbedding.get_next_input_positions_tensor(',
        '                    out=self.mrope_positions.np,',
        '                    out_offset=dst_start,',
        '                    mrope_position_delta=req.mrope_position_delta,',
        '                    context_len=num_computed_tokens + prompt_part_len,',
        '                    num_new_tokens=completion_part_len,',
        '                )',
        '                # LUMO_TREE_POS_MROPE: siblings share depth-based RoPE positions.',
        '                try:',
        '                    _ltree_offsets = getattr(self, "_lumo_tree_depth_offsets_np", None)',
        '                    if _ltree_offsets is None:',
        '                        _lspec = getattr(self.vllm_config, "speculative_config", None)',
        '                        _ltree_src = getattr(_lspec, "speculative_token_tree", None) if _lspec is not None else None',
        '                        _ltree_offsets = False',
        '                        if _ltree_src:',
        '                            _choices = __import__("ast").literal_eval(_ltree_src)',
        '                            _max_depth = max(len(_t) for _t in _choices)',
        '                            if len(_choices) > _max_depth:',
        '                                _ltree_offsets = np.array([0] + [len(_t) for _t in _choices], dtype=self.mrope_positions.np.dtype)',
        '                        self._lumo_tree_depth_offsets_np = _ltree_offsets',
        '                    if _ltree_offsets is not False and completion_part_len == len(_ltree_offsets):',
        '                        _base = self.mrope_positions.np[:, dst_start:dst_start + 1].copy()',
        '                        self.mrope_positions.np[:, dst_start:dst_end] = _base + _ltree_offsets.reshape(1, -1)',
        '                except Exception:',
        '                    pass',
        '',
        '                mrope_pos_ptr += completion_part_len',
    ])
    if mrope_anchor not in text:
        raise RuntimeError('mrope tree position anchor not found')
    text = text.replace(mrope_anchor, mrope_inject, 1)

    gm.write_text(text)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree metadata and depth-position patches')
LUMOTREEREJECT
'''

_FB_BLOCK = r'''
python3 - <<'LUMOFBPATHS'
from pathlib import Path
nl = chr(10)

eg = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py')
text = eg.read_text()
sentinel = '# LUMO_FB_PATHS_EAGLE'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b eagle path proposer already present')
else:
    patch = r"""

# LUMO_FB_PATHS_EAGLE: bounded K=2 depth=3 path proposer for Qwen MTP.
# Appended as a source edit before import.  It deliberately avoids propose_tree:
# root top-2 is taken from the first MTP logits and each root is extended by
# reusing the ordinary linear proposer path, one path at a time.
import os as _lumo_fb_os
import json as _lumo_fb_json
import torch as _lumo_fb_torch
import time as _lumo_fb_time
from dataclasses import replace as _lumo_fb_replace
from vllm.forward_context import set_forward_context as _lumo_fb_forward_context
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata as _LumoFBTreeMetadata
from vllm.v1.spec_decode.eagle import eagle_step_update_slot_mapping_and_metadata as _lumo_fb_step_update

_lumo_fb_orig_propose = EagleProposer.propose

def _lumo_fb_read_control(max_depth):
    depth = int(_lumo_fb_os.environ.get("LUMO_FB_DEPTH", str(max_depth)))
    k = int(_lumo_fb_os.environ.get("LUMO_FB_K", "2"))
    info = {"fb_control_source": "env"}
    path = _lumo_fb_os.environ.get("LUMO_FB_CONTROL_FILE", "/logs/fb_control.json")
    if path and _lumo_fb_os.path.exists(path):
        st0 = _lumo_fb_os.stat(path)
        with open(path) as fh:
            payload = _lumo_fb_json.load(fh)
        st1 = _lumo_fb_os.stat(path)
        if (st0.st_mtime_ns != st1.st_mtime_ns
                or st0.st_size != st1.st_size
                or getattr(st0, "st_ino", None) != getattr(st1, "st_ino", None)):
            raise RuntimeError(f"LUMO_FB_CONTROL_FILE changed during read: {path}")
        if payload.get("depth") is not None:
            depth = int(payload["depth"])
        if payload.get("k") is not None:
            k = int(payload["k"])
        info = {
            "fb_control_source": "file",
            "fb_control_file": path,
            "fb_control_mtime_ns": int(st1.st_mtime_ns),
        }
    if depth < 1 or depth > int(max_depth):
        raise RuntimeError(f"LUMO_FB active depth {depth} outside launch_n_max {max_depth}")
    if k < 1 or k > 2:
        raise RuntimeError(f"LUMO_FB active K {k} unsupported in this build")
    info["launch_n_max"] = int(max_depth)
    info["active_depth"] = int(depth)
    info["active_k"] = int(k)
    return int(depth), int(k), info

def _lumo_fb_sample_logits(self, hidden_states):
    if self.use_local_argmax_reduction:
        return None
    return self.model.compute_logits(hidden_states)

def _lumo_fb_policy_from_logits(logits, max_k=None):
    requested_k = int(max_k if max_k is not None else _lumo_fb_os.environ.get("LUMO_FB_K", "2"))
    if _lumo_fb_os.environ.get("LUMO_FB_ADAPTIVE") != "1":
        return requested_k, {"fb_policy_k": requested_k, "fb_policy_reason": "fixed_k"}
    try:
        vals = _lumo_fb_torch.topk(logits.float(), min(8, logits.shape[-1]), dim=-1).values.view(-1)
        probs = _lumo_fb_torch.softmax(vals, dim=-1)
        p1 = float(probs[0].item())
        p2 = float(probs[1].item()) if probs.numel() > 1 else 0.0
        ratio = p2 / max(p1, 1e-9)
        p1_max = float(_lumo_fb_os.environ.get("LUMO_FB_ADAPTIVE_P1_MAX", "0.45"))
        ratio_min = float(_lumo_fb_os.environ.get("LUMO_FB_ADAPTIVE_RATIO_MIN", "0.50"))
        k = 2 if (requested_k >= 2 and p1 < p1_max and ratio > ratio_min) else 1
        return k, {
            "fb_policy_k": k,
            "fb_root_p1": round(p1, 6),
            "fb_root_p2": round(p2, 6),
            "fb_root_ratio": round(ratio, 6),
            "fb_policy_reason": "uncertain_root" if k == 2 else "confident_root",
        }
    except Exception:
        return 1, {"fb_policy_k": 1, "fb_policy_reason": "policy_error"}

def _lumo_fb_extend_one(self, root_token, base_positions, base_hidden_states,
                        base_common_attn_metadata, batch_size, per_layer_attn_metadata,
                        num_rejected_tokens_gpu, draft_len=None):
    if draft_len is None:
        draft_len = self.num_speculative_tokens
    draft_token_ids_list = [root_token]
    positions = base_positions.clone()
    hidden_states = base_hidden_states.clone()
    cad = _lumo_fb_replace(
        base_common_attn_metadata,
        seq_lens=base_common_attn_metadata.seq_lens.clone(),
        _seq_lens_cpu=(None if base_common_attn_metadata._seq_lens_cpu is None
                       else base_common_attn_metadata._seq_lens_cpu.clone()),
        _num_computed_tokens_cpu=(None if base_common_attn_metadata._num_computed_tokens_cpu is None
                                  else base_common_attn_metadata._num_computed_tokens_cpu.clone()),
    )
    cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
        self._determine_batch_execution_and_padding(batch_size)
    )
    cad.num_actual_tokens = batch_size
    cad.max_query_len = 1
    cad.query_start_loc = self.arange[: batch_size + 1]
    cad.query_start_loc_cpu = _lumo_fb_torch.from_numpy(
        self.token_arange_np[: batch_size + 1]
    ).clone()
    if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
        cad.seq_lens -= num_rejected_tokens_gpu
        cad._seq_lens_cpu = None
        cad._num_computed_tokens_cpu = None
    block_size = self.block_size
    assert block_size > 0
    for token_index in range(draft_len - 1):
        input_ids = draft_token_ids_list[-1].int()
        positions_1d = positions[0] if self.uses_mrope else positions
        if self.uses_mrope:
            out_pos = self.mrope_positions[0, :batch_size]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            out_pos = self.xdrope_positions[0, :batch_size]
        else:
            out_pos = self.positions[:batch_size]
        _lumo_fb_step_update(
            positions_1d=positions_1d,
            block_table_tensor=cad.block_table_tensor,
            seq_lens=cad.seq_lens,
            block_size=block_size,
            max_model_len=self.max_model_len,
            out_clamped_positions=out_pos,
            out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
            input_batch_size=input_batch_size,
        )
        cad.slot_mapping = self._slot_mapping_buffer[:batch_size]
        if self.uses_mrope:
            self.mrope_positions[1:, :batch_size] = self.mrope_positions[0, :batch_size]
            positions = self.mrope_positions[:, :batch_size]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[0, :batch_size]
            positions = self.xdrope_positions[0, :batch_size]
        else:
            positions = self.positions[:batch_size]
        cad.max_seq_len = min(cad.max_seq_len + 1, self.max_model_len)
        if cad._seq_lens_cpu is not None:
            cad._seq_lens_cpu += 1
        if cad._num_computed_tokens_cpu is not None:
            cad._num_computed_tokens_cpu += 1
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cad, draft_index=token_index + 1,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        self.input_ids[:batch_size] = input_ids
        self.hidden_states[:batch_size] = hidden_states
        if self.supports_mm_inputs:
            self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)
            input_ids_arg = None
            inputs_embeds = self.inputs_embeds[:input_batch_size]
        else:
            input_ids_arg = self.input_ids[:input_batch_size]
            inputs_embeds = None
        model_kwargs = {
            "input_ids": input_ids_arg,
            "positions": self._get_positions(input_batch_size),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]
        with _lumo_fb_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=input_batch_size,
            num_tokens_across_dp=batch_size_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(input_batch_size),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        hidden_states = hidden_states[:batch_size]
        draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
        draft_token_ids_list.append(draft_token_ids)
    return _lumo_fb_torch.stack(draft_token_ids_list, dim=1)

def _lumo_fb_repeat_cpu_shadow(value, k):
    if value is None:
        return None
    return value[:1].repeat(k).clone()

def _lumo_fb_extend_paths_batched(self, root_tokens, base_positions, base_hidden_states,
                                  base_common_attn_metadata, per_layer_attn_metadata,
                                  num_rejected_tokens_gpu, draft_len=None):
    # LUMO_FB_BATCHED_PROPOSER: row-scaled K path growth.  The K path roots
    # share the prompt prefix but advance as K draft rows inside one MTP forward
    # per depth step instead of K serial single-row forwards.
    if draft_len is None:
        draft_len = self.num_speculative_tokens
    k = int(root_tokens.numel())
    root_tokens = root_tokens.reshape(k).to(device=base_hidden_states.device)
    draft_token_ids_list = [root_tokens]
    if self.uses_mrope:
        positions = base_positions[:, :1].repeat(1, k).clone()
    else:
        positions = base_positions[:1].repeat(k).clone()
    hidden_states = base_hidden_states[:1].repeat(k, 1).clone()
    cad = _lumo_fb_replace(
        base_common_attn_metadata,
        query_start_loc=self.arange[: k + 1],
        query_start_loc_cpu=_lumo_fb_torch.from_numpy(
            self.token_arange_np[: k + 1]
        ).clone(),
        seq_lens=base_common_attn_metadata.seq_lens[:1].repeat(k).clone(),
        _seq_lens_cpu=_lumo_fb_repeat_cpu_shadow(
            base_common_attn_metadata._seq_lens_cpu, k),
        _num_computed_tokens_cpu=_lumo_fb_repeat_cpu_shadow(
            base_common_attn_metadata._num_computed_tokens_cpu, k),
        num_reqs=k,
        num_actual_tokens=k,
        max_query_len=1,
        block_table_tensor=base_common_attn_metadata.block_table_tensor[:1].repeat(
            k, 1).clone(),
        slot_mapping=base_common_attn_metadata.slot_mapping[:1].repeat(k).clone(),
    )
    cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
        self._determine_batch_execution_and_padding(k)
    )
    if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
        cad.seq_lens -= num_rejected_tokens_gpu[:1].repeat(k)
        cad._seq_lens_cpu = None
        cad._num_computed_tokens_cpu = None
    block_size = self.block_size
    assert block_size > 0
    for token_index in range(draft_len - 1):
        input_ids = draft_token_ids_list[-1].int()
        positions_1d = positions[0] if self.uses_mrope else positions
        if self.uses_mrope:
            out_pos = self.mrope_positions[0, :k]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            out_pos = self.xdrope_positions[0, :k]
        else:
            out_pos = self.positions[:k]
        _lumo_fb_step_update(
            positions_1d=positions_1d,
            block_table_tensor=cad.block_table_tensor,
            seq_lens=cad.seq_lens,
            block_size=block_size,
            max_model_len=self.max_model_len,
            out_clamped_positions=out_pos,
            out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
            input_batch_size=input_batch_size,
        )
        if k > 1:
            slot_stride = int(_lumo_fb_os.environ.get(
                "LUMO_FB_BATCHED_SLOT_STRIDE", str(self.num_speculative_tokens)))
            slot_offsets = (
                _lumo_fb_torch.arange(k, device=self._slot_mapping_buffer.device,
                                      dtype=self._slot_mapping_buffer.dtype)
                * slot_stride
            )
            self._slot_mapping_buffer[:k] += slot_offsets
        cad.slot_mapping = self._slot_mapping_buffer[:k]
        if self.uses_mrope:
            self.mrope_positions[1:, :k] = self.mrope_positions[0, :k]
            positions = self.mrope_positions[:, :k]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[1:, :k] = self.xdrope_positions[0, :k]
            positions = self.xdrope_positions[0, :k]
        else:
            positions = self.positions[:k]
        cad.max_seq_len = min(cad.max_seq_len + 1, self.max_model_len)
        if cad._seq_lens_cpu is not None:
            cad._seq_lens_cpu += 1
        if cad._num_computed_tokens_cpu is not None:
            cad._num_computed_tokens_cpu += 1
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cad, draft_index=token_index + 1,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        self.input_ids[:k] = input_ids
        self.hidden_states[:k] = hidden_states
        if self.supports_mm_inputs:
            self.inputs_embeds[:k] = self.model.embed_input_ids(input_ids)
            input_ids_arg = None
            inputs_embeds = self.inputs_embeds[:input_batch_size]
        else:
            input_ids_arg = self.input_ids[:input_batch_size]
            inputs_embeds = None
        model_kwargs = {
            "input_ids": input_ids_arg,
            "positions": self._get_positions(input_batch_size),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]
        with _lumo_fb_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=input_batch_size,
            num_tokens_across_dp=batch_size_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(input_batch_size),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        hidden_states = hidden_states[:k]
        draft_token_ids = self._greedy_sample(last_hidden_states[:k])
        draft_token_ids_list.append(draft_token_ids)
    return _lumo_fb_torch.stack(draft_token_ids_list, dim=1)

def _lumo_fb_propose(self, target_token_ids, target_positions, target_hidden_states,
                     next_token_ids, token_indices_to_sample, common_attn_metadata,
                     sampling_metadata, mm_embed_inputs=None,
                     num_rejected_tokens_gpu=None, slot_mappings=None):
    global _LUMO_FB_DBG_FH
    _lumo_fb_prop_t0 = _lumo_fb_time.perf_counter_ns()
    if _lumo_fb_os.environ.get("LUMO_FB_PATHS") != "1":
        return _lumo_fb_orig_propose(
            self, target_token_ids, target_positions, target_hidden_states,
            next_token_ids, token_indices_to_sample, common_attn_metadata,
            sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)
    active_depth, requested_k, control_info = _lumo_fb_read_control(self.num_speculative_tokens)
    if common_attn_metadata.batch_size() != 1:
        return _lumo_fb_orig_propose(
            self, target_token_ids, target_positions, target_hidden_states,
            next_token_ids, token_indices_to_sample, common_attn_metadata,
            sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)
    batch_size = common_attn_metadata.batch_size()
    if self.method == "eagle3":
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
    num_tokens, token_indices_to_sample, common_attn_metadata = self.set_inputs_first_pass(
        target_token_ids=target_token_ids,
        next_token_ids=next_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        token_indices_to_sample=token_indices_to_sample,
        cad=common_attn_metadata,
        num_rejected_tokens_gpu=num_rejected_tokens_gpu,
    )
    per_layer_attn_metadata = {}
    attn_metadata = None
    for attn_group in self.draft_attn_groups:
        attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0,
        )
        for layer_name in attn_group.layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
    if isinstance(attn_metadata, _LumoFBTreeMetadata):
        return _lumo_fb_orig_propose(
            self, target_token_ids, target_positions, target_hidden_states,
            next_token_ids, token_indices_to_sample, common_attn_metadata,
            sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)
    cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
        self._determine_batch_execution_and_padding(num_tokens)
    )
    if self.supports_mm_inputs:
        mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)
        self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
            self.input_ids[:num_tokens],
            multimodal_embeddings=mm_embeds,
            is_multimodal=is_mm_embed,
        )
        input_ids = None
        inputs_embeds = self.inputs_embeds[:num_input_tokens]
    else:
        input_ids = self.input_ids[:num_input_tokens]
        inputs_embeds = None
    model_kwargs = {
        "input_ids": input_ids,
        "positions": self._get_positions(num_input_tokens),
        "inputs_embeds": inputs_embeds,
    }
    if self.pass_hidden_states_to_model:
        model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]
    with _lumo_fb_forward_context(
        per_layer_attn_metadata,
        self.vllm_config,
        num_tokens=num_input_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        slot_mapping=self._get_slot_mapping(num_input_tokens, common_attn_metadata.slot_mapping),
    ):
        ret_hidden_states = self.model(**model_kwargs)
        if not self.model_returns_tuple():
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
    sample_hidden_states = last_hidden_states[token_indices_to_sample]
    logits = _lumo_fb_sample_logits(self, sample_hidden_states)
    if logits is None:
        return _lumo_fb_orig_propose(
            self, target_token_ids, target_positions, target_hidden_states,
            next_token_ids, token_indices_to_sample, common_attn_metadata,
            sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)
    policy_k, policy_info = _lumo_fb_policy_from_logits(logits, requested_k)
    policy_info = dict(policy_info)
    policy_info.update(control_info)
    if getattr(sampling_metadata, "all_greedy", False):
        # In greedy decode, F_b must reduce exactly to the linear E path:
        # the single target argmax root is deterministic, so sibling root
        # diversity has no distributional role and only risks row interference.
        policy_k = 1
        policy_info = dict(policy_info)
        policy_info["fb_policy_k"] = 1
        policy_info["fb_policy_reason"] = "greedy_reduces_to_linear"
    if _lumo_fb_os.environ.get("LUMO_FB_DUP_PATH1") == "1":
        policy_k = 2
        policy_info = dict(policy_info)
        policy_info["fb_policy_k"] = 2
        policy_info["fb_policy_reason"] = "duplicate_path_gate"
    root_candidates = _lumo_fb_torch.topk(
        logits, min(16, logits.shape[-1]), dim=-1
    ).indices.view(-1)
    raw_roots = root_candidates[:2].view(2, 1)
    real_roots = []
    for tok in root_candidates.tolist():
        tok = int(tok)
        if tok != 0 and tok not in real_roots:
            real_roots.append(tok)
        if len(real_roots) == 2:
            break
    if len(real_roots) < 2:
        real_roots = [int(t) for t in raw_roots.view(-1).tolist()]
    roots = _lumo_fb_torch.tensor(
        real_roots[:2], dtype=raw_roots.dtype, device=raw_roots.device
    ).view(2, 1)
    if self.uses_mrope:
        positions = self.mrope_positions[:, token_indices_to_sample]
    else:
        positions = self.positions[token_indices_to_sample]
    base_hidden_states = hidden_states[token_indices_to_sample]
    # LUMO_FB_DUP_PATH1_TEST: duplicate-path discriminator keeps path0 as the
    # raw MTP top-1 chain so it remains directly comparable to K=1.
    _lumo_fb_path0_root = raw_roots[0] if (
        policy_k == 1 or _lumo_fb_os.environ.get("LUMO_FB_DUP_PATH1") == "1"
    ) else roots[0]
    if policy_k == 1:
        path0 = _lumo_fb_extend_one(
            self, _lumo_fb_path0_root, positions, base_hidden_states,
            common_attn_metadata, batch_size, dict(per_layer_attn_metadata),
            num_rejected_tokens_gpu, draft_len=active_depth)
        out = path0[:, :active_depth]
        try:
            if _lumo_fb_os.environ.get("LUMO_FB_DEBUG") == "1":
                import json as _j, time as _t
                try:
                    _LUMO_FB_DBG_FH
                except NameError:
                    _LUMO_FB_DBG_FH = open("/logs/fb_debug.jsonl", "a", buffering=1)
                _LUMO_FB_DBG_FH.write(_j.dumps({
                    "ts": round(_t.time(), 4),
                    "k": 1,
                    "launch_n_max": int(self.num_speculative_tokens),
                    "active_depth": int(active_depth),
                    "active_k": int(requested_k),
                    "fb_proposer_us": int((_lumo_fb_time.perf_counter_ns() - _lumo_fb_prop_t0) // 1000),
                    "raw_roots": raw_roots.view(-1).tolist(),
                    "roots": [int(out[0, 0].item())],
                    "root_candidates": root_candidates[:8].tolist(),
                    "draft": out.tolist(),
                    "policy": policy_info,
                }) + chr(10))
        except Exception:
            pass
        return out
    if _lumo_fb_os.environ.get("LUMO_FB_BATCHED_PROPOSER") == "1":
        root_vec = _lumo_fb_torch.cat([
            _lumo_fb_path0_root.reshape(-1)[:1],
            (_lumo_fb_path0_root if _lumo_fb_os.environ.get("LUMO_FB_DUP_PATH1") == "1"
             else roots[1]).reshape(-1)[:1],
        ], dim=0)
        paths = _lumo_fb_extend_paths_batched(
            self, root_vec, positions, base_hidden_states, common_attn_metadata,
            dict(per_layer_attn_metadata), num_rejected_tokens_gpu, draft_len=active_depth)
        path0 = paths[0:1, :active_depth]
        path1 = paths[1:2, :active_depth]
    elif _lumo_fb_os.environ.get("LUMO_FB_DUP_PATH1") == "1":
        path0 = _lumo_fb_extend_one(
            self, _lumo_fb_path0_root, positions, base_hidden_states,
            common_attn_metadata, batch_size, dict(per_layer_attn_metadata),
            num_rejected_tokens_gpu, draft_len=active_depth)
        path1 = path0.clone()
    else:
        path0 = _lumo_fb_extend_one(
            self, _lumo_fb_path0_root, positions, base_hidden_states,
            common_attn_metadata, batch_size, dict(per_layer_attn_metadata),
            num_rejected_tokens_gpu, draft_len=active_depth)
        path1 = _lumo_fb_extend_one(
            self, roots[1], positions, base_hidden_states,
            common_attn_metadata, batch_size, dict(per_layer_attn_metadata),
            num_rejected_tokens_gpu, draft_len=active_depth)
    out = _lumo_fb_torch.cat([path0, path1], dim=1)
    try:
        if _lumo_fb_os.environ.get("LUMO_FB_DEBUG") == "1":
            import json as _j, time as _t
            try:
                _LUMO_FB_DBG_FH
            except NameError:
                _LUMO_FB_DBG_FH = open("/logs/fb_debug.jsonl", "a", buffering=1)
            _LUMO_FB_DBG_FH.write(_j.dumps({
                "ts": round(_t.time(), 4),
                "launch_n_max": int(self.num_speculative_tokens),
                "active_depth": int(active_depth),
                "active_k": int(requested_k),
                "policy_k": int(policy_k),
                "fb_proposer_us": int((_lumo_fb_time.perf_counter_ns() - _lumo_fb_prop_t0) // 1000),
                "raw_roots": raw_roots.view(-1).tolist(),
                "roots": [int(out[0, 0].item()), int(out[0, active_depth].item())],
                "root_candidates": root_candidates[:8].tolist(),
                "draft": out.tolist(),
                "policy": policy_info,
            }) + chr(10))
    except Exception:
        pass
    return out

EagleProposer.propose = _lumo_fb_propose
"""
    eg.write_text(text + patch)
    import py_compile
    py_compile.compile(str(eg), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b eagle path proposer patch')

text = eg.read_text()
sentinel = '# LUMO_FB_DUP_PATH1_TEST'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b duplicate-path test patch already present')
else:
    old_path0 = nl.join([
        '    path0 = _lumo_fb_extend_one(',
        '        self, (raw_roots[0] if int(_lumo_fb_os.environ.get("LUMO_FB_K", "2")) == 1 else roots[0]), positions, base_hidden_states,',
        '        common_attn_metadata, batch_size, dict(per_layer_attn_metadata),',
        '        num_rejected_tokens_gpu)',
    ])
    new_path0 = nl.join([
        '    # LUMO_FB_DUP_PATH1_TEST: in duplicate-path discriminator mode,',
        '    # path0 must be the raw MTP top-1 chain so it is comparable to K=1.',
        '    _lumo_fb_path0_root = (raw_roots[0] if (int(_lumo_fb_os.environ.get("LUMO_FB_K", "2")) == 1',
        '                           or _lumo_fb_os.environ.get("LUMO_FB_DUP_PATH1") == "1")',
        '                           else roots[0])',
        '    path0 = _lumo_fb_extend_one(',
        '        self, _lumo_fb_path0_root, positions, base_hidden_states,',
        '        common_attn_metadata, batch_size, dict(per_layer_attn_metadata),',
        '        num_rejected_tokens_gpu)',
    ])
    old_path1 = nl.join([
        '    path1 = _lumo_fb_extend_one(',
        '        self, roots[1], positions, base_hidden_states,',
        '        common_attn_metadata, batch_size, dict(per_layer_attn_metadata),',
        '        num_rejected_tokens_gpu)',
    ])
    new_path1 = nl.join([
        '    if _lumo_fb_os.environ.get("LUMO_FB_DUP_PATH1") == "1":',
        '        path1 = path0.clone()',
        '    else:',
        '        path1 = _lumo_fb_extend_one(',
        '            self, roots[1], positions, base_hidden_states,',
        '            common_attn_metadata, batch_size, dict(per_layer_attn_metadata),',
        '            num_rejected_tokens_gpu)',
    ])
    if old_path0 not in text or old_path1 not in text:
        raise RuntimeError('F_b duplicate-path test anchor not found')
    text = text.replace(old_path0, new_path0, 1).replace(old_path1, new_path1, 1)
    eg.write_text(text)
    import py_compile
    py_compile.compile(str(eg), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b duplicate-path test patch')

sch = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py')
text = sch.read_text()
sentinel = '# LUMO_FB_PATHS_SCHED'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b scheduler path clone patch already present')
else:
    patch = r"""

# LUMO_FB_PATHS_SCHED: internal K=2 path requests + longest-accepted collapse.
import os as _lumo_fb_os
import json as _lumo_fb_json
from vllm.v1.core.sched.output import NewRequestData as _LumoFBNewRequestData

_lumo_fb_orig_schedule = Scheduler.schedule
_lumo_fb_orig_update_draft = Scheduler.update_draft_token_ids
_lumo_fb_orig_update_output = Scheduler.update_from_output

def _lumo_fb_sched_read_control(default_depth):
    depth = int(_lumo_fb_os.environ.get("LUMO_FB_DEPTH", str(default_depth)))
    k = int(_lumo_fb_os.environ.get("LUMO_FB_K", "2"))
    path = _lumo_fb_os.environ.get("LUMO_FB_CONTROL_FILE", "/logs/fb_control.json")
    if path and _lumo_fb_os.path.exists(path):
        st0 = _lumo_fb_os.stat(path)
        with open(path) as fh:
            payload = _lumo_fb_json.load(fh)
        st1 = _lumo_fb_os.stat(path)
        if (st0.st_mtime_ns != st1.st_mtime_ns
                or st0.st_size != st1.st_size
                or getattr(st0, "st_ino", None) != getattr(st1, "st_ino", None)):
            raise RuntimeError(f"LUMO_FB_CONTROL_FILE changed during read: {path}")
        if payload.get("depth") is not None:
            depth = int(payload["depth"])
        if payload.get("k") is not None:
            k = int(payload["k"])
    if depth < 1:
        raise RuntimeError(f"LUMO_FB active depth {depth} invalid")
    if k < 1 or k > 2:
        raise RuntimeError(f"LUMO_FB active K {k} unsupported in this build")
    return int(depth), int(k)

def _lumo_fb_block_ids_from_blocks(blocks):
    return tuple([blk.block_id for blk in group] for group in blocks)

def _lumo_fb_clone_blocks(self, parent, clone_id):
    copies = []
    for group_idx, manager in enumerate(self.kv_cache_manager.coordinator.single_type_managers):
        parent_blocks = list(manager.req_to_blocks.get(parent.request_id, ()))
        block_size = manager.block_size
        if getattr(manager, "mamba_cache_mode", None) == "align":
            # Mamba/GDN align mode keeps the recurrent state in logical cache
            # blocks beyond the token prefix. F_b verifier clones must keep the
            # same logical layout but use independent physical blocks; otherwise
            # the two sibling verifier rows overwrite each other's recurrent
            # state during the batched target forward.
            clone_blocks = []
            for src in parent_blocks:
                if src == getattr(manager, "_null_block", None):
                    clone_blocks.append(src)
                    continue
                dst = manager.block_pool.get_new_blocks(1)[0]
                clone_blocks.append(dst)
                copies.append(("mamba", group_idx, src.block_id, dst.block_id))
            manager.req_to_blocks[clone_id] = clone_blocks
            manager.num_cached_block[clone_id] = min(
                manager.num_cached_block.get(parent.request_id, len(clone_blocks)),
                len(clone_blocks),
            )
            if hasattr(manager, "last_state_block_idx") and parent.request_id in manager.last_state_block_idx:
                manager.last_state_block_idx[clone_id] = manager.last_state_block_idx[parent.request_id]
            continue
        valid_tokens = max(0, int(parent.num_tokens) - 1)
        full_blocks = min(valid_tokens // block_size, len(parent_blocks))
        has_partial = (valid_tokens % block_size) != 0 and full_blocks < len(parent_blocks)
        clone_blocks = list(parent_blocks[:full_blocks])
        manager.block_pool.touch(clone_blocks)
        if has_partial:
            src = parent_blocks[full_blocks]
            dst = manager.block_pool.get_new_blocks(1)[0]
            clone_blocks.append(dst)
            copies.append((src.block_id, dst.block_id))
            try:
                if type(manager.kv_cache_spec).__name__ == "FullAttentionSpec":
                    manager.new_block_ids.append(dst.block_id)
            except Exception:
                pass
        manager.req_to_blocks[clone_id] = clone_blocks
        manager.num_cached_block[clone_id] = min(
            manager.num_cached_block.get(parent.request_id, full_blocks),
            len(clone_blocks),
        )
    return copies

def _lumo_fb_transfer_blocks(self, parent_id, winner_id, loser_id):
    for manager in self.kv_cache_manager.coordinator.single_type_managers:
        winner_allocated = False
        if hasattr(manager, "_allocated_block_reqs"):
            winner_allocated = winner_id in manager._allocated_block_reqs
        old_parent = manager.req_to_blocks.pop(parent_id, [])
        if old_parent:
            manager.block_pool.free_blocks(reversed(old_parent))
        winner_blocks = manager.req_to_blocks.pop(winner_id, [])
        manager.req_to_blocks[parent_id] = winner_blocks
        if winner_id in manager.num_cached_block:
            manager.num_cached_block[parent_id] = manager.num_cached_block.pop(winner_id)
        manager.num_cached_block.pop(loser_id, None)
        loser_blocks = manager.req_to_blocks.pop(loser_id, [])
        if loser_blocks:
            manager.block_pool.free_blocks(reversed(loser_blocks))
        if hasattr(manager, "_allocated_block_reqs"):
            manager._allocated_block_reqs.discard(parent_id)
            manager._allocated_block_reqs.discard(winner_id)
            manager._allocated_block_reqs.discard(loser_id)
            if winner_allocated:
                manager._allocated_block_reqs.add(parent_id)
            if hasattr(manager, "last_state_block_idx"):
                if winner_id in manager.last_state_block_idx:
                    manager.last_state_block_idx[parent_id] = manager.last_state_block_idx.pop(winner_id)
                else:
                    manager.last_state_block_idx.pop(parent_id, None)
                manager.last_state_block_idx.pop(loser_id, None)

def _lumo_fb_make_clone(self, parent, path_idx, path):
    clone_id = f"{parent.request_id}::lumo_fb::{path_idx}"
    clone = Request(
        request_id=clone_id,
        prompt_token_ids=list(parent.all_token_ids),
        sampling_params=parent.sampling_params,
        pooling_params=parent.pooling_params,
        client_index=parent.client_index,
        arrival_time=parent.arrival_time,
        mm_features=list(parent.mm_features),
        lora_request=parent.lora_request,
        cache_salt=parent.cache_salt,
        priority=parent.priority,
        trace_headers=parent.trace_headers,
        block_hasher=None,
        resumable=False,
    )
    clone.status = RequestStatus.RUNNING
    # Flat-chain verifier rows must start at the current sampled token.
    # The token is present in all_token_ids but its KV is not computed yet.
    clone.num_computed_tokens = max(0, parent.num_tokens - 1)
    clone.num_cached_tokens = parent.num_cached_tokens
    clone.spec_token_ids = list(path)
    clone._lumo_fb_parent_id = parent.request_id
    clone._lumo_fb_path_idx = path_idx
    clone._lumo_fb_new_clone = True
    self.requests[clone_id] = clone
    return clone

def _lumo_fb_expand_pending(self):
    if _lumo_fb_os.environ.get("LUMO_FB_PATHS") != "1":
        return []
    copies = []
    new_running = []
    for req in self.running:
        paths = getattr(req, "_lumo_fb_pending_paths", None)
        if not paths:
            new_running.append(req)
            continue
        req._lumo_fb_pending_paths = None
        req._lumo_fb_waiting_parent = True
        for idx, path in enumerate(paths):
            clone = _lumo_fb_make_clone(self, req, idx, path)
            copies.extend(_lumo_fb_clone_blocks(self, req, clone.request_id))
            new_running.append(clone)
    self.running = new_running
    return copies

def _lumo_fb_update_draft_token_ids(self, draft_token_ids):
    if _lumo_fb_os.environ.get("LUMO_FB_PATHS") != "1":
        return _lumo_fb_orig_update_draft(self, draft_token_ids)
    for req_id, spec_token_ids in zip(draft_token_ids.req_ids, draft_token_ids.draft_token_ids):
        request = self.requests.get(req_id)
        if request is None or request.is_finished():
            continue
        if request.is_prefill_chunk:
            request.spec_token_ids = []
            continue
        _fb_sched_default_k = int(_lumo_fb_os.environ.get("LUMO_FB_K", "2"))
        _fb_sched_default_depth = (len(spec_token_ids) // 2
                                   if _fb_sched_default_k >= 2 and len(spec_token_ids) % 2 == 0
                                   else len(spec_token_ids))
        active_depth, requested_k = _lumo_fb_sched_read_control(_fb_sched_default_depth)
        if requested_k >= 2 and len(spec_token_ids) == 2 * active_depth:
            request._lumo_fb_pending_paths = [
                list(spec_token_ids[:active_depth]),
                list(spec_token_ids[active_depth:2 * active_depth]),
            ]
            request.spec_token_ids = []
        else:
            if _lumo_fb_os.environ.get("LUMO_FB_ASSERT_WIDTH") == "1" and len(spec_token_ids) != active_depth:
                raise RuntimeError(
                    f"LUMO_FB clone scheduler width mismatch: draft_len={len(spec_token_ids)} active_depth={active_depth} active_k={requested_k}")
            request.spec_token_ids = list(spec_token_ids[:active_depth])
        try:
            if _lumo_fb_os.environ.get("LUMO_FB_DEBUG") != "1":
                continue
            import json as _fbj, time as _fbt
            global _LUMO_FB_RAW_DBG_FH
            try:
                _LUMO_FB_RAW_DBG_FH
            except NameError:
                _LUMO_FB_RAW_DBG_FH = open("/logs/fb_raw_draft_debug.jsonl", "a", buffering=1)
            _LUMO_FB_RAW_DBG_FH.write(_fbj.dumps({
                "ts": round(_fbt.time(), 4),
                "req_id": req_id,
                "raw": list(spec_token_ids),
                "pending_paths": getattr(request, "_lumo_fb_pending_paths", None),
                "is_clone": "::lumo_fb::" in req_id,
            }) + chr(10))
        except Exception:
            pass

def _lumo_fb_schedule(self):
    copies = _lumo_fb_expand_pending(self)
    out = _lumo_fb_orig_schedule(self)
    if copies:
        out.lumo_fb_block_copies = copies
    fb_new_ids = {
        req_id for req_id in out.num_scheduled_tokens
        if "::lumo_fb::" in req_id and getattr(self.requests.get(req_id), "_lumo_fb_new_clone", False)
    }
    if fb_new_ids:
        keep = []
        req_data = out.scheduled_cached_reqs
        new_req_ids = []
        new_new_block_ids = []
        new_num_computed = []
        new_num_output = []
        for i, rid in enumerate(req_data.req_ids):
            if rid in fb_new_ids:
                req = self.requests[rid]
                out.scheduled_new_reqs.append(
                    _LumoFBNewRequestData.from_request(
                        req, _lumo_fb_block_ids_from_blocks(self.kv_cache_manager.get_blocks(rid))
                    )
                )
            else:
                keep.append(i)
        if len(keep) != len(req_data.req_ids):
            req_data.req_ids = [req_data.req_ids[i] for i in keep]
            req_data.new_block_ids = [req_data.new_block_ids[i] for i in keep]
            req_data.num_computed_tokens = [req_data.num_computed_tokens[i] for i in keep]
            req_data.num_output_tokens = [req_data.num_output_tokens[i] for i in keep]
            if req_data.new_token_ids:
                req_data.new_token_ids = [req_data.new_token_ids[i] for i in keep]
            req_data.resumed_req_ids = {rid for rid in req_data.resumed_req_ids if rid not in fb_new_ids}
            req_data.all_token_ids = {rid: toks for rid, toks in req_data.all_token_ids.items() if rid not in fb_new_ids}
    collapses = getattr(self, "_lumo_fb_pending_collapses", None)
    if collapses:
        out.lumo_fb_collapses = collapses
        self._lumo_fb_pending_collapses = []
    return out

def _lumo_fb_update_from_output(self, scheduler_output, model_runner_output):
    clone_ids = [rid for rid in scheduler_output.num_scheduled_tokens if "::lumo_fb::" in rid]
    if not clone_ids:
        return _lumo_fb_orig_update_output(self, scheduler_output, model_runner_output)
    groups = {}
    for rid in clone_ids:
        req = self.requests.get(rid)
        if req is not None:
            groups.setdefault(req._lumo_fb_parent_id, []).append(rid)
    outputs = {}
    spec_decoding_stats = None
    for parent_id, ids in groups.items():
        if len(ids) != 2:
            continue
        parent = self.requests[parent_id]
        rows = []
        for rid in sorted(ids, key=lambda x: self.requests[x]._lumo_fb_path_idx):
            idx = model_runner_output.req_id_to_index[rid]
            gen = model_runner_output.sampled_token_ids[idx] if model_runner_output.sampled_token_ids else []
            rows.append((rid, gen, max(len(gen) - 1, 0)))
        try:
            if _lumo_fb_os.environ.get("LUMO_FB_DEBUG") != "1":
                raise RuntimeError("fb accept debug disabled")
            import json as _fbj, time as _fbt
            global _LUMO_FB_ACCEPT_DBG_FH
            try:
                _LUMO_FB_ACCEPT_DBG_FH
            except NameError:
                _LUMO_FB_ACCEPT_DBG_FH = open("/logs/fb_accept_debug.jsonl", "a", buffering=1)
            _LUMO_FB_ACCEPT_DBG_FH.write(_fbj.dumps({
                "ts": round(_fbt.time(), 4),
                "parent": parent_id,
                "paths": [
                    {
                        "rid": rid,
                        "path_idx": getattr(self.requests.get(rid), "_lumo_fb_path_idx", None),
                        "draft": scheduler_output.scheduled_spec_decode_tokens.get(rid),
                        "generated": gen,
                        "accepted": acc,
                    }
                    for rid, gen, acc in rows
                ],
            }) + chr(10))
        except Exception:
            pass
        winner_id, generated_token_ids, accepted = max(rows, key=lambda r: (r[2], -self.requests[r[0]]._lumo_fb_path_idx))
        loser_id = rows[1][0] if rows[0][0] == winner_id else rows[0][0]
        parent.num_computed_tokens = max(0, parent.num_tokens - 1) + len(generated_token_ids)
        spec_decoding_stats = self.make_spec_decoding_stats(
            spec_decoding_stats,
            num_draft_tokens=3,
            num_accepted_tokens=accepted,
            num_invalid_spec_tokens=None,
            request_id=parent_id,
        )
        new_token_ids, stopped = self._update_request_with_output(parent, list(generated_token_ids))
        finish_reason = None
        kv_transfer_params = None
        routed_experts = None
        if stopped:
            routed_experts = self._get_routed_experts(parent)
            finish_reason = parent.get_finished_reason()
            finished = self._handle_stopped_request(parent)
            if finished:
                kv_transfer_params = self._free_request(parent)
        else:
            if parent not in self.running:
                self.running.append(parent)
        _lumo_fb_transfer_blocks(self, parent_id, winner_id, loser_id)
        for rid in ids:
            req = self.requests.pop(rid, None)
            if req is not None and req in self.running:
                self.running.remove(req)
            self.finished_req_ids.add(rid)
        self._lumo_fb_pending_collapses = getattr(self, "_lumo_fb_pending_collapses", [])
        self._lumo_fb_pending_collapses.append((parent_id, winner_id, loser_id, parent.num_computed_tokens, len(generated_token_ids)))
        outputs.setdefault(parent.client_index, []).append(
            EngineCoreOutput(
                request_id=parent_id,
                new_token_ids=new_token_ids,
                finish_reason=finish_reason,
                stop_reason=parent.stop_reason,
                events=parent.take_events(),
                kv_transfer_params=kv_transfer_params,
                trace_headers=parent.trace_headers,
                num_cached_tokens=parent.num_cached_tokens,
                num_external_computed_tokens=parent.num_external_computed_tokens,
                routed_experts=routed_experts,
                num_nans_in_logits=parent.num_nans_in_logits,
            )
        )
    engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}
    if (stats := self.make_stats(spec_decoding_stats, None, model_runner_output.cudagraph_stats, None)) is not None:
        if (eco := next(iter(engine_core_outputs.values()), None)) is None:
            engine_core_outputs[0] = eco = EngineCoreOutputs()
        eco.scheduler_stats = stats
    return engine_core_outputs

Scheduler.update_draft_token_ids = _lumo_fb_update_draft_token_ids
Scheduler.schedule = _lumo_fb_schedule
Scheduler.update_from_output = _lumo_fb_update_from_output
"""
    sch.write_text(text + patch)
    import py_compile
    py_compile.compile(str(sch), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b scheduler path clone/collapse patch')

gm = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = gm.read_text()
sentinel = '# LUMO_FB_PATHS_RUNNER'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b gpu runner collapse patch already present')
else:
    patch = r"""

# LUMO_FB_PATHS_RUNNER: GPU-side partial-block clone and winner-state rename.
import os as _lumo_fb_os
import time as _lumo_fb_time

_lumo_fb_orig_update_states = GPUModelRunner._update_states

def _lumo_fb_replace_req_id_in_batch(input_batch, old_id, new_id):
    idx = input_batch.req_id_to_index.pop(old_id, None)
    if idx is None:
        return None
    old_parent_idx = input_batch.req_id_to_index.pop(new_id, None)
    if old_parent_idx is not None and old_parent_idx != idx:
        input_batch.remove_request(new_id)
    input_batch.req_id_to_index[new_id] = idx
    input_batch._req_ids[idx] = new_id
    for attr in ("greedy_reqs", "random_reqs", "top_p_reqs", "top_k_reqs",
                 "frequency_penalties_reqs", "presence_penalties_reqs",
                 "repetition_penalties_reqs", "has_allowed_token_ids"):
        s = getattr(input_batch, attr, None)
        if s is not None and old_id in s:
            s.discard(old_id)
            s.add(new_id)
    if old_id in input_batch.num_logprobs:
        input_batch.num_logprobs[new_id] = input_batch.num_logprobs.pop(old_id)
    return idx

def _lumo_fb_foreach_copy_(dst_views, src_views):
    if not dst_views:
        return
    try:
        torch._foreach_copy_(dst_views, src_views)
        return
    except Exception:
        pass
    for dst, src in zip(dst_views, src_views):
        dst.copy_(src)

def _lumo_fb_copy_block_id(self, src, dst):
    copied_bytes = 0
    dst_views = []
    src_views = []
    seen = set()
    for kv in getattr(self, "kv_caches", []):
        if id(kv) in seen:
            continue
        seen.add(id(kv))
        try:
            if isinstance(kv, list):
                for t in kv:
                    dst_view = t[dst]
                    src_view = t[src]
                    dst_views.append(dst_view)
                    src_views.append(src_view)
                    copied_bytes += int(dst_view.numel() * t.element_size())
            else:
                dst_view = kv[dst]
                src_view = kv[src]
                dst_views.append(dst_view)
                src_views.append(src_view)
                copied_bytes += int(dst_view.numel() * kv.element_size())
        except Exception:
            pass
    _lumo_fb_foreach_copy_(dst_views, src_views)
    return copied_bytes

def _lumo_fb_copy_mamba_block_id(self, src, dst, group_idx=None):
    copied_bytes = 0
    dst_views = []
    src_views = []
    try:
        mamba_group_ids = self._get_mamba_copy_bufs().mamba_group_ids
        if group_idx is not None:
            mamba_group_ids = [int(group_idx)]
        forward_context = self.compilation_config.static_forward_context
        for gid in mamba_group_ids:
            for layer_name in self.kv_cache_config.kv_cache_groups[gid].layer_names:
                attention = forward_context[layer_name]
                for state in attention.kv_cache:
                    dst_view = state[int(dst)]
                    src_view = state[int(src)]
                    dst_views.append(dst_view)
                    src_views.append(src_view)
                    copied_bytes += int(dst_view.numel() * state.element_size())
        _lumo_fb_foreach_copy_(dst_views, src_views)
    except Exception:
        pass
    return copied_bytes

def _lumo_fb_apply_runner_collapses(self, scheduler_output):
    collapses = getattr(scheduler_output, "lumo_fb_collapses", None)
    if not collapses:
        return
    for collapse in collapses:
        accepted_prefix_len = None
        if len(collapse) == 3:
            parent_id, winner_id, loser_id = collapse
            corrected_num_computed = None
        elif len(collapse) == 4:
            parent_id, winner_id, loser_id, corrected_num_computed = collapse
        else:
            parent_id, winner_id, loser_id, corrected_num_computed, accepted_prefix_len = collapse
        win_state = self.requests.pop(winner_id, None)
        self.requests.pop(parent_id, None)
        if win_state is not None:
            win_state.req_id = parent_id
            if corrected_num_computed is not None:
                win_state.num_computed_tokens = int(corrected_num_computed)
            self.requests[parent_id] = win_state
        row_idx = _lumo_fb_replace_req_id_in_batch(self.input_batch, winner_id, parent_id)
        self.input_batch.remove_request(loser_id)
        self.requests.pop(loser_id, None)
        if winner_id in self.mamba_state_idx:
            self.mamba_state_idx[parent_id] = self.mamba_state_idx.pop(winner_id)
        self.mamba_state_idx.pop(loser_id, None)
        if accepted_prefix_len is not None and row_idx is not None:
            try:
                self.input_batch.num_accepted_tokens_cpu[row_idx] = int(accepted_prefix_len)
                self.input_batch.num_accepted_tokens_cpu_tensor[row_idx] = int(accepted_prefix_len)
                self._lumo_fb_accept_lens = getattr(self, "_lumo_fb_accept_lens", {})
                self._lumo_fb_accept_lens[parent_id] = int(accepted_prefix_len)
            except Exception:
                pass
        scheduler_output.finished_req_ids.discard(winner_id)
        scheduler_output.finished_req_ids.discard(loser_id)

def _lumo_fb_restore_parent_accept_lens(self):
    accept_lens = getattr(self, "_lumo_fb_accept_lens", None)
    if not accept_lens:
        return
    for rid, accept_len in list(accept_lens.items()):
        idx = self.input_batch.req_id_to_index.get(rid)
        if idx is None:
            continue
        try:
            self.input_batch.num_accepted_tokens_cpu[idx] = int(accept_len)
            self.input_batch.num_accepted_tokens_cpu_tensor[idx] = int(accept_len)
        except Exception:
            pass

def _lumo_fb_inherit_clone_mamba_state(self, scheduler_output):
    marker = "::lumo_fb::"
    accept_lens = getattr(self, "_lumo_fb_accept_lens", {})
    for rid in scheduler_output.num_scheduled_tokens:
        if marker not in rid:
            continue
        parent_id = rid.split(marker, 1)[0]
        clone_idx = self.input_batch.req_id_to_index.get(rid)
        if clone_idx is not None and parent_id in accept_lens:
            try:
                self.input_batch.num_accepted_tokens_cpu[clone_idx] = int(accept_lens[parent_id])
                self.input_batch.num_accepted_tokens_cpu_tensor[clone_idx] = int(accept_lens[parent_id])
            except Exception:
                pass
        if rid in self.mamba_state_idx:
            continue
        parent_idx = self.mamba_state_idx.get(parent_id)
        if parent_idx is None:
            continue
        req_state = self.requests.get(rid)
        if req_state is None:
            continue
        ok = True
        try:
            for group_id in self._get_mamba_copy_bufs().mamba_group_ids:
                if parent_idx >= len(req_state.block_ids[group_id]):
                    ok = False
                    break
        except Exception:
            ok = False
        if ok:
            self.mamba_state_idx[rid] = parent_idx

def _lumo_fb_state_debug(self, scheduler_output, phase):
    if _lumo_fb_os.environ.get("LUMO_FB_DEBUG") != "1":
        return
    try:
        import json as _fbj, time as _fbt
        global _LUMO_FB_STATE_DBG_FH
        try:
            _LUMO_FB_STATE_DBG_FH
        except NameError:
            _LUMO_FB_STATE_DBG_FH = open("/logs/fb_state_debug.jsonl", "a", buffering=1)
        marker = "::lumo_fb::"
        rows = []
        for rid in scheduler_output.num_scheduled_tokens:
            if marker not in rid:
                continue
            parent_id = rid.split(marker, 1)[0]
            req_state = self.requests.get(rid)
            parent_state = self.requests.get(parent_id)
            rows.append({
                "rid": rid,
                "parent": parent_id,
                "clone_idx": self.mamba_state_idx.get(rid),
                "parent_idx": self.mamba_state_idx.get(parent_id),
                "clone_num_computed": getattr(req_state, "num_computed_tokens", None),
                "parent_num_computed": getattr(parent_state, "num_computed_tokens", None),
                "clone_block_lens": [len(g) for g in getattr(req_state, "block_ids", ())] if req_state is not None else None,
                "parent_block_lens": [len(g) for g in getattr(parent_state, "block_ids", ())] if parent_state is not None else None,
                "clone_mamba_blocks": [
                    list(getattr(req_state, "block_ids", ())[gid])
                    for gid in self._get_mamba_copy_bufs().mamba_group_ids
                ] if req_state is not None else None,
                "parent_mamba_blocks": [
                    list(getattr(parent_state, "block_ids", ())[gid])
                    for gid in self._get_mamba_copy_bufs().mamba_group_ids
                ] if parent_state is not None else None,
                "clone_state_block_ids": [
                    getattr(req_state, "block_ids", ())[gid][self.mamba_state_idx[rid]]
                    for gid in self._get_mamba_copy_bufs().mamba_group_ids
                    if rid in self.mamba_state_idx and self.mamba_state_idx[rid] < len(getattr(req_state, "block_ids", ())[gid])
                ] if req_state is not None else None,
                "parent_state_block_ids": [
                    getattr(parent_state, "block_ids", ())[gid][self.mamba_state_idx[parent_id]]
                    for gid in self._get_mamba_copy_bufs().mamba_group_ids
                    if parent_id in self.mamba_state_idx and self.mamba_state_idx[parent_id] < len(getattr(parent_state, "block_ids", ())[gid])
                ] if parent_state is not None else None,
                "clone_accept_cpu": (int(self.input_batch.num_accepted_tokens_cpu[self.input_batch.req_id_to_index[rid]])
                                     if rid in self.input_batch.req_id_to_index else None),
                "parent_accept_cpu": (int(self.input_batch.num_accepted_tokens_cpu[self.input_batch.req_id_to_index[parent_id]])
                                      if parent_id in self.input_batch.req_id_to_index else None),
                "num_scheduled": scheduler_output.num_scheduled_tokens.get(rid),
                "draft": scheduler_output.scheduled_spec_decode_tokens.get(rid),
            })
        if rows:
            _LUMO_FB_STATE_DBG_FH.write(_fbj.dumps({
                "ts": round(_fbt.time(), 4),
                "phase": phase,
                "rows": rows,
            }) + chr(10))
    except Exception:
        pass

def _lumo_fb_update_states(self, scheduler_output):
    if _lumo_fb_os.environ.get("LUMO_FB_PATHS") == "1":
        _lumo_fb_apply_runner_collapses(self, scheduler_output)
    ret = _lumo_fb_orig_update_states(self, scheduler_output)
    if _lumo_fb_os.environ.get("LUMO_FB_PATHS") == "1":
        _lumo_fb_restore_parent_accept_lens(self)
        _lumo_fb_inherit_clone_mamba_state(self, scheduler_output)
        _lumo_fb_state_debug(self, scheduler_output, "after_update_states")
        _lumo_fb_copies = list(getattr(scheduler_output, "lumo_fb_block_copies", []) or [])
        _lumo_fb_copy_us = 0
        if _lumo_fb_copies:
            _lumo_fb_copy_t0 = _lumo_fb_time.perf_counter_ns()
            _lumo_fb_copy_bytes = 0
            _lumo_fb_copy_detail = []
            _lumo_fb_cuda_start = None
            _lumo_fb_cuda_end = None
            if _lumo_fb_os.environ.get("LUMO_FB_DEBUG") == "1":
                try:
                    _lumo_fb_cuda_start = torch.cuda.Event(enable_timing=True)
                    _lumo_fb_cuda_end = torch.cuda.Event(enable_timing=True)
                    _lumo_fb_cuda_start.record()
                except Exception:
                    _lumo_fb_cuda_start = None
                    _lumo_fb_cuda_end = None
            for item in _lumo_fb_copies:
                if len(item) >= 3 and item[0] == "mamba":
                    if len(item) >= 4:
                        _bytes = _lumo_fb_copy_mamba_block_id(self, item[2], item[3], item[1])
                        _lumo_fb_copy_detail.append({
                            "kind": "mamba", "group": int(item[1]),
                            "src": int(item[2]), "dst": int(item[3]),
                            "bytes": int(_bytes),
                        })
                    else:
                        _bytes = _lumo_fb_copy_mamba_block_id(self, item[1], item[2])
                        _lumo_fb_copy_detail.append({
                            "kind": "mamba_legacy_all_groups",
                            "src": int(item[1]), "dst": int(item[2]),
                            "bytes": int(_bytes),
                        })
                    _lumo_fb_copy_bytes += int(_bytes)
                else:
                    src, dst = item
                    _bytes = _lumo_fb_copy_block_id(self, int(src), int(dst))
                    _lumo_fb_copy_bytes += int(_bytes)
                    _lumo_fb_copy_detail.append({
                        "kind": "kv", "src": int(src), "dst": int(dst),
                        "bytes": int(_bytes),
                    })
            if _lumo_fb_cuda_start is not None and _lumo_fb_cuda_end is not None:
                try:
                    _lumo_fb_cuda_end.record()
                    _lumo_fb_cuda_end.synchronize()
                    _lumo_fb_copy_us = int(_lumo_fb_cuda_start.elapsed_time(_lumo_fb_cuda_end) * 1000.0)
                except Exception:
                    _lumo_fb_copy_us = int((_lumo_fb_time.perf_counter_ns() - _lumo_fb_copy_t0) // 1000)
            else:
                _lumo_fb_copy_us = int((_lumo_fb_time.perf_counter_ns() - _lumo_fb_copy_t0) // 1000)
        scheduler_output.lumo_fb_state_copy_us = int(_lumo_fb_copy_us)
        scheduler_output.lumo_fb_state_copy_bytes = int(locals().get("_lumo_fb_copy_bytes", 0))
        scheduler_output.lumo_fb_state_copy_detail = locals().get("_lumo_fb_copy_detail", [])
        scheduler_output.lumo_fb_kv_blocks_copied = int(getattr(
            scheduler_output, "lumo_fb_kv_blocks_copied",
            sum(1 for item in _lumo_fb_copies if not (len(item) >= 3 and item[0] == "mamba"))))
        scheduler_output.lumo_fb_mamba_blocks_copied = int(getattr(
            scheduler_output, "lumo_fb_mamba_blocks_copied",
            sum(1 for item in _lumo_fb_copies if len(item) >= 3 and item[0] == "mamba")))
    return ret

GPUModelRunner._update_states = _lumo_fb_update_states
"""
    gm.write_text(text + patch)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b gpu runner copy/collapse patch')

text = gm.read_text()
sentinel = '# LUMO_FB_ACCEPT_LEN_PREPROCESS_SYNC'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b mamba accepted-prefix sync patch already present')
else:
    patch = r"""

# LUMO_FB_ACCEPT_LEN_PREPROCESS_SYNC: after F_b collapses a verifier clone
# back into the parent request, the next Mamba forward must consume the
# winner's committed prefix length (accepted drafts + sampled token).  Upstream
# preprocess_mamba may reset num_accepted_tokens_cpu to 1 after internal state
# block copies; restore F_b's explicit accepted-prefix lengths before the GPU
# sync immediately following preprocess_mamba.
import json as _lumo_fb_ap_j
import time as _lumo_fb_ap_t

_lumo_fb_ap_prev_update_states = GPUModelRunner._update_states
_lumo_fb_ap_orig_preprocess_mamba = mamba_utils.preprocess_mamba

def _lumo_fb_ap_update_states(self, scheduler_output):
    ret = _lumo_fb_ap_prev_update_states(self, scheduler_output)
    if _lumo_fb_os.environ.get("LUMO_FB_PATHS") == "1":
        try:
            accept_lens = getattr(self, "_lumo_fb_accept_lens", {}) or {}
            marker = "::lumo_fb::"
            by_req = {}
            for req_id, accept_len in accept_lens.items():
                if req_id in self.input_batch.req_id_to_index:
                    by_req[req_id] = int(accept_len)
            for req_id in scheduler_output.num_scheduled_tokens:
                if marker not in req_id:
                    continue
                parent_id = req_id.split(marker, 1)[0]
                if parent_id in accept_lens and req_id in self.input_batch.req_id_to_index:
                    by_req[req_id] = int(accept_lens[parent_id])
            if by_req:
                self.input_batch._lumo_fb_accept_lens_by_req = by_req
        except Exception:
            pass
    return ret

def _lumo_fb_ap_preprocess_mamba(
    scheduler_output,
    kv_cache_config,
    cache_config,
    mamba_state_idx,
    input_batch,
    requests,
    forward_context,
    mamba_state_copy_funcs,
    copy_bufs,
):
    preserved = {}
    before = {}
    if _lumo_fb_os.environ.get("LUMO_FB_PATHS") == "1":
        try:
            explicit = getattr(input_batch, "_lumo_fb_accept_lens_by_req", {}) or {}
            for i, req_id in enumerate(input_batch.req_ids):
                if req_id in explicit:
                    val = int(explicit[req_id])
                    preserved[i] = val
                    before[req_id] = int(input_batch.num_accepted_tokens_cpu[i])
        except Exception:
            preserved = {}
            before = {}
    ret = _lumo_fb_ap_orig_preprocess_mamba(
        scheduler_output,
        kv_cache_config,
        cache_config,
        mamba_state_idx,
        input_batch,
        requests,
        forward_context,
        mamba_state_copy_funcs,
        copy_bufs,
    )
    if preserved:
        try:
            restored = {}
            after = {}
            for i, val in preserved.items():
                req_id = input_batch.req_ids[i]
                after[req_id] = int(input_batch.num_accepted_tokens_cpu[i])
                input_batch.num_accepted_tokens_cpu[i] = val
                input_batch.num_accepted_tokens_cpu_tensor[i] = val
                restored[req_id] = val
            if _lumo_fb_os.environ.get("LUMO_FB_DEBUG") == "1":
                global _LUMO_FB_AP_DBG_FH
                try:
                    _LUMO_FB_AP_DBG_FH
                except NameError:
                    _LUMO_FB_AP_DBG_FH = open("/logs/fb_mamba_preprocess_debug.jsonl", "a", buffering=1)
                _LUMO_FB_AP_DBG_FH.write(_lumo_fb_ap_j.dumps({
                    "ts": round(_lumo_fb_ap_t.time(), 4),
                    "before": before,
                    "after_preprocess": after,
                    "restored": restored,
                    "state_idx": {rid: mamba_state_idx.get(rid) for rid in restored},
                    "num_scheduled": {
                        rid: scheduler_output.num_scheduled_tokens.get(rid)
                        for rid in restored
                    },
                    "draft": {
                        rid: scheduler_output.scheduled_spec_decode_tokens.get(rid)
                        for rid in restored
                    },
                }) + chr(10))
        except Exception:
            pass
    return ret

GPUModelRunner._update_states = _lumo_fb_ap_update_states
mamba_utils.preprocess_mamba = _lumo_fb_ap_preprocess_mamba
"""
    gm.write_text(text + patch)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b mamba accepted-prefix sync patch')

text = gm.read_text()
sentinel = '# LUMO_FB_DRAFT_CPU_RESIZE'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b draft CPU resize patch already present')
else:
    anchor = nl.join([
        '        num_reqs = draft_token_ids.shape[0]',
        '        with torch.cuda.stream(self.draft_token_ids_copy_stream):',
        '            if not zeros_only:',
    ])
    inject = nl.join([
        '        num_reqs = draft_token_ids.shape[0]',
        '        # LUMO_FB_DRAFT_CPU_RESIZE: F_b flattens K=2 depth=3 into',
        '        # six proposed ids before the scheduler splits them into two',
        '        # verifier rows. Keep vLLM configured at depth=3, but grow the',
        '        # host handoff buffer to carry the flattened proposer output.',
        '        if getattr(draft_token_ids, "ndim", 0) == 2 and draft_token_ids.shape[1] > self.draft_token_ids_cpu.shape[1]:',
        '            self.draft_token_ids_cpu = torch.empty(',
        '                (self.max_num_reqs, int(draft_token_ids.shape[1])),',
        '                dtype=torch.int64,',
        '                device="cpu",',
        '                pin_memory=self.pin_memory,',
        '            )',
        '        with torch.cuda.stream(self.draft_token_ids_copy_stream):',
        '            if not zeros_only:',
    ])
    if anchor not in text:
        raise RuntimeError('F_b draft CPU resize anchor not found')
    text = text.replace(anchor, inject, 1)
    gm.write_text(text)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b draft CPU resize patch')

sch = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py')
text = sch.read_text()
sentinel = '# LUMO_FB_KVCACHEBLOCKS_FIX'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b KVCacheBlocks helper patch already present')
else:
    old = nl.join([
        'def _lumo_fb_block_ids_from_blocks(blocks):',
        '    return tuple([blk.block_id for blk in group] for group in blocks)',
    ])
    new = nl.join([
        'def _lumo_fb_block_ids_from_blocks(blocks):',
        '    # LUMO_FB_KVCACHEBLOCKS_FIX: vLLM 0.19 may return KVCacheBlocks here.',
        '    if hasattr(blocks, "get_block_ids"):',
        '        return blocks.get_block_ids()',
        '    return tuple([blk.block_id for blk in group] for group in blocks)',
    ])
    if old not in text:
        raise RuntimeError('F_b KVCacheBlocks helper anchor not found')
    text = text.replace(old, new, 1)
    sch.write_text(text)
    import py_compile
    py_compile.compile(str(sch), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b KVCacheBlocks helper patch')

gm = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = gm.read_text()
sentinel = '# LUMO_FB_DRAFT_CPU_COPY_WIDTH'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b draft CPU copy-width patch already present')
else:
    old = nl.join([
        '                self.draft_token_ids_cpu[:num_reqs].copy_(',
        '                    draft_token_ids, non_blocking=True',
        '                )',
    ])
    new = nl.join([
        '                # LUMO_FB_DRAFT_CPU_COPY_WIDTH: buffer may be wider',
        '                # than the current proposal after path rows fall back',
        '                # to ordinary depth-3 drafting.',
        '                _lumo_fb_w = int(draft_token_ids.shape[1])',
        '                self.draft_token_ids_cpu[:num_reqs, :_lumo_fb_w].copy_(',
        '                    draft_token_ids, non_blocking=True',
        '                )',
    ])
    if old not in text:
        raise RuntimeError('F_b draft CPU copy-width anchor not found')
    text = text.replace(old, new, 1)
    gm.write_text(text)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b draft CPU copy-width patch')

text = gm.read_text()
sentinel = '# LUMO_FB_DRAFT_CPU_READ_WIDTH'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b draft CPU read-width patch already present')
else:
    old = nl.join([
        '                _lumo_fb_w = int(draft_token_ids.shape[1])',
        '                self.draft_token_ids_cpu[:num_reqs, :_lumo_fb_w].copy_(',
        '                    draft_token_ids, non_blocking=True',
        '                )',
    ])
    new = nl.join([
        '                _lumo_fb_w = int(draft_token_ids.shape[1])',
        '                # LUMO_FB_DRAFT_CPU_READ_WIDTH: remember how many',
        '                # columns are valid for this copy. The host buffer',
        '                # can remain width-6 after an F_b root split; ordinary',
        '                # depth-3 rows must not expose stale tail columns.',
        '                self._lumo_fb_draft_cpu_width = _lumo_fb_w',
        '                self.draft_token_ids_cpu[:num_reqs, :_lumo_fb_w].copy_(',
        '                    draft_token_ids, non_blocking=True',
        '                )',
    ])
    if old not in text:
        raise RuntimeError('F_b draft CPU read-width write anchor not found')
    text = text.replace(old, new, 1)
    old = '        return self.draft_token_ids_cpu[: len(req_ids)].tolist(), req_ids'
    new = nl.join([
        '        # LUMO_FB_DRAFT_CPU_READ_WIDTH: trim to the width copied for',
        '        # this specific proposal rather than the maximum buffer width.',
        '        _lumo_fb_w = int(getattr(self, "_lumo_fb_draft_cpu_width", self.draft_token_ids_cpu.shape[1]))',
        '        return self.draft_token_ids_cpu[: len(req_ids), :_lumo_fb_w].tolist(), req_ids',
    ])
    if old not in text:
        raise RuntimeError('F_b draft CPU read-width read anchor not found')
    text = text.replace(old, new, 1)
    gm.write_text(text)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b draft CPU read-width patch')

text = gm.read_text()
sentinel = '# LUMO_FB_META_DEBUG'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b metadata debug patch already present')
else:
    patch = r"""

# LUMO_FB_META_DEBUG: bounded verifier-layout logging for F_b canaries.
import os as _lumo_fb_meta_os
_lumo_fb_orig_calc_spec_decode_metadata = GPUModelRunner._calc_spec_decode_metadata
_lumo_fb_meta_debug_count = 0

def _lumo_fb_calc_spec_decode_metadata(self, num_draft_tokens, cu_num_tokens):
    global _lumo_fb_meta_debug_count
    meta = _lumo_fb_orig_calc_spec_decode_metadata(self, num_draft_tokens, cu_num_tokens)
    if (_lumo_fb_meta_os.environ.get("LUMO_FB_PATHS") == "1"
            and _lumo_fb_meta_os.environ.get("LUMO_FB_DEBUG") == "1"
            and _lumo_fb_meta_debug_count < 64):
        try:
            import json as _fbj, time as _fbt
            global _LUMO_FB_META_DBG_FH
            try:
                _LUMO_FB_META_DBG_FH
            except NameError:
                _LUMO_FB_META_DBG_FH = open("/logs/fb_meta_debug.jsonl", "a", buffering=1)
            _LUMO_FB_META_DBG_FH.write(_fbj.dumps({
                "ts": round(_fbt.time(), 4),
                "num_draft_tokens": num_draft_tokens.tolist() if hasattr(num_draft_tokens, "tolist") else list(num_draft_tokens),
                "draft_token_ids": meta.draft_token_ids.detach().cpu().tolist(),
                "target_logits_indices": meta.target_logits_indices.detach().cpu().tolist(),
                "bonus_logits_indices": meta.bonus_logits_indices.detach().cpu().tolist(),
                "logits_indices": meta.logits_indices.detach().cpu().tolist(),
            }) + chr(10))
            _lumo_fb_meta_debug_count += 1
        except Exception:
            pass
    return meta

GPUModelRunner._calc_spec_decode_metadata = _lumo_fb_calc_spec_decode_metadata
"""
    gm.write_text(text + patch)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b metadata debug patch')

text = gm.read_text()
sentinel = '# LUMO_FB_PREPARE_SPEC_IDS'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b prepare spec-token patch already present')
else:
    patch = r"""

# LUMO_FB_PREPARE_SPEC_IDS: F_b verifier clone rows are scheduled as new
# internal requests, so vanilla _prepare_input_ids does not have previous-step
# draft tensors to scatter into the spec-token slots. Fill those slots from
# scheduler_output.scheduled_spec_decode_tokens before target forward and before
# SpecDecodeMetadata extracts draft_token_ids.
import os as _lumo_fb_prep_os
import torch as _lumo_fb_prep_torch

_lumo_fb_orig_prepare_input_ids = GPUModelRunner._prepare_input_ids

def _lumo_fb_prepare_input_ids(self, scheduler_output, num_reqs,
                               total_num_scheduled_tokens, cu_num_tokens):
    ret = _lumo_fb_orig_prepare_input_ids(
        self, scheduler_output, num_reqs, total_num_scheduled_tokens,
        cu_num_tokens)
    if _lumo_fb_prep_os.environ.get("LUMO_FB_PATHS") != "1":
        return ret
    flat_indices = []
    flat_values = []
    for req_id, toks in scheduler_output.scheduled_spec_decode_tokens.items():
        if "::lumo_fb::" not in req_id or not toks:
            continue
        req_idx = self.input_batch.req_id_to_index.get(req_id)
        if req_idx is None:
            continue
        draft_len = len(toks)
        row_end = int(cu_num_tokens[req_idx])
        row_start = row_end - (draft_len + 1)
        flat_indices.extend(range(row_start + 1, row_start + 1 + draft_len))
        flat_values.extend(int(t) for t in toks)
    if flat_indices:
        idx = _lumo_fb_prep_torch.tensor(
            flat_indices, dtype=_lumo_fb_prep_torch.int64,
            device=self.input_ids.gpu.device)
        vals = _lumo_fb_prep_torch.tensor(
            flat_values, dtype=self.input_ids.gpu.dtype,
            device=self.input_ids.gpu.device)
        self.input_ids.gpu.scatter_(dim=0, index=idx, src=vals)
    return ret

GPUModelRunner._prepare_input_ids = _lumo_fb_prepare_input_ids
"""
    gm.write_text(text + patch)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b prepare spec-token patch')

rs = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py')
text = rs.read_text()
sentinel = '# LUMO_FB_SHARED_ROOT_SAMPLE'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b shared-root sampler patch already present')
else:
    helper = r"""

# LUMO_FB_SHARED_ROOT_SAMPLE: K=2/depth=3 sibling rows verify the same
# logical request root. Random sampling must draw that root target token once
# and reuse it across siblings; otherwise path0/path1 independently sample
# different roots and the second-root win case is lost.
import json as _lumo_fb_rs_json
import os as _lumo_fb_rs_os
import time as _lumo_fb_rs_time
_LUMO_FB_SHARED_ROOT_PAIRS = None
_LUMO_FB_SAMPLER_DEBUG_COUNT = 0

def _lumo_fb_set_shared_root_pairs(pairs):
    global _LUMO_FB_SHARED_ROOT_PAIRS
    _LUMO_FB_SHARED_ROOT_PAIRS = pairs

def _lumo_fb_sample_from_probs(probs, generator=None):
    q = torch.empty_like(probs)
    if generator is None:
        q.exponential_()
    else:
        q.exponential_(generator=generator)
    return torch.argmax(probs / q).to(torch.int32)

def _lumo_fb_sampler_to_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [int(v) if not isinstance(v, (list, tuple)) else _lumo_fb_sampler_to_list(v)
                for v in value]
    try:
        return int(value)
    except Exception:
        return str(value)

def _lumo_fb_sampler_debug(
    tag,
    draft_token_ids,
    num_draft_tokens,
    max_spec_len,
    cu_num_draft_tokens,
    target_logits,
    bonus_token_ids,
    sampling_metadata,
    output_token_ids,
):
    global _LUMO_FB_SAMPLER_DEBUG_COUNT
    if _lumo_fb_rs_os.environ.get("LUMO_FB_DEBUG") != "1":
        return
    limit = int(_lumo_fb_rs_os.environ.get("LUMO_FB_SAMPLER_DEBUG_LIMIT", "128"))
    if _LUMO_FB_SAMPLER_DEBUG_COUNT >= limit:
        return
    _LUMO_FB_SAMPLER_DEBUG_COUNT += 1
    target_argmax = target_logits.argmax(dim=-1).to(torch.int32)
    row = {
        "event": "fb_sampler_debug",
        "idx": _LUMO_FB_SAMPLER_DEBUG_COUNT - 1,
        "tag": tag,
        "ts": _lumo_fb_rs_time.time(),
        "max_spec_len": int(max_spec_len),
        "num_draft_tokens": [int(x) for x in num_draft_tokens],
        "cu_num_draft_tokens": _lumo_fb_sampler_to_list(cu_num_draft_tokens),
        "draft_token_ids": _lumo_fb_sampler_to_list(draft_token_ids),
        "target_argmax": _lumo_fb_sampler_to_list(target_argmax),
        "target_logits_shape": list(target_logits.shape),
        "bonus_token_ids": _lumo_fb_sampler_to_list(bonus_token_ids),
        "output_token_ids": _lumo_fb_sampler_to_list(output_token_ids),
        "all_greedy": bool(getattr(sampling_metadata, "all_greedy", False)),
        "all_random": bool(getattr(sampling_metadata, "all_random", False)),
    }
    path = _lumo_fb_rs_os.environ.get(
        "LUMO_FB_SAMPLER_DEBUG_PATH", "/logs/fb_sampler_debug.jsonl")
    with open(path, "a") as fh:
        fh.write(_lumo_fb_rs_json.dumps(row) + "\n")

def _lumo_fb_rejection_sample_with_debug(
    tag,
    draft_token_ids,
    num_draft_tokens,
    max_spec_len,
    cu_num_draft_tokens,
    draft_probs,
    target_logits,
    bonus_token_ids,
    sampling_metadata,
):
    output_token_ids = rejection_sample(
        draft_token_ids, num_draft_tokens, max_spec_len,
        cu_num_draft_tokens, draft_probs, target_logits,
        bonus_token_ids, sampling_metadata)
    _lumo_fb_sampler_debug(
        tag, draft_token_ids, num_draft_tokens, max_spec_len,
        cu_num_draft_tokens, target_logits, bonus_token_ids,
        sampling_metadata, output_token_ids)
    return output_token_ids

def _lumo_fb_shared_root_rejection_sample(
    draft_token_ids,
    num_draft_tokens,
    max_spec_len,
    cu_num_draft_tokens,
    draft_probs,
    target_logits,
    bonus_token_ids,
    sampling_metadata,
):
    if (_lumo_fb_rs_os.environ.get("LUMO_FB_PATHS") != "1"
            or _lumo_fb_rs_os.environ.get("LUMO_FB_DISABLE_SHARED_ROOT") == "1"
            or draft_probs is not None
            or len(num_draft_tokens) % 2 != 0
            or len(num_draft_tokens) == 0):
        return _lumo_fb_rejection_sample_with_debug(
            "fallback_unpaired_or_disabled",
            draft_token_ids, num_draft_tokens, max_spec_len,
            cu_num_draft_tokens, draft_probs, target_logits,
            bonus_token_ids, sampling_metadata)
    fb_depth = int(num_draft_tokens[0])
    if fb_depth < 1 or any(int(n) != fb_depth for n in num_draft_tokens):
        return _lumo_fb_rejection_sample_with_debug(
            "fallback_uneven_depth",
            draft_token_ids, num_draft_tokens, max_spec_len,
            cu_num_draft_tokens, draft_probs, target_logits,
            bonus_token_ids, sampling_metadata)

    batch_size = len(num_draft_tokens)
    device = target_logits.device
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    pairs = globals().get("_LUMO_FB_SHARED_ROOT_PAIRS", None)
    if pairs:
        pairs = [(int(a), int(b)) for a, b in pairs
                 if 0 <= int(a) < batch_size and 0 <= int(b) < batch_size]
        covered = set()
        for a, b in pairs:
            covered.add(a)
            covered.add(b)
        if len(covered) != batch_size:
            return _lumo_fb_rejection_sample_with_debug(
                "fallback_incomplete_pairs",
                draft_token_ids, num_draft_tokens, max_spec_len,
                cu_num_draft_tokens, draft_probs, target_logits,
                bonus_token_ids, sampling_metadata)
    else:
        pairs = [(i, i + 1) for i in range(0, batch_size, 2)]

    if sampling_metadata.all_greedy:
        target_argmax = target_logits.argmax(dim=-1).to(torch.int32)
        for parent_idx, sibling_idx in pairs:
            parent_start = 0 if parent_idx == 0 else int(cu_num_draft_tokens[parent_idx - 1].item())
            shared_root = target_argmax[parent_start]
            for req_idx in (parent_idx, sibling_idx):
                start = 0 if req_idx == 0 else int(cu_num_draft_tokens[req_idx - 1].item())
                rejected = False
                for pos in range(fb_depth):
                    tok = shared_root if pos == 0 else target_argmax[start + pos]
                    output_token_ids[req_idx, pos] = tok
                    if int(draft_token_ids[start + pos].item()) != int(tok.item()):
                        rejected = True
                        break
                if not rejected:
                    output_token_ids[req_idx, fb_depth] = bonus_token_ids[req_idx, 0]
        _lumo_fb_sampler_debug(
            "shared_root_greedy",
            draft_token_ids, num_draft_tokens, max_spec_len,
            cu_num_draft_tokens, target_logits, bonus_token_ids,
            sampling_metadata, output_token_ids)
        return output_token_ids

    if sampling_metadata.all_random:
        greedy_by_req = [False] * batch_size
    else:
        greedy_by_req = (sampling_metadata.temperature == GREEDY_TEMPERATURE).detach().cpu().tolist()

    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    uniform_probs = generate_uniform_probs(
        draft_token_ids.shape[0], num_draft_tokens,
        sampling_metadata.generators, device)
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len, num_draft_tokens, cu_num_draft_tokens,
        draft_token_ids, draft_probs, target_probs, sampling_metadata, device)
    target_argmax = target_logits.argmax(dim=-1).to(torch.int32)

    for parent_idx, sibling_idx in pairs:
        start0 = 0 if parent_idx == 0 else int(cu_num_draft_tokens[parent_idx - 1].item())
        root_generator = sampling_metadata.generators.get(parent_idx)
        if greedy_by_req[parent_idx]:
            shared_root = target_argmax[start0]
        else:
            shared_root = _lumo_fb_sample_from_probs(target_probs[start0], root_generator)
        for req_idx in (parent_idx, sibling_idx):
            start = 0 if req_idx == 0 else int(cu_num_draft_tokens[req_idx - 1].item())
            rejected = False
            for pos in range(fb_depth):
                tok_idx = start + pos
                draft_id = draft_token_ids[tok_idx].to(torch.int32)
                if pos == 0:
                    if int(shared_root.item()) == int(draft_id.item()):
                        output_token_ids[req_idx, pos] = draft_id
                        continue
                    output_token_ids[req_idx, pos] = shared_root
                    rejected = True
                    break
                if greedy_by_req[req_idx]:
                    tok = target_argmax[tok_idx]
                    output_token_ids[req_idx, pos] = tok
                    if int(draft_id.item()) != int(tok.item()):
                        rejected = True
                        break
                    continue
                target_prob = target_probs[tok_idx, draft_id.to(torch.int64)]
                uniform_prob = uniform_probs[tok_idx]
                if target_prob > 0 and target_prob >= uniform_prob:
                    output_token_ids[req_idx, pos] = draft_id
                else:
                    output_token_ids[req_idx, pos] = recovered_token_ids[tok_idx]
                    rejected = True
                    break
            if not rejected:
                output_token_ids[req_idx, fb_depth] = bonus_token_ids[req_idx, 0]
    _lumo_fb_sampler_debug(
        "shared_root_random",
        draft_token_ids, num_draft_tokens, max_spec_len,
        cu_num_draft_tokens, target_logits, bonus_token_ids,
        sampling_metadata, output_token_ids)
    return output_token_ids
"""
    old = nl.join([
        '        output_token_ids = rejection_sample(',
        '            metadata.draft_token_ids,',
        '            metadata.num_draft_tokens,',
        '            metadata.max_spec_len,',
        '            metadata.cu_num_draft_tokens,',
        '            draft_probs,',
        '            target_logits,',
        '            bonus_token_ids,',
        '            sampling_metadata,',
        '        )',
    ])
    new = nl.join([
        '        output_token_ids = _lumo_fb_shared_root_rejection_sample(',
        '            metadata.draft_token_ids,',
        '            metadata.num_draft_tokens,',
        '            metadata.max_spec_len,',
        '            metadata.cu_num_draft_tokens,',
        '            draft_probs,',
        '            target_logits,',
        '            bonus_token_ids,',
        '            sampling_metadata,',
        '        )',
    ])
    if old not in text:
        raise RuntimeError('F_b shared-root sampler anchor not found')
    text = text.replace(old, new, 1) + helper
    rs.write_text(text)
    import py_compile
    py_compile.compile(str(rs), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b shared-root sampler patch')

sch = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py')
text = sch.read_text()
sentinel = '# LUMO_FB_INTERNAL_ROWS'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b internal-row scheduler patch already present')
else:
    patch = r"""

# LUMO_FB_INTERNAL_ROWS: Gate-2 path verifier rows without public scheduler
# request clones.  The scheduler keeps the parent as the only logical request,
# attaches internal verifier rows to SchedulerOutput, and owns only the fresh
# COW blocks allocated for those rows.
import os as _lumo_fb_ir_os
import json as _lumo_fb_ir_json
import time as _lumo_fb_ir_time

_lumo_fb_ir_prev_update_draft = Scheduler.update_draft_token_ids
_lumo_fb_ir_prev_schedule = Scheduler.schedule
_lumo_fb_ir_prev_update_output = Scheduler.update_from_output

def _lumo_fb_ir_enabled():
    return (_lumo_fb_ir_os.environ.get("LUMO_FB_PATHS") == "1"
            and _lumo_fb_ir_os.environ.get("LUMO_FB_INTERNAL_ROWS") == "1")

def _lumo_fb_ir_read_control(default_depth):
    depth = int(_lumo_fb_ir_os.environ.get("LUMO_FB_DEPTH", str(default_depth)))
    k = int(_lumo_fb_ir_os.environ.get("LUMO_FB_K", "2"))
    path = _lumo_fb_ir_os.environ.get("LUMO_FB_CONTROL_FILE", "/logs/fb_control.json")
    if path and _lumo_fb_ir_os.path.exists(path):
        st0 = _lumo_fb_ir_os.stat(path)
        with open(path) as fh:
            payload = _lumo_fb_ir_json.load(fh)
        st1 = _lumo_fb_ir_os.stat(path)
        if (st0.st_mtime_ns != st1.st_mtime_ns
                or st0.st_size != st1.st_size
                or getattr(st0, "st_ino", None) != getattr(st1, "st_ino", None)):
            raise RuntimeError(f"LUMO_FB_CONTROL_FILE changed during read: {path}")
        if payload.get("depth") is not None:
            depth = int(payload["depth"])
        if payload.get("k") is not None:
            k = int(payload["k"])
    if depth < 1:
        raise RuntimeError(f"LUMO_FB active depth {depth} invalid")
    if k < 1 or k > 2:
        raise RuntimeError(f"LUMO_FB active K {k} unsupported in this build")
    return int(depth), int(k)

def _lumo_fb_ir_sched_debug(event):
    if _lumo_fb_ir_os.environ.get("LUMO_FB_DEBUG") != "1":
        return
    try:
        global _LUMO_FB_IR_SCHED_DBG_FH
        try:
            _LUMO_FB_IR_SCHED_DBG_FH
        except NameError:
            _LUMO_FB_IR_SCHED_DBG_FH = open("/logs/fb_overhead_debug.jsonl", "a", buffering=1)
        event["ts"] = round(_lumo_fb_ir_time.time(), 4)
        _LUMO_FB_IR_SCHED_DBG_FH.write(_lumo_fb_ir_json.dumps(event) + chr(10))
    except Exception:
        pass

def _lumo_fb_ir_row_id(parent_id, path_idx):
    return f"{parent_id}::lumo_fb_ir::{path_idx}"

def _lumo_fb_ir_cdiv(a, b):
    return (int(a) + int(b) - 1) // int(b)

def _lumo_fb_ir_new_block(manager):
    return manager.block_pool.get_new_blocks(1)[0]

def _lumo_fb_ir_kernel_rows_enabled():
    return _lumo_fb_ir_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1"

def _lumo_fb_ir_kernel_arrange_mamba_blocks(manager, req_blocks, curr_idx):
    # Arrange one GDN row as [shared read, private writes...].
    # The GDN metadata patch reads the initial recurrent state from column 0
    # and writes token states to columns 1..num_spec+1.  This helper
    # creates that layout without copying recurrent state bytes.
    num_spec_blocks = int(getattr(manager.kv_cache_spec, "num_speculative_blocks", 0))
    if num_spec_blocks < 1:
        return list(req_blocks), [], None
    row_blocks = list(req_blocks)
    read_idx = int(curr_idx)
    last_idx = read_idx + 1 + num_spec_blocks
    if last_idx > len(row_blocks):
        raise RuntimeError(
            f"LUMO_FB_KERNEL_ROWS block table too short: need {last_idx}, "
            f"got {len(row_blocks)}")
    read_block = row_blocks[read_idx]
    if read_block == getattr(manager, "_null_block", None):
        raise RuntimeError("LUMO_FB_KERNEL_ROWS read state block is null")
    owned = []
    # Column 0 stays as the shared read state.  The extra speculative block
    # added by LUMO_FB_KERNEL_ROWS gives us enough private write columns.
    for logical_idx in range(read_idx + 1, last_idx):
        dst = _lumo_fb_ir_new_block(manager)
        row_blocks[logical_idx] = dst
        owned.append(dst)
    return row_blocks, owned, read_block

def _lumo_fb_ir_prepare_parent_kernel_blocks(self, parent, num_scheduled_tokens):
    # Parent/path0 already has the correct no-copy layout after the extra
    # Mamba speculative block: col0 is the live prefix state populated by
    # preprocess_mamba, and cols1..num_spec+1 are private speculative writes.
    return []

def _lumo_fb_ir_alloc_blocks(self, parent, row_id, num_scheduled_tokens):
    copies = []
    owned = []
    row_block_ids = []
    for group_idx, manager in enumerate(self.kv_cache_manager.coordinator.single_type_managers):
        parent_blocks = list(manager.req_to_blocks.get(parent.request_id, ()))
        row_blocks = list(parent_blocks)
        if not row_blocks:
            row_block_ids.append([])
            continue
        block_size = int(manager.block_size)
        if getattr(manager, "mamba_cache_mode", None) == "align":
            num_spec_blocks = int(getattr(manager.kv_cache_spec, "num_speculative_blocks", 0))
            curr_idx = max(0, _lumo_fb_ir_cdiv(parent.num_computed_tokens + num_scheduled_tokens, block_size) - 1)
            if _lumo_fb_ir_kernel_rows_enabled():
                row_blocks, new_blocks, read_block = _lumo_fb_ir_kernel_arrange_mamba_blocks(
                    manager, parent_blocks, curr_idx)
                owned.extend((group_idx, blk) for blk in new_blocks)
                _lumo_fb_ir_sched_debug({
                    "event": "kernel_row_blocks",
                    "row": row_id,
                    "parent": parent.request_id,
                    "group": group_idx,
                    "curr_idx": curr_idx,
                    "read_block": getattr(read_block, "block_id", None),
                    "write_blocks": [blk.block_id for blk in new_blocks],
                })
            else:
                last_idx = min(len(row_blocks), curr_idx + 1 + num_spec_blocks)
                for logical_idx in range(curr_idx, last_idx):
                    src = row_blocks[logical_idx]
                    if src == getattr(manager, "_null_block", None):
                        continue
                    dst = _lumo_fb_ir_new_block(manager)
                    row_blocks[logical_idx] = dst
                    owned.append((group_idx, dst))
                    if logical_idx == curr_idx:
                        copies.append(("mamba", group_idx, src.block_id, dst.block_id))
        else:
            start_idx = max(0, int(parent.num_computed_tokens) // block_size)
            end_idx = min(len(row_blocks), _lumo_fb_ir_cdiv(parent.num_computed_tokens + num_scheduled_tokens, block_size))
            for logical_idx in range(start_idx, end_idx):
                src = row_blocks[logical_idx]
                if src == getattr(manager, "_null_block", None):
                    continue
                dst = _lumo_fb_ir_new_block(manager)
                row_blocks[logical_idx] = dst
                owned.append((group_idx, dst))
                if logical_idx == start_idx and (int(parent.num_computed_tokens) % block_size) != 0:
                    copies.append((src.block_id, dst.block_id))
        row_block_ids.append([blk.block_id for blk in row_blocks])
    self._lumo_fb_ir_owned_blocks = getattr(self, "_lumo_fb_ir_owned_blocks", {})
    self._lumo_fb_ir_owned_blocks[row_id] = owned
    self._lumo_fb_ir_row_block_ids = getattr(self, "_lumo_fb_ir_row_block_ids", {})
    self._lumo_fb_ir_row_block_ids[row_id] = tuple(row_block_ids)
    return tuple(row_block_ids), copies

def _lumo_fb_ir_transfer_owned_to_parent(self, parent_id, row_id):
    owned_by_row = getattr(self, "_lumo_fb_ir_owned_blocks", {})
    row_block_ids_by_row = getattr(self, "_lumo_fb_ir_row_block_ids", {})
    owned = owned_by_row.pop(row_id, [])
    row_block_ids = row_block_ids_by_row.pop(row_id, None)
    if not owned or row_block_ids is None:
        return
    by_group = {}
    for group_idx, block in owned:
        by_group.setdefault(group_idx, {})[block.block_id] = block
    for group_idx, manager in enumerate(self.kv_cache_manager.coordinator.single_type_managers):
        if group_idx >= len(row_block_ids):
            continue
        parent_blocks = list(manager.req_to_blocks.get(parent_id, ()))
        parent_by_id = {blk.block_id: blk for blk in parent_blocks}
        owned_by_id = by_group.get(group_idx, {})
        new_ids = list(row_block_ids[group_idx])
        new_id_set = set(new_ids)
        new_blocks = []
        for block_id in new_ids:
            block = owned_by_id.get(block_id) or parent_by_id.get(block_id)
            if block is not None:
                new_blocks.append(block)
        if len(new_blocks) != len(new_ids):
            continue
        to_free = [
            blk for blk in parent_blocks
            if blk.block_id not in new_id_set
            and blk != getattr(manager, "_null_block", None)
        ]
        if to_free:
            manager.block_pool.free_blocks(reversed(to_free))
        manager.req_to_blocks[parent_id] = new_blocks
        manager.num_cached_block[parent_id] = min(
            manager.num_cached_block.get(parent_id, len(new_blocks)),
            len(new_blocks),
        )
        if hasattr(manager, "_allocated_block_reqs"):
            manager._allocated_block_reqs.add(parent_id)

def _lumo_fb_ir_free_owned(self, row_ids):
    owned_by_row = getattr(self, "_lumo_fb_ir_owned_blocks", {})
    row_block_ids_by_row = getattr(self, "_lumo_fb_ir_row_block_ids", {})
    for row_id in row_ids:
        row_block_ids_by_row.pop(row_id, None)
        owned = owned_by_row.pop(row_id, [])
        by_group = {}
        for group_idx, block in owned:
            by_group.setdefault(group_idx, []).append(block)
        for group_idx, blocks in by_group.items():
            try:
                manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]
                manager.block_pool.free_blocks(reversed(blocks))
            except Exception:
                pass

def _lumo_fb_ir_accept_from_generated(tokens):
    valid = []
    for tok in list(tokens or []):
        try:
            if int(tok) == -1:
                break
            valid.append(int(tok))
        except Exception:
            break
    return max(0, len(valid) - 1)

def _lumo_fb_ir_promote_manager_state(self, req_id, accepted_drafts):
    if not _lumo_fb_ir_kernel_rows_enabled():
        return
    req = self.requests.get(req_id)
    if req is None:
        return
    accepted_drafts = int(accepted_drafts)
    _seen = getattr(self, "_lumo_fb_kernel_seen_sample", None)
    if _seen is None:
        _seen = set()
        self._lumo_fb_kernel_seen_sample = _seen
    _first_sample = req_id not in _seen
    _seen.add(req_id)
    moved = []
    for group_idx, manager in enumerate(self.kv_cache_manager.coordinator.single_type_managers):
        if getattr(manager, "mamba_cache_mode", None) != "align":
            continue
        blocks = list(manager.req_to_blocks.get(req_id, ()))
        if not blocks:
            continue
        block_size = int(manager.block_size)
        num_sched = 0
        try:
            num_sched = int(getattr(self, "_lumo_fb_ir_last_sched_tokens", {}).get(req_id, 0))
        except Exception:
            num_sched = 0
        curr_idx = max(0, _lumo_fb_ir_cdiv(
            int(req.num_computed_tokens) + int(num_sched), block_size) - 1)
        # Closed-form kernel-row promotion:
        # - Logical verify columns 0..n are states after consuming 1..n+1
        #   tokens: the carried token plus the accepted draft prefix.
        # - Physical block-table column 0 is the read-only prefix, so logical
        #   column c is stored at physical offset c+1.
        # - Accepting a drafts consumes a+1 tokens, so promote offset a+1.
        # - The initial prompt-root sample is not a spec-verify row and is the
        #   only no-op.
        src_offset = 0 if _first_sample else accepted_drafts + 1
        src_idx = curr_idx + src_offset
        if src_idx >= len(blocks):
            continue
        blocks[curr_idx], blocks[src_idx] = blocks[src_idx], blocks[curr_idx]
        manager.req_to_blocks[req_id] = blocks
        moved.append({
            "group": int(group_idx),
            "curr_idx": int(curr_idx),
            "src_idx": int(src_idx),
            "curr_block": int(blocks[curr_idx].block_id),
        })
    if moved:
        _lumo_fb_ir_sched_debug({
            "event": "kernel_promote_manager_state",
            "rid": req_id,
            "accepted_drafts": int(accepted_drafts),
            "moves": moved,
        })

def _lumo_fb_ir_update_draft_token_ids(self, draft_token_ids):
    if not _lumo_fb_ir_enabled():
        return _lumo_fb_ir_prev_update_draft(self, draft_token_ids)
    for req_id, spec_token_ids in zip(draft_token_ids.req_ids, draft_token_ids.draft_token_ids):
        request = self.requests.get(req_id)
        if request is None or request.is_finished():
            continue
        if request.is_prefill_chunk:
            request.spec_token_ids = []
            request._lumo_fb_internal_paths = None
            continue
        toks = list(spec_token_ids)
        _fb_ir_default_k = int(_lumo_fb_ir_os.environ.get("LUMO_FB_K", "2"))
        _fb_ir_default_depth = (len(toks) // 2
                                if _fb_ir_default_k >= 2 and len(toks) % 2 == 0
                                else len(toks))
        active_depth, requested_k = _lumo_fb_ir_read_control(_fb_ir_default_depth)
        if requested_k >= 2 and len(toks) == 2 * active_depth:
            paths = [list(toks[:active_depth]), list(toks[active_depth:2 * active_depth])]
            if _lumo_fb_ir_os.environ.get("LUMO_FB_DUP_PATH1") == "1":
                paths[1] = list(paths[0])
            request._lumo_fb_internal_paths = paths
            request.spec_token_ids = list(paths[0])
        else:
            if _lumo_fb_ir_os.environ.get("LUMO_FB_ASSERT_WIDTH") == "1" and len(toks) != active_depth:
                raise RuntimeError(
                    f"LUMO_FB scheduler width mismatch: draft_len={len(toks)} active_depth={active_depth} active_k={requested_k}")
            request._lumo_fb_internal_paths = None
            request.spec_token_ids = toks[:active_depth]

def _lumo_fb_ir_schedule(self):
    if not _lumo_fb_ir_enabled():
        return _lumo_fb_ir_prev_schedule(self)
    _lumo_fb_sched_base_t0 = _lumo_fb_ir_time.perf_counter_ns()
    out = _lumo_fb_orig_schedule(self)
    _lumo_fb_sched_base_us = int((_lumo_fb_ir_time.perf_counter_ns() - _lumo_fb_sched_base_t0) // 1000)
    _lumo_fb_sched_t0 = _lumo_fb_ir_time.perf_counter_ns()
    rows_by_parent = {}
    copies = []
    state_fork_us = 0
    self._lumo_fb_ir_last_sched_tokens = dict(out.num_scheduled_tokens)
    for parent_id in list(out.num_scheduled_tokens.keys()):
        parent = self.requests.get(parent_id)
        paths = getattr(parent, "_lumo_fb_internal_paths", None) if parent is not None else None
        if not paths or len(paths) < 2:
            continue
        if parent_id not in out.scheduled_spec_decode_tokens:
            parent._lumo_fb_internal_paths = None
            continue
        num_sched = int(out.num_scheduled_tokens[parent_id])
        row_id = _lumo_fb_ir_row_id(parent_id, 1)
        _lumo_fb_fork_t0 = _lumo_fb_ir_time.perf_counter_ns()
        block_ids, row_copies = _lumo_fb_ir_alloc_blocks(self, parent, row_id, num_sched)
        _lumo_fb_ir_prepare_parent_kernel_blocks(self, parent, num_sched)
        state_fork_us += int((_lumo_fb_ir_time.perf_counter_ns() - _lumo_fb_fork_t0) // 1000)
        copies.extend(row_copies)
        rows_by_parent[parent_id] = {
            "paths": [list(p) for p in paths[:2]],
            "rows": [{
                "rid": row_id,
                "path_idx": 1,
                "draft": list(paths[1]),
                "block_ids": block_ids,
            }],
        }
        parent._lumo_fb_internal_paths = None
    scheduler_us = int((_lumo_fb_ir_time.perf_counter_ns() - _lumo_fb_sched_t0) // 1000)
    kv_copies = sum(1 for item in copies if not (len(item) >= 3 and item[0] == "mamba"))
    mamba_copies = sum(1 for item in copies if len(item) >= 3 and item[0] == "mamba")
    out.lumo_fb_scheduler_us = scheduler_us
    out.lumo_fb_base_scheduler_us = _lumo_fb_sched_base_us
    out.lumo_fb_state_fork_us = state_fork_us
    out.lumo_fb_kv_blocks_copied = kv_copies
    out.lumo_fb_mamba_blocks_copied = mamba_copies
    out.lumo_fb_internal_row_count = sum(len(bundle.get("rows", [])) for bundle in rows_by_parent.values())
    _lumo_fb_ir_sched_debug({
        "event": "schedule",
        "fb_base_scheduler_us": _lumo_fb_sched_base_us,
        "fb_scheduler_us": scheduler_us,
        "fb_state_fork_us": state_fork_us,
        "fb_kv_blocks_copied": kv_copies,
        "fb_mamba_blocks_copied": mamba_copies,
        "fb_internal_row_count": out.lumo_fb_internal_row_count,
        "fb_parent_count": len(rows_by_parent),
    })
    if rows_by_parent:
        out.lumo_fb_internal_rows = rows_by_parent
        existing = list(getattr(out, "lumo_fb_block_copies", []) or [])
        out.lumo_fb_block_copies = existing + copies
    return out

def _lumo_fb_ir_update_from_output(self, scheduler_output, model_runner_output):
    if not _lumo_fb_ir_enabled():
        return _lumo_fb_ir_prev_update_output(self, scheduler_output, model_runner_output)
    rows_by_parent = getattr(scheduler_output, "lumo_fb_internal_rows", {}) or {}
    internal_ids = []
    for bundle in rows_by_parent.values():
        for row in bundle.get("rows", []):
            internal_ids.append(row.get("rid"))
    internal_ids = [rid for rid in internal_ids if rid]
    if internal_ids:
        winners = getattr(scheduler_output, "lumo_fb_internal_winners", {}) or {}
        winner_ids = {
            data.get("winner_rid")
            for data in winners.values()
            if isinstance(data, dict) and data.get("winner_rid")
        }
        for parent_id, data in winners.items():
            if isinstance(data, dict) and data.get("winner_rid") in internal_ids:
                _lumo_fb_ir_promote_manager_state(
                    self, data.get("winner_rid"), int(data.get("accepted", 0)))
                _lumo_fb_ir_transfer_owned_to_parent(self, parent_id, data.get("winner_rid"))
            elif isinstance(data, dict) and data.get("winner_rid") == parent_id:
                _lumo_fb_ir_promote_manager_state(
                    self, parent_id, int(data.get("accepted", 0)))
        for rid in internal_ids:
            if rid in scheduler_output.num_scheduled_tokens:
                scheduler_output.total_num_scheduled_tokens -= scheduler_output.num_scheduled_tokens.pop(rid)
            scheduler_output.scheduled_spec_decode_tokens.pop(rid, None)
            scheduler_output.finished_req_ids.discard(rid)
        if hasattr(model_runner_output, "req_ids"):
            keep = [i for i, rid in enumerate(model_runner_output.req_ids) if rid not in internal_ids]
            if len(keep) != len(model_runner_output.req_ids):
                model_runner_output.req_ids = [model_runner_output.req_ids[i] for i in keep]
                model_runner_output.sampled_token_ids = [model_runner_output.sampled_token_ids[i] for i in keep]
                model_runner_output.req_id_to_index = {rid: i for i, rid in enumerate(model_runner_output.req_ids)}
        _lumo_fb_ir_free_owned(self, [rid for rid in internal_ids if rid not in winner_ids])
    elif _lumo_fb_ir_kernel_rows_enabled() and hasattr(model_runner_output, "req_ids"):
        try:
            for req_id, toks in zip(model_runner_output.req_ids, model_runner_output.sampled_token_ids):
                if req_id in scheduler_output.num_scheduled_tokens:
                    _lumo_fb_ir_promote_manager_state(
                        self, req_id, _lumo_fb_ir_accept_from_generated(toks))
        except Exception as e:
            _lumo_fb_ir_sched_debug({
                "event": "kernel_promote_manager_state_error",
                "error": repr(e),
            })
    return _lumo_fb_orig_update_output(self, scheduler_output, model_runner_output)

Scheduler.update_draft_token_ids = _lumo_fb_ir_update_draft_token_ids
Scheduler.schedule = _lumo_fb_ir_schedule
Scheduler.update_from_output = _lumo_fb_ir_update_from_output
"""
    sch.write_text(text + patch)
    import py_compile
    py_compile.compile(str(sch), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b internal-row scheduler patch')

gm = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = gm.read_text()
sentinel = '# LUMO_FB_INTERNAL_ROWS_RUNNER'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b internal-row runner patch already present')
else:
    patch = r"""

# LUMO_FB_INTERNAL_ROWS_RUNNER: materialize scheduler-attached F_b path rows
# inside the GPU runner for one target forward, then remove them before the
# scheduler observes outputs.  This is a Gate-2 duplicate-path isolation path:
# parent/path0 remains the committed row, while the second row tests whether
# batched GDN verification corrupts path0 under internal COW state blocks.
import os as _lumo_fb_ir_os
import json as _lumo_fb_ir_json
import time as _lumo_fb_ir_time
import torch as _lumo_fb_ir_torch
from vllm.v1.worker.gpu_input_batch import CachedRequestState as _LumoFBCachedRequestState

_lumo_fb_ir_prev_update_states_runner = GPUModelRunner._update_states
_lumo_fb_ir_prev_sample_tokens = GPUModelRunner.sample_tokens

def _lumo_fb_ir_runner_enabled():
    return (_lumo_fb_ir_os.environ.get("LUMO_FB_PATHS") == "1"
            and _lumo_fb_ir_os.environ.get("LUMO_FB_INTERNAL_ROWS") == "1")

def _lumo_fb_ir_is_row_id(req_id):
    return isinstance(req_id, str) and "::lumo_fb_ir::" in req_id

def _lumo_fb_ir_debug(event):
    if _lumo_fb_ir_os.environ.get("LUMO_FB_DEBUG") != "1":
        return
    try:
        global _LUMO_FB_IR_DBG_FH
        try:
            _LUMO_FB_IR_DBG_FH
        except NameError:
            _LUMO_FB_IR_DBG_FH = open("/logs/fb_internal_debug.jsonl", "a", buffering=1)
        event["ts"] = round(_lumo_fb_ir_time.time(), 4)
        _LUMO_FB_IR_DBG_FH.write(_lumo_fb_ir_json.dumps(event) + chr(10))
    except Exception:
        pass

def _lumo_fb_ir_state_block_ids(self, req_id):
    req_state = self.requests.get(req_id)
    state_idx = self.mamba_state_idx.get(req_id)
    if req_state is None or state_idx is None:
        return None
    try:
        out = []
        for gid in self._get_mamba_copy_bufs().mamba_group_ids:
            blocks = req_state.block_ids[gid]
            out.append(blocks[state_idx] if state_idx < len(blocks) else None)
        return out
    except Exception:
        return None

def _lumo_fb_ir_kernel_promote_state(self, req_id, accepted_drafts):
    if _lumo_fb_ir_os.environ.get("LUMO_FB_KERNEL_ROWS") != "1":
        return
    req_state = self.requests.get(req_id)
    curr_idx = self.mamba_state_idx.get(req_id)
    if req_state is None or curr_idx is None:
        return
    accepted_drafts = int(accepted_drafts)
    _seen = getattr(self, "_lumo_fb_kernel_seen_sample", None)
    if _seen is None:
        _seen = set()
        self._lumo_fb_kernel_seen_sample = _seen
    _first_sample = req_id not in _seen
    _seen.add(req_id)
    # Same closed-form rule as scheduler-side promotion. Full accept at
    # a=active_depth intentionally promotes offset active_depth+1.
    src_offset = 0 if _first_sample else accepted_drafts + 1
    if src_offset < 1:
        return
    moved = []
    try:
        all_groups = [list(group) for group in req_state.block_ids]
        for gid in self._get_mamba_copy_bufs().mamba_group_ids:
            blocks = all_groups[gid]
            src_idx = int(curr_idx) + src_offset
            if src_idx >= len(blocks):
                continue
            blocks[int(curr_idx)], blocks[src_idx] = blocks[src_idx], blocks[int(curr_idx)]
            all_groups[gid] = blocks
            moved.append({
                "group": int(gid),
                "curr_idx": int(curr_idx),
                "src_idx": int(src_idx),
                "curr_block": int(blocks[int(curr_idx)]),
            })
        if moved:
            req_state.block_ids = tuple(all_groups)
            idx = self.input_batch.req_id_to_index.get(req_id)
            if idx is not None:
                self.input_batch.block_table.clear_row(idx)
                self.input_batch.block_table.add_row(req_state.block_ids, idx)
            # The promoted SSM column is physical, but the conv kernel stores a
            # final speculative rolling state and uses num_accepted_tokens to
            # slice it back to the accepted prefix. Keep that logical offset in
            # sync for no-internal K=1 kernel-row steps.
            _lumo_fb_ir_set_accept_len(self, req_id, accepted_drafts + 1)
            _lumo_fb_ir_debug({
                "event": "kernel_promote_state",
                "rid": req_id,
                "accepted_drafts": int(accepted_drafts),
                "accepted_prefix_len": int(accepted_drafts) + 1,
                "moves": moved,
            })
    except Exception as e:
        _lumo_fb_ir_debug({
            "event": "kernel_promote_state_error",
            "rid": req_id,
            "accepted_drafts": int(accepted_drafts),
            "error": repr(e),
        })

def _lumo_fb_ir_accepted_from_tokens(tokens):
    valid = []
    for tok in list(tokens):
        try:
            if int(tok) == -1:
                break
            valid.append(int(tok))
        except Exception:
            break
    return max(0, len(valid) - 1)

def _lumo_fb_ir_set_accept_len(self, req_id, accept_len):
    idx = self.input_batch.req_id_to_index.get(req_id)
    if idx is None:
        return
    try:
        self.input_batch.num_accepted_tokens_cpu[idx] = int(accept_len)
        self.input_batch.num_accepted_tokens_cpu_tensor[idx] = int(accept_len)
        if hasattr(self, "num_accepted_tokens"):
            self.num_accepted_tokens.np[idx] = int(accept_len)
    except Exception:
        pass

def _lumo_fb_ir_mamba_curr_state_idx(self, req_state, num_scheduled_tokens):
    try:
        copy_bufs = self._get_mamba_copy_bufs()
        block_size = int(copy_bufs.mamba_spec.block_size)
        num_speculative_blocks = int(copy_bufs.mamba_spec.num_speculative_blocks)
        num_blocks = (
            (int(req_state.num_computed_tokens) + int(num_scheduled_tokens)
             + block_size - 1) // block_size
        ) + num_speculative_blocks
        return int(num_blocks - 1 - num_speculative_blocks)
    except Exception:
        return None

def _lumo_fb_ir_update_states_runner(self, scheduler_output):
    ret = _lumo_fb_ir_prev_update_states_runner(self, scheduler_output)
    if not _lumo_fb_ir_runner_enabled():
        return ret
    rows_by_parent = getattr(scheduler_output, "lumo_fb_internal_rows", {}) or {}
    if not rows_by_parent:
        return ret
    _lumo_fb_materialize_t0 = _lumo_fb_ir_time.perf_counter_ns()
    active = {}
    for parent_id, bundle in rows_by_parent.items():
        parent_state = self.requests.get(parent_id)
        parent_sched = scheduler_output.num_scheduled_tokens.get(parent_id)
        if parent_state is None or parent_sched is None:
            continue
        for row in bundle.get("rows", []):
            row_id = row["rid"]
            if row_id in self.requests:
                continue
            req_state = _LumoFBCachedRequestState(
                req_id=row_id,
                prompt_token_ids=(None if parent_state.prompt_token_ids is None else list(parent_state.prompt_token_ids)),
                prompt_embeds=parent_state.prompt_embeds,
                mm_features=list(parent_state.mm_features),
                sampling_params=parent_state.sampling_params,
                pooling_params=parent_state.pooling_params,
                generator=parent_state.generator,
                block_ids=tuple([list(group) for group in row["block_ids"]]),
                num_computed_tokens=parent_state.num_computed_tokens,
                output_token_ids=list(parent_state.output_token_ids),
                mrope_positions=parent_state.mrope_positions,
                mrope_position_delta=parent_state.mrope_position_delta,
                xdrope_positions=parent_state.xdrope_positions,
                lora_request=parent_state.lora_request,
            )
            self.requests[row_id] = req_state
            scheduler_output.num_scheduled_tokens[row_id] = int(parent_sched)
            scheduler_output.total_num_scheduled_tokens += int(parent_sched)
            scheduler_output.scheduled_spec_decode_tokens[row_id] = list(row["draft"])
            self.input_batch.add_request(req_state)
            self.input_batch.update_req_spec_token_ids(req_state, scheduler_output.scheduled_spec_decode_tokens)
            row_state_idx = _lumo_fb_ir_mamba_curr_state_idx(
                self, req_state, parent_sched)
            if row_state_idx is None:
                row_state_idx = self.mamba_state_idx.get(parent_id)
            if row_state_idx is not None:
                self.mamba_state_idx[row_id] = int(row_state_idx)
            # Internal rows are materialized after the base Mamba preprocess
            # copied num_accepted_tokens to GPU. Seed the row slot explicitly;
            # otherwise the batched GDN forward may read stale offset data for
            # the sibling row.
            _lumo_fb_ir_set_accept_len(self, row_id, 1)
            active[row_id] = parent_id
    if active:
        try:
            self.num_accepted_tokens.copy_to_gpu(len(self.input_batch.req_ids))
        except Exception:
            pass
        self.input_batch.refresh_metadata()
        self._lumo_fb_ir_active = active
        scheduler_output.lumo_fb_row_materialize_us = int(
            (_lumo_fb_ir_time.perf_counter_ns() - _lumo_fb_materialize_t0) // 1000)
        _lumo_fb_ir_debug({
            "event": "expanded",
            "fb_scheduler_us": getattr(scheduler_output, "lumo_fb_scheduler_us", None),
            "fb_state_fork_us": getattr(scheduler_output, "lumo_fb_state_fork_us", None),
            "fb_state_copy_us": getattr(scheduler_output, "lumo_fb_state_copy_us", None),
            "fb_state_copy_bytes": getattr(scheduler_output, "lumo_fb_state_copy_bytes", None),
            "fb_state_copy_detail": getattr(scheduler_output, "lumo_fb_state_copy_detail", None),
            "fb_kv_blocks_copied": getattr(scheduler_output, "lumo_fb_kv_blocks_copied", None),
            "fb_mamba_blocks_copied": getattr(scheduler_output, "lumo_fb_mamba_blocks_copied", None),
            "fb_row_materialize_us": getattr(scheduler_output, "lumo_fb_row_materialize_us", None),
            "rows": [{
                "rid": rid,
                "parent": parent,
                "idx": self.input_batch.req_id_to_index.get(rid),
                "parent_idx": self.input_batch.req_id_to_index.get(parent),
                "draft": scheduler_output.scheduled_spec_decode_tokens.get(rid),
                "mamba_idx": self.mamba_state_idx.get(rid),
                "parent_mamba_idx": self.mamba_state_idx.get(parent),
                "state_block_ids": _lumo_fb_ir_state_block_ids(self, rid),
                "parent_state_block_ids": _lumo_fb_ir_state_block_ids(self, parent),
            } for rid, parent in active.items()],
        })
    return ret

def _lumo_fb_ir_filter_drafts(self, internal_ids):
    try:
        req_ids = list(self._draft_token_req_ids or [])
        if not req_ids:
            return
        keep = [i for i, rid in enumerate(req_ids) if rid not in internal_ids]
        if len(keep) == len(req_ids):
            return
        self._draft_token_req_ids = [req_ids[i] for i in keep]
        toks = self._draft_token_ids
        if _lumo_fb_ir_torch.is_tensor(toks):
            idx = _lumo_fb_ir_torch.tensor(keep, dtype=_lumo_fb_ir_torch.long, device=toks.device)
            self._draft_token_ids = toks.index_select(0, idx)
            if hasattr(self, "draft_token_ids_cpu") and self.draft_token_ids_cpu is not None:
                try:
                    if self.draft_token_ids_event is not None:
                        self.draft_token_ids_event.synchronize()
                    cpu_idx = _lumo_fb_ir_torch.tensor(keep, dtype=_lumo_fb_ir_torch.long, device="cpu")
                    width = int(getattr(self, "_lumo_fb_draft_cpu_width", self.draft_token_ids_cpu.shape[1]))
                    self.draft_token_ids_cpu[:len(keep), :width].copy_(
                        self.draft_token_ids_cpu.index_select(0, cpu_idx)[:, :width])
                except Exception:
                    pass
        elif isinstance(toks, list):
            self._draft_token_ids = [toks[i] for i in keep]
    except Exception:
        pass

def _lumo_fb_ir_filter_model_output(model_output, internal_ids):
    if model_output is None or not hasattr(model_output, "req_ids"):
        return model_output
    req_ids = list(model_output.req_ids)
    keep = [i for i, rid in enumerate(req_ids) if rid not in internal_ids]
    if len(keep) == len(req_ids):
        return model_output
    model_output.req_ids = [req_ids[i] for i in keep]
    model_output.sampled_token_ids = [model_output.sampled_token_ids[i] for i in keep]
    model_output.req_id_to_index = {rid: i for i, rid in enumerate(model_output.req_ids)}
    if getattr(model_output, "logprobs", None) is not None:
        try:
            model_output.logprobs = [model_output.logprobs[i] for i in keep]
        except Exception:
            pass
    return model_output

def _lumo_fb_ir_prune_after_sample(self, scheduler_output, sampler_output,
                                   spec_decode_metadata=None,
                                   common_attn_metadata=None):
    active = getattr(self, "_lumo_fb_ir_active", None)
    if not (_lumo_fb_ir_runner_enabled() and active):
        return sampler_output, spec_decode_metadata, common_attn_metadata
    internal_ids = set(active.keys())
    req_ids = list(self.input_batch.req_ids)
    keep = [i for i, rid in enumerate(req_ids) if rid not in internal_ids]
    winners = {}
    try:
        raw = sampler_output.sampled_token_ids.detach().cpu().tolist()
        rows = []
        for i, rid in enumerate(req_ids):
            if rid in internal_ids or rid in set(active.values()):
                toks = raw[i] if i < len(raw) else []
                rows.append({
                    "rid": rid,
                    "parent": active.get(rid, rid),
                    "is_internal": rid in internal_ids,
                    "sampled": list(toks),
                    "accepted": max(0, len([t for t in toks if int(t) != -1]) - 1),
                    "mamba_idx": self.mamba_state_idx.get(rid),
                    "state_block_ids": _lumo_fb_ir_state_block_ids(self, rid),
                })
        by_parent = {}
        for i, rid in enumerate(req_ids):
            if rid in internal_ids:
                parent = active.get(rid)
            elif rid in set(active.values()):
                parent = rid
            else:
                continue
            toks = raw[i] if i < len(raw) else []
            valid = []
            for tok in toks:
                if int(tok) == -1:
                    break
                valid.append(int(tok))
            by_parent.setdefault(parent, []).append((rid, valid, max(0, len(valid) - 1)))
        for parent, row_data in by_parent.items():
            row_data.sort(key=lambda item: (item[2], 0 if item[0] == parent else -1), reverse=True)
            winner_rid, winner_tokens, winner_acc = row_data[0]
            parent_idx = req_ids.index(parent) if parent in req_ids else None
            winner_idx = req_ids.index(winner_rid) if winner_rid in req_ids else None
            if parent_idx is not None and winner_idx is not None and winner_idx != parent_idx:
                sampler_output.sampled_token_ids[parent_idx].copy_(sampler_output.sampled_token_ids[winner_idx])
                try:
                    parent_state = self.requests.get(parent)
                    winner_state = self.requests.get(winner_rid)
                    if parent_state is not None and winner_state is not None:
                        parent_state.block_ids = tuple([list(group) for group in winner_state.block_ids])
                        pidx = self.input_batch.req_id_to_index.get(parent)
                        if pidx is not None:
                            self.input_batch.block_table.clear_row(pidx)
                            self.input_batch.block_table.add_row(parent_state.block_ids, pidx)
                        if winner_rid in self.mamba_state_idx:
                            self.mamba_state_idx[parent] = self.mamba_state_idx[winner_rid]
                except Exception:
                    pass
            # Keep the runner's cached block table in sync with the scheduler's
            # persistent block manager. These are separate state stores; both
            # must promote the same accepted draft column.
            _lumo_fb_ir_kernel_promote_state(self, parent, winner_acc)
            winners[parent] = {
                "winner_rid": winner_rid,
                "winner_idx": 0 if winner_rid == parent else 1,
                "accepted": int(winner_acc),
                "accept_lens": [
                    int(acc) for rid, toks, acc in sorted(
                        row_data, key=lambda item: 0 if item[0] == parent else 1)
                ],
            }
        if rows:
            _lumo_fb_ir_debug({
                "event": "sampled",
                "fb_scheduler_us": getattr(scheduler_output, "lumo_fb_scheduler_us", None),
                "fb_state_fork_us": getattr(scheduler_output, "lumo_fb_state_fork_us", None),
                "fb_state_copy_us": getattr(scheduler_output, "lumo_fb_state_copy_us", None),
                "fb_state_copy_bytes": getattr(scheduler_output, "lumo_fb_state_copy_bytes", None),
                "fb_state_copy_detail": getattr(scheduler_output, "lumo_fb_state_copy_detail", None),
                "fb_kv_blocks_copied": getattr(scheduler_output, "lumo_fb_kv_blocks_copied", None),
                "fb_mamba_blocks_copied": getattr(scheduler_output, "lumo_fb_mamba_blocks_copied", None),
                "fb_row_materialize_us": getattr(scheduler_output, "lumo_fb_row_materialize_us", None),
                "rows": rows,
                "winners": winners,
            })
    except Exception:
        pass
    if winners:
        scheduler_output.lumo_fb_internal_winners = winners
        # The same sampled parent row may subsequently flow through the
        # no-active K=1 promotion hook after internal rows have been pruned.
        # Winner collapse already promoted the parent once; a second promotion
        # swaps from the newly promoted column and corrupts the next state.
        self._lumo_fb_ir_skip_noactive_promote_once = set(winners.keys())
    if len(keep) != len(req_ids):
        try:
            idx = _lumo_fb_ir_torch.tensor(keep, dtype=_lumo_fb_ir_torch.long,
                                           device=sampler_output.sampled_token_ids.device)
            sampler_output.sampled_token_ids = sampler_output.sampled_token_ids.index_select(0, idx)
            if sampler_output.logprobs_tensors is not None:
                sampler_output.logprobs_tensors = None
        except Exception:
            pass
    for rid in internal_ids:
        if rid in scheduler_output.num_scheduled_tokens:
            scheduler_output.total_num_scheduled_tokens -= scheduler_output.num_scheduled_tokens.pop(rid)
        scheduler_output.scheduled_spec_decode_tokens.pop(rid, None)
        scheduler_output.finished_req_ids.discard(rid)
    if spec_decode_metadata is not None:
        try:
            parent_drafts = int(spec_decode_metadata.num_draft_tokens[0])
            parent_sampled = parent_drafts + 1
            spec_decode_metadata.draft_token_ids = spec_decode_metadata.draft_token_ids[:parent_drafts]
            spec_decode_metadata.num_draft_tokens = spec_decode_metadata.num_draft_tokens[:1]
            spec_decode_metadata.cu_num_draft_tokens = spec_decode_metadata.cu_num_draft_tokens[:1]
            spec_decode_metadata.cu_num_sampled_tokens = spec_decode_metadata.cu_num_sampled_tokens[:1]
            spec_decode_metadata.target_logits_indices = spec_decode_metadata.target_logits_indices[:parent_drafts]
            spec_decode_metadata.bonus_logits_indices = spec_decode_metadata.bonus_logits_indices[:1]
            spec_decode_metadata.logits_indices = spec_decode_metadata.logits_indices[:parent_sampled]
        except Exception:
            pass
    if common_attn_metadata is not None:
        try:
            parent_query_tokens = int(common_attn_metadata.query_start_loc_cpu[1].item())
            common_attn_metadata.query_start_loc = common_attn_metadata.query_start_loc[:2]
            common_attn_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:2]
            common_attn_metadata.seq_lens = common_attn_metadata.seq_lens[:1]
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu = common_attn_metadata._seq_lens_cpu[:1]
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu = common_attn_metadata._num_computed_tokens_cpu[:1]
            if common_attn_metadata.dcp_local_seq_lens is not None:
                common_attn_metadata.dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens[:1]
            if common_attn_metadata.dcp_local_seq_lens_cpu is not None:
                common_attn_metadata.dcp_local_seq_lens_cpu = common_attn_metadata.dcp_local_seq_lens_cpu[:1]
            if common_attn_metadata.is_prefilling is not None:
                common_attn_metadata.is_prefilling = common_attn_metadata.is_prefilling[:1]
            common_attn_metadata.num_reqs = 1
            common_attn_metadata.num_actual_tokens = parent_query_tokens
            common_attn_metadata.max_query_len = parent_query_tokens
            common_attn_metadata.block_table_tensor = common_attn_metadata.block_table_tensor[:1]
            common_attn_metadata.slot_mapping = common_attn_metadata.slot_mapping[:parent_query_tokens]
        except Exception:
            pass
    _lumo_fb_ir_cleanup_rows(self, internal_ids)
    self._lumo_fb_ir_active = {}
    return sampler_output, spec_decode_metadata, common_attn_metadata

def _lumo_fb_ir_cleanup_rows(self, internal_ids):
    for rid in internal_ids:
        try:
            self.input_batch.remove_request(rid)
        except Exception:
            pass
        self.requests.pop(rid, None)
        self.mamba_state_idx.pop(rid, None)
    try:
        self.input_batch.condense()
        self.input_batch.refresh_metadata()
    except Exception:
        pass

def _lumo_fb_ir_sample_tokens(self, grammar_output):
    active = getattr(self, "_lumo_fb_ir_active", None)
    if _lumo_fb_ir_runner_enabled() and active:
        try:
            from vllm.v1.sample import rejection_sampler as _lumo_fb_ir_rs
            req_ids = list(self.input_batch.req_ids)
            req_index = {rid: i for i, rid in enumerate(req_ids)}
            pairs = []
            for row_id, parent_id in active.items():
                if parent_id in req_index and row_id in req_index:
                    pairs.append((req_index[parent_id], req_index[row_id]))
            if hasattr(_lumo_fb_ir_rs, "_lumo_fb_set_shared_root_pairs"):
                _lumo_fb_ir_rs._lumo_fb_set_shared_root_pairs(pairs or None)
        except Exception:
            pass
    output = _lumo_fb_ir_prev_sample_tokens(self, grammar_output)
    try:
        from vllm.v1.sample import rejection_sampler as _lumo_fb_ir_rs
        if hasattr(_lumo_fb_ir_rs, "_lumo_fb_set_shared_root_pairs"):
            _lumo_fb_ir_rs._lumo_fb_set_shared_root_pairs(None)
    except Exception:
        pass
    if not _lumo_fb_ir_runner_enabled():
        return output
    active = getattr(self, "_lumo_fb_ir_active", None)
    if not active:
        if _lumo_fb_ir_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1":
            try:
                model_output = getattr(output, "model_runner_output", output)
                raw_req_ids = list(getattr(model_output, "req_ids", []) or [])
                raw_samples = list(getattr(model_output, "sampled_token_ids", []) or [])
                # Keep the runner's cached block table in sync with the
                # scheduler-side manager promotion for K=1/no-internal events.
                skip_once = getattr(
                    self, "_lumo_fb_ir_skip_noactive_promote_once", set())
                for rid, toks in zip(raw_req_ids, raw_samples):
                    if rid in skip_once:
                        skip_once.discard(rid)
                        _lumo_fb_ir_debug({
                            "event": "kernel_noactive_promote_skipped",
                            "rid": rid,
                            "sampled": [int(t) for t in list(toks)[:8] if int(t) != -1],
                            "accepted": int(_lumo_fb_ir_accepted_from_tokens(toks)),
                        })
                        continue
                    _lumo_fb_ir_debug({
                        "event": "kernel_noactive_sample",
                        "rid": rid,
                        "sampled": [int(t) for t in list(toks)[:8] if int(t) != -1],
                        "accepted": int(_lumo_fb_ir_accepted_from_tokens(toks)),
                    })
                    _lumo_fb_ir_kernel_promote_state(
                        self, rid, _lumo_fb_ir_accepted_from_tokens(toks))
                self.input_batch.refresh_metadata()
            except Exception as e:
                _lumo_fb_ir_debug({
                    "event": "kernel_promote_noactive_error",
                    "error": repr(e),
                })
        return output
    self._lumo_fb_ir_active = {}
    internal_ids = set(active.keys())
    model_output = getattr(output, "model_runner_output", output)
    try:
        raw_req_ids = list(getattr(model_output, "req_ids", []) or [])
        raw_samples = list(getattr(model_output, "sampled_token_ids", []) or [])
        rows = []
        for rid, toks in zip(raw_req_ids, raw_samples):
            if rid in internal_ids or rid in set(active.values()):
                rows.append({
                    "rid": rid,
                    "parent": active.get(rid, rid),
                    "is_internal": rid in internal_ids,
                    "sampled": list(toks),
                    "accepted": max(0, len(toks) - 1),
                })
        if rows:
            _lumo_fb_ir_debug({"event": "sampled", "rows": rows})
    except Exception:
        pass
    _lumo_fb_ir_filter_drafts(self, internal_ids)
    filtered = _lumo_fb_ir_filter_model_output(model_output, internal_ids)
    if hasattr(output, "model_runner_output"):
        output.model_runner_output = filtered
    else:
        output = filtered
    _lumo_fb_ir_cleanup_rows(self, internal_ids)
    return output

GPUModelRunner._update_states = _lumo_fb_ir_update_states_runner
GPUModelRunner.sample_tokens = _lumo_fb_ir_sample_tokens
"""
    gm.write_text(text + patch)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b internal-row runner patch')

text = gm.read_text()
sentinel = '# LUMO_FB_INTERNAL_ROWS_PRE_DRAFT_CLEANUP'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b internal-row pre-draft cleanup already present')
else:
    old = nl.join([
        '        self._update_states_after_model_execute(',
        '            sampler_output.sampled_token_ids, scheduler_output',
        '        )',
        '        self.p2b_debug_exporter.export_state_snapshots(runner=self)',
    ])
    new = nl.join([
        '        self._update_states_after_model_execute(',
        '            sampler_output.sampled_token_ids, scheduler_output',
        '        )',
        '        # LUMO_FB_INTERNAL_ROWS_PRE_DRAFT_CLEANUP: internal F_b verifier',
        '        # rows must be removed before proposing the next draft, otherwise',
        '        # the ordinary EAGLE proposer sees batch_size=2 and falls back to',
        '        # width-3 drafting on alternating steps.',
        '        if "_lumo_fb_ir_prune_after_sample" in globals():',
        '            sampler_output = _lumo_fb_ir_prune_after_sample(',
        '                self, scheduler_output, sampler_output)',
        '        self.p2b_debug_exporter.export_state_snapshots(runner=self)',
    ])
    if old not in text:
        raise RuntimeError('F_b internal-row pre-draft cleanup anchor not found')
    text = text.replace(old, new, 1)
    gm.write_text(text)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b internal-row pre-draft cleanup patch')

text = gm.read_text()
sentinel = '# LUMO_FB_INTERNAL_ROWS_SPEC_META_PRUNE'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b internal-row spec metadata prune already present')
else:
    old = nl.join([
        '        if "_lumo_fb_ir_prune_after_sample" in globals():',
        '            sampler_output = _lumo_fb_ir_prune_after_sample(',
        '                self, scheduler_output, sampler_output)',
    ])
    new = nl.join([
        '        if "_lumo_fb_ir_prune_after_sample" in globals():',
        '            # LUMO_FB_INTERNAL_ROWS_SPEC_META_PRUNE: once internal rows',
        '            # are removed from the persistent batch, prune spec metadata',
        '            # too so the next EAGLE proposal keeps the parent-only F_b',
        '            # width-6 path instead of falling back on batch_size=2.',
        '            sampler_output, spec_decode_metadata, spec_decode_common_attn_metadata = _lumo_fb_ir_prune_after_sample(',
        '                self, scheduler_output, sampler_output, spec_decode_metadata,',
        '                spec_decode_common_attn_metadata)',
    ])
    if old not in text:
        raise RuntimeError('F_b internal-row spec metadata prune anchor not found')
    text = text.replace(old, new, 1)
    gm.write_text(text)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b internal-row spec metadata prune patch')

text = gm.read_text()
sentinel = '# LUMO_FB_ACTUAL_WIDTH_ASSERT'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b actual compute-width assert already present')
else:
    old = nl.join([
        '                sample_hidden_states = hidden_states[logits_indices]',
        '                logits = self.model.compute_logits(sample_hidden_states)',
    ])
    inject = nl.join([
        '                sample_hidden_states = hidden_states[logits_indices]',
        '                logits = self.model.compute_logits(sample_hidden_states)',
        '                # LUMO_FB_ACTUAL_WIDTH_ASSERT: hot-plug search must prove',
        '                # that target forward and lm_head compute at the active',
        '                # draft width, not at the over-allocated launch width.',
        '                import os as _fbaw_os',
        '                if use_spec_decode and _fbaw_os.environ.get("LUMO_FB_PATHS") == "1":',
        '                    import json as _fbaw_json, time as _fbaw_time',
        '                    _fbaw_ctrl_depth = None',
        '                    _fbaw_ctrl_path = _fbaw_os.environ.get("LUMO_FB_CONTROL_FILE", "/logs/fb_control.json")',
        '                    try:',
        '                        if _fbaw_os.path.exists(_fbaw_ctrl_path):',
        '                            with open(_fbaw_ctrl_path) as _fbaw_cf:',
        '                                _fbaw_payload = _fbaw_json.load(_fbaw_cf)',
        '                            if _fbaw_payload.get("assert_depth") is not None:',
        '                                _fbaw_ctrl_depth = int(_fbaw_payload["assert_depth"])',
        '                            if _fbaw_payload.get("depth") is not None:',
        '                                _fbaw_ctrl_depth = int(_fbaw_payload["depth"]) if _fbaw_ctrl_depth is None else _fbaw_ctrl_depth',
        '                        elif _fbaw_os.environ.get("LUMO_FB_DEPTH"):',
        '                            _fbaw_ctrl_depth = int(_fbaw_os.environ["LUMO_FB_DEPTH"])',
        '                    except (OSError, ValueError, _fbaw_json.JSONDecodeError):',
        '                        _fbaw_ctrl_depth = None',
        '                    _fbaw_expected_per_req = (_fbaw_ctrl_depth + 1) if _fbaw_ctrl_depth is not None else None',
        '                    _fbaw_event = {',
        '                        "ts": round(_fbaw_time.time(), 4),',
        '                        "cudagraph_mode": str(cudagraph_mode),',
        '                        "batch_desc": str(batch_desc),',
        '                        "active_depth": _fbaw_ctrl_depth,',
        '                        "expected_tokens_per_req": _fbaw_expected_per_req,',
        '                        "num_reqs": int(num_reqs),',
        '                        "num_reqs_padded": int(num_reqs_padded),',
        '                        "num_scheduled_tokens": [int(x) for x in num_scheduled_tokens_np.tolist()],',
        '                        "max_query_len": int(max_num_scheduled_tokens),',
        '                        "target_unpadded_tokens": int(num_tokens_unpadded),',
        '                        "target_forward_tokens": int(num_tokens_padded),',
        '                        "logits_indices_len": int(len(logits_indices)),',
        '                        "lm_head_rows": int(logits.shape[0]) if logits is not None else None,',
        '                        "sample_hidden_rows": int(sample_hidden_states.shape[0]),',
        '                    }',
        '                    if _fbaw_os.environ.get("LUMO_FB_DEBUG") == "1":',
        '                        global _LUMO_FB_ACTUAL_WIDTH_FH',
        '                        try:',
        '                            _LUMO_FB_ACTUAL_WIDTH_FH',
        '                        except NameError:',
        '                            _LUMO_FB_ACTUAL_WIDTH_FH = open("/logs/fb_actual_width_debug.jsonl", "a", buffering=1)',
        '                        _LUMO_FB_ACTUAL_WIDTH_FH.write(_fbaw_json.dumps(_fbaw_event) + chr(10))',
        '                    if _fbaw_os.environ.get("LUMO_FB_ASSERT_ACTUAL_WIDTH") == "1" and _fbaw_expected_per_req is not None:',
        '                        _fbaw_expected_total = int(num_reqs) * int(_fbaw_expected_per_req)',
        '                        if (int(max_num_scheduled_tokens) != int(_fbaw_expected_per_req)',
        '                                or int(num_tokens_unpadded) != _fbaw_expected_total',
        '                                or int(num_tokens_padded) != _fbaw_expected_total',
        '                                or int(len(logits_indices)) != _fbaw_expected_total',
        '                                or int(logits.shape[0]) != _fbaw_expected_total):',
        '                            raise RuntimeError(f"LUMO_FB_ASSERT_ACTUAL_WIDTH failed: {_fbaw_event}")',
    ])
    if old not in text:
        raise RuntimeError('F_b actual compute-width common logits anchor not found')
    text = text.replace(old, inject, 1)
    gm.write_text(text)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b actual compute-width assert patch')
LUMOFBPATHS
'''


def _prelaunch_for(config: str, tree: bool = False, tree_debug: bool = False, fb: bool = False) -> str:
    full = _track_b_runtime_prelaunch_shell()
    if config == "D":
        base = full  # full T1+T2+T3+T4 stack
        if "T2_T4_COMPOSITE" not in base:
            raise RuntimeError("config D expects the full prelaunch shell")
    else:  # E -- KEEP prefix only (no suffix T-patches; MTP doesn't use them)
        idx = full.find(_KEEP_MARKER)
        if idx < 0:
            raise RuntimeError("forced tool_choice marker not found")
        base = full[: idx + len(_KEEP_MARKER)]
        if "T1_SESSION_SCOPING" in base:
            raise RuntimeError("config E shell must not contain T1")
    # Tree drafting only activates when decoder self-attn runs the TREE_ATTN
    # backend (target verify + draft). vLLM's selector has no tree logic and does
    # not honor VLLM_ATTENTION_BACKEND in 0.19.0, so we source-edit the selector
    # (config F only). Realized KV is auto/bf16, which TreeAttention supports.
    dbg = "export LUMO_TREE_DRAFT_DEBUG=1\n" if (tree and tree_debug) else ""
    tree_blocks = _TREE_ATTN_BLOCK + _MROPE_TREE_BLOCK + _TREE_REJECTION_BLOCK
    fb_k = os.environ.get("LUMO_FB_K", "1")
    fb_dup = "export LUMO_FB_DUP_PATH1=1\n" if os.environ.get("LUMO_FB_DUP_PATH1") == "1" else ""
    fb_no_shared = "export LUMO_FB_DISABLE_SHARED_ROOT=1\n" if os.environ.get("LUMO_FB_DISABLE_SHARED_ROOT") == "1" else ""
    fb_internal = "export LUMO_FB_INTERNAL_ROWS=1\n" if os.environ.get("LUMO_FB_INTERNAL_ROWS") == "1" else ""
    fb_kernel_rows = "export LUMO_FB_KERNEL_ROWS=1\n" if os.environ.get("LUMO_FB_KERNEL_ROWS") == "1" else ""
    fb_adaptive = "export LUMO_FB_ADAPTIVE=1\n" if os.environ.get("LUMO_FB_ADAPTIVE") == "1" else ""
    fb_batched = "export LUMO_FB_BATCHED_PROPOSER=1\n" if os.environ.get("LUMO_FB_BATCHED_PROPOSER") == "1" else ""
    fb_p1 = f"export LUMO_FB_ADAPTIVE_P1_MAX={os.environ['LUMO_FB_ADAPTIVE_P1_MAX']}\n" if os.environ.get("LUMO_FB_ADAPTIVE_P1_MAX") else ""
    fb_ratio = f"export LUMO_FB_ADAPTIVE_RATIO_MIN={os.environ['LUMO_FB_ADAPTIVE_RATIO_MIN']}\n" if os.environ.get("LUMO_FB_ADAPTIVE_RATIO_MIN") else ""
    fb_depth = f"export LUMO_FB_DEPTH={os.environ['LUMO_FB_DEPTH']}\n" if os.environ.get("LUMO_FB_DEPTH") else ""
    fb_control = os.environ.get("LUMO_FB_CONTROL_FILE", "/logs/fb_control.json")
    fb_env = f"export LUMO_FB_PATHS=1\nexport LUMO_FB_K={fb_k}\n{fb_depth}export LUMO_FB_CONTROL_FILE={fb_control}\nexport LUMO_FB_ASSERT_WIDTH=1\nexport LUMO_FB_ASSERT_ACTUAL_WIDTH=1\nexport LUMO_FB_DEBUG=1\n{fb_dup}{fb_no_shared}{fb_internal}{fb_kernel_rows}{fb_adaptive}{fb_batched}{fb_p1}{fb_ratio}" if fb else ""
    return dbg + fb_env + base + _SPEC_TRACE_BLOCK + (tree_blocks if tree else "") + (_FB_BLOCK if fb else "") + (_FB_KERNEL_ROWS_BLOCK if fb and os.environ.get("LUMO_FB_KERNEL_ROWS") == "1" else "")


def _apply_kv_cache_dtype(src: str, kv_cache_dtype: str | None) -> str:
    """Rewrite the bundle's realized KV cache dtype. The base bundles request
    fp8_e5m2, which ModelServer._initial_kv_cache_dtype rewrites to auto for the
    fp8 checkpoint (vLLM rejects fp8_e5m2 KV on fp8 checkpoints). fp8_e4m3 is the
    one FP8 KV dtype the fp8 checkpoint accepts (it carries e4m3 k/v scales), so it
    passes through to a realized FP8 KV cache instead of falling back to auto."""
    if kv_cache_dtype is None:
        return src
    old = "    kv_cache_dtype: fp8_e5m2"
    if src.count(old) != 1:
        raise RuntimeError("kv_cache_dtype anchor not unique in bundle")
    return src.replace(old, f"    kv_cache_dtype: {kv_cache_dtype}", 1)


def _apply_gpu_memory_utilization(src: str, value: str | None) -> str:
    if value is None:
        return src
    gpu_mem = float(value)
    if not (0.0 < gpu_mem <= 0.95):
        raise RuntimeError(f"invalid gpu memory utilization override: {value}")
    old = "    gpu_memory_utilization: 0.9"
    if src.count(old) != 1:
        raise RuntimeError("gpu_memory_utilization anchor not unique in bundle")
    return src.replace(old, f"    gpu_memory_utilization: {gpu_mem:.2f}", 1)


def _d_bundle(kv_cache_dtype: str | None = None) -> str:
    base = "/tmp/lumo-track-b-bundle-qwen36/bundle.yaml"
    if kv_cache_dtype is None:
        return base
    src = _apply_kv_cache_dtype(Path(base).read_text(), kv_cache_dtype)
    out = Path(f"/tmp/lumo-track-b-bundle-qwen36-kv{kv_cache_dtype}"); out.mkdir(exist_ok=True)
    (out / "bundle.yaml").write_text(src)
    return str(out / "bundle.yaml")


def _mtp_bundle(n: int, tree: str | None = None, kv_cache_dtype: str | None = None) -> str:
    src = Path("/tmp/lumo-track-b-bundle-qwen36-off/bundle.yaml").read_text()
    src = _apply_kv_cache_dtype(src, kv_cache_dtype)
    src = _apply_gpu_memory_utilization(
        src, os.environ.get("LUMO_GPU_MEMORY_UTILIZATION"))
    kvtag = "" if kv_cache_dtype is None else f"-kv{kv_cache_dtype}"
    tag = (f"mtp{n}" if tree is None else f"mtp{n}tree") + kvtag
    src = src.replace("bundle_id: 712fd011-4b16-4051-9e8c-875405b70f5b",
                      f"bundle_id: e0000000-{tag}-4000-9000-config-e-qwen36")
    # speculative_token_tree passes through load_tuned_config (the
    # spec_decode_fields_only allowlist is advisory, not enforced); we still
    # add it to the allowlist below for provenance. vLLM only supports REGULAR
    # trees (uniform children/level). For a TREE, num_speculative_tokens must be
    # the NODE COUNT (len(tree_choices)), not the depth -- the runner's draft
    # output buffer is sized by num_speculative_tokens and propose_tree emits one
    # draft per tree node (else: RuntimeError size a(depth) != b(nodes)).
    import ast as _ast
    n_spec = n if tree is None else len(_ast.literal_eval(tree))
    spec_block = f"  spec_decode:\n    method: qwen3_5_mtp\n    num_speculative_tokens: {n_spec}"
    if tree is not None:
        # single-quote the tree so YAML keeps it a literal string for vLLM's
        # ast.literal_eval; embedded single quotes are not expected in node tuples.
        spec_block += f"\n    speculative_token_tree: '{tree}'"
    src = src.replace("  spec_decode: {}", spec_block)
    if tree is not None:
        src = src.replace("    - num_speculative_tokens\n",
                          "    - num_speculative_tokens\n    - speculative_token_tree\n")
    out = Path(f"/tmp/lumo-track-b-bundle-qwen36-{tag}"); out.mkdir(exist_ok=True)
    (out / "bundle.yaml").write_text(src)
    return str(out / "bundle.yaml")


def _default_tree(n: int) -> str:
    """Config F's default REGULAR tree: top-2 at the root, each extended as a
    linear chain to depth n (= two parallel depth-n candidate chains seeded by
    the MTP head's top-2 first tokens). Small on purpose -- enough branching to
    test the hypothesis, not so much that tree-attn/verifier overhead dominates.
    n=3 -> [(0,),(1,),(0,0),(1,0),(0,0,0),(1,0,0)] (6-node budget vs E's 3)."""
    nodes = {(root,) + (0,) * level for level in range(n) for root in (0, 1)}
    return str(sorted(nodes, key=lambda t: (len(t), t)))


def main() -> int:
    ap = argparse.ArgumentParser()
    # Configs (each self-contained, like D's suffix stack): D = full T1-T4 suffix;
    # E = native MTP linear chain; F = native MTP + branching top-k tree (E's
    # prelaunch + the tree-attn selector source-edit + an MTP bundle carrying
    # speculative_token_tree).
    ap.add_argument("--config", choices=["D", "E", "F", "Fb"], required=True)
    ap.add_argument("--mtp", type=int, default=1, help="num_speculative_tokens (MTP depth) for config E/F")
    ap.add_argument("--tree", default=None,
                    help="config F only: override the speculative_token_tree literal "
                         "(default: _default_tree(--mtp)). Must be a REGULAR tree whose "
                         "max depth equals --mtp.")
    ap.add_argument("--tree-debug", action="store_true",
                    help="config F only: export LUMO_TREE_DRAFT_DEBUG=1 so propose_tree "
                         "logs per-level proposed draft tokens to /logs/tree_draft_debug.jsonl")
    ap.add_argument("--kv-cache-dtype", default=None,
                    choices=["auto", "fp8_e5m2", "fp8_e4m3"],
                    help="override realized KV cache dtype. fp8_e4m3 is the FP8 KV that "
                         "the fp8 checkpoint accepts (fp8_e5m2 is rejected -> auto). Default: "
                         "use the bundle's value (fp8_e5m2 -> auto for this checkpoint).")
    args = ap.parse_args()
    is_tree = args.config == "F"
    is_fb = args.config == "Fb"
    if args.tree is not None and not is_tree:
        ap.error("--tree is only valid with --config F")
    tree = (args.tree or _default_tree(args.mtp)) if is_tree else None
    if args.config == "D":
        bundle = _d_bundle(kv_cache_dtype=args.kv_cache_dtype)
    else:  # E, F, or Fb -- MTP bundle (F adds speculative_token_tree)
        bundle = _mtp_bundle(args.mtp, tree=tree, kv_cache_dtype=args.kv_cache_dtype)
    server = ModelServer(
        registry_path=REPO / "model_registry.yaml", port=9950,
        container_name="lumo-vllm-track-b-suffix",
        logs_root=Path("/tmp/lumo-l0c-fp8-cutlass-run30-logs"),
        triton_cache_root=Path("/tmp/lumo-l0c-fp8-cutlass-run30-triton"),
        state_root=Path("/tmp/lumo-l0c-fp8-cutlass-run30-state"),
        proxy_port=8088, ready_timeout_s=900,
        prelaunch_shell=_prelaunch_for(args.config, tree=is_tree, tree_debug=args.tree_debug, fb=is_fb),
    )
    server.load_tuned_config(bundle)
    server.start("qwen3.6-27b")
    tree_desc = f" tree={tree}" if is_tree else ""
    mtp_desc = args.mtp if args.config in ("E", "F", "Fb") else "-"
    kv_desc = args.kv_cache_dtype or "bundle-default"
    print(f"READY config={args.config} mtp={mtp_desc}{tree_desc} kv={kv_desc} bundle={bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
