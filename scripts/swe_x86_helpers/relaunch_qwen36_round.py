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

_MTP_DRAFT_TRACE_BLOCK = r'''
python3 - <<'LUMOMTPDRAFTTRACE'
from pathlib import Path
p = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py')
text = p.read_text()
sentinel = '# LUMO_MTP_DRAFT_TRACE'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] MTP draft trace patch already present')
else:
    patch = r"""

# LUMO_MTP_DRAFT_TRACE: optional native MTP draft-token trace. This is
# observational only and is used to replay exact E3 drafts into the F_b
# verifier-isolation experiment.
import os as _lumo_mtp_trace_os
import json as _lumo_mtp_trace_json
import time as _lumo_mtp_trace_time

_lumo_mtp_trace_orig_propose = EagleProposer.propose
_lumo_mtp_trace_idx = 0
_lumo_mtp_replay_cache = None
_lumo_mtp_replay_idx = 0

def _lumo_mtp_replay_next(device):
    global _lumo_mtp_replay_cache, _lumo_mtp_replay_idx
    path = _lumo_mtp_trace_os.environ.get("LUMO_FB_REPLAY_DRAFT_FILE")
    if not path:
        return None
    if _lumo_mtp_replay_cache is None:
        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = _lumo_mtp_trace_json.loads(line)
                if payload.get("event") == "mtp_draft":
                    rows.append(payload["draft"])
        _lumo_mtp_replay_cache = rows
        _lumo_mtp_replay_idx = 0
    if _lumo_mtp_replay_idx >= len(_lumo_mtp_replay_cache):
        raise RuntimeError(
            f"LUMO_FB_REPLAY_DRAFT_FILE exhausted at {_lumo_mtp_replay_idx}")
    draft = _lumo_mtp_replay_cache[_lumo_mtp_replay_idx]
    _lumo_mtp_replay_idx += 1
    return torch.tensor(draft, dtype=torch.int64, device=device), int(_lumo_mtp_replay_idx - 1)

def _lumo_mtp_trace_propose(self, target_token_ids, target_positions,
                            target_hidden_states, next_token_ids,
                            token_indices_to_sample, common_attn_metadata,
                            sampling_metadata, mm_embed_inputs=None,
                            num_rejected_tokens_gpu=None,
                            slot_mappings=None):
    global _lumo_mtp_trace_idx
    out = _lumo_mtp_trace_orig_propose(
        self, target_token_ids, target_positions, target_hidden_states,
        next_token_ids, token_indices_to_sample, common_attn_metadata,
        sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)
    replay = _lumo_mtp_replay_next(out.device)
    if replay is not None:
        replay_out, replay_idx = replay
        out = replay_out[:, :out.shape[1]].contiguous()
    else:
        replay_idx = None
    path = _lumo_mtp_trace_os.environ.get("LUMO_MTP_DRAFT_TRACE_FILE")
    if path:
        try:
            global _LUMO_MTP_DRAFT_TRACE_FH
            try:
                _LUMO_MTP_DRAFT_TRACE_FH
            except NameError:
                _LUMO_MTP_DRAFT_TRACE_FH = open(path, "a", buffering=1)
            _LUMO_MTP_DRAFT_TRACE_FH.write(_lumo_mtp_trace_json.dumps({
                "event": "mtp_draft",
                "idx": int(_lumo_mtp_trace_idx),
                "replay_idx": replay_idx,
                "ts": round(_lumo_mtp_trace_time.time(), 4),
                "draft": out.detach().cpu().tolist(),
            }) + chr(10))
            _lumo_mtp_trace_idx += 1
        except Exception:
            pass
    return out

EagleProposer.propose = _lumo_mtp_trace_propose
"""
    p.write_text(text + patch)
    import py_compile
    py_compile.compile(str(p), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied MTP draft trace patch')
LUMOMTPDRAFTTRACE
'''

_NO_STALE_FB_PATCHES_BLOCK = r'''
python3 - <<'LUMONOSTALEFB'
from pathlib import Path
root = Path('/usr/local/lib/python3.12/dist-packages/vllm')
hits = []
for path in root.rglob('*.py'):
    try:
        text = path.read_text(errors='ignore')
    except Exception:
        continue
    if 'LUMO_FB' in text:
        hits.append(str(path.relative_to(root)))
if hits:
    raise RuntimeError(
        'Non-Fb launch found stale F_b vLLM source patches: '
        + ', '.join(hits[:20]))
print('[TRACK-B-PRELAUNCH] no stale F_b vLLM source patches found')
LUMONOSTALEFB
'''

_QWEN36_FP8_CONFIG_FIX_BLOCK = r'''
python3 - <<'LUMOQ36FP8CFG'
from pathlib import Path
import json

cfg_path = Path('/models/qwen3.6-27b-fp8/config.json')
if not cfg_path.exists():
    print('[TRACK-B-PRELAUNCH] qwen3.6 fp8 config not present; skip quant metadata fix')
else:
    cfg = json.loads(cfg_path.read_text())
    text_cfg = cfg.get('text_config') or {}
    layer_types = list(text_cfg.get('layer_types') or [])
    modules_to_not_convert = [
        'lm_head',
        'model.language_model.embed_tokens',
        'mtp.fc',
    ]
    for idx, layer_type in enumerate(layer_types):
        if layer_type == 'linear_attention':
            base = f'model.language_model.layers.{idx}.linear_attn'
            modules_to_not_convert.extend([
                f'{base}.conv1d',
                f'{base}.in_proj_a',
                f'{base}.in_proj_b',
            ])
    vision_depth = int((cfg.get('vision_config') or {}).get('depth') or 0)
    for prefix in ('model.visual', 'visual'):
        for idx in range(vision_depth):
            base = f'{prefix}.blocks.{idx}'
            modules_to_not_convert.extend([
                f'{base}.attn.proj',
                f'{base}.attn.qkv',
                f'{base}.mlp.linear_fc1',
                f'{base}.mlp.linear_fc2',
            ])
        modules_to_not_convert.extend([
            f'{prefix}.merger.linear_fc1',
            f'{prefix}.merger.linear_fc2',
            f'{prefix}.patch_embed.proj',
            f'{prefix}.pos_embed',
        ])
    existing = cfg.get('quantization_config') or {}
    desired = {
        'quant_method': 'fp8',
        'activation_scheme': 'dynamic',
        'weight_per_tensor': False,
        'act_per_tensor': False,
        'weight_block_size': [128, 128],
        'modules_to_not_convert': sorted(set(modules_to_not_convert)),
    }
    if existing == desired:
        print('[TRACK-B-PRELAUNCH] qwen3.6 fp8 quant metadata already present')
    else:
        cfg['quantization_config'] = desired
        bak = cfg_path.with_suffix('.json.lumo_pre_fp8_fix.bak')
        if not bak.exists():
            bak.write_text(cfg_path.read_text())
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=False) + '\n')
        print('[TRACK-B-PRELAUNCH] injected/updated qwen3.6 fp8 block-quant metadata')
LUMOQ36FP8CFG
'''


_CAUSAL_CONV_CUDAGRAPH_ASSERT_FIX_BLOCK = r'''
python3 - <<'LUMOCCONVASSERT'
from pathlib import Path
import py_compile

p = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py')
text = p.read_text()
sentinel = '# LUMO_CAUSAL_CONV_CUDAGRAPH_ASSERT_FIX'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] causal_conv1d cudagraph assert fix already present')
else:
    old = """        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
"""
    new = """        # LUMO_CAUSAL_CONV_CUDAGRAPH_ASSERT_FIX: during full decode CUDA-graph
        # capture, batch is the capture size while num_cache_lines is the Mamba
        # state pool width. When conv_state_indices is provided, the valid batch
        # invariant is the index-table length, not num_cache_lines >= batch.
        if conv_state_indices is None:
            assert num_cache_lines >= batch
        else:
            assert batch == conv_state_indices.shape[0], (
                f"ERROR: conv_state_indices should have shape ({batch},*) but got {conv_state_indices.shape}"
            )
        assert weight.stride(1) == 1  # Need this
"""
    if old not in text:
        raise RuntimeError('causal_conv1d cudagraph assert anchor not found')
    text = text.replace(old, new, 1)
    p.write_text(text)
    py_compile.compile(str(p), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied causal_conv1d cudagraph assert fix')
LUMOCCONVASSERT
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
                if IS_SPEC_DECODING:
                    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                else:
                    i_t = 0
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
            if (_lumo_fb_kernel_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1"
                    or _lumo_fb_kernel_os.environ.get("LUMO_FA_UNIQUE_NODES") == "1"):
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
    if not HAS_INITIAL_STATE_INDICES:
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

    old = '''        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
'''
    new = '''        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if HAS_INITIAL_STATE_INDICES:
            # LUMO_FB_KERNEL_ROWS_CONV_PREFIX_WRITE: mirror the SSM kernel's
            # per-token state table. Each private write column stores the conv
            # state after exactly idx_token + 1 verified tokens, so promoting a
            # partial internal-row winner never carries rejected suffix state.
            _lumo_fb_write_col = idx_token
            if _lumo_fb_write_col < FB_WRITE_COLS:
                _lumo_fb_conv_state = tl.zeros((NP2_STATELEN, BLOCK_N), dtype=tl.float32)
                if KERNEL_WIDTH >= 2:
                    _lumo_fb_conv_state = tl.where(
                        idx_tokens[:, None] == 0, col0[None, :], _lumo_fb_conv_state)
                if KERNEL_WIDTH >= 3:
                    _lumo_fb_conv_state = tl.where(
                        idx_tokens[:, None] == 1, col1[None, :], _lumo_fb_conv_state)
                if KERNEL_WIDTH >= 4:
                    _lumo_fb_conv_state = tl.where(
                        idx_tokens[:, None] == 2, col2[None, :], _lumo_fb_conv_state)
                if KERNEL_WIDTH >= 5:
                    _lumo_fb_conv_state = tl.where(
                        idx_tokens[:, None] == 3, col3[None, :], _lumo_fb_conv_state)
                if KERNEL_WIDTH >= 6:
                    _lumo_fb_conv_state = tl.where(
                        idx_tokens[:, None] == 4, col4[None, :], _lumo_fb_conv_state)
                conv_states_offset = tl.load(
                    conv_state_indices_ptr
                    + idx_seq * stride_state_indices
                    + _lumo_fb_write_col
                ).to(tl.int64)
                conv_state_ptrs_target = (
                    conv_state_ptr
                    + (conv_states_offset * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)
                )[None, :] + (idx_tokens * stride_conv_state_tok)[:, None]
                _lumo_fb_mask = (
                    (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
                )
                tl.store(conv_state_ptrs_target, _lumo_fb_conv_state, _lumo_fb_mask)

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
'''
    if old not in text:
        raise RuntimeError('F_b kernel-row causal_conv prefix-write anchor not found')
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

text = gl.read_text()
sentinel = '# LUMO_FB_RECURRENT_INDEX_DEBUG'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b recurrent index debug already present')
else:
    old = '''        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
'''
    new = '''        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        # LUMO_FB_RECURRENT_INDEX_DEBUG: per-step recurrent read/write index
        # trace for F_b kernel rows.  The parent/path0 row is row 0 in the
        # active K batch; siblings must never change row 0's read index.
        import os as _lumo_fb_ridx_os
        if _lumo_fb_ridx_os.environ.get("LUMO_FB_GDN_DEBUG") == "1":
            try:
                import json as _lumo_fb_ridx_json
                import time as _lumo_fb_ridx_time
                def _lumo_fb_ridx_head(_value, _limit=24):
                    if _value is None:
                        return None
                    if hasattr(_value, "detach"):
                        _tensor = _value.detach()
                        return {
                            "shape": list(_tensor.shape),
                            "head": _tensor.reshape(-1).cpu().tolist()[:_limit],
                        }
                    return _value
                def _lumo_fb_ridx_state_summary(_cache, _idx):
                    if _idx is None or int(_idx) < 0:
                        return None
                    _idx = int(_idx)
                    _row = _cache[_idx].detach().float()
                    _flat = _row.reshape(-1)
                    return {
                        "idx": _idx,
                        "shape": list(_row.shape),
                        "sum": float(_flat.sum().item()),
                        "abs_sum": float(_flat.abs().sum().item()),
                        "head": _flat[:8].cpu().tolist(),
                    }
                _lumo_fb_ridx_state_pre = None
                if _lumo_fb_ridx_os.environ.get("LUMO_FB_RIDX_STATE_SUMMARY", "0") == "1":
                    _lumo_fb_ridx_layers = _lumo_fb_ridx_os.environ.get(
                        "LUMO_FB_RIDX_STATE_LAYERS", "0,1,2")
                    _lumo_fb_ridx_do_state = True
                    if _lumo_fb_ridx_layers:
                        _lumo_fb_ridx_do_state = str(self.layer_idx) in {
                            _x.strip() for _x in _lumo_fb_ridx_layers.split(",") if _x.strip()}
                    if _lumo_fb_ridx_do_state and spec_initial_state_indices_tensor is not None:
                        _lumo_fb_ridx_parent_read = int(
                            spec_initial_state_indices_tensor.detach().reshape(-1)[0].item())
                        _lumo_fb_ridx_parent_writes = None
                        if spec_state_indices_tensor is not None:
                            _lumo_fb_ridx_parent_writes = (
                                spec_state_indices_tensor.detach().reshape(
                                    spec_state_indices_tensor.shape[0], -1)[0].cpu().tolist())
                        _lumo_fb_ridx_state_pre = {
                            "parent_read_idx": _lumo_fb_ridx_parent_read,
                            "parent_write_indices": _lumo_fb_ridx_parent_writes,
                            "conv_parent_read": _lumo_fb_ridx_state_summary(
                                conv_state, _lumo_fb_ridx_parent_read),
                            "ssm_parent_read": _lumo_fb_ridx_state_summary(
                                ssm_state, _lumo_fb_ridx_parent_read),
                        }
                with open("/logs/fb_recurrent_index_debug.jsonl", "a", buffering=1) as _lumo_fb_ridx_fh:
                    _lumo_fb_ridx_fh.write(_lumo_fb_ridx_json.dumps({
                        "event": "gdn_recurrent_indices",
                        "phase": "pre",
                        "ts": round(_lumo_fb_ridx_time.time(), 4),
                        "layer_idx": int(self.layer_idx) if self.layer_idx is not None else None,
                        "prefix": str(getattr(self, "prefix", "")),
                        "num_actual_tokens": int(num_actual_tokens),
                        "num_spec_decodes": int(getattr(attn_metadata, "num_spec_decodes", 0)),
                        "num_decodes": int(getattr(attn_metadata, "num_decodes", 0)),
                        "num_prefills": int(getattr(attn_metadata, "num_prefills", 0)),
                        "spec_query_start_loc": _lumo_fb_ridx_head(spec_query_start_loc),
                        "spec_sequence_masks": _lumo_fb_ridx_head(spec_sequence_masks),
                        "spec_token_indx": _lumo_fb_ridx_head(spec_token_indx),
                        "spec_initial_state_indices_tensor": _lumo_fb_ridx_head(spec_initial_state_indices_tensor),
                        "spec_state_indices_tensor": _lumo_fb_ridx_head(spec_state_indices_tensor),
                        "spec_initial_state_slot_tensor": _lumo_fb_ridx_head(spec_initial_state_slot_tensor),
                        "spec_write_state_slot_tensor": _lumo_fb_ridx_head(spec_write_state_slot_tensor),
                        "num_accepted_tokens": _lumo_fb_ridx_head(num_accepted_tokens),
                        "state_pre": _lumo_fb_ridx_state_pre,
                    }) + chr(10))
            except Exception:
                pass

        mixed_qkv = mixed_qkv[:num_actual_tokens]
'''
    if old not in text:
        raise RuntimeError('F_b recurrent index debug anchor not found')
    text = text.replace(old, new, 1)
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b recurrent index debug patch')

text = gl.read_text()
sentinel = '# LUMO_FB_RECURRENT_STATE_POST_DEBUG'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b recurrent state post debug already present')
else:
    old = '''        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
'''
    new = '''        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

        # LUMO_FB_RECURRENT_STATE_POST_DEBUG: post-kernel write checksum for
        # path0's private recurrent write slots. The following step should read
        # the promoted value from these slots, not a sibling row's state.
        import os as _lumo_fb_ridx_post_os
        if (
            _lumo_fb_ridx_post_os.environ.get("LUMO_FB_GDN_DEBUG") == "1"
            and _lumo_fb_ridx_post_os.environ.get("LUMO_FB_RIDX_STATE_SUMMARY", "0") == "1"
            and spec_sequence_masks is not None
            and spec_state_indices_tensor is not None
        ):
            try:
                _lumo_fb_ridx_layers = _lumo_fb_ridx_post_os.environ.get(
                    "LUMO_FB_RIDX_STATE_LAYERS", "0,1,2")
                _lumo_fb_ridx_do_state = True
                if _lumo_fb_ridx_layers:
                    _lumo_fb_ridx_do_state = str(self.layer_idx) in {
                        _x.strip() for _x in _lumo_fb_ridx_layers.split(",") if _x.strip()}
                if _lumo_fb_ridx_do_state:
                    import json as _lumo_fb_ridx_post_json
                    import time as _lumo_fb_ridx_post_time
                    def _lumo_fb_ridx_post_summary(_cache, _idx):
                        if _idx is None or int(_idx) < 0:
                            return None
                        _idx = int(_idx)
                        _row = _cache[_idx].detach().float()
                        _flat = _row.reshape(-1)
                        return {
                            "idx": _idx,
                            "shape": list(_row.shape),
                            "sum": float(_flat.sum().item()),
                            "abs_sum": float(_flat.abs().sum().item()),
                            "head": _flat[:8].cpu().tolist(),
                        }
                    _lumo_fb_parent_writes = (
                        spec_state_indices_tensor.detach().reshape(
                            spec_state_indices_tensor.shape[0], -1)[0].cpu().tolist())
                    _lumo_fb_state_post = {
                        "parent_write_indices": _lumo_fb_parent_writes,
                        "conv_parent_writes": [
                            _lumo_fb_ridx_post_summary(conv_state, _idx)
                            for _idx in _lumo_fb_parent_writes[:6]
                        ],
                        "ssm_parent_writes": [
                            _lumo_fb_ridx_post_summary(ssm_state, _idx)
                            for _idx in _lumo_fb_parent_writes[:6]
                        ],
                    }
                    with open("/logs/fb_recurrent_index_debug.jsonl", "a", buffering=1) as _lumo_fb_ridx_fh:
                        _lumo_fb_ridx_fh.write(_lumo_fb_ridx_post_json.dumps({
                            "event": "gdn_recurrent_state_checksums",
                            "phase": "post",
                            "ts": round(_lumo_fb_ridx_post_time.time(), 4),
                            "layer_idx": int(self.layer_idx) if self.layer_idx is not None else None,
                            "prefix": str(getattr(self, "prefix", "")),
                            "num_actual_tokens": int(num_actual_tokens),
                            "num_spec_decodes": int(getattr(attn_metadata, "num_spec_decodes", 0)),
                            "state_post": _lumo_fb_state_post,
                        }) + chr(10))
            except Exception:
                pass
'''
    if old not in text:
        raise RuntimeError('F_b recurrent state post debug anchor not found')
    text = text.replace(old, new, 1)
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b recurrent state post debug patch')

text = gl.read_text()
sentinel = '# LUMO_FB_RECURRENT_INPUT_DEBUG'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b recurrent input debug already present')
else:
    old = '''        if attn_metadata.num_prefills > 0:
            g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
'''
    new = '''        # LUMO_FB_RECURRENT_INPUT_DEBUG: checksum row0's recurrent inputs
        # immediately before the SSM update.  If K1 and K2 differ here, the leak
        # is upstream of the recurrent kernel; if they match, the fused batched
        # SSM update itself is changing row0.
        import os as _lumo_fb_rinput_os
        if (
            _lumo_fb_rinput_os.environ.get("LUMO_FB_TENSOR_DEBUG") == "1"
            and spec_sequence_masks is not None
            and getattr(attn_metadata, "num_spec_decodes", 0) > 0
        ):
            try:
                _lumo_fb_rinput_layers = _lumo_fb_rinput_os.environ.get(
                    "LUMO_FB_RIDX_STATE_LAYERS", "0,1,2")
                _lumo_fb_rinput_do = True
                if _lumo_fb_rinput_layers:
                    _lumo_fb_rinput_do = str(self.layer_idx) in {
                        _x.strip() for _x in _lumo_fb_rinput_layers.split(",") if _x.strip()}
                if _lumo_fb_rinput_do:
                    import json as _lumo_fb_rinput_json
                    import time as _lumo_fb_rinput_time
                    def _lumo_fb_rinput_summary(_tensor, _row_tokens=6):
                        if _tensor is None:
                            return None
                        _t = _tensor.detach().float()
                        if _t.ndim >= 2:
                            _row = _t[:, :_row_tokens].contiguous()
                        else:
                            _row = _t[:_row_tokens].contiguous()
                        _flat = _row.reshape(-1)
                        return {
                            "shape": list(_t.shape),
                            "row_shape": list(_row.shape),
                            "sum": float(_flat.sum().item()),
                            "abs_sum": float(_flat.abs().sum().item()),
                            "head": _flat[:8].cpu().tolist(),
                        }
                    _lumo_fb_rinput_state = {
                        "query_spec": _lumo_fb_rinput_summary(query_spec),
                        "key_spec": _lumo_fb_rinput_summary(key_spec),
                        "value_spec": _lumo_fb_rinput_summary(value_spec),
                        "a": _lumo_fb_rinput_summary(a),
                        "b": _lumo_fb_rinput_summary(b),
                    }
                    with open("/logs/fb_recurrent_index_debug.jsonl", "a", buffering=1) as _lumo_fb_rinput_fh:
                        _lumo_fb_rinput_fh.write(_lumo_fb_rinput_json.dumps({
                            "event": "gdn_recurrent_input_checksums",
                            "phase": "pre_ssm",
                            "ts": round(_lumo_fb_rinput_time.time(), 4),
                            "layer_idx": int(self.layer_idx) if self.layer_idx is not None else None,
                            "prefix": str(getattr(self, "prefix", "")),
                            "num_actual_tokens": int(num_actual_tokens),
                            "num_spec_decodes": int(getattr(attn_metadata, "num_spec_decodes", 0)),
                            "inputs": _lumo_fb_rinput_state,
                        }) + chr(10))
            except Exception:
                pass

        if attn_metadata.num_prefills > 0:
            g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
'''
    if old not in text:
        raise RuntimeError('F_b recurrent input debug anchor not found')
    text = text.replace(old, new, 1)
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b recurrent input debug patch')

text = gl.read_text()
sentinel = '# LUMO_FB_GDN_PROJECTION_INPUT_DEBUG'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b GDN projection input debug already present')
else:
    old = '''        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if hasattr(self, "in_proj_qkv"):
'''
    new = '''        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        # LUMO_FB_GDN_PROJECTION_INPUT_DEBUG
        import os as _lumo_fb_gdn_proj_os
        _lumo_fb_gdn_proj_do = False
        _lumo_fb_gdn_proj_pre = None
        _lumo_fb_gdn_proj_mixed = None
        if _lumo_fb_gdn_proj_os.environ.get("LUMO_FB_TENSOR_DEBUG") == "1":
            _lumo_fb_gdn_proj_layers = _lumo_fb_gdn_proj_os.environ.get(
                "LUMO_FB_RIDX_STATE_LAYERS", "0,1,2")
            _lumo_fb_gdn_proj_do = True
            if _lumo_fb_gdn_proj_layers:
                _lumo_fb_gdn_proj_do = str(self.layer_idx) in {
                    _x.strip() for _x in _lumo_fb_gdn_proj_layers.split(",") if _x.strip()}
        def _lumo_fb_gdn_proj_summary(_tensor, _tokens=1):
            if _tensor is None:
                return None
            _t = _tensor.detach().float()
            _row = _t[:_tokens].contiguous()
            _flat = _row.reshape(-1)
            return {
                "shape": list(_t.shape),
                "row_shape": list(_row.shape),
                "sum": float(_flat.sum().item()),
                "abs_sum": float(_flat.abs().sum().item()),
                "head": _flat[:8].cpu().tolist(),
            }
        if _lumo_fb_gdn_proj_do:
            try:
                _lumo_fb_gdn_proj_pre = {
                    "hidden_states_token0": _lumo_fb_gdn_proj_summary(hidden_states, 1),
                }
            except Exception:
                _lumo_fb_gdn_proj_pre = None
        if hasattr(self, "in_proj_qkv"):
'''
    if old not in text:
        raise RuntimeError('F_b GDN projection input debug pre anchor not found')
    text = text.replace(old, new, 1)
    old = '''            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
'''
    new = '''            b = b.contiguous()
            a = a.contiguous()
            if _lumo_fb_gdn_proj_do:
                _lumo_fb_gdn_proj_mixed = {
                    "mixed_qkv_token0": _lumo_fb_gdn_proj_summary(mixed_qkv, 1),
                    "ba_token0": _lumo_fb_gdn_proj_summary(ba, 1),
                }
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            if _lumo_fb_gdn_proj_do:
                _lumo_fb_gdn_proj_mixed = {
                    "mixed_qkvz_token0": _lumo_fb_gdn_proj_summary(mixed_qkvz, 1),
                    "ba_token0": _lumo_fb_gdn_proj_summary(ba, 1),
                }
'''
    if old not in text:
        raise RuntimeError('F_b GDN projection input debug projection anchor not found')
    text = text.replace(old, new, 1)
    old = '''        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
'''
    new = '''        if _lumo_fb_gdn_proj_do:
            try:
                import json as _lumo_fb_gdn_proj_json
                import time as _lumo_fb_gdn_proj_time
                with open("/logs/fb_recurrent_index_debug.jsonl", "a", buffering=1) as _lumo_fb_gdn_proj_fh:
                    _lumo_fb_gdn_proj_fh.write(_lumo_fb_gdn_proj_json.dumps({
                        "event": "gdn_projection_input_checksums",
                        "phase": "pre_core",
                        "ts": round(_lumo_fb_gdn_proj_time.time(), 4),
                        "layer_idx": int(self.layer_idx) if self.layer_idx is not None else None,
                        "prefix": str(getattr(self, "prefix", "")),
                        "num_tokens": int(num_tokens),
                        "pre": _lumo_fb_gdn_proj_pre,
                        "projection": _lumo_fb_gdn_proj_mixed,
                    }) + chr(10))
            except Exception:
                pass

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
'''
    if old not in text:
        raise RuntimeError('F_b GDN projection input debug write anchor not found')
    text = text.replace(old, new, 1)
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b GDN projection input debug patch')

text = gl.read_text()
sentinel = '# LUMO_FB_BATCH_INVARIANT_BA_PROJ'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b batch-invariant BA projection patch already present')
else:
    old = '''            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            if _lumo_fb_gdn_proj_do:
                _lumo_fb_gdn_proj_mixed = {
                    "mixed_qkvz_token0": _lumo_fb_gdn_proj_summary(mixed_qkvz, 1),
                    "ba_token0": _lumo_fb_gdn_proj_summary(ba, 1),
                }

            if self.gqa_interleaved_layout:
'''
    new = '''            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            # LUMO_FB_BATCH_INVARIANT_BA_PROJ: make the BA projection shape
            # independent across active K by padding spec rows to a fixed row
            # group and issuing one batched projection. This preserves the
            # row-scaled architecture; it is not a per-row projection loop.
            if _lumo_fb_gdn_proj_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1":
                try:
                    _lumo_fb_ctx = get_forward_context()
                    _lumo_fb_meta = _lumo_fb_ctx.attn_metadata
                    if isinstance(_lumo_fb_meta, dict):
                        _lumo_fb_meta = _lumo_fb_meta.get(self.prefix)
                    _lumo_fb_nspec = int(getattr(_lumo_fb_meta, "num_spec_decodes", 0))
                    _lumo_fb_qsl = getattr(_lumo_fb_meta, "spec_query_start_loc", None)
                    if (
                        _lumo_fb_nspec > 1
                        and getattr(_lumo_fb_meta, "num_prefills", 0) == 0
                        and getattr(_lumo_fb_meta, "num_decodes", 0) == 0
                        and _lumo_fb_qsl is not None
                    ):
                        _lumo_fb_qsl_cpu = _lumo_fb_qsl[: _lumo_fb_nspec + 1].detach().cpu().tolist()
                        _lumo_fb_spans = [
                            (int(_lumo_fb_qsl_cpu[_lumo_fb_i]),
                             int(_lumo_fb_qsl_cpu[_lumo_fb_i + 1]))
                            for _lumo_fb_i in range(_lumo_fb_nspec)
                        ]
                        _lumo_fb_row_len = max((_e - _s for _s, _e in _lumo_fb_spans), default=0)
                        _lumo_fb_pad_rows = max(
                            _lumo_fb_nspec,
                            int(_lumo_fb_gdn_proj_os.environ.get(
                                "LUMO_FB_PROJ_PAD_ROWS", "8")))
                        if _lumo_fb_row_len > 0 and _lumo_fb_pad_rows > _lumo_fb_nspec:
                            _lumo_fb_padded = hidden_states.new_zeros(
                                (_lumo_fb_pad_rows * _lumo_fb_row_len,
                                 hidden_states.shape[-1]))
                            for _lumo_fb_i, (_lumo_fb_s, _lumo_fb_e) in enumerate(_lumo_fb_spans):
                                _lumo_fb_len = _lumo_fb_e - _lumo_fb_s
                                _lumo_fb_ps = _lumo_fb_i * _lumo_fb_row_len
                                _lumo_fb_padded[_lumo_fb_ps:_lumo_fb_ps + _lumo_fb_len] = hidden_states[_lumo_fb_s:_lumo_fb_e]
                            _lumo_fb_qkvz_padded, _ = self.in_proj_qkvz(_lumo_fb_padded)
                            _lumo_fb_ba_padded, _ = self.in_proj_ba(_lumo_fb_padded)
                            _lumo_fb_qkvz_parts = []
                            _lumo_fb_ba_parts = []
                            for _lumo_fb_i, (_lumo_fb_s, _lumo_fb_e) in enumerate(_lumo_fb_spans):
                                _lumo_fb_len = _lumo_fb_e - _lumo_fb_s
                                _lumo_fb_ps = _lumo_fb_i * _lumo_fb_row_len
                                _lumo_fb_qkvz_parts.append(_lumo_fb_qkvz_padded[_lumo_fb_ps:_lumo_fb_ps + _lumo_fb_len])
                                _lumo_fb_ba_parts.append(_lumo_fb_ba_padded[_lumo_fb_ps:_lumo_fb_ps + _lumo_fb_len])
                            mixed_qkvz = torch.cat(_lumo_fb_qkvz_parts, dim=0)
                            ba = torch.cat(_lumo_fb_ba_parts, dim=0)
                        elif _lumo_fb_row_len > 0:
                            _lumo_fb_reshaped = hidden_states[:_lumo_fb_nspec * _lumo_fb_row_len]
                            mixed_qkvz, _ = self.in_proj_qkvz(_lumo_fb_reshaped)
                            ba, _ = self.in_proj_ba(_lumo_fb_reshaped)
                except Exception:
                    pass
            if _lumo_fb_gdn_proj_do:
                _lumo_fb_gdn_proj_mixed = {
                    "mixed_qkvz_token0": _lumo_fb_gdn_proj_summary(mixed_qkvz, 1),
                    "ba_token0": _lumo_fb_gdn_proj_summary(ba, 1),
                }

            if self.gqa_interleaved_layout:
'''
    if old not in text:
        raise RuntimeError('F_b batch-invariant BA projection anchor not found')
    text = text.replace(old, new, 1)
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b batch-invariant BA projection patch')

text = gl.read_text()
sentinel = '# LUMO_FB_BATCH_INVARIANT_GDN_OUT_PROJ'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b batch-invariant GDN out projection patch already present')
else:
    old = '''        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)
'''
    new = '''        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        # LUMO_FB_BATCH_INVARIANT_GDN_OUT_PROJ: use one fixed-shape batched
        # output projection for spec rows, then scatter the real rows back.
        _lumo_fb_out_done = False
        if _lumo_fb_gdn_proj_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1":
            try:
                _lumo_fb_ctx = get_forward_context()
                _lumo_fb_meta = _lumo_fb_ctx.attn_metadata
                if isinstance(_lumo_fb_meta, dict):
                    _lumo_fb_meta = _lumo_fb_meta.get(self.prefix)
                _lumo_fb_nspec = int(getattr(_lumo_fb_meta, "num_spec_decodes", 0))
                _lumo_fb_qsl = getattr(_lumo_fb_meta, "spec_query_start_loc", None)
                if (
                    _lumo_fb_nspec > 1
                    and getattr(_lumo_fb_meta, "num_prefills", 0) == 0
                    and getattr(_lumo_fb_meta, "num_decodes", 0) == 0
                    and _lumo_fb_qsl is not None
                ):
                    _lumo_fb_qsl_cpu = _lumo_fb_qsl[: _lumo_fb_nspec + 1].detach().cpu().tolist()
                    _lumo_fb_spans = [
                        (int(_lumo_fb_qsl_cpu[_lumo_fb_i]),
                         int(_lumo_fb_qsl_cpu[_lumo_fb_i + 1]))
                        for _lumo_fb_i in range(_lumo_fb_nspec)
                    ]
                    _lumo_fb_row_len = max((_e - _s for _s, _e in _lumo_fb_spans), default=0)
                    _lumo_fb_pad_rows = max(
                        _lumo_fb_nspec,
                        int(_lumo_fb_gdn_proj_os.environ.get(
                            "LUMO_FB_PROJ_PAD_ROWS", "8")))
                    if _lumo_fb_row_len > 0 and _lumo_fb_pad_rows > _lumo_fb_nspec:
                        _lumo_fb_padded = core_attn_out.new_zeros(
                            (_lumo_fb_pad_rows * _lumo_fb_row_len,
                             core_attn_out.shape[-1]))
                        for _lumo_fb_i, (_lumo_fb_s, _lumo_fb_e) in enumerate(_lumo_fb_spans):
                            _lumo_fb_len = _lumo_fb_e - _lumo_fb_s
                            _lumo_fb_ps = _lumo_fb_i * _lumo_fb_row_len
                            _lumo_fb_padded[_lumo_fb_ps:_lumo_fb_ps + _lumo_fb_len] = core_attn_out[_lumo_fb_s:_lumo_fb_e]
                        _lumo_fb_out_padded, _ = self.out_proj(_lumo_fb_padded)
                        for _lumo_fb_i, (_lumo_fb_s, _lumo_fb_e) in enumerate(_lumo_fb_spans):
                            _lumo_fb_len = _lumo_fb_e - _lumo_fb_s
                            _lumo_fb_ps = _lumo_fb_i * _lumo_fb_row_len
                            output[_lumo_fb_s:_lumo_fb_e] = _lumo_fb_out_padded[_lumo_fb_ps:_lumo_fb_ps + _lumo_fb_len]
                        _lumo_fb_out_done = True
                    elif _lumo_fb_row_len > 0:
                        output[:_lumo_fb_nspec * _lumo_fb_row_len], _ = self.out_proj(
                            core_attn_out[:_lumo_fb_nspec * _lumo_fb_row_len])
                        _lumo_fb_out_done = True
            except Exception:
                _lumo_fb_out_done = False
        if not _lumo_fb_out_done:
            output[:num_tokens], _ = self.out_proj(core_attn_out)
'''
    if old not in text:
        raise RuntimeError('F_b batch-invariant GDN out projection anchor not found')
    text = text.replace(old, new, 1)
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b batch-invariant GDN out projection patch')

fa = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_attn.py')
text = fa.read_text()
sentinel = '# LUMO_FB_SPLIT_PARTIAL_KV_ATTN'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b split partial-KV attention patch already present')
else:
    old = '''from dataclasses import dataclass
from typing import ClassVar
'''
    new = '''from dataclasses import dataclass
from typing import ClassVar
import os as _lumo_fb_fa_os
'''
    if old not in text:
        raise RuntimeError('F_b split partial-KV attention import anchor not found')
    text = text.replace(old, new, 1)

    old = '''        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata
'''
    new = '''        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        # LUMO_FB_SPLIT_PARTIAL_KV_ATTN: internal F_b rows must not copy the
        # parent's partially-filled KV block. When the verify batch has multiple
        # equal-width speculative rows, run attention as:
        #   shared prefix from row0's paged KV cache + dense causal suffix from
        #   this forward's K/V tensors.
        # This is the add-a-row architecture: sibling rows read the same prefix
        # blocks and own only their tiny speculative suffix.
        if (
            _lumo_fb_fa_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1"
            and _lumo_fb_fa_os.environ.get("LUMO_FB_NO_KV_PREFIX_COPY") == "1"
            and num_reqs > 1
            and max_query_len > 1
            and not use_cascade
            and self.dcp_world_size == 1
        ):
            query_lens = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
            prefix_lens = seq_lens[:num_reqs] - query_lens
            if int(query_lens.min().item()) == int(query_lens.max().item()):
                prefix_len = int(prefix_lens[0].item())
                if prefix_len > 0 and int(prefix_lens.min().item()) == int(prefix_lens.max().item()):
                    attn_metadata.lumo_fb_split_partial_kv = True
                    attn_metadata.lumo_fb_split_prefix_len = prefix_len
                    attn_metadata.lumo_fb_split_cu_prefix_query_lens = torch.tensor(
                        [0, num_actual_tokens], dtype=torch.int32, device=self.device)
                    attn_metadata.lumo_fb_split_prefix_kv_lens = torch.tensor(
                        [prefix_len], dtype=torch.int32, device=self.device)
        return attn_metadata
'''
    if old not in text:
        raise RuntimeError('F_b split partial-KV attention metadata anchor not found')
    text = text.replace(old, new, 1)

    old = '''        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
'''
    new = '''        if getattr(attn_metadata, "lumo_fb_split_partial_kv", False):
            # Shared-prefix + dense-suffix attention. Prefix reads row0's
            # block table up to the exact token prefix length, including a
            # partial final block. Suffix uses this forward's dense K/V and
            # therefore does not require copied prefix slots in row-private KV.
            cu_seqlens_q = attn_metadata.query_start_loc
            max_seqlen_q = attn_metadata.max_query_len
            prefix_len = int(attn_metadata.lumo_fb_split_prefix_len)
            sliding_window_size = (
                list(self.sliding_window)
                if self.sliding_window is not None
                else None
            )
            if sliding_window_size is not None and sliding_window_size != [-1, -1]:
                raise RuntimeError("LUMO_FB split partial-KV attention does not support sliding window")
            prefix_descale_shape = (
                attn_metadata.lumo_fb_split_cu_prefix_query_lens.shape[0] - 1,
                self.num_kv_heads,
            )
            suffix_descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
            prefix_output, prefix_lse = flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=attn_metadata.lumo_fb_split_cu_prefix_query_lens,
                seqused_k=attn_metadata.lumo_fb_split_prefix_kv_lens,
                max_seqlen_q=num_actual_tokens,
                max_seqlen_k=prefix_len,
                softmax_scale=self.scale,
                causal=False,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=attn_metadata.block_table[:1],
                softcap=self.logits_soft_cap,
                return_softmax_lse=True,
                scheduler_metadata=None,
                fa_version=self.vllm_flash_attn_version,
                q_descale=layer._q_scale.expand(prefix_descale_shape),
                k_descale=layer._k_scale.expand(prefix_descale_shape),
                v_descale=layer._v_scale.expand(prefix_descale_shape),
                num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,
                s_aux=self.sinks,
            )
            suffix_output, suffix_lse = flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key[:num_actual_tokens],
                v=value[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_q,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                softcap=self.logits_soft_cap,
                return_softmax_lse=True,
                fa_version=self.vllm_flash_attn_version,
                q_descale=layer._q_scale.expand(suffix_descale_shape),
                k_descale=layer._k_scale.expand(suffix_descale_shape),
                v_descale=layer._v_scale.expand(suffix_descale_shape),
                num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,
            )
            merge_attn_states(
                output[:num_actual_tokens],
                prefix_output,
                prefix_lse,
                suffix_output,
                suffix_lse,
            )
            return output

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
'''
    if old not in text:
        raise RuntimeError('F_b split partial-KV attention forward anchor not found')
    text = text.replace(old, new, 1)
    fa.write_text(text)
    import py_compile
    py_compile.compile(str(fa), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b split partial-KV attention patch')

gm = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = gm.read_text()
sentinel = '# LUMO_FB_INTERNAL_ROW_CAPACITY'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_b internal-row capacity patch already present')
else:
    old = '''        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs
'''
    new = '''        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs
        # LUMO_FB_INTERNAL_ROW_CAPACITY: internal verifier rows are not public
        # requests, but they do need runner-side batch slots. Keep scheduler
        # concurrency unchanged and over-allocate only the GPU runner buffers.
        if _lumo_fb_kernel_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1":
            _lumo_fb_branch_depth = int(
                _lumo_fb_kernel_os.environ.get("LUMO_FB_TREE_BRANCH_DEPTH", "2"))
            _lumo_fb_row_multiplier = int(
                _lumo_fb_kernel_os.environ.get(
                    "LUMO_FB_INTERNAL_ROW_MULTIPLIER",
                    str(2 ** max(1, _lumo_fb_branch_depth))))
            self.max_num_reqs = max(
                self.max_num_reqs,
                scheduler_config.max_num_seqs * _lumo_fb_row_multiplier)
'''
    if old not in text:
        raise RuntimeError('F_b internal-row capacity anchor not found')
    if 'import os as _lumo_fb_kernel_os' not in text:
        old_import = 'import time\n'
        if old_import not in text:
            raise RuntimeError('F_b internal-row capacity import anchor not found')
        text = text.replace(old_import, old_import + 'import os as _lumo_fb_kernel_os\n', 1)
    text = text.replace(old, new, 1)
    gm.write_text(text)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_b internal-row capacity patch')

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
                (
                    vllm_config.speculative_config.num_speculative_tokens
                    + (1 if (
                        _lumo_fb_kernel_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1"
                        or _lumo_fb_kernel_os.environ.get("LUMO_FA_UNIQUE_NODES") == "1"
                    ) else 0)
                )
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

_FA_UNIQUE_NODES_BLOCK = r'''
python3 - <<'LUMOFAUNIQUENODES'
from pathlib import Path

ga = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py')
text = ga.read_text()
sentinel = '# LUMO_FA_UNIQUE_NODES_GDN_ATTN'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_a unique-node GDN metadata patch already present')
else:
    if 'import os as _lumo_fb_kernel_os' not in text:
        old = 'from dataclasses import dataclass\n\nimport torch\n'
        new = 'from dataclasses import dataclass\nimport os as _lumo_fb_kernel_os\n\nimport torch\n'
        if old not in text:
            raise RuntimeError('F_a unique-node gdn_attn import anchor not found')
        text = text.replace(old, new, 1)
    if 'import ast as _lumo_fa_ast' not in text:
        text = text.replace(
            'import os as _lumo_fb_kernel_os\n',
            'import os as _lumo_fb_kernel_os\nimport ast as _lumo_fa_ast\nimport json as _lumo_fa_json\nimport time as _lumo_fa_time\n',
            1,
        )
    if '_lumo_fa_replay_gdn' not in text:
        text = text.replace(
            'import time as _lumo_fa_time\n',
            'import time as _lumo_fa_time\n'
            'try:\n'
            '    from vllm.model_executor.layers.mamba import gdn_linear_attn as _lumo_fa_replay_gdn\n'
            'except Exception:\n'
            '    _lumo_fa_replay_gdn = None\n',
            1,
        )

    old = '        m = common_attn_metadata\n\n        query_start_loc = m.query_start_loc\n'
    new = '        m = common_attn_metadata\n        fa_tree_parent_indices_tensor = None\n        fa_tree_depth_rows = None\n        fa_tree_depth_row_tensors = None\n        fa_tree_depth_query_start_tensors = None\n        fa_unique_node_mode = False\n        fa_unique_expanded_node_mode = False\n\n        query_start_loc = m.query_start_loc\n'
    if old not in text:
        raise RuntimeError('F_a unique-node metadata default anchor not found')
    text = text.replace(old, new, 1)

    old = """    spec_write_state_slot_tensor: torch.Tensor | None = None
    non_spec_state_indices_tensor: torch.Tensor | None = (
"""
    new = """    spec_write_state_slot_tensor: torch.Tensor | None = None
    # LUMO_FA_UNIQUE_NODES_GDN_ATTN: packed-tree verifier state map.
    # The target verifier remains one scheduler request; GDN layers process
    # root+selected unique nodes as one-token recurrent sequences whose
    # initial state is gathered from the parent node's write slot.
    fa_tree_parent_indices_tensor: torch.Tensor | None = None
    fa_tree_depth_rows: tuple[tuple[int, ...], ...] | None = None
    fa_tree_depth_row_tensors: tuple[torch.Tensor, ...] | None = None
    fa_tree_depth_query_start_tensors: tuple[torch.Tensor, ...] | None = None
    fa_unique_node_mode: bool = False
    fa_unique_expanded_node_mode: bool = False
    non_spec_state_indices_tensor: torch.Tensor | None = (
"""
    if old not in text:
        old = """    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
"""
        new = """    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    # LUMO_FA_UNIQUE_NODES_GDN_ATTN: packed-tree verifier metadata.
    fa_tree_parent_indices_tensor: torch.Tensor | None = None
    fa_tree_depth_rows: tuple[tuple[int, ...], ...] | None = None
    fa_tree_depth_row_tensors: tuple[torch.Tensor, ...] | None = None
    fa_tree_depth_query_start_tensors: tuple[torch.Tensor, ...] | None = None
    fa_unique_node_mode: bool = False
    fa_unique_expanded_node_mode: bool = False
    non_spec_state_indices_tensor: torch.Tensor | None = (
"""
        if old not in text:
            raise RuntimeError('F_a unique-node dataclass anchor not found')
    text = text.replace(old, new, 1)

    old = """            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]
"""
    new = """            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

            _spec = getattr(self.vllm_config, "speculative_config", None)
            _tree_src = getattr(_spec, "speculative_token_tree", None) if _spec is not None else None
            if _tree_src and int(num_spec_decodes) == 1:
                _choices = list(_lumo_fa_ast.literal_eval(_tree_src))
                _node_count = len(_choices)
                _path_to_idx = {tuple(_p): _i for _i, _p in enumerate(_choices)}
                _parents = [
                    _path_to_idx.get(tuple(_p[:-1]), -1)
                    for _p in _choices
                ]
                _max_depth = max((len(tuple(_p)) for _p in _choices), default=0)
                _is_spine = (
                    _node_count == _max_depth
                    and all(tuple(_p) == tuple([0] * len(tuple(_p)))
                            for _p in _choices)
                )
                if block_table_tensor.size(1) < _node_count + 2:
                    raise RuntimeError(
                        "LUMO_FA_UNIQUE_NODES requires root + node write "
                        f"state slots: need {_node_count + 2}, got "
                        f"{block_table_tensor.size(1)}")
                _row = block_table_tensor[spec_sequence_masks, :_node_count + 2][0]
                _write_slots = _row[1:_node_count + 2].contiguous()
                _initial_slots = torch.empty(
                    (_node_count + 1,), dtype=torch.int32,
                    device=query_start_loc.device)
                _initial_slots[0] = _row[0]
                for _i, _parent in enumerate(_parents):
                    _initial_slots[_i + 1] = _write_slots[0 if _parent < 0 else _parent + 1]
                spec_initial_state_indices_tensor = _initial_slots
                spec_initial_state_slot_tensor = None
                spec_write_state_slot_tensor = torch.zeros(
                    (_node_count + 1,), dtype=torch.int32,
                    device=query_start_loc.device)
                spec_state_indices_tensor = _write_slots.view(_node_count + 1, 1)
                spec_query_start_loc = torch.arange(
                    _node_count + 2, dtype=torch.int32,
                    device=query_start_loc.device)
                num_spec_decodes = _node_count + 1
                num_spec_decode_tokens = _node_count + 1
                num_accepted_tokens = torch.ones(
                    (_node_count + 1,), dtype=torch.int32,
                    device=query_start_loc.device)
                fa_unique_expanded_node_mode = True
                fa_tree_parent_indices_tensor = torch.tensor(
                    [-2] + _parents, dtype=torch.int32,
                    device=query_start_loc.device)
                _actual_parents = [-1]
                _depths = [0]
                for _parent in _parents:
                    _actual = 0 if int(_parent) < 0 else int(_parent) + 1
                    _actual_parents.append(_actual)
                    _depths.append(_depths[_actual] + 1)
                fa_tree_depth_rows = tuple(
                    tuple(_i for _i, _d in enumerate(_depths) if _d == _depth)
                    for _depth in range(max(_depths) + 1)
                )
                fa_tree_depth_row_tensors = tuple(
                    torch.tensor(_rows, dtype=torch.long, device=query_start_loc.device)
                    for _rows in fa_tree_depth_rows
                )
                fa_tree_depth_query_start_tensors = tuple(
                    torch.arange(len(_rows) + 1, dtype=torch.int32, device=query_start_loc.device)
                    for _rows in fa_tree_depth_rows
                )
                fa_unique_node_mode = True
                try:
                    _fh = globals().get("_LUMO_FA_UNIFIED_FH")
                    if _fh is None:
                        _fh = open("/logs/fb_debug.jsonl", "a", buffering=1)
                        globals()["_LUMO_FA_UNIFIED_FH"] = _fh
                    _fh.write(_lumo_fa_json.dumps({
                        "ts": round(_lumo_fa_time.time(), 4),
                        "event": "round_f_unified_step",
                        "stage": "stage3_spine_only_unique_node_state_tree" if _is_spine else "stage3_unique_node_state_tree_expanded",
                        "component_under_test": "fa_unique_node_state_tree_verifier",
                        "verifier_path": "LUMO_FA_UNIQUE_NODES/spine_expanded_parent_state_rows" if _is_spine else "LUMO_FA_UNIQUE_NODES/expanded_parent_state_rows",
                        "internal_rows_enabled": False,
                        "kernel_rows_enabled": False,
                        "no_kv_prefix_copy_enabled": True,
                        "candidate_pool_nodes": int(_node_count),
                        "selected_nodes": int(_node_count),
                        "verified_nodes": int(_node_count),
                        "unique_tree_nodes": int(_node_count),
                        "trimmed_nodes": 0,
                        "max_depth": int(_max_depth),
                        "sources": {"mtp_top1": int(_node_count), "mtp_alt": 0, "suffix": 0},
                        "path_rows": 0,
                        "scheduler_visible_clone_requests": 0,
                        "prefix_kv_copy_bytes": 0,
                        "recomputed_shared_prefix_nodes": 0,
                        "extra_proposer_for_trimmed_nodes": 0,
                        "accepted_path_commit_only": True,
                        "tree_attention": False,
                        "gdn_parent_gather": True,
                        "depth_positions": True,
                        "tree_sampler": False,
                        "top1_spine_accept_depth": None,
                        "accepted_depth": None,
                        "accepted_node_path": [],
                        "estimated_event_ms": None,
                        "event_budget_ms": None,
                        "tree_score": None,
                        "proposer_us": 0,
                        "trim_us": 0,
                        "verify_us": 0,
                        "tree_attention_us": 0,
                        "gdn_parent_gather_us": 0,
                        "depth_sync_us": 0,
                        "commit_us": 0,
                        "gdn_state_bytes_copied": 0,
                        "kv_suffix_bytes_copied": 0,
                        "physical_minimum_invariant_failures": [],
                        "parent_map": [-2] + [int(_p) for _p in _parents],
                        "state_rows": (
                            list(spec_state_indices_tensor.shape)
                            if spec_state_indices_tensor is not None else None
                        ),
                        "spine_chain_degenerate_unique_tree": bool(_is_spine),
                        "expanded_parent_state_rows": True,
                    }) + chr(10))
                except Exception:
                    pass
            elif spec_state_indices_tensor is not None and int(num_spec_decodes) == 1:
                _node_count = int(spec_state_indices_tensor.size(-1))
                _max_depth = int(_node_count)
                _parents = [-1] + [int(_i) for _i in range(_node_count - 1)]
                fa_tree_parent_indices_tensor = torch.tensor(
                    _parents, dtype=torch.int32, device=query_start_loc.device)
                _actual_parents = [-1]
                _depths = [0]
                for _parent in _parents[1:]:
                    _actual = 0 if int(_parent) < 0 else int(_parent) + 1
                    _actual_parents.append(_actual)
                    _depths.append(_depths[_actual] + 1)
                fa_tree_depth_rows = tuple(
                    tuple(_i for _i, _d in enumerate(_depths) if _d == _depth)
                    for _depth in range(max(_depths) + 1)
                )
                fa_tree_depth_row_tensors = tuple(
                    torch.tensor(_rows, dtype=torch.long, device=query_start_loc.device)
                    for _rows in fa_tree_depth_rows
                )
                fa_tree_depth_query_start_tensors = tuple(
                    torch.arange(len(_rows) + 1, dtype=torch.int32, device=query_start_loc.device)
                    for _rows in fa_tree_depth_rows
                )
                fa_unique_node_mode = True
                try:
                    _fh = globals().get("_LUMO_FA_UNIFIED_FH")
                    if _fh is None:
                        _fh = open("/logs/fb_debug.jsonl", "a", buffering=1)
                        globals()["_LUMO_FA_UNIFIED_FH"] = _fh
                    _fh.write(_lumo_fa_json.dumps({
                        "ts": round(_lumo_fa_time.time(), 4),
                        "event": "round_f_unified_step",
                        "stage": "stage3_spine_only_unique_node_state_tree",
                        "component_under_test": "fa_unique_node_state_tree_verifier",
                        "verifier_path": "LUMO_FA_UNIQUE_NODES/e3_spine_chain",
                        "internal_rows_enabled": False,
                        "kernel_rows_enabled": False,
                        "no_kv_prefix_copy_enabled": True,
                        "candidate_pool_nodes": int(_node_count),
                        "selected_nodes": int(_node_count),
                        "verified_nodes": int(_node_count),
                        "unique_tree_nodes": int(_node_count),
                        "trimmed_nodes": 0,
                        "max_depth": int(_max_depth),
                        "sources": {"mtp_top1": int(_node_count), "mtp_alt": 0, "suffix": 0},
                        "path_rows": 0,
                        "scheduler_visible_clone_requests": 0,
                        "prefix_kv_copy_bytes": 0,
                        "recomputed_shared_prefix_nodes": 0,
                        "extra_proposer_for_trimmed_nodes": 0,
                        "accepted_path_commit_only": True,
                        "tree_attention": False,
                        "gdn_parent_gather": False,
                        "depth_positions": True,
                        "tree_sampler": False,
                        "top1_spine_accept_depth": None,
                        "accepted_depth": None,
                        "accepted_node_path": [],
                        "estimated_event_ms": None,
                        "event_budget_ms": None,
                        "tree_score": None,
                        "proposer_us": 0,
                        "trim_us": 0,
                        "verify_us": 0,
                        "tree_attention_us": 0,
                        "gdn_parent_gather_us": 0,
                        "depth_sync_us": 0,
                        "commit_us": 0,
                        "gdn_state_bytes_copied": 0,
                        "kv_suffix_bytes_copied": 0,
                        "physical_minimum_invariant_failures": [],
                        "parent_map": [int(_p) for _p in _parents],
                        "state_rows": list(spec_state_indices_tensor.shape),
                        "spine_chain_degenerate_unique_tree": True,
                        "expanded_parent_state_rows": False,
                    }) + chr(10))
                except Exception:
                    pass
"""
    if old not in text:
        raise RuntimeError('F_a unique-node spec metadata anchor not found')
    text = text.replace(old, new, 1)

    old = """            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
"""
    new = """            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
"""
    if old not in text:
        raise RuntimeError('F_a unique-node cudagraph guard anchor not found')
    text = text.replace(old, new, 1)

    old = """            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            if spec_initial_state_indices_tensor is not None:
"""
    new = """            assert spec_sequence_masks is not None
            if bool(locals().get("fa_unique_node_mode", False)):
                _fa_state_cols = int(spec_state_indices_tensor.size(-1))
                self.spec_state_indices_tensor[
                    :num_spec_decodes, :_fa_state_cols
                ].copy_(spec_state_indices_tensor, non_blocking=True)
                spec_state_indices_tensor = self.spec_state_indices_tensor[
                    :batch_size, :_fa_state_cols
                ]
                spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)
            else:
                self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                    spec_state_indices_tensor, non_blocking=True
                )
                spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
                spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            if spec_initial_state_indices_tensor is not None:
"""
    if old not in text:
        raise RuntimeError('F_a unique-node cudagraph state copy anchor not found')
    text = text.replace(old, new, 1)

    old = """            spec_write_state_slot_tensor=spec_write_state_slot_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
"""
    new = """            spec_write_state_slot_tensor=spec_write_state_slot_tensor,
            fa_tree_parent_indices_tensor=fa_tree_parent_indices_tensor,
            fa_tree_depth_rows=fa_tree_depth_rows,
            fa_tree_depth_row_tensors=fa_tree_depth_row_tensors,
            fa_tree_depth_query_start_tensors=fa_tree_depth_query_start_tensors,
            fa_unique_node_mode=bool(fa_unique_node_mode),
            fa_unique_expanded_node_mode=bool(fa_unique_expanded_node_mode),
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
"""
    if old not in text:
        old = """            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
"""
        new = """            spec_state_indices_tensor=spec_state_indices_tensor,
            fa_tree_parent_indices_tensor=fa_tree_parent_indices_tensor,
            fa_tree_depth_rows=fa_tree_depth_rows,
            fa_tree_depth_row_tensors=fa_tree_depth_row_tensors,
            fa_tree_depth_query_start_tensors=fa_tree_depth_query_start_tensors,
            fa_unique_node_mode=bool(fa_unique_node_mode),
            fa_unique_expanded_node_mode=bool(fa_unique_expanded_node_mode),
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
"""
        if old not in text:
            raise RuntimeError('F_a unique-node metadata ctor anchor not found')
    text = text.replace(old, new, 1)

    ga.write_text(text)
    import py_compile
    py_compile.compile(str(ga), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_a unique-node GDN metadata patch')

gl = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py')
text = gl.read_text()
sentinel = '# LUMO_FA_UNIQUE_NODES_GDN_LINEAR'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_a unique-node GDN linear telemetry patch already present')
else:
    if 'import os as _lumo_fa_os' not in text:
        text = text.replace(
            'import torch\n',
            'import torch\nimport os as _lumo_fa_os\nimport json as _lumo_fa_json\nimport time as _lumo_fa_time\n',
            1,
        )
    if '# LUMO_FA_TREE_DELTA_TORCH' not in text:
        old = '\n@CustomOp.register("chunk_gated_delta_rule")\n'
        new = r"""
# LUMO_FA_TREE_DELTA_TORCH: topo-ordered tree-ancestor WY/UT delta update.
def _lumo_fa_tree_delta_torch(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    initial_state_indices: torch.Tensor | None,
    parent_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = q.squeeze(0)
    k = k.squeeze(0)
    v = v.squeeze(0)
    n = int(q.shape[0])
    h_k = int(k.shape[1])
    h_v = int(v.shape[1])
    if h_v % h_k != 0:
        raise RuntimeError(f"LUMO_FA_TREE_DELTA_TORCH requires value heads divisible by key heads, got {h_v=} {h_k=}")
    repeat = h_v // h_k
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    if use_qk_l2norm_in_kernel:
        q_f = q_f * torch.rsqrt(torch.sum(q_f * q_f, dim=-1, keepdim=True) + 1e-6)
        k_f = k_f * torch.rsqrt(torch.sum(k_f * k_f, dim=-1, keepdim=True) + 1e-6)
    if repeat != 1:
        q_f = q_f.repeat_interleave(repeat, dim=1)
        k_f = k_f.repeat_interleave(repeat, dim=1)

    g, _ = fused_gdn_gating(A_log=A_log, a=a, b=b, dt_bias=dt_bias)
    alpha = torch.exp(g.squeeze(0).to(torch.float32))
    beta = torch.sigmoid(b.to(torch.float32))
    q_f = q_f * (k.shape[-1] ** -0.5)
    v_f = v.to(torch.float32)

    parents = parent_indices.to(device=q_f.device, dtype=torch.long)
    actual_parents = torch.empty((n,), dtype=torch.long, device=q_f.device)
    actual_parents[0] = -1
    for i in range(1, n):
        parent = int(parents[i].item())
        actual_parents[i] = 0 if parent < 0 else parent + 1

    ancestor = torch.zeros((n, n), dtype=torch.bool, device=q_f.device)
    gamma = torch.empty((n, h_v), dtype=torch.float32, device=q_f.device)
    for i in range(n):
        parent = int(actual_parents[i].item())
        gamma[i] = alpha[i] if parent < 0 else gamma[parent] * alpha[i]
        while parent >= 0:
            ancestor[i, parent] = True
            parent = int(actual_parents[parent].item())

    if initial_state_indices is not None:
        prefix_idx = int(initial_state_indices[0].item())
    else:
        prefix_idx = int(ssm_state_indices.reshape(-1)[0].item())
    prefix_state = initial_state[prefix_idx].to(torch.float32)

    kk = torch.einsum("nhd,mhd->hnm", k_f, k_f)
    gamma_hn = gamma.transpose(0, 1)
    beta_hn = beta.transpose(0, 1)
    ratio = gamma_hn[:, :, None] / gamma_hn[:, None, :].clamp_min(1e-20)
    lower = ancestor.to(torch.float32).unsqueeze(0) * beta_hn[:, :, None] * ratio * kk
    system = torch.eye(n, dtype=torch.float32, device=q_f.device).unsqueeze(0) + lower

    initial_projection = torch.einsum("hvk,nhk->nhv", prefix_state, k_f)
    rhs = beta[:, :, None] * (v_f - gamma[:, :, None] * initial_projection)
    writes = torch.linalg.solve_triangular(
        system,
        rhs.permute(1, 0, 2).contiguous(),
        upper=False,
    )

    ancestor_or_self = ancestor.to(torch.float32) + torch.eye(n, dtype=torch.float32, device=q_f.device)
    coeff = ancestor_or_self.unsqueeze(0) * ratio
    states = (
        gamma[:, :, None, None] * prefix_state.unsqueeze(0)
        + torch.einsum("hij,hjv,hjk->ihvk", coeff, writes, k_f.permute(1, 0, 2))
    )
    output = torch.einsum("ihvk,ihk->ihv", states, q_f)

    write_indices = ssm_state_indices.reshape(n, -1)[:, 0].to(torch.long)
    initial_state.index_copy_(0, write_indices, states.to(initial_state.dtype))
    return output.unsqueeze(0).to(v.dtype), states.to(initial_state.dtype)

# LUMO_FA_TREE_DELTA_TRITON: graph-native fused forward verifier kernel.
@triton.jit(do_not_specialize=["N"])
def _lumo_fa_tree_delta_triton_kernel(
    q,
    k,
    v,
    g,
    beta_gated,
    out,
    state,
    ssm_state_indices,
    initial_state_indices,
    parent_indices,
    scale,
    N: tl.int64,
    HK: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BN: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_state_head: tl.constexpr,
    stride_state_value: tl.constexpr,
    stride_state_key: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    HAS_INITIAL_STATE_INDICES: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_hk = i_hv // (HV // HK)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_v[:, None] & m_k[None, :]

    head_state_off = (
        i_hv * stride_state_head
        + o_v[:, None] * stride_state_value
        + o_k[None, :] * stride_state_key
    )
    if HAS_INITIAL_STATE_INDICES:
        prefix_idx = tl.load(initial_state_indices + 0).to(tl.int64)
    else:
        prefix_idx = tl.load(ssm_state_indices + 0).to(tl.int64)
    prefix_h = tl.load(
        state + prefix_idx * stride_state_token + head_state_off,
        mask=m_h,
        other=0.0,
    ).to(tl.float32)

    for i in tl.static_range(0, BN):
        if i < N:
            if i == 0:
                parent_actual = tl.full((), -1, tl.int64)
            else:
                parent_raw = tl.load(parent_indices + i).to(tl.int64)
                parent_actual = tl.where(parent_raw < 0, 0, parent_raw + 1)
            parent_safe = tl.maximum(parent_actual, 0)
            parent_write_idx = tl.load(
                ssm_state_indices + parent_safe * stride_indices_seq
            ).to(tl.int64)
            parent_h = tl.load(
                state + parent_write_idx * stride_state_token + head_state_off,
                mask=m_h,
                other=0.0,
            ).to(tl.float32)
            h = tl.where(parent_actual >= 0, parent_h, prefix_h)

            q_i = tl.load(
                q + (i * HK + i_hk) * K + o_k,
                mask=m_k,
                other=0.0,
            ).to(tl.float32)
            k_i = tl.load(
                k + (i * HK + i_hk) * K + o_k,
                mask=m_k,
                other=0.0,
            ).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                q_i = q_i * tl.rsqrt(tl.sum(q_i * q_i) + 1e-6)
                k_i = k_i * tl.rsqrt(tl.sum(k_i * k_i) + 1e-6)
            q_i = q_i * scale

            g_i = tl.load(g + i * HV + i_hv).to(tl.float32)
            beta_i = tl.load(beta_gated + i * HV + i_hv).to(tl.float32)
            v_i = tl.load(
                v + (i * HV + i_hv) * V + o_v,
                mask=m_v,
                other=0.0,
            ).to(tl.float32)

            h = h * tl.exp(g_i)
            delta_v = (v_i - tl.sum(h * k_i[None, :], axis=1)) * beta_i
            h = h + delta_v[:, None] * k_i[None, :]
            o_i = tl.sum(h * q_i[None, :], axis=1)

            tl.store(
                out + (i * HV + i_hv) * V + o_v,
                o_i,
                mask=m_v,
            )
            write_idx = tl.load(ssm_state_indices + i * stride_indices_seq).to(tl.int64)
            tl.store(
                state + write_idx * stride_state_token + head_state_off,
                h,
                mask=m_h,
            )

def _lumo_fa_tree_delta_triton(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    initial_state_indices: torch.Tensor | None,
    parent_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, None]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    n = int(q.shape[1])
    hk = int(k.shape[2])
    hv = int(v.shape[2])
    key_dim = int(k.shape[3])
    value_dim = int(v.shape[3])
    if hv % hk != 0:
        raise RuntimeError(f"LUMO_FA_TREE_DELTA_TRITON requires value heads divisible by key heads, got {hv=} {hk=}")
    if key_dim > 256:
        raise RuntimeError(f"LUMO_FA_TREE_DELTA_TRITON supports K<=256, got {key_dim}")
    g, beta_gated = fused_gdn_gating(A_log=A_log, a=a, b=b, dt_bias=dt_bias)
    out = torch.empty_like(v)
    bk = triton.next_power_of_2(key_dim)
    bv = min(triton.next_power_of_2(value_dim), 32)
    bn = triton.next_power_of_2(n)
    grid = (triton.cdiv(value_dim, bv), hv)
    _lumo_fa_tree_delta_triton_kernel[grid](
        q,
        k,
        v,
        g,
        beta_gated,
        out,
        initial_state,
        ssm_state_indices,
        initial_state_indices,
        parent_indices,
        key_dim ** -0.5,
        n,
        hk,
        hv,
        key_dim,
        value_dim,
        bk,
        bv,
        bn,
        initial_state.stride(0),
        initial_state.stride(1),
        initial_state.stride(2),
        initial_state.stride(3),
        ssm_state_indices.stride(0),
        initial_state_indices is not None,
        use_qk_l2norm_in_kernel,
        num_warps=4,
    )
    return out, None

# LUMO_FA_ACTIVATION_REPLAY_COMMIT: accepted-path commit is standard linear GDN.
_LUMO_FA_REPLAY_LAYERS = []
_LUMO_FA_REPLAY_LAYER_SETS = {}
_LUMO_FA_REPLAY_ACTIVE_KEY = None

def _lumo_fa_replay_reset_if_first_layer(prefix: str) -> None:
    global _LUMO_FA_REPLAY_LAYERS, _LUMO_FA_REPLAY_ACTIVE_KEY
    if ".layers.0." in prefix:
        _LUMO_FA_REPLAY_LAYERS = []
        _LUMO_FA_REPLAY_ACTIVE_KEY = None

def _lumo_fa_replay_remember(record: dict) -> None:
    global _LUMO_FA_REPLAY_ACTIVE_KEY
    _key = int(record.get("num_tokens") or 0)
    if _LUMO_FA_REPLAY_ACTIVE_KEY is None:
        _LUMO_FA_REPLAY_ACTIVE_KEY = _key
        if _key > 0:
            _LUMO_FA_REPLAY_LAYER_SETS[_key] = _LUMO_FA_REPLAY_LAYERS
    _LUMO_FA_REPLAY_LAYERS.append(record)

def _lumo_fa_activation_replay_commit(accepted_token_count: int) -> None:
    n = int(accepted_token_count)
    if n <= 0:
        _LUMO_FA_REPLAY_LAYERS.clear()
        return
    for rec in list(_LUMO_FA_REPLAY_LAYERS):
        tokens = min(n, int(rec["num_tokens"]))
        if tokens <= 0:
            continue
        module = rec["module"]
        prefix_idx = int(rec["initial_state_indices"].reshape(-1)[0].item())
        device = rec["mixed_qkv_input"].device
        state_idx = torch.tensor([prefix_idx], dtype=torch.int32, device=device)
        state_cols = torch.full((1, tokens), prefix_idx, dtype=torch.int32, device=device)
        mixed = rec["mixed_qkv_input"][:tokens]
        mixed = causal_conv1d_update(
            mixed.transpose(0, 1).unsqueeze(0),
            rec["conv_state"],
            rec["conv_weights"],
            module.conv1d.bias,
            module.activation,
            conv_state_indices=state_idx,
            validate_data=False,
        ).squeeze(0).transpose(0, 1).contiguous()
        q, k, v = module.rearrange_mixed_qkv(mixed)
        fused_sigmoid_gating_delta_rule_update(
            A_log=module.A_log,
            a=rec["a"][:tokens],
            b=rec["b"][:tokens],
            dt_bias=module.dt_bias,
            q=q,
            k=k,
            v=v,
            initial_state=rec["ssm_state"],
            inplace_final_state=True,
            cu_seqlens=None,
            ssm_state_indices=state_cols,
            initial_state_indices=None,
            num_accepted_tokens=None,
            use_qk_l2norm_in_kernel=True,
        )

@CustomOp.register("chunk_gated_delta_rule")
"""
        if old not in text:
            raise RuntimeError('F_a tree-delta helper insertion anchor not found')
        text = text.replace(old, new, 1)
    old = """        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
"""
    new = """        fa_unique_node_mode = bool(getattr(attn_metadata, "fa_unique_node_mode", False))
        fa_unique_expanded_node_mode = bool(getattr(attn_metadata, "fa_unique_expanded_node_mode", False))
        if fa_unique_node_mode:
            try:
                global _LUMO_FA_UNIQUE_GDN_FH
                try:
                    _LUMO_FA_UNIQUE_GDN_FH
                except NameError:
                    _LUMO_FA_UNIQUE_GDN_FH = open("/logs/fa_unique_gdn_debug.jsonl", "a", buffering=1)
                _LUMO_FA_UNIQUE_GDN_FH.write(_lumo_fa_json.dumps({
                    "ts": round(_lumo_fa_time.time(), 4),
                    "event": "fa_unique_gdn_layer",
                    "layer": self.prefix,
                    "num_actual_tokens": int(num_actual_tokens),
                    "num_spec_decodes": int(attn_metadata.num_spec_decodes),
                    "num_spec_decode_tokens": int(attn_metadata.num_spec_decode_tokens),
                    "expanded_parent_state_rows": bool(fa_unique_expanded_node_mode),
                    "mixed_qkv_shape": list(mixed_qkv.shape),
                    "b_shape": list(b.shape),
                    "state_rows": list(spec_state_indices_tensor.shape) if spec_state_indices_tensor is not None else None,
                    "initial_rows": (
                        list(locals().get("spec_initial_state_indices_tensor").shape)
                        if locals().get("spec_initial_state_indices_tensor") is not None
                        else None
                    ),
                    "parents": (
                        None
                        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
                        else (
                            getattr(attn_metadata, "fa_tree_parent_indices_tensor", None)
                            .detach().cpu().tolist()
                            if getattr(attn_metadata, "fa_tree_parent_indices_tensor", None) is not None
                            else None
                        )
                    ),
                }) + chr(10))
            except Exception:
                pass

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
"""
    if old not in text:
        raise RuntimeError('F_a unique-node gdn_linear telemetry anchor not found')
    text = text.replace(old, new, 1)

    old = """        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=(
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
            )
"""
    new = """        # 1.1: Process the multi-query part
        _lumo_fa_replay_mixed_qkv_input = mixed_qkv_spec
        if spec_sequence_masks is not None and fa_unique_expanded_node_mode:
            _depth_rows = getattr(attn_metadata, "fa_tree_depth_rows", None)
            _depth_row_tensors = getattr(attn_metadata, "fa_tree_depth_row_tensors", None)
            _depth_query_start_tensors = getattr(attn_metadata, "fa_tree_depth_query_start_tensors", None)
            if _depth_rows is None:
                _parents_t = getattr(attn_metadata, "fa_tree_parent_indices_tensor", None)
                _parents = _parents_t.detach().cpu().tolist() if _parents_t is not None else []
                _depths = []
                for _i, _parent in enumerate(_parents):
                    if _i == 0:
                        _depths.append(0)
                    else:
                        _depths.append(1 if int(_parent) < 0 else _depths[int(_parent) + 1] + 1)
                _depth_rows = tuple(
                    tuple(i for i, d in enumerate(_depths) if d == depth)
                    for depth in range((max(_depths) + 1) if _depths else 0)
                )
                _depth_row_tensors = tuple(
                    torch.tensor(_rows, dtype=torch.long, device=mixed_qkv_spec.device)
                    for _rows in _depth_rows
                )
                _depth_query_start_tensors = tuple(
                    torch.arange(len(_rows) + 1, dtype=torch.int32, device=mixed_qkv_spec.device)
                    for _rows in _depth_rows
                )
            _conv_out = torch.empty_like(mixed_qkv_spec)
            for _rows, _row_idx, _sub_query_start in zip(
                _depth_rows, _depth_row_tensors, _depth_query_start_tensors):
                if not _rows:
                    continue
                _sub = causal_conv1d_update(
                    mixed_qkv_spec.index_select(0, _row_idx),
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=spec_state_indices_tensor.index_select(0, _row_idx),
                    num_accepted_tokens=num_accepted_tokens.index_select(0, _row_idx),
                    query_start_loc=_sub_query_start,
                    max_query_len=1,
                    block_idx_last_scheduled_token=spec_write_state_slot_tensor.index_select(0, _row_idx),
                    initial_state_idx=spec_write_state_slot_tensor.index_select(0, _row_idx),
                    initial_state_indices=spec_initial_state_indices_tensor.index_select(0, _row_idx),
                    validate_data=False,
                )
                _conv_out.index_copy_(0, _row_idx, _sub)
            mixed_qkv_spec = _conv_out
        elif spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=(
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
            )
"""
    if old not in text:
        print('[TRACK-B-PRELAUNCH] skip F_a expanded-node conv patch; F_b kernel-row hook absent')
    else:
        text = text.replace(old, new, 1)

    old = """        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )
"""
    new = """        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )
        if fa_unique_node_mode:
            try:
                _LUMO_FA_UNIQUE_GDN_FH.write(_lumo_fa_json.dumps({
                    "ts": round(_lumo_fa_time.time(), 4),
                    "event": "fa_unique_gdn_shapes",
                    "layer": self.prefix,
                    "expanded_parent_state_rows": bool(fa_unique_expanded_node_mode),
                    "query_spec_shape": list(query_spec.shape) if query_spec is not None else None,
                    "key_spec_shape": list(key_spec.shape) if key_spec is not None else None,
                    "value_spec_shape": list(value_spec.shape) if value_spec is not None else None,
                    "a_shape": list(a.shape),
                    "b_shape": list(b.shape),
                }) + chr(10))
            except Exception:
                pass
            if (
                fa_unique_expanded_node_mode
                and _lumo_fa_os.environ.get("LUMO_FA_ACTIVATION_REPLAY_COMMIT", "1") == "1"
                and spec_initial_state_indices_tensor is not None
                and query_spec is not None
            ):
                try:
                    _record_group_size = 0
                    _record_req_count = 0
                    try:
                        _depth_rows_for_record = getattr(attn_metadata, "fa_tree_depth_rows", None)
                        if _depth_rows_for_record is not None:
                            _record_group_size = int(len(_depth_rows_for_record))
                        if _record_group_size <= 0:
                            _record_group_size = int(_lumo_fa_os.environ.get("LUMO_FA_TREE_GROUP_SIZE", "4"))
                        _total_tree_rows = int(attn_metadata.num_spec_decode_tokens)
                        if _record_group_size > 0 and _total_tree_rows % _record_group_size == 0:
                            _record_req_count = int(_total_tree_rows // _record_group_size)
                    except Exception:
                        _record_group_size = 0
                        _record_req_count = 0
                    _lumo_fa_replay_reset_if_first_layer(self.prefix)
                    _lumo_fa_replay_remember({
                        "module": self,
                        "num_tokens": int(attn_metadata.num_spec_decode_tokens),
                        "tree_group_size": int(_record_group_size),
                        "tree_req_count": int(_record_req_count),
                        "mixed_qkv_input": _lumo_fa_replay_mixed_qkv_input.detach(),
                        "a": a.detach(),
                        "b": b.detach(),
                        "conv_state": conv_state,
                        "conv_prefix_state": conv_state.index_select(
                            0, spec_initial_state_indices_tensor.reshape(-1)[:1].to(torch.long)
                        ).squeeze(0).detach().clone(),
                        "conv_weights": conv_weights,
                        "ssm_state": ssm_state,
                        "ssm_prefix_state": ssm_state.index_select(
                            0, spec_initial_state_indices_tensor.reshape(-1)[:1].to(torch.long)
                        ).squeeze(0).detach().clone(),
                        "initial_state_indices": spec_initial_state_indices_tensor.detach(),
                    })
                except Exception:
                    pass
"""
    if old not in text:
        raise RuntimeError('F_a unique-node shape telemetry anchor not found')
    text = text.replace(old, new, 1)

    old = """        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[
                        : attn_metadata.num_spec_decodes + 1
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    initial_state_indices=spec_initial_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None
"""
    new = """        # 2.1: Process the multi-query part
        if (
            spec_sequence_masks is not None
            and fa_unique_expanded_node_mode
            and (
                _lumo_fa_os.environ.get("LUMO_FA_TREE_DELTA_TORCH") == "1"
                or _lumo_fa_os.environ.get("LUMO_FA_TREE_DELTA_TRITON") == "1"
            )
        ):
            def _lumo_fa_select_token(_tensor, _idx):
                if _tensor is None:
                    return None
                _tree_rows = int(query_spec.shape[1])
                if _tensor.ndim >= 3 and _tensor.shape[1] == attn_metadata.num_spec_decode_tokens:
                    return _tensor.index_select(1, _idx)
                if _tensor.ndim >= 1 and _tensor.shape[0] == attn_metadata.num_spec_decode_tokens:
                    return _tensor.index_select(0, _idx)
                if _tensor.ndim >= 1 and int(_tensor.shape[0]) == int(num_actual_tokens) and int(_tensor.shape[0]) >= _tree_rows:
                    return _tensor.narrow(0, int(_tensor.shape[0]) - _tree_rows, _tree_rows).index_select(0, _idx)
                return _tensor

            _parents_t = getattr(attn_metadata, "fa_tree_parent_indices_tensor", None)
            if _parents_t is None:
                raise RuntimeError("LUMO_FA_TREE_DELTA_TORCH requires fa_tree_parent_indices_tensor")
            _lumo_tree_delta_impl = (
                _lumo_fa_tree_delta_triton
                if _lumo_fa_os.environ.get("LUMO_FA_TREE_DELTA_TRITON") == "1"
                else _lumo_fa_tree_delta_torch
            )
            _all_rows = torch.arange(
                query_spec.shape[1],
                dtype=torch.long,
                device=query_spec.device,
            )
            _tree_a = _lumo_fa_select_token(a, _all_rows)
            _tree_b = _lumo_fa_select_token(b, _all_rows)
            core_attn_out_spec, last_recurrent_state = _lumo_tree_delta_impl(
                A_log=self.A_log,
                a=_tree_a,
                b=_tree_b,
                dt_bias=self.dt_bias,
                q=query_spec,
                k=key_spec,
                v=value_spec,
                initial_state=ssm_state,
                ssm_state_indices=spec_state_indices_tensor,
                initial_state_indices=spec_initial_state_indices_tensor,
                parent_indices=_parents_t,
                use_qk_l2norm_in_kernel=True,
            )
        elif spec_sequence_masks is not None and fa_unique_expanded_node_mode:
            def _lumo_fa_select_token(_tensor, _idx):
                if _tensor is None:
                    return None
                _tree_rows = int(query_spec.shape[1])
                if _tensor.ndim >= 3 and _tensor.shape[1] == attn_metadata.num_spec_decode_tokens:
                    return _tensor.index_select(1, _idx)
                if _tensor.ndim >= 1 and _tensor.shape[0] == attn_metadata.num_spec_decode_tokens:
                    return _tensor.index_select(0, _idx)
                if _tensor.ndim >= 1 and int(_tensor.shape[0]) == int(num_actual_tokens) and int(_tensor.shape[0]) >= _tree_rows:
                    return _tensor.narrow(0, int(_tensor.shape[0]) - _tree_rows, _tree_rows).index_select(0, _idx)
                return _tensor

            _parents_t = getattr(attn_metadata, "fa_tree_parent_indices_tensor", None)
            _parents = _parents_t.detach().cpu().tolist() if _parents_t is not None else []
            _depths = []
            for _i, _parent in enumerate(_parents):
                if _i == 0:
                    _depths.append(0)
                else:
                    _depths.append(1 if int(_parent) < 0 else _depths[int(_parent) + 1] + 1)
            core_attn_out_spec = None
            last_recurrent_state = None
            for _depth in range((max(_depths) + 1) if _depths else 0):
                _rows = [i for i, d in enumerate(_depths) if d == _depth]
                if not _rows:
                    continue
                _row_idx = torch.tensor(_rows, dtype=torch.long, device=query_spec.device)
                _sub_query_start = torch.arange(
                    len(_rows) + 1, dtype=torch.int32, device=query_spec.device)
                _sub_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=_lumo_fa_select_token(a, _row_idx),
                    b=_lumo_fa_select_token(b, _row_idx),
                    dt_bias=self.dt_bias,
                    q=query_spec.index_select(1, _row_idx),
                    k=key_spec.index_select(1, _row_idx),
                    v=value_spec.index_select(1, _row_idx),
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=_sub_query_start,
                    ssm_state_indices=spec_state_indices_tensor.index_select(0, _row_idx),
                    initial_state_indices=spec_initial_state_indices_tensor.index_select(0, _row_idx),
                    num_accepted_tokens=num_accepted_tokens.index_select(0, _row_idx),
                    use_qk_l2norm_in_kernel=True,
                )
                if core_attn_out_spec is None:
                    core_attn_out_spec = torch.empty(
                        (1, attn_metadata.num_spec_decode_tokens, *_sub_out.shape[2:]),
                        dtype=_sub_out.dtype, device=_sub_out.device)
                core_attn_out_spec.index_copy_(1, _row_idx, _sub_out)
        elif spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[
                        : attn_metadata.num_spec_decodes + 1
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    initial_state_indices=spec_initial_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None
"""
    if old not in text:
        print('[TRACK-B-PRELAUNCH] skip F_a expanded-node SSM patch; F_b kernel-row hook absent')
    else:
        text = text.replace(old, new, 1)

    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_a unique-node GDN linear telemetry patch')

gm = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = gm.read_text()
sentinel = '# LUMO_FA_ACTIVATION_REPLAY_COMMIT_RUNNER'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_a activation replay commit runner patch already present')
else:
    patch = r"""

# LUMO_FA_ACTIVATION_REPLAY_COMMIT_RUNNER: after tree verification samples the
# accepted linear path, replay that path through the standard GDN update.
import os as _lumo_fa_replay_os
import json as _lumo_fa_replay_json
import time as _lumo_fa_replay_time
from vllm.model_executor.layers.mamba import gdn_linear_attn as _lumo_fa_replay_gdn

_lumo_fa_replay_prev_sample_tokens = GPUModelRunner.sample_tokens

def _lumo_fa_replay_sample_tokens(self, grammar_output):
    out = _lumo_fa_replay_prev_sample_tokens(self, grammar_output)
    if (_lumo_fa_replay_os.environ.get("LUMO_FA_UNIQUE_NODES") == "1"
            and _lumo_fa_replay_os.environ.get("LUMO_FA_ACTIVATION_REPLAY_COMMIT", "1") == "1"):
        try:
            model_output = getattr(out, "model_runner_output", out)
            samples = list(getattr(model_output, "sampled_token_ids", []) or [])
            accepted_len = 0
            if samples:
                toks = samples[0]
                accepted_len = sum(1 for tok in list(toks) if int(tok) >= 0)
            if accepted_len > 0:
                _lumo_fa_replay_gdn._lumo_fa_activation_replay_commit(accepted_len)
            try:
                fh = globals().get("_LUMO_FA_REPLAY_COMMIT_FH")
                if fh is None:
                    fh = open("/logs/fa_activation_replay_commit.jsonl", "a", buffering=1)
                    globals()["_LUMO_FA_REPLAY_COMMIT_FH"] = fh
                fh.write(_lumo_fa_replay_json.dumps({
                    "ts": round(_lumo_fa_replay_time.time(), 4),
                    "event": "fa_activation_replay_commit",
                    "accepted_len": int(accepted_len),
                    "sample_head": [int(x) for x in (list(samples[0])[:8] if samples else [])],
                }) + chr(10))
            except Exception:
                pass
        except Exception as exc:
            try:
                fh = globals().get("_LUMO_FA_REPLAY_COMMIT_FH")
                if fh is None:
                    fh = open("/logs/fa_activation_replay_commit.jsonl", "a", buffering=1)
                    globals()["_LUMO_FA_REPLAY_COMMIT_FH"] = fh
                fh.write(_lumo_fa_replay_json.dumps({
                    "ts": round(_lumo_fa_replay_time.time(), 4),
                    "event": "fa_activation_replay_commit_error",
                    "error": repr(exc),
                }) + chr(10))
            except Exception:
                pass
    return out

GPUModelRunner.sample_tokens = _lumo_fa_replay_sample_tokens
"""
    gm.write_text(text + patch)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_a activation replay commit runner patch')
LUMOFAUNIQUENODES
'''

_FA_UNIQUE_BATCH4_DIAG_BLOCK = r'''
python3 - <<'LUMOFAUNIQUEBATCH4DIAG'
from pathlib import Path

gl = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py')
text = gl.read_text()
sentinel = '# LUMO_FA_ACTIVATION_REPLAY_BATCH4_DIAG'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_a activation replay batch4 diag already present')
else:
    start = text.find('def _lumo_fa_activation_replay_commit(accepted_token_count: int) -> None:\n')
    if start < 0:
        start = text.find('def _lumo_fa_activation_replay_commit(accepted_token_count) -> None:\n')
    end = text.find('\n@CustomOp.register("chunk_gated_delta_rule")\n', start)
    if start < 0 or end < 0:
        raise RuntimeError('F_a activation replay commit function anchor not found')
    new = r"""def _lumo_fa_activation_replay_commit(
    accepted_token_count,
    expected_total_tokens=None,
    expected_req_count=None,
) -> None:
    if isinstance(accepted_token_count, (list, tuple)):
        accepted_counts = [int(x) for x in accepted_token_count]
    else:
        try:
            accepted_counts = [int(x) for x in accepted_token_count.tolist()]
        except Exception:
            accepted_counts = [int(accepted_token_count)]
    if not accepted_counts or max(accepted_counts) <= 0:
        _LUMO_FA_REPLAY_LAYERS.clear()
        return
    try:
        _fh = globals().get("_LUMO_FA_REPLAY_COMMIT_DETAIL_FH")
        if _fh is None:
            _fh = open("/logs/fa_activation_replay_commit_detail.jsonl", "a", buffering=1)
            globals()["_LUMO_FA_REPLAY_COMMIT_DETAIL_FH"] = _fh
        _fh.write(_lumo_fa_json.dumps({
            "ts": round(_lumo_fa_time.time(), 4),
            "event": "fa_activation_replay_commit_detail",
            "accepted_counts": accepted_counts,
            "record_count": len(_LUMO_FA_REPLAY_LAYERS),
            "expected_total_tokens": expected_total_tokens,
            "expected_req_count": expected_req_count,
        }) + chr(10))
    except Exception:
        pass
    replay_layers = _LUMO_FA_REPLAY_LAYERS
    try:
        _expected_total = int(expected_total_tokens or 0)
        if _expected_total > 0 and _LUMO_FA_REPLAY_LAYER_SETS:
            _candidate_keys = sorted(
                int(k) for k in _LUMO_FA_REPLAY_LAYER_SETS
                if int(k) >= _expected_total
            )
            if not _candidate_keys and _expected_total in _LUMO_FA_REPLAY_LAYER_SETS:
                _candidate_keys = [_expected_total]
            if _candidate_keys:
                replay_layers = _LUMO_FA_REPLAY_LAYER_SETS[_candidate_keys[0]]
    except Exception:
        replay_layers = _LUMO_FA_REPLAY_LAYERS
    for rec in list(replay_layers):
        total_tokens = int(rec["num_tokens"])
        if total_tokens <= 0:
            continue
        record_group_size = int(rec.get("tree_group_size") or 0)
        if record_group_size > 0 and total_tokens % record_group_size == 0:
            group_size = record_group_size
        elif len(accepted_counts) > 1 and total_tokens % len(accepted_counts) == 0:
            group_size = max(1, total_tokens // len(accepted_counts))
        else:
            group_size = total_tokens
        record_req_count = int(rec.get("tree_req_count") or 0)
        req_count = max(1, total_tokens // group_size)
        if record_req_count > 0:
            req_count = min(req_count, record_req_count)
        if expected_req_count is not None:
            try:
                req_count = min(req_count, int(expected_req_count))
            except Exception:
                pass
        req_count = min(req_count, len(accepted_counts))
        module = rec["module"]
        device = rec["mixed_qkv_input"].device
        initial_flat = rec["initial_state_indices"].reshape(-1)
        try:
            _fh = globals().get("_LUMO_FA_REPLAY_COMMIT_DETAIL_FH")
            if _fh is not None:
                _fh.write(_lumo_fa_json.dumps({
                    "ts": round(_lumo_fa_time.time(), 4),
                    "event": "fa_activation_replay_record",
                    "total_tokens": total_tokens,
                    "group_size": int(group_size),
                    "req_count": int(req_count),
                    "record_group_size": int(record_group_size),
                    "record_req_count": int(record_req_count),
                    "initial_rows": int(initial_flat.numel()),
                    "mixed_shape": list(rec["mixed_qkv_input"].shape),
                }) + chr(10))
        except Exception:
            pass
        for req_i in range(req_count):
            n = int(accepted_counts[req_i])
            tokens = min(n, group_size, total_tokens - req_i * group_size)
            if tokens <= 0:
                continue
            base = req_i * group_size
            prefix_idx = int(initial_flat[base].item())
            state_idx = torch.tensor([prefix_idx], dtype=torch.int32, device=device)
            state_cols = torch.full((1, tokens), prefix_idx, dtype=torch.int32, device=device)
            mixed = rec["mixed_qkv_input"][base:base + tokens]
            mixed = causal_conv1d_update(
                mixed.transpose(0, 1).unsqueeze(0),
                rec["conv_state"],
                rec["conv_weights"],
                module.conv1d.bias,
                module.activation,
                conv_state_indices=state_idx,
                validate_data=False,
            ).squeeze(0).transpose(0, 1).contiguous()
            q, k, v = module.rearrange_mixed_qkv(mixed)
            fused_sigmoid_gating_delta_rule_update(
                A_log=module.A_log,
                a=rec["a"][base:base + tokens],
                b=rec["b"][base:base + tokens],
                dt_bias=module.dt_bias,
                q=q,
                k=k,
                v=v,
                initial_state=rec["ssm_state"],
                inplace_final_state=True,
                cu_seqlens=None,
                ssm_state_indices=state_cols,
                initial_state_indices=None,
                num_accepted_tokens=None,
                use_qk_l2norm_in_kernel=True,
            )
"""
    text = text[:start] + new + text[end:]
    text = sentinel + '\n' + text
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_a activation replay batch4 diag')

gm = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = gm.read_text()
sentinel = '# LUMO_FA_REPLAY_COMMIT_BATCH4_RUNNER_DIAG'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_a replay commit batch4 runner diag already present')
else:
    old = """            accepted_len = 0
            if samples:
                toks = samples[0]
                accepted_len = sum(1 for tok in list(toks) if int(tok) >= 0)
            if accepted_len > 0:
                _lumo_fa_replay_gdn._lumo_fa_activation_replay_commit(accepted_len)
"""
    new = """            accepted_lens = []
            for toks in samples:
                accepted_lens.append(sum(1 for tok in list(toks) if int(tok) >= 0))
            accepted_len = accepted_lens[0] if accepted_lens else 0
            expected_total_tokens = getattr(
                _lumo_fa_replay_gdn, "_LUMO_FA_LAST_TREE_TOTAL_ROWS", None)
            expected_req_count = getattr(
                _lumo_fa_replay_gdn, "_LUMO_FA_LAST_TREE_REQ_COUNT", None)
            if accepted_lens and max(accepted_lens) > 0:
                _lumo_fa_replay_gdn._lumo_fa_activation_replay_commit(
                    accepted_lens,
                    expected_total_tokens=expected_total_tokens,
                    expected_req_count=expected_req_count,
                )
"""
    if old not in text:
        raise RuntimeError('F_a replay commit runner accepted_len anchor not found')
    text = text.replace(old, new, 1)
    text = text.replace(
        """                    "accepted_len": int(accepted_len),
                    "sample_head": [int(x) for x in (list(samples[0])[:8] if samples else [])],
""",
        """                    "accepted_len": int(accepted_len),
                    "accepted_lens": [int(x) for x in accepted_lens],
                    "expected_total_tokens": expected_total_tokens,
                    "expected_req_count": expected_req_count,
                    "num_samples": int(len(samples)),
                    "sample_heads": [[int(x) for x in list(toks)[:8]] for toks in samples[:8]],
                    "sample_head": [int(x) for x in (list(samples[0])[:8] if samples else [])],
""",
        1,
    )
    text = sentinel + '\n' + text
    gm.write_text(text)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_a replay commit batch4 runner diag')
LUMOFAUNIQUEBATCH4DIAG
'''

_FA_REPLAY_STATE_COPY_COMMIT_BLOCK = r'''
python3 - <<'LUMOFASTATECOPYCOMMIT'
from pathlib import Path

gl = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py')
text = gl.read_text()
sentinel = '# LUMO_FA_ACTIVATION_REPLAY_STATE_COPY_COMMIT'
changed = False
record_anchor = '                        "initial_state_indices": spec_initial_state_indices_tensor.detach(),\n'
if '"state_indices": spec_state_indices_tensor.detach(),' not in text and record_anchor in text:
    text = text.replace(
        record_anchor,
        '                        "state_indices": spec_state_indices_tensor.detach(),\n'
        + record_anchor,
        1,
    )
    changed = True
start = text.find('def _lumo_fa_activation_replay_commit(\n')
if start < 0:
    start = text.find('def _lumo_fa_activation_replay_commit(accepted_token_count: int) -> None:\n')
if start < 0:
    start = text.find('def _lumo_fa_activation_replay_commit(accepted_token_count) -> None:\n')
end = text.find('\n@CustomOp.register("chunk_gated_delta_rule")\n', start)
if start < 0 or end < 0:
    raise RuntimeError('F_a activation replay commit function anchor not found for state-copy commit')
if sentinel not in text[start:end]:
    new = r"""def _lumo_fa_activation_replay_commit(
    accepted_token_count,
    expected_total_tokens=None,
    expected_req_count=None,
) -> None:
    # LUMO_FA_ACTIVATION_REPLAY_STATE_COPY_COMMIT: tree verification already
    # materialized the accepted row's conv and GDN recurrent states. Roll both
    # caches back to that row instead of replaying a second recurrence.
    _commit_t0 = _lumo_fa_time.perf_counter()
    if isinstance(accepted_token_count, (list, tuple)):
        accepted_counts = [int(x) for x in accepted_token_count]
    else:
        try:
            accepted_counts = [int(x) for x in accepted_token_count.tolist()]
        except Exception:
            accepted_counts = [int(accepted_token_count)]
    if not accepted_counts or max(accepted_counts) <= 0:
        _LUMO_FA_REPLAY_LAYERS.clear()
        return
    try:
        _fh = globals().get("_LUMO_FA_REPLAY_COMMIT_DETAIL_FH")
        if _fh is None:
            _fh = open("/logs/fa_activation_replay_commit_detail.jsonl", "a", buffering=1)
            globals()["_LUMO_FA_REPLAY_COMMIT_DETAIL_FH"] = _fh
        _fh.write(_lumo_fa_json.dumps({
            "ts": round(_lumo_fa_time.time(), 4),
            "event": "fa_activation_replay_commit_detail",
            "commit_mode": "state_copy",
            "accepted_counts": accepted_counts,
            "record_count": len(_LUMO_FA_REPLAY_LAYERS),
            "expected_total_tokens": expected_total_tokens,
            "expected_req_count": expected_req_count,
        }) + chr(10))
    except Exception:
        pass
    replay_layers = _LUMO_FA_REPLAY_LAYERS
    try:
        _expected_total = int(expected_total_tokens or 0)
        if _expected_total > 0 and _LUMO_FA_REPLAY_LAYER_SETS:
            _candidate_keys = sorted(
                int(k) for k in _LUMO_FA_REPLAY_LAYER_SETS
                if int(k) >= _expected_total
            )
            if not _candidate_keys and _expected_total in _LUMO_FA_REPLAY_LAYER_SETS:
                _candidate_keys = [_expected_total]
            if _candidate_keys:
                replay_layers = _LUMO_FA_REPLAY_LAYER_SETS[_candidate_keys[0]]
    except Exception:
        replay_layers = _LUMO_FA_REPLAY_LAYERS
    copied = 0
    missing_state_indices = 0
    for rec in list(replay_layers):
        total_tokens = int(rec["num_tokens"])
        if total_tokens <= 0:
            continue
        record_group_size = int(rec.get("tree_group_size") or 0)
        if record_group_size > 0 and total_tokens % record_group_size == 0:
            group_size = record_group_size
        elif len(accepted_counts) > 1 and total_tokens % len(accepted_counts) == 0:
            group_size = max(1, total_tokens // len(accepted_counts))
        else:
            group_size = total_tokens
        record_req_count = int(rec.get("tree_req_count") or 0)
        req_count = max(1, total_tokens // group_size)
        if record_req_count > 0:
            req_count = min(req_count, record_req_count)
        if expected_req_count is not None:
            try:
                req_count = min(req_count, int(expected_req_count))
            except Exception:
                pass
        req_count = min(req_count, len(accepted_counts))
        initial_flat = rec["initial_state_indices"].reshape(-1).to(torch.long)
        state_indices = rec.get("state_indices")
        if state_indices is None:
            missing_state_indices += 1
            continue
        state_flat = state_indices.reshape(total_tokens, -1)[:, 0].to(torch.long)
        try:
            _fh = globals().get("_LUMO_FA_REPLAY_COMMIT_DETAIL_FH")
            if _fh is not None:
                _fh.write(_lumo_fa_json.dumps({
                    "ts": round(_lumo_fa_time.time(), 4),
                    "event": "fa_activation_replay_record",
                    "commit_mode": "state_copy",
                    "total_tokens": total_tokens,
                    "group_size": int(group_size),
                    "req_count": int(req_count),
                    "record_group_size": int(record_group_size),
                    "record_req_count": int(record_req_count),
                    "initial_rows": int(initial_flat.numel()),
                    "state_rows": list(state_indices.shape),
                    "mixed_shape": list(rec["mixed_qkv_input"].shape),
                }) + chr(10))
        except Exception:
            pass
        for req_i in range(req_count):
            n = int(accepted_counts[req_i])
            tokens = min(n, group_size, total_tokens - req_i * group_size)
            if tokens <= 0:
                continue
            base = req_i * group_size
            final_row = base + tokens - 1
            prefix_idx = int(initial_flat[base].item())
            final_idx = int(state_flat[final_row].item())
            if final_idx != prefix_idx:
                rec["conv_state"][prefix_idx].copy_(rec["conv_state"][final_idx], non_blocking=True)
                rec["ssm_state"][prefix_idx].copy_(rec["ssm_state"][final_idx], non_blocking=True)
            copied += 1
    try:
        _fh = globals().get("_LUMO_FA_REPLAY_COMMIT_DETAIL_FH")
        if _fh is not None:
            _fh.write(_lumo_fa_json.dumps({
                "ts": round(_lumo_fa_time.time(), 4),
                "event": "fa_activation_replay_commit_summary",
                "commit_mode": "state_copy",
                "accepted_counts": accepted_counts,
                "copied_requests": int(copied),
                "missing_state_indices": int(missing_state_indices),
                "commit_enqueue_us": int((_lumo_fa_time.perf_counter() - _commit_t0) * 1000000),
            }) + chr(10))
    except Exception:
        pass
"""
    text = text[:start] + new + text[end:]
    changed = True
if changed:
    text = sentinel + '\n' + text
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_a activation replay state-copy commit')
else:
    print('[TRACK-B-PRELAUNCH] F_a activation replay state-copy commit already present')
LUMOFASTATECOPYCOMMIT
'''

_FA_UNIQUE_BATCH4_PACK_BLOCK = r'''
python3 - <<'LUMOFAUNIQUEBATCH4PACK'
from pathlib import Path

ga = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py')
text = ga.read_text()
sentinel = '# LUMO_FA_UNIQUE_BATCH4_PACK'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_a unique-node batch4 pack already present')
else:
    start = text.find('            if _tree_src and int(num_spec_decodes) == 1:\n')
    end = text.find('            elif spec_state_indices_tensor is not None and int(num_spec_decodes) == 1:\n', start)
    if start < 0 or end < 0:
        raise RuntimeError('F_a unique-node batch4 metadata anchor not found')
    new = r"""            if _tree_src and int(num_spec_decodes) >= 1:
                _pre_num_spec_decodes = int(num_spec_decodes)
                _choices = list(_lumo_fa_ast.literal_eval(_tree_src))
                _node_count = len(_choices)
                _group_size = _node_count + 1
                _path_to_idx = {tuple(_p): _i for _i, _p in enumerate(_choices)}
                _parents = [
                    _path_to_idx.get(tuple(_p[:-1]), -1)
                    for _p in _choices
                ]
                _max_depth = max((len(tuple(_p)) for _p in _choices), default=0)
                _is_spine = (
                    _node_count == _max_depth
                    and all(tuple(_p) == tuple([0] * len(tuple(_p)))
                            for _p in _choices)
                )
                if block_table_tensor.size(1) < _node_count + 2:
                    raise RuntimeError(
                        "LUMO_FA_UNIQUE_NODES requires root + node write "
                        f"state slots: need {_node_count + 2}, got "
                        f"{block_table_tensor.size(1)}")
                _rows = block_table_tensor[
                    spec_sequence_masks, :_node_count + 2
                ].contiguous()
                _req_count = int(_rows.shape[0])
                _write_slots_2d = _rows[:, 1:_node_count + 2].contiguous()
                _initial_slots_2d = torch.empty(
                    (_req_count, _group_size), dtype=torch.int32,
                    device=query_start_loc.device)
                _initial_slots_2d[:, 0] = _rows[:, 0]
                for _i, _parent in enumerate(_parents):
                    _initial_slots_2d[:, _i + 1] = _write_slots_2d[
                        :, 0 if _parent < 0 else _parent + 1
                    ]
                spec_initial_state_indices_tensor = (
                    _initial_slots_2d.reshape(-1).contiguous()
                )
                spec_initial_state_slot_tensor = None
                spec_write_state_slot_tensor = torch.zeros(
                    (_req_count * _group_size,), dtype=torch.int32,
                    device=query_start_loc.device)
                spec_state_indices_tensor = (
                    _write_slots_2d.reshape(-1, 1).contiguous()
                )
                spec_query_start_loc = torch.arange(
                    _req_count * _group_size + 1, dtype=torch.int32,
                    device=query_start_loc.device)
                num_spec_decodes = _req_count * _group_size
                num_spec_decode_tokens = _req_count * _group_size
                if _lumo_fa_replay_gdn is not None:
                    try:
                        _lumo_fa_replay_gdn._LUMO_FA_LAST_TREE_TOTAL_ROWS = int(num_spec_decode_tokens)
                        _lumo_fa_replay_gdn._LUMO_FA_LAST_TREE_REQ_COUNT = int(_req_count)
                    except Exception:
                        pass
                num_accepted_tokens = torch.ones(
                    (_req_count * _group_size,), dtype=torch.int32,
                    device=query_start_loc.device)
                fa_unique_expanded_node_mode = True
                _local_actual_parents = [-1]
                for _parent in _parents:
                    _local_actual_parents.append(
                        0 if int(_parent) < 0 else int(_parent) + 1
                    )
                _all_actual_parents = []
                for _req_i in range(_req_count):
                    _base = _req_i * _group_size
                    for _parent in _local_actual_parents:
                        _all_actual_parents.append(
                            -1 if int(_parent) < 0 else _base + int(_parent)
                        )
                fa_tree_parent_indices_tensor = torch.tensor(
                    _all_actual_parents, dtype=torch.int32,
                    device=query_start_loc.device)
                _depths = []
                for _parent in _all_actual_parents:
                    _depths.append(0 if int(_parent) < 0 else _depths[int(_parent)] + 1)
                fa_tree_depth_rows = tuple(
                    tuple(_i for _i, _d in enumerate(_depths) if _d == _depth)
                    for _depth in range(max(_depths) + 1)
                )
                fa_tree_depth_row_tensors = tuple(
                    torch.tensor(_rows, dtype=torch.long, device=query_start_loc.device)
                    for _rows in fa_tree_depth_rows
                )
                fa_tree_depth_query_start_tensors = tuple(
                    torch.arange(len(_rows) + 1, dtype=torch.int32, device=query_start_loc.device)
                    for _rows in fa_tree_depth_rows
                )
                fa_unique_node_mode = True
                try:
                    _fh = globals().get("_LUMO_FA_UNIFIED_FH")
                    if _fh is None:
                        _fh = open("/logs/fb_debug.jsonl", "a", buffering=1)
                        globals()["_LUMO_FA_UNIFIED_FH"] = _fh
                    _fh.write(_lumo_fa_json.dumps({
                        "ts": round(_lumo_fa_time.time(), 4),
                        "event": "round_f_unified_step",
                        "stage": "stage3_spine_only_unique_node_state_tree" if _is_spine else "stage3_unique_node_state_tree_expanded",
                        "component_under_test": "fa_unique_node_state_tree_verifier",
                        "verifier_path": "LUMO_FA_UNIQUE_NODES/batch_packed_parent_state_rows",
                        "pre_num_spec_decodes": int(_pre_num_spec_decodes),
                        "request_count": int(_req_count),
                        "tree_group_size": int(_group_size),
                        "internal_rows_enabled": False,
                        "kernel_rows_enabled": False,
                        "no_kv_prefix_copy_enabled": True,
                        "candidate_pool_nodes": int(_node_count),
                        "selected_nodes": int(_node_count),
                        "verified_nodes": int(_node_count),
                        "unique_tree_nodes": int(_node_count),
                        "trimmed_nodes": 0,
                        "max_depth": int(_max_depth),
                        "sources": {"mtp_top1": int(_node_count), "mtp_alt": 0, "suffix": 0},
                        "path_rows": 0,
                        "scheduler_visible_clone_requests": 0,
                        "prefix_kv_copy_bytes": 0,
                        "recomputed_shared_prefix_nodes": 0,
                        "extra_proposer_for_trimmed_nodes": 0,
                        "accepted_path_commit_only": True,
                        "tree_attention": False,
                        "gdn_parent_gather": True,
                        "depth_positions": True,
                        "tree_sampler": False,
                        "top1_spine_accept_depth": None,
                        "accepted_depth": None,
                        "accepted_node_path": [],
                        "estimated_event_ms": None,
                        "event_budget_ms": None,
                        "tree_score": None,
                        "proposer_us": 0,
                        "trim_us": 0,
                        "verify_us": 0,
                        "tree_attention_us": 0,
                        "gdn_parent_gather_us": 0,
                        "depth_sync_us": 0,
                        "commit_us": 0,
                        "gdn_state_bytes_copied": 0,
                        "kv_suffix_bytes_copied": 0,
                        "physical_minimum_invariant_failures": [],
                        "parent_map": [int(_p) for _p in _all_actual_parents],
                        "state_rows": (
                            list(spec_state_indices_tensor.shape)
                            if spec_state_indices_tensor is not None else None
                        ),
                        "initial_rows": list(spec_initial_state_indices_tensor.shape),
                        "spine_chain_degenerate_unique_tree": bool(_is_spine),
                        "expanded_parent_state_rows": True,
                    }) + chr(10))
                except Exception:
                    pass
"""
    text = text[:start] + new + text[end:]
    text = sentinel + '\n' + text
    ga.write_text(text)
    import py_compile
    py_compile.compile(str(ga), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_a unique-node batch4 pack')

gl = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py')
text = gl.read_text()
sentinel = '# LUMO_FA_TREE_DELTA_ACTUAL_PARENT_ROWS'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_a tree-delta actual-parent patch already present')
else:
    text = text.replace(
        """    if initial_state_indices is not None:
        prefix_idx = int(initial_state_indices[0].item())
    else:
        prefix_idx = int(ssm_state_indices.reshape(-1)[0].item())
    prefix_state = initial_state[prefix_idx].to(torch.float32)
""",
        """    if initial_state_indices is not None:
        prefix_indices = initial_state_indices.reshape(-1).to(torch.long)
    else:
        prefix_indices = ssm_state_indices.reshape(n, -1)[:, 0].to(torch.long)
    prefix_state = initial_state.index_select(0, prefix_indices).to(torch.float32)
""",
        1,
    )
    text = text.replace(
        """    parents = parent_indices.to(device=q_f.device, dtype=torch.long)
    actual_parents = torch.empty((n,), dtype=torch.long, device=q_f.device)
    actual_parents[0] = -1
    for i in range(1, n):
        parent = int(parents[i].item())
        actual_parents[i] = 0 if parent < 0 else parent + 1
""",
        """    actual_parents = parent_indices.to(device=q_f.device, dtype=torch.long)
""",
        1,
    )
    text = text.replace(
        '    initial_projection = torch.einsum("hvk,nhk->nhv", prefix_state, k_f)\n',
        '    initial_projection = torch.einsum("nhvk,nhk->nhv", prefix_state, k_f)\n',
        1,
    )
    text = text.replace(
        """    states = (
        gamma[:, :, None, None] * prefix_state.unsqueeze(0)
        + torch.einsum("hij,hjv,hjk->ihvk", coeff, writes, k_f.permute(1, 0, 2))
    )
""",
        """    states = (
        gamma[:, :, None, None] * prefix_state
        + torch.einsum("hij,hjv,hjk->ihvk", coeff, writes, k_f.permute(1, 0, 2))
    )
""",
        1,
    )
    old = """    if HAS_INITIAL_STATE_INDICES:
        prefix_idx = tl.load(initial_state_indices + 0).to(tl.int64)
    else:
        prefix_idx = tl.load(ssm_state_indices + 0).to(tl.int64)
    prefix_h = tl.load(
        state + prefix_idx * stride_state_token + head_state_off,
        mask=m_h,
        other=0.0,
    ).to(tl.float32)

    for i in tl.static_range(0, BN):
        if i < N:
            if i == 0:
                parent_actual = tl.full((), -1, tl.int64)
            else:
                parent_raw = tl.load(parent_indices + i).to(tl.int64)
                parent_actual = tl.where(parent_raw < 0, 0, parent_raw + 1)
            parent_safe = tl.maximum(parent_actual, 0)
            parent_write_idx = tl.load(
                ssm_state_indices + parent_safe * stride_indices_seq
            ).to(tl.int64)
            parent_h = tl.load(
                state + parent_write_idx * stride_state_token + head_state_off,
                mask=m_h,
                other=0.0,
            ).to(tl.float32)
            h = tl.where(parent_actual >= 0, parent_h, prefix_h)
"""
    new = """    for i in tl.static_range(0, BN):
        if i < N:
            parent_actual = tl.load(parent_indices + i).to(tl.int64)
            parent_safe = tl.maximum(parent_actual, 0)
            if HAS_INITIAL_STATE_INDICES:
                prefix_idx = tl.load(initial_state_indices + i).to(tl.int64)
            else:
                prefix_idx = tl.load(
                    ssm_state_indices + i * stride_indices_seq
                ).to(tl.int64)
            prefix_h = tl.load(
                state + prefix_idx * stride_state_token + head_state_off,
                mask=m_h,
                other=0.0,
            ).to(tl.float32)
            parent_write_idx = tl.load(
                ssm_state_indices + parent_safe * stride_indices_seq
            ).to(tl.int64)
            parent_h = tl.load(
                state + parent_write_idx * stride_state_token + head_state_off,
                mask=m_h,
                other=0.0,
            ).to(tl.float32)
            h = tl.where(parent_actual >= 0, parent_h, prefix_h)
"""
    if old not in text:
        raise RuntimeError('F_a tree-delta Triton parent anchor not found')
    text = text.replace(old, new, 1)
    text = text.replace(
        """                _depths = []
                for _i, _parent in enumerate(_parents):
                    if _i == 0:
                        _depths.append(0)
                    else:
                        _depths.append(1 if int(_parent) < 0 else _depths[int(_parent) + 1] + 1)
""",
        """                _depths = []
                for _parent in _parents:
                    _depths.append(0 if int(_parent) < 0 else _depths[int(_parent)] + 1)
""",
    )
    text = sentinel + '\n' + text
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_a tree-delta actual-parent rows')
LUMOFAUNIQUEBATCH4PACK
'''

_FA_UNIQUE_BATCH4_STARTUP_FIX_BLOCK = r'''
python3 - <<'LUMOFAUNIQUEBATCH4STARTUP'
from pathlib import Path
ga = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py')
text = ga.read_text()
alloc_anchor = (
    '        self.spec_write_state_slot_tensor: torch.Tensor = torch.empty(\n'
    '            (self.decode_cudagraph_max_bs,),\n'
    '            dtype=torch.int32,\n'
    '            device=device,\n'
    '        )\n'
)
if 'self.fa_tree_parent_indices_tensor' not in text and alloc_anchor in text:
    text = text.replace(
        alloc_anchor,
        alloc_anchor
        + '        self.fa_tree_parent_indices_tensor: torch.Tensor = torch.empty(\n'
        + '            (self.decode_cudagraph_max_bs,),\n'
        + '            dtype=torch.int32,\n'
        + '            device=device,\n'
        + '        )\n',
        1,
    )
bad = '                batch_size = max(int(batch_size), int(num_spec_decodes))\n'
changed = False
if 'self.fa_tree_parent_indices_tensor' in text and alloc_anchor in text:
    changed = True
if bad in text:
    text = text.replace(bad, '')
    changed = True
mask_anchor = (
    '                num_spec_decodes = _req_count * _group_size\n'
    '                num_spec_decode_tokens = _req_count * _group_size\n'
    '                num_accepted_tokens = torch.ones(\n'
)
if mask_anchor in text:
    text = text.replace(
        mask_anchor,
        '                num_spec_decodes = _req_count * _group_size\n'
        '                num_spec_decode_tokens = _req_count * _group_size\n'
        '                spec_sequence_masks = torch.ones(\n'
        '                    (num_spec_decodes,), dtype=torch.bool,\n'
        '                    device=query_start_loc.device)\n'
        '                num_accepted_tokens = torch.ones(\n',
        1,
    )
    changed = True
bad_mask_block = (
    '                spec_sequence_masks = torch.ones(\n'
    '                    (num_spec_decodes,), dtype=torch.bool,\n'
    '                    device=query_start_loc.device)\n'
)
if bad_mask_block in text:
    text = text.replace(bad_mask_block, '')
    changed = True
copy_anchor = (
    '            self.spec_sequence_masks[:num_spec_decodes].copy_(\n'
    '                spec_sequence_masks[:num_spec_decodes], non_blocking=True\n'
    '            )\n'
)
if copy_anchor in text and '_fa_spec_masks_for_copy' not in text:
    text = text.replace(
        copy_anchor,
        '            if bool(locals().get("fa_unique_node_mode", False)):\n'
        '                _fa_spec_masks_for_copy = torch.ones(\n'
        '                    (num_spec_decodes,), dtype=torch.bool,\n'
        '                    device=spec_state_indices_tensor.device)\n'
        '            else:\n'
        '                _fa_spec_masks_for_copy = spec_sequence_masks[:num_spec_decodes]\n'
        '            self.spec_sequence_masks[:num_spec_decodes].copy_(\n'
        '                _fa_spec_masks_for_copy, non_blocking=True\n'
        '            )\n',
        1,
    )
    changed = True
parent_anchor = (
    '                fa_tree_parent_indices_tensor = torch.tensor(\n'
    '                    _all_actual_parents, dtype=torch.int32,\n'
    '                    device=query_start_loc.device)\n'
)
if parent_anchor in text:
    text = text.replace(
        parent_anchor,
        '                _fa_parent_local = torch.tensor(\n'
        '                    _all_actual_parents, dtype=torch.int32,\n'
        '                    device=query_start_loc.device)\n'
        '                if (\n'
        '                    hasattr(self, "fa_tree_parent_indices_tensor")\n'
        '                    and int(_fa_parent_local.numel()) <= int(self.fa_tree_parent_indices_tensor.numel())\n'
        '                ):\n'
        '                    self.fa_tree_parent_indices_tensor[:_fa_parent_local.numel()].copy_(\n'
        '                        _fa_parent_local, non_blocking=True)\n'
        '                    fa_tree_parent_indices_tensor = self.fa_tree_parent_indices_tensor[\n'
        '                        :_fa_parent_local.numel()]\n'
        '                else:\n'
        '                    fa_tree_parent_indices_tensor = _fa_parent_local\n',
        1,
    )
    changed = True
depth_anchor = (
    '                fa_tree_depth_row_tensors = tuple(\n'
    '                    torch.tensor(_rows, dtype=torch.long, device=query_start_loc.device)\n'
    '                    for _rows in fa_tree_depth_rows\n'
    '                )\n'
    '                fa_tree_depth_query_start_tensors = tuple(\n'
    '                    torch.arange(len(_rows) + 1, dtype=torch.int32, device=query_start_loc.device)\n'
    '                    for _rows in fa_tree_depth_rows\n'
    '                )\n'
)
if depth_anchor in text:
    text = text.replace(
        depth_anchor,
        '                _fa_depth_cache = getattr(self, "_lumo_fa_tree_depth_cache", None)\n'
        '                if _fa_depth_cache is None:\n'
        '                    _fa_depth_cache = {}\n'
        '                    self._lumo_fa_tree_depth_cache = _fa_depth_cache\n'
        '                _fa_depth_key = (int(num_spec_decodes), fa_tree_depth_rows)\n'
        '                if _fa_depth_key not in _fa_depth_cache:\n'
        '                    _fa_depth_cache[_fa_depth_key] = (\n'
        '                        tuple(\n'
        '                            torch.tensor(_rows, dtype=torch.long, device=query_start_loc.device)\n'
        '                            for _rows in fa_tree_depth_rows\n'
        '                        ),\n'
        '                        tuple(\n'
        '                            torch.arange(len(_rows) + 1, dtype=torch.int32, device=query_start_loc.device)\n'
        '                            for _rows in fa_tree_depth_rows\n'
        '                        ),\n'
        '                    )\n'
        '                fa_tree_depth_row_tensors, fa_tree_depth_query_start_tensors = _fa_depth_cache[_fa_depth_key]\n',
        1,
    )
    changed = True
if changed:
    ga.write_text(text)
    import py_compile
    py_compile.compile(str(ga), doraise=True)
    print('[TRACK-B-PRELAUNCH] repaired stale F_a batch4 startup metadata')
else:
    print('[TRACK-B-PRELAUNCH] stale F_a batch4 startup metadata absent')
LUMOFAUNIQUEBATCH4STARTUP
'''

_FA_TREE_DELTA_VALID_N_BLOCK = r'''
python3 - <<'LUMOFATREEVALIDN'
from pathlib import Path
gl = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py')
text = gl.read_text()
sentinel = '# LUMO_FA_TREE_DELTA_VALID_N'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_a tree-delta valid-N patch already present')
else:
    text = text.replace(
        '    use_qk_l2norm_in_kernel: bool,\n) -> tuple[torch.Tensor, torch.Tensor]:\n',
        '    use_qk_l2norm_in_kernel: bool,\n    valid_n: int | None = None,\n) -> tuple[torch.Tensor, torch.Tensor]:\n',
        1,
    )
    text = text.replace(
        '    n = int(q.shape[0])\n',
        '    n = int(q.shape[0] if valid_n is None else valid_n)\n',
        1,
    )
    text = text.replace(
        '    q = q.squeeze(0)\n    k = k.squeeze(0)\n    v = v.squeeze(0)\n    n = int(q.shape[0] if valid_n is None else valid_n)\n',
        '    q = q.squeeze(0)\n    k = k.squeeze(0)\n    v = v.squeeze(0)\n    n = int(q.shape[0] if valid_n is None else valid_n)\n    q = q[:n]\n    k = k[:n]\n    v = v[:n]\n',
        1,
    )
    text = text.replace(
        '    use_qk_l2norm_in_kernel: bool,\n) -> tuple[torch.Tensor, None]:\n',
        '    use_qk_l2norm_in_kernel: bool,\n    valid_n: int | None = None,\n) -> tuple[torch.Tensor, None]:\n',
        1,
    )
    text = text.replace(
        '    n = int(q.shape[1])\n',
        '    n = int(q.shape[1] if valid_n is None else valid_n)\n',
        1,
    )
    text = text.replace(
        '    out = torch.empty_like(v)\n',
        '    out = torch.zeros_like(v)\n',
        1,
    )
    text = text.replace(
        '                parent_indices=_parents_t,\n                use_qk_l2norm_in_kernel=True,\n            )\n',
        '                parent_indices=_parents_t,\n                use_qk_l2norm_in_kernel=True,\n                valid_n=int(attn_metadata.num_spec_decode_tokens),\n            )\n',
        1,
    )
    text = sentinel + '\n' + text
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_a tree-delta valid-N patch')
LUMOFATREEVALIDN
'''

_FA_GDN_CORE_CUDAGRAPH_UNSAFE_BLOCK = r'''
python3 - <<'LUMOFAGDNUNSAFE'
from pathlib import Path
gl = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py')
text = gl.read_text()
sentinel = '# LUMO_FA_GDN_CORE_CUDAGRAPH_UNSAFE'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] GDN core cudagraph-unsafe tag already present')
else:
    old = """direct_register_custom_op(
    op_name="gdn_attention_core",
    op_func=gdn_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=gdn_attention_core_fake,
)
"""
    new = """direct_register_custom_op(
    op_name="gdn_attention_core",
    op_func=gdn_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=gdn_attention_core_fake,
    tags=(torch._C.Tag.cudagraph_unsafe,),
)
"""
    if old not in text:
        raise RuntimeError('gdn_attention_core registration anchor not found')
    text = sentinel + '\n' + text.replace(old, new, 1)
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] tagged gdn_attention_core as cudagraph_unsafe')
LUMOFAGDNUNSAFE
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
        try:
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
        except Exception as exc:
            info = {
                "fb_control_source": "env",
                "fb_control_file": path,
                "fb_control_error": repr(exc),
            }
    if depth < 1 or depth > int(max_depth):
        raise RuntimeError(f"LUMO_FB active depth {depth} outside launch_n_max {max_depth}")
    if k < 0 or k > 2:
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

def _lumo_fb_alt_record_from_logits(logits, row0_token, position):
    try:
        row = logits.reshape(-1, logits.shape[-1])[:1].float()
        topn = min(8, int(row.shape[-1]))
        vals, idx = _lumo_fb_torch.topk(row, topn, dim=-1)
        probs = _lumo_fb_torch.softmax(vals, dim=-1)
        row0 = int(row0_token.reshape(-1)[0].item())
        top_tokens = [int(x) for x in idx[0].detach().cpu().tolist()]
        top_probs = [float(x) for x in probs[0].detach().cpu().tolist()]
        row0_p = 0.0
        alt_tok = row0
        alt_p = 0.0
        for tok, prob in zip(top_tokens, top_probs):
            if tok == row0:
                row0_p = float(prob)
                break
        for tok, prob in zip(top_tokens, top_probs):
            if tok != row0 and tok != 0:
                alt_tok = int(tok)
                alt_p = float(prob)
                break
        if row0_p <= 0.0 and top_tokens and top_tokens[0] == row0:
            row0_p = float(top_probs[0])
        gap = float(row0_p - alt_p)
        ratio = float(alt_p / max(row0_p, 1e-9))
        return {
            "position": int(position),
            "row0_token": row0,
            "alt_token": int(alt_tok),
            "row0_p": row0_p,
            "alt_p": alt_p,
            "gap": gap,
            "ratio": ratio,
            "top_tokens": top_tokens,
            "top_probs": top_probs,
        }
    except Exception as exc:
        return {
            "position": int(position),
            "row0_token": int(row0_token.reshape(-1)[0].item()),
            "alt_token": int(row0_token.reshape(-1)[0].item()),
            "row0_p": 0.0,
            "alt_p": 0.0,
            "gap": 0.0,
            "ratio": 0.0,
            "error": repr(exc),
        }

def _lumo_fb_build_free_row1(row0, records):
    row0_1d = row0.reshape(-1)
    depth = int(row0_1d.numel())
    usable = [rec for rec in records[:depth]
              if int(rec.get("alt_token", rec.get("row0_token", -1))) != int(rec.get("row0_token", -1))]
    if not usable:
        return row0.clone(), {
            "row1_enabled": False,
            "flip_pos": -1,
            "gate_reason": "no_cached_alt",
            "candidate_valid": False,
        }
    p1_max = float(_lumo_fb_os.environ.get("LUMO_FB_FREE_ROW1_P1_MAX", "0.45"))
    ratio_min = float(_lumo_fb_os.environ.get("LUMO_FB_FREE_ROW1_RATIO_MIN", "0.50"))
    low_conf = [
        rec for rec in usable
        if float(rec.get("row0_p", 0.0)) < p1_max
        and float(rec.get("ratio", 0.0)) > ratio_min
    ]
    pool = low_conf if low_conf else usable
    best = max(pool, key=lambda rec: (
        float(rec.get("ratio", 0.0)),
        float(rec.get("alt_p", 0.0)),
        -int(rec.get("position", 0)),
    ))
    flip_pos = int(best.get("position", -1))
    row1 = row0.clone()
    if 0 <= flip_pos < depth:
        row1.reshape(-1)[flip_pos] = int(best["alt_token"])
    always = _lumo_fb_os.environ.get("LUMO_FB_FREE_ROW1_ALWAYS") == "1"
    gated = bool(low_conf)
    enabled = always or gated
    return row1, {
        "row1_enabled": bool(enabled),
        "flip_pos": int(flip_pos),
        "gate_reason": (
            "always_on"
            if always else
            (f"low_conf_pos{flip_pos}" if gated else "gate_closed")
        ),
        "candidate_valid": True,
    }

def _lumo_fb_free_row1_event(active_depth, row0, row1, records, decision,
                             requested_k, verified_rows, fb_proposer_us,
                             policy_info=None):
    row0_list = [int(x) for x in row0.reshape(-1)[:active_depth].detach().cpu().tolist()]
    row1_list = [int(x) for x in row1.reshape(-1)[:active_depth].detach().cpu().tolist()]
    rows_generated = 2 if decision.get("candidate_valid") else 1
    return {
        "event": "fb_free_row1_decision",
        "active_depth": int(active_depth),
        "active_k": int(requested_k),
        "row0": row0_list,
        "row1": row1_list,
        "row1_enabled": bool(decision.get("row1_enabled", False)),
        "row1_source": "mtp_cached_alt",
        "flip_pos": int(decision.get("flip_pos", -1)),
        "proposer_free": True,
        "extra_extend_one_calls": 0,
        "position_tree_enabled": False,
        "generated_rows": int(rows_generated if verified_rows > 1 else verified_rows),
        "candidate_rows": int(rows_generated),
        "verified_rows": int(verified_rows),
        "row0_p": [round(float(rec.get("row0_p", 0.0)), 6) for rec in records[:active_depth]],
        "row1_alt_p": [round(float(rec.get("alt_p", 0.0)), 6) for rec in records[:active_depth]],
        "row0_alt_ratio": [round(float(rec.get("ratio", 0.0)), 6) for rec in records[:active_depth]],
        "gate_reason": str(decision.get("gate_reason", "")),
        "fb_proposer_us": int(fb_proposer_us),
        "policy": policy_info or {},
    }

def _lumo_fb_unified_step_event(active_depth, records, decision, verified_rows,
                                proposer_us, trim_us=0, verify_us=0,
                                commit_us=0):
    candidate_alt_nodes = sum(
        1 for rec in records[:active_depth]
        if int(rec.get("alt_token", rec.get("row0_token", -1))) != int(rec.get("row0_token", -1))
    )
    candidate_pool_nodes = int(active_depth) + int(candidate_alt_nodes)
    selected_nodes = int(verified_rows)
    trimmed_nodes = max(0, candidate_pool_nodes - selected_nodes)
    invariant_failures = []
    if selected_nodes != int(verified_rows):
        invariant_failures.append("selected_nodes_ne_verified_nodes")
    return {
        "event": "round_f_unified_step",
        "stage": "stage2_cached_alt_shadow" if int(verified_rows) == int(active_depth) else "stage2_cached_alt_active_row_compat",
        "candidate_pool_nodes": int(candidate_pool_nodes),
        "selected_nodes": int(selected_nodes),
        "verified_nodes": int(verified_rows),
        "trimmed_nodes": int(trimmed_nodes),
        "max_depth": int(active_depth),
        "sources": {
            "mtp_top1": int(active_depth),
            "mtp_alt": int(candidate_alt_nodes),
            "suffix": 0,
        },
        "path_rows": 0 if int(verified_rows) == int(active_depth) else 1,
        "scheduler_visible_clone_requests": 0 if int(verified_rows) == int(active_depth) else 1,
        "prefix_kv_copy_bytes": 0,
        "recomputed_shared_prefix_nodes": 0,
        "extra_proposer_for_trimmed_nodes": 0,
        "accepted_path_commit_only": True,
        "tree_attention": False,
        "gdn_parent_gather": False,
        "depth_positions": True,
        "tree_sampler": False,
        "top1_spine_accept_depth": None,
        "accepted_depth": None,
        "accepted_node_path": [],
        "estimated_event_ms": None,
        "event_budget_ms": None,
        "tree_score": None,
        "proposer_us": int(proposer_us),
        "trim_us": int(trim_us),
        "verify_us": int(verify_us),
        "tree_attention_us": 0,
        "gdn_parent_gather_us": 0,
        "depth_sync_us": 0,
        "commit_us": int(commit_us),
        "gdn_state_bytes_copied": 0,
        "kv_suffix_bytes_copied": 0,
        "physical_minimum_invariant_failures": invariant_failures,
        "gate_reason": str(decision.get("gate_reason", "")),
        "row1_enabled": bool(decision.get("row1_enabled", False)),
    }

def _lumo_fb_spine_only_state_tree_event(active_depth, proposer_us,
                                         spine_source="lumo_fb_extend_one_k1_trunk",
                                         replay_idx=None):
    internal_rows = _lumo_fb_os.environ.get("LUMO_FB_INTERNAL_ROWS") == "1"
    kernel_rows = _lumo_fb_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1"
    no_kv_prefix_copy = _lumo_fb_os.environ.get("LUMO_FB_NO_KV_PREFIX_COPY") == "1"
    replay = spine_source == "replay_native_e3_mtp_draft"
    invariant_failures = []
    if not internal_rows:
        invariant_failures.append("lumo_fb_internal_rows_off")
    if not kernel_rows:
        invariant_failures.append("lumo_fb_kernel_rows_off")
    if not no_kv_prefix_copy:
        invariant_failures.append("lumo_fb_no_kv_prefix_copy_off")
    return {
        "event": "round_f_unified_step",
        "stage": "stage3_spine_only_state_tree",
        "component_under_test": (
            "verifier_replay_isolation"
            if replay else "proposer_wrapper_plus_real_k1_verifier"
        ),
        "verifier_path": "LUMO_FB_INTERNAL_ROWS/KERNEL_ROWS_K1",
        "internal_rows_enabled": bool(internal_rows),
        "kernel_rows_enabled": bool(kernel_rows),
        "no_kv_prefix_copy_enabled": bool(no_kv_prefix_copy),
        "candidate_pool_nodes": int(active_depth),
        "selected_nodes": int(active_depth),
        "verified_nodes": int(active_depth),
        "trimmed_nodes": 0,
        "max_depth": int(active_depth),
        "sources": {
            "mtp_top1": int(active_depth),
            "mtp_alt": 0,
            "suffix": 0,
        },
        "path_rows": 0,
        "scheduler_visible_clone_requests": 0,
        "prefix_kv_copy_bytes": 0,
        "recomputed_shared_prefix_nodes": 0,
        "extra_proposer_for_trimmed_nodes": 0,
        "accepted_path_commit_only": True,
        "tree_attention": False,
        "gdn_parent_gather": True,
        "depth_positions": True,
        "tree_sampler": False,
        "top1_spine_accept_depth": None,
        "accepted_depth": None,
        "accepted_node_path": [],
        "estimated_event_ms": None,
        "event_budget_ms": None,
        "tree_score": None,
        "proposer_us": int(proposer_us),
        "trim_us": 0,
        "verify_us": 0,
        "tree_attention_us": 0,
        "gdn_parent_gather_us": 0,
        "depth_sync_us": 0,
        "commit_us": 0,
        "gdn_state_bytes_copied": 0,
        "kv_suffix_bytes_copied": 0,
        "physical_minimum_invariant_failures": invariant_failures,
        "spine_source": str(spine_source),
        "replay_idx": replay_idx,
    }

def _lumo_fb_debug_write(event):
    if _lumo_fb_os.environ.get("LUMO_FB_DEBUG") != "1":
        return
    try:
        global _LUMO_FB_DBG_FH
        try:
            _LUMO_FB_DBG_FH
        except NameError:
            _LUMO_FB_DBG_FH = open("/logs/fb_debug.jsonl", "a", buffering=1)
        _LUMO_FB_DBG_FH.write(_lumo_fb_json.dumps(event) + chr(10))
    except Exception:
        pass

_lumo_fb_replay_cache = None
_lumo_fb_replay_idx = 0

def _lumo_fb_replay_next(device):
    global _lumo_fb_replay_cache, _lumo_fb_replay_idx
    path = _lumo_fb_os.environ.get("LUMO_FB_REPLAY_DRAFT_FILE")
    if not path:
        return None
    if _lumo_fb_replay_cache is None:
        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = _lumo_fb_json.loads(line)
                if payload.get("event") == "mtp_draft":
                    rows.append(payload["draft"])
        _lumo_fb_replay_cache = rows
        _lumo_fb_replay_idx = 0
    if _lumo_fb_replay_idx >= len(_lumo_fb_replay_cache):
        raise RuntimeError(
            f"LUMO_FB_REPLAY_DRAFT_FILE exhausted at {_lumo_fb_replay_idx}")
    draft = _lumo_fb_replay_cache[_lumo_fb_replay_idx]
    _lumo_fb_replay_idx += 1
    return _lumo_fb_torch.tensor(
        draft, dtype=_lumo_fb_torch.int64, device=device), int(_lumo_fb_replay_idx - 1)

def _lumo_fb_extend_one(self, root_token, base_positions, base_hidden_states,
                        base_common_attn_metadata, batch_size, per_layer_attn_metadata,
                        num_rejected_tokens_gpu, draft_len=None,
                        return_free_row1_metadata=False, root_logits=None):
    if draft_len is None:
        draft_len = self.num_speculative_tokens
    draft_token_ids_list = [root_token]
    free_row1_records = []
    if return_free_row1_metadata and root_logits is not None:
        free_row1_records.append(
            _lumo_fb_alt_record_from_logits(root_logits, root_token, 0))
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
        if return_free_row1_metadata:
            step_logits = _lumo_fb_sample_logits(self, last_hidden_states[:batch_size])
            if step_logits is None:
                draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
            else:
                draft_token_ids = _lumo_fb_torch.argmax(step_logits, dim=-1).to(_lumo_fb_torch.int64)
                free_row1_records.append(
                    _lumo_fb_alt_record_from_logits(
                        step_logits, draft_token_ids, token_index + 1))
        else:
            draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
        draft_token_ids_list.append(draft_token_ids)
    out = _lumo_fb_torch.stack(draft_token_ids_list, dim=1)
    if return_free_row1_metadata:
        return out, free_row1_records
    return out

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
            _lumo_fb_default_stride = (
                self.num_speculative_tokens + 1
                if _lumo_fb_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1"
                else self.num_speculative_tokens
            )
            slot_stride = int(_lumo_fb_os.environ.get(
                "LUMO_FB_BATCHED_SLOT_STRIDE", str(_lumo_fb_default_stride)))
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

def _lumo_fb_extend_top2_pos01_tree_batched(
        self, root_tokens, base_positions, base_hidden_states,
        base_common_attn_metadata, per_layer_attn_metadata,
        num_rejected_tokens_gpu, draft_len=None):
    # Bounded per-position top-2 tree flattened to rows:
    # branch at the first L draft positions (default L=3 => 8 rows), then
    # greedily extend each row to depth N. Row 0 is the ordinary top-1 E/K1
    # chain; the caller may splice in the exact single-row trunk as an extra
    # guard while the batched proposer is being stabilized.
    if draft_len is None:
        draft_len = self.num_speculative_tokens
    draft_len = int(draft_len)
    if draft_len < 2:
        return _lumo_fb_extend_paths_batched(
            self, root_tokens[:1], base_positions, base_hidden_states,
            base_common_attn_metadata, per_layer_attn_metadata,
            num_rejected_tokens_gpu, draft_len=draft_len)

    def _cpu_index(value, cpu_idx):
        if value is None:
            return None
        return value.index_select(0, cpu_idx).clone()

    def _run_step(input_ids, positions, hidden_states, cad, draft_index):
        k = int(input_ids.numel())
        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(k)
        )
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
            block_size=self.block_size,
            max_model_len=self.max_model_len,
            out_clamped_positions=out_pos,
            out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
            input_batch_size=input_batch_size,
        )
        if k > 1:
            _lumo_fb_default_stride = (
                self.num_speculative_tokens + 1
                if _lumo_fb_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1"
                else self.num_speculative_tokens
            )
            slot_stride = int(_lumo_fb_os.environ.get(
                "LUMO_FB_BATCHED_SLOT_STRIDE", str(_lumo_fb_default_stride)))
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
                common_attn_metadata=cad, draft_index=int(draft_index),
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        self.input_ids[:k] = input_ids.int()
        self.hidden_states[:k] = hidden_states
        if self.supports_mm_inputs:
            self.inputs_embeds[:k] = self.model.embed_input_ids(input_ids.int())
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
                next_hidden_states = ret_hidden_states
            else:
                last_hidden_states, next_hidden_states = ret_hidden_states
        next_hidden_states = next_hidden_states[:k]
        logits = _lumo_fb_sample_logits(self, last_hidden_states[:k])
        return next_hidden_states, positions, cad, logits

    root_tokens = root_tokens.reshape(-1)[:2].to(device=base_hidden_states.device)
    if root_tokens.numel() < 2:
        return _lumo_fb_extend_paths_batched(
            self, root_tokens[:1], base_positions, base_hidden_states,
            base_common_attn_metadata, per_layer_attn_metadata,
            num_rejected_tokens_gpu, draft_len=draft_len)
    branch_depth = int(_lumo_fb_os.environ.get("LUMO_FB_TREE_BRANCH_DEPTH", "3"))
    branch_depth = max(1, min(branch_depth, draft_len))

    k0 = 2
    if self.uses_mrope:
        positions = base_positions[:, :1].repeat(1, k0).clone()
    else:
        positions = base_positions[:1].repeat(k0).clone()
    hidden_states = base_hidden_states[:1].repeat(k0, 1).clone()
    cad = _lumo_fb_replace(
        base_common_attn_metadata,
        query_start_loc=self.arange[: k0 + 1],
        query_start_loc_cpu=_lumo_fb_torch.from_numpy(
            self.token_arange_np[: k0 + 1]
        ).clone(),
        seq_lens=base_common_attn_metadata.seq_lens[:1].repeat(k0).clone(),
        _seq_lens_cpu=_lumo_fb_repeat_cpu_shadow(
            base_common_attn_metadata._seq_lens_cpu, k0),
        _num_computed_tokens_cpu=_lumo_fb_repeat_cpu_shadow(
            base_common_attn_metadata._num_computed_tokens_cpu, k0),
        num_reqs=k0,
        num_actual_tokens=k0,
        max_query_len=1,
        block_table_tensor=base_common_attn_metadata.block_table_tensor[:1].repeat(
            k0, 1).clone(),
        slot_mapping=base_common_attn_metadata.slot_mapping[:1].repeat(k0).clone(),
    )
    if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
        cad.seq_lens -= num_rejected_tokens_gpu[:1].repeat(k0)
        cad._seq_lens_cpu = None
        cad._num_computed_tokens_cpu = None

    draft_token_ids_list = [root_tokens]
    k = k0
    for consumed_index in range(0, branch_depth - 1):
        hidden_after, positions_after, cad_after, logits_after = _run_step(
            draft_token_ids_list[-1], positions, hidden_states, cad,
            draft_index=consumed_index + 1)
        if logits_after is None:
            return _lumo_fb_extend_paths_batched(
                self, root_tokens, base_positions, base_hidden_states,
                base_common_attn_metadata, per_layer_attn_metadata,
                num_rejected_tokens_gpu, draft_len=draft_len)
        top2 = _lumo_fb_torch.topk(logits_after, 2, dim=-1).indices
        branch_idx = _lumo_fb_torch.arange(
            k, dtype=_lumo_fb_torch.long, device=root_tokens.device
        ).repeat_interleave(2)
        cpu_branch_idx = branch_idx.to(device="cpu")
        draft_token_ids_list = [
            col.index_select(0, branch_idx).clone()
            for col in draft_token_ids_list
        ]
        draft_token_ids_list.append(top2.reshape(-1))
        k = int(branch_idx.numel())
        hidden_states = hidden_after.index_select(0, branch_idx).clone()
        if self.uses_mrope:
            positions = positions_after.index_select(1, branch_idx).clone()
        else:
            positions = positions_after.index_select(0, branch_idx).clone()
        cad = _lumo_fb_replace(
            cad_after,
            query_start_loc=self.arange[: k + 1],
            query_start_loc_cpu=_lumo_fb_torch.from_numpy(
                self.token_arange_np[: k + 1]
            ).clone(),
            seq_lens=cad_after.seq_lens.index_select(0, branch_idx).clone(),
            _seq_lens_cpu=_cpu_index(cad_after._seq_lens_cpu, cpu_branch_idx),
            _num_computed_tokens_cpu=_cpu_index(
                cad_after._num_computed_tokens_cpu, cpu_branch_idx),
            num_reqs=k,
            num_actual_tokens=k,
            max_query_len=1,
            block_table_tensor=cad_after.block_table_tensor.index_select(
                0, branch_idx).clone(),
            slot_mapping=cad_after.slot_mapping.index_select(0, branch_idx).clone(),
        )

    for consumed_index in range(branch_depth - 1, draft_len - 1):
        hidden_states, positions, cad, logits = _run_step(
            draft_token_ids_list[-1], positions, hidden_states, cad,
            draft_index=consumed_index + 1)
        if logits is None:
            draft_token_ids = self._greedy_sample(hidden_states[:k])
        else:
            draft_token_ids = _lumo_fb_torch.argmax(logits, dim=-1).to(_lumo_fb_torch.int64)
        draft_token_ids_list.append(draft_token_ids)
    return _lumo_fb_torch.stack(draft_token_ids_list, dim=1)

def _lumo_fb_propose(self, target_token_ids, target_positions, target_hidden_states,
                     next_token_ids, token_indices_to_sample, common_attn_metadata,
                     sampling_metadata, mm_embed_inputs=None,
                     num_rejected_tokens_gpu=None, slot_mappings=None):
    global _LUMO_FB_DBG_FH
    _lumo_fb_prop_t0 = _lumo_fb_time.perf_counter_ns()
    if (_lumo_fb_os.environ.get("LUMO_FB_PATHS") != "1"
            and _lumo_fb_os.environ.get("LUMO_FB_KERNEL_ROWS") != "1"):
        return _lumo_fb_orig_propose(
            self, target_token_ids, target_positions, target_hidden_states,
            next_token_ids, token_indices_to_sample, common_attn_metadata,
            sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)
    active_depth, requested_k, control_info = _lumo_fb_read_control(self.num_speculative_tokens)
    if requested_k == 0:
        return _lumo_fb_orig_propose(
            self, target_token_ids, target_positions, target_hidden_states,
            next_token_ids, token_indices_to_sample, common_attn_metadata,
            sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)
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
    if requested_k == 1 and _lumo_fb_os.environ.get("LUMO_FB_REPLAY_DRAFT_FILE"):
        replay = _lumo_fb_replay_next(device=base_hidden_states.device)
        out, replay_idx = replay
        out = out[:, :active_depth].contiguous()
        spine_source = "replay_native_e3_mtp_draft"
        _lumo_fb_debug_write(_lumo_fb_spine_only_state_tree_event(
            active_depth=active_depth,
            proposer_us=int((_lumo_fb_time.perf_counter_ns() - _lumo_fb_prop_t0) // 1000),
            spine_source=spine_source,
            replay_idx=replay_idx,
        ))
        try:
            if _lumo_fb_os.environ.get("LUMO_FB_DEBUG") == "1":
                _lumo_fb_debug_write({
                    "ts": round(_lumo_fb_time.time(), 4),
                    "event": "fb_state_tree_spine_draft",
                    "active_depth": int(active_depth),
                    "active_k": int(requested_k),
                    "raw_roots": raw_roots.view(-1).tolist(),
                    "root_candidates": root_candidates[:8].tolist(),
                    "draft": out.tolist(),
                    "spine_source": spine_source,
                    "replay_idx": replay_idx,
                    "policy": policy_info,
                })
        except Exception:
            pass
        return out
    if (requested_k >= 2
            and _lumo_fb_os.environ.get("LUMO_FB_FREE_ROW1") == "1"):
        path0, free_records = _lumo_fb_extend_one(
            self, _lumo_fb_path0_root, positions, base_hidden_states,
            common_attn_metadata, batch_size, dict(per_layer_attn_metadata),
            num_rejected_tokens_gpu, draft_len=active_depth,
            return_free_row1_metadata=True, root_logits=logits)
        path0 = path0[:, :active_depth]
        while len(free_records) < active_depth:
            pos = len(free_records)
            tok = path0.reshape(-1)[pos]
            free_records.append({
                "position": int(pos),
                "row0_token": int(tok.item()),
                "alt_token": int(tok.item()),
                "row0_p": 0.0,
                "alt_p": 0.0,
                "gap": 0.0,
                "ratio": 0.0,
                "error": "missing_logits",
            })
        path1, free_decision = _lumo_fb_build_free_row1(path0, free_records)
        shadow = _lumo_fb_os.environ.get("LUMO_FB_FREE_ROW1_SHADOW") == "1"
        active_row1 = bool(free_decision.get("row1_enabled", False)) and not shadow
        verified_rows = 2 if active_row1 else 1
        out = (_lumo_fb_torch.cat([path0, path1[:, :active_depth]], dim=1)
               if active_row1 else path0)
        _free_proposer_us = int((_lumo_fb_time.perf_counter_ns() - _lumo_fb_prop_t0) // 1000)
        _lumo_fb_debug_write(_lumo_fb_free_row1_event(
            active_depth=active_depth,
            row0=path0,
            row1=path1[:, :active_depth],
            records=free_records,
            decision=free_decision,
            requested_k=requested_k,
            verified_rows=verified_rows,
            fb_proposer_us=_free_proposer_us,
            policy_info=policy_info,
        ))
        _lumo_fb_debug_write(_lumo_fb_unified_step_event(
            active_depth=active_depth,
            records=free_records,
            decision=free_decision,
            verified_rows=(active_depth if not active_row1 else 2 * active_depth),
            proposer_us=_free_proposer_us,
        ))
        return out
    if policy_k == 1:
        path0 = _lumo_fb_extend_one(
            self, _lumo_fb_path0_root, positions, base_hidden_states,
            common_attn_metadata, batch_size, dict(per_layer_attn_metadata),
            num_rejected_tokens_gpu, draft_len=active_depth)
        out = path0[:, :active_depth]
        _lumo_fb_debug_write(_lumo_fb_spine_only_state_tree_event(
            active_depth=active_depth,
            proposer_us=int((_lumo_fb_time.perf_counter_ns() - _lumo_fb_prop_t0) // 1000),
            spine_source="lumo_fb_extend_one_k1_trunk",
        ))
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
    if (_lumo_fb_os.environ.get("LUMO_FB_BATCHED_PROPOSER") == "1"
            or _lumo_fb_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1"):
        root_vec = _lumo_fb_torch.cat([
            _lumo_fb_path0_root.reshape(-1)[:1],
            (_lumo_fb_path0_root if _lumo_fb_os.environ.get("LUMO_FB_DUP_PATH1") == "1"
             else roots[1]).reshape(-1)[:1],
        ], dim=0)
        if _lumo_fb_os.environ.get("LUMO_FB_POSITION_TREE", "1") == "1":
            # Strict-superset invariant: K2 row 0 must be the exact K1 draft
            # chain. The row-tree batched proposer can perturb row0 under the
            # wider MTP batch shape, which lowers trunk acceptance even though
            # verify/commit remains lossless. Preserve correctness first by
            # splicing the canonical K1 trunk into row0; the later kernel work
            # can make this a single batch-shape-invariant proposer forward.
            _lumo_fb_k1_trunk = _lumo_fb_extend_one(
                self, _lumo_fb_path0_root, positions, base_hidden_states,
                common_attn_metadata, batch_size, dict(per_layer_attn_metadata),
                num_rejected_tokens_gpu, draft_len=active_depth)
            paths = _lumo_fb_extend_top2_pos01_tree_batched(
                self, root_vec, positions, base_hidden_states, common_attn_metadata,
                dict(per_layer_attn_metadata), num_rejected_tokens_gpu,
                draft_len=active_depth)
            if paths.shape[0] > 0:
                paths = paths.clone()
                paths[0, :active_depth] = _lumo_fb_k1_trunk[0, :active_depth]
            if (_lumo_fb_os.environ.get("LUMO_FB_SINGLE_FLIP_TREE", "1") == "1"
                    and paths.shape[0] > 1):
                trunk = _lumo_fb_k1_trunk[0, :active_depth].to(paths.device)
                selected = [0]
                seen_first_diff = set()
                for row_idx in range(1, int(paths.shape[0])):
                    row = paths[row_idx, :active_depth]
                    diffs = (row != trunk).nonzero(as_tuple=False).reshape(-1)
                    if int(diffs.numel()) == 0:
                        continue
                    first_diff = int(diffs[0].item())
                    if first_diff in seen_first_diff:
                        continue
                    if first_diff > 0 and not bool((row[:first_diff] == trunk[:first_diff]).all().item()):
                        continue
                    seen_first_diff.add(first_diff)
                    selected.append(row_idx)
                    if len(selected) >= active_depth + 1:
                        break
                if len(selected) > 1:
                    select_idx = _lumo_fb_torch.tensor(
                        selected, dtype=_lumo_fb_torch.long, device=paths.device)
                    paths = paths.index_select(0, select_idx).contiguous()
            out = paths[:, :active_depth].reshape(1, -1)
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
                        "row_count": int(paths.shape[0]),
                        "tree_shape": ("top2_single_flip_rows"
                                       if _lumo_fb_os.environ.get("LUMO_FB_SINGLE_FLIP_TREE", "1") == "1"
                                       else "top2_prefix_rows"),
                        "tree_branch_depth": int(_lumo_fb_os.environ.get(
                            "LUMO_FB_TREE_BRANCH_DEPTH", "3")),
                        "fb_proposer_us": int((_lumo_fb_time.perf_counter_ns() - _lumo_fb_prop_t0) // 1000),
                        "raw_roots": raw_roots.view(-1).tolist(),
                        "roots": [int(paths[i, 0].item()) for i in range(paths.shape[0])],
                        "second_tokens": [int(paths[i, 1].item()) for i in range(paths.shape[0])] if active_depth > 1 else [],
                        "root_candidates": root_candidates[:8].tolist(),
                        "draft": out.tolist(),
                        "paths": paths.tolist(),
                        "policy": policy_info,
                    }) + chr(10))
            except Exception:
                pass
            return out
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
    if k < 0 or k > 2:
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
    try:
        _, _requested_k = _lumo_fb_sched_read_control(
            int(_lumo_fb_os.environ.get("LUMO_FB_DEPTH", "1")))
        if _requested_k == 0:
            return _lumo_fb_orig_update_draft(self, draft_token_ids)
    except Exception:
        raise
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

def _lumo_fb_copy_block_id(self, src, dst, num_slots=None):
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
                    if num_slots is not None and dst_view.ndim >= 1:
                        slots = min(int(num_slots), int(dst_view.shape[0]))
                        dst_view = dst_view[:slots]
                        src_view = src_view[:slots]
                    dst_views.append(dst_view)
                    src_views.append(src_view)
                    copied_bytes += int(dst_view.numel() * t.element_size())
            else:
                dst_view = kv[dst]
                src_view = kv[src]
                if num_slots is not None and dst_view.ndim >= 1:
                    slots = min(int(num_slots), int(dst_view.shape[0]))
                    dst_view = dst_view[:slots]
                    src_view = src_view[:slots]
                dst_views.append(dst_view)
                src_views.append(src_view)
                copied_bytes += int(dst_view.numel() * kv.element_size())
        except Exception:
            pass
    _lumo_fb_foreach_copy_(dst_views, src_views)
    return copied_bytes

def _lumo_fb_copy_block_slot_range(self, src, dst, start_slot, num_slots):
    copied_bytes = 0
    dst_views = []
    src_views = []
    start_slot = int(start_slot)
    num_slots = int(num_slots)
    if num_slots <= 0:
        return 0
    seen = set()
    for kv in getattr(self, "kv_caches", []):
        if id(kv) in seen:
            continue
        seen.add(id(kv))
        try:
            if isinstance(kv, list):
                tensors = kv
            else:
                tensors = [kv]
            for t in tensors:
                dst_view = t[dst]
                src_view = t[src]
                if dst_view.ndim >= 1:
                    end_slot = min(start_slot + num_slots, int(dst_view.shape[0]))
                    if end_slot <= start_slot:
                        continue
                    dst_view = dst_view[start_slot:end_slot]
                    src_view = src_view[start_slot:end_slot]
                dst_views.append(dst_view)
                src_views.append(src_view)
                copied_bytes += int(dst_view.numel() * t.element_size())
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
                elif len(item) >= 4 and item[0] == "kv_partial":
                    _bytes = _lumo_fb_copy_block_id(
                        self, int(item[1]), int(item[2]), int(item[3]))
                    _lumo_fb_copy_bytes += int(_bytes)
                    _lumo_fb_copy_detail.append({
                        "kind": "kv_partial", "src": int(item[1]),
                        "dst": int(item[2]), "slots": int(item[3]),
                        "bytes": int(_bytes),
                    })
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
    if (_lumo_fb_prep_os.environ.get("LUMO_FB_PATHS") != "1"
            and _lumo_fb_prep_os.environ.get("LUMO_FB_KERNEL_ROWS") != "1"):
        return ret
    flat_indices = []
    flat_values = []
    for req_id, toks in scheduler_output.scheduled_spec_decode_tokens.items():
        if (("::lumo_fb::" not in req_id and "::lumo_fb_ir::" not in req_id)
                or not toks):
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
    if (_lumo_fb_rs_os.environ.get("LUMO_FB_DEBUG") != "1"
            and _lumo_fb_rs_os.environ.get("LUMO_FB_SAMPLER_TRACE") != "1"):
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
            or len(num_draft_tokens) < 2
            or (len(num_draft_tokens) % 2 != 0
                and _lumo_fb_rs_os.environ.get("LUMO_FB_POSITION_TREE") != "1")
            ):
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
        if (_lumo_fb_rs_os.environ.get("LUMO_FB_POSITION_TREE") == "1"
                and batch_size > 2):
            pairs = [(0, i) for i in range(1, batch_size)]
        else:
            pairs = [(i, i + 1) for i in range(0, batch_size, 2)]

    is_position_tree = (
        _lumo_fb_rs_os.environ.get("LUMO_FB_POSITION_TREE") == "1"
        and batch_size % 4 == 0
        and fb_depth >= 2
    )
    if sampling_metadata.all_greedy:
        greedy_by_req = [True] * batch_size
    elif sampling_metadata.all_random:
        greedy_by_req = [False] * batch_size
    else:
        greedy_by_req = (sampling_metadata.temperature == GREEDY_TEMPERATURE).detach().cpu().tolist()

    target_argmax = target_logits.argmax(dim=-1).to(torch.int32)
    target_probs = None
    uniform_probs = None
    recovered_token_ids = None
    if not sampling_metadata.all_greedy:
        target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
        uniform_probs = generate_uniform_probs(
            draft_token_ids.shape[0], num_draft_tokens,
            sampling_metadata.generators, device)
        recovered_token_ids = sample_recovered_tokens(
            max_spec_len, num_draft_tokens, cu_num_draft_tokens,
            draft_token_ids, draft_probs, target_probs, sampling_metadata, device)

    def _lumo_fb_row_start(req_idx):
        return 0 if req_idx == 0 else int(cu_num_draft_tokens[req_idx - 1].item())

    def _lumo_fb_logical_target(row_idx, pos):
        start = _lumo_fb_row_start(row_idx)
        tok_idx = start + pos
        if greedy_by_req[row_idx]:
            return target_argmax[tok_idx]
        return _lumo_fb_sample_from_probs(
            target_probs[tok_idx], sampling_metadata.generators.get(row_idx))

    if pairs:
        # Generic prefix-tree verifier for K>=2 row groups.  The earlier
        # pairwise root coupling overwrote the parent once per sibling row,
        # drawing multiple target samples for one logical token and breaking
        # the strict-superset invariant.  Here every logical prefix gets one
        # target sample, shared by all rows with that prefix.
        parent_to_rows = {}
        for parent_idx, sibling_idx in pairs:
            parent_to_rows.setdefault(int(parent_idx), set()).add(int(parent_idx))
            parent_to_rows[int(parent_idx)].add(int(sibling_idx))
        for parent_idx, rows_set in parent_to_rows.items():
            group_rows = sorted(rows_set)
            rejected = {int(req_idx): False for req_idx in group_rows}
            prefix_targets = {}
            for pos in range(fb_depth):
                for req_idx in group_rows:
                    if rejected[int(req_idx)]:
                        continue
                    start = _lumo_fb_row_start(req_idx)
                    prefix = tuple(
                        int(draft_token_ids[start + j].item())
                        for j in range(pos)
                    )
                    if prefix not in prefix_targets:
                        prefix_targets[prefix] = _lumo_fb_logical_target(req_idx, pos)
                    tok = prefix_targets[prefix]
                    draft_id = draft_token_ids[start + pos].to(torch.int32)
                    if int(draft_id.item()) == int(tok.item()):
                        output_token_ids[req_idx, pos] = draft_id
                    else:
                        output_token_ids[req_idx, pos] = tok
                        rejected[int(req_idx)] = True
                # Continue the loop so rows on other prefixes can still verify
                # their own deeper positions after a sibling rejects.
            for req_idx in group_rows:
                if not rejected[int(req_idx)]:
                    output_token_ids[req_idx, fb_depth] = bonus_token_ids[req_idx, 0]
        _lumo_fb_sampler_debug(
            "shared_tree_prefix_generic",
            draft_token_ids, num_draft_tokens, max_spec_len,
            cu_num_draft_tokens, target_logits, bonus_token_ids,
            sampling_metadata, output_token_ids)
        return output_token_ids

    if is_position_tree:
        # Top-2-at-position-0/1 tree rows are laid out in groups of four:
        # [root0/child0, root0/child1, root1/child0, root1/child1].  The root
        # target sample is one logical request token shared by all four rows;
        # the depth-1 sample is shared by the two rows with the same root.
        for group_start in range(0, batch_size, 4):
            group_rows = list(range(group_start, group_start + 4))
            shared_root = _lumo_fb_logical_target(group_rows[0], 0)
            child_samples = {
                0: _lumo_fb_logical_target(group_rows[0], 1),
                2: _lumo_fb_logical_target(group_rows[2], 1),
            }
            for local_idx, req_idx in enumerate(group_rows):
                start = _lumo_fb_row_start(req_idx)
                rejected = False
                root_draft = draft_token_ids[start].to(torch.int32)
                if int(root_draft.item()) == int(shared_root.item()):
                    output_token_ids[req_idx, 0] = root_draft
                else:
                    output_token_ids[req_idx, 0] = shared_root
                    rejected = True
                if rejected:
                    continue

                child_group = 0 if local_idx < 2 else 2
                shared_child = child_samples[child_group]
                child_draft = draft_token_ids[start + 1].to(torch.int32)
                if int(child_draft.item()) == int(shared_child.item()):
                    output_token_ids[req_idx, 1] = child_draft
                else:
                    output_token_ids[req_idx, 1] = shared_child
                    rejected = True
                if rejected:
                    continue

                for pos in range(2, fb_depth):
                    tok_idx = start + pos
                    draft_id = draft_token_ids[tok_idx].to(torch.int32)
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
            "shared_tree_position_tree",
            draft_token_ids, num_draft_tokens, max_spec_len,
            cu_num_draft_tokens, target_logits, bonus_token_ids,
            sampling_metadata, output_token_ids)
        return output_token_ids

    if sampling_metadata.all_greedy:
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
            "shared_tree_greedy",
            draft_token_ids, num_draft_tokens, max_spec_len,
            cu_num_draft_tokens, target_logits, bonus_token_ids,
            sampling_metadata, output_token_ids)
        return output_token_ids

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
        "shared_tree_random",
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
    return ((_lumo_fb_ir_os.environ.get("LUMO_FB_PATHS") == "1"
             and _lumo_fb_ir_os.environ.get("LUMO_FB_INTERNAL_ROWS") == "1")
            or _lumo_fb_ir_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1")

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
    if k < 0 or k > 2:
        raise RuntimeError(f"LUMO_FB active K {k} unsupported in this build")
    return int(depth), int(k)

def _lumo_fb_ir_read_internal_max_commit(default_value):
    value = _lumo_fb_ir_os.environ.get("LUMO_FB_INTERNAL_MAX_COMMIT")
    path = _lumo_fb_ir_os.environ.get("LUMO_FB_CONTROL_FILE", "/logs/fb_control.json")
    if path and _lumo_fb_ir_os.path.exists(path):
        try:
            with open(path) as fh:
                payload = _lumo_fb_ir_json.load(fh)
            if payload.get("internal_max_commit") is not None:
                value = payload.get("internal_max_commit")
        except Exception:
            pass
    if value is None:
        value = default_value
    return max(0, int(value))

def _lumo_fb_ir_default_internal_max_commit():
    try:
        return int(_lumo_fb_ir_os.environ.get("LUMO_FB_DEPTH", "2"))
    except Exception:
        return 2

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
                parent_write_blocks = [
                    blk.block_id for blk in parent_blocks[
                        curr_idx + 1:curr_idx + 1 + len(new_blocks)]
                ]
                row_write_blocks = [blk.block_id for blk in new_blocks]
                overlap = sorted(set(parent_write_blocks) & set(row_write_blocks))
                if overlap:
                    raise RuntimeError(
                        "LUMO_FB_KERNEL_ROWS Mamba row write blocks alias parent: "
                        f"parent={parent.request_id} row={row_id} group={group_idx} "
                        f"overlap={overlap}")
                _lumo_fb_ir_sched_debug({
                    "event": "kernel_row_blocks",
                    "row": row_id,
                    "parent": parent.request_id,
                    "group": group_idx,
                    "curr_idx": curr_idx,
                    "read_block": getattr(read_block, "block_id", None),
                    "parent_write_blocks": parent_write_blocks,
                    "write_blocks": row_write_blocks,
                    "write_disjoint_from_parent": not bool(overlap),
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
                    if (_lumo_fb_ir_kernel_rows_enabled()
                            and _lumo_fb_ir_os.environ.get("LUMO_FB_NO_KV_PREFIX_COPY") == "1"):
                        _lumo_fb_ir_sched_debug({
                            "event": "kv_partial_prefix_copy_skipped",
                            "parent": parent.request_id,
                            "row": row_id,
                            "group": group_idx,
                            "src": src.block_id,
                            "dst": dst.block_id,
                            "slots": int(parent.num_computed_tokens) % block_size,
                            "note": "experimental no-copy mode; requires split prefix/suffix attention to be correct when prefix is not block-aligned",
                        })
                    else:
                        copies.append((
                            "kv_partial", src.block_id, dst.block_id,
                            int(parent.num_computed_tokens) % block_size))
        row_block_ids.append([blk.block_id for blk in row_blocks])
    self._lumo_fb_ir_owned_blocks = getattr(self, "_lumo_fb_ir_owned_blocks", {})
    self._lumo_fb_ir_owned_blocks[row_id] = owned
    self._lumo_fb_ir_row_block_ids = getattr(self, "_lumo_fb_ir_row_block_ids", {})
    self._lumo_fb_ir_row_block_ids[row_id] = tuple(row_block_ids)
    return tuple(row_block_ids), copies

def _lumo_fb_ir_transfer_owned_to_parent(self, parent_id, row_id,
                                         target_block_id_groups=None):
    owned_by_row = getattr(self, "_lumo_fb_ir_owned_blocks", {})
    row_block_ids_by_row = getattr(self, "_lumo_fb_ir_row_block_ids", {})
    owned = owned_by_row.pop(row_id, [])
    row_block_ids = row_block_ids_by_row.pop(row_id, None)
    if row_block_ids is None and not target_block_id_groups:
        return
    target_groups = target_block_id_groups or row_block_ids
    if not owned:
        return
    by_group = {}
    for group_idx, block in owned:
        by_group.setdefault(group_idx, {})[block.block_id] = block
    transfer_summary = []
    for group_idx, manager in enumerate(self.kv_cache_manager.coordinator.single_type_managers):
        if group_idx >= len(target_groups):
            continue
        parent_blocks = list(manager.req_to_blocks.get(parent_id, ()))
        parent_by_id = {blk.block_id: blk for blk in parent_blocks}
        owned_by_id = by_group.get(group_idx, {})
        new_ids = list(target_groups[group_idx] or [])
        if not new_ids:
            if owned_by_id:
                manager.block_pool.free_blocks(reversed(list(owned_by_id.values())))
            continue
        new_id_set = set(new_ids)
        new_blocks = []
        for block_id in new_ids:
            block = owned_by_id.get(block_id) or parent_by_id.get(block_id)
            if block is not None:
                new_blocks.append(block)
        if len(new_blocks) != len(new_ids):
            if owned_by_id:
                manager.block_pool.free_blocks(reversed(list(owned_by_id.values())))
            continue
        first_changed = None
        for idx, block_id in enumerate(new_ids):
            if idx >= len(parent_blocks) or int(parent_blocks[idx].block_id) != int(block_id):
                first_changed = idx
                break
        to_free = [
            blk for blk in parent_blocks
            if blk.block_id not in new_id_set
            and blk != getattr(manager, "_null_block", None)
        ]
        if to_free:
            manager.block_pool.free_blocks(reversed(to_free))
        unused_owned = [
            blk for block_id, blk in owned_by_id.items()
            if block_id not in new_id_set
            and blk != getattr(manager, "_null_block", None)
        ]
        if unused_owned:
            manager.block_pool.free_blocks(reversed(unused_owned))
        manager.req_to_blocks[parent_id] = new_blocks
        cached_limit = len(new_blocks) if first_changed is None else int(first_changed)
        manager.num_cached_block[parent_id] = min(
            manager.num_cached_block.get(parent_id, cached_limit),
            cached_limit,
        )
        if hasattr(manager, "_allocated_block_reqs"):
            manager._allocated_block_reqs.add(parent_id)
        transfer_summary.append({
            "group": int(group_idx),
            "adopted_owned": sorted(int(block_id) for block_id in owned_by_id if block_id in new_id_set),
            "freed_owned": sorted(int(blk.block_id) for blk in unused_owned),
            "freed_parent": sorted(int(blk.block_id) for blk in to_free),
            "first_changed": first_changed,
        })
    if transfer_summary:
        _lumo_fb_ir_sched_debug({
            "event": "kv_pointer_transfer_owned_to_parent",
            "parent": parent_id,
            "rid": row_id,
            "groups": transfer_summary,
        })

def _lumo_fb_ir_mirror_manager_blocks_from_ids(self, req_id, block_id_groups):
    if not block_id_groups:
        return False
    ok = False
    for group_idx, manager in enumerate(self.kv_cache_manager.coordinator.single_type_managers):
        if group_idx >= len(block_id_groups):
            continue
        block_ids = list(block_id_groups[group_idx] or [])
        if not block_ids:
            continue
        existing = list(manager.req_to_blocks.get(req_id, ()))
        by_id = {blk.block_id: blk for blk in existing}
        new_blocks = []
        for block_id in block_ids:
            block = by_id.get(int(block_id))
            if block is None:
                new_blocks = []
                break
            new_blocks.append(block)
        if not new_blocks:
            continue
        manager.req_to_blocks[req_id] = new_blocks
        manager.num_cached_block[req_id] = min(
            manager.num_cached_block.get(req_id, len(new_blocks)),
            len(new_blocks),
        )
        if hasattr(manager, "_allocated_block_reqs"):
            manager._allocated_block_reqs.add(req_id)
        ok = True
    return ok

def _lumo_fb_ir_copy_winner_suffix_kv_to_parent(self, parent_id, winner_id,
                                                commit_len):
    # In split partial-KV mode, internal attention rows do not own a populated
    # copy of the parent's partial prefix block. Keep the parent's KV block
    # table and copy back only the accepted speculative suffix slots.
    if not (_lumo_fb_ir_kernel_rows_enabled()
            and _lumo_fb_ir_os.environ.get("LUMO_FB_NO_KV_PREFIX_COPY") == "1"):
        return 0
    parent_state = self.requests.get(parent_id)
    winner_state = self.requests.get(winner_id)
    if parent_state is None or winner_state is None:
        _lumo_fb_ir_debug({
            "event": "split_kv_suffix_commit_copy_missing_state",
            "parent": parent_id,
            "winner": winner_id,
            "has_parent": parent_state is not None,
            "has_winner": winner_state is not None,
        })
        return 0
    try:
        start_token = int(parent_state.num_computed_tokens)
        remaining = int(commit_len)
        copied_bytes = 0
        for group_idx, manager in enumerate(self.kv_cache_manager.coordinator.single_type_managers):
            if getattr(manager, "mamba_cache_mode", None) == "align":
                continue
            if group_idx >= len(parent_state.block_ids) or group_idx >= len(winner_state.block_ids):
                continue
            block_size = int(manager.block_size)
            parent_blocks = list(parent_state.block_ids[group_idx])
            winner_blocks = list(winner_state.block_ids[group_idx])
            pos = start_token
            left = remaining
            while left > 0:
                logical_idx = pos // block_size
                slot = pos % block_size
                if logical_idx >= len(parent_blocks) or logical_idx >= len(winner_blocks):
                    break
                n = min(left, block_size - slot)
                src = int(winner_blocks[logical_idx])
                dst = int(parent_blocks[logical_idx])
                if src != dst and "_lumo_fb_copy_block_slot_range" in globals():
                    copied_bytes += int(_lumo_fb_copy_block_slot_range(
                        self, src, dst, slot, n))
                pos += n
                left -= n
        _lumo_fb_ir_debug({
            "event": "split_kv_suffix_commit_copy",
            "parent": parent_id,
            "winner": winner_id,
            "commit_len": int(commit_len),
            "bytes": int(copied_bytes),
        })
        return int(copied_bytes)
    except Exception as e:
        _lumo_fb_ir_debug({
            "event": "split_kv_suffix_commit_copy_error",
            "parent": parent_id,
            "winner": winner_id,
            "commit_len": int(commit_len),
            "error": repr(e),
        })
        return 0

def _lumo_fb_ir_promote_internal_row_state(self, parent_id, row_id, accepted_drafts):
    if not _lumo_fb_ir_kernel_rows_enabled():
        return
    row_block_ids_by_row = getattr(self, "_lumo_fb_ir_row_block_ids", None)
    if not row_block_ids_by_row or row_id not in row_block_ids_by_row:
        return
    parent = self.requests.get(parent_id)
    if parent is None:
        return
    try:
        accepted_drafts = int(accepted_drafts)
        row_groups = [list(group) for group in row_block_ids_by_row[row_id]]
        num_sched = int(getattr(self, "_lumo_fb_ir_last_sched_tokens", {}).get(parent_id, 0))
        moved = []
        for group_idx, manager in enumerate(self.kv_cache_manager.coordinator.single_type_managers):
            if getattr(manager, "mamba_cache_mode", None) != "align":
                continue
            if group_idx >= len(row_groups) or not row_groups[group_idx]:
                continue
            block_size = int(manager.block_size)
            curr_idx = max(0, _lumo_fb_ir_cdiv(
                int(parent.num_computed_tokens) + int(num_sched), block_size) - 1)
            src_idx = curr_idx + accepted_drafts + 1
            if src_idx >= len(row_groups[group_idx]):
                continue
            row_groups[group_idx][curr_idx], row_groups[group_idx][src_idx] = (
                row_groups[group_idx][src_idx], row_groups[group_idx][curr_idx])
            moved.append({
                "group": int(group_idx),
                "curr_idx": int(curr_idx),
                "src_idx": int(src_idx),
                "curr_block": int(row_groups[group_idx][curr_idx]),
            })
        row_block_ids_by_row[row_id] = tuple(row_groups)
        if moved:
            _lumo_fb_ir_sched_debug({
                "event": "kernel_promote_internal_row_state",
                "parent": parent_id,
                "rid": row_id,
                "accepted_drafts": int(accepted_drafts),
                "moves": moved,
            })
    except Exception as e:
        _lumo_fb_ir_sched_debug({
            "event": "kernel_promote_internal_row_state_error",
            "parent": parent_id,
            "rid": row_id,
            "accepted_drafts": int(accepted_drafts),
            "error": repr(e),
        })

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

def _lumo_fb_ir_promote_manager_state(self, req_id, accepted_drafts,
                                      first_sample_noop=True):
    if not _lumo_fb_ir_kernel_rows_enabled():
        return
    req = self.requests.get(req_id)
    if req is None:
        return
    accepted_drafts = int(accepted_drafts)
    _seen = getattr(self, "_lumo_fb_manager_seen_sample", None)
    if _seen is None:
        _seen = set()
        self._lumo_fb_manager_seen_sample = _seen
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
        src_offset = 0 if (_first_sample and first_sample_noop) else accepted_drafts + 1
        src_idx = curr_idx + src_offset
        if src_idx >= len(blocks):
            continue
        blocks[curr_idx], blocks[src_idx] = blocks[src_idx], blocks[curr_idx]
        manager.req_to_blocks[req_id] = blocks
        row_block_ids_by_row = getattr(self, "_lumo_fb_ir_row_block_ids", None)
        if row_block_ids_by_row is not None and req_id in row_block_ids_by_row:
            try:
                row_groups = [list(group) for group in row_block_ids_by_row[req_id]]
                if int(group_idx) < len(row_groups):
                    row_groups[int(group_idx)] = [blk.block_id for blk in blocks]
                    row_block_ids_by_row[req_id] = tuple(row_groups)
            except Exception:
                pass
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
            "first_sample_noop": bool(first_sample_noop),
            "moves": moved,
        })

def _lumo_fb_ir_update_draft_token_ids(self, draft_token_ids):
    if not _lumo_fb_ir_enabled():
        return _lumo_fb_ir_prev_update_draft(self, draft_token_ids)
    try:
        _, _requested_k = _lumo_fb_ir_read_control(
            int(_lumo_fb_ir_os.environ.get("LUMO_FB_DEPTH", "1")))
        if _requested_k == 0:
            return _lumo_fb_ir_prev_update_draft(self, draft_token_ids)
    except Exception:
        raise
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
        if requested_k >= 2 and active_depth > 0 and len(toks) % active_depth == 0 and len(toks) >= 2 * active_depth:
            path_count = len(toks) // active_depth
            paths = [
                list(toks[i * active_depth:(i + 1) * active_depth])
                for i in range(path_count)
            ]
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
        _lumo_fb_fork_t0 = _lumo_fb_ir_time.perf_counter_ns()
        _lumo_fb_ir_prepare_parent_kernel_blocks(self, parent, num_sched)
        rows = []
        for path_idx, path in enumerate(paths[1:], 1):
            row_id = _lumo_fb_ir_row_id(parent_id, path_idx)
            block_ids, row_copies = _lumo_fb_ir_alloc_blocks(self, parent, row_id, num_sched)
            copies.extend(row_copies)
            rows.append({
                "rid": row_id,
                "path_idx": int(path_idx),
                "draft": list(path),
                "block_ids": block_ids,
            })
        if _lumo_fb_ir_kernel_rows_enabled():
            owned_by_row = getattr(self, "_lumo_fb_ir_owned_blocks", {})
            seen_mamba = {}
            for row in rows:
                row_id = row["rid"]
                for group_idx, blk in owned_by_row.get(row_id, []):
                    manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]
                    if getattr(manager, "mamba_cache_mode", None) != "align":
                        continue
                    owner = seen_mamba.get(blk.block_id)
                    if owner is not None:
                        raise RuntimeError(
                            "LUMO_FB_KERNEL_ROWS sibling Mamba write block alias: "
                            f"parent={parent_id} block={blk.block_id} "
                            f"owner={owner} row={row_id}")
                    seen_mamba[blk.block_id] = row_id
            _lumo_fb_ir_sched_debug({
                "event": "kernel_row_isolation_ok",
                "parent": parent_id,
                "mamba_write_block_count": int(len(seen_mamba)),
                "rows": [row["rid"] for row in rows],
            })
        state_fork_us += int((_lumo_fb_ir_time.perf_counter_ns() - _lumo_fb_fork_t0) // 1000)
        rows_by_parent[parent_id] = {
            "paths": [list(p) for p in paths],
            "rows": rows,
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
    rows_queue = list(getattr(self, "_lumo_fb_ir_rows_queue", []) or [])
    rows_queue.append(rows_by_parent)
    self._lumo_fb_ir_rows_queue = rows_queue[-16:]
    if rows_by_parent:
        out.lumo_fb_internal_rows = rows_by_parent
        existing = list(getattr(out, "lumo_fb_block_copies", []) or [])
        out.lumo_fb_block_copies = existing + copies
    return out

def _lumo_fb_ir_update_from_output(self, scheduler_output, model_runner_output):
    if not _lumo_fb_ir_enabled():
        return _lumo_fb_ir_prev_update_output(self, scheduler_output, model_runner_output)
    rows_queue = list(getattr(self, "_lumo_fb_ir_rows_queue", []) or [])
    queue_len_before = len(rows_queue)
    queued_rows_by_parent = rows_queue.pop(0) if rows_queue else None
    self._lumo_fb_ir_rows_queue = rows_queue
    scheduler_rows_by_parent = getattr(scheduler_output, "lumo_fb_internal_rows", None)
    runner_rows_by_parent = getattr(model_runner_output, "lumo_fb_internal_rows", None)
    rows_by_parent = (
        scheduler_rows_by_parent
        or runner_rows_by_parent
        or queued_rows_by_parent
        or {}
    )
    internal_ids = []
    for bundle in rows_by_parent.values():
        for row in bundle.get("rows", []):
            internal_ids.append(row.get("rid"))
    internal_ids = [rid for rid in internal_ids if rid]
    try:
        if (_lumo_fb_ir_os.environ.get("LUMO_FB_DEBUG") == "1"
                and (_lumo_fb_ir_kernel_rows_enabled()
                     or queue_len_before
                     or scheduler_rows_by_parent
                     or runner_rows_by_parent
                     or internal_ids)):
            _lumo_fb_ir_sched_debug({
                "event": "update_boundary",
                "queue_len_before": int(queue_len_before),
                "queue_len_after": int(len(rows_queue)),
                "queued_parent_count": int(len(queued_rows_by_parent or {})),
                "scheduler_parent_count": int(len(scheduler_rows_by_parent or {})),
                "runner_parent_count": int(len(runner_rows_by_parent or {})),
                "rows_parent_count": int(len(rows_by_parent or {})),
                "internal_id_count": int(len(internal_ids)),
                "model_req_ids": list(getattr(model_runner_output, "req_ids", []) or []),
                "sample_row_count": int(len(getattr(model_runner_output, "sampled_token_ids", []) or [])),
            })
    except Exception as e:
        _lumo_fb_ir_sched_debug({
            "event": "update_boundary_error",
            "error": repr(e),
        })
    if internal_ids:
        winners = (
            getattr(model_runner_output, "lumo_fb_internal_winners", None)
            or getattr(scheduler_output, "lumo_fb_internal_winners", None)
            or {}
        )
        if not winners:
            try:
                out_req_ids = list(getattr(model_runner_output, "req_ids", []) or [])
                out_samples = list(getattr(model_runner_output, "sampled_token_ids", []) or [])
                out_by_req = dict(zip(out_req_ids, out_samples))
                for parent_id, data in rows_by_parent.items():
                    toks = out_by_req.get(parent_id)
                    if toks is None:
                        continue
                    valid = []
                    for tok in list(toks):
                        if int(tok) == -1:
                            break
                        valid.append(int(tok))
                    candidates = [(parent_id, list((data.get("paths") or [[]])[0]))]
                    for row in data.get("rows", []):
                        candidates.append((row.get("rid"), list(row.get("draft") or [])))
                    scored = []
                    for rid, draft in candidates:
                        acc = 0
                        for pos, draft_tok in enumerate(draft):
                            if pos >= max(0, len(valid) - 1):
                                break
                            if int(draft_tok) != int(valid[pos]):
                                break
                            acc += 1
                        scored.append((rid, acc))
                    scored.sort(key=lambda item: (item[1], 0 if item[0] == parent_id else -1), reverse=True)
                    if scored:
                        winners[parent_id] = {
                            "winner_rid": scored[0][0],
                            "accepted": int(scored[0][1]),
                            "reconstructed": True,
                        }
                if winners:
                    _lumo_fb_ir_sched_debug({
                        "event": "reconstructed_internal_winners",
                        "winners": winners,
                    })
            except Exception as e:
                _lumo_fb_ir_sched_debug({
                    "event": "reconstruct_internal_winners_error",
                    "error": repr(e),
                })
        _lumo_fb_ir_sched_debug({
            "event": "scheduler_winner_source",
            "winner_parent_count": int(len(winners or {})),
            "winners": winners,
        })
        winner_ids = {
            data.get("winner_rid")
            for data in winners.values()
            if isinstance(data, dict) and data.get("winner_rid")
        }
        for parent_id, data in winners.items():
            if isinstance(data, dict) and data.get("winner_rid") in internal_ids:
                runner_block_ids = data.get("runner_parent_block_ids")
                if runner_block_ids:
                    try:
                        row_block_ids_by_row = getattr(
                            self, "_lumo_fb_ir_row_block_ids", {})
                        row_block_ids_by_row[data.get("winner_rid")] = tuple(
                            [list(group) for group in runner_block_ids])
                        self._lumo_fb_ir_row_block_ids = row_block_ids_by_row
                        _lumo_fb_ir_sched_debug({
                            "event": "kernel_transfer_runner_promoted_state",
                            "parent": parent_id,
                            "rid": data.get("winner_rid"),
                            "accepted": int(data.get(
                                "state_accepted", data.get("accepted", 0))),
                            "runner_parent_mamba_idx": data.get(
                                "runner_parent_mamba_idx"),
                        })
                    except Exception as e:
                        _lumo_fb_ir_sched_debug({
                            "event": "kernel_transfer_runner_promoted_state_error",
                            "parent": parent_id,
                            "rid": data.get("winner_rid"),
                            "error": repr(e),
                        })
                        runner_block_ids = None
                if not runner_block_ids:
                    _lumo_fb_ir_promote_internal_row_state(
                        self, parent_id, data.get("winner_rid"),
                        int(data.get("state_accepted", data.get("accepted", 0))))
                _lumo_fb_ir_transfer_owned_to_parent(
                    self, parent_id, data.get("winner_rid"),
                    target_block_id_groups=runner_block_ids)
            elif (isinstance(data, dict)
                  and data.get("winner_rid") == parent_id
                  and (_lumo_fb_ir_os.environ.get("LUMO_FB_PROMOTE_PARENT_WINNERS") == "1"
                       or _lumo_fb_ir_kernel_rows_enabled())):
                runner_block_ids = data.get("runner_parent_block_ids")
                if runner_block_ids and _lumo_fb_ir_mirror_manager_blocks_from_ids(
                        self, parent_id, runner_block_ids):
                    _lumo_fb_ir_sched_debug({
                        "event": "kernel_mirror_parent_winner_state",
                        "parent": parent_id,
                        "accepted": int(data.get("state_accepted",
                                                 data.get("accepted", 0))),
                    })
                else:
                    _lumo_fb_ir_promote_manager_state(
                        self, parent_id, int(data.get("accepted", 0)),
                        first_sample_noop=False)
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
    elif (_lumo_fb_ir_kernel_rows_enabled()
          and _lumo_fb_ir_os.environ.get("LUMO_FB_KERNEL_ROWS_NOACTIVE_PROMOTE", "1") != "0"
          and hasattr(model_runner_output, "req_ids")):
        try:
            runner_promoted = getattr(
                model_runner_output, "lumo_fb_noactive_promoted", {}) or {}
            for req_id, toks in zip(model_runner_output.req_ids, model_runner_output.sampled_token_ids):
                if req_id in scheduler_output.num_scheduled_tokens:
                    promoted = runner_promoted.get(req_id)
                    if promoted and _lumo_fb_ir_mirror_manager_blocks_from_ids(
                            self, req_id, promoted.get("runner_block_ids")):
                        _lumo_fb_ir_sched_debug({
                            "event": "kernel_noactive_mirror_runner_state",
                            "rid": req_id,
                            "accepted": int(promoted.get("accepted", 0)),
                        })
                    else:
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
    return ((_lumo_fb_ir_os.environ.get("LUMO_FB_PATHS") == "1"
             and _lumo_fb_ir_os.environ.get("LUMO_FB_INTERNAL_ROWS") == "1")
            or _lumo_fb_ir_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1")

def _lumo_fb_ir_kernel_rows_enabled():
    return _lumo_fb_ir_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1"

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

def _lumo_fb_ir_superset_diag(event):
    if _lumo_fb_ir_os.environ.get("LUMO_FB_SUPERSET_DIAG") != "1":
        return
    try:
        global _LUMO_FB_SUPERSET_DIAG_FH
        try:
            _LUMO_FB_SUPERSET_DIAG_FH
        except NameError:
            _LUMO_FB_SUPERSET_DIAG_FH = open(
                "/logs/fb_superset_diag.jsonl", "a", buffering=1)
        event["ts"] = round(_lumo_fb_ir_time.time(), 4)
        _LUMO_FB_SUPERSET_DIAG_FH.write(
            _lumo_fb_ir_json.dumps(event) + chr(10))
    except Exception:
        pass

def _lumo_fb_ir_read_internal_max_commit(default_value):
    try:
        control = _lumo_fb_ir_os.environ.get("LUMO_FB_CONTROL_FILE")
        if control:
            try:
                with open(control, "r") as _fh:
                    payload = _lumo_fb_ir_json.load(_fh)
                if payload.get("internal_max_commit") is not None:
                    return int(payload.get("internal_max_commit"))
            except (OSError, ValueError, TypeError, _lumo_fb_ir_json.JSONDecodeError):
                pass
        if _lumo_fb_ir_os.environ.get("LUMO_FB_INTERNAL_MAX_COMMIT") is not None:
            return int(_lumo_fb_ir_os.environ["LUMO_FB_INTERNAL_MAX_COMMIT"])
    except (ValueError, TypeError):
        pass
    return int(default_value)

def _lumo_fb_ir_default_internal_max_commit():
    try:
        return int(_lumo_fb_ir_os.environ.get("LUMO_FB_DEPTH", "2"))
    except Exception:
        return 2

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

def _lumo_fb_ir_write_state_block_ids(self, req_id):
    req_state = self.requests.get(req_id)
    state_idx = self.mamba_state_idx.get(req_id)
    if req_state is None or state_idx is None:
        return None
    try:
        out = []
        num_spec_blocks = int(self._get_mamba_copy_bufs().mamba_spec.num_speculative_blocks)
        for gid in self._get_mamba_copy_bufs().mamba_group_ids:
            blocks = req_state.block_ids[gid]
            start = int(state_idx) + 1
            end = min(len(blocks), start + num_spec_blocks)
            out.append(list(blocks[start:end]))
        return out
    except Exception:
        return None

def _lumo_fb_ir_kernel_promote_state(self, req_id, accepted_drafts,
                                     first_sample_noop=True):
    if _lumo_fb_ir_os.environ.get("LUMO_FB_KERNEL_ROWS") != "1":
        return
    req_state = self.requests.get(req_id)
    curr_idx = self.mamba_state_idx.get(req_id)
    if req_state is None or curr_idx is None:
        return
    accepted_drafts = int(accepted_drafts)
    _seen = getattr(self, "_lumo_fb_kernel_state_seen_sample", None)
    if _seen is None:
        _seen = set()
        self._lumo_fb_kernel_state_seen_sample = _seen
    _first_sample = req_id not in _seen
    _seen.add(req_id)
    # Same closed-form rule as scheduler-side promotion. Full accept at
    # a=active_depth intentionally promotes offset active_depth+1.
    src_offset = 0 if (_first_sample and first_sample_noop) else accepted_drafts + 1
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
            # The promoted state block is already the exact accepted-prefix
            # conv+SSM state. Reset the next conv read to offset zero; carrying
            # the old accepted length would slice inside the promoted block and
            # corrupt depth>1 internal-row collapses.
            _lumo_fb_ir_set_accept_len(self, req_id, 1)
            _lumo_fb_ir_debug({
                "event": "kernel_promote_state",
                "rid": req_id,
                "accepted_drafts": int(accepted_drafts),
                "accepted_prefix_len": int(accepted_drafts) + 1,
                "first_sample_noop": bool(first_sample_noop),
                "moves": moved,
            })
    except Exception as e:
        _lumo_fb_ir_debug({
            "event": "kernel_promote_state_error",
            "rid": req_id,
            "accepted_drafts": int(accepted_drafts),
            "first_sample_noop": bool(first_sample_noop),
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
        self._lumo_fb_accept_lens = getattr(self, "_lumo_fb_accept_lens", {})
        self._lumo_fb_accept_lens[req_id] = int(accept_len)
        self.input_batch.num_accepted_tokens_cpu[idx] = int(accept_len)
        self.input_batch.num_accepted_tokens_cpu_tensor[idx] = int(accept_len)
        if hasattr(self, "num_accepted_tokens"):
            self.num_accepted_tokens.np[idx] = int(accept_len)
    except Exception as e:
        _lumo_fb_ir_debug({
            "event": "set_accept_len_error",
            "rid": req_id,
            "accept_len": int(accept_len),
            "error": repr(e),
        })

def _lumo_fb_ir_head(value, limit=16):
    try:
        if value is None:
            return None
        if hasattr(value, "detach"):
            tensor = value.detach()
            shape = list(tensor.shape)
            flat = tensor.reshape(-1).cpu().tolist()
            return {"shape": shape, "head": flat[:limit]}
        if isinstance(value, dict):
            return {
                str(k): _lumo_fb_ir_head(v, limit)
                for k, v in list(value.items())[:limit]
            }
        if isinstance(value, (list, tuple)):
            return [_lumo_fb_ir_head(v, limit) for v in list(value)[:limit]]
        return value
    except Exception as e:
        return {"error": repr(e), "type": type(value).__name__}

def _lumo_fb_ir_debug_pre_base_update(self, label, scheduler_output,
                                      sampler_output,
                                      spec_decode_metadata=None,
                                      common_attn_metadata=None):
    if _lumo_fb_ir_os.environ.get("LUMO_FB_DEBUG") != "1":
        return
    try:
        req_ids = list(getattr(self.input_batch, "req_ids", []) or [])
        sched_tokens = dict(getattr(scheduler_output, "num_scheduled_tokens", {}) or {})
        drafts = dict(getattr(scheduler_output, "scheduled_spec_decode_tokens", {}) or {})
        active = dict(getattr(self, "_lumo_fb_ir_active", {}) or {})
        rows = []
        for rid in req_ids:
            rows.append({
                "rid": rid,
                "is_internal": _lumo_fb_ir_is_row_id(rid),
                "parent": active.get(rid, rid),
                "num_scheduled": sched_tokens.get(rid),
                "draft": list(drafts.get(rid, []) or [])[:8],
                "mamba_idx": self.mamba_state_idx.get(rid),
                "state_block_ids": _lumo_fb_ir_state_block_ids(self, rid),
                "write_state_block_ids": _lumo_fb_ir_write_state_block_ids(self, rid),
            })
        event = {
            "event": "pre_base_update",
            "label": label,
            "input_req_ids": req_ids,
            "active": active,
            "rows": rows,
            "scheduler_total_num_scheduled_tokens": getattr(
                scheduler_output, "total_num_scheduled_tokens", None),
            "scheduler_num_scheduled_tokens": sched_tokens,
            "sampler_sampled_token_ids": _lumo_fb_ir_head(
                getattr(sampler_output, "sampled_token_ids", None)),
            "input_num_accepted_tokens_cpu": _lumo_fb_ir_head(
                getattr(self.input_batch, "num_accepted_tokens_cpu", None)),
            "input_num_accepted_tokens_cpu_tensor": _lumo_fb_ir_head(
                getattr(self.input_batch, "num_accepted_tokens_cpu_tensor", None)),
        }
        if hasattr(self, "num_accepted_tokens"):
            event["runner_num_accepted_tokens"] = _lumo_fb_ir_head(
                getattr(getattr(self, "num_accepted_tokens", None), "np", None))
        if spec_decode_metadata is not None:
            event["spec"] = {
                "draft_token_ids": _lumo_fb_ir_head(
                    getattr(spec_decode_metadata, "draft_token_ids", None)),
                "num_draft_tokens": _lumo_fb_ir_head(
                    getattr(spec_decode_metadata, "num_draft_tokens", None)),
                "cu_num_draft_tokens": _lumo_fb_ir_head(
                    getattr(spec_decode_metadata, "cu_num_draft_tokens", None)),
                "cu_num_sampled_tokens": _lumo_fb_ir_head(
                    getattr(spec_decode_metadata, "cu_num_sampled_tokens", None)),
                "target_logits_indices": _lumo_fb_ir_head(
                    getattr(spec_decode_metadata, "target_logits_indices", None)),
                "bonus_logits_indices": _lumo_fb_ir_head(
                    getattr(spec_decode_metadata, "bonus_logits_indices", None)),
                "logits_indices": _lumo_fb_ir_head(
                    getattr(spec_decode_metadata, "logits_indices", None)),
            }
        if common_attn_metadata is not None:
            event["attn"] = {
                "num_reqs": getattr(common_attn_metadata, "num_reqs", None),
                "num_actual_tokens": getattr(
                    common_attn_metadata, "num_actual_tokens", None),
                "max_query_len": getattr(common_attn_metadata, "max_query_len", None),
                "query_start_loc": _lumo_fb_ir_head(
                    getattr(common_attn_metadata, "query_start_loc", None)),
                "query_start_loc_cpu": _lumo_fb_ir_head(
                    getattr(common_attn_metadata, "query_start_loc_cpu", None)),
                "seq_lens": _lumo_fb_ir_head(
                    getattr(common_attn_metadata, "seq_lens", None)),
                "_seq_lens_cpu": _lumo_fb_ir_head(
                    getattr(common_attn_metadata, "_seq_lens_cpu", None)),
                "_num_computed_tokens_cpu": _lumo_fb_ir_head(
                    getattr(common_attn_metadata, "_num_computed_tokens_cpu", None)),
                "block_table_tensor": _lumo_fb_ir_head(
                    getattr(common_attn_metadata, "block_table_tensor", None), 32),
                "slot_mapping": _lumo_fb_ir_head(
                    getattr(common_attn_metadata, "slot_mapping", None), 32),
            }
        _lumo_fb_ir_debug(event)
    except Exception as e:
        _lumo_fb_ir_debug({
            "event": "pre_base_update_debug_error",
            "label": label,
            "error": repr(e),
        })

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
        _lumo_fb_ir_superset_diag({
            "event": "runner_expanded_active",
            "active_count": int(len(active)),
            "parents": sorted(list(set(active.values())))[:8],
        })
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
                "write_state_block_ids": _lumo_fb_ir_write_state_block_ids(self, rid),
                "parent_write_state_block_ids": _lumo_fb_ir_write_state_block_ids(self, parent),
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
    except Exception as e:
        _lumo_fb_ir_debug({
            "event": "filter_drafts_error",
            "error": repr(e),
        })

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

def _lumo_fb_ir_copy_winner_suffix_kv_to_parent(self, parent_id, winner_id,
                                                commit_len):
    # Runner-side companion to split partial-KV attention. The internal row
    # verifies using dense suffix K/V plus shared parent prefix KV, so on an
    # internal win only the accepted suffix slots need to be copied into the
    # parent's existing attention blocks. Prefix KV is never materialized per row.
    if not (_lumo_fb_ir_kernel_rows_enabled()
            and _lumo_fb_ir_os.environ.get("LUMO_FB_NO_KV_PREFIX_COPY") == "1"):
        return 0
    parent_state = self.requests.get(parent_id)
    winner_state = self.requests.get(winner_id)
    if parent_state is None or winner_state is None:
        _lumo_fb_ir_debug({
            "event": "split_kv_suffix_commit_copy_missing_state",
            "parent": parent_id,
            "winner": winner_id,
            "has_parent": parent_state is not None,
            "has_winner": winner_state is not None,
        })
        return 0
    try:
        start_token = int(parent_state.num_computed_tokens)
        remaining = int(commit_len)
        copied_bytes = 0
        details = []
        try:
            mamba_group_ids = set(int(g) for g in self._get_mamba_copy_bufs().mamba_group_ids)
        except Exception:
            mamba_group_ids = set()
        for group_idx, manager in enumerate(self.kv_cache_config.kv_cache_groups):
            if int(group_idx) in mamba_group_ids:
                continue
            try:
                # group ids in req_state.block_ids match scheduler KV groups;
                # the runtime cache group object carries the same block size.
                block_size = int(getattr(manager.kv_cache_spec, "block_size", 0))
            except Exception:
                block_size = 0
            if block_size <= 0:
                continue
            if group_idx >= len(parent_state.block_ids) or group_idx >= len(winner_state.block_ids):
                continue
            parent_blocks = list(parent_state.block_ids[group_idx])
            winner_blocks = list(winner_state.block_ids[group_idx])
            pos = start_token
            left = remaining
            while left > 0:
                logical_idx = pos // block_size
                slot = pos % block_size
                if logical_idx >= len(parent_blocks) or logical_idx >= len(winner_blocks):
                    break
                n = min(left, block_size - slot)
                src = int(winner_blocks[logical_idx])
                dst = int(parent_blocks[logical_idx])
                if src != dst and "_lumo_fb_copy_block_slot_range" in globals():
                    copied = int(_lumo_fb_copy_block_slot_range(
                        self, src, dst, slot, n))
                    copied_bytes += copied
                    details.append({
                        "group": int(group_idx), "src": src, "dst": dst,
                        "slot": int(slot), "slots": int(n),
                        "bytes": int(copied),
                    })
                pos += n
                left -= n
        _lumo_fb_ir_debug({
            "event": "split_kv_suffix_commit_copy",
            "parent": parent_id,
            "winner": winner_id,
            "commit_len": int(commit_len),
            "bytes": int(copied_bytes),
            "details": details[:16],
        })
        return int(copied_bytes)
    except Exception as e:
        _lumo_fb_ir_debug({
            "event": "split_kv_suffix_commit_copy_error",
            "parent": parent_id,
            "winner": winner_id,
            "commit_len": int(commit_len),
            "error": repr(e),
        })
        return 0

def _lumo_fb_ir_pointer_swap_winner_suffix_to_parent(self, parent_id, winner_id,
                                                     commit_len):
    result = {
        "event": "split_kv_suffix_pointer_swap",
        "parent": parent_id,
        "winner": winner_id,
        "commit_len": int(commit_len),
        "bytes": 0,
        "groups": [],
        "swapped_blocks": 0,
        "partial_head": False,
    }
    if not (_lumo_fb_ir_kernel_rows_enabled()
            and _lumo_fb_ir_os.environ.get("LUMO_FB_NO_KV_PREFIX_COPY") == "1"):
        result["skipped"] = "disabled"
        return result
    parent_state = self.requests.get(parent_id)
    winner_state = self.requests.get(winner_id)
    if parent_state is None or winner_state is None:
        result["skipped"] = "missing_state"
        result["has_parent"] = parent_state is not None
        result["has_winner"] = winner_state is not None
        _lumo_fb_ir_debug(result)
        return result
    try:
        start_token = int(parent_state.num_computed_tokens)
        end_token = start_token + int(commit_len)
        result["start_token"] = int(start_token)
        result["end_token"] = int(end_token)
        try:
            mamba_group_ids = set(
                int(g) for g in self._get_mamba_copy_bufs().mamba_group_ids)
        except Exception:
            mamba_group_ids = set()
        merged_groups = [list(group) for group in parent_state.block_ids]
        for group_idx, manager in enumerate(self.kv_cache_config.kv_cache_groups):
            if int(group_idx) in mamba_group_ids:
                continue
            try:
                block_size = int(getattr(manager.kv_cache_spec, "block_size", 0))
            except Exception:
                block_size = 0
            if block_size <= 0:
                continue
            if group_idx >= len(parent_state.block_ids) or group_idx >= len(winner_state.block_ids):
                continue
            parent_blocks = list(parent_state.block_ids[group_idx])
            winner_blocks = list(winner_state.block_ids[group_idx])
            partial_head = (int(commit_len) > 0 and (start_token % block_size) != 0)
            if partial_head:
                result["partial_head"] = True
            first_idx = ((start_token + block_size - 1) // block_size
                         if partial_head else start_token // block_size)
            last_excl = (end_token + block_size - 1) // block_size
            swapped = []
            skipped = []
            if partial_head:
                skipped.append({
                    "logical_idx": int(start_token // block_size),
                    "reason": "partial_head",
                    "slot": int(start_token % block_size),
                })
            for logical_idx in range(int(first_idx), int(last_excl)):
                block_start = logical_idx * block_size
                if block_start >= end_token:
                    continue
                if logical_idx >= len(parent_blocks) or logical_idx >= len(winner_blocks):
                    skipped.append({
                        "logical_idx": int(logical_idx),
                        "reason": "missing_block",
                    })
                    continue
                src = int(winner_blocks[logical_idx])
                dst = int(parent_blocks[logical_idx])
                if src == dst:
                    continue
                parent_blocks[logical_idx] = src
                swapped.append({
                    "logical_idx": int(logical_idx),
                    "src": src,
                    "dst": dst,
                    "block_start": int(block_start),
                })
            if swapped:
                merged_groups[group_idx] = parent_blocks
                result["swapped_blocks"] += len(swapped)
            if swapped or skipped:
                result["groups"].append({
                    "group": int(group_idx),
                    "block_size": int(block_size),
                    "partial_head": bool(partial_head),
                    "first_idx": int(first_idx),
                    "last_excl": int(last_excl),
                    "swapped": swapped[:16],
                    "skipped": skipped[:16],
                })
        parent_state.block_ids = tuple(merged_groups)
        _lumo_fb_ir_debug(result)
        return result
    except Exception as e:
        result["error"] = repr(e)
        _lumo_fb_ir_debug(result)
        return result

def _lumo_fb_ir_prune_after_sample(self, scheduler_output, sampler_output,
                                   spec_decode_metadata=None,
                                   common_attn_metadata=None):
    active = getattr(self, "_lumo_fb_ir_active", None)
    if not (_lumo_fb_ir_runner_enabled() and active):
        if _lumo_fb_ir_runner_enabled():
            try:
                _count = int(getattr(self, "_lumo_fb_prune_noactive_diag_count", 0))
                if _count < 8:
                    self._lumo_fb_prune_noactive_diag_count = _count + 1
                    _lumo_fb_ir_superset_diag({
                        "event": "prune_no_active",
                        "has_active_attr": hasattr(self, "_lumo_fb_ir_active"),
                        "input_req_count": len(list(
                            getattr(self.input_batch, "req_ids", []) or [])),
                    })
            except Exception:
                pass
        return sampler_output, spec_decode_metadata, common_attn_metadata
    internal_ids = set(active.keys())
    req_ids = list(self.input_batch.req_ids)
    keep = [i for i, rid in enumerate(req_ids) if rid not in internal_ids]
    winners = {}
    _lumo_fb_ir_superset_diag({
        "event": "prune_active_entry",
        "active_count": int(len(active)),
        "input_req_count": int(len(req_ids)),
        "internal_count": int(len(internal_ids)),
    })
    try:
        raw = sampler_output.sampled_token_ids.detach().cpu().tolist()
        draft_by_rid = {
            rid: list(toks)
            for rid, toks in getattr(scheduler_output, "scheduled_spec_decode_tokens", {}).items()
        }
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
                    "draft": draft_by_rid.get(rid),
                    "mamba_idx": self.mamba_state_idx.get(rid),
                    "state_block_ids": _lumo_fb_ir_state_block_ids(self, rid),
                    "write_state_block_ids": _lumo_fb_ir_write_state_block_ids(self, rid),
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
            by_parent.setdefault(parent, []).append((
                rid, valid, max(0, len(valid) - 1), draft_by_rid.get(rid, [])))
        for parent, row_data in by_parent.items():
            ordered = sorted(row_data, key=lambda item: 0 if item[0] == parent else 1)
            canonical = {}
            max_depth = 0
            for _rid, _valid, _raw_acc, _draft in ordered:
                max_depth = max(max_depth, len(_draft or []))
            for depth in range(max_depth):
                prefixes = []
                for _rid, _valid, _raw_acc, _draft in ordered:
                    if len(_draft or []) > depth:
                        prefix = tuple(_draft[:depth])
                        if prefix not in prefixes:
                            prefixes.append(prefix)
                for prefix in prefixes:
                    for _rid, _valid, _raw_acc, _draft in ordered:
                        if len(_valid) > depth and len(_draft or []) > depth and tuple(_draft[:depth]) == prefix:
                            canonical[prefix] = int(_valid[depth])
                            break
            scored = []
            for rid, valid, raw_acc, draft in row_data:
                tree_acc = 0
                for depth, draft_tok in enumerate(draft or []):
                    tok = canonical.get(tuple(draft[:depth]))
                    if tok is None or int(draft_tok) != int(tok):
                        break
                    tree_acc += 1
                tree_acc = min(int(tree_acc), int(raw_acc))
                if rid != parent:
                    # Bound only to active draft depth. The row tree may branch
                    # for the first few positions and then continue linearly,
                    # but the full row is still a verified candidate path.
                    tree_acc = min(
                        tree_acc,
                        _lumo_fb_ir_read_internal_max_commit(
                            _lumo_fb_ir_default_internal_max_commit()))
                scored.append((rid, valid, raw_acc, draft, tree_acc))
            scored.sort(key=lambda item: (item[4], 0 if item[0] == parent else -1), reverse=True)
            winner_rid, winner_tokens, winner_raw_acc, winner_draft, winner_acc = scored[0]
            winner_is_internal = winner_rid != parent
            path0_raw_acc = 0
            path0_tree_acc = 0
            raw_best_acc = 0
            tree_best_acc = 0
            if scored:
                try:
                    ordered_scored = sorted(
                        scored, key=lambda item: 0 if item[0] == parent else 1)
                    path0_raw_acc = int(ordered_scored[0][2])
                    path0_tree_acc = int(ordered_scored[0][4])
                    raw_best_acc = max(int(item[2]) for item in scored)
                    tree_best_acc = max(int(item[4]) for item in scored)
                except Exception:
                    pass
            partial_head_fallback_result = None
            if winner_is_internal:
                try:
                    parent_state = self.requests.get(parent)
                    start_token = int(parent_state.num_computed_tokens) if parent_state is not None else 0
                    mamba_group_ids = set(
                        int(_gid) for _gid in self._get_mamba_copy_bufs().mamba_group_ids)
                    partial_groups = []
                    for group_idx, manager in enumerate(self.kv_cache_config.kv_cache_groups):
                        if int(group_idx) in mamba_group_ids:
                            continue
                        block_size = int(getattr(manager.kv_cache_spec, "block_size", 0))
                        if block_size > 0 and (start_token % block_size) != 0:
                            partial_groups.append({
                                "group": int(group_idx),
                                "block_size": int(block_size),
                                "slot": int(start_token % block_size),
                            })
                    if partial_groups:
                        partial_head_fallback_result = {
                            "event": "split_kv_suffix_pointer_swap",
                            "parent": parent,
                            "winner": winner_rid,
                            "commit_len": int(winner_acc) + 1,
                            "bytes": 0,
                            "groups": partial_groups,
                            "swapped_blocks": 0,
                            "partial_head": True,
                            "skipped": "partial_head_parent_fallback",
                            "start_token": int(start_token),
                        }
                        for _item in scored:
                            if _item[0] == parent:
                                winner_rid, winner_tokens, winner_raw_acc, winner_draft, winner_acc = _item
                                winner_is_internal = False
                                break
                except Exception as e:
                    partial_head_fallback_result = {
                        "event": "split_kv_suffix_pointer_swap",
                        "parent": parent,
                        "winner": winner_rid,
                        "commit_len": int(winner_acc) + 1,
                        "bytes": 0,
                        "swapped_blocks": 0,
                        "partial_head": True,
                        "skipped": "partial_head_detection_error",
                        "error": repr(e),
                    }
            state_accepted_drafts = int(winner_acc)
            def _lumo_fb_target_commit_tokens(_draft, _valid, _acc, _include_bonus=True):
                _draft = list(_draft or [])
                _valid = list(_valid or [])
                _out = []
                _limit = int(_acc) + (1 if _include_bonus else 0)
                for _depth in range(_limit):
                    _tok = canonical.get(tuple(_draft[:_depth]))
                    if _tok is None and _depth < len(_valid):
                        _tok = _valid[_depth]
                    if _tok is None:
                        break
                    _out.append(int(_tok))
                return _out
            winner_commit_tokens = _lumo_fb_target_commit_tokens(
                winner_draft, winner_tokens, winner_acc,
                _include_bonus=True)
            expected_commit_len = int(winner_acc) + 1
            if expected_commit_len <= 0:
                expected_commit_len = 1
                winner_commit_tokens = _lumo_fb_target_commit_tokens(
                    winner_draft, winner_tokens, 0, _include_bonus=True)
                state_accepted_drafts = 0
            if len(winner_commit_tokens) != expected_commit_len:
                raise RuntimeError(
                    "LUMO_FB target commit synthesis failed: "
                    f"parent={parent} winner={winner_rid} accepted={winner_acc} "
                    f"commit_len={len(winner_commit_tokens)}")
            second_pos0_capture = False
            second_pos1_capture = False
            try:
                diag_rows = sorted(row_data, key=lambda item: 0 if item[0] == parent else 1)
                parent_draft = []
                for rid, _valid, _raw_acc, draft in diag_rows:
                    if rid == parent:
                        parent_draft = list(draft or [])
                        break
                if parent_draft and winner_commit_tokens:
                    roots = []
                    for _rid, _valid, _raw_acc, draft in diag_rows:
                        draft = list(draft or [])
                        if draft and draft[0] not in roots:
                            roots.append(draft[0])
                    second_pos0_capture = (
                        len(roots) > 1
                        and int(winner_commit_tokens[0]) == int(roots[1])
                        and int(winner_commit_tokens[0]) != int(parent_draft[0])
                    )
                    if len(winner_commit_tokens) > 1:
                        kids = []
                        for _rid, _valid, _raw_acc, draft in diag_rows:
                            draft = list(draft or [])
                            if (len(draft) > 1 and parent_draft
                                    and int(draft[0]) == int(parent_draft[0])
                                    and draft[1] not in kids):
                                kids.append(draft[1])
                        second_pos1_capture = (
                            len(kids) > 1
                            and int(winner_commit_tokens[0]) == int(parent_draft[0])
                            and int(winner_commit_tokens[1]) == int(kids[1])
                            and int(winner_commit_tokens[1]) != int(kids[0])
                        )
            except Exception:
                second_pos0_capture = False
                second_pos1_capture = False
            parent_idx = req_ids.index(parent) if parent in req_ids else None
            winner_idx = req_ids.index(winner_rid) if winner_rid in req_ids else None
            runner_parent_block_ids = None
            runner_parent_mamba_idx = None
            split_kv_parent_state_applied = False
            kv_pointer_swap_result = partial_head_fallback_result
            if (winner_is_internal
                    and _lumo_fb_ir_kernel_rows_enabled()
                    and _lumo_fb_ir_os.environ.get("LUMO_FB_NO_KV_PREFIX_COPY") == "1"):
                try:
                    parent_state = self.requests.get(parent)
                    winner_state = self.requests.get(winner_rid)
                    if parent_state is not None and winner_state is not None:
                        kv_pointer_swap_result = _lumo_fb_ir_pointer_swap_winner_suffix_to_parent(
                            self, parent, winner_rid, len(winner_commit_tokens))
                        try:
                            _mamba_group_ids = set(
                                int(_gid) for _gid in self._get_mamba_copy_bufs().mamba_group_ids)
                        except Exception:
                            _mamba_group_ids = set()
                        merged_groups = []
                        for _gid, _parent_group in enumerate(parent_state.block_ids):
                            if (_gid < len(winner_state.block_ids)
                                    and int(_gid) in _mamba_group_ids):
                                merged_groups.append(list(winner_state.block_ids[_gid]))
                            else:
                                merged_groups.append(list(_parent_group))
                        parent_state.block_ids = tuple(merged_groups)
                        pidx = self.input_batch.req_id_to_index.get(parent)
                        if pidx is not None:
                            self.input_batch.block_table.clear_row(pidx)
                            self.input_batch.block_table.add_row(parent_state.block_ids, pidx)
                        if winner_rid in self.mamba_state_idx:
                            self.mamba_state_idx[parent] = self.mamba_state_idx[winner_rid]
                        split_kv_parent_state_applied = True
                except Exception as e:
                    _lumo_fb_ir_debug({
                        "event": "split_kv_parent_state_apply_error",
                        "parent": parent,
                        "winner": winner_rid,
                        "error": repr(e),
                    })
            if winner_idx is not None:
                try:
                    sampler_output.sampled_token_ids[winner_idx].fill_(-1)
                    for _pos, _tok in enumerate(winner_commit_tokens):
                        sampler_output.sampled_token_ids[winner_idx, _pos] = int(_tok)
                except Exception:
                    pass
            if parent_idx is not None and winner_idx is not None and winner_idx != parent_idx:
                sampler_output.sampled_token_ids[parent_idx].copy_(sampler_output.sampled_token_ids[winner_idx])
                try:
                    parent_state = self.requests.get(parent)
                    winner_state = self.requests.get(winner_rid)
                    if parent_state is not None and winner_state is not None:
                        if (not split_kv_parent_state_applied
                                and _lumo_fb_ir_kernel_rows_enabled()
                                and _lumo_fb_ir_os.environ.get("LUMO_FB_NO_KV_PREFIX_COPY") == "1"):
                            kv_pointer_swap_result = _lumo_fb_ir_pointer_swap_winner_suffix_to_parent(
                                self, parent, winner_rid, len(winner_commit_tokens))
                            try:
                                _mamba_group_ids = set(
                                    int(_gid) for _gid in self._get_mamba_copy_bufs().mamba_group_ids)
                            except Exception:
                                _mamba_group_ids = set()
                            merged_groups = []
                            for _gid, _parent_group in enumerate(parent_state.block_ids):
                                if (_gid < len(winner_state.block_ids)
                                        and int(_gid) in _mamba_group_ids):
                                    merged_groups.append(list(winner_state.block_ids[_gid]))
                                else:
                                    merged_groups.append(list(_parent_group))
                            parent_state.block_ids = tuple(merged_groups)
                        else:
                            parent_state.block_ids = tuple([list(group) for group in winner_state.block_ids])
                        pidx = self.input_batch.req_id_to_index.get(parent)
                        if pidx is not None:
                            self.input_batch.block_table.clear_row(pidx)
                            self.input_batch.block_table.add_row(parent_state.block_ids, pidx)
                        if winner_rid in self.mamba_state_idx:
                            self.mamba_state_idx[parent] = self.mamba_state_idx[winner_rid]
                except Exception:
                    pass
            # Promote exactly once in the runner for both parent and internal
            # winners, then make the scheduler manager mirror this exact block
            # table.  Doing independent scheduler-side arithmetic promotion can
            # desync row allocation from the runner-visible parent state.
            _lumo_fb_ir_set_accept_len(self, parent, int(state_accepted_drafts) + 1)
            if _lumo_fb_ir_kernel_rows_enabled():
                _lumo_fb_ir_kernel_promote_state(
                    self, parent, state_accepted_drafts, first_sample_noop=False)
            if _lumo_fb_ir_kernel_rows_enabled():
                try:
                    parent_state = self.requests.get(parent)
                    if parent_state is not None:
                        runner_parent_block_ids = tuple(
                            [list(group) for group in parent_state.block_ids])
                    runner_parent_mamba_idx = self.mamba_state_idx.get(parent)
                except Exception:
                    runner_parent_block_ids = None
                    runner_parent_mamba_idx = None
            winners[parent] = {
                "winner_rid": winner_rid,
                "winner_idx": 0 if winner_rid == parent else 1,
                "accepted": int(state_accepted_drafts),
                "tree_accepted": int(winner_acc),
                "state_accepted": int(state_accepted_drafts),
                "commit_len": int(len(winner_commit_tokens)),
                "path0_raw_acc": int(path0_raw_acc),
                "path0_tree_acc": int(path0_tree_acc),
                "raw_best_acc": int(raw_best_acc),
                "tree_best_acc": int(tree_best_acc),
                "winner_is_internal": bool(winner_is_internal),
                "second_pos0_capture": bool(second_pos0_capture),
                "second_pos1_capture": bool(second_pos1_capture),
                "accept_lens": [
                    int(acc) for rid, toks, raw_acc, draft, acc in sorted(
                        scored, key=lambda item: 0 if item[0] == parent else 1)
                ],
                "raw_accept_lens": [
                    int(raw_acc) for rid, toks, raw_acc, draft, acc in sorted(
                        scored, key=lambda item: 0 if item[0] == parent else 1)
                ],
                "commit_tokens": list(winner_commit_tokens),
                "winner_raw_tokens": list(winner_tokens[:len(winner_commit_tokens)]),
                "commit_source": "canonical_target",
                "internal_bonus_deferred": False,
                "kv_pointer_swap": kv_pointer_swap_result,
                "runner_parent_block_ids": runner_parent_block_ids,
                "runner_parent_mamba_idx": runner_parent_mamba_idx,
            }
            if _lumo_fb_ir_os.environ.get("LUMO_FB_SUPERSET_DIAG") == "1":
                try:
                    global _LUMO_FB_SUPERSET_DIAG_FH
                    try:
                        _LUMO_FB_SUPERSET_DIAG_FH
                    except NameError:
                        _LUMO_FB_SUPERSET_DIAG_FH = open(
                            "/logs/fb_superset_diag.jsonl", "a", buffering=1)
                    _diag_rows = []
                    for _rid, _valid, _raw_acc, _draft, _tree_acc in sorted(
                            scored, key=lambda item: 0 if item[0] == parent else 1):
                        _diag_rows.append({
                            "rid": _rid,
                            "is_parent": _rid == parent,
                            "raw_acc": int(_raw_acc),
                            "tree_acc": int(_tree_acc),
                            "draft": list(_draft or [])[:8],
                            "valid": list(_valid or [])[:8],
                        })
                    _LUMO_FB_SUPERSET_DIAG_FH.write(_lumo_fb_ir_json.dumps({
                        "ts": round(_lumo_fb_ir_time.time(), 4),
                        "parent": parent,
                        "winner_rid": winner_rid,
                        "winner_is_internal": bool(winner_is_internal),
                        "winner_acc": int(winner_acc),
                        "state_accepted": int(state_accepted_drafts),
                        "commit_len": int(len(winner_commit_tokens)),
                        "path0_raw_acc": int(path0_raw_acc),
                        "path0_tree_acc": int(path0_tree_acc),
                        "raw_best_acc": int(raw_best_acc),
                        "tree_best_acc": int(tree_best_acc),
                        "second_pos0_capture": bool(second_pos0_capture),
                        "second_pos1_capture": bool(second_pos1_capture),
                        "commit_tokens": list(winner_commit_tokens)[:8],
                        "kv_pointer_swap": kv_pointer_swap_result,
                        "rows": _diag_rows,
                    }) + chr(10))
                except Exception:
                    pass
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
    except Exception as e:
        _lumo_fb_ir_debug({
            "event": "prune_after_sample_error",
            "error": repr(e),
        })
        _lumo_fb_ir_superset_diag({
            "event": "prune_after_sample_error",
            "error": repr(e),
            "active_count": int(len(active)),
            "input_req_count": int(len(req_ids)),
        })
    if winners:
        self._lumo_fb_ir_last_winners = winners
        scheduler_output.lumo_fb_internal_winners = winners
        try:
            sampler_output.lumo_fb_internal_winners = winners
            sampler_output.lumo_fb_internal_rows = getattr(
                scheduler_output, "lumo_fb_internal_rows", None)
        except Exception:
            pass
        # A K>=2 internal-row winner collapse has already promoted the parent
        # state once in this active path.  The same parent sample can later
        # flow through the no-active kernel-row hook after internal rows are
        # pruned; skip that parent exactly once to avoid promoting it twice.
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
    last_winners = getattr(self, "_lumo_fb_ir_last_winners", None)
    if last_winners:
        try:
            model_output = getattr(output, "model_runner_output", output)
            req_ids = list(getattr(model_output, "req_ids", []) or [])
            for parent, data in list(last_winners.items()):
                if parent in req_ids and isinstance(data, dict):
                    idx = req_ids.index(parent)
                    commit_tokens = [int(t) for t in list(data.get("commit_tokens") or [])]
                    if commit_tokens:
                        model_output.sampled_token_ids[idx] = commit_tokens
            model_output.lumo_fb_internal_winners = last_winners
            if hasattr(output, "model_runner_output"):
                output.model_runner_output = model_output
            else:
                output = model_output
            _lumo_fb_ir_debug({
                "event": "patched_model_output_winners",
                "winners": last_winners,
            })
        except Exception as e:
            _lumo_fb_ir_debug({
                "event": "patch_model_output_winners_error",
                "error": repr(e),
            })
        self._lumo_fb_ir_last_winners = None
    active = getattr(self, "_lumo_fb_ir_active", None)
    if not active:
        if (_lumo_fb_ir_os.environ.get("LUMO_FB_KERNEL_ROWS") == "1"
                and _lumo_fb_ir_os.environ.get("LUMO_FB_KERNEL_ROWS_NOACTIVE_PROMOTE", "1") != "0"):
            try:
                model_output = getattr(output, "model_runner_output", output)
                raw_req_ids = list(getattr(model_output, "req_ids", []) or [])
                raw_samples = list(getattr(model_output, "sampled_token_ids", []) or [])
                noactive_promoted = {}
                # Keep the runner's cached block table in sync with the
                # scheduler-side manager promotion for K=1/no-internal events.
                skip_once = getattr(
                    self, "_lumo_fb_ir_skip_noactive_promote_once", None)
                for rid, toks in zip(raw_req_ids, raw_samples):
                    if skip_once is not None and rid in skip_once:
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
                    _accepted = _lumo_fb_ir_accepted_from_tokens(toks)
                    _lumo_fb_ir_kernel_promote_state(
                        self, rid, _accepted)
                    try:
                        _state = self.requests.get(rid)
                        if _state is not None:
                            noactive_promoted[rid] = {
                                "accepted": int(_accepted),
                                "runner_block_ids": tuple(
                                    [list(group) for group in _state.block_ids]),
                            }
                    except Exception:
                        pass
                if noactive_promoted:
                    model_output.lumo_fb_noactive_promoted = noactive_promoted
                    if hasattr(output, "model_runner_output"):
                        output.model_runner_output = model_output
                    else:
                        output = model_output
                self.input_batch.refresh_metadata()
            except Exception as e:
                _lumo_fb_ir_debug({
                    "event": "kernel_promote_noactive_error",
                    "error": repr(e),
                })
        return output
    if "_lumo_fb_ir_prune_after_sample" in globals():
        # The pre-update prune path must see the internal rows and active map so
        # it can collapse the winning row before vLLM mutates persistent state.
        _lumo_fb_ir_superset_diag({
            "event": "sample_tokens_active_preserved",
            "active_count": int(len(active)),
            "output_req_count": len(list(getattr(
                getattr(output, "model_runner_output", output), "req_ids", []) or [])),
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
    debug_call = nl.join([
        '        if "_lumo_fb_ir_debug_pre_base_update" in globals():',
        '            _lumo_fb_ir_debug_pre_base_update(',
        '                self, "before_update", scheduler_output, sampler_output,',
        '                spec_decode_metadata, spec_decode_common_attn_metadata)',
    ])
    update_call = nl.join([
        '        self._update_states_after_model_execute(',
        '            sampler_output.sampled_token_ids, scheduler_output',
        '        )',
    ])
    if debug_call not in text:
        if update_call not in text:
            raise RuntimeError('F_b pre-update trace upgrade anchor not found')
        text = text.replace(update_call, debug_call + nl + update_call, 1)
        gm.write_text(text)
        import py_compile
        py_compile.compile(str(gm), doraise=True)
        print('[TRACK-B-PRELAUNCH] upgraded F_b internal-row pre-update trace patch')
    else:
        print('[TRACK-B-PRELAUNCH] F_b internal-row pre-draft cleanup already present')
else:
    old = nl.join([
        '        self._update_states_after_model_execute(',
        '            sampler_output.sampled_token_ids, scheduler_output',
        '        )',
        '        self.p2b_debug_exporter.export_state_snapshots(runner=self)',
    ])
    new = nl.join([
        '        # LUMO_FB_INTERNAL_ROWS_PRE_DRAFT_CLEANUP: prune/collapse internal',
        '        # F_b verifier rows before vLLM updates persistent request state.',
        '        # Otherwise the ordinary state update observes sibling rows that',
        '        # share the logical Mamba state index and can corrupt path0 despite',
        '        # the kernel-level private write columns.',
        '        if "_lumo_fb_ir_prune_after_sample" in globals():',
        '            # LUMO_FB_INTERNAL_ROWS_SPEC_META_PRUNE: once internal rows',
        '            # are removed from the persistent batch, prune spec metadata',
        '            # too so the next EAGLE proposal keeps the parent-only F_b',
        '            # width-6 path instead of falling back on batch_size=2.',
        '            sampler_output, spec_decode_metadata, spec_decode_common_attn_metadata = _lumo_fb_ir_prune_after_sample(',
        '                self, scheduler_output, sampler_output, spec_decode_metadata,',
        '                spec_decode_common_attn_metadata)',
        '        if "_lumo_fb_ir_debug_pre_base_update" in globals():',
        '            _lumo_fb_ir_debug_pre_base_update(',
        '                self, "before_update", scheduler_output, sampler_output,',
        '                spec_decode_metadata, spec_decode_common_attn_metadata)',
        '        self._update_states_after_model_execute(',
        '            sampler_output.sampled_token_ids, scheduler_output',
        '        )',
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
        '                        "padding_tokens": int(num_tokens_padded) - int(num_tokens_unpadded),',
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
        '                        # num_tokens_padded may include CUDA-graph padding. The',
        '                        # contamination risk is paying/projection-sampling padded',
        '                        # speculative positions, so assert the actual unpadded',
        '                        # target, logits_indices, sample_hidden, and lm_head rows.',
        '                        _fbaw_scheduled_ok = all(int(x) == int(_fbaw_expected_per_req) for x in num_scheduled_tokens_np.tolist()[:int(num_reqs)])',
        '                        if (not _fbaw_scheduled_ok',
        '                                or int(max_num_scheduled_tokens) != int(_fbaw_expected_per_req)',
        '                                or int(num_tokens_unpadded) != _fbaw_expected_total',
        '                                or int(len(logits_indices)) != _fbaw_expected_total',
        '                                or int(sample_hidden_states.shape[0]) != _fbaw_expected_total',
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
    fa_unique = os.environ.get("LUMO_FA_UNIQUE_NODES") == "1"
    dbg = "export LUMO_TREE_DRAFT_DEBUG=1\n" if (tree and tree_debug) else ""
    tree_blocks = _TREE_ATTN_BLOCK + _MROPE_TREE_BLOCK + _TREE_REJECTION_BLOCK
    fb_k = os.environ.get("LUMO_FB_K", "1")
    fb_dup = "export LUMO_FB_DUP_PATH1=1\n" if os.environ.get("LUMO_FB_DUP_PATH1") == "1" else ""
    fb_no_shared = "export LUMO_FB_DISABLE_SHARED_ROOT=1\n" if os.environ.get("LUMO_FB_DISABLE_SHARED_ROOT") == "1" else ""
    fb_internal = "export LUMO_FB_INTERNAL_ROWS=1\n" if os.environ.get("LUMO_FB_INTERNAL_ROWS") == "1" else ""
    fb_kernel_rows = "export LUMO_FB_KERNEL_ROWS=1\n" if os.environ.get("LUMO_FB_KERNEL_ROWS") == "1" else ""
    fa_unique_env = "export LUMO_FA_UNIQUE_NODES=1\n" if fa_unique else ""
    # Batch-invariant vLLM must be enabled through the host-side ModelServer
    # knob so the launch command also gets a concrete attention backend. A raw
    # inner-container VLLM_BATCH_INVARIANT export makes vLLM fail at init with
    # attention_backend=None.
    fb_batch_invariant = ""
    fb_adaptive = "export LUMO_FB_ADAPTIVE=1\n" if os.environ.get("LUMO_FB_ADAPTIVE") == "1" else ""
    fb_batched = "export LUMO_FB_BATCHED_PROPOSER=1\n" if os.environ.get("LUMO_FB_BATCHED_PROPOSER") == "1" else ""
    fb_position_tree = f"export LUMO_FB_POSITION_TREE={os.environ['LUMO_FB_POSITION_TREE']}\n" if os.environ.get("LUMO_FB_POSITION_TREE") else ""
    fb_tree_branch_depth = f"export LUMO_FB_TREE_BRANCH_DEPTH={os.environ['LUMO_FB_TREE_BRANCH_DEPTH']}\n" if os.environ.get("LUMO_FB_TREE_BRANCH_DEPTH") else ""
    fb_replay_draft = f"export LUMO_FB_REPLAY_DRAFT_FILE={os.environ['LUMO_FB_REPLAY_DRAFT_FILE']}\n" if os.environ.get("LUMO_FB_REPLAY_DRAFT_FILE") else ""
    fb_free_row1 = "export LUMO_FB_FREE_ROW1=1\n" if os.environ.get("LUMO_FB_FREE_ROW1") == "1" else ""
    fb_free_row1_shadow = "export LUMO_FB_FREE_ROW1_SHADOW=1\n" if os.environ.get("LUMO_FB_FREE_ROW1_SHADOW") == "1" else ""
    fb_free_row1_always = "export LUMO_FB_FREE_ROW1_ALWAYS=1\n" if os.environ.get("LUMO_FB_FREE_ROW1_ALWAYS") == "1" else ""
    fb_free_row1_p1 = f"export LUMO_FB_FREE_ROW1_P1_MAX={os.environ['LUMO_FB_FREE_ROW1_P1_MAX']}\n" if os.environ.get("LUMO_FB_FREE_ROW1_P1_MAX") else ""
    fb_free_row1_ratio = f"export LUMO_FB_FREE_ROW1_RATIO_MIN={os.environ['LUMO_FB_FREE_ROW1_RATIO_MIN']}\n" if os.environ.get("LUMO_FB_FREE_ROW1_RATIO_MIN") else ""
    fb_sampler_trace = "export LUMO_FB_SAMPLER_TRACE=1\n" if os.environ.get("LUMO_FB_SAMPLER_TRACE") == "1" else ""
    fb_no_kv_prefix_copy = "export LUMO_FB_NO_KV_PREFIX_COPY=1\n" if os.environ.get("LUMO_FB_NO_KV_PREFIX_COPY") == "1" else ""
    fb_p1 = f"export LUMO_FB_ADAPTIVE_P1_MAX={os.environ['LUMO_FB_ADAPTIVE_P1_MAX']}\n" if os.environ.get("LUMO_FB_ADAPTIVE_P1_MAX") else ""
    fb_ratio = f"export LUMO_FB_ADAPTIVE_RATIO_MIN={os.environ['LUMO_FB_ADAPTIVE_RATIO_MIN']}\n" if os.environ.get("LUMO_FB_ADAPTIVE_RATIO_MIN") else ""
    fb_depth = f"export LUMO_FB_DEPTH={os.environ['LUMO_FB_DEPTH']}\n" if os.environ.get("LUMO_FB_DEPTH") else ""
    fb_debug = "export LUMO_FB_DEBUG=1\n" if os.environ.get("LUMO_FB_DEBUG") == "1" else ""
    fb_superset_diag = "export LUMO_FB_SUPERSET_DIAG=1\n" if os.environ.get("LUMO_FB_SUPERSET_DIAG") == "1" else ""
    fb_debug_exports = ""
    for _name in (
        "LUMO_FB_GDN_DEBUG",
        "LUMO_FB_RIDX_STATE_SUMMARY",
        "LUMO_FB_TENSOR_DEBUG",
        "LUMO_FB_RIDX_STATE_LAYERS",
        "LUMO_FA_TREE_DELTA_TORCH",
        "LUMO_FA_TREE_DELTA_TRITON",
        "LUMO_FA_CUDAGRAPH_UNSAFE_GDN_CORE",
    ):
        if os.environ.get(_name):
            fb_debug_exports += f"export {_name}={os.environ[_name]}\n"
    fb_control = os.environ.get("LUMO_FB_CONTROL_FILE", "/logs/fb_control.json")
    fb_seed_control = """python3 - <<'LUMOFBCTRL'
import json, os, pathlib, time
p = pathlib.Path(os.environ.get("LUMO_FB_CONTROL_FILE", "/logs/fb_control.json"))
p.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "k": int(os.environ.get("LUMO_FB_K", "1")),
    "updated_at": round(time.time(), 6),
    "source": "launch_env",
}
if os.environ.get("LUMO_FB_DEPTH"):
    payload["depth"] = int(os.environ["LUMO_FB_DEPTH"])
tmp = p.with_name(p.name + ".tmp")
tmp.write_text(json.dumps(payload, sort_keys=True) + "\\n")
tmp.replace(p)
print(f"[TRACK-B-PRELAUNCH] seeded F_b control {p}: {payload}")
LUMOFBCTRL
""" if fb else ""
    fb_env = f"export LUMO_FB_PATHS=1\nexport LUMO_FB_K={fb_k}\n{fb_depth}export LUMO_FB_CONTROL_FILE={fb_control}\nexport LUMO_FB_ASSERT_WIDTH=1\nexport LUMO_FB_ASSERT_ACTUAL_WIDTH=1\n{fb_debug}{fb_superset_diag}{fb_debug_exports}{fb_sampler_trace}{fb_dup}{fb_no_shared}{fb_internal}{fb_kernel_rows}{fa_unique_env}{fb_no_kv_prefix_copy}{fb_batch_invariant}{fb_adaptive}{fb_batched}{fb_position_tree}{fb_tree_branch_depth}{fb_replay_draft}{fb_free_row1}{fb_free_row1_shadow}{fb_free_row1_always}{fb_free_row1_p1}{fb_free_row1_ratio}{fb_p1}{fb_ratio}{fb_seed_control}" if fb else (fa_unique_env + fb_debug + fb_debug_exports + fb_replay_draft)
    mtp_draft_trace = f"export LUMO_MTP_DRAFT_TRACE_FILE={os.environ['LUMO_MTP_DRAFT_TRACE_FILE']}\n" if os.environ.get("LUMO_MTP_DRAFT_TRACE_FILE") else ""
    mtp_draft_trace_block = _MTP_DRAFT_TRACE_BLOCK if (
        os.environ.get("LUMO_MTP_DRAFT_TRACE_FILE")
        or os.environ.get("LUMO_FB_REPLAY_DRAFT_FILE")
    ) else ""
    stale_fb_guard = "" if (fb or fa_unique) else _NO_STALE_FB_PATCHES_BLOCK
    return (_QWEN36_FP8_CONFIG_FIX_BLOCK + _CAUSAL_CONV_CUDAGRAPH_ASSERT_FIX_BLOCK
            + stale_fb_guard + dbg + fb_env + mtp_draft_trace + base + _SPEC_TRACE_BLOCK
            + mtp_draft_trace_block
            + (tree_blocks if tree else "") + (_FB_BLOCK if fb else "")
            + (_FB_KERNEL_ROWS_BLOCK if ((fb and os.environ.get("LUMO_FB_KERNEL_ROWS") == "1") or fa_unique) else "")
            + (_FA_UNIQUE_NODES_BLOCK if fa_unique else "")
            + (_FA_UNIQUE_BATCH4_PACK_BLOCK if fa_unique else "")
            + (_FA_UNIQUE_BATCH4_STARTUP_FIX_BLOCK if fa_unique else "")
            + (_FA_TREE_DELTA_VALID_N_BLOCK if fa_unique else "")
            + (_FA_GDN_CORE_CUDAGRAPH_UNSAFE_BLOCK if (fa_unique and os.environ.get("LUMO_FA_CUDAGRAPH_UNSAFE_GDN_CORE") == "1") else "")
            + (_FA_UNIQUE_BATCH4_DIAG_BLOCK if fa_unique else "")
            + (_FA_REPLAY_STATE_COPY_COMMIT_BLOCK if fa_unique else ""))


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


def _apply_enforce_eager(src: str, value: str | None) -> str:
    if value is None or value.lower() not in {"1", "true", "yes", "on"}:
        return src
    line = "    enforce_eager: true"
    if "    enforce_eager:" in src:
        import re as _re
        return _re.sub(r"(?m)^    enforce_eager: .*$", line, src, count=1)
    anchor = "  vllm_config:\n"
    if src.count(anchor) != 1:
        raise RuntimeError("vllm_config anchor not unique in bundle")
    return src.replace(anchor, anchor + line + "\n", 1)


def _apply_cuda_graph_capture(src: str, value: str | None) -> str:
    if value is None:
        value = os.environ.get("LUMO_CUDAGRAPH_MODE")
    if value is None:
        value = os.environ.get("LUMO_CUDA_GRAPH_CAPTURE")
    if value is None:
        return src
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "full"}:
        normalized = "on"
    elif normalized in {"0", "false", "no", "off", "none"}:
        normalized = "off"
    elif normalized in {"piecewise", "pw"}:
        normalized = "piecewise"
    else:
        raise RuntimeError(f"unsupported LUMO_CUDAGRAPH_MODE={value!r}")
    if "  kernel_selection: {}\n" in src:
        return src.replace(
            "  kernel_selection: {}\n",
            "  kernel_selection:\n"
            f"    cuda_graph_capture: {normalized}\n",
            1,
        )
    import re as _re
    if _re.search(r"(?m)^  kernel_selection:\n", src) is None:
        raise RuntimeError("kernel_selection anchor missing in bundle")
    if _re.search(r"(?m)^    cuda_graph_capture: .*$", src):
        return _re.sub(
            r"(?m)^    cuda_graph_capture: .*$",
            f"    cuda_graph_capture: {normalized}",
            src,
            count=1,
        )
    return _re.sub(
        r"(?m)^  kernel_selection:\n",
        "  kernel_selection:\n"
        f"    cuda_graph_capture: {normalized}\n",
        src,
        count=1,
    )


def _apply_cuda_graph_capture_sizes(src: str, value: str | None) -> str:
    if value is None and os.environ.get("LUMO_FA_PACKED_CUDAGRAPH_SIZES") == "1":
        value = "4,8,12,16,24,32"
    if value is None:
        value = os.environ.get("LUMO_CUDAGRAPH_CAPTURE_SIZES")
    if value is None:
        return src
    sizes = []
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if part:
            sizes.append(int(part))
    sizes = sorted(set(sizes))
    if not sizes or sizes[0] <= 0:
        raise RuntimeError(f"unsupported LUMO_CUDAGRAPH_CAPTURE_SIZES={value!r}")
    yaml_value = "[" + ", ".join(str(size) for size in sizes) + "]"
    if "  kernel_selection: {}\n" in src:
        return src.replace(
            "  kernel_selection: {}\n",
            "  kernel_selection:\n"
            f"    cuda_graph_capture_sizes: {yaml_value}\n",
            1,
        )
    import re as _re
    if _re.search(r"(?m)^  kernel_selection:\n", src) is None:
        raise RuntimeError("kernel_selection anchor missing in bundle")
    if _re.search(r"(?m)^    cuda_graph_capture_sizes: .*$", src):
        return _re.sub(
            r"(?m)^    cuda_graph_capture_sizes: .*$",
            f"    cuda_graph_capture_sizes: {yaml_value}",
            src,
            count=1,
        )
    return _re.sub(
        r"(?m)^  kernel_selection:\n",
        "  kernel_selection:\n"
        f"    cuda_graph_capture_sizes: {yaml_value}\n",
        src,
        count=1,
    )


def _d_bundle(kv_cache_dtype: str | None = None) -> str:
    base = "/tmp/lumo-track-b-bundle-qwen36/bundle.yaml"
    enforce_eager = os.environ.get("LUMO_ENFORCE_EAGER")
    cuda_graph_capture = os.environ.get("LUMO_CUDAGRAPH_MODE") or os.environ.get("LUMO_CUDA_GRAPH_CAPTURE")
    cuda_graph_capture_sizes = os.environ.get("LUMO_CUDAGRAPH_CAPTURE_SIZES")
    packed_cg_sizes = os.environ.get("LUMO_FA_PACKED_CUDAGRAPH_SIZES")
    if kv_cache_dtype is None and enforce_eager is None and cuda_graph_capture is None and cuda_graph_capture_sizes is None and packed_cg_sizes is None:
        return base
    src = Path(base).read_text()
    src = _apply_kv_cache_dtype(src, kv_cache_dtype)
    src = _apply_enforce_eager(src, enforce_eager)
    src = _apply_cuda_graph_capture(src, cuda_graph_capture)
    src = _apply_cuda_graph_capture_sizes(src, cuda_graph_capture_sizes)
    kvtag = "" if kv_cache_dtype is None else f"-kv{kv_cache_dtype}"
    eager_tag = "-eager" if enforce_eager is not None else ""
    cg_tag = "" if cuda_graph_capture is None else f"-cg{cuda_graph_capture.strip().lower()}"
    cgs_tag = "-cgpacked" if packed_cg_sizes is not None else ("" if cuda_graph_capture_sizes is None else "-cgsizes")
    out = Path(f"/tmp/lumo-track-b-bundle-qwen36{kvtag}{eager_tag}{cg_tag}{cgs_tag}"); out.mkdir(exist_ok=True)
    (out / "bundle.yaml").write_text(src)
    return str(out / "bundle.yaml")


def _mtp_bundle(n: int, tree: str | None = None, kv_cache_dtype: str | None = None) -> str:
    src = Path("/tmp/lumo-track-b-bundle-qwen36-off/bundle.yaml").read_text()
    src = _apply_kv_cache_dtype(src, kv_cache_dtype)
    src = _apply_gpu_memory_utilization(
        src, os.environ.get("LUMO_GPU_MEMORY_UTILIZATION"))
    src = _apply_enforce_eager(src, os.environ.get("LUMO_ENFORCE_EAGER"))
    cuda_graph_capture = os.environ.get("LUMO_CUDAGRAPH_MODE") or os.environ.get("LUMO_CUDA_GRAPH_CAPTURE")
    src = _apply_cuda_graph_capture(src, cuda_graph_capture)
    cuda_graph_capture_sizes = os.environ.get("LUMO_CUDAGRAPH_CAPTURE_SIZES")
    src = _apply_cuda_graph_capture_sizes(src, cuda_graph_capture_sizes)
    kvtag = "" if kv_cache_dtype is None else f"-kv{kv_cache_dtype}"
    eager_tag = "-eager" if os.environ.get("LUMO_ENFORCE_EAGER") is not None else ""
    cg_tag = "" if cuda_graph_capture is None else f"-cg{cuda_graph_capture.strip().lower()}"
    cgs_tag = "-cgpacked" if os.environ.get("LUMO_FA_PACKED_CUDAGRAPH_SIZES") is not None else ("" if cuda_graph_capture_sizes is None else "-cgsizes")
    tag = (f"mtp{n}" if tree is None else f"mtp{n}tree") + kvtag + eager_tag + cg_tag + cgs_tag
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
    if ((is_fb and os.environ.get("LUMO_FB_KERNEL_ROWS") == "1")
            or os.environ.get("LUMO_FA_UNIQUE_NODES") == "1"):
        os.environ["LUMO_BATCH_INVARIANT_VLLM"] = "1"
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
