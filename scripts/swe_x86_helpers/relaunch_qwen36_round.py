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


def _tree_path_lcp_max_reference(
    parents: list[int],
    draft_tokens: list[int],
    parent_target_tokens: list[int],
    self_target_tokens: list[int],
) -> dict[str, object]:
    """Reference implementation for the N-spine greedy tree verifier.

    This mirrors the source-edited vLLM helper below and is intentionally kept
    small so host-side tests can prove the intended semantics without importing
    vLLM. The tree may contain any number of root-to-leaf spines. Ties prefer
    the earliest leaf in flattened tree order, which preserves path0 parity.
    """
    node_count = len(parents)
    children: dict[int, list[int]] = {-1: []}
    for node, parent in enumerate(parents):
        children.setdefault(int(parent), []).append(node)
        children.setdefault(node, [])
    leaves = [node for node in range(node_count) if not children.get(node)]
    if not leaves:
        leaves = list(range(node_count))

    best_path: list[int] = []
    best_lcp = -1
    path_scores: list[dict[str, object]] = []
    for leaf in leaves:
        path: list[int] = []
        node = int(leaf)
        guard = 0
        while 0 <= node < node_count and guard <= node_count:
            path.append(node)
            node = int(parents[node])
            guard += 1
        path.reverse()
        lcp = 0
        for node in path:
            if int(draft_tokens[node]) != int(parent_target_tokens[node]):
                break
            lcp += 1
        path_scores.append({
            "leaf": int(leaf),
            "path": [int(x) for x in path],
            "lcp": int(lcp),
        })
        if lcp > best_lcp:
            best_lcp = lcp
            best_path = path

    best_lcp = max(0, best_lcp)
    path0_lcp = int(path_scores[0]["lcp"]) if path_scores else 0
    out_tokens = [int(draft_tokens[node]) for node in best_path[:best_lcp]]
    if best_path:
        if best_lcp < len(best_path):
            out_tokens.append(int(parent_target_tokens[best_path[best_lcp]]))
        elif best_lcp > 0:
            out_tokens.append(int(self_target_tokens[best_path[best_lcp - 1]]))
        else:
            out_tokens.append(int(parent_target_tokens[best_path[0]]))
    accepted_row = int(best_path[best_lcp - 1]) + 1 if best_lcp > 0 else 0
    return {
        "accepted_len": int(best_lcp),
        "accepted_row": accepted_row,
        "accepted_node_ids": best_path[:best_lcp],
        "winner_path": best_path,
        "path0_lcp": path0_lcp,
        "superset_violation": bool(best_lcp < path0_lcp),
        "path_scores": path_scores,
        "output_tokens": out_tokens,
    }


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

# LUMO_MTP_DRAFT_TRACE: optional native MTP draft-token trace.
import os as _lumo_mtp_trace_os
import json as _lumo_mtp_trace_json
import time as _lumo_mtp_trace_time

_lumo_mtp_trace_orig_propose = EagleProposer.propose
_lumo_mtp_trace_idx = 0

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

_INDEPENDENT_ROWS_BLOCK = r'''
python3 - <<'LUMOINDEPENDENTROWS'
from pathlib import Path
import py_compile

scheduler = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py')
text = scheduler.read_text()
sentinel = '# LUMO_INDEPENDENT_ROWS'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] independent rows scheduler patch already present')
else:
    patch = r"""

# LUMO_INDEPENDENT_ROWS: hidden co-resident request rows for few-spine MTP.
# Each spine is a native vLLM sequence with its own recurrent GDN state.  The
# public request remains spine 0; sibling rows are kept alive until the request
# finishes and their EngineCoreOutput rows are suppressed from the client.
import os as _lumo_ir_os
from vllm.v1.request import Request as _LumoIRRequest

_lumo_ir_orig_add_request = Scheduler.add_request
_lumo_ir_orig_finish_requests = Scheduler.finish_requests
_lumo_ir_orig_update_from_output = Scheduler.update_from_output

def _lumo_ir_enabled():
    return _lumo_ir_os.environ.get("LUMO_INDEPENDENT_ROWS") == "1"

def _lumo_ir_spines():
    try:
        value = int(_lumo_ir_os.environ.get("LUMO_IR_SPINES", "2"))
    except Exception:
        value = 2
    return max(1, min(10, value))

def _lumo_ir_is_clone(req_id):
    return "::lumo_ir_s" in str(req_id)

def _lumo_ir_primary(req_id):
    return str(req_id).split("::lumo_ir_s", 1)[0]

def _lumo_ir_clone_id(req_id, spine):
    return f"{req_id}::lumo_ir_s{int(spine)}"

def _lumo_ir_init(self):
    if not hasattr(self, "_lumo_ir_groups"):
        self._lumo_ir_groups = {}
        self._lumo_ir_parent = {}
        self._lumo_ir_spine = {}

def _lumo_ir_clone_request(request, spine):
    clone = _LumoIRRequest(
        request_id=_lumo_ir_clone_id(request.request_id, spine),
        client_index=request.client_index,
        prompt_token_ids=(request.prompt_token_ids.copy()
                          if request.prompt_token_ids is not None else None),
        prompt_embeds=request.prompt_embeds,
        mm_features=list(request.mm_features),
        sampling_params=request.sampling_params,
        pooling_params=request.pooling_params,
        arrival_time=request.arrival_time + spine * 1e-6,
        lora_request=request.lora_request,
        cache_salt=request.cache_salt,
        priority=request.priority,
        trace_headers=request.trace_headers,
        block_hasher=getattr(request, "_block_hasher", None),
        resumable=False,
        reasoning_ended=(
            request.structured_output_request.reasoning_ended
            if request.structured_output_request is not None else None
        ),
    )
    clone._lumo_ir_hidden = True
    clone._lumo_ir_primary = request.request_id
    clone._lumo_ir_spine = spine
    return clone

def _lumo_ir_add_request(self, request):
    if (not _lumo_ir_enabled()
            or _lumo_ir_is_clone(request.request_id)
            or request.pooling_params is not None
            or request.request_id in self.requests):
        return _lumo_ir_orig_add_request(self, request)
    spine_count = _lumo_ir_spines()
    if spine_count <= 1:
        return _lumo_ir_orig_add_request(self, request)
    _lumo_ir_init(self)
    primary_id = request.request_id
    group = [primary_id] + [_lumo_ir_clone_id(primary_id, s)
                            for s in range(1, spine_count)]
    self._lumo_ir_groups[primary_id] = group
    for s, rid in enumerate(group):
        self._lumo_ir_parent[rid] = primary_id
        self._lumo_ir_spine[rid] = s
    request._lumo_ir_primary = primary_id
    request._lumo_ir_spine = 0
    _lumo_ir_orig_add_request(self, request)
    for spine in range(1, spine_count):
        _lumo_ir_orig_add_request(self, _lumo_ir_clone_request(request, spine))

def _lumo_ir_finish_requests(self, request_ids, finished_status):
    if not _lumo_ir_enabled() or request_ids is None:
        return _lumo_ir_orig_finish_requests(self, request_ids, finished_status)
    _lumo_ir_init(self)
    if isinstance(request_ids, str):
        requested = [request_ids]
    else:
        requested = list(request_ids)
    expanded = []
    seen = set()
    for rid in requested:
        primary = self._lumo_ir_parent.get(rid, _lumo_ir_primary(rid))
        members = self._lumo_ir_groups.get(primary, [rid])
        for member in members:
            if member not in seen:
                expanded.append(member)
                seen.add(member)
    finished = _lumo_ir_orig_finish_requests(self, expanded, finished_status)
    return [(rid, idx) for rid, idx in finished if not _lumo_ir_is_clone(rid)]

def _lumo_ir_update_from_output(self, scheduler_output, model_runner_output):
    outputs = _lumo_ir_orig_update_from_output(
        self, scheduler_output, model_runner_output)
    if not _lumo_ir_enabled():
        return outputs
    for client_index, eco in list(outputs.items()):
        if hasattr(eco, "outputs"):
            eco.outputs = [
                out for out in eco.outputs
                if not _lumo_ir_is_clone(getattr(out, "request_id", ""))
            ]
            if getattr(eco, "finished_requests", None):
                eco.finished_requests = {
                    rid for rid in eco.finished_requests
                    if not _lumo_ir_is_clone(rid)
                }
            if (not eco.outputs
                    and getattr(eco, "scheduler_stats", None) is None
                    and getattr(eco, "utility_output", None) is None
                    and not getattr(eco, "finished_requests", None)
                    and getattr(eco, "wave_complete", None) is None
                    and getattr(eco, "start_wave", None) is None):
                del outputs[client_index]
        else:
            filtered = [
                out for out in eco
                if not _lumo_ir_is_clone(getattr(out, "request_id", ""))
            ]
            if filtered:
                outputs[client_index] = filtered
            else:
                del outputs[client_index]
    return outputs

Scheduler.add_request = _lumo_ir_add_request
Scheduler.finish_requests = _lumo_ir_finish_requests
Scheduler.update_from_output = _lumo_ir_update_from_output
"""
    scheduler.write_text(text + patch)
    py_compile.compile(str(scheduler), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied independent rows scheduler patch')

eagle = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py')
text = eagle.read_text()
sentinel = '# LUMO_INDEPENDENT_ROWS_ROOTS'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] independent rows proposer patch already present')
else:
    helper = r"""

def _lumo_ir_choose_root_tokens(self, sample_hidden_states, fallback_token_ids):
    # LUMO_INDEPENDENT_ROWS_ROOTS: spine 0 is native top-1; later co-resident
    # rows use rank-s roots and then continue through the ordinary linear MTP
    # drafter with their own native sequence state.
    import os as _lumo_ir_os
    if _lumo_ir_os.environ.get("LUMO_INDEPENDENT_ROWS") != "1":
        return fallback_token_ids
    try:
        spines = max(1, min(10, int(_lumo_ir_os.environ.get("LUMO_IR_SPINES", "2"))))
    except Exception:
        spines = 2
    req_ids = getattr(self, "_lumo_ir_req_ids", None)
    if spines <= 1 or fallback_token_ids.shape[0] <= 1 or not req_ids:
        return fallback_token_ids
    spine_values = []
    for rid in list(req_ids)[: int(fallback_token_ids.shape[0])]:
        marker = "::lumo_ir_s"
        rid = str(rid)
        if marker not in rid:
            spine_values.append(0)
        else:
            try:
                spine_values.append(int(rid.rsplit(marker, 1)[1]))
            except Exception:
                spine_values.append(0)
    logits = self.model.compute_logits(sample_hidden_states)
    width = min(spines, int(logits.shape[-1]))
    topk = torch.topk(logits, width, dim=-1).indices
    spine_idx = torch.tensor(
        spine_values,
        dtype=torch.long,
        device=fallback_token_ids.device,
    ).clamp(min=0, max=width - 1).view(-1, 1)
    return topk.gather(1, spine_idx).squeeze(-1).to(dtype=fallback_token_ids.dtype)
"""
    text = text + helper
    old = """        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids = self._greedy_sample(sample_hidden_states)
            return draft_token_ids.view(-1, self.num_speculative_tokens)
"""
    new = """        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids = self._greedy_sample(sample_hidden_states)
            draft_token_ids = _lumo_ir_choose_root_tokens(
                self, sample_hidden_states, draft_token_ids)
            return draft_token_ids.view(-1, self.num_speculative_tokens)
"""
    if old not in text:
        raise RuntimeError('EagleProposer early root anchor not found')
    text = text.replace(old, new, 1)
    old = """        draft_token_ids = self._greedy_sample(sample_hidden_states)

        if self.allowed_attn_types is not None and not isinstance(
"""
    new = """        draft_token_ids = self._greedy_sample(sample_hidden_states)
        draft_token_ids = _lumo_ir_choose_root_tokens(
            self, sample_hidden_states, draft_token_ids)

        if self.allowed_attn_types is not None and not isinstance(
"""
    if old not in text:
        raise RuntimeError('EagleProposer linear root anchor not found')
    text = text.replace(old, new, 1)
    eagle.write_text(text)
    py_compile.compile(str(eagle), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied independent rows proposer patch')

runner = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = runner.read_text()
sentinel = '# LUMO_INDEPENDENT_ROWS_REQ_IDS'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] independent rows req-id patch already present')
else:
    old = """            else:
                mm_embed_inputs = None

            draft_token_ids = self.drafter.propose(
"""
    new = """            else:
                mm_embed_inputs = None

            # LUMO_INDEPENDENT_ROWS_REQ_IDS: expose active request ids to the
            # proposer so hidden clone roots are selected by request id, not by
            # unstable tensor row parity.
            if __import__("os").environ.get("LUMO_INDEPENDENT_ROWS") == "1":
                try:
                    _lumo_ir_batch = int(common_attn_metadata.batch_size())
                    self.drafter._lumo_ir_req_ids = list(
                        self.input_batch.req_ids[:_lumo_ir_batch])
                except Exception:
                    self.drafter._lumo_ir_req_ids = None
            draft_token_ids = self.drafter.propose(
"""
    count = text.count(old)
    if count < 1:
        raise RuntimeError('gpu_model_runner drafter propose anchor not found')
    text = text.replace(old, new, 1)
    runner.write_text(text)
    py_compile.compile(str(runner), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied independent rows req-id patch')

text = runner.read_text()
sentinel = '# LUMO_INDEPENDENT_ROWS_WINNER_COMMIT'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] independent rows winner-commit patch already present')
else:
    patch = r"""

# LUMO_INDEPENDENT_ROWS_WINNER_COMMIT: commit the longest accepted co-resident
# spine and clone its recurrent state back to the persistent sibling rows.
import json as _lumo_ir_json
import os as _lumo_ir_os2
import time as _lumo_ir_time
from vllm.v1.worker import mamba_utils as _lumo_ir_mamba_utils

_lumo_ir_orig_update_states_after_model_execute = (
    GPUModelRunner._update_states_after_model_execute)

def _lumo_ir_primary_id(req_id):
    return str(req_id).split("::lumo_ir_s", 1)[0]

def _lumo_ir_spine_id(req_id):
    req_id = str(req_id)
    marker = "::lumo_ir_s"
    if marker not in req_id:
        return 0
    try:
        return int(req_id.rsplit(marker, 1)[1])
    except Exception:
        return 0

def _lumo_ir_accept_count(row):
    for i, tok in enumerate(row):
        if int(tok) < 0:
            return int(i)
    return int(len(row))

def _lumo_ir_accept_counts_cpu(output_token_ids):
    # Keep the token matrix on GPU. vLLM's native state update only needs
    # accepted-token counts; synchronizing the whole sampled-token tensor here
    # can surface CUDA graph lifetime issues before native postprocess runs.
    return output_token_ids.ge(0).sum(dim=1).detach().cpu().tolist()

def _lumo_ir_copy_one_winner_state(
    self,
    src_req_id,
    dst_req_ids,
    src_accept_count,
):
    copy_bufs = self._get_mamba_copy_bufs()
    copy_bufs.offset = 0
    src_state = self.requests.get(src_req_id)
    src_block_idx = self.mamba_state_idx.get(src_req_id)
    if src_state is None or src_block_idx is None:
        return {"copied": 0, "missing": 1}
    state_copy_funcs = self.model.get_mamba_state_copy_func()
    forward_context = self.compilation_config.static_forward_context
    accept_token_bias = max(0, int(src_accept_count) - 1)
    copied = 0
    missing = 0
    src_block_idx = int(src_block_idx)
    for dst_req_id in dst_req_ids:
        if str(dst_req_id) == str(src_req_id):
            continue
        dst_state = self.requests.get(dst_req_id)
        dst_block_idx = self.mamba_state_idx.get(dst_req_id)
        if dst_state is None or dst_block_idx is None:
            missing += 1
            continue
        dst_block_idx = int(dst_block_idx)
        for mamba_group_id in copy_bufs.mamba_group_ids:
            src_block_ids = src_state.block_ids[mamba_group_id]
            dst_block_ids = dst_state.block_ids[mamba_group_id]
            if src_block_idx >= len(src_block_ids) or dst_block_idx >= len(dst_block_ids):
                missing += 1
                continue
            dst_block_id = dst_block_ids[dst_block_idx]
            layer_names = self.kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
            for layer_name in layer_names:
                attention = forward_context[layer_name]
                kv_caches = attention.kv_cache
                for state, state_copy_func in zip(kv_caches, state_copy_funcs):
                    if copy_bufs.offset >= copy_bufs.src_ptrs.np.shape[0]:
                        _lumo_ir_mamba_utils.do_mamba_copy_block(copy_bufs)
                        copy_bufs.offset = 0
                    copy_spec = state_copy_func(
                        state, src_block_ids, src_block_idx, accept_token_bias + 1)
                    off = copy_bufs.offset
                    copy_bufs.src_ptrs.np[off] = copy_spec.start_addr
                    copy_bufs.dst_ptrs.np[off] = state[dst_block_id].data_ptr()
                    copy_bufs.sizes.np[off] = copy_spec.num_elements * state.element_size()
                    copy_bufs.offset = off + 1
                    copied += 1
    _lumo_ir_mamba_utils.do_mamba_copy_block(copy_bufs)
    return {"copied": int(copied), "missing": int(missing)}

def _lumo_ir_winner_update_states_after_model_execute(
    self,
    output_token_ids,
    scheduler_output,
):
    if (_lumo_ir_os2.environ.get("LUMO_INDEPENDENT_ROWS") != "1"
            or _lumo_ir_os2.environ.get("LUMO_IR_WINNER_COMMIT", "1") != "1"
            or not torch.is_tensor(output_token_ids)
            or output_token_ids.dim() != 2):
        return _lumo_ir_orig_update_states_after_model_execute(
            self, output_token_ids, scheduler_output)

    num_rows = int(output_token_ids.shape[0])
    req_ids = [str(x) for x in list(self.input_batch.req_ids[:num_rows])]
    accept_counts = [int(x) for x in _lumo_ir_accept_counts_cpu(output_token_ids)]
    groups = {}
    for idx, req_id in enumerate(req_ids):
        groups.setdefault(_lumo_ir_primary_id(req_id), []).append(idx)
    winner_rows = {}
    for primary, indices in groups.items():
        if len(indices) <= 1:
            continue
        best_idx = indices[0]
        best_acc = accept_counts[best_idx]
        for idx in indices[1:]:
            acc = accept_counts[idx]
            if acc > best_acc or (
                    acc == best_acc
                    and _lumo_ir_spine_id(req_ids[idx]) < _lumo_ir_spine_id(req_ids[best_idx])):
                best_idx = idx
                best_acc = acc
        winner_rows[primary] = (best_idx, best_acc, indices)

    if not winner_rows:
        return _lumo_ir_orig_update_states_after_model_execute(
            self, output_token_ids, scheduler_output)

    trace_rows = []
    for primary, (winner_idx, winner_acc, indices) in winner_rows.items():
        winner_req_id = req_ids[winner_idx]
        winner_row = output_token_ids[winner_idx].clone()
        for idx in indices:
            output_token_ids[idx].copy_(winner_row)
            try:
                self.input_batch.num_accepted_tokens_cpu[idx] = int(winner_acc)
            except Exception:
                pass
    _lumo_ir_orig_update_states_after_model_execute(
        self, output_token_ids, scheduler_output)

    for primary, (winner_idx, winner_acc, indices) in winner_rows.items():
        winner_req_id = req_ids[winner_idx]
        copy_result = _lumo_ir_copy_one_winner_state(
            self, winner_req_id,
            [req_ids[i] for i in indices if req_ids[i] != winner_req_id],
            winner_acc)
        counts = {str(_lumo_ir_spine_id(req_ids[i])): int(accept_counts[i])
                  for i in indices}
        trace_rows.append({
            "primary": primary,
            "winner_req_id": winner_req_id,
            "winner_spine": int(_lumo_ir_spine_id(winner_req_id)),
            "winner_acc": int(winner_acc),
            "spine0_acc": int(counts.get("0", 0)),
            "counts": counts,
            "members": [req_ids[i] for i in indices],
            "copy": copy_result,
        })

    try:
        fh = globals().get("_LUMO_IR_WINNER_TRACE_FH")
        if fh is None:
            fh = open(_lumo_ir_os2.environ.get(
                "LUMO_IR_WINNER_TRACE_FILE",
                "/logs/independent_winner_trace.jsonl"), "a", buffering=1)
            globals()["_LUMO_IR_WINNER_TRACE_FH"] = fh
        for row in trace_rows:
            row["ts"] = round(_lumo_ir_time.time(), 4)
            row["event"] = "independent_winner_commit"
            fh.write(_lumo_ir_json.dumps(row) + chr(10))
    except Exception:
        pass

GPUModelRunner._update_states_after_model_execute = (
    _lumo_ir_winner_update_states_after_model_execute)
"""
    runner.write_text(text + patch)
    py_compile.compile(str(runner), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied independent rows winner-commit patch')
LUMOINDEPENDENTROWS
'''

_NO_STALE_TREE_EXPERIMENT_PATCHES_BLOCK = r'''
python3 - <<'LUMONOSTALETREEEXP'
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
        'Non-Fb launch found stale retired tree-experiment source patches: '
        + ', '.join(hits[:20]))
print('[TRACK-B-PRELAUNCH] no stale retired tree-experiment source patches found')
LUMONOSTALETREEEXP
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

_MAMBA_ALIGN_DUP_STATE_FREE_FIX_BLOCK = r"""
python3 - <<'LUMOMAMBADUPFREE'
from pathlib import Path
import py_compile

p = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py')
text = p.read_text()
sentinel = '# LUMO_MAMBA_ALIGN_DUP_STATE_FREE_FIX'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] Mamba align duplicate-state free fix already present')
else:
    old = '''                if blocks[last_state_block_idx] != self._null_block:
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])
                    blocks[last_state_block_idx] = self._null_block
'''
    new = '''                if blocks[last_state_block_idx] != self._null_block:
                    # LUMO_MAMBA_ALIGN_DUP_STATE_FREE_FIX: align-mode Mamba
                    # block tables can contain the same physical recurrent
                    # state block in multiple logical slots after speculative
                    # state promotion.  Freeing last_state_block_idx blindly
                    # can put a still-referenced block on the free queue; a
                    # later allocation then pops it with ref_cnt > 0.  Null the
                    # skipped slot, but only decrement the physical block when
                    # no other slot in this request still names it.
                    block = blocks[last_state_block_idx]
                    block_id = getattr(block, "block_id", None)
                    still_referenced = any(
                        idx != last_state_block_idx
                        and other != self._null_block
                        and getattr(other, "block_id", None) == block_id
                        for idx, other in enumerate(blocks)
                    )
                    blocks[last_state_block_idx] = self._null_block
                    if not still_referenced:
                        self.block_pool.free_blocks([block])
'''
    if old not in text:
        raise RuntimeError('Mamba align duplicate-state free anchor not found')
    text = text.replace(old, new, 1)
    old = '''            if num_required_blocks == len(req_blocks):
                return []
            else:
                assert num_required_blocks > len(req_blocks), (
                    "num_required_blocks "
                    f"{num_required_blocks} < len(req_blocks) {len(req_blocks)}"
                )
'''
    new = '''            if num_required_blocks < len(req_blocks):
                # LUMO_MAMBA_ALIGN_DUP_STATE_FREE_FIX: accepted speculative
                # prefixes can make the next main-model requirement shorter
                # than the previous align-mode speculative block table.  Trim
                # the stale tail before allocating new blocks, freeing only
                # physical blocks no remaining logical slot still references.
                removed_blocks = req_blocks[num_required_blocks:]
                del req_blocks[num_required_blocks:]
                remaining_ids = {
                    getattr(block, "block_id", None)
                    for block in req_blocks
                    if block != self._null_block
                }
                to_free = []
                seen_removed = set()
                for block in removed_blocks:
                    if block == self._null_block:
                        continue
                    block_id = getattr(block, "block_id", None)
                    if block_id in remaining_ids or block_id in seen_removed:
                        continue
                    seen_removed.add(block_id)
                    if getattr(block, "ref_cnt", 0) > 0:
                        to_free.append(block)
                if to_free:
                    self.block_pool.free_blocks(to_free)
            if num_required_blocks == len(req_blocks):
                return []
            else:
                assert num_required_blocks > len(req_blocks), (
                    "num_required_blocks "
                    f"{num_required_blocks} < len(req_blocks) {len(req_blocks)}"
                )
'''
    if old not in text:
        raise RuntimeError('Mamba align overlong block-table trim anchor not found')
    text = text.replace(old, new, 1)
    p.write_text(text)
    py_compile.compile(str(p), doraise=True)
print('[TRACK-B-PRELAUNCH] applied Mamba align duplicate-state free fix')
LUMOMAMBADUPFREE
"""

_FREE_QUEUE_MEMBERSHIP_FIX_BLOCK = r"""
python3 - <<'LUMOFREEQUEUEMEMBERSHIP'
from pathlib import Path
import py_compile

p = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py')
text = p.read_text()
sentinel = '# LUMO_FREE_QUEUE_MEMBERSHIP_FIX'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] free queue membership fix already present')
else:
    old = '''        # Connect the new block after the last block.
        last_block.next_free_block = block
        block.prev_free_block = last_block

        # Connect the fake tail after the new block.
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

        self.num_free_blocks += 1
'''
    new = '''        # LUMO_FREE_QUEUE_MEMBERSHIP_FIX: append is a queue-membership
        # operation.  A block that is already linked into the free list must not
        # be appended again; doing so leaves duplicate/stale references that can
        # later be popped after the block has been reallocated.
        if block.prev_free_block is not None or block.next_free_block is not None:
            return

        # Connect the new block after the last block.
        last_block.next_free_block = block
        block.prev_free_block = last_block

        # Connect the fake tail after the new block.
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

        self.num_free_blocks += 1
'''
    if old not in text:
        raise RuntimeError('FreeKVCacheBlockQueue append membership anchor not found')
    text = text.replace(old, new, 1)
    old = '''        # Add inter-connections between consecutive blocks
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # Connect the last block of <blocks> to the fake tail
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        self.num_free_blocks += len(blocks)
'''
    new = '''        # LUMO_FREE_QUEUE_MEMBERSHIP_FIX: append each physical block at most
        # once, and never append a block that is already linked into this free
        # list.  This is the global backstop for Mamba align-mode tables whose
        # logical slots can alias the same recurrent state block.
        linked_blocks = []
        seen_block_ids = set()
        for block in blocks:
            block_id = getattr(block, "block_id", id(block))
            if block_id in seen_block_ids:
                continue
            seen_block_ids.add(block_id)
            if block.prev_free_block is not None or block.next_free_block is not None:
                continue
            linked_blocks.append(block)

        if len(linked_blocks) == 0:
            return

        # Add inter-connections between consecutive blocks
        for block in linked_blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # Connect the last block of <blocks> to the fake tail
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        self.num_free_blocks += len(linked_blocks)
'''
    if old not in text:
        raise RuntimeError('FreeKVCacheBlockQueue append_n membership anchor not found')
    text = text.replace(old, new, 1)
    p.write_text(text)
    py_compile.compile(str(p), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied free queue membership fix')
LUMOFREEQUEUEMEMBERSHIP
"""

_BLOCK_POOL_DEDUP_FREE_FIX_BLOCK = r"""
python3 - <<'LUMOBLOCKPOOLDEDUP'
from pathlib import Path
import py_compile

p = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py')
text = p.read_text()
sentinel = '# LUMO_BLOCK_POOL_DEDUP_FREE_FIX'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] BlockPool dedup free fix already present')
else:
    old = '''        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )
'''
    new = '''        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        # LUMO_BLOCK_POOL_DEDUP_FREE_FIX: Mamba align block tables can contain
        # the same physical state block in multiple logical slots.  A single
        # request free must release that physical block once; otherwise append_n
        # can enqueue duplicate references and the second copy later trips
        # get_new_blocks with ref_cnt > 0.
        unique_blocks = []
        seen_block_ids = set()
        for block in blocks_list:
            block_id = getattr(block, "block_id", id(block))
            if block_id in seen_block_ids:
                continue
            seen_block_ids.add(block_id)
            unique_blocks.append(block)
        blocks_list = unique_blocks
        filtered_blocks = []
        for block in blocks_list:
            if (not block.is_null and block.ref_cnt == 0
                    and block.prev_free_block is not None
                    and block.next_free_block is not None):
                continue
            filtered_blocks.append(block)
        blocks_list = filtered_blocks
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )
'''
    if old not in text:
        raise RuntimeError('BlockPool free_blocks dedup anchor not found')
    text = text.replace(old, new, 1)
    p.write_text(text)
    py_compile.compile(str(p), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied BlockPool dedup free fix')
LUMOBLOCKPOOLDEDUP
"""


# tree per-path GDN prefix-state foundation: let the GDN SSM recurrent update read its initial
# state from one state slot and write the evolved per-token states to another
# slot table. This is the primitive needed for no-copy K-path rows: siblings
# read the shared prefix state, but each row stores its private evolution.
_TREE_GDN_PREFIX_STATE_BLOCK = r"""
python3 - <<'LUMOTREEGDNPREFIXSTATE'
from pathlib import Path

fg = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py')
text = fg.read_text()
sentinel = '# LUMO_TREE_GDN_PREFIX_STATE_SSM'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree per-path GDN prefix-state SSM patch already present')
else:
    old = '"IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,'
    new = old + '\n        "HAS_INITIAL_STATE_INDICES": lambda args: args["initial_state_indices"] is not None,'
    if old not in text:
        raise RuntimeError('tree per-path GDN prefix-state SSM heuristic anchor not found')
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
        raise RuntimeError('tree per-path GDN prefix-state SSM kernel arg anchor not found')
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
    # LUMO_TREE_GDN_PREFIX_STATE_SSM'''
    if old not in text:
        raise RuntimeError('tree per-path GDN prefix-state SSM constexpr anchor not found')
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
                # tree per-path GDN rows: read the shared prefix state from a separate
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
        raise RuntimeError('tree per-path GDN prefix-state SSM read anchor not found')
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
        raise RuntimeError('tree per-path GDN prefix-state SSM wrapper arg anchor not found')
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
        raise RuntimeError('tree per-path GDN prefix-state SSM launch arg anchor not found')
    text = text.replace(old, new, 1)

    fg.write_text(text)
    import py_compile
    py_compile.compile(str(fg), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree per-path GDN prefix-state SSM read/write patch')

gl = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py')
text = gl.read_text()
sentinel = '# LUMO_TREE_GDN_PREFIX_STATE_GDN_LINEAR'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree per-path GDN prefix-state gdn_linear patch already present')
else:
    old = '''        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
'''
    new = '''        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        # LUMO_TREE_GDN_PREFIX_STATE_GDN_LINEAR: optional separate read slot for SSM
        # prefix-state rows.  When None, upstream read/write semantics are unchanged.
        spec_initial_state_indices_tensor = getattr(
            attn_metadata, "spec_initial_state_indices_tensor", None)
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
'''
    if old not in text:
        raise RuntimeError('tree per-path GDN prefix-state gdn_linear metadata anchor not found')
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
        raise RuntimeError('tree per-path GDN prefix-state gdn_linear fused call anchor not found')
    text = text.replace(old, new, 1)

    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree per-path GDN prefix-state gdn_linear SSM hook')

ga = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py')
text = ga.read_text()
sentinel = '# LUMO_TREE_GDN_PREFIX_STATE_GDN_ATTN'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree per-path GDN prefix-state gdn_attn patch already present')
else:
    old = '''from dataclasses import dataclass

import torch
'''
    new = '''from dataclasses import dataclass
import os as _lumo_tree_gdn_prefix_os

import torch
'''
    if old not in text:
        raise RuntimeError('tree per-path GDN prefix-state gdn_attn import anchor not found')
    text = text.replace(old, new, 1)

    old = '''    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
'''
    new = '''    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    # LUMO_TREE_GDN_PREFIX_STATE_GDN_ATTN: separate read-only prefix state slot for
    # no-copy path rows. spec_state_indices_tensor remains the private write
    # table used by the recurrent kernels.
    spec_initial_state_indices_tensor: torch.Tensor | None = None
    spec_initial_state_slot_tensor: torch.Tensor | None = None
    spec_write_state_slot_tensor: torch.Tensor | None = None
    non_spec_state_indices_tensor: torch.Tensor | None = (
'''
    if old not in text:
        raise RuntimeError('tree per-path GDN prefix-state gdn_attn dataclass anchor not found')
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
        raise RuntimeError('tree per-path GDN prefix-state gdn_attn cudagraph alloc anchor not found')
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
        raise RuntimeError('tree per-path GDN prefix-state gdn_attn nonspec init anchor not found')
    text = text.replace(old, new, 1)

    old = '''            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]
'''
    new = '''            # tree per-path GDN prefix-state convention: block-table column 0 carries the
            # shared read-only prefix state.  Columns 1..num_spec+1 are the
            # private write table used by recurrent kernels.
            spec_initial_state_indices_tensor = None
            spec_initial_state_slot_tensor = None
            spec_write_state_slot_tensor = None
            if _lumo_tree_gdn_prefix_os.environ.get("LUMO_FA_UNIQUE_NODES") == "1":
                _tree_write_end = min(int(self.num_spec + 2), int(block_table_tensor.size(1)))
                if _tree_write_end <= 1:
                    raise RuntimeError(
                        "LUMO_FA_UNIQUE_NODES requires block_table prefix + write columns, "
                        f"got width {block_table_tensor.size(1)}")
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks, 1:_tree_write_end
                ]
                spec_initial_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks, 0
                ].contiguous()
                _fb_n = int(num_spec_decodes)
                spec_initial_state_slot_tensor = None
                spec_write_state_slot_tensor = (
                    spec_state_indices_tensor[:, 0].contiguous())

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]
'''
    if old not in text:
        raise RuntimeError('tree per-path GDN prefix-state gdn_attn spec initial anchor not found')
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
        raise RuntimeError('tree per-path GDN prefix-state gdn_attn cudagraph copy anchor not found')
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
        raise RuntimeError('tree per-path GDN prefix-state gdn_attn metadata ctor anchor not found')
    text = text.replace(old, new, 1)

    ga.write_text(text)
    import py_compile
    py_compile.compile(str(ga), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree per-path GDN prefix-state gdn_attn metadata hook')

text = gl.read_text()
sentinel = '# LUMO_TREE_GDN_PREFIX_STATE_GDN_LINEAR_CONV'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree per-path GDN prefix-state gdn_linear conv patch already present')
else:
    old = '''        spec_initial_state_indices_tensor = getattr(
            attn_metadata, "spec_initial_state_indices_tensor", None)
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
'''
    new = '''        spec_initial_state_indices_tensor = getattr(
            attn_metadata, "spec_initial_state_indices_tensor", None)
        # LUMO_TREE_GDN_PREFIX_STATE_GDN_LINEAR_CONV: conv update uses the same
        # block-table convention as SSM: read from shared prefix slot, write to
        # the row-private slot.
        spec_initial_state_slot_tensor = getattr(
            attn_metadata, "spec_initial_state_slot_tensor", None)
        spec_write_state_slot_tensor = getattr(
            attn_metadata, "spec_write_state_slot_tensor", None)
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
'''
    if old not in text:
        raise RuntimeError('tree per-path GDN prefix-state gdn_linear conv metadata anchor not found')
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
        raise RuntimeError('tree per-path GDN prefix-state gdn_linear conv call anchor not found')
    text = text.replace(old, new, 1)

    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree per-path GDN prefix-state gdn_linear conv hook')

cc = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py')
text = cc.read_text()
sentinel = '# LUMO_TREE_GDN_PREFIX_STATE_CONV_READ'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree per-path GDN prefix-state causal_conv read patch already present')
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
        raise RuntimeError('tree per-path GDN prefix-state causal_conv kernel arg anchor not found')
    text = text.replace(old, new, 1)

    old = '''    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
'''
    new = '''    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    HAS_INITIAL_STATE_INDICES: tl.constexpr,
    TREE_WRITE_COLS: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
'''
    if old not in text:
        raise RuntimeError('tree per-path GDN prefix-state causal_conv constexpr anchor not found')
    text = text.replace(old, new, 1)

    old = '''    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
    ).to(tl.int64)
'''
    new = '''    # cache_idx
    # LUMO_TREE_GDN_PREFIX_STATE_CONV_READ: no-copy path rows read the shared prefix
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
        raise RuntimeError('tree per-path GDN prefix-state causal_conv read anchor not found')
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
        raise RuntimeError('tree per-path GDN prefix-state causal_conv fanout anchor not found')
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
            # LUMO_TREE_GDN_PREFIX_STATE_CONV_PREFIX_WRITE: mirror the SSM kernel's
            # per-token state table. Each private write column stores the conv
            # state after exactly idx_token + 1 verified tokens, so promoting a
            # partial internal-row winner never carries rejected suffix state.
            _lumo_tree_write_col = idx_token
            if _lumo_tree_write_col < TREE_WRITE_COLS:
                _lumo_tree_conv_state = tl.zeros((NP2_STATELEN, BLOCK_N), dtype=tl.float32)
                if KERNEL_WIDTH >= 2:
                    _lumo_tree_conv_state = tl.where(
                        idx_tokens[:, None] == 0, col0[None, :], _lumo_tree_conv_state)
                if KERNEL_WIDTH >= 3:
                    _lumo_tree_conv_state = tl.where(
                        idx_tokens[:, None] == 1, col1[None, :], _lumo_tree_conv_state)
                if KERNEL_WIDTH >= 4:
                    _lumo_tree_conv_state = tl.where(
                        idx_tokens[:, None] == 2, col2[None, :], _lumo_tree_conv_state)
                if KERNEL_WIDTH >= 5:
                    _lumo_tree_conv_state = tl.where(
                        idx_tokens[:, None] == 3, col3[None, :], _lumo_tree_conv_state)
                if KERNEL_WIDTH >= 6:
                    _lumo_tree_conv_state = tl.where(
                        idx_tokens[:, None] == 4, col4[None, :], _lumo_tree_conv_state)
                conv_states_offset = tl.load(
                    conv_state_indices_ptr
                    + idx_seq * stride_state_indices
                    + _lumo_tree_write_col
                ).to(tl.int64)
                conv_state_ptrs_target = (
                    conv_state_ptr
                    + (conv_states_offset * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)
                )[None, :] + (idx_tokens * stride_conv_state_tok)[:, None]
                _lumo_tree_mask = (
                    (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
                )
                tl.store(conv_state_ptrs_target, _lumo_tree_conv_state, _lumo_tree_mask)

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
'''
    if old not in text:
        raise RuntimeError('tree per-path GDN prefix-state causal_conv prefix-write anchor not found')
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
        raise RuntimeError('tree per-path GDN prefix-state causal_conv wrapper arg anchor not found')
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
        raise RuntimeError('tree per-path GDN prefix-state causal_conv launch arg anchor not found')
    text = text.replace(old, new, 1)

    old = '''        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
'''
    new = '''        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        HAS_INITIAL_STATE_INDICES=initial_state_indices is not None,
        TREE_WRITE_COLS=max_query_len if initial_state_indices is not None else 1,
        NP2_STATELEN=np2_statelen,
'''
    if old not in text:
        raise RuntimeError('tree per-path GDN prefix-state causal_conv launch meta anchor not found')
    text = text.replace(old, new, 1)

    cc.write_text(text)
    import py_compile
    py_compile.compile(str(cc), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree per-path GDN prefix-state causal_conv read hook')
LUMOTREEGDNPREFIXSTATE
"""

_TREE_ATTN_BLOCK = r'''
python3 - <<'LUMOTREEATTN'
from pathlib import Path
import os
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

_TREE_PER_PATH_DRAFTER_BLOCK = r'''
python3 - <<'LUMOTREEPERPATHDRAFTER'
from pathlib import Path
nl = chr(10)
p = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py')
text = p.read_text()
sentinel = '# LUMO_TREE_PER_PATH_DRAFTER'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree per-path drafter already present')
else:
    method_anchor = '    def prepare_inputs(' + nl
    helper = r"""
    # LUMO_TREE_PER_PATH_DRAFTER: correctness-first tree drafter.
    #
    # The stock propose_tree drafts all tree nodes in one recurrent scan. That
    # shares GDN recurrence across sibling branches, so the even-node/top-1
    # spine diverges from the standalone linear MTP chain. This helper reuses
    # the normal one-token drafting loop once per root spine and repacks those
    # independent chains into vLLM's level-major tree token order. It is slower
    # by construction; it is the validation path before a fused STree kernel.
    def _lumo_tree_per_path_draft(
        self,
        *,
        batch_size: int,
        root_draft_token_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        max_depth: int,
    ) -> list[torch.Tensor] | None:
        import os as _lumo_tree_ppd_os
        if not self.cu_drafts_per_level:
            return None
        root_width = int(root_draft_token_ids.shape[1])
        if root_width <= 0:
            return None
        # This naive path intentionally supports regular N-spine trees: each
        # root has one child at every deeper level, so each level has root_width
        # nodes and the flattened order is [spine0, spine1, ...] per level.
        prev_cu = 0
        for cu in self.cu_drafts_per_level:
            level_width = int(cu) - int(prev_cu)
            if level_width != root_width:
                return None
            prev_cu = int(cu)

        def _clone_cpu_tensor(value):
            return value.clone() if value is not None else None

        def _clone_common_metadata():
            return replace(
                common_attn_metadata,
                query_start_loc=common_attn_metadata.query_start_loc.clone(),
                seq_lens=common_attn_metadata.seq_lens.clone(),
                query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu.clone(),
                _seq_lens_cpu=_clone_cpu_tensor(common_attn_metadata._seq_lens_cpu),
                _num_computed_tokens_cpu=_clone_cpu_tensor(
                    common_attn_metadata._num_computed_tokens_cpu),
            )

        chains: list[list[torch.Tensor]] = []
        for spine_idx in range(root_width):
            draft_token_ids = root_draft_token_ids[:, spine_idx].contiguous()
            chain_tokens = [draft_token_ids]
            spine_hidden_states = hidden_states.contiguous()
            spine_positions = positions.clone()
            if spine_positions.dim() == 0:
                spine_positions = spine_positions.view(1)
            spine_common = _clone_common_metadata()

            cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
                self._determine_batch_execution_and_padding(batch_size)
            )

            spine_common.num_actual_tokens = batch_size
            spine_common.max_query_len = 1
            spine_common.query_start_loc = self.arange[: batch_size + 1]
            spine_common.query_start_loc_cpu = torch.from_numpy(
                self.token_arange_np[: batch_size + 1]
            ).clone()

            block_size = self.block_size
            assert block_size > 0, "block_size has not been initialized."
            per_layer_attn_metadata: dict[str, object] = {}
            for token_index in range(max_depth - 1):
                input_ids = chain_tokens[-1].int()
                positions_1d = spine_positions[0] if self.uses_mrope else spine_positions
                if positions_1d.dim() == 0:
                    positions_1d = positions_1d.view(1)
                if self.uses_mrope:
                    out_pos = self.mrope_positions[0, :batch_size]
                elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                    out_pos = self.xdrope_positions[0, :batch_size]
                else:
                    out_pos = self.positions[:batch_size]
                eagle_step_update_slot_mapping_and_metadata(
                    positions_1d=positions_1d,
                    block_table_tensor=spine_common.block_table_tensor,
                    seq_lens=spine_common.seq_lens,
                    block_size=block_size,
                    max_model_len=self.max_model_len,
                    out_clamped_positions=out_pos,
                    out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
                    input_batch_size=input_batch_size,
                )
                spine_common.slot_mapping = self._slot_mapping_buffer[:batch_size]
                if self.uses_mrope:
                    self.mrope_positions[1:, :batch_size] = self.mrope_positions[
                        0, :batch_size
                    ]
                    spine_positions = self.mrope_positions[:, :batch_size].clone()
                elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                    self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[
                        0, :batch_size
                    ]
                    spine_positions = self.xdrope_positions[0, :batch_size].clone()
                else:
                    spine_positions = self.positions[:batch_size].clone()

                spine_common.max_seq_len = min(
                    spine_common.max_seq_len + 1, self.max_model_len
                )
                if spine_common._seq_lens_cpu is not None:
                    spine_common._seq_lens_cpu += 1
                if spine_common._num_computed_tokens_cpu is not None:
                    spine_common._num_computed_tokens_cpu += 1

                for attn_group in self.draft_attn_groups:
                    attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                        common_attn_metadata=spine_common,
                        draft_index=token_index + 1,
                    )
                    for layer_name in attn_group.layer_names:
                        per_layer_attn_metadata[layer_name] = attn_metadata

                self.input_ids[:batch_size] = input_ids
                self.hidden_states[:batch_size] = spine_hidden_states
                if self.supports_mm_inputs:
                    self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)
                    model_input_ids = None
                    inputs_embeds = self.inputs_embeds[:input_batch_size]
                else:
                    model_input_ids = self.input_ids[:input_batch_size]
                    inputs_embeds = None

                model_kwargs = {
                    "input_ids": model_input_ids,
                    "positions": self._get_positions(input_batch_size),
                    "inputs_embeds": inputs_embeds,
                }
                if self.pass_hidden_states_to_model:
                    model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

                with set_forward_context(
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
                        spine_hidden_states = ret_hidden_states
                    else:
                        last_hidden_states, spine_hidden_states = ret_hidden_states

                spine_hidden_states = spine_hidden_states[:batch_size].contiguous()
                draft_token_ids = self._greedy_sample(
                    last_hidden_states[:batch_size])
                chain_tokens.append(draft_token_ids)
            chains.append(chain_tokens)

        level_major = [root_draft_token_ids]
        for level in range(1, max_depth):
            level_major.append(
                torch.stack([chains[spine][level] for spine in range(root_width)], dim=1)
            )

        try:
            import json as _lumo_tree_ppd_json, time as _lumo_tree_ppd_time
            global _LUMO_TREE_PER_PATH_DRAFTER_FH
            try:
                _LUMO_TREE_PER_PATH_DRAFTER_FH
            except NameError:
                _LUMO_TREE_PER_PATH_DRAFTER_FH = open(
                    _lumo_tree_ppd_os.environ.get(
                        "LUMO_TREE_PER_PATH_DRAFTER_LOG",
                        "/logs/tree_per_path_drafter.jsonl"),
                    "a",
                    buffering=1,
                )
            _LUMO_TREE_PER_PATH_DRAFTER_FH.write(_lumo_tree_ppd_json.dumps({
                "event": "tree_per_path_drafter",
                "ts": round(_lumo_tree_ppd_time.time(), 4),
                "batch_size": int(batch_size),
                "spines": int(root_width),
                "depth": int(max_depth),
                "draft": torch.cat(level_major, dim=1).detach().cpu().tolist(),
            }) + chr(10))
        except Exception:
            pass
        return level_major

"""
    if method_anchor not in text:
        raise RuntimeError('tree per-path drafter method anchor not found')
    text = text.replace(method_anchor, helper + method_anchor, 1)

    call_anchor = """        draft_token_ids_list = [draft_token_ids]
        draft_hidden_states = hidden_states.view(batch_size, 1, -1)
"""
    call_new = """        draft_token_ids_list = [draft_token_ids]
        _lumo_tree_ppd = self._lumo_tree_per_path_draft(
            batch_size=batch_size,
            root_draft_token_ids=draft_token_ids,
            positions=positions,
            hidden_states=hidden_states.view(batch_size, -1),
            common_attn_metadata=common_attn_metadata,
            max_depth=len(self.cu_drafts_per_level),
        )
        if _lumo_tree_ppd is not None:
            return _lumo_tree_ppd
        draft_hidden_states = hidden_states.view(batch_size, 1, -1)
"""
    if call_anchor not in text:
        raise RuntimeError('tree per-path drafter call anchor not found')
    text = text.replace(call_anchor, call_new, 1)
    text = sentinel + '\n' + text
    p.write_text(text)
    import py_compile
    py_compile.compile(str(p), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree per-path drafter')
LUMOTREEPERPATHDRAFTER
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

_TREE_REJECTION_RANDOM_FIX_BLOCK = r'''
python3 - <<'LUMOTREEREJECTRANDOMFIX'
from pathlib import Path

rs = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py')
text = rs.read_text()
sentinel = '# LUMO_TREE_REJECTION_RANDOM_RATIO_FIX'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree rejection random ratio fix already present')
else:
    old_call = """            target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
            uniform_probs = generate_uniform_probs(
                num_tokens,
                num_draft_tokens,
                sampling_metadata.generators,
                device,
            )
            lumo_tree_prob_sample_kernel[(batch_size,)](
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                tree_parent_indices,
                tree_token_ids[0],
                tree_token_ids[1],
                target_probs,
                uniform_probs,
                max_spec_len,
                vocab_size,
            )
"""
    new_call = """            target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
            uniform_probs = generate_uniform_probs(
                num_tokens,
                num_draft_tokens,
                sampling_metadata.generators,
                device,
            )
            recovered_token_ids = sample_recovered_tokens(
                max_spec_len,
                num_draft_tokens,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                sampling_metadata,
                device,
            )
            lumo_tree_prob_sample_kernel[(batch_size,)](
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                tree_parent_indices,
                tree_token_ids[0],
                tree_token_ids[1],
                draft_probs,
                target_probs,
                recovered_token_ids,
                uniform_probs,
                max_spec_len,
                vocab_size,
                NO_DRAFT_PROBS=draft_probs is None,
            )
"""
    if old_call not in text:
        raise RuntimeError('tree rejection random ratio call anchor not found')
    text = text.replace(old_call, new_call, 1)

    start = text.find('def lumo_tree_prob_sample_kernel(')
    end_marker = '\n\n# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.\n@triton.jit(do_not_specialize=["max_spec_len"])\ndef rejection_greedy_sample_kernel('
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError('tree rejection probability kernel anchor not found')
    new_kernel = """def lumo_tree_prob_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    tree_parent_indices_ptr,  # [num_tokens], parent node or -1 for root
    parent_token_ids_ptr,  # [num_tokens], target sample at each node parent
    self_token_ids_ptr,  # [num_tokens], target sample at each node
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    current_parent = -1
    out_pos = 0
    done = False
    for _step in range(max_spec_len + 1):
        if not done:
            first_child = -1
            accepted_child = -1
            accepted_token_id = -1
            recovered_token_id = -1
            for pos in range(num_draft_tokens):
                parent = tl.load(tree_parent_indices_ptr + start_idx + pos)
                if parent == current_parent:
                    if first_child == -1:
                        first_child = pos
                        recovered_token_id = tl.load(
                            recovered_token_ids_ptr + start_idx + pos
                        )
                    draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                    if NO_DRAFT_PROBS:
                        draft_prob = 1.0
                    else:
                        draft_prob = tl.load(
                            draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                        )
                    target_prob = tl.load(
                        target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                    )
                    uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                    if (
                        (accepted_child == -1)
                        and (draft_prob > 0.0)
                        and (target_prob / draft_prob >= uniform_prob)
                    ):
                        accepted_child = pos
                        accepted_token_id = draft_token_id

            if first_child == -1:
                if current_parent >= 0:
                    token_id = tl.load(self_token_ids_ptr + start_idx + current_parent)
                    tl.store(
                        output_token_ids_ptr + req_idx * (max_spec_len + 1) + out_pos,
                        token_id,
                    )
                done = True
            elif accepted_child >= 0:
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + out_pos,
                    accepted_token_id,
                )
                out_pos += 1
                current_parent = accepted_child
            else:
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + out_pos,
                    recovered_token_id,
                )
                done = True
"""
    text = text[:start] + new_kernel + text[end:]
    text = sentinel + '\n' + text
    rs.write_text(text)
    import py_compile
    py_compile.compile(str(rs), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree rejection random ratio fix')
LUMOTREEREJECTRANDOMFIX
'''

_TREE_ACCEPTED_ROW_KERNEL_BLOCK = r'''
python3 - <<'LUMOTREEACCEPTEDROWKERNEL'
from pathlib import Path

rs = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py')
text = rs.read_text()
sentinel = '# LUMO_TREE_ACCEPTED_ROW_KERNEL'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree accepted-row kernel patch already present')
else:
    branch_old = """    if tree_parent_indices is not None and tree_token_ids is not None:
        assert tree_parent_indices.is_contiguous()
        assert tree_token_ids.is_contiguous()
        if sampling_metadata.all_greedy:
            lumo_tree_sample_kernel[(batch_size,)](
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                tree_parent_indices,
                tree_token_ids[0],
                tree_token_ids[1],
                max_spec_len,
            )
        else:
            target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
            uniform_probs = generate_uniform_probs(
                num_tokens,
                num_draft_tokens,
                sampling_metadata.generators,
                device,
            )
            recovered_token_ids = sample_recovered_tokens(
                max_spec_len,
                num_draft_tokens,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                sampling_metadata,
                device,
            )
            lumo_tree_prob_sample_kernel[(batch_size,)](
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                tree_parent_indices,
                tree_token_ids[0],
                tree_token_ids[1],
                draft_probs,
                target_probs,
                recovered_token_ids,
                uniform_probs,
                max_spec_len,
                vocab_size,
                NO_DRAFT_PROBS=draft_probs is None,
            )
        return output_token_ids
"""
    branch_new = """    if tree_parent_indices is not None and tree_token_ids is not None:
        assert tree_parent_indices.is_contiguous()
        assert tree_token_ids.is_contiguous()
        accepted_tree_rows = torch.zeros(
            (batch_size,),
            dtype=torch.int32,
            device=device,
        )
        if sampling_metadata.all_greedy:
            lumo_tree_sample_kernel[(batch_size,)](
                output_token_ids,
                accepted_tree_rows,
                cu_num_draft_tokens,
                draft_token_ids,
                tree_parent_indices,
                tree_token_ids[0],
                tree_token_ids[1],
                max_spec_len,
            )
        else:
            target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
            uniform_probs = generate_uniform_probs(
                num_tokens,
                num_draft_tokens,
                sampling_metadata.generators,
                device,
            )
            recovered_token_ids = sample_recovered_tokens(
                max_spec_len,
                num_draft_tokens,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                sampling_metadata,
                device,
            )
            lumo_tree_prob_sample_kernel[(batch_size,)](
                output_token_ids,
                accepted_tree_rows,
                cu_num_draft_tokens,
                draft_token_ids,
                tree_parent_indices,
                tree_token_ids[0],
                tree_token_ids[1],
                draft_probs,
                target_probs,
                recovered_token_ids,
                uniform_probs,
                max_spec_len,
                vocab_size,
                NO_DRAFT_PROBS=draft_probs is None,
            )
        try:
            _rows = [int(x) for x in accepted_tree_rows.detach().cpu().tolist()]
            globals()["_LUMO_TREE_LAST_ACCEPTED_ROWS_KERNEL"] = _rows
            from vllm.model_executor.layers.mamba import gdn_linear_attn as _lumo_tree_commit_gdn
            _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_ROWS = _rows
        except Exception:
            pass
        try:
            import json as _tapj, os as _tapo, time as _tapt
            global _LUMO_TREE_ACCEPT_PATH_FH
            try:
                _LUMO_TREE_ACCEPT_PATH_FH
            except NameError:
                _LUMO_TREE_ACCEPT_PATH_FH = open(
                    _tapo.environ.get("LUMO_TREE_ACCEPT_PATH_LOG", "/logs/tree_accept_path.jsonl"),
                    "a",
                    buffering=1,
                )
            _parents_cpu = tree_parent_indices.detach().cpu().tolist()
            _start = 0
            _now = round(_tapt.time(), 4)
            for _req_i, _n in enumerate(num_draft_tokens):
                _n = int(_n)
                _parents = [int(x) for x in _parents_cpu[_start:_start + _n]]
                _final_row = int(_rows[_req_i]) if _req_i < len(_rows) else 0
                _accepted_node_ids = []
                if _final_row > 0:
                    _node = _final_row - 1
                    _guard = 0
                    while 0 <= _node < _n and _guard <= _n:
                        _accepted_node_ids.append(int(_node))
                        _node = int(_parents[_node])
                        _guard += 1
                    _accepted_node_ids.reverse()
                _child_ranks = []
                for _node in _accepted_node_ids:
                    _parent = int(_parents[_node])
                    _rank = 0
                    for _pos in range(int(_node)):
                        if int(_parents[_pos]) == _parent:
                            _rank += 1
                    _child_ranks.append(int(_rank))
                _seen_alt = False
                _alt_tokens = 0
                for _rank in _child_ranks:
                    if int(_rank) != 0:
                        _seen_alt = True
                    if _seen_alt:
                        _alt_tokens += 1
                _LUMO_TREE_ACCEPT_PATH_FH.write(_tapj.dumps({
                    "ts": _now,
                    "req_index": int(_req_i),
                    "node_count": int(_n),
                    "accepted_node_ids": _accepted_node_ids,
                    "accepted_child_ranks": _child_ranks,
                    "accepted_len": int(len(_accepted_node_ids)),
                    "accepted_final_row": int(_final_row),
                    "has_alt_branch": bool(any(int(_rank) != 0 for _rank in _child_ranks)),
                    "accepted_alt_tokens": int(_alt_tokens),
                }) + chr(10))
                _start += _n
        except Exception:
            pass
        return output_token_ids
"""
    if branch_old not in text:
        raise RuntimeError('tree accepted-row kernel branch anchor not found')
    text = text.replace(branch_old, branch_new, 1)

    sig_old = """def lumo_tree_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
"""
    sig_new = """def lumo_tree_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    accepted_tree_rows_ptr,  # [batch_size], local row after accepted draft path
    cu_num_draft_tokens_ptr,  # [batch_size]
"""
    if sig_old not in text:
        raise RuntimeError('tree accepted-row greedy kernel signature anchor not found')
    text = text.replace(sig_old, sig_new, 1)

    greedy_old = """                if matched_child >= 0:
                    current_parent = matched_child
                else:
                    done = True
"""
    greedy_new = """                if matched_child >= 0:
                    current_parent = matched_child
                    tl.store(accepted_tree_rows_ptr + req_idx, current_parent + 1)
                else:
                    done = True
"""
    if greedy_old not in text:
        raise RuntimeError('tree accepted-row greedy store anchor not found')
    text = text.replace(greedy_old, greedy_new, 1)

    prob_sig_old = """def lumo_tree_prob_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
"""
    prob_sig_new = """def lumo_tree_prob_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    accepted_tree_rows_ptr,  # [batch_size], local row after accepted draft path
    cu_num_draft_tokens_ptr,  # [batch_size]
"""
    if prob_sig_old not in text:
        raise RuntimeError('tree accepted-row prob kernel signature anchor not found')
    text = text.replace(prob_sig_old, prob_sig_new, 1)

    prob_old = """                out_pos += 1
                current_parent = accepted_child
            else:
"""
    prob_new = """                out_pos += 1
                current_parent = accepted_child
                tl.store(accepted_tree_rows_ptr + req_idx, current_parent + 1)
            else:
"""
    if prob_old not in text:
        raise RuntimeError('tree accepted-row prob store anchor not found')
    text = text.replace(prob_old, prob_new, 1)

    text = sentinel + '\n' + text
    rs.write_text(text)
    import py_compile
    py_compile.compile(str(rs), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree accepted-row kernel patch')
LUMOTREEACCEPTEDROWKERNEL
'''

_CUDAGRAPH_RUNTIME_TELEMETRY_BLOCK = r'''
python3 - <<'LUMOCUDAGRAPHRUNTIMETELEMETRY'
from pathlib import Path

gm = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = gm.read_text()
sentinel = '# LUMO_CUDAGRAPH_RUNTIME_TELEMETRY'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] cudagraph runtime telemetry patch already present')
else:
    old = """        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )
"""
    new = """        cudagraph_stats = None
        if (
            self.vllm_config.observability_config.cudagraph_metrics
            or __import__("os").environ.get("LUMO_CUDAGRAPH_RUNTIME_TELEMETRY") == "1"
        ):
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )
            if __import__("os").environ.get("LUMO_CUDAGRAPH_RUNTIME_TELEMETRY") == "1":
                try:
                    import json as _lumo_cg_json, time as _lumo_cg_time
                    global _LUMO_CUDAGRAPH_RUNTIME_FH
                    try:
                        _LUMO_CUDAGRAPH_RUNTIME_FH
                    except NameError:
                        _LUMO_CUDAGRAPH_RUNTIME_FH = open("/logs/cudagraph_runtime_debug.jsonl", "a", buffering=1)
                    _LUMO_CUDAGRAPH_RUNTIME_FH.write(_lumo_cg_json.dumps({
                        "ts": round(_lumo_cg_time.time(), 4),
                        "event": "cudagraph_runtime",
                        "num_unpadded_tokens": int(num_tokens),
                        "num_padded_tokens": int(batch_descriptor.num_tokens),
                        "num_paddings": int(batch_descriptor.num_tokens - num_tokens),
                        "runtime_mode": str(cudagraph_mode),
                    }) + chr(10))
                except Exception:
                    pass
"""
    if old not in text:
        raise RuntimeError('cudagraph runtime telemetry anchor not found')
    text = text.replace(old, new, 1)
    text = sentinel + '\n' + text
    gm.write_text(text)
    import py_compile
    py_compile.compile(str(gm), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied cudagraph runtime telemetry patch')
LUMOCUDAGRAPHRUNTIMETELEMETRY
'''

_TREE_ACCEPTED_ROW_COMMIT_BLOCK = r'''
python3 - <<'LUMOTREEACCEPTEDROWCOMMIT'
from pathlib import Path

rs = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py')
text = rs.read_text()
sentinel = '# LUMO_TREE_ACCEPTED_ROW_COMMIT'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree accepted-row commit sampler patch already present')
else:
    anchor = """        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            tree_parent_indices=lumo_tree_parent_indices,
            tree_token_ids=lumo_tree_token_ids,
        )
"""
    inject = anchor + """
        # LUMO_TREE_ACCEPTED_ROW_COMMIT: for branched trees, a generated
        # sequence position is not the same as the flattened verifier row.
        # Record the actual accepted tree row per request so GDN state-copy
        # commits copy the accepted branch leaf, not the linear top-1 row.
        if lumo_tree_parent_indices is not None:
            try:
                from vllm.model_executor.layers.mamba import gdn_linear_attn as _lumo_tree_commit_gdn
                _rows = list(globals().get("_LUMO_TREE_LAST_ACCEPTED_ROWS_KERNEL", []) or [])
                if len(_rows) != len(metadata.num_draft_tokens):
                    _parents_cpu = lumo_tree_parent_indices.detach().cpu().tolist()
                    _draft_cpu = metadata.draft_token_ids.detach().cpu().tolist()
                    _out_cpu = output_token_ids.detach().cpu().tolist()
                    _vocab_size = int(logits.shape[-1])
                    _rows = []
                    _start = 0
                    for _req_i, _n in enumerate(metadata.num_draft_tokens):
                        _n = int(_n)
                        _parents = [int(x) for x in _parents_cpu[_start:_start + _n]]
                        _drafts = [int(x) for x in _draft_cpu[_start:_start + _n]]
                        _tokens = [
                            int(x) for x in _out_cpu[_req_i]
                            if 0 <= int(x) < _vocab_size
                        ]
                        _cur_parent = -1
                        _final_row = 0
                        for _tok in _tokens:
                            _matched = -1
                            for _pos in range(_n):
                                if _parents[_pos] == _cur_parent and _drafts[_pos] == _tok:
                                    _matched = _pos
                                    break
                            if _matched < 0:
                                break
                            _cur_parent = _matched
                            _final_row = _matched + 1
                        _rows.append(int(_final_row))
                        _start += _n
                _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_ROWS = _rows
            except Exception as _exc:
                try:
                    import json as _arj, time as _art
                    global _LUMO_TREE_ACCEPTED_ROW_ERR_FH
                    try:
                        _LUMO_TREE_ACCEPTED_ROW_ERR_FH
                    except NameError:
                        _LUMO_TREE_ACCEPTED_ROW_ERR_FH = open("/logs/tree_accepted_row_commit_error.jsonl", "a", buffering=1)
                    _LUMO_TREE_ACCEPTED_ROW_ERR_FH.write(_arj.dumps({
                        "ts": round(_art.time(), 4),
                        "error": repr(_exc),
                    }) + chr(10))
                except Exception:
                    pass
"""
    if anchor not in text:
        raise RuntimeError('tree accepted-row commit sampler anchor not found')
    text = text.replace(anchor, inject, 1)
    text = sentinel + '\n' + text
    rs.write_text(text)
    import py_compile
    py_compile.compile(str(rs), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree accepted-row commit sampler patch')

raise SystemExit(0)

gl = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py')
text = gl.read_text()
sentinel = '# LUMO_FA_BRANCH_ACCEPTED_ROW_STATE_COPY'
record_old = """                    try:
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
"""
record_new = """                    try:
                        _total_tree_rows = int(attn_metadata.num_spec_decode_tokens)
                        _req_hint = int(globals().get("_LUMO_FA_LAST_TREE_REQ_COUNT", 0) or 0)
                        if _req_hint > 0 and _total_tree_rows % _req_hint == 0:
                            _record_req_count = int(_req_hint)
                            _record_group_size = int(_total_tree_rows // _req_hint)
                        else:
                            _depth_rows_for_record = getattr(attn_metadata, "fa_tree_depth_rows", None)
                            if _depth_rows_for_record is not None:
                                _record_group_size = int(len(_depth_rows_for_record))
                            if _record_group_size <= 0:
                                _record_group_size = int(_lumo_fa_os.environ.get("LUMO_FA_TREE_GROUP_SIZE", "4"))
                            if _record_group_size > 0 and _total_tree_rows % _record_group_size == 0:
                                _record_req_count = int(_total_tree_rows // _record_group_size)
                    except Exception:
                        _record_group_size = 0
                        _record_req_count = 0
"""
if record_old in text:
    text = text.replace(record_old, record_new, 1)

start = text.find('def _lumo_fa_activation_replay_commit(\n')
if start < 0:
    start = text.find('def _lumo_fa_activation_replay_commit(accepted_token_count: int) -> None:\n')
if start < 0:
    start = text.find('def _lumo_fa_activation_replay_commit(accepted_token_count) -> None:\n')
end = text.find('\n@CustomOp.register("chunk_gated_delta_rule")\n', start)
if start < 0 or end < 0:
    raise RuntimeError('F_a activation replay commit function anchor not found for branch accepted-row patch')
if sentinel not in text[start:end]:
    new = r"""def _lumo_fa_activation_replay_commit(
    accepted_token_count,
    expected_total_tokens=None,
    expected_req_count=None,
) -> None:
    # LUMO_FA_BRANCH_ACCEPTED_ROW_STATE_COPY: copy the actual accepted tree
    # row per request. Flattened tree row order is not a linear path when the
    # verifier has siblings, so base + accepted_count - 1 corrupts branch state.
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
    accepted_tree_rows = list(globals().get("_LUMO_FA_LAST_ACCEPTED_TREE_ROWS", []) or [])
    try:
        _fh = globals().get("_LUMO_FA_REPLAY_COMMIT_DETAIL_FH")
        if _fh is None:
            _fh = open("/logs/fa_activation_replay_commit_detail.jsonl", "a", buffering=1)
            globals()["_LUMO_FA_REPLAY_COMMIT_DETAIL_FH"] = _fh
        _fh.write(_lumo_fa_json.dumps({
            "ts": round(_lumo_fa_time.time(), 4),
            "event": "fa_activation_replay_commit_detail",
            "commit_mode": "state_copy_branch_row",
            "accepted_counts": accepted_counts,
            "accepted_tree_rows": [int(x) for x in accepted_tree_rows],
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
    used_tree_rows = []
    for rec in list(replay_layers):
        total_tokens = int(rec["num_tokens"])
        if total_tokens <= 0:
            continue
        record_group_size = int(rec.get("tree_group_size") or 0)
        record_req_count = int(rec.get("tree_req_count") or 0)
        group_size = 0
        req_count = 0
        try:
            expected_req = int(expected_req_count or 0)
        except Exception:
            expected_req = 0
        if expected_req > 0 and total_tokens % expected_req == 0:
            req_count = expected_req
            group_size = total_tokens // expected_req
        elif record_req_count > 0 and total_tokens % record_req_count == 0:
            req_count = record_req_count
            group_size = total_tokens // record_req_count
        elif record_group_size > 0 and total_tokens % record_group_size == 0:
            group_size = record_group_size
            req_count = max(1, total_tokens // group_size)
        elif len(accepted_counts) > 1 and total_tokens % len(accepted_counts) == 0:
            req_count = len(accepted_counts)
            group_size = max(1, total_tokens // req_count)
        else:
            group_size = total_tokens
            req_count = 1
        req_count = min(max(1, int(req_count)), len(accepted_counts))
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
                    "commit_mode": "state_copy_branch_row",
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
            fallback_local = max(0, min(tokens - 1, group_size - 1))
            local_row = fallback_local
            if req_i < len(accepted_tree_rows):
                try:
                    candidate = int(accepted_tree_rows[req_i])
                    if 0 <= candidate < group_size:
                        local_row = candidate
                except Exception:
                    local_row = fallback_local
            final_row = base + local_row
            prefix_idx = int(initial_flat[base].item())
            final_idx = int(state_flat[final_row].item())
            if final_idx != prefix_idx:
                rec["conv_state"][prefix_idx].copy_(rec["conv_state"][final_idx], non_blocking=True)
                rec["ssm_state"][prefix_idx].copy_(rec["ssm_state"][final_idx], non_blocking=True)
            copied += 1
            used_tree_rows.append(int(local_row))
    try:
        _fh = globals().get("_LUMO_FA_REPLAY_COMMIT_DETAIL_FH")
        if _fh is not None:
            _fh.write(_lumo_fa_json.dumps({
                "ts": round(_lumo_fa_time.time(), 4),
                "event": "fa_activation_replay_commit_summary",
                "commit_mode": "state_copy_branch_row",
                "accepted_counts": accepted_counts,
                "accepted_tree_rows": [int(x) for x in accepted_tree_rows],
                "used_tree_rows": used_tree_rows[:16],
                "copied_requests": int(copied),
                "missing_state_indices": int(missing_state_indices),
                "commit_enqueue_us": int((_lumo_fa_time.perf_counter() - _commit_t0) * 1000000),
            }) + chr(10))
    except Exception:
        pass
"""
    text = text[:start] + new + text[end:]
    text = sentinel + '\n' + text

gl.write_text(text)
import py_compile
py_compile.compile(str(gl), doraise=True)
print('[TRACK-B-PRELAUNCH] applied branch accepted-row state-copy patch')
LUMOTREEACCEPTEDROWCOMMIT
'''

_TREE_PATH_LCP_MAX_BLOCK = r'''
python3 - <<'LUMOTREEPATHLCPMAX'
from pathlib import Path

rs = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py')
text = rs.read_text()
sentinel = '# LUMO_TREE_PATH_LCP_MAX'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] tree path-LCP-max verifier already present')
else:
    helper_anchor = '\n\ndef rejection_sample('
    helper = r"""

# LUMO_TREE_PATH_LCP_MAX: greedy N-spine verifier.
#
# Verify every root-to-leaf path against target argmax tokens and choose the
# longest accepted prefix. This is the multi-candidate greedy rule:
# accepted_len = max_path LCP(path, target_argmax). It is a strict superset of
# path0 because path0 is one of the leaves. The only recurrent state recorded
# for commit is the final accepted node on the winning path.
def _lumo_tree_path_lcp_max_greedy_sample(
    output_token_ids: torch.Tensor,
    accepted_tree_rows: torch.Tensor,
    num_draft_tokens,
    draft_token_ids: torch.Tensor,
    tree_parent_indices: torch.Tensor,
    parent_token_ids: torch.Tensor,
    self_token_ids: torch.Tensor,
    max_spec_len: int,
) -> torch.Tensor:
    parents_cpu = [int(x) for x in tree_parent_indices.detach().cpu().tolist()]
    drafts_cpu = [int(x) for x in draft_token_ids.detach().cpu().tolist()]
    parent_targets_cpu = [int(x) for x in parent_token_ids.detach().cpu().tolist()]
    self_targets_cpu = [int(x) for x in self_token_ids.detach().cpu().tolist()]
    if hasattr(num_draft_tokens, 'detach'):
        counts = [int(x) for x in num_draft_tokens.detach().cpu().tolist()]
    else:
        counts = [int(x) for x in num_draft_tokens]

    out_rows = []
    accepted_rows = []
    path_log_rows = []
    start = 0
    for req_i, node_count in enumerate(counts):
        node_count = int(node_count)
        parents = parents_cpu[start:start + node_count]
        drafts = drafts_cpu[start:start + node_count]
        parent_targets = parent_targets_cpu[start:start + node_count]
        self_targets = self_targets_cpu[start:start + node_count]

        children = {-1: []}
        for node, parent in enumerate(parents):
            parent = int(parent)
            children.setdefault(parent, []).append(node)
            children.setdefault(node, [])
        leaves = [node for node in range(node_count) if not children.get(node)]
        if not leaves:
            leaves = list(range(node_count))

        best_path = []
        best_lcp = -1
        best_leaf = -1
        path_scores = []
        for leaf in leaves:
            path = []
            node = int(leaf)
            guard = 0
            while 0 <= node < node_count and guard <= node_count:
                path.append(node)
                node = int(parents[node])
                guard += 1
            path.reverse()
            lcp = 0
            for node in path:
                if int(drafts[node]) != int(parent_targets[node]):
                    break
                lcp += 1
            path_scores.append({
                'leaf': int(leaf),
                'path': [int(x) for x in path],
                'lcp': int(lcp),
            })
            # Stable tie-break: earliest flattened leaf wins. With vLLM's sorted
            # tree choices this preserves the native top-1/path0 chain.
            if lcp > best_lcp:
                best_lcp = int(lcp)
                best_path = path
                best_leaf = int(leaf)

        best_lcp = max(0, int(best_lcp))
        path0_lcp = int(path_scores[0]['lcp']) if path_scores else 0
        row = []
        for node in best_path[:best_lcp]:
            row.append(int(drafts[node]))
        if best_path:
            if best_lcp < len(best_path):
                row.append(int(parent_targets[best_path[best_lcp]]))
            elif best_lcp > 0:
                row.append(int(self_targets[best_path[best_lcp - 1]]))
            else:
                row.append(int(parent_targets[best_path[0]]))
        row = row[:int(max_spec_len) + 1]
        out_rows.append(row)
        accepted_row = int(best_path[best_lcp - 1]) + 1 if best_lcp > 0 else 0
        accepted_rows.append(accepted_row)
        path_log_rows.append({
            'req_index': int(req_i),
            'node_count': int(node_count),
            'accepted_len': int(best_lcp),
            'accepted_final_row': int(accepted_row),
            'accepted_node_ids': [int(x) for x in best_path[:best_lcp]],
            'path0_lcp': int(path0_lcp),
            'superset_violation': bool(int(best_lcp) < int(path0_lcp)),
            'superset_delta': int(best_lcp) - int(path0_lcp),
            'winner_leaf': int(best_leaf),
            'winner_path': [int(x) for x in best_path],
            'path_scores': path_scores,
        })
        start += node_count

    output_token_ids.fill_(-1)
    for req_i, row in enumerate(out_rows):
        for pos, token_id in enumerate(row):
            output_token_ids[req_i, pos] = int(token_id)
    accepted_tree_rows.copy_(
        torch.tensor(accepted_rows, dtype=accepted_tree_rows.dtype,
                     device=accepted_tree_rows.device)
    )
    globals()['_LUMO_TREE_LAST_ACCEPTED_ROWS_KERNEL'] = [int(x) for x in accepted_rows]
    try:
        from vllm.model_executor.layers.mamba import gdn_linear_attn as _lumo_tree_commit_gdn
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_ROWS = [
            int(x) for x in accepted_rows
        ]
    except Exception:
        pass
    try:
        import json as _lcpj, os as _lcpo, time as _lcpt
        global _LUMO_TREE_PATH_LCP_FH
        try:
            _LUMO_TREE_PATH_LCP_FH
        except NameError:
            _LUMO_TREE_PATH_LCP_FH = open(
                _lcpo.environ.get('LUMO_TREE_PATH_LCP_LOG',
                                  '/logs/tree_path_lcp_max.jsonl'),
                'a',
                buffering=1,
            )
        _now = round(_lcpt.time(), 4)
        for row in path_log_rows:
            row = dict(row)
            row['ts'] = _now
            row['event'] = 'tree_path_lcp_max'
            _LUMO_TREE_PATH_LCP_FH.write(_lcpj.dumps(row) + chr(10))
    except Exception:
        pass
    return output_token_ids
"""
    if helper_anchor not in text:
        raise RuntimeError('tree path-LCP helper anchor not found')
    text = text.replace(helper_anchor, helper + helper_anchor, 1)

    old = """        if sampling_metadata.all_greedy:
            lumo_tree_sample_kernel[(batch_size,)](
                output_token_ids,
                accepted_tree_rows,
                cu_num_draft_tokens,
                draft_token_ids,
                tree_parent_indices,
                tree_token_ids[0],
                tree_token_ids[1],
                max_spec_len,
            )
        else:
"""
    new = """        if sampling_metadata.all_greedy:
            output_token_ids = _lumo_tree_path_lcp_max_greedy_sample(
                output_token_ids,
                accepted_tree_rows,
                num_draft_tokens,
                draft_token_ids,
                tree_parent_indices,
                tree_token_ids[0],
                tree_token_ids[1],
                max_spec_len,
            )
        else:
"""
    if old not in text:
        raise RuntimeError('tree path-LCP greedy branch anchor not found')
    text = text.replace(old, new, 1)
    text = sentinel + '\n' + text
    rs.write_text(text)
    import py_compile
    py_compile.compile(str(rs), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied tree path-LCP-max greedy verifier')
LUMOTREEPATHLCPMAX
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
    if 'import os as _lumo_tree_gdn_prefix_os' not in text:
        old = 'from dataclasses import dataclass\n\nimport torch\n'
        new = 'from dataclasses import dataclass\nimport os as _lumo_tree_gdn_prefix_os\n\nimport torch\n'
        if old not in text:
            raise RuntimeError('F_a unique-node gdn_attn import anchor not found')
        text = text.replace(old, new, 1)
    if 'import ast as _lumo_fa_ast' not in text:
        text = text.replace(
            'import os as _lumo_tree_gdn_prefix_os\n',
            'import os as _lumo_tree_gdn_prefix_os\nimport ast as _lumo_fa_ast\nimport json as _lumo_fa_json\nimport time as _lumo_fa_time\n',
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
                if block_table_tensor.size(1) < _node_count + 1:
                    raise RuntimeError(
                        "LUMO_FA_UNIQUE_NODES requires root + node state slots: "
                        f"need at least {_node_count + 1}, got "
                        f"{block_table_tensor.size(1)}")
                _row = block_table_tensor[spec_sequence_masks, :_node_count + 2][0]
                if _row.numel() >= _node_count + 2:
                    _write_slots = _row[1:_node_count + 2].contiguous()
                else:
                    _write_slots = _row[:_node_count + 1].contiguous()
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
                for _parent in _parents:
                    _depths.append(0 if int(_parent) < 0 else _depths[int(_parent)] + 1)
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
            _actual_conv_rows = int(mixed_qkv_spec.shape[0])
            _needs_row_clip = any(
                any(int(i) >= _actual_conv_rows for i in _rows)
                for _rows in _depth_rows
            )
            if _needs_row_clip or _depth_row_tensors is None or _depth_query_start_tensors is None:
                _depth_rows = tuple(
                    tuple(int(i) for i in _rows if int(i) < _actual_conv_rows)
                    for _rows in _depth_rows
                )
                _clip_cache = getattr(self, "_lumo_fa_conv_depth_clip_cache", None)
                if _clip_cache is None:
                    _clip_cache = {}
                    self._lumo_fa_conv_depth_clip_cache = _clip_cache
                _clip_key = (_actual_conv_rows, _depth_rows)
                if _clip_key not in _clip_cache:
                    _clip_cache[_clip_key] = (
                        tuple(
                            torch.tensor(_rows, dtype=torch.long, device=mixed_qkv_spec.device)
                            for _rows in _depth_rows
                        ),
                        tuple(
                            torch.arange(len(_rows) + 1, dtype=torch.int32, device=mixed_qkv_spec.device)
                            for _rows in _depth_rows
                        ),
                    )
                _depth_row_tensors, _depth_query_start_tensors = _clip_cache[_clip_key]
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
        print('[TRACK-B-PRELAUNCH] skip F_a expanded-node conv patch; tree per-path GDN prefix-state hook absent')
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
        if spec_sequence_masks is not None and fa_unique_expanded_node_mode:
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
            for _parent in _parents:
                _depths.append(0 if int(_parent) < 0 else _depths[int(_parent)] + 1)
            core_attn_out_spec = None
            last_recurrent_state = None
            for _depth in range((max(_depths) + 1) if _depths else 0):
                _rows = [i for i, d in enumerate(_depths) if d == _depth and i < int(query_spec.shape[1])]
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
        print('[TRACK-B-PRELAUNCH] skip F_a expanded-node SSM patch; tree per-path GDN prefix-state hook absent')
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

_FA_BRANCH_ACCEPTED_ROW_STATE_COPY_BLOCK = r'''
python3 - <<'LUMOFABRANCHROWCOPY'
from pathlib import Path

gl = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py')
text = gl.read_text()
sentinel = '# LUMO_FA_BRANCH_ACCEPTED_ROW_STATE_COPY'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] branch accepted-row state-copy patch already present')
else:
    record_old = """                    try:
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
"""
    record_new = """                    try:
                        _total_tree_rows = int(attn_metadata.num_spec_decode_tokens)
                        _req_hint = int(globals().get("_LUMO_FA_LAST_TREE_REQ_COUNT", 0) or 0)
                        if _req_hint > 0 and _total_tree_rows % _req_hint == 0:
                            _record_req_count = int(_req_hint)
                            _record_group_size = int(_total_tree_rows // _req_hint)
                        else:
                            _depth_rows_for_record = getattr(attn_metadata, "fa_tree_depth_rows", None)
                            if _depth_rows_for_record is not None:
                                _record_group_size = int(len(_depth_rows_for_record))
                            if _record_group_size <= 0:
                                _record_group_size = int(_lumo_fa_os.environ.get("LUMO_FA_TREE_GROUP_SIZE", "4"))
                            if _record_group_size > 0 and _total_tree_rows % _record_group_size == 0:
                                _record_req_count = int(_total_tree_rows // _record_group_size)
                    except Exception:
                        _record_group_size = 0
                        _record_req_count = 0
"""
    if record_old in text:
        text = text.replace(record_old, record_new, 1)

    detail_old = """            "commit_mode": "state_copy",
            "accepted_counts": accepted_counts,
            "record_count": len(_LUMO_FA_REPLAY_LAYERS),
"""
    detail_new = """            "commit_mode": "state_copy_branch_row",
            "accepted_counts": accepted_counts,
            "accepted_tree_rows": [int(x) for x in list(globals().get("_LUMO_FA_LAST_ACCEPTED_TREE_ROWS", []) or [])],
            "record_count": len(_LUMO_FA_REPLAY_LAYERS),
"""
    if detail_old in text:
        text = text.replace(detail_old, detail_new, 1)

    group_old = """        record_group_size = int(rec.get("tree_group_size") or 0)
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
"""
    group_new = """        record_group_size = int(rec.get("tree_group_size") or 0)
        record_req_count = int(rec.get("tree_req_count") or 0)
        try:
            expected_req = int(expected_req_count or 0)
        except Exception:
            expected_req = 0
        if expected_req > 0 and total_tokens % expected_req == 0:
            req_count = expected_req
            group_size = total_tokens // expected_req
        elif record_req_count > 0 and total_tokens % record_req_count == 0:
            req_count = record_req_count
            group_size = total_tokens // record_req_count
        elif record_group_size > 0 and total_tokens % record_group_size == 0:
            group_size = record_group_size
            req_count = max(1, total_tokens // group_size)
        elif len(accepted_counts) > 1 and total_tokens % len(accepted_counts) == 0:
            req_count = len(accepted_counts)
            group_size = max(1, total_tokens // req_count)
        else:
            group_size = total_tokens
            req_count = 1
        req_count = min(max(1, int(req_count)), len(accepted_counts))
        accepted_tree_rows = list(globals().get("_LUMO_FA_LAST_ACCEPTED_TREE_ROWS", []) or [])
"""
    if group_old not in text:
        raise RuntimeError('branch accepted-row state-copy group anchor not found')
    text = text.replace(group_old, group_new, 1)

    final_old = """            base = req_i * group_size
            final_row = base + tokens - 1
            prefix_idx = int(initial_flat[base].item())
"""
    final_new = """            base = req_i * group_size
            fallback_local = max(0, min(tokens - 1, group_size - 1))
            local_row = fallback_local
            if req_i < len(accepted_tree_rows):
                try:
                    candidate = int(accepted_tree_rows[req_i])
                    if 0 <= candidate < group_size:
                        local_row = candidate
                except Exception:
                    local_row = fallback_local
            final_row = base + local_row
            prefix_idx = int(initial_flat[base].item())
"""
    if final_old not in text:
        raise RuntimeError('branch accepted-row state-copy final_row anchor not found')
    text = text.replace(final_old, final_new, 1)

    summary_old = """                "commit_mode": "state_copy",
                "accepted_counts": accepted_counts,
                "copied_requests": int(copied),
"""
    summary_new = """                "commit_mode": "state_copy_branch_row",
                "accepted_counts": accepted_counts,
                "accepted_tree_rows": [int(x) for x in list(globals().get("_LUMO_FA_LAST_ACCEPTED_TREE_ROWS", []) or [])],
                "copied_requests": int(copied),
"""
    if summary_old in text:
        text = text.replace(summary_old, summary_new, 1)

    text = sentinel + '\n' + text
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied branch accepted-row state-copy patch')
LUMOFABRANCHROWCOPY
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
                if block_table_tensor.size(1) < _node_count + 1:
                    raise RuntimeError(
                        "LUMO_FA_UNIQUE_NODES requires root + node state slots: "
                        f"need at least {_node_count + 1}, got "
                        f"{block_table_tensor.size(1)}")
                _rows = block_table_tensor[
                    spec_sequence_masks, :_node_count + 2
                ].contiguous()
                _req_count = int(_rows.shape[0])
                if _rows.shape[1] >= _node_count + 2:
                    _write_slots_2d = _rows[:, 1:_node_count + 2].contiguous()
                else:
                    _write_slots_2d = _rows[:, :_node_count + 1].contiguous()
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
sentinel = '# LUMO_FA_SPEC_OUTPUT_ACTUAL_TOKEN_CLIP'
if sentinel in text:
    print('[TRACK-B-PRELAUNCH] F_a spec output actual-token clip already present')
else:
    old = """        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
"""
    new = """        elif spec_sequence_masks is not None:
            _lumo_spec_out = core_attn_out_spec.squeeze(0)
            core_attn_out[:num_actual_tokens] = _lumo_spec_out[:num_actual_tokens]
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
"""
    if old not in text:
        raise RuntimeError('F_a spec output actual-token clip anchor not found')
    text = text.replace(old, new, 1)
    text = sentinel + '\n' + text
    gl.write_text(text)
    import py_compile
    py_compile.compile(str(gl), doraise=True)
    print('[TRACK-B-PRELAUNCH] applied F_a spec output actual-token clip')
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

def _prelaunch_for(
    config: str,
    tree: bool = False,
    tree_debug: bool = False,
    fb: bool = False,
    independent_rows: bool = False,
    spines: int = 1,
) -> str:
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
    fa_unique = bool(tree)
    dbg = "export LUMO_TREE_DRAFT_DEBUG=1\n" if (tree and tree_debug) else ""
    tree_blocks = (
        _TREE_ATTN_BLOCK
        + _MROPE_TREE_BLOCK
        + _TREE_REJECTION_BLOCK
        + _TREE_REJECTION_RANDOM_FIX_BLOCK
        + _TREE_ACCEPTED_ROW_KERNEL_BLOCK
        + _TREE_ACCEPTED_ROW_COMMIT_BLOCK
        + _TREE_PATH_LCP_MAX_BLOCK
    )
    fa_unique_env = (
        "export LUMO_FA_UNIQUE_NODES=1\n"
        if fa_unique else ""
    )
    # Batch-invariant vLLM must be enabled through the host-side ModelServer
    # knob so the launch command also gets a concrete attention backend. A raw
    # inner-container VLLM_BATCH_INVARIANT export makes vLLM fail at init with
    # attention_backend=None.
    tree_debug_exports = ""
    fb_debug_exports = ""
    for _name in (
        "LUMO_TREE_PER_PATH_DRAFTER_LOG",
        "LUMO_TREE_PATH_LCP_LOG",
        "LUMO_TREE_ACCEPT_PATH_LOG",
    ):
        if os.environ.get(_name):
            tree_debug_exports += f"export {_name}={os.environ[_name]}\n"
    independent_env = (
        "export LUMO_INDEPENDENT_ROWS=1\n"
        f"export LUMO_IR_SPINES={int(spines)}\n"
        if independent_rows else ""
    )
    fb_env = fa_unique_env + tree_debug_exports + independent_env
    mtp_draft_trace = f"export LUMO_MTP_DRAFT_TRACE_FILE={os.environ['LUMO_MTP_DRAFT_TRACE_FILE']}\n" if os.environ.get("LUMO_MTP_DRAFT_TRACE_FILE") else ""
    mtp_draft_trace_block = _MTP_DRAFT_TRACE_BLOCK if (
        os.environ.get("LUMO_MTP_DRAFT_TRACE_FILE")
    ) else ""
    stale_fb_guard = "" if (fb or fa_unique) else _NO_STALE_TREE_EXPERIMENT_PATCHES_BLOCK
    mamba_dup_free_fix = (
        _MAMBA_ALIGN_DUP_STATE_FREE_FIX_BLOCK
        if fa_unique
        else ""
    )
    block_pool_dedup_free_fix = (
        _BLOCK_POOL_DEDUP_FREE_FIX_BLOCK
        if fa_unique
        else ""
    )
    free_queue_membership_fix = (
        _FREE_QUEUE_MEMBERSHIP_FIX_BLOCK
        if fa_unique
        else ""
    )
    return (_QWEN36_FP8_CONFIG_FIX_BLOCK + _CAUSAL_CONV_CUDAGRAPH_ASSERT_FIX_BLOCK
            + mamba_dup_free_fix + free_queue_membership_fix + block_pool_dedup_free_fix
            + stale_fb_guard + dbg + "export LUMO_CUDAGRAPH_RUNTIME_TELEMETRY=1\n"
            + fb_env + mtp_draft_trace + base + _CUDAGRAPH_RUNTIME_TELEMETRY_BLOCK + _SPEC_TRACE_BLOCK
            + mtp_draft_trace_block
            + (_INDEPENDENT_ROWS_BLOCK if independent_rows else "")
            + ((_TREE_PER_PATH_DRAFTER_BLOCK + tree_blocks) if tree else "")
            + (_TREE_GDN_PREFIX_STATE_BLOCK if fa_unique else "")
            + (_FA_UNIQUE_NODES_BLOCK if fa_unique else "")
            + (_FA_UNIQUE_BATCH4_PACK_BLOCK if fa_unique else "")
            + (_FA_UNIQUE_BATCH4_STARTUP_FIX_BLOCK if fa_unique else ""))


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


def _apply_max_num_seqs(src: str, value: str | None) -> str:
    if value is None:
        return src
    max_num_seqs = int(value)
    if max_num_seqs < 1:
        raise RuntimeError(f"invalid max_num_seqs override: {value}")
    old = "    max_num_seqs: 4"
    if src.count(old) != 1:
        raise RuntimeError("max_num_seqs anchor not unique in bundle")
    return src.replace(old, f"    max_num_seqs: {max_num_seqs}", 1)


def _apply_agentic_request_shaping(src: str) -> str:
    """Keep the proxy admission cap aligned with B4 agentic measurement.

    The legacy MTP bundle carries ``request_shaping.target_concurrency: 1``.
    The proxy maps that to one active eval request and no queue, which rejects
    the SWE B=4 workload before it reaches vLLM.  vLLM row capacity remains
    governed by ``max_num_seqs``; this only admits the four public Codex
    requests that the benchmark asks for.
    """
    import re as _re

    match = _re.search(r"(?m)^    max_num_seqs: ([0-9]+)$", src)
    if match is None:
        raise RuntimeError("max_num_seqs missing in bundle")
    max_num_seqs = int(match.group(1))
    requested_eval_cap = int(os.environ.get("LUMO_PROXY_CONCURRENCY_CAP_EVAL", "4"))
    eval_cap = max(1, min(requested_eval_cap, max_num_seqs))
    queue_depth = int(os.environ.get("LUMO_PROXY_ADMISSION_QUEUE_DEPTH", "128"))
    if queue_depth < 0:
        raise RuntimeError("LUMO_PROXY_ADMISSION_QUEUE_DEPTH must be >= 0")
    block = (
        "  request_shaping:\n"
        f"    concurrency_cap_eval: {eval_cap}\n"
        "    concurrency_cap_rollout: 0\n"
        f"    admission_queue_depth_max: {queue_depth}\n"
    )
    pattern = r"(?m)^  request_shaping:\n(?:^    .*\n)+"
    if _re.search(pattern, src) is None:
        raise RuntimeError("request_shaping block missing in bundle")
    return _re.sub(pattern, block, src, count=1)


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
    src = _apply_max_num_seqs(src, os.environ.get("LUMO_VLLM_MAX_NUM_SEQS"))
    src = _apply_agentic_request_shaping(src)
    src = _apply_enforce_eager(src, os.environ.get("LUMO_ENFORCE_EAGER"))
    cuda_graph_capture = os.environ.get("LUMO_CUDAGRAPH_MODE") or os.environ.get("LUMO_CUDA_GRAPH_CAPTURE")
    src = _apply_cuda_graph_capture(src, cuda_graph_capture)
    cuda_graph_capture_sizes = os.environ.get("LUMO_CUDAGRAPH_CAPTURE_SIZES")
    src = _apply_cuda_graph_capture_sizes(src, cuda_graph_capture_sizes)
    kvtag = "" if kv_cache_dtype is None else f"-kv{kv_cache_dtype}"
    eager_tag = "-eager" if os.environ.get("LUMO_ENFORCE_EAGER") is not None else ""
    cg_tag = "" if cuda_graph_capture is None else f"-cg{cuda_graph_capture.strip().lower()}"
    cgs_tag = "-cgpacked" if os.environ.get("LUMO_FA_PACKED_CUDAGRAPH_SIZES") is not None else ("" if cuda_graph_capture_sizes is None else "-cgsizes")
    # speculative_token_tree passes through load_tuned_config (the
    # spec_decode_fields_only allowlist is advisory, not enforced); we still
    # add it to the allowlist below for provenance. vLLM only supports REGULAR
    # trees (uniform children/level). For a TREE, num_speculative_tokens must be
    # the NODE COUNT (len(tree_choices)), not the depth -- the runner's draft
    # output buffer is sized by num_speculative_tokens and propose_tree emits one
    # draft per tree node (else: RuntimeError size a(depth) != b(nodes)).
    import ast as _ast
    n_spec = n if tree is None else len(_ast.literal_eval(tree))
    tag = (f"mtp{n}" if tree is None else f"mtp{n}tree{n_spec}") + kvtag + eager_tag + cg_tag + cgs_tag
    src = src.replace("bundle_id: 712fd011-4b16-4051-9e8c-875405b70f5b",
                      f"bundle_id: e0000000-{tag}-4000-9000-config-e-qwen36")
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


def _default_tree(n: int, spines: int | None = None) -> str:
    """Config F's default REGULAR N-spine tree.

    Each root is one candidate spine, extended linearly to depth n. The default
    remains top-2 for the FR7 gate, while LUMO_TREE_SPINES lets the same verifier
    exercise 2-10 spines without changing code.
    """
    if spines is None:
        spines = int(os.environ.get("LUMO_TREE_SPINES", "2"))
    if not (1 <= spines <= 10):
        raise RuntimeError(f"--spines must be in [1, 10], got {spines}")
    nodes = {
        (root,) + (0,) * level
        for level in range(n)
        for root in range(spines)
    }
    return str(sorted(nodes, key=lambda t: (len(t), t)))


def main() -> int:
    ap = argparse.ArgumentParser()
    # Configs: D is the legacy suffix stack. Fb is the maintained MTP path.
    # --row-mode=tree uses the consolidated speculative_token_tree path.
    # --row-mode=independent uses native linear MTP plus hidden co-resident
    # request rows, one per spine.
    ap.add_argument("--config", choices=["D", "Fb"], required=True)
    ap.add_argument("--mtp", type=int, default=5, help="MTP depth for config Fb")
    ap.add_argument("--row-mode", choices=["tree", "independent"], default="tree",
                    help="config Fb only: tree=speculative_token_tree, "
                         "independent=N native co-resident sequence rows")
    ap.add_argument("--spines", type=int, default=int(os.environ.get("LUMO_TREE_SPINES", "2")),
                    help="number of root spines/rows for config Fb; 1 is the E5-equivalent chain")
    ap.add_argument("--tree", default=None,
                    help="config Fb only: override the speculative_token_tree literal "
                         "(default: _default_tree(--mtp)). Must be a REGULAR tree whose "
                         "max depth equals --mtp.")
    ap.add_argument("--tree-debug", action="store_true",
                    help="config Fb only: export LUMO_TREE_DRAFT_DEBUG=1 so propose_tree "
                         "logs per-level proposed draft tokens to /logs/tree_draft_debug.jsonl")
    ap.add_argument("--kv-cache-dtype", default=None,
                    choices=["auto", "fp8_e5m2", "fp8_e4m3"],
                    help="override realized KV cache dtype. fp8_e4m3 is the FP8 KV that "
                         "the fp8 checkpoint accepts (fp8_e5m2 is rejected -> auto). Default: "
                         "use the bundle's value (fp8_e5m2 -> auto for this checkpoint).")
    args = ap.parse_args()
    if args.row_mode != "tree" and args.config != "Fb":
        ap.error("--row-mode is only valid with --config Fb")
    is_tree = args.config == "Fb" and args.row_mode == "tree"
    is_independent = args.config == "Fb" and args.row_mode == "independent"
    is_fb = False
    if is_tree or is_independent:
        os.environ["LUMO_BATCH_INVARIANT_VLLM"] = "1"
    if is_independent:
        if not (1 <= args.spines <= 10):
            raise RuntimeError(f"--spines must be in [1, 10], got {args.spines}")
        os.environ["LUMO_INDEPENDENT_ROWS"] = "1"
        os.environ["LUMO_IR_SPINES"] = str(args.spines)
        if os.environ.get("LUMO_VLLM_MAX_NUM_SEQS") is None:
            os.environ["LUMO_VLLM_MAX_NUM_SEQS"] = str(4 * args.spines)
    if args.tree is not None and not is_tree:
        ap.error("--tree is only valid with --config Fb --row-mode tree")
    tree = (args.tree or _default_tree(args.mtp, args.spines)) if is_tree else None
    if args.config == "D":
        bundle = _d_bundle(kv_cache_dtype=args.kv_cache_dtype)
    else:  # Fb -- MTP bundle; tree mode adds speculative_token_tree
        bundle = _mtp_bundle(args.mtp, tree=tree, kv_cache_dtype=args.kv_cache_dtype)
    server = ModelServer(
        registry_path=REPO / "model_registry.yaml",
        port=int(os.environ.get("LUMO_VLLM_PORT", "9950")),
        container_name=os.environ.get(
            "LUMO_VLLM_CONTAINER_NAME", "lumo-vllm-track-b-suffix"),
        logs_root=Path(os.environ.get(
            "LUMO_VLLM_LOGS_ROOT", "/tmp/lumo-l0c-fp8-cutlass-run30-logs")),
        triton_cache_root=Path(os.environ.get(
            "LUMO_VLLM_TRITON_CACHE_ROOT",
            "/tmp/lumo-l0c-fp8-cutlass-run30-triton")),
        state_root=Path(os.environ.get(
            "LUMO_VLLM_STATE_ROOT", "/tmp/lumo-l0c-fp8-cutlass-run30-state")),
        proxy_port=int(os.environ.get("LUMO_VLLM_PROXY_PORT", "8088")),
        ready_timeout_s=900,
        prelaunch_shell=_prelaunch_for(
            args.config,
            tree=is_tree,
            tree_debug=args.tree_debug,
            fb=is_fb,
            independent_rows=is_independent,
            spines=args.spines,
        ),
    )
    server.load_tuned_config(bundle)
    server.start("qwen3.6-27b")
    tree_desc = f" tree={tree}" if is_tree else ""
    row_desc = f" row_mode={args.row_mode} spines={args.spines}" if args.config == "Fb" else ""
    mtp_desc = args.mtp if args.config == "Fb" else "-"
    kv_desc = args.kv_cache_dtype or "bundle-default"
    print(f"READY config={args.config} mtp={mtp_desc}{row_desc}{tree_desc} kv={kv_desc} bundle={bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
