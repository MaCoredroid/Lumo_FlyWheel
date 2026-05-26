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
import argparse, sys
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
        '        try:',
        '            import json as _lj, time as _lt, os as _lo',
        '            global _LUMO_SPEC_FH',
        '            try:',
        '                _LUMO_SPEC_FH',
        '            except NameError:',
        '                _LUMO_SPEC_FH = open(_lo.environ.get("LUMO_PER_REQ_SPEC_TRACE", "/logs/per_req_spec_trace.jsonl"), "a", buffering=1)',
        '            _linv = (num_invalid_spec_tokens.get(request_id, 0) if num_invalid_spec_tokens else 0)',
        '            _LUMO_SPEC_FH.write(_lj.dumps({"ts": round(_lt.time(), 4), "rid": request_id, "draft": num_draft_tokens, "acc": num_accepted_tokens, "inv": _linv}) + chr(10))',
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


def _prelaunch_for(config: str, tree: bool = False, tree_debug: bool = False) -> str:
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
    return dbg + base + _SPEC_TRACE_BLOCK + (tree_blocks if tree else "")


def _mtp_bundle(n: int, tree: str | None = None) -> str:
    src = Path("/tmp/lumo-track-b-bundle-qwen36-off/bundle.yaml").read_text()
    tag = f"mtp{n}" if tree is None else f"mtp{n}tree"
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
    ap.add_argument("--config", choices=["D", "E", "F"], required=True)
    ap.add_argument("--mtp", type=int, default=1, help="num_speculative_tokens (MTP depth) for config E/F")
    ap.add_argument("--tree", default=None,
                    help="config F only: override the speculative_token_tree literal "
                         "(default: _default_tree(--mtp)). Must be a REGULAR tree whose "
                         "max depth equals --mtp.")
    ap.add_argument("--tree-debug", action="store_true",
                    help="config F only: export LUMO_TREE_DRAFT_DEBUG=1 so propose_tree "
                         "logs per-level proposed draft tokens to /logs/tree_draft_debug.jsonl")
    args = ap.parse_args()
    is_tree = args.config == "F"
    if args.tree is not None and not is_tree:
        ap.error("--tree is only valid with --config F")
    tree = (args.tree or _default_tree(args.mtp)) if is_tree else None
    if args.config == "D":
        bundle = "/tmp/lumo-track-b-bundle-qwen36/bundle.yaml"
    else:  # E or F -- MTP bundle (F adds speculative_token_tree)
        bundle = _mtp_bundle(args.mtp, tree=tree)
    server = ModelServer(
        registry_path=REPO / "model_registry.yaml", port=9950,
        container_name="lumo-vllm-track-b-suffix",
        logs_root=Path("/tmp/lumo-l0c-fp8-cutlass-run30-logs"),
        triton_cache_root=Path("/tmp/lumo-l0c-fp8-cutlass-run30-triton"),
        state_root=Path("/tmp/lumo-l0c-fp8-cutlass-run30-state"),
        proxy_port=8088, ready_timeout_s=900,
        prelaunch_shell=_prelaunch_for(args.config, tree=is_tree, tree_debug=args.tree_debug),
    )
    server.load_tuned_config(bundle)
    server.start("qwen3.6-27b")
    tree_desc = f" tree={tree}" if is_tree else ""
    mtp_desc = args.mtp if args.config in ("E", "F") else "-"
    print(f"READY config={args.config} mtp={mtp_desc}{tree_desc} bundle={bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
