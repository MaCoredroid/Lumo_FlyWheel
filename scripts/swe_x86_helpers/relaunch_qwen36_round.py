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


def _prelaunch_for(config: str, tree: bool = False) -> str:
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
    return base + _SPEC_TRACE_BLOCK + (_TREE_ATTN_BLOCK if tree else "")


def _mtp_bundle(n: int, tree: str | None = None) -> str:
    src = Path("/tmp/lumo-track-b-bundle-qwen36-off/bundle.yaml").read_text()
    tag = f"mtp{n}" if tree is None else f"mtp{n}tree"
    src = src.replace("bundle_id: 712fd011-4b16-4051-9e8c-875405b70f5b",
                      f"bundle_id: e0000000-{tag}-4000-9000-config-e-qwen36")
    # speculative_token_tree passes through load_tuned_config (the
    # spec_decode_fields_only allowlist is advisory, not enforced); we still
    # add it to the allowlist below for provenance. num_speculative_tokens must
    # equal the tree depth (max tuple length) so vLLM's MTP layer is re-forwarded
    # once per level. vLLM only supports REGULAR trees (uniform children/level).
    spec_block = f"  spec_decode:\n    method: qwen3_5_mtp\n    num_speculative_tokens: {n}"
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
        prelaunch_shell=_prelaunch_for(args.config, tree=is_tree),
    )
    server.load_tuned_config(bundle)
    server.start("qwen3.6-27b")
    tree_desc = f" tree={tree}" if is_tree else ""
    mtp_desc = args.mtp if args.config in ("E", "F") else "-"
    print(f"READY config={args.config} mtp={mtp_desc}{tree_desc} bundle={bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
