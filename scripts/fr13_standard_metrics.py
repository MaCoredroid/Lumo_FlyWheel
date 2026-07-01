#!/usr/bin/env python3
"""Emit FIELD-STANDARD speed/usage metrics for one arm, in the two standard reporting lenses.

  SERVING  (Artificial Analysis / vLLM): Output Speed (decode tok/s, cache-neutral), TPOT (ms/tok),
           TTFT (s, where prefix-caching shows), + accept/event & s_per_fwd for the decode story.
  AGENTIC  (AA Coding Agents / SWE-bench): wall-clock time / task (the standard agentic "speed" =
           a DURATION, not a rate), turns, token usage (input / cache-hit / reasoning / output),
           cache hit rate, resolve verdict.

Rationale: nobody reports an agentic run as a single tok/s; serving reports Output-Speed + TTFT
separately (Output-Speed is decode-only => cache-neutral by construction), and agentic reports
wall-clock-per-task + a token-usage breakdown + hit rate. See FR13 speed-lens research.

Usage: fr13_standard_metrics.py <arm_dir | swe_out_dir> [label] [--json]
Read-only; reuses fr13_decode_accounting.reduce_root. Safe to run post-hoc off the DGX.
"""
import json, glob, os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_decode_accounting import reduce_root, g


def resolve_paths(arg):
    arg = arg.rstrip('/')
    if os.path.isdir(os.path.join(arg, 'swe_out')):
        return os.path.join(arg, 'swe_out'), arg          # arm dir
    if os.path.basename(arg) == 'swe_out':
        return arg, os.path.dirname(arg)                   # swe_out dir
    return arg, arg


def _find(swe_out, name):
    hits = glob.glob(os.path.join(swe_out, '**', name), recursive=True)
    for p in hits:
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {}


def _iso(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def wall_clock(swe_out):
    """Agent wall-clock per task from runner_metadata: full span (started->ended) and the
    agent-only sum (all codex attempts, excl. the grader). Retry-aware."""
    m = _find(swe_out, 'runner_metadata.json')
    full = agent = None
    st, en = _iso(m.get('started_at', '')), _iso(m.get('ended_at', ''))
    if st and en:
        full = (en - st).total_seconds()
    agent = 0.0
    for k in ('codex', 'codex_retry', 'codex_retry2'):
        v = m.get(k)
        if isinstance(v, dict) and isinstance(v.get('elapsed_s'), (int, float)):
            agent += v['elapsed_s']
    return (full, agent or None, m)


def per_turn(arm_dir):
    """Token breakdown from proxy_pair_dumps. reasoning = call-A reasoning (the real per-turn
    thinking). answer = message text + function_call ARGS (the tool call = the real coding output),
    summed over call-A (initial) AND call-B (think_cap_continue) so forced-close answers are counted."""
    d = os.path.join(arm_dir, 'proxy_pair_dumps')
    inits = glob.glob(os.path.join(d, '*initial*'))
    conts = glob.glob(os.path.join(d, '*think_cap_continue*'))
    ins = []
    r_chars = a_chars = 0

    def scan(files, count_reasoning):
        nonlocal r_chars, a_chars
        for f in files:
            try:
                dd = json.load(open(f))
                r = dd.get('response')
                r = json.loads(r) if isinstance(r, str) else r
                if not isinstance(r, dict):
                    continue
                if count_reasoning:
                    u = r.get('usage', {}) or {}
                    it = u.get('input_tokens') or u.get('prompt_tokens')
                    if it is not None:
                        ins.append(int(it))
                for item in r.get('output', []) or []:
                    if not isinstance(item, dict):
                        continue
                    t = item.get('type')
                    if t == 'reasoning' and count_reasoning:
                        for c in item.get('content') or []:
                            r_chars += len(c.get('text', '') or '') if isinstance(c, dict) else 0
                    elif t == 'message':
                        for c in item.get('content') or []:
                            a_chars += len(c.get('text', '') or '') if isinstance(c, dict) else 0
                    elif t in ('function_call', 'tool_call'):
                        args = item.get('arguments') or ''
                        a_chars += len(args) if isinstance(args, str) else len(json.dumps(args))
            except Exception:
                pass

    scan(inits, True)     # call-A: reasoning + any answer
    scan(conts, False)    # call-B: forced answers only (skip the tiny close-stub reasoning)
    return {
        'turns': len(inits),
        'cap_fires': len(conts),
        'ctx': sorted(ins),
        'reasoning_tok_est': r_chars // 4,
        'answer_tok_est': a_chars // 4,   # message text + tool-call args
    }


def collect(arg, label=None):
    swe_out, arm_dir = resolve_paths(arg)
    label = label or os.path.basename(arm_dir)
    d, n = reduce_root(swe_out)
    gen = g(d, 'vllm:generation_tokens_total')
    dec = g(d, 'vllm:request_decode_time_seconds_sum')
    ttft_s = g(d, 'vllm:time_to_first_token_seconds_sum')
    ttft_c = g(d, 'vllm:time_to_first_token_seconds_count')
    drafts = g(d, 'vllm:spec_decode_num_drafts_total')
    acc_tok = g(d, 'vllm:spec_decode_num_accepted_tokens_total')
    prompt = g(d, 'vllm:prompt_tokens_total')
    pc_hit = g(d, 'vllm:prefix_cache_hits_total')
    pc_q = g(d, 'vllm:prefix_cache_queries_total')
    reqs = g(d, 'vllm:e2e_request_latency_seconds_count')
    cs = _find(swe_out, 'campaign_summary.json')
    full_s, agent_s, _ = wall_clock(swe_out)
    pt = per_turn(arm_dir)
    div = lambda x, y: (x / y) if y else None
    ctx = pt['ctx']
    return {
        'label': label,
        'resolve': cs.get('resolved_rate'),
        'verdict_counts': cs.get('verdict_counts'),
        # SERVING lens
        'output_speed_tok_s': div(gen, dec),
        'tpot_ms': div(1000 * dec, gen),
        'ttft_mean_s': div(ttft_s, ttft_c),
        'accept_per_event': div(acc_tok, drafts),
        's_per_fwd_ms': div(1000 * dec, drafts),
        # AGENTIC lens
        'wall_clock_full_s': full_s,       # started_at -> ended_at (incl. grader)
        'wall_clock_agent_s': agent_s,     # sum of codex attempts (excl. grader)
        'turns': pt['turns'],
        'cap_fires': pt['cap_fires'],
        'context_start_tok': ctx[0] if ctx else None,
        'context_final_tok': ctx[-1] if ctx else None,
        'tok_input_cumulative': prompt,
        'tok_cache_hit': pc_hit,
        'cache_hit_rate': div(pc_hit, pc_q),
        'tok_reasoning_est': pt['reasoning_tok_est'],
        'tok_answer_est': pt['answer_tok_est'],
        'tok_raw_generation': gen,     # incl. reasoning + answer + spec drafts + call-B
        'requests': reqs,
    }


def fmt(m):
    f = lambda v, p='%.2f': (p % v) if isinstance(v, (int, float)) else str(v)
    mn = lambda v: ('%.1f min' % (v / 60.0)) if isinstance(v, (int, float)) else '?'
    L = [f"==== {m['label']} — standard metrics ====",
         f"  resolve: {m['resolve']}   verdicts={m['verdict_counts']}",
         "  [SERVING lens — Artificial Analysis / vLLM]",
         f"    Output Speed  = {f(m['output_speed_tok_s'])} tok/s   (decode-only; cache-NEUTRAL)",
         f"    TPOT          = {f(m['tpot_ms'])} ms/token",
         f"    TTFT (mean)   = {f(m['ttft_mean_s'])} s   (prefix-cache shows HERE)",
         f"    accept/event  = {f(m['accept_per_event'])}    s_per_fwd = {f(m['s_per_fwd_ms'])} ms",
         "  [AGENTIC lens — AA Coding Agents / SWE-bench]",
         f"    wall-clock/task = {f(m['wall_clock_agent_s'])} s ({mn(m['wall_clock_agent_s'])} agent) | full {f(m['wall_clock_full_s'])} s ({mn(m['wall_clock_full_s'])} incl grader)",
         f"    turns         = {m['turns']}   (thinking-cap fires: {m['cap_fires']})",
         f"    context window: {m['context_start_tok']} -> {m['context_final_tok']} tok",
         f"    cache hit rate= {f(100*m['cache_hit_rate'],'%.0f') if isinstance(m['cache_hit_rate'],(int,float)) else 'NA'}%",
         "    TOKEN USAGE (AA breakdown):",
         f"      input (cumulative)   = {f(m['tok_input_cumulative'],'%.0f')}",
         f"      cache-hit (reused)   = {f(m['tok_cache_hit'],'%.0f')}",
         f"      reasoning (est)      = {m['tok_reasoning_est']}",
         f"      output/answer (est)  = {m['tok_answer_est']}   [text + tool-call args; codex-visible]",
         f"      raw generation       = {f(m['tok_raw_generation'],'%.0f')}   [incl. spec drafts + call-B]"]
    return "\n".join(L)


def main():
    args = [a for a in sys.argv[1:] if a != '--json']
    as_json = '--json' in sys.argv
    m = collect(args[0], args[1] if len(args) > 1 else None)
    print(json.dumps(m) if as_json else fmt(m))


if __name__ == '__main__':
    main()
