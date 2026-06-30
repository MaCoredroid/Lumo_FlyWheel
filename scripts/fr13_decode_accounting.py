#!/usr/bin/env python3
"""FR13 decode + cache accounting reducer (confound-proof).

Reports, for a swe_out root (one or more tasks), the DECODE decomposition that
attributes differences correctly (tree-shape vs kernel/cache overhead), AND the
CACHE-benefit metrics (where the prefix cache actually pays off -- prefill side):

  DECODE (the speed decomposition):
    tok/s (token-weighted) = gen / Sdecode_time     <- headline throughput, but
                                                        length-confounded; don't
                                                        attribute small effects on it
    accept/event           = accepted / drafts       <- TREE-shape effect
    s_per_fwd (ms)         = 1000 * Sdecode / drafts  <- KERNEL/CACHE overhead (the
                                                        least length-confounded metric;
                                                        compare THIS for sub-10% effects)
    -> tok/s ~= (accept/event + 1) / s_per_fwd

  CACHE benefit (prefill side -- this is WHERE the cache helps; decode is ~neutral):
    prefix hit rate = prefix_cache_hits / prefix_cache_queries  <- context reuse leverage
    prefill saved   = prefix_cache_hits tokens not re-prefilled
    TTFT (mean)     = time_to_first_token mean                  <- the direct win
    e2e tok/s       = gen / Se2e_latency                        <- request-level win (TTFT rolls in)

Usage: fr13_decode_accounting.py <swe_out_root> [label]
"""
import sys, re, glob, os


def parse(f):
    d = {}
    try:
        for ln in open(f):
            ln = ln.strip()
            if ln.startswith('#') or not ln:
                continue
            m = re.match(r'(\S+?)(\{[^}]*\})?\s+([0-9eE.+-]+)$', ln)
            if m:
                d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(3))
    except Exception:
        pass
    return d


def reduce_root(root):
    acc = {}
    n_pairs = 0
    for pre in glob.glob(os.path.join(root, '**', 'vllm_metrics_pre.txt'), recursive=True):
        post = pre.replace('_pre.txt', '_post.txt')
        if not os.path.exists(post):
            continue
        a, b = parse(pre), parse(post)
        for k in set(a) | set(b):
            acc[k] = acc.get(k, 0.0) + (b.get(k, 0) - a.get(k, 0))
        n_pairs += 1
    return acc, n_pairs


def g(d, k):
    return d.get(k, 0.0)


def main():
    root = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(root.rstrip('/'))
    d, n = reduce_root(root)
    gen = g(d, 'vllm:generation_tokens_total')
    dec = g(d, 'vllm:request_decode_time_seconds_sum')
    e2e = g(d, 'vllm:e2e_request_latency_seconds_sum')
    reqs = g(d, 'vllm:e2e_request_latency_seconds_count')
    ttft_s = g(d, 'vllm:time_to_first_token_seconds_sum')
    ttft_c = g(d, 'vllm:time_to_first_token_seconds_count')
    drafts = g(d, 'vllm:spec_decode_num_drafts_total')
    acc_tok = g(d, 'vllm:spec_decode_num_accepted_tokens_total')
    draft_tok = g(d, 'vllm:spec_decode_num_draft_tokens_total')
    prompt = g(d, 'vllm:prompt_tokens_total')
    pc_hit = g(d, 'vllm:prefix_cache_hits_total')
    pc_q = g(d, 'vllm:prefix_cache_queries_total')

    def r(x, dv, fmt='%.2f', unit=''):
        return (fmt % (x / dv) + unit) if dv else 'NA'

    print(f"== {label}  ({n} metric-pairs, {reqs:.0f} requests, gen={gen:.0f} tok) ==")
    print("  DECODE:")
    print(f"    tok/s (token-weighted) = {r(gen, dec)}        [gen {gen:.0f} / {dec:.0f}s decode]")
    print(f"    accept/event           = {r(acc_tok, drafts)}        [accepted {acc_tok:.0f} / {drafts:.0f} drafts]")
    print(f"    s_per_fwd              = {r(dec, drafts, '%.4f', 's')}  ({r(1000*dec, drafts, '%.2f', ' ms')}/fwd)  <- weight-floor-dominated; compare ONLY within matched context-length")
    print(f"    tok/draft              = {r(draft_tok, drafts)}  (WARMUP-CONTAMINATED counter: root double-count / phantom pos -- NOT cross-arm comparable)")
    print("  CACHE benefit (prefill side):")
    print(f"    prefix hit rate        = {r(100*pc_hit, pc_q, '%.0f', '%')}   [hits {pc_hit:.0f} / queries {pc_q:.0f}]")
    print(f"    prefill tokens saved   = {pc_hit:.0f}  (of {prompt:.0f} prompt tok = {r(100*pc_hit, prompt, '%.0f', '%')} not re-prefilled)")
    print(f"    TTFT (mean)            = {r(ttft_s, ttft_c, '%.2f', 's')}")
    print(f"    e2e tok/s              = {r(gen, e2e)}        [gen / {e2e:.0f}s e2e]   <- request-level win")


if __name__ == '__main__':
    main()
