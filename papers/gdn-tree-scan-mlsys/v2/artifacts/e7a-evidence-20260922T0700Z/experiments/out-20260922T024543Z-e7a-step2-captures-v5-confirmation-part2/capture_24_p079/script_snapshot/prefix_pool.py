#!/usr/bin/env python3
"""E7a step 2: fresh-prefix pool, native-reference scoring, and stratified frozen selection.

Design (frozen BEFORE any tree/candidate output is viewed — review-experiments.md):
  * pool   : deterministic prefixes cut from local plain-text sources (repository docs) at three token-length
             strata (short ~256, medium ~1024, long ~4096 tokens) with a fixed seed; each prefix records the
             source path, sha256 of the source text, character span, token count, and prefix sha256.
  * score  : the NATIVE reference server (stock vLLM MTP, FLASH_ATTN) is queried for the next-token top-5
             logprobs at each prefix end (max_tokens=1, temperature=0); the margin = logprob(top1) - logprob(top2).
             This is native-reference behavior only; no tree kernel output is consulted.
  * select : strata = context {short, long} x margin {small (< pool median), large (>= median)}; the pilot takes
             2 per cell = 8 prefixes; the confirmation set takes 8 per cell = 32 DISJOINT prefixes, medium
             context prefixes fill cells that run short (recorded). Selection uses seeded shuffles; IDs, seeds,
             strata and the pool manifest are written to a frozen JSON.
Counts define a bounded diagnostic design, not statistical power (review-experiments.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

STRATA_TOKENS = {"short": 256, "medium": 1024, "long": 4096}


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def collect_sources(roots: list[str], min_chars: int) -> list[Path]:
    out = []
    for r in roots:
        for p in sorted(Path(r).rglob("*.md")):
            try:
                if p.stat().st_size >= min_chars and not p.is_symlink():
                    out.append(p)
            except OSError:
                pass
    return out


def build_pool(args) -> dict:
    from transformers import AutoTokenizer  # inside the pinned image
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    rng = random.Random(args.seed)
    sources = collect_sources(args.source_roots, min_chars=2000)
    rng.shuffle(sources)
    pool = []
    per_stratum = {k: 0 for k in STRATA_TOKENS}
    for src in sources:
        if all(per_stratum[k] >= args.per_stratum for k in STRATA_TOKENS):
            break
        text = src.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"[ \t]+\n", "\n", text)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        for stratum, n_tok in STRATA_TOKENS.items():
            if per_stratum[stratum] >= args.per_stratum or len(ids) < n_tok + 8:
                continue
            start = rng.randrange(0, len(ids) - n_tok) if len(ids) > n_tok else 0
            piece = ids[start:start + n_tok]
            ptxt = tok.decode(piece, skip_special_tokens=True)
            # re-tokenize the decoded text so the served prompt and the recorded token count agree
            n_actual = len(tok(ptxt, add_special_tokens=False)["input_ids"])
            pid = f"p{len(pool):03d}"
            pool.append({"id": pid, "stratum": stratum, "target_tokens": n_tok, "tokens": n_actual,
                         "source": str(src), "source_sha256": sha256_text(text), "token_start": start,
                         "prefix_sha256": sha256_text(ptxt), "text": ptxt})
            per_stratum[stratum] += 1
    man = {"schema": "e7a.prefix_pool.v1", "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seed": args.seed, "tokenizer": args.tokenizer, "per_stratum": args.per_stratum, "strata_tokens": STRATA_TOKENS,
           "source_roots": args.source_roots, "n_sources_scanned": len(sources), "counts": per_stratum, "prefixes": pool}
    Path(args.out).write_text(json.dumps(man, indent=1))
    print(f"pool: {len(pool)} prefixes, counts={per_stratum} -> {args.out}")
    return man


def score_pool(args) -> dict:
    """Query the NATIVE reference server for next-token top-k logprobs at each prefix end."""
    man = json.load(open(args.pool))
    url = args.server.rstrip("/") + "/v1/completions"
    scored = []
    for p in man["prefixes"]:
        body = {"model": args.model, "prompt": p["text"], "max_tokens": 1, "temperature": 0.0, "logprobs": 5, "seed": args.seed}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.load(resp)
        lat = time.time() - t0
        ch = out["choices"][0]
        lp = ch["logprobs"]
        top = lp["top_logprobs"][0]  # dict token -> logprob
        ranked = sorted(top.items(), key=lambda kv: -kv[1])
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("inf")
        scored.append({**{k: v for k, v in p.items() if k != "text"},
                       "native_top1": ranked[0][0], "native_top1_logprob": ranked[0][1],
                       "native_top2": ranked[1][0] if len(ranked) > 1 else None, "native_margin_nats": margin,
                       "native_tokens_generated": ch.get("text"), "usage_prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
                       "latency_s": lat})
        print(f"  {p['id']} {p['stratum']:6s} tok={scored[-1]['usage_prompt_tokens']} margin={margin:.3f}", flush=True)
    res = {"schema": "e7a.prefix_scores.v1", "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "server": args.server,
           "model": args.model, "reference": "NATIVE stock MTP server; next-token top-5 logprobs at prefix end; temperature 0; no tree output consulted",
           "pool_sha256": hashlib.sha256(Path(args.pool).read_bytes()).hexdigest(), "prefixes": scored}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"scored {len(scored)} -> {args.out}")
    return res


class SelectionError(RuntimeError):
    pass


def select(args) -> dict:
    """Frozen stratified selection with FAIL-CLOSED guards (review 05):
    exact cardinalities, finite margins, unique ids/content hashes, disjoint pilot/confirmation, pool->scores
    hash chain, per-cell sufficiency (medium-context fill only when --allow-medium-fill, and then the ACTUAL
    context stratum is kept on the row), and a per-source-document cap for diversity."""
    sc = json.load(open(args.scores))
    if args.pool:
        pool_sha = hashlib.sha256(Path(args.pool).read_bytes()).hexdigest()
        if sc.get("pool_sha256") != pool_sha:
            raise SelectionError(f"hash chain broken: scores.pool_sha256={sc.get('pool_sha256')} != sha256(pool)={pool_sha}")
        pool_rows = {p["id"]: p for p in json.load(open(args.pool))["prefixes"]}
    else:
        pool_rows = None
    rows = sc["prefixes"]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SelectionError("duplicate prefix ids in scores")
    hashes = [r["prefix_sha256"] for r in rows]
    if len(hashes) != len(set(hashes)):
        raise SelectionError("duplicate prefix content hashes in scores")
    bad = [r["id"] for r in rows if not isinstance(r.get("native_margin_nats"), (int, float)) or r["native_margin_nats"] != r["native_margin_nats"] or r["native_margin_nats"] in (float("inf"), float("-inf"))]
    if bad:
        raise SelectionError(f"non-finite native margins for {bad}")
    if pool_rows is not None:
        for r in rows:
            if r["id"] not in pool_rows or pool_rows[r["id"]]["prefix_sha256"] != r["prefix_sha256"]:
                raise SelectionError(f"score row {r['id']} does not match the pool (id/hash)")
    margins = sorted(r["native_margin_nats"] for r in rows)
    med = margins[len(margins) // 2]
    rng = random.Random(args.seed)
    ctx = lambda r: r["stratum"]  # actual context stratum: short | medium | long
    mar = lambda r: "small" if r["native_margin_nats"] < med else "large"
    cells: dict[tuple, list] = {}
    for r in rows:
        cells.setdefault((ctx(r), mar(r)), []).append(r)
    for k in cells:
        rng.shuffle(cells[k])
    used: set[str] = set()
    per_source: dict[str, int] = {}
    fills: list[str] = []

    def take(cell_ctx: str, m: str, n: int, cap: int, label: str) -> list[dict]:
        chosen = []
        def try_pool(pool_ctx):
            nonlocal chosen
            for r in cells.get((pool_ctx, m), []):
                if len(chosen) >= n:
                    break
                if r["id"] in used or per_source.get(r["source"], 0) >= cap:
                    continue
                chosen.append({"id": r["id"], "prefix_sha256": r["prefix_sha256"], "target_cell": [cell_ctx, m],
                               "context_stratum_actual": r["stratum"], "tokens": r["tokens"], "native_margin_nats": r["native_margin_nats"],
                               "source": r["source"], "filled_from_medium": r["stratum"] != cell_ctx})
                used.add(r["id"]); per_source[r["source"]] = per_source.get(r["source"], 0) + 1
        try_pool(cell_ctx)
        if len(chosen) < n:
            if not args.allow_medium_fill:
                raise SelectionError(f"{label} cell ({cell_ctx},{m}) has only {len(chosen)} eligible prefixes, need {n} (per-source cap {cap}); refusing (use --allow-medium-fill to record medium-context substitutes)")
            before = len(chosen)
            try_pool("medium")
            fills.append(f"{label} cell ({cell_ctx},{m}): {len(chosen) - before} medium-context substitutes (actual stratum recorded on each row)")
            if len(chosen) < n:
                raise SelectionError(f"{label} cell ({cell_ctx},{m}) still short after medium fill: {len(chosen)}/{n}")
        return chosen

    pilot, confirm = [], []
    for c in ("short", "long"):
        for m in ("small", "large"):
            pilot += take(c, m, args.pilot_per_cell, args.max_per_source_pilot, "pilot")
    for c in ("short", "long"):
        for m in ("small", "large"):
            confirm += take(c, m, args.confirm_per_cell, args.max_per_source_confirm, "confirmation")
    want_p, want_c = 4 * args.pilot_per_cell, 4 * args.confirm_per_cell
    if len(pilot) != want_p or len(confirm) != want_c:
        raise SelectionError(f"cardinality mismatch: pilot {len(pilot)}/{want_p}, confirmation {len(confirm)}/{want_c}")
    pid, cid = {r["id"] for r in pilot}, {r["id"] for r in confirm}
    if pid & cid or len(pid) != len(pilot) or len(cid) != len(confirm):
        raise SelectionError("pilot/confirmation not disjoint or contain duplicates")
    frozen = {"schema": "e7a.frozen_prefixes.v2", "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": args.seed,
              "margin_median_nats": med, "strata": {"context": ["short", "long"], "margin": ["small(<median)", "large(>=median)"]},
              "pilot_per_cell": args.pilot_per_cell, "confirm_per_cell": args.confirm_per_cell,
              "max_per_source": {"pilot": args.max_per_source_pilot, "confirmation": args.max_per_source_confirm},
              "counts": {"pilot": len(pilot), "confirmation": len(confirm)}, "pilot": pilot, "confirmation": confirm,
              "medium_fills": fills, "n_source_documents_pilot": len({r["source"] for r in pilot}),
              "n_source_documents_confirmation": len({r["source"] for r in confirm}),
              "scores_sha256": hashlib.sha256(Path(args.scores).read_bytes()).hexdigest(), "pool_sha256": sc.get("pool_sha256"),
              "analysis_unit_note": "unit = prefix; prefixes from the same source document are recorded (source) and must be treated as related spans in any later resampling",
              "rule": "frozen before any tree/candidate output is viewed; pilot and confirmation are disjoint; counts are a bounded diagnostic design, not power"}
    Path(args.out).write_text(json.dumps(frozen, indent=1))
    print(f"frozen: pilot={len(pilot)} confirmation={len(confirm)} median margin={med:.3f} fills={fills} sources pilot/conf={frozen['n_source_documents_pilot']}/{frozen['n_source_documents_confirmation']} -> {args.out}")
    return frozen


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--tokenizer", default="/models/qwen3.6-27b-fp8"); b.add_argument("--source-roots", nargs="+", default=["/work/docs", "/work/research"])
    b.add_argument("--per-stratum", type=int, default=32); b.add_argument("--seed", type=int, default=20260921); b.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--pool", required=True); s.add_argument("--server", default="http://127.0.0.1:9951"); s.add_argument("--model", default="qwen3.6-27b")
    s.add_argument("--seed", type=int, default=20260921); s.add_argument("--out", required=True)
    f = sub.add_parser("select"); f.add_argument("--scores", required=True); f.add_argument("--pool", default=None, help="pool.json to validate the hash chain (recommended)")
    f.add_argument("--seed", type=int, default=20260921)
    f.add_argument("--pilot-per-cell", type=int, default=2); f.add_argument("--confirm-per-cell", type=int, default=8); f.add_argument("--out", required=True)
    f.add_argument("--allow-medium-fill", action="store_true", help="permit recorded medium-context substitutes when a cell is short (actual stratum kept on the row)")
    f.add_argument("--max-per-source-pilot", type=int, default=1); f.add_argument("--max-per-source-confirm", type=int, default=3)
    args = ap.parse_args()
    {"build": build_pool, "score": score_pool, "select": select}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
