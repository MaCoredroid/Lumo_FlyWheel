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


def select(args) -> dict:
    sc = json.load(open(args.scores))
    rows = [r for r in sc["prefixes"] if r["native_margin_nats"] != float("inf")]
    med = sorted(r["native_margin_nats"] for r in rows)[len(rows) // 2]
    rng = random.Random(args.seed)
    ctx = lambda r: "short" if r["stratum"] == "short" else ("long" if r["stratum"] == "long" else "medium")
    mar = lambda r: "small" if r["native_margin_nats"] < med else "large"
    cells = {}
    for r in rows:
        cells.setdefault((ctx(r), mar(r)), []).append(r["id"])
    for k in cells:
        rng.shuffle(cells[k])
    pilot, confirm, used = [], [], set()
    notes = []
    for c in ("short", "long"):
        for m in ("small", "large"):
            ids = [i for i in cells.get((c, m), []) if i not in used]
            take = ids[:args.pilot_per_cell]
            if len(take) < args.pilot_per_cell:
                fill = [i for i in cells.get(("medium", m), []) if i not in used][: args.pilot_per_cell - len(take)]
                notes.append(f"pilot cell ({c},{m}) filled {len(fill)} from medium context")
                take += fill
            pilot += [{"id": i, "cell": [c, m]} for i in take]; used.update(take)
    for c in ("short", "long"):
        for m in ("small", "large"):
            ids = [i for i in cells.get((c, m), []) if i not in used]
            take = ids[:args.confirm_per_cell]
            if len(take) < args.confirm_per_cell:
                fill = [i for i in cells.get(("medium", m), []) if i not in used][: args.confirm_per_cell - len(take)]
                notes.append(f"confirmation cell ({c},{m}) filled {len(fill)} from medium context")
                take += fill
            confirm += [{"id": i, "cell": [c, m]} for i in take]; used.update(take)
    frozen = {"schema": "e7a.frozen_prefixes.v1", "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": args.seed,
              "margin_median_nats": med, "strata": {"context": ["short", "long"], "margin": ["small(<median)", "large(>=median)"]},
              "pilot_per_cell": args.pilot_per_cell, "confirm_per_cell": args.confirm_per_cell,
              "pilot": pilot, "confirmation": confirm, "notes": notes,
              "scores_sha256": hashlib.sha256(Path(args.scores).read_bytes()).hexdigest(),
              "rule": "frozen before any tree/candidate output is viewed; pilot and confirmation are disjoint; counts are a bounded diagnostic design, not power"}
    Path(args.out).write_text(json.dumps(frozen, indent=1))
    print(f"frozen: pilot={len(pilot)} confirmation={len(confirm)} median margin={med:.3f} notes={notes} -> {args.out}")
    return frozen


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--tokenizer", default="/models/qwen3.6-27b-fp8"); b.add_argument("--source-roots", nargs="+", default=["/work/docs", "/work/research"])
    b.add_argument("--per-stratum", type=int, default=32); b.add_argument("--seed", type=int, default=20260921); b.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--pool", required=True); s.add_argument("--server", default="http://127.0.0.1:9951"); s.add_argument("--model", default="qwen3.6-27b")
    s.add_argument("--seed", type=int, default=20260921); s.add_argument("--out", required=True)
    f = sub.add_parser("select"); f.add_argument("--scores", required=True); f.add_argument("--seed", type=int, default=20260921)
    f.add_argument("--pilot-per-cell", type=int, default=2); f.add_argument("--confirm-per-cell", type=int, default=8); f.add_argument("--out", required=True)
    args = ap.parse_args()
    {"build": build_pool, "score": score_pool, "select": select}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
