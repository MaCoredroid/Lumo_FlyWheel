#!/usr/bin/env python3
"""Build a harness-aware pre-warm corpus (design §6b) for the Arctic suffix trie, on the HOST (no GPU).

Source = prior qwen-code request-dumps (/tmp/lumo_proxy_request_dumps/*.json) = full agentic conversations.
The MODEL-GENERATED content (assistant turns: text + tool_calls) carries the CROSS-TASK harness-repetitive
patterns the suffix tail needs from token 1 -- tool-call XML/JSON scaffolding, common imports/idioms,
edit-echoes. We extract those, tokenize with the qwen3.6 tokenizer, and write corpus.jsonl (one token-id
list per assistant segment). NEVER-REGRESS: pre-warm only ADDS candidates through the monotone committer.

Usage: fr13_build_prewarm_corpus.py <out.jsonl> [--dumps DIR] [--max N] [--exclude-substr S ...]
  --exclude-substr : skip dumps whose instance/text contains S (avoid leakage of the TEST tasks).
"""
import sys, os, json, glob, argparse, hashlib

def assistant_segments(dump):
    """Yield the model-GENERATED text segments from one request-dump's messages."""
    for m in dump.get("messages", []):
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            yield c
        elif isinstance(c, list):
            for part in c:
                t = part.get("text") if isinstance(part, dict) else None
                if t:
                    yield t
        for tc in (m.get("tool_calls") or []):
            fn = (tc.get("function") or {})
            # serialize the tool-call the way the model emits it (name + JSON args) -- the harness boilerplate
            name = fn.get("name", ""); args = fn.get("arguments", "")
            if name or args:
                yield f'{name}\n{args}' if not isinstance(args, (dict, list)) else f'{name}\n{json.dumps(args)}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dumps", default="/tmp/lumo_proxy_request_dumps")
    ap.add_argument("--max", type=int, default=400, help="max dumps to scan")
    ap.add_argument("--min-toks", type=int, default=8)
    ap.add_argument("--max-toks", type=int, default=256, help="chunk long segments to this")
    ap.add_argument("--exclude-substr", action="append", default=[])
    ap.add_argument("--model", default="/models/qwen3.6-27b-fp8")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    files = sorted(glob.glob(os.path.join(a.dumps, "*.json")))[:a.max]
    seen = set(); corpus = []; n_dumps = 0; n_seg = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        blob = json.dumps(d)[:4000]
        if any(s in blob for s in a.exclude_substr):
            continue
        n_dumps += 1
        for seg in assistant_segments(d):
            n_seg += 1
            ids = tok.encode(seg, add_special_tokens=False)
            # chunk long segments so the trie stores manageable patterns
            for i in range(0, len(ids), a.max_toks):
                chunk = ids[i:i + a.max_toks]
                if len(chunk) < a.min_toks:
                    continue
                h = hashlib.md5(bytes(str(chunk), "utf8")).hexdigest()
                if h in seen:      # dedup exact-repeat segments (they're already high-frequency in the trie)
                    continue
                seen.add(h)
                corpus.append(chunk)

    with open(a.out, "w") as fh:
        for c in corpus:
            fh.write(json.dumps(c) + "\n")
    ntok = sum(len(c) for c in corpus)
    print(f"scanned {n_dumps} dumps, {n_seg} assistant segments -> {len(corpus)} deduped corpus sequences, "
          f"{ntok} tokens (~{ntok*4/1e6:.1f}MB int32) -> {a.out}")

if __name__ == "__main__":
    main()
