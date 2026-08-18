#!/usr/bin/env python3
"""FR14 lane 5A: render the generation traces for the degeneration eyeball, and
compute the mechanical signatures BESIDE them -- never instead of them.

Mark's condition on this lane is that a human read the real outputs.  The
statistics below exist to tell the reader WHERE to look in an 8,000-token
trace, not to issue a verdict: every one of them can be passed by fluent
nonsense and failed by a legitimately repetitive answer (a numbered list of 40
items repeats "detection hint" 40 times by construction).  The verdict lives in
the note, written by whoever read the excerpts.

Signatures computed, all of which the campaign has named as low-precision-head
failure modes:

  type_token_ratio      -- unique words / words.  A repetition loop collapses it.
  max_line_repeat       -- the most times any single line repeats verbatim.
  longest_ngram_loop    -- the longest 8-gram that occurs more than once, and
                           how many times: the direct fingerprint of a decode
                           loop.
  tail_repeat_fraction  -- fraction of the LAST 25% of lines that are duplicates
                           of an earlier line.  Loops start late; a whole-trace
                           average dilutes them away.
  mid_word_breaks       -- occurrences of a lowercase letter, a space, then a
                           lowercase letter continuing what scans as one word
                           (``inva riant``): the classic single-flipped-token
                           artefact.
  nonascii_fraction     -- gibberish from a corrupted head is usually a burst of
                           CJK/symbol bytes in an English trace.
  unbalanced_delims     -- brace/bracket/paren/backtick imbalance in code output.
  tool_call_wellformed  -- tool calls parse as JSON and match the declared
                           parameter names.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter


def text_of(resp: dict) -> str:
    msg = resp["choices"][0]["message"]
    return (msg.get("reasoning_content") or "") + (msg.get("content") or "")


def signatures(t: str) -> dict:
    words = t.split()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    line_counts = Counter(lines)

    grams = Counter(tuple(words[i : i + 8]) for i in range(max(0, len(words) - 7)))
    top_gram, top_gram_n = ("", 0)
    if grams:
        g, n = grams.most_common(1)[0]
        top_gram, top_gram_n = " ".join(g), n

    tail = lines[int(len(lines) * 0.75) :]
    seen_before = set(lines[: int(len(lines) * 0.75)])
    tail_rep = sum(1 for ln in tail if ln in seen_before) / max(1, len(tail))

    # A real mid-word break leaves a fragment that is not an English word on
    # either side; approximate it by "short lowercase fragment followed by a
    # short lowercase fragment, inside a word-ish run", which over-reports and
    # is therefore safe to eyeball.
    mid_word_breaks = len(re.findall(r"[a-z]{1,3} [a-z]{1,3}(?=[^a-zA-Z]|$)", t))

    nonascii = sum(1 for c in t if ord(c) > 0x2FFF)

    delims = {
        "{}": t.count("{") - t.count("}"),
        "[]": t.count("[") - t.count("]"),
        "()": t.count("(") - t.count(")"),
        "```": t.count("```") % 2,
    }

    return {
        "chars": len(t),
        "words": len(words),
        "lines": len(lines),
        "type_token_ratio": (len(set(words)) / len(words)) if words else 0.0,
        "max_line_repeat": max(line_counts.values()) if line_counts else 0,
        "most_repeated_line": (
            line_counts.most_common(1)[0][0][:120] if line_counts else ""
        ),
        "longest_ngram_loop": {"ngram": top_gram[:160], "count": top_gram_n},
        "tail_repeat_fraction": tail_rep,
        "mid_word_break_candidates": mid_word_breaks,
        "nonascii_chars": nonascii,
        "nonascii_fraction": nonascii / max(1, len(t)),
        "unbalanced_delims": delims,
    }


def tool_check(resp: dict, prompt: dict) -> dict:
    msg = resp["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    declared = {
        t["function"]["name"]: set(t["function"]["parameters"]["properties"])
        for t in prompt.get("tools", [])
    }
    out = {"n_tool_calls": len(calls), "calls": []}
    for c in calls:
        fn = c.get("function", {})
        name = fn.get("name")
        raw = fn.get("arguments", "")
        rec = {"name": name, "raw_arguments": raw[:400], "known_function": name in declared}
        try:
            args = json.loads(raw)
            rec["json_parses"] = True
            rec["args"] = args
            if name in declared:
                rec["unknown_params"] = sorted(set(args) - declared[name])
        except Exception as e:
            rec["json_parses"] = False
            rec["parse_error"] = f"{type(e).__name__}: {e}"
        out["calls"].append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--excerpt-chars", type=int, default=1400)
    a = ap.parse_args()

    data = json.load(open(a.generations))
    report = {"schema": "fr14.lane5a.eyeball.v1", "traces": []}
    for item in data:
        p, r = item["prompt"], item["response"]
        rec = {
            "id": p["id"],
            "regime": p["regime"],
            "max_tokens": p["max_tokens"],
            "sampling": p["sampling"],
            "wall_s": item["wall_s"],
            "error": item["error"],
        }
        if r is None:
            rec["FAILED"] = True
            report["traces"].append(rec)
            continue
        t = text_of(r)
        rec["usage"] = r.get("usage")
        rec["finish_reason"] = r["choices"][0].get("finish_reason")
        rec["signatures"] = signatures(t)
        if "tools" in p:
            rec["tool_calls"] = tool_check(r, p)
        rec["head_excerpt"] = t[: a.excerpt_chars]
        rec["tail_excerpt"] = t[-a.excerpt_chars :]
        rec["full_text"] = t
        report["traces"].append(rec)

    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)

    for tr in report["traces"]:
        s = tr.get("signatures")
        if not s:
            print(f"{tr['id']:26s} FAILED: {tr.get('error')}")
            continue
        print(
            f"{tr['id']:26s} {tr['regime']:8s} {s['words']:6d}w  "
            f"ttr={s['type_token_ratio']:.3f}  maxline={s['max_line_repeat']:3d}  "
            f"8gram={s['longest_ngram_loop']['count']:3d}  "
            f"tailrep={s['tail_repeat_fraction']:.3f}  "
            f"nonascii={s['nonascii_fraction']:.5f}  "
            f"finish={tr['finish_reason']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
