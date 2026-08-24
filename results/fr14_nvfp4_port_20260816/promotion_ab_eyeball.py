#!/usr/bin/env python3
"""FR14 promotion A/B: render the REAL SWE agent generations for the degeneration
eyeball, and compute the mechanical signatures BESIDE them -- never instead.

Same doctrine as fr14_lane5a_eyeball.py: the statistics tell the reader WHERE to
look in a 60,000-word multi-turn trajectory; they never issue the verdict. The
verdict lives in the campaign note, written by whoever read the excerpts.

Input is the per-task agent trace the SWE runner writes,
``swe_out/verified/per_task/<instance>/qwen_trace.jsonl`` -- the actual served
token stream as the agent consumed it, including its tool calls. This is the
only place the campaign's generations survive: the offload proxy runs with
``raw_dumps_disabled=1``, so there is no raw-response dump to read instead.

Signatures (all named by the campaign as decode-degeneration modes):
  type_token_ratio      -- unique words / words; a repetition loop collapses it
  max_line_repeat       -- most times any single non-empty line repeats verbatim
  longest_ngram_loop    -- most-repeated 8-gram and its count: the direct
                           fingerprint of a decode loop
  tail_repeat_fraction  -- duplicate fraction of the LAST 25% of lines; loops
                           start late and a whole-trace average dilutes them
  mid_word_break_candidates -- "inva riant": the single-flipped-token artefact
  nonascii_fraction     -- CJK/symbol bursts in an English trace
  unbalanced_delims     -- brace/bracket/paren/backtick imbalance in code output
  tool_calls            -- every tool call, whether its arguments parse as JSON,
                           and the malformed ones verbatim
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def signatures(t: str) -> dict:
    words = t.split()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    line_counts = Counter(lines)

    grams = Counter(tuple(words[i : i + 8]) for i in range(max(0, len(words) - 7)))
    top_gram, top_gram_n = ("", 0)
    if grams:
        g, n = grams.most_common(1)[0]
        top_gram, top_gram_n = " ".join(g), n

    cut = int(len(lines) * 0.75)
    tail = lines[cut:]
    seen_before = set(lines[:cut])
    tail_rep = sum(1 for ln in tail if ln in seen_before) / max(1, len(tail))

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
            line_counts.most_common(1)[0][0][:160] if line_counts else ""
        ),
        "longest_ngram_loop": {"ngram": top_gram[:200], "count": top_gram_n},
        "tail_repeat_fraction": tail_rep,
        "mid_word_break_candidates": mid_word_breaks,
        "nonascii_chars": nonascii,
        "nonascii_fraction": nonascii / max(1, len(t)),
        "unbalanced_delims": delims,
    }


# Separates DEGENERATION from VACUITY. Both show no visible text and no tools; they
# differ by how much the model generated. Measured over the 10 ambiguous cases in the
# bank (visible<200 and tools==0): the degeneration carries 70755 thinking chars, the
# next-highest is 204. Any threshold in that range works, so this is a chasm rather
# than a tuned constant -- set at the round number nearest the low side.
_DEGEN_THINK_FLOOR = 2000
_DEGEN_VISIBLE_CEIL = 200


def _degeneration_flag(rec: dict, vis: dict) -> str:
    """The conjunction, not any single repetition statistic.

    Validated over all 125 banked traces: ZERO false positives, one true positive
    (Cqc16/13236). Its limit is stated honestly because negative validation cannot
    manufacture positives -- ONE positive example. A degeneration that kept narrating,
    or that called tools while looping, would pass all three terms and this would miss
    it. Treat as a floor, and keep c5 as the independent second opinion: c5 keys on the
    decode seam rather than on the prose, and it flagged 13236 at 0.3499 while every
    other task in the canonical sixteen sat inside [0.40, 0.70].
    """
    if (vis["chars"] <= _DEGEN_VISIBLE_CEIL
            and rec["n_tool_calls"] == 0
            and rec.get("thinking_chars", 0) >= _DEGEN_THINK_FLOOR):
        return ("   <<< DEGENERATION SUSPECTED (no visible output, no tools, "
                f"{rec.get('thinking_chars', 0)} thinking chars) -- confirm with c5")
    if vis["chars"] <= _DEGEN_VISIBLE_CEIL and rec["n_tool_calls"] == 0:
        return "   <<< VACUOUS (no visible output, no tools, little generation)"
    return ""


def harvest(trace_path: Path) -> dict:
    """Pull every assistant text block and every tool call out of the trace."""
    texts: list[str] = []
    thinking: list[str] = []
    tool_calls: list[dict] = []
    turns = 0
    stop_reasons: Counter = Counter()
    for raw in trace_path.read_text(errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        if not isinstance(rec, dict) or rec.get("type") != "assistant":
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        turns += 1
        if msg.get("stop_reason") is not None:
            stop_reasons[str(msg.get("stop_reason"))] += 1
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                texts.append(str(block.get("text") or ""))
            elif kind == "thinking":
                thinking.append(str(block.get("thinking") or ""))
            elif kind == "tool_use":
                arg = block.get("input")
                rec_call = {
                    "name": block.get("name"),
                    "input_is_object": isinstance(arg, dict),
                    "input_repr": json.dumps(arg, ensure_ascii=False)[:600]
                    if arg is not None
                    else None,
                }
                # The runner has already parsed the wire arguments into
                # ``input``; a tool call the parser could not decode arrives as
                # a string (or is missing), which is the malformation signature.
                if isinstance(arg, str):
                    try:
                        json.loads(arg)
                        rec_call["json_parses"] = True
                    except Exception as exc:
                        rec_call["json_parses"] = False
                        rec_call["parse_error"] = f"{type(exc).__name__}: {exc}"
                        rec_call["raw"] = arg[:600]
                tool_calls.append(rec_call)
    return {
        "assistant_turns": turns,
        "stop_reasons": dict(stop_reasons),
        "texts": texts,
        "thinking": thinking,
        "tool_calls": tool_calls,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True, help="…/<arm>/swe_out/verified")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--excerpt-chars", type=int, default=2000)
    a = ap.parse_args()

    root = Path(a.arm_dir) / "per_task"
    report = {
        "schema": "fr14.promotion_ab.eyeball.v1",
        "label": a.label,
        "arm_dir": str(a.arm_dir),
        "tasks": [],
    }
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        trace = task_dir / "qwen_trace.jsonl"
        rec: dict = {"instance_id": task_dir.name, "trace": str(trace)}
        if not trace.is_file():
            rec["MISSING_TRACE"] = True
            report["tasks"].append(rec)
            continue
        h = harvest(trace)
        visible = "\n".join(h["texts"])
        reasoning = "\n".join(h["thinking"])
        both = (reasoning + "\n" + visible) if reasoning else visible
        rec["assistant_turns"] = h["assistant_turns"]
        rec["stop_reasons"] = h["stop_reasons"]
        rec["n_tool_calls"] = len(h["tool_calls"])
        rec["malformed_tool_calls"] = [
            c for c in h["tool_calls"] if c.get("json_parses") is False
        ]
        rec["tool_name_counts"] = dict(
            Counter(str(c.get("name")) for c in h["tool_calls"])
        )
        rec["signatures_visible"] = signatures(visible)
        rec["signatures_all"] = signatures(both)
        rec["thinking_chars"] = len(reasoning)
        rec["head_excerpt"] = both[: a.excerpt_chars]
        rec["tail_excerpt"] = both[-a.excerpt_chars :]
        # The three longest single text blocks: a decode loop that ran until the
        # cap is longest, so this is where to look first.
        longest = sorted(h["texts"], key=len, reverse=True)[:3]
        rec["longest_blocks"] = [b[: a.excerpt_chars] for b in longest]
        patch = task_dir / "patch.diff"
        rec["patch_bytes"] = patch.stat().st_size if patch.is_file() else 0
        rec["patch"] = patch.read_text(errors="replace")[:4000] if patch.is_file() else ""
        report["tasks"].append(rec)

    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    print(f"== {a.label} ==")
    for t in report["tasks"]:
        s = t.get("signatures_all")
        v = t.get("signatures_visible")
        if not s:
            print(f"{t['instance_id']:26s} MISSING TRACE")
            continue
        # WHICH SERIES THE HEADLINE SHOWS, and why it changed on 2026-08-24.
        # It used to lead with signatures_ALL. That series mixes the model's output
        # with TOOL OUTPUT, and measured against the one known degeneration it is not
        # merely noisy, it is INVERTED: 13236 (degenerate) scores all-ttr 0.729, the
        # HIGHEST in the bank, because a degenerate trace pulls in almost no tool text
        # to dilute it, while healthy tasks sit at 0.17-0.35. All-tailrep does not
        # separate either -- degenerate 13236 is 0.538, healthy 14096 is 0.570. Two
        # healthy tasks were investigated as suspected degenerations on those numbers.
        # So the headline now leads with the VISIBLE series and the conjunction below.
        flag = _degeneration_flag(t, v)
        print(
            f"{t['instance_id']:26s} turns={t['assistant_turns']:4d} "
            f"tools={t['n_tool_calls']:4d} malformed={len(t['malformed_tool_calls'])} "
            f"visChars={v['chars']:6d} visTtr={v['type_token_ratio']:.3f} "
            f"visTailrep={v['tail_repeat_fraction']:.3f} "
            f"think={t.get('thinking_chars', 0):7d} "
            f"patch={t['patch_bytes']}B{flag}"
        )
        # all-text series retained on a second line: still the right place to see how
        # much transcript a task pulled in, just not a degeneration signal.
        print(
            f"{'':26s}   [all-text] {s['words']:7d}w ttr={s['type_token_ratio']:.3f} "
            f"maxline={s['max_line_repeat']:4d} 8gram={s['longest_ngram_loop']['count']:4d} "
            f"tailrep={s['tail_repeat_fraction']:.3f} "
            f"nonascii={s['nonascii_fraction']:.6f} "
            f"midword={s['mid_word_break_candidates']:5d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
