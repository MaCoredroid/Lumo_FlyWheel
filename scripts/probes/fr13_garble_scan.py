#!/usr/bin/env python3
"""fr13_garble_scan.py -- generation-degradation ("garble") detector for codex/qwen-code traces.

FR13 tree+cache campaign instrument (see FR13_TREECACHE_CAMPAIGN_20260704.md sec.57).
The tree+cache spec-decode runs occasionally *degenerate*: the model emits a coherent
reasoning preamble, then -- while producing an unconstrained tool-argument string (an
edit `file_path`, a `run_shell_command` `command`) or a verify `thinking` block --
derails into an off-task hallucinated document, a short-cycle repetition loop, or a
script-mixed (CJK) tail, and often collapses to an empty final answer. These episodes
are only visible at the sanctioned instrument: the codex_trace agent messages (the vLLM
docker log never contains generated tokens -- see memory feedback_garble_measure_codex_trace).

This scanner is a CHEAP, CONTENT-BASED catch instrument. It does NOT trust scalar metrics
(accept/TV/pass-rate are blind to small-rate per-token defects -- playbook class 12); it
reads the actual generated strings and flags the degradation signatures directly.

Detectors
  1. oversized_tool_arg   -- any tool_use arg string > --max-arg-bytes (default 8 KiB)
  2. short_field_bloat    -- a "should-be-short" field (file_path/path/pattern/glob/...) that
                             is huge (> --short-field-bytes, default 512 B). A real file path is
                             <200 B; a 22 KB `file_path` is unambiguous garble.
  3. oversized_block      -- a thinking/text block > --max-block-bytes (default 8 KiB)
  4. repetition_loop      -- a short token cycle (e.g. '2.1.2.1', '| | | |') repeated >= N times
  5. script_mix           -- CJK (or other non-latin) codepoint spike in a latin/code task
  6. offtask_document     -- prose/markdown/academic/package-manager/GitHub-issue document
                             markers appearing inside a tool arg or block (a python -c command
                             that contains '# 1. Introduction' or 'apt install' is derailed)
  7. empty_final_answer   -- the terminal result / last main-agent final text is empty/whitespace

It also reports per-turn generation-length statistics and decode-block-boundary crossing
flags: a turn generating G tokens crosses at least floor(G / block_size) mamba decode-block
boundaries regardless of prefill alignment (block_size default 1024). Crossing writes a
decode-side snapshot state into the prefix cache; whether that is the CAUSE or a CONSEQUENCE
of the garble is left to the caller (the garble onset offset per episode is reported so the
two can be told apart -- onset << first boundary => the runaway is not boundary-triggered).

Usage
  python3 scripts/probes/fr13_garble_scan.py TRACE [TRACE ...] [--json OUT.json]
  python3 scripts/probes/fr13_garble_scan.py 'output/fr13_live_gate/*/swe_out/**/agent_trace.jsonl' \
                                             'output/fr13_live_gate/*/swe_out/**/codex_trace.jsonl'

Exit code: 0 if every trace is CLEAN, 1 if any trace has >=1 garble episode (usable as a gate).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# ------------------------------------------------------------------ tokenizer (optional)
_TOK = None
_TOK_TRIED = False
DEFAULT_TOKENIZER = "/models/qwen3.6-27b-fp8/tokenizer.json"


def _load_tok(path: str):
    global _TOK, _TOK_TRIED
    if _TOK_TRIED:
        return _TOK
    _TOK_TRIED = True
    try:
        from tokenizers import Tokenizer  # type: ignore

        _TOK = Tokenizer.from_file(path)
    except Exception:
        _TOK = None
    return _TOK


def ntok(s: str, tok_path: str) -> int:
    """Token count; exact via the qwen tokenizer if available, else a chars/4 estimate."""
    if not s:
        return 0
    t = _load_tok(tok_path)
    if t is not None:
        return len(t.encode(s).ids)
    return max(1, round(len(s) / 4.0))


# ------------------------------------------------------------------ detector config / thresholds
# "should-be-short" tool-argument fields: a real value is a path/pattern/identifier, never a document.
SHORT_FIELDS = {
    "file_path", "path", "filename", "file", "pattern", "glob", "query",
    "absolute_path", "target_file", "directory", "dir", "url",
}

# off-task document markers: prose/markdown/academic/package-manager/issue-tracker fragments that
# have no business inside a shell command, a file path, or a code edit.
OFFTASK_PATTERNS = [
    (r"^\s*#{1,3}\s*\d", "markdown-numbered-heading"),
    (r"##\s*\d+(\.\d+)?\s+[A-Z]", "markdown-section"),
    (r"\brepository copy\b", "academic-repo-copy"),
    (r"\bWhite Rose\b", "academic-whiterose"),
    (r"\bLiterature Review\b", "academic-litreview"),
    (r"\bIntroduction\b.{0,40}\bobjectives\b", "academic-intro-objectives"),
    (r"\bapt(-get)? install\b", "pkg-manager-apt"),
    (r"\bpip install\b.{0,60}\bpython3-", "pkg-manager-mixed"),
    (r"username_\d", "github-issue-persona"),
    (r"@angular-devkit|node_modules/@", "hallucinated-js-stack"),
    (r"/etc/profile\.d", "shell-profile-doc"),
]
OFFTASK_RE = [(re.compile(p, re.IGNORECASE | re.MULTILINE), name) for p, name in OFFTASK_PATTERNS]

# Repetition loop. The 2026-07-06 fix: the naive r"(.{1,15})\1{6,}" FALSE-POSITIVES
# on normal INDENTED CODE (a repeated "\n        " newline+indent, or "\n    }" brace
# rows) — it flagged 2/3 replay samples that were on-task Python (fr13_garble_replay
# §65). Fix: the repeated unit must contain a NON-WHITESPACE, NON-STRUCTURAL char
# (i.e. not pure whitespace / braces / punctuation that legitimately repeats in code).
# Genuine garble cycles ("2.1.2.1", "| | |", "0000...") still match. Applied to a
# WHITESPACE-COLLAPSED view so indentation can't pad a cycle.
REPLOOP_RE = re.compile(r"(.{1,15}?)\1{6,}", re.DOTALL)
_REP_STRUCTURAL = set(" \t\n\r{}()[];,.:")

def _rep_is_genuine(unit: str) -> bool:
    # reject cycles whose unit is entirely whitespace / code-structural punctuation
    return any(ch not in _REP_STRUCTURAL for ch in unit) and \
        len(set(unit.strip())) >= 1 and unit.strip() != "" and \
        not all(ch in "{}()[];,. \t\n\r" for ch in unit)
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")


# ------------------------------------------------------------------ data model
@dataclass
class Episode:
    detector: str
    event: int
    turn: int
    level: str            # MAIN | SUB
    kind: str             # tool:<name> | thinking | text | final
    field: str            # arg name, if a tool arg
    bytes: int
    tokens: int
    signal: str           # human-readable trigger detail


@dataclass
class TurnStat:
    turn: int
    level: str
    event_first: int
    event_last: int
    gen_tokens: int
    crosses_boundary: bool          # gen_tokens >= block_size  => >=1 decode boundary GUARANTEED
    boundaries_min: int             # gen_tokens // block_size
    has_garble: bool = False


@dataclass
class TraceReport:
    trace: str
    verdict: str                    # CLEAN | GARBLE | UNREADABLE
    n_events: int
    n_turns: int
    n_episodes: int
    distinct_garble_events: list[int]
    total_gen_tokens: int
    max_turn_gen_tokens: int
    n_turns_over_block: int
    boundary_crossing_turns: list[int]
    episodes: list[dict]
    per_turn: list[dict] = field(default_factory=list)
    error: Optional[str] = None


# ------------------------------------------------------------------ trace parsing
def gen_text_of_msg(msg: dict) -> str:
    """Approx raw generated text of one assistant message (thinking + text + tool-call serialization)."""
    parts = []
    for c in msg.get("content", []) or []:
        ct = c.get("type")
        if ct == "text":
            parts.append(c.get("text", ""))
        elif ct == "thinking":
            parts.append(c.get("thinking", ""))
        elif ct == "tool_use":
            parts.append(json.dumps({"name": c.get("name"), "arguments": c.get("input", {})}, ensure_ascii=False))
    return "\n".join(parts)


def group_turns(events: list[dict]) -> list[dict]:
    """A turn = a maximal run of consecutive assistant events at the same parent_tool_use_id,
    bounded by any user/system/result event or a level change (one API completion)."""
    turns = []
    cur = None
    for idx, e in enumerate(events):
        if e.get("type") == "assistant":
            ptu = e.get("parent_tool_use_id")
            if cur is None or cur["ptu"] != ptu:
                if cur is not None:
                    turns.append(cur)
                cur = {"ptu": ptu, "events": [], "idx": []}
            cur["events"].append(e)
            cur["idx"].append(idx)
        else:
            if cur is not None:
                turns.append(cur)
                cur = None
    if cur is not None:
        turns.append(cur)
    return turns


# ------------------------------------------------------------------ per-block detection
def scan_block(detector_out: list[Episode], *, ev_idx: int, turn: int, level: str, kind: str,
               field_name: str, text: str, tok_path: str, args) -> bool:
    """Run the content detectors on one generated string. Returns True if any fired."""
    fired = False
    n = len(text.encode("utf-8", "ignore"))
    tks = ntok(text, tok_path)

    def add(det, signal):
        nonlocal fired
        fired = True
        detector_out.append(Episode(det, ev_idx, turn, level, kind, field_name, n, tks, signal))

    is_tool = kind.startswith("tool:")

    # 1 / 2 -- size
    if is_tool and field_name in SHORT_FIELDS and n > args.short_field_bytes:
        add("short_field_bloat", f"{field_name}={n}B (>{args.short_field_bytes}B; expected a short path/pattern)")
    elif is_tool and n > args.max_arg_bytes:
        add("oversized_tool_arg", f"{field_name}={n}B (>{args.max_arg_bytes}B)")
    elif (not is_tool) and n > args.max_block_bytes:
        add("oversized_block", f"{kind}={n}B (>{args.max_block_bytes}B)")

    # 4 -- repetition loop (whitespace-collapsed so indentation can't pad a cycle;
    # unit must be non-structural so normal indented code / brace rows don't fire)
    collapsed = re.sub(r"[ \t]+", " ", text)
    m = REPLOOP_RE.search(collapsed)
    if m:
        cyc = m.group(1)
        span = m.end() - m.start()
        if span >= 24 and _rep_is_genuine(cyc):
            add("repetition_loop", f"cycle {cyc!r} x~{span // max(1, len(cyc))} over {span}B")

    # 5 -- script-mix (CJK in a latin/code task)
    cjk = len(CJK_RE.findall(text))
    if cjk >= args.cjk_min and (cjk / max(1, len(text))) >= args.cjk_ratio:
        add("script_mix", f"{cjk} CJK codepoints ({100 * cjk / max(1, len(text)):.1f}% of block)")

    # 6 -- off-task document markers (only meaningful in a sizeable block/arg)
    if n >= args.offtask_min_bytes:
        hits = []
        for rgx, name in OFFTASK_RE:
            if rgx.search(text):
                hits.append(name)
        # a shell/path/edit field carrying document prose is the strongest signal
        strong = is_tool and (field_name in SHORT_FIELDS or field_name == "command")
        if hits and (len(hits) >= 2 or strong):
            add("offtask_document", f"markers={sorted(set(hits))}"
                                    + (f" inside tool arg '{field_name}'" if is_tool else ""))
    return fired


def scan_trace(path: str, args) -> TraceReport:
    try:
        with open(path) as f:
            raw = f.read().strip()
        obj = json.loads(raw)
        if isinstance(obj, dict):
            obj = obj.get("events") or obj.get("messages") or [obj]
        events = list(obj)
    except Exception as e:  # noqa
        return TraceReport(path, "UNREADABLE", 0, 0, 0, 0, 0, 0, [], [], error=repr(e))

    turns = group_turns(events)
    eps: list[Episode] = []

    # map event idx -> turn idx for garble<->turn correlation
    ev2turn = {}
    for ti, tn in enumerate(turns):
        for j in tn["idx"]:
            ev2turn[j] = ti

    # content detectors, per assistant block
    for idx, e in enumerate(events):
        if e.get("type") != "assistant":
            continue
        level = "MAIN" if e.get("parent_tool_use_id") is None else "SUB"
        turn = ev2turn.get(idx, -1)
        for c in e.get("message", {}).get("content", []) or []:
            ct = c.get("type")
            if ct == "thinking":
                scan_block(eps, ev_idx=idx, turn=turn, level=level, kind="thinking",
                           field_name="", text=c.get("thinking", ""), tok_path=args.tokenizer, args=args)
            elif ct == "text":
                scan_block(eps, ev_idx=idx, turn=turn, level=level, kind="text",
                           field_name="", text=c.get("text", ""), tok_path=args.tokenizer, args=args)
            elif ct == "tool_use":
                name = c.get("name")
                for k, v in (c.get("input", {}) or {}).items():
                    if isinstance(v, str):
                        scan_block(eps, ev_idx=idx, turn=turn, level=level, kind=f"tool:{name}",
                                   field_name=k, text=v, tok_path=args.tokenizer, args=args)

    # 7 -- empty final answer (terminal result / last main-agent final text)
    result_ev = next((e for e in reversed(events) if e.get("type") == "result"), None)
    final_text = ""
    if result_ev is not None:
        r = result_ev.get("result", "")
        final_text = r if isinstance(r, str) else json.dumps(r)
    else:
        last_a = next((e for e in reversed(events)
                       if e.get("type") == "assistant" and e.get("parent_tool_use_id") is None), None)
        if last_a:
            final_text = "".join(c.get("text", "") for c in last_a.get("message", {}).get("content", [])
                                 if c.get("type") == "text")
    if len(final_text.strip()) == 0:
        # only flag as garble if the trace also shows a completion (>1 turn) -- an empty
        # answer after real work is the collapse signature, not a legit no-op.
        if len(turns) > 1:
            eps.append(Episode("empty_final_answer", len(events) - 1, len(turns) - 1, "MAIN",
                               "final", "", 0, 0, "terminal answer is empty/whitespace"))

    # per-turn generation-length + boundary-crossing stats
    per_turn: list[TurnStat] = []
    garble_turns = {ep.turn for ep in eps}
    for ti, tn in enumerate(turns):
        g = sum(ntok(gen_text_of_msg(ev["message"]), args.tokenizer) for ev in tn["events"])
        ts = TurnStat(
            turn=ti,
            level="MAIN" if tn["ptu"] is None else "SUB",
            event_first=tn["idx"][0],
            event_last=tn["idx"][-1],
            gen_tokens=g,
            crosses_boundary=(g >= args.block_size),
            boundaries_min=g // args.block_size,
            has_garble=(ti in garble_turns),
        )
        per_turn.append(ts)

    total_gen = sum(t.gen_tokens for t in per_turn)
    max_gen = max((t.gen_tokens for t in per_turn), default=0)
    over = [t.turn for t in per_turn if t.crosses_boundary]
    # distinct garbled *generation* events (content detectors, excluding the structural empty-final)
    distinct_events = sorted({ep.event for ep in eps if ep.detector != "empty_final_answer"})
    # VERDICT (2026-07-06 fix): empty_final_answer ALONE is NOT garble — it is a
    # benign terminal state (model ends on a tool call with no closing message;
    # result subtype=success, is_error=False). A real GARBLE verdict requires a
    # CONTENT-degradation detector (oversized arg/block, repetition, script-mix,
    # off-task document). tcfix_i5 (relaunch): 21 clean turns, 0 boundary
    # crossings, only an empty-final => was mislabeled GARBLE. Empty-final is kept
    # as a reported detector hit (BENIGN_TERMINAL) but does not set the verdict.
    _content_eps = [ep for ep in eps if ep.detector != "empty_final_answer"]
    verdict = "GARBLE" if _content_eps else "CLEAN"
    return TraceReport(
        trace=path,
        verdict=verdict,
        n_events=len(events),
        n_turns=len(turns),
        n_episodes=len(eps),
        distinct_garble_events=distinct_events,
        total_gen_tokens=total_gen,
        max_turn_gen_tokens=max_gen,
        n_turns_over_block=len(over),
        boundary_crossing_turns=over,
        episodes=[asdict(ep) for ep in eps],
        per_turn=[asdict(t) for t in per_turn],
    )


# ------------------------------------------------------------------ reporting
def print_report(rep: TraceReport, args) -> None:
    print(f"\n=== {rep.trace}")
    if rep.error:
        print(f"    UNREADABLE: {rep.error}")
        return
    print(f"    verdict={rep.verdict}  events={rep.n_events} turns={rep.n_turns} "
          f"gen_tokens={rep.total_gen_tokens} max_turn={rep.max_turn_gen_tokens} "
          f"turns>=block({args.block_size})={rep.n_turns_over_block} {rep.boundary_crossing_turns}")
    if rep.episodes:
        print(f"    {len(rep.distinct_garble_events)} distinct garbled event(s) {rep.distinct_garble_events}"
              f" + {rep.n_episodes} detector hit(s):")
        for ep in rep.episodes:
            print(f"      ev#{ep['event']:<4} turn#{ep['turn']:<3} {ep['level']:4} {ep['kind']:>22} "
                  f"[{ep['detector']}] {ep['signal']}")
    if args.verbose:
        print("    boundary-crossing turns (gen_tokens >= block_size => >=1 decode snapshot write):")
        for t in rep.per_turn:
            if t["crosses_boundary"] or t["has_garble"]:
                tag = " GARBLE" if t["has_garble"] else ""
                print(f"      turn#{t['turn']:<3} {t['level']:4} ev{t['event_first']}-{t['event_last']} "
                      f"gen_tokens={t['gen_tokens']} boundaries_min={t['boundaries_min']}{tag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FR13 garble / generation-degradation scanner for codex traces.")
    ap.add_argument("traces", nargs="+",
                    help="agent_trace.jsonl / codex_trace.jsonl path(s); globs allowed "
                         "(the emitter still writes codex_trace.jsonl for backward-compat)")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="qwen tokenizer.json (falls back to chars/4 if unavailable)")
    ap.add_argument("--block-size", type=int, default=1024, help="mamba decode block size (default 1024)")
    ap.add_argument("--max-arg-bytes", type=int, default=8192)
    ap.add_argument("--max-block-bytes", type=int, default=8192)
    ap.add_argument("--short-field-bytes", type=int, default=512)
    ap.add_argument("--offtask-min-bytes", type=int, default=400)
    ap.add_argument("--cjk-min", type=int, default=20, help="min CJK codepoints to flag script_mix")
    ap.add_argument("--cjk-ratio", type=float, default=0.01, help="min CJK fraction of block to flag script_mix")
    ap.add_argument("--json", default=None, help="write the machine verdict here (else stdout)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    paths: list[str] = []
    for pat in args.traces:
        hits = sorted(glob.glob(pat, recursive=True))
        paths.extend(hits if hits else [pat])

    reports = [scan_trace(p, args) for p in paths]
    for rep in reports:
        print_report(rep, args)

    verdict = {
        "schema": "fr13.garble_scan.v1",
        "block_size": args.block_size,
        "tokenizer_exact": _load_tok(args.tokenizer) is not None,
        "n_traces": len(reports),
        "n_garble_traces": sum(1 for r in reports if r.verdict == "GARBLE"),
        "traces": [
            {k: v for k, v in asdict(r).items() if k != "per_turn" or args.verbose}
            for r in reports
        ],
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(verdict, f, indent=2)
        print(f"\n[json verdict -> {args.json}]")
    else:
        print("\n" + json.dumps({k: verdict[k] for k in
              ("schema", "block_size", "tokenizer_exact", "n_traces", "n_garble_traces")}, indent=2))

    any_garble = any(r.verdict == "GARBLE" for r in reports)
    return 1 if any_garble else 0


if __name__ == "__main__":
    sys.exit(main())
