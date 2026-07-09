#!/usr/bin/env python3
"""FR13 garble-watch: extract JUST the garble evidence from an agent session trace
and (optionally) diff a tree-arm task against native on the SAME instance.

User steer (2026-07-09): "watch the agent session; if garble happens we don't need the
whole session; compare its behavior with previous native traces." The garble = tree
spec-decode commits a mis-spelled NEAR-NEIGHBOR identifier in a tool-call parameter
(dropped underscore, truncation, camelCase, wrong-word) -> NameError / no-such-file /
command-not-found / a typo-loop of near-identical failing calls -> give-up.

This does NOT dump the session. It emits, per task:
  - the tool-call sequence (tool name + the salient path/command/param), compacted
  - every tool_result that carries an ERROR, with the error line
  - GARBLE FLAGS: undefined-name / not-found errors; typo-loops (consecutive
    near-duplicate tool inputs, edit-distance small); edit-distance-1 identifier drift
    between a used name and a previously-established name
  - verdict (from eval/predictions.jsonl if present: empty patch = give-up)

Trace formats handled (auto-detected, tolerant of truncation):
  (A) stream-json NDJSON  -> one JSON object per line (the NEW qwen_trace.jsonl)
  (B) single JSON array   -> legacy buffered `qwen --output-format json` (may be
                             truncated mid-string at 64KB; we salvage decodable prefix)
  (C) codex --json JSONL  -> legacy codex_trace.jsonl (item.command_execution/agent_message)

Usage:
  # one trace:
  python3 fr13_garble_watch.py trace <path/to/qwen_trace.jsonl>
  # a whole arm dir (globs per_task/<id>/{qwen,codex}_trace*.jsonl):
  python3 fr13_garble_watch.py arm <arm_dir>
  # diff a tree arm vs a native arm on shared instance_ids:
  python3 fr13_garble_watch.py compare --tree <tree_arm_dir> --native <native_arm_dir>
"""
import sys, os, json, glob, argparse, re, difflib

# ---------------------------------------------------------------- trace loading
def _iter_records(path):
    """Yield decoded JSON objects from a trace file, tolerant of truncation.
    Detects NDJSON vs single-array vs partial. Never raises on a bad tail."""
    try:
        raw = open(path, "r", errors="replace").read()
    except OSError:
        return
    s = raw.lstrip()
    if not s:
        return
    if s[0] == "[":
        # single JSON array (legacy buffered json; possibly truncated mid-string)
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                yield from (o for o in arr if isinstance(o, dict))
                return
        except json.JSONDecodeError:
            # salvage: decode successive objects with raw_decode from inside the array
            yield from _salvage_array(s)
            return
    # NDJSON / JSONL: one object per line
    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if not line or line in ("[", "]"):
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict):
            yield o

def _salvage_array(s):
    """Decode as many top-level objects as possible from a truncated `[{...},{...},..`."""
    dec = json.JSONDecoder()
    i = s.find("{")
    while i != -1 and i < len(s):
        try:
            obj, end = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            yield obj
        # advance to next '{'
        nxt = s.find("{", end)
        if nxt == -1:
            break
        i = nxt

# ---------------------------------------------------------- event normalization
def _norm_events(records):
    """Normalize heterogeneous trace records to a flat list of:
       {"role": "assistant"|"tool"|"result"|"other", "tools":[{name,input}],
        "errors":[str], "text":str, "raw_type":str}
    Handles stream-json (Anthropic Messages shape) AND codex item.* shape."""
    out = []
    for r in records:
        rt = r.get("type", "")
        # --- codex shape: {"type":"item.completed","item":{"type":"command_execution"|"agent_message",...}}
        if "item" in r and isinstance(r["item"], dict):
            it = r["item"]; itt = it.get("type", "")
            if itt == "command_execution":
                cmd = it.get("command") or it.get("cmd") or ""
                err = ""
                res = it.get("result") or it.get("output") or it.get("aggregated_output") or ""
                if isinstance(res, dict): res = json.dumps(res)[:400]
                if _is_error_text(str(res)): err = str(res)
                out.append({"role": "tool", "tools": [{"name": "shell", "input": {"command": cmd}}],
                            "errors": [err] if err else [], "text": "", "raw_type": itt})
            elif itt == "agent_message":
                out.append({"role": "assistant", "tools": [], "errors": [],
                            "text": it.get("text", "") or "", "raw_type": itt})
            continue
        # --- stream-json / Anthropic Messages shape
        msg = r.get("message") if isinstance(r.get("message"), dict) else None
        content = None
        if msg is not None:
            content = msg.get("content")
        elif isinstance(r.get("content"), (list, str)):
            content = r.get("content")
        role = (msg or r).get("role", "")
        if rt == "result" or role == "result":
            txt = r.get("result") or r.get("subtype") or ""
            out.append({"role": "result", "tools": [], "errors": [], "text": str(txt), "raw_type": rt})
            continue
        tools, errors, texts = [], [], []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for blk in content:
                if not isinstance(blk, dict): continue
                bt = blk.get("type", "")
                if bt == "tool_use":
                    tools.append({"name": blk.get("name", "?"), "input": blk.get("input", {}) or {}})
                elif bt == "tool_result":
                    c = blk.get("content", "")
                    if isinstance(c, list):
                        c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
                    c = str(c)
                    if blk.get("is_error") or _is_error_text(c):
                        errors.append(c[:500])
                elif bt == "text":
                    texts.append(blk.get("text", ""))
        if tools or errors or texts:
            out.append({"role": role or ("assistant" if tools or texts else "tool"),
                        "tools": tools, "errors": errors, "text": " ".join(texts), "raw_type": rt})
    return out

# ------------------------------------------------------------- garble detection
# The garble SYMPTOM in a trace = a tool-call parameter names an identifier/file/command
# that does not exist, because the tree wrongly accepted a mis-spelled near-neighbor.
# We anchor on the ERROR (the deterministic downstream fact), extract the offending
# NAME, and — the smoking gun — test whether that name is a near-neighbor (edit-dist
# <=2) of an identifier that DID appear successfully earlier in the session. Broad
# command-similarity is NOT used (normal exploration produces near-dup shell commands).
# STRONG error signals only — phrases that do NOT occur in normal source dumps / file
# listings (so a `sed`/`cat` of a file that merely CONTAINS the word "ImportError"
# won't false-positive). A bare keyword like "AttributeError" appearing in code is
# ignored; we require the runtime phrasing.
_ERR_PAT = re.compile(
    r"is not defined|command not found|No such file or directory|"
    r"No module named|has no attribute|Traceback \(most recent call last\)|"
    r"cannot find|does not exist|not recognized as",
    re.I)
# universal environment gaps present in the image REGARDLESS of decode backend —
# they hit tree AND native identically, so they are never a garble signal.
_ENV_DENY = {"rg", "ripgrep", "fd", "bat", "erfa", "pyerfa", "ag"}
# capture the offending NAME from the common error shapes
_NAME_FROM_ERR = [
    re.compile(r"name '([^']+)' is not defined"),
    re.compile(r"No module named '?([A-Za-z0-9_.]+)'?"),
    re.compile(r"module '[^']+' has no attribute '([^']+)'"),
    re.compile(r"has no attribute '([^']+)'"),
    re.compile(r"No such file or directory:?\s*'?([^'\n]+)'?"),
    re.compile(r"([A-Za-z0-9_./-]+): command not found"),
    re.compile(r"cannot find '?([A-Za-z0-9_./-]+)'?"),
]
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{4,}")

def _is_error_text(t):
    return bool(_ERR_PAT.search(t or ""))

def _salient(tool):
    """Compact one tool call's salient argument for display + identifier mining."""
    inp = tool.get("input", {}) or {}
    for k in ("command", "file_path", "path", "filePath", "old_string", "pattern",
              "query", "content", "new_string"):
        if k in inp and isinstance(inp[k], str) and inp[k].strip():
            return f"{k}={inp[k][:200]}"
    return json.dumps(inp)[:200]

def _extract_name(err):
    for pat in _NAME_FROM_ERR:
        m = pat.search(err)
        if m:
            return m.group(1).strip().split("/")[-1]
    return None

def analyze_trace(path):
    events = _norm_events(_iter_records(path))
    tool_calls, errors, flags = [], [], []
    seen_idents = set()          # identifiers ESTABLISHED anywhere in the session:
                                 # agent tool inputs AND tool results (source it read).
                                 # Mining results is what lets a garbled 'from_geodentic'
                                 # match the correct 'from_geodetic' the agent read from src.
    err_classes = []
    # first pass: build the established-identifier pool from BOTH inputs and results
    for ev in events:
        for t in ev.get("tools", []):
            for m in _IDENT.findall(_salient(t)):
                seen_idents.add(m)
        # mine result/text bodies too (bounded), keeping only api-ish names
        body = " ".join(ev.get("errors", []) + [ev.get("text", "")])[:20000]
        for m in _IDENT.findall(body):
            if ("_" in m or not m.islower()) and 6 <= len(m) <= 40:
                seen_idents.add(m)
    for ev in events:
        for t in ev.get("tools", []):
            sal = _salient(t)
            tool_calls.append((t["name"], sal))
        for e in ev.get("errors", []):
            if not e.strip():
                continue
            line = e.strip().splitlines()[0][:200]
            errors.append(line)
            m = _ERR_PAT.search(e)
            if not m:
                continue
            cls = m.group(0)
            nm = _extract_name(e)
            if nm and nm.lower() in _ENV_DENY:
                continue  # universal env gap (rg/erfa/...) — identical in both arms
            err_classes.append(cls.lower())
            if nm:
                # SMOKING GUN: the undefined name is a near-neighbor of an established id
                near = [k for k in seen_idents
                        if k != nm and _editle2(nm, k)]
                if near:
                    flags.append(f"NEAR-NEIGHBOR GARBLE: undefined '{nm}' ~ established "
                                 f"{near[:3]}  [{cls}]  ::  {line}")
                else:
                    flags.append(f"UNDEFINED '{nm}' [{cls}] :: {line}")
            else:
                flags.append(f"ERROR[{cls}] :: {line}")
    # error-loop: the same error class hit >=3x = a stuck typo/retry loop
    from collections import Counter
    loop = [(c, n) for c, n in Counter(err_classes).items() if n >= 3]
    for c, n in loop:
        flags.append(f"ERROR-LOOP: '{c}' repeated {n}x (stuck)")
    # garble verdict (STRONG): a near-neighbor identifier hit or a stuck error-loop.
    # plain UNDEFINED / generic ERROR are shown as context but do NOT alone = garble
    # (they are dominated by env noise that the tree-vs-native COMPARE cancels).
    strong = any(f.startswith(("NEAR-NEIGHBOR", "ERROR-LOOP")) for f in flags)
    return {"path": path, "n_events": len(events), "n_tools": len(tool_calls),
            "tool_calls": tool_calls, "errors": errors,
            "flags": _dedup(flags), "garble": strong,
            "err_classes": dict(Counter(err_classes))}

def _editle2(a, b, maxd=2):
    """True if Levenshtein(a,b) in [1, maxd] — a near-neighbor, not identical.
    This is the near-neighbor garble test (dropped underscore, truncation, one-char
    swap, camelCase slip are all edit-dist 1-2)."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > maxd:
        return False
    # classic DP, bounded
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        best = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            best = min(best, cur[j])
        if best > maxd:
            return False
        prev = cur
    return 1 <= prev[lb] <= maxd

def _dedup(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# --------------------------------------------------------------- verdict lookup
def _verdict(task_dir):
    pj = os.path.join(task_dir, "eval", "predictions.jsonl")
    if not os.path.isfile(pj):
        return "no-eval"
    try:
        rows = [json.loads(l) for l in open(pj) if l.strip()]
    except Exception:
        return "eval-parse-err"
    if not rows:
        return "give-up(empty-preds)"
    patch = rows[-1].get("model_patch") or rows[-1].get("prediction") or ""
    return "give-up(empty-patch)" if not str(patch).strip() else f"patch({len(str(patch))}B)"

def _find_trace(task_dir):
    for pat in ("qwen_trace.jsonl", "qwen_trace*.jsonl", "agent_trace*.jsonl", "codex_trace*.jsonl"):
        g = sorted(glob.glob(os.path.join(task_dir, pat)))
        if g: return g[0]
    return None

# ------------------------------------------------------------------- reporting
def _print_trace(a, indent=""):
    tag = "🔴 GARBLE" if a["garble"] else "🟢 clean"
    print(f"{indent}{tag}  events={a['n_events']} tools={a['n_tools']}  {a['path']}")
    for f in a["flags"]:
        print(f"{indent}    ! {f}")
    if not a["flags"] and a["errors"]:
        for e in a["errors"][:3]:
            print(f"{indent}    · err: {e}")

def cmd_trace(args):
    a = analyze_trace(args.path)
    _print_trace(a)

def cmd_arm(args):
    tds = sorted(glob.glob(os.path.join(args.arm_dir, "**", "per_task", "*"), recursive=True))
    tds = [d for d in tds if os.path.isdir(d)]
    if not tds:
        print(f"(no per_task dirs under {args.arm_dir})"); return
    ng = 0
    for td in tds:
        iid = os.path.basename(td)
        tr = _find_trace(td)
        v = _verdict(td)
        if not tr:
            print(f"  ?? {iid:32s} verdict={v}  (no trace yet)"); continue
        a = analyze_trace(tr)
        ng += a["garble"]
        print(f"  {'🔴' if a['garble'] else '🟢'} {iid:32s} verdict={v:22s} tools={a['n_tools']} flags={len(a['flags'])}")
        for f in a["flags"][:6]:
            print(f"        ! {f}")
    print(f"  --- {ng}/{len(tds)} tasks show a garble signature ---")

def cmd_compare(args):
    def by_iid(arm):
        m = {}
        for td in glob.glob(os.path.join(arm, "**", "per_task", "*"), recursive=True):
            if os.path.isdir(td):
                m[os.path.basename(td)] = td
        return m
    tree, nat = by_iid(args.tree), by_iid(args.native)
    shared = sorted(set(tree) & set(nat))
    if not shared:
        print(f"(no shared instance_ids: tree={len(tree)} native={len(nat)})"); return
    for iid in shared:
        print(f"\n=== {iid} ===")
        for label, td in (("TREE ", tree[iid]), ("NATIVE", nat[iid])):
            tr = _find_trace(td); v = _verdict(td)
            if not tr:
                print(f"  {label}: verdict={v} (no trace)"); continue
            a = analyze_trace(tr)
            print(f"  {label}: verdict={v}  {'🔴GARBLE' if a['garble'] else '🟢clean'} "
                  f"tools={a['n_tools']} flags={len(a['flags'])}")
            for f in a["flags"][:6]:
                print(f"        ! {f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("trace"); p.add_argument("path")
    p = sub.add_parser("arm"); p.add_argument("arm_dir")
    p = sub.add_parser("compare"); p.add_argument("--tree", required=True); p.add_argument("--native", required=True)
    args = ap.parse_args()
    {"trace": cmd_trace, "arm": cmd_arm, "compare": cmd_compare}[args.cmd](args)
