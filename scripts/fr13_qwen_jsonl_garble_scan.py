#!/usr/bin/env python3
"""Scan qwen-code proxy request-dump JSONLs (chatreq_*.json) for GARBLE in the
served (model-generated) content, with LOW false-positive noise.

The served code lives in assistant tool_call arguments. Garble (tree spec-decode
commits a wrong/near-neighbor identifier) corrupts that code -> undefined names /
syntax errors / malformed tool-call JSON.

NOISE CONTROL (v2): only SELF-CONTAINED code is syntax/undef-scored:
  - run_shell_command `python -c <script>`  (extracted via shlex, not a fragile regex)
  - assistant ```python fenced code blocks
  - write_file / create_file full-file contents
These SHOULD parse and (with their imports) have no undefined names, so a syntax
error / undefined name here is a real anomaly. EDIT `new_string` is an inherently
PARTIAL fragment (mid-function, references outside identifiers) -> it is COUNTED but
NOT syntax/undef-scored (that was 35% of the old false positives). `malformed_toolargs`
(a tool-call whose JSON args don't parse) is an absolute, unambiguous garble signal.

The reliable verdict is the tree(cat8/cat6)-vs-native RATE on the SAME 16 tasks
(native carries the identical self-contained-code baseline).
"""
import json, os, sys, glob, ast, builtins, re, argparse, shlex

BI = set(dir(builtins)) | {
    "self", "cls", "np", "torch", "os", "sys", "re", "math", "pd", "plt",
    "json", "time", "Path", "pytest", "__file__", "__name__", "true", "false", "null",
    "kwargs", "args",  # near-universal idioms, never garble
}


def _bound_names(tr):
    """All names BOUND by this AST (Store targets, imports, def/class, args, except-as)."""
    S = set()
    for node in ast.walk(tr):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            S.add(node.id)
        elif isinstance(node, ast.alias):
            S.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            S.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            S.update(node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            S.add(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            aa = node.args
            for a in aa.args + aa.posonlyargs + aa.kwonlyargs:
                S.add(a.arg)
            if aa.vararg:
                S.add(aa.vararg.arg)
            if aa.kwarg:
                S.add(aa.kwarg.arg)
    return S


def _py_from_shell(cmd):
    """Extract the python -c SCRIPT from a shell command via shlex (quoting-safe)."""
    try:
        toks = shlex.split(cmd)
    except Exception:
        return None
    for i, t in enumerate(toks):
        if re.fullmatch(r"python[0-9.]*", os.path.basename(t)):
            if "-c" in toks[i:]:
                j = toks.index("-c", i)
                if j + 1 < len(toks):
                    return toks[j + 1]
    return None


def extract(chatreq_path):
    """Yield (kind, code). kind in {script, fragment, malformed}. Only `script` is scored."""
    try:
        r = json.load(open(chatreq_path))
    except Exception:
        return
    for m in r.get("messages", []):
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str):
            for blk in re.findall(r"```(?:python)?\n(.*?)```", c, re.S):
                yield ("script", blk)
        for t in (m.get("tool_calls") or []):
            fn = t.get("function") or {}
            name = fn.get("name", "")
            try:
                a = json.loads(fn.get("arguments", ""))
            except Exception:
                yield ("malformed", fn.get("arguments", ""))
                continue
            if name in ("write_file", "create_file") and isinstance(a.get("content"), str):
                if a.get("file_path", "").endswith(".py"):
                    yield ("script", a["content"])
            elif name == "edit" and isinstance(a.get("new_string"), str):
                yield ("fragment", a["new_string"])          # count-only, NOT scored
            elif name == "run_shell_command" and isinstance(a.get("command"), str):
                py = _py_from_shell(a["command"])
                if py:
                    yield ("script", py)


_PY_MARKERS = re.compile(
    r"(^|\n)\s*(import |from |def |class |return\b|if |elif |else:|for |while |"
    r"with |try:|except|finally:|assert |raise |yield |print\s*\(|@\w|async |await )"
    r"|[^=!<>+\-*/%&|^:]=[^=]"        # a plain assignment
    r"|\blambda\b|\bself\.|\bprint\b"
)


def _looks_python(code):
    """True if the content has real Python structure. Filters out NON-Python content the
    agent sometimes quotes (PLY/BNF grammar rules, config/data, plain text) that fails
    ast.parse but is NOT garble. Garbled Python still has import/def/=/return markers, so
    this hides non-Python false positives WITHOUT hiding garble."""
    return bool(_PY_MARKERS.search(code))


def undefined(code, gdefined=frozenset()):
    """None => syntax error; else Load names not bound HERE, not builtins, and not bound
    ANYWHERE in the arm (gdefined). A leftover = a name never defined in the whole session
    = garble-suspect (a corrupted identifier). Real cross-fragment names (imported/assigned
    in some other script) live in gdefined and drop out => native-floor noise (kwargs/AltAz)
    is excluded, leaving only genuinely-never-defined corrupted identifiers."""
    try:
        tr = ast.parse(code)
    except Exception:
        return None
    L = {x.id for x in ast.walk(tr) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
    return L - _bound_names(tr) - BI - gdefined


def scan(d):
    root = os.path.join(d, "proxy_request_dumps")
    root = root if os.path.isdir(root) else d
    files = glob.glob(os.path.join(root, "**", "chatreq_*.json"), recursive=True)
    seen = set()
    scripts = []
    n_frag = n_malformed = 0
    gdefined = set()               # pass 1: every name bound ANYWHERE in the arm's served code
    for f in files:
        for kind, code in extract(f):
            key = (kind, code)
            if key in seen:
                continue
            seen.add(key)
            if kind == "malformed":
                n_malformed += 1
                continue
            if kind == "fragment":
                n_frag += 1
            else:
                scripts.append(code)
            try:                    # fragments rarely parse; scripts usually do — best-effort
                gdefined |= _bound_names(ast.parse(code))
            except Exception:
                pass
    n_script = n_syntax_bad = n_undef = n_nonpy = n_edge = 0    # pass 2: score scripts
    undef_ex = []
    for code in scripts:
        if not _looks_python(code):
            n_nonpy += 1                 # PLY grammar / config / text the agent quoted — not garble
            continue
        n_script += 1
        try:
            tr = ast.parse(code)
        except SyntaxError as e:
            # f-string parse errors from shlex-extracted `python -c` are nested-quote/brace
            # EXTRACTION artifacts, not garble (garble corrupts identifiers -> undef). Class
            # them as edges, not syntax_bad.
            if "f-string" in (e.msg or "").lower():
                n_edge += 1
            else:
                n_syntax_bad += 1
            continue
        L = {x.id for x in ast.walk(tr) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
        u = L - _bound_names(tr) - BI - gdefined
        if u:
            n_undef += 1
            if len(undef_ex) < 12:
                undef_ex.append(sorted(u)[:6])
    return dict(files=len(files), scripts=n_script, syntax_bad=n_syntax_bad, undef=n_undef,
                fragments=n_frag, malformed=n_malformed, nonpython=n_nonpy, fstring_edges=n_edge,
                undef_ex=undef_ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    for d in a.dirs:
        s = scan(d)
        print(d)
        sc = s["scripts"] or 1
        print(f"  chatreq={s['files']}  self_contained_scripts={s['scripts']}  "
              f"syntax_bad={s['syntax_bad']} ({100*s['syntax_bad']/sc:.1f}%)  "
              f"undef={s['undef']} ({100*s['undef']/sc:.1f}%)  "
              f"malformed_toolargs={s['malformed']}  edit_fragments={s['fragments']}(unscored)  nonpython={s.get('nonpython',0)}(skipped)  fstring_edges={s.get('fstring_edges',0)}")
        if s["undef_ex"]:
            print(f"  undef_examples: {s['undef_ex'][:6]}")


if __name__ == "__main__":
    main()
