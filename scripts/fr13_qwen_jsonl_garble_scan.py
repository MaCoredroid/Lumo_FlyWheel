#!/usr/bin/env python3
"""Scan qwen-code proxy request-dump JSONLs (chatreq_*.json) for GARBLE in the
served (model-generated) content.

The offload pair-dump (served-stream capture) is /v1/responses-only and qwen-code
hits /v1/chat/completions, so it captures 0. But the request dumps (chatreq_*.json,
LUMO_PROXY_REQUEST_DUMP_DIR) accumulate the full conversation INCLUDING the model's
prior assistant turns -> the served code lives in assistant tool_call arguments
(edit new_string, run_shell_command python) + assistant content code blocks.

Garble (tree spec-decode commits a wrong/near-neighbor identifier) corrupts that
generated code -> undefined names / syntax errors / malformed tool-call JSON. This
is the temp-0.6 gate's undefined-name methodology applied to the agentic served
code. The reliable signal is the tree(cat8/cat9)-vs-native RATE on the SAME tasks.
"""
import json, os, sys, glob, ast, builtins, re, argparse

BI = set(dir(builtins)) | {
    "self", "cls", "np", "torch", "os", "sys", "re", "math", "pd", "plt",
    "json", "time", "Path", "pytest", "__file__", "__name__",
}


def extract_code_snippets(chatreq_path):
    try:
        r = json.load(open(chatreq_path))
    except Exception:
        return
    for m in r.get("messages", []):
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str) and "```" in c:
            for blk in re.findall(r"```(?:python)?\n(.*?)```", c, re.S):
                yield ("content_block", blk)
        for t in (m.get("tool_calls") or []):
            fn = t.get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            try:
                a = json.loads(args)
            except Exception:
                yield ("malformed_toolargs", args)   # malformed JSON = a garble signal
                continue
            if name == "edit" and isinstance(a.get("new_string"), str):
                yield ("edit_new", a["new_string"])
            if name == "run_shell_command" and isinstance(a.get("command"), str):
                for pyc in re.findall(r'python[0-9.]* -c "(.*?)"', a["command"], re.S):
                    yield ("shell_py", pyc.replace("\\n", "\n").replace('\\"', '"'))


def undefined_names(code):
    try:
        tr = ast.parse(code)
    except Exception:
        return None            # syntax error
    L = {x.id for x in ast.walk(tr) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
    S = {x.id for x in ast.walk(tr) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
    A = set()
    for fn in ast.walk(tr):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for a in fn.args.args + fn.args.posonlyargs + fn.args.kwonlyargs:
                A.add(a.arg)
    # imported names count as defined
    for imp in ast.walk(tr):
        if isinstance(imp, ast.alias):
            S.add((imp.asname or imp.name).split(".")[0])
    return L - S - A - BI


def scan_dir(d):
    rd = os.path.join(d, "proxy_request_dumps")
    root = rd if os.path.isdir(rd) else d
    files = glob.glob(os.path.join(root, "**", "chatreq_*.json"), recursive=True)
    seen = set()
    n_snip = n_syntax_bad = n_with_undef = n_malformed = 0
    undef_examples = []
    for f in files:
        for kind, code in extract_code_snippets(f):
            key = (kind, code)
            if key in seen:
                continue
            seen.add(key)
            if kind == "malformed_toolargs":
                n_malformed += 1
                continue
            n_snip += 1
            undef = undefined_names(code)
            if undef is None:
                n_syntax_bad += 1
            elif undef:
                n_with_undef += 1
                if len(undef_examples) < 12:
                    undef_examples.append(sorted(undef)[:6])
    return dict(files=len(files), snippets=n_snip, syntax_bad=n_syntax_bad,
                with_undef=n_with_undef, malformed_toolargs=n_malformed,
                undef_examples=undef_examples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="arm dirs (each holds proxy_request_dumps/)")
    a = ap.parse_args()
    for d in a.dirs:
        s = scan_dir(d)
        print(f"{d}")
        print(f"  chatreq_files={s['files']} unique_code_snippets={s['snippets']} "
              f"syntax_bad={s['syntax_bad']} with_undef={s['with_undef']} "
              f"malformed_toolargs={s['malformed_toolargs']}")
        if s["snippets"]:
            print(f"  undef_rate={s['with_undef']/s['snippets']*100:.1f}%  "
                  f"syntax_bad_rate={s['syntax_bad']/s['snippets']*100:.1f}%  "
                  f"(malformed_toolargs is an ABSOLUTE garble count)")
        if s["undef_examples"]:
            print(f"  undef_examples: {s['undef_examples'][:6]}")


if __name__ == "__main__":
    main()
