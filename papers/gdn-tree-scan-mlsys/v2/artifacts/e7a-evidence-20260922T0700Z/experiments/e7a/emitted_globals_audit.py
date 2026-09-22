#!/usr/bin/env python3
"""Bounded CPU audit of EMITTED (patched) Python sources for unbound module-global references.

For each nested scope (function/lambda/comprehension/class body) the compiler's own scoping decision (symtable) is
used: a symbol that is resolved as GLOBAL (implicit or explicit), is referenced, is not bound at module level
(assignment / import / def / class / `global X` + assignment inside a function) and is not a builtin is reported
with the enclosing scope names and the source lines of its Name loads. Names bound only under `TYPE_CHECKING` are
module-level bindings for this purpose (they are reported only if never bound anywhere).

Modes:
  audit  <file.py> [--json out]                 -> report for one file
  delta  <orig.py> <patched.py> [--json out]    -> names undefined in patched that are NOT undefined in orig
                                                   (the patch's NEW dangling references), plus both full lists
Exit status: audit -> 0 always (report), delta -> 1 if NEW undefined names exist, else 0.
"""
import argparse, ast, builtins, hashlib, json, symtable, sys

BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__", "__loader__", "__package__",
                                 "__builtins__", "__path__", "__annotations__", "__debug__", "__cached__"}


def _module_bound(top: symtable.SymbolTable) -> set:
    bound = set()
    for s in top.get_symbols():
        if s.is_assigned() or s.is_imported() or s.is_namespace() or s.is_parameter():
            bound.add(s.get_name())

    def walk(t):
        for c in t.get_children():
            for s in c.get_symbols():
                if s.is_declared_global() and s.is_assigned():
                    bound.add(s.get_name())
            walk(c)
    walk(top)
    return bound


def _load_lines(tree: ast.AST):
    """name -> sorted list of line numbers where the name is loaded (any scope)."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            out.setdefault(node.id, set()).add(node.lineno)
    return {k: sorted(v) for k, v in out.items()}


def audit(path: str) -> dict:
    src = open(path, encoding="utf-8").read()
    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    tree = ast.parse(src, path)
    top = symtable.symtable(src, path, "exec")
    bound = _module_bound(top)
    star_import = any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree))
    loads = _load_lines(tree)
    undefined = {}

    def walk(t, chain):
        for c in t.get_children():
            name_chain = chain + [c.get_name()]
            for s in c.get_symbols():
                n = s.get_name()
                if s.is_referenced() and (s.is_global() or s.is_declared_global()) and n not in bound and n not in BUILTINS:
                    rec = undefined.setdefault(n, {"scopes": [], "lines": loads.get(n, [])})
                    rec["scopes"].append(".".join(name_chain))
            walk(c, name_chain)
    walk(top, [])
    # module-level loads of unbound names (rare; executed at import)
    for s in top.get_symbols():
        n = s.get_name()
        if s.is_referenced() and not s.is_assigned() and not s.is_imported() and not s.is_namespace() and n not in bound and n not in BUILTINS:
            undefined.setdefault(n, {"scopes": [], "lines": loads.get(n, [])})["scopes"].append("<module>")
    return {"path": path, "sha256": sha, "n_lines": src.count("\n") + 1, "star_import_present": star_import,
            "module_bound_count": len(bound), "undefined": dict(sorted(undefined.items()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audit", "delta"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.mode == "audit":
        rep = audit(a.files[0]); rc = 0
    else:
        o, p = audit(a.files[0]), audit(a.files[1])
        new = {k: v for k, v in p["undefined"].items() if k not in o["undefined"]}
        rep = {"orig": o, "patched": p, "new_undefined_in_patched": new, "n_new": len(new),
               "resolved_by_patch": sorted(set(o["undefined"]) - set(p["undefined"]))}
        rc = 1 if new else 0
    txt = json.dumps(rep, indent=2)
    if a.json:
        open(a.json, "w").write(txt)
    if a.mode == "delta":
        print(f"{a.files[1]}: NEW undefined globals = {sorted(new)}  (orig had {len(o['undefined'])}, patched has {len(p['undefined'])})")
    else:
        print(f"{a.files[0]}: undefined globals = {sorted(rep['undefined'])}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
