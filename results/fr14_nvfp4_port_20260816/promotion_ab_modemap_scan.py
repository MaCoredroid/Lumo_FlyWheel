"""Repo-wide: every module-level mapping KEYED BY fixed32 mode name, and whether
its key set is complete. Site 8 lived in the authority; site 11 lives in a module
that never imports it -- so scanning the authority alone is insufficient."""
import ast, sys, pathlib
ROSTER = {"tail6_fixed32", "hydra27_fixed32", "hydra31_fixed32"}
root = pathlib.Path("/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816")
rows = []
for sub in ("scripts", "src", "tests", "config"):
    for p in sorted((root / sub).rglob("*.py")):
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = []
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.append(k.value)
            ks = set(keys)
            if not ks or not (ks & ROSTER):
                continue
            if not ks <= ROSTER:
                continue          # mixed dict, not a pure mode index
            missing = ROSTER - ks
            rows.append((str(p.relative_to(root)), node.lineno, sorted(ks), sorted(missing)))
inc = [r for r in rows if r[3]]
print(f"pure mode-keyed mappings: {len(rows)}   INCOMPLETE: {len(inc)}")
for f, ln, ks, miss in rows:
    flag = "INCOMPLETE" if miss else "complete  "
    print(f"  {flag} {f}:{ln}  keys={ks} missing={miss}")
