#!/usr/bin/env python3
"""OFF-equivalence prover for the FR13 dead-flag cleanup.

Given a PATCHED source file that contains runtime-gated dead code (guarded by env
flags that default OFF), fold every deleted-flag reference to its OFF value,
constant-fold the resulting boolean/ternary/if, and dead-code-eliminate any
function/module-global that becomes unreferenced. The output is THE correct
"flags-deleted" source: any patcher edit that reproduces it is proven behavior-
preserving with the flags OFF (their default), so no EXACT_SEED / kept behavior
can have changed.

  cleanup_offeq.py fold   <patched.py>            -> stdout: folded (=target) source
  cleanup_offeq.py compare <patched_HEAD.py> <patched_NEW.py>
        -> exit 0 + "OFF_EQUIVALENT" iff ast(fold(HEAD)) == ast(NEW)
           else prints the first differing region.
"""
import ast, sys

# env-flag reads that fold to a constant (all default OFF => the ==\"1\" test is False,
# the !=\"1\" test is True). Matched on the ast of os.environ.get("NAME", d) OP "1".
DELETED_FLAGS = {
    "FR13_APC_BLOCK_REFOLD", "FR13_APC_REFOLD_TO_SNAPSHOT", "FR13_APC_PRE_SNAP_FIX",
    "FR13_TREE_GDN_SLOT_PIN", "FR13_DECODE_GDN_CAPTURE",
}
# module globals that are the flag gate (defined as `X = (os.environ.get(flag)==...)`)
GATE_GLOBALS = {"_FR13_REFOLD_ON"}


def _is_env_flag_read(node):
    """os.environ.get("FLAG", default) -> ('FLAG', ) if FLAG is a deleted flag."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "get" and node.args:
        base = node.func.value
        if isinstance(base, ast.Attribute) and base.attr == "environ":
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and a0.value in DELETED_FLAGS:
                return a0.value
    return None


def _is_gate_global(node):
    """_FR13_REFOLD_ON  OR  globals().get("_FR13_REFOLD_ON", False)."""
    if isinstance(node, ast.Name) and node.id in GATE_GLOBALS:
        return node.id
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "get" and node.args \
            and isinstance(node.func.value, ast.Call) \
            and isinstance(node.func.value.func, ast.Name) \
            and node.func.value.func.id == "globals":
        a0 = node.args[0]
        if isinstance(a0, ast.Constant) and a0.value in GATE_GLOBALS:
            return a0.value
    return None


class FlagFolder(ast.NodeTransformer):
    """Substitute deleted-flag reads with their OFF constant, then fold."""

    def visit_Compare(self, node):
        self.generic_visit(node)
        flag = _is_env_flag_read(node.left)
        if flag and len(node.ops) == 1 and isinstance(node.comparators[0], ast.Constant):
            op = node.ops[0]
            rhs = node.comparators[0].value
            # env default is "0" (OFF); the literal string the code compares against
            off_val = "0"
            if isinstance(op, ast.Eq):
                return ast.copy_location(ast.Constant(off_val == rhs), node)
            if isinstance(op, ast.NotEq):
                return ast.copy_location(ast.Constant(off_val != rhs), node)
        return node

    def visit_Name(self, node):
        # only fold Load-context reads; never the Store LHS of the gate's own def
        if isinstance(node.ctx, ast.Load) and _is_gate_global(node):
            return ast.copy_location(ast.Constant(False), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if _is_gate_global(node):
            return ast.copy_location(ast.Constant(False), node)
        return node


def _const(node):
    return node.value if isinstance(node, ast.Constant) else _MISS


_MISS = object()


class Simplifier(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        vals = node.values
        if isinstance(node.op, ast.Or):
            kept = [v for v in vals if _const(v) is not False]  # drop `... or False`
            if any(_const(v) is True for v in kept):
                return ast.copy_location(ast.Constant(True), node)
            if not kept:
                return ast.copy_location(ast.Constant(False), node)
            return kept[0] if len(kept) == 1 else ast.copy_location(ast.BoolOp(node.op, kept), node)
        else:  # And
            if any(_const(v) is False for v in vals):
                return ast.copy_location(ast.Constant(False), node)
            kept = [v for v in vals if _const(v) is not True]
            if not kept:
                return ast.copy_location(ast.Constant(True), node)
            return kept[0] if len(kept) == 1 else ast.copy_location(ast.BoolOp(node.op, kept), node)

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            return ast.copy_location(ast.Constant(not node.operand.value), node)
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        t = _const(node.test)
        if t is True:
            return node.body
        if t is False:
            return node.orelse
        return node

    def _fold_if(self, node):
        self.generic_visit(node)
        t = _const(node.test)
        if t is True:
            return node.body            # inline the body statements
        if t is False:
            return node.orelse          # drop body, keep else (may be [])
        return [node]

    def visit_If(self, node):
        return self._fold_if(node)


def _referenced_names(tree):
    used = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            used.add(n.id)
        if isinstance(n, ast.Attribute):
            used.add(n.attr)
    return used


def _dce_module(tree):
    """Drop top-level FunctionDefs / simple Assign globals that are never referenced.
    Iterate to fixpoint. Only removes names starting with the fr13 dead prefixes."""
    DEAD_PREFIX = ("_fr13_pathA_refold", "_FR13_REFOLD", "_fr13_ep_refold",
                   "_fr13_refold", "_FR13_PRE_SNAP", "_fr13_tree_gdn_slot_pin",
                   "_FR13_SLOT_PIN", "_fr13_decode_gdn_capture", "_FR13_DECODE_GDN")
    changed = True
    while changed:
        changed = False
        used = _referenced_names(tree)
        new_body = []
        for stmt in tree.body:
            drop = False
            if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith(DEAD_PREFIX):
                if stmt.name not in used:
                    drop = True
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                    and isinstance(stmt.targets[0], ast.Name) \
                    and stmt.targets[0].id.startswith(DEAD_PREFIX):
                if stmt.targets[0].id not in used:
                    drop = True
            if drop:
                changed = True
            else:
                new_body.append(stmt)
        tree.body = new_body
    return tree


def fold_source(src):
    tree = ast.parse(src)
    tree = FlagFolder().visit(tree)
    # simplify to fixpoint (folding an If can expose a foldable BoolOp above it)
    for _ in range(6):
        tree = Simplifier().visit(tree)
    ast.fix_missing_locations(tree)
    tree = _dce_module(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def main():
    mode = sys.argv[1]
    if mode == "fold":
        print(fold_source(open(sys.argv[2]).read()))
    elif mode == "compare":
        head_folded = fold_source(open(sys.argv[2]).read())
        new_norm = ast.unparse(ast.parse(open(sys.argv[3]).read()))
        if head_folded == new_norm:
            print("OFF_EQUIVALENT")
            sys.exit(0)
        # find first differing line
        h = head_folded.splitlines()
        n = new_norm.splitlines()
        for i, (a, b) in enumerate(zip(h, n)):
            if a != b:
                lo = max(0, i - 3)
                print(f"DIFFER at line {i}:")
                print("--- fold(HEAD) ---")
                print("\n".join(h[lo:i + 4]))
                print("--- NEW ---")
                print("\n".join(n[lo:i + 4]))
                break
        else:
            print(f"length differ: fold(HEAD)={len(h)} NEW={len(n)} lines")
        sys.exit(1)


if __name__ == "__main__":
    main()
