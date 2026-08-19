"""FR13 host-residual rung: the post-DFWD tail's per-step input-prep plan.

Single source of truth for the tree depth-position plan that
``# FR10_TREE_DEPTH_POSITIONS`` recomputes from scratch on every decode step.

Why this module exists
----------------------
Under fixed32 the speculative token tree is a *patch-time constant*: the
patcher rewrites ``_fr10_tree_src`` to the baked ``_FR13_FIXED32_TREE_SOURCE``
literal before the block runs, and then asserts the derived choices equal the
baked ``_FR13_FIXED32_CHOICES``. The whole derivation --
``ast.literal_eval`` + ``sorted`` + four list comprehensions + two
``np.array`` builds -- is therefore a pure function of a compile-time
constant, recomputed 1146 times in the banked capture.

Measured price (scripts/fr13_host_tail_cost_probe.py on GB10, p50):
34 us at 9 paths, 242 us at the deployed 31 paths, 921 us at 63. The banked
capture puts the whole post-DFWD host tail at 3.458 ms/step of GPU idle, so
this one derivation is ~7% of it.

Byte-safety
-----------
Nothing here touches a tensor, a sampler input, or an ordering. It returns the
same integers the incumbent expression returns, and the caller still builds
its own fresh, writable ``np.ndarray`` from them, so no cached array can be
aliased or mutated across steps. ``derive_tree_depth_plan`` is the reference
implementation used both to bake the literals at patch time and to check them
in tests, so the baked form cannot drift from the expression it replaces.
"""

from __future__ import annotations

import ast
from typing import Any

# ROUND 18 ADJUDICATION -- KEPT hydra27/tail6, deliberately not widened.
# CORRECTLY hydra27-only, and this guard is the one that fired first in round
# 18's boot -- working exactly as designed. The lever replaces the depth-
# position derivation with literals BAKED from one tree; on any other tree
# those literals feed wrong RoPE depth offsets into the position rewrite and
# every token in the step lands at the wrong position, silently. There is no
# widening that is safe here: hydra31 needs its own bake, or the derivation.
_FIXED32_MODES = ("hydra27_fixed32", "tail6_fixed32")


def strict_flag(raw: Any, name: str) -> bool:
    """Parse a strict ``"0"``/``"1"`` flag, raising on anything else.

    A typo must never be read as OFF: the campaign has shipped "candidate"
    arms that were silently the stock path.
    """
    if raw is None:
        raise RuntimeError(
            f"{name} must be exactly 0 or 1 (observed None). It is read once, "
            "at patch time, and baked into the injected source."
        )
    text = str(raw).strip()
    if text not in ("0", "1"):
        raise RuntimeError(
            f"{name} must be exactly 0 or 1 (observed {raw!r}). A typo is "
            "never read as OFF, because an arm that cannot arm must die at "
            "preflight rather than serve as stock."
        )
    return text == "1"


def assert_host_tail_prep_requires_fixed32(enabled: bool, mode: Any) -> None:
    """Fail loud when the baked-plan lever is armed outside fixed32.

    The lever replaces the depth-position derivation with literals baked from
    ``_FR13_FIXED32_TREE_SOURCE``. That source is only substituted into the
    injected block under fixed32; outside fixed32 ``_fr10_tree_src`` comes
    from ``SPEC_CONFIG``/``speculative_config`` at runtime and can legally be
    any tree, so baked literals would silently serve the WRONG depth offsets
    into the RoPE position rewrite (``positions[...] = base + depth_offsets``)
    and every token in the step would be placed at the wrong position. No
    downstream check would catch it: the topology assertion the lever replaces
    is the only guard, and the lever is what removes it.

    Two ways out: set FR13_FIXED32_MODE (one of
    ('hydra27_fixed32', 'tail6_fixed32')), or unset FR13_HOST_TAIL_PREP_BAKE.
    """
    if not enabled:
        return
    text = "" if mode is None else str(mode).strip()
    if text in _FIXED32_MODES:
        return
    raise RuntimeError(
        "FR13_HOST_TAIL_PREP_BAKE=1 requires fixed32 to be armed but "
        f"FR13_FIXED32_MODE is {mode!r}. The lever bakes the tree "
        "depth-position plan as a literal, which is only sound because the "
        "patcher substitutes a constant _fr10_tree_src under fixed32. "
        "Outside fixed32 the tree source is read at runtime, so the baked "
        "plan would feed wrong RoPE depth offsets into "
        "positions[...] = base + depth_offsets and nothing downstream would "
        "catch it. Set FR13_FIXED32_MODE (one of "
        f"{_FIXED32_MODES!r}) or unset FR13_HOST_TAIL_PREP_BAKE."
    )


def derive_tree_depth_plan(tree_src: str) -> dict:
    """Reference implementation of the incumbent depth-position derivation.

    Mirrors ``# FR10_TREE_DEPTH_POSITIONS`` statement for statement:

        choices = sorted(ast.literal_eval(src), key=lambda p: (len(p), p))
        depth_offsets = [0] + [len(c) for c in choices]
        spine = [c for c in choices if all(int(v) == 0 for v in c)]
        leaf  = [c for c in choices if not all(int(v) == 0 for v in c)]
        spine_first = [0] + [len(c) for c in spine] + [len(c) for c in leaf]
        tree_n = len(depth_offsets)

    Returns plain Python tuples of ints. The caller constructs its own arrays,
    so nothing returned here can be aliased into a served tensor.
    """
    choices = sorted(ast.literal_eval(tree_src), key=lambda p: (len(p), p))
    choices = tuple(tuple(choice) for choice in choices)
    depth_offsets = (0,) + tuple(len(choice) for choice in choices)
    spine = tuple(
        choice for choice in choices if all(int(v) == 0 for v in choice)
    )
    leaf = tuple(
        choice for choice in choices if not all(int(v) == 0 for v in choice)
    )
    spine_first = (
        (0,)
        + tuple(len(choice) for choice in spine)
        + tuple(len(choice) for choice in leaf)
    )
    return {
        "choices": choices,
        "depth_offsets": depth_offsets,
        "spine_first_depth_offsets": spine_first,
        "tree_n": len(depth_offsets),
    }


def baked_plan_source(tree_src: str, indent: str) -> str:
    """Emit the injected replacement for the derivation, as source text.

    ``indent`` is the leading whitespace of the statements being replaced, so
    the emitted block drops into the same suite. The emitted code rebuilds
    ``_fr10_choices`` as a fresh ``list`` and both offset vectors as fresh
    ``np.ndarray`` objects on every call, exactly as the incumbent does --
    only the parse, the sort and the four comprehensions are gone.
    """
    plan = derive_tree_depth_plan(tree_src)
    lines = [
        "# FR13_HOST_TAIL_PREP_BAKE: the fixed32 tree source is a patch-time",
        "# constant (_fr10_tree_src is substituted above), so this plan is a",
        "# pure function of a literal. Baked by",
        "# lumo_flywheel_serving.fr13_host_tail_prep.baked_plan_source; the",
        "# derivation it replaces is derive_tree_depth_plan in that module.",
        f"_fr10_choices = list({plan['choices']!r})",
        f"_fr10_depth_offsets = np.array({list(plan['depth_offsets'])!r},"
        " dtype=np.int64)",
        "_fr10_spine_first_depth_offsets = np.array(",
        f"    {list(plan['spine_first_depth_offsets'])!r}, dtype=np.int64",
        ")",
    ]
    return "".join(indent + line + "\n" for line in lines)


def plan_census(tree_src: str) -> dict:
    """Machine-readable summary for the design doc and the tests."""
    plan = derive_tree_depth_plan(tree_src)
    return {
        "paths": len(plan["choices"]),
        "tree_n": plan["tree_n"],
        "depth_offsets": list(plan["depth_offsets"]),
        "spine_first_depth_offsets": list(plan["spine_first_depth_offsets"]),
    }
