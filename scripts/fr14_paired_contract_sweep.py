#!/usr/bin/env python3
"""Enumerate PAIRED structures around the fixed32 drafter shape, and report stale sides.

WHY THIS EXISTS
---------------
Three campaign blockers in a row were one defect shape: **a paired structure
updated on one side only.**

  * TAW emitter vs its mirrors
  * a bash pin vs its Python twin
  * `_fr13_fixed32_drafter_proposal_end`'s census half vs its runtime-evidence
    half (the 12th site: the census half was made pass-aware when the split
    landed, the runtime half twenty lines later still said `graph_replays: 1`)

Each was found by a boot. A boot is an expensive way to learn that two literals
disagree, so this enumerates the pairs and checks both sides mechanically.

WHAT A "PAIR" IS HERE
---------------------
Two artifacts that must encode the same fact, where nothing forces them to
change together:

  emitter/validator   the runtime writes a census record; the reducer validates it
  half/half           two blocks inside one function that describe one step
  contract/consumer   a shared constant vs a hand-written literal of that constant
  twin/twin           the two launcher families, which must stay identical
  bash/python         a launcher literal vs the Python that parses or mirrors it

The split-graph change moved five quantities. Any site still carrying the
pre-split literal for one of them is a stale side:

  mtp_forward_calls   4      -> 4 (ungated) | 2 (gated)
  graph_replays       1      -> 1 | 2 (armed ungated replays lo then hi)
  graph_captures      0|1    -> 0|1|2 (a split capture opens two scopes)
  main_tail_length    6      -> 6 | 8 (gated hands off at draft position 3)
  arctic_requested    12     -> 12 | 14

Run:  python3 scripts/fr14_paired_contract_sweep.py [--json out.json]
Exit 1 if any enumerated pair has a stale side.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
# The OTHER patcher. Nothing in this family scanned it, which is why the FA2
# arm-identity resolver could carry a fall-through through an entire campaign:
# every check here walked the GDN patcher's blobs and none walked these.
FA2_PATCHER = REPO / "scripts" / "fr13_patch_fa2_tree_bias.py"
CENSUS = REPO / "scripts" / "fr13_fixed32_work_census.py"
TOPOLOGY = REPO / "scripts" / "fr13_fixed32_topology.py"
DRAFTER = REPO / "scripts" / "fr13_merged_drafter.py"
LAUNCHERS = (
    REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh",
    REPO / "scripts" / "fr14_armb_leg3_launch_nomiddleware.sh",
    # The third twin. It was outside this tuple, so its pin case drifted a
    # whole arm behind its siblings and nothing said so -- pair_launcher_twins
    # already listed "gqa_pair_splitk" among its required markers and simply
    # never looked at this file.
    REPO / "scripts" / "fr14_leg3_launch_nomiddleware.sh",
)


def all_injected_blobs():
    """EVERY large source string the patcher injects, not just the gdn one.

    This is the correction the 14th site forced. `injected_blob()` below returns
    the ONE blob bound to a named global; the eagle proposer -- where the pack
    identity lives -- is an anonymous `new = \'\'\'...\'\'\'` local, so nothing
    enumerated it and a shape literal sat there through three boots.

    Returns [(lineno, text, parsed_or_None)]. Blobs that will not parse even
    after dedent/wrapping are returned with None so callers can report coverage
    honestly instead of silently skipping them.
    """
    import textwrap

    src = PATCHER.read_text()
    out = []
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) > 2000
        ):
            parsed = None
            for attempt in (
                lambda s: s,
                textwrap.dedent,
                lambda s: "if True:\n" + s,
            ):
                try:
                    parsed = ast.parse(attempt(node.value))
                    break
                except SyntaxError:
                    continue
            out.append((node.lineno, node.value, parsed))
    out.sort(key=lambda r: r[0])
    return out


def eagle_blob() -> str:
    """The proposer blob: the one that carries the drafter pack identity."""
    for _lineno, text, _parsed in all_injected_blobs():
        if "_fr13_t_cols" in text and "_fr13_t_paths" in text:
            return text
    raise SystemExit("eagle proposer blob not found")


def injected_blob() -> str:
    """The runtime source the patcher execs into gdn_linear_attn.

    It is string content, so the patcher's own `ast.parse` does not cover it and
    neither does any import. Everything in it has to be checked textually.
    """
    tree = ast.parse(PATCHER.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None)
            == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
        ):
            return ast.literal_eval(node.value)
    raise SystemExit("injected runtime blob not found")


def _topology():
    sys.path.insert(0, str(REPO / "scripts"))
    import fr13_fixed32_topology as topo

    return topo


# ---------------------------------------------------------------------------
# The pairs.  Each check returns (ok, detail).
# ---------------------------------------------------------------------------

def pair_contract_vs_blob(blob, topo):
    """The shared shape contract vs the literal the blob must carry.

    The blob cannot import the contract, so this is the only thing keeping them
    equal.  This is the pair the 12th site lived in.
    """
    m = re.search(
        r'int\(proposal\["graph_replays"\]\),\s*\)\s*not in (.+)$',
        blob,
        re.M,
    )
    if m is None:
        return False, "blob no longer carries the (calls, replays) shape tuple"
    literal = ast.literal_eval(m.group(1).strip())
    if tuple(topo.LEGAL_STEP_SHAPES) != literal:
        return False, (
            f"contract {topo.LEGAL_STEP_SHAPES} != blob literal {literal}"
        )
    return True, f"agree on {literal}"


def pair_contract_vs_blob_captures(blob, topo):
    m = re.search(r'proposal\["graph_captures"\]\) not in \(([0-9, ]+)\)', blob)
    if m is None:
        return False, "blob no longer bounds graph_captures"
    literal = tuple(int(x) for x in m.group(1).replace(" ", "").split(","))
    if tuple(topo.LEGAL_GRAPH_CAPTURES) != literal:
        return False, (
            f"contract {topo.LEGAL_GRAPH_CAPTURES} != blob literal {literal}"
        )
    return True, f"agree on {literal}"


def pair_contract_vs_census(topo):
    """The census must consume the contract, not re-type it."""
    text = CENSUS.read_text()
    missing = [
        name
        for name in ("LEGAL_STEP_SHAPES", "LEGAL_GRAPH_CAPTURES",
                     "LEGAL_HANDOFF_SHAPES")
        if name not in text
    ]
    if missing:
        return False, f"census does not consume {missing}"
    return True, "census imports all three contract tuples"


def pair_census_halves():
    """`drafter` and `drafter_runtime` describe ONE step and must agree.

    The census already cross-checks them at runtime via `runtime_projection`;
    what this asserts is that neither half reintroduced a bare literal for a
    split-sensitive quantity.
    """
    text = CENSUS.read_text()
    stale = []
    if re.search(r'_expect\(\s*runtime_mtp_calls,\s*MTP_FORWARD_CALLS\b', text):
        stale.append("drafter_runtime.mtp_forward_calls pinned to 4")
    if re.search(r'"graph_replays"\)\s*,\s*\n?\s*1,', text):
        stale.append("drafter_runtime.graph_replays pinned to 1")
    if "graph_captures not in (0, 1)" in text:
        stale.append("drafter_runtime.graph_captures bounded at 0|1")
    # `reference_event` legitimately builds the UNGATED fixture; the pair that
    # matters is the validator's runtime_projection, which must derive.
    proj = text[text.index("runtime_projection = {"):]
    proj = proj[: proj.index("}")]
    if "ARCTIC_MAIN_TAIL_LENGTH" in proj:
        stale.append("runtime_projection.main_tail_length pinned to 6")
    return (not stale), (stale or "both halves derive from the observed shape")


def pair_emitter_halves(blob):
    """proposal_end writes a census half and a runtime-evidence half."""
    stale = []
    if re.search(r'"graph_replays":\s*1,', blob):
        stale.append('drafter_runtime emitter writes graph_replays: 1')
    if re.search(r'"mtp_forward_calls":\s*4,', blob):
        stale.append('drafter_runtime emitter writes mtp_forward_calls: 4')
    if re.search(r'"mtp_forward_rows":\s*4 \* batch,', blob):
        stale.append('drafter_runtime emitter writes mtp_forward_rows: 4*batch')
    if re.search(r'"main_tail_length":\s*6,', blob):
        stale.append('census emitter writes main_tail_length: 6')
    # Scope to the DRAFTER evidence chain by VARIABLE name: the target
    # forward-graph chain in the same function uses a bare `evidence` and
    # legitimately replays once per step, so a function-level scope would
    # produce a false positive on it.
    if re.search(
        r'int\(drafter_evidence\.get\("matching_replays", -1\)\)\s*\n?\s*!= 1\b',
        blob,
    ):
        stale.append("drafter_evidence demands matching_replays == 1")
    # and proposal_end's own check must be TIED to the proposal, not a literal
    if not re.search(
        r'int\(evidence\.get\("matching_replays", -1\)\)\s*\n\s*'
        r'!= int\(proposal\["graph_replays"\]\)',
        blob,
    ):
        stale.append("proposal_end evidence check is not tied to graph_replays")
    return (not stale), (stale or "both emitter halves derive from the proposal")


def pair_observer_vs_context(blob):
    """The per-forward observer vs the capture context it validates against."""
    stale = []
    if 'proposal.get("graph_captures") != 1' in blob:
        stale.append("tree-attention observer pins graph_captures to 1")
    if "tree_calls not in (0, 1, 2, 3)" in blob:
        stale.append("tree-attention observer bounds forwards at 4")
    return (not stale), (stale or "observer is segment- and pass-aware")


def pair_launcher_twins():
    """The two launcher families must carry identical FR14 blocks."""
    texts = [p.read_text() for p in LAUNCHERS]
    markers = (
        "FR14_SUFFIX_PASS_GATE",
        "FR14_GATE_SPLIT_GRAPH",
        "FR14_FUSED_DRAFT_TOPK",
        "gqa_pair_splitk",
        "_fr14_gate_incompat",
        "fr14_suffix_pass_gate.cfg",
    )
    mismatched = [
        m for m in markers
        if (m in texts[0]) != (m in texts[1])
    ]
    if mismatched:
        return False, f"present in only one launcher: {mismatched}"
    counts = {
        m: (texts[0].count(m), texts[1].count(m)) for m in markers
    }
    skewed = {m: c for m, c in counts.items() if c[0] != c[1]}
    if skewed:
        return False, f"occurrence counts differ: {skewed}"
    return True, f"{len(markers)} FR14 blocks identical across both families"


def pair_bash_cfg_vs_python_parser():
    """The launcher writes the gate sidecar; the gate module parses it."""
    launcher = LAUNCHERS[0].read_text()
    m = re.search(
        r'echo "\$\{FR14_SUFFIX_PASS_GATE_NGRAM:-\d+\} '
        r'\$\{FR14_SUFFIX_PASS_GATE_MIN_AGREE:-[\d.]+\} '
        r'\$\{FR14_SUFFIX_PASS_GATE_MIN_HISTORY:-\d+\}"',
        launcher,
    )
    if m is None:
        return False, "launcher no longer writes the 3-field sidecar"
    gate = (REPO / "scripts" / "fr14_suffix_pass_gate.py").read_text()
    if "len(raw) != 3" not in gate:
        return False, "gate parser no longer requires exactly 3 fields"
    return True, "3 fields written, 3 fields required"


def pair_handoff_contract_vs_blob(blob, topo):
    """The handoff interlock restates LEGAL_HANDOFF_SHAPES as a two-branch."""
    m = re.search(
        r'!= \((\d+) if int\(proposal\["mtp_forward_calls"\]\) == (\d+) '
        r'else (\d+)\)',
        blob,
    )
    if m is None:
        return False, "blob no longer carries the handoff interlock"
    gated_tail, gated_calls, ungated_tail = (int(x) for x in m.groups())
    literal = {(gated_calls, gated_tail),
               (topo.MTP_FORWARD_CALLS, ungated_tail)}
    if literal != set(topo.LEGAL_HANDOFF_SHAPES):
        return False, (
            f"contract {set(topo.LEGAL_HANDOFF_SHAPES)} != blob {literal}"
        )
    return True, f"agree on {sorted(literal)}"


def pair_pack_identity_under_both_shapes(topo):
    """THE 14th-site check: evaluate the pack identity, do not read it.

    The identity is `head + (arctic - arctic_in_head) + rescue == 31`. It is
    extracted from the eagle blob and EVALUATED under the ungated and the gated
    bindings, which is what a literal `15` cannot survive.
    """
    blob = eagle_blob()
    m = re.search(
        r"_fr14_head_cols\s*\n\s*\+ \(\s*\n\s*len\(_fr13_t_cols\)\s*\n"
        r"\s*- _fr14_arctic_in_head\s*\n\s*\)\s*\n\s*"
        r"\+ len\(_fr13_t_paths\)\s*\n\s*!= (\d+)",
        blob,
    )
    if m is None:
        return False, "pack identity is no longer the derived form"
    width = int(m.group(1))
    if width != topo.PHYSICAL_DRAFTS:
        return False, f"pack width {width} != {topo.PHYSICAL_DRAFTS}"
    hd_full = topo.N_MTP_HEAD_DEPTHS
    rescue = sum(length for _r, length in topo.PHYSICAL_BRANCH_CHAINS)
    for label, hd, cols in (
        ("ungated", hd_full, topo.ARCTIC_MAIN_TAIL_LENGTH),
        ("gated", topo.GATED_MTP_K, topo.GATED_ARCTIC_MAIN_TAIL_LENGTH),
    ):
        head = hd_full * (1 + topo.BRANCHES_PER_HEAD_DEPTH)
        in_head = hd_full - hd
        total = head + (cols - in_head) + rescue
        if total != topo.PHYSICAL_DRAFTS:
            return False, f"{label}: {total} != {topo.PHYSICAL_DRAFTS}"
    return True, "identity holds at 31 under both shapes"


def pair_profiles_vs_patcher_mode_table(blob, topo):
    """The profile table vs the patcher's mode literal (the blob cannot import)."""
    src = PATCHER.read_text()
    m = re.search(r"_FR13_FIXED32_MODES = \{(.*?)\n\}", src, re.S)
    if m is None:
        return False, "patcher mode table not found"
    body = m.group(1)
    for mode in (topo.PROFILE_HYDRA27, topo.PROFILE_HYDRA31):
        prof = topo.profile(mode)
        want = f'"{mode}": ({prof["valid_mask"]:#010X}, {prof["active_drafts"]})'
        want_lower = want.replace("0X", "0x")
        if want not in body and want_lower not in body:
            return False, f"patcher mode table missing/stale for {mode}: {want}"
    return True, "patcher mode table matches every profile"


def pair_profiles_vs_blob_needle(blob, topo):
    """The live topology needle's mask map must know every profile."""
    m = re.search(r"expected_masks = \{(.*?)\n\s*\}", blob, re.S)
    if m is None:
        return False, "topology needle mask map not found"
    body = m.group(1)
    for mode in (topo.PROFILE_HYDRA27, topo.PROFILE_HYDRA31):
        mask = topo.profile(mode)["valid_mask"]
        if f'"{mode}": {mask:#010X}' not in body.replace("0x", "0X").replace(
            "ABDFFFF", "ABDFFFF"
        ) and f'"{mode}": {mask:#010x}' not in body:
            return False, f"needle mask map missing {mode}"
    return True, "needle mask map matches every profile"


def pair_profiles_vs_census_modes(blob, topo):
    text = CENSUS.read_text()
    if "HYDRA31_MODE" not in text or "shape_profile" not in text:
        return False, "census does not know the hydra31 profile"
    if "_SHAPE_PROFILE_BY_MODE" not in text:
        return False, "census has no mode->profile mapping"
    return True, "census consumes the profile table"


def pair_topology_vs_drafter():
    """decide_fixed32's gated widths vs the topology contract."""
    topo = _topology()
    text = DRAFTER.read_text()
    # STAGE 2: the flat constants were replaced by the profile table, so the
    # check is that decide_fixed32 takes its widths from `profile(mode)` and
    # restates none of them.
    for name in ("profile", "fixed32_mode()", "GATED_MTP_K"):
        if name not in text:
            return False, f"decide_fixed32 does not consume {name}"
    for key in (
        "gated_main_tail_length",
        "gated_arctic_requested_tokens",
        "main_tail_length",
        "arctic_requested_tokens",
        "rescue_carry_slots",
        "physical_branch_chains",
    ):
        if f'prof["{key}"]' not in text and f'["{key}"]' not in text:
            return False, f"decide_fixed32 does not take {key} from the profile"
    if topo.GATED_MTP_K + topo.GATED_ARCTIC_MAIN_TAIL_LENGTH != 11:
        return False, "gated shape does not reach draft position 10"
    return True, "drafter consumes the topology constants"


# Shape literals adjudicated once, by asking of each: "does this value change
# under 3 post-root passes instead of 5?"  Anything NOT here that lands in a
# drafter-relevant arithmetic/comparison position is reported for adjudication.
#   value, context substring, verdict
ADJUDICATED_SHAPE_LITERALS = (
    (4, "len(num_draft_tokens) <= 4", "batch bound, not a pass count"),
    (4, "batch == 4", "batch size"),
    (4, 'int(work["batch_size"]) == 4', "batch size"),
    (4, 'context.get("batch_size", -1) != 4', "batch size"),
    (4, "rows + 4", "batch size (this lever is B4-only)"),
    (4, "int(_fr10_spine_steps) == 4", "loop still has 4 iterations; the split "
        "changes CAPTURE, not the loop count"),
    (4, "num_kv_heads == 4", "attention geometry"),
    (4, "passes=4", "default arg = the ungated value; every split call site "
        "passes it explicitly"),
    (4, "passes == 4", "derived from the pass count, not a restatement"),
    (4, "passes != 4", "derived from the pass count"),
    (4, '"rank1_lookup_tokens": 4 * batch', "rank-1 chain hangs off the ROOT "
        "runner-up; unaffected by skipped passes"),
    (2, '"rank2_lookup_tokens": 2 * batch', "rank-2 chain, same reason"),
    (4, '"rescue_carry_slots": 4 * batch', "4 permanently inactive rank-2 slots"),
    (3, '"arctic_lookup_calls": 3 * batch', "always main+rank1+rank2"),
    (6, "(_fr14_main + 6)", "6 = rank1(4)+rank2(2); main is derived"),
    (10, "_fr14_main + 10", "10 = rescue path columns, topology-fixed"),
    (16, '"drafter_pair_slots": 16 * batch', "KV remap path capacity "
        "(COMMIT_PATH_CAP), per tree path not per pass"),
    (5, "draft_events * 5", "GATE-VARIANT (5 head reads/step) but structurally "
        "unreachable: FR13_DRAFT_HEAD_M32_LIVE_AB is refused with the gate"),
)


# Lines in the UNPARSEABLE blobs adjudicated the same way. These are string
# fragments the patcher assembles, so they carry the shape of generated code
# rather than of executed code.
ADJUDICATED_TEXTUAL = (
    (31, "num_decode_draft_tokens", "verify row count, topology-fixed"),
    (32, "tree_n=32", "verify rows"),
    (4, "num_kv_heads", "attention geometry"),
    (16, "path_capacity", "KV remap capacity"),
    (4, "1 <= _fr13_mtp_B <= 4", "batch bound"),
    (4, "1 <= _fr13_mtp_batch_rows <= 4", "batch bound"),
    (4, "FR13_SLOT_REORDER (edits 1+4 of 5)", "docstring prose: the 5 EDITS of "
        "the slot-reorder patch, not MTP passes"),
)


# Category rules, each a stated ARGUMENT for why the magnitude cannot vary with
# the post-root pass count. A rule suppresses only literals whose line matches
# it; anything unmatched is reported for adjudication, so "0 unreviewed" stays
# meaningful.
ADJUDICATED_RULES = (
    (r"batch", {4, 2},
     "batch-size bound or per-batch multiplier; B is orthogonal to pass count"),
    (r"\b12\b", {12},
     "committer/TAW walk depth (WALK_CAP = MAX_PHYSICAL_DEPTH + 1). Gating "
     "does not change the tree's reach -- both shapes end at draft position 10"),
    (r"path_cap|path_capacity|paths\.shape|path_shape|alias_|row_guard|"
     r"pair_rows|pair_slots|slots_written|target_names|caches !=|tree_q_rows|"
     r"_ep_ptr_list", {16},
     "COMMIT_PATH_CAP / KV-remap capacities, fixed by the topology"),
    (r"48 \*|kernel_warps|production_bv", {6, 8},
     "GDN layer geometry and block-vector width"),
    (r"compares|products|nonfinite|mismatches|compared_", {5, 4, 2},
     "M32 draft-head lever tuple arities. GATE-VARIANT where they encode 5 head "
     "reads, but the lever is refused with the gate by both launchers"),
    (r"\(\"cuda\",\) \* 4|\(\"torch\.int32\",\) \* 4|len\(_fr13_dm_ret\) == 4|"
     r"len\(bank_shape\) != 4|direct_ring_inputs", {4},
     "tuple arity / fixed kernel inputs, not a pass count"),
)


def shape_literal_scan():
    """Every shape-magnitude literal in a drafter-relevant arithmetic position.

    Restated class: **no literal may encode the ungated 5-pass shape.** This
    walks ALL injected blobs -- the 14th site was in the one nobody enumerated.
    """
    import textwrap

    magnitudes = {15, 6, 10, 5, 4, 12, 16, 8, 14, 18}
    relevant = ("mtp", "drafter", "tail", "spine", "arctic", "pack",
                "tree_attn", "head", "draft", "wide", "fixed32")
    unadjudicated = []
    parsed = unparsed = 0
    # A blob that will not parse is not a blob we may ignore: 24 of them do not,
    # and one of those could hold the next 14th site. They get a line-oriented
    # check for the same magnitudes in the same contexts.
    textual = re.compile(
        r"(?:[=!<>]=|[-+*]|\breturn\b)\s*(?:\w+\s*[-+*]\s*)?"
        r"\b(15|12|14|16|18|10|8|6|5|4)\b"
    )
    for _lineno, text, tree in all_injected_blobs():
        if tree is None:
            unparsed += 1
            for raw in text.split("\n"):
                low = raw.lower()
                if not any(k in low for k in relevant):
                    continue
                # a comment cannot encode a runtime shape
                if raw.lstrip().startswith("#"):
                    continue
                m = textual.search(raw)
                if m is None:
                    continue
                value = int(m.group(1))
                if any(
                    value == v and frag in raw
                    for v, frag, _why in ADJUDICATED_SHAPE_LITERALS
                ):
                    continue
                if any(
                    frag in raw for _v, frag, _why in ADJUDICATED_TEXTUAL
                ):
                    continue
                unadjudicated.append(
                    f"{value} in unparseable blob `{raw.strip()[:70]}`"
                )
            continue
        parsed += 1
        lines = text.split("\n")
        # Relevance is a property of the ENCLOSING FUNCTION, not of the line.
        # Filtering on line text alone silently dropped drafter-internal lines
        # like `"rescue_carry_slots": 4 * batch` -- a hole in this detector that
        # its own mutation test found.
        scopes = []
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and any(
                k in fn.name.lower() for k in relevant
            ):
                scopes.append((fn.lineno, fn.end_lineno or fn.lineno))
        for node in ast.walk(tree):
            ctxs = []
            if isinstance(node, ast.BinOp):
                ctxs = [node.left, node.right]
            elif isinstance(node, ast.Compare):
                ctxs = [node.left] + list(node.comparators)
            elif isinstance(node, ast.arguments):
                ctxs = list(node.defaults) + [d for d in node.kw_defaults if d]
            for c in ctxs:
                if not (
                    isinstance(c, ast.Constant)
                    and isinstance(c.value, int)
                    and c.value in magnitudes
                ):
                    continue
                ln = getattr(c, "lineno", 1)
                ctx = lines[ln - 1] if ln - 1 < len(lines) else ""
                in_scope = any(a <= ln <= b for a, b in scopes)
                if not in_scope and not any(
                    k in ctx.lower() for k in relevant
                ):
                    continue
                if any(
                    c.value == v and frag in ctx
                    for v, frag, _why in ADJUDICATED_SHAPE_LITERALS
                ):
                    continue
                if any(
                    c.value in mags and re.search(pat, ctx)
                    for pat, mags, _why in ADJUDICATED_RULES
                ):
                    continue
                unadjudicated.append(f"{c.value} in `{ctx.strip()[:70]}`")
    if unadjudicated:
        return False, sorted(set(unadjudicated))
    return True, (
        f"{parsed} blobs scanned ({unparsed} unparseable, textually checked), "
        f"{len(ADJUDICATED_SHAPE_LITERALS)} literals adjudicated, 0 unreviewed"
    )


# ---------------------------------------------------------------------------
# The REPLAY dimension. shape_literal_scan()'s magnitudes cover COLUMNS (15/6/10)
# and PASSES (4/5). The split also moves REPLAYS 1 -> 2, and neither 1 nor 2 is a
# shape magnitude, so the 15th site was scanned without being flagged.
#
# Adjudication is keyed STRUCTURALLY -- (blob, enclosing function, ordinal) --
# never on the literal's text, and never on the raw line number.
#
#   * not text: `evidence.get("matching_replays") != 1` occurs TWICE in the
#     fixed_flush blob and means OPPOSITE things. At the forward-graph site one
#     replay per step is correct; at the drafter site an armed ungated step
#     replays twice and the literal was the 15th site. The runner's own first
#     draft keyed on text and marked both OK -- reproducing, to the character,
#     the blind spot it was written to expose.
#   * not line numbers: fixing the drafter site added lines and moved every
#     position after it. An allowlist that drifts on every edit trains people to
#     refresh it without reading, which is the same failure one step removed.
REPLAY_NAMES = (
    "replay", "replays", "graph_captures", "capture_count", "prior_replays",
    "matching_replays", "measured_replays", "unmeasured_replays",
)
REPLAY_LITERAL = re.compile(
    r"(?:[=!<>]=|[-+*]|\breturn\b|:)\s*(?:\w+\s*[-+*]\s*)?\b([12])\b"
)
# (enclosing function, ordinal within that function) -> rationale.
# NOT keyed on the blob's start line: round 5's own patcher edits shifted it from
# 39286 to 39294 and turned this pair stale for no semantic reason. Function
# names are unique across all injected blobs (asserted by the lint), so the
# function is the stable half of the position.
ADJUDICATED_REPLAY_POSITIONS = {
    ("_fr13_f32_flush_reconcile", 1):
        "FORWARD-graph evidence: the forward CUDA graph is replayed once per "
        "step regardless of the drafter's pass split. Correct; do not touch. "
        "Its character-identical twin in this same function was the 15th site.",
}


# Category adjudications keyed on the ENCLOSING FUNCTION -- structural identity,
# not the literal's text. Each states why its replay count cannot move with the
# drafter's pass split. A function that holds twins with OPPOSITE verdicts is
# excluded here and adjudicated per-ordinal above.
ADJUDICATED_REPLAY_FUNCTIONS = {
    # --- drafter chain: already derived from the proposal/evidence ---
    "_fr13_fixed32_drafter_graph_replay":
        "derived: prior_replays + 1, and the evidence check is tied to it",
    "_fr13_fixed32_drafter_proposal_end":
        "derived from proposal['graph_replays'] (12th-site fix)",
    "_fr13_fixed32_drafter_graph_capture_begin":
        "graph_replays must be 0 at capture begin in EVERY shape; the capture "
        "counter is derived from the segment",
    "_fr13_fixed32_observed_build_record":
        "holds BOTH chains: the target/provenance pair legitimately replays once "
        "per step, and the drafter pair was tied to runtime['graph_replays'] by "
        "the 13th-site fix",
    # --- target forward graph: one replay per step in every shape ---
    "_fr13_fixed32_forward_graph_registry": "forward graph, once per step",
    "_fr13_fixed32_observed_graph_replay": "forward graph emitter, once per step",
    "_fr13_fixed32_observed_take": "forward graph replay count, once per step",
    # --- committer graph: unaffected by the drafter's pass count ---
    "_fr13_fixed32_observed_commit": "committer graph, once per event",
    "_fr13_fixed32_failure_counts": "committer graph, once per event",
    "_fr13_f32_flush_boot_warm_metrics": "committer replays vs boot capacity",
    "_fr13_fixed32_warm_final_full_postprocess": "committer replays vs capacity",
    # --- draft-head / BM8 levers: gate-variant, but refused with the gate ---
    "_fr13_dh_fp8_note_replay": "draft-head FP8 lever, refused with the gate",
    "_fr13_dh_m32_note_production_replay": "M32 lever, refused with the gate",
    "_fr13_dh_u8_note_production_replay": "U8 lever, refused with the gate",
    "_fr13_dfwd_unified_bm8_production_replay_installed":
        "BM8 production, refused with the gate",
    # --- name collisions: not replay COUNTS at all ---
    "_lumo_tree_canonical_multidraft_sample": "a [:2] slice of replay flags",
    "__init__": "a comment mentioning pre-replay timing",
    "<module>": "a comment mentioning replay stack width",
}


def _enclosing_functions(tree):
    spans = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            spans.append((n.lineno, n.end_lineno or n.lineno, n.name))
    spans.sort(key=lambda s: (s[0], -s[1]))
    return spans


def replay_literal_scan():
    """Every literal 1 or 2 in a replay-counting position, in every blob."""
    unreviewed = []
    for lineno, text, tree in all_injected_blobs():
        spans = _enclosing_functions(tree) if tree is not None else []
        ordinals = {}
        for offset, raw in enumerate(text.split("\n"), start=1):
            low = raw.lower()
            if not any(n in low for n in REPLAY_NAMES):
                continue
            if raw.lstrip().startswith("#"):
                continue
            if REPLAY_LITERAL.search(raw) is None:
                continue
            fn = "<module>"
            for a, b, name in spans:
                if a <= offset <= b:
                    fn = name
            ordinals[fn] = ordinals.get(fn, 0) + 1
            key = (fn, ordinals[fn])
            if key in ADJUDICATED_REPLAY_POSITIONS:
                continue
            if fn in ADJUDICATED_REPLAY_FUNCTIONS:
                continue
            unreviewed.append(
                f"blob@{lineno} {fn}#{ordinals[fn]}: {raw.strip()[:80]}"
            )
    if unreviewed:
        return False, sorted(unreviewed)
    return True, (
        f"{len(ADJUDICATED_REPLAY_POSITIONS)} per-ordinal + "
        f"{len(ADJUDICATED_REPLAY_FUNCTIONS)} per-function adjudications, "
        "0 unreviewed"
    )


# The arm-identity resolvers this family is responsible for. Each entry names
# the file, a human label, and the arm names that MUST appear as explicit
# branches. The detector's real job is the last part of the rule: after the
# explicit branches, an unrecognised arm must REFUSE.
ARM_IDENTITY_RESOLVERS = (
    (FA2_PATCHER, "_FR13_FA2_QROW32_B1_IDENTITIES (injected blob)",
     ("nosplit", "split2", "visibility", "gqa_pair", "gqa_pair_splitk")),
    (REPO / "scripts" / "fr13_fixed32_contract.py",
     "_expected_runtime_fa2_identity B1 table",
     ("nosplit", "split2", "visibility", "gqa_pair", "gqa_pair_splitk")),
    (REPO / "scripts" / "fr13_qrow32_b1_pass_sidecar.py",
     "_SOURCE_STATUS_BY_ARM",
     ("nosplit", "split2", "visibility", "gqa_pair", "gqa_pair_splitk")),
)

# The refusal each resolver must contain. Keyed by phrase rather than by line,
# for the reason ADJUDICATED_REPLAY_POSITIONS spells out: text-keying on the
# thing being checked reproduces the blind spot. These are phrases the RESOLVER
# emits, so a resolver that stopped refusing loses its phrase.
ARM_IDENTITY_REFUSALS = (
    "has no pinned identity for arm",
    "has no pinned binary identity",
    "has no pinned modified-source set",
)

# A textual detector can be fooled by a comment that mentions the right words.
# So the core of this check is BEHAVIOURAL: every resolver is actually called
# with an arm name nobody wrote a branch for, and must raise. Text is used only
# for the half a call cannot see -- that each registry arm is branched
# EXPLICITLY rather than reached through a neighbour's entry.
UNKNOWN_ARM = "__fr14_unknown_arm__"


def _fa2_selector_namespace():
    """The B1 selector helpers, as injected into the served container."""
    import os

    src = FA2_PATCHER.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "_fr13_fa2_qrow32_b1_identity" in node.value
            and "_FR13_FA2_QROW32_B1_IDENTITIES" in node.value
        ):
            namespace = {"os": os}
            exec(compile("import os\n" + node.value, "<fa2_sel>", "exec"),
                 namespace)
            return namespace
    raise SystemExit("FA2 B1 selector blob not found")


def _strip_comments(text: str, suffix: str) -> str:
    """Drop comment lines before looking for an executable guard.

    Found by this family's own mutation test: the launcher check matched the
    phrase "has no pinned identity", the refusal's COMMENT contained that
    phrase, and deleting the actual `echo ... >&2; exit 2` left the check
    passing. A detector that a comment can satisfy is a detector that documents
    a guard instead of finding one -- the same text-keying mistake
    ADJUDICATED_REPLAY_POSITIONS was written to stop.
    """
    marker = "#" if suffix in (".sh", ".py") else "#"
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(marker):
            continue
        out.append(line)
    return "\n".join(out)


def _sidecar_status_table():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fr14_sweep_sidecar_status",
        REPO / "scripts" / "fr13_qrow32_b1_pass_sidecar.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("fr14_sweep_sidecar_status", module)
    spec.loader.exec_module(module)
    return module._SOURCE_STATUS_BY_ARM


def _resolver_probes():
    """(label, callable) pairs, each of which must RAISE on an unknown arm."""
    import importlib.util

    def _load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(name, module)
        spec.loader.exec_module(module)
        return module

    sys.path.insert(0, str(REPO / "scripts"))
    contract = _load(REPO / "scripts" / "fr13_fixed32_contract.py",
                     "fr13_fixed32_contract")
    sidecar = _load(REPO / "scripts" / "fr13_qrow32_b1_pass_sidecar.py",
                    "fr14_sweep_sidecar")
    selectors = _fa2_selector_namespace()
    return (
        ("_fr13_fa2_qrow32_b1_identity (injected blob)",
         lambda: selectors["_fr13_fa2_qrow32_b1_identity"](UNKNOWN_ARM)),
        ("_expected_runtime_fa2_identity (contract)",
         lambda: contract._expected_runtime_fa2_identity({
             "FR13_FA2_QROW32_B1_LIVE_AB_ARM": UNKNOWN_ARM,
             "FR13_FA2_QROW32_B1_SO_SHA256": "0" * 64,
         })),
        ("_source_status (sidecar)",
         lambda: sidecar._source_status(UNKNOWN_ARM)),
    )


# THE TWIN-EQUIVALENCE DETECTOR (site 23). The bash pin-arm resolver and its
# in-container Python twin decide the SAME question -- which arm's pins this
# boot must satisfy -- in two languages, in two files, ~2000 lines apart. The
# Python one carried a comment saying "This mirrors the bash pin case" while
# not mirroring it, and three separate rounds (2.1, 17, 23) were bought by that
# divergence.
#
# So the comment is replaced by an experiment: both resolvers are EXECUTED, on
# the same environments, and their answers compared. A claim of equivalence
# that is only asserted in prose is the thing this family exists to delete.
TWIN_RESOLVER_ENVS = (
    {},
    {"FR13_FA2_QROW32_B1_PRODUCTION_ARM": "nosplit"},
    {"FR13_FA2_QROW32_B1_PRODUCTION_ARM": "gqa_pair"},
    {"FR13_FA2_QROW32_B1_LIVE_AB_ARM": "split2"},
    {"FR13_FA2_QROW32_B1_LIVE_AB_ARM": "visibility"},
    {"FR13_FA2_QROW32_B1_LIVE_AB_ARM": "gqa_pair"},
    {"FR13_FA2_QROW32_B1_LIVE_AB_ARM": "gqa_pair_splitk"},
    # The case that cost round 9.
    {"FR13_FA2_QROW32_B1_TIER_B_ARM": "gqa_pair_splitk"},
    {"FR13_FA2_QROW32_B1_TIMING_ARM": "gqa_pair"},
)


def _bash_pin_arm(launcher, env):
    """Run the launcher's OWN bash resolver lines, unmodified."""
    import subprocess

    text = launcher.read_text()
    start = text.index("  _FR13_FA2_QROW32_B1_PIN_ARM=$FR13_FA2_QROW32_B1_LIVE_AB_ARM")
    end = text.index('  case "$_FR13_FA2_QROW32_B1_PIN_ARM" in', start)
    fragment = text[start:end]
    script = (
        "set -u\n"
        + "".join(
            f'{name}={value!r}\n'.replace("'", '"')
            for name, value in {
                "FR13_FA2_QROW32_B1_LIVE_AB_ARM": env.get(
                    "FR13_FA2_QROW32_B1_LIVE_AB_ARM", ""),
                "FR13_FA2_QROW32_B1_TIER_B_ARM": env.get(
                    "FR13_FA2_QROW32_B1_TIER_B_ARM", ""),
                "FR13_FA2_QROW32_B1_PRODUCTION_ARM": env.get(
                    "FR13_FA2_QROW32_B1_PRODUCTION_ARM", ""),
            }.items()
        )
        + fragment
        + '\nprintf "%s" "$_FR13_FA2_QROW32_B1_PIN_ARM"\n'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if out.returncode != 0:
        return f"REFUSED:{out.stderr.strip()[:60]}"
    return out.stdout


def _python_pin_arm(launcher, env):
    """Run the launcher's OWN in-container python resolver lines, unmodified."""
    import re
    import types

    lines = launcher.read_text().splitlines(keepends=True)
    target = next(
        i for i, line in enumerate(lines)
        if "binary identity is not qualified" in line and "B1" in line
    )
    opener = re.compile(r"<<\s*'?([A-Z]+)'?\s*$")
    start = max(i for i in range(target) if opener.search(lines[i]))
    tag = opener.search(lines[start]).group(1)
    end = next(i for i in range(target, len(lines)) if lines[i].strip() == tag)
    body = "".join(lines[start + 1:end])
    fragment = body[
        body.index("    _b1_pin_selectors = ("): body.index("    expected = {")
    ]

    class _Exit(Exception):
        pass

    namespace = {
        "os": types.SimpleNamespace(environ=dict(env)),
        "SystemExit": _Exit,
    }
    source = "if True:\n" + "\n".join(
        "    " + line for line in fragment.splitlines()
    )
    try:
        exec(compile(source, "<py_resolver>", "exec"), namespace)  # noqa: S102
    except _Exit as exc:
        return f"REFUSED:{str(exc)[:60]}"
    return namespace["b1_pin_arm"]


def pin_arm_resolver_twins():
    """Both resolvers, executed, on every arm the campaign can name."""
    disagreements = []
    for launcher in LAUNCHERS:
        for env in TWIN_RESOLVER_ENVS:
            bash_answer = _bash_pin_arm(launcher, env)
            python_answer = _python_pin_arm(launcher, env)
            if bash_answer != python_answer:
                named = ", ".join(f"{k}={v}" for k, v in env.items()) or "(none)"
                disagreements.append(
                    f"{launcher.name} [{named}]: bash={bash_answer!r} "
                    f"python={python_answer!r}"
                )
    if disagreements:
        return False, "; ".join(disagreements)
    return True, (
        f"{len(LAUNCHERS)} launchers x {len(TWIN_RESOLVER_ENVS)} selector "
        "environments: bash and python pin-arm resolvers agree, by execution"
    )


def arm_identity_resolvers_refuse_unknown_arms():
    """THE 17th-SITE DETECTOR: an identity resolver must refuse, never default.

    Arm S's fifth boot reached _fr13_fa2_qrow32_b1_identity with an arm that
    had no branch, and the function returned split2's pins. Nothing served,
    because the environment's declared sha happened not to match what it handed
    back -- an accident, not a guard. Had it matched, the run would have SERVED
    split-K while ATTESTING the incumbent.

    The rule is not "the split-K arm is present". It is: every arm the registry
    knows has an explicit branch, AND an arm nobody wrote a branch for refuses.
    The first half alone is what the campaign fixed instance by instance for
    seventeen sites; the second half is what makes an eighteenth impossible.
    """
    problems = []
    # Structural, not textual. The first version of this check asked whether
    # each arm name appeared ANYWHERE in the file, which every arm does -- the
    # registry names them all. Renaming a key in the identity TABLE therefore
    # passed. So: read the table's actual keys and compare them with the
    # registry's, which is the invariant that matters.
    selectors = _fa2_selector_namespace()
    registry = set(selectors["_FR13_FA2_QROW32_B1_ARMS"])
    identities = set(selectors["_FR13_FA2_QROW32_B1_IDENTITIES"])
    for arm in sorted(registry - identities):
        problems.append(
            f"fr13_patch_fa2_tree_bias.py: the injected identity table has no "
            f"branch for {arm!r}, which the arm registry knows"
        )
    for arm in sorted(identities - registry):
        problems.append(
            f"fr13_patch_fa2_tree_bias.py: the injected identity table pins "
            f"{arm!r}, which is not a registered arm"
        )
    sidecar_status = _sidecar_status_table()
    for arm in sorted(registry - set(sidecar_status)):
        problems.append(
            f"fr13_qrow32_b1_pass_sidecar.py: _SOURCE_STATUS_BY_ARM has no "
            f"entry for {arm!r}"
        )
    for path, label, required in ARM_IDENTITY_RESOLVERS:
        text = _strip_comments(path.read_text(), path.suffix)
        for arm in required:
            if f'"{arm}"' not in text:
                problems.append(
                    f"{path.name}: {label} has no branch for {arm!r}"
                )
    for label, probe in _resolver_probes():
        try:
            result = probe()
        except Exception:
            continue
        problems.append(
            f"{label} did NOT refuse an unknown arm; it answered with "
            f"{result!r} -- an identity resolver that defaults attests the "
            "wrong artifact"
        )
    if problems:
        return False, "; ".join(problems)
    return True, (
        f"{len(ARM_IDENTITY_RESOLVERS)} arm-identity resolvers: every registry "
        "arm branched explicitly, unknown arms refused by call"
    )


def launcher_pin_cases_refuse_unknown_arms():
    """The bash half of the same rule, across ALL THREE launcher twins.

    A `*)` that asserts an arm's pins is the shell spelling of the same defect,
    and it is worse in one respect: the arm has already passed the allowlist by
    the time it gets here, so the fall-through is the last thing between a
    selected arm and a mounted binary.
    """
    bad = []
    for launcher in LAUNCHERS:
        text = launcher.read_text()
        if 'case "$_FR13_FA2_QROW32_B1_PIN_ARM" in' not in text:
            bad.append(f"{launcher.name}: no B1 pin case at all")
            continue
        case = text[text.index('case "$_FR13_FA2_QROW32_B1_PIN_ARM" in'):]
        case = case[: case.index("\n  esac")]
        case = _strip_comments(case, launcher.suffix)
        for arm in ("visibility", "gqa_pair", "gqa_pair_splitk"):
            if f"    {arm})" not in case:
                bad.append(f"{launcher.name}: pin case has no {arm}) branch")
        if 'nosplit|split2|""' not in case:
            bad.append(
                f"{launcher.name}: the incumbent arms are not named explicitly"
            )
        # The refusal has to be EXECUTABLE: an echo to stderr and a non-zero
        # exit, in the default branch. Comments were stripped above.
        default = case[case.rindex("    *)"):]
        if "has no pinned identity" not in default or "exit 2" not in default:
            bad.append(
                f"{launcher.name}: pin case default does not refuse an "
                "unrecognised arm"
            )
    if bad:
        return False, "; ".join(bad)
    return True, (
        f"{len(LAUNCHERS)} launcher pin cases: every arm branched, "
        "unrecognised arms refused"
    )


PAIRS = (
    ("contract <-> injected blob (step shapes)", "contract/consumer",
     lambda blob, topo: pair_contract_vs_blob(blob, topo)),
    ("contract <-> injected blob (graph_captures)", "contract/consumer",
     lambda blob, topo: pair_contract_vs_blob_captures(blob, topo)),
    ("contract <-> census validator", "contract/consumer",
     lambda blob, topo: pair_contract_vs_census(topo)),
    ("census drafter <-> census drafter_runtime", "half/half",
     lambda blob, topo: pair_census_halves()),
    ("proposal_end census half <-> runtime half", "half/half",
     lambda blob, topo: pair_emitter_halves(blob)),
    ("tree-attn observer <-> capture context", "emitter/validator",
     lambda blob, topo: pair_observer_vs_context(blob)),
    ("launcher family A <-> launcher family B", "twin/twin",
     lambda blob, topo: pair_launcher_twins()),
    ("launcher sidecar <-> gate parser", "bash/python",
     lambda blob, topo: pair_bash_cfg_vs_python_parser()),
    ("topology contract <-> decide_fixed32", "contract/consumer",
     lambda blob, topo: pair_topology_vs_drafter()),
    ("handoff contract <-> injected blob", "contract/consumer",
     lambda blob, topo: pair_handoff_contract_vs_blob(blob, topo)),
    ("pack identity <-> both step shapes", "shape-literal",
     lambda blob, topo: pair_pack_identity_under_both_shapes(topo)),
    ("shape literals across ALL injected blobs", "shape-literal",
     lambda blob, topo: shape_literal_scan()),
    ("replay-count literals across ALL injected blobs", "shape-literal",
     lambda blob, topo: replay_literal_scan()),
    ("arm-identity resolvers <-> unknown-arm refusal", "fallback-pattern",
     lambda blob, topo: arm_identity_resolvers_refuse_unknown_arms()),
    ("launcher pin cases <-> unknown-arm refusal", "fallback-pattern",
     lambda blob, topo: launcher_pin_cases_refuse_unknown_arms()),
    ("bash pin-arm resolver <-> python twin", "twin/twin",
     lambda blob, topo: pin_arm_resolver_twins()),
)


def sweep():
    blob = injected_blob()
    topo = _topology()
    rows = []
    for name, kind, check in PAIRS:
        ok, detail = check(blob, topo)
        rows.append(
            {"pair": name, "kind": kind, "ok": bool(ok), "detail": detail}
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = sweep()
    stale = [r for r in rows if not r["ok"]]
    for r in rows:
        print(f"[{'OK  ' if r['ok'] else 'STALE'}] {r['pair']:48s} {r['detail']}")
    print(f"\n{len(rows)} pairs enumerated, {len(stale)} stale")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "schema": "fr14.paired_contract_sweep.v1",
                    "pairs": rows,
                    "n_pairs": len(rows),
                    "n_stale": len(stale),
                },
                indent=1,
            )
        )
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
