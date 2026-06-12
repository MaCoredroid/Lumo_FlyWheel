"""FIX-1 wiring gate: FR13_DRAFTER_SINGLE_LOGITS (FR13_B1_SPEED_ATTRIBUTION_BIND.md).

The caterpillar drafter paid the full-vocab bf16 lm-head read TWICE per step:
explicit compute_logits for top-2 packing AND self._greedy_sample, which
recomputes compute_logits in live vLLM eagle.py (measured +81.94 ms/draft of
the +92.25 ms/draft B=1 GPU-busy delta). FIX-1 takes draft tokens as argmax of
the SAME already-computed logits tensor, deletes the verified-unused root topk,
and skips the loop topk only in spine-only mode (cat9 needs top-2 packing).

Playbook class 9 (silent fallback / vacuous instrument): these are text-level
asserts that the flag exists end-to-end — default ON in the patch consumption,
exact legacy path preserved under OFF (the A/B instrument), an engagement
needle in the boot log, and docker -e passthrough in BOTH launchers (both
apply fr10_phase4_patch_vllm_tree_gdn.py in-container).
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHERS = [
    REPO / "scripts" / "fr10_launch_speed_server.sh",
    REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh",
]


def _eagle_consumption_new_snippet() -> str:
    """Extract the drafter replacement string literal from the patch source."""
    tree = ast.parse(PATCH.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_patch_eagle_tree_consumption_verify"
        ):
            for stmt in ast.walk(node):
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "new"
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                    and "FR13_DRAFTER_SINGLE_LOGITS" in stmt.value.value
                ):
                    return stmt.value.value
    raise AssertionError(
        "drafter replacement string with FR13_DRAFTER_SINGLE_LOGITS not found"
    )


def test_flag_default_on_and_branch_guarded_in_patch() -> None:
    snippet = _eagle_consumption_new_snippet()

    # Default ON; OFF only via explicit env or use_local_argmax_reduction
    # (get_top_tokens has different selection semantics -> exact legacy).
    assert (
        'os.environ.get("FR13_DRAFTER_SINGLE_LOGITS", "1") == "1"' in snippet
    )
    assert (
        'and not getattr(self, "use_local_argmax_reduction", False)' in snippet
    )

    # Reuse path: draft tokens = argmax of the SAME logits tensor, both steps.
    assert "draft_token_ids = _fr10_logits.argmax(dim=-1)" in snippet
    assert "draft_token_ids = _fr10_step_logits.argmax(dim=-1)" in snippet

    # Loop topk skipped ONLY in spine-only mode (cat9 keeps top-2 packing).
    assert "if not _fr10_is_spine_only:" in snippet

    # Engagement needle (class 9): boot log records which path engaged.
    assert "FR13_DRAFTER_SINGLE_LOGITS drafter path engaged" in snippet
    # Flag-state header in the FR10_METRICS artifact.
    assert '"single_logits": bool(_fr13_single_logits),' in snippet


def test_legacy_double_logits_path_preserved_verbatim_under_off() -> None:
    snippet = _eagle_consumption_new_snippet()

    # The A/B instrument: flag OFF must take the EXACT legacy statements.
    legacy_root = (
        "                _fr10_logits = self.model.compute_logits(sample_hidden_states)\n"
        "                _fr10_top2 = torch.topk(_fr10_logits, 2, dim=-1).indices\n"
        "                draft_token_ids = self._greedy_sample(sample_hidden_states)\n"
    )
    assert legacy_root in snippet

    legacy_loop = (
        "                    _fr10_step_logits = self.model.compute_logits(\n"
        "                        last_hidden_states[:batch_size]\n"
        "                    )\n"
        "                    _fr10_step_top2 = torch.topk(_fr10_step_logits, 2, dim=-1).indices\n"
        "                    draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])\n"
        "                    _fr10_spine_tokens.append(draft_token_ids)\n"
        "                    _fr10_leaf_tokens.append(_fr10_step_top2[:, 1])\n"
    )
    assert legacy_loop in snippet

    # No _greedy_sample call outside the legacy (else) branches: exactly the
    # two legacy call sites remain.
    assert snippet.count("self._greedy_sample(") == 2


def test_drafter_replacement_snippet_is_valid_method_body() -> None:
    snippet = _eagle_consumption_new_snippet()
    # The replacement is propose()-body code at 8-space indentation; it must
    # compile when re-homed under a method definition.
    wrapped = "class _C:\n    def propose(self):\n" + snippet
    compile(wrapped, "<fr13_drafter_single_logits_snippet>", "exec")


def test_both_launchers_default_and_pass_the_flag() -> None:
    for launcher in LAUNCHERS:
        text = launcher.read_text()
        assert (
            "FR13_DRAFTER_SINGLE_LOGITS=${FR13_DRAFTER_SINGLE_LOGITS:-1}" in text
        ), launcher
        assert (
            '-e FR13_DRAFTER_SINGLE_LOGITS="$FR13_DRAFTER_SINGLE_LOGITS" \\' in text
        ), launcher
