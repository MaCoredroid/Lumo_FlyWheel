"""Tests for the T4 plan-structure pre-drafter decision core."""

from __future__ import annotations

import pytest

from lumo_flywheel_serving.plan_structure_drafter import (
    DraftProposal,
    PlanFingerprint,
    PlanRegistry,
    looks_like_plan_reemission,
    propose,
)


PLAN_A = """## Plan
1. Read the failing test
2. Identify the root cause
3. Apply the fix
4. Re-run the test
"""

PLAN_A_REVISED = """## Plan
1. Looked at the failing test (done)
2. Found the bug in widget.py
3. Patched widget.py with the fix
4. Tests now passing
"""

PLAN_B_DIFFERENT_SHAPE = """### TODO
- [ ] Read the failing test
- [ ] Identify the root cause
- [ ] Apply the fix
"""

NOT_A_PLAN = """The agent then said hello world.
This is a paragraph of prose with no structure.
"""


def test_fingerprint_extracts_numbered_list_skeleton() -> None:
    fp = PlanFingerprint.fingerprint(PLAN_A)
    assert fp.line_count == 5  # 1 header + 4 numbered
    assert fp.structure[0] == "## "
    assert fp.structure[1] == "1. "
    assert fp.structure[-1] == "4. "


def test_fingerprint_same_shape_different_content_collides() -> None:
    fp_a = PlanFingerprint.fingerprint(PLAN_A)
    fp_a_rev = PlanFingerprint.fingerprint(PLAN_A_REVISED)
    assert fp_a == fp_a_rev


def test_fingerprint_different_shapes_dont_collide() -> None:
    fp_a = PlanFingerprint.fingerprint(PLAN_A)
    fp_b = PlanFingerprint.fingerprint(PLAN_B_DIFFERENT_SHAPE)
    assert fp_a != fp_b


def test_is_plan_shaped_rejects_random_prose() -> None:
    fp = PlanFingerprint.fingerprint(NOT_A_PLAN)
    assert not fp.is_plan_shaped()


def test_is_plan_shaped_rejects_one_line_emissions() -> None:
    fp = PlanFingerprint.fingerprint("1. Just one line\n")
    assert not fp.is_plan_shaped()


def test_is_plan_shaped_accepts_numbered_list() -> None:
    fp = PlanFingerprint.fingerprint(PLAN_A)
    assert fp.is_plan_shaped()


def test_is_plan_shaped_accepts_checklist() -> None:
    fp = PlanFingerprint.fingerprint(PLAN_B_DIFFERENT_SHAPE)
    assert fp.is_plan_shaped()


def test_registry_observe_increments_count() -> None:
    reg = PlanRegistry()
    fp1 = reg.observe(PLAN_A)
    assert fp1 is not None
    assert reg.counts[fp1] == 1
    reg.observe(PLAN_A_REVISED)  # same shape
    assert reg.counts[fp1] == 2


def test_registry_observe_skips_non_plan_emissions() -> None:
    reg = PlanRegistry()
    assert reg.observe(NOT_A_PLAN) is None
    assert len(reg.counts) == 0


def test_registry_activation_requires_min_emissions() -> None:
    reg = PlanRegistry()
    fp1 = reg.observe(PLAN_A)
    assert fp1 is not None
    assert not reg.is_activated(fp1)
    reg.observe(PLAN_A)
    assert not reg.is_activated(fp1)
    reg.observe(PLAN_A)
    assert reg.is_activated(fp1)  # third emission activates


def test_registry_best_activated_picks_highest_count() -> None:
    reg = PlanRegistry()
    for _ in range(4):
        reg.observe(PLAN_A)
    for _ in range(3):
        reg.observe(PLAN_B_DIFFERENT_SHAPE)
    best = reg.best_activated_fingerprint()
    assert best is not None
    assert best == PlanFingerprint.fingerprint(PLAN_A)
    assert reg.counts[best] == 4


def test_registry_best_activated_returns_none_below_threshold() -> None:
    reg = PlanRegistry()
    reg.observe(PLAN_A)
    reg.observe(PLAN_A)  # only 2 — below threshold
    assert reg.best_activated_fingerprint() is None


def test_looks_like_plan_reemission_recognizes_triggers() -> None:
    assert looks_like_plan_reemission("...\nupdated plan:\n")
    assert looks_like_plan_reemission("Revised plan after step 2\n")
    assert looks_like_plan_reemission("- [x] something\nplan:\n")
    assert not looks_like_plan_reemission("the user wants me to apply patches")


def test_looks_like_plan_reemission_only_recent_trigger() -> None:
    """A trigger phrase from 1KB back doesn't count — only the last
    ~512 chars of context."""

    pad = "x " * 600  # ~1200 chars
    text = "updated plan:\n" + pad + "\nthen something else"
    assert not looks_like_plan_reemission(text)


def test_propose_returns_none_when_no_trigger() -> None:
    reg = PlanRegistry()
    for _ in range(5):
        reg.observe(PLAN_A)
    assert propose(reg, "the user said hello") is None


def test_propose_returns_none_when_below_threshold() -> None:
    reg = PlanRegistry()
    reg.observe(PLAN_A)
    reg.observe(PLAN_A)  # only 2
    assert propose(reg, "...\nupdated plan:\n") is None


def test_propose_returns_skeleton_when_activated() -> None:
    reg = PlanRegistry()
    for _ in range(3):
        reg.observe(PLAN_A)
    proposal = propose(reg, "I'll provide an updated plan:\n")
    assert isinstance(proposal, DraftProposal)
    assert proposal.confidence == pytest.approx(0.7)
    assert proposal.text.startswith("## \n1. \n2. \n3. \n4. \n")
    assert "emissions=3" in proposal.reason


def test_render_skeleton_terminates_each_element_with_newline() -> None:
    fp = PlanFingerprint.fingerprint(PLAN_A)
    skeleton = fp.render_skeleton()
    assert skeleton.count("\n") == fp.line_count
    assert "1. \n" in skeleton
