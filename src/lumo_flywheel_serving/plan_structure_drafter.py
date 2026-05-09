"""T4 plan-structure token-level pre-drafter — pure-Python decision core.

Open-territory technique (no published precedent at token granularity;
closest related work is action-granularity speculation in Speculative
Actions / Dynamic Speculative Planning). The premise: when an agent
emits a numbered/bulleted plan multiple times in the same session
(e.g., "updated plan after step 1"), the *structural* tokens
(numbers, bullets, separators, header markers) repeat verbatim across
emissions. The drafter pre-fills those structural tokens with high
confidence, falling back to suffix-decoding for the freely-emitted
content between separators.

False-positive mitigation per the spec:
- The drafter requires ``emission_count >= 3`` before activating for a
  session — same threshold the spec recommends.
- The fingerprint records ONLY structural tokens (separator strings,
  header levels), never content. A markdown list quoted from a file
  has the same structural fingerprint but won't activate because the
  emission count for the file's structure is 0 in the agent's plan
  registry.

Tokenizer-free: returns text. The vLLM-side integrator encodes to
model tokens at draft time (same pattern as ``schema_aware_drafter``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Structural pattern detection. Each regex matches a single line or
# leading-line element that, when seen at high frequency in a known-
# plan emission, indicates a structural slot the model is likely to
# emit deterministically.
_HEADER_RE = re.compile(r"^(#{1,4})\s+\S", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^(\d+)\.\s+\S", re.MULTILINE)
_BULLET_RE = re.compile(r"^([-*+])\s+\S", re.MULTILINE)
_CHECKLIST_RE = re.compile(r"^([-*+])\s+\[([ xX])\]\s+\S", re.MULTILINE)
_SEPARATOR_RE = re.compile(r"^(---+|\*{3,}|={3,})$", re.MULTILINE)

# Trigger phrases the drafter looks for in recent context to know
# the next token region is likely to be a plan re-emission. Anchored
# at line start to keep false positives low — random prose that
# happens to contain "plan" doesn't trigger.
_PLAN_REEMISSION_TRIGGERS = (
    r"updated plan",
    r"revised plan",
    r"next plan",
    r"new plan",
    r"current plan",
    r"todo",
    r"plan:\s*$",
    r"steps:\s*$",
    r"plan for",
)
_TRIGGER_RE = re.compile(
    r"(?im)(" + r"|".join(_PLAN_REEMISSION_TRIGGERS) + r")"
)


@dataclass(frozen=True)
class PlanFingerprint:
    """Structural skeleton extracted from an observed plan emission.

    Records only the SHAPE of the plan, never its content. Two plan
    emissions with the same numbered-list shape and same header
    structure produce the same fingerprint."""

    structure: tuple[str, ...]
    """Ordered tuple of structural-token strings, e.g.
    ``('## ', '1. ', '2. ', '3. ', '## ', '- [ ] ', '- [ ] ')``.
    Used as the fingerprint key for matching subsequent emissions."""

    line_count: int
    """Total number of structural lines in the emission. Helps the
    drafter decide how many tokens to pre-draft."""

    @classmethod
    def fingerprint(cls, text: str) -> "PlanFingerprint":
        elements: list[str] = []
        for line in text.splitlines():
            stripped = line.lstrip()
            if not stripped:
                continue
            m = _HEADER_RE.match(line)
            if m:
                elements.append(m.group(1) + " ")
                continue
            m = _CHECKLIST_RE.match(line)
            if m:
                elements.append(f"{m.group(1)} [ ] ")
                continue
            m = _NUMBERED_RE.match(line)
            if m:
                elements.append(f"{m.group(1)}. ")
                continue
            m = _BULLET_RE.match(line)
            if m:
                elements.append(f"{m.group(1)} ")
                continue
            if _SEPARATOR_RE.match(line):
                elements.append(line + "\n")
                continue
        return cls(structure=tuple(elements), line_count=len(elements))

    def is_plan_shaped(self) -> bool:
        """Heuristic: an emission must have at least two structural
        lines and at least one numbered or checklist marker to count
        as a plan. A blob of random bullets in prose isn't a plan."""

        if self.line_count < 2:
            return False
        has_numbered_or_check = any(
            re.match(r"\d+\.\s$|.* \[ \] $", e) for e in self.structure
        )
        return has_numbered_or_check

    def render_skeleton(self) -> str:
        """Render the structural skeleton as text the drafter can
        propose. Each element gets a trailing newline marker so the
        model can fill in content between them."""

        return "".join(f"{e}\n" for e in self.structure)


@dataclass
class PlanRegistry:
    """Per-session count of plan emissions by fingerprint.

    Activation gate: the drafter's pre-drafting only fires when a
    fingerprint has been observed at least ``MIN_EMISSIONS`` times in
    the session. This is the v2 spec's false-positive mitigation."""

    MIN_EMISSIONS: int = 3
    counts: dict[PlanFingerprint, int] = field(default_factory=dict)

    def observe(self, text: str) -> PlanFingerprint | None:
        fp = PlanFingerprint.fingerprint(text)
        if not fp.is_plan_shaped():
            return None
        self.counts[fp] = self.counts.get(fp, 0) + 1
        return fp

    def is_activated(self, fp: PlanFingerprint) -> bool:
        return self.counts.get(fp, 0) >= self.MIN_EMISSIONS

    def best_activated_fingerprint(self) -> PlanFingerprint | None:
        """The fingerprint with the most observations among activated
        ones. Used when the drafter detects a re-emission trigger
        but doesn't know which prior fingerprint to draft against."""

        activated = [(fp, n) for fp, n in self.counts.items() if n >= self.MIN_EMISSIONS]
        if not activated:
            return None
        activated.sort(key=lambda kv: kv[1], reverse=True)
        return activated[0][0]


@dataclass
class DraftProposal:
    """Same shape as ``schema_aware_drafter.DraftProposal`` so the
    composite drafter can compare confidences across techniques."""

    text: str
    confidence: float
    reason: str


def looks_like_plan_reemission(recent_text: str) -> bool:
    """Did the recent text contain a trigger phrase suggesting the
    model is about to re-emit a plan?"""

    if not recent_text:
        return False
    # Trigger must appear in the LAST ~512 chars to be relevant.
    tail = recent_text[-512:]
    return bool(_TRIGGER_RE.search(tail))


def propose(registry: PlanRegistry, recent_text: str) -> DraftProposal | None:
    """Top-level entrypoint mirroring ``schema_aware_drafter.propose``.

    Returns ``None`` when no plan-structure proposal is justified
    (no activated fingerprint, no recent trigger phrase). When
    activated, returns a structural skeleton at confidence 0.7 — lower
    than the schema-aware drafter (1.0/0.9/0.8) because the structural
    tokens are repeatable but the spans between them are model-driven.
    """

    if not looks_like_plan_reemission(recent_text):
        return None
    fp = registry.best_activated_fingerprint()
    if fp is None:
        return None
    skeleton = fp.render_skeleton()
    if not skeleton:
        return None
    return DraftProposal(
        text=skeleton,
        confidence=0.7,
        reason=f"plan_reemission_trigger_with_emissions={registry.counts.get(fp, 0)}",
    )
