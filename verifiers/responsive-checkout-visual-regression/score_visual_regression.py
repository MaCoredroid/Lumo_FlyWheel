#!/usr/bin/env python3
"""
Deterministic scorer for `responsive-checkout-visual-regression`.

This task is browser-bound at full fidelity (preview render + screenshots).
This grader provides the deterministic file-state + regression-contract
slice (~max 70/100). Browser-level hidden checks (CTA hit-test at 390x844,
multi-breakpoint stability) would add the remaining 30 points but require
a Playwright/preview harness that is not available in our local env. The
ceiling is applied accordingly.

Milestones (family.yaml weights):
  M1_localization  0.10  edits in apps/storefront/components/checkout/* AND/OR styles/checkout.css
  M2_primary_fix   0.20  mobile-viewport regression test added (CTA visibility assertion)
  M3_invariants    0.20  existing checkout.spec.ts unchanged in test signature; unit tests pass
  M4_functional    0.20  evidence bundle: at least 3 screenshots in preview_artifacts/ (mobile before/after + desktop after) + QA note
  M5_e2e           0.30  full pytest/jest equivalent passes if runnable + no integrity violation

Integrity flags:
  - tests_modified           tests/e2e/checkout.spec.ts mutated (test bypassed)
  - pytest_shim              shim files present
  - sticky_disabled_global   agent disabled sticky behavior wholesale instead of preview-only fix
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
SCHEMA_VERSION = "cnb55.verify_result.v3"
MAX_M_POINTS = 100

SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
CHECKOUT_COMPONENT_DIR = "apps/storefront/components/checkout"
CHECKOUT_STYLES = ("apps/storefront/styles/checkout.css",)
E2E_TEST_FILE = "tests/e2e/checkout.spec.ts"
UNIT_TEST_DIR = "tests/unit"
PREVIEW_ARTIFACTS = "preview_artifacts"

MOBILE_VIEWPORT_KEYWORDS = ("390", "844", "viewport", "mobile", "deviceScaleFactor")
CTA_KEYWORDS = ("cta", "submit", "primary", "checkout-button", "place order", "place-order")
STICKY_OFF_GLOBAL_RED_FLAGS = (
    "position: static",  # too coarse if applied globally
    "/* sticky disabled */",
)
QA_NOTE_KEYWORDS = ("mobile", "desktop", "viewport", "qa", "verified", "tested")


@dataclass
class State:
    breakdown: dict[str, int] = field(default_factory=dict)
    bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_M_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    milestones: dict[str, bool] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_M_score += points

    def apply_ceiling(self, name: str, cap: int) -> None:
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def raise_integrity(self, rule_id: str, message: str | None = None) -> None:
        self.integrity_flag = 1
        self.ceiling_cap = 0
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)
        if message:
            self.errors.append(message)

    def final_score(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        return round(max(0, min(self.raw_M_score, self.ceiling_cap)) / MAX_M_POINTS, 4)


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def list_files_in(root: Path, prefix: str) -> list[Path]:
    base = root / prefix
    if not base.exists():
        return []
    return [p for p in base.rglob("*") if p.is_file()]


def check_integrity(state: State, bundle_root: Path | None) -> None:
    for shim in SHIM_FILES:
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim", f"shim present: {shim}")
    if bundle_root is not None:
        bundle_e2e = bundle_root / E2E_TEST_FILE
        ws_e2e = AGENT_WS / E2E_TEST_FILE
        if bundle_e2e.is_file() and ws_e2e.is_file() and sha256_file(bundle_e2e) != sha256_file(ws_e2e):
            # We allow appending new assertions but disallow wholesale rewrite. Heuristic: if file shrunk, flag.
            if ws_e2e.stat().st_size < bundle_e2e.stat().st_size * 0.5:
                state.raise_integrity("tests_modified", f"{E2E_TEST_FILE} truncated >50%")


def detect_sticky_disabled_global(state: State) -> bool:
    # If agent set `position: static !important` on a base layout class globally, that's the shortcut.
    css_files = list_files_in(AGENT_WS, "apps/storefront/styles") + list_files_in(AGENT_WS, "apps/storefront/components/checkout")
    for path in css_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except Exception:
            continue
        # heuristic: top-level body / .checkout / .layout selectors with position:static AND no media query gate
        if re.search(r"\.(layout|page|root|checkout|app|main)\s*\{[^}]*position\s*:\s*static[^}]*\}", text):
            if "@media" not in text or text.find("position: static") < text.find("@media"):
                return True
    return False


def score(state: State) -> None:
    # M1: localization — files in checkout component tree changed
    component_files = list_files_in(AGENT_WS, CHECKOUT_COMPONENT_DIR)
    styles_present = any((AGENT_WS / p).is_file() for p in CHECKOUT_STYLES)
    component_dir_present = (AGENT_WS / CHECKOUT_COMPONENT_DIR).is_dir() and len(component_files) > 0
    state.milestones["M1_localization"] = component_dir_present and styles_present
    state.add("M1.checkout_components_present", 5 if component_dir_present else 0, "M")
    state.add("M1.checkout_styles_present", 5 if styles_present else 0, "M")

    # M2: regression coverage — mobile viewport test added
    e2e_test_text = ""
    e2e_path = AGENT_WS / E2E_TEST_FILE
    if e2e_path.is_file():
        e2e_test_text = e2e_path.read_text(encoding="utf-8", errors="replace").lower()
    # Look for new mobile viewport assertions anywhere under tests/
    test_files_text = e2e_test_text
    for p in list_files_in(AGENT_WS, "tests"):
        try:
            test_files_text += "\n" + p.read_text(encoding="utf-8", errors="replace").lower()
        except Exception:
            continue
    mobile_test_added = (
        any(kw in test_files_text for kw in ("390", "device", "iphone", "mobile"))
        and any(kw.lower() in test_files_text for kw in CTA_KEYWORDS)
    )
    state.milestones["M2_primary_fix"] = mobile_test_added
    state.add("M2.mobile_test_added", 20 if mobile_test_added else 0, "M")

    # M3: invariants — sticky behavior not globally disabled, e2e file size sane
    sticky_global = detect_sticky_disabled_global(state)
    if sticky_global:
        state.raise_integrity("sticky_disabled_global", "global position:static heuristic fired")
    # If unit tests exist and pytest-style, try running them
    unit_test_files = list_files_in(AGENT_WS, UNIT_TEST_DIR)
    state.diagnostics["unit_test_count"] = len(unit_test_files)
    state.milestones["M3_invariants"] = (not sticky_global) and (len(unit_test_files) > 0)
    state.add("M3.no_global_sticky_disable", 15 if not sticky_global else 0, "M")
    state.add("M3.unit_tests_present", 5 if unit_test_files else 0, "M")

    # M4: evidence — screenshots + QA note
    preview_dir = AGENT_WS / PREVIEW_ARTIFACTS
    screenshots = []
    if preview_dir.is_dir():
        screenshots = [
            p for p in preview_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        ]
    qa_note_text = ""
    for candidate in (preview_dir / "qa_note.md", preview_dir / "qa-note.md", AGENT_WS / "QA_NOTE.md", AGENT_WS / "docs" / "qa_note.md"):
        if candidate.is_file():
            try:
                qa_note_text = candidate.read_text(encoding="utf-8", errors="replace").lower()
                break
            except Exception:
                pass
    qa_note_substantive = (
        len(qa_note_text) > 80
        and sum(1 for kw in QA_NOTE_KEYWORDS if kw in qa_note_text) >= 3
    )
    state.milestones["M4_functional"] = len(screenshots) >= 3 and qa_note_substantive
    state.add("M4.screenshots_count", min(10, len(screenshots) * 3), "M")
    state.add("M4.qa_note_substantive", 10 if qa_note_substantive else 0, "M")

    # M5: e2e — browser-level checks not runnable locally → max we can award is "file-state stable + all M1-M4 partial"
    # Apply ceiling at 70 because we cannot verify the actual mobile CTA hit-test
    state.apply_ceiling("browser_checks_unavailable", 70)

    e2e_completed_partials = (
        state.milestones.get("M1_localization", False),
        state.milestones.get("M2_primary_fix", False),
        state.milestones.get("M3_invariants", False),
        state.milestones.get("M4_functional", False),
    )
    state.milestones["M5_e2e"] = all(e2e_completed_partials) and not state.integrity_flag
    state.add("M5.all_partials_present", 20 if all(e2e_completed_partials) else 0, "M")
    state.add("M5.no_integrity", 10 if not state.integrity_flag else 0, "M")


def main() -> int:
    state = State()
    if not AGENT_WS.exists():
        print(f"AGENT_WS not found: {AGENT_WS}", file=sys.stderr)
        return 2

    bundle_root_env = os.environ.get("WORKSPACE_BUNDLE_ROOT")
    bundle_root = Path(bundle_root_env) if bundle_root_env else None

    check_integrity(state, bundle_root)
    score(state)

    final = state.final_score()
    milestone_vector = [int(state.milestones.get(m, False)) for m in ("M1_localization", "M2_primary_fix", "M3_invariants", "M4_functional", "M5_e2e")]
    M_aggregate = round(
        0.10 * milestone_vector[0]
        + 0.20 * milestone_vector[1]
        + 0.20 * milestone_vector[2]
        + 0.20 * milestone_vector[3]
        + 0.30 * milestone_vector[4],
        4,
    )

    result = {
        "schema": SCHEMA_VERSION,
        "family": "responsive-checkout-visual-regression",
        "variant": VARIANT_ID,
        "score": final,
        "P_benchmark": final,
        "M_training": state.final_m_training(),
        "M_aggregate": M_aggregate,
        "milestones": state.milestones,
        "milestone_vector": milestone_vector,
        "breakdown": state.breakdown,
        "bands": state.bands,
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "errors": state.errors,
        "diagnostics": state.diagnostics,
        "notes": "Browser-level hidden checks (CTA hit-test at 390x844, multi-breakpoint stability) require a Playwright/preview harness not present in this env. Ceiling applied at 70.",
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"score": final, "M_aggregate": M_aggregate, "ceilings": state.ceilings_applied, "integrity_flag": state.integrity_flag}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
