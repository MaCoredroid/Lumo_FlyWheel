#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import textwrap
from pathlib import Path


FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parents[2]
VERIFIER_ROOT = REPO / "verifiers" / "transcript-merge-regression"
VERIFIER_DATA = REPO / "verifier_data" / "transcript-merge-regression"

ALLOWED_WRITE_PATHS = [
    "replay/merge.py",
    "replay/render.py",
    "replay/incident_summary.py",
    "reports/incidents/transcript-merge.md",
    "tests/test_merge.py",
    "tests/test_render.py",
    "tests/test_incident_summary.py",
]

VARIANTS = [
    {
        "id": "v1-clean-baseline",
        "title": "clean baseline",
        "extra_relpath": "repo_evidence/baseline_readme.md",
        "extra_text": """\
        # Baseline evidence

        The visible replay shows two distinct `fetch_weather` tool-output blocks.
        They share a tool name but not an event identity.
        """,
        "stale_note_hint": "The visible replay looks like duplicate tool lines, so operator render filtering remains an acceptable mitigation until the reducer is rewritten.",
        "variant_note_phrases": [
            "visible replay",
            "same tool name",
        ],
    },
    {
        "id": "v2-noisy-distractor",
        "title": "noisy distractor",
        "extra_relpath": "repo_evidence/render_dedupe_memo.md",
        "extra_text": """\
        # Render dedupe memo (stale)

        Benign duplicate debug lines showed up during a trace replay, but this
        memo is stale and should not be treated as the root-cause analysis.
        """,
        "stale_note_hint": "The benign duplicate debug lines suggest a render-layer dedupe pass is safe enough for this variant.",
        "variant_note_phrases": [
            "benign duplicate debug lines",
            "render_dedupe_memo.md",
        ],
    },
    {
        "id": "v3-dirty-state",
        "title": "dirty state",
        "extra_relpath": "repo_evidence/in_progress_patch/drop_post_completion.diff",
        "extra_text": """\
        diff --git a/replay/merge.py b/replay/merge.py
        @@
        -    if seen_completion and event.get("debug_only"):
        -        continue
        +    if seen_completion:
        +        continue

        # Abandoned patch: it drops every post-completion fragment, including
        # legitimate deferred tool output. Do not complete this patch.
        """,
        "stale_note_hint": "An abandoned patch already drops everything after completion, which is good enough if duplicate lines disappear from the rendered transcript.",
        "variant_note_phrases": [
            "abandoned patch",
            "drop_post_completion.diff",
        ],
    },
    {
        "id": "v4-multi-corpus-objective",
        "title": "multi-corpus objective drift",
        "extra_relpath": "release_context/reliability_directive.md",
        "extra_text": """\
        # Reliability directive

        The incident summary is consumed by follow-on automation. Preserve the
        summary's semantics by grounding it in merged events, not rendered lines.
        """,
        "stale_note_hint": "Because the reliability directive only mentions duplicate operator output, render filtering is still a safe first fix.",
        "variant_note_phrases": [
            "reliability directive",
            "merged events",
        ],
    },
    {
        "id": "v5-recovery-in-thread",
        "title": "recovery in thread",
        "extra_relpath": "incident_context/INC-2410-rollback.md",
        "extra_text": """\
        # INC-2410 rollback

        The prior hotfix filtered duplicate tool lines after rendering. It was
        rolled back because legitimate deferred tool output disappeared from the
        operator transcript and the incident summary under-counted tool blocks.
        """,
        "stale_note_hint": "The rollback only affected a noisy tail case, so a narrower render filter is still acceptable here.",
        "variant_note_phrases": [
            "INC-2410",
            "rollback",
        ],
    },
]

BUGGY_MERGE = """\
from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def _event_key(event: dict) -> object:
    if event.get("kind") == "tool_output":
        # BUG: tool outputs need stable event identity, not role/name grouping.
        return (event.get("role"), event.get("tool_name"))
    return event.get("event_id") or f"{event.get('kind')}:{event.get('sequence', 0)}"


def merge_records(records: list[dict]) -> list[dict]:
    merged: list[dict] = []
    by_key: dict[object, dict] = {}
    seen_completion = False
    for event in sorted(
        records,
        key=lambda row: (
            row.get("sequence", 0),
            row.get("chunk_index", 0),
            row.get("event_id", ""),
        ),
    ):
        if event.get("kind") == "response.completed":
            seen_completion = True
            continue
        key = _event_key(event)
        existing = by_key.get(key)
        if existing is None:
            current = dict(event)
            current["content_parts"] = [event.get("content", "")]
            by_key[key] = current
            merged.append(current)
        else:
            existing["content_parts"].append(event.get("content", ""))
            existing["sequence"] = max(existing.get("sequence", 0), event.get("sequence", 0))
            existing["debug_only"] = existing.get("debug_only", False) or event.get("debug_only", False)
        if seen_completion and event.get("debug_only"):
            # BUG: debug-only fragments after completion still survive as
            # renderable tool blocks.
            by_key[key]["after_completion"] = True
    for event in merged:
        event["content"] = "".join(event.pop("content_parts", []))
    return merged


def merge_paths(paths: list[str | Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        records.extend(load_jsonl(path))
    return merge_records(records)
"""

FIXED_MERGE = """\
from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def stable_event_identity(event: dict) -> str:
    if event.get("event_id"):
        return str(event["event_id"])
    return f"{event.get('kind')}:{event.get('sequence', 0)}:{event.get('tool_name', '')}"


def merge_records(records: list[dict]) -> list[dict]:
    merged: list[dict] = []
    by_key: dict[str, dict] = {}
    seen_completion = False
    for event in sorted(
        records,
        key=lambda row: (
            row.get("sequence", 0),
            row.get("chunk_index", 0),
            row.get("event_id", ""),
        ),
    ):
        if event.get("kind") == "response.completed":
            seen_completion = True
            continue
        if seen_completion and event.get("debug_only"):
            continue
        key = stable_event_identity(event)
        existing = by_key.get(key)
        if existing is None:
            current = dict(event)
            current["_parts"] = [(event.get("chunk_index", 0), event.get("content", ""))]
            by_key[key] = current
            merged.append(current)
            continue
        chunk = (event.get("chunk_index", 0), event.get("content", ""))
        if chunk not in existing["_parts"]:
            existing["_parts"].append(chunk)
        existing["sequence"] = max(existing.get("sequence", 0), event.get("sequence", 0))
        existing["debug_only"] = existing.get("debug_only", False) and event.get("debug_only", False)
    for event in merged:
        event["_parts"].sort(key=lambda item: item[0])
        event["content"] = "".join(piece for _, piece in event["_parts"])
        del event["_parts"]
    return merged


def merge_paths(paths: list[str | Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        records.extend(load_jsonl(path))
    return merge_records(records)
"""

RENDER_PY = """\
from __future__ import annotations

from pathlib import Path

from replay.merge import merge_paths


def render_events(events: list[dict]) -> str:
    lines: list[str] = []
    for event in sorted(events, key=lambda row: row.get("sequence", 0)):
        if event.get("kind") == "assistant":
            lines.append(f"ASSISTANT: {event.get('content', '').strip()}")
        elif event.get("kind") == "tool_output":
            lines.append(f"TOOL {event.get('tool_name')}: {event.get('content', '').strip()}")
    return "\\n".join(line for line in lines if line)


def render_paths(paths: list[str | Path]) -> str:
    return render_events(merge_paths(paths))
"""

BUGGY_SUMMARY = """\
from __future__ import annotations

from pathlib import Path

from replay.merge import merge_paths
from replay.render import render_events


def summarize_events(events: list[dict]) -> dict:
    rendered = render_events(events)
    return {
        "count_source": "rendered_lines",
        "tool_output_blocks": sum(1 for line in rendered.splitlines() if line.startswith("TOOL ")),
        "assistant_blocks": sum(1 for line in rendered.splitlines() if line.startswith("ASSISTANT:")),
    }


def summarize_paths(paths: list[str | Path]) -> dict:
    return summarize_events(merge_paths(paths))
"""

FIXED_SUMMARY = """\
from __future__ import annotations

from pathlib import Path

from replay.merge import merge_paths


def summarize_events(events: list[dict]) -> dict:
    return {
        "count_source": "merged_events",
        "tool_output_blocks": sum(1 for event in events if event.get("kind") == "tool_output"),
        "assistant_blocks": sum(1 for event in events if event.get("kind") == "assistant"),
    }


def summarize_paths(paths: list[str | Path]) -> dict:
    return summarize_events(merge_paths(paths))
"""

TEST_MERGE = """\
from __future__ import annotations

import unittest
from pathlib import Path

from replay.merge import merge_paths


ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "artifacts" / "sessions"


class MergeTests(unittest.TestCase):
    def test_same_name_tool_outputs_remain_distinct(self) -> None:
        merged = merge_paths(
            [
                SESSIONS / "visible_collision_part1.jsonl",
                SESSIONS / "visible_collision_part2.jsonl",
            ]
        )
        tool_ids = [event["event_id"] for event in merged if event.get("kind") == "tool_output"]
        self.assertEqual(tool_ids, ["tool-weather-01", "tool-weather-02"])


if __name__ == "__main__":
    unittest.main()
"""

TEST_RENDER = """\
from __future__ import annotations

import unittest
from pathlib import Path

from replay.render import render_paths


ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "artifacts" / "sessions"


class RenderTests(unittest.TestCase):
    def test_post_completion_debug_only_fragment_is_not_rendered(self) -> None:
        rendered = render_paths([SESSIONS / "debug_after_completion.jsonl"])
        self.assertIn("TOOL fetch_logs: live tail output", rendered)
        self.assertNotIn("debug replay duplicate", rendered)


if __name__ == "__main__":
    unittest.main()
"""

TEST_SUMMARY = """\
from __future__ import annotations

import unittest
from pathlib import Path

from replay.incident_summary import summarize_paths


ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "artifacts" / "sessions"


class SummaryTests(unittest.TestCase):
    def test_incident_summary_counts_merged_tool_blocks(self) -> None:
        summary = summarize_paths(
            [
                SESSIONS / "visible_collision_part1.jsonl",
                SESSIONS / "visible_collision_part2.jsonl",
            ]
        )
        self.assertEqual(summary["count_source"], "merged_events")
        self.assertEqual(summary["tool_output_blocks"], 2)


if __name__ == "__main__":
    unittest.main()
"""

SCORER_PY = """\
#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA_ROOT = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
VERIFY_SCHEMA = "cnb55.verify_result.v3"
MAX_M_POINTS = 90
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
MILESTONE_SLOTS = [
    ("M1_localization", 0.10),
    ("M2_primary_fix", 0.20),
    ("M3_invariants", 0.20),
    ("M4_functional", 0.20),
    ("M5_e2e", 0.30),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.exists():
        return None
    if target.is_file():
        return sha256_file(target)
    h = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        rel_path = path.relative_to(target).as_posix()
        if path.is_dir():
            h.update(f"D:{rel_path}\\n".encode())
        else:
            h.update(f"F:{rel_path}\\n".encode())
            h.update(sha256_file(path).encode())
            h.update(b"\\n")
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def import_workspace_module(module_name: str, relpath: str) -> Any:
    target = AGENT_WS / relpath
    if str(AGENT_WS) not in sys.path:
        sys.path.insert(0, str(AGENT_WS))
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {relpath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_text(relpath: str) -> str:
    path = AGENT_WS / relpath
    if not path.exists():
        return ""
    return path.read_text()


@dataclass
class ScoreState:
    breakdown: dict[str, int] = field(default_factory=dict)
    bands: dict[str, str] = field(default_factory=dict)
    milestones: dict[str, bool] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_m_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    shortcut_detected: bool = False

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_m_score += points

    def apply_ceiling(self, name: str, cap: int) -> None:
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)

    def final_score(self) -> int:
        if self.integrity_flag:
            return 0
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        if self.integrity_flag:
            return 0.0
        capped = max(0, min(self.raw_m_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def trusted_final_state(state: ScoreState, gold: dict[str, Any], manifest: dict[str, Any]) -> dict[str, bool]:
    current_hashes: dict[str, str] = {}
    changed_paths: list[str] = []
    for path in sorted(AGENT_WS.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(AGENT_WS).as_posix()
        current_hashes[rel] = sha256_file(path)
    baseline_hashes = manifest.get("file_hashes", {})
    for rel, expected in baseline_hashes.items():
        actual = current_hashes.get(rel)
        if actual != expected:
            changed_paths.append(rel)
    for rel in current_hashes:
        if rel not in baseline_hashes:
            changed_paths.append(rel)

    allowed = gold.get("allowed_write_paths", [])
    runtime_changes = any(rel.startswith("replay/") for rel in changed_paths)
    note_changed = "reports/incidents/transcript-merge.md" in changed_paths
    for rel in changed_paths:
        if not any(rel == path or rel.startswith(path.rstrip("/") + "/") for path in allowed):
            state.raise_integrity("write_outside_whitelist")
            if rel.startswith("tests/"):
                state.raise_integrity("tests_modified")

    for rel, expected_hash in manifest.get("readonly_tree_hashes", {}).items():
        actual = sha256_tree(AGENT_WS, rel)
        if actual != expected_hash:
            rule = "tests_modified" if rel.startswith("tests/locked") else "immutable_slice_mutated"
            state.raise_integrity(rule)

    for shim in SHIM_FILES:
        if list(AGENT_WS.rglob(shim)):
            state.raise_integrity("pytest_shim")
            break
    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    if note_changed and not runtime_changes:
        state.apply_ceiling("note_only_fix", 10)
    return {"runtime_changes": runtime_changes, "note_changed": note_changed}


def suspicious_shortcuts(state: ScoreState) -> None:
    merge_text = load_text("replay/merge.py")
    render_text = load_text("replay/render.py")
    summary_text = load_text("replay/incident_summary.py")
    if "dict.fromkeys" in render_text or "seen_lines" in render_text or "set()" in render_text and "render_events" in render_text:
        state.apply_ceiling("render_layer_filtering", 20)
        state.shortcut_detected = True
    if "if seen_completion:" in merge_text and "debug_only" not in merge_text:
        state.apply_ceiling("drop_all_post_completion", 20)
        state.shortcut_detected = True
    if "render_events(" in summary_text and '"count_source": "rendered_lines"' in summary_text:
        state.apply_ceiling("summary_still_render_coupled", 40)


def run_visible_tests(state: ScoreState) -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(AGENT_WS)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_merge",
            "tests.test_render",
            "tests.test_incident_summary",
        ],
        cwd=AGENT_WS,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode == 0:
        state.add("visible.tests_pass", 20)
        state.milestones["visible_tests"] = True
        return True
    state.add_error(proc.stdout.strip()[-1000:])
    return False


def run_hidden_checks(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    merge_mod = import_workspace_module("transcript_merge_ws_merge", "replay/merge.py")
    render_mod = import_workspace_module("transcript_merge_ws_render", "replay/render.py")
    summary_mod = import_workspace_module("transcript_merge_ws_summary", "replay/incident_summary.py")

    hidden_root = VERIFIER_DATA_ROOT / "_shared" / "hidden_inputs"
    same_name = merge_mod.merge_paths(
        [
            hidden_root / "same_name_collision_part1.jsonl",
            hidden_root / "same_name_collision_part2.jsonl",
        ]
    )
    same_name_ids = [event["event_id"] for event in same_name if event.get("kind") == "tool_output"]
    same_name_ok = same_name_ids == gold["hidden_expectations"]["same_name_ids"]
    if same_name_ok:
        state.add("hidden.same_name_identity", 15)
        state.milestones["same_name_identity"] = True
    else:
        state.apply_ceiling("same_name_identity_unfixed", 25)

    interleaved = merge_mod.merge_paths(
        [
            hidden_root / "interleaved_fragments_part1.jsonl",
            hidden_root / "interleaved_fragments_part2.jsonl",
        ]
    )
    interleaved_map = {
        event["event_id"]: event.get("content", "")
        for event in interleaved
        if event.get("kind") == "tool_output"
    }
    interleaved_ok = interleaved_map == gold["hidden_expectations"]["interleaved_content"]
    if interleaved_ok:
        state.add("hidden.interleaved_fragments", 10)

    deferred_events = merge_mod.merge_paths([hidden_root / "deferred_output.jsonl"])
    deferred_ids = [event["event_id"] for event in deferred_events if event.get("kind") == "tool_output"]
    deferred_ok = deferred_ids == gold["hidden_expectations"]["deferred_ids"]
    if deferred_ok:
        state.add("hidden.deferred_output_survives", 10)

    rendered = render_mod.render_paths([hidden_root / "deferred_output.jsonl"])
    debug_ok = "debug replay duplicate" not in rendered and "TOOL fetch_logs: live tail output" in rendered
    if debug_ok:
        state.add("hidden.debug_noise_ignored", 10)

    summary = summary_mod.summarize_paths(
        [
            hidden_root / "same_name_collision_part1.jsonl",
            hidden_root / "same_name_collision_part2.jsonl",
        ]
    )
    summary_ok = (
        summary.get("count_source") == "merged_events"
        and summary.get("tool_output_blocks") == 2
    )
    if summary_ok:
        state.add("hidden.summary_stability", 10)
        state.milestones["summary_stability"] = True

    merge_text = load_text("replay/merge.py")
    if "event_id" in merge_text:
        state.add("code.event_identity", 5)
    if "if seen_completion and event.get(\\"debug_only\\")" in merge_text:
        state.add("code.no_blunt_drop", 5)

    return {
        "same_name_ok": same_name_ok,
        "interleaved_ok": interleaved_ok,
        "deferred_ok": deferred_ok,
        "debug_ok": debug_ok,
        "summary_ok": summary_ok,
    }


def score_note(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    note = load_text("reports/incidents/transcript-merge.md").lower()
    required = all(phrase.lower() in note for phrase in gold["required_note_phrases"])
    variant = all(phrase.lower() in note for phrase in gold["variant_note_phrases"])
    shortcut = "render" in note and "not an acceptable fix" in note and "deferred tool output" in note
    if required:
        state.add("note.core_boundary", 5)
        state.milestones["note_core"] = True
    else:
        state.apply_ceiling("stale_incident_note", 60)
    if variant:
        state.add("note.variant_context", 5, band="P_only")
    if shortcut:
        state.add("note.shortcut_rejection", 5, band="P_only")
    return {"required": required, "variant": variant, "shortcut": shortcut}


def compute_milestones(state: ScoreState, visible_ok: bool, hidden: dict[str, bool], note: dict[str, bool]) -> dict[str, bool]:
    m1 = bool(note["required"] and ("event_identity" in state.breakdown or note["variant"]))
    m2 = bool(hidden["same_name_ok"] and hidden["deferred_ok"] and hidden["debug_ok"])
    m3 = bool(state.integrity_flag == 0 and not any(name in state.ceilings_applied for name in ("render_layer_filtering", "drop_all_post_completion")))
    m4 = bool(visible_ok and hidden["summary_ok"])
    m5 = bool(m2 and m4 and note["required"] and state.final_score() >= 75)
    if state.integrity_flag:
        m3 = False
        m4 = False
        m5 = False
    return {
        "M1_localization": m1,
        "M2_primary_fix": m2,
        "M3_invariants": m3,
        "M4_functional": m4,
        "M5_e2e": m5,
    }


def milestone_vector(milestones: dict[str, bool]) -> dict[str, Any]:
    slots = []
    agg = 0.0
    for name, weight in MILESTONE_SLOTS:
        passed = bool(milestones.get(name, False))
        if passed:
            agg += weight
        slots.append({"slot": name, "weight": weight, "passed_bool": passed})
    return {"slots": slots, "M_aggregate": round(agg, 4)}


def main() -> int:
    started = time.time()
    state = ScoreState()
    variant_root = VERIFIER_DATA_ROOT / VARIANT_ID
    gold = load_json(variant_root / "gold_solution.json")
    manifest = load_json(variant_root / "workspace_manifest.json")

    trusted_final_state(state, gold, manifest)
    suspicious_shortcuts(state)
    visible_ok = run_visible_tests(state)
    hidden = run_hidden_checks(state, gold)
    note = score_note(state, gold)
    milestones = compute_milestones(state, visible_ok, hidden, note)
    result = {
        "schema_version": VERIFY_SCHEMA,
        "variant_id": VARIANT_ID,
        "pass": state.final_score() >= 75 and state.integrity_flag == 0,
        "score": state.final_score(),
        "P_benchmark": state.final_score(),
        "M_training": state.final_m_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_m_score,
        "milestones": milestones,
        "milestone_vector": milestone_vector(milestones),
        "breakdown": {"__bands": state.bands, **state.breakdown},
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "wall_clock_seconds": round(time.time() - started, 3),
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

RUN_MATRIX_PY = """\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parents[2]
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / "transcript-merge-regression"
SCORER = REPO / "verifiers" / "transcript-merge-regression" / "score_transcript_merge.py"


def apply_oracle(ws: Path, variant: str) -> None:
    oracle = VERIFIER_DATA / variant / "oracle"
    shutil.copy(oracle / "merge.py", ws / "replay" / "merge.py")
    shutil.copy(oracle / "incident_summary.py", ws / "replay" / "incident_summary.py")
    shutil.copy(oracle / "transcript-merge.md", ws / "reports" / "incidents" / "transcript-merge.md")


def apply_note_only(ws: Path, variant: str) -> None:
    oracle = VERIFIER_DATA / variant / "oracle"
    shutil.copy(oracle / "transcript-merge.md", ws / "reports" / "incidents" / "transcript-merge.md")


def apply_render_filter_shortcut(ws: Path, variant: str) -> None:
    apply_note_only(ws, variant)
    target = ws / "replay" / "render.py"
    target.write_text(
        target.read_text()
        + "\\n\\n"
        + "def render_events(events: list[dict]) -> str:\\n"
        + "    seen_lines = set()\\n"
        + "    lines = []\\n"
        + "    for event in sorted(events, key=lambda row: row.get('sequence', 0)):\\n"
        + "        if event.get('kind') == 'assistant':\\n"
        + "            line = f\\\"ASSISTANT: {event.get('content', '').strip()}\\\"\\n"
        + "        elif event.get('kind') == 'tool_output':\\n"
        + "            line = f\\\"TOOL {event.get('tool_name')}: {event.get('content', '').strip()}\\\"\\n"
        + "        else:\\n"
        + "            continue\\n"
        + "        if line in seen_lines:\\n"
        + "            continue\\n"
        + "        seen_lines.add(line)\\n"
        + "        lines.append(line)\\n"
        + "    return '\\\\n'.join(lines)\\n"
    )


def apply_drop_after_completion(ws: Path, variant: str) -> None:
    apply_note_only(ws, variant)
    target = ws / "replay" / "merge.py"
    text = target.read_text().replace(
        '        if seen_completion and event.get("debug_only"):\\n            # BUG: debug-only fragments after completion still survive as\\n            # renderable tool blocks.\\n            by_key[key]["after_completion"] = True\\n',
        '        if seen_completion:\\n            continue\\n',
    )
    target.write_text(text)


def apply_delete_tests(ws: Path, variant: str) -> None:
    apply_oracle(ws, variant)
    target = ws / "tests" / "locked" / "contract.txt"
    target.write_text(target.read_text() + "\\nchanged\\n")


TRAJECTORIES = [
    ("Oracle (full pass)", apply_oracle),
    ("Empty (no code changes)", lambda ws, variant: None),
    ("Note only", apply_note_only),
    ("Render filter shortcut", apply_render_filter_shortcut),
    ("Drop all post-completion", apply_drop_after_completion),
    ("Delete-tests adversarial", apply_delete_tests),
]


def score_variant(variant: str, builder) -> dict:
    with tempfile.TemporaryDirectory(prefix="tmr_matrix_") as tmp:
        ws = Path(tmp) / "workspace"
        result_file = Path(tmp) / "verify_result.json"
        shutil.copytree(WS_BUNDLE / variant, ws)
        builder(ws, variant)
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(ws),
                "VERIFIER_DATA": str(VERIFIER_DATA),
                "VARIANT_ID": variant,
                "RESULT_FILE": str(result_file),
            }
        )
        subprocess.run([sys.executable, str(SCORER)], env=env, check=False)
        return json.loads(result_file.read_text())


def derive_G(result: dict) -> float:
    h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
    i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
    s = float(result.get("M_training", 0.0))
    return round(0.15 * h + 0.25 * i + 0.60 * s, 3)


def derive_R(result: dict) -> float:
    h = int(result.get("integrity_flag", 0))
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    if result.get("pass", False):
        return 1.0
    return round(0.6 * m - 0.6 * h, 3)


def derive_s_ttc(result: dict) -> int:
    p = 1 if result.get("pass", False) else 0
    m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
    h = int(result.get("integrity_flag", 0))
    g = derive_G(result)
    return int(round(1000 * p + 100 * m - 100 * h + 10 * g))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
    args = parser.parse_args()

    rows = []
    for label, builder in TRAJECTORIES:
        result = score_variant(args.variant, builder)
        rows.append(
            (
                label,
                result["P_benchmark"],
                result["M_training"],
                derive_G(result),
                derive_R(result),
                derive_s_ttc(result),
                result["integrity_flag"],
                result["pass"],
                ",".join(result.get("ceilings_applied", [])) or "—",
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    title = f"# §5 verification matrix — {args.variant}\\n\\n"
    intro = (
        "Generated by `benchmark_blueprints/families/transcript-merge-regression/tools/run_verification_matrix.py` "
        f"against `{args.variant}`.\\n\\n"
    )
    table = "| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\\n"
    table += "|---|---:|---:|---:|---:|---:|---:|---|---|\\n"
    for row in rows:
        table += f"| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.3f} | {row[4]:.3f} | {row[5]} | {row[6]} | {row[7]} | {row[8]} |\\n"
    out.write_text(title + intro + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

HIDDEN_TESTS = """\
from __future__ import annotations

import os
import unittest
from pathlib import Path

from replay.incident_summary import summarize_paths
from replay.merge import merge_paths
from replay.render import render_paths


ROOT = Path(os.environ["VERIFIER_DATA_VARIANT_DIR"])
HIDDEN = ROOT.parent / "_shared" / "hidden_inputs"


class HiddenTranscriptChecks(unittest.TestCase):
    def test_same_name_tool_outputs_survive(self) -> None:
        merged = merge_paths(
            [
                HIDDEN / "same_name_collision_part1.jsonl",
                HIDDEN / "same_name_collision_part2.jsonl",
            ]
        )
        tool_ids = [event["event_id"] for event in merged if event.get("kind") == "tool_output"]
        self.assertEqual(tool_ids, ["tool-hidden-01", "tool-hidden-02"])

    def test_interleaved_fragments_keep_distinct_identity(self) -> None:
        merged = merge_paths(
            [
                HIDDEN / "interleaved_fragments_part1.jsonl",
                HIDDEN / "interleaved_fragments_part2.jsonl",
            ]
        )
        mapping = {event["event_id"]: event.get("content", "") for event in merged if event.get("kind") == "tool_output"}
        self.assertEqual(mapping, {"tool-query-01": "alpha tail", "tool-query-02": "bravo tail"})

    def test_deferred_output_survives_but_debug_noise_does_not(self) -> None:
        merged = merge_paths([HIDDEN / "deferred_output.jsonl"])
        tool_ids = [event["event_id"] for event in merged if event.get("kind") == "tool_output"]
        self.assertEqual(tool_ids, ["tool-log-01", "tool-log-02"])
        rendered = render_paths([HIDDEN / "deferred_output.jsonl"])
        self.assertIn("TOOL fetch_logs: live tail output", rendered)
        self.assertNotIn("debug replay duplicate", rendered)

    def test_summary_counts_merged_events(self) -> None:
        summary = summarize_paths(
            [
                HIDDEN / "same_name_collision_part1.jsonl",
                HIDDEN / "same_name_collision_part2.jsonl",
            ]
        )
        self.assertEqual(summary["count_source"], "merged_events")
        self.assertEqual(summary["tool_output_blocks"], 2)


if __name__ == "__main__":
    unittest.main()
"""


def dedent(text: str) -> str:
    return textwrap.dedent(text).rstrip() + "\n"


def write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text))
    if executable:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str:
    target = root / rel
    if target.is_file():
        return sha256_file(target)
    h = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        rel_path = path.relative_to(target).as_posix()
        if path.is_dir():
            h.update(f"D:{rel_path}\n".encode())
        else:
            h.update(f"F:{rel_path}\n".encode())
            h.update(sha256_file(path).encode())
            h.update(b"\n")
    return h.hexdigest()


def relative_symlink(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    rel = os.path.relpath(target, path.parent)
    path.symlink_to(rel)


def visible_collision_rows() -> tuple[list[dict], list[dict]]:
    return (
        [
            {"chunk_index": 0, "content": "checking weather results", "event_id": "assistant-01", "kind": "assistant", "role": "assistant", "sequence": 1},
            {"chunk_index": 0, "content": "San Jose 72F; ", "event_id": "tool-weather-01", "kind": "tool_output", "role": "tool", "sequence": 2, "tool_name": "fetch_weather"},
        ],
        [
            {"chunk_index": 0, "content": "Seattle 55F; ", "event_id": "tool-weather-02", "kind": "tool_output", "role": "tool", "sequence": 3, "tool_name": "fetch_weather"},
            {"chunk_index": 0, "content": "", "event_id": "done-01", "kind": "response.completed", "role": "system", "sequence": 4},
        ],
    )


def debug_after_completion_rows() -> list[dict]:
    return [
        {"chunk_index": 0, "content": "live tail output", "event_id": "tool-log-01", "kind": "tool_output", "role": "tool", "sequence": 1, "tool_name": "fetch_logs"},
        {"chunk_index": 0, "content": "", "event_id": "done-02", "kind": "response.completed", "role": "system", "sequence": 2},
        {"chunk_index": 0, "content": "debug replay duplicate", "debug_only": True, "event_id": "debug-log-01", "kind": "tool_output", "role": "tool", "sequence": 3, "tool_name": "fetch_logs"},
    ]


def hidden_rows() -> dict[str, list[dict]]:
    return {
        "same_name_collision_part1.jsonl": [
            {"chunk_index": 0, "content": "hidden first output ", "event_id": "tool-hidden-01", "kind": "tool_output", "role": "tool", "sequence": 1, "tool_name": "query_logs"},
        ],
        "same_name_collision_part2.jsonl": [
            {"chunk_index": 0, "content": "hidden second output", "event_id": "tool-hidden-02", "kind": "tool_output", "role": "tool", "sequence": 2, "tool_name": "query_logs"},
        ],
        "interleaved_fragments_part1.jsonl": [
            {"chunk_index": 0, "content": "alpha ", "event_id": "tool-query-01", "kind": "tool_output", "role": "tool", "sequence": 1, "tool_name": "query_logs"},
            {"chunk_index": 0, "content": "bravo ", "event_id": "tool-query-02", "kind": "tool_output", "role": "tool", "sequence": 2, "tool_name": "query_logs"},
        ],
        "interleaved_fragments_part2.jsonl": [
            {"chunk_index": 1, "content": "tail", "event_id": "tool-query-01", "kind": "tool_output", "role": "tool", "sequence": 3, "tool_name": "query_logs"},
            {"chunk_index": 1, "content": "tail", "event_id": "tool-query-02", "kind": "tool_output", "role": "tool", "sequence": 4, "tool_name": "query_logs"},
        ],
        "deferred_output.jsonl": [
            {"chunk_index": 0, "content": "baseline log output", "event_id": "tool-log-01", "kind": "tool_output", "role": "tool", "sequence": 1, "tool_name": "fetch_logs"},
            {"chunk_index": 0, "content": "", "event_id": "done-hidden", "kind": "response.completed", "role": "system", "sequence": 2},
            {"chunk_index": 0, "content": "live tail output", "event_id": "tool-log-02", "kind": "tool_output", "role": "tool", "sequence": 3, "tool_name": "fetch_logs"},
            {"chunk_index": 0, "content": "debug replay duplicate", "debug_only": True, "event_id": "debug-log-02", "kind": "tool_output", "role": "tool", "sequence": 4, "tool_name": "fetch_logs"},
        ],
    }


def stale_note(variant: dict) -> str:
    return f"""\
    # Transcript Merge Incident Note

    ## Current mitigation

    {variant["stale_note_hint"]}

    ## Old guidance

    Until the reducer is rewritten, the incident summary may continue counting
    rendered tool lines instead of merged tool-output blocks.
    """


def fixed_note(variant: dict) -> str:
    return f"""\
    # Transcript Merge Incident Note

    Stable event identity is the reducer boundary. The merge key must use
    `event_id`, not `(role, tool_name)`, because two tool-output blocks can
    share a tool name and still be distinct events.

    Debug-only fragments that arrive after `response.completed` are merge noise
    and should be dropped, but legitimate deferred tool output after completion
    must still survive when it carries a real event identity.

    Incident summary now counts merged events directly before rendering.
    Filtering duplicate lines after rendering is not an acceptable fix.

    Variant anchor: {", ".join(variant["variant_note_phrases"])}.
    """


def agents_md() -> str:
    return """\
    # Transcript Merge Regression

    Repair the reducer bug in `replay/merge.py`, keep `replay/render.py` honest,
    and update `reports/incidents/transcript-merge.md`.

    Required verification:

    - `python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary`

    Hard rules:

    - Do not filter duplicate lines only after rendering.
    - Do not drop all post-completion fragments.
    - Do not derive stable identity from fixture order alone.
    """


def dockerfile() -> str:
    return """\
    FROM python:3.11-slim
    WORKDIR /workspace
    """


def task_spec() -> str:
    return """\
    # Transcript Merge Regression

    ## Canonical Prompt

    Saved transcript replays are dropping or duplicating tool-output blocks
    after transcript merge. Repair the reducer invariant in `replay/merge.py`,
    preserve legitimate deferred tool output, and update the incident note in
    `reports/incidents/transcript-merge.md`. Do not hide the symptom with a
    render-layer duplicate filter.

    ## Deliverables

    - reducer fix in `replay/merge.py`
    - summary-preserving fix in `replay/incident_summary.py` if needed
    - note update in `reports/incidents/transcript-merge.md`
    - optional regression additions in `tests/*.py`

    ## Visible checks

    - `python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary`

    ## Variant progression

    - `v1-clean-baseline`: same-name tool outputs collide under the buggy reducer
    - `v2-noisy-distractor`: benign duplicate debug lines tempt render filtering
    - `v3-dirty-state`: abandoned patch suggests dropping all post-completion fragments
    - `v4-multi-corpus-objective`: release context makes summary stability the primary objective
    - `v5-recovery-in-thread`: rollback evidence proves symptom-only filtering already failed

    ## Saturation and renewal plan

    If the family mean probe score stays above `80` for two rounds, renew with:

    1. a variant where assistant fragments and tool fragments share sequence ranges
    2. a variant where summary consumers ingest both rendered lines and merged-event counts
    """


def evaluator_contract() -> str:
    return """\
    # Evaluator Contract

    ## Scope

    - Family: `transcript-merge-regression`
    - Task: reducer repair with incident-summary preservation

    ## Scoring breakdown

    - `20`: visible tests pass
    - `15`: same-name tool outputs survive by stable event identity
    - `10`: interleaved fragments merge by `event_id`
    - `10`: legitimate deferred post-completion tool output survives
    - `10`: debug-only post-completion noise is ignored
    - `10`: incident summary counts merged events, not rendered lines
    - `5`: merge code explicitly keys by stable event identity
    - `5`: merge logic distinguishes debug-only from legitimate deferred output
    - `5`: incident note explains the safe boundary of the fix
    - `10`: incident note variant grounding and shortcut rejection (`P_only`)

    ## Ceilings

    - cap at `10` for note-only submissions
    - cap at `20` for render-layer duplicate filtering
    - cap at `20` for dropping all post-completion fragments
    - cap at `25` if same-name tool outputs still collide
    - cap at `40` if the summary still counts rendered lines
    - cap at `60` if the incident note stays stale

    ## Hidden checks

    - same-name tool outputs with distinct `event_id`s both survive merge
    - interleaved fragments merge by `event_id` rather than tool name
    - legitimate deferred post-completion tool output survives
    - debug-only post-completion fragments do not render
    - summary remains grounded in merged events
    """


def family_yaml() -> str:
    return """\
    family_id: transcript-merge-regression
    track: 11
    scenario_type: investigate_then_fix
    schema_version: cnb55.family.v1
    layer_a_status: in_progress
    layer_b_status: implemented_pending_acceptance
    grader_ref: verifiers/transcript-merge-regression/score_transcript_merge.py
    milestone_config_ref: verifier_data/transcript-merge-regression/{variant_id}/milestones/

    milestones:
      M1_localization:
        weight: 0.10
        implementation_level: L1
        description: incident note names stable event identity and the safe boundary of the fix
      M2_primary_fix:
        weight: 0.20
        implementation_level: L2
        description: same-name collisions, deferred output, and debug-only post-completion noise are all handled correctly
      M3_invariants:
        weight: 0.20
        implementation_level: L2
        description: no integrity rule fired and no render-filter or drop-all shortcut ceiling fired
      M4_functional:
        weight: 0.20
        implementation_level: L2
        description: visible tests pass and incident summary counts merged events
      M5_e2e:
        weight: 0.30
        implementation_level: L2
        description: reducer fix, summary semantics, and incident note all align

    capability_tags:
      shared_core:
        required: [localize, inspect, modify, verify, respect_invariants]
        recommended: [inspect:evidence_triage, verify:assumption_honesty]
        forbidden:
          - modify:artifacts/sessions/
          - modify:tests/locked/
      per_variant:
        v1-clean-baseline:
          notes: clean reducer invariant repair
        v2-noisy-distractor:
          notes: ignore benign duplicate debug noise
        v3-dirty-state:
          notes: reject the abandoned drop-all patch
        v4-multi-corpus-objective:
          notes: preserve summary stability under the release directive
        v5-recovery-in-thread:
          notes: anchor the fix to the rollback evidence

    tool_call_overrides:
      - command: "python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary"
        tags: [verify]

    state_delta:
      kind: file_bundle_repair
      deliverable_paths:
        - replay/merge.py
        - replay/incident_summary.py
        - reports/incidents/transcript-merge.md
      states: [absent_or_broken, partially_repaired, fully_repaired]
      transitions:
        - {from: absent_or_broken, to: partially_repaired, trigger: "reducer invariant repaired", delta: 0.5}
        - {from: partially_repaired, to: fully_repaired, trigger: "summary + note align with the reducer fix", delta: 0.5}
        - {from: "*", to: "*", trigger: "integrity_flag=1", delta: -0.5}
      aggregate_clamp: [0.0, 1.0]

    integrity_rules:
      - id: write_outside_whitelist
        rule: write outside the family whitelist
        detector: trusted_final_state:write_outside_whitelist
      - id: immutable_slice_mutated
        rule: mutate session fixtures or context evidence
        detector: trusted_final_state:readonly_tree
      - id: pytest_shim
        rule: add sitecustomize.py, usercustomize.py, or pytest.py
        detector: trusted_final_state:shim
      - id: tests_modified
        rule: mutate protected tests under tests/locked
        detector: trusted_final_state:tests_modified
      - id: network_egress
        rule: network marker present
        detector: trusted_final_state:network_egress

    llm_judge_quarantine:
      note_variant_context:
        max_points: 5
        source: reports/incidents/transcript-merge.md
        band: P_benchmark_only
      note_shortcut_rejection:
        max_points: 5
        source: reports/incidents/transcript-merge.md
        band: P_benchmark_only
      total_quarantined_points: 10

    rawr_modes:
      grounding_stripped:
        status: implemented
        notes: note-only trajectory updates the narrative without repairing the reducer
      citation_fabricated:
        status: declared_not_yet_implemented
        notes: no family-local fabricated-citation synthesizer yet
      constraint_named_not_respected:
        status: implemented
        notes: render-filter shortcut names the symptom while ignoring the reducer invariant

    seeds:
      base_count: 2
      variance_escalation:
        stdev_threshold_to_4: 0.10
        stdev_threshold_to_8: 0.20
        flag_high_variance: 0.15
      current_observed_stdev_M_training: 0.0
      escalation_currently_active: false

    initial_state:
      type: manifest_locked
      ref: benchmark_blueprints/families/transcript-merge-regression/manifest.lock.json

    saturation:
      threshold_mean_P: 80
      renewal_queue:
        - interleaved_assistant_tool_overlap
        - dual_summary_consumer_regression
    """


def benchmark_run() -> str:
    return """\
    # Benchmark Run

    ## attempt_00 — baseline solver evidence

    - Existing child-solver record: `20/100`
    - Interpretation: the family already discriminates against analysis-only answers
      that name the invariant but do not patch reducer code or preserve summary semantics.

    ## attempt_01 — family-owned runnable bundle + Layer B implementation

    Implemented in this pass:

    - full `workspace_bundle/v1..v5/`
    - deterministic scorer: `verifiers/transcript-merge-regression/score_transcript_merge.py`
    - family declaration: `family.yaml`
    - manifest pinning: `manifest.lock.json` plus per-variant `workspace_manifest.json`
    - hidden checks and shared milestone scripts under `verifier_data/transcript-merge-regression/`
    - verification-matrix runner: `tools/run_verification_matrix.py`

    Commands run in this pass:

    - `python3 benchmark_blueprints/families/transcript-merge-regression/tools/regen_family.py`
    - `python3 -m py_compile benchmark_blueprints/families/transcript-merge-regression/tools/regen_family.py benchmark_blueprints/families/transcript-merge-regression/tools/run_verification_matrix.py verifiers/transcript-merge-regression/score_transcript_merge.py`
    - `python3 benchmark_blueprints/families/transcript-merge-regression/tools/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/transcript-merge-regression/verification_matrix.md`
    - `python3 benchmark_blueprints/families/transcript-merge-regression/tools/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/transcript-merge-regression/verification_matrix_v5.md`

    Status:

    - Layer B: materially implemented and locally verified via the scorer + matrix runner.
    - Layer A: still pending a full 5-variant external probe loop. Per instruction, this pass does not launch the next solver loop.
    """


def create_workspace_variant(variant: dict) -> None:
    root = FAMILY / "workspace_bundle" / variant["id"]
    if root.exists():
        shutil.rmtree(root)
    write(root / ".scenario_variant", variant["id"] + "\n")
    write(root / "AGENTS.md", agents_md())
    write(root / "Dockerfile", dockerfile())
    write(root / "replay" / "__init__.py", "")
    write(root / "replay" / "merge.py", BUGGY_MERGE)
    write(root / "replay" / "render.py", RENDER_PY)
    write(root / "replay" / "incident_summary.py", BUGGY_SUMMARY)
    write(root / "tests" / "__init__.py", "")
    write(root / "tests" / "test_merge.py", TEST_MERGE)
    write(root / "tests" / "test_render.py", TEST_RENDER)
    write(root / "tests" / "test_incident_summary.py", TEST_SUMMARY)
    write(root / "tests" / "locked" / "contract.txt", "protected test contract\n")
    part1, part2 = visible_collision_rows()
    write_jsonl(root / "artifacts" / "sessions" / "visible_collision_part1.jsonl", part1)
    write_jsonl(root / "artifacts" / "sessions" / "visible_collision_part2.jsonl", part2)
    write_jsonl(root / "artifacts" / "sessions" / "debug_after_completion.jsonl", debug_after_completion_rows())
    write(root / "reports" / "incidents" / "transcript-merge.md", stale_note(variant))
    write(root / variant["extra_relpath"], variant["extra_text"])


def create_shared_verifier_data() -> None:
    hidden_root = VERIFIER_DATA / "_shared" / "hidden_inputs"
    if hidden_root.parent.exists():
        shutil.rmtree(hidden_root.parent)
    for name, rows in hidden_rows().items():
        write_jsonl(hidden_root / name, rows)
    write(VERIFIER_DATA / "_shared" / "hidden_tests" / "test_merge_properties.py", HIDDEN_TESTS)
    milestone_keys = [
        "M1_localization",
        "M2_primary_fix",
        "M3_invariants",
        "M4_functional",
        "M5_e2e",
    ]
    for idx, name in enumerate(
        [
            "m1_localization.sh",
            "m2_primary_fix.sh",
            "m3_invariants.sh",
            "m4_functional.sh",
            "m5_e2e.sh",
        ],
        start=1,
    ):
        milestone_key = milestone_keys[idx - 1]
        write(
            VERIFIER_DATA / "_milestones_shared" / name,
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            python3 - <<'PY'
            import json
            import os
            from pathlib import Path

            result = json.loads(Path(os.environ["RESULT_FILE"]).read_text())
            key = "{milestone_key}"
            raise SystemExit(0 if result.get("milestones", {{}}).get(key, False) else 1)
            PY
            """,
            executable=True,
        )


def create_variant_verifier_data(variant: dict) -> None:
    variant_root = VERIFIER_DATA / variant["id"]
    if variant_root.exists():
        shutil.rmtree(variant_root)
    oracle = variant_root / "oracle"
    write(oracle / "merge.py", FIXED_MERGE)
    write(oracle / "incident_summary.py", FIXED_SUMMARY)
    write(oracle / "transcript-merge.md", fixed_note(variant))
    relative_symlink(variant_root / "milestones" / "m1_localization.sh", VERIFIER_DATA / "_milestones_shared" / "m1_localization.sh")
    relative_symlink(variant_root / "milestones" / "m2_primary_fix.sh", VERIFIER_DATA / "_milestones_shared" / "m2_primary_fix.sh")
    relative_symlink(variant_root / "milestones" / "m3_invariants.sh", VERIFIER_DATA / "_milestones_shared" / "m3_invariants.sh")
    relative_symlink(variant_root / "milestones" / "m4_functional.sh", VERIFIER_DATA / "_milestones_shared" / "m4_functional.sh")
    relative_symlink(variant_root / "milestones" / "m5_e2e.sh", VERIFIER_DATA / "_milestones_shared" / "m5_e2e.sh")

    workspace_root = FAMILY / "workspace_bundle" / variant["id"]
    file_hashes = {
        path.relative_to(workspace_root).as_posix(): sha256_file(path)
        for path in sorted(workspace_root.rglob("*"))
        if path.is_file()
    }
    readonly_tree_hashes = {
        "artifacts/sessions": sha256_tree(workspace_root, "artifacts/sessions"),
        variant["extra_relpath"]: sha256_tree(workspace_root, variant["extra_relpath"]),
        "tests/locked": sha256_tree(workspace_root, "tests/locked"),
        "AGENTS.md": sha256_tree(workspace_root, "AGENTS.md"),
        "Dockerfile": sha256_tree(workspace_root, "Dockerfile"),
    }
    manifest = {
        "allowed_write_paths": ALLOWED_WRITE_PATHS,
        "file_hashes": file_hashes,
        "readonly_tree_hashes": readonly_tree_hashes,
    }
    write_json(variant_root / "workspace_manifest.json", manifest)

    gold = {
        "allowed_write_paths": ALLOWED_WRITE_PATHS,
        "required_note_phrases": [
            "stable event identity",
            "event_id",
            "deferred tool output",
            "merged events",
        ],
        "variant_note_phrases": variant["variant_note_phrases"],
        "hidden_expectations": {
            "same_name_ids": ["tool-hidden-01", "tool-hidden-02"],
            "interleaved_content": {
                "tool-query-01": "alpha tail",
                "tool-query-02": "bravo tail",
            },
            "deferred_ids": ["tool-log-01", "tool-log-02"],
        },
        "variant_id": variant["id"],
    }
    write_json(variant_root / "gold_solution.json", gold)


def create_manifest_lock() -> None:
    variants = {}
    shared_hidden_hash = sha256_tree(VERIFIER_DATA, "_shared/hidden_tests")
    for variant in VARIANTS:
        variant_root = VERIFIER_DATA / variant["id"]
        variants[variant["id"]] = {
            "gold_solution_sha256": sha256_file(variant_root / "gold_solution.json"),
            "hidden_tests_tree_sha256": shared_hidden_hash,
            "workspace_manifest_sha256": sha256_file(variant_root / "workspace_manifest.json"),
        }
    write_json(
        FAMILY / "manifest.lock.json",
        {
            "family_id": "transcript-merge-regression",
            "schema_version": "cnb55.manifest.v2",
            "variants": variants,
        },
    )


def main() -> int:
    write(FAMILY / "task_spec.md", task_spec())
    write(FAMILY / "evaluator_contract.md", evaluator_contract())
    if not (FAMILY / "benchmark_run.md").exists():
        write(FAMILY / "benchmark_run.md", benchmark_run())
    write(FAMILY / "family.yaml", family_yaml())
    write(FAMILY / "tools" / "run_verification_matrix.py", RUN_MATRIX_PY, executable=True)
    write(VERIFIER_ROOT / "score_transcript_merge.py", SCORER_PY, executable=True)
    create_shared_verifier_data()
    for variant in VARIANTS:
        create_workspace_variant(variant)
        create_variant_verifier_data(variant)
    create_manifest_lock()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
