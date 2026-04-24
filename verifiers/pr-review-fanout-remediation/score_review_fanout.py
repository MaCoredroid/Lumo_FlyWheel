#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

VERIFY_RESULT_SCHEMA = "cnb55.verify_result.v3"
MAX_M_POINTS = 95
PASS_BAR = 70
MILESTONE_SLOTS = [
    ("M1_localization", 0.10),
    ("M2_primary_fix", 0.20),
    ("M3_invariants", 0.20),
    ("M4_functional", 0.20),
    ("M5_e2e", 0.30),
]
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def tree_hash(root: Path, rel: str) -> str:
    target = root / rel
    if not target.exists():
        return "MISSING"
    if target.is_file():
        return sha256_file(target)
    h = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        relpath = path.relative_to(target).as_posix()
        if path.is_dir():
            h.update(f"D:{relpath}\n".encode())
        else:
            h.update(f"F:{relpath}\n".encode())
            h.update(sha256_file(path).encode())
            h.update(b"\n")
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def file_text(relpath: str) -> str:
    path = AGENT_WS / relpath
    if not path.exists():
        return ""
    return path.read_text()


def lower_text(relpath: str) -> str:
    return file_text(relpath).lower()


def collect_workspace_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    return files


def rel_is_allowed(relpath: str, allowed_paths: list[str]) -> bool:
    for allowed in allowed_paths:
        allowed = allowed.rstrip("/")
        if relpath == allowed or relpath.startswith(f"{allowed}/"):
            return True
    return False


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def any_phrase(text: str, phrases: list[str]) -> bool:
    hay = normalize_text(text)
    return any(normalize_text(phrase) in hay for phrase in phrases)


def import_from_workspace(module_name: str, relpath: str) -> Any:
    path = AGENT_WS / relpath
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {relpath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class ScoreState:
    breakdown: dict[str, int] = field(default_factory=dict)
    bands: dict[str, str] = field(default_factory=dict)
    milestones: dict[str, bool] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    integrity_rules_fired: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_m_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    shortcut_detected: bool = False
    submission_present: bool = False

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

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)

    def final_score(self) -> int:
        if self.integrity_flag == 1:
            return 0
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        if self.integrity_flag == 1:
            return 0.0
        capped = max(0, min(self.raw_m_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def baseline_and_integrity(
    state: ScoreState, gold: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    current_files = collect_workspace_files(AGENT_WS)
    baseline_files = manifest.get("file_hashes", {})
    allowed_write_paths = gold.get("allowed_write_paths", [])

    changed_paths: list[str] = []
    for relpath, sha in baseline_files.items():
        current_sha = current_files.get(relpath)
        if current_sha != sha:
            changed_paths.append(relpath)
    for relpath in current_files:
        if relpath not in baseline_files:
            changed_paths.append(relpath)

    state.submission_present = any(
        rel_is_allowed(path, allowed_write_paths) for path in changed_paths
    )
    for output_path in [
        "artifacts/review/thread_triage.json",
        "artifacts/review/reply_draft.md",
        "artifacts/review/remediation_report.md",
    ]:
        if (AGENT_WS / output_path).exists():
            state.submission_present = True

    for relpath, expected_hash in gold.get("readonly_tree_hashes", {}).items():
        actual_hash = tree_hash(AGENT_WS, relpath)
        if actual_hash != expected_hash:
            if relpath == "tests/test_approval_router.py":
                state.raise_integrity("tests_modified")
            else:
                state.raise_integrity("immutable_slice_mutated")

    for relpath in changed_paths:
        if not rel_is_allowed(relpath, allowed_write_paths):
            state.raise_integrity("write_outside_whitelist")

    for shim in SHIM_FILES:
        if list(AGENT_WS.rglob(shim)):
            state.raise_integrity("pytest_shim")
            break

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    if not state.submission_present:
        state.apply_ceiling("no_submission", 0)

    return {
        "changed_paths": sorted(set(changed_paths)),
    }


def score_triage(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    triage_path = AGENT_WS / "artifacts/review/thread_triage.json"
    if not triage_path.exists():
        state.apply_ceiling("wrong_triage", 20)
        return {"triage_correct": False, "duplicate_correct": False}

    try:
        triage_data = json.loads(triage_path.read_text())
    except json.JSONDecodeError:
        state.errors.append("thread_triage.json is not valid JSON")
        state.apply_ceiling("wrong_triage", 20)
        return {"triage_correct": False, "duplicate_correct": False}

    entries = {}
    triage_items = []
    if isinstance(triage_data, list):
        triage_items = triage_data
    elif isinstance(triage_data, dict):
        maybe_threads = triage_data.get("threads")
        if isinstance(maybe_threads, list):
            triage_items = maybe_threads

    for item in triage_items:
        if isinstance(item, dict) and "thread_id" in item:
            entries[item["thread_id"]] = item

    expected = gold.get("expected_triage", {})
    correct = 0
    for thread_id, expectation in expected.items():
        actual = entries.get(thread_id, {})
        actual_disposition = actual.get("disposition") or actual.get("classification")
        if actual_disposition == expectation.get("disposition"):
            correct += 1

    if correct == len(expected):
        state.add("triage.classification", 25)
        triage_correct = True
    else:
        triage_correct = False
        state.apply_ceiling("wrong_triage", 20)

    duplicate_entry = entries.get("T-214-02", {})
    duplicate_target = duplicate_entry.get("duplicate_of")
    if duplicate_target is None and any_phrase(
        str(duplicate_entry.get("rationale", "")),
        ["duplicate of t-214-01", "same as t-214-01", "same fix as t-214-01"],
    ):
        duplicate_target = "T-214-01"
    duplicate_correct = (
        (duplicate_entry.get("disposition") or duplicate_entry.get("classification")) == "duplicate"
        and duplicate_target == "T-214-01"
        and len(str(duplicate_entry.get("rationale", "")).strip()) >= 12
    )
    if duplicate_correct:
        state.add("triage.duplicate_mapping", 10)
    else:
        state.apply_ceiling("missing_duplicate_mapping", 25)

    if triage_path.exists():
        state.milestones["triage_file_present"] = True

    return {
        "triage_correct": triage_correct,
        "duplicate_correct": duplicate_correct,
    }


def run_hidden_tests(state: ScoreState) -> bool:
    hidden_tests_dir = VERIFIER_DATA / VARIANT_ID / "hidden_tests"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(AGENT_WS)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix=f"{VARIANT_ID}_hidden_") as tmp:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(hidden_tests_dir),
            ],
            cwd=AGENT_WS,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            state.errors.append("hidden tests failed")
            return False
    return True


def score_behavior(state: ScoreState) -> dict[str, bool]:
    behavior_ok = False
    router_ok = False
    legacy_alias_present = "legacy_preview_hint" in file_text("src/policy/preview.py") or (
        "legacy_preview_hint" in file_text("src/policy/approval_router.py")
    )
    try:
        sys.path.insert(0, str(AGENT_WS))
        router_module = import_from_workspace(
            "workspace_approval_router", "src/policy/approval_router.py"
        )
        fallback = router_module.build_policy_preview(
            request_id="req-214",
            approval_state="manual_review",
            preview_enabled=False,
            actor="review-bot",
        )
        live = router_module.build_policy_preview(
            request_id="req-214",
            approval_state="manual_review",
            preview_enabled=True,
            actor="review-bot",
        )
        behavior_ok = (
            fallback.get("status") == "preview_unavailable"
            and fallback.get("approval_state") == "human_review_required"
            and fallback.get("requires_human_review") is True
            and fallback.get("actor") == "review-bot"
        )
        router_ok = (
            live.get("approval_state") == "human_review_required"
            and live.get("preview_path") == "/preview/req-214"
            and live.get("requires_human_review") is True
        )
    except Exception as exc:  # pragma: no cover - scorer guardrail
        state.errors.append(f"behavior import failed: {exc}")

    hidden_tests_ok = run_hidden_tests(state)
    if behavior_ok and hidden_tests_ok:
        state.add("runtime.fallback_contract", 20)
    else:
        state.apply_ceiling("missing_fallback_fix", 30)

    if router_ok:
        state.add("runtime.router_normalization", 10)

    if legacy_alias_present:
        state.apply_ceiling("legacy_alias_reintroduced", 20)

    return {
        "behavior_ok": behavior_ok,
        "router_ok": router_ok,
        "hidden_tests_ok": hidden_tests_ok,
    }


def score_regression_test(state: ScoreState) -> bool:
    text = file_text("tests/test_preview.py")
    text_norm = normalize_text(text)
    passed = (
        "requires_human_review" in text
        and '"human_review_required"' in text
        and "preview_unavailable" in text
        and any(
            marker in text_norm
            for marker in [
                "preview_enabled=false",
                "preview_enabled = false",
                "build_policy_preview(",
            ]
        )
        and any(
            marker in text_norm
            for marker in [
                "manual_review",
                'approval_state="manual_review"',
                "approval_state = \"manual_review\"",
            ]
        )
    )
    if passed:
        state.add("tests.preview_regression", 10)
    return passed


def score_docs(state: ScoreState, gold: dict[str, Any]) -> bool:
    docs_text = lower_text("docs/approval_policy.md")
    passed = all(
        phrase.lower() in docs_text for phrase in gold.get("required_docs_phrases", [])
    )
    if passed:
        state.add("docs.fallback_alignment", 10)
    return passed


def score_replies(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    reply_path = AGENT_WS / "artifacts/review/reply_draft.md"
    mapping_ok = False
    specificity_ok = False
    if reply_path.exists():
        reply = reply_path.read_text()
        required_ids = gold.get("required_reply_thread_ids", [])
        mapping_ok = all(thread_id in reply for thread_id in required_ids)
        duplicate_ok = "duplicate of `T-214-01`" in reply or "duplicate of T-214-01" in reply
        obsolete_ok = "T-214-03" in reply and any_phrase(
            reply,
            [
                "Did not apply",
                "did not apply",
                "Not applied",
                "not applied",
                "did not revive",
            ],
        )
        mapping_ok = mapping_ok and duplicate_ok and obsolete_ok
        specificity_ok = all(
            token in reply
            for token in [
                "src/policy/approval_router.py",
                "src/policy/preview.py",
                "tests/test_preview.py",
                "docs/approval_policy.md",
            ]
        )
    if mapping_ok:
        state.add("reply.thread_mapping", 5)
    else:
        state.apply_ceiling("generic_replies", 35)
    if specificity_ok:
        state.add("reply.specificity", 5, band="P_only")
    return {"mapping_ok": mapping_ok, "specificity_ok": specificity_ok}


def score_report(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    report_path = AGENT_WS / "artifacts/review/remediation_report.md"
    scope_ok = False
    variant_ok = False
    if report_path.exists():
        report = report_path.read_text()
        scope_ok = all(
            phrase in report for phrase in gold.get("report_scope_phrases", [])
        ) and any_phrase(
            report,
            [
                "Did not apply",
                "did not apply",
                "Not applied",
                "intentionally not applied",
                "left unchanged as required",
            ],
        )
        variant_ok = all(
            phrase in report for phrase in gold.get("required_report_phrases", [])
        )
    if scope_ok:
        state.add("report.scope", 5)
    else:
        state.apply_ceiling("missing_report_scope", 80)
    if variant_ok:
        state.add("report.variant_awareness", 5, band="P_only")
    else:
        if VARIANT_ID == "v4-multi-corpus-objective":
            state.apply_ceiling("objective_contract_missed", 15)
        elif VARIANT_ID == "v5-recovery-in-thread":
            state.apply_ceiling("incident_blind_recovery", 10)
    return {"scope_ok": scope_ok, "variant_ok": variant_ok}


def build_milestones(state: ScoreState, checks: dict[str, bool]) -> tuple[dict[str, bool], dict[str, Any]]:
    m1 = bool(state.milestones.get("triage_file_present") and checks["reply_mapping"])
    m2 = bool(checks["behavior_ok"] and checks["router_ok"])
    m3 = state.integrity_flag == 0
    m4 = bool(m2 and checks["test_ok"] and checks["docs_ok"] and checks["reply_mapping"])
    m5 = bool(m4 and checks["triage_correct"] and checks["report_scope"])
    milestones = {
        "M1_localization": m1,
        "M2_primary_fix": m2,
        "M3_invariants": m3,
        "M4_functional": m4,
        "M5_e2e": m5,
    }
    if state.integrity_flag == 1:
        milestones["M3_invariants"] = False
        milestones["M4_functional"] = False
        milestones["M5_e2e"] = False
    slots = []
    total = 0.0
    for name, weight in MILESTONE_SLOTS:
        passed = bool(milestones[name])
        slots.append({"name": name, "weight": weight, "passed_bool": passed})
        if passed:
            total += weight
    return milestones, {"slots": slots, "M_aggregate": round(total, 4)}


def main() -> int:
    gold = load_json(VERIFIER_DATA / VARIANT_ID / "gold_review_state.json")
    manifest = load_json(VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json")
    state = ScoreState()

    baseline_and_integrity(state, gold, manifest)
    triage = score_triage(state, gold)
    behavior = score_behavior(state)
    test_ok = score_regression_test(state)
    docs_ok = score_docs(state, gold)
    replies = score_replies(state, gold)
    report = score_report(state, gold)

    checks = {
        "triage_correct": triage["triage_correct"],
        "behavior_ok": behavior["behavior_ok"] and behavior["hidden_tests_ok"],
        "router_ok": behavior["router_ok"],
        "test_ok": test_ok,
        "docs_ok": docs_ok,
        "reply_mapping": replies["mapping_ok"],
        "report_scope": report["scope_ok"],
    }
    milestones, milestone_vector = build_milestones(state, checks)
    P_benchmark = state.final_score()
    M_training = state.final_m_training()
    pass_bar = int(gold.get("pass_bar", PASS_BAR))
    passed = P_benchmark >= pass_bar and state.integrity_flag == 0 and milestones["M5_e2e"]

    result = {
        "schema_version": VERIFY_RESULT_SCHEMA,
        "score": P_benchmark,
        "P_benchmark": P_benchmark,
        "M_training": M_training,
        "raw_score_pre_ceiling": state.raw_score,
        "pass": passed,
        "pass_bar": pass_bar,
        "shortcut_detected": state.shortcut_detected,
        "ceilings_applied": state.ceilings_applied,
        "milestones": milestones,
        "milestone_vector": milestone_vector,
        "breakdown": {
            **state.breakdown,
            "__bands": state.bands,
        },
        "errors": state.errors,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
