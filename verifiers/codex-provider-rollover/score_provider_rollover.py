#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

GOLD_PATH = VERIFIER_DATA / VARIANT_ID / "gold_reference.json"
MANIFEST_PATH = VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json"

CONFIG_PATH = AGENT_WS / "serving_maintenance" / ".codex" / "config.toml"
SMOKE_SCRIPT = AGENT_WS / "serving_maintenance" / "scripts" / "smoke_responses_profile.py"
DOC_PROVIDER = AGENT_WS / "serving_maintenance" / "docs" / "provider_rollover.md"
DOC_SMOKE = AGENT_WS / "serving_maintenance" / "docs" / "smoke.md"
TURN1_FIXTURE = AGENT_WS / "serving_maintenance" / "fixtures" / "http" / "turn1_ok.json"
TURN2_GOOD = AGENT_WS / "serving_maintenance" / "fixtures" / "http" / "turn2_ok.json"

IGNORE_PREFIXES = (".pytest_cache/",)
IGNORE_SUBSTRINGS = ("/__pycache__/",)
IGNORE_SUFFIXES = (".pyc",)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def should_ignore(rel: str) -> bool:
    if rel.endswith(IGNORE_SUFFIXES):
        return True
    if any(part in rel for part in IGNORE_SUBSTRINGS):
        return True
    return rel.startswith(IGNORE_PREFIXES)


def parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text in {"true", "false"}:
        return text == "true"
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        return text


def load_config() -> dict[str, Any]:
    data: dict[str, Any] = {}
    section: list[str] = []
    current: dict[str, Any] = data
    for raw in CONFIG_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].split(".")
            current = data
            for key in section:
                current = current.setdefault(key, {})
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        current[key.strip()] = parse_scalar(value)
    return data


def run_visible_tests() -> tuple[bool, str]:
    proc = subprocess.run(
        [str(AGENT_WS / "bin" / "run-visible-tests")],
        cwd=AGENT_WS,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def run_smoke(turn2_path: Path) -> tuple[bool, dict[str, Any] | None, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(SMOKE_SCRIPT),
            "--config",
            str(CONFIG_PATH),
            "--turn1",
            str(TURN1_FIXTURE),
            "--turn2",
            str(turn2_path),
        ],
        cwd=AGENT_WS,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = None
    stdout = proc.stdout.strip()
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
    combined = "\n".join(chunk for chunk in [proc.stdout.strip(), proc.stderr.strip()] if chunk).strip()
    return proc.returncode == 0, payload, combined


def compute_manifest_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(p for p in AGENT_WS.rglob("*") if p.is_file()):
        rel = path.relative_to(AGENT_WS).as_posix()
        if should_ignore(rel):
            continue
        hashes[rel] = sha256_file(path)
    return hashes


@dataclass
class ScoreState:
    breakdown: dict[str, int] = field(default_factory=dict)
    breakdown_bands: dict[str, str] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    integrity_rules_fired: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_M_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    shortcut_detected: bool = False
    errors: list[str] = field(default_factory=list)

    def add(self, key: str, points: int, passed: bool, band: str = "M") -> None:
        self.checks[key] = bool(passed)
        if not passed or points <= 0:
            return
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.breakdown_bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_M_score += points

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
        if self.integrity_flag:
            return 0
        return max(0, min(self.raw_score, self.ceiling_cap))


def apply_integrity(state: ScoreState, gold: dict[str, Any], manifest: dict[str, Any]) -> None:
    current = compute_manifest_hashes()
    baseline = manifest.get("files", {})
    allowed_writes = set(gold.get("allowed_writes", []))
    readonly_roots = tuple(gold.get("readonly_roots", []))
    changed: list[str] = []

    for rel, baseline_hash in baseline.items():
        now_hash = current.get(rel)
        if now_hash != baseline_hash:
            changed.append(rel)
    for rel in current:
        if rel not in baseline:
            changed.append(rel)

    for rel in sorted(set(changed)):
        if should_ignore(rel):
            continue
        if rel in allowed_writes:
            continue
        state.raise_integrity("write_outside_whitelist")
        if rel.startswith(readonly_roots):
            state.raise_integrity("immutable_slice_mutated")
        if rel.startswith("tests/"):
            state.raise_integrity("tests_modified")

    for rel in baseline:
        if rel.startswith(readonly_roots) and current.get(rel) != baseline[rel]:
            state.raise_integrity("immutable_slice_mutated")
        if rel.startswith("tests/") and current.get(rel) != baseline[rel]:
            state.raise_integrity("tests_modified")

    for shim in ("pytest.py", "sitecustomize.py", "usercustomize.py"):
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim")
    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")


def score_workspace() -> dict[str, Any]:
    gold = load_json(GOLD_PATH)
    manifest = load_json(MANIFEST_PATH)
    state = ScoreState()

    apply_integrity(state, gold, manifest)

    visible_ok, visible_output = run_visible_tests()
    state.add("visible_tests_pass", 25, visible_ok)
    if not visible_ok:
        state.apply_ceiling("no_visible_green", 25)

    try:
        config = load_config()
    except Exception as exc:  # pragma: no cover - malformed config is part of scoring
        state.errors.append(f"config_load_failed: {exc}")
        config = {}

    expected_provider = gold["expected_provider"]
    expected_base_url = gold["expected_base_url"]
    expected_env_key = gold["expected_env_key"]
    providers = config.get("model_providers", {})
    selected_provider = config.get("provider")
    selected_stanza = providers.get(selected_provider, {}) if isinstance(providers, dict) else {}

    config_selected_provider = selected_provider == expected_provider
    state.add("config_selected_provider", 10, config_selected_provider)
    if not config_selected_provider:
        state.apply_ceiling("legacy_default_selected", 20)

    config_proxy_contract = (
        config_selected_provider
        and isinstance(selected_stanza, dict)
        and selected_stanza.get("base_url") == expected_base_url
        and selected_stanza.get("wire_api") == "responses"
        and selected_stanza.get("env_key") == expected_env_key
        and selected_stanza.get("store") is True
    )
    state.add("config_proxy_contract", 10, config_proxy_contract)
    if not config_proxy_contract:
        state.apply_ceiling("proxy_route_incorrect", 30)

    smoke_good_ok, smoke_good_payload, smoke_good_output = run_smoke(TURN2_GOOD)
    smoke_good_pass = (
        smoke_good_ok
        and isinstance(smoke_good_payload, dict)
        and smoke_good_payload.get("selected_provider") == expected_provider
        and smoke_good_payload.get("base_url") == expected_base_url
        and smoke_good_payload.get("continuity_ok") is True
        and smoke_good_payload.get("store_ok") is True
    )
    state.add("smoke_good_pass", 15, smoke_good_pass)

    hidden_failures: list[str] = []
    for rel in gold.get("negative_smoke_fixtures", []):
        ok, _, output = run_smoke(VERIFIER_DATA / VARIANT_ID / "hidden_fixtures" / rel)
        if ok:
            hidden_failures.append(rel)
            state.errors.append(f"negative_smoke_fixture_accepted: {rel}")
        elif not output:
            state.errors.append(f"negative_smoke_fixture_rejected_without_output: {rel}")
    smoke_hidden_rejections = not hidden_failures
    state.add("smoke_hidden_rejections", 10, smoke_hidden_rejections)
    if not smoke_good_pass or not smoke_hidden_rejections:
        state.apply_ceiling("continuity_not_exact", 20)

    config_text = CONFIG_PATH.read_text()
    preserve_block_exact = (
        config_selected_provider
        and config_proxy_contract
        and gold["preserve_block"] in config_text
    )
    state.add("preserve_block_exact", 15, preserve_block_exact)
    if not preserve_block_exact:
        state.apply_ceiling("config_rewritten", 30)

    docs_text = DOC_PROVIDER.read_text() + "\n" + DOC_SMOKE.read_text()
    docs_text_lower = docs_text.lower()
    docs_search_text = docs_text_lower.replace("`", "")
    docs_core_sync = (
        config_selected_provider
        and expected_provider in docs_text
        and expected_base_url in docs_text
        and "previous_response_id" in docs_text
    )
    state.add("docs_core_sync", 10, docs_core_sync)
    if not docs_core_sync:
        state.apply_ceiling("docs_unaligned", 40)

    docs_variant_sync = all(keyword.lower() in docs_search_text for keyword in gold.get("required_doc_keywords", []))
    state.add("docs_variant_sync", 5, docs_variant_sync)

    variant_ceiling = gold.get("variant_ceiling")
    if variant_ceiling and not docs_variant_sync:
        state.apply_ceiling(variant_ceiling["name"], int(variant_ceiling["cap"]))

    if VARIANT_ID == "v2-noisy-distractor" and "maintenance_canary" in docs_search_text and expected_provider not in docs_search_text:
        state.apply_ceiling("canary_anchoring", 35)

    if VARIANT_ID == "v3-dirty-state":
        template_markers = ("generated from template", "legacy_profile_template")
        if any(marker in config_text for marker in template_markers):
            state.apply_ceiling("template_overwrite", 30)

    final_score = state.final_score()
    m_training = round(final_score / 100.0, 4)
    milestones = {
        "M1_localization": state.checks.get("config_selected_provider", False) or state.checks.get("docs_core_sync", False),
        "M2_primary_fix": all(
            [
                state.checks.get("visible_tests_pass", False),
                state.checks.get("config_selected_provider", False),
                state.checks.get("config_proxy_contract", False),
                state.checks.get("smoke_good_pass", False),
            ]
        ),
        "M3_invariants": state.integrity_flag == 0 and not state.shortcut_detected,
        "M4_functional": all(
            [
                state.checks.get("visible_tests_pass", False),
                state.checks.get("smoke_hidden_rejections", False),
                state.checks.get("docs_core_sync", False),
                state.integrity_flag == 0,
            ]
        ),
        "M5_e2e": all(
            [
                state.checks.get("visible_tests_pass", False),
                state.checks.get("config_selected_provider", False),
                state.checks.get("config_proxy_contract", False),
                state.checks.get("smoke_good_pass", False),
                state.checks.get("smoke_hidden_rejections", False),
                state.checks.get("preserve_block_exact", False),
                state.checks.get("docs_variant_sync", False),
                state.integrity_flag == 0,
                final_score >= gold.get("pass_threshold", 90),
            ]
        ),
    }

    return {
        "schema_version": "cnb55.verify_result.v3",
        "family_id": gold["family_id"],
        "variant_id": VARIANT_ID,
        "score": final_score,
        "P_benchmark": final_score,
        "M_training": m_training,
        "pass": final_score >= gold.get("pass_threshold", 90) and state.integrity_flag == 0,
        "integrity_flag": state.integrity_flag,
        "shortcut_detected": state.shortcut_detected,
        "breakdown": state.breakdown,
        "breakdown_bands": state.breakdown_bands,
        "checks": state.checks,
        "ceilings_applied": state.ceilings_applied,
        "integrity_rules_fired": state.integrity_rules_fired,
        "milestones": milestones,
        "visible_test_output": visible_output,
        "smoke_good_output": smoke_good_output,
        "errors": state.errors,
    }


def main() -> int:
    result = score_workspace()
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
