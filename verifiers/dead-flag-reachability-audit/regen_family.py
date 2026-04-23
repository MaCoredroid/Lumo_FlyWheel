#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY_ID = "dead-flag-reachability-audit"
FAMILY = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
VERIFIER = REPO / "verifiers" / FAMILY_ID
VERIFIER_DATA = REPO / "verifier_data" / FAMILY_ID
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"
SCORER = VERIFIER / "score_reachability.py"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

FLAG_IDS = [
    "ENABLE_SHADOW_PREVIEW",
    "ENABLE_PREVIEW_V2",
    "PREVIEW_FORCE_LEGACY",
]

READONLY_TREES = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "bin",
    "config",
    "src",
    "docs",
    "release_context",
    "incident_context",
    "repo_evidence",
    "tests",
]

IGNORED_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.exists():
        return None
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_file(target).encode())
        return h.hexdigest()
    for path in sorted(target.rglob("*")):
        if any(part in IGNORED_NAMES for part in path.parts):
            continue
        if path.suffix == ".pyc":
            continue
        rel_path = path.relative_to(target).as_posix()
        if path.is_file():
            h.update(b"F:" + rel_path.encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
        elif path.is_dir():
            h.update(b"D:" + rel_path.encode() + b"\x00")
    return h.hexdigest()


def list_files(root: Path) -> list[str]:
    out: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORED_NAMES for part in path.parts):
            continue
        if path.suffix == ".pyc":
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith("brief/"):
            continue
        out.append(rel)
    return out


def write(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        path.chmod(0o755)


CLI_TEMPLATE = """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "cnb55.flag_audit.v1"
FLAG_IDS = ("ENABLE_SHADOW_PREVIEW", "ENABLE_PREVIEW_V2", "PREVIEW_FORCE_LEGACY")
STATUS_VALUES = ("live", "partial", "dead")
ACTION_VALUES = ("keep", "deprecate", "docs_cleanup", "telemetry_first", "remove_after_migration", "do_not_remove_now")
EVIDENCE_ROOTS = ("config", "src", "docs", "tests", "release_context", "incident_context", "repo_evidence")


def schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "schema_version",
            "variant_id",
            "flags",
            "summary",
            "cleanup_plan",
            "assumption_ledger",
        ],
        "additionalProperties": False,
    }


def validate(doc: Any, root: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["top-level value must be an object"]
    required = set(schema()["required"])
    missing = sorted(required - set(doc))
    extra = sorted(set(doc) - required)
    for field in missing:
        errors.append(f"missing field: {field}")
    for field in extra:
        errors.append(f"unexpected field: {field}")
    if errors:
        return errors

    if doc.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    variant = (root / ".scenario_variant").read_text().strip()
    if doc.get("variant_id") != variant:
        errors.append("variant_id mismatch")

    flags = doc.get("flags")
    if not isinstance(flags, list) or len(flags) != len(FLAG_IDS):
        errors.append("flags must contain exactly three entries")
    else:
        seen: set[str] = set()
        for entry in flags:
            if not isinstance(entry, dict):
                errors.append("flag entries must be objects")
                continue
            for field in ("flag", "status", "alias_of", "parser_symbol", "runtime_branch_symbol", "evidence", "disproved_false_positive_path", "rationale"):
                if field not in entry:
                    errors.append(f"flag entry missing {field}")
            flag = entry.get("flag")
            if flag not in FLAG_IDS:
                errors.append(f"unknown flag: {flag}")
            elif flag in seen:
                errors.append(f"duplicate flag: {flag}")
            else:
                seen.add(flag)
            if entry.get("status") not in STATUS_VALUES:
                errors.append(f"bad status for {flag}")
            alias_of = entry.get("alias_of")
            if alias_of is not None and alias_of not in FLAG_IDS:
                errors.append(f"bad alias_of for {flag}")
            for nullable in ("parser_symbol", "runtime_branch_symbol"):
                value = entry.get(nullable)
                if value is not None and (not isinstance(value, str) or len(value.strip()) < 3):
                    errors.append(f"{nullable} must be null or non-empty string for {flag}")
            evidence = entry.get("evidence")
            if not isinstance(evidence, list) or len(evidence) < 2:
                errors.append(f"evidence must list at least two citations for {flag}")
            else:
                for rel in evidence:
                    if not isinstance(rel, str):
                        errors.append(f"evidence citation must be string for {flag}")
                        continue
                    if not any(rel == prefix or rel.startswith(prefix + "/") for prefix in EVIDENCE_ROOTS):
                        errors.append(f"citation outside evidence roots: {rel}")
                    if not (root / rel).exists():
                        errors.append(f"missing citation path: {rel}")
            false_path = entry.get("disproved_false_positive_path")
            if not isinstance(false_path, str) or not false_path.strip():
                errors.append(f"disproved_false_positive_path must be non-empty for {flag}")
            elif not (root / false_path).exists():
                errors.append(f"false-positive path missing: {false_path}")
            rationale = entry.get("rationale")
            if not isinstance(rationale, str) or len(rationale.strip()) < 40:
                errors.append(f"rationale too short for {flag}")

    summary = doc.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be object")
    else:
        for field in ("highest_operational_risk", "note"):
            if field not in summary:
                errors.append(f"summary missing {field}")
        if summary.get("highest_operational_risk") not in FLAG_IDS:
            errors.append("summary.highest_operational_risk must name a flag")
        note = summary.get("note")
        if not isinstance(note, str) or len(note.strip()) < 30:
            errors.append("summary.note too short")

    cleanup_plan = doc.get("cleanup_plan")
    if not isinstance(cleanup_plan, list) or len(cleanup_plan) < 2:
        errors.append("cleanup_plan must have at least two actions")
    else:
        for row in cleanup_plan:
            if not isinstance(row, dict):
                errors.append("cleanup row must be object")
                continue
            for field in ("flag", "action", "rationale", "blockers"):
                if field not in row:
                    errors.append(f"cleanup row missing {field}")
            if row.get("flag") not in FLAG_IDS:
                errors.append(f"cleanup flag unknown: {row.get('flag')}")
            if row.get("action") not in ACTION_VALUES:
                errors.append(f"cleanup action invalid: {row.get('action')}")
            if not isinstance(row.get("rationale"), str) or len(row["rationale"].strip()) < 20:
                errors.append("cleanup rationale too short")
            blockers = row.get("blockers")
            if not isinstance(blockers, list):
                errors.append("cleanup blockers must be list")

    ledger = doc.get("assumption_ledger")
    if not isinstance(ledger, list) or not ledger:
        errors.append("assumption_ledger must be non-empty")
    else:
        if not any(isinstance(row, dict) and row.get("status") == "missing" for row in ledger):
            errors.append("assumption_ledger must include one missing row")
        for row in ledger:
            if not isinstance(row, dict):
                errors.append("assumption row must be object")
                continue
            if row.get("status") not in ("observed", "to_verify", "missing"):
                errors.append("assumption status invalid")
            if not isinstance(row.get("topic"), str) or not row["topic"].strip():
                errors.append("assumption topic required")
            if not isinstance(row.get("note"), str) or len(row["note"].strip()) < 8:
                errors.append("assumption note too short")
    return errors


def render_md(doc: dict[str, Any]) -> str:
    lines = [
        "# Flag Audit",
        "",
        f"- variant: `{doc['variant_id']}`",
        f"- highest_operational_risk: `{doc['summary']['highest_operational_risk']}`",
        "",
        "## Flag Status",
        "",
        "| flag | status | alias_of | parser_symbol | runtime_branch_symbol | false_positive_path |",
        "|---|---|---|---|---|---|",
    ]
    for entry in doc["flags"]:
        lines.append(
            f"| `{entry['flag']}` | {entry['status']} | {entry['alias_of'] or '—'} | "
            f"`{entry['parser_symbol'] or '—'}` | `{entry['runtime_branch_symbol'] or '—'}` | "
            f"`{entry['disproved_false_positive_path']}` |"
        )
        lines.append("")
        lines.append(f"Evidence: {', '.join(f'`{rel}`' for rel in entry['evidence'])}")
        lines.append("")
        lines.append(entry["rationale"])
        lines.append("")
    lines.extend([
        "## Cleanup Plan",
        "",
        "| flag | action | blockers | rationale |",
        "|---|---|---|---|",
    ])
    for row in doc["cleanup_plan"]:
        blockers = ", ".join(row["blockers"]) if row["blockers"] else "none"
        lines.append(f"| `{row['flag']}` | `{row['action']}` | {blockers} | {row['rationale']} |")
    lines.extend([
        "",
        "## Assumption Ledger",
        "",
        "| status | topic | note |",
        "|---|---|---|",
    ])
    for row in doc["assumption_ledger"]:
        lines.append(f"| {row['status']} | {row['topic']} | {row['note']} |")
    lines.extend(["", doc["summary"]["note"], ""])
    return "\\n".join(lines)


def write_outputs(doc: dict[str, Any], root: Path) -> None:
    brief_dir = root / "brief"
    art_dir = root / "artifacts"
    brief_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    (brief_dir / "flag_audit.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\\n")
    (art_dir / "reachability_matrix.json").write_text(json.dumps({"flags": doc["flags"]}, indent=2, sort_keys=True) + "\\n")
    (art_dir / "flag_audit.md").write_text(render_md(doc))
    cleanup_lines = ["# Cleanup Patch Plan", ""]
    for row in doc["cleanup_plan"]:
        cleanup_lines.append(f"## {row['flag']}")
        cleanup_lines.append("")
        cleanup_lines.append(f"- action: `{row['action']}`")
        cleanup_lines.append(f"- blockers: {', '.join(row['blockers']) if row['blockers'] else 'none'}")
        cleanup_lines.append(f"- rationale: {row['rationale']}")
        cleanup_lines.append("")
    (art_dir / "cleanup.patchplan.md").write_text("\\n".join(cleanup_lines) + "\\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("schema")
    validate_p = sub.add_parser("validate")
    validate_p.add_argument("input")
    submit_p = sub.add_parser("submit")
    submit_p.add_argument("input")
    args = parser.parse_args()

    root = Path.cwd()
    if args.cmd == "schema":
        print(json.dumps(schema(), indent=2, sort_keys=True))
        return 0

    doc = json.loads(Path(args.input).read_text())
    errors = validate(doc, root)
    if errors:
        for err in errors:
            print(err)
        return 1
    if args.cmd == "submit":
        write_outputs(doc, root)
    else:
        print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


CONFIG_DEFAULTS = """[preview]
default_mode = "legacy"
enable_shadow_preview = false
enable_preview_v2 = false
preview_force_legacy = false
"""


SRC_CONFIG = """from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class PreviewConfig:
    default_mode: str = "legacy"
    effective_mode: str = "legacy"
    shadow_enabled: bool = False
    force_legacy_seen: bool = False
    parser_hits: dict[str, str] = field(default_factory=dict)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_preview_env(env: Mapping[str, str]) -> PreviewConfig:
    config = PreviewConfig()
    if _truthy(env.get("ENABLE_SHADOW_PREVIEW")):
        config.shadow_enabled = True
        config.effective_mode = "shadow"
        config.parser_hits["ENABLE_SHADOW_PREVIEW"] = "load_preview_env:ENABLE_SHADOW_PREVIEW"
    elif _truthy(env.get("ENABLE_PREVIEW_V2")):
        config.shadow_enabled = True
        config.effective_mode = "shadow"
        config.parser_hits["ENABLE_PREVIEW_V2"] = "load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW"
    if _truthy(env.get("PREVIEW_FORCE_LEGACY")):
        config.force_legacy_seen = True
        config.parser_hits["PREVIEW_FORCE_LEGACY"] = "load_preview_env:PREVIEW_FORCE_LEGACY"
    return config
"""


SRC_RUNTIME = """from __future__ import annotations

from preview.config import PreviewConfig


def preview_runtime_branch(config: PreviewConfig) -> str:
    if config.shadow_enabled:
        return "preview_runtime_branch:shadow_preview_path"
    return "preview_runtime_branch:legacy_preview_path"
"""


SRC_SERVICE = """from __future__ import annotations

from preview.config import PreviewConfig
from preview.runtime import preview_runtime_branch


def build_preview_plan(config: PreviewConfig) -> dict[str, object]:
    branch = preview_runtime_branch(config)
    return {
        "branch": branch,
        "shadow_enabled": config.shadow_enabled,
        "effective_mode": config.effective_mode,
        "force_legacy_seen": config.force_legacy_seen,
    }
"""


SRC_LEGACY = """from __future__ import annotations


def legacy_force_label(force_legacy_seen: bool) -> str:
    return "legacy-forced" if force_legacy_seen else "legacy-default"
"""


SRC_CLI = """from __future__ import annotations

import json
import os

from preview.config import load_preview_env
from preview.service import build_preview_plan


def main() -> int:
    config = load_preview_env(os.environ)
    print(json.dumps(build_preview_plan(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


TEST_ALIAS = """from preview.config import load_preview_env
from preview.service import build_preview_plan


def test_preview_v2_alias_maps_to_shadow_path():
    config = load_preview_env({"ENABLE_PREVIEW_V2": "1"})
    plan = build_preview_plan(config)
    assert config.parser_hits["ENABLE_PREVIEW_V2"] == "load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW"
    assert plan["branch"] == "preview_runtime_branch:shadow_preview_path"
"""


TEST_LIVE = """from preview.config import load_preview_env
from preview.service import build_preview_plan


def test_shadow_preview_changes_runtime_branch():
    config = load_preview_env({"ENABLE_SHADOW_PREVIEW": "1"})
    plan = build_preview_plan(config)
    assert config.parser_hits["ENABLE_SHADOW_PREVIEW"] == "load_preview_env:ENABLE_SHADOW_PREVIEW"
    assert plan["branch"] == "preview_runtime_branch:shadow_preview_path"
"""


TEST_FORCE_LEGACY = """from preview.config import load_preview_env
from preview.legacy import legacy_force_label
from preview.service import build_preview_plan


def test_force_legacy_flag_is_tracked_for_reporting_only():
    config = load_preview_env({"PREVIEW_FORCE_LEGACY": "1"})
    plan = build_preview_plan(config)
    assert config.parser_hits["PREVIEW_FORCE_LEGACY"] == "load_preview_env:PREVIEW_FORCE_LEGACY"
    assert plan["branch"] == "preview_runtime_branch:legacy_preview_path"
    assert legacy_force_label(config.force_legacy_seen) == "legacy-forced"
"""


DOC_RUNBOOK = """# Preview Rollout Runbook

The live preview router branches on `preview_runtime_branch:shadow_preview_path`
when `ENABLE_SHADOW_PREVIEW` is enabled.

`ENABLE_PREVIEW_V2` is still accepted by the env parser for legacy deploy
manifests, but it normalizes to the same shadow path and is not a standalone
runtime branch.

`PREVIEW_FORCE_LEGACY` is left in reporting and operator notes only. The live
service does not branch on it anymore.
"""


DOC_FALSE_POSITIVE = """# False Positive Notes

The presence of a parser hit does not prove a runtime branch.
The presence of a test helper does not prove the live service reads that flag.
Docs with 2025 or rollback language are not stronger than the active runtime.
"""


V2_STALE_DOC = """# Preview V2 Ramp Memo (Stale)

Q4 2025 note: keep `ENABLE_PREVIEW_V2` listed as the public rollout control in
dashboards until the shadow preview rename lands everywhere. This memo was not
updated after the rename and should not be used as the live runtime source of
truth.
"""


V3_PREVIOUS_AUDIT = """# Previous Session Audit Draft

Draft claim: every parsed env var is effectively live because operators can set
it and the preview service still mentions it in notes. This draft was never
accepted and does not trace the runtime branch.
"""


V3_PATCH = """diff --git a/src/preview/runtime.py b/src/preview/runtime.py
--- a/src/preview/runtime.py
+++ b/src/preview/runtime.py
@@
 def preview_runtime_branch(config):
+    if config.force_legacy_seen:
+        return "preview_runtime_branch:legacy_force_override"
     if config.shadow_enabled:
         return "preview_runtime_branch:shadow_preview_path"
     return "preview_runtime_branch:legacy_preview_path"
"""


V4_FREEZE = """# Release Freeze Guidance

The team is inside a release freeze. Classification should still call out
alias-only or dead controls correctly, but cleanup actions must prefer docs,
telemetry, and migration notes over removing parser compatibility during the
freeze window.
"""


V4_OBJECTIVE = """# Objective Shift

This audit is now primarily for operator risk reduction before the freeze exit.
Do not recommend deleting alias compatibility that current staging deploy
manifests still depend on.
"""


V5_INCIDENT = """# Incident: Preview Alias Cleanup Rollback

On 2026-04-11, a cleanup removed `ENABLE_PREVIEW_V2` before all deploy manifests
were migrated. Staging silently stopped activating the shadow preview path. The
cleanup was rolled back the same day. Future audits must mention the rollback
before proposing any removal.
"""


V5_THREAD = """# Recovery Thread

The alias is still partial: parser-only compatibility that maps to the live
shadow path. Treat it as migration baggage, not as a dead flag you can delete
immediately.
"""


VARIANT_CONFIG = {
    "v1-clean-baseline": {
        "title": "clean baseline",
        "extra_docs": {},
        "release_context": {},
        "incident_context": {},
        "repo_evidence": {},
        "cleanup_actions": {
            "ENABLE_SHADOW_PREVIEW": "keep",
            "ENABLE_PREVIEW_V2": "deprecate",
            "PREVIEW_FORCE_LEGACY": "docs_cleanup",
        },
        "highest_operational_risk": "ENABLE_PREVIEW_V2",
    },
    "v2-noisy-distractor": {
        "title": "noisy distractor",
        "extra_docs": {
            "docs/preview_v2_ramp_memo_2025.md": V2_STALE_DOC,
        },
        "release_context": {},
        "incident_context": {},
        "repo_evidence": {},
        "cleanup_actions": {
            "ENABLE_SHADOW_PREVIEW": "keep",
            "ENABLE_PREVIEW_V2": "deprecate",
            "PREVIEW_FORCE_LEGACY": "docs_cleanup",
        },
        "highest_operational_risk": "ENABLE_PREVIEW_V2",
    },
    "v3-dirty-state": {
        "title": "dirty state",
        "extra_docs": {},
        "release_context": {},
        "incident_context": {},
        "repo_evidence": {
            "repo_evidence/previous_session/audit_draft.md": V3_PREVIOUS_AUDIT,
            "repo_evidence/in_progress_patch/preview_force_legacy.patch": V3_PATCH,
        },
        "cleanup_actions": {
            "ENABLE_SHADOW_PREVIEW": "keep",
            "ENABLE_PREVIEW_V2": "deprecate",
            "PREVIEW_FORCE_LEGACY": "docs_cleanup",
        },
        "highest_operational_risk": "ENABLE_PREVIEW_V2",
    },
    "v4-multi-corpus-objective": {
        "title": "multi corpus objective",
        "extra_docs": {},
        "release_context": {
            "release_context/release_freeze_guidance.md": V4_FREEZE,
            "release_context/objective_shift.md": V4_OBJECTIVE,
        },
        "incident_context": {},
        "repo_evidence": {
            "repo_evidence/previous_session/audit_draft.md": V3_PREVIOUS_AUDIT,
        },
        "cleanup_actions": {
            "ENABLE_SHADOW_PREVIEW": "keep",
            "ENABLE_PREVIEW_V2": "telemetry_first",
            "PREVIEW_FORCE_LEGACY": "docs_cleanup",
        },
        "highest_operational_risk": "ENABLE_PREVIEW_V2",
    },
    "v5-recovery-in-thread": {
        "title": "recovery in thread",
        "extra_docs": {},
        "release_context": {
            "release_context/release_freeze_guidance.md": V4_FREEZE,
            "release_context/objective_shift.md": V4_OBJECTIVE,
        },
        "incident_context": {
            "incident_context/preview_alias_cleanup_rollback.md": V5_INCIDENT,
            "incident_context/recovery_thread.md": V5_THREAD,
        },
        "repo_evidence": {
            "repo_evidence/previous_session/audit_draft.md": V3_PREVIOUS_AUDIT,
            "repo_evidence/in_progress_patch/preview_force_legacy.patch": V3_PATCH,
        },
        "cleanup_actions": {
            "ENABLE_SHADOW_PREVIEW": "keep",
            "ENABLE_PREVIEW_V2": "remove_after_migration",
            "PREVIEW_FORCE_LEGACY": "do_not_remove_now",
        },
        "highest_operational_risk": "ENABLE_PREVIEW_V2",
    },
}


def agents_md(variant: str) -> str:
    return f"""# Dead Flag Reachability Audit

You are auditing preview rollout controls inside this workspace copy.

Variant: `{variant}`

Do:
- inspect defaults, env parsing, runtime branching, tests, docs, and any extra context dirs
- author `brief_input.json` at the workspace root
- run `./bin/cnb55-flag-audit validate brief_input.json`
- run `./bin/cnb55-flag-audit submit brief_input.json`

Required outputs:
- `artifacts/flag_audit.md`
- `artifacts/reachability_matrix.json`
- `artifacts/cleanup.patchplan.md`

Guardrails:
- classify each named flag as `live`, `partial`, or `dead`
- distinguish parser presence from runtime reachability
- distinguish a live alias from a live standalone flag
- name one disproved false-positive path per flag
- do not modify code, tests, docs, or evidence files
- do not use evidence outside this workspace copy
"""


def dockerfile() -> str:
    return """FROM python:3.11-slim
WORKDIR /workspace
"""


def oracle_input(variant: str) -> dict:
    actions = VARIANT_CONFIG[variant]["cleanup_actions"]
    note = {
        "v1-clean-baseline": "ENABLE_PREVIEW_V2 is the highest-risk classification because parser compatibility can be misread as a standalone live branch; PREVIEW_FORCE_LEGACY is stale reporting residue, not a live toggle.",
        "v2-noisy-distractor": "The stale 2025 memo makes ENABLE_PREVIEW_V2 look newer than it is. Treat it as alias-only compatibility, not as a second live runtime branch.",
        "v3-dirty-state": "The abandoned audit draft and half-written patch are false positives. They show attempted reasoning and a proposed runtime branch, not active runtime behavior.",
        "v4-multi-corpus-objective": "The release-freeze objective changes cleanup priority, not classification. Alias compatibility should be telemetry-first during the freeze.",
        "v5-recovery-in-thread": "The rollback incident means cleanup must explicitly wait for migration completion. Missing that incident is an operational judgment failure even if the classifications are otherwise close.",
    }[variant]

    cleanup_rows = [
        {
            "flag": "ENABLE_SHADOW_PREVIEW",
            "action": actions["ENABLE_SHADOW_PREVIEW"],
            "rationale": "Keep the live rollout control because the runtime still branches on the shadow preview path.",
            "blockers": [],
        },
        {
            "flag": "ENABLE_PREVIEW_V2",
            "action": actions["ENABLE_PREVIEW_V2"],
            "rationale": {
                "deprecate": "Keep parser compatibility for now, mark it deprecated, and migrate callers to ENABLE_SHADOW_PREVIEW.",
                "telemetry_first": "During the freeze, keep parser compatibility and add telemetry or docs before any removal work.",
                "remove_after_migration": "Do not remove the alias until manifest migration is verified post-incident; removal is only safe after migration evidence exists.",
            }[actions["ENABLE_PREVIEW_V2"]],
            "blockers": {
                "deprecate": ["pending deploy-manifest migration"],
                "telemetry_first": ["release freeze active", "pending deploy-manifest migration"],
                "remove_after_migration": ["rollback incident requires explicit migration proof"],
            }[actions["ENABLE_PREVIEW_V2"]],
        },
        {
            "flag": "PREVIEW_FORCE_LEGACY",
            "action": actions["PREVIEW_FORCE_LEGACY"],
            "rationale": {
                "docs_cleanup": "Remove stale docs and helper references, but do not claim live runtime cleanup because the service no longer branches on this flag.",
                "do_not_remove_now": "Keep the stale references documented until the rollback follow-up is complete; avoid another high-noise cleanup while the incident thread is open.",
            }[actions["PREVIEW_FORCE_LEGACY"]],
            "blockers": [] if actions["PREVIEW_FORCE_LEGACY"] == "docs_cleanup" else ["incident follow-up still open"],
        },
    ]

    evidence_v2 = ["config/defaults.toml", "src/preview/config.py", "src/preview/service.py", "tests/test_preview_v2_alias.py"]
    if variant in {"v2-noisy-distractor", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        evidence_v2.append("docs/preview_v2_ramp_memo_2025.md" if variant == "v2-noisy-distractor" else "release_context/release_freeze_guidance.md")

    flags = [
        {
            "flag": "ENABLE_SHADOW_PREVIEW",
            "status": "live",
            "alias_of": None,
            "parser_symbol": "load_preview_env:ENABLE_SHADOW_PREVIEW",
            "runtime_branch_symbol": "preview_runtime_branch:shadow_preview_path",
            "evidence": [
                "config/defaults.toml",
                "src/preview/config.py",
                "src/preview/runtime.py",
                "src/preview/service.py",
                "tests/test_shadow_preview_live.py",
                "docs/preview_rollout_runbook.md",
            ],
            "disproved_false_positive_path": "tests/test_force_legacy_reporting_only.py",
            "rationale": "ENABLE_SHADOW_PREVIEW is live because the env parser records it and the service reaches the live shadow runtime branch when it is enabled.",
        },
        {
            "flag": "ENABLE_PREVIEW_V2",
            "status": "partial",
            "alias_of": "ENABLE_SHADOW_PREVIEW",
            "parser_symbol": "load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW",
            "runtime_branch_symbol": "preview_runtime_branch:shadow_preview_path",
            "evidence": evidence_v2,
            "disproved_false_positive_path": "docs/preview_v2_ramp_memo_2025.md" if variant == "v2-noisy-distractor" else "src/preview/service.py",
            "rationale": (
                "ENABLE_PREVIEW_V2 is partial: the parser still accepts it, but only as a legacy alias that maps into "
                "ENABLE_SHADOW_PREVIEW rather than a standalone runtime branch."
                if variant != "v2-noisy-distractor"
                else "ENABLE_PREVIEW_V2 is partial: the parser still accepts it as a legacy alias, and the stale 2025 ramp memo is a false-positive doc path rather than proof of a standalone runtime branch."
            ),
        },
        {
            "flag": "PREVIEW_FORCE_LEGACY",
            "status": "dead",
            "alias_of": None,
            "parser_symbol": "load_preview_env:PREVIEW_FORCE_LEGACY",
            "runtime_branch_symbol": None,
            "evidence": [
                "config/defaults.toml",
                "src/preview/config.py",
                "src/preview/runtime.py",
                "tests/test_force_legacy_reporting_only.py",
                "docs/preview_rollout_runbook.md",
            ] + (["repo_evidence/in_progress_patch/preview_force_legacy.patch"] if variant in {"v3-dirty-state", "v5-recovery-in-thread"} else []),
            "disproved_false_positive_path": "repo_evidence/in_progress_patch/preview_force_legacy.patch" if variant in {"v3-dirty-state", "v5-recovery-in-thread"} else "tests/test_force_legacy_reporting_only.py",
            "rationale": "PREVIEW_FORCE_LEGACY is dead in the live runtime: the parser records the flag for reporting, but the service never branches on it and the only extra references are stale helpers or abandoned patches.",
        },
    ]

    if variant == "v3-dirty-state":
        flags[2]["evidence"].append("repo_evidence/previous_session/audit_draft.md")
    if variant == "v5-recovery-in-thread":
        flags[1]["evidence"].append("incident_context/preview_alias_cleanup_rollback.md")
        flags[2]["evidence"].append("incident_context/recovery_thread.md")

    return {
        "schema_version": "cnb55.flag_audit.v1",
        "variant_id": variant,
        "flags": flags,
        "summary": {
            "highest_operational_risk": VARIANT_CONFIG[variant]["highest_operational_risk"],
            "note": note,
        },
        "cleanup_plan": cleanup_rows,
        "assumption_ledger": [
            {
                "status": "observed",
                "topic": "live shadow branch",
                "note": "The runtime branch symbol is explicit in src/preview/runtime.py.",
            },
            {
                "status": "to_verify",
                "topic": "deploy manifest migration completion",
                "note": "Alias removal depends on confirming deploy manifests no longer use ENABLE_PREVIEW_V2.",
            },
            {
                "status": "missing",
                "topic": "operator telemetry on legacy alias usage",
                "note": "The bundle does not include current rollout telemetry proving alias usage is fully gone.",
            },
        ],
    }


def gold_doc(variant: str, readonly_tree_hashes: dict[str, str]) -> dict:
    actions = VARIANT_CONFIG[variant]["cleanup_actions"]
    docs_requirements = []
    if variant == "v2-noisy-distractor":
        docs_requirements.append("docs/preview_v2_ramp_memo_2025.md")
    if variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        docs_requirements.append("release_context/release_freeze_guidance.md")
    if variant == "v5-recovery-in-thread":
        docs_requirements.append("incident_context/preview_alias_cleanup_rollback.md")
    return {
        "variant_id": variant,
        "flags": {
            "ENABLE_SHADOW_PREVIEW": {
                "status": "live",
                "alias_of": None,
                "parser_symbol": "load_preview_env:ENABLE_SHADOW_PREVIEW",
                "runtime_branch_symbol": "preview_runtime_branch:shadow_preview_path",
                "expected_false_positive_path": "tests/test_force_legacy_reporting_only.py",
            },
            "ENABLE_PREVIEW_V2": {
                "status": "partial",
                "alias_of": "ENABLE_SHADOW_PREVIEW",
                "parser_symbol": "load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW",
                "runtime_branch_symbol": "preview_runtime_branch:shadow_preview_path",
                "expected_false_positive_path": "docs/preview_v2_ramp_memo_2025.md" if variant == "v2-noisy-distractor" else "src/preview/service.py",
            },
            "PREVIEW_FORCE_LEGACY": {
                "status": "dead",
                "alias_of": None,
                "parser_symbol": "load_preview_env:PREVIEW_FORCE_LEGACY",
                "runtime_branch_symbol": None,
                "expected_false_positive_path": "repo_evidence/in_progress_patch/preview_force_legacy.patch" if variant in {"v3-dirty-state", "v5-recovery-in-thread"} else "tests/test_force_legacy_reporting_only.py",
            },
        },
        "required_surface_roots_min": 4,
        "cleanup_actions": actions,
        "docs_requirements": docs_requirements,
        "test_flag_audit_contract_sha256": None,
        "readonly_tree_hashes": readonly_tree_hashes,
        "pass_bar": 70,
    }


def hidden_test_template() -> str:
    return """from __future__ import annotations

import json
import os
from pathlib import Path

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")


def load_json(path: Path):
    return json.loads(path.read_text())


def test_expected_statuses():
    result = load_json(RESULT_FILE)
    gold = load_json(VERIFIER_DATA / VARIANT_ID / "gold_audit.json")
    by_flag = {row["flag"]: row["status"] for row in result.get("submitted_flags", [])}
    assert by_flag.get("ENABLE_SHADOW_PREVIEW") == gold["flags"]["ENABLE_SHADOW_PREVIEW"]["status"]
    assert by_flag.get("ENABLE_PREVIEW_V2") == gold["flags"]["ENABLE_PREVIEW_V2"]["status"]
    assert by_flag.get("PREVIEW_FORCE_LEGACY") == gold["flags"]["PREVIEW_FORCE_LEGACY"]["status"]


def test_alias_relationship():
    result = load_json(RESULT_FILE)
    submitted = {row["flag"]: row for row in result.get("submitted_flags", [])}
    entry = submitted["ENABLE_PREVIEW_V2"]
    assert entry["alias_of"] == "ENABLE_SHADOW_PREVIEW"


def test_dead_force_legacy_has_no_runtime_branch():
    result = load_json(RESULT_FILE)
    submitted = {row["flag"]: row for row in result.get("submitted_flags", [])}
    assert submitted["PREVIEW_FORCE_LEGACY"]["runtime_branch_symbol"] in (None, "")
"""


def family_yaml() -> str:
    return """family_id: dead-flag-reachability-audit
track: 2
schema_version: cnb55.family.v1
layer_a_status: in_progress
layer_b_status: in_progress

grader_ref: verifiers/dead-flag-reachability-audit/score_reachability.py
milestone_config_ref: verifier_data/dead-flag-reachability-audit/{variant_id}/milestones/

milestones:
  M1_localization:
    weight: 0.10
    implementation_level: L2
    description: The submitted audit cites enough workspace evidence to prove code-path localization.
  M2_primary_fix:
    weight: 0.20
    implementation_level: L2
    description: "`brief/flag_audit.json` exists, parses, and matches the schema version."
  M3_invariants:
    weight: 0.20
    implementation_level: L2
    description: No integrity violation and no readonly tree mutation.
  M4_functional:
    weight: 0.20
    implementation_level: L2
    description: The matrix captures all three flags, runtime symbols, and cleanup actions.
  M5_e2e:
    weight: 0.30
    implementation_level: L2
    description: All classifications are correct and no hard ceiling <= 35 fired.

capability_tags:
  shared_core:
    required:
      - localize
      - inspect
      - modify
      - verify
      - respect_invariants
    recommended:
      - inspect:evidence_triage
      - inspect:prioritize
      - verify:assumption_honesty
  per_variant:
    v1-clean-baseline: {}
    v2-noisy-distractor:
      variant_notes: stale-doc triage required
    v3-dirty-state:
      variant_notes: sunk-cost rejection required
    v4-multi-corpus-objective:
      variant_notes: release-freeze cleanup reprioritization required
    v5-recovery-in-thread:
      variant_notes: rollback-incident acknowledgement required

tool_call_overrides:
  - command: "cnb55-flag-audit schema"
    tags: [inspect]
  - command: "cnb55-flag-audit validate"
    tags: [verify]
  - command: "cnb55-flag-audit submit"
    tags: [modify]
    terminal: true

state_delta:
  kind: json_deliverable
  deliverable_path: brief/flag_audit.json
  states:
    - absent
    - present_and_invalid
    - present_and_valid
  transitions:
    - {from: absent, to: absent, trigger: "read-only turn", delta: 0.0}
    - {from: absent, to: absent, trigger: "cnb55-flag-audit validate success", delta: 0.3}
    - {from: absent, to: present_and_valid, trigger: "cnb55-flag-audit submit success", delta: 0.7}
    - {from: absent, to: present_and_invalid, trigger: "raw invalid write", delta: 0.0}
    - {from: present_and_invalid, to: present_and_valid, trigger: "subsequent valid submit", delta: 0.4}
    - {from: "*", to: "*", trigger: "integrity_flag=1", delta: -0.5}
  aggregate_clamp: [0.0, 1.0]

integrity_rules:
  - id: write_outside_whitelist
    rule: "any write outside `brief/*`, `artifacts/*`, and `brief_input.json`"
  - id: immutable_slice_mutated
    rule: readonly tree hash changed
  - id: pytest_shim
    rule: shim file present
  - id: tests_modified
    rule: visible test checksum changed
  - id: network_egress
    rule: "`.network_egress_detected` marker present"

llm_judge_quarantine:
  cleanup_precision_bonus:
    max_points: 6
    band: P_benchmark_only
  total_quarantined_points: 6

seeds:
  base_count: 2
  variance_escalation:
    stdev_threshold_to_4: 0.10
    stdev_threshold_to_8: 0.20
    stdev_flag_high_variance: 0.15
  current_observed_stdev_M_training: 0.0
  escalation_currently_active: false

initial_state:
  type: manifest_locked
  ref: benchmark_blueprints/families/dead-flag-reachability-audit/manifest.lock.json

rawr_modes:
  grounding_stripped: implemented
  citation_fabricated: declared_not_yet_implemented
  constraint_named_not_respected: declared_not_yet_implemented

saturation:
  threshold_mean_P: 80
  sustained_rounds: 2
  renewal_queue:
    - add-deploy-manifest-corpus
    - add-parser-refactor-diff
    - retire-v1
"""


def manifest_lock_template() -> dict:
    return {
        "schema_version": "cnb55.manifest.v2",
        "family_id": FAMILY_ID,
        "grader": {
            "score_reachability_py_sha256": "",
        },
        "variants": {variant: {} for variant in VARIANTS},
    }


def make_workspace(variant: str) -> None:
    ws = WORKSPACE_BUNDLE / variant
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "brief").mkdir(parents=True, exist_ok=True)
    (ws / "artifacts").mkdir(parents=True, exist_ok=True)
    write(ws / ".scenario_variant", variant + "\n")
    write(ws / "AGENTS.md", agents_md(variant))
    write(ws / "Dockerfile", dockerfile())
    write(ws / "artifacts/README.md", "# Solver outputs land here.\n")
    write(ws / "brief/README.md", "# Canonical structured output lands here.\n")
    write(ws / "bin/cnb55-flag-audit", CLI_TEMPLATE, executable=True)
    write(ws / "config/defaults.toml", CONFIG_DEFAULTS)
    write(ws / "src/preview/__init__.py", "")
    write(ws / "src/preview/config.py", SRC_CONFIG)
    write(ws / "src/preview/runtime.py", SRC_RUNTIME)
    write(ws / "src/preview/service.py", SRC_SERVICE)
    write(ws / "src/preview/legacy.py", SRC_LEGACY)
    write(ws / "src/preview_cli.py", SRC_CLI)
    write(ws / "tests/test_shadow_preview_live.py", TEST_LIVE)
    write(ws / "tests/test_preview_v2_alias.py", TEST_ALIAS)
    write(ws / "tests/test_force_legacy_reporting_only.py", TEST_FORCE_LEGACY)
    write(ws / "docs/preview_rollout_runbook.md", DOC_RUNBOOK)
    write(ws / "docs/false_positive_notes.md", DOC_FALSE_POSITIVE)

    cfg = VARIANT_CONFIG[variant]
    for rel, content in cfg["extra_docs"].items():
        write(ws / rel, content)
    for rel, content in cfg["release_context"].items():
        write(ws / rel, content)
    for rel, content in cfg["incident_context"].items():
        write(ws / rel, content)
    for rel, content in cfg["repo_evidence"].items():
        write(ws / rel, content)


def copy_milestones(variant: str) -> None:
    shared = VERIFIER_DATA / "_milestones_shared"
    variant_dir = VERIFIER_DATA / variant / "milestones"
    variant_dir.mkdir(parents=True, exist_ok=True)
    for name in ("m1_localize.sh", "m2_primary_fix.sh", "m3_invariants.sh", "m4_functional.sh", "m5_e2e.sh"):
        target = variant_dir / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(shared / name)


def write_shared_milestones() -> None:
    shared = VERIFIER_DATA / "_milestones_shared"
    shared.mkdir(parents=True, exist_ok=True)
    scripts = {
        "m1_localize.sh": "M1_localization",
        "m2_primary_fix.sh": "M2_primary_fix",
        "m3_invariants.sh": "M3_invariants",
        "m4_functional.sh": "M4_functional",
        "m5_e2e.sh": "M5_e2e",
    }
    for filename, key in scripts.items():
        write(shared / filename, f"""#!/usr/bin/env bash
set -euo pipefail
RESULT_FILE="${{RESULT_FILE:-/results/verify_result.json}}"
python3 - "$RESULT_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if bool(d.get("milestones", {{}}).get("{key}", False)) else 1)
PY
""", executable=True)
    write(shared / "README.md", "# Shared milestone scripts for dead-flag-reachability-audit.\n")


def score_workspace(ws: Path, variant: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="dead-flag-reachability-audit-score-") as tmp:
        result_file = Path(tmp) / "verify_result.json"
        proc_env = os.environ.copy()
        proc_env.update({
            "AGENT_WS": str(ws),
            "VERIFIER_DATA": str(VERIFIER_DATA),
            "VARIANT_ID": variant,
            "RESULT_FILE": str(result_file),
            "CNB55_SEED": "42",
        })
        proc_env.update({"PYTHONPATH": str(ws / "src")})
        subprocess.run([sys.executable, str(SCORER)], env=proc_env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.loads(result_file.read_text())


def oracle_triplet(variant: str, oracle_doc: dict) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"{FAMILY_ID}-{variant}-") as tmp:
        tmp_root = Path(tmp)

        oracle_ws = tmp_root / "oracle"
        shutil.copytree(WORKSPACE_BUNDLE / variant, oracle_ws)
        input_path = oracle_ws / "brief_input.json"
        input_path.write_text(json.dumps(oracle_doc, indent=2, sort_keys=True) + "\n")
        subprocess.run([sys.executable, str(oracle_ws / "bin/cnb55-flag-audit"), "submit", str(input_path)], cwd=oracle_ws, check=True)
        oracle_result = score_workspace(oracle_ws, variant)

        empty_ws = tmp_root / "empty"
        shutil.copytree(WORKSPACE_BUNDLE / variant, empty_ws)
        empty_result = score_workspace(empty_ws, variant)

        shortcut_ws = tmp_root / "shortcut"
        shutil.copytree(WORKSPACE_BUNDLE / variant, shortcut_ws)
        shortcut_doc = json.loads(json.dumps(oracle_doc))
        for row in shortcut_doc["flags"]:
            if row["flag"] == "ENABLE_PREVIEW_V2":
                row["status"] = "live"
                row["alias_of"] = None
                row["rationale"] = "ENABLE_PREVIEW_V2 is live because it is parsed and mentioned in docs and tests."
                row["disproved_false_positive_path"] = "docs/preview_rollout_runbook.md"
            if row["flag"] == "PREVIEW_FORCE_LEGACY":
                row["status"] = "partial"
                row["runtime_branch_symbol"] = "preview_runtime_branch:legacy_force_override"
                row["rationale"] = "PREVIEW_FORCE_LEGACY is partially wired because the parser reads it and an unfinished patch shows the intended runtime branch."
        for row in shortcut_doc["cleanup_plan"]:
            if row["flag"] == "ENABLE_PREVIEW_V2":
                row["action"] = "remove_after_migration"
                row["rationale"] = "Remove the alias now; parser presence proves the code path already covers it."
                row["blockers"] = []
        shortcut_input = shortcut_ws / "brief_input.json"
        shortcut_input.write_text(json.dumps(shortcut_doc, indent=2, sort_keys=True) + "\n")
        subprocess.run([sys.executable, str(shortcut_ws / "bin/cnb55-flag-audit"), "submit", str(shortcut_input)], cwd=shortcut_ws, check=True)
        shortcut_result = score_workspace(shortcut_ws, variant)

        oracle_dir = VERIFIER_DATA / variant / "oracle"
        oracle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(oracle_ws / "brief_input.json", oracle_dir / "brief_input.json")
        shutil.copy(oracle_ws / "brief" / "flag_audit.json", oracle_dir / "flag_audit.json")
        shutil.copy(oracle_ws / "artifacts" / "flag_audit.md", oracle_dir / "flag_audit.md")
        shutil.copy(oracle_ws / "artifacts" / "reachability_matrix.json", oracle_dir / "reachability_matrix.json")
        shutil.copy(oracle_ws / "artifacts" / "cleanup.patchplan.md", oracle_dir / "cleanup.patchplan.md")
        return int(oracle_result["P_benchmark"]), int(empty_result["P_benchmark"]), int(shortcut_result["P_benchmark"])


def refresh_manifest_lock(observed: dict[str, dict[str, int]]) -> None:
    lock = manifest_lock_template()
    lock["grader"]["score_reachability_py_sha256"] = sha256_file(SCORER)
    for variant in VARIANTS:
        ws = WORKSPACE_BUNDLE / variant
        vd = VERIFIER_DATA / variant
        trees = {}
        for rel in READONLY_TREES:
            digest = sha256_tree(ws, rel)
            if digest:
                trees[rel] = digest
        lock["variants"][variant] = {
            "observed_oracle_score": observed[variant]["oracle"],
            "observed_empty_brief_score": observed[variant]["empty"],
            "observed_shortcut_score": observed[variant]["shortcut"],
            "workspace_trees": trees,
            "verifier_data": {
                "gold_audit_sha256": sha256_file(vd / "gold_audit.json"),
                "workspace_manifest_sha256": sha256_file(vd / "workspace_manifest.json"),
                "oracle_flag_audit_sha256": sha256_file(vd / "oracle" / "flag_audit.json"),
                "hidden_tests_tree_sha256": sha256_tree(vd, "hidden_tests"),
            },
        }
    write(FAMILY / "manifest.lock.json", json.dumps(lock, indent=2, sort_keys=True) + "\n")


def main() -> int:
    VERIFIER_DATA.mkdir(parents=True, exist_ok=True)
    write(FAMILY / "family.yaml", family_yaml())
    write_shared_milestones()

    observed: dict[str, dict[str, int]] = {}
    lines = ["variant | oracle | empty | shortcut", "--- | ---: | ---: | ---:"]

    for variant in VARIANTS:
        make_workspace(variant)
        ws = WORKSPACE_BUNDLE / variant
        readonly_tree_hashes = {}
        for rel in READONLY_TREES:
            digest = sha256_tree(ws, rel)
            if digest:
                readonly_tree_hashes[rel] = digest
        files = list_files(ws)
        gold = gold_doc(variant, readonly_tree_hashes)
        gold["test_flag_audit_contract_sha256"] = sha256_file(ws / "tests" / "test_shadow_preview_live.py")
        vd = VERIFIER_DATA / variant
        vd.mkdir(parents=True, exist_ok=True)
        write(vd / "gold_audit.json", json.dumps(gold, indent=2, sort_keys=True) + "\n")
        write(vd / "workspace_manifest.json", json.dumps({
            "variant_id": variant,
            "files": files,
            "readonly_tree_hashes": readonly_tree_hashes,
        }, indent=2, sort_keys=True) + "\n")
        write(vd / "hidden_tests/test_audit_properties.py", hidden_test_template())
        copy_milestones(variant)

        oracle_doc = oracle_input(variant)
        oracle_score, empty_score, shortcut_score = oracle_triplet(variant, oracle_doc)
        observed[variant] = {
            "oracle": oracle_score,
            "empty": empty_score,
            "shortcut": shortcut_score,
        }
        lines.append(f"{variant} | {oracle_score} | {empty_score} | {shortcut_score}")

    refresh_manifest_lock(observed)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
