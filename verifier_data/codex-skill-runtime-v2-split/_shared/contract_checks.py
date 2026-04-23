from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                data[key] = []
            else:
                data[key] = [item.strip().strip('"') for item in inner.split(",")]
        elif value in {"true", "false"}:
            data[key] = value == "true"
        else:
            data[key] = value.strip('"')
    return data


def read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def run_checks(workspace: Path, gold: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    primary_skill = workspace / "skills/oncall_handoff/SKILL.md"
    shared_contract = workspace / gold["shared_contract_path"]
    config = parse_toml(workspace / ".codex/config.toml")
    primary_automation = parse_toml(workspace / "automations/handoff-primary.toml")
    copy_automation = parse_toml(workspace / "automations/handoff-copy.toml")
    runbook = read_text(workspace / "docs/runbooks/oncall.md")
    primary_skill_text = read_text(primary_skill)
    shared_text = read_text(shared_contract)
    live_surface_paths = [
        ".codex/config.toml",
        "automations/handoff-primary.toml",
        "automations/handoff-copy.toml",
        "docs/runbooks/oncall.md",
        "skills/oncall_handoff/SKILL.md",
    ]
    if gold.get("require_reuse_skill"):
        live_surface_paths.append("skills/escalation_handoff/SKILL.md")
    live_text = "\n".join(read_text(workspace / rel) for rel in live_surface_paths if (workspace / rel).exists())

    checks["hidden.skill_bundle_exists"] = primary_skill.exists()
    checks["hidden.shared_contract"] = (
        shared_contract.exists()
        and all(marker in shared_text for marker in gold["shared_contract_required"])
        and gold["shared_contract_path"] in primary_skill_text
    )
    checks["hidden.config_paths"] = config.get("skill_paths") == gold["config_skill_paths"]
    checks["hidden.runbook_alignment"] = all(token in runbook for token in gold["runbook_required"]) and all(
        token not in runbook for token in gold["runbook_forbidden"]
    )
    checks["hidden.primary_automation"] = (
        primary_automation.get("canonical") is True
        and primary_automation.get("enabled") is True
        and primary_automation.get("skill") == "skills/oncall_handoff/SKILL.md"
        and gold["command_substring"] in str(primary_automation.get("command", ""))
    )
    checks["hidden.retired_automation"] = (
        copy_automation.get("canonical") is False
        and copy_automation.get("enabled") is False
        and copy_automation.get("status") == "retired"
        and copy_automation.get("replaced_by") == "handoff-primary.toml"
    )
    checks["hidden.legacy_refs_removed"] = all(token not in live_text for token in gold["forbidden_live_markers"])

    for rel_path, digest in gold.get("dirty_sentinel_hashes", {}).items():
        checks["hidden.dirty_sentinel_untouched"] = sha256_file(workspace / rel_path) == digest and checks.get(
            "hidden.dirty_sentinel_untouched", True
        )

    if gold.get("require_reuse_skill"):
        reuse_path = workspace / "skills/escalation_handoff/SKILL.md"
        reuse_text = read_text(reuse_path)
        checks["hidden.release_reuse_extension"] = (
            reuse_path.exists()
            and gold["shared_contract_path"] in reuse_text
            and all(token in reuse_text for token in gold["reuse_skill_required"])
            and all(token not in reuse_text for token in gold["reuse_skill_forbidden"])
        )

    if gold.get("require_incident_note"):
        checks["hidden.incident_note"] = all(token in runbook.lower() for token in gold["incident_keywords"])

    return checks
