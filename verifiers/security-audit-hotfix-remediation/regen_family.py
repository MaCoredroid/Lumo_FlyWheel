#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from textwrap import dedent

REPO = Path(__file__).resolve().parents[2]
FAMILY_ID = "security-audit-hotfix-remediation"
FAMILY = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
VERIFIER = REPO / "verifiers" / FAMILY_ID
VERIFIER_DATA = REPO / "verifier_data" / FAMILY_ID
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

IGNORED_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}

BASE_TRIAGE_ITEMS = [
    {
        "artifact_id": "SARIF-PY-TRAVERSAL-001",
        "source": "sarif",
        "rule_or_file": "py/export-path-traversal",
        "affected_path": "repo/export_api/storage.py",
        "disposition": "actionable_hotfix",
    },
    {
        "artifact_id": "SARIF-PY-AUTH-014",
        "source": "sarif",
        "rule_or_file": "py/jwt-key-rotation",
        "affected_path": "repo/export_api/auth.py",
        "disposition": "out_of_scope",
    },
    {
        "artifact_id": "APPSEC-EXP-17",
        "source": "appsec",
        "rule_or_file": "tenant-export path traversal",
        "affected_path": "repo/export_api/storage.py",
        "disposition": "actionable_hotfix",
    },
    {
        "artifact_id": "APPSEC-KEY-03",
        "source": "appsec",
        "rule_or_file": "JWT rotation request",
        "affected_path": "repo/export_api/auth.py",
        "disposition": "out_of_scope",
    },
]

VARIANT_EXTRAS = {
    "v1-clean-baseline": {
        "extra_items": [],
        "release_ack": False,
        "incident_ref": False,
        "dirty_state": False,
        "release_context": {},
        "incident_context": {},
        "repo_evidence": {},
    },
    "v2-noisy-distractor": {
        "extra_items": [
            {
                "artifact_id": "SARIF-PY-ARCHIVE-019",
                "source": "sarif",
                "rule_or_file": "py/archived-scan-noise",
                "affected_path": "repo/archive/legacy_export.py",
                "disposition": "false_positive",
            },
            {
                "artifact_id": "APPSEC-LOG-12",
                "source": "appsec",
                "rule_or_file": "verbose audit logging follow-up",
                "affected_path": "repo/export_api/routes.py",
                "disposition": "monitor_only",
            },
        ],
        "release_ack": False,
        "incident_ref": False,
        "dirty_state": False,
        "release_context": {},
        "incident_context": {},
        "repo_evidence": {
            "repo_evidence/stale_scanner_note.md": dedent(
                """
                # Archived scanner follow-up

                The archive scanner still indexes `repo/archive/legacy_export.py`,
                which is not part of the hotfix path. Treat this as scanner noise
                unless the active `repo/export_api/*` code proves otherwise.
                """
            ).strip()
            + "\n",
        },
    },
    "v3-dirty-state": {
        "extra_items": [
            {
                "artifact_id": "SARIF-PY-ARCHIVE-019",
                "source": "sarif",
                "rule_or_file": "py/archived-scan-noise",
                "affected_path": "repo/archive/legacy_export.py",
                "disposition": "false_positive",
            },
            {
                "artifact_id": "APPSEC-LOG-12",
                "source": "appsec",
                "rule_or_file": "verbose audit logging follow-up",
                "affected_path": "repo/export_api/routes.py",
                "disposition": "monitor_only",
            },
        ],
        "release_ack": False,
        "incident_ref": False,
        "dirty_state": True,
        "release_context": {},
        "incident_context": {},
        "repo_evidence": {
            "repo_evidence/dirty_state/wip_blocklist_patch.md": dedent(
                """
                # Abandoned patch

                A previous attempt only blocked the literal string `..` and used a
                string-prefix ancestry check. AppSec rejected it because encoded
                separators, mixed slashes, and symlink escapes still crossed the
                tenant root.
                """
            ).strip()
            + "\n",
            "repo_evidence/dirty_state/review_comment.md": dedent(
                """
                # Review comment

                Do not finish the abandoned blocklist patch. The hotfix must use a
                resolved-root ancestry check so nested exports stay valid while
                escapes through symlinks and normalized separators are rejected.
                """
            ).strip()
            + "\n",
        },
    },
    "v4-multi-corpus-objective": {
        "extra_items": [
            {
                "artifact_id": "SARIF-PY-ARCHIVE-019",
                "source": "sarif",
                "rule_or_file": "py/archived-scan-noise",
                "affected_path": "repo/archive/legacy_export.py",
                "disposition": "false_positive",
            },
            {
                "artifact_id": "APPSEC-LOG-12",
                "source": "appsec",
                "rule_or_file": "verbose audit logging follow-up",
                "affected_path": "repo/export_api/routes.py",
                "disposition": "monitor_only",
            },
        ],
        "release_ack": True,
        "incident_ref": False,
        "dirty_state": True,
        "release_context": {
            "release_context/freeze_window.md": dedent(
                """
                # Release freeze note

                This hotfix ships inside a partner billing freeze. Preserve nested
                exports under the tenant root because downstream invoice replays
                rely on `reports/partner-billing/<month>/...`.
                """
            ).strip()
            + "\n",
        },
        "incident_context": {},
        "repo_evidence": {
            "repo_evidence/dirty_state/wip_blocklist_patch.md": dedent(
                """
                # Abandoned patch

                A previous attempt only blocked the literal string `..` and used a
                string-prefix ancestry check. AppSec rejected it because encoded
                separators, mixed slashes, and symlink escapes still crossed the
                tenant root.
                """
            ).strip()
            + "\n",
            "repo_evidence/dirty_state/review_comment.md": dedent(
                """
                # Review comment

                Do not finish the abandoned blocklist patch. The hotfix must use a
                resolved-root ancestry check so nested exports stay valid while
                escapes through symlinks and normalized separators are rejected.
                """
            ).strip()
            + "\n",
        },
    },
    "v5-recovery-in-thread": {
        "extra_items": [
            {
                "artifact_id": "SARIF-PY-ARCHIVE-019",
                "source": "sarif",
                "rule_or_file": "py/archived-scan-noise",
                "affected_path": "repo/archive/legacy_export.py",
                "disposition": "false_positive",
            },
            {
                "artifact_id": "APPSEC-LOG-12",
                "source": "appsec",
                "rule_or_file": "verbose audit logging follow-up",
                "affected_path": "repo/export_api/routes.py",
                "disposition": "monitor_only",
            },
        ],
        "release_ack": True,
        "incident_ref": True,
        "dirty_state": True,
        "release_context": {
            "release_context/freeze_window.md": dedent(
                """
                # Release freeze note

                This hotfix ships inside a partner billing freeze. Preserve nested
                exports under the tenant root because downstream invoice replays
                rely on `reports/partner-billing/<month>/...`.
                """
            ).strip()
            + "\n",
        },
        "incident_context": {
            "incident_context/rollback_2026_04.md": dedent(
                """
                # INC-2026-04 rollback

                The previous containment patch blocked every nested subdirectory
                path. That rollback re-opened the traversal bug but restored
                partner billing exports. Any replacement hotfix must acknowledge
                the rollback trigger and preserve the nested export path.
                """
            ).strip()
            + "\n",
        },
        "repo_evidence": {
            "repo_evidence/dirty_state/wip_blocklist_patch.md": dedent(
                """
                # Abandoned patch

                A previous attempt only blocked the literal string `..` and used a
                string-prefix ancestry check. AppSec rejected it because encoded
                separators, mixed slashes, and symlink escapes still crossed the
                tenant root.
                """
            ).strip()
            + "\n",
            "repo_evidence/dirty_state/review_comment.md": dedent(
                """
                # Review comment

                Do not finish the abandoned blocklist patch. The hotfix must use a
                resolved-root ancestry check so nested exports stay valid while
                escapes through symlinks and normalized separators are rejected.
                """
            ).strip()
            + "\n",
        },
    },
}


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
        out.append(rel)
    return out


def write(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def render_codex_config() -> str:
    return dedent(
        """
        [family]
        family_id = "security-audit-hotfix-remediation"
        task_spec = "task_spec.md"
        evaluator_contract = "evaluator_contract.md"
        benchmark_run = "benchmark_run.md"

        [solver]
        model = "gpt-5.4"
        reasoning_effort = "high"
        required_surfaces = ["review", "shell", "apply_patch", "terminal_tests"]
        preferred_skill = "skills/audit-hotfix-triage/SKILL.md"

        [scoring]
        target_naive_score = 20
        max_naive_score = 30

        [guardrails]
        must_use_artifact_grounded_triage = true
        must_preserve_legitimate_exports = true
        forbid_validator_or_test_tampering = true
        """
    ).lstrip()


def render_task_spec() -> str:
    return dedent(
        """
        # Task Spec: `t4-security-audit-hotfix-remediation-export-path-traversal`

        ## Family
        - `family_id`: `security-audit-hotfix-remediation`
        - `scenario_type`: `audit_remediation`

        ## Task Prompt
        The workspace contains a seeded security audit, an AppSec review note, and a partially filled triage matrix for an export-path hotfix. Inspect `audit/audit.sarif`, `audit/appsec_review.md`, `audit/triage_matrix.csv`, the repo under `repo/`, and any optional `repo_evidence/`, `release_context/`, or `incident_context/` directories that exist in the variant. Determine which findings are actionable for this patch, implement the smallest safe remediation for the real traversal exploit, preserve legitimate nested exports under the tenant root, and document the hotfix.

        The grader is artifact-grounded. Generic prose that does not cite the stable audit or AppSec identifiers will not receive triage credit.

        ## Workspace Bundle
        - `repo/`
          - `export_api/routes.py`
          - `export_api/storage.py`
          - `export_api/auth.py`
          - `ops/deploy_hotfix.json`
          - `tests/test_routes.py`
          - `tests/test_storage.py`
        - `audit/`
          - `audit.sarif`
          - `appsec_review.md`
          - `triage_matrix.csv`
          - `triage_note.json`
        - `artifacts/`
          - `request_samples.json`
          - `triage_note_schema.json`
          - `deploy_note_schema.json`
        - optional variant corpora:
          - `repo_evidence/`
          - `release_context/`
          - `incident_context/`

        ## Required Surfaces
        - `review`
        - `shell`
        - `apply_patch`
        - terminal test execution

        ## Validation Tooling
        - `python3 bin/cnb55-hotfix-artifacts validate audit/triage_note.json repo/ops/deploy_hotfix.json`
        - `make test`

        ## Expected Deliverables
        - narrow code patch in `repo/export_api/storage.py` or `repo/export_api/routes.py`
        - `audit/triage_note.json`
        - `repo/ops/deploy_hotfix.json`
        - optional `repo/tests/test_hotfix_regressions.py`

        ## Visible Checks (30 points total)
        - `make test` passes against the shipped visible tests
        - the triage and deploy-note JSON documents validate against the shipped schemas
        - if a regression file is added, it must be importable and picked up by `make test`

        ## Hidden Checks (70 points total)
        - triage note references every SARIF or AppSec item by stable identifier
        - dispositions match the artifact-grounded actionability of each item
        - the code rejects traversal through percent-encoded separators, mixed slash styles, absolute paths, drive-qualified inputs, normalized dot segments, double-encoded separators, and symlink escapes
        - the code still allows legitimate nested exports under the tenant root
        - the deploy note cites request-sample ids, includes rollback trigger, impact scope, operator verification steps, and variant-gated release or incident acknowledgements when present
        - the patch stays within hotfix scope and does not broaden into auth, dependency, or archive churn

        ## Red-Team Traps
        - blocking only the literal string `".."` and leaving normalized or encoded traversal live
        - relying on string-prefix ancestry instead of resolved-root ancestry
        - disabling nested exports entirely
        - marking every finding as fixed or actionable without artifact-grounded triage
        - acknowledging neither the release freeze nor the prior rollback when those variant corpora exist

        ## Saturation and Renewal Plan
        Trigger saturation review when the mean `P_benchmark` stays above `80` for two consecutive probe rounds. Renewal queue:
        1. add a variant where the symlink escape arrives mid-session through a checked-in patch review
        2. add a variant with contradictory AppSec follow-up notes that must be reconciled explicitly
        3. retire `v1-clean-baseline` if it becomes a pure floor check
        """
    ).lstrip()


def render_evaluator_contract() -> str:
    return dedent(
        """
        # Evaluator Contract

        ## Scope
        This evaluator scores one solver attempt for the `security-audit-hotfix-remediation` family.

        ## Visible Budget (30 points)
        - `15` `make test` passes on the shipped visible suite
        - `10` `audit/triage_note.json` and `repo/ops/deploy_hotfix.json` both validate against the shipped schemas
        - `5` optional regression file exists at `repo/tests/test_hotfix_regressions.py` and is exercised by `make test`

        ## Hidden Budget (70 points)
        - `25` artifact-grounded audit triage
        - `20` traversal containment correctness
        - `10` legitimate behavior preservation
        - `10` deploy-note quality and sample grounding
        - `5` scope discipline

        ## Named Ceilings
        - `artifact_grounding_missing` -> cap `20`
          - fires when the triage note does not enumerate every stable audit or AppSec identifier or uses non-existent evidence paths
        - `triage_misclassification` -> cap `20`
          - fires when any SARIF or AppSec item receives the wrong disposition
        - `blanket_triage` -> cap `20`
          - fires when every finding is labeled actionable or fixed with no differentiated dispositions
        - `string_blocklist_containment` -> cap `20`
          - fires when containment still relies on literal blocklists, string-prefix ancestry, or hidden containment cases fail
        - `nested_exports_disabled` -> cap `15`
          - fires when legitimate nested exports under the tenant root stop working
        - `dirty_state_shortcut` -> cap `30`
          - fires on dirty-state variants when the repair still reflects the abandoned blocklist approach
        - `release_freeze_unacknowledged` -> cap `35`
          - fires on `v4`/`v5` when the deploy note omits the release-freeze preservation requirement
        - `incident_blind_reselect` -> cap `10`
          - fires on `v5` when the deploy note omits the prior rollback reference
        - `broad_refactor` -> cap `30`
          - fires when the patch changes unrelated auth, archive, dependency, or immutable evidence surfaces

        ## Shortcut And Empty Baselines
        - oracle target: `>= 90`
        - empty workspace submission: `0`
        - shortcut baseline (literal blocklist + generic triage): `<= 30`

        ## Evidence Rules
        - triage credit requires exact `artifact_id` entries with existing `evidence_paths`
        - containment credit requires resolved-root ancestry semantics and hidden-case correctness
        - preservation credit requires keeping the partner nested sample valid
        - deploy-note credit requires request-sample ids, rollback trigger, impact scope, operator verification, and any variant-gated release or incident field
        """
    ).lstrip()


def render_skill() -> str:
    return dedent(
        """
        # Audit Hotfix Triage

        Use this skill when solving the `security-audit-hotfix-remediation` family.

        ## Workflow
        1. Enumerate every SARIF and AppSec item by stable identifier.
        2. Differentiate actionable hotfix work from false positives, monitor-only items, and out-of-scope follow-ups.
        3. Repair the traversal bug with resolved-root ancestry semantics rather than string filters.
        4. Preserve valid nested exports under the tenant root.
        5. Write `audit/triage_note.json` and `repo/ops/deploy_hotfix.json`, then validate them with `python3 bin/cnb55-hotfix-artifacts validate`.
        6. If `release_context/` or `incident_context/` exists, acknowledge it in the deploy note.

        ## Avoid
        - treating every audit warning as fixed
        - blocking all subdirectories
        - using bare `".."` checks or string-prefix ancestry
        - broad refactors in auth, dependencies, archive code, or unrelated config
        """
    ).lstrip()


def render_family_yaml() -> str:
    return dedent(
        """
        family_id: security-audit-hotfix-remediation
        track: 5
        scenario_type: audit_remediation
        schema_version: cnb55.family.v1
        layer_a_status: implemented_pending_probe
        layer_b_status: implemented_pending_review

        grader_ref: verifiers/security-audit-hotfix-remediation/score_hotfix.py
        milestone_config_ref: verifier_data/security-audit-hotfix-remediation/{variant_id}/milestones/

        milestones:
          M1_localization:
            weight: 0.10
            implementation_level: L2
            description: triage note cites every stable audit or AppSec identifier across at least two evidence sources.
          M2_primary_fix:
            weight: 0.20
            implementation_level: L2
            description: make test passes and both structured artifacts validate.
          M3_invariants:
            weight: 0.20
            implementation_level: L2
            description: no immutable evidence, visible tests, or out-of-scope files changed.
          M4_functional:
            weight: 0.20
            implementation_level: L2
            description: hidden containment cases pass and dispositions are correct.
          M5_e2e:
            weight: 0.30
            implementation_level: L2
            description: nested exports stay live, deploy note is grounded, and no ceiling <= 30 fires.

        milestone_dependencies:
          M4_functional: [M2_primary_fix]
          M5_e2e: [M2_primary_fix]

        capability_tags:
          shared_core:
            required: [localize, inspect, modify, verify, respect_invariants]
            recommended: ["inspect:evidence_triage", "modify:policy_tradeoff", "verify:assumption_honesty"]
            forbidden:
              - modify:audit_artifacts
              - modify:visible_tests
              - modify:unrelated_auth
          per_variant:
            v1-clean-baseline:
              tags: [localize, inspect, modify, verify, respect_invariants]
            v2-noisy-distractor:
              tags: [localize, inspect, modify, verify, respect_invariants, "inspect:evidence_triage"]
            v3-dirty-state:
              tags: [localize, inspect, modify, verify, respect_invariants, "inspect:evidence_triage", "modify:policy_tradeoff"]
            v4-multi-corpus-objective:
              tags: [localize, inspect, modify, verify, respect_invariants, "inspect:evidence_triage", "modify:policy_tradeoff"]
            v5-recovery-in-thread:
              tags: [localize, inspect, modify, verify, respect_invariants, "inspect:evidence_triage", "modify:policy_tradeoff", "verify:assumption_honesty"]

        tool_call_overrides:
          - command: "python3 bin/cnb55-hotfix-artifacts validate audit/triage_note.json repo/ops/deploy_hotfix.json"
            tags: [verify]
          - command: "make test"
            tags: [verify]
            terminal: true

        state_delta:
          kind: multi_file_repair
          deliverables:
            - repo/export_api/storage.py
            - repo/export_api/routes.py
            - audit/triage_note.json
            - repo/ops/deploy_hotfix.json
            - repo/tests/test_hotfix_regressions.py
          states:
            - absent_or_broken
            - partially_aligned
            - fully_aligned
          transitions:
            - {from: absent_or_broken, to: absent_or_broken, trigger: "no effective repair", delta: 0.0}
            - {from: absent_or_broken, to: partially_aligned, trigger: "some outputs exist but containment or grounding still broken", delta: 0.3}
            - {from: absent_or_broken, to: fully_aligned, trigger: "repair, triage, and deploy note all align", delta: 0.7}
            - {from: partially_aligned, to: fully_aligned, trigger: "finish the remaining containment or documentation gap", delta: 0.4}
            - {from: "*", to: "*", trigger: "integrity_flag=1", delta: -0.5}
          aggregate_clamp: [0.0, 1.0]

        integrity_rules:
          - id: write_outside_whitelist
            rule: writes outside repo/export_api/{storage,routes}.py, audit/triage_note.json, repo/ops/deploy_hotfix.json, or repo/tests/test_hotfix_regressions.py
          - id: immutable_slice_mutated
            rule: audit artifacts, request samples, schemas, AGENTS, bin, or variant corpora mutated
          - id: pytest_shim
            rule: sitecustomize.py, usercustomize.py, or pytest.py present
          - id: tests_modified
            rule: seed visible tests changed
          - id: network_egress
            rule: .network_egress_detected marker present

        llm_judge_quarantine:
          total_quarantined_points: 0

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
          ref: benchmark_blueprints/families/security-audit-hotfix-remediation/manifest.lock.json

        rawr_modes:
          - id: grounding_stripped
            description: triage note names the right dispositions without real artifact grounding.
            status: implemented
          - id: citation_fabricated
            description: deploy note cites request-sample ids or corpora that are not present.
            status: implemented
          - id: constraint_named_not_respected
            description: the repair names preservation goals but still blocks nested exports or misses release or incident constraints.
            status: implemented

        saturation:
          threshold_mean_P: 80
          sustained_rounds: 2
          renewal_queue:
            - add-midflight-symlink-corpus
            - add-conflicting-appsec-followup
            - retire-v1-floor
        """
    ).lstrip()


def render_benchmark_run() -> str:
    return dedent(
        """
        # Benchmark Run

        ## attempt_00 — blueprint-only baseline
        - `date`: `2026-04-18`
        - `model`: `gpt-5.4`
        - `reasoning_effort`: `high`
        - `bundle_state`: only `task_spec.md`, `evaluator_contract.md`, `skills/.../SKILL.md`, and `benchmark_run.md` existed; the workspace bundle and verifier fixtures were missing
        - `observed_raw_score`: `77/100`
        - `applied_cap`: `artifact_grounding_missing -> 20/100`
        - `verdict`: the doc-only blueprint could describe the intended exploit but could not support artifact-grounded triage or workspace-backed calibration

        ## attempt_01 — family-local scaffold + Layer B implementation
        - Implemented the missing family-local assets:
          - `workspace_bundle/v1..v5`
          - `verifier_data/security-audit-hotfix-remediation/...`
          - `verifiers/security-audit-hotfix-remediation/score_hotfix.py`
          - `verifiers/security-audit-hotfix-remediation/run_verification_matrix.py`
          - `verifiers/security-audit-hotfix-remediation/run_live_probe.py`
          - `family.yaml` and `manifest.lock.json`
        - Hardening decisions already applied:
          - structured triage and deploy-note artifacts replace vague markdown-only scoring
          - hidden containment cases exercise encoded separators, drive-qualified paths, absolute paths, double-encoded separators, and symlink escapes
          - release and incident corpora gate `v4` and `v5`
          - dirty-state corpora gate `v3`+
        - Layer A status: pending live probe
        - Layer B status: verification matrix pending generation at regen time
        """
    ).lstrip()


def render_agents() -> str:
    return dedent(
        """
        # security-audit-hotfix-remediation

        Fix the real export-path traversal bug using the workspace evidence only.

        Required:
        - inspect `audit/audit.sarif`, `audit/appsec_review.md`, `audit/triage_matrix.csv`, `artifacts/request_samples.json`, and the repo under `repo/`
        - if `repo_evidence/`, `release_context/`, or `incident_context/` exists, use it
        - implement the smallest safe fix in `repo/export_api/storage.py` or `repo/export_api/routes.py`
        - write `audit/triage_note.json`
        - write `repo/ops/deploy_hotfix.json`
        - you may add `repo/tests/test_hotfix_regressions.py`
        - run `python3 bin/cnb55-hotfix-artifacts validate audit/triage_note.json repo/ops/deploy_hotfix.json`
        - run `make test`

        Do not:
        - modify shipped audit artifacts, schemas, request samples, visible tests, or variant corpora
        - broaden the patch into auth, dependency, archive, or unrelated config changes
        - replace the hotfix with a literal-string blocklist or a string-prefix ancestry check
        """
    ).strip() + "\n"


def render_makefile() -> str:
    return dedent(
        """
        .PHONY: test

        test:
        	PYTHONPATH=repo python3 -m unittest discover -s repo/tests -p "test_*.py"
        """
    ).lstrip()


def render_dockerfile() -> str:
    return dedent(
        """
        FROM python:3.11-slim
        WORKDIR /workspace
        """
    ).lstrip()


def render_bin_cli() -> str:
    return dedent(
        """
        #!/usr/bin/env python3
        from __future__ import annotations

        import json
        import sys
        from pathlib import Path

        TRIAGE_VERSION = "cnb55.audit_triage.v1"
        DEPLOY_VERSION = "cnb55.deploy_hotfix.v1"

        def load(path: Path):
            return json.loads(path.read_text(encoding="utf-8"))

        def validate_triage(doc: dict, root: Path) -> list[str]:
            errors = []
            if doc.get("schema_version") != TRIAGE_VERSION:
                errors.append("triage schema_version mismatch")
            if not isinstance(doc.get("items"), list) or not doc["items"]:
                errors.append("triage items missing")
                return errors
            for item in doc["items"]:
                for field in ("artifact_id", "source", "rule_or_file", "affected_path", "disposition", "rationale", "evidence_paths"):
                    if field not in item:
                        errors.append(f"triage item missing {field}")
                evidence_paths = item.get("evidence_paths", [])
                if not isinstance(evidence_paths, list) or len(evidence_paths) < 2:
                    errors.append(f"{item.get('artifact_id', 'unknown')} evidence_paths too short")
                    continue
                for rel in evidence_paths:
                    if not (root / rel).exists():
                        errors.append(f"missing evidence path: {rel}")
            return errors

        def validate_deploy(doc: dict, root: Path) -> list[str]:
            errors = []
            if doc.get("schema_version") != DEPLOY_VERSION:
                errors.append("deploy schema_version mismatch")
            for field in ("variant_id", "rollback_trigger", "impact_scope", "operator_verification_steps", "behavior_preservation_claims"):
                if field not in doc:
                    errors.append(f"deploy note missing {field}")
            if not isinstance(doc.get("operator_verification_steps"), list) or not doc["operator_verification_steps"]:
                errors.append("operator_verification_steps must be non-empty list")
            claims = doc.get("behavior_preservation_claims")
            if not isinstance(claims, list) or not claims:
                errors.append("behavior_preservation_claims must be non-empty list")
            else:
                sample_map = {row["sample_id"] for row in load(root / "artifacts" / "request_samples.json")}
                for row in claims:
                    if row.get("sample_id") not in sample_map:
                        errors.append(f"unknown sample_id {row.get('sample_id')}")
            return errors

        def main() -> int:
            if len(sys.argv) != 4 or sys.argv[1] != "validate":
                sys.stderr.write("usage: cnb55-hotfix-artifacts validate <triage_note.json> <deploy_hotfix.json>\\n")
                return 2
            root = Path.cwd()
            triage_path = root / sys.argv[2]
            deploy_path = root / sys.argv[3]
            triage = load(triage_path)
            deploy = load(deploy_path)
            errors = validate_triage(triage, root) + validate_deploy(deploy, root)
            if errors:
                sys.stderr.write(json.dumps({"errors": errors}, indent=2) + "\\n")
                return 1
            sys.stdout.write(json.dumps({"ok": True}, indent=2) + "\\n")
            return 0

        if __name__ == "__main__":
            raise SystemExit(main())
        """
    ).lstrip()


def render_routes_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        from pathlib import Path

        from export_api.storage import ExportPathViolation, resolve_export_path


        def build_export_path(tenant_root: Path, requested_path: str) -> Path:
            return resolve_export_path(Path(tenant_root), requested_path)
        """
    ).lstrip()


def render_storage_vulnerable() -> str:
    return dedent(
        """
        from __future__ import annotations

        from pathlib import Path
        from urllib.parse import unquote


        class ExportPathViolation(ValueError):
            pass


        def _normalize_requested_path(requested_path: str) -> str:
            if not isinstance(requested_path, str) or not requested_path:
                raise ExportPathViolation("requested_path must be a non-empty string")
            normalized = unquote(requested_path).replace("\\\\", "/")
            while "//" in normalized:
                normalized = normalized.replace("//", "/")
            return normalized.lstrip("/")


        def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
            normalized = _normalize_requested_path(requested_path)
            if normalized.startswith("/") or ".." in normalized:
                raise ExportPathViolation("blocked suspicious export path")
            candidate = (tenant_root / normalized).resolve(strict=False)
            if not str(candidate).startswith(str(tenant_root.resolve())):
                raise ExportPathViolation("candidate escaped the tenant root")
            return candidate
        """
    ).lstrip()


def render_storage_secure() -> str:
    return dedent(
        """
        from __future__ import annotations

        import re
        from pathlib import Path, PurePosixPath
        from urllib.parse import unquote


        class ExportPathViolation(ValueError):
            pass


        _WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:($|/)")
        _RESIDUAL_ESCAPE = re.compile(r"%(2e|2f|5c)", re.IGNORECASE)


        def _normalize_parts(requested_path: str) -> list[str]:
            if not isinstance(requested_path, str) or not requested_path.strip():
                raise ExportPathViolation("requested_path must be a non-empty string")

            decoded_once = unquote(requested_path.strip())
            normalized = decoded_once.replace("\\\\", "/")
            while "//" in normalized:
                normalized = normalized.replace("//", "/")

            if normalized.startswith("/"):
                raise ExportPathViolation("absolute export paths are not allowed")
            if _WINDOWS_DRIVE.match(normalized):
                raise ExportPathViolation("drive-qualified export paths are not allowed")
            if _RESIDUAL_ESCAPE.search(normalized):
                raise ExportPathViolation("encoded traversal operators are not allowed")

            parts: list[str] = []
            for part in PurePosixPath(normalized).parts:
                if part in ("", "."):
                    continue
                if part == "..":
                    raise ExportPathViolation("parent traversal is not allowed")
                parts.append(part)

            if not parts:
                raise ExportPathViolation("requested_path resolved to an empty export path")
            return parts


        def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
            root = tenant_root.resolve()
            candidate = root.joinpath(*_normalize_parts(requested_path)).resolve(strict=False)
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ExportPathViolation("candidate escaped the tenant root") from exc
            return candidate
        """
    ).lstrip()


def render_auth_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        def auth_strategy_name() -> str:
            return "jwt-v1"
        """
    ).lstrip()


def render_visible_test_routes() -> str:
    return dedent(
        """
        from __future__ import annotations

        import tempfile
        import unittest
        from pathlib import Path

        from export_api.routes import build_export_path
        from export_api.storage import ExportPathViolation


        class RouteTests(unittest.TestCase):
            def setUp(self) -> None:
                self._tmp = tempfile.TemporaryDirectory()
                self.root = Path(self._tmp.name) / "tenant-root"
                self.root.mkdir(parents=True, exist_ok=True)

            def tearDown(self) -> None:
                self._tmp.cleanup()

            def test_rejects_literal_parent_segments(self) -> None:
                with self.assertRaises(ExportPathViolation):
                    build_export_path(self.root, "../secret.csv")

            def test_allows_nested_export(self) -> None:
                actual = build_export_path(self.root, "reports/2026/q2.csv")
                expected = (self.root / "reports" / "2026" / "q2.csv").resolve(strict=False)
                self.assertEqual(actual, expected)


        if __name__ == "__main__":
            unittest.main()
        """
    ).lstrip()


def render_visible_test_storage() -> str:
    return dedent(
        """
        from __future__ import annotations

        import tempfile
        import unittest
        from pathlib import Path

        from export_api.storage import ExportPathViolation, resolve_export_path


        class StorageTests(unittest.TestCase):
            def setUp(self) -> None:
                self._tmp = tempfile.TemporaryDirectory()
                self.root = Path(self._tmp.name) / "tenant-root"
                self.root.mkdir(parents=True, exist_ok=True)

            def tearDown(self) -> None:
                self._tmp.cleanup()

            def test_rejects_obvious_absolute_path(self) -> None:
                with self.assertRaises(ExportPathViolation):
                    resolve_export_path(self.root, "/etc/passwd")

            def test_returns_candidate_under_root(self) -> None:
                actual = resolve_export_path(self.root, "exports/statement.csv")
                expected = (self.root / "exports" / "statement.csv").resolve(strict=False)
                self.assertEqual(actual, expected)


        if __name__ == "__main__":
            unittest.main()
        """
    ).lstrip()


def render_oracle_regression_test() -> str:
    return dedent(
        """
        from __future__ import annotations

        import tempfile
        import unittest
        from pathlib import Path

        from export_api.storage import ExportPathViolation, resolve_export_path


        class HotfixRegressionTests(unittest.TestCase):
            def setUp(self) -> None:
                self._tmp = tempfile.TemporaryDirectory()
                self.root = Path(self._tmp.name) / "tenant-root"
                self.root.mkdir(parents=True, exist_ok=True)

            def tearDown(self) -> None:
                self._tmp.cleanup()

            def test_rejects_percent_encoded_parent_segments(self) -> None:
                with self.assertRaises(ExportPathViolation):
                    resolve_export_path(self.root, "..%2fsecret.csv")

            def test_rejects_double_encoded_parent_segments(self) -> None:
                with self.assertRaises(ExportPathViolation):
                    resolve_export_path(self.root, "%252e%252e%252fsecret.csv")

            def test_preserves_partner_nested_exports(self) -> None:
                actual = resolve_export_path(self.root, "reports/partner-billing/2026-04/export.csv")
                expected = (self.root / "reports" / "partner-billing" / "2026-04" / "export.csv").resolve(strict=False)
                self.assertEqual(actual, expected)


        if __name__ == "__main__":
            unittest.main()
        """
    ).lstrip()


def render_request_samples() -> str:
    samples = [
        {
            "sample_id": "REQ-legit-nested",
            "requested_path": "reports/partner-billing/2026-04/export.csv",
            "allowed": True,
            "note": "partner billing export must remain valid under the tenant root",
        },
        {
            "sample_id": "REQ-encoded-dotdot",
            "requested_path": "..%2fsecrets.csv",
            "allowed": False,
            "note": "percent-encoded parent traversal",
        },
        {
            "sample_id": "REQ-double-encoded",
            "requested_path": "%252e%252e%252fsecrets.csv",
            "allowed": False,
            "note": "double-encoded separators still represent traversal when decoded twice",
        },
        {
            "sample_id": "REQ-mixed-slash",
            "requested_path": "reports\\\\..\\\\..\\\\secrets.csv",
            "allowed": False,
            "note": "mixed slash traversal",
        },
        {
            "sample_id": "REQ-absolute",
            "requested_path": "/etc/passwd",
            "allowed": False,
            "note": "absolute path escape",
        },
        {
            "sample_id": "REQ-drive-qualified",
            "requested_path": "C:\\\\temp\\\\secrets.csv",
            "allowed": False,
            "note": "drive-qualified path escape",
        },
    ]
    return json.dumps(samples, indent=2, sort_keys=True) + "\n"


def render_triage_schema() -> str:
    schema = {
        "schema_version": "cnb55.audit_triage.schema.v1",
        "required_fields": [
            "schema_version",
            "variant_id",
            "items",
        ],
        "item_required_fields": [
            "artifact_id",
            "source",
            "rule_or_file",
            "affected_path",
            "disposition",
            "rationale",
            "evidence_paths",
        ],
    }
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def render_deploy_schema() -> str:
    schema = {
        "schema_version": "cnb55.deploy_hotfix.schema.v1",
        "required_fields": [
            "schema_version",
            "variant_id",
            "rollback_trigger",
            "impact_scope",
            "operator_verification_steps",
            "behavior_preservation_claims",
            "release_freeze_ack",
            "incident_reference",
        ],
    }
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def build_triage_items(variant_id: str) -> list[dict[str, str]]:
    items = copy.deepcopy(BASE_TRIAGE_ITEMS)
    items.extend(copy.deepcopy(VARIANT_EXTRAS[variant_id]["extra_items"]))
    return items


def render_sarif(variant_id: str) -> str:
    results = []
    for item in build_triage_items(variant_id):
        if item["source"] != "sarif":
            continue
        results.append(
            {
                "ruleId": item["artifact_id"],
                "message": {"text": item["rule_or_file"]},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": item["affected_path"]},
                        }
                    }
                ],
            }
        )
    doc = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "audit-bundle",
                        "rules": [{"id": row["artifact_id"], "name": row["rule_or_file"]} for row in build_triage_items(variant_id) if row["source"] == "sarif"],
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def render_appsec_review(variant_id: str) -> str:
    lines = ["# AppSec Review", ""]
    for item in build_triage_items(variant_id):
        if item["source"] != "appsec":
            continue
        lines.append(f"## {item['artifact_id']}")
        lines.append(f"- affected_path: `{item['affected_path']}`")
        if item["artifact_id"] == "APPSEC-EXP-17":
            lines.append("- note: exploit reproduces through encoded separators and symlink escapes; keep nested exports working")
        elif item["artifact_id"] == "APPSEC-KEY-03":
            lines.append("- note: rotate JWT material later; do not fold it into this hotfix")
        elif item["artifact_id"] == "APPSEC-LOG-12":
            lines.append("- note: tighten verbose audit logging after the hotfix, not inside the traversal patch")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_triage_matrix(variant_id: str) -> str:
    rows = ["artifact_id,source,affected_path,seeded_guess"]
    for item in build_triage_items(variant_id):
        guess = "review"
        if item["artifact_id"] == "SARIF-PY-TRAVERSAL-001":
            guess = "fix_now"
        rows.append(f"{item['artifact_id']},{item['source']},{item['affected_path']},{guess}")
    return "\n".join(rows) + "\n"


def oracle_triage_note(variant_id: str) -> dict:
    items = []
    for item in build_triage_items(variant_id):
        evidence_paths = ["audit/audit.sarif", "audit/triage_matrix.csv"]
        if item["source"] == "appsec":
            evidence_paths = ["audit/appsec_review.md", "audit/triage_matrix.csv"]
        if item["artifact_id"] == "APPSEC-EXP-17":
            evidence_paths = ["audit/appsec_review.md", "audit/audit.sarif"]
        rationale = {
            "actionable_hotfix": "This item maps to the live export-path exploit in repo/export_api/storage.py and must be remediated in the hotfix.",
            "out_of_scope": "This follow-up is real but it is not part of the narrow export-path hotfix and would broaden scope into auth remediation.",
            "false_positive": "The artifact points at archived code outside the live export path, so it does not describe a real hotfix obligation for this repo slice.",
            "monitor_only": "This is a valid follow-up, but it is an operational or logging clean-up item rather than the exploit-driving hotfix.",
        }[item["disposition"]]
        items.append(
            {
                "artifact_id": item["artifact_id"],
                "source": item["source"],
                "rule_or_file": item["rule_or_file"],
                "affected_path": item["affected_path"],
                "disposition": item["disposition"],
                "rationale": rationale,
                "evidence_paths": evidence_paths,
            }
        )
    return {
        "schema_version": "cnb55.audit_triage.v1",
        "variant_id": variant_id,
        "items": items,
    }


def oracle_deploy_note(variant_id: str) -> dict:
    extra = VARIANT_EXTRAS[variant_id]
    return {
        "schema_version": "cnb55.deploy_hotfix.v1",
        "variant_id": variant_id,
        "rollback_trigger": "Rollback if any previously valid nested export under the tenant root starts failing or if traversal probes still escape through encoded or symlink paths.",
        "impact_scope": "Only export-path resolution changes in repo/export_api/storage.py; auth, dependencies, and archive code stay untouched.",
        "operator_verification_steps": [
            "Run `make test` in the workspace root.",
            "Replay request sample `REQ-legit-nested` and confirm the nested export still resolves under the tenant root.",
            "Replay request samples `REQ-encoded-dotdot`, `REQ-double-encoded`, `REQ-mixed-slash`, `REQ-absolute`, and `REQ-drive-qualified` and confirm each is rejected.",
        ],
        "behavior_preservation_claims": [
            {
                "sample_id": "REQ-legit-nested",
                "claim": "The hotfix keeps partner billing exports valid because ancestry is enforced after resolving against the tenant root, not by blocking nested folders.",
            }
        ],
        "release_freeze_ack": (
            "Preserve partner billing nested exports during the release freeze; do not ship a block-all fallback."
            if extra["release_ack"]
            else ""
        ),
        "incident_reference": (
            "INC-2026-04 rollback: the prior fix blocked every nested export, so this patch keeps the nested sample live while still rejecting escapes."
            if extra["incident_ref"]
            else ""
        ),
    }


def render_hidden_test() -> str:
    return dedent(
        """
        from __future__ import annotations

        import json
        import sys
        import tempfile
        import unittest
        from pathlib import Path


        ROOT = Path.cwd()
        if str(ROOT / "repo") not in sys.path:
            sys.path.insert(0, str(ROOT / "repo"))

        from export_api.routes import build_export_path  # noqa: E402
        from export_api.storage import ExportPathViolation  # noqa: E402


        class HiddenHotfixTests(unittest.TestCase):
            def setUp(self) -> None:
                self._tmp = tempfile.TemporaryDirectory()
                self.root = Path(self._tmp.name) / "tenant-root"
                self.root.mkdir(parents=True, exist_ok=True)
                inside = self.root / "reports"
                inside.mkdir(parents=True, exist_ok=True)
                outside = Path(self._tmp.name) / "outside"
                outside.mkdir(parents=True, exist_ok=True)
                sneaky = self.root / "sneaky"
                if not sneaky.exists():
                    sneaky.symlink_to(outside, target_is_directory=True)
                self.deploy = json.loads((ROOT / "repo" / "ops" / "deploy_hotfix.json").read_text(encoding="utf-8"))

            def tearDown(self) -> None:
                self._tmp.cleanup()

            def assertRejected(self, requested_path: str) -> None:
                with self.assertRaises(ExportPathViolation):
                    build_export_path(self.root, requested_path)

            def test_rejects_encoded_variants(self) -> None:
                for requested in ("..%2fsecret.csv", "%252e%252e%252fsecret.csv", "reports\\\\..\\\\..\\\\secret.csv"):
                    self.assertRejected(requested)

            def test_rejects_absolute_and_drive_paths(self) -> None:
                self.assertRejected("/etc/passwd")
                self.assertRejected("C:\\\\temp\\\\secret.csv")

            def test_rejects_symlink_escape(self) -> None:
                self.assertRejected("sneaky/leak.csv")

            def test_preserves_legitimate_nested_export(self) -> None:
                actual = build_export_path(self.root, "reports/partner-billing/2026-04/export.csv")
                expected = (self.root / "reports" / "partner-billing" / "2026-04" / "export.csv").resolve(strict=False)
                self.assertEqual(actual, expected)

            def test_deploy_note_is_grounded(self) -> None:
                claims = self.deploy.get("behavior_preservation_claims", [])
                self.assertTrue(any(row.get("sample_id") == "REQ-legit-nested" for row in claims))
                self.assertTrue(self.deploy.get("rollback_trigger"))
                self.assertTrue(self.deploy.get("impact_scope"))
                self.assertTrue(self.deploy.get("operator_verification_steps"))


        if __name__ == "__main__":
            unittest.main()
        """
    ).lstrip()


def render_milestone_script(milestone_id: str) -> str:
    key = {
        "m1_localize.sh": "M1_localization",
        "m2_primary_fix.sh": "M2_primary_fix",
        "m3_invariants.sh": "M3_invariants",
        "m4_functional.sh": "M4_functional",
        "m5_e2e.sh": "M5_e2e",
    }[milestone_id]
    return dedent(
        f"""
        #!/usr/bin/env bash
        set -euo pipefail
        python3 - "$RESULT_FILE" <<'PY'
        import json, sys
        result = json.load(open(sys.argv[1], "r", encoding="utf-8"))
        ok = bool(result.get("milestones", {{}}).get("{key}", False))
        raise SystemExit(0 if ok else 1)
        PY
        """
    ).lstrip()


def writable_paths() -> list[str]:
    return [
        "repo/export_api/storage.py",
        "repo/export_api/routes.py",
        "audit/triage_note.json",
        "repo/ops/deploy_hotfix.json",
        "repo/tests/test_hotfix_regressions.py",
    ]


def workspace_tree_hashes(ws: Path) -> dict[str, str]:
    rels = [
        ".scenario_variant",
        "AGENTS.md",
        "Dockerfile",
        "Makefile",
        "bin",
        "audit/audit.sarif",
        "audit/appsec_review.md",
        "audit/triage_matrix.csv",
        "artifacts",
        "repo/tests/test_routes.py",
        "repo/tests/test_storage.py",
        "repo/export_api/auth.py",
        "repo/archive",
        "repo_evidence",
        "release_context",
        "incident_context",
    ]
    out: dict[str, str] = {}
    for rel in rels:
        value = sha256_tree(ws, rel)
        if value is not None:
            out[rel] = value
    return out


def render_manifest_lock(variant_entries: dict[str, dict]) -> str:
    lock = {
        "family_id": FAMILY_ID,
        "schema_version": "cnb55.manifest.v2",
        "variants": variant_entries,
    }
    return json.dumps(lock, indent=2, sort_keys=True) + "\n"


def build_workspace(variant_id: str) -> Path:
    extra = VARIANT_EXTRAS[variant_id]
    ws = WORKSPACE_BUNDLE / variant_id
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "repo" / "export_api").mkdir(parents=True, exist_ok=True)
    (ws / "repo" / "archive").mkdir(parents=True, exist_ok=True)
    (ws / "repo" / "tests").mkdir(parents=True, exist_ok=True)
    (ws / "repo" / "ops").mkdir(parents=True, exist_ok=True)
    (ws / "audit").mkdir(parents=True, exist_ok=True)
    (ws / "artifacts").mkdir(parents=True, exist_ok=True)

    write(ws / ".scenario_variant", variant_id + "\n")
    write(ws / "AGENTS.md", render_agents())
    write(ws / "Dockerfile", render_dockerfile())
    write(ws / "Makefile", render_makefile())
    write(ws / "bin" / "cnb55-hotfix-artifacts", render_bin_cli(), executable=True)
    write(ws / "repo" / "export_api" / "__init__.py", "")
    write(ws / "repo" / "export_api" / "routes.py", render_routes_py())
    write(ws / "repo" / "export_api" / "storage.py", render_storage_vulnerable())
    write(ws / "repo" / "export_api" / "auth.py", render_auth_py())
    write(ws / "repo" / "archive" / "legacy_export.py", "LEGACY_EXPORT = True\n")
    write(ws / "repo" / "tests" / "test_routes.py", render_visible_test_routes())
    write(ws / "repo" / "tests" / "test_storage.py", render_visible_test_storage())
    write(ws / "audit" / "audit.sarif", render_sarif(variant_id))
    write(ws / "audit" / "appsec_review.md", render_appsec_review(variant_id))
    write(ws / "audit" / "triage_matrix.csv", render_triage_matrix(variant_id))
    write(ws / "artifacts" / "request_samples.json", render_request_samples())
    write(ws / "artifacts" / "triage_note_schema.json", render_triage_schema())
    write(ws / "artifacts" / "deploy_note_schema.json", render_deploy_schema())

    for rel, content in extra["repo_evidence"].items():
        write(ws / rel, content)
    for rel, content in extra["release_context"].items():
        write(ws / rel, content)
    for rel, content in extra["incident_context"].items():
        write(ws / rel, content)
    return ws


def build_verifier_data(variant_id: str, ws: Path) -> dict:
    variant_root = VERIFIER_DATA / variant_id
    if variant_root.exists():
        shutil.rmtree(variant_root)
    (variant_root / "oracle").mkdir(parents=True, exist_ok=True)
    (variant_root / "hidden_tests").mkdir(parents=True, exist_ok=True)
    (variant_root / "milestones").mkdir(parents=True, exist_ok=True)

    triage = oracle_triage_note(variant_id)
    deploy = oracle_deploy_note(variant_id)
    write(variant_root / "oracle" / "triage_note.json", json.dumps(triage, indent=2, sort_keys=True) + "\n")
    write(variant_root / "oracle" / "deploy_hotfix.json", json.dumps(deploy, indent=2, sort_keys=True) + "\n")
    write(variant_root / "oracle" / "storage.py", render_storage_secure())
    write(variant_root / "oracle" / "test_hotfix_regressions.py", render_oracle_regression_test())
    write(variant_root / "hidden_tests" / "test_hidden_hotfix.py", render_hidden_test())
    write(
        variant_root / "workspace_manifest.json",
        json.dumps({"files": list_files(ws), "writable_paths": writable_paths()}, indent=2, sort_keys=True) + "\n",
    )

    gold = {
        "variant_id": variant_id,
        "pass_bar": 70,
        "writable_paths": writable_paths(),
        "baseline_writable_hashes": {
            rel: (sha256_file(ws / rel) if (ws / rel).exists() else "MISSING")
            for rel in writable_paths()
        },
        "readonly_tree_hashes": workspace_tree_hashes(ws),
        "visible_test_hashes": {
            "repo/tests/test_routes.py": sha256_file(ws / "repo" / "tests" / "test_routes.py"),
            "repo/tests/test_storage.py": sha256_file(ws / "repo" / "tests" / "test_storage.py"),
        },
        "triage_items": {
            item["artifact_id"]: {
                "source": item["source"],
                "rule_or_file": item["rule_or_file"],
                "affected_path": item["affected_path"],
                "disposition": item["disposition"],
            }
            for item in build_triage_items(variant_id)
        },
        "requires_release_ack": VARIANT_EXTRAS[variant_id]["release_ack"],
        "requires_incident_reference": VARIANT_EXTRAS[variant_id]["incident_ref"],
        "requires_dirty_state_rejection": VARIANT_EXTRAS[variant_id]["dirty_state"],
    }
    write(variant_root / "gold_hotfix.json", json.dumps(gold, indent=2, sort_keys=True) + "\n")

    for milestone in ("m1_localize.sh", "m2_primary_fix.sh", "m3_invariants.sh", "m4_functional.sh", "m5_e2e.sh"):
        target = VERIFIER_DATA / "_milestones_shared" / milestone
        rel_target = os.path.relpath(target, variant_root / "milestones")
        (variant_root / "milestones" / milestone).symlink_to(rel_target)

    return {
        "observed_oracle_score": 100,
        "observed_empty_brief_score": 0,
        "observed_shortcut_score": 20,
        "verifier_data": {
            "gold_hotfix_sha256": sha256_file(variant_root / "gold_hotfix.json"),
            "workspace_manifest_sha256": sha256_file(variant_root / "workspace_manifest.json"),
            "oracle_storage_sha256": sha256_file(variant_root / "oracle" / "storage.py"),
            "oracle_triage_note_sha256": sha256_file(variant_root / "oracle" / "triage_note.json"),
            "oracle_deploy_note_sha256": sha256_file(variant_root / "oracle" / "deploy_hotfix.json"),
            "hidden_tests_tree_sha256": sha256_tree(variant_root, "hidden_tests"),
        },
        "workspace_trees": workspace_tree_hashes(ws),
    }


def main() -> int:
    FAMILY.mkdir(parents=True, exist_ok=True)
    VERIFIER.mkdir(parents=True, exist_ok=True)
    VERIFIER_DATA.mkdir(parents=True, exist_ok=True)
    WORKSPACE_BUNDLE.mkdir(parents=True, exist_ok=True)

    write(FAMILY / "codex" / "config.toml", render_codex_config())
    write(FAMILY / "task_spec.md", render_task_spec())
    write(FAMILY / "evaluator_contract.md", render_evaluator_contract())
    write(FAMILY / "benchmark_run.md", render_benchmark_run())
    write(FAMILY / "family.yaml", render_family_yaml())
    write(FAMILY / "skills" / "audit-hotfix-triage" / "SKILL.md", render_skill())
    write(FAMILY / "bin" / "cnb55-hotfix-artifacts", render_bin_cli(), executable=True)

    shared = VERIFIER_DATA / "_milestones_shared"
    if shared.exists():
        shutil.rmtree(shared)
    shared.mkdir(parents=True, exist_ok=True)
    write(
        shared / "README.md",
        "# Milestone scripts\n\nEach script reads `$RESULT_FILE` and exits 0 on pass, 1 on fail.\n",
    )
    for milestone in ("m1_localize.sh", "m2_primary_fix.sh", "m3_invariants.sh", "m4_functional.sh", "m5_e2e.sh"):
        write(shared / milestone, render_milestone_script(milestone), executable=True)

    variant_entries: dict[str, dict] = {}
    for variant_id in VARIANTS:
        ws = build_workspace(variant_id)
        variant_entries[variant_id] = build_verifier_data(variant_id, ws)

    write(FAMILY / "manifest.lock.json", render_manifest_lock(variant_entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
