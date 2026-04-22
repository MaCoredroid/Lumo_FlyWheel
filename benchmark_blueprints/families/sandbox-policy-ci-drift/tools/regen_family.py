#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
FAMILY = REPO / "benchmark_blueprints/families/sandbox-policy-ci-drift"
VERIFIERS = REPO / "verifiers/sandbox-policy-ci-drift"
VERIFIER_DATA = REPO / "verifier_data/sandbox-policy-ci-drift"
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"

FAMILY_ID = "sandbox-policy-ci-drift"
TASK_ID = "cnb55-core-sandbox-policy-ci-drift-local-ci-contract"
VERIFY_RESULT_SCHEMA = "cnb55.verify_result.v3"
VARIANTS: dict[str, dict[str, str]] = {
    "v1-clean-baseline": {
        "title": "Clean baseline",
        "note": "Baseline: repair parser, preview, CI workflow, config, and release note without dropping the deprecated input alias.",
    },
    "v2-noisy-distractor": {
        "title": "Noisy distractor",
        "note": "Ignore archived docs that still mention `manual-review` and hyphenated sandbox tokens.",
    },
    "v3-dirty-state": {
        "title": "Dirty state",
        "note": "A half-finished helper-path shortcut exists in repo evidence; do not stop at fixture-only normalization.",
    },
    "v4-multi-corpus-objective": {
        "title": "Multi-corpus objective drift",
        "note": "release_context/ makes operator-facing canonical preview output and doc alignment part of correctness, not polish.",
    },
    "v5-recovery-in-thread": {
        "title": "Recovery in thread",
        "note": "incident_context/ records a rollback caused by removing the deprecated alias. Preserve compatibility while keeping emitted output canonical.",
    },
}

EDITABLE_FILES = [
    ".github/workflows/ci.yml",
    "codex/config.py",
    "codex/config.toml",
    "codex/policy.py",
    "docs/releases/policy-rename.md",
    "scripts/render_workflow_preview.py",
    "scripts/run_ci.py",
]
CODE_FILES = [
    "codex/config.py",
    "codex/policy.py",
    "scripts/render_workflow_preview.py",
    "scripts/run_ci.py",
]
READONLY_HASH_RELS = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "Makefile",
    "codex/__init__.py",
    "scripts/__init__.py",
    "tests",
    "docs/archive",
    "repo_evidence",
    "release_context",
    "incident_context",
]
RUNTIME_CACHE_PREFIXES = (".pytest_cache/", "__pycache__/", ".mypy_cache/")


def write(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str:
    target = root / rel
    if not target.exists():
        return "MISSING"
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_file(target).encode())
        return h.hexdigest()
    for path in sorted(target.rglob("*")):
        rel_path = path.relative_to(target).as_posix()
        if path.is_file():
            h.update(b"F:" + rel_path.encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
        elif path.is_dir():
            h.update(b"D:" + rel_path.encode() + b"\x00")
    return h.hexdigest()


def collect_manifest(root: Path) -> dict[str, Any]:
    files: list[str] = []
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        files.append(rel)
        hashes[rel] = sha256_file(path)
    return {"files": files, "file_sha256": hashes}


def common_agents_md() -> str:
    return textwrap.dedent(
        """
        # Sandbox Policy CI Drift

        Repair the repo so local CI, the workflow preview generator, and the
        checked-in workflow all use the same Codex sandbox and approval-policy
        contract.

        Required outcomes:

        - Canonical active tokens are `workspace_write` and `on_request`.
        - The deprecated input spelling `workspace-write` must still parse from
          config files for compatibility.
        - Emitted preview JSON and the workflow file must stay canonical-only.
        - Update the release note so operators understand the compatibility rule.

        Editable surfaces:

        - `codex/policy.py`
        - `codex/config.py`
        - `scripts/render_workflow_preview.py`
        - `scripts/run_ci.py`
        - `.github/workflows/ci.yml`
        - `codex/config.toml`
        - `docs/releases/policy-rename.md`

        Immutable surfaces:

        - `tests/*`
        - `tests/fixtures/*`
        - `docs/archive/*`
        - `repo_evidence/*`
        - `release_context/*`
        - `incident_context/*`
        - `Makefile`
        - `Dockerfile`
        - `AGENTS.md`

        Final validation command:

        - `make ci`

        Variant details live in `.scenario_variant`.
        """
    )


def dockerfile_text() -> str:
    return textwrap.dedent(
        """
        FROM python:3.11-slim

        WORKDIR /workspace
        RUN python -m pip install --no-cache-dir pytest
        """
    )


def makefile_text() -> str:
    return textwrap.dedent(
        """
        PYTHON ?= python3

        .PHONY: ci

        ci:
        \t$(PYTHON) scripts/run_ci.py
        """
    )


def visible_fixture_toml() -> str:
    return textwrap.dedent(
        """
        [policy]
        sandbox = "workspace_write"
        approval_policy = "on_request"
        """
    )


def config_toml() -> str:
    return textwrap.dedent(
        """
        [policy]
        sandbox = "workspace_write"
        approval_policy = "on_request"
        """
    )


def policy_py(broken: bool) -> str:
    if broken:
        return textwrap.dedent(
            """
            from __future__ import annotations

            OLD_SANDBOX_NAMES = {
                "read-only": "read_only",
                "workspace-write": "workspace_write",
                "danger-full-access": "danger_full_access",
            }
            APPROVAL_NAMES = {"never", "on_request", "on_failure", "untrusted"}


            def parse_sandbox(value: str) -> str:
                raw = str(value).strip()
                if raw in OLD_SANDBOX_NAMES:
                    return OLD_SANDBOX_NAMES[raw]
                raise ValueError(f"unsupported sandbox policy: {value!r}")


            def parse_approval_policy(value: str) -> str:
                raw = str(value).strip()
                if raw in APPROVAL_NAMES:
                    return raw
                raise ValueError(f"unsupported approval policy: {value!r}")


            def normalize_policy(policy: dict[str, str]) -> dict[str, str]:
                return {
                    "sandbox": parse_sandbox(policy["sandbox"]),
                    "approval_policy": parse_approval_policy(policy["approval_policy"]),
                }


            def preview_contract(policy: dict[str, str]) -> dict[str, str]:
                parsed = normalize_policy(policy)
                approval = parsed["approval_policy"]
                if approval == "on_request":
                    approval = "manual-review"
                return {
                    "sandbox": parsed["sandbox"],
                    "approval_policy": approval,
                }
            """
        )
    return textwrap.dedent(
        """
        from __future__ import annotations

        SANDBOX_NAMES = {
            "read_only": "read_only",
            "workspace_write": "workspace_write",
            "danger_full_access": "danger_full_access",
        }
        DEPRECATED_SANDBOX_ALIASES = {
            "workspace-write": "workspace_write",
        }
        APPROVAL_NAMES = {"never", "on_request", "on_failure", "untrusted"}


        def parse_sandbox(value: str) -> str:
            raw = str(value).strip()
            if raw in SANDBOX_NAMES:
                return SANDBOX_NAMES[raw]
            if raw in DEPRECATED_SANDBOX_ALIASES:
                return DEPRECATED_SANDBOX_ALIASES[raw]
            raise ValueError(f"unsupported sandbox policy: {value!r}")


        def parse_approval_policy(value: str) -> str:
            raw = str(value).strip()
            if raw in APPROVAL_NAMES:
                return raw
            raise ValueError(f"unsupported approval policy: {value!r}")


        def normalize_policy(policy: dict[str, str]) -> dict[str, str]:
            return {
                "sandbox": parse_sandbox(policy["sandbox"]),
                "approval_policy": parse_approval_policy(policy["approval_policy"]),
            }


        def preview_contract(policy: dict[str, str]) -> dict[str, str]:
            return normalize_policy(policy)
        """
    )


def config_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        from pathlib import Path

        from codex.policy import normalize_policy


        def _parse_policy_table(text: str) -> dict[str, str]:
            in_policy = False
            out: dict[str, str] = {}
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    in_policy = line == "[policy]"
                    continue
                if not in_policy or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip().strip('"')
            return out


        def load_config(path: str | Path) -> dict[str, str]:
            target = Path(path)
            policy = _parse_policy_table(target.read_text(encoding="utf-8"))
            if "sandbox" not in policy or "approval_policy" not in policy:
                raise ValueError("config file must define sandbox and approval_policy")
            return normalize_policy(
                {
                    "sandbox": str(policy["sandbox"]),
                    "approval_policy": str(policy["approval_policy"]),
                }
            )
        """
    )


def render_preview_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import argparse
        import json
        from pathlib import Path

        from codex.config import load_config
        from codex.policy import preview_contract


        def render_preview(config_path: str | Path = "codex/config.toml") -> dict[str, object]:
            policy = load_config(config_path)
            preview_policy = preview_contract(policy)
            return {
                "schema_version": "sandbox_preview.v1",
                "sandbox": preview_policy["sandbox"],
                "approval_policy": preview_policy["approval_policy"],
                "jobs": [
                    {
                        "job_id": "policy-smoke",
                        "sandbox": preview_policy["sandbox"],
                        "approval_policy": preview_policy["approval_policy"],
                    },
                    {
                        "job_id": "preview-render",
                        "sandbox": preview_policy["sandbox"],
                        "approval_policy": preview_policy["approval_policy"],
                    },
                ],
            }


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--config", default="codex/config.toml")
            args = parser.parse_args()
            print(json.dumps(render_preview(args.config), indent=2, sort_keys=True))
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def broken_run_ci_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import os
        import shutil
        import subprocess
        import sys
        import tempfile
        from pathlib import Path


        def main() -> int:
            env = os.environ.copy()
            visible_fixture = Path("tests/fixtures/visible_config.toml")
            with tempfile.TemporaryDirectory(prefix="visible_fixture_") as tmp:
                temp_fixture = Path(tmp) / "visible_config.toml"
                shutil.copy(visible_fixture, temp_fixture)
                patched = temp_fixture.read_text(encoding="utf-8").replace(
                    "workspace_write",
                    "workspace-write",
                )
                temp_fixture.write_text(patched, encoding="utf-8")
                env["CODEX_CONFIG_FIXTURE"] = str(temp_fixture)
                return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], env=env)


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def fixed_run_ci_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import subprocess
        import sys


        def main() -> int:
            return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"])


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def workflow_yml(broken: bool) -> str:
    preview_approval = "manual-review" if broken else "on_request"
    return textwrap.dedent(
        f"""
        name: CI

        on:
          push:
            branches: [main]
          pull_request:

        jobs:
          policy-smoke:
            runs-on: ubuntu-latest
            env:
              SANDBOX_POLICY: workspace_write
              APPROVAL_POLICY: on_request
            steps:
              - uses: actions/checkout@v4
              - run: make ci

          preview-render:
            runs-on: ubuntu-latest
            env:
              SANDBOX_POLICY: workspace_write
              APPROVAL_POLICY: {preview_approval}
            steps:
              - uses: actions/checkout@v4
              - run: python3 scripts/render_workflow_preview.py --config codex/config.toml
        """
    )


def release_note_md(variant_id: str, broken: bool) -> str:
    extra = ""
    if not broken and variant_id == "v4-multi-corpus-objective":
        extra = "\nOperators consume preview JSON downstream, so active preview output must stay canonical-only.\n"
    if not broken and variant_id == "v5-recovery-in-thread":
        extra = "\nThe previous rename-only hotfix was rolled back because it removed the `workspace-write` compatibility path.\n"
    if broken:
        return textwrap.dedent(
            """
            # Policy Rename

            The repo moved from hyphenated sandbox tokens to underscored names.
            This release is fully breaking: old config spellings no longer work.

            Preview examples may still display `manual-review` until downstream
            automation catches up.
            """
        )
    return textwrap.dedent(
        f"""
        # Policy Rename

        Canonical active policy names are `workspace_write` for sandbox and
        `on_request` for approval.

        Compatibility:
        - local config parsing still accepts the deprecated input spelling
          `workspace-write`
        - emitted preview JSON and workflow examples stay canonical-only and
          always emit `workspace_write` plus `on_request`

        Reviewer validation:
        - run `make ci`

        {extra.strip()}
        """
    ).strip() + "\n"


def tests_init_py() -> str:
    return ""


def tests_helpers_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import os
        from pathlib import Path


        def fixture_path() -> Path:
            return Path(os.environ.get("CODEX_CONFIG_FIXTURE", "tests/fixtures/visible_config.toml"))
        """
    )


def test_policy_parser_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        from codex.config import load_config
        from tests.helpers import fixture_path


        def test_parser_accepts_visible_fixture() -> None:
            loaded = load_config(fixture_path())
            assert loaded["sandbox"] == "workspace_write"
            assert loaded["approval_policy"] == "on_request"
        """
    )


def test_config_loading_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        from codex.config import load_config
        from tests.helpers import fixture_path


        def test_config_loading_returns_canonical_tokens() -> None:
            loaded = load_config(fixture_path())
            assert loaded == {
                "sandbox": "workspace_write",
                "approval_policy": "on_request",
            }
        """
    )


def test_workflow_preview_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        from scripts.render_workflow_preview import render_preview
        from tests.helpers import fixture_path


        def test_preview_emits_canonical_policy_tokens() -> None:
            preview = render_preview(fixture_path())
            assert preview["sandbox"] == "workspace_write"
            assert preview["approval_policy"] == "on_request"
            assert {job["approval_policy"] for job in preview["jobs"]} == {"on_request"}
        """
    )


def archived_doc() -> str:
    return textwrap.dedent(
        """
        # Archived Preview Snapshot

        Historical snapshot only:
        - sandbox token: `workspace-write`
        - approval token: `manual-review`

        This file is stale and should not be copied into active workflow or
        release-note surfaces.
        """
    )


def helper_patch_note() -> str:
    return textwrap.dedent(
        """
        # Abandoned Helper Shortcut

        A previous attempt normalized only `tests/fixtures/visible_config.toml`
        before invoking pytest. That made one happy-path fixture green without
        fixing parser compatibility for real configs.
        """
    )


def release_context_note() -> str:
    return textwrap.dedent(
        """
        # Preview Consumer Contract

        Release operators ingest preview JSON directly to generate CI review
        screenshots. Active preview output must stay canonical-only even when
        deprecated local config spellings still parse.
        """
    )


def incident_context_note() -> str:
    return textwrap.dedent(
        """
        # Rollback 2026-04

        The previous rename-only hotfix removed `workspace-write` input support.
        Stored examples still used that spelling, so the patch was rolled back
        after breaking local dry runs. Preserve compatibility on input while
        keeping emitted preview and workflow tokens canonical.
        """
    )


def hidden_test_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import json
        import os
        import tempfile
        from pathlib import Path

        from codex.config import load_config
        from scripts.render_workflow_preview import render_preview


        ROOT = Path(os.environ["AGENT_WS"]).resolve()
        VARIANT = Path(__file__).resolve().parents[1].name
        GOLD = json.loads((Path(__file__).resolve().parents[1] / "gold_repair.json").read_text())


        def _temp_config(text: str) -> Path:
            tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
            tmp.write(text)
            tmp.flush()
            tmp.close()
            return Path(tmp.name)


        def test_actual_config_is_canonical() -> None:
            config_path = ROOT / "codex" / "config.toml"
            loaded = load_config(config_path)
            assert loaded == GOLD["expected_policy"]


        def test_deprecated_input_still_parses() -> None:
            alias = GOLD["deprecated_alias"]
            config_path = _temp_config(
                f"[policy]\\nsandbox = \\"{alias['input']}\\"\\napproval_policy = \\"on_request\\"\\n"
            )
            loaded = load_config(config_path)
            assert loaded["sandbox"] == alias["canonical"]


        def test_preview_output_stays_canonical_for_alias_input() -> None:
            alias = GOLD["deprecated_alias"]
            config_path = _temp_config(
                f"[policy]\\nsandbox = \\"{alias['input']}\\"\\napproval_policy = \\"on_request\\"\\n"
            )
            preview = render_preview(config_path)
            assert preview["sandbox"] == alias["canonical"]
            assert preview["approval_policy"] == GOLD["expected_policy"]["approval_policy"]
            assert {job["approval_policy"] for job in preview["jobs"]} == {GOLD["expected_policy"]["approval_policy"]}


        def test_workflow_file_uses_one_canonical_approval_token_family() -> None:
            workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
            assert "manual-review" not in workflow
            assert workflow.count("APPROVAL_POLICY: on_request") == 2
            assert workflow.count("SANDBOX_POLICY: workspace_write") == 2


        def test_run_ci_does_not_normalize_only_the_visible_fixture() -> None:
            run_ci = (ROOT / "scripts" / "run_ci.py").read_text(encoding="utf-8")
            assert "visible_config.toml" not in run_ci
            assert 'replace("workspace_write", "workspace-write")' not in run_ci
            assert "CODEX_CONFIG_FIXTURE" not in run_ci


        def test_release_note_documents_canonical_output_and_compatibility() -> None:
            note = (ROOT / "docs" / "releases" / "policy-rename.md").read_text(encoding="utf-8")
            assert "workspace-write" in note
            assert "workspace_write" in note
            assert "on_request" in note
            assert "manual-review" not in note
            assert "fully breaking" not in note
            assert "make ci" in note


        def test_archive_noise_stays_archived_when_present() -> None:
            archived = ROOT / "docs" / "archive" / "legacy-preview-contract.md"
            if archived.exists():
                active = (ROOT / "docs" / "releases" / "policy-rename.md").read_text(encoding="utf-8")
                assert "Historical snapshot only" in archived.read_text(encoding="utf-8")
                assert "manual-review" not in active


        def test_release_context_requires_preview_consumer_alignment_when_present() -> None:
            context = ROOT / "release_context" / "preview-consumer-contract.md"
            if context.exists():
                note = (ROOT / "docs" / "releases" / "policy-rename.md").read_text(encoding="utf-8")
                assert "preview" in note.lower()
                assert "canonical-only" in note.lower()


        def test_incident_context_requires_rollback_aware_compatibility_when_present() -> None:
            incident = ROOT / "incident_context" / "rollback_2026_04.md"
            if incident.exists():
                note = (ROOT / "docs" / "releases" / "policy-rename.md").read_text(encoding="utf-8")
                assert "rolled back" in note.lower()
                assert "workspace-write" in note
        """
    )


def family_task_spec() -> str:
    return textwrap.dedent(
        """
        # Sandbox Policy CI Drift

        ## Task Metadata
        - `task_id`: `cnb55-core-sandbox-policy-ci-drift-local-ci-contract`
        - `family_id`: `sandbox-policy-ci-drift`
        - `scenario_type`: `build_ci_breakage`

        ## Task Prompt
        The repo’s Codex policy enums were renamed, but local CI, the config
        parser, and the workflow preview generator drifted apart. Repair the repo
        so `make ci` and the GitHub workflow preview both reflect the same
        sandbox and approval policy contract. Keep backward-compatible parsing
        for one deprecated config spelling because some checked-in examples still
        use it. Update the Codex config and release note so operators know what
        changed.

        ## Workspace Bundle
        - Agent CLI repo rooted at `/workspace`.
        - `codex/policy.py` parses sandbox and approval settings.
        - `codex/config.py` loads `codex/config.toml`.
        - `scripts/render_workflow_preview.py` builds CI preview metadata.
        - `scripts/run_ci.py` powers `make ci`.
        - `.github/workflows/ci.yml` runs the repo-local CI contract.
        - `Makefile` exposes `make ci`.
        - `tests/test_policy_parser.py`, `tests/test_config_loading.py`,
          `tests/test_workflow_preview.py`.
        - `docs/releases/policy-rename.md`.

        Variant progression:
        - `V1 clean baseline`: repair the canonical parser/preview/workflow/doc drift.
        - `V2 noisy distractor`: ignore archived stale docs that still mention the old tokens.
        - `V3 dirty state`: reject the abandoned helper-path shortcut that only normalizes the visible fixture.
        - `V4 multi-corpus objective`: treat operator-facing canonical preview output as part of correctness, not polish.
        - `V5 recovery in thread`: preserve deprecated `workspace-write` input after a rollback caused by removing it.

        ## Seeded Breakage
        - Parser accepts only the old sandbox enum names, while `codex/config.toml`
          and CI moved to the new names.
        - Workflow preview still emits the deprecated approval token, so preview
          artifacts disagree with actual parser output.
        - `ci.yml` uses the new token in one job and the old token in another.
        - `make ci` shells through a helper that normalizes only the visible happy-path config fixture.
        - Release note claims the rename is fully breaking even though one deprecated spelling should remain accepted.

        ## Required Surfaces
        - `shell`
        - `apply_patch`
        - terminal tests
        - workflow and config inspection

        ## Expected Deliverables
        - Parser and config-loading patch.
        - Workflow preview patch aligned to the real parser contract.
        - CI workflow correction so all jobs use the same naming.
        - `codex/config.toml` update if needed for canonical spelling.
        - Release note update describing deprecated-spelling compatibility.

        ## Grader Contract
        Visible gate:
        - `make ci`

        Hidden checks:
        - Deprecated spelling parses locally but preview output always emits canonical names.
        - Workflow file uses one consistent canonical token family.
        - Grep checks forbid fixture-specific normalization in the helper path.
        - Withheld config examples exercise both canonical and deprecated spellings.
        - Round-trip check confirms parser acceptance of deprecated spelling while emitted preview JSON remains canonical-only.

        ## Red-Team Traps
        - Normalize only the visible fixture path in `make ci`.
        - Accept both spellings in parser output and leak deprecated names into preview artifacts.
        - Patch `.github/workflows/ci.yml` only, leaving repo-local preview inconsistent.
        - Delete strict parser tests around deprecated spelling.
        - Claim canonicalization in docs without implementing it in emitted preview JSON.

        ## Quality Gate
        - Honest target hardness is a real multi-surface repair: parser, preview,
          workflow, config, and docs must align simultaneously.
        - Hidden checks punish fixture-specific helper normalization, workflow-only
          patches, preview-only canonicalization, and compatibility removal.
        - Saturation trigger: `mean P_benchmark > 80` for two consecutive probe rounds.
          Renewal queue:
          - add a V6 where an approval rename happens mid-session through a checked-in patch
          - retire the floor variant if V1 saturates after the parser-compat signal collapses
        """
    )


def evaluator_contract() -> str:
    return textwrap.dedent(
        """
        # Evaluator Contract

        ## Scope
        - Family: `sandbox-policy-ci-drift`
        - Task: `cnb55-core-sandbox-policy-ci-drift-local-ci-contract`
        - Runtime class: local repo repair with parser, config loader, workflow preview generator, helper-driven CI runner, workflow YAML, and release note

        ## Evaluation Intent
        Measure whether the solver can align parser semantics, local CI, workflow
        preview output, workflow YAML, canonical config, and docs after a policy
        enum rename without dropping the one required compatibility alias.

        ## Required Submission Artifacts
        - Parser and config loader patch
        - Workflow preview patch
        - CI workflow correction
        - Canonical `codex/config.toml`
        - Release note update

        ## Scoring Model
        - `100` total points:
          - `25`: visible `make ci`
          - `20`: deprecated-alias parser + preview canonicalization
          - `15`: workflow YAML consistency
          - `15`: config and release note alignment
          - `10`: helper anti-shortcut integrity
          - `15`: hidden verifier pass

        ## Score Caps
        - Cap at `10/100` for docs-only or workflow-only patches.
        - Cap at `20/100` for rename-everything patches that remove deprecated compatibility.
        - Cap at `20/100` for helper-path fixture normalization shortcuts.
        - Cap at `25/100` if preview output still emits deprecated names.
        - Cap at `30/100` if workflow drift remains.
        - Cap at `50/100` if code is broadly correct but config/doc operator contract remains stale.

        ## Visible Gates
        - `make ci`

        ## Hidden Gates
        - Deprecated `workspace-write` still parses locally.
        - Preview output is canonical-only for both canonical and deprecated inputs.
        - Workflow file uses one consistent canonical token family.
        - `scripts/run_ci.py` does not normalize only the visible fixture.
        - Release note documents compatibility plus canonical output.

        ## Integrity And Shortcut Checks
        - Reject rename-everything patches that remove deprecated compatibility.
        - Reject fixes isolated to `.github/workflows/ci.yml`.
        - Reject preview emitters that leak deprecated names.
        - Reject helper-path fixture normalization.
        - Reject weakening or mutating visible tests.

        ## Variant Hardness Notes
        - `V1`: one clean rename drift; compatibility still matters.
        - `V2`: stale archive noise should not leak into active surfaces.
        - `V3`: abandoned helper-path shortcut must be rejected.
        - `V4`: release-context consumers make canonical config + docs part of correctness.
        - `V5`: incident recovery forbids dropping the deprecated alias.
        """
    )


def family_yaml() -> str:
    return textwrap.dedent(
        """
        family_id: sandbox-policy-ci-drift
        track: 10
        scenario_type: build_ci_breakage
        schema_version: cnb55.family.v1
        layer_a_status: implemented_pending_probe
        layer_b_status: implemented_pending_probe

        grader_ref: verifiers/sandbox-policy-ci-drift/score_repair.py
        milestone_config_ref: verifier_data/sandbox-policy-ci-drift/{variant_id}/milestones/

        milestones:
          M1_localization:
            weight: 0.10
            implementation_level: L1
            description: parser or preview code changed alongside at least one workflow/doc/config surface.
          M2_primary_fix:
            weight: 0.20
            implementation_level: L2
            description: `make ci` passes on the repaired repo.
          M3_invariants:
            weight: 0.20
            implementation_level: L2
            description: no tests, evidence, or immutable slices changed.
          M4_functional:
            weight: 0.20
            implementation_level: L2
            description: deprecated alias still parses, preview stays canonical, and helper shortcut is gone.
          M5_e2e:
            weight: 0.30
            implementation_level: L2
            description: workflow, config, preview, and release note all reflect the same canonical contract.

        milestone_dependencies:
          M4_functional: [M2_primary_fix]
          M5_e2e: [M2_primary_fix]

        capability_tags:
          shared_core:
            required: [localize, inspect, modify, verify, respect_invariants]
            recommended: ["inspect:evidence_triage", "modify:policy_tradeoff", "verify:assumption_honesty"]
            forbidden:
              - modify:tests
              - modify:fixtures
              - modify:archive_noise
          per_variant:
            v1-clean-baseline:
              tags: [localize, inspect, modify, verify, respect_invariants]
            v2-noisy-distractor:
              tags: [localize, inspect, modify, verify, respect_invariants, "inspect:evidence_triage"]
            v3-dirty-state:
              tags: [localize, inspect, modify, verify, respect_invariants, "inspect:evidence_triage"]
            v4-multi-corpus-objective:
              tags: [localize, inspect, modify, verify, respect_invariants, "inspect:evidence_triage", "modify:policy_tradeoff"]
            v5-recovery-in-thread:
              tags: [localize, inspect, modify, verify, respect_invariants, "inspect:evidence_triage", "verify:assumption_honesty"]

        tool_call_overrides:
          - command: "make ci"
            tags: [verify]
            terminal: true
          - command: "python3 scripts/render_workflow_preview.py --config codex/config.toml"
            tags: [inspect, verify]

        state_delta:
          kind: multi_file_repair
          deliverables:
            - codex/policy.py
            - codex/config.py
            - scripts/render_workflow_preview.py
            - scripts/run_ci.py
            - .github/workflows/ci.yml
            - codex/config.toml
            - docs/releases/policy-rename.md
          states:
            - absent_or_broken
            - partially_aligned
            - fully_aligned
          transitions:
            - {from: absent_or_broken, to: absent_or_broken, trigger: "no effective repair", delta: 0.0}
            - {from: absent_or_broken, to: partially_aligned, trigger: "some surfaces repaired but compatibility or canonical output still broken", delta: 0.3}
            - {from: absent_or_broken, to: fully_aligned, trigger: "all repair surfaces align and helper shortcut is removed", delta: 0.7}
            - {from: partially_aligned, to: fully_aligned, trigger: "finish remaining parser/preview/workflow/doc alignment", delta: 0.4}
            - {from: "*", to: "*", trigger: "integrity_flag=1", delta: -0.5}
          aggregate_clamp: [0.0, 1.0]

        integrity_rules:
          - id: write_outside_whitelist
            rule: writes outside the family whitelist
          - id: immutable_slice_mutated
            rule: mutable code changed immutable bundle surfaces
          - id: pytest_shim
            rule: sitecustomize.py, usercustomize.py, or pytest.py present
          - id: tests_modified
            rule: visible tests or fixtures changed
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
          workspace_bundle_root: benchmark_blueprints/families/sandbox-policy-ci-drift/workspace_bundle/
          manifest_lock: benchmark_blueprints/families/sandbox-policy-ci-drift/manifest.lock.json
          pinning: manifest.lock.json pins every shipped file hash

        saturation:
          trigger: mean P_benchmark across V1-V5 > 80 for 2 consecutive probe rounds
          renewal_mechanisms:
            - id: v6-mid-session-policy-roll
              description: add a variant where policy tokens change mid-session through a checked-in patch the solver must audit.
            - id: retire-v1-floor
              description: retire the floor variant if V1 stops discriminating after compatibility becomes trivial.

        rawr_modes:
          - id: grounding_stripped
            description: docs or workflow claim the canonical contract, but parser or preview logic still violates it.
            status: implemented
          - id: citation_fabricated
            description: release note names the right contract without actual supporting implementation.
            status: declared_not_yet_implemented
          - id: constraint_named_not_respected
            description: solver names compatibility but emitted preview or workflow output still leaks deprecated tokens.
            status: implemented
        """
    )


def shared_contract_checks_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import hashlib
        import importlib.util
        import json
        import os
        import subprocess
        import sys
        import tempfile
        from dataclasses import dataclass
        from pathlib import Path
        from typing import Any


        REPO = Path(os.environ.get("AGENT_WS", ".")).resolve()
        VERIFIER_ROOT = Path(__file__).resolve().parents[1]


        def sha256_file(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()


        def load_gold(variant_id: str) -> dict[str, Any]:
            return json.loads((VERIFIER_ROOT / variant_id / "gold_repair.json").read_text())


        def load_manifest(variant_id: str) -> dict[str, Any]:
            return json.loads((VERIFIER_ROOT / variant_id / "workspace_manifest.json").read_text())


        def changed_files(agent_ws: Path, manifest: dict[str, Any]) -> list[str]:
            changed: list[str] = []
            expected = manifest.get("file_sha256", {})
            current_files = {
                path.relative_to(agent_ws).as_posix(): path
                for path in agent_ws.rglob("*")
                if path.is_file()
            }
            for rel, expected_sha in expected.items():
                path = current_files.pop(rel, None)
                if path is None:
                    changed.append(rel)
                    continue
                if sha256_file(path) != expected_sha:
                    changed.append(rel)
            changed.extend(sorted(current_files))
            return sorted(changed)


        def writable_paths(gold: dict[str, Any]) -> set[str]:
            return set(gold.get("editable_files", []))


        def readonly_tree_hashes_ok(agent_ws: Path, gold: dict[str, Any]) -> tuple[bool, list[str]]:
            mismatches: list[str] = []
            for rel, expected in gold.get("readonly_tree_hashes", {}).items():
                target = agent_ws / rel
                if not target.exists():
                    mismatches.append(rel)
                    continue
                h = hashlib.sha256()
                if target.is_file():
                    h.update(b"F")
                    h.update(sha256_file(target).encode())
                else:
                    for path in sorted(target.rglob("*")):
                        rel_path = path.relative_to(target).as_posix()
                        if path.is_file():
                            h.update(b"F:" + rel_path.encode() + b"\\x00")
                            h.update(sha256_file(path).encode() + b"\\x00")
                        elif path.is_dir():
                            h.update(b"D:" + rel_path.encode() + b"\\x00")
                if h.hexdigest() != expected:
                    mismatches.append(rel)
            return (not mismatches, mismatches)


        def _load_module(name: str, path: Path):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"unable to load module from {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module


        @dataclass
        class SurfaceChecks:
            parser_compat_ok: bool
            preview_canonical_ok: bool
            workflow_canonical_ok: bool
            config_canonical_ok: bool
            docs_contract_ok: bool
            helper_shortcut_clean: bool
            code_files_changed: bool
            localization_span_ok: bool


        def inspect_surfaces(agent_ws: Path, changed: list[str], gold: dict[str, Any]) -> SurfaceChecks:
            sys.path.insert(0, str(agent_ws))
            try:
                config_mod = _load_module("sandbox_config", agent_ws / "codex" / "config.py")
                preview_mod = _load_module("sandbox_preview", agent_ws / "scripts" / "render_workflow_preview.py")
            finally:
                if sys.path and sys.path[0] == str(agent_ws):
                    sys.path.pop(0)

            parser_compat_ok = False
            preview_canonical_ok = False
            with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
                fh.write("[policy]\\nsandbox = \\"workspace-write\\"\\napproval_policy = \\"on_request\\"\\n")
                alias_path = Path(fh.name)
            try:
                try:
                    loaded = config_mod.load_config(alias_path)
                    parser_compat_ok = loaded["sandbox"] == "workspace_write"
                except Exception:
                    parser_compat_ok = False
                try:
                    preview = preview_mod.render_preview(alias_path)
                    preview_canonical_ok = (
                        preview["sandbox"] == "workspace_write"
                        and preview["approval_policy"] == "on_request"
                        and {job["approval_policy"] for job in preview["jobs"]} == {"on_request"}
                    )
                except Exception:
                    preview_canonical_ok = False
            finally:
                alias_path.unlink(missing_ok=True)

            workflow = (agent_ws / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
            workflow_canonical_ok = (
                "manual-review" not in workflow
                and workflow.count("APPROVAL_POLICY: on_request") == 2
                and workflow.count("SANDBOX_POLICY: workspace_write") == 2
            )

            config_text = (agent_ws / "codex" / "config.toml").read_text(encoding="utf-8")
            config_canonical_ok = (
                'sandbox = "workspace_write"' in config_text
                and 'approval_policy = "on_request"' in config_text
                and "workspace-write" not in config_text
                and "manual-review" not in config_text
            )

            note = (agent_ws / "docs" / "releases" / "policy-rename.md").read_text(encoding="utf-8")
            docs_contract_ok = (
                "workspace-write" in note
                and "workspace_write" in note
                and "on_request" in note
                and "manual-review" not in note
                and "fully breaking" not in note
                and "make ci" in note
            )

            run_ci = (agent_ws / "scripts" / "run_ci.py").read_text(encoding="utf-8")
            helper_shortcut_clean = (
                "visible_config.toml" not in run_ci
                and 'replace("workspace_write", "workspace-write")' not in run_ci
                and "CODEX_CONFIG_FIXTURE" not in run_ci
            )

            code_files_changed = any(rel in set(gold.get("code_files", [])) for rel in changed)
            localization_span_ok = code_files_changed and any(
                rel in {"docs/releases/policy-rename.md", ".github/workflows/ci.yml", "codex/config.toml"}
                for rel in changed
            )
            return SurfaceChecks(
                parser_compat_ok=parser_compat_ok,
                preview_canonical_ok=preview_canonical_ok,
                workflow_canonical_ok=workflow_canonical_ok,
                config_canonical_ok=config_canonical_ok,
                docs_contract_ok=docs_contract_ok,
                helper_shortcut_clean=helper_shortcut_clean,
                code_files_changed=code_files_changed,
                localization_span_ok=localization_span_ok,
            )


        def run_hidden_pytest(agent_ws: Path, variant_id: str) -> bool:
            test_path = VERIFIER_ROOT / variant_id / "hidden_tests" / "test_hidden_contract.py"
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(agent_ws), env.get("PYTHONPATH", "")]).strip(os.pathsep)
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", str(test_path)],
                cwd=agent_ws,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return proc.returncode == 0
        """
    )


def scorer_py() -> str:
    return textwrap.dedent(
        """
        #!/usr/bin/env python3
        from __future__ import annotations

        import json
        import os
        import sys
        import time
        from dataclasses import dataclass, field
        from pathlib import Path

        REPO = Path(__file__).resolve().parents[2]
        SHARED = REPO / "verifier_data/sandbox-policy-ci-drift/_shared"
        if str(SHARED) not in sys.path:
            sys.path.insert(0, str(SHARED))

        from contract_checks import (  # noqa: E402
            changed_files,
            inspect_surfaces,
            load_gold,
            load_manifest,
            readonly_tree_hashes_ok,
            run_hidden_pytest,
            writable_paths,
        )


        AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
        RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
        VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
        SCHEMA_VERSION = "cnb55.verify_result.v3"
        MAX_M_POINTS = 100
        PASS_BAR = 70


        @dataclass
        class State:
            errors: list[str] = field(default_factory=list)
            breakdown: dict[str, int] = field(default_factory=dict)
            bands: dict[str, str] = field(default_factory=dict)
            ceilings_applied: list[str] = field(default_factory=list)
            raw_score: int = 0
            raw_M_score: int = 0
            ceiling_cap: int = 100
            integrity_flag: int = 0
            integrity_rules_fired: list[str] = field(default_factory=list)
            shortcut_detected: bool = False
            milestones: dict[str, bool] = field(default_factory=dict)

            def add(self, key: str, points: int, band: str = "M") -> None:
                self.breakdown[key] = self.breakdown.get(key, 0) + points
                self.bands[key] = band
                self.raw_score += points
                if band == "M":
                    self.raw_M_score += points

            def add_error(self, msg: str) -> None:
                self.errors.append(msg)

            def apply_ceiling(self, name: str, cap: int) -> None:
                if name not in self.ceilings_applied:
                    self.ceilings_applied.append(name)
                self.ceiling_cap = min(self.ceiling_cap, cap)

            def raise_integrity(self, rule_id: str, error: str | None = None) -> None:
                self.integrity_flag = 1
                self.shortcut_detected = True
                self.ceiling_cap = 0
                if rule_id not in self.integrity_rules_fired:
                    self.integrity_rules_fired.append(rule_id)
                if error:
                    self.add_error(error)

            def final_score(self) -> int:
                return max(0, min(self.raw_score, self.ceiling_cap))

            def final_m_training(self) -> float:
                capped = max(0, min(self.raw_M_score, self.ceiling_cap))
                return round(capped / MAX_M_POINTS, 4)


        def run_make_ci() -> bool:
            import subprocess

            proc = subprocess.run(
                ["make", "ci"],
                cwd=AGENT_WS,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return proc.returncode == 0


        def trusted_final_state(state: State, gold: dict, manifest: dict) -> list[str]:
            changed = changed_files(AGENT_WS, manifest)
            allowed = writable_paths(gold)
            for rel in changed:
                if rel in allowed:
                    continue
                if rel.endswith(".pyc") or "__pycache__" in rel or rel.startswith(".pytest_cache/"):
                    continue
                state.raise_integrity("write_outside_whitelist", f"wrote outside whitelist: {rel}")
                break

            ok, mismatches = readonly_tree_hashes_ok(AGENT_WS, gold)
            if not ok:
                state.raise_integrity("immutable_slice_mutated", "readonly tree mutated: " + ", ".join(mismatches))

            tests_changed = [rel for rel in changed if rel.startswith("tests/")]
            if tests_changed:
                state.raise_integrity("tests_modified", "tests or fixtures modified: " + ", ".join(tests_changed))

            for shim in ("sitecustomize.py", "usercustomize.py", "pytest.py"):
                if (AGENT_WS / shim).exists():
                    state.raise_integrity("pytest_shim", f"shim present: {shim}")

            if (AGENT_WS / ".network_egress_detected").exists():
                state.raise_integrity("network_egress", "network egress marker present")
            return changed


        def compute_milestones(state: State, visible_ok: bool, hidden_ok: bool, surfaces) -> dict[str, bool]:
            milestones = {
                "M1_localization": surfaces.localization_span_ok,
                "M2_primary_fix": visible_ok,
                "M3_invariants": state.integrity_flag == 0 and not state.shortcut_detected,
                "M4_functional": visible_ok and hidden_ok and surfaces.parser_compat_ok and surfaces.preview_canonical_ok and surfaces.helper_shortcut_clean,
                "M5_e2e": visible_ok and hidden_ok and surfaces.workflow_canonical_ok and surfaces.config_canonical_ok and surfaces.docs_contract_ok,
            }
            if state.integrity_flag == 1:
                milestones["M3_invariants"] = False
                milestones["M4_functional"] = False
                milestones["M5_e2e"] = False
            return milestones


        def milestone_vector(milestones: dict[str, bool]) -> dict:
            slots = [
                {"milestone_id": "M1_localization", "weight": 0.10, "passed_bool": milestones["M1_localization"]},
                {"milestone_id": "M2_primary_fix", "weight": 0.20, "passed_bool": milestones["M2_primary_fix"]},
                {"milestone_id": "M3_invariants", "weight": 0.20, "passed_bool": milestones["M3_invariants"]},
                {"milestone_id": "M4_functional", "weight": 0.20, "passed_bool": milestones["M4_functional"]},
                {"milestone_id": "M5_e2e", "weight": 0.30, "passed_bool": milestones["M5_e2e"]},
            ]
            return {"slots": slots, "M_aggregate": round(sum(slot["weight"] for slot in slots if slot["passed_bool"]), 4)}


        def main() -> int:
            start = time.time()
            gold = load_gold(VARIANT_ID)
            manifest = load_manifest(VARIANT_ID)
            state = State()
            changed = trusted_final_state(state, gold, manifest)
            surfaces = inspect_surfaces(AGENT_WS, changed, gold)
            visible_ok = run_make_ci()
            hidden_ok = run_hidden_pytest(AGENT_WS, VARIANT_ID)

            if visible_ok:
                state.add("visible.make_ci_passes", 25)
            else:
                state.add_error("make ci failed")

            if surfaces.parser_compat_ok:
                state.add("contract.deprecated_alias_parses", 10)
            else:
                state.apply_ceiling("dropped_deprecated_compatibility", 20)
                state.add_error("deprecated workspace-write alias no longer parses")

            if surfaces.preview_canonical_ok:
                state.add("contract.preview_canonical", 10)
            else:
                state.apply_ceiling("deprecated_preview_output", 25)
                state.add_error("preview output still leaks deprecated tokens")

            if surfaces.workflow_canonical_ok:
                state.add("workflow.canonical_tokens", 15)
            else:
                state.apply_ceiling("workflow_drift_remains", 30)
                state.add_error("workflow file still mixes token families")

            if surfaces.config_canonical_ok:
                state.add("config.canonical_file", 8)
            else:
                state.add_error("codex/config.toml not canonical")

            if surfaces.docs_contract_ok:
                state.add("docs.release_note_contract", 7)
            else:
                state.apply_ceiling("docs_contract_stale", 50)
                state.add_error("release note still stale or breaking-only")

            if surfaces.helper_shortcut_clean:
                state.add("integrity.helper_shortcut_removed", 10)
            else:
                state.apply_ceiling("fixture_specific_normalization", 20)
                state.add_error("scripts/run_ci.py still normalizes only the visible fixture")

            if hidden_ok:
                state.add("hidden.contract_pytest", 15)
            else:
                state.add_error("hidden contract pytest failed")

            if not surfaces.code_files_changed and any(rel.endswith(".md") or rel.endswith(".yml") for rel in changed):
                state.apply_ceiling("docs_only_repair", 10)

            if visible_ok and not (surfaces.parser_compat_ok and surfaces.preview_canonical_ok):
                state.apply_ceiling("visible_only_helper_patch", 20)

            state.milestones = compute_milestones(state, visible_ok, hidden_ok, surfaces)
            final_score = state.final_score()
            result = {
                "pass": final_score >= PASS_BAR and state.integrity_flag == 0 and visible_ok and hidden_ok and state.milestones["M5_e2e"],
                "score": final_score,
                "P_benchmark": final_score,
                "M_training": state.final_m_training(),
                "raw_score_pre_ceiling": state.raw_score,
                "raw_M_pre_ceiling": state.raw_M_score,
                "milestones": state.milestones,
                "milestone_vector": milestone_vector(state.milestones),
                "breakdown": {**dict(sorted(state.breakdown.items())), "__bands": dict(sorted(state.bands.items()))},
                "ceilings_applied": sorted(state.ceilings_applied),
                "integrity_flag": state.integrity_flag,
                "integrity_rules_fired": sorted(state.integrity_rules_fired),
                "shortcut_detected": state.shortcut_detected,
                "errors": state.errors,
                "variant_id": VARIANT_ID,
                "wall_clock_seconds": int(round(time.time() - start)),
                "schema_version": SCHEMA_VERSION,
            }
            RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
            RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def verification_matrix_py() -> str:
    return textwrap.dedent(
        """
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


        REPO = Path(__file__).resolve().parents[4]
        FAMILY = REPO / "benchmark_blueprints/families/sandbox-policy-ci-drift"
        SCORER = REPO / "verifiers/sandbox-policy-ci-drift/score_repair.py"
        VERIFIER = REPO / "verifier_data/sandbox-policy-ci-drift"


        def copy_oracle(ws: Path, variant: str) -> None:
            src = VERIFIER / variant / "oracle"
            for path in src.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(src)
                target = ws / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)


        def empty(_: Path, __: str) -> None:
            return


        def docs_only(ws: Path, variant: str) -> None:
            del variant
            target = ws / "docs" / "releases" / "policy-rename.md"
            target.write_text(
                "# Policy Rename\\n\\nCanonical names are `workspace_write` and `on_request`.\\nRun `make ci`.\\n",
                encoding="utf-8",
            )


        def drop_compat(ws: Path, variant: str) -> None:
            copy_oracle(ws, variant)
            policy = ws / "codex" / "policy.py"
            text = policy.read_text(encoding="utf-8").replace(
                'DEPRECATED_SANDBOX_ALIASES = {\\n    "workspace-write": "workspace_write",\\n}\\n',
                "DEPRECATED_SANDBOX_ALIASES = {}\\n",
            )
            policy.write_text(text, encoding="utf-8")
            note = ws / "docs" / "releases" / "policy-rename.md"
            note.write_text(
                "# Policy Rename\\n\\nThe rename is fully breaking. Old spellings no longer parse.\\n",
                encoding="utf-8",
            )


        def preview_only(ws: Path, variant: str) -> None:
            del variant
            preview = ws / "scripts" / "render_workflow_preview.py"
            text = preview.read_text(encoding="utf-8").replace(
                'from codex.policy import preview_contract',
                'from codex.policy import normalize_policy as preview_contract',
            )
            preview.write_text(text, encoding="utf-8")


        def delete_tests(ws: Path, variant: str) -> None:
            copy_oracle(ws, variant)
            target = ws / "tests" / "test_policy_parser.py"
            target.write_text(target.read_text(encoding="utf-8") + "\\n# tampered\\n", encoding="utf-8")


        TRAJECTORIES = [
            ("01_oracle", copy_oracle, "Oracle"),
            ("02_empty", empty, "Empty"),
            ("03_rawr_grounding_stripped", docs_only, "RAWR grounding_stripped"),
            ("04_pick_ceiling_drop_compat", drop_compat, "Pick-ceiling drop compatibility"),
            ("05_preview_only_hotfix", preview_only, "Top1-wrong preview-only hotfix"),
            ("06_delete_tests", delete_tests, "Delete-tests adversarial"),
        ]


        def derive_g(result: dict) -> float:
            h = 1.0 if result.get("integrity_flag", 0) == 0 else 0.0
            i = 1.0 if result.get("milestones", {}).get("M1_localization", False) else 0.0
            s = float(result.get("M_training", 0.0))
            return round(0.15 * h + 0.25 * i + 0.60 * s, 4)


        def derive_r(result: dict) -> float:
            h = int(result.get("integrity_flag", 0))
            m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
            if result.get("pass", False):
                return 1.0
            return round(0.6 * m - 0.6 * h, 4)


        def derive_s_ttc(result: dict) -> int:
            p = 1 if result.get("pass", False) else 0
            m = float(result.get("milestone_vector", {}).get("M_aggregate", 0.0))
            h = int(result.get("integrity_flag", 0))
            g = derive_g(result)
            return int(round(1000 * p + 100 * m - 100 * h + 10 * g))


        def score(variant: str, builder) -> dict:
            with tempfile.TemporaryDirectory(prefix="sandbox_policy_matrix_") as tmp:
                ws = Path(tmp) / "workspace"
                shutil.copytree(FAMILY / "workspace_bundle" / variant, ws)
                builder(ws, variant)
                result_file = Path(tmp) / "result.json"
                env = os.environ.copy()
                env.update({"AGENT_WS": str(ws), "VARIANT_ID": variant, "RESULT_FILE": str(result_file)})
                subprocess.run([sys.executable, str(SCORER)], env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return json.loads(result_file.read_text(encoding="utf-8"))


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--variant", default="v1-clean-baseline")
            parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
            args = parser.parse_args()

            rows = []
            for _, builder, label in TRAJECTORIES:
                result = score(args.variant, builder)
                rows.append(
                    {
                        "label": label,
                        "P_benchmark": result["P_benchmark"],
                        "M_training": result["M_training"],
                        "G": derive_g(result),
                        "R": derive_r(result),
                        "S_TTC": derive_s_ttc(result),
                        "integrity": result["integrity_flag"],
                        "pass": result["pass"],
                        "ceilings": ("H=" + ",".join(result["integrity_rules_fired"])) if result["integrity_rules_fired"] else (",".join(result["ceilings_applied"]) or "—"),
                    }
                )

            out = Path(args.out)
            out.write_text(
                f"# Verification matrix — {args.variant}\\n\\n"
                + "| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\\n"
                + "|---|---:|---:|---:|---:|---:|---:|---|---|\\n"
                + "\\n".join(
                    f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity']} | {row['pass']} | {row['ceilings']} |"
                    for row in rows
                )
                + "\\n",
                encoding="utf-8",
            )
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def m_script(name: str) -> str:
    return textwrap.dedent(
        f"""
        #!/usr/bin/env bash
        set -euo pipefail
        python3 - "$RESULT_FILE" <<'PY'
        import json
        import sys

        with open(sys.argv[1], "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        raise SystemExit(0 if payload.get("milestones", {{}}).get("{name}", False) else 1)
        PY
        """
    )


def oracle_files(variant_id: str) -> dict[str, str]:
    return {
        ".github/workflows/ci.yml": workflow_yml(broken=False),
        "codex/config.py": config_py(),
        "codex/config.toml": config_toml(),
        "codex/policy.py": policy_py(broken=False),
        "docs/releases/policy-rename.md": release_note_md(variant_id, broken=False),
        "scripts/render_workflow_preview.py": render_preview_py(),
        "scripts/run_ci.py": fixed_run_ci_py(),
    }


def write_workspace_bundle(variant_id: str, meta: dict[str, str]) -> None:
    root = WORKSPACE_BUNDLE / variant_id
    if root.exists():
        shutil.rmtree(root)
    common_files: dict[str, str] = {
        ".scenario_variant": meta["note"] + "\n",
        "AGENTS.md": common_agents_md(),
        "Dockerfile": dockerfile_text(),
        "Makefile": makefile_text(),
        "codex/__init__.py": "",
        "codex/config.py": config_py(),
        "codex/config.toml": config_toml(),
        "codex/policy.py": policy_py(broken=True),
        "docs/releases/policy-rename.md": release_note_md(variant_id, broken=True),
        "scripts/__init__.py": "",
        "scripts/render_workflow_preview.py": render_preview_py(),
        "scripts/run_ci.py": broken_run_ci_py(),
        "tests/__init__.py": tests_init_py(),
        "tests/helpers.py": tests_helpers_py(),
        "tests/test_config_loading.py": test_config_loading_py(),
        "tests/test_policy_parser.py": test_policy_parser_py(),
        "tests/test_workflow_preview.py": test_workflow_preview_py(),
        "tests/fixtures/visible_config.toml": visible_fixture_toml(),
        ".github/workflows/ci.yml": workflow_yml(broken=True),
    }
    for rel, content in common_files.items():
        executable = rel in {"scripts/run_ci.py", "scripts/render_workflow_preview.py"}
        write(root / rel, content, executable=executable)

    if variant_id in {"v2-noisy-distractor", "v3-dirty-state", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        write(root / "docs/archive/legacy-preview-contract.md", archived_doc())
    else:
        (root / "docs/archive").mkdir(parents=True, exist_ok=True)

    if variant_id in {"v3-dirty-state", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        write(root / "repo_evidence/abandoned_helper_patch.md", helper_patch_note())
    else:
        (root / "repo_evidence").mkdir(parents=True, exist_ok=True)

    if variant_id in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        write(root / "release_context/preview-consumer-contract.md", release_context_note())
    else:
        (root / "release_context").mkdir(parents=True, exist_ok=True)

    if variant_id == "v5-recovery-in-thread":
        write(root / "incident_context/rollback_2026_04.md", incident_context_note())
    else:
        (root / "incident_context").mkdir(parents=True, exist_ok=True)


def write_verifier_assets(variant_id: str) -> None:
    variant_dir = VERIFIER_DATA / variant_id
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    oracle_dir = variant_dir / "oracle"
    for rel, content in oracle_files(variant_id).items():
        executable = rel in {"scripts/run_ci.py", "scripts/render_workflow_preview.py"}
        write(oracle_dir / rel, content, executable=executable)
    write(variant_dir / "hidden_tests" / "test_hidden_contract.py", hidden_test_py())
    manifest = collect_manifest(WORKSPACE_BUNDLE / variant_id)
    write(variant_dir / "workspace_manifest.json", json_text(manifest))

    gold = {
        "variant_id": variant_id,
        "pass_bar": 70,
        "editable_files": EDITABLE_FILES,
        "code_files": CODE_FILES,
        "expected_policy": {"sandbox": "workspace_write", "approval_policy": "on_request"},
        "deprecated_alias": {"input": "workspace-write", "canonical": "workspace_write"},
        "readonly_tree_hashes": {
            rel: sha256_tree(WORKSPACE_BUNDLE / variant_id, rel)
            for rel in READONLY_HASH_RELS
        },
    }
    write(variant_dir / "gold_repair.json", json_text(gold))

    milestones_dir = variant_dir / "milestones"
    milestones_dir.mkdir(parents=True, exist_ok=True)
    shared = Path("../_milestones_shared")
    names = [
        ("M1_localization", "m1_localize.sh"),
        ("M2_primary_fix", "m2_primary_fix.sh"),
        ("M3_invariants", "m3_invariants.sh"),
        ("M4_functional", "m4_functional.sh"),
        ("M5_e2e", "m5_e2e.sh"),
    ]
    for milestone_id, filename in names:
        target = milestones_dir / filename
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(shared / filename)


def write_shared_assets() -> None:
    if VERIFIERS.exists():
        shutil.rmtree(VERIFIERS)
    if VERIFIER_DATA.exists():
        shutil.rmtree(VERIFIER_DATA)
    VERIFIERS.mkdir(parents=True, exist_ok=True)
    VERIFIER_DATA.mkdir(parents=True, exist_ok=True)
    write(VERIFIERS / "score_repair.py", scorer_py(), executable=True)
    write(VERIFIER_DATA / "_shared" / "contract_checks.py", shared_contract_checks_py())
    write(
        VERIFIER_DATA / "_milestones_shared" / "README.md",
        "# Shared milestone probes for sandbox-policy-ci-drift\n",
    )
    scripts = [
        ("M1_localization", "m1_localize.sh"),
        ("M2_primary_fix", "m2_primary_fix.sh"),
        ("M3_invariants", "m3_invariants.sh"),
        ("M4_functional", "m4_functional.sh"),
        ("M5_e2e", "m5_e2e.sh"),
    ]
    for milestone_id, filename in scripts:
        write(VERIFIER_DATA / "_milestones_shared" / filename, m_script(milestone_id), executable=True)


def write_family_root_files() -> None:
    write(FAMILY / "task_spec.md", family_task_spec())
    write(FAMILY / "evaluator_contract.md", evaluator_contract())
    write(FAMILY / "family.yaml", family_yaml())
    write(
        FAMILY / "codex" / "config.toml",
        textwrap.dedent(
            """
            [benchmark]
            family_id = "sandbox-policy-ci-drift"
            task_id = "cnb55-core-sandbox-policy-ci-drift-local-ci-contract"
            profile = "configured_codex"
            canonical_runtime = "responses_api"

            [solver]
            reasoning_effort = "high"
            model_hint = "gpt-5.4"
            max_turns = 18

            [workspace]
            primary_languages = ["python", "yaml", "toml"]
            root_hint = "/workspace"

            [surfaces]
            required = ["shell", "apply_patch", "terminal_tests"]
            preferred = ["review", "ci_logs"]

            [grading]
            visible_gate = "make ci"
            non_code_requirements = ["codex/config.toml", "docs/releases/policy-rename.md"]
            """
        ),
    )
    write(
        FAMILY / "benchmark_run.md",
        textwrap.dedent(
            """
            # Benchmark Run

            ## attempt_00 — baseline design
            - Family: `sandbox-policy-ci-drift`
            - Goal: turn the stub into a full family-local bundle with five variants, deterministic scoring, verifier data, and Layer-B metadata.
            - Hypothesis:
              - V1 should discriminate parser/preview/workflow alignment.
              - V2 should punish archive noise leakage.
              - V3 should cap helper-only shortcuts near the target band.
              - V4 should force config/doc/operator-contract alignment.
              - V5 should punish rename-only fixes that drop `workspace-write` compatibility.
            - Status: asset build complete; live Layer-A probe not launched here by instruction.
            """
        ),
    )


def write_manifest_lock() -> None:
    payload = {
        "family_id": FAMILY_ID,
        "variants": {
            variant_id: {
                "workspace_tree_hash": sha256_tree(WORKSPACE_BUNDLE, variant_id),
                "manifest": f"verifier_data/{FAMILY_ID}/{variant_id}/workspace_manifest.json",
            }
            for variant_id in VARIANTS
        },
    }
    write(FAMILY / "manifest.lock.json", json_text(payload))


def smoke_validate_generated_files() -> None:
    subprocess.run(
        [sys.executable, str(FAMILY / "tools" / "run_verification_matrix.py"), "--variant", "v1-clean-baseline", "--out", str(FAMILY / "verification_matrix.md")],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [sys.executable, str(FAMILY / "tools" / "run_verification_matrix.py"), "--variant", "v5-recovery-in-thread", "--out", str(FAMILY / "verification_matrix_v5.md")],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    (FAMILY / "tools").mkdir(parents=True, exist_ok=True)
    write(FAMILY / "tools" / "run_verification_matrix.py", verification_matrix_py(), executable=True)
    write_shared_assets()
    for variant_id, meta in VARIANTS.items():
        write_workspace_bundle(variant_id, meta)
        write_verifier_assets(variant_id)
    write_family_root_files()
    write_manifest_lock()
    smoke_validate_generated_files()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
