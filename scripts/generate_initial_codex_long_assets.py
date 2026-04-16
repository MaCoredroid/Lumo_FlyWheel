#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import stat
from pathlib import Path
from textwrap import dedent

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_FAMILIES_DIR = ROOT / "scenario_families"
VERIFIERS_DIR = ROOT / "verifiers"
VERIFIER_DATA_DIR = ROOT / "verifier_data"

BASE_IMAGE = "python:3.12-bookworm@sha256:1e034bf5ce1ca754be43d1491516da937c0fa03fc29b97a8b6a5e1ce7cb8bbf3"
BASE_IMAGE_DIGEST = "sha256:1e034bf5ce1ca754be43d1491516da937c0fa03fc29b97a8b6a5e1ce7cb8bbf3"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _chmod_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _sha256_manifest(directory: Path) -> str:
    lines: list[str] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(directory.parent).as_posix()
        lines.append(f"{digest}  ./{rel}")
    return "\n".join(lines) + "\n"


def _reset_output_dirs() -> None:
    for path in (SCENARIO_FAMILIES_DIR, VERIFIERS_DIR, VERIFIER_DATA_DIR):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _dockerfile(smoke_command: str) -> str:
    return dedent(
        f"""
        FROM {BASE_IMAGE}

        COPY repo/ /workspace/
        WORKDIR /workspace

        RUN python -m pip install --no-cache-dir pytest
        RUN set -eux; \\
            set +e; \\
            {smoke_command} >/tmp/build-smoke.log 2>&1; \\
            status=$?; \\
            cat /tmp/build-smoke.log; \\
            test "$status" -ne 0
        """
    ).strip()


def _feature_repo(variant: dict) -> dict[str, str]:
    title = variant["title"]
    slug = variant["slug"]
    records = repr(variant["records"])
    return {
        ".scenario_variant": f"{variant['variant_id']}\n",
        "AGENTS.md": dedent(
            f"""
            {title} needs a Markdown mode for its local report CLI.

            The repo already supports JSON output, but the new tests expect
            `--format markdown` to produce a readable heading plus an owner table.
            Keep the JSON path working, add the Markdown renderer, and update
            `docs/usage.md` so the documented command matches the code.

            Do not remove or rewrite the existing tests.
            """
        ).strip(),
        "pyproject.toml": dedent(
            f"""
            [project]
            name = "{variant['project_name']}"
            version = "0.1.0"
            description = "{title} report renderer"
            requires-python = ">=3.12"
            """
        ).strip(),
        "docs/usage.md": dedent(
            f"""
            # {title}

            Run the local report tool with:

            ```bash
            python -m report_app.cli
            ```

            The CLI currently prints JSON for automation jobs.
            """
        ).strip(),
        "report_app/__init__.py": "",
        "report_app/service.py": dedent(
            f"""
            from __future__ import annotations

            TITLE = "{title}"
            REPORT_SLUG = "{slug}"
            RECORDS = {records}


            def build_sections() -> list[dict[str, object]]:
                return [dict(item) for item in RECORDS]
            """
        ).strip(),
        "report_app/rendering.py": dedent(
            """
            from __future__ import annotations

            import json


            def render_json(title: str, sections: list[dict[str, object]]) -> str:
                return json.dumps({"title": title, "sections": sections}, sort_keys=True)
            """
        ).strip(),
        "report_app/cli.py": dedent(
            """
            from __future__ import annotations

            import argparse

            from report_app.rendering import render_json
            from report_app.service import TITLE, build_sections


            def main(argv: list[str] | None = None) -> str:
                parser = argparse.ArgumentParser()
                parser.add_argument("--format", choices=["json"], default="json")
                args = parser.parse_args(argv)
                sections = build_sections()
                if args.format != "json":
                    raise ValueError(f"unsupported format: {args.format}")
                return render_json(TITLE, sections)


            if __name__ == "__main__":
                print(main())
            """
        ).strip(),
        "tests/test_cli.py": dedent(
            f"""
            from __future__ import annotations

            import json

            from report_app.cli import main
            from report_app.service import TITLE


            def test_json_output_is_still_supported() -> None:
                payload = json.loads(main([]))
                assert payload["title"] == TITLE
                assert payload["sections"][0]["owner"]


            def test_markdown_output_renders_heading_and_owner_table() -> None:
                output = main(["--format", "markdown"])
                assert output.startswith(f"# {{TITLE}}")
                assert "| Owner |" in output
                assert "{variant['records'][0]['owner']}" in output
            """
        ).strip(),
        "tests/test_docs.py": dedent(
            """
            from __future__ import annotations

            from pathlib import Path


            def test_usage_doc_mentions_markdown_mode() -> None:
                usage = Path("docs/usage.md").read_text(encoding="utf-8")
                assert "--format markdown" in usage
                assert "Markdown" in usage
            """
        ).strip(),
    }


def _migration_repo(variant: dict) -> dict[str, str]:
    sample = repr(variant["sample"])
    return {
        ".scenario_variant": f"{variant['variant_id']}\n",
        "AGENTS.md": dedent(
            f"""
            The {variant['domain']} rules flow was updated to the RulePlan v2 API.

            Tests now expect the repo to use `build_rule_plan()` from
            `norm_app.rules_v2` instead of the removed legacy helpers. Update the
            assembler, router, and CLI preview path so the v2 dataclass is used
            consistently and the deprecated import path disappears from the codebase.

            Keep the existing behavior the tests describe; do not remove the tests.
            """
        ).strip(),
        "pyproject.toml": dedent(
            f"""
            [project]
            name = "{variant['project_name']}"
            version = "0.1.0"
            description = "{variant['domain']} normalization flow"
            requires-python = ">=3.12"
            """
        ).strip(),
        "norm_app/__init__.py": "",
        "norm_app/legacy_rules.py": dedent(
            """
            from __future__ import annotations


            def build_plan(*_args, **_kwargs):
                raise RuntimeError("LegacyRuleSet was removed; migrate to norm_app.rules_v2")
            """
        ).strip(),
        "norm_app/rules_v2.py": dedent(
            """
            from __future__ import annotations

            from dataclasses import dataclass


            @dataclass(frozen=True)
            class RulePlan:
                slug: str
                route_bucket: str
                owner: str


            def build_rule_plan(title: str, owner: str, region: str) -> RulePlan:
                slug = title.strip().lower().replace(" ", "-")
                return RulePlan(slug=slug, route_bucket=f"{region}:{owner}", owner=owner)
            """
        ).strip(),
        "norm_app/assembler.py": dedent(
            """
            from __future__ import annotations

            from norm_app.legacy_rules import build_plan


            def compile_payload(record: dict[str, str]) -> dict[str, str]:
                plan = build_plan(record["title"], record["owner"], record["region"])
                return {"slug": plan.slug, "route_bucket": plan.route_bucket}
            """
        ).strip(),
        "norm_app/router.py": dedent(
            """
            from __future__ import annotations

            from norm_app.legacy_rules import build_plan


            def route_for(record: dict[str, str]) -> str:
                plan = build_plan(record["title"], record["owner"], record["region"])
                return f"{plan.route_bucket}:{plan.slug}"
            """
        ).strip(),
        "norm_app/cli.py": dedent(
            f"""
            from __future__ import annotations

            from norm_app.assembler import compile_payload
            from norm_app.router import route_for


            SAMPLE = {sample}


            def preview() -> dict[str, str]:
                compiled = compile_payload(SAMPLE)
                compiled["route"] = route_for(SAMPLE)
                return compiled
            """
        ).strip(),
        "tests/test_preview.py": dedent(
            f"""
            from __future__ import annotations

            from norm_app.cli import preview


            def test_preview_uses_ruleplan_v2_shape() -> None:
                payload = preview()
                assert payload["slug"] == "{variant['expected_slug']}"
                assert payload["route_bucket"] == "{variant['expected_bucket']}"
                assert payload["route"].startswith("{variant['expected_bucket']}")
            """
        ).strip(),
        "tests/test_contract.py": dedent(
            """
            from __future__ import annotations

            from pathlib import Path


            def test_repo_no_longer_relies_on_removed_legacy_api() -> None:
                source = Path("norm_app/assembler.py").read_text(encoding="utf-8")
                assert "build_rule_plan" in source
                assert "legacy_rules" not in source
            """
        ).strip(),
    }


def _build_repo(variant: dict) -> dict[str, str]:
    checklist = repr(variant["checklist"])
    return {
        ".scenario_variant": f"{variant['variant_id']}\n",
        "AGENTS.md": dedent(
            f"""
            Local CI for the {variant['domain']} service drifted after the package rename.

            `make ci` is the contract for this repo, and it should succeed again without
            changing the service behavior. Bring `pyproject.toml`, the workflow file, and
            any helper scripts back in sync with the current package name `ci_app`.

            Do not delete the CI checks or rewrite the tests to avoid the failure.
            """
        ).strip(),
        "pyproject.toml": dedent(
            f"""
            [project]
            name = "{variant['project_name']}"
            version = "0.1.0"
            description = "{variant['domain']} CI smoke project"
            requires-python = ">=3.12"

            [tool.lumo_ci]
            package = "{variant['legacy_package']}"
            workflow = ".github/workflows/ci.yml"
            """
        ).strip(),
        "Makefile": dedent(
            """
            PYTHON ?= python

            .PHONY: ci

            ci:
            	$(PYTHON) scripts/run_ci.py
            """
        ).strip(),
        ".github/workflows/ci.yml": dedent(
            f"""
            name: ci

            on:
              push:
              pull_request:

            jobs:
              test:
                runs-on: ubuntu-latest
                steps:
                  - uses: actions/checkout@v4
                  - uses: actions/setup-python@v5
                    with:
                      python-version: "3.12"
                  - run: python -m pip install pytest
                  - run: python scripts/run_ci.py --package {variant['legacy_package']}
            """
        ).strip(),
        "scripts/run_ci.py": dedent(
            """
            from __future__ import annotations

            import argparse
            import subprocess
            import sys
            import tomllib
            from pathlib import Path


            def main(argv: list[str] | None = None) -> int:
                parser = argparse.ArgumentParser()
                parser.add_argument("--package")
                args = parser.parse_args(argv)

                config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
                configured_package = config["tool"]["lumo_ci"]["package"]
                package_name = args.package or configured_package
                if package_name != "ci_app":
                    print(f"ci package drift: {package_name}", file=sys.stderr)
                    return 2
                return subprocess.call([sys.executable, "-m", "pytest", "-q"])


            if __name__ == "__main__":
                raise SystemExit(main())
            """
        ).strip(),
        "ci_app/__init__.py": "",
        "ci_app/service.py": dedent(
            f"""
            from __future__ import annotations

            CHECKLIST = {checklist}


            def run_checks() -> list[str]:
                return [item["label"] for item in CHECKLIST if item["required"]]
            """
        ).strip(),
        "ci_app/cli.py": dedent(
            """
            from __future__ import annotations

            from ci_app.service import run_checks


            def main() -> str:
                return ",".join(run_checks())
            """
        ).strip(),
        "tests/test_service.py": dedent(
            f"""
            from __future__ import annotations

            from ci_app.cli import main


            def test_service_contract_still_returns_required_labels() -> None:
                output = main()
                assert "{variant['checklist'][0]['label']}" in output
                assert "{variant['checklist'][1]['label']}" in output
            """
        ).strip(),
    }


def _investigate_repo(variant: dict) -> dict[str, str]:
    events = "\n".join(json.dumps(item, sort_keys=True) for item in variant["events"])
    return {
        ".scenario_variant": f"{variant['variant_id']}\n",
        "AGENTS.md": dedent(
            f"""
            An integration regression in the {variant['domain']} alert stream started after
            the dedupe logic was simplified.

            Read `logs/failure.log`, reproduce the problem through the tests, and fix the
            dedupe key so alerts are collapsed by the correct time window and environment.
            The code should preserve distinct incidents instead of merging them together.

            Do not delete the failing integration coverage.
            """
        ).strip(),
        "pyproject.toml": dedent(
            f"""
            [project]
            name = "{variant['project_name']}"
            version = "0.1.0"
            description = "{variant['domain']} alert investigation fixture"
            requires-python = ">=3.12"
            """
        ).strip(),
        "logs/failure.log": events,
        "investigate_app/__init__.py": "",
        "investigate_app/dedupe.py": dedent(
            """
            from __future__ import annotations


            def fingerprint(event: dict[str, str]) -> str:
                return f"{event['service']}::{event['title']}"


            def collapse(events: list[dict[str, str]]) -> list[dict[str, str]]:
                grouped: dict[str, dict[str, str]] = {}
                for event in events:
                    grouped[fingerprint(event)] = event
                return list(grouped.values())
            """
        ).strip(),
        "investigate_app/parser.py": dedent(
            """
            from __future__ import annotations

            import json
            from pathlib import Path


            def load_events(path: str = "logs/failure.log") -> list[dict[str, str]]:
                return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
            """
        ).strip(),
        "tests/test_integration.py": dedent(
            f"""
            from __future__ import annotations

            from investigate_app.dedupe import collapse
            from investigate_app.parser import load_events


            def test_distinct_windows_are_not_merged() -> None:
                collapsed = collapse(load_events())
                keys = {{(item["environment"], item["window_start"]) for item in collapsed}}
                assert len(collapsed) == {variant['expected_count']}
                assert ("{variant['events'][0]['environment']}", "{variant['events'][0]['window_start']}") in keys
                assert ("{variant['events'][1]['environment']}", "{variant['events'][1]['window_start']}") in keys
            """
        ).strip(),
        "tests/test_contract.py": dedent(
            """
            from __future__ import annotations

            from pathlib import Path


            def test_failure_log_mentions_the_window_collision() -> None:
                text = Path("logs/failure.log").read_text(encoding="utf-8")
                assert "window_start" in text
                assert "environment" in text
            """
        ).strip(),
    }


def _cross_layer_repo(variant: dict) -> dict[str, str]:
    defaults = json.dumps({"owner": variant["default_owner"], "status": "pending"}, indent=2)
    return {
        ".scenario_variant": f"{variant['variant_id']}\n",
        "AGENTS.md": dedent(
            f"""
            The {variant['domain']} sync flow needs owner routing wired through every layer.

            Tests now expect the backend store, service API, CLI entrypoint, default config,
            and docs to understand an `owner` field. Add the field end-to-end while keeping
            the existing `name` and `status` behavior intact.

            Do not remove the schema checks or the CLI coverage.
            """
        ).strip(),
        "pyproject.toml": dedent(
            f"""
            [project]
            name = "{variant['project_name']}"
            version = "0.1.0"
            description = "{variant['domain']} cross-layer sync fixture"
            requires-python = ">=3.12"
            """
        ).strip(),
        "config/defaults.json": defaults,
        "docs/cli.md": dedent(
            f"""
            # {variant['domain']} sync CLI

            Current command:

            ```bash
            python -m sync_app.cli --name {variant['records'][0]['name']} --status pending
            ```
            """
        ).strip(),
        "sync_app/__init__.py": "",
        "sync_app/store.py": dedent(
            """
            from __future__ import annotations


            def make_record(name: str, status: str) -> dict[str, str]:
                return {"name": name, "status": status}
            """
        ).strip(),
        "sync_app/service.py": dedent(
            """
            from __future__ import annotations

            from sync_app.store import make_record


            def sync_item(name: str, status: str, owner: str | None = None) -> dict[str, str]:
                del owner
                return make_record(name, status)
            """
        ).strip(),
        "sync_app/cli.py": dedent(
            """
            from __future__ import annotations

            import argparse
            import json

            from sync_app.service import sync_item


            def main(argv: list[str] | None = None) -> str:
                parser = argparse.ArgumentParser()
                parser.add_argument("--name", required=True)
                parser.add_argument("--status", required=True)
                args = parser.parse_args(argv)
                payload = sync_item(args.name, args.status)
                return json.dumps(payload, sort_keys=True)


            if __name__ == "__main__":
                print(main())
            """
        ).strip(),
        "tests/test_sync.py": dedent(
            f"""
            from __future__ import annotations

            import json

            from sync_app.cli import main
            from sync_app.service import sync_item


            def test_owner_is_persisted_by_the_service() -> None:
                payload = sync_item("{variant['records'][0]['name']}", "pending", owner="{variant['default_owner']}")
                assert payload["owner"] == "{variant['default_owner']}"


            def test_cli_accepts_owner_flag() -> None:
                payload = json.loads(
                    main(
                        [
                            "--name",
                            "{variant['records'][0]['name']}",
                            "--status",
                            "pending",
                            "--owner",
                            "{variant['default_owner']}",
                        ]
                    )
                )
                assert payload["owner"] == "{variant['default_owner']}"
            """
        ).strip(),
        "tests/test_docs.py": dedent(
            """
            from __future__ import annotations

            from pathlib import Path


            def test_docs_and_defaults_call_out_owner_field() -> None:
                docs = Path("docs/cli.md").read_text(encoding="utf-8")
                defaults = Path("config/defaults.json").read_text(encoding="utf-8")
                assert "--owner" in docs
                assert '"owner"' in defaults
            """
        ).strip(),
    }


FAMILIES = [
    {
        "family_id": "report-cli-markdown-evolution",
        "scenario_type": "feature_evolution",
        "description": "Add a Markdown output mode to a local Python reporting CLI and update the docs to match.",
        "breakage_class": {
            "injection_method": "spec_delta_tests",
            "description": "Tests now require a Markdown mode plus updated usage docs, but the repo still only supports JSON rendering.",
            "surfaces": ["cli", "renderer", "docs"],
        },
        "functional_check": {
            "id": "pytest_suite",
            "command": "cd /workspace && python -m pytest -q",
            "description": "Run the report CLI test suite",
            "timeout_seconds": 120,
        },
        "expected_final_state": [
            {"cli_markdown_check": "The CLI accepts --format markdown."},
            {"renderer_check": "A Markdown renderer exists and is wired in."},
            {"docs_check": "Usage docs mention the Markdown command."},
            {"test_suite_check": "The pytest suite passes from Phase 2."},
        ],
        "milestones": [
            ("m1_cli_markdown", "CLI accepts --format markdown", 0.2),
            ("m2_renderer_markdown", "Markdown renderer exists and is wired in", 0.35),
            ("m3_docs_updated", "Usage docs mention the Markdown workflow", 0.45),
        ],
        "shortcut_resistance": {
            "notes": "Phase 3 checks the CLI source, renderer source, and docs text directly while test-file checksums catch test deletion or edits.",
            "known_exploits_tested": [
                "Delete docs assertions from the pytest suite",
                "Stub the CLI to always print a canned Markdown string",
                "Shadow pytest to exit 0 without exercising the renderer",
            ],
        },
        "difficulty_estimate": {
            "target_solve_rate": "35-55%",
            "rationale": "Requires coordinated CLI, renderer, and docs changes with clear regression tests.",
        },
        "variants": [
            {
                "variant_id": "inventory-ops",
                "project_name": "inventory-ops-report",
                "title": "Inventory Ops Report",
                "slug": "inventory-ops",
                "records": [
                    {"label": "blocked-picks", "count": 4, "owner": "Mae"},
                    {"label": "late-recounts", "count": 2, "owner": "Noah"},
                ],
            },
            {
                "variant_id": "incident-triage",
                "project_name": "incident-triage-report",
                "title": "Incident Triage Report",
                "slug": "incident-triage",
                "records": [
                    {"label": "stale-pages", "count": 3, "owner": "Jules"},
                    {"label": "escalations", "count": 1, "owner": "Ivy"},
                ],
            },
            {
                "variant_id": "release-readiness",
                "project_name": "release-readiness-report",
                "title": "Release Readiness Report",
                "slug": "release-readiness",
                "records": [
                    {"label": "blocked-rollouts", "count": 2, "owner": "Sam"},
                    {"label": "hotfixes", "count": 1, "owner": "Rin"},
                ],
            },
        ],
        "smoke_command": "python -m pytest -q",
    },
    {
        "family_id": "normalizer-api-migration",
        "scenario_type": "migration_refactor",
        "description": "Migrate a Python codebase off a removed legacy planning helper and onto a dataclass-based RulePlan v2 API.",
        "breakage_class": {
            "injection_method": "api_swap",
            "description": "The deprecated legacy builder now raises, and the repo must be migrated to rules_v2.build_rule_plan().",
            "surfaces": ["imports", "call_sites", "cli_preview"],
        },
        "functional_check": {
            "id": "pytest_suite",
            "command": "cd /workspace && python -m pytest -q",
            "description": "Run the normalization preview test suite",
            "timeout_seconds": 120,
        },
        "expected_final_state": [
            {"legacy_import_check": "No legacy_rules imports remain in the package."},
            {"ruleplan_usage_check": "RulePlan v2 helpers are used in the assembler and router."},
            {"test_suite_check": "The pytest suite passes from Phase 2."},
        ],
        "milestones": [
            ("m1_legacy_imports_removed", "Removed legacy_rules imports", 0.25),
            ("m2_ruleplan_v2_used", "Assembler and router use RulePlan v2", 0.35),
            ("m3_tests_passing", "Pytest suite passes", 0.4),
        ],
        "shortcut_resistance": {
            "notes": "Trusted greps reject stale legacy imports and require RulePlan v2 symbols even if Phase 2 were spoofed.",
            "known_exploits_tested": [
                "Shadow pytest to exit 0",
                "Monkeypatch the legacy helper to stop raising",
                "Delete the contract test that checks the new import path",
            ],
        },
        "difficulty_estimate": {
            "target_solve_rate": "30-50%",
            "rationale": "Requires a repo-wide import migration plus adapting the call-site field names.",
        },
        "variants": [
            {
                "variant_id": "alert-routing",
                "project_name": "alert-routing-normalizer",
                "domain": "alert routing",
                "sample": {"title": "Disk Pressure", "owner": "ops", "region": "us-east"},
                "expected_slug": "disk-pressure",
                "expected_bucket": "us-east:ops",
            },
            {
                "variant_id": "billing-ledger",
                "project_name": "billing-ledger-normalizer",
                "domain": "billing ledger",
                "sample": {"title": "Invoice Retry", "owner": "finance", "region": "eu-west"},
                "expected_slug": "invoice-retry",
                "expected_bucket": "eu-west:finance",
            },
            {
                "variant_id": "catalog-sync",
                "project_name": "catalog-sync-normalizer",
                "domain": "catalog sync",
                "sample": {"title": "Missing SKU", "owner": "catalog", "region": "ap-south"},
                "expected_slug": "missing-sku",
                "expected_bucket": "ap-south:catalog",
            },
        ],
        "smoke_command": "python -m pytest -q",
    },
    {
        "family_id": "ci-config-coverage-drift",
        "scenario_type": "build_ci_breakage",
        "description": "Repair a Python repo whose local CI contract drifted after a package rename.",
        "breakage_class": {
            "injection_method": "config_drift",
            "description": "The package was renamed to ci_app, but pyproject and CI workflow files still point at the old package token.",
            "surfaces": ["pyproject", "workflow", "make_ci_contract"],
        },
        "functional_check": {
            "id": "make_ci",
            "command": "cd /workspace && make ci",
            "description": "Run the repo's local CI command",
            "timeout_seconds": 120,
        },
        "expected_final_state": [
            {"pyproject_check": "tool.lumo_ci.package points at ci_app."},
            {"workflow_check": "The workflow calls make ci or ci_app, not the legacy package."},
            {"test_suite_check": "The local make ci contract passes from Phase 2."},
        ],
        "milestones": [
            ("m1_pyproject_synced", "pyproject points at ci_app", 0.25),
            ("m2_workflow_synced", "CI workflow matches the local contract", 0.35),
            ("m3_ci_passing", "make ci passes", 0.4),
        ],
        "shortcut_resistance": {
            "notes": "Phase 3 checks the config and workflow files directly while checksum manifests stop test deletion.",
            "known_exploits_tested": [
                "Delete tests and keep make ci empty",
                "Hardcode scripts/run_ci.py to return 0",
                "Shadow pytest to claim success without fixing the config drift",
            ],
        },
        "difficulty_estimate": {
            "target_solve_rate": "25-45%",
            "rationale": "Requires understanding the CI contract across config, workflow, and helper-script layers.",
        },
        "variants": [
            {
                "variant_id": "inventory-gate",
                "project_name": "inventory-gate-ci",
                "domain": "inventory gate",
                "legacy_package": "inventory_gate_legacy",
                "checklist": [
                    {"label": "schema-check", "required": True},
                    {"label": "render-check", "required": True},
                ],
            },
            {
                "variant_id": "payments-gate",
                "project_name": "payments-gate-ci",
                "domain": "payments gate",
                "legacy_package": "payments_gate_legacy",
                "checklist": [
                    {"label": "queue-check", "required": True},
                    {"label": "ledger-check", "required": True},
                ],
            },
            {
                "variant_id": "search-gate",
                "project_name": "search-gate-ci",
                "domain": "search gate",
                "legacy_package": "search_gate_legacy",
                "checklist": [
                    {"label": "ranking-check", "required": True},
                    {"label": "fixture-check", "required": True},
                ],
            },
        ],
        "smoke_command": "make ci",
    },
    {
        "family_id": "alert-dedupe-investigation",
        "scenario_type": "investigate_then_fix",
        "description": "Investigate a failing alert-stream integration test using log evidence and repair the dedupe key.",
        "breakage_class": {
            "injection_method": "logic_regression",
            "description": "A simplified dedupe fingerprint ignores window and environment fields, causing distinct incidents to merge.",
            "surfaces": ["log_analysis", "dedupe_key", "integration_tests"],
        },
        "functional_check": {
            "id": "pytest_suite",
            "command": "cd /workspace && python -m pytest -q",
            "description": "Run the alert-stream integration tests",
            "timeout_seconds": 120,
        },
        "expected_final_state": [
            {"fingerprint_check": "The dedupe fingerprint includes environment and window_start."},
            {"test_suite_check": "The pytest suite passes from Phase 2."},
        ],
        "milestones": [
            ("m1_window_key_used", "Fingerprint includes the time window", 0.3),
            ("m2_environment_key_used", "Fingerprint includes the environment", 0.3),
            ("m3_tests_passing", "Pytest suite passes", 0.4),
        ],
        "shortcut_resistance": {
            "notes": "Phase 3 inspects the dedupe source for the required fields and uses test checksums to catch deleted integration coverage.",
            "known_exploits_tested": [
                "Delete the failing integration test",
                "Shadow pytest to report success",
                "Return a canned dedupe result without fixing the source key",
            ],
        },
        "difficulty_estimate": {
            "target_solve_rate": "30-50%",
            "rationale": "Starts from logs and failing tests, then requires a targeted logic fix in the reducer.",
        },
        "variants": [
            {
                "variant_id": "payments-oncall",
                "project_name": "payments-oncall-alerts",
                "domain": "payments on-call",
                "events": [
                    {"service": "payments", "title": "Queue Lag", "environment": "prod", "window_start": "2026-04-15T10:00:00Z"},
                    {"service": "payments", "title": "Queue Lag", "environment": "prod", "window_start": "2026-04-15T10:05:00Z"},
                ],
                "expected_count": 2,
            },
            {
                "variant_id": "search-oncall",
                "project_name": "search-oncall-alerts",
                "domain": "search on-call",
                "events": [
                    {"service": "search", "title": "Shard Saturation", "environment": "staging", "window_start": "2026-04-15T09:00:00Z"},
                    {"service": "search", "title": "Shard Saturation", "environment": "prod", "window_start": "2026-04-15T09:00:00Z"},
                ],
                "expected_count": 2,
            },
            {
                "variant_id": "inventory-oncall",
                "project_name": "inventory-oncall-alerts",
                "domain": "inventory on-call",
                "events": [
                    {"service": "inventory", "title": "Recount Drift", "environment": "prod", "window_start": "2026-04-15T07:00:00Z"},
                    {"service": "inventory", "title": "Recount Drift", "environment": "prod", "window_start": "2026-04-15T08:00:00Z"},
                ],
                "expected_count": 2,
            },
        ],
        "smoke_command": "python -m pytest -q",
    },
    {
        "family_id": "owner-field-cross-layer",
        "scenario_type": "cross_layer_changes",
        "description": "Thread a new owner field through backend storage, service logic, CLI flags, config defaults, and docs.",
        "breakage_class": {
            "injection_method": "schema_extension",
            "description": "Tests now require an owner field to exist across storage, service, CLI, defaults, and docs, but the repo only handles name/status.",
            "surfaces": ["store", "service", "cli", "config", "docs"],
        },
        "functional_check": {
            "id": "pytest_suite",
            "command": "cd /workspace && python -m pytest -q",
            "description": "Run the cross-layer owner-field tests",
            "timeout_seconds": 120,
        },
        "expected_final_state": [
            {"store_check": "The storage layer persists owner."},
            {"cli_check": "The CLI accepts --owner and passes it through."},
            {"defaults_check": "Default config and docs mention owner."},
            {"test_suite_check": "The pytest suite passes from Phase 2."},
        ],
        "milestones": [
            ("m1_store_owner_added", "Storage layer persists owner", 0.25),
            ("m2_cli_owner_wired", "CLI and service pass through owner", 0.35),
            ("m3_defaults_and_tests", "Defaults/docs mention owner and tests pass", 0.4),
        ],
        "shortcut_resistance": {
            "notes": "Phase 3 inspects store, CLI, defaults, and docs directly and checks that the tests were not modified.",
            "known_exploits_tested": [
                "Delete the owner assertions from the test suite",
                "Shadow pytest to exit 0",
                "Return a canned JSON payload from the CLI without changing the store or defaults",
            ],
        },
        "difficulty_estimate": {
            "target_solve_rate": "35-60%",
            "rationale": "Requires coordinated changes across backend, service, CLI, config, and docs layers.",
        },
        "variants": [
            {
                "variant_id": "project-board",
                "project_name": "project-board-sync",
                "domain": "project board",
                "default_owner": "pm-oncall",
                "records": [{"name": "launch-checklist"}],
            },
            {
                "variant_id": "warehouse-queue",
                "project_name": "warehouse-queue-sync",
                "domain": "warehouse queue",
                "default_owner": "ops-lead",
                "records": [{"name": "picker-backlog"}],
            },
            {
                "variant_id": "release-gate",
                "project_name": "release-gate-sync",
                "domain": "release gate",
                "default_owner": "release-captain",
                "records": [{"name": "patch-train"}],
            },
        ],
        "smoke_command": "python -m pytest -q",
    },
]


def _family_yaml(family: dict) -> dict:
    family_id = family["family_id"]
    variants = []
    for variant in family["variants"]:
        variants.append(
            {
                "variant_id": variant["variant_id"],
                "repo_source": "authored",
                "injected_breakage": family["breakage_class"]["description"],
                "env_dockerfile": f"variants/{variant['variant_id']}/Dockerfile",
                "base_image_digest": BASE_IMAGE_DIGEST,
            }
        )
    milestones = []
    for milestone_id, description, partial_credit in family["milestones"]:
        milestones.append(
            {
                "id": milestone_id,
                "description": description,
                "check_script": f"verifiers/{family_id}/milestones/{milestone_id}.sh",
                "partial_credit": partial_credit,
            }
        )
    return {
        "family_id": family_id,
        "scenario_type": family["scenario_type"],
        "description": family["description"],
        "repo_pattern": {
            "language": "python",
            "framework": "pytest",
            "structure": "Small Python repos rooted at /workspace with source files, docs/config, and pytest coverage.",
            "base_image": BASE_IMAGE,
        },
        "breakage_class": family["breakage_class"],
        "grading_invariant": {
            "type": "state_based",
            "description": "Final state is graded by trusted Phase 3 file inspections plus the Phase 2 command result.",
            "verifier_script": f"verifiers/{family_id}/verify.sh",
            "functional_checks": [family["functional_check"]],
            "expected_final_state": family["expected_final_state"],
        },
        "milestones": milestones,
        "shortcut_resistance": family["shortcut_resistance"],
        "difficulty_estimate": family["difficulty_estimate"],
        "variants": variants,
    }


def _verifier_data(family: dict) -> dict:
    family_id = family["family_id"]
    if family_id == "report-cli-markdown-evolution":
        return {
            "variants": {
                variant["variant_id"]: {
                    "project_name": variant["project_name"],
                    "project_name_pattern": f'name = "{variant["project_name"]}"',
                    "checksum_file": f'test_checksums/{variant["variant_id"]}.sha256',
                    "m1_file": "report_app/cli.py",
                    "m1_pattern": '"markdown"',
                    "m2_file": "report_app/rendering.py",
                    "m2_pattern": "def render_markdown",
                    "m3_file": "docs/usage.md",
                    "m3_pattern": "--format markdown",
                }
                for variant in family["variants"]
            }
        }
    if family_id == "normalizer-api-migration":
        return {
            "variants": {
                variant["variant_id"]: {
                    "project_name": variant["project_name"],
                    "project_name_pattern": f'name = "{variant["project_name"]}"',
                    "checksum_file": f'test_checksums/{variant["variant_id"]}.sha256',
                    "m1_glob": "norm_app",
                    "m1_forbidden_pattern": "legacy_rules",
                    "m2_file_a": "norm_app/assembler.py",
                    "m2_file_b": "norm_app/router.py",
                    "m2_pattern": "build_rule_plan",
                }
                for variant in family["variants"]
            }
        }
    if family_id == "ci-config-coverage-drift":
        return {
            "variants": {
                variant["variant_id"]: {
                    "project_name": variant["project_name"],
                    "project_name_pattern": f'name = "{variant["project_name"]}"',
                    "checksum_file": f'test_checksums/{variant["variant_id"]}.sha256',
                    "m1_file": "pyproject.toml",
                    "m1_pattern": 'package = "ci_app"',
                    "m2_file": ".github/workflows/ci.yml",
                    "m2_required_pattern": "make ci",
                    "m2_forbidden_pattern": variant["legacy_package"],
                }
                for variant in family["variants"]
            }
        }
    if family_id == "alert-dedupe-investigation":
        return {
            "variants": {
                variant["variant_id"]: {
                    "project_name": variant["project_name"],
                    "project_name_pattern": f'name = "{variant["project_name"]}"',
                    "checksum_file": f'test_checksums/{variant["variant_id"]}.sha256',
                    "m1_file": "investigate_app/dedupe.py",
                    "m1_pattern": "window_start",
                    "m2_file": "investigate_app/dedupe.py",
                    "m2_pattern": "environment",
                }
                for variant in family["variants"]
            }
        }
    if family_id == "owner-field-cross-layer":
        return {
            "variants": {
                variant["variant_id"]: {
                    "project_name": variant["project_name"],
                    "project_name_pattern": f'name = "{variant["project_name"]}"',
                    "checksum_file": f'test_checksums/{variant["variant_id"]}.sha256',
                    "m1_file": "sync_app/store.py",
                    "m1_pattern": '"owner"',
                    "m2_file": "sync_app/cli.py",
                    "m2_pattern": '"--owner"',
                    "m3_file_a": "config/defaults.json",
                    "m3_file_b": "docs/cli.md",
                    "m3_pattern": "owner",
                }
                for variant in family["variants"]
            }
        }
    raise KeyError(f"Unknown family verifier data template: {family_id}")


def _milestone_script(family_id: str, milestone_id: str) -> str:
    if family_id == "report-cli-markdown-evolution":
        if milestone_id == "m1_cli_markdown":
            body = """
            check_m1_cli_markdown() {
                local agent_ws="$1"
                local config_path="$2"
                local variant_id="$3"
                local file pattern
                file=$(jq -r --arg v "$variant_id" '.variants[$v].m1_file' "$config_path")
                pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m1_pattern' "$config_path")
                grep -Fq -- "$pattern" "$agent_ws/$file"
            }
            """
        elif milestone_id == "m2_renderer_markdown":
            body = """
            check_m2_renderer_markdown() {
                local agent_ws="$1"
                local config_path="$2"
                local variant_id="$3"
                local file pattern
                file=$(jq -r --arg v "$variant_id" '.variants[$v].m2_file' "$config_path")
                pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m2_pattern' "$config_path")
                grep -Fq -- "$pattern" "$agent_ws/$file"
            }
            """
        else:
            body = """
            check_m3_docs_updated() {
                local agent_ws="$1"
                local config_path="$2"
                local variant_id="$3"
                local file pattern
                file=$(jq -r --arg v "$variant_id" '.variants[$v].m3_file' "$config_path")
                pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m3_pattern' "$config_path")
                grep -Fq -- "$pattern" "$agent_ws/$file"
            }
            """
    elif family_id == "normalizer-api-migration":
        if milestone_id == "m1_legacy_imports_removed":
            body = """
            check_m1_legacy_imports_removed() {
                local agent_ws="$1"
                local config_path="$2"
                local variant_id="$3"
                local scope forbidden
                scope=$(jq -r --arg v "$variant_id" '.variants[$v].m1_glob' "$config_path")
                forbidden=$(jq -r --arg v "$variant_id" '.variants[$v].m1_forbidden_pattern' "$config_path")
                ! grep -R -F -- "$forbidden" "$agent_ws/$scope"
            }
            """
        elif milestone_id == "m2_ruleplan_v2_used":
            body = """
            check_m2_ruleplan_v2_used() {
                local agent_ws="$1"
                local config_path="$2"
                local variant_id="$3"
                local file_a file_b pattern
                file_a=$(jq -r --arg v "$variant_id" '.variants[$v].m2_file_a' "$config_path")
                file_b=$(jq -r --arg v "$variant_id" '.variants[$v].m2_file_b' "$config_path")
                pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m2_pattern' "$config_path")
                grep -Fq -- "$pattern" "$agent_ws/$file_a" && grep -Fq -- "$pattern" "$agent_ws/$file_b"
            }
            """
        else:
            body = """
            check_m3_tests_passing() {
                local functional_dir="$1"
                [ -f "$functional_dir/pytest_suite_exit_code" ] && [ "$(cat "$functional_dir/pytest_suite_exit_code")" = "0" ]
            }
            """
    elif family_id == "ci-config-coverage-drift":
        if milestone_id == "m1_pyproject_synced":
            body = """
            check_m1_pyproject_synced() {
                local agent_ws="$1"
                local config_path="$2"
                local variant_id="$3"
                local file pattern
                file=$(jq -r --arg v "$variant_id" '.variants[$v].m1_file' "$config_path")
                pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m1_pattern' "$config_path")
                grep -Fq -- "$pattern" "$agent_ws/$file"
            }
            """
        elif milestone_id == "m2_workflow_synced":
            body = """
            check_m2_workflow_synced() {
                local agent_ws="$1"
                local config_path="$2"
                local variant_id="$3"
                local file required forbidden
                file=$(jq -r --arg v "$variant_id" '.variants[$v].m2_file' "$config_path")
                required=$(jq -r --arg v "$variant_id" '.variants[$v].m2_required_pattern' "$config_path")
                forbidden=$(jq -r --arg v "$variant_id" '.variants[$v].m2_forbidden_pattern' "$config_path")
                grep -Fq -- "$required" "$agent_ws/$file" && ! grep -Fq -- "$forbidden" "$agent_ws/$file"
            }
            """
        else:
            body = """
            check_m3_ci_passing() {
                local functional_dir="$1"
                [ -f "$functional_dir/make_ci_exit_code" ] && [ "$(cat "$functional_dir/make_ci_exit_code")" = "0" ]
            }
            """
    elif family_id == "alert-dedupe-investigation":
        if milestone_id == "m1_window_key_used":
            body = """
            check_m1_window_key_used() {
                local agent_ws="$1"
                local config_path="$2"
                local variant_id="$3"
                local file pattern
                file=$(jq -r --arg v "$variant_id" '.variants[$v].m1_file' "$config_path")
                pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m1_pattern' "$config_path")
                grep -Fq -- "$pattern" "$agent_ws/$file"
            }
            """
        elif milestone_id == "m2_environment_key_used":
            body = """
            check_m2_environment_key_used() {
                local agent_ws="$1"
                local config_path="$2"
                local variant_id="$3"
                local file pattern
                file=$(jq -r --arg v "$variant_id" '.variants[$v].m2_file' "$config_path")
                pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m2_pattern' "$config_path")
                grep -Fq -- "$pattern" "$agent_ws/$file"
            }
            """
        else:
            body = """
            check_m3_tests_passing() {
                local functional_dir="$1"
                [ -f "$functional_dir/pytest_suite_exit_code" ] && [ "$(cat "$functional_dir/pytest_suite_exit_code")" = "0" ]
            }
            """
    elif family_id == "owner-field-cross-layer":
        if milestone_id == "m1_store_owner_added":
            body = """
            check_m1_store_owner_added() {
                local agent_ws="$1"
                local config_path="$2"
                local variant_id="$3"
                local file pattern
                file=$(jq -r --arg v "$variant_id" '.variants[$v].m1_file' "$config_path")
                pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m1_pattern' "$config_path")
                grep -Fq -- "$pattern" "$agent_ws/$file"
            }
            """
        elif milestone_id == "m2_cli_owner_wired":
            body = """
            check_m2_cli_owner_wired() {
                local agent_ws="$1"
                local config_path="$2"
                local variant_id="$3"
                local file pattern
                file=$(jq -r --arg v "$variant_id" '.variants[$v].m2_file' "$config_path")
                pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m2_pattern' "$config_path")
                grep -Fq -- "$pattern" "$agent_ws/$file"
            }
            """
        else:
            body = """
            check_m3_defaults_and_tests() {
                local agent_ws="$1"
                local functional_dir="$2"
                local config_path="$3"
                local variant_id="$4"
                local file_a file_b pattern
                file_a=$(jq -r --arg v "$variant_id" '.variants[$v].m3_file_a' "$config_path")
                file_b=$(jq -r --arg v "$variant_id" '.variants[$v].m3_file_b' "$config_path")
                pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m3_pattern' "$config_path")
                grep -Fq -- "$pattern" "$agent_ws/$file_a" && \
                    grep -Fq -- "$pattern" "$agent_ws/$file_b" && \
                    [ -f "$functional_dir/pytest_suite_exit_code" ] && [ "$(cat "$functional_dir/pytest_suite_exit_code")" = "0" ]
            }
            """
    else:
        raise KeyError(f"Unknown milestone family: {family_id}")
    return dedent(body).strip()


def _verify_script(family: dict) -> str:
    family_id = family["family_id"]
    milestone_ids = [milestone_id for milestone_id, _desc, _weight in family["milestones"]]
    source_lines = "\n".join(f"source /verifier/milestones/{milestone_id}.sh" for milestone_id in milestone_ids)
    if family_id == "report-cli-markdown-evolution":
        milestone_logic = dedent(
            """
            if check_m1_cli_markdown "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
              write_result '.milestones.m1_cli_markdown = true'
            else
              add_error "CLI still does not expose markdown output"
            fi

            if check_m2_renderer_markdown "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
              write_result '.milestones.m2_renderer_markdown = true'
            else
              add_error "Markdown renderer is still missing"
            fi

            if check_m3_docs_updated "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
              write_result '.milestones.m3_docs_updated = true'
            else
              add_error "Usage docs do not mention --format markdown"
            fi
            """
        ).strip()
    elif family_id == "normalizer-api-migration":
        milestone_logic = dedent(
            """
            if check_m1_legacy_imports_removed "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
              write_result '.milestones.m1_legacy_imports_removed = true'
            else
              add_error "legacy_rules imports still remain"
            fi

            if check_m2_ruleplan_v2_used "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
              write_result '.milestones.m2_ruleplan_v2_used = true'
            else
              add_error "RulePlan v2 helpers are not wired into assembler and router"
            fi

            if check_m3_tests_passing "$FUNCTIONAL_DIR"; then
              write_result '.milestones.m3_tests_passing = true'
            else
              add_error "Phase 2 pytest suite did not pass"
            fi
            """
        ).strip()
    elif family_id == "ci-config-coverage-drift":
        milestone_logic = dedent(
            """
            if check_m1_pyproject_synced "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
              write_result '.milestones.m1_pyproject_synced = true'
            else
              add_error "pyproject.toml still points at the legacy package"
            fi

            if check_m2_workflow_synced "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
              write_result '.milestones.m2_workflow_synced = true'
            else
              add_error "workflow file is still out of sync with make ci"
            fi

            if check_m3_ci_passing "$FUNCTIONAL_DIR"; then
              write_result '.milestones.m3_ci_passing = true'
            else
              add_error "Phase 2 make ci did not pass"
            fi
            """
        ).strip()
    elif family_id == "alert-dedupe-investigation":
        milestone_logic = dedent(
            """
            if check_m1_window_key_used "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
              write_result '.milestones.m1_window_key_used = true'
            else
              add_error "dedupe fingerprint still ignores window_start"
            fi

            if check_m2_environment_key_used "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
              write_result '.milestones.m2_environment_key_used = true'
            else
              add_error "dedupe fingerprint still ignores environment"
            fi

            if check_m3_tests_passing "$FUNCTIONAL_DIR"; then
              write_result '.milestones.m3_tests_passing = true'
            else
              add_error "Phase 2 pytest suite did not pass"
            fi
            """
        ).strip()
    elif family_id == "owner-field-cross-layer":
        milestone_logic = dedent(
            """
            if check_m1_store_owner_added "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
              write_result '.milestones.m1_store_owner_added = true'
            else
              add_error "store layer does not persist owner"
            fi

            if check_m2_cli_owner_wired "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
              write_result '.milestones.m2_cli_owner_wired = true'
            else
              add_error "CLI does not accept --owner"
            fi

            if check_m3_defaults_and_tests "$AGENT_WS" "$FUNCTIONAL_DIR" "$CONFIG_PATH" "$VARIANT_ID"; then
              write_result '.milestones.m3_defaults_and_tests = true'
            else
              add_error "defaults/docs/test path for owner is incomplete"
            fi
            """
        ).strip()
    else:
        raise KeyError(f"Unknown verify family: {family_id}")

    return dedent(
        f"""
        #!/usr/bin/env bash
        set -euo pipefail

        AGENT_WS="/agent/workspace"
        FUNCTIONAL_DIR="/functional"
        CONFIG_PATH="/verifier_data/variant_expectations.json"
        RESULT_FILE="/results/verify_result.json"
        START_TS=$(date +%s)

        mkdir -p /results
        printf '{{"pass": false, "milestones": {{}}, "errors": [], "shortcut_detected": false, "wall_clock_seconds": 0}}' > "$RESULT_FILE"

        write_result() {{
          local jq_expr="$1"
          tmp=$(mktemp)
          jq "$jq_expr" "$RESULT_FILE" > "$tmp"
          mv "$tmp" "$RESULT_FILE"
        }}

        add_error() {{
          local message="$1"
          tmp=$(mktemp)
          jq --arg message "$message" '.errors += [$message]' "$RESULT_FILE" > "$tmp"
          mv "$tmp" "$RESULT_FILE"
        }}

        {source_lines}

        if [ ! -f "$AGENT_WS/.scenario_variant" ]; then
          add_error "missing .scenario_variant marker"
          jq --argjson elapsed "$(( $(date +%s) - START_TS ))" '.wall_clock_seconds = $elapsed' "$RESULT_FILE" > "$RESULT_FILE.tmp"
          mv "$RESULT_FILE.tmp" "$RESULT_FILE"
          cat "$RESULT_FILE"
          exit 0
        fi

        VARIANT_ID=$(tr -d '\\n' < "$AGENT_WS/.scenario_variant")
        if ! jq -e --arg v "$VARIANT_ID" '.variants[$v]' "$CONFIG_PATH" >/dev/null; then
          add_error "unknown variant marker: $VARIANT_ID"
          jq --argjson elapsed "$(( $(date +%s) - START_TS ))" '.wall_clock_seconds = $elapsed' "$RESULT_FILE" > "$RESULT_FILE.tmp"
          mv "$RESULT_FILE.tmp" "$RESULT_FILE"
          cat "$RESULT_FILE"
          exit 0
        fi

        PROJECT_PATTERN=$(jq -r --arg v "$VARIANT_ID" '.variants[$v].project_name_pattern' "$CONFIG_PATH")
        if ! grep -Fq -- "$PROJECT_PATTERN" "$AGENT_WS/pyproject.toml"; then
          add_error "project identity does not match variant marker"
        fi

        CHECKSUM_FILE=$(jq -r --arg v "$VARIANT_ID" '.variants[$v].checksum_file' "$CONFIG_PATH")
        if [ -f "/verifier_data/$CHECKSUM_FILE" ]; then
          sed 's|  \\./|  /agent/workspace/|' "/verifier_data/$CHECKSUM_FILE" > /tmp/checksums_remapped.sha256
          if ! sha256sum -c /tmp/checksums_remapped.sha256 --quiet >/dev/null 2>&1; then
            write_result '.shortcut_detected = true'
            add_error "test files were modified"
          fi
        else
          add_error "missing verifier checksum manifest for $VARIANT_ID"
        fi

        {milestone_logic}

        if jq -e '.errors | length == 0' "$RESULT_FILE" >/dev/null && \
           jq -e '[.milestones[]] | length > 0 and all' "$RESULT_FILE" >/dev/null; then
          write_result '.pass = true'
        fi

        jq --argjson elapsed "$(( $(date +%s) - START_TS ))" '.wall_clock_seconds = $elapsed' "$RESULT_FILE" > "$RESULT_FILE.tmp"
        mv "$RESULT_FILE.tmp" "$RESULT_FILE"
        cat "$RESULT_FILE"
        """
    ).strip()


def _variant_expectation_files(family: dict) -> None:
    family_id = family["family_id"]
    family_dir = SCENARIO_FAMILIES_DIR / family_id
    verifier_dir = VERIFIER_DATA_DIR / family_id
    expectations = _verifier_data(family)
    _write(verifier_dir / "variant_expectations.json", json.dumps(expectations, indent=2, sort_keys=True))
    for variant in family["variants"]:
        tests_dir = family_dir / "variants" / variant["variant_id"] / "repo" / "tests"
        _write(
            verifier_dir / "test_checksums" / f"{variant['variant_id']}.sha256",
            _sha256_manifest(tests_dir),
        )


def _family_files(family: dict) -> None:
    family_id = family["family_id"]
    family_dir = SCENARIO_FAMILIES_DIR / family_id
    _write_yaml(family_dir / "family.yaml", _family_yaml(family))

    verifier_dir = VERIFIERS_DIR / family_id
    _write(verifier_dir / "verify.sh", _verify_script(family))
    _chmod_executable(verifier_dir / "verify.sh")

    for milestone_id, _desc, _weight in family["milestones"]:
        milestone_path = verifier_dir / "milestones" / f"{milestone_id}.sh"
        _write(milestone_path, _milestone_script(family_id, milestone_id))
        _chmod_executable(milestone_path)

    for variant in family["variants"]:
        variant_dir = family_dir / "variants" / variant["variant_id"]
        if family_id == "report-cli-markdown-evolution":
            repo_files = _feature_repo(variant)
        elif family_id == "normalizer-api-migration":
            repo_files = _migration_repo(variant)
        elif family_id == "ci-config-coverage-drift":
            repo_files = _build_repo(variant)
        elif family_id == "alert-dedupe-investigation":
            repo_files = _investigate_repo(variant)
        elif family_id == "owner-field-cross-layer":
            repo_files = _cross_layer_repo(variant)
        else:
            raise KeyError(f"Unknown family renderer: {family_id}")
        for relative_path, content in repo_files.items():
            _write(variant_dir / "repo" / relative_path, content)
        _write(variant_dir / "Dockerfile", _dockerfile(family["smoke_command"]))

    _variant_expectation_files(family)


def _readme() -> str:
    return dedent(
        """
        # Codex-Long Initial Authored Pack

        This repo now carries an initial real authored scenario pack for the Codex-Long
        framework. The pack is intentionally below the signed-off freeze threshold from
        LLD-13, so it does not include `split_assignment.yaml` or
        `benchmark_manifest.lock`.

        Included families:

        - `report-cli-markdown-evolution` (`feature_evolution`)
        - `normalizer-api-migration` (`migration_refactor`)
        - `ci-config-coverage-drift` (`build_ci_breakage`)
        - `alert-dedupe-investigation` (`investigate_then_fix`)
        - `owner-field-cross-layer` (`cross_layer_changes`)

        Each family contains three concrete authored variants with:

        - a broken repo and visible `AGENTS.md`
        - a Dockerfile with a build-time smoke failure
        - a verifier tree rooted at `verifiers/<family_id>/`
        - milestone helpers under `verifiers/<family_id>/milestones/`
        - verifier data and immutable test checksum manifests under `verifier_data/<family_id>/`

        Regenerate the pack with:

        ```bash
        .venv/bin/python scripts/generate_initial_codex_long_assets.py
        ```

        Validate the pack with:

        ```bash
        .venv/bin/python scripts/validate_codex_long_assets.py
        ```
        """
    ).strip()


def main() -> None:
    _reset_output_dirs()
    _write(SCENARIO_FAMILIES_DIR / "README.md", _readme())
    for family in FAMILIES:
        _family_files(family)


if __name__ == "__main__":
    main()
