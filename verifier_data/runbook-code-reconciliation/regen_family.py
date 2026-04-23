#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import os
import shutil
import stat
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY = REPO / "benchmark_blueprints/families/runbook-code-reconciliation"
VERIFIER = REPO / "verifier_data/runbook-code-reconciliation"

PRIMARY_ENTRYPOINT = "python src/release_preview/cli.py generate"
CURRENT_FLAG = "--config"
DEPRECATED_FLAG = "--settings"
CURRENT_ENV = "RELEASE_PREVIEW_CONFIG"
DEPRECATED_ENV = "PREVIEW_SETTINGS_PATH"
DEFAULT_CONFIG = "configs/release_preview.toml"
LEGACY_ALIAS = "python scripts/release_preview_helper.py build-preview"
LEGACY_ALIAS_NAME = "build-preview"
CURRENT_COMMAND = f"{PRIMARY_ENTRYPOINT} {CURRENT_FLAG} {DEFAULT_CONFIG}"
REQUIRED_EVIDENCE_COMMANDS = [
    "python src/release_preview/cli.py --help",
    "python src/release_preview/cli.py generate --help",
    "python scripts/release_preview_helper.py build-preview --help",
    "pytest -q tests/test_release_preview_cli.py",
]

VARIANTS = {
    "v1-clean-baseline": {
        "title": "Clean baseline",
        "current_fragment": "Current operator path: `python src/release_preview/cli.py generate --config configs/release_preview.toml`\nCurrent env fallback: `RELEASE_PREVIEW_CONFIG`\n",
        "legacy_fragment": "Older operator note: `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`\nOlder env: `PREVIEW_SETTINGS_PATH`\n",
        "runbook_mode": "legacy",
        "release_context": None,
        "incident_context": None,
    },
    "v2-noisy-distractor": {
        "title": "Noisy distractor",
        "current_fragment": "Current entrypoint is `python src/release_preview/cli.py generate`, but an old paste mistakenly kept `PREVIEW_SETTINGS_PATH` in the example.\n",
        "legacy_fragment": "Older operator note: `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`\nOlder env: `PREVIEW_SETTINGS_PATH`\n",
        "runbook_mode": "legacy",
        "release_context": None,
        "incident_context": None,
    },
    "v3-dirty-state": {
        "title": "Dirty state",
        "current_fragment": "Current path is `python src/release_preview/cli.py generate --config configs/release_preview.toml`, but a transition draft suggested leaving both paths in the runbook.\n",
        "legacy_fragment": "Compatibility helper still works: `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`.\n",
        "runbook_mode": "dual",
        "release_context": None,
        "incident_context": None,
    },
    "v4-multi-corpus-objective": {
        "title": "Multi-corpus objective drift",
        "current_fragment": "Current path is `python src/release_preview/cli.py generate --config configs/release_preview.toml`.\n",
        "legacy_fragment": "Compatibility helper still works, but that does not make it the preferred runbook path.\n",
        "runbook_mode": "dual",
        "release_context": {
            "README.md": "# Release context\n\nThese notes are supporting context only.\n",
            "speed_guidance.md": "Keep release-preview docs compact and avoid extra churn. Operators like the compatibility helper because it feels fast, but this note does not override the live CLI contract.\n",
        },
        "incident_context": None,
    },
    "v5-recovery-in-thread": {
        "title": "Recovery in thread",
        "current_fragment": "Current path is `python src/release_preview/cli.py generate --config configs/release_preview.toml`.\n",
        "legacy_fragment": "Compatibility helper still works. A previous docs fix wrongly described it as removed.\n",
        "runbook_mode": "dual",
        "release_context": {
            "README.md": "# Release context\n\nThese notes are supporting context only.\n",
            "speed_guidance.md": "Operators still want the shortest path, but the documented default must follow the live CLI and config names.\n",
        },
        "incident_context": {
            "README.md": "# Incident context\n\nThis directory captures a prior bad documentation fix.\n",
            "rollback_2026_02_21.md": "A prior fix claimed the compatibility helper alias was removed. That statement was rolled back because the alias still existed for backward compatibility; only the recommended default had changed.\n",
        },
    },
}


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def json_dump(path: Path, payload: object) -> None:
    write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
        rp = path.relative_to(target).as_posix()
        if path.is_dir():
            h.update(b"D:" + rp.encode() + b"\x00")
        elif path.is_file():
            h.update(b"F:" + rp.encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
    return h.hexdigest()


def set_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def cli_py() -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        from __future__ import annotations

        import argparse
        import os

        CURRENT_FLAG = "{CURRENT_FLAG}"
        DEPRECATED_FLAG = "{DEPRECATED_FLAG}"
        CURRENT_ENV = "{CURRENT_ENV}"
        DEPRECATED_ENV = "{DEPRECATED_ENV}"
        DEFAULT_CONFIG = "{DEFAULT_CONFIG}"


        def build_parser() -> argparse.ArgumentParser:
            parser = argparse.ArgumentParser(prog="release-preview")
            sub = parser.add_subparsers(dest="command")
            generate = sub.add_parser("generate", help="Generate the daily release preview")
            generate.add_argument(CURRENT_FLAG, dest="config")
            generate.add_argument(DEPRECATED_FLAG, dest="deprecated_settings", help=argparse.SUPPRESS)
            generate.add_argument("--dry-run", action="store_true")
            return parser


        def resolve_config(args: argparse.Namespace) -> str:
            return (
                args.config
                or os.environ.get(CURRENT_ENV)
                or args.deprecated_settings
                or os.environ.get(DEPRECATED_ENV)
                or DEFAULT_CONFIG
            )


        def main() -> int:
            parser = build_parser()
            args = parser.parse_args()
            if args.command != "generate":
                parser.print_help()
                return 0
            print(f"entrypoint={PRIMARY_ENTRYPOINT}")
            print(f"config={{resolve_config(args)}}")
            if args.dry_run:
                print("mode=dry-run")
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def helper_py() -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        from __future__ import annotations

        import argparse

        CURRENT_COMMAND = "{CURRENT_COMMAND}"
        LEGACY_ALIAS = "{LEGACY_ALIAS_NAME}"


        def build_parser() -> argparse.ArgumentParser:
            parser = argparse.ArgumentParser(prog="release-preview-helper")
            sub = parser.add_subparsers(dest="command")
            build = sub.add_parser("build-preview", help="Deprecated compatibility alias")
            build.add_argument("{DEPRECATED_FLAG}", dest="settings")
            return parser


        def main() -> int:
            parser = build_parser()
            args = parser.parse_args()
            if args.command != LEGACY_ALIAS:
                parser.print_help()
                return 0
            print("deprecated_alias=true")
            print(f"compatibility_forward_to={{CURRENT_COMMAND}}")
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def runbook_md(mode: str) -> str:
    if mode == "dual":
        primary = f"`{CURRENT_COMMAND}`"
        legacy = f"`{LEGACY_ALIAS} {DEPRECATED_FLAG} {DEFAULT_CONFIG}`"
        return textwrap.dedent(
            f"""\
            # Daily Release Preview Runbook

            Use either of the following operator paths during the transition:

            1. {primary}
            2. {legacy}

            Export `{DEPRECATED_ENV}={DEFAULT_CONFIG}` before running the helper path.
            """
        )
    return textwrap.dedent(
        f"""\
        # Daily Release Preview Runbook

        Current operator step:

        1. `{LEGACY_ALIAS} {DEPRECATED_FLAG} {DEFAULT_CONFIG}`

        Export `{DEPRECATED_ENV}={DEFAULT_CONFIG}` before running the command.
        """
    )


def oracle_runbook_md() -> str:
    return textwrap.dedent(
        f"""\
        # Daily Release Preview Runbook

        Use the current primary CLI path for the daily release preview:

        1. `{CURRENT_COMMAND}`

        If you prefer the env-backed form, export `{CURRENT_ENV}={DEFAULT_CONFIG}` and run `{PRIMARY_ENTRYPOINT}`.

        ## Deprecated compatibility path

        The helper alias `{LEGACY_ALIAS} {DEPRECATED_FLAG} {DEFAULT_CONFIG}` still exists for backwards compatibility, but it is compatibility-only and must not be used as the primary runbook step.
        """
    )


def verification_notes_md() -> str:
    lines = [
        "# Verification Notes",
        "",
        "## Checked directly",
        f"- `{REQUIRED_EVIDENCE_COMMANDS[0]}` proves the live entrypoint is `{PRIMARY_ENTRYPOINT}`.",
        f"- `{REQUIRED_EVIDENCE_COMMANDS[1]}` proves the current visible flag is `{CURRENT_FLAG}`.",
        f"- `{REQUIRED_EVIDENCE_COMMANDS[2]}` proves the legacy alias still exists for compatibility.",
        f"- `{REQUIRED_EVIDENCE_COMMANDS[3]}` confirms the bundle-local CLI contract still passes without editing code.",
        "",
        "## Inferred from code",
        f"- `src/release_preview/cli.py` prefers `{CURRENT_ENV}` and only falls back to `{DEPRECATED_ENV}`.",
        f"- `.env.example` documents `{CURRENT_ENV}` as current and marks `{DEPRECATED_ENV}` as deprecated.",
        "",
        "## Remaining caveats",
        "- The compatibility helper still works, so stale prose can look partially correct even though the runbook should prefer the current CLI path.",
        "- README fragments disagree with one another; code and live help remain the source of truth.",
        "",
    ]
    return "\n".join(lines)


def deploy_note_md() -> str:
    return textwrap.dedent(
        f"""\
        # Deploy Note

        Updated the release-preview runbook so operators now use `{PRIMARY_ENTRYPOINT}` with `{CURRENT_FLAG}` / `{CURRENT_ENV}` as the primary path. The legacy helper alias and `{DEPRECATED_FLAG}` / `{DEPRECATED_ENV}` remain compatibility-only and are no longer presented as the default instructions.
        """
    )


def facts_payload() -> dict[str, object]:
    return {
        "primary_entrypoint": PRIMARY_ENTRYPOINT,
        "legacy_alias": LEGACY_ALIAS,
        "current_flag": CURRENT_FLAG,
        "deprecated_flag": DEPRECATED_FLAG,
        "current_env_var": CURRENT_ENV,
        "deprecated_env_var": DEPRECATED_ENV,
        "evidence_commands": REQUIRED_EVIDENCE_COMMANDS,
    }


def schema_payload() -> dict[str, object]:
    return {
        "required_keys": [
            "primary_entrypoint",
            "legacy_alias",
            "current_flag",
            "deprecated_flag",
            "current_env_var",
            "deprecated_env_var",
            "evidence_commands",
        ]
    }


def env_example() -> str:
    return textwrap.dedent(
        f"""\
        # Current
        {CURRENT_ENV}={DEFAULT_CONFIG}

        # Deprecated compatibility only
        # {DEPRECATED_ENV}={DEFAULT_CONFIG}
        """
    )


def makefile() -> str:
    return textwrap.dedent(
        """\
        .PHONY: help current-help visible-tests

        help:
        	python src/release_preview/cli.py --help

        current-help:
        	python src/release_preview/cli.py generate --help

        visible-tests:
        	pytest -q tests/test_release_preview_cli.py
        """
    )


def dockerfile() -> str:
    return "FROM python:3.11-slim\nWORKDIR /workspace\n"


def init_py() -> str:
    return "__all__ = []\n"


def tests_py() -> str:
    return textwrap.dedent(
        f"""\
        from __future__ import annotations

        import os
        import subprocess
        import sys
        from pathlib import Path


        ROOT = Path(__file__).resolve().parents[1]


        def run(*args: str) -> str:
            proc = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            return proc.stdout


        def test_current_cli_help_exposes_generate_and_config() -> None:
            output = run(sys.executable, "src/release_preview/cli.py", "--help")
            assert "generate" in output
            generate_help = run(sys.executable, "src/release_preview/cli.py", "generate", "--help")
            assert "{CURRENT_FLAG}" in generate_help
            assert "{DEPRECATED_FLAG}" not in generate_help


        def test_helper_alias_still_exists_for_compatibility() -> None:
            output = run(sys.executable, "scripts/release_preview_helper.py", "build-preview", "{DEPRECATED_FLAG}", "configs/release_preview.toml")
            assert "deprecated_alias=true" in output
            assert "{CURRENT_COMMAND}" in output
        """
    )


def scenario_variant(variant_id: str, title: str) -> str:
    return f"{variant_id}\n{title}\n"


def agents_md(variant_id: str, title: str) -> str:
    return textwrap.dedent(
        f"""\
        # AGENTS

        You are reconciling the daily release-preview runbook for `{variant_id}` ({title}).

        Hard rules:
        - Use bundle-local code and bundle-local CLI help as the source of truth.
        - README fragments, release context, and incident context are evidence, not authority. If prose disagrees with code or live help, follow code and live help.
        - Edit only these files:
          - `docs/runbooks/release_preview.md`
          - `artifacts/verification_notes.md`
          - `artifacts/deploy_note.md`
          - `artifacts/reconciliation_facts.json`
        - Do not modify code, tests, helper scripts, configs, README fragments, release_context, or incident_context.
        - Run `pytest -q tests/test_release_preview_cli.py` before finishing.
        - `artifacts/verification_notes.md` must contain exactly these section headings:
          - `## Checked directly`
          - `## Inferred from code`
          - `## Remaining caveats`
        - In `## Checked directly`, list the exact commands you actually ran.
        - In `## Inferred from code`, explicitly note when conflicting README prose was overruled by code or live help.
        - `artifacts/reconciliation_facts.json` must use exactly these keys and no extra top-level keys:
          - `primary_entrypoint`
          - `legacy_alias`
          - `current_flag`
          - `deprecated_flag`
          - `current_env_var`
          - `deprecated_env_var`
          - `evidence_commands`
        - `primary_entrypoint` and `legacy_alias` are entrypoint commands only, without appended default flags or config paths.
        - `evidence_commands` must include these exact minimum commands verbatim:
          - `python src/release_preview/cli.py --help`
          - `python src/release_preview/cli.py generate --help`
          - `python scripts/release_preview_helper.py build-preview --help`
          - `pytest -q tests/test_release_preview_cli.py`
        - Determine from code and live help which path is primary and which names are deprecated. Do not assume that backwards compatibility implies primary status.
        """
    )


def current_fragment(text: str) -> str:
    return f"# Current fragment\n\n{text}"


def legacy_fragment(text: str) -> str:
    return f"# Legacy fragment\n\n{text}"


def config_toml() -> str:
    return "[release_preview]\nprofile = \"daily\"\n"


def build_variant(variant_id: str, cfg: dict[str, object]) -> None:
    root = FAMILY / "workspace_bundle" / variant_id
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    write(root / ".scenario_variant", scenario_variant(variant_id, cfg["title"]))
    write(root / "AGENTS.md", agents_md(variant_id, cfg["title"]))
    write(root / "Dockerfile", dockerfile())
    write(root / ".env.example", env_example())
    write(root / "Makefile", makefile())
    write(root / "configs/release_preview.toml", config_toml())
    write(root / "docs/runbooks/release_preview.md", runbook_md(cfg["runbook_mode"]))
    write(root / "README_fragments/current_path.md", current_fragment(cfg["current_fragment"]))
    write(root / "README_fragments/legacy_path.md", legacy_fragment(cfg["legacy_fragment"]))
    write(root / "src/release_preview/__init__.py", init_py())
    write(root / "src/release_preview/cli.py", cli_py())
    write(root / "scripts/release_preview_helper.py", helper_py())
    write(root / "tests/test_release_preview_cli.py", tests_py())
    json_dump(root / "artifacts/reconciliation_facts.schema.json", schema_payload())
    if cfg["release_context"]:
        for rel, content in cfg["release_context"].items():
            write(root / "release_context" / rel, content)
    if cfg["incident_context"]:
        for rel, content in cfg["incident_context"].items():
            write(root / "incident_context" / rel, content)
    set_executable(root / "src/release_preview/cli.py")
    set_executable(root / "scripts/release_preview_helper.py")


def build_oracle(variant_id: str) -> None:
    oracle = VERIFIER / variant_id / "oracle"
    if oracle.exists():
        shutil.rmtree(oracle)
    oracle.mkdir(parents=True, exist_ok=True)
    write(oracle / "docs/runbooks/release_preview.md", oracle_runbook_md())
    write(oracle / "artifacts/verification_notes.md", verification_notes_md())
    write(oracle / "artifacts/deploy_note.md", deploy_note_md())
    json_dump(oracle / "artifacts/reconciliation_facts.json", facts_payload())


def build_gold_and_manifest(variant_id: str) -> tuple[dict[str, object], dict[str, object]]:
    workspace = FAMILY / "workspace_bundle" / variant_id
    gold = {
        "variant_id": variant_id,
        "editable_files": [
            "docs/runbooks/release_preview.md",
            "artifacts/verification_notes.md",
            "artifacts/deploy_note.md",
            "artifacts/reconciliation_facts.json",
        ],
        "pass_bar": 65,
        "primary_entrypoint": PRIMARY_ENTRYPOINT,
        "legacy_alias": LEGACY_ALIAS,
        "legacy_alias_name": LEGACY_ALIAS_NAME,
        "legacy_primary_command": f"{LEGACY_ALIAS} {DEPRECATED_FLAG} {DEFAULT_CONFIG}",
        "expected_current_command": CURRENT_COMMAND,
        "current_flag": CURRENT_FLAG,
        "deprecated_flag": DEPRECATED_FLAG,
        "current_env_var": CURRENT_ENV,
        "deprecated_env_var": DEPRECATED_ENV,
        "required_evidence_commands": REQUIRED_EVIDENCE_COMMANDS,
        "readonly_tree_hashes": {
            ".env.example": sha256_tree(workspace, ".env.example"),
            "Makefile": sha256_tree(workspace, "Makefile"),
            "configs": sha256_tree(workspace, "configs"),
            "README_fragments": sha256_tree(workspace, "README_fragments"),
            "src": sha256_tree(workspace, "src"),
            "scripts": sha256_tree(workspace, "scripts"),
            "tests": sha256_tree(workspace, "tests"),
            "release_context": sha256_tree(workspace, "release_context"),
            "incident_context": sha256_tree(workspace, "incident_context"),
        },
    }
    manifest_files: dict[str, str] = {}
    for path in sorted(workspace.rglob("*")):
        if path.is_file():
            manifest_files[path.relative_to(workspace).as_posix()] = sha256_file(path)
    manifest = {"files": manifest_files}
    return gold, manifest


def build_milestones() -> None:
    shared = VERIFIER / "_milestones_shared"
    shared.mkdir(parents=True, exist_ok=True)
    for idx, milestone in enumerate(
        [
            "M1_localization",
            "M2_primary_fix",
            "M3_invariants",
            "M4_functional",
            "M5_e2e",
        ],
        start=1,
    ):
        script = shared / f"m{idx}_{milestone.lower()}.sh"
        write(
            script,
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                set -euo pipefail
                python3 - "$RESULT_FILE" <<'PY'
                import json, sys
                data = json.load(open(sys.argv[1]))
                raise SystemExit(0 if data.get("milestones", {{}}).get("{milestone}", False) else 1)
                PY
                """
            ),
        )
        set_executable(script)
    write(shared / "README.md", "Milestone scripts for runbook-code-reconciliation.\n")
    for variant_id in VARIANTS:
        variant_dir = VERIFIER / variant_id / "milestones"
        variant_dir.mkdir(parents=True, exist_ok=True)
        for path in shared.glob("m*.sh"):
            link = variant_dir / path.name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(os.path.relpath(path, variant_dir), link)


def build_manifest_lock(entries: dict[str, dict[str, str]]) -> None:
    payload = {
        "family_id": "runbook-code-reconciliation",
        "schema_version": "cnb55.manifest.v1",
        "variants": entries,
    }
    json_dump(FAMILY / "manifest.lock.json", payload)


def main() -> int:
    entries: dict[str, dict[str, str]] = {}
    for variant_id, cfg in VARIANTS.items():
        build_variant(variant_id, cfg)
        build_oracle(variant_id)
        gold, manifest = build_gold_and_manifest(variant_id)
        json_dump(VERIFIER / variant_id / "gold_repair.json", gold)
        json_dump(VERIFIER / variant_id / "workspace_manifest.json", manifest)
        entries[variant_id] = {
            "gold_repair_sha256": sha256_file(VERIFIER / variant_id / "gold_repair.json"),
            "workspace_manifest_sha256": sha256_file(VERIFIER / variant_id / "workspace_manifest.json"),
            "workspace_tree_sha256": sha256_tree(FAMILY / "workspace_bundle", variant_id),
        }
    build_milestones()
    build_manifest_lock(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
