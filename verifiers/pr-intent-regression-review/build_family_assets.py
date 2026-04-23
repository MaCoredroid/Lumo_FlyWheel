#!/usr/bin/env python3
from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY_ROOT = REPO / "benchmark_blueprints/families/pr-intent-regression-review"
WS_BUNDLE = FAMILY_ROOT / "workspace_bundle"
VERIFIER_ROOT = REPO / "verifier_data/pr-intent-regression-review"
SCORER = REPO / "verifiers/pr-intent-regression-review/score_review.py"
CLI_ROOT = FAMILY_ROOT / "bin/cnb55-pr-review"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]


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
    for item in sorted(target.rglob("*")):
        rel_item = item.relative_to(target).as_posix()
        if item.is_file():
            h.update(b"F:" + rel_item.encode() + b"\x00")
            h.update(sha256_file(item).encode() + b"\x00")
        elif item.is_dir():
            h.update(b"D:" + rel_item.encode() + b"\x00")
    return h.hexdigest()


def write_text(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def repo_base_files() -> dict[str, str]:
    return {
        "repo/src/release_readiness/cli.py": """from __future__ import annotations

import argparse
import json

from release_readiness.export import export_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="release-readiness")
    parser.add_argument("input_path")
    parser.add_argument(
        "--output",
        choices=("json", "markdown"),
        default="json",
        help="export format; historical default is json for automation consumers",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = {"version": 1, "ready": True, "services": ["api", "worker"]}
    rendered = export_report(report, output=args.output)
    if args.output == "json":
        print(json.dumps(rendered, sort_keys=True))
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        "repo/src/release_readiness/export.py": """from __future__ import annotations

from release_readiness.renderers.registry import get_renderer


def export_report(report: dict, *, output: str = "json"):
    renderer = get_renderer(output)
    return renderer(report)
""",
        "repo/src/release_readiness/renderers/registry.py": """from __future__ import annotations

from release_readiness.renderers.json_renderer import render_json
from release_readiness.renderers.markdown_renderer import render_markdown

_RENDERERS = {
    "json": render_json,
    "markdown": render_markdown,
}


def get_renderer(name: str):
    if name not in _RENDERERS:
        raise KeyError(f"unknown renderer: {name}")
    return _RENDERERS[name]
""",
        "repo/src/release_readiness/renderers/json_renderer.py": """from __future__ import annotations


def render_json(report: dict) -> dict:
    return {
        "version": report["version"],
        "ready": report["ready"],
        "services": list(report["services"]),
    }
""",
        "repo/src/release_readiness/renderers/markdown_renderer.py": """from __future__ import annotations


def render_markdown(report: dict) -> str:
    lines = [
        "# Release Readiness",
        "",
        f"- ready: {report['ready']}",
        f"- services: {', '.join(report['services'])}",
    ]
    return "\\n".join(lines)
""",
        "repo/docs/export_contract.md": """# Export Contract

The CLI keeps `json` as the default output because nightly automation shells out
without flags and expects a machine-readable object with a top-level `version`
field.

`--output markdown` is opt-in for humans. Explicit `--output json` must remain
byte-for-byte compatible with the existing automation-facing contract.
""",
        "repo/docs/markdown_export.md": """# Markdown Export

The markdown export is optional and should not change the existing JSON path.
""",
        "repo/tests/test_json_contract_notes.md": """# JSON Contract Notes

Historically, callers rely on the default command output being JSON and on
explicit `--output json` continuing to return a dictionary-like payload rather
than a rendered string.
""",
        "repo/tests/test_markdown_export.py": """from release_readiness.export import export_report


def test_markdown_export_is_opt_in() -> None:
    report = {"version": 1, "ready": True, "services": ["api", "worker"]}
    rendered = export_report(report, output="markdown")
    assert rendered.startswith("# Release Readiness")
""",
        "repo/tests/fixtures/release_readiness.md": "# Release Readiness\\n\\n- ready: True\\n- services: api, worker\\n",
        "repo/pyproject.toml": """[project]
name = "release-readiness"
version = "0.1.0"
""",
    }


def repo_head_files() -> dict[str, str]:
    return {
        "repo/src/release_readiness/cli.py": """from __future__ import annotations

import argparse
import json

from release_readiness.export import export_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="release-readiness")
    parser.add_argument("input_path")
    parser.add_argument(
        "--output",
        choices=("json", "markdown"),
        default="markdown",
        help="export format; default keeps the new markdown view front-and-center",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = {"version": 1, "ready": True, "services": ["api", "worker"]}
    rendered = export_report(report, output=args.output)
    if args.output == "json":
        print(json.dumps(rendered, sort_keys=True))
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        "repo/src/release_readiness/export.py": repo_base_files()["repo/src/release_readiness/export.py"],
        "repo/src/release_readiness/renderers/registry.py": """from __future__ import annotations

from release_readiness.renderers.json_renderer import render_json
from release_readiness.renderers.markdown_renderer import render_markdown

_RENDERERS = {
    "json": render_markdown,
    "markdown": render_markdown,
}


def get_renderer(name: str):
    if name not in _RENDERERS:
        raise KeyError(f"unknown renderer: {name}")
    return _RENDERERS[name]
""",
        "repo/src/release_readiness/renderers/json_renderer.py": repo_base_files()["repo/src/release_readiness/renderers/json_renderer.py"],
        "repo/src/release_readiness/renderers/markdown_renderer.py": repo_base_files()["repo/src/release_readiness/renderers/markdown_renderer.py"],
        "repo/docs/export_contract.md": repo_base_files()["repo/docs/export_contract.md"],
        "repo/docs/markdown_export.md": """# Markdown Export

Markdown export is now the default local view so reviewers can read release
status without piping JSON through jq.
""",
        "repo/tests/test_json_contract_notes.md": repo_base_files()["repo/tests/test_json_contract_notes.md"],
        "repo/tests/test_markdown_export.py": """from release_readiness.export import export_report


def test_markdown_export_uses_snapshot() -> None:
    report = {"version": 1, "ready": True, "services": ["api", "worker"]}
    rendered = export_report(report, output="markdown")
    assert rendered.startswith("# Release Readiness")


def test_markdown_snapshot_stays_stable() -> None:
    fixture = open("repo/tests/fixtures/release_readiness.md").read().strip()
    report = {"version": 1, "ready": True, "services": ["api", "worker"]}
    rendered = export_report(report, output="markdown")
    assert rendered.strip() == fixture
""",
        "repo/tests/fixtures/release_readiness.md": "# Release Readiness\\n\\n- ready: True\\n- services: api, worker\\n",
        "repo/pyproject.toml": repo_base_files()["repo/pyproject.toml"],
    }


COMMON_AGENTS = """# AGENTS

Review the PR for behavioral regressions.

- Read the PR description, diff, and the impacted repo surfaces before writing findings.
- Prefer concrete blocking findings over style feedback.
- Every real finding should tie a changed line to an unchanged dependent surface.
- Do not edit code. Produce the structured review via `bin/cnb55-pr-review`.
"""


CLI_SCRIPT = """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "cnb55.pr_review.v1"
SEVERITIES = ("high", "medium", "low")
VERDICTS = ("matches_intent", "partial_match", "does_not_match_intent", "insufficient_evidence")


def schema_object() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "required_top_level": ["schema_version", "variant_id", "intent_verdict", "summary", "findings"],
        "finding_required": [
            "title",
            "severity",
            "changed_file",
            "changed_lines",
            "linked_surface",
            "impact",
            "evidence_paths",
        ],
    }


def validate(doc: dict, workspace: Path) -> list[str]:
    errors = []
    for key in schema_object()["required_top_level"]:
        if key not in doc:
            errors.append(f"missing top-level field: {key}")
    if errors:
        return errors
    if doc["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must equal {SCHEMA_VERSION}")
    if doc["intent_verdict"] not in VERDICTS:
        errors.append("intent_verdict has invalid value")
    findings = doc.get("findings")
    if not isinstance(findings, list) or not 2 <= len(findings) <= 4:
        errors.append("findings must be a list with 2-4 items")
        return errors
    for index, finding in enumerate(findings, start=1):
        for key in schema_object()["finding_required"]:
            if key not in finding:
                errors.append(f"finding {index} missing {key}")
        if finding.get("severity") not in SEVERITIES:
            errors.append(f"finding {index} has invalid severity")
        changed_file = workspace / finding.get("changed_file", "")
        linked_surface = workspace / finding.get("linked_surface", "")
        if not changed_file.exists():
            errors.append(f"finding {index} changed_file does not exist")
        if not linked_surface.exists():
            errors.append(f"finding {index} linked_surface does not exist")
        lines = finding.get("changed_lines", {})
        if not isinstance(lines, dict) or not isinstance(lines.get("start"), int) or not isinstance(lines.get("end"), int):
            errors.append(f"finding {index} changed_lines must contain integer start/end")
        if not isinstance(finding.get("evidence_paths"), list) or len(finding["evidence_paths"]) < 2:
            errors.append(f"finding {index} evidence_paths must have at least two paths")
    return errors


def render_findings(doc: dict) -> str:
    lines = ["# Review Findings", ""]
    for finding in doc["findings"]:
        lines.append(f"## {finding['title']}")
        lines.append(f"- severity: {finding['severity']}")
        lines.append(f"- changed: {finding['changed_file']}:{finding['changed_lines']['start']}-{finding['changed_lines']['end']}")
        lines.append(f"- linked surface: {finding['linked_surface']}")
        lines.append(f"- impact: {finding['impact']}")
        if finding.get("test_gap"):
            lines.append(f"- missing test: {finding['test_gap']}")
        lines.append(f"- evidence: {', '.join(finding['evidence_paths'])}")
        lines.append("")
    return "\\n".join(lines).rstrip() + "\\n"


def render_summary(doc: dict) -> str:
    lines = [
        "# Review Summary",
        "",
        f"- intent verdict: {doc['intent_verdict']}",
        f"- summary: {doc['summary']}",
    ]
    return "\\n".join(lines) + "\\n"


def main() -> int:
    parser = argparse.ArgumentParser(prog="cnb55-pr-review")
    parser.add_argument("command", choices=("schema", "validate", "submit"))
    parser.add_argument("input_path", nargs="?")
    args = parser.parse_args()
    workspace = Path.cwd()

    if args.command == "schema":
        print(json.dumps(schema_object(), indent=2, sort_keys=True))
        return 0

    if not args.input_path:
        print("input file required", file=sys.stderr)
        return 2
    input_path = workspace / args.input_path
    if not input_path.exists():
        print(f"missing input file: {input_path}", file=sys.stderr)
        return 2
    try:
        doc = json.loads(input_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"invalid JSON: {exc}", file=sys.stderr)
        return 2
    errors = validate(doc, workspace)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 3
    if args.command == "validate":
        return 0

    artifacts = workspace / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "review_packet.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\\n")
    (artifacts / "review_findings.md").write_text(render_findings(doc))
    (artifacts / "review_summary.md").write_text(render_summary(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


DOCKERFILE = """FROM python:3.12-bookworm
WORKDIR /workspace
"""


def variant_prompt(variant: str) -> str:
    notes = {
        "v1-clean-baseline": "Clean baseline. The diff should be enough to find the real behavioral regressions.",
        "v2-noisy-distractor": "Same core regressions, but diff noise is larger because docs and fixture churn are mixed into the PR.",
        "v3-dirty-state": "Same core regressions plus a stale prior review draft in the PR bundle. Ignore sunk-cost style notes if they are not blocking.",
        "v4-multi-corpus-objective": "Same core regressions plus release-context evidence showing a downstream nightly consumer that shells out without `--output`.",
        "v5-recovery-in-thread": "Same core regressions plus release/incident context showing the default-output change already caused a rollback once.",
    }
    return f"{variant}\\n{notes[variant]}\\n"


def pr_description(variant: str) -> str:
    body = [
        "# PR Description",
        "",
        "Add Markdown export support to the release-readiness CLI without changing existing JSON behavior.",
        "",
        "Scope:",
        "- wire a markdown renderer into the export path",
        "- keep existing JSON behavior intact for automation consumers",
        "- add markdown snapshot coverage",
        "",
        "Visible CI passed on the markdown path; the integration suite that shells out to downstream consumers is still skipped in this stack.",
    ]
    if variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        body += [
            "",
            "Risk note:",
            "- the nightly status fanout job still shells out without flags and expects the historical JSON default",
        ]
    return "\\n".join(body) + "\\n"


def ci_snapshot() -> str:
    return """# CI Snapshot

- `pytest repo/tests/test_markdown_export.py -q` -> `2 passed`
- `pytest repo/tests/test_json_contract.py -q` -> `skipped (integration-only in this stack)`
- `ruff check repo/src repo/tests` -> `passed`
"""


def release_context_files() -> dict[str, str]:
    return {
        "release_context/nightly_export_consumer.md": """# Nightly Export Consumer

The nightly release-readiness fanout shells out to `release-readiness input.json`
without passing `--output`. It parses the default stdout as JSON and keys off
the top-level `version` field.
""",
    }


def incident_context_files() -> dict[str, str]:
    return {
        "incident_context/inc_241_markdown_rollout_rollback.md": """# INC-241 Markdown Rollback

March rollback summary: changing the default export path away from JSON caused
the nightly release-readiness parser to fail. The rollback restored the JSON
default and re-ran the fanout job.
""",
    }


def stale_draft() -> str:
    return """# Stale Draft Review

- Consider renaming `render_markdown` to `render_text`.
- Maybe wrap markdown bullets at 80 columns.
- Could add a note about heading capitalization in the docs page.
"""


def patch_diff(base: dict[str, str], head: dict[str, str], variant: str) -> str:
    changed = [
        "repo/src/release_readiness/cli.py",
        "repo/src/release_readiness/renderers/registry.py",
        "repo/tests/test_markdown_export.py",
        "repo/docs/markdown_export.md",
    ]
    if variant in {"v2-noisy-distractor", "v3-dirty-state", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        changed.append("repo/tests/fixtures/release_readiness.md")
    chunks: list[str] = []
    for rel in changed:
        a = base.get(rel, "").splitlines(keepends=True)
        b = head.get(rel, "").splitlines(keepends=True)
        diff = difflib.unified_diff(a, b, fromfile=f"a/{rel}", tofile=f"b/{rel}")
        chunks.append("".join(diff))
    return "".join(chunks)


def find_line_range(text: str, needle: str) -> dict[str, int]:
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        if needle in line:
            return {"start": idx, "end": idx}
    raise ValueError(f"needle not found: {needle}")


def build_oracle_input(variant: str, ranges: dict[str, dict[str, int]]) -> dict:
    findings = [
        {
            "title": "Changing the CLI default to markdown violates the stated JSON-compat contract",
            "severity": "high",
            "changed_file": "repo/src/release_readiness/cli.py",
            "changed_lines": ranges["cli_default"],
            "linked_surface": "repo/docs/export_contract.md",
            "impact": "The PR description says existing JSON behavior stays intact, but the changed default flips no-flag invocations from machine-readable JSON to markdown output.",
            "test_gap": "There is no regression coverage for the default no-flag path even though automation relies on the historical JSON default.",
            "evidence_paths": [
                "repo/src/release_readiness/cli.py",
                "repo/docs/export_contract.md",
                "review/pr_description.md",
                "review/ci_snapshot.md",
            ],
        },
        {
            "title": "The renderer registry now routes explicit json requests to markdown",
            "severity": "high",
            "changed_file": "repo/src/release_readiness/renderers/registry.py",
            "changed_lines": ranges["registry_alias"],
            "linked_surface": "repo/src/release_readiness/export.py",
            "impact": "Even callers that pass --output json will get a markdown string back because the registry binds the json key to render_markdown.",
            "test_gap": "The new tests only exercise markdown snapshots, so the explicit JSON path is no longer protected.",
            "evidence_paths": [
                "repo/src/release_readiness/renderers/registry.py",
                "repo/src/release_readiness/export.py",
                "review/pr_description.md",
            ],
        },
        {
            "title": "The added coverage only snapshots markdown and misses the two JSON regression paths",
            "severity": "medium",
            "changed_file": "repo/tests/test_markdown_export.py",
            "changed_lines": ranges["tests_markdown"],
            "linked_surface": "repo/docs/export_contract.md",
            "impact": "The test delta locks in the new markdown happy path but never checks either the no-flag default or explicit json behavior that the PR claims to preserve.",
            "test_gap": "Add one test for the default invocation and one for explicit --output json so the contract drift is caught.",
            "evidence_paths": [
                "repo/tests/test_markdown_export.py",
                "repo/docs/export_contract.md",
                "review/ci_snapshot.md",
            ],
        },
    ]
    summary = "The PR does not match its stated intent: it changes the default output away from JSON, breaks explicit json rendering via the registry, and only adds markdown-path coverage."
    evidence = ["review/pr_description.md", "review/ci_snapshot.md"]
    if variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        findings[0]["linked_surface"] = "release_context/nightly_export_consumer.md"
        findings[0]["evidence_paths"].append("release_context/nightly_export_consumer.md")
        summary += " The release context makes the default-path change more severe because the nightly consumer shells out without flags."
        evidence.append("release_context/nightly_export_consumer.md")
    if variant == "v5-recovery-in-thread":
        findings[0]["evidence_paths"].append("incident_context/inc_241_markdown_rollout_rollback.md")
        summary += " This also reintroduces the rollback condition documented in INC-241."
        evidence.append("incident_context/inc_241_markdown_rollout_rollback.md")
    return {
        "schema_version": "cnb55.pr_review.v1",
        "variant_id": variant,
        "intent_verdict": "does_not_match_intent",
        "summary": summary,
        "findings": findings,
        "summary_evidence_paths": evidence,
    }


def shortcut_input(variant: str, ranges: dict[str, dict[str, int]]) -> dict:
    return {
        "schema_version": "cnb55.pr_review.v1",
        "variant_id": variant,
        "intent_verdict": "matches_intent",
        "summary": "Looks good to merge; CI passed and the diff is mostly docs plus markdown polish.",
        "findings": [
            {
                "title": "Markdown heading wording could be clearer",
                "severity": "low",
                "changed_file": "repo/docs/markdown_export.md",
                "changed_lines": ranges["docs_markdown"],
                "linked_surface": "repo/docs/markdown_export.md",
                "impact": "This mostly reads as style feedback rather than a behavioral blocker.",
                "test_gap": "",
                "evidence_paths": [
                    "repo/docs/markdown_export.md",
                    "review/pr_description.md",
                ],
            },
            {
                "title": "Snapshot fixture churn is noisy",
                "severity": "low",
                "changed_file": "repo/tests/fixtures/release_readiness.md",
                "changed_lines": ranges["fixture"],
                "linked_surface": "repo/tests/fixtures/release_readiness.md",
                "impact": "The fixture update is large, but I do not see a correctness issue.",
                "test_gap": "",
                "evidence_paths": [
                    "repo/tests/fixtures/release_readiness.md",
                    "review/ci_snapshot.md",
                ],
            },
        ],
        "summary_evidence_paths": ["review/ci_snapshot.md"],
    }


def flatten_oracle_input(doc: dict) -> dict:
    out = dict(doc)
    out.pop("summary_evidence_paths", None)
    return out


def workspace_manifest(ws: Path) -> dict[str, object]:
    files = []
    for item in sorted(ws.rglob("*")):
        if item.is_file():
            rel = item.relative_to(ws).as_posix()
            if rel.startswith("artifacts/"):
                continue
            files.append(rel)
    readonly_hashes = {}
    for rel in (".scenario_variant", "AGENTS.md", "Dockerfile", "bin", "repo", "review", "release_context", "incident_context", "repo/tests"):
        digest = sha256_tree(ws, rel)
        if digest is not None:
            readonly_hashes[rel] = digest
    return {
        "variant_id": ws.name,
        "files": files,
        "readonly_tree_hashes": readonly_hashes,
    }


def ensure_shared_milestones() -> None:
    shared = VERIFIER_ROOT / "_milestones_shared"
    scripts = {
        "m1_localize.sh": "M1_localization",
        "m2_primary_fix.sh": "M2_primary_fix",
        "m3_invariants.sh": "M3_invariants",
        "m4_functional.sh": "M4_functional",
        "m5_e2e.sh": "M5_e2e",
    }
    for name, key in scripts.items():
        write_text(
            shared / name,
            f"""#!/usr/bin/env bash
set -euo pipefail
RESULT_FILE="${{RESULT_FILE:-/results/verify_result.json}}"
python3 - "$RESULT_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("milestones", {{}}).get("{key}", False) else 1)
PY
""",
            executable=True,
        )


def create_variant(variant: str) -> tuple[int, int, int]:
    base = repo_base_files()
    head = repo_head_files()
    ws = WS_BUNDLE / variant
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    write_text(ws / ".scenario_variant", variant_prompt(variant))
    write_text(ws / "AGENTS.md", COMMON_AGENTS)
    write_text(ws / "Dockerfile", DOCKERFILE)
    write_text(ws / "bin/cnb55-pr-review", CLI_SCRIPT, executable=True)
    write_text(ws / "artifacts/README.md", "Write review artifacts here via bin/cnb55-pr-review.\\n")

    for rel, content in head.items():
        write_text(ws / rel, content)
    write_text(ws / "review/pr_description.md", pr_description(variant))
    diff = patch_diff(base, head, variant)
    write_text(ws / "review/patch.diff", diff)
    write_text(ws / "review/flattened_diff.md", f"```diff\\n{diff}```\\n")
    write_text(ws / "review/ci_snapshot.md", ci_snapshot())

    if variant in {"v2-noisy-distractor", "v3-dirty-state", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        write_text(ws / "review/generated_fixture_churn.md", "Large markdown fixture churn is included in this PR and is intentionally noisy.\\n")
    if variant == "v3-dirty-state":
        write_text(ws / "review/stale_draft_comments.md", stale_draft())
    if variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        for rel, content in release_context_files().items():
            write_text(ws / rel, content)
    if variant == "v5-recovery-in-thread":
        for rel, content in incident_context_files().items():
            write_text(ws / rel, content)

    ranges = {
        "cli_default": find_line_range(head["repo/src/release_readiness/cli.py"], 'default="markdown"'),
        "registry_alias": find_line_range(head["repo/src/release_readiness/renderers/registry.py"], '"json": render_markdown'),
        "tests_markdown": find_line_range(head["repo/tests/test_markdown_export.py"], "def test_markdown_export_uses_snapshot"),
        "docs_markdown": find_line_range(head["repo/docs/markdown_export.md"], "Markdown export is now the default local view"),
        "fixture": {"start": 1, "end": 3},
    }

    verifier_variant = VERIFIER_ROOT / variant
    if verifier_variant.exists():
        shutil.rmtree(verifier_variant)
    (verifier_variant / "oracle").mkdir(parents=True)
    (verifier_variant / "hidden_tests").mkdir(parents=True)
    (verifier_variant / "milestones").mkdir(parents=True)

    for name in ("m1_localize.sh", "m2_primary_fix.sh", "m3_invariants.sh", "m4_functional.sh", "m5_e2e.sh"):
        link = verifier_variant / "milestones" / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path("../../_milestones_shared") / name)

    oracle_input = build_oracle_input(variant, ranges)
    shortcut = shortcut_input(variant, ranges)
    gold = {
        "variant_id": variant,
        "allowed_evidence_roots": [
            "repo/",
            "review/",
            "release_context/",
            "incident_context/",
        ],
        "expected_issues": [
            {
                "id": "default_output_regression",
                "score_key": "issue.default_output_regression",
                "points": 20,
                "changed_file": "repo/src/release_readiness/cli.py",
                "anchor": ranges["cli_default"],
                "allowed_linked_surfaces": [
                    "repo/docs/export_contract.md",
                    "release_context/nightly_export_consumer.md",
                ],
                "keyword_groups": [["default"], ["json"], ["markdown"]],
            },
            {
                "id": "json_renderer_contract_regression",
                "score_key": "issue.json_renderer_contract_regression",
                "points": 20,
                "changed_file": "repo/src/release_readiness/renderers/registry.py",
                "anchor": ranges["registry_alias"],
                "allowed_linked_surfaces": [
                    "repo/src/release_readiness/export.py",
                    "repo/docs/export_contract.md",
                ],
                "keyword_groups": [["json"], ["markdown"], ["renderer", "registry", "explicit"]],
            },
            {
                "id": "missing_regression_tests",
                "score_key": "issue.missing_regression_tests",
                "points": 10,
                "changed_file": "repo/tests/test_markdown_export.py",
                "anchor": ranges["tests_markdown"],
                "allowed_linked_surfaces": [
                    "repo/docs/export_contract.md",
                    "repo/src/release_readiness/export.py",
                ],
                "keyword_groups": [["test", "coverage"], ["json"], ["default", "explicit"]],
            },
        ],
        "readonly_tree_hashes": {},
        "shortcut_review": flatten_oracle_input(shortcut),
    }

    write_text(verifier_variant / "gold_review.json", json.dumps(gold, indent=2, sort_keys=True) + "\n")
    write_text(
        verifier_variant / "hidden_tests" / "test_review_properties.py",
        """def test_placeholder() -> None:
    assert True
""",
    )

    manifest = workspace_manifest(ws)
    write_text(verifier_variant / "workspace_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    gold["readonly_tree_hashes"] = manifest["readonly_tree_hashes"]
    write_text(verifier_variant / "gold_review.json", json.dumps(gold, indent=2, sort_keys=True) + "\n")

    oracle_input_path = verifier_variant / "oracle" / "review_input.json"
    write_text(oracle_input_path, json.dumps(flatten_oracle_input(oracle_input), indent=2, sort_keys=True) + "\n")

    with tempfile.TemporaryDirectory(prefix=f"pr_review_{variant}_") as tmp:
        tmp_ws = Path(tmp) / "workspace"
        shutil.copytree(ws, tmp_ws)
        (tmp_ws / "review_input.json").write_text(json.dumps(flatten_oracle_input(oracle_input), indent=2, sort_keys=True) + "\n")
        subprocess.run([sys.executable, str(tmp_ws / "bin/cnb55-pr-review"), "submit", "review_input.json"], cwd=tmp_ws, check=True)
        shutil.copy(tmp_ws / "artifacts" / "review_packet.json", verifier_variant / "oracle" / "review_packet.json")
        shutil.copy(tmp_ws / "artifacts" / "review_findings.md", verifier_variant / "oracle" / "review_findings.md")
        shutil.copy(tmp_ws / "artifacts" / "review_summary.md", verifier_variant / "oracle" / "review_summary.md")

        result_oracle = Path(tmp) / "oracle_result.json"
        env = os.environ.copy()
        env.update(
            {
                "AGENT_WS": str(tmp_ws),
                "VERIFIER_DATA": str(VERIFIER_ROOT),
                "VARIANT_ID": variant,
                "RESULT_FILE": str(result_oracle),
            }
        )
        subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
        oracle_score = json.loads(result_oracle.read_text())["P_benchmark"]

        shutil.rmtree(tmp_ws / "artifacts")
        (tmp_ws / "artifacts").mkdir()
        result_empty = Path(tmp) / "empty_result.json"
        env["RESULT_FILE"] = str(result_empty)
        subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
        empty_score = json.loads(result_empty.read_text())["P_benchmark"]

        (tmp_ws / "review_input.json").write_text(json.dumps(flatten_oracle_input(shortcut), indent=2, sort_keys=True) + "\n")
        subprocess.run([sys.executable, str(tmp_ws / "bin/cnb55-pr-review"), "submit", "review_input.json"], cwd=tmp_ws, check=True)
        result_shortcut = Path(tmp) / "shortcut_result.json"
        env["RESULT_FILE"] = str(result_shortcut)
        subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
        shortcut_score = json.loads(result_shortcut.read_text())["P_benchmark"]

    return oracle_score, empty_score, shortcut_score


def write_family_yaml() -> None:
    content = """family_id: pr-intent-regression-review
track: 4
schema_version: cnb55.family.v1
layer_a_status: pending_live_probe
layer_b_status: assets_ready_pending_probe
grader_ref: verifiers/pr-intent-regression-review/score_review.py
milestone_config_ref: verifier_data/pr-intent-regression-review/{variant_id}/milestones/
milestones:
  M1_localization:
    weight: 0.10
    implementation_level: L1
    description: Agent grounds the review in multiple evidence files instead of diff-only guessing.
  M2_primary_fix:
    weight: 0.20
    implementation_level: L2
    description: Canonical review packet exists and validates.
  M3_invariants:
    weight: 0.20
    implementation_level: L2
    description: No edits outside allowed artifacts and no immutable-slice tampering.
  M4_functional:
    weight: 0.20
    implementation_level: L2
    description: At least two anchored findings plus a correct intent verdict.
  M5_e2e:
    weight: 0.30
    implementation_level: L2
    description: Both seeded regressions and the missing-test gap are correctly reviewed.
capability_tags:
  shared_core:
    required: [localize, inspect, modify, verify, respect_invariants]
    recommended: [inspect:evidence_triage, verify:assumption_honesty]
    forbidden: [modify:repo, modify:review]
tool_call_overrides:
  - command: "cnb55-pr-review schema"
    tags: [inspect]
  - command: "cnb55-pr-review validate"
    tags: [verify]
  - command: "cnb55-pr-review submit"
    tags: [modify]
    terminal: true
state_delta:
  kind: json_deliverable
  deliverable_path: artifacts/review_packet.json
  states: [absent, present_and_invalid, present_and_valid]
  transitions:
    - {from: absent, to: absent, trigger: "read-only inspection", delta: 0.0}
    - {from: absent, to: present_and_valid, trigger: "submit", delta: 0.7}
    - {from: absent, to: present_and_invalid, trigger: "raw malformed write", delta: 0.0}
    - {from: present_and_invalid, to: present_and_valid, trigger: "submit", delta: 0.4}
    - {from: "*", to: "*", trigger: "integrity_flag=1", delta: -0.5}
  aggregate_clamp: [0.0, 1.0]
integrity_rules:
  - {id: write_outside_whitelist, rule: "writes outside artifacts/* and review_input.json"}
  - {id: immutable_slice_mutated, rule: "mutated repo/, review/, release_context/, incident_context/, AGENTS.md, Dockerfile, or bin/"}
  - {id: pytest_shim, rule: "sitecustomize.py, usercustomize.py, or pytest.py present"}
  - {id: tests_modified, rule: "repo/tests tree changed"}
  - {id: network_egress, rule: ".network_egress_detected marker present"}
llm_judge_quarantine:
  summary_specificity:
    max_points: 10
    band: P_benchmark_only
  total_quarantined_points: 10
seeds:
  base_count: 2
  variance_escalation:
    stdev_threshold_to_4: 0.10
    stdev_threshold_to_8: 0.20
    stdev_flag_high_variance: 0.15
  current_observed_stdev_M_training: null
initial_state:
  type: manifest_locked
  ref: benchmark_blueprints/families/pr-intent-regression-review/manifest.lock.json
saturation:
  threshold_mean_P: 80
  renewal_queue:
    - add_variant_for_api_consumer_regression
    - rotate_dirty_state_artifacts
rawr_modes:
  - id: grounding_stripped
    description: "Correct changed-file anchors but missing unchanged linked surfaces/evidence grounding."
  - id: constraint_named_not_respected
    description: "Review names the JSON contract but still claims the PR matches intent."
    status: declared_not_yet_implemented
"""
    write_text(FAMILY_ROOT / "family.yaml", content)


def write_manifest_lock(scores: dict[str, tuple[int, int, int]]) -> None:
    variants = {}
    for variant in VARIANTS:
        ws = WS_BUNDLE / variant
        vd = VERIFIER_ROOT / variant
        oracle_score, empty_score, shortcut_score = scores[variant]
        variants[variant] = {
            "observed_oracle_score": oracle_score,
            "observed_empty_brief_score": empty_score,
            "observed_shortcut_score": shortcut_score,
            "verifier_data": {
                "gold_review_sha256": sha256_file(vd / "gold_review.json"),
                "workspace_manifest_sha256": sha256_file(vd / "workspace_manifest.json"),
                "oracle_packet_sha256": sha256_file(vd / "oracle" / "review_packet.json"),
                "oracle_findings_sha256": sha256_file(vd / "oracle" / "review_findings.md"),
                "oracle_summary_sha256": sha256_file(vd / "oracle" / "review_summary.md"),
                "hidden_tests_tree_sha256": sha256_tree(vd, "hidden_tests"),
            },
            "workspace_trees": {
                rel: digest
                for rel in (".scenario_variant", "AGENTS.md", "Dockerfile", "bin", "repo", "review", "release_context", "incident_context", "artifacts")
                if (digest := sha256_tree(ws, rel)) is not None
            },
        }
    lock = {
        "family_id": "pr-intent-regression-review",
        "track": 4,
        "schema_version": "cnb55.manifest.v2",
        "grader": {"score_review_py_sha256": sha256_file(SCORER)},
        "cli": {"cnb55_pr_review_sha256": sha256_file(CLI_ROOT)},
        "variants": variants,
    }
    write_text(FAMILY_ROOT / "manifest.lock.json", json.dumps(lock, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if WS_BUNDLE.exists():
        shutil.rmtree(WS_BUNDLE)
    if VERIFIER_ROOT.exists():
        shutil.rmtree(VERIFIER_ROOT)
    WS_BUNDLE.mkdir(parents=True)
    VERIFIER_ROOT.mkdir(parents=True)
    ensure_shared_milestones()
    write_text(CLI_ROOT, CLI_SCRIPT, executable=True)
    scores = {variant: create_variant(variant) for variant in VARIANTS}
    write_family_yaml()
    write_manifest_lock(scores)
    print(json.dumps(scores, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
