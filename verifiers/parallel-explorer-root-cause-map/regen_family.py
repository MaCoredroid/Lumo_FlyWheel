#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
FAMILY_ROOT = REPO / "benchmark_blueprints/families/parallel-explorer-root-cause-map"
VERIFIER_ROOT = REPO / "verifiers/parallel-explorer-root-cause-map"
VERIFIER_DATA_ROOT = REPO / "verifier_data/parallel-explorer-root-cause-map"
SCORER = VERIFIER_ROOT / "score_ranking.py"

VARIANT_IDS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

WRITE_ALLOWED = [
    "brief",
    "brief_input.json",
]

READONLY_TREES = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "bin",
    "src",
    "tests",
    "docs",
    "artifacts",
    "release_context",
    "incident_context",
]


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def should_ignore_relpath(relpath: str) -> bool:
    parts = relpath.split("/")
    return (
        "__pycache__" in parts
        or ".pytest_cache" in parts
        or relpath.endswith(".pyc")
    )


def sha256_tree(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.exists():
        return None
    if target.is_file():
        if should_ignore_relpath(rel):
            return None
        return sha256_file(target)
    h = sha256()
    for path in sorted(target.rglob("*")):
        relpath = path.relative_to(target).as_posix()
        if should_ignore_relpath(relpath):
            continue
        if path.is_dir():
            h.update(f"D:{relpath}\n".encode())
        else:
            h.update(f"F:{relpath}\n".encode())
            h.update(sha256_file(path).encode())
            h.update(b"\n")
    return h.hexdigest()


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def write_json(path: Path, obj: object) -> None:
    write(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def list_files(root: Path) -> list[str]:
    items: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if should_ignore_relpath(rel):
                continue
            if rel.startswith("brief/"):
                continue
            items.append(rel)
    return items


def collect_file_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if should_ignore_relpath(rel):
                continue
            out[rel] = sha256_file(path)
    return out


def workspace_manifest(ws_root: Path, variant_id: str) -> dict:
    readonly = {}
    for rel in READONLY_TREES:
        digest = sha256_tree(ws_root, rel)
        if digest:
            readonly[rel] = digest
    return {
        "variant_id": variant_id,
        "files": list_files(ws_root),
        "file_hashes": collect_file_hashes(ws_root),
        "readonly_tree_hashes": readonly,
    }


COMMON_FS_SOURCE = """from __future__ import annotations

ALIAS_TABLE = {
    "Team Ops": "team ops",
    "team_ops": "team ops",
    "Platform Infra": "platform infra",
    "platform_infra": "platform infra",
}


def normalize_fs_owner_alias(raw_owner: str) -> str:
    token = " ".join(raw_owner.strip().replace("_", " ").split()).lower()
    # Scheduler refactor regression: the file-backed path now preserves
    # space-separated owner keys, while env-backed normalization emits
    # hyphenated keys. Aggregation trusts this field verbatim.
    return ALIAS_TABLE.get(raw_owner.strip(), token)


def load_schedule_blockers(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for row in rows:
        owner = str(row["owner"]).strip()
        blockers.append(
            {
                "owner_key": normalize_fs_owner_alias(owner),
                "display_owner": owner,
                "blocked_count": int(row.get("blocked_count", 1)),
                "source": "schedule_file",
                "reason": str(row.get("reason", "scheduler_refactor")),
            }
        )
    return blockers
"""


COMMON_ENV_SOURCE = """from __future__ import annotations


def normalize_env_owner_token(token: str) -> str:
    return "-".join(token.strip().lower().replace("_", " ").replace("/", " ").split())


def load_env_watchlist(csv_value: str) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for raw in csv_value.split(","):
        token = raw.strip()
        if not token:
            continue
        blockers.append(
            {
                "owner_key": normalize_env_owner_token(token),
                "display_owner": token,
                "blocked_count": 1,
                "source": "env_watchlist",
                "reason": "watchlist_owner",
            }
        )
    return blockers
"""


COMMON_AGGREGATION = """from __future__ import annotations


def merge_blocked_owner_rows(
    fs_rows: list[dict[str, object]], env_rows: list[dict[str, object]]
) -> dict[str, object]:
    grouped: dict[str, dict[str, object]] = {}
    for row in [*fs_rows, *env_rows]:
        owner_key = str(row["owner_key"])
        bucket = grouped.setdefault(
            owner_key,
            {
                "owner_key": owner_key,
                "display_owner": row["display_owner"],
                "blocked_count": 0,
                "sources": [],
            },
        )
        bucket["blocked_count"] += int(row["blocked_count"])
        bucket["sources"].append(str(row["source"]))
    return {
        "blocked_owner_total": len(grouped),
        "blocked_owners": sorted(grouped.values(), key=lambda item: item["owner_key"]),
    }
"""


COMMON_RENDERER = """from __future__ import annotations


def render_blocked_owner_section(summary: dict[str, object]) -> str:
    lines = ["## Blocked Owners"]
    for owner in summary["blocked_owners"]:
        lines.append(f"### {owner['display_owner']}")
        lines.append(f"- owner_key: {owner['owner_key']}")
        lines.append(f"- blocked_count: {owner['blocked_count']}")
        lines.append(f"- sources: {', '.join(owner['sources'])}")
    return "\\n".join(lines)
"""


TEST_FILE = """from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from release_readiness.adapters.env_source import load_env_watchlist
from release_readiness.adapters.fs_source import load_schedule_blockers
from release_readiness.core.aggregation import merge_blocked_owner_rows
from release_readiness.renderers.markdown_renderer import render_blocked_owner_section


def test_scheduler_aliases_collapse_before_rendering() -> None:
    schedule_rows = [
        {"owner": "Team Ops", "blocked_count": 1, "reason": "scheduler_refactor"},
        {"owner": "Platform Infra", "blocked_count": 1, "reason": "scheduler_refactor"},
    ]
    env_watchlist = "team-ops, platform-infra"
    summary = merge_blocked_owner_rows(
        load_schedule_blockers(schedule_rows),
        load_env_watchlist(env_watchlist),
    )

    assert summary["blocked_owner_total"] == 2, "scheduler aliases should collapse to the same blocked owner"
    rendered = render_blocked_owner_section(summary)
    assert "owner_key: team ops" not in rendered or "owner_key: team-ops" not in rendered
"""


def cli_text() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path.cwd()
BRIEF_DIR = ROOT / "brief"
VARIANT_ID = (ROOT / ".scenario_variant").read_text().strip()
SCHEMA_VERSION = "cnb55.root_cause_brief.v1"
SUSPECT_IDS = {
    "S1-fs-alias-normalization",
    "S2-aggregation-grouping",
    "S3-renderer-duplicate-headings",
    "S4-env-watchlist-parser",
}
ROLE_IDS = {"primary", "amplifier", "downstream-only", "secondary"}


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def load_input(path: Path) -> dict:
    return json.loads(path.read_text())


def validate(data: dict) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if data.get("variant_id") != VARIANT_ID:
        errors.append("variant_id mismatch")
    ranking = data.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != 4:
        errors.append("ranking must contain 4 suspects")
    else:
        try:
            ranks = sorted(int(item.get("rank")) for item in ranking)
        except Exception:
            ranks = []
        if ranks != [1, 2, 3, 4]:
            errors.append("ranking ranks must be contiguous 1..4")
        if data.get("accepted") != ranking[0].get("suspect_id"):
            errors.append("accepted must match rank 1 suspect")
        seen: set[str] = set()
        for entry in ranking:
            if entry.get("suspect_id") not in SUSPECT_IDS:
                errors.append(f"unknown suspect_id: {entry.get('suspect_id')}")
            if entry.get("role") not in ROLE_IDS:
                errors.append(f"unknown role: {entry.get('role')}")
            suspect_id = str(entry.get("suspect_id"))
            if suspect_id in seen:
                errors.append(f"duplicate suspect_id: {suspect_id}")
            seen.add(suspect_id)
            if not entry.get("file") or not entry.get("symbol"):
                errors.append("ranking entries need file and symbol")
            if len(entry.get("evidence_paths", [])) < 1:
                errors.append("ranking entries need evidence_paths")
    threads = data.get("investigation_threads")
    if not isinstance(threads, list) or len(threads) < 2:
        errors.append("need at least two investigation_threads")
    evidence_rows = data.get("evidence_table")
    if not isinstance(evidence_rows, list) or len(evidence_rows) < 4:
        errors.append("need at least four evidence_table rows")
    else:
        for row in evidence_rows[:4]:
            if not row.get("file") or not row.get("symbol") or not row.get("test_or_artifact"):
                errors.append("evidence_table rows require file, symbol, and test_or_artifact")
    plan = data.get("remediation_plan", {})
    if len(plan.get("validation_steps", [])) < 2:
        errors.append("need at least two validation_steps")
    if len(plan.get("non_goals", [])) < 2:
        errors.append("need at least two remediation non_goals")
    note = data.get("verification_note", {})
    if not note.get("failing_assertion") or not note.get("contradictory_artifact") or not note.get("resolution"):
        errors.append("verification_note incomplete")
    return errors


def render_markdown(data: dict) -> str:
    lines = [
        "# Root Cause Brief",
        "",
        f"- Variant: `{data['variant_id']}`",
        f"- Accepted suspect: `{data['accepted']}`",
        "",
        "## Ranked Suspects",
    ]
    for entry in data["ranking"]:
        lines.extend(
            [
                f"### {entry['rank']}. {entry['suspect_id']}",
                f"- Role: `{entry['role']}`",
                f"- File: `{entry['file']}`",
                f"- Symbol: `{entry['symbol']}`",
                f"- Summary: {entry['summary']}",
                f"- Evidence: {', '.join(entry['evidence_paths'])}",
                "",
            ]
        )
    lines.append("## Investigation Threads")
    for thread in data["investigation_threads"]:
        lines.extend(
            [
                f"- Question: {thread['question']}",
                f"  Files: {', '.join(thread['files'])}",
                f"  Finding: {thread['finding']}",
            ]
        )
    lines.extend(["", "## Evidence Table", ""])
    for row in data["evidence_table"]:
        lines.extend(
            [
                f"- Claim: {row['claim']}",
                f"  File: `{row['file']}`",
                f"  Symbol: `{row['symbol']}`",
                f"  Artifact: `{row['test_or_artifact']}`",
                f"  Why misleading: {row['why_misleading']}",
            ]
        )
    plan = data["remediation_plan"]
    lines.extend(
        [
            "",
            "## Remediation Plan",
            f"- Patch target: `{plan['patch_target_file']}` :: `{plan['patch_target_symbol']}`",
            f"- Why smallest safe patch: {plan['why_smallest_safe_patch']}",
            f"- Validation steps: {', '.join(plan['validation_steps'])}",
            f"- Non-goals: {', '.join(plan['non_goals'])}",
            "",
            "## Verification Note",
            f"- Failing assertion: {data['verification_note']['failing_assertion']}",
            f"- Contradictory artifact: `{data['verification_note']['contradictory_artifact']}`",
            f"- Resolution: {data['verification_note']['resolution']}",
            "",
        ]
    )
    return "\\n".join(lines)


def cmd_schema() -> int:
    schema = {
        "schema_version": SCHEMA_VERSION,
        "variant_id": VARIANT_ID,
        "accepted": "S1-fs-alias-normalization",
        "ranking": [
            {
                "suspect_id": "S1-fs-alias-normalization",
                "rank": 1,
                "file": "src/release_readiness/adapters/fs_source.py",
                "symbol": "normalize_fs_owner_alias",
                "role": "primary",
                "summary": "true root cause summary",
                "evidence_paths": ["src/release_readiness/adapters/fs_source.py"],
            }
        ],
        "investigation_threads": [
            {"question": "thread question", "files": ["path"], "finding": "finding"}
        ],
        "evidence_table": [
            {
                "claim": "claim",
                "file": "path",
                "symbol": "symbol",
                "test_or_artifact": "artifact",
                "why_misleading": "why",
            }
        ],
        "remediation_plan": {
            "patch_target_file": "src/release_readiness/adapters/fs_source.py",
            "patch_target_symbol": "normalize_fs_owner_alias",
            "why_smallest_safe_patch": "why",
            "validation_steps": ["pytest tests/test_root_cause_map.py -q", "inspect rendered output"],
            "non_goals": ["do not patch renderer", "do not broad-rewrite aggregation"],
        },
        "verification_note": {
            "failing_assertion": "exact assertion",
            "contradictory_artifact": "artifact path",
            "resolution": "resolution",
        },
    }
    print(json.dumps(schema, indent=2, sort_keys=True))
    return 0


def cmd_validate(path: Path) -> int:
    try:
        data = load_input(path)
    except Exception as exc:
        return fail(f"unable to read {path}: {exc}")
    errors = validate(data)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("valid")
    return 0


def cmd_submit(path: Path) -> int:
    try:
        data = load_input(path)
    except Exception as exc:
        return fail(f"unable to read {path}: {exc}")
    errors = validate(data)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    (BRIEF_DIR / "manager_brief.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\\n")
    (BRIEF_DIR / "manager_brief.md").write_text(render_markdown(data) + "\\n")
    print("submitted")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        return fail("usage: cnb55-brief {schema|validate|submit} [file]")
    command = argv[1]
    if command == "schema":
        return cmd_schema()
    if command in {"validate", "submit"}:
        if len(argv) != 3:
            return fail(f"usage: cnb55-brief {command} FILE")
        path = Path(argv[2])
        return cmd_validate(path) if command == "validate" else cmd_submit(path)
    return fail(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
"""


@dataclass(frozen=True)
class Variant:
    variant_id: str
    label: str
    contradictory_artifact_path: str
    required_context_path: str
    pass_bar: int
    extra_agent_notes: tuple[str, ...]
    release_context_files: dict[str, str]
    incident_context_files: dict[str, str]
    extra_artifacts: dict[str, str]


VARIANTS = [
    Variant(
        "v1-clean-baseline",
        "V1 clean baseline",
        "docs/triage-policy.md",
        "",
        78,
        (
            "The visible symptom is duplicated blocked-owner headings. Treat visible output as suspect, not authoritative.",
        ),
        {},
        {},
        {},
    ),
    Variant(
        "v2-noisy-distractor",
        "V2 noisy distractor",
        "artifacts/logs/operator_note_2026_04_15.md",
        "",
        74,
        (
            "An operator handoff note leans hard on the renderer. It may be stale or downstream-only.",
        ),
        {},
        {},
        {
            "artifacts/logs/operator_note_2026_04_15.md": """# Operator Note — 2026-04-15

The markdown report doubled headings again after the scheduler refactor.
Quick thought from on-call: maybe `render_blocked_owner_section()` stopped
deduping. I did not trace upstream owner keys before writing this note.
""",
        },
    ),
    Variant(
        "v3-dirty-state",
        "V3 dirty state",
        "artifacts/wip/renderer_hotfix.patch",
        "",
        70,
        (
            "There is an abandoned renderer hotfix under `artifacts/wip/`. It is context, not proof.",
        ),
        {},
        {},
        {
            "artifacts/wip/README.md": """# Abandoned WIP

This patch was drafted during the incident but never merged. The on-call
stopped after realizing the duplicate headings could be symptom-only.
""",
            "artifacts/wip/renderer_hotfix.patch": """diff --git a/src/release_readiness/renderers/markdown_renderer.py b/src/release_readiness/renderers/markdown_renderer.py
@@
-        lines.append(f"### {owner['display_owner']}")
+        if owner['display_owner'] not in seen:
+            lines.append(f"### {owner['display_owner']}")

# abandoned: this hid the duplicate display symptom without changing upstream
# owner keys
""",
        },
    ),
    Variant(
        "v4-multi-corpus-objective",
        "V4 multi-corpus objective",
        "release_context/hotfix_window.md",
        "release_context/hotfix_window.md",
        66,
        (
            "Release context changes the acceptable patch surface: fix the source normalization with minimal blast radius.",
        ),
        {
            "release_context/hotfix_window.md": """# Hotfix Window

Friday cut is frozen except for low-blast-radius fixes. Do not churn
`merge_blocked_owner_rows()` unless the adapter path cannot be repaired in
place. Downstream dashboards key off the current aggregation shape.
""",
            "release_context/downstream_contracts.md": """# Downstream Contracts

`blocked_owner_total` consumers assume the aggregation output shape stays
stable for this release. A hotfix that rewrites grouping semantics is higher
risk than repairing source normalization before aggregation.
""",
        },
        {},
        {},
    ),
    Variant(
        "v5-recovery-in-thread",
        "V5 recovery in thread",
        "incident_context/INC-447-renderer-rollback.md",
        "incident_context/INC-447-renderer-rollback.md",
        60,
        (
            "Incident context records a prior renderer-only hotfix rollback. Do not re-select that path without acknowledging the rollback.",
        ),
        {
            "release_context/hotfix_window.md": """# Hotfix Window

Friday cut is frozen except for low-blast-radius fixes. Keep the patch
adapter-local and preserve the aggregation contract.
""",
        },
        {
            "incident_context/INC-447-renderer-rollback.md": """# INC-447 Renderer Rollback

The first hotfix attempted to dedupe headings inside
`render_blocked_owner_section()`. It reduced the visible duplicate headings
but left `blocked_owner_total` wrong, so the patch was rolled back.
""",
            "incident_context/prior_hotfix_summary.md": """# Prior Hotfix Summary

Renderer-only mitigation masked the report symptom and confused operators.
The rollback note explicitly says the upstream owner keys were still split.
""",
        },
        {},
    ),
]


def agents_md(variant: Variant) -> str:
    extra_notes = "\n".join(f"- {note}" for note in variant.extra_agent_notes)
    context_note = ""
    if variant.release_context_files:
        context_note += "- `release_context/` is present for this variant.\n"
    if variant.incident_context_files:
        context_note += "- `incident_context/` is present for this variant.\n"
    return f"""# Agent Instructions — `parallel-explorer-root-cause-map`

## Task

Investigate why `release_readiness` over-reports blocked owners after the
scheduler refactor. Produce a unified root-cause brief plus one bounded
remediation plan. Do not patch the repo. Your only writable outputs are
`brief_input.json` and the generated files under `brief/`.

## Required workflow

1. Read at least two different surfaces before deciding: one adapter / aggregation path and one renderer / artifact path.
2. Author `brief_input.json` at the workspace root.
3. Run `./bin/cnb55-brief validate brief_input.json`.
4. Run `./bin/cnb55-brief submit brief_input.json`.

## Inputs

- `src/release_readiness/adapters/fs_source.py`
- `src/release_readiness/adapters/env_source.py`
- `src/release_readiness/core/aggregation.py`
- `src/release_readiness/renderers/markdown_renderer.py`
- `tests/test_root_cause_map.py`
- `docs/triage-policy.md`
- `artifacts/logs/`
- `artifacts/review/incident_thread.md`
{context_note}
## Suspects To Rank

- `S1-fs-alias-normalization`
- `S2-aggregation-grouping`
- `S3-renderer-duplicate-headings`
- `S4-env-watchlist-parser`

## What Good Looks Like

- the accepted suspect identifies the true source-normalization defect
- the aggregation layer is named as the place where the defect becomes visible in totals
- the renderer is ruled out as downstream-only
- the exact failing assertion is quoted
- at least one contradictory artifact is explicitly explained away
- the remediation plan names the smallest safe patch target and explicit non-goals

## Variant-specific notes

{extra_notes}

## Rules

- Do not modify `src/`, `tests/`, `docs/`, `artifacts/`, `release_context/`, or `incident_context/`.
- Do not add shim files such as `sitecustomize.py`, `usercustomize.py`, or `pytest.py`.
- Do not use the network.
"""


def docs_triage_policy() -> str:
    return """# Triage Policy

Document date: 2026-03-01

When blocked-owner headings are duplicated in the markdown report, the visible
symptom often appears in `render_blocked_owner_section()`. That symptom is not
enough to declare the renderer primary. Check upstream owner keys before
editing rendering code.
"""


def incident_thread() -> str:
    return """# Incident Thread Summary

- 09:12 on-call: renderer headings are duplicated, maybe the markdown path regressed.
- 09:19 follow-up: runtime snapshot shows both `team ops` and `team-ops`
  entering the owner map before rendering.
- 09:23 review: the scheduler refactor touched file-backed owner normalization.
- 09:31 recommendation: confirm adapter normalization before touching renderer.
"""


def runtime_snapshot() -> str:
    return json.dumps(
        {
            "observed_owner_keys": ["team ops", "team-ops", "platform infra", "platform-infra"],
            "rendered_headings": ["Team Ops", "team-ops", "Platform Infra", "platform-infra"],
            "blocked_owner_total": 4,
            "note": "duplicate owner keys were already present before markdown rendering",
        },
        indent=2,
        sort_keys=True,
    ) + "\n"


def rendered_excerpt() -> str:
    return """## Blocked Owners
### Team Ops
- owner_key: team ops
### team-ops
- owner_key: team-ops
### Platform Infra
- owner_key: platform infra
### platform-infra
- owner_key: platform-infra
"""


def operator_note_base() -> str:
    return """# Operator Note — 2026-04-14

The report duplicates blocked-owner headings after the scheduler refactor.
That makes the renderer suspicious, but I did not inspect the upstream owner
keys before writing this note.
"""


def brief_readme() -> str:
    return "Generated deliverables appear here after `./bin/cnb55-brief submit brief_input.json`.\n"


def oracle_brief(variant: Variant) -> dict:
    contradictory = variant.contradictory_artifact_path
    rows = [
        {
            "suspect_id": "S1-fs-alias-normalization",
            "rank": 1,
            "file": "src/release_readiness/adapters/fs_source.py",
            "symbol": "normalize_fs_owner_alias",
            "role": "primary",
            "summary": "The file-backed scheduler path emits space-preserving owner keys like `team ops`, while the env path emits hyphenated keys like `team-ops`; aggregation trusts `owner_key` verbatim, so aliases split before totals are counted.",
            "evidence_paths": [
                "src/release_readiness/adapters/fs_source.py",
                "src/release_readiness/core/aggregation.py",
                "tests/test_root_cause_map.py",
                "artifacts/logs/runtime_snapshot_2026_04_14.json",
            ],
        },
        {
            "suspect_id": "S2-aggregation-grouping",
            "rank": 2,
            "file": "src/release_readiness/core/aggregation.py",
            "symbol": "merge_blocked_owner_rows",
            "role": "amplifier",
            "summary": "Aggregation is where the bug becomes visible because it keys buckets on the already-split `owner_key`, but it is consuming bad upstream normalization rather than inventing new aliases itself.",
            "evidence_paths": [
                "src/release_readiness/core/aggregation.py",
                "artifacts/logs/runtime_snapshot_2026_04_14.json",
                "tests/test_root_cause_map.py",
            ],
        },
        {
            "suspect_id": "S4-env-watchlist-parser",
            "rank": 3,
            "file": "src/release_readiness/adapters/env_source.py",
            "symbol": "normalize_env_owner_token",
            "role": "secondary",
            "summary": "The env path is consistent and hyphen-normalized; it matters only because it disagrees with the file-backed path after the refactor.",
            "evidence_paths": [
                "src/release_readiness/adapters/env_source.py",
                "artifacts/logs/runtime_snapshot_2026_04_14.json",
            ],
        },
        {
            "suspect_id": "S3-renderer-duplicate-headings",
            "rank": 4,
            "file": "src/release_readiness/renderers/markdown_renderer.py",
            "symbol": "render_blocked_owner_section",
            "role": "downstream-only",
            "summary": "The renderer faithfully prints the already-duplicated owner buckets; it explains the visible symptom but not the inflated `blocked_owner_total`.",
            "evidence_paths": [
                "src/release_readiness/renderers/markdown_renderer.py",
                "artifacts/logs/rendered_report_excerpt.md",
                "artifacts/logs/runtime_snapshot_2026_04_14.json",
            ],
        },
    ]
    threads = [
        {
            "question": "Where do owner aliases diverge before the count is computed?",
            "files": [
                "src/release_readiness/adapters/fs_source.py",
                "src/release_readiness/adapters/env_source.py",
                "src/release_readiness/core/aggregation.py",
            ],
            "finding": "The scheduler file path preserves space-separated owner keys, the env path hyphenates them, and aggregation buckets on the mismatched keys without re-canonicalizing.",
        },
        {
            "question": "Is the renderer a cause or just a visible symptom?",
            "files": [
                "src/release_readiness/renderers/markdown_renderer.py",
                "artifacts/logs/rendered_report_excerpt.md",
                contradictory,
            ],
            "finding": "The renderer only reflects the split buckets it is handed; contradictory artifacts over-emphasize the symptom, but the runtime snapshot shows the duplication existed before rendering.",
        },
    ]
    note_resolution = "This artifact is downstream-only and misleading because the runtime snapshot already shows split owner keys before rendering."
    if "rollback" in contradictory:
        note_resolution = "This rollback note is essential context: it proves a renderer-only fix was already tried and failed because the upstream owner keys remained split."
    if "hotfix_window" in contradictory:
        note_resolution = "This release-context file is a constraint, not the cause; it rules out a broad aggregation rewrite and reinforces that the smallest safe patch belongs in source normalization."
    if "renderer_hotfix.patch" in contradictory:
        note_resolution = "This abandoned patch is sunk-cost evidence, not proof of root cause; it would hide duplicate headings without fixing the inflated blocked-owner total."
    return {
        "schema_version": "cnb55.root_cause_brief.v1",
        "variant_id": variant.variant_id,
        "accepted": "S1-fs-alias-normalization",
        "ranking": rows,
        "investigation_threads": threads,
        "evidence_table": [
            {
                "claim": "The file-backed adapter emits a space-preserving owner key that disagrees with the env path.",
                "file": "src/release_readiness/adapters/fs_source.py",
                "symbol": "normalize_fs_owner_alias",
                "test_or_artifact": "artifacts/logs/runtime_snapshot_2026_04_14.json",
                "why_misleading": "Visible markdown duplication happens later and should not be mistaken for the first divergence.",
            },
            {
                "claim": "Aggregation trusts `owner_key` verbatim, so the upstream mismatch becomes duplicate blocked owners.",
                "file": "src/release_readiness/core/aggregation.py",
                "symbol": "merge_blocked_owner_rows",
                "test_or_artifact": "tests/test_root_cause_map.py::test_scheduler_aliases_collapse_before_rendering",
                "why_misleading": "A dedupe in aggregation would mask the symptom but leaves the root normalization bug in place.",
            },
            {
                "claim": "The renderer prints what it is given and does not create new owner buckets.",
                "file": "src/release_readiness/renderers/markdown_renderer.py",
                "symbol": "render_blocked_owner_section",
                "test_or_artifact": "artifacts/logs/rendered_report_excerpt.md",
                "why_misleading": "The duplicated headings are downstream-visible and can trick a shallow read into blaming formatting.",
            },
            {
                "claim": "The failing test is asserting a unique-owner invariant that breaks before rendering.",
                "file": "tests/test_root_cause_map.py",
                "symbol": "test_scheduler_aliases_collapse_before_rendering",
                "test_or_artifact": 'assert summary["blocked_owner_total"] == 2, "scheduler aliases should collapse to the same blocked owner"',
                "why_misleading": "A renderer-only patch cannot make this assertion true because the count is wrong upstream.",
            },
        ],
        "remediation_plan": {
            "patch_target_file": "src/release_readiness/adapters/fs_source.py",
            "patch_target_symbol": "normalize_fs_owner_alias",
            "why_smallest_safe_patch": "Repairing file-backed owner canonicalization keeps the aggregation contract stable and removes the duplicate buckets before they can inflate totals or render duplicate headings.",
            "validation_steps": [
                "pytest tests/test_root_cause_map.py -q",
                "confirm runtime snapshot keys collapse to `team-ops` and `platform-infra` before rendering",
            ],
            "non_goals": [
                "do not patch the renderer first",
                "do not broad-rewrite aggregation during the hotfix window",
            ],
        },
        "verification_note": {
            "failing_assertion": 'assert summary["blocked_owner_total"] == 2, "scheduler aliases should collapse to the same blocked owner"',
            "contradictory_artifact": contradictory,
            "resolution": note_resolution,
        },
    }


def shortcut_brief(variant: Variant) -> dict:
    brief = oracle_brief(variant)
    by_id = {entry["suspect_id"]: entry for entry in brief["ranking"]}
    order = [
        by_id["S3-renderer-duplicate-headings"],
        by_id["S1-fs-alias-normalization"],
        by_id["S2-aggregation-grouping"],
        by_id["S4-env-watchlist-parser"],
    ]
    for idx, entry in enumerate(order, start=1):
        entry["rank"] = idx
    order[0]["role"] = "primary"
    brief["accepted"] = "S3-renderer-duplicate-headings"
    brief["ranking"] = order
    brief["remediation_plan"]["patch_target_file"] = "src/release_readiness/renderers/markdown_renderer.py"
    brief["remediation_plan"]["patch_target_symbol"] = "render_blocked_owner_section"
    return brief


def write_workspace(variant: Variant) -> Path:
    root = FAMILY_ROOT / "workspace_bundle" / variant.variant_id
    if root.exists():
        shutil.rmtree(root)
    (root / "brief").mkdir(parents=True, exist_ok=True)
    write(root / ".scenario_variant", variant.variant_id + "\n")
    write(root / "AGENTS.md", agents_md(variant))
    write(root / "Dockerfile", "FROM python:3.12-bookworm\nWORKDIR /workspace\n")
    write(root / "brief/README.md", brief_readme())
    write(root / "bin/cnb55-brief", cli_text())
    os.chmod(root / "bin/cnb55-brief", 0o755)

    for rel in [
        "src/release_readiness/__init__.py",
        "src/release_readiness/adapters/__init__.py",
        "src/release_readiness/core/__init__.py",
        "src/release_readiness/renderers/__init__.py",
    ]:
        write(root / rel, "")

    write(root / "src/release_readiness/adapters/fs_source.py", COMMON_FS_SOURCE)
    write(root / "src/release_readiness/adapters/env_source.py", COMMON_ENV_SOURCE)
    write(root / "src/release_readiness/core/aggregation.py", COMMON_AGGREGATION)
    write(root / "src/release_readiness/renderers/markdown_renderer.py", COMMON_RENDERER)
    write(root / "tests/test_root_cause_map.py", TEST_FILE)
    write(root / "docs/triage-policy.md", docs_triage_policy())
    write(root / "artifacts/logs/runtime_snapshot_2026_04_14.json", runtime_snapshot())
    write(root / "artifacts/logs/rendered_report_excerpt.md", rendered_excerpt())
    write(root / "artifacts/logs/operator_note_2026_04_14.md", operator_note_base())
    write(root / "artifacts/review/incident_thread.md", incident_thread())

    for rel, content in variant.extra_artifacts.items():
        write(root / rel, content)
    for rel, content in variant.release_context_files.items():
        write(root / rel, content)
    for rel, content in variant.incident_context_files.items():
        write(root / rel, content)
    return root


def write_milestone_scripts() -> None:
    shared = VERIFIER_DATA_ROOT / "_milestones_shared"
    shared.mkdir(parents=True, exist_ok=True)
    scripts = {
        "m1_localize.sh": "python3 - <<'PY'\nimport json, os\nr=json.loads(open(os.environ['RESULT_FILE']).read())\nraise SystemExit(0 if r['milestones']['M1_localization'] else 1)\nPY\n",
        "m2_primary_fix.sh": "python3 - <<'PY'\nimport json, os\nr=json.loads(open(os.environ['RESULT_FILE']).read())\nraise SystemExit(0 if r['milestones']['M2_primary_fix'] else 1)\nPY\n",
        "m3_invariants.sh": "python3 - <<'PY'\nimport json, os\nr=json.loads(open(os.environ['RESULT_FILE']).read())\nraise SystemExit(0 if r['milestones']['M3_invariants'] else 1)\nPY\n",
        "m4_functional.sh": "python3 - <<'PY'\nimport json, os\nr=json.loads(open(os.environ['RESULT_FILE']).read())\nraise SystemExit(0 if r['milestones']['M4_functional'] else 1)\nPY\n",
        "m5_e2e.sh": "python3 - <<'PY'\nimport json, os\nr=json.loads(open(os.environ['RESULT_FILE']).read())\nraise SystemExit(0 if r['milestones']['M5_e2e'] else 1)\nPY\n",
    }
    for name, content in scripts.items():
        write(shared / name, content)
        os.chmod(shared / name, 0o755)


def regen_oracle(variant: Variant, ws_root: Path) -> tuple[int, int, int]:
    vd = VERIFIER_DATA_ROOT / variant.variant_id
    oracle_dir = vd / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    oracle_input = oracle_brief(variant)
    write_json(oracle_dir / "brief_input.json", oracle_input)

    with tempfile.TemporaryDirectory(prefix="regen_root_cause_") as tmp:
        tmp_ws = Path(tmp) / "workspace"
        shutil.copytree(ws_root, tmp_ws)
        write_json(tmp_ws / "brief_input.json", oracle_input)
        subprocess.run([sys.executable, str(tmp_ws / "bin/cnb55-brief"), "submit", "brief_input.json"], cwd=tmp_ws, check=True)
        shutil.copy(tmp_ws / "brief/manager_brief.json", oracle_dir / "manager_brief.json")
        shutil.copy(tmp_ws / "brief/manager_brief.md", oracle_dir / "manager_brief.md")

        def score_workspace(workspace: Path) -> int:
            result_file = Path(tmp) / "verify_result.json"
            env = os.environ.copy()
            env.update(
                {
                    "AGENT_WS": str(workspace),
                    "VERIFIER_DATA": str(VERIFIER_DATA_ROOT),
                    "RESULT_FILE": str(result_file),
                    "VARIANT_ID": variant.variant_id,
                }
            )
            subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
            return json.loads(result_file.read_text())["score"]

        oracle_score = score_workspace(tmp_ws)

        empty_ws = Path(tmp) / "empty_workspace"
        shutil.copytree(ws_root, empty_ws)
        empty_score = score_workspace(empty_ws)

        shortcut_ws = Path(tmp) / "shortcut_workspace"
        shutil.copytree(ws_root, shortcut_ws)
        write_json(shortcut_ws / "brief_input.json", shortcut_brief(variant))
        subprocess.run([sys.executable, str(shortcut_ws / "bin/cnb55-brief"), "submit", "brief_input.json"], cwd=shortcut_ws, check=True)
        shortcut_score = score_workspace(shortcut_ws)

    return oracle_score, empty_score, shortcut_score


def update_gold_and_manifest(variant: Variant, ws_root: Path, oracle_score: int, empty_score: int, shortcut_score: int) -> None:
    vd = VERIFIER_DATA_ROOT / variant.variant_id
    vd.mkdir(parents=True, exist_ok=True)
    manifest = workspace_manifest(ws_root, variant.variant_id)
    write_json(vd / "workspace_manifest.json", manifest)

    gold = {
        "variant_id": variant.variant_id,
        "pass_bar": variant.pass_bar,
        "primary_suspect_id": "S1-fs-alias-normalization",
        "aggregation_suspect_id": "S2-aggregation-grouping",
        "renderer_suspect_id": "S3-renderer-duplicate-headings",
        "env_suspect_id": "S4-env-watchlist-parser",
        "primary_file": "src/release_readiness/adapters/fs_source.py",
        "primary_symbol": "normalize_fs_owner_alias",
        "aggregation_file": "src/release_readiness/core/aggregation.py",
        "aggregation_symbol": "merge_blocked_owner_rows",
        "renderer_file": "src/release_readiness/renderers/markdown_renderer.py",
        "renderer_symbol": "render_blocked_owner_section",
        "patch_target_file": "src/release_readiness/adapters/fs_source.py",
        "patch_target_symbol": "normalize_fs_owner_alias",
        "required_test_quote": 'assert summary["blocked_owner_total"] == 2, "scheduler aliases should collapse to the same blocked owner"',
        "contradictory_artifact_path": variant.contradictory_artifact_path,
        "required_context_path": variant.required_context_path.lower(),
        "allowed_write_paths": WRITE_ALLOWED,
        "readonly_tree_hashes": manifest["readonly_tree_hashes"],
        "baseline_scores": {
            "oracle": oracle_score,
            "empty": empty_score,
            "renderer_shortcut": shortcut_score,
        },
    }
    write_json(vd / "gold_ranking.json", gold)

    milestones = vd / "milestones"
    if milestones.exists():
        shutil.rmtree(milestones)
    milestones.mkdir(parents=True, exist_ok=True)
    for name in ["m1_localize.sh", "m2_primary_fix.sh", "m3_invariants.sh", "m4_functional.sh", "m5_e2e.sh"]:
        target = Path("../_milestones_shared") / name
        os.symlink(target, milestones / name)


def refresh_manifest_lock(summary: dict[str, dict[str, int]]) -> None:
    lock = {
        "schema_version": "cnb55.manifest.v2",
        "grader": {
            "score_ranking_py_sha256": sha256_file(SCORER),
            "verify_sh_sha256": sha256_file(VERIFIER_ROOT / "verify.sh"),
        },
        "cli": {
            "cnb55_brief_sha256": sha256_file(FAMILY_ROOT / "workspace_bundle" / "v1-clean-baseline" / "bin" / "cnb55-brief")
        },
        "variants": {},
    }
    for variant_id in VARIANT_IDS:
        ws_root = FAMILY_ROOT / "workspace_bundle" / variant_id
        vd = VERIFIER_DATA_ROOT / variant_id
        lock["variants"][variant_id] = {
            "observed_oracle_score": summary[variant_id]["oracle"],
            "observed_empty_brief_score": summary[variant_id]["empty"],
            "observed_shortcut_score": summary[variant_id]["shortcut"],
            "verifier_data": {
                "gold_ranking_sha256": sha256_file(vd / "gold_ranking.json"),
                "workspace_manifest_sha256": sha256_file(vd / "workspace_manifest.json"),
                "oracle_brief_json_sha256": sha256_file(vd / "oracle" / "manager_brief.json"),
                "oracle_brief_md_sha256": sha256_file(vd / "oracle" / "manager_brief.md"),
            },
            "workspace_trees": {
                rel: digest
                for rel in READONLY_TREES
                if (digest := sha256_tree(ws_root, rel))
            },
        }
    write_json(FAMILY_ROOT / "manifest.lock.json", lock)


def main() -> int:
    write_milestone_scripts()
    summary: dict[str, dict[str, int]] = {}
    for variant in VARIANTS:
        ws_root = write_workspace(variant)
        update_gold_and_manifest(variant, ws_root, 0, 0, 0)
        oracle_score, empty_score, shortcut_score = regen_oracle(variant, ws_root)
        update_gold_and_manifest(variant, ws_root, oracle_score, empty_score, shortcut_score)
        summary[variant.variant_id] = {
            "oracle": oracle_score,
            "empty": empty_score,
            "shortcut": shortcut_score,
        }
    refresh_manifest_lock(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
