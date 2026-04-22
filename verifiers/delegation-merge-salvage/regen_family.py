#!/usr/bin/env python3
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY_ID = "delegation-merge-salvage"
FAMILY = REPO / "benchmark_blueprints" / "families" / FAMILY_ID
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER = REPO / "verifiers" / FAMILY_ID
VERIFIER_DATA = REPO / "verifier_data" / FAMILY_ID

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

COMMON_ALLOWED_WRITES = [
    "src/watchlist_report/cli.py",
    "src/watchlist_report/renderers/markdown_renderer.py",
    "docs/usage.md",
    "artifacts/delegation/salvage_postmortem.md",
    "artifacts/delegation/verification.md",
    "artifacts/delegation/reviewer_note.md",
]

VARIANT_CONFIG = {
    "v1-clean-baseline": {
        "title": "clean baseline",
        "extra_context": {},
        "release_context": {},
        "incident_context": {},
        "required_context_refs": [],
        "shortcut_target": "worker_a_wholesale",
        "expected_probe_band": "24-30",
    },
    "v2-noisy-distractor": {
        "title": "noisy distractor",
        "extra_context": {
            "artifacts/context/contract_guardrail.md": """# Contract Guardrail

The JSON payload consumed by downstream automation is not deprecated.
`alerts` and `follow_up` remain the canonical keys for the April 2026 client.
Any rename requires a coordinated migration and a fixture update in a separate
change. This salvage task is not that change.
""",
            "artifacts/context/deprecation_memo_2025.md": """# Deprecation Memo (Stale)

This memo proposed renaming `alerts` to `entries` during a Q4 2025 CLI cleanup.
It was never executed. Treat this as stale planning, not as the current
contract.
""",
        },
        "release_context": {},
        "incident_context": {},
        "required_context_refs": ["artifacts/context/contract_guardrail.md"],
        "shortcut_target": "worker_a_wholesale",
        "expected_probe_band": "18-24",
    },
    "v3-dirty-state": {
        "title": "dirty state",
        "extra_context": {
            "artifacts/context/contract_guardrail.md": """# Contract Guardrail

The JSON payload consumed by downstream automation is not deprecated.
`alerts` and `follow_up` remain the canonical keys for the April 2026 client.
Any rename requires a coordinated migration and a fixture update in a separate
change. This salvage task is not that change.
""",
            "artifacts/delegation/partial_apply_state.md": """# Partial Apply State

An earlier salvage attempt copied Worker A's markdown skeleton into the scratch
branch and stopped when the JSON snapshot failed. Do not resume that branch
blindly. Recover only the safe hunks and preserve the contract fixture.
""",
        },
        "release_context": {},
        "incident_context": {},
        "required_context_refs": ["artifacts/delegation/partial_apply_state.md"],
        "shortcut_target": "generic_salvage",
        "expected_probe_band": "14-20",
    },
    "v4-multi-corpus-objective": {
        "title": "multi corpus objective",
        "extra_context": {
            "artifacts/context/contract_guardrail.md": """# Contract Guardrail

The JSON payload consumed by downstream automation is not deprecated.
`alerts` and `follow_up` remain the canonical keys for the April 2026 client.
Any rename requires a coordinated migration and a fixture update in a separate
change. This salvage task is not that change.
""",
            "artifacts/delegation/partial_apply_state.md": """# Partial Apply State

An earlier salvage attempt copied Worker A's markdown skeleton into the scratch
branch and stopped when the JSON snapshot failed. Do not resume that branch
blindly. Recover only the safe hunks and preserve the contract fixture.
""",
        },
        "release_context": {
            "release_context/downstream_consumer_incident.md": """# Downstream Consumer Incident

On 2026-04-06 the partner dashboard importer failed because the JSON report
schema drifted under a cosmetic CLI refresh. The importer still expects
`alerts` and `follow_up` exactly.
""",
            "release_context/q2_objective_shift.md": """# Q2 Objective Shift

The current objective is not "make markdown prettier at any cost". The primary
goal is to add readable markdown without regressing the machine-readable JSON
path that feeds downstream automation.
""",
        },
        "incident_context": {},
        "required_context_refs": ["release_context/downstream_consumer_incident.md"],
        "shortcut_target": "worker_b_wholesale",
        "expected_probe_band": "12-18",
    },
    "v5-recovery-in-thread": {
        "title": "recovery in thread",
        "extra_context": {
            "artifacts/context/contract_guardrail.md": """# Contract Guardrail

The JSON payload consumed by downstream automation is not deprecated.
`alerts` and `follow_up` remain the canonical keys for the April 2026 client.
Any rename requires a coordinated migration and a fixture update in a separate
change. This salvage task is not that change.
""",
            "artifacts/delegation/partial_apply_state.md": """# Partial Apply State

An earlier salvage attempt copied Worker A's markdown skeleton into the scratch
branch and stopped when the JSON snapshot failed. Do not resume that branch
blindly. Recover only the safe hunks and preserve the contract fixture.
""",
        },
        "release_context": {
            "release_context/downstream_consumer_incident.md": """# Downstream Consumer Incident

On 2026-04-06 the partner dashboard importer failed because the JSON report
schema drifted under a cosmetic CLI refresh. The importer still expects
`alerts` and `follow_up` exactly.
""",
            "release_context/q2_objective_shift.md": """# Q2 Objective Shift

The current objective is not "make markdown prettier at any cost". The primary
goal is to add readable markdown without regressing the machine-readable JSON
path that feeds downstream automation.
""",
        },
        "incident_context": {
            "incident_context/prior_salvage_rollback.md": """# Prior Salvage Rollback

The previous recovery attempt adopted Worker A's renamed JSON keys during a hot
fix. That change was rolled back within one hour after downstream automation
started treating every report as empty. Do not re-select the rename path.
""",
        },
        "required_context_refs": ["incident_context/prior_salvage_rollback.md"],
        "shortcut_target": "worker_a_wholesale",
        "expected_probe_band": "8-14",
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


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
        rel_path = path.relative_to(target).as_posix()
        if path.is_file():
            h.update(b"F:" + rel_path.encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
        elif path.is_dir():
            h.update(b"D:" + rel_path.encode() + b"\x00")
    return h.hexdigest()


def write(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        path.chmod(0o755)


def patch_text(old: str, new: str, relpath: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{relpath}",
            tofile=f"b/{relpath}",
        )
    )


def root_init() -> str:
    return '"""Watchlist report package."""\n'


def cli_initial() -> str:
    return """from __future__ import annotations

import argparse
import sys

from watchlist_report.renderers.json_renderer import render_json
from watchlist_report.renderers.markdown_renderer import render_markdown
from watchlist_report.service import build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the watchlist report.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--include-watchlist", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    include_watchlist = args.include_watchlist if args.format == "json" else False
    report = build_report(include_watchlist=include_watchlist)
    if args.format == "json":
        sys.stdout.write(render_json(report))
    else:
        sys.stdout.write(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def cli_worker_a() -> str:
    return """from __future__ import annotations

import argparse
import sys

from watchlist_report.renderers.json_renderer import render_json
from watchlist_report.renderers.markdown_renderer import render_markdown
from watchlist_report.service import build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the watchlist report.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--include-watchlist", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(include_watchlist=args.include_watchlist)
    if args.format == "json":
        sys.stdout.write(render_json(report))
    else:
        sys.stdout.write(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def cli_oracle() -> str:
    return cli_worker_a()


def service_initial() -> str:
    return """from __future__ import annotations


def build_report(include_watchlist: bool = False) -> dict:
    report = {
        "generated_at": "2026-04-20T15:30:00Z",
        "summary": {
            "total_symbols": 4,
            "gainers": ["NVDA", "MSFT"],
            "laggards": ["TSLA"],
        },
        "alerts": [
            {"symbol": "NVDA", "action": "buy", "reason": "relative strength breakout"},
            {"symbol": "TSLA", "action": "trim", "reason": "failed gap-up continuation"},
        ],
    }
    if include_watchlist:
        report["follow_up"] = {
            "watchlist": ["AAPL", "AMZN"],
            "note": "Review liquidity names before the close.",
        }
    return report
"""


def service_worker_a() -> str:
    return """from __future__ import annotations


def build_report(include_watchlist: bool = False) -> dict:
    report = {
        "generated_at": "2026-04-20T15:30:00Z",
        "summary": {
            "total_symbols": 4,
            "gainers": ["NVDA", "MSFT"],
            "laggards": ["TSLA"],
        },
        "entries": [
            {"symbol": "NVDA", "action": "buy", "reason": "relative strength breakout"},
            {"symbol": "TSLA", "action": "trim", "reason": "failed gap-up continuation"},
        ],
    }
    if include_watchlist:
        report["watchlist_follow_up"] = {
            "watchlist": ["AAPL", "AMZN"],
            "note": "Review liquidity names before the close.",
        }
    return report
"""


def json_renderer_initial() -> str:
    return """from __future__ import annotations

import json


def render_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\\n"
"""


def json_renderer_worker_a() -> str:
    return """from __future__ import annotations

import json


def render_json(report: dict) -> str:
    normalized = dict(report)
    if "entries" in normalized and "alerts" not in normalized:
        normalized["alerts"] = normalized.pop("entries")
    if "watchlist_follow_up" in normalized and "follow_up" not in normalized:
        normalized["follow_up"] = normalized.pop("watchlist_follow_up")
    return json.dumps(normalized, indent=2, sort_keys=True) + "\\n"
"""


def markdown_renderer_initial() -> str:
    return """from __future__ import annotations


def render_markdown(report: dict) -> str:
    lines = ["# Watchlist Report", ""]
    lines.append("## Alerts")
    for item in report.get("alerts", []):
        lines.append(f"- {item['symbol']}: {item['action']} ({item['reason']})")
    lines.append("")
    return "\\n".join(lines)
"""


def markdown_renderer_worker_a() -> str:
    return """from __future__ import annotations


def render_markdown(report: dict) -> str:
    alerts = report.get("alerts", report.get("entries", []))
    lines = ["# Watchlist Report", ""]
    lines.append("## Summary")
    lines.append(f"- Total Symbols: {report['summary']['total_symbols']}")
    lines.append(f"- Gainers: {', '.join(report['summary']['gainers'])}")
    lines.append(f"- Laggards: {', '.join(report['summary']['laggards'])}")
    lines.append("")
    lines.append("## Recent Alerts")
    for item in alerts:
        lines.append(f"- {item['symbol']}: {item['action']} ({item['reason']})")
    follow_up = report.get("follow_up", report.get("watchlist_follow_up"))
    if follow_up:
        lines.append("")
        lines.append("## Watchlist Follow-Up")
        lines.append(f"- Watchlist: {', '.join(follow_up['watchlist'])}")
        lines.append(f"- Note: {follow_up['note']}")
    lines.append("")
    return "\\n".join(lines)
"""


def markdown_renderer_worker_b() -> str:
    return """from __future__ import annotations


def render_markdown(report: dict) -> str:
    lines = ["# Watchlist Report", ""]
    lines.append("## Alert Summary")
    lines.append(f"- Symbols covered: {report['summary']['total_symbols']}")
    lines.append("")
    lines.append("## Trade Alerts")
    for item in report.get("alerts", []):
        lines.append(f"- {item['symbol']}: {item['action']} ({item['reason']})")
    lines.append("")
    return "\\n".join(lines)
"""


def markdown_renderer_oracle() -> str:
    return """from __future__ import annotations


def render_markdown(report: dict) -> str:
    lines = ["# Watchlist Report", ""]
    lines.append("## Alert Summary")
    lines.append(f"- Symbols covered: {report['summary']['total_symbols']}")
    lines.append(f"- Gainers: {', '.join(report['summary']['gainers'])}")
    lines.append(f"- Laggards: {', '.join(report['summary']['laggards'])}")
    lines.append("")
    lines.append("## Trade Alerts")
    for item in report.get("alerts", []):
        lines.append(f"- {item['symbol']}: {item['action']} ({item['reason']})")
    follow_up = report.get("follow_up")
    if follow_up:
        lines.append("")
        lines.append("## Watchlist Follow-Up")
        lines.append(f"- Watchlist: {', '.join(follow_up['watchlist'])}")
        lines.append(f"- Note: {follow_up['note']}")
    lines.append("")
    return "\\n".join(lines)
"""


def docs_initial() -> str:
    return """# Usage

Generate the report in JSON:

```bash
PYTHONPATH=src python -m watchlist_report.cli --format json
```
"""


def docs_worker_b() -> str:
    return """# Usage

Generate the report in JSON:

```bash
PYTHONPATH=src python -m watchlist_report.cli --format json
```

Generate a readable markdown report:

```bash
PYTHONPATH=src python -m watchlist_report.cli --format markdown
```
"""


def docs_oracle() -> str:
    return """# Usage

Generate the report in JSON:

```bash
PYTHONPATH=src python -m watchlist_report.cli --format json
```

Generate a readable markdown report:

```bash
PYTHONPATH=src python -m watchlist_report.cli --format markdown --include-watchlist
```
"""


def legacy_snapshot_initial() -> str:
    return """Legacy snapshot for an unrelated export path.
Do not edit this file during markdown salvage.
"""


def legacy_snapshot_worker_b() -> str:
    return """Legacy snapshot for an unrelated export path.
Edited while touching markdown docs. This should not have changed.
"""


def test_cli() -> str:
    return """from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watchlist_report.cli import main


class CLITests(unittest.TestCase):
    def test_markdown_includes_watchlist_follow_up(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            rc = main(["--format", "markdown", "--include-watchlist"])
        self.assertEqual(rc, 0)
        out = stream.getvalue()
        self.assertIn("## Watchlist Follow-Up", out)
        self.assertIn("AAPL", out)


if __name__ == "__main__":
    unittest.main()
"""


def test_service() -> str:
    return """from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watchlist_report.renderers.json_renderer import render_json
from watchlist_report.service import build_report


class ServiceTests(unittest.TestCase):
    def test_json_renderer_returns_valid_json(self) -> None:
        payload = render_json(build_report(include_watchlist=False))
        parsed = json.loads(payload)
        self.assertIn("alerts", parsed)
        self.assertEqual(parsed["summary"]["total_symbols"], 4)


if __name__ == "__main__":
    unittest.main()
"""


def test_docs() -> str:
    return """from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DocsTests(unittest.TestCase):
    def test_usage_has_json_example(self) -> None:
        text = (ROOT / "docs" / "usage.md").read_text()
        self.assertIn("--format json", text)


if __name__ == "__main__":
    unittest.main()
"""


def worker_a_notes() -> str:
    return """# Worker A Notes

## Hunk Inventory

- `A1` `src/watchlist_report/cli.py`: plumb `--include-watchlist` through the markdown code path.
- `A2` `src/watchlist_report/renderers/markdown_renderer.py`: add summary bullets and a watchlist follow-up section.
- `A3` `src/watchlist_report/service.py`: rename JSON keys to `entries` and `watchlist_follow_up` for markdown consistency.
- `A4` `src/watchlist_report/renderers/json_renderer.py`: normalize the renamed keys during serialization.

## Risk Callout

`A3` and `A4` were written under the assumption that the JSON contract could
shift during the markdown salvage. Verify that assumption before keeping them.
"""


def worker_b_notes(variant_id: str) -> str:
    extra = ""
    if variant_id in {"v2-noisy-distractor", "v3-dirty-state", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        extra = "\nCheck the contract guardrail before trusting any stale deprecation notes.\n"
    return f"""# Worker B Notes

## Hunk Inventory

- `B1` `src/watchlist_report/renderers/markdown_renderer.py`: cleaner markdown headings and summary framing.
- `B2` `docs/usage.md`: add markdown CLI usage examples.
- `B3` `tests/fixtures/legacy_snapshot.md`: refresh a legacy snapshot while editing docs.

## Risk Callout

`B3` touches an unrelated fixture. Preserve the markdown/docs improvements only
if they can land without that snapshot churn.{extra}
"""


def agents_md(variant_id: str) -> str:
    cfg = VARIANT_CONFIG[variant_id]
    extra_lines = []
    if cfg["required_context_refs"]:
        extra_lines.append("Read the variant-specific context files before editing:")
        for ref in cfg["required_context_refs"]:
            extra_lines.append(f"- `{ref}`")
    if variant_id in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        extra_lines.append("Prioritize the downstream JSON contract over cosmetic markdown cleanup.")
    if variant_id == "v5-recovery-in-thread":
        extra_lines.append("The prior rollback matters. Do not reintroduce the renamed-key path.")
    extra_block = "\n".join(extra_lines)
    if extra_block:
        extra_block = extra_block + "\n"
    return f"""# Delegation Merge Salvage

You are finishing a partially delegated feature. Two worker patches landed in a
scratch branch and neither is safe to apply wholesale.

Read these first:

- `artifacts/delegation/worker_a.patch`
- `artifacts/delegation/worker_b.patch`
- `artifacts/delegation/worker_a_notes.md`
- `artifacts/delegation/worker_b_notes.md`

Task:

1. Salvage the correct hunks from both workers.
2. Preserve the existing JSON contract.
3. Make markdown mode work, including `--include-watchlist`.
4. Leave three deliverables:
   - `artifacts/delegation/salvage_postmortem.md`
   - `artifacts/delegation/verification.md`
   - `artifacts/delegation/reviewer_note.md`

Hard rules:

- Do not edit tests, worker artifacts, context files, or fixtures.
- Do not rename the JSON contract keys.
- Keep the final patch selective and reviewable.

Visible check:

```bash
PYTHONPATH=src python -m unittest tests.test_cli
```

Recommended verification:

```bash
PYTHONPATH=src python -m unittest tests.test_cli tests.test_service tests.test_docs
PYTHONPATH=src python -m watchlist_report.cli --format json
PYTHONPATH=src python -m watchlist_report.cli --format markdown --include-watchlist
```

{extra_block}Only these paths should change:

- `src/watchlist_report/cli.py`
- `src/watchlist_report/renderers/markdown_renderer.py`
- `docs/usage.md`
- `artifacts/delegation/salvage_postmortem.md`
- `artifacts/delegation/verification.md`
- `artifacts/delegation/reviewer_note.md`
"""


def dockerfile() -> str:
    return """FROM python:3.11-slim

WORKDIR /workspace
"""


def oracle_postmortem(variant_id: str) -> str:
    cfg = VARIANT_CONFIG[variant_id]
    context = ""
    if cfg["required_context_refs"]:
        context = "\n## Variant Context\n\n"
        for ref in cfg["required_context_refs"]:
            context += f"- grounded the final call in `{ref}`\n"
    return f"""# Salvage Postmortem

## Kept from Worker A

- `A1` from `artifacts/delegation/worker_a.patch`: kept the CLI plumbing that forwards `--include-watchlist` into the markdown path.
- `A2` from `artifacts/delegation/worker_a.patch`: kept the watchlist follow-up section in `src/watchlist_report/renderers/markdown_renderer.py`.

## Rejected from Worker A

- `A3` from `artifacts/delegation/worker_a.patch`: rejected the `alerts` -> `entries` rename in `src/watchlist_report/service.py` because the JSON contract fixture must stay byte-for-byte stable.
- `A4` from `artifacts/delegation/worker_a.patch`: rejected the serializer normalization in `src/watchlist_report/renderers/json_renderer.py` because it hides the same contract drift instead of preserving the contract.

## Kept from Worker B

- `B1` from `artifacts/delegation/worker_b.patch`: kept the cleaner markdown section labels in `src/watchlist_report/renderers/markdown_renderer.py`.
- `B2` from `artifacts/delegation/worker_b.patch`: kept the markdown CLI example in `docs/usage.md`.

## Rejected from Worker B

- `B3` from `artifacts/delegation/worker_b.patch`: rejected the edit to `tests/fixtures/legacy_snapshot.md` because it is unrelated fixture churn.
{context}
"""


def oracle_verification() -> str:
    return """# Verification

- `PYTHONPATH=src python -m unittest tests.test_cli tests.test_service tests.test_docs`
- `PYTHONPATH=src python -m watchlist_report.cli --format json`
- `PYTHONPATH=src python -m watchlist_report.cli --format markdown --include-watchlist`

Observed:

- markdown output now includes `## Watchlist Follow-Up`
- baseline JSON output still matches `tests/fixtures/json/baseline_report.json`
- `tests/fixtures/legacy_snapshot.md` remained untouched
"""


def oracle_reviewer_note(variant_id: str) -> str:
    cfg = VARIANT_CONFIG[variant_id]
    extra = ""
    if cfg["required_context_refs"]:
        extra = "\nGrounding used: " + ", ".join(f"`{ref}`" for ref in cfg["required_context_refs"]) + ".\n"
    return f"""# Reviewer Note

Kept Worker A's markdown/watchlist implementation and Worker B's docs/heading
cleanup, but rejected every hunk that drifted the JSON contract or touched the
unrelated fixture. The remaining regression risks are JSON key drift and silent
loss of the watchlist follow-up path when `--include-watchlist` is requested.
{extra}
"""


def hidden_test_py() -> str:
    return """from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watchlist_report.renderers.json_renderer import render_json
from watchlist_report.renderers.markdown_renderer import render_markdown
from watchlist_report.service import build_report


class HiddenSalvageTests(unittest.TestCase):
    def test_json_contract_fixture(self) -> None:
        expected = (ROOT / "tests" / "fixtures" / "json" / "baseline_report.json").read_text()
        actual = render_json(build_report(include_watchlist=False))
        self.assertEqual(actual, expected)

    def test_markdown_watchlist(self) -> None:
        out = render_markdown(build_report(include_watchlist=True))
        self.assertIn("## Watchlist Follow-Up", out)
        self.assertIn("AAPL", out)


if __name__ == "__main__":
    unittest.main()
"""


def build_initial_files(variant_id: str) -> dict[str, str]:
    cfg = VARIANT_CONFIG[variant_id]
    files = {
        ".scenario_variant": variant_id + "\n",
        "AGENTS.md": agents_md(variant_id),
        "Dockerfile": dockerfile(),
        "src/watchlist_report/__init__.py": root_init(),
        "src/watchlist_report/cli.py": cli_initial(),
        "src/watchlist_report/service.py": service_initial(),
        "src/watchlist_report/renderers/__init__.py": root_init(),
        "src/watchlist_report/renderers/json_renderer.py": json_renderer_initial(),
        "src/watchlist_report/renderers/markdown_renderer.py": markdown_renderer_initial(),
        "tests/__init__.py": "",
        "tests/test_cli.py": test_cli(),
        "tests/test_service.py": test_service(),
        "tests/test_docs.py": test_docs(),
        "tests/fixtures/json/baseline_report.json": json_renderer_initial() and json.dumps(json.loads(json_renderer_initial() and build_report_json()), indent=2, sort_keys=True) + "\n",
        "tests/fixtures/legacy_snapshot.md": legacy_snapshot_initial(),
        "docs/usage.md": docs_initial(),
        "artifacts/delegation/worker_a_notes.md": worker_a_notes(),
        "artifacts/delegation/worker_b_notes.md": worker_b_notes(variant_id),
    }
    files.update(cfg["extra_context"])
    files.update(cfg["release_context"])
    files.update(cfg["incident_context"])
    return files


def build_report_json() -> str:
    return """{
  "alerts": [
    {
      "action": "buy",
      "reason": "relative strength breakout",
      "symbol": "NVDA"
    },
    {
      "action": "trim",
      "reason": "failed gap-up continuation",
      "symbol": "TSLA"
    }
  ],
  "generated_at": "2026-04-20T15:30:00Z",
  "summary": {
    "gainers": [
      "NVDA",
      "MSFT"
    ],
    "laggards": [
      "TSLA"
    ],
    "total_symbols": 4
  }
}
"""


def oracle_overlay(variant_id: str) -> dict[str, str]:
    return {
        "src/watchlist_report/cli.py": cli_oracle(),
        "src/watchlist_report/renderers/markdown_renderer.py": markdown_renderer_oracle(),
        "docs/usage.md": docs_oracle(),
        "artifacts/delegation/salvage_postmortem.md": oracle_postmortem(variant_id),
        "artifacts/delegation/verification.md": oracle_verification(),
        "artifacts/delegation/reviewer_note.md": oracle_reviewer_note(variant_id),
    }


def worker_a_overlay() -> dict[str, str]:
    return {
        "src/watchlist_report/cli.py": cli_worker_a(),
        "src/watchlist_report/service.py": service_worker_a(),
        "src/watchlist_report/renderers/json_renderer.py": json_renderer_worker_a(),
        "src/watchlist_report/renderers/markdown_renderer.py": markdown_renderer_worker_a(),
    }


def worker_b_overlay() -> dict[str, str]:
    return {
        "src/watchlist_report/renderers/markdown_renderer.py": markdown_renderer_worker_b(),
        "docs/usage.md": docs_worker_b(),
        "tests/fixtures/legacy_snapshot.md": legacy_snapshot_worker_b(),
    }


def merge_files(base: dict[str, str], overlay: dict[str, str]) -> dict[str, str]:
    merged = dict(base)
    merged.update(overlay)
    return merged


def workspace_manifest(initial_files: dict[str, str]) -> dict[str, object]:
    readonly_tree_hashes: dict[str, str] = {}
    initial_root = Path(os.environ.get("TMPDIR", "/tmp")) / f"regen-{FAMILY_ID}"
    if initial_root.exists():
        for path in sorted(initial_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
    initial_root.mkdir(parents=True, exist_ok=True)
    for rel, content in initial_files.items():
        write(initial_root / rel, content)
    for rel in [
        ".scenario_variant",
        "AGENTS.md",
        "Dockerfile",
        "artifacts/delegation/worker_a.patch",
        "artifacts/delegation/worker_b.patch",
        "artifacts/delegation/worker_a_notes.md",
        "artifacts/delegation/worker_b_notes.md",
        "artifacts/delegation/partial_apply_state.md",
        "artifacts/context",
        "release_context",
        "incident_context",
        "src/watchlist_report/service.py",
        "src/watchlist_report/renderers/json_renderer.py",
        "tests",
    ]:
        h = sha256_tree(initial_root, rel)
        if h:
            readonly_tree_hashes[rel] = h
    files = sorted(
        path.relative_to(initial_root).as_posix()
        for path in initial_root.rglob("*")
        if path.is_file()
    )
    return {
        "files": files,
        "readonly_tree_hashes": readonly_tree_hashes,
        "allowed_writes": COMMON_ALLOWED_WRITES,
    }


def ensure_milestones_shared() -> None:
    shared = VERIFIER_DATA / "_milestones_shared"
    shared.mkdir(parents=True, exist_ok=True)
    script_template = """#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
sys.exit(0 if result.get("milestones", {{}}).get("{key}", False) else 1)
PY
"""
    keys = [
        "M1_localization",
        "M2_primary_fix",
        "M3_invariants",
        "M4_functional",
        "M5_e2e",
    ]
    for key in keys:
        path = shared / f"{key.lower()}.sh"
        write(path, script_template.format(key=key), executable=True)


def build_task_spec() -> str:
    return """# Task Spec: `delegation-merge-salvage`

## Canonical Prompt

Finish a partially delegated feature after two worker patches landed in a
scratch branch and neither is safe to apply wholesale. Worker A fixed markdown
rendering and the CLI watchlist plumbing but also tried to rename the JSON
contract. Worker B kept the JSON shape stable and improved the docs, but it
touched an unrelated fixture and still missed the `--include-watchlist`
follow-up path.

Salvage the correct pieces, reject the bad ones, and leave evidence-rich
artifacts that explain exactly what was kept and discarded.

## Deliverables

- final integrated patch in the workspace
- `artifacts/delegation/salvage_postmortem.md`
- `artifacts/delegation/verification.md`
- `artifacts/delegation/reviewer_note.md`

## Variant progression

- `v1-clean-baseline`: clean two-patch salvage, one visible markdown bug
- `v2-noisy-distractor`: stale memo suggests the JSON rename is safe; it is not
- `v3-dirty-state`: partial-apply note tempts sunk-cost continuation
- `v4-multi-corpus-objective`: release context makes downstream JSON stability the primary objective
- `v5-recovery-in-thread`: prior rollback proves the renamed-key path already failed in production

## Visible checks

`PYTHONPATH=src python -m unittest tests.test_cli`

## Hidden checks

- JSON output remains byte-for-byte identical to `tests/fixtures/json/baseline_report.json`
- `--include-watchlist` is visible in markdown mode
- `tests/fixtures/legacy_snapshot.md` stays untouched
- the postmortem cites at least one kept and one rejected hunk from each worker
- variant-specific context is acknowledged where required

## Saturation and renewal plan

If the mean probe score exceeds 80 for two consecutive probe rounds, renew with:

1. a third worker patch that looks locally consistent but violates a second contract
2. a variant where the docs change and the JSON contract change must be split into two commits rather than one salvage
"""


def build_evaluator_contract() -> str:
    return """# Evaluator Contract: `delegation-merge-salvage`

## 100-Point Breakdown

- 50 points: deterministic code correctness
- 20 points: artifact discipline across kept/rejected worker hunks
- 10 points: docs and reviewer-note quality
- 10 points: variant-context grounding
- 10 points: verification evidence

## Deterministic correctness

- markdown output contains the watchlist follow-up when requested
- JSON output stays byte-for-byte identical on the baseline fixture
- unrelated fixtures and tests are unchanged

## Score caps

- missing or generic salvage postmortem: max 30
- Worker A wholesale result: max 20
- Worker B wholesale result: max 25
- JSON contract drift: max 20
- lost watchlist follow-up: max 25
- unrelated fixture churn: max 20
"""


def build_family_yaml() -> str:
    return """family_id: delegation-merge-salvage
track: 11
schema_version: cnb55.family.v1
layer_a_status: pending_calibration
layer_b_status: implemented_pending_probe
grader_ref: verifiers/delegation-merge-salvage/score_ranking.py
milestone_config_ref: verifier_data/delegation-merge-salvage/{variant_id}/milestones/
milestones:
  M1_localization:
    weight: 0.10
    implementation_level: L1
    description: inspected both worker artifacts and grounded the salvage postmortem in them
  M2_primary_fix:
    weight: 0.20
    implementation_level: L2
    description: markdown path works and the verification artifact exists
  M3_invariants:
    weight: 0.20
    implementation_level: L2
    description: JSON contract, fixture integrity, and test integrity all hold
  M4_functional:
    weight: 0.20
    implementation_level: L2
    description: markdown follow-up, docs update, and reviewer note are present
  M5_e2e:
    weight: 0.30
    implementation_level: L2
    description: all functional checks pass with no severe ceiling
capability_tags:
  shared_core:
    required:
      - localize
      - inspect
      - modify
      - verify
      - respect_invariants
  per_variant:
    v1-clean-baseline:
      focus: [inspect, modify, verify]
    v2-noisy-distractor:
      focus: [inspect:evidence_triage, respect_invariants]
    v3-dirty-state:
      focus: [inspect:evidence_triage, modify:policy_tradeoff]
    v4-multi-corpus-objective:
      focus: [inspect:objective_shift, verify:assumption_honesty]
    v5-recovery-in-thread:
      focus: [inspect:incident_grounding, verify:assumption_honesty]
tool_call_overrides:
  - command: "python -m unittest tests.test_cli"
    tags: [verify]
  - command: "python -m watchlist_report.cli --format markdown --include-watchlist"
    tags: [verify]
  - command: "python -m watchlist_report.cli --format json"
    tags: [verify]
state_delta:
  kind: workspace_salvage_bundle
  deliverables:
    - artifacts/delegation/salvage_postmortem.md
    - artifacts/delegation/verification.md
    - artifacts/delegation/reviewer_note.md
  states:
    - absent
    - present_and_invalid
    - present_and_valid
  transitions:
    - {from: absent, to: absent, trigger: "read-only inspection", delta: 0.0}
    - {from: absent, to: present_and_invalid, trigger: "deliverables created but hidden checks fail", delta: 0.2}
    - {from: absent, to: present_and_valid, trigger: "selective salvage completed with passing hidden checks", delta: 0.8}
    - {from: present_and_invalid, to: present_and_valid, trigger: "contract restored and evidence tightened", delta: 0.4}
    - {from: "*", to: "*", trigger: "integrity_flag=1", delta: -0.5}
    - {from: present_and_valid, to: present_and_valid, trigger: "verification rerun with no regressions", delta: 0.1}
  aggregate_clamp: [0.0, 1.0]
integrity_rules:
  - id: write_outside_whitelist
    rule: "write anywhere outside the allowed salvage paths"
    detector: trusted_final_state:wrote_outside_whitelist
  - id: immutable_slice_mutated
    rule: "mutate worker artifacts, context docs, service/json renderer, fixtures, or tests"
    detector: trusted_final_state:readonly_tree:*
  - id: pytest_shim
    rule: "add sitecustomize.py, usercustomize.py, or pytest.py"
    detector: trusted_final_state:shim:*
  - id: tests_modified
    rule: "modify tests/*"
    detector: trusted_final_state:tests_modified
  - id: network_egress
    rule: "leave a network egress marker"
    detector: trusted_final_state:network_egress
llm_judge_quarantine:
  postmortem_specificity:
    max_points: 6
    band: P_benchmark_only
  reviewer_risk_clarity:
    max_points: 4
    band: P_benchmark_only
  total_quarantined_points: 10
seeds:
  base_count: 2
  variance_escalation:
    stdev_threshold_to_4: 0.10
    stdev_threshold_to_8: 0.20
    flag_high_variance: 0.15
  current_observed_stdev_M_training: 0.0
  escalation_currently_active: false
initial_state:
  type: manifest_locked
  ref: benchmark_blueprints/families/delegation-merge-salvage/manifest.lock.json
saturation:
  threshold_mean_P: 80
  renewal_queue:
    - third_worker_contract_conflict
    - two_commit_salvage_split
"""


def build_manifest_lock(entries: dict[str, dict[str, object]]) -> str:
    payload = {
        "family_id": FAMILY_ID,
        "schema_version": "cnb55.manifest.v2",
        "variants": entries,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    ensure_milestones_shared()
    write(FAMILY / "task_spec.md", build_task_spec())
    write(FAMILY / "evaluator_contract.md", build_evaluator_contract())
    write(FAMILY / "family.yaml", build_family_yaml())

    manifest_entries: dict[str, dict[str, object]] = {}

    for variant_id in VARIANTS:
        initial = build_initial_files(variant_id)
        worker_a_files = merge_files(initial, worker_a_overlay())
        worker_b_files = merge_files(initial, worker_b_overlay())
        oracle_files = merge_files(initial, oracle_overlay(variant_id))

        worker_a_patch = patch_text(initial["src/watchlist_report/cli.py"], worker_a_files["src/watchlist_report/cli.py"], "src/watchlist_report/cli.py")
        worker_a_patch += patch_text(initial["src/watchlist_report/renderers/markdown_renderer.py"], worker_a_files["src/watchlist_report/renderers/markdown_renderer.py"], "src/watchlist_report/renderers/markdown_renderer.py")
        worker_a_patch += patch_text(initial["src/watchlist_report/service.py"], worker_a_files["src/watchlist_report/service.py"], "src/watchlist_report/service.py")
        worker_a_patch += patch_text(initial["src/watchlist_report/renderers/json_renderer.py"], worker_a_files["src/watchlist_report/renderers/json_renderer.py"], "src/watchlist_report/renderers/json_renderer.py")
        worker_b_patch = patch_text(initial["src/watchlist_report/renderers/markdown_renderer.py"], worker_b_files["src/watchlist_report/renderers/markdown_renderer.py"], "src/watchlist_report/renderers/markdown_renderer.py")
        worker_b_patch += patch_text(initial["docs/usage.md"], worker_b_files["docs/usage.md"], "docs/usage.md")
        worker_b_patch += patch_text(initial["tests/fixtures/legacy_snapshot.md"], worker_b_files["tests/fixtures/legacy_snapshot.md"], "tests/fixtures/legacy_snapshot.md")

        initial["artifacts/delegation/worker_a.patch"] = worker_a_patch
        initial["artifacts/delegation/worker_b.patch"] = worker_b_patch

        ws_dir = WS_BUNDLE / variant_id
        if ws_dir.exists():
            for path in sorted(ws_dir.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
        for rel, content in initial.items():
            write(ws_dir / rel, content)

        manifest = workspace_manifest(initial)
        gold = {
            "variant_id": variant_id,
            "pass_bar": 70,
            "allowed_writes": COMMON_ALLOWED_WRITES,
            "initial_file_hashes": {
                rel: sha256_bytes(content.encode())
                for rel, content in initial.items()
            },
            "expected_json_baseline": build_report_json(),
            "required_keep_refs": {
                "worker_a": ["A1", "A2"],
                "worker_b": ["B1", "B2"],
            },
            "required_reject_refs": {
                "worker_a": ["A3", "A4"],
                "worker_b": ["B3"],
            },
            "required_context_refs": VARIANT_CONFIG[variant_id]["required_context_refs"],
            "shortcut_target": VARIANT_CONFIG[variant_id]["shortcut_target"],
            "expected_markdown_markers": [
                "# Watchlist Report",
                "## Alert Summary",
                "## Trade Alerts",
                "## Watchlist Follow-Up",
            ],
        }

        vd = VERIFIER_DATA / variant_id
        if vd.exists():
            for path in sorted(vd.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
        write(vd / "gold_solution.json", json.dumps(gold, indent=2, sort_keys=True) + "\n")
        write(vd / "workspace_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        write(vd / "hidden_tests" / "test_salvage_properties.py", hidden_test_py())
        write(vd / "oracle" / "salvage_postmortem.md", oracle_files["artifacts/delegation/salvage_postmortem.md"])
        write(vd / "oracle" / "verification.md", oracle_files["artifacts/delegation/verification.md"])
        write(vd / "oracle" / "reviewer_note.md", oracle_files["artifacts/delegation/reviewer_note.md"])
        for rel, content in oracle_overlay(variant_id).items():
            if rel.startswith("artifacts/delegation/"):
                continue
            write(vd / "oracle_workspace" / rel, content)
        for milestone_name in [
            "m1_localization.sh",
            "m2_primary_fix.sh",
            "m3_invariants.sh",
            "m4_functional.sh",
            "m5_e2e.sh",
        ]:
            target = Path("../_milestones_shared") / milestone_name.replace("m1_", "m1_localization" if milestone_name == "m1_localization.sh" else milestone_name)
            # Resolved explicitly below for portability.
        link_map = {
            "m1_localization.sh": VERIFIER_DATA / "_milestones_shared" / "m1_localization.sh",
            "m2_primary_fix.sh": VERIFIER_DATA / "_milestones_shared" / "m2_primary_fix.sh",
            "m3_invariants.sh": VERIFIER_DATA / "_milestones_shared" / "m3_invariants.sh",
            "m4_functional.sh": VERIFIER_DATA / "_milestones_shared" / "m4_functional.sh",
            "m5_e2e.sh": VERIFIER_DATA / "_milestones_shared" / "m5_e2e.sh",
        }
        milestone_dir = vd / "milestones"
        milestone_dir.mkdir(parents=True, exist_ok=True)
        for name, target in link_map.items():
            rel_target = os.path.relpath(target, milestone_dir)
            link = milestone_dir / name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(rel_target)

        manifest_entries[variant_id] = {
            "workspace_manifest_sha256": sha256_file(vd / "workspace_manifest.json"),
            "gold_solution_sha256": sha256_file(vd / "gold_solution.json"),
            "hidden_tests_tree_sha256": sha256_tree(vd, "hidden_tests"),
        }

    write(FAMILY / "manifest.lock.json", build_manifest_lock(manifest_entries))
    print(f"regenerated {FAMILY_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
