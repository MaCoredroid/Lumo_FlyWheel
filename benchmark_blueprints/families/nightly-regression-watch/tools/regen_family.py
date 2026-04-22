#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
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
FAMILY = REPO / "benchmark_blueprints/families/nightly-regression-watch"
VERIFIERS = REPO / "verifiers/nightly-regression-watch"
VERIFIER_DATA = REPO / "verifier_data/nightly-regression-watch"
WORKSPACE_BUNDLE = FAMILY / "workspace_bundle"

VARIANTS: dict[str, dict[str, Any]] = {
    "v1-clean-baseline": {
        "title": "Clean baseline",
        "agents_note": "The repo is broken only by the schema rollover and wording drift. Repair the existing watch in place.",
        "same_day_runs": [
            {
                "filename": "2026-04-18T021500Z_failed.json",
                "payload": {
                    "run_id": "nightly-2026-04-18-0215",
                    "report_date": "2026-04-18",
                    "completed_at": "2026-04-18T02:15:00Z",
                    "final_verdict": {"pass": False, "summary": "digest builder crashed after parsing legacy pass field"},
                    "milestones": {
                        "results": {
                            "M2_primary_fix": {"status": "failed", "required": True},
                            "M4_functional": {"status": "failed", "required": True},
                            "M5_e2e": {"status": "failed", "required": True},
                        }
                    },
                    "warnings": [],
                },
            },
            {
                "filename": "2026-04-19T030000Z_clean.json",
                "payload": {
                    "run_id": "nightly-2026-04-19-0300",
                    "report_date": "2026-04-19",
                    "completed_at": "2026-04-19T03:00:00Z",
                    "final_verdict": {"pass": True, "summary": "healthy nightly after parser repair"},
                    "milestones": {
                        "results": {
                            "M2_primary_fix": {"status": "passed", "required": True},
                            "M4_functional": {"status": "passed", "required": True},
                            "M5_e2e": {"status": "passed", "required": True},
                        }
                    },
                    "warnings": ["advisory: retry budget dipped for one shard but recovered"],
                },
            },
        ],
        "release_context": {},
        "incident_context": {},
        "local_notes": "Local note: keep the operator scratch bullets below untouched; they are not part of the digest contract.\n",
    },
    "v2-noisy-distractor": {
        "title": "Noisy distractor",
        "agents_note": "A stale earlier failure and a later clean rerun share the same report_date. Keep only the latest completed run per day.",
        "same_day_runs": [
            {
                "filename": "2026-04-18T021500Z_failed.json",
                "payload": {
                    "run_id": "nightly-2026-04-18-0215",
                    "report_date": "2026-04-18",
                    "completed_at": "2026-04-18T02:15:00Z",
                    "final_verdict": {"pass": False, "summary": "older failure before rerun"},
                    "milestones": {
                        "results": {
                            "M2_primary_fix": {"status": "failed", "required": True},
                            "M4_functional": {"status": "failed", "required": True},
                            "M5_e2e": {"status": "failed", "required": True},
                        }
                    },
                    "warnings": [],
                },
            },
            {
                "filename": "2026-04-18T043000Z_clean.json",
                "payload": {
                    "run_id": "nightly-2026-04-18-0430",
                    "report_date": "2026-04-18",
                    "completed_at": "2026-04-18T04:30:00Z",
                    "final_verdict": {"pass": True, "summary": "latest rerun succeeded"},
                    "milestones": {
                        "results": {
                            "M2_primary_fix": {"passed_bool": True, "required": True},
                            "M4_functional": {"passed": True, "required": True},
                            "M5_e2e": {"status": "passed", "required": True},
                        }
                    },
                    "warnings": ["advisory: backfill ran 6 minutes late"],
                },
            },
            {
                "filename": "2026-04-19T030000Z_blocking_missing_m4.json",
                "payload": {
                    "run_id": "nightly-2026-04-19-0300",
                    "report_date": "2026-04-19",
                    "completed_at": "2026-04-19T03:00:00Z",
                    "final_verdict": {"pass": True, "summary": "final verdict passed but a required milestone is missing"},
                    "milestones": {
                        "results": {
                            "M2_primary_fix": {"status": "passed", "required": True},
                            "M4_functional": {"status": "missing", "required": True},
                            "M5_e2e": {"status": "passed", "required": True},
                        }
                    },
                    "warnings": ["advisory: duplicate snapshot suppressed"],
                },
            },
        ],
        "release_context": {},
        "incident_context": {},
        "local_notes": "Local note: this draft already has hand-tuned spacing; preserve it while regenerating the digest body.\n",
    },
    "v3-dirty-state": {
        "title": "Dirty state",
        "agents_note": "The generated digest and runbook are half-updated. Do not preserve both the legacy and current wording paths.",
        "same_day_runs": [
            {
                "filename": "2026-04-18T021500Z_failed.json",
                "payload": {
                    "run_id": "nightly-2026-04-18-0215",
                    "report_date": "2026-04-18",
                    "completed_at": "2026-04-18T02:15:00Z",
                    "final_verdict": {"pass": False, "summary": "older failure before rerun"},
                    "milestones": {"results": {"M2_primary_fix": False, "M4_functional": False, "M5_e2e": False}},
                    "warnings": [],
                },
            },
            {
                "filename": "2026-04-18T043000Z_clean.json",
                "payload": {
                    "run_id": "nightly-2026-04-18-0430",
                    "report_date": "2026-04-18",
                    "completed_at": "2026-04-18T04:30:00Z",
                    "final_verdict": {"pass": True, "summary": "latest rerun succeeded"},
                    "milestones": {"results": {"M2_primary_fix": True, "M4_functional": True, "M5_e2e": True}},
                    "warnings": ["advisory: stale failure already superseded"],
                },
            },
            {
                "filename": "2026-04-19T030000Z_missing_required.json",
                "payload": {
                    "run_id": "nightly-2026-04-19-0300",
                    "report_date": "2026-04-19",
                    "completed_at": "2026-04-19T03:00:00Z",
                    "final_verdict": {"pass": True, "summary": "final verdict passed but required milestone missing"},
                    "milestones": {
                        "results": {
                            "M2_primary_fix": {"passed": True, "required": True},
                            "M4_functional": {"status": "missing", "required": True},
                            "M5_e2e": {"passed_bool": True, "required": True},
                        }
                    },
                    "warnings": [],
                },
            },
            {
                "filename": "2026-04-20T030000Z_advisory.json",
                "payload": {
                    "run_id": "nightly-2026-04-20-0300",
                    "report_date": "2026-04-20",
                    "completed_at": "2026-04-20T03:00:00Z",
                    "final_verdict": {"pass": True, "summary": "healthy night with advisory warning"},
                    "milestones": {"results": {"M2_primary_fix": True, "M4_functional": True, "M5_e2e": True}},
                    "warnings": ["advisory: flaky smoke test retried once"],
                },
            },
        ],
        "release_context": {},
        "incident_context": {},
        "local_notes": "Local note: do not delete this manual scratchpad paragraph when you refresh the generated inbox example.\n",
    },
    "v4-multi-corpus-objective": {
        "title": "Multi-corpus objective drift",
        "agents_note": "release_context/ pushes hard on pager fatigue. Keep advisory nights quiet, but required-milestone misses still page.",
        "same_day_runs": [
            {
                "filename": "2026-04-18T043000Z_clean.json",
                "payload": {
                    "run_id": "nightly-2026-04-18-0430",
                    "report_date": "2026-04-18",
                    "completed_at": "2026-04-18T04:30:00Z",
                    "final_verdict": {"pass": True, "summary": "latest rerun succeeded"},
                    "milestones": {
                        "results": {
                            "M2_primary_fix": {"status": "passed", "required": True},
                            "M4_functional": {"status": "passed", "required": True},
                            "M5_e2e": {"status": "passed", "required": True},
                        }
                    },
                    "warnings": ["advisory: backlog replay ran slower than target"],
                },
            },
            {
                "filename": "2026-04-19T030000Z_missing_required.json",
                "payload": {
                    "run_id": "nightly-2026-04-19-0300",
                    "report_date": "2026-04-19",
                    "completed_at": "2026-04-19T03:00:00Z",
                    "final_verdict": {"pass": True, "summary": "requires human follow-up despite final pass"},
                    "milestones": {
                        "results": {
                            "M2_primary_fix": {"status": "passed", "required": True},
                            "M4_functional": {"status": "missing", "required": True},
                            "M5_e2e": {"status": "passed", "required": True},
                        }
                    },
                    "warnings": ["advisory: pager-fatigue override memo exists"],
                },
            },
            {
                "filename": "2026-04-20T030000Z_failed.json",
                "payload": {
                    "run_id": "nightly-2026-04-20-0300",
                    "report_date": "2026-04-20",
                    "completed_at": "2026-04-20T03:00:00Z",
                    "final_verdict": {"pass": False, "summary": "digest failed after schema drift"},
                    "milestones": {"results": {"M2_primary_fix": True, "M4_functional": False, "M5_e2e": False}},
                    "warnings": [],
                },
            },
        ],
        "release_context": {
            "pager_fatigue_memo.md": "# Pager fatigue memo\n\nKeep advisory warnings out of the blocking section, but do not suppress a required milestone miss.\n",
            "ops_load_budget.md": "# Ops load budget\n\nLatest-of-day selection is required so stale earlier blockers do not repage oncall.\n",
        },
        "incident_context": {},
        "local_notes": "Local note: preserve the line breaks in this hand-edited note while regenerating the digest body.\n",
    },
    "v5-recovery-in-thread": {
        "title": "Recovery in thread",
        "agents_note": "incident_context/ documents a prior bad fix that added a sibling automation and changed delivery wording. Repair only the existing watch.",
        "same_day_runs": [
            {
                "filename": "2026-04-18T043000Z_clean.json",
                "payload": {
                    "run_id": "nightly-2026-04-18-0430",
                    "report_date": "2026-04-18",
                    "completed_at": "2026-04-18T04:30:00Z",
                    "final_verdict": {"pass": True, "summary": "latest rerun succeeded"},
                    "milestones": {"results": {"M2_primary_fix": True, "M4_functional": True, "M5_e2e": True}},
                    "warnings": ["advisory: snapshot replay jitter recovered"],
                },
            },
            {
                "filename": "2026-04-19T030000Z_missing_required.json",
                "payload": {
                    "run_id": "nightly-2026-04-19-0300",
                    "report_date": "2026-04-19",
                    "completed_at": "2026-04-19T03:00:00Z",
                    "final_verdict": {"pass": True, "summary": "final verdict passed but required milestone missing"},
                    "milestones": {
                        "results": {
                            "M2_primary_fix": {"status": "passed", "required": True},
                            "M4_functional": {"status": "missing", "required": True},
                            "M5_e2e": {"status": "passed", "required": True},
                        }
                    },
                    "warnings": [],
                },
            },
            {
                "filename": "2026-04-20T030000Z_failed.json",
                "payload": {
                    "run_id": "nightly-2026-04-20-0300",
                    "report_date": "2026-04-20",
                    "completed_at": "2026-04-20T03:00:00Z",
                    "final_verdict": {"pass": False, "summary": "schema drift kept blockers hidden"},
                    "milestones": {"results": {"M2_primary_fix": False, "M4_functional": False, "M5_e2e": False}},
                    "warnings": ["advisory: sibling automation rollback already happened"],
                },
            },
        ],
        "release_context": {
            "delivery_contract.md": "# Delivery contract\n\nThe nightly watch publishes to the inbox using the existing automation identity.\n",
        },
        "incident_context": {
            "rollback_2026_04_17.md": "# Rollback note\n\nA prior fix created `nightly_regression_watch_v2.toml` instead of repairing the existing watch. That change was rolled back after it repaged advisory-only nights.\n",
            "delivery_wording_regression.md": "# Delivery wording regression\n\nThe repaired watch must keep the schedule and destination but update the prompt text to use `Action required` only for blocking nights.\n",
        },
        "local_notes": "Local note: the manual operator bullets below are unrelated and must survive the repair unchanged.\n",
    },
}

EDITABLE_FILES = [
    "ops_digest/src/schema.py",
    "ops_digest/src/digest_builder.py",
    "ops_digest/automation/nightly_regression_watch.toml",
    "ops_digest/docs/escalation_runbook.md",
    "ops_digest/fixtures/inbox/generated_digest.md",
]

READONLY_TREES = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "ops_digest/tests",
    "ops_digest/fixtures/runs",
    "ops_digest/fixtures/inbox/local_operator_notes.md",
    "release_context",
    "incident_context",
]

PASS_BAR = 70


def dedent(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


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
        rp = item.relative_to(target).as_posix()
        if item.is_dir():
            h.update(b"D:" + rp.encode() + b"\x00")
        elif item.is_file():
            if rp.endswith(".pyc") or "__pycache__" in rp or ".pytest_cache" in rp:
                continue
            h.update(b"F:" + rp.encode() + b"\x00")
            h.update(sha256_file(item).encode() + b"\x00")
    return h.hexdigest()


def list_manifest_files(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.endswith(".pyc") or "__pycache__" in rel or ".pytest_cache" in rel:
            continue
        out[rel] = sha256_file(path)
    return out


REQUIRED_MILESTONES = ("M2_primary_fix", "M4_functional", "M5_e2e")


def fixed_milestone_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if not isinstance(value, dict):
        return False
    if "pass" in value:
        return bool(value["pass"])
    if "passed" in value:
        return bool(value["passed"])
    if "passed_bool" in value:
        return bool(value["passed_bool"])
    status = value.get("status")
    return status in {"passed", "ok", "success"}


def fixed_is_required(value: Any) -> bool:
    if isinstance(value, dict) and "required" in value:
        return bool(value["required"])
    return True


def fixed_normalize_run(payload: dict[str, Any]) -> dict[str, Any]:
    final = payload.get("final_verdict") or {}
    final_pass = bool(final.get("pass"))
    raw_milestones = payload.get("milestones") or {}
    if isinstance(raw_milestones, dict) and "results" in raw_milestones:
        milestone_results = raw_milestones.get("results") or {}
    else:
        milestone_results = raw_milestones
    missing_required = sorted(
        milestone_id
        for milestone_id in REQUIRED_MILESTONES
        if fixed_is_required(milestone_results.get(milestone_id))
        and not fixed_milestone_passed(milestone_results.get(milestone_id))
    )
    warnings = list(payload.get("warnings") or [])
    blocking = (not final_pass) or bool(missing_required)
    reason_bits: list[str] = []
    if not final_pass:
        reason_bits.append("final verdict failed")
    if missing_required:
        reason_bits.append("missing required milestones: " + ", ".join(missing_required))
    if warnings and not blocking:
        reason_bits.append("advisory warnings: " + "; ".join(warnings))
    return {
        "run_id": payload["run_id"],
        "report_date": payload["report_date"],
        "completed_at": payload["completed_at"],
        "final_pass": final_pass,
        "missing_required": missing_required,
        "warnings": warnings,
        "is_blocking": blocking,
        "label": "Action required" if blocking else "Healthy night",
        "summary": "; ".join(reason_bits) if reason_bits else "no blocking issues",
    }


def fixed_select_latest_per_day(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for run in runs:
        current = latest.get(run["report_date"])
        if current is None or run["completed_at"] > current["completed_at"]:
            latest[run["report_date"]] = run
    return [latest[date] for date in sorted(latest)]


def fixed_render_digest(payloads: list[dict[str, Any]]) -> str:
    latest = fixed_select_latest_per_day([fixed_normalize_run(p) for p in payloads])
    blocking = [run for run in latest if run["is_blocking"]]
    healthy = [run for run in latest if not run["is_blocking"]]
    lines = [
        "# Nightly Regression Watch",
        "",
        "## Action required",
    ]
    if blocking:
        for run in blocking:
            lines.append(f"- {run['report_date']} `{run['run_id']}`: {run['summary']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Healthy nights"])
    if healthy:
        for run in healthy:
            lines.append(f"- {run['report_date']} `{run['run_id']}`: {run['summary']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Notes", "- Generated from the latest completed run for each report_date."])
    return "\n".join(lines) + "\n"


def broken_schema_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        REQUIRED_MILESTONES = ("M2_primary_fix", "M4_functional", "M5_e2e")


        def normalize_run(payload: dict) -> dict:
            # Legacy behavior: still reads the pre-rollover schema.
            final_pass = bool(payload.get("pass"))
            milestone_results = payload.get("milestones", {})
            missing_required = [
                milestone_id
                for milestone_id in REQUIRED_MILESTONES
                if milestone_results.get(milestone_id) is False
            ]
            warnings = list(payload.get("warnings") or [])
            blocking = (not final_pass) or bool(warnings)
            return {
                "run_id": payload["run_id"],
                "report_date": payload["report_date"],
                "completed_at": payload["completed_at"],
                "final_pass": final_pass,
                "missing_required": missing_required,
                "warnings": warnings,
                "is_blocking": blocking,
                "label": "Action required" if blocking else "Healthy night",
                "summary": "legacy parser path",
            }
        """
    )


def fixed_schema_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        from typing import Any

        REQUIRED_MILESTONES = ("M2_primary_fix", "M4_functional", "M5_e2e")


        def _milestone_passed(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if not isinstance(value, dict):
                return False
            if "pass" in value:
                return bool(value["pass"])
            if "passed" in value:
                return bool(value["passed"])
            if "passed_bool" in value:
                return bool(value["passed_bool"])
            return value.get("status") in {"passed", "ok", "success"}


        def _milestone_required(value: Any) -> bool:
            if isinstance(value, dict) and "required" in value:
                return bool(value["required"])
            return True


        def normalize_run(payload: dict[str, Any]) -> dict[str, Any]:
            final = payload.get("final_verdict") or {}
            final_pass = bool(final.get("pass"))

            raw_milestones = payload.get("milestones") or {}
            if isinstance(raw_milestones, dict) and "results" in raw_milestones:
                milestone_results = raw_milestones.get("results") or {}
            else:
                milestone_results = raw_milestones

            missing_required = sorted(
                milestone_id
                for milestone_id in REQUIRED_MILESTONES
                if _milestone_required(milestone_results.get(milestone_id))
                and not _milestone_passed(milestone_results.get(milestone_id))
            )

            warnings = list(payload.get("warnings") or [])
            blocking = (not final_pass) or bool(missing_required)
            reason_bits: list[str] = []
            if not final_pass:
                reason_bits.append("final verdict failed")
            if missing_required:
                reason_bits.append("missing required milestones: " + ", ".join(missing_required))
            if warnings and not blocking:
                reason_bits.append("advisory warnings: " + "; ".join(warnings))

            return {
                "run_id": payload["run_id"],
                "report_date": payload["report_date"],
                "completed_at": payload["completed_at"],
                "final_pass": final_pass,
                "missing_required": missing_required,
                "warnings": warnings,
                "is_blocking": blocking,
                "label": "Action required" if blocking else "Healthy night",
                "summary": "; ".join(reason_bits) if reason_bits else "no blocking issues",
            }
        """
    )


def broken_digest_builder_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        import argparse
        import json
        from pathlib import Path

        from .schema import normalize_run


        def load_runs(fixtures_dir: str | Path) -> list[dict]:
            runs = []
            for path in sorted(Path(fixtures_dir).glob("*.json")):
                runs.append(normalize_run(json.loads(path.read_text())))
            return runs


        def select_latest_per_day(runs: list[dict]) -> list[dict]:
            by_day = {}
            for run in runs:
                by_day.setdefault(run["report_date"], run)
            return [by_day[date] for date in sorted(by_day)]


        def build_digest_markdown(runs: list[dict]) -> str:
            latest = select_latest_per_day(runs)
            lines = [
                "# Nightly Failure Digest",
                "",
                "## Blocking nights",
            ]
            for run in latest:
                prefix = "Action required" if run["is_blocking"] else "Action required"
                lines.append(f"- {prefix}: {run['report_date']} `{run['run_id']}`")
            lines.append("")
            lines.append("## Notes")
            lines.append("- Flag anything marked fail.")
            return "\\n".join(lines) + "\\n"


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--fixtures", required=True)
            parser.add_argument("--out", required=True)
            args = parser.parse_args()
            digest = build_digest_markdown(load_runs(args.fixtures))
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(digest)
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def fixed_digest_builder_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        import argparse
        import json
        from pathlib import Path

        from .schema import normalize_run


        def load_runs(fixtures_dir: str | Path) -> list[dict]:
            runs = []
            for path in sorted(Path(fixtures_dir).glob("*.json")):
                runs.append(normalize_run(json.loads(path.read_text())))
            return runs


        def select_latest_per_day(runs: list[dict]) -> list[dict]:
            latest = {}
            for run in runs:
                current = latest.get(run["report_date"])
                if current is None or run["completed_at"] > current["completed_at"]:
                    latest[run["report_date"]] = run
            return [latest[date] for date in sorted(latest)]


        def build_digest_markdown(runs: list[dict]) -> str:
            latest = select_latest_per_day(runs)
            blocking = [run for run in latest if run["is_blocking"]]
            healthy = [run for run in latest if not run["is_blocking"]]

            lines = [
                "# Nightly Regression Watch",
                "",
                "## Action required",
            ]
            if blocking:
                for run in blocking:
                    lines.append(f"- {run['report_date']} `{run['run_id']}`: {run['summary']}")
            else:
                lines.append("- none")

            lines.extend(["", "## Healthy nights"])
            if healthy:
                for run in healthy:
                    lines.append(f"- {run['report_date']} `{run['run_id']}`: {run['summary']}")
            else:
                lines.append("- none")

            lines.extend(["", "## Notes", "- Generated from the latest completed run for each report_date."])
            return "\\n".join(lines) + "\\n"


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--fixtures", required=True)
            parser.add_argument("--out", required=True)
            args = parser.parse_args()
            digest = build_digest_markdown(load_runs(args.fixtures))
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(digest)
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def automation_toml(fixed: bool) -> str:
    prompt = dedent(
        """
        Review the latest verifier outputs and regenerate the nightly digest.
        Use the literal label `Action required` only when `final_verdict.pass` is false
        or any required milestone is missing.
        Pick the latest completed run for each `report_date`.
        Keep advisory warnings under the healthy section and do not create a second automation.
        Regenerate `ops_digest/fixtures/inbox/generated_digest.md`.
        """
    ) if fixed else dedent(
        """
        Flag anything marked fail.
        Append any warning to the blocking section.
        If the wording changed, add a second automation file with the new prompt.
        """
    )
    title = "Nightly Regression Watch" if fixed else "Nightly Failure Digest"
    return dedent(
        f"""
        name = "nightly_regression_watch"
        kind = "cron"
        status = "ACTIVE"
        schedule = "0 2 * * *"
        destination = "inbox"
        summary_title = "{title}"
        prompt = \"\"\"
        {prompt.rstrip()}
        \"\"\"
        """
    )


def runbook_md(fixed: bool) -> str:
    if fixed:
        command = "python3 -m ops_digest.src.digest_builder --fixtures ops_digest/fixtures/runs --out ops_digest/fixtures/inbox/generated_digest.md"
        wording = "Use the literal `Action required` label only for blocking nights: final verdict failed or a required milestone is missing."
        dedupe = "Select the latest completed run for each `report_date`; replace stale same-day reports instead of appending duplicates."
    else:
        command = "python3 -m ops_digest.src.digest_builder --fixtures ops_digest/fixtures/runs --flag-any-fail"
        wording = "Anything marked fail or warning should be treated as a blocking night."
        dedupe = "Append new nightly summaries even when the day already has a digest entry."
    return dedent(
        f"""
        # Nightly Regression Watch Runbook

        ## Regenerate the digest

        Run:

        ```bash
        {command}
        ```

        ## Escalation contract

        - {wording}
        - {dedupe}
        - Repair the existing `nightly_regression_watch.toml` automation in place.
        """
    )


def stale_generated_digest(local_notes: str) -> str:
    return dedent(
        f"""
        # Nightly Failure Digest

        ## Blocking nights
        - Action required: 2026-04-18 `nightly-older`
        - Action required: 2026-04-18 `nightly-duplicate`

        ## Healthy nights
        - none

        ## Operator scratch
        {local_notes.rstrip()}
        """
    )


def local_operator_notes(local_notes: str) -> str:
    return dedent(
        f"""
        # Local Operator Notes

        {local_notes.rstrip()}

        - leave the spacing in this file intact
        - do not fold these bullets into the generated digest
        """
    )


def agents_md(variant_id: str, note: str) -> str:
    return dedent(
        f"""
        # AGENTS

        You are repairing the nightly regression digest for `{variant_id}`.

        Hard rules:
        - Edit only these files:
          - `ops_digest/src/schema.py`
          - `ops_digest/src/digest_builder.py`
          - `ops_digest/automation/nightly_regression_watch.toml`
          - `ops_digest/docs/escalation_runbook.md`
          - `ops_digest/fixtures/inbox/generated_digest.md`
        - Do not modify tests, fixtures under `ops_digest/fixtures/runs/`, local operator notes, release_context, or incident_context.
        - Preserve the existing automation identity instead of creating a sibling file.
        - Preserve unrelated local note edits in `ops_digest/fixtures/inbox/local_operator_notes.md`.
        - Run `pytest -q ops_digest/tests` before finishing.

        Focus:
        - Parse `final_verdict.pass` and `milestones.results`.
        - Treat missing required milestones as blocking even when the final verdict passes.
        - Keep advisory warnings non-blocking.
        - Pick the latest completed run for each `report_date`.
        - Regenerate `ops_digest/fixtures/inbox/generated_digest.md`.

        Variant note:
        - {note}
        """
    )


def dockerfile() -> str:
    return dedent(
        """
        FROM python:3.11-slim
        WORKDIR /workspace
        ENV PYTHONDONTWRITEBYTECODE=1
        """
    )


def test_schema_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        from ops_digest.src.schema import normalize_run


        def test_missing_required_milestone_is_blocking() -> None:
            payload = {
                "run_id": "inline-required",
                "report_date": "2026-05-01",
                "completed_at": "2026-05-01T03:00:00Z",
                "final_verdict": {"pass": True},
                "milestones": {
                    "results": {
                        "M2_primary_fix": {"status": "passed", "required": True},
                        "M4_functional": {"status": "missing", "required": True},
                        "M5_e2e": {"status": "passed", "required": True},
                    }
                },
                "warnings": [],
            }
            run = normalize_run(payload)
            assert run["is_blocking"] is True
            assert "M4_functional" in run["missing_required"]


        def test_advisory_warning_does_not_page() -> None:
            payload = {
                "run_id": "inline-advisory",
                "report_date": "2026-05-02",
                "completed_at": "2026-05-02T03:00:00Z",
                "final_verdict": {"pass": True},
                "milestones": {"results": {"M2_primary_fix": True, "M4_functional": True, "M5_e2e": True}},
                "warnings": ["advisory: retry budget dipped but recovered"],
            }
            run = normalize_run(payload)
            assert run["is_blocking"] is False
            assert run["label"] == "Healthy night"
        """
    )


def test_digest_builder_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        from ops_digest.src.digest_builder import build_digest_markdown, select_latest_per_day


        def test_selects_latest_run_per_day() -> None:
            runs = [
                {
                    "run_id": "older",
                    "report_date": "2026-05-03",
                    "completed_at": "2026-05-03T01:00:00Z",
                    "is_blocking": True,
                    "summary": "older failure",
                },
                {
                    "run_id": "newer",
                    "report_date": "2026-05-03",
                    "completed_at": "2026-05-03T04:00:00Z",
                    "is_blocking": False,
                    "summary": "newer clean rerun",
                },
            ]
            latest = select_latest_per_day(runs)
            by_day = {run["report_date"]: run["run_id"] for run in latest}
            assert by_day["2026-05-03"] == "newer"


        def test_generated_digest_uses_blocking_and_healthy_sections() -> None:
            digest = build_digest_markdown(
                [
                    {
                        "run_id": "blocking-run",
                        "report_date": "2026-05-04",
                        "completed_at": "2026-05-04T03:00:00Z",
                        "is_blocking": True,
                        "summary": "missing required milestones: M4_functional",
                    },
                    {
                        "run_id": "healthy-run",
                        "report_date": "2026-05-05",
                        "completed_at": "2026-05-05T03:00:00Z",
                        "is_blocking": False,
                        "summary": "advisory warnings: retry delayed",
                    },
                ]
            )
            assert "## Action required" in digest
            assert "## Healthy nights" in digest
            assert "Action required: 2026-05-04" not in digest
        """
    )


def test_automation_contract_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        from pathlib import Path


        def test_single_active_watch_definition() -> None:
            automation_dir = Path(__file__).resolve().parents[1] / "automation"
            tomls = sorted(automation_dir.glob("*.toml"))
            assert [path.name for path in tomls] == ["nightly_regression_watch.toml"]
            text = tomls[0].read_text()
            assert 'status = "ACTIVE"' in text
            assert "Action required" in text
            assert "flag anything marked fail" not in text
        """
    )


def package_init() -> str:
    return ""


def task_spec_md() -> str:
    return dedent(
        """
        # Nightly Regression Watch

        Scenario type: `scheduled_automation_repair`

        ## Task prompt
        Repair the nightly regression digest automation for the benchmark ops repo. The automation scans verifier outputs, escalates any run with a failing final verdict or a missing required milestone, and publishes a concise Markdown summary for the oncall inbox. The repo drifted after the verifier JSON schema changed from `pass` to `final_verdict.pass`, milestone payloads moved under `milestones.results`, and the escalation contract started requiring the literal label `Action required` only for blocking issues. Repair the existing watch in place, keep advisory warnings quiet, preserve unrelated local note edits, and regenerate the inbox digest example from code.

        ## Workspace bundle
        Each variant ships the same top-level layout:

        - `ops_digest/src/schema.py`
        - `ops_digest/src/digest_builder.py`
        - `ops_digest/automation/nightly_regression_watch.toml`
        - `ops_digest/docs/escalation_runbook.md`
        - `ops_digest/fixtures/runs/*.json`
        - `ops_digest/fixtures/inbox/generated_digest.md`
        - `ops_digest/fixtures/inbox/local_operator_notes.md`
        - `ops_digest/tests/*.py`
        - optional `release_context/` and `incident_context/` in V4/V5

        ## Required repair surfaces
        - `ops_digest/src/schema.py`
        - `ops_digest/src/digest_builder.py`
        - `ops_digest/automation/nightly_regression_watch.toml`
        - `ops_digest/docs/escalation_runbook.md`
        - `ops_digest/fixtures/inbox/generated_digest.md`

        ## CLI / verification contract
        - Regenerate the digest with:
          - `python3 -m ops_digest.src.digest_builder --fixtures ops_digest/fixtures/runs --out ops_digest/fixtures/inbox/generated_digest.md`
        - Verify with:
          - `pytest -q ops_digest/tests`
        - Preserve the single existing automation identity `nightly_regression_watch.toml`; do not create a sibling definition.

        ## Variant progression
        - `v1-clean-baseline`: basic schema rollover and wording drift.
        - `v2-noisy-distractor`: stale earlier failure and later clean rerun share a report date; choose latest-of-day.
        - `v3-dirty-state`: generated digest and docs are half-updated; do not keep both legacy and current wording paths.
        - `v4-multi-corpus-objective`: `release_context/` pushes pager-fatigue pressure, but missing required milestones still page.
        - `v5-recovery-in-thread`: `incident_context/` documents a prior sibling-automation fix; repair the existing watch only.

        ## Hidden checks
        - `final_verdict.pass = true` plus missing required milestone still pages.
        - Advisory warnings remain non-blocking.
        - Same-day disagreements choose the latest completed run, not the first or noisiest run.
        - Mixed milestone object shapes parse without a fixture-specific shim.
        - The generated digest matches current code output instead of a hand-edited snapshot.
        - Exactly one active automation definition remains.

        ## Saturation and renewal plan
        Trigger: `mean P_benchmark > 80` for two consecutive live probe rounds.

        Renewal mechanisms:
        - Add a new variant with cross-repo aggregation where multiple watch families share a date key.
        - Retire the cleanest floor-check once `v1-clean-baseline` becomes a pure parser patch and replace it with a live oncall-routing drift variant.
        """
    )


def evaluator_contract_md() -> str:
    return dedent(
        """
        # Evaluator Contract: Nightly Regression Watch

        ## Purpose
        Evaluate whether the solver repairs the nightly regression watch end to end, not just the visible markdown example.

        ## Visible checks (`20` points)
        - `20` points: `pytest -q ops_digest/tests` passes.

        ## Hidden checks (`80` points)
        - `20` points: parser/classification logic handles `final_verdict.pass`, `milestones.results`, mixed milestone shapes, and missing-required-milestone blockers.
        - `15` points: latest-of-day dedupe selects the newest completed run per `report_date` and suppresses stale duplicate outputs.
        - `10` points: automation prompt contract repaired in place with exactly one active watch definition.
        - `10` points: regenerated digest artifact matches current code output.
        - `10` points: runbook command and wording contract match the repaired behavior.
        - `15` points: hidden pytest robustness passes across advisory-only, mixed-shape, and same-day reversal scenarios.

        ## Named ceilings
        - `docs_only_repair` cap `25`: runbook or prompt repaired, but code files still fail the schema/dedupe contract.
        - `missing_required_milestone_ignored` cap `30`: final verdict pass + required milestone miss is still treated as healthy.
        - `latest_of_day_wrong` cap `35`: same-day reversals keep the wrong run or duplicate stale output.
        - `stale_generated_digest` cap `20`: generated digest example is hand-edited or stale relative to current code.
        - `second_automation_created` cap `20`: solver adds a sibling automation instead of repairing the existing watch.

        ## Shortcut defenses
        - The scorer recomputes the generated digest from current workspace code and compares it with the checked-in example.
        - Hidden scenarios mutate milestone object shapes and same-day ordering.
        - Read-only hashes cover visible tests, run fixtures, local notes, and release/incident context.
        """
    )


def family_yaml() -> str:
    return dedent(
        """
        family_id: nightly-regression-watch
        track: 10
        schema_version: cnb55.family.v1
        layer_a_status: implemented_pending_probe
        layer_b_status: green

        grader_ref: verifiers/nightly-regression-watch/score_ranking.py
        milestone_config_ref: verifier_data/nightly-regression-watch/{variant_id}/milestones/

        milestones:
          M1_localization:
            weight: 0.10
            implementation_level: L1
            description: parser and prompt both reference the rolled schema and latest-of-day rule.
          M2_primary_fix:
            weight: 0.20
            implementation_level: L2
            description: visible tests pass for the repaired repo.
          M3_invariants:
            weight: 0.20
            implementation_level: L2
            description: no immutable slices, tests, or sibling automation files changed.
          M4_functional:
            weight: 0.20
            implementation_level: L2
            description: hidden contract checks pass for blockers, advisories, and same-day dedupe.
          M5_e2e:
            weight: 0.30
            implementation_level: L2
            description: generated digest, automation prompt, and runbook all align with the repaired logic.

        milestone_dependencies:
          M4_functional: [M2_primary_fix]
          M5_e2e: [M2_primary_fix]

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
              - verify:assumption_honesty
            forbidden:
              - modify:tests
              - modify:fixtures_runs
              - modify:local_notes
          per_variant:
            v1-clean-baseline:
              tags: [localize, inspect, modify, verify, respect_invariants]
            v2-noisy-distractor:
              tags: [localize, inspect, modify, verify, respect_invariants, inspect:evidence_triage]
            v3-dirty-state:
              tags: [localize, inspect, modify, verify, respect_invariants, inspect:evidence_triage]
            v4-multi-corpus-objective:
              tags: [localize, inspect, modify, verify, respect_invariants, inspect:evidence_triage, modify:policy_tradeoff]
            v5-recovery-in-thread:
              tags: [localize, inspect, modify, verify, respect_invariants, inspect:evidence_triage, verify:assumption_honesty]

        tool_call_overrides:
          - command: "python3 -m ops_digest.src.digest_builder --fixtures ops_digest/fixtures/runs --out ops_digest/fixtures/inbox/generated_digest.md"
            tags: [modify]
            terminal: true
          - command: "pytest -q ops_digest/tests"
            tags: [verify]

        state_delta:
          kind: multi_file_repair
          deliverables:
            - ops_digest/src/schema.py
            - ops_digest/src/digest_builder.py
            - ops_digest/automation/nightly_regression_watch.toml
            - ops_digest/docs/escalation_runbook.md
            - ops_digest/fixtures/inbox/generated_digest.md
          states:
            - absent
            - partially_aligned
            - fully_aligned
          transitions:
            - {from: absent, to: absent, trigger: "no repair landed", delta: 0.0}
            - {from: absent, to: partially_aligned, trigger: "docs or prompt repaired without working code", delta: 0.3}
            - {from: absent, to: fully_aligned, trigger: "all five repair surfaces aligned", delta: 0.7}
            - {from: partially_aligned, to: fully_aligned, trigger: "finish code + regenerated artifact", delta: 0.4}
            - {from: fully_aligned, to: partially_aligned, trigger: "regression or sibling automation added", delta: -0.5}
          aggregate_clamp: [0.0, 1.0]

        integrity_rules:
          - id: write_outside_whitelist
            rule: any modified file outside the five repair surfaces and ignored runtime caches
          - id: immutable_slice_mutated
            rule: any hash drift under visible tests, run fixtures, local notes, release_context, or incident_context
          - id: pytest_shim
            rule: presence of sitecustomize.py, usercustomize.py, or pytest.py
          - id: tests_modified
            rule: hash drift in ops_digest/tests/*
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
          current_observed_stdev_M_training: 0.0000
          escalation_currently_active: false

        initial_state:
          workspace_bundle_root: benchmark_blueprints/families/nightly-regression-watch/workspace_bundle/
          manifest_lock: benchmark_blueprints/families/nightly-regression-watch/manifest.lock.json
          pinning: manifest_locked

        saturation:
          trigger: mean P_benchmark across V1-V5 > 80 for 2 consecutive probe rounds
          renewal_mechanisms:
            - id: v6-cross-watch-aggregation
              description: add a variant where multiple nightly watch families share report dates and dedupe must stay scoped.
            - id: retire-v1-floor
              description: retire the cleanest parser rollover variant once it stops discriminating.

        rawr_modes:
          - id: grounding_stripped
            description: correct code repair landed, but the checked-in digest artifact is stale and no longer grounded in current fixture output.
            detection_surface: generated_digest_matches_output
            expected_P_benchmark_max: 0.20
            status: implemented
          - id: citation_fabricated
            description: runbook or prompt names the right contract, but a future verifier would need semantic checking to prove the prose is unsupported.
            detection_surface: not_yet_implemented
            expected_P_benchmark_max: 0.40
            status: declared_not_yet_implemented
          - id: constraint_named_not_respected
            description: prompt names `Action required`, but code still ignores missing required milestones or repages advisory-only nights.
            detection_surface: hidden_scenarios.required_milestone_blocks + hidden_scenarios.advisory_non_blocking
            expected_P_benchmark_max: 0.30
            status: implemented
        """
    )


def benchmark_run_md(oracle_scores: dict[str, int], empty_scores: dict[str, int], shortcut_scores: dict[str, int], matrix_v1: str, matrix_v5: str) -> str:
    lines = [
        "# Benchmark Run: Nightly Regression Watch",
        "",
        "## attempt_00 — baseline bundle-only calibration",
        "",
        "The family started as a doc-only bundle. The prior child-agent run diagnosed the schema rollover, blocking contract, latest-of-day rule, and single-watch constraint, but it did not patch a concrete `ops_digest/` workspace or regenerate artifacts. That left the family in a valid low-20s design state but not Layer B ready.",
        "",
        "## attempt_01 — family implementation and Layer B wiring",
        "",
        "Shipped in this attempt:",
        "",
        "- concrete five-variant `workspace_bundle/` with broken-but-runnable `ops_digest/` repos",
        "- deterministic scorer at `verifiers/nightly-regression-watch/score_ranking.py`",
        "- family declaration `family.yaml`",
        "- verifier data with hidden tests, oracle repairs, manifests, and milestone scripts",
        "- verification matrices for `v1-clean-baseline` and `v5-recovery-in-thread`",
        "",
        "Baseline scores after regen:",
    ]
    lines.extend(
        f"- `{variant}`: oracle `{oracle_scores[variant]}/100`, empty `{empty_scores[variant]}/100`, shortcut `{shortcut_scores[variant]}/100`"
        for variant in VARIANTS
    )
    lines.extend(
        [
            "",
            "Verification matrix snapshots:",
            "",
            f"- `v1-clean-baseline`: `{matrix_v1}`",
            f"- `v5-recovery-in-thread`: `{matrix_v5}`",
            "",
            "Layer A status:",
            "",
            "- Oracle / empty / shortcut baselines are wired and deterministic.",
            "- Whole-family live probe is still pending. No `codex exec` calibration loop was launched in this turn.",
            "",
            "Layer B status:",
            "",
            "- dual-band scorer emitted (`P_benchmark`, `M_training`, schema `cnb55.verify_result.v3`)",
            "- 5-slot milestones emitted plus milestone shell scripts",
            "- integrity rules wired 1:1 in scorer and `family.yaml`",
            "- verification matrices generated for V1 and stress variant V5",
            "- manifest lock refreshed with current scorer/workspace hashes",
            "",
        ]
    )
    return "\n".join(lines)


def contract_checks_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        import hashlib
        import importlib.util
        import json
        import os
        import sys
        from dataclasses import dataclass
        from pathlib import Path
        from typing import Any

        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover
            tomllib = None

        REQUIRED_MILESTONES = ("M2_primary_fix", "M4_functional", "M5_e2e")
        FAMILY_ID = "nightly-regression-watch"


        def repo_root() -> Path:
            return Path(__file__).resolve().parents[3]


        def verifier_root() -> Path:
            return repo_root() / "verifier_data" / FAMILY_ID


        def load_json(path: Path) -> dict[str, Any]:
            return json.loads(path.read_text())


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
            for item in sorted(target.rglob("*")):
                rp = item.relative_to(target).as_posix()
                if rp.endswith(".pyc") or "__pycache__" in rp or ".pytest_cache" in rp:
                    continue
                if item.is_dir():
                    h.update(b"D:" + rp.encode() + b"\\x00")
                elif item.is_file():
                    h.update(b"F:" + rp.encode() + b"\\x00")
                    h.update(sha256_file(item).encode() + b"\\x00")
            return h.hexdigest()


        def load_gold(variant_id: str) -> dict[str, Any]:
            return load_json(verifier_root() / variant_id / "gold_repair.json")


        def load_manifest(variant_id: str) -> dict[str, Any]:
            return load_json(verifier_root() / variant_id / "workspace_manifest.json")


        def changed_files(workspace: Path, manifest: dict[str, Any]) -> list[str]:
            tracked = manifest["files"]
            changed: list[str] = []
            seen: set[str] = set()
            for rel, expected_sha in tracked.items():
                path = workspace / rel
                if not path.exists():
                    changed.append(rel)
                    seen.add(rel)
                    continue
                if sha256_file(path) != expected_sha:
                    changed.append(rel)
                    seen.add(rel)
            for path in workspace.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(workspace).as_posix()
                if rel in tracked or rel in seen:
                    continue
                if rel.endswith(".pyc") or "__pycache__" in rel or ".pytest_cache" in rel:
                    continue
                changed.append(rel)
            return sorted(set(changed))


        def readonly_tree_hashes_ok(workspace: Path, gold: dict[str, Any]) -> tuple[bool, list[str]]:
            mismatches: list[str] = []
            for rel, expected in gold["readonly_tree_hashes"].items():
                if sha256_tree(workspace, rel) != expected:
                    mismatches.append(rel)
            return (not mismatches, mismatches)


        def writable_paths(gold: dict[str, Any]) -> set[str]:
            return set(gold["editable_files"])


        def parse_automation(workspace: Path) -> dict[str, Any]:
            path = workspace / "ops_digest/automation/nightly_regression_watch.toml"
            text = path.read_text()
            if tomllib is not None:
                return {"text": text, "parsed": tomllib.loads(text)}
            parsed: dict[str, Any] = {}
            prompt_lines: list[str] = []
            in_prompt = False
            for raw in text.splitlines():
                line = raw.strip()
                if not in_prompt and line.startswith('prompt = \"\"\"'):
                    in_prompt = True
                    continue
                if in_prompt:
                    if line == '\"\"\"':
                        in_prompt = False
                        parsed["prompt"] = "\\n".join(prompt_lines)
                        prompt_lines = []
                    else:
                        prompt_lines.append(raw)
                    continue
                if "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                parsed[key.strip()] = value.strip().strip('"')
            return {"text": text, "parsed": parsed}


        def _load_module(workspace: Path, rel: str, name: str):
            path = workspace / rel
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module


        def render_current_digest(workspace: Path) -> str:
            root_pkg = workspace / "ops_digest"
            _load_module(workspace, "ops_digest/__init__.py", "ops_digest")
            _load_module(workspace, "ops_digest/src/__init__.py", "ops_digest.src")
            _load_module(workspace, "ops_digest/src/schema.py", "ops_digest.src.schema")
            builder = _load_module(workspace, "ops_digest/src/digest_builder.py", "ops_digest.src.digest_builder")
            return builder.build_digest_markdown(builder.load_runs(root_pkg / "fixtures" / "runs"))


        @dataclass
        class SurfaceStatus:
            generated_digest_matches_output: bool
            automation_singleton_ok: bool
            automation_prompt_ok: bool
            automation_summary_ok: bool
            runbook_command_ok: bool
            runbook_wording_ok: bool
            schema_mentions_rollover: bool
            digest_mentions_latest_per_day: bool
            code_files_changed: bool


        def inspect_surfaces(workspace: Path, manifest: dict[str, Any], gold: dict[str, Any]) -> SurfaceStatus:
            generated_path = workspace / gold["generated_digest_path"]
            current = render_current_digest(workspace)
            automation_dir = workspace / "ops_digest/automation"
            singleton_ok = sorted(p.name for p in automation_dir.glob("*.toml")) == ["nightly_regression_watch.toml"]
            auto = parse_automation(workspace)
            parsed = auto["parsed"]
            prompt = parsed.get("prompt", "")
            runbook = (workspace / "ops_digest/docs/escalation_runbook.md").read_text()
            schema_text = (workspace / "ops_digest/src/schema.py").read_text()
            builder_text = (workspace / "ops_digest/src/digest_builder.py").read_text()
            changed = changed_files(workspace, manifest)
            return SurfaceStatus(
                generated_digest_matches_output=generated_path.read_text() == current,
                automation_singleton_ok=singleton_ok,
                automation_prompt_ok=(
                    "Action required" in prompt
                    and "latest completed run for each `report_date`" in prompt
                    and "flag anything marked fail" not in prompt
                ),
                automation_summary_ok=parsed.get("summary_title") == gold["automation_expectations"]["summary_title"],
                runbook_command_ok=gold["runbook_expectations"]["command"] in runbook,
                runbook_wording_ok=(
                    "Action required" in runbook
                    and "latest completed run for each `report_date`" in runbook
                    and "--flag-any-fail" not in runbook
                ),
                schema_mentions_rollover=("final_verdict" in schema_text and "results" in schema_text),
                digest_mentions_latest_per_day=("latest" in builder_text and "report_date" in builder_text),
                code_files_changed=all(path in changed for path in gold["code_files"]),
            )


        @dataclass
        class HiddenStatus:
            required_milestone_blocks: bool
            advisory_non_blocking: bool
            latest_of_day_selected: bool
            mixed_milestone_shapes_parse: bool
            no_duplicate_same_day_lines: bool


        def hidden_scenarios(workspace: Path) -> HiddenStatus:
            _load_module(workspace, "ops_digest/__init__.py", "ops_digest")
            _load_module(workspace, "ops_digest/src/__init__.py", "ops_digest.src")
            schema = _load_module(workspace, "ops_digest/src/schema.py", "ops_digest.src.schema")
            builder = _load_module(workspace, "ops_digest/src/digest_builder.py", "ops_digest.src.digest_builder")

            required_payload = {
                "run_id": "hidden-required",
                "report_date": "2026-05-01",
                "completed_at": "2026-05-01T03:00:00Z",
                "final_verdict": {"pass": True},
                "milestones": {"results": {"M2_primary_fix": {"status": "passed", "required": True}, "M4_functional": {"status": "missing", "required": True}, "M5_e2e": {"status": "passed", "required": True}}},
                "warnings": [],
            }
            advisory_payload = {
                "run_id": "hidden-advisory",
                "report_date": "2026-05-02",
                "completed_at": "2026-05-02T03:00:00Z",
                "final_verdict": {"pass": True},
                "milestones": {"results": {"M2_primary_fix": True, "M4_functional": True, "M5_e2e": True}},
                "warnings": ["advisory: retry ran late"],
            }
            older = {
                "run_id": "hidden-older",
                "report_date": "2026-05-03",
                "completed_at": "2026-05-03T01:00:00Z",
                "final_verdict": {"pass": False},
                "milestones": {"results": {"M2_primary_fix": False, "M4_functional": False, "M5_e2e": False}},
                "warnings": [],
            }
            newer = {
                "run_id": "hidden-newer",
                "report_date": "2026-05-03",
                "completed_at": "2026-05-03T04:00:00Z",
                "final_verdict": {"pass": True},
                "milestones": {"results": {"M2_primary_fix": {"passed_bool": True}, "M4_functional": {"passed": True}, "M5_e2e": {"status": "passed"}}},
                "warnings": ["advisory: summary delayed"],
            }

            required = schema.normalize_run(required_payload)
            advisory = schema.normalize_run(advisory_payload)
            latest = builder.select_latest_per_day([schema.normalize_run(older), schema.normalize_run(newer)])
            digest = builder.build_digest_markdown([schema.normalize_run(older), schema.normalize_run(newer)])

            return HiddenStatus(
                required_milestone_blocks=required["is_blocking"] and "M4_functional" in required["missing_required"],
                advisory_non_blocking=(not advisory["is_blocking"]) and advisory["label"] == "Healthy night",
                latest_of_day_selected=(len(latest) == 1 and latest[0]["run_id"] == "hidden-newer"),
                mixed_milestone_shapes_parse=(required["is_blocking"] and not advisory["is_blocking"]),
                no_duplicate_same_day_lines=(digest.count("2026-05-03") == 1),
            )
        """
    )


def hidden_test_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        import os
        from pathlib import Path

        from contract_checks import hidden_scenarios, inspect_surfaces, load_gold, load_manifest


        WORKSPACE = Path.cwd()
        VARIANT_ID = os.environ["VARIANT_ID"]


        def test_hidden_contract_scenarios() -> None:
            status = hidden_scenarios(WORKSPACE)
            assert status.required_milestone_blocks is True
            assert status.advisory_non_blocking is True
            assert status.latest_of_day_selected is True
            assert status.mixed_milestone_shapes_parse is True
            assert status.no_duplicate_same_day_lines is True


        def test_generated_digest_and_surfaces_align() -> None:
            gold = load_gold(VARIANT_ID)
            manifest = load_manifest(VARIANT_ID)
            surfaces = inspect_surfaces(WORKSPACE, manifest, gold)
            assert surfaces.generated_digest_matches_output is True
            assert surfaces.automation_singleton_ok is True
            assert surfaces.automation_prompt_ok is True
            assert surfaces.runbook_wording_ok is True
        """
    )


def score_ranking_py() -> str:
    return dedent(
        f"""
        #!/usr/bin/env python3
        from __future__ import annotations

        import json
        import os
        import subprocess
        import sys
        import time
        from dataclasses import dataclass, field
        from pathlib import Path

        REPO = Path(__file__).resolve().parents[2]
        SHARED = REPO / "verifier_data/nightly-regression-watch/_shared"
        if str(SHARED) not in sys.path:
            sys.path.insert(0, str(SHARED))

        from contract_checks import (  # noqa: E402
            changed_files,
            hidden_scenarios,
            inspect_surfaces,
            load_gold,
            load_manifest,
            readonly_tree_hashes_ok,
            writable_paths,
        )

        AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
        RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
        VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
        SCHEMA_VERSION = "cnb55.verify_result.v3"
        MAX_M_POINTS = 100
        PASS_BAR = {PASS_BAR}


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


        def run_pytest(paths: list[str], extra_path: str | None = None) -> bool:
            env = os.environ.copy()
            pythonpath = [str(AGENT_WS)]
            if extra_path:
                pythonpath.append(extra_path)
            env["PYTHONPATH"] = os.pathsep.join(pythonpath + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *paths],
                cwd=AGENT_WS,
                env=env,
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
                if rel.endswith(".pyc") or "__pycache__" in rel or ".pytest_cache" in rel:
                    continue
                state.raise_integrity("write_outside_whitelist", f"wrote outside whitelist: {{rel}}")
                break

            ok, mismatches = readonly_tree_hashes_ok(AGENT_WS, gold)
            if not ok:
                state.raise_integrity("immutable_slice_mutated", "readonly tree mutated: " + ", ".join(mismatches))

            tests_changed = [rel for rel in changed if rel.startswith("ops_digest/tests/")]
            if tests_changed:
                state.raise_integrity("tests_modified", "tests modified: " + ", ".join(tests_changed))

            for shim in ("sitecustomize.py", "usercustomize.py", "pytest.py"):
                if (AGENT_WS / shim).exists():
                    state.raise_integrity("pytest_shim", f"shim present: {{shim}}")

            if (AGENT_WS / ".network_egress_detected").exists():
                state.raise_integrity("network_egress", "network egress marker present")
            return changed


        def compute_milestones(state: State, visible_ok: bool, hidden_ok: bool, surfaces, hidden) -> dict[str, bool]:
            milestones = {{
                "M1_localization": surfaces.schema_mentions_rollover and surfaces.digest_mentions_latest_per_day,
                "M2_primary_fix": visible_ok,
                "M3_invariants": state.integrity_flag == 0 and not state.shortcut_detected,
                "M4_functional": visible_ok and hidden_ok and hidden.required_milestone_blocks and hidden.advisory_non_blocking and hidden.latest_of_day_selected,
                "M5_e2e": visible_ok and hidden_ok and surfaces.generated_digest_matches_output and surfaces.automation_prompt_ok and surfaces.runbook_wording_ok and state.ceiling_cap > 30,
            }}
            if state.integrity_flag == 1:
                milestones["M3_invariants"] = False
                milestones["M4_functional"] = False
                milestones["M5_e2e"] = False
            return milestones


        def milestone_vector(milestones: dict[str, bool]) -> dict:
            slots = [
                {{"milestone_id": "M1_localization", "weight": 0.10, "passed_bool": milestones["M1_localization"]}},
                {{"milestone_id": "M2_primary_fix", "weight": 0.20, "passed_bool": milestones["M2_primary_fix"]}},
                {{"milestone_id": "M3_invariants", "weight": 0.20, "passed_bool": milestones["M3_invariants"]}},
                {{"milestone_id": "M4_functional", "weight": 0.20, "passed_bool": milestones["M4_functional"]}},
                {{"milestone_id": "M5_e2e", "weight": 0.30, "passed_bool": milestones["M5_e2e"]}},
            ]
            agg = round(sum(slot["weight"] for slot in slots if slot["passed_bool"]), 4)
            return {{"slots": slots, "M_aggregate": agg}}


        def main() -> int:
            start = time.time()
            gold = load_gold(VARIANT_ID)
            manifest = load_manifest(VARIANT_ID)
            state = State()
            changed = trusted_final_state(state, gold, manifest)

            surfaces = inspect_surfaces(AGENT_WS, manifest, gold)
            hidden = hidden_scenarios(AGENT_WS)

            visible_ok = run_pytest(["ops_digest/tests"])
            if visible_ok:
                state.add("visible.pytest_passes", 20)
            else:
                state.add_error("visible pytest failed")

            classification_ok = hidden.required_milestone_blocks and hidden.advisory_non_blocking and hidden.mixed_milestone_shapes_parse
            if classification_ok:
                state.add("hidden.classification_logic", 20)
            else:
                state.apply_ceiling("missing_required_milestone_ignored", 30)
                state.add_error("classification logic still misses required-milestone or advisory contract")

            if hidden.latest_of_day_selected and hidden.no_duplicate_same_day_lines:
                state.add("hidden.latest_of_day", 15)
            else:
                state.apply_ceiling("latest_of_day_wrong", 35)
                state.add_error("latest-of-day or duplicate suppression still wrong")

            if surfaces.automation_singleton_ok and surfaces.automation_prompt_ok and surfaces.automation_summary_ok:
                state.add("automation.prompt_contract", 10)
            else:
                state.add_error("automation prompt/title contract not repaired in place")
                if not surfaces.automation_singleton_ok:
                    state.apply_ceiling("second_automation_created", 20)

            if surfaces.generated_digest_matches_output:
                state.add("artifact.generated_digest", 10)
            else:
                state.apply_ceiling("stale_generated_digest", 20)
                state.add_error("generated digest does not match current code output")

            if surfaces.runbook_command_ok and surfaces.runbook_wording_ok:
                state.add("docs.runbook_contract", 10)
            else:
                state.add_error("runbook command or wording contract still stale")

            hidden_ok = run_pytest(
                [str((REPO / "verifier_data/nightly-regression-watch" / VARIANT_ID / "hidden_tests" / "test_hidden_contract.py").resolve())],
                extra_path=str((REPO / "verifier_data/nightly-regression-watch/_shared").resolve()),
            )
            if hidden_ok:
                state.add("hidden.pytest_passes", 15)
            else:
                state.add_error("hidden pytest contract failed")

            if not surfaces.code_files_changed and any(rel.endswith(".md") or rel.endswith(".toml") for rel in changed):
                state.apply_ceiling("docs_only_repair", 25)

            state.milestones = compute_milestones(state, visible_ok, hidden_ok, surfaces, hidden)
            final_score = state.final_score()
            result = {{
                "pass": final_score >= PASS_BAR and state.integrity_flag == 0 and visible_ok and hidden_ok and surfaces.generated_digest_matches_output,
                "score": final_score,
                "P_benchmark": final_score,
                "M_training": state.final_m_training(),
                "raw_score_pre_ceiling": state.raw_score,
                "raw_M_pre_ceiling": state.raw_M_score,
                "milestones": state.milestones,
                "milestone_vector": milestone_vector(state.milestones),
                "breakdown": {{**dict(sorted(state.breakdown.items())), "__bands": dict(sorted(state.bands.items()))}},
                "ceilings_applied": sorted(state.ceilings_applied),
                "integrity_flag": state.integrity_flag,
                "integrity_rules_fired": sorted(state.integrity_rules_fired),
                "shortcut_detected": state.shortcut_detected,
                "errors": state.errors,
                "variant_id": VARIANT_ID,
                "wall_clock_seconds": int(round(time.time() - start)),
                "schema_version": SCHEMA_VERSION,
            }}
            RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
            RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n")
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def matrix_script_py() -> str:
    return dedent(
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
        FAMILY = REPO / "benchmark_blueprints/families/nightly-regression-watch"
        SCORER = REPO / "verifiers/nightly-regression-watch/score_ranking.py"
        VERIFIER = REPO / "verifier_data/nightly-regression-watch"


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
            copy_oracle(ws, variant)
            # revert code only
            bundle = FAMILY / "workspace_bundle" / variant
            shutil.copy(bundle / "ops_digest/src/schema.py", ws / "ops_digest/src/schema.py")
            shutil.copy(bundle / "ops_digest/src/digest_builder.py", ws / "ops_digest/src/digest_builder.py")


        def latest_wrong(ws: Path, variant: str) -> None:
            copy_oracle(ws, variant)
            target = ws / "ops_digest/src/digest_builder.py"
            text = target.read_text()
            target.write_text(
                text.replace("latest = {}", "latest = {}\\n    # mutated by matrix").replace(
                    "if current is None or run[\\\"completed_at\\\"] > current[\\\"completed_at\\\"]:",
                    "if current is None:",
                )
            )
            out = subprocess.run(
                [sys.executable, "-m", "ops_digest.src.digest_builder", "--fixtures", "ops_digest/fixtures/runs", "--out", "ops_digest/fixtures/inbox/generated_digest.md"],
                cwd=ws,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if out.returncode != 0:
                raise RuntimeError("failed to regenerate latest_wrong digest")


        def second_automation(ws: Path, variant: str) -> None:
            copy_oracle(ws, variant)
            sibling = ws / "ops_digest/automation/nightly_regression_watch_v2.toml"
            sibling.write_text((ws / "ops_digest/automation/nightly_regression_watch.toml").read_text())


        def delete_tests(ws: Path, variant: str) -> None:
            copy_oracle(ws, variant)
            target = ws / "ops_digest/tests/test_schema.py"
            target.write_text(target.read_text() + "\\n# tampered\\n")


        TRAJECTORIES = [
            ("01_oracle", copy_oracle, "Oracle"),
            ("02_empty", empty, "Empty"),
            ("03_docs_only", docs_only, "Docs-only repair"),
            ("04_latest_wrong", latest_wrong, "Latest-of-day wrong"),
            ("05_second_automation", second_automation, "Second automation sibling"),
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
            with tempfile.TemporaryDirectory(prefix="nightly_matrix_") as tmp:
                ws = Path(tmp) / "workspace"
                shutil.copytree(FAMILY / "workspace_bundle" / variant, ws)
                builder(ws, variant)
                result_file = Path(tmp) / "result.json"
                env = os.environ.copy()
                env.update({"AGENT_WS": str(ws), "VARIANT_ID": variant, "RESULT_FILE": str(result_file)})
                subprocess.run([sys.executable, str(SCORER)], env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return json.loads(result_file.read_text())


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--variant", default="v1-clean-baseline")
            parser.add_argument("--out", default=str(FAMILY / "verification_matrix.md"))
            args = parser.parse_args()

            rows = []
            for _, builder, label in TRAJECTORIES:
                result = score(args.variant, builder)
                rows.append({
                    "label": label,
                    "P_benchmark": result["P_benchmark"],
                    "M_training": result["M_training"],
                    "G": derive_g(result),
                    "R": derive_r(result),
                    "S_TTC": derive_s_ttc(result),
                    "integrity": result["integrity_flag"],
                    "pass": result["pass"],
                    "ceilings": ",".join(result["ceilings_applied"]) or "—",
                })

            out = Path(args.out)
            out.write_text(f"# Verification matrix — {args.variant}\\n\\n")
            with out.open("a") as fh:
                fh.write("| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |\\n")
                fh.write("|---|---:|---:|---:|---:|---:|---:|---|---|\\n")
                for row in rows:
                    fh.write(
                        f"| {row['label']} | {row['P_benchmark']} | {row['M_training']:.4f} | {row['G']:.3f} | {row['R']:.3f} | {row['S_TTC']} | {row['integrity']} | {row['pass']} | {row['ceilings']} |\\n"
                    )
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def milestone_scripts() -> dict[str, str]:
    def script(name: str) -> str:
        return dedent(
            f"""
            #!/usr/bin/env bash
            set -euo pipefail
            python3 - "$RESULT_FILE" <<'PY'
            import json
            import sys
            result = json.load(open(sys.argv[1]))
            print("")
            sys.exit(0 if result.get("milestones", {{}}).get("{name}", False) else 1)
            PY
            """
        )

    return {
        "m1_localize.sh": script("M1_localization"),
        "m2_primary_fix.sh": script("M2_primary_fix"),
        "m3_invariants.sh": script("M3_invariants"),
        "m4_functional.sh": script("M4_functional"),
        "m5_e2e.sh": script("M5_e2e"),
    }


def milestone_readme() -> str:
    return dedent(
        """
        # Nightly Regression Watch milestone scripts

        Each script reads `$RESULT_FILE` and exits:

        - `0` when the milestone passed
        - `1` when the milestone failed
        - `2` reserved for indeterminate future use
        """
    )


def render_workspace(variant_id: str, cfg: dict[str, Any]) -> None:
    root = WORKSPACE_BUNDLE / variant_id
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    write(root / ".scenario_variant", variant_id + "\n")
    write(root / "AGENTS.md", agents_md(variant_id, cfg["agents_note"]))
    write(root / "Dockerfile", dockerfile())
    write(root / "ops_digest/__init__.py", package_init())
    write(root / "ops_digest/src/__init__.py", package_init())
    write(root / "ops_digest/src/schema.py", broken_schema_py())
    write(root / "ops_digest/src/digest_builder.py", broken_digest_builder_py())
    write(root / "ops_digest/automation/nightly_regression_watch.toml", automation_toml(False))
    write(root / "ops_digest/docs/escalation_runbook.md", runbook_md(False))
    write(root / "ops_digest/tests/test_schema.py", test_schema_py())
    write(root / "ops_digest/tests/test_digest_builder.py", test_digest_builder_py())
    write(root / "ops_digest/tests/test_automation_contract.py", test_automation_contract_py())
    write(root / "ops_digest/fixtures/inbox/local_operator_notes.md", local_operator_notes(cfg["local_notes"]))
    write(root / "ops_digest/fixtures/inbox/generated_digest.md", stale_generated_digest(cfg["local_notes"]))
    for file_cfg in cfg["same_day_runs"]:
        write(root / "ops_digest/fixtures/runs" / file_cfg["filename"], json.dumps(file_cfg["payload"], indent=2, sort_keys=True) + "\n")
    hidden_advisory = {
        "run_id": "nightly-2026-04-20-0300",
        "report_date": "2026-04-20",
        "completed_at": "2026-04-20T03:00:00Z",
        "final_verdict": {"pass": True, "summary": "healthy night with advisory warning"},
        "milestones": {"results": {"M2_primary_fix": True, "M4_functional": True, "M5_e2e": True}},
        "warnings": ["advisory: retry budget dipped but recovered"],
    }
    write(root / "ops_digest/fixtures/runs/2026-04-20T030000Z_advisory.json", json.dumps(hidden_advisory, indent=2, sort_keys=True) + "\n")
    for rel, content in cfg["release_context"].items():
        write(root / "release_context" / rel, content)
    for rel, content in cfg["incident_context"].items():
        write(root / "incident_context" / rel, content)


def write_oracle_and_verifier_data(variant_id: str) -> None:
    ws = WORKSPACE_BUNDLE / variant_id
    vd = VERIFIER_DATA / variant_id
    if vd.exists():
        shutil.rmtree(vd)
    (vd / "oracle").mkdir(parents=True)
    bundle_payloads = [
        json.loads(path.read_text())
        for path in sorted((ws / "ops_digest/fixtures/runs").glob("*.json"))
    ]
    digest = fixed_render_digest(bundle_payloads)
    oracle_files = {
        "ops_digest/src/schema.py": fixed_schema_py(),
        "ops_digest/src/digest_builder.py": fixed_digest_builder_py(),
        "ops_digest/automation/nightly_regression_watch.toml": automation_toml(True),
        "ops_digest/docs/escalation_runbook.md": runbook_md(True),
        "ops_digest/fixtures/inbox/generated_digest.md": digest,
    }
    for rel, content in oracle_files.items():
        write(vd / "oracle" / rel, content)
    write(vd / "hidden_tests/test_hidden_contract.py", hidden_test_py())
    readonly_hashes = {
        rel: sha256_tree(ws, rel)
        for rel in READONLY_TREES
        if sha256_tree(ws, rel) is not None
    }
    gold = {
        "variant_id": variant_id,
        "editable_files": EDITABLE_FILES,
        "code_files": ["ops_digest/src/schema.py", "ops_digest/src/digest_builder.py"],
        "generated_digest_path": "ops_digest/fixtures/inbox/generated_digest.md",
        "runbook_expectations": {
            "command": "python3 -m ops_digest.src.digest_builder --fixtures ops_digest/fixtures/runs --out ops_digest/fixtures/inbox/generated_digest.md",
        },
        "automation_expectations": {
            "summary_title": "Nightly Regression Watch",
            "schedule": "0 2 * * *",
            "destination": "inbox",
        },
        "readonly_tree_hashes": readonly_hashes,
    }
    write(vd / "gold_repair.json", json.dumps(gold, indent=2, sort_keys=True) + "\n")
    manifest = {"variant_id": variant_id, "files": list_manifest_files(ws)}
    write(vd / "workspace_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_shared_files() -> None:
    write(FAMILY / "task_spec.md", task_spec_md())
    write(FAMILY / "evaluator_contract.md", evaluator_contract_md())
    write(FAMILY / "family.yaml", family_yaml())
    write(VERIFIERS / "score_ranking.py", score_ranking_py())
    executable(VERIFIERS / "score_ranking.py")
    write(VERIFIER_DATA / "_shared/contract_checks.py", contract_checks_py())
    write(FAMILY / "tools/run_verification_matrix.py", matrix_script_py())
    executable(FAMILY / "tools/run_verification_matrix.py")
    write(VERIFIER_DATA / "_milestones_shared/README.md", milestone_readme())
    for name, content in milestone_scripts().items():
        path = VERIFIER_DATA / "_milestones_shared" / name
        write(path, content)
        executable(path)


def write_milestone_symlinks(variant_id: str) -> None:
    milestone_dir = VERIFIER_DATA / variant_id / "milestones"
    milestone_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "m1_localize.sh": "m1_localize.sh",
        "m2_primary_fix.sh": "m2_primary_fix.sh",
        "m3_invariants.sh": "m3_invariants.sh",
        "m4_functional.sh": "m4_functional.sh",
        "m5_e2e.sh": "m5_e2e.sh",
    }
    for link_name, target_name in mapping.items():
        link = milestone_dir / link_name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path("..") / ".." / "_milestones_shared" / target_name)


def apply_oracle_to_workspace(variant_id: str, workspace: Path) -> None:
    src = VERIFIER_DATA / variant_id / "oracle"
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)


def run_score(workspace: Path, variant_id: str) -> dict[str, Any]:
    result_file = workspace.parent / f"{variant_id}_score_result.json"
    env = os.environ.copy()
    env.update({"AGENT_WS": str(workspace), "VARIANT_ID": variant_id, "RESULT_FILE": str(result_file)})
    subprocess.run([sys.executable, str(VERIFIERS / "score_ranking.py")], env=env, check=True)
    return json.loads(result_file.read_text())


def score_baselines() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    oracle_scores: dict[str, int] = {}
    empty_scores: dict[str, int] = {}
    shortcut_scores: dict[str, int] = {}
    for variant_id in VARIANTS:
        with tempfile.TemporaryDirectory(prefix=f"nightly_{variant_id}_") as tmp:
            ws = Path(tmp) / "workspace"
            shutil.copytree(WORKSPACE_BUNDLE / variant_id, ws)
            empty_scores[variant_id] = run_score(ws, variant_id)["score"]
            apply_oracle_to_workspace(variant_id, ws)
            oracle_scores[variant_id] = run_score(ws, variant_id)["score"]
        with tempfile.TemporaryDirectory(prefix=f"nightly_shortcut_{variant_id}_") as tmp:
            ws = Path(tmp) / "workspace"
            shutil.copytree(WORKSPACE_BUNDLE / variant_id, ws)
            shutil.copy(VERIFIER_DATA / variant_id / "oracle" / "ops_digest/automation/nightly_regression_watch.toml", ws / "ops_digest/automation/nightly_regression_watch.toml")
            shutil.copy(VERIFIER_DATA / variant_id / "oracle" / "ops_digest/docs/escalation_runbook.md", ws / "ops_digest/docs/escalation_runbook.md")
            shortcut_scores[variant_id] = run_score(ws, variant_id)["score"]
    return oracle_scores, empty_scores, shortcut_scores


def write_manifest_lock(oracle_scores: dict[str, int], empty_scores: dict[str, int], shortcut_scores: dict[str, int]) -> None:
    lock = {
        "family_id": "nightly-regression-watch",
        "schema_version": "cnb55.manifest.v2",
        "grader": {
            "score_ranking_py_sha256": sha256_file(VERIFIERS / "score_ranking.py"),
        },
        "variants": {},
    }
    for variant_id in VARIANTS:
        ws = WORKSPACE_BUNDLE / variant_id
        vd = VERIFIER_DATA / variant_id
        lock["variants"][variant_id] = {
            "observed_oracle_score": oracle_scores[variant_id],
            "observed_empty_brief_score": empty_scores[variant_id],
            "observed_shortcut_score": shortcut_scores[variant_id],
            "workspace_trees": {
                rel: sha256_tree(ws, rel)
                for rel in READONLY_TREES + ["ops_digest/src/schema.py", "ops_digest/src/digest_builder.py", "ops_digest/automation/nightly_regression_watch.toml", "ops_digest/docs/escalation_runbook.md"]
                if sha256_tree(ws, rel) is not None
            },
            "verifier_data": {
                "gold_repair_sha256": sha256_file(vd / "gold_repair.json"),
                "workspace_manifest_sha256": sha256_file(vd / "workspace_manifest.json"),
                "hidden_tests_sha256": sha256_tree(vd, "hidden_tests"),
            },
        }
    write(FAMILY / "manifest.lock.json", json.dumps(lock, indent=2, sort_keys=True) + "\n")


def run_matrix(variant_id: str, out_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(FAMILY / "tools/run_verification_matrix.py"), "--variant", variant_id, "--out", str(out_path)],
        check=True,
        cwd=REPO,
    )


def main() -> int:
    WORKSPACE_BUNDLE.mkdir(parents=True, exist_ok=True)
    VERIFIERS.mkdir(parents=True, exist_ok=True)
    VERIFIER_DATA.mkdir(parents=True, exist_ok=True)

    write_shared_files()
    for variant_id, cfg in VARIANTS.items():
        render_workspace(variant_id, cfg)
        write_oracle_and_verifier_data(variant_id)
        write_milestone_symlinks(variant_id)

    oracle_scores, empty_scores, shortcut_scores = score_baselines()
    write_manifest_lock(oracle_scores, empty_scores, shortcut_scores)
    matrix_v1 = FAMILY / "verification_matrix.md"
    matrix_v5 = FAMILY / "verification_matrix_v5.md"
    run_matrix("v1-clean-baseline", matrix_v1)
    run_matrix("v5-recovery-in-thread", matrix_v5)
    write(FAMILY / "benchmark_run.md", benchmark_run_md(oracle_scores, empty_scores, shortcut_scores, matrix_v1.name, matrix_v5.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
