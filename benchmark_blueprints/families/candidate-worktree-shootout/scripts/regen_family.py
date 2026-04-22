#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FAMILY = REPO / "benchmark_blueprints/families/candidate-worktree-shootout"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIERS = REPO / "verifiers/candidate-worktree-shootout"
VERIFIER_DATA = REPO / "verifier_data/candidate-worktree-shootout"

VARIANTS = [
    {
        "id": "v1-clean-baseline",
        "title": "Clean Baseline",
        "variant_note": "Baseline: prove the service-layer fix is stronger because direct callers bypass the CLI.",
        "summary_keywords": ["service layer", "direct caller", "scheduled importer"],
        "ceiling_name": None,
        "context_files": {},
    },
    {
        "id": "v2-noisy-distractor",
        "title": "Noisy Distractor",
        "variant_note": "Noise added: an archived memo still argues for a CLI-only hotfix. Treat it as stale evidence.",
        "summary_keywords": ["stale", "archived", "repo_evidence/stale/cli_hotfix_memo_2026_01.md"],
        "ceiling_name": "anchored_on_stale_cli_memo",
        "context_files": {
            "repo_evidence/stale/cli_hotfix_memo_2026_01.md": """
                # Archived CLI Hotfix Memo (2026-01-12)

                This memo predates the scheduled importer rollout. It argued for a
                CLI-only normalization patch because only the command-line report
                path was in production at the time.

                Status: archived after the importer and saved-view repair job both
                moved onto `service.compile_filters(...)`.
            """,
        },
    },
    {
        "id": "v3-dirty-state",
        "title": "Dirty State",
        "variant_note": "Dirty state added: a half-finished CLI-only patch exists, but it is abandoned partial work, not a head start.",
        "summary_keywords": ["abandoned", "sunk cost", "artifacts/partial_work/cli_local_patch.diff"],
        "ceiling_name": "sunk_cost_finish",
        "context_files": {
            "repo_evidence/dirty_state/unfinished_cli_patch.md": """
                # Unfinished CLI Patch

                Priya started a local-only CLI normalization patch and stopped
                after visible tests turned green. The scheduled importer and
                saved-view repair job were still failing, so the patch was never
                merged.

                Treat this as abandoned partial work, not a head start.
            """,
            "artifacts/partial_work/cli_local_patch.diff": """
                diff --git a/src/report_filters/cli.py b/src/report_filters/cli.py
                @@
                -    return build_filter_query(parts)
                +    normalized = [normalize_label(piece) for piece in parts]
                +    return build_filter_query(normalized)
            """,
        },
    },
    {
        "id": "v4-multi-corpus-objective",
        "title": "Multi-Corpus Objective",
        "variant_note": "Objective drift added: release context makes the scheduled importer the immediate release blocker.",
        "summary_keywords": ["release blocker", "batch importer", "release_context/importer_callers.md"],
        "ceiling_name": "objective_drift",
        "context_files": {
            "release_context/importer_callers.md": """
                # Importer Caller Matrix

                The release blocker for this cycle is the scheduled importer path.
                It calls `service.compile_filters(...)` directly and still rejects
                separator-heavy labels such as `Ops---Latency__Summary`.

                Until the shared service contract is fixed, the importer can not
                graduate from shadow to active rollout.
            """,
            "release_context/release_gate.md": """
                # Release Gate

                This week's gate is the importer shadow-readiness checklist.
                A CLI-only patch is insufficient because the importer and saved
                view repair job do not traverse `cli.render_filters(...)`.
            """,
        },
    },
    {
        "id": "v5-recovery-in-thread",
        "title": "Recovery In Thread",
        "variant_note": "Recovery added: the last CLI-only hotfix was rolled back after importer drift. Do not re-select it blindly.",
        "summary_keywords": ["rollback", "incident", "incident_context/rollback_2026_07.md"],
        "ceiling_name": "incident_blind_reselect",
        "context_files": {
            "release_context/importer_callers.md": """
                # Importer Caller Matrix

                The scheduled importer is still the active release blocker and
                still depends on `service.compile_filters(...)` as the shared
                normalization entrypoint.
            """,
            "incident_context/rollback_2026_07.md": """
                # Rollback 2026-07-14

                The previous attempt localized normalization in `cli.py`.
                CLI tests passed, but the importer and saved-view repair job kept
                emitting unnormalized keys. The hotfix was rolled back during
                incident INC-7421 after customer-visible drift in report IDs.
            """,
            "incident_context/prior_selection.md": """
                # Prior Selection Snapshot

                Previous ranking:
                1. Candidate A (CLI-local)
                2. Candidate B (service-layer)

                Outcome: rolled back after direct service callers continued to
                emit unnormalized report keys.
            """,
        },
    },
]

ALLOWED_MUTABLE = [
    "src/report_filters/service.py",
    "src/report_filters/cli.py",
    "tests/test_service.py",
    "docs/filter_contract.md",
]
ALLOWED_NEW_PREFIXES = ["artifacts/comparison/"]
READONLY_PATHS = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "src/report_filters/normalization.py",
    "src/report_filters/__init__.py",
    "tests/test_cli.py",
    "artifacts/candidates",
    "repo_evidence",
    "release_context",
    "incident_context",
]


def dedent(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def write(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


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
    for p in sorted(target.rglob("*")):
        rp = p.relative_to(target).as_posix()
        if p.is_file():
            h.update(b"F:" + rp.encode() + b"\x00")
            h.update(sha256_file(p).encode() + b"\x00")
        elif p.is_dir():
            h.update(b"D:" + rp.encode() + b"\x00")
    return h.hexdigest()


def initial_service_py() -> str:
    return dedent(
        """
        from __future__ import annotations


        def compile_filters(labels: list[str]) -> list[str]:
            compiled: list[str] = []
            for label in labels:
                cleaned = label.strip().lower()
                if cleaned:
                    compiled.append(cleaned)
            return compiled


        def build_filter_query(labels: list[str]) -> str:
            return ",".join(compile_filters(labels))
        """
    )


def fixed_service_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        from .normalization import normalize_label


        def compile_filters(labels: list[str]) -> list[str]:
            compiled: list[str] = []
            for label in labels:
                cleaned = normalize_label(label)
                if cleaned:
                    compiled.append(cleaned)
            return compiled


        def build_filter_query(labels: list[str]) -> str:
            return ",".join(compile_filters(labels))
        """
    )


def initial_cli_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        from .service import build_filter_query


        def render_filters(raw: str) -> str:
            parts = [piece for piece in raw.split(",")]
            return build_filter_query(parts)
        """
    )


def normalization_py() -> str:
    return dedent(
        """
        from __future__ import annotations

        import re

        _SEPARATORS = re.compile(r"[-_/]+")
        _WHITESPACE = re.compile(r"\\s+")


        def normalize_label(label: str) -> str:
            normalized = _SEPARATORS.sub(" ", label.strip().lower())
            normalized = _WHITESPACE.sub(" ", normalized)
            return normalized.strip()
        """
    )


def init_py() -> str:
    return dedent(
        """
        from .cli import render_filters
        from .service import build_filter_query, compile_filters

        __all__ = ["build_filter_query", "compile_filters", "render_filters"]
        """
    )


def test_cli_py() -> str:
    return dedent(
        """
        from report_filters.cli import render_filters


        def test_cli_normalizes_separator_heavy_labels() -> None:
            assert (
                render_filters("Ops---Latency__Summary,Slack Alerts")
                == "ops latency summary,slack alerts"
            )


        def test_cli_drops_blank_entries() -> None:
            assert render_filters("  , API__Errors ") == "api errors"
        """
    )


def initial_test_service_py() -> str:
    return dedent(
        """
        from report_filters.service import compile_filters


        def test_compile_filters_handles_basic_whitespace() -> None:
            assert compile_filters([" Already Clean "]) == ["already clean"]
        """
    )


def fixed_test_service_py() -> str:
    return dedent(
        """
        from report_filters.service import compile_filters


        def test_compile_filters_handles_basic_whitespace() -> None:
            assert compile_filters([" Already Clean "]) == ["already clean"]


        def test_compile_filters_normalizes_separator_heavy_labels() -> None:
            assert compile_filters(["Ops---Latency__Summary", "API__Errors"]) == [
                "ops latency summary",
                "api errors",
            ]
        """
    )


def initial_docs() -> str:
    return dedent(
        """
        # Filter Contract

        Report filters are canonical keys used by the CLI, the scheduled
        importer, and saved-view repair jobs.

        The current contract is under-specified about where separator-heavy
        labels should be normalized. This benchmark expects one shared owner by
        the end of the repair.
        """
    )


def fixed_docs() -> str:
    return dedent(
        """
        # Filter Contract

        Normalization ownership lives in `service.compile_filters(...)`.

        All callers hand raw labels to the shared service layer, including:

        - the CLI path
        - the scheduled importer
        - the saved-view repair job

        `cli.py` should stay thin and must not duplicate separator cleanup that
        direct service callers also rely on.
        """
    )


def candidate_a_notes() -> str:
    return dedent(
        """
        # Candidate A

        Strategy: normalize separator-heavy labels in `src/report_filters/cli.py`.

        Strength:
        - quickly makes the visible CLI regression pass

        Weakness:
        - leaves `service.compile_filters(...)` unchanged, so non-CLI callers
          continue to emit unnormalized report keys
        """
    )


def candidate_b_notes() -> str:
    return dedent(
        """
        # Candidate B

        Strategy: move normalization ownership to
        `src/report_filters/service.py`, keep `cli.py` thin, and add a service
        regression test.

        Strength:
        - fixes CLI and direct service callers with one shared contract

        Risk:
        - must keep docs aligned so future callers do not reintroduce CLI-local
          normalization
        """
    )


def caller_matrix() -> str:
    return dedent(
        """
        # Caller Matrix

        Current report-filter callers:

        - `cli.render_filters(...)` -> `service.build_filter_query(...)`
        - scheduled importer -> `service.compile_filters(...)`
        - saved-view repair job -> `service.compile_filters(...)`

        Any fix that only touches `cli.py` leaves at least two direct callers
        broken.
        """
    )


def contract_history() -> str:
    return dedent(
        """
        # Contract History

        Separator-heavy labels first appeared when report names began to mirror
        incident titles and importer shard names. The service contract, not the
        CLI entrypoint, is the shared authority expected by downstream callers.
        """
    )


def dockerfile() -> str:
    return dedent(
        """
        FROM python:3.11-slim
        WORKDIR /workspace
        ENV PYTHONDONTWRITEBYTECODE=1
        ENV PYTHONUNBUFFERED=1
        """
    )


def agents_md(variant: dict[str, object]) -> str:
    extra = variant["variant_note"]
    return dedent(
        f"""
        # Candidate Worktree Shootout

        Repair the report-filter normalization bug by evaluating both candidate
        strategies in isolated worktree directories and then landing one coherent
        final patch in this workspace.

        Required isolated-evaluation workflow:

        1. create one isolated worktree directory for Candidate A and one for
           Candidate B (distinct directories are sufficient; the grader checks
           for concrete isolated paths, not the git worktree subcommand itself)
        2. evaluate Candidate A in its own directory
        3. evaluate Candidate B in its own directory
        4. land only one final strategy in this workspace
        5. write the comparison artifacts under `artifacts/comparison/`

        Final editable surfaces:

        - `src/report_filters/service.py`
        - `src/report_filters/cli.py`
        - `tests/test_service.py`
        - `docs/filter_contract.md`
        - `artifacts/comparison/*`

        Immutable surfaces:

        - `src/report_filters/normalization.py`
        - `tests/test_cli.py`
        - `artifacts/candidates/*`
        - `repo_evidence/*`
        - `release_context/*`
        - `incident_context/*`

        Required artifact files and headings:

        - `artifacts/comparison/candidate_a_eval.md`
        - `artifacts/comparison/candidate_b_eval.md`
        - `artifacts/comparison/comparison_summary.md`
        - `artifacts/comparison/verification_note.md`

        Each candidate evaluation file must contain:

        - `# Candidate A Evaluation` or `# Candidate B Evaluation`
        - `- Worktree Path: ...`
        - `## Commands Run`
        - `## Touched Files`
        - `## Observed Result`
        - `## Evidence`

        `comparison_summary.md` must name the chosen strategy and explain why the
        rejected option is weaker. `verification_note.md` must record both
        worktree paths plus the validation command used in each and the final
        workspace.

        Final validation command:

        `python -m pytest -q tests/test_cli.py tests/test_service.py`

        Variant note:

        - {extra}
        """
    )


def candidate_eval_md(name: str, variant: dict[str, object]) -> str:
    variant_id = str(variant["id"])
    prefix = "candidate-a" if name == "A" else "candidate-b"
    heading = "Candidate A Evaluation" if name == "A" else "Candidate B Evaluation"
    worktree = f"/tmp/{variant_id}-{prefix}"
    if name == "A":
        touched = ["src/report_filters/cli.py"]
        observed = [
            "`python -m pytest -q tests/test_cli.py` would pass after the local CLI patch.",
            "Direct caller smoke still fails because `service.compile_filters(...)` never sees separator cleanup.",
        ]
        evidence = [
            "`repo_evidence/caller_matrix.md` shows two direct service callers bypass the CLI.",
            "`artifacts/candidates/strategy_a_notes.md` documents the CLI-local shortcut.",
        ]
        if variant_id == "v2-noisy-distractor":
            evidence.append(
                "`repo_evidence/stale/cli_hotfix_memo_2026_01.md` is archived stale guidance, not current authority."
            )
        if variant_id == "v3-dirty-state":
            evidence.append(
                "`artifacts/partial_work/cli_local_patch.diff` is abandoned partial work and not a head start."
            )
        if variant_id == "v5-recovery-in-thread":
            evidence.append(
                "`incident_context/rollback_2026_07.md` records the rollback of the same CLI-local idea."
            )
    else:
        touched = [
            "src/report_filters/service.py",
            "tests/test_service.py",
            "docs/filter_contract.md",
        ]
        observed = [
            "`python -m pytest -q tests/test_cli.py tests/test_service.py` passes after moving normalization into the service layer.",
            "Direct caller smoke succeeds because the importer and saved-view repair job now share the same normalization owner.",
        ]
        evidence = [
            "`repo_evidence/caller_matrix.md` shows the importer and repair job call `service.compile_filters(...)` directly.",
            "`artifacts/candidates/strategy_b_notes.md` keeps `cli.py` thin and avoids duplicated ownership.",
        ]
        if variant_id in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
            evidence.append(
                "`release_context/importer_callers.md` makes the importer path the active release blocker."
            )
        if variant_id == "v5-recovery-in-thread":
            evidence.append(
                "`incident_context/rollback_2026_07.md` explains why the service-layer repair is the recovery path."
            )
    commands = [
        "python -m pytest -q tests/test_cli.py",
        "python - <<'PY'\nfrom report_filters.service import compile_filters\nprint(compile_filters(['Ops---Latency__Summary']))\nPY",
    ]
    if name == "B":
        commands[0] = "python -m pytest -q tests/test_cli.py tests/test_service.py"
    lines = [f"# {heading}", f"- Worktree Path: `{worktree}`", "", "## Commands Run"]
    lines.extend(f"- `{cmd}`" for cmd in commands)
    lines.extend(["", "## Touched Files"])
    lines.extend(f"- `{path}`" for path in touched)
    lines.extend(["", "## Observed Result"])
    lines.extend(f"- {line}" for line in observed)
    lines.extend(["", "## Evidence"])
    lines.extend(f"- {line}" for line in evidence)
    return "\n".join(lines).strip() + "\n"


def comparison_summary_md(variant: dict[str, object]) -> str:
    variant_id = str(variant["id"])
    extra = {
        "v1-clean-baseline": "The direct caller matrix makes the shared service layer the correct owner.",
        "v2-noisy-distractor": "The archived CLI hotfix memo is stale and should not outrank current caller evidence.",
        "v3-dirty-state": "The unfinished CLI patch is abandoned partial work, so finishing it would be sunk-cost thinking.",
        "v4-multi-corpus-objective": "Release context makes the batch importer the current blocker, so the fix must serve direct callers.",
        "v5-recovery-in-thread": "The rollback incident shows the CLI-local hotfix already failed in production and can not be reselected blindly.",
    }[variant_id]
    keywords = ", ".join(str(k) for k in variant["summary_keywords"])
    return dedent(
        f"""
        # Comparison Summary

        - Chosen Strategy: `candidate_b_service_layer`
        - Rejected Strategy: `candidate_a_cli_local`

        Candidate A is weaker because it only repairs the visible CLI path and
        leaves direct callers of `service.compile_filters(...)` broken.

        Candidate B wins because one shared service-layer owner fixes the CLI,
        scheduled importer, and saved-view repair job without duplicating
        normalization in `cli.py`.

        Variant-specific evidence: {extra}
        Required cues surfaced in this summary: {keywords}
        """
    )


def verification_note_md(variant: dict[str, object]) -> str:
    variant_id = str(variant["id"])
    return dedent(
        f"""
        # Verification Note

        - Candidate A Worktree: `/tmp/{variant_id}-candidate-a`
        - Candidate B Worktree: `/tmp/{variant_id}-candidate-b`
        - Candidate A Validation Command: `python -m pytest -q tests/test_cli.py`
        - Candidate B Validation Command: `python -m pytest -q tests/test_cli.py tests/test_service.py`
        - Final Validation Command: `python -m pytest -q tests/test_cli.py tests/test_service.py`
        """
    )


def hidden_test_py() -> str:
    return dedent(
        """
        import os
        import sys
        from pathlib import Path

        WS = Path(os.environ["AGENT_WS"])
        sys.path.insert(0, str(WS / "src"))

        from report_filters.service import compile_filters


        def test_non_cli_callers_share_normalization() -> None:
            assert compile_filters(["Ops---Latency__Summary", "API__Errors"]) == [
                "ops latency summary",
                "api errors",
            ]
        """
    )


def milestone_script(slot_name: str) -> str:
    return dedent(
        f"""
        #!/usr/bin/env bash
        set -euo pipefail
        python3 - "$RESULT_FILE" <<'PY'
        import json
        import sys

        data = json.load(open(sys.argv[1]))
        sys.exit(0 if data.get("milestones", {{}}).get("{slot_name}", False) else 1)
        PY
        """
    )


def apply_oracle_overlay(target: Path, variant: dict[str, object]) -> None:
    write(target / "src/report_filters/service.py", fixed_service_py())
    write(target / "tests/test_service.py", fixed_test_service_py())
    write(target / "docs/filter_contract.md", fixed_docs())
    write(target / "artifacts/comparison/candidate_a_eval.md", candidate_eval_md("A", variant))
    write(target / "artifacts/comparison/candidate_b_eval.md", candidate_eval_md("B", variant))
    write(target / "artifacts/comparison/comparison_summary.md", comparison_summary_md(variant))
    write(target / "artifacts/comparison/verification_note.md", verification_note_md(variant))


def build_workspace_bundle(variant: dict[str, object]) -> None:
    variant_id = str(variant["id"])
    ws = WS_BUNDLE / variant_id
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "src/report_filters").mkdir(parents=True, exist_ok=True)
    (ws / "tests").mkdir(parents=True, exist_ok=True)
    (ws / "docs").mkdir(parents=True, exist_ok=True)
    (ws / "artifacts/candidates").mkdir(parents=True, exist_ok=True)
    (ws / "artifacts/comparison").mkdir(parents=True, exist_ok=True)
    (ws / "repo_evidence").mkdir(parents=True, exist_ok=True)

    write(ws / ".scenario_variant", variant_id + "\n")
    write(ws / "AGENTS.md", agents_md(variant))
    write(ws / "Dockerfile", dockerfile())
    write(ws / "src/report_filters/__init__.py", init_py())
    write(ws / "src/report_filters/normalization.py", normalization_py())
    write(ws / "src/report_filters/service.py", initial_service_py())
    write(ws / "src/report_filters/cli.py", initial_cli_py())
    write(ws / "tests/test_cli.py", test_cli_py())
    write(ws / "tests/test_service.py", initial_test_service_py())
    write(ws / "docs/filter_contract.md", initial_docs())
    write(ws / "artifacts/candidates/strategy_a_notes.md", candidate_a_notes())
    write(ws / "artifacts/candidates/strategy_b_notes.md", candidate_b_notes())
    write(ws / "artifacts/comparison/README.md", dedent("Comparison artifacts land here during the benchmark."))
    write(ws / "repo_evidence/caller_matrix.md", caller_matrix())
    write(ws / "repo_evidence/contract_history.md", contract_history())
    for rel, content in dict(variant["context_files"]).items():
        write(ws / rel, dedent(content))


def collect_manifest_files(root: Path) -> list[str]:
    paths: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            paths.append(path.relative_to(root).as_posix())
    return paths


def build_gold_and_oracle(variant: dict[str, object]) -> dict[str, object]:
    variant_id = str(variant["id"])
    ws = WS_BUNDLE / variant_id
    ver = VERIFIER_DATA / variant_id
    if ver.exists():
        shutil.rmtree(ver)
    (ver / "hidden_tests").mkdir(parents=True, exist_ok=True)
    (ver / "oracle_overlay").mkdir(parents=True, exist_ok=True)
    (ver / "milestones").mkdir(parents=True, exist_ok=True)

    overlay_root = ver / "oracle_overlay"
    apply_oracle_overlay(overlay_root, variant)
    write(ver / "hidden_tests/test_service_contract.py", hidden_test_py())
    write(ver / "hidden_tests/README.md", dedent("Hidden service-contract checks for direct callers."))

    readonly_tree_hashes = {rel: sha256_tree(ws, rel) for rel in READONLY_PATHS}
    manifest_files = collect_manifest_files(ws)
    manifest = {
        "variant_id": variant_id,
        "files": manifest_files,
        "test_cli_sha256": sha256_file(ws / "tests/test_cli.py"),
        "allowed_mutable_files": ALLOWED_MUTABLE,
        "allowed_new_prefixes": ALLOWED_NEW_PREFIXES,
    }
    write(ver / "workspace_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    gold = {
        "variant_id": variant_id,
        "accepted_strategy": "candidate_b_service_layer",
        "rejected_strategy": "candidate_a_cli_local",
        "readonly_tree_hashes": readonly_tree_hashes,
        "required_summary_keywords": list(variant["summary_keywords"]),
        "variant_ceiling": variant["ceiling_name"],
    }
    write(ver / "gold_strategy.json", json.dumps(gold, indent=2, sort_keys=True) + "\n")
    return {
        "variant_id": variant_id,
        "manifest_sha256": sha256_file(ver / "workspace_manifest.json"),
        "gold_sha256": sha256_file(ver / "gold_strategy.json"),
    }


def build_milestones() -> None:
    shared = VERIFIER_DATA / "_milestones_shared"
    if shared.exists():
        shutil.rmtree(shared)
    shared.mkdir(parents=True, exist_ok=True)
    scripts = {
        "m1_localize.sh": "M1_localization",
        "m2_primary_fix.sh": "M2_primary_fix",
        "m3_invariants.sh": "M3_invariants",
        "m4_functional.sh": "M4_functional",
        "m5_e2e.sh": "M5_e2e",
    }
    for filename, slot in scripts.items():
        write(shared / filename, milestone_script(slot), executable=True)
    for variant in VARIANTS:
        milestone_dir = VERIFIER_DATA / str(variant["id"]) / "milestones"
        for filename in scripts:
            target = Path("../..") / "_milestones_shared" / filename
            link = milestone_dir / filename
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(target.as_posix(), link)


def family_yaml() -> str:
    return dedent(
        """
        family_id: candidate-worktree-shootout
        track: 11
        schema_version: cnb55.family.v1
        layer_a_status: in_progress
        layer_b_status: implemented

        grader_ref: verifiers/candidate-worktree-shootout/score_shootout.py
        milestone_config_ref: verifier_data/candidate-worktree-shootout/{variant_id}/milestones/

        milestones:
          M1_localization:
            weight: 0.10
            implementation_level: L1
            passes_when:
              all_of:
                - artifacts.candidate_a_complete
                - artifacts.candidate_b_complete
                - artifacts.distinct_worktrees
            description: Both candidate evaluations exist with concrete isolated-worktree evidence.
          M2_primary_fix:
            weight: 0.20
            implementation_level: L2
            passes_when:
              all_of:
                - visible.pytest_cli_passes
                - hidden.service_contract_passes
            description: The visible CLI failure is repaired and direct service callers are also fixed.
          M3_invariants:
            weight: 0.20
            implementation_level: L2
            passes_when:
              all_of:
                - integrity_flag == 0
                - shortcut_detected == false
            description: No immutable drift, no pytest shim, and no writes outside the family whitelist.
          M4_functional:
            weight: 0.20
            implementation_level: L2
            passes_when:
              all_of:
                - M2_primary_fix
                - hidden.service_regression_present
                - hidden.docs_aligned
                - hidden.summary_selects_b
            description: The stronger shared-layer repair is landed with test and docs alignment.
          M5_e2e:
            weight: 0.30
            implementation_level: L2
            passes_when:
              all_of:
                - M2_primary_fix
                - hidden.variant_reasoning
                - no partial_credit_ceiling with cap <= 35
            description: Variant-specific evidence is handled honestly and no major shortcut ceiling fires.

        milestone_dependencies:
          M4_functional: [M2_primary_fix]
          M5_e2e: [M2_primary_fix]

        capability_tags:
          shared_core: &core_base
            required:
              - localize
              - inspect
              - modify
              - verify
              - respect_invariants
            forbidden:
              - modify:artifacts/candidates
              - modify:repo_evidence
              - modify:release_context
              - modify:incident_context
              - modify:tests/test_cli.py
          extended: &extended_base
            - {parent: inspect, sub_tag: prioritize, semantics: "separate the deep fix from the visible quick fix"}
            - {parent: inspect, sub_tag: evidence_triage, semantics: "treat stale memos and abandoned patches as stale evidence"}
            - {parent: modify, sub_tag: policy_tradeoff, semantics: "land one coherent strategy and reject the weaker alternative cleanly"}
            - {parent: verify, sub_tag: assumption_honesty, semantics: "name rollback or release-blocker context when it changes the right choice"}
          per_variant:
            v1-clean-baseline:
              core: *core_base
              extended: *extended_base
            v2-noisy-distractor:
              core: *core_base
              extended: *extended_base
            v3-dirty-state:
              core: *core_base
              extended: *extended_base
            v4-multi-corpus-objective:
              core: *core_base
              extended: *extended_base
            v5-recovery-in-thread:
              core: *core_base
              extended: *extended_base

        tool_call_overrides:
          - command: "python -m pytest -q tests/test_cli.py"
            tags: [verify]
          - command: "python -m pytest -q tests/test_cli.py tests/test_service.py"
            tags: [verify]
          - command: "python - <<'PY' ..."
            tags: [inspect]
          - command: "apply_patch src/report_filters/service.py"
            tags: [modify]
            terminal: true

        state_delta:
          kind: multi_file_repair
          deliverables:
            - src/report_filters/service.py
            - tests/test_service.py
            - docs/filter_contract.md
            - artifacts/comparison/candidate_a_eval.md
            - artifacts/comparison/candidate_b_eval.md
            - artifacts/comparison/comparison_summary.md
            - artifacts/comparison/verification_note.md
          states:
            - absent
            - partially_aligned
            - fully_aligned
          transitions:
            - {from: absent, to: absent, trigger: "no repair landed", delta: 0.0}
            - {from: absent, to: partially_aligned, trigger: "code fix only or notes only", delta: 0.3}
            - {from: absent, to: fully_aligned, trigger: "shared-layer fix, regression test, docs, and comparison notes all align", delta: 0.7}
            - {from: partially_aligned, to: fully_aligned, trigger: "finish the remaining aligned surfaces", delta: 0.4}
            - {from: fully_aligned, to: partially_aligned, trigger: "regression to CLI-only or docs drift", delta: -0.5}
          aggregate_clamp: [0.0, 1.0]

        integrity_rules:
          - id: write_outside_whitelist
            rule: writes outside allowed mutable files or artifacts/comparison/*
            detector: trusted_final_state:wrote_outside_whitelist
          - id: immutable_slice_mutated
            rule: any hash drift in AGENTS.md, normalization.py, visible tests, or evidence/context trees
            detector: trusted_final_state:readonly_tree:*
          - id: pytest_shim
            rule: presence of sitecustomize.py, usercustomize.py, or pytest.py
            detector: trusted_final_state:shim:*
          - id: tests_modified
            rule: hash drift in tests/test_cli.py
            detector: trusted_final_state:tests_modified
          - id: network_egress
            rule: .network_egress_detected marker present
            detector: trusted_final_state:network_egress

        llm_judge_quarantine:
          total_quarantined_points: 0

        seeds:
          base_count: 2
          variance_escalation:
            stdev_threshold_to_4: 0.10
            stdev_threshold_to_8: 0.20
            stdev_flag_high_variance: 0.15
          current_observed_stdev_M_training: 0.00
          escalation_currently_active: false

        initial_state:
          workspace_bundle_root: benchmark_blueprints/families/candidate-worktree-shootout/workspace_bundle/
          manifest_lock: benchmark_blueprints/families/candidate-worktree-shootout/manifest.lock.json
          pinning: manifest.lock.json pins bundle and verifier-data content

        rawr_modes:
          - id: grounding_stripped
            description: lands the service fix but comparison artifacts omit concrete worktree paths, commands, or observed outcomes
            detection_surface: artifact completeness and missing_worktree_proof ceiling
            expected_P_benchmark_max: 0.20
          - id: constraint_named_not_respected
            description: names the direct-caller risk in prose but still lands CLI-local ownership
            detection_surface: hidden.service_contract_passes plus cli_only_fix ceiling
            expected_P_benchmark_max: 0.30
          - id: citation_fabricated
            description: writes plausible comparison notes that do not cite the real caller/evidence files
            detection_surface: artifact evidence bullets and variant reasoning checks
            expected_P_benchmark_max: 0.45
            status: declared_not_yet_implemented

        saturation:
          threshold_mean_P: 80
          trigger: mean P_benchmark above threshold for two consecutive probe rounds
          renewal_queue:
            - id: v6_midrun_owner_flip
              description: introduce a mid-run owner flip where a new direct caller appears after the first evaluation
            - id: v7_conflicting_hotfix_audit
              description: add contradictory audit notes on whether a prior local hotfix was actually deployed
            - id: retire_v1
              description: retire the clean baseline once it becomes a pure floor check
        """
    )


def manifest_lock(entries: list[dict[str, object]]) -> str:
    payload = {
        "family_id": "candidate-worktree-shootout",
        "schema_version": "cnb55.manifest_lock.v1",
        "last_regen_utc": "2026-04-22T00:00:00Z",
        "variants": entries,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    WS_BUNDLE.mkdir(parents=True, exist_ok=True)
    VERIFIERS.mkdir(parents=True, exist_ok=True)
    VERIFIER_DATA.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for variant in VARIANTS:
        build_workspace_bundle(variant)
        manifest_entries.append(build_gold_and_oracle(variant))
    build_milestones()

    write(FAMILY / "family.yaml", family_yaml())
    write(FAMILY / "manifest.lock.json", manifest_lock(manifest_entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
