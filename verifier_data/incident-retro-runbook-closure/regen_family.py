#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import stat
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILY = REPO / "benchmark_blueprints/families/incident-retro-runbook-closure"
VERIFIER = REPO / "verifier_data/incident-retro-runbook-closure"

VARIANTS = {
    "v1-clean-baseline": {
        "title": "Clean baseline",
        "extra_agents": "Read retro/action_items.json first. The workspace is otherwise clean.",
        "retro_notes": "",
        "timeline_lines": "",
        "release_context": None,
        "incident_context": None,
    },
    "v2-noisy-distractor": {
        "title": "Noisy distractor",
        "extra_agents": "Supporting prose repeats the retired command. Do not let it override action_items.json.",
        "retro_notes": "\nSupporting note: the January playbook still calls the retired `queue-drain audit-post-drain` command and points operators at #queue-drain-war-room.\n",
        "timeline_lines": "\n09:29 operator pasted `queue-drain audit-post-drain --cluster atlas-a --include-stuck-shards` from an old paste buffer.\n",
        "release_context": None,
        "incident_context": None,
    },
    "v3-dirty-state": {
        "title": "Dirty state",
        "extra_agents": "The follow-up note and helper comments are half-updated. Do not preserve both command names as acceptable operator paths.",
        "retro_notes": "\nSupporting note: a prior cleanup draft said to keep both command names in the runbook until the next retro.\n",
        "timeline_lines": "\n09:37 cleanup draft suggested leaving both the retired and current commands in the operator doc.\n",
        "release_context": None,
        "incident_context": None,
    },
    "v4-multi-corpus-objective": {
        "title": "Multi-corpus objective drift",
        "extra_agents": "release_context/ emphasizes fast closure. It is supporting context only; post-drain verification still comes from action_items.json.",
        "retro_notes": "\nSupporting note: an incident commander wanted faster closure wording after queue depth hit zero.\n",
        "timeline_lines": "\n09:42 incident commander asked whether closure could happen once queue depth reached zero, before post-drain verification finished.\n",
        "release_context": {
            "README.md": "# Release context\n\nThese notes are supporting context only.\n",
            "q2_page_closure_guidance.md": "Page-closure guidance: restore service quickly after queue depth reaches zero, but do not overrule the authoritative retro action items.\n",
        },
        "incident_context": None,
    },
    "v5-recovery-in-thread": {
        "title": "Recovery in thread",
        "extra_agents": "incident_context/ shows a prior bad fix changed cadence and destination. Preserve both and repair only the prompt content.",
        "retro_notes": "\nSupporting note: a previous wake-up change touched cadence instead of prompt content and had to be rolled back.\n",
        "timeline_lines": "\n09:51 rollback note: the thread wake cadence was changed to 60 minutes in a prior attempt and the follow-up sequence was missed.\n",
        "release_context": {
            "README.md": "# Release context\n\nThese notes are supporting context only.\n",
            "q2_page_closure_guidance.md": "Fast closure matters, but closure is still blocked on the authoritative post-drain verification sequence.\n",
        },
        "incident_context": {
            "README.md": "# Incident context\n\nThis directory describes a rolled-back automation fix.\n",
            "rollback_2026_02_18.md": "A prior fix changed queue-drain-watch cadence from 30 minutes to 60 and destination from thread to inbox. That fix was rolled back because it preserved the stale prompt text and skipped the required verification sequence.\n",
        },
    },
}

ACTION_ITEMS = {
    "family_id": "incident-retro-runbook-closure",
    "verification_command": "queue-drain verify-post-drain",
    "retired_command": "queue-drain audit-post-drain",
    "verification_sequence": [
        "queue-drain quiesce --cluster atlas-a",
        "queue-drain verify-post-drain --cluster atlas-a --include-stuck-shards",
        "python3 repo/scripts/queue_drain_helper.py --emit-shard-report atlas-a",
        "Escalate to #storage-sre with the shard report if any shard remains stuck",
    ],
    "escalation_target": "#storage-sre",
    "retired_escalation_target": "#queue-drain-war-room",
    "informational_note_tokens": [
        "mirror-screenshot-only",
        "timeline-typo-only",
    ],
}

AUTOMATION_EXPECTATIONS = {
    "kind": "heartbeat",
    "destination": "thread",
    "schedule_minutes": 30,
    "expected_prompt_command": ACTION_ITEMS["verification_command"],
    "expected_prompt_target": ACTION_ITEMS["escalation_target"],
}

OPS_NOTE_SCHEMA = {
    "required_lines": [
        "## Applied repair",
        "- verification_command:",
        "- escalation_target:",
        "## Deferred informational retro notes",
    ]
}


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def json_dump(path: Path, payload: object) -> None:
    write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def helper_py(fixed: bool) -> str:
    current = ACTION_ITEMS["verification_command"] if fixed else ACTION_ITEMS["retired_command"]
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        from __future__ import annotations

        import argparse

        CURRENT_VERIFY_COMMAND = "{current}"
        RETIRED_VERIFY_COMMAND = "{ACTION_ITEMS["retired_command"]}"


        def build_verification_command(cluster: str) -> str:
            return f"{{CURRENT_VERIFY_COMMAND}} --cluster {{cluster}} --include-stuck-shards"


        def emit_shard_report(cluster: str) -> str:
            return f"python3 repo/scripts/queue_drain_helper.py --emit-shard-report {{cluster}}"


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--emit-verification", action="store_true")
            parser.add_argument("--emit-shard-report")
            parser.add_argument("--cluster", default="atlas-a")
            args = parser.parse_args()
            if args.emit_verification:
                print(build_verification_command(args.cluster))
                return 0
            if args.emit_shard_report:
                print(emit_shard_report(args.emit_shard_report))
                return 0
            parser.print_help()
            return 1


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def runbook_md(fixed: bool, variant_id: str) -> str:
    command = ACTION_ITEMS["verification_command"] if fixed else ACTION_ITEMS["retired_command"]
    target = ACTION_ITEMS["escalation_target"] if fixed else ACTION_ITEMS["retired_escalation_target"]
    sequence = [
        ACTION_ITEMS["verification_sequence"][0],
        f"{command} --cluster atlas-a --include-stuck-shards",
        ACTION_ITEMS["verification_sequence"][2],
        f"Escalate to {target} with the shard report if any shard remains stuck",
    ]
    dual_path = ""
    if not fixed and variant_id == "v3-dirty-state":
        dual_path = "\nLegacy note: until cleanup lands, operators may also reference `queue-drain verify-post-drain`.\n"
    return textwrap.dedent(
        f"""\
        # Queue Drain Closure

        Use this runbook after the queue depth returns to zero.

        ## Required closure sequence

        1. `{sequence[0]}`
        2. `{sequence[1]}`
        3. `{sequence[2]}`
        4. `{sequence[3]}`
        {dual_path}
        ## Escalation

        Escalate unresolved stuck shards to `{target}` with the shard report.
        """
    )


def automation_toml(fixed: bool) -> str:
    command = ACTION_ITEMS["verification_command"] if fixed else ACTION_ITEMS["retired_command"]
    target = ACTION_ITEMS["escalation_target"] if fixed else ACTION_ITEMS["retired_escalation_target"]
    return textwrap.dedent(
        f'''\
        name = "queue-drain-watch"
        kind = "heartbeat"
        destination = "thread"
        schedule_minutes = 30
        prompt = """
        Wake up in this thread and verify queue-drain closure.
        1. Run `queue-drain quiesce --cluster atlas-a`
        2. Run `{command} --cluster atlas-a --include-stuck-shards`
        3. Escalate unresolved stuck shards to `{target}`
        """
        '''
    )


def followup_md(fixed: bool, variant_id: str) -> str:
    if fixed:
        body = textwrap.dedent(
            f"""\
            ## Applied repair
            - verification_command: {ACTION_ITEMS["verification_command"]}
            - escalation_target: {ACTION_ITEMS["escalation_target"]}
            - repaired_surfaces: runbook, helper, automation_prompt, followup_note

            ## Deferred informational retro notes
            - mirror-screenshot-only
            - timeline-typo-only
            """
        )
    else:
        extra = ""
        if variant_id == "v3-dirty-state":
            extra = "\n- operators may reference either queue-drain audit-post-drain or queue-drain verify-post-drain during transition\n"
        body = textwrap.dedent(
            f"""\
            ## Applied repair
            - verification_command: {ACTION_ITEMS["retired_command"]}
            - escalation_target: {ACTION_ITEMS["retired_escalation_target"]}
            - mirror-screenshot-only
            - timeline-typo-only
            {extra}
            ## Deferred informational retro notes
            - none
            """
        )
    return body


def agents_md(variant_id: str, cfg: dict[str, object]) -> str:
    return textwrap.dedent(
        f"""\
        # AGENTS

        You are repairing the queue-drain closure workflow for `{variant_id}`.

        Hard rules:
        - Read `retro/action_items.json` first and treat it as authoritative.
        - Edit only these files:
          - `repo/runbooks/queue_drain.md`
          - `repo/scripts/queue_drain_helper.py`
          - `repo/.codex/automations/queue-drain-watch/automation.toml`
          - `repo/ops/notes/queue_drain_followup.md`
        - Do not modify tests, retro artifacts, release_context, incident_context, or expectation files.
        - Preserve the automation schedule and destination.
        - Run `pytest -q repo/tests/test_queue_drain_helper.py repo/tests/test_runbook_contract.py` before finishing.

        Variant note:
        {cfg["extra_agents"]}
        """
    )


def incident_retro_md(cfg: dict[str, object]) -> str:
    return textwrap.dedent(
        f"""\
        # Incident retrospective

        The queue drained, but the closure workflow drifted from the intended repair.
        The action items below were copied into `retro/action_items.json` and are authoritative.
        {cfg["retro_notes"]}
        Informational notes:
        - mirror-screenshot-only
        - timeline-typo-only
        """
    )


def operator_timeline(cfg: dict[str, object]) -> str:
    return textwrap.dedent(
        f"""\
        09:10 queue depth hit zero on atlas-a
        09:16 on-call copied the previous helper output into the incident thread
        09:21 stuck shard persisted after closure draft
        {cfg["timeline_lines"]}
        """
    )


def dockerfile() -> str:
    return "FROM python:3.12-bookworm\nWORKDIR /workspace\n"


def helper_tests() -> str:
    return textwrap.dedent(
        """\
        from __future__ import annotations

        import importlib.util
        from pathlib import Path


        ROOT = Path(__file__).resolve().parents[2]
        ACTION_ITEMS = ROOT / "retro" / "action_items.json"
        HELPER = ROOT / "repo" / "scripts" / "queue_drain_helper.py"


        def load_helper():
            spec = importlib.util.spec_from_file_location("queue_drain_helper", HELPER)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            return module


        def test_helper_emits_authoritative_verification_command():
            import json
            expected = json.loads(ACTION_ITEMS.read_text())["verification_command"]
            helper = load_helper()
            actual = helper.build_verification_command("atlas-a")
            assert actual.startswith(expected)


        def test_helper_no_longer_uses_retired_primary_command():
            import json
            payload = json.loads(ACTION_ITEMS.read_text())
            helper = load_helper()
            actual = helper.build_verification_command("atlas-a")
            assert payload["retired_command"] not in actual
        """
    )


def runbook_tests() -> str:
    return textwrap.dedent(
        """\
        from __future__ import annotations

        import json
        from pathlib import Path


        ROOT = Path(__file__).resolve().parents[2]
        ACTION_ITEMS = json.loads((ROOT / "retro" / "action_items.json").read_text())
        RUNBOOK = (ROOT / "repo" / "runbooks" / "queue_drain.md").read_text()


        def test_runbook_contains_authoritative_sequence():
            for step in ACTION_ITEMS["verification_sequence"]:
                assert step in RUNBOOK


        def test_runbook_does_not_leave_retired_command_as_an_operator_step():
            assert ACTION_ITEMS["retired_command"] not in RUNBOOK
        """
    )


def oracle_files(variant_id: str) -> dict[str, str]:
    return {
        "repo/runbooks/queue_drain.md": runbook_md(True, variant_id),
        "repo/scripts/queue_drain_helper.py": helper_py(True),
        "repo/.codex/automations/queue-drain-watch/automation.toml": automation_toml(True),
        "repo/ops/notes/queue_drain_followup.md": followup_md(True, variant_id),
    }


def initial_files(variant_id: str, cfg: dict[str, object]) -> dict[str, str]:
    return {
        "AGENTS.md": agents_md(variant_id, cfg),
        "Dockerfile": dockerfile(),
        ".scenario_variant": variant_id + "\n",
        "repo/runbooks/queue_drain.md": runbook_md(False, variant_id),
        "repo/scripts/queue_drain_helper.py": helper_py(False),
        "repo/.codex/automations/queue-drain-watch/automation.toml": automation_toml(False),
        "repo/ops/notes/queue_drain_followup.md": followup_md(False, variant_id),
        "repo/tests/test_queue_drain_helper.py": helper_tests(),
        "repo/tests/test_runbook_contract.py": runbook_tests(),
        "retro/action_items.json": json.dumps(ACTION_ITEMS, indent=2, sort_keys=True) + "\n",
        "retro/incident_2026_02_14_retro.md": incident_retro_md(cfg),
        "retro/operator_timeline.txt": operator_timeline(cfg),
        "artifacts/automation_expectations.json": json.dumps(AUTOMATION_EXPECTATIONS, indent=2, sort_keys=True) + "\n",
        "artifacts/ops_note_schema.json": json.dumps(OPS_NOTE_SCHEMA, indent=2, sort_keys=True) + "\n",
    }


def hash_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path, rel: str) -> str:
    import hashlib

    target = root / rel
    h = hashlib.sha256()
    if not target.exists():
        return "MISSING"
    if target.is_file():
        h.update(b"F")
        h.update(hash_file(target).encode())
        return h.hexdigest()
    for path in sorted(target.rglob("*")):
        rp = path.relative_to(target).as_posix()
        if "/__pycache__/" in f"/{rp}/" or rp.startswith("__pycache__/"):
            continue
        if "/.pytest_cache/" in f"/{rp}/" or rp.startswith(".pytest_cache/"):
            continue
        if rp.endswith((".pyc", ".pyo")):
            continue
        if path.is_dir():
            h.update(b"D:" + rp.encode() + b"\x00")
        elif path.is_file():
            h.update(b"F:" + rp.encode() + b"\x00")
            h.update(hash_file(path).encode() + b"\x00")
    return h.hexdigest()


def manifest_for(workspace: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(workspace.rglob("*")):
        if path.is_file():
            rel = path.relative_to(workspace).as_posix()
            out[rel] = hash_file(path)
    return out


def ensure_variant_assets() -> None:
    shared_hidden = VERIFIER / "_shared/test_hidden_contract.py"
    write(
        shared_hidden,
        textwrap.dedent(
            """\
            from __future__ import annotations

            import os
            from pathlib import Path

            from contract_checks import inspect_surfaces, load_gold, validate_automation_schema, validate_followup_schema


            WORKSPACE = Path(os.environ["AGENT_WS"])
            VARIANT_ID = os.environ["VARIANT_ID"]


            def test_hidden_authoritative_alignment():
                gold = load_gold(VARIANT_ID)
                surfaces = inspect_surfaces(WORKSPACE, gold)
                assert surfaces.helper_command_matches_authority
                assert surfaces.runbook_command_matches_authority
                assert surfaces.runbook_sequence_matches_authority
                assert surfaces.automation_prompt_matches_authority


            def test_hidden_schema_and_note_contract():
                assert validate_automation_schema(WORKSPACE)
                assert validate_followup_schema(WORKSPACE)
            """
        ),
    )
    write(
        VERIFIER / "_milestones_shared/README.md",
        "# Shared milestone scripts for incident-retro-runbook-closure.\n",
    )
    for name in ("m1_localize", "m2_primary_fix", "m3_invariants", "m4_functional", "m5_e2e"):
        script = VERIFIER / "_milestones_shared" / f"{name}.sh"
        write(
            script,
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                set -euo pipefail
                RESULT_FILE="${{RESULT_FILE:-/results/verify_result.json}}"
                python3 - "$RESULT_FILE" <<'PY'
                import json
                import sys
                data = json.load(open(sys.argv[1]))
                mapping = {{
                    "m1_localize": "M1_localization",
                    "m2_primary_fix": "M2_primary_fix",
                    "m3_invariants": "M3_invariants",
                    "m4_functional": "M4_functional",
                    "m5_e2e": "M5_e2e",
                }}
                key = mapping["{name}"]
                sys.exit(0 if data.get("milestones", {{}}).get(key, False) else 1)
                PY
                """
            ),
        )
        executable(script)

    for variant_id, cfg in VARIANTS.items():
        workspace = FAMILY / "workspace_bundle" / variant_id
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        for rel, content in initial_files(variant_id, cfg).items():
            write(workspace / rel, content)
        executable(workspace / "repo/scripts/queue_drain_helper.py")
        if cfg["release_context"]:
            for rel, content in cfg["release_context"].items():
                write(workspace / "release_context" / rel, content)
        if cfg["incident_context"]:
            for rel, content in cfg["incident_context"].items():
                write(workspace / "incident_context" / rel, content)

        variant_verifier = VERIFIER / variant_id
        if variant_verifier.exists():
            shutil.rmtree(variant_verifier)
        (variant_verifier / "oracle").mkdir(parents=True)
        (variant_verifier / "hidden_tests").mkdir(parents=True)
        (variant_verifier / "milestones").mkdir(parents=True)

        gold = {
            "variant_id": variant_id,
            "pass_bar": 65,
            "editable_files": [
                "repo/runbooks/queue_drain.md",
                "repo/scripts/queue_drain_helper.py",
                "repo/.codex/automations/queue-drain-watch/automation.toml",
                "repo/ops/notes/queue_drain_followup.md",
            ],
            "automation_expectations": AUTOMATION_EXPECTATIONS,
            "readonly_tree_hashes": {
                "retro": tree_hash(workspace, "retro"),
                "artifacts": tree_hash(workspace, "artifacts"),
                "repo/tests": tree_hash(workspace, "repo/tests"),
                "release_context": tree_hash(workspace, "release_context"),
                "incident_context": tree_hash(workspace, "incident_context"),
            },
        }
        json_dump(variant_verifier / "gold_repair.json", gold)
        json_dump(variant_verifier / "workspace_manifest.json", {"files": manifest_for(workspace)})
        for rel, content in oracle_files(variant_id).items():
            write(variant_verifier / "oracle" / rel, content)
        os.symlink("../_shared/test_hidden_contract.py", variant_verifier / "hidden_tests" / "test_hidden_contract.py")
        for name in ("m1_localize", "m2_primary_fix", "m3_invariants", "m4_functional", "m5_e2e"):
            os.symlink(f"../../_milestones_shared/{name}.sh", variant_verifier / "milestones" / f"{name}.sh")

    lock = {
        "family_id": "incident-retro-runbook-closure",
        "schema_version": "cnb55.manifest.v1",
        "variants": {},
    }
    for variant_id in VARIANTS:
        ws = FAMILY / "workspace_bundle" / variant_id
        vv = VERIFIER / variant_id
        lock["variants"][variant_id] = {
            "workspace_tree_sha256": tree_hash(ws, "."),
            "gold_repair_sha256": hash_file(vv / "gold_repair.json"),
            "workspace_manifest_sha256": hash_file(vv / "workspace_manifest.json"),
        }
    json_dump(FAMILY / "manifest.lock.json", lock)


if __name__ == "__main__":
    ensure_variant_assets()
