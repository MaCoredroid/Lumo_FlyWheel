#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FAMILY = REPO / "benchmark_blueprints/families/request-path-evidence-brief"
WS_BUNDLE = FAMILY / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data/request-path-evidence-brief"
SCORER = REPO / "verifiers/request-path-evidence-brief/score_trace.py"
MANIFEST_LOCK = FAMILY / "manifest.lock.json"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]
READONLY_RELS = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "sync_app",
    "config",
    "docs",
    "ops",
    "release_context",
    "incident_context",
    "tests",
]
ALLOWED_OUTPUTS = [
    "artifacts/request_path_brief.md",
    "artifacts/path_map.json",
    "artifacts/docs_correction.md",
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
    for path in sorted(target.rglob("*")):
        rel_path = path.relative_to(target).as_posix()
        if rel_path == "__pycache__" or rel_path.startswith("__pycache__/"):
            continue
        if path.name.endswith(".pyc"):
            continue
        if path.is_file():
            h.update(b"F:" + rel_path.encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
        elif path.is_dir():
            h.update(b"D:" + rel_path.encode() + b"\x00")
    return h.hexdigest()


def write(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    if executable:
        path.chmod(0o755)


def json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def common_agents_md() -> str:
    return textwrap.dedent(
        """
        # Request Path Evidence Brief

        You are tracing how `--owner`, `owner_source`, and `routing_key` move through this repo.

        Rules:
        - Do not edit repo behavior.
        - The only allowed writes are:
          - `artifacts/request_path_brief.md`
          - `artifacts/path_map.json`
          - `artifacts/docs_correction.md`
        - Use repo-local evidence only.
        - `request_path_brief.md` must explicitly state whether the support note is correct.
        - In markdown outputs, cite concrete evidence with backticked `path::symbol` tokens, for example `sync_app/service.py::_resolve_owner`.
        - `artifacts/path_map.json` must include `schema_version: "cnb55.request_path_map.v1"` and include:
          - `variant_id`
          - `live_path[]` with `file`, `symbol`, `role`, `caller_symbol`, `callee_symbol`
          - `field_derivations.owner_source`
          - `field_derivations.routing_key`
          - `field_derivations.emission`
          - `test_observations[]`
          - `rejected_decoys[]`
        """
    )


def common_dockerfile() -> str:
    return textwrap.dedent(
        """
        FROM python:3.11-slim
        WORKDIR /workspace
        ENV PYTHONDONTWRITEBYTECODE=1
        """
    )


def cli_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import argparse
        import json

        from sync_app.service import sync_item


        def main(argv: list[str] | None = None) -> str:
            parser = argparse.ArgumentParser()
            parser.add_argument("--name", required=True)
            parser.add_argument("--status", required=True)
            parser.add_argument("--owner")
            args = parser.parse_args(argv)
            payload = sync_item(args.name, args.status, owner=args.owner)
            return json.dumps(payload, sort_keys=True)


        if __name__ == "__main__":
            print(main())
        """
    )


def service_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import json
        from pathlib import Path

        from sync_app.serializer import build_routing_key, serialize_payload
        from sync_app.store import make_record


        def _load_defaults() -> dict[str, str]:
            defaults_path = Path(__file__).resolve().parents[1] / "config" / "defaults.json"
            return json.loads(defaults_path.read_text(encoding="utf-8"))


        def _resolve_owner(owner: str | None) -> tuple[str, str]:
            if owner and owner.strip():
                return owner.strip(), "explicit"
            return _load_defaults()["owner"], "default"


        def sync_item(name: str, status: str, owner: str | None = None) -> dict[str, str]:
            effective_owner, owner_source = _resolve_owner(owner)
            record = make_record(name=name, status=status, owner=effective_owner)
            routing_key = build_routing_key(effective_owner, name)
            return serialize_payload(record, owner_source=owner_source, routing_key=routing_key)
        """
    )


def store_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations


        def make_record(name: str, status: str, owner: str) -> dict[str, str]:
            return {"name": name, "status": status, "owner": owner}


        def legacy_make_record_with_owner_source(
            name: str,
            status: str,
            owner: str,
            owner_source: str,
        ) -> dict[str, str]:
            payload = make_record(name=name, status=status, owner=owner)
            payload["owner_source"] = owner_source
            return payload


        def legacy_build_routing_key(owner: str, name: str) -> str:
            compact_name = "-".join(name.lower().split())
            return f"{owner.lower()}:{compact_name}"
        """
    )


def serializer_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations


        def _slugify(text: str) -> str:
            return "-".join(text.lower().split())


        def build_routing_key(owner: str, name: str) -> str:
            return f"{_slugify(owner)}:{_slugify(name)}"


        def serialize_payload(
            record: dict[str, str],
            owner_source: str,
            routing_key: str,
        ) -> dict[str, str]:
            payload = dict(record)
            payload["owner_source"] = owner_source
            payload["routing_key"] = routing_key
            return payload


        def draft_owner_source_from_record(record: dict[str, str]) -> str:
            return record.get("owner_source", "unknown")
        """
    )


def test_sync_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import json
        from pathlib import Path

        from sync_app.cli import main
        from sync_app.service import sync_item


        def _default_owner() -> str:
            return json.loads(Path("config/defaults.json").read_text(encoding="utf-8"))["owner"]


        def test_service_resolves_explicit_owner_before_serialization() -> None:
            payload = sync_item("Launch Checklist", "pending", owner="pm-oncall")
            assert payload == {
                "name": "Launch Checklist",
                "status": "pending",
                "owner": "pm-oncall",
                "owner_source": "explicit",
                "routing_key": "pm-oncall:launch-checklist",
            }


        def test_service_uses_default_owner_when_flag_is_missing() -> None:
            payload = sync_item("Launch Checklist", "pending")
            assert payload == {
                "name": "Launch Checklist",
                "status": "pending",
                "owner": _default_owner(),
                "owner_source": "default",
                "routing_key": f"{_default_owner()}:launch-checklist",
            }


        def test_cli_accepts_owner_flag_and_preserves_existing_fields() -> None:
            payload = json.loads(
                main(
                    [
                        "--name",
                        "Launch Checklist",
                        "--status",
                        "pending",
                        "--owner",
                        "pm-oncall",
                    ]
                )
            )
            assert payload == {
                "name": "Launch Checklist",
                "status": "pending",
                "owner": "pm-oncall",
                "owner_source": "explicit",
                "routing_key": "pm-oncall:launch-checklist",
            }
        """
    )


def test_docs_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import json
        from pathlib import Path


        def test_docs_reference_owner_path_surfaces() -> None:
            docs = Path("docs/cli.md").read_text(encoding="utf-8")
            data_flow = Path("docs/data_flow.md").read_text(encoding="utf-8")
            defaults = json.loads(Path("config/defaults.json").read_text(encoding="utf-8"))
            assert "--owner" in docs
            assert "owner_source" in docs
            assert "routing_key" in docs
            assert "sync_item" in data_flow
            assert defaults["owner"] in docs
        """
    )


def test_trace_outputs_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import json
        from pathlib import Path


        def test_path_outputs_exist_and_parse() -> None:
            base = Path("artifacts")
            path_map = base / "path_map.json"
            brief = base / "request_path_brief.md"
            correction = base / "docs_correction.md"
            assert path_map.exists()
            assert brief.exists()
            assert correction.exists()

            data = json.loads(path_map.read_text(encoding="utf-8"))
            assert data["schema_version"] == "cnb55.request_path_map.v1"
            assert data["variant_id"] == Path(".scenario_variant").read_text(encoding="utf-8").strip()
            assert isinstance(data["live_path"], list) and data["live_path"]
            assert isinstance(data["field_derivations"], dict)
            assert isinstance(data["test_observations"], list) and data["test_observations"]
            assert isinstance(data["rejected_decoys"], list) and data["rejected_decoys"]
        """
    )


def test_conftest_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import sys
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[1]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        """
    )


def cli_md() -> str:
    return textwrap.dedent(
        """
        # project board sync CLI

        Current command:

        ```bash
        python -m sync_app.cli --name launch-checklist --status pending --owner pm-oncall
        ```

        The emitted payload includes:
        - `owner`
        - `owner_source`
        - `routing_key`

        If `--owner` is omitted, the service falls back to the default owner from `config/defaults.json`.
        """
    )


def data_flow_md(variant: str) -> str:
    base = textwrap.dedent(
        """
        # Data Flow Note

        The CLI entrypoint calls `sync_item`, which resolves the effective owner, persists the base record, and then emits the payload.

        Historical notes often over-compressed the flow because the stored record already contains `owner`. That shortcut is not enough to prove where `owner_source` or `routing_key` are decided.
        """
    ).strip()
    if variant == "v4-multi-corpus-objective":
        return (
            base
            + "\n\nPlanned cleanup for the next release may move more payload assembly into the serializer. Treat release planning notes as future-state only unless the code matches them.\n"
        )
    if variant == "v5-recovery-in-thread":
        return (
            base
            + "\n\nA previous docs correction over-anchored on storage, then got rolled back during incident follow-up. The incident notes are historical evidence, not proof that the implementation changed.\n"
        )
    return base + "\n"


def support_note_md() -> str:
    return textwrap.dedent(
        """
        # Escalation Summary

        Support note from 2026-04-10:

        - `owner_source` in the exported project-board payload looks like it comes from storage because the record already has `owner` when it comes back.
        - `routing_key` is probably computed before the CLI applies `--owner`.
        - Please confirm whether the fix belongs in the storage layer or only in docs.
        """
    )


def previous_session_md() -> str:
    return textwrap.dedent(
        """
        # Previous Session Draft

        Draft claim:
        - Store helper likely became the source of truth after the serializer cleanup.
        - We can probably reuse the old support explanation if the tests still pass.

        This file was left unfinished and was never merged into the docs.
        """
    )


def future_serializer_split_md() -> str:
    return textwrap.dedent(
        """
        # Planned Serializer Split

        Next-quarter plan:
        - Move more payload assembly into serializer helpers.
        - Consider whether owner attribution should also be computed there.

        This document is design intent only. It does not assert that the live code already moved `owner_source` into serializer.
        """
    )


def release_notes_md() -> str:
    return textwrap.dedent(
        """
        # Release Notes 2026-06

        Follow-up work:
        - Untangle historical support notes from live code paths.
        - Keep design notes separate from implemented behavior during request-path audits.
        """
    )


def prior_correction_md() -> str:
    return textwrap.dedent(
        """
        # Prior Docs Correction

        Rolled-back draft:
        - Claimed storage decided `owner_source`.
        - Claimed `routing_key` was built before CLI owner precedence applied.

        This draft was reverted after incident review found that it described a stale theory, not the live path.
        """
    )


def rollback_note_md() -> str:
    return textwrap.dedent(
        """
        # Incident Follow-up

        The first docs correction for this support thread was rolled back.

        Reason:
        - The correction repeated the support note's storage-layer explanation without tracing the live call chain.
        - Future-state serializer planning notes were cited as if they described the current implementation.
        """
    )


def artifacts_readme() -> str:
    return textwrap.dedent(
        """
        Write only these files here:
        - `request_path_brief.md`
        - `path_map.json`
        - `docs_correction.md`
        """
    )


def workspace_files(variant: str) -> dict[str, str]:
    files = {
        ".scenario_variant": variant,
        "AGENTS.md": common_agents_md(),
        "Dockerfile": common_dockerfile(),
        "sync_app/__init__.py": "",
        "sync_app/cli.py": cli_py(),
        "sync_app/service.py": service_py(),
        "sync_app/store.py": store_py(),
        "sync_app/serializer.py": serializer_py(),
        "config/defaults.json": json_text({"owner": "pm-oncall", "status": "pending"}),
        "docs/cli.md": cli_md(),
        "docs/data_flow.md": data_flow_md(variant),
        "ops/support_note.md": support_note_md(),
        "tests/test_sync.py": test_sync_py(),
        "tests/test_docs.py": test_docs_py(),
        "tests/test_trace_outputs.py": test_trace_outputs_py(),
        "tests/conftest.py": test_conftest_py(),
        "artifacts/README.md": artifacts_readme(),
    }
    if variant in {"v3-dirty-state", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        files["ops/previous_session_path.md"] = previous_session_md()
    if variant in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        files["release_context/future_serializer_split.md"] = future_serializer_split_md()
        files["release_context/release_notes_2026_06.md"] = release_notes_md()
    if variant == "v5-recovery-in-thread":
        files["incident_context/prior_docs_correction.md"] = prior_correction_md()
        files["incident_context/rollback_note.md"] = rollback_note_md()
    return files


def live_path() -> list[dict[str, object]]:
    return [
        {
            "step": 1,
            "file": "sync_app/cli.py",
            "symbol": "main",
            "role": "cli_entry",
            "caller_symbol": "argv.--owner",
            "callee_symbol": "sync_app.service.sync_item",
        },
        {
            "step": 2,
            "file": "sync_app/service.py",
            "symbol": "sync_item",
            "role": "service_orchestrator",
            "caller_symbol": "sync_app.cli.main",
            "callee_symbol": "sync_app.service._resolve_owner",
        },
        {
            "step": 3,
            "file": "sync_app/service.py",
            "symbol": "_resolve_owner",
            "role": "owner_selection",
            "caller_symbol": "sync_app.service.sync_item",
            "callee_symbol": "sync_app.store.make_record",
        },
        {
            "step": 4,
            "file": "sync_app/store.py",
            "symbol": "make_record",
            "role": "storage_write_shape",
            "caller_symbol": "sync_app.service.sync_item",
            "callee_symbol": "sync_app.serializer.build_routing_key",
        },
        {
            "step": 5,
            "file": "sync_app/serializer.py",
            "symbol": "build_routing_key",
            "role": "derived_field",
            "caller_symbol": "sync_app.service.sync_item",
            "callee_symbol": "sync_app.serializer.serialize_payload",
        },
        {
            "step": 6,
            "file": "sync_app/serializer.py",
            "symbol": "serialize_payload",
            "role": "emission",
            "caller_symbol": "sync_app.service.sync_item",
            "callee_symbol": None,
        },
    ]


def base_derivations() -> dict[str, dict[str, str]]:
    return {
        "owner_source": {
            "file": "sync_app/service.py",
            "symbol": "_resolve_owner",
            "why": "selects explicit vs default owner and labels the source",
        },
        "routing_key": {
            "file": "sync_app/serializer.py",
            "symbol": "build_routing_key",
            "why": "derives the key from the resolved owner and task name",
        },
        "emission": {
            "file": "sync_app/serializer.py",
            "symbol": "serialize_payload",
            "why": "emits owner_source and routing_key into the final payload",
        },
    }


def test_observations() -> list[dict[str, object]]:
    return [
        {
            "file": "tests/test_sync.py",
            "symbol": "test_cli_accepts_owner_flag_and_preserves_existing_fields",
            "proves": ["cli_forwards_owner", "owner_source_explicit", "routing_key_uses_resolved_owner"],
        },
        {
            "file": "tests/test_sync.py",
            "symbol": "test_service_uses_default_owner_when_flag_is_missing",
            "proves": ["default_owner_path", "owner_source_default"],
        },
    ]


def variant_decoy(variant: str) -> dict[str, object]:
    if variant == "v1-clean-baseline":
        return {
            "file": "sync_app/store.py",
            "symbol": "legacy_make_record_with_owner_source",
            "reason": "not imported or called from the live service path; storage only returns the base record",
            "reason_keywords": ["not imported", "not called", "base record", "storage only"],
        }
    if variant == "v2-noisy-distractor":
        return {
            "file": "sync_app/store.py",
            "symbol": "legacy_build_routing_key",
            "reason": "looks plausible but the live code builds routing_key in serializer, not in the store helper",
            "reason_keywords": ["serializer", "not live", "store helper"],
        }
    if variant == "v3-dirty-state":
        return {
            "file": "ops/previous_session_path.md",
            "symbol": "draft_path",
            "reason": "previous session draft is stale prose, not a live caller or callee",
            "reason_keywords": ["stale", "draft", "not live", "prose"],
        }
    if variant == "v4-multi-corpus-objective":
        return {
            "file": "release_context/future_serializer_split.md",
            "symbol": "planned_state",
            "reason": "release planning describes future cleanup only and does not replace the live code path",
            "reason_keywords": ["future", "planned", "not implemented", "cleanup"],
        }
    return {
        "file": "incident_context/prior_docs_correction.md",
        "symbol": "rolled_back_patch",
        "reason": "rolled-back correction repeated the stale theory and was explicitly invalidated by incident follow-up",
        "reason_keywords": ["rolled back", "incident", "stale theory", "invalidated"],
    }


def forbidden_positive_files(variant: str) -> list[str]:
    if variant == "v3-dirty-state":
        return ["ops/previous_session_path.md"]
    if variant == "v4-multi-corpus-objective":
        return ["release_context/future_serializer_split.md"]
    if variant == "v5-recovery-in-thread":
        return ["incident_context/prior_docs_correction.md"]
    return []


def required_brief_tokens() -> list[str]:
    return [
        "sync_app/cli.py::main",
        "sync_app/service.py::sync_item",
        "sync_app/service.py::_resolve_owner",
        "sync_app/serializer.py::build_routing_key",
        "sync_app/serializer.py::serialize_payload",
    ]


def required_correction_tokens(variant: str) -> list[str]:
    tokens = [
        "ops/support_note.md::escalation_summary",
        "sync_app/service.py::_resolve_owner",
        "sync_app/store.py::make_record",
        "sync_app/serializer.py::build_routing_key",
        "sync_app/serializer.py::serialize_payload",
    ]
    if variant == "v5-recovery-in-thread":
        tokens[0] = "incident_context/rollback_note.md::incident_follow_up"
    return tokens


def oracle_path_map(variant: str) -> dict[str, object]:
    decoy = variant_decoy(variant)
    return {
        "schema_version": "cnb55.request_path_map.v1",
        "variant_id": variant,
        "summary": "The support note is incorrect. owner_source is decided in service, routing_key is derived in serializer after owner resolution, and both fields are emitted during payload serialization.",
        "support_note_verdict": {
            "file": "ops/support_note.md",
            "symbol": "escalation_summary",
            "verdict": "incorrect",
        },
        "live_path": live_path(),
        "field_derivations": base_derivations(),
        "test_observations": test_observations(),
        "rejected_decoys": [
            {
                "file": decoy["file"],
                "symbol": decoy["symbol"],
                "reason": decoy["reason"],
            }
        ],
    }


def oracle_brief_md(variant: str) -> str:
    decoy = variant_decoy(variant)
    extra = ""
    if variant == "v5-recovery-in-thread":
        extra = (
            "\nPrior correction note: `incident_context/rollback_note.md::incident_follow_up` explains why the earlier store-layer explanation was rolled back.\n"
        )
    return textwrap.dedent(
        f"""
        # Request Path Brief

        Verdict: the support note is incorrect.

        Live path:
        1. `sync_app/cli.py::main` parses `--owner` and forwards it into `sync_app/service.py::sync_item`.
        2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner`, where the effective owner and `owner_source` are actually chosen.
        3. `sync_app/store.py::make_record` only persists the base record after owner selection; it does not decide `owner_source`.
        4. `sync_app/serializer.py::build_routing_key` derives `routing_key` from the resolved owner and item name.
        5. `sync_app/serializer.py::serialize_payload` is the emission step that exposes both `owner_source` and `routing_key`.

        Test evidence:
        - `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`
        - `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`

        Rejected decoy:
        - `{decoy["file"]}::{decoy["symbol"]}` is not live because {decoy["reason"]}.
        {extra}
        """
    ).strip() + "\n"


def oracle_correction_md(variant: str) -> str:
    header = "`ops/support_note.md::escalation_summary` is incorrect."
    body = (
        "The live code resolves the effective owner in `sync_app/service.py::_resolve_owner`, "
        "stores only the base record via `sync_app/store.py::make_record`, derives `routing_key` in "
        "`sync_app/serializer.py::build_routing_key`, and emits both fields in "
        "`sync_app/serializer.py::serialize_payload`."
    )
    if variant == "v5-recovery-in-thread":
        header = "`incident_context/rollback_note.md::incident_follow_up` shows the earlier correction was rolled back."
        body += " The replacement correction should explicitly say the old store-layer explanation was stale, not reassert it."
    return f"{header}\n\n{body}\n"


def shortcut_path_map(variant: str) -> dict[str, object]:
    return {
        "schema_version": "cnb55.request_path_map.v1",
        "variant_id": variant,
        "summary": "Storage decides everything important.",
        "support_note_verdict": {
            "file": "ops/support_note.md",
            "symbol": "escalation_summary",
            "verdict": "correct",
        },
        "live_path": [
            {
                "step": 1,
                "file": "sync_app/store.py",
                "symbol": "legacy_make_record_with_owner_source",
                "role": "storage_decision",
                "caller_symbol": "ops/support_note.md",
                "callee_symbol": "sync_app/store.py::legacy_build_routing_key",
            }
        ],
        "field_derivations": {
            "owner_source": {"file": "sync_app/store.py", "symbol": "legacy_make_record_with_owner_source"},
            "routing_key": {"file": "sync_app/store.py", "symbol": "legacy_build_routing_key"},
            "emission": {"file": "sync_app/store.py", "symbol": "legacy_make_record_with_owner_source"},
        },
        "test_observations": [],
        "rejected_decoys": [],
    }


def write_workspace_bundle() -> None:
    shutil.rmtree(WS_BUNDLE, ignore_errors=True)
    for variant in VARIANTS:
        root = WS_BUNDLE / variant
        for rel, content in workspace_files(variant).items():
            write(root / rel, content)


def workspace_manifest(root: Path) -> dict[str, object]:
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in ALLOWED_OUTPUTS
    )
    return {"schema_version": "cnb55.workspace_manifest.v1", "files": files}


def write_milestones_shared() -> None:
    shared = VERIFIER_DATA / "_milestones_shared"
    scripts = {
        "m1_localize.sh": "M1_localization",
        "m2_primary_fix.sh": "M2_primary_fix",
        "m3_invariants.sh": "M3_invariants",
        "m4_functional.sh": "M4_functional",
        "m5_e2e.sh": "M5_e2e",
    }
    for name, milestone in scripts.items():
        write(
            shared / name,
            textwrap.dedent(
                f"""
                #!/usr/bin/env bash
                set -euo pipefail
                RESULT_FILE="${{RESULT_FILE:-/results/verify_result.json}}"
                python3 - "$RESULT_FILE" <<'PY'
                import json, sys
                data = json.load(open(sys.argv[1]))
                sys.exit(0 if data.get("milestones", {{}}).get("{milestone}", False) else 1)
                PY
                """
            ),
            executable=True,
        )


def hidden_test_py() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import json
        import os
        from pathlib import Path

        RESULT_FILE = Path(os.environ["RESULT_FILE"])


        def test_functional_milestone_present() -> None:
            result = json.loads(RESULT_FILE.read_text())
            assert "M4_functional" in result["milestones"]


        def test_no_shortcut_for_oracle() -> None:
            result = json.loads(RESULT_FILE.read_text())
            assert isinstance(result["shortcut_detected"], bool)
        """
    )


def write_verifier_data() -> None:
    shutil.rmtree(VERIFIER_DATA, ignore_errors=True)
    write_milestones_shared()
    for variant in VARIANTS:
        variant_dir = VERIFIER_DATA / variant
        root = WS_BUNDLE / variant
        manifest = workspace_manifest(root)
        write(variant_dir / "workspace_manifest.json", json_text(manifest))
        gold = {
            "schema_version": "cnb55.request_path_gold.v1",
            "variant_id": variant,
            "pass_bar": 40,
            "live_path": live_path(),
            "field_derivations": base_derivations(),
            "test_observations": test_observations(),
            "required_decoy": variant_decoy(variant),
            "forbidden_positive_files": forbidden_positive_files(variant),
            "required_brief_tokens": required_brief_tokens(),
            "required_correction_tokens": required_correction_tokens(variant),
            "readonly_tree_hashes": {
                rel: sha256_tree(root, rel)
                for rel in READONLY_RELS
                if sha256_tree(root, rel) is not None
            },
        }
        write(variant_dir / "gold_path.json", json_text(gold))
        write(variant_dir / "oracle" / "path_map.json", json_text(oracle_path_map(variant)))
        write(variant_dir / "oracle" / "request_path_brief.md", oracle_brief_md(variant))
        write(variant_dir / "oracle" / "docs_correction.md", oracle_correction_md(variant))
        write(variant_dir / "hidden_tests" / "test_trace_properties.py", hidden_test_py())
        milestones = variant_dir / "milestones"
        milestones.mkdir(parents=True, exist_ok=True)
        for name in ("m1_localize.sh", "m2_primary_fix.sh", "m3_invariants.sh", "m4_functional.sh", "m5_e2e.sh"):
            target = VERIFIER_DATA / "_milestones_shared" / name
            link = milestones / name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(os.path.relpath(target, milestones))


def populate_artifacts(root: Path, variant: str, kind: str) -> None:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    if kind == "empty":
        return
    if kind == "oracle":
        source = VERIFIER_DATA / variant / "oracle"
        for name in ("path_map.json", "request_path_brief.md", "docs_correction.md"):
            shutil.copy2(source / name, artifacts / name)
        return
    if kind == "shortcut":
        write(artifacts / "path_map.json", json_text(shortcut_path_map(variant)))
        write(artifacts / "request_path_brief.md", "Storage decides the payload, so the support note is correct.\n")
        write(artifacts / "docs_correction.md", "No correction needed.\n")
        return
    if kind == "grounding_stripped":
        write(artifacts / "path_map.json", json_text(oracle_path_map(variant)))
        write(artifacts / "request_path_brief.md", "The flow looks straightforward.\n")
        write(artifacts / "docs_correction.md", "Docs should be updated.\n")
        return
    raise ValueError(f"unknown artifact kind: {kind}")


def run_score(root: Path, variant: str) -> dict[str, object]:
    result_file = root / "verify_result.json"
    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(root),
            "VERIFIER_DATA": str(VERIFIER_DATA),
            "VARIANT_ID": variant,
            "RESULT_FILE": str(result_file),
        }
    )
    subprocess.run(
        [sys.executable, str(SCORER)],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return json.loads(result_file.read_text())


def observed_score(variant: str, kind: str) -> int:
    with tempfile.TemporaryDirectory(prefix=f"request_path_{variant}_") as tmp:
        root = Path(tmp) / "workspace"
        shutil.copytree(WS_BUNDLE / variant, root)
        populate_artifacts(root, variant, kind)
        return int(run_score(root, variant)["P_benchmark"])


def build_manifest_lock() -> None:
    payload: dict[str, object] = {
        "schema_version": "cnb55.manifest.v2",
        "last_regen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "grader": {"score_trace_py_sha256": sha256_file(SCORER)},
        "variants": {},
    }
    for variant in VARIANTS:
        variant_dir = VERIFIER_DATA / variant
        payload["variants"][variant] = {
            "observed_oracle_score": observed_score(variant, "oracle"),
            "observed_empty_brief_score": observed_score(variant, "empty"),
            "observed_shortcut_score": observed_score(variant, "shortcut"),
            "observed_grounding_stripped_score": observed_score(variant, "grounding_stripped"),
            "verifier_data": {
                "gold_path_sha256": sha256_file(variant_dir / "gold_path.json"),
                "hidden_tests_tree_sha256": sha256_tree(variant_dir, "hidden_tests"),
                "oracle_path_map_sha256": sha256_file(variant_dir / "oracle" / "path_map.json"),
                "oracle_brief_md_sha256": sha256_file(variant_dir / "oracle" / "request_path_brief.md"),
                "oracle_correction_md_sha256": sha256_file(variant_dir / "oracle" / "docs_correction.md"),
                "workspace_manifest_sha256": sha256_file(variant_dir / "workspace_manifest.json"),
            },
            "workspace_trees": {
                rel: sha256_tree(WS_BUNDLE / variant, rel)
                for rel in READONLY_RELS
                if sha256_tree(WS_BUNDLE / variant, rel) is not None
            },
        }
    write(MANIFEST_LOCK, json_text(payload))


def main() -> int:
    write_workspace_bundle()
    write_verifier_data()
    build_manifest_lock()
    print(MANIFEST_LOCK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
