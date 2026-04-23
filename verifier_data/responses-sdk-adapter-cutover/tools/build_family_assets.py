#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FAMILY = REPO / "benchmark_blueprints/families/responses-sdk-adapter-cutover"
VERIFIER_DATA = REPO / "verifier_data/responses-sdk-adapter-cutover"
VERIFIERS = REPO / "verifiers/responses-sdk-adapter-cutover"

VARIANTS = {
    "v1-clean-baseline": {
        "required_doc_keywords": ["event ordering", "tool-result correlation"],
        "pass_bar": 40,
        "weights": {
            "visible.pytest": 10,
            "config.responses_wire": 10,
            "config.responses_mode": 5,
            "docs.event_ordering": 5,
            "docs.tool_result_correlation": 5,
            "hidden.interleaved_order": 15,
            "hidden.function_alias_normalization": 15,
            "hidden.multi_block_message": 0,
            "hidden.replay_roundtrip": 15,
            "hidden.reordered_chunk_stability": 0,
            "hidden.future_event_passthrough": 0,
            "hidden.legacy_path_removed": 15,
            "docs.variant_complete": 5,
        },
    },
    "v2-noisy-distractor": {
        "required_doc_keywords": ["event ordering", "tool-result correlation"],
        "pass_bar": 40,
        "weights": {
            "visible.pytest": 10,
            "config.responses_wire": 5,
            "config.responses_mode": 5,
            "docs.event_ordering": 5,
            "docs.tool_result_correlation": 5,
            "hidden.interleaved_order": 10,
            "hidden.multi_block_message": 15,
            "hidden.replay_roundtrip": 15,
            "hidden.reordered_chunk_stability": 0,
            "hidden.future_event_passthrough": 0,
            "hidden.legacy_path_removed": 20,
            "docs.variant_complete": 10,
        },
    },
    "v3-dirty-state": {
        "required_doc_keywords": ["event ordering", "tool-result correlation"],
        "pass_bar": 40,
        "weights": {
            "visible.pytest": 10,
            "config.responses_wire": 5,
            "config.responses_mode": 5,
            "docs.event_ordering": 5,
            "docs.tool_result_correlation": 5,
            "hidden.interleaved_order": 10,
            "hidden.multi_block_message": 10,
            "hidden.replay_roundtrip": 15,
            "hidden.reordered_chunk_stability": 15,
            "hidden.future_event_passthrough": 0,
            "hidden.legacy_path_removed": 15,
            "docs.variant_complete": 5,
        },
    },
    "v4-multi-corpus-objective": {
        "required_doc_keywords": ["event ordering", "tool-result correlation", "event-sourced"],
        "pass_bar": 40,
        "weights": {
            "visible.pytest": 10,
            "config.responses_wire": 5,
            "config.responses_mode": 5,
            "docs.event_ordering": 5,
            "docs.tool_result_correlation": 5,
            "hidden.interleaved_order": 10,
            "hidden.multi_block_message": 10,
            "hidden.replay_roundtrip": 15,
            "hidden.reordered_chunk_stability": 10,
            "hidden.future_event_passthrough": 0,
            "hidden.legacy_path_removed": 10,
            "docs.variant_complete": 15,
        },
    },
    "v5-recovery-in-thread": {
        "required_doc_keywords": ["event ordering", "tool-result correlation", "incident recovery", "future event"],
        "pass_bar": 40,
        "weights": {
            "visible.pytest": 10,
            "config.responses_wire": 5,
            "config.responses_mode": 5,
            "docs.event_ordering": 5,
            "docs.tool_result_correlation": 5,
            "hidden.interleaved_order": 10,
            "hidden.multi_block_message": 10,
            "hidden.replay_roundtrip": 10,
            "hidden.reordered_chunk_stability": 10,
            "hidden.future_event_passthrough": 15,
            "hidden.legacy_path_removed": 10,
            "docs.variant_complete": 5,
        },
    },
}

ALLOWED_WRITES = [
    "config/runtime.toml",
    "docs/migrations/responses-cutover.md",
    "src/incident_handoff/client.py",
    "src/incident_handoff/adapter.py",
    "src/incident_handoff/replay.py",
    "src/incident_handoff/render.py",
]

READONLY_ROOTS = [
    "tests/",
    "transcripts/",
    "release_context/",
    "incident_context/",
]

LEGACY_MARKERS = [
    "chat_completions",
    "legacy wrapper",
    "legacy_messages",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.endswith(".pyc") or "/__pycache__/" in rel or rel.startswith(".pytest_cache/"):
            continue
        hashes[rel] = sha256_file(path)
    return hashes


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    manifest_payload = {
        "family_id": "responses-sdk-adapter-cutover",
        "schema_version": "cnb55.manifest.v2",
        "variants": {},
        "grader": {
            "score_responses_cutover_py_sha256": sha256_file(VERIFIERS / "score_responses_cutover.py"),
            "verify_sh_sha256": sha256_file(VERIFIERS / "verify.sh"),
        },
    }

    for variant_id, variant_meta in VARIANTS.items():
        workspace = FAMILY / "workspace_bundle" / variant_id
        hashes = file_hashes(workspace)
        verifier_variant = VERIFIER_DATA / variant_id
        write_json(
            verifier_variant / "workspace_manifest.json",
            {
                "variant_id": variant_id,
                "files": hashes,
            },
        )
        write_json(
            verifier_variant / "gold_reference.json",
            {
                "variant_id": variant_id,
                "pass_bar": variant_meta["pass_bar"],
                "weights": variant_meta["weights"],
                "allowed_writes": ALLOWED_WRITES,
                "readonly_roots": READONLY_ROOTS,
                "legacy_live_markers": LEGACY_MARKERS,
                "required_doc_keywords": variant_meta["required_doc_keywords"],
            },
        )
        manifest_payload["variants"][variant_id] = {
            "workspace_manifest_sha256": sha256_file(verifier_variant / "workspace_manifest.json"),
            "gold_reference_sha256": sha256_file(verifier_variant / "gold_reference.json"),
        }

    write_json(FAMILY / "manifest.lock.json", manifest_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
