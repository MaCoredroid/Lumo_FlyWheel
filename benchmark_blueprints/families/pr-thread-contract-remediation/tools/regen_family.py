#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
FAMILY_ROOT = REPO / "benchmark_blueprints" / "families" / "pr-thread-contract-remediation"
WORKSPACE_ROOT = FAMILY_ROOT / "workspace_bundle"
VERIFIER_DATA_ROOT = REPO / "verifier_data" / "pr-thread-contract-remediation"
VERIFIER_ROOT = REPO / "verifiers" / "pr-thread-contract-remediation"
SCORER = VERIFIER_ROOT / "score_pr_thread_contract.py"
MANIFEST_LOCK = FAMILY_ROOT / "manifest.lock.json"

VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

ALLOWED_WRITE_PATHS = [
    "queue_api/handlers.py",
    "queue_api/serializers.py",
    "queue_api/service.py",
    "tests/test_handlers.py",
    "tests/test_service.py",
    "docs/release_notes.md",
    "CHANGELOG.md",
    "review/reviewer_replies.json",
    "review/verification_note.md",
]

BASE_ROWS = [
    {"queue_id": "q-001", "owner": "bob"},
    {"queue_id": "q-002", "owner": "bob"},
    {"queue_id": "q-003", "owner": "alice"},
    {"queue_id": "q-004", "owner": None},
]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.exists():
        return None
    if target.is_file():
        return sha256_file(target)
    h = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rel_path = path.relative_to(target).as_posix()
        if path.is_dir():
            h.update(f"D:{rel_path}\n".encode())
        else:
            h.update(f"F:{rel_path}\n".encode())
            h.update(sha256_file(path).encode())
            h.update(b"\n")
    return h.hexdigest()


def thread_export(variant_id: str) -> str:
    threads = [
        {
            "thread_id": "RTR-101",
            "state": "unresolved",
            "path": "queue_api/serializers.py",
            "subject": "Unowned response buckets should omit owner entirely",
            "body": (
                "Clients already treat a missing `owner` key as unassigned. "
                "Serializing `owner: null` changed the response contract. "
                "Please fix the default and export payloads and leave a reply "
                "with the changed files."
            ),
        },
        {
            "thread_id": "RTR-102",
            "state": "unresolved",
            "path": "queue_api/service.py",
            "subject": "Append unowned after current owner order",
            "body": (
                "When `include_unowned=true`, append the unowned bucket after "
                "the pre-existing owner buckets. Do not globally re-sort the "
                "owners, especially on equal counts."
            ),
        },
        {
            "thread_id": "RTR-103",
            "state": "unresolved",
            "path": "docs/release_notes.md",
            "subject": "Document the contract nuance and close with evidence",
            "body": (
                "Please document that unowned buckets omit `owner` instead of "
                "serializing `null`, keep request-side owner filter semantics "
                "unchanged, and reply with the concrete files/tests you touched."
            ),
        },
        {
            "thread_id": "RTR-104",
            "state": "resolved",
            "path": "queue_api/serializers.py",
            "subject": "Rename owner_label",
            "body": "Resolved earlier in a separate cleanup diff. No follow-up here.",
        },
        {
            "thread_id": "RTR-105",
            "state": "outdated",
            "path": "queue_api/service.py",
            "subject": "Alphabetize owner buckets",
            "body": (
                "This was attached to an older diff and is no longer the right "
                "direction. The current contract is to preserve owner order."
            ),
        },
    ]
    if variant_id == "v4-multi-corpus-objective":
        threads[2]["body"] += " The release_context files explain why mobile compatibility is the real objective this cycle."
    if variant_id == "v5-recovery-in-thread":
        threads[2]["body"] += " Mention the INC-742 rollback when you close this thread."
    return json.dumps(threads, indent=2, sort_keys=True) + "\n"


def baseline_handlers() -> str:
    return """from __future__ import annotations

from queue_api import serializers, service


class QueueRepository:
    def __init__(self, rows):
        self._rows = list(rows)

    def list_rows(self):
        return list(self._rows)


def parse_owner_filter(params):
    if "owner" not in params:
        return service.MISSING
    return params.get("owner")


def get_queue_summary(params, repository):
    buckets = service.build_summary(
        repository.list_rows(),
        owner_filter=parse_owner_filter(params),
        include_unowned=bool(params.get("include_unowned")),
    )
    serializer = serializers.serialize_export_bucket if params.get("mode") == "export" else serializers.serialize_bucket
    return {"summary": [serializer(bucket) for bucket in buckets]}
"""


def baseline_service() -> str:
    return """from __future__ import annotations

MISSING = object()


def build_summary(rows, owner_filter=MISSING, include_unowned=False):
    buckets = []
    by_key = {}
    for row in rows:
        owner = row.get("owner")
        if owner_filter is not MISSING:
            if owner_filter is None:
                if owner is not None:
                    continue
            elif owner != owner_filter:
                continue
        key = owner if owner is not None else "__unowned__"
        bucket = by_key.get(key)
        if bucket is None:
            bucket = {"owner": owner, "count": 0, "queue_ids": []}
            by_key[key] = bucket
            buckets.append(bucket)
        bucket["count"] += 1
        bucket["queue_ids"].append(row["queue_id"])
    if not include_unowned:
        return [bucket for bucket in buckets if bucket["owner"] is not None]
    return sorted(
        buckets,
        key=lambda bucket: (bucket["owner"] is None, -bucket["count"], bucket["owner"] or ""),
    )
"""


def fixed_service() -> str:
    return """from __future__ import annotations

MISSING = object()


def build_summary(rows, owner_filter=MISSING, include_unowned=False):
    buckets = []
    by_key = {}
    for row in rows:
        owner = row.get("owner")
        if owner_filter is not MISSING:
            if owner_filter is None:
                if owner is not None:
                    continue
            elif owner != owner_filter:
                continue
        key = owner if owner is not None else "__unowned__"
        bucket = by_key.get(key)
        if bucket is None:
            bucket = {"owner": owner, "count": 0, "queue_ids": []}
            by_key[key] = bucket
            buckets.append(bucket)
        bucket["count"] += 1
        bucket["queue_ids"].append(row["queue_id"])
    owned = [bucket for bucket in buckets if bucket["owner"] is not None]
    if not include_unowned:
        return owned
    unowned = [bucket for bucket in buckets if bucket["owner"] is None]
    return owned + unowned
"""


def baseline_serializers() -> str:
    return """from __future__ import annotations


def _base(bucket):
    return {"count": bucket["count"], "queue_ids": list(bucket["queue_ids"])}


def serialize_bucket(bucket):
    payload = _base(bucket)
    payload["owner"] = bucket["owner"]
    return payload


def serialize_export_bucket(bucket):
    payload = _base(bucket)
    payload["owner"] = bucket["owner"]
    payload["owner_label"] = bucket["owner"] or "unowned"
    return payload
"""


def fixed_serializers() -> str:
    return """from __future__ import annotations


def _base(bucket):
    return {"count": bucket["count"], "queue_ids": list(bucket["queue_ids"])}


def _with_owner(payload, owner):
    if owner is not None:
        payload["owner"] = owner
    return payload


def serialize_bucket(bucket):
    return _with_owner(_base(bucket), bucket["owner"])


def serialize_export_bucket(bucket):
    payload = _with_owner(_base(bucket), bucket["owner"])
    payload["owner_label"] = bucket["owner"] or "unowned"
    return payload
"""


def baseline_test_handlers() -> str:
    return """import unittest

from queue_api.handlers import QueueRepository, get_queue_summary


ROWS = [
    {"queue_id": "q-1", "owner": "bob"},
    {"queue_id": "q-2", "owner": "alice"},
    {"queue_id": "q-3", "owner": "alice"},
    {"queue_id": "q-4", "owner": None},
]


class HandlerTests(unittest.TestCase):
    def test_default_summary_omits_owner_key_for_unowned_bucket(self):
        payload = get_queue_summary({"include_unowned": True}, QueueRepository(ROWS))
        self.assertNotIn("owner", payload["summary"][-1])

    def test_explicit_null_owner_filter_still_returns_only_unowned_rows(self):
        payload = get_queue_summary({"owner": None, "include_unowned": True}, QueueRepository(ROWS))
        self.assertEqual(payload["summary"], [{"count": 1, "queue_ids": ["q-4"]}])


if __name__ == "__main__":
    unittest.main()
"""


def fixed_test_handlers() -> str:
    return """import unittest

from queue_api.handlers import QueueRepository, get_queue_summary


ROWS = [
    {"queue_id": "q-1", "owner": "bob"},
    {"queue_id": "q-2", "owner": "alice"},
    {"queue_id": "q-3", "owner": "alice"},
    {"queue_id": "q-4", "owner": None},
]


class HandlerTests(unittest.TestCase):
    def test_default_summary_omits_owner_key_for_unowned_bucket(self):
        payload = get_queue_summary({"include_unowned": True}, QueueRepository(ROWS))
        self.assertNotIn("owner", payload["summary"][-1])

    def test_explicit_null_owner_filter_still_returns_only_unowned_rows(self):
        payload = get_queue_summary({"owner": None, "include_unowned": True}, QueueRepository(ROWS))
        self.assertEqual(payload["summary"], [{"count": 1, "queue_ids": ["q-4"]}])

    def test_export_mode_omits_owner_key_for_unowned_bucket(self):
        payload = get_queue_summary({"include_unowned": True, "mode": "export"}, QueueRepository(ROWS))
        self.assertNotIn("owner", payload["summary"][-1])


if __name__ == "__main__":
    unittest.main()
"""


def baseline_test_service() -> str:
    return """import unittest

from queue_api.service import build_summary


class ServiceTests(unittest.TestCase):
    def test_include_unowned_keeps_existing_owner_order(self):
        rows = [
            {"queue_id": "q-1", "owner": "bob"},
            {"queue_id": "q-2", "owner": "alice"},
            {"queue_id": "q-3", "owner": "alice"},
            {"queue_id": "q-4", "owner": None},
        ]
        buckets = build_summary(rows, include_unowned=True)
        self.assertEqual([bucket["owner"] for bucket in buckets], ["bob", "alice", None])

    def test_without_include_unowned_drops_unowned_bucket(self):
        rows = [
            {"queue_id": "q-1", "owner": "bob"},
            {"queue_id": "q-2", "owner": None},
        ]
        buckets = build_summary(rows, include_unowned=False)
        self.assertEqual([bucket["owner"] for bucket in buckets], ["bob"])


if __name__ == "__main__":
    unittest.main()
"""


def fixed_test_service() -> str:
    return """import unittest

from queue_api.service import build_summary


class ServiceTests(unittest.TestCase):
    def test_include_unowned_keeps_existing_owner_order(self):
        rows = [
            {"queue_id": "q-1", "owner": "bob"},
            {"queue_id": "q-2", "owner": "alice"},
            {"queue_id": "q-3", "owner": "alice"},
            {"queue_id": "q-4", "owner": None},
        ]
        buckets = build_summary(rows, include_unowned=True)
        self.assertEqual([bucket["owner"] for bucket in buckets], ["bob", "alice", None])

    def test_without_include_unowned_drops_unowned_bucket(self):
        rows = [
            {"queue_id": "q-1", "owner": "bob"},
            {"queue_id": "q-2", "owner": None},
        ]
        buckets = build_summary(rows, include_unowned=False)
        self.assertEqual([bucket["owner"] for bucket in buckets], ["bob"])

    def test_equal_count_owner_buckets_keep_insertion_order(self):
        rows = [
            {"queue_id": "q-1", "owner": "charlie"},
            {"queue_id": "q-2", "owner": "alice"},
            {"queue_id": "q-3", "owner": "charlie"},
            {"queue_id": "q-4", "owner": "alice"},
            {"queue_id": "q-5", "owner": None},
        ]
        buckets = build_summary(rows, include_unowned=True)
        self.assertEqual([bucket["owner"] for bucket in buckets], ["charlie", "alice", None])


if __name__ == "__main__":
    unittest.main()
"""


def baseline_release_notes(variant_id: str) -> str:
    lines = [
        "# Release Notes",
        "",
        "## Queue summary API",
        "",
        "- follow-up patch pending for owner-aware filter review feedback",
    ]
    if variant_id in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        lines.append("- mobile rollout is still blocked on contract clarification")
    return "\n".join(lines) + "\n"


def fixed_release_notes(variant_id: str) -> str:
    body = [
        "# Release Notes",
        "",
        "## Queue summary API",
        "",
        "- unowned summary buckets now omit `owner` instead of serializing `null`",
        "- request-side owner filter semantics remain unchanged",
        "- `include_unowned=true` appends the unowned bucket after the existing owner buckets without re-sorting them",
    ]
    if variant_id in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        body.append("- mobile compatibility depends on keeping the omit-not-null response contract")
    if variant_id == "v5-recovery-in-thread":
        body.append("- this follows the INC-742 rollback of the earlier `owner: null` hotfix")
    return "\n".join(body) + "\n"


def oracle_replies(variant_id: str) -> str:
    replies = [
        {
            "thread_id": "RTR-101",
            "status": "resolved",
            "body": (
                "Updated queue_api/serializers.py so both response paths omit "
                "the owner key for unowned buckets, and added regression "
                "coverage in tests/test_handlers.py for the export path."
            ),
            "evidence": ["queue_api/serializers.py", "tests/test_handlers.py"],
        },
        {
            "thread_id": "RTR-102",
            "status": "resolved",
            "body": (
                "Adjusted queue_api/service.py to preserve the existing owner "
                "bucket order and append unowned last, with equal-count coverage "
                "added in tests/test_service.py."
            ),
            "evidence": ["queue_api/service.py", "tests/test_service.py"],
        },
        {
            "thread_id": "RTR-103",
            "status": "resolved",
            "body": (
                "Updated docs/release_notes.md with the omit-not-null contract "
                "note and confirmed request-side owner filter semantics are unchanged."
            ),
            "evidence": ["docs/release_notes.md", "review/verification_note.md"],
        },
    ]
    if variant_id == "v4-multi-corpus-objective":
        replies[2]["body"] += " I also referenced release_context/mobile_client_contract.md because that compatibility contract is the current objective."
        replies[2]["evidence"].append("release_context/mobile_client_contract.md")
    if variant_id == "v5-recovery-in-thread":
        replies[2]["body"] += " The reply now calls out the INC-742 rollback so we do not reintroduce the null-owner hotfix."
        replies[2]["evidence"].extend(["release_context/mobile_client_contract.md", "incident_context/inc_742_owner_null_rollback.md"])
    return json.dumps(replies, indent=2, sort_keys=True) + "\n"


def generic_replies() -> str:
    replies = [
        {"thread_id": "RTR-101", "status": "resolved", "body": "fixed", "evidence": []},
        {"thread_id": "RTR-102", "status": "resolved", "body": "addressed", "evidence": []},
        {"thread_id": "RTR-103", "status": "resolved", "body": "done", "evidence": []},
    ]
    return json.dumps(replies, indent=2, sort_keys=True) + "\n"


def verification_note() -> str:
    return "Ran `python3 -m unittest tests.test_handlers tests.test_service`.\n"


HIDDEN_TESTS = """from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


def evaluate(root: Path) -> dict[str, bool]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from queue_api.handlers import QueueRepository, get_queue_summary

    rows = [
        {"queue_id": "q-1", "owner": "bob"},
        {"queue_id": "q-2", "owner": "alice"},
        {"queue_id": "q-3", "owner": "alice"},
        {"queue_id": "q-4", "owner": None},
    ]
    default_payload = get_queue_summary({"include_unowned": True}, QueueRepository(rows))
    export_payload = get_queue_summary({"include_unowned": True, "mode": "export"}, QueueRepository(rows))

    equal_rows = [
        {"queue_id": "q-10", "owner": "charlie"},
        {"queue_id": "q-11", "owner": "alice"},
        {"queue_id": "q-12", "owner": "charlie"},
        {"queue_id": "q-13", "owner": "alice"},
        {"queue_id": "q-14", "owner": None},
    ]
    equal_payload = get_queue_summary({"include_unowned": True}, QueueRepository(equal_rows))
    absent_payload = get_queue_summary({}, QueueRepository(rows))
    null_payload = get_queue_summary({"owner": None, "include_unowned": True}, QueueRepository(rows))
    named_payload = get_queue_summary({"owner": "alice"}, QueueRepository(rows))

    service_text = (root / "queue_api" / "service.py").read_text().lower()

    return {
        "default_omits_owner": "owner" not in default_payload["summary"][-1],
        "export_omits_owner": "owner" not in export_payload["summary"][-1],
        "owner_order_preserved": [bucket.get("owner") for bucket in default_payload["summary"]] == ["bob", "alice", None],
        "equal_count_order_preserved": [bucket.get("owner") for bucket in equal_payload["summary"]] == ["charlie", "alice", None],
        "request_semantics_preserved": (
            [bucket.get("owner") for bucket in absent_payload["summary"]] == ["bob", "alice"]
            and null_payload["summary"] == [{"count": 1, "queue_ids": ["q-4"]}]
            and named_payload["summary"] == [{"count": 2, "owner": "alice", "queue_ids": ["q-2", "q-3"]}]
        ),
        "global_owner_sort_detected": "sorted(" in service_text,
    }


class HiddenContractTests(unittest.TestCase):
    def setUp(self):
        root = Path(os.environ["AGENT_WS"])
        self.diagnostics = evaluate(root)

    def test_default_omits_owner(self):
        self.assertTrue(self.diagnostics["default_omits_owner"])

    def test_export_omits_owner(self):
        self.assertTrue(self.diagnostics["export_omits_owner"])

    def test_owner_order_preserved(self):
        self.assertTrue(self.diagnostics["owner_order_preserved"])

    def test_equal_count_order_preserved(self):
        self.assertTrue(self.diagnostics["equal_count_order_preserved"])

    def test_request_semantics_preserved(self):
        self.assertTrue(self.diagnostics["request_semantics_preserved"])
"""


MILESTONE_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" "$1" <<'PY'
import json
import sys
path, key = sys.argv[1], sys.argv[2]
data = json.load(open(path))
sys.exit(0 if data.get("milestones", {}).get(key, False) else 1)
PY
"""


VARIANT_META = {
    "v1-clean-baseline": {
        "extra_files": {},
        "required_release_note_phrases": [
            "omit `owner` instead of serializing `null`",
            "request-side owner filter semantics remain unchanged",
        ],
        "required_reply_evidence": {
            "RTR-101": ["queue_api/serializers.py", "tests/test_handlers.py"],
            "RTR-102": ["queue_api/service.py", "tests/test_service.py"],
            "RTR-103": ["docs/release_notes.md"],
        },
        "required_reply_phrases": {},
    },
    "v2-noisy-distractor": {
        "extra_files": {
            "repo_evidence/january_readability_memo.md": (
                "# January readability memo\n\n"
                "This older note suggested alphabetical sorting for scanability.\n"
                "It predates the current contract and should not drive the fix.\n"
            ),
        },
        "required_release_note_phrases": [
            "omit `owner` instead of serializing `null`",
            "request-side owner filter semantics remain unchanged",
        ],
        "required_reply_evidence": {
            "RTR-101": ["queue_api/serializers.py", "tests/test_handlers.py"],
            "RTR-102": ["queue_api/service.py", "tests/test_service.py"],
            "RTR-103": ["docs/release_notes.md"],
        },
        "required_reply_phrases": {},
    },
    "v3-dirty-state": {
        "extra_files": {
            "repo_evidence/january_readability_memo.md": (
                "# January readability memo\n\n"
                "This older note suggested alphabetical sorting for scanability.\n"
                "It predates the current contract and should not drive the fix.\n"
            ),
            "repo_evidence/abandoned_patch.md": (
                "# Abandoned patch note\n\n"
                "A prior patch normalized missing owner filters to None and sorted all\n"
                "owner buckets for readability. It was abandoned because it changed\n"
                "request semantics and hid the real response-contract bug.\n"
            ),
        },
        "required_release_note_phrases": [
            "omit `owner` instead of serializing `null`",
            "request-side owner filter semantics remain unchanged",
        ],
        "required_reply_evidence": {
            "RTR-101": ["queue_api/serializers.py", "tests/test_handlers.py"],
            "RTR-102": ["queue_api/service.py", "tests/test_service.py"],
            "RTR-103": ["docs/release_notes.md"],
        },
        "required_reply_phrases": {},
    },
    "v4-multi-corpus-objective": {
        "extra_files": {
            "repo_evidence/january_readability_memo.md": (
                "# January readability memo\n\n"
                "This older note suggested alphabetical sorting for scanability.\n"
                "It predates the current contract and should not drive the fix.\n"
            ),
            "repo_evidence/abandoned_patch.md": (
                "# Abandoned patch note\n\n"
                "A prior patch normalized missing owner filters to None and sorted all\n"
                "owner buckets for readability. It was abandoned because it changed\n"
                "request semantics and hid the real response-contract bug.\n"
            ),
            "release_context/mobile_client_contract.md": (
                "# Mobile client contract\n\n"
                "iOS 7.4 treats an omitted owner key as unassigned, but `owner: null`\n"
                "is parsed as an explicit cleared assignment. Preserve the response\n"
                "contract and do not change request-side owner filter semantics.\n"
            ),
        },
        "required_release_note_phrases": [
            "omit `owner` instead of serializing `null`",
            "request-side owner filter semantics remain unchanged",
            "mobile compatibility depends on keeping the omit-not-null response contract",
        ],
        "required_reply_evidence": {
            "RTR-101": ["queue_api/serializers.py", "tests/test_handlers.py"],
            "RTR-102": ["queue_api/service.py", "tests/test_service.py"],
            "RTR-103": ["docs/release_notes.md", "release_context/mobile_client_contract.md"],
        },
        "required_reply_phrases": {},
    },
    "v5-recovery-in-thread": {
        "extra_files": {
            "repo_evidence/january_readability_memo.md": (
                "# January readability memo\n\n"
                "This older note suggested alphabetical sorting for scanability.\n"
                "It predates the current contract and should not drive the fix.\n"
            ),
            "repo_evidence/abandoned_patch.md": (
                "# Abandoned patch note\n\n"
                "A prior patch normalized missing owner filters to None and sorted all\n"
                "owner buckets for readability. It was abandoned because it changed\n"
                "request semantics and hid the real response-contract bug.\n"
            ),
            "release_context/mobile_client_contract.md": (
                "# Mobile client contract\n\n"
                "iOS 7.4 treats an omitted owner key as unassigned, but `owner: null`\n"
                "is parsed as an explicit cleared assignment. Preserve the response\n"
                "contract and do not change request-side owner filter semantics.\n"
            ),
            "incident_context/inc_742_owner_null_rollback.md": (
                "# INC-742 owner-null rollback\n\n"
                "The prior hotfix serialized `owner: null` for unowned buckets. Mobile\n"
                "parity checks failed, so the patch was rolled back. The recovery path is\n"
                "to omit the key and document the contract clearly.\n"
            ),
        },
        "required_release_note_phrases": [
            "omit `owner` instead of serializing `null`",
            "request-side owner filter semantics remain unchanged",
            "mobile compatibility depends on keeping the omit-not-null response contract",
        ],
        "required_reply_evidence": {
            "RTR-101": ["queue_api/serializers.py", "tests/test_handlers.py"],
            "RTR-102": ["queue_api/service.py", "tests/test_service.py"],
            "RTR-103": [
                "docs/release_notes.md",
                "release_context/mobile_client_contract.md",
                "incident_context/inc_742_owner_null_rollback.md",
            ],
        },
        "required_reply_phrases": {
            "RTR-103": ["INC-742", "rollback"],
        },
        "required_incident_phrase": "INC-742",
    },
}


def write_workspace_bundle(variant_id: str) -> None:
    root = WORKSPACE_ROOT / variant_id
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    write(root / ".scenario_variant", f"{variant_id}\n")
    write(
        root / "AGENTS.md",
        (
            "# Solve The Review Threads\n\n"
            "Read `review/pr_481_threads.json` first. Only unresolved actionable\n"
            "threads should drive the patch. Preserve unrelated work, fix the\n"
            "owner-summary contract bug narrowly, update `docs/release_notes.md`,\n"
            "write `review/reviewer_replies.json`, and record what you ran in\n"
            "`review/verification_note.md`.\n"
        ),
    )
    write(root / "Dockerfile", "FROM python:3.11-slim\nWORKDIR /workspace\n")
    write(root / "queue_api" / "__init__.py", "")
    write(root / "queue_api" / "handlers.py", baseline_handlers())
    write(root / "queue_api" / "service.py", baseline_service())
    write(root / "queue_api" / "serializers.py", baseline_serializers())
    write(root / "tests" / "__init__.py", "")
    write(root / "tests" / "test_handlers.py", baseline_test_handlers())
    write(root / "tests" / "test_service.py", baseline_test_service())
    write(root / "tests" / "locked" / "reply_contract.txt", "do not modify protected fixtures\n")
    write(root / "docs" / "release_notes.md", baseline_release_notes(variant_id))
    write(root / "CHANGELOG.md", "# Changelog\n\n- owner-aware filter work in progress\n")
    write(root / "review" / "pr_481_threads.json", thread_export(variant_id))
    write(root / "review" / "pr_481_patch.diff", "--- a/queue_api\n+++ b/queue_api\n@@ seeded review export only @@\n")
    write(
        root / "review" / "review_summary.md",
        (
            "# Review Summary\n\n"
            "- unresolved actionable: RTR-101, RTR-102, RTR-103\n"
            "- non-actionable: RTR-104 (resolved), RTR-105 (outdated)\n"
        ),
    )
    write(
        root / "artifacts" / "expected_reply_schema.json",
        json.dumps(
            {
                "type": "array",
                "items": {
                    "required": ["thread_id", "status", "body", "evidence"],
                    "properties": {
                        "thread_id": {"type": "string"},
                        "status": {"type": "string"},
                        "body": {"type": "string"},
                        "evidence": {"type": "array"},
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    write(root / "artifacts" / "sample_requests" / "owner_filter_cases.json", json.dumps(BASE_ROWS, indent=2, sort_keys=True) + "\n")
    for relpath, text in VARIANT_META[variant_id]["extra_files"].items():
        write(root / relpath, text)


def write_verifier_data(variant_id: str) -> None:
    variant_root = VERIFIER_DATA_ROOT / variant_id
    if variant_root.exists():
        shutil.rmtree(variant_root)
    variant_root.mkdir(parents=True)

    gold = {
        "variant_id": variant_id,
        "pass_bar": PASS_BAR if (PASS_BAR := 80) else 80,
        "allowed_write_paths": ALLOWED_WRITE_PATHS,
        "actionable_thread_ids": ["RTR-101", "RTR-102", "RTR-103"],
        "ignored_thread_ids": ["RTR-104", "RTR-105"],
        "required_release_note_phrases": VARIANT_META[variant_id]["required_release_note_phrases"],
        "required_reply_evidence": VARIANT_META[variant_id]["required_reply_evidence"],
        "required_reply_phrases": VARIANT_META[variant_id].get("required_reply_phrases", {}),
        "required_regression_test_names": [
            "test_export_mode_omits_owner_key_for_unowned_bucket",
            "test_equal_count_owner_buckets_keep_insertion_order",
        ],
        "required_incident_phrase": VARIANT_META[variant_id].get("required_incident_phrase", ""),
    }
    write(variant_root / "gold_solution.json", json.dumps(gold, indent=2, sort_keys=True) + "\n")

    oracle_root = variant_root / "oracle"
    write(oracle_root / "queue_api" / "handlers.py", baseline_handlers())
    write(oracle_root / "queue_api" / "service.py", fixed_service())
    write(oracle_root / "queue_api" / "serializers.py", fixed_serializers())
    write(oracle_root / "tests" / "test_handlers.py", fixed_test_handlers())
    write(oracle_root / "tests" / "test_service.py", fixed_test_service())
    write(oracle_root / "docs" / "release_notes.md", fixed_release_notes(variant_id))
    write(oracle_root / "review" / "reviewer_replies.json", oracle_replies(variant_id))
    write(oracle_root / "review" / "verification_note.md", verification_note())

    milestone_root = variant_root / "milestones"
    milestone_root.mkdir(parents=True, exist_ok=True)
    for name, key in (
        ("m1_localization.sh", "M1_localization"),
        ("m2_primary_fix.sh", "M2_primary_fix"),
        ("m3_invariants.sh", "M3_invariants"),
        ("m4_functional.sh", "M4_functional"),
        ("m5_e2e.sh", "M5_e2e"),
    ):
        script = milestone_root / name
        write(script, MILESTONE_SCRIPT)
        script.chmod(0o755)


def build_workspace_manifest(variant_id: str) -> None:
    root = WORKSPACE_ROOT / variant_id
    manifest = {
        "variant_id": variant_id,
        "allowed_write_paths": ALLOWED_WRITE_PATHS,
        "file_hashes": {},
        "readonly_tree_hashes": {},
    }
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            manifest["file_hashes"][rel] = sha256_file(path)
    for rel in (
        "review/pr_481_threads.json",
        "review/pr_481_patch.diff",
        "review/review_summary.md",
        "artifacts",
        "tests/locked",
        "repo_evidence",
        "release_context",
        "incident_context",
    ):
        digest = sha256_tree(root, rel)
        if digest:
            manifest["readonly_tree_hashes"][rel] = digest
    write(VERIFIER_DATA_ROOT / variant_id / "workspace_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def build_hidden_tests() -> None:
    shared_root = VERIFIER_DATA_ROOT / "_shared" / "hidden_tests"
    if shared_root.exists():
        shutil.rmtree(shared_root)
    shared_root.mkdir(parents=True)
    write(shared_root / "test_contract_hidden.py", HIDDEN_TESTS)


def overlay_oracle(workspace: Path, variant_id: str) -> None:
    oracle_root = VERIFIER_DATA_ROOT / variant_id / "oracle"
    for path in oracle_root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(oracle_root)
            target = workspace / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def make_shortcut(workspace: Path) -> None:
    write(workspace / "review" / "reviewer_replies.json", generic_replies())
    write(workspace / "review" / "verification_note.md", verification_note())


def score_workspace(workspace: Path, variant_id: str) -> int:
    result_file = workspace / ".." / "verify_result.json"
    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(workspace),
            "VERIFIER_DATA": str(VERIFIER_DATA_ROOT),
            "RESULT_FILE": str(result_file),
            "VARIANT_ID": variant_id,
        }
    )
    subprocess.run([sys.executable, str(SCORER)], env=env, check=True)
    return json.loads(Path(result_file).read_text())["score"]


def build_manifest_lock() -> None:
    lock = {"family_id": "pr-thread-contract-remediation", "schema_version": "cnb55.manifest.v2", "variants": {}}
    hidden_tree = sha256_tree(VERIFIER_DATA_ROOT / "_shared", "hidden_tests")
    for variant_id in VARIANTS:
        with tempfile.TemporaryDirectory(prefix=f"prthread_{variant_id}_") as tmp:
            ws_oracle = Path(tmp) / "oracle"
            shutil.copytree(WORKSPACE_ROOT / variant_id, ws_oracle)
            overlay_oracle(ws_oracle, variant_id)
            oracle_score = score_workspace(ws_oracle, variant_id)

            ws_empty = Path(tmp) / "empty"
            shutil.copytree(WORKSPACE_ROOT / variant_id, ws_empty)
            empty_score = score_workspace(ws_empty, variant_id)

            ws_shortcut = Path(tmp) / "shortcut"
            shutil.copytree(WORKSPACE_ROOT / variant_id, ws_shortcut)
            make_shortcut(ws_shortcut)
            shortcut_score = score_workspace(ws_shortcut, variant_id)

        lock["variants"][variant_id] = {
            "gold_solution_sha256": sha256_file(VERIFIER_DATA_ROOT / variant_id / "gold_solution.json"),
            "workspace_manifest_sha256": sha256_file(VERIFIER_DATA_ROOT / variant_id / "workspace_manifest.json"),
            "oracle_tree_sha256": sha256_tree(VERIFIER_DATA_ROOT / variant_id, "oracle"),
            "hidden_tests_tree_sha256": hidden_tree,
            "observed_oracle_score": oracle_score,
            "observed_empty_score": empty_score,
            "observed_shortcut_score": shortcut_score,
        }
    write(MANIFEST_LOCK, json.dumps(lock, indent=2, sort_keys=True) + "\n")


def main() -> int:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    VERIFIER_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for variant_id in VARIANTS:
        write_workspace_bundle(variant_id)
    build_hidden_tests()
    for variant_id in VARIANTS:
        write_verifier_data(variant_id)
        build_workspace_manifest(variant_id)
    build_manifest_lock()
    print(f"wrote {WORKSPACE_ROOT}")
    print(f"wrote {VERIFIER_DATA_ROOT}")
    print(f"wrote {MANIFEST_LOCK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
