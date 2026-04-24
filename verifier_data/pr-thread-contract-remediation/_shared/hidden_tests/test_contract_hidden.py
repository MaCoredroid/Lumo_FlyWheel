from __future__ import annotations

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
