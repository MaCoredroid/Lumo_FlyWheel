import unittest

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
