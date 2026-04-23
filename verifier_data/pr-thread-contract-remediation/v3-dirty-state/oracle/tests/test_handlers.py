import unittest

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
