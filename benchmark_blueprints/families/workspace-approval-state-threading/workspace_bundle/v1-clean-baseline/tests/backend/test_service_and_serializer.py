import unittest

from backend.workspaces.service import normalize_workspace_row
from backend.api.serializers import serialize_workspace


class WorkspaceContractTests(unittest.TestCase):
    def test_risk_level_still_serializes(self) -> None:
        row = normalize_workspace_row({'workspace_id': 'ws-1', 'workspace_name': 'One'})
        payload = serialize_workspace(row)
        self.assertEqual(payload['risk_level'], 'medium')


if __name__ == '__main__':
    unittest.main()
