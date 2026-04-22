#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILY_ROOT = REPO_ROOT / "benchmark_blueprints" / "families" / "workspace-approval-state-threading"
WORKSPACE_ROOT = FAMILY_ROOT / "workspace_bundle"
VERIFIER_ROOT = REPO_ROOT / "verifier_data" / "workspace-approval-state-threading"

VARIANTS = [
    {
        "id": "v1-clean-baseline",
        "label": "V1 clean baseline",
        "filtered_row": {"workspace_id": "ws-legacy-03", "approval_state": "manual_review", "source": "legacy_fallback"},
        "variant_note": "Baseline rollout; prove legacy fallback without extra context.",
        "keywords": ["legacy_fallback", "service", "api", "cli", "workspace-admin-approval-state.png"],
        "extra_readonly": {},
    },
    {
        "id": "v2-noisy-distractor",
        "label": "V2 noisy distractor",
        "filtered_row": {"workspace_id": "ws-legacy-03", "approval_state": "manual_review", "source": "legacy_fallback"},
        "variant_note": "Ignore stale approval_mode snapshots and update the real contract.",
        "keywords": ["stale approval_mode", "legacy_fallback", "service", "api", "cli"],
        "extra_readonly": {
            "seed_data/stale_ops_snapshot.json": json.dumps(
                {
                    "generated_at": "2026-02-01T14:00:00Z",
                    "fields": ["workspace_id", "risk_level", "approval_mode"],
                    "note": "stale export; do not treat as current schema",
                },
                indent=2,
            )
            + "\n",
            "seed_data/stale_memo.md": (
                "# Stale memo\n\n"
                "Older operators still mention `approval_mode`, but the runtime schema moved to "
                "`approval_state`. This memo is stale evidence.\n"
            ),
        },
    },
    {
        "id": "v3-dirty-state",
        "label": "V3 dirty state",
        "filtered_row": {"workspace_id": "ws-legacy-03", "approval_state": "manual_review", "source": "legacy_fallback"},
        "variant_note": "Reject the abandoned frontend-only alias patch and thread the real field.",
        "keywords": ["frontend-only alias patch", "legacy_fallback", "service", "api", "cli"],
        "extra_readonly": {
            "incident_context/abandoned_alias_patch.md": (
                "# Abandoned patch\n\n"
                "A previous local patch proposed rendering `approval_state` from `risk_level` in the "
                "table layer only. It was never shipped because the API and CLI remained stale.\n"
            ),
        },
    },
    {
        "id": "v4-multi-corpus-objective",
        "label": "V4 multi-corpus objective",
        "filtered_row": {"workspace_id": "ws-blocked-02", "approval_state": "blocked", "source": "explicit"},
        "variant_note": "Release-readiness preview must foreground the blocked row, not the legacy fallback row.",
        "keywords": ["release-readiness", "blocked row", "workspace-admin-approval-state.png"],
        "extra_readonly": {
            "release_context/launch_readiness.md": (
                "# Launch readiness objective\n\n"
                "For this cycle the preview artifact shown to operators must foreground the blocked row "
                "`ws-blocked-02` because it is the launch blocker. Keep legacy fallback documented, but "
                "the screenshot contract should highlight blocked review state.\n"
            ),
        },
    },
    {
        "id": "v5-recovery-in-thread",
        "label": "V5 recovery in thread",
        "filtered_row": {"workspace_id": "ws-legacy-03", "approval_state": "manual_review", "source": "legacy_fallback"},
        "variant_note": "Acknowledge the earlier rollback of the risk_level alias hotfix before describing the new fix.",
        "keywords": ["rollback", "risk_level alias", "legacy_fallback"],
        "extra_readonly": {
            "incident_context/prior_hotfix_rollback.md": (
                "# Prior rollback\n\n"
                "The earlier hotfix mapped `approval_state` directly from `risk_level` in the serializer and "
                "frontend. That hotfix was rolled back after operators noticed the CLI export still omitted "
                "the field. The new rollout note must acknowledge the rollback and describe the real fix.\n"
            ),
        },
    },
]

READONLY_ROOTS = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "bin",
    "seed_data",
    "release_context",
    "incident_context",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_tree(root: Path, relpath: str) -> str:
    target = root / relpath
    if not target.exists():
        return "MISSING"
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(b"F")
        digest.update(sha256_file(target).encode())
        return digest.hexdigest()
    for item in sorted(target.rglob("*")):
        rel = item.relative_to(target).as_posix()
        if item.is_dir():
            digest.update(f"D:{rel}\0".encode())
        else:
            digest.update(f"F:{rel}\0".encode())
            digest.update(sha256_file(item).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def write_text(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.chmod(path, mode)


def write_json(path: Path, data: object) -> None:
    write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def seed_rows() -> list[dict[str, str]]:
    return [
        {"workspace_id": "ws-approved-01", "workspace_name": "Northwind", "risk_level": "medium", "approval_state": "approved"},
        {"workspace_id": "ws-blocked-02", "workspace_name": "Redwood", "risk_level": "high", "approval_state": "blocked"},
        {"workspace_id": "ws-legacy-03", "workspace_name": "Legacy Labs", "risk_level": "low"},
    ]


def broken_files(variant: dict[str, object]) -> dict[str, str]:
    filtered = variant["filtered_row"]
    return {
        ".scenario_variant": variant["id"] + "\n",
        "AGENTS.md": (
            "# Workspace approval threading task\n\n"
            "Implement `approval_state` end to end across backend, API, CLI, frontend, config, docs, tests, "
            "preview artifact, and rollout note.\n\n"
            "Writable paths:\n"
            "- backend/**\n"
            "- cli/**\n"
            "- frontend/**\n"
            "- config/defaults.toml\n"
            "- docs/runbooks/workspace-approvals.md\n"
            "- tests/**\n"
            "- artifacts/preview/workspace_admin_capture.json\n"
            "- artifacts/rollout/approval_state_rollout_note.json\n\n"
            "Read-only paths:\n"
            "- seed_data/**\n"
            "- release_context/** when present\n"
            "- incident_context/** when present\n"
            "- AGENTS.md\n"
            "- Dockerfile\n"
            "- bin/run-visible-tests\n"
            "- .scenario_variant\n\n"
            "Do not fabricate `approval_state` in the frontend or alias it from `risk_level`.\n"
            "Run `bin/run-visible-tests` before you stop.\n"
        ),
        "Dockerfile": (
            "FROM python:3.11-slim\n"
            "WORKDIR /workspace\n"
            "COPY . /workspace\n"
            "CMD [\"python3\", \"-m\", \"unittest\", \"discover\", \"-s\", \"tests\", \"-p\", \"test_*.py\"]\n"
        ),
        "bin/run-visible-tests": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "python3 -m unittest discover -s tests -p 'test_*.py'\n"
        ),
        "backend/workspaces/models.py": (
            "from __future__ import annotations\n\n"
            "DEFAULT_RISK_LEVEL = \"medium\"\n"
            "DEFAULT_APPROVAL_STATE = \"manual_review\"\n"
        ),
        "backend/workspaces/service.py": (
            "from __future__ import annotations\n\n"
            "from backend.workspaces.models import DEFAULT_RISK_LEVEL\n\n\n"
            "def normalize_workspace_row(raw_row: dict, default_policy: dict | None = None) -> dict:\n"
            "    normalized = {\n"
            "        \"workspace_id\": raw_row[\"workspace_id\"],\n"
            "        \"workspace_name\": raw_row[\"workspace_name\"],\n"
            "        \"risk_level\": raw_row.get(\"risk_level\", DEFAULT_RISK_LEVEL),\n"
            "    }\n"
            "    if raw_row.get(\"approval_state\"):\n"
            "        normalized[\"approval_state\"] = raw_row[\"approval_state\"]\n"
            "    if raw_row.get(\"approval_mode\"):\n"
            "        normalized[\"approval_mode\"] = raw_row[\"approval_mode\"]\n"
            "    return normalized\n"
        ),
        "backend/api/serializers.py": (
            "from __future__ import annotations\n\n\n"
            "def serialize_workspace(row: dict) -> dict:\n"
            "    payload = {\n"
            "        \"workspace_id\": row[\"workspace_id\"],\n"
            "        \"workspace_name\": row[\"workspace_name\"],\n"
            "        \"risk_level\": row[\"risk_level\"],\n"
            "    }\n"
            "    if row.get(\"approval_state\") and row.get(\"risk_level\") != \"low\":\n"
            "        payload[\"approval_state\"] = row[\"approval_state\"]\n"
            "    return payload\n"
        ),
        "cli/export_workspace.py": (
            "from __future__ import annotations\n\n"
            "import json\n"
            "from pathlib import Path\n\n"
            "from backend.workspaces.service import normalize_workspace_row\n\n"
            "SEED_PATH = Path(__file__).resolve().parents[1] / 'seed_data' / 'mixed_workspaces.json'\n\n\n"
            "def export_workspace_snapshot(rows: list[dict], default_policy: dict | None = None) -> list[dict]:\n"
            "    export_rows = []\n"
            "    for row in rows:\n"
            "        normalized = normalize_workspace_row(row, default_policy=default_policy)\n"
            "        export_rows.append({\n"
            "            'workspace_id': normalized['workspace_id'],\n"
            "            'workspace_name': normalized['workspace_name'],\n"
            "            'risk_level': normalized['risk_level'],\n"
            "        })\n"
            "    return export_rows\n\n\n"
            "def main() -> None:\n"
            "    rows = json.loads(SEED_PATH.read_text())\n"
            "    print(json.dumps(export_workspace_snapshot(rows), indent=2, sort_keys=True))\n\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
        "frontend/src/pages/WorkspaceAdmin.tsx": (
            "import { WorkspaceTable } from '../components/WorkspaceTable';\n\n"
            "export function WorkspaceAdminPage() {\n"
            "  return <WorkspaceTable title=\"Workspace approvals\" />;\n"
            "}\n"
        ),
        "frontend/src/components/WorkspaceTable.tsx": (
            "export const TABLE_COLUMNS = ['workspace', 'risk_level'];\n\n"
            "export function WorkspaceTable(props: { title: string }) {\n"
            "  return `${props.title}: ${TABLE_COLUMNS.join(',')}`;\n"
            "}\n\n"
            "export function renderRiskBadge(row: { risk_level: string }) {\n"
            "  return row.risk_level;\n"
            "}\n"
        ),
        "config/defaults.toml": (
            "[workspace_policy]\n"
            "risk_level = \"medium\"\n"
            "approval_mode = \"manual_review\"\n"
        ),
        "docs/runbooks/workspace-approvals.md": (
            "# Workspace approvals\n\n"
            "Operators currently validate the `risk_level` column and the `approval_mode` config token.\n"
            "The screenshot contract remains `workspace-admin-risk-level.png` until the UI rewrite lands.\n"
        ),
        "artifacts/preview/workspace_admin_capture.json": json.dumps(
            {
                "screenshot_name": "workspace-admin-risk-level.png",
                "columns": ["workspace", "risk_level"],
                "filtered_row": {
                    "workspace_id": filtered["workspace_id"],
                    "approval_state": "MISSING",
                    "source": "unknown",
                },
                "badge_values": ["medium", "high", "low"],
            },
            indent=2,
        )
        + "\n",
        "tests/backend/test_service_and_serializer.py": (
            "import unittest\n\n"
            "from backend.workspaces.service import normalize_workspace_row\n"
            "from backend.api.serializers import serialize_workspace\n\n\n"
            "class WorkspaceContractTests(unittest.TestCase):\n"
            "    def test_risk_level_still_serializes(self) -> None:\n"
            "        row = normalize_workspace_row({'workspace_id': 'ws-1', 'workspace_name': 'One'})\n"
            "        payload = serialize_workspace(row)\n"
            "        self.assertEqual(payload['risk_level'], 'medium')\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "tests/frontend/test_workspace_table_contract.py": (
            "import unittest\n"
            "from pathlib import Path\n\n\n"
            "class WorkspaceTableContractTests(unittest.TestCase):\n"
            "    def test_table_mentions_risk_level(self) -> None:\n"
            "        text = Path('frontend/src/components/WorkspaceTable.tsx').read_text()\n"
            "        self.assertIn('risk_level', text)\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "tests/e2e/test_workspace_admin_preview.py": (
            "import json\n"
            "import unittest\n"
            "from pathlib import Path\n\n\n"
            "class WorkspacePreviewTests(unittest.TestCase):\n"
            "    def test_preview_uses_old_name(self) -> None:\n"
            "        data = json.loads(Path('artifacts/preview/workspace_admin_capture.json').read_text())\n"
            "        self.assertEqual(data['screenshot_name'], 'workspace-admin-risk-level.png')\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "tests/test_mixed_dataset_consistency.py": (
            "import unittest\n\n\n"
            "class MixedDatasetConsistencyTests(unittest.TestCase):\n"
            "    def test_placeholder(self) -> None:\n"
            "        self.assertTrue(True)\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "seed_data/mixed_workspaces.json": json.dumps(seed_rows(), indent=2) + "\n",
    } | variant["extra_readonly"]


def oracle_files(variant: dict[str, object]) -> dict[str, str]:
    filtered = variant["filtered_row"]
    return {
        "backend/workspaces/models.py": (
            "from __future__ import annotations\n\n"
            "DEFAULT_RISK_LEVEL = \"medium\"\n"
            "DEFAULT_APPROVAL_STATE = \"manual_review\"\n"
            "LEGACY_APPROVAL_SOURCE = \"legacy_fallback\"\n"
            "EXPLICIT_APPROVAL_SOURCE = \"explicit\"\n"
        ),
        "backend/workspaces/service.py": (
            "from __future__ import annotations\n\n"
            "from backend.workspaces.models import (\n"
            "    DEFAULT_APPROVAL_STATE,\n"
            "    DEFAULT_RISK_LEVEL,\n"
            "    EXPLICIT_APPROVAL_SOURCE,\n"
            "    LEGACY_APPROVAL_SOURCE,\n"
            ")\n\n\n"
            "def normalize_workspace_row(raw_row: dict, default_policy: dict | None = None) -> dict:\n"
            "    default_state = (default_policy or {}).get('approval_state', DEFAULT_APPROVAL_STATE)\n"
            "    approval_state = raw_row.get('approval_state') or default_state\n"
            "    approval_state_source = EXPLICIT_APPROVAL_SOURCE if raw_row.get('approval_state') else LEGACY_APPROVAL_SOURCE\n"
            "    return {\n"
            "        'workspace_id': raw_row['workspace_id'],\n"
            "        'workspace_name': raw_row['workspace_name'],\n"
            "        'risk_level': raw_row.get('risk_level', DEFAULT_RISK_LEVEL),\n"
            "        'approval_state': approval_state,\n"
            "        'approval_state_source': approval_state_source,\n"
            "    }\n"
        ),
        "backend/api/serializers.py": (
            "from __future__ import annotations\n\n\n"
            "def serialize_workspace(row: dict) -> dict:\n"
            "    return {\n"
            "        'workspace_id': row['workspace_id'],\n"
            "        'workspace_name': row['workspace_name'],\n"
            "        'risk_level': row['risk_level'],\n"
            "        'approval_state': row['approval_state'],\n"
            "        'approval_state_source': row['approval_state_source'],\n"
            "    }\n"
        ),
        "cli/export_workspace.py": (
            "from __future__ import annotations\n\n"
            "import json\n"
            "from pathlib import Path\n\n"
            "from backend.api.serializers import serialize_workspace\n"
            "from backend.workspaces.service import normalize_workspace_row\n\n"
            "SEED_PATH = Path(__file__).resolve().parents[1] / 'seed_data' / 'mixed_workspaces.json'\n"
            "DEFAULT_POLICY = {'approval_state': 'manual_review'}\n\n\n"
            "def export_workspace_snapshot(rows: list[dict], default_policy: dict | None = None) -> list[dict]:\n"
            "    policy = default_policy or DEFAULT_POLICY\n"
            "    return [serialize_workspace(normalize_workspace_row(row, policy)) for row in rows]\n\n\n"
            "def main() -> None:\n"
            "    rows = json.loads(SEED_PATH.read_text())\n"
            "    print(json.dumps(export_workspace_snapshot(rows), indent=2, sort_keys=True))\n\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
        "frontend/src/pages/WorkspaceAdmin.tsx": (
            "import { WorkspaceTable } from '../components/WorkspaceTable';\n\n"
            "export function WorkspaceAdminPage() {\n"
            "  return <WorkspaceTable title=\"Workspace approvals\" filterMode=\"approval_state\" />;\n"
            "}\n"
        ),
        "frontend/src/components/WorkspaceTable.tsx": (
            "export const TABLE_COLUMNS = ['workspace', 'risk_level', 'approval_state'];\n"
            "export const APPROVAL_FALLBACK_BADGE = 'manual_review';\n\n"
            "export function WorkspaceTable(props: { title: string; filterMode: string }) {\n"
            "  return `${props.title}: ${TABLE_COLUMNS.join(',')} filtered-by=${props.filterMode}`;\n"
            "}\n\n"
            "export function renderApprovalStateBadge(row: { approval_state?: string }) {\n"
            "  return row.approval_state || APPROVAL_FALLBACK_BADGE;\n"
            "}\n\n"
            "export const APPROVAL_STATE_COLUMN_LABEL = 'Approval state';\n"
        ),
        "config/defaults.toml": (
            "[workspace_policy]\n"
            "risk_level = \"medium\"\n"
            "approval_state = \"manual_review\"\n"
        ),
        "docs/runbooks/workspace-approvals.md": (
            "# Workspace approvals\n\n"
            "Operators now validate the `approval_state` column end to end across service, API, CLI, and UI.\n\n"
            "Legacy rows without `approval_state` render as `manual_review` until the backfill completes. "
            "The UI must show `legacy_fallback` provenance for those rows.\n\n"
            f"Variant note: {variant['variant_note']}\n\n"
            "The preview artifact is `workspace-admin-approval-state.png`.\n"
        ),
        "artifacts/preview/workspace_admin_capture.json": json.dumps(
            {
                "screenshot_name": "workspace-admin-approval-state.png",
                "columns": ["workspace", "risk_level", "approval_state"],
                "filtered_row": filtered,
                "badge_values": ["approved", "blocked", "manual_review"],
                "notes": variant["variant_note"],
            },
            indent=2,
        )
        + "\n",
        "artifacts/rollout/approval_state_rollout_note.json": json.dumps(
            {
                "schema_version": "cnb55.rollout_note.v1",
                "legacy_row_fallback": "Rows missing approval_state render as manual_review with legacy_fallback provenance until the backfill completes.",
                "consistency_surfaces": ["service", "api", "cli", "frontend", "preview"],
                "screenshot_name": "workspace-admin-approval-state.png",
                "variant_notes": variant["variant_note"],
            },
            indent=2,
        )
        + "\n",
        "tests/backend/test_service_and_serializer.py": (
            "import json\n"
            "import unittest\n"
            "from pathlib import Path\n\n"
            "from backend.api.serializers import serialize_workspace\n"
            "from backend.workspaces.service import normalize_workspace_row\n\n\n"
            "class WorkspaceContractTests(unittest.TestCase):\n"
            "    def test_legacy_row_fallback_and_serializer(self) -> None:\n"
            "        rows = json.loads(Path('seed_data/mixed_workspaces.json').read_text())\n"
            "        legacy = normalize_workspace_row(rows[2], {'approval_state': 'manual_review'})\n"
            "        self.assertEqual(legacy['approval_state'], 'manual_review')\n"
            "        self.assertEqual(legacy['approval_state_source'], 'legacy_fallback')\n"
            "        payload = serialize_workspace(legacy)\n"
            "        self.assertEqual(payload['approval_state'], 'manual_review')\n"
            "        self.assertEqual(payload['approval_state_source'], 'legacy_fallback')\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "tests/frontend/test_workspace_table_contract.py": (
            "import unittest\n"
            "from pathlib import Path\n\n\n"
            "class WorkspaceTableContractTests(unittest.TestCase):\n"
            "    def test_table_mentions_approval_state_and_fallback(self) -> None:\n"
            "        text = Path('frontend/src/components/WorkspaceTable.tsx').read_text()\n"
            "        self.assertIn('approval_state', text)\n"
            "        self.assertIn('Approval state', text)\n"
            "        self.assertIn('manual_review', text)\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "tests/e2e/test_workspace_admin_preview.py": (
            "import json\n"
            "import unittest\n"
            "from pathlib import Path\n\n\n"
            "class WorkspacePreviewTests(unittest.TestCase):\n"
            "    def test_preview_matches_current_contract(self) -> None:\n"
            "        data = json.loads(Path('artifacts/preview/workspace_admin_capture.json').read_text())\n"
            "        self.assertEqual(data['screenshot_name'], 'workspace-admin-approval-state.png')\n"
            "        self.assertIn('approval_state', data['columns'])\n"
            f"        self.assertEqual(data['filtered_row']['workspace_id'], '{filtered['workspace_id']}')\n"
            f"        self.assertEqual(data['filtered_row']['approval_state'], '{filtered['approval_state']}')\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "tests/test_mixed_dataset_consistency.py": (
            "import json\n"
            "import unittest\n"
            "from pathlib import Path\n\n"
            "from backend.api.serializers import serialize_workspace\n"
            "from backend.workspaces.service import normalize_workspace_row\n"
            "from cli.export_workspace import export_workspace_snapshot\n\n\n"
            "class MixedDatasetConsistencyTests(unittest.TestCase):\n"
            "    def test_service_api_cli_agree(self) -> None:\n"
            "        rows = json.loads(Path('seed_data/mixed_workspaces.json').read_text())\n"
            "        normalized = [normalize_workspace_row(row, {'approval_state': 'manual_review'}) for row in rows]\n"
            "        serialized = [serialize_workspace(row) for row in normalized]\n"
            "        cli_rows = export_workspace_snapshot(rows, {'approval_state': 'manual_review'})\n"
            "        self.assertEqual(serialized, cli_rows)\n"
            "        legacy = serialized[2]\n"
            "        self.assertEqual(legacy['approval_state'], 'manual_review')\n"
            "        self.assertEqual(legacy['approval_state_source'], 'legacy_fallback')\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
    }


def hidden_test_text() -> str:
    return (
        "from __future__ import annotations\n\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import importlib.util\n\n\n"
        "def _load_module(name: str, path: Path):\n"
        "    spec = importlib.util.spec_from_file_location(name, path)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    assert spec.loader is not None\n"
        "    spec.loader.exec_module(module)\n"
        "    return module\n\n\n"
        "def main() -> None:\n"
        "    agent_ws = Path(os.environ['AGENT_WS'])\n"
        "    if str(agent_ws) not in sys.path:\n"
        "        sys.path.insert(0, str(agent_ws))\n"
        "    gold = json.loads(Path(os.environ['GOLD_FILE']).read_text())\n"
        "    rows = json.loads((agent_ws / 'seed_data' / 'mixed_workspaces.json').read_text())\n"
        "    service = _load_module('service_mod', agent_ws / 'backend' / 'workspaces' / 'service.py')\n"
        "    serializers = _load_module('serializers_mod', agent_ws / 'backend' / 'api' / 'serializers.py')\n"
        "    cli = _load_module('cli_mod', agent_ws / 'cli' / 'export_workspace.py')\n"
        "    normalized = [service.normalize_workspace_row(row, {'approval_state': 'manual_review'}) for row in rows]\n"
        "    serialized = [serializers.serialize_workspace(row) for row in normalized]\n"
        "    assert cli.export_workspace_snapshot(rows, {'approval_state': 'manual_review'}) == serialized\n"
        "    legacy = serialized[2]\n"
        "    assert legacy['approval_state'] == 'manual_review'\n"
        "    assert legacy['approval_state_source'] == 'legacy_fallback'\n"
        "    preview = json.loads((agent_ws / 'artifacts' / 'preview' / 'workspace_admin_capture.json').read_text())\n"
        "    assert preview['filtered_row']['workspace_id'] == gold['preview_filtered_row']['workspace_id']\n"
        "    assert preview['filtered_row']['approval_state'] == gold['preview_filtered_row']['approval_state']\n\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


def milestone_script(slot_key: str) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "RESULT_FILE=\"${RESULT_FILE:?RESULT_FILE must be set}\"\n"
        "python3 - \"$RESULT_FILE\" <<'PY'\n"
        "import json, pathlib, sys\n"
        "data = json.loads(pathlib.Path(sys.argv[1]).read_text())\n"
        f"value = data.get('milestones', {{}}).get('{slot_key}', False)\n"
        "raise SystemExit(0 if value else 1)\n"
        "PY\n"
    )


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def file_list(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def build_variant(variant: dict[str, object]) -> None:
    workspace = WORKSPACE_ROOT / variant["id"]
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    for rel, text in broken_files(variant).items():
        mode = 0o755 if rel == "bin/run-visible-tests" else 0o644
        write_text(workspace / rel, text, mode=mode)

    verifier_variant_root = VERIFIER_ROOT / variant["id"]
    if verifier_variant_root.exists():
        shutil.rmtree(verifier_variant_root)
    oracle_root = verifier_variant_root / "oracle"
    copy_tree(workspace, oracle_root)
    for rel, text in oracle_files(variant).items():
        write_text(oracle_root / rel, text, mode=0o755 if rel == "bin/run-visible-tests" else 0o644)

    readonly_hashes = {
        rel: sha256_tree(workspace, rel)
        for rel in READONLY_ROOTS
        if (workspace / rel).exists()
    }

    gold = {
        "variant_id": variant["id"],
        "pass_bar": 40,
        "default_approval_state": "manual_review",
        "legacy_row_id": "ws-legacy-03",
        "preview_filtered_row": variant["filtered_row"],
        "rollout_note_keywords": variant["keywords"],
        "readonly_tree_hashes": readonly_hashes,
        "visible_test_sha256": sha256_file(oracle_root / "tests" / "test_mixed_dataset_consistency.py"),
    }
    write_json(verifier_variant_root / "gold_ranking.json", gold)
    write_json(
        verifier_variant_root / "workspace_manifest.json",
        {
            "variant_id": variant["id"],
            "files": file_list(workspace),
            "readonly_tree_hashes": readonly_hashes,
            "visible_test_sha256": gold["visible_test_sha256"],
        },
    )
    write_text(verifier_variant_root / "hidden_tests" / "test_workspace_approval_state.py", hidden_test_text())
    for key, filename in [
        ("M1_localization", "m1_localization.sh"),
        ("M2_primary_fix", "m2_primary_fix.sh"),
        ("M3_invariants", "m3_invariants.sh"),
        ("M4_functional", "m4_functional.sh"),
        ("M5_e2e", "m5_e2e.sh"),
    ]:
        write_text(verifier_variant_root / "milestones" / filename, milestone_script(key), mode=0o755)


def write_manifest_lock() -> None:
    data = {
        "family_id": "workspace-approval-state-threading",
        "variants": [],
    }
    for variant in VARIANTS:
        workspace = WORKSPACE_ROOT / variant["id"]
        data["variants"].append(
            {
                "variant_id": variant["id"],
                "tree_hash": sha256_tree(WORKSPACE_ROOT, variant["id"]),
                "file_count": len(file_list(workspace)),
            }
        )
    write_json(FAMILY_ROOT / "manifest.lock.json", data)


def main() -> None:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    VERIFIER_ROOT.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        build_variant(variant)
    write_manifest_lock()


if __name__ == "__main__":
    main()
