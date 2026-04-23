#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILY_ROOT = REPO_ROOT / "benchmark_blueprints" / "families" / "codex-provider-rollover"
WORKSPACE_ROOT = FAMILY_ROOT / "workspace_bundle"
VERIFIER_ROOT = REPO_ROOT / "verifier_data" / "codex-provider-rollover"
SCORER = REPO_ROOT / "verifiers" / "codex-provider-rollover" / "score_provider_rollover.py"
VERIFY_SH = REPO_ROOT / "verifiers" / "codex-provider-rollover" / "verify.sh"
LOCK_PATH = FAMILY_ROOT / "manifest.lock.json"

EXPECTED_PROVIDER = "responses_proxy"
EXPECTED_BASE_URL = "http://127.0.0.1:11434/v1/responses"
EXPECTED_ENV_KEY = "OPENAI_API_KEY"
LEGACY_PROVIDER = "legacy_vllm"

VARIANTS = [
    {
        "id": "v1-clean-baseline",
        "variant_note": "Repair the maintained proxy-backed Responses profile and exact follow-up chaining.",
        "extra_workspace": {},
        "required_doc_keywords": [
            "responses_proxy",
            "proxy-backed Responses path",
            "previous_response_id",
        ],
        "variant_ceiling": {"name": "docs_unaligned", "cap": 40},
    },
    {
        "id": "v2-noisy-distractor",
        "variant_note": "Ignore the stale canary path and keep the canonical maintenance profile selected.",
        "extra_workspace": {
            "release_context/stale_canary_note.md": (
                "# Stale canary note\n\n"
                "Older rehearsal notes reference `maintenance_canary`, but the supported maintenance profile "
                "for this family remains `responses_proxy`.\n"
            ),
        },
        "required_doc_keywords": [
            "responses_proxy",
            "maintenance_canary is not the maintained default",
            "previous_response_id",
        ],
        "variant_ceiling": {"name": "canary_anchoring", "cap": 35},
    },
    {
        "id": "v3-dirty-state",
        "variant_note": "Do not regenerate the entire config from the stale template; preserve the local tuning block.",
        "extra_workspace": {
            "serving_maintenance/templates/legacy_profile_template.toml": (
                "# legacy_profile_template\n"
                "provider = \"legacy_vllm\"\n"
                "model = \"gpt-5.4\"\n"
            ),
            "serving_maintenance/scripts/rewrite_profile_from_template.py": (
                "from pathlib import Path\n\n"
                "TEMPLATE = Path('serving_maintenance/templates/legacy_profile_template.toml')\n"
                "CONFIG = Path('serving_maintenance/.codex/config.toml')\n\n"
                "def main() -> None:\n"
                "    CONFIG.write_text('# generated from template\\n' + TEMPLATE.read_text())\n\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            ),
        },
        "required_doc_keywords": [
            "preserve the local tuning block exactly",
            "responses_proxy",
            "previous_response_id",
        ],
        "variant_ceiling": {"name": "template_overwrite", "cap": 30},
    },
    {
        "id": "v4-multi-corpus-objective",
        "variant_note": "Align the fix to the maintenance profile called out in release context, not just a generic smoke patch.",
        "extra_workspace": {
            "release_context/maintenance_window.md": (
                "# Maintenance window\n\n"
                "This cutover is specifically for the `maintenance-responses` profile used during proxy-backed "
                "Responses maintenance. Docs must reference that profile and the proxy route together.\n"
            ),
            "release_context/operator_ticket.md": (
                "# Operator ticket\n\n"
                "Turn-two retrieval broke in the maintenance profile even though one-turn smoke looked healthy. "
                "The rollout note must explain why exact `previous_response_id` continuity is part of the fix.\n"
            ),
        },
        "required_doc_keywords": [
            "maintenance-responses",
            "proxy-backed Responses path",
            "previous_response_id",
        ],
        "variant_ceiling": {"name": "maintenance_profile_drift", "cap": 45},
    },
    {
        "id": "v5-recovery-in-thread",
        "variant_note": "Acknowledge the earlier rollback caused by `store = false` and incomplete follow-up validation.",
        "extra_workspace": {
            "release_context/maintenance_window.md": (
                "# Maintenance window\n\n"
                "This remains the `maintenance-responses` profile repair path. Do not regress back to the direct "
                "endpoint during rollback recovery.\n"
            ),
            "incident_context/store_false_rollback.md": (
                "# Rollback note\n\n"
                "A previous hotfix switched the selected provider but left `store = false` and did not verify "
                "exact follow-up chaining. The rollback note must explicitly call that out.\n"
            ),
        },
        "required_doc_keywords": [
            "rollback",
            "store = true",
            "maintenance-responses",
            "previous_response_id",
        ],
        "variant_ceiling": {"name": "rollback_blind", "cap": 35},
    },
]

READONLY_ROOTS = [
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "bin",
    "tests",
    "serving_maintenance/fixtures",
    "release_context",
    "incident_context",
    "serving_maintenance/templates",
    "serving_maintenance/scripts/rewrite_profile_from_template.py",
]

ALLOWED_WRITES = [
    "serving_maintenance/.codex/config.toml",
    "serving_maintenance/scripts/smoke_responses_profile.py",
    "serving_maintenance/docs/provider_rollover.md",
    "serving_maintenance/docs/smoke.md",
]

PRESERVE_BLOCK = (
    "# Preserve this tuning block byte-for-byte.\n"
    "max_output_tokens = 6400\n"
    "reasoning_summary = \"auto\"\n"
    "tool_retry_budget = 3\n"
    "proxy_read_timeout_ms = 9000\n"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def agents_md(variant: dict[str, object]) -> str:
    return (
        "# Codex Provider Rollover\n\n"
        f"{variant['variant_note']}\n\n"
        "Writable paths:\n"
        "- serving_maintenance/.codex/config.toml\n"
        "- serving_maintenance/scripts/smoke_responses_profile.py\n"
        "- serving_maintenance/docs/provider_rollover.md\n"
        "- serving_maintenance/docs/smoke.md\n\n"
        "Read-only paths:\n"
        "- serving_maintenance/fixtures/**\n"
        "- tests/**\n"
        "- release_context/** when present\n"
        "- incident_context/** when present\n"
        "- serving_maintenance/templates/** when present\n"
        "- AGENTS.md\n"
        "- Dockerfile\n"
        "- .scenario_variant\n\n"
        "Do not rewrite the whole TOML file from a template. Preserve the local tuning block exactly.\n"
        "Run `bin/run-visible-tests` before you stop.\n"
    )


def broken_config_text(include_canary: bool) -> str:
    canary = ""
    if include_canary:
        canary = (
            "\n[model_providers.maintenance_canary]\n"
            "name = \"Maintenance Canary\"\n"
            "base_url = \"http://127.0.0.1:11435/v1/responses\"\n"
            "env_key = \"OPENAI_API_KEY\"\n"
            "wire_api = \"responses\"\n"
            "store = true\n"
        )
    return (
        "# Local maintenance profile after provider rollover.\n"
        "# Keep the local tuning block below exactly as written.\n\n"
        f"provider = \"{LEGACY_PROVIDER}\"\n"
        "model = \"gpt-5.4\"\n"
        "model_reasoning_effort = \"high\"\n"
        "model_verbosity = \"low\"\n"
        "approval_policy = \"never\"\n"
        "sandbox_mode = \"danger-full-access\"\n"
        "profile = \"maintenance-responses\"\n\n"
        f"[model_providers.{LEGACY_PROVIDER}]\n"
        "name = \"Legacy direct vLLM\"\n"
        "base_url = \"http://127.0.0.1:8001/v1\"\n"
        "env_key = \"VLLM_API_KEY\"\n"
        "wire_api = \"responses\"\n"
        "store = false\n\n"
        f"[model_providers.{EXPECTED_PROVIDER}]\n"
        "name = \"Responses proxy\"\n"
        f"base_url = \"{EXPECTED_BASE_URL}\"\n"
        f"env_key = \"{EXPECTED_ENV_KEY}\"\n"
        "wire_api = \"responses\"\n"
        "store = true\n"
        f"{canary}\n"
        f"{PRESERVE_BLOCK}"
    )


def fixed_config_text(include_canary: bool) -> str:
    return broken_config_text(include_canary).replace(
        f"provider = \"{LEGACY_PROVIDER}\"",
        f"provider = \"{EXPECTED_PROVIDER}\"",
        1,
    )


def broken_smoke_script() -> str:
    return (
        "#!/usr/bin/env python3\n"
        "from __future__ import annotations\n\n"
        "import argparse\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "EXPECTED_PROVIDER = \"responses_proxy\"\n"
        "EXPECTED_BASE_URL = \"http://127.0.0.1:11434/v1/responses\"\n\n"
        "def load_json(path: Path) -> dict:\n"
        "    return json.loads(path.read_text())\n\n"
        "def parse_scalar(raw: str):\n"
        "    text = raw.strip()\n"
        "    if text in {'true', 'false'}:\n"
        "        return text == 'true'\n"
        "    if text.startswith('\"') and text.endswith('\"'):\n"
        "        return text[1:-1]\n"
        "    try:\n"
        "        return int(text)\n"
        "    except ValueError:\n"
        "        return text\n\n"
        "def load_config(path: Path) -> dict:\n"
        "    data = {}\n"
        "    current = data\n"
        "    for raw in path.read_text().splitlines():\n"
        "        line = raw.strip()\n"
        "        if not line or line.startswith('#'):\n"
        "            continue\n"
        "        if line.startswith('[') and line.endswith(']'):\n"
        "            current = data\n"
        "            for part in line[1:-1].split('.'):\n"
        "                current = current.setdefault(part, {})\n"
        "            continue\n"
        "        if '=' not in line:\n"
        "            continue\n"
        "        key, value = line.split('=', 1)\n"
        "        current[key.strip()] = parse_scalar(value)\n"
        "    return data\n\n"
        "def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:\n"
        "    config = load_config(config_path)\n"
        "    selected = config.get('provider')\n"
        "    providers = config.get('model_providers', {})\n"
        "    turn1 = load_json(turn1_path)\n"
        "    turn2 = load_json(turn2_path)\n"
        "    return {\n"
        "        'selected_provider': selected,\n"
        "        'base_url': providers.get(selected, {}).get('base_url'),\n"
        "        'turn1_id': turn1.get('response', {}).get('id'),\n"
        "        'observed_previous_response_id': turn2.get('request', {}).get('previous_response_id'),\n"
        "        'continuity_ok': bool(turn2.get('request', {}).get('previous_response_id')),\n"
        "        'store_ok': True,\n"
        "        'status': 'ok',\n"
        "    }\n\n"
        "def main() -> int:\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--config', type=Path, required=True)\n"
        "    parser.add_argument('--turn1', type=Path, required=True)\n"
        "    parser.add_argument('--turn2', type=Path, required=True)\n"
        "    args = parser.parse_args()\n"
        "    print(json.dumps(run_smoke(args.config, args.turn1, args.turn2), indent=2, sort_keys=True))\n"
        "    return 0\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )


def fixed_smoke_script() -> str:
    return (
        "#!/usr/bin/env python3\n"
        "from __future__ import annotations\n\n"
        "import argparse\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "EXPECTED_PROVIDER = \"responses_proxy\"\n"
        "EXPECTED_BASE_URL = \"http://127.0.0.1:11434/v1/responses\"\n"
        "EXPECTED_ENV_KEY = \"OPENAI_API_KEY\"\n\n"
        "class SmokeFailure(RuntimeError):\n"
        "    pass\n\n"
        "def load_json(path: Path) -> dict:\n"
        "    return json.loads(path.read_text())\n\n"
        "def parse_scalar(raw: str):\n"
        "    text = raw.strip()\n"
        "    if text in {'true', 'false'}:\n"
        "        return text == 'true'\n"
        "    if text.startswith('\"') and text.endswith('\"'):\n"
        "        return text[1:-1]\n"
        "    try:\n"
        "        return int(text)\n"
        "    except ValueError:\n"
        "        return text\n\n"
        "def parse_config(path: Path) -> dict:\n"
        "    data = {}\n"
        "    current = data\n"
        "    for raw in path.read_text().splitlines():\n"
        "        line = raw.strip()\n"
        "        if not line or line.startswith('#'):\n"
        "            continue\n"
        "        if line.startswith('[') and line.endswith(']'):\n"
        "            current = data\n"
        "            for part in line[1:-1].split('.'):\n"
        "                current = current.setdefault(part, {})\n"
        "            continue\n"
        "        if '=' not in line:\n"
        "            continue\n"
        "        key, value = line.split('=', 1)\n"
        "        current[key.strip()] = parse_scalar(value)\n"
        "    return data\n\n"
        "def load_config(path: Path) -> tuple[str, dict]:\n"
        "    data = parse_config(path)\n"
        "    selected = data.get('provider')\n"
        "    providers = data.get('model_providers', {})\n"
        "    if selected != EXPECTED_PROVIDER:\n"
        "        raise SmokeFailure(f'selected provider must be {EXPECTED_PROVIDER}, got {selected}')\n"
        "    stanza = providers.get(selected, {})\n"
        "    if stanza.get('base_url') != EXPECTED_BASE_URL:\n"
        "        raise SmokeFailure('selected provider must use the proxy-backed Responses path')\n"
        "    if stanza.get('wire_api') != 'responses':\n"
        "        raise SmokeFailure('selected provider must use wire_api=responses')\n"
        "    if stanza.get('env_key') != EXPECTED_ENV_KEY:\n"
        "        raise SmokeFailure('selected provider must use OPENAI_API_KEY')\n"
        "    if stanza.get('store') is not True:\n"
        "        raise SmokeFailure('selected provider must keep store = true')\n"
        "    return selected, stanza\n\n"
        "def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:\n"
        "    selected, stanza = load_config(config_path)\n"
        "    turn1 = load_json(turn1_path)\n"
        "    turn2 = load_json(turn2_path)\n"
        "    turn1_id = turn1.get('response', {}).get('id')\n"
        "    observed_previous = turn2.get('request', {}).get('previous_response_id')\n"
        "    if observed_previous != turn1_id:\n"
        "        raise SmokeFailure('turn two must use exact previous_response_id continuity')\n"
        "    if turn2.get('request', {}).get('provider') != selected:\n"
        "        raise SmokeFailure('turn two must use the selected provider id')\n"
        "    if turn2.get('request', {}).get('base_url') != stanza.get('base_url'):\n"
        "        raise SmokeFailure('turn two must use the selected provider base_url')\n"
        "    if turn2.get('request', {}).get('store') is not True:\n"
        "        raise SmokeFailure('turn two must keep store = true')\n"
        "    response_previous = turn2.get('response', {}).get('previous_response_id')\n"
        "    if response_previous not in (None, turn1_id):\n"
        "        raise SmokeFailure('turn two response must preserve previous_response_id continuity')\n"
        "    return {\n"
        "        'selected_provider': selected,\n"
        "        'base_url': stanza.get('base_url'),\n"
        "        'expected_previous_response_id': turn1_id,\n"
        "        'observed_previous_response_id': observed_previous,\n"
        "        'continuity_ok': True,\n"
        "        'store_ok': True,\n"
        "        'status': 'ok',\n"
        "    }\n\n"
        "def main() -> int:\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--config', type=Path, required=True)\n"
        "    parser.add_argument('--turn1', type=Path, required=True)\n"
        "    parser.add_argument('--turn2', type=Path, required=True)\n"
        "    args = parser.parse_args()\n"
        "    try:\n"
        "        payload = run_smoke(args.config, args.turn1, args.turn2)\n"
        "    except SmokeFailure as exc:\n"
        "        print(str(exc), flush=True)\n"
        "        return 1\n"
        "    print(json.dumps(payload, indent=2, sort_keys=True))\n"
        "    return 0\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )


def broken_provider_doc(variant: dict[str, object]) -> str:
    extra = ""
    if variant["id"] == "v2-noisy-distractor":
        extra = "\nThe rehearsal canary note still references `maintenance_canary`.\n"
    elif variant["id"] == "v3-dirty-state":
        extra = "\nAn old template helper still exists, but this doc does not say how to preserve local tuning.\n"
    elif variant["id"] in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        extra = "\nOperators still describe the fix as a generic smoke repair rather than the maintenance-responses profile.\n"
    return (
        "# Provider rollover\n\n"
        f"Current profile still points at `{LEGACY_PROVIDER}` and the direct endpoint `http://127.0.0.1:8001/v1`.\n"
        "Update the file before the next maintenance run.\n"
        f"{extra}"
    )


def fixed_provider_doc(variant: dict[str, object]) -> str:
    lines = [
        "# Provider rollover",
        "",
        f"Select `{EXPECTED_PROVIDER}` as the maintained default provider.",
        f"It must point at the proxy-backed Responses path `{EXPECTED_BASE_URL}`.",
        "Do not select the legacy direct endpoint.",
    ]
    if variant["id"] == "v2-noisy-distractor":
        lines.append("`maintenance_canary` is not the maintained default.")
    if variant["id"] == "v3-dirty-state":
        lines.append("Preserve the local tuning block exactly instead of regenerating the file from a template.")
    if variant["id"] in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        lines.append("This repair applies to the `maintenance-responses` profile called out in release context.")
        lines.append("Keep the docs grounded in the proxy-backed Responses path.")
    if variant["id"] == "v5-recovery-in-thread":
        lines.append("A rollback already occurred because the earlier hotfix left `store = true` unchecked in follow-up retrieval.")
        lines.append("Call out the rollback explicitly so the same failure mode does not recur.")
    return "\n".join(lines) + "\n"


def broken_smoke_doc() -> str:
    return (
        "# Smoke procedure\n\n"
        "1. Run the first maintenance turn.\n"
        "2. Check that the response id exists.\n"
        "3. Stop after the first success.\n"
    )


def fixed_smoke_doc() -> str:
    return (
        "# Smoke procedure\n\n"
        "1. Run the maintenance profile with `responses_proxy`.\n"
        f"2. Confirm the selected provider uses `{EXPECTED_BASE_URL}`.\n"
        "3. Run a second turn and require exact `previous_response_id` continuity.\n"
        "4. Confirm the follow-up request keeps `store = true`.\n"
    )


def turn_fixtures() -> dict[str, str]:
    turn1 = {
        "request": {
            "provider": EXPECTED_PROVIDER,
            "base_url": EXPECTED_BASE_URL,
            "store": True,
        },
        "response": {
            "id": "resp_turn_001",
            "status": "completed",
        },
    }
    turn2_ok = {
        "request": {
            "provider": EXPECTED_PROVIDER,
            "base_url": EXPECTED_BASE_URL,
            "store": True,
            "previous_response_id": "resp_turn_001",
        },
        "response": {
            "id": "resp_turn_002",
            "status": "completed",
            "previous_response_id": "resp_turn_001",
        },
    }
    turn2_missing_store = {
        "request": {
            "provider": EXPECTED_PROVIDER,
            "base_url": EXPECTED_BASE_URL,
            "store": False,
            "previous_response_id": "resp_turn_001",
        },
        "response": {
            "id": "resp_turn_002",
            "status": "failed",
            "error": "turn two did not persist follow-up state",
        },
    }
    return {
        "serving_maintenance/fixtures/http/turn1_ok.json": json.dumps(turn1, indent=2, sort_keys=True) + "\n",
        "serving_maintenance/fixtures/http/turn2_ok.json": json.dumps(turn2_ok, indent=2, sort_keys=True) + "\n",
        "serving_maintenance/fixtures/http/turn2_missing_store.json": json.dumps(turn2_missing_store, indent=2, sort_keys=True) + "\n",
    }


def hidden_negative_fixtures() -> dict[str, dict[str, object]]:
    return {
        "turn2_wrong_previous.json": {
            "request": {
                "provider": EXPECTED_PROVIDER,
                "base_url": EXPECTED_BASE_URL,
                "store": True,
                "previous_response_id": "resp_wrong_999",
            },
            "response": {"id": "resp_turn_002", "status": "completed"},
        },
        "turn2_wrong_provider.json": {
            "request": {
                "provider": LEGACY_PROVIDER,
                "base_url": "http://127.0.0.1:8001/v1",
                "store": True,
                "previous_response_id": "resp_turn_001",
            },
            "response": {"id": "resp_turn_002", "status": "completed"},
        },
        "turn2_missing_store.json": {
            "request": {
                "provider": EXPECTED_PROVIDER,
                "base_url": EXPECTED_BASE_URL,
                "store": False,
                "previous_response_id": "resp_turn_001",
            },
            "response": {"id": "resp_turn_002", "status": "completed"},
        },
    }


def visible_tests() -> dict[str, str]:
    return {
        "tests/test_config_profile.py": (
            "from __future__ import annotations\n\n"
            "import unittest\n"
            "from pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "CONFIG = ROOT / 'serving_maintenance' / '.codex' / 'config.toml'\n"
            "EXPECTED_PROVIDER = 'responses_proxy'\n"
            "EXPECTED_BASE_URL = 'http://127.0.0.1:11434/v1/responses'\n\n"
            "def parse_scalar(raw: str):\n"
            "    text = raw.strip()\n"
            "    if text in {'true', 'false'}:\n"
            "        return text == 'true'\n"
            "    if text.startswith('\"') and text.endswith('\"'):\n"
            "        return text[1:-1]\n"
            "    try:\n"
            "        return int(text)\n"
            "    except ValueError:\n"
            "        return text\n\n"
            "def load_config() -> dict:\n"
            "    data = {}\n"
            "    current = data\n"
            "    for raw in CONFIG.read_text().splitlines():\n"
            "        line = raw.strip()\n"
            "        if not line or line.startswith('#'):\n"
            "            continue\n"
            "        if line.startswith('[') and line.endswith(']'):\n"
            "            current = data\n"
            "            for part in line[1:-1].split('.'):\n"
            "                current = current.setdefault(part, {})\n"
            "            continue\n"
            "        if '=' not in line:\n"
            "            continue\n"
            "        key, value = line.split('=', 1)\n"
            "        current[key.strip()] = parse_scalar(value)\n"
            "    return data\n\n"
            "class ConfigProfileTests(unittest.TestCase):\n"
            "    def test_selected_provider_is_proxy(self) -> None:\n"
            "        data = load_config()\n"
            "        self.assertEqual(data['provider'], EXPECTED_PROVIDER)\n"
            "        provider = data['model_providers'][EXPECTED_PROVIDER]\n"
            "        self.assertEqual(provider['base_url'], EXPECTED_BASE_URL)\n"
            "        self.assertEqual(provider['wire_api'], 'responses')\n"
            "        self.assertIs(provider['store'], True)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "tests/test_smoke_profile.py": (
            "from __future__ import annotations\n\n"
            "import json\n"
            "import subprocess\n"
            "import sys\n"
            "import unittest\n"
            "from pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "CONFIG = ROOT / 'serving_maintenance' / '.codex' / 'config.toml'\n"
            "SCRIPT = ROOT / 'serving_maintenance' / 'scripts' / 'smoke_responses_profile.py'\n"
            "FIXTURES = ROOT / 'serving_maintenance' / 'fixtures' / 'http'\n\n"
            "def run(turn2_name: str) -> subprocess.CompletedProcess[str]:\n"
            "    return subprocess.run(\n"
            "        [\n"
            "            sys.executable,\n"
            "            str(SCRIPT),\n"
            "            '--config', str(CONFIG),\n"
            "            '--turn1', str(FIXTURES / 'turn1_ok.json'),\n"
            "            '--turn2', str(FIXTURES / turn2_name),\n"
            "        ],\n"
            "        cwd=ROOT,\n"
            "        capture_output=True,\n"
            "        text=True,\n"
            "        check=False,\n"
            "    )\n\n"
            "class SmokeProfileTests(unittest.TestCase):\n"
            "    def test_good_followup_passes(self) -> None:\n"
            "        proc = run('turn2_ok.json')\n"
            "        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)\n"
            "        payload = json.loads(proc.stdout)\n"
            "        self.assertTrue(payload['continuity_ok'])\n"
            "        self.assertTrue(payload['store_ok'])\n\n"
            "    def test_missing_store_fails(self) -> None:\n"
            "        proc = run('turn2_missing_store.json')\n"
            "        self.assertNotEqual(proc.returncode, 0)\n"
            "        self.assertIn('store', proc.stdout + proc.stderr)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "tests/test_docs_sync.py": (
            "from __future__ import annotations\n\n"
            "import unittest\n"
            "from pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "PROVIDER_DOC = ROOT / 'serving_maintenance' / 'docs' / 'provider_rollover.md'\n"
            "SMOKE_DOC = ROOT / 'serving_maintenance' / 'docs' / 'smoke.md'\n\n"
            "class DocsSyncTests(unittest.TestCase):\n"
            "    def test_docs_reference_current_provider_and_continuity(self) -> None:\n"
            "        provider_text = PROVIDER_DOC.read_text()\n"
            "        smoke_text = SMOKE_DOC.read_text()\n"
            "        self.assertIn('responses_proxy', provider_text)\n"
            "        self.assertIn('http://127.0.0.1:11434/v1/responses', provider_text + smoke_text)\n"
            "        self.assertIn('previous_response_id', smoke_text)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
    }


def workspace_files_for_variant(variant: dict[str, object]) -> dict[str, str]:
    files = {
        ".scenario_variant": variant["id"] + "\n",
        "AGENTS.md": agents_md(variant),
        "Dockerfile": (
            "FROM python:3.11-slim\n"
            "WORKDIR /workspace\n"
            "COPY . /workspace\n"
            "CMD [\"bash\", \"bin/run-visible-tests\"]\n"
        ),
        "bin/run-visible-tests": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "python3 -m unittest discover -s tests -p 'test_*.py'\n"
        ),
        "serving_maintenance/.codex/config.toml": broken_config_text(variant["id"] == "v2-noisy-distractor"),
        "serving_maintenance/scripts/smoke_responses_profile.py": broken_smoke_script(),
        "serving_maintenance/docs/provider_rollover.md": broken_provider_doc(variant),
        "serving_maintenance/docs/smoke.md": broken_smoke_doc(),
    }
    files.update(turn_fixtures())
    files.update(visible_tests())
    files.update(variant["extra_workspace"])
    return files


def oracle_files_for_variant(variant: dict[str, object]) -> dict[str, str]:
    return {
        "serving_maintenance/.codex/config.toml": fixed_config_text(variant["id"] == "v2-noisy-distractor"),
        "serving_maintenance/scripts/smoke_responses_profile.py": fixed_smoke_script(),
        "serving_maintenance/docs/provider_rollover.md": fixed_provider_doc(variant),
        "serving_maintenance/docs/smoke.md": fixed_smoke_doc(),
    }


def workspace_manifest(workspace: Path) -> dict[str, object]:
    files: dict[str, str] = {}
    for path in sorted(p for p in workspace.rglob("*") if p.is_file()):
        rel = path.relative_to(workspace).as_posix()
        if rel.endswith(".pyc") or "/__pycache__/" in rel:
            continue
        files[rel] = sha256_file(path)
    return {
        "schema_version": "cnb55.workspace_manifest.v1",
        "files": files,
    }


def maybe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src, dst.parent)
    os.symlink(rel, dst)


def milestone_scripts() -> dict[str, str]:
    return {
        "m1_localize.sh": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "python3 - \"$RESULT_FILE\" <<'PY'\n"
            "import json, sys\n"
            "data = json.load(open(sys.argv[1]))\n"
            "checks = data.get('checks', {})\n"
            "ok = checks.get('config_selected_provider') or checks.get('docs_core_sync')\n"
            "raise SystemExit(0 if ok else 1)\n"
            "PY\n"
        ),
        "m2_primary_fix.sh": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "python3 - \"$RESULT_FILE\" <<'PY'\n"
            "import json, sys\n"
            "data = json.load(open(sys.argv[1]))\n"
            "checks = data.get('checks', {})\n"
            "ok = all(checks.get(k) for k in ['visible_tests_pass', 'config_selected_provider', 'config_proxy_contract', 'smoke_good_pass'])\n"
            "raise SystemExit(0 if ok else 1)\n"
            "PY\n"
        ),
        "m3_invariants.sh": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "python3 - \"$RESULT_FILE\" <<'PY'\n"
            "import json, sys\n"
            "data = json.load(open(sys.argv[1]))\n"
            "ok = data.get('integrity_flag') == 0\n"
            "raise SystemExit(0 if ok else 1)\n"
            "PY\n"
        ),
        "m4_functional.sh": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "python3 - \"$RESULT_FILE\" <<'PY'\n"
            "import json, sys\n"
            "data = json.load(open(sys.argv[1]))\n"
            "checks = data.get('checks', {})\n"
            "ok = all(checks.get(k) for k in ['visible_tests_pass', 'smoke_hidden_rejections', 'docs_core_sync']) and data.get('integrity_flag') == 0\n"
            "raise SystemExit(0 if ok else 1)\n"
            "PY\n"
        ),
        "m5_e2e.sh": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "python3 - \"$RESULT_FILE\" <<'PY'\n"
            "import json, sys\n"
            "data = json.load(open(sys.argv[1]))\n"
            "checks = data.get('checks', {})\n"
            "ok = all(checks.get(k) for k in ['visible_tests_pass', 'config_selected_provider', 'config_proxy_contract', 'smoke_good_pass', 'smoke_hidden_rejections', 'preserve_block_exact', 'docs_variant_sync']) and data.get('integrity_flag') == 0 and data.get('P_benchmark', 0) >= 90\n"
            "raise SystemExit(0 if ok else 1)\n"
            "PY\n"
        ),
    }


def build_family() -> None:
    clear_dir(WORKSPACE_ROOT)
    clear_dir(VERIFIER_ROOT)

    shared_milestones = VERIFIER_ROOT / "_milestones_shared"
    shared_milestones.mkdir(parents=True, exist_ok=True)
    for name, content in milestone_scripts().items():
        write_text(shared_milestones / name, content, 0o755)

    hidden_fixture_payloads = hidden_negative_fixtures()
    lock = {
        "family_id": "codex-provider-rollover",
        "schema_version": "cnb55.manifest.v2",
        "grader": {
            "score_provider_rollover_py_sha256": sha256_file(SCORER),
            "verify_sh_sha256": sha256_file(VERIFY_SH),
        },
        "variants": {},
    }

    for variant in VARIANTS:
        variant_id = variant["id"]
        workspace = WORKSPACE_ROOT / variant_id
        workspace.mkdir(parents=True, exist_ok=True)
        for rel, content in workspace_files_for_variant(variant).items():
            mode = 0o755 if rel.startswith("bin/") or rel.endswith(".py") and "/scripts/" in rel else 0o644
            write_text(workspace / rel, content, mode)

        verifier_variant = VERIFIER_ROOT / variant_id
        verifier_variant.mkdir(parents=True, exist_ok=True)
        oracle_root = verifier_variant / "oracle"
        hidden_root = verifier_variant / "hidden_fixtures"
        oracle_root.mkdir(parents=True, exist_ok=True)
        hidden_root.mkdir(parents=True, exist_ok=True)

        for rel, content in oracle_files_for_variant(variant).items():
            mode = 0o755 if rel.endswith(".py") else 0o644
            write_text(oracle_root / rel, content, mode)

        for name, payload in hidden_fixture_payloads.items():
            write_json(hidden_root / name, payload)

        gold = {
            "family_id": "codex-provider-rollover",
            "variant_id": variant_id,
            "expected_provider": EXPECTED_PROVIDER,
            "expected_base_url": EXPECTED_BASE_URL,
            "expected_env_key": EXPECTED_ENV_KEY,
            "allowed_writes": ALLOWED_WRITES,
            "readonly_roots": READONLY_ROOTS,
            "preserve_block": PRESERVE_BLOCK,
            "negative_smoke_fixtures": sorted(hidden_fixture_payloads),
            "required_doc_keywords": variant["required_doc_keywords"],
            "variant_ceiling": variant["variant_ceiling"],
            "pass_threshold": 90,
        }
        write_json(verifier_variant / "gold_reference.json", gold)
        write_json(verifier_variant / "workspace_manifest.json", workspace_manifest(workspace))

        milestone_dir = verifier_variant / "milestones"
        milestone_dir.mkdir(parents=True, exist_ok=True)
        for name in milestone_scripts():
            maybe_symlink(shared_milestones / name, milestone_dir / name)

        lock["variants"][variant_id] = {
            "gold_reference_sha256": sha256_file(verifier_variant / "gold_reference.json"),
            "workspace_manifest_sha256": sha256_file(verifier_variant / "workspace_manifest.json"),
        }

    LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    build_family()
