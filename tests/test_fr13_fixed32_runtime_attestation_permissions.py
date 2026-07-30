from __future__ import annotations

import argparse
import importlib.util
import json
import stat
import sys
from pathlib import Path


CONTRACT_PATH = Path("scripts/fr13_fixed32_contract.py")
sys.path.insert(0, str(CONTRACT_PATH.parent))
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_runtime_attestation_permissions_contract",
    CONTRACT_PATH,
)
assert CONTRACT_SPEC is not None and CONTRACT_SPEC.loader is not None
contract = importlib.util.module_from_spec(CONTRACT_SPEC)
CONTRACT_SPEC.loader.exec_module(contract)


def test_runtime_attestation_cli_publishes_host_readable_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "fr13_fixed32_runtime_attestation.json"
    payload = {"overall_canonical_sha256": "runtime-digest"}
    monkeypatch.setattr(
        contract,
        "parse_args",
        lambda: argparse.Namespace(
            command="runtime-attestation",
            output=output,
        ),
    )
    monkeypatch.setattr(contract, "build_runtime_attestation", lambda: payload)
    monkeypatch.setattr(
        contract,
        "validate_runtime_attestation",
        lambda value: value,
    )

    assert contract.main() == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_generic_atomic_json_output_remains_private(tmp_path: Path) -> None:
    output = tmp_path / "private.json"

    contract.atomic_write_json(output, {"secret": True})

    assert stat.S_IMODE(output.stat().st_mode) == 0o600
