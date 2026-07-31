from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import fr13_fixed32_contract as contract  # noqa: E402

EXPECTED_SHA256 = "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d"
EXPECTED_SIZE = 299_183_936
QROW16_EXPECTED_SHA256 = (
    "35ba18c9bab4b37362aa3b26441e8a58edfcd3d0a75692fda90fc131a0b3307c"
)
QROW16_EXPECTED_SIZE = 299_554_080
FA2_PATH = REPO / contract.FA2_REPO_RELATIVE


def test_suffix_only_fa2_identity_is_pinned() -> None:
    assert contract.FA2_SHA256 == EXPECTED_SHA256
    assert contract.FA2_SIZE == EXPECTED_SIZE
    assert contract.QROW16_FA2_SHA256 == QROW16_EXPECTED_SHA256
    assert contract.QROW16_FA2_SIZE == QROW16_EXPECTED_SIZE


@pytest.mark.skipif(not FA2_PATH.is_file(), reason="ignored FA2 binary is not staged")
def test_staged_suffix_only_fa2_matches_identity() -> None:
    assert not FA2_PATH.is_symlink()
    assert FA2_PATH.stat().st_size == EXPECTED_SIZE
    assert contract.sha256_file(FA2_PATH) == EXPECTED_SHA256


def _runtime_payload(*, size: int, sha256: str) -> dict[str, object]:
    arctic_files = [
        {
            "path": "arctic_inference/suffix_decoding/cache.py",
            "size": 1,
            "sha256": hashlib.sha256(b"x").hexdigest(),
        }
    ]
    payload: dict[str, object] = {
        "schema": contract.RUNTIME_SCHEMA,
        "canonical_format": contract.CANONICAL_FORMAT,
        "python": {"version": "3.12.3", "implementation": "CPython"},
        "vllm": {"version": contract.VLLM_VERSION, "module_path": "/vllm.py"},
        "forked_fa2": {
            "source": {
                "path": str(contract.CONTAINER_FA2_SOURCE),
                "size": size,
                "sha256": sha256,
            },
            "destination": {
                "path": str(contract.CONTAINER_FA2_DESTINATION),
                "size": size,
                "sha256": sha256,
            },
            "byte_identical": True,
        },
        "arctic": {
            "version": contract.ARCTIC_VERSION,
            "files": arctic_files,
            "canonical_sha256": hashlib.sha256(
                contract.canonical_bytes(arctic_files)
            ).hexdigest(),
            "cache_class_module": "arctic_inference.suffix_decoding.cache",
            "cache_class_qualname": "SuffixDecodingCache",
            "pinned_source_url": contract.ARCTIC_SDIST_URL,
            "pinned_source_sha256": contract.ARCTIC_SDIST_SHA256,
        },
    }
    payload["overall_canonical_sha256"] = hashlib.sha256(
        contract.canonical_bytes(payload)
    ).hexdigest()
    return payload


def test_runtime_identity_defaults_to_stock_when_qrow_is_off() -> None:
    env = {
        "FR13_FA2_QROW16_LIVE_PAGED_AB": "0",
        "FR13_FA2_QROW16_PRODUCTION": "0",
        "FR13_FA2_QROW16_SO_SHA256": QROW16_EXPECTED_SHA256,
    }
    assert contract._expected_runtime_fa2_identity(env) == (
        EXPECTED_SIZE,
        EXPECTED_SHA256,
    )
    contract.validate_runtime_attestation(
        _runtime_payload(size=EXPECTED_SIZE, sha256=EXPECTED_SHA256)
    )


@pytest.mark.parametrize(
    "selector",
    ["FR13_FA2_QROW16_LIVE_PAGED_AB", "FR13_FA2_QROW16_PRODUCTION"],
)
def test_runtime_identity_selects_only_pinned_qrow_candidate(selector: str) -> None:
    env = {
        "FR13_FA2_QROW16_LIVE_PAGED_AB": "0",
        "FR13_FA2_QROW16_PRODUCTION": "0",
        "FR13_FA2_QROW16_SO_SHA256": QROW16_EXPECTED_SHA256,
    }
    env[selector] = "1"
    assert contract._expected_runtime_fa2_identity(env) == (
        QROW16_EXPECTED_SIZE,
        QROW16_EXPECTED_SHA256,
    )
    contract.validate_runtime_attestation(
        _runtime_payload(
            size=QROW16_EXPECTED_SIZE,
            sha256=QROW16_EXPECTED_SHA256,
        )
    )


def test_runtime_identity_rejects_arbitrary_qrow_declaration() -> None:
    env = {
        "FR13_FA2_QROW16_LIVE_PAGED_AB": "1",
        "FR13_FA2_QROW16_PRODUCTION": "0",
        "FR13_FA2_QROW16_SO_SHA256": "0" * 64,
    }
    with pytest.raises(contract.ContractError, match="not the pinned candidate"):
        contract._expected_runtime_fa2_identity(env)
    with pytest.raises(contract.ContractError, match="source FA2 mismatch"):
        contract.validate_runtime_attestation(
            _runtime_payload(size=QROW16_EXPECTED_SIZE, sha256="0" * 64)
        )


def test_runtime_identity_rejects_candidate_when_qrow_is_off() -> None:
    env = {
        "FR13_FA2_QROW16_LIVE_PAGED_AB": "0",
        "FR13_FA2_QROW16_PRODUCTION": "0",
    }
    assert contract._expected_runtime_fa2_identity(env) == (
        EXPECTED_SIZE,
        EXPECTED_SHA256,
    )
    source = {
        "path": str(contract.CONTAINER_FA2_SOURCE),
        "size": QROW16_EXPECTED_SIZE,
        "sha256": QROW16_EXPECTED_SHA256,
    }
    destination = {
        "path": str(contract.CONTAINER_FA2_DESTINATION),
        "size": QROW16_EXPECTED_SIZE,
        "sha256": QROW16_EXPECTED_SHA256,
    }
    with pytest.raises(contract.ContractError, match="container FA2 identity mismatch"):
        contract._require_built_runtime_fa2_identity(
            source,
            destination,
            env=env,
        )


def test_runtime_identity_rejects_wrong_candidate_path() -> None:
    payload = _runtime_payload(
        size=QROW16_EXPECTED_SIZE,
        sha256=QROW16_EXPECTED_SHA256,
    )
    payload["forked_fa2"]["source"]["path"] = "/tmp/not-the-mounted-fa2.so"
    payload["overall_canonical_sha256"] = hashlib.sha256(
        contract.canonical_bytes(
            {
                key: value
                for key, value in payload.items()
                if key != "overall_canonical_sha256"
            }
        )
    ).hexdigest()
    with pytest.raises(contract.ContractError, match="source FA2 mismatch"):
        contract.validate_runtime_attestation(payload)
