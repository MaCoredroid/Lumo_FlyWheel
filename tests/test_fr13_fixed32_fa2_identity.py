from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import fr13_fixed32_contract as contract  # noqa: E402

EXPECTED_SHA256 = "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d"
EXPECTED_SIZE = 299_183_936
FA2_PATH = REPO / contract.FA2_REPO_RELATIVE


def test_suffix_only_fa2_identity_is_pinned() -> None:
    assert contract.FA2_SHA256 == EXPECTED_SHA256
    assert contract.FA2_SIZE == EXPECTED_SIZE


@pytest.mark.skipif(not FA2_PATH.is_file(), reason="ignored FA2 binary is not staged")
def test_staged_suffix_only_fa2_matches_identity() -> None:
    assert not FA2_PATH.is_symlink()
    assert FA2_PATH.stat().st_size == EXPECTED_SIZE
    assert contract.sha256_file(FA2_PATH) == EXPECTED_SHA256
