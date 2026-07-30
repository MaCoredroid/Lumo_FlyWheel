from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import fr13_fixed32_contract as contract  # noqa: E402

EXPECTED_NSYS_PREFIX = (
    "/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys",
    "profile",
    "--delay",
    "1200",
    "--duration",
    "300",
    "--trace=cuda,cuda-sw,nvtx",
    "--cuda-graph-trace=node",
    "--cuda-flush-interval",
    "100",
    "--discard-environment=true",
    "--sample=none",
    "--cpuctxsw=none",
    "--force-overwrite=true",
    "-o",
    "/logs/fr13_fixed32_b1_real_swe",
)


def test_nsys_prefix_is_the_literal_profile_contract() -> None:
    assert contract.NSYS_PROFILE_PREFIX == EXPECTED_NSYS_PREFIX


def test_direct_pid1_is_required_for_acceptance() -> None:
    direct = contract.expected_pid1_argv(1)
    wrapped = [
        *EXPECTED_NSYS_PREFIX,
        "vllm",
        *direct[2:],
    ]

    assert contract.validate_process_pid1_argv(
        direct,
        1,
        attribution_only=False,
    ) == direct
    with pytest.raises(contract.ContractError, match="PID1 argv mismatch"):
        contract.validate_process_pid1_argv(
            wrapped,
            1,
            attribution_only=False,
        )


def test_exact_nsys_pid1_is_required_for_attribution() -> None:
    expected = [
        *EXPECTED_NSYS_PREFIX,
        "vllm",
        *contract.expected_pid1_argv(1)[2:],
    ]

    assert contract.validate_process_pid1_argv(
        expected,
        1,
        attribution_only=True,
    ) == expected
    with pytest.raises(contract.ContractError, match="PID1 argv mismatch"):
        contract.validate_process_pid1_argv(
            contract.expected_pid1_argv(1),
            1,
            attribution_only=True,
        )


@pytest.mark.parametrize(
    ("index", "replacement"),
    (
        (0, "/tmp/nsys"),
        (1, "launch"),
        (2, "--duration"),
        (3, "1199"),
        (4, "--delay"),
        (5, "301"),
        (6, "--trace=cuda,nvtx"),
        (7, "--cuda-graph-trace=graph"),
        (8, "--cuda-flush-interval=101"),
        (9, "99"),
        (10, "--discard-environment=false"),
        (11, "--sample=process-tree"),
        (12, "--cpuctxsw=process-tree"),
        (13, "--force-overwrite=false"),
        (14, "--output"),
        (15, "/logs/other"),
    ),
)
def test_nsys_pid1_prefix_tamper_fails(
    index: int,
    replacement: str,
) -> None:
    argv = contract.expected_process_pid1_argv(1, attribution_only=True)
    argv[index] = replacement

    with pytest.raises(contract.ContractError, match="PID1 argv mismatch"):
        contract.validate_process_pid1_argv(
            argv,
            1,
            attribution_only=True,
        )


def test_nsys_pid1_wrapped_vllm_tamper_fails() -> None:
    argv = contract.expected_process_pid1_argv(1, attribution_only=True)
    argv[-1] = "other.middleware"

    with pytest.raises(contract.ContractError, match="PID1 argv mismatch"):
        contract.validate_process_pid1_argv(
            argv,
            1,
            attribution_only=True,
        )


def test_live_attestation_receives_the_selector_explicitly() -> None:
    serve = (
        REPO / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"
    ).read_text(encoding="utf-8")

    assert '"${FR13_FIXED32_ATTRIBUTION_ONLY:-0}" <<\'PY\'' in serve
    assert "attribution_only_text = sys.argv[6]" in serve
    assert "attribution_only_text = os.environ" not in serve
