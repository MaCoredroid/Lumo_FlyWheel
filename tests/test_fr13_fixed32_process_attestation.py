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
    "--session-new=%q{LUMO_NSYS_SESSION_NAME}",
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


def test_default_graph_and_eager_pid1_are_exact_distinct_contracts() -> None:
    default = contract.expected_process_pid1_argv(
        4,
        attribution_only=False,
    )
    eager = contract.expected_process_pid1_argv(
        4,
        attribution_only=False,
        eager_diagnostic=True,
    )
    graph = contract.expected_process_pid1_argv(
        4,
        attribution_only=False,
        graph_diagnostic=True,
    )

    assert default == contract.expected_pid1_argv(4)
    kv_index = default.index("--kv-cache-memory-bytes")
    assert default[kv_index : kv_index + 2] == [
        "--kv-cache-memory-bytes",
        str(contract.FIXED32_B4_KV_CACHE_MEMORY_BYTES),
    ]
    assert len(default) == 49
    assert graph == default
    assert len(graph) == 49
    assert eager == [*default, "--enforce-eager"]
    assert len(eager) == 50
    assert contract.validate_process_pid1_argv(
        graph,
        4,
        attribution_only=False,
        graph_diagnostic=True,
    ) == graph
    assert contract.validate_process_pid1_argv(
        eager,
        4,
        attribution_only=False,
        eager_diagnostic=True,
    ) == eager
    with pytest.raises(contract.ContractError, match="PID1 argv mismatch"):
        contract.validate_process_pid1_argv(
            eager,
            4,
            attribution_only=False,
        )
    with pytest.raises(contract.ContractError, match="PID1 argv mismatch"):
        contract.validate_process_pid1_argv(
            default,
            4,
            attribution_only=False,
            eager_diagnostic=True,
        )
    with pytest.raises(contract.ContractError, match="PID1 argv mismatch"):
        contract.validate_process_pid1_argv(
            eager,
            4,
            attribution_only=False,
            graph_diagnostic=True,
        )


def test_b1_omits_manual_kv_cache_and_b4_rejects_kv_tamper() -> None:
    b1 = contract.expected_pid1_argv(1)
    b4 = contract.expected_pid1_argv(4)

    assert "--kv-cache-memory-bytes" not in b1
    kv_index = b4.index("--kv-cache-memory-bytes")
    b4[kv_index + 1] = str(contract.FIXED32_B4_KV_CACHE_MEMORY_BYTES - 1)
    with pytest.raises(contract.ContractError, match="PID1 argv mismatch"):
        contract.validate_process_pid1_argv(
            b4,
            4,
            attribution_only=False,
            graph_diagnostic=True,
        )


def test_eager_diagnostic_is_not_an_attribution_contract() -> None:
    with pytest.raises(
        contract.ContractError,
        match="eager diagnostic cannot be attribution-only",
    ):
        contract.expected_process_pid1_argv(
            4,
            attribution_only=True,
            eager_diagnostic=True,
        )


def test_eager_diagnostic_is_b4_only() -> None:
    with pytest.raises(
        contract.ContractError,
        match="eager diagnostic requires concurrency 4",
    ):
        contract.expected_process_pid1_argv(
            1,
            attribution_only=False,
            eager_diagnostic=True,
        )


def test_streamk_eager_diagnostic_is_exact_b1_contract() -> None:
    default = contract.expected_pid1_argv(1)
    eager = contract.expected_process_pid1_argv(
        1,
        attribution_only=False,
        streamk_eager_diagnostic=True,
    )

    assert eager == [*default, "--enforce-eager"]
    assert contract.validate_process_pid1_argv(
        eager,
        1,
        attribution_only=False,
        streamk_eager_diagnostic=True,
    ) == eager
    with pytest.raises(
        contract.ContractError,
        match="Stream-K eager diagnostic requires concurrency 1",
    ):
        contract.expected_process_pid1_argv(
            4,
            attribution_only=False,
            streamk_eager_diagnostic=True,
        )


def test_sfwd_byte_diagnostic_is_eager_at_b1_and_b4() -> None:
    """The SFWD conv/post-prep + prior-reuse byte gates boot --enforce-eager.

    B1 is fr13_run_b1_sfwd_conv_postprep_gate.sh /
    fr13_run_b1_sfwd_prior_reuse_gate.sh (stock wave); B4 is
    fr13_run_b4_sfwd_embedded_gate_live_gate.sh.
    """
    for concurrency in (1, 4):
        default = contract.expected_pid1_argv(concurrency)
        eager = contract.expected_process_pid1_argv(
            concurrency,
            attribution_only=False,
            sfwd_byte_diagnostic=True,
        )

        assert eager == [*default, "--enforce-eager"]
        assert contract.validate_process_pid1_argv(
            eager,
            concurrency,
            attribution_only=False,
            sfwd_byte_diagnostic=True,
        ) == eager
        with pytest.raises(contract.ContractError, match="PID1 argv mismatch"):
            contract.validate_process_pid1_argv(
                eager,
                concurrency,
                attribution_only=False,
            )
        with pytest.raises(contract.ContractError, match="PID1 argv mismatch"):
            contract.validate_process_pid1_argv(
                default,
                concurrency,
                attribution_only=False,
                sfwd_byte_diagnostic=True,
            )


def test_sfwd_byte_diagnostic_composes_with_the_b1_cutlass_byte_wave() -> None:
    """conv/post-prep byte A/B also rides identity_wide256_fullgrid_b1_byte_ab."""
    expected = [*contract.expected_pid1_argv(1), "--enforce-eager"]

    assert contract.validate_process_pid1_argv(
        expected,
        1,
        attribution_only=False,
        streamk_eager_diagnostic=True,
        sfwd_byte_diagnostic=True,
    ) == expected


def test_sfwd_byte_diagnostic_excludes_graph_and_attribution() -> None:
    with pytest.raises(
        contract.ContractError,
        match="process diagnostics are mutually exclusive",
    ):
        contract.expected_process_pid1_argv(
            4,
            attribution_only=False,
            graph_diagnostic=True,
            sfwd_byte_diagnostic=True,
        )
    with pytest.raises(
        contract.ContractError,
        match="eager diagnostic cannot be attribution-only",
    ):
        contract.expected_process_pid1_argv(
            1,
            attribution_only=True,
            sfwd_byte_diagnostic=True,
        )


def test_graph_diagnostic_is_b4_only_and_not_attribution() -> None:
    with pytest.raises(
        contract.ContractError,
        match="graph diagnostic requires concurrency 4",
    ):
        contract.expected_process_pid1_argv(
            1,
            attribution_only=False,
            graph_diagnostic=True,
        )
    with pytest.raises(
        contract.ContractError,
        match="graph diagnostic cannot be attribution-only",
    ):
        contract.expected_process_pid1_argv(
            4,
            attribution_only=True,
            graph_diagnostic=True,
        )


def test_eager_and_graph_diagnostics_are_mutually_exclusive() -> None:
    with pytest.raises(
        contract.ContractError,
        match="process diagnostics are mutually exclusive",
    ):
        contract.expected_process_pid1_argv(
            4,
            attribution_only=False,
            eager_diagnostic=True,
            graph_diagnostic=True,
        )
    with pytest.raises(
        contract.ContractError,
        match="process diagnostics are mutually exclusive",
    ):
        contract.expected_process_pid1_argv(
            1,
            attribution_only=False,
            eager_diagnostic=True,
            streamk_eager_diagnostic=True,
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
        (2, "--session-new=unrelated-session"),
        (3, "--duration"),
        (4, "1199"),
        (5, "--delay"),
        (6, "301"),
        (7, "--trace=cuda,nvtx"),
        (8, "--cuda-graph-trace=graph"),
        (9, "--cuda-flush-interval=101"),
        (10, "99"),
        (11, "--discard-environment=false"),
        (12, "--sample=process-tree"),
        (13, "--cpuctxsw=process-tree"),
        (14, "--force-overwrite=false"),
        (15, "--output"),
        (16, "/logs/other"),
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

    assert '"${FR13_FIXED32_ATTRIBUTION_ONLY:-0}" \\' in serve
    assert '"${FR13_FIXED32_BATCH_GDN_BYTE_AB:-0}" \\' in serve
    assert '"${FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB:-0}" \\' in serve
    assert '"${FR13_FIXED32_CUTLASS_WAVE:-stock}" \\' in serve
    assert '"$FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB" \\' in serve
    assert '"$FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB" \\' in serve
    assert '"$FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB" \\' in serve
    # The capture SHAPE is a selector too, and is passed by the same explicit
    # discipline. Default-off: unset LUMO_NSYS_START_LATER attests the
    # canonical wall-gated B1 prefix byte for byte.
    assert '"${LUMO_NSYS_START_LATER:-0}" \\' in serve
    assert '"${LUMO_NSYS_OUTPUT:-}" <<\'PY\'' in serve
    assert "attribution_only_text = sys.argv[7]" in serve
    assert "batch_gdn_byte_ab_text = sys.argv[8]" in serve
    assert "batch_gdn_graph_byte_ab_text = sys.argv[9]" in serve
    assert "cutlass_wave = sys.argv[10]" in serve
    assert "sfwd_b4_byte_ab_text = sys.argv[11]" in serve
    assert "sfwd_conv_postprep_byte_ab_text = sys.argv[12]" in serve
    assert "sfwd_prior_reuse_byte_ab_text = sys.argv[13]" in serve
    assert "nsys_start_later_text = sys.argv[14]" in serve
    assert "nsys_capture_output = sys.argv[15] or None" in serve
    assert "deferred_capture=nsys_start_later_text == \"1\"," in serve
    assert "nsys_start_later_text = os.environ" not in serve
    assert "nsys_capture_output = os.environ" not in serve
    assert "attribution_only_text = os.environ" not in serve
    assert "batch_gdn_byte_ab_text = os.environ" not in serve
    assert "batch_gdn_graph_byte_ab_text = os.environ" not in serve
    assert "sfwd_conv_postprep_byte_ab_text = os.environ" not in serve
    assert "sfwd_prior_reuse_byte_ab_text = os.environ" not in serve
    assert "eager_diagnostic=(" in serve
    assert "batch_gdn_byte_ab_text == \"1\"" in serve
    assert '"persistent_b4_m128_byte_ab",' in serve
    assert 'or sfwd_b4_byte_ab_text == "1"' in serve
    assert (
        "graph_diagnostic=batch_gdn_graph_byte_ab_text == \"1\"" in serve
    )
    assert "sfwd_byte_diagnostic=(" in serve
    assert 'sfwd_conv_postprep_byte_ab_text == "1"' in serve
    assert 'or sfwd_prior_reuse_byte_ab_text == "1"' in serve
