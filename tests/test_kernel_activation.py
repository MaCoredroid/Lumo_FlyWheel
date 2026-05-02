import pytest

from lumo_flywheel_serving.kernel_activation import (
    PHASE_A_BACKEND_DISPATCH_DRIFT,
    PhaseABackendDispatchDriftError,
    phase_a_fp8_gemm_backend_identities,
    resolve_kernel_runtime_activation,
    verify_phase_a_backend_identity_manifest,
)


def test_phase_a_fp8_gemm_backend_identities_pin_supported_dispatch_hook() -> None:
    identities = phase_a_fp8_gemm_backend_identities(["cutlass", "cublas"])

    assert set(identities) == {"cublas", "cutlass"}
    assert identities["cublas"]["action_space_value"] == "cublas"
    assert identities["cublas"]["resolved_runtime_name"] == "torch_scaled_mm"
    assert identities["cublas"]["support_status"] == "supported"
    assert identities["cublas"]["supported"] is True
    assert identities["cutlass"]["resolved_runtime_name"] == "CutlassFP8ScaledMMLinearKernel"
    assert identities["cutlass"]["support_status"] == "supported"
    assert identities["cutlass"]["supported"] is True
    for identity in identities.values():
        assert identity["repo_dispatch_hook_source_path"] == "src/lumo_flywheel_serving/kernel_activation.py"
        assert identity["repo_dispatch_hook_symbol"].endswith("._apply_fp8_gemm_kernel")
        assert identity["content_hash_algorithm"] == "sha256"
        assert identity["content_hash_scope"] == "repo_owned_runtime_activation_dispatch_hook"
        assert len(identity["content_hash"]) == 64


def test_unsupported_fp8_gemm_backend_keeps_structured_identity_evidence() -> None:
    plan = resolve_kernel_runtime_activation(
        {
            "combo_id": "combo_999",
            "attention_backend": "vllm-default",
            "deltanet_kernel": "triton-chunked-delta-v2",
            "fp8_gemm_kernel": "triton_fp8_scaled_mm",
            "torch_compile_mode": "default",
            "cuda_graph_capture": "off",
        }
    )

    assert plan.supported is False
    assert plan.unsupported_knobs[0].axis == "fp8_gemm_kernel"
    identity = plan.resolved["fp8_gemm_backend_identity"]
    assert identity["action_space_value"] == "triton_fp8_scaled_mm"
    assert identity["resolved_runtime_name"] is None
    assert identity["support_status"] == "unsupported"
    assert identity["supported"] is False


def test_phase_a_backend_identity_manifest_detects_dispatch_drift() -> None:
    manifest = {
        "phase_a_backend_identities": phase_a_fp8_gemm_backend_identities(["cublas", "cutlass"])
    }
    verified = verify_phase_a_backend_identity_manifest(manifest)
    assert set(verified) == {"cublas", "cutlass"}

    manifest["phase_a_backend_identities"]["cublas"]["content_hash"] = "0" * 64
    with pytest.raises(PhaseABackendDispatchDriftError) as excinfo:
        verify_phase_a_backend_identity_manifest(manifest)

    assert PHASE_A_BACKEND_DISPATCH_DRIFT in str(excinfo.value)
    assert excinfo.value.mismatches[0]["action_space_value"] == "cublas"
    assert excinfo.value.mismatches[0]["field"] == "content_hash"
