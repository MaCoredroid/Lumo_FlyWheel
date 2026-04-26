from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


KERNEL_SELECTION_RUNTIME_UNSUPPORTED = "l0a_kernel_selection_runtime_unsupported_knobs"


@dataclass(frozen=True)
class UnsupportedKernelKnob:
    axis: str
    value: str
    reason: str
    required_runtime_hook: str

    def as_dict(self) -> dict[str, str]:
        return {
            "axis": self.axis,
            "value": self.value,
            "reason": self.reason,
            "required_runtime_hook": self.required_runtime_hook,
        }


@dataclass(frozen=True)
class KernelRuntimeActivationPlan:
    kernel_selection: dict[str, Any]
    supported: bool
    launch_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    resolved: dict[str, Any] = field(default_factory=dict)
    unsupported_knobs: list[UnsupportedKernelKnob] = field(default_factory=list)

    @property
    def restart_required(self) -> bool:
        return bool(self.kernel_selection)

    @property
    def activation_id(self) -> str:
        payload = {
            "kernel_selection": self.kernel_selection,
            "launch_args": self.launch_args,
            "env": self.env,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "restart_required": self.restart_required,
            "activation_id": self.activation_id,
            "kernel_selection": dict(self.kernel_selection),
            "launch_args": list(self.launch_args),
            "env": dict(self.env),
            "resolved": dict(self.resolved),
            "unsupported_knobs": [knob.as_dict() for knob in self.unsupported_knobs],
        }


class KernelRuntimeActivationError(RuntimeError):
    def __init__(self, plan: KernelRuntimeActivationPlan) -> None:
        self.plan = plan
        unsupported = ", ".join(
            f"{knob.axis}={knob.value} ({knob.required_runtime_hook})"
            for knob in plan.unsupported_knobs
        )
        super().__init__(
            f"HALT_REASON: {KERNEL_SELECTION_RUNTIME_UNSUPPORTED}; unsupported runtime kernel knob(s): {unsupported}"
        )


def resolve_kernel_runtime_activation(kernel_selection: dict[str, Any] | None) -> KernelRuntimeActivationPlan:
    selection = dict(kernel_selection or {})
    launch_args: list[str] = []
    env: dict[str, str] = {}
    compilation_config: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    unsupported: list[UnsupportedKernelKnob] = []

    if not selection:
        return KernelRuntimeActivationPlan(
            kernel_selection={},
            supported=True,
            resolved={
                "attention_backend": "vllm-auto",
                "deltanet_kernel": "vllm-auto",
                "fp8_gemm_kernel": "vllm-auto",
                "torch_compile_mode": "vllm-default",
                "cuda_graph_capture": "vllm-default",
            },
        )

    allowed_axes = {
        "combo_id",
        "attention_backend",
        "deltanet_kernel",
        "fp8_gemm_kernel",
        "torch_compile_mode",
        "cuda_graph_capture",
    }
    for axis in sorted(set(selection) - allowed_axes):
        unsupported.append(
            UnsupportedKernelKnob(
                axis=axis,
                value=str(selection[axis]),
                reason="unknown L0a kernel_selection axis",
                required_runtime_hook="add a repo-owned vLLM launch/runtime mapping for this axis",
            )
        )

    _apply_attention_backend(selection.get("attention_backend"), launch_args, resolved, unsupported)
    _apply_deltanet_kernel(selection.get("deltanet_kernel"), launch_args, resolved, unsupported)
    _apply_fp8_gemm_kernel(selection.get("fp8_gemm_kernel"), env, resolved, unsupported)
    compile_mode_applied = _apply_torch_compile_mode(
        selection.get("torch_compile_mode"),
        compilation_config,
        resolved,
        unsupported,
    )
    _apply_cuda_graph_capture(
        selection.get("cuda_graph_capture"),
        launch_args,
        compilation_config,
        resolved,
        unsupported,
        compile_mode_applied=compile_mode_applied,
    )
    if compilation_config:
        launch_args.extend(["--compilation-config", json.dumps(compilation_config, sort_keys=True)])

    return KernelRuntimeActivationPlan(
        kernel_selection=selection,
        supported=not unsupported,
        launch_args=launch_args,
        env=env,
        resolved=resolved,
        unsupported_knobs=unsupported,
    )


def require_supported_kernel_runtime_activation(kernel_selection: dict[str, Any] | None) -> KernelRuntimeActivationPlan:
    plan = resolve_kernel_runtime_activation(kernel_selection)
    if not plan.supported:
        raise KernelRuntimeActivationError(plan)
    return plan


def _apply_attention_backend(
    value: Any,
    launch_args: list[str],
    resolved: dict[str, Any],
    unsupported: list[UnsupportedKernelKnob],
) -> None:
    if value is None:
        resolved["attention_backend"] = "vllm-auto"
        return
    normalized = str(value)
    if normalized == "vllm-default":
        resolved["attention_backend"] = "vllm-auto"
    elif normalized == "flash-attn-4":
        launch_args.extend(["--attention-config", json.dumps({"backend": "FLASH_ATTN", "flash_attn_version": 4})])
        resolved["attention_backend"] = "FLASH_ATTN"
        resolved["flash_attn_version"] = 4
    elif normalized == "flash-attn-3":
        launch_args.extend(["--attention-config", json.dumps({"backend": "FLASH_ATTN", "flash_attn_version": 3})])
        resolved["attention_backend"] = "FLASH_ATTN"
        resolved["flash_attn_version"] = 3
    elif normalized == "flashinfer":
        launch_args.extend(["--attention-backend", "FLASHINFER"])
        resolved["attention_backend"] = "FLASHINFER"
    elif normalized == "triton":
        launch_args.extend(["--attention-backend", "TRITON_ATTN"])
        resolved["attention_backend"] = "TRITON_ATTN"
    else:
        unsupported.append(
            UnsupportedKernelKnob(
                axis="attention_backend",
                value=normalized,
                reason="vLLM 0.19 attention backend is not known to this repo mapping",
                required_runtime_hook="map to --attention-backend or --attention-config",
            )
        )


def _apply_deltanet_kernel(
    value: Any,
    launch_args: list[str],
    resolved: dict[str, Any],
    unsupported: list[UnsupportedKernelKnob],
) -> None:
    if value is None:
        resolved["deltanet_kernel"] = "vllm-auto"
        return
    normalized = str(value)
    if normalized == "triton-chunked-delta-v2":
        launch_args.extend(["--gdn-prefill-backend", "triton"])
        resolved["deltanet_kernel"] = "triton"
    else:
        unsupported.append(
            UnsupportedKernelKnob(
                axis="deltanet_kernel",
                value=normalized,
                reason="vLLM exposes only GDN prefill backend selection for this model",
                required_runtime_hook="add an exact DeltaNet state/update kernel selector in vLLM or repo launch patches",
            )
        )


def _apply_fp8_gemm_kernel(
    value: Any,
    env: dict[str, str],
    resolved: dict[str, Any],
    unsupported: list[UnsupportedKernelKnob],
) -> None:
    if value is None:
        resolved["fp8_gemm_kernel"] = "vllm-auto"
        return
    normalized = str(value)
    if normalized == "cublas":
        env["VLLM_DISABLED_KERNELS"] = ",".join(
            [
                "MarlinFP8ScaledMMLinearKernel",
                "FlashInferFP8ScaledMMLinearKernel",
                "CutlassFP8ScaledMMLinearKernel",
            ]
        )
        resolved["fp8_gemm_kernel"] = "torch_scaled_mm"
        resolved["fp8_gemm_kernel_activation"] = "disabled non-Torch FP8 scaled-mm kernels"
    elif normalized == "cutlass":
        env["VLLM_DISABLED_KERNELS"] = ",".join(
            [
                "MarlinFP8ScaledMMLinearKernel",
                "FlashInferFP8ScaledMMLinearKernel",
                "PerTensorTorchFP8ScaledMMLinearKernel",
                "ChannelWiseTorchFP8ScaledMMLinearKernel",
                "RowWiseTorchFP8ScaledMMLinearKernel",
            ]
        )
        resolved["fp8_gemm_kernel"] = "CutlassFP8ScaledMMLinearKernel"
        resolved["fp8_gemm_kernel_activation"] = "disabled non-CUTLASS FP8 scaled-mm kernels"
    else:
        unsupported.append(
            UnsupportedKernelKnob(
                axis="fp8_gemm_kernel",
                value=normalized,
                reason="repo has no safe exact dense FP8 GEMM launch hook for this value",
                required_runtime_hook="add a dense FP8 GEMM backend selector; --moe-backend is not equivalent",
            )
        )


def _apply_torch_compile_mode(
    value: Any,
    compilation_config: dict[str, Any],
    resolved: dict[str, Any],
    unsupported: list[UnsupportedKernelKnob],
) -> bool:
    if value is None:
        resolved["torch_compile_mode"] = "vllm-default"
        return False
    normalized = str(value)
    if normalized == "default":
        resolved["torch_compile_mode"] = "vllm-default"
        return False
    mode_options = {
        "reduce-overhead": {
            "triton.cudagraphs": True,
        },
        "max-autotune": {
            "max_autotune": True,
            "triton.cudagraphs": True,
            "coordinate_descent_tuning": True,
        },
        "max-autotune-no-cudagraphs": {
            "max_autotune": True,
            "coordinate_descent_tuning": True,
        },
    }
    if normalized in mode_options:
        compilation_config["mode"] = "VLLM_COMPILE"
        inductor_config = compilation_config.setdefault("inductor_compile_config", {})
        inductor_config.update(mode_options[normalized])
        resolved["torch_compile_mode"] = normalized
        resolved["torch_compile_mode_activation"] = {
            "vllm_compilation_config.mode": "VLLM_COMPILE",
            "vllm_compilation_config.inductor_compile_config": dict(mode_options[normalized]),
        }
        return True
    else:
        unsupported.append(
            UnsupportedKernelKnob(
                axis="torch_compile_mode",
                value=normalized,
                reason="vLLM compilation config does not safely expose torch.compile(mode=...) for this serving path",
                required_runtime_hook="add a repo-owned vLLM compilation_config mapping with parity/runtime proof",
            )
        )
        return False


def _apply_cuda_graph_capture(
    value: Any,
    launch_args: list[str],
    compilation_config: dict[str, Any],
    resolved: dict[str, Any],
    unsupported: list[UnsupportedKernelKnob],
    *,
    compile_mode_applied: bool,
) -> None:
    if value is None:
        resolved["cuda_graph_capture"] = "vllm-default"
        return
    normalized = str(value)
    if normalized == "off":
        if compile_mode_applied:
            compilation_config["cudagraph_mode"] = "NONE"
            resolved["cuda_graph_capture"] = "NONE"
            resolved["cuda_graph_capture_activation"] = "vLLM compilation_config cudagraph_mode NONE"
        else:
            launch_args.append("--enforce-eager")
            resolved["cuda_graph_capture"] = "off"
    elif normalized == "on":
        compilation_config["cudagraph_mode"] = "FULL"
        resolved["cuda_graph_capture"] = "FULL"
    else:
        resolved["cuda_graph_capture"] = "unknown"
        unsupported.append(
            UnsupportedKernelKnob(
                axis="cuda_graph_capture",
                value=normalized,
                reason="expected on/off CUDA graph selection",
                required_runtime_hook="map to --enforce-eager or --compilation-config cudagraph_mode",
            )
        )
