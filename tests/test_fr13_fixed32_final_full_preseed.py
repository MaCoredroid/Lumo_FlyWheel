from __future__ import annotations

import ast
import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


PATCHER_PATH = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py")
KERNEL_PATH = Path(
    "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
)
PATCHER_SPEC = importlib.util.spec_from_file_location(
    "fr13_final_full_preseed_patcher",
    PATCHER_PATH,
)
assert PATCHER_SPEC is not None and PATCHER_SPEC.loader is not None
patcher = importlib.util.module_from_spec(PATCHER_SPEC)
PATCHER_SPEC.loader.exec_module(patcher)


def _runtime_functions(*names: str, mode: str) -> dict[str, object]:
    tree = ast.parse(patcher._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE)
    wanted = set(names)
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace: dict[str, object] = {"torch": torch}
    exec(patcher._fr13_fixed32_runtime_bindings(mode), namespace)
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            "<fixed32-observed-runtime>",
            "exec",
        ),
        namespace,
    )
    return namespace


def _kernel_functions(
    *names: str,
    namespace: dict[str, object],
) -> dict[str, object]:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    wanted = set(names)
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            "<fixed32-kernel-runtime>",
            "exec",
        ),
        namespace,
    )
    return namespace


@pytest.mark.parametrize(
    ("mode", "active_nodes", "valid_mask"),
    (
        ("tail6_fixed32", 21, 0x7A9CE73F),
        ("hydra27_fixed32", 27, 0x7ABDFFFF),
    ),
)
def test_final_full_preseed_requires_physical_32_row_descriptor(
    mode: str,
    active_nodes: int,
    valid_mask: int,
) -> None:
    namespace = _runtime_functions(
        "_fr13_fixed32_graph_descriptor",
        "_fr13_fixed32_final_full_preseed_postcheck_required",
        "_fr13_fixed32_final_full_preseed_needed",
        mode=mode,
    )
    namespace.update(
        {
            "_FR13_FIXED32_PROFILE_MEMORY_SCOPE": False,
            "_FR13_FIXED32_PROFILE_CAPTURE_SCOPE": None,
        }
    )
    assert namespace["_FR13_FIXED32_MODE"] == mode
    assert namespace["_FR13_FIXED32_VALID_MASK"] == valid_mask
    needed = namespace["_fr13_fixed32_final_full_preseed_needed"]
    postcheck = namespace[
        "_fr13_fixed32_final_full_preseed_postcheck_required"
    ]

    assert postcheck("FULL") is True
    assert needed("FULL", 32, 1, True, False, 0) is True
    with pytest.raises(
        RuntimeError,
        match="full-graph descriptor drift",
    ):
        needed("FULL", active_nodes, 1, True, False, 0)

    namespace["_FR13_FIXED32_PROFILE_MEMORY_SCOPE"] = True
    assert postcheck("FULL") is False
    assert needed("FULL", 32, 1, True, False, 0) is False
    namespace["_FR13_FIXED32_PROFILE_MEMORY_SCOPE"] = False
    assert postcheck("PIECEWISE") is False
    assert needed("PIECEWISE", 64, 1, False, False, 0) is False

    namespace["_FR13_FIXED32_PRESEED_CAP"] = 4
    namespace["_FR13_FIXED32_PRESEEDED_BATCHES"] = {1, 2, 3, 4}
    assert needed("FULL", 32, 1, True, False, 0) is False
    assert needed("FULL", 128, 4, True, False, 0) is False


_GPU_RUNNER_FIXTURE = """\
class Runner:
    def __init__(
        self,
        num_warmups,
        cudagraph_mode=CUDAGraphMode.NONE,
        max_num_seqs=4,
    ):
        self.compilation_config = SimpleNamespace(
            cudagraph_num_of_warmups=num_warmups,
            cudagraph_mode=cudagraph_mode,
        )
        self.scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
        self.input_batch = SimpleNamespace(vocab_size=248320)
        self.calls = []

    def _dummy_run(self, num_tokens, **kwargs):
        self.calls.append((num_tokens, kwargs))

    def _warmup_and_capture(
        self,
        desc,
        cudagraph_runtime_mode,
        profile_seq_lens=None,
        allow_microbatching=False,
        num_warmups=None,
    ):
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
        for _ in range(num_warmups):
            self._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                force_attention=force_attention,
                uniform_decode=desc.uniform,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
                profile_seq_lens=profile_seq_lens,
            )
        self._dummy_run(
            desc.num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            uniform_decode=desc.uniform,
            allow_microbatching=allow_microbatching,
            skip_eplb=True,
            remove_lora=False,
            num_active_loras=desc.num_active_loras,
            is_graph_capturing=True,
            profile_seq_lens=profile_seq_lens,
        )

    def capture_model(self):
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0
        return 1
"""


class _CUDAGraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


def _install_fake_gdn(
    monkeypatch: pytest.MonkeyPatch,
    fake_gdn: SimpleNamespace,
) -> None:
    modules = {
        "vllm": ModuleType("vllm"),
        "vllm.model_executor": ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": ModuleType(
            "vllm.model_executor.layers"
        ),
        "vllm.model_executor.layers.mamba": ModuleType(
            "vllm.model_executor.layers.mamba"
        ),
    }
    for module in modules.values():
        module.__path__ = []
    modules["vllm.model_executor.layers.mamba"].gdn_linear_attn = fake_gdn
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_normalized_one_warmup_gets_explicit_capture_shaped_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "gpu_model_runner.py"
    source.write_text(_GPU_RUNNER_FIXTURE)
    monkeypatch.setattr(patcher, "GPU_MODEL_RUNNER_PATH", source)
    monkeypatch.setattr(patcher, "_FR13_FIXED32_MODE", "tail6_fixed32")
    assert patcher._patch_gpu_model_runner_fixed32_final_full_preseed()

    holder: dict[str, object] = {}

    def needed(*args: object) -> bool:
        holder["needed"] = args
        return True

    def assert_ready(batch: int) -> None:
        calls = holder["runner"].calls
        assert calls[-1][1]["cudagraph_runtime_mode"] is _CUDAGraphMode.NONE
        assert calls[-1][1]["is_graph_capturing"] is True
        holder["ready"] = batch

    fake_gdn = SimpleNamespace(
        _fr13_fixed32_final_full_preseed_postcheck_required=(
            lambda mode: mode == "FULL"
        ),
        _fr13_fixed32_final_full_preseed_needed=needed,
        _fr13_fixed32_warm_final_full_postprocess=lambda vocab: holder.update(
            warm_vocab=vocab
        ),
        _fr13_fixed32_assert_final_full_preseed_ready=assert_ready,
    )
    _install_fake_gdn(monkeypatch, fake_gdn)
    namespace = {
        "CUDAGraphMode": _CUDAGraphMode,
        "SimpleNamespace": SimpleNamespace,
        "_fr13_cfwd_timer": lambda: holder.update(cfwd_ready=True),
        "_fr13_dfwd_timer": lambda: holder.update(dfwd_ready=True),
    }
    exec(source.read_text(), namespace)
    runner = namespace["Runner"](1)
    holder["runner"] = runner
    descriptor = SimpleNamespace(
        num_tokens=32,
        num_reqs=1,
        uniform=True,
        has_lora=False,
        num_active_loras=0,
    )
    runner._warmup_and_capture(descriptor, _CUDAGraphMode.FULL)

    assert len(runner.calls) == 3
    stock, producer, capture = runner.calls
    assert stock[1]["cudagraph_runtime_mode"] is _CUDAGraphMode.NONE
    assert "is_graph_capturing" not in stock[1]
    assert producer[1]["cudagraph_runtime_mode"] is _CUDAGraphMode.NONE
    assert producer[1]["is_graph_capturing"] is True
    assert capture[1]["cudagraph_runtime_mode"] is _CUDAGraphMode.FULL
    assert capture[1]["is_graph_capturing"] is True
    assert holder["ready"] == 1
    assert holder["needed"] == ("FULL", 32, 1, True, False, 0)
    assert holder["warm_vocab"] == 248320
    assert holder["cfwd_ready"] is True
    assert holder["dfwd_ready"] is True


def test_ready_metadata_with_stale_lease_fails_before_full_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "gpu_model_runner.py"
    source.write_text(_GPU_RUNNER_FIXTURE)
    monkeypatch.setattr(patcher, "GPU_MODEL_RUNNER_PATH", source)
    monkeypatch.setattr(patcher, "_FR13_FIXED32_MODE", "tail6_fixed32")
    assert patcher._patch_gpu_model_runner_fixed32_final_full_preseed()

    fake_gdn = SimpleNamespace(
        _fr13_fixed32_final_full_preseed_postcheck_required=lambda mode: True,
        _fr13_fixed32_final_full_preseed_needed=lambda *args: False,
        _fr13_fixed32_warm_final_full_postprocess=lambda vocab: None,
        _fr13_fixed32_assert_final_full_preseed_ready=lambda batch: (
            (_ for _ in ()).throw(RuntimeError("stale fixed32 lease"))
        ),
    )
    _install_fake_gdn(monkeypatch, fake_gdn)
    namespace = {
        "CUDAGraphMode": _CUDAGraphMode,
        "SimpleNamespace": SimpleNamespace,
        "_fr13_cfwd_timer": lambda: None,
        "_fr13_dfwd_timer": lambda: None,
    }
    exec(source.read_text(), namespace)
    runner = namespace["Runner"](1)
    descriptor = SimpleNamespace(
        num_tokens=32,
        num_reqs=1,
        uniform=True,
        has_lora=False,
        num_active_loras=0,
    )
    with pytest.raises(RuntimeError, match="stale fixed32 lease"):
        runner._warmup_and_capture(descriptor, _CUDAGraphMode.FULL)

    assert len(runner.calls) == 1
    assert (
        runner.calls[0][1]["cudagraph_runtime_mode"]
        is _CUDAGraphMode.NONE
    )


def test_default_eager_runtime_has_no_b4_diagnostic_boot_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "gpu_model_runner.py"
    source.write_text(_GPU_RUNNER_FIXTURE)
    monkeypatch.setattr(patcher, "GPU_MODEL_RUNNER_PATH", source)
    monkeypatch.setattr(patcher, "_FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.setattr(
        patcher,
        "_FR13_FIXED32_BATCH_GDN_BYTE_AB",
        "0",
    )
    assert patcher._patch_gpu_model_runner_fixed32_final_full_preseed()

    text = source.read_text(encoding="utf-8")
    assert "FR13_FIXED32_EAGER_B4_BOOT_WARM" not in text
    namespace = {
        "CUDAGraphMode": _CUDAGraphMode,
        "SimpleNamespace": SimpleNamespace,
        "logger": SimpleNamespace(warning=lambda *args: None),
    }
    exec(text, namespace)
    runner = namespace["Runner"](1, max_num_seqs=1)
    assert runner.capture_model() == 0
    assert runner.calls == []


def test_eager_b4_diagnostic_boot_is_zero_traffic_and_boundary_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "gpu_model_runner.py"
    source.write_text(_GPU_RUNNER_FIXTURE)
    monkeypatch.setattr(patcher, "GPU_MODEL_RUNNER_PATH", source)
    monkeypatch.setattr(patcher, "_FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.setattr(
        patcher,
        "_FR13_FIXED32_BATCH_GDN_BYTE_AB",
        "1",
    )
    assert patcher._patch_gpu_model_runner_fixed32_final_full_preseed()

    state: dict[str, object] = {
        "observed": None,
        "pending": None,
        "work_census": [],
        "api_requests": [],
        "boot_warm": None,
    }

    def needed(*descriptor: object) -> bool:
        state["descriptor"] = descriptor
        return True

    def warm(vocab_size: int) -> None:
        assert state["observed"] is None
        assert state["pending"] is None
        state["boot_warm"] = {
            "ready": True,
            "capacity": 4,
            "vocab_size": vocab_size,
            "observed_event_absent": True,
            "pending_event_absent": True,
        }

    def assert_ready(batch: int) -> None:
        evidence = state["boot_warm"]
        assert isinstance(evidence, dict)
        assert evidence["ready"] is True
        assert evidence["capacity"] == batch
        assert state["work_census"] == []
        assert state["api_requests"] == []
        state["generation_1_boundary_ready"] = True

    fake_gdn = SimpleNamespace(
        _fr13_fixed32_final_full_preseed_needed=needed,
        _fr13_fixed32_warm_final_full_postprocess=warm,
        _fr13_fixed32_assert_final_full_preseed_ready=assert_ready,
    )
    _install_fake_gdn(monkeypatch, fake_gdn)
    namespace = {
        "CUDAGraphMode": _CUDAGraphMode,
        "SimpleNamespace": SimpleNamespace,
        "logger": SimpleNamespace(warning=lambda *args: None),
        "_fr13_cfwd_timer": lambda: state.update(cfwd_ready=True),
        "_fr13_dfwd_timer": lambda: state.update(dfwd_ready=True),
    }
    text = source.read_text(encoding="utf-8")
    assert "FR13_FIXED32_EAGER_B4_BOOT_WARM" in text
    exec(text, namespace)
    runner = namespace["Runner"](1)
    assert runner.capture_model() == 0

    assert state["descriptor"] == ("FULL", 128, 4, True, False, 0)
    assert len(runner.calls) == 1
    num_tokens, kwargs = runner.calls[0]
    assert num_tokens == 128
    assert kwargs == {
        "cudagraph_runtime_mode": _CUDAGraphMode.NONE,
        "force_attention": True,
        "uniform_decode": True,
        "allow_microbatching": False,
        "skip_eplb": True,
        "remove_lora": False,
        "is_graph_capturing": True,
        "num_active_loras": 0,
        "profile_seq_lens": None,
    }
    assert state["generation_1_boundary_ready"] is True
    assert state["cfwd_ready"] is True
    assert state["dfwd_ready"] is True
    assert state["work_census"] == []
    assert state["api_requests"] == []


@pytest.mark.parametrize(
    ("mode", "max_num_seqs", "error"),
    (
        (_CUDAGraphMode.FULL, 4, "requires CUDAGraphMode.NONE"),
        (_CUDAGraphMode.NONE, 1, "requires max_num_seqs=4"),
    ),
)
def test_eager_b4_diagnostic_boot_rejects_wrong_runtime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: _CUDAGraphMode,
    max_num_seqs: int,
    error: str,
) -> None:
    source = tmp_path / "gpu_model_runner.py"
    source.write_text(_GPU_RUNNER_FIXTURE)
    monkeypatch.setattr(patcher, "GPU_MODEL_RUNNER_PATH", source)
    monkeypatch.setattr(patcher, "_FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.setattr(
        patcher,
        "_FR13_FIXED32_BATCH_GDN_BYTE_AB",
        "1",
    )
    assert patcher._patch_gpu_model_runner_fixed32_final_full_preseed()
    namespace = {
        "CUDAGraphMode": _CUDAGraphMode,
        "SimpleNamespace": SimpleNamespace,
        "logger": SimpleNamespace(warning=lambda *args: None),
    }
    exec(source.read_text(encoding="utf-8"), namespace)
    runner = namespace["Runner"](
        1,
        cudagraph_mode=mode,
        max_num_seqs=max_num_seqs,
    )
    with pytest.raises(RuntimeError, match=error):
        runner.capture_model()
    assert runner.calls == []


def test_postprocess_boot_warm_is_unmeasured_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _runtime_functions(
        "_fr13_fixed32_warm_final_full_postprocess",
        mode="tail6_fixed32",
    )
    capacity = 4
    calls: list[tuple[str, int]] = []
    with torch.inference_mode():
        taw_state = torch.zeros(1)
        tail_state = torch.zeros(1)
    namespace.update(
        {
            "_FR13_FIXED32_BOOT_WARM_EVIDENCE": None,
            "_FR13_FIXED32_OBSERVED_CURRENT": None,
            "_FR13_FIXED32_PENDING_EVENT": None,
            "_FR13_FIXED32_PRESEED_CAP": capacity,
            "_FR13_FIXED32_PRESEEDED_BATCHES": {1, 2, 3, 4},
            "_FR13_EAGER_PACK_STACKS": {"spec_idx": torch.empty(1)},
            "_fr13_fixed32_boot_preseed_allowed": lambda: True,
        }
    )
    def taw_warm(*_args: object, **kwargs: object) -> dict[str, object]:
        assert torch.is_inference_mode_enabled()
        taw_state.add_(1)
        taw_state.zero_()
        calls.append(("taw", int(kwargs["max_batch_size"])))
        return {
            "ready": True,
            "classification": "unmeasured_boot",
            "batches": (1, 2, 3, 4),
            "executions": 4,
            "cache_lease_current": True,
            "rng_state_restored": True,
            "staging_state_restored": True,
            "measured_state_restored": True,
        }

    taw = SimpleNamespace(fr13_fixed32_taw_warm_execute=taw_warm)
    tail_evidence = {
        "ready": True,
        "classification": "unmeasured_boot",
        "hardware_scope": "device_postprocess_kernels",
        "wrapper_bookkeeping_warmed": False,
        "copy_source_dtype": "torch.int64",
        "copy_destination_dtype": "torch.int32",
        "batches": (1, 2, 3, 4),
        "output_copy_pairs": 4,
        "slot_copy_pairs": 10,
        "spec_copy_pairs": 4,
        "flags_zero_fills": 1,
        "persistent_copy_state_restored": True,
        "flags_state_restored": True,
    }
    committer_evidence = {
            "ready": True,
            "classification": "unmeasured_boot",
            "batches": (1, 2, 3, 4),
            "replays": 4,
            "conv_commit_direct_launches": 4,
            "conv_commit_gather_launches": 0,
            "conv_commit_scatter_launches": 0,
            "route_lease_current": True,
            "bank_state_restored": True,
            "conv_bank_state_restored": True,
            "conv_staging_state_restored": True,
            "alias_destination_contract": "exact_alias_only_16x3",
            "input_state_restored": True,
            "measured_state_restored": True,
            "scratch_overwrite_proven": True,
    }
    def tail_warm(*_args: object) -> tuple[dict[str, object], dict[str, object]]:
        assert torch.is_inference_mode_enabled()
        tail_state.add_(1)
        tail_state.zero_()
        calls.append(("tail", capacity))
        return dict(tail_evidence), dict(committer_evidence)

    namespace["_fr13_fixed32_warm_device_postprocess_tail"] = tail_warm
    monkeypatch.setitem(sys.modules, "_fr13_device_multidraft_kernel", taw)

    warm = namespace["_fr13_fixed32_warm_final_full_postprocess"]
    assert not torch.is_inference_mode_enabled()
    evidence = warm(248320)
    assert not torch.is_inference_mode_enabled()
    assert taw_state.item() == 0
    assert tail_state.item() == 0
    assert evidence == {
        "ready": True,
        "classification": "unmeasured_boot",
        "mode": "tail6_fixed32",
        "capacity": 4,
        "vocab_size": 248320,
        "batches": (1, 2, 3, 4),
        "taw_executions": 4,
        "hardware_scope": "device_postprocess_kernels",
        "wrapper_bookkeeping_warmed": False,
        "copy_source_dtype": "torch.int64",
        "copy_destination_dtype": "torch.int32",
        "output_copy_pairs": 4,
        "slot_copy_pairs": 10,
        "spec_copy_pairs": 4,
        "flags_zero_fills": 1,
        "persistent_copy_state_restored": True,
        "flags_state_restored": True,
        "conv_commit_direct_launches": 4,
        "conv_commit_gather_launches": 0,
        "conv_commit_scatter_launches": 0,
        "committer_replays": 4,
        "observed_event_absent": True,
        "pending_event_absent": True,
    }
    assert calls == [("taw", 4), ("tail", 4)]
    assert warm(248320) == evidence
    assert calls == [("taw", 4), ("tail", 4)]

    namespace["_FR13_FIXED32_BOOT_WARM_EVIDENCE"] = None
    namespace["_FR13_FIXED32_OBSERVED_CURRENT"] = object()
    with pytest.raises(RuntimeError, match="measured/capture state"):
        warm(248320)


@pytest.mark.parametrize("committer_fails", (False, True))
def test_postprocess_tail_uses_int32_destinations_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
    committer_fails: bool,
) -> None:
    namespace = _runtime_functions(
        "_fr13_fixed32_warm_device_postprocess_tail",
        mode="tail6_fixed32",
    )
    capacity = 4
    slot_paths = torch.arange(64, dtype=torch.int32).view(4, 16)
    slot_lens = torch.arange(4, dtype=torch.int32)
    spec_paths = (slot_paths + 100).clone()
    spec_lens = (slot_lens + 100).clone()
    flags = torch.arange(8, dtype=torch.int32).view(4, 2)
    saved = tuple(
        tensor.clone()
        for tensor in (
            slot_paths,
            slot_lens,
            spec_paths,
            spec_lens,
            flags,
        )
    )
    namespace.update(
        {
            "_LUMO_FA_FIXED32_SLOT_PATHS": slot_paths,
            "_LUMO_FA_FIXED32_SLOT_LENS": slot_lens,
            "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR": spec_paths,
            "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR": spec_lens,
            "_FR13_EAGER_PACK_STACKS": {"flags": flags},
        }
    )
    allocation_dtypes: list[torch.dtype] = []

    class _TorchProxy:
        int32 = torch.int32
        int64 = torch.int64
        cuda = torch.cuda

        @staticmethod
        def is_tensor(value: object) -> bool:
            return torch.is_tensor(value)

        @staticmethod
        def equal(left: torch.Tensor, right: torch.Tensor) -> bool:
            return torch.equal(left, right)

        @staticmethod
        def empty(*args: object, **kwargs: object) -> torch.Tensor:
            allocation_dtypes.append(kwargs["dtype"])  # type: ignore[arg-type]
            return torch.empty(*args, **kwargs)

    namespace["torch"] = _TorchProxy
    taw = SimpleNamespace(
        fr13_fixed32_taw_warm_products=lambda *args, **kwargs: (
            torch.arange(
                int(kwargs["batch_size"]) * 32,
                dtype=torch.int64,
            ).view(int(kwargs["batch_size"]), 32),
            torch.ones(int(kwargs["batch_size"]), dtype=torch.int64),
            torch.arange(
                int(kwargs["batch_size"]) * 16,
                dtype=torch.int64,
            ).view(int(kwargs["batch_size"]), 16),
            torch.ones(int(kwargs["batch_size"]), dtype=torch.int64),
            torch.arange(
                int(kwargs["batch_size"]), dtype=torch.int64
            ),
        )
    )
    kernel = ModuleType("lumo_flywheel_serving.fr10_gdn_tree_kernel")

    def _warm_committer() -> dict[str, object]:
        if committer_fails:
            raise RuntimeError("injected committer failure")
        return {"ready": True}

    kernel.warm_fixed32_committer_graphs_all_batches = _warm_committer
    package = ModuleType("lumo_flywheel_serving")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "lumo_flywheel_serving", package)
    monkeypatch.setitem(
        sys.modules,
        "lumo_flywheel_serving.fr10_gdn_tree_kernel",
        kernel,
    )

    warm_tail = namespace["_fr13_fixed32_warm_device_postprocess_tail"]
    if committer_fails:
        with pytest.raises(RuntimeError, match="injected committer failure"):
            warm_tail(taw, torch.device("cpu"), capacity, 248320)
    else:
        evidence, committer = warm_tail(
            taw, torch.device("cpu"), capacity, 248320
        )
        assert evidence["copy_source_dtype"] == "torch.int64"
        assert evidence["copy_destination_dtype"] == "torch.int32"
        assert evidence["output_copy_pairs"] == capacity
        assert committer == {"ready": True}

    assert allocation_dtypes == [torch.int32] * (2 * capacity)
    for current, original in zip(
        (slot_paths, slot_lens, spec_paths, spec_lens, flags),
        saved,
        strict=True,
    ):
        assert torch.equal(current, original)


def test_committer_warmup_readiness_binds_full_restore_contract() -> None:
    route = object()
    conv_state = object()
    evidence = {
        "ready": True,
        "classification": "unmeasured_boot",
        "mode": "tail6_fixed32",
        "max_batch_size": 4,
        "batches": (1, 2, 3, 4),
        "replays": 4,
        "conv_commit_direct_launches": 4,
        "conv_commit_gather_launches": 0,
        "conv_commit_scatter_launches": 0,
        "route_lease_current": True,
        "bank_state_restored": True,
        "conv_bank_state_restored": True,
        "conv_staging_state_restored": True,
        "alias_destination_contract": "exact_alias_only_16x3",
        "input_state_restored": True,
        "measured_state_restored": True,
        "scratch_overwrite_proven": True,
        "scratch_restored": (
            "accepted_paths",
            "accepted_lens",
            "node_mat",
            "qbuf",
        ),
        "scratch_fully_overwritten": (
            "abuf",
            "bbuf",
            "kbuf",
            "vbuf",
            "ssi",
        ),
        "scratch_immutable": (
            "cu",
            "path_offsets",
            "batch_offsets",
            "graph",
            "scratch",
        ),
    }
    namespace = _kernel_functions(
        "fixed32_committer_warmup_counters",
        namespace={
            "_FR13_FIXED32_MODE": "tail6_fixed32",
            "_FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY": 4,
            "_FR13_FIXED32_COMMITTER_FAST_ROUTE": {"state": route},
            "_FR13_FIXED32_CONV_PREGATHER": {"state": conv_state},
            "_FR13_FIXED32_COMMITTER_WARMUP": {
                "route": route,
                "conv_state": conv_state,
                "evidence": evidence,
            },
        },
    )
    counters = namespace["fixed32_committer_warmup_counters"]
    assert counters()["ready"] is True
    evidence["scratch_overwrite_proven"] = False
    assert counters()["ready"] is False


def test_committer_graph_body_fully_overwrites_declared_scratch() -> None:
    source = KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    graph_body = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_committer_graph_body"
    )
    body_source = ast.get_source_segment(source, graph_body)
    assert body_source is not None
    for statement in (
        'state["abuf"].fill_(-1e4)',
        'state["bbuf"].zero_()',
        'state["kbuf"].zero_()',
        'state["vbuf"].zero_()',
        'state["ssi"].fill_(state["scratch"])',
        "node_mat[:, 0] = 0",
        "node_mat[:, 1:] = (",
    ):
        assert statement in body_source


@pytest.mark.parametrize("mode", ("tail6_fixed32", "hydra27_fixed32"))
@pytest.mark.parametrize(("capacity", "batch"), ((1, 1), (4, 4)))
def test_preseed_postcondition_binds_all_current_bank_aliases(
    mode: str,
    capacity: int,
    batch: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _runtime_functions(
        "_fr13_fixed32_assert_final_full_preseed_ready",
        mode=mode,
    )
    order = tuple(f"gdn.{index}" for index in range(48))
    ssm_banks = tuple(object() for _ in order)
    conv_banks = tuple(object() for _ in order)
    layers = {
        name: SimpleNamespace(
            _fr13_replay_ssm_state=ssm_banks[index],
            _fr13_replay_conv_state=conv_banks[index],
        )
        for index, name in enumerate(order)
    }
    staging = torch.empty((48, capacity, 10))
    commit_ssi = object()
    accepted_paths = object()
    accepted_lens = object()
    pregather_state = {
        "mode": mode,
        "banks": conv_banks,
        "ssm_banks": ssm_banks,
        "layer_order": order,
        "staging": staging,
        "row_elems": 10,
        "anchor": torch.empty(1),
        "commit_spec_state_indices": commit_ssi,
        "accepted_paths": accepted_paths,
        "accepted_lens": accepted_lens,
        "contract": {
            "commit_route": "fixed32_direct_source_col0",
            "commit_bank_overlap_policy": "exact_alias_only_16x3",
            "commit_bank_partial_overlap": False,
            "commit_bank_alias_groups": 16,
            "commit_bank_alias_width": 3,
            "commit_bank_destination_guard": "alias_row_unique",
            "commit_null_row_rejected": True,
        },
    }
    committer_state = {"banks": ssm_banks}
    kernel = ModuleType("lumo_flywheel_serving.fr10_gdn_tree_kernel")
    kernel._FR13_FIXED32_CONV_PREGATHER = {"state": pregather_state}
    kernel._FR13_FIXED32_COMMITTER_FAST_ROUTE = {
        "state": committer_state
    }
    kernel.audit_fixed32_conv_commit_lease = lambda: {
        "lease_audited": True,
        "route": "fixed32_direct_source_col0",
    }
    kernel.fixed32_conv_col0_pregather_counters = lambda: {
        "preseeded": True,
        "pointer_entries": 48,
        "max_batch_size": capacity,
        "preseeded_batches": tuple(range(1, capacity + 1)),
    }
    kernel.fixed32_committer_counters = lambda: {
        "captures": capacity,
        "preseeded_graphs": capacity,
        "preseeded_batches": tuple(range(1, capacity + 1)),
        "required_capacity": capacity,
        "all_batches_ready": True,
    }
    kernel.fixed32_committer_warmup_counters = lambda: {
        "ready": True,
        "classification": "unmeasured_boot",
        "mode": mode,
        "max_batch_size": capacity,
        "batches": tuple(range(1, capacity + 1)),
        "replays": capacity,
        "conv_commit_direct_launches": capacity,
        "conv_commit_gather_launches": 0,
        "conv_commit_scatter_launches": 0,
        "route_lease_current": True,
        "bank_state_restored": True,
        "conv_bank_state_restored": True,
        "conv_staging_state_restored": True,
        "alias_destination_contract": "exact_alias_only_16x3",
        "input_state_restored": True,
        "measured_state_restored": True,
        "scratch_overwrite_proven": True,
    }
    package = ModuleType("lumo_flywheel_serving")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "lumo_flywheel_serving", package)
    monkeypatch.setitem(
        sys.modules,
        "lumo_flywheel_serving.fr10_gdn_tree_kernel",
        kernel,
    )
    taw = SimpleNamespace(
        fr13_fixed32_taw_preseeded_counts=lambda *args, **kwargs: (
            torch.full((int(kwargs["batch_size"]),), 31, dtype=torch.int32)
        ),
        fr13_fixed32_taw_warmup_counters=lambda *args, **kwargs: {
            "ready": True,
            "classification": "unmeasured_boot",
            "mode": mode,
            "valid_mask": namespace["_FR13_FIXED32_VALID_MASK"],
            "max_batch_size": capacity,
            "vocab_size": int(kwargs["vocab_size"]),
            "batches": tuple(range(1, capacity + 1)),
            "executions": capacity,
            "cache_lease_current": True,
            "rng_state_restored": True,
            "staging_state_restored": True,
            "measured_state_restored": True,
        },
    )
    monkeypatch.setitem(sys.modules, "_fr13_device_multidraft_kernel", taw)
    namespace.update(
        {
            "_FR13_FIXED32_PRESEED_CAP": capacity,
            "_FR13_FIXED32_PRESEEDED_BATCHES": set(
                range(1, capacity + 1)
            ),
            "_FR13_EAGER_PACK_STACKS": {
                "layer_order": order,
                "spec_idx": commit_ssi,
            },
            "_FR13_REPLAY_LAYERS": layers,
            "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR": accepted_paths,
            "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR": accepted_lens,
            "_FR13_FIXED32_BOOT_WARM_EVIDENCE": {
                "ready": True,
                "classification": "unmeasured_boot",
                "mode": mode,
                "capacity": capacity,
                "vocab_size": 248320,
                "batches": tuple(range(1, capacity + 1)),
                "taw_executions": capacity,
                "hardware_scope": "device_postprocess_kernels",
                "wrapper_bookkeeping_warmed": False,
                "copy_source_dtype": "torch.int64",
                "copy_destination_dtype": "torch.int32",
                "output_copy_pairs": capacity,
                "slot_copy_pairs": capacity * (capacity + 1) // 2,
                "spec_copy_pairs": capacity,
                "flags_zero_fills": 1,
                "persistent_copy_state_restored": True,
                "flags_state_restored": True,
                "conv_commit_direct_launches": capacity,
                "conv_commit_gather_launches": 0,
                "conv_commit_scatter_launches": 0,
                "committer_replays": capacity,
                "observed_event_absent": True,
                "pending_event_absent": True,
            },
        }
    )
    assert_ready = namespace[
        "_fr13_fixed32_assert_final_full_preseed_ready"
    ]
    assert_ready(batch)

    namespace["_FR13_EAGER_PACK_STACKS"]["spec_idx"] = object()
    with pytest.raises(RuntimeError, match="did not publish lease"):
        assert_ready(batch)
    namespace["_FR13_EAGER_PACK_STACKS"]["spec_idx"] = commit_ssi

    namespace["_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR"] = object()
    with pytest.raises(RuntimeError, match="did not publish lease"):
        assert_ready(batch)
    namespace["_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR"] = accepted_paths

    namespace["_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR"] = object()
    with pytest.raises(RuntimeError, match="did not publish lease"):
        assert_ready(batch)
    namespace["_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR"] = accepted_lens

    pregather_state["ssm_banks"] = (*ssm_banks[:-1], object())
    with pytest.raises(RuntimeError, match="did not publish lease"):
        assert_ready(batch)
    pregather_state["ssm_banks"] = ssm_banks

    layers[order[-1]]._fr13_replay_conv_state = object()
    with pytest.raises(
        RuntimeError,
        match="did not publish lease",
    ):
        assert_ready(batch)
