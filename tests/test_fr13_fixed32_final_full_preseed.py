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
    def __init__(self, num_warmups):
        self.compilation_config = SimpleNamespace(
            cudagraph_num_of_warmups=num_warmups
        )
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
        _fr13_fixed32_assert_final_full_preseed_ready=assert_ready,
    )
    _install_fake_gdn(monkeypatch, fake_gdn)
    namespace = {
        "CUDAGraphMode": _CUDAGraphMode,
        "SimpleNamespace": SimpleNamespace,
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
        _fr13_fixed32_assert_final_full_preseed_ready=lambda batch: (
            (_ for _ in ()).throw(RuntimeError("stale fixed32 lease"))
        ),
    )
    _install_fake_gdn(monkeypatch, fake_gdn)
    namespace = {
        "CUDAGraphMode": _CUDAGraphMode,
        "SimpleNamespace": SimpleNamespace,
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
        "layer_order": order,
        "staging": staging,
        "row_elems": 10,
        "anchor": torch.empty(1),
        "commit_spec_state_indices": commit_ssi,
        "accepted_paths": accepted_paths,
        "accepted_lens": accepted_lens,
        "contract": {
            "commit_route": "fixed32_two_launch_col0",
            "commit_bank_nonoverlap": True,
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
        "route": "fixed32_two_launch_col0",
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
        )
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

    layers[order[-1]]._fr13_replay_conv_state = object()
    with pytest.raises(
        RuntimeError,
        match="did not publish lease",
    ):
        assert_ready(batch)
