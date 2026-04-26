from __future__ import annotations

import importlib.util
import logging
import pickle
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_SCRIPT = REPO_ROOT / "docker" / "patches" / "apply_p2b_vllm_debug_export.py"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.nvidia-vllm"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("p2b_vllm_debug_patch", PATCH_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


class _FakeTensor:
    def __init__(self, data, *, dtype: str = "torch.float16", device: str = "cuda:0"):
        self.data = data
        self.dtype = dtype
        self.device = device

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        if isinstance(self.data, list):
            return (len(self.data),)
        return ()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def detach(self):
        return self

    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def to(self, *_args, dtype=None, **_kwargs):
        return _FakeTensor(self.data, dtype=dtype or self.dtype, device="cpu")

    def contiguous(self):
        return self

    def __getitem__(self, item):
        if isinstance(item, int):
            return _FakeTensor(self.data[item], dtype=self.dtype, device=self.device)
        if isinstance(item, tuple) and item[0] is Ellipsis and isinstance(item[1], slice):
            token_slice = item[1]
            if self.ndim == 1:
                return _FakeTensor(self.data[token_slice], dtype=self.dtype, device=self.device)
            return _FakeTensor([row[token_slice] for row in self.data], dtype=self.dtype, device=self.device)
        raise TypeError(f"unsupported fake tensor index: {item!r}")


def _fake_torch_module() -> types.ModuleType:
    torch_module = types.ModuleType("torch")
    torch_module.Tensor = _FakeTensor
    torch_module.float32 = "torch.float32"
    torch_module.inference_mode = _FakeInferenceMode

    def save(payload, path):
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)

    def load(path, **_kwargs):
        with Path(path).open("rb") as handle:
            return pickle.load(handle)

    torch_module.save = save
    torch_module.load = load
    return torch_module


def _load_helper_module(helper_source: str):
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: logging.getLogger(name)
    vllm_module = types.ModuleType("vllm")
    torch_module = _fake_torch_module()
    old_vllm = sys.modules.get("vllm")
    old_logger = sys.modules.get("vllm.logger")
    old_torch = sys.modules.get("torch")
    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.logger"] = logger_module
    sys.modules["torch"] = torch_module
    module = types.ModuleType("p2b_debug_export_under_test")
    sys.modules[module.__name__] = module
    try:
        exec(helper_source, module.__dict__)
    finally:
        if old_vllm is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = old_vllm
        if old_logger is None:
            sys.modules.pop("vllm.logger", None)
        else:
            sys.modules["vllm.logger"] = old_logger
        if old_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = old_torch
    return module


def test_p2b_vllm_debug_export_patch_targets_expected_vllm_hooks(tmp_path: Path) -> None:
    patch = _load_patch_module()
    target = tmp_path / "vllm" / "v1" / "worker"
    target.mkdir(parents=True)
    gpu_model_runner = target / "gpu_model_runner.py"
    gpu_model_runner.write_text(
        "".join(replacement.before for replacement in patch.REPLACEMENTS),
        encoding="utf-8",
    )

    applied = patch.apply_patch_to_root(tmp_path, skip_version_check=True)

    assert applied == [replacement.label for replacement in patch.REPLACEMENTS]
    patched = gpu_model_runner.read_text(encoding="utf-8")
    assert "from vllm.v1.worker.p2b_debug_export import P2BDebugExporter" in patched
    assert "self.p2b_debug_exporter = P2BDebugExporter.from_env()" in patched
    assert "self.p2b_debug_exporter.export_logits(" in patched
    assert "self.p2b_debug_exporter.export_state_snapshots(runner=self)" in patched
    assert (target / "p2b_debug_export.py").is_file()


def test_p2b_vllm_debug_export_is_disabled_by_default_and_probe_keyed() -> None:
    patch = _load_patch_module()

    assert patch.PATCH_VERSION == "0.19.0"
    assert "LUMO_P2B_VLLM_DEBUG_EXPORT" in patch.DEBUG_ENV_VARS
    assert "LUMO_P2B_DEBUG_PROBE_REQUEST_IDS" in patch.DEBUG_ENV_VARS
    assert "LUMO_P2B_DEBUG_LOGITS_MAX_TOKENS" in patch.DEBUG_ENV_VARS
    assert "LUMO_P2B_DEBUG_LOGITS_MAX_EXPORTS" in patch.DEBUG_ENV_VARS
    assert '_env_value("LUMO_P2B_VLLM_DEBUG_EXPORT", "VLLM_LUMO_P2B_DEBUG_EXPORT")' in patch.HELPER_MODULE
    assert "VLLM_LUMO_P2B_DEBUG_EXPORT" in patch.HELPER_MODULE
    assert "VLLM_LUMO_P2B_DEBUG_EXPORT_DIR" in patch.HELPER_MODULE
    assert "VLLM_LUMO_P2B_DEBUG_PROBE_REQUEST_IDS" in patch.HELPER_MODULE
    assert "VLLM_LUMO_P2B_DEBUG_LOGITS_MAX_TOKENS" in patch.HELPER_MODULE
    assert "VLLM_LUMO_P2B_DEBUG_LOGITS_MAX_EXPORTS" in patch.HELPER_MODULE
    assert "return cls.disabled()" in patch.HELPER_MODULE
    assert ") or bool(probe_request_ids)" in patch.HELPER_MODULE
    assert "if not self._matches_probe_request(req_id):" in patch.HELPER_MODULE
    assert "fnmatchcase(req_id, pattern)" in patch.HELPER_MODULE
    assert "generated_token_index not in self.config.state_tokens" in patch.HELPER_MODULE
    assert "frozenset({1, 1024})" in patch.HELPER_MODULE
    assert "full_vocab_logits_before_sampling" in patch.HELPER_MODULE
    assert "self._exported_logits_count" in patch.HELPER_MODULE
    assert "self._written_logits_count" in patch.HELPER_MODULE
    assert "def _next_generated_token_index(" in patch.HELPER_MODULE
    assert "qwen35_mamba_deltanet_recurrent_state" in patch.HELPER_MODULE
    assert "qwen35_mamba_deltanet_recurrent_state_diagnostic" in patch.HELPER_MODULE
    assert "state_diag_req_" in patch.HELPER_MODULE


def test_p2b_vllm_debug_export_bounds_logits_and_records_source_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    patch = _load_patch_module()
    helper = _load_helper_module(patch.HELPER_MODULE)
    monkeypatch.setenv("LUMO_P2B_VLLM_DEBUG_EXPORT", "1")
    monkeypatch.setenv("LUMO_P2B_DEBUG_EXPORT_DIR", str(tmp_path))
    monkeypatch.setenv("LUMO_P2B_DEBUG_PROBE_REQUEST_IDS", "req-*")
    monkeypatch.setenv("LUMO_P2B_DEBUG_LOGITS_MAX_TOKENS", "4")
    monkeypatch.setenv("LUMO_P2B_DEBUG_LOGITS_MAX_EXPORTS", "1")

    exporter = helper.P2BDebugExporter.from_env()
    logits = _FakeTensor([list(range(10)), list(range(10, 20))])
    exporter.export_logits(
        logits=logits,
        req_ids=["req-alpha", "ignored"],
        output_token_ids=[[], []],
    )
    exporter.export_logits(
        logits=_FakeTensor([list(range(100, 110)), list(range(110, 120))]),
        req_ids=["req-alpha", "ignored"],
        output_token_ids=[[7], []],
    )

    exports = sorted(tmp_path.glob("logits_req_req-alpha_tok_*.pt"))
    assert [path.name for path in exports] == ["logits_req_req-alpha_tok_000001.pt"]
    payload = helper.torch.load(exports[0], weights_only=False)
    assert payload["kind"] == "full_vocab_logits_before_sampling"
    assert payload["request_id"] == "req-alpha"
    assert payload["generated_token_index"] == 1
    assert payload["source_shape"] == (10,)
    assert payload["source_dtype"] == "torch.float16"
    assert payload["source_numel"] == 10
    assert payload["logits_max_tokens"] == 4
    assert payload["logits_is_truncated"] is True
    assert payload["saved_shape"] == (4,)
    assert payload["saved_dtype"] == "torch.float32"
    assert payload["logits"].data == [0, 1, 2, 3]
    assert payload["logits"].dtype == "torch.float32"
    assert payload["logits"].device == "cpu"


def test_p2b_vllm_debug_export_state_uses_default_mamba_cache_block(
    monkeypatch, tmp_path: Path
) -> None:
    patch = _load_patch_module()
    helper = _load_helper_module(patch.HELPER_MODULE)
    monkeypatch.setenv("LUMO_P2B_VLLM_DEBUG_EXPORT", "1")
    monkeypatch.setenv("LUMO_P2B_DEBUG_EXPORT_DIR", str(tmp_path))
    monkeypatch.setenv("LUMO_P2B_DEBUG_PROBE_REQUEST_IDS", "req-state")
    monkeypatch.setenv("LUMO_P2B_DEBUG_STATE_TOKENS", "1")

    exporter = helper.P2BDebugExporter.from_env()
    req_state = types.SimpleNamespace(
        output_token_ids=[123],
        block_ids=[[7]],
        num_prompt_tokens=5,
        num_computed_tokens=6,
    )
    runner = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(mamba_cache_mode="none"),
        mamba_state_idx={},
        input_batch=types.SimpleNamespace(req_ids=["req-state"], num_reqs=1),
        requests={"req-state": req_state},
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(layer_names=["model.layers.0.linear_attn"])
            ]
        ),
        compilation_config=types.SimpleNamespace(
            static_forward_context={
                "model.layers.0.linear_attn": types.SimpleNamespace(
                    kv_cache=[
                        _FakeTensor(
                            [
                                [10, 11],
                                [20, 21],
                                [30, 31],
                                [40, 41],
                                [50, 51],
                                [60, 61],
                                [70, 71],
                                [80, 81],
                            ]
                        ),
                        _FakeTensor(
                            [
                                [100, 101],
                                [200, 201],
                                [300, 301],
                                [400, 401],
                                [500, 501],
                                [600, 601],
                                [700, 701],
                                [800, 801],
                            ]
                        ),
                    ]
                )
            }
        ),
    )
    runner._get_mamba_copy_bufs = lambda: types.SimpleNamespace(
        mamba_group_ids=[0],
        mamba_spec=types.SimpleNamespace(block_size=16),
    )

    exporter.export_state_snapshots(runner=runner)

    exports = sorted(tmp_path.glob("state_req_req-state_tok_*.pt"))
    assert [path.name for path in exports] == ["state_req_req-state_tok_000001.pt"]
    payload = helper.torch.load(exports[0], weights_only=False)
    layer_payload = payload["layers"]["model.layers.0.linear_attn"]
    assert payload["kind"] == "qwen35_mamba_deltanet_recurrent_state"
    assert payload["request_id"] == "req-state"
    assert payload["generated_token_index"] == 1
    assert payload["mamba_cache_mode"] == "none"
    assert payload["state_block_idx"] == 0
    assert payload["state_block_idx_source"] == "cache_mode_none_first_block"
    assert payload["request_num_prompt_tokens"] == 5
    assert payload["request_num_computed_tokens"] == 6
    assert payload["request_output_token_count"] == 1
    assert [entry["state_role"] for entry in layer_payload] == [
        "conv_state",
        "recurrent_ssm_state",
    ]
    assert layer_payload[1]["block_id"] == 7
    assert layer_payload[1]["source_shape"] == (2,)
    assert layer_payload[1]["source_dtype"] == "torch.float16"
    assert layer_payload[1]["source_device"] == "cuda:0"
    assert layer_payload[1]["saved_shape"] == (2,)
    assert layer_payload[1]["saved_dtype"] == "torch.float16"
    assert layer_payload[1]["tensor"].data == [800, 801]
    assert layer_payload[1]["tensor"].device == "cpu"


def test_p2b_vllm_debug_export_failures_are_non_strict_by_default(monkeypatch, tmp_path: Path) -> None:
    patch = _load_patch_module()
    helper = _load_helper_module(patch.HELPER_MODULE)
    monkeypatch.setenv("LUMO_P2B_VLLM_DEBUG_EXPORT", "1")
    monkeypatch.setenv("LUMO_P2B_DEBUG_EXPORT_DIR", str(tmp_path))
    monkeypatch.setenv("LUMO_P2B_DEBUG_PROBE_REQUEST_IDS", "*")
    monkeypatch.setattr(
        helper.torch,
        "save",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("disk full")),
    )

    exporter = helper.P2BDebugExporter.from_env()
    exporter.export_logits(logits=_FakeTensor([[1] * 8]), req_ids=["any"], output_token_ids=[[]])

    assert list(tmp_path.glob("*.tmp")) == []


def test_nvidia_vllm_dockerfile_applies_repo_owned_patch() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY docker/patches/apply_p2b_vllm_debug_export.py" in dockerfile
    assert "RUN python3 /tmp/apply_p2b_vllm_debug_export.py" in dockerfile
