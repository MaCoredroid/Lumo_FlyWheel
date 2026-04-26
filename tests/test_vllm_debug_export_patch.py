from __future__ import annotations

import importlib.util
import sys
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
    assert '_env_value("LUMO_P2B_VLLM_DEBUG_EXPORT", "VLLM_LUMO_P2B_DEBUG_EXPORT")' in patch.HELPER_MODULE
    assert "VLLM_LUMO_P2B_DEBUG_EXPORT" in patch.HELPER_MODULE
    assert "VLLM_LUMO_P2B_DEBUG_EXPORT_DIR" in patch.HELPER_MODULE
    assert "VLLM_LUMO_P2B_DEBUG_PROBE_REQUEST_IDS" in patch.HELPER_MODULE
    assert "return cls.disabled()" in patch.HELPER_MODULE
    assert ") or bool(probe_request_ids)" in patch.HELPER_MODULE
    assert "if not self._matches_probe_request(req_id):" in patch.HELPER_MODULE
    assert '"*" in self.config.probe_request_ids' in patch.HELPER_MODULE
    assert "generated_token_index not in self.config.state_tokens" in patch.HELPER_MODULE
    assert "frozenset({1, 1024})" in patch.HELPER_MODULE
    assert "full_vocab_logits_before_sampling" in patch.HELPER_MODULE
    assert "self._exported_logits_count" in patch.HELPER_MODULE
    assert "def _next_generated_token_index(" in patch.HELPER_MODULE
    assert "qwen35_mamba_deltanet_recurrent_state" in patch.HELPER_MODULE


def test_nvidia_vllm_dockerfile_applies_repo_owned_patch() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY docker/patches/apply_p2b_vllm_debug_export.py" in dockerfile
    assert "RUN python3 /tmp/apply_p2b_vllm_debug_export.py" in dockerfile
