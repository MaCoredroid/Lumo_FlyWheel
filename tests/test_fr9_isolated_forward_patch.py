from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO_ROOT / "docker" / "patches" / "apply_fr9_isolated_forward_probe.py"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("apply_fr9_isolated_forward_probe", PATCH_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fr9_isolated_forward_helper_compiles() -> None:
    module = _load_patch_module()

    compile(module.HELPER_MODULE, str(module.HELPER_MODULE_PATH), "exec")


def test_fr9_isolated_forward_patch_applies_to_expected_anchors(tmp_path: Path) -> None:
    module = _load_patch_module()
    target_root = tmp_path / "site-packages"
    gpu_model_runner = target_root / module.GPU_MODEL_RUNNER
    gpu_worker = target_root / module.GPU_WORKER
    gpu_model_runner.parent.mkdir(parents=True)

    gpu_model_runner.write_text(
        "from vllm.v1.worker.p2b_debug_export import P2BDebugExporter\n"
        "\n"
        "class GPUModelRunner:\n"
        "    def __init__(self):\n"
        "        self.p2b_debug_exporter = P2BDebugExporter.from_env()\n"
        "\n"
        "    def sample_tokens(self):\n"
        "        self.p2b_debug_exporter.export_state_snapshots(runner=self)\n",
        encoding="utf-8",
    )
    gpu_worker.write_text(
        "from typing import Any\n"
        "\n"
        "class Worker:\n"
        "    def sleep(self, level: int = 1) -> None:\n"
        "        pass\n",
        encoding="utf-8",
    )

    applied = module.apply_patch_to_root(target_root, skip_version_check=True)

    assert applied == [
        "GPUModelRunner import FR9 probe",
        "GPUModelRunner construct FR9 probe",
        "GPUModelRunner sample_tokens FR9 hook",
        "Worker collective_rpc FR9 probe",
    ]
    patched_runner = gpu_model_runner.read_text(encoding="utf-8")
    patched_worker = gpu_worker.read_text(encoding="utf-8")
    assert "FR9IsolatedForwardProbe.from_env()" in patched_runner
    assert "fr9_isolated_forward_probe.maybe_run(" in patched_runner
    assert "def lumo_fr9_isolated_forward_probe" in patched_worker
    assert (target_root / module.HELPER_MODULE_PATH).is_file()


def test_dockerfile_runs_fr9_patch_after_p2b_patch() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.nvidia-vllm").read_text(encoding="utf-8")

    p2b_index = dockerfile.index("apply_p2b_vllm_debug_export.py")
    fr9_index = dockerfile.index("apply_fr9_isolated_forward_probe.py")
    assert p2b_index < fr9_index
