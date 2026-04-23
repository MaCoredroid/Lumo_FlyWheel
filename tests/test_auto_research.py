from __future__ import annotations

from pathlib import Path

from lumo_flywheel_serving.auto_research import OfflineAutoResearchRunner, SyntheticWorkloadDistribution
from lumo_flywheel_serving.registry import load_registry
from lumo_flywheel_serving.tuned_config import load_tuned_config_bundle


def _write_registry(path: Path) -> None:
    path.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    hf_revision: 2e1b21350ce589fcaafbb3c7d7eac526a7aed582
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.90
    max_num_batched_tokens: 8192
    max_num_seqs: 4
""",
        encoding="utf-8",
    )


def test_offline_auto_research_produces_bundle_for_proposal_ranking_family(tmp_path: Path) -> None:
    registry_path = tmp_path / "model_registry.yaml"
    _write_registry(registry_path)
    model_config = load_registry(registry_path)["qwen3.5-27b"]
    workload = SyntheticWorkloadDistribution.from_file(
        Path("benchmark_blueprints/families/proposal-ranking-manager-judgment/serving_workload.yaml"),
        model_config=model_config,
        family_id="proposal-ranking-manager-judgment",
    )
    runner = OfflineAutoResearchRunner(
        model_config=model_config,
        family_id="proposal-ranking-manager-judgment",
        output_root=tmp_path / "tuned_configs",
        workload=workload,
        iteration_cap=6,
    )

    result = runner.run()

    assert result.status == "produced_bundle"
    assert result.bundle_path is not None
    bundle = load_tuned_config_bundle(result.bundle_path)
    assert bundle.family_id == "proposal-ranking-manager-judgment"
    assert bundle.regression_guard["delta"] > 0
    assert bundle.vllm_config["gpu_memory_utilization"] <= 0.08
