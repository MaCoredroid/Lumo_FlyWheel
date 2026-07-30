from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"


def test_fixed32_nvtx_profile_is_opt_in_and_phase_scoped() -> None:
    text = PATCHER.read_text(encoding="utf-8")

    assert '"FR13_FIXED32_NVTX_PROFILE", "0"' in text
    assert 'torch.cuda.nvtx.range_push("fr13.fixed32." + str(phase))' in text
    assert "torch.cuda.nvtx.range_pop()" in text
    assert 'torch.cuda.nvtx.range_push("fr13.fixed32.step")' in text
    assert "_fr13_fixed32_step_nvtx_close()" in text
    assert "_FR13_FIXED32_STEP_NVTX_THREAD" in text
    for phase in ("sfwd", "dfwd", "cfwd"):
        assert f"'{phase}', _fr13_{phase}_ev is not None" in text
        assert f"_fr13_{phase}_nvtx" in text
    assert "'postprocess', _fr13_sfwd_ev is not None" in text
    assert "_fr13_postprocess_nvtx" in text


def test_postprocess_closes_before_deferred_sample_phases() -> None:
    text = PATCHER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    patch_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_gpu_model_runner_sfwd_gpu_timer"
    )
    postprocess_inject = next(
        ast.literal_eval(node.value)
        for node in ast.walk(patch_function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "postprocess_end_inject"
            for target in node.targets
        )
    )
    assert postprocess_inject.index(
        "_fr13_fixed32_nvtx_end(_fr13_postprocess_nvtx)"
    ) < postprocess_inject.index("self.execute_model_state = ExecuteModelState(")
    assert "def _patch_gpu_model_runner_cfwd_gpu_timer()" in text
    assert "postprocess end anchor " in text
    assert "(ExecuteModelState assignment) not found" in text


def test_fixed32_nvtx_profile_flag_reaches_the_server() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert '-e FR13_FIXED32_NVTX_PROFILE="${FR13_FIXED32_NVTX_PROFILE:-0}"' in text
