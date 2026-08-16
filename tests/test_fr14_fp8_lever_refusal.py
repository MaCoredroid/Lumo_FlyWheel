"""FR14: the fp8-only GEMM levers must refuse a non-fp8 checkpoint.

Why this has teeth rather than being paranoia: the levers do not crash under a
non-fp8 checkpoint, they measure NOTHING. OPT-A / ``FR13_GB10_FP8_GEMV_CFG``
guards on ``weight_block_size == [128,128]``; ``FR13_FIXED32_B1_FP8_QUANT_REGCACHE``
targets ``per_token_group_quant.cu``; every non-``stock``
``FR13_FIXED32_CUTLASS_WAVE`` variant targets
``scaled_mm_blockwise_sm120_fp8.cu``. An arm named ``cat9-opta`` would boot,
pass every flag assertion, write a full evidence bundle and produce a lever
verdict about a code path it never entered.

Arm B's checkpoint declares ``quantization_config.quant_method == "modelopt"``
(ModelOpt / NVFP4), which is a DIFFERENT non-fp8 value from arm A's
``"compressed-tensors"``. The guard is a ``!= "fp8"`` test, so both are refused
-- but "should be fine, it's a != test" is exactly the reasoning that misses a
config the probe cannot parse. These tests run the launcher's OWN guard text,
extracted from the launcher so a copy cannot drift, against real and synthetic
configs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"

BEGIN = "# FR14 FP8-LEVER REFUSAL (2026-08-16)"
END = "unset _fr14_served_quant_method _fr14_armed_fp8_levers _fr14_quant_probe"

ARM_B_ROOT = Path("/models/qwen3.8-27b-nvfp4-radixark")
ARM_A_ROOT = Path("/models/qwen3.8-27b-nvfp4")
FP8_ROOT = Path("/models/qwen3.8-27b-fp8")


def _guard_script() -> str:
    text = LAUNCHER.read_text(encoding="utf-8")
    start = text.index(BEGIN)
    end = text.index(END, start) + len(END)
    body = text[start:end]
    assert "SERVED_MODEL_PATH" in body
    return "set -uo pipefail\n" + body + "\necho GUARD_PASSED\n"


def _run(model_root: Path, **env: str) -> subprocess.CompletedProcess[str]:
    environment = {"PATH": "/usr/bin:/bin", "SERVED_MODEL_PATH": str(model_root)}
    environment.update(env)
    return subprocess.run(
        ["bash", "-c", _guard_script()],
        capture_output=True,
        text=True,
        env=environment,
    )


def _synthetic(tmp_path: Path, quantization_config: object | None) -> Path:
    root = tmp_path / "model"
    root.mkdir(exist_ok=True)
    payload: dict[str, object] = {"architectures": ["X"]}
    if quantization_config is not None:
        payload["quantization_config"] = quantization_config
    (root / "config.json").write_text(json.dumps(payload), encoding="ascii")
    return root


ARMED = (
    {"FR13_FIXED32_CUTLASS_WAVE": "identity_wide256_fullgrid_b1"},
    {"FR13_GB10_FP8_GEMV_CFG": "1"},
    {"FR13_FIXED32_B1_FP8_QUANT_REGCACHE": "1"},
    {"FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO": "/tmp/x.so"},
    {"FR13_FIXED32_CUTLASS_WAVE_SO": "/tmp/y.so"},
)


@pytest.mark.parametrize("armed", ARMED)
def test_modelopt_checkpoint_refuses_every_fp8_lever(
    tmp_path: Path, armed: dict[str, str]
) -> None:
    root = _synthetic(
        tmp_path,
        {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"},
    )
    result = _run(root, **armed)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "FR14 REFUSAL" in result.stderr
    assert "quant_method      : modelopt" in result.stderr
    # The refusal names the specific lever, so the operator does not have to
    # guess which of five knobs tripped it.
    (name,) = armed
    assert name in result.stderr


def test_modelopt_checkpoint_boots_with_no_levers_armed(tmp_path: Path) -> None:
    root = _synthetic(tmp_path, {"quant_method": "modelopt"})
    result = _run(root)
    assert result.returncode == 0, result.stderr
    assert "GUARD_PASSED" in result.stdout


def test_fp8_checkpoint_still_arms_the_levers(tmp_path: Path) -> None:
    """The guard must not have become "refuse everything"."""
    root = _synthetic(
        tmp_path,
        {"quant_method": "fp8", "weight_block_size": [128, 128]},
    )
    result = _run(root, FR13_FIXED32_CUTLASS_WAVE="identity_wide256_fullgrid_b1")
    assert result.returncode == 0, result.stderr
    assert "GUARD_PASSED" in result.stdout


def test_compressed_tensors_checkpoint_is_also_refused(tmp_path: Path) -> None:
    root = _synthetic(tmp_path, {"quant_method": "compressed-tensors"})
    result = _run(root, FR13_GB10_FP8_GEMV_CFG="1")
    assert result.returncode == 2
    assert "quant_method      : compressed-tensors" in result.stderr


@pytest.mark.parametrize(
    "config",
    (None, {}, "not-a-mapping", {"quant_method": None}),
    ids=("absent", "empty", "not-a-mapping", "null-method"),
)
def test_unparseable_or_absent_quant_config_is_refused_not_assumed(
    tmp_path: Path, config: object | None
) -> None:
    """An unreadable config must not be silently treated as fp8.

    The probe swallows every exception and prints an empty string; the guard
    then compares "" != "fp8" and refuses. That is the fail-closed direction,
    and it is worth a test because the swallow is broad.
    """
    root = _synthetic(tmp_path, config)
    result = _run(root, FR13_FIXED32_CUTLASS_WAVE="identity_wide256_fullgrid_b1")
    assert result.returncode == 2
    assert "FR14 REFUSAL" in result.stderr


def test_missing_config_json_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    result = _run(root, FR13_GB10_FP8_GEMV_CFG="1")
    assert result.returncode == 2
    assert "<absent>" in result.stderr


@pytest.mark.parametrize(
    ("root", "expected_method", "refused"),
    (
        (ARM_B_ROOT, "modelopt", True),
        (ARM_A_ROOT, "compressed-tensors", True),
        (FP8_ROOT, "fp8", False),
    ),
    ids=("arm_b_radixark", "arm_a_unsloth", "fp8_baseline"),
)
def test_against_the_real_checkpoints_on_disk(
    root: Path, expected_method: str, refused: bool
) -> None:
    """Synthetic configs cannot catch a real file the probe mis-reads."""
    if not (root / "config.json").is_file():
        pytest.skip(f"checkpoint not present: {root}")
    result = _run(root, FR13_FIXED32_CUTLASS_WAVE="identity_wide256_fullgrid_b1")
    if refused:
        assert result.returncode == 2, result.stdout + result.stderr
        assert f"quant_method      : {expected_method}" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert "GUARD_PASSED" in result.stdout
