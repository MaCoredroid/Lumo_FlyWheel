"""FR14 arm B: the boot-time NVFP4 lm_head loader patch.

The patch itself edits four files inside the pinned container image, so it
cannot be executed against the real tree from the host. What CAN be checked
offline -- and is worth checking, because a silent failure here costs 6.695 ms
of a floor that is otherwise fiction -- is the patcher's own contract:

* every anchor is required exactly once, and a missing/duplicated anchor aborts
  the boot rather than half-applying;
* the fail-closed decision is BAKED at patch time, not re-read from os.environ
  in the curated EngineCore worker process (the fail-open class this repo has
  been bitten by repeatedly);
* the emitted source is valid Python;
* it is idempotent within a mode and REFUSES to cross modes.

The fixtures below are minimal files containing the exact anchor strings the
patcher matches, built from the patcher's own module constants so the test
cannot drift away from the thing it is testing.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr14_patch_nvfp4_lmhead.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"


def _patcher_constants() -> dict[str, str]:
    """Read the anchor literals out of the patcher WITHOUT importing it.

    Importing would immediately try to read the container's site-packages, so
    the module body is parsed and only the string assignments are evaluated.
    """
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    wanted = {"LMHEAD_NEEDLE", "MODELOPT_NEEDLE", "VPE_NEEDLE"}
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    missing = wanted - set(found)
    assert not missing, f"patcher no longer defines {sorted(missing)}"
    return found


def _build_fake_site_packages(root: Path, *, duplicate: str | None = None) -> Path:
    """A site-packages tree carrying exactly the four anchors, once each."""
    anchors = _patcher_constants()
    sp = root / "vllm"
    (sp / "model_executor" / "models").mkdir(parents=True)
    (sp / "model_executor" / "layers" / "quantization").mkdir(parents=True)

    lm_body = anchors["LMHEAD_NEEDLE"]
    if duplicate == "lm_head":
        lm_body = lm_body + lm_body
    head_file = (
        "import typing\n\n\n"
        "class _Stub:\n"
        "    def _build(self, config, prefix):\n"
        "        if True:\n"
        "            if True:\n"
        f"{lm_body}"
        "                return self.lm_head\n"
    )
    for name in ("qwen3_5.py", "qwen3_5_mtp.py"):
        (sp / "model_executor" / "models" / name).write_text(head_file, encoding="ascii")

    modelopt_body = anchors["MODELOPT_NEEDLE"]
    if duplicate == "modelopt":
        modelopt_body = modelopt_body + modelopt_body
    # The anchor is a 12-space `if` followed by an 8-space `return None`, so it
    # needs exactly one enclosing block inside the method, not two.
    (sp / "model_executor" / "layers" / "quantization" / "modelopt.py").write_text(
        "class _Cfg:\n"
        "    def get_quant_method(self, layer, prefix):\n"
        "        quant_algo = None\n"
        "        if True:\n"
        f"{modelopt_body}",
        encoding="ascii",
    )

    (sp / "model_executor" / "layers" / "vocab_parallel_embedding.py").write_text(
        "class _Emb:\n"
        "    def weight_loader(self, param, loaded_weight, output_dim=None):\n"
        f"{anchors['VPE_NEEDLE']}"
        "        raise AssertionError\n",
        encoding="ascii",
    )
    return sp


def _run_patcher(sp: Path, require: str | None) -> subprocess.CompletedProcess[str]:
    """Run the patcher with SP redirected at the fake tree."""
    driver = sp.parent / "_driver.py"
    driver.write_text(
        "import runpy\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "source = Path(sys.argv[1]).read_text()\n"
        "source = source.replace(\n"
        '    \'SP = Path("/usr/local/lib/python3.12/dist-packages/vllm")\',\n'
        "    'SP = Path(sys.argv[2])',\n"
        ")\n"
        "assert 'SP = Path(sys.argv[2])' in source, 'SP anchor moved'\n"
        "ns = {'__name__': '__main__', 'sys': sys}\n"
        "exec(compile(source, str(sys.argv[1]), 'exec'), ns)\n",
        encoding="ascii",
    )
    env = dict(os.environ)
    env.pop("FR14_REQUIRE_NVFP4_LMHEAD", None)
    if require is not None:
        env["FR14_REQUIRE_NVFP4_LMHEAD"] = require
    return subprocess.run(
        [sys.executable, str(driver), str(PATCHER), str(sp)],
        capture_output=True,
        text=True,
        env=env,
    )


def _emitted(sp: Path) -> str:
    return (sp / "model_executor" / "models" / "qwen3_5.py").read_text(encoding="ascii")


def test_patcher_module_is_importable_source() -> None:
    """A syntax error here would kill the boot 40 s into a container start."""
    spec = importlib.util.spec_from_file_location("fr14_patch_nvfp4_lmhead", PATCHER)
    assert spec is not None
    compile(PATCHER.read_text(encoding="utf-8"), str(PATCHER), "exec")


def test_required_mode_bakes_an_unconditional_raise(tmp_path: Path) -> None:
    """The fail-closed decision must not survive as an os.environ read.

    vLLM v1 builds the model in a separate worker whose environment is curated
    (this repo's own launcher notes only 14/66 FR13_* vars reach it). A runtime
    env read would therefore be fail-OPEN precisely when the env is dropped:
    the banner would still print and the assertion would silently not fire.
    """
    sp = _build_fake_site_packages(tmp_path)
    result = _run_patcher(sp, "1")
    assert result.returncode == 0, result.stderr

    text = _emitted(sp)
    assert "FR14_LMHEAD_QUANT_ROUTE_REQUIRED" in text
    assert 'if _fr14_qm != "ModelOptNvFp4LinearMethod":' in text
    assert "raise RuntimeError(" in text
    # The emitted enforcement must not depend on the worker's environment.
    assert "os.environ" not in text
    assert "FR14_REQUIRE_NVFP4_LMHEAD" in text  # named in the message only
    # quant_config is actually threaded through (G1).
    assert "quant_config=self.quant_config," in text
    compile(text, "qwen3_5.py", "exec")

    # The MTP drafter gets the identical treatment: it loads the TARGET head.
    mtp = (sp / "model_executor" / "models" / "qwen3_5_mtp.py").read_text()
    assert "FR14_LMHEAD_QUANT_ROUTE_REQUIRED" in mtp
    compile(mtp, "qwen3_5_mtp.py", "exec")


def test_permissive_mode_emits_no_raise(tmp_path: Path) -> None:
    sp = _build_fake_site_packages(tmp_path)
    result = _run_patcher(sp, "0")
    assert result.returncode == 0, result.stderr

    text = _emitted(sp)
    assert "FR14_LMHEAD_QUANT_ROUTE_PERMISSIVE" in text
    assert "raise RuntimeError(" not in text
    # The banner still fires, so a permissive boot is observable rather than silent.
    assert "FR14_LMHEAD_QUANT_ROUTE lm_head quant_method=%s" in text
    compile(text, "qwen3_5.py", "exec")


def test_patch_is_idempotent_within_a_mode(tmp_path: Path) -> None:
    sp = _build_fake_site_packages(tmp_path)
    assert _run_patcher(sp, "1").returncode == 0
    first = _emitted(sp)
    second_run = _run_patcher(sp, "1")
    assert second_run.returncode == 0, second_run.stderr
    assert "already-patched" in second_run.stdout
    assert _emitted(sp) == first


def test_patch_refuses_to_cross_enforcement_modes(tmp_path: Path) -> None:
    """Re-running permissive over a required tree must not silently no-op.

    The sentinel encodes the mode for exactly this reason: a bare
    "FR14_LMHEAD_QUANT_ROUTE" sentinel would match either emission and leave
    the first mode live while reporting success.
    """
    sp = _build_fake_site_packages(tmp_path)
    assert _run_patcher(sp, "1").returncode == 0
    crossed = _run_patcher(sp, "0")
    assert crossed.returncode != 0
    assert "OTHER enforcement mode" in (crossed.stderr + crossed.stdout)
    # The required-mode raise is still in place; nothing was downgraded.
    assert 'if _fr14_qm != "ModelOptNvFp4LinearMethod":' in _emitted(sp)


@pytest.mark.parametrize("duplicate", ("lm_head", "modelopt"))
def test_duplicated_anchor_aborts_rather_than_half_applying(
    tmp_path: Path, duplicate: str
) -> None:
    sp = _build_fake_site_packages(tmp_path, duplicate=duplicate)
    result = _run_patcher(sp, "1")
    assert result.returncode != 0
    assert "expected exactly 1" in (result.stderr + result.stdout)


def test_missing_anchor_aborts(tmp_path: Path) -> None:
    sp = _build_fake_site_packages(tmp_path)
    vpe = sp / "model_executor" / "layers" / "vocab_parallel_embedding.py"
    vpe.write_text("class _Emb:\n    pass\n", encoding="ascii")
    result = _run_patcher(sp, "1")
    assert result.returncode != 0
    assert "occurs 0 times" in (result.stderr + result.stdout)


def test_invalid_requirement_value_is_refused(tmp_path: Path) -> None:
    sp = _build_fake_site_packages(tmp_path)
    result = _run_patcher(sp, "yes")
    assert result.returncode != 0
    assert "must be exactly '0' or '1'" in (result.stderr + result.stdout)
    # Nothing was written.
    assert "FR14_LMHEAD_QUANT_ROUTE" not in _emitted(sp)


def test_launcher_runs_the_patch_fail_closed_by_default() -> None:
    """The production boot flow, not just the smoke script, applies the patch."""
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "export FR14_REQUIRE_NVFP4_LMHEAD=${FR14_REQUIRE_NVFP4_LMHEAD:-1}" in text
    assert "python3 /workspace/scripts/fr14_patch_nvfp4_lmhead.py\n" in text

    # Ordering is load-bearing: the lm_head patch must land after the fr10
    # tree/GDN patcher (which the DVK shim lives in) and before the FA2 patcher
    # and the serve line, so every downstream verifier sees a fully patched tree.
    fr10 = text.index("python3 /workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py\n")
    lmhead = text.index("python3 /workspace/scripts/fr14_patch_nvfp4_lmhead.py\n")
    fa2 = text.index("python3 /workspace/scripts/fr13_patch_fa2_tree_bias.py")
    serve = text.index("vllm serve $SERVED_MODEL_PATH")
    assert fr10 < lmhead < fa2 < serve

    # The env sweeper forwards FR<N>_* into the container, which is the only
    # reason the patcher can read the requirement at patch time.
    assert "grep -E '^(FR[0-9]+_|LUMO_|VLLM_)'" in text


def test_patch_source_is_pinned_in_the_runtime_manifest_closure() -> None:
    from scripts import fr13_runtime_manifest

    assert (
        "scripts/fr14_patch_nvfp4_lmhead.py"
        in fr13_runtime_manifest.FIXED32_HOST_SCRIPT_SOURCE
    )
