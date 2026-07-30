from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


_PATCHER_PATH = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py")
_PATCHER_SPEC = importlib.util.spec_from_file_location(
    "fr13_runrow_bias_patcher",
    _PATCHER_PATH,
)
assert _PATCHER_SPEC is not None and _PATCHER_SPEC.loader is not None
patcher = importlib.util.module_from_spec(_PATCHER_SPEC)
_PATCHER_SPEC.loader.exec_module(patcher)


_MAMBA_UTILS_FIXTURE = """\
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
)


def _fr10_tree_accept_token_bias(
    req_id: str,
    batch_index: int,
    linear_bias: int,
    *,
    phase: str,
) -> int:
    if not _fr10_tree_mamba_mode_active():
        return int(linear_bias)
    path = _FR10_TREE_ACCEPTED_PATH_BY_REQ_ID.get(str(req_id))
    if not path:
        raise RuntimeError("host accepted path missing")
    return int(path[int(linear_bias)])
"""


class _HostLookupTouched(dict[str, list[int]]):
    def get(self, key: str, default: object = None) -> object:
        raise AssertionError(f"host accepted-path lookup touched for {key}")


def _load_patched_bias_helper(source: str) -> object:
    parsed = ast.parse(source)
    helper = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr10_tree_accept_token_bias"
    )
    namespace = {
        "os": os,
        "_fr13_mecopy": SimpleNamespace(_FR13_IN_PREPROCESS=None),
        "_fr10_tree_mamba_mode_active": lambda: True,
        "_FR10_TREE_ACCEPTED_PATH_BY_REQ_ID": _HostLookupTouched(),
    }
    exec(
        compile(
            ast.Module(body=[helper], type_ignores=[]),
            "<patched-mamba-utils>",
            "exec",
        ),
        namespace,
    )
    return namespace


def test_runrow_bias_bypasses_host_path_at_block_rollover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mamba_utils = tmp_path / "mamba_utils.py"
    mamba_utils.write_text(_MAMBA_UTILS_FIXTURE)
    monkeypatch.setattr(patcher, "MAMBA_UTILS_PATH", mamba_utils)

    assert patcher._patch_mamba_utils_preprocess_context_flag()
    patched = mamba_utils.read_text()
    namespace = _load_patched_bias_helper(patched)
    helper = namespace["_fr10_tree_accept_token_bias"]
    phase_state = namespace["_fr13_mecopy"]

    monkeypatch.delenv("FR13_APC_COMMIT_TO_RUNNING_ROW", raising=False)
    assert helper("req-pre", 3, 1023, phase="preprocess") == 1023
    assert phase_state._FR13_IN_PREPROCESS is True

    assert helper("req-post", 3, 1024, phase="postprocess") == 1024
    assert phase_state._FR13_IN_PREPROCESS is False

    flag_pos = patched.index("_FR13_IN_PREPROCESS =")
    runrow_pos = patched.index("FR13_APC_COMMIT_TO_RUNNING_ROW")
    host_read_pos = patched.index("_FR10_TREE_ACCEPTED_PATH_BY_REQ_ID.get")
    assert flag_pos < runrow_pos < host_read_pos


def test_nonrunrow_preprocess_still_reaches_host_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mamba_utils = tmp_path / "mamba_utils.py"
    mamba_utils.write_text(_MAMBA_UTILS_FIXTURE)
    monkeypatch.setattr(patcher, "MAMBA_UTILS_PATH", mamba_utils)
    assert patcher._patch_mamba_utils_preprocess_context_flag()
    namespace = _load_patched_bias_helper(mamba_utils.read_text())

    monkeypatch.setenv("FR13_APC_COMMIT_TO_RUNNING_ROW", "0")
    monkeypatch.setenv("FR13_APC_CONV_FIX", "0")
    with pytest.raises(
        AssertionError,
        match="host accepted-path lookup touched",
    ):
        namespace["_fr10_tree_accept_token_bias"](
            "legacy-pre",
            0,
            1,
            phase="preprocess",
        )
