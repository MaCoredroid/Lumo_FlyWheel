from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace

import torch


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
SEQUENCE = REPO / "scripts" / "fr13_fixed32_floor_timers_seq.sh"
DRIVER = REPO / "scripts" / "fr13_b4_campaign_driver.sh"
FLOOR_GATE = REPO / "scripts" / "fr13_floor_gate.py"


def _eagle_consumption_new_snippet() -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_patch_eagle_tree_consumption_verify"
        ):
            for statement in ast.walk(node):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "new"
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                    and "FR13_DRAFT_VOCAB_ROOT" in statement.value.value
                ):
                    return statement.value.value
    raise AssertionError("root draft-vocabulary replacement snippet not found")


def _nested_function_factory(name: str, self_obj: object):
    snippet = _eagle_consumption_new_snippet()
    wrapped = "class _Holder:\n    def propose(self):\n" + snippet
    tree = ast.parse(wrapped)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    source = ast.unparse(function)
    factory_source = (
        "def _factory(self):\n"
        + textwrap.indent(source, "    ")
        + f"\n    return {name}\n"
    )
    namespace: dict[str, object] = {}
    exec(factory_source, namespace)
    return namespace["_factory"](self_obj)


def test_root_mode_is_default_off_strict_and_fixed32_only() -> None:
    snippet = _eagle_consumption_new_snippet()

    assert 'os.environ.get(\n                "FR13_DRAFT_VOCAB_ROOT", "0"' in snippet
    assert '_fr13_dvk_root_raw not in ("0", "1")' in snippet
    assert "_fr13_dvk_root and not _fr13_is_fixed32" in snippet
    assert "_fr13_dvk_root and not _fr13_single_logits" in snippet
    assert "_fr13_dvk_root and _fr13_selfcheck" in snippet
    assert "_fr13_dvk_root and _fr13_dvk_configured <= 0" in snippet


def test_subset_setup_precedes_and_remaps_every_root_selection() -> None:
    snippet = _eagle_consumption_new_snippet()

    assert snippet.index("def _fr13_dvk_logits") < snippet.index(
        "_fr10_consumes_root_leaf ="
    )
    assert snippet.index("if not _fr13_dvk_root:") > snippet.index(
        "_fr10_wide_topk[0] = _fr13_dvk_real_ids("
    )
    assert (
        "_fr10_logits, _fr10_root_map = _fr13_dvk_logits(" in snippet
    )
    assert (
        "draft_token_ids = _fr13_dvk_real_ids(\n"
        "                    draft_token_ids, _fr10_root_map"
    ) in snippet
    assert (
        "_fr10_root_topk = _fr13_dvk_real_ids(\n"
        "                        _fr10_root_topk, _fr10_root_map"
    ) in snippet
    assert (
        "_fr10_wide_topk[0] = _fr13_dvk_real_ids(\n"
        "                        _fr10_wide_topk[0], _fr10_root_map"
    ) in snippet
    assert "[FR13_DRAFT_VOCAB_ROOT] engaged " in snippet


def test_every_loop_selection_uses_the_map_from_its_logits_call() -> None:
    snippet = _eagle_consumption_new_snippet()

    assert snippet.count(
        "_fr10_step_logits, _fr10_step_map = _fr13_dvk_logits("
    ) == 2
    assert snippet.count("_fr10_step_map = None") == 2
    assert "draft_token_ids, _fr10_step_map" in snippet
    assert snippet.count("_fr10_step_top2, _fr10_step_map") == 2
    assert (
        "_fr13_dg_wt = _fr13_dvk_real_ids(\n"
        "                            _fr13_dg_wt, _fr10_step_map"
    ) in snippet
    assert "\n            _fr13_dvk_map = (" not in snippet
    assert "if _fr13_dvk_map is not None:" not in snippet


def test_full_vocabulary_size_is_not_derived_from_subset_root_logits() -> None:
    snippet = _eagle_consumption_new_snippet()

    assert "_fr13_dvk_full = int(self.model.lm_head.weight.shape[0])" in snippet
    assert snippet.count(
        "_fr13_dvk, _fr13_full_vocab_size = _fr13_dvk_prepare()"
    ) == 1
    assert "_fr13_dvk, _ = _fr13_dvk_prepare()" in snippet
    assert snippet.count(
        "_fr10_logits.shape[-1]\n"
        "                                if _fr13_full_vocab_size is None\n"
        "                                else _fr13_full_vocab_size"
    ) == 1
    assert snippet.count(
        "_fr10_logits.shape[-1]\n"
        "                        if _fr13_full_vocab_size is None\n"
        "                        else _fr13_full_vocab_size"
    ) == 1


class _QuantMethod:
    def __init__(self, output: torch.Tensor | None, error: Exception | None = None):
        self.output = output
        self.error = error

    def apply(self, _head, _hidden, *, bias):
        assert bias is None
        if self.error is not None:
            raise self.error
        return self.output


class _Model:
    def __init__(self, full_logits: torch.Tensor):
        self.full_logits = full_logits
        self.calls = 0

    def compute_logits(self, _hidden):
        self.calls += 1
        return self.full_logits


def test_helper_pairs_subset_logits_with_map_and_full_fallback_with_none() -> None:
    subset_logits = torch.tensor([[1.0, 2.0]])
    full_logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    id_map = torch.tensor([7, 11])
    owner = SimpleNamespace(
        _fr13_dvk_shim=SimpleNamespace(
            quant_method=_QuantMethod(subset_logits)
        ),
        _fr13_dvk_map_t=id_map,
        model=_Model(full_logits),
    )
    helper = _nested_function_factory("_fr13_dvk_logits", owner)

    logits, returned_map = helper(torch.zeros(1, 3))

    assert logits is subset_logits
    assert returned_map is id_map
    assert owner.model.calls == 0

    owner._fr13_dvk_shim.quant_method = _QuantMethod(
        None, RuntimeError("forced apply failure")
    )
    fallback_logits, fallback_map = helper(torch.zeros(1, 3))

    assert fallback_logits is full_logits
    assert fallback_map is None
    assert owner._fr13_dvk_dead is True
    assert owner.model.calls == 1


def test_formal_campaign_binds_mode_and_runtime_engagement() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    sequence = SEQUENCE.read_text(encoding="utf-8")
    driver = DRIVER.read_text(encoding="utf-8")
    floor_gate = FLOOR_GATE.read_text(encoding="utf-8")

    assert "FR13_DRAFT_VOCAB_ROOT=${FR13_DRAFT_VOCAB_ROOT:-0}" in launcher
    assert 'case "$FR13_DRAFT_VOCAB_ROOT" in' in launcher
    assert "FR13_DRAFT_VOCAB_ROOT=1 requires FR13_FIXED32_MODE" in launcher
    assert '-e FR13_DRAFT_VOCAB_ROOT="$FR13_DRAFT_VOCAB_ROOT" \\' in launcher
    assert "FR13_DRAFT_VOCAB_ROOT=${FR13_DRAFT_VOCAB_ROOT:-0}" in sequence
    assert 'case "$FR13_DRAFT_VOCAB_ROOT" in' in sequence
    assert "0|1) export FR13_DRAFT_VOCAB_ROOT" in sequence
    assert '--draft-vocab-root "$FR13_DRAFT_VOCAB_ROOT" \\' in driver
    assert '"FR13_DRAFT_VOCAB_ROOT": str(draft_vocab_root)' in floor_gate
    assert '"--draft-vocab-root"' in floor_gate
    assert "DRAFT_VOCAB_SHIM_ENGAGED" in floor_gate
    assert "DRAFT_VOCAB_ROOT_ENGAGED" in floor_gate
    assert "DRAFT_VOCAB_DISABLED in log" in floor_gate


def test_drafter_replacement_snippet_compiles_as_a_method_body() -> None:
    snippet = _eagle_consumption_new_snippet()
    compile(
        "class _C:\n    def propose(self):\n" + snippet,
        "<fr13_draft_vocab_root_snippet>",
        "exec",
    )
