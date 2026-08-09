"""FR13_MAMBA_SPEC_BLOCKS_CDIV patch-site gate (mamba spec-block right-sizing).

ONE patch-time flag, default OFF, byte-inert when OFF; the anchor-replace +
exact-count house pattern used by the neighbouring mamba patchers
(_patch_mamba_utils_preprocess_context_flag, _patch_mamba_state_utils_tree_conv
_node_copy).

Instrument under test:
  - scripts/fr10_phase4_patch_vllm_tree_gdn.py
      _fr13_mamba_spec_blocks_cdiv()                     strict 0/1 env read
      _patch_mamba_abstract_spec_blocks_cdiv()           the anchor replace
      _fr13_assert_mamba_spec_blocks_cdiv_slot_demand()  the fail-loud preflight
  - scripts/fr13_launch_forked_fa2_tree_server.sh        strict 0/1 + docker -e
  - scripts/fr13_canonical_env.sh                        default-OFF export
  - scripts/fr13_required_tree_flags.sh                  comment-only QUEUED

WHY THE PREFLIGHT EXISTS (the thing these tests really protect): the lever's
premise is a units error. MambaSpec.num_speculative_blocks counts mamba STATE
SLOTS -- one page per (conv_state, ssm_state) pair, sized by
page_size_bytes = sum(prod(shape) * itemsize) and independent of block_size --
so ceil-dividing it by mamba_block_size does not "right-size" anything. The GDN
speculative path indexes those slots per draft NODE (stock FLA
fused_recurrent.py stores one recurrent state per speculative token to
ssm_state_indices[b, i_t]; the FR13 tree conv writeback scatters tree_n node
windows to spec_state_indices[b, :tree_n]). The preflight refuses =1 while
1 + cdiv(num_spec_tokens, mamba_block_size) < num_spec_tokens + 1, and opens by
itself if a per-node scratch rehome ever lowers that demand.
"""

from __future__ import annotations

import ast
import importlib.util
import py_compile
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
CANONICAL_ENV = REPO / "scripts" / "fr13_canonical_env.sh"
REGISTRY = REPO / "scripts" / "fr13_required_tree_flags.sh"

TEXT = PATCHER.read_text()
LAUNCHER_TEXT = LAUNCHER.read_text()
CANONICAL_TEXT = CANONICAL_ENV.read_text()
REGISTRY_TEXT = REGISTRY.read_text()

PRISTINE = Path("/tmp/vllm_pristine_019/extracted/vllm")
PRISTINE_ABSTRACT = PRISTINE / "model_executor" / "layers" / "mamba" / "abstract.py"

# The verbatim stock cu130-nightly construction site (abstract.py L44-59 of
# vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f
# 78bbbc38e776). Kept here so the end-to-end apply is hermetic; the
# pristine-gated test below asserts this really is what ships.
STOCK_ABSTRACT = '''# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


class MambaBase:
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        mamba_block_size = vllm_config.cache_config.mamba_block_size
        assert mamba_block_size is not None
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        return MambaSpec(
            shapes=tuple(self.get_state_shape()),
            dtypes=self.get_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,
            mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            ),
        )
'''

STOCK_ANCHOR = (
    "            num_speculative_blocks=(\n"
    "                vllm_config.speculative_config.num_speculative_tokens\n"
    "                if vllm_config.speculative_config\n"
    "                else 0\n"
    "            ),\n"
)


def _load_patcher(monkeypatch, *, flag: str) -> types.ModuleType:
    """Import the patcher fresh with FR13_MAMBA_SPEC_BLOCKS_CDIV=flag."""
    monkeypatch.setenv("FR13_MAMBA_SPEC_BLOCKS_CDIV", flag)
    spec = importlib.util.spec_from_file_location(
        f"_fr13_patcher_cdiv_{flag}", PATCHER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# flag hygiene
# --------------------------------------------------------------------------


def test_flag_default_off_and_never_defaults_on() -> None:
    assert 'os.environ.get(\n    "FR13_MAMBA_SPEC_BLOCKS_CDIV", "0"\n)' in TEXT
    # No site may default ON, and no site may read it without a default.
    assert '"FR13_MAMBA_SPEC_BLOCKS_CDIV", "1"' not in TEXT
    assert "'FR13_MAMBA_SPEC_BLOCKS_CDIV', '1'" not in TEXT
    assert 'environ.get("FR13_MAMBA_SPEC_BLOCKS_CDIV")' not in TEXT
    assert "environ.get('FR13_MAMBA_SPEC_BLOCKS_CDIV')" not in TEXT
    # Exactly one env read: the module-scope one.
    assert TEXT.count('"FR13_MAMBA_SPEC_BLOCKS_CDIV", "0"') == 1


def test_strict_zero_one(monkeypatch) -> None:
    for bad in ("2", "true", "yes", "01", "-1"):
        module = _load_patcher(monkeypatch, flag=bad)
        with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
            module._fr13_mamba_spec_blocks_cdiv()
    assert _load_patcher(monkeypatch, flag="0")._fr13_mamba_spec_blocks_cdiv() is False
    assert _load_patcher(monkeypatch, flag="1")._fr13_mamba_spec_blocks_cdiv() is True


def test_empty_env_is_off(monkeypatch) -> None:
    monkeypatch.delenv("FR13_MAMBA_SPEC_BLOCKS_CDIV", raising=False)
    spec = importlib.util.spec_from_file_location("_fr13_patcher_cdiv_unset", PATCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._fr13_mamba_spec_blocks_cdiv() is False


# --------------------------------------------------------------------------
# patcher AST wiring
# --------------------------------------------------------------------------


def test_patch_fn_and_preflight_are_defined_once() -> None:
    tree = ast.parse(TEXT)
    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for fn in (
        "_fr13_mamba_spec_blocks_cdiv",
        "_patch_mamba_abstract_spec_blocks_cdiv",
        "_fr13_assert_mamba_spec_blocks_cdiv_slot_demand",
    ):
        assert names.count(fn) == 1, f"{fn} must be defined exactly once"


def test_patch_step_registered_against_the_abstract_path() -> None:
    step = "(MAMBA_ABSTRACT_PATH, _patch_mamba_abstract_spec_blocks_cdiv()),"
    assert TEXT.count(step) == 1
    # It must live inside main()'s patch_steps list.
    steps_at = TEXT.index("    patch_steps = [")
    assert steps_at < TEXT.index(step)


def test_preflight_runs_in_main_before_any_patch_step() -> None:
    call = "    _fr13_assert_mamba_spec_blocks_cdiv_slot_demand()\n"
    assert TEXT.count(call) == 1
    main_at = TEXT.index("def main() -> int:")
    call_at = TEXT.index(call, main_at)
    steps_at = TEXT.index("    patch_steps = [", main_at)
    assert main_at < call_at < steps_at


def test_anchor_is_exact_count_guarded() -> None:
    fn_start = TEXT.index("def _patch_mamba_abstract_spec_blocks_cdiv()")
    fn_end = TEXT.index(
        "def _fr13_assert_mamba_spec_blocks_cdiv_slot_demand()", fn_start
    )
    body = TEXT[fn_start:fn_end]
    # Both anchors are count-asserted (not `in`-tested), house fail-loud style.
    assert "if text.count(anchor) != 1:" in body
    assert "if text.count(import_anchor) != 1:" in body
    assert body.count("raise RuntimeError(") == 2
    # Idempotency sentinel + OFF short-circuit.
    assert 'if "FR13_MAMBA_SPEC_BLOCKS_CDIV" in text:\n        return False' in body
    assert "if not _fr13_mamba_spec_blocks_cdiv():\n        return False" in body
    # Single-shot replaces.
    assert body.count("text.replace(anchor, inject, 1)") == 1


def test_patcher_carries_the_stock_anchor_verbatim() -> None:
    assert TEXT.count(
        '        "            num_speculative_blocks=(\\n"\n'
        '        "                vllm_config.speculative_config.'
        'num_speculative_tokens\\n"\n'
    ) == 1


# --------------------------------------------------------------------------
# end-to-end apply
# --------------------------------------------------------------------------


def test_off_arm_is_a_byte_noop(monkeypatch, tmp_path) -> None:
    module = _load_patcher(monkeypatch, flag="0")
    target = tmp_path / "abstract.py"
    target.write_text(STOCK_ABSTRACT)
    module.MAMBA_ABSTRACT_PATH = target
    assert module._patch_mamba_abstract_spec_blocks_cdiv() is False
    assert target.read_text() == STOCK_ABSTRACT


def test_on_arm_rewrites_to_cdiv_and_compiles(monkeypatch, tmp_path) -> None:
    module = _load_patcher(monkeypatch, flag="1")
    target = tmp_path / "abstract.py"
    target.write_text(STOCK_ABSTRACT)
    module.MAMBA_ABSTRACT_PATH = target

    assert module._patch_mamba_abstract_spec_blocks_cdiv() is True
    patched = target.read_text()

    # The token-count expression is gone; the slot expression is cdiv-by-block.
    assert STOCK_ANCHOR not in patched
    assert (
        "            num_speculative_blocks=(\n"
        "                cdiv(\n"
        "                    vllm_config.speculative_config."
        "num_speculative_tokens,\n"
        "                    mamba_block_size,\n"
        "                )\n"
        "                if vllm_config.speculative_config\n"
        "                else 0\n"
        "            ),\n"
    ) in patched
    # cdiv must be bound, exactly once, from the version-correct module.
    assert patched.count(
        "from vllm.utils.math_utils import cdiv  # FR13_MAMBA_SPEC_BLOCKS_CDIV"
    ) == 1
    bound = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "vllm.utils.math_utils"
        and any((a.asname or a.name) == "cdiv" for a in node.names)
        for node in ast.parse(patched).body
    )
    assert bound, "patched abstract.py has no cdiv binding"
    ast.parse(patched)
    py_compile.compile(str(target), doraise=True)

    # Second application is a no-op (sentinel), not a double patch.
    assert module._patch_mamba_abstract_spec_blocks_cdiv() is False
    assert target.read_text() == patched


def test_missing_anchor_fails_loud(monkeypatch, tmp_path) -> None:
    module = _load_patcher(monkeypatch, flag="1")
    target = tmp_path / "abstract.py"
    target.write_text(STOCK_ABSTRACT.replace(STOCK_ANCHOR, "            x=1,\n"))
    module.MAMBA_ABSTRACT_PATH = target
    with pytest.raises(RuntimeError, match="construction anchor count 0 != 1"):
        module._patch_mamba_abstract_spec_blocks_cdiv()


def test_duplicated_anchor_fails_loud(monkeypatch, tmp_path) -> None:
    module = _load_patcher(monkeypatch, flag="1")
    target = tmp_path / "abstract.py"
    target.write_text(STOCK_ABSTRACT.replace(STOCK_ANCHOR, STOCK_ANCHOR * 2))
    module.MAMBA_ABSTRACT_PATH = target
    with pytest.raises(RuntimeError, match="construction anchor count 2 != 1"):
        module._patch_mamba_abstract_spec_blocks_cdiv()


@pytest.mark.skipif(
    not PRISTINE_ABSTRACT.exists(), reason="pristine vLLM tree not extracted"
)
def test_stock_fixture_matches_the_shipped_abstract() -> None:
    """The hermetic fixture above must be the real construction site."""
    shipped = PRISTINE_ABSTRACT.read_text()
    assert shipped.count(STOCK_ANCHOR) == 1
    assert (
        shipped.count("from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec\n")
        == 1
    )


# --------------------------------------------------------------------------
# the fail-loud slot-demand preflight
# --------------------------------------------------------------------------


def test_preflight_is_inert_when_flag_off(monkeypatch) -> None:
    module = _load_patcher(monkeypatch, flag="0")
    monkeypatch.setenv("NUM_SPECULATIVE_TOKENS", "31")
    monkeypatch.setenv("MAMBA_BLOCK_SIZE", "1024")
    module._fr13_assert_mamba_spec_blocks_cdiv_slot_demand()


def test_preflight_refuses_the_production_geometry(monkeypatch) -> None:
    """31 draft tokens / 1024-token mamba block: reserves 2 slots, needs 32."""
    module = _load_patcher(monkeypatch, flag="1")
    monkeypatch.setenv("NUM_SPECULATIVE_TOKENS", "31")
    monkeypatch.setenv("MAMBA_BLOCK_SIZE", "1024")
    with pytest.raises(RuntimeError) as excinfo:
        module._fr13_assert_mamba_spec_blocks_cdiv_slot_demand()
    message = str(excinfo.value)
    assert "would reserve 2 mamba state slots" in message
    assert "indexes 32 of them PER DRAFT NODE" in message
    # The message must name the mechanism, not just fail.
    assert "ssm_state_indices[b, i_t]" in message
    assert "spec_state_indices[b, :tree_n]" in message
    assert "STATE-SLOT count, not a token range" in message


def test_preflight_opens_when_demand_is_met(monkeypatch) -> None:
    """A per-node scratch rehome that lowers the demand opens the gate with no
    edit here: block_size 1 makes cdiv the identity, so reserved == demanded."""
    module = _load_patcher(monkeypatch, flag="1")
    monkeypatch.setenv("NUM_SPECULATIVE_TOKENS", "31")
    monkeypatch.setenv("MAMBA_BLOCK_SIZE", "1")
    module._fr13_assert_mamba_spec_blocks_cdiv_slot_demand()


def test_preflight_ignores_non_spec_boots(monkeypatch) -> None:
    module = _load_patcher(monkeypatch, flag="1")
    monkeypatch.setenv("NUM_SPECULATIVE_TOKENS", "0")
    monkeypatch.setenv("MAMBA_BLOCK_SIZE", "1024")
    module._fr13_assert_mamba_spec_blocks_cdiv_slot_demand()


def test_preflight_rejects_a_nonpositive_block_size(monkeypatch) -> None:
    module = _load_patcher(monkeypatch, flag="1")
    monkeypatch.setenv("NUM_SPECULATIVE_TOKENS", "31")
    monkeypatch.setenv("MAMBA_BLOCK_SIZE", "0")
    with pytest.raises(RuntimeError, match="MAMBA_BLOCK_SIZE must be positive"):
        module._fr13_assert_mamba_spec_blocks_cdiv_slot_demand()


# --------------------------------------------------------------------------
# launcher / registry wiring
# --------------------------------------------------------------------------


def test_launcher_validates_strict_and_forwards_default_off() -> None:
    assert 'case "${FR13_MAMBA_SPEC_BLOCKS_CDIV:-0}" in' in LAUNCHER_TEXT
    assert (
        'echo "FR13_MAMBA_SPEC_BLOCKS_CDIV must be 0 or 1" >&2; exit 2 ;;'
        in LAUNCHER_TEXT
    )
    # The patcher runs INSIDE the container, so the -e line is what carries it.
    assert (
        '-e FR13_MAMBA_SPEC_BLOCKS_CDIV="${FR13_MAMBA_SPEC_BLOCKS_CDIV:-0}" \\'
        in LAUNCHER_TEXT
    )
    assert LAUNCHER_TEXT.count("FR13_MAMBA_SPEC_BLOCKS_CDIV:-1") == 0
    # Validation must precede the docker run that forwards it.
    assert LAUNCHER_TEXT.index(
        'case "${FR13_MAMBA_SPEC_BLOCKS_CDIV:-0}" in'
    ) < LAUNCHER_TEXT.index('-e FR13_MAMBA_SPEC_BLOCKS_CDIV=')


def test_canonical_env_exports_default_off() -> None:
    assert (
        'export FR13_MAMBA_SPEC_BLOCKS_CDIV="${FR13_MAMBA_SPEC_BLOCKS_CDIV:-0}"'
        in CANONICAL_TEXT
    )


def test_registry_entry_is_comment_only_and_carries_the_verdict() -> None:
    line = next(
        ln
        for ln in REGISTRY_TEXT.splitlines()
        if "FR13_MAMBA_SPEC_BLOCKS_CDIV" in ln
    )
    # Comment-only: an array string would make the variant harness NEEDS
    # assertion demand the flag be literally present in container_env.txt.
    assert line.lstrip().startswith("#")
    assert '"FR13_MAMBA_SPEC_BLOCKS_CDIV=' not in REGISTRY_TEXT
    assert "QUEUED (built 2026-08-09, default 0=OFF)" in line
    # The measured motivation and the blocking verdict both have to be on record.
    assert "384 of 692 pool pages" in line
    assert "89-93% of LRU evictions are mamba pops" in line
    assert "DOES NOT SHIP AS-IS" in line
    assert "fused_recurrent.py" in line
    assert "_fr13_assert_mamba_spec_blocks_cdiv_slot_demand" in line
