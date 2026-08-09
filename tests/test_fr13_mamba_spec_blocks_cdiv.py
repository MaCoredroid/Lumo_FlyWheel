"""FR13_MAMBA_SPEC_BLOCKS_CDIV patch-site gate (mamba physical-page narrowing).

ONE patch-time flag, default OFF, byte-inert when OFF; the anchor-replace +
exact-count house pattern used by the neighbouring mamba patchers
(_patch_mamba_utils_preprocess_context_flag, _patch_mamba_state_utils_tree_conv
_node_copy).

Instrument under test:
  - scripts/fr10_phase4_patch_vllm_tree_gdn.py
      _fr13_mamba_spec_blocks_cdiv()                     strict 0/1 env read
      _patch_mamba_abstract_spec_blocks_cdiv()           the PHYSICAL cut
      _patch_gdn_attn_mamba_spec_scratch_table()         the LOGICAL rehome
      _fr13_assert_mamba_spec_blocks_cdiv_slot_demand()  the 2-slot floor
      _fr13_assert_mamba_spec_blocks_cdiv_coherent()     both halves or neither
  - scripts/fr13_launch_forked_fa2_tree_server.sh        strict 0/1 + docker -e
  - scripts/fr13_canonical_env.sh                        default-OFF export
  - scripts/fr13_required_tree_flags.sh                  comment-only QUEUED

WHAT THESE TESTS REALLY PROTECT: the flag arms TWO patch sites that are only
safe together. MambaSpec.num_speculative_blocks counts mamba STATE SLOTS -- one
page per (conv_state, ssm_state) pair, sized by page_size_bytes =
sum(prod(shape) * itemsize) and independent of block_size -- so ceil-dividing it
by mamba_block_size is a units error IF TAKEN ALONE: the GDN speculative path
indexes those slots per draft NODE. The lever is legal only because the second
site keeps the LOGICAL window num_spec + 1 columns wide, republishing the single
align spare page across columns 1..num_spec, so no consumer is ever narrowed --
only the PHYSICAL page count drops (3 * 32 -> 3 * 2 per request).

Hence the two fail-louds. The preflight enforces the floor that makes the
scratch column real (1 + cdiv(...) >= 2; a NULL_BLOCK_ID filler would fail the
conv commit row guard's strict (0, BANK_ROWS) check on all num_spec + 1
columns). The coherence assert refuses a HALF-APPLIED pair, which is the one
state nothing else would catch: Python slicing narrows silently, so an
abstract.py cut without the gdn_attn rehome short-feeds the kernels a 2-column
window instead of raising.
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
        "_patch_gdn_attn_mamba_spec_scratch_table",
        "_fr13_assert_mamba_spec_blocks_cdiv_slot_demand",
        "_fr13_assert_mamba_spec_blocks_cdiv_coherent",
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
        "def _patch_gdn_attn_mamba_spec_scratch_table()", fn_start
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


def test_preflight_admits_the_production_geometry(monkeypatch) -> None:
    """31 draft tokens / 1024-token mamba block reserves col0 + one scratch.

    This is the case the preflight used to REFUSE. The scratch rehome
    (_patch_gdn_attn_mamba_spec_scratch_table, same flag) lowers the physical
    demand to 2, exactly as the old refusal said it would: it "opens by itself
    once a per-node scratch rehome lands".
    """
    module = _load_patcher(monkeypatch, flag="1")
    monkeypatch.setenv("NUM_SPECULATIVE_TOKENS", "31")
    monkeypatch.setenv("MAMBA_BLOCK_SIZE", "1024")
    module._fr13_assert_mamba_spec_blocks_cdiv_slot_demand()
    assert module._FR13_MAMBA_SPEC_SCRATCH_PHYSICAL_COLS == 2


def test_preflight_refuses_a_reservation_below_the_scratch_floor(
    monkeypatch,
) -> None:
    """One slot is col0 with NO scratch page -- the conv row guard rejects a
    NULL_BLOCK_ID filler, so the floor of 2 is load-bearing."""
    module = _load_patcher(monkeypatch, flag="1")
    monkeypatch.setenv("NUM_SPECULATIVE_TOKENS", "31")
    monkeypatch.setenv("MAMBA_BLOCK_SIZE", "1024")
    monkeypatch.setattr(module, "_FR13_MAMBA_SPEC_SCRATCH_PHYSICAL_COLS", 3)
    with pytest.raises(RuntimeError) as excinfo:
        module._fr13_assert_mamba_spec_blocks_cdiv_slot_demand()
    message = str(excinfo.value)
    assert "would reserve 2 mamba state slots" in message
    # The message must name the mechanism, not just fail.
    assert "ONE real scratch page" in message
    assert "(0, BANK_ROWS)" in message
    assert "NULL_BLOCK_ID is not a legal" in message


def test_preflight_still_scales_with_the_block_size(monkeypatch) -> None:
    """block_size 1 makes cdiv the identity: 32 reserved slots clears the
    floor of 2 just as well. The floor is a minimum, not an equality."""
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
# the gdn_attn scratch window (the logical half of the lever)
# --------------------------------------------------------------------------


STOCK_SPEC_SLICE = (
    "                spec_state_indices_tensor = block_table_tensor[\n"
    "                    spec_sequence_masks_cpu, : self.num_spec + 1\n"
    "                ]\n"
)

STOCK_GDN_ATTN = (
    '"""Backend for GatedDeltaNet attention."""\n'
    "\n"
    "import torch\n"
    "\n"
    "\n"
    "class GDNAttentionBackend(AttentionBackend):\n"
    "    pass\n"
    "\n"
    "\n"
    "class GDNAttentionMetadataBuilder:\n"
    "    def build(self):\n"
    "        if True:\n"
    "            if True:\n"
    + STOCK_SPEC_SLICE
    + "            else:\n"
    + STOCK_SPEC_SLICE
    + "        return spec_state_indices_tensor\n"
)


def _patched_gdn_attn(monkeypatch, tmp_path, *, flag: str):
    module = _load_patcher(monkeypatch, flag=flag)
    target = tmp_path / "gdn_attn.py"
    target.write_text(STOCK_GDN_ATTN)
    module.GDN_ATTN_PATH = target
    return module, target


def test_scratch_window_off_arm_is_a_byte_noop(monkeypatch, tmp_path) -> None:
    module, target = _patched_gdn_attn(monkeypatch, tmp_path, flag="0")
    assert module._patch_gdn_attn_mamba_spec_scratch_table() is False
    assert target.read_text() == STOCK_GDN_ATTN


def test_scratch_window_rewrites_both_sites_and_compiles(
    monkeypatch, tmp_path
) -> None:
    module, target = _patched_gdn_attn(monkeypatch, tmp_path, flag="1")
    assert module._patch_gdn_attn_mamba_spec_scratch_table() is True
    patched = target.read_text()

    # BOTH slice sites are rewritten -- a single-site patch would leave one
    # path silently short-fed by the narrowed align gather.
    assert STOCK_SPEC_SLICE not in patched
    assert patched.count("_fr13_mamba_spec_scratch_window(") == 3  # def + 2 uses
    assert patched.count(module._FR13_MAMBA_SPEC_SCRATCH_SENTINEL) == 3
    # The logical width still comes from the TOKEN count, never from the
    # narrowed block count.
    assert patched.count("self.num_spec,\n") == 2
    ast.parse(patched)
    py_compile.compile(str(target), doraise=True)

    # Second application is a no-op (sentinel), not a double patch.
    assert module._patch_gdn_attn_mamba_spec_scratch_table() is False
    assert target.read_text() == patched


def test_scratch_window_anchor_counts_fail_loud(monkeypatch, tmp_path) -> None:
    module, target = _patched_gdn_attn(monkeypatch, tmp_path, flag="1")
    # Exactly two slice sites are required: one is anchor drift, not a subset.
    target.write_text(STOCK_GDN_ATTN.replace(STOCK_SPEC_SLICE, "            x = 1\n", 1))
    with pytest.raises(RuntimeError, match="spec-window anchor count 1 != 2"):
        module._patch_gdn_attn_mamba_spec_scratch_table()

    target.write_text(STOCK_GDN_ATTN.replace(STOCK_SPEC_SLICE, "            x = 1\n"))
    with pytest.raises(RuntimeError, match="spec-window anchor count 0 != 2"):
        module._patch_gdn_attn_mamba_spec_scratch_table()

    target.write_text(
        STOCK_GDN_ATTN.replace(
            "class GDNAttentionBackend(AttentionBackend):\n", "class Other:\n", 1
        )
    )
    with pytest.raises(RuntimeError, match="backend class anchor count 0 != 1"):
        module._patch_gdn_attn_mamba_spec_scratch_table()


def test_scratch_window_semantics(monkeypatch, tmp_path) -> None:
    """The emitted helper must map col0 -> col0 and the single align spare
    across every speculative column, contiguously and strictly positive."""
    torch = pytest.importorskip("torch")
    module, target = _patched_gdn_attn(monkeypatch, tmp_path, flag="1")
    module._patch_gdn_attn_mamba_spec_scratch_table()
    src = target.read_text()
    helper = src[
        src.index("def _fr13_mamba_spec_scratch_window") : src.index(
            "class GDNAttentionBackend"
        )
    ]
    namespace: dict = {"torch": torch}
    exec(helper, namespace)  # noqa: S102 - executing our own emitted source
    widen = namespace["_fr13_mamba_spec_scratch_window"]

    # The narrowed align gather: col0 = running page, col1 = the one spare.
    narrowed = torch.tensor([[101, 102], [201, 202]], dtype=torch.int32)
    out = widen(narrowed, 31)
    assert tuple(out.shape) == (2, 32)
    assert out.dtype == torch.int32
    # The FLA kernels index ssm_state_indices with a bare `+ i_t`.
    assert out.stride(1) == 1
    assert out[:, 0].tolist() == [101, 201]
    for row, scratch in ((0, 102), (1, 202)):
        assert set(out[row, 1:].tolist()) == {scratch}
    # The conv commit row guard requires every column strictly > 0.
    assert bool((out > 0).all())

    # Only two DISTINCT physical pages back the whole 32-wide window.
    assert len(set(out[0].tolist())) == 2

    # Already-full-width tables (mamba_cache_mode all/none, or the physical
    # narrowing not in force) must behave exactly like the stock slice.
    wide = (torch.arange(2 * 40, dtype=torch.int32).reshape(2, 40) + 1)
    assert torch.equal(widen(wide, 31), wide[:, :32])

    # A window with no scratch column at all is a wiring bug, not a silent
    # narrowing.
    with pytest.raises(RuntimeError, match="need col0 plus one scratch column"):
        widen(torch.tensor([[7]], dtype=torch.int32), 31)


def test_scratch_patch_step_runs_after_patch_gdn_attn() -> None:
    """_patch_gdn_attn rewrites gdn_attn.py wholesale; this patch's anchors are
    the stock slice sites it leaves intact, so ordering is asserted."""
    scratch = "(GDN_ATTN_PATH, _patch_gdn_attn_mamba_spec_scratch_table()),"
    base = "(GDN_ATTN_PATH, _patch_gdn_attn()),"
    assert TEXT.count(scratch) == 1
    assert TEXT.index("    patch_steps = [") < TEXT.index(scratch)
    assert TEXT.index(base) < TEXT.index(scratch)


# --------------------------------------------------------------------------
# the two halves ship together
# --------------------------------------------------------------------------


def test_coherence_assert_runs_in_main_after_the_patch_steps() -> None:
    call = "    _fr13_assert_mamba_spec_blocks_cdiv_coherent()\n"
    assert TEXT.count(call) == 1
    main_at = TEXT.index("def main() -> int:")
    steps_at = TEXT.index("    patch_steps = [", main_at)
    assert steps_at < TEXT.index(call, main_at)


def test_coherence_is_inert_when_flag_off(monkeypatch, tmp_path) -> None:
    module = _load_patcher(monkeypatch, flag="0")
    module.MAMBA_ABSTRACT_PATH = tmp_path / "abstract.py"
    module.GDN_ATTN_PATH = tmp_path / "gdn_attn.py"
    module.MAMBA_ABSTRACT_PATH.write_text(STOCK_ABSTRACT)
    module.GDN_ATTN_PATH.write_text(STOCK_GDN_ATTN)
    module._fr13_assert_mamba_spec_blocks_cdiv_coherent()


def test_coherence_accepts_both_halves_applied(monkeypatch, tmp_path) -> None:
    module = _load_patcher(monkeypatch, flag="1")
    module.MAMBA_ABSTRACT_PATH = tmp_path / "abstract.py"
    module.GDN_ATTN_PATH = tmp_path / "gdn_attn.py"
    module.MAMBA_ABSTRACT_PATH.write_text(STOCK_ABSTRACT)
    module.GDN_ATTN_PATH.write_text(STOCK_GDN_ATTN)
    assert module._patch_mamba_abstract_spec_blocks_cdiv() is True
    assert module._patch_gdn_attn_mamba_spec_scratch_table() is True
    module._fr13_assert_mamba_spec_blocks_cdiv_coherent()


def test_coherence_refuses_a_half_applied_pair(monkeypatch, tmp_path) -> None:
    """The failure mode this guards: abstract.py narrowed to a 2-column align
    gather while gdn_attn still slices [: num_spec + 1] off it. Python slicing
    narrows SILENTLY, so nothing else would raise."""
    module = _load_patcher(monkeypatch, flag="1")
    module.MAMBA_ABSTRACT_PATH = tmp_path / "abstract.py"
    module.GDN_ATTN_PATH = tmp_path / "gdn_attn.py"

    # physical half only
    module.MAMBA_ABSTRACT_PATH.write_text(STOCK_ABSTRACT)
    module.GDN_ATTN_PATH.write_text(STOCK_GDN_ATTN)
    assert module._patch_mamba_abstract_spec_blocks_cdiv() is True
    with pytest.raises(RuntimeError, match="incoherent"):
        module._fr13_assert_mamba_spec_blocks_cdiv_coherent()

    # logical half only
    module.MAMBA_ABSTRACT_PATH.write_text(STOCK_ABSTRACT)
    module.GDN_ATTN_PATH.write_text(STOCK_GDN_ATTN)
    assert module._patch_gdn_attn_mamba_spec_scratch_table() is True
    with pytest.raises(RuntimeError, match="incoherent"):
        module._fr13_assert_mamba_spec_blocks_cdiv_coherent()


@pytest.mark.skipif(
    not (PRISTINE / "v1" / "attention" / "backends" / "gdn_attn.py").exists(),
    reason="pristine vLLM tree not extracted",
)
def test_stock_gdn_attn_carries_exactly_two_spec_slice_sites() -> None:
    """The hermetic fixture above must match the real anchor multiplicity."""
    shipped = (PRISTINE / "v1" / "attention" / "backends" / "gdn_attn.py").read_text()
    assert shipped.count(STOCK_SPEC_SLICE) == 2
    assert shipped.count("class GDNAttentionBackend(AttentionBackend):\n") == 1


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
    # The measured motivation stays on record.
    assert "384 of 692 pool pages" in line
    assert "89-93% of LRU evictions are mamba pops" in line
    assert "fused_recurrent.py" in line
    assert "_fr13_assert_mamba_spec_blocks_cdiv_slot_demand" in line
    # BOTH halves must be described -- the entry is the only place a reader
    # learns the physical cut never ships without the logical rehome.
    assert "_patch_gdn_attn_mamba_spec_scratch_table" in line
    assert "_fr13_assert_mamba_spec_blocks_cdiv_coherent" in line
    # The superseded verdict must stay legible as superseded, not be erased.
    assert "DOES NOT SHIP AS-IS" in line
    assert "SUPERSEDES" in line
    # The distinction from the deleted lever is the whole argument.
    assert "FR13_SPEC_BLOCKS_CAP" in line
    assert "consumer widths are untouched" in line.lower()
    # The one combination that is still unsafe has to be named.
    assert "KNOWN CONSTRAINT" in line
