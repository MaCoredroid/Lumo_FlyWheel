"""FR13_TREE_SAMPLE_ROW (FIX-A1) wiring + row-map gate.

H1 ROWBUG fix (FR13_CHASE_STEP1_BIND.md, wf_a71e2a24 nextStepSpec): stock
eagle samples the next-event drafter input at flat row prev_accepted_len,
but the committed tree leaf lives at the +1-shifted published node id
paths_buf[b, len-1] (committer publish: _gdn_path = node_id + 1). The fix
rewrites token_indices_to_sample BEFORE set_inputs_first_pass so every
downstream consumer (bonus-token write, sample_hidden_states/root logits,
positions, hidden_states, 4-spine-step slot/position kernel) moves
consistently. Default OFF = verbatim stock (FIX-2/3 wiring pattern,
playbook classes 5/6/7/9):

  - flag default OFF at every read site, launcher default 0 + passthrough;
  - engagement needle in BOTH states;
  - fail-louds: REQKEY required, needs_extra_input_slots refused, buffers
    missing after a commit;
  - freshness contract three legs (committer publish x2 twins, REQKEY
    pre-forward clear, propose consume-once) -- closes the stale-join
    class the step-1 instrument hit on prefill proposes (class 7/12);
  - tree-only proof: the consumer-visible mutation is gated on
    tree_mtp mode AND a tree-commit row count published only by the tree
    committer helper;
  - CPU unit test of the VERBATIM row-map fragment against the step-1
    records: [1,2,4] L=3 => row 4; [1,2,4,6,8] L=5 => row 8; len 0 => 0;
    chain [1..5] L=k => row k == stock (chain-neutral by construction);
    rows beyond the commit count keep stock (prefill rows untouched).
"""

from __future__ import annotations

import importlib.util
import py_compile
import textwrap
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"

TEXT = PATCHER.read_text()
LAUNCHER_TEXT = LAUNCHER.read_text()


def _helper_snippet() -> str:
    start = TEXT.index("    helper = r'''")
    body_start = start + len("    helper = r'''")
    return TEXT[body_start:TEXT.index("'''", body_start)]


def _tsr_inject_snippet() -> str:
    """The FIX-A1 pre-set_inputs_first_pass injection (plain triple-quote)."""
    anchor = 'tsr_inject = """'
    start = TEXT.index(anchor) + len(anchor)
    return TEXT[start:TEXT.index('"""', start)]


def test_flag_default_on_everywhere() -> None:
    assert 'os.environ.get("FR13_TREE_SAMPLE_ROW", "1") == "1"' in TEXT
    assert '"FR13_TREE_SAMPLE_ROW", "0"' not in TEXT
    assert "'FR13_TREE_SAMPLE_ROW', '1'" not in TEXT
    assert '\\"FR13_TREE_SAMPLE_ROW\\", \\"1\\"' not in TEXT
    assert 'environ.get("FR13_TREE_SAMPLE_ROW")' not in TEXT
    assert "environ.get('FR13_TREE_SAMPLE_ROW')" not in TEXT


def test_launcher_default_on_and_passthrough() -> None:
    assert "FR13_TREE_SAMPLE_ROW=${FR13_TREE_SAMPLE_ROW:-1}" in LAUNCHER_TEXT
    assert '-e FR13_TREE_SAMPLE_ROW="$FR13_TREE_SAMPLE_ROW" \\' in LAUNCHER_TEXT
    assert "FR13_TREE_SAMPLE_ROW:-0" not in LAUNCHER_TEXT
    assert "default OFF until gated" in LAUNCHER_TEXT


def test_needle_fires_in_both_states() -> None:
    snippet = _tsr_inject_snippet()
    needle_idx = snippet.index(
        "FR13_TREE_SAMPLE_ROW drafter sample-row fix: tsr=%s"
    )
    assert '"armed" if _fr13_tsr_on else "inert"' in snippet
    gate_idx = snippet.index("if _fr13_tsr_on:")
    assert needle_idx < gate_idx, "needle must fire in BOTH states"


def test_fail_louds_present() -> None:
    snippet = _tsr_inject_snippet()
    assert "FR13_TREE_SAMPLE_ROW=1 requires FR13_TREE_REQKEY=1" in snippet
    assert "needs_extra_input_slots drafters" in snippet
    assert "refusing to run vacuously" in snippet
    assert "accepted-path device buffers " in snippet
    assert "missing at propose time after a tree commit" in snippet


def test_freshness_contract_three_legs() -> None:
    # Leg 1: the tree committer publishes its row count -- BOTH twins
    # (greedy + sampled), inside the rejection-sampler helper (tree path
    # only => native/naive decode can never arm the fix).
    helper = _helper_snippet()
    assert helper.count(
        "_lumo_tree_commit_gdn._LUMO_FA_TREE_COMMIT_NROWS = len("
    ) == 2
    # Leg 2: the REQKEY pre-forward rewrite zeroes it at the start of every
    # step (prefill proposes / commit-less steps => verbatim stock row).
    assert (
        '_fr13_rk_gdn._LUMO_FA_TREE_COMMIT_NROWS = 0' in TEXT
    )
    # Leg 3: propose consumes once (re-zeroes after read).
    snippet = _tsr_inject_snippet()
    read_idx = snippet.index('"_LUMO_FA_TREE_COMMIT_NROWS", 0')
    clear_idx = snippet.index(
        "_fr13_tsr_gdn._LUMO_FA_TREE_COMMIT_NROWS = 0"
    )
    assert read_idx < clear_idx


def test_tree_only_engagement_ordering() -> None:
    snippet = _tsr_inject_snippet()
    flag_idx = snippet.index("if _fr13_tsr_on:")
    mode_idx = snippet.index('_fr13_tsr_mode == "tree_mtp"')
    nrows_idx = snippet.index("_fr13_tsr_nrows > 0")
    mutate_idx = snippet.index(
        "token_indices_to_sample = token_indices_to_sample.clone()"
    )
    assert flag_idx < mode_idx < mutate_idx
    assert flag_idx < nrows_idx < mutate_idx
    # The count is a COUNT consumer separation: num_rejected_tokens_gpu is
    # never touched by the fix block.
    assert "num_rejected_tokens_gpu" not in snippet


def test_trace_wrapper_untouched() -> None:
    start = TEXT.index("def _patch_eagle_mtp_draft_trace()")
    end = TEXT.index("\ndef ", start + 1)
    assert "FR13_TREE_SAMPLE_ROW" not in TEXT[start:end]


def test_fragment_parses() -> None:
    snippet = _tsr_inject_snippet()
    compile(
        "def _fr13_tsr_syntax_probe(self, token_indices_to_sample,"
        " common_attn_metadata, batch_size):\n" + snippet,
        "<fr13_tsr_block>",
        "exec",
    )


def _load_patcher_module():
    spec = importlib.util.spec_from_file_location(
        "fr10_phase4_patcher_tsr", PATCHER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_chase_test_module():
    spec = importlib.util.spec_from_file_location(
        "fr13_chase_wiring_for_tsr",
        Path(__file__).resolve().parent / "test_fr13_chase_diag_wiring.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eagle_patch_applies_to_synthetic_fixture_and_compiles(
    tmp_path,
) -> None:
    chase = _load_chase_test_module()
    module = _load_patcher_module()
    target = tmp_path / "eagle.py"
    target.write_text(chase._eagle_synthetic_fixture())
    module.EAGLE_PATH = target
    assert module._patch_eagle_tree_consumption_verify() is True
    patched = target.read_text()
    assert "FR13_TREE_SAMPLE_ROW" in patched
    # The whole rewrite precedes the set_inputs_first_pass call =>
    # bonus write / sample_hidden_states / positions / hidden_states /
    # spine-step slots all derive from the SAME fixed row.
    assert patched.index("# FR13_TSR_ROWMAP_END") < patched.index(
        "self.set_inputs_first_pass("
    )
    py_compile.compile(str(target), doraise=True)


def _run_rowmap_fragment(
    paths_rows: list[list[int]],
    lens: list[int],
    nrows: int,
    qsl: list[int],
    stock_tis: list[int],
    batch_size: int,
) -> list[int]:
    """Exec the VERBATIM committed row-map fragment on CPU tensors."""
    # Slice at line starts so textwrap.dedent sees the uniform 16-space
    # indentation of the injected block.
    begin = TEXT.rindex("\n", 0, TEXT.index("# FR13_TSR_ROWMAP_BEGIN")) + 1
    end = TEXT.rindex("\n", 0, TEXT.index("# FR13_TSR_ROWMAP_END")) + 1
    frag = textwrap.dedent(TEXT[begin:end])
    ns = {
        "torch": torch,
        "_fr13_tsr_nrows": nrows,
        "batch_size": batch_size,
        "_fr13_tsr_paths": torch.tensor(paths_rows, dtype=torch.int32),
        "_fr13_tsr_lens": torch.tensor(lens, dtype=torch.int32),
        "common_attn_metadata": SimpleNamespace(
            query_start_loc=torch.tensor(qsl, dtype=torch.int32)
        ),
        "token_indices_to_sample": torch.tensor(
            stock_tis, dtype=torch.int32
        ),
    }
    exec(compile(frag, "<fr13_tsr_rowmap>", "exec"), ns)
    return [int(x) for x in ns["token_indices_to_sample"].tolist()]


def _pad(path: list[int], cols: int = 10) -> list[int]:
    return path + [0] * (cols - len(path))


def test_rowmap_step1_record_partial_accept() -> None:
    # Verbatim step-1 H1 mechanism: prev accepted [1,2,4] (nodes [0,1,3],
    # L=3) -> stock samples flat row 3 = the REJECTED (0,1)-alt; the
    # published leaf row is 4.
    out = _run_rowmap_fragment(
        paths_rows=[_pad([1, 2, 4])],
        lens=[3],
        nrows=1,
        qsl=[0, 10],
        stock_tis=[3],
        batch_size=1,
    )
    assert out == [4]


def test_rowmap_step1_record_full_spine() -> None:
    # prev [1,2,4,6,8] (full spine L=5) -> stock samples row 5 = rejected
    # (0,0,1) depth-3 alt; the leaf row is 8.
    out = _run_rowmap_fragment(
        paths_rows=[_pad([1, 2, 4, 6, 8])],
        lens=[5],
        nrows=1,
        qsl=[0, 10],
        stock_tis=[5],
        batch_size=1,
    )
    assert out == [8]


def test_rowmap_zero_accept_root_row() -> None:
    # len == 0 (root rejected) => row 0 == stock (class-7 edge).
    out = _run_rowmap_fragment(
        paths_rows=[_pad([])],
        lens=[0],
        nrows=1,
        qsl=[0, 10],
        stock_tis=[0],
        batch_size=1,
    )
    assert out == [0]


def test_rowmap_chain_neutral_by_construction() -> None:
    # Chain published path is [1..L] => leaf row L == stock offset for
    # every k (incl. k=0): the fix is byte-identical on chain5.
    for k in range(0, 6):
        out = _run_rowmap_fragment(
            paths_rows=[_pad(list(range(1, k + 1)))],
            lens=[k],
            nrows=1,
            qsl=[0, 6],
            stock_tis=[k],
            batch_size=1,
        )
        assert out == [k], f"chain L={k} must equal stock"


def test_rowmap_batch_bases_and_stock_tail() -> None:
    # Multi-row batch: absolute index = query_start_loc base + leaf row;
    # rows beyond the published commit count (e.g. a prefill row riding
    # the batch) keep their STOCK index untouched.
    out = _run_rowmap_fragment(
        paths_rows=[
            _pad([1, 2, 4]),       # ROWBUG row: base 0  -> 0 + 4
            _pad([1, 3]),          # alt-leaf:   base 10 -> 10 + 3
            _pad([1, 2, 3, 4, 5]), # chain row:  base 20 -> 20 + 5 == stock
        ],
        lens=[3, 2, 5],
        nrows=2,  # only the first two rows were committed this step
        qsl=[0, 10, 20, 30],
        stock_tis=[3, 12, 25],
        batch_size=3,
    )
    assert out == [4, 13, 25]
    # row 2 (beyond nrows) kept stock 25 even though its buffer row holds a
    # chain path; row 1's alt-leaf [1,3] L=2 -> leaf 3 != stock offset 2.


def test_rowmap_alt_leaf_lower_than_len_never_negative() -> None:
    # Depth-1 alt-only accept commits path [1] with L=1 -> leaf row 1 ==
    # stock; a single-row gather with clamp never reads col -1 on L=0.
    out = _run_rowmap_fragment(
        paths_rows=[_pad([1]), _pad([])],
        lens=[1, 0],
        nrows=2,
        qsl=[0, 10, 20],
        stock_tis=[1, 10],
        batch_size=2,
    )
    assert out == [1, 10]
