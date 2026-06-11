"""FR13 nondeterminism-chase fixes: behavioral CPU tests.

Covers the three flag-gated fixes (all default ON):

1. FR13_TREE_PER_REQ_GEN — the sampled tree committer seeds its numpy
   rejection-sampling rng from the per-request torch.Generator
   (sampling_metadata.generators) instead of the global torch RNG, so
   same-seed reruns no longer diverge with the boot's global RNG offset.
2. FR13_TREE_REQKEY — the committers publish accepted tree paths/lens keyed
   by request id, and the model-runner pre-forward rewrite installs the
   CURRENT slot occupants' own values (zeros for a first spec step) into the
   persistent batch-position-keyed device buffers.
3. FR13_TREE_REMAP_SEQ — race-free gather-then-scatter state remap kernel
   (GPU-only; covered here by wiring assertions, exercised by the re-probe).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PATCHER_PATH = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
PATCHER_TEXT = PATCHER_PATH.read_text()


@pytest.fixture(autouse=True)
def _legacy_route_escape_hatch(monkeypatch):
    # FR13_REPLAY_ROUTE now defaults ON; these CPU tests exec the committer
    # fragments, whose replay branch imports the triton kernel and requires
    # registered GDN replay layers (GPU/serving-only). Pin the explicit
    # legacy escape hatch =0 -- the RNG/path-publish logic under test is
    # route-independent; default-ON wiring is asserted in
    # test_fr13_replay_route_wiring.py.
    monkeypatch.setenv("FR13_REPLAY_ROUTE", "0")


def _extract_function(name: str) -> str:
    start = PATCHER_TEXT.index(f"def {name}(")
    ends = [
        PATCHER_TEXT.find("\ndef ", start + 1),
        PATCHER_TEXT.find("\n'''", start + 1),
    ]
    end = min(e for e in ends if e != -1)
    return PATCHER_TEXT[start:end]


def _install_gdn_stub(monkeypatch, *, row_req_ids):
    names = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.mamba",
    ]
    for name in names:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    gdn = types.ModuleType("vllm.model_executor.layers.mamba.gdn_linear_attn")
    gdn._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR = torch.zeros((8, 3), dtype=torch.int32)
    gdn._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR = torch.zeros((8,), dtype=torch.int32)
    if row_req_ids is not None:
        gdn._LUMO_FA_SAMPLER_ROW_REQ_IDS = list(row_req_ids)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.mamba.gdn_linear_attn", gdn
    )
    setattr(sys.modules["vllm.model_executor.layers.mamba"], "gdn_linear_attn", gdn)
    return gdn


def _committer_namespace() -> dict:
    ns: dict = {"torch": torch}
    exec(_extract_function("_lumo_tree_canonical_multidraft_sample"), ns)
    exec(_extract_function("_lumo_tree_path_lcp_max_greedy_sample"), ns)
    return ns


def _call_sampled(ns, *, generators):
    output_token_ids = torch.full((1, 3), -1, dtype=torch.long)
    accepted_tree_rows = torch.zeros((1,), dtype=torch.int32)
    torch_state = torch.random.get_rng_state()
    try:
        # Deterministic logits, independent of the ambient global RNG state.
        torch.manual_seed(0)
        target_logits = torch.randn(2, 8)
        tree_self_logits = torch.randn(2, 8)
    finally:
        torch.random.set_rng_state(torch_state)
    out = ns["_lumo_tree_canonical_multidraft_sample"](
        output_token_ids,
        accepted_tree_rows,
        [2],
        torch.tensor([1, 2]),
        torch.tensor([-1, 0]),
        target_logits,
        tree_self_logits,
        None,
        torch.tensor([[5]]),
        2,
        generators=generators,
    )
    return out.tolist()


def test_per_req_gen_isolates_committer_from_global_rng(monkeypatch) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.delenv("FR13_TREE_PER_REQ_GEN", raising=False)  # default ON
    monkeypatch.setenv("FR13_TREE_REQKEY", "1")
    ns = _committer_namespace()

    _install_gdn_stub(monkeypatch, row_req_ids=["req-a"])
    torch.manual_seed(1)
    torch.rand(7)  # perturb the global stream
    gen_a = torch.Generator()
    gen_a.manual_seed(1313)
    out_a = _call_sampled(ns, generators={0: gen_a})

    _install_gdn_stub(monkeypatch, row_req_ids=["req-a"])
    torch.manual_seed(99)
    torch.rand(123)  # different global stream offset
    gen_b = torch.Generator()
    gen_b.manual_seed(1313)
    out_b = _call_sampled(ns, generators={0: gen_b})

    assert out_a == out_b


def test_per_req_gen_flag_off_restores_legacy_global_rng(monkeypatch) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.setenv("FR13_TREE_PER_REQ_GEN", "0")
    monkeypatch.setenv("FR13_TREE_REQKEY", "1")
    ns = _committer_namespace()

    _install_gdn_stub(monkeypatch, row_req_ids=["req-a"])
    torch.manual_seed(42)
    out_a = _call_sampled(ns, generators={0: torch.Generator().manual_seed(1313)})

    _install_gdn_stub(monkeypatch, row_req_ids=["req-a"])
    torch.manual_seed(42)
    out_b = _call_sampled(ns, generators={0: torch.Generator().manual_seed(1313)})

    # Legacy path is reproducible given the SAME global torch RNG state and
    # must ignore the per-request generator entirely.
    assert out_a == out_b


def test_committer_publishes_request_keyed_paths(monkeypatch) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.delenv("FR13_TREE_PER_REQ_GEN", raising=False)
    monkeypatch.delenv("FR13_TREE_REQKEY", raising=False)  # default ON
    ns = _committer_namespace()

    gdn = _install_gdn_stub(monkeypatch, row_req_ids=["req-a"])
    _call_sampled(ns, generators={0: torch.Generator().manual_seed(1313)})

    by_req = getattr(gdn, "_LUMO_FA_TREE_ACCEPT_BY_REQ", None)
    assert by_req is not None and "req-a" in by_req
    path, length = by_req["req-a"]
    assert length == int(gdn._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR[0].item())
    assert path == gdn._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR[0, :length].tolist()


def test_committer_fails_loud_without_row_req_ids(monkeypatch) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.delenv("FR13_TREE_REQKEY", raising=False)  # default ON
    ns = _committer_namespace()

    _install_gdn_stub(monkeypatch, row_req_ids=None)
    with pytest.raises(RuntimeError, match="FR13_TREE_REQKEY|publish accepted rows"):
        _call_sampled(ns, generators={0: torch.Generator().manual_seed(1313)})


def test_greedy_committer_publishes_request_keyed_paths(monkeypatch) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.delenv("FR13_TREE_REQKEY", raising=False)  # default ON
    ns = _committer_namespace()

    gdn = _install_gdn_stub(monkeypatch, row_req_ids=["req-g"])
    ns["_lumo_tree_path_lcp_max_greedy_sample"](
        torch.full((1, 3), -1, dtype=torch.long),
        torch.zeros((1,), dtype=torch.int32),
        [2],
        torch.tensor([1, 2]),
        torch.tensor([-1, 0]),
        torch.tensor([1, 2]),
        torch.tensor([9, 9]),
        torch.tensor([[5]]),
        2,
    )
    by_req = getattr(gdn, "_LUMO_FA_TREE_ACCEPT_BY_REQ", None)
    assert by_req is not None and "req-g" in by_req


def _load_patcher_module():
    spec = importlib.util.spec_from_file_location("fr10_phase4_patcher", PATCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REQKEY_ANCHOR = (
    "            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens\n"
    "            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)\n"
    "            self.num_decode_draft_tokens.copy_to_gpu()\n"
)


def _patched_reqkey_block(tmp_path) -> str:
    module = _load_patcher_module()
    target = tmp_path / "gpu_model_runner.py"
    target.write_text(_REQKEY_ANCHOR)
    module.GPU_MODEL_RUNNER_PATH = target
    assert module._patch_gpu_model_runner_tree_reqkey() is True
    patched = target.read_text()
    assert "# FR13_TREE_REQKEY_REWRITE" in patched
    block = patched[len(_REQKEY_ANCHOR):]
    return "\n".join(
        line[12:] if line.strip() else line for line in block.splitlines()
    )


def _exec_reqkey_block(monkeypatch, tmp_path, *, by_req, req_ids, draft_flags, gdn):
    block = _patched_reqkey_block(tmp_path)
    if by_req is not None:
        gdn._LUMO_FA_TREE_ACCEPT_BY_REQ = dict(by_req)
    ns = {
        "torch": torch,
        "self": SimpleNamespace(input_batch=SimpleNamespace(req_ids=list(req_ids))),
        "num_reqs": len(req_ids),
        "num_decode_draft_tokens": np.array(draft_flags, dtype=np.int32),
    }
    exec(compile(block, "<fr13_reqkey_block>", "exec"), ns)
    return gdn


def test_model_runner_rewrite_installs_request_values_and_prunes(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.delenv("FR13_TREE_REQKEY", raising=False)  # default ON
    gdn = _install_gdn_stub(monkeypatch, row_req_ids=None)
    # Garbage left in the slot by a previous occupant.
    gdn._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR[0] = torch.tensor([3, 2, 1])
    gdn._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR[0] = 3

    _exec_reqkey_block(
        monkeypatch,
        tmp_path,
        by_req={"req-a": ([2, 3], 2), "req-zombie": ([1], 1)},
        req_ids=["req-a", "req-b"],
        draft_flags=[0, -1],  # req-a is the only spec row
        gdn=gdn,
    )

    assert gdn._LUMO_FA_SAMPLER_ROW_REQ_IDS == ["req-a", "req-b"]
    assert gdn._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR[0].tolist() == [2, 3, 0]
    assert int(gdn._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR[0].item()) == 2
    assert "req-zombie" not in gdn._LUMO_FA_TREE_ACCEPT_BY_REQ


def test_model_runner_rewrite_zeros_first_spec_step(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.delenv("FR13_TREE_REQKEY", raising=False)  # default ON
    gdn = _install_gdn_stub(monkeypatch, row_req_ids=None)
    # The literal FR13 bug: previous occupant's path/lens still in the slot.
    gdn._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR[0] = torch.tensor([4, 5, 6])
    gdn._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR[0] = 2

    _exec_reqkey_block(
        monkeypatch,
        tmp_path,
        by_req={},
        req_ids=["req-new"],
        draft_flags=[5],
        gdn=gdn,
    )

    assert gdn._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR[0].tolist() == [0, 0, 0]
    assert int(gdn._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR[0].item()) == 0


def test_model_runner_rewrite_flag_off_is_inert(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.setenv("FR13_TREE_REQKEY", "0")
    gdn = _install_gdn_stub(monkeypatch, row_req_ids=None)
    gdn._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR[0] = torch.tensor([4, 5, 6])
    gdn._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR[0] = 2

    _exec_reqkey_block(
        monkeypatch,
        tmp_path,
        by_req=None,
        req_ids=["req-new"],
        draft_flags=[5],
        gdn=gdn,
    )

    # Legacy slot-keyed behavior preserved when the flag is off.
    assert gdn._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR[0].tolist() == [4, 5, 6]
    assert int(gdn._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR[0].item()) == 2
    assert not hasattr(gdn, "_LUMO_FA_SAMPLER_ROW_REQ_IDS")


def test_remap_race_fix_wiring() -> None:
    kernel_text = (REPO / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py").read_text()
    assert "_linear_remap_rows_gather_kernel" in kernel_text
    assert 'os.environ.get("FR13_TREE_REMAP_SEQ", "1") == "1"' in kernel_text
    # All source slices must be in registers before any store: single program
    # owns every path column for an element block.
    assert "src_bank[:, None] * ROW_ELEMS + offs[None, :]" in kernel_text
    assert "dst_bank[:, None] * ROW_ELEMS + offs[None, :]" in kernel_text


def test_launcher_forwards_fr13_chase_flags() -> None:
    launcher = (REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh").read_text()
    for flag in ("FR13_TREE_PER_REQ_GEN", "FR13_TREE_REQKEY", "FR13_TREE_REMAP_SEQ"):
        assert f'-e {flag}="${{{flag}:-1}}"' in launcher


def test_patcher_registers_reqkey_step_and_passes_generators() -> None:
    assert "(GPU_MODEL_RUNNER_PATH, _patch_gpu_model_runner_tree_reqkey())" in PATCHER_TEXT
    assert (
        "(GPU_MODEL_RUNNER_PATH, _patch_gpu_model_runner_replay_draft_reqkey())"
        in PATCHER_TEXT
    )
    assert 'generators=getattr(sampling_metadata, "generators", None),' in PATCHER_TEXT
    # The sampled committer must accept the per-request generators.
    sampled = _extract_function("_lumo_tree_canonical_multidraft_sample")
    assert "generators=None" in sampled
    assert "generators.get(req_i)" in sampled


PRISTINE = Path("/tmp/vllm_pristine_019/extracted/vllm")


@pytest.mark.skipif(not PRISTINE.exists(), reason="pristine vLLM tree not present")
def test_reqkey_patch_applies_to_pristine_model_runner(tmp_path) -> None:
    import py_compile

    module = _load_patcher_module()
    target = tmp_path / "gpu_model_runner.py"
    target.write_text((PRISTINE / "v1" / "worker" / "gpu_model_runner.py").read_text())
    module.GPU_MODEL_RUNNER_PATH = target
    assert module._patch_gpu_model_runner_tree_reqkey() is True
    py_compile.compile(str(target), doraise=True)
    # Idempotent on the sentinel.
    assert module._patch_gpu_model_runner_tree_reqkey() is False


@pytest.mark.skipif(not PRISTINE.exists(), reason="pristine vLLM tree not present")
def test_replay_draft_reqkey_patch_applies_to_pristine_model_runner(tmp_path) -> None:
    import py_compile

    module = _load_patcher_module()
    target = tmp_path / "gpu_model_runner.py"
    target.write_text((PRISTINE / "v1" / "worker" / "gpu_model_runner.py").read_text())
    module.GPU_MODEL_RUNNER_PATH = target
    assert module._patch_gpu_model_runner_replay_draft_reqkey() is True
    py_compile.compile(str(target), doraise=True)
    text = target.read_text()
    assert "# FR13_REPLAY_DRAFT_REQKEY" in text
    assert "_fr13_replay_draft_token_req_ids" in text
    assert "_fr13_replay_prev_sampled_req_ids" in text
    assert "_fr13_replay_scatter_first_spec_drafts()" in text
    assert "prev_req_id_to_index" in text
    assert "missing request-keyed draft rows" in text
    assert "missing request-keyed sampled " in text
    assert "still contains async draft-token placeholders" in text
    assert "still contains async sampled-token placeholders" in text
    # The repair must source from the actual drafter tensor, not replace -1
    # placeholders with a constant token.
    repair_start = text.index("def _fr13_replay_scatter_first_spec_drafts")
    repair_end = text.index("        _fr13_replay_scatter_first_spec_drafts()", repair_start)
    repair_block = text[repair_start:repair_end]
    assert "self._draft_token_ids.to(" in repair_block
    assert "_fr13_prev_sampled.to(" in repair_block
    assert "fill_(0)" not in repair_block
    # Idempotent on the sentinel.
    assert module._patch_gpu_model_runner_replay_draft_reqkey() is False


@pytest.mark.skipif(not PRISTINE.exists(), reason="pristine vLLM tree not present")
def test_tree_lcp_patch_applies_to_pristine_rejection_sampler(tmp_path) -> None:
    import py_compile

    module = _load_patcher_module()
    target = tmp_path / "rejection_sampler.py"
    target.write_text(
        (PRISTINE / "v1" / "sample" / "rejection_sampler.py").read_text()
    )
    module.REJECTION_SAMPLER_PATH = target
    assert module._patch_rejection_sampler_tree_lcp() is True
    py_compile.compile(str(target), doraise=True)
    text = target.read_text()
    assert "FR13_TREE_PER_REQ_GEN" in text
    assert "FR13_TREE_REQKEY" in text
