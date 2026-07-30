from __future__ import annotations

import importlib.util
import py_compile
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest


PATCHER_PATH = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").resolve()
ANCHOR = (
    "        use_spec_decode = "
    "len(scheduler_output.scheduled_spec_decode_tokens) > 0\n"
)


def _load_fixed32_patcher(
    monkeypatch: pytest.MonkeyPatch,
) -> types.ModuleType:
    monkeypatch.setenv("FR13_FIXED32_MODE", "tail6_fixed32")
    name = "fr13_fixed32_prefill_owner_patcher"
    spec = importlib.util.spec_from_file_location(name, PATCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_runtime_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[types.SimpleNamespace, types.SimpleNamespace]:
    gdn = types.SimpleNamespace(
        _LUMO_FA_SPEC_ROW_REQ_IDS=["stale-spec-owner"],
        _FR13_FIXED32_SPEC_BATCH_INDICES=(0,),
        _LUMO_FA_TREE_COMMIT_NROWS=99,
    )
    cache = types.SimpleNamespace(
        active_requests=set(),
        cached_requests=set(),
    )
    calls = types.SimpleNamespace(
        stages=[],
        prewarms=0,
    )

    merged = types.ModuleType("fr13_merged_drafter")
    merged.merged_on = lambda: True
    merged.get_cache = lambda: cache

    def maybe_prewarm(_cache) -> None:
        assert _cache is cache
        calls.prewarms += 1

    def stage_fixed32_step(
        _cache,
        request_ids,
        token_rows,
        prompt_lengths,
        computed_lengths,
        scheduled_lengths,
        scheduled_draft_lengths,
        committed_lengths,
        discard_rows,
        step_seq,
        *,
        restart_request_ids=(),
    ) -> None:
        assert _cache is cache
        calls.stages.append(
            {
                "request_ids": tuple(str(req_id) for req_id in request_ids),
                "token_rows_shape": tuple(token_rows.shape),
                "prompt_lengths": tuple(int(value) for value in prompt_lengths),
                "computed_lengths": tuple(
                    int(value) for value in computed_lengths
                ),
                "scheduled_lengths": tuple(
                    int(value) for value in scheduled_lengths
                ),
                "scheduled_draft_lengths": tuple(
                    int(value) for value in scheduled_draft_lengths
                ),
                "committed_lengths": tuple(
                    int(value) for value in committed_lengths
                ),
                "discard_rows": tuple(bool(value) for value in discard_rows),
                "step_seq": int(step_seq),
                "restart_request_ids": frozenset(
                    str(req_id) for req_id in restart_request_ids
                ),
            }
        )

    merged.maybe_prewarm = maybe_prewarm
    merged.stage_fixed32_step = stage_fixed32_step
    monkeypatch.setitem(sys.modules, "fr13_merged_drafter", merged)

    packages = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType(
            "vllm.model_executor.layers"
        ),
        "vllm.model_executor.layers.mamba": types.ModuleType(
            "vllm.model_executor.layers.mamba"
        ),
    }
    for package in packages.values():
        package.__path__ = []
    packages["vllm.model_executor.layers.mamba"].gdn_linear_attn = gdn
    for name, package in packages.items():
        monkeypatch.setitem(sys.modules, name, package)
    return gdn, calls


def _runtime_self(
    request_ids: list[str],
    prompt_lengths: list[int],
    computed_lengths: list[int],
    *,
    committed_lengths: list[int] | None = None,
    discard_rows: list[bool] | None = None,
) -> types.SimpleNamespace:
    rows = len(request_ids)
    if committed_lengths is None:
        committed_lengths = list(prompt_lengths)
    if discard_rows is None:
        discard_rows = [False] * rows
    width = max(committed_lengths, default=1)
    token_ids = np.zeros((rows, width), dtype=np.int64)
    block_rows = np.arange(max(rows, 1) * 64, dtype=np.int64).reshape(
        max(rows, 1), 64
    )
    block_table = types.SimpleNamespace(
        block_table=types.SimpleNamespace(np=block_rows)
    )
    mamba_spec = type("MambaSpec", (), {"block_size": 1024})()
    return types.SimpleNamespace(
        input_batch=types.SimpleNamespace(
            req_ids=list(request_ids),
            num_prompt_tokens=np.asarray(prompt_lengths, dtype=np.int64),
            num_computed_tokens_cpu=np.asarray(
                computed_lengths, dtype=np.int64
            ),
            num_tokens_no_spec=np.asarray(
                committed_lengths, dtype=np.int64
            ),
            token_ids_cpu=token_ids,
            block_table=[block_table],
        ),
        discard_request_mask=types.SimpleNamespace(
            np=np.asarray(discard_rows, dtype=np.bool_)
        ),
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(kv_cache_spec=mamba_spec)
            ]
        ),
    )


def _patched_prelude(
    tmp_path: Path,
    patcher: types.ModuleType,
) -> tuple[types.ModuleType, str]:
    target = tmp_path / "gpu_model_runner.py"
    target.write_text(
        "def run(self, scheduler_output, num_reqs):\n"
        + ANCHOR
        + "        return use_spec_decode\n"
    )
    patcher.GPU_MODEL_RUNNER_PATH = target
    assert patcher._patch_gpu_model_runner_row_req_ids_fresh() is True
    assert patcher._patch_gpu_model_runner_merged_drafter() is True
    py_compile.compile(str(target), doraise=True)
    name = "fr13_fixed32_prefill_owner_runtime"
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, target.read_text()


def test_long_b1_chunked_prefill_publishes_owner_before_proposal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patcher = _load_fixed32_patcher(monkeypatch)
    gdn, calls = _install_runtime_stubs(monkeypatch)
    runtime, source = _patched_prelude(tmp_path, patcher)
    runner = _runtime_self(
        ["request-0"],
        [22_869],
        [0],
        committed_lengths=[22_869],
        discard_rows=[True],
    )
    scheduler_output = types.SimpleNamespace(
        scheduled_spec_decode_tokens={},
        num_scheduled_tokens={"request-0": 1_024},
        scheduled_new_reqs=[
            types.SimpleNamespace(req_id="request-0")
        ],
        scheduled_cached_reqs=types.SimpleNamespace(resumed_req_ids=set()),
    )

    assert runtime.run(runner, scheduler_output, 1) is False
    assert gdn._LUMO_FA_SAMPLER_ROW_REQ_IDS == ["request-0"]
    assert gdn._LUMO_FA_SPEC_ROW_REQ_IDS is None
    assert gdn._FR13_FIXED32_SPEC_BATCH_INDICES is None
    assert gdn._FR13_FIXED32_BATCH_ROWS == 1
    assert gdn._FR13_FIXED32_SPEC_ROWS == 0
    assert gdn._LUMO_FA_TREE_COMMIT_NROWS == 0
    assert gdn._LUMO_FA_STEP_SEQ == runner._fr13_rf_seq == 1
    assert calls.stages == [
        {
            "request_ids": ("request-0",),
            "token_rows_shape": (1, 22_869),
            "prompt_lengths": (22_869,),
            "computed_lengths": (0,),
            "scheduled_lengths": (1_024,),
            "scheduled_draft_lengths": (0,),
            "committed_lengths": (22_869,),
            "discard_rows": (True,),
            "step_seq": 1,
            "restart_request_ids": frozenset({"request-0"}),
        }
    ]
    assert source.index("# FR13_MERGED_DRAFTER_LIFECYCLE") < source.index(
        "use_spec_decode ="
    )
    assert (
        "_FR13_FIXED32_ACCEPTED_OUTPUT_CURRENT = None" in source
    )


def test_b4_mixed_prefill_uses_current_full_owner_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patcher = _load_fixed32_patcher(monkeypatch)
    gdn, calls = _install_runtime_stubs(monkeypatch)
    runtime, _source = _patched_prelude(tmp_path, patcher)
    request_ids = ["request-0", "request-1", "request-2", "request-3"]
    runner = _runtime_self(
        request_ids,
        [2_048, 2_049, 2_050, 2_051],
        [1_024, 2_049, 1_024, 2_051],
        committed_lengths=[2_048, 2_050, 2_050, 2_052],
        discard_rows=[True, False, True, False],
    )
    scheduler_output = types.SimpleNamespace(
        scheduled_spec_decode_tokens={
            "request-1": [1] * 31,
            "request-3": [1] * 31,
        },
        num_scheduled_tokens={req_id: 32 for req_id in request_ids},
        scheduled_new_reqs=[
            types.SimpleNamespace(req_id="request-0"),
            types.SimpleNamespace(req_id="request-2"),
        ],
        scheduled_cached_reqs=types.SimpleNamespace(
            resumed_req_ids={"request-1", "request-3"}
        ),
    )

    assert runtime.run(runner, scheduler_output, 4) is True
    assert gdn._LUMO_FA_SAMPLER_ROW_REQ_IDS == request_ids
    assert gdn._LUMO_FA_SPEC_ROW_REQ_IDS is None
    assert [req_id for req_id, _page in gdn._LUMO_FA_SPEC_ROW_CONV_COL0] == [
        "request-1",
        "request-3",
    ]
    assert calls.stages == [
        {
            "request_ids": tuple(request_ids),
            "token_rows_shape": (4, 2_052),
            "prompt_lengths": (2_048, 2_049, 2_050, 2_051),
            "computed_lengths": (1_024, 2_049, 1_024, 2_051),
            "scheduled_lengths": (32, 32, 32, 32),
            "scheduled_draft_lengths": (0, 31, 0, 31),
            "committed_lengths": (2_048, 2_050, 2_050, 2_052),
            "discard_rows": (True, False, True, False),
            "step_seq": 1,
            "restart_request_ids": frozenset(request_ids),
        }
    ]


@pytest.mark.parametrize(
    ("request_ids", "num_reqs"),
    (
        (["request-0"], 2),
        (["duplicate", "duplicate"], 2),
        ([""], 1),
        ([f"request-{index}" for index in range(5)], 5),
    ),
)
def test_fixed32_full_owner_publish_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request_ids: list[str],
    num_reqs: int,
) -> None:
    patcher = _load_fixed32_patcher(monkeypatch)
    _install_runtime_stubs(monkeypatch)
    runtime, _source = _patched_prelude(tmp_path, patcher)
    runner = _runtime_self(
        request_ids,
        [1] * len(request_ids),
        [0] * len(request_ids),
    )
    scheduler_output = types.SimpleNamespace(
        scheduled_spec_decode_tokens={},
        num_scheduled_tokens={req_id: 1 for req_id in request_ids},
    )

    with pytest.raises(RuntimeError, match="full row-owner publish drift"):
        runtime.run(runner, scheduler_output, num_reqs)


def test_emitted_fixed32_step_sequence_has_one_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patcher = _load_fixed32_patcher(monkeypatch)
    target = tmp_path / "gpu_model_runner.py"
    target.write_text(
        "def run(self, scheduler_output, num_reqs, num_decode_draft_tokens):\n"
        + ANCHOR
        + "        if use_spec_decode:\n"
        "            self.num_decode_draft_tokens.np[:num_reqs] = "
        "num_decode_draft_tokens\n"
        "            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)\n"
        "            self.num_decode_draft_tokens.copy_to_gpu()\n"
        "        return use_spec_decode\n"
    )
    patcher.GPU_MODEL_RUNNER_PATH = target
    assert patcher._patch_gpu_model_runner_row_req_ids_fresh() is True
    assert patcher._patch_gpu_model_runner_tree_reqkey() is True
    assert patcher._patch_gpu_model_runner_merged_drafter() is True
    py_compile.compile(str(target), doraise=True)
    source = target.read_text()

    assert source.count(
        "self._fr13_rf_seq = getattr(self, '_fr13_rf_seq', 0) + 1"
    ) == 1
    assert source.index("_LUMO_FA_SAMPLER_ROW_REQ_IDS = _fr13_rf_req_ids") < (
        source.index("use_spec_decode =")
    )
    assert source.index("_fr13_fixed32_observed_begin(") > source.index(
        "if use_spec_decode:"
    )


def test_fixed32_async_gate_waits_through_drafter_proposal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patcher = _load_fixed32_patcher(monkeypatch)
    target = tmp_path / "gpu_model_runner.py"
    target.write_text(
        "import threading\n"
        "sample_entered = threading.Event()\n"
        "sample_continue = threading.Event()\n"
        "sample_released = threading.Event()\n"
        "sample_finish = threading.Event()\n"
        "class GPUModelRunner:\n"
        "    def __init__(self):\n"
        "        self.execute_model_state = None\n"
        "        self.execute_calls = 0\n"
        "    def execute_model(self):\n"
        "        self.execute_calls += 1\n"
        "        self.execute_model_state = object()\n"
        "        return None\n"
        "    def _copy_draft_token_ids_to_cpu(self, scheduler_output):\n"
        "        return None\n"
        "    def sample_tokens(self, grammar_output=None):\n"
        "        self.execute_model_state = None\n"
        "        sample_entered.set()\n"
        "        assert sample_continue.wait(2)\n"
        "        scheduler_output = object()\n"
        "        if True:\n"
        "            if True:\n"
        "                self._copy_draft_token_ids_to_cpu(scheduler_output)\n"
        "                # FR13_FIXED32_DRAFTER_PROPOSAL_SEALED\n"
        "                sample_released.set()\n"
        "                assert sample_finish.wait(2)\n"
        "        return 'sampled'\n"
    )
    patcher.GPU_MODEL_RUNNER_PATH = target
    assert patcher._patch_gpu_model_runner_exec_lock() is True
    source = target.read_text()
    gate_source = source.split(
        "# FR13_FIXED32_FLUSH: queue-only SIGUSR2 control plane",
        1,
    )[0]
    namespace: dict[str, object] = {}
    exec(compile(gate_source, str(target), "exec"), namespace)
    runner = namespace["GPUModelRunner"]()

    assert runner.execute_model() is None
    sample_result: list[object] = []
    sample_thread = threading.Thread(
        target=lambda: sample_result.append(runner.sample_tokens(None))
    )
    sample_thread.start()
    assert namespace["sample_entered"].wait(1)

    execute_done = threading.Event()

    def execute_again() -> None:
        runner.execute_model()
        execute_done.set()

    execute_thread = threading.Thread(target=execute_again)
    execute_thread.start()
    assert not execute_done.wait(0.05)

    namespace["sample_continue"].set()
    assert namespace["sample_released"].wait(1)
    execute_thread.join(1)
    assert not execute_thread.is_alive()
    assert sample_thread.is_alive()
    pending = namespace["_FR13_FIXED32_SAMPLE_PENDING"]
    assert len(pending) == 1
    assert namespace["_FR13_FIXED32_SAMPLE_FAILURE"] is None

    namespace["sample_finish"].set()
    sample_thread.join(1)
    assert not sample_thread.is_alive()
    assert sample_result == ["sampled"]
    assert execute_done.is_set()
    assert runner.execute_calls == 2
    assert len(pending) == 1

    assert runner.sample_tokens(None) == "sampled"
    assert pending == {}
    assert namespace["_FR13_FIXED32_SAMPLE_FAILURE"] is None


def test_fixed32_async_gate_raises_on_a_completed_unsealed_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patcher = _load_fixed32_patcher(monkeypatch)
    target = tmp_path / "gpu_model_runner.py"
    target.write_text(
        "class GPUModelRunner:\n"
        "    def __init__(self):\n"
        "        self.execute_model_state = None\n"
        "    def execute_model(self):\n"
        "        self.execute_model_state = object()\n"
        "        return None\n"
        "    def sample_tokens(self, should_seal=True):\n"
        "        self.execute_model_state = None\n"
        "        if should_seal:\n"
        "            if True:\n"
        "                # FR13_FIXED32_DRAFTER_PROPOSAL_SEALED\n"
        "        return 'sampled'\n"
    )
    patcher.GPU_MODEL_RUNNER_PATH = target
    assert patcher._patch_gpu_model_runner_exec_lock() is True
    source = target.read_text()
    gate_source = source.split(
        "# FR13_FIXED32_FLUSH: queue-only SIGUSR2 control plane",
        1,
    )[0]
    namespace: dict[str, object] = {}
    exec(compile(gate_source, str(target), "exec"), namespace)
    runner = namespace["GPUModelRunner"]()

    assert runner.execute_model() is None
    with pytest.raises(
        RuntimeError,
        match="sample returned before proposal seal",
    ):
        runner.sample_tokens(False)

    assert namespace["_FR13_FIXED32_SAMPLE_PENDING"] == {}
    assert namespace["_FR13_FIXED32_SAMPLE_FAILURE"] == (
        "sample returned before fixed32 proposal seal",
        None,
    )


@pytest.mark.parametrize(
    ("release_first", "failure_phase"),
    (
        (False, "before"),
        (True, "after"),
    ),
)
def test_fixed32_async_gate_preserves_the_original_sample_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    release_first: bool,
    failure_phase: str,
) -> None:
    patcher = _load_fixed32_patcher(monkeypatch)
    target = tmp_path / "gpu_model_runner.py"
    target.write_text(
        "class GPUModelRunner:\n"
        "    def __init__(self):\n"
        "        self.execute_model_state = None\n"
        "    def execute_model(self):\n"
        "        self.execute_model_state = object()\n"
        "        return None\n"
        "    def sample_tokens(self, release_first):\n"
        "        self.execute_model_state = None\n"
        "        if release_first:\n"
        "            if True:\n"
        "                # FR13_FIXED32_DRAFTER_PROPOSAL_SEALED\n"
        "        raise ValueError('original sample failure')\n"
    )
    patcher.GPU_MODEL_RUNNER_PATH = target
    assert patcher._patch_gpu_model_runner_exec_lock() is True
    source = target.read_text()
    gate_source = source.split(
        "# FR13_FIXED32_FLUSH: queue-only SIGUSR2 control plane",
        1,
    )[0]
    namespace: dict[str, object] = {}
    exec(compile(gate_source, str(target), "exec"), namespace)
    runner = namespace["GPUModelRunner"]()

    assert runner.execute_model() is None
    with pytest.raises(ValueError, match="original sample failure") as sample_error:
        runner.sample_tokens(release_first)

    failure = namespace["_FR13_FIXED32_SAMPLE_FAILURE"]
    assert failure[0] == (
        f"sample raised {failure_phase} fixed32 proposal seal: "
        "ValueError:original sample failure"
    )
    assert failure[1] is sample_error.value
    with pytest.raises(
        RuntimeError,
        match=(
            f"prior sample failed: sample raised {failure_phase} "
            "fixed32 proposal seal: ValueError:original sample failure"
        ),
    ) as execute_error:
        runner.execute_model()
    assert execute_error.value.__cause__ is sample_error.value
