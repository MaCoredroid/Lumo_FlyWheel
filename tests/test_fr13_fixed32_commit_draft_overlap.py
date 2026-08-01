from __future__ import annotations

import ast
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
PATCHER_PATH = ROOT / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER_PATH = ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"
SERVE_PATH = ROOT / "scripts/fr13_bigdenom_swe_serve_variant.sh"
KERNEL_SOURCE = KERNEL_PATH.read_text(encoding="utf-8")
PATCHER_SOURCE = PATCHER_PATH.read_text(encoding="utf-8")
KERNEL_TREE = ast.parse(KERNEL_SOURCE)
PATCHER_TREE = ast.parse(PATCHER_SOURCE)

OVERLAP_FUNCTIONS = {
    "fixed32_commit_draft_overlap_requested",
    "_fr13_fixed32_commit_draft_overlap_drain",
    "_fr13_fixed32_commit_draft_overlap_runtime",
    "fixed32_commit_draft_overlap_begin",
    "fixed32_commit_draft_overlap_state_enqueued",
    "fixed32_commit_draft_overlap_stream",
    "fixed32_commit_draft_overlap_seal",
    "fixed32_commit_draft_overlap_fence",
    "fixed32_commit_draft_overlap_snapshot",
}
OVERLAP_GLOBALS = {
    "_FR13_FIXED32_COMMIT_DRAFT_OVERLAP_ARM",
    "_FR13_FIXED32_COMMIT_DRAFT_OVERLAP_PATHS",
    "_FR13_FIXED32_COMMIT_DRAFT_OVERLAP_STATE",
}


def _function_source(tree: ast.Module, source: str, name: str) -> str:
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def _embedded_function_source(name: str) -> str:
    matches = []
    for node in ast.walk(PATCHER_TREE):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if f"def {name}(" not in node.value:
            continue
        try:
            embedded = ast.parse(node.value)
        except SyntaxError:
            continue
        for item in embedded.body:
            if isinstance(item, ast.FunctionDef) and item.name == name:
                matches.append(ast.get_source_segment(node.value, item) or "")
    assert len(matches) == 1
    return matches[0]


class _FakeEvent:
    _next_id = 0

    def __init__(self, log: list[tuple], *, enable_timing: bool = False):
        self.log = log
        self.enable_timing = enable_timing
        self.complete = False
        self.event_id = _FakeEvent._next_id
        _FakeEvent._next_id += 1

    def record(self, stream=None) -> None:
        self.complete = True
        self.log.append(("record", self.event_id, getattr(stream, "name", None)))

    def query(self) -> bool:
        return self.complete

    def elapsed_time(self, end) -> float:
        assert self.enable_timing and end.enable_timing
        return 2.5


class _FakeStream:
    def __init__(self, log: list[tuple], name: str):
        self.log = log
        self.name = name

    def wait_event(self, event: _FakeEvent) -> None:
        self.log.append(("wait", self.name, event.event_id))


class _FakeCuda:
    def __init__(self):
        self.log: list[tuple] = []
        self.default = _FakeStream(self.log, "default")
        self.active = self.default
        self.capturing = False

    def is_available(self) -> bool:
        return True

    def is_current_stream_capturing(self) -> bool:
        return self.capturing

    def Stream(self):
        return _FakeStream(self.log, "commit")

    def Event(self, *, enable_timing: bool = False):
        return _FakeEvent(self.log, enable_timing=enable_timing)

    def current_stream(self):
        return self.active

    @contextlib.contextmanager
    def stream(self, stream):
        prior = self.active
        self.active = stream
        try:
            yield
        finally:
            self.active = prior


def _overlap_namespace(tmp_path: Path) -> tuple[dict[str, object], _FakeCuda]:
    selected = []
    for node in KERNEL_TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in OVERLAP_GLOBALS
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id in OVERLAP_GLOBALS:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in OVERLAP_FUNCTIONS:
            selected.append(node)
    fake_cuda = _FakeCuda()
    namespace = {
        "Path": Path,
        "os": os,
        "json": json,
        "torch": SimpleNamespace(cuda=fake_cuda),
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), KERNEL_PATH, "exec"), namespace)
    namespace["_FR13_FIXED32_COMMIT_DRAFT_OVERLAP_PATHS"] = (
        str(tmp_path / "overlap.arm"),
    )
    return namespace, fake_cuda


def _arm(namespace: dict[str, object], tmp_path: Path) -> None:
    (tmp_path / "overlap.arm").write_text(
        str(namespace["_FR13_FIXED32_COMMIT_DRAFT_OVERLAP_ARM"]) + "\n",
        encoding="ascii",
    )


def test_arm_requires_exact_launcher_attestation(
    tmp_path: Path,
) -> None:
    namespace, _ = _overlap_namespace(tmp_path)
    requested = namespace["fixed32_commit_draft_overlap_requested"]
    assert requested() is False
    (tmp_path / "overlap.arm").write_text("1\n", encoding="ascii")
    assert requested() is False
    _arm(namespace, tmp_path)
    assert requested() is True


def test_two_slot_lifecycle_joins_after_tail_and_reconciles(
    tmp_path: Path,
) -> None:
    namespace, cuda = _overlap_namespace(tmp_path)
    _arm(namespace, tmp_path)
    identity = {
        "mode": "hydra27_fixed32",
        "batch": 4,
        "step_seq": 17,
        "request_ids": ("r0", "r1", "r2", "r3"),
    }
    stream = namespace["fixed32_commit_draft_overlap_begin"](**identity)
    assert stream.name == "commit"
    namespace["fixed32_commit_draft_overlap_state_enqueued"](**identity)
    assert namespace["fixed32_commit_draft_overlap_stream"](**identity) is stream
    namespace["fixed32_commit_draft_overlap_seal"](**identity)
    assert namespace["fixed32_commit_draft_overlap_fence"]() is True

    snapshot = namespace["fixed32_commit_draft_overlap_snapshot"](flush=True)
    assert snapshot == {
        "schema": "fr13.fixed32.commit_draft_overlap.v1",
        "armed": True,
        "arm_contract": "fr13.fixed32.k64.commit_draft_overlap.v1",
        "begun": 1,
        "sealed": 1,
        "fenced": 1,
        "flush_fenced": 0,
        "timed_spans": 1,
        "tail_gpu_ms_total": 2.5,
        "tail_gpu_ms_per_span": 2.5,
        "pending": False,
        "order_reconciled": True,
        "by_batch": {"1": 0, "2": 0, "3": 0, "4": 1},
        "event_slots": 2,
        "streams": 1,
    }
    records = [entry for entry in cuda.log if entry[0] == "record"]
    waits = [entry for entry in cuda.log if entry[0] == "wait"]
    assert records[0][2] == "default"
    assert records[1][2] == "commit"
    assert records[2][2] == "commit"
    assert waits[0][1] == "commit"
    assert waits[-1][1] == "default"


def test_lifecycle_fails_closed_on_capture_identity_and_order(
    tmp_path: Path,
) -> None:
    namespace, cuda = _overlap_namespace(tmp_path)
    _arm(namespace, tmp_path)
    begin = namespace["fixed32_commit_draft_overlap_begin"]
    cuda.capturing = True
    with pytest.raises(RuntimeError, match="cannot start during capture"):
        begin(
            mode="tail6_fixed32",
            batch=1,
            step_seq=1,
            request_ids=("r0",),
        )
    cuda.capturing = False
    identity = dict(
        mode="tail6_fixed32",
        batch=1,
        step_seq=1,
        request_ids=("r0",),
    )
    begin(**identity)
    with pytest.raises(RuntimeError, match="unsealed tail"):
        namespace["fixed32_commit_draft_overlap_fence"]()
    with pytest.raises(RuntimeError, match="identity"):
        begin(
            mode="tail6_fixed32",
            batch=1,
            step_seq=2,
            request_ids=("",),
        )


def test_patcher_orders_state_and_target_kv_on_one_side_stream() -> None:
    commit = _embedded_function_source("_fr13_fixed32_device_commit_route")
    remap = _function_source(
        PATCHER_TREE, PATCHER_SOURCE, "_patch_gpu_model_runner_attn_kv_remap_apply"
    )
    assert commit.index("_fixed_overlap_begin(") < commit.index("_fixed_conv_commit(")
    assert commit.index("_fixed_conv_commit(") < commit.index("_fixed_replay(")
    assert commit.index("_fixed_replay(") < commit.index("_fixed_overlap_state_done(")
    assert "torch.cuda.stream(_fixed_overlap_stream)" in commit
    assert remap.index("_fr13_f32_overlap_stream(") < remap.index("_fr13_f32_kv16(")
    assert remap.index("_fr13_f32_kv16(") < remap.index("_fr13_f32_overlap_seal(")
    assert "torch.cuda.stream(_fr13_f32_overlap)" in remap


def test_dfwd_fence_precedes_copy_and_connector_finalization() -> None:
    replay = _function_source(
        PATCHER_TREE, PATCHER_SOURCE, "_patch_gpu_model_runner_replay_draft_reqkey"
    )
    timer_end = replay.index('"                _fr13_dfwd_end(_fr13_dfwd_ev)')
    first_fence = replay.index(
        '"                    _fr13_f32_overlap_fence()', timer_end
    )
    draft_copy = replay.index(
        '"                self._copy_draft_token_ids_to_cpu(scheduler_output)',
        timer_end,
    )
    fallback = replay.rindex('"            _fr13_f32_overlap_fence()')
    connector = replay.index('"            self.finalize_kv_connector()', fallback)
    assert timer_end < first_fence < draft_copy
    assert fallback < connector


def test_overlap_keeps_target_and_drafter_kv_owners_disjoint() -> None:
    remap = _function_source(
        PATCHER_TREE, PATCHER_SOURCE, "_patch_gpu_model_runner_attn_kv_remap_apply"
    )
    assert "'_fr13_fixed32_kv_caches'" in remap
    assert "'_fr13_fixed32_mtp_kv_cache'" in remap
    target_call = remap.split("_fr13_f32_kv16(", 1)[1].split(")\\n\"", 1)[0]
    assert "kv_caches=_fr13_f32_kvs" in target_call
    assert "_fr13_f32_mtp_kv" not in target_call
    assert "'mtp_kv': _fr13_f32_mtp_kv" in remap


def test_flush_persists_terminal_tail_census_after_global_sync() -> None:
    flush = _embedded_function_source("_fr13_f32_flush_one")
    assert flush.index("torch.cuda.synchronize()") < flush.index(
        "fixed32_commit_draft_overlap_snapshot(flush=True)"
    )
    assert 'generation=request["generation"]' in flush
    assert 'action=request["action"]' in flush
    assert "overlap_path.exists()" in flush
    snapshot = _function_source(
        KERNEL_TREE, KERNEL_SOURCE, "fixed32_commit_draft_overlap_snapshot"
    )
    assert "begun == sealed == fences == spans" in snapshot
    assert "request_ids" not in snapshot


def test_launcher_confines_candidate_to_k64_root_fixed32_b1_or_b4() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    serve = SERVE_PATH.read_text(encoding="utf-8")
    assert "FR13_FIXED32_COMMIT_DRAFT_OVERLAP must be 0 or 1" in launcher
    assert "FR13_FIXED32_COMMIT_DRAFT_OVERLAP must be exactly 0 or 1" in serve
    assert '"$MAX_NUM_SEQS" == "1" || "$MAX_NUM_SEQS" == "4"' in launcher
    assert '"$FR13_DRAFT_VOCAB_ROOT" == "1"' in launcher
    assert '"${FR13_DRAFT_VOCAB_K:-65536}" == "65536"' in launcher
    assert '"${FR13_COMMIT_OVERLAP:-0}" == "0"' in launcher
    assert '"${FR13_REPLAY_MULTISTREAM:-0}" == "0"' in launcher
    assert '"${FR13_FIXED32_COMMITTER_LAYER_BATCH:-0}" == "0"' in launcher
    assert "fr13.fixed32.k64.commit_draft_overlap.v1" in launcher
    assert 'chmod 0400 "$LOG_DIR/fr13_fixed32_commit_draft_overlap.arm"' in launcher
    assert '-e FR13_FIXED32_COMMIT_DRAFT_OVERLAP=' in launcher
