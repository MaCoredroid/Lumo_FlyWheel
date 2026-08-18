"""`FR13_DFWD_SPLIT` must never synchronize inside a CUDA capture window.

Reconciliation of the Arm G failure (runroot
`output/fr14_promoab_Giso_20260818T074147Z`): the
`cudaErrorStreamCaptureUnsupported` in `logs/fr13_dfwd_split.err` was
**secondary**. Ordering in `docker_after_tasks.log` is unambiguous — the
tree-attention refusal and `EngineDeadError` at 07:47:27, then `CUDAGraph.cpp:275
... reset` warnings at 07:47:28 (both graph objects destroyed with the capture
still open), then a single atexit `dump()` whose device-wide sync could not run.
Exactly one entry in the file; no periodic-dump entries during the run.

The hazard is nevertheless **real independently of that defect**, which is why it
gets its own guard: the periodic dump fires from inside the drafter's
instrumented model span (`end('model')`, every 25 levels), and the drafter graph
capture executes exactly that span. Whether a 25-boundary lands inside a capture
window is a race on the pre-capture forward count. A device-wide sync there is
not merely refused — it can INVALIDATE the in-flight capture, so an instrument
would be breaking the thing it measures.

The helper is injected source text, so these tests extract and exec it against a
fake torch, the same discipline that caught the 11th integration site.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"


def _helper_source():
    src = PATCHER.read_text()
    m = re.search(
        r"class _Fr13DfwdSplit:.*?\n_FR13_DFWD_SPLIT = _Fr13DfwdSplit\(\)",
        src,
        re.S,
    )
    assert m, "FR13_DFWD_SPLIT helper not found in the patcher"
    return m.group(0)


class _FakeEvent:
    def __init__(self, *a, **k):
        pass

    def record(self):
        pass

    def elapsed_time(self, other):
        return 1.0


def _install_fake_torch(monkeypatch, *, capturing, sync_raises=None):
    calls = {"sync": 0}

    def synchronize():
        calls["sync"] += 1
        if sync_raises is not None:
            raise sync_raises

    cuda = types.SimpleNamespace(
        Event=_FakeEvent,
        synchronize=synchronize,
        is_current_stream_capturing=lambda: capturing,
    )
    fake = types.ModuleType("torch")
    fake.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", fake)
    return calls


def _make(monkeypatch, tmp, **kw):
    """Exec the injected helper and return an armed instance."""
    calls = _install_fake_torch(monkeypatch, **kw)
    monkeypatch.setenv("FR13_DFWD_SPLIT", "1")
    monkeypatch.setenv("FR13_DFWD_SPLIT_JSON", str(Path(tmp) / "split.json"))
    ns = {}
    exec(_helper_source(), ns)
    inst = ns["_FR13_DFWD_SPLIT"]
    assert inst.on, "helper did not arm from FR13_DFWD_SPLIT=1"
    return inst, calls


@pytest.fixture
def tmp():
    import tempfile

    return tempfile.mkdtemp()


def test_dump_does_not_sync_while_the_current_stream_is_capturing(
    monkeypatch, tmp
):
    inst, calls = _make(monkeypatch, tmp, capturing=True)
    inst.dump()
    assert calls["sync"] == 0, "an instrument must not sync inside a capture"
    assert inst.done is False, "the dump must stay pending, not be marked done"


def test_dump_syncs_and_writes_when_no_capture_is_active(monkeypatch, tmp):
    import json

    inst, calls = _make(monkeypatch, tmp, capturing=False)
    inst.pairs["model"].append((_FakeEvent(), _FakeEvent()))
    inst.dump()
    assert calls["sync"] == 1
    written = list(Path(tmp).glob("split.json.*"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["schema"] == "fr13.dfwd_split.v1"


def test_a_capture_on_another_stream_is_recorded_as_a_deferral(monkeypatch, tmp):
    """is_current_stream_capturing() cannot see a capture on a different stream."""
    err = RuntimeError(
        "CUDA error: operation not permitted when stream is capturing"
    )
    inst, calls = _make(monkeypatch, tmp, capturing=False, sync_raises=err)
    monkeypatch.setattr(inst, "_defer", lambda why: recorded.append(why))
    recorded = []
    inst.dump()
    assert calls["sync"] == 1
    assert recorded and "stream is capturing" in recorded[0]
    assert inst.done is False


def test_a_real_failure_is_still_reported_with_its_traceback(monkeypatch, tmp):
    """The guard must not swallow genuine instrument faults."""
    inst, calls = _make(
        monkeypatch, tmp, capturing=False, sync_raises=RuntimeError("disk on fire")
    )
    written = []
    monkeypatch.setattr(inst, "_defer", lambda why: written.append(("defer", why)))
    inst.dump()
    assert not written, "a non-capture failure must not be classified as a deferral"


def test_periodic_dump_is_guarded_too(monkeypatch, tmp):
    """end() fires dump() every 25 model levels -- from inside the capture span."""
    inst, calls = _make(monkeypatch, tmp, capturing=True)
    for _ in range(25):
        inst.end("model", _FakeEvent())
    assert calls["sync"] == 0
    assert len(inst.pairs["model"]) == 25, "timing pairs are still collected"


def test_the_guard_precedes_the_sync_in_source_order():
    """Order matters: checking after the call would already have failed."""
    helper = _helper_source()
    body = helper[helper.index("    def dump(self):"):]
    guard = body.index("is_current_stream_capturing()")
    sync = body.index("_t.cuda.synchronize()")
    assert guard < sync
