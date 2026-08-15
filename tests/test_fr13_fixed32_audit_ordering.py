"""The audit must be able to run on the evidence the failure leaves behind.

THE DEFECT, as observed rather than imagined
--------------------------------------------
A task's runner metadata is written first as ``runner_metadata.pending.json``
and promoted to ``runner_metadata.json`` by the end-of-campaign finalizer. The
fixed32 chat-task traffic audit required the PROMOTED file.

But the finalizer promotes nothing until it has reconciled the campaign's token
accounting, and it raises on the first task when that reconciliation fails. So a
campaign whose accounting is wrong leaves EVERY task unpromoted -- and the audit
whose job is to diagnose exactly that class of failure could not run at all. It
demanded the artifact whose absence is the symptom.

Both arms of the 2026-08-15 width-4 screen hit this: 16 of 16 tasks pending,
zero promoted, the audit dead on the alphabetically-first task
(astropy__astropy-12907), six hours of otherwise clean serving per arm left
undiagnosable by its own instrument. The apparent "12907 is corrupt" reading was
an artifact of alphabetical order; nothing was wrong with 12907.

WHAT THESE TESTS PIN
--------------------
1. The audit reader accepts the pending form. Nothing else does -- the opt-in
   defaults to False, so every promoted-metadata gate is bit-unchanged.
2. Promotion state travels INTO the evidence. An audit run on unpromoted
   metadata is weaker than one run on published metadata, and that difference
   must be legible in the artifact rather than reconstructed from a directory
   listing months later.
3. The promoted form still wins when both exist, so a stale pending file can
   never shadow published metadata.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fr13_floor_gate as fg  # noqa: E402


def _task(tmp: Path, task_id: str, *, promoted: bool, pending: bool, **over):
    d = tmp / task_id
    d.mkdir(parents=True, exist_ok=True)
    body = {"instance_id": task_id, "ended_at": "2026-08-15T08:03:25Z"}
    body.update(over)
    if promoted:
        (d / "runner_metadata.json").write_text(json.dumps(body), encoding="ascii")
    if pending:
        (d / "runner_metadata.pending.json").write_text(
            json.dumps(body), encoding="ascii"
        )
    return d


def test_promoted_metadata_is_read_and_reported_as_promoted(tmp_path) -> None:
    d = _task(tmp_path, "astropy__astropy-12907", promoted=True, pending=False)
    meta, path, promoted = fg.fixed32_runner_metadata(d, allow_unpromoted=False)
    assert promoted is True
    assert path.name == "runner_metadata.json"
    assert meta["instance_id"] == "astropy__astropy-12907"


def test_pending_metadata_is_refused_by_default(tmp_path) -> None:
    """Every non-audit caller keeps the old behaviour, byte for byte."""
    d = _task(tmp_path, "astropy__astropy-12907", promoted=False, pending=True)
    with pytest.raises(fg.GateError) as excinfo:
        fg.fixed32_runner_metadata(d, allow_unpromoted=False)
    # and the error still names the PROMOTED path, so the message a reader has
    # seen for months does not change shape. Compare the FILENAME, not the whole
    # string: pytest's tmp_path is named after the test, so the word "pending"
    # appears in the directory component and would make a naive check pass or
    # fail for reasons that have nothing to do with the code.
    assert "runner_metadata.json" in str(excinfo.value)
    assert "runner_metadata.pending.json" not in str(excinfo.value)


def test_pending_metadata_is_accepted_when_the_audit_opts_in(tmp_path) -> None:
    """THE FIX. This is the exact shape both screen arms were in."""
    d = _task(tmp_path, "astropy__astropy-12907", promoted=False, pending=True)
    meta, path, promoted = fg.fixed32_runner_metadata(d, allow_unpromoted=True)
    assert promoted is False
    assert path.name == "runner_metadata.pending.json"
    assert meta["instance_id"] == "astropy__astropy-12907"


def test_promoted_wins_when_both_exist(tmp_path) -> None:
    """A stale pending file must never shadow published metadata."""
    d = _task(tmp_path, "astropy__astropy-12907", promoted=True, pending=True)
    _meta, path, promoted = fg.fixed32_runner_metadata(d, allow_unpromoted=True)
    assert promoted is True
    assert path.name == "runner_metadata.json"


def test_missing_both_forms_still_fails(tmp_path) -> None:
    d = tmp_path / "astropy__astropy-12907"
    d.mkdir(parents=True)
    with pytest.raises(fg.GateError):
        fg.fixed32_runner_metadata(d, allow_unpromoted=True)


def test_task_directories_opt_in_defaults_to_off() -> None:
    """The signature itself is the guarantee that existing gates are unchanged."""
    import inspect

    sig = inspect.signature(fg.task_directories)
    param = sig.parameters["allow_unpromoted_metadata"]
    assert param.default is False
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_only_the_audit_opts_in() -> None:
    """Exactly one caller may read unpromoted metadata."""
    src = (REPO / "scripts" / "fr13_floor_gate.py").read_text(encoding="utf-8")
    assert src.count("allow_unpromoted_metadata=True") == 1
    # exactly one direct opt-in; task_directories forwards its parameter rather
    # than hardcoding True, which is what keeps the default reachable
    assert src.count("allow_unpromoted=True") == 1
    assert "allow_unpromoted=allow_unpromoted_metadata" in src
    start = src.index("def build_fixed32_chat_traffic_audit")
    end = src.index("\ndef ", start + 1)
    audit = src[start:end]
    assert "allow_unpromoted_metadata=True" in audit


def test_the_audit_payload_carries_promotion_state() -> None:
    src = (REPO / "scripts" / "fr13_floor_gate.py").read_text(encoding="utf-8")
    start = src.index("def build_fixed32_chat_traffic_audit")
    end = src.index("\ndef ", start + 1)
    audit = src[start:end]
    assert '"metadata_promotion"' in audit
    assert '"unpromoted_metadata_task_ids"' in audit
    assert '"all_task_metadata_promoted"' in audit
    # the flag must be derived from what was read, not hardcoded optimistic
    assert "not unpromoted_metadata_task_ids" in audit
    assert "unpromoted_metadata_task_ids.append(task_id)" in audit


def test_the_helper_explains_why_the_exception_exists() -> None:
    """The reason is the load-bearing part; a future reader must not delete it blind."""
    src = (REPO / "scripts" / "fr13_floor_gate.py").read_text(encoding="utf-8")
    start = src.index("def fixed32_runner_metadata")
    end = src.index("\ndef ", start + 1)
    helper = src[start:end]
    for phrase in ("promotes nothing", "absence is the symptom", "defaults to False"):
        assert phrase in helper, phrase
