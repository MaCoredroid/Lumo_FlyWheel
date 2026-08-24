"""SITE 26's siblings: the kill path, and the evaluator's identity.

(2) THE KILL PATH. The standing suspicion was that the remote kill is IDLE-based
    and a gaplessly streaming task never trips it, so the 9000s budget fired
    ~45 minutes late or not at all. MEASURED ON DISK, THE HYPOTHESIS IS REFUTED:

      * enforcement is HOST-SIDE, on the SSH subprocess, and it is a true WALL
        deadline -- ``max(timeout_s, 30) + 120`` in the instance-image runner
        (the only runner a fixed32 arm may use) and the same expression in the
        legacy remote runner's stall-watchdog monitor;
      * on expiry the harness sets ``timed_out`` and docker-kills the remote
        container, which is what produced the two capped terminals;
      * the two capped tasks ran 9125s (Cqc12 astropy__astropy-13579) and 9124s
        (Cqc15 astropy__astropy-13398) measured pre-bracket to trace-fetch,
        against a 9000s budget plus the 120s teardown buffer -- on time to
        within the 5s poll, not 45 minutes late. The ~195-minute figure does not
        match either runroot; both ran 152 minutes;
      * ``codex-bench-eval-swe``, whose header describes stream-idle and
        turn-limit semantics, is the EVAL harness invoked by ``_run_eval`` on a
        finished patch. It is not in the agent's kill path at all.

    So there is nothing to fix, and this file pins the semantics instead: a wall
    deadline that quietly became idle-based would be exactly the defect that was
    suspected, and now it cannot happen unnoticed.

(3) THE FOURTH SHADOW. Which evaluator binary actually resolves, recorded into
    provenance and refused when it comes from another checkout.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_runner() -> Any:
    path = SCRIPTS / "run_swe_bench_q36_a.py"
    spec = importlib.util.spec_from_file_location("fr14_killpath_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER_SOURCE = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# (2) the wall deadline is real, host-side, and not idle-based                 #
# --------------------------------------------------------------------------- #
def test_wall_deadline_kills_a_process_that_never_goes_idle() -> None:
    """The monitor's wall is a WALL: a busy, growing task is still killed.

    The trace file grows on every poll, so an idle-based kill would never fire.
    The wall must fire anyway -- that is the whole difference the hypothesis
    turned on.
    """
    runner = _load_runner()
    with tempfile.TemporaryDirectory(prefix="fr14-wall-") as raw:
        trace_path = Path(raw) / "trace.jsonl"
        trace_path.write_text("", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys,time\n"
                    "path=sys.argv[1]\n"
                    "for _ in range(600):\n"
                    "    open(path,'a').write('x'*64+'\\n')\n"
                    "    time.sleep(0.05)\n"
                ),
                str(trace_path),
            ]
        )
        started = time.monotonic()
        try:
            result = runner._monitor_proc_with_stall_watchdog(
                proc,
                trace_path=trace_path,
                wall_timeout_s=1.0,
                # The stall watchdog is OFF, exactly as the campaign runs it.
                stall_kill_s=0.0,
                poll_s=0.1,
            )
        finally:
            if proc.poll() is None:  # pragma: no cover - defensive
                proc.kill()
                proc.wait(timeout=10)
        grown_bytes = trace_path.stat().st_size
    elapsed = time.monotonic() - started
    assert result["timed_out"] is True
    assert result["stall_killed"] is False
    # It fired at the wall, not at the 30s the process would have run for.
    assert elapsed < 10.0
    # And the task WAS streaming throughout, so an idle rule would never have
    # fired on it -- which is precisely the hypothesis this refutes.
    assert grown_bytes > 0


def test_wall_deadline_does_not_fire_on_a_process_that_finishes() -> None:
    runner = _load_runner()
    with tempfile.TemporaryDirectory(prefix="fr14-wall-ok-") as raw:
        trace_path = Path(raw) / "trace.jsonl"
        trace_path.write_text("", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        result = runner._monitor_proc_with_stall_watchdog(
            proc,
            trace_path=trace_path,
            wall_timeout_s=30.0,
            stall_kill_s=0.0,
            poll_s=0.1,
        )
    assert result["timed_out"] is False
    assert result["returncode"] == 0


def test_the_instance_image_runner_carries_a_true_wall_deadline() -> None:
    """The runner a fixed32 arm must use enforces the wall on the SSH process.

    ``_run_agent_instance`` is where the campaign's kill actually happens
    (``_run_agent_remote`` delegates to it whenever the agent env is
    ``instance_image``, which the provenance validator REQUIRES). Its
    ``subprocess.run(..., timeout=...)`` is the deadline, and TimeoutExpired is
    what sets ``timed_out`` and stops the remote container.
    """
    assert "max(timeout_s, 30) + 120" in RUNNER_SOURCE
    assert "except subprocess.TimeoutExpired:" in RUNNER_SOURCE
    assert "_stop_remote_agent_container(" in RUNNER_SOURCE
    # The legacy remote path computes the same wall for its monitor.
    assert "wall_timeout_s = None if timeout_s <= 0 else max(timeout_s, 30) + 120" in (
        RUNNER_SOURCE
    )


def test_the_campaign_budget_is_the_tighter_of_wall_and_budget() -> None:
    runner = _load_runner()
    assert runner._campaign_budget_effective_timeout_s(20_000, 9_000.0) == 9_000
    assert runner._campaign_budget_effective_timeout_s(5_000, 9_000.0) == 5_000
    assert runner._campaign_budget_effective_timeout_s(0, 9_000.0) == 9_000


def test_a_capped_terminal_is_attributed_only_when_the_wall_actually_ran_out() -> None:
    """The attribution is measured, never assumed."""
    runner = _load_runner()
    capped = runner._attribute_campaign_budget_cap(
        {"timed_out": True, "elapsed_s": 9_005.0},
        budget_s=9_000.0,
        requested_wall_s=20_000,
    )
    assert capped["budget_capped"] is True
    assert capped["campaign_budget_was_the_binding_limit"] is True

    early = runner._attribute_campaign_budget_cap(
        {"timed_out": True, "elapsed_s": 100.0},
        budget_s=9_000.0,
        requested_wall_s=20_000,
    )
    assert early["budget_capped"] is False

    not_binding = runner._attribute_campaign_budget_cap(
        {"timed_out": True, "elapsed_s": 9_005.0},
        budget_s=9_000.0,
        requested_wall_s=1_000,
    )
    assert not_binding["budget_capped"] is False
    assert not_binding["campaign_budget_was_the_binding_limit"] is False


def test_the_eval_harness_is_not_in_the_agent_kill_path() -> None:
    """codex-bench-eval-swe scores a finished patch; it never runs the agent."""
    # The entry point is resolved in exactly one place, and that place is the
    # evaluator resolver -- not any of the four agent-runner bodies.
    resolver = RUNNER_SOURCE.index("def _local_evaluator_identity(")
    resolver_end = RUNNER_SOURCE.index("def _remote_evaluator_identity(")
    executable_hits = [
        line
        for line in RUNNER_SOURCE.splitlines()
        if 'shutil.which("codex-bench-eval-swe")' in line
        and not line.lstrip().startswith("#")
    ]
    assert len(executable_hits) == 1
    assert 'shutil.which("codex-bench-eval-swe")' in (
        RUNNER_SOURCE[resolver:resolver_end]
    )
    for runner_name in (
        "def _run_agent_local(",
        "def _run_agent_remote(",
        "def _run_agent_instance(",
        "def _run_agent_dispatch(",
    ):
        assert runner_name in RUNNER_SOURCE
    # The eval harness is invoked only from _run_eval, on a finished patch.
    assert RUNNER_SOURCE.count("cbe_exe") == 2


# --------------------------------------------------------------------------- #
# (3) the fourth shadow: which evaluator resolves, and is it ours              #
# --------------------------------------------------------------------------- #
def test_local_evaluator_identity_names_every_part_of_the_resolution() -> None:
    runner = _load_runner()
    identity = runner._local_evaluator_identity()
    assert identity["mode"] == "local_console_script"
    assert identity["resolved_via"] in {"PATH", "repo_venv_fallback"}
    assert set(identity) >= {
        "which",
        "resolved_via",
        "repo_root",
        "venv_link_target",
        "path",
        "interpreter",
        "inside_repo",
        "interpreter_inside_repo",
    }
    if identity["readable"]:
        assert len(identity["sha256"]) == 64


def test_the_shared_venv_shadow_is_visible_in_the_identity() -> None:
    """On this host the fallback resolves OUTSIDE the repo. Say so, in the record.

    If someone later makes ``.venv`` a real in-repo virtualenv this assertion
    flips to the happy branch rather than failing -- the point is that the
    record distinguishes the two, not that the shadow is permanent.
    """
    runner = _load_runner()
    identity = runner._local_evaluator_identity()
    if identity["resolved_via"] == "repo_venv_fallback" and identity["readable"]:
        realpath = Path(identity["realpath"])
        repo = Path(identity["repo_root"]).resolve()
        assert identity["inside_repo"] == realpath.is_relative_to(repo)


def test_fixed32_refuses_an_evaluator_from_another_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verdict scored by another checkout's evaluator is not our evidence."""
    runner = _load_runner()
    monkeypatch.setattr(runner, "EVAL_HOST", None, raising=False)
    monkeypatch.setattr(
        runner,
        "_local_evaluator_identity",
        lambda: {
            "mode": "local_console_script",
            "which": None,
            "resolved_via": "repo_venv_fallback",
            "repo_root": str(REPO),
            "venv_link_target": "/home/mark/shared/lumoFlyWheel/.venv",
            "path": "/home/mark/shared/lumoFlyWheel/.venv/bin/codex-bench-eval-swe",
            "realpath": "/home/mark/shared/lumoFlyWheel/.venv/bin/codex-bench-eval-swe",
            "readable": True,
            "bytes": 221,
            "sha256": "0" * 64,
            "interpreter": "/home/mark/shared/lumoFlyWheel/.venv/bin/python3",
            "inside_repo": False,
            "interpreter_inside_repo": False,
        },
    )
    with tempfile.TemporaryDirectory(prefix="fr14-eval-") as raw:
        root = Path(raw)
        with pytest.raises(runner.Fixed32BoundaryError) as error:
            runner._run_eval(
                instance_id="astropy__astropy-12907",
                patch_path=root / "patch.diff",
                output_dir=root / "out",
                dataset_name="princeton-nlp/SWE-bench_Verified",
                model_name="qwen",
                timeout_s=60,
                eval_log_path=root / "eval.log",
                fixed32=True,
            )
    message = str(error.value)
    assert "local evaluator resolves outside the repository" in message
    assert "repo_venv_fallback" in message
    assert "lumoFlyWheel/.venv" in message


def test_a_non_fixed32_run_still_uses_the_local_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal is scoped to evidence-grade arms; ad-hoc runs are unchanged."""
    runner = _load_runner()
    monkeypatch.setattr(runner, "EVAL_HOST", None, raising=False)
    seen: dict[str, Any] = {}

    def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    with tempfile.TemporaryDirectory(prefix="fr14-eval-ok-") as raw:
        root = Path(raw)
        meta = runner._run_eval(
            instance_id="astropy__astropy-12907",
            patch_path=root / "patch.diff",
            output_dir=root / "out",
            dataset_name="princeton-nlp/SWE-bench_Verified",
            model_name="qwen",
            timeout_s=60,
            eval_log_path=root / "eval.log",
        )
    assert meta["exit_code"] == 0
    assert meta["evaluator"]["mode"] == "local_console_script"
    assert seen["cmd"][0] == meta["evaluator"]["path"]


def test_the_evaluator_identity_reaches_the_task_record() -> None:
    """Both eval routes attach an ``evaluator`` block to the summary."""
    assert '"evaluator": evaluator,' in RUNNER_SOURCE
    assert '"evaluator": _remote_evaluator_identity(host)' in RUNNER_SOURCE
    assert "fixed32=fixed32_bracket is not None," in RUNNER_SOURCE
