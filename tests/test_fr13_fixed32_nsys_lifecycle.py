from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "scripts" / "fr13_fixed32_b1_nsys_profile.sh"
SEQUENCE = REPO / "scripts" / "fr13_fixed32_floor_timers_seq.sh"
VARIANT = REPO / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"
GUARD = REPO / "scripts" / "gpu_oom_guard.sh"
CONTAINER_ID = "a" * 64
FOREIGN_CONTAINER_ID = "b" * 64
CONTAINER_STARTED_AT = "2026-07-30T12:00:00.123456789Z"
CONTAINER_INIT_PID = "4242"
CONTAINER_RESTART_COUNT = "0"
SESSION_ID = "10206"
SESSION_NAME = "fr13-fixed32-20260730T120000Z-p4242"
INJECTION_LIB = (
    "/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-sbsa-armv8"
    "/libToolsInjection64.so"
)
CONTAINER_NSYS_BIN = "/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys"
CONTAINER_TIMEOUT_BIN = "/usr/bin/timeout"
CONTAINER_BASH_BIN = "/bin/bash"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _shell_function(path: Path, name: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{name}() {{")
    for end in range(start + 1, len(lines)):
        if lines[end] == "}":
            return "\n".join(lines[start : end + 1])
    raise AssertionError(f"unterminated shell function: {name}")


def _container_nsys_exec(
    command: str,
    *,
    timeout_s: str = "10",
    kill_after_s: str = "2",
) -> str:
    return (
        f"exec {CONTAINER_ID} {CONTAINER_TIMEOUT_BIN} --signal=TERM "
        f"--kill-after={kill_after_s}s {timeout_s}s "
        f"{CONTAINER_NSYS_BIN} {command}"
    )


def _incarnation_attestation() -> str:
    return (
        "inspect --format "
        "{{.Id}} {{.Name}} {{.State.Status}} {{.State.Running}} "
        "{{.State.StartedAt}} {{.State.Pid}} {{.RestartCount}} "
        f"{{{{.State.ExitCode}}}} {CONTAINER_ID}"
    )


def _write_process_identity(
    path: Path,
    session_id: str | None = SESSION_ID,
    *,
    injection: bool = True,
    session_name: str = SESSION_NAME,
) -> None:
    """Write a process identity artifact AND the engine-side Nsight attestation.

    ``session_id=None`` models the pre-delay runtime: nsys sets
    NSYS_PROFILING_SESSION_ID when the session actually starts, so before the
    capture delay elapses the engine legitimately has no id yet. That is
    tolerated; what is not tolerated is a missing/mismatched injection pair or a
    session name from another run.

    FR14: the Nsight variables are no longer read from /proc/<engine>/environ.
    setproctitle("VLLM::EngineCore") NUL-fills that process's argv+environ block
    -- measured 22 variables to zero in this image -- so the engine publishes its
    own os.environ to a /logs artifact and the profiler attests THAT. The fixture
    writes both, since both exist at runtime.
    """
    engine_environ = []
    if session_id is not None:
        engine_environ.append(f"NSYS_PROFILING_SESSION_ID={session_id}")
    if injection:
        engine_environ.extend(
            [
                f"NVTX_INJECTION64_PATH={INJECTION_LIB}",
                f"NSYSDK_INJECTION64_PATH={INJECTION_LIB}",
            ]
        )
    path.write_text(
        json.dumps(
            {
                "schema": "fr13-fixed32-process-identity-v1",
                "pid1": {
                    "pid": 1,
                    "argv": ["nsys", "profile", "vllm", "serve"],
                    "environ": [f"LUMO_NSYS_SESSION_NAME={session_name}"],
                    "forked_fa2_maps": [],
                },
                "engine_core": {
                    "pid": 321,
                    "argv": ["VLLM::EngineCore"],
                    "environ": sorted(engine_environ),
                    "forked_fa2_maps": [],
                },
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    engine_env = {"LUMO_NSYS_SESSION_NAME": session_name}
    if injection:
        engine_env["NVTX_INJECTION64_PATH"] = INJECTION_LIB
        engine_env["NSYSDK_INJECTION64_PATH"] = INJECTION_LIB
    if session_id is not None:
        engine_env["NSYS_PROFILING_SESSION_ID"] = session_id
    Path(f"{path}.attestation.json").write_text(
        json.dumps(
            {
                "schema": "fr13.fixed32.enginecore_nsight_attestation.v1",
                "pid": 321,
                "environ": engine_env,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

def _write_identity_docker(
    path: Path,
    *,
    mutable_name_file: bool = False,
) -> None:
    name_source = '$(<"$CONTAINER_NAME_FILE")' if mutable_name_file else "$CONTAINER_NAME"
    _write_executable(
        path,
        f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
if [[ "$1" == "inspect" && "$2" == "--format" ]]; then
  [[ "$4" == "$CONTAINER_ID" ]] || exit 1
  [[ ! -e "${{CONTAINER_MISSING_FILE:-/nonexistent}}" ]] || exit 1
  if [[ "$3" == "{{{{.Id}}}} {{{{.Name}}}} {{{{.State.Status}}}} {{{{.State.Running}}}} {{{{.State.StartedAt}}}} {{{{.State.Pid}}}} {{{{.RestartCount}}}} {{{{.State.ExitCode}}}}" ]]; then
    if [[ -n "${{CONTAINER_STATE_FILE:-}}" ]]; then
      read -r state_status state_running state_started_at state_pid \
        state_restart_count state_exit_code < "$CONTAINER_STATE_FILE"
    else
      state_status=${{CONTAINER_STATUS:-running}}
      state_running=${{CONTAINER_RUNNING:-true}}
      state_started_at=${{CONTAINER_STARTED_AT:-{CONTAINER_STARTED_AT}}}
      state_pid=${{CONTAINER_STATE_PID:-{CONTAINER_INIT_PID}}}
      state_restart_count=${{CONTAINER_RESTART_COUNT:-{CONTAINER_RESTART_COUNT}}}
      state_exit_code=${{CONTAINER_EXIT_CODE:-0}}
    fi
    printf '%s /%s %s %s %s %s %s %s\\n' \
      "$CONTAINER_ID" "{name_source}" \
      "$state_status" "$state_running" "$state_started_at" "$state_pid" \
      "$state_restart_count" "$state_exit_code"
  elif [[ "$3" == "{{{{.Id}}}} {{{{.Name}}}}" ]]; then
    printf '%s /%s\\n' "$CONTAINER_ID" "{name_source}"
  else
    exit 1
  fi
  exit 0
fi
if [[ "$1" == "exec" ]]; then
  [[ "$2" == "$CONTAINER_ID" ]] || exit 1
  [[ "$3" == "{CONTAINER_TIMEOUT_BIN}" ]] || exit 1
  [[ "$4" == "--signal=TERM" && "$5" == --kill-after=* ]] || exit 1
  if [[ "$7" == "{CONTAINER_NSYS_BIN}" ]]; then
    FAKE_NSYS_CONTAINER_CONTEXT=1 exec /usr/bin/timeout \
      "$4" "$5" "$6" "$CONTAINER_NSYS" "${{@:8}}"
  fi
  if [[ "$7" == "{CONTAINER_BASH_BIN}" ]]; then
    [[ ! -e "${{ENGINE_CORE_DEAD_FILE:-/nonexistent}}" ]] || exit 1
    printf '%s\\n' "${{ENGINE_CORE_START_TICKS:-9001}}"
    exit 0
  fi
  exit 1
  exit 0
fi
exit 1
""",
    )


def _write_engine_snapshot_docker(path: Path) -> None:
    _write_executable(
        path,
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
if [[ "$1" == "inspect" && "$2" == "--format" ]]; then
  [[ "$4" == "$CONTAINER_ID" ]] || exit 1
  [[ "$3" == "{{.Id}} {{.Name}} {{.State.Status}} {{.State.Running}} {{.State.StartedAt}} {{.State.Pid}} {{.RestartCount}} {{.State.ExitCode}}" ]] \
    || exit 1
  restart_count=$CONTAINER_RESTART_COUNT
  [[ ! -e "$IDENTITY_DRIFT_FILE" ]] || restart_count=1
  printf '%s /%s exited false %s 0 %s 0\\n' \
    "$CONTAINER_ID" "$CONTAINER_NAME" "$CONTAINER_STARTED_AT" "$restart_count"
  exit 0
fi
if [[ "$1" == "cp" ]]; then
  [[ "$2" == "$CONTAINER_ID:/logs/fr13_fixed32_engine_ingress.jsonl" ]] \
    || exit 1
  [[ "$CP_MODE" != "cp_failure" ]] || exit 42
  /bin/cp -- "$ENGINE_LEDGER_SOURCE" "$3" || exit 1
  if [[ "$CP_MODE" == "identity_drift" ]]; then
    : > "$IDENTITY_DRIFT_FILE"
  fi
  exit 0
fi
exit 1
""",
    )


def _fake_control_processes(tmp_path: Path) -> tuple[Path, Path, Path]:
    driver = tmp_path / "driver.sh"
    variant = tmp_path / "variant.sh"
    runner = tmp_path / "runner.sh"
    _write_executable(
        driver,
        """#!/usr/bin/env bash
bash "$VARIANT_SCRIPT" "$ARM_VALUE" "$KIND_VALUE" "$SUBSET_VALUE"
rc=$?
printf 'driver_done\\n' >> "$EVENTS"
exit "$rc"
""",
    )
    _write_executable(
        variant,
        """#!/usr/bin/env bash
printf '%s\\n' "$$" > "$VARIANT_PID_FILE"
bash "$RUNNER_SCRIPT"
rc=$?
printf 'variant_done\\n' >> "$EVENTS"
exit "$rc"
""",
    )
    _write_executable(
        runner,
        """#!/usr/bin/env bash
printf '%s\\n' "$$" > "$RUNNER_PID_FILE"
sleep 0.25
printf 'runner_done\\n' >> "$EVENTS"
""",
    )
    return driver, variant, runner


def _process_env(
    *,
    driver: Path,
    variant: Path,
    runner: Path,
    tmp_path: Path,
) -> dict[str, str]:
    return {
        **os.environ,
        "VARIANT_SCRIPT": os.fspath(variant),
        "RUNNER_SCRIPT": os.fspath(runner),
        "ARM_VALUE": "tail-test",
        "KIND_VALUE": "tail6_fixed32",
        "SUBSET_VALUE": "exact4.json",
        "EVENTS": os.fspath(tmp_path / "events.log"),
        "VARIANT_PID_FILE": os.fspath(tmp_path / "variant.pid"),
        "RUNNER_PID_FILE": os.fspath(tmp_path / "runner.pid"),
    }


def test_profile_lifecycle_contract_is_async_exact_and_recoverable() -> None:
    text = PROFILE.read_text(encoding="utf-8")
    sequence = SEQUENCE.read_text(encoding="utf-8")
    variant = VARIANT.read_text(encoding="utf-8")

    assert 'bash scripts/fr13_b4_campaign_driver.sh \\\n' in text
    assert '>"$RUNROOT/driver.log" 2>&1 &' in text
    assert "DRIVER_PID=$!" in text
    assert "(( ${#argv[@]} == 2 + $# ))" in text
    assert "freeze_exact_control_ancestors" in text
    assert '"$NSYS_SESSION_STATE" == "Collection"' in text
    assert 'test("^profile-[0-9]+$")' not in text
    assert "NSYS_INJECTED_SESSION_ID" in text
    assert "NSYS_EXPECTED_SESSION_NAME" in text
    assert "PROFILE_CONTAINER_CIDFILE" in text
    assert "capture_nsys_session_baseline" not in text
    assert (
        'docker exec "$PROFILE_CONTAINER_ID" \\\n'
        '      "$NSYS_CONTAINER_TIMEOUT_BIN" --signal=TERM'
        in text
    )
    assert '"$NSYS_CONTAINER_BIN" sessions list --output-format=json' in text
    assert '"$NSYS_CONTAINER_BIN" stop --session="$NSYS_SESSION_ID"' in text
    assert f"NSYS_CONTAINER_BIN={CONTAINER_NSYS_BIN}" in text
    assert f"NSYS_CONTAINER_TIMEOUT_BIN={CONTAINER_TIMEOUT_BIN}" in text
    assert f"NSYS_CONTAINER_BASH_BIN={CONTAINER_BASH_BIN}" in text
    assert "PROFILE_CONTAINER_STARTED_AT" in text
    assert "PROFILE_CONTAINER_INIT_PID" in text
    assert "PROFILE_CONTAINER_RESTART_COUNT" in text
    assert "NSYS_STOPPED_START_TICKS" in text
    assert "thaw_exact_control_ancestors" in text
    assert "--type=info" in text
    assert "--expected-report-identity" in text
    assert "--expected-report-sha256" in text
    assert "--kill-after=" in text
    assert "LUMO_NSYS_REPORT_STABLE_POLLS" in text
    assert 'docker rename "$PROFILE_CONTAINER_ID" "$PRESERVED_CONTAINER"' in text
    assert 'docker rm -f "$PROFILE_CONTAINER_ID"' in text
    assert "skipping container removal; recovery state is intentionally preserved" in text
    assert "pkill" not in text
    assert "docker ps -q --filter" not in text
    assert '_fixed32_container_identity_matches "$CONTAINER_RUNTIME_REF"' in variant
    assert '--fixed32-container "$CONTAINER_RUNTIME_REF"' in variant
    assert (
        '"$CONTAINER_RUNTIME_REF" "$FIXED32_PRODUCER_PID" "$FIXED32_MODE"'
        in variant
    )
    assert "PROCESS_IDENTITY_TMP=" in variant
    assert 'mv -T "$PROCESS_IDENTITY_TMP"' in variant
    attribution_branch = sequence.index(
        'if [[ "${FR13_FIXED32_ATTRIBUTION_ONLY:-0}" == "1" ]]'
    )
    floor_order_branch = sequence.index('case "${FR13_FLOOR_ORDER:-TH}"')
    assert attribution_branch < floor_order_branch


@pytest.mark.parametrize(
    (
        "session_state",
        "session_present",
        "post_collection",
        "terminal_ok",
        "control_frozen",
        "expected_eligible",
    ),
    (
        ("Waiting", "1", "0", "0", "1", "0"),
        ("", "0", "0", "0", "1", "0"),
        ("", "0", "1", "0", "1", "0"),
        ("ContainerExited", "0", "1", "1", "0", "0"),
        ("ContainerExited", "0", "1", "1", "1", "1"),
    ),
)
def test_report_stability_requires_proven_guarded_terminal_transition(
    session_state: str,
    session_present: str,
    post_collection: str,
    terminal_ok: str,
    control_frozen: str,
    expected_eligible: str,
) -> None:
    script = r"""
source "$1"
NSYS_SESSION_QUERY_OK=1
NSYS_SESSION_ID=$EXPECTED_SESSION_ID
NSYS_SESSION_STATE=$SESSION_STATE
NSYS_SESSION_PRESENT=$SESSION_PRESENT
NSYS_POST_COLLECTION_OBSERVED=$POST_COLLECTION
NSYS_CONTAINER_TERMINAL_OK=$TERMINAL_OK
if report_stability_is_eligible "$CONTROL_FROZEN"; then
  actual=1
else
  actual=0
fi
[[ "$actual" == "$EXPECTED_ELIGIBLE" ]]
"""
    completed = subprocess.run(
        ["bash", "-c", script, "--", os.fspath(PROFILE)],
        cwd=REPO,
        env={
            **os.environ,
            "EXPECTED_SESSION_ID": SESSION_ID,
            "SESSION_STATE": session_state,
            "SESSION_PRESENT": session_present,
            "POST_COLLECTION": post_collection,
            "TERMINAL_OK": terminal_ok,
            "CONTROL_FROZEN": control_frozen,
            "EXPECTED_ELIGIBLE": expected_eligible,
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    (
        "boot_timeout_s",
        "session_timeout_s",
        "collection_timeout_s",
        "expected_error",
    ),
    (
        ("1200", "1500", "1500", None),
        (
            "1199",
            "1500",
            "1500",
            "EngineCore identity timeout must cover the server health timeout",
        ),
        (
            "1200",
            "1499",
            "1500",
            "session discovery timeout must cover delay plus capture duration",
        ),
        (
            "1200",
            "1500",
            "1499",
            "Collection-entry timeout must cover delay plus capture duration",
        ),
    ),
)
def test_profile_timeouts_cover_canonical_delayed_collection(
    boot_timeout_s: str,
    session_timeout_s: str,
    collection_timeout_s: str,
    expected_error: str | None,
) -> None:
    script = r"""
source "$1"
LUMO_NSYS_DELAY_S=1200
LUMO_NSYS_DURATION_S=300
HEALTH_TIMEOUT_S=1200
LUMO_NSYS_BOOT_TIMEOUT_S=$2
LUMO_NSYS_SESSION_TIMEOUT_S=$3
LUMO_NSYS_COLLECTION_TIMEOUT_S=$4
validate_nsys_delayed_collection_timeouts
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            boot_timeout_s,
            session_timeout_s,
            collection_timeout_s,
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    if expected_error is None:
        assert completed.returncode == 0, completed.stderr
    else:
        assert completed.returncode == 1
        assert expected_error in completed.stderr


@pytest.mark.parametrize(
    ("task_wall_s", "duration_s", "expected_error"),
    (
        ("900", "300", None),
        ("300", "300", None),
        ("299", "300", "must cover the Nsight capture duration"),
        ("0", "300", "must be a strict positive integer"),
        ("-1", "300", "must be a strict positive integer"),
        ("invalid", "300", "must be a strict positive integer"),
    ),
)
def test_profile_task_wall_is_positive_and_covers_capture(
    task_wall_s: str,
    duration_s: str,
    expected_error: str | None,
) -> None:
    script = r"""
source "$1"
LUMO_NSYS_SWE_AGENT_WALL_S=$2
LUMO_NSYS_DURATION_S=$3
validate_nsys_attribution_task_wall
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            task_wall_s,
            duration_s,
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    if expected_error is None:
        assert completed.returncode == 0, completed.stderr
    else:
        assert completed.returncode == 1
        assert expected_error in completed.stderr


@pytest.mark.parametrize("cp_mode", ("success", "cp_failure", "identity_drift"))
def test_terminal_engine_ledger_snapshot_is_exact_fresh_and_host_owned(
    tmp_path: Path,
    cp_mode: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    docker_calls = tmp_path / "docker.calls"
    lifecycle_log = tmp_path / "lifecycle.log"
    engine_ledger = tmp_path / "engine-ledger.source.jsonl"
    snapshot = tmp_path / "engine-ledger.snapshot.jsonl"
    identity_drift_file = tmp_path / "identity.drift"
    engine_ledger.write_text('{"kind":"engine_ingress"}\n', encoding="ascii")
    engine_ledger.chmod(0o600)
    _write_engine_snapshot_docker(fake_docker)
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
CONTAINER=$CONTAINER_NAME
PROFILE_CONTAINER_ID=$CONTAINER_ID
PROFILE_CONTAINER_STARTED_AT=$CONTAINER_STARTED_AT
PROFILE_CONTAINER_INIT_PID=$CONTAINER_INIT_PID
PROFILE_CONTAINER_RESTART_COUNT=$CONTAINER_RESTART_COUNT
PROFILE_CONTAINER_RUNNING=0
PROFILE_CONTAINER_STATUS=exited
PROFILE_CONTAINER_EXIT_CODE=0
NSYS_CONTAINER_TERMINAL_OK=1
NSYS_PROVEN_REPORT_IDENTITY=1:2:1:3:4:5
NSYS_PROVEN_REPORT_SHA256=$(printf 'a%.0s' {1..64})
if snapshot_terminal_engine_ledger "$3"; then
  snapshot_rc=0
else
  snapshot_rc=$?
fi
printf '%s\n' "$NSYS_LIFECYCLE_ERROR" > "$4"
if [[ "$CP_MODE" == "success" ]]; then
  [[ "$snapshot_rc" == 0 ]]
  [[ "$ENGINE_LEDGER_SNAPSHOT" == "$3" ]]
else
  [[ "$snapshot_rc" == 1 ]]
fi
"""
    error_file = tmp_path / "snapshot.error"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(lifecycle_log),
            os.fspath(snapshot),
            os.fspath(error_file),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "CONTAINER_ID": CONTAINER_ID,
            "CONTAINER_NAME": "profile-test-container",
            "CONTAINER_STARTED_AT": CONTAINER_STARTED_AT,
            "CONTAINER_INIT_PID": CONTAINER_INIT_PID,
            "CONTAINER_RESTART_COUNT": CONTAINER_RESTART_COUNT,
            "DOCKER_CALLS": os.fspath(docker_calls),
            "ENGINE_LEDGER_SOURCE": os.fspath(engine_ledger),
            "IDENTITY_DRIFT_FILE": os.fspath(identity_drift_file),
            "CP_MODE": cp_mode,
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    calls = docker_calls.read_text(encoding="utf-8").splitlines()
    assert calls[0] == _incarnation_attestation()
    assert calls[1].startswith(
        f"cp {CONTAINER_ID}:/logs/fr13_fixed32_engine_ingress.jsonl "
    )
    assert not list(tmp_path.glob("engine-ledger.snapshot.jsonl.tmp.*"))
    if cp_mode == "success":
        assert calls[2] == _incarnation_attestation()
        assert snapshot.read_bytes() == engine_ledger.read_bytes()
        assert snapshot.is_file() and not snapshot.is_symlink()
        assert snapshot.stat().st_uid == os.getuid()
        assert os.access(snapshot, os.R_OK)
        assert error_file.read_text(encoding="utf-8") == "\n"
    else:
        assert not snapshot.exists()
        expected_error = {
            "cp_failure": "docker cp failed",
            "identity_drift": "identity drifted during",
        }[cp_mode]
        assert expected_error in error_file.read_text(encoding="utf-8")


def test_delayed_session_deadlines_start_after_engine_identity(
    tmp_path: Path,
) -> None:
    lifecycle_log = tmp_path / "lifecycle.log"
    readability = tmp_path / "readability.json"
    boundary = tmp_path / "boundary"
    report = tmp_path / "capture.nsys-rep"
    boundary.touch()
    time.sleep(0.01)
    report.write_bytes(b"report")
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
NSYS_SESSION_ID=''
NSYS_SESSION_STATE=''
NSYS_SESSION_PRESENT=0
NSYS_SESSION_QUERY_OK=0
NSYS_INJECTED_SESSION_ID=''
NSYS_POST_COLLECTION_OBSERVED=0
NSYS_PROVEN_REPORT_IDENTITY=''
NSYS_PROVEN_REPORT_SHA256=''
LUMO_NSYS_BOOT_TIMEOUT_S=10
LUMO_NSYS_SESSION_TIMEOUT_S=1
LUMO_NSYS_COLLECTION_TIMEOUT_S=1
LUMO_NSYS_COLLECTION_MAX_S=10
LUMO_NSYS_REPORT_TIMEOUT_S=10
LUMO_NSYS_HASH_TIMEOUT_S=2
LUMO_NSYS_HASH_KILL_AFTER_S=1
LUMO_NSYS_REPORT_STABLE_POLLS=1
LUMO_NSYS_POLL_S=1
ARM=tail6_fixture
SUBSET=subset_fixture
NSYS_EXPECTED_DRIVER_SCRIPT=driver_fixture
NSYS_EXPECTED_VARIANT_SCRIPT=variant_fixture
NSYS_EXPECTED_VARIANT_KIND=tail6_fixed32
refresh_count=0

refresh_run_nsys_session() {
  (( refresh_count += 1 ))
  case "$refresh_count" in
    1)
      SECONDS=5
      NSYS_SESSION_QUERY_OK=0
      ;;
    2)
      NSYS_INJECTED_SESSION_ID=42
      NSYS_IDENTITY_ATTESTED=1
      NSYS_SESSION_ID=42
      NSYS_SESSION_STATE=Collection
      NSYS_SESSION_PRESENT=1
      NSYS_SESSION_QUERY_OK=1
      ;;
    *)
      NSYS_SESSION_STATE=ContainerExited
      NSYS_SESSION_PRESENT=0
      NSYS_SESSION_QUERY_OK=1
      NSYS_POST_COLLECTION_OBSERVED=1
      NSYS_CONTAINER_TERMINAL_OK=1
      ;;
  esac
  return 0
}
freeze_exact_control_ancestors() { return 0; }
_process_identity_is_live() { return 0; }
verify_report_readable() {
  printf '{"readable":true}\n' > "$2"
}
fail_nsys_lifecycle() {
  NSYS_LIFECYCLE_ERROR=$1
  return 1
}
sleep() { :; }

SECONDS=0
wait_for_fresh_stable_report 123 456 "$3" "$4" "$5"
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(lifecycle_log),
            os.fspath(report),
            os.fspath(readability),
            os.fspath(boundary),
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    assert "starting delayed-session deadlines" in lifecycle_log.read_text(
        encoding="utf-8"
    )


def test_attribution_sequence_is_tail_only_and_acceptance_orders_are_unchanged() -> None:
    script = r"""
run_variant() {
  printf '%s %s %s %s\n' "$1" "$2" "$3" "$4"
}
export BSIZE=1
export CONC=1
export TAG=test
export FR13_FIXED32_ATTRIBUTION_ONLY=$2
export FR13_FLOOR_ORDER=$3
export LUMO_NSYS_WRAP_VLLM=$4
source "$1"
"""
    cases = (
        (
            ("1", "HT", "1"),
            ["tail6_fixed32_test tail6_fixed32 31 1"],
        ),
        (
            ("0", "TH", "0"),
            [
                "tail6_fixed32_test tail6_fixed32 31 1",
                "hydra27_fixed32_test hydra27_fixed32 31 1",
            ],
        ),
        (
            ("0", "HT", "0"),
            [
                "hydra27_fixed32_test hydra27_fixed32 31 1",
                "tail6_fixed32_test tail6_fixed32 31 1",
            ],
        ),
    )
    for arguments, expected in cases:
        completed = subprocess.run(
            ["bash", "-c", script, "--", os.fspath(SEQUENCE), *arguments],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.splitlines() == expected


def test_freeze_stops_only_exact_shell_ancestors_and_runner_continues(
    tmp_path: Path,
) -> None:
    driver, variant, runner = _fake_control_processes(tmp_path)
    lifecycle_log = tmp_path / "lifecycle.log"
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
bash "$3" &
driver_pid=$!
driver_start=$(_process_start_ticks "$driver_pid")
for (( attempt=0; attempt < 100; attempt++ )); do
  [[ -s "$4" && -s "$5" ]] && break
  sleep 0.01
done
freeze_exact_control_ancestors \
  "$driver_pid" "$driver_start" "$3" "$6" \
  "$ARM_VALUE" "$KIND_VALUE" "$SUBSET_VALUE"
variant_pid=$(<"$4")
runner_pid=$(<"$5")
[[ "$(_process_state "$driver_pid")" == "T" ]]
[[ "$(_process_state "$variant_pid")" == "T" ]]
[[ "$(_process_state "$runner_pid")" != "T" ]]
for (( attempt=0; attempt < 100; attempt++ )); do
  grep -q '^runner_done$' "$EVENTS" 2>/dev/null && break
  sleep 0.01
done
grep -q '^runner_done$' "$EVENTS"
! grep -q '^variant_done$' "$EVENTS"
[[ ${#NSYS_STOPPED_PIDS[@]} == 2 ]]
thaw_exact_control_ancestors
wait "$driver_pid"
grep -q '^variant_done$' "$EVENTS"
grep -q '^driver_done$' "$EVENTS"
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(lifecycle_log),
            os.fspath(driver),
            os.fspath(tmp_path / "variant.pid"),
            os.fspath(tmp_path / "runner.pid"),
            os.fspath(variant),
        ],
        cwd=tmp_path,
        env=_process_env(
            driver=driver,
            variant=variant,
            runner=runner,
            tmp_path=tmp_path,
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    lifecycle = lifecycle_log.read_text(encoding="utf-8")
    assert "stopped campaign driver" in lifecycle
    assert "stopped variant shell" in lifecycle
    assert "descendants remain runnable" in lifecycle
    assert lifecycle.count("thawed exact pid=") == 2


def test_wait_requires_fresh_stable_report_and_pinned_nsys_readability(
    tmp_path: Path,
) -> None:
    driver, variant, runner = _fake_control_processes(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nsys = tmp_path / "nsys"
    fake_docker = fake_bin / "docker"
    session_generation = tmp_path / "session.generation"
    session_done = tmp_path / "session.done"
    report = tmp_path / "fresh.nsys-rep"
    readability = tmp_path / "readability.info"
    boundary = tmp_path / "run.boundary"
    cidfile = tmp_path / "container.cid"
    process_identity = tmp_path / "process_identity.json"
    container_state = tmp_path / "container.state"
    nsys_calls = tmp_path / "nsys.calls"
    docker_calls = tmp_path / "docker.calls"
    lifecycle_log = tmp_path / "lifecycle.log"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    cidfile.write_text(f"{CONTAINER_ID}\n", encoding="ascii")
    _write_process_identity(process_identity)
    container_state.write_text(
        (
            f"running true {CONTAINER_STARTED_AT} {CONTAINER_INIT_PID} "
            f"{CONTAINER_RESTART_COUNT} 0\n"
        ),
        encoding="ascii",
    )
    _write_identity_docker(fake_docker)
    _write_executable(
        fake_nsys,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "sessions list" ]]; then
  [[ "${FAKE_NSYS_CONTAINER_CONTEXT:-0}" == "1" ]] || exit 18
  if [[ -e "$SESSION_DONE" ]]; then
    phase=done
    printf '[{"id":"7777","state":"Collection","name":"unrelated-session","accessible":true}]\\n'
  elif [[ -e "$SESSION_GENERATION" ]]; then
    phase=generation
    printf '[{"id":"7777","state":"Collection","name":"unrelated-session","accessible":true},{"id":"10206","state":"Generation","name":"%s","accessible":true}]\\n' "$EXPECTED_SESSION_NAME"
  else
    phase=collection
    printf '[{"id":"7777","state":"Collection","name":"unrelated-session","accessible":true},{"id":"10206","state":"Collection","name":"%s","accessible":true}]\\n' "$EXPECTED_SESSION_NAME"
  fi
  printf 'sessions list phase=%s\\n' "$phase" >> "$NSYS_CALLS"
  exit 0
fi
printf '%s\\n' "$*" >> "$NSYS_CALLS"
if [[ "$1" == "export" ]]; then
  [[ "${FAKE_NSYS_CONTAINER_CONTEXT:-0}" == "0" ]] || exit 19
  if [[ -L "$REPORT_PATH" ]]; then
    printf 'export_symlink\\n' >> "$NSYS_CALLS"
    exit 20
  fi
  output=""
  for arg in "$@"; do
    case "$arg" in
      --output=*) output=${arg#--output=} ;;
    esac
  done
  grep -q 'complete' "$REPORT_PATH" || exit 9
  printf '{"readable":true}\\n' > "$output"
  exit 0
fi
if [[ "$1" == "stop" ]]; then
  [[ "${FAKE_NSYS_CONTAINER_CONTEXT:-0}" == "1" ]] || exit 18
  : > "$SESSION_GENERATION"
  exit 0
fi
exit 8
""",
    )
    _write_executable(
        runner,
        """#!/usr/bin/env bash
printf '%s\\n' "$$" > "$RUNNER_PID_FILE"
sleep 0.12
: > "$SESSION_GENERATION"
printf 'complete\\n' > "$REPORT_PATH"
# Keep the report byte-stable through more than three profiler polls while the
# exact session remains in Generation. Those polls must not count as stable.
sleep 0.25
: > "$SESSION_DONE"
printf 'complete\\n' > "$REPORT_PATH.target"
rm -f "$REPORT_PATH"
ln -s "$REPORT_PATH.target" "$REPORT_PATH"
printf 'exited false %s 0 %s 0\\n' \
  "$CONTAINER_STARTED_AT" "$CONTAINER_RESTART_COUNT" \
  > "$CONTAINER_STATE_FILE.tmp"
mv -T "$CONTAINER_STATE_FILE.tmp" "$CONTAINER_STATE_FILE"
sleep 0.25
printf 'complete\\n' > "$REPORT_PATH.tmp"
mv -T "$REPORT_PATH.tmp" "$REPORT_PATH"
printf 'runner_done\\n' >> "$EVENTS"
""",
    )
    env = {
        **_process_env(
            driver=driver,
            variant=variant,
            runner=runner,
            tmp_path=tmp_path,
        ),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CONTAINER_ID": CONTAINER_ID,
        "CONTAINER_NAME": "profile-test-container",
        "DOCKER_CALLS": os.fspath(docker_calls),
        "CONTAINER_NSYS": os.fspath(fake_nsys),
        "CONTAINER_STATE_FILE": os.fspath(container_state),
        "CONTAINER_STARTED_AT": CONTAINER_STARTED_AT,
        "CONTAINER_RESTART_COUNT": CONTAINER_RESTART_COUNT,
        "EXPECTED_SESSION_NAME": SESSION_NAME,
        "NSYS_CALLS": os.fspath(nsys_calls),
        "SESSION_GENERATION": os.fspath(session_generation),
        "SESSION_DONE": os.fspath(session_done),
        "REPORT_PATH": os.fspath(report),
    }
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
LUMO_NSYS_BIN=$3
JQ_BIN=$(command -v jq)
NSYS_SESSION_ID=''
NSYS_SESSION_NAME=''
NSYS_SESSION_STATE=''
NSYS_SESSION_PRESENT=0
NSYS_SESSION_QUERY_OK=0
NSYS_INJECTED_SESSION_ID=''
NSYS_EXPECTED_SESSION_NAME=$EXPECTED_SESSION_NAME
PROFILE_CONTAINER_ID=''
PROFILE_CONTAINER_CIDFILE=$9
PROCESS_IDENTITY=${10}
NSIGHT_ATTESTATION="$PROCESS_IDENTITY.attestation.json"
RUN_BOUNDARY=$7
CONTAINER=$CONTAINER_NAME
NSYS_EXPECTED_DRIVER_SCRIPT=$4
NSYS_EXPECTED_VARIANT_SCRIPT=$8
NSYS_EXPECTED_VARIANT_KIND=$KIND_VALUE
LUMO_NSYS_SESSION_TIMEOUT_S=3
LUMO_NSYS_COLLECTION_TIMEOUT_S=3
LUMO_NSYS_COLLECTION_MAX_S=3
LUMO_NSYS_REPORT_TIMEOUT_S=3
LUMO_NSYS_STOP_TIMEOUT_S=1
LUMO_NSYS_STOP_KILL_AFTER_S=1
LUMO_NSYS_REPORT_STABLE_POLLS=3
LUMO_NSYS_POLL_S=0.05
bash "$4" &
DRIVER_PID=$!
DRIVER_START_TICKS=$(_process_start_ticks "$DRIVER_PID")
ARM=$ARM_VALUE
SUBSET=$SUBSET_VALUE
if ! wait_for_fresh_stable_report \
    "$DRIVER_PID" "$DRIVER_START_TICKS" "$5" "$6" "$7"; then
  printf 'lifecycle failure: %s\n' "$NSYS_LIFECYCLE_ERROR" >&2
  thaw_exact_control_ancestors
  wait "$DRIVER_PID" || true
  exit 30
fi
[[ "$NSYS_PROVEN_REPORT_IDENTITY" == \
  "$(stat -c '%d:%i:%h:%s:%Y:%Z' "$5")" ]] || exit 35
[[ "$NSYS_PROVEN_REPORT_SHA256" == \
  "$(sha256sum "$5" | awk '{print $1}')" ]] || exit 36
grep -q '^runner_done$' "$EVENTS" || exit 31
! grep -q '^variant_done$' "$EVENTS" || exit 32
[[ ${#NSYS_STOPPED_PIDS[@]} == 2 ]] || exit 33
thaw_exact_control_ancestors
wait "$DRIVER_PID" || exit 34
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(lifecycle_log),
            os.fspath(fake_nsys),
            os.fspath(driver),
            os.fspath(report),
            os.fspath(readability),
            os.fspath(boundary),
            os.fspath(variant),
            os.fspath(cidfile),
            os.fspath(process_identity),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert readability.read_text(encoding="utf-8") == '{"readable":true}\n'
    calls = nsys_calls.read_text(encoding="utf-8").splitlines()
    export_index = next(
        index for index, call in enumerate(calls) if call.startswith("export ")
    )
    assert "sessions list phase=generation" in calls[:export_index]
    if "sessions list phase=done" in calls:
        assert calls.index("sessions list phase=done") < export_index
    assert "export_symlink" not in calls
    lifecycle = lifecycle_log.read_text(encoding="utf-8")
    assert f"attested run container id={CONTAINER_ID}" in lifecycle
    assert (
        f"attested Nsight-injected EngineCore identity "
        f"session_name={SESSION_NAME} injected_session_id={SESSION_ID}"
    ) in lifecycle
    assert f"bound exact run Nsight session id={SESSION_ID}" in lifecycle
    assert "bound exact run Nsight session id=7777" not in lifecycle
    assert "Nsight collection window opened with teardown control frozen" in lifecycle
    assert "Nsight report generation began" in lifecycle
    assert "accepted exact exited0 container incarnation" in lifecycle
    assert "stable, readable, and cryptographically latched" in lifecycle
    assert lifecycle.index("accepted exact exited0 container incarnation") < (
        lifecycle.index("stable, readable, and cryptographically latched")
    )
    docker_execs = [
        call
        for call in docker_calls.read_text(encoding="utf-8").splitlines()
        if call.startswith("exec ") and f" {CONTAINER_NSYS_BIN} " in call
    ]
    assert docker_execs
    assert all(
        call.startswith(f"exec {CONTAINER_ID} {CONTAINER_TIMEOUT_BIN} ")
        and f" {CONTAINER_NSYS_BIN} " in call
        for call in docker_execs
    )


@pytest.mark.parametrize("terminal_at_inspect", (3, 4))
def test_generation_accepts_terminal_transition_around_engine_liveness(
    tmp_path: Path,
    terminal_at_inspect: int,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    inspect_count = tmp_path / "inspect.count"
    docker_calls = tmp_path / "docker.calls"
    process_identity = tmp_path / "process_identity.json"
    boundary = tmp_path / "run.boundary"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    _write_process_identity(process_identity)
    _write_executable(
        fake_docker,
        f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
if [[ "$1" == "inspect" && "$2" == "--format" ]]; then
  count=$(cat "$INSPECT_COUNT" 2>/dev/null || printf '0')
  count=$((count + 1))
  printf '%s\\n' "$count" > "$INSPECT_COUNT"
  if (( count >= TERMINAL_AT_INSPECT )); then
    printf '%s /%s exited false %s 0 %s 0\\n' \
      "$CONTAINER_ID" "$CONTAINER_NAME" "$CONTAINER_STARTED_AT" \
      "$CONTAINER_RESTART_COUNT"
  else
    printf '%s /%s running true %s %s %s 0\\n' \
      "$CONTAINER_ID" "$CONTAINER_NAME" "$CONTAINER_STARTED_AT" \
      "$CONTAINER_INIT_PID" "$CONTAINER_RESTART_COUNT"
  fi
  exit 0
fi
if [[ "$1" == "exec" && "$7" == "{CONTAINER_BASH_BIN}" ]]; then
  printf '9001\\n'
  exit 0
fi
exit 91
""",
    )
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
JQ_BIN=$(command -v jq)
RUN_BOUNDARY=$3
PROCESS_IDENTITY=$4
NSIGHT_ATTESTATION="$PROCESS_IDENTITY.attestation.json"
PROCESS_IDENTITY_STAT=$(stat -c '%d:%i:%h:%s:%Y:%Z' "$PROCESS_IDENTITY")
CONTAINER=$CONTAINER_NAME
PROFILE_CONTAINER_ID=$CONTAINER_ID
PROFILE_CONTAINER_STARTED_AT=$CONTAINER_STARTED_AT
PROFILE_CONTAINER_INIT_PID=$CONTAINER_INIT_PID
PROFILE_CONTAINER_RESTART_COUNT=$CONTAINER_RESTART_COUNT
PROFILE_CONTAINER_RUNNING=1
PROFILE_CONTAINER_STATUS=running
PROFILE_CONTAINER_EXIT_CODE=0
ENGINE_CORE_PID=321
ENGINE_CORE_START_TICKS=9001
ENGINE_CORE_LIVENESS_OK=1
NSYS_INJECTED_SESSION_ID=$EXPECTED_SESSION_ID
NSYS_IDENTITY_ATTESTED=1
NSYS_SESSION_ID=$EXPECTED_SESSION_ID
NSYS_EXPECTED_SESSION_NAME=$EXPECTED_SESSION_NAME
NSYS_SESSION_NAME=$EXPECTED_SESSION_NAME
NSYS_SESSION_STATE=Generation
NSYS_COLLECTION_OBSERVED=1
NSYS_POST_COLLECTION_OBSERVED=1
refresh_run_nsys_session
[[ $? == 0 ]]
[[ "$PROFILE_CONTAINER_RUNNING" == 0 ]]
[[ "$NSYS_CONTAINER_TERMINAL_OK" == 1 ]]
[[ "$NSYS_SESSION_STATE" == "ContainerExited" ]]
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(tmp_path / "lifecycle.log"),
            os.fspath(boundary),
            os.fspath(process_identity),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "CONTAINER_ID": CONTAINER_ID,
            "CONTAINER_NAME": "profile-test-container",
            "CONTAINER_STARTED_AT": CONTAINER_STARTED_AT,
            "CONTAINER_INIT_PID": CONTAINER_INIT_PID,
            "CONTAINER_RESTART_COUNT": CONTAINER_RESTART_COUNT,
            "DOCKER_CALLS": os.fspath(docker_calls),
            "INSPECT_COUNT": os.fspath(inspect_count),
            "TERMINAL_AT_INSPECT": str(terminal_at_inspect),
            "EXPECTED_SESSION_ID": SESSION_ID,
            "EXPECTED_SESSION_NAME": SESSION_NAME,
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    calls = docker_calls.read_text(encoding="utf-8")
    assert f" {CONTAINER_NSYS_BIN} sessions list " not in calls


def test_session_binding_ignores_unrelated_name_and_stops_only_exact_in_container(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_nsys = tmp_path / "nsys"
    docker_calls = tmp_path / "docker.calls"
    nsys_calls = tmp_path / "nsys.calls"
    snapshot = tmp_path / "sessions.json"
    boundary = tmp_path / "run.boundary"
    cidfile = tmp_path / "container.cid"
    process_identity = tmp_path / "process_identity.json"
    lifecycle_log = tmp_path / "lifecycle.log"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    cidfile.write_text(f"{CONTAINER_ID}\n", encoding="ascii")
    _write_process_identity(process_identity)
    _write_identity_docker(fake_docker)
    _write_executable(
        fake_nsys,
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$NSYS_CALLS"
if [[ "$1 $2" == "sessions list" ]]; then
  [[ "${FAKE_NSYS_CONTAINER_CONTEXT:-0}" == "1" ]] || exit 18
  cat "$SESSION_SNAPSHOT"
  exit 0
fi
if [[ "$1" == "stop" ]]; then
  [[ "${FAKE_NSYS_CONTAINER_CONTEXT:-0}" == "1" ]] || exit 18
  exit 0
fi
exit 9
""",
    )
    unrelated = [
        {
            "id": "7777",
            "state": "Collection",
            "name": "unrelated-session",
            "accessible": True,
        }
    ]
    exact = [
        *unrelated,
        {
            "id": SESSION_ID,
            "state": "Collection",
            "name": SESSION_NAME,
            "accessible": True,
        },
    ]
    snapshot.write_text(json.dumps(unrelated), encoding="ascii")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CONTAINER_ID": CONTAINER_ID,
        "CONTAINER_NAME": "profile-test-container",
        "DOCKER_CALLS": os.fspath(docker_calls),
        "CONTAINER_NSYS": os.fspath(fake_nsys),
        "NSYS_CALLS": os.fspath(nsys_calls),
        "SESSION_SNAPSHOT": os.fspath(snapshot),
        "EXACT_SNAPSHOT": json.dumps(exact, separators=(",", ":")),
    }
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
LUMO_NSYS_BIN=$3
JQ_BIN=$(command -v jq)
RUN_BOUNDARY=$4
PROFILE_CONTAINER_CIDFILE=$5
PROCESS_IDENTITY=$6
NSIGHT_ATTESTATION="$PROCESS_IDENTITY.attestation.json"
CONTAINER=$CONTAINER_NAME
NSYS_EXPECTED_SESSION_NAME=$7
LUMO_NSYS_STOP_TIMEOUT_S=1
LUMO_NSYS_STOP_KILL_AFTER_S=1
refresh_run_nsys_session
[[ -z "$NSYS_SESSION_ID" ]]
if stop_exact_nsys_session; then
  exit 20
fi
printf '%s\n' "$EXACT_SNAPSHOT" > "$SESSION_SNAPSHOT"
refresh_run_nsys_session
[[ "$NSYS_SESSION_ID" == "10206" ]]
stop_exact_nsys_session
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(lifecycle_log),
            os.fspath(fake_nsys),
            os.fspath(boundary),
            os.fspath(cidfile),
            os.fspath(process_identity),
            SESSION_NAME,
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    stop_calls = [
        line
        for line in nsys_calls.read_text(encoding="utf-8").splitlines()
        if line.startswith("stop ")
    ]
    assert stop_calls == [f"stop --session={SESSION_ID}"]
    docker_execs = [
        line
        for line in docker_calls.read_text(encoding="utf-8").splitlines()
        if line.startswith("exec ") and f" {CONTAINER_NSYS_BIN} " in line
    ]
    assert docker_execs[-1] == _container_nsys_exec(
        f"stop --session={SESSION_ID}",
        timeout_s="1",
        kill_after_s="1",
    )
    assert all(
        line.startswith(f"exec {CONTAINER_ID} {CONTAINER_TIMEOUT_BIN} ")
        and f" {CONTAINER_NSYS_BIN} " in line
        for line in docker_execs
    )
    lifecycle = lifecycle_log.read_text(encoding="utf-8")
    assert f"bound exact run Nsight session id={SESSION_ID}" in lifecycle
    assert "bound exact run Nsight session id=7777" not in lifecycle


def test_engine_environ_without_session_id_binds_by_run_unique_name(
    tmp_path: Path,
) -> None:
    """The observed runtime: no NSYS_PROFILING_SESSION_ID reaches EngineCore.

    Nsight injects NSYS_PROFILING_SESSION_ID into the `vllm serve` root it
    launches. vLLM's setproctitle("VLLM::EngineCore") then overwrites the front
    of the engine subprocess's argv+environ block, so the variable is absent
    from /proc/<engine>/environ. The run-unique session name must still bind the
    session, and the stop must still target only that exact session.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_nsys = tmp_path / "nsys"
    docker_calls = tmp_path / "docker.calls"
    nsys_calls = tmp_path / "nsys.calls"
    snapshot = tmp_path / "sessions.json"
    boundary = tmp_path / "run.boundary"
    cidfile = tmp_path / "container.cid"
    process_identity = tmp_path / "process_identity.json"
    lifecycle_log = tmp_path / "lifecycle.log"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    cidfile.write_text(f"{CONTAINER_ID}\n", encoding="ascii")
    _write_process_identity(process_identity, None)
    _write_identity_docker(fake_docker)
    _write_executable(
        fake_nsys,
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$NSYS_CALLS"
if [[ "$1 $2" == "sessions list" ]]; then
  [[ "${FAKE_NSYS_CONTAINER_CONTEXT:-0}" == "1" ]] || exit 18
  cat "$SESSION_SNAPSHOT"
  exit 0
fi
if [[ "$1" == "stop" ]]; then
  [[ "${FAKE_NSYS_CONTAINER_CONTEXT:-0}" == "1" ]] || exit 18
  exit 0
fi
exit 9
""",
    )
    unrelated = [
        {
            "id": "7777",
            "state": "Collection",
            "name": "unrelated-session",
            "accessible": True,
        }
    ]
    exact = [
        *unrelated,
        {
            "id": SESSION_ID,
            "state": "Collection",
            "name": SESSION_NAME,
            "accessible": True,
        },
    ]
    snapshot.write_text(json.dumps(unrelated), encoding="ascii")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CONTAINER_ID": CONTAINER_ID,
        "CONTAINER_NAME": "profile-test-container",
        "DOCKER_CALLS": os.fspath(docker_calls),
        "CONTAINER_NSYS": os.fspath(fake_nsys),
        "NSYS_CALLS": os.fspath(nsys_calls),
        "SESSION_SNAPSHOT": os.fspath(snapshot),
        "EXACT_SNAPSHOT": json.dumps(exact, separators=(",", ":")),
    }
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
LUMO_NSYS_BIN=$3
JQ_BIN=$(command -v jq)
RUN_BOUNDARY=$4
PROFILE_CONTAINER_CIDFILE=$5
PROCESS_IDENTITY=$6
NSIGHT_ATTESTATION="$PROCESS_IDENTITY.attestation.json"
CONTAINER=$CONTAINER_NAME
NSYS_EXPECTED_SESSION_NAME=$7
LUMO_NSYS_STOP_TIMEOUT_S=1
LUMO_NSYS_STOP_KILL_AFTER_S=1
refresh_run_nsys_session || exit 21
# The identity is attested even though no session ID was injected.
(( NSYS_IDENTITY_ATTESTED == 1 )) || exit 22
[[ -z "$NSYS_INJECTED_SESSION_ID" ]] || exit 23
# The unrelated session must not bind, and stop must refuse while unbound.
[[ -z "$NSYS_SESSION_ID" ]] || exit 24
if stop_exact_nsys_session; then
  exit 20
fi
printf '%s\n' "$EXACT_SNAPSHOT" > "$SESSION_SNAPSHOT"
refresh_run_nsys_session || exit 25
[[ "$NSYS_SESSION_ID" == "10206" ]] || exit 26
[[ "$NSYS_SESSION_NAME" == "$NSYS_EXPECTED_SESSION_NAME" ]] || exit 27
stop_exact_nsys_session
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(lifecycle_log),
            os.fspath(fake_nsys),
            os.fspath(boundary),
            os.fspath(cidfile),
            os.fspath(process_identity),
            SESSION_NAME,
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    stop_calls = [
        line
        for line in nsys_calls.read_text(encoding="utf-8").splitlines()
        if line.startswith("stop ")
    ]
    assert stop_calls == [f"stop --session={SESSION_ID}"]
    lifecycle = lifecycle_log.read_text(encoding="utf-8")
    assert (
        f"bound exact run Nsight session id={SESSION_ID} name={SESSION_NAME} "
        "via run-unique session name"
    ) in lifecycle
    assert "bound exact run Nsight session id=7777" not in lifecycle
    assert "injected_session_id=<resolved by run-unique name>" in lifecycle


@pytest.mark.parametrize(
    ("identity_kwargs", "expected_evidence", "expected_message"),
    (
        pytest.param(
            {"injection": False},
            "engine_core.NVTX_INJECTION64_PATH=[]",
            # FR14: an engine that is not under Nsight injection is now caught by
            # the ENGINE-PUBLISHED attestation rather than by reading the
            # engine's (setproctitle-destroyed) /proc environ, so the refusal
            # names the binding that failed.
            "does not bind this process to this run's session",
            id="engine-not-under-nsight-injection",
        ),
        pytest.param(
            {"session_name": "fr13-fixed32-20260730T120000Z-p9999"},
            "pid1.LUMO_NSYS_SESSION_NAME=[\"fr13-fixed32-20260730T120000Z-p9999\"]",
            # pid1's environ is NOT setproctitle'd, so the frontend pin is still
            # attested from the process identity and keeps its original refusal.
            "does not attest the run-unique Nsight session",
            id="frontend-pinned-a-different-run",
        ),
    ),
)
def test_identity_without_nsight_evidence_fails_closed_and_self_diagnoses(
    tmp_path: Path,
    identity_kwargs: dict[str, object],
    expected_evidence: str,
    expected_message: str,
) -> None:
    """A rejected attestation must name the evidence it actually found."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    docker_calls = tmp_path / "docker.calls"
    boundary = tmp_path / "run.boundary"
    cidfile = tmp_path / "container.cid"
    process_identity = tmp_path / "process_identity.json"
    lifecycle_log = tmp_path / "lifecycle.log"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    cidfile.write_text(f"{CONTAINER_ID}\n", encoding="ascii")
    _write_process_identity(process_identity, None, **identity_kwargs)
    _write_identity_docker(fake_docker)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CONTAINER_ID": CONTAINER_ID,
        "CONTAINER_NAME": "profile-test-container",
        "DOCKER_CALLS": os.fspath(docker_calls),
        "CONTAINER_NSYS": "/nonexistent-nsys",
    }
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
JQ_BIN=$(command -v jq)
RUN_BOUNDARY=$3
PROFILE_CONTAINER_CIDFILE=$4
PROCESS_IDENTITY=$5
NSIGHT_ATTESTATION="$PROCESS_IDENTITY.attestation.json"
CONTAINER=$CONTAINER_NAME
NSYS_EXPECTED_SESSION_NAME=$6
refresh_run_identity_evidence
(( $? == 2 )) || exit 21
(( NSYS_IDENTITY_ATTESTED == 0 )) || exit 22
printf '%s\n' "$NSYS_LIFECYCLE_ERROR"
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(lifecycle_log),
            os.fspath(boundary),
            os.fspath(cidfile),
            os.fspath(process_identity),
            SESSION_NAME,
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert expected_message in completed.stdout, completed.stdout
    lifecycle = lifecycle_log.read_text(encoding="utf-8")
    assert "process identity Nsight evidence:" in lifecycle
    assert expected_evidence in lifecycle


@pytest.mark.parametrize(
    ("session_snapshot", "expected_error"),
    (
        (
            [
                {
                    "id": SESSION_ID,
                    "state": "Collection",
                    "name": "other-run",
                    "accessible": True,
                }
            ],
            "does not match",
        ),
        (
            [
                {
                    "id": "7777",
                    "state": "Collection",
                    "name": SESSION_NAME,
                    "accessible": True,
                }
            ],
            "different session ID",
        ),
    ),
)
def test_session_identity_mismatch_fails_closed(
    tmp_path: Path,
    session_snapshot: list[dict[str, object]],
    expected_error: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_nsys = tmp_path / "nsys"
    boundary = tmp_path / "run.boundary"
    cidfile = tmp_path / "container.cid"
    process_identity = tmp_path / "process_identity.json"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    cidfile.write_text(f"{CONTAINER_ID}\n", encoding="ascii")
    _write_process_identity(process_identity)
    _write_identity_docker(fake_docker)
    _write_executable(
        fake_nsys,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "sessions list" ]]; then
  printf '%s\\n' "$SESSION_SNAPSHOT"
  exit 0
fi
exit 9
""",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CONTAINER_ID": CONTAINER_ID,
        "CONTAINER_NAME": "profile-test-container",
        "DOCKER_CALLS": os.fspath(tmp_path / "docker.calls"),
        "CONTAINER_NSYS": os.fspath(fake_nsys),
        "SESSION_SNAPSHOT": json.dumps(session_snapshot, separators=(",", ":")),
        "EXPECTED_ERROR": expected_error,
    }
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
LUMO_NSYS_BIN=$3
JQ_BIN=$(command -v jq)
RUN_BOUNDARY=$4
PROFILE_CONTAINER_CIDFILE=$5
PROCESS_IDENTITY=$6
NSIGHT_ATTESTATION="$PROCESS_IDENTITY.attestation.json"
CONTAINER=$CONTAINER_NAME
NSYS_EXPECTED_SESSION_NAME=$7
refresh_run_nsys_session
[[ $? == 2 ]]
[[ -z "$NSYS_SESSION_ID" ]]
[[ "$NSYS_LIFECYCLE_ERROR" == *"$EXPECTED_ERROR"* ]]
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(tmp_path / "lifecycle.log"),
            os.fspath(fake_nsys),
            os.fspath(boundary),
            os.fspath(cidfile),
            os.fspath(process_identity),
            SESSION_NAME,
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr


def test_session_query_fails_closed_on_post_exec_container_name_drift(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_nsys = tmp_path / "nsys"
    boundary = tmp_path / "run.boundary"
    cidfile = tmp_path / "container.cid"
    process_identity = tmp_path / "process_identity.json"
    container_name_file = tmp_path / "container.name"
    docker_calls = tmp_path / "docker.calls"
    expected_container_name = "profile-test-container"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    cidfile.write_text(f"{CONTAINER_ID}\n", encoding="ascii")
    _write_process_identity(process_identity)
    container_name_file.write_text(f"{expected_container_name}\n", encoding="ascii")
    _write_identity_docker(fake_docker, mutable_name_file=True)
    _write_executable(
        fake_nsys,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "sessions list" ]]; then
  printf 'foreign-container\\n' > "$CONTAINER_NAME_FILE"
  printf '[{"id":"10206","state":"Collection","name":"%s","accessible":true}]\\n' "$EXPECTED_SESSION_NAME"
  exit 0
fi
exit 9
""",
    )
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
JQ_BIN=$(command -v jq)
RUN_BOUNDARY=$3
PROFILE_CONTAINER_CIDFILE=$4
PROCESS_IDENTITY=$5
NSIGHT_ATTESTATION="$PROCESS_IDENTITY.attestation.json"
CONTAINER=$6
NSYS_EXPECTED_SESSION_NAME=$7
refresh_run_nsys_session
[[ $? == 2 ]]
[[ -z "$NSYS_SESSION_ID" ]]
[[ "$NSYS_LIFECYCLE_ERROR" == *"changed during Nsight session query"* ]]
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(tmp_path / "lifecycle.log"),
            os.fspath(boundary),
            os.fspath(cidfile),
            os.fspath(process_identity),
            expected_container_name,
            SESSION_NAME,
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "CONTAINER_ID": CONTAINER_ID,
            "CONTAINER_NAME_FILE": os.fspath(container_name_file),
            "CONTAINER_NSYS": os.fspath(fake_nsys),
            "DOCKER_CALLS": os.fspath(docker_calls),
            "EXPECTED_SESSION_NAME": SESSION_NAME,
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    calls = docker_calls.read_text(encoding="utf-8").splitlines()
    exact_exec = _container_nsys_exec("sessions list --output-format=json")
    running_attestation = _incarnation_attestation()
    exec_index = calls.index(exact_exec)
    assert calls[exec_index - 1 : exec_index + 2] == [
        running_attestation,
        exact_exec,
        running_attestation,
    ]


def test_same_id_container_restart_fails_incarnation_reattestation(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_nsys = tmp_path / "nsys"
    boundary = tmp_path / "run.boundary"
    cidfile = tmp_path / "container.cid"
    process_identity = tmp_path / "process_identity.json"
    container_state = tmp_path / "container.state"
    docker_calls = tmp_path / "docker.calls"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    cidfile.write_text(f"{CONTAINER_ID}\n", encoding="ascii")
    _write_process_identity(process_identity)
    container_state.write_text(
        (
            f"running true {CONTAINER_STARTED_AT} {CONTAINER_INIT_PID} "
            f"{CONTAINER_RESTART_COUNT} 0\n"
        ),
        encoding="ascii",
    )
    _write_identity_docker(fake_docker)
    _write_executable(
        fake_nsys,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "sessions list" ]]; then
  printf '[{"id":"10206","state":"Collection","name":"%s","accessible":true}]\\n' "$EXPECTED_SESSION_NAME"
  exit 0
fi
exit 9
""",
    )
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
JQ_BIN=$(command -v jq)
RUN_BOUNDARY=$3
PROFILE_CONTAINER_CIDFILE=$4
PROCESS_IDENTITY=$5
NSIGHT_ATTESTATION="$PROCESS_IDENTITY.attestation.json"
CONTAINER=$CONTAINER_NAME
NSYS_EXPECTED_SESSION_NAME=$EXPECTED_SESSION_NAME
refresh_run_nsys_session
[[ "$NSYS_SESSION_ID" == "10206" ]]
printf 'running true 2026-07-30T12:05:00.000000000Z 5252 1 0\n' \
  > "$CONTAINER_STATE_FILE.tmp"
mv -T "$CONTAINER_STATE_FILE.tmp" "$CONTAINER_STATE_FILE"
refresh_run_nsys_session
[[ $? == 2 ]]
[[ "$NSYS_LIFECYCLE_ERROR" == *"identity/incarnation"* ]]
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(tmp_path / "lifecycle.log"),
            os.fspath(boundary),
            os.fspath(cidfile),
            os.fspath(process_identity),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "CONTAINER_ID": CONTAINER_ID,
            "CONTAINER_NAME": "profile-test-container",
            "CONTAINER_NSYS": os.fspath(fake_nsys),
            "CONTAINER_STATE_FILE": os.fspath(container_state),
            "DOCKER_CALLS": os.fspath(docker_calls),
            "EXPECTED_SESSION_NAME": SESSION_NAME,
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    exec_calls = [
        call
        for call in docker_calls.read_text(encoding="utf-8").splitlines()
        if call.startswith("exec ") and f" {CONTAINER_NSYS_BIN} " in call
    ]
    assert exec_calls == [
        _container_nsys_exec("sessions list --output-format=json")
    ]


@pytest.mark.parametrize(
    ("failure_mode", "expected_error", "expected_success"),
    (
        ("identity_mutation", "process identity changed", False),
        (
            "engine_death_collection",
            "EngineCore process identity is no longer live",
            False,
        ),
        ("engine_death_generation", "", True),
    ),
)
def test_bound_session_revalidates_process_file_and_engine_start_identity(
    tmp_path: Path,
    failure_mode: str,
    expected_error: str,
    expected_success: bool,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_nsys = tmp_path / "nsys"
    boundary = tmp_path / "run.boundary"
    cidfile = tmp_path / "container.cid"
    process_identity = tmp_path / "process_identity.json"
    engine_dead = tmp_path / "engine.dead"
    session_snapshot = tmp_path / "sessions.json"
    docker_calls = tmp_path / "docker.calls"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    cidfile.write_text(f"{CONTAINER_ID}\n", encoding="ascii")
    _write_process_identity(process_identity)
    session_snapshot.write_text(
        json.dumps(
            [
                {
                    "id": SESSION_ID,
                    "state": "Collection",
                    "name": SESSION_NAME,
                    "accessible": True,
                }
            ],
            separators=(",", ":"),
        ),
        encoding="ascii",
    )
    _write_identity_docker(fake_docker)
    _write_executable(
        fake_nsys,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "sessions list" ]]; then
  cat "$SESSION_SNAPSHOT"
  exit 0
fi
exit 9
""",
    )
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
JQ_BIN=$(command -v jq)
RUN_BOUNDARY=$3
PROFILE_CONTAINER_CIDFILE=$4
PROCESS_IDENTITY=$5
NSIGHT_ATTESTATION="$PROCESS_IDENTITY.attestation.json"
CONTAINER=$CONTAINER_NAME
NSYS_EXPECTED_SESSION_NAME=$EXPECTED_SESSION_NAME
refresh_run_nsys_session
[[ "$NSYS_SESSION_ID" == "10206" ]]
case "$FAILURE_MODE" in
  identity_mutation) printf ' ' >> "$PROCESS_IDENTITY" ;;
  engine_death_collection) : > "$ENGINE_CORE_DEAD_FILE" ;;
  engine_death_generation)
    : > "$ENGINE_CORE_DEAD_FILE"
    printf '%s\n' "$GENERATION_SNAPSHOT" > "$SESSION_SNAPSHOT"
    ;;
esac
refresh_run_nsys_session
refresh_rc=$?
if [[ "$EXPECTED_SUCCESS" == "1" ]]; then
  [[ "$refresh_rc" == 0 ]]
  [[ "$NSYS_SESSION_STATE" == "Generation" ]]
  [[ "$NSYS_POST_COLLECTION_OBSERVED" == 1 ]]
  [[ "$ENGINE_CORE_LIVENESS_OK" == 0 ]]
else
  [[ "$refresh_rc" == 2 ]]
  [[ "$NSYS_LIFECYCLE_ERROR" == *"$EXPECTED_ERROR"* ]]
fi
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(tmp_path / "lifecycle.log"),
            os.fspath(boundary),
            os.fspath(cidfile),
            os.fspath(process_identity),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "CONTAINER_ID": CONTAINER_ID,
            "CONTAINER_NAME": "profile-test-container",
            "CONTAINER_NSYS": os.fspath(fake_nsys),
            "DOCKER_CALLS": os.fspath(docker_calls),
            "ENGINE_CORE_DEAD_FILE": os.fspath(engine_dead),
            "SESSION_SNAPSHOT": os.fspath(session_snapshot),
            "GENERATION_SNAPSHOT": json.dumps(
                [
                    {
                        "id": SESSION_ID,
                        "state": "Generation",
                        "name": SESSION_NAME,
                        "accessible": True,
                    }
                ],
                separators=(",", ":"),
            ),
            "EXPECTED_SESSION_NAME": SESSION_NAME,
            "FAILURE_MODE": failure_mode,
            "EXPECTED_ERROR": expected_error,
            "EXPECTED_SUCCESS": "1" if expected_success else "0",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    nsys_execs = [
        call
        for call in docker_calls.read_text(encoding="utf-8").splitlines()
        if call.startswith("exec ") and f" {CONTAINER_NSYS_BIN} " in call
    ]
    expected_query_count = 1 if failure_mode == "identity_mutation" else 2
    assert nsys_execs == [
        _container_nsys_exec("sessions list --output-format=json")
    ] * expected_query_count


@pytest.mark.parametrize(
    ("bound", "session_state", "post_collection", "exit_code", "expected_error"),
    (
        (False, "", False, "0", "before Nsight session binding"),
        (True, "Collection", False, "0", "during Nsight Collection"),
        (True, "Generation", True, "7", "exited nonzero"),
    ),
)
def test_terminal_container_exit_fails_without_a_safe_success_transition(
    tmp_path: Path,
    bound: bool,
    session_state: str,
    post_collection: bool,
    exit_code: str,
    expected_error: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    boundary = tmp_path / "run.boundary"
    process_identity = tmp_path / "process_identity.json"
    docker_calls = tmp_path / "docker.calls"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    _write_process_identity(process_identity)
    _write_identity_docker(fake_docker)
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
RUN_BOUNDARY=$3
PROCESS_IDENTITY=$4
NSIGHT_ATTESTATION="$PROCESS_IDENTITY.attestation.json"
PROCESS_IDENTITY_STAT=$(stat -c '%d:%i:%h:%s:%Y:%Z' "$PROCESS_IDENTITY")
CONTAINER=$CONTAINER_NAME
PROFILE_CONTAINER_ID=$CONTAINER_ID
PROFILE_CONTAINER_STARTED_AT=$CONTAINER_STARTED_AT
PROFILE_CONTAINER_INIT_PID=$CONTAINER_INIT_PID
PROFILE_CONTAINER_RESTART_COUNT=$CONTAINER_RESTART_COUNT
NSYS_INJECTED_SESSION_ID=10206
NSYS_IDENTITY_ATTESTED=1
ENGINE_CORE_PID=321
ENGINE_CORE_START_TICKS=9001
if [[ "$BOUND" == "1" ]]; then
  NSYS_SESSION_ID=10206
  NSYS_SESSION_NAME=$EXPECTED_SESSION_NAME
fi
NSYS_EXPECTED_SESSION_NAME=$EXPECTED_SESSION_NAME
NSYS_SESSION_STATE=$SESSION_STATE
NSYS_POST_COLLECTION_OBSERVED=$POST_COLLECTION
refresh_run_nsys_session
[[ $? == 2 ]]
[[ "$NSYS_LIFECYCLE_ERROR" == *"$EXPECTED_ERROR"* ]]
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(tmp_path / "lifecycle.log"),
            os.fspath(boundary),
            os.fspath(process_identity),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "BOUND": "1" if bound else "0",
            "CONTAINER_ID": CONTAINER_ID,
            "CONTAINER_NAME": "profile-test-container",
            "CONTAINER_STARTED_AT": CONTAINER_STARTED_AT,
            "CONTAINER_INIT_PID": CONTAINER_INIT_PID,
            "CONTAINER_RESTART_COUNT": CONTAINER_RESTART_COUNT,
            "CONTAINER_STATUS": "exited",
            "CONTAINER_RUNNING": "false",
            "CONTAINER_STATE_PID": "0",
            "CONTAINER_EXIT_CODE": exit_code,
            "DOCKER_CALLS": os.fspath(docker_calls),
            "EXPECTED_SESSION_NAME": SESSION_NAME,
            "SESSION_STATE": session_state,
            "POST_COLLECTION": "1" if post_collection else "0",
            "EXPECTED_ERROR": expected_error,
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert not any(
        call.startswith("exec ")
        for call in docker_calls.read_text(encoding="utf-8").splitlines()
    )


@pytest.mark.parametrize("second_query", ("absent", "failure"))
def test_stop_refuses_when_exact_session_cannot_be_freshly_revalidated(
    tmp_path: Path,
    second_query: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_nsys = tmp_path / "nsys"
    boundary = tmp_path / "run.boundary"
    cidfile = tmp_path / "container.cid"
    process_identity = tmp_path / "process_identity.json"
    query_count = tmp_path / "query.count"
    nsys_calls = tmp_path / "nsys.calls"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    cidfile.write_text(f"{CONTAINER_ID}\n", encoding="ascii")
    _write_process_identity(process_identity)
    _write_identity_docker(fake_docker)
    _write_executable(
        fake_nsys,
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$NSYS_CALLS"
if [[ "$1 $2" == "sessions list" ]]; then
  [[ "${FAKE_NSYS_CONTAINER_CONTEXT:-0}" == "1" ]] || exit 18
  count=$(cat "$QUERY_COUNT" 2>/dev/null || printf '0')
  count=$((count + 1))
  printf '%s\\n' "$count" > "$QUERY_COUNT"
  if (( count == 1 )); then
    printf '[{"id":"10206","state":"Collection","name":"%s","accessible":true}]\\n' "$EXPECTED_SESSION_NAME"
  elif [[ "$SECOND_QUERY" == "absent" ]]; then
    printf '[]\\n'
  else
    exit 17
  fi
  exit 0
fi
if [[ "$1" == "stop" ]]; then
  exit 99
fi
exit 9
""",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CONTAINER_ID": CONTAINER_ID,
        "CONTAINER_NAME": "profile-test-container",
        "DOCKER_CALLS": os.fspath(tmp_path / "docker.calls"),
        "CONTAINER_NSYS": os.fspath(fake_nsys),
        "NSYS_CALLS": os.fspath(nsys_calls),
        "QUERY_COUNT": os.fspath(query_count),
        "EXPECTED_SESSION_NAME": SESSION_NAME,
        "SECOND_QUERY": second_query,
    }
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
LUMO_NSYS_BIN=$3
JQ_BIN=$(command -v jq)
RUN_BOUNDARY=$4
PROFILE_CONTAINER_CIDFILE=$5
PROCESS_IDENTITY=$6
NSIGHT_ATTESTATION="$PROCESS_IDENTITY.attestation.json"
CONTAINER=$CONTAINER_NAME
NSYS_EXPECTED_SESSION_NAME=$EXPECTED_SESSION_NAME
LUMO_NSYS_STOP_TIMEOUT_S=1
LUMO_NSYS_STOP_KILL_AFTER_S=1
refresh_run_nsys_session
[[ "$NSYS_SESSION_ID" == "10206" ]]
if stop_exact_nsys_session; then
  exit 20
fi
if [[ "$SECOND_QUERY" == "failure" ]]; then
  [[ "$NSYS_LIFECYCLE_ERROR" == *"container-scoped Nsight session query failed"* ]]
fi
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(tmp_path / "lifecycle.log"),
            os.fspath(fake_nsys),
            os.fspath(boundary),
            os.fspath(cidfile),
            os.fspath(process_identity),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    calls = nsys_calls.read_text(encoding="utf-8").splitlines()
    assert calls.count("sessions list --output-format=json") == 2
    assert not any(call.startswith("stop ") for call in calls)
    docker_calls = (tmp_path / "docker.calls").read_text(encoding="utf-8").splitlines()
    exec_calls = [
        call
        for call in docker_calls
        if call.startswith("exec ") and f" {CONTAINER_NSYS_BIN} " in call
    ]
    assert exec_calls == [
        _container_nsys_exec("sessions list --output-format=json"),
        _container_nsys_exec("sessions list --output-format=json"),
    ]
    if second_query == "failure":
        running_attestation = _incarnation_attestation()
        assert docker_calls[-3:] == [
            running_attestation,
            exec_calls[-1],
            running_attestation,
        ]


def test_timeout_preserves_exact_container_then_exit_trap_thaws(
    tmp_path: Path,
) -> None:
    driver, variant, runner = _fake_control_processes(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    docker_calls = tmp_path / "docker.calls"
    container_name_file = tmp_path / "container.name"
    _write_executable(
        fake_docker,
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
if [[ "$1" == "inspect" && "$2" == "--format" ]]; then
  [[ "$4" == "$CONTAINER_ID" ]] || exit 1
  printf '%s /%s running true %s %s %s 0\\n' \
    "$CONTAINER_ID" "$(<"$CONTAINER_NAME_FILE")" \
    "$CONTAINER_STARTED_AT" "$CONTAINER_INIT_PID" "$CONTAINER_RESTART_COUNT"
  exit 0
fi
case "$1" in
  inspect)
    [[ "$2" == "$PRESERVED_CONTAINER_NAME" ]] && exit 1
    exit 1
    ;;
  rename)
    [[ "$2" == "$CONTAINER_ID" && "$3" == "$PRESERVED_CONTAINER_NAME" ]] || exit 2
    printf '%s\\n' "$PRESERVED_CONTAINER_NAME" > "$CONTAINER_NAME_FILE"
    exit 0
    ;;
  rm)
    exit 99
    ;;
esac
exit 98
""",
    )
    stamp = "teststamp"
    original_name = "fr13-bigdenom-tail-test"
    preserved_name = f"{original_name}-preserved-{stamp}"
    container_name_file.write_text(f"{original_name}\n", encoding="ascii")
    env = {
        **_process_env(
            driver=driver,
            variant=variant,
            runner=runner,
            tmp_path=tmp_path,
        ),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "DOCKER_CALLS": os.fspath(docker_calls),
        "CONTAINER_ID": CONTAINER_ID,
        "CONTAINER_NAME_FILE": os.fspath(container_name_file),
        "PRESERVED_CONTAINER_NAME": preserved_name,
        "CONTAINER_STARTED_AT": CONTAINER_STARTED_AT,
        "CONTAINER_INIT_PID": CONTAINER_INIT_PID,
        "CONTAINER_RESTART_COUNT": CONTAINER_RESTART_COUNT,
    }
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
RUNROOT_ABS=$3
RUNROOT=$3
REPORT=$3/fresh.nsys-rep
STAMP=$4
CONTAINER=$5
PROFILE_CONTAINER_ID=$CONTAINER_ID
PROFILE_CONTAINER_STARTED_AT=$CONTAINER_STARTED_AT
PROFILE_CONTAINER_INIT_PID=$CONTAINER_INIT_PID
PROFILE_CONTAINER_RESTART_COUNT=$CONTAINER_RESTART_COUNT
bash "$6" >/dev/null 2>&1 &
DRIVER_PID=$!
DRIVER_START_TICKS=$(_process_start_ticks "$DRIVER_PID")
for (( attempt=0; attempt < 100; attempt++ )); do
  [[ -s "$7" && -s "$8" ]] && break
  sleep 0.01
done
freeze_exact_control_ancestors \
  "$DRIVER_PID" "$DRIVER_START_TICKS" "$6" "$9" \
  "$ARM_VALUE" "$KIND_VALUE" "$SUBSET_VALUE"
trap profile_cleanup EXIT
preserve_recoverable_container "report timeout"
exit 7
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(tmp_path / "lifecycle.log"),
            os.fspath(tmp_path),
            stamp,
            original_name,
            os.fspath(driver),
            os.fspath(tmp_path / "variant.pid"),
            os.fspath(tmp_path / "runner.pid"),
            os.fspath(variant),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 7, completed.stderr
    assert container_name_file.read_text(encoding="ascii").strip() == preserved_name
    calls = docker_calls.read_text(encoding="utf-8")
    assert f"rename {CONTAINER_ID} {preserved_name}" in calls
    assert f"rename {original_name} {preserved_name}" not in calls
    assert "\nrm " not in f"\n{calls}"
    recovery = (tmp_path / "nsys_recovery_state.txt").read_text(encoding="utf-8")
    assert f"CONTAINER_ID={CONTAINER_ID}" in recovery
    assert f"PRESERVED_CONTAINER={preserved_name}" in recovery

    events = tmp_path / "events.log"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if events.exists() and "variant_done\n" in events.read_text(encoding="utf-8"):
            break
        time.sleep(0.02)
    assert "variant_done\n" in events.read_text(encoding="utf-8")
    assert "driver_done\n" in events.read_text(encoding="utf-8")


def test_cleanup_and_preservation_never_follow_a_reused_container_name(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    docker_calls = tmp_path / "docker.calls"
    _write_executable(
        fake_docker,
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
if [[ "$1" == "inspect" && "$2" == "--format" ]]; then
  # The original immutable ID is gone. A foreign container now owns the old
  # name, but only a name lookup would discover it.
  [[ "$4" == "$ORIGINAL_ID" ]] && exit 1
fi
if [[ "$1" == "inspect" && "$2" == "$ORIGINAL_NAME" ]]; then
  printf '[{"Id":"%s","Name":"/%s"}]\\n' "$FOREIGN_ID" "$ORIGINAL_NAME"
  exit 0
fi
if [[ "$1" == "rm" || "$1" == "rename" ]]; then
  exit 99
fi
exit 1
""",
    )
    original_name = "fr13-bigdenom-reused"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "DOCKER_CALLS": os.fspath(docker_calls),
        "ORIGINAL_ID": CONTAINER_ID,
        "FOREIGN_ID": FOREIGN_CONTAINER_ID,
        "ORIGINAL_NAME": original_name,
        "CONTAINER_STARTED_AT": CONTAINER_STARTED_AT,
        "CONTAINER_INIT_PID": CONTAINER_INIT_PID,
        "CONTAINER_RESTART_COUNT": CONTAINER_RESTART_COUNT,
    }
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
RUNROOT_ABS=$3
REPORT=$3/report.nsys-rep
STAMP=teststamp
CONTAINER=$4
PROFILE_CONTAINER_ID=$5
PROFILE_CONTAINER_STARTED_AT=$CONTAINER_STARTED_AT
PROFILE_CONTAINER_INIT_PID=$CONTAINER_INIT_PID
PROFILE_CONTAINER_RESTART_COUNT=$CONTAINER_RESTART_COUNT
if preserve_recoverable_container "identity disappeared"; then
  exit 20
fi
PRESERVE_RECOVERABLE_STATE=0
trap profile_cleanup EXIT
exit 0
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(tmp_path / "lifecycle.log"),
            os.fspath(tmp_path),
            original_name,
            CONTAINER_ID,
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    calls = docker_calls.read_text(encoding="utf-8")
    assert _incarnation_attestation() in calls
    assert original_name not in calls
    assert FOREIGN_CONTAINER_ID not in calls
    assert "\nrm " not in f"\n{calls}"
    assert "\nrename " not in f"\n{calls}"


def test_variant_teardown_skips_a_profiler_preserved_container(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    docker_calls = tmp_path / "docker.calls"
    _write_executable(
        fake_docker,
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
if [[ "$1" == "inspect" && "$2" == "--format" && "$4" == "$CONTAINER_ID" ]]; then
  if [[ "$3" == "{{.Id}}" ]]; then
    printf '%s\\n' "$CONTAINER_ID"
    exit 0
  fi
  printf '%s /%s-preserved-run\\n' "$CONTAINER_ID" "$ORIGINAL_NAME"
  exit 0
fi
exit 99
""",
    )
    harness = tmp_path / "variant_cleanup_harness.sh"
    _write_executable(
        harness,
        "\n".join(
            (
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                _shell_function(VARIANT, "_fixed32_container_identity_matches"),
                _shell_function(VARIANT, "_fixed32_container_incarnation_matches"),
                _shell_function(VARIANT, "_fixed32_classify_container_state"),
                # site 27: the removal helper now takes evidence before it
                # judges, so the harness needs the capture helper too
                _shell_function(VARIANT, "_fixed32_capture_container_log"),
                _shell_function(VARIANT, "_fixed32_remove_attested_container"),
                "CONTAINER=$ORIGINAL_NAME",
                "FIXED32_CONTAINER_STARTED_AT=$CONTAINER_STARTED_AT",
                "FIXED32_CONTAINER_INIT_PID=$CONTAINER_INIT_PID",
                "FIXED32_CONTAINER_RESTART_COUNT=$CONTAINER_RESTART_COUNT",
                (
                    'if _fixed32_remove_attested_container "$CONTAINER_ID" '
                    '"$LOG_OUTPUT"; then exit 20; fi'
                ),
            )
        )
        + "\n",
    )
    completed = subprocess.run(
        ["bash", os.fspath(harness)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "DOCKER_CALLS": os.fspath(docker_calls),
            "CONTAINER_ID": CONTAINER_ID,
            "ORIGINAL_NAME": "fr13-bigdenom-run",
            "LOG_OUTPUT": os.fspath(tmp_path / "docker.log"),
            "CONTAINER_STARTED_AT": CONTAINER_STARTED_AT,
            "CONTAINER_INIT_PID": CONTAINER_INIT_PID,
            "CONTAINER_RESTART_COUNT": CONTAINER_RESTART_COUNT,
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    calls = docker_calls.read_text(encoding="utf-8").splitlines()
    assert calls == [
        # site 27: evidence first, before any attestation that could skip it
        f"logs {CONTAINER_ID}",
        (
            "inspect --format {{.Id}} {{.Name}} {{.State.Status}} "
            "{{.State.Running}} {{.State.Paused}} {{.State.StartedAt}} "
            "{{.State.Pid}} {{.RestartCount}} "
            f"{CONTAINER_ID}"
        ),
        (
            "inspect --format {{.Id}} {{.Name}} {{.State.Status}} "
            "{{.State.Running}} {{.State.Paused}} {{.State.StartedAt}} "
            "{{.State.Pid}} {{.RestartCount}} "
            f"{CONTAINER_ID}"
        ),
    ]
    # site 27: it used to read "logs/removal". The capture now happens BEFORE
    # this refusal, so only removal is skipped -- the message had to stop
    # claiming otherwise, and the log below must EXIST despite the refusal.
    assert (
        "container removal skipped without exact incarnation re-attestation"
        in completed.stderr
    )


@pytest.mark.parametrize(
    ("drift_mode", "expected_error"),
    (
        (
            "restart_before_logs",
            "container removal skipped without exact incarnation re-attestation",
        ),
        (
            # SITE 27 moved which check catches this. The drift is injected
            # after the `logs` call; with the capture now happening FIRST, the
            # first re-attestation is the one that runs after it and so it is
            # the one that sees the drift. The second check still exists and
            # still guards removal -- what changed is that a drift can no
            # longer slip past an earlier gate that used to run before the
            # capture. Removal is refused either way, which is the property.
            "started_at_after_logs",
            "container removal skipped without exact incarnation re-attestation",
        ),
    ),
)
def test_variant_teardown_preserves_container_after_incarnation_drift(
    tmp_path: Path,
    drift_mode: str,
    expected_error: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    docker_calls = tmp_path / "docker.calls"
    after_logs = tmp_path / "after.logs"
    _write_executable(
        fake_docker,
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
if [[ "$1" == "inspect" && "$2" == "--format" && "$4" == "$CONTAINER_ID" ]]; then
  if [[ "$3" == "{{.Id}}" ]]; then
    printf '%s\\n' "$CONTAINER_ID"
    exit 0
  fi
  started_at=$CONTAINER_STARTED_AT
  restart_count=$CONTAINER_RESTART_COUNT
  if [[ "$DRIFT_MODE" == "restart_before_logs" ]]; then
    restart_count=1
  elif [[ "$DRIFT_MODE" == "started_at_after_logs" && -e "$AFTER_LOGS" ]]; then
    started_at=2026-07-31T00:00:00.000000000Z
  fi
  printf '%s /%s running true false %s %s %s\\n' \
    "$CONTAINER_ID" "$CONTAINER_NAME" "$started_at" \
    "$CONTAINER_INIT_PID" "$restart_count"
  exit 0
fi
if [[ "$1" == "logs" && "$2" == "$CONTAINER_ID" ]]; then
  : > "$AFTER_LOGS"
  printf 'exact container log\\n'
  exit 0
fi
if [[ "$1" == "rm" ]]; then
  exit 99
fi
exit 1
""",
    )
    harness = tmp_path / "variant_incarnation_cleanup_harness.sh"
    _write_executable(
        harness,
        "\n".join(
            (
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                _shell_function(VARIANT, "_fixed32_container_incarnation_matches"),
                _shell_function(VARIANT, "_fixed32_classify_container_state"),
                # site 27: the removal helper now takes evidence before it
                # judges, so the harness needs the capture helper too
                _shell_function(VARIANT, "_fixed32_capture_container_log"),
                _shell_function(VARIANT, "_fixed32_remove_attested_container"),
                "CONTAINER=$CONTAINER_NAME",
                "FIXED32_CONTAINER_STARTED_AT=$CONTAINER_STARTED_AT",
                "FIXED32_CONTAINER_INIT_PID=$CONTAINER_INIT_PID",
                "FIXED32_CONTAINER_RESTART_COUNT=$CONTAINER_RESTART_COUNT",
                (
                    'if _fixed32_remove_attested_container "$CONTAINER_ID" '
                    '"$LOG_OUTPUT"; then exit 20; fi'
                ),
            )
        )
        + "\n",
    )
    completed = subprocess.run(
        ["bash", os.fspath(harness)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "DOCKER_CALLS": os.fspath(docker_calls),
            "CONTAINER_ID": CONTAINER_ID,
            "CONTAINER_NAME": "fr13-bigdenom-run",
            "CONTAINER_STARTED_AT": CONTAINER_STARTED_AT,
            "CONTAINER_INIT_PID": CONTAINER_INIT_PID,
            "CONTAINER_RESTART_COUNT": CONTAINER_RESTART_COUNT,
            "DRIFT_MODE": drift_mode,
            "AFTER_LOGS": os.fspath(after_logs),
            "LOG_OUTPUT": os.fspath(tmp_path / "docker.log"),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    calls = docker_calls.read_text(encoding="utf-8").splitlines()
    assert not any(call.startswith("rm ") for call in calls)
    # SITE 27. This used to assert that restart_before_logs produced NO log --
    # the defect, written down as an expectation: an incarnation that drifted
    # before the capture meant no evidence at all, and a death in that state
    # was unreadable. Capture now happens FIRST and unconditionally, so both
    # drift modes keep their log; what the drift still prevents is REMOVAL,
    # which is the judgment the attestation is actually for.
    assert calls[0] == f"logs {CONTAINER_ID}", (
        "the capture is no longer the first thing teardown does"
    )
    assert len(calls) == 3
    assert after_logs.exists(), (
        "evidence must survive an incarnation drift in either direction"
    )
    # (Non-emptiness is not asserted here: this harness's fake docker emits
    # nothing, so it would test the fake rather than the code. That the helper
    # refuses to install an empty capture over a good one is proved by
    # test_the_capture_is_monotonic, whose fake does emit.)
    assert expected_error in completed.stderr


@pytest.mark.parametrize(
    ("post_rm_state", "expected_outcome", "expected_text"),
    (
        (
            "exact",
            "exact_preserved",
            "fixed32 exact container preserved after cleanup",
        ),
        (
            "absent",
            "absent",
            "fixed32 container absent after cleanup",
        ),
        (
            "drifted",
            "drifted",
            "fixed32 container incarnation drifted after cleanup",
        ),
        (
            "unproven",
            "removal_unproven",
            "fixed32 container removal/preservation unproven after cleanup",
        ),
    ),
)
def test_variant_teardown_classifies_failed_container_removal(
    tmp_path: Path,
    post_rm_state: str,
    expected_outcome: str,
    expected_text: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    docker_calls = tmp_path / "docker.calls"
    after_rm = tmp_path / "after.rm"
    _write_executable(
        fake_docker,
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
if [[ "$1" == "inspect" && "$2" == "--format" && "$4" == "$CONTAINER_ID" ]]; then
  if [[ -e "$AFTER_RM" ]]; then
    case "$POST_RM_STATE" in
      absent|unproven) exit 1 ;;
      drifted) restart_count=1 ;;
      exact) restart_count=$CONTAINER_RESTART_COUNT ;;
      *) exit 98 ;;
    esac
  else
    restart_count=$CONTAINER_RESTART_COUNT
  fi
  if [[ "$3" == "{{.Id}}" ]]; then
    printf '%s\\n' "$CONTAINER_ID"
  else
    printf '%s /%s running true false %s %s %s\\n' \
      "$CONTAINER_ID" "$CONTAINER_NAME" "$CONTAINER_STARTED_AT" \
      "$CONTAINER_INIT_PID" "$restart_count"
  fi
  exit 0
fi
if [[ "$1" == "logs" && "$2" == "$CONTAINER_ID" ]]; then
  printf 'exact container log\\n'
  exit 0
fi
if [[ "$1" == "rm" && "$2" == "-f" && "$3" == "$CONTAINER_ID" ]]; then
  : > "$AFTER_RM"
  exit 99
fi
if [[ "$1" == "ps" && "$2" == "-aq" && "$3" == "--no-trunc" ]]; then
  [[ "$POST_RM_STATE" != "unproven" ]] || exit 97
  exit 0
fi
exit 96
""",
    )
    harness = tmp_path / "variant_failed_removal_harness.sh"
    _write_executable(
        harness,
        "\n".join(
            (
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                _shell_function(VARIANT, "_fixed32_container_incarnation_matches"),
                _shell_function(VARIANT, "_fixed32_classify_container_state"),
                _shell_function(
                    VARIANT,
                    "_fixed32_record_container_cleanup_failure",
                ),
                # site 27: the removal helper now takes evidence before it
                # judges, so the harness needs the capture helper too
                _shell_function(VARIANT, "_fixed32_capture_container_log"),
                _shell_function(VARIANT, "_fixed32_remove_attested_container"),
                "CONTAINER=$CONTAINER_NAME",
                "FIXED32_CONTAINER_STARTED_AT=$CONTAINER_STARTED_AT",
                "FIXED32_CONTAINER_INIT_PID=$CONTAINER_INIT_PID",
                "FIXED32_CONTAINER_RESTART_COUNT=$CONTAINER_RESTART_COUNT",
                "FIXED32_CONTAINER_CLEANUP_OUTCOME=",
                (
                    'if _fixed32_remove_attested_container "$CONTAINER_ID" '
                    '"$LOG_OUTPUT"; then exit 20; fi'
                ),
                (
                    '_fixed32_record_container_cleanup_failure "$CONTAINER_ID" '
                    '"cleanup attestation/removal failure"'
                ),
                'printf "%s\\n" "$FIXED32_CONTAINER_CLEANUP_OUTCOME" > "$OUTCOME"',
            )
        )
        + "\n",
    )
    arm_dir = tmp_path / "arm"
    arm_dir.mkdir()
    outcome = tmp_path / "outcome"
    completed = subprocess.run(
        ["bash", os.fspath(harness)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "DOCKER_CALLS": os.fspath(docker_calls),
            "CONTAINER_ID": CONTAINER_ID,
            "CONTAINER_NAME": "fr13-bigdenom-run",
            "CONTAINER_STARTED_AT": CONTAINER_STARTED_AT,
            "CONTAINER_INIT_PID": CONTAINER_INIT_PID,
            "CONTAINER_RESTART_COUNT": CONTAINER_RESTART_COUNT,
            "POST_RM_STATE": post_rm_state,
            "AFTER_RM": os.fspath(after_rm),
            "LOG_OUTPUT": os.fspath(tmp_path / "docker.log"),
            "ARMDIR": os.fspath(arm_dir),
            "OUTCOME": os.fspath(outcome),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert outcome.read_text(encoding="ascii") == f"{expected_outcome}\n"
    assert expected_text in completed.stderr
    calls = docker_calls.read_text(encoding="ascii").splitlines()
    assert calls.count(f"rm -f {CONTAINER_ID}") == 1
    preserved = arm_dir / "fixed32_container_preserved.txt"
    cleanup_failure = arm_dir / "fixed32_container_cleanup_failure.txt"
    if expected_outcome == "exact_preserved":
        assert expected_text in preserved.read_text(encoding="ascii")
        assert not cleanup_failure.exists()
    else:
        assert not preserved.exists()
        assert expected_text in cleanup_failure.read_text(encoding="ascii")


def test_stop_timeout_kills_a_term_ignoring_nsys_process(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_nsys = tmp_path / "nsys"
    nsys_calls = tmp_path / "nsys.calls"
    boundary = tmp_path / "run.boundary"
    process_identity = tmp_path / "process_identity.json"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    _write_process_identity(process_identity)
    _write_identity_docker(fake_docker)
    _write_executable(
        fake_nsys,
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$NSYS_CALLS"
if [[ "$1 $2" == "sessions list" ]]; then
  printf '[{"id":"10206","state":"Collection","name":"%s","accessible":true}]\\n' "$EXPECTED_SESSION_NAME"
  exit 0
fi
if [[ "$1" == "stop" ]]; then
  trap '' TERM
  while :; do :; done
fi
exit 9
""",
    )
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
LUMO_NSYS_BIN=$3
NSYS_SESSION_ID=10206
NSYS_INJECTED_SESSION_ID=10206
NSYS_IDENTITY_ATTESTED=1
NSYS_SESSION_NAME=$4
NSYS_EXPECTED_SESSION_NAME=$4
PROCESS_IDENTITY=$5
NSIGHT_ATTESTATION="$PROCESS_IDENTITY.attestation.json"
RUN_BOUNDARY=$6
PROCESS_IDENTITY_STAT=$(stat -c '%d:%i:%h:%s:%Y:%Z' "$PROCESS_IDENTITY")
ENGINE_CORE_PID=321
ENGINE_CORE_START_TICKS=9001
PROFILE_CONTAINER_ID=$CONTAINER_ID
PROFILE_CONTAINER_STARTED_AT=$CONTAINER_STARTED_AT
PROFILE_CONTAINER_INIT_PID=$CONTAINER_INIT_PID
PROFILE_CONTAINER_RESTART_COUNT=$CONTAINER_RESTART_COUNT
CONTAINER=$CONTAINER_NAME
JQ_BIN=$(command -v jq)
LUMO_NSYS_STOP_TIMEOUT_S=0.1
LUMO_NSYS_STOP_KILL_AFTER_S=0.2
if stop_exact_nsys_session; then
  exit 20
fi
"""
    started = time.monotonic()
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(tmp_path / "lifecycle.log"),
            os.fspath(fake_nsys),
            SESSION_NAME,
            os.fspath(process_identity),
            os.fspath(boundary),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "NSYS_CALLS": os.fspath(nsys_calls),
            "EXPECTED_SESSION_NAME": SESSION_NAME,
            "CONTAINER_ID": CONTAINER_ID,
            "CONTAINER_NAME": "profile-test-container",
            "DOCKER_CALLS": os.fspath(tmp_path / "docker.calls"),
            "CONTAINER_NSYS": os.fspath(fake_nsys),
            "CONTAINER_STARTED_AT": CONTAINER_STARTED_AT,
            "CONTAINER_INIT_PID": CONTAINER_INIT_PID,
            "CONTAINER_RESTART_COUNT": CONTAINER_RESTART_COUNT,
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stderr
    assert elapsed < 2
    assert nsys_calls.read_text(encoding="utf-8").splitlines() == [
        "sessions list --output-format=json",
        f"stop --session={SESSION_ID}",
    ]
    docker_calls = (tmp_path / "docker.calls").read_text(
        encoding="utf-8"
    ).splitlines()
    assert _container_nsys_exec(
        f"stop --session={SESSION_ID}",
        timeout_s="0.1",
        kill_after_s="0.2",
    ) in docker_calls


def test_export_timeout_kills_a_term_ignoring_nsys_process_group(
    tmp_path: Path,
) -> None:
    fake_nsys = tmp_path / "nsys"
    parent_pid_path = tmp_path / "parent.pid"
    child_pid_path = tmp_path / "child.pid"
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"report")
    _write_executable(
        fake_nsys,
        """#!/usr/bin/env bash
printf '%s\n' "$$" > "$PARENT_PID_PATH"
trap '' TERM
bash -c 'trap "" TERM; printf "%s\\n" "$$" > "$CHILD_PID_PATH"; while :; do :; done' &
while :; do :; done
""",
    )
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
LUMO_NSYS_BIN=$3
LUMO_NSYS_EXPORT_TIMEOUT_S=0.1
LUMO_NSYS_EXPORT_KILL_AFTER_S=0.2
if verify_report_readable "$4" "$5"; then
  exit 20
fi
"""
    started = time.monotonic()
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "--",
            os.fspath(PROFILE),
            os.fspath(tmp_path / "lifecycle.log"),
            os.fspath(fake_nsys),
            os.fspath(report),
            os.fspath(tmp_path / "readability.json"),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PARENT_PID_PATH": os.fspath(parent_pid_path),
            "CHILD_PID_PATH": os.fspath(child_pid_path),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stderr
    assert elapsed < 2
    for pid_path in (parent_pid_path, child_pid_path):
        pid = int(pid_path.read_text(encoding="ascii"))
        for _ in range(100):
            stat_path = Path(f"/proc/{pid}/stat")
            if not stat_path.exists() or stat_path.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"timed-out nsys export process remains live: {pid}")


def test_profiler_preflight_rejects_a_stopped_exact_name_without_removing_it(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    docker_calls = tmp_path / "docker.calls"
    _write_executable(
        fake_docker,
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
if [[ "$1 $2" == "ps -q" ]]; then
  exit 0
fi
if [[ "$1 $2 $3" == "ps -aq --filter" ]]; then
  printf '%s\\n' "$STOPPED_ID"
  exit 0
fi
exit 9
""",
    )
    completed = subprocess.run(
        ["bash", os.fspath(PROFILE)],
        cwd=REPO,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "DOCKER_CALLS": os.fspath(docker_calls),
            "STOPPED_ID": FOREIGN_CONTAINER_ID,
            "STAMP": "20260730T120000Z",
            "TAG": "stopped-name-test",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "exact container name already exists" in completed.stderr
    calls = docker_calls.read_text(encoding="utf-8")
    assert "ps -q" in calls
    assert "ps -aq --filter name=^/" in calls
    assert "\nrm " not in f"\n{calls}"


def test_fixed32_oom_guard_exits_on_name_drift_without_kill_or_rm(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_free = fake_bin / "free"
    docker_calls = tmp_path / "docker.calls"
    _write_executable(
        fake_docker,
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
if [[ "$1" == "inspect" && "$2" == "--format" && "$4" == "$CONTAINER_ID" ]]; then
  printf '%s /foreign-reused-name true\\n' "$CONTAINER_ID"
  exit 0
fi
exit 99
""",
    )
    _write_executable(
        fake_free,
        """#!/usr/bin/env bash
printf '              total        used        free      shared  buff/cache   available\\n'
printf 'Mem:         120000      119999           1           0           0           1\\n'
""",
    )
    completed = subprocess.run(
        ["bash", os.fspath(GUARD)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "REPO": os.fspath(tmp_path),
            "DOCKER_CALLS": os.fspath(docker_calls),
            "CONTAINER_ID": CONTAINER_ID,
            "GPU_GUARD_CONTAINER_ID": CONTAINER_ID,
            "GPU_GUARD_EXPECTED_NAME": "expected-run-name",
            "GPU_GUARD_LOG": os.fspath(tmp_path / "guard.log"),
            "GPU_GUARD_POLL_S": "1",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )

    assert completed.returncode == 0, completed.stderr
    calls = docker_calls.read_text(encoding="utf-8").splitlines()
    assert calls == [
        (
            "inspect --format "
            "{{.Id}} {{.Name}} {{.State.Running}} "
            f"{CONTAINER_ID}"
        )
    ]
    assert "changed name; guard EXIT" in completed.stderr
