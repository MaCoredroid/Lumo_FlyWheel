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
SESSION_ID = "10206"
SESSION_NAME = "fr13-fixed32-20260730T120000Z-p4242"


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


def _write_process_identity(path: Path, session_id: str = SESSION_ID) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "fr13-fixed32-process-identity-v1",
                "pid1": {
                    "pid": 1,
                    "argv": ["nsys", "profile", "vllm", "serve"],
                    "environ": [f"LUMO_NSYS_SESSION_NAME={SESSION_NAME}"],
                    "forked_fa2_maps": [],
                },
                "engine_core": {
                    "pid": 321,
                    "argv": ["VLLM::EngineCore"],
                    "environ": [f"NSYS_PROFILING_SESSION_ID={session_id}"],
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
  printf '%s /%s\\n' "$CONTAINER_ID" "{name_source}"
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
    assert "NSYS_STOPPED_START_TICKS" in text
    assert "thaw_exact_control_ancestors" in text
    assert "--type=info" in text
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
    nsys_calls = tmp_path / "nsys.calls"
    docker_calls = tmp_path / "docker.calls"
    lifecycle_log = tmp_path / "lifecycle.log"
    boundary.touch()
    os.utime(boundary, (time.time() - 10, time.time() - 10))
    cidfile.write_text(f"{CONTAINER_ID}\n", encoding="ascii")
    _write_process_identity(process_identity)
    _write_identity_docker(fake_docker)
    _write_executable(
        fake_nsys,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "sessions list" ]]; then
  if [[ -e "$SESSION_DONE" ]]; then
    phase=done
    printf '[{"id":"7777","state":"Collection","name":"%s","accessible":true}]\\n' "$EXPECTED_SESSION_NAME"
  elif [[ -e "$SESSION_GENERATION" ]]; then
    phase=generation
    printf '[{"id":"7777","state":"Collection","name":"%s","accessible":true},{"id":"10206","state":"Generation","name":"%s","accessible":true}]\\n' "$EXPECTED_SESSION_NAME" "$EXPECTED_SESSION_NAME"
  else
    phase=collection
    printf '[{"id":"7777","state":"Collection","name":"%s","accessible":true},{"id":"10206","state":"Collection","name":"%s","accessible":true}]\\n' "$EXPECTED_SESSION_NAME" "$EXPECTED_SESSION_NAME"
  fi
  printf 'sessions list phase=%s\\n' "$phase" >> "$NSYS_CALLS"
  exit 0
fi
printf '%s\\n' "$*" >> "$NSYS_CALLS"
if [[ "$1" == "export" ]]; then
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
NSYS_BASELINE_JSON='[{"id":"9000","state":"Collection","name":"old-session","accessible":true}]'
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
wait_for_fresh_stable_report \
  "$DRIVER_PID" "$DRIVER_START_TICKS" "$5" "$6" "$7"
grep -q '^runner_done$' "$EVENTS"
! grep -q '^variant_done$' "$EVENTS"
[[ ${#NSYS_STOPPED_PIDS[@]} == 2 ]]
thaw_exact_control_ancestors
wait "$DRIVER_PID"
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
    done_polls = [
        call for call in calls[:export_index] if call == "sessions list phase=done"
    ]
    assert len(done_polls) >= 3
    lifecycle = lifecycle_log.read_text(encoding="utf-8")
    assert f"attested run container id={CONTAINER_ID}" in lifecycle
    assert f"attested EngineCore-injected Nsight session id={SESSION_ID}" in lifecycle
    assert f"bound exact run Nsight session id={SESSION_ID}" in lifecycle
    assert "bound exact run Nsight session id=7777" not in lifecycle
    assert "Nsight collection window opened with teardown control frozen" in lifecycle
    assert "Nsight report generation began" in lifecycle
    assert "stable across 3 polls and readable via pinned nsys" in lifecycle


def test_session_binding_ignores_unrelated_fresh_same_name_and_stops_only_exact(
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
  cat "$SESSION_SNAPSHOT"
  exit 0
fi
if [[ "$1" == "stop" ]]; then
  exit 0
fi
exit 9
""",
    )
    unrelated = [
        {
            "id": "7777",
            "state": "Collection",
            "name": SESSION_NAME,
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
CONTAINER=$CONTAINER_NAME
NSYS_EXPECTED_SESSION_NAME=$7
NSYS_BASELINE_JSON='[{"id":"9000","state":"Collection","name":"old","accessible":true}]'
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
    lifecycle = lifecycle_log.read_text(encoding="utf-8")
    assert f"bound exact run Nsight session id={SESSION_ID}" in lifecycle
    assert "bound exact run Nsight session id=7777" not in lifecycle


def test_exact_injected_session_with_wrong_name_fails_closed(
    tmp_path: Path,
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
  printf '[{"id":"10206","state":"Collection","name":"other-run","accessible":true}]\\n'
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
    }
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
LUMO_NSYS_BIN=$3
JQ_BIN=$(command -v jq)
RUN_BOUNDARY=$4
PROFILE_CONTAINER_CIDFILE=$5
PROCESS_IDENTITY=$6
CONTAINER=$CONTAINER_NAME
NSYS_EXPECTED_SESSION_NAME=$7
NSYS_BASELINE_JSON='[]'
refresh_run_nsys_session
[[ $? == 2 ]]
[[ -z "$NSYS_SESSION_ID" ]]
[[ "$NSYS_LIFECYCLE_ERROR" == *"does not match"* ]]
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
CONTAINER=$CONTAINER_NAME
NSYS_EXPECTED_SESSION_NAME=$EXPECTED_SESSION_NAME
NSYS_BASELINE_JSON='[]'
LUMO_NSYS_STOP_TIMEOUT_S=1
LUMO_NSYS_STOP_KILL_AFTER_S=1
refresh_run_nsys_session
[[ "$NSYS_SESSION_ID" == "10206" ]]
if stop_exact_nsys_session; then
  exit 20
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
  printf '%s /%s\\n' "$CONTAINER_ID" "$(<"$CONTAINER_NAME_FILE")"
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
    }
    script = r"""
source "$1"
NSYS_LIFECYCLE_LOG=$2
RUNROOT_ABS=$3
REPORT=$3/report.nsys-rep
STAMP=teststamp
CONTAINER=$4
PROFILE_CONTAINER_ID=$5
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
    assert f"inspect --format {{{{.Id}}}} {{{{.Name}}}} {CONTAINER_ID}" in calls
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
                _shell_function(VARIANT, "_fixed32_remove_attested_container"),
                "CONTAINER=$ORIGINAL_NAME",
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
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    calls = docker_calls.read_text(encoding="utf-8").splitlines()
    assert calls == [
        f"inspect --format {{{{.Id}}}} {{{{.Name}}}} {CONTAINER_ID}"
    ]
    assert "logs/removal skipped without exact re-attestation" in completed.stderr


def test_stop_timeout_kills_a_term_ignoring_nsys_process(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_nsys = tmp_path / "nsys"
    nsys_calls = tmp_path / "nsys.calls"
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
NSYS_SESSION_NAME=$4
NSYS_EXPECTED_SESSION_NAME=$4
NSYS_BASELINE_JSON='[]'
PROFILE_CONTAINER_ID=$CONTAINER_ID
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
