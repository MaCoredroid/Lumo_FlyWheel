from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import pytest


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
SERVE = REPO / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"
OFFLOAD = REPO / "scripts" / "swe_x86_helpers" / "offload_codex_proxy.sh"
RELAUNCH = REPO / "scripts" / "swe_x86_helpers" / "relaunch_proxy_remote.sh"
CONTRACT = REPO / "scripts" / "fr13_fixed32_contract.py"
FLOOR = REPO / "scripts" / "fr13_floor_gate.py"
DEPTH = REPO / "scripts" / "fr13_depth_acceptance.py"
EXACT4 = REPO / "config" / "fr13_fixed32" / "subset_b4_four.json"
CONTAINER_ID = "a" * 64
CONTAINER_STARTED_AT = "2026-07-31T12:00:00.123456789Z"
CONTAINER_INIT_PID = "4242"
CONTAINER_RESTART_COUNT = "0"


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_valid_engine_ledger_fixture(
    arm_dir: Path,
    source_ledger: Path,
) -> None:
    import sys

    sys.path.insert(0, os.fspath(REPO / "scripts"))
    sys.path.insert(0, os.fspath(REPO / "src"))
    from fr13_floor_gate import validate_canonical_subset
    from lumo_flywheel_serving.inference_proxy import Fixed32EngineIngress

    subset = validate_canonical_subset(EXACT4)
    task_ids = subset["task_ids"]
    secret_path = arm_dir / "engine-secret.json"
    secret_path.write_text(
        json.dumps(
            {
                "schema": "fr13-fixed32-ingress-secrets-v1",
                "task_hmac_key_hex": "1" * 64,
                "engine_bearer": "fr13_engine_" + "2" * 64,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    secret_path.chmod(0o600)
    engine = Fixed32EngineIngress(
        secret_file=secret_path,
        canonical_task_ids=task_ids,
        ledger_path=source_ledger,
    )
    for route in ("chat", "responses"):
        for reason in ("missing_bearer", "invalid_engine_bearer"):
            engine.reject(route=route, task_key_id=None, reason=reason)
    begin = engine.begin(
        {
            "schema": "fr13-fixed32-ingress-begin-v1",
            "canonical_task_count": len(task_ids),
            "canonical_task_set_sha256": engine.canonical_task_set_sha256,
        }
    )
    finalize = engine.finalize({"schema": "fr13-fixed32-ingress-finalize-v1"})
    engine.ledger.close()
    for name, payload in (
        ("fixed32_engine_ingress_begin.json", begin),
        ("fixed32_engine_ingress_finalize.json", finalize),
    ):
        (arm_dir / name).write_text(
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )


def test_fixed32_launcher_stages_private_secret_and_installs_middleware() -> None:
    launcher = source(LAUNCHER)
    contract = source(CONTRACT)

    assert "stat -c '%a' \"$FR13_FIXED32_INGRESS_SECRET_FILE\"" in launcher
    assert "fixed32 ingress secret file mode must be exactly 600" in launcher
    assert (
        "$FR13_FIXED32_INGRESS_SECRET_FILE:"
        "$FR13_FIXED32_CONTAINER_INGRESS_SECRET_SOURCE:ro"
    ) in launcher
    assert "os.O_RDONLY | os.O_NOFOLLOW" in launcher
    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW" in launcher
    assert "os.fchown(target_fd, 0, 0)" in launcher
    assert (
        "--middleware "
        "lumo_flywheel_serving.inference_proxy."
        "Fixed32EngineIngressMiddleware"
    ) in launcher
    assert '"--middleware",' in contract
    assert (
        '"lumo_flywheel_serving.inference_proxy.Fixed32EngineIngressMiddleware",'
    ) in contract


def test_fixed32_offload_uses_stdin_secret_and_disables_raw_dumps() -> None:
    offload = source(OFFLOAD)
    relaunch = source(RELAUNCH)
    serve = source(SERVE)

    assert "cat > $REMOTE_FIXED32_SECRET" in offload
    assert '< "$FIXED32_SECRET_LOCAL"' in offload
    assert "stat -c '%a' $REMOTE_FIXED32_SECRET" in offload
    assert "unset LUMO_PROXY_PAIR_DUMP_DIR LUMO_PROXY_REQUEST_DUMP_DIR" in offload
    assert "export LUMO_PROXY_FIXED32_DISABLE_RAW_DUMPS=1" in offload
    assert 'if [[ -z "$FIXED32_TASK_IDS" ]]; then' in offload
    assert 'if [ "${LUMO_PROXY_FIXED32_DISABLE_RAW_DUMPS:-0}" = "1" ]; then' in relaunch
    assert (
        'if [ "${LUMO_PROXY_FIXED32_DISABLE_RAW_DUMPS:-0}" != "1" ]; then' in relaunch
    )
    assert 'if [[ -z "$FIXED32_MODE" ]]; then' in serve
    assert "fixed32 requires OFFLOAD_AGENT=1" in serve


def test_fixed32_preflight_covers_deny_default_alternate_routes() -> None:
    proxy = source(OFFLOAD)
    engine = source(SERVE)
    floor = source(FLOOR)
    depth = source(DEPTH)

    for text in (proxy, floor, depth):
        assert '"denied_alternate_routes"' in text
        assert '"/admin/invalidate"' in text
        assert '"/admin/load_tuned_config"' in text
    for text in (engine, floor, depth):
        assert '"/v1/completions"' in text
        assert '"/reset_prefix_cache"' in text
    assert proxy.count("status_code") > 0
    assert engine.count("status_code") > 0


def test_fixed32_campaign_closes_ingress_before_fetch_and_terminal_audit() -> None:
    serve = source(SERVE)

    runner = serve.index(".venv/bin/python scripts/run_swe_bench_q36_a.py")
    proxy_finalize = serve.index(
        'bash "$OFFLOAD_HELPER" control "$OFFLOAD_HOST" finalize',
        runner,
    )
    engine_finalize = serve.index(
        'finalize "$ARMDIR/fixed32_engine_ingress_finalize.json"',
        proxy_finalize,
    )
    fetch = serve.index(
        'bash "$OFFLOAD_HELPER" fetch "$OFFLOAD_HOST" "$ARMDIR_ABS"',
        engine_finalize,
    )
    assert runner < proxy_finalize < engine_finalize < fetch

    teardown = serve[serve.index("teardown(){") : serve.index("\ntrap teardown EXIT")]
    stop_proxy = teardown.index('bash "$OFFLOAD_HELPER" stop "$OFFLOAD_HOST"')
    final_flush = teardown.index("--action final")
    ledger_snapshot = teardown.index("_fixed32_snapshot_engine_ingress_ledger")
    audit = teardown.index("write_fixed32_chat_traffic_audit")
    remove = teardown.index("_fixed32_remove_attested_container")
    assert stop_proxy < final_flush < ledger_snapshot < audit < remove
    assert "if (( flush_rc == 0 && fixed32_container_attested == 1 )); then" in teardown
    assert "if (( ledger_snapshot_rc == 0 )); then" in teardown
    assert "container_cleanup_rc=$?" in teardown
    assert "_fixed32_record_container_cleanup_failure" in teardown
    assert "_fixed32_remove_attested_stopped_container" in teardown
    assert "(( rc == 0 )) && rc=17" in teardown
    assert 'case "$FIXED32_CONTAINER_CLEANUP_OUTCOME" in' in serve
    assert "fixed32 exact container preserved after" in serve
    assert "fixed32_container_cleanup_failure.txt" in serve
    assert '"engine-ledger materialization failure"' in teardown
    assert 'chmod 700 "$ARMDIR" "$ARMDIR/logs"' in serve
    assert "_fixed32_container_incarnation_matches" in serve
    assert "_fixed32_stopped_container_incarnation_matches" in serve
    assert ".State.StartedAt" in serve
    assert ".RestartCount" in serve
    assert "build_fixed32_chat_traffic_audit" in serve
    assert '"$FR13_FIXED32_B1_DIAGNOSTIC" <<\'PY\'' in serve
    assert "concurrency=int(concurrency_text)" not in serve
    assert "fr13-fixed32-eager-kernel-terminal-v1" in teardown
    assert "fr13-fixed32-eager-kernel-traffic-audit-skip-v1" in teardown
    assert '"authenticated_engine_ledger_snapshotted":true' in teardown
    assert '"graph_census_audit_used":false' in teardown


@pytest.mark.parametrize(
    ("copy_mode", "expected_error"),
    (
        ("success", None),
        ("copy_failure", "snapshot docker cp failed"),
        ("name_drift", "incarnation drifted during snapshot"),
        ("restart_drift", "incarnation drifted during snapshot"),
        (
            "copied_symlink",
            "not a host-owned readable nonempty regular file",
        ),
        (
            "copied_hardlink",
            "not a host-owned readable nonempty regular file",
        ),
        (
            "invalid_content",
            "finalized engine-ledger validation/publication failed",
        ),
        (
            "finalize_mismatch",
            "finalized engine-ledger validation/publication failed",
        ),
        (
            "inplace_mutation",
            "validated engine ledger bytes changed before publication",
        ),
    ),
)
def test_terminal_engine_ledger_snapshot_is_private_fresh_and_attested(
    tmp_path: Path,
    copy_mode: str,
    expected_error: str | None,
) -> None:
    serve = source(SERVE)
    functions = serve[
        serve.index("_fixed32_container_identity_matches() {") : serve.index(
            "\nteardown(){"
        )
    ]
    publication_ready = tmp_path / "publication.ready"
    publication_continue = tmp_path / "publication.continue"
    if copy_mode == "inplace_mutation":
        synchronization_point = (
            "        source_before_publish = os.fstat(snapshot_fd)\n"
        )
        assert functions.count(synchronization_point) == 1
        functions = functions.replace(
            synchronization_point,
            (
                '        Path(os.environ["PUBLICATION_READY"]).touch()\n'
                '        while not Path(os.environ["PUBLICATION_CONTINUE"]).exists():\n'
                "            __import__('time').sleep(0.01)\n"
                f"{synchronization_point}"
            ),
        )
    arm_dir = tmp_path / "arm"
    logs_dir = arm_dir / "logs"
    logs_dir.mkdir(parents=True)
    arm_dir.chmod(0o700)
    logs_dir.chmod(0o700)
    destination = logs_dir / "fr13_fixed32_engine_ingress.jsonl"
    destination.write_text("stale\n", encoding="ascii")
    destination.chmod(0)
    source_ledger = tmp_path / "engine-ledger.source.jsonl"
    _write_valid_engine_ledger_fixture(arm_dir, source_ledger)
    if copy_mode == "finalize_mismatch":
        finalize_path = arm_dir / "fixed32_engine_ingress_finalize.json"
        finalize = json.loads(finalize_path.read_text(encoding="ascii"))
        finalize["ledger_records"] += 1
        finalize_path.write_text(
            json.dumps(
                finalize,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )
    docker_calls = tmp_path / "docker.calls"
    copy_finished = tmp_path / "copy.finished"
    symlink_victim = tmp_path / "symlink.victim"
    symlink_victim.write_text("unchanged\n", encoding="ascii")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
if [[ "$1" == "inspect" && "$2" == "--format" ]]; then
  [[ "$4" == "$CONTAINER_ID" ]] || exit 1
  name=$CONTAINER_NAME
  restart_count=$CONTAINER_RESTART_COUNT
  if [[ -e "$COPY_FINISHED" && "$COPY_MODE" == "name_drift" ]]; then
    name=foreign
  elif [[ -e "$COPY_FINISHED" && "$COPY_MODE" == "restart_drift" ]]; then
    restart_count=1
  fi
  printf '%s /%s running true false %s %s %s\\n' \
    "$CONTAINER_ID" "$name" "$CONTAINER_STARTED_AT" \
    "$CONTAINER_INIT_PID" "$restart_count"
  exit 0
fi
if [[ "$1" == "cp" ]]; then
  [[ "$2" == "$CONTAINER_ID:/logs/fr13_fixed32_engine_ingress.jsonl" ]] \
    || exit 1
  [[ "$COPY_MODE" != "copy_failure" ]] || exit 42
  case "$COPY_MODE" in
    copied_symlink)
      ln -s "$SYMLINK_VICTIM" "$3" || exit 1
      ;;
    copied_hardlink)
      ln "$SOURCE_LEDGER" "$3" || exit 1
      ;;
    invalid_content)
      printf 'not-json\\n' > "$3" || exit 1
      ;;
    *)
      /bin/cp -- "$SOURCE_LEDGER" "$3" || exit 1
      ;;
  esac
  : > "$COPY_FINISHED"
  exit 0
fi
exit 1
""",
        encoding="ascii",
    )
    fake_docker.chmod(0o755)
    command = [
        "bash",
        "-c",
        functions
        + r"""
ARMDIR=$1
ARMDIR_ABS=$1
CONTAINER=$2
SUBSET=$5
_capture_fixed32_container_incarnation "$3" || exit 99
if _fixed32_snapshot_engine_ingress_ledger "$3"; then
  snapshot_rc=0
else
  snapshot_rc=$?
fi
printf '%s\n' "$snapshot_rc" > "$4"
""",
        "--",
        os.fspath(arm_dir),
        "fixed32-test",
        CONTAINER_ID,
        os.fspath(tmp_path / "snapshot.rc"),
        os.fspath(EXACT4),
    ]
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CONTAINER_ID": CONTAINER_ID,
        "CONTAINER_NAME": "fixed32-test",
        "CONTAINER_STARTED_AT": CONTAINER_STARTED_AT,
        "CONTAINER_INIT_PID": CONTAINER_INIT_PID,
        "CONTAINER_RESTART_COUNT": CONTAINER_RESTART_COUNT,
        "COPY_MODE": copy_mode,
        "SOURCE_LEDGER": os.fspath(source_ledger),
        "DOCKER_CALLS": os.fspath(docker_calls),
        "COPY_FINISHED": os.fspath(copy_finished),
        "SYMLINK_VICTIM": os.fspath(symlink_victim),
        "PUBLICATION_READY": os.fspath(publication_ready),
        "PUBLICATION_CONTINUE": os.fspath(publication_continue),
    }
    if copy_mode == "inplace_mutation":
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 10
        while not publication_ready.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                process.kill()
                raise AssertionError("publication synchronization point timed out")
            time.sleep(0.01)
        assert publication_ready.exists(), process.communicate(timeout=1)
        snapshots = list(arm_dir.glob(".fixed32_engine_ingress_snapshot.*/*"))
        assert len(snapshots) == 1
        original = snapshots[0].read_bytes()
        marker = b'"seq":0'
        marker_offset = original.find(marker)
        assert marker_offset >= 0
        mutated = bytearray(original)
        mutated[marker_offset + len(marker) - 1] = ord("1")
        assert len(mutated) == len(original)
        snapshots[0].write_bytes(mutated)
        publication_continue.touch()
        stdout, stderr = process.communicate(timeout=10)
        completed = subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout,
            stderr,
        )
    else:
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    assert completed.returncode == 0, completed.stderr
    calls = docker_calls.read_text(encoding="ascii").splitlines()
    assert calls[0].startswith("inspect --format ")
    assert calls[1].startswith("inspect --format ")
    assert calls[2].startswith(
        f"cp {CONTAINER_ID}:/logs/fr13_fixed32_engine_ingress.jsonl "
    )
    assert not list(arm_dir.glob(".fixed32_engine_ingress_snapshot.*"))
    assert symlink_victim.read_bytes() == b"unchanged\n"
    if expected_error is None:
        assert (tmp_path / "snapshot.rc").read_text(encoding="ascii") == "0\n", (
            completed.stderr
        )
        assert calls[3].startswith("inspect --format ")
        assert calls[4].startswith("inspect --format ")
        assert destination.read_bytes() == source_ledger.read_bytes()
        assert destination.stat().st_uid == os.getuid()
        assert destination.stat().st_mode & 0o777 == 0o600
        assert "snapshotted from immutable container" in completed.stdout
    else:
        assert (tmp_path / "snapshot.rc").read_text(encoding="ascii") == "1\n"
        assert expected_error in completed.stderr
        destination.chmod(0o600)
        assert destination.read_bytes() == b"stale\n"


def test_floor_and_depth_require_exact_ingress_and_trace_evidence() -> None:
    floor = source(FLOOR)
    depth = source(DEPTH)
    required_gates = (
        "fixed32_ingress_proxy_engine_exact",
        "fixed32_zero_campaign_rejections",
        "fixed32_raw_proxy_dumps_disabled",
    )

    assert "fr13.canonical_swe_verified_fixed32_floor_gate.v11" in floor
    assert "fr13-fixed32-chat-task-provenance-audit-v3" in floor
    assert "fr13.depth_acceptance.fixed32.v2" in depth
    for gate in required_gates:
        assert gate in floor
        assert gate in depth
    assert "trace_model_request_id_sha256s" in floor
    assert "task successful request evidence differs from terminal trace" in floor
    assert "trace_request_id_sha256s" in depth
    assert "task trace/engine request evidence differs" in depth
