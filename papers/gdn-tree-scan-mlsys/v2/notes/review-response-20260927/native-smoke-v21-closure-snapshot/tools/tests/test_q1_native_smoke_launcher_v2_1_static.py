"""Native-smoke launcher v2.1: static contract + ACTUAL-SHELL mocked-docker controls for the cleanup contract (reviewer finding on v2 stop_engine).
No GPU, no real docker: a mock `docker`/`nvidia-smi` on PATH drives the shipped q1_native_smoke_cleanup_v2_1.sh functions and the launcher's exact trap line.
Controls: success (verified stop + rm), stop failure / still running (rc 8, no stopped timestamp, logs+inspect preserved), stopped but rm fails (rc 9),
finish_run terminal semantics (driver rc wins; cleanup failure otherwise non-zero), trap semantics (cleanup failure surfaces on a clean exit, never masks a failure)."""
import json, os, stat, subprocess, sys, tempfile, textwrap

HERE = os.path.dirname(os.path.abspath(__file__)); TOOLS = os.path.dirname(HERE); CAMP = os.path.dirname(TOOLS)
L2 = os.path.join(TOOLS, "run_q1_native_smoke_v2.sh"); L = os.path.join(TOOLS, "run_q1_native_smoke_v2_1.sh"); LIB = os.path.join(TOOLS, "q1_native_smoke_cleanup_v2_1.sh")
GATE_KEYS = ["image_id", "smoke_design", "hooks_v2", "driver_v2", "patcher_v2", "config_v2", "fa2_install", "job_builder", "fa2_source_index", "token_fixtures", "prefix_token_ids",
             "stock_runner_copy", "fa2_patcher", "lm_head_shim", "fork_binary", "launcher", "cleanup_lib"]

MOCK_DOCKER = r'''#!/usr/bin/env bash
# mock docker: state via env MOCK_EXISTS MOCK_STOP_RC MOCK_STATUS MOCK_RUNNING MOCK_RM_RC; removal recorded in $MOCK_STATE/removed
echo "docker $*" >> "$MOCK_LOG"
name=lumotree-q1-ref-test
case "$1" in
  ps) if [[ "$MOCK_EXISTS" == "1" && ! -e "$MOCK_STATE/removed" ]]; then echo "$name"; fi; exit 0;;
  stop) exit "$MOCK_STOP_RC";;
  logs) echo "engine log line"; exit 0;;
  inspect) if [[ "$2" == "-f" ]]; then case "$3" in *Status*) echo "$MOCK_STATUS";; *Running*) echo "$MOCK_RUNNING";; *ExitCode*) echo 137;; esac; exit 0; fi; echo '[{"State": {"Status": "'"$MOCK_STATUS"'"}}]'; exit 0;;
  rm) if [[ "$MOCK_RM_RC" == "0" ]]; then touch "$MOCK_STATE/removed"; fi; exit "$MOCK_RM_RC";;
  *) exit 0;;
esac
'''
MOCK_NVSMI = "#!/usr/bin/env bash\necho 'pid, process_name, used_gpu_memory [MiB]'\n"


def _env(tmp, exists="1", stop_rc="0", status="exited", running="false", rm_rc="0"):
    b = os.path.join(tmp, "bin"); os.makedirs(b, exist_ok=True)
    for n, body in (("docker", MOCK_DOCKER), ("nvidia-smi", MOCK_NVSMI)):
        p = os.path.join(b, n); open(p, "w").write(body); os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    out = os.path.join(tmp, "out"); os.makedirs(out, exist_ok=True); st = os.path.join(tmp, "state"); os.makedirs(st, exist_ok=True)
    env = {**os.environ, "PATH": b + os.pathsep + os.environ["PATH"], "MOCK_LOG": os.path.join(tmp, "docker.log"), "MOCK_STATE": st, "MOCK_EXISTS": exists, "MOCK_STOP_RC": stop_rc,
           "MOCK_STATUS": status, "MOCK_RUNNING": running, "MOCK_RM_RC": rm_rc, "OUT": out, "CONTAINER": "lumotree-q1-ref-test"}
    return env, out


def _sh(env, script):
    return subprocess.run(["bash", "-c", textwrap.dedent(script)], env=env, capture_output=True, text=True, timeout=60)


PRELUDE = f'''set -u; . "{LIB}"; RECEIPT_WRITTEN=0; write_receipt() {{ echo "$1" > "$OUT/receipt_status.txt"; }};'''


def _rd(out, rel):
    p = os.path.join(out, rel); return open(p).read().strip() if os.path.exists(p) else None


# ------------------------------------------------------------------ static contract
def test_v2_1_static_contract_and_v2_bytes_preserved():
    s = open(L).read(); lib = open(LIB).read()
    assert '. "$CLEANUP_LIB"' in s and "q1_native_smoke_cleanup_v2_1.sh" in s and "stop_engine() {" not in s and 'finish_run "$DRV_RC"; exit $?' in s
    for k in GATE_KEYS:
        assert f'"{k}"' in s, k
    assert "cleanup_state" in s and "cleanup_err" in s and "_cleanup_rc=${crc}" in s
    trap = [l for l in s.splitlines() if l.startswith("trap '")][0]
    assert "rc=$?; stop_engine; crc=$?" in trap and 'if [[ $rc -eq 0 && $crc -ne 0 ]]; then exit $crc; fi' in trap and "ABORTED_OR_FAILED_rc=${rc}_cleanup_rc=${crc}" in trap
    assert "stop_engine() {" in lib and "finish_run() {" in lib and 'date -u +%FT%TZ > "$OUT/engine_stopped_utc.txt"' in lib and "return 8" in lib and "return 9" in lib
    import re
    assert not any(re.match(r"^\s*exit\b", l) or re.search(r"[;&|]\s*exit\b", l) for l in lib.splitlines() if not l.strip().startswith("#"))      # the lib never exits the caller (returns only)
    assert subprocess.run(["bash", "-n", L]).returncode == 0 and subprocess.run(["bash", "-n", LIB]).returncode == 0
    fz = json.load(open(os.path.join(CAMP, "FREEZE-Q1-FULLMODEL-DESIGN-v2.json")))
    import hashlib
    assert hashlib.sha256(open(L2, "rb").read()).hexdigest() == fz["files"]["tools/run_q1_native_smoke_v2.sh"]["sha256"]                    # v2 launcher bytes untouched (frozen)
    r = subprocess.run(["bash", L, "--dry-run"], env={**os.environ, "RUN_ID": "DRYRUN-V21"}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "(dry-run: nothing launched, nothing written)" in r.stdout and not os.path.exists(os.path.join(CAMP, "runs", "q1-native-smoke", "DRYRUN-V21"))


# ------------------------------------------------------------------ mocked-shell cleanup controls
def test_stop_engine_success_verified_stop_then_remove():
    tmp = tempfile.mkdtemp(prefix="smoke-clean-"); env, out = _env(tmp)
    r = _sh(env, PRELUDE + ' stop_engine; echo "rc=$?"')
    assert "rc=0" in r.stdout, (r.stdout, r.stderr)
    assert _rd(out, "engine_cleanup_state.txt") == "stopped_and_removed" and _rd(out, "engine_stopped_utc.txt") and _rd(out, "engine.log") == "engine log line" and os.path.exists(os.path.join(out, "engine_inspect.json"))
    log = open(env["MOCK_LOG"]).read(); assert log.index("docker stop") < log.index("docker logs") < log.index("docker rm")
    r2 = _sh(env, PRELUDE + ' stop_engine; echo "rc=$?"'); assert "rc=0" in r2.stdout and _rd(out, "engine_cleanup_state.txt") == "stopped_and_removed"   # idempotent after verified cleanup


def test_stop_engine_stop_failure_still_running_is_rc8_without_stopped_timestamp():
    tmp = tempfile.mkdtemp(prefix="smoke-clean-"); env, out = _env(tmp, stop_rc="1", status="running", running="true")
    r = _sh(env, PRELUDE + ' stop_engine; echo "rc=$?"')
    assert "rc=8" in r.stdout and "not verifiably stopped" in r.stdout
    assert _rd(out, "engine_stopped_utc.txt") is None and _rd(out, "engine_cleanup_state.txt").startswith("not_stopped stop_rc=1 status=running running=true")
    assert _rd(out, "engine.log") == "engine log line" and os.path.exists(os.path.join(out, "engine_inspect.json")) and "not verifiably stopped" in _rd(out, "engine_cleanup.err")
    assert "docker rm" not in open(env["MOCK_LOG"]).read()                                                                                   # never removes an unstopped owned container


def test_stop_engine_stop_rc0_but_inspect_still_running_is_rc8():
    tmp = tempfile.mkdtemp(prefix="smoke-clean-"); env, out = _env(tmp, stop_rc="0", status="running", running="true")                       # docker stop "succeeded" yet the container runs
    r = _sh(env, PRELUDE + ' stop_engine; echo "rc=$?"')
    assert "rc=8" in r.stdout and _rd(out, "engine_stopped_utc.txt") is None and "docker rm" not in open(env["MOCK_LOG"]).read()


def test_stop_engine_stopped_but_remove_fails_is_rc9():
    tmp = tempfile.mkdtemp(prefix="smoke-clean-"); env, out = _env(tmp, rm_rc="1")
    r = _sh(env, PRELUDE + ' stop_engine; echo "rc=$?"')
    assert "rc=9" in r.stdout and _rd(out, "engine_stopped_utc.txt") and _rd(out, "engine_cleanup_state.txt").startswith("stopped_but_not_removed") and "could not be removed" in _rd(out, "engine_cleanup.err")


def test_stop_engine_no_owned_container_is_rc0():
    tmp = tempfile.mkdtemp(prefix="smoke-clean-"); env, out = _env(tmp, exists="0")
    r = _sh(env, PRELUDE + ' stop_engine; echo "rc=$?"')
    assert "rc=0" in r.stdout and _rd(out, "engine_cleanup_state.txt") == "no_owned_container" and _rd(out, "engine_stopped_utc.txt") is None


def test_finish_run_terminal_semantics():
    for drv, stop_rc, status, running, rm_rc, want in ((0, "0", "exited", "false", "0", 0), (0, "1", "running", "true", "0", 8), (0, "0", "exited", "false", "1", 9), (2, "1", "running", "true", "0", 2), (2, "0", "exited", "false", "0", 2)):
        tmp = tempfile.mkdtemp(prefix="smoke-fin-"); env, out = _env(tmp, stop_rc=stop_rc, status=status, running=running, rm_rc=rm_rc)
        r = _sh(env, PRELUDE + f' finish_run {drv}; echo "rc=$?"')
        assert f"rc={want}" in r.stdout, (drv, stop_rc, status, rm_rc, r.stdout, r.stderr)
        assert _rd(out, "receipt_status.txt") == f"COMPLETED_driver_rc={drv}_cleanup_rc={want if drv == 0 else (8 if stop_rc == '1' else 0)}" and _rd(out, "gpu_contention_after.txt")


def test_launcher_trap_line_surfaces_cleanup_failure_without_masking_driver_failure():
    trap = [l for l in open(L).read().splitlines() if l.startswith("trap '")][0]
    for exit_code, stop_rc, status, running, want in ((0, "1", "running", "true", 8), (6, "1", "running", "true", 6), (0, "0", "exited", "false", 0), (2, "0", "exited", "false", 2)):
        tmp = tempfile.mkdtemp(prefix="smoke-trap-"); env, out = _env(tmp, stop_rc=stop_rc, status=status, running=running)
        r = _sh(env, PRELUDE + "\n" + trap + f"\nexit {exit_code}\n")
        assert r.returncode == want, (exit_code, stop_rc, r.returncode, r.stdout, r.stderr)
        assert _rd(out, "receipt_status.txt") == f"ABORTED_OR_FAILED_rc={exit_code}_cleanup_rc={8 if stop_rc == '1' else 0}"
