"""Native-smoke launcher v2.3 (= v2.2 + FINALIZED marker written BEFORE the sealed receipt; receipt-write failure -> non-zero): static contract + ACTUAL-SHELL
mocked-docker controls, including the two COMBINED controls the reviewer required, now also asserting the marker is inside the receipt-time snapshot
(engine.finalized not null) and that NOTHING is written after the receipt:
  (A) a failed `docker ps -a` during cleanup is refused (rc 8, state docker_query_failed), never read as 'no owned container';
  (B) the ACTUAL finalize() composed with the ACTUAL launcher EXIT trap on a failed stop: exactly ONE docker stop call, and the receipt / cleanup
      state / cleanup.err bytes at receipt time are IDENTICAL after process EXIT (no second cleanup, no rewrite), for driver rc 0 and rc 2, plus an
      unexpected abort before finalization (finalized exactly once by the trap).  No GPU, no real docker."""
import hashlib, json, os, stat, subprocess, sys, tempfile, textwrap

HERE = os.path.dirname(os.path.abspath(__file__)); TOOLS = os.path.dirname(HERE); CAMP = os.path.dirname(TOOLS)
L = os.path.join(TOOLS, "run_q1_native_smoke_v2_3.sh"); LIB = os.path.join(TOOLS, "q1_native_smoke_cleanup_v2_3.sh")
L22, LIB22 = os.path.join(TOOLS, "run_q1_native_smoke_v2_2.sh"), os.path.join(TOOLS, "q1_native_smoke_cleanup_v2_2.sh")
L21, LIB21, L2 = os.path.join(TOOLS, "run_q1_native_smoke_v2_1.sh"), os.path.join(TOOLS, "q1_native_smoke_cleanup_v2_1.sh"), os.path.join(TOOLS, "run_q1_native_smoke_v2.sh")

MOCK_DOCKER = r'''#!/usr/bin/env bash
echo "docker $*" >> "$MOCK_LOG"
name=lumotree-q1-ref-test
case "$1" in
  ps) if [[ "${MOCK_PS_RC:-0}" != "0" ]]; then echo "Cannot connect to the Docker daemon" >&2; exit "$MOCK_PS_RC"; fi; if [[ "$MOCK_EXISTS" == "1" && ! -e "$MOCK_STATE/removed" ]]; then echo "$name"; fi; exit 0;;
  stop) exit "$MOCK_STOP_RC";;
  logs) echo "engine log line"; exit 0;;
  inspect) if [[ "$2" == "-f" ]]; then case "$3" in *Status*) echo "$MOCK_STATUS";; *Running*) echo "$MOCK_RUNNING";; *ExitCode*) echo 137;; esac; exit 0; fi; echo '[{"State": {"Status": "'"$MOCK_STATUS"'"}}]'; exit 0;;
  rm) if [[ "$MOCK_RM_RC" == "0" ]]; then touch "$MOCK_STATE/removed"; fi; exit "$MOCK_RM_RC";;
  *) exit 0;;
esac
'''
MOCK_NVSMI = "#!/usr/bin/env bash\necho 'pid, process_name, used_gpu_memory [MiB]'\n"


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _env(tmp, exists="1", stop_rc="0", status="exited", running="false", rm_rc="0", ps_rc="0"):
    b = os.path.join(tmp, "bin"); os.makedirs(b, exist_ok=True)
    for n, body in (("docker", MOCK_DOCKER), ("nvidia-smi", MOCK_NVSMI)):
        p = os.path.join(b, n); open(p, "w").write(body); os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    out = os.path.join(tmp, "out"); os.makedirs(out, exist_ok=True); st = os.path.join(tmp, "state"); os.makedirs(st, exist_ok=True)
    return {**os.environ, "PATH": b + os.pathsep + os.environ["PATH"], "MOCK_LOG": os.path.join(tmp, "docker.log"), "MOCK_STATE": st, "MOCK_EXISTS": exists, "MOCK_STOP_RC": stop_rc,
            "MOCK_STATUS": status, "MOCK_RUNNING": running, "MOCK_RM_RC": rm_rc, "MOCK_PS_RC": ps_rc, "OUT": out, "CONTAINER": "lumotree-q1-ref-test"}, out


def _sh(env, script): return subprocess.run(["bash", "-c", textwrap.dedent(script)], env=env, capture_output=True, text=True, timeout=60)


def _rd(out, rel):
    p = os.path.join(out, rel); return open(p).read().strip() if os.path.exists(p) else None


def _size(out, rel):
    p = os.path.join(out, rel); return os.path.getsize(p) if os.path.exists(p) else -1


# the launcher-side write_receipt is replaced by a stub that (1) writes the receipt file, (2) counts calls, (3) snapshots the sizes/hashes of the sealed files AT RECEIPT TIME
PRELUDE = f'''set -u; . "{LIB}"; RECEIPT_WRITTEN=0
write_receipt() {{ if [[ "${{MOCK_RECEIPT_RC:-0}}" != "0" ]]; then echo "receipt write failed" >&2; return "$MOCK_RECEIPT_RC"; fi
  echo "$1" > "$OUT/receipt_status.txt"; echo "$1" >> "$OUT/receipt_calls.log";
  python3 - "$OUT" <<'PY'
import os, sys, hashlib, json
out = sys.argv[1]; snap = {{}}
for rel in ("receipt_status.txt", "engine_cleanup_state.txt", "engine_cleanup.err", "engine_stopped_utc.txt", "engine.log", "FINALIZED.txt", "gpu_contention_after.txt"):
    p = os.path.join(out, rel); snap[rel] = [os.path.getsize(p), hashlib.sha256(open(p, "rb").read()).hexdigest()] if os.path.exists(p) else None
snap["engine.finalized_at_receipt_time"] = open(os.path.join(out, "FINALIZED.txt")).read().strip() if os.path.exists(os.path.join(out, "FINALIZED.txt")) else None
json.dump(snap, open(os.path.join(out, "receipt_time_snapshot.json"), "w"))
PY
}}
'''


def _trap_line():
    return [l for l in open(L).read().splitlines() if l.startswith("trap '")][0]


def _stops(env): return open(env["MOCK_LOG"]).read().count("docker stop ")


def _after_exit_equals_receipt_time(out):
    snap = json.load(open(os.path.join(out, "receipt_time_snapshot.json"))); fin = snap.pop("engine.finalized_at_receipt_time")
    now = {rel: ([_size(out, rel), sha(os.path.join(out, rel))] if os.path.exists(os.path.join(out, rel)) else None) for rel in snap}
    listing_after = sorted(f for f in os.listdir(out) if f not in ("receipt_time_snapshot.json", "receipt_calls.log"))
    return snap == now and fin is not None and fin == _rd(out, "FINALIZED.txt") == _rd(out, "receipt_status.txt"), snap, now, fin, listing_after


# ------------------------------------------------------------------ static contract + prior drafts preserved
def test_v2_2_static_contract_and_prior_drafts_preserved():
    s = open(L).read(); lib = open(LIB).read(); body = s[s.index('if [[ $DRY == 1 ]]'):]
    assert body.index('REFUSES LAUNCH:")') < body.index('. "$CLEANUP_LIB"') < body.index("trap 'rc=$?; on_exit_trap $rc") < body.index('python3 "$TOOLS/q1_reference_job.py"')   # sourced only after the gate validated its hash
    assert '"cleanup_lib": sha(cleanup_lib)' in s and "q1_native_smoke_cleanup_v2_3.sh" in s and "stop_engine;" not in s and "stop_engine\n" not in s and s.count('finalize "') == 3
    wr = s[s.index("write_receipt() {"):s.index("trap 'rc=$?")]; assert "os.replace(tmp, os.path.join(out, \"RUN-RECEIPT.json\"))" in wr and wr.rstrip().endswith("return $?\n}") is False and "  return $?" in wr
    li = lib.index("finalize() {"); assert lib.index('FINALIZED.txt"', li) < lib.index("write_receipt ", li) and "return 10" in lib                                  # marker BEFORE the seal; receipt failure -> rc 10
    assert "RUNNING_NAMES=$(docker ps --format '{{.Names}}') ||" in s and "ALL_NAMES=$(docker ps -a --format '{{.Names}}') ||" in s and "query failure is not an empty result" in s
    assert "owned_container_state() {" in lib and 'echo "query_failed"; return 1' in lib and "docker_query_failed" in lib and "FINALIZED=1" in lib and "on_exit_trap() {" in lib
    import re
    assert not any(re.match(r"^\s*exit\b", l) or re.search(r"[;&|]\s*exit\b", l) for l in lib.splitlines() if not l.strip().startswith("#"))
    assert subprocess.run(["bash", "-n", L]).returncode == 0 and subprocess.run(["bash", "-n", LIB]).returncode == 0
    v2 = json.load(open(os.path.join(CAMP, "FREEZE-Q1-FULLMODEL-DESIGN-v2.json")))["files"]; v21 = json.load(open(os.path.join(CAMP, "FREEZE-Q1-FULLMODEL-DESIGN-v2.1.json")))["files"]; v22 = json.load(open(os.path.join(CAMP, "FREEZE-Q1-FULLMODEL-DESIGN-v2.2.json")))["files"]
    assert sha(L2) == v2["tools/run_q1_native_smoke_v2.sh"]["sha256"] and sha(L21) == v21["tools/run_q1_native_smoke_v2_1.sh"]["sha256"] and sha(LIB21) == v21["tools/q1_native_smoke_cleanup_v2_1.sh"]["sha256"]
    assert sha(L22) == v22["tools/run_q1_native_smoke_v2_2.sh"]["sha256"] and sha(LIB22) == v22["tools/q1_native_smoke_cleanup_v2_2.sh"]["sha256"]
    r = subprocess.run(["bash", L, "--dry-run"], env={**os.environ, "RUN_ID": "DRYRUN-V23"}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "(dry-run: nothing launched, nothing written)" in r.stdout and not os.path.exists(os.path.join(CAMP, "runs", "q1-native-smoke", "DRYRUN-V23"))


# ------------------------------------------------------------------ isolated controls (carried from v2.1) against the v2.2 library
def test_isolated_controls_success_stopfail_running_rmfail_absent():
    for kw, want_rc, want_state in (({}, 0, "stopped_and_removed"), ({"stop_rc": "1", "status": "running", "running": "true"}, 8, "not_stopped stop_rc=1"), ({"status": "running", "running": "true"}, 8, "not_stopped stop_rc=0"),
                                    ({"rm_rc": "1"}, 9, "stopped_but_not_removed"), ({"exists": "0"}, 0, "no_owned_container")):
        tmp = tempfile.mkdtemp(prefix="smoke22-iso-"); env, out = _env(tmp, **kw)
        r = _sh(env, PRELUDE + ' stop_engine; echo "rc=$?"')
        assert f"rc={want_rc}" in r.stdout and (_rd(out, "engine_cleanup_state.txt") or "").startswith(want_state), (kw, r.stdout, r.stderr, _rd(out, "engine_cleanup_state.txt"))
        assert (_rd(out, "engine_stopped_utc.txt") is not None) == (want_rc in (0, 9) and kw.get("exists", "1") == "1"), kw


# ------------------------------------------------------------------ COMBINED CONTROL A: docker enumeration failure is refused, never 'absent'
def test_combined_A_docker_ps_failure_refuses_and_finalize_exits_nonzero():
    tmp = tempfile.mkdtemp(prefix="smoke22-A-"); env, out = _env(tmp, ps_rc="1")
    r = _sh(env, PRELUDE + ' stop_engine; echo "rc=$?"')
    assert "rc=8" in r.stdout and _rd(out, "engine_cleanup_state.txt") == "not_verified docker_query_failed" and "enumeration failed" in _rd(out, "engine_cleanup.err")
    assert _rd(out, "engine_stopped_utc.txt") is None and "docker stop" not in open(env["MOCK_LOG"]).read() and "docker rm" not in open(env["MOCK_LOG"]).read()
    tmp2 = tempfile.mkdtemp(prefix="smoke22-A2-"); env2, out2 = _env(tmp2, ps_rc="1")                   # composed: actual finalize + actual trap, driver rc 0
    r2 = _sh(env2, PRELUDE + "\n" + _trap_line() + '\nfinalize "COMPLETED_driver_rc=0" 0; exit $?\n')
    assert r2.returncode == 8 and _rd(out2, "receipt_status.txt") == "COMPLETED_driver_rc=0_cleanup_rc=8" and _rd(out2, "engine_cleanup_state.txt") == "not_verified docker_query_failed"
    assert open(os.path.join(out2, "receipt_calls.log")).read().count("\n") == 1 and _after_exit_equals_receipt_time(out2)[0]


# ------------------------------------------------------------------ COMBINED CONTROL B: actual finalize + actual EXIT trap on a FAILED stop
def test_combined_B_finalize_plus_exit_trap_on_failed_stop_one_cleanup_files_unchanged():
    for drv, want in ((2, 2), (0, 8)):
        tmp = tempfile.mkdtemp(prefix="smoke22-B-"); env, out = _env(tmp, stop_rc="1", status="running", running="true")
        r = _sh(env, PRELUDE + "\n" + _trap_line() + f'\nfinalize "COMPLETED_driver_rc={drv}" {drv}; exit $?\n')
        assert r.returncode == want, (drv, r.returncode, r.stdout, r.stderr)
        assert _stops(env) == 1, open(env["MOCK_LOG"]).read()                                                       # exactly ONE stop call (v2.1 made two)
        assert open(os.path.join(out, "receipt_calls.log")).read().count("\n") == 1                                  # exactly ONE receipt write
        same, snap, now, fin, listing = _after_exit_equals_receipt_time(out); assert same, (snap, now, fin, listing)                           # cleanup.err / state / receipt bytes identical after EXIT (v2.1: 150 -> 300 bytes)
        assert _rd(out, "receipt_status.txt") == f"COMPLETED_driver_rc={drv}_cleanup_rc=8" and _rd(out, "FINALIZED.txt") == f"COMPLETED_driver_rc={drv}_cleanup_rc=8" and _rd(out, "engine_stopped_utc.txt") is None
        assert _rd(out, "engine.log") == "engine log line" and os.path.exists(os.path.join(out, "engine_inspect.json"))
    # success path composed: one stop, one rm, one receipt, rc 0
    tmp = tempfile.mkdtemp(prefix="smoke22-B-ok-"); env, out = _env(tmp)
    r = _sh(env, PRELUDE + "\n" + _trap_line() + '\nfinalize "COMPLETED_driver_rc=0" 0; exit $?\n')
    assert r.returncode == 0 and _stops(env) == 1 and open(env["MOCK_LOG"]).read().count("docker rm ") == 1 and open(os.path.join(out, "receipt_calls.log")).read().count("\n") == 1 and _after_exit_equals_receipt_time(out)[0]
    assert _rd(out, "receipt_status.txt") == "COMPLETED_driver_rc=0_cleanup_rc=0" and _rd(out, "engine_cleanup_state.txt") == "stopped_and_removed"


def test_unexpected_abort_before_finalization_is_finalized_exactly_once_by_the_trap():
    for exit_code, stop_rc, status, running, want in ((5, "1", "running", "true", 5), (0, "1", "running", "true", 8), (0, "0", "exited", "false", 0), (6, "0", "exited", "false", 6)):
        tmp = tempfile.mkdtemp(prefix="smoke22-abort-"); env, out = _env(tmp, stop_rc=stop_rc, status=status, running=running)
        r = _sh(env, PRELUDE + "\n" + _trap_line() + f"\nexit {exit_code}\n")
        assert r.returncode == want and _stops(env) == 1 and open(os.path.join(out, "receipt_calls.log")).read().count("\n") == 1, (exit_code, stop_rc, r.returncode, open(env["MOCK_LOG"]).read())
        assert _rd(out, "receipt_status.txt") == f"ABORTED_OR_FAILED_rc={exit_code}_cleanup_rc={8 if stop_rc == '1' else 0}" and _after_exit_equals_receipt_time(out)[0]


def test_receipt_write_failure_is_nonzero_and_marker_precedes_seal():
    for drv, want in ((0, 10), (2, 2)):
        tmp = tempfile.mkdtemp(prefix="smoke23-rcpt-"); env, out = _env(tmp); env["MOCK_RECEIPT_RC"] = "1"
        r = _sh(env, PRELUDE + "\n" + _trap_line() + f'\nfinalize "COMPLETED_driver_rc={drv}" {drv}; exit $?\n')
        assert r.returncode == want and "RECEIPT WRITE FAILED" in r.stderr and _rd(out, "FINALIZED.txt") == f"COMPLETED_driver_rc={drv}_cleanup_rc=0" and _rd(out, "receipt_status.txt") is None, (drv, r.returncode, r.stderr)
        assert _stops(env) == 1                                                                     # finalized once; the trap did not re-run cleanup after the failed receipt
    tmp = tempfile.mkdtemp(prefix="smoke23-order-"); env, out = _env(tmp, stop_rc="1", status="running", running="true")
    r = _sh(env, PRELUDE + "\n" + _trap_line() + '\nfinalize "COMPLETED_driver_rc=0" 0; exit $?\n')
    snap = json.load(open(os.path.join(out, "receipt_time_snapshot.json")))
    assert r.returncode == 8 and snap["engine.finalized_at_receipt_time"] == "COMPLETED_driver_rc=0_cleanup_rc=8" and snap["FINALIZED.txt"] is not None and snap["gpu_contention_after.txt"] is not None   # marker + contention file existed at seal time
    same, *_ = _after_exit_equals_receipt_time(out); assert same
