"""Static contract of the native-smoke launcher (tools/run_q1_native_smoke_v2.sh): gate binding keys, exact smoke scope, ordered steps, failure/receipt
behaviour, no execution paths reachable in --dry-run.  Nothing here boots an engine or touches docker."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); TOOLS = os.path.dirname(HERE); CAMP = os.path.dirname(TOOLS)
L = os.path.join(TOOLS, "run_q1_native_smoke_v2.sh")
GATE_KEYS = ["image_id", "smoke_design", "hooks_v2", "driver_v2", "patcher_v2", "config_v2", "fa2_install", "job_builder", "fa2_source_index", "token_fixtures", "prefix_token_ids",
             "stock_runner_copy", "fa2_patcher", "lm_head_shim", "fork_binary", "launcher"]
SCOPE = {"arm": "aligned_nonpacked", "process": "A", "repeats": 2, "fresh_processes": 1, "case_ids": ["calibration-short_available__c0__root-only"], "expected_requests": 2, "save_kv_bytes": True, "block": "calibration"}


def test_launcher_gate_binding_and_scope_text():
    s = open(L).read()
    assert 'g.get("gate") != "Q1-NATIVE-SMOKE"' in s and 'approved_run_id' in s and "GATE-Q1-NATIVE-SMOKE.json" in s and "Q1-NATIVE-SMOKE-DESIGN.json" in s
    for k in GATE_KEYS:
        assert f'"{k}"' in s, k
    for k, v in SCOPE.items():
        assert f'"{k}"' in s, k
    assert 'ARM=aligned_nonpacked; PROCESS=A; REPEATS=2; CASE=calibration-short_available__c0__root-only; EXPECTED_REQUESTS=2' in s
    assert "q1_reference_job.py" in s and "--smoke" in s and "q1_spec_off_engine_config_v2.py" in s and "q1_reference_driver_v2.py" in s and "prefix-plan" in s
    assert "28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857" in s and "FORK_SIZE=300123792" in s and "ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc" in s


def test_launcher_ordered_steps_failure_rules_and_receipts():
    s = open(L).read(); body = s[s.index('if [[ $DRY == 1 ]]'):]                    # executable body only
    steps = ['REFUSES LAUNCH:")', 'nvidia-smi --query-compute-apps', 'docker image inspect', 'python3 "$TOOLS/q1_reference_job.py"', 'python3 "$TOOLS/q1_spec_off_engine_config_v2.py"',
             '"LAUNCH-BINDING.json"), "x")', '"engine_launch_argv.json"), "x")', 'curl -fsS --max-time 5 "$URL/health"', 'q1_ref_fa2_install_receipt.json", "problems")', 'python3 "$TOOLS/q1_reference_driver_v2.py"',
             'write_receipt "COMPLETED_driver_rc=']
    idx = [body.index(k) for k in steps]
    assert idx == sorted(idx), list(zip(steps, idx))                                  # gate -> contention -> image -> job -> config -> binding -> argv -> boot/health -> receipts -> driver -> receipt
    assert body.index('python3 "$TOOLS/q1_reference_job.py"') < body.index('docker_run.out') < body.index('curl -fsS --max-time 5 "$URL/health"')   # engine boots only after job+config+binding
    assert "trap 'stop_engine;" in s and "ABORTED_OR_FAILED" in s and "ENGINE_NOT_HEALTHY" in s and "BOOT_RECEIPTS_NOT_CLEAN" in s and "docker stop" in s and "docker logs" in s and "docker rm" in s
    assert "retry" not in s.lower().replace("no retry", "") and "--speculative-config" not in s and "held_out" not in s and "evaluation" not in s.replace("evaluation/", "")
    assert "grep -qE '^lumotree-'" in s and ':9950 ' in s and "port 9950" in s                # any lumotree container or a bound port refuses
    assert "files_recursive_excluding_object_store" in s and "engine_healthy_utc" in s and "driver_exit" in s and "RUN-RECEIPT.json" in s


def test_launcher_syntax_and_dry_run_touch_nothing():
    assert subprocess.run(["bash", "-n", L]).returncode == 0
    before = sorted(os.listdir(os.path.join(CAMP, "runs")))
    r = subprocess.run(["bash", L, "--dry-run"], env={**os.environ, "RUN_ID": "DRYRUN-STATIC"}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "(dry-run: nothing launched, nothing written)" in r.stdout and "CASE=calibration-short_available__c0__root-only REPEATS=2 EXPECTED_REQUESTS=2" in r.stdout
    assert sorted(os.listdir(os.path.join(CAMP, "runs"))) == before and not os.path.exists(os.path.join(CAMP, "runs", "q1-native-smoke"))
    r2 = subprocess.run(["bash", L], env={**os.environ, "RUN_ID": "UNSET"}, capture_output=True, text=True, timeout=120)
    assert r2.returncode == 3 and "RUN_ID must be set" in r2.stdout and not os.path.exists(os.path.join(CAMP, "runs", "q1-native-smoke"))


def test_prefix_bytes_and_smoke_case_are_the_frozen_fixture():
    fx = json.load(open(os.path.join(CAMP, "fullmodel", "fixtures", "token-fixtures.v1.json")))
    pre = [p for p in fx["prefixes"] if p["prefix_id"] == "calibration-short_available"][0]; case = [c for c in fx["cases"] if c["case_id"] == SCOPE["case_ids"][0]][0]
    p = os.path.join(CAMP, "prefix-plan", pre["token_ids_u32le"])
    import hashlib
    assert os.path.getsize(p) == 4 * pre["prefix_len"] == 4 * 13487 and hashlib.sha256(open(p, "rb").read()).hexdigest() == pre["token_ids_sha256"]
    assert case["chain_tokens"] == [11352, 25559] and case["chain_positions"] == [13487, 13488] and case["held_out"] is False and case["prefix_id"] == pre["prefix_id"]
