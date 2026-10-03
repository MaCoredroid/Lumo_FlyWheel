"""Focused CPU regression for the Q1.2b HELD-OUT reduction repair v2.2.1 (parent instruction 2026-09-28T01:42:32Z).

Why: the frozen v2.2 runner's HELPER_FILES still names q1_component_runner_v2_1.py, so its REAL helper_hashes() emits the redundant
self-helper entry {q1_component_runner_v2_1.py: frozen v2.1 bytes} and no q1_component_runner_v2_2.py key; the v2.2 reducer refused the
preserved held-out run q12b-heldout-20260928T013539Z with exactly two malformed findings.  The earlier synthetic suites built the helper
map by hand (keyed v2_2) and bypassed the real producer.  Here every process attestation carries the REAL R22.helper_hashes() output.

Proven: (1) the real producer has the defect; (2) reducer v2.2 reproduces the exact original refusal; (3) reducer v2.2.1 reduces the same
preserved evidence to a verdict in a SEPARATE directory under a parent authorization, without touching the run; (4) the default binding
constants equal the preserved real run; (5) wrong/partial/extra helper entries, wrong runner identities on attestation / records / launch
binding, wrong run id, changed original evidence, missing or mismatched authorization, and an output inside the run all refuse;
(6) numerical functions are byte-identical to v2.2; (7) launcher static contract + dry run.
"""
import argparse, copy, difflib, glob, hashlib, inspect, json, os, shutil, subprocess, sys, tempfile

os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pytest
import torch
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__)); TOOLS = os.path.dirname(HERE); CAMP = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS); sys.path.insert(0, HERE)
import q1_2b_fixtures as FX  # noqa: E402
import q1_component_runner_v2_1 as R21  # noqa: E402
import q1_component_runner_v2_2 as R22  # noqa: E402
import q1_2b_reduce_v2_2 as RD22  # noqa: E402
import q1_2b_reduce_v2_2_1 as RD221  # noqa: E402
import q1_policy_evaluator_v2_1 as PE  # noqa: E402
import q1_policy_build_v2_1 as PB  # noqa: E402
from test_q1_2b_v2_1_pipeline import StubBackend, _edit, _rewrite_result, _rewrite_record  # noqa: E402

RUNNER21, RUNNER22 = os.path.join(TOOLS, "q1_component_runner_v2_1.py"), os.path.join(TOOLS, "q1_component_runner_v2_2.py")
LAUNCHER22 = os.path.join(TOOLS, "run_q1_2b_component_v2_2.sh"); REDUCER22, REDUCER221 = os.path.join(TOOLS, "q1_2b_reduce_v2_2.py"), os.path.join(TOOLS, "q1_2b_reduce_v2_2_1.py")
REPAIR_LAUNCHER = os.path.join(TOOLS, "run_q1_2b_reduction_repair_v2_2_1.sh")
POL21 = os.path.join(CAMP, "policy", "q1_component_numerical_policy.v2.1.json"); POLICY2 = os.path.join(CAMP, "policy", "q1_component_numerical_policy.v2.json")
CONTRACT2 = os.path.join(CAMP, "..", "..", "p0", "monitor", "review-response-20260927", "Q1-PAIRED-NUMERICAL-CONTRACT-v2.json")
REAL_RUN = os.path.join(CAMP, "runs", "q1.2b-heldout", "q12b-heldout-20260928T013539Z")
POWER_RUN = "q12b-calibration-retry-20260928T010045Z"
pytestmark = pytest.mark.skipif(not os.path.exists(POL21), reason="frozen policy v2.1 not present")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


FROZEN_V21 = sha(RUNNER21)
REAL_HELPERS = R22.helper_hashes()          # the REAL producer inside the frozen v2.2 runner


# ------------------------------------------------------------------ (1) the real producer
def test_real_helper_hashes_expose_the_redundant_self_entry():
    assert FROZEN_V21 == "b7241e8b925a9b5d9e921673d84f84b701f9b1404ed71d727bc5ade37d15cb65" == json.load(open(os.path.join(CAMP, "FREEZE-Q1_2B-INIT-v1.1.json")))["files"]["tools/q1_component_runner_v2_1.py"]
    assert R22.HELPER_FILES[0] == "q1_component_runner_v2_1.py" and "q1_component_runner_v2_2.py" not in R22.HELPER_FILES
    assert REAL_HELPERS["q1_component_runner_v2_1.py"] == FROZEN_V21 and "q1_component_runner_v2_2.py" not in REAL_HELPERS
    assert set(REAL_HELPERS) == (set(RD22.HELPER_KEYS) - {"q1_component_runner_v2_2.py"}) | {"q1_component_runner_v2_1.py"}
    assert set(R21.helper_hashes()) == set(REAL_HELPERS) and R21.helper_hashes()["q1_component_runner_v2_1.py"] == FROZEN_V21   # v2.1 legitimately names itself
    assert RD221.DEFECTIVE_SELF_HELPER == {"key_present_instead": "q1_component_runner_v2_1.py", "value_required": FROZEN_V21, "key_absent": "q1_component_runner_v2_2.py"}
    assert RD221.BOUND_RUNNER_SHA256 == sha(RUNNER22) == RD22.BOUND_RUNNER_SHA256 and RD221.BOUND_LAUNCHER_SHA256 == sha(LAUNCHER22)


# ------------------------------------------------------------------ synthetic held-out stage whose attestation carries the REAL helper map
def make_stage(tmp, layers=2):
    fxroot = os.path.join(tmp, "fixtures"); FX.generate(fxroot, blocks=("calibration", "evaluation"), layers=layers, fixtures_per_block=2)
    fm_p = os.path.join(fxroot, "manifest.json"); obs_p = os.path.join(fxroot, "expected_observations.json"); pol_p = os.path.join(tmp, "policy.v2.1.json")
    pol = PB.build(POLICY2, sha(POLICY2), fm_p, obs_p); json.dump(pol, open(pol_p, "w"), indent=1)
    fm = json.load(open(fm_p)); ho = sorted(e["fixture_id"] for e in fm["fixtures"] if e["block"] == "evaluation")
    run = os.path.join(tmp, "run", "q12b-heldout-SYNTH"); os.makedirs(os.path.join(run, "tensors")); os.makedirs(os.path.join(run, "fixtures"))
    for e in fm["fixtures"]:
        if e["block"] == "evaluation":
            dst = os.path.join(run, "fixtures", e["path"]); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(os.path.join(fxroot, e["path"]), dst)
    helpers = {"q1_component_runner_v2_2.py": sha(RUNNER22), **{k: v for k, v in REAL_HELPERS.items() if k != "q1_component_runner_v2_1.py"}}   # launcher v2.2 map (real bytes)
    expect = {"image_id": "img", "fixture_manifest": sha(fm_p), "expected_observations": sha(obs_p), "policy": sha(pol_p), "policy_base_v2": sha(POLICY2), "contract_v2": sha(CONTRACT2),
              "runner": sha(RUNNER22), "reducer": sha(REDUCER22), "launcher": sha(LAUNCHER22), "fixtures_tool": helpers["q1_2b_fixtures.py"], "oracle": helpers["q1_oracle.py"],
              "evaluator_v1": helpers["q1_policy_evaluator.py"], "evaluator_v2": helpers["q1_policy_evaluator_v2.py"], "evaluator_v2_1": helpers["q1_policy_evaluator_v2_1.py"],
              "bf16_ulp": helpers["q1_bf16_ulp.py"], "topology": helpers["scripts/fr13_fixed32_topology.py"], "kernel": helpers["src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"],
              "native_module": "n" * 64, "verification_V2": "v" * 64, "tree_conv_fused": helpers["src/lumo_flywheel_serving/fr13_tree_conv_fused.py"]}
    scope = {"repeats": 2, "block": "evaluation", "negatives": 0, "processes": ["A", "B"], "held_out_fixture_ids": ho, "calibration_power_run_id": POWER_RUN}
    gate = {"gate": "Q1.2b", "approved": True, "approved_run_id": "q12b-heldout-SYNTH", "reviewed_hashes": dict(expect), "reviewed_scope": copy.deepcopy(scope)}
    for name, src in (("GATE-Q1.2b.snapshot.json", None), ("fixture_manifest.snapshot.json", fm_p), ("expected_observations.snapshot.json", obs_p), ("policy.snapshot.json", pol_p),
                      ("policy_base_v2.snapshot.json", POLICY2), ("contract_v2.snapshot.json", CONTRACT2)):
        if src is None:
            json.dump(gate, open(os.path.join(run, name), "w"), indent=1)
        else:
            shutil.copy2(src, os.path.join(run, name))
    expect["gate"] = sha(os.path.join(run, "GATE-Q1.2b.snapshot.json"))
    lb = {"schema": "lumo.review-response.q1-2b-launch-binding.v2", "expect": expect, "helpers": helpers, "scope": copy.deepcopy(scope), "run_id": "q12b-heldout-SYNTH",
          "runtime": {"torch": torch.__version__, "triton": "stub", "gpu_name": "stub", "image_id": "img"},
          "snapshots": {"gate": "GATE-Q1.2b.snapshot.json", "fixture_manifest": "fixture_manifest.snapshot.json", "expected_observations": "expected_observations.snapshot.json",
                        "policy": "policy.snapshot.json", "policy_base_v2": "policy_base_v2.snapshot.json", "contract_v2": "contract_v2.snapshot.json"}}
    json.dump(lb, open(os.path.join(run, "LAUNCH-BINDING.json"), "w"), indent=1)
    return {"tmp": tmp, "fxroot": fxroot, "fm_p": fm_p, "obs_p": obs_p, "pol_p": pol_p, "run": run, "gate": gate, "expect": expect, "helpers": helpers, "fm": fm, "obs": json.load(open(obs_p)), "ho": ho}


def run_process(st, tag, backend, utc_offset):
    approval = R22.gate_approval(os.path.join(st["run"], "GATE-Q1.2b.snapshot.json"))
    pol = PE.load_policy_verified_v2_1(st["pol_p"], sha(st["pol_p"]), approval); assert pol["_approved_ok"]
    bound, pr = PE.bind_expected_manifest(pol, st["obs"], sha(st["obs_p"]), "evaluation"); assert not pr, pr
    args = argparse.Namespace(process_tag=tag, repeats=2, negatives=0, limit=0, init_only=False, block="evaluation", out=os.path.join(st["run"], f"proc{tag}"),
                              tensor_store=os.path.join(st["run"], "tensors"), fixtures_root=st["fxroot"], image_id="img", expected_policy_sha256=sha(st["pol_p"]))
    bp, facts = R22.block_authorization(st["gate"], pol, st["fm"], args); assert bp == [], bp
    os.makedirs(args.out, exist_ok=True)
    att = {"utc": f"2026-09-28T03:00:0{utc_offset}+00:00", "process_tag": tag, "pid": 1, "hostname_in_container": f"c{utc_offset:011d}", "container_name_env": f"lumotree-review-q12b-{tag}",
           "policy_sha256": pol["_loaded_sha256"], "kernel_module_sha256": REAL_HELPERS["src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"], "topology_sha256": REAL_HELPERS["scripts/fr13_fixed32_topology.py"],
           "native_module_sha256": "n" * 64, "native_function_source_sha256": "f" * 64, "runner_sha256": sha(RUNNER22), "oracle_sha256": REAL_HELPERS["q1_oracle.py"], "fixtures_tool_sha256": REAL_HELPERS["q1_2b_fixtures.py"],
           "image_id_arg": "img", "torch": torch.__version__, "triton": "stub", "gpu_name": "stub", "vllm": "stub-vllm", "torch_cuda": None, "expected_observations_sha256": sha(st["obs_p"]),
           "fixture_manifest_sha256": sha(st["fm_p"]), "block": "evaluation", "repeats": 2, "helper_hashes": dict(REAL_HELPERS), "block_authorization": facts}   # REAL (defective) helper map
    entries = [e for e in st["fm"]["fixtures"] if e["block"] == "evaluation"]
    return R22.run_process(backend, args, entries, bound, pol, att, True)


def original_from_run(run, findings):
    return {"run_id": os.path.basename(run), "block": "evaluation", "gate_file_sha256": sha(os.path.join(run, "GATE-Q1.2b.snapshot.json")),
            "run_receipt_sha256": sha(os.path.join(run, "RUN-RECEIPT.json")), "original_summary_sha256": sha(os.path.join(run, "summary.v2.json")), "launch_binding_sha256": sha(os.path.join(run, "LAUNCH-BINDING.json")),
            "gate_snapshot_sha256": sha(os.path.join(run, "GATE-Q1.2b.snapshot.json")), "procA_result_sha256": sha(os.path.join(run, "procA", "result.json")), "procB_result_sha256": sha(os.path.join(run, "procB", "result.json")),
            "procA_inventory_sha256": sha(os.path.join(run, "procA", "raw_inventory.jsonl")), "procB_inventory_sha256": sha(os.path.join(run, "procB", "raw_inventory.jsonl")),
            "original_reducer_sha256": sha(REDUCER22), "original_reducer_exit": 2, "original_findings": list(findings), "procA_exit": "exit=0", "procB_exit": "exit=0", "receipt_status": "COMPLETED_reduce_rc=2"}


def auth_for(original, reduction_id="RED-1", **over):
    a = {"schema": "lumo.parent.q12b-heldout-reduction-authorization.v1", "approved": True, "authorizes": "CPU_REDUCTION_ONLY", "run_id": original["run_id"], "reduction_id": reduction_id,
         "repaired_reducer_sha256": sha(REDUCER221), "original_evidence": {k: original[k] for k in ("run_receipt_sha256", "original_summary_sha256", "launch_binding_sha256", "gate_snapshot_sha256", "procA_result_sha256", "procB_result_sha256")},
         "approved_utc": "2026-09-28T02:00:00+00:00", "owner": "test"}
    a.update(over); return a


@pytest.fixture(scope="module")
def stage():
    tmp = tempfile.mkdtemp(prefix="q12b-v221-")
    st = make_stage(tmp)
    st["resA"] = run_process(st, "A", StubBackend(2), 1); st["resB"] = run_process(st, "B", StubBackend(2), 2)
    s22, code22 = RD22.reduce_run(st["run"], "evaluation", "none")                        # the ORIGINAL reducer, as the launcher ran it
    json.dump(s22, open(os.path.join(st["run"], "summary.v2.json"), "x"), indent=1, default=str)
    json.dump({"schema": "lumo.review-response.q1-2b-run-receipt.v2", "run": st["run"], "status": f"COMPLETED_reduce_rc={code22}", "files_recursive_excluding_tensor_store": {},
               "tensor_store": {"objects": 0, "bytes": 0}, "proc_exit": {"A": "exit=0", "B": "exit=0"}, "reduce_exit": f"exit={code22}"}, open(os.path.join(st["run"], "RUN-RECEIPT.json"), "x"), indent=1)
    st["s22"], st["code22"] = s22, code22
    st["original"] = original_from_run(st["run"], s22["findings"]["malformed"]); st["auth"] = auth_for(st["original"])
    yield st
    shutil.rmtree(tmp, ignore_errors=True)


def _snapshot(run):
    return {os.path.relpath(os.path.join(r, f), run): sha(os.path.join(r, f)) for r, _, fs in os.walk(run) for f in fs if not os.path.relpath(os.path.join(r, f), run).startswith("tensors/")}


# ------------------------------------------------------------------ (2) reproduction, (3) repair
def test_v2_2_reducer_reproduces_the_original_refusal_on_the_real_helper_map(stage):
    assert stage["code22"] == 2
    assert stage["s22"]["findings"]["malformed"] == ["procA: loaded helper q1_component_runner_v2_2.py hash != launch-bound helper map", "procB: loaded helper q1_component_runner_v2_2.py hash != launch-bound helper map"]
    assert stage["s22"]["findings"]["malformed"] == RD221.ORIGINAL_RUN["original_findings"]                     # identical text to the preserved real run
    for t in "AB":
        att = json.load(open(os.path.join(stage["run"], f"proc{t}", "result.json")))["attestation"]
        assert att["helper_hashes"] == REAL_HELPERS and att["runner_sha256"] == sha(RUNNER22)


def test_v2_2_1_reduces_the_preserved_evidence_in_a_separate_directory(stage):
    before = _snapshot(stage["run"]); out = os.path.join(stage["tmp"], "reductions", "RED-1"); os.makedirs(out)
    s, code = RD221.reduce_run(stage["run"], "evaluation", "all", out_dir=out, authorization=stage["auth"], original=stage["original"])
    assert code == 0, (code, s["findings"])
    assert s["schema"] == "lumo.review-response.q1-2b-summary.v2.2.1" and s["aggregate"]["A"]["aggregate"] == "PASS" and s["aggregate"]["B"]["aggregate"] == "PASS" and s["cases_expected"] == 240
    assert s["recomputed_records"] == 480 and s["negative_power"] == {"A": {}, "B": {}} and s["negative_power_note"]["power_source_calibration_run_id"] == POWER_RUN
    b = s["reduction"]; assert b["original_reducer_exit"] == 2 and b["original_findings"] == stage["s22"]["findings"]["malformed"] and b["binding"]["repaired_reducer_sha256"] == sha(REDUCER221)
    assert b["binding"]["authorization"]["reduction_id"] == "RED-1" and "did NOT pass" in b["claim_boundary"]
    assert b["original_launch_binding_names_reducer"] == sha(REDUCER22) == json.load(open(os.path.join(stage["run"], "LAUNCH-BINDING.json")))["expect"]["reducer"] != sha(REDUCER221)   # intended-original positive
    assert b["binding"]["authorization"]["repaired_reducer_sha256"] == sha(REDUCER221)                                                                                         # new bytes bound by the authorization only
    assert _snapshot(stage["run"]) == before                                                                    # the original run is untouched (no new/changed files)
    # every metrics record's execution.runner_sha256 is the bound v2.2 runner
    recs = glob.glob(os.path.join(stage["run"], "procA", "raw", "*", "*.json")); assert recs
    assert all(json.load(open(p)).get("execution", {}).get("runner_sha256") == sha(RUNNER22) for p in recs)


def test_v2_2_1_default_binding_refuses_a_different_run(stage):
    out = os.path.join(stage["tmp"], "reductions", "RED-default"); os.makedirs(out)
    s, code = RD221.reduce_run(stage["run"], "evaluation", "none", out_dir=out, authorization=stage["auth"])            # original=None -> the REAL preserved run binding
    assert code == 2 and any("run id" in p for p in s["findings"]["malformed"])


# ------------------------------------------------------------------ (4) the default constants ARE the preserved real run
@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_RUN, "RUN-RECEIPT.json")), reason="preserved held-out run not present on this host")
def test_default_original_binding_matches_the_preserved_real_run():
    o = RD221.ORIGINAL_RUN
    for rel, key in (("RUN-RECEIPT.json", "run_receipt_sha256"), ("summary.v2.json", "original_summary_sha256"), ("LAUNCH-BINDING.json", "launch_binding_sha256"), ("GATE-Q1.2b.snapshot.json", "gate_snapshot_sha256"),
                     ("procA/result.json", "procA_result_sha256"), ("procB/result.json", "procB_result_sha256"), ("procA/raw_inventory.jsonl", "procA_inventory_sha256"), ("procB/raw_inventory.jsonl", "procB_inventory_sha256")):
        assert sha(os.path.join(REAL_RUN, rel)) == o[key], rel
    s = json.load(open(os.path.join(REAL_RUN, "summary.v2.json"))); assert s["findings"]["malformed"] == o["original_findings"] and not s["findings"]["structural"]
    rc = json.load(open(os.path.join(REAL_RUN, "RUN-RECEIPT.json"))); assert rc["status"] == "COMPLETED_reduce_rc=2" and rc["proc_exit"] == {"A": "exit=0", "B": "exit=0"} and rc["reduce_exit"] == "exit=2"
    for t in "AB":
        att = json.load(open(os.path.join(REAL_RUN, f"proc{t}", "result.json")))["attestation"]
        assert att["helper_hashes"] == REAL_HELPERS and att["runner_sha256"] == RD221.BOUND_RUNNER_SHA256 == sha(RUNNER22)     # the real evidence has exactly the defective shape + correct runner identity
    lb = json.load(open(os.path.join(REAL_RUN, "LAUNCH-BINDING.json"))); assert lb["expect"]["runner"] == sha(RUNNER22) and lb["expect"]["reducer"] == o["original_reducer_sha256"] == sha(REDUCER22)
    pr, facts = RD221.bind_original_run(REAL_RUN, "evaluation", o, "/tmp/never-created-out", auth_for(o, reduction_id="never-created-out"))   # intended-original positive on the REAL preserved evidence
    assert pr == [] and facts["repaired_reducer_sha256"] == sha(REDUCER221)


# ------------------------------------------------------------------ (5) adversarial refusals
def _clone(stage):
    tmp = tempfile.mkdtemp(prefix="q12b-v221-m-"); run = os.path.join(tmp, "q12b-heldout-SYNTH"); shutil.copytree(stage["run"], run)
    return tmp, run


REFUSALS = {
    # helper-map shape / values (re-bound to the mutated evidence so ONLY the reconciliation rule can fire)
    "helper_map_extra_entry": ("rebind", lambda run: _rewrite_result(run, "A", lambda j: j["attestation"]["helper_hashes"].update({"extra.py": "e" * 64})), "exact bound defective shape"),
    "helper_map_v21_value_wrong": ("rebind", lambda run: _rewrite_result(run, "B", lambda j: j["attestation"]["helper_hashes"].update({"q1_component_runner_v2_1.py": "0" * 64})), "frozen v2.1 runner bytes"),
    "helper_map_v21_entry_missing": ("rebind", lambda run: _rewrite_result(run, "A", lambda j: j["attestation"]["helper_hashes"].pop("q1_component_runner_v2_1.py")), "exact bound defective shape"),
    "helper_map_contains_v22_key": ("rebind", lambda run: _rewrite_result(run, "A", lambda j: j["attestation"]["helper_hashes"].update({"q1_component_runner_v2_2.py": sha(RUNNER22)})), "exact bound defective shape"),
    "helper_map_other_value_wrong": ("rebind", lambda run: _rewrite_result(run, "B", lambda j: j["attestation"]["helper_hashes"].update({"q1_oracle.py": "0" * 64})), "loaded helper q1_oracle.py"),
    "attestation_runner_is_v21": ("rebind", lambda run: _rewrite_result(run, "A", lambda j: j["attestation"].update(runner_sha256=FROZEN_V21)), "runner_sha256"),
    "record_execution_runner_wrong": ("rebind", lambda run: _rewrite_record(run, "A", "raw/*/L0__out__node00.json", lambda r: r["execution"].update(runner_sha256="0" * 64)), "execution runner identity"),
    "launch_binding_runner_is_v21": ("rebind", lambda run: _edit(os.path.join(run, "LAUNCH-BINDING.json"), lambda j: (j["expect"].update(runner=FROZEN_V21), j["helpers"].update({"q1_component_runner_v2_2.py": FROZEN_V21}))), "pinned ORIGINAL reducer bytes"),
    # reducer identity: the immutable original binding must name the ORIGINAL reducer; the authorization must name the NEW one (reviewer blocker on draft eb117260)
    "launch_binding_reducer_rewritten_to_new": ("rebind", lambda run: _edit(os.path.join(run, "LAUNCH-BINDING.json"), lambda j: j["expect"].update(reducer=sha(REDUCER221))), "pinned ORIGINAL reducer bytes"),
    "launch_binding_reducer_unknown": ("rebind", lambda run: _edit(os.path.join(run, "LAUNCH-BINDING.json"), lambda j: j["expect"].update(reducer="0" * 64)), "pinned ORIGINAL reducer bytes"),
    "original_reducer_pin_wrong": ("orig", lambda o: o.update(original_reducer_sha256=sha(REDUCER221)), "pinned ORIGINAL reducer bytes"),
    "auth_names_old_reducer_as_repaired": ("auth", lambda a: dict(a, repaired_reducer_sha256=sha(REDUCER22)), "names reducer bytes"),
    # original evidence / identity (bound BEFORE the mutation -> the binding itself must refuse)
    "original_summary_changed": ("keep", lambda run: _edit(os.path.join(run, "summary.v2.json"), lambda j: j["findings"]["malformed"].append("x")), "summary.v2.json sha"),
    "original_receipt_changed": ("keep", lambda run: _edit(os.path.join(run, "RUN-RECEIPT.json"), lambda j: j.update(status="COMPLETED_reduce_rc=0")), "RUN-RECEIPT.json sha"),
    "original_procA_result_changed": ("keep", lambda run: _rewrite_result(run, "A", lambda j: j.update(note="edited")), "procA/result.json sha"),
    "original_inventory_changed": ("keep", lambda run: open(os.path.join(run, "procB", "raw_inventory.jsonl"), "a").write("\n"), "procB/raw_inventory.jsonl sha"),
    "run_id_differs": ("orig", lambda o: o.update(run_id="OTHER"), "run id"),
    "original_findings_differ": ("orig", lambda o: o.update(original_findings=["procA: something else"]), "original failing summary findings"),
    "block_calibration": ("block", None, "block 'calibration' != bound original block"),
    # authorization
    "auth_missing": ("auth", lambda a: None, "authorization missing"),
    "auth_not_approved": ("auth", lambda a: dict(a, approved=False), "not approved"),
    "auth_not_cpu_only": ("auth", lambda a: dict(a, authorizes="GPU_RERUN"), "CPU-reduction-only"),
    "auth_wrong_run": ("auth", lambda a: dict(a, run_id="OTHER"), "authorization run id"),
    "auth_wrong_reducer_bytes": ("auth", lambda a: dict(a, repaired_reducer_sha256=sha(REDUCER22)), "names reducer bytes"),
    "auth_wrong_evidence": ("auth", lambda a: dict(a, original_evidence=dict(a["original_evidence"], run_receipt_sha256="0" * 64)), "original_evidence.run_receipt_sha256"),
    "auth_reduction_id_mismatch": ("auth", lambda a: dict(a, reduction_id="OTHER"), "reduction_id"),
    # output placement
    "out_inside_run": ("out_inside", None, "OUTSIDE the original run"),
    "out_missing": ("out_missing", None, "--out"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_v2_2_1_refusals(stage, case):
    mode, mut, needle = REFUSALS[case]
    tmp, run = _clone(stage)
    try:
        original = copy.deepcopy(stage["original"]); auth = copy.deepcopy(stage["auth"]); block = "evaluation"; out = os.path.join(tmp, "reductions", "RED-1"); os.makedirs(out)
        if mode == "rebind":
            mut(run); original = original_from_run(run, stage["s22"]["findings"]["malformed"]); auth = auth_for(original)
        elif mode == "keep":
            mut(run)
        elif mode == "orig":
            mut(original); auth = auth_for(original)
        elif mode == "auth":
            auth = mut(auth)
        elif mode == "block":
            block = "calibration"
        elif mode == "out_inside":
            out = os.path.join(run, "reduction-inside"); os.makedirs(out)
        elif mode == "out_missing":
            out = None
        s, code = RD221.reduce_run(run, block, "none", out_dir=out, authorization=auth, original=original)
        assert code == 2, (case, code, s["findings"])
        allf = [p for k in s["findings"] for p in s["findings"][k]]
        assert any(needle in p for p in allf), (case, allf)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fixed_runner_shape_is_NOT_accepted_by_v2_2_1(stage):
    """A helper map WITHOUT the redundant entry (what a corrected runner would emit) is a different runner and must refuse here: v2.2.1 is bound to the exact preserved v2.2 evidence."""
    tmp, run = _clone(stage)
    try:
        for t in "AB":
            _rewrite_result(run, t, lambda j: (j["attestation"]["helper_hashes"].pop("q1_component_runner_v2_1.py"), j["attestation"]["helper_hashes"].update({"q1_component_runner_v2_2.py": sha(RUNNER22)})))
        original = original_from_run(run, stage["s22"]["findings"]["malformed"]); out = os.path.join(tmp, "r", "RED-1"); os.makedirs(out)
        s, code = RD221.reduce_run(run, "evaluation", "none", out_dir=out, authorization=auth_for(original), original=original)
        assert code == 2 and sum("exact bound defective shape" in p for p in s["findings"]["malformed"]) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ (5b) CLI no-original-write contract (second reviewer issue): guard BEFORE any mkdir/write
def _cli(run, out, auth_path, extra=()):
    return subprocess.run([sys.executable, REDUCER221, "--run", run, "--block", "evaluation", "--recompute", "none", "--out", out, "--authorization", auth_path, *extra],
                          capture_output=True, text=True, timeout=600, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""})


@pytest.mark.parametrize("kind", ["inside", "same", "symlink_to_run", "symlink_into_run", "ancestor", "existing_file"])
def test_cli_invalid_output_refuses_before_any_filesystem_mutation(stage, kind):
    tmp, run = _clone(stage)
    try:
        auth_p = os.path.join(tmp, "auth.json"); json.dump(stage["auth"], open(auth_p, "w"))
        if kind == "inside": out = os.path.join(run, "reduction")
        elif kind == "same": out = run
        elif kind == "symlink_to_run": os.symlink(run, os.path.join(tmp, "link")); out = os.path.join(tmp, "link", "red")
        elif kind == "symlink_into_run": os.symlink(os.path.join(run, "procA"), os.path.join(tmp, "linkA")); out = os.path.join(tmp, "linkA", "red")
        elif kind == "ancestor": out = os.path.dirname(run)
        else: out = os.path.join(tmp, "afile"); open(out, "w").write("x")
        before = _snapshot(run); listing = sorted(os.listdir(tmp))
        r = _cli(run, out, auth_p)
        assert r.returncode == 2 and "REFUSED before any filesystem mutation" in r.stderr and ("OUTSIDE the original run" in r.stderr or "existing file" in r.stderr), (kind, r.stderr[-400:])
        assert _snapshot(run) == before and sorted(os.listdir(tmp)) == listing                                          # nothing created or written anywhere
        assert not os.path.exists(os.path.join(run, "reduction")) and not os.path.exists(os.path.join(run, "summary.v2_2_1.json"))
        assert RD221.output_guard(run, out) is not None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_valid_separate_output_writes_only_there(stage):
    tmp, run = _clone(stage)
    try:
        auth_p = os.path.join(tmp, "auth.json"); json.dump(stage["auth"], open(auth_p, "w")); out = os.path.join(tmp, "reductions", "RED-1")
        assert RD221.output_guard(run, out) is None
        before = _snapshot(run); r = _cli(run, out, auth_p)
        assert r.returncode == 2 and "REFUSED before any filesystem mutation" not in r.stderr                                   # the CLI default binding is the REAL preserved run: refuses on run id, AFTER creating the separate output
        s = json.load(open(os.path.join(out, "summary.v2_2_1.json"))); assert any("run id" in p for p in s["findings"]["malformed"]) and s["reduction"]["out_dir"] == out
        assert _snapshot(run) == before and sorted(os.listdir(out)) == ["summary.v2_2_1.json"]                                # the original run untouched; only the separate output received a file
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ (6) unchanged numerical path, (7) launcher
def test_v2_2_1_numerical_path_identical_to_v2_2():
    a, b = open(REDUCER22).read().splitlines(), open(REDUCER221).read().splitlines()
    d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
    removed = [l[1:] for l in d if l.startswith("-")]
    assert len(removed) == 7 and not any(k in l for l in removed for k in ("evaluate_case", "aggregate_v2_1", "reference_only_eligibility", "_recompute", "verify_record", "return summary, 6", "return summary, 4", "return summary, 5"))
    for n in ("verify_record", "load_inventory", "process_independence", "_recompute_metrics", "reducer_runtime_identity", "canonical_sha", "_case_rel", "sha256_file", "_load_json", "_is_int"):
        assert inspect.getsource(getattr(RD22, n)) == inspect.getsource(getattr(RD221, n)), n
    assert RD221.PE21 is RD22.PE21 and RD221.STRUCT_KEYS == RD22.STRUCT_KEYS and RD221.NEG_TAGS == RD22.NEG_TAGS and RD221.EXPECT_KEYS == RD22.EXPECT_KEYS and RD221.HELPER_KEYS == RD22.HELPER_KEYS
    assert sha(os.path.join(TOOLS, "q1_policy_evaluator_v2_1.py")) == json.load(open(os.path.join(CAMP, "FREEZE-Q1_2B-v2.1.json")))["files"]["tools/q1_policy_evaluator_v2_1.py"]
    for fz in ("FREEZE-Q1_2B-HELDOUT-v1.json",):
        for rel, want in json.load(open(os.path.join(CAMP, fz)))["files"].items():
            assert sha(os.path.join(CAMP, rel)) == want["sha256"], rel                                     # the held-out freeze bytes (runner/launcher/reducer v2.2, tests, log) are unchanged


def test_repair_launcher_static_contract_and_dry_run():
    L = open(REPAIR_LAUNCHER).read()
    for tok in ("q1_2b_reduce_v2_2_1.py", ":ro", "--out /reductions/", "--authorization", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "AUTH-Q1.2b-HELDOUT-REDUCTION.json", "runs/q1.2b-heldout-reduction/",
                "--block evaluation --recompute all", "REDUCTION-BINDING.json", "REDUCTION-RECEIPT.json", "AUTHORIZATION.snapshot.json", "q12b-heldout-20260928T013539Z", "bind_original_run("):
        assert tok in L, tok
    assert "--gpus" not in L and "summary.v2.json\", \"x\"" not in L
    assert subprocess.run(["bash", "-n", REPAIR_LAUNCHER]).returncode == 0
    r = subprocess.run(["bash", REPAIR_LAUNCHER, "--dry-run"], env={**os.environ, "REDUCTION_ID": "DRYRUN-TEST"}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "(dry-run: nothing launched)" in r.stdout and "read-only" in r.stdout and f"REDUCER_V2_2_1_SHA={sha(REDUCER221)}" in r.stdout
    assert not os.path.exists(os.path.join(CAMP, "runs", "q1.2b-heldout-reduction", "DRYRUN-TEST"))
