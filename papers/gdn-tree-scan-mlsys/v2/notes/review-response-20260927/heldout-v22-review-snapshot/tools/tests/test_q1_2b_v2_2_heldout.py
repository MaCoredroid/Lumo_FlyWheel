"""Targeted CPU tests for the Q1.2b HELD-OUT launch package v2.2 (parent instruction 2026-09-28T01:09:47Z).

What is proven here (no GPU, no docker, no model; the REAL held-out fixture tensors are never loaded and no held-out candidate
output exists or is inspected -- only manifest/policy METADATA is read from the frozen files):
  S  static: runner/launcher/reducer v2.2 differ from the frozen v2.1 bytes only in the held-out block guard / identity /
     scope hunks; every numerical function is byte-identical; evaluator/policy/fixtures/oracle/bf16 bytes equal the v2.1 freeze;
     the reducer's bound runner/launcher constants equal the exact files.
  A  authorization on the REAL manifest + policy v2.1: the held-out block is admitted only under a complete gate scope, and
     malformed block strings / unauthorized gates / missing or incomplete fixture sets / negatives / limits / unbound runner refuse.
  C  complete coverage: the bound held-out domain is exactly the manifest's evaluation fixtures (5760 cases); a CPU stub run of
     the v2.2 process seals every bound case exactly once with ZERO negative products; reducer v2.2 accepts it (PASS) and refuses
     wrong runner/launcher/reducer bytes, non-zero negatives, unnamed fixtures, injected negatives and the wrong block.
"""
import argparse, copy, difflib, glob, hashlib, inspect, json, os, shutil, subprocess, sys, tempfile

os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["CUDA_VISIBLE_DEVICES"] = ""   # reducer runtime pins
import pytest
import torch
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__)); TOOLS = os.path.dirname(HERE); CAMP = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS); sys.path.insert(0, HERE)
import q1_2b_fixtures as FX  # noqa: E402
import q1_component_runner_v2_1 as R21  # noqa: E402
import q1_component_runner_v2_2 as R22  # noqa: E402
import q1_2b_reduce_v2_1 as RD21  # noqa: E402
import q1_2b_reduce_v2_2 as RD22  # noqa: E402
import q1_policy_evaluator_v2_1 as PE  # noqa: E402
import q1_policy_build_v2_1 as PB  # noqa: E402
from test_q1_2b_v2_1_pipeline import StubBackend, _edit, _rewrite_result  # noqa: E402  (stub backend + tamper helpers only; no fixtures/tests imported)

RUNNER21, RUNNER22 = os.path.join(TOOLS, "q1_component_runner_v2_1.py"), os.path.join(TOOLS, "q1_component_runner_v2_2.py")
LAUNCHER21, LAUNCHER22 = os.path.join(TOOLS, "run_q1_2b_component_v2_1.sh"), os.path.join(TOOLS, "run_q1_2b_component_v2_2.sh")
REDUCER21, REDUCER22 = os.path.join(TOOLS, "q1_2b_reduce_v2_1.py"), os.path.join(TOOLS, "q1_2b_reduce_v2_2.py")
REAL_FM = os.path.join(CAMP, "fixtures", "q1_2b", "manifest.json"); REAL_OBS = os.path.join(CAMP, "fixtures", "q1_2b", "expected_observations.json")
POL21 = os.path.join(CAMP, "policy", "q1_component_numerical_policy.v2.1.json"); POLICY2 = os.path.join(CAMP, "policy", "q1_component_numerical_policy.v2.json")
CONTRACT2 = os.path.join(CAMP, "..", "..", "p0", "monitor", "review-response-20260927", "Q1-PAIRED-NUMERICAL-CONTRACT-v2.json")
HELD_OUT_IDS = ["evaluation/fx0_ordinary-random", "evaluation/fx1_mixed-stress"]
POWER_RUN = "q12b-calibration-retry-20260928T010045Z"
pytestmark = pytest.mark.skipif(not (os.path.exists(POL21) and os.path.exists(REAL_FM)), reason="frozen policy v2.1 / fixture manifest not present")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def _removed_added(a_path, b_path):
    a, b = open(a_path).read().splitlines(), open(b_path).read().splitlines()
    d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
    return [l[1:] for l in d if l.startswith("-")], [l[1:] for l in d if l.startswith("+")]


# ------------------------------------------------------------------ S: static / minimal diff
def test_reducer_bound_constants_equal_exact_files():
    assert RD22.BOUND_RUNNER_SHA256 == sha(RUNNER22) and RD22.BOUND_LAUNCHER_SHA256 == sha(LAUNCHER22)
    assert RD22.HELPER_KEYS[0] == "q1_component_runner_v2_2.py" and RD22.HELPER_TO_EXPECT["q1_component_runner_v2_2.py"] == "runner"
    assert "q1_component_runner_v2_1.py" not in RD22.HELPER_KEYS and RD22.EXPECT_KEYS == RD21.EXPECT_KEYS and RD22.MANDATORY_ATT == RD21.MANDATORY_ATT


def test_frozen_v2_1_dependencies_unchanged():
    for fz in ("FREEZE-Q1_2B-v2.1.json", "FREEZE-Q1_2B-INIT-v1.1.json"):
        files = json.load(open(os.path.join(CAMP, fz)))["files"]
        for rel, want in files.items():
            p = os.path.join(CAMP, rel)
            assert os.path.exists(p), rel
            assert sha(p) == (want["sha256"] if isinstance(want, dict) else want), rel


def test_runner_v2_2_diff_is_the_block_guard_only():
    removed, added = _removed_added(RUNNER21, RUNNER22)
    assert removed == ['"""Q1.2b candidate GDN component runner **v2.1** (runs INSIDE the pinned image, GPU; case loop CPU-testable via a backend).',
                       '    eligible = (args.block == "calibration" and args.repeats >= 2 and args.limit == 0)']
    assert any("def block_authorization(" in l for l in added) and any('eligible = (args.block in SUPPORTED_BLOCKS and not block_problems' in l for l in added)
    assert not any(k in l for l in added for k in ("evaluate_case", "_paired", "kappa", "threshold", "reference_only_eligibility", "metrics_record(", "run_negatives("))
    same = [n for n in dir(R21) if not n.startswith("__") and n not in ("main", "block_authorization") and (inspect.isfunction(getattr(R21, n)) or inspect.isclass(getattr(R21, n)))
            and getattr(getattr(R21, n), "__module__", None) == R21.__name__]
    assert len(same) > 20
    for n in same:
        assert inspect.getsource(getattr(R21, n)) == inspect.getsource(getattr(R22, n)), n
    for n in ("REQUIRED_ENV", "EXPECTED_KERNEL_SHA", "EXPECTED_TOPOLOGY_SHA", "EXPECTED_NATIVE_SHA", "HELPER_FILES", "REPO_HELPER_FILES", "SCHEMA_RESULT", "N5_TARGET_PATH", "N1_SIBLING_PATH", "N3_INSTANCES", "N4_PREVIOUS_PATH"):
        assert getattr(R21, n) == getattr(R22, n), n
    src = open(RUNNER22).read()
    assert 'refusals += block_problems' in src and 'att["block_authorization"] = block_facts' in src and '([], {"skipped": "init_only"}) if args.init_only else block_authorization(' in src
    assert 'if args.negatives and meta["block"] == "calibration":' in src          # negative mutations stay calibration-only inside the case loop too
    assert R22.SUPPORTED_BLOCKS == ("calibration", "evaluation") and R22.HELD_OUT_BLOCK == "evaluation"


def test_reducer_v2_2_numerical_path_unchanged():
    removed, _ = _removed_added(REDUCER21, REDUCER22)
    assert len(removed) == 5 and any("STRUCT_KEYS:" in l for l in removed) and any("expected_neg_counts = {" in l for l in removed) and sum("q1_component_runner_v2_1.py" in l for l in removed) == 3   # docstring + HELPER_KEYS + HELPER_TO_EXPECT
    assert not any(k in l for l in removed for k in ("evaluate_case", "aggregate_v2_1", "reference_only_eligibility", "_recompute_metrics", "verify_record", "return summary, 6", "return summary, 4"))
    for n in ("verify_record", "load_inventory", "process_independence", "_recompute_metrics", "reducer_runtime_identity", "canonical_sha", "_case_rel", "sha256_file", "_load_json", "_is_int", "main"):
        assert inspect.getsource(getattr(RD21, n)) == inspect.getsource(getattr(RD22, n)), n
    assert RD22.PE21 is RD21.PE21 and sha(os.path.join(TOOLS, "q1_policy_evaluator_v2_1.py")) == json.load(open(os.path.join(CAMP, "FREEZE-Q1_2B-v2.1.json")))["files"]["tools/q1_policy_evaluator_v2_1.py"]
    assert RD22.STRUCT_KEYS == RD21.STRUCT_KEYS and RD22.NEG_TAGS == RD21.NEG_TAGS and RD22.TENSOR_SHAPES == RD21.TENSOR_SHAPES


def test_launcher_v2_2_static_contract_and_dry_run(tmp_path):
    L = open(LAUNCHER22).read()
    for tok in ("BLOCK=evaluation", "NEGATIVES=0", "q1_component_runner_v2_2.py", "q1_2b_reduce_v2_2.py", "runs/q1.2b-heldout/", "held_out_fixture_ids", "calibration_power_run_id",
                "GATE-Q1.2b-HELDOUT.json", 'g.get("gate") != "Q1.2b"', '"tree_conv_fused"', "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL=0", "--recompute all", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1"):
        assert tok in L, tok
    assert "q1_component_runner_v2_1.py" not in L and "q1_2b_reduce_v2_1.py" not in L and "runs/q1.2b/$RUN_ID" not in L
    removed, added = _removed_added(LAUNCHER21, LAUNCHER22)
    assert all(any(k in l for k in ("launcher v2.1", "GATE=", "REPEATS=2; BLOCK=calibration", "OUT=$REPO/$C/runs/q1.2b/", "q1_component_runner_v2_1.py", "q1_2b_reduce_v2_1.py", '"scope": {')) for l in removed), removed
    assert subprocess.run(["bash", "-n", LAUNCHER22]).returncode == 0
    r = subprocess.run(["bash", LAUNCHER22, "--dry-run"], env={**os.environ, "RUN_ID": "DRYRUN-TEST"}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "(dry-run: nothing launched)" in r.stdout and "runs/q1.2b-heldout/DRYRUN-TEST" in r.stdout and "--block evaluation" in r.stdout and "--negatives 0" in r.stdout
    assert not os.path.exists(os.path.join(CAMP, "runs", "q1.2b-heldout", "DRYRUN-TEST"))


# ------------------------------------------------------------------ A: authorization on the REAL manifest / policy metadata
def _gate(block="evaluation", **over):
    g = {"gate": "Q1.2b", "approved": True, "approved_run_id": "RUN-HO-1", "reviewed_hashes": {"policy": sha(POL21), "runner": sha(RUNNER22)},
         "reviewed_scope": ({"repeats": 2, "block": "evaluation", "negatives": 0, "processes": ["A", "B"], "held_out_fixture_ids": list(HELD_OUT_IDS), "calibration_power_run_id": POWER_RUN}
                            if block == "evaluation" else {"repeats": 2, "block": "calibration", "negatives": 1, "processes": ["A", "B"]})}
    g.update(over)
    return g


def _pol(gate, path=POL21):
    approval = {"approved": gate.get("approved") is True, "policy_sha256": (gate.get("reviewed_hashes") or {}).get("policy"), "stage": gate.get("gate"), "expected_observations_sha256": None,
                "approved_run_id": gate.get("approved_run_id")}
    return PE.load_policy_verified_v2_1(path, sha(path), approval)


def _args(block="evaluation", negatives=0, limit=0, init_only=False):
    return argparse.Namespace(init_only=init_only, expected_policy_sha256=sha(POL21), block=block, negatives=negatives, limit=limit)


def _fm():
    return json.load(open(REAL_FM))


def test_held_out_authorized_with_complete_coverage_on_real_metadata():
    g = _gate(); pol = _pol(g); fm = _fm()
    assert pol["_integrity_ok"] and pol["_approved_ok"]
    assert R22.authorize_scope(g, pol, _args())["ok"]                                   # unchanged numerical authorization path (gate "Q1.2b")
    pr, facts = R22.block_authorization(g, pol, fm, _args())
    assert pr == [] and facts["complete_coverage"] is True and facts["supported"] is True
    assert sorted(facts["manifest_fixture_ids"]) == sorted(facts["policy_domain_held_out_ids"]) == sorted(facts["gate_named_held_out_ids"]) == HELD_OUT_IDS
    assert facts["calibration_power_run_id"] == POWER_RUN and facts["args_negatives"] == 0 and facts["args_limit"] == 0
    obs = json.load(open(REAL_OBS))
    bound, problems = PE.bind_expected_manifest(pol, obs, sha(REAL_OBS), "evaluation")
    assert not problems and len(bound) == pol["domain"]["q1_2b"]["cases_evaluation_held_out"] == 5760
    per = {}
    for c in bound.values():
        per[c["fixture_id"]] = per.get(c["fixture_id"], 0) + 1
    assert per == {HELD_OUT_IDS[0]: 2880, HELD_OUT_IDS[1]: 2880} and all(cid.startswith("evaluation/") for cid in bound)
    assert not any(c["fixture_id"].startswith("calibration/") for c in bound.values())


def test_calibration_block_keeps_v2_1_semantics_under_v2_2():
    g = _gate("calibration"); pol = _pol(g)
    pr, facts = R22.block_authorization(g, pol, _fm(), _args("calibration", negatives=1))
    assert pr == [] and facts["complete_coverage"] is True and "gate_named_held_out_ids" not in facts


REFUSALS = {
    "malformed_block_held_out": (lambda g, fm, pol: (g, fm, pol, _args("held_out")), "unsupported block"),
    "malformed_block_capitalised": (lambda g, fm, pol: (g, fm, pol, _args("Evaluation")), "unsupported block"),
    "malformed_block_empty": (lambda g, fm, pol: (g, fm, pol, _args("")), "unsupported block"),
    "malformed_block_slash": (lambda g, fm, pol: (g, fm, pol, _args("evaluation/")), "unsupported block"),
    "unauthorized_calibration_gate_for_held_out": (lambda g, fm, pol: (_gate("calibration"), fm, pol, _args()), "does not authorize block 'evaluation'"),
    "unauthorized_held_out_gate_for_calibration": (lambda g, fm, pol: (g, fm, pol, _args("calibration", negatives=1)), "does not authorize block 'calibration'"),
    "unauthorized_named_ids_missing": (lambda g, fm, pol: (_gate(reviewed_scope={k: v for k, v in _gate()["reviewed_scope"].items() if k != "held_out_fixture_ids"}), fm, pol, _args()), "does not name exactly"),
    "unauthorized_named_ids_partial": (lambda g, fm, pol: (_gate(reviewed_scope={**_gate()["reviewed_scope"], "held_out_fixture_ids": HELD_OUT_IDS[:1]}), fm, pol, _args()), "does not name exactly"),
    "unauthorized_named_ids_extra": (lambda g, fm, pol: (_gate(reviewed_scope={**_gate()["reviewed_scope"], "held_out_fixture_ids": HELD_OUT_IDS + ["evaluation/fx9_unknown"]}), fm, pol, _args()), "does not name exactly"),
    "unauthorized_named_ids_duplicate": (lambda g, fm, pol: (_gate(reviewed_scope={**_gate()["reviewed_scope"], "held_out_fixture_ids": [HELD_OUT_IDS[0], HELD_OUT_IDS[0]]}), fm, pol, _args()), "does not name exactly"),
    "unauthorized_named_ids_not_list": (lambda g, fm, pol: (_gate(reviewed_scope={**_gate()["reviewed_scope"], "held_out_fixture_ids": "evaluation/*"}), fm, pol, _args()), "does not name exactly"),
    "negatives_requested_by_args": (lambda g, fm, pol: (g, fm, pol, _args(negatives=1)), "calibration-only"),
    "negatives_in_gate_scope": (lambda g, fm, pol: (_gate(reviewed_scope={**_gate()["reviewed_scope"], "negatives": 1}), fm, pol, _args()), "calibration-only"),
    "limit_nonzero": (lambda g, fm, pol: (g, fm, pol, _args(limit=1)), "complete coverage"),
    "power_run_id_missing": (lambda g, fm, pol: (_gate(reviewed_scope={k: v for k, v in _gate()["reviewed_scope"].items() if k != "calibration_power_run_id"}), fm, pol, _args()), "calibration_power_run_id"),
    "runner_bytes_not_bound": (lambda g, fm, pol: (_gate(reviewed_hashes={"policy": sha(POL21), "runner": sha(RUNNER21)}), fm, pol, _args()), "bind this runner"),
    "manifest_missing_one_held_out_entry": (lambda g, fm, pol: (g, {**fm, "fixtures": [e for e in fm["fixtures"] if e["fixture_id"] != HELD_OUT_IDS[1]]}, pol, _args()), "policy domain evaluation_held_out ids"),
    "manifest_without_held_out": (lambda g, fm, pol: (g, {**fm, "fixtures": [e for e in fm["fixtures"] if e["block"] != "evaluation"]}, pol, _args()), "no complete typed unique fixture set"),
    "manifest_duplicate_held_out_entry": (lambda g, fm, pol: (g, {**fm, "fixtures": fm["fixtures"] + [copy.deepcopy(next(e for e in fm["fixtures"] if e["block"] == "evaluation"))]}, pol, _args()), "no complete typed unique fixture set"),
    "manifest_entry_untyped_sha": (lambda g, fm, pol: (g, {**fm, "fixtures": [({**e, "sha256": "abc"} if e["block"] == "evaluation" else e) for e in fm["fixtures"]]}, pol, _args()), "no complete typed unique fixture set"),
    "manifest_entry_wrong_prefix": (lambda g, fm, pol: (g, {**fm, "fixtures": [({**e, "fixture_id": "calibration/x"} if e["fixture_id"] == HELD_OUT_IDS[0] else e) for e in fm["fixtures"]]}, pol, _args()), "no complete typed unique fixture set"),
    "policy_domain_missing_held_out_id": (lambda g, fm, pol: (g, fm, {**pol, "domain": {"q1_2b": {**pol["domain"]["q1_2b"], "fixture_ids": {**pol["domain"]["q1_2b"]["fixture_ids"], "evaluation_held_out": HELD_OUT_IDS[:1]}}}}, _args()), "policy domain evaluation_held_out ids"),
    "policy_domain_absent": (lambda g, fm, pol: (g, fm, {k: v for k, v in pol.items() if k != "domain"}, _args()), "policy domain evaluation_held_out ids"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_held_out_refusals(case):
    mut, needle = REFUSALS[case]
    g0 = _gate(); g, fm, pol, args = mut(g0, _fm(), _pol(g0))
    pr, facts = R22.block_authorization(g, pol, fm, args)
    assert pr and any(needle in p for p in pr), (case, pr)
    assert facts["complete_coverage"] is False


def test_init_only_scope_skips_block_authorization_in_main_source():
    src = open(RUNNER22).read()
    i = src.index('block_problems, block_facts = ([], {"skipped": "init_only"}) if args.init_only else block_authorization(gate_doc, pol, manifest, args)')
    assert src.index("    manifest = json.load(open(args.manifest))") < i < src.index("    if refusals:\n        att[\"refusals\"] = refusals")   # after manifest/policy load, before the refusal receipt


# ------------------------------------------------------------------ C: complete coverage on CPU (synthetic 2-layer fixtures, both blocks generated, held-out selected)
def make_stage(tmp, block="evaluation", layers=2):
    fxroot = os.path.join(tmp, "fixtures"); FX.generate(fxroot, blocks=("calibration", "evaluation"), layers=layers, fixtures_per_block=2)
    fm_p = os.path.join(fxroot, "manifest.json"); obs_p = os.path.join(fxroot, "expected_observations.json"); pol_p = os.path.join(tmp, "policy.v2.1.json")
    pol = PB.build(POLICY2, sha(POLICY2), fm_p, obs_p); json.dump(pol, open(pol_p, "w"), indent=1)
    fm = json.load(open(fm_p)); ho = sorted(e["fixture_id"] for e in fm["fixtures"] if e["block"] == block)
    run = os.path.join(tmp, "run", "RUN-1"); os.makedirs(os.path.join(run, "tensors")); os.makedirs(os.path.join(run, "fixtures"))
    for e in fm["fixtures"]:
        if e["block"] != block:
            continue
        dst = os.path.join(run, "fixtures", e["path"]); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(os.path.join(fxroot, e["path"]), dst)
    helpers = {"q1_component_runner_v2_2.py": sha(RUNNER22), "q1_oracle.py": sha(os.path.join(TOOLS, "q1_oracle.py")), "q1_2b_fixtures.py": sha(os.path.join(TOOLS, "q1_2b_fixtures.py")),
               "q1_policy_evaluator.py": sha(os.path.join(TOOLS, "q1_policy_evaluator.py")), "q1_policy_evaluator_v2.py": sha(os.path.join(TOOLS, "q1_policy_evaluator_v2.py")),
               "q1_policy_evaluator_v2_1.py": sha(os.path.join(TOOLS, "q1_policy_evaluator_v2_1.py")), "q1_bf16_ulp.py": sha(os.path.join(TOOLS, "q1_bf16_ulp.py")),
               "scripts/fr13_fixed32_topology.py": "t" * 64, "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py": "k" * 64, "src/lumo_flywheel_serving/fr13_tree_conv_fused.py": "c" * 64}
    expect = {"image_id": "img", "fixture_manifest": sha(fm_p), "expected_observations": sha(obs_p), "policy": sha(pol_p), "policy_base_v2": sha(POLICY2), "contract_v2": sha(CONTRACT2),
              "runner": sha(RUNNER22), "reducer": sha(REDUCER22), "launcher": sha(LAUNCHER22), "fixtures_tool": helpers["q1_2b_fixtures.py"], "oracle": helpers["q1_oracle.py"],
              "evaluator_v1": helpers["q1_policy_evaluator.py"], "evaluator_v2": helpers["q1_policy_evaluator_v2.py"], "evaluator_v2_1": helpers["q1_policy_evaluator_v2_1.py"],
              "bf16_ulp": helpers["q1_bf16_ulp.py"], "topology": "t" * 64, "kernel": "k" * 64, "native_module": "n" * 64, "verification_V2": "v" * 64, "tree_conv_fused": "c" * 64}
    scope = ({"repeats": 2, "block": "evaluation", "negatives": 0, "processes": ["A", "B"], "held_out_fixture_ids": ho, "calibration_power_run_id": POWER_RUN} if block == "evaluation"
             else {"repeats": 2, "block": "calibration", "negatives": 1, "processes": ["A", "B"]})
    gate = {"gate": "Q1.2b", "approved": True, "approved_run_id": "RUN-1", "reviewed_hashes": dict(expect), "reviewed_scope": copy.deepcopy(scope)}
    for name, src in (("GATE-Q1.2b.snapshot.json", None), ("fixture_manifest.snapshot.json", fm_p), ("expected_observations.snapshot.json", obs_p), ("policy.snapshot.json", pol_p),
                      ("policy_base_v2.snapshot.json", POLICY2), ("contract_v2.snapshot.json", CONTRACT2)):
        if src is None:
            json.dump(gate, open(os.path.join(run, name), "w"), indent=1)
        else:
            shutil.copy2(src, os.path.join(run, name))
    expect["gate"] = sha(os.path.join(run, "GATE-Q1.2b.snapshot.json"))
    lb = {"schema": "lumo.review-response.q1-2b-launch-binding.v2", "expect": expect, "helpers": helpers, "scope": copy.deepcopy(scope), "run_id": "RUN-1",
          "runtime": {"torch": torch.__version__, "triton": "stub", "gpu_name": "stub", "image_id": "img"},
          "snapshots": {"gate": "GATE-Q1.2b.snapshot.json", "fixture_manifest": "fixture_manifest.snapshot.json", "expected_observations": "expected_observations.snapshot.json",
                        "policy": "policy.snapshot.json", "policy_base_v2": "policy_base_v2.snapshot.json", "contract_v2": "contract_v2.snapshot.json"}}
    json.dump(lb, open(os.path.join(run, "LAUNCH-BINDING.json"), "w"), indent=1)
    return {"tmp": tmp, "fxroot": fxroot, "fm_p": fm_p, "obs_p": obs_p, "pol_p": pol_p, "run": run, "gate": gate, "expect": expect, "helpers": helpers, "fm": fm, "obs": json.load(open(obs_p)), "block": block, "ho": ho}


def run_process(st, tag, backend, utc_offset=0):
    block = st["block"]; negatives = 0 if block == "evaluation" else 1
    approval = R22.gate_approval(os.path.join(st["run"], "GATE-Q1.2b.snapshot.json"))
    pol = PE.load_policy_verified_v2_1(st["pol_p"], sha(st["pol_p"]), approval); assert pol["_approved_ok"], (pol["_load_problems"], pol["_auth_problems"])
    bound, pr = PE.bind_expected_manifest(pol, st["obs"], sha(st["obs_p"]), block); assert not pr, pr
    args = argparse.Namespace(process_tag=tag, repeats=2, negatives=negatives, limit=0, init_only=False, block=block, out=os.path.join(st["run"], f"proc{tag}"),
                              tensor_store=os.path.join(st["run"], "tensors"), fixtures_root=st["fxroot"], image_id="img", expected_policy_sha256=sha(st["pol_p"]))
    assert R22.authorize_scope(st["gate"], pol, args)["ok"]
    bp, facts = R22.block_authorization(st["gate"], pol, st["fm"], args); assert bp == [] and facts["complete_coverage"] is True, bp
    os.makedirs(args.out, exist_ok=True)
    att = {"utc": f"2026-09-28T02:00:0{utc_offset}+00:00", "process_tag": tag, "pid": 1, "hostname_in_container": f"c{utc_offset:011d}", "container_name_env": f"lumotree-review-q12b-{tag}",
           "policy_sha256": pol["_loaded_sha256"], "kernel_module_sha256": "k" * 64, "topology_sha256": "t" * 64, "native_module_sha256": "n" * 64, "native_function_source_sha256": "f" * 64,
           "runner_sha256": sha(RUNNER22), "oracle_sha256": st["helpers"]["q1_oracle.py"], "fixtures_tool_sha256": st["helpers"]["q1_2b_fixtures.py"], "image_id_arg": "img",
           "torch": torch.__version__, "triton": "stub", "gpu_name": "stub", "vllm": "stub-vllm", "torch_cuda": None, "expected_observations_sha256": sha(st["obs_p"]),
           "fixture_manifest_sha256": sha(st["fm_p"]), "block": block, "repeats": 2, "helper_hashes": dict(st["helpers"]), "block_authorization": facts}
    entries = [e for e in st["fm"]["fixtures"] if e["block"] == block]
    return R22.run_process(backend, args, entries, bound, pol, att, True), bound


@pytest.fixture(scope="module")
def ho_stage():
    tmp = tempfile.mkdtemp(prefix="q12b-v22-ho-")
    st = make_stage(tmp, "evaluation")
    (st["resA"], st["boundA"]), (st["resB"], _) = run_process(st, "A", StubBackend(2), 1), run_process(st, "B", StubBackend(2), 2)
    yield st
    shutil.rmtree(tmp, ignore_errors=True)


def test_synthetic_policy_domain_names_held_out_like_the_frozen_one(ho_stage):
    pol = json.load(open(ho_stage["pol_p"])); d = pol["domain"]["q1_2b"]
    assert sorted(d["fixture_ids"]["evaluation_held_out"]) == ho_stage["ho"] and len(ho_stage["ho"]) == 2 and d["held_out_rule"] == json.load(open(POL21))["domain"]["q1_2b"]["held_out_rule"]


def test_held_out_process_seals_every_bound_case_once_with_zero_negatives(ho_stage):
    for tag in ("A", "B"):
        res = ho_stage[f"res{tag}"]; bound = ho_stage["boundA"]
        assert res["eligible"] is True and res["characterization_complete"] is True and res["integrity_ok"] is True and res["block"] == "evaluation" and res["negatives"] == 0
        assert res["fixtures_expected"] == res["fixtures_done"] == 2 and res["attestation"]["block_authorization"]["complete_coverage"] is True
        sealed = []
        for fx in res["fixtures"]:
            assert fx["status"] == "DONE" and fx["structural_ok"] is True and fx["negatives"] == [] and fx["block"] == "evaluation"
            assert not any(k in fx["structural"] for k in RD22.NEG_STRUCT_KEYS) and fx["structural"]["expected_case_product_sealed"] is True
            sealed += [c["case_id"] for c in fx["cases"]]
        assert sorted(sealed) == sorted(bound) and len(sealed) == len(set(sealed)) == 240         # 2 fixtures x 2 layers x (32 outputs + 28 states)
        assert glob.glob(os.path.join(ho_stage["run"], f"proc{tag}", "raw", "*", "NEG__*")) == []
        inv = [json.loads(l) for l in open(os.path.join(ho_stage["run"], f"proc{tag}", "raw_inventory.jsonl"))]
        assert not any(e["kind"] == "negative" for e in inv) and sum(1 for e in inv if e["kind"] == "metrics") == 240


def test_reducer_v2_2_accepts_held_out_run_and_records_power_source(ho_stage):
    s, code = RD22.reduce_run(ho_stage["run"], "evaluation", "all")
    assert code == 0, (code, s["findings"])
    assert s["aggregate"]["A"]["aggregate"] == "PASS" and s["aggregate"]["B"]["aggregate"] == "PASS" and s["cases_expected"] == 240 and s["cases_evaluated"] == {"A": 240, "B": 240}
    assert s["negative_power"] == {"A": {}, "B": {}} and s["negative_power_note"]["power_source_calibration_run_id"] == POWER_RUN and s["negative_power_note"]["negative_mutations_executed"] == 0
    assert s["recomputed_records"] == 480 and s["tensors_verified"] > 0 and s["reducer_runtime"]["torch_threads"] == 1 and s["reducer_runtime"]["cuda_available"] is False
    assert s["process_independence"]["independent"] is True and s["policy"]["approved_ok"] is True


def _clone(st):
    tmp = tempfile.mkdtemp(prefix="q12b-v22-m-"); run = os.path.join(tmp, "RUN-1"); shutil.copytree(st["run"], run)
    return tmp, run


def _lb(run, fn):
    _edit(os.path.join(run, "LAUNCH-BINDING.json"), fn)


REDUCER_REFUSALS = {
    "runner_bytes_are_v2_1": (lambda run: _lb(run, lambda j: (j["expect"].update(runner=sha(RUNNER21)), j["helpers"].update({"q1_component_runner_v2_2.py": sha(RUNNER21)}))), 2, "exact v2.2 bytes"),
    "launcher_bytes_are_v2_1": (lambda run: _lb(run, lambda j: j["expect"].update(launcher=sha(LAUNCHER21))), 2, "exact v2.2 bytes"),
    "reducer_bytes_are_v2_1": (lambda run: _lb(run, lambda j: j["expect"].update(reducer=sha(REDUCER21))), 2, "exact v2.2 bytes"),
    "scope_negatives_nonzero": (lambda run: _lb(run, lambda j: j["scope"].update(negatives=1)), 2, "held-out scope invalid"),
    "scope_named_ids_missing": (lambda run: _lb(run, lambda j: j["scope"].pop("held_out_fixture_ids")), 2, "held-out scope invalid"),
    "scope_power_run_missing": (lambda run: _lb(run, lambda j: j["scope"].pop("calibration_power_run_id")), 2, "held-out scope invalid"),
    "scope_named_ids_differ_from_gate": (lambda run: _lb(run, lambda j: j["scope"].update(held_out_fixture_ids=j["scope"]["held_out_fixture_ids"][:1])), 2, "!= gate.reviewed_scope"),
    "process_negatives_flag_nonzero": (lambda run: _rewrite_result(run, "A", lambda j: j.update(negatives=1)), 2, "negatives 0"),
    "block_authorization_not_complete": (lambda run: _rewrite_result(run, "B", lambda j: j["attestation"]["block_authorization"].update(complete_coverage=False)), 2, "complete-coverage"),
    "block_authorization_names_other_ids": (lambda run: _rewrite_result(run, "B", lambda j: j["attestation"]["block_authorization"].update(gate_named_held_out_ids=["evaluation/other"])), 2, "complete-coverage"),
    "negatives_injected_into_fixture": (lambda run: _rewrite_result(run, "A", lambda j: j["fixtures"][0].update(negatives=[{"tag": "N1_sibling_substitution", "records_expected": 2, "records_sealed": 2}])), 5, "calibration-only"),
    "negative_struct_flags_injected": (lambda run: _rewrite_result(run, "A", lambda j: j["fixtures"][1]["structural"].update(negative_product_exact=True, n5_poisoned_equals_clean_candidate=True)), 5, "calibration-only"),
    "attestation_runner_is_v2_1": (lambda run: _rewrite_result(run, "A", lambda j: j["attestation"].update(runner_sha256=sha(RUNNER21))), 2, "runner_sha256 != bound runner"),
}


@pytest.mark.parametrize("case", sorted(REDUCER_REFUSALS))
def test_reducer_v2_2_refusals(ho_stage, case):
    mut, want_code, needle = REDUCER_REFUSALS[case]
    tmp, run = _clone(ho_stage)
    try:
        mut(run)
        s, code = RD22.reduce_run(run, "evaluation", "none")
        assert code == want_code, (case, code, s["findings"])
        allf = [p for k in s["findings"] for p in s["findings"][k]]
        assert any(needle in p for p in allf), (case, allf)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reducer_v2_2_refuses_wrong_block_and_v2_1_reducer_refuses_v2_2_run(ho_stage):
    s, code = RD22.reduce_run(ho_stage["run"], "calibration", "none")
    assert code == 2 and any("scope invalid for block=calibration" in p for p in s["findings"]["malformed"])
    s21, code21 = RD21.reduce_run(ho_stage["run"], "evaluation", "none")
    assert code21 == 2 and any("helpers missing or incomplete" in p for p in s21["findings"]["malformed"])   # the v2.1 reducer cannot consume a v2.2 launch (separate versions)


def test_calibration_path_still_passes_under_v2_2():
    tmp = tempfile.mkdtemp(prefix="q12b-v22-cal-")
    try:
        st = make_stage(tmp, "calibration")
        run_process(st, "A", StubBackend(2), 3); run_process(st, "B", StubBackend(2), 4)
        s, code = RD22.reduce_run(st["run"], "calibration", "all")
        assert code == 0, (code, s["findings"])
        assert s["negative_power"]["A"]["N1_sibling_substitution"] == ["FAIL"] * 4 and s["negative_power"]["B"]["N3_ring_swap"] == ["FAIL"] * 4 and "negative_power_note" not in s
        assert s["cases_expected"] == 240 and s["recomputed_records"] == 480 + 2 * 2 * 8
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
