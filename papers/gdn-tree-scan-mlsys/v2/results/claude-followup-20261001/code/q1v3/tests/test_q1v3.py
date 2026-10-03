"""CPU-only tests for q1v3 (run with CUDA_VISIBLE_DEVICES='' python3 -m pytest -q tests)."""
import json, os, subprocess, sys, tempfile

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import q1v3_common as Q  # noqa: E402
import cases as CS  # noqa: E402
import client as CL  # noqa: E402
import reduce as R  # noqa: E402
import patch_runner as PR  # noqa: E402
import make_cand_launch as MK  # noqa: E402
import sim_engine as SIM  # noqa: E402

CODEX = "/home/mark/lumotree-review-20260927/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927"
GEN = f"{CODEX}/identity/generated_source/probe-20260928T035847Z/logs/generated"
STOCK = f"{CODEX}/identity/native_source/vllm__v1__worker__gpu_model_runner.py"


# ------------------------------------------------------------------ case set
def test_case_file_structure():
    doc = CS.validate(check_sources=False)
    assert len(doc["cases"]) == 33 and len(doc["negative_controls"]) == 10
    assert all(len(c["cycles"]) >= 3 for c in doc["cases"])
    used = {x["path_name"] for c in doc["cases"] for x in c["cycles"]}
    assert {"R0", "S11", "B5", "B13", "B21", "S3"} <= used                      # root-only, max depth, off-spine, padding-adjacent
    kinds = {x["z_kind"] for c in doc["cases"] for x in c["cycles"]}
    assert kinds == {"bonus", "correction", "sibling"}
    held = {p for p, v in doc["prefixes"].items() if v["held_out"]}
    assert {"eval-short", "eval-medium", "eval-long"} <= held


def test_boundary_cases_cross_the_block():
    doc = CS.load()
    pref = doc["prefixes"]["bnd-25535"]; P, B = pref["reference_prompt_len"], pref["block_boundary"]
    assert (P, B) == (25535, 25600)
    want = {"K1-cross-inside": "inside", "K2-cross-last-draft": "last", "K3-pending-at-boundary": "pending"}
    for c in doc["cases"]:
        if c["prefix"] != "bnd-25535":
            continue
        kind = want[c["case_id"].split("__")[1]]
        k = 5; s = c["cycle_start"][k]; L = c["cycles"][k]["L"]
        commit = range(P + s, P + s + L + 1); z_pos = P + s + L + 1
        if kind == "inside":
            assert B in commit and B != commit[-1]
        elif kind == "last":
            assert commit[-1] == B
        else:
            assert z_pos == B and B not in commit


def test_negative_control_sibling_is_valid_same_length():
    doc = CS.load()
    for n in doc["negative_controls"]:
        base = {c["case_id"]: c for c in doc["cases"]}[n["base_case"]]
        cy = base["cycles"][n["cycle"]]
        assert cy["L"] >= 2
        if n["mutation"] == "NC_SIB":
            sib = Q.validate_path(n["sibling_nodes"])
            assert len(sib) == len(cy["nodes"]) and sib != cy["nodes"] and sib[:-1] == cy["nodes"][:-1]


def test_plan_orders_and_cold_first_case():
    doc = CS.load()
    a = CL.plan(doc, "A", "r"); c = CL.plan(doc, "CAND", "r")
    assert len(a) == 33 and len(c) == 33 + 1 + 10
    firsts = [o for o in a if o["cache_salt"]]
    assert len(firsts) == 7 and all(o["o0_mode"] == "capture" and o["natural_digest"] for o in firsts)
    assert [o["mutation"]["mutation"] for o in c if o["mutation"]][-2:] == ["NC_SIB", "NC_SIB"]


# ------------------------------------------------------------------ patchers / launch generation
@pytest.mark.skipif(not os.path.exists(STOCK), reason="captured stock runner not present")
def test_native_patch_anchors(tmp_path):
    rec = PR.run("native", STOCK, None, False, None)
    assert rec["problems"] == [] and rec["files"]["runner"]["matches_reference_copy"]
    dst = tmp_path / "runner.py"; dst.write_text(open(STOCK).read())
    assert PR.run("native", str(dst), None, True, None)["applied"]
    assert dst.read_text().count(PR.MARK) == 4
    again = PR.run("native", str(dst), None, True, None)
    assert not again["applied"] and any("double patch" in p for p in again["problems"])


@pytest.mark.skipif(not os.path.exists(GEN), reason="captured generated sources not present")
def test_cand_patch_anchors(tmp_path):
    r, j = tmp_path / "r.py", tmp_path / "j.py"
    r.write_text(open(f"{GEN}/gpu_model_runner.patched.py").read()); j.write_text(open(f"{GEN}/rejection_sampler.patched.py").read())
    rec = PR.run("cand", str(r), str(j), True, None)
    assert rec["applied"] and rec["problems"] == []
    assert r.read_text().count(PR.MARK) == 6 and j.read_text().count(PR.MARK) == 2
    assert j.read_text().startswith("# LUMO_TREE_PATH_LCP_MAX\nimport q1v3_hooks")


def test_make_cand_launch(tmp_path):
    if not all(os.path.exists(p) for p, _ in MK.SOURCES.values()):
        pytest.skip("launcher sources not present")
    rec = MK.generate(str(tmp_path))
    for f in rec["files"].values():
        assert subprocess.run(["bash", "-n", f["generated"]]).returncode == 0
    launcher = open(rec["files"]["launcher"]["generated"]).read()
    assert launcher.count("python3 /q1v3/patch_runner.py --mode cand --apply") == 1
    assert '"KV_CACHE_MEMORY_BYTES|$KV_CACHE_MEMORY_BYTES|$_fixed32_expected_kv_cache_memory_bytes"' in launcher   # B1 pin rule untouched


# ------------------------------------------------------------------ paged state access
@pytest.mark.parametrize("page_src,page_dst", [(4, 16), (64, 1024), (16, 4)])
def test_kv_reblock_roundtrip(page_src, page_dst):
    n, P = 40, 3 * page_dst + 5
    def cache(page):
        nb = P // page + 4
        return torch.zeros(nb, 2, page, 1, 2, dtype=torch.bfloat16).permute(1, 0, 2, 3, 4), list(range(1, nb))[::-1]
    a, ra = cache(page_src); b, rb = cache(page_dst)
    src = torch.randn(2, P, 1, 2).to(torch.bfloat16)
    Q.write_kv(a, ra, page_src, 0, src)
    logical = Q.read_kv(a, ra, page_src, 0, P)
    assert torch.equal(logical, src)
    Q.write_kv(b, rb, page_dst, 0, logical)
    assert torch.equal(Q.read_kv(b, rb, page_dst, 0, P), src)
    assert torch.equal(Q.read_kv(b, rb, page_dst, 7, 9), src[:, 7:16])
    assert float(b[:, 0].abs().sum()) == 0.0                                    # null block untouched


def test_gdn_rows_and_pad():
    conv = torch.zeros(5, 34, 6, dtype=torch.bfloat16); ssm = torch.zeros(5, 2, 3, 3)
    c, s = torch.randn(3, 6).to(torch.bfloat16), torch.randn(2, 3, 3)
    Q.write_gdn(conv, ssm, 2, c, s)
    rc, rs = Q.read_gdn(conv, ssm, 2)
    assert torch.equal(rc, c) and torch.equal(rs, s) and float(conv[2, 3:].abs().sum()) == 0
    with pytest.raises(RuntimeError):
        Q.read_gdn(conv, ssm, 0)


def test_metrics():
    a = torch.tensor([1.0, 3.0, 3.0, 0.5])
    assert Q.greedy(a) == 1 and Q.top_margin(a) == 0.0 and Q.kl_div(a, a) == 0.0
    assert Q.rel_l2(a * 1.01, a) == pytest.approx(0.01, rel=1e-6)
    assert Q.topk_ids(torch.tensor([0.0, 2.0, 2.0, 1.0]), 2) == [1, 2]


# ------------------------------------------------------------------ end-to-end on the CPU engine stand-in
def _synthetic_doc(tmp):
    pool = list(range(11, 400))
    prefixes, cases = {}, []
    for key, block, held, prompt in (("cal-short", "calibration", False, list(range(100, 110))), ("eval-short", "evaluation", True, list(range(200, 212)))):
        req = os.path.join(tmp, f"{key}.json"); json.dump({"messages": [], "tools": None}, open(req, "w"))
        prefixes[key] = {"block": block, "held_out": held, "request_file": req, "request_sha256": Q.sha256_file(req), "reference_prompt_len": len(prompt),
                         "reference_prompt_sha256": "x", "prompt": prompt}
        for s in ("C2-nc-base", "C1-spine-bonus"):
            cases.append(CS.build_case(f"{key}__{s}", key, pool, CS.SCHEDULES[s]))
    ncs = [{"nc_id": f"cal-short__{n}", "base_case": "cal-short__C2-nc-base", "mutation": n, "cycle": 1, "description": d,
            "must_fail_surfaces_at_cycle": t, "sibling_nodes": CS.NC_SIBLING_PATH if n == "NC_SIB" else None} for n, d, t in CS.NEGATIVE_CONTROLS]
    doc = {"schema": "q1v3.cases.v1", "prefixes": prefixes, "cases": cases, "negative_controls": ncs, "candidate_repeats": ["cal-short__C2-nc-base"]}
    doc["record_sha256"] = Q.record_digest(doc)
    path = os.path.join(tmp, "cases.json"); json.dump(doc, open(path, "w"))
    return doc, path


def _run_arm(monkeypatch, tmp, doc, cases_path, arm, bug=None):
    import q1v3_hooks as HK
    run = os.path.join(tmp, "run")
    out = os.path.join(run, arm); os.makedirs(out, exist_ok=True)
    kind = "cand" if arm == "CAND" else "native"
    for k, v in {"Q1V3_ARM_KIND": kind, "Q1V3_ARM": arm, "Q1V3_OUT": out, "Q1V3_RUN": run, "Q1V3_CASES": cases_path}.items():
        monkeypatch.setenv(k, v)
    hooks = HK.Hooks()
    by = {c["case_id"]: c for c in doc["cases"]}
    for o in CL.plan(doc, arm, "run"):
        Q.write_json(os.path.join(out, "control.json"), dict(o, seq=0))
        prompt = doc["prefixes"][by[o["case_id"]]["prefix"]]["prompt"]
        eng = SIM.Engine(kind, prompt, page=4 if kind == "native" else 16, taps=3 if kind == "native" else 34,
                         nblocks=40 if kind == "native" else 12, bug=bug)
        emitted = eng.run(hooks)
        rec = json.load(open(os.path.join(out, "cases", f"{o['obs_id']}.json")))
        assert rec["valid"], (o["obs_id"], rec["problems"])
        if not o["mutation"] or o["mutation"]["mutation"] != "NC_STALE":
            assert emitted == by[o["case_id"]]["output_tokens"], o["obs_id"]
    return run


@pytest.fixture()
def synth(tmp_path):
    doc, path = _synthetic_doc(str(tmp_path))
    return str(tmp_path), doc, path


def test_end_to_end_correct_candidate_is_equivalent(monkeypatch, synth):
    tmp, doc, path = synth
    for arm in ("A", "B", "V", "CAND"):
        run = _run_arm(monkeypatch, tmp, doc, path, arm)
    V = R.reduce(run, doc, CL.plan)
    assert V["gates"] == {k: True for k in V["gates"]}, (V["reasons"], V["negative_controls"])
    assert V["verdict"] == "EQUIVALENT"
    assert V["reference"]["A_vs_B"] == "EXACT" and V["candidate"]["max_by_surface"]["gdn_rel"] == 0.0
    nc = {n["mutation"]: n for n in V["negative_controls"]}
    assert all(n["detected"] and n["cycle0_clean"] for n in nc.values())
    assert nc["NC_KV"]["target"]["kv"]["value"] > 0.02 and nc["NC_STALE"]["structural"]
    assert V["candidate_repeat"][0]["bitwise_equal_all_cycles"]


def test_end_to_end_replay_bug_is_not_equivalent(monkeypatch, synth):
    tmp, doc, path = synth
    for arm in ("A", "B", "V"):
        run = _run_arm(monkeypatch, tmp, doc, path, arm)
    run = _run_arm(monkeypatch, tmp, doc, path, "CAND", bug="replay_short")
    V = R.reduce(run, doc, CL.plan)
    assert V["verdict"] == "NOT_EQUIVALENT", V["reasons"]
    assert any("gdn_rel" in f for c in V["candidate"]["failing_cells"] for f in c["fails"])


def test_missing_arm_is_inconclusive(monkeypatch, synth):
    tmp, doc, path = synth
    for arm in ("A", "B", "CAND"):
        run = _run_arm(monkeypatch, tmp, doc, path, arm)
    V = R.reduce(run, doc, CL.plan)
    assert V["verdict"] == "INCONCLUSIVE" and not V["gates"]["R1_integrity"]
