"""Offline gate for the DEPLOYMENT temp-0.6 drift reducer (CPU, no GPU).

Proves cmd_deploy_temp06_drift is NON-VACUOUS:
  (1) id-aligned: q (spec verify) and p (recurrent oracle) are forced onto the
      SAME served stream, aligned by STEP INDEX + TOKEN ID, not string keys;
  (2) n_scored > 0 on banked synthetic q,p;
  (3) NEG-CONTROL: identical q==p -> TV ~ 0; pushed-apart q,p -> TV rises (the
      reducer measures real drift, it is not always-zero);
  (4) FAIL-LOUD: wrong schema / src mismatch / disengaged q or p are rejected.

It builds the q-sinks (verify_topk_*) + p-sinks (oracle_topk_*) on disk exactly
in the layout the GPU producers emit (capture-q-deploy qsinks, bigdenom rescore
<arm>_sinks), so the reducer's real sink-resolution path is exercised.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


measure = _load("fr13_measure", REPO / "scripts" / "fr13_measure.py")


# --------------------------------------------------------------------------- #
# synthetic stream builders                                                    #
# --------------------------------------------------------------------------- #
def _logsm(logits: dict[int, float]) -> dict[str, float]:
    """log_softmax over a small support, keyed by str(token_id)."""
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    z = sum(exps.values())
    return {str(k): math.log(e / z) for k, e in exps.items()}


def _served_lp(lp: dict[str, float], served: int) -> float:
    return lp.get(str(served), float("-inf"))  # served id may be outside support


def _write_q_sink(qdir: Path, pid: int, steps: list[dict]) -> None:
    qdir.mkdir(parents=True, exist_ok=True)
    with (qdir / f"p{pid}_rep0.jsonl").open("w", encoding="utf-8") as f:
        for st in steps:
            ids = sorted(st["logits"], key=lambda k: -st["logits"][k])
            lp = _logsm(st["logits"])
            f.write(json.dumps({
                "step": st["step"],
                "served_token_id": st["served"],
                "verify_argmax_id": ids[0],
                "verify_argmax_logprob": lp[str(ids[0])],
                "verify_served_logprob": _served_lp(lp, st["served"]),
                "verify_topk_ids": ids,
                "verify_topk_logprobs": [lp[str(i)] for i in ids],
            }) + "\n")


def _write_p_sink(pdir: Path, pid: int, steps: list[dict]) -> None:
    pdir.mkdir(parents=True, exist_ok=True)
    with (pdir / f"p{pid}_rep0.jsonl").open("w", encoding="utf-8") as f:
        for st in steps:
            ids = sorted(st["logits"], key=lambda k: -st["logits"][k])
            lp = _logsm(st["logits"])
            f.write(json.dumps({
                "step": st["step"],
                "served_token_id": st["served"],
                "oracle_argmax_id": ids[0],
                "oracle_argmax_logprob": lp[str(ids[0])],
                "served_logprob_in_oracle": _served_lp(lp, st["served"]),
                "oracle_topk_ids": ids,
                "oracle_topk_logprobs": [lp[str(i)] for i in ids],
            }) + "\n")


def _q_artifact(tmp: Path, arm: str, src: str, n_steps: int, qsink_dir: Path) -> Path:
    art = {
        "schema": "fr13.recurrent_decode_oracle.capture_q_deploy.v1",
        "arm": arm, "src": src, "regime": "deployment", "instrument": "ON",
        "q_id_keyed": True, "qsink_dir": str(qsink_dir),
        "total_positions": n_steps, "SPEC_VERIFY_ENGAGED": True,
        "per_prompt": [{"prompt_id": 0, "served_len": n_steps,
                        "n_q_steps": n_steps, "qsink": str(qsink_dir / "p0_rep0.jsonl")}],
    }
    pth = tmp / f"q_{arm}.json"
    pth.write_text(json.dumps(art), encoding="utf-8")
    return pth


def _p_artifact(tmp: Path, arm: str, src: str, n_steps: int) -> Path:
    # bigdenom layout: rescore artifact next to <arm>_sinks/ (sink_dir may be None).
    art = {
        "schema": "fr13.recurrent_decode_oracle.rescore.v1",
        "arm": arm, "src": src, "RECURRENT_PATH_ENGAGED": True,
        "sink_dir": None, "full_topk_all_positions": False,
        "total_positions": n_steps,
        "per_prompt": [{"prompt_id": 0, "served_len": n_steps,
                        "n_oracle_steps": n_steps, "positions": [], "flips": []}],
    }
    pth = tmp / f"rescore_{arm}.json"
    pth.write_text(json.dumps(art), encoding="utf-8")
    return pth


def _run(tmp: Path, q: Path, p: Path, **kw) -> dict:
    import argparse
    out = tmp / "drift.json"
    ns = argparse.Namespace(q=str(q), p=str(p), temp=0.6, per_position_floor=None,
                            native_floor_p95=None, dump_positions=True, out=str(out))
    for k, v in kw.items():
        setattr(ns, k, v)
    rc = measure.cmd_deploy_temp06_drift(ns)
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# tests                                                                        #
# --------------------------------------------------------------------------- #
def test_identical_q_p_drift_is_zero_neg_control(tmp_path):
    """NEG-CONTROL part A: q == p (same dist) => per-position TV ~ 0."""
    src = str(tmp_path / "arm_src.json")
    # IDENTICAL dists at every step.
    steps = [{"step": i, "served": 100,
              "logits": {100: 3.0, 200: 1.0, 300: 0.5}} for i in range(8)]
    _write_q_sink(tmp_path / "armX_qsinks", 0, steps)
    _write_p_sink(tmp_path / "armX_sinks", 0, steps)
    q = _q_artifact(tmp_path, "armX", src, 8, tmp_path / "armX_qsinks")
    p = _p_artifact(tmp_path, "armX", src, 8)
    res = _run(tmp_path, q, p)
    assert res["n_positions_scored"] == 8           # NON-VACUOUS
    assert res["mean_tv_q_p_at_temp"] < 1e-9        # identical => zero drift
    assert res["max_tv_q_p_at_temp"] < 1e-9
    assert res["non_vacuity"]["q_p_same_served_stream"] is True
    assert res["non_vacuity"]["n_positions_scored_gt0"] is True


def test_pushed_apart_q_p_drift_is_nonzero(tmp_path):
    """NEG-CONTROL part B: q and p diverge => TV RISES (reducer is not always-0)."""
    src = str(tmp_path / "arm_src.json")
    q_steps, p_steps = [], []
    for i in range(8):
        # q peaks on 100, p peaks on 200 -> real distributional drift.
        q_steps.append({"step": i, "served": 100,
                        "logits": {100: 4.0, 200: 0.0, 300: -1.0}})
        p_steps.append({"step": i, "served": 100,
                        "logits": {100: 0.0, 200: 4.0, 300: -1.0}})
    _write_q_sink(tmp_path / "armY_qsinks", 0, q_steps)
    _write_p_sink(tmp_path / "armY_sinks", 0, p_steps)
    q = _q_artifact(tmp_path, "armY", src, 8, tmp_path / "armY_qsinks")
    p = _p_artifact(tmp_path, "armY", src, 8)
    res = _run(tmp_path, q, p)
    assert res["n_positions_scored"] == 8
    assert res["mean_tv_q_p_at_temp"] > 0.3         # large, real drift
    # depth-matched floor verdict: a small floor -> ABOVE; a big floor -> WITHIN.
    res_above = _run(tmp_path, q, p, native_floor_p95=0.01)
    assert res_above["within_floor_verdict"] == "ABOVE_floor"
    res_within = _run(tmp_path, q, p, native_floor_p95=1.0)
    assert res_within["within_floor_verdict"] == "WITHIN_floor"


def test_id_alignment_not_positional_only(tmp_path):
    """q and p with DIFFERENT support sets at a step still align by TOKEN ID
    (the union TV counts mass on the symmetric difference, not a positional zip)."""
    src = str(tmp_path / "arm_src.json")
    # step 0: q support {100,200}, p support {200,300}; only id 200 overlaps.
    q_steps = [{"step": 0, "served": 100, "logits": {100: 2.0, 200: 0.0}}]
    p_steps = [{"step": 0, "served": 100, "logits": {200: 2.0, 300: 0.0}}]
    _write_q_sink(tmp_path / "armZ_qsinks", 0, q_steps)
    _write_p_sink(tmp_path / "armZ_sinks", 0, p_steps)
    q = _q_artifact(tmp_path, "armZ", src, 1, tmp_path / "armZ_qsinks")
    p = _p_artifact(tmp_path, "armZ", src, 1)
    res = _run(tmp_path, q, p)
    pos = res["forks"][0]["positions"][0]
    # id-keyed union: q has 100,200; p has 200,300; overlap is {200} only.
    assert pos["n_support_overlap"] == 1
    assert pos["q_support"] == 2 and pos["p_support"] == 2
    assert pos["tv_q_p_at_temp"] > 0.0


def test_temp_scaling_changes_tv(tmp_path):
    """The /0.6 temp scaling is applied (T=1 vs T=0.6 give different TV)."""
    src = str(tmp_path / "arm_src.json")
    q_steps = [{"step": 0, "served": 100, "logits": {100: 1.0, 200: 0.0}}]
    p_steps = [{"step": 0, "served": 100, "logits": {100: 0.0, 200: 1.0}}]
    _write_q_sink(tmp_path / "armT_qsinks", 0, q_steps)
    _write_p_sink(tmp_path / "armT_sinks", 0, p_steps)
    q = _q_artifact(tmp_path, "armT", src, 1, tmp_path / "armT_qsinks")
    p = _p_artifact(tmp_path, "armT", src, 1)
    r06 = _run(tmp_path, q, p, temp=0.6)
    r10 = _run(tmp_path, q, p, temp=1.0)
    # sharper temp (0.6) makes the peaked dists MORE separated -> larger TV.
    assert r06["mean_tv_q_p_at_temp"] > r10["mean_tv_q_p_at_temp"]


def test_fail_loud_wrong_q_schema(tmp_path):
    src = str(tmp_path / "s.json")
    steps = [{"step": 0, "served": 1, "logits": {1: 1.0, 2: 0.0}}]
    _write_q_sink(tmp_path / "armW_qsinks", 0, steps)
    _write_p_sink(tmp_path / "armW_sinks", 0, steps)
    q = _q_artifact(tmp_path, "armW", src, 1, tmp_path / "armW_qsinks")
    bad = json.loads(q.read_text())
    bad["schema"] = "fr13.measure.capture_q.v1"  # the OFF-distribution q, not deploy
    q.write_text(json.dumps(bad), encoding="utf-8")
    p = _p_artifact(tmp_path, "armW", src, 1)
    with pytest.raises(RuntimeError, match="capture_q_deploy"):
        _run(tmp_path, q, p)


def test_fail_loud_src_mismatch(tmp_path):
    steps = [{"step": 0, "served": 1, "logits": {1: 1.0, 2: 0.0}}]
    _write_q_sink(tmp_path / "armM_qsinks", 0, steps)
    _write_p_sink(tmp_path / "armM_sinks", 0, steps)
    q = _q_artifact(tmp_path, "armM", "/srcA.json", 1, tmp_path / "armM_qsinks")
    p = _p_artifact(tmp_path, "armM", "/srcB.json", 1)  # DIFFERENT stream
    with pytest.raises(RuntimeError, match="q.src != p.src"):
        _run(tmp_path, q, p)


def test_fail_loud_q_disengaged(tmp_path):
    src = str(tmp_path / "s.json")
    steps = [{"step": 0, "served": 1, "logits": {1: 1.0, 2: 0.0}}]
    _write_q_sink(tmp_path / "armD_qsinks", 0, steps)
    _write_p_sink(tmp_path / "armD_sinks", 0, steps)
    q = _q_artifact(tmp_path, "armD", src, 1, tmp_path / "armD_qsinks")
    bad = json.loads(q.read_text())
    bad["SPEC_VERIFY_ENGAGED"] = False
    q.write_text(json.dumps(bad), encoding="utf-8")
    p = _p_artifact(tmp_path, "armD", src, 1)
    with pytest.raises(RuntimeError, match="SPEC_VERIFY_ENGAGED"):
        _run(tmp_path, q, p)


def test_fail_loud_p_not_recurrent(tmp_path):
    src = str(tmp_path / "s.json")
    steps = [{"step": 0, "served": 1, "logits": {1: 1.0, 2: 0.0}}]
    _write_q_sink(tmp_path / "armR_qsinks", 0, steps)
    _write_p_sink(tmp_path / "armR_sinks", 0, steps)
    q = _q_artifact(tmp_path, "armR", src, 1, tmp_path / "armR_qsinks")
    p = _p_artifact(tmp_path, "armR", src, 1)
    bad = json.loads(p.read_text())
    bad["RECURRENT_PATH_ENGAGED"] = False
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="RECURRENT_PATH_ENGAGED"):
        _run(tmp_path, q, p)
