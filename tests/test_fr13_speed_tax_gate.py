from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gate = _load_script("fr13_speed_tax_gate", REPO / "scripts" / "fr13_speed_tax_gate.py")


# ---------------------------------------------------------------------------
# fixtures: synthetic probe run dirs (the canonical probe output shape)
# ---------------------------------------------------------------------------

PROMPTS = ["p one", "p two", "p three", "p four"]


def _probe_doc(
    *,
    mode: str,
    decode_seconds: float,
    spec_drafts: float,
    spec_draft_tokens: float,
    accept_per_event: float,
    seed: int | None = 1313,
    prompts: list[str] | None = None,
    with_metric_delta: bool = True,
) -> dict:
    summary = {
        "mode": mode,
        "accepted_per_draft_event": accept_per_event,
        "accepted_per_draft_token": 0.2,
        "warm_decode_tps": 7.65,
        "per_request_decode_tps_mean": 6.81,
        "per_request_decode_tps_median": 6.79,
    }
    if with_metric_delta:
        summary["metric_delta"] = {
            "decode_seconds": decode_seconds,
            "generation_tokens": 512.0,
            "spec_accepted_tokens": accept_per_event * spec_drafts,
            "spec_draft_tokens": spec_draft_tokens,
            "spec_drafts": spec_drafts,
        }
    return {
        "schema": "fr10.quick_decode_tps.v1",
        "seed": seed,
        "temperature": 0.0,
        "top_p": 1.0,
        "batch_size": 1,
        "max_tokens": 128,
        "samples_per_prompt": 1,
        "mode": mode,
        "prompts": prompts if prompts is not None else list(PROMPTS),
        "modes": {mode: summary},
        "summary": summary,
    }


def _write_arm(tmp_path: Path, window: str, doc: dict, campaign_header: dict | None = None) -> Path:
    run_dir = tmp_path / window
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{window}_probe.json").write_text(json.dumps(doc), encoding="utf-8")
    if campaign_header is not None:
        (tmp_path / "run_header.json").write_text(
            json.dumps(campaign_header), encoding="utf-8"
        )
    return run_dir


CAMPAIGN_HEADER = {
    "run": "synthetic",
    "boots": {
        "boot_tree": {
            "BATCH_INVARIANT": 1,
            "FR13_BI_TREE_ATTN": 1,
            "TREE": "[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]",
            "windows": ["tree_win"],
        },
        "boot_native": {
            "BATCH_INVARIANT": 1,
            "windows": ["native_win"],
        },
    },
}


def _reduce(tmp_path: Path, argv: list[str]) -> dict:
    out = tmp_path / "reduce.json"
    rc = gate.main(["reduce", *argv, "--out", str(out)])
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _standard_pair(tmp_path: Path) -> tuple[Path, Path]:
    native = _write_arm(
        tmp_path,
        "native_win",
        _probe_doc(
            mode="naive_mtp",
            decode_seconds=27.0,
            spec_drafts=127.0,
            spec_draft_tokens=635.0,
            accept_per_event=3.047,
        ),
        campaign_header=CAMPAIGN_HEADER,
    )
    tree = _write_arm(
        tmp_path,
        "tree_win",
        _probe_doc(
            mode="tree_mtp",
            decode_seconds=66.9,
            spec_drafts=170.0,
            spec_draft_tokens=1530.0,
            accept_per_event=2.024,
        ),
    )
    return native, tree


# ---------------------------------------------------------------------------
# 1. hand-roll guard
# ---------------------------------------------------------------------------


def test_handroll_guard_rejects_tps_accept_inputs() -> None:
    with pytest.raises(gate.HandRollGuardError):
        gate.measured_per_forward(
            {"warm_decode_tps": 7.65, "accepted_per_draft_event": 2.02},
            source="probe_metric_delta",
        )


def test_handroll_guard_rejects_non_metrics_source() -> None:
    with pytest.raises(gate.HandRollGuardError):
        gate.measured_per_forward(
            {"decode_seconds": 10.0, "spec_drafts": 100.0},
            source="tps_divided_by_accept",
        )


def test_missing_metrics_emits_unavailable_never_falls_back(tmp_path: Path) -> None:
    # arm has TPS + accept in the probe summary but NO metric_delta and no
    # before/after snapshots: per-forward must be UNAVAILABLE, not TPS/accept.
    native, _ = _standard_pair(tmp_path)
    no_metrics = _write_arm(
        tmp_path,
        "no_metrics_win",
        _probe_doc(
            mode="tree_mtp",
            decode_seconds=0.0,
            spec_drafts=0.0,
            spec_draft_tokens=0.0,
            accept_per_event=2.0,
            with_metric_delta=False,
        ),
    )
    doc = _reduce(
        tmp_path,
        [
            "--arm",
            f"native={native}",
            "--arm",
            f"nm={no_metrics}",
            "--baseline",
            "native",
            "--route",
            "nm=legacy",
            "--topology",
            "nm=caterpillar9",
        ],
    )
    arm = doc["arms"]["nm"]
    assert arm["per_forward"]["status"] == "UNAVAILABLE"
    assert "seconds_per_forward_sum_basis" not in arm["per_forward"]
    assert arm["ratio_vs_baseline"]["status"] == "UNAVAILABLE"
    row = [r for r in doc["ladder_rows"] if "arm=nm" in r][0]
    assert "UNAVAILABLE" in row
    # raw TPS/accept are still reported as raw numbers, never as per-forward
    assert arm["raw"]["warm_decode_tps"] == 7.65


# ---------------------------------------------------------------------------
# 2. reduce on synthetic fixtures: per-forward basis + ratio + validity
# ---------------------------------------------------------------------------


def test_reduce_per_forward_and_ratio(tmp_path: Path) -> None:
    native, tree = _standard_pair(tmp_path)
    doc = _reduce(
        tmp_path,
        [
            "--arm",
            f"native={native}",
            "--arm",
            f"tree={tree}",
            "--baseline",
            "native",
            "--route",
            "tree=legacy",
            "--topology",
            "tree=caterpillar9",
            "--flags",
            "tree=BATCH_INVARIANT=1",
        ],
    )
    tree_arm = doc["arms"]["tree"]
    native_arm = doc["arms"]["native"]
    pf_t = tree_arm["per_forward"]
    pf_n = native_arm["per_forward"]
    assert pf_t["status"] == "MEASURED"
    assert pf_t["seconds_per_forward_sum_basis"] == pytest.approx(66.9 / 170.0)
    assert pf_t["tokens_per_draft"] == pytest.approx(9.0)
    assert pf_t["drafts_equal_forwards"] is True
    assert pf_n["seconds_per_forward_sum_basis"] == pytest.approx(27.0 / 127.0)
    assert pf_n["tokens_per_draft"] == pytest.approx(5.0)
    ratio = tree_arm["ratio_vs_baseline"]
    assert ratio["status"] == "MEASURED_RATIO"
    assert ratio["ratio_vs_baseline"] == pytest.approx((66.9 / 170.0) / (27.0 / 127.0))


def test_reduce_refuses_ratio_on_prompt_mismatch(tmp_path: Path) -> None:
    native, _ = _standard_pair(tmp_path)
    other_prompts = _write_arm(
        tmp_path,
        "other_prompts_win",
        _probe_doc(
            mode="tree_mtp",
            decode_seconds=60.0,
            spec_drafts=150.0,
            spec_draft_tokens=1350.0,
            accept_per_event=2.0,
            prompts=["different prompt"],
        ),
    )
    doc = _reduce(
        tmp_path,
        [
            "--arm",
            f"native={native}",
            "--arm",
            f"bad={other_prompts}",
            "--baseline",
            "native",
            "--route",
            "bad=legacy",
            "--topology",
            "bad=caterpillar9",
            "--flags",
            "bad=BATCH_INVARIANT=1",
        ],
    )
    ratio = doc["arms"]["bad"]["ratio_vs_baseline"]
    assert ratio["status"] == "REFUSED"
    assert any("prompts_sha256" in f for f in ratio["pairing"]["failures"])


def test_reduce_refuses_ratio_on_bi_mismatch(tmp_path: Path) -> None:
    native, tree = _standard_pair(tmp_path)
    doc = _reduce(
        tmp_path,
        [
            "--arm",
            f"native={native}",
            "--arm",
            f"tree={tree}",
            "--baseline",
            "native",
            "--route",
            "tree=legacy",
            "--topology",
            "tree=caterpillar9",
            "--flags",
            "tree=BATCH_INVARIANT=0",
        ],
    )
    ratio = doc["arms"]["tree"]["ratio_vs_baseline"]
    assert ratio["status"] == "REFUSED"
    assert any("batch_invariant" in f for f in ratio["pairing"]["failures"])


def test_drafts_not_equal_forwards_invalidates_ratio(tmp_path: Path) -> None:
    native, _ = _standard_pair(tmp_path)
    # 1531/170 is not an integer -> drafts==forwards proof fails
    bad = _write_arm(
        tmp_path,
        "bad_drafts_win",
        _probe_doc(
            mode="tree_mtp",
            decode_seconds=66.9,
            spec_drafts=170.0,
            spec_draft_tokens=1531.0,
            accept_per_event=2.0,
        ),
    )
    doc = _reduce(
        tmp_path,
        [
            "--arm",
            f"native={native}",
            "--arm",
            f"bad={bad}",
            "--baseline",
            "native",
            "--route",
            "bad=legacy",
            "--topology",
            "bad=caterpillar9",
            "--flags",
            "bad=BATCH_INVARIANT=1",
        ],
    )
    arm = doc["arms"]["bad"]
    assert arm["per_forward"]["drafts_equal_forwards"] is False
    assert arm["ratio_vs_baseline"]["status"] == "REFUSED"


# ---------------------------------------------------------------------------
# 3. raw metrics_before/metrics_after files take precedence
# ---------------------------------------------------------------------------

PROM_TEMPLATE = """# HELP vllm:request_decode_time_seconds Histogram
vllm:request_decode_time_seconds_sum{{engine="0",model_name="m"}} {decode}
vllm:spec_decode_num_drafts_total{{engine="0",model_name="m"}} {drafts}
vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="m"}} {draft_tokens}
vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="m"}} {accepted}
vllm:generation_tokens_total{{engine="0",model_name="m"}} {gen}
"""


def test_metrics_files_precedence_and_parsing(tmp_path: Path) -> None:
    native, tree = _standard_pair(tmp_path)
    before = tmp_path / "metrics_before.txt"
    after = tmp_path / "metrics_after.txt"
    before.write_text(
        PROM_TEMPLATE.format(decode=10.0, drafts=10.0, draft_tokens=90.0, accepted=20.0, gen=50.0),
        encoding="utf-8",
    )
    after.write_text(
        PROM_TEMPLATE.format(decode=50.0, drafts=110.0, draft_tokens=990.0, accepted=220.0, gen=550.0),
        encoding="utf-8",
    )
    doc = _reduce(
        tmp_path,
        [
            "--arm",
            f"native={native}",
            "--arm",
            f"tree={tree}",
            "--baseline",
            "native",
            "--route",
            "tree=legacy",
            "--topology",
            "tree=caterpillar9",
            "--flags",
            "tree=BATCH_INVARIANT=1",
            "--metrics-before",
            f"tree={before}",
            "--metrics-after",
            f"tree={after}",
        ],
    )
    pf = doc["arms"]["tree"]["per_forward"]
    assert pf["source"] == "metrics_files"
    # delta: decode 40s over 100 drafts, 900 draft tokens -> 9 tokens/draft
    assert pf["seconds_per_forward_sum_basis"] == pytest.approx(0.4)
    assert pf["tokens_per_draft"] == pytest.approx(9.0)
    assert pf["drafts_equal_forwards"] is True


# ---------------------------------------------------------------------------
# 4. header completeness (class 9: flag-state + seed headers)
# ---------------------------------------------------------------------------


def test_header_completeness_flags_missing_seed(tmp_path: Path) -> None:
    native, _ = _standard_pair(tmp_path)
    seedless = _write_arm(
        tmp_path,
        "seedless_win",
        _probe_doc(
            mode="tree_mtp",
            decode_seconds=66.9,
            spec_drafts=170.0,
            spec_draft_tokens=1530.0,
            accept_per_event=2.0,
            seed=None,
        ),
    )
    doc = _reduce(
        tmp_path,
        [
            "--arm",
            f"native={native}",
            "--arm",
            f"sl={seedless}",
            "--baseline",
            "native",
            "--route",
            "sl=legacy",
            "--topology",
            "sl=caterpillar9",
            "--flags",
            "sl=BATCH_INVARIANT=1",
        ],
    )
    header = doc["arms"]["sl"]["header"]
    assert header["header_complete"] is False
    assert "seed" in header["header_missing"]
    assert "sl" in doc["header_incomplete_arms"]
    row = [r for r in doc["ladder_rows"] if "arm=sl" in r][0]
    assert "HDR-INCOMPLETE" in row


def test_strict_headers_exits_nonzero(tmp_path: Path) -> None:
    native, tree = _standard_pair(tmp_path)
    out = tmp_path / "strict.json"
    rc = gate.main(
        [
            "reduce",
            "--arm",
            f"native={native}",
            "--arm",
            f"tree={tree}",
            "--baseline",
            "native",
            # tree gets no --route/--flags -> route+BI missing -> incomplete
            "--strict-headers",
            "--out",
            str(out),
        ]
    )
    assert rc == 2


# ---------------------------------------------------------------------------
# 5. topology format + N_PAD cap
# ---------------------------------------------------------------------------


def test_topology_n_pad_math() -> None:
    cases = {
        "chain5": (5, 5, 8, True),
        "caterpillar9": (9, 5, 16, True),
        "caterpillar12_w3_d5": (12, 5, 16, True),
        "caterpillar13_d6": (13, 6, 16, True),
        "caterpillar15_w3_d6": (15, 6, 16, True),
        "node16_REJECTED": (16, 6, 32, False),
    }
    for shape, (n, depth, n_pad, allowed) in cases.items():
        paths = gate.SHAPE_CATALOG[shape]
        gate.validate_topology(paths)
        stats = gate.topology_stats(paths)
        assert stats["num_draft_nodes"] == n, shape
        assert stats["depth"] == depth, shape
        assert stats["n_pad"] == n_pad, shape
        assert stats["allowed_by_n_pad_cap"] is allowed, shape


def test_tree_literal_roundtrip_matches_deployed_caterpillar() -> None:
    deployed = "[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]"
    assert gate.parse_tree_literal(deployed) == gate.SHAPE_CATALOG["caterpillar9"]
    assert gate.parse_tree_literal("caterpillar " + deployed) == gate.SHAPE_CATALOG[
        "caterpillar9"
    ]
    assert gate.tree_literal(gate.SHAPE_CATALOG["caterpillar9"]) == deployed


def test_validate_topology_rejects_orphans_and_sibling_gaps() -> None:
    with pytest.raises(ValueError):
        gate.validate_topology([(0,), (0, 0, 0)])  # missing parent (0,0)
    with pytest.raises(ValueError):
        gate.validate_topology([(0,), (0, 2)])  # skips child index 1


# ---------------------------------------------------------------------------
# 6. traffic model
# ---------------------------------------------------------------------------


def test_traffic_model_rows() -> None:
    # legacy at N=9 = 27 + 2a + 1 (w78aq6xum)
    assert gate.predicted_state_rows("legacy", 9, 2.0) == pytest.approx(32.0)
    assert gate.predicted_state_rows("replay", 9, 2.0) == pytest.approx(4.0)
    # replay is N-invariant
    assert gate.predicted_state_rows("replay", 15, 2.0) == gate.predicted_state_rows(
        "replay", 5, 2.0
    )
    assert gate.predicted_state_rows("native", None, None) == pytest.approx(7.0)


def test_traffic_model_gb_and_floor() -> None:
    tm = gate.traffic_model("native", None, None)
    assert tm["status"] == "MODEL"
    assert tm["state_gb_per_fwd"] == pytest.approx(7 * 3.146e-3 * 48)
    assert tm["total_gb_per_fwd"] == pytest.approx(27.0 + 7 * 3.146e-3 * 48)
    assert tm["predicted_floor_ms_per_fwd"] == pytest.approx(
        (27.0 + 7 * 3.146e-3 * 48) / 273.0 * 1000.0
    )
    # weights-only floor constant is carried through
    assert tm["weights_only_floor_ms"] == pytest.approx(98.9)


# ---------------------------------------------------------------------------
# 7. sweep matrix generation
# ---------------------------------------------------------------------------


def test_sweep_matrix_caps_and_routes(tmp_path: Path) -> None:
    out_dir = tmp_path / "sweep"
    rc = gate.main(["sweep", "--out-dir", str(out_dir)])
    assert rc == 0
    matrix = json.loads((out_dir / "sweep_matrix.json").read_text(encoding="utf-8"))
    assert matrix["n_pad_cap"]["cap"] == 16
    shapes = matrix["shapes"]
    assert shapes["node16_REJECTED"]["allowed_by_n_pad_cap"] is False
    # rejected shape produces no arms
    arm_shapes = {a["shape"] for a in matrix["arms"]}
    assert "node16_REJECTED" not in arm_shapes
    # each allowed shape gets legacy + replay arms
    for shape in ("chain5", "caterpillar9", "caterpillar13_d6", "caterpillar15_w3_d6"):
        routes = {a["route"] for a in matrix["arms"] if a["shape"] == shape}
        assert routes == {"legacy", "replay"}
    # replay arms carry the post-merge label (accept bug FIXED, default ON);
    # BOTH routes pin FR13_REPLAY_ROUTE explicitly so the sweep stays
    # unambiguous under either launcher default.
    for arm in matrix["arms"]:
        if arm["route"] == "replay":
            assert "FIXED" in arm["label"]
            assert arm["env"]["FR13_REPLAY_ROUTE"] == "1"
        else:
            assert arm["env"]["FR13_REPLAY_ROUTE"] == "0"
        assert arm["env"]["BATCH_INVARIANT"] == "1"
        assert arm["env"]["FR10_METRICS"] == "1"
    commands = (out_dir / "sweep_commands.sh").read_text(encoding="utf-8")
    assert "metrics_before.txt" in commands and "metrics_after.txt" in commands
    assert "merged to main + DEFAULT ON" in commands
    assert "native MTP-5 reference" in commands


# ---------------------------------------------------------------------------
# 8. fit: linear-in-N vs N-invariant
# ---------------------------------------------------------------------------


def _fit_reduce_doc(points: list[tuple[str, str, int, float, float]]) -> dict:
    """points: (arm, route, N, per_forward_s, accept)"""
    arms = {}
    for arm, route, n, pf, acc in points:
        topo = {"num_draft_nodes": n, "depth": 5, "verifier_n": n + 1, "n_pad": 16,
                "allowed_by_n_pad_cap": True, "n_pad_cap": 16}
        arms[arm] = {
            "header": {"route": route, "label": None},
            "topology_stats": None if route == "native" else topo,
            "raw": {"accepted_per_draft_event": acc},
            "per_forward": {
                "status": "MEASURED",
                "seconds_per_forward_sum_basis": pf,
                "drafts_equal_forwards": True,
            },
        }
    return {"schema": "fr13.speed_tax_gate.reduce.v1", "arms": arms}


def test_fit_discriminates_linear_vs_invariant(tmp_path: Path) -> None:
    # legacy points exactly linear in N; replay points exactly constant
    doc = _fit_reduce_doc(
        [
            ("native", "native", 5, 0.21, 3.0),
            ("leg5", "legacy", 5, 0.30, 2.2),
            ("leg9", "legacy", 9, 0.38, 2.0),
            ("leg13", "legacy", 13, 0.46, 2.0),
            ("leg15", "legacy", 15, 0.50, 2.0),
            ("rep5", "replay", 5, 0.251, 2.2),
            ("rep9", "replay", 9, 0.249, 2.0),
            ("rep13", "replay", 13, 0.250, 2.0),
            ("rep15", "replay", 15, 0.250, 2.0),
        ]
    )
    reduce_path = tmp_path / "r.json"
    reduce_path.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "fit.json"
    rc = gate.main(["fit", "--reduce", str(reduce_path), "--out", str(out)])
    assert rc == 0
    fit = json.loads(out.read_text(encoding="utf-8"))
    legacy = fit["fits"]["legacy"]
    replay = fit["fits"]["replay"]
    assert legacy["preferred_model_by_dof_adjusted_mse"] == "linear_in_N"
    assert legacy["linear_in_N_fit"]["slope_s_per_node"] == pytest.approx(0.02, abs=1e-9)
    assert "slope_ci95_normal_approx" in legacy["linear_in_N_fit"]
    assert replay["preferred_model_by_dof_adjusted_mse"] == "N_invariant_constant"
    assert replay["N_invariant_fit"]["constant_s"] == pytest.approx(0.25, abs=1e-3)
    # tax computed against the native reference
    assert fit["native_reference_seconds_per_forward"] == pytest.approx(0.21)
    assert legacy["points"][0]["tax_vs_native_s"] == pytest.approx(0.09)
    # the fit output is labeled as a fit, not a measurement
    assert "FIT" in fit["label"]


def test_fit_excludes_invalid_points(tmp_path: Path) -> None:
    doc = _fit_reduce_doc(
        [
            ("leg9", "legacy", 9, 0.38, 2.0),
            ("leg13", "legacy", 13, 0.46, 2.0),
        ]
    )
    doc["arms"]["leg13"]["per_forward"]["drafts_equal_forwards"] = False
    reduce_path = tmp_path / "r.json"
    reduce_path.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "fit.json"
    rc = gate.main(["fit", "--reduce", str(reduce_path), "--out", str(out)])
    assert rc == 0
    fit = json.loads(out.read_text(encoding="utf-8"))
    assert len(fit["fits"]["legacy"]["points"]) == 1
