"""The TAW native-production width-4 pair: hardening, attestation, runner, verdict.

Three things must hold or this screen is worthless, and each has its own section
below.

  1. AUTHORITATIVE ZERO. A declared-off arm must BE off. Before the hardening a
     stray sidecar in the container log directory could arm the candidate on an
     arm whose recorded environment said ``...PRODUCTION=0``, which destroys the
     single-variable delta in the direction that manufactures a null.
  2. ATTESTABLE ENGAGEMENT. The candidate arm must be able to PROVE it served
     and the stock arm must be able to prove it did not. Absence of an artifact
     is only evidence if the artifact is one the runtime would have written.
  3. ONE LEVER. FA2 became the registry production default at 32e240e15, so
     "leave FA2 alone" no longer holds FA2 fixed. Both arms must NAME the same
     FA2 production configuration or the pair measures two levers at once.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "scripts/fr13_run_b4_taw_width4_timing.sh"
REDUCER = REPO / "scripts/fr13_b4_taw_width4_pair_reduce.py"
KERNEL = REPO / "scripts/fr13_device_multidraft_kernel.py"
PATCHER = REPO / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
FA2_RUNNER = REPO / "scripts/fr13_run_b4_gqa_width4_timing.sh"
SEALED = REPO / "results/fr13_b4_width4_nsys_20260813/fr13_b4_batch_conditioned_wall.json"
ATTRIBUTION = REPO / "results/fr13_b4_width4_nsys_20260813/attribution_final.json"
SUBSET16 = REPO / "config/fr13_fixed32/subset_b4_sixteen.json"
POOL16_SHA = "47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c"
EXACT4_SHA = "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
PRODUCTION_ROUTE = "fixed32_native_precompute_production_candidate_return"
REFERENCE_ROUTE = "fixed32_pytorch_exact_float_triton_integer_commit"
DIAGNOSTIC_ROUTE = "fixed32_native_precompute_byte_ab_reference_return"
FAKE_COMMIT = "0" * 40


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


taw = _load("taw_for_width4_pair", KERNEL)


def _reducer():
    """One instance for the whole suite: two exec_module calls would give two
    distinct PairError classes and `pytest.raises` would stop matching."""
    module = sys.modules.get("taw_w4_pair_reduce")
    if module is None:
        module = _load("taw_w4_pair_reduce", REDUCER)
    return module


# ==========================================================================
# 1. authoritative zero
# ==========================================================================
def _arm_sources(environ: dict, sidecars: tuple[str, ...]) -> list[str]:
    return taw._fr13_fixed32_taw_native_arm_sources(
        environ=environ,
        name="FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION",
        sidecars=sidecars,
    )


def test_explicit_zero_vetoes_a_stray_sidecar(tmp_path: Path) -> None:
    """The fix. A declared-off arm stays off however the log directory looks."""
    sidecar = tmp_path / "production.arm"
    sidecar.write_text("1\n", encoding="ascii")
    assert _arm_sources(
        {"FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION": "0"}, (str(sidecar),)
    ) == []


def test_absent_env_still_lets_a_sidecar_arm(tmp_path: Path) -> None:
    """The PRESERVED semantics: unset delegates to the sidecar, as before."""
    sidecar = tmp_path / "production.arm"
    sidecar.write_text("1\n", encoding="ascii")
    assert _arm_sources({}, (str(sidecar),)) == [f"sidecar:{sidecar}"]


def test_explicit_one_arms_without_any_sidecar() -> None:
    assert _arm_sources(
        {"FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION": "1"}, ()
    ) == ["env:FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION"]


def test_zero_still_reports_a_malformed_sidecar(tmp_path: Path) -> None:
    """Zero removes a sidecar's power to ARM, never the report that it is wrong.

    A corrupted arm directory is still a corrupted arm directory; silently
    ignoring it under `0` would trade one blind spot for another.
    """
    sidecar = tmp_path / "production.arm"
    sidecar.write_text("2\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="sidecar must contain exactly 1"):
        _arm_sources(
            {"FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION": "0"}, (str(sidecar),)
        )


def test_a_non_binary_env_is_still_refused() -> None:
    with pytest.raises(RuntimeError, match="must be unset, 0, or 1"):
        _arm_sources({"FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION": "yes"}, ())


def test_selector_reports_reference_when_zero_vetoes_the_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the selector, which is what the commit path calls."""
    production = tmp_path / "production.arm"
    production.write_text("1\n", encoding="ascii")
    monkeypatch.setenv("FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.delenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", raising=False)
    monkeypatch.setenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION", "0")
    # No PASS bundle exists anywhere; if the sidecar could still arm production
    # the selector would raise looking for one instead of returning reference.
    assert (
        taw._fr13_fixed32_taw_native_selector(
            diagnostic_sidecars=(), production_sidecars=(str(production),)
        )
        == "reference"
    )


# ==========================================================================
# 2. the engagement artifact
# ==========================================================================
def _bundle_payload(mode: str = "tail6_fixed32", batches=(2, 3, 4)) -> dict:
    topology = taw._fr13_fixed32_topology()
    qualified = sorted(set(batches))
    record = {
        "schema": "fr13.fixed32.taw_native_precompute.live_pass.v2",
        "status": "pass",
        "candidate": taw._FR13_FIXED32_TAW_NATIVE_CANDIDATE,
        "source_contract_schema": taw._FR13_FIXED32_TAW_SOURCE_SCHEMA,
        "source_contract_sha256": taw._FR13_FIXED32_TAW_SOURCE_SHA256,
        "task_marker": "swe_verified:campaign4_" + "a" * 64,
        "mode": mode,
        "valid_mask": int(topology.VALID_MASK_BY_MODE[mode]),
        "topology_binding": taw._fr13_fixed32_taw_topology_binding(topology),
        "geometry": taw._FR13_FIXED32_TAW_GEOMETRY,
        "probability_mismatches": 0,
        "product_mismatches": 0,
        "evidence_route": "full_graph_replay",
        "reference_returned": True,
        "candidate_returned": False,
    }
    return {
        "schema": "fr13.fixed32.taw_native_precompute.pass_bundle.v1",
        "status": "production_ready",
        "candidate": taw._FR13_FIXED32_TAW_NATIVE_CANDIDATE,
        "source_contract_schema": taw._FR13_FIXED32_TAW_SOURCE_SCHEMA,
        "source_contract_sha256": taw._FR13_FIXED32_TAW_SOURCE_SHA256,
        "mode": mode,
        "valid_mask": int(topology.VALID_MASK_BY_MODE[mode]),
        "topology_binding": taw._fr13_fixed32_taw_topology_binding(topology),
        "required_production_batches": list(
            taw._FR13_FIXED32_TAW_REQUIRED_PRODUCTION_BATCHES
        ),
        "qualified_batches": qualified,
        "batch_passes": {
            str(batch): {**record, "batch_size": batch, "covered_batches": [batch]}
            for batch in qualified
        },
    }


def _write_bundle(path: Path, **kwargs) -> str:
    raw = json.dumps(_bundle_payload(**kwargs)).encode("ascii")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _events(routes_by_batch, *, pid: int = 4242, mode: str = "tail6_fixed32"):
    rows = []
    for batch, route, count in routes_by_batch:
        for _ in range(count):
            rows.append(
                {
                    "schema": "fr13-fixed32-work-census-v12",
                    "event_complete": True,
                    "event_index": len(rows),
                    "producer_pid": pid,
                    "mode": mode,
                    "batch_size": batch,
                    "taw": {"route": route},
                }
            )
    return rows


def _binding(events, *, pid: int = 4242, action: str = "final"):
    canonical = json.dumps(
        events, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return {
        "action": action,
        "boundary_snapshot_sha256": "b" * 64,
        "complete_work_census_events": len(events),
        "events_sha256": hashlib.sha256(canonical).hexdigest(),
        "generation": 1,
        "nonce": "c" * 64,
        "producer_pid": pid,
    }


@pytest.fixture()
def armed_production(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A production arm exactly as the launcher builds one, minus the GPU."""
    bundle = tmp_path / "production_pass.json"
    sha = _write_bundle(bundle)
    engagement = tmp_path / "engagement.json"
    monkeypatch.setenv("FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.delenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", raising=False)
    monkeypatch.setenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION", "1")
    monkeypatch.setenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_PATH", str(bundle)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION_ENGAGEMENT_JSON",
        str(engagement),
    )
    return {"bundle": bundle, "bundle_sha256": sha, "engagement": engagement}


def test_production_arm_writes_served_candidate_counts_per_batch(
    armed_production,
) -> None:
    events = _events(
        [
            (1, REFERENCE_ROUTE, 30),
            (2, PRODUCTION_ROUTE, 5),
            (3, PRODUCTION_ROUTE, 7),
            (4, PRODUCTION_ROUTE, 11),
        ]
    )
    record = taw.fr13_fixed32_taw_native_production_engagement_finalize(
        events, _binding(events)
    )
    assert record["status"] == "ENGAGED"
    assert record["schema"] == (
        "fr13.fixed32.taw_native_precompute.production_engagement.v1"
    )
    assert record["served_candidate_calls_by_batch"] == {"2": 5, "3": 7, "4": 11}
    assert record["served_candidate_calls"] == 23
    assert record["reference_calls_by_batch"] == {"1": 30}
    assert record["production_bundle_sha256"] == armed_production["bundle_sha256"]
    assert record["qualified_batches"] == [2, 3, 4]
    assert record["pinned_min_batch"] == taw._FR13_FIXED32_TAW_PINNED_MIN_BATCH
    assert record["candidate_returned"] is True
    assert record["observer_accounting_only"] is True
    assert record["performance_measurement"] is False
    written = json.loads(
        armed_production["engagement"].read_text(encoding="ascii")
    )
    assert written == record


def test_stock_arm_writes_nothing_and_that_absence_is_the_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engagement = tmp_path / "engagement.json"
    monkeypatch.setenv("FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.delenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", raising=False)
    monkeypatch.setenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION", "0")
    monkeypatch.setenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION_ENGAGEMENT_JSON",
        str(engagement),
    )
    events = _events([(4, REFERENCE_ROUTE, 12)])
    assert (
        taw.fr13_fixed32_taw_native_production_engagement_finalize(
            events, _binding(events)
        )
        is None
    )
    assert not engagement.exists()


def test_a_served_candidate_on_a_declared_off_arm_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The independent check that authoritative zero actually held."""
    monkeypatch.setenv("FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.delenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", raising=False)
    monkeypatch.setenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION", "0")
    monkeypatch.setenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION_ENGAGEMENT_JSON",
        str(tmp_path / "engagement.json"),
    )
    events = _events([(4, PRODUCTION_ROUTE, 3)])
    with pytest.raises(RuntimeError, match="production selector is off"):
        taw.fr13_fixed32_taw_native_production_engagement_finalize(
            events, _binding(events)
        )


def test_an_armed_arm_that_never_served_is_fatal(armed_production) -> None:
    events = _events([(1, REFERENCE_ROUTE, 40)])
    with pytest.raises(RuntimeError, match="never served the candidate"):
        taw.fr13_fixed32_taw_native_production_engagement_finalize(
            events, _binding(events)
        )


def test_a_diagnostic_route_on_the_production_arm_is_fatal(
    armed_production,
) -> None:
    events = _events(
        [(4, PRODUCTION_ROUTE, 3), (4, DIAGNOSTIC_ROUTE, 1)]
    )
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        taw.fr13_fixed32_taw_native_production_engagement_finalize(
            events, _binding(events)
        )


def test_a_served_batch_outside_the_qualified_scope_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "production_pass.json"
    _write_bundle(bundle, batches=(4,))
    monkeypatch.setenv("FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.delenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", raising=False)
    monkeypatch.setenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION", "1")
    monkeypatch.setenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_PATH", str(bundle)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION_ENGAGEMENT_JSON",
        str(tmp_path / "engagement.json"),
    )
    events = _events([(4, PRODUCTION_ROUTE, 3), (2, PRODUCTION_ROUTE, 1)])
    with pytest.raises(RuntimeError, match="outside the qualified scope"):
        taw.fr13_fixed32_taw_native_production_engagement_finalize(
            events, _binding(events)
        )


def test_engagement_refuses_a_flush_binding_that_is_not_the_final_one(
    armed_production,
) -> None:
    events = _events([(4, PRODUCTION_ROUTE, 3)])
    with pytest.raises(RuntimeError, match="flush drift"):
        taw.fr13_fixed32_taw_native_production_engagement_finalize(
            events, _binding(events, action="incremental")
        )


def test_engagement_refuses_events_that_do_not_hash_to_the_binding(
    armed_production,
) -> None:
    events = _events([(4, PRODUCTION_ROUTE, 3)])
    binding = _binding(events)
    binding["events_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="flush drift"):
        taw.fr13_fixed32_taw_native_production_engagement_finalize(events, binding)


def test_engagement_is_under_the_taw_source_contract() -> None:
    """Evidence that decides a verdict is pinned like the math it judges."""
    assert (
        "fr13_fixed32_taw_native_production_engagement_finalize"
        in taw._FR13_FIXED32_TAW_SOURCE_FUNCTIONS
    )


def test_the_three_routes_have_exactly_one_definition_each() -> None:
    """The publish site and the reader must not be able to drift apart."""
    text = KERNEL.read_text(encoding="utf-8")
    for constant, value in (
        ("_FR13_FIXED32_TAW_NATIVE_PRODUCTION_ROUTE", PRODUCTION_ROUTE),
        ("_FR13_FIXED32_TAW_NATIVE_DIAGNOSTIC_ROUTE", DIAGNOSTIC_ROUTE),
        ("_FR13_FIXED32_TAW_NATIVE_REFERENCE_ROUTE", REFERENCE_ROUTE),
    ):
        assert f'{constant} = (\n    "{value}"\n)' in text
        assert text.count(f'"{value}"') == 1, constant


def test_the_final_flush_actually_calls_the_emitter() -> None:
    """Six campaign fossils were runners bound to an artifact nothing wrote."""
    text = PATCHER.read_text(encoding="utf-8")
    assert (
        "taw_module.fr13_fixed32_taw_native_production_engagement_finalize(\n"
        "                    events, flush_binding\n"
        "                )"
    ) in text
    # and it must sit inside the final-flush branch, after the sibling finalizer
    cfwd = text.index("taw_module.fr13_fixed32_cfwd_logit_direct_live_finalize(")
    ours = text.index(
        "taw_module.fr13_fixed32_taw_native_production_engagement_finalize("
    )
    guard = text.rindex('if action == "final" and _gdn is not None:', 0, cfwd)
    assert guard < cfwd < ours


# ==========================================================================
# 3. runner wiring
# ==========================================================================
def test_runner_parses_and_is_executable() -> None:
    assert RUNNER.is_file() and not RUNNER.is_symlink()
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_runner_is_disabled_unless_explicitly_enabled() -> None:
    proc = subprocess.run(
        ["bash", str(RUNNER)], capture_output=True, text=True, cwd=REPO
    )
    assert proc.returncode == 2
    assert "disabled" in proc.stderr


@pytest.mark.parametrize("value", ["2", "yes", "01"])
def test_runner_refuses_a_non_binary_enable_flag(value: str) -> None:
    proc = subprocess.run(
        ["bash", str(RUNNER)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "FR13_RUN_B4_TAW_WIDTH4_TIMING": value},
    )
    assert proc.returncode == 2
    assert "must be exactly 0 or 1" in proc.stderr


@pytest.mark.parametrize(
    "missing",
    [
        "TAW_PRODUCTION_BUNDLE",
        "TAW_PRODUCTION_BUNDLE_SHA256",
        "TAW_BYTE_GATE_JSON",
        "TAW_BYTE_GATE_SHA256",
        "QROW32_GQA_PAIR_FA2_SO",
        "QROW32_GQA_PAIR_DUAL_GATE_JSON",
        "QROW32_GQA_PAIR_DUAL_GATE_SHA256",
    ],
)
def test_runner_refuses_a_missing_credential_input(missing: str) -> None:
    """A missing PASS bundle must cost seconds, not the stock arm's hours."""
    env = {
        "PATH": "/usr/bin:/bin",
        "FR13_RUN_B4_TAW_WIDTH4_TIMING": "1",
        "RUNROOT": str(REPO / "output/never_created"),
        "TAG": "unittest",
        "TAW_PRODUCTION_BUNDLE": "/nonexistent/bundle.json",
        "TAW_PRODUCTION_BUNDLE_SHA256": "0" * 64,
        "TAW_BYTE_GATE_JSON": "/nonexistent/gate.json",
        "TAW_BYTE_GATE_SHA256": "0" * 64,
        "QROW32_GQA_PAIR_FA2_SO": "/nonexistent/fa2.so",
        "QROW32_GQA_PAIR_DUAL_GATE_JSON": "/nonexistent/dual.json",
        "QROW32_GQA_PAIR_DUAL_GATE_SHA256": "0" * 64,
    }
    env.pop(missing)
    proc = subprocess.run(
        ["bash", str(RUNNER)], capture_output=True, text=True, cwd=REPO, env=env
    )
    assert proc.returncode != 0
    assert missing in proc.stderr


def test_runner_binds_the_sixteen_task_pool_not_exact4() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "config/fr13_fixed32/subset_b4_sixteen.json" in text
    assert f"SUBSET_SHA256={POOL16_SHA}" in text
    assert "subset_b4_four.json" not in text
    assert EXACT4_SHA not in text
    assert "TASK_COUNT=16" in text
    assert "astropy__astropy-14995" in text


def test_runner_serves_the_pool_regime_with_no_agent_wall() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "FR13_B4_TASK_REFILL=1" in text
    assert "AGENT_WALL_S= \\" in text, "the agent wall must be passed EMPTY"
    assert "AGENT_WALL_S=5400" not in text


def test_runner_keeps_the_qualified_geometry_pinned() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "MAX_NUM_SEQS_OVR=4" in text
    assert "SWE_CONCURRENCY=4" in text
    assert "export BSIZE=4" in text
    assert "export CONC=4" in text
    assert "CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in text
    assert "ENFORCE_EAGER=0" in text


def test_runner_differs_between_arms_in_exactly_one_variable() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'run_arm "$STOCK_ARM" 0' in text
    assert 'run_arm "$CANDIDATE_ARM" 1' in text
    # the single variable, and only it, is parameterised by the arm
    assert (
        'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION="$taw_production"' in text
    )
    assert "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \\" in text
    # the stock arm must be PROVEN not to have engaged the candidate
    assert "emitted a TAW production engagement on the stock-commit arm" in text


def test_runner_pins_fa2_on_and_identically_on_both_arms() -> None:
    """32e240e15 made FA2 the default, so 'leave it alone' stopped being safe.

    Both arms must NAME the same FA2 production arm. Naming it is what removes
    the dependence on whether the registry default happens to fire; naming it
    `gqa_pair` rather than empty is what keeps the pair a measurement of the
    configuration that actually ships.
    """
    text = RUNNER.read_text(encoding="utf-8")
    assert "FA2_PRODUCTION_ARM=gqa_pair" in text
    assert 'FR13_FA2_QROW32_B4_PRODUCTION_ARM="$FA2_PRODUCTION_ARM"' in text
    # not a per-arm parameter: it must not depend on $taw_production
    arm_body = text[text.index("run_arm() {") : text.index("# Both orders run")]
    assert "$taw_production" in arm_body
    fa2_lines = [
        line
        for line in arm_body.splitlines()
        if "FR13_FA2_QROW32_B4_PRODUCTION_ARM=" in line
        or "FR13_FA2_QROW32_B4_DUAL_GATE_JSON=" in line
        or "FORKED_FA2_SO=" in line
    ]
    assert fa2_lines, "the FA2 pin must be inside run_arm"
    for line in fa2_lines:
        assert "taw_production" not in line
    # and the registry default must still BE the value being pinned to
    assert 'FR13_FA2_QROW32_B4_PRODUCTION_ARM_DEFAULT:-}" == "$FA2_PRODUCTION_ARM"' in text


def test_runner_never_turns_fa2_off() -> None:
    """The other way to get it wrong: measuring TAW against a config nothing ships."""
    text = RUNNER.read_text(encoding="utf-8")
    assert "FR13_FA2_QROW32_B4_PRODUCTION_ARM= \\" not in text
    assert 'FR13_FA2_QROW32_B4_PRODUCTION_ARM=""' not in text


def test_runner_requires_the_admission_ledger_before_reducing() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "fr13_task_refill_ledger.jsonl" in text
    assert "fr13_task_refill_summary.json" in text
    assert "the width-4 window is DEFINED by" in text
    assert "--self-check" in text, "the reducer must resolve before GPU time"


def test_runner_lays_arms_out_for_the_sealed_window_reducer() -> None:
    """pass_00/<mode>_* is what fr13_b4_width4_window_reduce.discover_arms globs."""
    text = RUNNER.read_text(encoding="utf-8")
    assert "PASS_INDEX=${PASS_INDEX:-0}" in text
    assert "PASS_ROOT=${PASS_ROOT:-$RUNROOT_ABS}" in text
    assert "PASS_DIR=\"$PASS_ROOT/pass_$(printf '%02d' \"$PASS_INDEX\")\"" in text
    assert 'RUNROOT="$PASS_DIR"' in text
    assert "fr13_b4_width4_window_reduce.py" in text
    assert "fr13_b4_taw_width4_pair_reduce.py" in text
    derived = subprocess.run(
        [
            "bash",
            "-c",
            "RUNROOT_ABS=/repo/output/RUN\n"
            + "\n".join(
                line
                for line in text.splitlines()
                if line.startswith(("PASS_INDEX=", "PASS_ROOT=${", "PASS_DIR="))
            )
            + '\necho "$PASS_DIR"',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert derived == "/repo/output/RUN/pass_00", derived
    # and both arm names must start with the mode, or the glob misses them
    for line in text.splitlines():
        if line.startswith(("STOCK_ARM=", "CANDIDATE_ARM=")):
            assert line.split("=", 1)[1].startswith('"${FIXED32_MODE}_')


def test_runner_validates_the_bundle_through_the_runtimes_own_entrypoint() -> None:
    """Not a reimplementation: the function the container itself will call."""
    text = RUNNER.read_text(encoding="utf-8")
    assert "module._fr13_fixed32_taw_native_production_pass(" in text
    assert "module._fr13_fixed32_taw_source_contract(topology, batch_size=4)" in text
    assert "the gate must be re-earned at the commit that will serve" in text
    # the FA2 control variable carries the same HEAD binding via its sidecar
    assert "--expected-source-commit" in text


def test_runner_task_list_matches_the_subset_file_and_the_fa2_runner() -> None:
    """Four independent hardcodings of one ordered list; drift fails at boot."""
    from_file = ",".join(
        json.loads(SUBSET16.read_text(encoding="ascii"))["instance_ids"]
    )
    runner_line = next(
        line
        for line in RUNNER.read_text(encoding="utf-8").splitlines()
        if line.startswith("TASK_IDS=")
    )
    assert runner_line.split("=", 1)[1] == from_file
    fa2_line = next(
        line
        for line in FA2_RUNNER.read_text(encoding="utf-8").splitlines()
        if line.startswith("TASK_IDS=")
    )
    assert runner_line == fa2_line, "the two width-4 runners must serve one pool"
    assert len(from_file.split(",")) == 16


# --------------------------------------------------------------------------
# runner refusals, EXECUTED rather than pattern-matched
# --------------------------------------------------------------------------
def _run_block(start: str, end: str, prelude: str) -> subprocess.CompletedProcess:
    """Execute one named block of run_arm in isolation, with fixtures bound."""
    text = RUNNER.read_text(encoding="utf-8")
    begin = text.index(start)
    finish = text.index(end, begin) + len(end)
    script = (
        "set -uo pipefail\n"
        + prelude
        + "\ncheck() {\n"
        + text[begin:finish]
        + "\n  return 0\n}\ncheck\necho \"rc=$?\"\n"
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, cwd=REPO
    )


SINGLE_VARIABLE_START = '  [[ "$(grep -Fxc "FR13_FIXED32_MODE=$FIXED32_MODE"'
SINGLE_VARIABLE_END = (
    'did not run the declared single-variable B4 pool16 TAW selector" >&2; return 4; }'
)
CONSISTENCY_START = (
    "  if [[ \"$(grep -Fxc 'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0'"
)
CONSISTENCY_END = "\n  fi\n"


def _container_env(tmp_path: Path, **overrides) -> Path:
    values = {
        "FR13_FIXED32_MODE": "tail6_fixed32",
        "FR13_DRAFT_VOCAB_ROOT": "1",
        "FR13_DRAFT_VOCAB_K": "65536",
        "FR13_DRAFT_VOCAB_BLOCKS": "/workspace/scripts/fr13_dvk_subset_blocks.json",
        "FR13_B4_TASK_REFILL": "1",
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE": "0",
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION": "1",
        "FR13_FA2_QROW32_B4_PRODUCTION_ARM": "gqa_pair",
        "FR13_FA2_QROW32_B4_TIMING_ARM": "",
        "FR13_FA2_QROW32_B4_DUAL_GATE_SHA256": "e" * 64,
        "FR13_FA2_QROW32_SO_SHA256": (
            "af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb"
        ),
        "FR13_FA2_QROW32_LIVE_PAGED_AB": "0",
    }
    values.update(overrides)
    path = tmp_path / "container_env.txt"
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in sorted(values.items())),
        encoding="ascii",
    )
    return path


def _single_variable_prelude(container_env: Path, taw_production: str) -> str:
    return (
        f'container_env={container_env!s}\n'
        'FIXED32_MODE=tail6_fixed32\n'
        'DRAFT_VOCAB_ROOT=1\n'
        'DRAFT_VOCAB_K=65536\n'
        'DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json\n'
        f'taw_production={taw_production}\n'
        'FA2_PRODUCTION_ARM=gqa_pair\n'
        f'QROW32_GQA_PAIR_DUAL_GATE_SHA256={"e" * 64}\n'
        'CANDIDATE_SHA256='
        'af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb\n'
        'arm=unittest_arm\n'
    )


def test_a_matching_environment_passes_the_single_variable_check(
    tmp_path: Path,
) -> None:
    proc = _run_block(
        SINGLE_VARIABLE_START,
        SINGLE_VARIABLE_END,
        _single_variable_prelude(_container_env(tmp_path), "1"),
    )
    assert "rc=0" in proc.stdout, proc.stderr


@pytest.mark.parametrize(
    "override",
    [
        {"FR13_FA2_QROW32_B4_PRODUCTION_ARM": ""},
        {"FR13_FA2_QROW32_B4_PRODUCTION_ARM": "stock_dispatch"},
        {"FR13_FA2_QROW32_B4_TIMING_ARM": "gqa_pair"},
        {"FR13_FA2_QROW32_B4_DUAL_GATE_SHA256": "f" * 64},
        {"FR13_FA2_QROW32_SO_SHA256": "0" * 64},
        {"FR13_FIXED32_TAW_NATIVE_PRECOMPUTE": "1"},
        {"FR13_B4_TASK_REFILL": "0"},
    ],
)
def test_an_fa2_or_geometry_mismatch_between_arms_is_refused(
    tmp_path: Path, override: dict
) -> None:
    """An arm whose FA2 differs from the pin is refused, so the two cannot differ."""
    proc = _run_block(
        SINGLE_VARIABLE_START,
        SINGLE_VARIABLE_END,
        _single_variable_prelude(_container_env(tmp_path, **override), "1"),
    )
    assert "rc=4" in proc.stdout, proc.stdout + proc.stderr
    assert "single-variable" in proc.stderr


def _consistency_prelude(
    container_env: Path,
    *,
    engagement: Path,
    production_sidecar: Path,
    served_events: int,
) -> str:
    return (
        f"container_env={container_env!s}\n"
        f"engagement={engagement!s}\n"
        f"production_sidecar={production_sidecar!s}\n"
        f"served_events={served_events}\n"
        "arm=unittest_arm\n"
    )


def test_declared_off_with_no_served_evidence_is_consistent(tmp_path: Path) -> None:
    env = _container_env(
        tmp_path, FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION="0"
    )
    proc = _run_block(
        CONSISTENCY_START,
        CONSISTENCY_END,
        _consistency_prelude(
            env,
            engagement=tmp_path / "absent.json",
            production_sidecar=tmp_path / "absent.arm",
            served_events=0,
        ),
    )
    assert "rc=0" in proc.stdout, proc.stderr


@pytest.mark.parametrize("witness", ["engagement", "sidecar", "census"])
def test_declared_off_can_never_coexist_with_candidate_served_evidence(
    tmp_path: Path, witness: str
) -> None:
    """The rule, stated exactly: PRODUCTION=0 and served evidence are exclusive."""
    env = _container_env(
        tmp_path, FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION="0"
    )
    engagement = tmp_path / "engagement.json"
    sidecar = tmp_path / "production.arm"
    served = 0
    if witness == "engagement":
        engagement.write_text("{}", encoding="ascii")
    elif witness == "sidecar":
        sidecar.write_text("1\n", encoding="ascii")
    else:
        served = 17
    proc = _run_block(
        CONSISTENCY_START,
        CONSISTENCY_END,
        _consistency_prelude(
            env,
            engagement=engagement,
            production_sidecar=sidecar,
            served_events=served,
        ),
    )
    assert "rc=4" in proc.stdout, proc.stdout + proc.stderr
    assert "PRODUCTION=0 yet carries TAW candidate-served evidence" in proc.stderr


def test_declared_on_leaves_the_consistency_block_inert(tmp_path: Path) -> None:
    env = _container_env(tmp_path)
    engagement = tmp_path / "engagement.json"
    engagement.write_text("{}", encoding="ascii")
    proc = _run_block(
        CONSISTENCY_START,
        CONSISTENCY_END,
        _consistency_prelude(
            env,
            engagement=engagement,
            production_sidecar=tmp_path / "absent.arm",
            served_events=99,
        ),
    )
    assert "rc=0" in proc.stdout, proc.stderr


# --------------------------------------------------------------------------
# the HEAD-bound credential preflight, EXECUTED
# --------------------------------------------------------------------------
def _preflight_source() -> str:
    text = RUNNER.read_text(encoding="utf-8")
    return text.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


def _gate_payload(bundle_sha: str, commit: str, mode: str = "tail6_fixed32") -> dict:
    return {
        "schema": "fr13.fixed32.tail23_all_parent.exact4_b4_live_gate.v1",
        "status": "pass",
        "candidate": taw._FR13_FIXED32_TAW_NATIVE_CANDIDATE,
        "mode": mode,
        "source_commit": commit,
        "source_contract_schema": taw._FR13_FIXED32_TAW_SOURCE_SCHEMA,
        "source_contract_sha256": taw._FR13_FIXED32_TAW_SOURCE_SHA256,
        "production_bundle_sha256": bundle_sha,
        "live_bundle_sha256": bundle_sha,
        "qualified_batches": [2, 3, 4],
        "required_production_batches": [4],
        "probability_mismatches": 0,
        "product_mismatches": 0,
        "reference_always_served": True,
        "candidate_returned": False,
        "timing_eligible": False,
    }


def _run_preflight(tmp_path: Path, *, gate_overrides=None, bundle_kwargs=None):
    script = tmp_path / "preflight.py"
    script.write_text(_preflight_source(), encoding="utf-8")
    bundle = tmp_path / "production_pass.json"
    bundle_sha = _write_bundle(bundle, **(bundle_kwargs or {}))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    ).stdout.strip()
    gate_payload = _gate_payload(bundle_sha, head)
    gate_payload.update(gate_overrides or {})
    gate = tmp_path / "byte_gate.json"
    raw = json.dumps(gate_payload).encode("ascii")
    gate.write_bytes(raw)
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "scripts/fr13_device_multidraft_kernel.py",
            taw._FR13_FIXED32_TAW_SOURCE_SCHEMA,
            taw._FR13_FIXED32_TAW_SOURCE_SHA256,
            taw._FR13_FIXED32_TAW_NATIVE_CANDIDATE,
            str(bundle),
            bundle_sha,
            str(gate),
            hashlib.sha256(raw).hexdigest(),
            head,
            "tail6_fixed32",
            "0x7a9ce7ff",
            "23",
            str(tmp_path / "binding.json"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_preflight_accepts_a_bundle_earned_at_this_head(tmp_path: Path) -> None:
    proc = _run_preflight(tmp_path)
    assert proc.returncode == 0, proc.stderr
    binding = json.loads((tmp_path / "binding.json").read_text(encoding="ascii"))
    assert binding["status"] == "bound"
    assert binding["qualified_batches"] == [2, 3, 4]
    assert binding["pinned_min_batch"] == taw._FR13_FIXED32_TAW_PINNED_MIN_BATCH


def test_preflight_refuses_a_stale_head_bundle(tmp_path: Path) -> None:
    """The HEAD-binding doctrine, shared with the FA2 dual gate."""
    proc = _run_preflight(tmp_path, gate_overrides={"source_commit": FAKE_COMMIT})
    assert proc.returncode != 0
    assert "re-earned at the commit that will serve" in proc.stderr


@pytest.mark.parametrize(
    "override",
    [
        {"status": "fail"},
        {"probability_mismatches": 1},
        {"product_mismatches": 1},
        {"candidate_returned": True},
        {"reference_always_served": False},
        {"timing_eligible": True},
        {"production_bundle_sha256": "0" * 64},
        {"qualified_batches": [4]},
        {"mode": "hydra27_fixed32"},
    ],
)
def test_preflight_refuses_a_gate_that_does_not_publish_this_bundle(
    tmp_path: Path, override: dict
) -> None:
    proc = _run_preflight(tmp_path, gate_overrides=override)
    assert proc.returncode != 0
    assert "byte-gate verdict does not publish this PASS bundle" in proc.stderr


def test_preflight_refuses_a_bundle_bound_to_a_different_source_contract(
    tmp_path: Path,
) -> None:
    """This is the check the commit that adds this runner deliberately trips.

    Hardening `_fr13_fixed32_taw_native_arm_sources` changes the TAW source
    contract digest, so every bundle earned before it -- including 69f01aae --
    stops validating. That is the intended behaviour of a source-bound
    credential, and it is what forces the re-gate.
    """
    script = tmp_path / "preflight.py"
    script.write_text(_preflight_source(), encoding="utf-8")
    bundle = tmp_path / "production_pass.json"
    payload = _bundle_payload()
    payload["source_contract_sha256"] = "9" * 64
    raw = json.dumps(payload).encode("ascii")
    bundle.write_bytes(raw)
    bundle_sha = hashlib.sha256(raw).hexdigest()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=REPO, check=True,
    ).stdout.strip()
    gate = tmp_path / "byte_gate.json"
    gate_raw = json.dumps(_gate_payload(bundle_sha, head)).encode("ascii")
    gate.write_bytes(gate_raw)
    proc = subprocess.run(
        [
            sys.executable, str(script),
            "scripts/fr13_device_multidraft_kernel.py",
            taw._FR13_FIXED32_TAW_SOURCE_SCHEMA,
            taw._FR13_FIXED32_TAW_SOURCE_SHA256,
            taw._FR13_FIXED32_TAW_NATIVE_CANDIDATE,
            str(bundle), bundle_sha,
            str(gate), hashlib.sha256(gate_raw).hexdigest(),
            head, "tail6_fixed32", "0x7a9ce7ff", "23",
            str(tmp_path / "binding.json"),
        ],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode != 0
    assert "different candidate/source" in (proc.stderr + proc.stdout)


# ==========================================================================
# 4. the verdict reducer
# ==========================================================================
def test_self_check_resolves_both_topologies_and_the_emitter() -> None:
    proc = subprocess.run(
        [sys.executable, str(REDUCER), "--self-check"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    assert "hydra27_fixed32" in proc.stdout
    assert "tail6_fixed32" in proc.stdout


def test_self_check_fails_if_the_engagement_emitter_disappears(
    tmp_path: Path,
) -> None:
    """The reducer must never be bound to an artifact nothing writes."""
    module = _reducer()
    fake_repo = tmp_path / "repo"
    (fake_repo / "scripts").mkdir(parents=True)
    (fake_repo / "scripts/fr13_device_multidraft_kernel.py").write_text(
        "# no emitter here\n", encoding="utf-8"
    )
    for relative in (
        "results/fr13_b4_width4_nsys_20260813/fr13_b4_batch_conditioned_wall.json",
        "results/fr13_b4_width4_nsys_20260813/attribution_final.json",
        "config/fr13_fixed32/subset_b4_sixteen.json",
    ):
        target = fake_repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / relative).read_bytes())
    assert module.self_check(fake_repo) == 2


@pytest.mark.parametrize(
    "mode,mde,wall",
    [
        ("hydra27_fixed32", 4.204845067020671, 413.14178365521565),
        ("tail6_fixed32", 6.417803846730505, 411.05488226730876),
    ],
)
def test_thresholds_come_from_the_sealed_artifact(
    mode: str, mde: float, wall: float
) -> None:
    module = _reducer()
    t = module.load_sealed_thresholds(REPO, mode)
    assert t["mde_ms"] == pytest.approx(mde, rel=0, abs=1e-9)
    assert t["sealed_width4_step_wall_ms"] == pytest.approx(wall, rel=0, abs=1e-9)
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    assert t["mde_ms"] == sealed["pooled"][mode]["batch_conditioned_full_width"]["mde_ms"]


def test_the_addressable_pile_is_read_from_the_attribution_not_retyped() -> None:
    module = _reducer()
    pile = module.load_addressable_pile(REPO)
    groups = json.loads(ATTRIBUTION.read_text(encoding="utf-8"))["phase_kernels"][
        "cfwd"
    ]["groups_ms_per_step"]
    expected = groups["elementwise"]["ms_per_step"] + groups["other"]["ms_per_step"]
    assert pile["small_kernel_ms_per_step"] == pytest.approx(expected, abs=1e-12)
    # the README's "37.1 ms/step across 321k tiny instances"
    assert pile["small_kernel_ms_per_step"] == pytest.approx(37.1, abs=0.1)
    assert pile["small_kernel_instances"] == pytest.approx(321_712, abs=1000)


def test_the_instrument_cannot_resolve_a_tenth_of_the_pile() -> None:
    """The disclosure that keeps a null honest, asserted rather than hoped for."""
    module = _reducer()
    for mode in ("hydra27_fixed32", "tail6_fixed32"):
        t = module.load_sealed_thresholds(REPO, mode)
        assert t["illustrative_tenth_of_pile_ms"] < t["mde_ms"]
        assert t["instrument_can_resolve_a_tenth_of_the_pile"] is False


def test_blended_basis_carries_its_own_larger_mde() -> None:
    module = _reducer()
    for mode in ("hydra27_fixed32", "tail6_fixed32"):
        t = module.load_sealed_thresholds(REPO, mode)
        assert t["blended_basis"]["mde_ms"] > t["mde_ms"]


def _thresholds():
    return _reducer().load_sealed_thresholds(REPO, "tail6_fixed32")


@pytest.mark.parametrize(
    "improvement,band,disposition",
    [
        (12.0, "GAIN_CLEARS_FOUR_PASS_MDE", "LEVER_SURVIVES_FUND_A_FOUR_PASS_CAMPAIGN"),
        (
            6.417803846730505,
            "GAIN_CLEARS_FOUR_PASS_MDE",
            "LEVER_SURVIVES_FUND_A_FOUR_PASS_CAMPAIGN",
        ),
        (1.0, "GAIN_BELOW_FOUR_PASS_MDE", "NOT_RESOLVED_AT_THIS_SIZE"),
        (0.0, "NO_GAIN_OR_REGRESSION", "NOT_RESOLVED_AT_THIS_SIZE"),
        (-3.0, "NO_GAIN_OR_REGRESSION", "NOT_RESOLVED_AT_THIS_SIZE"),
    ],
)
def test_verdict_bands_are_continuous_and_correctly_ordered(
    improvement: float, band: str, disposition: str
) -> None:
    module = _reducer()
    verdict = module.judge(improvement, _thresholds())
    assert verdict["band"] == band
    assert verdict["lever_disposition"] == disposition
    assert verdict["clears_four_pass_mde"] == (improvement >= _thresholds()["mde_ms"])


def test_a_sub_mde_result_is_never_called_a_null() -> None:
    module = _reducer()
    for improvement in (1.0, 0.0, -3.0):
        verdict = module.judge(improvement, _thresholds())
        assert "not a null" in verdict["reading"] or "not as null" in verdict["reading"]
        assert verdict["is_significance_test"] is False
        assert verdict["n_paired_draws"] == 1


def test_does_not_claim_covers_the_load_bearing_limits() -> None:
    module = _reducer()
    joined = " ".join(module.DOES_NOT_CLAIM).lower()
    for needle in (
        "statistical significance",
        "pre-registered effect size",
        "null means no effect",
        "strong placebo",
        "numerics verdict",
        "whole-arm throughput",
        "cap verdict",
        "agent-quality",
        "promotion",
    ):
        assert needle in joined, needle


# --------------------------------------------------------------------------
# reducer: fail-closed provenance
# --------------------------------------------------------------------------
def _good_engagement(**overrides):
    record = {
        "schema": "fr13.fixed32.taw_native_precompute.production_engagement.v1",
        "status": "ENGAGED",
        "candidate": "fixed32_all_parent_commit_v2",
        "route": PRODUCTION_ROUTE,
        "candidate_returned": True,
        "reference_returned": False,
        "observer_accounting_only": True,
        "flush_action": "final",
        "finalized_by_fixed32_flush": True,
        "mode": "tail6_fixed32",
        "source_contract_sha256": taw._FR13_FIXED32_TAW_SOURCE_SHA256,
        "production_bundle_sha256": "a" * 64,
        "qualified_batches": [2, 3, 4],
        "pinned_min_batch": 2,
        "served_candidate_calls_by_batch": {"2": 3, "3": 9, "4": 800},
        "served_candidate_calls": 812,
    }
    record.update(overrides)
    return record


def _validate(stock, candidate):
    module = _reducer()
    module.validate_pair_engagement(
        stock,
        candidate,
        expected_mode="tail6_fixed32",
        expected_bundle_sha256="a" * 64,
        expected_source_contract_sha256=taw._FR13_FIXED32_TAW_SOURCE_SHA256,
    )


def test_engagement_on_the_stock_arm_is_a_hard_failure() -> None:
    module = _reducer()
    with pytest.raises(module.PairError, match="leaked across the pair"):
        _validate(_good_engagement(), _good_engagement())


def test_missing_candidate_engagement_is_a_hard_failure() -> None:
    module = _reducer()
    with pytest.raises(module.PairError, match="emitted no TAW production engagement"):
        _validate(None, None)


def test_a_correct_pair_is_accepted() -> None:
    _validate(None, _good_engagement())


@pytest.mark.parametrize(
    "override",
    [
        {"status": "BYPASSED"},
        {"candidate_returned": False},
        {"reference_returned": True},
        {"route": DIAGNOSTIC_ROUTE},
        {"mode": "hydra27_fixed32"},
        {"flush_action": "incremental"},
        {"observer_accounting_only": False},
        {"candidate": "something_else"},
    ],
)
def test_a_drifted_engagement_is_refused(override: dict) -> None:
    module = _reducer()
    with pytest.raises(module.PairError, match="did not serve the TAW native"):
        _validate(None, _good_engagement(**override))


def test_a_bundle_the_runner_did_not_validate_is_refused() -> None:
    module = _reducer()
    with pytest.raises(module.PairError, match="not the bundle the runner validated"):
        _validate(None, _good_engagement(production_bundle_sha256="b" * 64))


def test_a_different_source_contract_is_refused() -> None:
    module = _reducer()
    with pytest.raises(module.PairError, match="not the contract this verdict"):
        _validate(None, _good_engagement(source_contract_sha256="b" * 64))


# --------------------------------------------------------------------------
# reducer: treated widths and strata
# --------------------------------------------------------------------------
def test_treated_widths_are_the_authorised_set_not_the_observed_one() -> None:
    """A qualified width with no events is still treated, never a control."""
    module = _reducer()
    engagement = _good_engagement(
        served_candidate_calls_by_batch={"4": 800}, served_candidate_calls=800
    )
    assert module.authorised_widths(engagement) == (2, 3, 4)


def test_a_served_batch_outside_the_authorised_set_is_refused() -> None:
    module = _reducer()
    engagement = _good_engagement(
        qualified_batches=[4],
        served_candidate_calls_by_batch={"2": 1, "4": 800},
    )
    with pytest.raises(module.PairError, match="unauthorised batches"):
        module.authorised_widths(engagement)


def test_an_engagement_without_a_qualified_set_is_refused() -> None:
    module = _reducer()
    with pytest.raises(module.PairError, match="must not be guessed"):
        module.authorised_widths(_good_engagement(qualified_batches=None))


def _bc(by_width: dict[int, tuple[int, float]]) -> dict:
    return {
        "available": True,
        "by_width": {
            str(w): {"steps": n, "mean_ms": ms, "fraction": 0.0, "sd_ms": 0.0}
            for w, (n, ms) in by_width.items()
        },
    }


def test_strata_label_every_authorised_width_as_treated() -> None:
    module = _reducer()
    stock = _bc({1: (40, 270.0), 2: (300, 300.0), 3: (900, 362.0), 4: (3000, 411.0)})
    cand = _bc({1: (44, 268.0), 2: (300, 296.0), 3: (900, 356.0), 4: (3000, 399.0)})
    strata = module.per_width_strata(stock, cand, (2, 3, 4))
    roles = {r["width"]: r["role"] for r in strata["rows"]}
    assert roles == {1: "placebo", 2: "treated", 3: "treated", 4: "treated"}
    assert strata["treated_width"] == 4


def test_a_thin_width_one_control_is_refused_as_a_did_basis() -> None:
    """The EXPECTED state for this lever, and it must be named, not silently used."""
    module = _reducer()
    stock = _bc({1: (40, 270.0), 4: (3000, 411.0)})
    cand = _bc({1: (44, 268.0), 4: (3000, 399.0)})
    strata = module.per_width_strata(stock, cand, (2, 3, 4))
    did = strata["difference_in_differences"]
    assert did["available"] is False
    assert "EXPECTED case" in did["reason"]


def test_a_fat_placebo_width_is_used_and_detects_an_arm_wide_shift() -> None:
    module = _reducer()
    stock = _bc({1: (500, 270.0), 4: (3000, 411.0)})
    cand = _bc({1: (500, 250.0), 4: (3000, 391.0)})
    strata = module.per_width_strata(stock, cand, (2, 3, 4))
    did = strata["difference_in_differences"]
    assert did["available"] is True
    assert did["control_width"] == 1
    assert did["additive_effect_ms"] == pytest.approx(0.0, abs=0.5)
    assert "OVERSTATES" in did["confound_direction"]


def test_positive_always_means_the_candidate_is_better() -> None:
    module = _reducer()
    deltas = module.delta_block(
        {"step_wall_ms": 411.0, "per_request_step_tps": 16.0},
        {"step_wall_ms": 399.0, "per_request_step_tps": 17.0},
    )
    assert deltas["step_wall_ms"]["improvement"] == pytest.approx(12.0)
    assert deltas["step_wall_ms"]["candidate_minus_stock"] == pytest.approx(-12.0)
    assert deltas["per_request_step_tps"]["improvement"] == pytest.approx(1.0)


def test_reducer_requires_both_arm_directories() -> None:
    proc = subprocess.run(
        [sys.executable, str(REDUCER), "--runroot", str(REPO / "output/none")],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode == 2
    assert "--stock-arm is required" in proc.stderr


def test_reducer_delegates_the_windowing_and_does_not_reimplement_it() -> None:
    text = REDUCER.read_text(encoding="utf-8")
    assert "import fr13_b4_width4_window_reduce as w4" in text
    assert "w4.reduce_window_arm(" in text
    # the sealed reducer must be used, not copied
    assert "def reduce_window_arm" not in text
    assert "def derive_width4_window" not in text
    assert "def window_counter_bracket" not in text
