from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
SERVE = REPO / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"
OFFLOAD = REPO / "scripts" / "swe_x86_helpers" / "offload_codex_proxy.sh"
RELAUNCH = (
    REPO / "scripts" / "swe_x86_helpers" / "relaunch_proxy_remote.sh"
)
CONTRACT = REPO / "scripts" / "fr13_fixed32_contract.py"
FLOOR = REPO / "scripts" / "fr13_floor_gate.py"
DEPTH = REPO / "scripts" / "fr13_depth_acceptance.py"


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_fixed32_launcher_stages_private_secret_and_installs_middleware() -> None:
    launcher = source(LAUNCHER)
    contract = source(CONTRACT)

    assert 'stat -c \'%a\' "$FR13_FIXED32_INGRESS_SECRET_FILE"' in launcher
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
        '"lumo_flywheel_serving.inference_proxy.'
        'Fixed32EngineIngressMiddleware",'
    ) in contract


def test_fixed32_offload_uses_stdin_secret_and_disables_raw_dumps() -> None:
    offload = source(OFFLOAD)
    relaunch = source(RELAUNCH)
    serve = source(SERVE)

    assert "cat > $REMOTE_FIXED32_SECRET" in offload
    assert '< "$FIXED32_SECRET_LOCAL"' in offload
    assert "stat -c '%a' $REMOTE_FIXED32_SECRET" in offload
    assert (
        "unset LUMO_PROXY_PAIR_DUMP_DIR LUMO_PROXY_REQUEST_DUMP_DIR"
        in offload
    )
    assert "export LUMO_PROXY_FIXED32_DISABLE_RAW_DUMPS=1" in offload
    assert 'if [[ -z "$FIXED32_TASK_IDS" ]]; then' in offload
    assert (
        'if [ "${LUMO_PROXY_FIXED32_DISABLE_RAW_DUMPS:-0}" = "1" ]; then'
        in relaunch
    )
    assert (
        'if [ "${LUMO_PROXY_FIXED32_DISABLE_RAW_DUMPS:-0}" != "1" ]; then'
        in relaunch
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

    teardown = serve[
        serve.index("teardown(){") : serve.index("\ntrap teardown EXIT")
    ]
    stop_proxy = teardown.index(
        'bash "$OFFLOAD_HELPER" stop "$OFFLOAD_HOST"'
    )
    final_flush = teardown.index("--action final")
    audit = teardown.index("write_fixed32_chat_traffic_audit")
    assert stop_proxy < final_flush < audit
    assert "build_fixed32_chat_traffic_audit" in serve


def test_floor_and_depth_require_exact_ingress_and_trace_evidence() -> None:
    floor = source(FLOOR)
    depth = source(DEPTH)
    required_gates = (
        "fixed32_ingress_proxy_engine_exact",
        "fixed32_zero_campaign_rejections",
        "fixed32_raw_proxy_dumps_disabled",
    )

    assert "fr13.canonical_swe_verified_fixed32_floor_gate.v11" in floor
    assert "fr13-fixed32-chat-task-provenance-audit-v2" in floor
    assert "fr13.depth_acceptance.fixed32.v2" in depth
    for gate in required_gates:
        assert gate in floor
        assert gate in depth
    assert "trace_model_request_id_sha256s" in floor
    assert "task successful request evidence differs from terminal trace" in floor
    assert "trace_request_id_sha256s" in depth
    assert "task trace/engine request evidence differs" in depth
