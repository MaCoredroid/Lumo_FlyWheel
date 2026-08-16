from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHERS = [
    REPO / "scripts" / "fr10_launch_speed_server.sh",
    REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh",
]


def test_nsys_wrapper_is_off_by_default_and_mounts_only_when_enabled() -> None:
    for launcher_path in LAUNCHERS:
        text = launcher_path.read_text()

        assert "LUMO_NSYS_WRAP_VLLM=${LUMO_NSYS_WRAP_VLLM:-0}" in text
        assert (
            "LUMO_NSYS_OUTPUT=${LUMO_NSYS_OUTPUT:-/logs/nsys_vllm_${CONTAINER}}" in text
        )
        assert "NSYS_DOCKER_ARGS=()" in text
        assert 'if _lumo_truthy "$LUMO_NSYS_WRAP_VLLM"; then' in text
        assert "for nsight_mount in /opt/nvidia /usr/local/cuda-13.0; do" in text
        assert 'NSYS_DOCKER_ARGS+=(-v "$nsight_mount:$nsight_mount:ro")' in text
        assert '"${NSYS_DOCKER_ARGS[@]}" \\' in text

        # Normal launches should not bind host Nsight paths unless the opt-in
        # array is populated by LUMO_NSYS_WRAP_VLLM.
        assert "-v /opt/nvidia:/opt/nvidia:ro" not in text
        assert "-v /usr/local/cuda-13.0:/usr/local/cuda-13.0:ro" not in text


def test_nsys_wrapper_wraps_the_in_container_vllm_serve_exec() -> None:
    for launcher_path in LAUNCHERS:
        text = launcher_path.read_text()

        assert '-e LUMO_NSYS_WRAP_VLLM="$LUMO_NSYS_WRAP_VLLM"' in text
        assert '-e LUMO_NSYS_BIN="$LUMO_NSYS_BIN"' in text
        assert '-e LUMO_NSYS_DELAY_S="$LUMO_NSYS_DELAY_S"' in text
        assert '-e LUMO_NSYS_DURATION_S="$LUMO_NSYS_DURATION_S"' in text
        assert '-e LUMO_NSYS_FLUSH_MS="$LUMO_NSYS_FLUSH_MS"' in text
        assert '-e LUMO_NSYS_OUTPUT="$LUMO_NSYS_OUTPUT"' in text
        assert "NSYS_PREFIX=()" in text
        assert "--cuda-graph-trace=node" in text
        # Without a periodic CUPTI flush, per-kernel rows (incl. graph node-level
        # kernels) are dropped as incomplete at the delayed-duration session stop
        # on GB10; fr13_b1_profile_bind exports had ZERO kernel tables.
        assert "LUMO_NSYS_FLUSH_MS=${LUMO_NSYS_FLUSH_MS:-100}" in text
        assert '--cuda-flush-interval \\"\\$LUMO_NSYS_FLUSH_MS\\"' in text
        # GB10 GPU-timestamp drop workaround: ALL per-kernel rows are dropped as
        # "incomplete" at session stop even with periodic flushes unless
        # CuptiUseRawGpuTimestamps=false is written to the in-container nsys
        # user config before the wrapped vllm serve exec.
        assert (
            "LUMO_NSYS_CONFIG_DIRECTIVES="
            "${LUMO_NSYS_CONFIG_DIRECTIVES:-CuptiUseRawGpuTimestamps=false}"
        ) in text
        assert '-e LUMO_NSYS_CONFIG_DIRECTIVES="$LUMO_NSYS_CONFIG_DIRECTIVES"' in text
        assert 'NSYS_CFG_PATH=\\$(\\"\\$LUMO_NSYS_BIN\\" -z)' in text
        assert (
            "printf '%s\\n' \\\"\\$LUMO_NSYS_CONFIG_DIRECTIVES\\\" | tr ';' '\\n' "
            '>> \\"\\$NSYS_CFG_PATH\\"'
        ) in text
        # Trace selector is env-tunable: GB10+CUDA13 hardware-trace kernel rows
        # are dropped in delayed sessions; cuda,cuda-sw forces software records.
        assert "LUMO_NSYS_TRACE=${LUMO_NSYS_TRACE:-cuda,nvtx}" in text
        assert '-e LUMO_NSYS_TRACE="$LUMO_NSYS_TRACE"' in text
        assert '--trace=\\"\\$LUMO_NSYS_TRACE\\"' in text
        assert '-o \\"\\$LUMO_NSYS_OUTPUT\\"' in text
        # The nsys prefix must wrap the exec in EVERY launcher, and no
        # launcher may keep an unwrapped exec beside it. Stated
        # model-agnostically: fr10_launch_speed_server.sh is an FR10-era
        # vehicle that FR14 deliberately did not re-point, so pinning a model
        # identity here would couple two unrelated things.
        assert 'exec \\"\\${NSYS_PREFIX[@]}\\" vllm serve ' in text
        assert "exec vllm serve " not in text


def test_fixed32_launcher_serves_the_fr14_checkpoint_through_one_variable() -> None:
    """FR14: the checkpoint is a single point of truth, not two buried literals.

    Pin the definition AND the use, so neither can drift alone, and pin the
    readonly so a caller cannot re-point the serve line from the environment
    (fr13_fixed32_contract.expected_pid1_argv compares PID1 argv exactly).
    """
    text = (REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh").read_text()

    assert "SERVED_MODEL_PATH=/models/qwen3.8-27b-nvfp4\n" in text
    assert "SERVED_MODEL_NAME=qwen3.8-27b-nvfp4\n" in text
    assert "readonly SERVED_MODEL_PATH SERVED_MODEL_NAME\n" in text
    assert (
        'exec \\"\\${NSYS_PREFIX[@]}\\" vllm serve '
        "$SERVED_MODEL_PATH --served-model-name $SERVED_MODEL_NAME"
    ) in text
    assert "/models/qwen3.6-27b-fp8" not in text


def test_fixed32_launcher_omits_environment_from_nsys_report() -> None:
    text = (REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh").read_text()

    assert "--discard-environment=true" in text


def test_fixed32_launcher_pins_session_and_container_creation_identity() -> None:
    text = (REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh").read_text()

    assert '\\"--session-new=%q{LUMO_NSYS_SESSION_NAME}\\"' in text
    assert 'FR13_FIXED32_CIDFILE="$LOG_DIR/fr13_fixed32_container.cid"' in text
    assert '[[ ! -e "$FR13_FIXED32_CIDFILE" && ! -L "$FR13_FIXED32_CIDFILE" ]]' in text
    assert "FR13_FIXED32_DOCKER_ARGS=(\n    --cidfile\n" in text
    assert (
        'docker ps -aq --filter "name=^/${CONTAINER}$"'
        in text
    )
    assert 'GPU_GUARD_CONTAINER_ID="$FR13_FIXED32_CONTAINER_ID"' in text
    assert 'GPU_GUARD_EXPECTED_NAME="$CONTAINER"' in text
