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
        assert "LUMO_NSYS_OUTPUT=${LUMO_NSYS_OUTPUT:-/logs/nsys_vllm_${CONTAINER}}" in text
        assert "NSYS_DOCKER_ARGS=()" in text
        assert 'if _lumo_truthy "$LUMO_NSYS_WRAP_VLLM"; then' in text
        assert 'for nsight_mount in /opt/nvidia /usr/local/cuda-13.0; do' in text
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
        assert '--cuda-graph-trace=node' in text
        # Without a periodic CUPTI flush, per-kernel rows (incl. graph node-level
        # kernels) are dropped as incomplete at the delayed-duration session stop
        # on GB10; fr13_b1_profile_bind exports had ZERO kernel tables.
        assert "LUMO_NSYS_FLUSH_MS=${LUMO_NSYS_FLUSH_MS:-100}" in text
        assert '--cuda-flush-interval \\"\\$LUMO_NSYS_FLUSH_MS\\"' in text
        assert '--trace=cuda,nvtx' in text
        assert '-o \\"\\$LUMO_NSYS_OUTPUT\\"' in text
        assert (
            'exec \\"\\${NSYS_PREFIX[@]}\\" vllm serve '
            "/models/qwen3.6-27b-fp8 --served-model-name qwen3.6-27b"
        ) in text
        assert "exec vllm serve /models/qwen3.6-27b-fp8" not in text
