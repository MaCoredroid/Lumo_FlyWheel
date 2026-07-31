from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
FLOOR_GATE = REPO / "scripts" / "fr13_floor_gate.py"


def _load_floor_gate():
    spec = importlib.util.spec_from_file_location("fr13_floor_gate", FLOOR_GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed32_launcher_forces_exact_scheduler_request_id_binding() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert (
        '[[ "${VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:-1}" == "1" ]]'
        in text
    )
    assert "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1" in text
    assert "export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION" in text
    assert (
        '"VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1"'
        in text
    )


def test_fixed32_request_id_binding_is_repinned_after_local_env_and_not_forwarded(
    tmp_path: Path,
) -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    late_source = text.index(
        "\nset -a\n",
        text.index("fixed32 exact container name already exists"),
    )
    repin = text.index(
        "fixed32 request-ID binding changed after local environment loading",
        late_source,
    )
    forwarding = text.index("FR13_ENV_FORWARD_ARGS=()", repin)
    docker_run = text.index("docker run -d", forwarding)
    assert late_source < repin < forwarding < docker_run
    assert (
        '&& "$_v" == "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"' in text
    )

    post_local_fragment = text[
        late_source + 1 : text.index("\n_lumo_truthy() {", repin)
    ]
    forwarding_fragment = text[
        forwarding : text.index(
            '\n\nif [[ -n "${FR13_FIXED32_MODE:-}" ]]; then',
            forwarding,
        )
    ]
    harness = "\n".join(
        (
            "set -euo pipefail",
            "REPO=$1",
            "_FR13_LOCAL_ENV_SOURCED=0",
            "FR13_FIXED32_MODE=tail6_fixed32",
            "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1",
            "export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION",
            post_local_fragment,
            forwarding_fragment,
            'printf "ARG=%s\\n" "${FR13_ENV_FORWARD_ARGS[@]}"',
        )
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    local_env = repo / ".lumo.local.env"
    local_env.write_text(
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=0\n",
        encoding="ascii",
    )
    conflict = subprocess.run(
        ["bash", "-c", harness, "--", os.fspath(repo)],
        env={"PATH": os.environ["PATH"]},
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert conflict.returncode == 2
    assert (
        "fixed32 request-ID binding changed after local environment loading"
        in conflict.stderr
    )

    local_env.write_text(
        "VLLM_REQUEST_BINDING_SENTINEL=kept\n",
        encoding="ascii",
    )
    accepted = subprocess.run(
        ["bash", "-c", harness, "--", os.fspath(repo)],
        env={"PATH": os.environ["PATH"]},
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "ARG=VLLM_REQUEST_BINDING_SENTINEL=kept" in accepted.stdout
    assert "ARG=VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1" not in accepted.stdout


def test_fixed32_gate_attests_exact_scheduler_request_id_binding(
    tmp_path: Path,
) -> None:
    floor_gate = _load_floor_gate()
    task_ids = list(floor_gate.EVIDENCE_SETS[4]["task_ids"])

    required = floor_gate.fixed32_required_env(
        tmp_path,
        mode="hydra27_fixed32",
        task_ids=task_ids,
    )

    assert required["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] == "1"
