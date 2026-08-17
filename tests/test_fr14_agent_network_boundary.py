"""FR14 agent network boundary — runner-side wiring.

STOP FINDING 3231eeff7: the SWE agent reached the internet through the shell
(`python -c "import urllib.request"`) and pulled its own gold patch. The fix is
a network boundary, not a setting. These tests pin the runner half of it:

  * every agent docker-run site joins the FR14 bridge, never --network=host,
  * fixed32 REFUSES to launch when the boundary is disabled, unverifiable, or
    verifies a different network than the one the container would join,
  * OPENAI_BASE_URL is re-pointed at the bridge gateway DERIVED from the
    verified network (the proxy keeps its 127.0.0.1 bind),
  * a mid-run fingerprint change is fatal.

The shell script's own behaviour (rules, ordering, DNAT) is measured live on
alienware and recorded in
results/fr14_nvfp4_port_20260816/agent_net_boundary_probe.md; this file needs no
docker, no GPU and no network.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_swe_bench_q36_a as runner  # noqa: E402

BOUNDARY_SCRIPT = SCRIPTS / "fr14_agent_network_boundary.sh"
GATEWAY = "172.31.99.1"
FINGERPRINT = "e3cc51795829dca6a7ac83a86e7a8e52f4937b469882bbfb2e70c356c6f1b5e4"


def _receipt(net: str = "fr14-agent-isolated", fingerprint: str = FINGERPRINT) -> str:
    return (
        f"FR14_NET_BOUNDARY_VERIFIED net={net} bridge=fr14agent0 "
        f"gateway={GATEWAY} proxy_port=8023 rules=9 fingerprint={fingerprint}\n"
    )


@pytest.fixture(autouse=True)
def _isolated_boundary_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(runner.FR14_AGENT_NET_ENV, raising=False)
    monkeypatch.setattr(runner, "_FR14_BOUNDARY_CACHE", {})
    monkeypatch.setattr(runner, "_FR14_BOUNDARY_APPLIED", set())
    monkeypatch.setattr(runner, "AGENT_HOST", None)
    yield


def _stub_boundary_ssh(monkeypatch: pytest.MonkeyPatch, results: dict[str, tuple]):
    """Replace the ssh hop with canned (rc, stdout) per subcommand."""
    calls: list[str] = []

    def fake(host: str, subcommand: str):
        calls.append(f"{host}:{subcommand}")
        rc, out = results[subcommand]
        return subprocess.CompletedProcess(
            args=["ssh"], returncode=rc, stdout=out, stderr=""
        )

    monkeypatch.setattr(runner, "_fr14_boundary_ssh", fake)
    return calls


# ---------------------------------------------------------------- network name
def test_offload_default_is_the_boundary_not_host(monkeypatch):
    monkeypatch.setattr(runner, "AGENT_HOST", "alienware")
    assert runner._fr14_agent_network() == "fr14-agent-isolated"
    assert runner._fr14_boundary_enabled() is True


def test_local_agent_keeps_host_networking(monkeypatch):
    monkeypatch.setattr(runner, "AGENT_HOST", None)
    assert runner._fr14_agent_network() == "host"


@pytest.mark.parametrize("value", ["host", "0", "off", "none", "no", "", "  "])
def test_explicit_opt_out_spellings(monkeypatch, value):
    monkeypatch.setattr(runner, "AGENT_HOST", "alienware")
    monkeypatch.setenv(runner.FR14_AGENT_NET_ENV, value)
    assert runner._fr14_agent_network() == "host"
    assert runner._fr14_boundary_enabled() is False


# ------------------------------------------------------- docker-run rendering
def test_no_agent_docker_run_site_hardcodes_host_networking():
    """The three agent sites must all go through _fr14_agent_network().

    Guards the regression that started all of this: a new agent docker-run site
    (or a revert of one of the three) silently restores full host egress.
    """
    source = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")
    docker_run_lines = [
        line for line in source.splitlines() if "docker run " in line
    ]
    assert docker_run_lines, "expected the agent docker-run sites to exist"
    for line in docker_run_lines:
        assert "--network=host" not in line, line
    network_flags = [
        line for line in source.splitlines()
        if "--network=" in line and "docker run" in line
    ]
    assert len(network_flags) == 2, network_flags  # the two .format templates
    for line in network_flags:
        assert "--network={agent_network}" in line, line


@pytest.mark.parametrize("agent", ["qwen_code", "codex"])
def test_agent_template_carries_the_bridge(monkeypatch, agent):
    monkeypatch.setattr(runner, "AGENT_HOST", "alienware")
    monkeypatch.setenv("SWE_AGENT", agent)
    template = runner._agent_template()
    assert "--network=fr14-agent-isolated" in template
    assert "{agent_network}" not in template
    # the placeholder must not survive into .format(); rendering with the two
    # call sites' kwargs has to succeed.
    rendered = template.format(
        container_name="c", workspace="/w", endpoint="http://x/v1",
        model="m", session_id="s",
    )
    assert "--network=fr14-agent-isolated" in rendered


def test_instance_agent_command_joins_the_bridge(monkeypatch):
    monkeypatch.setattr(runner, "AGENT_HOST", "alienware")
    cmd = runner._instance_agent_command(
        container_name="c1", image="img:1", endpoint=f"http://{GATEWAY}:8023/v1",
        model="m", host_out_dir="/o", bundle_src="/b", agents_md_b64="YQ==",
        prompt_b64="Yg==", base_commit="abc", session_id="s",
    )
    assert "--network=fr14-agent-isolated" in cmd
    assert "--network=host" not in cmd


# ------------------------------------------------------------ endpoint rewrite
@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("http://127.0.0.1:8023/v1", f"http://{GATEWAY}:8023/v1"),
        ("http://localhost:8023/v1", f"http://{GATEWAY}:8023/v1"),
        ("http://[::1]:8023/v1", f"http://{GATEWAY}:8023/v1"),
        ("http://127.0.0.1/v1", f"http://{GATEWAY}/v1"),
        # already an off-loopback address: untouched
        ("http://10.0.0.113:8023/v1", "http://10.0.0.113:8023/v1"),
        (f"http://{GATEWAY}:8023/v1", f"http://{GATEWAY}:8023/v1"),
    ],
)
def test_endpoint_rewrite(endpoint, expected):
    assert runner._fr14_rewrite_endpoint(endpoint, GATEWAY) == expected


def test_endpoint_gateway_is_derived_not_literal():
    """Changing the subnet in the boundary script must follow through."""
    assert runner._fr14_rewrite_endpoint(
        "http://127.0.0.1:8023/v1", "10.77.0.1"
    ) == "http://10.77.0.1:8023/v1"


# ------------------------------------------------------------------- receipts
def test_receipt_parse_round_trip():
    fields = runner._fr14_parse_verified(_receipt().strip())
    assert fields["gateway"] == GATEWAY
    assert fields["fingerprint"] == FINGERPRINT


@pytest.mark.parametrize(
    "line",
    [
        "FR14_NET_BOUNDARY_VERIFIED net=n bridge=b gateway=g proxy_port=1",
        "FR14_NET_BOUNDARY_VERIFIED net=n bridge=b gateway=g proxy_port=1 "
        "fingerprint=deadbeef",
        "FR14_NET_BOUNDARY_VERIFIED net=n bridge=b gateway=g proxy_port=1 "
        "fingerprint=" + "z" * 64,
    ],
)
def test_receipt_rejects_malformed(line):
    with pytest.raises(runner.Fixed32BoundaryError):
        runner._fr14_parse_verified(line)


# ---------------------------------------------------------------- fail-closed
def test_gate_accepts_a_verified_boundary(monkeypatch):
    calls = _stub_boundary_ssh(monkeypatch, {
        "apply": (0, "FR14_NET_BOUNDARY_APPLIED ok\n"),
        "verify": (0, _receipt()),
    })
    observation = runner._fr14_require_network_boundary("alienware")
    assert observation["gateway"] == GATEWAY
    assert calls == ["alienware:apply", "alienware:verify"]
    # apply runs once per host per process; verify runs every time.
    runner._fr14_require_network_boundary("alienware")
    assert calls.count("alienware:apply") == 1
    assert calls.count("alienware:verify") == 2


def test_gate_refuses_when_boundary_disabled(monkeypatch):
    monkeypatch.setenv(runner.FR14_AGENT_NET_ENV, "host")
    with pytest.raises(runner.Fixed32BoundaryError, match="3231eeff7"):
        runner._fr14_require_network_boundary("alienware")


def test_gate_does_not_depend_on_when_agent_host_was_assigned(monkeypatch):
    """AGENT_HOST is a module global assigned in main(); the gate must decide
    from the host it was handed, not from import-time state."""
    monkeypatch.setattr(runner, "AGENT_HOST", None)
    _stub_boundary_ssh(monkeypatch, {"apply": (0, ""), "verify": (0, _receipt())})
    assert runner._fr14_require_network_boundary("alienware")["gateway"] == GATEWAY


def test_gate_refuses_a_local_agent(monkeypatch):
    monkeypatch.setattr(runner, "AGENT_HOST", "alienware")
    with pytest.raises(runner.Fixed32BoundaryError, match="offload host"):
        runner._fr14_require_network_boundary(None)


def test_gate_refuses_when_verify_fails(monkeypatch):
    _stub_boundary_ssh(monkeypatch, {
        "apply": (0, ""),
        "verify": (3, "FAIL: missing rules: filter/INPUT[0]"),
    })
    with pytest.raises(runner.Fixed32BoundaryError, match="did not verify"):
        runner._fr14_require_network_boundary("alienware")


def test_gate_refuses_a_silent_verify(monkeypatch):
    _stub_boundary_ssh(monkeypatch, {"apply": (0, ""), "verify": (0, "ok\n")})
    with pytest.raises(runner.Fixed32BoundaryError, match="no receipt"):
        runner._fr14_require_network_boundary("alienware")


def test_gate_refuses_a_different_network(monkeypatch):
    monkeypatch.setenv(runner.FR14_AGENT_NET_ENV, "fr14-somewhere-else")
    _stub_boundary_ssh(monkeypatch, {"apply": (0, ""), "verify": (0, _receipt())})
    with pytest.raises(runner.Fixed32BoundaryError, match="would join"):
        runner._fr14_require_network_boundary("alienware")


def test_gate_refuses_a_mid_run_fingerprint_change(monkeypatch):
    state = {"verify": (0, _receipt())}
    _stub_boundary_ssh(monkeypatch, {"apply": (0, ""), **state})
    runner._fr14_require_network_boundary("alienware")
    _stub_boundary_ssh(monkeypatch, {
        "apply": (0, ""), "verify": (0, _receipt(fingerprint="a" * 64)),
    })
    with pytest.raises(runner.Fixed32BoundaryError, match="CHANGED mid-run"):
        runner._fr14_require_network_boundary("alienware")


def test_fixed32_runtime_mode_calls_the_gate(monkeypatch):
    """The gate must be wired into the fixed32 precondition, not just exist."""
    monkeypatch.setenv("SWE_AGENT_ENV", "instance_image")
    monkeypatch.setenv("SWE_AGENT", "qwen_code")
    seen: list[str | None] = []
    monkeypatch.setattr(
        runner, "_fr14_require_network_boundary",
        lambda host: seen.append(host) or {"gateway": GATEWAY},
    )
    monkeypatch.setattr(runner, "_validate_fixed32_retry_policy", lambda: None)
    runner._validate_fixed32_agent_runtime_mode(remote_host="alienware")
    assert seen == ["alienware"]


# ------------------------------------------------------------- shell contract
def test_boundary_script_is_present_and_executable():
    assert BOUNDARY_SCRIPT.is_file()
    assert BOUNDARY_SCRIPT.stat().st_mode & 0o111, "must be executable"
    assert runner._FR14_BOUNDARY_SCRIPT == BOUNDARY_SCRIPT


def test_boundary_script_parses_and_exposes_the_contract():
    subprocess.run(
        ["bash", "-n", str(BOUNDARY_SCRIPT)], check=True, capture_output=True
    )
    text = BOUNDARY_SCRIPT.read_text(encoding="utf-8")
    for subcommand in ("apply", "verify", "fingerprint", "gateway", "probe",
                       "control", "status", "teardown"):
        assert f"  {subcommand})" in text, subcommand
    # every rule is scoped to the FR14 bridge interface — nothing on this host
    # may match alienware's other docker stacks.
    for rule_array in ("INPUT_RULES", "DOCKER_USER_RULES", "NAT_PRE_RULES",
                       "IP6_INPUT_RULES", "IP6_FORWARD_RULES"):
        assert rule_array in text
    assert "$BRIDGE_IF" in text
    # fast-fail, not timeouts: a hanging turn is a corrupted measurement.
    assert "--reject-with icmp-port-unreachable" in text
    assert "icmp6-port-unreachable" in text


def test_boundary_script_gateway_matches_the_runner_expectation():
    out = subprocess.run(
        ["bash", str(BOUNDARY_SCRIPT), "gateway"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert out == GATEWAY
