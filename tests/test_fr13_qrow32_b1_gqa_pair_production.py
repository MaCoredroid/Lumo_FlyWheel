"""The B1 GQA-pair production/timing path: selector, credential, wiring.

The B1 no-split arm is covered by tests/test_fr13_qrow32_b1_selectors.py; this
file covers only what the GQA-pair arm adds.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts/fr13_patch_fa2_tree_bias.py"
SIDECAR = REPO / "scripts/fr13_qrow32_b1_pass_sidecar.py"
LAUNCHER = REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"
TIMING_RUNNER = REPO / "scripts/fr13_run_b1_fa2_qrow32_gqa_pair_timing.sh"
GATE_ROOT = (
    REPO
    / "output/fr13_fa2_qrow32_gqa_pair_b1_byte_gate_20260812T020320Z"
    / "hydra27_fixed32_fa2_qrow32_gqa_pair_k64_b1_gate_gqapair20260812T020320Z"
)
SEALED_GATE = GATE_ROOT / "qrow32_gqa_pair_live_verification.json"
SEALED_LIVE = GATE_ROOT / "logs/fr13_fa2_qrow32_b1_gqa_pair_live_paged_ab.json"
GQA_PAIR_SENTINEL = 1179791670


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Selector: acceptance, rejection, and how the sentinel reaches the served path
# --------------------------------------------------------------------------


def _selector_namespace(monkeypatch: pytest.MonkeyPatch, *, arm: str = "gqa_pair"):
    namespace = {"os": __import__("os"), "torch": torch}
    patcher = _module(PATCHER, "qrow32_b1_gqa_pair_patcher")
    exec(patcher.FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS, namespace)

    gdn = types.ModuleType("gdn_linear_attn")
    gdn._FR13_FIXED32_PROFILE_CAPTURE_SCOPE = None
    gdn._FR13_FIXED32_PROFILE_MEMORY_SCOPE = False
    gdn._FR13_FIXED32_CAPTURE_CONTEXT = None
    packages = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.mamba": types.ModuleType(
            "vllm.model_executor.layers.mamba"
        ),
        "vllm.model_executor.layers.mamba.gdn_linear_attn": gdn,
    }
    packages["vllm.model_executor.layers.mamba"].gdn_linear_attn = gdn
    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)

    identity = namespace["_fr13_fa2_qrow32_b1_identity"](arm)
    monkeypatch.setenv("FR13_DRAFT_VOCAB_ROOT", "1")
    monkeypatch.setenv("FR13_DRAFT_VOCAB_K", "65536")
    monkeypatch.setenv("FR13_FA2_QROW32_B1_SO_SHA256", identity["candidate_sha256"])
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_SO_SIZE", str(identity["candidate_size"])
    )
    monkeypatch.setenv("FR13_FA2_QROW32_B1_FA2_HEAD", identity["fa2_head"])
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256",
        identity["source_closure_sha256"],
    )
    monkeypatch.setenv("FR13_FA2_QROW32_B1_SOURCE_COMMIT", "1" * 40)
    monkeypatch.setenv("FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256", "2" * 64)
    monkeypatch.setenv("FR13_FA2_QROW32_B1_PRODUCTION_ARM", arm)
    monkeypatch.setenv("FR13_FA2_QROW32_B1_INTERNAL_ATTESTED", "1")
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_EXACT4_TASK_IDS",
        ",".join(namespace["_FR13_FA2_QROW32_B1_CANONICAL_TASK_IDS"]),
    )
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256",
        namespace["_FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256"],
    )
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR_SHA256", "3" * 64
    )
    return namespace, gdn


def _b1_geometry():
    fused_qkv = torch.empty((32, 32, 256), dtype=torch.bfloat16)
    interleaved_kv = torch.empty((1, 2, 1024, 4, 256), dtype=torch.bfloat16)
    return {
        "query": fused_qkv[:, :24, :],
        "key_cache": interleaved_kv[:, 0],
        "value_cache": interleaved_kv[:, 1],
        "cu_seqlens_q": torch.tensor([0, 32], dtype=torch.int32),
        "max_seqlen_q": 32,
        "seqused_k": torch.tensor([32], dtype=torch.int32),
        "max_seqlen_k": 32,
        "causal": False,
        "window_size": [-1, -1],
        "block_table": torch.tensor([[0]], dtype=torch.int32),
        "softcap": 0.0,
        "num_splits": 0,
        "tree_bias": torch.zeros((32, 32), dtype=torch.float32),
    }


def test_production_selector_accepts_gqa_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, _ = _selector_namespace(monkeypatch)
    assert (
        namespace["_fr13_fa2_qrow32_b1_arm"]("FR13_FA2_QROW32_B1_PRODUCTION_ARM")
        == "gqa_pair"
    )


@pytest.mark.parametrize("arm", ["split2", "visibility", "qrow16", "nosplit2"])
def test_production_selector_rejects_non_production_arms(
    monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """Gate-only instruments must never be able to answer production traffic.

    split2 and visibility ARE registered arms, so a membership test against the
    arm registry would have admitted them here; neither was byte-qualified as a
    served dispatch.
    """
    namespace, _ = _selector_namespace(monkeypatch)
    monkeypatch.setenv("FR13_FA2_QROW32_B1_PRODUCTION_ARM", arm)
    with pytest.raises(RuntimeError, match="must be empty or one of"):
        namespace["_fr13_fa2_qrow32_b1_arm"]("FR13_FA2_QROW32_B1_PRODUCTION_ARM")


def test_production_binds_the_gqa_pair_binary_not_the_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GQA-pair arm must refuse to run against the no-split binary.

    This is the failure the arm-blind require_identity() would have allowed:
    the GQA-pair dispatch gate does not exist in the no-split .so, so the
    sentinel would be inert and the run would have timed the incumbent kernel
    while reporting the candidate.
    """
    namespace, gdn = _selector_namespace(monkeypatch)
    incumbent = namespace["_fr13_fa2_qrow32_b1_identity"](None)
    candidate = namespace["_fr13_fa2_qrow32_b1_identity"]("gqa_pair")
    assert incumbent["candidate_sha256"] != candidate["candidate_sha256"]

    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_SO_SHA256", incumbent["candidate_sha256"]
    )
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_SO_SIZE", str(incumbent["candidate_size"])
    )
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256",
        incumbent["source_closure_sha256"],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    gdn._FR13_FIXED32_CAPTURE_CONTEXT = {
        "descriptor": {"num_reqs": 1},
        "graph_id": 7,
    }
    with pytest.raises(RuntimeError, match="pinned identity drifted"):
        namespace["_fr13_fa2_qrow32_b1_production_begin"](
            layer=types.SimpleNamespace(
                layer_name=namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"][0]
            ),
            **_b1_geometry(),
        )


def test_sentinel_reaches_the_served_operand_without_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tag IS the batch stride, and at B1 it costs nothing to apply."""
    namespace, gdn = _selector_namespace(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    gdn._FR13_FIXED32_CAPTURE_CONTEXT = {
        "descriptor": {"num_reqs": 1},
        "graph_id": 11,
    }
    geometry = _b1_geometry()
    original = geometry["tree_bias"]
    layer_name = namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"][0]

    selection = namespace["_fr13_fa2_qrow32_b1_production_begin"](
        layer=types.SimpleNamespace(layer_name=layer_name), **geometry
    )
    assert selection["arm"] == "gqa_pair"
    assert selection["candidate_served"] is True
    assert selection["num_splits"] == 0
    tagged = selection["tree_bias"]
    # The stride the forked flash_api dispatch predicates on.
    assert int(tagged.stride(0)) == GQA_PAIR_SENTINEL
    assert tuple(tagged.shape) == (1, 32, 32)
    # Zero-copy: the served operand aliases the incumbent's bytes.
    assert tagged.data_ptr() == original.data_ptr()
    assert torch.equal(tagged[0], original)


def test_sentinel_matches_the_cpp_dispatch_predicate() -> None:
    """The Python tag and the C++ gate must name the same constant."""
    patcher = _module(PATCHER, "qrow32_b1_gqa_pair_sentinel")
    assert (
        patcher.FIXED32_QUERY_GQA_PAIR32_B1_BATCH_STRIDE_SENTINEL
        == GQA_PAIR_SENTINEL
    )
    gate = patcher.FIXED32_QUERY_GQA_PAIR32_B1_API_GATE
    assert "kFr13Qrow32GqaPairB1BatchStrideSentinel" in gate
    assert "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1(params, stream);" in gate
    assert "            && params.b == 1\n" in gate
    assert "            && params.total_q == 32\n" in gate


def test_capture_end_requires_all_sixteen_layers_in_the_full_b1_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MAX_NUM_SEQS=1 means one FULL graph, so a partial engagement is fatal."""
    namespace, gdn = _selector_namespace(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    gdn._FR13_FIXED32_CAPTURE_CONTEXT = {
        "descriptor": {"num_reqs": 1},
        "graph_id": 21,
    }
    layers = namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"]

    for layer_name in layers[:-1]:
        selection = namespace["_fr13_fa2_qrow32_b1_production_begin"](
            layer=types.SimpleNamespace(layer_name=layer_name), **_b1_geometry()
        )
        namespace["_fr13_fa2_qrow32_b1_production_end"](selection, completed=True)

    with pytest.raises(RuntimeError, match="did not capture all target tree layers"):
        namespace["_fr13_fa2_qrow32_b1_production_capture_end"](
            21, "sig", "FULL", 1
        )

    selection = namespace["_fr13_fa2_qrow32_b1_production_begin"](
        layer=types.SimpleNamespace(layer_name=layers[-1]), **_b1_geometry()
    )
    namespace["_fr13_fa2_qrow32_b1_production_end"](selection, completed=True)
    record = namespace["_fr13_fa2_qrow32_b1_production_record"](
        arm="gqa_pair", runtime_mode="FULL", graph_id=21,
        graph_signature="sig", layers=sorted(layers), calls=16,
    )
    identity = namespace["_fr13_fa2_qrow32_b1_identity"]("gqa_pair")
    # The engagement record must attest the arm that ran, not the incumbent.
    assert record["arm"] == "gqa_pair"
    assert record["selector_sentinel"] == GQA_PAIR_SENTINEL
    assert record["candidate_so_size"] == identity["candidate_size"]
    assert (
        record["fa2_source_closure_sha256"] == identity["source_closure_sha256"]
    )
    assert record["dispatch"] == "qrow32 B1 GQA-pair exact geometry; no fallback"
    assert record["layer_count"] == 16


def test_capture_end_refuses_a_non_b1_full_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, _ = _selector_namespace(monkeypatch)
    with pytest.raises(RuntimeError, match="captured outside FULL B1"):
        namespace["_fr13_fa2_qrow32_b1_production_capture_end"](21, "sig", "FULL", 4)


# --------------------------------------------------------------------------
# Credential
# --------------------------------------------------------------------------


def _sealed_source_commit() -> str:
    return json.loads(SEALED_GATE.read_text(encoding="ascii"))["source_commit"]


def test_credential_validators_accept_the_real_sealed_gate() -> None:
    """The sealed b5dab3f0 artifact and the evidence it binds must validate."""
    module = _module(SIDECAR, "qrow32_b1_sidecar_sealed")
    gate_payload, gate_raw = module.load_json(SEALED_GATE)
    assert (
        module._digest(gate_raw)
        == "b5dab3f0a0939c30ec52a3326491b33589ff21478846266ecb4430e15a73120c"
    )
    summary = module.validate_gqa_pair_gate(
        gate_payload, source_commit=gate_payload["source_commit"]
    )

    live_payload, live_raw = module.load_json(SEALED_LIVE)
    # The gate binds its evidence by digest; follow it.
    assert module._digest(live_raw) == summary["live_result_sha256"]
    live_summary = module.validate_live_result(
        live_payload,
        candidate_sha256=module.GQA_PAIR_CANDIDATE_SHA256,
        arm="gqa_pair",
    )
    assert live_summary["layers_sha256"] == summary["layers_sha256"]
    # What the gate actually established.
    assert live_payload["output_raw_byte_mismatches"] == 0
    assert live_payload["lse_raw_byte_mismatches"] == 0
    assert live_payload["layer_count"] == 16
    assert live_payload["selector_sentinel"] == GQA_PAIR_SENTINEL
    assert live_payload["served_return"] == "qrow16 captured graph output unchanged"


def test_credential_refuses_a_gate_from_another_commit() -> None:
    """A gate that predates the plumbing did not exercise the serving code."""
    module = _module(SIDECAR, "qrow32_b1_sidecar_commit")
    payload, _ = module.load_json(SEALED_GATE)
    with pytest.raises(ValueError, match="not produced at the production plumbing commit"):
        module.validate_gqa_pair_gate(payload, source_commit="f" * 40)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("performance_measurement", True, "eligibility fields were rewritten"),
        ("served_return", "candidate output served", "eligibility fields were rewritten"),
        ("status", "FAIL", "is not a PASS"),
        ("schema", "fr13.other.v1", "schema drifted"),
        ("candidate_so_sha256", "a" * 64, "candidate identity drifted"),
        ("topology", "tail6_fixed32", "operating point drifted"),
        ("batch_size", 4, "operating point drifted"),
        ("subset_sha256", "b" * 64, "operating point drifted"),
    ],
)
def test_credential_is_fail_closed_on_the_gate_body(
    field: str, value: object, message: str
) -> None:
    module = _module(SIDECAR, "qrow32_b1_sidecar_failclosed")
    payload, _ = module.load_json(SEALED_GATE)
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        module.validate_gqa_pair_gate(payload, source_commit=payload["source_commit"])


def test_credential_requires_every_bound_digest() -> None:
    module = _module(SIDECAR, "qrow32_b1_sidecar_digests")
    for key in (
        "live_result_sha256",
        "layers_sha256",
        "health_sha256",
        "traffic_audit_sha256",
        "block_map_sha256",
        "diagnostic_binding_sha256",
        "patch_source_sha256",
    ):
        payload, _ = module.load_json(SEALED_GATE)
        payload.pop(key)
        with pytest.raises(ValueError, match=f"GQA-pair gate {key}"):
            module.validate_gqa_pair_gate(
                payload, source_commit=payload["source_commit"]
            )


def _issue_gqa_pair_credential(module, tmp_path: Path):
    """Issue a real GQA-pair credential the way the host launcher does."""
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(b"gqa-pair-b1-candidate")
    module.GQA_PAIR_CANDIDATE_SIZE = candidate.stat().st_size
    module.GQA_PAIR_CANDIDATE_SHA256 = module.sha256_file(candidate)

    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO, check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    patch_sha256 = module.sha256_file(PATCHER)

    live_payload, _ = module.load_json(SEALED_LIVE)
    live_payload["candidate_so_sha256"] = module.GQA_PAIR_CANDIDATE_SHA256
    live_payload["candidate_so_size"] = module.GQA_PAIR_CANDIDATE_SIZE
    live_payload["source_commit"] = source_commit
    live_payload["patch_source_sha256"] = patch_sha256
    live = tmp_path / "live.json"
    live.write_bytes(module.canonical_bytes(live_payload) + b"\n")

    layers_sha256 = module.validate_live_result(
        live_payload,
        candidate_sha256=module.GQA_PAIR_CANDIDATE_SHA256,
        arm="gqa_pair",
    )["layers_sha256"]

    gate_payload, _ = module.load_json(SEALED_GATE)
    gate_payload["candidate_so_sha256"] = module.GQA_PAIR_CANDIDATE_SHA256
    gate_payload["candidate_so_size"] = module.GQA_PAIR_CANDIDATE_SIZE
    gate_payload["source_commit"] = source_commit
    gate_payload["patch_source_sha256"] = patch_sha256
    gate_payload["live_result_sha256"] = module.sha256_file(live)
    gate_payload["layers_sha256"] = layers_sha256
    gate = tmp_path / "gate.json"
    gate.write_bytes(module.canonical_bytes(gate_payload) + b"\n")

    out = tmp_path / "pass.json"
    issued = module.issue_gqa_pair_sidecar(
        gate=gate,
        expected_gate_sha256=module.sha256_file(gate),
        live_result=live,
        candidate_so=candidate,
        expected_candidate_sha256=module.GQA_PAIR_CANDIDATE_SHA256,
        arm="gqa_pair",
        patch_source=PATCHER,
        expected_source_commit=source_commit,
        out=out,
    )
    return candidate, source_commit, patch_sha256, gate, live, out, issued


def test_credential_round_trips_and_records_what_the_gate_established(
    tmp_path: Path,
) -> None:
    module = _module(SIDECAR, "qrow32_b1_sidecar_roundtrip")
    candidate, source_commit, patch_sha256, gate, _, out, issued = (
        _issue_gqa_pair_credential(module, tmp_path)
    )
    assert issued["arm"] == "gqa_pair"
    assert issued["selector_sentinel"] == GQA_PAIR_SENTINEL
    assert issued["output_raw_byte_mismatches"] == 0
    assert issued["lse_raw_byte_mismatches"] == 0
    assert issued["layer_count"] == 16
    # The gate's own eligibility statements are carried, not reinterpreted.
    assert issued["gate_performance_measurement"] is False
    assert issued["gate_served_return"] == "qrow16 captured graph output unchanged"
    # The scope gap is recorded rather than laundered.
    assert issued["gate_task_ids"] == ["astropy__astropy-12907"]
    assert len(issued["production_task_ids"]) == 4
    assert issued["gate_scope_narrower_than_production"] is True
    assert issued["gate_sha256"] == module.sha256_file(gate)

    verified = module.verify_gqa_pair_sidecar(
        sidecar_path=out,
        expected_sidecar_sha256=module.sha256_file(out),
        candidate_so=candidate,
        expected_candidate_sha256=module.GQA_PAIR_CANDIDATE_SHA256,
        arm="gqa_pair",
        patch_source=PATCHER,
        expected_source_commit=source_commit,
        expected_patch_source_sha256=patch_sha256,
    )
    assert verified["canonical_sha256"] == issued["canonical_sha256"]


def test_credential_verify_needs_no_git_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pinned runtime image ships no git; verification must not need one.

    This is the failure that killed the B4 candidate arm at container init and
    was still latent on the B1 path. The GQA-pair verify must inherit the fix.
    """
    module = _module(SIDECAR, "qrow32_b1_sidecar_nogit")
    candidate, source_commit, patch_sha256, _, _, out, issued = (
        _issue_gqa_pair_credential(module, tmp_path)
    )
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    def _no_subprocess(*args, **kwargs):
        raise AssertionError("verify must not shell out (git is absent)")

    monkeypatch.setattr(module.subprocess, "run", _no_subprocess)

    verified = module.verify_gqa_pair_sidecar(
        sidecar_path=out,
        expected_sidecar_sha256=module.sha256_file(out),
        candidate_so=candidate,
        expected_candidate_sha256=module.GQA_PAIR_CANDIDATE_SHA256,
        arm="gqa_pair",
        patch_source=PATCHER,
        expected_source_commit=source_commit,
        expected_patch_source_sha256=patch_sha256,
    )
    assert verified["patch_source_sha256"] == patch_sha256


def test_credential_refuses_a_drifted_gate_artifact(tmp_path: Path) -> None:
    """Swapping the gate under a declared digest must fail closed."""
    module = _module(SIDECAR, "qrow32_b1_sidecar_drift")
    candidate, source_commit, _, gate, live, _, _ = _issue_gqa_pair_credential(
        module, tmp_path
    )
    declared = module.sha256_file(gate)
    payload, _ = module.load_json(gate)
    payload["health_sha256"] = "c" * 64
    gate.write_bytes(module.canonical_bytes(payload) + b"\n")
    assert module.sha256_file(gate) != declared

    with pytest.raises(ValueError, match="gate raw SHA-256 mismatch"):
        module.validate_gqa_pair_binding(
            gate=gate,
            expected_gate_sha256=declared,
            live_result=live,
            candidate_so=candidate,
            expected_candidate_sha256=module.GQA_PAIR_CANDIDATE_SHA256,
            arm="gqa_pair",
            patch_source=PATCHER,
            expected_source_commit=source_commit,
        )


def test_credential_refuses_a_live_result_the_gate_did_not_bind(
    tmp_path: Path,
) -> None:
    """The wrapper cannot be paired with a body it never referenced."""
    module = _module(SIDECAR, "qrow32_b1_sidecar_unbound")
    candidate, source_commit, _, gate, live, _, _ = _issue_gqa_pair_credential(
        module, tmp_path
    )
    payload, _ = module.load_json(live)
    payload["seq_len"] = int(payload.get("seq_len", 1)) + 1
    live.write_bytes(module.canonical_bytes(payload) + b"\n")

    with pytest.raises(ValueError, match="not the one the gate bound"):
        module.validate_gqa_pair_binding(
            gate=gate,
            expected_gate_sha256=module.sha256_file(gate),
            live_result=live,
            candidate_so=candidate,
            expected_candidate_sha256=module.GQA_PAIR_CANDIDATE_SHA256,
            arm="gqa_pair",
            patch_source=PATCHER,
            expected_source_commit=source_commit,
        )


def test_credential_refuses_the_wrong_candidate_binary(tmp_path: Path) -> None:
    module = _module(SIDECAR, "qrow32_b1_sidecar_wrongso")
    candidate, source_commit, _, gate, live, _, _ = _issue_gqa_pair_credential(
        module, tmp_path
    )
    impostor = tmp_path / "impostor.so"
    impostor.write_bytes(b"not-the-candidate")

    with pytest.raises(ValueError, match="candidate SO identity mismatch"):
        module.validate_gqa_pair_binding(
            gate=gate,
            expected_gate_sha256=module.sha256_file(gate),
            live_result=live,
            candidate_so=impostor,
            expected_candidate_sha256=module.GQA_PAIR_CANDIDATE_SHA256,
            arm="gqa_pair",
            patch_source=PATCHER,
            expected_source_commit=source_commit,
        )


def test_credential_refuses_a_non_gqa_pair_arm(tmp_path: Path) -> None:
    module = _module(SIDECAR, "qrow32_b1_sidecar_wrongarm")
    candidate, source_commit, _, gate, live, _, _ = _issue_gqa_pair_credential(
        module, tmp_path
    )
    with pytest.raises(ValueError, match="arm must be gqa_pair"):
        module.validate_gqa_pair_binding(
            gate=gate,
            expected_gate_sha256=module.sha256_file(gate),
            live_result=live,
            candidate_so=candidate,
            expected_candidate_sha256=module.GQA_PAIR_CANDIDATE_SHA256,
            arm="nosplit",
            patch_source=PATCHER,
            expected_source_commit=source_commit,
        )


def test_gqa_pair_credential_commands_exist_and_are_separate() -> None:
    """The incumbent no-split credential path must stay untouched."""
    text = SIDECAR.read_text(encoding="utf-8")
    for command in ("validate-gqa-pair", "issue-gqa-pair", "verify-gqa-pair"):
        assert f'"{command}"' in text
    module = _module(SIDECAR, "qrow32_b1_sidecar_cli")
    # The no-split commands still refuse any arm but nosplit.
    assert module.ARM == "nosplit"
    with pytest.raises(ValueError, match="production arm must be nosplit"):
        module.issue_sidecar(
            live_result=SEALED_LIVE,
            expected_live_sha256="0" * 64,
            candidate_so=SIDECAR,
            expected_candidate_sha256="0" * 64,
            arm="gqa_pair",
            patch_source=PATCHER,
            expected_source_commit="0" * 40,
            out=Path("/nonexistent/pass.json"),
        )


# --------------------------------------------------------------------------
# Launcher wiring
# --------------------------------------------------------------------------


def test_launcher_admits_gqa_pair_and_pins_it_to_its_own_binary() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert (
        'FR13_FA2_QROW32_B1_PRODUCTION_ARM must be empty, nosplit, or gqa_pair'
        in text
    )
    assert '""|nosplit|gqa_pair) ;;' in text
    # Pin selection must fall back to the production arm; keying on the live
    # arm alone checked every production run against the incumbent pins.
    assert "_FR13_FA2_QROW32_B1_PIN_ARM=$FR13_FA2_QROW32_B1_LIVE_AB_ARM" in text
    assert (
        "_FR13_FA2_QROW32_B1_PIN_ARM=$FR13_FA2_QROW32_B1_PRODUCTION_ARM" in text
    )
    assert 'case "$_FR13_FA2_QROW32_B1_PIN_ARM" in' in text
    # The GQA-pair branch still carries its own pins.
    assert (
        '"$FR13_FA2_QROW32_B1_SO_SHA256" == '
        '"3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae"'
        in text
    )


def test_runtime_contract_resolves_the_production_arm_to_its_own_binary() -> None:
    """The in-container contract must admit gqa_pair AND pin its binary.

    It carried the same two defects the launcher did: the arm was rejected
    outright, and the binary was resolved from the LIVE arm only, so a
    production launch (which has no live arm) would have been required to
    present the split2/incumbent .so.
    """
    # The contract imports its sibling topology module, so scripts/ must be
    # importable before it loads.
    sys.path.insert(0, str(REPO / "scripts"))
    contract = _module(
        REPO / "scripts/fr13_fixed32_contract.py", "b1_gqa_pair_contract"
    )
    env = {
        "FR13_FA2_QROW16_LIVE_PAGED_AB": "0",
        "FR13_FA2_QROW16_PRODUCTION": "0",
        "FR13_FA2_QROW32_LIVE_PAGED_AB": "0",
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "",
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "gqa_pair",
        "FR13_FA2_QROW32_B1_SO_SHA256": contract.QROW32_B1_GQA_PAIR_FA2_SHA256,
    }
    assert contract._expected_runtime_fa2_identity(env) == (
        contract.QROW32_B1_GQA_PAIR_FA2_SIZE,
        contract.QROW32_B1_GQA_PAIR_FA2_SHA256,
    )

    # Presenting the incumbent binary to the GQA-pair production arm fails.
    env["FR13_FA2_QROW32_B1_SO_SHA256"] = contract.QROW32_B1_SPLIT2_FA2_SHA256
    with pytest.raises(contract.ContractError, match="not the pinned candidate"):
        contract._expected_runtime_fa2_identity(env)

    # The no-split production arm still resolves to the incumbent binary.
    env["FR13_FA2_QROW32_B1_PRODUCTION_ARM"] = "nosplit"
    assert contract._expected_runtime_fa2_identity(env) == (
        contract.QROW32_B1_SPLIT2_FA2_SIZE,
        contract.QROW32_B1_SPLIT2_FA2_SHA256,
    )

    # And the gate-only instruments are still refused as production arms.
    for refused in ("split2", "visibility"):
        env["FR13_FA2_QROW32_B1_PRODUCTION_ARM"] = refused
        with pytest.raises(
            contract.ContractError,
            match="must be empty, nosplit, or gqa_pair",
        ):
            contract._expected_runtime_fa2_identity(env)


def test_launcher_pairs_the_timing_arm_with_the_selector() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert (
        'FR13_FA2_QROW32_B1_TIMING_ARM must be empty, qrow16_stock, or gqa_pair'
        in text
    )
    for message in (
        "gqa_pair timing arm must serve the candidate",
        "qrow16_stock timing arm must carry no B1 selector",
        "qrow16_stock timing arm requires qrow16 production",
        "GQA-pair production requires the gqa_pair timing arm",
    ):
        assert message in text


def test_launcher_issues_and_verifies_the_gqa_pair_credential() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "fr13_qrow32_b1_pass_sidecar.py issue-gqa-pair" in text
    assert "--gate \"$FR13_FA2_QROW32_B1_GQA_PAIR_GATE_HOST\"" in text
    assert (
        "--live-result \"$FR13_FA2_QROW32_B1_GQA_PAIR_LIVE_RESULT_JSON\"" in text
    )
    # In-container verification selects the GQA-pair subcommand...
    assert "_fr13_b1_verify_command=verify-gqa-pair" in text
    # ...and stays digest-based, because the image has no git.
    assert (
        '--expected-patch-source-sha256 "\\$FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256"'
        in text
    )


def test_launcher_passes_the_gate_binding_into_the_container() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for name in (
        "FR13_FA2_QROW32_B1_TIMING_ARM",
        "FR13_FA2_QROW32_B1_GQA_PAIR_GATE_JSON",
        "FR13_FA2_QROW32_B1_GQA_PAIR_GATE_SHA256",
    ):
        assert f"  -e {name}=" in text
        assert f"  {name}\n" in text


def test_launcher_keeps_the_b1_selectors_mutually_exclusive() -> None:
    """One private FA2 selector per call site, enforced at the shared anchor."""
    text = LAUNCHER.read_text(encoding="utf-8")
    assert (
        "FR13 qrow32 B1 live A/B and production arms are mutually exclusive"
        in text
    )
    assert (
        "FR13 qrow32 B1 and existing FA2 private selectors are mutually exclusive"
        in text
    )
    assert (
        "FR13 qrow32 B4 and existing FA2 private selectors are mutually exclusive"
        in text
    )
    # The patcher installs exactly one of these at the tree_attn call site.
    anchor = re.search(
        r"elif \[\[ -n \"\$FR13_FA2_QROW32_B1_PRODUCTION_ARM\" \]\]; "
        r"then printf '%s' '--fixed32-query-tile32-b1-production'",
        text,
    )
    assert anchor is not None


def test_launcher_keeps_credential_env_private() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "FR13 qrow32 production sidecar credentials are launcher-private" in text
    assert "FR13 qrow32 B1 internal attestation is launcher-private" in text


# --------------------------------------------------------------------------
# Timing runner
# --------------------------------------------------------------------------


def test_timing_runner_is_disabled_by_default_and_pass_gated() -> None:
    text = TIMING_RUNNER.read_text(encoding="utf-8")
    assert "FR13_RUN_B1_QROW32_GQA_PAIR_TIMING must be exactly 0 or 1" in text
    assert 'QROW32_GQA_PAIR_B1_GATE_JSON:?' in text
    assert 'QROW32_GQA_PAIR_B1_LIVE_RESULT_JSON:?' in text
    assert "validate-gqa-pair" in text
    assert "verify-gqa-pair" in text


def test_timing_runner_runs_exact4_at_batch_one_in_full_graphs() -> None:
    text = TIMING_RUNNER.read_text(encoding="utf-8")
    assert (
        "SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
        in text
    )
    assert "export BSIZE=1" in text
    assert "export CONC=1" in text
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in text
    assert "ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in text
    for pin in (
        "'MAX_NUM_SEQS=1'",
        "'SWE_CONCURRENCY=1'",
        "'ENFORCE_EAGER=0'",
        "'CUDAGRAPH_MODE=FULL_AND_PIECEWISE'",
    ):
        assert pin in text


def test_timing_runner_declares_the_two_binary_delta_honestly() -> None:
    """The pair is not single-variable and must not claim to be."""
    text = TIMING_RUNNER.read_text(encoding="utf-8")
    assert '"single_variable": False' in text
    assert '"arm_delta_disclosure"' in text
    assert '"why_not_single_variable"' in text
    assert '"residual_confound"' in text
    assert '"candidate_only_overhead": "none"' in text
    assert "arm_delta_spans_two_binaries=1" in text
    assert "single_variable=0" in text
    # And it must refuse the degenerate pair.
    assert "stock and candidate binaries are identical" in text


def test_timing_runner_gates_reduction_and_forbids_sentinel_leak() -> None:
    text = TIMING_RUNNER.read_text(encoding="utf-8")
    assert "--work-census" in text
    assert '"work_census_gate"' in text
    assert "emitted a GQA-pair engagement on the stock arm" in text


def test_timing_runner_verdict_is_step_wall_with_floor_ratio() -> None:
    text = TIMING_RUNNER.read_text(encoding="utf-8")
    assert '"decision_metric": "step_wall_ms"' in text
    assert '"step_wall_ms_delta"' in text
    assert '"step_wall_ms_delta_frac"' in text
    assert '"step_wall_to_optimistic_floor_ratio"' in text
    assert "promotion_verdict(stock_phases, candidate_phases)" in text
    # Batch-1 equivalence of per-request and aggregate is asserted, not assumed.
    assert "is not batch-1: per-request != aggregate" in text
    assert '"formal_floor_acceptance_eligible": False' in text


def test_timing_runner_summary_argv_arity_matches_its_invocation() -> None:
    """A mis-numbered argv slice would silently mislabel the summary."""
    text = TIMING_RUNNER.read_text(encoding="utf-8")
    invocation = text.split('"$PYTHON_BIN" - \\', 1)[1].split("<<'PY'", 1)[0]
    # Count shell words passed as arguments (one per line, backslash-continued).
    args = [
        line.strip().rstrip("\\").strip()
        for line in invocation.splitlines()
        if line.strip().rstrip("\\").strip()
    ]
    tokens: list[str] = []
    for chunk in args:
        tokens.extend(part for part in chunk.split(" ") if part)
    assert len(tokens) == 38, tokens
    body = re.search(r"<<'PY'\n(.*?)\nPY\n", text, re.S).group(1)
    highest = max(
        int(match)
        for match in re.findall(r"sys\.argv\[(\d+)(?::\d+)?\]", body)
    )
    slices = [
        int(end)
        for end in re.findall(r"sys\.argv\[\d+:(\d+)\]", body)
    ]
    # argv[0] is "-", so the last positional is index 38.
    assert max([highest] + slices) <= 39
    assert max(slices) == 39


# --------------------------------------------------------------------------
# The launcher's host-side FA2 identity preflight (inline Python, pre-docker)
# --------------------------------------------------------------------------


_STUB_CONTRACT = '''
import hashlib
from pathlib import Path

FA2_REPO_RELATIVE = "stock_fa2.so"
IMAGE_REFERENCE = "stub-image@sha256:{image}"
FA2_SHA256 = "{stock_sha}"
FA2_SIZE = {stock_size}
QROW32_B4_GQA_PAIR_FA2_SHA256 = "b4" * 32
QROW32_B4_GQA_PAIR_FA2_SIZE = 1
QROW32_B1_SPLIT2_FA2_SHA256 = "{split2_sha}"
QROW32_B1_SPLIT2_FA2_SIZE = {split2_size}
QROW32_B1_VISIBILITY_FA2_SHA256 = "{vis_sha}"
QROW32_B1_VISIBILITY_FA2_SIZE = {vis_size}
QROW32_B1_GQA_PAIR_FA2_SHA256 = "{pair_sha}"
QROW32_B1_GQA_PAIR_FA2_SIZE = {pair_size}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _docker_image_record():
    return None


def fixed32_tree_text():
    return "TREE"


def speculative_config_text():
    return "SPEC"
'''


def _launcher_fa2_preflight_source() -> str:
    text = LAUNCHER.read_text(encoding="utf-8")
    marker = '"$_FR13_FA2_QROW32_B4_CANDIDATE_MODE" <<\'PY\'\n'
    body = text.split(marker, 1)[1]
    return body.split("\nPY\n", 1)[0]


def _run_launcher_fa2_preflight(
    tmp_path: Path, *, arm_env: dict, candidate_name: str
):
    """Execute the launcher's real preflight against stubbed pins.

    The genuine pinned binaries are ~300 MB, so the contract module is
    shadowed with a stub whose pins point at small fixtures. The code under
    test is the launcher's own bytes, extracted verbatim.
    """
    import hashlib
    import os

    stage = tmp_path / "stage"
    stage.mkdir(parents=True)
    binaries = {}
    for name, payload in {
        "stock_fa2.so": b"stock",
        "split2": b"split2-binary",
        "visibility": b"visibility-binary",
        "gqa_pair": b"gqa-pair-binary",
    }.items():
        path = stage / f"{name}.so" if name != "stock_fa2.so" else stage / name
        path.write_bytes(payload)
        binaries[name] = (
            path,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
        )

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir(parents=True)
    (stub_dir / "fr13_fixed32_contract.py").write_text(
        _STUB_CONTRACT.format(
            image="0" * 64,
            stock_sha=binaries["stock_fa2.so"][1],
            stock_size=binaries["stock_fa2.so"][2],
            split2_sha=binaries["split2"][1],
            split2_size=binaries["split2"][2],
            vis_sha=binaries["visibility"][1],
            vis_size=binaries["visibility"][2],
            pair_sha=binaries["gqa_pair"][1],
            pair_size=binaries["gqa_pair"][2],
        ),
        encoding="utf-8",
    )
    script = tmp_path / "preflight.py"
    script.write_text(_launcher_fa2_preflight_source(), encoding="utf-8")

    fa2_path, fa2_sha, _ = binaries[candidate_name]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(stub_dir)
    env.update(arm_env)
    return subprocess.run(
        [
            sys.executable,
            str(script),
            str(stage),                     # repo
            "stub-image@sha256:" + "0" * 64,  # image
            str(fa2_path),                  # FORKED_FA2_SO
            "TREE",
            "SPEC",
            "0",                            # qrow16 candidate
            "",                             # qrow16 sha
            "0",                            # qrow32 candidate
            "",                             # qrow32 sha
            "1",                            # qrow32 B1 candidate
            fa2_sha,                        # declared B1 sha
            "0",                            # qrow32 B4 candidate
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def test_launcher_preflight_pins_the_production_arm_not_the_live_arm(
    tmp_path: Path,
) -> None:
    """A GQA-pair PRODUCTION launch has no live arm.

    Keying the host-side binary-identity preflight on
    FR13_FA2_QROW32_B1_LIVE_AB_ARM alone resolved the empty live arm to the
    incumbent split2 pins, so the candidate .so was rejected with
    "binary identity is not qualified" and the candidate arm of the timing
    pair could never boot. This is the same arm-blind defect fixed in the
    bash pin case, the patcher and the runtime contract.
    """
    result = _run_launcher_fa2_preflight(
        tmp_path,
        arm_env={
            "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "",
            "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "gqa_pair",
        },
        candidate_name="gqa_pair",
    )
    assert result.returncode == 0, result.stderr
    assert "not qualified" not in result.stderr


def test_launcher_preflight_still_refuses_a_mismatched_binary(
    tmp_path: Path,
) -> None:
    """The widened resolution must not become permissive."""
    result = _run_launcher_fa2_preflight(
        tmp_path,
        arm_env={
            "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "",
            "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "gqa_pair",
        },
        candidate_name="split2",
    )
    assert result.returncode != 0
    assert "B1 binary identity is not qualified" in result.stderr


def test_launcher_preflight_keeps_the_live_arm_and_nosplit_resolutions(
    tmp_path: Path,
) -> None:
    """The live arm still wins, and the empty/nosplit case is unchanged."""
    # A live visibility gate resolves to the visibility binary even though a
    # production arm is not set.
    ok = _run_launcher_fa2_preflight(
        tmp_path / "a",
        arm_env={
            "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "visibility",
            "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "",
        },
        candidate_name="visibility",
    )
    assert ok.returncode == 0, ok.stderr

    # The no-split production arm keeps the incumbent split2 identity.
    nosplit = _run_launcher_fa2_preflight(
        tmp_path / "b",
        arm_env={
            "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "",
            "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "nosplit",
        },
        candidate_name="split2",
    )
    assert nosplit.returncode == 0, nosplit.stderr

    # A live arm and a production arm cannot disagree in practice, but if a
    # caller set both the live arm must decide, because it decides the .so.
    both = _run_launcher_fa2_preflight(
        tmp_path / "c",
        arm_env={
            "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "visibility",
            "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "gqa_pair",
        },
        candidate_name="visibility",
    )
    assert both.returncode == 0, both.stderr


def test_launcher_exports_the_arm_variables_to_the_preflight() -> None:
    """os.environ only sees exported variables."""
    text = LAUNCHER.read_text(encoding="utf-8")
    export_line = (
        "export FR13_FA2_QROW32_B1_LIVE_AB_ARM "
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM"
    )
    assert export_line in text
    assert text.index(export_line) < text.index(
        'fixed32 qrow32 B1 binary identity is not qualified'
    )


def test_build_checklist_lists_every_gqa_pair_pin_site() -> None:
    """A rebuild that follows the checklist must not leave a pin stale.

    Each pinned site is a hard-fail comparison; this campaign has already
    been burned by a rebuild that updated only some of them.
    """
    build = REPO / "scripts/fr13_build_fa2_qrow32_gqa_pair_b1_sm121a.sh"
    checklist = build.read_text(encoding="utf-8")
    pinned = subprocess.run(
        [
            "grep",
            "-rl",
            "3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae",
            str(REPO / "scripts"),
        ],
        capture_output=True,
        text=True,
    ).stdout.split()
    for path in pinned:
        name = Path(path).name
        if name == build.name or Path(path).suffix not in (".sh", ".py"):
            continue
        assert name in checklist, f"{name} pins the candidate but is not in the checklist"


def test_timing_runner_is_in_the_runtime_manifest() -> None:
    """The manifest is this pair's provenance record; the runner is in it."""
    manifest = (REPO / "scripts/fr13_runtime_manifest.py").read_text(
        encoding="utf-8"
    )
    assert '"scripts/fr13_run_b1_fa2_qrow32_gqa_pair_timing.sh",' in manifest


# --------------------------------------------------------------------------
# Bypass at the operating points the runtime visits before the B1 graph exists
# --------------------------------------------------------------------------


def test_boot_profile_forward_bypasses_instead_of_killing_engine_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM's memory-profiling warmup is neither capturing nor eager.

    With ENFORCE_EAGER=0 it runs at init before any graph exists. Raising
    there killed engine-core init on the candidate arm; the legacy nosplit
    path never hit it because it only ever ran eager-pinned.
    """
    namespace, _gdn = _selector_namespace(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.delenv("ENFORCE_EAGER", raising=False)
    geometry = _b1_geometry()
    bias = geometry["tree_bias"]

    selection = namespace["_fr13_fa2_qrow32_b1_production_begin"](
        layer=types.SimpleNamespace(
            layer_name=namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"][0]
        ),
        **geometry,
    )

    assert selection["candidate_served"] is False
    assert selection["bypass_reason"] == "outside_capture"
    # The UNTAGGED operand and the caller's num_splits: stock dispatch.
    assert selection["tree_bias"] is bias
    assert selection["num_splits"] == 0
    namespace["_fr13_fa2_qrow32_b1_production_end"](selection, completed=True)
    assert (
        namespace["_FR13_FA2_QROW32_B1_BYPASS_COUNTS"]["outside_capture"] == 1
    )
    # Nothing engaged, so no graph was recorded.
    assert namespace["_FR13_FA2_QROW32_B1_PRODUCTION_GRAPHS"] == {}


def test_the_full_b1_capture_still_engages_the_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bypassing outside capture must not weaken the qualified point."""
    namespace, gdn = _selector_namespace(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    gdn._FR13_FIXED32_CAPTURE_CONTEXT = {
        "descriptor": {"num_reqs": 1},
        "graph_id": 31,
    }

    selection = namespace["_fr13_fa2_qrow32_b1_production_begin"](
        layer=types.SimpleNamespace(
            layer_name=namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"][0]
        ),
        **_b1_geometry(),
    )

    assert selection["candidate_served"] is True
    assert selection.get("bypass_reason") is None
    assert int(selection["tree_bias"].stride(0)) == GQA_PAIR_SENTINEL
    namespace["_fr13_fa2_qrow32_b1_production_end"](selection, completed=True)


def test_a_bypass_carrying_the_sentinel_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bypass accounted as stock must not have engaged the candidate."""
    namespace, _gdn = _selector_namespace(monkeypatch)
    leaked = namespace["_fr13_fa2_qrow32_b1_candidate_tree_bias"](
        torch.zeros((32, 32), dtype=torch.float32), "gqa_pair"
    )
    assert int(leaked.stride(0)) == GQA_PAIR_SENTINEL

    with pytest.raises(RuntimeError, match="carried the candidate sentinel"):
        namespace["_fr13_fa2_qrow32_b1_production_end"](
            {
                "arm": "gqa_pair",
                "candidate_served": False,
                "bypass_reason": "outside_capture",
                "tree_bias": leaked,
                "num_splits": 0,
            },
            completed=True,
        )


def test_a_forged_bypass_selection_is_rejected_by_the_end_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, _gdn = _selector_namespace(monkeypatch)
    with pytest.raises(RuntimeError, match="bypass drifted"):
        namespace["_fr13_fa2_qrow32_b1_production_end"](
            {
                "arm": "gqa_pair",
                "candidate_served": True,
                "bypass_reason": "outside_capture",
            },
            completed=True,
        )
    with pytest.raises(RuntimeError, match="bypass drifted"):
        namespace["_fr13_fa2_qrow32_b1_production_end"](
            {
                "arm": "gqa_pair",
                "candidate_served": False,
                "bypass_reason": "invented_reason",
            },
            completed=True,
        )


def test_the_legacy_nosplit_arm_is_unaffected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nosplit keeps its own sentinel when captured, and bypasses the same."""
    namespace, gdn = _selector_namespace(monkeypatch, arm="nosplit")
    nosplit_sentinel = namespace["_FR13_FA2_QROW32_B1_ARMS"]["nosplit"]["sentinel"]
    assert nosplit_sentinel != GQA_PAIR_SENTINEL
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.delenv("ENFORCE_EAGER", raising=False)
    geometry = _b1_geometry()
    bias = geometry["tree_bias"]
    bypass = namespace["_fr13_fa2_qrow32_b1_production_begin"](
        layer=types.SimpleNamespace(
            layer_name=namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"][0]
        ),
        **geometry,
    )
    assert bypass["bypass_reason"] == "outside_capture"
    assert bypass["tree_bias"] is bias
    namespace["_fr13_fa2_qrow32_b1_production_end"](bypass, completed=True)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    gdn._FR13_FIXED32_CAPTURE_CONTEXT = {
        "descriptor": {"num_reqs": 1},
        "graph_id": 41,
    }
    selection = namespace["_fr13_fa2_qrow32_b1_production_begin"](
        layer=types.SimpleNamespace(
            layer_name=namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"][0]
        ),
        **_b1_geometry(),
    )
    assert selection["candidate_served"] is True
    assert int(selection["tree_bias"].stride(0)) == nosplit_sentinel
    namespace["_fr13_fa2_qrow32_b1_production_end"](selection, completed=True)


def test_the_engagement_record_reports_the_bypass_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bypasses are declared in the artifact, not hidden."""
    namespace, _gdn = _selector_namespace(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.delenv("ENFORCE_EAGER", raising=False)
    for layer_name in namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"]:
        selection = namespace["_fr13_fa2_qrow32_b1_production_begin"](
            layer=types.SimpleNamespace(layer_name=layer_name), **_b1_geometry()
        )
        namespace["_fr13_fa2_qrow32_b1_production_end"](selection, completed=True)

    record = namespace["_fr13_fa2_qrow32_b1_production_record"](
        arm="gqa_pair", runtime_mode="FULL", graph_id=51,
        graph_signature="sig",
        layers=sorted(namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"]),
        calls=16,
    )

    # One profiling pass over the 16 target layers; the live boot signature is
    # two passes, i.e. outside_capture == 32.
    assert record["bypass_counts"]["outside_capture"] == 16
    assert record["candidate_scope"] == "final_fixed32_b1_full_graph_only"
    assert record["candidate_served"] is True


def test_the_helper_block_no_longer_aborts_outside_capture() -> None:
    """The installed block is what runs in the container."""
    patcher = _module(PATCHER, "qrow32_b1_bypass_helpers")
    helpers = patcher.FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS
    assert "FR13 qrow32 B1 production ran outside capture or eager" not in helpers
    assert "_fr13_fa2_qrow32_b1_bypass(" in helpers
    # The qualified point stays fail-closed.
    assert "did not capture all target tree layers" in helpers
