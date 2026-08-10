"""The B4 GQA-pair production/timing path: credential, call site, env wiring."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts/fr13_patch_fa2_tree_bias.py"
SIDECAR = REPO / "scripts/fr13_qrow32_b4_pass_sidecar.py"
LAUNCHER = REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"
CONTRACT = REPO / "scripts/fr13_fixed32_contract.py"
TIMING_RUNNER = REPO / "scripts/fr13_run_b4_fa2_qrow32_gqa_pair_timing.sh"
BYTE_GATE_RUNNER = REPO / "scripts/fr13_run_b4_fa2_qrow32_gqa_pair_live_gate.sh"

# The ten caller-supplied names this path introduces. FR13_FA2_QROW32_SO_SHA256
# and friends are deliberately reused from the byte gate: the binary identity
# is the same pinned candidate, so re-declaring it under a B4 alias would
# create two names that could disagree.
NEW_ENV_NAMES = (
    "FR13_FA2_QROW32_B4_TIMING_ARM",
    "FR13_FA2_QROW32_B4_PRODUCTION_ARM",
    "FR13_FA2_QROW32_B4_PRODUCTION_PASS_SIDECAR",
    "FR13_FA2_QROW32_B4_PRODUCTION_PASS_SIDECAR_SHA256",
    "FR13_FA2_QROW32_B4_PRODUCTION_ENGAGEMENT_JSON",
    "FR13_FA2_QROW32_B4_DUAL_GATE_JSON",
    "FR13_FA2_QROW32_B4_DUAL_GATE_SHA256",
    "FR13_FA2_QROW32_B4_EXACT4_TASK_IDS",
    "FR13_FA2_QROW32_B4_EXACT4_SUBSET_SHA256",
    "FR13_FA2_QROW32_B4_PATCH_SOURCE_SHA256",
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _dual_gate_payload(module, source_commit: str) -> dict[str, object]:
    return {
        "schema": module.DUAL_GATE_SCHEMA,
        "status": "PASS",
        "candidate_arm": module.ARM,
        "candidate_so_sha256": module.CANDIDATE_SHA256,
        "candidate_so_size": module.CANDIDATE_SIZE,
        "fa2_head": module.FA2_HEAD,
        "fa2_source_closure_sha256": module.SOURCE_CLOSURE_SHA256,
        "selector_sentinel": module.SELECTOR_SENTINEL,
        "source_commit": source_commit,
        "task_ids": list(module.EXACT4_TASK_IDS),
        "subset_sha256": module.EXACT4_SUBSET_SHA256,
        "qualified_topologies": list(module.QUALIFIED_TOPOLOGIES),
        "tail23_verification_sha256": "a" * 64,
        "hydra27_verification_sha256": "b" * 64,
        "layer_count_per_topology": 16,
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "fallback_allowed": False,
        "performance_measurement": False,
        "timing_eligible": False,
        "production_eligible": False,
    }


@pytest.fixture()
def sidecar_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _module(SIDECAR, "qrow32_b4_sidecar")
    candidate_bytes = b"pinned gqa-pair candidate binary"
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(candidate_bytes)
    monkeypatch.setattr(module, "CANDIDATE_SIZE", len(candidate_bytes))
    monkeypatch.setattr(
        module, "CANDIDATE_SHA256", hashlib.sha256(candidate_bytes).hexdigest()
    )
    source_commit = _head()
    payload = _dual_gate_payload(module, source_commit)
    return module, candidate, source_commit, payload, tmp_path


def _write_gate(tmp_path: Path, payload: dict[str, object]) -> tuple[Path, str]:
    raw = json.dumps(payload).encode("ascii")
    path = tmp_path / "dual_gate_verification.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _issue(module, candidate, source_commit, gate, gate_sha, out):
    return module.issue_sidecar(
        dual_gate=gate,
        expected_dual_gate_sha256=gate_sha,
        candidate_so=candidate,
        expected_candidate_sha256=module.CANDIDATE_SHA256,
        arm=module.ARM,
        patch_source=REPO / "scripts/fr13_patch_fa2_tree_bias.py",
        expected_source_commit=source_commit,
        out=out,
    )


def test_sidecar_binds_a_dual_gate_pass_to_the_production_credential(
    sidecar_fixture,
) -> None:
    module, candidate, source_commit, payload, tmp_path = sidecar_fixture
    gate, gate_sha = _write_gate(tmp_path, payload)
    out = tmp_path / "production_pass.json"
    issued = _issue(module, candidate, source_commit, gate, gate_sha, out)

    assert issued["status"] == "PASS"
    assert issued["arm"] == "gqa_pair"
    assert issued["selector_sentinel"] == 0x20014
    assert issued["dual_gate_sha256"] == gate_sha
    assert issued["source_commit"] == source_commit
    assert issued["qualified_topologies"] == ["Tail23", "Hydra27"]
    # The gate's own eligibility flags are carried verbatim, not laundered.
    assert issued["dual_gate_timing_eligible"] is False
    assert issued["dual_gate_production_eligible"] is False

    verified = module.verify_sidecar(
        sidecar_path=out,
        expected_sidecar_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        candidate_so=candidate,
        expected_candidate_sha256=module.CANDIDATE_SHA256,
        arm=module.ARM,
        patch_source=REPO / "scripts/fr13_patch_fa2_tree_bias.py",
        expected_source_commit=source_commit,
    )
    assert verified["canonical_sha256"] == issued["canonical_sha256"]

    # Issuing is single-shot: an existing credential is never silently replaced.
    with pytest.raises(module.SidecarError, match="refusing to replace"):
        _issue(module, candidate, source_commit, gate, gate_sha, out)


def test_sidecar_rejects_a_wrong_sha_candidate_binary(sidecar_fixture) -> None:
    module, candidate, source_commit, payload, tmp_path = sidecar_fixture
    gate, gate_sha = _write_gate(tmp_path, payload)

    # Declared digest is not the pinned candidate at all.
    with pytest.raises(module.SidecarError, match="not the pinned GQA-pair binary"):
        module.issue_sidecar(
            dual_gate=gate,
            expected_dual_gate_sha256=gate_sha,
            candidate_so=candidate,
            expected_candidate_sha256="0" * 64,
            arm=module.ARM,
            patch_source=REPO / "scripts/fr13_patch_fa2_tree_bias.py",
            expected_source_commit=source_commit,
            out=tmp_path / "a.json",
        )

    # Declared digest is right, but the file on disk is a different binary.
    candidate.write_bytes(b"a different binary entirely")
    with pytest.raises(module.SidecarError, match="candidate SO identity mismatch"):
        _issue(module, candidate, source_commit, gate, gate_sha, tmp_path / "b.json")
    assert not (tmp_path / "a.json").exists()
    assert not (tmp_path / "b.json").exists()


def test_sidecar_rejects_a_gate_from_another_commit(sidecar_fixture) -> None:
    module, candidate, source_commit, payload, tmp_path = sidecar_fixture
    # The reason the runroot is a parameter: a gate produced before the
    # production plumbing existed carries the older commit and is refused.
    payload["source_commit"] = "0" * 40
    gate, gate_sha = _write_gate(tmp_path, payload)
    with pytest.raises(module.SidecarError, match="production plumbing commit"):
        _issue(module, candidate, source_commit, gate, gate_sha, tmp_path / "c.json")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "FAIL", "not a PASS"),
        ("candidate_arm", "qrow32", "candidate identity drifted"),
        ("selector_sentinel", 0x20013, "candidate identity drifted"),
        ("output_raw_byte_mismatches", 1, "raw-byte equality drifted"),
        ("lse_raw_byte_mismatches", 1, "raw-byte equality drifted"),
        ("qualified_topologies", ["Hydra27"], "exact4 scope drifted"),
        ("subset_sha256", "0" * 64, "exact4 scope drifted"),
        ("production_eligible", True, "eligibility fields were rewritten"),
        ("timing_eligible", True, "eligibility fields were rewritten"),
    ],
)
def test_sidecar_is_fail_closed_on_a_doctored_gate(
    sidecar_fixture, field: str, value: object, message: str
) -> None:
    module, candidate, source_commit, payload, tmp_path = sidecar_fixture
    payload[field] = value
    gate, gate_sha = _write_gate(tmp_path, payload)
    with pytest.raises(module.SidecarError, match=message):
        _issue(module, candidate, source_commit, gate, gate_sha, tmp_path / "d.json")


def test_sidecar_rejects_a_gate_whose_bytes_do_not_match_the_declaration(
    sidecar_fixture,
) -> None:
    module, candidate, source_commit, payload, tmp_path = sidecar_fixture
    gate, _ = _write_gate(tmp_path, payload)
    with pytest.raises(module.SidecarError, match="raw SHA-256 mismatch"):
        _issue(module, candidate, source_commit, gate, "f" * 64, tmp_path / "e.json")


def test_sidecar_verify_rejects_a_tampered_credential(sidecar_fixture) -> None:
    module, candidate, source_commit, payload, tmp_path = sidecar_fixture
    gate, gate_sha = _write_gate(tmp_path, payload)
    out = tmp_path / "production_pass.json"
    _issue(module, candidate, source_commit, gate, gate_sha, out)

    tampered = json.loads(out.read_text(encoding="ascii"))
    tampered["production_scope"] = "everything, always"
    raw = module.canonical_bytes(tampered) + b"\n"
    out.chmod(0o600)
    out.write_bytes(raw)
    with pytest.raises(module.SidecarError, match="canonical digest mismatch"):
        module.verify_sidecar(
            sidecar_path=out,
            expected_sidecar_sha256=hashlib.sha256(raw).hexdigest(),
            candidate_so=candidate,
            expected_candidate_sha256=module.CANDIDATE_SHA256,
            arm=module.ARM,
            patch_source=REPO / "scripts/fr13_patch_fa2_tree_bias.py",
            expected_source_commit=source_commit,
        )


# --------------------------------------------------------------------------
# Call-site mutual exclusion
# --------------------------------------------------------------------------

TREE_ATTN_STUB = '''from __future__ import annotations

import ast
import os
from dataclasses import dataclass

from vllm.v1.attention.ops.triton_unified_attention import unified_attention

logger = init_logger(__name__)

# FR13_TREE_ATTN_OP_CAPTURE
_prefill_native_installed = (
    os.environ.get("FR13_FA2_PREFILL_NATIVE", "0") == "1"
)


def _get_depth_counts():
    return ()


class TreeAttentionImpl:
    def forward(
        self,
        layer,
        query,
        key,
        value,
        key_cache,
        value_cache,
        attn_metadata,
        output,
        num_decode_tokens,
        descale_shape,
    ):
        if decode_meta := attn_metadata.decode_metadata:
            unified_attention(
                q=query[:num_decode_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_decode_tokens],
                cu_seqlens_q=decode_meta.query_start_loc,
                max_seqlen_q=decode_meta.max_query_len,
                seqused_k=decode_meta.seq_lens,
                max_seqlen_k=decode_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                qq_bias=decode_meta.tree_attn_bias,
                window_size=self.sliding_window,
                block_table=decode_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )
'''

# Every one of these rewrites the same `if not _fr13_reordered:` decode call.
CONTENDING_SELECTORS = (
    "fixed32_query_tile16_live_ab",
    "fixed32_query_tile32_live_ab",
    "fixed32_query_tile32_b1_live_ab",
    "fixed32_query_tile32_b1_production",
    "fixed32_query_tile32_b4_production",
    "fixed32_query_tile16_production",
)


def _stub(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text(TREE_ATTN_STUB, encoding="utf-8")
    return path


def test_b4_production_wrapper_feeds_the_selection_bias_to_the_served_call(
    tmp_path: Path,
) -> None:
    patcher = _module(PATCHER, "qrow32_b4_patcher")
    path = _stub(tmp_path, "tree_attn.py")
    assert patcher._patch_tree_attn(path, fixed32_query_tile32_b4_production=True)
    text = path.read_text(encoding="utf-8")
    impl = text.split("class TreeAttentionImpl", 1)[1]

    assert "_fr13_fa2_qrow32_b4_production_begin(" in impl
    assert impl.count("_fr13_fa2_qrow32_b4_production_end(") == 2
    # The SERVED call -- the one writing output[:num_decode_tokens] -- is the
    # one that receives the tagged bias. That is how the sentinel reaches the
    # C++ dispatch for real traffic instead of an offline replay.
    served = impl[impl.index("out=output[:num_decode_tokens],") :]
    assert '_fr13_qrow32_b4_selection["tree_bias"]' in served
    assert '_fr13_qrow32_b4_selection["num_splits"]' in served
    ast.parse(text)


@pytest.mark.parametrize("first", CONTENDING_SELECTORS)
@pytest.mark.parametrize("second", CONTENDING_SELECTORS)
def test_contending_decode_selectors_are_pairwise_mutually_exclusive(
    tmp_path: Path, first: str, second: str
) -> None:
    patcher = _module(PATCHER, "qrow32_b4_patcher_mutex")
    path = _stub(tmp_path, f"tree_attn_{first}_{second}.py")
    if first == second:
        # A single selector must still install cleanly.
        assert patcher._patch_tree_attn(path, **{first: True})
        return
    with pytest.raises(ValueError, match="mutually exclusive"):
        patcher._patch_tree_attn(path, **{first: True, second: True})
    # Nothing was written: the refusal happens before any rewrite.
    assert path.read_text(encoding="utf-8") == TREE_ATTN_STUB


def test_only_one_selector_survives_the_argparse_front_door() -> None:
    text = PATCHER.read_text(encoding="utf-8")
    front_door = text[text.index("private_selectors = sum(") :]
    for name in CONTENDING_SELECTORS:
        assert f"args.{name}," in front_door.split(")\n    )", 1)[0]
    assert "qrow16/qrow32 private selectors are mutually exclusive" in front_door


def test_patcher_installs_the_b4_capture_end_hook_and_source_requirement() -> None:
    text = PATCHER.read_text(encoding="utf-8")
    assert "_patch_cuda_graph_qrow32_b4_production" in text
    assert "# FR13_FA2_QROW32_B4_PRODUCTION_CAPTURE_END" in text
    assert "_fr13_fa2_qrow32_b4_production_capture_end" in text
    assert "--fixed32-query-tile32-b4-production" in text
    assert "a combined qrow32 B4 source/production patch requires " in text
    assert '"--fixed32-query-gqa-pair32"' in text


def test_b4_helper_block_serves_the_gqa_pair_sentinel_fail_closed() -> None:
    patcher = _module(PATCHER, "qrow32_b4_patcher_helpers")
    helpers = patcher.FIXED32_QUERY_TILE32_B4_PRODUCTION_HELPERS
    ast.parse(helpers)
    assert "_FR13_FA2_QROW32_B4_BATCH_STRIDE_SENTINEL = 131092" in helpers
    assert (
        "af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb"
        in helpers
    )
    assert "299813360" in helpers
    assert "FR13 qrow32 B4 production has no launcher attestation" in helpers
    assert "FR13 qrow32 B4 pinned identity drifted" in helpers
    assert "FR13 qrow32 B4 production exact4 identity drifted" in helpers
    assert "FR13 qrow32 B4 production silently fell back" in helpers
    assert "FR13 qrow32 B4 production geometry drifted" in helpers
    assert 'int(descriptor.get("num_reqs", -1)) != 4' in helpers
    assert "torch.empty_strided(" in helpers
    assert '"candidate_served": True' in helpers


# --------------------------------------------------------------------------
# Environment threading
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", NEW_ENV_NAMES)
def test_new_envs_are_guarded_defaulted_and_passed_into_the_container(
    name: str,
) -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    guard_block = launcher[
        launcher.index("_FR13_M32_GUARD_NAMES=(") : launcher.index("\n)\ndeclare -A")
    ]
    # In the guard list, so .lumo.local.env cannot rewrite it behind the run.
    assert f"\n  {name}\n" in guard_block
    # Defaulted exactly once, so an unset caller env is not an unbound variable.
    assert launcher.count(f"\n{name}=${{{name}:-") == 1
    # Threaded into the container.
    assert f'-e {name}="$' in launcher


def test_launcher_pins_the_b4_candidate_identity_and_head_binding() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    block = launcher[
        launcher.index("_FR13_FA2_QROW32_B4_CANDIDATE_MODE=0") :
        launcher.index("_FR13_FA2_QROW32_B1_CANDIDATE_MODE=0")
    ]
    assert (
        '"$FR13_FA2_QROW32_SO_SHA256" == '
        '"af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb"'
    ) in block
    assert '"$FR13_FA2_QROW32_SO_SIZE" == "299813360"' in block
    assert (
        '"$FR13_FA2_QROW32_SOURCE_CLOSURE_SHA256" == '
        '"9c3f9e751da7b783e9d07d8e40d5bc2234b99e719a1048668bd6c82244ed2d81"'
    ) in block
    # The candidate .so on disk, not merely the declaration.
    assert '"$(sha256sum "$FORKED_FA2_SO" | cut -d\' \' -f1)"' in block
    assert '"$FR13_FA2_QROW32_SOURCE_COMMIT" == "$(git rev-parse HEAD)"' in block
    assert (
        '"$FR13_FA2_QROW32_B4_PATCH_SOURCE_SHA256" == '
        '"$(sha256sum scripts/fr13_patch_fa2_tree_bias.py | cut -d\' \' -f1)"'
    ) in block
    assert "astropy__astropy-12907,astropy__astropy-13033" in block
    assert '"$MAX_NUM_SEQS" == "4"' in block
    assert '"${SWE_CONCURRENCY:-}" == "4"' in block


def test_launcher_issues_verifies_and_privately_scopes_the_credential() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "fr13_qrow32_b4_pass_sidecar.py issue" in launcher
    assert "fr13_qrow32_b4_pass_sidecar.py verify" in launcher
    assert "FR13_FA2_QROW32_B4_INTERNAL_ATTESTED=1" in launcher
    assert "FR13 qrow32 B4 internal attestation is launcher-private" in launcher
    assert "FR13 qrow32 production sidecar credentials are launcher-private" in launcher
    assert (
        "FR13 qrow32 B4 and existing FA2 private selectors are mutually exclusive"
        in launcher
    )
    assert "FR13_FA2_QROW32_B4_PRODUCTION_ARM must be empty or gqa_pair" in launcher
    assert (
        "FR13_FA2_QROW32_B4_TIMING_ARM must be empty, stock_dispatch, or gqa_pair"
        in launcher
    )
    assert "--fixed32-query-tile32-b4-production" in launcher


def test_env_contract_pins_the_b4_timing_binary_and_arm_agreement() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    contract = _module(CONTRACT, "fr13_b4_contract")
    identity = contract._expected_runtime_fa2_identity
    pinned = (
        contract.QROW32_B4_GQA_PAIR_FA2_SIZE,
        contract.QROW32_B4_GQA_PAIR_FA2_SHA256,
    )
    # Both arms declare the SAME binary; that is the point of the pair.
    assert identity(
        {
            "FR13_FA2_QROW32_B4_TIMING_ARM": "stock_dispatch",
            "FR13_FA2_QROW32_SO_SHA256": pinned[1],
        }
    ) == pinned
    assert identity(
        {
            "FR13_FA2_QROW32_B4_TIMING_ARM": "gqa_pair",
            "FR13_FA2_QROW32_B4_PRODUCTION_ARM": "gqa_pair",
            "FR13_FA2_QROW32_SO_SHA256": pinned[1],
        }
    ) == pinned

    for env, message in (
        (
            {
                "FR13_FA2_QROW32_B4_TIMING_ARM": "gqa_pair",
                "FR13_FA2_QROW32_SO_SHA256": pinned[1],
            },
            "must agree on the served kernel",
        ),
        (
            {"FR13_FA2_QROW32_B4_PRODUCTION_ARM": "gqa_pair"},
            "must agree on the served kernel",
        ),
        ({"FR13_FA2_QROW32_B4_TIMING_ARM": "bogus"}, "must be empty, stock_dispatch"),
        (
            {"FR13_FA2_QROW32_B4_PRODUCTION_ARM": "qrow32"},
            "must be empty or gqa_pair",
        ),
        (
            {
                "FR13_FA2_QROW32_B4_TIMING_ARM": "stock_dispatch",
                "FR13_FA2_QROW32_SO_SHA256": "0" * 64,
            },
            "not the pinned",
        ),
        (
            {
                "FR13_FA2_QROW32_B4_TIMING_ARM": "stock_dispatch",
                "FR13_FA2_QROW32_SO_SHA256": pinned[1],
                "FR13_FA2_QROW32_LIVE_PAGED_AB": "1",
                "FR13_FA2_QROW32_LIVE_PAGED_AB_ARM": "gqa_pair",
            },
            "mutually exclusive",
        ),
        (
            {
                "FR13_FA2_QROW32_B4_TIMING_ARM": "stock_dispatch",
                "FR13_FA2_QROW32_SO_SHA256": pinned[1],
                "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "nosplit",
            },
            "mutually exclusive",
        ),
    ):
        with pytest.raises(contract.ContractError, match=message):
            identity(env)


# --------------------------------------------------------------------------
# Timing runner
# --------------------------------------------------------------------------


def test_timing_runner_is_default_off_and_a_single_variable_pair() -> None:
    text = TIMING_RUNNER.read_text(encoding="utf-8")
    assert "FR13_RUN_B4_QROW32_GQA_PAIR_TIMING:-0" in text
    assert 'run_arm "$STOCK_ARM" stock_dispatch ""' in text
    assert 'run_arm "$CANDIDATE_ARM" gqa_pair gqa_pair' in text
    # One FORKED_FA2_SO for both arms: the binary is not the variable.
    assert text.count('FORKED_FA2_SO="$QROW32_GQA_PAIR_FA2_SO"') == 1
    assert "config/fr13_fixed32/subset_b4_four.json" in text
    assert (
        "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5" in text
    )
    assert "export BSIZE=4" in text and "export CONC=4" in text
    assert "DRAFT_VOCAB_K=65536" in text
    assert 'FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K"' in text
    assert "ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in text
    assert "promotion_verdict" in text
    assert "fr13_b4_timing_math" in text
    # Work-census gated reduction, on both arms.
    assert "--work-census" in text
    assert 'work_census_gate") or {}).get("status") != "pass"' in text
    # The stock arm must not emit an engagement: a sentinel leak invalidates
    # the pair rather than merely biasing it.
    assert "emitted a GQA-pair engagement on the stock-dispatch arm" in text
    assert "formal_floor_acceptance_eligible=0" in text
    assert '"production_default_enabled": False' in text


def test_timing_runner_revalidates_the_gate_binding_before_launching() -> None:
    text = TIMING_RUNNER.read_text(encoding="utf-8")
    assert "SIDECAR=scripts/fr13_qrow32_b4_pass_sidecar.py" in text
    assert '"$SIDECAR" validate' in text
    assert '"$SIDECAR" verify' in text
    assert '"$GATE" validate-candidate' in text
    assert '--expected-source-commit "$SOURCE_COMMIT"' in text
    # No runroot of any previous gate is baked in.
    assert "fr13_fa2_qrow32_gqa_pair" in text
    assert not re.search(r"output/fr13_[a-z0-9_]*_20\d{6}T\d{6}Z", text)


def test_timing_runner_inline_summary_argv_matches_the_shell_invocation() -> None:
    text = TIMING_RUNNER.read_text(encoding="utf-8")
    marker = '"$PYTHON_BIN" - \\'
    invocation = text[text.index(marker) + len(marker) : text.index("<<'PY'")]
    # Count the arguments the shell actually passes to the inline reducer.
    passed = len(re.findall(r'"\$[^"]*"', invocation))
    body = text.split("<<'PY'\n", 1)[1].rsplit("\nPY\n", 1)[0]
    ast.parse(body)
    highest = max(
        int(match)
        for match in re.findall(r"sys\.argv\[(?:\d+:)?(\d+)\]", body)
    )
    # argv[0] is the script itself, so the reducer's highest slice bound must
    # be exactly one past the number of passed arguments.
    assert highest == passed + 1, (passed, highest)
