from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SIDECAR_SCRIPT = REPO / "scripts" / "fr13_bm8_pass_sidecar.py"
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
FA2_PATCHER = REPO / "scripts" / "fr13_patch_fa2_tree_bias.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
REAL_PASS = (
    REPO
    / "results"
    / "fr13_fixed32_bm8_b1_live_pass_20260731T180804Z"
    / "run_evidence"
    / "live_pass.json"
)
REAL_PASS_SHA256 = (
    "570caf42e3e75ff0d3717042b0dfc58b23a90041e71103f70a07f6d7563445b5"
)
QUALIFIED_SOURCE_SHA256 = (
    "3baccaa1a83907e15561b1cf807f15a41bd4764513bb43c4046b434937c3274b"
)


def _module():
    spec = importlib.util.spec_from_file_location("bm8_sidecar", SIDECAR_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patcher_module():
    spec = importlib.util.spec_from_file_location("bm8_patcher", PATCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fa2_patcher_module():
    spec = importlib.util.spec_from_file_location("bm8_fa2_patcher", FA2_PATCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _live(candidate_source_sha256: str) -> dict[str, object]:
    calls = []
    for index in range(4):
        output_sha = _sha(f"call-{index}".encode("ascii"))
        calls.append(
            {
                "call_index": index,
                "seq_len": 2048 + index,
                "bytes": 12288,
                "raw_byte_mismatches": 0,
                "stock_sha256": output_sha,
                "candidate_sha256": output_sha,
            }
        )
    identity = {
        "schema": "fr13.fixed32.dfwd_unified_bm8.identity.v1",
        "source_commit": "a" * 40,
        "production_enabled": False,
        "candidate": {
            "kernel": "kernel_unified_attention_2d",
            "stock_block_m": 16,
            "stock_block_q": 2,
            "candidate_block_m": 8,
            "candidate_block_q": 1,
            "required_calls": 4,
        },
        "files": {
            "patcher": {
                "path": "/workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py",
                "sha256": "1" * 64,
            },
            "unified_attention": {
                "path": (
                    "/usr/local/lib/python3.12/dist-packages/vllm/v1/"
                    "attention/ops/triton_unified_attention.py"
                ),
                "sha256": candidate_source_sha256,
            },
            "eagle_replay_hook": {
                "path": (
                    "/usr/local/lib/python3.12/dist-packages/vllm/v1/"
                    "spec_decode/eagle.py"
                ),
                "sha256": "2" * 64,
            },
        },
    }
    return {
        "schema": "fr13.fixed32.dfwd_unified_bm8_live_ab.v1",
        "status": "PASS",
        "suite": "SWE-Verified",
        "instance_id": "astropy__astropy-12907",
        "concurrency": 1,
        "batch_size": 1,
        "task_marker": "swe_verified:astropy__astropy-12907",
        "candidate_identity": identity,
        "calls": calls,
        "geometry": {
            "query_shape": [1, 24, 256],
            "kv_heads": 4,
            "stock_block_m": 16,
            "stock_block_q": 2,
            "candidate_block_m": 8,
            "candidate_block_q": 1,
            "valid_query_heads_per_kv": 6,
        },
        "candidate_dispatch": "launcher-private BM8 exact B1 selector",
        "candidate_dispatches": 4,
        "served_return": "stock captured drafter graph unchanged",
        "performance_measurement": False,
    }


def test_issue_and_verify_bind_real_live_pass_to_exact_triton_source(
    tmp_path: Path,
) -> None:
    module = _module()
    sidecar = tmp_path / "production-pass.json"
    issued = module.issue_sidecar(
        live_result=REAL_PASS,
        expected_live_sha256=REAL_PASS_SHA256,
        expected_candidate_source_sha256=QUALIFIED_SOURCE_SHA256,
        out=sidecar,
    )

    assert issued["status"] == "PASS"
    assert issued["instance_id"] == "astropy__astropy-12907"
    assert issued["candidate_artifact_kind"] == "triton_jit_source"
    assert issued["qualified_unified_attention_sha256"] == (
        QUALIFIED_SOURCE_SHA256
    )
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600


def test_sidecar_verification_rejects_source_drift(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "triton_unified_attention.py"
    source.write_bytes(b"qualified BM8 source")
    source_sha256 = module.sha256_file(source)
    live = tmp_path / "live.json"
    live.write_text(json.dumps(_live(source_sha256), sort_keys=True) + "\n")
    sidecar = tmp_path / "production-pass.json"
    module.issue_sidecar(
        live_result=live,
        expected_live_sha256=module.sha256_file(live),
        expected_candidate_source_sha256=source_sha256,
        out=sidecar,
    )

    verified = module.verify_sidecar(
        sidecar_path=sidecar,
        expected_sidecar_sha256=module.sha256_file(sidecar),
        candidate_source=source,
        expected_candidate_source_sha256=source_sha256,
    )
    assert verified["required_runtime"] == "fixed32 B1 FULL"
    source.write_bytes(b"different source")
    with pytest.raises(
        ValueError, match="attested BM8 candidate source SHA-256 mismatch"
    ):
        module.verify_sidecar(
            sidecar_path=sidecar,
            expected_sidecar_sha256=module.sha256_file(sidecar),
            candidate_source=source,
            expected_candidate_source_sha256=source_sha256,
        )


def test_bm8_production_selector_is_default_off_and_fail_closed() -> None:
    patcher = PATCHER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert 'FR13_DFWD_UNIFIED_BM8_PRODUCTION", "0") != "1"' in patcher
    assert "_fr13_dfwd_unified_bm8_production_begin" in patcher
    assert "_fr13_dfwd_unified_bm8_production_end" in patcher
    assert "_fr13_dfwd_unified_bm8_production_call" in patcher
    assert "_fr13_dfwd_unified_bm8_production_replay_installed" in patcher
    assert "descriptor.get(\"runtime_mode\") == \"FULL\"" in patcher
    assert "physical_rows_per_request\", -1)) == 32" in patcher
    assert "dispatches - dispatches_before != 4" in patcher
    assert 'cu_seqlens_q.dtype == torch.int32' in patcher
    assert 'seqused_k.dtype == torch.int32' in patcher
    assert 'block_table.dtype == torch.int32' in patcher
    assert 'qq_bias.dtype == torch.float32' in patcher
    assert 'alibi_slopes is None' in patcher
    assert 'os.environ.pop("FR13_DFWD_UNIFIED_BM8_INTERNAL", None)' in patcher
    assert "qualified source drifted" in patcher
    assert '"status": "CAPTURED_PENDING_REPLAY"' in patcher
    assert 'published["status"] = "ENGAGED"' in patcher
    assert 'published["graph_captures"] = 1' in patcher
    assert 'published["measured_replays"] = 1' in patcher
    assert 'published["unmeasured_replays"] = 0' in patcher
    assert "_fr13_dg_all.pop(_fr13_dg_key, None)" in patcher

    assert (
        "FR13_DFWD_UNIFIED_BM8_PRODUCTION="
        "${FR13_DFWD_UNIFIED_BM8_PRODUCTION:-0}"
    ) in launcher
    assert "BM8 live A/B and production are mutually exclusive" in launcher
    assert "fr13_bm8_pass_sidecar.py issue" in launcher
    assert "fr13_bm8_pass_sidecar.py verify" in launcher
    assert "FR13_DFWD_UNIFIED_BM8_INTERNAL_PRODUCTION_ATTESTED=1" in launcher
    assert "--dfwd-unified-bm8-production" in launcher
    assert "DFWD unified BM8 target FA2 route missing" in launcher
    assert REAL_PASS_SHA256 in launcher
    assert QUALIFIED_SOURCE_SHA256 in launcher


def test_bm8_production_call_is_composed_after_target_fa2_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree_path = tmp_path / "tree_attn.py"
    tree_path.write_text(
        '''from __future__ import annotations

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
''',
        encoding="utf-8",
    )
    patcher = _patcher_module()
    fa2_patcher = _fa2_patcher_module()
    monkeypatch.setattr(patcher, "TREE_ATTN_PATH", tree_path)
    monkeypatch.setenv("FR13_DFWD_UNIFIED_BM8_PRODUCTION", "1")

    assert patcher._patch_tree_attn_op_capture()
    helper_only = tree_path.read_text(encoding="utf-8")
    helper_only_impl = helper_only.split("class TreeAttentionImpl", 1)[1]
    assert "# FR13_DFWD_UNIFIED_BM8_PRODUCTION_CALL" in helper_only
    assert "_fr13_dfwd_unified_bm8_production_call(" not in helper_only_impl
    assert "if decode_meta := attn_metadata.decode_metadata:" in helper_only_impl

    assert fa2_patcher._patch_tree_attn(
        tree_path,
        fixed32_query_tile16_production=True,
        dfwd_unified_bm8_production=True,
    )
    composed = tree_path.read_text(encoding="utf-8")
    tree_impl = composed.split("class TreeAttentionImpl", 1)[1]
    route = (
        'if os.environ.get("FR13_FA2_TREE_BIAS", "0") == "1" '
        "and use_tree_bias:"
    )
    guarded_call = "_fr13_dfwd_unified_bm8_production_call("
    assert route in tree_impl
    assert "_fr13_fa2_qrow16_production_begin(" in tree_impl
    assert tree_impl.count(guarded_call) == 1
    assert fa2_patcher._DFWD_UNIFIED_BM8_FALLBACK not in tree_impl
    assert tree_impl.index(route) < tree_impl.index(guarded_call)
    assert tree_impl.index(guarded_call) < tree_impl.index(
        "_fr13_tree_attn_op_capture("
    )

    unchanged, did_patch = (
        fa2_patcher._patch_dfwd_unified_bm8_production_call(composed)
    )
    assert not did_patch
    assert unchanged == composed


def test_launcher_preflight_rejects_padded_head_and_refreshes_relaunch_sidecar(
    tmp_path: Path,
) -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    preflight_start = launcher.index(
        'if [[ "$FR13_DFWD_UNIFIED_BM8_PRODUCTION" == "1" ]]; then'
    )
    preflight_end = launcher.index("# FR13_EAGER_PACK", preflight_start)
    preflight = launcher[preflight_start:preflight_end]
    assert '"$FR13_DRAFT_HEAD_PAD_ROWS" == "0"' in preflight
    assert '"$FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB" == "0"' in preflight

    issue_start = launcher.index(
        'FR13_DFWD_UNIFIED_BM8_PRODUCTION_PASS_SIDECAR_HOST=',
        preflight_end,
    )
    issue_end = launcher.index(
        "FR13_DFWD_UNIFIED_BM8_PRODUCTION_PASS_SIDECAR=/logs/",
        issue_start,
    )
    issue = launcher[issue_start:issue_end]
    assert issue.index('rm -f -- "$FR13_DFWD_UNIFIED_BM8_PRODUCTION_PASS_SIDECAR_HOST"') < (
        issue.index("fr13_bm8_pass_sidecar.py issue")
    )

    module = _module()
    source_sha256 = _sha(b"qualified source")
    live = tmp_path / "live.json"
    live.write_text(json.dumps(_live(source_sha256), sort_keys=True) + "\n")
    sidecar = tmp_path / "production-pass.json"
    arguments = {
        "live_result": live,
        "expected_live_sha256": module.sha256_file(live),
        "expected_candidate_source_sha256": source_sha256,
        "out": sidecar,
    }
    first = module.issue_sidecar(**arguments)
    sidecar.unlink()
    second = module.issue_sidecar(**arguments)
    assert second == first


def test_production_capture_requires_four_dispatches_and_clears_selector(
    tmp_path: Path, monkeypatch
) -> None:
    source = _patcher_module()._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_fr13_dfwd_unified_bm8_production_begin",
            "_fr13_dfwd_unified_bm8_production_end",
            "_fr13_dfwd_unified_bm8_production_replay_installed",
        }
    ]
    assert len(selected) == 3

    candidate_source = tmp_path / "triton_unified_attention.py"
    candidate_source.write_bytes(b"qualified runtime source")
    source_sha256 = _sha(candidate_source.read_bytes())
    unified = types.SimpleNamespace(
        __file__=str(candidate_source),
        _FR13_DFWD_UNIFIED_BM8_DISPATCHES=0,
    )
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.v1.attention.ops.triton_unified_attention":
            return unified
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("FR13_DFWD_UNIFIED_BM8_PRODUCTION", "1")
    monkeypatch.setenv("FR13_DFWD_UNIFIED_BM8_LIVE_AB", "0")
    monkeypatch.setenv(
        "FR13_DFWD_UNIFIED_BM8_INTERNAL_PRODUCTION_ATTESTED", "1"
    )
    monkeypatch.setenv(
        "FR13_DFWD_UNIFIED_BM8_PRODUCTION_PASS_SIDECAR_SHA256", "a" * 64
    )
    monkeypatch.setenv(
        "FR13_DFWD_UNIFIED_BM8_QUALIFIED_SOURCE_SHA256", source_sha256
    )
    capture_json = tmp_path / "production_capture.json"
    monkeypatch.setenv(
        "FR13_DFWD_UNIFIED_BM8_PRODUCTION_CAPTURE_JSON", str(capture_json)
    )
    manifest = {
        "schema": "fr13-fixed32-forward-graph-manifest-v2",
        "mode": "tail6_fixed32",
        "batch_size": 1,
        "physical_rows_per_request": 32,
        "descriptor": {
            "runtime_mode": "FULL",
            "num_tokens": 32,
            "num_reqs": 1,
            "uniform": True,
            "has_lora": False,
            "num_active_loras": 0,
        },
    }
    context = {
        "graph_id": 22,
        "batch_size": 1,
        "mode": "tail6_fixed32",
    }
    namespace = {
        "os": os,
        "_FR13_FIXED32_MODE": "tail6_fixed32",
        "_FR13_FIXED32_CAPTURE_FROZEN": True,
        "_FR13_FIXED32_CAPTURE_MANIFESTS": {
            11: ("c" * 64, json.dumps(manifest))
        },
        "_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT": context,
        "_FR13_DFWD_UNIFIED_BM8_PRODUCTION_PENDING": {},
        "_FR13_FIXED32_DRAFTER_GRAPH_LIFECYCLE": {},
        "_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT": {
            "measured": True,
            "batch_size": 1,
            "mode": "tail6_fixed32",
        },
        "_fr13_fixed32_manifest_entry": lambda entry, _label: entry,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<bm8-prod>", "exec"),
        namespace,
    )

    namespace["_fr13_dfwd_unified_bm8_production_begin"](22, 1)
    assert "FR13_DFWD_UNIFIED_BM8_INTERNAL" not in os.environ
    context["bm8_production"]["guarded_calls"] = 4
    unified._FR13_DFWD_UNIFIED_BM8_DISPATCHES = 4
    namespace["_fr13_dfwd_unified_bm8_production_end"](22, 1, "d" * 64)

    assert "FR13_DFWD_UNIFIED_BM8_INTERNAL" not in os.environ
    assert not capture_json.exists()
    namespace["_FR13_FIXED32_DRAFTER_GRAPH_LIFECYCLE"][22] = {
        "captures": 1,
        "measured_replays": 1,
        "unmeasured_replays": 0,
    }
    namespace["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"].update(
        {"graph_id": 22, "graph_signature": "d" * 64, "graph_replays": 1}
    )
    namespace["_fr13_dfwd_unified_bm8_production_replay_installed"](
        22, 1, "d" * 64
    )
    capture = json.loads(capture_json.read_text(encoding="ascii"))
    assert capture["schema"].endswith("production_capture.v2")
    assert capture["status"] == "ENGAGED"
    assert capture["runtime_mode"] == "FULL"
    assert capture["candidate"]["calls"] == 4
    assert capture["qualified_source_sha256"] == source_sha256
    assert capture["graph_captures"] == 1
    assert capture["measured_replays"] == 1
    assert capture["unmeasured_replays"] == 0

    namespace["_fr13_dfwd_unified_bm8_production_begin"](22, 1)
    context["bm8_production"]["guarded_calls"] = 3
    unified._FR13_DFWD_UNIFIED_BM8_DISPATCHES = 7
    with pytest.raises(RuntimeError, match="did not capture four calls"):
        namespace["_fr13_dfwd_unified_bm8_production_end"](
            22, 1, "e" * 64
        )
    assert "FR13_DFWD_UNIFIED_BM8_INTERNAL" not in os.environ


def test_per_call_guard_covers_exact_geometry_and_never_leaks_selector(
    monkeypatch,
) -> None:
    patcher_tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    patch_function = next(
        node
        for node in patcher_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_tree_attn_op_capture"
    )
    helper_source = next(
        node.value.value
        for node in ast.walk(patch_function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "production_helper"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )

    class Tensor:
        def __init__(self, dtype, shape):
            self.dtype = dtype
            self.shape = shape
            self.ndim = len(shape)

    class Cuda:
        capturing = True

        @staticmethod
        def is_available():
            return True

        @classmethod
        def is_current_stream_capturing(cls):
            return cls.capturing

    torch = types.SimpleNamespace(
        bfloat16=object(), int32=object(), float32=object(), cuda=Cuda
    )
    production = {"guarded_calls": 0}
    gdn = types.SimpleNamespace(
        _FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT={
            "graph_id": 1,
            "batch_size": 1,
            "capturing": True,
            "bm8_production": production,
        }
    )
    packages = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType(
            "vllm.model_executor.layers"
        ),
        "vllm.model_executor.layers.mamba": types.ModuleType(
            "vllm.model_executor.layers.mamba"
        ),
    }
    packages["vllm.model_executor.layers.mamba"].gdn_linear_attn = gdn
    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)
    calls = []

    def unified_attention(**kwargs):
        assert os.environ["FR13_DFWD_UNIFIED_BM8_INTERNAL"] == "1"
        calls.append(kwargs)
        return "candidate"

    namespace = {"os": os, "torch": torch, "unified_attention": unified_attention}
    exec(compile(helper_source, "<bm8-call-guard>", "exec"), namespace)
    guarded = namespace["_fr13_dfwd_unified_bm8_production_call"]
    monkeypatch.setenv("FR13_DFWD_UNIFIED_BM8_PRODUCTION", "1")
    monkeypatch.setenv("FR13_DFWD_UNIFIED_BM8_LIVE_AB", "0")
    monkeypatch.setenv(
        "FR13_DFWD_UNIFIED_BM8_INTERNAL_PRODUCTION_ATTESTED", "1"
    )
    kwargs = {
        "layer": types.SimpleNamespace(layer_name="mtp.layers.0.self_attn.attn"),
        "q": Tensor(torch.bfloat16, (1, 24, 256)),
        "k": Tensor(torch.bfloat16, (8, 1024, 4, 256)),
        "v": Tensor(torch.bfloat16, (8, 1024, 4, 256)),
        "out": Tensor(torch.bfloat16, (1, 24, 256)),
        "cu_seqlens_q": Tensor(torch.int32, (2,)),
        "max_seqlen_q": 1,
        "seqused_k": Tensor(torch.int32, (1,)),
        "max_seqlen_k": 22872,
        "softmax_scale": 0.0625,
        "causal": True,
        "window_size": (-1, -1),
        "block_table": Tensor(torch.int32, (1, 8)),
        "softcap": 0.0,
        "q_descale": None,
        "k_descale": object(),
        "v_descale": object(),
        "qq_bias": Tensor(torch.float32, (1, 1)),
    }

    assert guarded(**kwargs) == "candidate"
    assert len(calls) == 1
    assert production["guarded_calls"] == 1
    assert "FR13_DFWD_UNIFIED_BM8_INTERNAL" not in os.environ

    wrong = dict(kwargs)
    wrong["cu_seqlens_q"] = Tensor(torch.float32, (2,))
    with pytest.raises(RuntimeError, match="production geometry drifted"):
        guarded(**wrong)
    assert "FR13_DFWD_UNIFIED_BM8_INTERNAL" not in os.environ

    def failing_attention(**_kwargs):
        assert os.environ["FR13_DFWD_UNIFIED_BM8_INTERNAL"] == "1"
        raise RuntimeError("candidate failure")

    namespace["unified_attention"] = failing_attention
    with pytest.raises(RuntimeError, match="candidate failure"):
        guarded(**kwargs)
    assert production["guarded_calls"] == 1
    assert "FR13_DFWD_UNIFIED_BM8_INTERNAL" not in os.environ

    gdn._FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT = None
    Cuda.capturing = False
    namespace["unified_attention"] = lambda **_kwargs: "stock"
    assert guarded(**kwargs) == "stock"
    assert "FR13_DFWD_UNIFIED_BM8_INTERNAL" not in os.environ
