from __future__ import annotations

import hashlib
import importlib.util
import json
import os
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
LIVE_GATE = REPO / "scripts/fr13_run_b1_k64_qrow32_split2_live_gate.sh"
TIMING_RUNNER = REPO / "scripts/fr13_run_b1_k64_qrow32_b1_sfwd_stack_timing.sh"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selector_namespace(monkeypatch: pytest.MonkeyPatch, *, profile_scope):
    namespace = {"os": os, "torch": torch}
    patcher = _module(PATCHER, "qrow32_b1_fa2_patcher")
    exec(patcher.FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS, namespace)

    gdn = types.ModuleType("gdn_linear_attn")
    gdn._FR13_FIXED32_PROFILE_CAPTURE_SCOPE = profile_scope
    gdn._FR13_FIXED32_PROFILE_MEMORY_SCOPE = profile_scope is not None
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

    monkeypatch.setenv("FR13_FA2_QROW32_B1_LIVE_AB_ARM", "split2")
    monkeypatch.setenv("FR13_DRAFT_VOCAB_ROOT", "1")
    monkeypatch.setenv("FR13_DRAFT_VOCAB_K", "65536")
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_SO_SHA256",
        namespace["_FR13_FA2_QROW32_B1_CANDIDATE_SHA256"],
    )
    monkeypatch.setenv("FR13_FA2_QROW32_B1_SO_SIZE", "300154616")
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_FA2_HEAD",
        namespace["_FR13_FA2_QROW32_B1_FA2_HEAD"],
    )
    monkeypatch.setenv(
        "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256",
        namespace["_FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256"],
    )
    monkeypatch.setenv("FR13_FA2_QROW32_B1_SOURCE_COMMIT", "1" * 40)
    monkeypatch.setenv("FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256", "2" * 64)
    monkeypatch.setenv("FR13_FA2_QROW32_B1_PRODUCTION_ARM", "nosplit")
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
    interleaved_kv = torch.empty(
        (1, 2, 1024, 4, 256), dtype=torch.bfloat16
    )
    return {
        "flash_fn": object(),
        "query": fused_qkv[:, :24, :],
        "key_cache": interleaved_kv[:, 0],
        "value_cache": interleaved_kv[:, 1],
        "cu_seqlens_q": torch.tensor([0, 32], dtype=torch.int32),
        "max_seqlen_q": 32,
        "seqused_k": torch.tensor([32], dtype=torch.int32),
        "max_seqlen_k": 32,
        "softmax_scale": 1.0,
        "causal": False,
        "window_size": [-1, -1],
        "block_table": torch.tensor([[0]], dtype=torch.int32),
        "softcap": 0.0,
        "num_splits": 1,
        "tree_bias": torch.zeros((32, 32), dtype=torch.float32),
    }


def _profile_descriptor() -> dict[str, object]:
    return {
        "runtime_mode": "FULL",
        "num_tokens": 32,
        "num_reqs": 1,
        "uniform": True,
        "has_lora": False,
        "num_active_loras": 0,
    }


def _comparison(seed: str, dtype: str, shape: list[int]) -> dict[str, object]:
    digest = hashlib.sha256(seed.encode("ascii")).hexdigest()
    return {
        "dtype": dtype,
        "shape": shape,
        "bytes": 4096,
        "raw_byte_mismatches": 0,
        "reference_sha256": digest,
        "candidate_sha256": digest,
    }


def _live_payload(
    module,
    candidate_sha256: str,
    source_commit: str,
    patch_sha256: str,
    *,
    arm: str = "nosplit",
) -> dict[str, object]:
    arm_contract = module.LIVE_ARMS[arm]
    layers = []
    for index in range(3, 64, 4):
        name = f"language_model.model.layers.{index}.self_attn.attn"
        layers.append(
            {
                "layer_name": name,
                "output": _comparison(name + "-o", "torch.bfloat16", [32, 24, 256]),
                "lse": _comparison(name + "-l", "torch.float32", [1, 24, 32]),
            }
        )
    return {
        "schema": module.LIVE_SCHEMA,
        "status": "PASS",
        "suite": "SWE-Verified",
        "instance_id": module.EXACT4_TASK_IDS[0],
        "concurrency": 1,
        "batch_size": 1,
        "physical_rows": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "runtime_mode": "FULL",
        "candidate_so_sha256": candidate_sha256,
        "candidate_so_size": module.CANDIDATE_SIZE,
        "arm": arm,
        "selector_sentinel": arm_contract["selector_sentinel"],
        "candidate_num_splits": arm_contract["num_splits"],
        "split_scratch_allocation": arm_contract["split_scratch_allocation"],
        "reference_selector_sentinel": 1179791667,
        "reference_dispatch": "qrow16 incumbent exact geometry; no fallback",
        "candidate_dispatch": arm_contract["candidate_dispatch"],
        "fa2_head": module.FA2_HEAD,
        "fa2_source_closure_sha256": module.SOURCE_CLOSURE_SHA256,
        "source_commit": source_commit,
        "patch_source_sha256": patch_sha256,
        "layer_count": 16,
        "layers": layers,
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "fallback_allowed": False,
        "served_return": "qrow16 captured graph output unchanged",
        "performance_measurement": False,
    }


def test_selectors_admit_live_nosplit_but_keep_qrow16_served() -> None:
    text = PATCHER.read_text()
    helpers = text.split("FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS", 1)[1]
    production = helpers.split("def _fr13_fa2_qrow32_b1_production_begin", 1)[1]

    assert '"nosplit": {' in helpers
    assert '"sentinel": 1179791668' in helpers
    assert '"num_splits": 0' in helpers
    assert '"split2": {' in helpers
    assert '"sentinel": 1179791669' in helpers
    assert '"num_splits": 2' in helpers
    assert 'os.environ.get(env_name, "")' in helpers
    assert 'tree_bias = _fr13_fa2_qrow32_b1_live_register(' in text
    assert "_FR13_FA2_QROW32_B1_QROW16_REFERENCE_SENTINEL = 1179791667" in helpers
    assert '"reference_sha256"' in helpers
    assert '"served_return": "qrow16 captured graph output unchanged"' in helpers
    assert "torch.cuda.synchronize()" not in production
    assert '"candidate_served": True, "fallback_allowed": False' in helpers
    assert "FR13 qrow32 B1 production silently fell back" in helpers


def test_nosplit_live_call_uses_hidden_sentinel_and_zero_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, _ = _selector_namespace(monkeypatch, profile_scope=None)
    monkeypatch.setenv("FR13_FA2_QROW32_B1_LIVE_AB_ARM", "nosplit")
    assert namespace["_fr13_fa2_qrow32_b1_arm"](
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM"
    ) == "nosplit"

    bundle = _b1_geometry()
    bundle["flash_fn"] = lambda **kwargs: kwargs
    called = namespace["_fr13_fa2_qrow32_b1_live_call"](
        bundle, object(), arm="nosplit"
    )
    assert called["num_splits"] == 0
    assert tuple(called["tree_bias"].stride()) == (1179791668, 32, 1)

    monkeypatch.setenv("FR13_FA2_QROW32_B1_PRODUCTION_ARM", "nosplit")
    assert namespace["_fr13_fa2_qrow32_b1_arm"](
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM"
    ) == "nosplit"
    monkeypatch.setenv("FR13_FA2_QROW32_B1_PRODUCTION_ARM", "split2")
    with pytest.raises(RuntimeError, match="must be empty or nosplit"):
        namespace["_fr13_fa2_qrow32_b1_arm"](
            "FR13_FA2_QROW32_B1_PRODUCTION_ARM"
        )


def test_split2_is_not_raw_byte_eligible_against_no_split_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, _ = _selector_namespace(monkeypatch, profile_scope=None)
    require_same_reduction = namespace[
        "_fr13_fa2_qrow32_b1_require_same_reduction"
    ]

    for reference_num_splits in (0, 1):
        with pytest.raises(
            RuntimeError,
            match=(
                "raw-byte qualification requires identical reduction topology: "
                "reference_partitions=1 candidate_partitions=2"
            ),
        ):
            require_same_reduction("split2", reference_num_splits)

    assert require_same_reduction("split2", 2) is None


def test_live_and_production_paths_enforce_same_reduction_before_selection() -> None:
    text = PATCHER.read_text()
    helpers = text.split("FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS", 1)[1]
    live = helpers.split("def _fr13_fa2_qrow32_b1_live_replay", 1)[1].split(
        "def _fr13_fa2_qrow32_b1_production_begin", 1
    )[0]
    production = helpers.split(
        "def _fr13_fa2_qrow32_b1_production_begin", 1
    )[1].split("def _fr13_fa2_qrow32_b1_production_end", 1)[0]

    assert "arm, bundle[\"num_splits\"]" in live
    assert "arm, num_splits" in production
    assert live.index("arm, bundle[\"num_splits\"]") < live.index(
        "_fr13_fa2_qrow32_b1_live_call("
    )
    assert production.index("arm, num_splits") < production.index(
        "_fr13_fa2_qrow32_b1_candidate_tree_bias("
    )


def test_fa2_interface_allows_only_exact_private_b1_split2_tag() -> None:
    patcher = _module(PATCHER, "qrow32_b1_interface_guard")
    namespace = {"torch": torch}
    exec(patcher.FR13_FA2_QROW32_B1_SPLIT2_INTERFACE_HELPER, namespace)
    allowed = namespace["_fr13_fa2_qrow32_b1_split2_interface_allowed"]

    def tagged_bias(**overrides):
        values = {
            "is_cuda": True,
            "dtype": torch.float32,
            "shape": (1, 32, 32),
            "stride": lambda: (1179791669, 32, 1),
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    assert allowed(2, tagged_bias())
    assert not allowed(1, tagged_bias())
    assert not allowed(3, tagged_bias())
    assert not allowed(2, None)
    assert not allowed(2, tagged_bias(is_cuda=False))
    assert not allowed(2, tagged_bias(dtype=torch.bfloat16))
    assert not allowed(2, tagged_bias(shape=(32, 32)))
    assert not allowed(
        2, tagged_bias(stride=lambda: (1179791667, 32, 1))
    )
    assert not allowed(
        2, tagged_bias(stride=lambda: (1179791669, 33, 1))
    )


def test_fa2_interface_patcher_replaces_generic_split_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patcher = _module(PATCHER, "qrow32_b1_interface_patcher")
    interface = tmp_path / "flash_attn_interface.py"
    interface.write_text(
        '''import torch

DEFAULT_FA_VERSION = 2

def flash_attn_varlen_func(
    q,
    k,
    v,
    out=None,
    dropout_p=0.0,
    return_softmax_lse=False,
    num_splits=0,
    fa_version=2,
    s_aux=None,
    cp_world_size=1,
):
    if fa_version == 2:
        if num_splits > 1:
            raise NotImplementedError("FA2 does not support num_splits > 1")
        out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(
            q,
            k,
            v,
            out,
            return_softmax_lse and dropout_p > 0,
            num_splits,
            None,
        )
    return out, softmax_lse
''',
        encoding="ascii",
    )

    assert patcher._patch_flash_attn_interface(interface)
    assert not patcher._patch_flash_attn_interface(interface)
    patched = interface.read_text(encoding="ascii")
    assert patched.count("# FR13_FA2_QROW32_B1_SPLIT2_INTERFACE") == 1
    assert "and not _fr13_fa2_qrow32_b1_split2_interface_allowed(" in patched
    assert "if tree_bias is not None" in patched
    assert "*(([tree_bias] if tree_bias is not None else []))" in patched

    calls = []

    def recorder(name):
        def call(*args):
            calls.append((name, args))
            return "served-output", "served-lse"

        return call

    fake_torch = types.SimpleNamespace(
        float32=object(),
        ops=types.SimpleNamespace(
            _vllm_fa2_C=types.SimpleNamespace(
                varlen_fwd=recorder("stock"),
                varlen_fwd_tree_bias=recorder("tree_bias"),
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    namespace = {}
    exec(compile(patched, str(interface), "exec"), namespace)
    flash = namespace["flash_attn_varlen_func"]
    tagged = types.SimpleNamespace(
        is_cuda=True,
        dtype=fake_torch.float32,
        shape=(1, 32, 32),
        stride=lambda: (1179791669, 32, 1),
    )

    assert flash("q", "k", "v", num_splits=2, tree_bias=tagged) == (
        "served-output",
        "served-lse",
    )
    assert [(name, args[-2]) for name, args in calls] == [
        ("tree_bias", tagged)
    ]

    ordinary = types.SimpleNamespace(
        **{**tagged.__dict__, "stride": lambda: (32, 32, 1)}
    )
    qrow16 = types.SimpleNamespace(
        **{**tagged.__dict__, "stride": lambda: (1179791667, 32, 1)}
    )
    for rejected_bias, rejected_splits in (
        (None, 2),
        (ordinary, 2),
        (qrow16, 2),
        (tagged, 3),
    ):
        calls.clear()
        with pytest.raises(NotImplementedError, match="num_splits > 1"):
            flash(
                "q",
                "k",
                "v",
                num_splits=rejected_splits,
                tree_bias=rejected_bias,
            )
        assert calls == []

    calls.clear()
    flash("q", "k", "v", num_splits=1)
    assert [name for name, _ in calls] == ["stock"]


def test_exact_geometry_accepts_fused_qkv_row_stride_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, _ = _selector_namespace(monkeypatch, profile_scope=None)
    geometry = _b1_geometry()
    query = geometry["query"]
    key_cache = geometry["key_cache"]
    value_cache = geometry["value_cache"]
    assert tuple(query.stride()) == (32 * 256, 256, 1)
    assert tuple(key_cache.stride()) == (2 * 1024 * 4 * 256, 4 * 256, 256, 1)
    assert tuple(value_cache.stride()) == tuple(key_cache.stride())
    assert value_cache.storage_offset() - key_cache.storage_offset() == 1024 * 4 * 256
    exact_geometry = namespace["_fr13_fa2_qrow32_b1_exact_geometry"]
    exact_args = {
        key: value
        for key, value in geometry.items()
        if key not in {"flash_fn", "softmax_scale"}
    }

    assert exact_geometry(**exact_args)
    assert namespace["_fr13_fa2_qrow32_b1_geometry_mismatches"](
        **exact_args
    ) == ()

    wrong_head = torch.empty((32, 24, 257), dtype=torch.bfloat16)[..., :256]
    assert tuple(wrong_head.stride()) == (24 * 257, 257, 1)
    assert not exact_geometry(**{**exact_args, "query": wrong_head})

    wrong_element = torch.empty((32, 24, 512), dtype=torch.bfloat16)[..., ::2]
    assert tuple(wrong_element.shape) == (32, 24, 256)
    assert int(wrong_element.stride(-1)) == 2
    assert not exact_geometry(**{**exact_args, "query": wrong_element})

    compact_key = torch.empty((1, 1024, 4, 256), dtype=torch.bfloat16)
    compact_value = torch.empty_like(compact_key)
    assert tuple(compact_key.stride()) == (1024 * 4 * 256, 4 * 256, 256, 1)
    assert not exact_geometry(
        **{
            **exact_args,
            "key_cache": compact_key,
            "value_cache": compact_value,
        }
    )


def test_geometry_failure_reports_only_non_secret_operand_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, _ = _selector_namespace(monkeypatch, profile_scope=None)
    geometry = _b1_geometry()
    geometry["max_seqlen_q"] = 1

    mismatches = namespace["_fr13_fa2_qrow32_b1_geometry_mismatches"](
        **{
            key: value
            for key, value in geometry.items()
            if key not in {"flash_fn", "softmax_scale"}
        }
    )
    assert mismatches == ("max_seqlen_q=1",)

    with pytest.raises(RuntimeError) as error:
        namespace["_fr13_fa2_qrow32_b1_live_register"](
            layer=types.SimpleNamespace(layer_name="unused.before.capture"),
            **geometry,
        )
    assert str(error.value).endswith("geometry drifted: max_seqlen_q=1")


def test_live_register_bypasses_partial_events_before_and_after_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, _ = _selector_namespace(monkeypatch, profile_scope=None)
    geometry = _b1_geometry()
    geometry["query"] = torch.empty((25, 24, 256), dtype=torch.bfloat16)
    geometry["max_seqlen_q"] = 25
    original_tree_bias = geometry["tree_bias"]

    selected = namespace["_fr13_fa2_qrow32_b1_live_register"](
        layer=types.SimpleNamespace(layer_name="partial.before.replay"),
        **geometry,
    )

    assert selected is original_tree_bias
    assert namespace["_FR13_FA2_QROW32_B1_LIVE_ATTEMPTED"] is False
    assert namespace["_FR13_FA2_QROW32_B1_LIVE_GRAPHS"] == {}

    namespace["_FR13_FA2_QROW32_B1_LIVE_ATTEMPTED"] = True
    exact = _b1_geometry()
    exact_tree_bias = exact["tree_bias"]
    selected = namespace["_fr13_fa2_qrow32_b1_live_register"](
        layer=types.SimpleNamespace(layer_name="exact.after.replay"), **exact
    )

    assert selected is exact_tree_bias
    assert namespace["_FR13_FA2_QROW32_B1_LIVE_GRAPHS"] == {}


def test_live_register_profile_capture_bypasses_final_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_scope = {
        "descriptor": _profile_descriptor(),
        "graph_id": 41,
        "completed": False,
    }
    namespace, _ = _selector_namespace(
        monkeypatch, profile_scope=profile_scope
    )
    geometry = _b1_geometry()
    geometry["query"] = torch.empty((128, 24, 256), dtype=torch.bfloat16)

    selected = namespace["_fr13_fa2_qrow32_b1_live_register"](
        layer=types.SimpleNamespace(layer_name="profile.throwaway"), **geometry
    )

    assert tuple(selected.shape) == (1, 32, 32)
    assert int(selected.stride(0)) == namespace[
        "_FR13_FA2_QROW32_B1_QROW16_REFERENCE_SENTINEL"
    ]
    assert namespace["_FR13_FA2_QROW32_B1_LIVE_GRAPHS"] == {}


def test_live_register_profile_bypass_requires_profile_memory_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, gdn = _selector_namespace(
        monkeypatch,
        profile_scope={
            "descriptor": _profile_descriptor(),
            "graph_id": 41,
            "completed": False,
        },
    )
    gdn._FR13_FIXED32_PROFILE_MEMORY_SCOPE = False

    with pytest.raises(RuntimeError, match="profile capture scope drifted"):
        namespace["_fr13_fa2_qrow32_b1_live_register"](
            layer=types.SimpleNamespace(layer_name="profile.throwaway"),
            **_b1_geometry(),
        )


def test_profile_bypass_rejects_non_b1_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _profile_descriptor()
    descriptor["num_tokens"] = 128
    namespace, _ = _selector_namespace(
        monkeypatch,
        profile_scope={
            "descriptor": descriptor,
            "graph_id": None,
            "completed": False,
        },
    )

    with pytest.raises(RuntimeError, match="profile capture scope drifted"):
        namespace["_fr13_fa2_qrow32_b1_live_register"](
            layer=types.SimpleNamespace(layer_name="profile.throwaway"),
            **_b1_geometry(),
        )


def test_live_register_final_capture_enforces_geometry_and_all_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, gdn = _selector_namespace(monkeypatch, profile_scope=None)
    gdn._FR13_FIXED32_CAPTURE_CONTEXT = {
        "descriptor": {"runtime_mode": "FULL", "num_reqs": 1},
        "graph_id": 77,
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    geometry = _b1_geometry()
    drifted = dict(geometry)
    drifted["query"] = torch.empty((128, 24, 256), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="live gate geometry drifted"):
        namespace["_fr13_fa2_qrow32_b1_live_register"](
            layer=types.SimpleNamespace(
                layer_name=namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"][0]
            ),
            **drifted,
        )

    for layer_name in namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"]:
        selected = namespace["_fr13_fa2_qrow32_b1_live_register"](
            layer=types.SimpleNamespace(layer_name=layer_name), **geometry
        )
        assert int(selected.stride(0)) == namespace[
            "_FR13_FA2_QROW32_B1_QROW16_REFERENCE_SENTINEL"
        ]

    graph = namespace["_FR13_FA2_QROW32_B1_LIVE_GRAPHS"][77]
    assert set(graph) == set(namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"])
    assert len(graph) == 16


def test_production_profile_warmup_bypasses_before_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, _ = _selector_namespace(
        monkeypatch,
        profile_scope={
            "descriptor": _profile_descriptor(),
            "graph_id": None,
            "completed": False,
        },
    )
    geometry = _b1_geometry()
    geometry.pop("flash_fn")
    geometry.pop("softmax_scale")
    geometry["query"] = torch.empty((128, 24, 256), dtype=torch.bfloat16)

    selection = namespace["_fr13_fa2_qrow32_b1_production_begin"](
        layer=types.SimpleNamespace(layer_name="profile.warmup"), **geometry
    )

    assert selection["profile_capture_bypass"] is True
    assert selection["candidate_served"] is False
    assert int(selection["tree_bias"].stride(0)) == namespace[
        "_FR13_FA2_QROW32_B1_QROW16_REFERENCE_SENTINEL"
    ]
    namespace["_fr13_fa2_qrow32_b1_production_end"](
        selection, completed=True
    )
    assert namespace["_FR13_FA2_QROW32_B1_PRODUCTION_GRAPHS"] == {}


def test_nosplit_production_final_capture_enforces_geometry_and_engages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, gdn = _selector_namespace(monkeypatch, profile_scope=None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    geometry = _b1_geometry()
    geometry.pop("flash_fn")
    geometry.pop("softmax_scale")

    with pytest.raises(RuntimeError, match="no final fixed32 capture context"):
        namespace["_fr13_fa2_qrow32_b1_production_begin"](
            layer=types.SimpleNamespace(
                layer_name=namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"][0]
            ),
            **geometry,
        )

    gdn._FR13_FIXED32_CAPTURE_CONTEXT = {
        "descriptor": _profile_descriptor(),
        "graph_id": 91,
    }
    drifted = dict(geometry)
    drifted["query"] = torch.empty((128, 24, 256), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="production geometry drifted"):
        namespace["_fr13_fa2_qrow32_b1_production_begin"](
            layer=types.SimpleNamespace(
                layer_name=namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"][0]
            ),
            **drifted,
        )

    layer_name = namespace["_FR13_FA2_QROW32_B1_TARGET_LAYERS"][0]
    selection = namespace["_fr13_fa2_qrow32_b1_production_begin"](
        layer=types.SimpleNamespace(layer_name=layer_name), **geometry
    )
    assert selection["arm"] == "nosplit"
    assert selection["candidate_served"] is True
    assert selection["num_splits"] == 0
    assert int(selection["tree_bias"].stride(0)) == 1179791668
    namespace["_fr13_fa2_qrow32_b1_production_end"](
        selection, completed=True
    )
    assert namespace["_FR13_FA2_QROW32_B1_PRODUCTION_GRAPHS"] == {
        91: {"layers": {layer_name}, "arm": "nosplit"}
    }


def test_launcher_requires_exact_binary_source_graph_and_real_gate() -> None:
    text = LAUNCHER.read_text()

    assert (
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM must be empty, nosplit, split2, "
        "visibility, or gqa_pair"
        in text
    )
    assert "FR13_FA2_QROW32_B1_PRODUCTION_ARM must be empty or nosplit" in text
    assert "FR13 qrow32 B1 live gate requires the canonical K64/root1 real task" in text
    assert '"${FR13_FIXED32_MODE:-}" == "hydra27_fixed32"' in text
    assert '"${ENFORCE_EAGER:-0}" == "0"' in text
    assert '"${CUDAGRAPH_MODE:-}" == "FULL_AND_PIECEWISE"' in text
    assert "a9d8a6887b8b27b3a83af60bba7945eb66caff174ba710c2ee2aea92b8e7081a" in text
    assert "22b8c2016443a151bf50f62166f7cc3b9ce45137138d948b76fdfded74c395ff" in text
    assert "--patch-source scripts/fr13_patch_fa2_tree_bias.py" in text
    assert "--patch-source /workspace/scripts/fr13_patch_fa2_tree_bias.py" in text
    assert "FR13_FA2_QROW32_B1_INTERNAL_ATTESTED=1" in text
    assert "astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398" in text


def test_live_gate_is_authenticated_one_task_non_timing_qrow16_served() -> None:
    text = LIVE_GATE.read_text()

    assert "FR13_RUN_QROW32_SPLIT2_LIVE_GATE" in text
    assert "subset_b1_diagnostic_one.json" in text
    assert "fixed32_chat_traffic_audit.json" in text
    assert "all(value is True for value in checks.values())" not in text
    assert "any(value is not True for value in checks.values())" in text
    assert 'ingress = traffic.get("ingress")' in text
    assert 'ingress.get("exact_proxy_engine_attempt_parity") is not True' in text
    assert '"served_return": "qrow16 captured graph output unchanged"' in text
    assert "FR13_SFWD_GPU_TIMER=0" in text
    assert "ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in text
    assert "fr13_qrow32_b1_pass_sidecar.py validate-source" in text
    assert 'PYTHONPATH="$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}"' in text
    assert 'LIVE_ARM=${FR13_QROW32_B1_LIVE_ARM:-split2}' in text
    assert 'arm=live_arm' in text


def test_live_gate_inline_contract_import_resolves_from_repo_root() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "scripts")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts import fr13_fixed32_contract, fr13_qrow32_b1_pass_sidecar",
        ],
        cwd=REPO,
        env=environment,
        check=True,
    )


def test_timing_runner_is_pass_gated_exact4_graph_only() -> None:
    text = TIMING_RUNNER.read_text()

    assert "FR13_RUN_QROW32_NOSPLIT_TIMING" in text
    assert ': "${QROW32_B1_PASS:?set QROW32_B1_PASS' in text
    assert "fr13_qrow32_b1_pass_sidecar.py validate-source" in text
    assert "fr13_qrow32_b1_pass_sidecar.py verify" in text
    assert "FR13_FA2_QROW32_B1_PRODUCTION_ARM=nosplit" in text
    assert "ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in text
    assert "fr13_qrow32_split2_timing.py" in text
    assert "exact16_rule=only_after_exact4_u95_clears_cap" in text
    assert "subset_b4_four.json" in text
    assert "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0" in text


def test_sidecar_is_binary_source_and_nosplit_bound(tmp_path: Path) -> None:
    module = _module(SIDECAR, "qrow32_sidecar")
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(b"combined-qrow32-candidate")
    module.CANDIDATE_SIZE = candidate.stat().st_size
    module.CANDIDATE_SHA256 = module.sha256_file(candidate)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    patch_sha256 = module.sha256_file(PATCHER)
    live = tmp_path / "live.json"
    live.write_text(
        json.dumps(
            _live_payload(module, module.CANDIDATE_SHA256, source_commit, patch_sha256),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    sidecar = tmp_path / "pass.json"
    issued = module.issue_sidecar(
        live_result=live,
        expected_live_sha256=module.sha256_file(live),
        candidate_so=candidate,
        expected_candidate_sha256=module.CANDIDATE_SHA256,
        arm="nosplit",
        patch_source=PATCHER,
        expected_source_commit=source_commit,
        out=sidecar,
    )
    verified = module.verify_sidecar(
        sidecar_path=sidecar,
        expected_sidecar_sha256=module.sha256_file(sidecar),
        candidate_so=candidate,
        expected_candidate_sha256=module.CANDIDATE_SHA256,
        arm="nosplit",
        patch_source=PATCHER,
        expected_source_commit=source_commit,
    )
    assert issued["arm"] == "nosplit"
    assert issued["num_splits"] == 0
    assert issued["selector_sentinel"] == 1179791668
    assert verified["source_commit"] == source_commit
    assert verified["patch_source_sha256"] == patch_sha256

    split2 = _live_payload(
        module,
        module.CANDIDATE_SHA256,
        source_commit,
        patch_sha256,
        arm="split2",
    )
    split2_summary = module.validate_live_result(
        split2,
        candidate_sha256=module.CANDIDATE_SHA256,
        arm="split2",
    )
    assert split2_summary["arm"] == "split2"

    with pytest.raises(ValueError, match="production arm must be nosplit"):
        module.issue_sidecar(
            live_result=live,
            expected_live_sha256=module.sha256_file(live),
            candidate_so=candidate,
            expected_candidate_sha256=module.CANDIDATE_SHA256,
            arm="split2",
            patch_source=PATCHER,
            expected_source_commit=source_commit,
            out=tmp_path / "nosplit-pass.json",
        )

    stale = tmp_path / "stale-live.json"
    stale.write_text(
        json.dumps(
            _live_payload(
                module,
                module.CANDIDATE_SHA256,
                "f" * 40,
                patch_sha256,
                arm="nosplit",
            ),
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="live source commit drifted"):
        module.issue_sidecar(
            live_result=stale,
            expected_live_sha256=module.sha256_file(stale),
            candidate_so=candidate,
            expected_candidate_sha256=module.CANDIDATE_SHA256,
            arm="nosplit",
            patch_source=PATCHER,
            expected_source_commit=source_commit,
            out=tmp_path / "stale-pass.json",
        )

    with pytest.raises(
        ValueError,
        match="live arm must be nosplit, split2, visibility, or gqa_pair",
    ):
        module.validate_live_result(
            _live_payload(
                module, module.CANDIDATE_SHA256, source_commit, patch_sha256
            ),
            candidate_sha256=module.CANDIDATE_SHA256,
            arm="no_split",
        )
