from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"
VARIANT = ROOT / "scripts/fr13_bigdenom_swe_serve_variant.sh"
RUNNER = ROOT / "scripts/run_swe_bench_q36_a.py"
B1_GATE_RUNNER = ROOT / "scripts/fr13_run_b1_sfwd_conv_postprep_gate.sh"
B4_EMBEDDED_RUNNER = ROOT / "scripts/fr13_run_b4_sfwd_embedded_gate_live_gate.sh"
GATE = ROOT / "scripts/fr13_sfwd_conv_postprep_gate.py"
MODULE = ROOT / "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py"
SELECTOR = "FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION"
BYTE_SELECTOR = "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB"
EMBED_SELECTOR = "FR13_FIXED32_SFWD_EMBED_GATE_CTA"
QROW16_SHA256 = "1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86"
QROW16_PASS_SHA256 = (
    "36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77"
)
QROW32_SHA256 = "a9d8a6887b8b27b3a83af60bba7945eb66caff174ba710c2ee2aea92b8e7081a"
BLOCKS_SHA256 = "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"


def _load_patcher(name: str):
    spec = importlib.util.spec_from_file_location(name, PATCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generated_literals() -> str:
    source = PATCHER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    values: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            values.append(node.value.value)
    return "\n".join(values)


def _literal_tuple(path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, tuple)
            return value
    raise AssertionError(f"{path} lacks {name}")


def _sfwd_profile_capture_runtime() -> dict[str, object]:
    patcher = _load_patcher("fr13_sfwd_profile_capture_runtime")
    tree = ast.parse(patcher._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_fr13_fixed32_sfwd_conv_postprep_profile_capture_active"
    )
    namespace: dict[str, object] = {
        "_FR13_FIXED32_PROFILE_CAPTURE_SCOPE": None,
        "_FR13_FIXED32_PROFILE_MEMORY_SCOPE": False,
        "_FR13_FIXED32_CAPTURE_CONTEXT": None,
        "_FR13_FIXED32_CAPTURE_MANIFESTS": {},
        "_FR13_FIXED32_OBSERVED_CURRENT": None,
        "_FR13_FIXED32_PENDING_EVENT": None,
        "_FR13_FIXED32_CAPTURE_FROZEN": False,
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "<fr13-sfwd-profile-capture>",
            "exec",
        ),
        namespace,
    )
    return namespace


def _embedded_gate_payload(
    *, batch: int, source_commit: str, manifest_sha256: str
) -> dict[str, object]:
    task_ids = (
        "astropy__astropy-12907",
        "astropy__astropy-13033",
        "astropy__astropy-13236",
        "astropy__astropy-13398",
    )[:batch]
    return {
        "schema": (
            "fr13.fixed32.sfwd_conv_postprep.embedded_gate."
            f"{'b1' if batch == 1 else 'exact4_b4'}_gate.v1"
        ),
        "status": "pass",
        "candidate": "fixed32_sfwd_conv_postprep_embedded_gate_cta_v1",
        "source_commit": source_commit,
        "source_manifest_sha256": manifest_sha256,
        "task_ids": list(task_ids),
        "task_markers": [f"swe_verified:{task_id}" for task_id in task_ids],
        "batch_size": batch,
        "concurrency": batch,
        "physical_rows_per_request": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "embedded_gate_cta": True,
        "programs_per_request": 40,
        "layer_count": 48,
        "compared_byte_surfaces": [
            "query_spec",
            "key_spec",
            "value_spec",
            "value_tree",
            "g",
            "beta",
            "commit_source_stage",
        ],
        "reference_returned": True,
        "candidate_returned": False,
        "decision_exact": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_enabled": False,
        "records_sha256": "1" * 64,
        "live_pass_sha256": "2" * 64,
        "health_sha256": "3" * 64,
        "container_env_sha256": "4" * 64,
        "engine_ledger_chain_head_sha256": "5" * 64,
    }


def test_selector_is_default_off_and_baked_only_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SELECTOR, raising=False)
    monkeypatch.delenv(BYTE_SELECTOR, raising=False)
    monkeypatch.delenv(EMBED_SELECTOR, raising=False)
    patcher = _load_patcher("fr13_sfwd_conv_postprep_default_off")
    assert patcher._FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION == "0"
    assert patcher._FR13_FIXED32_SFWD_CONV_POSTPREP_IMPORT == ""
    bindings = patcher._fr13_fixed32_runtime_bindings("hydra27_fixed32")
    assert "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION = False\n" in bindings
    assert "_FR13_FIXED32_SFWD_EMBED_GATE_CTA = False\n" in bindings

    monkeypatch.setenv(SELECTOR, "1")
    monkeypatch.setenv("MAX_NUM_SEQS", "4")
    patcher = _load_patcher("fr13_sfwd_conv_postprep_explicit")
    assert "launch_fixed32_sfwd_conv_postprep_fusion" in (
        patcher._FR13_FIXED32_SFWD_CONV_POSTPREP_IMPORT
    )
    bindings = patcher._fr13_fixed32_runtime_bindings("tail6_fixed32")
    assert "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION = True\n" in bindings


@pytest.mark.parametrize("raw", ("", "true", "2"))
def test_selector_rejects_every_noncanonical_value(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    monkeypatch.setenv(SELECTOR, raw)
    monkeypatch.setenv("MAX_NUM_SEQS", "1")
    patcher = _load_patcher(f"fr13_sfwd_conv_postprep_bad_{raw!r}")
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        patcher._fr13_fixed32_runtime_bindings("hydra27_fixed32")


@pytest.mark.parametrize("batch", (1, 4))
def test_sfwd_profile_capture_guard_covers_throwaway_full_capture_only(
    batch: int,
) -> None:
    runtime = _sfwd_profile_capture_runtime()
    active = runtime[
        "_fr13_fixed32_sfwd_conv_postprep_profile_capture_active"
    ]
    assert callable(active)
    assert active() is False

    runtime["_FR13_FIXED32_PROFILE_MEMORY_SCOPE"] = True
    runtime["_FR13_FIXED32_PROFILE_CAPTURE_SCOPE"] = {
        "descriptor": {
            "runtime_mode": "FULL",
            "num_tokens": 32 * batch,
            "num_reqs": batch,
            "uniform": True,
            "has_lora": False,
            "num_active_loras": 0,
        },
        "graph_id": None,
        "completed": False,
    }
    assert active() is True
    runtime["_FR13_FIXED32_PROFILE_CAPTURE_SCOPE"]["graph_id"] = 41
    assert active() is True

    runtime["_FR13_FIXED32_PROFILE_MEMORY_SCOPE"] = False
    with pytest.raises(RuntimeError, match="profile capture scope drifted"):
        active()


def test_sfwd_profile_capture_records_and_seals_before_full_dummy() -> None:
    generated = _generated_literals()
    start = generated.index(
        "_fr13_conv_postprep_profile_capture = ("
    )
    candidate = generated.index(
        "_fr13_conv_postprep_active = bool(", start
    )
    pending = generated.index(
        '"profile_capture_pending"', candidate
    )
    capture_guard = generated.index(
        "capture lacks preseeded ", candidate
    )
    profile_preseed = generated.index(
        "_fr13_fixed32_preseed_sfwd_conv_postprep_profile_capture("
    )
    full_dummy = generated.index(
        "cudagraph_runtime_mode=cudagraph_runtime_mode,", profile_preseed
    )
    assert start < candidate < capture_guard < pending
    assert profile_preseed < full_dummy


def test_sfwd_profile_producer_runs_capture_shaped_forward_before_sealing(
    tmp_path: Path,
) -> None:
    """The profile scope must publish its own eager operands.

    Sealing alone is not enough: the stock warmup loop in
    ``_warmup_and_capture`` builds ``for_cudagraph_capture=False`` metadata, so
    the GDN builder reports ``num_spec_decodes == 0`` and no layer ever creates
    the conv/post-prep output cache the seal depends on.
    """
    patcher_module = _load_patcher("fr13_sfwd_profile_producer_wiring")
    runner = tmp_path / "gpu_model_runner.py"
    runner.write_text(
        "                profile_seq_lens=profile_seq_lens,\n"
        "            )\n"
        "        self._dummy_run(\n"
        "            desc.num_tokens,\n"
        "            cudagraph_runtime_mode=cudagraph_runtime_mode,\n",
        encoding="utf-8",
    )
    patcher_module.GPU_MODEL_RUNNER_PATH = runner
    patcher_module._FR13_FIXED32_MODE = "hydra27_fixed32"
    assert patcher_module._patch_gpu_model_runner_fixed32_final_full_preseed()
    patched = runner.read_text(encoding="utf-8")
    assert "# FR13_FIXED32_SFWD_PROFILE_PRODUCER" in patched

    producer_needed = patched.index(
        "_fr13_fixed32_sfwd_conv_postprep_profile_producer_needed(\n"
    )
    producer_capturing = patched.index(
        "                is_graph_capturing=True,\n", producer_needed
    )
    profile_preseed = patched.index(
        "_fr13_fixed32_preseed_sfwd_conv_postprep_profile_capture(\n",
        producer_capturing,
    )
    full_dummy = patched.index(
        "            cudagraph_runtime_mode=cudagraph_runtime_mode,\n",
        profile_preseed,
    )
    assert producer_needed < producer_capturing < profile_preseed < full_dummy
    # The producer forward is eager and capture-shaped, never a graph capture.
    producer_block = patched[producer_needed:profile_preseed]
    assert "cudagraph_runtime_mode=CUDAGraphMode.NONE," in producer_block
    assert "force_attention=True," in producer_block

    # The injection is idempotent: a second pass must not stack producers.
    assert not patcher_module._patch_gpu_model_runner_fixed32_final_full_preseed()
    assert runner.read_text(encoding="utf-8") == patched
    assert patched.count(
        "_fr13_fixed32_sfwd_conv_postprep_profile_producer_needed("
    ) == 1


@pytest.mark.parametrize(
    ("fusion", "byte_ab", "enforce_eager"),
    (
        ("0", "1", "1"),  # standalone SFWD conv/post-prep byte gate
        ("1", "0", "1"),  # fusion pinned eager
        ("0", "0", "0"),  # fusion off, graph boot
    ),
)
def test_sfwd_profile_producer_and_seal_are_inert_without_graph_fusion(
    monkeypatch: pytest.MonkeyPatch,
    fusion: str,
    byte_ab: str,
    enforce_eager: str,
) -> None:
    """Only a FUSION+graph boot may arm the profile producer or the seal.

    The standalone byte gate runs ENFORCE_EAGER=1 with cudagraph_mode NONE, so
    it never reaches _warmup_and_capture at all; even if it did, both entry
    points must be no-ops there. Pinning this keeps the profile-capture repair
    from leaking into the eager gate shape.
    """
    patcher_module = _load_patcher("fr13_sfwd_profile_inert")
    monkeypatch.setattr(
        patcher_module, "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION", fusion
    )
    monkeypatch.setattr(
        patcher_module, "_FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB", byte_ab
    )
    monkeypatch.setenv("ENFORCE_EAGER", enforce_eager)
    monkeypatch.setenv("MAX_NUM_SEQS", "1")
    bindings = patcher_module._fr13_fixed32_runtime_bindings("hydra27_fixed32")
    assert "_FR13_FIXED32_SFWD_CONV_POSTPREP_GRAPH = False" in bindings

    tree = ast.parse(patcher_module._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE)
    wanted = {
        "_fr13_fixed32_sfwd_conv_postprep_profile_producer_needed",
        "_fr13_fixed32_preseed_sfwd_conv_postprep_profile_capture",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace: dict[str, object] = {}
    exec(bindings, namespace)
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            "<fr13-sfwd-inert>",
            "exec",
        ),
        namespace,
    )
    # No torch, no registries, no profile scope: both must bail out on the
    # graph-fusion selector alone, before touching anything else.
    needed = namespace[
        "_fr13_fixed32_sfwd_conv_postprep_profile_producer_needed"
    ]
    seal = namespace[
        "_fr13_fixed32_preseed_sfwd_conv_postprep_profile_capture"
    ]
    assert needed("FULL", 32, 1, True, False, 0) is False
    assert seal(1) is None


def test_sfwd_profile_producer_never_runs_in_eager_boot_warm(
    tmp_path: Path,
) -> None:
    """capture_model's eager boot-warm block must not gain a profile producer.

    ENFORCE_EAGER boots return from capture_model before any capture, so the
    producer belongs solely to _warmup_and_capture.
    """
    patcher_module = _load_patcher("fr13_sfwd_eager_boot_warm")
    source = ast.parse(PATCHER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_gpu_model_runner_fixed32_final_full_preseed"
    )
    producer = "_fr13_fixed32_sfwd_conv_postprep_profile_producer_needed"
    body = ast.unparse(function)
    assert body.count(producer) == 1
    eager_marker = "FR13_FIXED32_EAGER_SFWD_CONV_POSTPREP_BOOT_WARM"
    # The producer belongs to the _warmup_and_capture inject only; the eager
    # boot-warm inject (capture_model) must never reference it.
    runner = tmp_path / "gpu_model_runner.py"
    runner.write_text(
        "                profile_seq_lens=profile_seq_lens,\n"
        "            )\n"
        "        self._dummy_run(\n"
        "            desc.num_tokens,\n"
        "            cudagraph_runtime_mode=cudagraph_runtime_mode,\n",
        encoding="utf-8",
    )
    patcher_module.GPU_MODEL_RUNNER_PATH = runner
    patcher_module._FR13_FIXED32_MODE = "hydra27_fixed32"
    assert patcher_module._patch_gpu_model_runner_fixed32_final_full_preseed()
    patched = runner.read_text(encoding="utf-8")
    assert patched.count(producer) == 1
    assert eager_marker not in patched


def test_fused_branch_reports_its_work_to_the_forward_census() -> None:
    """The census expects 48 fused calls; the fused branch must publish them.

    The conv_* counters stay at zero under fusion because the pregather stage
    and the per-layer consumes are subsumed, so this call site is the only
    thing proving the fused kernels reached the captured graph. If it is
    dropped the census fails inverted (expects 48, observes 0).
    """
    source = PATCHER.read_text(encoding="utf-8")
    fragment = next(
        node.value.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "conv_replacement"
        and isinstance(node.value, ast.Constant)
    )
    call = "_fr13_fixed32_observed_sfwd_conv_postprep("
    assert fragment.count(call) == 1
    index = fragment.index(call)
    # Guarded exactly like the unfused conv observers.
    guard = fragment.rindex(
        "_fr13_fixed32_observed_event_active()", 0, index
    )
    assert "_fr13_conv_postprep_capturing" in fragment[guard - 200 : index]
    assert "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION" in fragment[guard - 200 : index]
    # It must sit after the fused launch and before the once-only engagement
    # marker, so it counts every call rather than only the first per layer.
    launch = fragment.rindex("launch_fixed32_sfwd_conv_postprep_fusion(", 0, index)
    marker = fragment.index("[FR13_SFWD_CONV_POSTPREP] production engaged", index)
    assert launch < index < marker


def test_sfwd_profile_seal_fails_loud_without_eager_operands() -> None:
    patcher_module = _load_patcher("fr13_sfwd_profile_seal_failloud")
    tree = ast.parse(patcher_module._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE)
    seal = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_fr13_fixed32_preseed_sfwd_conv_postprep_profile_capture"
    )
    source = ast.unparse(seal)
    # A missing seal must raise instead of silently returning None into the
    # FULL capture, which is how the 2026-08-08 arm died.
    assert "if evidence is None:\n        return None" not in source
    assert "evidence is None" in source
    assert "profile output preseed is incomplete" in source

    producer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_fr13_fixed32_sfwd_conv_postprep_profile_producer_needed"
    )
    producer_source = ast.unparse(producer)
    assert (
        "fixed32_sfwd_conv_postprep_profile_producer_pending"
        in producer_source
    )
    assert "_fr13_fixed32_sfwd_conv_postprep_profile_capture_active()" in (
        producer_source
    )


@pytest.mark.parametrize("raw", ("", "true", "2"))
def test_embedded_gate_selector_is_exact_and_subordinate(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    monkeypatch.setenv(EMBED_SELECTOR, raw)
    patcher = _load_patcher(f"fr13_sfwd_embed_bad_{raw!r}")
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        patcher._fr13_fixed32_runtime_bindings("hydra27_fixed32")

    monkeypatch.setenv(EMBED_SELECTOR, "1")
    monkeypatch.setenv(BYTE_SELECTOR, "0")
    monkeypatch.setenv("MAX_NUM_SEQS", "1")
    patcher = _load_patcher(f"fr13_sfwd_embed_naked_{raw!r}")
    with pytest.raises(RuntimeError, match="production or byte gate"):
        patcher._fr13_fixed32_runtime_bindings("hydra27_fixed32")

    monkeypatch.setenv(BYTE_SELECTOR, "1")
    patcher = _load_patcher(f"fr13_sfwd_embed_exact_{raw!r}")
    bindings = patcher._fr13_fixed32_runtime_bindings("hydra27_fixed32")
    assert "_FR13_FIXED32_SFWD_EMBED_GATE_CTA = True\n" in bindings

    monkeypatch.setenv(BYTE_SELECTOR, "0")
    monkeypatch.setenv(SELECTOR, "1")
    patcher = _load_patcher(f"fr13_sfwd_embed_production_{raw!r}")
    bindings = patcher._fr13_fixed32_runtime_bindings("hydra27_fixed32")
    assert "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION = True\n" in bindings
    assert "_FR13_FIXED32_SFWD_EMBED_GATE_CTA = True\n" in bindings


def test_patch_contract_rejects_non_k64_and_accepts_credentialed_full_graph_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patcher = _load_patcher("fr13_sfwd_conv_postprep_contract")
    monkeypatch.setattr(patcher, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION", "1"
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB", "0"
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE", ""
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_GDN_PATH_BV_PRODUCTION", ""
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB", "0"
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION", "0"
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB", "0"
    )
    monkeypatch.setattr(
        patcher, "_fr13_fixed32_eager_boot_warm_contract", lambda: None
    )
    exact = {
        "MAX_NUM_SEQS": "1",
        "SWE_CONCURRENCY": "1",
        "ENFORCE_EAGER": "1",
        "FR13_DRAFT_VOCAB_ROOT": "1",
        "FR13_DRAFT_VOCAB_K": "32768",
        "FR13_FIXED32_CONV_SOURCE_BATCH": "0",
        "FR13_RING_EXPORT": "1",
        "FR13_FLAGS_INKERNEL": "1",
        "FR13_TREE_RUNROW_INIT": "1",
        "FR13_TREE_CONV_FUSED": "1",
        "FR13_CONV_WB_BATCHED": "1",
        "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB": "0",
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION": "0",
        "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB": "0",
        "FR13_FA2_QROW16_PRODUCTION": "1",
        "FR13_FA2_QROW16_SO_SHA256": QROW16_SHA256,
        "FR13_FA2_QROW16_LIVE_PASS_SHA256": QROW16_PASS_SHA256,
        "FR13_FIXED32_B1_DIAGNOSTIC": "0",
    }
    for name, value in exact.items():
        monkeypatch.setenv(name, value)
    expected_mask, expected_active = patcher._FR13_FIXED32_MODES[
        "hydra27_fixed32"
    ]
    monkeypatch.setenv("FR13_FIXED32_VALID_MASK", hex(expected_mask))
    monkeypatch.setenv("FR13_FIXED32_ACTIVE_NODES", str(expected_active))
    with pytest.raises(RuntimeError, match="conv/post-prep fusion requires"):
        patcher._fr13_fixed32_validate_patch_env()

    monkeypatch.setenv("FR13_DRAFT_VOCAB_K", "65536")
    monkeypatch.setenv("ENFORCE_EAGER", "0")
    monkeypatch.setenv("CUDAGRAPH_MODE", "PIECEWISE")
    with pytest.raises(RuntimeError, match="conv/post-prep fusion requires"):
        patcher._fr13_fixed32_validate_patch_env()

    monkeypatch.setenv("CUDAGRAPH_MODE", "FULL_AND_PIECEWISE")
    for name in (
        "FR13_FIXED32_WORK_CENSUS",
        "FR13_FIXED32_DEVICE_PUBLISH",
        "FR13_FIXED32_ACCEPT_PACK",
        "FR13_FIXED32_REQKEY_DEVICE",
        "FR13_FIXED32_KV_REMAP16",
        "FR13_FIXED32_COMMIT_DEVICE_FILL",
        "FR13_DEVICE_MULTIDRAFT",
        "FR13_DRAFTER_GRAPH",
        "FR13_DRAFTER_SINGLE_LOGITS",
        "FR13_DM_DEPTHSYNC",
        "FR13_TAW",
        "FR13_PARENT_GATHER",
        "FR13_SUBTREE_PARALLEL",
        "FR13_EAGER_PACK",
        "FR13_COMMIT_BATCH_OUTPUT",
        "FR13_COMMITTER_NATIVE",
        "FR13_COMMITTER_BATCHED",
        "FR13_COMMITTER_GRAPH",
        "FR13_REPLAY_ROUTE",
        "FR13_ATTN_KV_REMAP",
        "FR13_SLOT_REORDER",
        "FR13_KV_REMAP_SYNCFREE",
        "FR13_CONV_WB_FUSED",
        "FR13_CONV_PREGATHER",
        "FR13_CONV_COMMITTED_PATH",
        "FR13_APC_COMMIT_TO_RUNNING_ROW",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("FR13_FIXED32_TAW_WALK_CAP", "12")
    source_commit = "1" * 40
    manifest = tmp_path / "source_manifest.json"
    source_paths = patcher._FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_PATHS
    manifest.write_text(
        json.dumps(
            {
                "schema": "fr13.fixed32.sfwd_conv_postprep.source_manifest.v1",
                "candidate": "fixed32_sfwd_conv_postprep_frontier5_direct_v1",
                "source_commit": source_commit,
                "files": {
                    relative: {
                        "bytes": len((ROOT / relative).read_bytes()),
                        "sha256": hashlib.sha256(
                            (ROOT / relative).read_bytes()
                        ).hexdigest(),
                    }
                    for relative in source_paths
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.chmod(0o400)
    live_pass = tmp_path / "live_pass.json"
    live_payload = {
        "schema": "fr13.fixed32.sfwd_conv_postprep.live_pass.v1",
        "status": "byte_pass_source_only",
        "candidate": "fixed32_sfwd_conv_postprep_frontier5_direct_v1",
        "source_commit": source_commit,
        "source_manifest_sha256": manifest_sha256,
        "fixed32_mode": "hydra27_fixed32",
        "batch_size": 1,
        "task_count": 1,
        "task_marker": "swe_verified:astropy__astropy-12907",
        "layer_count": 48,
        "physical_rows_per_request": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks_sha256": BLOCKS_SHA256,
        "qrow16_production": True,
        "qrow16_fa2_sha256": QROW16_SHA256,
        "qrow16_live_pass_sha256": QROW16_PASS_SHA256,
        "compared_byte_surfaces": [
            "query_spec",
            "key_spec",
            "value_spec",
            "value_tree",
            "g",
            "beta",
            "commit_source_stage",
        ],
        "layers": [
            {
                "layer_key": f"0x{index + 1:x}",
                "layer_prefix_sha256": f"{index + 1:064x}",
            }
            for index in range(48)
        ],
        "real_task_authenticated": True,
        "reference_always_served": True,
        "candidate_returned": False,
        "reference_decision": "serve_incumbent",
        "candidate_decision": "shadow_only",
        "decision_exact": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
        "comparisons": 336,
        "mismatches": 0,
        "differing_bytes": 0,
        "errors": 0,
    }
    live_pass.write_text(
        json.dumps(live_payload, sort_keys=True) + "\n", encoding="ascii"
    )
    live_pass.chmod(0o400)
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_JSON", str(live_pass)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_SHA256",
        hashlib.sha256(live_pass.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_PATH", str(manifest)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256",
        manifest_sha256,
    )
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT", source_commit
    )
    patcher._fr13_fixed32_validate_patch_env()

    monkeypatch.setenv("FR13_FA2_QROW16_PRODUCTION", "0")
    monkeypatch.setenv("FR13_FA2_QROW16_SO_SHA256", "")
    monkeypatch.setenv("FR13_FA2_QROW16_LIVE_PASS_SHA256", "")
    monkeypatch.setenv("FR13_FA2_QROW32_B1_LIVE_AB_ARM", "")
    monkeypatch.setenv("FR13_FA2_QROW32_B1_PRODUCTION_ARM", "nosplit")
    monkeypatch.setenv("FR13_FA2_QROW32_B1_SO_SHA256", QROW32_SHA256)
    monkeypatch.setenv("FR13_FA2_QROW32_B1_SO_SIZE", "300154616")
    patcher._fr13_fixed32_validate_patch_env()

    monkeypatch.setenv("FR13_FA2_QROW16_PRODUCTION", "1")
    with pytest.raises(RuntimeError, match="one credentialed Qrow arm"):
        patcher._fr13_fixed32_validate_patch_env()
    monkeypatch.setenv("FR13_FA2_QROW16_PRODUCTION", "0")

    manifest.chmod(0o600)
    manifest_payload = json.loads(manifest.read_text(encoding="ascii"))
    manifest_payload["files"][source_paths[-1]]["sha256"] = "0" * 64
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True) + "\n", encoding="ascii"
    )
    manifest.chmod(0o400)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    live_payload["source_manifest_sha256"] = manifest_sha256
    live_pass.chmod(0o600)
    live_pass.write_text(
        json.dumps(live_payload, sort_keys=True) + "\n", encoding="ascii"
    )
    live_pass.chmod(0o400)
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256",
        manifest_sha256,
    )
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_SHA256",
        hashlib.sha256(live_pass.read_bytes()).hexdigest(),
    )
    with pytest.raises(RuntimeError, match="not bound to the runtime candidate"):
        patcher._fr13_fixed32_validate_patch_env()


@pytest.mark.parametrize("batch", (1, 4))
def test_patch_contract_admits_source_bound_embedded_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    batch: int,
) -> None:
    monkeypatch.setenv(SELECTOR, "1")
    monkeypatch.setenv(BYTE_SELECTOR, "0")
    monkeypatch.setenv(EMBED_SELECTOR, "1")
    patcher = _load_patcher(f"fr13_sfwd_embedded_production_b{batch}")
    monkeypatch.setattr(patcher, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION", "1"
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB", "0"
    )
    monkeypatch.setattr(patcher, "_FR13_FIXED32_SFWD_EMBED_GATE_CTA", "1")
    monkeypatch.setattr(patcher, "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE", "")
    monkeypatch.setattr(patcher, "_FR13_FIXED32_GDN_PATH_BV_PRODUCTION", "")
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB", "0"
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION", "0"
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB", "0"
    )
    monkeypatch.setattr(
        patcher, "_fr13_fixed32_eager_boot_warm_contract", lambda: None
    )
    qrow16 = batch == 1
    exact = {
        "MAX_NUM_SEQS": str(batch),
        "SWE_CONCURRENCY": str(batch),
        "ENFORCE_EAGER": "1",
        "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
        "FR13_DRAFT_VOCAB_ROOT": "1",
        "FR13_DRAFT_VOCAB_K": "65536",
        "FR13_FIXED32_CONV_SOURCE_BATCH": "0",
        "FR13_RING_EXPORT": "1",
        "FR13_FLAGS_INKERNEL": "1",
        "FR13_TREE_RUNROW_INIT": "1",
        "FR13_TREE_CONV_FUSED": "1",
        "FR13_CONV_WB_BATCHED": "1",
        "FR13_FIXED32_B1_DIAGNOSTIC": "0",
        "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB": "0",
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION": "0",
        "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB": "0",
        "FR13_FA2_QROW16_LIVE_PAGED_AB": "0",
        "FR13_FA2_QROW16_PRODUCTION": "1" if qrow16 else "0",
        "FR13_FA2_QROW16_SO_SHA256": QROW16_SHA256 if qrow16 else "",
        "FR13_FA2_QROW16_LIVE_PASS_SHA256": (
            QROW16_PASS_SHA256 if qrow16 else ""
        ),
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "",
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "",
        "FR13_FIXED32_CUTLASS_WAVE": "stock",
        "FR13_FIXED32_CUTLASS_WAVE_PRODUCTION": "0",
    }
    for name, value in exact.items():
        monkeypatch.setenv(name, value)
    expected_mask, expected_active = patcher._FR13_FIXED32_MODES[
        "hydra27_fixed32"
    ]
    monkeypatch.setenv("FR13_FIXED32_VALID_MASK", hex(expected_mask))
    monkeypatch.setenv("FR13_FIXED32_ACTIVE_NODES", str(expected_active))
    for name in (
        "FR13_FIXED32_WORK_CENSUS",
        "FR13_FIXED32_DEVICE_PUBLISH",
        "FR13_FIXED32_ACCEPT_PACK",
        "FR13_FIXED32_REQKEY_DEVICE",
        "FR13_FIXED32_KV_REMAP16",
        "FR13_FIXED32_COMMIT_DEVICE_FILL",
        "FR13_DEVICE_MULTIDRAFT",
        "FR13_DRAFTER_GRAPH",
        "FR13_DRAFTER_SINGLE_LOGITS",
        "FR13_DM_DEPTHSYNC",
        "FR13_TAW",
        "FR13_PARENT_GATHER",
        "FR13_SUBTREE_PARALLEL",
        "FR13_EAGER_PACK",
        "FR13_COMMIT_BATCH_OUTPUT",
        "FR13_COMMITTER_NATIVE",
        "FR13_COMMITTER_BATCHED",
        "FR13_COMMITTER_GRAPH",
        "FR13_REPLAY_ROUTE",
        "FR13_ATTN_KV_REMAP",
        "FR13_SLOT_REORDER",
        "FR13_KV_REMAP_SYNCFREE",
        "FR13_CONV_WB_FUSED",
        "FR13_CONV_PREGATHER",
        "FR13_CONV_COMMITTED_PATH",
        "FR13_APC_COMMIT_TO_RUNNING_ROW",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("FR13_FIXED32_TAW_WALK_CAP", "12")

    source_commit = "6" * 40
    source_paths = patcher._FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_PATHS
    manifest = tmp_path / f"embedded_b{batch}.source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "fr13.fixed32.sfwd_conv_postprep.source_manifest.v1",
                "candidate": "fixed32_sfwd_conv_postprep_frontier5_direct_v1",
                "source_commit": source_commit,
                "files": {
                    relative: {
                        "bytes": len((ROOT / relative).read_bytes()),
                        "sha256": hashlib.sha256(
                            (ROOT / relative).read_bytes()
                        ).hexdigest(),
                    }
                    for relative in source_paths
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.chmod(0o400)
    production_pass = tmp_path / f"embedded_b{batch}.production_pass.json"
    production_pass.write_text(
        json.dumps(
            _embedded_gate_payload(
                batch=batch,
                source_commit=source_commit,
                manifest_sha256=manifest_sha256,
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    production_pass.chmod(0o400)
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_JSON",
        str(production_pass),
    )
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_SHA256",
        hashlib.sha256(production_pass.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_PATH", str(manifest)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256",
        manifest_sha256,
    )
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT", source_commit
    )
    patcher._fr13_fixed32_validate_patch_env()

    monkeypatch.setenv("ENFORCE_EAGER", "0")
    with pytest.raises(RuntimeError, match="embedded-gate B4|embedded gate"):
        patcher._fr13_fixed32_validate_patch_env()


def test_patch_contract_rejects_naked_serving_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SELECTOR, "1")
    monkeypatch.setenv("MAX_NUM_SEQS", "1")
    patcher = _load_patcher("fr13_sfwd_conv_postprep_naked_selector")
    monkeypatch.setattr(patcher, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION", "1"
    )
    monkeypatch.setattr(
        patcher, "_FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB", "0"
    )
    monkeypatch.setattr(patcher, "_fr13_fixed32_eager_boot_warm_contract", lambda: None)
    exact = {
        "MAX_NUM_SEQS": "1",
        "SWE_CONCURRENCY": "1",
        "ENFORCE_EAGER": "0",
        "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
        "FR13_DRAFT_VOCAB_ROOT": "1",
        "FR13_DRAFT_VOCAB_K": "65536",
        "FR13_FIXED32_CONV_SOURCE_BATCH": "0",
        "FR13_RING_EXPORT": "1",
        "FR13_FLAGS_INKERNEL": "1",
        "FR13_TREE_RUNROW_INIT": "1",
        "FR13_TREE_CONV_FUSED": "1",
        "FR13_CONV_WB_BATCHED": "1",
        "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB": "0",
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION": "0",
        "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB": "0",
        "FR13_FA2_QROW16_PRODUCTION": "1",
        "FR13_FA2_QROW16_SO_SHA256": QROW16_SHA256,
        "FR13_FA2_QROW16_LIVE_PASS_SHA256": QROW16_PASS_SHA256,
        "FR13_FIXED32_B1_DIAGNOSTIC": "0",
    }
    for name, value in exact.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="source-bound PASS/manifest"):
        patcher._fr13_fixed32_validate_patch_env()


def test_generated_route_serves_all_direct_outputs_and_has_no_fallback() -> None:
    generated = _generated_literals()
    assert "launch_fixed32_sfwd_conv_postprep_fusion(" in generated
    assert "not 1 <= _fr13_conv_postprep_b <= 4" in generated
    assert "int(_fr10_tree_n) != 32" in generated
    assert "qualification_profile=\"k64_root\"" in generated
    assert "draft_vocab_k=65536" in generated
    assert "draft_vocab_root=1" in generated
    assert "source_only_qualification=True" in generated
    assert "capture_binding=_fr13_conv_postprep_binding" in generated
    assert "capture lacks preseeded " in generated
    assert "output bindings" in generated
    assert "_fr13_fixed32_preseed_sfwd_conv_postprep_capture" in generated
    assert "_fr13_fixed32_require_sfwd_conv_postprep_pass()" in PATCHER.read_text(
        encoding="utf-8"
    )
    assert "conv_tap=None" in generated
    assert "(1, _fr13_conv_postprep_rows, 16, 128)" in generated
    assert "(1, _fr13_conv_postprep_rows, 48, 128)" in generated
    assert "(_fr13_conv_postprep_rows, 48)" in generated
    assert "if not _fr13_conv_postprep_active:\n                        mixed_qkv_spec = _fr10_tree_conv_out" in generated
    assert "if _FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION:" in generated
    assert "value_tree = _fr13_conv_postprep_value_tree" in generated
    assert "g_tree = _fr13_conv_postprep_g" in generated
    assert "beta_tree = _fr13_conv_postprep_beta" in generated
    assert "_fr13_conv_postprep_candidate\n                        or (" in generated
    assert "[FR13_SFWD_CONV_POSTPREP] production engaged " in generated


def test_source_closure_is_identical_in_host_patcher_and_runtime() -> None:
    gate_files = _literal_tuple(GATE, "SOURCE_FILES")
    patcher_files = _literal_tuple(
        PATCHER, "_FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_PATHS"
    )
    module_tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    module_assignment = next(
        node
        for node in module_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "SOURCE_FILES"
    )
    names = {
        "SOURCE_RELATIVE_PATH": (
            "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py"
        ),
        "KERNEL_SOURCE_RELATIVE_PATH": (
            "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion_kernel.py"
        ),
    }
    module_files = tuple(
        item.value if isinstance(item, ast.Constant) else names[item.id]
        for item in module_assignment.value.elts
    )
    assert gate_files == patcher_files == module_files
    assert len(gate_files) == 19


def test_launcher_and_real_task_runner_forward_the_selector() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    variant = VARIANT.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert f"{SELECTOR}=${{{SELECTOR}:-0}}" in launcher
    assert f'-e {SELECTOR}="${SELECTOR}"' in launcher
    assert f'-e {BYTE_SELECTOR}="${BYTE_SELECTOR}"' in launcher
    assert f'-e {EMBED_SELECTOR}="${EMBED_SELECTOR}"' in launcher
    assert f"{SELECTOR}=${{{SELECTOR}:-0}}" in variant
    assert f"{BYTE_SELECTOR}=${{{BYTE_SELECTOR}:-0}}" in variant
    assert f"{EMBED_SELECTOR}=${{{EMBED_SELECTOR}:-0}}" in variant
    assert (
        f'|| ( "${SELECTOR}" == "1" \\\n'
        '           && "${ENFORCE_EAGER:-0}" == "1" )'
    ) in variant
    assert SELECTOR in runner
    assert BYTE_SELECTOR in runner
    assert EMBED_SELECTOR in runner
    for source in (launcher, variant):
        assert f"{SELECTOR} must be exactly 0 or 1" in source
        assert source.count(SELECTOR) >= 7
    assert "_fr13_sfwd_qrow32_production=1" in launcher
    assert "identity_onen_n5120_fullgrid_b1|identity_wide256_fullgrid_b1" in launcher
    assert "source-qualified embedded Hydra27 eager B1/B4" in launcher
    assert "fr13_sfwd_conv_postprep_gate.py validate-pass" in launcher


def test_embedded_gate_runners_bind_real_b1_and_exact4_b4() -> None:
    b1 = B1_GATE_RUNNER.read_text(encoding="utf-8")
    b4 = B4_EMBEDDED_RUNNER.read_text(encoding="utf-8")
    assert f"{EMBED_SELECTOR}=${{{EMBED_SELECTOR}:-0}}" in b1
    assert "validate-embedded" in b1
    for token in (
        "config/fr13_fixed32/subset_b4_four.json",
        "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5",
        "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff",
        "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d",
        "MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4",
        "FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536",
        "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=1",
        "FR13_FIXED32_SFWD_EMBED_GATE_CTA=1",
        "--batch-size 4",
        "validate-embedded",
    ):
        assert token in b4
    assert "PROBE_ONLY=1" not in b4
    assert "ACCEPT_SPEED_PROBE=1" not in b4
