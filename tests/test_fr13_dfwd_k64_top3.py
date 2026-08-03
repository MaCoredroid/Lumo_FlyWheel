from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CUDA = REPO / "csrc" / "fr13_dfwd_k64_top3.cu"
BUILDER = REPO / "scripts" / "fr13_build_dfwd_k64_top3.py"
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
LIVE_GATE = REPO / "scripts" / "fr13_run_b1_kernel_live_gate.sh"
RUNNER = REPO / "scripts" / "fr13_run_b1_dfwd_k64_top3.sh"


def _eagle_snippet() -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_patch_eagle_tree_consumption_verify"
        ):
            for statement in ast.walk(node):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "new"
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                    and "FR13_DFWD_K64_TOP3" in statement.value.value
                ):
                    return statement.value.value
    raise AssertionError("DFWD K64 top3 replacement snippet not found")


def test_cuda_source_is_exact_one_launch_k64_mapped_top3() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "constexpr int kVocab = 65536;" in source
    assert "constexpr int kTopK = 3;" in source
    assert "constexpr int kThreads = 256;" in source
    assert "<<<1, kThreads, 0," in source
    assert source.count("fr13_dfwd_k64_mapped_top3_kernel<<<") == 1
    assert "for (int index = thread; index < kVocab; index += kThreads)" in source
    assert "__bfloat162float(logits[index])" in source
    assert "mapped_first = id_map[block_first.index]" in source
    assert "spine_output[0] = mapped_first;" in source
    assert "top3_output[0] = mapped_first;" in source
    assert "top3_output[1] = id_map[block_second.index];" in source
    assert "top3_output[2] = id_map[block_third.index];" in source
    assert "properties->major == 12" in source
    assert "properties->minor == 1" in source
    assert "mapped_top3_out(Tensor(a!) spine_output" in source


def test_cuda_source_scans_logits_once_and_has_no_workspace() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert source.count("logits[index]") == 1
    assert "atomic" not in source.lower()
    assert "cudaMalloc" not in source
    assert "torch::empty" not in source
    assert "at::empty" not in source
    assert "__shared__ Candidate warp_candidates[kWarps][kTopK]" in source
    assert "__syncthreads();" in source


def test_builder_is_sm121a_default_off_and_claims_no_live_result() -> None:
    source = BUILDER.read_text(encoding="ascii")

    assert 'EXPECTED_TORCH = "2.11.0+cu130"' in source
    assert 'EXPECTED_ARCH = "12.1a"' in source
    assert '"status": "BUILT_UNQUALIFIED"' in source
    assert '"performance_measurement": False' in source
    assert '"byte_equality_claim": False' in source
    assert '"real_task_correctness": False' in source
    assert '"production_default_enabled": False' in source
    assert '"minimum_redundant_argmax_logit_bytes_eliminated_per_event": 655360' in source
    assert '"minimum_reduction_launches_eliminated_per_event": 45' in source


def test_runtime_selector_is_default_off_and_exact_geometry_only() -> None:
    snippet = _eagle_snippet()

    assert '"FR13_DFWD_K64_TOP3", "0"' in snippet
    assert '_fr13_dfwd_top3_raw not in ("0", "1")' in snippet
    assert "not _fr13_is_fixed32" in snippet
    assert "not _fr13_dvk_root" in snippet
    assert "not _fr13_single_logits" in snippet
    assert "_fr13_dvk_configured != 65536" in snippet
    assert "int(batch_size) != 1" in snippet
    assert "!= (3, 3, 3, 3, 3)" in snippet
    assert '"/tmp/fr13_dfwd_k64_top3.abi3.so"' in snippet
    assert "_fr13_top3_so.is_symlink()" in snippet
    assert "_fr13_top3_digest.hexdigest() != _fr13_top3_expected" in snippet
    assert "torch.ops.fr13_dfwd_top3.mapped_top3_out" in snippet


def test_runtime_writes_graph_outputs_directly_and_fails_closed() -> None:
    snippet = _eagle_snippet()

    assert "_fr13_dfwd_top3_select(" in snippet
    assert '_dg["spine"][token_index]' in snippet
    assert '_dg["wide"][token_index, :, :3]' in snippet
    assert "if _fr13_dg_cap and not _fr13_dfwd_top3:" in snippet
    assert "if _fr13_dfwd_top3:" in snippet
    assert "FR13 DFWD K64 loop top3 requires graph capture" in snippet
    assert "FR13 DFWD K64 top3 runtime geometry/lifecycle drifted" in snippet
    assert "FR13 DFWD K64 top3 graph capture count drifted" in snippet
    assert '"[FR13_DFWD_K64_TOP3] graph captured_calls=4 "' in snippet


def test_replacement_snippet_still_compiles_as_a_method_body() -> None:
    compile(
        "class _C:\n    def propose(self):\n" + _eagle_snippet(),
        "<fr13_dfwd_k64_top3_snippet>",
        "exec",
    )


def test_launcher_mounts_only_an_attested_read_only_candidate() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "FR13_DFWD_K64_TOP3=${FR13_DFWD_K64_TOP3:-0}" in launcher
    assert "FR13_DFWD_K64_TOP3_SO=${FR13_DFWD_K64_TOP3_SO:-}" in launcher
    assert "FR13_DFWD_K64_TOP3_SHA256=${FR13_DFWD_K64_TOP3_SHA256:-}" in launcher
    assert 'case "$FR13_DFWD_K64_TOP3" in' in launcher
    assert "FR13_DFWD_K64_TOP3=0 forbids candidate binary credentials" in launcher
    assert '"$MAX_NUM_SEQS" == "1"' in launcher
    assert '"$FR13_DRAFT_VOCAB_ROOT" == "1"' in launcher
    assert '"${FR13_DRAFT_VOCAB_K:-65536}" == "65536"' in launcher
    assert '"${CUDAGRAPH_MODE:-}" == "FULL_AND_PIECEWISE"' in launcher
    assert '"$FR13_DFWD_K64_TOP3_SO" == /*' in launcher
    assert '! -L "$FR13_DFWD_K64_TOP3_SO"' in launcher
    assert '== "$FR13_DFWD_K64_TOP3_SHA256"' in launcher
    assert (
        '-v "$FR13_DFWD_K64_TOP3_SO:'
        '/tmp/fr13_dfwd_k64_top3.abi3.so:ro"'
    ) in launcher
    assert '"${FR13_DFWD_K64_TOP3_DOCKER_ARGS[@]}" \\' in launcher
    assert '|| "$_v" == "FR13_DFWD_K64_TOP3_SO"' in launcher


def test_real_b1_gate_selects_only_k64_top3_and_requires_graph_marker() -> None:
    gate = LIVE_GATE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "FR13_GATE_DFWD_TOP3=${FR13_GATE_DFWD_TOP3:-0}" in gate
    assert "FR13_GATE_DFWD_TOP3 must be the only enabled kernel candidate" in gate
    assert 'FR13_GATE_DFWD_TOP3_SO must be an absolute regular non-symlink file' in gate
    assert 'FR13_DFWD_K64_TOP3="$FR13_GATE_DFWD_TOP3" \\' in gate
    assert 'FR13_DFWD_K64_TOP3_SO="$FR13_GATE_DFWD_TOP3_SO" \\' in gate
    assert 'FR13_DFWD_K64_TOP3_SHA256="$FR13_GATE_DFWD_TOP3_SHA256" \\' in gate
    assert "[FR13_DFWD_K64_TOP3] ready B1 K64 mapped width3" in gate
    assert "[FR13_DFWD_K64_TOP3] engaged stock_argmax_topk_map_copy=0" in gate
    assert "[FR13_DFWD_K64_TOP3] graph captured_calls=4" in gate

    assert "export FR13_B1_WORKLOAD_PROFILE=k64_root" in runner
    assert "export FR13_GATE_DFWD_TOP3=1" in runner
    assert "export FR13_GATE_TAW_NATIVE=0" in runner
    assert "export FR13_GATE_GDN_BV=0" in runner
    assert "export FR13_FIXED32_CUTLASS_WAVE=stock" in runner
    assert 'exec bash "$SCRIPT_DIR/fr13_run_b1_kernel_live_gate.sh"' in runner
