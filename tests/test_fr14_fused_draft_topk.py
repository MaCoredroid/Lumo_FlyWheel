"""Torch-free pins for the FR14 fused full-vocabulary draft top-k lever.

Runs on the CPU-only host: every constant is lifted out of the real source by
text/AST inspection rather than re-typed, so the pin cannot drift from a copy.

What is pinned, and why:

* The kernel is compiled for exactly ONE geometry (V = 248320, k = 3) and one
  SM.  A silent geometry drift is the failure mode that cost this campaign a
  full re-run before ("self-misdescription"), so the literals are asserted.
* The env flag is strict `"0"`/`"1"`, defaults OFF, and is mutually exclusive
  with the K64 top3 op.  "A typo must never be read as OFF" is campaign
  doctrine (`fr13_host_tail_prep.strict_flag`).
* The candidate is admitted only under the K0 (full-vocabulary) drafter
  profile.  Under K64 the logits row is 65536 wide and the kernel would be
  reading past it; the guard, not a comment, is what prevents that.
* The build attestation claims nothing it has not measured.
* The gate is the thing that makes this Tier-A, so the probe's pre-registered
  verdict rule and its powered negative control are pinned too.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KERNEL = REPO / "csrc" / "fr14_dfwd_full_topk.cu"
BUILDER = REPO / "scripts" / "fr14_build_dfwd_full_topk.py"
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
PROBE = (
    REPO
    / "results"
    / "fr14_nvfp4_port_20260816"
    / "fr14_fused_draft_topk_probe.py"
)
EVIDENCE = (
    REPO
    / "results"
    / "fr14_nvfp4_port_20260816"
    / "fr14_fused_draft_topk_probe_result.json"
)
NOTE = REPO / "results" / "fr14_nvfp4_port_20260816" / "fused_draft_topk.md"
ATTESTATION = (
    REPO
    / "results"
    / "fr14_nvfp4_port_20260816"
    / "fr14_fused_draft_topk_build_attestation.json"
)
REPRODUCIBILITY = (
    REPO
    / "results"
    / "fr14_nvfp4_port_20260816"
    / "fr14_fused_draft_topk_build_reproducibility.json"
)


def _cu_constant(name: str) -> int:
    text = KERNEL.read_text()
    match = re.search(rf"constexpr int {name} = ([0-9_]+);", text)
    assert match, f"{name} not found in {KERNEL}"
    return int(match.group(1).replace("_", ""))


def _injected_drafter_body() -> str:
    """The `new = \"\"\"...\"\"\"` replacement body the patcher writes into eagle.py."""
    tree = ast.parse(PATCHER.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_patch_eagle_tree_consumption_verify"
        ):
            for statement in ast.walk(node):
                if (
                    isinstance(statement, ast.Assign)
                    and getattr(statement.targets[0], "id", None) == "new"
                    and isinstance(statement.value, ast.Constant)
                ):
                    return statement.value.value
    raise AssertionError("injected drafter body not found")


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------
def test_kernel_geometry_is_the_served_geometry():
    assert _cu_constant("kVocab") == 248320
    assert _cu_constant("kTopK") == 3
    assert _cu_constant("kThreads") == 256
    assert _cu_constant("kMaxRows") == 4
    assert _cu_constant("kVecWidth") == 8


def test_kernel_is_sm121_only():
    text = KERNEL.read_text()
    assert "properties->major == 12" in text
    assert "properties->minor == 1" in text


def test_kernel_requires_bf16_contiguous_logits_and_int64_outputs():
    text = KERNEL.read_text()
    assert "logits.scalar_type() == at::kBFloat16" in text
    assert "logits.stride(0) == kVocab && logits.stride(1) == 1" in text
    assert "spine_output.scalar_type() == at::kLong" in text
    assert "spine_output.data_ptr() != topk_output.data_ptr()" in text


def test_selection_is_a_total_order_so_reduction_order_cannot_matter():
    text = KERNEL.read_text()
    # value in the high half, (0xFFFFFFFF - index) in the low half: unique keys
    assert "0xffffffffu - static_cast<uint32_t>(index)" in text
    assert "fr14_order_bits" in text
    # NaN outranks every finite value, matching ATen's max semantics
    assert "if (isnan(value)) {" in text


def test_emission_order_reverses_equal_value_runs():
    """ATen's topk emits (value desc, index DESC) while argmax is index ASC."""
    text = KERNEL.read_text()
    assert "fr14_emit_order" in text
    assert "index DESCENDING" in text
    # the spine must be taken BEFORE the reorder (argmax semantics)
    assert (
        "spine_output[row] = static_cast<int64_t>(fr14_key_index(first));\n"
        "      fr14_emit_order(first, second, third);" in text
    )


def test_op_namespace_and_schema():
    text = KERNEL.read_text()
    assert "TORCH_LIBRARY_FRAGMENT(fr14_fused_draft_topk, library)" in text
    assert (
        '"select_out(Tensor(a!) spine_output, Tensor(b!) topk_output, Tensor "'
        in text
    )
    assert '"scratch_numel(int rows, int blocks_per_row) -> int"' in text


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------
def test_builder_claims_nothing_it_has_not_measured():
    text = BUILDER.read_text()
    assert '"status": "BUILT_UNQUALIFIED"' in text
    assert '"performance_measurement": False' in text
    assert '"byte_equality_claim": False' in text
    assert '"real_task_correctness": False' in text
    assert '"production_default_enabled": False' in text


def test_builder_credential_is_the_cubin_not_the_so_sha():
    text = BUILDER.read_text()
    assert '"sha256_is_reproducibility_credential": False' in text
    assert '"is_reproducibility_credential": True' in text
    assert '"cubin_sha256": digest' in text
    assert "cuobjdump" in text and "--extract-elf" in text
    assert 'EXPECTED_ARCH = "12.1a"' in text


def test_kernel_uses_a_named_namespace_so_the_build_is_reproducible():
    """An anonymous namespace puts a per-build hash inside the cubin symbols."""
    text = KERNEL.read_text()
    assert "namespace fr14_fused_draft_topk_impl {" in text
    assert "\nnamespace {\n" not in text


def test_banked_build_is_reproducible_and_is_the_gated_binary():
    if not REPRODUCIBILITY.is_file() or not ATTESTATION.is_file():
        import pytest

        pytest.skip("build evidence not yet banked")
    repro = json.loads(REPRODUCIBILITY.read_text())
    assert repro["reproducible"] is True
    assert len(repro["distinct_cubin_sha256"]) == 1
    assert len(repro["distinct_so_sha256"]) == 1
    assert len(repro["named_namespace_builds"]) >= 3
    attestation = json.loads(ATTESTATION.read_text())
    assert attestation["device_code"]["cubin_sha256"] == repro["distinct_cubin_sha256"][0]
    assert attestation["status"] == "BUILT_UNQUALIFIED"
    if EVIDENCE.is_file():
        # the gate must have run against THAT binary, not some other build
        assert json.loads(EVIDENCE.read_text())["so_sha256"] == (
            attestation["binary"]["sha256"]
        )


# --------------------------------------------------------------------------
# Patcher integration
# --------------------------------------------------------------------------
def test_flag_is_strict_and_defaults_off():
    body = _injected_drafter_body()
    assert 'os.environ.get(\n                "FR14_FUSED_DRAFT_TOPK", "0"\n            )' in body
    assert 'if _fr14_fused_topk_raw not in ("0", "1"):' in body
    assert '"FR14_FUSED_DRAFT_TOPK must be exactly 0 or 1"' in body
    assert '_fr14_fused_topk = _fr14_fused_topk_raw == "1"' in body


def test_candidate_is_admitted_only_under_the_k0_fixed32_width3_profile():
    body = _injected_drafter_body()
    guard = body.split("if _fr14_fused_topk and (", 1)[1].split("):", 1)[0]
    assert "not _fr13_is_fixed32" in guard
    assert "_fr13_dvk_root" in guard            # ROOT must be 0
    assert "_fr13_dvk_configured != 0" in guard  # K must be 0
    assert "not _fr13_single_logits" in guard
    assert "_fr13_dfwd_top3" in guard            # mutually exclusive with K64
    assert "not _fr10_is_wide" in guard
    assert "!= (3, 3, 3, 3, 3)" in guard


def test_binary_identity_is_verified_before_the_op_is_bound():
    body = _injected_drafter_body()
    assert '"FR14_FUSED_DRAFT_TOPK_SHA256"' in body
    assert '"FR14 fused draft top-k binary identity is missing"' in body
    assert '"FR14 fused draft top-k binary identity mismatch"' in body
    prepare = body.split("def _fr14_fused_topk_prepare", 1)[1]
    assert prepare.index("binary identity mismatch") < prepare.index(
        "torch.ops.load_library"
    )


def test_runtime_geometry_and_capture_lifecycle_are_asserted_every_call():
    body = _injected_drafter_body()
    select = body.split("def _fr14_fused_topk_select", 1)[1].split(
        "def _fr13_dh_pad_logits", 1
    )[0]
    assert "tuple(_logits.shape) != (_fr14_ft_rows, 248320)" in select
    assert "_logits.dtype != torch.bfloat16" in select
    # root runs eager, the four loop reads run inside the drafter graph capture
    assert '(_site == "root" and _fr14_ft_capturing)' in select
    assert '(_site == "loop" and not _fr14_ft_capturing)' in select


def test_all_four_deployed_selection_sites_are_routed():
    body = _injected_drafter_body()
    # root spine, root wide, loop spine, loop wide -- two call sites, each
    # emitting both the spine and the width-3 leaves from ONE launch.
    assert body.count("def _fr14_fused_topk_select(") == 1
    assert (
        body.count("_fr14_fused_topk_select(")
        - body.count("def _fr14_fused_topk_select(")
        == 2
    )
    assert "_fr10_wide_topk[0] = _fr14_root_wide" in body
    assert "_fr13_dg_wt = _fr14_step_wide" in body
    # and the graph-buffer copies are skipped, because the kernel wrote them
    assert "if _fr13_dg_cap and not (\n                        _fr13_dfwd_top3 or _fr14_fused_topk\n                    ):" in body


def test_stock_path_is_untouched_when_the_flag_is_off():
    body = _injected_drafter_body()
    # the two ATen calls the K0 arm ships today must still be present verbatim
    assert "draft_token_ids = _fr10_logits.argmax(dim=-1)" in body
    assert "_fr10_wide_topk[0] = torch.topk(" in body
    assert "draft_token_ids = _fr10_step_logits.argmax(dim=-1)" in body
    assert "_fr13_dg_wt = torch.topk(" in body


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------
def test_probe_geometry_matches_the_kernel():
    tree = ast.parse(PROBE.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            values[node.targets[0].id] = node.value.value
    assert values["VOCAB"] == _cu_constant("kVocab")
    assert values["TOPK"] == _cu_constant("kTopK")
    assert values["HEAD_READS_PER_STEP"] == 5


def test_probe_compares_raw_bytes_and_has_a_powered_negative_control():
    text = PROBE.read_text()
    assert ".view(torch.uint8)" in text
    assert "POWERED NEGATIVE CONTROL" in text
    assert "negative_control_all_fire" in text
    assert "torch_self_disagreements" in text
    assert "report[\"gate\"][\"gate_pass\"]" in text
    assert "return 0 if passed else 3" in text


def test_probe_reference_is_exactly_what_ships():
    text = PROBE.read_text()
    assert "spine = logits.argmax(dim=-1)" in text
    assert "wide = torch.topk(logits, TOPK, dim=-1).indices" in text


def test_probe_disclaims_the_claims_it_cannot_make():
    text = PROBE.read_text()
    assert '"analysis_only": True' in text
    assert '"acceptance_valid": False' in text
    assert '"step_envelope_claim": False' in text
    assert "step_wall_ms" not in text.split('"""', 2)[2]


def test_banked_gate_evidence_is_a_zero_mismatch_pass():
    if not EVIDENCE.is_file():
        # The kernel lands before its GPU window in a shared tree; the artifact
        # is the PROMOTION gate, and `test_flag_defaults_off` is what keeps the
        # un-gated kernel out of a serve in the meantime.
        import pytest

        pytest.skip("gate evidence not yet banked (flag is default OFF)")
    report = json.loads(EVIDENCE.read_text())
    gate = report["gate"]
    assert gate["gate_pass"] is True
    assert gate["mismatch_total"] == 0
    assert gate["mismatch_on_tie_cases"] == 0
    assert gate["mismatch_on_plain_cases"] == 0
    assert gate["torch_self_disagreements"] == 0
    assert gate["negative_control_all_fire"] is True
    # the gate is only Tier-A if it actually swept adversarial ties at scale
    assert gate["cases_evaluated"] >= 1000
    assert gate["total_configs"] >= 5000
    assert report["geometry"]["vocab"] == 248320
    assert report["geometry"]["topk"] == 3


def test_probe_gates_the_cuda_graph_replay_path():
    text = PROBE.read_text()
    assert "def run_graph_gate(" in text
    assert "graph.replay()" in text
    assert "ticket_self_cleaned" in text
    assert 'report["graph_gate"]["graph_gate_pass"]' in text


def test_banked_graph_gate_is_a_pass():
    if not EVIDENCE.is_file():
        import pytest

        pytest.skip("gate evidence not yet banked (flag is default OFF)")
    graph_gate = json.loads(EVIDENCE.read_text())["graph_gate"]
    assert graph_gate["graph_gate_pass"] is True
    assert graph_gate["mismatching_replays"] == 0
    assert graph_gate["ticket_self_cleaned"] is True
    assert graph_gate["replays"] >= 16


def test_design_note_exists_and_states_the_measured_saving():
    if not NOTE.is_file():
        import pytest

        pytest.skip("design note lands with the measured numbers")
    text = NOTE.read_text()
    assert "PLACEHOLDER_" not in text, "note still carries unfilled placeholders"
    assert "step_wall_ms" in text  # promotion is judged on the instruments
    assert "TPS" in text           # and explicitly not on TPS
