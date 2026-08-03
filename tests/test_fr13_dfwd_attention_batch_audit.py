from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import fr13_dfwd_attention_batch_audit as audit


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "fr13_fixed32_dfwd_attention_batch_audit_20260803"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("mode", "tail6_fixed32", "mode must be hydra27_fixed32"),
        ("valid_mask", 0, "Hydra27 valid mask drifted"),
        ("logical_drafts", 26, "logical draft count"),
        ("physical_rows", 31, "physical rows must be 32"),
        ("draft_vocab_root", 0, "draft vocabulary root must be 1"),
        ("draft_vocab_k", 32_768, "draft vocabulary K must be 65536"),
        ("batch_size", 2, "requires B1"),
        ("post_root_sites", 3, "site count must be 4"),
        ("kernel", "another_kernel", "kernel identity drifted"),
        ("block_m", 16, "BM8 geometry drifted"),
        ("tile_rows", 48, "paged tile geometry drifted"),
        ("causal", False, "must remain causal"),
        ("full_window", False, "must use a full window"),
        ("max_query_length", 2, "must remain 1"),
    ),
)
def test_exact_topology_and_algebra_guards_fail_closed(
    field: str, value: object, message: str
) -> None:
    contract = replace(audit.fixed32_hydra27_contract(), **{field: value})
    with pytest.raises(ValueError, match=message):
        audit.validate_contract(contract)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("q_shape", (2, 24, 256), "query shape drifted"),
        ("q_strides", (1, 1, 1), "query layout"),
        ("out_strides", (1, 1, 1), "output layout"),
        ("kv_inner_shape", (512, 4, 256), "KV inner shape"),
        ("kv_inner_strides", (1, 1, 1), "KV inner layout"),
        ("q_dtype", "float16", "data dtype drifted"),
        ("metadata_dtype", "int64", "metadata dtype drifted"),
        ("cu_seqlens_q_values", (0, 2), "exact B1 Q1"),
        ("seqused_k_shape", (2,), "sequence length layout"),
        ("block_table_rank", 1, "block-table layout"),
        ("block_table_last_stride", 2, "block-table layout"),
    ),
)
def test_exact_tensor_layout_guards_fail_closed(
    field: str, value: object, message: str
) -> None:
    contract = audit.fixed32_hydra27_contract()
    sites = list(contract.sites)
    sites[2] = replace(sites[2], **{field: value})
    with pytest.raises(ValueError, match=message):
        audit.validate_contract(replace(contract, sites=tuple(sites)))


def test_alias_guards_bind_disjoint_views_and_reused_workspaces() -> None:
    contract = audit.fixed32_hydra27_contract()
    assert audit.validate_contract(contract) is contract

    sites = list(contract.sites)
    sites[0] = replace(sites[0], out_storage=sites[0].q_storage)
    with pytest.raises(ValueError, match="within-site buffers alias"):
        audit.validate_contract(replace(contract, sites=tuple(sites)))

    sites = list(contract.sites)
    sites[3] = replace(sites[3], q_storage="private_level3_query")
    with pytest.raises(ValueError, match="workspace ownership drifted for q_storage"):
        audit.validate_contract(replace(contract, sites=tuple(sites)))


def test_recurrence_proves_attention_sites_are_not_batch_independent() -> None:
    verdict = audit.fusion_verdict()
    assert verdict["eligible"] is False
    assert verdict["classification"] == "recurrent_nonconsecutive_attention_sites"
    assert verdict["current_attention_launches_per_event"] == 4
    assert verdict["minimum_lossless_attention_launches_per_event"] == 4
    assert verdict["launches_removed"] == 0
    assert len(verdict["blocked_edges"]) == 3

    for level, edge in enumerate(verdict["blocked_edges"]):
        assert edge["from_attention_site"] == level
        assert edge["to_attention_site"] == level + 1
        assert edge["blocking_path"] == [
            f"attention_{level}.output",
            f"model_{level}.post_attention_hidden",
            f"lm_head_{level}.logits",
            f"top3_{level}.spine_rank0",
            f"model_{level + 1}.input_ids",
            f"attention_{level + 1}.query",
        ]
        assert edge["state_transition"] == {
            "sequence_length_update": "in_place_increment_or_rollover",
            "kv_suffix_write_before_next_attention": True,
        }

    assert "runner-up branches are packing-only" in verdict["wide_branch_note"]
    assert verdict["alias_guard"] == {
        "query_workspace_reused_across_sites": True,
        "output_workspace_reused_across_sites": True,
        "sequence_lengths_mutated_in_place": True,
        "paged_kv_suffix_mutated_between_sites": True,
        "deferred_batching_requires_query_and_length_snapshots": True,
    }


def test_static_launch_work_and_byte_ledger_is_exact() -> None:
    ledger = audit.static_ledger(first_sequence_length=1024)
    assert ledger["launches"] == {
        "kernel_unified_attention_2d": 4,
        "ctas_per_launch": 4,
        "ctas_per_event": 16,
        "lossless_attention_only_fused_launches": None,
    }
    assert ledger["bytes"]["q_reads_per_event"] == 49_152
    assert ledger["bytes"]["output_writes_per_event"] == 49_152
    assert ledger["bytes"]["kv_payload_per_sequence_row"] == 4096
    assert ledger["bytes"]["vector_page_index_loads_per_full_tile"] == 512
    assert ledger["bytes"]["scalar_page_index_loads_per_full_tile"] == 16
    assert ledger["bytes"]["scalar_page_index_logical_saving_per_full_tile"] == 496
    assert ledger["bytes"]["scalar_page_index_saving_vs_kv_payload_fraction"] == (
        496 / 131_072
    )
    assert ledger["work"] == {
        "tile_rows": 32,
        "bm8_physical_qk_pv_flops_per_full_tile_across_four_kv_heads": 1_048_576,
        "live_head_qk_pv_flops_per_full_tile_across_four_kv_heads": 786_432,
        "physical_to_live_head_work_ratio": 4 / 3,
        "tile_count_per_event": (
            "ceil(S/32)+ceil((S+1)/32)+ceil((S+2)/32)+ceil((S+3)/32)"
        ),
    }
    assert ledger["evaluation"] == {
        "first_sequence_length": 1024,
        "sequence_lengths": [1024, 1025, 1026, 1027],
        "sum_sequence_rows": 4102,
        "tile_counts": [32, 33, 33, 33],
        "tile_count": 131,
        "kv_payload_read_bytes": 16_801_792,
        "bm8_physical_qk_pv_flops": 137_363_456,
        "live_head_qk_pv_flops": 103_022_592,
        "vector_page_index_logical_bytes": 67_072,
        "scalar_page_index_logical_bytes": 2_096,
        "same_width_common_prefix_stage_extra_bytes": 8_388_608,
    }


def test_manual_kv_or_index_preparation_is_not_a_safe_traffic_win() -> None:
    ledger = audit.static_ledger()
    stage = ledger["same_width_common_prefix_stage"]
    assert stage == {
        "original_common_prefix_reads": "4*4096*S",
        "staged_common_prefix_traffic": "6*4096*S",
        "traffic_delta_bytes": "+8192*S",
        "standalone_preparation_launch_delta": 1,
        "verdict": "strictly_more_traffic",
    }
    index = ledger["index_preparation"]
    assert index["tiles_per_page"] == 32
    assert index["block_table_stable_across_sites"] is True
    assert index["once_per_event_expanded_tile_map_can_be_exact"] is True
    assert index["source_page_table_unique_bytes"] == "4*ceil((S+3)/1024)"
    assert index["expanded_tile_map_write_bytes"] == "4*ceil((S+3)/32)"
    assert index["expanded_map_entries_per_source_page"] == 32
    assert index["expanded_map_consumer_scalar_bytes"] == "16*T"
    assert index["direct_scalar_lookup_needs_expanded_map"] is False
    assert index["physical_byte_saving_proven_host_only"] is False
    assert (
        ledger["bytes"]["scalar_page_index_saving_vs_kv_payload_fraction"]
        < 0.004
    )


def test_patcher_contains_the_exact_recurrent_and_alias_anchors() -> None:
    source = (
        ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
    ).read_text(encoding="utf-8")
    loop_start = source.index("for token_index in range(_fr10_spine_steps):")
    loop_end = source.index("if _fr13_dg_cap:", loop_start)
    loop = source[loop_start:loop_end]
    model = loop.index("ret_hidden_states = self.model(**model_kwargs)")
    hidden = loop.index("hidden_states = hidden_states[:batch_size]", model)
    logits = loop.index("_fr10_step_logits", hidden)
    top3 = loop.index("_fr13_dfwd_top3_select(", logits)
    append = loop.index("_fr10_spine_tokens.append(draft_token_ids)", top3)
    assert model < hidden < logits < top3 < append
    assert "common_attn_metadata.max_seq_len + 1" in loop
    assert "eagle_step_update_slot_mapping_and_metadata(" in loop
    assert "self.hidden_states[:batch_size] = hidden_states" in loop

    capture = source.index("# The drafter reuses intermediate buffers")
    assert source.index("query_snapshot.copy_(q)", capture) > capture
    assert source.index("seq_lens_snapshot.copy_(seqused_k)", capture) > capture


def test_published_artifact_is_generated_and_sanitized() -> None:
    payload_path = ARTIFACT / "audit.json"
    raw = payload_path.read_bytes()
    published = json.loads(raw)
    assert published == audit.build_audit()
    assert published["status"] == "STOP_RECURRENCE_BLOCKS_ATTENTION_ONLY_FUSION"
    assert published["implementation"] == {
        "runtime_kernel_changed": False,
        "launcher_changed": False,
        "production_credentials_changed": False,
        "default_behavior_changed": False,
        "stop_reason": (
            "No lossless attention-only fusion or defensible shared-KV/index "
            "traffic reduction remains after the recurrent barriers."
        ),
    }
    assert published["sanitization"] == {
        "absolute_host_paths": False,
        "request_payloads": False,
        "environment_dump": False,
        "runtime_logs": False,
    }
    assert b"/home/" not in raw
    assert b"/workspace/" not in raw
    assert b"instance_id" not in raw

    sums = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        sums[relative] = digest
    for name in ("README.md", "audit.json"):
        relative = (
            "results/fr13_fixed32_dfwd_attention_batch_audit_20260803/" + name
        )
        assert sums[relative] == hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest()


def test_audit_has_no_gpu_or_timing_claim() -> None:
    payload = audit.build_audit()
    scope = payload["scope"]
    assert scope["gpu_used"] is False
    assert scope["docker_used"] is False
    assert scope["runtime_used"] is False
    assert scope["synthetic_timing_used"] is False
    assert payload["source_evidence"]["old_attribution_is_current_speed_claim"] is False
    assert payload["next_safe_reduction_review"]["physical_hbm_saving_proven"] is False


def test_source_manifest_is_relative_and_byte_bound() -> None:
    manifest = audit.source_manifest()
    assert set(manifest) == {
        "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
        "scripts/fr13_fixed32_topology.py",
        "scripts/fr13_dfwd_attention_batch_audit.py",
    }
    for relative, record in manifest.items():
        assert not relative.startswith("/")
        assert record["sha256"] == hashlib.sha256(
            (ROOT / relative).read_bytes()
        ).hexdigest()


def test_static_ledger_rejects_nonpositive_sequence_length() -> None:
    for value in (0, -1):
        with pytest.raises(ValueError, match="must be positive"):
            audit.static_ledger(value)
