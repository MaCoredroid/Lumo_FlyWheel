#!/usr/bin/env python3
"""Install reviewed packed-CFWD definitions without changing TAW source bytes."""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import ModuleType
from typing import Any


BASE_SOURCE_SHA256 = "088454e0605c5d41aee7b385c6d0ff66e6a7ddb999a9697258762d0aac9fe166"
CANDIDATE = "fixed32_cfwd_logit_direct_packed_physical_slots_v3"
CANDIDATE_SCHEMA = "fr13.fixed32.cfwd_logit_direct_packed_physical_slots.v3"
CANDIDATE_SOURCE_SHA256 = "a7a7b6582cdc11e930916f5e65583195fd31a3b664e8f567bb33a24ea1a64ee0"
INTEGRATION_SOURCE_SCHEMA = "fr13.fixed32.cfwd_logit_direct.integration_source.v2"
INTEGRATION_SOURCE_SHA256 = "421465c6c04de8c26e3ea724a7d2f0d3f00fe50b4fdc9f57c35e71e71212297b"
CHANGED_FUNCTIONS = ('_fr13_cfwd_logit_direct_state', 'fr13_fixed32_cfwd_logit_direct_capture_begin', 'fr13_fixed32_cfwd_logit_direct_capture_end', '_fr13_cfwd_logit_direct_walk_cuda', '_fr13_cfwd_logit_direct_compare', 'fr13_fixed32_cfwd_logit_direct_warm_execute')
CHANGED_KERNELS = ('_fr13_fixed32_taw_packed_physical_slot_commit_kernel', '_fr13_cfwd_logit_direct_compare_kernel')


if False:  # Parsed and installed into the credential-bound base module.
    @triton.jit
    def _fr13_fixed32_taw_packed_physical_slot_commit_kernel(
        self_token,
        event,
        bonus_token,
        output_tokens,
        output_lens,
        accepted_path_rows,
        accepted_lens,
        last_row,
        PHYSICAL_DRAFTS: tl.constexpr,
        PHYSICAL_ROWS: tl.constexpr,
        WALK_CAP: tl.constexpr,
        OUTPUT_CAPACITY: tl.constexpr,
        PATH_CAPACITY: tl.constexpr,
    ):
        """Walk fixed physical slots from packed emitted-token events."""
        request = tl.program_id(0)
        output_columns = tl.arange(0, OUTPUT_CAPACITY)
        path_columns = tl.arange(0, PATH_CAPACITY)
        tl.store(
            output_tokens + request * OUTPUT_CAPACITY + output_columns,
            -1,
        )
        tl.store(
            accepted_path_rows + request * PATH_CAPACITY + path_columns,
            0,
        )

        current = -1
        alive = True
        output_len = 0
        path_len = 0
        final_row = 0
        for _level in tl.static_range(0, WALK_CAP):
            parent_slot = tl.maximum(
                0,
                tl.minimum(current + 1, PHYSICAL_ROWS - 1),
            )
            packed_event = tl.load(
                event + request * PHYSICAL_ROWS + parent_slot
            ).to(tl.int64)
            has_kids = alive & ((packed_event & 0x800000) != 0)
            leaf = alive & ((packed_event & 0x800000) == 0)
            current_valid = (current >= 0) & (current < PHYSICAL_DRAFTS)

            safe_current = tl.maximum(
                0,
                tl.minimum(current, PHYSICAL_DRAFTS - 1),
            )
            sampled_self = tl.load(
                self_token + request * PHYSICAL_DRAFTS + safe_current
            ).to(tl.int64)
            sampled_bonus = tl.load(bonus_token + request).to(tl.int64)
            leaf_token = tl.where(current_valid, sampled_self, sampled_bonus)
            tl.store(
                output_tokens + request * OUTPUT_CAPACITY + output_len,
                leaf_token,
                mask=leaf,
            )

            emitted_token = packed_event & 0x3FFFF
            tl.store(
                output_tokens + request * OUTPUT_CAPACITY + output_len,
                emitted_token,
                mask=has_kids,
            )
            output_len += leaf.to(tl.int64) + has_kids.to(tl.int64)

            accepted_row = (packed_event >> 18) & 0x1F
            is_accepted = has_kids & (accepted_row != 0)
            accepted_node = accepted_row - 1
            tl.store(
                accepted_path_rows + request * PATH_CAPACITY + path_len,
                accepted_row,
                mask=is_accepted,
            )
            path_len += is_accepted.to(tl.int64)
            current = tl.where(is_accepted, accepted_node, current)
            alive = (alive & (~leaf)) & is_accepted
            final_row = tl.where(is_accepted, accepted_row, final_row)

        tl.store(output_lens + request, output_len)
        tl.store(accepted_lens + request, path_len)
        tl.store(last_row + request, final_row)


    @triton.jit
    def _fr13_cfwd_logit_direct_compare_kernel(
        count_enable,
        compared_events,
        decision_mismatches,
        walk_mismatches,
        self_source_indices,
        target_parent_slots,
        child_table,
        child_counts,
        ref_self_token,
        cand_self_token,
        ref_source,
        ref_selected_token,
        ref_rejected_token,
        ref_accepted,
        cand_event,
        ref_output_tokens,
        cand_output_tokens,
        ref_output_lens,
        cand_output_lens,
        ref_accepted_path_rows,
        cand_accepted_path_rows,
        ref_accepted_lens,
        cand_accepted_lens,
        ref_last_row,
        cand_last_row,
        TARGET_ROWS: tl.constexpr,
        PHYSICAL_ROWS: tl.constexpr,
        SELF_N: tl.constexpr,
        TARGET_N: tl.constexpr,
        OUTPUT_N: tl.constexpr,
        BATCH_N: tl.constexpr,
        PATH_N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Count exact decision and walk mismatches in one captured program."""
        offsets = tl.arange(0, BLOCK)
        enabled = tl.load(count_enable) != 0

        self_mask = enabled & (offsets < SELF_N)
        target_mask = enabled & (offsets < TARGET_N)
        output_mask = enabled & (offsets < OUTPUT_N)
        batch_mask = enabled & (offsets < BATCH_N)
        path_mask = enabled & (offsets < PATH_N)
        self_source = tl.load(
            self_source_indices + offsets,
            mask=self_mask,
            other=0,
        ).to(tl.int64)
        target_request = offsets // TARGET_ROWS
        target_local = offsets - target_request * TARGET_ROWS
        target_parent = tl.load(
            target_parent_slots + target_local,
            mask=target_mask,
            other=0,
        ).to(tl.int64)
        physical_target_offset = target_request * PHYSICAL_ROWS + target_parent
        reference_source = tl.load(
            ref_source + offsets,
            mask=target_mask,
            other=0,
        ).to(tl.int64)
        reference_selected = tl.load(
            ref_selected_token + offsets,
            mask=target_mask,
            other=0,
        ).to(tl.int64)
        reference_rejected = tl.load(
            ref_rejected_token + offsets,
            mask=target_mask,
            other=0,
        ).to(tl.int64)
        reference_accepted = tl.load(
            ref_accepted + offsets,
            mask=target_mask,
            other=0,
        ) != 0
        reference_child_count = tl.load(
            child_counts + physical_target_offset,
            mask=target_mask,
            other=0,
        ).to(tl.int64)
        reference_accepted_node = tl.load(
            child_table
            + target_request * PHYSICAL_ROWS * 3
            + target_parent * 3
            + reference_source,
            mask=target_mask & reference_accepted,
            other=-1,
        ).to(tl.int64)
        reference_emitted = tl.where(
            reference_accepted,
            reference_selected,
            reference_rejected,
        )
        reference_accepted_row = tl.where(
            reference_accepted,
            reference_accepted_node + 1,
            0,
        )
        reference_event = tl.where(
            reference_child_count > 0,
            reference_emitted | (reference_accepted_row << 18) | 0x800000,
            0,
        )
        candidate_event = tl.load(
            cand_event + physical_target_offset,
            mask=target_mask,
            other=0,
        ).to(tl.int64)

        self_bad = tl.sum(
            tl.where(
                self_mask,
                tl.load(ref_self_token + offsets, mask=self_mask, other=0)
                != tl.load(
                    cand_self_token + self_source,
                    mask=self_mask,
                    other=0,
                ),
                0,
            ),
            axis=0,
        )
        event_bad = tl.sum(
            tl.where(
                target_mask,
                reference_event != candidate_event,
                0,
            ),
            axis=0,
        )
        selected_bad = tl.sum(
            tl.where(
                target_mask,
                (reference_event & 0x3FFFF) != (candidate_event & 0x3FFFF),
                0,
            ),
            axis=0,
        )
        rejected_bad = tl.sum(
            tl.where(
                target_mask,
                (reference_event & 0x800000) != (candidate_event & 0x800000),
                0,
            ),
            axis=0,
        )
        accepted_bad = tl.sum(
            tl.where(
                target_mask,
                ((reference_event >> 18) & 0x1F)
                != ((candidate_event >> 18) & 0x1F),
                0,
            ),
            axis=0,
        )
        output_bad = tl.sum(
            tl.where(
                output_mask,
                tl.load(ref_output_tokens + offsets, mask=output_mask, other=0)
                != tl.load(cand_output_tokens + offsets, mask=output_mask, other=0),
                0,
            ),
            axis=0,
        )
        output_lens_bad = tl.sum(
            tl.where(
                batch_mask,
                tl.load(ref_output_lens + offsets, mask=batch_mask, other=0)
                != tl.load(cand_output_lens + offsets, mask=batch_mask, other=0),
                0,
            ),
            axis=0,
        )
        path_bad = tl.sum(
            tl.where(
                path_mask,
                tl.load(ref_accepted_path_rows + offsets, mask=path_mask, other=0)
                != tl.load(cand_accepted_path_rows + offsets, mask=path_mask, other=0),
                0,
            ),
            axis=0,
        )
        accepted_lens_bad = tl.sum(
            tl.where(
                batch_mask,
                tl.load(ref_accepted_lens + offsets, mask=batch_mask, other=0)
                != tl.load(cand_accepted_lens + offsets, mask=batch_mask, other=0),
                0,
            ),
            axis=0,
        )
        last_row_bad = tl.sum(
            tl.where(
                batch_mask,
                tl.load(ref_last_row + offsets, mask=batch_mask, other=0)
                != tl.load(cand_last_row + offsets, mask=batch_mask, other=0),
                0,
            ),
            axis=0,
        )

        first = offsets == 0
        event_delta = tl.where(first, enabled.to(tl.int64), 0)
        self_delta = tl.where(first, self_bad.to(tl.int64), 0)
        source_delta = tl.where(first, event_bad.to(tl.int64), 0)
        selected_delta = tl.where(first, selected_bad.to(tl.int64), 0)
        rejected_delta = tl.where(first, rejected_bad.to(tl.int64), 0)
        accepted_delta = tl.where(first, accepted_bad.to(tl.int64), 0)
        output_delta = tl.where(first, output_bad.to(tl.int64), 0)
        output_lens_delta = tl.where(first, output_lens_bad.to(tl.int64), 0)
        path_delta = tl.where(first, path_bad.to(tl.int64), 0)
        accepted_lens_delta = tl.where(
            first, accepted_lens_bad.to(tl.int64), 0
        )
        last_row_delta = tl.where(first, last_row_bad.to(tl.int64), 0)
        tl.atomic_add(compared_events + offsets, event_delta, mask=first)
        tl.atomic_add(
            decision_mismatches + offsets,
            self_delta,
            mask=first,
        )
        tl.atomic_add(
            decision_mismatches + 1 + offsets,
            source_delta,
            mask=first,
        )
        tl.atomic_add(
            decision_mismatches + 2 + offsets,
            selected_delta,
            mask=first,
        )
        tl.atomic_add(
            decision_mismatches + 3 + offsets,
            rejected_delta,
            mask=first,
        )
        tl.atomic_add(
            decision_mismatches + 4 + offsets,
            accepted_delta,
            mask=first,
        )
        tl.atomic_add(walk_mismatches + offsets, output_delta, mask=first)
        tl.atomic_add(
            walk_mismatches + 1 + offsets,
            output_lens_delta,
            mask=first,
        )
        tl.atomic_add(
            walk_mismatches + 2 + offsets,
            path_delta,
            mask=first,
        )
        tl.atomic_add(
            walk_mismatches + 3 + offsets,
            accepted_lens_delta,
            mask=first,
        )
        tl.atomic_add(
            walk_mismatches + 4 + offsets,
            last_row_delta,
            mask=first,
        )


    def _fr13_cfwd_logit_direct_state(
        entry: dict[str, Any],
        *,
        graph_id: int | None,
    ) -> dict[str, Any]:
        candidate = _fr13_cfwd_logit_direct_load()
        mode = str(entry["mode"])
        batch = int(entry["batch_size"])
        device = entry["child_table"].device
        workspace = candidate.allocate_workspace(device=device, batch_size=batch)
        metadata_binding = candidate.prepare_metadata_binding(
            self_source_indices=entry["native_self_source_indices"],
            target_source_indices=entry["native_target_source_indices"],
            child_table=entry["child_table"],
            child_counts=entry["child_counts"],
            self_uniform_levels=entry["all_parent_self_uniform_levels"],
            target_parent_slots=entry["all_parent_target_parent_slots"],
            target_uniform_levels=entry["all_parent_target_uniform_levels"],
            batch_size=batch,
            mode=mode,
        )
        return {
            "graph_id": graph_id,
            "mode": mode,
            "batch_size": batch,
            "device": device,
            "workspace": workspace,
            "metadata_binding": metadata_binding,
            "walk_entry": _fr13_cfwd_logit_direct_walk_entry(entry),
            "self_source_indices": entry["native_self_source_indices"],
            "target_parent_slots": entry["all_parent_target_parent_slots"],
            "child_table": entry["child_table"],
            "child_counts": entry["child_counts"],
            "count_enable": torch.zeros((1,), dtype=torch.int32, device=device),
            "compared_events": torch.zeros((1,), dtype=torch.int64, device=device),
            "decision_mismatches": torch.zeros(
                (5,), dtype=torch.int64, device=device
            ),
            "walk_mismatches": torch.zeros((5,), dtype=torch.int64, device=device),
            "bound_calls": 0,
        }


    def fr13_fixed32_cfwd_logit_direct_capture_begin(
        graph_id: int,
        *,
        mode: str,
        batch_size: int,
    ) -> None:
        """Bind prewarmed committer state to the target graph identity."""
        global _FR13_CFWD_LOGIT_DIRECT_CAPTURE
        if int(batch_size) not in (1, 4):
            return
        selector = _fr13_cfwd_logit_direct_selector(
            mode=mode, batch_size=int(batch_size)
        )
        if selector == "reference":
            return
        if _FR13_CFWD_LOGIT_DIRECT_CAPTURE is not None:
            raise RuntimeError("FR13 CFWD logit-direct captures overlapped")
        identity = int(graph_id)
        if identity <= 0 or identity in _FR13_CFWD_LOGIT_DIRECT_GRAPHS:
            raise RuntimeError("FR13 CFWD logit-direct graph identity was reused")
        _, valid_mask = _fr13_fixed32_runtime_contract(mode)
        entry = _fr13_cfwd_logit_direct_entry(mode, int(batch_size))
        key = fr13_fixed32_taw_cache_key(
            mode,
            valid_mask,
            int(batch_size),
            entry["child_table"].device,
        )
        state = _FR13_CFWD_LOGIT_DIRECT_WARM.get(key)
        if (
            not isinstance(state, dict)
            or state.get("graph_id") is not None
            or state.get("mode") != mode
            or state.get("batch_size") != int(batch_size)
            or state.get("device") != entry["child_table"].device
            or state.get("bound_calls") != 0
        ):
            raise RuntimeError("FR13 CFWD logit-direct prewarmed state drift")
        state["graph_id"] = identity
        _FR13_CFWD_LOGIT_DIRECT_GRAPHS[identity] = state
        _FR13_CFWD_LOGIT_DIRECT_CAPTURE = state


    def fr13_fixed32_cfwd_logit_direct_capture_end(
        graph_id: int,
        *,
        mode: str,
        batch_size: int,
    ) -> None:
        """Verify target capture excludes CFWD, then bind its external call site."""
        global _FR13_CFWD_LOGIT_DIRECT_CAPTURE
        if int(batch_size) not in (1, 4):
            return
        selector = _fr13_cfwd_logit_direct_selector(
            mode=mode, batch_size=int(batch_size)
        )
        if selector == "reference":
            return
        state = _FR13_CFWD_LOGIT_DIRECT_CAPTURE
        if (
            not isinstance(state, dict)
            or state.get("graph_id") != int(graph_id)
            or state.get("mode") != mode
            or state.get("batch_size") != int(batch_size)
            or state.get("bound_calls") != 0
        ):
            raise RuntimeError("FR13 CFWD logit-direct capture binding drift")
        state["bound_calls"] = 1
        _FR13_CFWD_LOGIT_DIRECT_CAPTURE = None


    def _fr13_cfwd_logit_direct_walk_cuda(
        topology,
        entry: dict[str, Any],
        bonus_flat,
        decisions: tuple[Any, Any, Any, Any, Any],
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Walk physical-slot decisions without topology-map indirection."""
        if triton is None or tl is None:
            raise RuntimeError("FR13 CFWD logit-direct walk requires Triton")
        batch = int(entry["batch_size"])
        physical_drafts = int(topology.PHYSICAL_DRAFTS)
        physical_rows = int(topology.PHYSICAL_ROWS)
        expected = (
            (decisions[0], (batch, physical_drafts), torch.long),
            (decisions[1], (batch, physical_rows), torch.long),
        )
        if any(
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != bonus_flat.device
            or not value.is_contiguous()
            for value, shape, dtype in expected
        ):
            raise RuntimeError("FR13 CFWD logit-direct decision layout drift")
        _fr13_fixed32_taw_packed_physical_slot_commit_kernel[(batch,)](
            *decisions,
            bonus_flat,
            entry["output_tokens"],
            entry["output_lens"],
            entry["accepted_path_rows"],
            entry["accepted_lens"],
            entry["last_row"],
            PHYSICAL_DRAFTS=physical_drafts,
            PHYSICAL_ROWS=physical_rows,
            WALK_CAP=int(topology.WALK_CAP),
            OUTPUT_CAPACITY=int(topology.OUTPUT_PUBLISH_CAPACITY),
            PATH_CAPACITY=int(topology.ACCEPTED_PATH_CAPACITY),
            num_warps=1,
        )
        return (
            entry["output_tokens"],
            entry["output_lens"],
            entry["accepted_path_rows"],
            entry["accepted_lens"],
            entry["last_row"],
        )


    def _fr13_cfwd_logit_direct_compare(
        state: dict[str, Any],
        reference_decisions: tuple[Any, Any, Any, Any, Any],
        candidate_decisions: tuple[Any, Any],
        reference_walk: tuple[Any, Any, Any, Any, Any],
        candidate_walk: tuple[Any, Any, Any, Any, Any],
    ) -> None:
        if triton is None or tl is None:
            raise RuntimeError("FR13 CFWD logit-direct comparator requires Triton")
        batch = int(state["batch_size"])
        _fr13_cfwd_logit_direct_compare_kernel[(1,)](
            state["count_enable"],
            state["compared_events"],
            state["decision_mismatches"],
            state["walk_mismatches"],
            state["self_source_indices"],
            state["target_parent_slots"],
            state["child_table"],
            state["child_counts"],
            reference_decisions[0],
            candidate_decisions[0],
            reference_decisions[1],
            reference_decisions[2],
            reference_decisions[3],
            reference_decisions[4],
            candidate_decisions[1],
            reference_walk[0],
            candidate_walk[0],
            reference_walk[1],
            candidate_walk[1],
            reference_walk[2],
            candidate_walk[2],
            reference_walk[3],
            candidate_walk[3],
            reference_walk[4],
            candidate_walk[4],
            TARGET_ROWS=17,
            PHYSICAL_ROWS=32,
            SELF_N=batch * 13,
            TARGET_N=batch * 17,
            OUTPUT_N=batch * 32,
            BATCH_N=batch,
            PATH_N=batch * 16,
            BLOCK=128,
            num_warps=4,
        )


    def fr13_fixed32_cfwd_logit_direct_warm_execute(
        device,
        *,
        mode: str,
        valid_mask: int,
        max_batch_size: int,
        vocab_size: int,
    ) -> dict[str, Any]:
        """Compile every candidate launch outside capture without serving output."""
        capacity = int(max_batch_size)
        selector = _fr13_cfwd_logit_direct_selector(
            mode=mode, batch_size=capacity
        )
        if selector == "reference":
            return {"ready": False, "requested": False, "batches": ()}
        vocab = int(vocab_size)
        if capacity not in (1, 4) or vocab != 248_320:
            raise RuntimeError(
                "FR13 CFWD logit-direct warm requires exact B1/B4 and vocab 248320"
            )
        normalized_device = torch.device(device)
        if (
            normalized_device.type != "cuda"
            or torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError("FR13 CFWD logit-direct warm requires uncaptured CUDA")
        topology, runtime_mask = _fr13_fixed32_runtime_contract(mode)
        if int(valid_mask) != int(runtime_mask):
            raise RuntimeError("FR13 CFWD logit-direct warm valid-mask drift")
        batches = (1,) if capacity == 1 else (1, 4)
        max_rows = max(batches) * int(topology.PHYSICAL_DRAFTS)
        logits = torch.zeros(
            (max_rows, vocab), dtype=torch.float32, device=normalized_device
        )
        for batch in batches:
            key = fr13_fixed32_taw_cache_key(
                mode, runtime_mask, batch, normalized_device
            )
            entry = _FR13_FIXED32_TAW_CACHE.get(key)
            if entry is None:
                raise RuntimeError("FR13 CFWD logit-direct warm cache miss")
            state = _FR13_CFWD_LOGIT_DIRECT_WARM.get(key)
            if state is None:
                state = _fr13_cfwd_logit_direct_state(entry, graph_id=None)
                _FR13_CFWD_LOGIT_DIRECT_WARM[key] = state
            drafts = (
                torch.arange(
                    int(topology.PHYSICAL_DRAFTS),
                    dtype=torch.long,
                    device=normalized_device,
                )
                .remainder(vocab)
                .repeat(batch, 1)
            )
            uniforms = torch.full(
                (batch, int(topology.WALK_CAP), 3),
                0.1,
                dtype=torch.float32,
                device=normalized_device,
            )
            bonus = torch.zeros(batch, dtype=torch.long, device=normalized_device)
            candidate = _fr13_cfwd_logit_direct_load()
            decisions = candidate.launch_logit_direct_fixed32(
                self_logits=logits[: batch * int(topology.PHYSICAL_DRAFTS)],
                target_logits=logits[: batch * int(topology.PHYSICAL_DRAFTS)],
                self_source_indices=entry["native_self_source_indices"],
                target_source_indices=entry["native_target_source_indices"],
                drafts=drafts,
                child_table=entry["child_table"],
                child_counts=entry["child_counts"],
                self_uniform_levels=entry["all_parent_self_uniform_levels"],
                target_parent_slots=entry["all_parent_target_parent_slots"],
                target_uniform_levels=entry["all_parent_target_uniform_levels"],
                uniforms=uniforms,
                workspace=state["workspace"],
                batch_size=batch,
                mode=mode,
                metadata_binding=state["metadata_binding"],
            )
            walk = _fr13_cfwd_logit_direct_walk_cuda(
                topology, state["walk_entry"], bonus, decisions
            )
            reference_warm = (
                torch.zeros((batch, 13), dtype=torch.long, device=normalized_device),
                torch.zeros((batch, 17), dtype=torch.long, device=normalized_device),
                torch.zeros((batch, 17), dtype=torch.long, device=normalized_device),
                torch.zeros((batch, 17), dtype=torch.long, device=normalized_device),
                torch.zeros((batch, 17), dtype=torch.bool, device=normalized_device),
            )
            _fr13_cfwd_logit_direct_compare(
                state, reference_warm, decisions, walk, walk
            )
        return {
            "ready": True,
            "requested": True,
            "selector": selector,
            "classification": "unmeasured_boot",
            "batches": batches,
            "candidate": _FR13_CFWD_LOGIT_DIRECT_CANDIDATE,
            "source_sha256": _FR13_CFWD_LOGIT_DIRECT_SOURCE_SHA256,
        }


    def _fr13_cfwd_logit_direct_integration_source_contract() -> dict[str, str]:
        """Bind the unchanged base plus the explicit CFWD-only overlay."""
        global _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_CACHE
        if _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_CACHE is not None:
            return dict(_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_CACHE)

        try:
            base_tree = ast.parse(Path(__file__).resolve().read_text(encoding="utf-8"))
            overlay_tree = ast.parse(
                Path(_FR13_CFWD_LOGIT_DIRECT_OVERLAY_PATH)
                .resolve()
                .read_text(encoding="utf-8")
            )
        except (OSError, SyntaxError) as error:
            raise RuntimeError(
                "FR13 CFWD logit-direct cannot inspect composed integration source"
            ) from error
        expected_functions = set(
            _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_FUNCTIONS
        )
        expected_kernels = set(
            _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_KERNEL_SOURCE_FUNCTIONS
        )
        expected = expected_functions | expected_kernels
        overlay_names = set(_FR13_CFWD_LOGIT_DIRECT_OVERLAY_DEFINITIONS)
        definitions: dict[str, list[Any]] = {name: [] for name in expected}
        for tree, use_names in (
            (base_tree, expected - overlay_names),
            (overlay_tree, expected & overlay_names),
        ):
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in use_names
                ):
                    definitions[node.name].append(node)
        if any(len(nodes) != 1 for nodes in definitions.values()):
            raise RuntimeError(
                "FR13 CFWD composed integration source is incomplete or ambiguous"
            )

        normalized = {
            name: ast.dump(
                definitions[name][0],
                annotate_fields=True,
                include_attributes=False,
            )
            for name in sorted(expected)
        }
        canonical = json.dumps(
            {
                "schema": _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SCHEMA,
                "candidate": {
                    "name": _FR13_CFWD_LOGIT_DIRECT_CANDIDATE,
                    "schema": _FR13_CFWD_LOGIT_DIRECT_SCHEMA,
                    "source_sha256": _FR13_CFWD_LOGIT_DIRECT_SOURCE_SHA256,
                },
                "geometry": _FR13_FIXED32_TAW_GEOMETRY,
                "functions": {
                    name: normalized[name] for name in sorted(expected_functions)
                },
                "kernels": {
                    name: normalized[name] for name in sorted(expected_kernels)
                },
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
        if digest != _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SHA256:
            raise RuntimeError(
                "FR13 CFWD composed integration source identity drifted: " + digest
            )
        contract = {
            "integration_source_schema": (
                _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SCHEMA
            ),
            "integration_source_sha256": digest,
        }
        _FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_CACHE = dict(contract)
        return contract


def _definitions(tree: ast.AST) -> dict[str, ast.AST]:
    wanted = {
        *CHANGED_FUNCTIONS,
        *CHANGED_KERNELS,
        "_fr13_cfwd_logit_direct_integration_source_contract",
    }
    found: dict[str, list[ast.AST]] = {name: [] for name in wanted}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in found:
                found[node.name].append(node)
    if any(len(nodes) != 1 for nodes in found.values()):
        raise RuntimeError("packed CFWD overlay definition set drifted")
    return {name: nodes[0] for name, nodes in found.items()}


def install(module: ModuleType) -> dict[str, str]:
    """Install only reviewed CFWD names into an unchanged TAW base module."""
    overlay_path = Path(__file__).resolve()
    base_path = Path(module.__file__).resolve()
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != BASE_SOURCE_SHA256:
        raise RuntimeError("packed CFWD overlay base source identity drifted")
    topology = module._fr13_fixed32_topology()
    taw_before = module._fr13_fixed32_taw_source_contract(topology, batch_size=1)
    taw_functions = {
        name: getattr(module, name)
        for name in module._FR13_FIXED32_TAW_SOURCE_FUNCTIONS
    }

    tree = ast.parse(overlay_path.read_text(encoding="utf-8"))
    definitions = _definitions(tree)
    namespace = module.__dict__
    namespace.update(
        {
            "_FR13_CFWD_LOGIT_DIRECT_CANDIDATE": CANDIDATE,
            "_FR13_CFWD_LOGIT_DIRECT_SCHEMA": CANDIDATE_SCHEMA,
            "_FR13_CFWD_LOGIT_DIRECT_SOURCE_SHA256": CANDIDATE_SOURCE_SHA256,
            "_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SCHEMA": (
                INTEGRATION_SOURCE_SCHEMA
            ),
            "_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SHA256": (
                INTEGRATION_SOURCE_SHA256
            ),
            "_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_KERNEL_SOURCE_FUNCTIONS": (
                CHANGED_KERNELS
            ),
            "_FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_CACHE": None,
            "_FR13_CFWD_LOGIT_DIRECT_MODULE": None,
            "_FR13_CFWD_LOGIT_DIRECT_OVERLAY_PATH": str(overlay_path),
            "_FR13_CFWD_LOGIT_DIRECT_OVERLAY_DEFINITIONS": (
                *CHANGED_FUNCTIONS,
                *CHANGED_KERNELS,
            ),
        }
    )
    function_nodes = [
        copy.deepcopy(definitions[name])
        for name in (
            *CHANGED_FUNCTIONS,
            "_fr13_cfwd_logit_direct_integration_source_contract",
        )
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=function_nodes, type_ignores=[])),
            str(overlay_path),
            "exec",
        ),
        namespace,
    )
    if module.triton is not None:
        kernel_nodes = [copy.deepcopy(definitions[name]) for name in CHANGED_KERNELS]
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=kernel_nodes, type_ignores=[])
                ),
                str(overlay_path),
                "exec",
            ),
            namespace,
        )

    taw_after = module._fr13_fixed32_taw_source_contract(topology, batch_size=1)
    if taw_after != taw_before or any(
        getattr(module, name) is not value for name, value in taw_functions.items()
    ):
        raise RuntimeError("packed CFWD overlay changed the TAW source contract")
    contract = module._fr13_cfwd_logit_direct_integration_source_contract()
    if contract != {
        "integration_source_schema": INTEGRATION_SOURCE_SCHEMA,
        "integration_source_sha256": INTEGRATION_SOURCE_SHA256,
    }:
        raise RuntimeError("packed CFWD overlay integration contract drifted")
    return contract
