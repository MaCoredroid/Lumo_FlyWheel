# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import json
import os
_FR13_FIXED32_MODE = ''

def _fr13_fixed32_tsr_nonprefix_update(
    token_indices_to_sample,
    query_start_loc,
    compact_leaf_rows,
    spec_batch_indices,
):
    """Write compact accepted leaves into their exact full proposer rows."""
    compact_batch = len(spec_batch_indices)
    if (
        not isinstance(spec_batch_indices, tuple)
        or compact_batch < 1
        or any(type(index) is not int for index in spec_batch_indices)
        or tuple(sorted(set(spec_batch_indices))) != spec_batch_indices
        or token_indices_to_sample.ndim != 1
        or query_start_loc.ndim != 1
        or compact_leaf_rows.ndim != 1
        or int(compact_leaf_rows.numel()) != compact_batch
        or any(
            not 0 <= index < int(token_indices_to_sample.numel())
            or index + 1 >= int(query_start_loc.numel())
            for index in spec_batch_indices
        )
        or query_start_loc.device != token_indices_to_sample.device
        or compact_leaf_rows.device != token_indices_to_sample.device
    ):
        raise RuntimeError("FR13 fixed32 TSR nonprefix geometry drift")
    index_tensor = torch.tensor(
        spec_batch_indices,
        dtype=torch.long,
        device=token_indices_to_sample.device,
    )
    base_rows = query_start_loc.index_select(0, index_tensor)
    mapped_rows = (
        base_rows.to(token_indices_to_sample.dtype)
        + compact_leaf_rows.to(token_indices_to_sample.dtype)
    )
    result = token_indices_to_sample.clone()
    result.index_copy_(0, index_tensor, mapped_rows)
    return result
from importlib.util import find_spec
from typing import Any, cast

import numpy as np
import torch
from e8_head_gate import primary_logits as _e8_primary, legacy_logits as _e8_legacy, finish_proposal as _e8_finish
_E8_ARM = 'on'
_E8_QUALIFY = False
import torch.nn as nn

from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    replace,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.tree_attn import (
    TreeAttentionMetadata,
    TreeAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    compute_new_slot_mapping,
    copy_and_expand_eagle_inputs_kernel,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
    eagle_step_update_slot_mapping_and_metadata,
    extend_all_queries_by_N,
    next_power_of_2,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


# FR13_DFWD_SPLIT: per-level 3-way drafter timing (default OFF; env FR13_DFWD_SPLIT=1)
class _Fr13DfwdSplit:
    def __init__(self):
        import os as _os
        self.on = _os.environ.get("FR13_DFWD_SPLIT", "0") == "1"
        if not self.on:
            # env-first-then-flag-file: bare FR13_* env may not reach the
            # EngineCore worker (the proven FR13_GDN_SUBOP_MAB pattern); the
            # patcher main() writes the flag inside the container at boot.
            try:
                with open("/logs/fr13_dfwd_split.flag") as _fh:
                    self.on = _fh.read().strip() == "1"
            except Exception:  # noqa: BLE001
                pass
        if self.on:
            try:
                from vllm.logger import init_logger as _il
                _il("vllm.fr13_dfwd_split").info(
                    "FR13_DFWD_SPLIT ENGAGED (worker pid=%s)",
                    __import__("os").getpid(),
                )
            except Exception:  # noqa: BLE001
                pass
        self.pairs = {"model": [], "sample": [], "lmhead": []}
        self.pathmap = set()
        self.done = False
        if self.on:
            import atexit as _ax
            _ax.register(self.dump)

    def begin(self, k):
        if not self.on:
            return None
        import torch as _t
        ev = _t.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def end(self, k, start_ev):
        if start_ev is None:
            return
        import torch as _t
        ev = _t.cuda.Event(enable_timing=True)
        ev.record()
        self.pairs[k].append((start_ev, ev))
        # periodic dump: teardown is docker rm -f (SIGKILL, no atexit); mirror
        # the span-timer's survival pattern. Level-end only, every 25 levels.
        if k == "model" and len(self.pairs["model"]) % 25 == 0:
            self.done = False
            self.dump()
            self.done = False

    def _defer(self, why):
        # Record the deferral WITHOUT a traceback. During a fatal mid-capture
        # teardown this is the difference between a log that names the real
        # defect and one that points at the instrument -- see Arm G,
        # output/fr14_promoab_Giso_20260818T074147Z, where a secondary
        # cudaErrorStreamCaptureUnsupported here outranked the primary refusal
        # in the diagnosis.
        try:
            with open("/logs/fr13_dfwd_split.err", "a") as fh:
                fh.write("DEFERRED: " + str(why) + chr(10))
        except Exception:  # noqa: BLE001
            pass

    def dump(self):
        if not self.on:
            return
        self.done = True
        import json as _j, os as _os, torch as _t
        try:
            # NEVER synchronize inside a capture window. cudaDeviceSynchronize is
            # cudaErrorStreamCaptureUnsupported there, and worse it can
            # INVALIDATE an in-flight capture: an instrument must not be able to
            # break the thing it is measuring. This window is reachable in normal
            # operation, not just on the failure path -- the periodic dump fires
            # from inside the drafter's instrumented model span, and the drafter
            # graph capture executes exactly that span.
            if _t.cuda.is_current_stream_capturing():
                self.done = False
                self._defer("current stream is capturing")
                return
            _t.cuda.synchronize()
            out = {"schema": "fr13.dfwd_split.v1"}
            for k, ps in self.pairs.items():
                tot = sum(a.elapsed_time(b) for a, b in ps) / 1000.0
                out[k + "_seconds"] = tot
                out["n_" + k] = len(ps)
            out["other_seconds"] = max(
                0.0, out.get("level_seconds", 0.0)
                - out.get("model_seconds", 0.0) - out.get("head_seconds", 0.0)
            )
            p = _os.environ.get(
                "FR13_DFWD_SPLIT_JSON", "/logs/fr13_dfwd_split.json"
            ) + "." + str(_os.getpid())
            with open(p, "w") as fh:
                _j.dump(out, fh, indent=1)
        except Exception as _e:  # noqa: BLE001
            # A capture on ANOTHER stream still fails a device-wide sync, and
            # is_current_stream_capturing() cannot see it. Treat that as a
            # deferral too, so the instrument never presents a capture-scoping
            # artifact as its own failure.
            if "stream is capturing" in str(_e):
                self.done = False
                self._defer("a stream is capturing (not the current one)")
                return
            try:
                import traceback as _tb
                with open("/logs/fr13_dfwd_split.err", "a") as fh:
                    fh.write(repr(_e) + chr(10) + _tb.format_exc() + chr(10))
            except Exception:
                pass


_FR13_DFWD_SPLIT = _Fr13DfwdSplit()


class SpecDecodeBaseProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool,
        runner=None,
    ):
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.pass_hidden_states_to_model = pass_hidden_states_to_model

        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

        # Unifying eagle, draft model, and parallel drafting support.
        # DFlash always uses parallel drafting (all tokens in one pass),
        # but has an additional slot for the next_token_id (does not shift like EAGLE)
        self.parallel_drafting: bool = self.speculative_config.parallel_drafting
        self.extra_slots_per_request = (
            1 if not self.parallel_drafting else self.num_speculative_tokens
        )
        self.net_num_new_slots_per_request = self.extra_slots_per_request - (
            1 if (self.pass_hidden_states_to_model and self.method != "dflash") else 0
        )
        self.needs_extra_input_slots = self.net_num_new_slots_per_request > 0

        self.parallel_drafting_token_id: int = 0
        self.parallel_drafting_hidden_state_tensor: torch.Tensor | None = None
        if self.parallel_drafting:
            self._init_parallel_drafting_params()
        self.use_local_argmax_reduction: bool = (
            self.speculative_config.use_local_argmax_reduction
        )

        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)

        # Can be specialized by methods like DFlash to reduce the limit
        self.max_query_tokens = self.max_num_tokens
        self.max_positions = self.max_num_tokens

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )

        self.draft_attn_groups: list[AttentionGroup] = []
        self.kv_cache_gid: int = -1
        self.eagle3_use_aux_hidden_state: bool = (
            self._get_eagle3_use_aux_hidden_state_from_config()
        )

        self.compilation_config = self.vllm_config.compilation_config

        # Cudagraph dispatcher for PIECEWISE-only dispatching in eagle.
        # Keys are initialized later via initialize_cudagraph_keys() called from
        # gpu_model_runner._check_and_update_cudagraph_mode after
        # adjust_cudagraph_sizes_for_spec_decode is called.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        # Use draft model's M-RoPE setting, not target model's
        # Draft models may be text-only even if target is multimodal
        self.uses_mrope = self.draft_model_config.uses_mrope
        self.uses_xdrope_dim = self.vllm_config.model_config.uses_xdrope_dim
        self.draft_uses_xdrope_dim = self.draft_model_config.uses_xdrope_dim
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros(
                (3, self.max_positions + 1), dtype=torch.int64, device=device
            )
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions = torch.zeros(
                (self.uses_xdrope_dim, self.max_positions + 1),
                dtype=torch.int64,
                device=device,
            )
        else:
            # RoPE need (max_num_tokens,)
            self.positions = torch.zeros(
                self.max_positions,
                dtype=torch.int64,
                device=device,
            )
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # Will be set when we initialize the attention backend
        self.block_size: int = -1

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        max_num_slots_for_arange = max(self.max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32
        )

        if self.needs_extra_input_slots:
            self._raise_if_padded_drafter_batch_disabled()
            self._raise_if_multimodal()
            self._raise_if_mrope()

        self.is_rejected_token_mask: torch.Tensor | None = None
        self.is_masked_token_mask: torch.Tensor | None = None
        if self.needs_extra_input_slots:
            # For draft models and parallel drafting, we need to keep track of
            # which tokens are rejected to update the slot mapping with padding slots.
            self.is_rejected_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )
            # For parallel drafting, we also need to keep track of which tokens
            # are parallel-padding tokens used to sample at later positions.
            # We populate this tensor even when using draft models for simplicity.
            self.is_masked_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=device,
        )

        self.backup_next_token_ids = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )

        self._slot_mapping_buffer = torch.zeros(
            self.max_positions,
            dtype=torch.int64,
            device=device,
        )

        # Determine allowed attention backends once during initialization.
        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            from vllm.v1.attention.backends.mla.indexer import (
                DeepseekV32IndexerMetadata,
            )
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
                ROCMAiterMLASparseMetadata,
            )
            from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

            rocm_types = [
                TritonAttentionMetadata,
                RocmAttentionMetadata,
                ROCMAiterMLASparseMetadata,
                DeepseekV32IndexerMetadata,
            ]
            # ROCM_AITER_FA is an optional backend
            # We check is_enabled() here to avoid importing the backend module during
            # auto-discovery when VLLM_ROCM_USE_AITER=0, which would trigger aiter
            # import and JIT compilation warnings. Explicit backend selection via
            # attention_config still works because the backend module is loaded
            # directly when selected, not through this auto-discovery path.
            # Check if backend module exists to allow explicit selection
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)

            # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadata,
            )

            rocm_types.append(MLACommonMetadata)

            # FlexAttention backend support
            from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

            rocm_types.append(FlexAttentionMetadata)

            self.allowed_attn_types = tuple(rocm_types)

        # Parse the speculative token tree.
        spec_token_tree = None
        try:
            spec_env = os.environ.get("SPEC_CONFIG")
            if spec_env:
                spec_token_tree = json.loads(spec_env).get("speculative_token_tree")
        except Exception:
            spec_token_tree = None
        if spec_token_tree is None:
            spec_token_tree = self.speculative_config.speculative_token_tree
        assert spec_token_tree is not None
        self.tree_choices: list[tuple[int, ...]] = sorted(
            ast.literal_eval(spec_token_tree), key=lambda _p: (len(_p), _p)
        )
        tree_depth = len(self.tree_choices[-1])
        # Precompute per-level properties of the tree.
        num_drafts_per_level = [0] * tree_depth
        for node in self.tree_choices:
            num_drafts_per_level[len(node) - 1] += 1
        self.cu_drafts_per_level = [num_drafts_per_level[0]]
        self.child_drafts_per_level = [num_drafts_per_level[0]]
        for level in range(1, tree_depth):
            self.cu_drafts_per_level.append(
                self.cu_drafts_per_level[-1] + num_drafts_per_level[level]
            )
            self.child_drafts_per_level.append(
                num_drafts_per_level[level] // num_drafts_per_level[level - 1]
            )
        # Precompute draft position offsets in flattened tree.
        self.tree_draft_pos_offsets = torch.arange(
            1, len(self.tree_choices) + 1, device=device, dtype=torch.int32
        ).repeat(self.max_batch_size, 1)

    def _raise_if_padded_drafter_batch_disabled(self):
        if self.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting only "
                "supports padded drafter batch. Please unset "
                "disable_padded_drafter_batch in the speculative_config."
            )

    def _raise_if_multimodal(self):
        if self.supports_mm_inputs:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting "
                "does not support multimodal models yet"
            )

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting "
                "does not support M-RoPE yet"
            )

    def _init_parallel_drafting_params(self):
        # For parallel drafting, we need the token ID to use for masked slots
        # And for EAGLE + parallel drafting, we need the hidden state tensor to use
        # for those masked slots.

        model_hf_config = self.draft_model_config.hf_config
        # DFlash stores mask_token_id in dflash_config
        dflash_config = getattr(model_hf_config, "dflash_config", None)
        if dflash_config and "mask_token_id" in dflash_config:
            self.parallel_drafting_token_id = dflash_config["mask_token_id"]
        elif hasattr(model_hf_config, "pard_token"):
            self.parallel_drafting_token_id = model_hf_config.pard_token
        elif hasattr(model_hf_config, "ptd_token_id"):
            self.parallel_drafting_token_id = model_hf_config.ptd_token_id
        else:
            raise ValueError(
                "For parallel drafting, the draft model config must have "
                "`pard_token`, `ptd_token_id`, or "
                "`dflash_config.mask_token_id` specified in its config.json."
            )

        if self.pass_hidden_states_to_model:
            self.parallel_drafting_hidden_state_tensor = torch.empty(
                self.hidden_size, dtype=self.dtype, device=self.device
            )

    def _get_positions(self, num_tokens: int):
        if self.uses_mrope:
            return self.mrope_positions[:, :num_tokens]
        if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            return self.xdrope_positions[:, :num_tokens]
        return self.positions[:num_tokens]

    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        if self.uses_mrope:
            self.mrope_positions[:, :num_tokens] = positions
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[:, :num_tokens] = positions
        else:
            # Convert M-RoPE positions if target model uses M-RoPE
            # but draft doesn't, For text inputs, all M-RoPE
            # dimensions are identical
            if self.vllm_config.model_config.uses_mrope:
                positions = positions[0]
            self.positions[:num_tokens] = positions

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return slot_mapping dict for EAGLE layers.

        If slot_mapping is provided, copies it into the buffer first.
        """
        if slot_mapping is not None:
            num_actual = slot_mapping.shape[0]
            self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)
            if num_tokens > num_actual:
                self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)

        view = self._slot_mapping_buffer[:num_tokens]
        return {name: view for name in self._draft_attn_layer_names}

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize cudagraph dispatcher keys for eagle.

        Eagle only supports PIECEWISE cudagraphs (via mixed_mode).
        This should be called after adjust_cudagraph_sizes_for_spec_decode.
        """
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)
        return _e8_legacy(self, hidden_states, _E8_ARM, _E8_QUALIFY).argmax(dim=-1)

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        batch_size = common_attn_metadata.batch_size()

        if self.method in ("eagle3", "dflash"):
            assert isinstance(
                self.model,
                (
                    Eagle3LlamaForCausalLM,
                    Eagle3DeepseekV2ForCausalLM,
                    DFlashQwen3ForCausalLM,
                ),
            )
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size

        # FR13_TREE_SAMPLE_ROW (FIX-A1): sample the drafter at the committed
        # tree LEAF's flat verify row (+1-shifted published node id), not the
        # stock linear row prev_accepted_len. Default OFF = verbatim stock.
        _fr13_tsr_on = True  # FR13_TREE_SAMPLE_ROW baked ON
        logger.info_once(
            "FR13_TREE_SAMPLE_ROW drafter sample-row fix: tsr=%s (%s)",
            "1",  # FR13_TREE_SAMPLE_ROW baked ON
            "armed" if _fr13_tsr_on else "inert",
        )
        if _fr13_tsr_on:
            if False:  # FR13_TREE_REQKEY baked ON; dep-guard never fires
                raise RuntimeError(
                    "FR13_TREE_SAMPLE_ROW=1 requires FR13_TREE_REQKEY=1: the "
                    "pre-forward rewrite provides the per-step freshness "
                    "clear of _LUMO_FA_TREE_COMMIT_NROWS"
                )
            if self.needs_extra_input_slots:
                raise RuntimeError(
                    "FR13_TREE_SAMPLE_ROW: needs_extra_input_slots drafters "
                    "recompute token_indices_to_sample inside "
                    "set_inputs_first_pass; the sample-row fix would be "
                    "silently discarded (refusing to run vacuously)"
                )
            from vllm.model_executor.layers.mamba import (
                gdn_linear_attn as _fr13_tsr_gdn,
            )
            _fr13_tsr_nrows = int(getattr(
                _fr13_tsr_gdn, "_LUMO_FA_TREE_COMMIT_NROWS", 0
            ))
            # Consume-once: zero after read so a later propose without a
            # tree commit this step (prefill propose, capture warmup) can
            # never join a stale path (class-12 trap from step 1).
            _fr13_tsr_gdn._LUMO_FA_TREE_COMMIT_NROWS = 0
            _fr13_tsr_mode = os.environ.get(
                "FR10_DECODE_MODE_DEFAULT", "tree_mtp"
            )
            try:
                from vllm.v1.sample import (
                    rejection_sampler as _fr13_tsr_rs,
                )
                _fr13_tsr_mode = getattr(
                    _fr13_tsr_rs, "_FR10_DECODE_MODE", _fr13_tsr_mode
                )
            except Exception:
                pass
            # Tree-only engagement: _LUMO_FA_TREE_COMMIT_NROWS > 0 is set
            # exclusively by the tree committer helper (greedy + sampled
            # twins), which only runs on the tree-verify path; native/naive
            # decode and prefill proposes always see 0 here.
            if (
                _fr13_tsr_mode == "tree_mtp"
                and token_indices_to_sample is not None
                and _fr13_tsr_nrows > 0
            ):
                _fr13_tsr_paths = getattr(
                    _fr13_tsr_gdn, "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR", None
                )
                _fr13_tsr_lens = getattr(
                    _fr13_tsr_gdn, "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR", None
                )
                if _fr13_tsr_paths is None or _fr13_tsr_lens is None:
                    raise RuntimeError(
                        "FR13_TREE_SAMPLE_ROW: accepted-path device buffers "
                        "missing at propose time after a tree commit"
                    )
                # FR13_TSR_ROWMAP_BEGIN (pure row map; unit-tested verbatim:
                # row = base + (len > 0 ? paths[b, len-1] : 0))
                _fr13_tsr_fixed = bool(getattr(
                    _fr13_tsr_gdn, "_FR13_FIXED32_MODE", None
                ))
                if _fr13_tsr_fixed:
                    _fr13_tsr_batch_rows = int(getattr(
                        _fr13_tsr_gdn, "_FR13_FIXED32_BATCH_ROWS", -1
                    ))
                    _fr13_tsr_spec_rows = int(getattr(
                        _fr13_tsr_gdn, "_FR13_FIXED32_SPEC_ROWS", -1
                    ))
                    _fr13_tsr_spec_indices = getattr(
                        _fr13_tsr_gdn,
                        "_FR13_FIXED32_SPEC_BATCH_INDICES",
                        None,
                    )
                    _fr13_tsr_sampler_rids = tuple(
                        str(_fr13_tsr_rid)
                        for _fr13_tsr_rid in (
                            getattr(
                                _fr13_tsr_gdn,
                                "_LUMO_FA_SAMPLER_ROW_REQ_IDS",
                                None,
                            ) or ()
                        )
                    )
                    _fr13_tsr_spec_rids = tuple(
                        str(_fr13_tsr_rid)
                        for _fr13_tsr_rid in (
                            getattr(
                                _fr13_tsr_gdn,
                                "_LUMO_FA_SPEC_ROW_REQ_IDS",
                                None,
                            ) or ()
                        )
                    )
                    if (
                        not 1 <= _fr13_tsr_nrows <= 4
                        or _fr13_tsr_spec_rows != _fr13_tsr_nrows
                        or _fr13_tsr_batch_rows != int(batch_size)
                        or not 1 <= _fr13_tsr_batch_rows <= 4
                        or not isinstance(_fr13_tsr_spec_indices, tuple)
                        or len(_fr13_tsr_spec_indices) != _fr13_tsr_nrows
                        or any(
                            type(_fr13_tsr_i) is not int
                            for _fr13_tsr_i in _fr13_tsr_spec_indices
                        )
                        or tuple(sorted(set(_fr13_tsr_spec_indices)))
                        != _fr13_tsr_spec_indices
                        or any(
                            not 0 <= _fr13_tsr_i < _fr13_tsr_batch_rows
                            for _fr13_tsr_i in _fr13_tsr_spec_indices
                        )
                        or len(_fr13_tsr_sampler_rids)
                        != _fr13_tsr_batch_rows
                        or len(_fr13_tsr_spec_rids) != _fr13_tsr_nrows
                        or len(set(_fr13_tsr_sampler_rids))
                        != len(_fr13_tsr_sampler_rids)
                        or len(set(_fr13_tsr_spec_rids))
                        != len(_fr13_tsr_spec_rids)
                        or tuple(
                            _fr13_tsr_sampler_rids[_fr13_tsr_i]
                            for _fr13_tsr_i in _fr13_tsr_spec_indices
                        ) != _fr13_tsr_spec_rids
                        or int(_fr13_tsr_paths.size(0)) < _fr13_tsr_nrows
                        or int(_fr13_tsr_lens.numel()) < _fr13_tsr_nrows
                        or int(token_indices_to_sample.numel())
                        != _fr13_tsr_batch_rows
                        or int(common_attn_metadata.query_start_loc.numel())
                        < _fr13_tsr_batch_rows + 1
                    ):
                        raise RuntimeError(
                            "FR13 fixed32 TSR compact/full row-map drift"
                        )
                    _fr13_tsr_n = _fr13_tsr_nrows
                    _fr13_tsr_nonprefix = (
                        _fr13_tsr_spec_indices
                        != tuple(range(_fr13_tsr_n))
                    )
                else:
                    _fr13_tsr_n = min(
                        int(_fr13_tsr_nrows),
                        int(batch_size),
                        int(_fr13_tsr_paths.size(0)),
                    )
                    _fr13_tsr_spec_indices = tuple(range(_fr13_tsr_n))
                    _fr13_tsr_nonprefix = False
                _fr13_tsr_len_n = _fr13_tsr_lens[:_fr13_tsr_n].to(torch.long)
                _fr13_tsr_leaf = _fr13_tsr_paths[:_fr13_tsr_n].gather(
                    1,
                    (_fr13_tsr_len_n - 1).clamp(min=0).unsqueeze(1),
                ).squeeze(1)
                _fr13_tsr_zero_row = torch.zeros_like(_fr13_tsr_leaf)
                _fr13_tsr_leaf = torch.where(
                    _fr13_tsr_len_n > 0,
                    _fr13_tsr_leaf,
                    _fr13_tsr_zero_row,
                )
                if _fr13_tsr_nonprefix:
                    token_indices_to_sample = (
                        _fr13_fixed32_tsr_nonprefix_update(
                            token_indices_to_sample,
                            common_attn_metadata.query_start_loc,
                            _fr13_tsr_leaf,
                            _fr13_tsr_spec_indices,
                        )
                    )
                else:
                    _fr13_tsr_base = common_attn_metadata.query_start_loc[
                        :_fr13_tsr_n
                    ]
                    token_indices_to_sample = token_indices_to_sample.clone()
                    token_indices_to_sample[:_fr13_tsr_n] = (
                        _fr13_tsr_base.to(token_indices_to_sample.dtype)
                        + _fr13_tsr_leaf.to(token_indices_to_sample.dtype)
                    )
                # FR13_TSR_ROWMAP_END
        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )

        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),
        ):
            _fr13_ds_md = _FR13_DFWD_SPLIT.begin('model')
            ret_hidden_states = self.model(**model_kwargs)
            _FR13_DFWD_SPLIT.end('model', _fr13_ds_md)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        # FR13_FIXED32_MTP_KV_POSTFORWARD: first-pass MTP KV is now fresh and flat-mapped.
        if False:
            try:
                from vllm.model_executor.layers.mamba import (
                    gdn_linear_attn as _fr13_mtp_kv_gdn,
                )
                from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
                    launch_attn_kv_linear_remap_syncfree_fixed1_drafter as _fr13_mtp_kv1,
                )
                _fr13_mtp_payload = getattr(
                    self, '_fr13_fixed32_mtp_kv_payload', None
                )
                _fr13_mtp_needs_remap = int(_fr13_tsr_nrows) > 0
                if (_fr13_mtp_payload is not None) != _fr13_mtp_needs_remap:
                    raise RuntimeError(
                        'FR13 fixed32 MTP KV payload/commit freshness drift'
                    )
                if _fr13_mtp_payload is not None:
                    _fr13_mtp_keys = {
                        'schema', 'mode', 'compact_batch', 'batch_rows',
                        'spec_batch_indices', 'full_request_ids',
                        'spec_request_ids', 'measured',
                        'forward_step_index', 'event_index', 'mtp_kv',
                        'accepted_paths', 'accepted_lens',
                        'batch_indices', 'target_group_count',
                        'target_cache_count', 'target_plane_counts',
                        'accepted_paths_shape', 'accepted_lens_shape',
                        'permutation_group_id',
                        'permutation_slot_mapping',
                        'permutation_query_start_loc',
                        'permutation_spans', 'slot_restore_complete',
                    }
                    _fr13_mtp_proposal = getattr(
                        _fr13_mtp_kv_gdn,
                        '_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT', None,
                    )
                    _fr13_mtp_pending = getattr(
                        _fr13_mtp_kv_gdn,
                        '_FR13_FIXED32_PENDING_EVENT', None,
                    )
                    _fr13_mtp_measured = _fr13_mtp_payload.get('measured')
                    _fr13_mtp_B = int(
                        _fr13_mtp_payload.get('compact_batch', -1)
                    )
                    _fr13_mtp_batch_rows = int(
                        _fr13_mtp_payload.get('batch_rows', -1)
                    )
                    _fr13_mtp_spec_indices = _fr13_mtp_payload.get('spec_batch_indices')
                    if (
                        not isinstance(_fr13_mtp_payload, dict)
                        or set(_fr13_mtp_payload) != _fr13_mtp_keys
                        or _fr13_mtp_payload.get('schema')
                        != 'fr13-fixed32-mtp-kv-payload-v1'
                        or type(_fr13_mtp_measured) is not bool
                        or not 1 <= _fr13_mtp_B <= 4
                        or _fr13_mtp_B != int(_fr13_tsr_nrows)
                        or not 1 <= _fr13_mtp_batch_rows <= 4
                        or _fr13_mtp_batch_rows != int(batch_size)
                        or _fr13_mtp_batch_rows
                        != int(common_attn_metadata.num_reqs)
                        or not isinstance(_fr13_mtp_spec_indices, tuple)
                        or len(_fr13_mtp_spec_indices) != _fr13_mtp_B
                        or _fr13_mtp_spec_indices
                        != tuple(_fr13_tsr_spec_indices)
                        or _fr13_mtp_payload.get('slot_restore_complete')
                        is not True
                        or _fr13_mtp_payload.get(
                            'permutation_query_start_loc'
                        ) is not common_attn_metadata.query_start_loc
                        or not isinstance(_fr13_mtp_proposal, dict)
                        or _fr13_mtp_proposal.get('mode')
                        != _fr13_mtp_payload.get('mode')
                        or int(_fr13_mtp_proposal.get('batch_size', -1))
                        != _fr13_mtp_batch_rows
                        or tuple(_fr13_mtp_proposal.get('request_ids', ()))
                        != tuple(
                            _fr13_mtp_payload.get('full_request_ids', ())
                        )
                        or bool(_fr13_mtp_proposal.get('measured'))
                        is not _fr13_mtp_measured
                    ):
                        raise RuntimeError(
                            'FR13 fixed32 MTP KV payload identity drift'
                        )
                    if _fr13_mtp_measured:
                        if (
                            not isinstance(_fr13_mtp_pending, dict)
                            or _fr13_mtp_pending.get(
                                'target_kv_complete'
                            ) is not True
                            or _fr13_mtp_pending.get(
                                'drafter_kv_complete'
                            ) is not None
                            or _fr13_mtp_pending.get('kv_complete')
                            is not None
                            or int(_fr13_mtp_pending.get(
                                'forward_step_index', -1
                            )) != int(_fr13_mtp_payload.get(
                                'forward_step_index', -2
                            ))
                            or tuple(_fr13_mtp_pending.get(
                                'request_ids', ()
                            )) != tuple(_fr13_mtp_payload.get(
                                'spec_request_ids', ()
                            ))
                        ):
                            raise RuntimeError(
                                'FR13 fixed32 measured MTP KV lifecycle drift'
                            )
                    elif _fr13_mtp_pending is not None:
                        raise RuntimeError(
                            'FR13 fixed32 unmeasured MTP saw pending event'
                        )
                    _fr13_mtp_slot_mapping = (
                        self._slot_mapping_buffer[:slot_mapping_size]
                    )
                    _fr13_mtp_kv1(
                        kv_caches=(_fr13_mtp_payload['mtp_kv'],),
                        slot_mapping=_fr13_mtp_slot_mapping,
                        query_start_loc=(
                            common_attn_metadata.query_start_loc
                        ),
                        accepted_paths=(
                            _fr13_mtp_payload['accepted_paths']
                        ),
                        num_accepted_tokens=(
                            _fr13_mtp_payload['accepted_lens']
                        ),
                        num_spec_decodes=_fr13_mtp_B,
                        batch_indices=(
                            _fr13_mtp_payload['batch_indices']
                        ),
                    )
                    if _fr13_mtp_measured:
                        _fr13_mtp_observed = _fr13_mtp_pending.get('observed_work')
                        _fr13_mtp_kv_gdn._fr13_fixed32_observed_kv(
                            _fr13_mtp_observed,
                            _fr13_mtp_B,
                            _fr13_mtp_payload['target_group_count'],
                            _fr13_mtp_payload['target_cache_count'],
                            _fr13_mtp_payload['target_plane_counts'],
                            1,
                            (int(_fr13_mtp_payload['mtp_kv'].shape[0]),),
                            _fr13_mtp_payload['accepted_paths_shape'],
                            _fr13_mtp_payload['accepted_lens_shape'],
                        )
                        _fr13_mtp_events = getattr(
                            _fr13_mtp_kv_gdn,
                            '_FR13_FIXED32_CENSUS_EVENTS', None,
                        )
                        if (
                            not isinstance(_fr13_mtp_events, list)
                            or int(_fr13_mtp_payload.get(
                                'event_index', -1
                            )) != len(_fr13_mtp_events)
                        ):
                            raise RuntimeError(
                                'FR13 fixed32 MTP KV event index drift'
                            )
                        _fr13_mtp_pending['event_index'] = int(
                            _fr13_mtp_payload['event_index']
                        )
                        _fr13_mtp_pending['drafter_kv_complete'] = True
                        _fr13_mtp_pending['kv_complete'] = True
                    self._fr13_fixed32_mtp_kv_payload = None
            except Exception as _fr13_mtp_kv_exc:
                raise RuntimeError(
                    'FR13 fixed32 MTP KV1/final census completion failed: '
                    + type(_fr13_mtp_kv_exc).__name__ + ':'
                    + str(_fr13_mtp_kv_exc)
                ) from _fr13_mtp_kv_exc
        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            if 'propose() PARALLEL early-exit' not in _FR13_DFWD_SPLIT.pathmap:
                _FR13_DFWD_SPLIT.pathmap.add('propose() PARALLEL early-exit')
                try:
                    from vllm.logger import init_logger as _il
                    _il('vllm.fr13_dfwd_split').info('FR13_PATHMAP: %s', 'propose() PARALLEL early-exit')
                except Exception:
                    pass
            _fr13_ds_sm = _FR13_DFWD_SPLIT.begin('sample')
            draft_token_ids = self._greedy_sample(sample_hidden_states)
            _FR13_DFWD_SPLIT.end('sample', _fr13_ds_sm)
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        _fr10_active_decode_mode = os.environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp")
        try:
            from vllm.v1.sample import rejection_sampler as _fr10_rs_mode
            _fr10_active_decode_mode = getattr(
                _fr10_rs_mode, "_FR10_DECODE_MODE", _fr10_active_decode_mode
            )
        except Exception:
            pass
        _fr10_caterpillar_choices = [
            (0,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1),
            (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 1),
        ]
        _fr10_spine_only_choices = [
            (0,), (0, 0), (0, 0, 0), (0, 0, 0, 0),
            (0, 0, 0, 0, 0),
        ]
        # FR13_RESHAPE_DEPTH3: pure depth-3 spine floor-probe (chain3) and the
        # rank-1 depth-3 deploy shape (cat3w). Both are sorted (len, path)
        # tree_choices. ADDITIVE, exact-match guarded, default cat9 path
        # untouched. chain3 = trivial depth-3 truncation of chain5 (no width);
        # cat3w layers a root runner-up (1,) + a d1 runner-up (0,1) onto the
        # depth-3 spine. Downstream consumers (parent/ancestry masks, committer
        # path enum, eager-pack replay rows, conv-fusion prior windows) ALL
        # auto-adapt off the SPEC_CONFIG tree_choices -- only the drafter
        # packing order is hand-rolled here.
        _fr10_chain3_choices = [
            (0,), (0, 0), (0, 0, 0),
        ]
        _fr10_cat3w_choices = [
            (0,), (1,), (0, 0), (0, 1), (0, 0, 0),
        ]
        # FR13_RESHAPE_DEPTH5: two depth-5 reshape candidates that layer a
        # root runner-up (1,) onto the existing depth-5 spine. Both are
        # sorted (len, path) tree_choices, ADDITIVE, exact-match guarded,
        # default cat9 path untouched, FAIL-LOUD on disengagement.
        #   cat6root (6 nodes, depth-5, pad8) = pure depth-5 spine + (1,)
        #     root sibling, NO interior leaves (like cat3w's root-sibling
        #     slot but on the full chain5 spine).
        #   cat10 (10 nodes, depth-5, pad16) = cat9 + (1,) root sibling
        #     (the caterpillar with a root runner-up prepended at slot 1).
        _fr10_cat6root_choices = [
            (0,), (1,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0),
        ]
        _fr10_cat10_choices = [
            (0,), (1,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1),
            (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 1),
        ]
        # FR13_RESHAPE_333 (wide-shape B=4 probe): depth-3 tree with 3
        # candidates PER spine depth (root + the 2 interior spine depths),
        # i.e. spine (top-1), rank-1 leaf (top-2), AND rank-2 leaf (top-3).
        # 9 nodes, depth 3, pad16. Sorted (len, path) tree order:
        #   slot 0 (0,)     spine0 (root draft = argmax)
        #   slot 1 (1,)     root rank-1 leaf (root top-2)
        #   slot 2 (2,)     root rank-2 leaf (root top-3)  <-- NEW child-rank 2
        #   slot 3 (0,0)    spine1 (loop step 0 argmax)
        #   slot 4 (0,1)    d1 rank-1 leaf (step 0 top-2)
        #   slot 5 (0,2)    d1 rank-2 leaf (step 0 top-3)  <-- NEW child-rank 2
        #   slot 6 (0,0,0)  spine2 (loop step 1 argmax)
        #   slot 7 (0,0,1)  d2 rank-1 leaf (step 1 top-2)
        #   slot 8 (0,0,2)  d2 rank-2 leaf (step 1 top-3)  <-- NEW child-rank 2
        # The rank-2 token is a RUNNER-UP READ of topk(logits, 3)[:, 2] from
        # the SAME spine logits (no extra lm-head). It is NEVER fed into a
        # forward/recurrent state -- only packed into a leaf slot. Lossless
        # by construction: parent/ancestry masks, committer path enum,
        # eager-pack replay rows, conv prior windows ALL auto-derive from
        # SPEC_CONFIG tree_choices (only the drafter packing is hand-rolled).
        _fr10_threethree_choices = [
            (0,), (1,), (2,),
            (0, 0), (0, 1), (0, 2),
            (0, 0, 0), (0, 0, 1), (0, 0, 2),
        ]
        # HYDRA23 is the one non-caterpillar topology admitted to the optimized
        # wide drafter. Its four conditional descendants are filled by Arctic
        # and keyed by their full paths, never by the zero-spine depth alias.
        _fr13_hydra23_choices = [
            (0,), (1,), (2,),
            (0, 0), (0, 1), (0, 2), (1, 0),
            (0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0),
            (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2),
            (1, 0, 0, 0),
            (0, 0, 0, 0, 0), (1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ]
        _fr13_hydra23_tail_paths = [
            (1, 0), (1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0, 0),
        ]
        _fr13_fixed32_choices = [
            (0,), (1,), (2,),
            (0, 0), (0, 1), (0, 2), (1, 0), (2, 0),
            (0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0),
            (2, 0, 0),
            (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2),
            (1, 0, 0, 0), (2, 0, 0, 0),
            (0, 0, 0, 0, 0), (0, 0, 0, 0, 1),
            (0, 0, 0, 0, 2), (1, 0, 0, 0, 0),
            (2, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (2, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ]
        _fr13_fixed32_tail_paths = [
            (1, 0), (1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0, 0),
            (2, 0), (2, 0, 0), (2, 0, 0, 0), (2, 0, 0, 0, 0),
            (2, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0),
        ]
        # THE TREE THIS MODE SERVES. The list above is hydra27's; tail10
        # respends the four slots hydra27 disarms, so ids >= 17 carry different
        # paths and the two lists are NOT equal. A single list here meant the
        # exact-shape predicate could only ever recognise one profile, and a
        # correct hydra31 tree read as a mode/topology mismatch on the first
        # real request. Mirrored because this blob is planted text that cannot
        # import; kept honest by tests/test_fr14_gdn_schedule_contract_parity.
        _fr13_tail10_choices = [
            (0,), (1,), (2,),
            (0, 0), (0, 1), (0, 2), (1, 0), (2, 0),
            (0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0), (2, 0, 0),
            (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (1, 0, 0, 0),
            (0, 0, 0, 0, 0), (0, 0, 0, 0, 1), (0, 0, 0, 0, 2),
            (1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ]
        _fr13_fixed32_choices_by_mode = {
            "": _fr13_fixed32_choices,
            "tail6_fixed32": _fr13_fixed32_choices,
            "hydra27_fixed32": _fr13_fixed32_choices,
            "hydra31_fixed32": _fr13_tail10_choices,
        }
        _fr13_fixed32_expected_choices = _fr13_fixed32_choices_by_mode.get(
            _FR13_FIXED32_MODE
        )
        if _FR13_FIXED32_MODE and _fr13_fixed32_expected_choices is None:
            raise RuntimeError(
                "FR13 fixed32 propose guard has no tree for mode "
                + repr(_FR13_FIXED32_MODE)
                + "; known modes are "
                + repr(sorted(_fr13_fixed32_choices_by_mode))
            )
        _fr10_tree_choices_current = [
            tuple(_x) for _x in getattr(self, "tree_choices", [])
        ]
        _fr13_decode_ok = _fr10_active_decode_mode == "tree_mtp"
        _fr13_spec_ok = int(self.num_speculative_tokens) == 31
        _fr13_tree_ok = (
            _fr10_tree_choices_current == _fr13_fixed32_expected_choices
        )
        _fr13_is_fixed32 = (
            bool(_FR13_FIXED32_MODE)
            and _fr13_decode_ok
            and _fr13_spec_ok
            and _fr13_tree_ok
        )
        if bool(_FR13_FIXED32_MODE) != _fr13_is_fixed32:
            # WHICH PREDICATE FAILED, and both trees. The old message said
            # exact_shape=False and nodes=31 -- a correct node count and a bare
            # False -- which named the symptom and hid every cause.
            _fr13_first_diff = next(
                (
                    _i
                    for _i in range(
                        max(
                            len(_fr10_tree_choices_current),
                            len(_fr13_fixed32_expected_choices or ()),
                        )
                    )
                    if _fr10_tree_choices_current[_i : _i + 1]
                    != (_fr13_fixed32_expected_choices or [])[_i : _i + 1]
                ),
                None,
            )
            raise RuntimeError(
                "FR13 fixed32 mode/topology mismatch: mode="
                + repr(_FR13_FIXED32_MODE)
                + " decode_mode_ok=" + str(_fr13_decode_ok)
                + " (" + repr(_fr10_active_decode_mode) + ")"
                + " spec_tokens_ok=" + str(_fr13_spec_ok)
                + " (observed " + str(int(self.num_speculative_tokens))
                + " against audited 31)"
                + " tree_ok=" + str(_fr13_tree_ok)
                + " nodes=" + str(len(_fr10_tree_choices_current))
                + " against audited "
                + str(len(_fr13_fixed32_expected_choices or ()))
                + (
                    ""
                    if _fr13_first_diff is None
                    else (
                        "; first differing path ["
                        + str(_fr13_first_diff)
                        + "]: observed "
                        + repr(
                            _fr10_tree_choices_current[
                                _fr13_first_diff : _fr13_first_diff + 1
                            ]
                        )
                        + " against audited "
                        + repr(
                            (_fr13_fixed32_expected_choices or [])[
                                _fr13_first_diff : _fr13_first_diff + 1
                            ]
                        )
                    )
                )
            )
        _fr13_hydra23_armed = os.path.exists("/logs/fr13_hydra23.arm")
        _fr13_is_hydra23 = (
            _fr10_active_decode_mode == "tree_mtp"
            and int(self.num_speculative_tokens) == 23
            and _fr10_tree_choices_current == _fr13_hydra23_choices
        )
        if _fr13_hydra23_armed != _fr13_is_hydra23:
            raise RuntimeError(
                "FR13_HYDRA23 sidecar/topology mismatch: armed="
                + str(_fr13_hydra23_armed)
                + " exact_shape=" + str(_fr13_is_hydra23)
                + " nodes=" + str(len(_fr10_tree_choices_current))
            )
        if _fr13_is_hydra23:
            if (
                not os.path.exists("/logs/fr13_tail_mode.arm")
                or not os.path.exists(
                    "/logs/fr13_draft_source_merged.arm"
                )
            ):
                raise RuntimeError(
                    "FR13_HYDRA23 requires tail-mode and merged-drafter sidecars"
                )
            logger.info_once(
                "FR13_HYDRA23 exact topology engaged: nodes=%d rescue_chains=%s",
                len(_fr13_hydra23_choices),
                "rank1:4",
            )
        _fr10_is_caterpillar = (
            _fr10_active_decode_mode == "tree_mtp"
            and int(self.num_speculative_tokens) == 9
            and _fr10_tree_choices_current == _fr10_caterpillar_choices
        )
        _fr10_is_spine_only = (
            _fr10_active_decode_mode == "tree_mtp"
            and int(self.num_speculative_tokens) == 5
            and _fr10_tree_choices_current == _fr10_spine_only_choices
        )
        _fr10_is_chain3 = (
            _fr10_active_decode_mode == "tree_mtp"
            and int(self.num_speculative_tokens) == 3
            and _fr10_tree_choices_current == _fr10_chain3_choices
        )
        _fr10_is_cat3w = (
            _fr10_active_decode_mode == "tree_mtp"
            and int(self.num_speculative_tokens) == 5
            and _fr10_tree_choices_current == _fr10_cat3w_choices
        )
        _fr10_is_cat6root = (
            _fr10_active_decode_mode == "tree_mtp"
            and int(self.num_speculative_tokens) == 6
            and _fr10_tree_choices_current == _fr10_cat6root_choices
        )
        _fr10_is_cat10 = (
            _fr10_active_decode_mode == "tree_mtp"
            and int(self.num_speculative_tokens) == 10
            and _fr10_tree_choices_current == _fr10_cat10_choices
        )
        # FR13_RESHAPE_333: exact-match guard for the 3-3-3 wide shape. Same
        # num_speculative_tokens (9) as cat9, so the tree_choices comparison
        # (NOT the count) is what disambiguates them -- the two choice lists
        # are disjoint, so at most one of _fr10_is_caterpillar/_fr10_is_333 is
        # ever True. Default cat9/cat3w/cat6root/cat10/chain* untouched.
        _fr10_is_333 = (
            _fr10_active_decode_mode == "tree_mtp"
            and int(self.num_speculative_tokens) == 9
            and _fr10_tree_choices_current == _fr10_threethree_choices
        )
        _fr10_wide_src_choices = _fr10_tree_choices_current
        # FR13_RESHAPE_WIDE: GENERAL width-N caterpillar drafter. Engages for
        # ANY tree_choices that is a single all-zeros spine with arbitrary-width
        # leaf children hanging DIRECTLY off each spine node (incl root), and
        # that matches NONE of the hand-rolled exact shapes above (so the
        # verified default cat9/333/cat6root/cat10/cat3w/chain* paths stay
        # byte-identical -- wide is purely additive). The caterpillar condition
        # is: every node path p has an all-zeros PARENT path (p[:-1] all 0), so
        # every off-spine node is a direct child of a spine node and its token
        # is a pure topk read off the SAME spine logits -- no extra forward, no
        # recurrent feed. That makes ANY width K (top-5/top-10/top-20/...)
        # lossless by the drafter-agnostic committer, exactly like 333's rank-2
        # read, just generalized to read rank-1..K-1 at any spine depth. Width
        # per depth, spine depth, and the flat packing order are ALL derived
        # from tree_choices (no per-shape hand-rolled stack).
        _fr10_wide_choices_ok = (
            _fr10_active_decode_mode == "tree_mtp"
            and len(_fr10_wide_src_choices) > 0
            and all(
                len(_p) >= 1 and all(_e == 0 for _e in _p[:-1])
                for _p in _fr10_wide_src_choices
            )
        )
        _fr10_is_wide = (
            (_fr10_wide_choices_ok or _fr13_is_hydra23 or _fr13_is_fixed32)
            and not (
                _fr10_is_caterpillar
                or _fr10_is_spine_only
                or _fr10_is_chain3
                or _fr10_is_cat3w
                or _fr10_is_cat6root
                or _fr10_is_cat10
                or _fr10_is_333
            )
        )
        # Build the wide plan from tree_choices: spine depth D (longest
        # all-zeros path), per-parent-position width (max child-rank+1 among
        # nodes whose parent is the spine node at that position; parent_pos =
        # len(path)-1, 0=root, 1=spine0, ...), forward-step count (D-1), and
        # the flat packing plan (parent_pos, child_rank) in sorted tree order.
        # FAIL-LOUD on a spine gap (a missing all-zeros rung breaks the forward
        # chain) so a malformed tree can never silently mis-pack.
        _fr10_wide_D = 0
        _fr10_wide_width = {}
        _fr10_wide_plan = []
        _fr10_wide_paths = []
        _fr10_wide_spine_steps = 0
        if _fr10_is_wide:
            _fr10_wide_D = max(
                len(_p)
                for _p in _fr10_wide_src_choices
                if all(_e == 0 for _e in _p)
            )
            _fr10_wide_spine_set = set(_fr10_wide_src_choices)
            for _fr10_wL in range(1, _fr10_wide_D + 1):
                if tuple([0] * _fr10_wL) not in _fr10_wide_spine_set:
                    raise RuntimeError(
                        "FR13_RESHAPE_WIDE: spine gap, missing all-zeros path "
                        "of length " + str(_fr10_wL) + " in "
                        + repr(_fr10_wide_src_choices)
                    )
            for _p in _fr10_wide_src_choices:
                _fr10_wpp = len(_p) - 1
                _fr10_wrk = _p[-1]
                _fr10_wide_width[_fr10_wpp] = max(
                    _fr10_wide_width.get(_fr10_wpp, 0), _fr10_wrk + 1
                )
            _fr10_wide_spine_steps = _fr10_wide_D - 1
            _fr10_wide_plan = [
                (len(_p) - 1, _p[-1]) for _p in _fr10_wide_src_choices
            ]
            _fr10_wide_paths = list(_fr10_wide_src_choices)
            # NOTE: logger.info_once hashes (msg, *args) to dedup, so EVERY arg
            # must be hashable -- pass the widths dict + tree as STRINGS, never
            # the dict/list objects (a dict arg raises TypeError: unhashable
            # type: 'dict' and kills the engine at warmup).
            logger.info_once(
                "FR13_RESHAPE_WIDE engaged: depth=%d spine_steps=%d "
                "widths=%s nodes=%d tree=%s",
                _fr10_wide_D,
                _fr10_wide_spine_steps,
                str({
                    _fr10_wk: _fr10_wide_width[_fr10_wk]
                    for _fr10_wk in sorted(_fr10_wide_width)
                }),
                len(_fr10_wide_src_choices),
                repr(_fr10_wide_src_choices),
            )
        # FR13_RESHAPE_DEPTH3: number of post-root spine forward steps and the
        # depth steps (1-based) at which the runner-up leaf is consumed. cat9
        # and chain5 keep the original depth-5 spine (4 steps, leaves at every
        # step for cat9). The depth-3 shapes run 2 steps; cat3w consumes the
        # d1 leaf at step 1 only (its (0,1) sibling); chain3 consumes none.
        # FR13_RESHAPE_DEPTH5: cat6root and cat10 keep the depth-5 spine (4
        # steps). cat6root consumes NO interior leaves (root sibling only);
        # cat10 consumes the d1..d4 leaves like cat9 (root sibling extra).
        # FR13_RESHAPE_333: 3-3-3 is depth-3 -> 2 post-root spine forward
        # steps (loop steps 0,1 == depths 2,3), same as chain3/cat3w.
        if _fr10_is_chain3 or _fr10_is_cat3w or _fr10_is_333:
            _fr10_spine_steps = 2
        elif _fr10_is_wide:
            # FR13_RESHAPE_WIDE: D-1 post-root spine forwards (derived from the
            # tree's all-zeros spine depth), so the step count always matches
            # the committed tree depth (no over-run mutating KV/seq_lens).
            _fr10_spine_steps = _fr10_wide_spine_steps
            # accept>5 TAIL mode (sidecar /logs/fr13_tail_mode.arm; FR13_* env is stripped in the
            # worker): cap native MTP forwards at the HEAD (head_depth-1 = 4). The deep chain
            # (depths head_depth..wide_D-1) is Arctic-retrieved + appended after the loop, NOT
            # MTP-drafted -- else the native path runs ~wide_D-1 (~20) slow autoregressive forwards.
            if os.path.exists("/logs/fr13_tail_mode.arm"):
                _fr13_tail_hd = 5   # == fr13_merged_drafter.TAIL_HEAD_DEPTH; mismatch fail-louds at packer
                if _fr10_wide_spine_steps > (_fr13_tail_hd - 1):
                    _fr10_spine_steps = _fr13_tail_hd - 1
        else:
            _fr10_spine_steps = 4
        # FR13_RESHAPE_333: 3-3-3 consumes a rank-1 leaf at BOTH interior
        # spine steps (loop steps 0,1 -> token_index+1 in {1,2}); it ALSO
        # consumes a rank-2 leaf at the same steps (handled separately via
        # _fr10_leaf2_steps below).
        if (
            _fr10_is_spine_only
            or _fr10_is_chain3
            or _fr10_is_cat6root
            or _fr10_is_wide
        ):
            # FR13_RESHAPE_WIDE collects ALL its leaves via its own per-depth
            # topk capture (_fr10_wide_topk), so the legacy rank-1/rank-2 leaf
            # collection is disabled here (empty _fr10_leaf_steps).
            _fr10_leaf_steps = frozenset()
        elif _fr10_is_cat3w:
            _fr10_leaf_steps = frozenset({1})
        elif _fr10_is_333:
            _fr10_leaf_steps = frozenset({1, 2})
        else:
            _fr10_leaf_steps = frozenset({1, 2, 3, 4})
        # FR13_RESHAPE_333: the SECOND runner-up (rank-2 / top-3) interior
        # leaf steps. ONLY 3-3-3 reads a child-rank-2 token; every other
        # shape leaves this empty (so its topk stays k=2, no behavior change).
        if _fr10_is_333:
            _fr10_leaf2_steps = frozenset({1, 2})
        else:
            _fr10_leaf2_steps = frozenset()
        if (
            _fr10_is_caterpillar
            or _fr10_is_spine_only
            or _fr10_is_chain3
            or _fr10_is_cat3w
            or _fr10_is_cat6root
            or _fr10_is_cat10
            or _fr10_is_333
            or _fr10_is_wide
        ):
            # FR10_CATERPILLAR_NATIVE_SPINE_TOP2: read-only drafter fix.
            # Run the native causal MTP spine unchanged for depth 5. At each
            # post-root spine step, read the runner-up token from the same
            # logits and pack it into the caterpillar leaf slot. Leaves are
            # never fed back into any forward or recurrent state. The 5-node
            # spine-only diagnostic uses the same native causal MTP spine and
            # packs only those five spine tokens.
            # FR13_DRAFTER_SINGLE_LOGITS (FIX-1, default ON): compute the
            # full-vocab bf16 lm-head logits ONCE per drafter step and take
            # the spine draft token as argmax of that SAME tensor. Legacy
            # (flag OFF = the A/B instrument) additionally calls
            # self._greedy_sample, which RECOMPUTES compute_logits in live
            # vLLM eagle.py — a second ~2.5 GB lm-head read per step.
            # use_local_argmax_reduction routes _greedy_sample through
            # get_top_tokens (shard-local logits.max on a separately
            # computed lm-head output, different selection semantics), so
            # that config falls back to the exact legacy path.
            _fr13_single_logits = (
                True  # E8 source-controlled arm; no worker-env dependency
                and not getattr(self, "use_local_argmax_reduction", False)
            )
            logger.info_once(
                "FR13_DRAFTER_SINGLE_LOGITS drafter path engaged: "
                "single_logits=%s (env=%s, use_local_argmax_reduction=%s)",
                _fr13_single_logits,
                '1',  # E8 actual source arm
                getattr(self, "use_local_argmax_reduction", False),
            )
            # FR13_FIX1_SELFCHECK (default OFF; DIAGNOSTIC ONLY, like
            # FR13_FORCE_SPINE_COMMIT): in-process dual-path byte-identity
            # proof for FIX-1. With the single-logits path serving, ALSO run
            # legacy self._greedy_sample (the second compute_logits) per
            # drafter step and assert torch.equal against the
            # argmax-of-same-logits draft tokens. Needs no cross-boot
            # anything: this is the decisive OFF==ON instrument on a
            # substrate whose cross-boot floor is non-deterministic under
            # BI=0 AND BI=1 (FR13_B1_FIX1_CONFIRM_BIND.md step 1).
            _fr13_selfcheck = (
                _fr13_single_logits
                and _E8_QUALIFY  # existing FIX1 check; baked from qualification manifest
            )
            if _fr13_selfcheck:
                logger.info_once(
                    "FR13_FIX1_SELFCHECK engaged: dual-path drafter "
                    "byte-identity assert active (diagnostic, default OFF)"
                )
                if not hasattr(self, "_fr13_fix1_sc_stats"):
                    self._fr13_fix1_sc_stats = {
                        "steps_checked": 0,
                        "rows_checked": 0,
                        "mismatch_steps": 0,
                    }

            def _fr13_sc_check(_site, _new_ids, _legacy_ids):
                _st = self._fr13_fix1_sc_stats
                _st["steps_checked"] += 1
                _st["rows_checked"] += int(_new_ids.numel())
                _ok = bool(torch.equal(_new_ids, _legacy_ids))
                if not _ok:
                    _st["mismatch_steps"] += 1
                    _bad = (
                        (_new_ids != _legacy_ids)
                        .nonzero(as_tuple=False)
                        .flatten()
                        .tolist()
                    )
                    logger.error(
                        "FR13_FIX1_SELFCHECK MISMATCH at %s step %d: "
                        "rows=%s argmax=%s greedy_sample=%s",
                        _site,
                        _st["steps_checked"],
                        _bad,
                        _new_ids.flatten()[_bad].tolist(),
                        _legacy_ids.flatten()[_bad].tolist(),
                    )
                try:
                    import json as _sc_json

                    with open(
                        os.environ.get(
                            "FR13_FIX1_SELFCHECK_DUMP",
                            "/logs/fr13_fix1_selfcheck.json",
                        ),
                        "w",
                    ) as _sc_fh:
                        _sc_json.dump(_st, _sc_fh)
                except Exception:
                    pass
                if not _ok or _st["steps_checked"] % 50 == 0:
                    logger.info(
                        "FR13_FIX1_SELFCHECK needle: steps=%d rows=%d "
                        "mismatch_steps=%d",
                        _st["steps_checked"],
                        _st["rows_checked"],
                        _st["mismatch_steps"],
                    )
                if not _ok:
                    raise AssertionError(
                        "FR13_FIX1_SELFCHECK: argmax-of-same-logits != "
                        "_greedy_sample at " + _site
                    )

            # FR13_DRAFT_VOCAB_K (FR-Spec revival, 2026-07-26): draft
            # argmax/topk over a block-aligned lm_head subset. Gather-mode
            # indices are mapped back to real vocab ids before any token is
            # consumed. The root head remains full by default; the exact
            # fixed32 candidate opts it into the same subset with
            # FR13_DRAFT_VOCAB_ROOT=1.
            _fr13_dvk_root_raw = os.environ.get(
                "FR13_DRAFT_VOCAB_ROOT", "0"
            )
            if _fr13_dvk_root_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_VOCAB_ROOT must be exactly 0 or 1"
                )
            _fr13_dvk_root = _fr13_dvk_root_raw == "1"
            if _fr13_dvk_root and not _fr13_is_fixed32:
                raise RuntimeError(
                    "FR13_DRAFT_VOCAB_ROOT=1 requires exact fixed32 topology"
                )
            if _fr13_dvk_root and not _fr13_single_logits:
                raise RuntimeError(
                    "FR13_DRAFT_VOCAB_ROOT=1 requires "
                    "FR13_DRAFTER_SINGLE_LOGITS=1"
                )
            if _fr13_dvk_root and _fr13_selfcheck:
                raise RuntimeError(
                    "FR13_DRAFT_VOCAB_ROOT=1 is incompatible with "
                    "FR13_FIX1_SELFCHECK=1"
                )
            _fr13_dvk_configured = int(
                os.environ.get("FR13_DRAFT_VOCAB_K", "0") or 0
            )
            _fr13_dvk_configured -= _fr13_dvk_configured % 128
            if _fr13_dvk_root and _fr13_dvk_configured <= 0:
                raise RuntimeError(
                    "FR13_DRAFT_VOCAB_ROOT=1 requires "
                    "FR13_DRAFT_VOCAB_K>=128"
                )
            _fr13_dfwd_top3_raw = os.environ.get(
                "FR13_DFWD_K64_TOP3", "0"
            )
            if _fr13_dfwd_top3_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DFWD_K64_TOP3 must be exactly 0 or 1"
                )
            _fr13_dfwd_top3 = _fr13_dfwd_top3_raw == "1"
            if _fr13_dfwd_top3 and (
                not _fr13_is_fixed32
                or not _fr13_dvk_root
                or not _fr13_single_logits
                or _fr13_dvk_configured != 65536
                or not _fr10_is_wide
                or int(batch_size) != 1
                or tuple(
                    int(_fr10_wide_width.get(_fr13_top3_depth, 0))
                    for _fr13_top3_depth in range(5)
                )
                != (3, 3, 3, 3, 3)
            ):
                raise RuntimeError(
                    "FR13 DFWD K64 top3 requires B1 exact fixed32, root1, "
                    "K64 single logits, and width3 at all five head depths"
                )
            # FR14_FUSED_DRAFT_TOPK (default OFF): the K0 analogue of the K64
            # top3 op. Under the served full-vocabulary drafter profile every
            # one of the five head reads runs argmax(248320) AND
            # topk(248320, 3) as two separate ATen calls -- an ATen multi-block
            # radix select is a chain of kernels, and the pair measures
            # ~68 us/head read at this geometry against ~8 us for one fused
            # launch. Selection is byte-identical (ids AND order, including the
            # tie-break, which argmax and topk do NOT agree on -- see
            # results/fr14_nvfp4_port_20260816/fused_draft_topk.md).
            _fr14_fused_topk_raw = os.environ.get(
                "FR14_FUSED_DRAFT_TOPK", "0"
            )
            if _fr14_fused_topk_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR14_FUSED_DRAFT_TOPK must be exactly 0 or 1"
                )
            _fr14_fused_topk = _fr14_fused_topk_raw == "1"
            if _fr14_fused_topk and (
                not _fr13_is_fixed32
                or _fr13_dvk_root
                or _fr13_dvk_configured != 0
                or not _fr13_single_logits
                or _fr13_dfwd_top3
                or not _fr10_is_wide
                or int(batch_size) not in (1, 2, 3, 4)
                or tuple(
                    int(_fr10_wide_width.get(_fr14_fused_depth, 0))
                    for _fr14_fused_depth in range(5)
                )
                != (3, 3, 3, 3, 3)
            ):
                raise RuntimeError(
                    "FR14 fused draft top-k requires exact fixed32 at the K0 "
                    "full-vocabulary profile (ROOT=0, K=0), single logits, no "
                    "K64 top3, no sibling-dedup slack, and width3 at all five "
                    "head depths"
                )
            _fr14_fused_topk_blocks = int(
                os.environ.get("FR14_FUSED_DRAFT_TOPK_BLOCKS", "64") or 64
            )
            if _fr14_fused_topk and not (1 <= _fr14_fused_topk_blocks <= 121):
                raise RuntimeError(
                    "FR14_FUSED_DRAFT_TOPK_BLOCKS must be in 1..121"
                )
            _fr13_dh_rows_raw = os.environ.get(
                "FR13_DRAFT_HEAD_PAD_ROWS", "0"
            )
            _fr13_dh_ab_raw = os.environ.get(
                "FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB", "0"
            )
            _fr13_dh_m32_live_raw = os.environ.get(
                "FR13_DRAFT_HEAD_M32_LIVE_AB", "0"
            )
            _fr13_dh_m32_prod_raw = os.environ.get(
                "FR13_DRAFT_HEAD_M32_PRODUCTION", "0"
            )
            _fr13_dh_u8_live_raw = os.environ.get(
                "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB", "0"
            )
            _fr13_dh_u8_quality_raw = os.environ.get(
                "FR13_DRAFT_HEAD_M1_R64_U8_QUALITY_GATE", "0"
            )
            _fr13_dh_u8_taw_quality_raw = os.environ.get(
                "FR13_DRAFT_HEAD_M1_R64_U8_TAW_QUALITY_GATE", "0"
            )
            _fr13_dh_u8_prod_raw = os.environ.get(
                "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION", "0"
            )
            _fr13_dh_m4_u8_live_raw = os.environ.get(
                "FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB", "0"
            )
            _fr13_dh_m4_u8_quality_raw = os.environ.get(
                "FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE", "0"
            )
            _fr13_dh_m4_u8_prod_raw = os.environ.get(
                "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION", "0"
            )
            _fr13_dh_fp8_raw = os.environ.get(
                "FR13_DRAFT_HEAD_FP8", "0"
            )
            _fr13_dh_fp8_static_io_raw = os.environ.get(
                "FR13_DRAFT_HEAD_FP8_STATIC_IO", "0"
            )
            _fr13_dh_fp8_arm = os.environ.get(
                "FR13_DRAFT_HEAD_FP8_ARM", ""
            )
            if _fr13_dh_rows_raw not in ("0", "32", "64", "128"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_PAD_ROWS must be exactly one of "
                    "0, 32, 64, or 128"
                )
            if _fr13_dh_ab_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB must be exactly 0 or 1"
                )
            if _fr13_dh_m32_live_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_M32_LIVE_AB must be exactly 0 or 1"
                )
            if _fr13_dh_m32_prod_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_M32_PRODUCTION must be exactly 0 or 1"
                )
            if _fr13_dh_u8_live_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB must be exactly 0 or 1"
                )
            if _fr13_dh_u8_quality_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_M1_R64_U8_QUALITY_GATE must be exactly 0 or 1"
                )
            if _fr13_dh_u8_taw_quality_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_M1_R64_U8_TAW_QUALITY_GATE must be "
                    "exactly 0 or 1"
                )
            if _fr13_dh_u8_prod_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION must be exactly 0 or 1"
                )
            if _fr13_dh_m4_u8_live_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB must be exactly 0 or 1"
                )
            if _fr13_dh_m4_u8_quality_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE must be exactly 0 or 1"
                )
            if _fr13_dh_m4_u8_prod_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION must be exactly 0 or 1"
                )
            if _fr13_dh_fp8_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_FP8 must be exactly 0 or 1"
                )
            if _fr13_dh_fp8_static_io_raw not in ("0", "1"):
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_FP8_STATIC_IO must be exactly 0 or 1"
                )
            _fr13_dh_rows = int(_fr13_dh_rows_raw)
            _fr13_dh_ab = _fr13_dh_ab_raw == "1"
            _fr13_dh_m32_live = _fr13_dh_m32_live_raw == "1"
            _fr13_dh_m32_prod = _fr13_dh_m32_prod_raw == "1"
            _fr13_dh_u8_live = _fr13_dh_u8_live_raw == "1"
            _fr13_dh_u8_quality = _fr13_dh_u8_quality_raw == "1"
            _fr13_dh_u8_taw_quality = (
                _fr13_dh_u8_taw_quality_raw == "1"
            )
            _fr13_dh_u8_prod = _fr13_dh_u8_prod_raw == "1"
            _fr13_dh_u8_active = _fr13_dh_u8_live or _fr13_dh_u8_prod
            _fr13_dh_m4_u8_live = _fr13_dh_m4_u8_live_raw == "1"
            if _fr13_dh_u8_quality and not _fr13_dh_u8_live:
                raise RuntimeError(
                    "FR13 draft-head U8 quality gate requires LIVE_AB=1"
                )
            if _fr13_dh_u8_taw_quality and not _fr13_dh_u8_quality:
                raise RuntimeError(
                    "FR13 draft-head U8 TAW quality requires candidate-served "
                    "quality mode"
                )
            _fr13_dh_m4_u8_quality = _fr13_dh_m4_u8_quality_raw == "1"
            _fr13_dh_m4_u8_prod = _fr13_dh_m4_u8_prod_raw == "1"
            _fr13_dh_m4_u8_active = (
                _fr13_dh_m4_u8_live or _fr13_dh_m4_u8_prod
            )
            if _fr13_dh_m4_u8_quality and not _fr13_dh_m4_u8_live:
                raise RuntimeError(
                    "FR13 draft-head M4 U8 quality requires LIVE_AB=1"
                )
            _fr13_dh_fp8 = _fr13_dh_fp8_raw == "1"
            _fr13_dh_fp8_static_io = (
                _fr13_dh_fp8_static_io_raw == "1"
            )
            if _fr13_dh_fp8_static_io and not _fr13_dh_fp8:
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_FP8_STATIC_IO=1 requires "
                    "FR13_DRAFT_HEAD_FP8=1"
                )
            if _fr13_dh_fp8:
                if (
                    not _fr13_dh_fp8_arm
                    or len(_fr13_dh_fp8_arm) > 200
                    or any(
                        _fr13_dh_arm_char not in (
                            "abcdefghijklmnopqrstuvwxyz"
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                        )
                        for _fr13_dh_arm_char in _fr13_dh_fp8_arm
                    )
                ):
                    raise RuntimeError(
                        "FR13_DRAFT_HEAD_FP8=1 requires a canonical "
                        "FR13_DRAFT_HEAD_FP8_ARM"
                    )
            elif _fr13_dh_fp8_arm:
                raise RuntimeError(
                    "FR13_DRAFT_HEAD_FP8=0 forbids FR13_DRAFT_HEAD_FP8_ARM"
                )
            _fr13_dh_modes = sum(
                int(value)
                for value in (
                    bool(_fr13_dh_rows),
                    _fr13_dh_ab,
                    _fr13_dh_m32_live,
                    _fr13_dh_m32_prod,
                    _fr13_dh_u8_live,
                    _fr13_dh_u8_prod,
                    _fr13_dh_m4_u8_active,
                    _fr13_dh_fp8,
                )
            )
            if _fr13_dh_modes > 1:
                raise RuntimeError(
                    "FR13 draft-head candidate, diagnostics, and production "
                    "modes are mutually exclusive"
                )
            if (
                _fr13_dh_m32_prod
                and os.environ.get(
                    "FR13_DRAFT_HEAD_M32_INTERNAL_PRODUCTION_ATTESTED"
                )
                != "1"
            ):
                raise RuntimeError(
                    "FR13 draft-head M32 production has no launcher attestation"
                )
            if (
                _fr13_dh_u8_prod
                and os.environ.get(
                    "FR13_DRAFT_HEAD_M1_R64_U8_"
                    "INTERNAL_PRODUCTION_ATTESTED"
                )
                != "1"
            ):
                raise RuntimeError(
                    "FR13 draft-head U8 production has no validator attestation"
                )
            if (
                _fr13_dh_m4_u8_prod
                and os.environ.get(
                    "FR13_DRAFT_HEAD_M4_R64_U8_INTERNAL_PRODUCTION_ATTESTED"
                )
                != "1"
            ):
                raise RuntimeError(
                    "FR13 draft-head M4 U8 production has no validator attestation"
                )
            if (
                _fr13_dh_rows
                or _fr13_dh_ab
                or _fr13_dh_m32_live
                or _fr13_dh_m32_prod
                or _fr13_dh_u8_active
                or _fr13_dh_m4_u8_active
                or _fr13_dh_fp8
            ) and (
                not _fr13_is_fixed32
                or not _fr13_dvk_root
                or not _fr13_single_logits
                or _fr13_dvk_configured != 65536
            ):
                raise RuntimeError(
                    "FR13 draft-head padding requires exact fixed32, root "
                    "subset, single-logits, and FR13_DRAFT_VOCAB_K=65536"
                )
            if _fr13_dh_fp8 and (
                _FR13_FIXED32_MODE != "hydra27_fixed32"
                or int(batch_size) not in (1, 4)
                or not _fr10_is_wide
                or tuple(
                    int(_fr10_wide_width.get(_fr13_dh_k64_depth, 0))
                    for _fr13_dh_k64_depth in range(5)
                )
                != (3, 3, 3, 3, 3)
            ):
                raise RuntimeError(
                    "FR13 direct K64 head requires exact Hydra27 physical32 "
                    "K64/root1 B1 or B4 geometry"
                )
            if _fr13_dh_m4_u8_active and int(batch_size) != 4:
                raise RuntimeError(
                    "FR13 draft-head M4 U8 live A/B requires exact B4 geometry"
                )
            _fr13_dh_source_sha = os.environ.get(
                "FR13_DRAFT_HEAD_M32_QUALIFIED_SOURCE_SHA256", ""
            )
            if (_fr13_dh_m32_live or _fr13_dh_m32_prod) and (
                len(_fr13_dh_source_sha) != 64
                or any(
                    value not in "0123456789abcdef"
                    for value in _fr13_dh_source_sha
                )
            ):
                raise RuntimeError(
                    "FR13 draft-head M32 requires its qualified source SHA-256"
                )
            self._fr13_dh_pad_rows = _fr13_dh_rows
            self._fr13_dh_ab_active = _fr13_dh_ab
            self._fr13_dh_m32_live_active = _fr13_dh_m32_live
            self._fr13_dh_m32_production_active = _fr13_dh_m32_prod
            self._fr13_dh_m32_selected_root_calls = 0
            self._fr13_dh_m32_selected_capture_calls = 0
            self._fr13_dh_m32_fallback_calls = 0
            self._fr13_dh_m32_graph_attestation = None
            self._fr13_dh_u8_live_active = _fr13_dh_u8_live
            self._fr13_dh_u8_production_active = _fr13_dh_u8_prod
            self._fr13_dh_u8_selected_root_calls = 0
            self._fr13_dh_u8_selected_capture_calls = 0
            self._fr13_dh_u8_fallback_calls = 0
            self._fr13_dh_u8_graph_attestation = None
            self._fr13_dh_m4_u8_live_active = _fr13_dh_m4_u8_live
            self._fr13_dh_m4_u8_production_active = _fr13_dh_m4_u8_prod
            self._fr13_dh_m4_u8_active = _fr13_dh_m4_u8_active
            self._fr13_dh_m4_u8_fallback_calls = 0
            self._fr13_dh_fp8_active = _fr13_dh_fp8
            self._fr13_dh_fp8_selected_root_calls = 0
            self._fr13_dh_fp8_selected_capture_calls = 0
            self._fr13_dh_fp8_fallback_calls = 0
            self._fr13_dh_fp8_graph_attestations = {}

            def _fr13_dvk_prepare():
                if (
                    _fr13_dvk_configured <= 0
                    or getattr(self, "_fr13_dvk_dead", False)
                ):
                    return 0, None
                _fr13_dvk_full = int(self.model.lm_head.weight.shape[0])
                if getattr(self, "_fr13_dvk_shim", None) is None:
                    try:
                        import types as _fr13_dvk_types
                        _fr13_dvk_lm = self.model.lm_head
                        # GATHER mode (FR13_DRAFT_VOCAB_BLOCKS=<json>): the
                        # subset is measured top-(K/128) 128-id blocks from
                        # our own trace corpus, not the contiguous id prefix.
                        # 128-block granularity keeps fp8 block-scale rows
                        # aligned; sliced argmax rows then need mapping back
                        # to real vocab ids via _fr13_dvk_map_t.
                        _fr13_dvk_bl = os.environ.get(
                            "FR13_DRAFT_VOCAB_BLOCKS", ""
                        )
                        _fr13_dvk_idx = None
                        _fr13_dvk_blk = None
                        if _fr13_dvk_bl:
                            import json as _fr13_dvk_json
                            with open(_fr13_dvk_bl) as _fr13_dvk_f:
                                _fr13_dvk_js = _fr13_dvk_json.load(_fr13_dvk_f)
                            _fr13_dvk_blk = torch.tensor(
                                _fr13_dvk_js["subsets"][
                                    str(_fr13_dvk_configured)
                                ],
                                dtype=torch.long,
                                device=_fr13_dvk_lm.weight.device,
                            )
                            _fr13_dvk_idx = (
                                _fr13_dvk_blk[:, None] * 128
                                + torch.arange(
                                    128, device=_fr13_dvk_blk.device
                                )[None, :]
                            ).reshape(-1)
                        _fr13_dvk_sh = _fr13_dvk_types.SimpleNamespace()
                        for _fr13_dvk_a in dir(_fr13_dvk_lm):
                            if _fr13_dvk_a.startswith("__"):
                                continue
                            _fr13_dvk_v = getattr(_fr13_dvk_lm, _fr13_dvk_a)
                            if isinstance(_fr13_dvk_v, torch.Tensor):
                                if (
                                    _fr13_dvk_v.dim() >= 1
                                    and _fr13_dvk_v.shape[0]
                                    >= _fr13_dvk_configured // 128
                                    and _fr13_dvk_v.shape[0] > 1
                                ):
                                    if _fr13_dvk_idx is not None:
                                        _fr13_dvk_v = _fr13_dvk_v.index_select(
                                            0,
                                            _fr13_dvk_idx
                                            if _fr13_dvk_v.shape[0]
                                            > _fr13_dvk_configured // 2
                                            else _fr13_dvk_blk,
                                        ).contiguous()
                                    else:
                                        _fr13_dvk_n = (
                                            _fr13_dvk_configured
                                            if _fr13_dvk_v.shape[0]
                                            > _fr13_dvk_configured // 2
                                            else _fr13_dvk_configured // 128
                                        )
                                        _fr13_dvk_v = _fr13_dvk_v[
                                            :_fr13_dvk_n
                                        ]
                                setattr(_fr13_dvk_sh, _fr13_dvk_a, _fr13_dvk_v)
                            elif not callable(_fr13_dvk_v):
                                setattr(_fr13_dvk_sh, _fr13_dvk_a, _fr13_dvk_v)
                        _fr13_dvk_sh.quant_method = _fr13_dvk_lm.quant_method
                        self._fr13_dvk_shim = _fr13_dvk_sh
                        self._fr13_dvk_map_t = _fr13_dvk_idx
                        print(
                            "[FR13_DRAFT_VOCAB] shim built "
                            f"K={_fr13_dvk_configured} "
                            f"(head rows {_fr13_dvk_lm.weight.shape[0]}"
                            f"->{_fr13_dvk_configured}) "
                            f"mode={'gather' if _fr13_dvk_idx is not None else 'contig'}",
                            flush=True,
                        )
                    except Exception as _fr13_dvk_e:
                        self._fr13_dvk_dead = True
                        print(
                            f"[FR13_DRAFT_VOCAB] DISABLED (shim build failed): {_fr13_dvk_e!r}",
                            flush=True,
                        )
                # FR14 ARM B -- DVK PHASE 1: dequantise the sliced NVFP4 rows
                # to BF16 at boot.
                #
                # Under the RadixArk aggressive checkpoint lm_head is a 4-tensor
                # ModelOpt NVFP4 set, so the shim above walks a QUANTIZED head
                # for the first time. Two facts make the slice itself correct
                # as-is, and they are worth stating because they are not
                # obvious:
                #
                #  1. The scales on disk are [out, in/16], so a row slice picks
                #     consistent rows of weight AND scale. But that is NOT the
                #     tensor we walk: FlashInferCutlassNvFp4LinearKernel
                #     .process_weights_after_loading replaces weight_scale with
                #     swizzle_blockscale(...), which keeps the logical shape
                #     [248320, 320] and interleaves rows via
                #     reshape(M/128,4,32,K/4,4).permute(0,1,4,3,2,5).
                #  2. That permutation never crosses a 128-ROW TILE BOUNDARY,
                #     and it is identical within every tile. The FR13 DVK block
                #     map's 128-id granularity -- chosen years earlier for fp8
                #     block-scale alignment -- is therefore exactly what makes
                #     the swizzled scale sliceable. Proven numerically at the
                #     real [248320, 320] shape, with a non-128-aligned control
                #     slice that correctly FAILS, in
                #     results/fr14_nvfp4_port_20260816/
                #     radixark_dvk_swizzle_check.py.
                #
                # So the index_select above is correct and no de-swizzle /
                # re-swizzle step is needed. What remains is that the FIVE K64
                # draft-head reads then feed BF16 GEMV units and a 128-block
                # map that are SEALED FR13 artifacts. Phase 1 keeps them
                # byte-identical by dequantising the 65,536 sliced rows once,
                # here, at boot: 671,088,640 B of BF16, exactly the number
                # scripts/fr13_hardware_floor_ledger.py pins as
                # SUBSET_HEAD_BYTES. (PHASE 2 -- reading those slices AS NVFP4
                # at 188,743,680 B, worth 8.834 ms -- needs an FP4 GEMV unit
                # and carries its own byte gate. It is NOT this.)
                #
                # FAIL-CLOSED ON PURPOSE, no _fr13_dvk_dead fallback: if this
                # cannot be done, the run would quietly fall back to reading
                # the FULL head five times, which is a different byte profile
                # from the pinned floor. A wrong number is worse than no boot.
                if (
                    not getattr(self, "_fr13_dvk_dead", False)
                    and getattr(self, "_fr13_dvk_shim", None) is not None
                    and not getattr(self, "_fr13_dvk_dequantised", False)
                    and hasattr(self._fr13_dvk_shim, "weight_scale")
                ):
                    _fr13_dvkq_sh = self._fr13_dvk_shim
                    # self.model.lm_head, not the local _fr13_dvk_lm: that
                    # name only exists on the branch that just BUILT the shim.
                    _fr13_dvkq_lm = self.model.lm_head
                    _fr13_dvkq_qm = type(
                        _fr13_dvkq_lm.quant_method
                    ).__name__
                    if _fr13_dvkq_qm != "ModelOptNvFp4LinearMethod":
                        raise RuntimeError(
                            "FR14 DVK dequant-at-slice is qualified only for "
                            "ModelOptNvFp4LinearMethod; the head resolved to "
                            f"{_fr13_dvkq_qm}"
                        )
                    _fr13_dvkq_w = _fr13_dvkq_sh.weight
                    _fr13_dvkq_s = _fr13_dvkq_sh.weight_scale
                    _fr13_dvkq_pad = int(
                        getattr(_fr13_dvkq_lm, "weights_padding_cols", 0)
                    )
                    # Every one of these is load-bearing for the arithmetic
                    # below, so none of them is allowed to be "probably fine".
                    # 248320 % 32 == 0 and 5120 % 32 == 0, so CUTLASS asks for
                    # no padding; a non-zero pad would silently add K columns
                    # the dequant would then materialise as real weights.
                    if (
                        _fr13_dvkq_pad != 0
                        or _fr13_dvkq_w.dtype != torch.uint8
                        or _fr13_dvkq_w.dim() != 2
                        or _fr13_dvkq_w.shape[0] != _fr13_dvk_configured
                        or _fr13_dvkq_w.shape[1] * 2 != 5120
                        or _fr13_dvkq_s.dim() != 2
                        or _fr13_dvkq_s.shape[0] != _fr13_dvk_configured
                        or _fr13_dvkq_s.shape[1] != 5120 // 16
                        or _fr13_dvk_configured % 128 != 0
                    ):
                        raise RuntimeError(
                            "FR14 DVK dequant-at-slice geometry contract "
                            f"failed: weight={tuple(_fr13_dvkq_w.shape)}"
                            f"/{_fr13_dvkq_w.dtype} "
                            f"scale={tuple(_fr13_dvkq_s.shape)}"
                            f"/{_fr13_dvkq_s.dtype} "
                            f"padding_cols={_fr13_dvkq_pad} "
                            f"K={_fr13_dvk_configured}"
                        )
                    _fr13_dvkq_gs = getattr(
                        _fr13_dvkq_lm, "weight_global_scale", None
                    )
                    if (
                        _fr13_dvkq_gs is None
                        or _fr13_dvkq_gs.numel() != 1
                    ):
                        raise RuntimeError(
                            "FR14 DVK dequant-at-slice needs the head's "
                            "weight_global_scale (ModelOpt renames "
                            "weight_scale_2 to it in "
                            "process_weights_after_loading)"
                        )
                    # vLLM's OWN dequant, not a reimplementation: this is the
                    # exact call EmulationNvFp4LinearKernel.apply_weights makes
                    # (nvfp4_emulation_utils.run_nvfp4_emulations), so the
                    # BF16 rows we produce are by construction the same numbers
                    # the NVFP4 GEMM is computing against. swizzle=True because
                    # the FlashInfer kernel swizzled the scale; the emulation
                    # kernel passes False only because it never swizzles.
                    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
                        dequantize_to_dtype as _fr13_dvkq_dequant,
                        kE2M1ToFloat_handle as _fr13_dvkq_lut,
                    )

                    # BIRTH DEFECT, first device run (boot probe
                    # 20260817T011303Z): break_fp4_bytes does
                    # `kE2M1[abs_vals]` where kE2M1 is a MODULE-LEVEL tensor
                    # that ships on CPU. The only thing that ever moves it is
                    # EmulationNvFp4LinearKernel.process_weights_after_loading
                    # -- and we run the FlashInfer kernel, so that hook never
                    # fires and the lookup table stays on CPU while the indices
                    # are on CUDA:
                    #   RuntimeError: indices should be either on cpu or on the
                    #   same device as the indexed tensor (cpu)
                    # Do exactly what the emulation kernel does, for the same
                    # reason it does it. Upstream's own comment says why it has
                    # to happen HERE and not inside the dequant: `.to(device)`
                    # is illegal during CUDA graph capture. _fr13_dvk_prepare
                    # runs at the first real propose, after capture finished --
                    # assert that rather than trust it, because a capture-time
                    # first call would fail in a much more confusing way.
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "FR14 DVK dequant-at-slice reached during CUDA "
                            "graph capture; the NVFP4 lookup table cannot be "
                            "moved to the device there"
                        )
                    if _fr13_dvkq_lut.val.device != _fr13_dvkq_w.device:
                        _fr13_dvkq_lut.val = _fr13_dvkq_lut.val.to(
                            _fr13_dvkq_w.device
                        )

                    # CHUNKED over 8192-row groups. break_fp4_bytes expands
                    # every packed byte into two int64 indices before the
                    # lookup, so a whole-head pass would peak at several GB of
                    # transient int64/float32 on a unified-memory pool that
                    # already holds the 46 GiB KV reservation. 8192 is a whole
                    # number of the 128-row tiles the swizzle inversion needs,
                    # so each chunk is independently de-swizzlable -- the same
                    # property that makes the row slice legal at all.
                    _fr13_dvkq_chunk = 8192
                    if _fr13_dvk_configured % _fr13_dvkq_chunk != 0:
                        raise RuntimeError(
                            "FR14 DVK dequant chunk size must divide K: "
                            f"K={_fr13_dvk_configured} "
                            f"chunk={_fr13_dvkq_chunk}"
                        )
                    _fr13_dvkq_bf16 = torch.empty(
                        (_fr13_dvk_configured, 5120),
                        dtype=torch.bfloat16,
                        device=_fr13_dvkq_w.device,
                    )
                    for _fr13_dvkq_lo in range(
                        0, _fr13_dvk_configured, _fr13_dvkq_chunk
                    ):
                        _fr13_dvkq_hi = _fr13_dvkq_lo + _fr13_dvkq_chunk
                        _fr13_dvkq_bf16[_fr13_dvkq_lo:_fr13_dvkq_hi] = (
                            _fr13_dvkq_dequant(
                                _fr13_dvkq_w.data[
                                    _fr13_dvkq_lo:_fr13_dvkq_hi
                                ].view(torch.uint8),
                                _fr13_dvkq_s.data[
                                    _fr13_dvkq_lo:_fr13_dvkq_hi
                                ],
                                _fr13_dvkq_gs,
                                torch.bfloat16,
                                16,
                                swizzle=True,
                            )
                        )
                    _fr13_dvkq_bf16 = _fr13_dvkq_bf16.contiguous()
                    if (
                        tuple(_fr13_dvkq_bf16.shape)
                        != (_fr13_dvk_configured, 5120)
                        or _fr13_dvkq_bf16.dtype != torch.bfloat16
                        or tuple(_fr13_dvkq_bf16.stride()) != (5120, 1)
                        or not _fr13_dvkq_bf16.is_contiguous()
                    ):
                        raise RuntimeError(
                            "FR14 DVK dequant produced "
                            f"{tuple(_fr13_dvkq_bf16.shape)}"
                            f"/{_fr13_dvkq_bf16.dtype}"
                            f"/{tuple(_fr13_dvkq_bf16.stride())}, not a "
                            f"contiguous BF16 [{_fr13_dvk_configured}, 5120]"
                        )
                    _fr13_dvkq_sh.weight = _fr13_dvkq_bf16
                    # Drop every quantisation-only attribute the shim inherited.
                    # Leaving a stale swizzled scale or an alpha next to a BF16
                    # weight is how a downstream reader silently applies a
                    # scale twice.
                    for _fr13_dvkq_a in (
                        "weight_scale",
                        "weight_scale_2",
                        "weight_global_scale",
                        "input_scale",
                        "input_global_scale",
                        "input_global_scale_inv",
                        "alpha",
                        "weights_padding_cols",
                        "marlin_input_dtype",
                    ):
                        if hasattr(_fr13_dvkq_sh, _fr13_dvkq_a):
                            delattr(_fr13_dvkq_sh, _fr13_dvkq_a)
                    # The shim is now a K64 BF16 head, so its declared widths
                    # must say so. output_size_per_partition and logical_widths
                    # were copied verbatim from the full head (248320) and are
                    # what a quant method would size its output by.
                    _fr13_dvkq_sh.output_size_per_partition = (
                        _fr13_dvk_configured
                    )
                    _fr13_dvkq_sh.logical_widths = [_fr13_dvk_configured]
                    _fr13_dvkq_sh.input_size_per_partition = 5120
                    if (
                        list(_fr13_dvkq_sh.logical_widths)
                        != [_fr13_dvkq_sh.output_size_per_partition]
                        or sum(_fr13_dvkq_sh.logical_widths)
                        != _fr13_dvkq_sh.weight.shape[0]
                    ):
                        raise RuntimeError(
                            "FR14 DVK shim widths are inconsistent with the "
                            "dequantised weight: "
                            f"logical_widths={_fr13_dvkq_sh.logical_widths} "
                            "output_size_per_partition="
                            f"{_fr13_dvkq_sh.output_size_per_partition} "
                            f"rows={_fr13_dvkq_sh.weight.shape[0]}"
                        )
                    # Hand the shim the unquantized method so
                    # `_sh.quant_method.apply(_sh, h, bias=None)` is a plain
                    # BF16 GEMM against the dequantised rows -- the same call
                    # the sealed FR13 sub-arms assert on by class NAME.
                    from vllm.model_executor.layers.vocab_parallel_embedding import (  # noqa: E501
                        UnquantizedEmbeddingMethod as _fr13_dvkq_um,
                    )

                    _fr13_dvkq_sh.quant_method = _fr13_dvkq_um()
                    self._fr13_dvk_dequantised = True
                    print(
                        "[FR14_DVK_DEQUANT] phase1 nvfp4->bf16 at slice "
                        f"K={_fr13_dvk_configured} "
                        f"packed_in={_fr13_dvkq_w.shape} "
                        f"swizzled_scale_in={_fr13_dvkq_s.shape} "
                        f"bf16_out={tuple(_fr13_dvkq_bf16.shape)} "
                        f"bytes={_fr13_dvkq_bf16.numel() * 2} "
                        f"logical_widths={_fr13_dvkq_sh.logical_widths} "
                        f"output_size_per_partition={_fr13_dvkq_sh.output_size_per_partition} "
                        "quant_method=UnquantizedEmbeddingMethod",
                        flush=True,
                    )
                if (
                    _fr13_dh_fp8
                    and not getattr(self, "_fr13_dh_fp8_ready", False)
                ):
                    _fr13_dh_fp8_sh = self._fr13_dvk_shim
                    _fr13_dh_fp8_w = _fr13_dh_fp8_sh.weight
                    if (
                        type(_fr13_dh_fp8_sh.quant_method).__name__
                        != "UnquantizedEmbeddingMethod"
                        or tuple(_fr13_dh_fp8_w.shape) != (65536, 5120)
                        or tuple(_fr13_dh_fp8_w.stride()) != (5120, 1)
                        or _fr13_dh_fp8_w.dtype != torch.bfloat16
                        or not _fr13_dh_fp8_w.is_contiguous()
                    ):
                        raise RuntimeError(
                            "FR13 draft-head FP8 requires contiguous BF16 "
                            "UnquantizedEmbeddingMethod "
                            "weight[65536,5120] stride[5120,1]"
                        )
                    if tuple(torch.cuda.get_device_capability()) != (12, 1):
                        raise RuntimeError(
                            "FR13 draft-head FP8 is qualified only on SM121"
                        )
                    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
                        CUTLASS_BLOCK_FP8_SUPPORTED as _fr13_dh_fp8_supported,
                    )
                    if not _fr13_dh_fp8_supported:
                        raise RuntimeError(
                            "FR13 draft-head FP8 requires vLLM CUTLASS "
                            "block-FP8 support"
                        )
                    from vllm.utils.deep_gemm import (
                        per_block_cast_to_fp8 as _fr13_dh_fp8_quant_weight,
                    )
                    (
                        self._fr13_dh_fp8_weight,
                        self._fr13_dh_fp8_weight_scale,
                    ) = _fr13_dh_fp8_quant_weight(
                        _fr13_dh_fp8_w,
                        block_size=[128, 128],
                        use_ue8m0=False,
                    )
                    _fr13_dh_fp8_qw = self._fr13_dh_fp8_weight
                    _fr13_dh_fp8_ws = self._fr13_dh_fp8_weight_scale
                    if (
                        tuple(_fr13_dh_fp8_qw.shape) != (65536, 5120)
                        or tuple(_fr13_dh_fp8_qw.stride()) != (5120, 1)
                        or not _fr13_dh_fp8_qw.is_contiguous()
                        or _fr13_dh_fp8_qw.dtype
                        != torch.float8_e4m3fn
                        or tuple(_fr13_dh_fp8_ws.shape) != (512, 40)
                        or tuple(_fr13_dh_fp8_ws.stride()) != (40, 1)
                        or _fr13_dh_fp8_ws.dtype != torch.float32
                        or not _fr13_dh_fp8_ws.is_contiguous()
                    ):
                        raise RuntimeError(
                            "FR13 draft-head FP8 weight quantization drifted "
                            "from qweight[65536,5120]/scale[512,40]"
                        )
                    if _fr13_dh_fp8_static_io:
                        from vllm.model_executor.layers.quantization.utils.quant_utils import (
                            get_fp8_min_max as _fr13_dh_fp8_min_max,
                        )

                        (
                            self._fr13_dh_fp8_min,
                            self._fr13_dh_fp8_max,
                        ) = _fr13_dh_fp8_min_max()
                        self._fr13_dh_fp8_weight_t = (
                            _fr13_dh_fp8_qw.t()
                        )
                        self._fr13_dh_fp8_weight_scale_t = (
                            _fr13_dh_fp8_ws.t()
                        )
                        self._fr13_dh_fp8_static_aq = (None,) + tuple(
                            torch.empty(
                                (_fr13_dh_b, 5120),
                                dtype=torch.float8_e4m3fn,
                                device=_fr13_dh_fp8_qw.device,
                            )
                            for _fr13_dh_b in range(1, 5)
                        )
                        self._fr13_dh_fp8_static_as = (None,) + tuple(
                            torch.empty(
                                (40, _fr13_dh_b),
                                dtype=torch.float32,
                                device=_fr13_dh_fp8_qw.device,
                            ).permute(1, 0)
                            for _fr13_dh_b in range(1, 5)
                        )
                        self._fr13_dh_fp8_static_out = (None,) + tuple(
                            torch.empty(
                                (_fr13_dh_b, 65536),
                                dtype=torch.bfloat16,
                                device=_fr13_dh_fp8_qw.device,
                            )
                            for _fr13_dh_b in range(1, 5)
                        )
                        if (
                            tuple(self._fr13_dh_fp8_weight_t.shape)
                            != (5120, 65536)
                            or tuple(self._fr13_dh_fp8_weight_t.stride())
                            != (1, 5120)
                            or tuple(
                                self._fr13_dh_fp8_weight_scale_t.shape
                            )
                            != (40, 512)
                            or tuple(
                                self._fr13_dh_fp8_weight_scale_t.stride()
                            )
                            != (1, 40)
                            or any(
                                tuple(
                                    self._fr13_dh_fp8_static_aq[
                                        _fr13_dh_b
                                    ].stride()
                                )
                                != (5120, 1)
                                or tuple(
                                    self._fr13_dh_fp8_static_as[
                                        _fr13_dh_b
                                    ].stride()
                                )
                                != (1, _fr13_dh_b)
                                or tuple(
                                    self._fr13_dh_fp8_static_out[
                                        _fr13_dh_b
                                    ].stride()
                                )
                                != (65536, 1)
                                for _fr13_dh_b in range(1, 5)
                            )
                        ):
                            raise RuntimeError(
                                "FR13 draft-head FP8 static I/O layout drifted"
                            )
                    self._fr13_dh_fp8_ready = True
                    print(
                        "[FR13_DRAFT_HEAD_FP8] static weight ready "
                        "qweight_shape=(65536,5120) "
                        "qweight_stride=(5120,1) "
                        "scale_shape=(512,40) scale_stride=(40,1) "
                        "block=(128,128) ue8m0=0 "
                        "retained_candidate_bytes=335626240 "
                        "calls_per_event=5 mandatory_event_bytes=1678131200",
                        flush=True,
                    )
                    if _fr13_dh_fp8_static_io:
                        print(
                            "[FR13_DRAFT_HEAD_FP8_STATIC_IO] ready "
                            "batches=1,2,3,4 raw_out_ops=1 "
                            "workspace_reused_across_root_and_loop=1",
                            flush=True,
                        )
                if (
                    _fr13_dh_u8_active
                    and not getattr(self, "_fr13_dh_u8_ready", False)
                ):
                    import hashlib as _fr13_dh_u8_hashlib
                    import pathlib as _fr13_dh_u8_pathlib

                    _fr13_dh_u8_sh = self._fr13_dvk_shim
                    _fr13_dh_u8_w = _fr13_dh_u8_sh.weight
                    if (
                        type(_fr13_dh_u8_sh.quant_method).__name__
                        != "UnquantizedEmbeddingMethod"
                        or tuple(_fr13_dh_u8_w.shape) != (65536, 5120)
                        or tuple(_fr13_dh_u8_w.stride()) != (5120, 1)
                        or _fr13_dh_u8_w.dtype != torch.bfloat16
                        or not _fr13_dh_u8_w.is_contiguous()
                        or tuple(torch.cuda.get_device_capability()) != (12, 1)
                    ):
                        raise RuntimeError(
                            "FR13 draft-head U8 requires SM121 and contiguous "
                            "BF16 UnquantizedEmbeddingMethod weight[65536,5120]"
                        )
                    _fr13_dh_u8_paths = {
                        "candidate_so_sha256": _fr13_dh_u8_pathlib.Path(
                            "/tmp/fr13_bf16_k64_m1_r64_u8.abi3.so"
                        ),
                        "candidate_source_sha256": _fr13_dh_u8_pathlib.Path(
                            "/workspace/csrc/"
                            "fr13_bf16_gemvx_k64_m1_shuffle_r64_u8.cu"
                        ),
                        "build_attestation_sha256": _fr13_dh_u8_pathlib.Path(
                            "/workspace/results/"
                            "fr13_fixed32_dfwd_k64_m1_r64_u8_"
                            "linked_build_20260805/build_attestation.json"
                        ),
                        "patch_source_sha256": _fr13_dh_u8_pathlib.Path(
                            "/workspace/scripts/"
                            "fr10_phase4_patch_vllm_tree_gdn.py"
                        ),
                        "runner_sha256": _fr13_dh_u8_pathlib.Path(
                            "/workspace/scripts/"
                            "fr13_run_b1_dfwd_k64_m1_r64_u8_live_gate.sh"
                        ),
                        "subset_sha256": _fr13_dh_u8_pathlib.Path(
                            "/workspace/config/fr13_fixed32/"
                            "subset_b1_diagnostic_one.json"
                        ),
                        "vocab_blocks_sha256": _fr13_dh_u8_pathlib.Path(
                            "/workspace/scripts/fr13_dvk_subset_blocks.json"
                        ),
                        "fa2_sha256": _fr13_dh_u8_pathlib.Path(
                            "/tmp/fr13_fork_fa2.so"
                        ),
                        "taw_source_sha256": _fr13_dh_u8_pathlib.Path(
                            "/workspace/scripts/fr13_device_multidraft_kernel.py"
                        ),
                    }
                    _fr13_dh_u8_env = {
                        "candidate_so_sha256": (
                            "FR13_DRAFT_HEAD_M1_R64_U8_SO_SHA256"
                        ),
                        "candidate_source_sha256": (
                            "FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_SHA256"
                        ),
                        "build_attestation_sha256": (
                            "FR13_DRAFT_HEAD_M1_R64_U8_BUILD_ATTESTATION_SHA256"
                        ),
                        "patch_source_sha256": (
                            "FR13_DRAFT_HEAD_M1_R64_U8_PATCH_SOURCE_SHA256"
                        ),
                        "runner_sha256": (
                            "FR13_DRAFT_HEAD_M1_R64_U8_RUNNER_SHA256"
                        ),
                        "subset_sha256": (
                            "FR13_DRAFT_HEAD_M1_R64_U8_SUBSET_SHA256"
                        ),
                        "vocab_blocks_sha256": (
                            "FR13_DRAFT_HEAD_M1_R64_U8_VOCAB_BLOCKS_SHA256"
                        ),
                        "fa2_sha256": (
                            "FR13_DRAFT_HEAD_M1_R64_U8_FA2_SHA256"
                        ),
                        "taw_source_sha256": (
                            "FR13_DRAFT_HEAD_M1_R64_U8_TAW_SOURCE_SHA256"
                        ),
                    }
                    _fr13_dh_u8_identities = {}
                    for (
                        _fr13_dh_u8_key,
                        _fr13_dh_u8_path,
                    ) in _fr13_dh_u8_paths.items():
                        _fr13_dh_u8_expected = os.environ.get(
                            _fr13_dh_u8_env[_fr13_dh_u8_key], ""
                        )
                        if (
                            len(_fr13_dh_u8_expected) != 64
                            or any(
                                _fr13_dh_u8_char
                                not in "0123456789abcdef"
                                for _fr13_dh_u8_char in _fr13_dh_u8_expected
                            )
                            or not _fr13_dh_u8_path.is_file()
                            or _fr13_dh_u8_path.is_symlink()
                            or _fr13_dh_u8_hashlib.sha256(
                                _fr13_dh_u8_path.read_bytes()
                            ).hexdigest()
                            != _fr13_dh_u8_expected
                        ):
                            raise RuntimeError(
                                "FR13 draft-head U8 input identity drifted: "
                                + _fr13_dh_u8_key
                            )
                        _fr13_dh_u8_identities[_fr13_dh_u8_key] = (
                            _fr13_dh_u8_expected
                        )
                    if (
                        _fr13_dh_u8_identities["candidate_so_sha256"]
                        != "8b27df4f3c6a5a0574261ee984159582a87615c3e6d83f2a267f4fa46a3e421e"
                        or _fr13_dh_u8_identities["candidate_source_sha256"]
                        != "af0044edd84ff58d353a816f6887894d05a62b221e0efa5af933c2c59676b01b"
                        or _fr13_dh_u8_identities[
                            "build_attestation_sha256"
                        ]
                        != "e7ec95d1fff3b665373ad7b3a14f7e3fad346cf77a5f2f992a90a689e5672c8f"
                        or _fr13_dh_u8_identities["subset_sha256"]
                        != "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
                        or _fr13_dh_u8_identities["vocab_blocks_sha256"]
                        != "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
                    ):
                        raise RuntimeError(
                            "FR13 draft-head U8 pinned qualification inputs drifted"
                        )
                    _fr13_dh_u8_source_commit = os.environ.get(
                        "FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_COMMIT", ""
                    )
                    _fr13_dh_u8_instance_id = os.environ.get(
                        "FR13_DRAFT_HEAD_M1_R64_U8_INSTANCE_ID", ""
                    )
                    if (
                        len(_fr13_dh_u8_source_commit) != 40
                        or any(
                            _fr13_dh_u8_char not in "0123456789abcdef"
                            for _fr13_dh_u8_char in _fr13_dh_u8_source_commit
                        )
                        or _fr13_dh_u8_instance_id
                        != "astropy__astropy-12907"
                    ):
                        raise RuntimeError(
                            "FR13 draft-head U8 source/task identity drifted"
                        )
                    _fr13_dh_u8_identities.update(
                        {
                            "source_commit": _fr13_dh_u8_source_commit,
                            "instance_id": _fr13_dh_u8_instance_id,
                            "candidate_so_bytes": 117904,
                        }
                    )
                    if _fr13_dh_u8_paths[
                        "candidate_so_sha256"
                    ].stat().st_size != 117904:
                        raise RuntimeError(
                            "FR13 draft-head U8 candidate binary size drifted"
                        )
                    if _fr13_dh_u8_prod:
                        _fr13_dh_u8_credential_path = (
                            _fr13_dh_u8_pathlib.Path(
                                os.environ.get(
                                    "FR13_DRAFT_HEAD_M1_R64_U8_"
                                    "PRODUCTION_PASS_SIDECAR",
                                    "",
                                )
                            )
                        )
                        _fr13_dh_u8_credential_sha = os.environ.get(
                            "FR13_DRAFT_HEAD_M1_R64_U8_"
                            "PRODUCTION_PASS_SIDECAR_SHA256",
                            "",
                        )
                        if (
                            str(_fr13_dh_u8_credential_path)
                            != "/logs/fr13_dfwd_k64_m1_r64_u8."
                            "production_credential.json"
                            or not _fr13_dh_u8_credential_path.is_file()
                            or _fr13_dh_u8_credential_path.is_symlink()
                            or len(_fr13_dh_u8_credential_sha) != 64
                            or _fr13_dh_u8_hashlib.sha256(
                                _fr13_dh_u8_credential_path.read_bytes()
                            ).hexdigest()
                            != _fr13_dh_u8_credential_sha
                        ):
                            raise RuntimeError(
                                "FR13 draft-head U8 production credential drifted"
                            )
                    torch.ops.load_library(
                        str(_fr13_dh_u8_paths["candidate_so_sha256"])
                    )
                    self._fr13_dh_u8_op = getattr(
                        torch.ops.fr13_bf16_k64_head,
                        "gemvx_m1_shuffle_r64_u8_out",
                    )
                    self._fr13_dh_u8_output = torch.empty(
                        (1, 65536),
                        dtype=torch.bfloat16,
                        device=_fr13_dh_u8_w.device,
                    )
                    if _fr13_dh_u8_live:
                        self._fr13_dh_u8_compares = torch.zeros(
                            (5,),
                            dtype=torch.int64,
                            device=_fr13_dh_u8_w.device,
                        )
                        self._fr13_dh_u8_mismatches = torch.zeros(
                            (5,),
                            dtype=torch.int64,
                            device=_fr13_dh_u8_w.device,
                        )
                        self._fr13_dh_u8_nonfinite = torch.zeros(
                            (5,),
                            dtype=torch.int64,
                            device=_fr13_dh_u8_w.device,
                        )
                        self._fr13_dh_u8_count_enable = torch.zeros(
                            (),
                            dtype=torch.int64,
                            device=_fr13_dh_u8_w.device,
                        )
                        from vllm.model_executor.layers.mamba import (
                            gdn_linear_attn as _fr13_dh_u8_gdn,
                        )

                        _fr13_dh_u8_contract_value = _fr13_dh_u8_contract()
                        _fr13_dh_u8_gdn._fr13_draft_head_u8_live_register(
                            self._fr13_dh_u8_compares,
                            self._fr13_dh_u8_mismatches,
                            self._fr13_dh_u8_nonfinite,
                            _fr13_dh_u8_contract_value["geometry"],
                            _fr13_dh_u8_contract_value["candidate"],
                            _fr13_dh_u8_identities,
                        )
                    self._fr13_dh_u8_ready = True
                    print(
                        "[FR13_DRAFT_HEAD_M1_R64_U8] ready "
                        + (
                            (
                                "quality_gate=1 candidate_served=1 "
                                "raw_bf16_drift_diagnostic=1 "
                                if _fr13_dh_u8_quality
                                else "shadow_only=1 incumbent_served=1 "
                            )
                            + "depths=root,1,2,3,4"
                            if _fr13_dh_u8_live
                            else "production=1 credential_attested=1 "
                            "preallocated_output=1"
                        ),
                        flush=True,
                    )
                if (
                    _fr13_dh_m4_u8_active
                    and not getattr(self, "_fr13_dh_m4_u8_ready", False)
                ):
                    import hashlib as _fr13_dh_m4_u8_hashlib
                    import pathlib as _fr13_dh_m4_u8_pathlib

                    _fr13_dh_m4_u8_sh = self._fr13_dvk_shim
                    _fr13_dh_m4_u8_w = _fr13_dh_m4_u8_sh.weight
                    if (
                        type(_fr13_dh_m4_u8_sh.quant_method).__name__
                        != "UnquantizedEmbeddingMethod"
                        or tuple(_fr13_dh_m4_u8_w.shape) != (65536, 5120)
                        or tuple(_fr13_dh_m4_u8_w.stride()) != (5120, 1)
                        or _fr13_dh_m4_u8_w.dtype != torch.bfloat16
                        or not _fr13_dh_m4_u8_w.is_contiguous()
                        or tuple(torch.cuda.get_device_capability()) != (12, 1)
                    ):
                        raise RuntimeError(
                            "FR13 draft-head M4 U8 requires SM121 and contiguous "
                            "BF16 UnquantizedEmbeddingMethod weight[65536,5120]"
                        )
                    _fr13_dh_m4_u8_paths = {
                        "candidate_so_sha256": _fr13_dh_m4_u8_pathlib.Path(
                            "/tmp/fr13_bf16_k64_m4_r64_u8.abi3.so"
                        ),
                        "candidate_source_sha256": _fr13_dh_m4_u8_pathlib.Path(
                            "/workspace/csrc/"
                            "fr13_bf16_gemvx_k64_m4_shuffle_r64_u8.cu"
                        ),
                        "build_attestation_sha256": _fr13_dh_m4_u8_pathlib.Path(
                            "/workspace/results/"
                            "fr13_fixed32_dfwd_k64_m4_r64_u8_"
                            "linked_build_20260805/build_attestation.json"
                        ),
                        "patch_source_sha256": _fr13_dh_m4_u8_pathlib.Path(
                            "/workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py"
                        ),
                        "runner_sha256": _fr13_dh_m4_u8_pathlib.Path(
                            "/workspace/scripts/"
                            "fr13_run_b4_dfwd_k64_m4_r64_u8_live_gate.sh"
                        ),
                        "subset_sha256": _fr13_dh_m4_u8_pathlib.Path(
                            "/workspace/config/fr13_fixed32/subset_b4_four.json"
                        ),
                        "vocab_blocks_sha256": _fr13_dh_m4_u8_pathlib.Path(
                            "/workspace/scripts/fr13_dvk_subset_blocks.json"
                        ),
                        "fa2_sha256": _fr13_dh_m4_u8_pathlib.Path(
                            "/tmp/fr13_fork_fa2.so"
                        ),
                        "taw_source_sha256": _fr13_dh_m4_u8_pathlib.Path(
                            "/workspace/scripts/fr13_device_multidraft_kernel.py"
                        ),
                    }
                    _fr13_dh_m4_u8_env = {
                        "candidate_so_sha256": "FR13_DRAFT_HEAD_M4_R64_U8_SO_SHA256",
                        "candidate_source_sha256": "FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_SHA256",
                        "build_attestation_sha256": "FR13_DRAFT_HEAD_M4_R64_U8_BUILD_ATTESTATION_SHA256",
                        "patch_source_sha256": "FR13_DRAFT_HEAD_M4_R64_U8_PATCH_SOURCE_SHA256",
                        "runner_sha256": "FR13_DRAFT_HEAD_M4_R64_U8_RUNNER_SHA256",
                        "subset_sha256": "FR13_DRAFT_HEAD_M4_R64_U8_SUBSET_SHA256",
                        "vocab_blocks_sha256": "FR13_DRAFT_HEAD_M4_R64_U8_VOCAB_BLOCKS_SHA256",
                        "fa2_sha256": "FR13_DRAFT_HEAD_M4_R64_U8_FA2_SHA256",
                        "taw_source_sha256": (
                            "FR13_DRAFT_HEAD_M4_R64_U8_TAW_SOURCE_SHA256"
                        ),
                    }
                    _fr13_dh_m4_u8_identities = {}
                    for (
                        _fr13_dh_m4_u8_key,
                        _fr13_dh_m4_u8_path,
                    ) in _fr13_dh_m4_u8_paths.items():
                        _fr13_dh_m4_u8_expected = os.environ.get(
                            _fr13_dh_m4_u8_env[_fr13_dh_m4_u8_key], ""
                        )
                        if (
                            len(_fr13_dh_m4_u8_expected) != 64
                            or any(
                                char not in "0123456789abcdef"
                                for char in _fr13_dh_m4_u8_expected
                            )
                            or not _fr13_dh_m4_u8_path.is_file()
                            or _fr13_dh_m4_u8_path.is_symlink()
                            or _fr13_dh_m4_u8_hashlib.sha256(
                                _fr13_dh_m4_u8_path.read_bytes()
                            ).hexdigest() != _fr13_dh_m4_u8_expected
                        ):
                            raise RuntimeError(
                                "FR13 draft-head M4 U8 input identity drifted: "
                                + _fr13_dh_m4_u8_key
                            )
                        _fr13_dh_m4_u8_identities[_fr13_dh_m4_u8_key] = (
                            _fr13_dh_m4_u8_expected
                        )
                    _fr13_dh_m4_u8_pins = {
                        "candidate_so_sha256": "6cb24782495ff1c1457ebbf9cbcfcd6ca7b372378d3b435f80054688432a365f",
                        "candidate_source_sha256": "a52361be1c9052a46509cc230ea320c4beb6d15f261327edc835d8da3ae00d9e",
                        "build_attestation_sha256": "b31ba7fb24fce81b0dceb97d77134f21107511e97538be15cb778c6ac4da5926",
                        "subset_sha256": "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5",
                        "vocab_blocks_sha256": "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff",
                        "fa2_sha256": "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d",
                    }
                    if any(
                        _fr13_dh_m4_u8_identities[key] != value
                        for key, value in _fr13_dh_m4_u8_pins.items()
                    ):
                        raise RuntimeError(
                            "FR13 draft-head M4 U8 pinned qualification inputs drifted"
                        )
                    _fr13_dh_m4_u8_source_commit = os.environ.get(
                        "FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_COMMIT", ""
                    )
                    _fr13_dh_m4_u8_task_ids = os.environ.get(
                        "FR13_DRAFT_HEAD_M4_R64_U8_TASK_IDS", ""
                    )
                    _fr13_dh_m4_u8_expected_tasks = (
                        "astropy__astropy-12907,astropy__astropy-13033,"
                        "astropy__astropy-13236,astropy__astropy-13398"
                    )
                    if (
                        len(_fr13_dh_m4_u8_source_commit) != 40
                        or any(
                            char not in "0123456789abcdef"
                            for char in _fr13_dh_m4_u8_source_commit
                        )
                        or _fr13_dh_m4_u8_task_ids
                        != _fr13_dh_m4_u8_expected_tasks
                        or _fr13_dh_m4_u8_paths[
                            "candidate_so_sha256"
                        ].stat().st_size != 134320
                    ):
                        raise RuntimeError(
                            "FR13 draft-head M4 U8 source/task/binary identity drifted"
                        )
                    _fr13_dh_m4_u8_identities.update(
                        {
                            "source_commit": _fr13_dh_m4_u8_source_commit,
                            "task_ids": _fr13_dh_m4_u8_task_ids.split(","),
                            "candidate_so_bytes": 134320,
                        }
                    )
                    if _fr13_dh_m4_u8_prod:
                        _fr13_dh_m4_u8_credential_path = (
                            _fr13_dh_m4_u8_pathlib.Path(
                                os.environ.get(
                                    "FR13_DRAFT_HEAD_M4_R64_U8_"
                                    "PRODUCTION_PASS_SIDECAR",
                                    "",
                                )
                            )
                        )
                        _fr13_dh_m4_u8_credential_sha = os.environ.get(
                            "FR13_DRAFT_HEAD_M4_R64_U8_"
                            "PRODUCTION_PASS_SIDECAR_SHA256",
                            "",
                        )
                        if (
                            str(_fr13_dh_m4_u8_credential_path)
                            != "/logs/fr13_dfwd_k64_m4_r64_u8."
                            "production_credential.json"
                            or not _fr13_dh_m4_u8_credential_path.is_file()
                            or _fr13_dh_m4_u8_credential_path.is_symlink()
                            or len(_fr13_dh_m4_u8_credential_sha) != 64
                            or _fr13_dh_m4_u8_hashlib.sha256(
                                _fr13_dh_m4_u8_credential_path.read_bytes()
                            ).hexdigest()
                            != _fr13_dh_m4_u8_credential_sha
                        ):
                            raise RuntimeError(
                                "FR13 draft-head M4 U8 production credential drifted"
                            )
                    torch.ops.load_library(
                        str(_fr13_dh_m4_u8_paths["candidate_so_sha256"])
                    )
                    self._fr13_dh_m4_u8_op = getattr(
                        torch.ops.fr13_bf16_k64_head,
                        "gemvx_m4_shuffle_r64_u8_out",
                    )
                    self._fr13_dh_m4_u8_output = torch.empty(
                        (4, 65536),
                        dtype=torch.bfloat16,
                        device=_fr13_dh_m4_u8_w.device,
                    )
                    if _fr13_dh_m4_u8_live:
                        self._fr13_dh_m4_u8_compares = torch.zeros(
                            (5,), dtype=torch.int64,
                            device=_fr13_dh_m4_u8_w.device,
                        )
                        self._fr13_dh_m4_u8_mismatches = torch.zeros(
                            (5,), dtype=torch.int64,
                            device=_fr13_dh_m4_u8_w.device,
                        )
                        self._fr13_dh_m4_u8_nonfinite = torch.zeros(
                            (5,), dtype=torch.int64,
                            device=_fr13_dh_m4_u8_w.device,
                        )
                        self._fr13_dh_m4_u8_count_enable = torch.zeros(
                            (), dtype=torch.int64,
                            device=_fr13_dh_m4_u8_w.device,
                        )
                        from vllm.model_executor.layers.mamba import (
                            gdn_linear_attn as _fr13_dh_m4_u8_gdn,
                        )
                        _fr13_dh_m4_u8_contract_value = _fr13_dh_m4_u8_contract()
                        _fr13_dh_m4_u8_gdn._fr13_draft_head_m4_u8_live_register(
                            self._fr13_dh_m4_u8_compares,
                            self._fr13_dh_m4_u8_mismatches,
                            self._fr13_dh_m4_u8_nonfinite,
                            _fr13_dh_m4_u8_contract_value["geometry"],
                            _fr13_dh_m4_u8_contract_value["candidate"],
                            _fr13_dh_m4_u8_identities,
                        )
                    self._fr13_dh_m4_u8_ready = True
                    print(
                        "[FR13_DRAFT_HEAD_M4_R64_U8] ready "
                        + (
                            "production=1 credential_attested=1 candidate_served=1"
                            if _fr13_dh_m4_u8_prod
                            else (
                                "quality_gate=1 candidate_served=1 taw_exact=1"
                                if _fr13_dh_m4_u8_quality
                                else "shadow_only=1 incumbent_served=1"
                            )
                        )
                        + " batch=4 depths=root,1,2,3,4",
                        flush=True,
                    )
                if (
                    (
                        _fr13_dh_rows
                        or _fr13_dh_ab
                        or _fr13_dh_m32_live
                        or _fr13_dh_m32_prod
                    )
                    and not getattr(self, "_fr13_dh_pad_ready", False)
                ):
                    _fr13_dh_sh = self._fr13_dvk_shim
                    _fr13_dh_w = _fr13_dh_sh.weight
                    if (
                        type(_fr13_dh_sh.quant_method).__name__
                        != "UnquantizedEmbeddingMethod"
                        or tuple(_fr13_dh_w.shape) != (65536, 5120)
                        or tuple(_fr13_dh_w.stride()) != (5120, 1)
                        or _fr13_dh_w.dtype != torch.bfloat16
                        or not _fr13_dh_w.is_contiguous()
                    ):
                        raise RuntimeError(
                            "FR13 draft-head M32 requires contiguous BF16 "
                            "UnquantizedEmbeddingMethod "
                            "weight[65536,5120] stride[5120,1]"
                        )
                    _fr13_dh_active_rows = tuple(
                        sorted(
                            (32, 64, 128)
                            if _fr13_dh_ab
                            else {
                                32
                                if (_fr13_dh_m32_live or _fr13_dh_m32_prod)
                                else _fr13_dh_rows
                            }
                        )
                    )
                    self._fr13_dh_pad_inputs = {
                        _fr13_dh_r: torch.empty(
                            (_fr13_dh_r, 5120),
                            dtype=torch.bfloat16,
                            device=_fr13_dh_w.device,
                        )
                        for _fr13_dh_r in _fr13_dh_active_rows
                    }
                    self._fr13_dh_pad_outputs = {
                        _fr13_dh_r: torch.empty(
                            (_fr13_dh_r, 65536),
                            dtype=torch.bfloat16,
                            device=_fr13_dh_w.device,
                        )
                        for _fr13_dh_r in _fr13_dh_active_rows
                    }
                    self._fr13_dh_ab_mismatches = torch.zeros(
                        (3,), dtype=torch.int64, device=_fr13_dh_w.device
                    )
                    self._fr13_dh_ab_compares = torch.zeros(
                        (3,), dtype=torch.int64, device=_fr13_dh_w.device
                    )
                    self._fr13_dh_ab_root_checks = 0
                    self._fr13_dh_pad_seen_eager = False
                    self._fr13_dh_pad_seen_capture_batches = set()
                    self._fr13_dh_pad_row_indices = {
                        (_fr13_dh_r, _fr13_dh_b): torch.tensor(
                            tuple(
                                _fr13_dh_i % _fr13_dh_b
                                for _fr13_dh_i in range(_fr13_dh_r)
                            ),
                            dtype=torch.long,
                            device=_fr13_dh_w.device,
                        )
                        for _fr13_dh_r in _fr13_dh_active_rows
                        for _fr13_dh_b in (2, 3, 4)
                    }
                    if _fr13_dh_m32_live:
                        self._fr13_dh_m32_live_count_enable = torch.zeros(
                            (), dtype=torch.int64, device=_fr13_dh_w.device
                        )
                        from vllm.model_executor.layers.mamba import (
                            gdn_linear_attn as _fr13_dh_live_gdn,
                        )

                        _fr13_dh_contract = _fr13_dh_m32_contract()
                        _fr13_dh_live_gdn._fr13_draft_head_m32_live_register(
                            self._fr13_dh_ab_compares,
                            self._fr13_dh_ab_mismatches,
                            _fr13_dh_contract["geometry"],
                            _fr13_dh_contract["candidate"],
                        )
                    self._fr13_dh_pad_ready = True
                    print(
                        "[FR13_DRAFT_HEAD_PAD] static buffers ready "
                        f"candidate_rows={_fr13_dh_rows} "
                        f"all_row_byte_ab={int(_fr13_dh_ab)} "
                        f"m32_live_ab={int(_fr13_dh_m32_live)} "
                        f"m32_production={int(_fr13_dh_m32_prod)} "
                        f"allocated_rows={_fr13_dh_active_rows} "
                        "vocab=65536 hidden=5120 "
                        "method=UnquantizedEmbeddingMethod",
                        flush=True,
                    )
                return _fr13_dvk_configured, _fr13_dvk_full

            def _fr13_dfwd_top3_prepare():
                if not _fr13_dfwd_top3:
                    return
                if getattr(self, "_fr13_dfwd_top3_ready", False):
                    return
                import hashlib as _fr13_top3_hashlib
                import pathlib as _fr13_top3_pathlib

                _fr13_top3_so = _fr13_top3_pathlib.Path(
                    "/tmp/fr13_dfwd_k64_top3.abi3.so"
                )
                _fr13_top3_expected = os.environ.get(
                    "FR13_DFWD_K64_TOP3_SHA256", ""
                )
                if (
                    len(_fr13_top3_expected) != 64
                    or any(
                        _fr13_top3_char not in "0123456789abcdef"
                        for _fr13_top3_char in _fr13_top3_expected
                    )
                    or not _fr13_top3_so.is_file()
                    or _fr13_top3_so.is_symlink()
                ):
                    raise RuntimeError(
                        "FR13 DFWD K64 top3 binary identity is missing"
                    )
                _fr13_top3_digest = _fr13_top3_hashlib.sha256()
                with _fr13_top3_so.open("rb") as _fr13_top3_handle:
                    for _fr13_top3_chunk in iter(
                        lambda: _fr13_top3_handle.read(1024 * 1024), b""
                    ):
                        _fr13_top3_digest.update(_fr13_top3_chunk)
                if _fr13_top3_digest.hexdigest() != _fr13_top3_expected:
                    raise RuntimeError(
                        "FR13 DFWD K64 top3 binary identity mismatch"
                    )
                _fr13_top3_map = getattr(self, "_fr13_dvk_map_t", None)
                if (
                    not torch.is_tensor(_fr13_top3_map)
                    or tuple(_fr13_top3_map.shape) != (65536,)
                    or tuple(_fr13_top3_map.stride()) != (1,)
                    or _fr13_top3_map.dtype != torch.int64
                    or not _fr13_top3_map.is_contiguous()
                ):
                    raise RuntimeError(
                        "FR13 DFWD K64 top3 requires the pinned gather ID map"
                    )
                torch.ops.load_library(str(_fr13_top3_so))
                self._fr13_dfwd_top3_op = (
                    torch.ops.fr13_dfwd_top3.mapped_top3_out
                )
                self._fr13_dfwd_top3_root_spine = torch.empty(
                    (1,), dtype=torch.int64, device=_fr13_top3_map.device
                )
                self._fr13_dfwd_top3_root_wide = torch.empty(
                    (1, 3), dtype=torch.int64, device=_fr13_top3_map.device
                )
                self._fr13_dfwd_top3_root_calls = 0
                self._fr13_dfwd_top3_capture_calls = 0
                self._fr13_dfwd_top3_ready = True
                print(
                    "[FR13_DFWD_K64_TOP3] ready B1 K64 mapped width3 "
                    "launches_per_head=1 graph_direct_outputs=1",
                    flush=True,
                )

            def _fr13_dfwd_top3_select(
                _logits, _id_map, _spine_output, _top3_output, _site
            ):
                if not getattr(self, "_fr13_dfwd_top3_ready", False):
                    raise RuntimeError(
                        "FR13 DFWD K64 top3 selected before static preparation"
                    )
                _fr13_top3_capturing = torch.cuda.is_current_stream_capturing()
                if (
                    tuple(_logits.shape) != (1, 65536)
                    or tuple(_logits.stride()) != (65536, 1)
                    or _logits.dtype != torch.bfloat16
                    or tuple(_id_map.shape) != (65536,)
                    or tuple(_id_map.stride()) != (1,)
                    or _id_map.dtype != torch.int64
                    or tuple(_spine_output.shape) != (1,)
                    or tuple(_spine_output.stride()) != (1,)
                    or _spine_output.dtype != torch.int64
                    or tuple(_top3_output.shape) != (1, 3)
                    or tuple(_top3_output.stride()) != (3, 1)
                    or _top3_output.dtype != torch.int64
                    or _logits.device != _id_map.device
                    or _spine_output.device != _logits.device
                    or _top3_output.device != _logits.device
                    or (_site == "root" and _fr13_top3_capturing)
                    or (_site == "loop" and not _fr13_top3_capturing)
                ):
                    raise RuntimeError(
                        "FR13 DFWD K64 top3 runtime geometry/lifecycle drifted"
                    )
                self._fr13_dfwd_top3_op(
                    _spine_output, _top3_output, _logits, _id_map
                )
                if _site == "root":
                    self._fr13_dfwd_top3_root_calls += 1
                else:
                    self._fr13_dfwd_top3_capture_calls += 1
                if not getattr(self, "_fr13_dfwd_top3_engaged", False):
                    self._fr13_dfwd_top3_engaged = True
                    print(
                        "[FR13_DFWD_K64_TOP3] engaged "
                        "stock_argmax_topk_map_copy=0",
                        flush=True,
                    )
                return _spine_output, _top3_output

            # ---- FR14_FUSED_DRAFT_TOPK (K0 full-vocabulary fused select) ----
            # One launch replaces `logits.argmax(-1)` + `torch.topk(logits, 3)`
            # on the SAME materialised logits row. Byte-exact by gate, not by
            # argument: results/fr14_nvfp4_port_20260816/
            # fr14_fused_draft_topk_probe.py runs 6 840 configurations at the
            # real geometry (V=248320, k=3, rows 1..4, five CTA counts),
            # including 320 planted exact-tie plateaus, and requires zero
            # raw-byte mismatches with a powered negative control.
            def _fr14_fused_topk_prepare(_vocab):
                if not _fr14_fused_topk:
                    return
                _fr14_ft_rows = int(batch_size)
                if _fr14_ft_rows in getattr(
                    self, "_fr14_fused_topk_buffers", {}
                ):
                    return
                import hashlib as _fr14_ft_hashlib
                import pathlib as _fr14_ft_pathlib

                _fr14_ft_so = _fr14_ft_pathlib.Path(
                    os.environ.get(
                        "FR14_FUSED_DRAFT_TOPK_SO",
                        "/tmp/fr14_dfwd_full_topk.abi3.so",
                    )
                )
                _fr14_ft_expected = os.environ.get(
                    "FR14_FUSED_DRAFT_TOPK_SHA256", ""
                )
                if (
                    len(_fr14_ft_expected) != 64
                    or any(
                        _fr14_ft_char not in "0123456789abcdef"
                        for _fr14_ft_char in _fr14_ft_expected
                    )
                    or not _fr14_ft_so.is_file()
                    or _fr14_ft_so.is_symlink()
                ):
                    raise RuntimeError(
                        "FR14 fused draft top-k binary identity is missing"
                    )
                _fr14_ft_digest = _fr14_ft_hashlib.sha256()
                with _fr14_ft_so.open("rb") as _fr14_ft_handle:
                    for _fr14_ft_chunk in iter(
                        lambda: _fr14_ft_handle.read(1024 * 1024), b""
                    ):
                        _fr14_ft_digest.update(_fr14_ft_chunk)
                if _fr14_ft_digest.hexdigest() != _fr14_ft_expected:
                    raise RuntimeError(
                        "FR14 fused draft top-k binary identity mismatch"
                    )
                if int(_vocab) != 248320:
                    raise RuntimeError(
                        "FR14 fused draft top-k is compiled for the pinned "
                        "248320 vocabulary only"
                    )
                if not getattr(self, "_fr14_fused_topk_ready", False):
                    torch.ops.load_library(str(_fr14_ft_so))
                    self._fr14_fused_topk_op = (
                        torch.ops.fr14_fused_draft_topk.select_out
                    )
                    self._fr14_fused_topk_buffers = {}
                    self._fr14_fused_topk_root_calls = 0
                    self._fr14_fused_topk_capture_calls = 0
                    self._fr14_fused_topk_ready = True
                _fr14_ft_dev = self.model.lm_head.weight.device
                # Static per-batch homes: the drafter graph bakes these
                # addresses, so they are allocated once per batch size and
                # never reallocated.
                self._fr14_fused_topk_buffers[_fr14_ft_rows] = (
                    torch.zeros(
                        int(
                            torch.ops.fr14_fused_draft_topk.scratch_numel(
                                _fr14_ft_rows, _fr14_fused_topk_blocks
                            )
                        ),
                        dtype=torch.int64,
                        device=_fr14_ft_dev,
                    ),
                    torch.empty(
                        (_fr14_ft_rows,),
                        dtype=torch.int64,
                        device=_fr14_ft_dev,
                    ),
                    torch.empty(
                        (_fr14_ft_rows, 3),
                        dtype=torch.int64,
                        device=_fr14_ft_dev,
                    ),
                )
                print(
                    "[FR14_FUSED_DRAFT_TOPK] ready K0 full-vocab width3 "
                    f"rows={_fr14_ft_rows} "
                    f"blocks={_fr14_fused_topk_blocks} "
                    "launches_per_head=1 stock_argmax_topk=0",
                    flush=True,
                )

            def _fr14_fused_topk_select(
                _logits, _spine_output, _wide_output, _site
            ):
                if not getattr(self, "_fr14_fused_topk_ready", False):
                    raise RuntimeError(
                        "FR14 fused draft top-k selected before static "
                        "preparation"
                    )
                _fr14_ft_rows = int(batch_size)
                _fr14_ft_scratch = self._fr14_fused_topk_buffers[
                    _fr14_ft_rows
                ][0]
                _fr14_ft_capturing = torch.cuda.is_current_stream_capturing()
                if (
                    tuple(_logits.shape) != (_fr14_ft_rows, 248320)
                    or tuple(_logits.stride()) != (248320, 1)
                    or _logits.dtype != torch.bfloat16
                    or tuple(_spine_output.shape) != (_fr14_ft_rows,)
                    or tuple(_spine_output.stride()) != (1,)
                    or _spine_output.dtype != torch.int64
                    or tuple(_wide_output.shape) != (_fr14_ft_rows, 3)
                    or tuple(_wide_output.stride()) != (3, 1)
                    or _wide_output.dtype != torch.int64
                    or _spine_output.device != _logits.device
                    or _wide_output.device != _logits.device
                    or _fr14_ft_scratch.device != _logits.device
                    or (_site == "root" and _fr14_ft_capturing)
                    or (_site == "loop" and not _fr14_ft_capturing)
                ):
                    raise RuntimeError(
                        "FR14 fused draft top-k runtime geometry/lifecycle "
                        "drifted"
                    )
                self._fr14_fused_topk_op(
                    _spine_output,
                    _wide_output,
                    _logits,
                    _fr14_ft_scratch,
                    _fr14_fused_topk_blocks,
                )
                if _site == "root":
                    self._fr14_fused_topk_root_calls += 1
                else:
                    self._fr14_fused_topk_capture_calls += 1
                if not getattr(self, "_fr14_fused_topk_engaged", False):
                    self._fr14_fused_topk_engaged = True
                    print(
                        "[FR14_FUSED_DRAFT_TOPK] engaged "
                        "stock_argmax_topk=0",
                        flush=True,
                    )
                return _spine_output, _wide_output

            def _fr13_dh_pad_logits(_sh, _h, _rows):
                _fr13_dh_batch = int(_h.shape[0]) if _h.ndim == 2 else 0
                if (
                    not getattr(self, "_fr13_dh_pad_ready", False)
                    or _rows not in (32, 64, 128)
                    or _fr13_dh_batch not in (1, 2, 3, 4)
                    or tuple(_h.shape) != (_fr13_dh_batch, 5120)
                    or tuple(_h.stride()) != (5120, 1)
                    or _h.dtype != torch.bfloat16
                    or _h.device != _sh.weight.device
                ):
                    raise RuntimeError(
                        "FR13 draft-head padding engaged outside exact B1-B4 "
                        "BF16 hidden[B,5120] stride[5120,1] contract"
                    )
                _fr13_dh_capturing = torch.cuda.is_current_stream_capturing()
                if (
                    _fr13_dh_rows
                    and not self._fr13_dh_pad_seen_eager
                    and _fr13_dh_capturing
                ):
                    raise RuntimeError(
                        "FR13 direct padded draft head reached capture before "
                        "one eager GEMM launch"
                    )
                _fr13_dh_in = self._fr13_dh_pad_inputs[_rows]
                _fr13_dh_out = self._fr13_dh_pad_outputs[_rows]
                if _fr13_dh_batch == 1:
                    _fr13_dh_in.copy_(_h.expand_as(_fr13_dh_in))
                else:
                    torch.index_select(
                        _h,
                        0,
                        self._fr13_dh_pad_row_indices[
                            (_rows, _fr13_dh_batch)
                        ],
                        out=_fr13_dh_in,
                    )
                torch.mm(_fr13_dh_in, _sh.weight.t(), out=_fr13_dh_out)
                if _fr13_dh_rows and not self._fr13_dh_pad_seen_eager:
                    print(
                        "[FR13_DRAFT_HEAD_PAD] engaged "
                        f"candidate_rows={_rows} "
                        f"source_rows={_fr13_dh_batch} eager_launch=1",
                        flush=True,
                    )
                    self._fr13_dh_pad_seen_eager = True
                if (
                    _fr13_dh_rows
                    and _fr13_dh_capturing
                    and _fr13_dh_batch
                    not in self._fr13_dh_pad_seen_capture_batches
                ):
                    print(
                        "[FR13_DRAFT_HEAD_PAD] captured "
                        f"candidate_rows={_rows} "
                        f"source_rows={_fr13_dh_batch}",
                        flush=True,
                    )
                    self._fr13_dh_pad_seen_capture_batches.add(
                        _fr13_dh_batch
                    )
                if (
                    tuple(_fr13_dh_in.stride()) != (5120, 1)
                    or tuple(_sh.weight.t().stride()) != (1, 5120)
                    or tuple(_fr13_dh_out.stride()) != (65536, 1)
                ):
                    raise RuntimeError(
                        "FR13 draft-head M32 operand strides drifted"
                    )
                return _fr13_dh_out[:_fr13_dh_batch]

            def _fr13_dh_m32_contract():
                return {
                    "geometry": {
                        "batch_size": 1,
                        "calls_per_event": 5,
                        "input_shape": [1, 5120],
                        "input_stride": [5120, 1],
                        "weight_shape": [65536, 5120],
                        "weight_stride": [5120, 1],
                        "weight_transpose_stride": [1, 5120],
                        "output_shape": [1, 65536],
                        "output_stride": [65536, 1],
                        "dtype": "torch.bfloat16",
                    },
                    "candidate": {
                        "method": "UnquantizedEmbeddingMethod",
                        "operation": (
                            "replicate hidden row to M32 then torch.mm "
                            "with weight.t"
                        ),
                        "gemm_mnk": [32, 65536, 5120],
                        "candidate_input_stride": [5120, 1],
                        "candidate_weight_transpose_stride": [1, 5120],
                        "candidate_output_stride": [65536, 1],
                        "served_rows": 1,
                    },
                }

            def _fr13_dh_u8_contract():
                return {
                    "geometry": {
                        "batch_size": 1,
                        "calls_per_event": 5,
                        "depths": ["root", 1, 2, 3, 4],
                        "input_shape": [1, 5120],
                        "input_stride": [5120, 1],
                        "weight_shape": [65536, 5120],
                        "weight_stride": [5120, 1],
                        "output_shape": [1, 65536],
                        "output_stride": [65536, 1],
                        "dtype": "torch.bfloat16",
                    },
                    "candidate": {
                        "operation": (
                            "fr13_bf16_k64_head::"
                            "gemvx_m1_shuffle_r64_u8_out"
                        ),
                        "device": "sm121",
                        "grid": [1024, 1, 1],
                        "block": [16, 64, 1],
                        "rows_per_cta": 64,
                        "lane_products": 320,
                        "unroll_steps": 8,
                        "single_accumulator": True,
                        "reduction_strides": [8, 4, 2, 1],
                        "served_rows": 1,
                        "shadow_only": bool(
                            _fr13_dh_u8_live and not _fr13_dh_u8_quality
                        ),
                    },
                }

            def _fr13_dh_m4_u8_contract():
                return {
                    "geometry": {
                        "batch_size": 4,
                        "calls_per_event": 5,
                        "depths": ["root", 1, 2, 3, 4],
                        "input_shape": [4, 5120],
                        "input_stride": [5120, 1],
                        "weight_shape": [65536, 5120],
                        "weight_stride": [5120, 1],
                        "output_shape": [4, 65536],
                        "output_stride": [65536, 1],
                        "dtype": "torch.bfloat16",
                    },
                    "candidate": {
                        "operation": (
                            "fr13_bf16_k64_head::"
                            "gemvx_m4_shuffle_r64_u8_out"
                        ),
                        "device": "sm121",
                        "grid": [1024, 1, 1],
                        "block": [16, 64, 1],
                        "rows_per_cta": 64,
                        "batch_rows": 4,
                        "weight_reuse": 4,
                        "lane_products_per_row": 320,
                        "unroll_steps": 8,
                        "independent_accumulators": 4,
                        "reduction_strides": [8, 4, 2, 1],
                        "served_rows": 4,
                        "shadow_only": bool(
                            _fr13_dh_m4_u8_live
                            and not _fr13_dh_m4_u8_quality
                        ),
                    },
                }

            def _fr13_dh_fp8_contract():
                return {
                    "geometry": {
                        "calls_per_event": 5,
                        "input_hidden": 5120,
                        "vocab_rows": 65536,
                        "weight_shape": [65536, 5120],
                        "weight_stride": [5120, 1],
                        "weight_scale_shape": [512, 40],
                        "weight_scale_stride": [40, 1],
                        "weight_block": [128, 128],
                        "activation_group": 128,
                    },
                    "candidate": {
                        "operation": "vllm_cutlass_block_fp8_scaled_mm",
                        "device": "sm121",
                        "weight_dtype_bytes": 1,
                        "weight_scale_dtype": "torch.float32",
                        "activation_scale_layout": "column_major",
                        "use_ue8m0": False,
                        "output_dtype": "torch.bfloat16",
                        "proposal_logits_source": "fp8_output_direct",
                        "bf16_shadow_calls": 0,
                        "activation_io": (
                            "static_preallocated_raw_out_ops"
                            if _fr13_dh_fp8_static_io
                            else "wrapper_allocated"
                        ),
                    },
                    "traffic": {
                        "bf16_weight_bytes_per_call_removed": 671088640,
                        "fp8_weight_bytes_per_call": 335544320,
                        "fp32_weight_scale_bytes_per_call": 81920,
                        "mandatory_bytes_per_call": 335626240,
                        "mandatory_bytes_per_event": 1678131200,
                        "retained_candidate_bytes": 335626240,
                        "baseline_mandatory_bytes_per_event": 32666638208,
                        "candidate_mandatory_bytes_per_event": 30989326208,
                        "floor_bandwidth_gbps": 273,
                        "candidate_weight_floor_ms": 113.514015414,
                        "one_sided_u95_cap_ms": 130.541117726,
                    },
                }

            def _fr13_dh_m32_atomic_json(_path, _payload):
                import json as _fr13_dh_json
                import pathlib as _fr13_dh_pathlib

                _fr13_dh_out_path = _fr13_dh_pathlib.Path(_path)
                _fr13_dh_out_path.parent.mkdir(parents=True, exist_ok=True)
                _fr13_dh_temp = _fr13_dh_out_path.with_name(
                    _fr13_dh_out_path.name
                    + ".tmp."
                    + str(os.getpid())
                )
                _fr13_dh_temp.write_text(
                    _fr13_dh_json.dumps(
                        _payload,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="ascii",
                )
                _fr13_dh_temp.replace(_fr13_dh_out_path)

            def _fr13_dh_m32_measured_proposal():
                from vllm.model_executor.layers.mamba import (
                    gdn_linear_attn as _fr13_dh_gdn,
                )

                _fr13_dh_proposal = getattr(
                    _fr13_dh_gdn,
                    "_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT",
                    None,
                )
                if (
                    not isinstance(_fr13_dh_proposal, dict)
                    or _fr13_dh_proposal.get("measured")
                    not in (True, False)
                    or int(_fr13_dh_proposal.get("batch_size", -1)) != 1
                    or _fr13_dh_proposal.get("mode")
                    != _fr13_dh_gdn._FR13_FIXED32_MODE
                ):
                    raise RuntimeError(
                        "FR13 draft-head M32 has no exact-B1 proposal"
                    )
                return _fr13_dh_proposal["measured"] is True

            def _fr13_dh_fp8_measured_proposal(_batch_size):
                from vllm.model_executor.layers.mamba import (
                    gdn_linear_attn as _fr13_dh_fp8_gdn,
                )

                _fr13_dh_fp8_proposal = getattr(
                    _fr13_dh_fp8_gdn,
                    "_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT",
                    None,
                )
                if (
                    not isinstance(_fr13_dh_fp8_proposal, dict)
                    or _fr13_dh_fp8_proposal.get("measured")
                    not in (True, False)
                    or int(_batch_size) not in (1, 2, 3, 4)
                    or int(
                        _fr13_dh_fp8_proposal.get("batch_size", -1)
                    )
                    != int(_batch_size)
                    or _fr13_dh_fp8_proposal.get("mode")
                    != _fr13_dh_fp8_gdn._FR13_FIXED32_MODE
                ):
                    raise RuntimeError(
                        "FR13 draft-head FP8 has no exact fixed32 proposal"
                    )
                return (
                    _fr13_dh_fp8_proposal["measured"] is True,
                    _fr13_dh_fp8_proposal,
                )

            def _fr13_dh_m32_note_production(_capturing):
                if getattr(
                    self, "_fr13_dh_m32_engagement_written", False
                ):
                    return
                if _capturing:
                    self._fr13_dh_m32_selected_capture_calls += 1
                else:
                    self._fr13_dh_m32_selected_root_calls += 1
                if (
                    self._fr13_dh_m32_selected_root_calls > 1
                    or self._fr13_dh_m32_selected_capture_calls > 4
                ):
                    raise RuntimeError(
                        "FR13 draft-head M32 selection exceeded its first "
                        "root/capture lifecycle before replay engagement"
                    )

            def _fr13_dh_m32_note_production_replay(
                _graph_id, _graph_signature, _batch_size
            ):
                if not getattr(
                    self, "_fr13_dh_m32_production_active", False
                ):
                    return
                if getattr(
                    self, "_fr13_dh_m32_engagement_written", False
                ):
                    return
                from vllm.model_executor.layers.mamba import (
                    gdn_linear_attn as _fr13_dh_replay_gdn,
                )

                _fr13_dh_proposal = getattr(
                    _fr13_dh_replay_gdn,
                    "_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT",
                    None,
                )
                _fr13_dh_signature = str(_graph_signature)
                _fr13_dh_lifecycle = getattr(
                    _fr13_dh_replay_gdn,
                    "_FR13_FIXED32_DRAFTER_GRAPH_LIFECYCLE",
                    {},
                ).get(int(_graph_id))
                if (
                    not isinstance(_fr13_dh_proposal, dict)
                    or int(_batch_size) != 1
                    or int(_graph_id) <= 0
                    or int(_fr13_dh_proposal.get("graph_id", -1))
                    != int(_graph_id)
                    or _fr13_dh_proposal.get("graph_signature")
                    != _fr13_dh_signature
                    or int(_fr13_dh_proposal.get("graph_replays", -1)) != 1
                    or self._fr13_dh_m32_selected_root_calls != 1
                    or self._fr13_dh_m32_fallback_calls != 0
                    or not isinstance(_fr13_dh_lifecycle, dict)
                    or int(_fr13_dh_lifecycle.get("captures", -1)) != 1
                    or int(_fr13_dh_lifecycle.get("batch_size", -1)) != 1
                    or _fr13_dh_lifecycle.get("graph_signature")
                    != _fr13_dh_signature
                    or _fr13_dh_lifecycle.get("capture_origin")
                    not in ("measured", "unmeasured")
                    or _fr13_dh_signature
                    != (
                        "d9a4ddece41d146e9949b9f8ff7c2603"
                        "b8948d157b28ef69244e44469b36150c"
                    )
                ):
                    raise RuntimeError(
                        "FR13 draft-head M32 production replay engagement "
                        "drifted"
                    )
                _fr13_dh_graph_attestation = getattr(
                    self, "_fr13_dh_m32_graph_attestation", None
                )
                if _fr13_dh_graph_attestation is None:
                    if self._fr13_dh_m32_selected_capture_calls != 4:
                        raise RuntimeError(
                            "FR13 draft-head M32 graph capture did not select "
                            "four loop heads"
                        )
                    _fr13_dh_graph_attestation = {
                        "graph_id": int(_graph_id),
                        "graph_signature": _fr13_dh_signature,
                        "capture_origin": _fr13_dh_lifecycle[
                            "capture_origin"
                        ],
                    }
                    self._fr13_dh_m32_graph_attestation = (
                        _fr13_dh_graph_attestation
                    )
                elif (
                    self._fr13_dh_m32_selected_capture_calls != 0
                    or _fr13_dh_graph_attestation.get("graph_id")
                    != int(_graph_id)
                    or _fr13_dh_graph_attestation.get("graph_signature")
                    != _fr13_dh_signature
                    or _fr13_dh_graph_attestation.get("capture_origin")
                    != _fr13_dh_lifecycle.get("capture_origin")
                ):
                    raise RuntimeError(
                        "FR13 draft-head M32 replay left its attested graph"
                    )
                self._fr13_dh_m32_selected_root_calls = 0
                self._fr13_dh_m32_selected_capture_calls = 0
                if _fr13_dh_proposal.get("measured") is not True:
                    return
                if int(_fr13_dh_lifecycle.get("measured_replays", 0)) < 1:
                    raise RuntimeError(
                        "FR13 draft-head M32 engagement lacks a measured replay"
                    )
                _fr13_dh_contract = _fr13_dh_m32_contract()
                _fr13_dh_m32_atomic_json(
                    os.environ.get(
                        "FR13_DRAFT_HEAD_M32_PRODUCTION_ENGAGEMENT_JSON",
                        "/logs/fr13_draft_head_m32.production_engagement.json",
                    ),
                    {
                        "schema": (
                            "fr13.fixed32."
                            "draft_head_m32_production_engagement.v1"
                        ),
                        "status": "ENGAGED",
                        "source_commit": os.environ.get(
                            "FR13_DRAFT_HEAD_M32_SOURCE_COMMIT", ""
                        ),
                        "candidate_source_sha256": _fr13_dh_source_sha,
                        "production_pass_sidecar_sha256": os.environ.get(
                            "FR13_DRAFT_HEAD_M32_"
                            "PRODUCTION_PASS_SIDECAR_SHA256",
                            "",
                        ),
                        "geometry": _fr13_dh_contract["geometry"],
                        "candidate": _fr13_dh_contract["candidate"],
                        "selected_root_calls": 1,
                        "captured_loop_calls": 4,
                        "fallback_calls": (
                            self._fr13_dh_m32_fallback_calls
                        ),
                        "drafter_graph_id": int(_graph_id),
                        "drafter_graph_signature": _fr13_dh_signature,
                        "observed_measured_replays_at_least": 1,
                        "capture_origin": _fr13_dh_graph_attestation[
                            "capture_origin"
                        ],
                        "execution_basis": "cudagraph_replay",
                        "forward_step_index": int(
                            _fr13_dh_proposal["forward_step_index"]
                        ),
                        "runtime_mode": "FULL",
                    },
                )
                self._fr13_dh_m32_engagement_written = True

            def _fr13_dh_u8_note_production():
                if getattr(
                    self, "_fr13_dh_u8_engagement_written", False
                ):
                    return
                from vllm.model_executor.layers.mamba import (
                    gdn_linear_attn as _fr13_dh_u8_selection_gdn,
                )

                _fr13_dh_u8_capturing = getattr(
                    _fr13_dh_u8_selection_gdn,
                    "_fr13_fixed32_drafter_u8_head_selection",
                )(1)
                if _fr13_dh_u8_capturing:
                    self._fr13_dh_u8_selected_capture_calls += 1
                else:
                    self._fr13_dh_u8_selected_root_calls += 1
                if (
                    self._fr13_dh_u8_selected_root_calls > 1
                    or self._fr13_dh_u8_selected_capture_calls > 4
                ):
                    raise RuntimeError(
                        "FR13 draft-head U8 selection exceeded its first "
                        "root/capture lifecycle before replay engagement"
                    )

            def _fr13_dh_u8_note_production_replay(
                _graph_id, _graph_signature, _batch_size
            ):
                if not getattr(
                    self, "_fr13_dh_u8_production_active", False
                ):
                    return
                if getattr(
                    self, "_fr13_dh_u8_engagement_written", False
                ):
                    return
                from vllm.model_executor.layers.mamba import (
                    gdn_linear_attn as _fr13_dh_u8_replay_gdn,
                )

                _fr13_dh_u8_proposal = getattr(
                    _fr13_dh_u8_replay_gdn,
                    "_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT",
                    None,
                )
                _fr13_dh_u8_signature = str(_graph_signature)
                _fr13_dh_u8_lifecycle = getattr(
                    _fr13_dh_u8_replay_gdn,
                    "_FR13_FIXED32_DRAFTER_GRAPH_LIFECYCLE",
                    {},
                ).get(int(_graph_id))
                if (
                    not isinstance(_fr13_dh_u8_proposal, dict)
                    or int(_batch_size) != 1
                    or int(_graph_id) <= 0
                    or int(_fr13_dh_u8_proposal.get("graph_id", -1))
                    != int(_graph_id)
                    or _fr13_dh_u8_proposal.get("graph_signature")
                    != _fr13_dh_u8_signature
                    or int(_fr13_dh_u8_proposal.get("graph_replays", -1))
                    != 1
                    or self._fr13_dh_u8_selected_root_calls != 1
                    or self._fr13_dh_u8_fallback_calls != 0
                    or not isinstance(_fr13_dh_u8_lifecycle, dict)
                    or int(_fr13_dh_u8_lifecycle.get("captures", -1)) != 1
                    or int(_fr13_dh_u8_lifecycle.get("batch_size", -1)) != 1
                    or _fr13_dh_u8_lifecycle.get("graph_signature")
                    != _fr13_dh_u8_signature
                    or _fr13_dh_u8_lifecycle.get("capture_origin")
                    not in ("measured", "unmeasured")
                    or _fr13_dh_u8_signature
                    != (
                        "d9a4ddece41d146e9949b9f8ff7c2603"
                        "b8948d157b28ef69244e44469b36150c"
                    )
                ):
                    raise RuntimeError(
                        "FR13 draft-head U8 production replay engagement drifted"
                    )
                _fr13_dh_u8_expected_capture_calls = int(
                    _fr13_dh_u8_lifecycle.get("mtp_forward_calls", -1)
                )
                _fr13_dh_u8_lifecycle_capture_calls = int(
                    _fr13_dh_u8_lifecycle.get("draft_head_u8_calls", -1)
                )
                _fr13_dh_u8_attestation = getattr(
                    self, "_fr13_dh_u8_graph_attestation", None
                )
                if _fr13_dh_u8_attestation is None:
                    if (
                        _fr13_dh_u8_expected_capture_calls != 4
                        or _fr13_dh_u8_lifecycle_capture_calls != 4
                        or self._fr13_dh_u8_selected_capture_calls != 4
                    ):
                        raise RuntimeError(
                            "FR13 draft-head U8 graph capture head count drifted"
                        )
                    _fr13_dh_u8_attestation = {
                        "graph_id": int(_graph_id),
                        "graph_signature": _fr13_dh_u8_signature,
                        "captured_loop_calls": 4,
                        "capture_origin": _fr13_dh_u8_lifecycle[
                            "capture_origin"
                        ],
                    }
                    self._fr13_dh_u8_graph_attestation = (
                        _fr13_dh_u8_attestation
                    )
                elif (
                    self._fr13_dh_u8_selected_capture_calls != 0
                    or _fr13_dh_u8_attestation.get("graph_id")
                    != int(_graph_id)
                    or _fr13_dh_u8_attestation.get("graph_signature")
                    != _fr13_dh_u8_signature
                    or _fr13_dh_u8_attestation.get("captured_loop_calls") != 4
                    or _fr13_dh_u8_attestation.get("capture_origin")
                    != _fr13_dh_u8_lifecycle.get("capture_origin")
                ):
                    raise RuntimeError(
                        "FR13 draft-head U8 replay left its attested graph"
                    )
                self._fr13_dh_u8_selected_root_calls = 0
                self._fr13_dh_u8_selected_capture_calls = 0
                if _fr13_dh_u8_proposal.get("measured") is not True:
                    return
                if int(
                    _fr13_dh_u8_lifecycle.get("measured_replays", 0)
                ) < 1:
                    raise RuntimeError(
                        "FR13 draft-head U8 engagement lacks a measured replay"
                    )
                _fr13_dh_u8_contract_value = _fr13_dh_u8_contract()
                _fr13_dh_m32_atomic_json(
                    os.environ.get(
                        "FR13_DRAFT_HEAD_M1_R64_U8_"
                        "PRODUCTION_ENGAGEMENT_JSON",
                        "/logs/fr13_dfwd_k64_m1_r64_u8."
                        "production_engagement.json",
                    ),
                    {
                        "schema": (
                            "fr13.fixed32.dfwd_k64_m1_r64_u8_"
                            "production_engagement.v1"
                        ),
                        "status": "ENGAGED",
                        "source_commit": os.environ.get(
                            "FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_COMMIT", ""
                        ),
                        "candidate_so_sha256": os.environ.get(
                            "FR13_DRAFT_HEAD_M1_R64_U8_SO_SHA256", ""
                        ),
                        "candidate_source_sha256": os.environ.get(
                            "FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_SHA256", ""
                        ),
                        "production_credential_sha256": os.environ.get(
                            "FR13_DRAFT_HEAD_M1_R64_U8_"
                            "PRODUCTION_PASS_SIDECAR_SHA256",
                            "",
                        ),
                        "geometry": _fr13_dh_u8_contract_value["geometry"],
                        "qualification_candidate": (
                            _fr13_dh_u8_contract_value["candidate"]
                        ),
                        "selector": "fr13_bf16_k64_m1_r64_u8_direct",
                        "selected_root_calls": 1,
                        "captured_loop_calls": 4,
                        "fallback_calls": 0,
                        "drafter_graph_id": int(_graph_id),
                        "drafter_graph_signature": _fr13_dh_u8_signature,
                        "observed_measured_replays_at_least": 1,
                        "capture_origin": _fr13_dh_u8_attestation[
                            "capture_origin"
                        ],
                        "execution_basis": "cudagraph_replay",
                        "forward_step_index": int(
                            _fr13_dh_u8_proposal["forward_step_index"]
                        ),
                        "runtime_mode": "FULL",
                        "candidate_served": True,
                        "incumbent_head_calls": 0,
                        "steady_state_synchronizations": 0,
                        "performance_claim": False,
                    },
                )
                self._fr13_dh_u8_engagement_written = True

            def _fr13_dh_fp8_note_selection(_batch_size):
                if getattr(
                    self, "_fr13_dh_fp8_engagement_written", False
                ):
                    return
                if int(_batch_size) not in (1, 2, 3, 4):
                    raise RuntimeError(
                        "FR13 draft-head FP8 selection left B1-B4"
                    )
                from vllm.model_executor.layers.mamba import (
                    gdn_linear_attn as _fr13_dh_fp8_selection_gdn,
                )

                _fr13_dh_fp8_classify = getattr(
                    _fr13_dh_fp8_selection_gdn,
                    "_fr13_fixed32_drafter_fp8_head_selection",
                )
                _capturing = _fr13_dh_fp8_classify(_batch_size)
                if _capturing:
                    self._fr13_dh_fp8_selected_capture_calls += 1
                else:
                    self._fr13_dh_fp8_selected_root_calls += 1
                if (
                    self._fr13_dh_fp8_selected_root_calls > 1
                    or self._fr13_dh_fp8_selected_capture_calls > 4
                ):
                    raise RuntimeError(
                        "FR13 draft-head FP8 selection exceeded its first "
                        "root/capture lifecycle before replay engagement"
                    )

            def _fr13_dh_fp8_note_replay(
                _graph_id, _graph_signature, _batch_size
            ):
                if not getattr(self, "_fr13_dh_fp8_active", False):
                    return
                if getattr(
                    self, "_fr13_dh_fp8_engagement_written", False
                ):
                    return
                from vllm.model_executor.layers.mamba import (
                    gdn_linear_attn as _fr13_dh_fp8_replay_gdn,
                )

                (
                    _fr13_dh_fp8_measured,
                    _fr13_dh_fp8_proposal,
                ) = _fr13_dh_fp8_measured_proposal(_batch_size)
                _fr13_dh_fp8_signature = str(_graph_signature)
                _fr13_dh_fp8_lifecycle = getattr(
                    _fr13_dh_fp8_replay_gdn,
                    "_FR13_FIXED32_DRAFTER_GRAPH_LIFECYCLE",
                    {},
                ).get(int(_graph_id))
                if (
                    int(_graph_id) <= 0
                    or int(_fr13_dh_fp8_proposal.get("graph_id", -1))
                    != int(_graph_id)
                    or _fr13_dh_fp8_proposal.get("graph_signature")
                    != _fr13_dh_fp8_signature
                    or int(
                        _fr13_dh_fp8_proposal.get("graph_replays", -1)
                    )
                    != 1
                    or self._fr13_dh_fp8_selected_root_calls != 1
                    or self._fr13_dh_fp8_fallback_calls != 0
                    or not isinstance(_fr13_dh_fp8_lifecycle, dict)
                    or int(
                        _fr13_dh_fp8_lifecycle.get("captures", -1)
                    )
                    != 1
                    or int(
                        _fr13_dh_fp8_lifecycle.get("batch_size", -1)
                    )
                    != int(_batch_size)
                    or _fr13_dh_fp8_lifecycle.get("graph_signature")
                    != _fr13_dh_fp8_signature
                    or _fr13_dh_fp8_lifecycle.get("capture_origin")
                    not in ("measured", "unmeasured")
                    or _fr13_dh_fp8_signature
                    != (
                        "d9a4ddece41d146e9949b9f8ff7c2603"
                        "b8948d157b28ef69244e44469b36150c"
                    )
                ):
                    raise RuntimeError(
                        "FR13 draft-head FP8 replay engagement drifted"
                    )
                _fr13_dh_fp8_attestation = (
                    self._fr13_dh_fp8_graph_attestations.get(
                        int(_graph_id)
                    )
                )
                _fr13_dh_fp8_expected_capture_calls = int(
                    _fr13_dh_fp8_lifecycle.get(
                        "mtp_forward_calls", -1
                    )
                )
                _fr13_dh_fp8_lifecycle_capture_calls = int(
                    _fr13_dh_fp8_lifecycle.get(
                        "draft_head_fp8_calls", -1
                    )
                )
                _fr13_dh_fp8_observed_capture_calls = int(
                    self._fr13_dh_fp8_selected_capture_calls
                )
                if _fr13_dh_fp8_attestation is None:
                    # FULL capture can run on a sibling Eagle object; the
                    # authenticated GDN lifecycle remains process-global.
                    if (
                        _fr13_dh_fp8_expected_capture_calls != 4
                        or _fr13_dh_fp8_lifecycle_capture_calls
                        != _fr13_dh_fp8_expected_capture_calls
                        or _fr13_dh_fp8_observed_capture_calls
                        not in (
                            0,
                            _fr13_dh_fp8_lifecycle_capture_calls,
                        )
                    ):
                        raise RuntimeError(
                            "FR13 draft-head FP8 graph capture head count "
                            "drifted: observed_object_local="
                            f"{_fr13_dh_fp8_observed_capture_calls} "
                            "observed_lifecycle="
                            f"{_fr13_dh_fp8_lifecycle_capture_calls} "
                            "expected_from_mtp="
                            f"{_fr13_dh_fp8_expected_capture_calls} "
                            f"graph_id={int(_graph_id)}"
                        )
                    _fr13_dh_fp8_attestation = {
                        "graph_id": int(_graph_id),
                        "graph_signature": _fr13_dh_fp8_signature,
                        "batch_size": int(_batch_size),
                        "captured_loop_calls": (
                            _fr13_dh_fp8_expected_capture_calls
                        ),
                        "capture_origin": _fr13_dh_fp8_lifecycle[
                            "capture_origin"
                        ],
                    }
                    self._fr13_dh_fp8_graph_attestations[
                        int(_graph_id)
                    ] = _fr13_dh_fp8_attestation
                elif (
                    self._fr13_dh_fp8_selected_capture_calls != 0
                    or _fr13_dh_fp8_attestation.get("graph_signature")
                    != _fr13_dh_fp8_signature
                    or _fr13_dh_fp8_attestation.get("batch_size")
                    != int(_batch_size)
                    or _fr13_dh_fp8_attestation.get(
                        "captured_loop_calls"
                    )
                    != _fr13_dh_fp8_expected_capture_calls
                    or _fr13_dh_fp8_attestation.get("capture_origin")
                    != _fr13_dh_fp8_lifecycle.get("capture_origin")
                ):
                    raise RuntimeError(
                        "FR13 draft-head FP8 replay left its attested graph"
                    )
                self._fr13_dh_fp8_selected_root_calls = 0
                self._fr13_dh_fp8_selected_capture_calls = 0
                if not _fr13_dh_fp8_measured:
                    return
                if int(
                    _fr13_dh_fp8_lifecycle.get("measured_replays", 0)
                ) < 1:
                    raise RuntimeError(
                        "FR13 draft-head FP8 engagement lacks a measured replay"
                    )
                _fr13_dh_fp8_contract_value = _fr13_dh_fp8_contract()
                _fr13_dh_m32_atomic_json(
                    os.environ.get(
                        "FR13_DRAFT_HEAD_FP8_ENGAGEMENT_JSON",
                        "/logs/fr13_draft_head_fp8.engagement.json",
                    ),
                    {
                        "schema": (
                            "fr13.fixed32.draft_head_fp8_engagement.v1"
                        ),
                        "status": "ENGAGED",
                        "arm": _fr13_dh_fp8_arm,
                        "source_commit": os.environ.get(
                            "FR13_DRAFT_HEAD_FP8_SOURCE_COMMIT", ""
                        ),
                        "candidate_source_sha256": os.environ.get(
                            "FR13_DRAFT_HEAD_FP8_SOURCE_SHA256", ""
                        ),
                        "served_batch_size": int(_batch_size),
                        "geometry": _fr13_dh_fp8_contract_value[
                            "geometry"
                        ],
                        "candidate": _fr13_dh_fp8_contract_value[
                            "candidate"
                        ],
                        "traffic": _fr13_dh_fp8_contract_value["traffic"],
                        "selected_root_calls": 1,
                        "captured_loop_calls": (
                            _fr13_dh_fp8_attestation[
                                "captured_loop_calls"
                            ]
                        ),
                        "fallback_calls": 0,
                        "drafter_graph_id": int(_graph_id),
                        "drafter_graph_signature": (
                            _fr13_dh_fp8_signature
                        ),
                        "observed_measured_replays_at_least": 1,
                        "capture_origin": _fr13_dh_fp8_attestation[
                            "capture_origin"
                        ],
                        "execution_basis": "cudagraph_replay",
                        "forward_step_index": int(
                            _fr13_dh_fp8_proposal["forward_step_index"]
                        ),
                        "runtime_mode": "FULL",
                        "steady_state_synchronizations": 0,
                    },
                )
                self._fr13_dh_fp8_engagement_written = True

            def _fr13_dh_u8_logits(_sh, _h):
                if (
                    not getattr(self, "_fr13_dh_u8_ready", False)
                    or tuple(_h.shape) != (1, 5120)
                    or tuple(_h.stride()) != (5120, 1)
                    or _h.dtype != torch.bfloat16
                    or not _h.is_contiguous()
                    or _h.device != _sh.weight.device
                    or tuple(_sh.weight.shape) != (65536, 5120)
                    or tuple(_sh.weight.stride()) != (5120, 1)
                    or _sh.weight.dtype != torch.bfloat16
                    or not _sh.weight.is_contiguous()
                    or tuple(self._fr13_dh_u8_output.shape) != (1, 65536)
                    or tuple(self._fr13_dh_u8_output.stride()) != (65536, 1)
                    or self._fr13_dh_u8_output.dtype != torch.bfloat16
                    or self._fr13_dh_u8_output.device != _h.device
                ):
                    raise RuntimeError(
                        "FR13 draft-head U8 left exact B1 BF16 head geometry"
                    )
                if getattr(
                    self, "_fr13_dh_u8_production_active", False
                ):
                    _fr13_dh_u8_note_production()
                    self._fr13_dh_u8_op(
                        self._fr13_dh_u8_output, _h, _sh.weight
                    )
                    if not getattr(self, "_fr13_dh_u8_engaged", False):
                        self._fr13_dh_u8_engaged = True
                        print(
                            "[FR13_DRAFT_HEAD_M1_R64_U8] engaged "
                            "candidate_served=1 incumbent_head_calls=0",
                            flush=True,
                        )
                    return self._fr13_dh_u8_output
                from vllm.model_executor.layers.mamba import (
                    gdn_linear_attn as _fr13_dh_u8_runtime_gdn,
                )

                _fr13_dh_u8_depth = getattr(
                    _fr13_dh_u8_runtime_gdn,
                    "_fr13_draft_head_u8_live_depth",
                )(1)
                _fr13_dh_u8_proposal = getattr(
                    _fr13_dh_u8_runtime_gdn,
                    "_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT",
                    None,
                )
                if not isinstance(_fr13_dh_u8_proposal, dict):
                    raise RuntimeError(
                        "FR13 draft-head U8 lost its authenticated proposal"
                    )
                if _fr13_dh_u8_depth == 0:
                    self._fr13_dh_u8_count_enable.fill_(
                        int(_fr13_dh_u8_proposal["measured"] is True)
                    )
                _fr13_dh_u8_reference = _sh.quant_method.apply(
                    _sh, _h, bias=None
                )
                self._fr13_dh_u8_op(
                    self._fr13_dh_u8_output, _h, _sh.weight
                )
                if (
                    tuple(_fr13_dh_u8_reference.shape) != (1, 65536)
                    or tuple(_fr13_dh_u8_reference.stride()) != (65536, 1)
                    or _fr13_dh_u8_reference.dtype != torch.bfloat16
                    or _fr13_dh_u8_reference.device != _h.device
                    or _fr13_dh_u8_reference.data_ptr()
                    == self._fr13_dh_u8_output.data_ptr()
                ):
                    raise RuntimeError(
                        "FR13 draft-head U8 incumbent/candidate output alias drifted"
                    )
                self._fr13_dh_u8_mismatches[
                    _fr13_dh_u8_depth
                ].add_(
                    torch.count_nonzero(
                        self._fr13_dh_u8_output.view(torch.int16)
                        != _fr13_dh_u8_reference.view(torch.int16)
                    )
                    * self._fr13_dh_u8_count_enable
                )
                self._fr13_dh_u8_nonfinite[_fr13_dh_u8_depth].add_(
                    torch.count_nonzero(
                        ~torch.isfinite(self._fr13_dh_u8_output)
                    )
                    * self._fr13_dh_u8_count_enable
                )
                self._fr13_dh_u8_compares[_fr13_dh_u8_depth].add_(
                    self._fr13_dh_u8_count_enable
                )
                if not getattr(self, "_fr13_dh_u8_engaged", False):
                    self._fr13_dh_u8_engaged = True
                    print(
                        "[FR13_DRAFT_HEAD_M1_R64_U8] engaged "
                        + (
                            "candidate_served=1 finite_gate=1 "
                            "raw_bf16_drift_diagnostic=1"
                            if _fr13_dh_u8_quality
                            else "full_bf16_vector=65536 incumbent_served=1 "
                            "shadow_only=1"
                        ),
                        flush=True,
                    )
                if _fr13_dh_u8_quality:
                    return self._fr13_dh_u8_output
                return _fr13_dh_u8_reference

            def _fr13_dh_m4_u8_logits(_sh, _h):
                if (
                    not getattr(self, "_fr13_dh_m4_u8_ready", False)
                    or tuple(_h.shape) != (4, 5120)
                    or tuple(_h.stride()) != (5120, 1)
                    or _h.dtype != torch.bfloat16
                    or not _h.is_contiguous()
                    or _h.device != _sh.weight.device
                    or tuple(_sh.weight.shape) != (65536, 5120)
                    or tuple(_sh.weight.stride()) != (5120, 1)
                    or _sh.weight.dtype != torch.bfloat16
                    or not _sh.weight.is_contiguous()
                    or tuple(self._fr13_dh_m4_u8_output.shape) != (4, 65536)
                    or tuple(self._fr13_dh_m4_u8_output.stride()) != (65536, 1)
                    or self._fr13_dh_m4_u8_output.dtype != torch.bfloat16
                    or self._fr13_dh_m4_u8_output.device != _h.device
                ):
                    raise RuntimeError(
                        "FR13 draft-head M4 U8 left exact B4 BF16 head geometry"
                    )
                from vllm.model_executor.layers.mamba import (
                    gdn_linear_attn as _fr13_dh_m4_u8_runtime_gdn,
                )

                if _fr13_dh_m4_u8_prod:
                    self._fr13_dh_m4_u8_op(
                        self._fr13_dh_m4_u8_output, _h, _sh.weight
                    )
                    if not getattr(
                        self, "_fr13_dh_m4_u8_engaged", False
                    ):
                        self._fr13_dh_m4_u8_engaged = True
                        print(
                            "[FR13_DRAFT_HEAD_M4_R64_U8] engaged batch=4 "
                            "production=1 candidate_served=1 incumbent_calls=0",
                            flush=True,
                        )
                    return self._fr13_dh_m4_u8_output
                _fr13_dh_m4_u8_depth = getattr(
                    _fr13_dh_m4_u8_runtime_gdn,
                    "_fr13_draft_head_m4_u8_live_depth",
                )(4)
                _fr13_dh_m4_u8_proposal = getattr(
                    _fr13_dh_m4_u8_runtime_gdn,
                    "_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT",
                    None,
                )
                if not isinstance(_fr13_dh_m4_u8_proposal, dict):
                    raise RuntimeError(
                        "FR13 draft-head M4 U8 lost its authenticated proposal"
                    )
                if _fr13_dh_m4_u8_depth == 0:
                    self._fr13_dh_m4_u8_count_enable.fill_(
                        int(_fr13_dh_m4_u8_proposal["measured"] is True)
                    )
                _fr13_dh_m4_u8_reference = _sh.quant_method.apply(
                    _sh, _h, bias=None
                )
                self._fr13_dh_m4_u8_op(
                    self._fr13_dh_m4_u8_output, _h, _sh.weight
                )
                if (
                    tuple(_fr13_dh_m4_u8_reference.shape) != (4, 65536)
                    or tuple(_fr13_dh_m4_u8_reference.stride()) != (65536, 1)
                    or _fr13_dh_m4_u8_reference.dtype != torch.bfloat16
                    or _fr13_dh_m4_u8_reference.device != _h.device
                    or _fr13_dh_m4_u8_reference.data_ptr()
                    == self._fr13_dh_m4_u8_output.data_ptr()
                ):
                    raise RuntimeError(
                        "FR13 draft-head M4 U8 incumbent/candidate output alias drifted"
                    )
                self._fr13_dh_m4_u8_mismatches[
                    _fr13_dh_m4_u8_depth
                ].add_(
                    torch.count_nonzero(
                        self._fr13_dh_m4_u8_output.view(torch.int16)
                        != _fr13_dh_m4_u8_reference.view(torch.int16)
                    )
                    * self._fr13_dh_m4_u8_count_enable
                )
                self._fr13_dh_m4_u8_compares[
                    _fr13_dh_m4_u8_depth
                ].add_(self._fr13_dh_m4_u8_count_enable)
                self._fr13_dh_m4_u8_nonfinite[
                    _fr13_dh_m4_u8_depth
                ].add_(
                    torch.count_nonzero(
                        ~torch.isfinite(self._fr13_dh_m4_u8_output)
                    )
                    * self._fr13_dh_m4_u8_count_enable
                )
                if not getattr(self, "_fr13_dh_m4_u8_engaged", False):
                    self._fr13_dh_m4_u8_engaged = True
                    print(
                        "[FR13_DRAFT_HEAD_M4_R64_U8] engaged batch=4 "
                        + (
                            "candidate_served=1 finite_gate=1 taw_exact=1"
                            if _fr13_dh_m4_u8_quality
                            else "full_bf16_matrix=4x65536 incumbent_served=1 "
                            "shadow_only=1"
                        ),
                        flush=True,
                    )
                if _fr13_dh_m4_u8_quality:
                    return self._fr13_dh_m4_u8_output
                return _fr13_dh_m4_u8_reference

            def _fr13_dh_fp8_logits(_h):
                _fr13_dh_fp8_batch = (
                    int(_h.shape[0]) if _h.ndim == 2 else 0
                )
                _fr13_dh_fp8_qw = getattr(
                    self, "_fr13_dh_fp8_weight", None
                )
                _fr13_dh_fp8_ws = getattr(
                    self, "_fr13_dh_fp8_weight_scale", None
                )
                if (
                    not getattr(self, "_fr13_dh_fp8_ready", False)
                    or _fr13_dh_fp8_batch not in (1, 2, 3, 4)
                    or tuple(_h.shape) != (_fr13_dh_fp8_batch, 5120)
                    or tuple(_h.stride()) != (5120, 1)
                    or _h.dtype != torch.bfloat16
                    or not _h.is_contiguous()
                    or _fr13_dh_fp8_qw.device != _h.device
                    or _fr13_dh_fp8_ws.device != _h.device
                ):
                    raise RuntimeError(
                        "FR13 draft-head FP8 left exact B1-B4 BF16 "
                        "hidden[B,5120] contract"
                    )
                if _fr13_dh_fp8_static_io:
                    _fr13_dh_fp8_aq = self._fr13_dh_fp8_static_aq[
                        _fr13_dh_fp8_batch
                    ]
                    _fr13_dh_fp8_as = self._fr13_dh_fp8_static_as[
                        _fr13_dh_fp8_batch
                    ]
                    torch.ops._C.per_token_group_fp8_quant(
                        _h,
                        _fr13_dh_fp8_aq,
                        _fr13_dh_fp8_as,
                        128,
                        1e-10,
                        self._fr13_dh_fp8_min,
                        self._fr13_dh_fp8_max,
                        False,
                        True,
                        False,
                    )
                else:
                    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                        per_token_group_quant_fp8 as _fr13_dh_fp8_quant_act,
                    )
                    _fr13_dh_fp8_aq, _fr13_dh_fp8_as = (
                        _fr13_dh_fp8_quant_act(
                            _h,
                            128,
                            column_major_scales=True,
                            use_ue8m0=False,
                        )
                    )
                if (
                    tuple(_fr13_dh_fp8_aq.shape)
                    != (_fr13_dh_fp8_batch, 5120)
                    or tuple(_fr13_dh_fp8_aq.stride()) != (5120, 1)
                    or _fr13_dh_fp8_aq.dtype != torch.float8_e4m3fn
                    or tuple(_fr13_dh_fp8_as.shape)
                    != (_fr13_dh_fp8_batch, 40)
                    or tuple(_fr13_dh_fp8_as.stride())
                    != (1, _fr13_dh_fp8_batch)
                    or _fr13_dh_fp8_as.dtype != torch.float32
                ):
                    raise RuntimeError(
                        "FR13 draft-head FP8 activation quantization drifted"
                    )
                if _fr13_dh_fp8_static_io:
                    _fr13_dh_fp8_out = self._fr13_dh_fp8_static_out[
                        _fr13_dh_fp8_batch
                    ]
                    torch.ops._C.cutlass_scaled_mm(
                        _fr13_dh_fp8_out,
                        _fr13_dh_fp8_aq,
                        self._fr13_dh_fp8_weight_t,
                        _fr13_dh_fp8_as,
                        self._fr13_dh_fp8_weight_scale_t,
                        None,
                    )
                else:
                    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
                        cutlass_scaled_mm as _fr13_dh_fp8_scaled_mm,
                    )
                    _fr13_dh_fp8_out = _fr13_dh_fp8_scaled_mm(
                        _fr13_dh_fp8_aq,
                        _fr13_dh_fp8_qw,
                        _fr13_dh_fp8_as,
                        _fr13_dh_fp8_ws,
                        [128, 128],
                        torch.bfloat16,
                    )
                if (
                    tuple(_fr13_dh_fp8_out.shape)
                    != (_fr13_dh_fp8_batch, 65536)
                    or tuple(_fr13_dh_fp8_out.stride()) != (65536, 1)
                    or _fr13_dh_fp8_out.dtype != torch.bfloat16
                ):
                    raise RuntimeError(
                        "FR13 draft-head FP8 CUTLASS output drifted"
                    )
                _fr13_dh_fp8_note_selection(_fr13_dh_fp8_batch)
                if not getattr(self, "_fr13_dh_fp8_engaged", False):
                    self._fr13_dh_fp8_engaged = True
                    print(
                        "[FR13_DRAFT_HEAD_FP8] engaged "
                        f"batch={_fr13_dh_fp8_batch} "
                        "proposal_logits=fp8_output_direct "
                        "bf16_shadow_calls=0",
                        flush=True,
                    )
                return _fr13_dh_fp8_out

            def _fr13_dvk_logits(_h):
                # Pair the logits with the only map valid for those rows.
                # A full-head fallback always returns map=None.
                try:
                    _sh = self._fr13_dvk_shim
                    _fr13_dh_rows_on = int(
                        getattr(self, "_fr13_dh_pad_rows", 0)
                    )
                    _fr13_dh_ab_on = getattr(
                        self, "_fr13_dh_ab_active", False
                    )
                    _fr13_dh_m32_live_on = getattr(
                        self, "_fr13_dh_m32_live_active", False
                    )
                    _fr13_dh_m32_prod_on = getattr(
                        self, "_fr13_dh_m32_production_active", False
                    )
                    _fr13_dh_fp8_on = getattr(
                        self, "_fr13_dh_fp8_active", False
                    )
                    _fr13_dh_u8_on = getattr(
                        self, "_fr13_dh_u8_live_active", False
                    ) or getattr(
                        self, "_fr13_dh_u8_production_active", False
                    )
                    _fr13_dh_m4_u8_on = getattr(
                        self, "_fr13_dh_m4_u8_active", False
                    )
                    _fr13_dh_capturing = False
                    if _fr13_dh_m4_u8_on:
                        _logits = _fr13_dh_m4_u8_logits(_sh, _h)
                    elif _fr13_dh_u8_on:
                        _logits = _fr13_dh_u8_logits(_sh, _h)
                    elif _fr13_dh_fp8_on:
                        _logits = _fr13_dh_fp8_logits(_h)
                    elif _fr13_dh_m32_live_on or _fr13_dh_m32_prod_on:
                        _fr13_dh_capturing = (
                            torch.cuda.is_current_stream_capturing()
                        )
                        _fr13_dh_measured = (
                            _fr13_dh_m32_measured_proposal()
                        )
                        if (
                            _fr13_dh_m32_live_on
                            and not _fr13_dh_capturing
                        ):
                            self._fr13_dh_m32_live_count_enable.fill_(
                                int(_fr13_dh_measured)
                            )
                        _fr13_dh_candidate = _fr13_dh_pad_logits(
                            _sh, _h, 32
                        )
                        if _fr13_dh_m32_live_on:
                            _fr13_dh_reference = (
                                _sh.quant_method.apply(
                                    _sh, _h, bias=None
                                )
                            )
                            _fr13_dh_count_enable = (
                                self._fr13_dh_m32_live_count_enable
                            )
                            self._fr13_dh_ab_mismatches[0].add_(
                                torch.count_nonzero(
                                    _fr13_dh_candidate.view(torch.int16)
                                    != _fr13_dh_reference.view(
                                        torch.int16
                                    )
                                )
                                * _fr13_dh_count_enable
                            )
                            self._fr13_dh_ab_compares[0].add_(
                                _fr13_dh_count_enable
                            )
                            _logits = _fr13_dh_reference
                        else:
                            _fr13_dh_m32_note_production(
                                _fr13_dh_capturing
                            )
                            _logits = _fr13_dh_candidate
                    elif _fr13_dh_rows_on or _fr13_dh_ab_on:
                        if _fr13_dh_ab_on and not torch.cuda.is_current_stream_capturing():
                            _fr13_dh_checks = int(
                                self._fr13_dh_ab_root_checks
                            )
                            if _fr13_dh_checks:
                                _fr13_dh_bad = tuple(
                                    int(_fr13_dh_v)
                                    for _fr13_dh_v in self._fr13_dh_ab_mismatches.tolist()
                                )
                                if any(_fr13_dh_bad):
                                    raise AssertionError(
                                        "FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB "
                                        "full-logit mismatch counts "
                                        f"rows32_64_128={_fr13_dh_bad}"
                                    )
                                if _fr13_dh_checks % 128 == 0:
                                    _fr13_dh_done = tuple(
                                        int(_fr13_dh_v)
                                        for _fr13_dh_v in self._fr13_dh_ab_compares.tolist()
                                    )
                                    print(
                                        "[FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB] "
                                        f"PASS root_checks={_fr13_dh_checks} "
                                        f"compares32_64_128={_fr13_dh_done} "
                                        "full_logit_bit_mismatches=0",
                                        flush=True,
                                    )
                            self._fr13_dh_ab_root_checks = (
                                _fr13_dh_checks + 1
                            )
                        if _fr13_dh_ab_on:
                            _fr13_dh_reference = _sh.quant_method.apply(
                                _sh, _h, bias=None
                            )
                            for _fr13_dh_i, _fr13_dh_r in enumerate(
                                (32, 64, 128)
                            ):
                                _fr13_dh_candidate = _fr13_dh_pad_logits(
                                    _sh, _h, _fr13_dh_r
                                )
                                self._fr13_dh_ab_mismatches[_fr13_dh_i].add_(
                                    torch.count_nonzero(
                                        _fr13_dh_candidate.view(torch.int16)
                                        != _fr13_dh_reference.view(torch.int16)
                                    )
                                )
                                self._fr13_dh_ab_compares[_fr13_dh_i].add_(1)
                            _logits = _fr13_dh_reference
                        else:
                            _logits = _fr13_dh_pad_logits(
                                _sh, _h, _fr13_dh_rows_on
                            )
                    else:
                        _logits = _sh.quant_method.apply(
                            _sh, _h, bias=None
                        )
                    return _logits, getattr(
                        self, "_fr13_dvk_map_t", None
                    )
                except Exception as _e:
                    if getattr(self, "_fr13_dh_m4_u8_active", False):
                        self._fr13_dh_m4_u8_fallback_calls += 1
                        raise RuntimeError(
                            "FR13 draft-head M4 U8 live A/B failed its strict "
                            "runtime contract; incumbent fallback is forbidden"
                        ) from _e
                    if getattr(
                        self, "_fr13_dh_u8_production_active", False
                    ):
                        self._fr13_dh_u8_fallback_calls += 1
                        raise RuntimeError(
                            "FR13 draft-head U8 production failed its strict "
                            "runtime contract; incumbent fallback is forbidden"
                        ) from _e
                    if getattr(self, "_fr13_dh_u8_live_active", False):
                        raise RuntimeError(
                            "FR13 draft-head U8 live A/B failed its strict "
                            "runtime contract; incumbent fallback is forbidden"
                        ) from _e
                    if getattr(self, "_fr13_dh_fp8_active", False):
                        self._fr13_dh_fp8_fallback_calls += 1
                        raise RuntimeError(
                            "FR13 draft-head FP8 failed its strict runtime "
                            "contract; BF16 fallback is forbidden"
                        ) from _e
                    if getattr(
                        self, "_fr13_dh_m32_production_active", False
                    ):
                        raise RuntimeError(
                            "FR13 draft-head M32 production failed its strict "
                            "runtime contract"
                        ) from _e
                    if getattr(
                        self, "_fr13_dh_m32_live_active", False
                    ):
                        raise RuntimeError(
                            "FR13 draft-head M32 live A/B failed its strict "
                            "runtime contract"
                        ) from _e
                    if (
                        getattr(self, "_fr13_dh_pad_rows", 0)
                        or getattr(self, "_fr13_dh_ab_active", False)
                    ):
                        raise RuntimeError(
                            "FR13 draft-head padding failed its strict runtime "
                            "contract"
                        ) from _e
                    self._fr13_dvk_dead = True
                    print(
                        f"[FR13_DRAFT_VOCAB] DISABLED (apply failed): {_e!r}",
                        flush=True,
                    )
                    return self.model.compute_logits(_h), None

            def _fr13_dvk_real_ids(_ids, _id_map):
                return _ids if _id_map is None else _id_map[_ids]

            if _fr13_dvk_root:
                _fr13_dvk, _fr13_full_vocab_size = _fr13_dvk_prepare()
            else:
                _fr13_dvk = 0
                _fr13_full_vocab_size = None
            _fr13_dfwd_top3_prepare()
            _fr14_fused_topk_prepare(
                int(self.model.lm_head.weight.shape[0])
                if _fr14_fused_topk
                else 0
            )

            # FR13_RESHAPE_DEPTH3: cat3w consumes the root runner-up as its
            # (1,) root-sibling leaf. cat9/chain5/chain3 never consume the
            # root runner-up (every choice starts with 0), so it stays None
            # for them and the root top-2 is not read (no extra lm-head work).
            # FR13_RESHAPE_DEPTH5: cat6root and cat10 ALSO carry the (1,) root
            # sibling, so they read the rank-2 token from the SAME root logits
            # (one topk, no extra lm-head read), identical to cat3w.
            # FR13_RESHAPE_333: 3-3-3 carries BOTH a (1,) root rank-1 leaf and
            # a (2,) root rank-2 leaf, so it reads topk(root_logits, 3) and
            # consumes indices[:, 1] (rank-1) + indices[:, 2] (rank-2). The
            # rank-2 token is a runner-up read from the SAME logits (no extra
            # lm-head), never fed into a forward/recurrent state.
            _fr10_consumes_root_leaf = (
                _fr10_is_cat3w or _fr10_is_cat6root or _fr10_is_cat10
                or _fr10_is_333
            )
            _fr10_consumes_root_leaf2 = _fr10_is_333
            _fr10_root_leaf_token = None
            _fr10_root_leaf2_token = None
            # FR13_RESHAPE_333: read 3 (top-3) only when the root rank-2 leaf
            # is consumed; otherwise keep k=2 (byte-identical to legacy).
            _fr10_root_topk_k = 3 if _fr10_consumes_root_leaf2 else 2
            _fr10_root_map = None
            _fr13_root_top3 = None
            _fr14_root_wide = None
            if _fr13_single_logits:
                # Root top-2 is verified unused for cat9/chain5/chain3 (no
                # tree node consumes the root runner-up: every choice starts
                # with 0). cat3w/cat6root/cat10 read the rank-2 token from the
                # SAME root logits tensor (one topk, no extra lm-head read).
                # 3-3-3 additionally reads rank-2 (index 2) from that topk.
                _fr13_ds_lm = _FR13_DFWD_SPLIT.begin('lmhead')
                if (
                    _fr13_dvk_root
                    and _fr13_dvk > 0
                    and not getattr(self, "_fr13_dvk_dead", False)
                ):
                    _fr10_logits, _fr10_root_map = _fr13_dvk_logits(
                        sample_hidden_states
                    )
                    if not getattr(self, "_fr13_dvk_dead", False):
                        if not getattr(
                            self, "_fr13_dvk_root_engaged", False
                        ):
                            self._fr13_dvk_root_engaged = True
                            print(
                                "[FR13_DRAFT_VOCAB_ROOT] engaged "
                                f"K={_fr13_dvk} "
                                f"mode={'gather' if _fr10_root_map is not None else 'contig'} "
                                f"full_vocab={_fr13_full_vocab_size}",
                                flush=True,
                            )
                else:
                    _fr10_logits = _e8_primary(self, sample_hidden_states, "root", _E8_ARM, _E8_QUALIFY)
                _FR13_DFWD_SPLIT.end('lmhead', _fr13_ds_lm)
                if _fr13_dfwd_top3:
                    draft_token_ids, _fr13_root_top3 = (
                        _fr13_dfwd_top3_select(
                            _fr10_logits,
                            _fr10_root_map,
                            self._fr13_dfwd_top3_root_spine,
                            self._fr13_dfwd_top3_root_wide,
                            "root",
                        )
                    )
                elif _fr14_fused_topk:
                    # ONE launch for BOTH the root spine argmax and the root
                    # width-3 leaves; _fr14_root_wide is consumed by the wide
                    # capture below instead of a second full-vocab topk.
                    _fr14_root_bufs = self._fr14_fused_topk_buffers[
                        int(batch_size)
                    ]
                    draft_token_ids, _fr14_root_wide = _fr14_fused_topk_select(
                        _fr10_logits,
                        _fr14_root_bufs[1],
                        _fr14_root_bufs[2],
                        "root",
                    )
                else:
                    draft_token_ids = _fr10_logits.argmax(dim=-1)
                    draft_token_ids = _fr13_dvk_real_ids(
                        draft_token_ids, _fr10_root_map
                    )
                if _fr10_consumes_root_leaf:
                    _fr10_root_topk = torch.topk(
                        _fr10_logits, _fr10_root_topk_k, dim=-1
                    ).indices
                    _fr10_root_topk = _fr13_dvk_real_ids(
                        _fr10_root_topk, _fr10_root_map
                    )
                    _fr10_root_leaf_token = _fr10_root_topk[:, 1]
                    if _fr10_consumes_root_leaf2:
                        _fr10_root_leaf2_token = _fr10_root_topk[:, 2]
                if _fr13_selfcheck:
                    _fr13_sc_check(
                        "root",
                        draft_token_ids,
                        self._greedy_sample(sample_hidden_states),
                    )
            else:
                _fr10_logits = _e8_primary(self, sample_hidden_states, "root", _E8_ARM, _E8_QUALIFY)
                _fr10_top2 = torch.topk(
                    _fr10_logits, _fr10_root_topk_k, dim=-1
                ).indices
                draft_token_ids = self._greedy_sample(sample_hidden_states)
                if _fr10_consumes_root_leaf:
                    _fr10_root_leaf_token = _fr10_top2[:, 1]
                    if _fr10_consumes_root_leaf2:
                        _fr10_root_leaf2_token = _fr10_top2[:, 2]
            _fr10_spine_tokens = [draft_token_ids]
            _fr10_leaf_tokens = []
            # FR13_RESHAPE_333: parallel list of the rank-2 (top-3) interior
            # leaves, collected only at _fr10_leaf2_steps (3-3-3 only).
            _fr10_leaf2_tokens = []
            # FR13_RESHAPE_WIDE: per-parent-position topk INDICES (pos -> [B, w]
            # tensor), captured off the SAME spine logits as the argmax. pos 0
            # = root (read here), pos p>=1 = spine_{p-1} (read in the loop). The
            # leaf at (parent_pos, rank>0) is wide_topk[parent_pos][:, rank] --
            # a pure runner-up read, never fed forward. _fr10_logits is the root
            # lm-head output (bound in BOTH the single-logits and legacy
            # branches above).
            _fr10_wide_topk = {}
            # FR13_DEDUP_SIBLINGS: capture a few SPARE topk ranks beyond the
            # per-position width so the sibling-dedup pass (before packing) has
            # real distinct model candidates to swap a collided branch for. The
            # packer only reads ranks 0..width-1, so the spare columns are inert
            # unless the dedup consumes them. Default ON (correctness+budget; a
            # duplicate sibling wastes a verify node AND creates a temp-0
            # argmax-tie). off => byte-identical prior capture width.
            _fr13_dedup_sib = (
                __import__('os').environ.get('FR13_DEDUP_SIBLINGS', '1') == '1'
                and not _fr13_is_fixed32
            )
            _fr13_dedup_slack = 3 if _fr13_dedup_sib else 0
            if _fr10_is_wide:
                _fr10_w_root = _fr10_wide_width.get(0, 1)
                if _fr10_w_root > 1:
                    if _fr13_dfwd_top3:
                        if _fr10_w_root != 3 or _fr13_root_top3 is None:
                            raise RuntimeError(
                                "FR13 DFWD K64 root top3 width drifted"
                            )
                        _fr10_wide_topk[0] = _fr13_root_top3
                    elif _fr14_fused_topk:
                        if (
                            _fr10_w_root != 3
                            or _fr13_dedup_slack != 0
                            or _fr14_root_wide is None
                        ):
                            raise RuntimeError(
                                "FR14 fused draft top-k root width drifted"
                            )
                        _fr10_wide_topk[0] = _fr14_root_wide
                    else:
                        _fr10_wide_topk[0] = torch.topk(
                            _fr10_logits,
                            min(_fr10_w_root + _fr13_dedup_slack,
                                int(_fr10_logits.shape[-1])),
                            dim=-1,
                        ).indices
                        _fr10_wide_topk[0] = _fr13_dvk_real_ids(
                            _fr10_wide_topk[0], _fr10_root_map
                        )

            # FR13_MERGED_DRAFTER_SEAM DELETED 2026-07-27 (cleanup+bake,
            # FR13_CLEANUP_BAKE_PLAN.md): the head-merge decide_and_fill path
            # (adaptive MTP-k + Arctic grow-to-cat33333) was dormant-by-design in
            # tail mode and Front-2/merge closed as a no-go. Tail mode (decide_tail,
            # below) is the shipped design. git history has the seam.

            if self.allowed_attn_types is not None:
                for group_md in per_group_attn_metadata:
                    if not isinstance(group_md, self.allowed_attn_types):
                        raise ValueError(
                            f"Unsupported attention metadata type for speculative "
                            "decoding with FR10 caterpillar native-spine drafting: "
                            f"{type(group_md)}. Supported types are: "
                            f"{self.allowed_attn_types}"
                        )

            cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
                self._determine_batch_execution_and_padding(batch_size)
            )

            common_attn_metadata.num_actual_tokens = batch_size
            common_attn_metadata.max_query_len = 1
            common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
            common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
                self.token_arange_np[: batch_size + 1]
            ).clone()

            if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
                common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
                common_attn_metadata._seq_lens_cpu = None
                common_attn_metadata._num_computed_tokens_cpu = None

            block_size = self.block_size
            assert block_size > 0, "block_size has not been initialized."
            # FR13_DRAFTER_GRAPH (R4): capture the WHOLE 4-iteration spine
            # loop as ONE CUDA graph per batch size — the ~93ms/step drafter
            # cost is host python BETWEEN piecewise pieces (2g named:
            # dispatch + set_forward_context + sampling glue x4; metadata
            # builds measured ~0 by the meta-reuse dead-heat A/B).
            # Mechanics: lazy capture_begin/capture_end around the EXISTING
            # loop on the first eligible live call per B (capture records
            # without executing; the immediate replay below produces this
            # step's real outputs and KV/seq_lens mutations exactly once =
            # eager-equivalent). During capture the inner model call is
            # forced EAGER (replaying piecewise child graphs inside a parent
            # capture is illegal; recording the pieces' kernels flat is the
            # point) and max_seq_len is baked at max_model_len (upper-bound
            # semantics; kernels bound by the live seq_lens device tensor).
            # Replay path: copy root+hidden into static buffers, replay,
            # append static output views (final torch.cat COPIES them out),
            # then apply the loop's host post-conditions (+4 shadows).
            # FR14_GATE_SPLIT_GRAPH (lever 2, default OFF). The drafter's four
            # post-root MTP forwards are captured as TWO 2-pass graphs, `lo` and
            # `hi`, sharing one memory pool and the SAME static buffers. An
            # ungated step replays lo then hi -- still exactly four forwards, so
            # every per-step invariant that counts four is untouched -- and a
            # gated step replays lo alone, leaving Arctic to fill draft
            # positions 3..10 instead of 5..10. Measured on GB10
            # (results/fr14_nvfp4_port_20260816/suffix_gate_graph_microbench.json):
            # the split is BIT-EXACT against the single graph, the second
            # capture costs 0.275 ms once and 23 MB of pool, and the extra
            # launch costs nothing measurable.
            # The decision was staged before this forward by
            # fr13_merged_drafter.stage_fixed32_step, so nothing here reads
            # device memory, syncs, or looks at a draft token.
            _fr14_gate_fired = False
            _fr14_split_on = False
            # the MTP head depth a GATED step leaves behind, taken from the one
            # place that defines it, so no arithmetic downstream restates it
            _fr14_gate_mtp_k = None
            if _fr13_is_fixed32:
                import sys as _fr14_gate_sys
                if "/workspace/scripts" not in _fr14_gate_sys.path:
                    _fr14_gate_sys.path.insert(0, "/workspace/scripts")
                import fr13_merged_drafter as _fr14_gate_md
                _fr14_split_on = bool(_fr14_gate_md.fr14_gate().enabled)
                if _fr14_split_on:
                    _fr14_gate_fired, _fr14_gate_decisions = (
                        _fr14_gate_md.fr14_gate_pending()
                    )
                    if len(_fr14_gate_decisions) != int(batch_size):
                        raise RuntimeError(
                            "FR14 suffix pass gate decision count != batch"
                        )
                    from fr14_suffix_pass_gate import (
                        SuffixPassGate as _fr14_gate_shape_cls,
                    )
                    _fr14_gate_mtp_k = int(
                        _fr14_gate_shape_cls.step_shape(True)[0]
                    )
            _fr14_seg_passes = 2 if _fr14_split_on else 4
            _fr13_dg_on = (
                os.environ.get("FR13_DRAFTER_GRAPH", "0") == "1"
                and int(_fr10_spine_steps) == 4
                and _fr13_single_logits
                and not (self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0)
                and not torch.cuda.is_current_stream_capturing()
                and not _fr13_selfcheck
            )
            if not _fr13_dvk_root:
                # Keep the accepted root-off execution order: build the loop
                # subset only after the unchanged full root head.
                _fr13_dvk, _ = _fr13_dvk_prepare()
            _fr13_ds_on = (
                os.environ.get("FR13_DFWD_SPLIT_NEEDLE", "0") == "1"
                and not torch.cuda.is_current_stream_capturing()
            )
            if _fr13_ds_on:
                torch.cuda.synchronize()
                _fr13_ds_t0 = __import__("time").monotonic()
            _fr13_dg_key = int(batch_size)
            _fr13_f32_dg_gdn = None
            if _fr13_is_fixed32:
                from vllm.model_executor.layers.mamba import (
                    gdn_linear_attn as _fr13_f32_dg_gdn,
                )
                if not _fr13_dg_on:
                    raise RuntimeError(
                        "FR13 fixed32 requires the full four-forward drafter graph"
                    )
            _fr13_dg_all = getattr(self, "_fr13_dg_graphs", None)
            if _fr13_dg_all is None:
                _fr13_dg_all = self._fr13_dg_graphs = {}
                self._fr13_dg_calls = {}
            if (
                os.environ.get("FR13_DRAFTER_GRAPH", "0") == "1"
                and not _fr13_dg_on
                and not getattr(self, "_fr13_dg_elig_reported", False)
            ):
                self._fr13_dg_elig_reported = True
                print(
                    "[FR13_DRAFTER_GRAPH] INELIGIBLE first-call report: "
                    f"spine_steps={_fr10_spine_steps} "
                    f"single_logits={_fr13_single_logits} "
                    f"mrope={self.uses_mrope} "
                    f"xdrope={self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0} "
                    f"mm={self.supports_mm_inputs} "
                    f"capturing={torch.cuda.is_current_stream_capturing()} "
                    f"nrt_none={num_rejected_tokens_gpu is None} "
                    f"selfcheck={_fr13_selfcheck}",
                    flush=True,
                )
            _fr13_dg_cap = False
            _fr13_dg_g = None
            if _fr13_dg_on and _fr13_dg_key in _fr13_dg_all:
                _dg = _fr13_dg_all[_fr13_dg_key]
                _dg["root"].copy_(_fr10_spine_tokens[-1])
                _dg["hidden"].copy_(hidden_states[:batch_size])
                # per-call device inputs -> static homes (addresses baked in
                # the graph): positions + adjusted seq_lens.
                _dg["pos"].copy_(positions)
                _dg["slen"].copy_(common_attn_metadata.seq_lens)
                # Segments: [(graph, signature, passes)]. One 4-pass segment
                # when the gate is off (byte-identical to the shipped path);
                # two 2-pass segments when it is armed. A gated step stops
                # after the first.
                _fr14_segs = _dg["segments"]
                _fr14_run = (
                    1 if (_fr14_gate_fired and len(_fr14_segs) > 1)
                    else len(_fr14_segs)
                )
                _fr14_done = 0
                for _fr14_si in range(_fr14_run):
                    _fr14_g, _fr14_sig, _fr14_np = _fr14_segs[_fr14_si]
                    _fr14_g.replay()
                    if _fr13_is_fixed32:
                        _fr13_f32_dg_gdn._fr13_fixed32_drafter_graph_replay(
                            id(_fr14_g),
                            _fr14_sig,
                            _fr13_dg_key,
                            _fr14_np,
                            _fr14_si,
                        )
                        _fr13_dh_m32_note_production_replay(
                            id(_fr14_g),
                            _fr14_sig,
                            _fr13_dg_key,
                        )
                        _fr13_dh_u8_note_production_replay(
                            id(_fr14_g),
                            _fr14_sig,
                            _fr13_dg_key,
                        )
                        _fr13_dh_fp8_note_replay(
                            id(_fr14_g),
                            _fr14_sig,
                            _fr13_dg_key,
                        )
                        from vllm.v1.attention.ops.triton_unified_attention import (
                            _fr13_dfwd_unified_bm8_live_replay,
                        )
                        _fr13_dfwd_unified_bm8_live_replay(
                            id(_fr14_g),
                            _fr13_dg_key,
                        )
                    _fr14_done += _fr14_np
                for _dg_i in range(_fr14_done):
                    _fr10_spine_tokens.append(_dg["spine"][_dg_i])
                    if (_dg_i + 1) in _fr10_leaf_steps:
                        _fr10_leaf_tokens.append(_dg["leaf"][_dg_i])
                        if (_dg_i + 1) in _fr10_leaf2_steps:
                            _fr10_leaf2_tokens.append(_dg["leaf2"][_dg_i])
                    if _fr10_is_wide:
                        _dg_wp = _fr10_wide_width.get(_dg_i + 1, 1)
                        if _dg_wp > 1:
                            _fr10_wide_topk[_dg_i + 1] = _dg["wide"][
                                _dg_i
                            ][:, : min(
                                _dg_wp + _fr13_dedup_slack,
                                int(_dg["wide"].shape[2]),
                            )]
                common_attn_metadata.max_seq_len = min(
                    common_attn_metadata.max_seq_len + _fr14_done,
                    self.max_model_len,
                )
                if common_attn_metadata._seq_lens_cpu is not None:
                    common_attn_metadata._seq_lens_cpu += _fr14_done
                if common_attn_metadata._num_computed_tokens_cpu is not None:
                    common_attn_metadata._num_computed_tokens_cpu += _fr14_done
                self._fr13_dg_calls[_fr13_dg_key] = (
                    self._fr13_dg_calls.get(_fr13_dg_key, 0) + 1
                )
                if self._fr13_dg_calls[_fr13_dg_key] % 2048 == 1:
                    print(
                        f"[FR13_DRAFTER_GRAPH] replay bs={_fr13_dg_key} "
                        f"calls={self._fr13_dg_calls[_fr13_dg_key]}",
                        flush=True,
                    )
                _fr10_spine_steps = 0  # loop below no-ops; tail code proceeds
            elif _fr13_dg_on:
                # capture this call: static buffers + rebinds, then record.
                _fr13_dg_cap = True
                _fr13_ds_on = False
                _dg_dev = hidden_states.device
                _dg = {
                    "root": torch.zeros(
                        batch_size, dtype=_fr10_spine_tokens[-1].dtype,
                        device=_dg_dev,
                    ),
                    "hidden": torch.zeros_like(hidden_states[:batch_size]),
                    "spine": torch.zeros(
                        4, batch_size, dtype=torch.int64, device=_dg_dev
                    ),
                    "leaf": torch.zeros(
                        4, batch_size, dtype=torch.int64, device=_dg_dev
                    ),
                    "leaf2": torch.zeros(
                        4, batch_size, dtype=torch.int64, device=_dg_dev
                    ),
                    "wide": torch.zeros(
                        4, batch_size,
                        (max(list(_fr10_wide_width.values()) or [1])
                         + _fr13_dedup_slack) if _fr10_is_wide else 1,
                        dtype=torch.int64, device=_dg_dev,
                    ),
                }
                _dg["root"].copy_(_fr10_spine_tokens[-1])
                _dg["hidden"].copy_(hidden_states[:batch_size])
                # static homes for per-call device inputs: the graph bakes
                # THESE addresses; replay refreshes their contents.
                _dg["pos"] = torch.zeros_like(positions)
                _dg["pos"].copy_(positions)
                positions = _dg["pos"]
                _dg["slen"] = torch.zeros_like(common_attn_metadata.seq_lens)
                _dg["slen"].copy_(common_attn_metadata.seq_lens)
                common_attn_metadata.seq_lens = _dg["slen"]
                _fr10_spine_tokens[-1] = _dg["root"]
                hidden_states = _dg["hidden"]
                common_attn_metadata.max_seq_len = self.max_model_len - 8
                cudagraph_runtime_mode = type(cudagraph_runtime_mode).NONE
                if getattr(self, "_fr13_dg_pool", None) is None:
                    self._fr13_dg_pool = torch.cuda.graph_pool_handle()
                if getattr(self, "_fr13_dg_stream", None) is None:
                    self._fr13_dg_stream = torch.cuda.Stream()
                # FR14_GATE_SPLIT_GRAPH: the capture step is NEVER gated -- it
                # records every segment and then replays all of them, so this
                # call's outputs and KV mutations stay eager-equivalent.
                _fr14_gate_fired = False
                _fr14_segs = []
                _fr13_dg_g = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                # capture must run on a NON-default stream (raw begin/end
                # form; the context-manager does this internally).
                _fr13_dg_prev_stream = torch.cuda.current_stream()
                torch.cuda.set_stream(self._fr13_dg_stream)
                if _fr13_is_fixed32:
                    _fr13_f32_dg_gdn._fr13_fixed32_drafter_graph_capture_begin(
                        id(_fr13_dg_g),
                        _fr13_dg_key,
                        _fr14_seg_passes,
                        len(_fr14_segs),
                    )
                _fr13_dg_g.capture_begin(pool=self._fr13_dg_pool)
            # FR13_RESHAPE_DEPTH3: cat9/chain5 keep range(4) (depth-5 spine);
            # the depth-3 shapes (chain3/cat3w) run 2 post-root steps. Each
            # extra spine forward mutates seq_lens/slot_mapping/KV, so the step
            # count MUST match the committed tree depth -- do not over-run.
            for token_index in range(_fr10_spine_steps):
                if _fr13_dg_cap and _fr14_split_on and token_index == 2:
                    # close `lo`, open `hi`. Both halves record into the SAME
                    # static buffers and share one pool, which is what makes
                    # replay(lo)+replay(hi) bit-identical to one 4-pass graph.
                    _fr13_dg_g.capture_end()
                    torch.cuda.set_stream(_fr13_dg_prev_stream)
                    _fr14_seg_sig = None
                    if _fr13_is_fixed32:
                        _fr14_seg_sig = (
                            _fr13_f32_dg_gdn._fr13_fixed32_drafter_graph_capture_end(
                                id(_fr13_dg_g),
                                _fr13_dg_key,
                                _fr14_seg_passes,
                                len(_fr14_segs),
                            )
                        )
                    _fr14_segs.append(
                        (_fr13_dg_g, _fr14_seg_sig, _fr14_seg_passes)
                    )
                    _fr13_dg_g = torch.cuda.CUDAGraph()
                    _fr13_dg_prev_stream = torch.cuda.current_stream()
                    torch.cuda.set_stream(self._fr13_dg_stream)
                    if _fr13_is_fixed32:
                        _fr13_f32_dg_gdn._fr13_fixed32_drafter_graph_capture_begin(
                            id(_fr13_dg_g),
                            _fr13_dg_key,
                            _fr14_seg_passes,
                            len(_fr14_segs),
                        )
                    _fr13_dg_g.capture_begin(pool=self._fr13_dg_pool)
                input_ids = _fr10_spine_tokens[-1].int()
                positions_1d = positions[0] if self.uses_mrope else positions
                if self.uses_mrope:
                    out_pos = self.mrope_positions[0, :batch_size]
                elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                    out_pos = self.xdrope_positions[0, :batch_size]
                else:
                    out_pos = self.positions[:batch_size]
                eagle_step_update_slot_mapping_and_metadata(
                    positions_1d=positions_1d,
                    block_table_tensor=common_attn_metadata.block_table_tensor,
                    seq_lens=common_attn_metadata.seq_lens,
                    block_size=block_size,
                    max_model_len=self.max_model_len,
                    out_clamped_positions=out_pos,
                    out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
                    input_batch_size=input_batch_size,
                )
                common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]
                if self.uses_mrope:
                    self.mrope_positions[1:, :batch_size] = self.mrope_positions[
                        0, :batch_size
                    ]
                    positions = self.mrope_positions[:, :batch_size]
                elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                    self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[
                        0, :batch_size
                    ]
                    positions = self.xdrope_positions[0, :batch_size]
                else:
                    positions = self.positions[:batch_size]

                common_attn_metadata.max_seq_len = min(
                    common_attn_metadata.max_seq_len + 1, self.max_model_len
                )
                if common_attn_metadata._seq_lens_cpu is not None:
                    common_attn_metadata._seq_lens_cpu += 1
                if common_attn_metadata._num_computed_tokens_cpu is not None:
                    common_attn_metadata._num_computed_tokens_cpu += 1

                _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
                    common_attn_metadata, draft_index=token_index + 1
                )

                self.input_ids[:batch_size] = input_ids
                self.hidden_states[:batch_size] = hidden_states
                if self.supports_mm_inputs:
                    self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:input_batch_size]
                else:
                    input_ids = self.input_ids[:input_batch_size]
                    inputs_embeds = None

                model_kwargs = {
                    "input_ids": input_ids,
                    "positions": self._get_positions(input_batch_size),
                    "inputs_embeds": inputs_embeds,
                }
                if self.pass_hidden_states_to_model:
                    model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

                with set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=input_batch_size,
                    num_tokens_across_dp=batch_size_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=self._get_slot_mapping(input_batch_size),
                ):
                    ret_hidden_states = self.model(**model_kwargs)
                    if _fr13_is_fixed32:
                        _fr13_f32_dg_gdn._fr13_fixed32_drafter_mtp_forward(
                            int(batch_size),
                            _fr13_dg_cap,
                        )
                    if not self.model_returns_tuple():
                        last_hidden_states = ret_hidden_states
                        hidden_states = ret_hidden_states
                    else:
                        last_hidden_states, hidden_states = ret_hidden_states

                hidden_states = hidden_states[:batch_size]
                _fr13_step_top3 = None
                _fr14_step_wide = None
                if _fr13_single_logits:
                    # Single lm-head read: spine token = argmax of the SAME
                    # logits tensor _greedy_sample would have recomputed.
                    # Top-2 packing is kept ONLY for the caterpillar leaf
                    # slots; spine-only (chain5) never consumes the
                    # runner-up (packing and the FR10_METRICS log both emit
                    # no leaves in spine-only mode), so its topk is skipped.
                    if _fr13_dvk > 0 and not getattr(self, "_fr13_dvk_dead", False):
                        _fr10_step_logits, _fr10_step_map = _fr13_dvk_logits(
                            last_hidden_states[:batch_size]
                        )
                    else:
                        _fr10_step_logits = _e8_primary(self, last_hidden_states[:batch_size], "loop", _E8_ARM, _E8_QUALIFY)
                        _fr10_step_map = None
                    if _fr13_dfwd_top3:
                        if not _fr13_dg_cap:
                            raise RuntimeError(
                                "FR13 DFWD K64 loop top3 requires graph capture"
                            )
                        draft_token_ids, _fr13_step_top3 = (
                            _fr13_dfwd_top3_select(
                                _fr10_step_logits,
                                _fr10_step_map,
                                _dg["spine"][token_index],
                                _dg["wide"][token_index, :, :3],
                                "loop",
                            )
                        )
                    elif _fr14_fused_topk:
                        if not _fr13_dg_cap:
                            raise RuntimeError(
                                "FR14 fused draft top-k loop select requires "
                                "graph capture"
                            )
                        # Writes STRAIGHT into the drafter graph's static
                        # spine/wide buffers: one launch replaces argmax, the
                        # multi-kernel topk, and both buffer copies.
                        draft_token_ids, _fr14_step_wide = (
                            _fr14_fused_topk_select(
                                _fr10_step_logits,
                                _dg["spine"][token_index],
                                _dg["wide"][token_index, :, :3],
                                "loop",
                            )
                        )
                    else:
                        draft_token_ids = _fr10_step_logits.argmax(dim=-1)
                        draft_token_ids = _fr13_dvk_real_ids(
                            draft_token_ids, _fr10_step_map
                        )
                    if _fr13_selfcheck:
                        _fr13_sc_check(
                            "loop",
                            draft_token_ids,
                            self._greedy_sample(last_hidden_states[:batch_size]),
                        )
                    if _fr13_dg_cap and not (
                        _fr13_dfwd_top3 or _fr14_fused_topk
                    ):
                        # FR13_DRAFTER_GRAPH: route through the static out
                        # buffer so replay reproduces the token chain.
                        _dg["spine"][token_index].copy_(draft_token_ids)
                        draft_token_ids = _dg["spine"][token_index]
                    _fr10_spine_tokens.append(draft_token_ids)
                    # FR13_RESHAPE_DEPTH3: collect the runner-up leaf only at
                    # the depths this shape consumes. cat9 = {1,2,3,4} (every
                    # step), so this is byte-identical to the legacy
                    # "not spine_only" branch; cat3w = {1} (d1 (0,1) only);
                    # chain5/chain3 = {} (no leaves).
                    # FR13_RESHAPE_333: this step is in BOTH _fr10_leaf_steps
                    # and _fr10_leaf2_steps ({1,2}); read top-3 ONCE and pack
                    # rank-1 (index 1) + rank-2 (index 2). The topk widens to
                    # k=3 ONLY for 3-3-3 (every other shape keeps k=2 == the
                    # legacy byte-identical read).
                    if (token_index + 1) in _fr10_leaf_steps:
                        _fr10_step_topk_k = (
                            3 if (token_index + 1) in _fr10_leaf2_steps else 2
                        )
                        _fr10_step_top2 = torch.topk(
                            _fr10_step_logits, _fr10_step_topk_k, dim=-1
                        ).indices
                        _fr10_step_top2 = _fr13_dvk_real_ids(
                            _fr10_step_top2, _fr10_step_map
                        )
                        if _fr13_dg_cap:
                            _dg["leaf"][token_index].copy_(_fr10_step_top2[:, 1])
                            _fr10_leaf_tokens.append(_dg["leaf"][token_index])
                            if (token_index + 1) in _fr10_leaf2_steps:
                                _dg["leaf2"][token_index].copy_(
                                    _fr10_step_top2[:, 2]
                                )
                                _fr10_leaf2_tokens.append(
                                    _dg["leaf2"][token_index]
                                )
                        else:
                            _fr10_leaf_tokens.append(_fr10_step_top2[:, 1])
                            if (token_index + 1) in _fr10_leaf2_steps:
                                _fr10_leaf2_tokens.append(_fr10_step_top2[:, 2])
                else:
                    if _fr13_dvk > 0 and not getattr(self, "_fr13_dvk_dead", False):
                        _fr10_step_logits, _fr10_step_map = _fr13_dvk_logits(
                            last_hidden_states[:batch_size]
                        )
                    else:
                        _fr10_step_logits = _e8_primary(self, last_hidden_states[:batch_size], "loop", _E8_ARM, _E8_QUALIFY)
                        _fr10_step_map = None
                    # FR13_RESHAPE_333: widen the legacy topk to k=3 only when
                    # this step packs a rank-2 leaf; otherwise keep k=2 (op
                    # order unchanged vs legacy cat9: topk before greedy_sample).
                    _fr10_step_topk_k = (
                        3 if (token_index + 1) in _fr10_leaf2_steps else 2
                    )
                    _fr10_step_top2 = torch.topk(
                        _fr10_step_logits, _fr10_step_topk_k, dim=-1
                    ).indices
                    _fr10_step_top2 = _fr13_dvk_real_ids(
                        _fr10_step_top2, _fr10_step_map
                    )
                    draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
                    _fr10_spine_tokens.append(draft_token_ids)
                    # FR13_RESHAPE_DEPTH3: only the leaf append is depth-gated.
                    # cat9 = {1,2,3,4} => appends every step = byte-identical;
                    # cat3w = {1}; chain shapes = {}.
                    # FR13_RESHAPE_333: rank-1 at _fr10_leaf_steps, rank-2 at
                    # _fr10_leaf2_steps (both {1,2} for 3-3-3).
                    if (token_index + 1) in _fr10_leaf_steps:
                        _fr10_leaf_tokens.append(_fr10_step_top2[:, 1])
                        if (token_index + 1) in _fr10_leaf2_steps:
                            _fr10_leaf2_tokens.append(_fr10_step_top2[:, 2])
                # FR13_RESHAPE_WIDE: capture this spine node's topk INDICES for
                # the leaf packing (parent_pos = token_index+1). Read off the
                # SAME _fr10_step_logits the spine argmax came from (one topk,
                # no extra lm-head, no recurrent feed). Width is per-depth, so a
                # depth that only carries the spine (width 1) reads nothing.
                if _fr10_is_wide:
                    _fr10_w_p = _fr10_wide_width.get(token_index + 1, 1)
                    if _fr10_w_p > 1:
                        if _fr13_dfwd_top3:
                            if _fr10_w_p != 3 or _fr13_step_top3 is None:
                                raise RuntimeError(
                                    "FR13 DFWD K64 loop top3 width drifted"
                                )
                            _fr13_dg_wt = _fr13_step_top3
                        elif _fr14_fused_topk:
                            if (
                                _fr10_w_p != 3
                                or _fr13_dedup_slack != 0
                                or _fr14_step_wide is None
                            ):
                                raise RuntimeError(
                                    "FR14 fused draft top-k loop width drifted"
                                )
                            _fr13_dg_wt = _fr14_step_wide
                        else:
                            # FR13_DEDUP_SIBLINGS: +slack spare ranks (see root capture)
                            _fr13_dg_wt = torch.topk(
                                _fr10_step_logits,
                                min(_fr10_w_p + _fr13_dedup_slack,
                                    int(_fr10_step_logits.shape[-1])),
                                dim=-1,
                            ).indices
                            _fr13_dg_wt = _fr13_dvk_real_ids(
                                _fr13_dg_wt, _fr10_step_map
                            )
                            if _fr13_dg_cap:
                                _dg["wide"][
                                    token_index, :, : _fr13_dg_wt.shape[1]
                                ].copy_(_fr13_dg_wt)
                                _fr13_dg_wt = _dg["wide"][
                                    token_index, :, : _fr13_dg_wt.shape[1]
                                ]
                        _fr10_wide_topk[token_index + 1] = _fr13_dg_wt

            if _fr13_dg_cap:
                # end recording; the immediate replay executes the recorded
                # kernels ONCE (capture itself executes nothing) => this
                # call's outputs + KV/seq_lens mutations are eager-equivalent.
                _fr13_dg_g.capture_end()
                torch.cuda.set_stream(_fr13_dg_prev_stream)
                _fr14_seg_sig = None
                if _fr13_is_fixed32:
                    _fr14_seg_sig = (
                        _fr13_f32_dg_gdn._fr13_fixed32_drafter_graph_capture_end(
                            id(_fr13_dg_g),
                            _fr13_dg_key,
                            _fr14_seg_passes,
                            len(_fr14_segs),
                        )
                    )
                _fr14_segs.append((_fr13_dg_g, _fr14_seg_sig, _fr14_seg_passes))
                if len(_fr14_segs) != (2 if _fr14_split_on else 1):
                    raise RuntimeError(
                        "FR14 drafter split capture produced "
                        + repr(len(_fr14_segs))
                        + " segments"
                    )
                if sum(_seg[2] for _seg in _fr14_segs) != 4:
                    raise RuntimeError(
                        "FR14 drafter capture segments do not sum to four "
                        "post-root MTP forwards"
                    )
                # replay every recorded segment: capture itself executes
                # nothing, so this is what makes the capture call's outputs and
                # KV/seq_lens mutations eager-equivalent.
                for _fr14_si, (_fr14_g, _fr14_sig, _fr14_np) in enumerate(
                    _fr14_segs
                ):
                    _fr14_g.replay()
                    if _fr13_is_fixed32:
                        _fr13_f32_dg_gdn._fr13_fixed32_drafter_graph_replay(
                            id(_fr14_g),
                            _fr14_sig,
                            _fr13_dg_key,
                            _fr14_np,
                            _fr14_si,
                        )
                        _fr13_dh_m32_note_production_replay(
                            id(_fr14_g),
                            _fr14_sig,
                            _fr13_dg_key,
                        )
                        _fr13_dh_u8_note_production_replay(
                            id(_fr14_g),
                            _fr14_sig,
                            _fr13_dg_key,
                        )
                        _fr13_dh_fp8_note_replay(
                            id(_fr14_g),
                            _fr14_sig,
                            _fr13_dg_key,
                        )
                        from vllm.v1.attention.ops.triton_unified_attention import (
                            _fr13_dfwd_unified_bm8_live_replay,
                        )
                        _fr13_dfwd_unified_bm8_live_replay(
                            id(_fr14_g),
                            _fr13_dg_key,
                        )
                # `lo` is the segment every step replays, so the single-graph
                # handles stay bound to it.
                _fr13_dg_g = _fr14_segs[0][0]
                _dg["fixed32_signature"] = _fr14_segs[0][1]
                _dg["segments"] = _fr14_segs
                _dg["graph"] = _fr13_dg_g
                _fr13_dg_all[_fr13_dg_key] = _dg
                if _fr13_dfwd_top3:
                    if (
                        self._fr13_dfwd_top3_capture_calls != 4
                        or self._fr13_dfwd_top3_root_calls < 1
                    ):
                        raise RuntimeError(
                            "FR13 DFWD K64 top3 graph capture count drifted"
                        )
                    print(
                        "[FR13_DFWD_K64_TOP3] graph captured_calls=4 "
                        "root_calls_at_least=1",
                        flush=True,
                    )
                if _fr13_is_fixed32:
                    try:
                        _fr13_f32_dg_gdn._fr13_dfwd_unified_bm8_production_replay_installed(
                            id(_fr13_dg_g),
                            _fr13_dg_key,
                            _dg["fixed32_signature"],
                        )
                    except Exception:
                        _fr13_dg_all.pop(_fr13_dg_key, None)
                        _dg["graph"] = None
                        raise
                print(
                    f"[FR13_DRAFTER_GRAPH] captured bs={_fr13_dg_key} "
                    "(full 4-iter spine loop, inner model flat-recorded)",
                    flush=True,
                )

            if _fr13_ds_on:
                torch.cuda.synchronize()
                _fr13_ds_t1 = __import__("time").monotonic()
            _fr13_hydra_path_tokens = {}
            _fr13_fixed32_path_tokens = {}
            # accept>5 TAIL append (sidecar /logs/fr13_tail_mode.arm): the native MTP head loop
            # above produced head_depth spine tokens (depths 0..head_depth-1) + head branches
            # (byte-identical baseline). Retrieve the deep Arctic chain (depths head_depth..wide_D-1)
            # and APPEND to _fr10_spine_tokens (head_depth -> wide_D) so the wide packer sees wide_D
            # spine tensors. Standard tail mode is add-only. HYDRA23 trades two
            # head siblings for four conditional nodes, so it is distribution-
            # lossless through the parent-aware committer but not accept-monotone
            # versus tail6. Own try/except: NEVER break the drafter.
            if _fr10_is_wide and os.path.exists("/logs/fr13_tail_mode.arm"):
                _fr13_t_skip = ""
                _fr13_hydra_contract_error = False
                try:
                    import sys as _fr13_t_sys
                    if "/workspace/scripts" not in _fr13_t_sys.path:
                        _fr13_t_sys.path.insert(0, "/workspace/scripts")
                    import fr13_merged_drafter as _fr13_t
                    # FR14_GATE_SPLIT_GRAPH: on a gated step only two post-root
                    # forwards ran, so the MTP head is 3 deep and Arctic owns
                    # draft positions 3..10 -- an 8-token main chain. Everything
                    # else in this block is already parametric in _fr13_t_hd,
                    # including the rank-1/rank-2 root seeds at
                    # _fr13_t_stack[_fr13_t_hd (+1)], so the seam moves with it.
                    # the head is ALWAYS this many depths of physical
                    # columns; _fr13_t_hd is how many of them MTP actually
                    # filled this step. Every identity below derives from the
                    # pair -- none of them may restate either as a literal.
                    _fr13_t_hd_full = int(
                        getattr(_fr13_t, "TAIL_HEAD_DEPTH", 5)
                    )
                    _fr13_t_hd = (
                        _fr14_gate_mtp_k if _fr14_gate_fired
                        else _fr13_t_hd_full
                    )
                    _fr13_t_len = int(_fr10_wide_D) - _fr13_t_hd
                    if not _fr13_t.merged_on():
                        _fr13_t_skip = "merged_off"
                    elif not (_fr13_t_len > 0 and len(_fr10_spine_tokens) == _fr13_t_hd):
                        _fr13_t_skip = "geom_len%d_hd%d_tlen%d" % (len(_fr10_spine_tokens), _fr13_t_hd, _fr13_t_len)
                    else:
                        from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr13_t_gdn
                        _fr13_t_cache = _fr13_t.get_cache()
                        _fr13_t_B = int(_fr10_spine_tokens[0].shape[0])
                        if _fr13_is_fixed32:
                            # The proposal wrapper owns the current full padded
                            # batch. Never let a stale compact spec-row list win.
                            _fr13_t_ids = getattr(
                                _fr13_t_gdn,
                                "_LUMO_FA_SAMPLER_ROW_REQ_IDS",
                                None,
                            )
                            _fr13_t_proposal = getattr(
                                _fr13_t_gdn,
                                "_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT",
                                None,
                            )
                            if (
                                not isinstance(_fr13_t_proposal, dict)
                                or _fr13_t_ids is None
                                or len(_fr13_t_ids) != _fr13_t_B
                                or tuple(str(_r) for _r in _fr13_t_ids)
                                != tuple(_fr13_t_proposal.get("request_ids", ()))
                                or int(_fr13_t_proposal.get("batch_size", -1))
                                != _fr13_t_B
                            ):
                                raise RuntimeError(
                                    "FR13 fixed32 drafter row-owner drift: "
                                    + repr(
                                        (
                                            _fr13_t_ids,
                                            _fr13_t_B,
                                            _fr13_t_proposal,
                                        )
                                    )
                                )
                        else:
                            _fr13_t_ids = getattr(
                                _fr13_t_gdn,
                                "_LUMO_FA_SPEC_ROW_REQ_IDS",
                                None,
                            )
                            # B=4 mixed prefill/decode: compact spec IDs can be
                            # shorter than the padded proposal rows.
                            if (
                                _fr13_t_ids is None
                                or len(_fr13_t_ids) != _fr13_t_B
                            ):
                                _fr13_t_ids = getattr(
                                    _fr13_t_gdn,
                                    "_LUMO_FA_SAMPLER_ROW_REQ_IDS",
                                    None,
                                )
                        if _fr13_t_cache is None:
                            _fr13_t_skip = "cache_none"
                        elif _fr13_t_ids is None:
                            _fr13_t_skip = "ids_none"
                        elif len(_fr13_t_ids) != _fr13_t_B:
                            _fr13_t_skip = "ids%d_ne_B%d" % (len(_fr13_t_ids), _fr13_t_B)
                        else:
                            # FR13_TAIL_HOSTCOPY_BATCHED (2026-07-25): ONE
                            # stacked DtoH instead of hd separate .cpu()
                            # syncs + one .item() — each was a full stream
                            # sync waiting on the drafter forwards (and the
                            # R4 replay). Values byte-identical.
                            _fr13_t_host_cols = [
                                _fr10_spine_tokens[_d].detach().reshape(-1)
                                for _d in range(_fr13_t_hd)
                            ]
                            if _fr13_is_hydra23:
                                _fr13_hydra_contract_error = True
                                _fr13_t_root_wtk = _fr10_wide_topk.get(0)
                                if (
                                    _fr13_t_root_wtk is None
                                    or int(_fr13_t_root_wtk.shape[1]) < 2
                                ):
                                    raise RuntimeError(
                                        "FR13_HYDRA23 root topk needs column 1"
                                    )
                                # Keep the rescue seed in the existing
                                # single stacked DtoH synchronization.
                                _fr13_t_host_cols.append(
                                    _fr13_t_root_wtk[:, 1].detach().reshape(-1),
                                )
                            elif _fr13_is_fixed32:
                                _fr13_t_root_wtk = _fr10_wide_topk.get(0)
                                if (
                                    _fr13_t_root_wtk is None
                                    or int(_fr13_t_root_wtk.shape[1]) < 3
                                ):
                                    raise RuntimeError(
                                        "FR13 fixed32 root topk needs ranks 1 and 2"
                                    )
                                _fr13_t_host_cols.extend(
                                    [
                                        _fr13_t_root_wtk[:, 1].detach().reshape(-1),
                                        _fr13_t_root_wtk[:, 2].detach().reshape(-1),
                                    ]
                                )
                            _fr13_t_accept_host = None
                            _fr13_t_next_host = None
                            if _fr13_is_fixed32:
                                _fr13_t_step_seq = int(getattr(
                                    _fr13_t_gdn,
                                    "_LUMO_FA_STEP_SEQ",
                                    -1,
                                ))
                                _fr13_t_accept = getattr(
                                    _fr13_t_gdn,
                                    "_FR13_FIXED32_ACCEPTED_OUTPUT_CURRENT",
                                    None,
                                )
                                _fr13_t_spec_B = 0
                                if _fr13_t_accept is not None:
                                    _fr13_t_spec_ids = tuple(
                                        str(_r) for _r in
                                        _fr13_t_accept.get("request_ids", ())
                                    )
                                    _fr13_t_full_ids = tuple(
                                        str(_r) for _r in
                                        _fr13_t_accept.get(
                                            "full_request_ids", ()
                                        )
                                    )
                                    _fr13_t_accept_tokens = (
                                        _fr13_t_accept.get("output_tokens")
                                    )
                                    _fr13_t_accept_lens = (
                                        _fr13_t_accept.get("output_lens")
                                    )
                                    _fr13_t_spec_B = len(_fr13_t_spec_ids)
                                    if (
                                        set(_fr13_t_accept)
                                        != {
                                            "step_seq",
                                            "request_ids",
                                            "full_request_ids",
                                            "output_tokens",
                                            "output_lens",
                                        }
                                        or int(_fr13_t_accept.get(
                                            "step_seq", -1
                                        )) != _fr13_t_step_seq
                                        or _fr13_t_full_ids
                                        != tuple(str(_r) for _r in _fr13_t_ids)
                                        or not 1 <= _fr13_t_spec_B <= _fr13_t_B
                                        or len(set(_fr13_t_spec_ids))
                                        != _fr13_t_spec_B
                                        or any(
                                            _r not in _fr13_t_full_ids
                                            for _r in _fr13_t_spec_ids
                                        )
                                        or not torch.is_tensor(
                                            _fr13_t_accept_tokens
                                        )
                                        or not torch.is_tensor(
                                            _fr13_t_accept_lens
                                        )
                                        or tuple(_fr13_t_accept_tokens.shape)
                                        != (_fr13_t_spec_B, 32)
                                        or tuple(_fr13_t_accept_lens.shape)
                                        != (_fr13_t_spec_B,)
                                        or _fr13_t_accept_tokens.dtype
                                        != torch.int64
                                        or _fr13_t_accept_lens.dtype
                                        != torch.int64
                                        or _fr13_t_accept_tokens.device
                                        != _fr10_spine_tokens[0].device
                                        or _fr13_t_accept_lens.device
                                        != _fr10_spine_tokens[0].device
                                    ):
                                        raise RuntimeError(
                                            "FR13 fixed32 accepted-output "
                                            "device record drift"
                                        )
                                _fr13_t_host_matrix = torch.stack(
                                    _fr13_t_host_cols
                                )
                                _fr13_t_payload_parts = [
                                    _fr13_t_host_matrix.reshape(-1)
                                ]
                                if _fr13_t_accept is not None:
                                    _fr13_t_payload_parts.extend(
                                        [
                                            _fr13_t_accept_tokens.reshape(-1),
                                            _fr13_t_accept_lens.reshape(-1),
                                        ]
                                    )
                                _fr13_t_payload_parts.append(
                                    next_token_ids.detach().reshape(-1).to(
                                        dtype=_fr13_t_host_matrix.dtype
                                    )
                                )
                                _fr13_t_payload = torch.cat(
                                    _fr13_t_payload_parts
                                ).cpu()
                                _fr13_t_offset = (
                                    len(_fr13_t_host_cols) * _fr13_t_B
                                )
                                _fr13_t_stack = _fr13_t_payload[
                                    :_fr13_t_offset
                                ].reshape(
                                    len(_fr13_t_host_cols),
                                    _fr13_t_B,
                                )
                                if _fr13_t_accept is not None:
                                    _fr13_t_accept_count = (
                                        _fr13_t_spec_B * 32
                                    )
                                    _fr13_t_accept_rows_host = (
                                        _fr13_t_payload[
                                            _fr13_t_offset:
                                            _fr13_t_offset
                                            + _fr13_t_accept_count
                                        ].reshape(_fr13_t_spec_B, 32)
                                    )
                                    _fr13_t_offset += _fr13_t_accept_count
                                    _fr13_t_accept_lens_host = (
                                        _fr13_t_payload[
                                            _fr13_t_offset:
                                            _fr13_t_offset
                                            + _fr13_t_spec_B
                                        ]
                                    )
                                    _fr13_t_offset += _fr13_t_spec_B
                                    _fr13_t_accept_host = {
                                        "step_seq": _fr13_t_step_seq,
                                        "request_ids": _fr13_t_spec_ids,
                                        "full_request_ids": _fr13_t_full_ids,
                                        "output_rows": tuple(
                                            tuple(int(_x) for _x in _row)
                                            for _row in
                                            _fr13_t_accept_rows_host.tolist()
                                        ),
                                        "output_lens": tuple(
                                            int(_x) for _x in
                                            _fr13_t_accept_lens_host.tolist()
                                        ),
                                    }
                                _fr13_t_next_host = tuple(
                                    int(_x) for _x in
                                    _fr13_t_payload[
                                        _fr13_t_offset:
                                        _fr13_t_offset + _fr13_t_B
                                    ].tolist()
                                )
                                if (
                                    len(_fr13_t_next_host) != _fr13_t_B
                                    or _fr13_t_offset + _fr13_t_B
                                    != int(_fr13_t_payload.numel())
                                ):
                                    raise RuntimeError(
                                        "FR13 fixed32 host lifecycle payload drift"
                                    )
                            else:
                                _fr13_t_stack = torch.stack(
                                    _fr13_t_host_cols
                                ).cpu()
                            _fr13_hydra_contract_error = False
                            _fr13_t_head = [
                                [int(_x) for _x in _fr13_t_stack[_d].tolist()]
                                for _d in range(_fr13_t_hd)]
                            _fr13_t_hydra_seeds = (
                                {
                                    1: [
                                        int(_x)
                                        for _x in _fr13_t_stack[_fr13_t_hd].tolist()
                                    ],
                                }
                                if _fr13_is_hydra23
                                else (
                                    {
                                        1: [
                                            int(_x)
                                            for _x in _fr13_t_stack[
                                                _fr13_t_hd
                                            ].tolist()
                                        ],
                                        2: [
                                            int(_x)
                                            for _x in _fr13_t_stack[
                                                _fr13_t_hd + 1
                                            ].tolist()
                                        ],
                                    }
                                    if _fr13_is_fixed32 else None
                                )
                            )
                            # FR13_DVK_DRAFTID_DUMP (instrumented runs only):
                            # spine draft ids are ALREADY on host here — one
                            # json line per drafter call, zero extra syncs.
                            # Run with DVK off => full-head argmax ids => the
                            # whole K/subset accept-curve computes offline.
                            _fr13_dvkd = os.environ.get(
                                "FR13_DVK_DRAFTID_DUMP", "")
                            if _fr13_dvkd:
                                try:
                                    _fr13_dvkd_fh = getattr(
                                        self, "_fr13_dvkd_fh", None)
                                    if _fr13_dvkd_fh is None:
                                        _fr13_dvkd_fh = self._fr13_dvkd_fh = (
                                            open(_fr13_dvkd, "a", buffering=1))
                                    _fr13_dvkd_fh.write(
                                        __import__("json").dumps(
                                            _fr13_t_head) + chr(10))
                                except Exception:
                                    self._fr13_dvkd_fh = None
                            _fr13_t_vocab = int(
                                _fr10_logits.shape[-1]
                                if _fr13_full_vocab_size is None
                                else _fr13_full_vocab_size
                            )
                            _fr13_t_pad = int(_fr13_t_stack[0, 0])
                            if _fr13_is_fixed32:
                                _fr13_t.finalize_fixed32_step(
                                    _fr13_t_cache,
                                    [str(_r) for _r in _fr13_t_ids],
                                    _fr13_t_accept_host,
                                    _fr13_t_next_host,
                                    _fr13_t_step_seq,
                                    _fr13_t_vocab,
                                )
                                _fr13_t_work_serial = (
                                    _fr13_t.get_fixed32_drafter_work_serial()
                                )
                                _fr13_t_cols = _fr13_t.decide_fixed32(
                                    _fr13_t_cache,
                                    [str(_r) for _r in _fr13_t_ids],
                                    _fr13_t_head,
                                    _fr13_t_hydra_seeds,
                                    _fr10_spine_tokens[0].device,
                                    _fr13_t_pad,
                                    vocab_size=_fr13_t_vocab,
                                    gated=_fr14_gate_fired,
                                )
                                _fr13_t_work = (
                                    _fr13_t.get_fixed32_drafter_last_work()
                                )
                                if (
                                    not isinstance(_fr13_t_work, dict)
                                    or int(_fr13_t_work.get("serial", -1))
                                    != int(_fr13_t_work_serial) + 1
                                ):
                                    raise RuntimeError(
                                        "FR13 fixed32 drafter work snapshot is stale"
                                    )
                                _fr13_f32_dg_gdn._fr13_fixed32_drafter_observed_arctic(
                                    _fr13_t_work
                                )
                            else:
                                _fr13_t_cols = _fr13_t.decide_tail(
                                    _fr13_t_cache, [str(_r) for _r in _fr13_t_ids], _fr13_t_head,
                                    _fr13_t_hd, _fr13_t_len, _fr10_spine_tokens[0].device,
                                    _fr13_t_pad, vocab_size=_fr13_t_vocab,
                                    hydra_seed_per_rank=_fr13_t_hydra_seeds,
                                    hydra_branch_chains=((1, 4),))
                            if _fr13_t_cols is not None:
                                _fr10_spine_tokens.extend(_fr13_t_cols)
                                if _fr14_gate_fired:
                                    # The four depth-4/5 runner-up columns have
                                    # no MTP logits behind them any more. Fill
                                    # them by repeating their own spine token --
                                    # the last-resort pad this drafter already
                                    # deploys, whose committer tie convention is
                                    # proven on device by
                                    # scripts/fr13_greedy_pointmass_dup_gate.py.
                                    # A repeat can never match a DISTINCT model
                                    # token, so the tree stays monotone-lossless
                                    # and the validity mask never changes.
                                    for _fr14_pd in (3, 4):
                                        _fr14_pw = int(
                                            _fr10_wide_width.get(_fr14_pd, 1)
                                        )
                                        if _fr14_pw > 1:
                                            _fr10_wide_topk[_fr14_pd] = (
                                                _fr10_spine_tokens[_fr14_pd]
                                                .reshape(-1, 1)
                                                .repeat(1, _fr14_pw)
                                            )
                                    if len(_fr10_spine_tokens) != int(
                                        _fr10_wide_D
                                    ):
                                        raise RuntimeError(
                                            "FR14 gated step published "
                                            + repr(len(_fr10_spine_tokens))
                                            + " spine columns, expected "
                                            + repr(int(_fr10_wide_D))
                                        )
                                if _fr13_is_hydra23:
                                    _fr13_hydra_contract_error = True
                                    _fr13_t_paths = dict(
                                        _fr13_t.get_tail_path_tokens()
                                    )
                                    if set(_fr13_t_paths) != set(
                                        _fr13_hydra23_tail_paths
                                    ):
                                        raise RuntimeError(
                                            "FR13_HYDRA23 Arctic path set mismatch: "
                                            + repr(sorted(_fr13_t_paths))
                                        )
                                    for _fr13_hp, _fr13_ht in _fr13_t_paths.items():
                                        if (
                                            tuple(_fr13_ht.shape)
                                            != tuple(_fr10_spine_tokens[0].shape)
                                            or _fr13_ht.dtype != torch.int64
                                            or _fr13_ht.device
                                            != _fr10_spine_tokens[0].device
                                        ):
                                            raise RuntimeError(
                                                "FR13_HYDRA23 bad column for "
                                                + repr(_fr13_hp)
                                            )
                                    _fr13_hydra_path_tokens = _fr13_t_paths
                                    _fr13_hydra_contract_error = False
                                elif _fr13_is_fixed32:
                                    _fr13_t_paths = dict(
                                        _fr13_t.get_tail_path_tokens()
                                    )
                                    if set(_fr13_t_paths) != set(
                                        _fr13_fixed32_tail_paths
                                    ):
                                        raise RuntimeError(
                                            "FR13 fixed32 Arctic path set mismatch: "
                                            + repr(sorted(_fr13_t_paths))
                                        )
                                    # FR14_GATE_SPLIT_GRAPH (14th site). The
                                    # head is _fr13_t_hd_full*3 physical columns
                                    # HOWEVER they are filled; on a gated step
                                    # (_fr13_t_hd_full - _fr13_t_hd) of the
                                    # Arctic columns land in the head as spine
                                    # tokens, so counting them again as tail
                                    # columns double-counts and the identity
                                    # reads 33. The 31 does NOT vary with the
                                    # pass count -- the pack width is fixed by
                                    # the topology -- so it stays a literal;
                                    # the 15 did vary, and no longer appears.
                                    _fr14_head_cols = _fr13_t_hd_full * 3
                                    _fr14_arctic_in_head = (
                                        _fr13_t_hd_full - _fr13_t_hd
                                    )
                                    if (
                                        _fr14_head_cols
                                        + (
                                            len(_fr13_t_cols)
                                            - _fr14_arctic_in_head
                                        )
                                        + len(_fr13_t_paths)
                                        != 31
                                    ):
                                        raise RuntimeError(
                                            "FR13 fixed32 drafter pack is not "
                                            "31 columns: head="
                                            + repr(_fr14_head_cols)
                                            + " arctic="
                                            + repr(len(_fr13_t_cols))
                                            + " in_head="
                                            + repr(_fr14_arctic_in_head)
                                            + " rescue="
                                            + repr(len(_fr13_t_paths))
                                        )
                                    _fr13_fixed32_path_tokens = _fr13_t_paths
                                # Direction-2 d6-branch: merge the tail-branch wide_topk (free packer keys
                                # head_depth-1.. ; the head fills only 0..head_depth-2) so the wide packer
                                # fills the d6/d7 tail branch nodes from the arctic runner-ups. Empty {}
                                # when FR13_TAIL_BRANCHES is unset (spine-only == tail6, byte-identical, no
                                # drift). Monotone-lossless: branches only ADD candidates (committer p(S)).
                                _fr13_t_wtk = _fr13_t.get_tail_wide_topk()
                                if _fr13_t_wtk:
                                    _fr10_wide_topk.update(_fr13_t_wtk)
                            else:
                                _fr13_t_skip = "decide_none"
                except Exception as _fr13_t_exc:
                    if _fr13_is_fixed32:
                        raise
                    if _fr13_is_hydra23 and _fr13_hydra_contract_error:
                        raise
                    _fr13_t_skip = "exc:%r" % (_fr13_t_exc,)
                # POST-TRY RECONCILIATION (OUTSIDE the try -- red-team FINDING 1, make-or-break): tail
                # mode serves a wide_D-deep tree, so the wide packer REQUIRES wide_D spine tensors with
                # NO fall-back-to-head. If the append produced fewer (skip/cold/exc), PAD to wide_D by
                # repeating the last spine token -- a [batch] int64 tensor (torch.stack needs matching
                # shape), NOT a scalar. Lossless: a repeated deep token ~never matches the model past the
                # head, and the committer verifies each row against its OWN target -> never a wrong accept.
                if _fr13_is_fixed32:
                    if _fr13_t_skip:
                        raise RuntimeError(
                            "FR13 fixed32 drafter skipped: " + _fr13_t_skip
                        )
                    if len(_fr10_spine_tokens) != int(_fr10_wide_D):
                        raise RuntimeError(
                            "FR13 fixed32 spine pack is not exactly depth 11"
                        )
                    if set(_fr13_fixed32_path_tokens) != set(
                        _fr13_fixed32_tail_paths
                    ):
                        raise RuntimeError(
                            "FR13 fixed32 rescue columns are incomplete"
                        )
                else:
                    while len(_fr10_spine_tokens) < int(_fr10_wide_D):
                        _fr10_spine_tokens.append(_fr10_spine_tokens[-1])
                if _fr13_is_hydra23:
                    _fr13_h_root = _fr10_wide_topk.get(0)
                    if _fr13_h_root is None or int(_fr13_h_root.shape[1]) < 2:
                        raise RuntimeError(
                            "FR13_HYDRA23 cannot reconcile without root topk"
                        )
                    for _fr13_h_rank, _fr13_h_len in ((1, 4),):
                        _fr13_h_prev = _fr13_h_root[:, _fr13_h_rank]
                        for _fr13_h_j in range(_fr13_h_len):
                            _fr13_h_path = (
                                (_fr13_h_rank,)
                                + (0,) * (_fr13_h_j + 1)
                            )
                            _fr13_h_col = _fr13_hydra_path_tokens.get(
                                _fr13_h_path
                            )
                            if (
                                _fr13_h_col is None
                                or tuple(_fr13_h_col.shape)
                                != tuple(_fr13_h_prev.shape)
                                or _fr13_h_col.dtype != torch.int64
                                or _fr13_h_col.device != _fr13_h_prev.device
                            ):
                                _fr13_h_col = _fr13_h_prev
                            _fr13_hydra_path_tokens[_fr13_h_path] = _fr13_h_col
                            _fr13_h_prev = _fr13_h_col
                # Direction-2 branch PAD-fallback (mirror the spine pad above): the branched tail tree has
                # branch nodes (rk>0) whose _fr10_wide_topk keys are filled by decide_tail's merge ONLY when
                # the tail path engaged; on a cold/dummy forward (decide_tail None/skip) those keys are absent
                # and the packer fail-louds. Pad any missing branch key with the (pad-reconciled) spine token
                # repeated -> LOSSLESS (a repeated token never matches a DISTINCT model token; committer p(S)).
                # Head branches (pp<head_depth) are MTP-filled so present; this only backstops the tail. No-op
                # for the spine-only tail (its plan has no rk>0 past the head) => byte-identical for tail6.
                for _fr13_pp, _fr13_rk in _fr10_wide_plan:
                    if _fr13_rk <= 0:
                        continue
                    _fr13_wt = _fr10_wide_topk.get(_fr13_pp)
                    if _fr13_wt is None or _fr13_rk >= int(_fr13_wt.shape[1]):
                        _fr13_w = max(int(_fr13_rk) + 1, 3)
                        _fr10_wide_topk[_fr13_pp] = (
                            _fr10_spine_tokens[_fr13_pp].unsqueeze(1).expand(-1, _fr13_w).contiguous())
                if _fr13_t_skip:   # fail-LOUD (throttled) so the real guard miss surfaces, not the packer error
                    try:
                        globals()["_fr13_tail_skip_n"] = globals().get("_fr13_tail_skip_n", 0) + 1
                        if globals()["_fr13_tail_skip_n"] % 50 == 1:
                            import logging as _fr13_t_log
                            _fr13_t_log.getLogger("vllm.fr13_tail").warning(
                                "[FR13_TAIL] append skipped x%d (padded to wide_D=%s): %s",
                                globals()["_fr13_tail_skip_n"], _fr10_wide_D, _fr13_t_skip)
                    except Exception:
                        pass

            # FR13_DEDUP_SIBLINGS: enforce DISTINCT tokens within each sibling
            # group (nodes sharing a parent_pos). Native topk branches are
            # distinct by construction; the merged/arctic/pad fills can re-emit
            # an already-used token -> a wasted verify node AND a temp-0 argmax
            # TIE (where the greedy max-LCP committer and the per-node rejection
            # committer legitimately differ). The target argmax is UNIQUE, so
            # making siblings distinct guarantees at most one matches => no tie
            # => point-mass rejection == greedy byte-for-byte. Correctness-safe:
            # ANY tree is lossless (committer verifies each row vs its OWN
            # target). ONE batched collision check (1 sync) => no-op fast path
            # for distinct-by-construction configs (deployed tail6). Repairs a
            # collided branch from the widened spare topk ranks; if none spare,
            # a distinct dead-end token (a repeated token can never out-match a
            # distinct model token so this only ever REMOVES a wasted node).
            if _fr13_dedup_sib and _fr10_is_wide:
                _fr13_dd_any = None
                _fr13_dd_groups = []
                for _fr13_gp in sorted({_pp for _pp, _rk in _fr10_wide_plan if _rk > 0}):
                    _fr13_w = int(_fr10_wide_width.get(_fr13_gp, 1))
                    _fr13_wt0 = _fr10_wide_topk.get(_fr13_gp)
                    if _fr13_w <= 1 or _fr13_wt0 is None:
                        continue
                    _fr13_cols = int(_fr13_wt0.shape[1])
                    _fr13_sp = _fr10_spine_tokens[_fr13_gp]
                    _fr13_used = [_fr13_sp]
                    _fr13_gc = torch.zeros_like(_fr13_sp, dtype=torch.bool)
                    for _fr13_r in range(1, min(_fr13_w, _fr13_cols)):
                        _fr13_c = _fr13_wt0[:, _fr13_r]
                        for _fr13_u in _fr13_used:
                            _fr13_gc = _fr13_gc | (_fr13_c == _fr13_u)
                        _fr13_used.append(_fr13_c)
                    _fr13_dd_any = _fr13_gc if _fr13_dd_any is None else (_fr13_dd_any | _fr13_gc)
                    _fr13_dd_groups.append((_fr13_gp, _fr13_w, _fr13_cols))
                if _fr13_dd_any is not None and bool(_fr13_dd_any.any().item()):
                    _fr13_vsz = int(
                        _fr10_logits.shape[-1]
                        if _fr13_full_vocab_size is None
                        else _fr13_full_vocab_size
                    )
                    _fr13_nb = int(_fr10_spine_tokens[0].shape[0])
                    for _fr13_gp, _fr13_w, _fr13_cols in _fr13_dd_groups:
                        _fr13_wt = _fr10_wide_topk[_fr13_gp].clone()
                        _fr13_sp = _fr10_spine_tokens[_fr13_gp]
                        _fr13_seen = [[int(_fr13_sp[_b].item())] for _b in range(_fr13_nb)]
                        for _fr13_r in range(1, min(_fr13_w, _fr13_cols)):
                            for _fr13_b in range(_fr13_nb):
                                _fr13_tk = int(_fr13_wt[_fr13_b, _fr13_r].item())
                                if _fr13_tk not in _fr13_seen[_fr13_b]:
                                    _fr13_seen[_fr13_b].append(_fr13_tk)
                                    continue
                                _fr13_nw = None
                                for _fr13_sr in range(_fr13_w, _fr13_cols):
                                    _fr13_cd = int(_fr13_wt[_fr13_b, _fr13_sr].item())
                                    if _fr13_cd not in _fr13_seen[_fr13_b]:
                                        _fr13_nw = _fr13_cd
                                        break
                                if _fr13_nw is None:
                                    _fr13_cd = (_fr13_tk + 1) % _fr13_vsz
                                    while _fr13_cd in _fr13_seen[_fr13_b]:
                                        _fr13_cd = (_fr13_cd + 1) % _fr13_vsz
                                    _fr13_nw = _fr13_cd
                                _fr13_wt[_fr13_b, _fr13_r] = _fr13_nw
                                _fr13_seen[_fr13_b].append(_fr13_nw)
                        _fr10_wide_topk[_fr13_gp] = _fr13_wt

            if _fr10_is_spine_only or _fr10_is_chain3:
                # FR13_RESHAPE_DEPTH3 chain3 = pure depth-3 spine = the same
                # spine-only stack, just depth 3 (3 tokens, no leaves). The
                # sorted (len, path) order [(0,), (0,0), (0,0,0)] == the spine
                # token order [spine0, spine1, spine2], so the plain stack is
                # the correct flat packing.
                _fr10_packed = torch.stack(_fr10_spine_tokens, dim=1)
            elif _fr10_is_cat3w:
                # FR13_RESHAPE_DEPTH3 cat3w: 5-node flat order = vLLM sorted
                # (len, path) tree order. Node id (0-based slot) -> path:
                #   0 (0,)   spine0 (root draft)
                #   1 (1,)   ROOT_LEAF (root runner-up)
                #   2 (0,0)  spine1 (depth-2 spine, loop step 0)
                #   3 (0,1)  leaf0  (depth-1 runner-up, loop step 0 top2[:,1])
                #   4 (0,0,0) spine2 (depth-3 spine, loop step 1)
                if _fr10_root_leaf_token is None:
                    raise RuntimeError(
                        "FR13_RESHAPE_DEPTH3 cat3w engaged but root runner-up "
                        "token was not captured (drafter packing would be "
                        "vacuous)"
                    )
                _fr10_packed = torch.stack(
                    [
                        _fr10_spine_tokens[0],
                        _fr10_root_leaf_token,
                        _fr10_spine_tokens[1],
                        _fr10_leaf_tokens[0],
                        _fr10_spine_tokens[2],
                    ],
                    dim=1,
                )
            elif _fr10_is_cat6root:
                # FR13_RESHAPE_DEPTH5 cat6root: 6-node flat order = vLLM
                # sorted (len, path) tree order. Node id (0-based slot) ->
                # path:
                #   0 (0,)         spine0 (root draft)
                #   1 (1,)         ROOT_LEAF (root runner-up)
                #   2 (0,0)        spine1 (loop step 0)
                #   3 (0,0,0)      spine2 (loop step 1)
                #   4 (0,0,0,0)    spine3 (loop step 2)
                #   5 (0,0,0,0,0)  spine4 (loop step 3)
                # No interior leaves (_fr10_leaf_steps == {}); the only
                # off-spine node is the root sibling.
                if _fr10_root_leaf_token is None:
                    raise RuntimeError(
                        "FR13_RESHAPE_DEPTH5 cat6root engaged but root "
                        "runner-up token was not captured (drafter packing "
                        "would be vacuous)"
                    )
                _fr10_packed = torch.stack(
                    [
                        _fr10_spine_tokens[0],
                        _fr10_root_leaf_token,
                        _fr10_spine_tokens[1],
                        _fr10_spine_tokens[2],
                        _fr10_spine_tokens[3],
                        _fr10_spine_tokens[4],
                    ],
                    dim=1,
                )
            elif _fr10_is_cat10:
                # FR13_RESHAPE_DEPTH5 cat10: 10-node flat order = vLLM sorted
                # (len, path) tree order = cat9 with the (1,) root sibling
                # inserted at slot 1. Node id (0-based slot) -> path:
                #   0 (0,)         spine0 (root draft)
                #   1 (1,)         ROOT_LEAF (root runner-up)
                #   2 (0,0)        spine1 (loop step 0)
                #   3 (0,1)        leaf0  (d1 runner-up, step 0 top2[:,1])
                #   4 (0,0,0)      spine2 (loop step 1)
                #   5 (0,0,1)      leaf1  (d2 runner-up, step 1 top2[:,1])
                #   6 (0,0,0,0)    spine3 (loop step 2)
                #   7 (0,0,0,1)    leaf2  (d3 runner-up, step 2 top2[:,1])
                #   8 (0,0,0,0,0)  spine4 (loop step 3)
                #   9 (0,0,0,0,1)  leaf3  (d4 runner-up, step 3 top2[:,1])
                if _fr10_root_leaf_token is None:
                    raise RuntimeError(
                        "FR13_RESHAPE_DEPTH5 cat10 engaged but root runner-up "
                        "token was not captured (drafter packing would be "
                        "vacuous)"
                    )
                _fr10_packed = torch.stack(
                    [
                        _fr10_spine_tokens[0],
                        _fr10_root_leaf_token,
                        _fr10_spine_tokens[1],
                        _fr10_leaf_tokens[0],
                        _fr10_spine_tokens[2],
                        _fr10_leaf_tokens[1],
                        _fr10_spine_tokens[3],
                        _fr10_leaf_tokens[2],
                        _fr10_spine_tokens[4],
                        _fr10_leaf_tokens[3],
                    ],
                    dim=1,
                )
            elif _fr10_is_333:
                # FR13_RESHAPE_333: 9-node flat order = vLLM sorted (len, path)
                # tree order. 3 candidates per depth (spine, rank-1, rank-2).
                # Node id (0-based slot) -> path:
                #   0 (0,)     spine0 (root draft = argmax)
                #   1 (1,)     ROOT_LEAF  (root rank-1, root topk[:,1])
                #   2 (2,)     ROOT_LEAF2 (root rank-2, root topk[:,2])
                #   3 (0,0)    spine1 (loop step 0 argmax)
                #   4 (0,1)    leaf0  (d2 rank-1, step 0 topk[:,1])
                #   5 (0,2)    leaf2_0 (d2 rank-2, step 0 topk[:,2])
                #   6 (0,0,0)  spine2 (loop step 1 argmax)
                #   7 (0,0,1)  leaf1  (d3 rank-1, step 1 topk[:,1])
                #   8 (0,0,2)  leaf2_1 (d3 rank-2, step 1 topk[:,2])
                # _fr10_leaf_tokens = [d2_rank1, d3_rank1] (steps {1,2});
                # _fr10_leaf2_tokens = [d2_rank2, d3_rank2] (steps {1,2}).
                if (
                    _fr10_root_leaf_token is None
                    or _fr10_root_leaf2_token is None
                ):
                    raise RuntimeError(
                        "FR13_RESHAPE_333 engaged but root runner-up tokens "
                        "(rank-1/rank-2) were not captured (drafter packing "
                        "would be vacuous)"
                    )
                if len(_fr10_leaf_tokens) != 2 or len(_fr10_leaf2_tokens) != 2:
                    raise RuntimeError(
                        "FR13_RESHAPE_333 engaged but interior leaf lists are "
                        "the wrong length (rank-1="
                        + str(len(_fr10_leaf_tokens))
                        + " rank-2="
                        + str(len(_fr10_leaf2_tokens))
                        + ", expected 2 each)"
                    )
                _fr10_packed = torch.stack(
                    [
                        _fr10_spine_tokens[0],
                        _fr10_root_leaf_token,
                        _fr10_root_leaf2_token,
                        _fr10_spine_tokens[1],
                        _fr10_leaf_tokens[0],
                        _fr10_leaf2_tokens[0],
                        _fr10_spine_tokens[2],
                        _fr10_leaf_tokens[1],
                        _fr10_leaf2_tokens[1],
                    ],
                    dim=1,
                )
            elif _fr10_is_wide:
                # FR13_RESHAPE_WIDE general packer: emit one column per
                # tree_choices node in sorted (len, path) order (== the
                # tree_choices order vLLM already requires). rank 0 -> the
                # spine token at depth parent_pos+1 (_fr10_spine_tokens[
                # parent_pos]); rank k>0 -> the leaf = that parent spine
                # node's topk column k (_fr10_wide_topk[parent_pos][:, k]).
                # FAIL-LOUD if a needed spine token or topk column is missing
                # (would otherwise mis-pack silently).
                if len(_fr10_spine_tokens) != _fr10_wide_D:
                    raise RuntimeError(
                        "FR13_RESHAPE_WIDE: collected "
                        + str(len(_fr10_spine_tokens))
                        + " spine tokens, expected D=" + str(_fr10_wide_D)
                )
                _fr10_wide_cols = []
                for _fr10_path, (_fr10_pp, _fr10_rk) in zip(
                    _fr10_wide_paths, _fr10_wide_plan
                ):
                    if (
                        (
                            _fr13_is_hydra23
                            and _fr10_path in _fr13_hydra_path_tokens
                        )
                        or (
                            _fr13_is_fixed32
                            and _fr10_path in _fr13_fixed32_path_tokens
                        )
                    ):
                        _fr10_wide_cols.append(
                            (
                                _fr13_fixed32_path_tokens[_fr10_path]
                                if _fr13_is_fixed32
                                else _fr13_hydra_path_tokens[_fr10_path]
                            )
                        )
                    elif _fr10_rk == 0:
                        _fr10_wide_cols.append(_fr10_spine_tokens[_fr10_pp])
                    else:
                        _fr10_wt = _fr10_wide_topk.get(_fr10_pp)
                        if (
                            _fr10_wt is None
                            or _fr10_rk >= int(_fr10_wt.shape[1])
                        ):
                            raise RuntimeError(
                                "FR13_RESHAPE_WIDE packing: missing topk "
                                "rank=" + str(_fr10_rk) + " at parent_pos="
                                + str(_fr10_pp) + " (width store "
                                + ("absent" if _fr10_wt is None
                                   else "cols=" + str(int(_fr10_wt.shape[1])))
                                + ")"
                            )
                        _fr10_wide_cols.append(_fr10_wt[:, _fr10_rk])
                if _fr13_ds_on:
                    torch.cuda.synchronize()
                    _fr13_ds_t2 = __import__("time").monotonic()
                _fr10_packed = torch.stack(_fr10_wide_cols, dim=1)
                if _fr13_is_fixed32:
                    _fr13_f32_dg_gdn._fr13_fixed32_drafter_observed_publish(
                        tuple(int(_d) for _d in _fr10_packed.shape),
                        str(_fr10_packed.dtype),
                        _fr10_packed.device.type,
                        tuple(_fr10_wide_paths),
                    )
                if _fr13_ds_on:
                    torch.cuda.synchronize()
                    _fr13_ds_t3 = __import__("time").monotonic()
                    _fr13_ds_c = globals().get("_FR13_DS_CNT", 0) + 1
                    globals()["_FR13_DS_CNT"] = _fr13_ds_c
                    _fr13_ds_acc = globals().get("_FR13_DS_ACC", [0.0, 0.0, 0.0])
                    _fr13_ds_acc[0] += _fr13_ds_t1 - _fr13_ds_t0
                    _fr13_ds_acc[1] += _fr13_ds_t2 - _fr13_ds_t1
                    _fr13_ds_acc[2] += _fr13_ds_t3 - _fr13_ds_t2
                    globals()["_FR13_DS_ACC"] = _fr13_ds_acc
                    if _fr13_ds_c % 256 == 1:
                        print(
                            "[FR13_DFWD_SPLIT] n=%d loop=%.1fms tail+mid=%.1fms pack=%.1fms"
                            % (
                                _fr13_ds_c,
                                1e3 * _fr13_ds_acc[0] / _fr13_ds_c,
                                1e3 * _fr13_ds_acc[1] / _fr13_ds_c,
                                1e3 * _fr13_ds_acc[2] / _fr13_ds_c,
                            ),
                            flush=True,
                        )
            else:
                _fr10_packed = torch.stack(
                    [
                        _fr10_spine_tokens[0],
                        _fr10_spine_tokens[1],
                        _fr10_leaf_tokens[0],
                        _fr10_spine_tokens[2],
                        _fr10_leaf_tokens[1],
                        _fr10_spine_tokens[3],
                        _fr10_leaf_tokens[2],
                        _fr10_spine_tokens[4],
                        _fr10_leaf_tokens[3],
                    ],
                    dim=1,
                )
            try:
                import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
                if _fr10_lo.environ.get("FR10_METRICS", "0") == "1":
                    global _LUMO_CATERPILLAR_DRAFTER_FH
                    try:
                        _LUMO_CATERPILLAR_DRAFTER_FH
                    except NameError:
                        _LUMO_CATERPILLAR_DRAFTER_FH = open(
                            _fr10_lo.environ.get(
                                "LUMO_CATERPILLAR_DRAFTER_LOG",
                                "/logs/fr10_caterpillar_drafter.jsonl",
                            ),
                            "a",
                            buffering=1,
                        )
                    _LUMO_CATERPILLAR_DRAFTER_FH.write(
                        _fr10_lj.dumps({
                            "event": "fr10_caterpillar_native_spine_top2",
                            "ts": round(_fr10_lt.time(), 4),
                            "single_logits": bool(_fr13_single_logits),
                            "spine_only": bool(_fr10_is_spine_only),
                            # FR13_RESHAPE_DEPTH3: shape tag + cat9 cosmetic
                            # slot maps retained for cat9; new shapes report
                            # their own num_spec via the draft width.
                            "shape": (
                                "chain3" if _fr10_is_chain3
                                else "cat3w" if _fr10_is_cat3w
                                else "cat6root" if _fr10_is_cat6root
                                else "cat10" if _fr10_is_cat10
                                else "333" if _fr10_is_333
                                else "wide" if _fr10_is_wide
                                else "chain5" if _fr10_is_spine_only
                                else "cat9"
                            ),
                            "spine_slots": [0, 1, 3, 5, 7],
                            "leaf_slots": [2, 4, 6, 8],
                            "draft": _fr10_packed.detach().cpu().tolist(),
                            "spine": torch.stack(_fr10_spine_tokens, dim=1).detach().cpu().tolist(),
                            "leaves": (
                                []
                                if not _fr10_leaf_tokens
                                else torch.stack(_fr10_leaf_tokens, dim=1).detach().cpu().tolist()
                            ),
                            # FR13_RESHAPE_333: rank-2 (child-rank-2) leaves;
                            # empty for every shape except 3-3-3.
                            "leaves2": (
                                []
                                if not _fr10_leaf2_tokens
                                else torch.stack(_fr10_leaf2_tokens, dim=1).detach().cpu().tolist()
                            ),
                            "root_leaf2": (
                                None
                                if _fr10_root_leaf2_token is None
                                else _fr10_root_leaf2_token.detach().cpu().tolist()
                            ),
                        }) + chr(10)
                    )
            except Exception:
                pass
            _e8_finish(self, _fr10_packed, _E8_ARM, _E8_QUALIFY)
            return _fr10_packed

        _fr10_tree_draft_branch_seen = any(
            isinstance(md, TreeAttentionMetadata) for md in per_group_attn_metadata
        )
        _fr10_tree_expected = (
            _fr10_active_decode_mode == "tree_mtp"
            and (
                "speculative_token_tree" in os.environ.get("SPEC_CONFIG", "")
                or len(_fr10_tree_choices_current) == 9
            )
        )
        if (
            _fr10_tree_expected
            and not (
                _fr10_is_caterpillar
                or _fr10_is_spine_only
                or _fr10_is_chain3
                or _fr10_is_cat3w
                or _fr10_is_cat6root
                or _fr10_is_cat10
                or _fr10_is_333
                or _fr10_is_wide
            )
            and os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1"
        ):
            raise RuntimeError(
                "FR10 caterpillar drafter disengaged: "
                + "num_speculative_tokens="
                + str(int(self.num_speculative_tokens))
                + " tree_choices="
                + repr(_fr10_tree_choices_current)
            )
        try:
            import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
            if _fr10_lo.environ.get("FR10_METRICS", "0") == "1":
                global _LUMO_TREE_DRAFT_BRANCH_FH
                try:
                    _LUMO_TREE_DRAFT_BRANCH_FH
                except NameError:
                    _LUMO_TREE_DRAFT_BRANCH_FH = open(
                        _fr10_lo.environ.get(
                            "LUMO_TREE_DRAFT_BRANCH_LOG",
                            "/logs/fr10_tree_draft_branch.jsonl",
                        ),
                        "a",
                        buffering=1,
                    )
                _LUMO_TREE_DRAFT_BRANCH_FH.write(
                    _fr10_lj.dumps({
                        "event": "tree_draft_branch",
                        "ts": round(_fr10_lt.time(), 4),
                        "tree_branch_seen": bool(_fr10_tree_draft_branch_seen),
                        "metadata_types": [
                            type(md).__name__ for md in per_group_attn_metadata
                        ],
                        "num_speculative_tokens": int(self.num_speculative_tokens),
                        "speculative_token_tree": _fr10_lo.environ.get("SPEC_CONFIG"),
                    }) + chr(10)
                )
        except Exception:
            pass

        if _fr10_tree_draft_branch_seen:
            # FR10_TREE_DRAFT_CONSUMPTION_VERIFY: no separate spine overlay.
            # vLLM's native tree drafter consumes the MTP head directly:
            # child-rank 0 is the fed-back spine, child-rank 1 is recorded as
            # the side leaf and must not advance the spine recurrent state.
            logits = self.model.compute_logits(sample_hidden_states)
            if 'propose() tree-branch' not in _FR13_DFWD_SPLIT.pathmap:
                _FR13_DFWD_SPLIT.pathmap.add('propose() tree-branch')
                try:
                    from vllm.logger import init_logger as _il
                    _il('vllm.fr13_dfwd_split').info('FR13_PATHMAP: %s', 'propose() tree-branch')
                except Exception:
                    pass
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
                slot_mappings=slot_mappings,
            )
            # [batch_size, num_tree_tokens]
            return torch.cat(draft_token_ids_list, dim=1)

        draft_token_ids = self._greedy_sample(sample_hidden_states)

        if self.allowed_attn_types is not None:
            for group_md in per_group_attn_metadata:
                if not isinstance(group_md, self.allowed_attn_types):
                    raise ValueError(
                        f"Unsupported attention metadata type for speculative "
                        "decoding with num_speculative_tokens > 1: "
                        f"{type(group_md)}. Supported types are: "
                        f"{self.allowed_attn_types}"
                    )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size)
        )

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            # Invalidate the CPU-side shadows to avoid H<>D sync.
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        block_size = self.block_size
        assert block_size > 0, "block_size has not been initialized."
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            # Use fused kernel for slot mapping and metadata updates.
            # Write clamped positions directly into the positions buffer to
            # avoid an extra D2D copy for the common (non-mrope) case.
            positions_1d = positions[0] if self.uses_mrope else positions
            if self.uses_mrope:
                out_pos = self.mrope_positions[0, :batch_size]
            elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                out_pos = self.xdrope_positions[0, :batch_size]
            else:
                out_pos = self.positions[:batch_size]
            eagle_step_update_slot_mapping_and_metadata(
                positions_1d=positions_1d,
                block_table_tensor=common_attn_metadata.block_table_tensor,
                seq_lens=common_attn_metadata.seq_lens,
                block_size=block_size,
                max_model_len=self.max_model_len,
                out_clamped_positions=out_pos,
                out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
                input_batch_size=input_batch_size,
            )
            common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]
            if self.uses_mrope:
                self.mrope_positions[1:, :batch_size] = self.mrope_positions[
                    0, :batch_size
                ]
                positions = self.mrope_positions[:, :batch_size]
            elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[
                    0, :batch_size
                ]
                positions = self.xdrope_positions[0, :batch_size]
            else:
                positions = self.positions[:batch_size]
            # Increment the maximum sequence length. We increment max_seq_len
            # unconditionally even though some seq_lens may have been capped above,
            # as max_seq_len serves as an upper bound for sequence lengths.
            common_attn_metadata.max_seq_len = min(
                common_attn_metadata.max_seq_len + 1, self.max_model_len
            )

            # Also update the CPU-side shadow; NOTE: this is hacky and should be
            # removed in when common_attn_metadata.seq_lens_cpu is deprecated.
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu += 1
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu += 1

            # Rebuild attention metadata
            _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
                common_attn_metadata, draft_index=token_index + 1
            )

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # Run the model.
            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        if not self.needs_extra_input_slots:
            # Default EAGLE pathway: no reshaping of input tensors needed.
            # Simply rotate the input ids and leave the positions unchanged,
            # Inserting the next token ids at the last slot in each request.
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]
            # Shift the input ids by one token.
            # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            # Replace the last token with the next token.
            # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
            self.input_ids[token_indices_to_sample] = next_token_ids

            # copy inputs to buffer for cudagraph
            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]
            self._set_positions(num_tokens, target_positions)

            self.hidden_states[:num_tokens] = target_hidden_states

            return num_tokens, token_indices_to_sample, cad
        else:
            assert self.is_rejected_token_mask is not None
            assert self.is_masked_token_mask is not None
            # 1.
            # Call a custom triton kernel to copy input_ids and positions
            # into the correct slots in the preallocated buffers self.input_ids,
            # self.positions.
            batch_size = cad.batch_size()
            # Since we might have to copy a lot of data for prefills, we select the
            # block size based on the max query length and limit to max 256 slots/block.
            max_num_tokens_per_request = (
                cad.max_query_len + self.net_num_new_slots_per_request
            )
            BLOCK_SIZE_TOKENS = min(256, next_power_of_2(max_num_tokens_per_request))
            num_blocks = (
                max_num_tokens_per_request + BLOCK_SIZE_TOKENS - 1
            ) // BLOCK_SIZE_TOKENS
            total_num_input_tokens = target_token_ids.shape[0]
            total_num_output_tokens = total_num_input_tokens + (
                self.net_num_new_slots_per_request * batch_size
            )

            token_indices_to_sample = torch.empty(
                batch_size * self.extra_slots_per_request,
                dtype=torch.int32,
                device=self.device,
            )

            # Destination indices to write target_hidden_states into drafting buffer.
            out_hidden_state_mapping = torch.empty(
                total_num_input_tokens, dtype=torch.int32, device=self.device
            )

            # Kernel grid: one program per request (row)
            grid = (batch_size, num_blocks)
            query_start_loc = cad.query_start_loc
            query_end_loc = cad.query_start_loc[1:] - 1
            if num_rejected_tokens_gpu is not None:
                query_end_loc = query_end_loc - num_rejected_tokens_gpu

            copy_and_expand_eagle_inputs_kernel[grid](
                # (Padded) Inputs from the target model
                target_token_ids_ptr=target_token_ids,
                target_positions_ptr=target_positions,
                next_token_ids_ptr=next_token_ids,  # sampled tokens, one per request
                # Outputs to the drafting buffers
                out_input_ids_ptr=self.input_ids,
                out_positions_ptr=self.positions,  # Doesn't support mrope for now
                out_is_rejected_token_mask_ptr=self.is_rejected_token_mask,
                out_is_masked_token_mask_ptr=self.is_masked_token_mask,
                out_new_token_indices_ptr=token_indices_to_sample,
                out_hidden_state_mapping_ptr=out_hidden_state_mapping,
                # Input metadata
                query_start_loc_ptr=query_start_loc,
                query_end_loc_ptr=query_end_loc,
                padding_token_id=0,
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                # Sizing info
                # Note that we can deduce batch_size for free from the grid size
                total_input_tokens=total_num_input_tokens,
                num_padding_slots_per_request=self.extra_slots_per_request,
                shift_input_ids=self.pass_hidden_states_to_model,
                BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
            )
            if self.pass_hidden_states_to_model:
                assert self.parallel_drafting_hidden_state_tensor is not None
                self.hidden_states[out_hidden_state_mapping] = target_hidden_states
                # Use torch.where to avoid DtoH sync from boolean indexing
                mask = self.is_masked_token_mask[:total_num_output_tokens]
                torch.where(
                    mask.unsqueeze(1),
                    self.parallel_drafting_hidden_state_tensor,
                    self.hidden_states[:total_num_output_tokens],
                    out=self.hidden_states[:total_num_output_tokens],
                )

            # 2.
            # Recompute the slot mapping based on the new positions and
            # rejection mask.
            assert self.block_size > 0, "block_size has not been initialized."
            new_slot_mapping = compute_new_slot_mapping(
                cad=cad,
                new_positions=self.positions[:total_num_output_tokens],
                is_rejected_token_mask=self.is_rejected_token_mask[
                    :total_num_output_tokens
                ],
                block_size=self.block_size,
                num_new_tokens=self.net_num_new_slots_per_request,
                max_model_len=self.max_model_len,
            )

            # 3. Update the common attention metadata with the new (meta)data
            new_cad = extend_all_queries_by_N(
                cad,
                N=self.net_num_new_slots_per_request,
                arange=self.arange,
                new_slot_mapping=new_slot_mapping,
            )

            return total_num_output_tokens, token_indices_to_sample, new_cad

    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]

        return model_kwargs, num_input_tokens

    def build_per_group_and_layer_attn_metadata(
        self, common_attn_metadata: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=draft_index
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def model_returns_tuple(self) -> bool:
        return self.method not in ("mtp", "draft_model", "dflash")

    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids for each request based on the sampled
        token ids from the CPU. If a request has no sampled token ids (e.g.,
        during the initial decoding steps), it falls back to using the request
        state to get the next token id.
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[req_id]
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.int32, device=self.input_ids.device
        )
        return next_token_ids

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead. This is denoted
        the "backup" token id. It also counts rejected tokens via `sampled_token_ids`.
        """
        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
        seq_lens_list = (gpu_input_batch.num_tokens_no_spec[:num_reqs] - 1).tolist()
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(seq_lens_list[i])
                for i in range(num_reqs)
            ],
            dtype=np.int32,
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        next_token_ids = torch.empty(batch_size, dtype=torch.int32, device=device)
        valid_sampled_tokens_count = next_token_ids.new_empty(batch_size)

        # Kernel grid: one program per request (row)
        grid = (batch_size,)

        # Find the next power of 2 for block sizes
        BLOCK_SIZE_TOKENS = next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[grid](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        grid = (num_reqs,)
        eagle_prepare_inputs_padded_kernel[grid](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.max_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    def propose_tree(
        self,
        batch_size: int,
        # [num_tokens, vocab_size]
        logits: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        # [num_tokens, hidden_size]
        hidden_states: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> list[torch.Tensor]:
        if 'propose_tree ENTERED' not in _FR13_DFWD_SPLIT.pathmap:
            _FR13_DFWD_SPLIT.pathmap.add('propose_tree ENTERED')
            try:
                from vllm.logger import init_logger as _il
                _il('vllm.fr13_dfwd_split').info('FR13_PATHMAP: %s', 'propose_tree ENTERED')
            except Exception:
                pass
        tree_attn_metadata_builder = self.draft_attn_groups[0].get_metadata_builder()
        assert isinstance(tree_attn_metadata_builder, TreeAttentionMetadataBuilder)

        total_num_drafts = self.cu_drafts_per_level[0]
        level_num_drafts = total_num_drafts
        # Sample a draft token for each child at the tree root level.
        num_children = self.child_drafts_per_level[0]
        if num_children == 1:
            draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
        else:
            draft_token_ids = torch.topk(logits, num_children, dim=-1).indices.view(
                batch_size, -1
            )
        draft_token_ids_list = [draft_token_ids]
        # FR10_TREE_DRAFT_CONSUMPTION_VERIFY: runtime node placement and q.
        _fr10_depth_choices = {}
        for _fr10_choice in self.tree_choices:
            _fr10_depth_choices.setdefault(len(_fr10_choice), []).append(tuple(_fr10_choice))
        _fr10_runtime_slot_by_choice = {
            tuple(_fr10_choice): int(_fr10_i)
            for _fr10_i, _fr10_choice in enumerate(self.tree_choices)
        }

        def _fr10_record_tree_consumption(
            _fr10_depth,
            _fr10_level_logits,
            _fr10_level_tokens,
            _fr10_level_num_parents,
            _fr10_level_num_children,
        ):
            import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
            if _fr10_lo.environ.get("FR10_METRICS", "0") != "1":
                return
            _fr10_now = round(_fr10_lt.time(), 4)
            def _fr10_log_consumption_error(_fr10_exc):
                global _LUMO_TREE_DRAFT_CONSUMPTION_ERR_FH
                try:
                    _LUMO_TREE_DRAFT_CONSUMPTION_ERR_FH
                except NameError:
                    _LUMO_TREE_DRAFT_CONSUMPTION_ERR_FH = open(
                        _fr10_lo.environ.get(
                            "LUMO_TREE_DRAFT_CONSUMPTION_ERROR_LOG",
                            "/logs/fr10_tree_draft_consumption_errors.jsonl",
                        ),
                        "a",
                        buffering=1,
                    )
                _LUMO_TREE_DRAFT_CONSUMPTION_ERR_FH.write(
                    _fr10_lj.dumps({
                        "event": "tree_draft_consumption_error",
                        "ts": _fr10_now,
                        "depth": int(_fr10_depth),
                        "level_tokens_shape": [
                            int(_x) for _x in getattr(_fr10_level_tokens, "shape", [])
                        ],
                        "level_logits_shape": [
                            int(_x) for _x in getattr(_fr10_level_logits, "shape", [])
                        ],
                        "level_num_parents": int(_fr10_level_num_parents),
                        "level_num_children": int(_fr10_level_num_children),
                        "error_type": type(_fr10_exc).__name__,
                        "error": str(_fr10_exc),
                    }) + chr(10)
                )
            try:
                global _LUMO_TREE_DRAFT_CONSUMPTION_FH
                try:
                    _LUMO_TREE_DRAFT_CONSUMPTION_FH
                except NameError:
                    _LUMO_TREE_DRAFT_CONSUMPTION_FH = open(
                        _fr10_lo.environ.get(
                            "LUMO_TREE_DRAFT_CONSUMPTION_LOG",
                            "/logs/fr10_tree_draft_consumption.jsonl",
                        ),
                        "a",
                        buffering=1,
                    )
                _fr10_choices = _fr10_depth_choices.get(int(_fr10_depth), [])
                _fr10_parent_choices = (
                    [tuple()]
                    if int(_fr10_depth) == 1
                    else _fr10_depth_choices.get(int(_fr10_depth) - 1, [])
                )
                _fr10_parent_slot = {
                    tuple(_fr10_choice): int(_fr10_i)
                    for _fr10_i, _fr10_choice in enumerate(_fr10_parent_choices)
                }
                _fr10_top = torch.topk(
                    _fr10_level_logits, int(_fr10_level_num_children), dim=-1
                )
                for _fr10_choice in _fr10_choices:
                    _fr10_parent = tuple(_fr10_choice[:-1])
                    _fr10_rank = int(_fr10_choice[-1])
                    _fr10_p_slot = int(_fr10_parent_slot.get(_fr10_parent, -1))
                    _fr10_flat_col = (
                        _fr10_p_slot * int(_fr10_level_num_children) + _fr10_rank
                    )
                    if (
                        _fr10_p_slot < 0
                        or _fr10_rank >= int(_fr10_level_num_children)
                        or _fr10_flat_col >= int(_fr10_level_tokens.size(1))
                    ):
                        _LUMO_TREE_DRAFT_CONSUMPTION_FH.write(
                            _fr10_lj.dumps({
                                "event": "tree_draft_consumption",
                                "ts": _fr10_now,
                                "depth": int(_fr10_depth),
                                "path": [int(_x) for _x in _fr10_choice],
                                "parent_path": [int(_x) for _x in _fr10_parent],
                                "child_rank": int(_fr10_rank),
                                "runtime_slot": int(_fr10_runtime_slot_by_choice.get(_fr10_choice, -1)),
                                "placement_ok": False,
                                "reason": "choice_not_represented_by_level_topk",
                            }) + chr(10)
                        )
                        continue
                    for _fr10_b in range(int(batch_size)):
                        _fr10_row = (
                            int(_fr10_b) * int(_fr10_level_num_parents) + _fr10_p_slot
                        )
                        _fr10_placed = _fr10_level_tokens[_fr10_b, _fr10_flat_col]
                        _fr10_expected = _fr10_top.indices[_fr10_row, _fr10_rank]
                        _fr10_probs = torch.softmax(
                            _fr10_level_logits[_fr10_row].float(), dim=-1
                        )
                        _fr10_q = _fr10_probs[_fr10_placed.to(torch.long)]
                        _LUMO_TREE_DRAFT_CONSUMPTION_FH.write(
                            _fr10_lj.dumps({
                                "event": "tree_draft_consumption",
                                "ts": _fr10_now,
                                "req_index": int(_fr10_b),
                                "depth": int(_fr10_depth),
                                "path": [int(_x) for _x in _fr10_choice],
                                "parent_path": [int(_x) for _x in _fr10_parent],
                                "runtime_slot": int(_fr10_runtime_slot_by_choice.get(_fr10_choice, -1)),
                                "parent_runtime_slot": int(_fr10_runtime_slot_by_choice.get(_fr10_parent, -1)),
                                "parent_level_slot": int(_fr10_p_slot),
                                "child_rank": int(_fr10_rank),
                                "rank_kind": "top1" if int(_fr10_rank) == 0 else ("top2" if int(_fr10_rank) == 1 else "topN"),
                                "flat_level_col": int(_fr10_flat_col),
                                "placed_token": int(_fr10_placed.detach().cpu().item()),
                                "expected_topk_token": int(_fr10_expected.detach().cpu().item()),
                                "q_prob": float(_fr10_q.detach().cpu().item()),
                                "placement_ok": bool(
                                    int(_fr10_placed.detach().cpu().item())
                                    == int(_fr10_expected.detach().cpu().item())
                                ),
                            }) + chr(10)
                        )
            except Exception as _fr10_exc:
                _fr10_log_consumption_error(_fr10_exc)

        _fr10_record_tree_consumption(1, logits, draft_token_ids, 1, num_children)
        draft_hidden_states = hidden_states.view(batch_size, 1, -1)

        # Initialize empty tensors for concatenation with the level outputs.
        tree_input_ids = torch.empty(
            0, device=self.input_ids.device, dtype=self.input_ids.dtype
        )
        tree_positions = torch.empty(
            0, device=self.positions.device, dtype=self.positions.dtype
        )
        tree_hidden_states = torch.empty(
            0, device=self.hidden_states.device, dtype=self.hidden_states.dtype
        )
        # Precompute the draft token positions.
        flattened_draft_positions = (
            positions.view(batch_size, -1) + self.tree_draft_pos_offsets[:batch_size, :]
        )
        tree_depth = len(self.cu_drafts_per_level)
        for level in range(tree_depth - 1):
            # Get draft positions for RoPE.
            draft_positions = positions + (level + 1)
            exceeds_max_model_len = (positions + total_num_drafts) >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            draft_positions = torch.where(
                exceeds_max_model_len,
                0,
                draft_positions,
            ).view(batch_size, -1)

            if level_num_drafts > 1:
                # Repeat the positions for each draft at this level.
                draft_positions = draft_positions.repeat_interleave(
                    level_num_drafts, dim=1
                )

            if num_children > 1:
                # Repeat draft hidden states for each child.
                draft_hidden_states = draft_hidden_states.repeat_interleave(
                    num_children, dim=1
                )

            # Concatenate the draft tokens, positions, and hidden states.
            tree_input_ids = torch.cat([tree_input_ids, draft_token_ids], dim=1)
            tree_positions = torch.cat([tree_positions, draft_positions], dim=1)
            tree_hidden_states = torch.cat(
                [tree_hidden_states, draft_hidden_states], dim=1
            )

            # Build new attention metadata for the next level of drafts.
            # This is necessary to support tree attention.
            query_len = total_num_drafts
            common_attn_metadata = replace(
                common_attn_metadata,
                query_start_loc=query_len * self.arange[: batch_size + 1],
                seq_lens=common_attn_metadata.seq_lens + level_num_drafts,
                num_actual_tokens=batch_size * query_len,
                max_query_len=query_len,
            )
            attn_metadata = tree_attn_metadata_builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=level + 1
            )

            # Apply new attention metadata to all draft layers.
            per_layer_attn_metadata = {}
            for attn_group in self.draft_attn_groups:
                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata

            # Consider max model length.
            attn_metadata.max_seq_len = min(
                attn_metadata.max_seq_len, self.max_model_len
            )
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            block_size = tree_attn_metadata_builder.kv_cache_spec.block_size
            query_positions = flattened_draft_positions[:, level : level + query_len]
            block_numbers = query_positions // block_size
            block_ids = attn_metadata.block_table.gather(dim=1, index=block_numbers)
            slot_mapping = block_ids * block_size + query_positions % block_size
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping[exceeds_max_model_len] = PADDING_SLOT_ID
            attn_metadata.slot_mapping = slot_mapping.view(-1)

            # Copy inputs to buffer for cudagraph.
            num_tokens = attn_metadata.num_actual_tokens
            input_ids = tree_input_ids.view(-1)
            self.input_ids[:num_tokens] = input_ids
            self.positions[:num_tokens] = tree_positions.view(-1)
            self.hidden_states[:num_tokens] = tree_hidden_states.view(num_tokens, -1)

            cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                num_tokens
            )
            num_input_tokens = batch_desc.num_tokens
            # Run the model.
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    num_input_tokens, attn_metadata.slot_mapping
                ),
            ):
                last_hidden_states, hidden_states = self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=None,
                )

            # Get the output hidden states for the draft tokens.
            draft_hidden_states = hidden_states[:num_tokens].view(
                batch_size, query_len, -1
            )[:, -level_num_drafts:]
            draft_last_hidden_states = last_hidden_states[:num_tokens].view(
                batch_size, query_len, -1
            )[:, -level_num_drafts:]

            # Get the output logits for the draft tokens.
            logits = self.model.compute_logits(
                draft_last_hidden_states.reshape(batch_size * level_num_drafts, -1)
            )

            # Sample a draft token for each child at the next tree level.
            num_children = self.child_drafts_per_level[level + 1]
            if num_children == 1:
                draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
            else:
                draft_token_ids = torch.topk(logits, num_children, dim=-1).indices.view(
                    batch_size, -1
                )
            draft_token_ids_list.append(draft_token_ids)
            _fr10_record_tree_consumption(
                level + 2, logits, draft_token_ids, level_num_drafts, num_children
            )

            # Update the # drafts counters for the next tree level.
            level_num_drafts = self.cu_drafts_per_level[level + 1] - total_num_drafts
            total_num_drafts = self.cu_drafts_per_level[level + 1]
        return draft_token_ids_list

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
            for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_seq_lens_cpu = common_attn_metadata.seq_lens_cpu - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offsets = (
            self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded
        )

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offsets + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=new_seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices

    def get_model_name(self, model: nn.Module) -> str:
        if hasattr(model, "module"):  # multi-GPU
            model = model.module
        return model.__class__.__name__

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Return a VllmConfig with kernel-level overrides for the proposer.
        Subclasses may override to apply additional config changes.
        """
        spec_cfg = self.speculative_config
        if spec_cfg.moe_backend is not None:
            return replace(
                self.vllm_config,
                kernel_config=replace(
                    self.vllm_config.kernel_config,
                    moe_backend=spec_cfg.moe_backend,
                ),
            )
        return self.vllm_config

    def _get_model(self) -> nn.Module:
        """
        Default method to call get_model(). Can be overridden by subclasses which
        need to customize model loading.
        """
        from vllm.compilation.backends import set_model_tag

        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("eagle_head"):
            model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
            )
        return model

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        self.model = self._get_model()

        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self._draft_attn_layer_names = (
            set(all_attn_layers.keys()) - target_attn_layer_names
        )

        if self.supports_mm_inputs:
            # Even if the target model is multimodal, we can also use
            # text-only draft models
            try:
                dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
                self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
            except (NotImplementedError, AttributeError, TypeError):
                logger.warning(
                    "Draft model does not support multimodal inputs, "
                    "falling back to text-only mode"
                )
                self.supports_mm_inputs = False

        if supports_multimodal(target_model):
            # handle multimodality
            assert hasattr(target_model, "config")
            if self.get_model_name(target_model) in [
                "Exaone4_5_ForConditionalGeneration",
                "GlmOcrForConditionalGeneration",
                "HunYuanVLForConditionalGeneration",
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
                "Gemma4ForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            elif self.get_model_name(target_model) == "KimiK25ForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.media_placeholder_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
            target_language_model = cast(
                SupportsMultiModal, target_model
            ).get_language_model()
        else:
            target_language_model = target_model

        self._maybe_share_embeddings(target_language_model)
        self._maybe_share_lm_head(target_language_model)

        if (
            self.parallel_drafting
            and self.pass_hidden_states_to_model
            and self.parallel_drafting_hidden_state_tensor is not None
        ):
            flat_mask = self.model.mask_hidden.view(-1)
            if self.eagle3_use_aux_hidden_state:
                # EAGLE3: mask_hidden stores all aux hidden states,
                # project through combine_hidden_states
                self.parallel_drafting_hidden_state_tensor.copy_(
                    self.model.combine_hidden_states(flat_mask)
                )
            else:
                self.parallel_drafting_hidden_state_tensor.copy_(flat_mask)

    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own embedding layers, and some may
        have a duplicate copy of the target model's embedding layers. In these cases,
        we share the target model's embedding layers with the draft model to save
        memory.
        """
        if get_pp_group().world_size == 1:
            inner_model = getattr(target_language_model, "model", None)
            if inner_model is None:
                raise AttributeError("Target model does not have 'model' attribute")
            if hasattr(inner_model, "embed_tokens"):
                target_embed_tokens = inner_model.embed_tokens
            elif hasattr(inner_model, "embedding"):
                target_embed_tokens = inner_model.embedding
            else:
                raise AttributeError(
                    "Target model does not have 'embed_tokens' or 'embedding' attribute"
                )

            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE model
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # TODO: Offload to CPU for comparison to avoid extra GPU memory
                    # usage in CI testing environments with limited GPU memory
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP model
                share_embeddings = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model embedding weights with the draft model."
                )

            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own LM head, and some may have a
        duplicate copy of the target model's LM head. In these cases, we share
        the target model's LM head with the draft model to save memory.
        """
        share_lm_head = False
        if hasattr(self.model, "has_own_lm_head"):
            # EAGLE model
            if not self.model.has_own_lm_head:
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model without its own lm_head in the checkpoint. "
                    "Sharing target model lm_head weights with the draft model."
                )
            elif (
                hasattr(target_language_model, "lm_head")
                and hasattr(target_language_model.lm_head, "weight")
                and hasattr(self.model.lm_head, "weight")
                and isinstance(target_language_model.lm_head.weight, torch.Tensor)
                and isinstance(self.model.lm_head.weight, torch.Tensor)
                # TODO: Offload to CPU for comparison to avoid extra GPU memory
                # usage in CI testing environments with limited GPU memory
                and torch.equal(
                    target_language_model.lm_head.weight.cpu(),
                    self.model.lm_head.weight.cpu(),
                )
            ):
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model with lm_head identical to the target model. "
                    "Sharing target model lm_head weights with the draft model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct lm_head weights. "
                    "Keeping separate lm_head weights from the target model."
                )
        else:
            # MTP model
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )

        if share_lm_head and hasattr(target_language_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_language_model.lm_head

            # MTP models call compute_logits via shared_head.head (a
            # ParallelLMHead inside each MTP layer), not self.model.lm_head.
            # If the checkpoint omits a copy of the lm_head weights at the
            # MTP layer path, shared_head.head stays uninitialised and
            # produces NaN logits. Always share it explicitly.
            inner = getattr(self.model, "model", None)
            layers = getattr(inner, "layers", None) if inner else None
            if layers is not None:
                items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
                for layer in items:
                    sh = getattr(layer, "shared_head", None)
                    if sh is not None and hasattr(sh, "head"):
                        del sh.head
                        sh.head = target_language_model.lm_head
                        logger.info(
                            "Shared target model lm_head with MTP shared_head.head."
                        )

        if self.use_local_argmax_reduction:
            if not hasattr(self.model, "get_top_tokens"):
                raise ValueError(
                    "use_local_argmax_reduction is enabled but draft model "
                    f"{self.model.__class__.__name__} does not implement "
                    "get_top_tokens()."
                )
            # Warn if draft model has vocab remapping, which forces fallback
            # to the full-logits path (negating the optimization).
            if (
                hasattr(self.model, "draft_id_to_target_id")
                and self.model.draft_id_to_target_id is not None
            ):
                logger.warning(
                    "use_local_argmax_reduction is enabled but draft model "
                    "uses draft_id_to_target_id vocab remapping. The "
                    "optimization will be bypassed (falling back to full "
                    "logits gather + argmax)."
                )
            else:
                logger.info(
                    "Using local argmax reduction for draft token generation "
                    "(communication: O(2*tp_size) vs O(vocab_size))."
                )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        # FIXME: when using tree-based specdec, adjust number of forward-passes
        # according to the depth of the tree.
        only_one_forward_pass = is_graph_capturing or self.parallel_drafting
        for fwd_idx in range(
            1 if only_one_forward_pass else self.num_speculative_tokens
        ):
            if fwd_idx <= 1:
                cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                    self._determine_batch_execution_and_padding(
                        num_tokens, use_cudagraphs=use_cudagraphs
                    )
                )

            # Make sure to use EAGLE's own buffer during cudagraph capture.
            if (
                self._draft_attn_layer_names
                and slot_mappings is not None
                and next(iter(self._draft_attn_layer_names)) in slot_mappings
            ):
                slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
            else:
                slot_mapping_dict = slot_mappings or {}

            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                kwargs = dict(
                    input_ids=input_ids,
                    positions=self._get_positions(num_input_tokens),
                    inputs_embeds=inputs_embeds,
                )
                if self.pass_hidden_states_to_model:
                    kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]
                self.model(**kwargs)

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """
        Some eagle3 heads (e.g., nvidia/gpt-oss-120b-Eagle3-v2) do not use auxiliary
        hidden states and directly uses the last layer output just like eagle1.
        They might indicate this by setting "use_aux_hidden_state" to False
        inside the "eagle_config" dict of their hf_config.
        """
        if self.method != "eagle3":
            return False
        # Assume that eagle3 heads use aux hidden states by default
        use_aux_hidden_state = True
        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is not None:
            use_aux_hidden_state = eagle_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all drafting layers belong to the same KVCacheGroup.
        Need this assumption to ensure all drafting layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self._draft_attn_layer_names
                    ]
                )
            )
            == 1
        ), "All drafting layers should belong to the same kv cache group"

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """
        Initialize AttentionGroups for draft layers using kv_cache_config.
        Called from the model runner's initialize_metadata_builders.
        """
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        # Find which kv_cache_group the draft layers belong to
        self.validate_same_kv_cache_group(kv_cache_config)
        kv_cache_spec = None
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if self._draft_attn_layer_names & set(group.layer_names):
                self.kv_cache_gid = gid
                kv_cache_spec = group.kv_cache_spec
                break

        attention_groups: dict[tuple[str, str], AttentionGroup] = {}
        if kv_cache_spec is not None:
            for layer_name in self._draft_attn_layer_names:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                backend_key = attn_backend.full_cls_name()
                if backend_key not in attention_groups:
                    layer_kv_cache_spec = kv_cache_spec
                    if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                        layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                            layer_name
                        ]

                    kernel_block_size = (
                        kernel_block_sizes[self.kv_cache_gid]
                        if kernel_block_sizes is not None
                        and self.kv_cache_gid < len(kernel_block_sizes)
                        else None
                    )
                    attn_group = AttentionGroup(
                        backend=attn_backend,
                        layer_names=[layer_name],
                        kv_cache_spec=layer_kv_cache_spec,
                        kv_cache_group_id=self.kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    attention_groups[backend_key] = attn_group
                else:
                    attention_groups[backend_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        self.block_size = (
            self.draft_attn_groups[0].get_metadata_builder().kv_cache_spec.block_size
        )
        logger.debug("Using block size %d for drafting layers", self.block_size)

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
        num_tokens_padded = batch_desc.num_tokens

        # Extra coordination when running data-parallel since we need to
        # coordinate across ranks
        # TODO(Flechman): support DBO ubatching
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.vllm_config.parallel_config,
                    allow_microbatching=False,
                    num_tokens_padded=num_tokens_padded,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )
            assert not should_ubatch, "DBO ubatching not implemented for EAGLE"

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct
                # batch_descriptor
                cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # Assert to make sure the agreed upon token count is correct
                # otherwise num_tokens_across_dp will no-longer be valid
                assert batch_desc.num_tokens == num_tokens_padded
                num_tokens_across_dp[dp_rank] = num_tokens_padded

        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp


class EagleProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )


# NOTE(woosuk): Currently, the below code is not used and we always use argmax
# to sample the draft tokens. We will use this after we find a way to manage
# the draft prob tensor.
# Refer to https://github.com/vllm-project/vllm/pull/16899 for the details.
# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    assert sampling_metadata.temperature is not None

    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)
    # consistent with sampler.py's _SAMPLING_EPS threshold
    temperature = sampling_metadata.temperature
    # Avoid division by zero if there are greedy requests.
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temperature = torch.where(is_greedy, 1.0, temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = torch.empty_like(probs)
    q.exponential_()
    # NOTE(woosuk): We shouldn't use `probs.div_(q)` because the draft_probs
    # will be used later for rejection sampling.
    next_token_ids = probs.div(q).argmax(dim=-1).view(-1)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(is_greedy, greedy_token_ids, next_token_ids)
    return next_token_ids, probs


# FR10_MTP_DRAFT_TRACE: final drafter tensor trace for native-vs-tree parity.
import json as _fr10_mtp_trace_json
import os as _fr10_mtp_trace_os
import time as _fr10_mtp_trace_time

_fr10_mtp_trace_orig_propose = EagleProposer.propose
_fr10_mtp_trace_idx = 0

def _fr10_mtp_trace_propose(self, target_token_ids, target_positions,
                            target_hidden_states, next_token_ids,
                            token_indices_to_sample, common_attn_metadata,
                            sampling_metadata, mm_embed_inputs=None,
                            num_rejected_tokens_gpu=None,
                            slot_mappings=None):
    global _fr10_mtp_trace_idx
    out = _fr10_mtp_trace_orig_propose(
        self, target_token_ids, target_positions, target_hidden_states,
        next_token_ids, token_indices_to_sample, common_attn_metadata,
        sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)
    trace_path = _fr10_mtp_trace_os.environ.get("LUMO_MTP_DRAFT_TRACE_FILE")
    if trace_path or _fr10_mtp_trace_os.environ.get("FR10_METRICS", "0") == "1":
        try:
            if not trace_path:
                trace_path = "/logs/fr10_mtp_draft_trace.jsonl"
            global _FR10_MTP_DRAFT_TRACE_FH
            try:
                _FR10_MTP_DRAFT_TRACE_FH
            except NameError:
                _FR10_MTP_DRAFT_TRACE_FH = open(trace_path, "a", buffering=1)
            try:
                from vllm.v1.sample import rejection_sampler as _fr10_rs_mode
                mode = getattr(
                    _fr10_rs_mode,
                    "_FR10_DECODE_MODE",
                    _fr10_mtp_trace_os.environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp"),
                )
            except Exception:
                mode = _fr10_mtp_trace_os.environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp")
            _FR10_MTP_DRAFT_TRACE_FH.write(_fr10_mtp_trace_json.dumps({
                "event": "mtp_draft",
                "idx": int(_fr10_mtp_trace_idx),
                "ts": round(_fr10_mtp_trace_time.time(), 4),
                "mode": str(mode),
                "speculative_token_tree": _fr10_mtp_trace_os.environ.get("SPEC_CONFIG"),
                "shape": [int(x) for x in out.shape],
                "draft": out.detach().cpu().tolist(),
            }) + chr(10))
            _fr10_mtp_trace_idx += 1
        except Exception:
            pass
    return out

EagleProposer.propose = _fr10_mtp_trace_propose
