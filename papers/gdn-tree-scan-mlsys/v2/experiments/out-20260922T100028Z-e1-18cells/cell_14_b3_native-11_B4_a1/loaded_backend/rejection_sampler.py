# LUMO_TREE_PATH_LCP_MAX
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import replace

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplerOutput
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
GREEDY_TEMPERATURE: tl.constexpr = 0
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 128


class RejectionSampler(nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def __init__(self, sampler: Sampler):
        super().__init__()
        self.sampler = sampler
        logprobs_mode = self.sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode.startswith("processed")
        self.is_logits_logprobs_mode = logprobs_mode.endswith("logits")

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        _fr13_sg_bso = globals().pop('_FR13_SG_BONUS_OUT', None)
        if _fr13_sg_bso is not None:
            bonus_sampler_output = _fr13_sg_bso
        else:
            bonus_sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=replace(
                    sampling_metadata,
                    max_num_logprobs=-1,
                ),
                predict_bonus_token=True,
                # Override the logprobs mode to return logits (needed
                # later for accepted token logprobs).
                logprobs_mode_override="processed_logits"
                if self.is_processed_logprobs_mode
                else "raw_logits",
            )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            # Clone raw_target_logits before applying processors to preserve
            # the original raw logits for logprobs computation, since
            # apply_logits_processors modifies the tensor in-place.
            target_logits = target_logits.clone()
        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
        _fr10_target_logits_pre_sampling_constraints = None
        try:
            import os as _fr10_constraint_os
            if (
                _fr10_constraint_os.environ.get("FR10_METRICS", "0") == "1"
                or _fr10_constraint_os.environ.get("LUMO_TREE_SAMPLER_DEBUG_LOG")
            ):
                _fr10_target_logits_pre_sampling_constraints = (
                    target_logits.detach().to(torch.float32).clone()
                )
        except Exception:
            _fr10_target_logits_pre_sampling_constraints = None

        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        lumo_tree_parent_indices = getattr(metadata, "tree_parent_indices", None)
        lumo_tree_token_ids = None
        lumo_tree_self_logits = None
        if lumo_tree_parent_indices is not None:
            tree_self_logits = logits[metadata.tree_self_logits_indices]
            tree_self_logits = tree_self_logits.to(torch.float32)
            if not _FR13_EAGER_PACK:
                if not self.is_processed_logprobs_mode:
                    tree_self_logits = tree_self_logits.clone()
            # FR13_EAGER_PACK 2g(i): under the flag the defensive clone is
            # dropped -- the advanced-index gather above ALWAYS materializes
            # fresh storage (never aliases `logits`), so in-place mutation by
            # processors/constraints cannot touch `logits` with or without
            # the clone; it is a bitwise-identity third copy. (Red-team
            # sign-off item: deletes a stock-mirroring defensive copy.)
            tree_self_logits = self.apply_logits_processors(
                tree_self_logits, sampling_metadata, metadata
            )
            lumo_tree_self_logits = apply_sampling_constraints(
                tree_self_logits,
                metadata.cu_num_draft_tokens,
                sampling_metadata,
            )
            if sampling_metadata.all_greedy:
                if _FR13_EAGER_PACK:
                    # FR13_EAGER_PACK 2g(ii): persistent [2, T] int32 output
                    # buffer; copy_ casts the int64 argmax to int32 in-kernel,
                    # removing the stack/cat + two cast kernels + contiguous +
                    # the per-step allocation. The argmax inputs and the
                    # resulting token values are UNTOUCHED.
                    lumo_tree_token_ids = _fr13_eager_pack_tree_ids_buf(
                        int(target_logits.size(0)), target_logits.device
                    )
                    lumo_tree_token_ids[0].copy_(target_logits.argmax(dim=-1))
                    lumo_tree_token_ids[1].copy_(
                        lumo_tree_self_logits.argmax(dim=-1)
                    )
                else:
                    lumo_tree_token_ids = torch.stack(
                        [
                            target_logits.argmax(dim=-1).to(torch.int32),
                            lumo_tree_self_logits.argmax(dim=-1).to(torch.int32),
                        ],
                        dim=0,
                    ).contiguous()

        try:
            import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
            if (
                _fr10_lo.environ.get("FR10_METRICS", "0") == "1"
                or _fr10_lo.environ.get("LUMO_TREE_SAMPLER_DEBUG_LOG")
            ):
                global _LUMO_TREE_SAMPLER_DEBUG_FH
                try:
                    _LUMO_TREE_SAMPLER_DEBUG_FH
                except NameError:
                    _LUMO_TREE_SAMPLER_DEBUG_FH = open(
                        _fr10_lo.environ.get(
                            "LUMO_TREE_SAMPLER_DEBUG_LOG",
                        ) or "/logs/tree_sampler_debug.jsonl",
                        "a",
                        buffering=1,
                    )
                _LUMO_TREE_SAMPLER_DEBUG_FH.write(
                    _fr10_lj.dumps({
                        "event": "sampler_metadata",
                        "ts": round(_fr10_lt.time(), 4),
                        "has_tree_parent_indices": lumo_tree_parent_indices is not None,
                        "has_tree_self_logits": lumo_tree_self_logits is not None,
                        "all_greedy": bool(sampling_metadata.all_greedy),
                    }) + chr(10)
                )
                if not sampling_metadata.all_greedy:
                    _fr10_draft_ids_cpu = [
                        int(_x) for _x in metadata.draft_token_ids.detach().cpu().tolist()
                    ]
                    _fr10_target_indices_cpu = [
                        int(_x) for _x in metadata.target_logits_indices.detach().cpu().tolist()
                    ]
                    _fr10_target_probs_cpu = (
                        target_logits.softmax(dim=-1, dtype=torch.float32)
                        .detach()
                        .cpu()
                    )
                    _fr10_pre_logits_cpu = (
                        None
                        if _fr10_target_logits_pre_sampling_constraints is None
                        else _fr10_target_logits_pre_sampling_constraints.detach().cpu()
                    )
                    try:
                        _fr10_temperatures_cpu = [
                            float(_x)
                            for _x in sampling_metadata.temperature.detach().cpu().tolist()
                        ]
                    except Exception:
                        _fr10_temperatures_cpu = []
                    if lumo_tree_parent_indices is not None:
                        _fr10_parents_cpu = [
                            int(_x) for _x in lumo_tree_parent_indices.detach().cpu().tolist()
                        ]
                        _fr10_self_indices_cpu = [
                            int(_x)
                            for _x in metadata.tree_self_logits_indices.detach().cpu().tolist()
                        ]
                        _fr10_rows = []
                        _fr10_start = 0
                        for _fr10_req_i, _fr10_node_count in enumerate(metadata.num_draft_tokens):
                            _fr10_node_count = int(_fr10_node_count)
                            for _fr10_node in range(_fr10_node_count):
                                _fr10_flat = _fr10_start + _fr10_node
                                _fr10_tok = int(_fr10_draft_ids_cpu[_fr10_flat])
                                _fr10_prob_row = _fr10_target_probs_cpu[_fr10_flat]
                                _fr10_raw_prob_draft = None
                                _fr10_temp_prob_draft = None
                                if _fr10_pre_logits_cpu is not None:
                                    _fr10_pre_row = _fr10_pre_logits_cpu[_fr10_flat]
                                    _fr10_raw_prob_draft = float(
                                        torch.softmax(
                                            _fr10_pre_row, dim=-1, dtype=torch.float32
                                        )[_fr10_tok].item()
                                    )
                                    _fr10_temp = (
                                        _fr10_temperatures_cpu[_fr10_req_i]
                                        if _fr10_req_i < len(_fr10_temperatures_cpu)
                                        else 1.0
                                    )
                                    if _fr10_temp == 0.0:
                                        _fr10_temp = 1.0
                                    _fr10_temp_prob_draft = float(
                                        torch.softmax(
                                            _fr10_pre_row / float(_fr10_temp),
                                            dim=-1,
                                            dtype=torch.float32,
                                        )[_fr10_tok].item()
                                    )
                                _fr10_rows.append({
                                    "req_index": int(_fr10_req_i),
                                    "node_id": int(_fr10_node),
                                    "parent_node_id": int(_fr10_parents_cpu[_fr10_flat]),
                                    "target_logits_index": int(_fr10_target_indices_cpu[_fr10_flat]),
                                    "self_logits_index": int(_fr10_self_indices_cpu[_fr10_flat]),
                                    "draft_token_id": int(_fr10_tok),
                                    "target_argmax": int(torch.argmax(_fr10_prob_row).item()),
                                    "target_raw_prob_draft": _fr10_raw_prob_draft,
                                    "target_temp_prob_draft": _fr10_temp_prob_draft,
                                    "target_prob_draft": float(_fr10_prob_row[_fr10_tok].item()),
                                })
                            _fr10_start += _fr10_node_count
                        _LUMO_TREE_SAMPLER_DEBUG_FH.write(
                            _fr10_lj.dumps({
                                "event": "tree_logit_gather",
                                "ts": round(_fr10_lt.time(), 4),
                                "rows": _fr10_rows,
                            }) + chr(10)
                        )
                    elif len(_fr10_draft_ids_cpu):
                        _fr10_rows = []
                        _fr10_start = 0
                        for _fr10_req_i, _fr10_node_count in enumerate(metadata.num_draft_tokens):
                            _fr10_node_count = int(_fr10_node_count)
                            for _fr10_pos in range(_fr10_node_count):
                                _fr10_flat = _fr10_start + _fr10_pos
                                _fr10_tok = int(_fr10_draft_ids_cpu[_fr10_flat])
                                _fr10_prob_row = _fr10_target_probs_cpu[_fr10_flat]
                                _fr10_raw_prob_draft = None
                                _fr10_temp_prob_draft = None
                                if _fr10_pre_logits_cpu is not None:
                                    _fr10_pre_row = _fr10_pre_logits_cpu[_fr10_flat]
                                    _fr10_raw_prob_draft = float(
                                        torch.softmax(
                                            _fr10_pre_row, dim=-1, dtype=torch.float32
                                        )[_fr10_tok].item()
                                    )
                                    _fr10_temp = (
                                        _fr10_temperatures_cpu[_fr10_req_i]
                                        if _fr10_req_i < len(_fr10_temperatures_cpu)
                                        else 1.0
                                    )
                                    if _fr10_temp == 0.0:
                                        _fr10_temp = 1.0
                                    _fr10_temp_prob_draft = float(
                                        torch.softmax(
                                            _fr10_pre_row / float(_fr10_temp),
                                            dim=-1,
                                            dtype=torch.float32,
                                        )[_fr10_tok].item()
                                    )
                                _fr10_rows.append({
                                    "req_index": int(_fr10_req_i),
                                    "position": int(_fr10_pos),
                                    "target_logits_index": int(_fr10_target_indices_cpu[_fr10_flat]),
                                    "draft_logits_index": int(_fr10_target_indices_cpu[_fr10_flat] + 1),
                                    "draft_token_id": int(_fr10_tok),
                                    "target_argmax": int(torch.argmax(_fr10_prob_row).item()),
                                    "target_raw_prob_draft": _fr10_raw_prob_draft,
                                    "target_temp_prob_draft": _fr10_temp_prob_draft,
                                    "target_prob_draft": float(_fr10_prob_row[_fr10_tok].item()),
                                })
                            _fr10_start += _fr10_node_count
                        _LUMO_TREE_SAMPLER_DEBUG_FH.write(
                            _fr10_lj.dumps({
                                "event": "linear_logit_gather",
                                "ts": round(_fr10_lt.time(), 4),
                                "rows": _fr10_rows,
                            }) + chr(10)
                        )
                    _fr10_capture_path = _fr10_lo.environ.get("FR10_SPINE_LOGIT_CAPTURE")
                    _fr10_capture_has_spec = any(
                        int(_x) > 0 for _x in metadata.num_draft_tokens
                    )
                    if (
                        _fr10_capture_path
                        and _fr10_capture_has_spec
                    ):
                        _fr10_capture_seen = int(
                            globals().get("_FR10_SPINE_LOGIT_CAPTURE_SEEN", 0)
                        )
                        _fr10_capture_skip = int(
                            _fr10_lo.environ.get("FR10_SPINE_LOGIT_CAPTURE_SKIP", "0")
                        )
                        _fr10_capture_limit = int(
                            _fr10_lo.environ.get("FR10_SPINE_LOGIT_CAPTURE_LIMIT", "1")
                        )
                        _fr10_capture_saved = int(
                            globals().get("_FR10_SPINE_LOGIT_CAPTURE_SAVED", 0)
                        )
                        globals()["_FR10_SPINE_LOGIT_CAPTURE_SEEN"] = (
                            _fr10_capture_seen + 1
                        )
                        if (
                            _fr10_capture_seen >= _fr10_capture_skip
                            and _fr10_capture_saved < _fr10_capture_limit
                        ):
                            try:
                                from pathlib import Path as _fr10_Path
                                _fr10_cap = {
                                    "schema": "fr10.spine_logit_capture.v1",
                                    "capture_call_index": int(_fr10_capture_seen),
                                    "capture_saved_index": int(_fr10_capture_saved),
                                    "mode": str(
                                        getattr(
                                            __import__("vllm.v1.sample.rejection_sampler", fromlist=["_FR10_DECODE_MODE"]),
                                            "_FR10_DECODE_MODE",
                                            _fr10_lo.environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp"),
                                        )
                                    ),
                                    "has_tree_parent_indices": lumo_tree_parent_indices is not None,
                                    "num_draft_tokens": [
                                        int(_x) for _x in metadata.num_draft_tokens
                                    ],
                                    "draft_token_ids": metadata.draft_token_ids.detach().cpu(),
                                    "target_logits_indices": metadata.target_logits_indices.detach().cpu(),
                                    "target_logits": target_logits.detach().to(torch.float32).cpu(),
                                }
                                if lumo_tree_parent_indices is not None:
                                    _fr10_cap["tree_parent_indices"] = (
                                        lumo_tree_parent_indices.detach().cpu()
                                    )
                                    _fr10_cap["tree_self_logits_indices"] = (
                                        metadata.tree_self_logits_indices.detach().cpu()
                                    )
                                    _fr10_cap["tree_self_logits"] = (
                                        lumo_tree_self_logits.detach().to(torch.float32).cpu()
                                        if lumo_tree_self_logits is not None else None
                                    )
                                _fr10_out = _fr10_Path(_fr10_capture_path)
                                _fr10_out.parent.mkdir(parents=True, exist_ok=True)
                                _fr10_call_out = _fr10_out.with_name(
                                    _fr10_out.stem
                                    + ".call"
                                    + str(int(_fr10_capture_saved))
                                    + _fr10_out.suffix
                                )
                                torch.save(_fr10_cap, _fr10_call_out)
                                if _fr10_capture_saved == 0:
                                    torch.save(_fr10_cap, _fr10_out)
                                globals()["_FR10_SPINE_LOGIT_CAPTURE_SAVED"] = (
                                    _fr10_capture_saved + 1
                                )
                                if _fr10_capture_saved + 1 >= _fr10_capture_limit:
                                    globals()["_FR10_SPINE_LOGIT_CAPTURED"] = True
                            except Exception as _fr10_capture_exc:
                                raise RuntimeError(
                                    "FR10 spine logit capture failed: "
                                    + type(_fr10_capture_exc).__name__
                                    + ":"
                                    + str(_fr10_capture_exc)
                                ) from _fr10_capture_exc
        except RuntimeError as _fr10_debug_exc:
            if str(_fr10_debug_exc).startswith("FR10 spine logit capture failed:"):
                raise
        except Exception:
            pass

        if (
            globals().get(
                "_FR10_DECODE_MODE",
                __import__("os").environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp"),
            )
            == "tree_mtp"
            and hasattr(metadata, "tree_parent_indices")
            and lumo_tree_parent_indices is None
            and __import__("os").environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1"
        ):
            raise RuntimeError(
                "FR10 sampled tree committer disengaged: missing_tree_parent_indices"
            )

        # FR13_COMMIT_ARGMAX_GATE (default OFF; DIAGNOSTIC ONLY, like
        # FR13_FIX1_SELFCHECK / FR13_FORCE_SPINE_COMMIT): publish the verify
        # forward's RAW logit tensors so the greedy committer can capture, at
        # each served row, whether committed_token_id == argmax(verify_logits
        # [the row the committer actually indexed]). Channel-1 = committer
        # row-mapping; channel-2 split records the verify argmax + top-2 margin
        # for the reduce's clean-forward comparison. OFF = nothing published
        # (the committer's gate reads the global as None and no-ops). EAGER-only
        # (the committer syncs/.item()s); NEVER bind =1 into a serving/speed
        # boot. The two tensors are the EXACT post-constraint logits the argmax
        # token-ids were computed from (target_logits = parent-edge dist;
        # lumo_tree_self_logits = self/node dist) -- same rows, same values.
        # FR13_FORK_MARGIN_DUMP reuses this SAME read-only publish: the
        # committer-fork classifier needs the verify target_logits rows to read
        # the deciding parent_target's top-2 margin. Publish when EITHER
        # diagnostic flag is armed; OFF for both => globals stay None => the
        # committer's gates no-op (byte-identical served path, bug-class #10).
        import os as _fr13_cag_os
        if (
            _fr13_cag_os.environ.get("FR13_COMMIT_ARGMAX_GATE", "0") == "1"
            or _fr13_cag_os.environ.get("FR13_FORK_MARGIN_DUMP", "0") == "1"
        ):
            globals()["_FR13_CAG_TARGET_LOGITS"] = target_logits
            globals()["_FR13_CAG_SELF_LOGITS"] = lumo_tree_self_logits
            globals()["_FR13_CAG_NUM_DRAFT"] = [
                int(_x) for _x in metadata.num_draft_tokens
            ]
            globals()["_FR13_CAG_STEP"] = int(
                globals().get("_FR13_CAG_STEP", -1)
            ) + 1
        else:
            globals()["_FR13_CAG_TARGET_LOGITS"] = None
            globals()["_FR13_CAG_SELF_LOGITS"] = None

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            tree_parent_indices=lumo_tree_parent_indices,
            tree_token_ids=lumo_tree_token_ids,
            tree_self_logits=lumo_tree_self_logits,
        )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        _fr13_sg_capchk("fwd-tail")
        _fr13_sg_so = SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )
        _fr13_sg_capchk("post-so-build")
        return _fr13_sg_so

    def _get_logprobs_tensors(
        self,
        max_num_logprobs: int,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
        cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]

        # Collect target and bonus logits.
        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices
        final_logits = torch.zeros_like(logits, dtype=torch.float32)
        final_logits[target_logits_indices] = target_logits.to(torch.float32)
        final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)

        # NOTE: To avoid cpu-gpu synchronization, we now simply compute indices for
        # all draft tokens, including the rejected ones. The rejected tokens will
        # be filtered out in the `parse_output`.
        logit_start_indices = cu_num_sampled_tokens
        offsets = torch.arange(
            sampled_token_ids.shape[-1],
            device=logit_start_indices.device,
            dtype=logit_start_indices.dtype,
        )
        accepted_logit_indices = (
            logit_start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        ).flatten()
        accepted_logit_indices.clamp_(max=final_logits.shape[0] - 1)
        accepted_tokens = sampled_token_ids.clone().flatten()
        # we replace rejected token ids with 0 to avoid gather_logprobs error
        accepted_tokens[accepted_tokens == PLACEHOLDER_TOKEN_ID] = 0

        # Compute logprobs for accepted tokens.
        accepted_logits = final_logits[accepted_logit_indices]
        accepted_logprobs = (
            accepted_logits
            if self.is_logits_logprobs_mode
            else self.sampler.compute_logprobs(accepted_logits)
        )
        return self.sampler.gather_logprobs(
            accepted_logprobs,
            max_num_logprobs,
            accepted_tokens.to(torch.int64),
        )

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        """Parse the output of the rejection sampler.
        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID` by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.
            discard_req_indices: Optional row indices to discard tokens in.
            logprobs_tensors: Optional logprobs tensors to filter.
        Returns:
            A list of lists of token IDs.
        """
        output_token_ids_np = output_token_ids.cpu().numpy()
        # Create mask for valid tokens.
        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        output_logprobs = None
        if logprobs_tensors is not None:
            cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
            output_logprobs = filtered_tensors.tolists(cu_num_tokens)

        if len(discard_req_indices) > 0:
            valid_mask[discard_req_indices] = False
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs, output_logprobs

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        # FR13_SG_TARGET_LOGITS (S1 =2): consume-in-order handoff — the
        # =2 wrapper ran this method eagerly OUTSIDE the capture into
        # per-key statics for BOTH call sites (target, then tree_self —
        # forward's call order). Penalties build per-step host tensors
        # that cannot run inside a graph (in-graph statics-fed penalty
        # compute is the follow-up refinement). Unset => stock path.
        _fr13_sg_tlq = globals().get('_FR13_SG_TL_QUEUE')
        if _fr13_sg_tlq:
            return _fr13_sg_tlq.pop(0)
        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
            sampling_metadata.bad_words_token_ids or has_penalties
        )

        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Calculate indices of target logits.
        if sampling_metadata.allowed_token_ids_mask is not None or has_penalties:
            num_requests = len(metadata.num_draft_tokens)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            logits = self.apply_penalties(
                logits, sampling_metadata, metadata, repeat_indices, output_token_ids
            )

            # Apply allowed token ids.
            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(processor, MinTokensLogitsProcessor):
                logits = processor.apply_with_spec_decode(
                    logits, metadata.num_draft_tokens
                )

        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None

        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]
        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]
        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]

        logits = apply_all_penalties(
            logits,
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        return logits

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if spec_token_ids is None:
            return output_token_ids

        result = []
        for out, spec in zip(output_token_ids, spec_token_ids):
            if len(spec) == 0:
                continue
            result.append(out)
            for i in range(len(spec) - 1):
                result.append([*result[-1], spec[i]])
        return result

_FR13_FIXED32_MODE = ''
_FR13_FIXED32_VALID_MASK = 0
_FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS = False

def _fr13_fixed32_compact_sampler_row_plan(
    num_draft_tokens,
    batch_rows,
    spec_batch_indices,
):
    """Validate the full sampler rows and return their compact spec map."""
    if (
        not isinstance(num_draft_tokens, (list, tuple))
        or not 1 <= len(num_draft_tokens) <= 4
        or type(batch_rows) is not int
        or batch_rows != len(num_draft_tokens)
        or not isinstance(spec_batch_indices, tuple)
    ):
        raise RuntimeError(
            "FR13 fixed32 full/compact sampler row contract is missing"
        )
    full_counts = tuple(num_draft_tokens)
    if any(
        type(value) is not int or value not in (0, 31)
        for value in full_counts
    ):
        raise RuntimeError(
            "FR13 fixed32 sampler rows must be zero or physical-31; "
            "truncated positive drafts are forbidden"
        )
    expected_indices = tuple(
        index for index, value in enumerate(full_counts) if value == 31
    )
    if (
        not 1 <= len(spec_batch_indices) <= 4
        or any(type(index) is not int for index in spec_batch_indices)
        or tuple(sorted(set(spec_batch_indices))) != spec_batch_indices
        or any(not 0 <= index < batch_rows for index in spec_batch_indices)
        or spec_batch_indices != expected_indices
    ):
        raise RuntimeError(
            "FR13 fixed32 published compact spec-row indices drifted"
        )
    return full_counts, spec_batch_indices


def _fr13_fixed32_live_sampler_row_plan(num_draft_tokens):
    from vllm.model_executor.layers.mamba import (
        gdn_linear_attn as _fr13_f32_plan_gdn,
    )

    return _fr13_fixed32_compact_sampler_row_plan(
        num_draft_tokens,
        getattr(_fr13_f32_plan_gdn, "_FR13_FIXED32_BATCH_ROWS", None),
        getattr(
            _fr13_f32_plan_gdn,
            "_FR13_FIXED32_SPEC_BATCH_INDICES",
            None,
        ),
    )


def _fr13_fixed32_prepare_mixed_sampler_io(
    output_token_ids,
    accepted_tree_rows,
    bonus_token_ids,
    spec_batch_indices,
    generators,
):
    """Build compact-spec destinations while retaining full sampler output."""
    full_batch = int(output_token_ids.shape[0])
    compact_batch = len(spec_batch_indices)
    if (
        output_token_ids.ndim != 2
        or tuple(output_token_ids.shape) != (full_batch, 32)
        or accepted_tree_rows.ndim != 1
        or tuple(accepted_tree_rows.shape) != (full_batch,)
        or bonus_token_ids.ndim < 1
        or int(bonus_token_ids.shape[0]) != full_batch
        or int(bonus_token_ids.numel()) != full_batch
        or not 1 <= compact_batch < full_batch <= 4
        or output_token_ids.device != accepted_tree_rows.device
        or output_token_ids.device != bonus_token_ids.device
    ):
        raise RuntimeError("FR13 fixed32 mixed sampler I/O geometry drift")
    index_tensor = torch.tensor(
        spec_batch_indices,
        dtype=torch.long,
        device=output_token_ids.device,
    )
    compact_output = torch.empty(
        (compact_batch, 32),
        dtype=output_token_ids.dtype,
        device=output_token_ids.device,
    )
    compact_accepted = torch.empty(
        (compact_batch,),
        dtype=accepted_tree_rows.dtype,
        device=accepted_tree_rows.device,
    )
    compact_bonus = bonus_token_ids.index_select(0, index_tensor).contiguous()
    compact_generators = None
    if generators:
        if (
            not hasattr(generators, "get")
            or not hasattr(generators, "keys")
        ):
            raise RuntimeError(
                "FR13 fixed32 mixed sampler generators have no index map"
            )
        generator_keys = tuple(generators.keys())
        if any(
            type(key) is not int or not 0 <= key < full_batch
            for key in generator_keys
        ):
            raise RuntimeError(
                "FR13 fixed32 mixed sampler generator key is out of range"
            )
        compact_generators = {}
        for compact_index, full_index in enumerate(spec_batch_indices):
            generator = generators.get(full_index)
            if generator is not None:
                compact_generators[compact_index] = generator
        if not compact_generators:
            compact_generators = None
    return (
        index_tensor,
        compact_output,
        compact_accepted,
        compact_bonus,
        compact_generators,
    )


def _fr13_fixed32_finish_mixed_sampler_io(
    output_token_ids,
    accepted_tree_rows,
    bonus_token_ids,
    index_tensor,
    compact_output,
    compact_accepted,
):
    """Restore compact spec results and bonus-only zero rows to full order."""
    output_token_ids.fill_(-1)
    output_token_ids[:, 0].copy_(bonus_token_ids.reshape(-1))
    output_token_ids.index_copy_(0, index_tensor, compact_output)
    accepted_tree_rows.zero_()
    accepted_tree_rows.index_copy_(0, index_tensor, compact_accepted)
    return output_token_ids


# FR13_EAGER_PACK (FIX-2): ONE flag, default OFF; OFF = exact legacy
# committer/sampler transports (the A/B instrument, same pattern as
# FR13_DRAFTER_SINGLE_LOGITS). ON changes NO computed value — only HOW the
# same ints move: one packed DtoH for all six committer inputs PLUS the
# stacked 48x2 replay-flag matrix (replaces ~102 tiny blocking readbacks),
# pinned staged HtoD for the committer outputs, and ONE batched all-layer
# replay launch (replaces 48 eager launches + 48 flag clears). Env is read
# once at module import (flag plan).
_FR13_EAGER_PACK = True
_FR13_EAGER_PACK_NEEDLE_DONE = False
_FR13_EAGER_PACK_STAGE = {}

# FR13_COMMITTER_SYNCKILL (OPT-1 G2, ONE flag, default OFF, gated UNDER
# FR13_GPU_COMMITTER). The GPU committer's first draft still blocks the MAIN
# launching thread: the EAGER_PACK packed committer-input DtoH+sync (:6761) and
# the kernel's output .cpu() readback both gate the main thread, so the flag-ON
# path was SLOWER than OFF (the decision is on-GPU + lossless but the run-ahead
# is not restored). G2 keeps the committer inputs DEVICE-resident (skips :6761),
# runs the decision kernel on device tensors, writes the SAME-STEP serving
# outputs (output_token_ids / accepted_tree_rows / GDN path/len) DEVICE->DEVICE
# from the kernel's device outputs, and moves the host materialisation onto a
# side stream + CUDA event (native AsyncGPUModelRunnerOutput shape) so the main
# thread launches the next forward WITHOUT blocking. PURE-INTEGER, location-only
# -> byte-identical to the host committer for every input; OFF == the current
# :6761 + first-draft-kernel/legacy path verbatim. NEVER bind into a config
# without FR13_GPU_COMMITTER=1 (synckill is meaningless without the GPU
# committer). Read once at import (flag plan).

# FR13_COMMIT_ARGMAX_GATE (ONE flag, default OFF, inert): per-served-token
# in-process committer-row argmax gate. At each committed/served token the
# greedy committer records, to a jsonl, the row it ACTUALLY indexed into the
# verify-forward logits and whether the served token id == argmax of THAT row
# (channel-1: committer row-mapping). It also records the verify-forward argmax
# id + its top-2 margin so the reduce can compare against a clean teacher-forced
# single-forward reference (channel-2: verify-forward losslessness). OFF =
# nothing happens (legacy-verbatim: the committer never touches the published
# logit globals, which the OFF call-site leaves None). EAGER-only: the tap
# syncs + .item()s the logit rows, so it must run on an eager diagnostic boot
# (fail-loud if it ever runs during CUDA-graph capture). NEVER bind =1 into a
# serving/speed config (same class as FR13_FORCE_SPINE_COMMIT). Read once at
# import (flag plan); the call-site publishes the raw logit tensors only when
# armed.
_FR13_COMMIT_ARGMAX_GATE = (
    __import__('os').environ.get('FR13_COMMIT_ARGMAX_GATE', '0') == '1'
    or __import__('os').path.exists('/logs/fr13_commit_argmax_gate.arm')
)
_FR13_COMMIT_ARGMAX_GATE_NEEDLE_DONE = False
_FR13_COMMIT_ARGMAX_GATE_FH = None
_FR13_COMMIT_ARGMAX_GATE_STATS = {
    'served_tokens': 0,
    'ch1_mismatch': 0,
    'spine_path_idx_set': 0,
}

# FR13_FORK_MARGIN_DUMP (ONE flag, default OFF, READ-ONLY) — per-spec-step
# committer-fork classifier sink. The LCP-committer
# (_lumo_tree_path_lcp_max_greedy_sample) scores every root-to-leaf path's lcp
# (= longest prefix where draft==parent_target) and commits the max-lcp path's
# prefix+bonus. A FORK is a step where the WINNER path is NOT the spine (a leaf
# wins the lcp tie-break, or the lcp boundary shifts under co-residency). To
# decide whether a fork is (A) a GENUINE leaf-LCP win (the deciding
# parent_target was CONFIDENT, verify top1-top2 > 1.0 nat -> margin-damp would
# lose a real accept) or (B) a SUB-1-NAT NEAR-TIE (verify nearly indifferent ->
# a deterministic rank-2 "don't let a leaf win the lcp boundary on a sub-1-nat
# parent_target" rule would stop the fork while genuine (A) wins keep serving),
# we dump, per request per step: each path's lcp + nodes; the WINNER and SPINE
# lcp-boundary (divergence) nodes + their VERIFY top-2 logprobs (parent_target
# margin = top1-top2); the parent_target id, spine-token id, draft id at the
# divergence node; the chosen best_path/best_leaf + committed row. The dump
# CHANGES NOTHING SERVED (the committed `row` is computed exactly as OFF; this
# block only reads target_logits rows and writes a jsonl) => default-OFF is
# byte-identical to the locked path (bug-class #10: OFF call-site never
# publishes the logit globals, so an OFF boot can never arm a captured run).
# EAGER-only (it syncs/.item()s the logit rows; fail-loud during CUDA-graph
# capture via the shared row_stats guard). NEVER bind =1 into a serving/speed
# config (same class as FR13_FORCE_SPINE_COMMIT / FR13_COMMIT_ARGMAX_GATE).
_FR13_FORK_MARGIN_DUMP = (
    __import__('os').environ.get('FR13_FORK_MARGIN_DUMP', '0') == '1'
)
_FR13_FORK_MARGIN_DUMP_NEEDLE_DONE = False
_FR13_FORK_MARGIN_DUMP_FH = None
_FR13_FORK_MARGIN_DUMP_STATS = {
    'steps_seen': 0,
    'forks_seen': 0,
}


def _fr13_fork_margin_dump_needle(armed):
    """Engagement needle (class 9): fires once, in BOTH flag states."""
    global _FR13_FORK_MARGIN_DUMP_NEEDLE_DONE
    if _FR13_FORK_MARGIN_DUMP_NEEDLE_DONE:
        return
    _FR13_FORK_MARGIN_DUMP_NEEDLE_DONE = True
    msg = (
        'FR13_FORK_MARGIN_DUMP committer-fork classifier: armed=%d (%s)' % (
            1 if armed else 0,
            'armed-eager-only-readonly' if armed else 'inert',
        )
    )
    try:
        from vllm.logger import init_logger as _fmd_init_logger
        _fmd_init_logger('vllm.fr13_fork_margin_dump').info(msg)
    except Exception:
        print(msg, flush=True)


def _fr13_fork_margin_dump_fh():
    """Append-mode jsonl handle, opened once (class 12 raw-counter sink)."""
    global _FR13_FORK_MARGIN_DUMP_FH
    if _FR13_FORK_MARGIN_DUMP_FH is None:
        import os as _fmd_os
        _fmd_path = _fmd_os.environ.get(
            'FR13_FORK_MARGIN_DUMP_PATH',
            '/logs/fr13_fork_margin_dump.jsonl',
        )
        try:
            _fmd_dir = _fmd_os.path.dirname(_fmd_path)
            if _fmd_dir:
                _fmd_os.makedirs(_fmd_dir, exist_ok=True)
        except Exception:
            pass
        _FR13_FORK_MARGIN_DUMP_FH = open(_fmd_path, 'a', buffering=1)
    return _FR13_FORK_MARGIN_DUMP_FH


def _fr13_commit_argmax_gate_needle(armed):
    """Engagement needle (class 9): fires once, in BOTH flag states."""
    global _FR13_COMMIT_ARGMAX_GATE_NEEDLE_DONE
    if _FR13_COMMIT_ARGMAX_GATE_NEEDLE_DONE:
        return
    _FR13_COMMIT_ARGMAX_GATE_NEEDLE_DONE = True
    msg = (
        'FR13_COMMIT_ARGMAX_GATE committer-row gate: armed=%d (%s)' % (
            1 if armed else 0,
            'armed-eager-only' if armed else 'inert',
        )
    )
    try:
        from vllm.logger import init_logger as _cag_init_logger
        _cag_init_logger('vllm.fr13_commit_argmax_gate').info(msg)
    except Exception:
        print(msg, flush=True)


def _fr13_commit_argmax_gate_fh():
    """Append-mode jsonl handle, opened once (class 12 raw-counter sink)."""
    global _FR13_COMMIT_ARGMAX_GATE_FH
    if _FR13_COMMIT_ARGMAX_GATE_FH is None:
        import os as _cag_os
        _cag_path = _cag_os.environ.get(
            'FR13_COMMIT_ARGMAX_GATE_DUMP',
            '/logs/fr13_commit_argmax_gate.jsonl',
        )
        try:
            _cag_dir = _cag_os.path.dirname(_cag_path)
            if _cag_dir:
                _cag_os.makedirs(_cag_dir, exist_ok=True)
        except Exception:
            pass
        _FR13_COMMIT_ARGMAX_GATE_FH = open(_cag_path, 'a', buffering=1)
    return _FR13_COMMIT_ARGMAX_GATE_FH


def _fr13_commit_argmax_gate_row_stats(logit_row):
    """Top-2 over one verify-forward logit row (eager .item() syncs).

    Returns (argmax_id, argmax_logit, runner_up_id, runner_up_logit). The
    top-2 margin (argmax_logit - runner_up_logit) is the channel-2 margin the
    reduce uses to decide whether a verify-vs-clean argmax flip is a clear
    deviation or a near-tie. NEVER call during CUDA-graph capture.
    """
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            'FR13_COMMIT_ARGMAX_GATE is eager-only: row stats called during '
            'CUDA-graph capture (class 6 fail-loud)'
        )
    _top = torch.topk(logit_row.to(torch.float32), 2, dim=-1)
    _vals = _top.values.tolist()
    _ids = _top.indices.tolist()
    return (
        int(_ids[0]),
        float(_vals[0]),
        int(_ids[1]) if len(_ids) > 1 else -1,
        float(_vals[1]) if len(_vals) > 1 else float('-inf'),
    )


def _fr13_eager_pack_stage(kind, elems, device, dtype):
    """Persistent staging (device tensor, pinned CPU mirror, event, rec-flag).

    Allocated once; REALLOCATED only when capacity grows (amortized
    init-time, class 6 — the committer is eager host code, never CUDA-graph
    captured, and B=1 steady state hits a fixed size). The CUDA event guards
    pinned-buffer reuse: recorded after the transport that last touches the
    pinned buffer on-stream and synchronized before the next host write, so
    a host rewrite can never race an in-flight async copy.
    """
    entry = _FR13_EAGER_PACK_STAGE.get(kind)
    if (
        entry is None
        or entry[0].numel() < elems
        or entry[0].device != device
        or entry[0].dtype != dtype
    ):
        cap = max(int(elems), 256)
        entry = (
            torch.empty((cap,), dtype=dtype, device=device),
            torch.empty((cap,), dtype=dtype, pin_memory=True),
            torch.cuda.Event(),
            [False],
        )
        _FR13_EAGER_PACK_STAGE[kind] = entry
    return entry


def _fr13_eager_pack_tree_ids_buf(rows, device):
    """Persistent contiguous [2, rows] int32 tree-token buffer (FIX-2 2g).

    Exact-size keyed cache (allocated once per (rows, device); B=1 steady
    state is a single entry), so the ON path has NO per-step allocation and
    the tensor layout is identical to the legacy stack().contiguous().
    """
    key = ('tree_ids', int(rows), str(device))
    buf = _FR13_EAGER_PACK_STAGE.get(key)
    if buf is None:
        buf = torch.empty((2, int(rows)), dtype=torch.int32, device=device)
        _FR13_EAGER_PACK_STAGE[key] = buf
    return buf


def _fr13_eager_pack_needle(pack_on, is_cuda, layers, packed_elems,
                            replay_batched, stacked_rings,
                            boundary_legacy_loop):
    """Engagement needle (class 9): fires once, in BOTH flag states."""
    global _FR13_EAGER_PACK_NEEDLE_DONE
    if _FR13_EAGER_PACK_NEEDLE_DONE:
        return
    _FR13_EAGER_PACK_NEEDLE_DONE = True
    msg = (
        'FR13_EAGER_PACK committer path engaged: pack=%d cuda=%d layers=%d '
        'packed_dtoh_elems=%d replay_batched=%d stacked_rings=%d '
        'boundary_legacy_loop=%d' % (
            1 if pack_on else 0,
            1 if is_cuda else 0,
            int(layers),
            int(packed_elems),
            1 if replay_batched else 0,
            1 if stacked_rings else 0,
            1 if boundary_legacy_loop else 0,
        )
    )
    try:
        from vllm.logger import init_logger as _ep_init_logger
        _ep_init_logger('vllm.fr13_eager_pack').info(msg)
    except Exception:
        print(msg, flush=True)


# FR13_REPLAY_BOUNDARY tap A (PRODUCER): immediately around the committer's
# _fr13_replay_launch for the probed layer(s), snapshot the replay's inputs
# (scan-time prev-lens/window snapshot, h0 source row bytes PRE-launch) and the
# as-written dst bank rows (linear columns 0..max(len-1,0)) POST-launch.
# Publishes {req_id: rows written} in the gdn module for tap C's stale-read
# discriminator. Eager-only diagnostics (syncs + .item()); the shared emit()
# fail-louds if ever called during CUDA-graph capture.
def _fr13_boundary_replay_pre(gdn_mod, layer, bank, rows):
    torch.cuda.synchronize()
    pre = []
    spec_idx = layer._fr13_replay_spec_idx
    prev_lens = layer._fr13_replay_prev_lens
    for _b in range(rows):
        _prev_len = int(prev_lens[_b].detach().cpu().item())
        _h0_col = max(_prev_len - 1, 0)
        _window = [
            int(_x) for _x in spec_idx[_b].detach().cpu().tolist()
        ]
        _h0_row = int(_window[min(_h0_col, len(_window) - 1)])
        _first8, _sha, _nb = gdn_mod._fr13_boundary_row_digest(bank[_h0_row])
        pre.append({
            'prev_len_snapshot': _prev_len,
            'h0_col': _h0_col,
            'h0_src_row': _h0_row,
            'h0_first8_prelaunch': _first8,
            'h0_sha4096_prelaunch': _sha,
            'window_written': _window,
            'window_digest_pre': gdn_mod._fr13_boundary_window_digest(
                bank, _window
            ),
        })
    return pre


def _fr13_boundary_replay_post(
    gdn_mod, prefix, layer, bank, rows,
    accepted_node_paths, accepted_lens_list, pre,
):
    torch.cuda.synchronize()
    # SPEC-row req_ids (aligned with the spec rows this producer iterates), NOT the
    # full sampler batch: else _last_written is keyed to the wrong req on mixed
    # prefill+decode batches and the Tap C stale_read verdict becomes a FALSE
    # positive (it compares src_row against a different req's written rows). Same
    # bug class as the leaf-publish fix.
    _row_req_ids = getattr(gdn_mod, '_LUMO_FA_SPEC_ROW_REQ_IDS', None) or []
    _last_written = getattr(gdn_mod, '_FR13_BOUNDARY_LAST_WRITTEN_BY_REQ', None)
    if _last_written is None:
        _last_written = {}
        gdn_mod._FR13_BOUNDARY_LAST_WRITTEN_BY_REQ = _last_written
    for _b in range(rows):
        _path = [int(_x) for _x in accepted_node_paths[_b]]
        _alen = int(accepted_lens_list[_b])
        _info = pre[_b] if pre is not None and _b < len(pre) else {}
        _window = _info.get('window_written') or [
            int(_x)
            for _x in layer._fr13_replay_spec_idx[_b].detach().cpu().tolist()
        ]
        _dst = []
        _rows_written = []
        for _c in range(max(_alen, 1)):
            _row_id = int(_window[_c])
            _first8, _sha, _nb = gdn_mod._fr13_boundary_row_digest(
                bank[_row_id]
            )
            _dst.append({
                'col': int(_c),
                'row': _row_id,
                'first8': _first8,
                'sha4096': _sha,
            })
            _rows_written.append(_row_id)
        _req_id = (
            str(_row_req_ids[_b]) if _b < len(_row_req_ids) else None
        )
        if _req_id is not None:
            _last_written[_req_id] = {
                'event': int(getattr(gdn_mod, '_FR13_BOUNDARY_EVENT', 0)),
                # FR13_CHASE_DIAG instrument (ii): keep the published PATH and
                # the ROWBUG class (H1: published leaf flat row != accepted
                # len => the stock linear sample-row math picks a wrong row)
                # with the as-written digests so the next event's B_JOIN
                # verdict can label every transition in-process.
                'accepted_path': [int(_x) for _x in _path],
                'rowbug': bool(_path) and int(_path[-1]) != _alen,
                # 'rows' is read by the PRE-EXISTING tap-C consumer
                # (lastw.get("rows")) for its stale-read verdict; the chase
                # edit that added accepted_path/rowbug dropped it, making
                # tap-C stale=always-True (wf_a71e2a24 FAIL-2). Keep it.
                'rows': [int(_x) for _x in _rows_written],
                'by_col': {
                    str(_d['col']): _d['row'] for _d in _dst
                },
                'sha_by_row': {
                    str(_d['row']): _d['sha4096'] for _d in _dst
                },
                'accepted_len': _alen,
                'window_written': _window,
            }
        gdn_mod._fr13_boundary_emit({
            'tap': 'A',
            'layer': str(prefix),
            'slot': int(_b),
            'req_id': _req_id,
            'accepted_path': _path,
            'accepted_len': _alen,
            'prev_len_snapshot': _info.get('prev_len_snapshot'),
            'h0_col': _info.get('h0_col'),
            'h0_src_row': _info.get('h0_src_row'),
            'h0_first8_prelaunch': _info.get('h0_first8_prelaunch'),
            'h0_sha4096_prelaunch': _info.get('h0_sha4096_prelaunch'),
            'window_written': _window,
            'dst_rows': _dst,
            'num_replay_rows': int(rows),
            'window_digest_pre': _info.get('window_digest_pre'),
            'window_digest_post': gdn_mod._fr13_boundary_window_digest(
                bank, _window
            ),
        })


# =========================================================================== #
# FR13_REPLAY_DURABLE_AB (PRIME lead, FR13_TOTAL_DRIFT_REANALYSIS_LEADS_BIND):
# OBSERVE-ONLY per-event durable-state A/B of OUR replay kernel's published GDN
# recurrent state (H_ours) vs NATIVE MTP's sequential recurrent kernel
# (fused_sigmoid_gating_delta_rule_update) run over the SAME accepted token
# chain from the SAME cloned h0/conv-state.  Default OFF -> byte-identical to
# the locked cat9 build.  Served stream NEVER touched: every native arm runs on
# a detached CLONE of h0 + a flat gather of the activation rings, with
# inplace_final_state=False so native writes its durable state into a FRESH
# tensor (the served ssm_state bank is read-only here).  NOT a reroute: the
# VERIFY scan and the served replay launch are unchanged; this only ADDS an
# observe-only native run on clones and records max_abs(H_ours - H_native_seq).
# Bug classes: #9 (silent/vacuous -> loud stage markers + record-count assert),
# #10 (codegen-identity-not-spec -> the very thing being measured), #12
# (cross-event accumulation -> per-event index for back-loading).
# =========================================================================== #
_FR13_RDAB_FLAG = None
_FR13_RDAB_FH = None
_FR13_RDAB_HEADER_DONE = False
_FR13_RDAB_STAGE_SEEN = set()
_FR13_RDAB_STAGE_COUNTS = {}
_FR13_RDAB_RECORDS = 0




def _fr13_conv_commit_to_col0(
    replay_layers,
    accepted_path_buf,
    accepted_lens_buf,
    replay_rows,
    do_commit,
):
    """STATELESS-TREE conv committer (post-accept, host-eager; conv has no kernel).

    Copies THIS step's committed-leaf conv window into col 0 (the req running row)
    for every committed (row, layer). Leaf node id = accepted_paths[b, acc_len-1]
    (post-refill = THIS step's accept); dst col0 == spec_state_indices[b,0] ==
    block_table[b,0] == the SSM RUNROW_COMMIT target, so conv + SSM commit to /
    init from the SAME running row. Byte-neutral on no-cache decode (col0 receives
    the same window the current accepted-leaf-node read carries).
    BURN DELETED 2026-07-27 (FR13_APC_BURN_NODE_BANK): the spec-col burn was proven
    redundant for the served path — every served reader is col-0 under
    RUNROW_INIT=1 and spec cols are regenerated fresh next forward. The legacy
    runrow=0 path (where burn was load-bearing) is retired; callers fail loud.
    """
    if replay_rows <= 0 or not do_commit:
        return
    import torch
    _layer_items = (
        enumerate(replay_layers)
        if isinstance(replay_layers, tuple)
        else (
            (_prefix, replay_layers[_prefix])
            for _prefix in sorted(replay_layers)
        )
    )
    for _prefix, _layer in _layer_items:
        _conv = getattr(_layer, "_fr13_replay_conv_state", None)
        _ssi = getattr(_layer, "_fr13_replay_spec_idx", None)
        if _conv is None or _ssi is None:
            raise RuntimeError(
                "FR13 STATELESS-TREE conv commit: missing staged conv_state/"
                "spec_idx for layer " + str(_prefix)
            )
        _spec_cols = int(_ssi.shape[1])
        _dev = _ssi.device
        _rows = torch.arange(replay_rows, device=_dev)
        _alen = accepted_lens_buf[:replay_rows].to(_dev)
        _leaf_pos = (_alen - 1).clamp(min=0)
        _leaf_node = accepted_path_buf[:replay_rows].to(_dev).gather(
            1, _leaf_pos.view(-1, 1).to(torch.long)
        ).view(-1)
        # zero-accept rows commit the ROOT window: node col 0.
        _leaf_node = torch.where(
            _alen > 0, _leaf_node,
            torch.zeros_like(_leaf_node),
        ).clamp(0, _spec_cols - 1).to(torch.long)
        _dst = _ssi[:replay_rows, 0].to(torch.long)
        if do_commit:
            # NOTE (2026-07-12): the ACTIVE ship path is FR13_TREE_CONV_FUSED (the
            # fused conv bank is +1-ANCHORED, tree_n=10, col0=anchor, tree-node k
            # -> col k+1), so accepted_path_buf=node_id+1 is the CORRECT column
            # here. A tried "-1 node-index" fix (from mis-reading the DEAD non-
            # fused L3482 forward) was empirically REFUTED: same-config A/B showed
            # +1 => clean coherent code (with the _rows identifier garble),
            # -1 => 95% syntax-broken degeneracy. Keep +1. The _rows garble root
            # is NOT this committer column; re-hunt on the fused conv path.
            _src = _ssi[_rows, _leaf_node].to(torch.long)
            # index_select snapshots the source rows before the write -> no alias
            # hazard even when a leaf row coincides with col 0.
            _conv.index_copy_(0, _dst, _conv.index_select(0, _src))


def _fr13_replay_durable_ab_enabled():
    """Resolve FR13_REPLAY_DURABLE_AB for the EngineCore worker.

    The GDN committer runs in the mp/spawn EngineCore worker, whose curated env
    drops bare FR13_* masters (measured 2026-06-14: SUBOP_MAB master dropped).
    The patcher runs in pid 1 where the var IS present and writes a sidecar flag
    into /logs (host + worker visible bind mount).  Env first (explicit wins,
    incl. "0"), else the sidecar.  Default OFF (no env, no sidecar) ->
    byte-identical locked path.  Cached after first resolution.
    """
    global _FR13_RDAB_FLAG
    if _FR13_RDAB_FLAG is not None:
        return _FR13_RDAB_FLAG
    val = __import__('os').environ.get("FR13_REPLAY_DURABLE_AB")
    if val is not None and val != "":
        _FR13_RDAB_FLAG = val == "1"
        return _FR13_RDAB_FLAG
    try:
        flag_path = __import__('os').environ.get(
            "FR13_REPLAY_DURABLE_AB_FLAG_FILE",
            "/logs/fr13_replay_durable_ab.flag",
        )
        with open(flag_path, "r") as _fh:
            _FR13_RDAB_FLAG = _fh.read().strip() == "1"
    except Exception:
        _FR13_RDAB_FLAG = False
    return _FR13_RDAB_FLAG


def _fr13_rdab_stage(tag, msg, once=True, level="error"):
    """Class-9 stage marker: grep-able `FR13_RDAB_STAGE=<tag>` ERROR line so the
    failing stage (env-in-worker / engaged / record-written / arm-fail) is
    unmistakable in the worker log flood.  Always bumps a monotone counter so a
    reducer can assert non-zero records vs non-zero engaged (vacuous-boot
    discriminator).  Callers guard behind _fr13_replay_durable_ab_enabled().
    """
    _FR13_RDAB_STAGE_COUNTS[tag] = int(_FR13_RDAB_STAGE_COUNTS.get(tag, 0)) + 1
    if once:
        if tag in _FR13_RDAB_STAGE_SEEN:
            return
        _FR13_RDAB_STAGE_SEEN.add(tag)
    try:
        emit = getattr(logger, level, None) or logger.error
        emit("FR13_RDAB_STAGE=%s %s", tag, msg)
    except Exception:
        pass


def _fr13_rdab_emit(record):
    """Append one JSONL durable-AB record; eager-only (fail-loud on capture)."""
    global _FR13_RDAB_FH, _FR13_RDAB_HEADER_DONE, _FR13_RDAB_RECORDS
    try:
        if (
            torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "FR13_REPLAY_DURABLE_AB is eager-only: emit during CUDA-graph "
                "capture (boot ENFORCE_EAGER=1)"
            )
    except RuntimeError:
        raise
    except Exception:
        pass
    if _FR13_RDAB_FH is None:
        _path = __import__('os').environ.get(
            "FR13_REPLAY_DURABLE_AB_PATH",
            "/logs/fr13_replay_durable_ab.jsonl",
        )
        _parent = __import__('os').path.dirname(_path)
        if _parent:
            __import__('os').makedirs(_parent, exist_ok=True)
        _FR13_RDAB_FH = open(_path, "a", buffering=1)
    if not _FR13_RDAB_HEADER_DONE:
        _FR13_RDAB_HEADER_DONE = True
        _FR13_RDAB_FH.write(__import__("json").dumps({
            "tap": "header",
            "regime": "eager",
            "ts": round(__import__("time").time(), 6),
            "pid": __import__('os').getpid(),
            "ref_kernel": "fused_sigmoid_gating_delta_rule_update",
            "flags": {
                _k: __import__('os').environ.get(_k)
                for _k in (
                    "FR13_REPLAY_DURABLE_AB",
                    "FR13_REPLAY_DURABLE_AB_LAYERS",
                    "FR13_REPLAY_ROUTE",
                    "FR10_ENABLE_TREE_GDN",
                    "VLLM_BATCH_INVARIANT",
                    "FR13_EAGER_PACK",
                )
            },
        }) + "\n")
    _rec = dict(record)
    _rec["regime"] = "eager"
    _rec["ts"] = round(__import__("time").time(), 6)
    _FR13_RDAB_FH.write(
        __import__("json").dumps(_rec, default=str) + "\n"
    )
    _FR13_RDAB_RECORDS += 1


def _fr13_rdab_layer_match(prefix):
    """Optional layer filter (default: every GDN layer)."""
    pats = __import__('os').environ.get("FR13_REPLAY_DURABLE_AB_LAYERS", "")
    if not pats.strip():
        return True
    return any(
        _p.strip() and _p.strip() in str(prefix)
        for _p in pats.split(",")
    )


def _fr13_replay_durable_ab(
    gdn_mod, prefix, layer, bank, rows,
    accepted_node_paths, accepted_lens_list, event_index,
):
    """OBSERVE-ONLY: per accepted-chain row, compare OUR replay's published
    durable GDN state (H_ours = bank row at the final accepted column, just
    written by _fr13_replay_launch) against NATIVE MTP's sequential recurrent
    kernel run over the SAME root+accepted chain from the SAME CLONED h0.

    H_native_seq = fused_sigmoid_gating_delta_rule_update(...) final state, with
    inplace_final_state=False so native writes into a FRESH (M,HV,V,K) tensor;
    the served bank is read-only here.  The accepted chain is a LINEAR sequence
    of length M = acc_len+1 (root node 0 + accepted path), passed as a single
    B=1 varlen sequence (cu_seqlens=[0,M]) WITHOUT ssm_state_indices /
    num_accepted_tokens -> IS_CONTINUOUS_BATCHING / IS_SPEC_DECODING are both
    False, dodging the reduced-row M5/M1 tree-slice geometry that device-
    asserted in the conv/scan A/B (kernel-valid by construction, never caught).
    """
    if not _fr13_replay_durable_ab_enabled():
        return
    if not _fr13_rdab_layer_match(prefix):
        return
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update as _rdab_native,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    spec_idx = layer._fr13_replay_spec_idx
    prev_lens = layer._fr13_replay_prev_lens
    ring_k = layer._fr13_replay_ring_k
    ring_v = layer._fr13_replay_ring_v
    ring_a = layer._fr13_replay_ring_a
    ring_b = layer._fr13_replay_ring_b
    A_log = layer.A_log
    dt_bias = layer.dt_bias
    n_pad = int(ring_k.size(1))
    # ring layout: k(B,N,KH,DK) v(B,N,VH,DV) a/b(B,N,VH); bank(rows,VH,DV,DK)
    num_kh = int(ring_k.size(2))
    num_vh = int(ring_v.size(2))
    dim_k = int(ring_k.size(3))
    dim_v = int(ring_v.size(3))
    scale = float(dim_k ** -0.5)
    for _b in range(rows):
        try:
            _path = [int(x) for x in accepted_node_paths[_b]]
            _alen = int(accepted_lens_list[_b])
            _prev_len = int(prev_lens[_b].detach().cpu().item())
            _h0_col = max(_prev_len - 1, 0)
            _spec_row = spec_idx[_b].detach().cpu().tolist()
            _h0_row = int(_spec_row[min(_h0_col, len(_spec_row) - 1)])
            # node chain = root(0) + accepted path tokens, clamped to ring.
            _nodes = [0] + [
                max(0, min(int(n), n_pad - 1)) for n in _path[:_alen]
            ]
            _M = len(_nodes)
            _node_t = torch.tensor(
                _nodes, device=ring_k.device, dtype=torch.long
            )
            # flat (1, M, ...) gather of the SAME activations the replay reads;
            # CLONE so nothing aliases the served rings.
            _k = ring_k[_b].index_select(0, _node_t).unsqueeze(0).contiguous()
            _v = ring_v[_b].index_select(0, _node_t).unsqueeze(0).contiguous()
            _a = ring_a[_b].index_select(0, _node_t).unsqueeze(0).contiguous()
            _bb = ring_b[_b].index_select(0, _node_t).unsqueeze(0).contiguous()
            # q is irrelevant to the durable state (q-side never touches it);
            # native still needs a tensor of the right (1,M,KH,DK) shape.
            _q = torch.zeros_like(_k)
            # h0 CLONE shaped (1, HV, V, K): IS_CONTINUOUS_BATCHING=False reads
            # h0 + bos*HV*V*K with bos=0 -> our cloned single row.
            _h0 = bank[_h0_row].to(torch.float32).unsqueeze(0).clone()
            _cu = torch.tensor(
                [0, _M], device=ring_k.device, dtype=torch.int32
            )
            _out, _final = _rdab_native(
                A_log=A_log,
                a=_a,
                b=_bb,
                dt_bias=dt_bias,
                q=_q,
                k=_k,
                v=_v,
                initial_state=_h0,
                inplace_final_state=False,
                cu_seqlens=_cu,
                use_qk_l2norm_in_kernel=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _h_native = _final[_M - 1].to(torch.float32)
            # H_ours = the bank row our replay just published at the final
            # accepted linear column (clamp(acc_len-1,0)).
            _final_col = max(_alen - 1, 0)
            _final_row = int(_spec_row[min(_final_col, len(_spec_row) - 1)])
            _h_ours = bank[_final_row].to(torch.float32)
            _diff = (_h_ours - _h_native).abs()
            _max_abs = float(_diff.max().item())
            # per-head max for first-nonzero head locality.
            _per_head = _diff.reshape(num_vh, -1).amax(dim=1)
            _nz_heads = int((_per_head > 0).sum().item())
            _fr13_rdab_emit({
                "tap": "durable_ab",
                "layer": str(prefix),
                "slot": int(_b),
                "event": int(event_index),
                "accepted_len": int(_alen),
                "M_chain": int(_M),
                "node_chain": _nodes,
                "h0_src_row": int(_h0_row),
                "final_dst_row": int(_final_row),
                "max_abs_h_ours_vs_native": _max_abs,
                "num_nonzero_heads": _nz_heads,
                "num_vh": int(num_vh),
                "ref_final_finite": bool(torch.isfinite(_h_native).all().item()),
            })
            _fr13_rdab_stage(
                "record-written",
                "first record-written ok layer=" + str(prefix)
                + " max_abs=" + repr(_max_abs),
            )
        except Exception as _rdab_exc:  # observe-only: never poison the served
            _fr13_rdab_stage(
                "arm-fail",
                "native A/B arm failed layer=" + str(prefix)
                + " slot=" + str(_b) + " "
                + type(_rdab_exc).__name__ + ":" + str(_rdab_exc),
                once=False,
            )


# LUMO_TREE_PATH_LCP_MAX: greedy N-spine verifier.
#
# Gate B is greedy/deterministic, so max-LCP over root-to-leaf paths is the
# correct deterministic tree accept rule. The same rows are diagnostics only for
# sampled Gate C; sampled production must use the distribution-preserving tree
# rejection sampler, not this max selector.
def _fr13_sg_capchk(tag):
    """FR13_CAPDBG phase probe, TRI-STATE (boot-24 fix): torch's
    is_current_stream_capturing() is `status != None`, which stays True for
    an INVALIDATED capture (the allocator needs it that way) — the old probe
    was structurally blind to the exact event we hunt. Read the raw
    cudaStreamIsCapturing status via ctypes instead: 0=None 1=Active
    2=Invalidated. Fires once at the FIRST probe seeing status != 1, and
    tracks the last Active tag so the failure is bracketed. Inert unless
    FR13_CAPDBG=1 AND a capture is armed; a status query, never a sync."""
    import os
    if os.environ.get("FR13_CAPDBG") != "1":
        return
    if not globals().get("_FR13_SG_CAPCHK_ARMED"):
        return
    st = -1
    try:
        import ctypes
        lib = globals().get("_FR13_SG_CUDART")
        if lib is None:
            for _cand in (None, "libcudart.so.13", "libcudart.so.12",
                          "libcudart.so"):
                try:
                    _lib = ctypes.CDLL(_cand)
                    _lib.cudaStreamIsCapturing  # symbol probe
                    lib = _lib
                    break
                except (OSError, AttributeError):
                    continue
            globals()["_FR13_SG_CUDART"] = lib
        if lib is None:
            st = -3
        else:
            _st = ctypes.c_int(-1)
            _rc = lib.cudaStreamIsCapturing(
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                ctypes.byref(_st))
            st = int(_st.value) if _rc == 0 else -(100 + int(_rc))
    except Exception:
        st = -2
    if st == 1:
        globals()["_FR13_SG_CAPCHK_LAST_ACTIVE"] = tag
        return
    if not globals().get("_FR13_SG_CAPCHK_FIRED"):
        globals()["_FR13_SG_CAPCHK_FIRED"] = True
        print(
            "[FR13_CAPDBG] capture status " + str(st) + " at phase: "
            + str(tag) + " (last Active: "
            + str(globals().get("_FR13_SG_CAPCHK_LAST_ACTIVE")) + ")",
            flush=True,
        )


def _fr13_sg_consumed_ptrs(stacks, tbl, pbuf, lbuf, banks, perm):
    """boot-32 ptr audit: addresses of every tensor the batched committer
    launch consumes. Baked at capture; compared at replay — a mismatch names
    the stale-consumption hole behind the boot-31 garble."""
    d = {}
    for k in ('spec_idx', 'prev_lens', 'ring_k', 'ring_v', 'ring_a',
              'ring_b', 'A_log', 'dt_bias'):
        try:
            d['stacks.' + k] = int(stacks[k].data_ptr())
        except Exception:
            d['stacks.' + k] = -1
    try:
        d['tbl.anchor'] = int(tbl[4].data_ptr())
        d['tbl.off16'] = int(tbl[1].data_ptr())
    except Exception:
        pass
    d['pbuf'] = int(pbuf.data_ptr())
    d['lbuf'] = int(lbuf.data_ptr())
    d['perm'] = int(perm.data_ptr())
    for i, b in enumerate(banks):
        if i in (0, len(banks) - 1):
            d['bank.%d' % i] = int(b.data_ptr())
    return d


def _fr13_sg_commit_device_route(products, output_token_ids, accepted_tree_rows, dm):
    """S1-full (FR13_STEP_GRAPH=2) in-capture committer: consume TAW defer
    products entirely on-device — product fill + GDN buffer fills + conv col0
    commit + ONE batched replay launch. ALL host publishes are deferred to
    fr13_sg_post_replay_publish (wrapper calls it after capture/replay). Host
    validation reads (flags tolist, ptr compares) are capture-illegal and run
    only on STAGED steps of the same boot (warmup guarantees >=1). Every
    missing-state case RAISES => capture aborts => S1 DISABLED staged fallback."""
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _g
    _fr13_sg_capchk("route-entry (sampler+walk done)")
    row_buf, row_len, path_buf, path_len = products
    nreq = int(row_buf.shape[0])
    gdn_paths, _gdn_rows = dm.fr13_taw_products_device(
        row_buf, row_len, path_buf, path_len, output_token_ids,
        accepted_tree_rows)
    pbuf = getattr(_g, '_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR', None)
    lbuf = getattr(_g, '_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR', None)
    perm = getattr(dm, '_FR13_SG_PERM', None)
    stacks = getattr(_g, '_FR13_EAGER_PACK_STACKS', None)
    layers = getattr(_g, '_FR13_REPLAY_LAYERS', None)
    tbl = getattr(_g, '_FR13_EAGER_PACK_BANK_TBL', None)
    if (pbuf is None or lbuf is None or perm is None or stacks is None
            or not layers or tbl is None):
        raise RuntimeError(
            'S1 device route: missing device state '
            f'(pbuf={pbuf is not None} lbuf={lbuf is not None} '
            f'perm={perm is not None} stacks={stacks is not None} '
            f'layers={bool(layers)} tbl={tbl is not None})')
    if int(perm.numel()) != nreq or int(pbuf.size(0)) < nreq:
        raise RuntimeError(
            f'S1 device route rows mismatch: perm={int(perm.numel())} '
            f'nreq={nreq} pbuf_rows={int(pbuf.size(0))}')
    import os as _os
    _runrow_commit = _os.environ.get('FR13_APC_COMMIT_TO_RUNNING_ROW', '1') == '1'
    _runrow_init = _os.environ.get('FR13_TREE_RUNROW_INIT', '1') == '1'
    # FR13_APC_BURN_NODE_BANK DELETED 2026-07-27 (burn redundant for the served path)
    if not (_runrow_commit and _runrow_init):
        raise RuntimeError(
            'S1 device route requires stateless-tree defaults '
            '(commit=init=1)')
    globals()['_FR13_SG_DEFER_STASH'] = (row_buf, row_len, path_buf, path_len)
    import os as _os2
    if _os2.environ.get("FR13_STEP_GRAPH_SCOPE", "full") == "half":
        # SCOPE=half (bisect + S1-mid shape): capture = sampler+walk+products;
        # the state-commit half (fills+conv+launch) runs EAGERLY post-replay
        # via _fr13_sg_commit_state_part.
        return output_token_ids
    cols = int(pbuf.size(1))
    k = min(cols, int(gdn_paths.shape[1]))
    # fill 1: committer-row order (conv col0 commit consumes this ordering,
    # mirroring the staged tail's first fill)
    pbuf[:nreq, :cols].zero_()
    lbuf[:nreq].zero_()
    pbuf[:nreq, :k].copy_(gdn_paths[:, :k].to(pbuf.dtype))
    lbuf[:nreq].copy_(path_len.to(lbuf.dtype))
    _fr13_sg_capchk("post-fill1")
    _fr13_conv_commit_to_col0(layers, pbuf, lbuf, nreq, _runrow_commit)
    _fr13_sg_capchk("post-conv-commit")
    # fill 2: replay (spec-row) order via the wrapper-refilled perm static —
    # the staged tail's host-list reorder, expressed as index_select
    pbuf[:nreq, :k].copy_(gdn_paths.index_select(0, perm)[:, :k].to(pbuf.dtype))
    lbuf[:nreq].copy_(path_len.index_select(0, perm).to(lbuf.dtype))
    from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
        launch_tree_gdn_replay_all_layers as _fr13_sg_launch_all,
    )
    _order = list(stacks['layer_order'])
    _banks = [layers[_n]._fr13_replay_ssm_state for _n in _order]
    if torch.cuda.is_current_stream_capturing():
        _g._FR13_SG_CONSUMED_PTRS = _fr13_sg_consumed_ptrs(
            stacks, tbl, pbuf, lbuf, _banks, perm)
    # boot-42 TWO-PHASE commit: independent layers first, post-full-attn
    # (remap-adjacent) layers in a second stream-ordered launch — the one
    # batched launch broke the staged path's sequential cross-layer
    # dependency (boots 39-41 state conviction).
    _adj_split = int(stacks.get('adj_split', len(_order)))
    for _ph_lo, _ph_hi in ((0, _adj_split), (_adj_split, len(_order))):
        if _ph_lo == _ph_hi:
            continue
        _fr13_sg_launch_all(
            bank_anchor=tbl[4], bank_off16=tbl[1][_ph_lo:_ph_hi],
            bank_shape=tbl[2], bank_stride=tbl[3],
            spec_state_indices=stacks['spec_idx'][_ph_lo:_ph_hi],
            prev_lens=stacks['prev_lens'][_ph_lo:_ph_hi],
            accepted_paths=pbuf, accepted_lens=lbuf,
            k_rings=stacks['ring_k'][_ph_lo:_ph_hi],
            v_rings=stacks['ring_v'][_ph_lo:_ph_hi],
            a_rings=stacks['ring_a'][_ph_lo:_ph_hi],
            b_rings=stacks['ring_b'][_ph_lo:_ph_hi],
            A_logs=stacks['A_log'][_ph_lo:_ph_hi],
            dt_biases=stacks['dt_bias'][_ph_lo:_ph_hi],
            num_layers=_ph_hi - _ph_lo, num_spec_decodes=nreq,
            output_scale=float(stacks['output_scale']),
            use_qk_l2norm_in_kernel=True,
            runrow_commit=_runrow_commit, runrow_init=_runrow_init,
            burn_node_bank=False,  # burn DELETED 2026-07-27 (kernel kwarg kept, dead)
            banks_list=_banks[_ph_lo:_ph_hi],
        )
    _fr13_sg_capchk("post-committer-launch")
    # NOTE: flags deliberately NOT zeroed in-graph (boot-22 death: the
    # staged tail validates flags==1 BEFORE its launch call; the lib-entry
    # consume-once guard then absorbs the launch => no double-commit, and
    # the next forward re-stages flags fresh)
    _fr13_sg_capchk("route-return")
    return output_token_ids


def _fr13_sg_commit_state_part(dm, stash):
    """SCOPE=half eager state-commit: fills+conv+ONE batched launch from the
    graph's stashed device products — identical ops to the in-graph commit
    half, just outside the capture. Uses the same perm static."""
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _g
    row_buf, row_len, path_buf, path_len = stash
    nreq = int(row_buf.shape[0])
    otid = torch.empty(nreq, row_buf.shape[1], dtype=torch.long, device=row_buf.device)
    atr = torch.empty(nreq, dtype=torch.long, device=row_buf.device)
    gdn_paths, _r = dm.fr13_taw_products_device(
        row_buf, row_len, path_buf, path_len, otid, atr)
    pbuf = getattr(_g, '_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR')
    lbuf = getattr(_g, '_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR')
    perm = getattr(dm, '_FR13_SG_PERM')
    stacks = getattr(_g, '_FR13_EAGER_PACK_STACKS')
    layers = getattr(_g, '_FR13_REPLAY_LAYERS')
    tbl = getattr(_g, '_FR13_EAGER_PACK_BANK_TBL')
    import os as _os3
    _rc = _os3.environ.get('FR13_APC_COMMIT_TO_RUNNING_ROW', '1') == '1'
    _ri = _os3.environ.get('FR13_TREE_RUNROW_INIT', '1') == '1'
    cols = int(pbuf.size(1))
    k = min(cols, int(gdn_paths.shape[1]))
    pbuf[:nreq, :cols].zero_()
    lbuf[:nreq].zero_()
    pbuf[:nreq, :k].copy_(gdn_paths[:, :k].to(pbuf.dtype))
    lbuf[:nreq].copy_(path_len.to(lbuf.dtype))
    _fr13_conv_commit_to_col0(layers, pbuf, lbuf, nreq, _rc)
    pbuf[:nreq, :k].copy_(gdn_paths.index_select(0, perm)[:, :k].to(pbuf.dtype))
    lbuf[:nreq].copy_(path_len.index_select(0, perm).to(lbuf.dtype))
    from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
        launch_tree_gdn_replay_all_layers as _la)
    _order = list(stacks['layer_order'])
    _banks = [layers[_n]._fr13_replay_ssm_state for _n in _order]
    from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
        _fr13_native_committer_all_layers_device as _dc)
    _dc(banks_list=_banks, spec_state_indices=stacks['spec_idx'],
        accepted_paths=pbuf, accepted_lens=lbuf,
        k_rings=stacks['ring_k'], v_rings=stacks['ring_v'],
        a_rings=stacks['ring_a'], b_rings=stacks['ring_b'],
        A_logs=stacks['A_log'], dt_biases=stacks['dt_bias'],
        num_layers=len(_order), num_spec_decodes=nreq,
        output_scale=float(stacks['output_scale']),
        use_qk_l2norm_in_kernel=True, burn_node_bank=False)  # burn DELETED 2026-07-27
    # NOTE: flags deliberately NOT zeroed in-graph (boot-22 death: the
    # staged tail validates flags==1 BEFORE its launch call; the lib-entry
    # consume-once guard then absorbs the launch => no double-commit, and
    # the next forward re-stages flags fresh)


def fr13_sg_post_replay_publish(stash, ndt_host):
    """S1-full (=2) host tail, called by the wrapper AFTER capture/replay
    (outside the graph): ONE batched DtoH materialization from the graph's
    static product buffers, then the same publishes the staged tail does —
    _LUMO_TREE_LAST_* globals, gdn freshness lists, by_req publish, and the
    CONV_PREGATHER trigger. Boundary/durable-AB instrument paths are =2-
    ineligible (wrapper skips), so their counters are not replicated here."""
    import sys
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _g
    dm = sys.modules.get('_fr13_device_multidraft_kernel')
    if dm is None or stash is None:
        raise RuntimeError('S1 post-replay publish: missing dm module/stash')
    row_buf, row_len, path_buf, path_len = stash
    (out_rows, accepted_rows, accepted_lens, accepted_node_paths,
     accepted_token_rows) = dm.fr13_taw_materialize(
        row_buf, row_len, path_buf, path_len)
    globals()['_LUMO_TREE_LAST_ACCEPTED_ROWS_KERNEL'] = [
        int(x) for x in accepted_rows]
    globals()['_LUMO_TREE_LAST_ACCEPTED_LENS_KERNEL'] = [
        int(x) for x in accepted_lens]
    globals()['_LUMO_TREE_LAST_ACCEPTED_NODE_PATHS_KERNEL'] = [
        [int(x) for x in row] for row in accepted_node_paths]
    _gdn_paths_h = [
        [int(n) + 1 for n in p[: int(l)]]
        for p, l in zip(accepted_node_paths, accepted_lens)]
    _gdn_rows_h = [(p[-1] if p else 0) for p in _gdn_paths_h]
    _g._LUMO_FA_TREE_COMMIT_NROWS = len(_gdn_paths_h)
    _g._LUMO_FA_LAST_ACCEPTED_TREE_ROWS = [int(x) for x in _gdn_rows_h]
    _g._LUMO_FA_LAST_ACCEPTED_TREE_LENS = [int(x) for x in accepted_lens]
    _g._LUMO_FA_LAST_ACCEPTED_TREE_NODE_PATHS = [
        list(r) for r in _gdn_paths_h]
    _g._LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS = [
        [int(x) for x in row] for row in accepted_token_rows]
    _rid = getattr(_g, '_LUMO_FA_SAMPLER_ROW_REQ_IDS', None)
    if _rid is None or len(_rid) < len(_gdn_paths_h):
        raise RuntimeError(
            'S1 post-replay publish: missing/short sampler row req ids')
    _by_req = getattr(_g, '_LUMO_FA_TREE_ACCEPT_BY_REQ', None)
    if _by_req is None:
        _by_req = {}
        _g._LUMO_FA_TREE_ACCEPT_BY_REQ = _by_req
    for _i in range(len(_gdn_paths_h)):
        if ndt_host is not None and int(ndt_host[_i]) <= 0:
            continue
        _by_req[str(_rid[_i])] = (
            list(_gdn_paths_h[_i]), int(accepted_lens[_i]))
    if getattr(_g, '_FR13_CONV_PREGATHER_ON', False):
        _stacks = getattr(_g, '_FR13_EAGER_PACK_STACKS', None)
        _layers = getattr(_g, '_FR13_REPLAY_LAYERS', None)
        try:
            from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
                launch_conv_col0_pregather as _fr13_sg_cpg,
            )
            _order = list(_stacks['layer_order']) if _stacks else []
            _banks = [_layers[_n]._fr13_replay_conv_state for _n in _order]
            _pairs = getattr(_g, '_LUMO_FA_SPEC_ROW_CONV_COL0', None)
            _seq = getattr(_g, '_LUMO_FA_STEP_SEQ', None)
            if (all(_b is not None for _b in _banks)
                    and _pairs is not None and _seq is not None):
                _fr13_sg_cpg(
                    conv_banks=_banks,
                    ssi_stack=_stacks['spec_idx'][:, :, 0].contiguous(),
                    num_spec_decodes=len(_gdn_paths_h),
                    req_ids_token=(_pairs, int(_seq)),
                )
        except Exception as _fr13_sg_cpg_exc:
            raise RuntimeError(
                'S1 post-replay pregather failed: '
                + repr(_fr13_sg_cpg_exc)) from _fr13_sg_cpg_exc


def _fr13_fixed32_taw_work_callback(payload):
    """Capture one exact TAW work payload without materializing device data."""
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _g
    if getattr(_g, "_FR13_FIXED32_PENDING_TAW", None) is not None:
        raise RuntimeError("FR13 fixed32 TAW callback overlapped an event")
    _g._FR13_FIXED32_PENDING_TAW = payload


try:
    # Baked as a literal by the patcher prelude (pid 1, where the FR13_*
    # master env is present -- the mp/spawn EngineCore worker's curated env
    # drops bare masters, so reading os.environ here would silently disarm).
    # A bare exec of this source (the patcher self-test) has no prelude, and
    # OFF is the byte-identical incumbent, so NameError defaults OFF.
    _FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS
except NameError:
    _FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS = False


def _fr13_fixed32_device_commit_route(
    products,
    output_token_ids,
    accepted_tree_rows,
):
    """Consume the fixed TAW products through one strict device-only route."""
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _g
    if not isinstance(products, tuple) or len(products) != 5:
        raise RuntimeError("FR13 fixed32 TAW must return exactly five tensors")
    (
        device_output,
        device_output_lens,
        device_paths,
        device_lens,
        device_last_rows,
    ) = products
    tensors = (
        device_output,
        device_output_lens,
        device_paths,
        device_lens,
        device_last_rows,
    )
    if any(not torch.is_tensor(_x) for _x in tensors):
        raise RuntimeError("FR13 fixed32 TAW returned a non-tensor product")
    batch = int(device_output.shape[0])
    expected_shapes = (
        (batch, 32),
        (batch,),
        (batch, 16),
        (batch,),
        (batch,),
    )
    if (
        not 1 <= batch <= 4
        or tuple(tuple(_x.shape) for _x in tensors) != expected_shapes
        or any(_x.dtype != torch.int64 for _x in tensors)
        or any(_x.device != device_output.device for _x in tensors)
    ):
        raise RuntimeError(
            "FR13 fixed32 TAW product geometry/dtype/device drift"
        )
    if (
        int(output_token_ids.shape[0]) != batch
        or int(output_token_ids.shape[1]) != 32
        or int(accepted_tree_rows.shape[0]) != batch
        or output_token_ids.dtype != torch.int32
        or accepted_tree_rows.dtype != torch.int32
        or output_token_ids.device != device_output.device
        or accepted_tree_rows.device != device_output.device
    ):
        raise RuntimeError(
            "FR13 fixed32 caller output geometry/dtype/device drift"
        )
    _fixed_batch_rows = int(
        getattr(_g, "_FR13_FIXED32_BATCH_ROWS", -1)
    )
    _fixed_spec_rows = int(
        getattr(_g, "_FR13_FIXED32_SPEC_ROWS", -1)
    )
    _fixed_observed = getattr(_g, "_FR13_FIXED32_OBSERVED_CURRENT", None)
    _fixed_pure_event = isinstance(_fixed_observed, dict)
    if (
        _fixed_spec_rows != batch
        or _fixed_batch_rows < batch
        or (
            _fixed_pure_event
            and (
                _fixed_batch_rows != batch
                or int(
                    getattr(
                        _g,
                        "_FR13_FIXED32_CURRENT_FORWARD_STEP",
                        -1,
                    )
                )
                != int(_fixed_observed.get("forward_step_index", -2))
            )
        )
        or (
            not _fixed_pure_event
            and _fixed_batch_rows == batch
            and not getattr(
                _g, "_FR13_FIXED32_EAGER_KERNEL_DIAGNOSTIC", False
            )
        )
    ):
        raise RuntimeError(
            "FR13 fixed32 commit pure/mixed row geometry drift"
        )

    output_token_ids.copy_(device_output)
    accepted_tree_rows.copy_(device_last_rows)
    slot_paths = getattr(_g, "_LUMO_FA_FIXED32_SLOT_PATHS", None)
    slot_lens = getattr(_g, "_LUMO_FA_FIXED32_SLOT_LENS", None)
    spec_paths = getattr(_g, "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR", None)
    spec_lens = getattr(_g, "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR", None)
    if (
        slot_paths is None
        or slot_lens is None
        or spec_paths is None
        or spec_lens is None
        or int(slot_paths.shape[1]) != 16
        or int(spec_paths.shape[1]) != 16
        or int(slot_paths.shape[0]) < batch
        or int(spec_paths.shape[0]) < batch
        or slot_paths.dtype != torch.int32
        or slot_lens.dtype != torch.int32
        or spec_paths.dtype != torch.int32
        or spec_lens.dtype != torch.int32
        or any(
            tensor.device != device_output.device
            for tensor in (slot_paths, slot_lens, spec_paths, spec_lens)
        )
    ):
        raise RuntimeError("FR13 fixed32 persistent path buffers are missing")
    sampler_req_ids = tuple(
        str(value)
        for value in (
            getattr(_g, "_LUMO_FA_SAMPLER_ROW_REQ_IDS", None) or ()
        )
    )
    spec_req_ids = tuple(
        str(value)
        for value in (
            getattr(_g, "_LUMO_FA_SPEC_ROW_REQ_IDS", None) or ()
        )
    )
    spec_batch_indices = getattr(
        _g, "_FR13_FIXED32_SPEC_BATCH_INDICES", None
    )
    if (
        not 1 <= _fixed_batch_rows <= 4
        or len(sampler_req_ids) != _fixed_batch_rows
        or len(spec_req_ids) != batch
        or not isinstance(spec_batch_indices, tuple)
        or len(spec_batch_indices) != batch
        or any(type(value) is not int for value in spec_batch_indices)
        or len(set(sampler_req_ids)) != len(sampler_req_ids)
        or len(set(spec_req_ids)) != len(spec_req_ids)
        or any(req_id not in sampler_req_ids for req_id in spec_req_ids)
    ):
        raise RuntimeError("FR13 fixed32 compact-spec request map drift")
    slot_indices = tuple(
        sampler_req_ids.index(req_id) for req_id in spec_req_ids
    )
    if slot_indices != spec_batch_indices:
        raise RuntimeError("FR13 fixed32 compact-spec batch-index map drift")
    step_seq = int(getattr(_g, "_LUMO_FA_STEP_SEQ", -1))
    if (
        step_seq <= 0
        or getattr(_g, "_FR13_FIXED32_ACCEPTED_OUTPUT_CURRENT", None)
        is not None
    ):
        raise RuntimeError(
            "FR13 fixed32 accepted-output freshness publish drift"
        )
    _g._FR13_FIXED32_ACCEPTED_OUTPUT_CURRENT = {
        "step_seq": step_seq,
        "request_ids": spec_req_ids,
        "full_request_ids": sampler_req_ids,
        "output_tokens": device_output,
        "output_lens": device_output_lens,
    }
    # Compact spec rows are always dense; only pure events also have
    # sampler-row == compact-spec-row order. The slot family is sparse: only
    # the rows named by slot_indices are written, every other row keeps its
    # previous content, and both forms below honour that.
    #
    # OFF (default) is the incumbent statement-for-statement: 2*B + 2 ATen
    # launches, one per slot row plus the two compact copies. ON is the
    # constant-4-launch form. Byte identity, disjoint-storage and
    # distinct-index preconditions are enforced inside the module, per event.
    from lumo_flywheel_serving.fr13_fixed32_commit_slot_scatter import (
        publish_committer_paths as _fr13_f32_publish_paths,
    )
    _fr13_f32_publish_paths(
        slot_paths=slot_paths,
        slot_lens=slot_lens,
        spec_paths=spec_paths,
        spec_lens=spec_lens,
        device_paths=device_paths,
        device_lens=device_lens,
        slot_indices=slot_indices,
        batch=batch,
        batched=_FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS,
    )
    _g._LUMO_FA_TREE_COMMIT_NROWS = batch

    stacks = getattr(_g, "_FR13_EAGER_PACK_STACKS", None)
    order = stacks.get("fixed32_order") if stacks else None
    layers = stacks.get("fixed32_layers") if stacks else None
    banks = stacks.get("fixed32_banks") if stacks else None
    conv_banks = stacks.get("fixed32_conv_banks") if stacks else None
    bank_anchor = stacks.get("fixed32_bank_anchor") if stacks else None
    bank_shape = stacks.get("fixed32_bank_shape") if stacks else None
    bank_stride = stacks.get("fixed32_bank_stride") if stacks else None
    output_scale = stacks.get("fixed32_output_scale") if stacks else None
    qk_l2norm = stacks.get("fixed32_qk_l2norm") if stacks else None
    if (
        stacks is None
        or not isinstance(order, tuple)
        or not isinstance(layers, tuple)
        or not isinstance(banks, tuple)
        or not isinstance(conv_banks, tuple)
        or len(order) != 48
        or len(layers) != 48
        or len(banks) != 48
        or len(conv_banks) != 48
        or bank_anchor is not banks[0]
        or not isinstance(bank_shape, tuple)
        or len(bank_shape) != 4
        or not isinstance(bank_stride, int)
        or not isinstance(output_scale, float)
        or qk_l2norm is not True
        or int(stacks.get("num_layers", 0)) != 48
    ):
        raise RuntimeError(
            "FR13 fixed32 persistent 48-layer bank tuples are missing"
        )
    if getattr(_g, "_FR13_FIXED32_PENDING_EVENT", None) is not None:
        raise RuntimeError("FR13 fixed32 prior event did not complete KV remap")

    import lumo_flywheel_serving.fr10_gdn_tree_kernel as _fixed_tree_kernel
    from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
        fixed32_committer_counters as _fixed_commit_counters,
        fixed32_conv_col0_commit_counters as _fixed_conv_commit_counters,
        fixed32_conv_col0_pregather_counters as _fixed_pregather_counters,
        launch_fixed32_conv_commit_to_col0 as _fixed_conv_commit,
        launch_tree_gdn_replay_all_layers as _fixed_replay,
    )
    _fixed_commit_before = _fixed_commit_counters()
    _fixed_conv_commit_before = _fixed_conv_commit_counters()
    _fixed_pregather_before = _fixed_pregather_counters()

    _fixed_conv_commit(
        conv_banks=conv_banks,
        spec_state_indices=stacks["spec_idx"],
        accepted_paths=spec_paths,
        accepted_lens=spec_lens,
        num_spec_decodes=batch,
    )
    _fixed_replay(
        bank_anchor=bank_anchor,
        bank_off16=stacks["spec_idx"],
        bank_shape=bank_shape,
        bank_stride=bank_stride,
        spec_state_indices=stacks["spec_idx"],
        prev_lens=stacks["prev_lens"],
        accepted_paths=spec_paths[:batch],
        accepted_lens=spec_lens[:batch],
        k_rings=stacks["ring_k"],
        k_norm_rings=stacks["ring_k_norm"],
        gate_rings=stacks["ring_gate"],
        v_rings=stacks["ring_v"],
        a_rings=stacks["ring_a"],
        b_rings=stacks["ring_b"],
        A_logs=stacks["A_log"],
        dt_biases=stacks["dt_bias"],
        num_layers=48,
        num_spec_decodes=batch,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=qk_l2norm,
        runrow_commit=True,
        runrow_init=True,
        burn_node_bank=False,
        banks_list=banks,
    )
    if (
        not _fixed_pure_event
        and not getattr(
            _g, "_FR13_FIXED32_EAGER_KERNEL_DIAGNOSTIC", False
        )
    ):
        _nonpure_replays = getattr(
            _g, "_FR13_FIXED32_NONPURE_COMMIT_REPLAYS_BY_BATCH", None
        )
        if (
            not isinstance(_nonpure_replays, dict)
            or batch not in _nonpure_replays
            or type(_nonpure_replays[batch]) is not int
            or _nonpure_replays[batch] < 0
        ):
            raise RuntimeError(
                "FR13 fixed32 nonpure committer replay ledger drift"
            )
        _nonpure_replays[batch] += 1
    stacks["flags"][:, 0].zero_()
    _fixed_commit_after = _fixed_commit_counters()
    _fixed_conv_commit_after = _fixed_conv_commit_counters()
    _fixed_pregather_after = _fixed_pregather_counters()
    _fixed_commit_route = getattr(
        _fixed_tree_kernel, "_FR13_FIXED32_COMMITTER_FAST_ROUTE", {}
    ).get("state")
    _fixed_pregather_route = getattr(
        _fixed_tree_kernel, "_FR13_FIXED32_CONV_PREGATHER", {}
    ).get("state")
    if (
        not isinstance(_fixed_commit_route, dict)
        or not isinstance(_fixed_pregather_route, dict)
        or batch not in _fixed_commit_route.get("states_by_batch", {})
    ):
        raise RuntimeError("FR13 fixed32 runtime route state disappeared")
    _fixed_commit_contract = _fixed_commit_route[
        "states_by_batch"
    ][batch].get("contract")
    _fixed_pregather_contract = {
        "route": _fixed_pregather_route.get("contract", {}).get("route"),
        "layers": int(
            _fixed_pregather_route.get("contract", {}).get("layers", -1)
        ),
        "row_elems": int(_fixed_pregather_route.get("row_elems", -1)),
        "block": int(_fixed_pregather_route.get("block", -1)),
        "graph_capture_stages": int(
            _fixed_pregather_route.get("graph_capture_stages", -1)
        ),
    }
    _fixed_conv_commit_contract = {
        "route": _fixed_pregather_route.get("contract", {}).get(
            "commit_route"
        ),
        "layers": int(
            _fixed_pregather_route.get("contract", {}).get("layers", -1)
        ),
        "row_elems": int(_fixed_pregather_route.get("row_elems", -1)),
        "channels": int(_fixed_pregather_route.get("conv_c", -1)),
        "state_length": int(_fixed_pregather_route.get("conv_l", -1)),
        "source_rows_per_batch": int(
            _fixed_pregather_route.get("source_rows_per_batch", -1)
        ),
        "block": int(_fixed_pregather_route.get("block", -1)),
        "staging_reused": _fixed_pregather_route.get("contract", {}).get(
            "commit_staging_reused"
        ),
        "source_staging_reused": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_source_staging_reused"),
        "source_pointer_entries": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_source_pointer_entries"),
        "row_guard_route": _fixed_pregather_route.get("contract", {}).get(
            "commit_row_guard_route"
        ),
        "row_guard_kernel_launches": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_row_guard_kernel_launches_per_event"),
        "row_guard_programs_per_request": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_row_guard_programs_per_request"),
        "row_guard_physical_rows": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_row_guard_physical_rows"),
        "row_guard_path_capacity": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_row_guard_path_capacity"),
        "row_guard_alias_width": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_row_guard_alias_width"),
        "row_guard_compare_capacity": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_row_guard_compare_capacity"),
        "row_guard_path_validation_programs_per_request": (
            _fixed_pregather_route.get("contract", {}).get(
                "commit_row_guard_path_validation_programs_per_request"
            )
        ),
        "row_guard_path_vector_loads_per_request": (
            _fixed_pregather_route.get("contract", {}).get(
                "commit_row_guard_path_vector_loads_per_request"
            )
        ),
        "row_guard_alias_validation_programs_per_event": (
            _fixed_pregather_route.get("contract", {}).get(
                "commit_row_guard_alias_validation_programs_per_event"
            )
        ),
        "row_guard_alias_vector_loads_per_event": (
            _fixed_pregather_route.get("contract", {}).get(
                "commit_row_guard_alias_vector_loads_per_event"
            )
        ),
        "row_guard_selected_row_loads_per_program": (
            _fixed_pregather_route.get("contract", {}).get(
                "commit_row_guard_selected_row_loads_per_program"
            )
        ),
        "row_guard_peer_topology_proof": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_row_guard_peer_topology_proof"),
        "row_guard_torch_index_transforms": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_row_guard_torch_index_transforms"),
        "row_guard_async_scalar_reductions": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_row_guard_async_scalar_reductions"),
        "row_guard_async_assertions": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_row_guard_async_assertions"),
        "full_node_writebacks": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_full_node_writebacks"),
        "conv_remaps": _fixed_pregather_route.get("contract", {}).get(
            "commit_conv_remaps"
        ),
        "commit_bank_overlap_policy": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_bank_overlap_policy"),
        "commit_bank_partial_overlap": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_bank_partial_overlap"),
        "commit_bank_alias_groups": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_bank_alias_groups"),
        "commit_bank_alias_width": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_bank_alias_width"),
        "commit_bank_destination_guard": _fixed_pregather_route.get(
            "contract", {}
        ).get("commit_bank_destination_guard"),
        "commit_null_row_rejected": _fixed_pregather_route.get(
            "contract", {}
        ).get(
            "commit_null_row_rejected"
        ),
        "ssi_bound": _fixed_pregather_route.get("contract", {}).get(
            "commit_ssi_bound"
        ),
        "paths_bound": _fixed_pregather_route.get("contract", {}).get(
            "commit_paths_bound"
        ),
    }
    if _fixed_pure_event:
        _g._fr13_fixed32_observed_commit(
            _FR13_FIXED32_MODE,
            batch,
            len(layers),
            tuple(int(value) for value in spec_paths.shape),
            tuple(int(value) for value in spec_lens.shape),
            _fixed_commit_before,
            _fixed_commit_after,
            _fixed_commit_contract,
            _fixed_conv_commit_before,
            _fixed_conv_commit_after,
            _fixed_conv_commit_contract,
            _fixed_pregather_before,
            _fixed_pregather_after,
            _fixed_pregather_contract,
        )

    taw_work = getattr(_g, "_FR13_FIXED32_PENDING_TAW", None)
    _g._FR13_FIXED32_PENDING_TAW = None
    if (
        not isinstance(taw_work, dict)
        or set(taw_work)
        != {"mode", "valid_mask", "batch_size", "taw"}
        or taw_work.get("mode") != _FR13_FIXED32_MODE
        or type(taw_work.get("valid_mask")) is not int
        or taw_work.get("valid_mask") != _FR13_FIXED32_VALID_MASK
        or type(taw_work.get("batch_size")) is not int
        or taw_work.get("batch_size") != batch
        or not isinstance(taw_work.get("taw"), dict)
    ):
        raise RuntimeError("FR13 fixed32 TAW work callback did not bind")
    _fixed_pack_tensors = (
        device_output,
        output_token_ids,
        device_last_rows,
        accepted_tree_rows,
        device_paths,
        slot_paths[:batch],
        device_lens,
        slot_lens[:batch],
        spec_paths[:batch],
        spec_lens[:batch],
    )
    if _fixed_pure_event:
        _g._fr13_fixed32_observed_publish_pack(
            batch,
            tuple(
                tuple(int(value) for value in tensor.shape)
                for tensor in _fixed_pack_tensors
            ),
            tuple(str(tensor.dtype) for tensor in _fixed_pack_tensors),
            tuple(tensor.device.type for tensor in _fixed_pack_tensors),
            2,
            2,
            2,
            int(taw_work.get("taw", {}).get("loop_iterations", -1)),
        )
        forward_step = int(
            getattr(_g, "_FR13_FIXED32_CURRENT_FORWARD_STEP", -1)
        )
        observed_work = _g._fr13_fixed32_observed_take(
            _FR13_FIXED32_MODE,
            batch,
            forward_step,
        )
        _g._FR13_FIXED32_PENDING_EVENT = {
            "mode": _FR13_FIXED32_MODE,
            "batch_size": batch,
            "forward_step_index": forward_step,
            "request_ids": tuple(observed_work["request_ids"]),
            "taw": taw_work,
            # The verify event remains pending through KV and the immediately
            # following same-runner-step proposal. Proposal end is the sole seal.
            "observed_work": observed_work,
        }
    return output_token_ids


def _lumo_tree_canonical_multidraft_sample(
    output_token_ids: torch.Tensor,
    accepted_tree_rows: torch.Tensor,
    num_draft_tokens,
    draft_token_ids: torch.Tensor,
    tree_parent_indices: torch.Tensor,
    target_logits: torch.Tensor,
    tree_self_logits: torch.Tensor,
    draft_probs: torch.Tensor | None,
    bonus_token_ids: torch.Tensor,
    max_spec_len: int,
    generators=None,
    all_greedy: bool = False,
) -> torch.Tensor:
    """Reference sampled tree committer using the verified FR10 rule.

    ``all_greedy`` (temp-0 committer unification): route temp-0/greedy commits
    through THIS same multidraft committer with POINT-MASS target/self rows so
    the accept rule reduces byte-for-byte and rng-free to the greedy longest-
    prefix committer (byte-gate scripts/fr13_greedy_pointmass_byte_gate.py). This
    lets the separate greedy path-LCP committer be deleted.
    """
    if all_greedy and draft_probs is not None:
        raise RuntimeError(
            'FR13 greedy-via-rejection: all_greedy with draft_probs!=None is '
            'not a valid combination (greedy MTP always passes draft_probs=None)'
        )
    if _FR13_FIXED32_MODE:
        if all_greedy or draft_probs is not None or int(max_spec_len) != 31:
            raise RuntimeError(
                "FR13 fixed32 requires sampled temp>0, no draft_probs, "
                "and max_spec_len=31"
            )
        import importlib.util as _fr13_f32_ilu
        import sys as _fr13_f32_sys
        _fr13_f32_dm = _fr13_f32_sys.modules.get(
            "_fr13_device_multidraft_kernel"
        )
        if _fr13_f32_dm is None:
            _fr13_f32_path = __import__("os").environ.get(
                "FR13_DEVICE_MULTIDRAFT_KERNEL",
                "/workspace/scripts/fr13_device_multidraft_kernel.py",
            )
            _fr13_f32_spec = _fr13_f32_ilu.spec_from_file_location(
                "_fr13_device_multidraft_kernel",
                _fr13_f32_path,
            )
            if _fr13_f32_spec is None or _fr13_f32_spec.loader is None:
                raise RuntimeError("FR13 fixed32 device module cannot load")
            _fr13_f32_dm = _fr13_f32_ilu.module_from_spec(_fr13_f32_spec)
            _fr13_f32_spec.loader.exec_module(_fr13_f32_dm)
            _fr13_f32_sys.modules[
                "_fr13_device_multidraft_kernel"
            ] = _fr13_f32_dm
        _fr13_f32_dm.fr13_fixed32_taw_set_work_callback(
            _fr13_fixed32_taw_work_callback
        )
        (
            _fr13_f32_full_counts,
            _fr13_f32_spec_indices,
        ) = _fr13_fixed32_live_sampler_row_plan(num_draft_tokens)
        _fr13_f32_full_batch = len(_fr13_f32_full_counts)
        _fr13_f32_batch = len(_fr13_f32_spec_indices)
        _fr13_f32_mixed = _fr13_f32_batch != _fr13_f32_full_batch
        _fr13_f32_rows = 31 * _fr13_f32_batch
        if (
            int(target_logits.ndim) != 2
            or int(tree_self_logits.ndim) != 2
            or int(target_logits.shape[1]) <= 0
            or int(target_logits.shape[1])
            != int(tree_self_logits.shape[1])
            or int(draft_token_ids.ndim) != 1
            or int(tree_parent_indices.ndim) != 1
            or int(draft_token_ids.numel()) != _fr13_f32_rows
            or int(tree_parent_indices.numel()) != _fr13_f32_rows
            or int(target_logits.shape[0]) != _fr13_f32_rows
            or int(tree_self_logits.shape[0]) != _fr13_f32_rows
            or int(output_token_ids.ndim) != 2
            or tuple(output_token_ids.shape)
            != (_fr13_f32_full_batch, 32)
            or int(accepted_tree_rows.ndim) != 1
            or tuple(accepted_tree_rows.shape)
            != (_fr13_f32_full_batch,)
            or int(bonus_token_ids.ndim) < 1
            or int(bonus_token_ids.shape[0]) != _fr13_f32_full_batch
            or int(bonus_token_ids.numel()) != _fr13_f32_full_batch
        ):
            raise RuntimeError(
                "FR13 fixed32 compact draft and full sampler geometry drift"
            )
        _fr13_f32_index_tensor = None
        _fr13_f32_output = output_token_ids
        _fr13_f32_accepted = accepted_tree_rows
        _fr13_f32_bonus = bonus_token_ids
        _fr13_f32_generators = generators
        if _fr13_f32_mixed:
            (
                _fr13_f32_index_tensor,
                _fr13_f32_output,
                _fr13_f32_accepted,
                _fr13_f32_bonus,
                _fr13_f32_generators,
            ) = _fr13_fixed32_prepare_mixed_sampler_io(
                output_token_ids,
                accepted_tree_rows,
                bonus_token_ids,
                _fr13_f32_spec_indices,
                generators,
            )
        _fr13_f32_counts = (
            _fr13_f32_dm.fr13_fixed32_taw_preseeded_counts(
                draft_token_ids.device,
                mode=_FR13_FIXED32_MODE,
                valid_mask=_FR13_FIXED32_VALID_MASK,
                batch_size=_fr13_f32_batch,
            )
        )
        _fr13_f32_commit_result = (
            _fr13_f32_dm.fr13_fixed32_cfwd_logit_direct_commit(
                _fr13_f32_counts,
                draft_token_ids,
                tree_parent_indices,
                target_logits,
                tree_self_logits,
                _fr13_f32_bonus,
                max_spec_len,
                generators=_fr13_f32_generators,
                all_greedy=False,
                mode=_FR13_FIXED32_MODE,
            )
        )
        _fr13_f32_dm.fr13_fixed32_cfwd_logit_direct_live_prepare_replay(
            mode=_FR13_FIXED32_MODE,
            batch_size=_fr13_f32_batch,
            enabled=False,
        )
        _fr13_f32_output = _fr13_fixed32_device_commit_route(
            _fr13_f32_commit_result,
            _fr13_f32_output,
            _fr13_f32_accepted,
        )
        if _fr13_f32_mixed:
            return _fr13_fixed32_finish_mixed_sampler_io(
                output_token_ids,
                accepted_tree_rows,
                bonus_token_ids,
                _fr13_f32_index_tensor,
                _fr13_f32_output,
                _fr13_f32_accepted,
            )
        return _fr13_f32_output
    _fr13_sg_capchk("committer-fn-entry")
    if __import__('os').environ.get('FR13_FORCE_SPINE_COMMIT', '0') == '1':
        # FR13_FORCE_SPINE_COMMIT is a GREEDY-committer diagnostic. The
        # sampled committer's sequential child walk cannot force-commit the
        # spine without silently changing acceptance distributions; fail loud
        # so a temp>0 run cannot masquerade as a forced-spine A/B.
        raise RuntimeError(
            'FR13_FORCE_SPINE_COMMIT=1 is diagnostic-only for the GREEDY '
            'tree committer; unset it for sampled (temp>0) decoding'
        )
    import numpy as _fr10_np
    from lumo_flywheel_serving.fr10_tree_rejection_sampler import (
        sample_deterministic_multidraft_rejection_step as _fr10_sample_det_step,
        sample_multidraft_rejection_step as _fr10_sample_step,
    )

    # FR13_SAMPLER_SYNC_KILL (2026-07-25, torchprof-named #1 site): the three
    # DtoH .cpu() syncs below ran EVERY step even on the deployed device
    # committer path, whose per-request host loop is SKIPPED (their only
    # consumer). Each sync drains the async pipeline -> the measured 37%%-of-
    # window host stall. Under device mode: parents_cpu comes from a static
    # per-topology cache (the tree never changes intra-boot) and drafts/
    # counts DtoH are eliminated (needle uses tensor numel). Host-reference
    # mode (FR13_DEVICE_MULTIDRAFT=0) keeps the original conversions
    # byte-for-byte.
    _fr13_dm_pre = (
        __import__('os').environ.get('FR13_DEVICE_MULTIDRAFT', '1') == '1'
        and not __import__('os').path.exists('/logs/fr13_device_multidraft_off.arm')
        and draft_probs is None
    )
    if _fr13_dm_pre:
        _fr13_pc_key = (
            int(tree_parent_indices.data_ptr()), int(tree_parent_indices.numel())
        )
        _fr13_pc_cache = globals().setdefault('_FR13_PARENTS_CPU_CACHE', {})
        parents_cpu = _fr13_pc_cache.get(_fr13_pc_key)
        if parents_cpu is None:
            parents_cpu = [
                int(x) for x in tree_parent_indices.detach().cpu().tolist()
            ]
            _fr13_pc_cache[_fr13_pc_key] = parents_cpu
        drafts_cpu = None
        counts = None
    else:
        parents_cpu = [int(x) for x in tree_parent_indices.detach().cpu().tolist()]
        drafts_cpu = [int(x) for x in draft_token_ids.detach().cpu().tolist()]
        if hasattr(num_draft_tokens, 'detach'):
            counts = [int(x) for x in num_draft_tokens.detach().cpu().tolist()]
        else:
            counts = [int(x) for x in num_draft_tokens]
    # FR13_DEVICE_MULTIDRAFT (BAKED default ON = the deployed cat6/cat9 committer): the temp>0 multidraft
    # per-node decision runs ON-DEVICE (scripts/fr13_device_multidraft_kernel.py)
    # so the [nodes x vocab] host softmax DtoH below + the Python per-node
    # interpreter loop are ELIMINATED. The device committer computes the SAME
    # SpecInfer/multi-draft residual-mix accept rule (per-node source weights ~
    # min-overlap, accept ~ min(1,p/q_mix), residual fallback) and produces the
    # SAME five committer products (distribution-lossless, NOT byte: device rng
    # draws differ but follow the identical distributions; proven offline by
    # scripts/fr13_device_multidraft_offline_gate.py). BAKED DEFAULT-ON (the
    # deployed committer); FR13_DEVICE_MULTIDRAFT=0 restores the host-reference
    # path (byte-identical to HEAD's host-ref, kept for A/B). Fail-loud on
    # disengagement (no silent host fallback, bug-class 9).
    _fr13_device_multidraft = (
        __import__('os').environ.get('FR13_DEVICE_MULTIDRAFT', '1') == '1'
        and not __import__('os').path.exists('/logs/fr13_device_multidraft_off.arm')
        and draft_probs is None
    )
    if _fr13_device_multidraft:
        # device path decides every request -> the [nodes x vocab] host softmax
        # DtoH is SKIPPED (its only consumer is the host per-node loop, which is
        # skipped). The actual on-device commit runs at the loop anchor below.
        target_probs_cpu = None
        self_probs_cpu = None
    elif all_greedy:
        # host fallback, all_greedy: POINT-MASS rows (one-hot on argmax) so the
        # host det-step reduces to greedy longest-prefix, matching the device
        # all_greedy path. Only reached when FR13_DEVICE_MULTIDRAFT=0.
        import numpy as _fr13_gu_np
        _tgt_am = target_logits.argmax(dim=-1).detach().cpu().numpy()
        _self_am = tree_self_logits.argmax(dim=-1).detach().cpu().numpy()
        _vsz = int(target_logits.size(-1))
        target_probs_cpu = _fr13_gu_np.zeros(
            (int(target_logits.size(0)), _vsz), dtype=_fr13_gu_np.float32)
        target_probs_cpu[_fr13_gu_np.arange(_tgt_am.shape[0]), _tgt_am] = 1.0
        self_probs_cpu = _fr13_gu_np.zeros(
            (int(tree_self_logits.size(0)), _vsz), dtype=_fr13_gu_np.float32)
        self_probs_cpu[_fr13_gu_np.arange(_self_am.shape[0]), _self_am] = 1.0
    else:
        target_probs_cpu = target_logits.softmax(dim=-1, dtype=torch.float32).detach().cpu().numpy()
        self_probs_cpu = tree_self_logits.softmax(dim=-1, dtype=torch.float32).detach().cpu().numpy()
    draft_probs_cpu = (
        None if draft_probs is None else draft_probs.detach().cpu().numpy()
    )
    # FR13_TREE_PER_REQ_GEN: seed the rejection-sampling rng from the
    # per-request torch.Generator (sampling_metadata.generators, the same
    # mechanism stock vLLM uses for seeded requests) instead of the GLOBAL
    # torch CUDA RNG. The global stream's offset depends on every prior tree
    # event in the boot AND the draws were consumed sequentially across
    # requests in batch order, so same-seed reruns diverged at the first
    # sampled token (FR13 nondeterminism chase candidate #1). Default ON:
    # the per-draw distributions are unchanged; only the stream source moves
    # to the request's own seeded generator. FR13_TREE_PER_REQ_GEN=0 restores
    # the legacy single global-seeded rng.
    _fr13_per_req_gen = (
        True  # FR13_TREE_PER_REQ_GEN baked ON (locked); legacy global-seeded rng path dead
    )
    rng_global = None
    rng = None

    out_rows = []
    accepted_rows = []
    accepted_lens = []
    sample_log_rows = []
    accepted_node_paths = []
    accepted_token_rows = []
    try:
        import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
        if (
            _fr10_lo.environ.get('FR10_METRICS', '0') == '1'
            or _fr10_lo.environ.get('LUMO_TREE_SAMPLER_DEBUG_LOG')
        ):
            global _LUMO_TREE_SAMPLE_DEBUG_FH
            try:
                _LUMO_TREE_SAMPLE_DEBUG_FH
            except NameError:
                _LUMO_TREE_SAMPLE_DEBUG_FH = open(
                    (
                        _fr10_lo.environ.get('LUMO_TREE_SAMPLER_DEBUG_LOG')
                        or '/logs/tree_sampler_debug.jsonl'
                    ),
                    'a',
                    buffering=1,
                )
            _LUMO_TREE_SAMPLE_DEBUG_FH.write(_fr10_lj.dumps({
                'event': 'sample_helper_enter',
                'ts': round(_fr10_lt.time(), 4),
                'max_spec_len': int(max_spec_len),
                'has_draft_probs': draft_probs is not None,
            }) + chr(10))
    except Exception:
        pass
    start = 0
    # FR13_DEVICE_MULTIDRAFT loop-skip (BAKED default ON). flag-off => _fr13_dm_counts
    # IS counts, so the legacy host per-node Python loop below runs byte-for-byte
    # unchanged. flag-on => the device committer fills the five product lists and
    # the legacy loop iterates EMPTY (the host walk never runs; the [nodes x
    # vocab] softmax DtoH above was already skipped).
    _fr13_dm_counts = counts if counts is not None else []
    if _fr13_device_multidraft:
        try:
            import importlib.util as _fr13_dm_ilu, os as _fr13_dm_os
            _fr13_dm_path = _fr13_dm_os.environ.get(
                'FR13_DEVICE_MULTIDRAFT_KERNEL',
                '/workspace/scripts/'
                'fr13_device_multidraft_kernel.py',
            )
            _fr13_dm = __import__('sys').modules.get('_fr13_device_multidraft_kernel')
            if _fr13_dm is None:
                _fr13_dm_spec = _fr13_dm_ilu.spec_from_file_location(
                    '_fr13_device_multidraft_kernel', _fr13_dm_path)
                _fr13_dm = _fr13_dm_ilu.module_from_spec(_fr13_dm_spec)
                _fr13_dm_spec.loader.exec_module(_fr13_dm)
                __import__('sys').modules['_fr13_device_multidraft_kernel'] = _fr13_dm
            if not globals().get('_FR13_DEVICE_MULTIDRAFT_NEEDLE_DONE'):
                globals()['_FR13_DEVICE_MULTIDRAFT_NEEDLE_DONE'] = True
                try:
                    from vllm.logger import init_logger as _fr13_dm_il
                    _fr13_dm_il('vllm.fr13_device_multidraft').info(
                        'FR13_DEVICE_MULTIDRAFT engaged: device-side temp>0 '
                        'multidraft committer (no [nodes x vocab] softmax DtoH, '
                        'no per-node Python loop), n_req=%d' % int(num_draft_tokens.numel() if hasattr(num_draft_tokens, 'numel') else len(num_draft_tokens))
                    )
                except Exception:
                    print('FR13_DEVICE_MULTIDRAFT engaged', flush=True)
            _fr13_dm_ret = _fr13_dm.fr13_device_multidraft_commit(
                num_draft_tokens,
                draft_token_ids,
                tree_parent_indices,
                target_logits,
                tree_self_logits,
                draft_probs,
                bonus_token_ids,
                max_spec_len,
                generators=generators,
                all_greedy=all_greedy,
            )
            if isinstance(_fr13_dm_ret, tuple) and len(_fr13_dm_ret) == 4:
                # FR13_STEP_GRAPH=2 in-capture route: TAW defer products
                # (device tensors). The staged tail below (host lists, host
                # publishes) never runs inside a capture; the =2 wrapper
                # calls fr13_sg_post_replay_publish after capture/replay.
                return _fr13_sg_commit_device_route(
                    _fr13_dm_ret, output_token_ids, accepted_tree_rows,
                    _fr13_dm)
            (
                out_rows, accepted_rows, accepted_lens,
                accepted_node_paths, accepted_token_rows,
            ) = _fr13_dm_ret
        except Exception as _fr13_dm_exc:
            raise RuntimeError(
                'FR13_DEVICE_MULTIDRAFT failed (no silent fallback, class 9): '
                + type(_fr13_dm_exc).__name__ + ':' + str(_fr13_dm_exc)
            ) from _fr13_dm_exc
        _fr13_dm_counts = []
    for req_i, node_count in enumerate(_fr13_dm_counts):
        node_count = int(node_count)
        _fr13_gen = None
        if _fr13_per_req_gen and generators:
            _fr13_gen = generators.get(req_i)
        if _fr13_gen is not None:
            _fr13_seed = int(
                torch.randint(
                    0,
                    2**31 - 1,
                    (1,),
                    device=_fr13_gen.device,
                    generator=_fr13_gen,
                ).cpu().item()
            )
            rng = _fr10_np.random.default_rng(_fr13_seed)
        else:
            if rng_global is None:
                rng_global = _fr10_np.random.default_rng(
                    int(
                        torch.randint(
                            0, 2**31 - 1, (1,), device=output_token_ids.device
                        ).cpu().item()
                    )
                )
            rng = rng_global
        parents = parents_cpu[start:start + node_count]
        drafts = drafts_cpu[start:start + node_count]
        current_parent = -1
        accepted_row = 0
        accepted_path = []
        step_trace_rows = []
        row = []
        for _step in range(int(max_spec_len) + 1):
            children = [
                node for node, parent in enumerate(parents)
                if int(parent) == int(current_parent)
            ]
            if not children:
                if current_parent >= 0:
                    row.append(
                        int(rng.choice(
                            self_probs_cpu.shape[1],
                            p=self_probs_cpu[start + current_parent],
                        ))
                    )
                elif req_i < int(bonus_token_ids.numel()):
                    row.append(int(bonus_token_ids.reshape(-1)[req_i].detach().cpu().item()))
                break

            _target_row = int(start + children[0])
            _child_drafts = [int(drafts[child]) for child in children]
            _target_probs = target_probs_cpu[_target_row]
            if draft_probs_cpu is None:
                step = _fr10_sample_det_step(
                    _target_probs,
                    _child_drafts,
                    rng=rng,
                )
                _target_prob_at_draft_tokens = [
                    float(_target_probs[int(_tok)]) for _tok in _child_drafts
                ]
                _overlap_mass = float(sum(_target_prob_at_draft_tokens))
                if _overlap_mass > 0.0:
                    _selected_draft_token = int(_child_drafts[int(step.source_index)])
                    _q_mix_token = float(sum(
                        _prob / _overlap_mass
                        for _tok, _prob in zip(
                            _child_drafts, _target_prob_at_draft_tokens
                        )
                        if int(_tok) == int(_selected_draft_token)
                    ))
                    _canonical_accept_prob = (
                        min(
                            1.0,
                            float(_target_probs[int(_selected_draft_token)])
                            / _q_mix_token,
                        )
                        if _q_mix_token > 0.0 else 0.0
                    )
                else:
                    _canonical_accept_prob = 0.0
            else:
                step = _fr10_sample_step(
                    _target_probs,
                    [draft_probs_cpu[start + child] for child in children],
                    rng=rng,
                )
                _target_prob_at_draft_tokens = [
                    float(_target_probs[int(_tok)]) for _tok in _child_drafts
                ]
                _canonical_accept_prob = None
            _selected_child = int(children[int(step.source_index)])
            step_trace_rows.append({
                'step': int(_step),
                'parent_node_id': int(current_parent),
                'child_node_ids': [int(x) for x in children],
                'target_prob_row': int(_target_row),
                'target_argmax': int(_fr10_np.argmax(_target_probs)),
                'draft_token_ids': [int(x) for x in _child_drafts],
                'target_prob_at_draft_token_ids': _target_prob_at_draft_tokens,
                'canonical_accept_prob': (
                    None if _canonical_accept_prob is None
                    else float(_canonical_accept_prob)
                ),
                'selected_source_index': int(step.source_index),
                'selected_child_node_id': int(_selected_child),
                'selected_token_id': int(step.token_id),
                'accepted': bool(step.accepted),
            })
            row.append(int(step.token_id))
            if not step.accepted:
                break
            accepted_child = _selected_child
            if int(step.token_id) != int(drafts[accepted_child]):
                break
            current_parent = accepted_child
            accepted_row = int(current_parent)
            accepted_path.append(int(current_parent))
        out_rows.append(row[:int(max_spec_len) + 1])
        accepted_rows.append(int(accepted_row))
        accepted_lens.append(int(len(accepted_path)))
        accepted_node_paths.append([int(x) for x in accepted_path])
        accepted_token_rows.append([int(drafts[x]) for x in accepted_path])
        final_root = int(accepted_path[0]) if accepted_path else None
        sample_log_rows.append({
            'event': 'tree_sample_accept',
            'policy': 'canonical_multidraft',
            'req_index': int(req_i),
            'node_count': int(node_count),
            'accepted_len': int(len(accepted_path)),
            'accepted_final_row': int(accepted_row),
            'accepted_node_ids': [int(x) for x in accepted_path],
            'accepted_root': final_root,
            'emitted_tokens': [int(x) for x in row[:int(max_spec_len) + 1]],
            'draft_token_ids': [int(x) for x in drafts],
            'committer_step_trace': step_trace_rows,
        })
        start += node_count

    # FR13_COMMIT_BATCH_OUTPUT (default OFF => legacy per-element writes; ON =>
    # ONE host-build + ONE H2D copy). The legacy double loop does
    # output_token_ids[req_i, pos] = int(token_id) per element -- each scalar->
    # CUDA-tensor write is a kernel launch + implicit sync (~B*len syncs/step).
    # The greedy committer was already batched (FR13_EAGER_PACK); the multidraft
    # committer was NOT. Batched build is BYTE-IDENTICAL (same values, same -1
    # padding). Phase-3 committer-decomposition target (surrounding ~80ms).
    if __import__('os').environ.get('FR13_COMMIT_BATCH_OUTPUT', '0') == '1':
        _ot_cols = int(output_token_ids.size(1))
        _ot_host = [
            [int(t) for t in row[:_ot_cols]]
            + [-1] * (_ot_cols - len(row[:_ot_cols]))
            for row in out_rows
        ]
        output_token_ids.fill_(-1)
        if _ot_host:
            output_token_ids[: len(_ot_host), :_ot_cols].copy_(
                torch.tensor(
                    _ot_host, dtype=output_token_ids.dtype,
                    device=output_token_ids.device,
                )
            )
    else:
        output_token_ids.fill_(-1)
        for req_i, row in enumerate(out_rows):
            for pos, token_id in enumerate(row):
                output_token_ids[req_i, pos] = int(token_id)
    accepted_tree_rows.copy_(
        torch.tensor(accepted_rows, dtype=accepted_tree_rows.dtype,
                     device=accepted_tree_rows.device)
    )
    globals()['_LUMO_TREE_LAST_ACCEPTED_ROWS_KERNEL'] = [int(x) for x in accepted_rows]
    globals()['_LUMO_TREE_LAST_ACCEPTED_LENS_KERNEL'] = [int(x) for x in accepted_lens]
    globals()['_LUMO_TREE_LAST_ACCEPTED_NODE_PATHS_KERNEL'] = [
        [int(x) for x in row] for row in accepted_node_paths
    ]
    try:
        from vllm.model_executor.layers.mamba import gdn_linear_attn as _lumo_tree_commit_gdn
        accepted_gdn_node_paths = []
        accepted_gdn_rows = []
        for _accepted_path, _accepted_len, _accepted_row in zip(
            accepted_node_paths, accepted_lens, accepted_rows
        ):
            _gdn_path = [
                int(_node_id) + 1
                for _node_id in _accepted_path[: int(_accepted_len)]
            ]
            accepted_gdn_node_paths.append(_gdn_path)
            accepted_gdn_rows.append(
                int(_gdn_path[-1]) if _gdn_path else 0
            )
        _accepted_path_buf = getattr(
            _lumo_tree_commit_gdn, "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR", None
        )
        _accepted_lens_buf = getattr(
            _lumo_tree_commit_gdn, "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR", None
        )
        if _accepted_path_buf is None or _accepted_lens_buf is None:
            raise RuntimeError("missing_accepted_path_device_tensor")
        if int(_accepted_path_buf.size(0)) < len(accepted_gdn_node_paths):
            raise RuntimeError("accepted_path_device_tensor_batch_too_small")
        _accepted_path_cols = int(_accepted_path_buf.size(1))
        _accepted_path_rows = []
        for _accepted_path in accepted_gdn_node_paths:
            _row = [0 for _ in range(_accepted_path_cols)]
            for _pos, _node_id in enumerate(_accepted_path[:_accepted_path_cols]):
                _row[_pos] = int(_node_id)
            _accepted_path_rows.append(_row)
        if _accepted_path_rows:
            _accepted_path_buf[: len(_accepted_path_rows), :_accepted_path_cols].copy_(
                torch.tensor(
                    _accepted_path_rows,
                    dtype=_accepted_path_buf.dtype,
                    device=_accepted_path_buf.device,
                )
            )
            _accepted_lens_buf[: len(accepted_lens)].copy_(
                torch.tensor(
                    accepted_lens,
                    dtype=_accepted_lens_buf.dtype,
                    device=_accepted_lens_buf.device,
                )
            )
        # FR13_TREE_SAMPLE_ROW freshness publish (FIX-A1): sampled-committer
        # twin of the greedy publish above (see there for the contract).
        _lumo_tree_commit_gdn._LUMO_FA_TREE_COMMIT_NROWS = len(
            accepted_gdn_node_paths
        )
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_ROWS = [
            int(x) for x in accepted_gdn_rows
        ]
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_LENS = [
            int(x) for x in accepted_lens
        ]
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_NODE_PATHS = [
            [int(x) for x in row] for row in accepted_gdn_node_paths
        ]
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS = [
            [int(x) for x in row] for row in accepted_token_rows
        ]
        if True:
            # FR13_TREE_REQKEY: also publish accepted paths keyed by request
            # id (see the greedy committer for the rationale). The
            # model-runner pre-forward rewrite consumes this dict.
            _fr13_row_req_ids = getattr(
                _lumo_tree_commit_gdn, '_LUMO_FA_SAMPLER_ROW_REQ_IDS', None
            )
            if _fr13_row_req_ids is None:
                raise RuntimeError(
                    'FR13_TREE_REQKEY missing sampler-row request ids'
                )
            if len(_fr13_row_req_ids) < len(accepted_gdn_node_paths):
                raise RuntimeError(
                    'FR13_TREE_REQKEY sampler-row request ids shorter than '
                    'committer rows: '
                    f'{len(_fr13_row_req_ids)} < {len(accepted_gdn_node_paths)}'
                )
            _fr13_by_req = getattr(
                _lumo_tree_commit_gdn, '_LUMO_FA_TREE_ACCEPT_BY_REQ', None
            )
            if _fr13_by_req is None:
                _fr13_by_req = {}
                _lumo_tree_commit_gdn._LUMO_FA_TREE_ACCEPT_BY_REQ = _fr13_by_req
            # dbg16 CORRECTION (refutes the dbg13 spec-keying): committer rows
            # cover ALL metadata rows (num_draft_tokens is built over num_reqs
            # with zeros -- zero-count decode rows and mid-prefill rows both
            # get rows), so the FULL-BATCH sampler list IS the right key
            # space, exact length. The pb-specific hazard is different: a
            # by_req entry for a row that never ran a TREE verify poisons the
            # chain packer (variant-B ring lookups have no rows for it) -- so
            # publish entries ONLY for rows with a real tree (count > 0).
            _fr13_commit_key_ids = _fr13_row_req_ids
            if len(_fr13_commit_key_ids) != len(accepted_gdn_node_paths):
                raise RuntimeError(
                    'FR13_TREE_REQKEY: sampler-row id list does not match '
                    'committer rows: sampler_ids='
                    + repr(_fr13_commit_key_ids)
                    + ' committer_rows='
                    + str(len(accepted_gdn_node_paths))
                )
            for _fr13_i in range(len(accepted_gdn_node_paths)):
                if int(num_draft_tokens[_fr13_i]) <= 0:
                    continue
                _fr13_by_req[str(_fr13_commit_key_ids[_fr13_i])] = (
                    [int(_x) for _x in accepted_gdn_node_paths[_fr13_i]],
                    int(accepted_lens[_fr13_i]),
                )
        if True:
            # FR13_REPLAY_ROUTE durable-state publish (Option 1, committer
            # publish site) -- sampled committer twin of the greedy block;
            # see the greedy committer for the full rationale.
            from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
                launch_tree_gdn_replay as _fr13_replay_launch,
            )
            _fr13_replay_layers = getattr(
                _lumo_tree_commit_gdn, '_FR13_REPLAY_LAYERS', None
            )
            if not _fr13_replay_layers:
                raise RuntimeError(
                    'FR13_REPLAY_ROUTE: no registered GDN replay layers'
                )
            _fr13_row_req_ids = getattr(
                _lumo_tree_commit_gdn, '_LUMO_FA_SAMPLER_ROW_REQ_IDS', None
            )
            _fr13_spec_req_ids = getattr(
                _lumo_tree_commit_gdn, '_LUMO_FA_SPEC_ROW_REQ_IDS', None
            )
            if _fr13_row_req_ids is None or _fr13_spec_req_ids is None:
                raise RuntimeError(
                    'FR13_REPLAY_ROUTE: missing request-keyed replay row ids'
                )
            if len(_fr13_row_req_ids) < len(accepted_gdn_node_paths):
                raise RuntimeError(
                    'FR13_REPLAY_ROUTE sampler-row request ids shorter than '
                    'committer rows: '
                    f'{len(_fr13_row_req_ids)} < {len(accepted_gdn_node_paths)}'
                )
            # dbg12 fix: committer-row index by SPEC-row position (the same
            # exact-length-checked list the by_req publish uses above), NOT
            # the full-batch sampler position.
            _fr13_idx_by_req = {
                str(_fr13_rid): int(_fr13_i)
                for _fr13_i, _fr13_rid in enumerate(_fr13_commit_key_ids)
            }
            _fr13_replay_gdn_node_paths = []
            _fr13_replay_lens = []
            _fr13_missing_replay_req_ids = []
            for _fr13_rid in _fr13_spec_req_ids:
                _fr13_src_i = _fr13_idx_by_req.get(str(_fr13_rid))
                if _fr13_src_i is None or _fr13_src_i >= len(accepted_gdn_node_paths):
                    _fr13_missing_replay_req_ids.append(str(_fr13_rid))
                    continue
                _fr13_replay_gdn_node_paths.append(
                    [int(_x) for _x in accepted_gdn_node_paths[_fr13_src_i]]
                )
                _fr13_replay_lens.append(int(accepted_lens[_fr13_src_i]))
            if _fr13_missing_replay_req_ids:
                raise RuntimeError(
                    'FR13_REPLAY_ROUTE: missing committer rows for staged '
                    'spec req ids: ' + repr(_fr13_missing_replay_req_ids[:8])
                )
            _fr13_replay_rows = len(_fr13_replay_gdn_node_paths)
            _fr13_bnd_on = _lumo_tree_commit_gdn._fr13_boundary_on()
            # FR13 STATELESS-TREE (default all-OFF -> byte-identical): one lifecycle,
            # fail loud on partial enable. (Sampled-committer twin; live @ temp>0.)
            # rejection_sampler.py namespace has no bare `os` -> import locally.
            import os
            _fr13_runrow_commit = (
                os.environ.get("FR13_APC_COMMIT_TO_RUNNING_ROW", "1") == "1"
            )
            _fr13_runrow_init = (
                os.environ.get("FR13_TREE_RUNROW_INIT", "1") == "1"
            )
            # BURN DELETED 2026-07-27 (was FR13_APC_BURN_NODE_BANK, baked OFF since
            # 2026-07-22: red-team wf_16247424-fb2 + live SWE proved the spec-col
            # burn redundant for the served path — every served reader is col-0
            # under RUNROW_INIT=1). The legacy runrow=0 path, where burn was
            # load-bearing, is RETIRED: commit/init must both be 1 (fail loud).
            if _fr13_runrow_commit != _fr13_runrow_init:
                raise RuntimeError(
                    "FR13 STATELESS-TREE: COMMIT_TO_RUNNING_ROW/TREE_RUNROW_INIT "
                    "must be set together (got commit=%r init=%r)" % (
                        _fr13_runrow_commit, _fr13_runrow_init,
                    )
                )
            if not _fr13_runrow_commit or not _fr13_runrow_init:
                raise RuntimeError(
                    "FR13 STATELESS-TREE: the legacy non-stateless path "
                    "(COMMIT_TO_RUNNING_ROW/TREE_RUNROW_INIT=0) was RETIRED "
                    "2026-07-27 with the burn deletion (it read col>=1 rows "
                    "and depended on the burn; got commit=%r init=%r)" % (
                        _fr13_runrow_commit,
                        _fr13_runrow_init,
                    )
                )
            # STATELESS-TREE conv committer (post-accept; gated -> no-op when OFF).
            _fr13_conv_commit_to_col0(
                _fr13_replay_layers,
                _accepted_path_buf,
                _accepted_lens_buf,
                _fr13_replay_rows,
                _fr13_runrow_commit,
            )
            if _fr13_replay_rows:
                if int(_accepted_path_buf.size(0)) < _fr13_replay_rows:
                    raise RuntimeError(
                        'FR13_REPLAY_ROUTE accepted-path buffer too small '
                        'for compact staged replay rows'
                    )
                _fr13_replay_path_rows = []
                for _fr13_path in _fr13_replay_gdn_node_paths:
                    _fr13_row = [0 for _ in range(_accepted_path_cols)]
                    for _fr13_pos, _fr13_node_id in enumerate(
                        _fr13_path[:_accepted_path_cols]
                    ):
                        _fr13_row[_fr13_pos] = int(_fr13_node_id)
                    _fr13_replay_path_rows.append(_fr13_row)
                _accepted_path_buf[
                    :_fr13_replay_rows, :_accepted_path_cols
                ].copy_(
                    torch.tensor(
                        _fr13_replay_path_rows,
                        dtype=_accepted_path_buf.dtype,
                        device=_accepted_path_buf.device,
                    )
                )
                _accepted_lens_buf[:_fr13_replay_rows].copy_(
                    torch.tensor(
                        _fr13_replay_lens,
                        dtype=_accepted_lens_buf.dtype,
                        device=_accepted_lens_buf.device,
                    )
                )
                # FR13_REPLAY_DURABLE_AB (PRIME lead, observe-only): canonical/
                # sampled committer variant. Default OFF -> no effect.
                _fr13_rdab_on = _fr13_replay_durable_ab_enabled()
                if _fr13_bnd_on or _fr13_rdab_on:
                    # Shared boundary event counter: increment ONCE per commit
                    # event; every tap record between commit k and commit k+1
                    # then carries event=k (tap A of commit k carries k).
                    _lumo_tree_commit_gdn._FR13_BOUNDARY_EVENT = int(getattr(
                        _lumo_tree_commit_gdn, '_FR13_BOUNDARY_EVENT', 0
                    )) + 1
                # FR13_SAMPLED_REPLAY_BATCHED (default OFF => byte-identical per-layer loop):
                # the sampled/deployed committer replays GDN state via ~48 per-layer
                # launch_tree_gdn_replay calls (measured ~72ms/step, 81% of the committer).
                # The GREEDY committer already batches this into ONE
                # launch_tree_gdn_replay_all_layers over the GLOBAL _FR13_EAGER_PACK_STACKS
                # (patcher ~8990-9108). Port that batched dispatch here (same gate: needs
                # stacks + NOT boundary/durable/APC-publish, which require the per-layer
                # publish). Semantics-preserving sibling kernel => byte-identical replay.
                # When active, the per-layer loop below iterates EMPTY.
                _fr13_sbr_stacks = getattr(
                    _lumo_tree_commit_gdn, '_FR13_EAGER_PACK_STACKS', None
                )
                # SIDECAR-armed like multistream: EngineCore curation DROPS FR13_* env
                # (env-only read here was vacuous in deployment, same class as the
                # multistream B=4 root cause). /logs/*.arm is worker-env-drop-proof.
                _fr13_sbr_active = (
                    (__import__('os').environ.get('FR13_SAMPLED_REPLAY_BATCHED', '0') == '1'
                     or __import__('os').environ.get('FR13_COMMITTER_NATIVE_BATCHED', '0') == '1'
                     or __import__('os').environ.get('FR13_COMMITTER_GRAPH', '0') == '1'
                     or __import__('os').path.exists('/logs/fr13_committer_batched.arm')
                     or __import__('os').path.exists('/tmp/fr13_committer_batched.arm')
                     or __import__('os').path.exists('/logs/fr13_committer_graph.arm')
                     or __import__('os').path.exists('/tmp/fr13_committer_graph.arm'))
                    and _fr13_sbr_stacks is not None
                    and int(_fr13_sbr_stacks.get('num_layers', 0)) > 0
                    and _fr13_replay_rows > 0
                    and not _fr13_bnd_on
                    and not _fr13_rdab_on
                )
                if _fr13_sbr_active:
                    _ep_order = list(_fr13_sbr_stacks['layer_order'])
                    # boot-42: layer_order is now PARTITIONED (independent
                    # first, post-full-attn adjacent last) — validate as a
                    # SET; row alignment is carried by the stacks themselves.
                    if sorted(_ep_order) != sorted(_fr13_replay_layers):
                        raise RuntimeError(
                            'FR13_SAMPLED_REPLAY_BATCHED: stacked layer order != '
                            'registered replay layers'
                        )
                    _ep_flag_rows = _fr13_sbr_stacks['flags'].detach().cpu().tolist()
                    _ep_banks = []
                    for _ep_i, _ep_prefix in enumerate(_ep_order):
                        _ep_layer = _fr13_replay_layers[_ep_prefix]
                        _ep_bank = getattr(_ep_layer, '_fr13_replay_ssm_state', None)
                        if _ep_bank is None or int(_ep_flag_rows[_ep_i][0]) != 1:
                            raise RuntimeError(
                                'FR13_SAMPLED_REPLAY_BATCHED: stale/missing scan '
                                'flags for layer ' + str(_ep_prefix)
                            )
                        if int(_ep_flag_rows[_ep_i][1]) != _fr13_replay_rows:
                            raise RuntimeError(
                                'FR13_SAMPLED_REPLAY_BATCHED: committer rows '
                                + str(_fr13_replay_rows) + ' != staged '
                                + str(int(_ep_flag_rows[_ep_i][1]))
                                + ' for layer ' + str(_ep_prefix)
                            )
                        _ep_banks.append(_ep_bank)
                    from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
                        build_replay_bank_pointer_table as _ep_build_tbl,
                        launch_tree_gdn_replay_all_layers as _ep_launch_all,
                    )
                    _ep_tbl = getattr(
                        _lumo_tree_commit_gdn, '_FR13_EAGER_PACK_BANK_TBL', None
                    )
                    _ep_ptrs_now = [int(_b.data_ptr()) for _b in _ep_banks]
                    if _ep_tbl is None:
                        _ep_ptr_list, _ep_bank_shape, _ep_bank_stride = (
                            _ep_build_tbl(_ep_banks)
                        )
                        _ep_off16_dev = torch.tensor(
                            [(_p - _ep_ptr_list[0]) // 16 for _p in _ep_ptr_list],
                            dtype=torch.int64, device=_ep_banks[0].device,
                        )
                        _ep_tbl = (
                            _ep_ptr_list, _ep_off16_dev, _ep_bank_shape,
                            _ep_bank_stride, _ep_banks[0],
                        )
                        _lumo_tree_commit_gdn._FR13_EAGER_PACK_BANK_TBL = _ep_tbl
                    elif _ep_tbl[0] != _ep_ptrs_now:
                        raise RuntimeError(
                            'FR13_SAMPLED_REPLAY_BATCHED: GDN bank data_ptr '
                            'changed since pointer-table build'
                        )
                    _ep_launch_all(
                        bank_anchor=_ep_tbl[4],
                        bank_off16=_ep_tbl[1],
                        bank_shape=_ep_tbl[2],
                        bank_stride=_ep_tbl[3],
                        spec_state_indices=_fr13_sbr_stacks['spec_idx'],
                        prev_lens=_fr13_sbr_stacks['prev_lens'],
                        accepted_paths=_accepted_path_buf,
                        accepted_lens=_accepted_lens_buf,
                        k_rings=_fr13_sbr_stacks['ring_k'],
                        v_rings=_fr13_sbr_stacks['ring_v'],
                        a_rings=_fr13_sbr_stacks['ring_a'],
                        b_rings=_fr13_sbr_stacks['ring_b'],
                        A_logs=_fr13_sbr_stacks['A_log'],
                        dt_biases=_fr13_sbr_stacks['dt_bias'],
                        num_layers=len(_ep_order),
                        num_spec_decodes=_fr13_replay_rows,
                        output_scale=float(_fr13_sbr_stacks['output_scale']),
                        use_qk_l2norm_in_kernel=True,
                        runrow_commit=_fr13_runrow_commit,
                        runrow_init=_fr13_runrow_init,
                        burn_node_bank=False,  # burn DELETED 2026-07-27
                        banks_list=_ep_banks,
                    )
                    _fr13_sbr_stacks['flags'][:, 0].fill_(0)
                    # FR13_CONV_PREGATHER trigger (2026-07-24): one launch
                    # stages every layer's conv col0 (post-commit truth) for
                    # next step's NPR read; host req-id token guards freshness
                    # (composition-change steps fall back to legacy gather).
                    if getattr(_lumo_tree_commit_gdn, '_FR13_CONV_PREGATHER_ON', False):
                        try:
                            from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
                                launch_conv_col0_pregather as _fr13_cpg,
                            )
                            _fr13_cpg_banks = [
                                _fr13_replay_layers[_n]._fr13_replay_conv_state
                                for _n in _ep_order
                            ]
                            # FR13_CPG_ROWID_TOKEN: composite (req_ids, col0
                            # page-ids) freshness token. If the host col0
                            # publish is unavailable this step, REFUSE to
                            # stage (consume then takes the legacy gather) --
                            # a req-id-only token cannot see a col0 page
                            # slide/realloc under stable batch composition.
                            # SPEC-ROW (rid, col0) pairs, NOT full-batch col0:
                            # prefill rows' running blocks race every chunk
                            # and would poison the all-or-nothing token.
                            _fr13_cpg_pairs = getattr(
                                _lumo_tree_commit_gdn,
                                '_LUMO_FA_SPEC_ROW_CONV_COL0', None)
                            _fr13_cpg_seq = getattr(
                                _lumo_tree_commit_gdn,
                                '_LUMO_FA_STEP_SEQ', None)
                            if (all(_b is not None for _b in _fr13_cpg_banks)
                                    and _fr13_cpg_pairs is not None
                                    and _fr13_cpg_seq is not None):
                                _fr13_cpg(
                                    conv_banks=_fr13_cpg_banks,
                                    ssi_stack=_fr13_sbr_stacks['spec_idx'][:, :, 0].contiguous(),
                                    num_spec_decodes=_fr13_replay_rows,
                                    req_ids_token=(
                                        _fr13_cpg_pairs,
                                        int(_fr13_cpg_seq),
                                    ),
                                )
                        except Exception as _fr13_cpg_exc:
                            raise RuntimeError(
                                'FR13_CONV_PREGATHER trigger failed: '
                                + repr(_fr13_cpg_exc)
                            ) from _fr13_cpg_exc
                # FR13_REPLAY_MULTISTREAM (default OFF => serial loop below is byte-identical):
                # overlap the WRITE-INDEPENDENT per-layer replays across N CUDA streams to hide the
                # latency-bound per-layer kernel (measured 1.386ms/layer = 14x its ~0.1ms bandwidth
                # floor). Each layer writes only its own ssm_bank => order-independent => byte-safe.
                # Distinct from refuted batched-fused (strided cross-bank). See
                # FR13_REPLAY_MULTISTREAM_DESIGN.md. Gate excludes bnd/rdab (serial order) + sbr.
                # FR13_REPLAY_MULTISTREAM enable: env OR sidecar. The EngineCore worker curation DROPS
                # FR13_* env (=> os.environ reads '0' at worker runtime), so read the /logs/*.arm sidecar
                # the launcher wrote (worker-env-drop-proof, same pattern as _fr13_committer_native_on).
                # This env-only read was the B=4 VACUOUS root cause.
                _fr13_ms_enable = (
                    __import__('os').environ.get(
                        'FR13_REPLAY_MULTISTREAM', '0') == '1'
                    or __import__('os').path.exists(
                        '/logs/fr13_replay_multistream.arm')
                    or __import__('os').path.exists(
                        '/tmp/fr13_replay_multistream.arm')
                )
                # FR13_COMMIT_OVERLAP v2 (replay half): reuse the multistream
                # machinery but DEFER the default-stream join to the next
                # forward's fence (gpu_model_runner begin-inject waits the
                # stashed events), so the ~14ms replay overlaps the drafter
                # instead of blocking it. SIDECAR-armed (worker env drop),
                # capture-guarded via the shared _fr13_ms_on gate below.
                _fr13_ov2_enable = (
                    __import__('os').path.exists('/logs/fr13_commit_overlap.arm')
                    or __import__('os').path.exists('/tmp/fr13_commit_overlap.arm')
                )
                _fr13_ms_enable = _fr13_ms_enable or _fr13_ov2_enable
                _fr13_ms_on = (
                    _fr13_ms_enable
                    and not _fr13_bnd_on and not _fr13_rdab_on and not _fr13_sbr_active
                    and int(_fr13_replay_rows) > 0
                    # CUDA-graph capture cannot tolerate cross-stream events / stream-switch
                    # (poisons the capture -> EngineCore init FAILS: observed ms_strm die at
                    # 'Capturing CUDA graphs (decode, FULL)'). The accepted-path replay is
                    # data-dependent => eager anyway, so fall back to the serial loop during
                    # capture and overlap only on real (eager) decode steps.
                    and not (torch.cuda.is_available()
                             and torch.cuda.is_current_stream_capturing())
                )
                if _fr13_ms_on:
                    _fr13_ms_pool = getattr(
                        _lumo_tree_commit_gdn, '_FR13_REPLAY_STREAMS', None
                    )
                    if _fr13_ms_pool is None:
                        # N streams: env (dropped in worker) OR sidecar content OR default 4.
                        _fr13_ms_n_s = __import__('os').environ.get(
                            'FR13_REPLAY_MULTISTREAM_N', '')
                        if not _fr13_ms_n_s:
                            try:
                                _fr13_ms_n_s = open(
                                    '/logs/fr13_replay_multistream.arm'
                                ).read().strip()
                            except Exception:
                                _fr13_ms_n_s = ''
                        _fr13_ms_n = (
                            int(_fr13_ms_n_s) if _fr13_ms_n_s.isdigit() else 4)
                        _fr13_ms_pool = [
                            torch.cuda.Stream() for _ in range(_fr13_ms_n)
                        ]
                        _lumo_tree_commit_gdn._FR13_REPLAY_STREAMS = _fr13_ms_pool
                    if not getattr(
                        _lumo_tree_commit_gdn, '_FR13_MS_ANNOUNCED', False
                    ):
                        __import__('sys').stderr.write(
                            '[FR13_REPLAY_MULTISTREAM] ENGAGED N='
                            + str(len(_fr13_ms_pool)) + ' streams over '
                            + str(len(_fr13_replay_layers)) + ' layers\n')
                        _lumo_tree_commit_gdn._FR13_MS_ANNOUNCED = True
                    _fr13_ms_default = torch.cuda.current_stream()
                    # scan (default stream) wrote ssm_bank + rings; pool streams MUST wait on this
                    # event before reading them (else stale-state garble).
                    _fr13_ms_ready = torch.cuda.Event()
                    _fr13_ms_ready.record(_fr13_ms_default)
                    _fr13_ms_used = []
                    for _fr13_ms_i, _fr13_prefix in enumerate(
                        sorted(_fr13_replay_layers)
                    ):
                        _fr13_layer = _fr13_replay_layers[_fr13_prefix]
                        _fr13_flags = getattr(
                            _fr13_layer, '_fr13_replay_flags', None)
                        _fr13_ssm_bank = getattr(
                            _fr13_layer, '_fr13_replay_ssm_state', None)
                        if _fr13_flags is None or _fr13_ssm_bank is None:
                            raise RuntimeError(
                                'FR13_REPLAY_MULTISTREAM: missing scan-time '
                                'staging for layer ' + str(_fr13_prefix))
                        if _fr13_ov2_enable:
                            # overlap mode: the 2 blocking .item() validations
                            # per layer (~24ms/step host dispatch tax) become
                            # ONE device-side async assert -- fail-loud is
                            # preserved (CUDA assert kills the context), zero
                            # host syncs, so the drafter starts immediately.
                            torch._assert_async(
                                (
                                    (_fr13_flags[0] == 1)
                                    & (_fr13_flags[1] == _fr13_replay_rows)
                                ),
                                'FR13_REPLAY_MULTISTREAM overlap: stale '
                                'staging or row mismatch',
                            )
                        elif int(_fr13_flags[0].item()) != 1:
                            raise RuntimeError(
                                'FR13_REPLAY_MULTISTREAM: stale or missing '
                                'scan-time staging for layer '
                                + str(_fr13_prefix))
                        elif int(_fr13_flags[1].item()) != _fr13_replay_rows:
                            raise RuntimeError(
                                'FR13_REPLAY_MULTISTREAM: committer rows '
                                + str(_fr13_replay_rows) + ' != staged '
                                + str(int(_fr13_flags[1].item()))
                                + ' for layer ' + str(_fr13_prefix))
                        _fr13_ms_s = _fr13_ms_pool[
                            _fr13_ms_i % len(_fr13_ms_pool)]
                        if _fr13_ms_s not in _fr13_ms_used:
                            _fr13_ms_used.append(_fr13_ms_s)
                        _fr13_ms_s.wait_event(_fr13_ms_ready)
                        with torch.cuda.stream(_fr13_ms_s):
                            _fr13_replay_launch(
                                state_bank=_fr13_ssm_bank,
                                spec_state_indices=(
                                    _fr13_layer._fr13_replay_spec_idx),
                                prev_lens=_fr13_layer._fr13_replay_prev_lens,
                                accepted_paths=_accepted_path_buf,
                                accepted_lens=_accepted_lens_buf,
                                k_ring=_fr13_layer._fr13_replay_ring_k,
                                v_ring=_fr13_layer._fr13_replay_ring_v,
                                a_ring=_fr13_layer._fr13_replay_ring_a,
                                b_ring=_fr13_layer._fr13_replay_ring_b,
                                A_log=_fr13_layer.A_log,
                                dt_bias=_fr13_layer.dt_bias,
                                num_spec_decodes=_fr13_replay_rows,
                                output_scale=float(
                                    _fr13_layer._fr13_replay_output_scale),
                                use_qk_l2norm_in_kernel=True,
                                runrow_commit=_fr13_runrow_commit,
                                runrow_init=_fr13_runrow_init,
                                burn_node_bank=False,  # burn DELETED 2026-07-27
                            )
                            _fr13_flags[0].fill_(0)
                    if _fr13_ov2_enable:
                        # FR13_COMMIT_OVERLAP v2: DEFER the join -- stash one
                        # completion event per used pool stream on the torch
                        # module (cross-module reachable); the next forward's
                        # fence (gpu_model_runner begin-inject) waits them
                        # before the verify reads col0/rings. The drafter
                        # proceeds immediately on the default stream.
                        _fr13_ov2_evts = []
                        for _fr13_ms_s in _fr13_ms_used:
                            _fr13_ms_je = torch.cuda.Event()
                            _fr13_ms_je.record(_fr13_ms_s)
                            _fr13_ov2_evts.append(_fr13_ms_je)
                        torch._fr13_ov2_evts = _fr13_ov2_evts
                    else:
                        # join: default stream waits for every used pool stream
                        for _fr13_ms_s in _fr13_ms_used:
                            _fr13_ms_je = torch.cuda.Event()
                            _fr13_ms_je.record(_fr13_ms_s)
                            _fr13_ms_default.wait_event(_fr13_ms_je)
                _fr13_lo_prefixes = (
                    [] if (
                        _fr13_sbr_active or _fr13_ms_on
                    )
                    else sorted(_fr13_replay_layers)
                )
                # FR13_COMMITTER_LAYOUT_ONCE (default OFF): each loop iteration validates
                # its layer's staging flags via .item() = 2 blocking D2H syncs/layer
                # (~2*(L-1) redundant host stalls). When ON, batch-read ALL layers' flags
                # in ONE .tolist() up front and validate from the host list -- identical
                # values, identical raises, only the sync count changes. The per-layer
                # replay below is untouched, so the committed GDN state is bit-identical.
                _fr13_lo_on = __import__('os').environ.get(
                    'FR13_COMMITTER_LAYOUT_ONCE', '0'
                ) == '1'
                _fr13_lo_flags_host = None
                if _fr13_lo_on and _fr13_lo_prefixes:
                    if any(
                        getattr(_fr13_replay_layers[_p], '_fr13_replay_flags', None)
                        is None
                        or getattr(
                            _fr13_replay_layers[_p], '_fr13_replay_ssm_state', None
                        ) is None
                        for _p in _fr13_lo_prefixes
                    ):
                        raise RuntimeError(
                            'FR13_REPLAY_ROUTE: stale or missing scan-time staging '
                            '(layout-once precheck)'
                        )
                    _fr13_lo_flags_host = torch.stack([
                        _fr13_replay_layers[_p]._fr13_replay_flags[:2]
                        for _p in _fr13_lo_prefixes
                    ]).to('cpu').tolist()
                    if not getattr(
                        _lumo_tree_commit_gdn, '_FR13_LO_ANNOUNCED', False
                    ):
                        __import__('sys').stderr.write(
                            '[FR13_COMMITTER_LAYOUT_ONCE ENGAGED] batched '
                            + str(len(_fr13_lo_prefixes))
                            + '-layer flag validation (1 D2H vs 2/layer)\n'
                        )
                        _lumo_tree_commit_gdn._FR13_LO_ANNOUNCED = True
                for _fr13_i, _fr13_prefix in enumerate(_fr13_lo_prefixes):
                    _fr13_layer = _fr13_replay_layers[_fr13_prefix]
                    _fr13_flags = getattr(
                        _fr13_layer, '_fr13_replay_flags', None
                    )
                    _fr13_ssm_bank = getattr(
                        _fr13_layer, '_fr13_replay_ssm_state', None
                    )
                    if _fr13_lo_on:
                        if (
                            _fr13_flags is None
                            or _fr13_ssm_bank is None
                            or int(_fr13_lo_flags_host[_fr13_i][0]) != 1
                        ):
                            raise RuntimeError(
                                'FR13_REPLAY_ROUTE: stale or missing scan-time '
                                'staging for layer ' + str(_fr13_prefix)
                            )
                        if int(_fr13_lo_flags_host[_fr13_i][1]) != _fr13_replay_rows:
                            raise RuntimeError(
                                'FR13_REPLAY_ROUTE: committer rows '
                                + str(_fr13_replay_rows)
                                + ' != staged spec decodes '
                                + str(int(_fr13_lo_flags_host[_fr13_i][1]))
                                + ' for layer ' + str(_fr13_prefix)
                            )
                    else:
                        if (
                            _fr13_flags is None
                            or _fr13_ssm_bank is None
                            or int(_fr13_flags[0].item()) != 1
                        ):
                            raise RuntimeError(
                                'FR13_REPLAY_ROUTE: stale or missing scan-time '
                                'staging for layer ' + str(_fr13_prefix)
                            )
                        if int(_fr13_flags[1].item()) != _fr13_replay_rows:
                            raise RuntimeError(
                                'FR13_REPLAY_ROUTE: committer rows '
                                + str(_fr13_replay_rows)
                                + ' != staged spec decodes '
                                + str(int(_fr13_flags[1].item()))
                                + ' for layer ' + str(_fr13_prefix)
                            )
                    _fr13_bnd_pre = None
                    _fr13_bnd_layer_on = (
                        _fr13_bnd_on
                        and _lumo_tree_commit_gdn._fr13_boundary_layer_match(
                            _fr13_prefix
                        )
                    )
                    if _fr13_bnd_layer_on:
                        _fr13_bnd_pre = _fr13_boundary_replay_pre(
                            _lumo_tree_commit_gdn, _fr13_layer,
                            _fr13_ssm_bank, _fr13_replay_rows,
                        )
                    _fr13_replay_launch(
                        state_bank=_fr13_ssm_bank,
                        spec_state_indices=_fr13_layer._fr13_replay_spec_idx,
                        prev_lens=_fr13_layer._fr13_replay_prev_lens,
                        accepted_paths=_accepted_path_buf,
                        accepted_lens=_accepted_lens_buf,
                        k_ring=_fr13_layer._fr13_replay_ring_k,
                        v_ring=_fr13_layer._fr13_replay_ring_v,
                        a_ring=_fr13_layer._fr13_replay_ring_a,
                        b_ring=_fr13_layer._fr13_replay_ring_b,
                        A_log=_fr13_layer.A_log,
                        dt_bias=_fr13_layer.dt_bias,
                        num_spec_decodes=_fr13_replay_rows,
                        output_scale=float(
                            _fr13_layer._fr13_replay_output_scale
                        ),
                        use_qk_l2norm_in_kernel=True,
                        runrow_commit=_fr13_runrow_commit,
                        runrow_init=_fr13_runrow_init,
                        burn_node_bank=False,  # burn DELETED 2026-07-27
                    )
                    # FR13: vestigial EXACT_SEED committer call REMOVED (twin of the
                    # removal in the eager-pack branch above; same rationale -- never
                    # published, superseded by prefill-capture, cost 48 syncs/step).
                    _fr13_flags[0].fill_(0)
                    if _fr13_bnd_layer_on:
                        _fr13_boundary_replay_post(
                            _lumo_tree_commit_gdn, _fr13_prefix, _fr13_layer,
                            _fr13_ssm_bank, _fr13_replay_rows,
                            _fr13_replay_gdn_node_paths, _fr13_replay_lens,
                            _fr13_bnd_pre,
                        )
                    if _fr13_rdab_on:
                        # OBSERVE-ONLY durable-state A/B (sampled committer):
                        # H_ours vs native MTP sequential recurrent on the SAME
                        # cloned h0 + accepted chain. event_index = shared
                        # per-commit counter (back-loading axis).
                        _fr13_replay_durable_ab(
                            _lumo_tree_commit_gdn, _fr13_prefix, _fr13_layer,
                            _fr13_ssm_bank, _fr13_replay_rows,
                            _fr13_replay_gdn_node_paths, _fr13_replay_lens,
                            int(getattr(
                                _lumo_tree_commit_gdn,
                                '_FR13_BOUNDARY_EVENT', 0,
                            )),
                        )
    except Exception as _fr10_commit_globals_exc:
        if __import__('os').environ.get('FR10_ALLOW_LINEAR_FALLBACK', '0') != '1':
            raise RuntimeError(
                'FR10 tree committer failed to publish accepted rows: '
                + type(_fr10_commit_globals_exc).__name__
                + ':'
                + str(_fr10_commit_globals_exc)
            ) from _fr10_commit_globals_exc
    try:
        import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
        if (
            _fr10_lo.environ.get('FR10_METRICS', '0') == '1'
            or _fr10_lo.environ.get('LUMO_TREE_PATH_LCP_LOG')
        ):
            global _LUMO_TREE_SAMPLE_ACCEPT_FH
            try:
                _LUMO_TREE_SAMPLE_ACCEPT_FH
            except NameError:
                _LUMO_TREE_SAMPLE_ACCEPT_FH = open(
                    _fr10_lo.environ.get('LUMO_TREE_PATH_LCP_LOG',
                                         '/logs/tree_path_lcp_max.jsonl'),
                    'a',
                    buffering=1,
                )
            _now = round(_fr10_lt.time(), 4)
            for _row in sample_log_rows:
                _row = dict(_row)
                _row['ts'] = _now
                _LUMO_TREE_SAMPLE_ACCEPT_FH.write(_fr10_lj.dumps(_row) + chr(10))
    except Exception:
        pass
    return output_token_ids


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    tree_parent_indices: torch.Tensor | None = None,
    tree_token_ids: torch.Tensor | None = None,
    tree_self_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )

    if (
        tree_parent_indices is not None
        and tree_token_ids is not None
        and sampling_metadata.all_greedy
    ):
        accepted_tree_rows = torch.empty(
            (batch_size,), dtype=torch.int32, device=device
        )
        # FR13 committer unification (temp-0): the greedy commit is the
        # POINT-MASS (all_greedy) specialization of the SAME multidraft
        # committer -- it reduces byte-for-byte and rng-free to greedy
        # longest-prefix (scripts/fr13_greedy_pointmass_byte_gate.py 0/4000).
        # FR13_GREEDY_VIA_REJECTION=1 serves the unified path so the separate
        # greedy path-LCP committer becomes dead. FR13_GREEDY_UNIFY_GATE=1
        # additionally dual-runs BOTH on the SAME real trees and records byte
        # mismatches (settles the duplicate-sibling tie the offline gate defers)
        # WITHOUT changing served output/state (old committer runs last + wins).
        # FR13 committer UNIFICATION (temp-0/all_greedy): ALWAYS the point-mass
        # rejection committer. Validated: offline 0/4000 + dup 0/2000 (new==greedy
        # longest-prefix) + VIA-mode live (temp-0 clean, graph mode) + temp-0.6 gate
        # (accept 4.363, 0 fatal). The old path-LCP greedy committer is removed -- it
        # was an eager-only replay path, dual-run-incompatible in graph mode.
        if tree_self_logits is None:
            raise RuntimeError(
                "FR13 greedy unification: all_greedy commit requires tree_self_logits"
            )
        _fr13_gu_gen = getattr(sampling_metadata, "generators", None)
        return _lumo_tree_canonical_multidraft_sample(
            output_token_ids,
            accepted_tree_rows,
            num_draft_tokens,
            draft_token_ids,
            tree_parent_indices,
            target_logits,
            tree_self_logits,
            None,
            bonus_token_ids,
            max_spec_len,
            generators=_fr13_gu_gen,
            all_greedy=True,
        )

    if tree_parent_indices is not None and not sampling_metadata.all_greedy:
        try:
            import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
            if (
                _fr10_lo.environ.get("FR10_METRICS", "0") == "1"
                or _fr10_lo.environ.get("LUMO_TREE_SAMPLER_DEBUG_LOG")
            ):
                global _LUMO_TREE_SAMPLER_BRANCH_FH
                try:
                    _LUMO_TREE_SAMPLER_BRANCH_FH
                except NameError:
                    _LUMO_TREE_SAMPLER_BRANCH_FH = open(
                        _fr10_lo.environ.get(
                            "LUMO_TREE_SAMPLER_DEBUG_LOG",
                        ) or "/logs/tree_sampler_debug.jsonl",
                        "a",
                        buffering=1,
                    )
                _LUMO_TREE_SAMPLER_BRANCH_FH.write(
                    _fr10_lj.dumps({
                        "event": "sampler_branch_enter",
                        "ts": round(_fr10_lt.time(), 4),
                        "batch_size": int(batch_size),
                        "max_spec_len": int(max_spec_len),
                        "has_tree_self_logits": tree_self_logits is not None,
                    }) + chr(10)
                )
        except Exception:
            pass
        if tree_self_logits is None:
            raise RuntimeError("FR10 sampled tree committer missing self logits")
        accepted_tree_rows = torch.empty(
            (batch_size,), dtype=torch.int32, device=device
        )
        # FR13_COMMIT_FULL_GPU_TIMER (diagnostic, default OFF => byte-identical):
        # brackets the WHOLE committer (device multidraft walk + output-row
        # assembly + GDN publish/replay) with cuda events. Compared against
        # FR13_MULTIDRAFT_GPU_TIMER (inner walk only), the DELTA = the surrounding
        # host assembly/publish -- the real decomposition of the committer span.
        # Reliable (globals() accumulator + json every 50; the built-in CFWD timer
        # counter is not firing). Uses synchronize() => diagnostic run only.
        _fr13_cf2 = __import__('os').environ.get('FR13_COMMIT_FULL_GPU_TIMER', '0') == '1'
        if _fr13_cf2:
            _cf2_s = torch.cuda.Event(enable_timing=True)
            _cf2_e = torch.cuda.Event(enable_timing=True)
            _cf2_s.record()
        _fr13_cf2_out = _lumo_tree_canonical_multidraft_sample(
            output_token_ids,
            accepted_tree_rows,
            num_draft_tokens,
            draft_token_ids,
            tree_parent_indices,
            target_logits,
            tree_self_logits,
            draft_probs,
            bonus_token_ids,
            max_spec_len,
            generators=getattr(sampling_metadata, "generators", None),
        )
        if _fr13_cf2:
            _cf2_e.record()
            _cf2_e.synchronize()
            _cf2_g = globals()
            _cf2_g['_FR13_CF2_S'] = _cf2_g.get('_FR13_CF2_S', 0.0) + _cf2_s.elapsed_time(_cf2_e) / 1000.0
            _cf2_g['_FR13_CF2_N'] = _cf2_g.get('_FR13_CF2_N', 0) + 1
            if _cf2_g['_FR13_CF2_N'] % 50 == 0:
                try:
                    import json as _cf2_json
                    _cf2_json.dump(
                        {"gpu_seconds": _cf2_g['_FR13_CF2_S'], "n_spans": _cf2_g['_FR13_CF2_N']},
                        open(__import__('os').environ.get(
                            'FR13_COMMIT_FULL_GPU_TIMER_JSON',
                            '/logs/fr13_commit_full_gpu.json'), 'w'),
                    )
                except Exception:
                    pass
        return _fr13_cf2_out

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Compute probability distribution from target logits.
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Rejection sampling for random sampling requests.
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return output_token_ids


def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Process logits based on sampling metadata.

    This function applies temperature scaling to the logits,
    as well as top-k and top-p. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be processed.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Processed logits if non-greedy sampling is used,
        otherwise returns the original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    return apply_top_k_top_p(logits, top_k, top_p)


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_kernel[(batch_size,)](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
    )
    return expanded_x


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """
    Generates a batch of uniform random samples, with optional seeding
    if available.

    This method creates a tensor of shape `(num_tokens, )` filled
    with uniform random values in the range [0, 1). If `generators` is provided,
    the requests with their own seeds will use the provided `torch.Generator`
    for reproducibility. The samples for the other requests will be generated
    without a seed.

    Args:
        num_tokens: int
            Total number of tokens.
        num_draft_tokens: List[List[int]]
            Number of draft tokens per request.
        generators: Optional[Dict[int, torch.Generator]]
            A dictionary mapping indices in the batch to
            `torch.Generator` objects.
        device: torch.device
            The device on which to allocate the tensor.
    Returns:
        uniform_rand: torch.Tensor
            A tensor of shape `(num_tokens, )` containing uniform
            random values in the range [0, 1).
    """
    # NOTE(woosuk): We deliberately use float64 instead of float32 here
    # because when using float32, there's a non-negligible chance that
    # uniform_prob is sampled to be exact 0.0 as reported in
    # https://github.com/pytorch/pytorch/issues/16706. Using float64
    # mitigates the issue.
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float64,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    inv_q = q.reciprocal()

    recovered_token_ids = torch.empty_like(draft_token_ids)
    BLOCK_SIZE = 8192
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size,
        BLOCK_SIZE,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                target_argmax_id,
            )
            if draft_token_id != target_argmax_id:
                # Reject.
                rejected = True

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            if NO_DRAFT_PROBS:
                draft_prob = 1
            else:
                draft_prob = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
            target_prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
            )
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            # NOTE(woosuk): While the draft probability should never be 0,
            # we check it to avoid NaNs. If it happens to be 0, we reject.
            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                # Accept.
                token_id = draft_token_id
            else:
                # Reject. Use recovered token.
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:  # noqa: SIM108
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    max_val = float("-inf")
    recovered_id = 0
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
            prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=(vocab_mask & (vocab_offset != draft_token_id)),
                other=0.0,
            )
        else:
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            prob = tl.maximum(target_prob - draft_prob, 0.0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        # Local tile reduction
        score = prob * inv_q
        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    tl.store(output_token_ids_ptr + token_idx, recovered_id)
