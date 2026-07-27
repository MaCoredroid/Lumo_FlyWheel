# FR13 boot-29 WARMUP CAPTURE (appended to gpu_model_runner.py by the
# fr10_phase4 patcher — see _patch_gpu_model_runner_warmup_capture).
# Boots 24-28 refuted every live-capture lever (strip hoists, pools, all
# capture modes, exec-lock serialization, hooks): the live window is
# structurally hostile (~80 threads incl. NCCL watchdogs). vLLM's own ~180
# FULL graphs capture fine at warmup in this same process => capture the =2
# graph there too. The refill machinery (uniforms/perm/tls/stls/bonus/
# topology, composition guards) makes the capture moment irrelevant to
# replay correctness.


def _fr13_sg_warmup_capture(self):
    import ast
    import numpy as _np
    import vllm.v1.spec_decode.metadata as _mdmod
    from vllm.v1.sample.metadata import SamplingMetadata as _SM
    from vllm.v1.sample.logits_processor import LogitsProcessors as _LP
    device = self.device
    sc = getattr(self, "speculative_config", None) or getattr(
        self.vllm_config, "speculative_config", None)
    tree_str = getattr(sc, "speculative_token_tree", None)
    choices = ast.literal_eval(tree_str) if isinstance(tree_str, str) else (
        list(tree_str) if tree_str else [])
    n = len(choices)
    if n == 0:
        raise RuntimeError("FR13 warmup-capture: no speculative token tree")
    path_to_idx = {tuple(p): i for i, p in enumerate(choices)}
    parents_template = [path_to_idx.get(tuple(p[:-1]), -1) for p in choices]
    maxB = int(self.scheduler_config.max_num_seqs)

    def _make_md(B):
        ndt = [n] * B
        parents = []
        target = []
        self_idx = []
        ss = 0
        for _r in range(B):
            for node_idx, par in enumerate(parents_template):
                pl = 0 if par < 0 else par + 1
                parents.append(par)
                target.append(ss + pl)
                self_idx.append(ss + node_idx + 1)
            ss += n + 1
        cu_d = _np.cumsum(ndt, dtype=_np.int32)
        cu_s = _np.cumsum([n + 1] * B, dtype=_np.int32)
        md = _mdmod.SpecDecodeMetadata(
            draft_token_ids=torch.zeros(
                n * B, dtype=torch.int32, device=device),
            num_draft_tokens=list(ndt),
            cu_num_draft_tokens=torch.from_numpy(cu_d).to(device),
            cu_num_sampled_tokens=torch.from_numpy(cu_s).to(device),
            target_logits_indices=torch.tensor(
                target, dtype=torch.int32, device=device),
            bonus_logits_indices=torch.tensor(
                [r * (n + 1) + n for r in range(B)],
                dtype=torch.int32, device=device),
            logits_indices=torch.arange(
                (n + 1) * B, dtype=torch.int32, device=device),
        )
        md.tree_parent_indices = torch.tensor(
            parents, dtype=torch.int32, device=device)
        md.tree_self_logits_indices = torch.tensor(
            self_idx, dtype=torch.int32, device=device)
        return md

    def _make_sm(B):
        def _f(v):
            return torch.full((B,), v, device=device)
        return _SM(
            temperature=_f(0.6), all_greedy=False, all_random=True,
            top_p=None, top_k=None, generators={}, max_num_logprobs=None,
            logprob_token_ids=None, no_penalties=True, prompt_token_ids=None,
            frequency_penalties=_f(0.0), presence_penalties=_f(0.0),
            repetition_penalties=_f(1.0),
            output_token_ids=[[] for _ in range(B)],
            spec_token_ids=[[] for _ in range(B)],
            allowed_token_ids_mask=None, bad_words_token_ids={},
            logitsprocs=_LP(),
        )

    _orig_make_dummy = _mdmod.SpecDecodeMetadata.make_dummy
    _orig_sm = getattr(self.input_batch, "sampling_metadata", None)
    # committer-publish host structures: live steps set these in the
    # pre-forward; warmup must fabricate them (REQKEY fail-loud otherwise)
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _tk
    _orig_rowids = getattr(_tk, "_LUMO_FA_SAMPLER_ROW_REQ_IDS", None)
    _orig_specids = getattr(_tk, "_LUMO_FA_SPEC_ROW_REQ_IDS", None)
    _orig_last = getattr(_tk, "_LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS", None)
    captured = 0
    _rng_state = torch.cuda.get_rng_state(device)
    try:
        for B in range(1, maxB + 1):
            md = _make_md(B)
            _mdmod.SpecDecodeMetadata.make_dummy = classmethod(
                lambda cls, draft_token_ids, device, _md=md: _md)
            self.input_batch.sampling_metadata = _make_sm(B)
            # spec-mode gate: the gdn builder AND the =2 wrapper both key on
            # this persistent buffer (all -1 at warmup => no tree route, no
            # layer staging => stale-flags raise)
            self.num_decode_draft_tokens.np[:B] = n
            self.num_decode_draft_tokens.np[B:] = -1
            self.num_decode_draft_tokens.copy_to_gpu()
            logits = None
            for _phase in ("warm", "capture"):
                _fr13_wids = ["fr13-warmup-%d" % _r for _r in range(B)]
                _tk._LUMO_FA_SAMPLER_ROW_REQ_IDS = _fr13_wids
                _tk._LUMO_FA_SPEC_ROW_REQ_IDS = list(_fr13_wids)
                hs = self._dummy_run(
                    (n + 1) * B,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    force_attention=True,
                    uniform_decode=True,
                )
                hidden = hs[0] if isinstance(hs, tuple) else hs
                _gen = globals().get("_FR13_SG_WU_GEN")
                if _gen is None:
                    _gen = torch.Generator(device=device)
                    _gen.manual_seed(torch.initial_seed() & 0x7FFFFFFF)
                    globals()["_FR13_SG_WU_GEN"] = _gen
                _h = hidden[: (n + 1) * B]
                logits = self.model.compute_logits(
                    torch.rand(_h.shape, dtype=_h.dtype, device=_h.device,
                               generator=_gen))
                # route through sample_tokens: the =2 capture harness wraps
                # ITS call to _sample (calling self._sample directly bypasses
                # the wrapper entirely — boot-29f: done 0/4, all-eager)
                import types as _types
                _ns = _types.SimpleNamespace(
                    scheduled_spec_decode_tokens={
                        _w: [0] * n for _w in _fr13_wids})
                self._fr13_sg_warmup_mode = True
                try:
                    self.execute_model_state = (
                        _ns, logits, md, None, hidden, hidden,
                        None, None, None, None)
                    out = self.sample_tokens(None)
                finally:
                    self._fr13_sg_warmup_mode = False
                    self.execute_model_state = None
                del out
            try:
                torch.cuda.set_rng_state(_rng_state, device)
            except Exception as _rse:
                print("[FR13_STEP_GRAPH] warmup rng-restore failed: "
                      + str(_rse)[:120], flush=True)
            ent = getattr(self, "_fr13_sg_graphs", {}).get(
                (tuple([n] * B), tuple(logits.shape)))
            if ent is not None:
                captured += 1
                print(
                    "[FR13_STEP_GRAPH] WARMUP-CAPTURED B=%d key=%s"
                    % (B, str((tuple([n] * B), tuple(logits.shape)))),
                    flush=True,
                )
            else:
                print(
                    "[FR13_STEP_GRAPH] warmup capture MISSING for B=%d "
                    "(staged fallback remains for this key)" % B,
                    flush=True,
                )
    finally:
        _mdmod.SpecDecodeMetadata.make_dummy = _orig_make_dummy
        if _orig_sm is not None:
            self.input_batch.sampling_metadata = _orig_sm
        _tk._LUMO_FA_SAMPLER_ROW_REQ_IDS = _orig_rowids
        _tk._LUMO_FA_SPEC_ROW_REQ_IDS = _orig_specids
        _tk._LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS = _orig_last
        _by_req = getattr(_tk, "_LUMO_FA_TREE_ACCEPT_BY_REQ", None)
        if isinstance(_by_req, dict):
            for _k in [k for k in _by_req if str(k).startswith(
                    "fr13-warmup-")]:
                _by_req.pop(_k, None)
    print(
        "[FR13_STEP_GRAPH] warmup-capture done: %d/%d keys" % (captured, maxB),
        flush=True,
    )


_fr13_sg_orig_capture_model = GPUModelRunner.capture_model


def _fr13_sg_capture_model_wrapped(self, *a, **k):
    _r = _fr13_sg_orig_capture_model(self, *a, **k)
    import os as _os
    if _os.environ.get("FR13_STEP_GRAPH", "0") == "2":
        try:
            _fr13_sg_warmup_capture(self)
        except Exception as _e:
            import traceback as _tb
            _tb.print_exc()
            print(
                "[FR13_STEP_GRAPH] warmup-capture FAILED (live ladder "
                "remains): " + type(_e).__name__ + ": " + str(_e)[:200],
                flush=True,
            )
    return _r


GPUModelRunner.capture_model = _fr13_sg_capture_model_wrapped
