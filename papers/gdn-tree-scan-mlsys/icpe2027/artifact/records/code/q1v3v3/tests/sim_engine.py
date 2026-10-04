"""Tiny CPU stand-in for the two engines, used only by the q1v3 tests.

It reproduces exactly the runner surfaces the hooks touch (request/batch/block-table objects, per-layer
kv_cache tensors with vLLM's interleaved K/V page layout, GDN conv/SSM rows, attention metadata dicts,
sampler output, the fixed32 TAW 5-tuple, the KV16 remap entry point and _prepare_input_ids) and calls
the hook entry points in the engine's order. The "model" is a deterministic toy recurrence: a correct
tree commit produces bitwise the native sequential state; the negative controls must break it.
"""
from __future__ import annotations

import sys, types
import numpy as np
import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import q1v3_common as Q  # noqa: E402

VOCAB, D, H, HD = 97, 4, 1, 2
GDN = [f"language_model.model.layers.{i}.linear_attn" for i in range(64) if i % 4 != 3]
ATT = [f"language_model.model.layers.{i}.self_attn.attn" for i in range(3, 64, 4)]


def _install_fake_modules():
    """project_serving.fr10_gdn_tree_kernel (KV16 entry point) and vllm...gdn_linear_attn (publication)."""
    if "project_serving.fr10_gdn_tree_kernel" not in sys.modules:
        pkg = types.ModuleType("project_serving"); pkg.__path__ = []
        tk = types.ModuleType("project_serving.fr10_gdn_tree_kernel")
        tk.launch_attn_kv_linear_remap_syncfree_fixed16 = _kv16
        pkg.fr10_gdn_tree_kernel = tk
        sys.modules["project_serving"] = pkg; sys.modules["project_serving.fr10_gdn_tree_kernel"] = tk
    names = ["vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.mamba", "vllm.model_executor.layers.mamba.gdn_linear_attn"]
    for i, n in enumerate(names):
        if n not in sys.modules:
            m = types.ModuleType(n); m.__path__ = []
            sys.modules[n] = m
            if i:
                setattr(sys.modules[names[i - 1]], n.rsplit(".", 1)[1], m)
    return sys.modules["project_serving.fr10_gdn_tree_kernel"], sys.modules["vllm.model_executor.layers.mamba.gdn_linear_attn"]


def _kv16(*, kv_caches, slot_mapping, accepted_paths, num_accepted_tokens, **_):
    """Same semantics as the production remap: committed depth m+1 slot <- slot of accepted row ap[m]."""
    L = int(num_accepted_tokens[0])
    for kvc in kv_caches:
        page = int(kvc.shape[2])
        for m in range(L):
            src, dst = int(slot_mapping[int(accepted_paths[0, m])]), int(slot_mapping[m + 1])
            for pl in (0, 1):
                kvc[pl, dst // page, dst % page] = kvc[pl, src // page, src % page].clone()


class NS(types.SimpleNamespace):
    pass


class Table:
    def __init__(self, blocks, page):
        self.block_size = page; self._b = np.array([blocks], dtype=np.int64); self.num_blocks_per_row = np.array([len(blocks)])

    def get_numpy_array(self):
        return self._b


def kv_val(tok, pos, layer):
    return torch.tensor([[[math_f(tok, pos, layer, j)] for j in range(HD)]], dtype=torch.float32).reshape(H, HD).to(torch.bfloat16)


def math_f(tok, pos, layer, j):
    return float(np.sin(0.37 * tok + 0.11 * pos + 0.07 * layer + 0.5 * j))


class Engine:
    """kind = native | cand. page = attention kernel page (native 4, candidate 16)."""

    def __init__(self, kind, prompt, page, taps, nblocks, bug=None):
        self.kind, self.prompt, self.page, self.taps, self.bug = kind, list(prompt), page, taps, bug
        self.tk, self.g = _install_fake_modules()
        ctx = {}
        self.rows = {n: 2 + (i % 3) for i, n in enumerate(GDN)}            # alias-like distinct running rows
        for n in GDN:
            ctx[n] = NS(kv_cache=[[torch.zeros(6, taps, D, dtype=torch.bfloat16), torch.zeros(6, 2, 3, 3, dtype=torch.float32)]])
        self.blocks = list(range(1, nblocks))[::-1]                          # non-identity physical order
        for n in ATT:
            base = torch.zeros(nblocks, 2, page, H, HD, dtype=torch.bfloat16)
            ctx[n] = NS(kv_cache=[base.permute(1, 0, 2, 3, 4)])               # vLLM interleaved K/V page layout
        self.ctx = ctx
        self.runner = NS(input_batch=NS(req_ids=["req-1"], req_id_to_index={"req-1": 0}, block_table=[Table(self.blocks, page)]),
                         requests={"req-1": NS(prompt_token_ids=list(prompt), num_computed_tokens=0)},
                         compilation_config=NS(static_forward_context=ctx),
                         kv_cache_config=NS(kv_cache_groups=[NS(layer_names=list(ATT), kv_cache_spec=NS(block_size=page))] +
                                            [NS(layer_names=[n for j, n in enumerate(GDN) if j % 3 == g], kv_cache_spec=NS(block_size=1024)) for g in range(3)],
                                            kv_cache_tensors=[NS(size=123)], num_blocks=nblocks),
                         query_start_loc=NS(np=np.array([0, 32])), input_ids=NS(gpu=torch.zeros(64, dtype=torch.int64)))
        self.runner.kv_cache_config.kv_cache_groups = [self.runner.kv_cache_config.kv_cache_groups[0]] + self.runner.kv_cache_config.kv_cache_groups[1:]
        self.runner._prepare_input_ids = self._prepare_input_ids
        self.pending = None; self.drafts = None

    # ------------------------------------------------------------- toy model
    def _md(self):
        if self.kind == "native":
            return {n: NS(non_spec_state_indices_tensor=torch.tensor([self.rows[n]])) for n in GDN}
        return {n: NS(spec_state_indices_tensor=torch.tensor([[self.rows[n]] + [0] * 31])) for n in GDN}

    def _slot(self, pos):
        return self.blocks[pos // self.page] * self.page + pos % self.page

    def _step_state(self, conv, ssm, tok, li):
        e = torch.tensor([np.cos(0.13 * tok + 0.29 * li + 0.41 * d) for d in range(D)], dtype=torch.float32).to(torch.bfloat16)
        conv = torch.cat([conv[1:Q.LIVE_TAPS], e.unsqueeze(0)], 0)
        ssm = 0.9 * ssm + 0.01 * float(np.sin(tok * 0.7 + li))
        return conv, ssm

    def _consume(self, toks, pos0, state=None):
        """Sequential recurrence over tokens from the running rows (or a given state); returns per-layer states."""
        st = state or {n: (self.ctx[n].kv_cache[0][0][self.rows[n], :Q.LIVE_TAPS].clone(), self.ctx[n].kv_cache[0][1][self.rows[n]].clone()) for n in GDN}
        for t in toks:
            st = {n: self._step_state(st[n][0], st[n][1], t, li) for li, n in enumerate(GDN)}
        return st

    def _write_state(self, st):
        for n in GDN:
            self.ctx[n].kv_cache[0][0][self.rows[n], :Q.LIVE_TAPS] = st[n][0]
            self.ctx[n].kv_cache[0][1][self.rows[n]] = st[n][1]

    def _logits(self, st, kv_upto):
        # depends on the recurrent/conv state only (KV is checked through its own O1 surface in the tests)
        h = sum(float(st[n][1].sum()) + float(st[n][0].float().sum()) for n in GDN[:6])
        v = torch.arange(VOCAB, dtype=torch.float32)
        return (torch.sin(0.05 * v * h + 0.3 * v) * 8.0).to(torch.bfloat16)

    def _write_kv(self, tok, pos, slot):
        for li, a in enumerate(ATT):
            kvc = self.ctx[a].kv_cache[0]; val = kv_val(tok, pos, li)
            kvc[0, slot // self.page, slot % self.page] = val; kvc[1, slot // self.page, slot % self.page] = -val

    # ------------------------------------------------------------- drive one request
    def _prepare_input_ids(self):
        self.runner.input_ids.gpu[0] = self.pending
        if self.drafts is not None:
            self.runner.input_ids.gpu[1:32] = self.drafts

    def run(self, H, max_steps=400):
        r, req = self.runner, self.runner.requests["req-1"]
        P = len(self.prompt)
        # prefill in chunks; like the deployed engine, the candidate's drafter runs after every chunk and only the
        # last chunk's drafts feed tree step 0 (v1 indexed forced rows by raw drafter calls and missed this)
        cuts = [0, P // 3, (2 * P) // 3, P] if P >= 6 else [0, P]
        assert all(b - a != Q.N_ROWS for a, b in zip(cuts, cuts[1:])), "a 32-token prefill chunk would look like a tree step"
        for a, b in zip(cuts[:-2], cuts[1:-1]):
            req.num_computed_tokens = a
            H.on_pre_forward(r, NS(num_scheduled_tokens={"req-1": b - a}), self._md(), None, None, None)
            if self.kind == "cand":
                H.on_drafts(r, torch.full((1, 31), 3, dtype=torch.int64))
        a = cuts[-2]
        req.num_computed_tokens = a
        H.on_pre_forward(r, NS(num_scheduled_tokens={"req-1": P - a}), self._md(), None, None, None)
        st = self._consume(self.prompt, 0)
        self._write_state(st)
        for p, t in enumerate(self.prompt):
            self._write_kv(t, p, self._slot(p))
        lg = self._logits(st, P).unsqueeze(0)
        H.on_logits(r, lg, None)
        so = NS(sampled_token_ids=torch.tensor([[int(torch.argmax(lg[0]))]]))
        H.on_sampled(r, so, NS(num_scheduled_tokens={"req-1": P - a}))
        self.pending = int(so.sampled_token_ids[0, 0]); req.num_computed_tokens = P
        emitted = [self.pending]
        if self.kind == "cand":
            self.drafts = torch.full((31,), 5, dtype=torch.int64)
            d = self.drafts.unsqueeze(0).clone(); H.on_drafts(r, d); self.drafts = d[0].clone()
        for _ in range(max_steps):
            if emitted[-1] == Q.TERMINAL_TOKEN:
                break
            if self.kind == "native":
                emitted.append(self._native_step(H))
            else:
                emitted += self._tree_step(H)
        return emitted

    def _native_step(self, H):
        r, req = self.runner, self.runner.requests["req-1"]
        pos = req.num_computed_tokens
        self._prepare_input_ids()
        H.on_pre_forward(r, NS(num_scheduled_tokens={"req-1": 1}), self._md(), None, torch.tensor([pos]), None)
        tok = int(r.input_ids.gpu[0])
        st = self._consume([tok], pos); self._write_state(st); self._write_kv(tok, pos, self._slot(pos))
        lg = self._logits(st, pos + 1).unsqueeze(0)
        H.on_logits(r, lg, None)
        so = NS(sampled_token_ids=torch.tensor([[int(torch.argmax(lg[0]))]]))
        H.on_sampled(r, so, NS(num_scheduled_tokens={"req-1": 1}))
        req.num_computed_tokens = pos + 1
        self.pending = int(so.sampled_token_ids[0, 0])
        return self.pending

    def _tree_step(self, H):
        r, req = self.runner, self.runner.requests["req-1"]
        p = req.num_computed_tokens
        self.runner._prepare_input_ids()                                     # may be wrapped by NC_STALE
        H.on_pre_forward(r, NS(num_scheduled_tokens={"req-1": 32}), self._md(), None, torch.tensor([p + Q.depth(i) for i in range(32)]), None)
        toks = [int(x) for x in r.input_ids.gpu[:32]]
        pre = self._consume([], p)
        logits, sm = [], [self._slot(p + i) for i in range(32)]            # tree row i written at slot p+i (flat map)
        for row in range(32):
            path = Q.path_to(row) if row not in Q.INACTIVE_ROWS else [0]
            st = self._consume([toks[x] for x in path], p, dict(pre))
            self._write_kv(toks[row], p + Q.depth(row), sm[row])
            logits.append(self._logits(st, p + len(path)))
        lg = torch.stack(logits)
        H.on_logits(r, lg, None)
        nat = (torch.full((1, 32), -1, dtype=torch.int64), torch.ones(1, dtype=torch.int64), torch.zeros(1, 16, dtype=torch.int64),
               torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64))
        nat[0][0, 0] = int(torch.argmax(lg[0]))
        prod = H.on_taw_products(nat)
        out = [int(x) for x in prod[0][0, :int(prod[1][0])]]
        L = int(prod[3][0]); rows = [int(x) for x in prod[2][0, :L]]
        commit = [toks[0]] + [toks[x] for x in rows]
        if self.bug == "replay_short" and L >= 2:
            commit = commit[:-1]
        self._write_state(self._consume(commit, p, dict(pre)))             # GDN replay + conv commit of the accepted rows
        self.tk.launch_attn_kv_linear_remap_syncfree_fixed16(kv_caches=[self.ctx[a].kv_cache[0] for a in ATT], slot_mapping=torch.tensor(sm),
                                                             accepted_paths=prod[2], num_accepted_tokens=prod[3], num_spec_decodes=1)
        self.g._PROJECT_FA_ACCEPTED_TREE_PATHS_TENSOR = prod[2].to(torch.int32); self.g._PROJECT_FA_ACCEPTED_TREE_LENS_TENSOR = prod[3].to(torch.int32)
        self.g._FR13_FIXED32_PENDING_EVENT = None
        req.num_computed_tokens = p + L + 1
        self.pending = out[-1]
        d = torch.full((1, 31), 7, dtype=torch.int64); H.on_drafts(r, d); self.drafts = d[0].clone()
        H.on_sealed(r)
        return out
