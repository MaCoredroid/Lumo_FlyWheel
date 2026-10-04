#!/usr/bin/env python3
"""Naive baselines that MATERIALIZE A FULL fp32 RECURRENT STATE FOR EVERY TREE NODE (no repo version exists for the fixed32 tree).

Node-state bank per layer (all 48 layers resident, because acceptance is only known after the whole forward):
  bank [L, 34, Hv, V, K] fp32 : slot 0 = NULL (native-kernel convention), slots 1..32 = node states (slot = node + 1),
                                slot 33 = durable running state.  Logical node-state bytes = L * 32 * 3 MiB = 4.5 GiB.
commit = copy the accepted leaf's state into the durable slot for all layers (index_select into a staging buffer + copy;
         two passes over L * 3 MiB).  An index-swap commit (vLLM MTP style: point the next step's initial-state index at the
         leaf slot) would make commit ~free but keep all 32 slots live per request; reported in the README, not timed.

naive_torch_node  : the literal baseline -- pure PyTorch, sequential per-node update in topological (physical) order
                    S_n = exp(g_n) * S_parent ; S_n += beta_n * (v_n - S_n k_n) k_n^T ; o_n = S_n (q_n * scale)
                    (q/k fp32 x * rsqrt(sum x^2 + 1e-6), softplus threshold 20, sigmoid beta: the q1_oracle C2 math in fp32).
naive_native_paths: the deployed vLLM native kernel fused_sigmoid_gating_delta_rule_update (sha 000ab899...) run over Lumo's
                    own two-level path cover (level 0: spine 0-1-4-9-14 from the durable slot; level 1: 11 paths from their
                    parent's node slot) with per-token state stores (IS_SPEC_DECODING: initial state read from column 15,
                    token t's state stored to column t).  Same launch structure as Lumo (2 launches/layer) but every node's
                    state is written to HBM: isolates the cost of full materialization versus transient cut states."""
from __future__ import annotations

import kbench_common as C
from kbench_base import Method

SLOTS = C.N_PHYS + 2            # NULL + 32 nodes + durable
DURABLE = C.N_PHYS + 1          # 33
NATIVE_COLS = 16                # ssm_state_indices columns; column 15 holds the initial-state slot


class _NaiveBase(Method):
    deployment_mode = {"verify": "not deployed (baseline); graph mode = whole 48-layer verify captured", "commit": "leaf-state copy into the durable slot (graph-captured in graph mode)"}

    def _common_setup(self, ops_cpu):
        torch = self.torch
        import fr13_fixed32_topology as T
        self.parent = [int(p) for p in T.PHYSICAL_PARENT]
        C.require(len(self.parent) == C.N_PHYS and self.parent[0] == -1 and all(self.parent[i] < i for i in range(1, C.N_PHYS)), "parent must be topological (parent < child)")
        self.levels = [[(list(p), int(par)) for p, par in lvl] for lvl in T.SUBTREE_LEVELS]
        self.ops_cpu = ops_cpu
        self.ops = C.stage_operands(ops_cpu, self.device)
        self.bank = torch.zeros((self.L, SLOTS, C.H_V, C.D_V, C.D_K), dtype=torch.float32, device=self.device)
        self.out = torch.zeros((self.L, C.N_PHYS, C.H_V, C.D_V), dtype=torch.bfloat16, device=self.device)
        self.commit_stage = torch.zeros((self.L, 1, C.H_V, C.D_V, C.D_K), dtype=torch.float32, device=self.device)
        self.leaf_slot = torch.zeros((1,), dtype=torch.int64, device=self.device)
        self.leaf_tab = torch.tensor([self.paths[p][-1] + 1 for p in self.path_ids], dtype=torch.int64, device=self.device)
        self.pid_index = {p: j for j, p in enumerate(self.path_ids)}

    def reset(self):
        self.bank.zero_()
        for l, inp in enumerate(self.ops):
            self.bank[l, DURABLE].copy_(inp["S0"])
        C.sync(self.torch, self.device)

    def set_leaf(self, j):
        self.leaf_slot.copy_(self.leaf_tab[j:j + 1])

    def commit_body(self):
        self.torch.index_select(self.bank, 1, self.leaf_slot, out=self.commit_stage)
        self.bank[:, DURABLE:DURABLE + 1].copy_(self.commit_stage)

    def commit(self, i):
        self.set_leaf(i % len(self.path_ids))
        self.commit_body()

    def verify_layer(self, l):
        raise NotImplementedError

    def verify(self):
        for l in range(self.L):
            self.verify_layer(l)

    def verify_layer0(self):
        self.verify_layer(0)

    def numerics(self, path_ids):
        torch = self.torch
        self.reset()
        self.verify()
        C.sync(torch, self.device)
        outs = [self.out[l].detach().cpu().clone() for l in range(self.L)]
        states = {}
        for pid in path_ids:
            self.reset()
            self.verify()
            self.set_leaf(self.pid_index[pid])
            self.commit_body()
            C.sync(torch, self.device)
            st = self.bank[:, DURABLE].detach().cpu().clone()
            states[pid] = [st[l] for l in range(self.L)]
        self.reset()
        return outs, states

    def prepare_graphs(self):
        self._graph("verify", self.verify)
        self._graph("commit", self.commit_body)
        self._graph("verify_layer0", self.verify_layer0)
        self.reset()

    def experiments(self, mode):
        n = len(self.path_ids)
        if mode == "graph":
            ex = []
            if self.graph_ok("verify", "commit"):
                rv, rc = self._replay("verify"), self._replay("commit")

                def step(i, mark):
                    rv(); mark("verify")
                    self.set_leaf(i % n); rc(); mark("commit")
                ex.append({"name": "step", "fn": step})
            if self.graph_ok("verify_layer0"):
                r0 = self._replay("verify_layer0")
                ex.append({"name": "verify_layer0", "fn": lambda i, mark: (r0(), mark("verify"))})
            return ex

        def step(i, mark):
            self.verify(); mark("verify")
            self.commit(i); mark("commit")
        return [{"name": "step", "fn": step}, {"name": "verify_layer0", "fn": lambda i, mark: (self.verify_layer0(), mark("verify"))}]

    def memory_accounting(self):
        return {"node_state_bank": {"allocated_bytes": C.tensor_bytes([self.bank]), "shape": list(self.bank.shape),
                                    "logical_node_state_bytes_per_layer": C.N_PHYS * C.STATE_BYTES,
                                    "logical_node_state_bytes_all_layers": C.N_PHYS * C.STATE_BYTES * self.L,
                                    "note": "every node's fp32 state materialized for all 48 layers (needed until acceptance is known)"},
                "commit_staging": {"bytes": C.tensor_bytes([self.commit_stage])},
                "verify_output": {"bytes": C.tensor_bytes([self.out])},
                "durable_state_logical_bytes": self.L * C.STATE_BYTES}


class NaiveTorchNode(_NaiveBase):
    name = "naive_torch_node"

    def setup(self, ops_cpu):
        torch = self.torch
        self._common_setup(ops_cpu)
        N, HV, H, K, V = C.N_PHYS, C.H_V, C.H_K, C.D_K, C.D_V
        self.head_map = (torch.arange(HV, device=self.device) // (HV // H)).contiguous()
        self.qn = torch.zeros((N, HV, K), dtype=torch.float32, device=self.device)
        self.kn = torch.zeros((N, HV, K), dtype=torch.float32, device=self.device)
        self.vf = torch.zeros((N, HV, V), dtype=torch.float32, device=self.device)
        self.decay = torch.zeros((N, HV), dtype=torch.float32, device=self.device)
        self.beta = torch.zeros((N, HV), dtype=torch.float32, device=self.device)
        self.reset()

    def _prep(self, inp):
        torch = self.torch
        F = torch.nn.functional
        q = inp["q"].float()
        k = inp["k"].float()
        qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        torch.index_select(qn, 1, self.head_map, out=self.qn)
        self.qn.mul_(C.SCALE)
        torch.index_select(kn, 1, self.head_map, out=self.kn)
        self.vf.copy_(inp["v"])
        g = -torch.exp(inp["A_log"].float()) * F.softplus(inp["a"].float() + inp["dt_bias"].float(), beta=1.0, threshold=20.0)
        torch.exp(g, out=self.decay)
        torch.sigmoid(inp["b"].float(), out=self.beta)

    def verify_layer(self, l):
        torch = self.torch
        self._prep(self.ops[l])
        bank, out = self.bank[l], self.out[l]
        for node in range(C.N_PHYS):
            p = self.parent[node]
            s_par = bank[DURABLE] if p < 0 else bank[p + 1]
            s = bank[node + 1]
            torch.mul(s_par, self.decay[node].view(C.H_V, 1, 1), out=s)               # S = exp(g) * S_parent
            sk = torch.bmm(s, self.kn[node].unsqueeze(-1)).squeeze(-1)                # [HV,V]
            r = (self.vf[node] - sk) * self.beta[node].unsqueeze(-1)
            s.baddbmm_(r.unsqueeze(-1), self.kn[node].unsqueeze(1))                   # S += r k^T
            out[node].copy_(torch.bmm(s, self.qn[node].unsqueeze(-1)).squeeze(-1))    # o = S (q * scale)


class NaiveNativePaths(_NaiveBase):
    name = "naive_native_paths"

    def setup(self, ops_cpu):
        torch = self.torch
        self._common_setup(ops_cpu)
        from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as mod
        self.native_sha = C.sha256_file(mod.__file__)
        C.require(self.native_sha == C.EXPECTED_NATIVE_SHA or getattr(self.args, "allow_native_sha_mismatch", False),
                  f"native fused_sigmoid_gating sha {self.native_sha} != pinned {C.EXPECTED_NATIVE_SHA}")
        self.fn = mod.fused_sigmoid_gating_delta_rule_update
        self.lv = []
        covered = []
        for lvl in self.levels:
            order, cu, idx = [], [0], []
            for nodes, par in lvl:
                C.require(len(nodes) < NATIVE_COLS, f"path length {len(nodes)} >= {NATIVE_COLS} columns")
                order += nodes
                cu.append(len(order))
                row = [n + 1 for n in nodes] + [0] * (NATIVE_COLS - len(nodes))
                row[NATIVE_COLS - 1] = DURABLE if par < 0 else par + 1
                C.require(par < 0 or par in covered, f"path {nodes} parent {par} not produced by an earlier level")
                idx.append(row)
            covered += order
            T = len(order)
            self.lv.append({"order": torch.tensor(order, dtype=torch.int64, device=self.device), "T": T, "paths": len(lvl),
                            "cu": torch.tensor(cu, dtype=torch.int32, device=self.device),
                            "ssi": torch.tensor(idx, dtype=torch.int32, device=self.device).contiguous(),
                            "nacc": torch.full((len(lvl),), NATIVE_COLS, dtype=torch.int32, device=self.device),
                            "q": torch.zeros((1, T, C.H_K, C.D_K), dtype=torch.bfloat16, device=self.device),
                            "k": torch.zeros((1, T, C.H_K, C.D_K), dtype=torch.bfloat16, device=self.device),
                            "v": torch.zeros((1, T, C.H_V, C.D_V), dtype=torch.bfloat16, device=self.device),
                            "a": torch.zeros((1, T, C.H_V), dtype=torch.bfloat16, device=self.device),
                            "b": torch.zeros((1, T, C.H_V), dtype=torch.bfloat16, device=self.device)})
        C.require(sorted(covered) == list(range(C.N_PHYS)), f"path cover does not cover the 32 physical rows: {sorted(covered)}")
        C.require([lv["paths"] for lv in self.lv] == [1, 11] and [lv["T"] for lv in self.lv] == [5, 27], f"two-level cover drifted {[(lv['paths'], lv['T']) for lv in self.lv]}")
        self.reset()

    def verify_layer(self, l):
        torch = self.torch
        inp, bank_l, out_l = self.ops[l], self.bank[l], self.out[l]
        for lv in self.lv:
            o = lv["order"]
            torch.index_select(inp["q"], 0, o, out=lv["q"][0])          # charged gather into path order
            torch.index_select(inp["k"], 0, o, out=lv["k"][0])
            torch.index_select(inp["v"], 0, o, out=lv["v"][0])
            torch.index_select(inp["a"], 0, o, out=lv["a"][0])
            torch.index_select(inp["b"], 0, o, out=lv["b"][0])
            y, _ = self.fn(A_log=inp["A_log"], a=lv["a"], b=lv["b"], dt_bias=inp["dt_bias"], q=lv["q"], k=lv["k"], v=lv["v"], scale=C.SCALE,
                           initial_state=bank_l, inplace_final_state=True, cu_seqlens=lv["cu"], ssm_state_indices=lv["ssi"],
                           num_accepted_tokens=lv["nacc"], use_qk_l2norm_in_kernel=True)
            out_l.index_copy_(0, o, y.reshape(lv["T"], C.H_V, C.D_V))   # charged inverse remap to physical order

    def memory_accounting(self):
        rec = super().memory_accounting()
        rec["native_kernel_sha256"] = self.native_sha
        rec["path_staging"] = {"bytes": C.tensor_bytes([lv[n] for lv in self.lv for n in ("q", "k", "v", "a", "b")])}
        return rec
