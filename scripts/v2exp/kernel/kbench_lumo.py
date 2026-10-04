#!/usr/bin/env python3
"""LumoTree deployed fixed32 route (Hydra27, B1): two-level path-parallel GDN scan with transient cut states (verify) +
fixed16 native committer CUDA graph (accepted-path replay/commit).

Binding: the M1/Q1.2b `Backend` of tools/q1_component_runner_v2_2.py (sha ab27fc9c..., the ONLY binding M1 allows) builds the
deployment-shaped persistent storage (16 page-shared conv/SSM alias pages, rings, SSI groups) and runs the PUBLIC production boot
(subtree_preseed -> committer preseed for all B -> conv pregather -> warm).  Timing then calls the SAME kernel entry points with the
SAME kwargs as Backend.scan / Backend.publish, minus their host-side work that is not part of the deployed step
(per-call .to(device), torch.cuda.synchronize(), host torch.tensor() for the accepted path, counter reads):
  verify = 48 x launch_tree_gdn_prepared(...)           (2 Triton launches/layer: level-0 spine path + 11 level-1 paths)
  commit = accepted_paths/lens D2D copy from a device table (device-resident acceptance, FR13_FIXED32_COMMIT_DEVICE_FILL)
           + launch_tree_gdn_replay_all_layers(...)     (device asserts + one replay of the preseeded fixed16 committer graph:
                                                         fixed16 staging gather + 48 native fused_sigmoid_gating updates)
Deployment: verify runs inside vLLM's FULL_AND_PIECEWISE decode CUDA graph; commit is an eager wrapper replaying its own graph."""
from __future__ import annotations

import os

import kbench_common as C
from kbench_base import Method


class LumoMethod(Method):
    name = "lumo_fixed32"
    deployment_mode = {"verify": "captured inside the FULL_AND_PIECEWISE model-forward CUDA graph (graph mode is the deployment number)",
                       "commit": "eager wrapper (metadata D2D copy + device asserts) + replay of the preseeded fixed16 committer CUDA graph; identical in eager and graph mode",
                       "env": C.DEPLOYED_ENV}

    def setup(self, ops_cpu):
        torch = self.torch
        bad = {k: (os.environ.get(k), v) for k, v in C.DEPLOYED_ENV.items() if os.environ.get(k) != v}
        C.require(not bad, f"deployed fixed32 env not set exactly (got, want): {bad}")
        import lumo_flywheel_serving.fr10_gdn_tree_kernel as kernel        # /workspace/src first on sys.path (setup_paths)
        C.require(C.sha256_file(kernel.__file__) == C.EXPECTED_KERNEL_SHA, f"kernel {kernel.__file__} is not the M1-pinned bytes")
        C.require(kernel._FR13_FIXED32_MODE == "hydra27_fixed32", f"kernel fixed32 mode {kernel._FR13_FIXED32_MODE!r}")
        C.require(tuple(kernel._FR13_FIXED32_PARENT) == tuple(self.topology_parent()), "kernel fixed32 parent != topology authority")
        geom = kernel._read_tree_gdn_geom_override() or {}
        C.require(int(geom.get("BV", -1)) == C.LUMO_BV, f"tree GDN BV override {geom} != BV={C.LUMO_BV}")
        import q1_component_runner_v2_2 as R
        C.require(C.sha256_file(R.__file__) == C.EXPECTED_RUNNER_SHA, "Q1 runner bytes differ from the M1 pin")
        C.require(dict(R.REQUIRED_ENV) == C.DEPLOYED_ENV, "Q1 runner REQUIRED_ENV differs from the deployed env table")
        C.require((R.H, R.HV, R.K, R.V, R.N) == (C.H_K, C.H_V, C.D_K, C.D_V, C.N_PHYS), f"Q1 runner geometry {(R.H, R.HV, R.K, R.V, R.N)}")
        C.require(R.CAPACITY == 1, "B1 benchmark requires the capacity-1 Backend")
        self.kernel, self.R = kernel, R
        self.ops_cpu = ops_cpu
        self.be = R.Backend(self.device, self.L)                            # storage + production boot (preseed + warm)
        C.require(bool(self.be.boot.get("warm_ready")), f"fixed32 boot not warm-ready: {self.be.boot}")
        self.be.load_instances(ops_cpu)
        self.ops = C.stage_operands(ops_cpu, self.device)
        be = self.be
        for l, inp in enumerate(self.ops):
            for n, shp in (("q", (C.N_PHYS, C.H_K, C.D_K)), ("k", (C.N_PHYS, C.H_K, C.D_K)), ("v", (C.N_PHYS, C.H_V, C.D_V)), ("a", (C.N_PHYS, C.H_V)), ("b", (C.N_PHYS, C.H_V))):
                C.require(tuple(inp[n].shape) == shp and inp[n].dtype == torch.bfloat16, f"layer {l} {n} {tuple(inp[n].shape)} {inp[n].dtype}")
        C.require(tuple(be.banks[0].shape[1:]) == (C.H_V, C.D_V, C.D_K) and be.banks[0].dtype == torch.float32, "SSM bank geometry")
        C.require(tuple(be.out.shape) == (self.L, C.N_PHYS, C.H_V, C.D_V), "verification output geometry")
        # exactly Backend.scan's kwargs (runner v2.2 :385-389) with pre-staged device operands
        self.layer_kwargs = [dict(q=inp["q"], k=inp["k"], v=inp["v"], g=be.g0, beta=be.beta0, raw_a=inp["a"], raw_b=inp["b"], A_log=be.A_logs[l], dt_bias=be.dt_biases[l],
                                  h0=be.banks[l], h0_indices=be.spec_idx[l], h0_num_accepted_tokens=be.prev_lens, h0_is_bank=True, h0_index_row=0, h0_batch_index=0,
                                  h0_use_accepted_column=False, n_actual=C.N_PHYS, n_pad=C.N_PHYS, strict_mask=be.strict, visible_mask=be.visible, out=be.out[l], state=None,
                                  output_scale=be.scale, use_qk_l2norm_in_kernel=True, ring_k=be.ring_k[l, 0], ring_v=be.ring_v[l, 0], ring_a=be.ring_a[l, 0],
                                  ring_b=be.ring_b[l, 0], staging_flags=be.flags, staging_rows=1) for l, inp in enumerate(self.ops)]
        # exactly Backend.publish's kwargs (runner v2.2 :414-418)
        self.publish_kwargs = dict(bank_anchor=be.banks[0], bank_off16=be.spec_idx, bank_shape=tuple(be.banks[0].shape), bank_stride=int(be.banks[0].stride(0)),
                                   spec_state_indices=be.spec_idx, prev_lens=be.prev_lens, accepted_paths=be.accepted_paths, accepted_lens=be.accepted_lens,
                                   k_rings=be.ring_k, k_norm_rings=None, gate_rings=None, v_rings=be.ring_v, a_rings=be.ring_a, b_rings=be.ring_b, A_logs=be.A_logs,
                                   dt_biases=be.dt_biases, num_layers=self.L, num_spec_decodes=1, output_scale=be.scale, use_qk_l2norm_in_kernel=True,
                                   runrow_commit=True, runrow_init=True, burn_node_bank=False, banks_list=be.banks)
        # device-resident accepted-path table (one row per accepted path), copied D2D inside the timed commit
        rows, lens = [], []
        for pid in self.path_ids:
            drafts = self.paths[pid][1:]
            C.require(len(drafts) <= 15, f"{pid}: {len(drafts)} drafts exceed the 15 consumed committer columns")
            rows.append(drafts + [0] * (16 - len(drafts)))
            lens.append(len(drafts))
        self.paths_tab = torch.tensor(rows, dtype=torch.int32, device=self.device)
        self.lens_tab = torch.tensor(lens, dtype=torch.int32, device=self.device)
        self.pid_index = {p: j for j, p in enumerate(self.path_ids)}
        self.reset()

    @staticmethod
    def topology_parent():
        import fr13_fixed32_topology as T
        return [int(p) for p in T.PHYSICAL_PARENT]

    def reset(self):
        self.be.restore_O0(self.ops_cpu, 0)
        self.torch.cuda.synchronize()

    # ---- step pieces ------------------------------------------------------------------------------------------
    def verify(self):
        launch = self.kernel.launch_tree_gdn_prepared
        for kw in self.layer_kwargs:
            launch(**kw)

    def verify_layer0(self):
        self.kernel.launch_tree_gdn_prepared(**self.layer_kwargs[0])

    def set_path(self, j):
        self.be.accepted_paths.copy_(self.paths_tab[j:j + 1])
        self.be.accepted_lens.copy_(self.lens_tab[j:j + 1])

    def commit(self, i):
        self.set_path(i % len(self.path_ids))
        self.kernel.launch_tree_gdn_replay_all_layers(**self.publish_kwargs)

    # ---- numerics ---------------------------------------------------------------------------------------------
    def durable_states(self):
        return self.torch.stack([self.be.banks[l][self.R.run_row(l)] for l in range(self.L)]).detach().cpu().clone()

    def numerics(self, path_ids):
        self.reset()
        self.verify()
        self.torch.cuda.synchronize()
        outs = [self.be.out[l].detach().cpu().clone() for l in range(self.L)]
        states = {}
        for pid in path_ids:
            self.reset()
            self.verify()
            self.set_path(self.pid_index[pid])
            self.kernel.launch_tree_gdn_replay_all_layers(**self.publish_kwargs)
            self.torch.cuda.synchronize()
            st = self.durable_states()
            states[pid] = [st[l] for l in range(self.L)]
        self.reset()
        return outs, states

    # ---- timing -----------------------------------------------------------------------------------------------
    def prepare_graphs(self):
        self._graph("verify", self.verify)
        self._graph("verify_layer0", self.verify_layer0)
        self.reset()

    def experiments(self, mode):
        if mode == "graph":
            ex = []
            if self.graph_ok("verify"):
                rv = self._replay("verify")

                def step(i, mark):
                    rv(); mark("verify")
                    self.commit(i); mark("commit")
                ex.append({"name": "step", "fn": step, "note": "verify = graph replay of 48 scans; commit = deployed eager wrapper + committer graph"})
            if self.graph_ok("verify_layer0"):
                r0 = self._replay("verify_layer0")
                ex.append({"name": "verify_layer0", "fn": lambda i, mark: (r0(), mark("verify")), "note": "one GDN layer's scan (layer 0)"})
            return ex

        def step(i, mark):
            self.verify(); mark("verify")
            self.commit(i); mark("commit")
        return [{"name": "step", "fn": step, "note": "eager per-layer launcher calls (host launch overhead included)"},
                {"name": "verify_layer0", "fn": lambda i, mark: (self.verify_layer0(), mark("verify")), "note": "one GDN layer's scan (layer 0)"}]

    # ---- accounting -------------------------------------------------------------------------------------------
    def memory_accounting(self):
        torch, k, be = self.torch, self.kernel, self.be
        st = k.subtree_get(C.N_PHYS, C.H_V, C.D_V, C.D_K, be.device)
        C.require(st.get("schedule") == "fixed32", f"subtree schedule {st.get('schedule')!r} != fixed32")
        lvl0_len = int(st["levels"][0][2])
        lvl0_nodes = [int(x) for x in st["levels"][0][0].reshape(-1).tolist() if int(x) >= 0]
        lvl1_parents = sorted({int(x) for x in st["levels"][1][1].tolist()})
        C.require(lvl0_len == 5 and lvl0_nodes == [0, 1, 4, 9, 14], f"level-0 path {lvl0_nodes}")
        C.require(set(lvl1_parents) <= set(lvl0_nodes), f"level-1 parents {lvl1_parents} not all level-0 cut nodes")
        n_c = len(lvl0_nodes)       # EXPORT_MODE=1: every level-0 node's post-state is exported (5 cut states)
        export = st["export"]
        geom = k._read_tree_gdn_geom_override() or {}
        b_v = int(geom.get("BV"))
        out = {"cut_state_buffer": {"tensor": "kernel._FR13_SUBTREE_CACHE[...]['export'] (fp32 [32,48,128,128], shared by all 48 layers, reused every layer)",
                                    "allocated_bytes": int(export.numel() * export.element_size()), "shape": list(export.shape), "dtype": str(export.dtype),
                                    "logical_cut_nodes": lvl0_nodes, "N_c": n_c, "level1_parent_cut_nodes": lvl1_parents,
                                    "logical_bytes_written_per_layer": n_c * C.STATE_BYTES, "logical_bytes_written_per_step": n_c * C.STATE_BYTES * self.L,
                                    "note": "transient: overwritten by the next layer's level-0 launch; only N_c rows are ever written/read"},
               "paper_formula": C.m_cut_formula(n_c, b_v=b_v),
               "replay_rings": {"bytes": C.tensor_bytes([be.ring_k, be.ring_v, be.ring_a, be.ring_b]),
                                "note": "in-kernel byte copies of k (pre-norm), v, raw a, raw b for all 32 rows x 48 layers (the commit's only input besides the bank)"},
               "verify_output": {"bytes": C.tensor_bytes([be.out])},
               "durable_state_logical_bytes": self.L * C.STATE_BYTES,
               "deployment_page_storage": {"bytes": C.tensor_bytes(be.raws), "note": "16 page-shared conv+SSM alias pages (24 rows x 4 MiB), deployed page geometry; not method scratch"}}
        try:
            fr = k._FR13_FIXED32_COMMITTER_FAST_ROUTE["state"]["states_by_batch"][1]
            tens = {n: t for n, t in fr.items() if torch.is_tensor(t)}
            out["committer_fixed16_staging"] = {"bytes": C.tensor_bytes(tens.values()), "tensors": {n: [list(t.shape), str(t.dtype)] for n, t in tens.items()}}
        except Exception as e:  # noqa: BLE001
            out["committer_fixed16_staging"] = {"error": f"{type(e).__name__}: {e}"}
        return out
