#!/usr/bin/env python3
"""TreeWY and Weaver author implementations, invoked through the FROZEN M1 adapters (m1_adapters_v3 plan builders + m1_loaders).

Every call is built by the M1 plan builders (same callables, args, kwargs, Ref bindings, stash layouts, DFS gather/remap, final
deferred flush) and dispatched exactly as M1's ImageExecutorV3._dispatch does (resolve_refs -> author callable -> bind_result),
minus the per-call torch.cuda.synchronize(), argument digests and logging that M1 added for its untimed evidence.
Plans are built ONCE before timing (plan building performs host->device leaf copies and host checks); during timing the per-step
accepted leaf is copied device-to-device from a pre-staged table into the persistent leaf buffer the plans reference.

Weaver (author serving route, gdn_backend._fused_tree_verify_forward):
  verify/layer = fused_gdn_gating -> g/beta stash, l2norm_fwd_strided(q) (allocates), l2norm_fwd_strided(k, out=k stash),
                 v copy -> v stash, tree_gdn_triton_verify(norm=False, precision="tf32", bf16_mode="none")
  commit       = advance_ssm_states_along_accept_paths (one launch, all layers, replays the accepted path from the stashes)
  topology     = build_tree_structure_into_fast + build_tree_ancestor_masks: built once and reused (M1 policy); production rebuilds
                 it every step, so it is timed as a separate 'topology_rebuild' experiment.
  weaver_aligned_local = same author kernels with M1's LOCAL fp32 rsqrt(+1e-6) normalization and unrounded fp32 g/beta.
TreeWY (tree_wy_tree_commit_capture_triton, mask-taking wrapper, dot_bf16 author default):
  step/layer   = physical->DFS gather (charged), ONE fused call that first commits the PREVIOUS step's accepted leaf from the
                 vt/kk/gc stash and then verifies this tree, DFS->physical output remap (charged).
  verify-only  = same step with the positive gcum sentinel set (untimed) so the in-kernel commit is skipped.
  final_flush  = one extra fused call per layer (finite-session flush; its verification output is discarded)."""
from __future__ import annotations

import sys

import kbench_common as C
from kbench_base import Method


class _M1Executor:
    """Lean re-statement of m1_executor_image_v3.ImageExecutorV3._dispatch (no sync, no digests)."""

    def __init__(self, torch):
        import m1_cycle_driver_v2 as D2
        import m1_adapters_v3 as A3
        self.torch, self.D2, self.A3 = torch, D2, A3
        self.results = {}
        self._fns = {}

    def resolve(self, name):
        fn = self._fns.get(name)
        if fn is None:
            parts = name.split(".")
            for i in range(len(parts) - 1, 0, -1):
                mod = sys.modules.get(".".join(parts[:i]))
                if mod is not None:
                    obj = mod
                    for p in parts[i:]:
                        obj = getattr(obj, p)
                    fn = obj
                    break
            C.require(fn is not None, f"callable {name!r} is not bound (M1 loader not executed?)")
            self._fns[name] = fn
        return fn

    def run(self, entry, check=False):
        args, kwargs = self.D2.resolve_refs(entry, self.results)
        if check:
            self.A3.assert_same_device(self.A3.direct_tensors(args, kwargs), entry["callable"])
            if entry.get("precheck"):
                entry["precheck"](args, kwargs)
        if entry["kind"] in ("torch_op", "local_helper"):
            result = entry["callable_obj"](*args, **kwargs)
        else:
            result = self.resolve(entry["callable"])(*args, **kwargs)
        return self.D2.bind_result(entry, result, self.results, kwargs)


def _load_author_modules():
    import m1_loaders as LD
    rec = LD.load(execute=True)          # manifest-verified author/helper bytes, isolated namespace, one declared shim
    C.require(rec.get("executed"), "M1 author loader did not execute")
    return {"created": rec.get("created"), "capability_flags": rec.get("capability_flags"), "merge_dot_precision": rec.get("merge_dot_precision"),
            "all_bound": rec["plan"]["all_bound"], "manifest": rec["plan"]["manifest"]}


# =====================================================================================================================  Weaver
class WeaverMethod(Method):
    deployment_mode = {"verify": "author serving route (SGLang); graph mode = decode CUDA graph as in SGLang serving",
                       "commit": "advance_ssm_states_along_accept_paths, one launch for all layers",
                       "topology": "static topology reused across steps (M1 policy); per-step rebuild timed separately as topology_rebuild"}

    def __init__(self, torch, device, layers, path_table, args, variant="weaver_author_default"):
        super().__init__(torch, device, layers, path_table, args)
        C.require(variant in ("weaver_author_default", "weaver_aligned_local"), variant)
        self.variant = variant
        self.name = variant

    def setup(self, ops_cpu):
        torch = self.torch
        self.loader = _load_author_modules()
        self.X = _M1Executor(torch)
        A3 = self.X.A3
        self.ops_cpu = ops_cpu
        self.ops = C.stage_operands(ops_cpu, self.device)
        self.bufs = A3.weaver_alloc(self.variant, 1, 2, self.L, device=self.device)
        b = self.bufs
        C.require(tuple(b["ssm_states"].shape) == (self.L, 2, C.H_V, C.D_V, C.D_K), f"ssm_states {tuple(b['ssm_states'].shape)}")
        C.require(tuple(b["k_stash"].shape) == (self.L, 1, C.N_PHYS, C.H_K * C.D_K), f"k_stash {tuple(b['k_stash'].shape)}")
        self.slot = int(b["cache_indices"][0])
        self.build = A3.weaver_build_plan(b)
        for e in self.build:                                   # alloc_tree_structure_buffers, build_tree_structure_into_fast, ancestor masks
            self.X.run(e, check=True)
        C.require("tree" in self.X.results and "ancestor_masks" in self.X.results, "Weaver topology build did not bind tree/ancestor_masks")
        tree = self.X.results["tree"]
        C.require(tuple(tree.anc_u8.shape) == (1, C.N_PHYS, C.N_PHYS) and int(tree.max_depth) == C.MAX_DEPTH, "Weaver TreeStructure geometry")
        self.layer_plans = [A3.weaver_prepare_calls(l, self.ops[l], b, self.variant)["calls"] for l in range(self.L)]
        self.replay = A3.weaver_replay_plan(b, self.paths["root-only"])
        self.leaf_tab = torch.tensor([self.paths[p][-1] for p in self.path_ids], dtype=torch.int64, device=self.device)
        self.pid_index = {p: j for j, p in enumerate(self.path_ids)}
        self.reset()
        for plan in self.layer_plans:                          # one checked eager pass (author preconditions on resolved tensors)
            for e in plan:
                self.X.run(e, check=True)
        self.X.run(self.replay, check=True)
        C.sync(torch, self.device)
        self.reset()

    def reset(self):
        b = self.bufs
        b["ssm_states"].zero_()
        for l, inp in enumerate(self.ops):
            b["ssm_states"][l, self.slot].copy_(inp["S0"])
        for n in ("k_stash", "v_stash", "g_stash", "beta_stash", "last_correct_steps"):
            b[n].zero_()
        C.sync(self.torch, self.device)

    def verify(self):
        run = self.X.run
        for plan in self.layer_plans:
            for e in plan:
                run(e)

    def verify_layer0(self):
        for e in self.layer_plans[0]:
            self.X.run(e)

    def set_leaf(self, j):
        self.bufs["last_correct_steps"].copy_(self.leaf_tab[j:j + 1])

    def commit_body(self):
        self.X.run(self.replay)

    def commit(self, i):
        self.set_leaf(i % len(self.path_ids))
        self.commit_body()

    def topology_rebuild(self):
        for e in self.build[1:]:
            self.X.run(e)

    def numerics(self, path_ids):
        torch = self.torch
        self.reset()
        outs = []
        for plan in self.layer_plans:
            for e in plan:
                self.X.run(e)
            outs.append(self.X.results["o"][0].detach().cpu().clone())       # [T,HV,V], token order == physical order
        states = {}
        for pid in path_ids:
            self.reset()
            self.verify()
            self.set_leaf(self.pid_index[pid])
            self.commit_body()
            C.sync(torch, self.device)
            st = self.bufs["ssm_states"][:, self.slot].detach().cpu().clone()
            states[pid] = [st[l] for l in range(self.L)]
        self.reset()
        return outs, states

    def prepare_graphs(self):
        self._graph("verify", self.verify)
        self._graph("commit", self.commit_body)
        self._graph("verify_layer0", self.verify_layer0)
        self._graph("topology_rebuild", self.topology_rebuild)
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
                ex.append({"name": "step", "fn": step, "note": "verify graph (48 x prep+verify) + leaf D2D copy + commit graph"})
            if self.graph_ok("verify_layer0"):
                r0 = self._replay("verify_layer0")
                ex.append({"name": "verify_layer0", "fn": lambda i, mark: (r0(), mark("verify"))})
            if self.graph_ok("topology_rebuild"):
                rt = self._replay("topology_rebuild")
                ex.append({"name": "topology_rebuild", "fn": lambda i, mark: (rt(), mark("topology")), "note": "per-step production rebuild NOT included in step"})
            return ex

        def step(i, mark):
            self.verify(); mark("verify")
            self.commit(i); mark("commit")
        return [{"name": "step", "fn": step}, {"name": "verify_layer0", "fn": lambda i, mark: (self.verify_layer0(), mark("verify"))},
                {"name": "topology_rebuild", "fn": lambda i, mark: (self.topology_rebuild(), mark("topology")), "note": "per-step production rebuild NOT included in step"}]

    def memory_accounting(self):
        b = self.bufs
        out = {"stashes": {"bytes": C.tensor_bytes([b["k_stash"], b["v_stash"], b["g_stash"], b["beta_stash"]]),
                           "note": "per-layer per-slot k (normalized), v, g, beta of all 32 rows: what the accepted replay reads"},
               "ssm_states_allocated": {"bytes": C.tensor_bytes([b["ssm_states"]]), "pool_slots": int(b["ssm_states"].shape[1]),
                                        "note": "pool=2 (slot 0 = PAD convention unused, slot 1 durable)"},
               "durable_state_logical_bytes": self.L * C.STATE_BYTES,
               "topology": {"bytes": C.tensor_bytes([t for t in self.X.results["tree"] if self.torch.is_tensor(t)] + [b["ancestor_masks"], b["parent_tokens"]])}}
        try:
            mod = sys.modules["sglang.srt.layers.attention.fla.gdn_tree_triton"]
            ws = [t for t in getattr(mod, "_WORKSPACE", {}).values() if self.torch.is_tensor(t)]
            tri = [v[0] for v in getattr(mod, "_TRI_CACHE", {}).values() if isinstance(v, tuple) and self.torch.is_tensor(v[0])]
            out["verifier_workspace_cache"] = {"bytes": C.tensor_bytes(ws + tri), "entries": len(ws) + len(tri),
                                               "note": "author _WORKSPACE (beta, A, QKD, Ainv) + _TRI_CACHE; never evicted"}
        except Exception as e:  # noqa: BLE001
            out["verifier_workspace_cache"] = {"error": f"{type(e).__name__}: {e}"}
        out["per_call_allocations"] = "prefix [B,NBT,HV] fp32, U [B*HV,NBT,V] fp32, O [B,T,HV,V] bf16, q_norm [1,T,H,K] per layer (captured by peak allocated)"
        return out


# =====================================================================================================================  TreeWY
class TreeWYMethod(Method):
    deployment_mode = {"step": "one fused call per layer = commit(previous accepted leaf) + verify(this tree); graph mode = decode CUDA graph (vLLM fork)",
                       "commit": "not separable: fused into the next verify; estimated as step - verify_only (sentinel) and reported as derived",
                       "flush": "final deferred flush = one extra fused call per layer (finite-session overhead)"}

    def __init__(self, torch, device, layers, path_table, args, dot_bf16=True):
        super().__init__(torch, device, layers, path_table, args)
        self.dot_bf16 = bool(dot_bf16)
        self.name = "treewy_author_default" + ("" if self.dot_bf16 else "_dotbf16_false")

    def setup(self, ops_cpu):
        torch = self.torch
        self.loader = _load_author_modules()
        self.X = _M1Executor(torch)
        A3 = self.X.A3
        import m1_adapters as A1
        import m1_topology_mapping as M
        self.A1, self.M = A1, M
        self.ops_cpu = ops_cpu
        self.ops = C.stage_operands(ops_cpu, self.device)
        self.bufs = A3.treewy_alloc(2, self.L, 1, device=self.device)
        b = self.bufs
        C.require(tuple(b["ssm_state"].shape) == (self.L, 2, C.H_V, C.D_V, C.D_K), f"ssm_state {tuple(b['ssm_state'].shape)}")
        C.require(tuple(b["vt_stash"].shape) == (self.L, 2, C.H_V, C.N_PHYS, C.D_V), f"vt_stash {tuple(b['vt_stash'].shape)}")
        self.slot = int(b["slots"][0])
        for e in A3.treewy_build_plan(b):
            self.X.run(e, check=True)
        variant = "treewy_author_default"
        self.gather = [A3.treewy_gather_plan(l, self.ops[l], b) for l in range(self.L)]
        self.calls = [A3.treewy_call_args(l, self.ops[l], b, 0, variant, dot_bf16=self.dot_bf16) for l in range(self.L)]
        self.flush_calls = [A3.treewy_call_args(l, self.ops[l], b, 0, variant, dot_bf16=self.dot_bf16, flush=True) for l in range(self.L)]
        self.remap = [A3.treewy_remap_plan(l, b) for l in range(self.L)]
        self.leaf_tab = torch.tensor([A1.treewy_leaf_author_id(self.paths[p]) for p in self.path_ids], dtype=torch.int64, device=self.device)
        self.pid_index = {p: j for j, p in enumerate(self.path_ids)}
        self.reset()
        for l in range(self.L):                                # one checked eager pass
            self.X.run(self.gather[l], check=True); self.X.run(self.calls[l], check=True); self.X.run(self.remap[l], check=True)
        for l in range(self.L):
            self.X.run(self.flush_calls[l], check=True)
        C.sync(torch, self.device)
        self.reset()

    def reset(self):
        b = self.bufs
        b["ssm_state"].zero_()
        for l, inp in enumerate(self.ops):
            b["ssm_state"][l, self.slot].copy_(inp["S0"])
        self.A1.treewy_reset_stash(b)                          # vt/kk zero + POSITIVE gcum sentinel
        b["leaf_full"].zero_(); b["out_phys"].zero_()
        for n in ("dfs.q", "dfs.k", "dfs.v", "dfs.a", "dfs.b"):
            b[n].zero_()
        C.sync(self.torch, self.device)

    def set_sentinel(self):
        self.bufs["gc_stash"][:, self.slot].fill_(1.0)         # author mechanism: positive gcum => skip the in-kernel commit

    def set_leaf(self, j):
        self.bufs["leaf_full"].copy_(self.leaf_tab[j:j + 1])

    def step_body(self):
        run = self.X.run
        for l in range(self.L):
            run(self.gather[l]); run(self.calls[l]); run(self.remap[l])

    def layer0_body(self):
        self.X.run(self.gather[0]); self.X.run(self.calls[0]); self.X.run(self.remap[0])

    def flush_body(self):
        for e in self.flush_calls:
            self.X.run(e)

    def numerics(self, path_ids):
        torch = self.torch
        self.reset()
        self.set_sentinel()
        self.step_body()                                       # first verification from S0 (commit skipped by the sentinel)
        C.sync(torch, self.device)
        outs = [self.bufs["out_phys"][l][0].detach().cpu().clone() for l in range(self.L)]
        states = {}
        for pid in path_ids:
            self.reset()
            self.set_sentinel()
            self.step_body()
            self.set_leaf(self.pid_index[pid])
            self.flush_body()                                  # commits the accepted leaf inside the fused call
            C.sync(torch, self.device)
            st = self.bufs["ssm_state"][:, self.slot].detach().cpu().clone()
            states[pid] = [st[l] for l in range(self.L)]
        self.reset()
        return outs, states

    def prepare_graphs(self):
        self._graph("step", self.step_body)
        self._graph("flush", self.flush_body)
        self._graph("layer0", self.layer0_body)
        self.reset()

    def experiments(self, mode):
        n = len(self.path_ids)
        if mode == "graph":
            ex = []
            if self.graph_ok("step"):
                rs = self._replay("step")
                ex.append({"name": "step", "fn": lambda i, mark: (self.set_leaf(i % n), rs(), mark("fused_commit_prev_and_verify")),
                           "note": "steady state: the in-kernel commit of the previous accepted leaf is active"})
                ex.append({"name": "verify_only_sentinel", "fn": lambda i, mark: (self.set_leaf(i % n), rs(), mark("verify_only")),
                           "pre": lambda i: self.set_sentinel(), "note": "positive gcum sentinel set before each iteration (untimed): commit skipped"})
            if self.graph_ok("flush"):
                rf = self._replay("flush")
                ex.append({"name": "final_flush", "fn": lambda i, mark: (self.set_leaf(i % n), rf(), mark("flush"))})
            if self.graph_ok("layer0"):
                r0 = self._replay("layer0")
                ex.append({"name": "layer0_step", "fn": lambda i, mark: (self.set_leaf(i % n), r0(), mark("fused_commit_prev_and_verify"))})
            return ex
        return [{"name": "step", "fn": lambda i, mark: (self.set_leaf(i % n), self.step_body(), mark("fused_commit_prev_and_verify"))},
                {"name": "verify_only_sentinel", "fn": lambda i, mark: (self.set_leaf(i % n), self.step_body(), mark("verify_only")), "pre": lambda i: self.set_sentinel()},
                {"name": "final_flush", "fn": lambda i, mark: (self.set_leaf(i % n), self.flush_body(), mark("flush"))},
                {"name": "layer0_step", "fn": lambda i, mark: (self.set_leaf(i % n), self.layer0_body(), mark("fused_commit_prev_and_verify"))}]

    def memory_accounting(self):
        b = self.bufs
        per_slot = (C.H_V * C.N_PHYS * C.D_V + C.N_PHYS * C.H_V * C.D_K + C.H_V * C.N_PHYS) * 4
        return {"stashes_allocated": {"bytes": C.tensor_bytes([b["vt_stash"], b["kk_stash"], b["gc_stash"]]), "pool_slots": int(b["vt_stash"].shape[1]),
                                      "logical_bytes_one_slot_all_layers": per_slot * self.L,
                                      "note": "vt [Hv,N,V], kk [N,Hv,K], gc [Hv,N] fp32 per layer per slot: the deferred-commit factors"},
                "ssm_state_allocated": {"bytes": C.tensor_bytes([b["ssm_state"]]), "pool_slots": int(b["ssm_state"].shape[1])},
                "durable_state_logical_bytes": self.L * C.STATE_BYTES,
                "dfs_staging_and_remap": {"bytes": C.tensor_bytes([b[n] for n in ("dfs.q", "dfs.k", "dfs.v", "dfs.a", "dfs.b", "out_phys")]),
                                          "note": "M1 physical->DFS operand staging + inverse-remapped output (charged gather/remap)"},
                "masks": {"bytes": C.tensor_bytes([b["anc_s"], b["anc_i"]])},
                "per_call_allocations": "out [1,N,Hv,V] bf16 per layer call (captured by peak allocated)"}
