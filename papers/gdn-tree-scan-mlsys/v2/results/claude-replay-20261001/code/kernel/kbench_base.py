#!/usr/bin/env python3
"""Method interface shared by every benchmarked method (Lumo, Weaver, TreeWY, naive baselines).

A method owns persistent device storage built in setup(); reset() restores the durable state to S0 (untimed);
experiments(mode) returns the timed experiments for mode "eager" or "graph".  Each experiment is a dict
  {"name", "fn": fn(i, mark), "pre": optional untimed fn(i), "note"}
where fn enqueues GPU work for iteration i and calls mark(label) at the END of each region.  In graph mode the
CUDA graphs are captured by prepare_graphs(); a failed capture is recorded and the affected experiment is skipped
(never silently replaced by eager timing)."""
from __future__ import annotations

import kbench_common as C


class Method:
    name = "base"
    deployment_mode = {}        # what the deployed system does for each phase (documentation in the JSON)

    def __init__(self, torch, device, layers, path_table, args):
        self.torch, self.device, self.L, self.args = torch, device, int(layers), args
        C.require(self.L == C.GDN_LAYERS or getattr(args, "allow_layer_subset", False),
                  f"{layers} layers requested; the benchmark geometry is {C.GDN_LAYERS} GDN layers")
        self.paths = path_table                         # {path_id: nodes}
        self.path_ids = sorted(path_table, key=lambda p: (len(path_table[p]), p))   # 28 paths, cycled over iterations
        self.graphs, self.capture = {}, {}
        self.ops_cpu = None

    # ---- lifecycle -------------------------------------------------------------------------------------------
    def setup(self, ops_cpu):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def path_for_iter(self, i):
        return self.path_ids[i % len(self.path_ids)]

    # ---- numerics (untimed, eager, from S0): returns (outputs_phys[l] cpu, {path_id: [L x [HV,V,K]] cpu}) -----
    def numerics(self, path_ids):
        raise NotImplementedError

    # ---- timing ----------------------------------------------------------------------------------------------
    def prepare_graphs(self):
        """Capture every graph used by experiments('graph'); fill self.graphs / self.capture."""
        raise NotImplementedError

    def experiments(self, mode):
        raise NotImplementedError

    def _graph(self, key, fn):
        g, err = C.capture_graph(self.torch, fn, warmup_iters=int(getattr(self.args, "capture_warmup", 3)))
        self.graphs[key] = g
        self.capture[key] = {"ok": g is not None, "error": err}
        return g

    def _replay(self, key):
        g = self.graphs.get(key)
        C.require(g is not None, f"{self.name}: graph {key!r} not captured")
        return g.replay

    def graph_ok(self, *keys):
        return all(self.capture.get(k, {}).get("ok") for k in keys)

    # ---- accounting ------------------------------------------------------------------------------------------
    def memory_accounting(self):
        return {}

    def describe(self):
        return {"method": self.name, "deployment_mode": self.deployment_mode}
