"""Passive gate diagnostic hooks (drop-in for q1v3_hooks; same entry points, NOTHING is modified).

Per candidate tree step (B=1, 32 scheduled tokens) records: the 32 tree input tokens and their
positions, the drafter's last fixed32 work snapshot (gated flag, head_fill_columns), the 31 drafts
the drafter published for the next step, and the natural TAW products (accepted rows / tokens).
Output: $Q1V3_OUT/trace.<pid>.jsonl.  Disabled unless Q1V3_ARM_KIND=cand and Q1V3_OUT are set."""
import json, os, sys, time

MAX_STEPS = int(os.environ.get("GATEDIAG_MAX_STEPS", "6000"))


class _Passive:
    def __init__(self):
        self.out = os.environ.get("Q1V3_OUT", "")
        self.enabled = os.environ.get("Q1V3_ARM_KIND") == "cand" and bool(self.out)
        self.n = 0
        self.fh = None
        if self.enabled:
            os.makedirs(self.out, exist_ok=True)
            self.fh = open(os.path.join(self.out, f"trace.{os.getpid()}.jsonl"), "a", buffering=1)
            self._w({"what": "boot", "pid": os.getpid(), "utc": time.time()})

    def _w(self, rec):
        if self.fh is not None and self.n < MAX_STEPS:
            self.fh.write(json.dumps(rec, default=str) + "\n")

    @staticmethod
    def _work():
        for m in list(sys.modules.values()):
            f = getattr(m, "get_fixed32_drafter_last_work", None)
            if callable(f):
                try:
                    return f()
                except Exception as e:
                    return {"error": repr(e)}
        return None

    def on_pre_forward(self, runner, scheduler_output, attn_metadata, input_ids, positions, logits_indices):
        try:
            ids = list(runner.input_batch.req_ids)
            if len(ids) != 1:
                return
            rid = ids[0]
            ns = int(scheduler_output.num_scheduled_tokens[rid])
            if ns != 32:
                return
            src = input_ids if input_ids is not None else runner.input_ids.gpu
            qsl = int(runner.query_start_loc.np[0])
            p = None
            if positions is not None:
                pp = positions[0] if positions.dim() == 2 else positions
                p = [int(x) for x in pp[qsl:qsl + 32].tolist()]
            self.n += 1
            self._w({"what": "tree", "step": self.n, "rid": str(rid),
                     "ncomp": int(runner.requests[rid].num_computed_tokens),
                     "ids": [int(x) for x in src[qsl:qsl + 32].tolist()], "pos0": (p[0] if p else None),
                     "depths": ([x - p[0] for x in p] if p else None), "work": self._work()})
        except Exception as e:
            self._w({"what": "err", "where": "pre_forward", "e": repr(e)})

    def on_drafts(self, runner, draft_token_ids):
        try:
            if hasattr(draft_token_ids, "dim") and draft_token_ids.dim() == 2 and int(draft_token_ids.shape[0]) == 1:
                self._w({"what": "drafts", "step": self.n, "tokens": [int(x) for x in draft_token_ids[0].tolist()],
                         "work": self._work()})
        except Exception as e:
            self._w({"what": "err", "where": "drafts", "e": repr(e)})

    def on_taw_products(self, products):
        try:
            out_t, out_l, rows_t, lens_t, last_t = products
            self._w({"what": "taw", "step": self.n, "tokens": [int(x) for x in out_t[0].tolist()],
                     "len": int(out_l[0]), "rows": [int(x) for x in rows_t[0].tolist()],
                     "acc": int(lens_t[0]), "last": int(last_t[0])})
        except Exception as e:
            self._w({"what": "err", "where": "taw", "e": repr(e)})
        return products

    def __getattr__(self, name):
        return lambda *a, **k: None


_INSTANCE = None


def H():
    global _INSTANCE
    if _INSTANCE is None:
        try:
            _INSTANCE = _Passive()
        except Exception as e:
            sys.stderr.write(f"[gatediag] hooks disabled: {e!r}\n")

            class _N:
                def __getattr__(self, n):
                    return (lambda products: products) if n == "on_taw_products" else (lambda *a, **k: None)
            _INSTANCE = _N()
    return _INSTANCE
