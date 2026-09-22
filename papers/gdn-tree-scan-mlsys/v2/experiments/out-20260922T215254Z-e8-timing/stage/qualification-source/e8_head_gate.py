"""Untimed same-input head gate, plus tiny in-memory head counts in clean timing.

The gate compares both computations inside each proposal; it never requires
independent boots to produce identical continuations. No model forward is replayed.
"""
import atexit
import hashlib
import json
import os
from pathlib import Path

STATS = {"schema": "e8.head.v1", "primary_head_calls": 0, "legacy_head_calls": 0,
         "proposals": 0, "qualified_proposals": 0, "checked_heads": 0,
         "root_heads": 0, "loop_heads": 0, "failures": [], "closed": False}
OWNER_PID = None


def claim_owner():
    global OWNER_PID
    pid = os.getpid()
    if OWNER_PID == pid:
        return
    path = Path(os.environ.get("E8_HEAD_REPORT", "/logs/e8_head_gate.json"))
    owner = path.with_name("e8_head_owner.json")
    with owner.open("x") as f:
        json.dump({"pid": pid}, f)
        f.write("\n")
    OWNER_PID = pid
    STATS["owner_pid"] = pid


def require(condition, message):
    if not condition:
        STATS["failures"].append(message)
        write_report()
        raise RuntimeError("E8 qualification failed: " + message)


def write_report():
    if OWNER_PID != os.getpid():
        return  # Import-only/forked non-owner processes cannot overwrite evidence.
    path = Path(os.environ.get("E8_HEAD_REPORT", "/logs/e8_head_gate.json"))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(STATS, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)  # Any sink error propagates; there is no valid-prefix fallback.


def close():
    if OWNER_PID != os.getpid():
        return
    STATS["closed"] = True
    write_report()


atexit.register(close)


def equal(a, b):
    import torch
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def state(model, hidden):
    import torch
    # Version/pointer census detects in-place writes or replacement of model
    # parameters/buffers by the head. Hidden bytes and RNG states are compared
    # separately. This is not a claim of full-model cross-boot state equality.
    def version(t):
        try:
            return int(t._version)
        except RuntimeError:  # Inference tensors do not carry a version counter.
            return None
    census = tuple((kind, name, int(t.data_ptr()), version(t), tuple(t.shape), str(t.dtype))
                   for kind, iterator in (("parameter", model.named_parameters()), ("buffer", model.named_buffers()))
                   for name, t in iterator)
    rng = [torch.random.get_rng_state().clone()]
    if hidden.is_cuda:
        rng.append(torch.cuda.get_rng_state(hidden.device).clone())
    return census, hidden.detach().clone(), rng


def compute(owner, hidden, qualify):
    if not qualify:
        return owner.model.compute_logits(hidden)
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    watched = {int(t.untyped_storage().data_ptr()) for t in
               [hidden, *owner.model.parameters(), *owner.model.buffers()]}

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            for v in value:
                yield from tensors(v)
        elif isinstance(value, dict):
            for v in value.values():
                yield from tensors(v)

    class ReadOnlyHead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            STATS["dispatch_ops_checked"] = STATS.get("dispatch_ops_checked", 0) + 1
            for idx, arg in enumerate(func._schema.arguments):
                if arg.alias_info is not None and arg.alias_info.is_write:
                    value = args[idx] if idx < len(args) else kwargs.get(arg.name)
                    require(not any(int(t.untyped_storage().data_ptr()) in watched for t in tensors(value)),
                            "head writes hidden/model state via " + str(func))
            return func(*args, **kwargs)

    with ReadOnlyHead():
        return owner.model.compute_logits(hidden)


def check_state(before, model, hidden):
    after = state(model, hidden)
    require(before[0] == after[0], "head changed model parameter/buffer identity or version")
    require(equal(before[1], hidden), "head mutated hidden input")
    require(len(before[2]) == len(after[2]) and all(equal(a, b) for a, b in zip(before[2], after[2])), "head changed RNG state")


def primary_logits(owner, hidden, site, arm, qualify):
    import torch
    claim_owner()
    STATS.update(arm=arm, qualify=qualify)
    require(not getattr(owner, "use_local_argmax_reduction", False), "local argmax reduction outside E8 scope")
    if qualify:
        before = state(owner.model, hidden)
        require(getattr(owner, "_e8_pending", None) is None, "primary head before prior pair closed")
    logits = compute(owner, hidden, qualify)
    STATS["primary_head_calls"] += 1
    if qualify:
        check_state(before, owner.model, hidden)
        require(logits.ndim == 2 and logits.shape[0] == 1, "qualification requires actual B1 head input")
        require(not bool(torch.isnan(logits).any()) and bool(torch.isfinite(logits).any(dim=-1).all()), "invalid head logits")
        owner._e8_pending = (site, logits.detach().clone(), hidden.detach().clone())
    return logits


def legacy_logits(owner, hidden, arm, qualify):
    import torch
    if qualify:
        pending = getattr(owner, "_e8_pending", None)
        require(pending is not None, "legacy head without paired primary")
        require(equal(pending[2], hidden), "legacy head input differs from primary")
        before = state(owner.model, hidden)
    logits = compute(owner, hidden, qualify)
    STATS["legacy_head_calls"] += 1
    if qualify:
        check_state(before, owner.model, hidden)
        site, primary, _ = pending
        require(equal(primary, logits), "same-input full head logits differ")
        # At temperature zero the target distribution is a point mass on argmax.
        # Check both its token and top-two candidate order, including ties.
        pids = primary.argmax(dim=-1)
        lids = logits.argmax(dim=-1)
        ptop = torch.topk(primary, 2, dim=-1).indices
        ltop = torch.topk(logits, 2, dim=-1).indices
        require(equal(pids, lids), "greedy point-mass/token mismatch")
        require(equal(ptop, ltop), "same-input ordered top-two mismatch")
        if site == "root":
            require(not getattr(owner, "_e8_candidates", []), "unclosed proposal at next root")
            owner._e8_candidates = []
        owner._e8_candidates.append((site, pids.detach().clone(), lids.detach().clone(), ptop[:, 1].detach().clone(), ltop[:, 1].detach().clone()))
        owner._e8_pending = None
        STATS["checked_heads"] += 1
        STATS[site + "_heads"] += 1
    return logits


def finish_proposal(owner, packed, arm, qualify):
    import torch
    STATS["proposals"] += 1
    if qualify:
        rows = getattr(owner, "_e8_candidates", [])
        require(getattr(owner, "_e8_pending", None) is None, "proposal has an unpaired head")
        require([r[0] for r in rows] == ["root", "loop", "loop", "loop", "loop"], "not the frozen depth-five drafter")
        # E1 Cat10 means ROOT + nine drafts (source's legacy cat9 packing),
        # NOT the source variable _fr10_is_cat10 which means ten drafts.
        on, off = [rows[0][1]], [rows[0][2]]
        for _, p, l, pleaf, lleaf in rows[1:]:
            on.extend((p, pleaf)); off.extend((l, lleaf))
        on, off = torch.stack(on, dim=1), torch.stack(off, dim=1)
        require(tuple(packed.shape) == (1, 9), "packed candidate shape is not Cat10 B1")
        require(equal(on, off) and equal(packed, on), "full ordered candidate tree differs from paired reconstruction")
        owner._e8_candidates = []
        STATS["qualified_proposals"] += 1
        write_report()
