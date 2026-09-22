"""E1 per-step EVENT RECORDER v3 (capture-only runtime; imported by the emitted served source via e1_event_recorder_shim.py).
Env: E1_RECORD=<jsonl> enables it (unset -> every call returns before any state mutation). Contract: NEVER raises into the
served path; a failing sink sets an in-memory failure flag, tries once to write a fallback flag file, and every later event
(if any can be written) carries the failure counters; the joiner refuses a run whose first event is not `recorder_probe` or
whose events report errors/sink failures (review-17 E1 I5).
Events (every event carries the recorder's own monotonic forward sequence `seq`, incremented at EVERY forward entry - pure
or mixed - so a physical-step chain can never bridge a forward the FR13 timer did not index; I2):
  recorder_probe {t}                                                     - first write (sink validation)
  forward_entry  {seq, num_reqs, num_tokens, max_num_scheduled_tokens, t} - top of _fr13_sfwd_begin (every forward)
  chain_break    {seq, reason}                                           - the timer's wall_break sites
  physical_step  {seq, physical_step_id (= issued fwd_index), t_start, num_reqs, request_ids, cg_mode} - the single wall_mark call
  output_rows    {seq, kind: "pure"|"nonpure", physical_step_id (pure) or null, num_reqs, rows:[{row, request_id,
                  step_idx, emitted_ids, n_emitted, num_draft_tokens, discarded}]}
                 - the EVERY-FORWARD OUTPUT LEDGER (v4): the model runner's synchronous output bookkeeping
                   (`_bookkeeping_sync`, non-async scheduling) runs for prefill/mixed forwards too; their emitted ids
                   (e.g. the API's first token) are recorded with kind "nonpure" and NO physical step, so the joiner
                   can bind the COMPLETE per-request sequence to the API stream, while throughput uses only the
                   prespecified pure subset (rows whose seq carries a physical_step). Rows are in the runner's own
                   request order (req_ids passed EXPLICITLY at the call site - I3). Async scheduling -> error.
  run_close      {n_events, n_forwards, n_physical_steps, n_output_records, sink_failed} - the COMPLETE CLOSE SEAL (last event; atexit or explicit close())
  error          {where, msg, errors, sink_failures}
Every event carries a 1-based write counter `n`; the joiner requires n contiguous and the file to END with a run_close
whose totals match, and refuses when the fail-flag sidecar exists."""
import json, os, threading, time
_LOCK = threading.Lock()
_CUR = {"seq": 0, "fwd_index": None, "seq_of_fwd": None, "seen": set(), "step_idx": {}, "errors": 0, "sink_failures": 0, "probed": False, "sink_failed": False,
        "n": 0, "n_forwards": 0, "n_physical_steps": 0, "n_output_records": 0, "closed": False}
def _path(): return os.environ.get("E1_RECORD")
def _flag_files():
    return [os.environ.get("E1_RECORD_FAILFLAG") or (os.path.dirname(_path() or "/logs/x") + "/e1_recorder_FAILED.flag"), "/tmp/e1_recorder_FAILED.flag"]
def _mark_sink_failed(msg):
    _CUR["sink_failed"] = True; _CUR["sink_failures"] += 1
    for fp in _flag_files():
        try:
            with open(fp, "a") as f: f.write(json.dumps({"t": time.time(), "msg": str(msg)[:300], "errors": _CUR["errors"], "sink_failures": _CUR["sink_failures"]}) + "\n")
            break
        except Exception:  # noqa: BLE001
            continue
def _emit(rec):
    """Best-effort append; NEVER raises. Returns True on success."""
    p = _path()
    if not p: return False
    try:
        with _LOCK:
            _CUR["n"] += 1; rec = dict(rec); rec["n"] = _CUR["n"]; rec.setdefault("errors", _CUR["errors"]); rec.setdefault("sink_failures", _CUR["sink_failures"]); line = json.dumps(rec) + "\n"
            with open(p, "a") as f: f.write(line); f.flush()
        return True
    except Exception as e:  # noqa: BLE001
        try: _mark_sink_failed(repr(e))
        except Exception:  # noqa: BLE001
            pass
        return False
def _error(where, e):
    _CUR["errors"] += 1; _emit({"event": "error", "where": where, "msg": repr(e)[:300]})
def close(reason="atexit"):
    """COMPLETE CLOSE SEAL (review-17 recheck case 2): the LAST event of a valid run. Carries the totals the joiner must
    find in the file (n_events written so far + this one, forwards, physical steps, output records, errors, sink failures).
    A run whose file does not end with a matching run_close is INVALID (tail lost to a sink failure or an unclean stop).
    Registered with atexit at first use; may also be called explicitly. Idempotent; never raises."""
    try:
        if not _path() or not _CUR["probed"] or _CUR["closed"]: return
        _CUR["closed"] = True
        _emit({"event": "run_close", "reason": str(reason), "n_events": _CUR["n"] + 1, "n_forwards": _CUR["n_forwards"], "n_physical_steps": _CUR["n_physical_steps"], "n_output_records": _CUR["n_output_records"],
               "sink_failed": _CUR["sink_failed"], "t": time.perf_counter(), "t_wall_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    except Exception:  # noqa: BLE001
        pass
def probe():
    """Validate the sink once (first use); register the close seal. Never raises."""
    if not _path() or _CUR["probed"]: return _CUR["probed"] and not _CUR["sink_failed"]
    _CUR["probed"] = True
    try:
        import atexit; atexit.register(close, "atexit")
    except Exception:  # noqa: BLE001
        pass
    ok = _emit({"event": "recorder_probe", "t": time.perf_counter(), "t_wall_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid()})
    if not ok:
        try: _mark_sink_failed("probe write failed")
        except Exception:  # noqa: BLE001
            pass
    return ok
def _safe(fn):
    def w(**kw):
        if not _path(): return
        try: fn(**kw)
        except Exception as e:  # noqa: BLE001
            try: _error(fn.__name__, e)
            except Exception:  # noqa: BLE001
                pass
    w.__name__ = fn.__name__; return w
@_safe
def forward_entry(*, num_reqs, num_tokens=None, max_num_scheduled_tokens=None):
    probe(); _CUR["seq"] += 1; _CUR["n_forwards"] += 1; _CUR["fwd_index"] = None; _CUR["seq_of_fwd"] = None
    _emit({"event": "forward_entry", "seq": _CUR["seq"], "num_reqs": int(num_reqs), "num_tokens": None if num_tokens is None else int(num_tokens),
           "max_num_scheduled_tokens": None if max_num_scheduled_tokens is None else int(max_num_scheduled_tokens), "t": time.perf_counter()})
@_safe
def chain_break(*, reason):
    probe(); _emit({"event": "chain_break", "seq": _CUR["seq"], "reason": str(reason)[:120]})
@_safe
def physical_step(*, fwd_index, num_reqs, request_ids, cg_mode=None):
    probe(); rids = [str(r) for r in list(request_ids)]
    if len(rids) != int(num_reqs) or len(set(rids)) != len(rids): raise ValueError(f"request id list inconsistent: num_reqs={num_reqs} ids={rids}")
    _CUR["fwd_index"] = int(fwd_index); _CUR["seq_of_fwd"] = _CUR["seq"]; _CUR["n_physical_steps"] += 1
    _emit({"event": "physical_step", "seq": _CUR["seq"], "physical_step_id": int(fwd_index), "t_start": time.perf_counter(), "t_wall_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "num_reqs": int(num_reqs), "request_ids": rids, "cg_mode": None if cg_mode is None else str(cg_mode)})
@_safe
def commit_rows(*, req_ids, sampled_lists, num_draft_tokens=None, async_scheduling=False):
    """Every-forward output ledger (route-independent). req_ids: the runner's request ids in the SAME order as
    sampled_lists (explicit at the call site); sampled_lists[i]: host list of the ids emitted for request i by THIS
    forward (already parsed; empty when discarded). Prefill/mixed forwards (no pure physical step) are recorded as
    kind "nonpure" with physical_step_id null — never an error (review-17 E1 recheck)."""
    probe()
    if async_scheduling: raise RuntimeError("async scheduling: output boundary unsupported (run invalid)")
    seq = _CUR["seq"]
    if seq in _CUR["seen"]: raise RuntimeError(f"second output ledger record for forward seq {seq}")
    _CUR["seen"].add(seq)
    fi = _CUR["fwd_index"]; pure = fi is not None and _CUR["seq_of_fwd"] == seq
    rids = [str(r) for r in list(req_ids)]; lists = [list(x) if x is not None else [] for x in list(sampled_lists)]
    if len(rids) < len(lists) or len(set(rids[:len(lists)])) != len(lists): raise ValueError(f"req_ids ({len(rids)}) do not index the {len(lists)} sampled rows uniquely")
    rows = []
    for b, ids in enumerate(lists):
        rid = rids[b]; k = _CUR["step_idx"].get(rid, 0); _CUR["step_idx"][rid] = k + 1; em = [int(x) for x in ids if int(x) >= 0]
        rows.append({"row": b, "request_id": rid, "step_idx": k, "emitted_ids": em, "n_emitted": len(em), "num_draft_tokens": (None if num_draft_tokens is None else int(num_draft_tokens[b])), "discarded": len(em) == 0})
    _CUR["n_output_records"] += 1
    _emit({"event": "output_rows", "seq": seq, "kind": "pure" if pure else "nonpure", "physical_step_id": int(fi) if pure else None, "num_reqs": len(lists), "rows": rows})
