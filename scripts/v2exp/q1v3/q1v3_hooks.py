#!/usr/bin/env python3
"""q1v3 in-engine hooks (imported by the patched vLLM runner / rejection sampler inside the container).

Env (set by the q1v3 launchers):  Q1V3_ARM_KIND = native | cand,  Q1V3_ARM (A/B/V/P/CAND),  Q1V3_OUT (this
process's output dir),  Q1V3_RUN (run root; common O0 store at $Q1V3_RUN/o0),  Q1V3_CASES (frozen case file).
Without Q1V3_ARM_KIND/Q1V3_OUT every hook is a no-op.

Per request the host client writes $Q1V3_OUT/control.json; the next request seen in the engine is bound to it.
SAFETY CONTRACT: no hook ever raises into the engine (a raise inside the fixed32 sample path poisons the boot).
Instrumentation failures are recorded as case `problems` (case invalid); forcing simply stops for that case.

Native (spec-off) step semantics: the decode step consuming stream[i] runs at num_computed = P + i.
Candidate (fixed32) tree step t consumes root stream[cycle_start[t]] (t < K) or the final pending token (t = K).
Observation points (identical stream indices in both arms):
  O0      pre-forward of the step consuming stream[0]: natural digest (cold first case only), then common-O0 import
  O1_k    pre-forward of the step consuming z_k (stream index b_k): state after r_k + accepted drafts of cycle k
  logits  every consumed stream index (native: its decode row; candidate: the tree row that consumed it)
"""
from __future__ import annotations

import hashlib, json, os, struct, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q1v3_common as Q  # noqa: E402

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

SCHEMA = "q1v3.case-record.v1"
HOOKS_SHA256 = Q.sha256_file(os.path.abspath(__file__))
NC_RESTORE = ("NC_CONV", "NC_GDN")


class Case:
    def __init__(self, ctrl, by_id):
        fx = by_id[ctrl["case_id"]]
        self.ctrl, self.fx, self.obs_id = ctrl, fx, ctrl["obs_id"]
        self.stream, self.outputs, self.cycles = fx["stream"], fx["output_tokens"], fx["cycles"]
        self.K, self.cstart = len(fx["cycles"]), fx["cycle_start"]
        self.bounds = [fx["cycle_start"][k] + c["L"] + 1 for k, c in enumerate(fx["cycles"])]
        self.mut = ctrl.get("mutation")
        self.rid = self.P = self.prompt_sha = None
        self.problems, self.structural, self.sealed = [], [], False
        self.taw_calls = self.draft_calls = 0
        self.cur = None
        self.pre = {}
        self.rec = {"consumed": [], "logits": [], "o1": {}, "o0": {}, "natural_drafts": [], "taw": [], "publication": [],
                    "prefill_chunks": [], "mutation_log": [], "natural_sampled": []}

    def problem(self, msg):
        self.problems.append(f"{Q.utc()} {msg}")


class Hooks:
    def __init__(self):
        self.kind = os.environ.get("Q1V3_ARM_KIND", "")
        self.out = os.environ.get("Q1V3_OUT", "")
        self.enabled = bool(self.kind in ("native", "cand") and self.out)
        self.case = None
        if not self.enabled:
            return
        self.arm = os.environ.get("Q1V3_ARM", "?")
        self.run_root = os.environ.get("Q1V3_RUN", os.path.dirname(self.out))
        cases_path = os.environ.get("Q1V3_CASES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.v1.json"))
        doc = json.load(open(cases_path))
        if Q.record_digest(doc) != doc["record_sha256"]:
            raise RuntimeError("q1v3 cases file digest mismatch")   # import-time only (before any request)
        self.cases_sha256 = Q.sha256_file(cases_path)
        self.by_id = {c["case_id"]: c for c in doc["cases"]}
        for d in ("cases", "objects"):
            os.makedirs(os.path.join(self.out, d), exist_ok=True)
        self.ctrl_path = os.path.join(self.out, "control.json")
        self._ctrl_sig = None
        self.boot_done = False
        self.wrapped = False
        self.kv_armed = False
        self.stale = None
        self.verified_o0 = set()
        self.o0_cache = {}
        self._event("hooks_import", pid=os.getpid(), hooks_sha256=HOOKS_SHA256, cases_sha256=self.cases_sha256)

    # ------------------------------------------------------------------ plumbing
    def _event(self, what, **kw):
        try:
            with open(os.path.join(self.out, f"events.{os.getpid()}.jsonl"), "a") as f:
                f.write(json.dumps({"utc": Q.utc(), "what": what, **kw}, default=str) + "\n")
        except Exception:
            pass

    def _fail(self, where, exc):
        c = self.case
        msg = f"{where}: {type(exc).__name__}: {exc}"
        self._event("hook_exception", where=where, error=msg, tb=traceback.format_exc()[-4000:])
        if c is not None and not c.sealed:
            c.problem(msg)
            try:
                self._seal()
            except Exception:
                pass

    def _load_ctrl(self):
        try:
            st = os.stat(self.ctrl_path)
        except FileNotFoundError:
            return None
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self._ctrl_sig:
            return None
        self._ctrl_sig = sig
        return json.load(open(self.ctrl_path))

    def _bind(self, runner):
        ctrl = self._load_ctrl()
        if ctrl is not None:
            if self.case is not None and not self.case.sealed:
                self.case.problem("control replaced before seal")
                self._seal()
            self.case = Case(ctrl, self.by_id)
            self.stale, self.kv_armed = None, False
            self._event("control", obs_id=ctrl.get("obs_id"))
        c = self.case
        if c is None or c.sealed:
            return None
        ids = list(runner.input_batch.req_ids)
        if len(ids) != 1:
            if c.rid is not None:
                c.problem(f"{len(ids)} requests in batch (B1 required)"); self._seal()
            return None
        rid = ids[0]
        if c.rid is None:
            st = runner.requests[rid]
            p = [int(t) for t in (st.prompt_token_ids or [])]
            c.rid, c.P = rid, len(p)
            c.prompt_sha = hashlib.sha256(struct.pack(f"<{len(p)}I", *p)).hexdigest()
            self._event("bind", obs_id=c.obs_id, rid=rid, P=c.P, prompt_sha256=c.prompt_sha)
        elif rid != c.rid:
            c.problem(f"second request {rid} while bound to {c.rid}"); self._seal(); return None
        return rid

    def _save(self, c, rel, t):
        return Q.save_tensor(self.out, f"objects/{c.obs_id}/{rel}", t)

    def _boot(self, runner):
        if self.boot_done:
            return
        self.boot_done = True
        b = {"utc": Q.utc(), "pid": os.getpid(), "arm": self.arm, "kind": self.kind, "hooks_sha256": HOOKS_SHA256, "cases_sha256": self.cases_sha256}
        try:
            kcc = runner.kv_cache_config
            b["kv_cache"] = {"num_blocks": int(getattr(kcc, "num_blocks", -1)),
                             "tensor_bytes_total": int(sum(int(t.size) for t in kcc.kv_cache_tensors)),
                             "groups": [{"layers": len(g.layer_names), "spec": type(g.kv_cache_spec).__name__,
                                         "block_size": int(getattr(g.kv_cache_spec, "block_size", -1))} for g in kcc.kv_cache_groups]}
            cc = runner.vllm_config.cache_config
            b["cache_config"] = {k: str(getattr(cc, k, None)) for k in ("block_size", "mamba_block_size", "mamba_cache_mode", "mamba_ssm_cache_dtype",
                                                                         "enable_prefix_caching", "kv_cache_memory_bytes", "gpu_memory_utilization", "num_gpu_blocks")}
            sc = runner.vllm_config.speculative_config
            b["speculative"] = None if sc is None else {"method": str(getattr(sc, "method", None)), "k": getattr(sc, "num_speculative_tokens", None)}
            ctx = runner.compilation_config.static_forward_context
            gdn, att = Q.select_layers(ctx)
            b["layers"] = {"gdn": len(gdn), "attention": len(att), "mtp_layers": sorted(k for k in ctx if k.startswith("mtp."))}
            b["packed_recurrent_decode"] = sorted({str(getattr(ctx[n], "enable_packed_recurrent_decode", None)) for n in gdn})
            conv, ssm = Q.gdn_kv(ctx[gdn[0]])
            kvc = Q.attn_kv(ctx[att[0]])
            b["shapes"] = {"conv": list(conv.shape), "conv_dtype": str(conv.dtype), "ssm": list(ssm.shape), "ssm_dtype": str(ssm.dtype),
                           "kv": list(kvc.shape), "kv_dtype": str(kvc.dtype), "kv_stride": list(kvc.stride())}
            b["attention_backend"] = str(getattr(getattr(ctx[att[0]], "attn_backend", None), "get_name", lambda: None)())
            b["env"] = {k: v for k, v in os.environ.items() if k.startswith(("Q1V3_", "VLLM_ENABLE_FLA", "FR13_FIXED32_MODE", "FR14_FUSED_DRAFT_TOPK", "VLLM_BATCH_INVARIANT"))}
        except Exception as e:
            b["error"] = f"{type(e).__name__}: {e}"
        Q.write_json(os.path.join(self.out, f"boot.{os.getpid()}.json"), b)

    # ------------------------------------------------------------------ state access
    def _rows(self, runner, attn_md, idx, gdn):
        if not isinstance(attn_md, dict):
            raise RuntimeError(f"attention metadata is {type(attn_md).__name__}, expected per-layer dict")
        rows = {}
        for n in gdn:
            md = attn_md[n]
            if self.kind == "native":
                row = int(md.non_spec_state_indices_tensor[idx].item())
            else:
                row = int(md.spec_state_indices_tensor[idx, 0].item())
            if row <= 0:
                raise RuntimeError(f"{n}: invalid running state row {row}")
            rows[n] = row
        return rows

    def _tables(self, runner, idx, att):
        gid_of = {}
        for gid, g in enumerate(runner.kv_cache_config.kv_cache_groups):
            for name in g.layer_names:
                gid_of[name] = gid
        out = {}
        for n in att:
            bt = runner.input_batch.block_table[gid_of[n]]
            nb = int(bt.num_blocks_per_row[idx])
            out[n] = ([int(x) for x in bt.get_numpy_array()[idx, :nb]], int(bt.block_size))
        return out

    def _layers(self, runner):
        ctx = runner.compilation_config.static_forward_context
        gdn, att = Q.select_layers(ctx)
        return ctx, gdn, att

    def _capture(self, c, runner, attn_md, idx, tag, n_ctx):
        """State the next forward continues from: GDN running row (conv live taps + SSM) and target KV for [P, n_ctx)."""
        ctx, gdn, att = self._layers(runner)
        rows = self._rows(runner, attn_md, idx, gdn)
        convs, ssms, pad_nz = [], [], 0
        for n in gdn:
            conv, ssm = Q.gdn_kv(ctx[n])
            r = rows[n]
            convs.append(conv[r, :Q.LIVE_TAPS]); ssms.append(ssm[r])
            if int(conv.shape[1]) > Q.LIVE_TAPS:
                pad_nz += int((conv[r, Q.LIVE_TAPS:] != 0).sum().item())
        conv_t, ssm_t = torch.stack(convs), torch.stack(ssms)
        tables = self._tables(runner, idx, att)
        n_new = n_ctx - c.P
        kv_t = torch.stack([Q.read_kv(Q.attn_kv(ctx[n]), tables[n][0], tables[n][1], c.P, n_new) for n in att]) if n_new > 0 else None
        rec = {"tag": tag, "n_ctx": n_ctx, "rows": [rows[n] for n in gdn], "conv_pad_nonzero": pad_nz,
               "finite": bool(torch.isfinite(ssm_t).all().item() and torch.isfinite(conv_t.float()).all().item()),
               "conv": self._save(c, f"{tag}/conv.bin", conv_t), "ssm": self._save(c, f"{tag}/ssm.bin", ssm_t),
               "kv": None if kv_t is None else self._save(c, f"{tag}/kv.bin", kv_t), "kv_start": c.P,
               "pages": sorted({tables[n][1] for n in att})}
        return rec

    # ------------------------------------------------------------------ common O0
    def _o0_dir(self, c):
        return os.path.join(self.run_root, "o0", c.fx["prefix"])

    def _natural_digest(self, c, runner, attn_md, idx):
        ctx, gdn, att = self._layers(runner)
        rows, tables = self._rows(runner, attn_md, idx, gdn), self._tables(runner, idx, att)
        conv = torch.stack([Q.gdn_kv(ctx[n])[0][rows[n], :Q.LIVE_TAPS] for n in gdn])
        ssm = torch.stack([Q.gdn_kv(ctx[n])[1][rows[n]] for n in gdn])
        kv = hashlib.sha256()
        for n in att:
            kv.update(Q.tensor_to_bytes(Q.read_kv(Q.attn_kv(ctx[n]), tables[n][0], tables[n][1], 0, c.P)))
        return {"conv_sha256": Q.sha256_bytes(Q.tensor_to_bytes(conv)), "ssm_sha256": Q.sha256_bytes(Q.tensor_to_bytes(ssm)),
                "kv_sha256": kv.hexdigest(), "P": c.P}

    def _o0_capture(self, c, runner, attn_md, idx):
        d = self._o0_dir(c)
        ctx, gdn, att = self._layers(runner)
        rows, tables = self._rows(runner, attn_md, idx, gdn), self._tables(runner, idx, att)
        os.makedirs(d, exist_ok=True)
        conv = torch.stack([Q.gdn_kv(ctx[n])[0][rows[n], :Q.LIVE_TAPS] for n in gdn])
        ssm = torch.stack([Q.gdn_kv(ctx[n])[1][rows[n]] for n in gdn])
        m = {"schema": "q1v3.o0.v1", "prefix": c.fx["prefix"], "P": c.P, "prompt_sha256": c.prompt_sha, "arm": self.arm, "obs_id": c.obs_id,
             "gdn_layers": gdn, "attn_layers": att, "conv": Q.save_tensor(d, "conv.bin", conv), "ssm": Q.save_tensor(d, "ssm.bin", ssm), "kv": []}
        for i, n in enumerate(att):
            m["kv"].append(Q.save_tensor(d, f"kv_{i:02d}.bin", Q.read_kv(Q.attn_kv(ctx[n]), tables[n][0], tables[n][1], 0, c.P)))
        m["utc"] = Q.utc()
        Q.write_json(os.path.join(d, "manifest.json"), m)   # written last: its presence marks a complete O0
        return {"mode": "capture", "dir": d, "conv_sha256": m["conv"]["sha256"], "ssm_sha256": m["ssm"]["sha256"],
                "kv_sha256": [x["sha256"] for x in m["kv"]]}

    def _o0_import(self, c, runner, attn_md, idx):
        d = self._o0_dir(c)
        m = json.load(open(os.path.join(d, "manifest.json")))
        if m["P"] != c.P or m["prompt_sha256"] != c.prompt_sha:
            raise RuntimeError(f"O0 prefix identity mismatch: O0 P={m['P']} sha={m['prompt_sha256'][:12]} vs request P={c.P} sha={c.prompt_sha[:12]}")
        ctx, gdn, att = self._layers(runner)
        if m["gdn_layers"] != gdn or m["attn_layers"] != att:
            raise RuntimeError("O0 layer registry differs from this engine")
        verify = d not in self.verified_o0
        t0 = time.time()
        rows, tables = self._rows(runner, attn_md, idx, gdn), self._tables(runner, idx, att)
        # Host-side cache of ONE prefix's O0 (cases are ordered by prefix): sha256 is verified on the first load per
        # process; later imports of the same prefix copy the already-verified host tensors (no disk re-read).
        if self.o0_cache.get("dir") != d or self.o0_cache.get("manifest_sha") != Q.sha256_file(os.path.join(d, "manifest.json")):
            self.o0_cache = {}
            self.o0_cache = {"dir": d, "manifest_sha": Q.sha256_file(os.path.join(d, "manifest.json")),
                             "conv": Q.load_tensor(d, m["conv"], verify), "ssm": Q.load_tensor(d, m["ssm"], verify),
                             "kv": [Q.load_tensor(d, m["kv"][i], verify) for i in range(len(att))]}
        conv, ssm = self.o0_cache["conv"], self.o0_cache["ssm"]
        ok, pad_nz = True, 0
        for i, n in enumerate(gdn):
            cs, ss = Q.gdn_kv(ctx[n])
            if int(cs.shape[1]) > Q.LIVE_TAPS:
                pad_nz += int((cs[rows[n], Q.LIVE_TAPS:] != 0).sum().item())
            Q.write_gdn(cs, ss, rows[n], conv[i], ssm[i])
            bc, bs = Q.read_gdn(cs, ss, rows[n])
            ok &= bool(torch.equal(bc, conv[i].to(bc.device)) and torch.equal(bs, ssm[i].to(bs.device)))
        nbytes = conv.numel() * 2 + ssm.numel() * 4
        for i, n in enumerate(att):
            src = self.o0_cache["kv"][i].to(Q.attn_kv(ctx[n]).device)
            kvc = Q.attn_kv(ctx[n])
            Q.write_kv(kvc, tables[n][0], tables[n][1], 0, src)
            ok &= bool(torch.equal(Q.read_kv(kvc, tables[n][0], tables[n][1], 0, c.P), src))
            nbytes += src.numel() * 2
            del src
        self.verified_o0.add(d)
        return {"mode": "import", "dir": d, "readback_equal": ok, "bytes": int(nbytes), "seconds": round(time.time() - t0, 3),
                "sha_verified_this_import": verify, "conv_pad_nonzero_before": pad_nz, "source_arm": m["arm"], "source_obs_id": m["obs_id"]}

    def _o0(self, c, runner, attn_md, idx):
        if c.ctrl.get("natural_digest"):
            c.rec["o0"]["natural"] = self._natural_digest(c, runner, attn_md, idx)
        mode = c.ctrl.get("o0_mode", "import")
        if mode == "capture" and not os.path.exists(os.path.join(self._o0_dir(c), "manifest.json")):
            c.rec["o0"]["capture"] = self._o0_capture(c, runner, attn_md, idx)
        r = self._o0_import(c, runner, attn_md, idx)
        c.rec["o0"]["import"] = r
        if not r["readback_equal"]:
            c.problem("common-O0 import readback differs from source")

    # ------------------------------------------------------------------ token at the consumed row
    @staticmethod
    def _row_token(runner, idx, input_ids, positions):
        if input_ids is None:
            input_ids = runner.input_ids.gpu
        qsl = int(runner.query_start_loc.np[idx])
        tok = int(input_ids[qsl].item())
        pos = int(positions[qsl].item()) if positions is not None and positions.dim() == 1 else (int(positions[0, qsl].item()) if positions is not None else None)
        return qsl, tok, pos

    # ------------------------------------------------------------------ hook entry points
    def on_pre_forward(self, runner, scheduler_output, attn_metadata, input_ids, positions, logits_indices):
        if not self.enabled:
            return
        try:
            rid = self._bind(runner)
            if rid is None:
                return
            self._boot(runner)
            c = self.case
            idx = runner.input_batch.req_id_to_index[rid]
            ncomp = int(runner.requests[rid].num_computed_tokens)
            ns = int(scheduler_output.num_scheduled_tokens[rid])
            if self.kind == "native":
                self._pre_native(c, runner, attn_metadata, idx, ncomp, ns, input_ids, positions)
            else:
                self._install_wrappers(runner)
                self._pre_cand(c, runner, attn_metadata, idx, ncomp, ns, input_ids, positions)
        except Exception as e:
            self._fail("on_pre_forward", e)

    def _pre_native(self, c, runner, attn_md, idx, ncomp, ns, input_ids, positions):
        c.cur = None
        if ncomp < c.P or ns != 1:
            c.rec["prefill_chunks"].append({"ncomp": ncomp, "scheduled": ns}); return
        i = ncomp - c.P
        qsl, tok, pos = self._row_token(runner, idx, input_ids, positions)
        c.rec["consumed"].append({"i": i, "token": tok, "position": pos})
        if i >= len(c.stream):
            c.problem(f"decode beyond stream (i={i})"); return
        if tok != c.stream[i]:
            c.structural.append({"check": "consumed_token", "i": i, "got": tok, "want": c.stream[i]})
        c.cur = {"i": i, "rows": [qsl]}
        if i == 0:
            self._o0(c, runner, attn_md, idx)
        if i in c.bounds:
            k = c.bounds.index(i)
            c.rec["o1"][str(k)] = self._capture(c, runner, attn_md, idx, f"o1_{k}", c.P + i)

    def _pre_cand(self, c, runner, attn_md, idx, ncomp, ns, input_ids, positions):
        c.cur = None
        if ns != Q.N_ROWS:
            c.rec["prefill_chunks"].append({"ncomp": ncomp, "scheduled": ns}); return
        i = ncomp - c.P
        if i in c.cstart:
            t = c.cstart.index(i)
        elif i == len(c.stream) - 1:
            t = c.K
        else:
            c.problem(f"unexpected tree step at stream index {i}"); return
        qsl, tok, pos = self._row_token(runner, idx, input_ids, positions)
        c.rec["consumed"].append({"step": t, "i": i, "root_token": tok, "position": pos})
        if tok != c.stream[i]:
            c.structural.append({"check": "root_token", "step": t, "i": i, "got": tok, "want": c.stream[i]})
        want = c.cycles[t]["row_tokens"] if t < c.K else c.fx["flush_rows"]
        src = input_ids if input_ids is not None else runner.input_ids.gpu
        got = [int(x) for x in src[qsl:qsl + Q.N_ROWS].tolist()]
        c.rec.setdefault("tree_inputs", []).append({"step": t, "ids": got})
        bad = [r for r in range(1, Q.N_ROWS) if got[r] != int(want[r])]
        if bad:   # a harness forcing failure, not candidate behaviour: invalidates the observation (R1)
            c.problem(f"tree step {t}: {len(bad)} forced draft rows not consumed (first {bad[:6]})")
        nodes = c.cycles[t]["nodes"] if t < c.K else [0]
        c.cur = {"t": t, "i": i, "qsl": qsl, "nodes": nodes}
        if t == 0:
            self._o0(c, runner, attn_md, idx)
        else:
            k = t - 1
            if c.mut and c.mut["cycle"] == k and c.mut["mutation"] in NC_RESTORE:
                self._apply_restore(c, runner, attn_md, idx)
            c.rec["o1"][str(k)] = self._capture(c, runner, attn_md, idx, f"o1_{k}", c.P + i)
        if c.mut and c.mut["cycle"] == t and c.mut["mutation"] in NC_RESTORE:
            self._snapshot_pre(c, runner, attn_md, idx)

    # ---------------------------------------------------------- negative controls (candidate only)
    def _snapshot_pre(self, c, runner, attn_md, idx):
        ctx, gdn, att = self._layers(runner)
        rows = self._rows(runner, attn_md, idx, gdn)
        which = 0 if c.mut["mutation"] == "NC_CONV" else 1
        c.pre = {"rows": rows, "which": which, "data": {n: Q.gdn_kv(ctx[n])[which][rows[n]].clone() for n in gdn}}
        c.rec["mutation_log"].append({"utc": Q.utc(), "what": f"{c.mut['mutation']} pre-step snapshot", "step": c.cur["t"]})

    def _apply_restore(self, c, runner, attn_md, idx):
        if not c.pre:
            c.problem("mutation restore without snapshot"); return
        ctx, gdn, att = self._layers(runner)
        rows = self._rows(runner, attn_md, idx, gdn)
        for n in gdn:
            Q.gdn_kv(ctx[n])[c.pre["which"]][rows[n]].copy_(c.pre["data"][n])
        c.rec["mutation_log"].append({"utc": Q.utc(), "what": f"{c.mut['mutation']} applied (running rows restored)", "same_rows": rows == c.pre["rows"]})
        c.pre = {}

    def _install_wrappers(self, runner):
        """Inert pass-through wrappers armed only by a negative-control observation (one step each)."""
        hooks = self
        if not self.wrapped:
            self.wrapped = True
            import lumo_flywheel_serving.fr10_gdn_tree_kernel as TK   # GMR imports the KV16 entry point at call time
            orig_kv16 = TK.launch_attn_kv_linear_remap_syncfree_fixed16

            def kv16(**kw):
                if hooks.kv_armed:
                    hooks.kv_armed = False
                    try:
                        ap = kw["accepted_paths"]
                        kw = dict(kw, accepted_paths=torch.clamp(ap + 1, max=Q.N_ROWS - 1).to(ap.dtype))
                        hooks._mut_log("NC_KV applied: accepted_paths+1 passed to KV16 remap")
                    except Exception as e:
                        hooks._mut_log(f"NC_KV not applied: {e}")
                return orig_kv16(**kw)

            TK.launch_attn_kv_linear_remap_syncfree_fixed16 = kv16
        if getattr(runner, "_q1v3_prep_wrapped", False):
            return
        runner._q1v3_prep_wrapped = True
        orig_prep = runner._prepare_input_ids

        def prep(*a, **k):
            r = orig_prep(*a, **k)
            if hooks.stale is not None:
                tok, hooks.stale = hooks.stale, None
                try:
                    if len(runner.input_batch.req_ids) == 1:
                        runner.input_ids.gpu[0].fill_(int(tok))
                        hooks._mut_log(f"NC_STALE applied: root input overwritten with {tok}")
                    else:
                        hooks._mut_log("NC_STALE not applied: batch != 1")
                except Exception as e:
                    hooks._mut_log(f"NC_STALE not applied: {e}")
            return r

        runner._prepare_input_ids = prep
        self._event("wrappers_installed", kv16=True, prepare_input_ids=True)

    def _mut_log(self, what):
        if self.case is not None:
            self.case.rec["mutation_log"].append({"utc": Q.utc(), "what": what})

    # ---------------------------------------------------------- logits
    def on_logits(self, runner, logits, logits_indices):
        if not self.enabled or self.case is None or self.case.sealed or self.case.cur is None:
            return
        c = self.case
        try:
            cur = c.cur
            if self.kind == "native":
                i = cur["i"]
                row = logits[0] if int(logits.shape[0]) == 1 else logits[runner.input_batch.req_id_to_index[c.rid]]
                c.rec["logits"].append({"i": [i], "obj": self._save(c, f"logits/i{i:04d}.bin", row.detach().unsqueeze(0))})
            else:
                q = cur["qsl"]
                if int(logits.shape[0]) < q + Q.N_ROWS:
                    c.problem(f"tree logits have {int(logits.shape[0])} rows < {q + Q.N_ROWS}"); return
                sel = torch.tensor([q + n for n in cur["nodes"]], device=logits.device)
                ii = [cur["i"] + Q.depth(n) for n in cur["nodes"]]
                c.rec["logits"].append({"i": ii, "step": cur["t"], "nodes": cur["nodes"],
                                        "obj": self._save(c, f"logits/t{cur['t']:02d}.bin", logits.index_select(0, sel).detach())})
        except Exception as e:
            self._fail("on_logits", e)

    # ---------------------------------------------------------- sampling / forcing
    def on_sampled(self, runner, sampler_output, scheduler_output, grammar_output=None):
        if not self.enabled or self.case is None or self.case.sealed or self.case.rid is None:
            return
        c = self.case
        try:
            if grammar_output is not None:
                c.problem("grammar output present"); return
            idx = runner.input_batch.req_id_to_index[c.rid]
            after = int(runner.requests[c.rid].num_computed_tokens) + int(scheduler_output.num_scheduled_tokens[c.rid])
            j = after - c.P
            ids = sampler_output.sampled_token_ids
            if self.kind == "cand":
                if j == 0 and int(scheduler_output.num_scheduled_tokens[c.rid]) != Q.N_ROWS:
                    c.rec["natural_sampled"].append({"j": 0, "natural": int(ids[idx, 0].item())})
                    ids[idx, 0].fill_(int(c.outputs[0]))
                return
            if j < 0:
                return
            if j >= len(c.outputs):
                c.problem(f"sampling beyond forced outputs (j={j})"); self._seal(); return
            c.rec["natural_sampled"].append({"j": j, "natural": int(ids[idx, 0].item())})
            ids[idx, 0].fill_(int(c.outputs[j]))
            if j == len(c.outputs) - 1:
                self._seal()
        except Exception as e:
            self._fail("on_sampled", e)

    def on_drafts(self, runner, draft_token_ids):
        if not self.enabled or self.case is None or self.case.sealed or self.case.rid is None:
            return
        c = self.case
        try:
            # v2: the engine's drafter also runs after every prefill chunk, so the raw call count is not the tree step
            # (v1 defect: forced rows went into discarded prefill-chunk drafts or one cycle early). The drafts produced
            # now feed tree step j = number of tree steps already consumed; prefill-chunk calls all see j == 0 and only
            # the last one reaches the tree forward.
            j = len(c.rec["consumed"])
            c.draft_calls += 1
            idx = runner.input_batch.req_id_to_index[c.rid]
            if not (torch.is_tensor(draft_token_ids) and draft_token_ids.dim() == 2 and int(draft_token_ids.shape[1]) == Q.N_DRAFTS):
                c.problem(f"draft tensor contract drift {getattr(draft_token_ids, 'shape', None)}"); return
            c.rec["natural_drafts"].append({"call": c.draft_calls - 1, "for_step": j, "tokens": [int(x) for x in draft_token_ids[idx].tolist()]})
            rows = c.cycles[j]["row_tokens"] if j < c.K else (c.fx["flush_rows"] if j == c.K else None)
            if rows is not None:
                draft_token_ids[idx].copy_(torch.tensor(rows[1:], dtype=draft_token_ids.dtype, device=draft_token_ids.device))
        except Exception as e:
            self._fail("on_drafts", e)

    def on_taw_products(self, products):
        if not self.enabled or self.case is None or self.case.sealed or self.case.rid is None:
            return products
        c = self.case
        try:
            j = c.taw_calls
            c.taw_calls += 1
            if not (isinstance(products, (tuple, list)) and len(products) == 5):
                c.problem("TAW products are not a 5-tuple"); return products
            out_t, out_l, rows_t, lens_t, last_t = products
            if int(out_t.shape[0]) != 1 or tuple(out_t.shape) != (1, Q.N_ROWS) or tuple(rows_t.shape) != (1, Q.MAX_PATH):
                c.problem(f"TAW geometry {tuple(out_t.shape)} {tuple(rows_t.shape)}"); return products
            natural = {"tokens": out_t[0].tolist(), "len": int(out_l[0]), "rows": rows_t[0].tolist(), "acc": int(lens_t[0]), "last": int(last_t[0])}
            dev = out_t.device
            if j < c.K:
                cy = c.cycles[j]
                nodes = list(cy["nodes"])
                toks = [cy["row_tokens"][n] for n in nodes[1:]] + [cy["z"]]
                commit_nodes = nodes
                if c.mut and c.mut["cycle"] == j and c.mut["mutation"] == "NC_SIB":
                    commit_nodes = Q.validate_path(c.mut["sibling_nodes"])
                    if len(commit_nodes) != len(nodes):
                        c.problem("NC_SIB sibling length differs"); return products
                    c.rec["mutation_log"].append({"utc": Q.utc(), "what": f"NC_SIB applied: committed rows {commit_nodes[1:]} for emitted path {nodes[1:]}"})
            elif j == c.K:
                toks, commit_nodes = [Q.TERMINAL_TOKEN], [0]
            else:
                return products
            L = len(commit_nodes) - 1
            o = torch.full((1, Q.N_ROWS), -1, dtype=torch.int64, device=dev); o[0, :len(toks)] = torch.tensor(toks, dtype=torch.int64, device=dev)
            r = torch.zeros((1, Q.MAX_PATH), dtype=torch.int64, device=dev)
            if L:
                r[0, :L] = torch.tensor(commit_nodes[1:], dtype=torch.int64, device=dev)
            out_t.copy_(o); out_l.fill_(len(toks)); rows_t.copy_(r); lens_t.fill_(L); last_t.fill_(commit_nodes[-1])
            c.rec["taw"].append({"call": j, "natural": natural, "forced_tokens": toks, "forced_rows": commit_nodes[1:]})
            if c.mut and c.mut["cycle"] == j:
                if c.mut["mutation"] == "NC_KV":
                    self.kv_armed = True
                elif c.mut["mutation"] == "NC_STALE":
                    self.stale = int(c.cycles[j]["row_tokens"][0])
            if j == c.K:
                self._seal()
        except Exception as e:
            self._fail("on_taw_products", e)
        return products

    def on_sealed(self, runner):
        """After commit + KV remap + drafter proposal: publication readback of the committed path."""
        if not self.enabled or self.case is None or self.case.sealed or self.case.rid is None:
            return
        c = self.case
        try:
            from vllm.model_executor.layers.mamba import gdn_linear_attn as g
            idx = runner.input_batch.req_id_to_index[c.rid]
            p = getattr(g, "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR", None); ln = getattr(g, "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR", None)
            c.rec["publication"].append({"after_taw_call": c.taw_calls - 1,
                                         "paths": None if p is None else [int(x) for x in p[idx].tolist()],
                                         "len": None if ln is None else int(ln[idx].item()),
                                         "pending_event_cleared": getattr(g, "_FR13_FIXED32_PENDING_EVENT", "unset") is None})
        except Exception as e:
            self._fail("on_sealed", e)

    # ---------------------------------------------------------- sealing
    def _completeness(self, c):
        pr = []
        got = sorted({i for r in c.rec["logits"] for i in r["i"]})
        if got != list(range(len(c.stream))):
            pr.append(f"logits coverage {len(got)}/{len(c.stream)}")
        if sorted(c.rec["o1"]) != sorted(str(k) for k in range(c.K)):
            pr.append(f"O1 coverage {sorted(c.rec['o1'])}")
        if "import" not in c.rec["o0"]:
            pr.append("common O0 not imported")
        if self.kind == "cand":
            if c.taw_calls < c.K + 1:
                pr.append(f"TAW calls {c.taw_calls} < {c.K + 1}")
            if c.draft_calls < c.K + 1:
                pr.append(f"draft calls {c.draft_calls} < {c.K + 1}")
            if len(c.rec.get("tree_inputs", [])) != c.K + 1:
                pr.append(f"tree-input checks {len(c.rec.get('tree_inputs', []))} != {c.K + 1}")
            if not c.rec["natural_sampled"]:
                pr.append("prefill root not forced")
        else:
            if len(c.rec["natural_sampled"]) != len(c.outputs):
                pr.append(f"forced {len(c.rec['natural_sampled'])}/{len(c.outputs)} tokens")
        if c.mut and not c.rec["mutation_log"]:
            pr.append("mutation armed but never applied")
        return pr

    def _seal(self):
        c = self.case
        if c is None or c.sealed:
            return
        c.sealed = True
        problems = list(c.problems) + self._completeness(c)
        doc = {"schema": SCHEMA, "arm": self.arm, "kind": self.kind, "obs_id": c.obs_id, "case_id": c.fx["case_id"], "control": c.ctrl,
               "P": c.P, "prompt_sha256": c.prompt_sha, "valid": not problems, "problems": problems, "structural_mismatches": c.structural,
               "records": c.rec, "hooks_sha256": HOOKS_SHA256, "cases_sha256": self.cases_sha256, "sealed_utc": Q.utc(), "pid": os.getpid()}
        doc["record_sha256"] = Q.record_digest(doc)
        Q.write_json(os.path.join(self.out, "cases", f"{c.obs_id}.json"), doc)
        self._event("sealed", obs_id=c.obs_id, valid=doc["valid"])


class _Disabled:
    enabled = False

    def __getattr__(self, name):
        if name == "on_taw_products":
            return lambda products: products
        return lambda *a, **k: None


_INSTANCE = None


def H():
    global _INSTANCE
    if _INSTANCE is None:
        try:
            _INSTANCE = Hooks()
            if not _INSTANCE.enabled:
                _INSTANCE = _Disabled()
        except Exception as e:  # a broken configuration must not take the engine down mid-request
            sys.stderr.write(f"[q1v3] hooks disabled: {type(e).__name__}: {e}\n")
            _INSTANCE = _Disabled()
    return _INSTANCE
