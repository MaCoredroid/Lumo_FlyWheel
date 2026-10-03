#!/usr/bin/env python3
"""Q1 reference-engine runtime hooks **v2.1** (imported by the patched stock gpu_model_runner inside the spec-off reference engine).

v2.1 = v2 + ONE bounded repair from the executed smoke q1-native-smoke-20260928T022909Z: on a multimodal-capable architecture the pinned
stock runner feeds `inputs_embeds` and passes `input_ids=None` to the forward (gpu_model_runner.py:3214-3266); the consumed token ids are the
runner's prepared buffer `self.input_ids.gpu` -- the very tensor the embed path reads (:3227).  v2.1 reads that buffer at the request's
query_start_loc when the argument is None, records the token source and the positions shape in the consumed trace, and records the runner
input-path facts at boot.  Everything else (attestation, KV/GDN capture, authentication, sealing) is the v2 text (see hooks_v2_to_v2_1.diff).

v2 =

Same five host anchors as v1 (A1 import, A2 pre-forward O0/O1, A3 O2 raw logits, A4 grammar guard, A5 teacher forcing) with the six
repair groups of q1-fullmodel-implementation-review.md and the native smoke design:

  R1  boot attestation must prove the pinned patched FA2 fork is the executed binary (sha of the installed .so, interface sha, fork symbol
      `varlen_fwd_tree_bias` present, `get_flash_attn_version()==2`), the packed flag on EVERY GDN layer, resolved cache config
      (mamba_cache_mode, block sizes, kernel block sizes, group specs, dtypes), async-scheduling flag; any mismatch is FATAL at boot.
  R2  attention KV extraction by cache group: layer -> kv_cache_group (from kv_cache_config), that group's block table row (kernel-block
      ids after the runner's own conversion), backend logical layout (2, num_blocks, kernel_block, kv_heads, head) after inverse
      permutation, materialized extent = the request's computed tokens at the boundary; full blocks + a valid tail only; per-layer
      shape/dtype/stride/storage-offset/group/block-map/extent/canonical order recorded; missing/duplicate/out-of-range/truncated -> INVALID.
  R3  repeat-unique observation identity (run/arm/process/repeat/case) in every name; payloads content-addressed (blocks, tails, GDN rows)
      with exact-byte verification on reuse; per-observation manifest = shared full blocks + own tail; O0 bytes are complete (recurrent, conv,
      KV) and restorable from the manifest (CPU round-trip helper `reconstruct_logical_kv`).
  R4  the consumed token is read from the PREPARED GPU `input_ids` at the request's `query_start_loc` and its absolute position from the
      prepared `positions`, both compared with the fixture chain (async replacement is handled because the prepared tensors are read after
      `_prepare_input_ids`); runtime prompt IDs + sha are REQUIRED; exactly one scheduled token per forced step; 48 GDN + 16 attention layers
      required with per-layer metadata rows/groups; expected packed attribute on every GDN layer; finite full-vocabulary logits with the
      configured vocab size.
  R5  a case is sealed VALID only if O0, O1 and O2 are present, the consumed trace equals the chain, coverage is complete and every value is
      finite; every failure is sealed exactly once (with whether the request is still active) and never left pending; a control change with
      an unsealed case seals it INVALID first.
  R6  Q1_REF_HOOKS_JOB pointing to a missing/unreadable job is FATAL; job run/arm identity, expected packed flag, archive policy and fixture
      sha are asserted at boot (hooks disabled only when the env var is absent, i.e. no instrumentation requested).
"""
from __future__ import annotations

import hashlib, json, os, struct, time

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

SCHEMA = "lumo.q1.fullmodel.reference-hooks.v2.1"
JOB_ENV = "Q1_REF_HOOKS_JOB"
GDN_LAYER_SUFFIX = ".linear_attn"
ATTN_LAYER_SUFFIX = ".self_attn.attn"
REQUIRED_GDN, REQUIRED_ATTN = 48, 16


def sha256_bytes(b) -> str:
    return hashlib.sha256(bytes(b)).hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def tensor_bytes(t):
    if t.is_cuda:
        torch.cuda.current_stream(t.device).synchronize()
    c = t.detach().contiguous().cpu()
    if c.dtype == torch.bfloat16:
        c = c.view(torch.int16)
    return c.numpy().tobytes()


DTYPE_BYTES = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4, "torch.int16": 2, "torch.int32": 4, "torch.int64": 8, "torch.float64": 8}


def packed_flag(layer) -> str:
    """GDN layer `enable_packed_recurrent_decode` as "0"/"1".  The attribute must EXIST and be a plain bool / 0-1 int (v2 preliminary
    repair review, defect 2): a missing attribute is never read as False, in the primary arm or the packed control."""
    if not hasattr(layer, "enable_packed_recurrent_decode"):
        raise RuntimeError("enable_packed_recurrent_decode attribute missing")
    v = getattr(layer, "enable_packed_recurrent_decode")
    if isinstance(v, bool) or (isinstance(v, int) and v in (0, 1)):
        return str(int(v))
    raise RuntimeError(f"enable_packed_recurrent_decode has unsupported type/value {type(v).__name__}:{v!r}")


def finite_or_raise(t, what: str):
    """Finiteness on the EXACT materialized slice being serialized (never on unmaterialized suffix slots)."""
    if not bool(torch.isfinite(t.detach().float()).all().item()):
        raise RuntimeError(f"nonfinite {what}")


def expected_object_lengths(doc: dict) -> dict:
    """sha256 -> exact expected byte length for every raw object a sealed case document references: GDN rows from recorded shape x dtype,
    K/V full blocks and tails from the recorded logical geometry (kernel block, kv heads, head dim, dtype, valid tail tokens), O2 from
    its recorded length which must equal vocab x 4 (float32).  Raises ValueError on an inconsistent manifest."""
    out = {}
    def put(s, n):
        if not isinstance(s, str) or len(s) != 64:
            raise ValueError(f"malformed object reference {s!r}")
        if s in out and out[s] != n:
            raise ValueError(f"object {s[:12]} referenced with two lengths {out[s]} / {n}")
        out[s] = int(n)
    for tag in ("o0", "o1"):
        o = doc.get(tag) or {}
        for name, g in (o.get("gdn") or {}).items():
            for part in ("conv", "ssm"):
                r = g[part]; n = 1
                for d in r["shape"]:
                    n *= int(d)
                put(r["sha256"], n * DTYPE_BYTES[r["dtype"]])
        for name, a in (o.get("attention") or {}).items():
            per_tok = int(a["shape"][3]) * int(a["shape"][4]) * DTYPE_BYTES[a["dtype"]]; kbs = int(a["kernel_block_size"])
            for blk in a.get("full_blocks", []):
                put(blk["k"], kbs * per_tok); put(blk["v"], kbs * per_tok)
                if blk.get("bytes") is not None and int(blk["bytes"]) != 2 * kbs * per_tok:
                    raise ValueError(f"{name}: full block {blk.get('physical_block')} recorded bytes {blk['bytes']} != 2 x {kbs} x {per_tok}")
            t = a.get("tail")
            if t:
                vt = int(t["valid_tokens"])
                if not (0 < vt < kbs):
                    raise ValueError(f"{name}: tail valid_tokens {vt} not in (0, {kbs})")
                put(t["k"], vt * per_tok); put(t["v"], vt * per_tok)
                if t.get("bytes") is not None and int(t["bytes"]) != 2 * vt * per_tok:
                    raise ValueError(f"{name}: tail recorded bytes {t['bytes']} != 2 x {vt} x {per_tok}")
    o2 = doc.get("o2") or {}
    if o2.get("raw"):
        n = int(o2["raw"]["bytes"])
        if o2.get("vocab") is not None and n != int(o2["vocab"]) * 4:
            raise ValueError(f"O2 raw bytes {n} != vocab {o2.get('vocab')} x 4")
        put(o2["raw"]["sha256"], n)
    return out


def authenticate_objects(doc: dict, objects_root: str):
    """Hash + exact-length authenticate EVERY referenced raw object once (reads deduplicated by sha).  Returns (problems, count, bytes)."""
    try:
        want = expected_object_lengths(doc)
    except (KeyError, ValueError, TypeError) as e:
        return [f"object manifest malformed: {type(e).__name__}: {e}"], 0, 0
    pr, n, nb = [], 0, 0
    for s, ln in sorted(want.items()):
        p = os.path.join(objects_root, s + ".bin")
        if not os.path.exists(p):
            pr.append(f"object {s[:12]} missing"); continue
        size = os.path.getsize(p)
        if size != ln:
            pr.append(f"object {s[:12]} length {size} != expected {ln}"); continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for ch in iter(lambda: f.read(1 << 22), b""):
                h.update(ch)
        if h.hexdigest() != s:
            pr.append(f"object {s[:12]} content sha mismatch (same-size corruption)"); continue
        n += 1; nb += size
    return pr, n, nb


class ContentStore:
    """Content-addressed write-once payload store with exact-byte verification on reuse."""

    def __init__(self, root):
        self.root = root; os.makedirs(root, exist_ok=True); self.written = 0; self.reused = 0; self.bytes_written = 0

    def put(self, b: bytes, meta: dict) -> dict:
        sha = sha256_bytes(b); p = os.path.join(self.root, sha + ".bin")
        if os.path.exists(p):
            if os.path.getsize(p) != len(b) or open(p, "rb").read() != b:
                raise RuntimeError(f"content-address collision/corruption {sha[:12]}")
            self.reused += 1
        else:
            tmp = p + f".tmp.{os.getpid()}.{time.monotonic_ns()}"
            with open(tmp, "wb") as f:
                f.write(b)
            os.replace(tmp, p); self.written += 1; self.bytes_written += len(b)
        return {"sha256": sha, "bytes": len(b), **meta}


class CaseState:
    def __init__(self, control: dict, job: dict):
        for k in ("case_id", "chain_tokens", "chain_positions", "prefix_len", "prefix_sha256", "arm", "repeat", "process", "run_id"):
            if k not in control:
                raise ValueError(f"control file missing {k}")
        if control["run_id"] != job["run_id"] or control["arm"] != job["arm"] or control["process"] != job["process"]:
            raise ValueError("control run/arm/process != job identity")
        jc = {c["case_id"]: c for c in job["cases"]}
        if control["case_id"] not in jc:
            raise ValueError("control case not in the bound job case list")
        if list(control["chain_tokens"]) != list(jc[control["case_id"]]["chain_tokens"]) or list(control["chain_positions"]) != list(jc[control["case_id"]]["chain_positions"]):
            raise ValueError("control chain != job chain")
        if not (isinstance(control["repeat"], int) and 0 <= control["repeat"] < int(job["repeats"])):
            raise ValueError("control repeat outside the job's repeat range")
        self.control = control; self.case_id = control["case_id"]
        self.chain = [int(x) for x in control["chain_tokens"]]; self.positions = [int(x) for x in control["chain_positions"]]
        self.prefix_len = int(control["prefix_len"])
        self.obs_id = f"{job['run_id']}__{job['arm']}__p{job['process']}__r{control['repeat']}__{self.case_id}"
        self.req_id = None; self.forced = 0; self.consumed_trace = []; self.natural_sampled = []; self.records = {}; self.invalid_reasons = []
        self.o0 = self.o1 = self.o2 = None; self.sealed = False; self.request_active = False

    @property
    def z_step_num_computed(self):
        return self.prefix_len + len(self.chain) - 1

    def invalidate(self, why):
        self.invalid_reasons.append(f"{utc()} {why}")


class Q1RefHooks:
    def __init__(self, job_path=None):
        self.job_path = job_path if job_path is not None else os.environ.get(JOB_ENV)
        self.enabled = self.job_path is not None
        self.case = None; self.control_mtime = None; self.fault = None; self.shadow = {}; self.seq = 0; self.boot = None
        if not self.enabled:
            return
        if not os.path.exists(self.job_path):
            raise RuntimeError(f"{JOB_ENV} set but job file missing: {self.job_path}")       # R6: fatal, never silently disabled
        raw = open(self.job_path, "rb").read(); self.job = json.loads(raw); self.job_sha = sha256_bytes(raw)
        exp_sha = os.environ.get("Q1_REF_HOOKS_JOB_SHA256")
        if exp_sha and exp_sha != self.job_sha:
            raise RuntimeError("hooks job sha != Q1_REF_HOOKS_JOB_SHA256")
        for k in ("run_id", "arm", "process", "repeats", "expect_packed_flag", "cases", "out_dir", "control_path", "terminal_token_id", "archive", "fixtures", "layer_coverage_required", "vocab_required"):
            if k not in self.job:
                raise RuntimeError(f"hooks job missing {k}")
        if os.environ.get("Q1_REF_ARM") not in (None, self.job["arm"]) or os.environ.get("Q1_REF_RUN_ID") not in (None, self.job["run_id"]):
            raise RuntimeError("hooks job run/arm identity != container env")
        if os.environ.get("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE") not in (None, self.job["expect_packed_flag"]):
            raise RuntimeError("container packed flag != job expectation")
        self.out_dir = self.job["out_dir"]; self.control_path = self.job["control_path"]
        os.makedirs(os.path.join(self.out_dir, "cases"), exist_ok=True)
        self.store = ContentStore(os.path.join(self.out_dir, "objects"))
        self._write(f"hooks_boot.{self.job['arm']}.p{self.job['process']}.{os.getpid()}.json", {"schema": SCHEMA, "utc": utc(), "job_sha256": self.job_sha, "job": self.job, "pid": os.getpid()})

    # ---------------------------------------------------------------- io
    def _write(self, name, obj):
        p = os.path.join(self.out_dir, name)
        if os.path.exists(p):
            raise FileExistsError(p)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1, allow_nan=False, default=str)
        os.replace(tmp, p); return p

    def _load_control(self):
        if not os.path.exists(self.control_path):
            return None
        st = os.stat(self.control_path)
        if self.control_mtime == (st.st_mtime_ns, st.st_size):
            return None
        self.control_mtime = (st.st_mtime_ns, st.st_size)
        return json.load(open(self.control_path))

    # ------------------------------------------------------------ boot attestation (R1) -- called once from on_pre_forward
    def _fa2_facts(self):
        """Executed-attention facts (overridable for CPU tests; the production path reads the live vLLM/FA2 modules).

        The pinned interface (flash_attn_interface.py 9caca958...) has DEFAULT_FA_VERSION and is_fa_version_supported but NO
        get_flash_attn_version; the supported selector is vllm.v1.attention.backends.fa_utils.get_flash_attn_version(requires_alibi=False,
        head_size=None).  Neither DEFAULT_FA_VERSION nor the global selector proves dispatch: the ACTIVE version is the field each
        FlashAttentionImpl stored at construction (flash_attn.py:630-633) and passes as fa_version (:809/:837); _attest_boot reads it per layer."""
        import vllm
        from vllm.vllm_flash_attn import flash_attn_interface as fai
        so = os.path.join(os.path.dirname(fai.__file__), "_vllm_fa2_C.abi3.so")
        facts = {"installed_so_sha256": hashlib.sha256(open(so, "rb").read()).hexdigest(), "interface_sha256": hashlib.sha256(open(fai.__file__, "rb").read()).hexdigest(),
                 "interface_default_fa_version": getattr(fai, "DEFAULT_FA_VERSION", None), "interface_has_version_getter": hasattr(fai, "get_flash_attn_version"),
                 "fork_symbol_varlen_fwd_tree_bias": hasattr(torch.ops._vllm_fa2_C, "varlen_fwd_tree_bias"), "varlen_fwd": hasattr(torch.ops._vllm_fa2_C, "varlen_fwd"), "vllm": vllm.__version__,
                 "selector": "vllm.v1.attention.backends.fa_utils.get_flash_attn_version", "fa_version_selected": None, "selector_error": None, "fa2_supported": None,
                 "fa_utils_sha256": None, "flash_attn_backend_sha256": None, "attention_layer_sha256": None}
        try:
            from vllm.v1.attention.backends import fa_utils, flash_attn as fa_backend
            facts["fa_utils_sha256"] = hashlib.sha256(open(fa_utils.__file__, "rb").read()).hexdigest()
            facts["flash_attn_backend_sha256"] = hashlib.sha256(open(fa_backend.__file__, "rb").read()).hexdigest()
            try:
                from vllm.model_executor.layers.attention import attention as attn_layer_mod
                facts["attention_layer_sha256"] = hashlib.sha256(open(attn_layer_mod.__file__, "rb").read()).hexdigest()
            except Exception as e:  # noqa: BLE001
                facts["attention_layer_error"] = f"{type(e).__name__}: {e}"
            facts["fa2_supported"] = bool(fai.is_fa_version_supported(2)) if hasattr(fai, "is_fa_version_supported") else None
            v = fa_utils.get_flash_attn_version(requires_alibi=False, head_size=None)
            facts["fa_version_selected"] = int(v) if v is not None else None
        except Exception as e:  # noqa: BLE001
            facts["selector_error"] = f"{type(e).__name__}: {e}"
        return facts

    def _cache_facts(self, runner):
        cc = runner.vllm_config.cache_config; sc = runner.vllm_config.scheduler_config
        return {"mamba_cache_mode": getattr(cc, "mamba_cache_mode", None), "block_size": getattr(cc, "block_size", None), "mamba_block_size": getattr(cc, "mamba_block_size", None),
                "mamba_ssm_cache_dtype": str(getattr(cc, "mamba_ssm_cache_dtype", None)), "cache_dtype": str(getattr(cc, "cache_dtype", None)), "enable_prefix_caching": getattr(cc, "enable_prefix_caching", None),
                "kernel_block_sizes": list(getattr(runner, "_kernel_block_sizes", []) or []),
                "groups": [{"gid": i, "spec": type(g.kv_cache_spec).__name__, "block_size": getattr(g.kv_cache_spec, "block_size", None), "layers": len(g.layer_names),
                            "num_speculative_blocks": getattr(g.kv_cache_spec, "num_speculative_blocks", None), "dtype": str(getattr(g.kv_cache_spec, "dtype", None))} for i, g in enumerate(runner.kv_cache_config.kv_cache_groups)],
                "speculative_config_is_none": runner.vllm_config.speculative_config is None, "async_scheduling": bool(getattr(runner, "use_async_scheduling", getattr(sc, "async_scheduling", None))), "max_num_seqs": int(sc.max_num_seqs)}

    def _attest_boot(self, runner):
        if self.boot is not None:
            return
        b = {"utc": utc(), "problems": []}
        try:
            fa = self._fa2_facts(); b["fa2"] = fa
            exp_fork = self.job.get("fa2_expected_sha256", "28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857")
            if fa["installed_so_sha256"] != exp_fork:
                b["problems"].append("executed FA2 binary is not the pinned fork")
            if fa.get("fa_version_selected") != 2 or not fa["fork_symbol_varlen_fwd_tree_bias"] or fa.get("fa2_supported") is False:
                b["problems"].append(f"FA2 selector/fork symbols not as required (fa_utils selector -> {fa.get('fa_version_selected')!r}, error {fa.get('selector_error')!r}; stock/FA3/FA4/Triton refused)")
            src = self.job.get("fa2_sources") or {}
            for key, fact in (("flash_attn_interface_patched", "interface_sha256"), ("fa_utils", "fa_utils_sha256"), ("flash_attn_backend", "flash_attn_backend_sha256"), ("attention_layer", "attention_layer_sha256")):
                if not src.get(key) or fa.get(fact) != src.get(key):
                    b["problems"].append(f"FA2 source {key}: runtime sha {str(fa.get(fact))[:12]} != pinned {str(src.get(key))[:12]}")
        except Exception as e:  # noqa: BLE001
            b["problems"].append(f"fa2 attestation failed: {type(e).__name__}: {e}")
        try:
            cf = self._cache_facts(runner); b["cache"] = cf
            if cf["mamba_cache_mode"] != "align":
                b["problems"].append("mamba_cache_mode != align")
            if not cf["speculative_config_is_none"] or cf["max_num_seqs"] != 1:
                b["problems"].append("reference must be spec-off with max_num_seqs 1")
        except Exception as e:  # noqa: BLE001
            b["problems"].append(f"cache attestation failed: {type(e).__name__}: {e}")
        ctx = runner.compilation_config.static_forward_context
        gdn = sorted(k for k in ctx if k.endswith(GDN_LAYER_SUFFIX)); attn = sorted(k for k in ctx if k.endswith(ATTN_LAYER_SUFFIX))
        b["layers"] = {"gdn": len(gdn), "attention": len(attn)}
        b["runner_input_path"] = {"supports_mm_inputs": getattr(runner, "supports_mm_inputs", None), "uses_mrope": getattr(runner, "uses_mrope", None),
                                  "has_prepared_input_ids_buffer": hasattr(getattr(runner, "input_ids", None), "gpu"),
                                  "note": "v2.1: input_ids arg is None on the embed path; consumed ids are read from runner.input_ids.gpu at query_start_loc"}
        if len(gdn) != REQUIRED_GDN or len(attn) != REQUIRED_ATTN:
            b["problems"].append(f"layer coverage {len(gdn)}/{len(attn)} != 48/16")
        bad, missing = [], []
        for n in gdn:
            try:
                if packed_flag(ctx[n]) != str(self.job["expect_packed_flag"]):
                    bad.append(n)
            except RuntimeError as e:
                missing.append(f"{n}: {e}")
        if bad or missing:
            b["problems"].append(f"{len(bad)} GDN layers have the wrong packed attribute; {len(missing)} missing/untyped ({missing[:2]})")
        b["packed_attribute_all_layers_match"] = not bad and not missing; b["packed_attribute_missing_or_untyped"] = len(missing)
        impl_bad = []                                                   # actual dispatched implementation per attention layer (not the global selector)
        for n in attn:
            layer = ctx[n]; ab = getattr(layer, "attn_backend", None); impl = getattr(layer, "impl", None)
            bname = ab.get_name() if (ab is not None and hasattr(ab, "get_name")) else None
            icls = type(impl).__name__ if impl is not None else None; imod = type(impl).__module__ if impl is not None else None
            ver = getattr(impl, "vllm_flash_attn_version", None) if (impl is not None and hasattr(impl, "vllm_flash_attn_version")) else None
            if bname != "FLASH_ATTN" or icls != "FlashAttentionImpl" or not str(imod or "").endswith("v1.attention.backends.flash_attn") or ver != 2:
                impl_bad.append({"layer": n, "backend": bname, "impl": f"{imod}.{icls}", "vllm_flash_attn_version": ver})
        b["attention_impl"] = {"required": "attn_backend.get_name()==FLASH_ATTN, impl vllm.v1.attention.backends.flash_attn.FlashAttentionImpl, impl.vllm_flash_attn_version==2", "layers": len(attn), "bad": impl_bad[:4], "all_layers_fa2": not impl_bad}
        if impl_bad:
            b["problems"].append(f"{len(impl_bad)} attention layers do not dispatch FlashAttentionImpl with active vllm_flash_attn_version 2")
        b["fatal"] = bool(b["problems"])
        self.boot = b
        self._write(f"boot_attestation.{self.job['arm']}.p{self.job['process']}.json", b)
        if b["fatal"]:
            self.fault = "boot attestation failed: " + "; ".join(b["problems"])

    # ---------------------------------------------------------------- binding
    def _bind(self, runner):
        ctrl = self._load_control()
        if ctrl is not None:
            if self.case is not None and not self.case.sealed:
                self.case.invalidate("control file changed while the case was still unsealed"); self._seal_case()
            try:
                self.case = CaseState(ctrl, self.job)
            except Exception as e:  # noqa: BLE001
                self._write(f"cases/INVALID_CONTROL.{utc().replace(':', '')}.{sha256_bytes(json.dumps(ctrl, sort_keys=True).encode())[:12]}.json", {"schema": SCHEMA, "control": ctrl, "error": f"{type(e).__name__}: {e}", "utc": utc()})
                self.case = None; return None
        if self.case is None or self.case.sealed:
            return None
        req_ids = list(runner.input_batch.req_ids)
        if len(req_ids) != 1:
            self.case.invalidate(f"{len(req_ids)} requests in batch; instrumentation requires exactly one"); return None
        rid = req_ids[0]
        if self.case.req_id is None:
            st = runner.requests[rid]
            if st.prompt_token_ids is None:
                self.case.invalidate("runtime prompt token ids unavailable (parity gate cannot run)"); return None
            got = sha256_bytes(struct.pack("<%dI" % len(st.prompt_token_ids), *st.prompt_token_ids))
            if st.num_prompt_tokens != self.case.prefix_len or got != self.case.control["prefix_sha256"]:
                self.case.invalidate(f"runtime prompt ids ({st.num_prompt_tokens}, {got[:12]}) != fixture ({self.case.prefix_len}, {self.case.control['prefix_sha256'][:12]})"); return None
            self.case.req_id = rid; self.case.request_active = True
        elif self.case.req_id != rid:
            self.case.invalidate(f"second request {rid} while case bound to {self.case.req_id}"); return None
        return rid

    # ---------------------------------------------------------------- KV extraction (R2)
    def _layer_groups(self, runner):
        m = {}
        for gid, g in enumerate(runner.kv_cache_config.kv_cache_groups):
            for name in g.layer_names:
                m[name] = gid
        return m

    def _attention_kv(self, runner, rid, name, layer, ctx_len, gid_of, blocks_seen):
        gid = gid_of.get(name)
        if gid is None:
            raise RuntimeError(f"{name}: not in any kv cache group")
        bt = runner.input_batch.block_table[gid]
        idx = runner.input_batch.req_id_to_index[rid]
        kbs = int(bt.block_size)                                      # kernel block size after the runner's own conversion
        nblk = int(bt.num_blocks_per_row[idx])
        row = [int(x) for x in bt.get_numpy_array()[idx, :nblk]]
        need = (ctx_len + kbs - 1) // kbs
        if need > nblk or any(b <= 0 for b in row[:need]) or len(set(row[:need])) != need:
            raise RuntimeError(f"{name}: block table does not cover {ctx_len} tokens (have {nblk} kernel blocks, need {need}) or has null/duplicate blocks")
        kvc = layer.kv_cache[0] if isinstance(layer.kv_cache, (list, tuple)) else layer.kv_cache
        if kvc.dim() != 5 or int(kvc.shape[0]) != 2 or int(kvc.shape[2]) != kbs:
            raise RuntimeError(f"{name}: kv cache logical shape {tuple(kvc.shape)} is not (2, num_blocks, {kbs}, kv_heads, head)")
        geom = self.job.get("declared_geometry") or {}
        if not all(k in geom for k in ("kv_dtype", "attn_kv_heads", "head_dim")):
            raise RuntimeError(f"{name}: job declares no attention KV geometry (kv_dtype / attn_kv_heads / head_dim required)")
        if str(kvc.dtype) != str(geom["kv_dtype"]) or int(kvc.shape[3]) != int(geom["attn_kv_heads"]) or int(kvc.shape[4]) != int(geom["head_dim"]):
            raise RuntimeError(f"{name}: kv cache {kvc.dtype} x (kv_heads {int(kvc.shape[3])}, head {int(kvc.shape[4])}) != declared {geom['kv_dtype']} x ({geom['attn_kv_heads']}, {geom['head_dim']})")
        rec = {"group": gid, "kernel_block_size": kbs, "shape": list(kvc.shape), "dtype": str(kvc.dtype), "stride": list(kvc.stride()), "storage_offset": int(kvc.storage_offset()), "ptr": kvc.data_ptr(),
               "materialized_tokens": ctx_len, "full_blocks": [], "tail": None, "canonical_order": "token-major: block i covers tokens [i*kbs, (i+1)*kbs); K then V per block; tail = first (ctx_len mod kbs) slots of the last block"}
        full, tail_n = ctx_len // kbs, ctx_len % kbs
        digest = hashlib.sha256()
        for i in range(full):
            b = row[i]
            if b >= int(kvc.shape[1]):
                raise RuntimeError(f"{name}: block id {b} out of range")
            key = (name, b, "full")
            if key not in blocks_seen:
                finite_or_raise(kvc[0, b], f"K in {name} group {gid} physical block {b} (logical {i}, full)"); finite_or_raise(kvc[1, b], f"V in {name} group {gid} physical block {b} (logical {i}, full)")
                kb = tensor_bytes(kvc[0, b]); vb = tensor_bytes(kvc[1, b])
                blocks_seen[key] = (self.store.put(kb, {"kind": "kv_block_k"}), self.store.put(vb, {"kind": "kv_block_v"}))
            k_ref, v_ref = blocks_seen[key]
            rec["full_blocks"].append({"logical_index": i, "physical_block": b, "k": k_ref["sha256"], "v": v_ref["sha256"], "bytes": k_ref["bytes"] + v_ref["bytes"]})
            digest.update(bytes.fromhex(k_ref["sha256"])); digest.update(bytes.fromhex(v_ref["sha256"]))
        if tail_n:
            b = row[full]
            if b >= int(kvc.shape[1]):
                raise RuntimeError(f"{name}: tail block id {b} out of range")
            finite_or_raise(kvc[0, b, :tail_n], f"K in {name} group {gid} physical block {b} (logical {full}, tail {tail_n} valid tokens)"); finite_or_raise(kvc[1, b, :tail_n], f"V in {name} group {gid} physical block {b} (logical {full}, tail {tail_n} valid tokens)")
            kb = tensor_bytes(kvc[0, b, :tail_n]); vb = tensor_bytes(kvc[1, b, :tail_n])
            k_ref = self.store.put(kb, {"kind": "kv_tail_k"}); v_ref = self.store.put(vb, {"kind": "kv_tail_v"})
            rec["tail"] = {"logical_index": full, "physical_block": b, "valid_tokens": tail_n, "k": k_ref["sha256"], "v": v_ref["sha256"], "bytes": k_ref["bytes"] + v_ref["bytes"]}
            digest.update(bytes.fromhex(k_ref["sha256"])); digest.update(bytes.fromhex(v_ref["sha256"]))
        rec["logical_digest"] = digest.hexdigest()
        rec["checked"] = {"declared_geometry": dict(geom), "finite_full_blocks": full, "finite_tail_tokens": tail_n, "unmaterialized_slots_ignored": (kbs - tail_n) if tail_n else 0}
        return rec

    def _snapshot_state(self, runner, rid, attn_metadata, tag, ctx_len):
        ctx = runner.compilation_config.static_forward_context
        gdn = sorted(k for k in ctx if k.endswith(GDN_LAYER_SUFFIX)); attn = sorted(k for k in ctx if k.endswith(ATTN_LAYER_SUFFIX))
        if len(gdn) != REQUIRED_GDN or len(attn) != REQUIRED_ATTN:
            raise RuntimeError(f"layer coverage {len(gdn)}/{len(attn)} != 48/16")
        idx = runner.input_batch.req_id_to_index[rid]
        gid_of = self._layer_groups(runner)
        rec = {"tag": tag, "utc": utc(), "obs_id": self.case.obs_id, "materialized_tokens": ctx_len, "gdn": {}, "attention": {}}
        conv_all, ssm_all = hashlib.sha256(), hashlib.sha256()
        for name in gdn:
            layer = ctx[name]; md = attn_metadata.get(name) if isinstance(attn_metadata, dict) else None
            if md is None or getattr(md, "spec_sequence_masks", None) is not None:
                raise RuntimeError(f"{name}: no non-spec GDN metadata")
            sidx = md.non_spec_state_indices_tensor
            row = int(sidx[idx].item()) if sidx is not None else None
            if row is None or row <= 0:
                raise RuntimeError(f"{name}: invalid native state row {row}")
            if packed_flag(layer) != str(self.job["expect_packed_flag"]):                      # raises on a missing/untyped attribute
                raise RuntimeError(f"{name}: packed attribute != arm")
            kv = layer.kv_cache[0] if isinstance(layer.kv_cache, (list, tuple)) and len(layer.kv_cache) and isinstance(layer.kv_cache[0], (list, tuple)) else layer.kv_cache
            conv_state, ssm_state = kv[0], kv[1]
            sh = self.shadow.get(name)
            if sh is None:
                sh = {"conv": torch.empty_like(conv_state[row]), "ssm": torch.empty_like(ssm_state[row])}; self.shadow[name] = sh
            sh["conv"].copy_(conv_state[row]); sh["ssm"].copy_(ssm_state[row])            # R-SNAP into once-allocated shadows
            cb, sb = tensor_bytes(sh["conv"]), tensor_bytes(sh["ssm"])
            if not (torch.isfinite(sh["ssm"].float()).all().item() and torch.isfinite(sh["conv"].float()).all().item()):
                raise RuntimeError(f"{name}: nonfinite recurrent/conv state")
            cref = self.store.put(cb, {"kind": "gdn_conv_row"}); sref = self.store.put(sb, {"kind": "gdn_ssm_row"})
            conv_all.update(bytes.fromhex(cref["sha256"])); ssm_all.update(bytes.fromhex(sref["sha256"]))
            rec["gdn"][name] = {"group": gid_of.get(name), "state_row": row, "mamba_state_idx_runner": (runner.mamba_state_idx.get(rid) if hasattr(runner, "mamba_state_idx") else None),
                                "conv": {"sha256": cref["sha256"], "shape": list(sh["conv"].shape), "dtype": str(sh["conv"].dtype), "stride": list(conv_state.stride()), "storage_offset": int(conv_state.storage_offset()), "ptr": conv_state.data_ptr()},
                                "ssm": {"sha256": sref["sha256"], "shape": list(sh["ssm"].shape), "dtype": str(sh["ssm"].dtype), "stride": list(ssm_state.stride()), "storage_offset": int(ssm_state.storage_offset()), "ptr": ssm_state.data_ptr()}}
        blocks_seen = {}
        kv_all = hashlib.sha256()
        for name in attn:
            r = self._attention_kv(runner, rid, name, ctx[name], ctx_len, gid_of, blocks_seen); rec["attention"][name] = r; kv_all.update(bytes.fromhex(r["logical_digest"]))
        rec["conv_all_sha256"] = conv_all.hexdigest(); rec["ssm_all_sha256"] = ssm_all.hexdigest(); rec["attn_kv_all_sha256"] = kv_all.hexdigest()
        rec["logical_digest"] = hashlib.sha256((rec["conv_all_sha256"] + rec["ssm_all_sha256"] + rec["attn_kv_all_sha256"]).encode()).hexdigest()
        rec["objects_written_so_far"] = self.store.written; rec["objects_reused_so_far"] = self.store.reused
        return rec

    # ---------------------------------------------------------------- hooks
    def on_pre_forward(self, runner, scheduler_output, attn_metadata, input_ids, positions, logits_indices):
        if not self.enabled:
            return
        try:
            self._attest_boot(runner)
            rid = self._bind(runner)
            if self.fault:                                                   # fatal boot attestation: every admitted case is sealed INVALID with the fault (never pending)
                if self.case is not None and not self.case.sealed:
                    self.case.invalidate(self.fault); self._seal_case()
                return
            if self.case is not None and not self.case.sealed and self.case.invalid_reasons:
                self._seal_case(); return                                    # R5: a binding failure is sealed immediately, never left pending
            if rid is None or self.case is None or self.case.sealed:
                return
            st = runner.requests[rid]; ncomp = int(st.num_computed_tokens); n_sched = int(scheduler_output.num_scheduled_tokens[rid]); self.seq += 1
            if n_sched == 1:
                idx = runner.input_batch.req_id_to_index[rid]
                token_source = "input_ids_arg"
                if input_ids is None:                                        # v2.1: embed path (multimodal-capable model) passes input_ids=None
                    buf = getattr(getattr(runner, "input_ids", None), "gpu", None)
                    if buf is None:
                        self.case.invalidate("prepared token ids unavailable at the pre-forward boundary (input_ids arg None and runner.input_ids.gpu missing)"); self._seal_case(); return
                    input_ids = buf; token_source = "runner.input_ids.gpu (embed path; input_ids arg None)"
                if positions is None:
                    self.case.invalidate("prepared positions unavailable at the pre-forward boundary"); self._seal_case(); return
                qsl = int(runner.query_start_loc.np[idx])
                tok = int(input_ids[qsl].item()); pos = int(positions[qsl].item()) if positions.dim() == 1 else int(positions[0, qsl].item())
                self.case.consumed_trace.append({"num_computed_before": ncomp, "token": tok, "position": pos, "seq": self.seq, "token_source": token_source, "positions_shape": list(positions.shape)})
                k = ncomp - self.case.prefix_len
                if 0 <= k < len(self.case.chain) and (tok != self.case.chain[k] or pos != self.case.positions[k]):
                    self.case.invalidate(f"consumed (token {tok}, pos {pos}) at chain step {k} != fixture ({self.case.chain[k]}, {self.case.positions[k]})"); self._seal_case(); return
                if ncomp == self.case.prefix_len:
                    self.case.o0 = self._snapshot_state(runner, rid, attn_metadata, "O0", ncomp)
                elif ncomp == self.case.z_step_num_computed:
                    self.case.o1 = self._snapshot_state(runner, rid, attn_metadata, "O1", ncomp)
            else:
                self.case.consumed_trace.append({"num_computed_before": ncomp, "prefill_tokens": n_sched, "seq": self.seq, "apc_hit_tokens": ncomp})
        except Exception as e:  # noqa: BLE001
            if self.case is not None and not self.case.sealed:
                self.case.invalidate(f"on_pre_forward: {type(e).__name__}: {e}"); self._seal_case()
            else:
                self.fault = f"on_pre_forward: {type(e).__name__}: {e}"

    def on_logits(self, runner, logits, logits_indices, grammar_output=None):
        if not self.enabled or self.case is None or self.case.sealed or self.case.req_id is None:
            return
        try:
            rid = self.case.req_id; st = runner.requests[rid]
            if int(st.num_computed_tokens) != self.case.z_step_num_computed:
                return
            idx = runner.input_batch.req_id_to_index[rid]
            row = logits[idx].detach()
            if row.is_cuda:
                torch.cuda.current_stream(row.device).synchronize()
            row32 = row.to(torch.float32).cpu().contiguous(); b = row32.numpy().tobytes()
            finite = bool(torch.isfinite(row32).all().item())
            if int(row32.numel()) != int(self.job["vocab_required"]):
                self.case.invalidate(f"logits row has {row32.numel()} entries != required vocab {self.job['vocab_required']}"); self._seal_case(); return
            mx = row32.max().item() if finite else None
            arg = int(torch.nonzero(row32 == mx).min().item()) if finite else None
            second = None
            if finite:
                s = row32.clone(); s[arg] = -float("inf"); second = s.max().item()
            ref = self.store.put(b, {"kind": "o2_logits_f32"})
            self.case.o2 = {"tag": "O2", "utc": utc(), "obs_id": self.case.obs_id, "num_computed_before": int(st.num_computed_tokens), "logits_dtype": str(logits.dtype), "vocab": int(row32.numel()), "all_finite": finite,
                            "argmax_smallest_id": arg, "top1": mx, "top2": second, "margin": (mx - second) if (finite and second is not None) else None, "exact_tie_count": int((row32 == mx).sum().item()) if finite else None,
                            "raw": ref}
            if not finite:
                self.case.invalidate("nonfinite O2 logits")
        except Exception as e:  # noqa: BLE001
            self.case.invalidate(f"on_logits: {type(e).__name__}: {e}"); self._seal_case()

    def on_grammar(self, runner, grammar_output):
        if not self.enabled or self.case is None or self.case.sealed or self.case.req_id is None:
            return
        if grammar_output is not None:
            self.case.invalidate("grammar_output present; raw greedy protocol forbids grammar masks"); self._seal_case()

    def on_sampled(self, runner, sampler_output, scheduler_output, grammar_output=None):
        if not self.enabled or self.case is None or self.case.sealed or self.case.req_id is None:
            return
        try:
            if grammar_output is not None:
                self.case.invalidate("grammar_output present before sampling"); self._seal_case(); return
            rid = self.case.req_id; idx = runner.input_batch.req_id_to_index[rid]; st = runner.requests[rid]
            after = int(st.num_computed_tokens) + int(scheduler_output.num_scheduled_tokens[rid])
            if after < self.case.prefix_len:
                return
            k = after - self.case.prefix_len
            ids = sampler_output.sampled_token_ids
            self.case.natural_sampled.append({"after_num_computed": after, "natural": int(ids[idx, 0].item())})
            forced = self.case.chain[k] if k < len(self.case.chain) else int(self.job["terminal_token_id"])
            ids[idx, 0].copy_(torch.tensor(forced, dtype=ids.dtype, device=ids.device))
            self.case.forced += 1
            if k >= len(self.case.chain):
                self.case.request_active = False; self._seal_case()
        except Exception as e:  # noqa: BLE001
            self.case.invalidate(f"on_sampled: {type(e).__name__}: {e}"); self._seal_case()

    # ---------------------------------------------------------------- sealing (R5)
    def _mandatory_observation_problems(self, c):
        pr = []
        if c.o0 is None: pr.append("O0 missing")
        if c.o1 is None: pr.append("O1 missing")
        if c.o2 is None: pr.append("O2 missing")
        toks = [t["token"] for t in c.consumed_trace if "token" in t]
        if toks != c.chain: pr.append(f"consumed trace {toks[:6]}.. != chain")
        poss = [t["position"] for t in c.consumed_trace if "position" in t]
        if poss != c.positions: pr.append("consumed positions != chain positions")
        if c.o0 is not None and (len(c.o0["gdn"]) != REQUIRED_GDN or len(c.o0["attention"]) != REQUIRED_ATTN): pr.append("O0 layer coverage incomplete")
        if c.o1 is not None and (len(c.o1["gdn"]) != REQUIRED_GDN or len(c.o1["attention"]) != REQUIRED_ATTN): pr.append("O1 layer coverage incomplete")
        if c.o2 is not None and not c.o2["all_finite"]: pr.append("O2 nonfinite")
        if c.o0 is not None and c.o0["materialized_tokens"] != c.prefix_len: pr.append("O0 extent != |P|")
        if c.o1 is not None and c.o1["materialized_tokens"] != c.z_step_num_computed: pr.append("O1 extent != |P|+1+L")
        return pr

    def _seal_case(self):
        c = self.case
        if c is None or c.sealed:
            return
        problems = list(c.invalid_reasons) + self._mandatory_observation_problems(c)
        doc = {"schema": SCHEMA, "obs_id": c.obs_id, "case_id": c.case_id, "run_id": self.job["run_id"], "arm": self.job["arm"], "process": self.job["process"], "repeat": c.control["repeat"],
               "control": c.control, "req_id": c.req_id, "request_still_active_at_seal": bool(c.request_active and problems), "valid": not problems, "problems": problems,
               "o0": c.o0, "o1": c.o1, "o2": c.o2, "consumed_trace": c.consumed_trace, "natural_sampled": c.natural_sampled, "forced_count": c.forced, "job_sha256": self.job_sha,
               "boot_attestation_ref": f"boot_attestation.{self.job['arm']}.p{self.job['process']}.json", "sealed_utc": utc(), "pid": os.getpid()}
        body = dict(doc); body["record_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        self._write(f"cases/{c.obs_id}.json", body)
        c.sealed = True


def reconstruct_logical_kv(rec_attention_layer, objects_root, verify=True):
    """CPU round-trip: rebuild the logical K/V byte stream for one attention layer from its manifest (full blocks + valid tail).
    With verify=True every object is hash + exact-length authenticated (length derived from the recorded geometry) before use."""
    a = rec_attention_layer
    per_tok = int(a["shape"][3]) * int(a["shape"][4]) * DTYPE_BYTES[a["dtype"]]; kbs = int(a["kernel_block_size"])
    def read(s, n):
        p = os.path.join(objects_root, s + ".bin"); b = open(p, "rb").read()
        if verify and (len(b) != n or hashlib.sha256(b).hexdigest() != s):
            raise RuntimeError(f"object {s[:12]}: length {len(b)} / sha mismatch vs expected {n} bytes")
        return b
    out_k, out_v = bytearray(), bytearray()
    for blk in a["full_blocks"]:
        out_k += read(blk["k"], kbs * per_tok); out_v += read(blk["v"], kbs * per_tok)
    t = a.get("tail")
    if t:
        out_k += read(t["k"], int(t["valid_tokens"]) * per_tok); out_v += read(t["v"], int(t["valid_tokens"]) * per_tok)
    return bytes(out_k), bytes(out_v)

_INSTANCE = None


def instance():
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = Q1RefHooks()
    return _INSTANCE
