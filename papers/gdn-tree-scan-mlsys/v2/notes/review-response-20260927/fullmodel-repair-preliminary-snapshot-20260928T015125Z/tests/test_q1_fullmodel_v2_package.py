"""CPU tests for the full-model package v2: hooks v2 (KV extraction by cache group, repeat-unique content-addressed archive, prepared-input
verification, mandatory-observation sealing, fatal job/boot rules), driver v2 (authenticated seals, stop rules, verdict), engine config v2
(patched FA2 both arms, real mounts, job binding), job builder (smoke scope). No GPU, no model, no vllm import."""
import argparse, copy, hashlib, json, os, shutil, struct, sys, tempfile, types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); TOOLS = os.path.dirname(HERE); CAMP = os.path.dirname(TOOLS); sys.path.insert(0, TOOLS)
import q1_reference_hooks_v2 as H  # noqa: E402
import q1_reference_driver_v2 as DRV  # noqa: E402
import q1_spec_off_engine_config_v2 as CFG  # noqa: E402
import q1_reference_job as JOB  # noqa: E402
import q1_patch_reference_runner_v2 as P2  # noqa: E402

FIXTURES = os.path.join(CAMP, "fullmodel", "fixtures", "token-fixtures.v1.json")
FORK_SHA = "28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857"
NG, NA, KBS, NB = 48, 16, 8, 6           # tiny cache: kernel block 8 tokens, 6 blocks; 4 kv heads x 4 head dim for speed


# ------------------------------------------------------------------ fake stock runner surface (only what the hooks read)
class _Req:
    def __init__(self, ids, ncomp):
        self.prompt_token_ids = list(ids); self.num_prompt_tokens = len(ids); self.num_computed_tokens = ncomp; self.output_token_ids = []


class _BT:
    def __init__(self, block_size, rows):
        self.block_size = block_size; self.num_blocks_per_row = [len(r) for r in rows]; self._np = torch.zeros((len(rows), 16), dtype=torch.int32).numpy()
        for i, r in enumerate(rows):
            self._np[i, :len(r)] = r
    def get_numpy_array(self): return self._np


class _Layer:
    def __init__(self, kv, packed=None):
        self.kv_cache = [kv]
        if packed is not None:
            self.enable_packed_recurrent_decode = packed


class FlashAttentionImpl:              # named + module-tagged like the pinned vllm.v1.attention.backends.flash_attn.FlashAttentionImpl
    def __init__(self, version=2):
        self.vllm_flash_attn_version = version
FlashAttentionImpl.__module__ = "vllm.v1.attention.backends.flash_attn"


class _Backend:
    def __init__(self, name="FLASH_ATTN"): self._n = name
    def get_name(self): return self._n


class _AttnLayer(_Layer):
    def __init__(self, kv):
        super().__init__(kv); self.impl = FlashAttentionImpl(); self.attn_backend = _Backend()


class _MD:
    def __init__(self, row):
        self.spec_sequence_masks = None; self.non_spec_state_indices_tensor = torch.tensor([row], dtype=torch.int32); self.num_prefills = 0; self.num_decodes = 1


class FakeRunner:
    def __init__(self, prefix, packed=False):
        rid = "req-1"; self.rid = rid
        self.requests = {rid: _Req(prefix, 0)}
        self.input_batch = types.SimpleNamespace(req_ids=[rid], req_id_to_index={rid: 0}, block_table=None)
        ctx = {}
        self.gdn_conv = [torch.randn(4, 6, 3, dtype=torch.bfloat16) for _ in range(NG)]; self.gdn_ssm = [torch.randn(4, 3, 4, 4) * (i + 1) for i in range(NG)]
        for i in range(NG):
            ctx[f"language_model.model.layers.{i}.linear_attn"] = _Layer((self.gdn_conv[i], self.gdn_ssm[i]), packed)
        self.attn_kv = [torch.randn(2, NB, KBS, 4, 4, dtype=torch.bfloat16) for _ in range(NA)]
        for i in range(NA):
            ctx[f"language_model.model.layers.{100 + i}.self_attn.attn"] = _AttnLayer(self.attn_kv[i])
        self.compilation_config = types.SimpleNamespace(static_forward_context=ctx)
        class FullAttentionSpec:  # noqa: D401 -- named like the vLLM spec so the attestation records the group kind
            block_size, num_speculative_blocks, dtype = KBS, 0, "bf16"
        class MambaSpec:
            block_size, num_speculative_blocks, dtype = KBS, 0, "fp32"
        g0 = types.SimpleNamespace(kv_cache_spec=FullAttentionSpec(), layer_names=[f"language_model.model.layers.{100 + i}.self_attn.attn" for i in range(NA)])
        g1 = types.SimpleNamespace(kv_cache_spec=MambaSpec(), layer_names=[f"language_model.model.layers.{i}.linear_attn" for i in range(NG)])
        self.kv_cache_config = types.SimpleNamespace(kv_cache_groups=[g0, g1]); self._kernel_block_sizes = [KBS, KBS]
        self.vllm_config = types.SimpleNamespace(cache_config=types.SimpleNamespace(mamba_cache_mode="align", block_size=KBS, mamba_block_size=KBS, mamba_ssm_cache_dtype="float32", cache_dtype="auto", enable_prefix_caching=True),
                                                 scheduler_config=types.SimpleNamespace(max_num_seqs=1, async_scheduling=False), speculative_config=None)
        self.use_async_scheduling = False; self.mamba_state_idx = {rid: 1}
        self.query_start_loc = types.SimpleNamespace(np=[0, 1])
        self.set_blocks(list(range(1, NB)))   # blocks 1..5 belong to the request
        self.md = {n: _MD(1) for n in ctx if n.endswith(".linear_attn")}

    def set_blocks(self, blocks):
        self.input_batch.block_table = [_BT(KBS, [blocks]), _BT(KBS, [[1]])]


FA2_SRC = {"flash_attn_interface_stock": "s" * 64, "flash_attn_interface_patched": "i" * 64, "fa_utils": "u" * 64, "flash_attn_backend": "b" * 64, "attention_layer": "l" * 64, "fa2_patcher": "p" * 64, "fork_binary": FORK_SHA}
GEOM = {"attn_kv_heads": 4, "head_dim": 4, "kv_dtype": "torch.bfloat16", "ssm_dtype": "torch.float32", "conv_dtype": "torch.bfloat16"}


class CpuHooks(H.Q1RefHooks):
    """Production hooks with synthetic FA2 facts (no vllm on the CPU host); cache facts come from the fake runner."""
    fa2 = {"installed_so_sha256": FORK_SHA, "interface_sha256": "i" * 64, "fa_version_selected": 2, "fork_symbol_varlen_fwd_tree_bias": True, "varlen_fwd": True, "vllm": "stub",
           "fa_utils_sha256": "u" * 64, "flash_attn_backend_sha256": "b" * 64, "attention_layer_sha256": "l" * 64, "fa2_supported": True, "selector_error": None, "interface_has_version_getter": False}

    def _fa2_facts(self):
        return dict(self.fa2)


def make_job(tmp, prefix, chain, repeats=2, **over):
    fx = json.load(open(FIXTURES))
    case = {"case_id": "case-x", "prefix_id": "px", "block": "calibration", "path_id": "root-only", "accepted_len": len(chain) - 2, "chain_tokens": chain, "chain_positions": list(range(len(prefix), len(prefix) + len(chain))), "z_position": len(prefix) + len(chain) - 1, "chain_sha256": "x"}
    job = {"schema": JOB.SCHEMA, "run_id": "RUN-T", "arm": "aligned_nonpacked", "process": "A", "repeats": repeats, "expect_packed_flag": "0", "smoke": True,
           "fixtures": {"path": os.path.relpath(FIXTURES, CAMP), "sha256": hashlib.sha256(open(FIXTURES, "rb").read()).hexdigest(), "canonical": fx["canonical_sha256_excluding_timestamp"]},
           "cases": [case], "prefixes": {"px": {"prefix_len": len(prefix), "token_ids_sha256": hashlib.sha256(struct.pack("<%dI" % len(prefix), *prefix)).hexdigest(), "token_ids_u32le": "px.u32le"}},
           "expected_requests": repeats, "out_dir": os.path.join(tmp, "hooks"), "control_path": os.path.join(tmp, "control.json"), "terminal_token_id": 7,
           "archive": {"save_state_bytes": True, "save_kv_bytes": True}, "layer_coverage_required": {"gdn": 48, "attention": 16}, "vocab_required": 16, "fa2_expected_sha256": FORK_SHA,
           "declared_geometry": dict(GEOM), "fa2_sources": dict(FA2_SRC)}
    job.update(over)
    jp = os.path.join(tmp, "job.json"); json.dump(job, open(jp, "w")); return jp, job


def control(job, prefix, chain, repeat=0, **over):
    c = {"case_id": "case-x", "chain_tokens": chain, "chain_positions": list(range(len(prefix), len(prefix) + len(chain))), "prefix_len": len(prefix), "prefix_sha256": job["prefixes"]["px"]["token_ids_sha256"],
         "arm": "aligned_nonpacked", "repeat": repeat, "process": "A", "run_id": "RUN-T"}
    c.update(over); json.dump(c, open(job["control_path"], "w"))


def drive(hooks, runner, prefix, chain, wrong_token_at=None, wrong_pos_at=None, grammar=None, logits_nan=False, mutate_kv_before_o1=False):
    """Emulate the stock runner's per-step calls: one prefill step for P, then one decode step per chain token; forcing at each sampling."""
    rid = runner.rid; runner.requests[rid].num_computed_tokens = 0
    so = types.SimpleNamespace(num_scheduled_tokens={rid: len(prefix)})
    hooks.on_pre_forward(runner, so, runner.md, torch.tensor(prefix), torch.arange(len(prefix)), torch.tensor([len(prefix) - 1]))
    logits = torch.zeros((1, 16)); logits[0, 3] = 5.0; hooks.on_logits(runner, logits, torch.tensor([0])); hooks.on_grammar(runner, grammar)
    ids = torch.tensor([[3]]); hooks.on_sampled(runner, types.SimpleNamespace(sampled_token_ids=ids), so, grammar); forced = [int(ids[0, 0])]
    runner.requests[rid].num_computed_tokens = len(prefix)
    for step in range(len(chain)):
        ncomp = runner.requests[rid].num_computed_tokens; so = types.SimpleNamespace(num_scheduled_tokens={rid: 1})
        tok = chain[step] if step != wrong_token_at else 99999; pos = ncomp if step != wrong_pos_at else ncomp + 5
        if mutate_kv_before_o1 and step == len(chain) - 1:
            runner.attn_kv[0][0, 1, 0].add_(1.0)
        hooks.on_pre_forward(runner, so, runner.md, torch.tensor([tok]), torch.tensor([pos]), torch.tensor([0]))
        logits = torch.zeros((1, 16)); logits[0, 5] = 2.0; logits[0, 9] = 2.0
        if logits_nan and step == len(chain) - 1:
            logits[0, 1] = float("nan")
        hooks.on_logits(runner, logits, torch.tensor([0])); hooks.on_grammar(runner, grammar)
        ids = torch.tensor([[9]]); hooks.on_sampled(runner, types.SimpleNamespace(sampled_token_ids=ids), so, grammar); forced.append(int(ids[0, 0]))
        runner.requests[rid].num_computed_tokens = ncomp + 1
    return forced


@pytest.fixture
def env(tmp_path):
    prefix = list(range(100, 100 + 21)); chain = [3, 22]      # |P| = 21 -> O0: 2 full blocks + 5-token tail; root-only chain [r, z]
    jp, job = make_job(str(tmp_path), prefix, chain)
    return {"tmp": str(tmp_path), "prefix": prefix, "chain": chain, "jp": jp, "job": job}


def test_full_chain_valid_seal_kv_extent_and_reconstruction(env):
    control(env["job"], env["prefix"], env["chain"], repeat=0)
    hooks = CpuHooks(env["jp"]); runner = FakeRunner(env["prefix"])
    forced = drive(hooks, runner, env["prefix"], env["chain"]); assert forced == env["chain"] + [7]
    docs = sorted(os.listdir(os.path.join(env["job"]["out_dir"], "cases"))); assert docs == ["RUN-T__aligned_nonpacked__pA__r0__case-x.json"]
    d = json.load(open(os.path.join(env["job"]["out_dir"], "cases", docs[0])))
    assert d["valid"] is True and d["problems"] == [] and d["o2"]["argmax_smallest_id"] == 5 and d["o2"]["exact_tie_count"] == 2
    assert len(d["o0"]["gdn"]) == 48 and len(d["o0"]["attention"]) == 16 and d["o0"]["materialized_tokens"] == 21 and d["o1"]["materialized_tokens"] == 22
    a0 = d["o0"]["attention"]["language_model.model.layers.100.self_attn.attn"]
    assert [b["physical_block"] for b in a0["full_blocks"]] == [1, 2] and a0["tail"]["physical_block"] == 3 and a0["tail"]["valid_tokens"] == 5 and a0["group"] == 0 and a0["kernel_block_size"] == KBS
    a1 = d["o1"]["attention"]["language_model.model.layers.100.self_attn.attn"]; assert a1["tail"]["valid_tokens"] == 6 and a1["full_blocks"][0]["k"] == a0["full_blocks"][0]["k"]   # shared full blocks, new tail
    k, v = H.reconstruct_logical_kv(a0, os.path.join(env["job"]["out_dir"], "objects"))
    exp_k = b"".join(H.tensor_bytes(runner.attn_kv[0][0, b]) for b in (1, 2)) + H.tensor_bytes(runner.attn_kv[0][0, 3, :5])
    assert k == exp_k and len(v) == len(exp_k)
    assert [t["token"] for t in d["consumed_trace"] if "token" in t] == env["chain"] and [t["position"] for t in d["consumed_trace"] if "position" in t] == [21, 22]
    boot = json.load(open(os.path.join(env["job"]["out_dir"], "boot_attestation.aligned_nonpacked.pA.json"))); assert boot["fatal"] is False and boot["packed_attribute_all_layers_match"]
    assert boot["attention_impl"]["all_layers_fa2"] is True and boot["packed_attribute_missing_or_untyped"] == 0 and a0["checked"]["finite_tail_tokens"] == 5 and a0["checked"]["unmaterialized_slots_ignored"] == 3


def test_second_repeat_has_unique_identity_and_dedups_objects(env):
    control(env["job"], env["prefix"], env["chain"], repeat=0); hooks = CpuHooks(env["jp"]); runner = FakeRunner(env["prefix"]); drive(hooks, runner, env["prefix"], env["chain"])
    w0 = hooks.store.written
    control(env["job"], env["prefix"], env["chain"], repeat=1); runner2 = FakeRunner(env["prefix"])
    for i in range(NA): runner2.attn_kv[i].copy_(runner.attn_kv[i])
    for i in range(NG): runner2.gdn_conv[i].copy_(runner.gdn_conv[i]); runner2.gdn_ssm[i].copy_(runner.gdn_ssm[i])
    drive(hooks, runner2, env["prefix"], env["chain"])
    docs = sorted(os.listdir(os.path.join(env["job"]["out_dir"], "cases"))); assert len(docs) == 2 and docs[1].endswith("__r1__case-x.json")
    d0, d1 = (json.load(open(os.path.join(env["job"]["out_dir"], "cases", x))) for x in docs)
    assert d0["o0"]["logical_digest"] == d1["o0"]["logical_digest"] and hooks.store.written == w0 + 0 and hooks.store.reused > 0


@pytest.mark.parametrize("fault", ["wrong_token", "wrong_position", "grammar", "nan_logits", "missing_prompt_ids", "wrong_prompt_sha", "control_wrong_run", "packed_mismatch", "fa2_not_fork", "layer_missing", "block_table_short"])
def test_faults_seal_invalid_exactly_once(env, fault):
    job = env["job"]; jp = env["jp"]
    if fault == "control_wrong_run":
        control(job, env["prefix"], env["chain"], run_id="OTHER")
    else:
        control(job, env["prefix"], env["chain"], **({"prefix_sha256": "0" * 64} if fault == "wrong_prompt_sha" else {}))
    hooks = CpuHooks(jp)
    if fault == "fa2_not_fork":
        hooks.fa2 = dict(CpuHooks.fa2, installed_so_sha256="d5c5c5de" + "0" * 56)
    runner = FakeRunner(env["prefix"], packed=(True if fault == "packed_mismatch" else False))
    if fault == "missing_prompt_ids":
        runner.requests[runner.rid].prompt_token_ids = None
    if fault == "layer_missing":
        runner.compilation_config.static_forward_context.pop("language_model.model.layers.5.linear_attn")
    if fault == "block_table_short":
        runner.set_blocks([1, 2])
    drive(hooks, runner, env["prefix"], env["chain"], wrong_token_at=(1 if fault == "wrong_token" else None), wrong_pos_at=(1 if fault == "wrong_position" else None),
          grammar=(object() if fault == "grammar" else None), logits_nan=(fault == "nan_logits"))
    cases_dir = os.path.join(job["out_dir"], "cases"); docs = os.listdir(cases_dir) if os.path.exists(cases_dir) else []
    if fault == "control_wrong_run":
        assert docs and all(x.startswith("INVALID_CONTROL") for x in docs); return
    assert len(docs) == 1, docs
    d = json.load(open(os.path.join(cases_dir, docs[0]))); assert d["valid"] is False and d["problems"], fault
    verdict_path = os.path.join(env["tmp"], "drv")


def test_job_missing_or_sha_mismatch_is_fatal(env, monkeypatch):
    with pytest.raises(RuntimeError, match="job file missing"):
        H.Q1RefHooks(os.path.join(env["tmp"], "nope.json"))
    monkeypatch.setenv("Q1_REF_HOOKS_JOB_SHA256", "f" * 64)
    with pytest.raises(RuntimeError, match="job sha"):
        H.Q1RefHooks(env["jp"])
    monkeypatch.delenv("Q1_REF_HOOKS_JOB_SHA256")
    monkeypatch.setenv("Q1_REF_ARM", "native_default_packed")
    with pytest.raises(RuntimeError, match="identity"):
        H.Q1RefHooks(env["jp"])
    monkeypatch.delenv("Q1_REF_ARM"); monkeypatch.delenv(H.JOB_ENV, raising=False)
    assert H.Q1RefHooks(None).enabled is False        # no instrumentation requested -> inert


def test_control_change_seals_unsealed_case_invalid_first(env):
    control(env["job"], env["prefix"], env["chain"], repeat=0); hooks = CpuHooks(env["jp"]); runner = FakeRunner(env["prefix"])
    so = types.SimpleNamespace(num_scheduled_tokens={runner.rid: len(env["prefix"])})
    hooks.on_pre_forward(runner, so, runner.md, torch.tensor(env["prefix"]), torch.arange(len(env["prefix"])), torch.tensor([0]))   # bound, unsealed
    import time; time.sleep(0.01); control(env["job"], env["prefix"], env["chain"], repeat=1)
    hooks.on_pre_forward(runner, so, runner.md, torch.tensor(env["prefix"]), torch.arange(len(env["prefix"])), torch.tensor([0]))
    docs = sorted(os.listdir(os.path.join(env["job"]["out_dir"], "cases"))); assert docs == ["RUN-T__aligned_nonpacked__pA__r0__case-x.json"]
    d = json.load(open(os.path.join(env["job"]["out_dir"], "cases", docs[0]))); assert d["valid"] is False and any("control file changed" in p for p in d["problems"])


# ------------------------------------------------------------------ driver v2
def _fake_engine(env, hooks_factory):
    """Transport stub that runs the hooks against a fake runner per request and seals like the engine would."""
    state = {"runner": None}
    def post(url, payload):
        c = json.load(open(env["job"]["control_path"])); hooks = hooks_factory(c)
        runner = FakeRunner(env["prefix"])
        drive(hooks, runner, env["prefix"], c["chain_tokens"])
        return {"usage": {"prompt_tokens": len(payload["prompt"])}, "choices": [{"text": "x"}]}
    return post


def test_driver_authenticates_seals_and_completes(env):
    pp = os.path.join(env["tmp"], "pp"); os.makedirs(pp); open(os.path.join(pp, "px.u32le"), "wb").write(struct.pack("<%dI" % len(env["prefix"]), *env["prefix"]))
    hooks = CpuHooks(env["jp"])
    v, recs = DRV.run(env["job"], hashlib.sha256(open(env["jp"], "rb").read()).hexdigest(), pp, os.path.join(env["tmp"], "drv"), env["job"]["control_path"], "http://x", env["job"]["out_dir"],
                      post_fn=_fake_engine(env, lambda c: hooks), wait_fn=lambda p, a, b: os.path.exists(p))
    assert v["complete"] and v["requests"] == 2 and v["authenticated_valid"] == 2 and v["repeat_o2_argmax_identical"] is True and v["stop_reason"] is None
    assert os.path.exists(os.path.join(env["tmp"], "drv", "DRIVER-VERDICT.aligned_nonpacked.pA.json"))


def test_driver_stops_on_invalid_seal_and_never_admits_next(env):
    pp = os.path.join(env["tmp"], "pp"); os.makedirs(pp); open(os.path.join(pp, "px.u32le"), "wb").write(struct.pack("<%dI" % len(env["prefix"]), *env["prefix"]))
    hooks = CpuHooks(env["jp"]); calls = []
    def post(url, payload):
        c = json.load(open(env["job"]["control_path"])); calls.append(c["repeat"]); runner = FakeRunner(env["prefix"])
        drive(hooks, runner, env["prefix"], c["chain_tokens"], wrong_token_at=1)                       # engine consumed a wrong token -> invalid seal
        return {"usage": {"prompt_tokens": len(payload["prompt"])}, "choices": [{"text": ""}]}
    v, recs = DRV.run(env["job"], hashlib.sha256(open(env["jp"], "rb").read()).hexdigest(), pp, os.path.join(env["tmp"], "drv"), env["job"]["control_path"], "http://x", env["job"]["out_dir"], post_fn=post, wait_fn=lambda p, a, b: os.path.exists(p))
    assert not v["complete"] and v["requests"] == 1 and calls == [0] and "seal not authenticated" in v["stop_reason"] and recs[0]["seal_authenticated"] is False


def test_driver_refuses_tampered_seal(env):
    good = {"obs_id": "o", "case_id": "c", "run_id": "RUN-T", "arm": "aligned_nonpacked", "process": "A", "repeat": 0, "job_sha256": "j", "valid": True, "o2": None}
    body = dict(good); body["record_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    p = os.path.join(env["tmp"], "seal.json"); json.dump(body, open(p, "w"))
    ok, pr, _ = DRV.authenticate_seal(p, {"obs_id": "o", "run_id": "RUN-T", "repeat": 0, "job_sha256": "j"}, env["tmp"]); assert ok, pr
    body2 = dict(body); body2["valid"] = True; body2["repeat"] = 1; json.dump(body2, open(p, "w"))
    ok, pr, _ = DRV.authenticate_seal(p, {"obs_id": "o", "run_id": "RUN-T", "repeat": 0, "job_sha256": "j"}, env["tmp"]); assert not ok and any("record_sha256" in x for x in pr)


# ------------------------------------------------------------------ config v2, job builder, patcher v2
def test_config_v2_patched_fa2_both_arms_and_real_paths():
    cfg = CFG.build("RUN", "/logs", "a" * 64); assert cfg["problems"] == [], cfg["problems"]
    for arm, c in cfg["arms"].items():
        assert c["fa2"]["sha256"] == FORK_SHA and "/tmp/fr13_fork_fa2.so" in c["rendered_command_NOT_EXECUTED"] and "q1_reference_fa2_install.py" in c["in_container_script"]
        assert "q1_patch_reference_runner_v2.py" in c["in_container_script"] and CFG.CAMPAIGN_REL in c["env"]["PYTHONPATH"] and "--speculative-config" not in c["rendered_command_NOT_EXECUTED"]
        assert c["env"]["Q1_REF_HOOKS_JOB_SHA256"] == "a" * 64 and "test -s /logs/q1_ref_hooks_job.json" in c["in_container_script"]
    assert cfg["arms"]["aligned_nonpacked"]["env"]["VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"] == "0" and cfg["arms"]["native_default_packed"]["env"]["VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"] == "1"
    assert cfg["derivation"]["fork_binary"]["sha256"] == FORK_SHA


def test_job_builder_smoke_scope_and_refusals(tmp_path):
    job = JOB.build(FIXTURES, "R", "aligned_nonpacked", "A", 2, [JOB.SMOKE_CASE], "/logs/q1_ref", "/logs/c.json", True, True)
    assert job["expected_requests"] == 2 and job["cases"][0]["chain_tokens"][0] == 11352 and job["expect_packed_flag"] == "0" and job["archive"]["save_kv_bytes"]
    with pytest.raises(SystemExit):
        JOB.build(FIXTURES, "R", "aligned_nonpacked", "A", 3, [JOB.SMOKE_CASE], "/logs", "/c", True, True)          # smoke scope is R=2
    with pytest.raises(SystemExit):
        JOB.build(FIXTURES, "R", "aligned_nonpacked", "A", 2, ["evaluation-short_available__c0__root-only"], "/logs", "/c", True, False)   # held out


def test_patcher_v2_imports_hooks_v2_with_same_five_anchors():
    src = open(os.path.join(CAMP, "identity", "native_source", "vllm__v1__worker__gpu_model_runner.py")).read()
    out, counts = P2.patch_text(src); assert counts == {"A1": 1, "A2": 1, "A3": 1, "A4": 1, "A5": 1} and "import q1_reference_hooks_v2 as _q1_ref_hooks" in out and out.count(P2.MARK) == 5


# ------------------------------------------------------------------ v2 preliminary repair review: focused controls (defects 1-3 + FA2 API/source binding)
def _sealed(env, **drive_kw):
    control(env["job"], env["prefix"], env["chain"], repeat=0); hooks = CpuHooks(env["jp"]); runner = FakeRunner(env["prefix"])
    return hooks, runner


def _seal_doc(env):
    docs = sorted(os.listdir(os.path.join(env["job"]["out_dir"], "cases"))); assert len(docs) == 1, docs
    p = os.path.join(env["job"]["out_dir"], "cases", docs[0]); return p, json.load(open(p))


def _refs(doc):
    r = []
    for tag in ("o0", "o1"):
        for g in (doc[tag] or {}).get("gdn", {}).values(): r += [g["conv"]["sha256"], g["ssm"]["sha256"]]
        for a in (doc[tag] or {}).get("attention", {}).values():
            for b in a["full_blocks"]: r += [b["k"], b["v"]]
            if a["tail"]: r += [a["tail"]["k"], a["tail"]["v"]]
    if doc["o2"] and doc["o2"].get("raw"): r.append(doc["o2"]["raw"]["sha256"])
    return r


def _corrupt(objects_root, sha, mode):
    p = os.path.join(objects_root, sha + ".bin"); b = bytearray(open(p, "rb").read())
    if mode == "same_size":
        b[len(b) // 2] ^= 0x5A
    elif mode == "truncate":
        b = b[:-1]
    elif mode == "extend":
        b += b"\x00"
    open(p, "wb").write(bytes(b))


@pytest.mark.parametrize("target,mode", [("gdn_conv", "same_size"), ("gdn_ssm", "truncate"), ("kv_block_k", "same_size"), ("kv_block_v", "truncate"), ("kv_tail_k", "same_size"), ("kv_tail_v", "extend"), ("o2", "same_size"), ("o2", "truncate")])
def test_driver_authenticates_every_raw_object_hash_and_length(env, target, mode):
    hooks, runner = _sealed(env); drive(hooks, runner, env["prefix"], env["chain"]); p, d = _seal_doc(env); objects_root = os.path.join(env["job"]["out_dir"], "objects")
    expect = {"obs_id": d["obs_id"], "case_id": "case-x", "run_id": "RUN-T", "arm": "aligned_nonpacked", "process": "A", "repeat": 0, "job_sha256": hooks.job_sha}
    ok, pr, doc = DRV.authenticate_seal(p, expect, objects_root); assert ok, pr                                   # positive control: all bytes authenticate
    assert doc["_raw_objects"]["authenticated"] == len(set(_refs(d))) and doc["_raw_objects"]["authenticated"] >= 48 * 2 + 16 * 3 + 1 and doc["_raw_objects"]["bytes"] > 0
    a0 = d["o0"]["attention"]["language_model.model.layers.101.self_attn.attn"]; g0 = d["o0"]["gdn"]["language_model.model.layers.7.linear_attn"]
    sha = {"gdn_conv": g0["conv"]["sha256"], "gdn_ssm": g0["ssm"]["sha256"], "kv_block_k": a0["full_blocks"][1]["k"], "kv_block_v": a0["full_blocks"][0]["v"], "kv_tail_k": a0["tail"]["k"], "kv_tail_v": a0["tail"]["v"], "o2": d["o2"]["raw"]["sha256"]}[target]
    _corrupt(objects_root, sha, mode)
    ok2, pr2, _ = DRV.authenticate_seal(p, expect, objects_root)
    assert not ok2 and any(("content sha mismatch" if mode == "same_size" else "length") in x for x in pr2), (target, mode, pr2)
    if target.startswith("kv_"):
        with pytest.raises(RuntimeError, match="length .* / sha mismatch"):
            H.reconstruct_logical_kv(a0, objects_root)


def test_expected_object_lengths_reject_inconsistent_manifests(env):
    hooks, runner = _sealed(env); drive(hooks, runner, env["prefix"], env["chain"]); _, d = _seal_doc(env)
    want = H.expected_object_lengths(d); a0 = d["o0"]["attention"]["language_model.model.layers.100.self_attn.attn"]
    assert want[a0["full_blocks"][0]["k"]] == KBS * 4 * 4 * 2 and want[a0["tail"]["k"]] == 5 * 4 * 4 * 2 and want[d["o2"]["raw"]["sha256"]] == 16 * 4
    bad = copy.deepcopy(d); bad["o0"]["attention"]["language_model.model.layers.100.self_attn.attn"]["tail"]["valid_tokens"] = 0
    with pytest.raises(ValueError, match="valid_tokens"):
        H.expected_object_lengths(bad)
    bad = copy.deepcopy(d); bad["o2"]["raw"]["bytes"] = 60
    with pytest.raises(ValueError, match="O2 raw bytes"):
        H.expected_object_lengths(bad)
    bad = copy.deepcopy(d); bad["o0"]["attention"]["language_model.model.layers.100.self_attn.attn"]["full_blocks"][0]["bytes"] = 1
    with pytest.raises(ValueError, match="recorded bytes"):
        H.expected_object_lengths(bad)


def test_packed_flag_requires_existing_typed_attribute():
    assert H.packed_flag(types.SimpleNamespace(enable_packed_recurrent_decode=True)) == "1" and H.packed_flag(types.SimpleNamespace(enable_packed_recurrent_decode=0)) == "0"
    for bad in (types.SimpleNamespace(), types.SimpleNamespace(enable_packed_recurrent_decode=None), types.SimpleNamespace(enable_packed_recurrent_decode="0"), types.SimpleNamespace(enable_packed_recurrent_decode=2)):
        with pytest.raises(RuntimeError, match="enable_packed_recurrent_decode"):
            H.packed_flag(bad)


@pytest.mark.parametrize("kind", ["missing", "string_zero", "none_value"])
def test_missing_or_untyped_packed_attribute_is_fatal_in_primary_arm(env, kind):
    control(env["job"], env["prefix"], env["chain"], repeat=0); hooks = CpuHooks(env["jp"])
    runner = FakeRunner(env["prefix"], packed=(None if kind == "missing" else False))
    if kind != "missing":
        for n, layer in runner.compilation_config.static_forward_context.items():
            if n.endswith(".linear_attn"):
                layer.enable_packed_recurrent_decode = "0" if kind == "string_zero" else None
    drive(hooks, runner, env["prefix"], env["chain"])
    boot = json.load(open(os.path.join(env["job"]["out_dir"], "boot_attestation.aligned_nonpacked.pA.json")))
    assert boot["fatal"] is True and boot["packed_attribute_all_layers_match"] is False and boot["packed_attribute_missing_or_untyped"] == 48 and any("missing/untyped" in p for p in boot["problems"])
    docs = os.listdir(os.path.join(env["job"]["out_dir"], "cases")); assert len(docs) == 1
    d = json.load(open(os.path.join(env["job"]["out_dir"], "cases", docs[0]))); assert d["valid"] is False and any("boot attestation failed" in p for p in d["problems"])


@pytest.mark.parametrize("where", ["full_block_k", "full_block_v", "tail_valid_slot_o0", "tail_slot_valid_only_at_o1"])
def test_nonfinite_materialized_kv_invalidates_with_block_identity(env, where):
    control(env["job"], env["prefix"], env["chain"], repeat=0); hooks = CpuHooks(env["jp"]); runner = FakeRunner(env["prefix"])
    kv = runner.attn_kv[2]                                              # layer 102; O0 = 21 tokens -> blocks 1,2 full + block 3 slots 0..4 valid; O1 = 22 -> slots 0..5 valid
    if where == "full_block_k": kv[0, 1, 3, 0, 0] = float("nan")
    if where == "full_block_v": kv[1, 2, 7, 3, 3] = float("inf")
    if where == "tail_valid_slot_o0": kv[0, 3, 4, 0, 0] = float("nan")
    if where == "tail_slot_valid_only_at_o1": kv[1, 3, 5, 1, 1] = float("nan")
    drive(hooks, runner, env["prefix"], env["chain"])
    docs = os.listdir(os.path.join(env["job"]["out_dir"], "cases")); assert len(docs) == 1
    d = json.load(open(os.path.join(env["job"]["out_dir"], "cases", docs[0]))); assert d["valid"] is False
    msg = " ".join(d["problems"]); assert "nonfinite" in msg and "layers.102.self_attn.attn group 0 physical block" in msg, msg
    if where == "tail_slot_valid_only_at_o1":
        assert d["o0"] is not None and d["o1"] is None and "tail 6 valid tokens" in msg          # O0 (5 valid slots) sealed the block fine; O1 materialized slot 5 and refused it
    elif where == "tail_valid_slot_o0":
        assert d["o0"] is None and "tail 5 valid tokens" in msg
    else:
        assert d["o0"] is None and "full)" in msg


def test_nonfinite_in_unmaterialized_slots_and_unused_blocks_is_ignored(env):
    control(env["job"], env["prefix"], env["chain"], repeat=0); hooks = CpuHooks(env["jp"]); runner = FakeRunner(env["prefix"])
    runner.attn_kv[2][0, 3, 7, 0, 0] = float("nan"); runner.attn_kv[2][1, 3, 6, 2, 2] = float("inf")     # tail block slots 6,7 are never materialized (O1 tail = 6 valid tokens)
    runner.attn_kv[5][0, 5, 0, 0, 0] = float("nan"); runner.attn_kv[5][1, 0, 0, 0, 0] = float("nan")     # block 5 (allocated, beyond need) and null block 0
    drive(hooks, runner, env["prefix"], env["chain"])
    _, d = _seal_doc(env); assert d["valid"] is True, d["problems"]
    a = d["o1"]["attention"]["language_model.model.layers.102.self_attn.attn"]; assert a["tail"]["valid_tokens"] == 6 and a["checked"]["unmaterialized_slots_ignored"] == 2 and a["checked"]["finite_full_blocks"] == 2
    k, v = H.reconstruct_logical_kv(a, os.path.join(env["job"]["out_dir"], "objects")); assert len(k) == (2 * KBS + 6) * 4 * 4 * 2 == len(v)


@pytest.mark.parametrize("kind", ["kv_heads", "head_dim", "dtype", "absent"])
def test_declared_kv_geometry_enforced_at_extraction(env, kind):
    over = {"declared_geometry": {**GEOM, **({"attn_kv_heads": 8} if kind == "kv_heads" else {"head_dim": 256} if kind == "head_dim" else {"kv_dtype": "torch.float16"} if kind == "dtype" else {})}}
    if kind == "absent":
        over = {"declared_geometry": {}}
    jp, job = make_job(env["tmp"] + "/g", env["prefix"], env["chain"], **over) if os.makedirs(env["tmp"] + "/g", exist_ok=True) is None else None
    control(job, env["prefix"], env["chain"], repeat=0); hooks = CpuHooks(jp); runner = FakeRunner(env["prefix"])
    drive(hooks, runner, env["prefix"], env["chain"])
    docs = os.listdir(os.path.join(job["out_dir"], "cases")); assert len(docs) == 1
    d = json.load(open(os.path.join(job["out_dir"], "cases", docs[0]))); assert d["valid"] is False and any(("declares no attention KV geometry" if kind == "absent" else "!= declared") in p for p in d["problems"]), d["problems"]


@pytest.mark.parametrize("kind", ["ok", "impl_version_3", "impl_version_missing", "backend_not_flash", "impl_wrong_class", "impl_missing"])
def test_attention_dispatch_attested_per_layer(env, kind):
    control(env["job"], env["prefix"], env["chain"], repeat=0); hooks = CpuHooks(env["jp"]); runner = FakeRunner(env["prefix"])
    layer = runner.compilation_config.static_forward_context["language_model.model.layers.109.self_attn.attn"]
    if kind == "impl_version_3": layer.impl.vllm_flash_attn_version = 3
    if kind == "impl_version_missing": del layer.impl.vllm_flash_attn_version
    if kind == "backend_not_flash": layer.attn_backend = _Backend("TRITON_ATTN")
    if kind == "impl_wrong_class": layer.impl = types.SimpleNamespace(vllm_flash_attn_version=2)
    if kind == "impl_missing": del layer.impl
    drive(hooks, runner, env["prefix"], env["chain"])
    boot = json.load(open(os.path.join(env["job"]["out_dir"], "boot_attestation.aligned_nonpacked.pA.json")))
    if kind == "ok":
        assert boot["fatal"] is False and boot["attention_impl"]["all_layers_fa2"] is True and boot["attention_impl"]["layers"] == 16
    else:
        assert boot["fatal"] is True and boot["attention_impl"]["all_layers_fa2"] is False and boot["attention_impl"]["bad"][0]["layer"].endswith("layers.109.self_attn.attn")
        assert any("do not dispatch FlashAttentionImpl" in p for p in boot["problems"])


@pytest.mark.parametrize("kind", ["fa_utils_sha", "backend_sha", "layer_sha", "patched_interface_sha", "selector_none", "selector_3", "fa2_unsupported"])
def test_fa2_source_and_selector_binding_is_fatal_on_mismatch(env, kind):
    control(env["job"], env["prefix"], env["chain"], repeat=0); hooks = CpuHooks(env["jp"])
    mut = {"fa_utils_sha": {"fa_utils_sha256": "x" * 64}, "backend_sha": {"flash_attn_backend_sha256": "x" * 64}, "layer_sha": {"attention_layer_sha256": None}, "patched_interface_sha": {"interface_sha256": "s" * 64},
           "selector_none": {"fa_version_selected": None, "selector_error": "AttributeError: module has no attribute get_flash_attn_version"}, "selector_3": {"fa_version_selected": 3}, "fa2_unsupported": {"fa2_supported": False}}[kind]
    hooks.fa2 = dict(CpuHooks.fa2, **mut)
    drive(hooks, FakeRunner(env["prefix"]), env["prefix"], env["chain"])
    boot = json.load(open(os.path.join(env["job"]["out_dir"], "boot_attestation.aligned_nonpacked.pA.json")))
    assert boot["fatal"] is True and any(("FA2 selector" if kind.startswith(("selector", "fa2_")) else "FA2 source") in p for p in boot["problems"]), boot["problems"]
    assert "stock" not in kind or True


def test_real_fa2_facts_use_fa_utils_selector_and_hash_the_pinned_sources(tmp_path, monkeypatch):
    """Runs the PRODUCTION _fa2_facts (not the CPU override) against stub vllm modules whose __file__ are the extracted pinned byte copies."""
    NS = os.path.join(CAMP, "identity", "native_source"); idx = json.load(open(os.path.join(NS, "fa2_source_index.json")))
    d = str(tmp_path); shutil.copy2(os.path.join(NS, "vllm__vllm_flash_attn__flash_attn_interface.py"), os.path.join(d, "flash_attn_interface.py")); open(os.path.join(d, "_vllm_fa2_C.abi3.so"), "wb").write(b"fork")
    def mod(name, file=None, **attrs):
        m = types.ModuleType(name); m.__file__ = file or os.path.join(d, name.split(".")[-1] + ".py"); m.__path__ = []
        for k, v in attrs.items(): setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m); return m
    vllm = mod("vllm", __version__="stub"); vfa = mod("vllm.vllm_flash_attn")
    fai = mod("vllm.vllm_flash_attn.flash_attn_interface", os.path.join(d, "flash_attn_interface.py"), DEFAULT_FA_VERSION=2, is_fa_version_supported=lambda v: v == 2)   # NO get_flash_attn_version, like the pinned bytes
    vfa.flash_attn_interface = fai; vllm.vllm_flash_attn = vfa
    v1 = mod("vllm.v1"); va = mod("vllm.v1.attention"); vb = mod("vllm.v1.attention.backends")
    fu = mod("vllm.v1.attention.backends.fa_utils", os.path.join(NS, "vllm__v1__attention__backends__fa_utils.py"), get_flash_attn_version=lambda requires_alibi=False, head_size=None: 2)
    fb = mod("vllm.v1.attention.backends.flash_attn", os.path.join(NS, "vllm__v1__attention__backends__flash_attn.py"))
    vb.fa_utils = fu; vb.flash_attn = fb; va.backends = vb; v1.attention = va; vllm.v1 = v1
    me = mod("vllm.model_executor"); ml = mod("vllm.model_executor.layers"); ma = mod("vllm.model_executor.layers.attention")
    at = mod("vllm.model_executor.layers.attention.attention", os.path.join(NS, "vllm__model_executor__layers__attention__attention.py")); ma.attention = at; ml.attention = ma; me.layers = ml; vllm.model_executor = me
    hooks = H.Q1RefHooks.__new__(H.Q1RefHooks)
    f = H.Q1RefHooks._fa2_facts(hooks)
    assert f["interface_has_version_getter"] is False and f["interface_default_fa_version"] == 2 and f["fa_version_selected"] == 2 and f["selector_error"] is None and f["fa2_supported"] is True
    assert f["interface_sha256"] == idx["files"]["flash_attn_interface_stock"]["sha256"] == "9caca9584061cfd60c9ebd404d9a8881a62fe7894242efefcba5b88fc086d5b6"
    assert f["fa_utils_sha256"] == idx["files"]["fa_utils"]["sha256"] and f["flash_attn_backend_sha256"] == idx["files"]["flash_attn_backend"]["sha256"] and f["attention_layer_sha256"] == idx["files"]["attention_layer"]["sha256"]
    assert f["installed_so_sha256"] == hashlib.sha256(b"fork").hexdigest() and f["fork_symbol_varlen_fwd_tree_bias"] is False   # host has no fork extension; the real engine must show True
    del fu.get_flash_attn_version                                                                     # selector absent -> None + recorded error (never a silent 2)
    f2 = H.Q1RefHooks._fa2_facts(hooks); assert f2["fa_version_selected"] is None and f2["selector_error"].startswith("AttributeError")
    fu.get_flash_attn_version = lambda requires_alibi=False, head_size=None: 4
    assert H.Q1RefHooks._fa2_facts(hooks)["fa_version_selected"] == 4


def test_job_builder_binds_fa2_sources_geometry_and_install_constants():
    import q1_reference_fa2_install as INST
    idx = json.load(open(os.path.join(CAMP, "identity", "native_source", "fa2_source_index.json")))
    job = JOB.build(FIXTURES, "R", "aligned_nonpacked", "A", 2, [JOB.SMOKE_CASE], "/logs/q1_ref", "/logs/c.json", True, True)
    assert job["fa2_sources"]["fa_utils"] == idx["files"]["fa_utils"]["sha256"] and job["fa2_sources"]["flash_attn_interface_patched"] == INST.EXPECTED_PATCHED_IFACE_SHA and job["fa2_expected_sha256"] == FORK_SHA
    assert idx["files"]["flash_attn_interface_stock"]["sha256"] == INST.EXPECTED_STOCK_IFACE_SHA and job["declared_geometry"] == {"attn_kv_heads": 4, "head_dim": 256, "kv_dtype": "torch.bfloat16", "ssm_dtype": "torch.float32", "conv_dtype": "torch.bfloat16"}
    assert INST.PINNED_SOURCES["v1/attention/backends/fa_utils.py"] == idx["files"]["fa_utils"]["sha256"] and INST.PINNED_SOURCES["model_executor/layers/attention/attention.py"] == idx["files"]["attention_layer"]["sha256"]
    for k in ("flash_attn_interface_stock", "fa_utils", "flash_attn_backend", "attention_layer"):
        e = idx["files"][k]; assert hashlib.sha256(open(os.path.join(CAMP, e["local_copy"]), "rb").read()).hexdigest() == e["sha256"] and os.path.getsize(os.path.join(CAMP, e["local_copy"])) == e["bytes"]
