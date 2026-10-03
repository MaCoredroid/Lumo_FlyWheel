#!/usr/bin/env python3
"""Spec-off sequential target reference engine configuration **v2** (CPU-only renderer; never launches).

v2 applies the implementation review (q1-fullmodel-implementation-review.md) and the parent's resolved choices:
  R1  BOTH reference arms load the PINNED PATCHED FA2 binary (sha 28570f83..., 300123792 bytes) through the ordinary one-token causal
      FLASH_ATTN backend: the fork is mounted read-only and installed by tools/q1_reference_fa2_install.py (binary copy + the minimal
      `_patch_flash_attn_interface` edit only); stock/FA3/FA4 selection must be refused by the in-process boot attestation.
  R6  real mounted campaign paths: the repository at /workspace (ro), the campaign tools dir on PYTHONPATH (so the patched runner can
      import q1_reference_hooks), the patcher invoked at its actual path, an explicit hash-bound hooks job JSON at Q1_REF_HOOKS_JOB
      (missing/unreadable job is FATAL in the instrumented engine), host/container control paths, terminal token binding.
  cache: explicit `--mamba-cache-mode align` retained; the boot receipt must record the resolved mode/group specs/block sizes.
  arms: aligned_nonpacked (VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0, PRIMARY) and native_default_packed (=1, control) differ in exactly
        that variable plus the arm name; the native smoke uses ONLY the primary arm (Q1-NATIVE-SMOKE-DESIGN.json).
v1 (`q1_spec_off_engine_config.py`, stock FA2) is preserved unchanged and superseded.
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, os, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(CAMPAIGN, "..", "..", "..", "..", ".."))
CAMPAIGN_REL = os.path.relpath(CAMPAIGN, ROOT)          # papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927

SCHEMA = "lumo.q1.fullmodel.spec-off-engine-config.v2.1"   # v2.1 = v2 binding patcher/hooks v2.1 (embed-path token-source repair); serve args/env unchanged
IMAGE_DIGEST = "vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
IMAGE_ID = "sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc"
EXPECTED_VLLM = "0.19.2rc1.dev134+gfe9c3d6c5"
MODEL_PATH = "/models/qwen3.8-27b-nvfp4-radixark"
REJECTED_MODEL_PATHS = ("/models/qwen3.6-27b-fp8",)
FORK_SO_HOST = "/home/mark/fr14_splitk_build_20260818/_vllm_fa2_qrow32_gqa_pair_splitk_b1_sm121a.abi3.so"
FORK_SO_SHA = "28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857"
FORK_SO_SIZE = 300123792
STOCK_SO_SHA_IN_IMAGE = "d5c5c5de6eb01b2531edef5c3c31c69b60049c7bba50c90bc1cee25f733c49e3"   # must NOT be the executed binary
LMHEAD_SHIM = "scripts/fr14_patch_nvfp4_lmhead.py"
FA2_PATCHER = "scripts/fr13_patch_fa2_tree_bias.py"
BASE_LAUNCHER = "scripts/fr13_launch_native_mtp_server.sh"
CANDIDATE_LAUNCHER = "scripts/fr14_leg3_launch_nomiddleware.sh"
REF_FA2_INSTALL = "tools/q1_reference_fa2_install.py"
REF_PATCHER = "tools/q1_patch_reference_runner_v2_1.py"
REF_HOOKS = "tools/q1_reference_hooks_v2_1.py"

SERVE_ARGS_COMMON = [
    "--served-model-name", "q1-spec-off-reference", "--host", "0.0.0.0", "--port", "9950",
    "--max-num-seqs", "1", "--gpu-memory-utilization", "0.6", "--max-model-len", "131072", "--seed", "0",
    "--attention-backend", "FLASH_ATTN", "--gdn-prefill-backend", "triton",
    "--enable-prefix-caching", "--enable-chunked-prefill",
    "--mamba-block-size", "1024", "--mamba-ssm-cache-dtype", "float32", "--block-size", "1024", "--mamba-cache-mode", "align",
    "--max-num-batched-tokens", "4096", "--long-prefill-token-threshold", "1024",
]
FORBIDDEN_SERVE_ARGS = ("--speculative-config", "--chat-template", "--enable-auto-tool-choice", "--tool-call-parser", "--reasoning-parser", "--enforce-eager", "--kv-cache-dtype")
ARMS = {
    "aligned_nonpacked": {"VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": "0", "role": "PRIMARY_REFERENCE", "executed_gdn_decode_operator": "vllm.model_executor.layers.fla.ops.fused_sigmoid_gating_delta_rule_update"},
    "native_default_packed": {"VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": "1", "role": "NAMED_CONTROL_DIAGNOSTIC_ONLY", "executed_gdn_decode_operator": "vllm.model_executor.layers.fla.ops.fused_recurrent_gated_delta_rule_packed_decode"},
}
COMMON_ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "VLLM_BATCH_INVARIANT": "0", "VLLM_SERVER_DEV_MODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def assert_not_launcher():
    if os.environ.get("Q1_ALLOW_ENGINE_LAUNCH") is not None:
        raise SystemExit("q1_spec_off_engine_config_v2.py renders configuration only; launching is a separately gated parent action")


def arm_config(arm, run_id, log_dir, job_sha256):
    spec = ARMS[arm]
    tools_in = f"/workspace/{CAMPAIGN_REL}/tools"
    env = dict(COMMON_ENV)
    env.update({"PYTHONPATH": f"/workspace/src:{tools_in}", "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": spec["VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"],
                "Q1_REF_ARM": arm, "Q1_REF_RUN_ID": run_id, "Q1_REF_HOOKS_JOB": "/logs/q1_ref_hooks_job.json", "Q1_REF_HOOKS_JOB_SHA256": job_sha256,
                "Q1_REF_CONTAINER_NAME": f"lumotree-q1-ref-{arm}-{run_id}"})
    serve = ["vllm", "serve", MODEL_PATH] + SERVE_ARGS_COMMON
    container = f"lumotree-q1-ref-{arm}-{run_id}"
    docker = ["docker", "run", "-d", "--name", container, "--cidfile", f"{log_dir}/engine.cid", "--gpus", "all", "--ipc=host", "--memory=105g", "--memory-swap=105g",
              "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", "-p", "9950:9950",
              "-v", "/home/mark/lumotree-review-20260927:/workspace:ro", "-v", "/models:/models:ro", "-v", f"{log_dir}:/logs", "-v", f"{FORK_SO_HOST}:/tmp/fr13_fork_fa2.so:ro"]
    for k in sorted(env):
        docker += ["-e", f"{k}={env[k]}"]
    docker += ["--entrypoint", "bash", IMAGE_DIGEST, "-lc"]
    in_container = " && ".join([
        "set -euo pipefail",
        "python3 -c 'import vllm,sys; assert vllm.__version__==\"%s\", vllm.__version__'" % EXPECTED_VLLM,
        "test -s /logs/q1_ref_hooks_job.json",
        "python3 -c 'import hashlib,os,sys; assert hashlib.sha256(open(\"/logs/q1_ref_hooks_job.json\",\"rb\").read()).hexdigest()==os.environ[\"Q1_REF_HOOKS_JOB_SHA256\"], \"job sha\"'",
        f"python3 /workspace/{LMHEAD_SHIM}",
        f"python3 /workspace/{CAMPAIGN_REL}/{REF_FA2_INSTALL} --fork /tmp/fr13_fork_fa2.so --patcher /workspace/{FA2_PATCHER} --receipt /logs/q1_ref_fa2_install_receipt.json",
        f"python3 /workspace/{CAMPAIGN_REL}/{REF_PATCHER} --apply --receipt /logs/q1_ref_patch_receipt.json",
        "exec " + " ".join(shlex.quote(x) for x in serve),
    ])
    return {"arm": arm, "role": spec["role"], "executed_gdn_decode_operator": spec["executed_gdn_decode_operator"],
            "image": {"digest": IMAGE_DIGEST, "id": IMAGE_ID, "expected_vllm": EXPECTED_VLLM}, "model_path": MODEL_PATH, "container_name": container,
            "fa2": {"binary_host_path": FORK_SO_HOST, "sha256": FORK_SO_SHA, "bytes": FORK_SO_SIZE, "stock_binary_sha256_in_image_must_not_execute": STOCK_SO_SHA_IN_IMAGE,
                    "dispatch": "one-token causal varlen_fwd through the fork (interface dispatches varlen_fwd_tree_bias only with a tree bias; none here)", "declared_geometry_difference": "candidate fixed32 (32,24,256) noncausal tree bias vs reference one-token causal; same binary/family"},
            "env": env, "serve_argv": serve, "docker_argv_without_script": docker, "in_container_script": in_container,
            "rendered_command_NOT_EXECUTED": " ".join(shlex.quote(x) for x in docker) + " " + shlex.quote(in_container),
            "boot_receipt_must_attest": ["vllm.__version__ == expected", "installed _vllm_fa2_C.abi3.so sha == fork (install receipt) and stock sha != executed", "flash_attn_interface sha after the minimal patch",
                                         "vllm.v1.attention.backends.fa_utils.get_flash_attn_version(requires_alibi=False, head_size=None) == 2 (the interface has no getter), fa_utils/flash_attn backend/attention layer/patched interface shas == fa2_source_index, torch.ops._vllm_fa2_C.varlen_fwd_tree_bias present (fork symbols loaded), and EVERY attention layer attn_backend.get_name()==FLASH_ATTN with impl vllm.v1.attention.backends.flash_attn.FlashAttentionImpl and impl.vllm_flash_attn_version==2", "attention backend FLASH_ATTN with FA2 selected; FA3/FA4/Triton refused",
                                         "cache_config.mamba_cache_mode == 'align'; resolved block_size / mamba_block_size / kernel block sizes / group specs", "mamba_ssm_cache_dtype float32; attention KV bf16",
                                         "envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE == arm value and every GDN layer attribute equal", "speculative_config is None", "hooks job sha == Q1_REF_HOOKS_JOB_SHA256; job run/arm identity == env", "patch receipt 5/5; lm_head shim receipt; async scheduling flag recorded"]}


def validate(cfg):
    problems = []
    for arm, c in cfg["arms"].items():
        argv = c["serve_argv"]
        for f in FORBIDDEN_SERVE_ARGS:
            if f in argv:
                problems.append(f"{arm}: forbidden serve arg {f}")
        if argv[2] != MODEL_PATH or argv[2] in REJECTED_MODEL_PATHS:
            problems.append(f"{arm}: wrong model path {argv[2]}")
        for k in ("--max-num-seqs", "--seed", "--mamba-block-size", "--mamba-ssm-cache-dtype", "--block-size", "--mamba-cache-mode", "--attention-backend"):
            if k not in argv:
                problems.append(f"{arm}: missing {k}")
        if c["env"].get("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE") != ARMS[arm]["VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"]:
            problems.append(f"{arm}: packed flag not pinned")
        s = c["in_container_script"]
        if "/tmp/fr13_fork_fa2.so" not in " ".join(c["docker_argv_without_script"]) or REF_FA2_INSTALL not in s or "q1_ref_fa2_install_receipt.json" not in s:
            problems.append(f"{arm}: forked FA2 not mounted/installed")
        if f"/workspace/{CAMPAIGN_REL}/{REF_PATCHER}" not in s or f"/workspace/{CAMPAIGN_REL}/tools" not in c["env"]["PYTHONPATH"]:
            problems.append(f"{arm}: campaign paths not mounted on the launch (patcher/PYTHONPATH)")
        if "test -s /logs/q1_ref_hooks_job.json" not in s or c["env"].get("Q1_REF_HOOKS_JOB") != "/logs/q1_ref_hooks_job.json" or not c["env"].get("Q1_REF_HOOKS_JOB_SHA256"):
            problems.append(f"{arm}: hooks job not required/bound")
        if "--speculative-config" in s:
            problems.append(f"{arm}: speculative config leaked")
    a, b = cfg["arms"]["aligned_nonpacked"], cfg["arms"]["native_default_packed"]
    diff = {k for k in set(a["env"]) | set(b["env"]) if a["env"].get(k) != b["env"].get(k)}
    if diff != {"VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", "Q1_REF_ARM", "Q1_REF_CONTAINER_NAME"}:
        problems.append(f"arms differ in unexpected env keys: {sorted(diff)}")
    if a["serve_argv"] != b["serve_argv"]:
        problems.append("arms differ in serve argv")
    return problems


def build(run_id="RUN_ID_PLACEHOLDER", log_dir="/LOG_DIR_PLACEHOLDER", job_sha256="JOB_SHA_PLACEHOLDER"):
    assert_not_launcher()
    fork_ok = os.path.exists(FORK_SO_HOST) and os.path.getsize(FORK_SO_HOST) == FORK_SO_SIZE
    cfg = {"schema": SCHEMA, "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "CONFIGURATION_ONLY_NOT_LAUNCHED",
           "supersedes": "fullmodel/spec-off-engine-config.v1.json (stock FA2; not approved) -- kept",
           "derivation": {"base_launcher": {"path": BASE_LAUNCHER, "sha256": sha256_file(os.path.join(ROOT, BASE_LAUNCHER))}, "candidate_launcher": {"path": CANDIDATE_LAUNCHER, "sha256": sha256_file(os.path.join(ROOT, CANDIDATE_LAUNCHER))},
                          "lm_head_shim": {"path": LMHEAD_SHIM, "sha256": sha256_file(os.path.join(ROOT, LMHEAD_SHIM))}, "fa2_patcher": {"path": FA2_PATCHER, "sha256": sha256_file(os.path.join(ROOT, FA2_PATCHER)), "used_function": "_patch_flash_attn_interface only"},
                          "fork_binary": {"host_path": FORK_SO_HOST, "present": fork_ok, "sha256": (sha256_file(FORK_SO_HOST) if fork_ok else None), "expected_sha256": FORK_SO_SHA, "expected_bytes": FORK_SO_SIZE, "install_pattern": "fr14_leg3_launch_nomiddleware.sh:6701,7331-7332"},
                          "reference_fa2_install": {"path": REF_FA2_INSTALL, "sha256": sha256_file(os.path.join(CAMPAIGN, REF_FA2_INSTALL))},
                          "reference_patcher": {"path": REF_PATCHER, "sha256": sha256_file(os.path.join(CAMPAIGN, REF_PATCHER))}, "reference_hooks": {"path": REF_HOOKS, "sha256": sha256_file(os.path.join(CAMPAIGN, REF_HOOKS))},
                          "campaign_rel": CAMPAIGN_REL},
           "arms": {arm: arm_config(arm, run_id, log_dir, job_sha256) for arm in ARMS}}
    cfg["problems"] = validate(cfg)
    if fork_ok and cfg["derivation"]["fork_binary"]["sha256"] != FORK_SO_SHA:
        cfg["problems"].append("host fork binary sha != pinned")
    if not fork_ok:
        cfg["problems"].append("pinned fork binary not present on host")
    return cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(CAMPAIGN, "fullmodel", "spec-off-engine-config.v2.1.json"))
    ap.add_argument("--run-id", default="RUN_ID_PLACEHOLDER"); ap.add_argument("--log-dir", default="/LOG_DIR_PLACEHOLDER"); ap.add_argument("--job-sha256", default="JOB_SHA_PLACEHOLDER")
    a = ap.parse_args()
    cfg = build(a.run_id, a.log_dir, a.job_sha256)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=1, allow_nan=False); f.write("\n")
    print(json.dumps({"out": os.path.relpath(a.out, CAMPAIGN), "file_sha256": sha256_file(a.out), "problems": cfg["problems"]}, indent=1))
    return 0 if not cfg["problems"] else 2


if __name__ == "__main__":
    sys.exit(main())
