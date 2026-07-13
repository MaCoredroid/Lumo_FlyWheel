#!/usr/bin/env python3
"""FR13 config fingerprint — a HASHABLE config per arm so confounds are easy to track.

Captures each arm's confound-relevant config from its run artifacts:
  - container_env.txt : the vLLM/model container env (FR13_*/FR10_*/MAMBA_*/APC_*/
    VLLM_*/BATCH_INVARIANT/SPEC_CONFIG(tree)/NUM_SPEC/GPU_UTIL/ENFORCE_EAGER/...)
  - proxy_env.txt / offload_proxy_env.txt : agent-side pins (nudge, forced temp)
  - <arm>.runlog : --agent-wall-s (WALL), --concurrency (CONC), max_num_seqs (BSIZE)

Normalizes to a sorted dict, prints a short sha256 config_hash + key count, and with
--diff surfaces EVERY key that differs across arms = a candidate confound. For a clean
fix-vs-before comparison the ONLY diffs should be the intended knobs (tree kind /
FR13_ATTN_KV_REMAP); anything else is a confound to fix before quoting numbers.
"""
import json, os, re, hashlib, argparse, glob

KEY_PREFIXES = (
    "FR13_", "FR10_", "MAMBA_", "APC_", "NATIVE_", "VLLM_", "LUMO_BATCH",
    "BATCH_INVARIANT", "SPEC_CONFIG", "NUM_SPEC", "GPU_UTIL", "ENFORCE_EAGER",
    "MAX_NUM", "ATTENTION_BACKEND", "TREE", "LUMO_FB_", "LUMO_WIDTH",
)
# per-run-unique / path noise excluded from the hash
EXCLUDE = re.compile(
    r"(_PATH$|_DIR$|_JSON$|_LOG$|_OUT$|_FILE$|PID|HOST|PORT|CONTAINER|_AT$|"
    r"TIMESTAMP|_SO$|SHA256|RUNROOT|^TAG$|DUMP_DIR|SIDECAR|OUTPUT|NSYS)", re.I,
)
# agent-side keys we DO want (from proxy env)
PROXY_KEYS = (
    "LUMO_PROXY_AUTO_CONTINUE", "LUMO_PROXY_FORCE_TEMPERATURE",
    "LUMO_PROXY_TEMPERATURE", "LUMO_PROXY_FORCE_TEMP",
)


def parse_env_file(path):
    d = {}
    if os.path.exists(path):
        for line in open(path, errors="ignore"):
            m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line.strip())
            if m:
                d[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return d


def scan_runlog(arm_dir):
    """Pull WALL / CONC / BSIZE from the arm's runlog (agent-side, not in container env)."""
    out = {}
    logs = glob.glob(os.path.join(os.path.dirname(arm_dir.rstrip("/")), os.path.basename(arm_dir.rstrip("/")) + ".runlog"))
    logs += glob.glob(os.path.join(arm_dir, "*.log"))
    txt = ""
    for lg in logs:
        try:
            txt += open(lg, errors="ignore").read()
        except Exception:
            pass
    for key, pat in (
        ("AGENT_WALL_S", r"--agent-wall-s[= ]([0-9]+)"),
        ("SWE_CONCURRENCY", r"--concurrency[= ]([0-9]+)"),
        ("MAX_NUM_SEQS", r"max_num_seqs['\"=: ]+([0-9]+)"),
        ("PREFIX_CACHE", r"enable_prefix_caching['\"=: ]+(True|False)"),
    ):
        m = re.search(pat, txt)
        if m:
            out[key] = m.group(1)
    # "no wall" if --agent-wall-s absent anywhere in the runlog
    if "AGENT_WALL_S" not in out and txt:
        out["AGENT_WALL_S"] = "ABSENT(no-wall)"
    return out


def config_of_arm(arm_dir):
    cfg = {}
    for k, v in parse_env_file(os.path.join(arm_dir, "container_env.txt")).items():
        if any(k.startswith(p) for p in KEY_PREFIXES) and not EXCLUDE.search(k):
            cfg[k] = v
    for src in ("proxy_env.txt", "offload_proxy_env.txt"):
        penv = parse_env_file(os.path.join(arm_dir, src))
        for k in PROXY_KEYS:
            if k in penv:
                cfg[k] = penv[k]
    cfg.update(scan_runlog(arm_dir))
    return cfg


def fp(cfg):
    canon = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="arm dirs (each with container_env.txt etc.)")
    ap.add_argument("--diff", action="store_true", help="surface every differing key across arms")
    a = ap.parse_args()
    cfgs = {}
    for arm in a.arms:
        cfg = config_of_arm(arm)
        cfgs[arm] = cfg
        print(f"config_hash={fp(cfg)}  keys={len(cfg):3d}  {arm}")
    if a.diff and len(a.arms) >= 2:
        allk = sorted(set().union(*[set(c) for c in cfgs.values()]))
        print(f"\n=== CONFIG DIFF ({len(a.arms)} arms) — each differing key = candidate confound ===")
        ndiff = 0
        for k in allk:
            vals = {arm: cfgs[arm].get(k, "<absent>") for arm in a.arms}
            if len(set(vals.values())) > 1:
                ndiff += 1
                print(f"  {k}:")
                for arm in a.arms:
                    print(f"      {os.path.basename(arm.rstrip('/')):32s} = {vals[arm]}")
        print(f"\n{ndiff} differing keys. EXPECTED intended diffs: SPEC_CONFIG/TREE/NUM_SPEC (tree kind), "
              f"FR13_ATTN_KV_REMAP (the fix). ANY OTHER diff = confound to fix.")


if __name__ == "__main__":
    main()
