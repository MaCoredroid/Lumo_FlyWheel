#!/usr/bin/env python3
"""FR13 config manifest + confound hash — ALL run config in ONE hashable place.

For each arm dir (with container_env.txt + git_head.txt, written by the bigdenom
runner) it emits a canonical manifest and TWO hashes:

  full_hash    = hash of the whole whitelisted config (tree/cache/remap + harness)
  harness_hash = hash of ONLY the "must-match-across-arms" keys (wall, nudge,
                 concurrency, batch, temp, subset, model, seed, git). Two arms with
                 the SAME harness_hash are apples-to-apples; a DIFFERENT harness_hash
                 is a CONFOUND and the tool prints exactly which key differs.

Experiment keys (tree / num_spec / cache-mechanism / remap) are EXPECTED to vary
per arm, so they live in `experiment` and are excluded from harness_hash.

Usage:
  fr13_config_manifest.py ARMDIR [ARMDIR ...]      # write manifest.json per arm + compare
"""
import json, os, sys, hashlib, glob

# Keys that MUST be identical across every arm for a fair A/B. A difference here is a
# confound (the whole point of this tool).
HARNESS_KEYS = [
    "AGENT_WALL_S",              # wall / timeout (EMPTY = no-wall)
    "LUMO_PROXY_AUTO_CONTINUE",  # nudge (0 = OFF, the honest give-up gate)
    "SWE_CONCURRENCY",           # codex concurrency
    "MAX_NUM_SEQS", "MAX_NUM_SEQS_OVR",  # vLLM B
    "DEPLOY_FORCE_TEMP", "LUMO_PROXY_FORCE_TEMP",  # sampling temp
    "SWE_AGENT_ENV",             # per-instance image vs local
    "LUMO_AGENT", "SWE_AGENT",   # which agent (qwen_code)
    "SEED", "VLLM_SEED",
]
# Keys that are EXPECTED to vary per arm (the independent variable). Recorded, not hashed
# into harness_hash.
EXPERIMENT_KEYS = [
    "SPEC_CONFIG",               # holds the tree
    "FR13_ATTN_KV_REMAP",        # the garble fix
    "FR13_ENABLE_APC", "FR13_APC_EXACT_SEED", "NATIVE_ENABLE_APC",
    "FR13_APC_COMMIT_TO_RUNNING_ROW", "FR13_TREE_RUNROW_INIT", "FR13_APC_BURN_NODE_BANK",
    "MAMBA_BLOCK_SIZE", "MAMBA_SSM_CACHE_DTYPE", "APC_BLOCK_SIZE",
    "FR13_DEVICE_MULTIDRAFT",
]


def parse_env(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            d[k.strip()] = v
    return d


def read_first(path):
    try:
        return open(path).read().strip()
    except Exception:
        return None


def h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def manifest_for(armdir):
    env = parse_env(os.path.join(armdir, "container_env.txt"))
    git = read_first(os.path.join(armdir, "git_head.txt"))
    # tree extracted from SPEC_CONFIG for readability
    tree = None
    try:
        tree = json.loads(env.get("SPEC_CONFIG", "") or "{}").get("speculative_token_tree")
    except Exception:
        pass
    harness = {k: env.get(k, "<unset>") for k in HARNESS_KEYS}
    harness["git_head"] = git
    experiment = {k: env.get(k, "<unset>") for k in EXPERIMENT_KEYS}
    experiment["tree"] = tree
    m = dict(arm=os.path.basename(armdir.rstrip("/")),
             harness=harness, experiment=experiment,
             harness_hash=h(harness), full_hash=h({**harness, **experiment}))
    return m


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__); sys.exit(2)
    manifests = []
    for d in dirs:
        # accept a run-root: expand to arm subdirs holding container_env.txt
        if not os.path.exists(os.path.join(d, "container_env.txt")):
            subs = [os.path.dirname(f) for f in glob.glob(os.path.join(d, "*", "container_env.txt"))]
            if subs:
                dirs = dirs + sorted(subs)
                continue
        m = manifest_for(d)
        json.dump(m, open(os.path.join(d, "config_manifest.json"), "w"), indent=2, sort_keys=True)
        manifests.append((d, m))
    if not manifests:
        print("no arm dirs with container_env.txt found under:", sys.argv[1:]); return
    print("=== per-arm harness (must-match) ===")
    for d, m in manifests:
        print(f"  {m['arm']:<32} harness_hash={m['harness_hash']}  wall={m['harness']['AGENT_WALL_S']!r} "
              f"nudge={m['harness']['LUMO_PROXY_AUTO_CONTINUE']!r} conc={m['harness']['SWE_CONCURRENCY']!r} "
              f"B={m['harness'].get('MAX_NUM_SEQS_OVR') or m['harness'].get('MAX_NUM_SEQS')!r}")
    hashes = {m["harness_hash"] for _, m in manifests}
    if len(hashes) == 1:
        print(f"\nOK — all {len(manifests)} arms share harness_hash {hashes.pop()} => APPLES (no harness confound).")
    else:
        print(f"\n*** CONFOUND *** {len(hashes)} distinct harness_hashes across arms. Differing keys:")
        base = manifests[0][1]["harness"]
        for d, m in manifests[1:]:
            diff = {k: (base.get(k), m["harness"].get(k)) for k in base if base.get(k) != m["harness"].get(k)}
            if diff:
                print(f"  {manifests[0][1]['arm']} vs {m['arm']}: {diff}")
    print("\n=== per-arm experiment (independent variable) ===")
    for d, m in manifests:
        e = m["experiment"]
        print(f"  {m['arm']:<32} remap={e['FR13_ATTN_KV_REMAP']!r} apc={e['FR13_ENABLE_APC']!r} "
              f"exact_seed={e['FR13_APC_EXACT_SEED']!r} tree={e['tree']}")


if __name__ == "__main__":
    main()
