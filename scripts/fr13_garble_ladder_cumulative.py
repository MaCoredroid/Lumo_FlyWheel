"""FR13 garble ladder CUMULATIVE-ONSET: is the col-0 mamba corruption cumulative (early onset) or step-specific?

CLEAN_A = /v1/completions(committed[:150], max_tokens=5) -> node0 at pos150 (sequential, an EARLY step before
the garble). CLEAN_B = /v1/completions(committed[:191], max_tokens=5) -> node0 at pos191 (the garble slot).
LIVE = /v1/completions(pids) -> node0 at every step. Diff LIVE node0 at pos150 vs CLEAN_A and pos191 vs CLEAN_B.
If layer-0 GDN diverges at pos150 too => CUMULATIVE (col-0 corrupt early); if clean at pos150 and only diverged
at pos191 => STEP-SPECIFIC (step-11 branch commit is the discrete corrupting event).
"""
import sys, json, time, urllib.request
EP = "http://127.0.0.1:9950"


def post(p, b, t=300):
    r = urllib.request.Request(EP + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    s = urllib.request.urlopen(r, timeout=t).read().decode()
    return json.loads(s) if s.strip() else {}


def wait_health(m=600):
    t0 = time.time()
    while time.time() - t0 < m:
        try:
            with urllib.request.urlopen(EP + "/health", timeout=5) as r:
                if r.status == 200:
                    print(f"[health] {time.time()-t0:.0f}s", flush=True); return True
        except Exception:
            pass
        time.sleep(6)
    return False


def main():
    if not wait_health():
        print("NEVER HEALTHY"); sys.exit(2)
    M = json.loads(urllib.request.urlopen(EP + "/v1/models", timeout=10).read())["data"][0]["id"]
    ex = json.load(open("output/fr13_garble_ladder/exact_tokens.json"))
    pids, gen, gi = ex["prompt_ids"], ex["gen_ids"], ex["garble_gen_idx"]
    # pos_early = an earlier step (gen idx ~23 => pos150); pos_garble = gi (pos ~192)
    early_gi = 23
    xargs = {"fr10_decode_mode": "tree_mtp"}
    for tag, gidx in [("EARLY", early_gi), ("GARBLE", gi)]:
        try:
            post("/reset_prefix_cache", {}, 30)
        except Exception:
            pass
        committed = pids + gen[:gidx]
        d = post("/v1/completions", {"model": M, "prompt": committed, "max_tokens": 5, "temperature": 0.0,
                 "seed": 0, "return_token_ids": True, "vllm_xargs": xargs})
        first = [int(x) for x in (d["choices"][0].get("token_ids") or [])][:1]
        print(f"[CLEAN_{tag}] committed_len={len(committed)} (pos{len(committed)}) first_tok={first}", flush=True)
    try:
        post("/reset_prefix_cache", {}, 30)
    except Exception:
        pass
    d2 = post("/v1/completions", {"model": M, "prompt": pids, "max_tokens": 300, "temperature": 0.0,
              "seed": 0, "return_token_ids": True, "vllm_xargs": xargs})
    g = [int(x) for x in (d2["choices"][0].get("token_ids") or [])]
    print(f"[LIVE] gen_len={len(g)} garble={any(t in {1748,10630} for t in g)}", flush=True)
    print(f"DONE - CLEAN_EARLY node0 at pos{len(pids)+early_gi}, CLEAN_GARBLE at pos{len(pids)+gi}, LIVE covers both. Run cumulative diff.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
