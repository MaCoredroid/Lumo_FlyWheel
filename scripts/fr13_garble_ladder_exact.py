"""FR13 garble ladder EXACT-TOKEN: teacher-force the exact LIVE committed IDs (no re-tokenization confound).

Uses exact_tokens.json (prompt_ids + gen_ids + garble_gen_idx from a prior /v1/completions LIVE). CLEAN =
/v1/completions(prompt=committed_to_garble ids, max_tokens=5) -> node0 at the EXACT pos192 tokenization,
predicting '_row' (captured as LAYER_HIDDEN call0). LIVE = /v1/completions(prompt=prompt_ids) -> garble node0
'_rows' at pos192 (same exact prefix). Diff => clean per-layer localization (no 4-token content gap).
"""
import sys, json, time, urllib.request
EP = "http://127.0.0.1:9950"


def post(p, b, t=300):
    r = urllib.request.Request(EP + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as resp:
        s = resp.read().decode()
    return json.loads(s) if s else None


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
    committed = pids + gen[:gi]   # exact IDs up to the garble (pos = len(committed))
    print(f"[exact] prompt_ids={len(pids)} committed_to_garble={len(committed)} (garble at pos {len(committed)})", flush=True)
    try:
        post("/reset_prefix_cache", {}, 30)
    except Exception:
        pass
    xargs = {"fr10_decode_mode": "tree_mtp"}
    # ---- CLEAN FIRST: exact committed IDs, max_tokens=5 (spec-decode runs -> node0 captured as call0) ----
    d = post("/v1/completions", {"model": M, "prompt": committed, "max_tokens": 5, "temperature": 0.0,
             "seed": 0, "return_token_ids": True, "logprobs": 5, "vllm_xargs": xargs})
    ch = d["choices"][0]
    print(f"[CLEAN] first gen ids={[int(x) for x in (ch.get('token_ids') or [])][:3]} text={ch.get('text','')[:20]!r} "
          f"(should start '_row_count')", flush=True)
    try:
        post("/reset_prefix_cache", {}, 30)
    except Exception:
        pass
    # ---- LIVE: full generation from prompt_ids -> garble node0 at pos len(committed) ----
    d2 = post("/v1/completions", {"model": M, "prompt": pids, "max_tokens": 300, "temperature": 0.0,
              "seed": 0, "return_token_ids": True, "vllm_xargs": xargs})
    g = [int(x) for x in (d2["choices"][0].get("token_ids") or [])]
    has = any(t in {1748, 10630} for t in g)
    print(f"[LIVE] gen_len={len(g)} garble_present={has}", flush=True)
    print("DONE - call0=CLEAN node0 (exact pos), later call=garble node0 (exact same pos). Run the diff.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
