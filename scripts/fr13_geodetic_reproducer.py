#!/usr/bin/env python3
"""FR13 focused garble reproducer — the FAST fix-selection inner loop.

The live-SWE arm pinned the garble concretely: the tree corrupts the astropy API
`EarthLocation.from_geodetic` -> `from_geodentic`/`from_geodec`/`from_geodeti` (~13%).
Live-SWE arms are ~70min/task (meander-dominated) — too slow to iterate a fix. This probe
elicits the SAME real identifier deterministically (seconds/run), tree vs native vs
no-spec (point --endpoint at the respective server), and measures:

  (1) CORRUPTION RATE — how often the generated code mis-spells from_geodetic to a
      near-neighbor (edit-distance 1-2). Tree should be high, native/no-spec ~0.
  (2) ACCEPT-TIME logprob on the corrupting token (the WITHIN-boot forward-drift signal,
      robust to cross-boot autotune noise, unlike the rate). When the model emits the
      corruption, we read its OWN logprob for the wrong token: a HIGH logprob = the
      tree-verify forward inflated the near-neighbor prob (drift) and the sampler faithfully
      committed it; a LOW logprob = correct-but-unlucky temp-0.6 tail (no drift).

A fix (drift-free tree-verify logits) must drop BOTH the rate AND the corrupting-token
logprob toward the native reference.

Usage:
  python3 fr13_geodetic_reproducer.py --endpoint http://127.0.0.1:9950/v1 \
        --model qwen3.6-27b --arm cat8 --n 24 --out out/geo_cat8.jsonl
"""
import sys, json, re, argparse, urllib.request, concurrent.futures, collections

CANON = "from_geodetic"   # the real API the tree garbles
RELATED = {"from_geodetic", "to_geodetic", "geodetic"}  # correct spellings (not garble)

# identifier-dense prompts that FORCE many `EarthLocation.from_geodetic(...)` emissions.
# short + concrete so they complete in the token budget; thinking OFF (server-side).
PROMPTS = [
    ("geo_roundtrip",
     "Write a short Python function `roundtrip_earth_location(lon_deg, lat_deg, height_m)` "
     "that: builds an astropy EarthLocation via `EarthLocation.from_geodetic(lon_deg, lat_deg, "
     "height_m)`, reads back `.to_geodetic()`, builds ANOTHER EarthLocation with "
     "`EarthLocation.from_geodetic` from those read-back values, and returns both. Call "
     "`EarthLocation.from_geodetic` at least THREE times, spelled IDENTICALLY each time. "
     "Output ONLY a ```python block, no prose."),
    ("geo_itrs",
     "Write a short Python snippet that constructs three astropy EarthLocation objects with "
     "`EarthLocation.from_geodetic(lon, lat, height)` for three different sites, then makes an "
     "ITRS coordinate at each via `loc.get_itrs(obstime)`. Use `EarthLocation.from_geodetic` "
     "for EVERY site, spelled the same. Output ONLY a ```python block."),
    ("geo_list",
     "Write a Python loop that iterates a list of (lon,lat,height) tuples and for each calls "
     "`EarthLocation.from_geodetic(lon, lat, height)`, appending the result to a list. Then "
     "print each location's `.to_geodetic()`. Use `EarthLocation.from_geodetic` inside the "
     "loop. Output ONLY a ```python block."),
]

def _editle2(a, b, maxd=2):
    if a == b: return False
    la, lb = len(a), len(b)
    if abs(la - lb) > maxd: return False
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0]*lb; best = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
            best = min(best, cur[j])
        if best > maxd: return False
        prev = cur
    return 1 <= prev[lb] <= maxd

def gen(endpoint, model, prompt, seed):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.6, "max_tokens": 600, "seed": seed,
                       "logprobs": True, "top_logprobs": 5,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    j = json.loads(urllib.request.urlopen(r, timeout=300).read())
    ch = j["choices"][0]
    toks = [(t["token"], t.get("logprob")) for t in (ch.get("logprobs", {}) or {}).get("content", [])]
    return ch["message"].get("content") or "", toks

def find_garbles(text, toks):
    """Return list of (garbled_word, min_token_logprob_over_its_span). A garble = a token
    subsequence forming an identifier that is edit-dist 1-2 of `from_geodetic` but not a
    correct related spelling."""
    # candidate identifiers containing 'geod'
    out = []
    for m in re.finditer(r"[A-Za-z_]*geod[A-Za-z_]*", text):
        w = m.group(0)
        if w in RELATED:
            continue
        if _editle2(w, CANON) or _editle2(w, "to_geodetic"):
            # locate the token span covering this identifier, take the min logprob
            lp = _span_min_logprob(toks, w)
            out.append((w, lp))
    return out

def _span_min_logprob(toks, word):
    text = "".join(t for t, _ in toks)
    k = text.find(word)
    if k < 0: return None
    end = k + len(word); off = 0; mins = []
    for t, lp in toks:
        a, b = off, off + len(t); off = b
        if a < end and b > k and lp is not None:
            mins.append(lp)
    return round(min(mins), 3) if mins else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=4); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = [(name, p, s) for (name, p) in PROMPTS for s in range(a.n)]
    rows = []
    fh = open(a.out + ".partial", "w")
    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(gen, a.endpoint, a.model, p, s): (name, s) for (name, p, s) in tasks}
        for f in concurrent.futures.as_completed(futs):
            name, s = futs[f]
            try:
                text, toks = f.result()
            except Exception as e:
                print(f"  [{name}#{s}] ERR {e}"); continue
            canon_n = len(re.findall(r"\bfrom_geodetic\b", text))
            garbles = find_garbles(text, toks)
            rec = {"arm": a.arm, "prompt": name, "seed": s, "canon": canon_n,
                   "garbles": garbles, "n_garble": len(garbles)}
            rows.append(rec); fh.write(json.dumps(rec) + "\n"); fh.flush()
    fh.close()
    with open(a.out, "w") as g:
        for r in rows: g.write(json.dumps(r) + "\n")
    # summary
    tot_canon = sum(r["canon"] for r in rows)
    tot_garble = sum(r["n_garble"] for r in rows)
    samples_with = sum(1 for r in rows if r["n_garble"] > 0)
    denom = tot_canon + tot_garble or 1
    garble_lps = [lp for r in rows for (_, lp) in r["garbles"] if lp is not None]
    spell = collections.Counter(w for r in rows for (w, _) in r["garbles"])
    print(f"\n=== [{a.arm}] geodetic reproducer  n={len(rows)} gens ===")
    print(f"  from_geodetic emissions (correct) : {tot_canon}")
    print(f"  near-neighbor GARBLES             : {tot_garble}  ({100*tot_garble/denom:.1f}% of the identifier)")
    print(f"  samples with >=1 garble           : {samples_with}/{len(rows)}")
    if spell: print(f"  garble spellings                  : {dict(spell)}")
    if garble_lps:
        garble_lps.sort()
        med = garble_lps[len(garble_lps)//2]
        hi = sum(1 for x in garble_lps if x > -2.3)  # prob>0.10 = tree CONFIDENT in wrong token
        print(f"  corrupting-token logprob median   : {med:.3f} (prob {2.718**med:.3f})")
        print(f"  HIGH-confidence corruptions (>0.10): {hi}/{len(garble_lps)}  <- DRIFT signal if many")
    print(f"  -> arm '{a.arm}': {'GARBLING' if tot_garble else 'clean'}")

if __name__ == "__main__":
    main()
