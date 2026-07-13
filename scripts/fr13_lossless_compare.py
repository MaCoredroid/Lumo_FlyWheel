#!/usr/bin/env python3
"""Losslessness (QUALITY) gate: is cat8's greedy committed output token-for-token == native's?
A DIVERGENCE = the M-perturbation flipped the committed argmax = a WRONG token = quality issue (not just
speed). Compares output_text saved by the probe. First divergence position = the first quality flip."""
import json, sys, os
RR = sys.argv[1] if len(sys.argv) > 1 else "output/fr13_matched_proof"
def txt(arm):
    f = f"{RR}/{arm}/accept_speed_greedy.json"
    return json.load(open(f)).get("output_text", "") if os.path.exists(f) else None
a = txt("probe_cat8"); b = txt("probe_native")
if a is None or b is None:
    print("missing output_text (cat8=%s native=%s) — probe must save it" % (a is not None, b is not None)); sys.exit(1)
print(f"cat8 output chars={len(a)}  native output chars={len(b)}")
# first char divergence
n = min(len(a), len(b)); i = 0
while i < n and a[i] == b[i]: i += 1
if i == len(a) == len(b):
    print("LOSSLESS: cat8 greedy output == native, char-for-char (identical). M-perturbation is SPEED-ONLY, quality OK.")
elif i >= n and len(a) != len(b):
    print(f"identical for the shorter length ({n}) then one is longer (len diff, likely ignore_eos past-EOS tail). Prefix LOSSLESS.")
else:
    print(f"DIVERGENCE at char {i} (of {n}): QUALITY FLIP — cat8 committed a different token than native.")
    print(f"  ...cat8:   ...{a[max(0,i-40):i]}[[{a[i:i+30]}]]")
    print(f"  ...native: ...{b[max(0,i-40):i]}[[{b[i:i+30]}]]")
    # how far in (as % of the natural response) — early divergence = real quality flip
    print(f"  divergence at {100*i/max(len(b),1):.1f}% of native's output")
