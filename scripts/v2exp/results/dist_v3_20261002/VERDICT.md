# Distribution diagnostic v3 — verdict under frozen DIST-PROTOCOL-v3.md

Collection: 10 fresh boots, 800 continuations each (20 requests x 40, 24 tokens), 0 errors; finished 2026-10-02 04:41 PDT.
Analysis run once: analyze_dist_v3.py boots.json (perms 300, rng 20261002).

| Rule | Value | Outcome |
|---|---|---|
| 1. E_AA (max AR-AR excess, 6 pairs) | 29.375 (range 18.5-29.4) | noise scale |
| 2. Sensitivity: mean NEG-vs-AR excess > E_AA | 21.998 vs 29.375 | **FAIL** |
| 3. MTP-5 m_k | 22.03, 26.10 | uninformative (rule 2) |
| 3. LumoTree m_k | 23.90, 30.00, 24.97 | uninformative (rule 2) |

**Verdict: uninformative for every arm** (rule 2). The design cannot separate a 0.60->0.65 temperature shift
from boot-to-boot variation of plain decoding, so it supports no distributional claim, positive or negative.
Descriptive only (not a decision): every AR-AR pair differs strongly (z 6.1-9.4), i.e. sampled outputs of the same
plain-decoding configuration vary across boots by more than the temperature control; LumoTree and MTP-5 boots
fall in the same range as AR-AR pairs.
