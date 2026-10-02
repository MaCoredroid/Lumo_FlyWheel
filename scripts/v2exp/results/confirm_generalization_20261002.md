# Generalization check on a disjoint request set (2026-10-02)

## Setup

- **Request set:** 43 requests frozen before any replay on them (manifest `results/confirm_set_20261002/MANIFEST.json`, sha256 prefix 0dc24282bb12).
  - Captured from native MTP-5 serving the Qwen Code agent on 4 SWE-bench Verified tasks: astropy-14096, 14309, 14508 and 14995.
  - Spread across 5 agent conversations.
  - None of the conversations appears in the 43-request tuning corpus.
- **Configuration:** same replay harness, sampling (temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.0, max 1024 tokens), images and launchers as the tuning-corpus replay.
- **Replicates:** one run per arm.

## Results

| Arm | Rate, confirmation set (tok/s) | Accepted per step | Rate, tuning corpus (tok/s) |
|---|---|---|---|
| LumoTree (deployed hydra27_fixed32) | **30.67** | 4.72 | 28.38–29.41 (6 runs, same code) |
| native vLLM MTP-5 | 27.41 | 3.31 | 26.01–26.23 (4 runs) |
| SGLang EAGLE s7k1d8 | 30.24 | — | 29.18–30.90 (6 runs, chat API and identical tokens) |

Rates are pooled decode rates, (N − R) / (Σe2e − Σttft).

**LumoTree vs native MTP-5:**
- Pooled: +11.9% here, against about +10% on the tuning corpus.
- Per request (median ratio): 1.13.
- LumoTree is faster on 33 of 43 requests.

**LumoTree vs SGLang s7k1d8:**
- Pooled: +1.4% here, against −3% to −4% on the tuning corpus.
- Per request (median ratio): 0.94.
- LumoTree is faster on 18 of 43 requests.

The LumoTree per-step split here is similar to the tuning corpus: 185.2 ms wall per step, of which 110.0 ms target, 48.2 ms drafter and 21.0 ms commit.

## Reading

- **Ranking holds on unseen tasks.** It is LumoTree ≈ SGLang EAGLE s7k1d8 > native MTP-5, the same ranking as on the tuning corpus.
- **Lead over MTP-5 reproduces:** about +12% here.
- **LumoTree vs SGLang is parity.** The sign flips between the two request sets, and both differences fall within SGLang's own run-to-run spread of about ±3%.
- **Caveat:** single replicates per arm on this set.
