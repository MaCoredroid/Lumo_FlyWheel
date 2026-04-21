# Rust prototype — synthetic benchmark notes

**Authored:** Diego, 2026-02-14 (prototype sprint from the transformer-runtime team)

## Setup

- Synthetic workload: 1k pre-canned tool-call requests pulled from production
  fixtures anonymized on 2026-02-10.
- Target service: a minimal Rust rewrite of `src/respproxy/hot_path.py` +
  `normalizer.py`. Tool-schema validation code-generated via a derive macro.
- Base image: rust:1.84-bookworm.

## Numbers

| p95 metric                          | Python (current v0.19.0) | Rust prototype |
| ----------------------------------- | ------------------------ | -------------- |
| normalize_request                   | 52 ms                    |  8 ms          |
| schema validate (code-gen)          | 84 ms                    | 11 ms          |
| dispatch_to_skill_pack              | 61 ms                    | 15 ms          |
| total service-side hot path         | ~380 ms                  | ~80 ms         |

Numbers are synthetic. No production traffic was tested. I have not
reproduced the p95 distribution on real traffic patterns.

## Not-yet-built

- skill-pack binding: currently linked directly; a FFI layer for Python skill
  packs would add latency we haven't measured.
- streaming path: not prototyped at all.
- hot-reloading skill packs: not prototyped.

## Opinion

I think the end-to-end number is correct within 30%. Real-world p95 is likely
to be 80-130ms. The hardest part is not the rewrite; it is the skill-pack
binding surface.
