# Literature Notes: GDN Tree-Scan MLSys Conversion

## Source Blueprint
- Live page: https://macoredroid.github.io/Lumo_FlyWheel/gdn-tree-scan.html
- Local source: `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel-gh-pages/gdn-tree-scan.html`
- Verification: live HTML and local HTML shared the same SHA-256 hash on 2026-06-18.

## Key Positioning Notes
- The contribution is not generic tree speculative decoding; it is served-realization equivalence for a Qwen GDN hybrid inside vLLM.
- SGLang is the closest adjacent production served comparison, but most cited SGLang mechanisms are cache/scheduler-layer mechanisms rather than an apple-to-apple fused vLLM/FA2/GDN kernel baseline.
- The headline number is clean B=1 token-weighted decode throughput: cat6root 23.8768 tok/s vs native E5 18.7962 tok/s, or +27.0%.
- The per-request-equal latency view is +4.0%; it should be retained as a caveat, not used as the throughput headline.
- End-to-end task wall time remains prefill/agent-heavy, especially because prefix caching is off for this hybrid in the measured setup.
