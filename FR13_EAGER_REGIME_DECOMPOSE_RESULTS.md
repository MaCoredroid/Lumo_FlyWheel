# FR13 eager regime/decompose results

Date: 2026-06-07

Config held fixed unless noted:
- SWE-4 subset: `docs/reports/auto_research/swe-bench-agentic-b4-four-verified-20260530.json`
- Probe: `scripts/fr12_deliverable_swe4_probe.py`
- Sampling: `temperature=0.6`, `top_p=0.95`, `seed=1313`, `max_tokens=64`
- Eager B=1: `--enforce-eager`, `MAX_NUM_SEQS=1`, probe `--batch-size 1`

## Branches on, tree eager B=1

Artifact:
`output/fr13_regime_decompose_20260607T021451Z/tree_branches_on_eager_b1_swe4_spp4_mt64.json`

Result:
- samples per prompt: `4`
- records: `16`
- accepted/event: `0.736936936936937`
- accepted/draft-token: `0.08188188188188188`
- warm decode TPS: `3.396447755143011`
- spec accepted tokens: `409`
- spec draft events: `555`
- spec draft tokens: `4995`

Interpretation:
Eager B=1 branches-on reproduces the low-accept regime. It is not an exclusively
B=4/cuda-graph-captured failure. The next eager-only decomposition is
spine-only branches-off vs native E5.
