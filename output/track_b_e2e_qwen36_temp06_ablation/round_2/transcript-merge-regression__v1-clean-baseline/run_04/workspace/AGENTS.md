# Transcript Merge Regression

Repair the reducer bug in `replay/merge.py`, keep `replay/render.py` honest,
and update `reports/incidents/transcript-merge.md`.

Required verification:

- `python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary`

Hard rules:

- Do not filter duplicate lines only after rendering.
- Do not drop all post-completion fragments.
- Do not derive stable identity from fixture order alone.
