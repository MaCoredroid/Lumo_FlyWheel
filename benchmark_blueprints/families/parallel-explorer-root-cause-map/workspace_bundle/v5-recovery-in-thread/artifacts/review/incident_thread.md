# Incident Thread Summary

- 09:12 on-call: renderer headings are duplicated, maybe the markdown path regressed.
- 09:19 follow-up: runtime snapshot shows both `team ops` and `team-ops`
  entering the owner map before rendering.
- 09:23 review: the scheduler refactor touched file-backed owner normalization.
- 09:31 recommendation: confirm adapter normalization before touching renderer.
