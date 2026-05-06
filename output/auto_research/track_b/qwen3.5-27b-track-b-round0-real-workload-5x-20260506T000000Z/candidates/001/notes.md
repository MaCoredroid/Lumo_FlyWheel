# Candidate 001 Notes

- Measurement caveat: this candidate expects the controller to dispatch the counted warm completions with enough overlap for vLLM continuous batching to engage; if the harness serializes completions strictly, the config may have little effect.
