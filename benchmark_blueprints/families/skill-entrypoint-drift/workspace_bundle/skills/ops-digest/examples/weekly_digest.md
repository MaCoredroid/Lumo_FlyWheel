Run this from `skills/ops-digest/` when you want the example to stay skill-bundle-relative:

```sh
python ../../tools/digest_runner.py --input ../../fixtures/incidents/sample_events.json --summary-length short
```

The output should start with `# Ops Digest`, then `## Summary`, then `## Events`.
