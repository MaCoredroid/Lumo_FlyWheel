
# project board sync CLI

Current command:

```bash
python -m sync_app.cli --name launch-checklist --status pending --owner pm-oncall
```

The emitted payload includes:
- `owner`
- `owner_source`
- `routing_key`

If `--owner` is omitted, the service falls back to the default owner from `config/defaults.json`.
