# Usage

## Basic invocation

```bash
python -m release_readiness --format json
```

Output is written to stdout. Configuration is read from environment variables
(prefix `RELEASE_READINESS_`) and optionally from a `.env` file in the working
directory.

## Input sources

The CLI reads section records from one of two sources, selected by
`RELEASE_READINESS_SOURCE`:

- `env` (default): reads a JSON array from `RELEASE_READINESS_RECORDS`.
- `fs`: reads a JSON array from the path in `RELEASE_READINESS_FS_PATH`
  (default: `records.json` in the working directory).

Alternatively, pass `--stdin` to read records as JSON from standard input.

## Example

```bash
export RELEASE_READINESS_RECORDS='[
  {"owner": "Sam", "label": "blocked-rollouts", "count": 2},
  {"owner": "Rin", "label": "hotfixes", "count": 1}
]'
python -m release_readiness --format json
```

## Available output formats

Available formats depend on what renderers are registered. Run the CLI with
`--help` to see the current list. To add a new format, see
`docs/renderers.md`.
