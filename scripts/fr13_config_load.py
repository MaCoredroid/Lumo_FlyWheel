#!/usr/bin/env python3
"""FR13 config reader: config/fr13_config.yaml -> shell `export VAR=value` lines.

The launchers source this LIVE:  eval "$(python3 scripts/fr13_config_load.py)"
so a single YAML is the source of truth for the toggleable (non-locked) flags.
The baked PIPELINE flags are NOT here (locked in code; FR13_PIPELINE_LOCK.md).

Leaf keys are the exact env var names; section headers are flattened. Fail-loud:
a missing/malformed YAML aborts (prints to stderr, exit 1) so a launcher never
silently runs a wrong config.
"""
import sys, os, shlex

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "config", "fr13_config.yaml")
    try:
        import yaml
    except ImportError:
        sys.stderr.write("fr13_config_load: PyYAML not available\n"); return 1
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        sys.stderr.write(f"fr13_config_load: cannot read {path}: {e}\n"); return 1
    if not isinstance(cfg, dict):
        sys.stderr.write(f"fr13_config_load: {path} is not a mapping\n"); return 1

    seen = {}
    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)                      # flatten section headers
            else:
                key = str(k)
                if not key.replace("_", "").isalnum() or not (key[0].isalpha() or key[0] == "_"):
                    sys.stderr.write(f"fr13_config_load: bad env key {key!r}\n"); sys.exit(1)
                val = "" if v is None else str(v)
                if key in seen and seen[key] != val:
                    sys.stderr.write(f"fr13_config_load: duplicate key {key} with conflicting values\n"); sys.exit(1)
                seen[key] = val
                print(f"export {key}={shlex.quote(val)}")
    walk(cfg)
    return 0

if __name__ == "__main__":
    sys.exit(main())
