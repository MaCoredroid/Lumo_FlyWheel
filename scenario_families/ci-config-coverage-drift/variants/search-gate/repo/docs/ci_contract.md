`make ci` is the canonical local CI entrypoint for this repo.

The workflow preview helper is part of the contract:
- it should derive preview job ids from the current package name
- it should emit pytest `-k` selector expressions for the requested search
  suites
- selector tokens should stay stable when suite labels contain namespaces or
  extra punctuation

The required search checks remain `ranking-check` and `fixture-check`.
