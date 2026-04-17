Local CI for the search gate service drifted after the package rename.

`make ci` is the contract for this repo, and it should succeed again without
changing the service behavior. Bring `pyproject.toml`,
`.github/workflows/ci.yml`, and the workflow preview helper back in sync with
the current package name `ci_app`.

Keep the search gate behavior stable:
- required checks stay `ranking-check` and `fixture-check`
- workflow preview jobs should keep search suite order stable
- workflow preview selectors should remain predictable even when search labels
  carry namespaces or extra punctuation

Do not delete the CI checks or rewrite the tests to avoid the failure.
