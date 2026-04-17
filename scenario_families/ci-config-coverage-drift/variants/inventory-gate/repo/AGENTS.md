Local CI for the inventory gate service drifted after the package rename.

`make ci` is the contract for this repo, and it should succeed again without
changing the service behavior. Bring `pyproject.toml`,
`.github/workflows/ci.yml`, and the workflow preview helper back in sync with
the current package name `ci_app`.

Keep the inventory gate behavior stable:
- required checks stay `schema-check` and `render-check`
- workflow preview artifacts stay paired with the requested inventory checks
- preview report paths should remain stable even when inventory labels carry
  extra separators or punctuation

Do not delete the CI checks or rewrite the tests to avoid the failure.
