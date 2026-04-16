Local CI for the search gate service drifted after the package rename.

`make ci` is the contract for this repo, and it should succeed again without
changing the service behavior. Bring `pyproject.toml`, the workflow file, and
any helper scripts back in sync with the current package name `ci_app`.

Do not delete the CI checks or rewrite the tests to avoid the failure.
