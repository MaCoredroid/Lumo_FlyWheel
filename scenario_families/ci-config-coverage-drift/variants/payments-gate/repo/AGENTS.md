Local CI for the payments gate service drifted after the package rename.

`make ci` is the contract for this repo, and it should succeed again without
changing the service behavior. Bring `pyproject.toml`,
`.github/workflows/ci.yml`, and the workflow preview helper back in sync with
the current package name `ci_app`.

Keep the payment gate behavior stable:
- required checks stay `queue-check` and `ledger-check`
- the receipt audit lane stays optional

Do not delete the CI checks or rewrite the tests to avoid the failure.
