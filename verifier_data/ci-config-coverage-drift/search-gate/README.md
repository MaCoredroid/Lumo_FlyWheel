Rich verifier bundle for `ci-config-coverage-drift/search-gate`.

The visible repo only fails because the package rename drift broke `make ci`.
The hidden bundle extends that benchmark with a workflow-preview helper that
must also track the renamed package and translate search suite labels into
stable preview job ids plus boolean-friendly `pytest -k` selector expressions.
