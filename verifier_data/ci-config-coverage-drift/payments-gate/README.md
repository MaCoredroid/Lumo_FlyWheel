Rich verifier bundle for `ci-config-coverage-drift/payments-gate`.

The visible repo only fails because the package rename drift broke `make ci`.
The hidden bundle extends that benchmark with a workflow-preview helper that
must also track the renamed package and normalize separator-heavy payment lane
labels into stable dispatch job ids.
