Rich verifier bundle for `ci-config-coverage-drift/inventory-gate`.

The visible repo only fails because the package rename drift broke `make ci`.
The hidden bundle extends that benchmark with a workflow-preview helper that
must also track the renamed package and normalize separator-heavy inventory
labels into stable preview job ids, artifact names, and `reports/*.json`
outputs.
