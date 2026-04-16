check_m3_tests_passing() {
    local functional_dir="$1"
    [ -f "$functional_dir/pytest_suite_exit_code" ] && [ "$(cat "$functional_dir/pytest_suite_exit_code")" = "0" ]
}
