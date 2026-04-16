check_m3_ci_passing() {
    local functional_dir="$1"
    [ -f "$functional_dir/make_ci_exit_code" ] && [ "$(cat "$functional_dir/make_ci_exit_code")" = "0" ]
}
