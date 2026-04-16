check_m3_defaults_and_tests() {
    local agent_ws="$1"
    local functional_dir="$2"
    local config_path="$3"
    local variant_id="$4"
    local file_a file_b pattern extra_file extra_pattern target
    file_a=$(jq -r --arg v "$variant_id" '.variants[$v].m3_file_a' "$config_path")
    file_b=$(jq -r --arg v "$variant_id" '.variants[$v].m3_file_b' "$config_path")
    pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m3_pattern' "$config_path")
    extra_file=$(jq -r --arg v "$variant_id" '.variants[$v].m3_extra_file // empty' "$config_path")
    extra_pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m3_extra_pattern // empty' "$config_path")
    grep -Fq -- "$pattern" "$agent_ws/$file_a" || return 1
    grep -Fq -- "$pattern" "$agent_ws/$file_b" || return 1
    if [ -n "$extra_pattern" ]; then
        target="$agent_ws/${extra_file:-$file_b}"
        grep -Fq -- "$extra_pattern" "$target" || return 1
    fi
    [ -f "$functional_dir/pytest_suite_exit_code" ] && [ "$(cat "$functional_dir/pytest_suite_exit_code")" = "0" ]
}
