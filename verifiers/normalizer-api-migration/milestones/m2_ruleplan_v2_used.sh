check_m2_ruleplan_v2_used() {
    local agent_ws="$1"
    local config_path="$2"
    local variant_id="$3"
    local file_a file_b pattern extra_pattern forbidden_pattern cli_file cli_pattern
    file_a=$(jq -r --arg v "$variant_id" '.variants[$v].m2_file_a' "$config_path")
    file_b=$(jq -r --arg v "$variant_id" '.variants[$v].m2_file_b' "$config_path")
    pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m2_pattern' "$config_path")
    extra_pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m2_extra_pattern // ""' "$config_path")
    forbidden_pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m2_forbidden_pattern // ""' "$config_path")
    cli_file=$(jq -r --arg v "$variant_id" '.variants[$v].m2_cli_file // ""' "$config_path")
    cli_pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m2_cli_pattern // ""' "$config_path")

    grep -Fq -- "$pattern" "$agent_ws/$file_a" || return 1
    grep -Fq -- "$pattern" "$agent_ws/$file_b" || return 1
    if [ -n "$extra_pattern" ]; then
        grep -Fq -- "$extra_pattern" "$agent_ws/$file_a" || return 1
        grep -Fq -- "$extra_pattern" "$agent_ws/$file_b" || return 1
    fi
    if [ -n "$forbidden_pattern" ]; then
        ! grep -Fq -- "$forbidden_pattern" "$agent_ws/$file_a" || return 1
        ! grep -Fq -- "$forbidden_pattern" "$agent_ws/$file_b" || return 1
    fi
    if [ -n "$cli_file" ] && [ -n "$cli_pattern" ]; then
        grep -Fq -- "$cli_pattern" "$agent_ws/$cli_file" || return 1
    fi
}
