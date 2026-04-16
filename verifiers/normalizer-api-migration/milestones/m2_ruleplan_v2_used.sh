check_m2_ruleplan_v2_used() {
    local agent_ws="$1"
    local config_path="$2"
    local variant_id="$3"
    local file_a file_b pattern
    file_a=$(jq -r --arg v "$variant_id" '.variants[$v].m2_file_a' "$config_path")
    file_b=$(jq -r --arg v "$variant_id" '.variants[$v].m2_file_b' "$config_path")
    pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m2_pattern' "$config_path")
    grep -Fq -- "$pattern" "$agent_ws/$file_a" && grep -Fq -- "$pattern" "$agent_ws/$file_b"
}
