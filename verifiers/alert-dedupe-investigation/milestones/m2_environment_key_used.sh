check_m2_environment_key_used() {
    local agent_ws="$1"
    local config_path="$2"
    local variant_id="$3"
    local file pattern
    file=$(jq -r --arg v "$variant_id" '.variants[$v].m2_file' "$config_path")
    pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m2_pattern' "$config_path")
    grep -Fq -- "$pattern" "$agent_ws/$file"
}
