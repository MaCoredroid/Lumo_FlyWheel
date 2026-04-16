check_m3_docs_updated() {
    local agent_ws="$1"
    local config_path="$2"
    local variant_id="$3"
    local file pattern
    file=$(jq -r --arg v "$variant_id" '.variants[$v].m3_file' "$config_path")
    pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m3_pattern' "$config_path")
    grep -Fq -- "$pattern" "$agent_ws/$file"
}
