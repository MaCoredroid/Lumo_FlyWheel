check_m2_workflow_synced() {
    local agent_ws="$1"
    local config_path="$2"
    local variant_id="$3"
    local file required forbidden
    file=$(jq -r --arg v "$variant_id" '.variants[$v].m2_file' "$config_path")
    required=$(jq -r --arg v "$variant_id" '.variants[$v].m2_required_pattern' "$config_path")
    forbidden=$(jq -r --arg v "$variant_id" '.variants[$v].m2_forbidden_pattern' "$config_path")
    grep -Fq -- "$required" "$agent_ws/$file" && ! grep -Fq -- "$forbidden" "$agent_ws/$file"
}
