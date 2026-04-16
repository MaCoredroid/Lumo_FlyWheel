check_m1_legacy_imports_removed() {
    local agent_ws="$1"
    local config_path="$2"
    local variant_id="$3"
    local scope forbidden
    scope=$(jq -r --arg v "$variant_id" '.variants[$v].m1_glob' "$config_path")
    forbidden=$(jq -r --arg v "$variant_id" '.variants[$v].m1_forbidden_pattern' "$config_path")
    ! grep -R -F -- "$forbidden" "$agent_ws/$scope"
}
