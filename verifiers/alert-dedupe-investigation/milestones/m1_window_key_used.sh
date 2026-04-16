check_m1_window_key_used() {
    local agent_ws="$1"
    local config_path="$2"
    local variant_id="$3"
    local file pattern extra_file extra_pattern target
    file=$(jq -r --arg v "$variant_id" '.variants[$v].m1_file' "$config_path")
    pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m1_pattern' "$config_path")
    extra_file=$(jq -r --arg v "$variant_id" '.variants[$v].m1_extra_file // empty' "$config_path")
    extra_pattern=$(jq -r --arg v "$variant_id" '.variants[$v].m1_extra_pattern // empty' "$config_path")
    grep -Fq -- "$pattern" "$agent_ws/$file" || return 1
    if [ -n "$extra_pattern" ]; then
        target="$agent_ws/${extra_file:-$file}"
        grep -Fq -- "$extra_pattern" "$target" || return 1
    fi
}
