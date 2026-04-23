# Provider rollover

The maintenance cutover is for the `maintenance-responses` profile, not a generic Responses smoke. That profile must select `responses_proxy` and route traffic through `http://127.0.0.1:11434/v1/responses`.

The rollover is only healthy when follow-up turns preserve exact conversation continuity. Turn two must send the exact turn-one response id in `previous_response_id`, and the proxy-backed response must report that same id back on the follow-up response.

This is why the maintenance note calls out the profile and proxy route together: one-turn success is insufficient if the maintenance-responses profile cannot retrieve turn two from exact `previous_response_id` continuity.
