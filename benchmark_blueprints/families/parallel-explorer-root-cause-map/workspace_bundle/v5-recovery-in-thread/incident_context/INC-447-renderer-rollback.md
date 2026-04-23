# INC-447 Renderer Rollback

The first hotfix attempted to dedupe headings inside
`render_blocked_owner_section()`. It reduced the visible duplicate headings
but left `blocked_owner_total` wrong, so the patch was rolled back.
