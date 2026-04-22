# Prior rollback

The earlier hotfix mapped `approval_state` directly from `risk_level` in the serializer and frontend. That hotfix was rolled back after operators noticed the CLI export still omitted the field. The new rollout note must acknowledge the rollback and describe the real fix.
