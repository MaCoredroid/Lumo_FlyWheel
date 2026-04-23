# AppSec Review

## APPSEC-EXP-17
- affected_path: `repo/export_api/storage.py`
- note: exploit reproduces through encoded separators and symlink escapes; keep nested exports working

## APPSEC-KEY-03
- affected_path: `repo/export_api/auth.py`
- note: rotate JWT material later; do not fold it into this hotfix

## APPSEC-LOG-12
- affected_path: `repo/export_api/routes.py`
- note: tighten verbose audit logging after the hotfix, not inside the traversal patch
