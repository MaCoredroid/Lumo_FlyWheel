import sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
reps = [
 ('RUNROOT=output/fr14_promoab_${ARM_KIND}${PROMOAB_ARM_SUFFIX:-}_$TS',
  'RUNROOT=/home/mark/shared/lumotree-v2exp-runs/tree/fr14_promoab_${ARM_KIND}${PROMOAB_ARM_SUFFIX:-}_$TS'),
 ('[[ -z "$(docker ps -aq)" ]] || { echo "docker must be empty before boot" >&2; exit 2; }',
  '[[ -z "$(docker ps -q)" ]] || { echo "no container may be running before boot" >&2; exit 2; }'),
 ('  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" "$PROMOAB_KIND" "$SUBSET" \\',
  '  REPO="$REPO" V2EXP_SERVE_ONLY=1 V2EXP_READY_FILE="$RUNROOT_ABS/READY.json" V2EXP_STOP_FILE="$RUNROOT_ABS/STOP" \\\n'
  '  bash /home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp_serve_only_variant.sh "$ARM" "$PROMOAB_KIND" "$SUBSET" \\'),
]
for a, b in reps:
    assert t.count(a) == 1, a
    t = t.replace(a, b)
t = t.replace('#!/usr/bin/env bash', '#!/usr/bin/env bash\n# v2exp copy of results/fr14_nvfp4_port_20260816/promotion_ab_arm_tail10.sh (the Cqc10 vehicle): serve-only\n# fixed32 arm for fixed-input replay. Changes: run root, running-container check, serve-only variant.', 1)
open(dst, 'w').write(t); print('ok')
