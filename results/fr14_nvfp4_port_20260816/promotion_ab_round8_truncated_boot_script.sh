set -euo pipefail
unset FR10_ALLOW_LINEAR_FALLBACK
if [[ -n "${FR13_FIXED32_MODE:-}" ]]; then
  python3 - <<'PY'
import json
import os
import re
import stat
from pathlib import Path


def reject_duplicate_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f'duplicate JSON key: {key!r}')
        payload[key] = value
    return payload


def reject_nonfinite(value):
    raise ValueError(f'non-finite JSON constant: {value}')


source = Path('/run/fr13_fixed32_ingress_secret.host')
target = Path('/run/fr13_fixed32_ingress_secret')
source_info = os.lstat(source)
if (
    not stat.S_ISREG(source_info.st_mode)
    or stat.S_IMODE(source_info.st_mode) != 0o600
    or source_info.st_size <= 0
    or source_info.st_size > 16 * 1024
):
    raise SystemExit('fixed32 staged secret identity is invalid')
source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
try:
    observed = os.fstat(source_fd)
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_dev != source_info.st_dev
        or observed.st_ino != source_info.st_ino
    ):
        raise SystemExit('fixed32 staged secret changed before copy')
    raw = b''
    while len(raw) <= 16 * 1024:
        chunk = os.read(source_fd, 16 * 1024 + 1 - len(raw))
        if not chunk:
            break
        raw += chunk
finally:
    os.close(source_fd)
if not raw or len(raw) > 16 * 1024:
    raise SystemExit('fixed32 staged secret size changed during copy')
payload = json.loads(
    raw.decode('ascii'),
    object_pairs_hook=reject_duplicate_keys,
    parse_constant=reject_nonfinite,
)
if (
    not isinstance(payload, dict)
    or set(payload) != {'schema', 'task_hmac_key_hex', 'engine_bearer'}
    or payload.get('schema') != 'fr13-fixed32-ingress-secrets-v1'
    or not isinstance(payload.get('task_hmac_key_hex'), str)
    or re.fullmatch(r'[0-9a-f]{64}', payload['task_hmac_key_hex']) is None
    or not isinstance(payload.get('engine_bearer'), str)
    or len(payload['engine_bearer']) < 32
    or any(
        ord(char) < 33 or ord(char) > 126
        for char in payload['engine_bearer']
    )
):
    raise SystemExit('fixed32 staged secret JSON contract mismatch')
target_fd = os.open(
    target,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
    0o600,
)
try:
    os.fchmod(target_fd, 0o600)
    os.fchown(target_fd, 0, 0)
    offset = 0
    while offset < len(raw):
        written = os.write(target_fd, raw[offset:])
        if written <= 0:
            raise OSError('short fixed32 secret copy')
        offset += written
    os.fsync(target_fd)
finally:
    os.close(target_fd)
target_info = os.lstat(target)
if (
    not stat.S_ISREG(target_info.st_mode)
    or stat.S_IMODE(target_info.st_mode) != 0o600
    or target_info.st_uid != 0
    or target_info.st_gid != 0
    or target_info.st_size != len(raw)
):
    raise SystemExit('fixed32 runtime secret identity mismatch')
identity = {
    'schema': 'fr13-fixed32-ingress-secret-identity-v1',
    'path': str(target),
    'regular': True,
    'symlink': False,
    'uid': target_info.st_uid,
    'gid': target_info.st_gid,
    'mode': '0600',
    'bytes': target_info.st_size,
}
identity_path = Path('/logs/fr13_fixed32_ingress_secret_identity.json')
temporary = identity_path.with_name(identity_path.name + '.tmp')
temporary.write_text(
    json.dumps(
        identity,
        ensure_ascii=True,
        separators=(',', ':'),
        sort_keys=True,
    )
    + '\n',
    encoding='ascii',
)
temporary.replace(identity_path)
PY
fi
if [[ -n ${FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO_SHA256:-} ]]; then
  if [[ $FR13_FIXED32_B1_FP8_QUANT_REGCACHE == 1 ]]; then
    python3 /workspace/scripts/fr13_fp8_quant_regcache_runtime.py install       --source /tmp/fr13_fp8_quant_regcache.abi3.so       --destination /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so       --attestation /logs/fr13_fixed32_b1_fp8_quant_regcache.binary.json       --selector $FR13_FIXED32_B1_FP8_QUANT_REGCACHE       --expected-sha256 $FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO_SHA256       --patch-source /workspace/scripts/fr13_patch_fp8_quant_fixed32.py       --source-commit $FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SOURCE_COMMIT       --production-sidecar /tmp/fr13_fp8_quant_regcache.pass.json       --expected-production-sidecar-sha256 $FR13_FIXED32_B1_FP8_QUANT_REGCACHE_PASS_SHA256       --smoke-load
  else
    python3 /workspace/scripts/fr13_fp8_quant_regcache_runtime.py install       --source /tmp/fr13_fp8_quant_regcache.abi3.so       --destination /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so       --attestation /logs/fr13_fixed32_b1_fp8_quant_regcache.binary.json       --selector $FR13_FIXED32_B1_FP8_QUANT_REGCACHE       --expected-sha256 $FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO_SHA256       --patch-source /workspace/scripts/fr13_patch_fp8_quant_fixed32.py       --source-commit $FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SOURCE_COMMIT       --smoke-load
  fi
fi
if [[ ${FR13_FIXED32_CUTLASS_WAVE:-stock} != stock ]]; then
  if [[ ${FR13_FIXED32_CUTLASS_WAVE_PRODUCTION:-0} == 1 ]]; then
    python3 /workspace/scripts/fr13_cutlass_wave_binary.py install       --source /tmp/fr13_cutlass_wave.abi3.so       --destination /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so       --attestation /logs/fr13_fixed32_cutlass_streamk_binary.json       --selector $FR13_FIXED32_CUTLASS_WAVE       --qualification-profile $FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE       --diagnostic-task-profile $FR13_FIXED32_CUTLASS_WAVE_DIAGNOSTIC_TASK_PROFILE       --production-pass-sidecar $FR13_FIXED32_CUTLASS_WAVE_PRODUCTION_PASS_SIDECAR       --expected-production-pass-sha256 $FR13_FIXED32_CUTLASS_WAVE_PRODUCTION_PASS_SIDECAR_SHA256       --fixed32-mode $FR13_FIXED32_MODE       --patch-source /workspace/scripts/fr13_patch_cutlass_fixed32_wave.py
  elif [[ $FR13_FIXED32_CUTLASS_WAVE == persistent_b4_m128_static_byte_ab ]]; then
    python3 /workspace/scripts/fr13_cutlass_wave_binary.py install       --source /tmp/fr13_cutlass_wave.abi3.so       --destination /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so       --attestation /logs/fr13_fixed32_cutlass_streamk_binary.json       --selector $FR13_FIXED32_CUTLASS_WAVE       --qualification-profile $FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE       --diagnostic-task-profile $FR13_FIXED32_CUTLASS_WAVE_DIAGNOSTIC_TASK_PROFILE       --resource-credential /tmp/fr13_cutlass_wave_resource_credential.json       --expected-resource-credential-sha256 $FR13_FIXED32_CUTLASS_WAVE_RESOURCE_CREDENTIAL_SHA256
  else
    python3 /workspace/scripts/fr13_cutlass_wave_binary.py install       --source /tmp/fr13_cutlass_wave.abi3.so       --destination /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so       --attestation /logs/fr13_fixed32_cutlass_streamk_binary.json       --selector $FR13_FIXED32_CUTLASS_WAVE       --qualification-profile $FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE       --diagnostic-task-profile $FR13_FIXED32_CUTLASS_WAVE_DIAGNOSTIC_TASK_PROFILE
  fi
fi
cp /tmp/fr13_fork_fa2.so /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
sha256sum /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so | tee /logs/fr13_forked_fa2.sha256
if [[ 0 == 1 ]]; then
  python3 /workspace/scripts/fr13_qrow16_pass_sidecar.py verify     --sidecar      --expected-sidecar-sha256      --candidate-so /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so     --expected-candidate-sha256 
  export FR13_FA2_QROW16_INTERNAL_PRODUCTION_ATTESTED=1
fi
if [[ -n ${FR13_FA2_QROW32_B1_PRODUCTION_ARM} ]]; then
  # Both arms verify by digest, never by git: this runs inside the pinned
  # serving image, which ships no git binary. The GQA-pair arm has its own
  # subcommand because its credential binds a different evidence chain.
  _fr13_b1_verify_command=verify
  if [[ $FR13_FA2_QROW32_B1_PRODUCTION_ARM == gqa_pair ]]; then
    _fr13_b1_verify_command=verify-gqa-pair
  fi
  python3 /workspace/scripts/fr13_qrow32_b1_pass_sidecar.py $_fr13_b1_verify_command     --sidecar $FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR     --expected-sidecar-sha256 $FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR_SHA256     --candidate-so /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so     --expected-candidate-sha256 $FR13_FA2_QROW32_B1_SO_SHA256     --arm $FR13_FA2_QROW32_B1_PRODUCTION_ARM     --patch-source /workspace/scripts/fr13_patch_fa2_tree_bias.py     --expected-source-commit $FR13_FA2_QROW32_B1_SOURCE_COMMIT     --expected-patch-source-sha256 $FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256
  unset _fr13_b1_verify_command
  export FR13_FA2_QROW32_B1_INTERNAL_ATTESTED=1
fi
if [[ -n ${FR13_FA2_QROW32_B1_TIER_B_ARM} ]]; then
  # SITE 19. The attestation above is exported only inside the production-arm
  # block, so a tier-B serve reached _fr13_fa2_qrow32_b1_production_begin and
  # died on has
