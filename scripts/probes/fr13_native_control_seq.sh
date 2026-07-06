# FR13 char-8 amplifier control: NATIVE MTP-5 (fr9 spec method), cache-OFF (native path
# carries no tree/APC flags), on the 8 fr9-RESOLVED tasks. Isolates spec-method (tree) vs
# harness/env. native recovers ~8/8 => tree is the amplifier; native ~0/8 => harness/env.
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_native natE5_${TAG}  5  5  1
