#!/usr/bin/env python3
"""FR14 TreeAttention-v2: measure the sm_121a hardware facts the design depends on.

Reads them from the driver, never from documentation. Three of the design's open
items are decided here:

  * CLUSTER_LAUNCH + DSMEM  -> gates fallback F2 (3 CTAs share one staged KV tile
    through distributed shared memory, keeping 12 CTAs and the compute floor).
  * MAX_SHARED_MEMORY_PER_BLOCK_OPTIN -> the real ceiling behind the 96 KB the
    promoted gqa_pair kernel already ships, and the exact budget a 6-head CTA needs.
  * MULTIPROCESSOR_COUNT / L2 -> the occupancy and reuse terms in the cost model.
"""
import ctypes
import json

CU = ctypes.CDLL("libcuda.so.1")

ATTRS = {
    "compute_capability_major": 75,
    "compute_capability_minor": 76,
    "multiprocessor_count": 16,
    "l2_cache_size_bytes": 38,
    "max_shared_memory_per_block_bytes": 8,
    "max_shared_memory_per_block_optin_bytes": 97,
    "max_shared_memory_per_multiprocessor_bytes": 81,
    "max_blocks_per_multiprocessor": 106,
    "max_threads_per_block": 1,
    "max_threads_per_multiprocessor": 39,
    "max_registers_per_block": 12,
    "warp_size": 10,
    "clock_rate_khz": 13,
    "memory_clock_rate_khz": 36,
    "global_memory_bus_width_bits": 37,
    "cluster_launch": 120,
    "unified_addressing": 41,
    "can_map_host_memory": 19,
    "integrated": 18,
}


def main():
    out = {}
    rc = CU.cuInit(0)
    out["cuInit_rc"] = rc
    if rc != 0:
        print(json.dumps(out, indent=2, sort_keys=True))
        raise SystemExit(2)

    dev = ctypes.c_int()
    CU.cuDeviceGet(ctypes.byref(dev), 0)

    name = ctypes.create_string_buffer(256)
    CU.cuDeviceGetName(name, 256, dev)
    out["device_name"] = name.value.decode()

    for key, attr in ATTRS.items():
        val = ctypes.c_int()
        r = CU.cuDeviceGetAttribute(ctypes.byref(val), ctypes.c_int(attr), dev)
        out[key] = val.value if r == 0 else f"QUERY_FAILED_rc{r}"

    # Derived, from measured fields only.
    try:
        bus = out["global_memory_bus_width_bits"]
        mclk = out["memory_clock_rate_khz"]
        if isinstance(bus, int) and isinstance(mclk, int) and mclk > 0:
            # The driver reports the EFFECTIVE data rate for LPDDR5X (8533 MT/s),
            # not the half-rate clock, so no DDR doubling is applied here. An
            # earlier revision doubled it and produced 546 GB/s; 273 GB/s is the
            # figure that reconciles with the campaign's measured bandwidth.
            out["derived_peak_dram_gbytes_per_s"] = round(
                mclk * 1e3 * bus / 8 / 1e9, 1
            )
    except Exception:
        pass

    out["verdict_cluster_launch_supported"] = out.get("cluster_launch") == 1
    optin = out.get("max_shared_memory_per_block_optin_bytes")
    if isinstance(optin, int):
        out["verdict_smem_headroom_over_96KB_bytes"] = optin - 96 * 1024
        out["verdict_g6_96KB_fits"] = optin >= 96 * 1024
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
