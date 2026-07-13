import torch
import sys
sys.path.insert(0, "src")
from lumo_flywheel_serving.fr10_gdn_tree_kernel import launch_attn_kv_linear_remap

# cat9 sorted: spine choices -> flat rows 1,2,4,6,8 ; verify batch = 10 tokens
# (offset0=anchor, offsets1..9=choices). Full-spine accept, acc=5.
device = "cpu"
nb, bs, H, D = 1, 64, 1, 1
kv = torch.zeros(2, nb, bs, H, D)
# mark each slot's K = slot value, V = slot+1000, so we can trace copies
slots = torch.arange(bs).view(nb, bs, 1, 1).float()
kv[0] = slots            # K
kv[1] = slots + 1000.0   # V
kv_orig = kv.clone()

# request 0 occupies verify offsets 0..9 -> flat batch idx 0..9 -> physical slots 100..109
slot_mapping = torch.zeros(10, dtype=torch.long)
for off in range(10):
    slot_mapping[off] = 100 // 1  # placeholder, set below
# put each offset at a distinct slot: offset j -> slot 20+j (all in block0)
for off in range(10):
    slot_mapping[off] = 20 + off
query_start_loc = torch.tensor([0, 10], dtype=torch.long)
# accepted_paths (flat rows) for full spine: [1,2,4,6,8]. REAL width = max path
# length (5), NOT the 10-token verify span -- the guard must compare qsl spacing
# (=10) to the offsets, NOT to path_cols(=5). Regression guard for that bug.
accepted_paths = torch.tensor([[1, 2, 4, 6, 8]], dtype=torch.int32)
num_accepted = torch.tensor([5], dtype=torch.int32)

n_foreign = launch_attn_kv_linear_remap(
    kv_caches=[kv],
    slot_mapping=slot_mapping,
    query_start_loc=query_start_loc,
    accepted_paths=accepted_paths,
    num_accepted_tokens=num_accepted,
    num_spec_decodes=1,
)
print("n_foreign =", n_foreign, "(expect 3: depths 3,4,5 -> rows 4,6,8 != 3,4,5)")

def K(off):  # K value now at verify offset `off` (slot 20+off)
    return float(kv[0, 0, 20 + off, 0, 0])
def Korig(off):
    return float(kv_orig[0, 0, 20 + off, 0, 0])

# dst offset m+1 should now hold src offset accepted_paths[m]'s ORIGINAL K
checks = []
# m=0: off1 <- off1 (noop); m=1: off2<-off2 (noop)
checks.append(("off1 noop", K(1), Korig(1)))
checks.append(("off2 noop", K(2), Korig(2)))
# m=2: dst off3 <- src off4
checks.append(("off3 <- orig off4", K(3), Korig(4)))
# m=3: dst off4 <- src off6  (off4 was ALSO a src for m=2; gather-then-scatter must read ORIG off4)
checks.append(("off4 <- orig off6", K(4), Korig(6)))
# m=4: dst off5 <- src off8
checks.append(("off5 <- orig off8", K(5), Korig(8)))
# untouched offsets (6,7,8,9 and 0) keep original
checks.append(("off6 untouched", K(6), Korig(6)))
checks.append(("off0 anchor untouched", K(0), Korig(0)))

ok = True
for name, got, want in checks:
    good = abs(got - want) < 1e-6
    ok &= good
    print(("  PASS " if good else "  FAIL ") + f"{name}: got {got} want {want}")
# also verify V copied in lockstep
vok = float(kv[1,0,20+3,0,0]) == Korig(4)+1000.0 and float(kv[1,0,20+4,0,0]) == Korig(6)+1000.0
print("  V-lockstep:", "PASS" if vok else "FAIL")
print("RESULT:", "ALL PASS" if (ok and vok and n_foreign==3) else "FAILURE")
