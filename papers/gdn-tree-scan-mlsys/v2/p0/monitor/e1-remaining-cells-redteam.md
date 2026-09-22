# Remaining E1 cells: independent raw audit

Read-only audit of cells 7–18 in the unchanged authorized 18-cell campaign. No GPU work, stress suites, source/campaign edits, added boots or repeats. The first six raw audits remain in e1-first-native-cell-redteam.md, e1-native-qualification-redteam.md and e1-first-tree-timing-redteam.md.

Known T1 deviation is retained: request seed 20260921 in every arm; native engine seed 20260921, tree engine seed 0. Raw-support PASS and runner VALID do not assert perfect frozen-configuration compliance. This audit does not establish whole-engine seed invariance.

Each completed cell is reconstructed from all-phase direct API token IDs and recorder events; only final-row clipping is allowed. Unique same-cohort pure physical intervals, phase/occupancy exclusions, all prompt/cohort floors, slow-interval diagnostic, terminal seal, frozen configs, prior native qualification provenance, model API/mount identity and five loaded source hashes are checked. Read-only model identity is compared without rehashing large model weights during timing.

## Completed raw audits

| Cell | Block | Arm | Actual B | Status | Intervals | API-bound tokens | Unique wall s | Rate tokens/s |
|---:|---:|---|---:|---|---:|---:|---:|---:|
| 7 | 2 | tree | 1 | VALID | 314 | 995 | 78.922469652258 | 12.607309482130 |
| 8 | 2 | tree | 4 | VALID | 61 | 804 | 17.220201850869 | 46.689348183189 |
| 9 | 2 | native-5 | 4 | VALID | 64 | 805 | 15.327652007341 | 52.519459576355 |
| 10 | 2 | native-5 | 1 | VALID | 312 | 995 | 72.829188602977 | 13.662104701237 |
| 11 | 2 | native-11 | 1 | VALID | 249 | 986 | 86.185229492374 | 11.440475424936 |
| 12 | 2 | native-11 | 4 | VALID | 52 | 768 | 18.164709499106 | 42.279784327837 |
| 13 | 3 | native-11 | 1 | VALID | 249 | 986 | 86.224372775294 | 11.435281791722 |
| 14 | 3 | native-11 | 4 | VALID | 52 | 740 | 18.181556114927 | 40.700586645192 |
| 15 | 3 | tree | 4 | VALID | 62 | 828 | 17.427238135599 | 47.511831396201 |
| 16 | 3 | tree | 1 | VALID | 298 | 965 | 74.922258021310 | 12.880017574024 |
| 17 | 3 | native-5 | 1 | VALID | 312 | 995 | 72.803675578907 | 13.666892393662 |
| 18 | 3 | native-5 | 4 | VALID | 60 | 808 | 14.358079833910 | 56.274934346842 |

## Per-cell raw identities

### Cell 7, sealed 2026-09-22T10:54:54Z

`{"cell_result.json": "afacaea3ce3fd38a6b905c1493697e8d89d5da5c041f29ff2ff57b83abc40740", "docker_inspect.json": "5bb8d9297a1f21425c7dd1c81f7e2cf76cf91e983ac6d790e568d44fd61736c0", "e1_join.json": "9842b37eefd9b867f4192f11d45b00e5f95cb82d4172825d16ec0a38f1c5cf4e", "e1_manifest.json": "b9fcfd772d74667063afe5c851aaf48727de383640ad1634204c8ea8df0aaa99", "logs/e1_events.jsonl": "9c75160a4fc18c432e4077f4f8f2c6bb25d703a8687950c250cd465aba6925ab"}`

Per-unit support: `{"p015": {"intervals": 40, "tokens": 123, "wall_s": 9.977840754203498}, "p017": {"intervals": 40, "tokens": 123, "wall_s": 9.976352604106069}, "p021": {"intervals": 38, "tokens": 124, "wall_s": 9.467790333554149}, "p058": {"intervals": 35, "tokens": 126, "wall_s": 8.89524121209979}, "p072": {"intervals": 39, "tokens": 125, "wall_s": 9.697898291051388}, "p083": {"intervals": 34, "tokens": 125, "wall_s": 8.609861607663333}, "p085": {"intervals": 46, "tokens": 126, "wall_s": 11.636924442835152}, "p095": {"intervals": 42, "tokens": 123, "wall_s": 10.66056040674448}}`

Exclusions: `{"intervening forward": 7, "preflight/warmup": 24, "terminal": 1}`; excluded durations: `{"intervening forward": 15.041653294116259, "preflight/warmup": 12.989665478467941}`.

### Cell 8, sealed 2026-09-22T11:01:15Z

`{"cell_result.json": "10ce7db48befe05f77574a6eaaadb32d6f10fab71f1f27d46288f47714f9e878", "docker_inspect.json": "19dfa0a6a90783ae7fe72992b4e55ffced4990489f34dda6d900b258d315f591", "e1_join.json": "29ef568e203094b0066ed1dee68d43ed00041dc67e9afc1758d1d3d1490e1127", "e1_manifest.json": "0185446cc18f54597a037ea722c7cefb44cebc2ad98cc3b22f8f4d6e18b795cf", "logs/e1_events.jsonl": "7edf4ba21bb07cc03374266c638b7ca2c06e10476993e96a88c150ccb78414ec"}`

Per-unit support: `{"A(1-4)": {"intervals": 30, "tokens": 393, "wall_s": 8.290255506522954}, "B(5-8)": {"intervals": 31, "tokens": 411, "wall_s": 8.929946344345808}}`

Exclusions: `{"cohort change": 5, "intervening forward": 1, "occupancy ramp/drain": 16, "preflight/warmup": 13, "terminal": 1}`; excluded durations: `{"cohort change": 1.3434383366256952, "intervening forward": 13.55398887488991, "occupancy ramp/drain": 4.108253509737551, "preflight/warmup": 10.982605428434908}`.

### Cell 9, sealed 2026-09-22T11:08:20Z

`{"cell_result.json": "3545c6639417489580ce16425e8dd6c88e6ed3fe0c8d78ef9ca845ad61462850", "docker_inspect.json": "c8fd2d4f711577a6308eeb4a25662120c1848175f98dd0d412065df0afa445a9", "e1_join.json": "599ba9887be8bdc4fc0f81a6d640c41cb46a761ad2e612e09df8495e3bf05d08", "e1_manifest.json": "10ac059dedd79346cce838f0205d707cebf70235a301ae465eb8e6f7f45999f3", "logs/e1_events.jsonl": "4991490305ef108557336200cbe5f5d2f9208297f38f58c97883978916199bc9"}`

Per-unit support: `{"A(1-4)": {"intervals": 29, "tokens": 375, "wall_s": 6.824562408961356}, "B(5-8)": {"intervals": 35, "tokens": 430, "wall_s": 8.50308959838003}}`

Exclusions: `{"cohort change": 6, "intervening forward": 1, "occupancy ramp/drain": 18, "preflight/warmup": 16, "terminal": 1}`; excluded durations: `{"cohort change": 1.380207097157836, "intervening forward": 12.819973330944777, "occupancy ramp/drain": 4.150077861733735, "preflight/warmup": 6.021758025512099}`.

### Cell 10, sealed 2026-09-22T11:16:19Z

`{"cell_result.json": "e224ba112641393cf121471389205868696d8e852cb024019ea1be02831e76e4", "docker_inspect.json": "48626591c78dee38e367dc6a00b1cd8aa59f4050e9528079ae5556a3449f3290", "e1_join.json": "74ebca57b323b5e609f8ec6ab706cf370fd46a6ad04c198888c7fc168d0dfd2f", "e1_manifest.json": "fde0afe46068aa4217d4e4c6bdf5aaa38da2d0aadc7b9d50afbdfe34439be25b", "logs/e1_events.jsonl": "fe73b65ebf3d06df57052393f1b0ef2ada581aeaf360a68751f0a6697d350e57"}`

Per-unit support: `{"p015": {"intervals": 39, "tokens": 121, "wall_s": 9.031439428217709}, "p017": {"intervals": 36, "tokens": 123, "wall_s": 8.35722023062408}, "p021": {"intervals": 39, "tokens": 126, "wall_s": 9.037083177827299}, "p058": {"intervals": 43, "tokens": 126, "wall_s": 10.111984508112073}, "p072": {"intervals": 29, "tokens": 125, "wall_s": 6.717668409459293}, "p083": {"intervals": 34, "tokens": 125, "wall_s": 7.988685255870223}, "p085": {"intervals": 46, "tokens": 126, "wall_s": 10.806427562609315}, "p095": {"intervals": 46, "tokens": 123, "wall_s": 10.77868003025651}}`

Exclusions: `{"intervening forward": 7, "preflight/warmup": 20, "terminal": 1}`; excluded durations: `{"intervening forward": 14.58576712757349, "preflight/warmup": 6.4938802206888795}`.

### Cell 11, sealed 2026-09-22T11:24:37Z

`{"cell_result.json": "e2153d7e26ccb3cdda971627d6261a5c8b5bacc8844d2b648e4613f41e8c3dc4", "docker_inspect.json": "40c6fa66e34d70784a4ae2a9ecdc8e35d7c957b05c7cae9c91cd9640af5541d2", "e1_join.json": "0066f4347b762fc3d6f6994072b9b4728e2d508c3b3338211dcd981dd7c3dcb5", "e1_manifest.json": "26bb2cdd6dfd6837785e94794ac8eb3c7a61258a93d24d32ee57e75c678e407c", "logs/e1_events.jsonl": "02caa95b5b2acc1905e2223a54116f510532ac6dee99d666da22b2b079d29c46"}`

Per-unit support: `{"p015": {"intervals": 35, "tokens": 122, "wall_s": 12.01776082534343}, "p017": {"intervals": 28, "tokens": 122, "wall_s": 9.64555855281651}, "p021": {"intervals": 24, "tokens": 124, "wall_s": 8.248385896906257}, "p058": {"intervals": 35, "tokens": 124, "wall_s": 12.176716109737754}, "p072": {"intervals": 26, "tokens": 118, "wall_s": 8.968201107345521}, "p083": {"intervals": 26, "tokens": 125, "wall_s": 9.064057923853397}, "p085": {"intervals": 38, "tokens": 125, "wall_s": 13.208547896705568}, "p095": {"intervals": 37, "tokens": 126, "wall_s": 12.856001179665327}}`

Exclusions: `{"intervening forward": 7, "preflight/warmup": 20, "terminal": 1}`; excluded durations: `{"intervening forward": 16.559478397481143, "preflight/warmup": 8.987610596232116}`.

### Cell 12, sealed 2026-09-22T11:31:51Z

`{"cell_result.json": "62c69c0b48c378e613eea7c7cf77fe6ea760ffa5f66814acc1fe2d9f3bcb3a7d", "docker_inspect.json": "b528a85ab3c5b37dd8d9689564b2067680a44cc6baa2922e104a82cbb09d6c42", "e1_join.json": "125a75c7167d64a35362abb22b474c01c9393e22f9a710a34b7192fd0a08f141", "e1_manifest.json": "88a7f3874b9417d9df28c69689c60a984f7a5b2a720a9e424d0f955464ac0a25", "logs/e1_events.jsonl": "b5e870a04a743ce2c8e19c3b5ad03d10ed59c193bb40f3ec512875f043f837a4"}`

Per-unit support: `{"A(1-4)": {"intervals": 25, "tokens": 372, "wall_s": 8.633423606865108}, "B(5-8)": {"intervals": 27, "tokens": 396, "wall_s": 9.531285892240703}}`

Exclusions: `{"cohort change": 5, "intervening forward": 1, "occupancy ramp/drain": 20, "preflight/warmup": 12, "terminal": 1}`; excluded durations: `{"cohort change": 1.6914071748033166, "intervening forward": 13.933068477548659, "occupancy ramp/drain": 6.7870330242440104, "preflight/warmup": 5.713044461794198}`.

### Cell 13, sealed 2026-09-22T11:40:08Z

`{"cell_result.json": "227bf44eb992ad111fa68b8173b6dbccf6cc4ac221f919b955d3c30bb4c541b9", "docker_inspect.json": "4aca10af2df71cd8f104c961ecf464d2a727ddc92df015a1dccf06e3126dbb2c", "e1_join.json": "e49a602c47c57a489ce42aa1f7f4a6582f6befed957279a5196cff6a1f7b106f", "e1_manifest.json": "dd2adaa49a6578befcb9a2a5437aca22886165663d9e04dde00cd46e7a33568b", "logs/e1_events.jsonl": "f8ce35fd2fd77cf5d778a6e487861b0dc916f3dd5fa0693e868131b000825475"}`

Per-unit support: `{"p015": {"intervals": 35, "tokens": 122, "wall_s": 12.028114078566432}, "p017": {"intervals": 28, "tokens": 122, "wall_s": 9.655591601505876}, "p021": {"intervals": 24, "tokens": 124, "wall_s": 8.25539623759687}, "p058": {"intervals": 35, "tokens": 124, "wall_s": 12.175592318177223}, "p072": {"intervals": 26, "tokens": 118, "wall_s": 8.987771199084818}, "p083": {"intervals": 26, "tokens": 125, "wall_s": 9.038591790013015}, "p085": {"intervals": 38, "tokens": 125, "wall_s": 13.192870846018195}, "p095": {"intervals": 37, "tokens": 126, "wall_s": 12.890444704331458}}`

Exclusions: `{"intervening forward": 7, "preflight/warmup": 20, "terminal": 1}`; excluded durations: `{"intervening forward": 16.44848145544529, "preflight/warmup": 8.953775773756206}`.

### Cell 14, sealed 2026-09-22T11:47:21Z

`{"cell_result.json": "2082b17f56d429e6fe0aace848a8d83ae1294a8b23a259e67a1ee059ec4e81ca", "docker_inspect.json": "fc6efaec254715d1ac6d6c06897fb50f1649936b8c6e97a23279f3406be7926f", "e1_join.json": "c5571cd74e0e33ad3fafe1034d3fd462a045104bde0ff80457e682da2ebe6621", "e1_manifest.json": "1595b9036635037e48e462776d1f91624fef4183dce2786f29926245b8788715", "logs/e1_events.jsonl": "665ed79ba71e1b4ce96b5302fb16fed14396663e5ecb03201ec681c950607f94"}`

Per-unit support: `{"A(1-4)": {"intervals": 26, "tokens": 366, "wall_s": 8.987729423679411}, "B(5-8)": {"intervals": 26, "tokens": 374, "wall_s": 9.193826691247523}}`

Exclusions: `{"cohort change": 5, "intervening forward": 1, "occupancy ramp/drain": 20, "preflight/warmup": 12, "terminal": 1}`; excluded durations: `{"cohort change": 1.7038072776049376, "intervening forward": 13.722598793916404, "occupancy ramp/drain": 6.584592264145613, "preflight/warmup": 6.353110584430397}`.

### Cell 15, sealed 2026-09-22T11:53:59Z

`{"cell_result.json": "2ddf04e502865337c7667b7db16b8a51ce9ac89c9452da931b2c4047b880dab0", "docker_inspect.json": "ac6ce07179636f7c0e2bbf1c4a603c95e1d9e69f60c6b4ce5e49b73d57e58ea7", "e1_join.json": "b486959ad5b94d79815768baa18ebf8f41209551ad5e69d3f028767160c2867f", "e1_manifest.json": "22a868d74138ac66d83674b65f7d1310d901da53b977cc92b7a7e72cf940d3e0", "logs/e1_events.jsonl": "b42bb83f96bec3079003255fd5f70cda6cbb7ae7643069156321be0029b400d2"}`

Per-unit support: `{"A(1-4)": {"intervals": 32, "tokens": 418, "wall_s": 8.890990647487342}, "B(5-8)": {"intervals": 30, "tokens": 410, "wall_s": 8.536247488111258}}`

Exclusions: `{"cohort change": 4, "intervening forward": 1, "occupancy ramp/drain": 11, "preflight/warmup": 13, "terminal": 1}`; excluded durations: `{"cohort change": 1.0772226722911, "intervening forward": 13.375443710945547, "occupancy ramp/drain": 3.141772633418441, "preflight/warmup": 12.551428072154522}`.

### Cell 16, sealed 2026-09-22T12:01:22Z

`{"cell_result.json": "4a0b66b01ad61a20924fd1162bb20de66c19792186b1b09c199cf0b1283e4820", "docker_inspect.json": "a8f0d4597ab29f6c3b4689c0c6049aeeac918424d125760488decf5943a7651e", "e1_join.json": "5a50d451e504ef74ed208613507d878900162d39d73a3593b35635246d5e46eb", "e1_manifest.json": "6270c2e3b3b7397c4fa10c7cd6231c3f8ad347b08a9aee645740dd77fd7871d5", "logs/e1_events.jsonl": "fd9374da463d08abff0b722cf83aaf55e3e467ae7d6d1dc9fa3534bc11eb77a8"}`

Per-unit support: `{"p015": {"intervals": 40, "tokens": 122, "wall_s": 9.981940879486501}, "p017": {"intervals": 40, "tokens": 125, "wall_s": 9.960125944577157}, "p021": {"intervals": 38, "tokens": 99, "wall_s": 9.436939221806824}, "p058": {"intervals": 36, "tokens": 126, "wall_s": 9.160203848034143}, "p072": {"intervals": 29, "tokens": 123, "wall_s": 7.245372891426086}, "p083": {"intervals": 29, "tokens": 121, "wall_s": 7.385676600970328}, "p085": {"intervals": 45, "tokens": 125, "wall_s": 11.386581439524889}, "p095": {"intervals": 41, "tokens": 124, "wall_s": 10.365417195484042}}`

Exclusions: `{"intervening forward": 7, "preflight/warmup": 22, "terminal": 1}`; excluded durations: `{"intervening forward": 15.109945885837078, "preflight/warmup": 12.529697596095502}`.

### Cell 17, sealed 2026-09-22T12:09:20Z

`{"cell_result.json": "c848fe314f88f05774fcdefd48a2ba9ab02d4a2f3a34efafac408e45abb2b493", "docker_inspect.json": "31c1244f36f07bfc66c975a227a40c3d0aa1d178af70aaa0bad2d00ecd97b826", "e1_join.json": "fee768289f243c9474a24a3b4736a3c2fc74d98867be23d376574748cca99db3", "e1_manifest.json": "50e93b422c2dc2b424e936d3fa706d7d369ca39276e2e694562d38c940bb6d92", "logs/e1_events.jsonl": "9fcd9f904555ce663ee9491760508e344ab27798835fac44072feeb1514ea7b0"}`

Per-unit support: `{"p015": {"intervals": 39, "tokens": 121, "wall_s": 9.024577124975622}, "p017": {"intervals": 36, "tokens": 123, "wall_s": 8.353571319952607}, "p021": {"intervals": 39, "tokens": 126, "wall_s": 9.009168059565127}, "p058": {"intervals": 43, "tokens": 126, "wall_s": 10.113523843698204}, "p072": {"intervals": 29, "tokens": 125, "wall_s": 6.7291785487905145}, "p083": {"intervals": 34, "tokens": 125, "wall_s": 7.993085094727576}, "p085": {"intervals": 46, "tokens": 126, "wall_s": 10.773230584338307}, "p095": {"intervals": 46, "tokens": 123, "wall_s": 10.807341002859175}}`

Exclusions: `{"intervening forward": 7, "preflight/warmup": 20, "terminal": 1}`; excluded durations: `{"intervening forward": 14.71710820775479, "preflight/warmup": 6.5415726732462645}`.

### Cell 18, sealed 2026-09-22T12:16:26Z

`{"cell_result.json": "e3b6beed2f11d13087bdb8c47bfffacb136133f99d8bb2d397547d68b960f488", "docker_inspect.json": "0253c593a40ef7b42a523bced3d13a5cd382121a9125c644255a972c5bb8ac47", "e1_join.json": "3d30302ddc567124ec08d5e09484f88c7966936316adb83978e43ece3a554149", "e1_manifest.json": "6081ec8565779217daf8fc0aa6a7aa63b9c7e06cecb1b2e96e5dcadf835e1c0a", "logs/e1_events.jsonl": "72881fc87a0d7a1cf669fc9766821ebc6b9bbe5072f8f81eececc71ae3d8319d"}`

Per-unit support: `{"A(1-4)": {"intervals": 27, "tokens": 395, "wall_s": 6.359670618548989}, "B(5-8)": {"intervals": 33, "tokens": 413, "wall_s": 7.99840921536088}}`

Exclusions: `{"cohort change": 5, "intervening forward": 1, "occupancy ramp/drain": 20, "preflight/warmup": 16, "terminal": 1}`; excluded durations: `{"cohort change": 1.1649372214451432, "intervening forward": 12.960298639722168, "occupancy ramp/drain": 4.574078337289393, "preflight/warmup": 5.2470338540151715}`.

## Current issues / completion

No new material raw-measurement finding. All 18 cells are now independently audited; the final aggregate and retrospective diagnostic are recorded below. Known T1 remains an as-executed configuration deviation.

Aggregate review: PASS independent reconstruction, completed after all 18 cells sealed.

Full counters, all raw/source hashes, qualification provenance and diagnostic outputs are in e1-remaining-cells-redteam.json.

## Reproducer

The following program is sent on stdin using ssh mark@100.103.10.122 'python3 - CELL_INDEX'. It reads only the completed cell; it neither imports campaign measurement code nor writes remote files.

```python
from pathlib import Path
import json,hashlib,math,sys,re
from collections import Counter,defaultdict
R=Path("/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T100028Z-e1-18cells")
S=R/"campaign_snapshot"; idx=int(sys.argv[1])
cell=next(c for c in json.loads((S/"e1_cells.json").read_text())["cells"] if c["index"]==idx)
N=cell["max_num_seqs"]; K=cell["num_speculative_tokens"]; is_native=cell["arm"]!="tree"
preflight_boot=is_native and idx<=4
C=R/("cell_%02d_b%d_%s_%s_a1"%(idx,cell["block"],cell["arm"],cell["batch"]))
result=json.loads((C/"cell_result.json").read_text())
assert result["terminal_seal"]["sealed"] and result["status"] in ("VALID","INSUFFICIENT_SUPPORT"),result
assert result["terminal_seal"]["campaign_snapshot_sha256sums"]==hashlib.sha256((S/"SHA256SUMS").read_bytes()).hexdigest()
assert result["terminal_seal"]["gates"]["verify"]=="PASS"
if preflight_boot: assert result["terminal_seal"]["gates"]["preflight_live"]=="PASS" and result["terminal_seal"]["gates"]["preflight_final"]=="PASS"
for D in [S,C/"script_snapshot"]:
 for line in (D/"SHA256SUMS").read_text().splitlines():
  h,n=line.split(None,1); assert hashlib.sha256((D/n.lstrip("*")).read_bytes()).hexdigest()==h,(D,n)
cfg=json.loads((C/"docker_inspect.json").read_text())[0]; env=dict(x.split("=",1) for x in cfg["Config"]["Env"] if "=" in x); cmd=" ".join(cfg["Config"]["Cmd"])
assert env["SPEC_CONFIG"]==cell["spec_config"]
if is_native: assert env["FR10_DECODE_MODE_DEFAULT"]=="naive_mtp" and env["FR10_ENABLE_TREE_GDN"]=="0"
else:
 assert env["FR10_DECODE_MODE_DEFAULT"]=="tree_mtp" and env["FR10_ENABLE_TREE_GDN"]=="1"
 for key,value in {"FR13_ATTN_KV_REMAP":"1","FR13_SLOT_REORDER":"0","FR13_KV_REMAP_SYNCFREE":"1","FR13_REPLAY_ROUTE":"1","FR13_TREE_RUNROW_INIT":"1"}.items(): assert env.get(key, "1" if key=="FR13_TREE_RUNROW_INIT" else None)==value,(key,env.get(key))
 assert env.get("FR13_EAGER_PACK","1")=="1" and env.get("E7A_CAPTURE_SHIM","0")=="0" and not env.get("E7B_SUBSTITUTE")
for flag in ["--no-async-scheduling","--no-enable-prefix-caching","--enforce-eager",("--attention-backend 'FLASH_ATTN'" if is_native else "--attention-backend 'TREE_ATTN'"),f"--max-num-seqs '{N}'","--gpu-memory-utilization '0.6'","--max-model-len '16384'"]: assert flag in cmd,flag
if is_native: assert "--seed '20260921'" in cmd
else: assert "--seed " not in cmd # observed engine-default 0, tracked configuration deviation
assert cfg["Config"]["Image"].endswith("3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776")
assert any(m["Destination"]=="/models" and not m["RW"] for m in cfg["Mounts"])
assert env.get("FR10_METRICS","0")!="1" and env.get("FR13_SFWD_GPU_TIMER")=="1" and env.get("E1_RECORD")=="/logs/e1_events.jsonl"
for key in ["LUMO_MTP_DRAFT_TRACE_FILE","LUMO_TREE_PATH_LCP_LOG","LUMO_TREE_SAMPLER_DEBUG_LOG","FR13_FINAL_LOGIT_CAPTURE","FR10_LAYER_HIDDEN_CAPTURE","FR10_TREE_GDN_CAPTURE_PAYLOAD"]: assert not env.get(key)
log=(C/"docker_logs.txt").read_text(errors="replace")
if is_native: assert "ENGAGED" not in log
else: assert "FR13_ATTN_KV_REMAP ENGAGED" in log and "FR13_SLOT_REORDER ENGAGED" not in log
gates={}
for name in (["native_preflight_live.json","native_preflight_final.json","cell_verify.json"] if preflight_boot else ["cell_verify.json"]):
 j=json.loads((C/name).read_text()); assert j["all_pass"] and all(c["pass"] for c in j["checks"])
 gates[name]=len(j["checks"])
 if "preflight" in name: assert j["reference_scores"]["sha256"]==hashlib.sha256((S/"prefix_scores.json").read_bytes()).hexdigest()
trace=(C/"driver_trace.txt").read_text()
if preflight_boot: assert trace.index("preflight_verdict=PASS")<trace.index("workload_start_utc")
ev=[json.loads(x) for x in (C/"logs/e1_events.jsonl").read_text().splitlines() if x.strip()]
assert [x["n"] for x in ev]==list(range(1,len(ev)+1))
assert ev[0]["event"]=="recorder_probe" and ev[-1]["event"]=="run_close"
assert ev[-1]["n_events"]==len(ev) and not ev[-1].get("sink_failed")
assert not (C/"logs/e1_recorder_FAILED.flag").exists()
assert not any(x["event"]=="error" or x.get("errors",0)>0 or x.get("sink_failures",0)>0 for x in ev)
fe={}; ps={}; outs={}; breaks=set()
for e in ev:
 k=e["event"]
 if k=="forward_entry": assert e["seq"] not in fe; fe[e["seq"]]=e
 if k=="physical_step": assert e["physical_step_id"] not in ps; ps[e["physical_step_id"]]=e
 if k=="output_rows": assert e["seq"] not in outs; outs[e["seq"]]=e
 if k=="chain_break": breaks.add(e["seq"])
assert sorted(fe)==list(range(min(fe),max(fe)+1))
assert ev[-1]["n_forwards"]==len(fe) and ev[-1]["n_physical_steps"]==len(ps) and ev[-1]["n_output_records"]==len(outs)
byreq=defaultdict(list)
for seq,o in sorted(outs.items()):
 assert seq in fe and o["kind"] in ["pure","nonpure"]
 if o["kind"]=="pure": assert ps[o["physical_step_id"]]["seq"]==seq
 for row in o["rows"]:
  rid=row["request_id"]; assert row["step_idx"]==len(byreq[rid])
  assert row["n_emitted"]==len(row["emitted_ids"])
  byreq[rid].append((seq,row["emitted_ids"]))
for pid,p in ps.items():
 o=outs[p["seq"]]; rids=[x["request_id"] for x in o["rows"]]
 assert o["kind"]=="pure" and len(rids)==len(set(rids))==p["num_reqs"]
 assert set(rids)==set(p["request_ids"]) and 1<=p["num_reqs"]<=N
man=json.loads((C/"e1_manifest.json").read_text())
frozen=json.loads((S/"frozen_prefixes.json").read_text()); expected=[x["id"] for x in frozen["pilot"]]
prefhash={x["id"]:x["prefix_sha256"] for x in frozen["pilot"]}
pool={x["id"]:x for x in json.loads((S/"prefix_pool.json").read_text())["prefixes"]}
phases={}; rid2meta={}; api={}; bounds={}; trunc={}
for m in man["requests"]:
 rid=[r for r in byreq if r==m["response_id"] or r.startswith(m["response_id"]+"-")]
 assert len(rid)==1; rid=rid[0]
 cap=C/"cohort"/("req_"+m["slot"]+"_"+m["prefix_id"])/"capture_request.json"
 c=json.loads(cap.read_text()); toks=c["response_logprobs_tokens"]
 assert all(isinstance(x,str) and x.startswith("token_id:") for x in toks)
 ids=[int(x[9:]) for x in toks]
 assert len(ids)==m["n_api_tokens"] and c["response_id"]==m["response_id"]
 req=c["request"]; assert req["return_tokens_as_token_ids"] is True and req["temperature"]==0 and req["seed"]==20260921
 assert c["max_tokens"]==(128 if m["phase"]=="timed" else 32)
 assert m["finish_reason"] in ["length","stop"]
 if m["finish_reason"]=="length": assert len(ids)==m["max_tokens"]
 text=pool[m["prefix_id"]]["text"]
 assert hashlib.sha256(text.encode()).hexdigest()==prefhash[m["prefix_id"]]==c["prompt_sha256"]==c["prefix_sha256"]
 body={"model":req["model"],"prompt":text,"max_tokens":req["max_tokens"],"temperature":req["temperature"],"seed":req["seed"],"logprobs":req["logprobs"],"echo":req["echo"],"return_tokens_as_token_ids":req["return_tokens_as_token_ids"]}
 assert hashlib.sha256(json.dumps(body).encode()).hexdigest()==c["request_body_sha256"]
 stream=[t for _,rowids in byreq[rid] for t in rowids]; counts={seq:len(rowids) for seq,rowids in byreq[rid]}
 if stream!=ids:
  lastseq,lastids=byreq[rid][-1]; headlen=len(stream)-len(lastids)
  assert len(ids)<len(stream) and stream[:len(ids)]==ids and len(ids)>=headlen
  counts[lastseq]=len(ids)-headlen; trunc[rid]=len(stream)-len(ids)
 api[rid]=ids; bounds[rid]=counts; phases[rid]=m["phase"]; rid2meta[rid]=m
assert set(phases)==set(byreq)
assert [m["prefix_id"] for m in man["requests"] if m["phase"]=="timed"]==expected
expected_phase_counts={"warmup":(2 if N==1 else 4),"timed":8}
if preflight_boot: expected_phase_counts["preflight"]=8
assert dict(Counter(phases.values()))==expected_phase_counts
assert [m["prefix_id"] for m in man["requests"] if m["phase"]=="warmup"]==([expected[0]]*2 if N==1 else expected[:4])
assert any(p["num_reqs"]==N and all(phases[r]=="warmup" for r in p["request_ids"]) for p in ps.values())
cohorts={frozenset(expected[:4]):"A(1-4)",frozenset(expected[4:]):"B(5-8)"}
ret=[]; excluded=Counter(); per=defaultdict(lambda:{"intervals":0,"tokens":0,"wall_s":0.0}); excwalls=defaultdict(float)
for pid,p in sorted(ps.items()):
 nxt=ps.get(pid+1); rids=p["request_ids"]; reason=None
 if nxt is None: reason="terminal"
 elif any(phases[rid]!="timed" for rid in rids): reason="preflight/warmup"
 elif nxt["seq"]!=p["seq"]+1: reason="intervening forward"
 elif any(p["seq"]<=b<=nxt["seq"] for b in breaks): reason="chain break"
 elif set(nxt["request_ids"])!=set(rids): reason="cohort change"
 elif len(rids)!=N: reason="occupancy ramp/drain"
 elif any(x.get("discarded") for x in outs[p["seq"]]["rows"]): reason="discarded"
 if reason:
  excluded[reason]+=1
  if nxt: excwalls[reason]+=nxt["t_start"]-p["t_start"]
  continue
 wall=nxt["t_start"]-p["t_start"]; assert wall>0
 tok=sum(bounds[rid][p["seq"]] for rid in rids); ret.append((pid,p["seq"],tok,wall))
 name=rid2meta[rids[0]]["prefix_id"] if N==1 else cohorts[frozenset(rid2meta[r]["prefix_id"] for r in rids)]
 unit=per[name]; unit["intervals"]+=1; unit["tokens"]+=tok; unit["wall_s"]+=wall
T=sum(x[2] for x in ret); W=sum(x[3] for x in ret); rate=T/W if W else None
j=json.loads((C/"e1_join.json").read_text())
assert j["n_usable"]==len(ret) and j["sum_emitted_tokens_api_bound_pure_support"]==T
assert math.isclose(j["sum_wall_s_unique_physical_steps"],W,rel_tol=1e-12)
assert j["tokens_per_wall_second"]==rate or math.isclose(j["tokens_per_wall_second"],rate,rel_tol=1e-12)
assert set(x["physical_step_id"] for x in j["steps"])==set(x[0] for x in ret)
if result["status"]=="VALID":
 assert set(per)==(set(expected) if N==1 else set(cohorts.values())) and all(x["intervals"]>=1 for x in per.values())
 assert result["primary"]["sum_emitted_tokens_api_bound_pure_support"]==T
 assert math.isclose(result["primary"]["tokens_per_wall_second"],rate,rel_tol=1e-12)
qual=dict(json.loads((S/"e1_qualification_manifest.json").read_text())["qualified"])
if (R/"qualification.json").exists(): qual.update(json.loads((R/"qualification.json").read_text())["qualified"])
qualkey=cell["arm"]+"/"+cell["batch"]
assert qualkey in qual
qualpaths=qual[qualkey].get("evidence",[])
if is_native:
 expected_idx={("native-5",1):1,("native-5",4):2,("native-11",4):3,("native-11",1):4}[(cell["arm"],N)]
 first=R/("cell_%02d_b1_%s_%s_a1"%(expected_idx,cell["arm"],cell["batch"]))
 assert set(qualpaths)=={str(first/n) for n in ["native_preflight_live.json","native_preflight_final.json","cell_result.json"]}
 for n in ["native_preflight_live.json","native_preflight_final.json"]:
  g=json.loads((first/n).read_text());assert g["all_pass"] and all(x["pass"] for x in g["checks"])
 firstresult=json.loads((first/"cell_result.json").read_text());assert firstresult["terminal_seal"]["sealed"] and firstresult["status"]=="VALID"
 assert qual[qualkey]["utc"]==firstresult["terminal_seal"]["utc"]
assert re.search(r"vllm serve ['\"]?/models/qwen3\.6-27b-fp8['\"]? --served-model-name ['\"]?qwen3\.6-27b['\"]?(?:\s|$)",cmd)
model=json.loads((C/"models.json").read_text())["data"]
assert len(model)==1 and model[0]["id"]=="qwen3.6-27b" and model[0]["root"]=="/models/qwen3.6-27b-fp8" and model[0]["max_model_len"]==16384
firstcfg=json.loads((R/"cell_01_b1_native-5_B1_a1/docker_inspect.json").read_text())[0]
assert [(m["Source"],m["Destination"],m["RW"]) for m in cfg["Mounts"] if m["Destination"]=="/models"]==[(m["Source"],m["Destination"],m["RW"]) for m in firstcfg["Mounts"] if m["Destination"]=="/models"]
assert (C/"worktree_head.txt").read_text()==(R/"cell_01_b1_native-5_B1_a1/worktree_head.txt").read_text()
slow=[x for x in ret if x[3]>1.5]; trim=[x for x in ret if x[3]<=1.5]
trimrate=sum(x[2] for x in trim)/sum(x[3] for x in trim) if trim else None
diag=result["diagnostic_over_cap"]
assert diag["cap_s"]==1.5 and diag["n_intervals_over_cap"]==len(slow) and diag["physical_step_ids"]==[x[0] for x in slow]
assert math.isclose(diag["wall_s_over_cap"],sum(x[3] for x in slow),rel_tol=1e-12,abs_tol=1e-12)
assert diag["tokens_per_wall_second_trimmed_DIAGNOSTIC_ONLY"]==trimrate or math.isclose(diag["tokens_per_wall_second_trimmed_DIAGNOSTIC_ONLY"],trimrate,rel_tol=1e-12)

summary={"cell":idx,"block":cell["block"],"arm":cell["arm"],"batch":N,"status":result["status"],"seal_utc":result["terminal_seal"]["utc"],"gates":gates,
"events":len(ev),"forwards":len(fe),"physical_steps":len(ps),"output_records":len(outs),"nonpure_outputs":sum(o["kind"]=="nonpure" for o in outs.values()),
"phase_requests":dict(Counter(phases.values())),"phase_api_tokens":{ph:sum(len(api[r]) for r in api if phases[r]==ph) for ph in set(phases.values())},
"api_tokens":sum(map(len,api.values())),"ledger_tokens":sum(len(ids) for seqs in byreq.values() for _,ids in seqs),"clipped_tokens":sum(trunc.values()),"clipped_requests":len(trunc),
"support":{"intervals":len(ret),"tokens":T,"wall_s":W,"rate":rate,"over_1p5_s":sum(x[3]>1.5 for x in ret)},
"excluded":dict(excluded),"excluded_walls":dict(excwalls),"per_unit":dict(per),"finish_counts":dict(Counter(m["finish_reason"] for m in man["requests"]))}
hashes={}
for n in ["logs/e1_events.jsonl","e1_manifest.json","docker_inspect.json","cell_result.json","e1_join.json","native_preflight_live.json","native_preflight_final.json"]:
 p=C/n
 if p.exists(): hashes[n]=hashlib.sha256(p.read_bytes()).hexdigest()
summary["hashes"]=hashes
summary["loaded_hashes"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (C/"loaded_backend").glob("*.py")}
expected_loaded={"gpu_model_runner.py":"b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40","eagle.py":"aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7","tree_attn.py":"a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97","gdn_linear_attn.py":"723974c703d1ad1545399b78526e5d50ae7f7f362e434bad384a0a3327595b28","rejection_sampler.py":"5be4ad43409da82b63bfd0177926467e9a2b59da553ee6dd04d3130a5d77b12f"}
assert summary["loaded_hashes"]==expected_loaded
engine_init=next(l for l in log.splitlines() if "Initializing a V1 LLM engine" in l)
assert ("seed="+str(20260921 if is_native else 0)+",") in engine_init
summary["qualification_provenance"]=qual[qualkey]
summary["model_identity"]={k:model[0][k] for k in ["id","root","max_model_len"]}
summary["diagnostic_over_cap"]=diag
summary["seed_scope"]={"request_seed":20260921,"engine_seed":20260921 if is_native else 0,"tree_engine_seed_deviation_recorded":not is_native}
print(json.dumps(summary,indent=1))
print("PASS independent raw support variant (tree engine seed deviation recorded)",idx)

```

## Final aggregate reconstruction

PASS independent primary/support/paired-block/precision reconstruction. All 18 planned cells and one immutable attempt per cell are accounted for. No result was replaced. Known T1 seed noncompliance is retained and raw-support PASS is separate from perfect frozen-config compliance.

Aggregate source SHA256: `98beec9efac26f560783d9d77af3e2154edbd6b24ebadfe141b0cafce6e48a68`; aggregate.json SHA256: `dbd4542ee9899da3eb592fdf8a0b935010f5c8035de9d5cb75e581343c7deb1d`. Embedded corrected raw reproducer SHA256: `9443381ca6604255a15af363ec719df616e362dd616fef647ad3b5519049a5ef`.

Arm rates are arithmetic means over whole boot blocks, not pooled-token rates. Contrasts use paired blocks within B1/B4 separately. The finite prescribed bootstrap matches the independently enumerated 27 ordered resample outcomes at both endpoints; with only three blocks the 95% interval is coarse. Precision uses the native-5 mean rate denominator.

| Batch | Arm | Block 1 | Block 2 | Block 3 | Mean rate |
|---|---|---:|---:|---:|---:|
| B1 | native-5 | 13.682245138976391 | 13.66210470123698 | 13.666892393661975 | 13.67041407795845 |
| B1 | native-11 | 11.439958475412476 | 11.44047542493633 | 11.435281791722367 | 11.438571897357058 |
| B1 | tree | 12.597382772294214 | 12.607309482129649 | 12.88001757402356 | 12.694903276149141 |
| B4 | native-5 | 57.33075818918667 | 52.51945957635484 | 56.27493434684242 | 55.37505070412798 |
| B4 | native-11 | 43.47118832311055 | 42.27978432783668 | 40.70058664519177 | 42.15051976537967 |
| B4 | tree | 47.65063886822262 | 46.68934818318858 | 47.51183139620072 | 47.283939482537306 |

| Batch | Contrast | Mean difference | 95% interval | Half-width / native-5 mean | Status |
|---|---|---:|---|---:|---|
| B1 | tree_minus_native5 | -0.975510801809308 | [-1.0848623666821773, -0.786874819638415] | 0.010898994915019574 | within target |
| B1 | tree_minus_native11 | 1.2563313787920827 | [1.1574242968817376, 1.4447357823011924] | 0.0105085143647076 | within target |
| B1 | native11_minus_native5 | -2.2318421806013906 | [-2.242286663563915, -2.22162927630065] | 0.0007555508979267849 | within target |
| B4 | tree_minus_native5 | -8.09111122159067 | [-9.680119320964046, -5.830111393166263] | 0.0347630194360326 | within target |
| B4 | tree_minus_native11 | 5.133419717157639 | [4.1794505451120685, 6.8112447510089495] | 0.023763357075361483 | within target |
| B4 | native11_minus_native5 | -13.224530938748309 | [-15.574347701650652, -10.23967524851816] | 0.04816855592273808 | within target |

## Retrospective timed API-ID stream diagnostic

Full timed API-ID streams only, not only retained rate-support tokens. Observed equality/divergence across repeated blocks within arm/batch, and across arms within block/batch. Retrospective diagnostic; no thresholds, inference, causal attribution, replacements or rate/support changes.

Matched prefixes do not by themselves establish matching continuations. Across arms within the same block/batch, 14 of 144 prompt-stream pairs are exactly equal; repeated-block comparisons within an arm/batch have 72 of 144 equal pairs. The sole early timed stop is cell 16, prefix p021, with 101 API token IDs; it is preserved without replacement or extension. These are observations only, with no attribution of cause. The following counts describe the recorded streams only. Each repeated-block row has 24 comparisons: eight prompts times three block pairs. Each across-arm row has 24 comparisons: eight prompts times three blocks for the stated fixed arm pair. Full exact IDs, request/capture hashes, counts, first differing token IDs and zero/one-based divergence positions for all 288 comparisons are preserved in the companion JSON.

| Comparison | Batch | Left arm | Right arm | Equal / compared | Divergent | First divergence position range (1-based) |
|---|---|---|---|---:|---:|---|
| repeated_blocks | B1 | native-5 | native-5 | 24 / 24 | 0 | None |
| repeated_blocks | B1 | native-11 | native-11 | 24 / 24 | 0 | None |
| repeated_blocks | B1 | tree | tree | 8 / 24 | 16 | [1, 75] |
| repeated_blocks | B4 | native-5 | native-5 | 7 / 24 | 17 | [1, 119] |
| repeated_blocks | B4 | native-11 | native-11 | 5 / 24 | 19 | [3, 75] |
| repeated_blocks | B4 | tree | tree | 4 / 24 | 20 | [1, 42] |
| across_arms_same_block | B1 | native-5 | native-11 | 3 / 24 | 21 | [3, 25] |
| across_arms_same_block | B1 | native-5 | tree | 3 / 24 | 21 | [1, 75] |
| across_arms_same_block | B1 | native-11 | tree | 1 / 24 | 23 | [1, 108] |
| across_arms_same_block | B4 | native-5 | native-11 | 3 / 24 | 21 | [1, 120] |
| across_arms_same_block | B4 | native-5 | tree | 1 / 24 | 23 | [1, 97] |
| across_arms_same_block | B4 | native-11 | tree | 3 / 24 | 21 | [1, 109] |

Auditor-only correction: the original added model CLI check incorrectly required quotes. It was replaced with quote-tolerant exact-value matching, preserving the model API and read-only mount checks. The original failed check and correction are retained under auditor_corrections in JSON; this is not a campaign defect.

Final limitations: this bounded audit verifies recorded event/API support, source/configuration provenance and the frozen numerical reduction. It does not turn three boot blocks into a high-resolution uncertainty estimate, establish equal continuations or whole-engine seed invariance, or erase the documented T1 freeze deviation.

## Complete frozen support reporting

All 18 cells have per-prompt (B1) / per-cohort (B4) retained interval/token/wall totals, full timed API token totals and per-unit/per-phase excluded-interval reason/count/duration breakdowns under per_cell_support_reporting in the companion JSON. B1 retained s/token is wall over the same supported tokens. Exclusions are attributed once to the starting prompt/cohort. Phase exclusions retain preflight/warmup labels; B4 occupancy subsets retain cohort identity and are not reclassified as full B4. Terminal intervals explicitly have unobservable duration because there is no successor; no zero wall duration is invented.

The per-unit exclusion reduction was independently derived from raw physical-step, output-row and phase records and checked against each earlier independent cell total. Its source is preserved and hash-bound in the JSON: `97a2b00f654933146d1ef53a6c82473189e979e4a7e60c44f82e3c9d47f61636`.

| Cell | B | Unit | Retained intervals | Supported tokens | Retained wall s | Full timed API tokens |
|---:|---|---|---:|---:|---:|---:|
| 1 | B1 | p072 | 29 | 125 | 6.735802257434 | 128 |
| 1 | B1 | p017 | 36 | 123 | 8.331856121309 | 128 |
| 1 | B1 | p015 | 39 | 121 | 9.032349215820 | 128 |
| 1 | B1 | p021 | 39 | 126 | 9.022395360284 | 128 |
| 1 | B1 | p085 | 46 | 126 | 10.774224679917 | 128 |
| 1 | B1 | p095 | 46 | 123 | 10.767720747739 | 128 |
| 1 | B1 | p058 | 43 | 126 | 10.074119511992 | 128 |
| 1 | B1 | p083 | 34 | 125 | 7.983515222557 | 128 |
| 2 | B4 | A(1-4) | 28 | 426 | 6.594811411574 | 512 |
| 2 | B4 | B(5-8) | 28 | 342 | 6.801139108837 | 512 |
| 3 | B4 | A(1-4) | 26 | 417 | 8.967796535231 | 512 |
| 3 | B4 | B(5-8) | 26 | 374 | 9.228163375519 | 512 |
| 4 | B1 | p072 | 26 | 118 | 8.957621013746 | 128 |
| 4 | B1 | p017 | 28 | 122 | 9.656124436297 | 128 |
| 4 | B1 | p015 | 35 | 122 | 12.034819783643 | 128 |
| 4 | B1 | p021 | 24 | 124 | 8.247500425205 | 128 |
| 4 | B1 | p085 | 38 | 125 | 13.227761984803 | 128 |
| 4 | B1 | p095 | 37 | 126 | 12.834117038175 | 128 |
| 4 | B1 | p058 | 35 | 124 | 12.164388533682 | 128 |
| 4 | B1 | p083 | 26 | 125 | 9.066790820099 | 128 |
| 5 | B1 | p072 | 39 | 125 | 9.712802954949 | 128 |
| 5 | B1 | p017 | 40 | 123 | 9.992829789408 | 128 |
| 5 | B1 | p015 | 40 | 123 | 9.965609173290 | 128 |
| 5 | B1 | p021 | 38 | 124 | 9.483131636865 | 128 |
| 5 | B1 | p085 | 46 | 126 | 11.674322916195 | 128 |
| 5 | B1 | p095 | 42 | 123 | 10.641424522735 | 128 |
| 5 | B1 | p058 | 35 | 126 | 8.903224444948 | 128 |
| 5 | B1 | p083 | 34 | 125 | 8.611314945854 | 128 |
| 6 | B4 | A(1-4) | 26 | 339 | 7.229520715773 | 512 |
| 6 | B4 | B(5-8) | 25 | 345 | 7.124956291169 | 512 |
| 7 | B1 | p072 | 39 | 125 | 9.697898291051 | 128 |
| 7 | B1 | p017 | 40 | 123 | 9.976352604106 | 128 |
| 7 | B1 | p015 | 40 | 123 | 9.977840754203 | 128 |
| 7 | B1 | p021 | 38 | 124 | 9.467790333554 | 128 |
| 7 | B1 | p085 | 46 | 126 | 11.636924442835 | 128 |
| 7 | B1 | p095 | 42 | 123 | 10.660560406744 | 128 |
| 7 | B1 | p058 | 35 | 126 | 8.895241212100 | 128 |
| 7 | B1 | p083 | 34 | 125 | 8.609861607663 | 128 |
| 8 | B4 | A(1-4) | 30 | 393 | 8.290255506523 | 512 |
| 8 | B4 | B(5-8) | 31 | 411 | 8.929946344346 | 512 |
| 9 | B4 | A(1-4) | 29 | 375 | 6.824562408961 | 512 |
| 9 | B4 | B(5-8) | 35 | 430 | 8.503089598380 | 512 |
| 10 | B1 | p072 | 29 | 125 | 6.717668409459 | 128 |
| 10 | B1 | p017 | 36 | 123 | 8.357220230624 | 128 |
| 10 | B1 | p015 | 39 | 121 | 9.031439428218 | 128 |
| 10 | B1 | p021 | 39 | 126 | 9.037083177827 | 128 |
| 10 | B1 | p085 | 46 | 126 | 10.806427562609 | 128 |
| 10 | B1 | p095 | 46 | 123 | 10.778680030257 | 128 |
| 10 | B1 | p058 | 43 | 126 | 10.111984508112 | 128 |
| 10 | B1 | p083 | 34 | 125 | 7.988685255870 | 128 |
| 11 | B1 | p072 | 26 | 118 | 8.968201107346 | 128 |
| 11 | B1 | p017 | 28 | 122 | 9.645558552817 | 128 |
| 11 | B1 | p015 | 35 | 122 | 12.017760825343 | 128 |
| 11 | B1 | p021 | 24 | 124 | 8.248385896906 | 128 |
| 11 | B1 | p085 | 38 | 125 | 13.208547896706 | 128 |
| 11 | B1 | p095 | 37 | 126 | 12.856001179665 | 128 |
| 11 | B1 | p058 | 35 | 124 | 12.176716109738 | 128 |
| 11 | B1 | p083 | 26 | 125 | 9.064057923853 | 128 |
| 12 | B4 | A(1-4) | 25 | 372 | 8.633423606865 | 512 |
| 12 | B4 | B(5-8) | 27 | 396 | 9.531285892241 | 512 |
| 13 | B1 | p072 | 26 | 118 | 8.987771199085 | 128 |
| 13 | B1 | p017 | 28 | 122 | 9.655591601506 | 128 |
| 13 | B1 | p015 | 35 | 122 | 12.028114078566 | 128 |
| 13 | B1 | p021 | 24 | 124 | 8.255396237597 | 128 |
| 13 | B1 | p085 | 38 | 125 | 13.192870846018 | 128 |
| 13 | B1 | p095 | 37 | 126 | 12.890444704331 | 128 |
| 13 | B1 | p058 | 35 | 124 | 12.175592318177 | 128 |
| 13 | B1 | p083 | 26 | 125 | 9.038591790013 | 128 |
| 14 | B4 | A(1-4) | 26 | 366 | 8.987729423679 | 512 |
| 14 | B4 | B(5-8) | 26 | 374 | 9.193826691248 | 512 |
| 15 | B4 | A(1-4) | 32 | 418 | 8.890990647487 | 512 |
| 15 | B4 | B(5-8) | 30 | 410 | 8.536247488111 | 512 |
| 16 | B1 | p072 | 29 | 123 | 7.245372891426 | 128 |
| 16 | B1 | p017 | 40 | 125 | 9.960125944577 | 128 |
| 16 | B1 | p015 | 40 | 122 | 9.981940879487 | 128 |
| 16 | B1 | p021 | 38 | 99 | 9.436939221807 | 101 |
| 16 | B1 | p085 | 45 | 125 | 11.386581439525 | 128 |
| 16 | B1 | p095 | 41 | 124 | 10.365417195484 | 128 |
| 16 | B1 | p058 | 36 | 126 | 9.160203848034 | 128 |
| 16 | B1 | p083 | 29 | 121 | 7.385676600970 | 128 |
| 17 | B1 | p072 | 29 | 125 | 6.729178548791 | 128 |
| 17 | B1 | p017 | 36 | 123 | 8.353571319953 | 128 |
| 17 | B1 | p015 | 39 | 121 | 9.024577124976 | 128 |
| 17 | B1 | p021 | 39 | 126 | 9.009168059565 | 128 |
| 17 | B1 | p085 | 46 | 126 | 10.773230584338 | 128 |
| 17 | B1 | p095 | 46 | 123 | 10.807341002859 | 128 |
| 17 | B1 | p058 | 43 | 126 | 10.113523843698 | 128 |
| 17 | B1 | p083 | 34 | 125 | 7.993085094728 | 128 |
| 18 | B4 | A(1-4) | 27 | 395 | 6.359670618549 | 512 |
| 18 | B4 | B(5-8) | 33 | 413 | 7.998409215361 | 512 |
