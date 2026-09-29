import hashlib,json,os,sys
from pathlib import Path
prefix=Path(sys.argv[1]); total=0; rows=[]; limit=512*1024*1024
while True:
 first=sys.stdin.buffer.read(4*1024*1024)
 if not first:break
 p=prefix.parent/(prefix.name+f".{len(rows):04d}.enc"); h=hashlib.sha256(); n=0
 with p.open("xb") as f:
  block=first
  while block:
   f.write(block); h.update(block); n+=len(block)
   if n>=limit:break
   block=sys.stdin.buffer.read(min(4*1024*1024,limit-n))
  f.flush();os.fsync(f.fileno())
 total+=n;rows.append({"file":p.name,"bytes":n,"sha256":h.hexdigest()})
 print(json.dumps({"chunk":p.name,"bytes":n,"total_bytes":total}),flush=True)
with (prefix.parent/(prefix.name+".chunks.json")).open("x") as f:json.dump({"bytes":total,"chunks":rows},f,indent=2)
