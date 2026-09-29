"""Verify encrypted backup pointers and remote object availability without printing credentials."""
from pathlib import Path
import argparse,concurrent.futures,datetime,json,subprocess,urllib.request
p=argparse.ArgumentParser();p.add_argument('--branch',required=True);p.add_argument('--out',required=True);a=p.parse_args()
base=Path(__file__).resolve().parent;root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
commit=subprocess.check_output(['git','rev-parse',a.branch],text=True).strip()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+a.branch],text=True).split()
assert len(remote)==2 and remote[0]==commit,'remote branch differs'
objects=[]
for manifest in sorted(base.glob('*.chunks.json')):
 doc=json.loads(manifest.read_text());assert sum(r['bytes'] for r in doc['chunks'])==doc['bytes']
 for row in doc['chunks']:
  rel=(base/row['file']).relative_to(root).as_posix()
  pointer=subprocess.check_output(['git','show',commit+':'+rel],text=True)
  expected=f"version https://git-lfs.github.com/spec/v1\noid sha256:{row['sha256']}\nsize {row['bytes']}\n"
  assert pointer==expected,rel
  objects.append({'oid':row['sha256'],'size':row['bytes'],'path':rel})
assert objects,'no complete archives'
# Authentication stays in memory. The public receipt contains no signed URLs or headers.
auth=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','git@github.com-primary','git-lfs-authenticate','MaCoredroid/Lumo_FlyWheel.git','download']))
headers=dict(auth['header']);headers.update({'Accept':'application/vnd.git-lfs+json','Content-Type':'application/vnd.git-lfs+json'})
rows=[]
for start in range(0,len(objects),20):
 batch=objects[start:start+20]
 body={'operation':'download','transfers':['basic'],'objects':[{'oid':r['oid'],'size':r['size']} for r in batch],'ref':{'name':'refs/heads/'+a.branch}}
 request=urllib.request.Request(auth['href'].rstrip('/')+'/objects/batch',data=json.dumps(body).encode(),headers=headers)
 response=json.load(urllib.request.urlopen(request,timeout=60));found={r['oid']:r for r in response['objects']}
 def check(row):
  r=found[row['oid']];assert r.get('error') is None and r['size']==row['size']
  d=r['actions']['download'];req=urllib.request.Request(d['href'],headers=d.get('header',{}),method='HEAD')
  with urllib.request.urlopen(req,timeout=60) as result:
   assert result.status==200 and int(result.headers['Content-Length'])==row['size']
  return dict(row,remote_head_status=200)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows.extend(pool.map(check,batch))
 print(json.dumps({'objects_verified':len(rows),'total':len(objects)}),flush=True)
receipt={'schema':'lumo.encrypted-git-lfs-backup-verification.v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'branch':a.branch,'commit':commit,'remote_branch_matches':True,'complete_archive_count':len(list(base.glob('*.chunks.json'))),'objects':rows,'bytes':sum(r['size'] for r in rows),'verification':'Git pointers equal archive write-time SHA256/size manifests; authenticated remote LFS batch and HTTP HEAD for every object. Full decrypt/decompression/tar listing checked separately. No keys or signed URLs recorded.'}
Path(a.out).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='objects'}))
