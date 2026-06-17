import torch, time, importlib.util
spec = importlib.util.spec_from_file_location("km", "/workspace/scripts/fr13_device_multidraft_kernel.py")
km = importlib.util.module_from_spec(spec); spec.loader.exec_module(km)
dev="cuda"; vocab=151936
cfgs=[("cat6",6,5,[-1,0,1,2,3,0]), ("cat9",9,5,[-1,0,1,2,3,0,1,2,3]), ("E5_chain",5,5,[-1,0,1,2,3])]
for label,nodes,maxspec,parents in cfgs:
    tl=torch.randn(nodes,vocab,device=dev); sl=torch.randn(nodes,vocab,device=dev)
    dti=torch.randint(0,vocab,(nodes,),device=dev,dtype=torch.long)
    tpi=torch.tensor(parents,device=dev,dtype=torch.long)
    ndt=torch.tensor([nodes],device=dev,dtype=torch.long)
    bonus=torch.randint(0,vocab,(1,),device=dev,dtype=torch.long)
    for _ in range(8): km.fr13_device_multidraft_commit(ndt,dti,tpi,tl,sl,None,bonus,maxspec)
    torch.cuda.synchronize(); t0=time.perf_counter(); N=60
    for _ in range(N): km.fr13_device_multidraft_commit(ndt,dti,tpi,tl,sl,None,bonus,maxspec)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/N*1000
    print(f"  committer {label:9s} ({nodes} nodes): {dt:.2f} ms/call")
