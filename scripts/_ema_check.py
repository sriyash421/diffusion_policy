import sys, torch, dill
p = torch.load(open(sys.argv[1],'rb'), pickle_module=dill, map_location='cpu')
sd = p['state_dicts']; m, e = sd['model'], sd['ema_model']
diffs=[]
for k,v in m.items():
    if k in e and v.dtype.is_floating_point and v.numel()>0:
        diffs.append((k, float((v-e[k]).abs().max()), float(v.abs().max())))
nz=[d for _,d,_ in diffs if d>0]
print(f'  compared {len(diffs)} float tensors; {len(nz)} differ between live and EMA')
print(f'  max |live-ema| = {max(d for _,d,_ in diffs):.6g}')
print(f'  mean relative lag = {sum(d/(s+1e-12) for _,d,s in diffs)/len(diffs):.4%}')
print('  -> EMA ACTIVE (lagging live weights, as intended)' if nz else '  -> EMA INERT')
