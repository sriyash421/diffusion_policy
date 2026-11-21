"""Exact-update training, checkpoint evaluation and full-trajectory test evaluation."""
import argparse,csv,json,os,random,sys
from pathlib import Path
from setup_data import ROOT,OUT,maze
import numpy as np
import torch,hydra
from diffusion_policy.dataset.sim2real_image_dataset import Sim2RealImageDataset
from diffusion_policy.common.sampler import get_collate_fn
from diffusion_policy.common.pytorch_util import dict_apply
from l2s.eval_openloop import BatchedRollout,build_episodes,eval_candidates,_pad_steps
from l2s.verifier import MazeVerifier,snap_cardinal_torch
sys.path.insert(0,str(OUT.parent/'unet64_20261008'))
from run import writecsv
from evaluation_common import eval_candidates,validate_candidates

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--setting',choices=['11br','11tl','11tr','11bl','17br','17tl','17tr','17bl','interp30','interp30c'],required=True);ap.add_argument('--algo',choices=['bc_unet','dt_bon','dt_st_traj','dt_st_step'],required=True);ap.add_argument('--variant',choices=['clean','noisy','delta'],default='clean');ap.add_argument('--seed',type=int,required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',default='cuda');ap.add_argument('--backbone',choices=['encoder','gpt2'],default='encoder');ap.add_argument('--out-root',default=None);ap.add_argument('--kv-cache',action='store_true');ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--schedule',choices=['cosine','constant'],default='cosine');ap.add_argument('--updates',type=int,default=10000);ap.add_argument('--tag',default='');ap.add_argument('--batch',type=int,default=64);ap.add_argument('--micro',type=int,default=0);ap.add_argument('--dataset',default='',help='zarr overriding the setting\'s demos (e.g. the DAgger-relabelled data of maze_interp_20261105/dagger_bc_data.py)');ap.add_argument('--val-ratio',type=float,default=.2);ap.add_argument('--horizon',type=int,default=0,help='bc_unet: prediction horizon (0 = the setting\'s, 64); below it, checkpoint validation runs closed loop (re-plan every step, execute the first action) for the setting\'s horizon');args=ap.parse_args()
    device=args.device;spec=json.loads((OUT/'datasets.json').read_text())[args.setting];spec=dict(spec,dataset=args.dataset) if args.dataset else spec;H=args.horizon or spec['horizon'];CL=spec['horizon'] if H<spec['horizon'] else 0;isdt=args.algo.startswith('dt_')
    out=(Path(args.out_root) if args.out_root else OUT)/(('cpu_smoke' if device=='cpu' else 'smoke') if args.smoke else 'runs')/args.setting/(args.algo+'__'+args.variant+('__gpt2' if args.backbone=='gpt2' else '')+('__'+args.tag if args.tag else ''))/f'seed_{args.seed}';out.mkdir(parents=True,exist_ok=True)
    total=2 if args.smoke else args.updates
    def trained():  # TRAINED records its update count; older runs (no count) trained 10k
        if not (out/'TRAINED').exists():return False
        import re as _re;m=_re.search(r'updates=(\d+)',(out/'TRAINED').read_text());return (int(m.group(1)) if m else 10000)>=total
    if trained():return
    import fcntl
    lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    if trained():return
    # several runs packed on one GPU (maze_traj17 train.sbatch): cap each process's share, or one caching allocator starves the other
    if device=='cuda' and os.environ.get('CUDA_MEM_FRACTION'):torch.cuda.set_per_process_memory_fraction(float(os.environ['CUDA_MEM_FRACTION']))
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    shape=dict(obs=dict(image=dict(shape=[3]+spec['shape'],type='rgb'),agent_pos=dict(shape=[2],type='low_dim'),goal_pos=dict(shape=[2],type='low_dim')),action=dict(shape=[2]))
    encoder=dict(_target_='diffusion_policy.model.vision.multi_image_obs_encoder.FlattenObsEncoder',shape_meta=shape)
    if isdt:
        pc=dict(_target_='diffusion_policy.policy.dt_search_policy.DTSearchPolicy',shape_meta=shape,obs_encoder=encoder,horizon=1,n_action_steps=1,n_obs_steps=1,max_actions=1 if args.algo=='dt_bon' else 8,reward_mode='step' if args.algo=='dt_st_step' else 'traj',stop_radius=1.,n_emb=256,n_layer=4,n_head=4,p_drop=.1,log_std_init=-1.,log_std_min=-2.,snap_cardinal=True,maze_path=None,device=device,verifier_noise=0.)
    else:
        pc=json.loads((OUT.parent/'unet64_20261008/production/config.json').read_text())['policy'];pc.update(shape_meta=shape,obs_encoder=encoder,horizon=H,n_action_steps=H)
    if args.variant!='clean':
        assert args.algo in ['dt_st_traj','dt_st_step']
        pc['_target_']='dt_obsnoise_policy.NoisyDTSearchPolicy' if args.variant=='noisy' else 'goal_delta_policy.GoalDeltaPolicy'
    if args.backbone=='gpt2':  # GPT-2 trunk (maze_delta_gpt2_20261026/gpt2_dt.py); only clean (BFS) and delta (x,y)
        assert args.algo.startswith('dt_') and args.variant in ['clean','delta']
        pc['_target_']='gpt2_dt.GPT2GoalDeltaPolicy' if args.variant=='delta' else 'gpt2_dt.GPT2DTSearchPolicy';pc['n_positions']=1024
    if args.kv_cache:  # rollouts encode one new token per step (DTSearchPolicy kv_cache); the BFS encoder trunk only
        assert isdt and args.variant=='clean' and args.backbone=='encoder';pc['kv_cache']=True
    ds=Sim2RealImageDataset(spec['dataset'],shape,horizon=max(100,spec['expert_max']) if isdt else H,pad_before=0,pad_after=0 if isdt else H-1,n_obs_steps=None if isdt else 1,seed=0,val_ratio=args.val_ratio,use_cache=False,use_disk=False,return_sequences=isdt,zero_pad_action_after=not isdt,val_split='pair')
    model=hydra.utils.instantiate(pc);model.set_normalizer(ds.get_normalizer());model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.95,.999),eps=1e-8,weight_decay=1e-6)
    every=1 if args.smoke else 1000;batch_size=2 if args.smoke else args.batch
    # Smaller DT microbatches limit quadratic attention memory; valid-token weighting preserves the loss.
    micro=args.micro or (8 if isdt else batch_size)  # --micro: rows per forward/backward; the loss is token-weighted, so any value gives the same update
    selection='latest' if isdt else 'validation_n1'
    config=dict(horizon=H,validation_closed_loop_steps=CL,backbone=args.backbone,kv_cache=args.kv_cache,context_rollout='batch' if args.kv_cache else 'microbatch',variant=args.variant,feedback_type='goal_delta' if args.variant=='delta' else 'bfs',ranking='negative_euclidean' if args.variant=='delta' else 'negative_bfs',setting=args.setting,algo=args.algo,train_seed=args.seed,policy=pc,data=spec,updates=total,batch_size=batch_size,microbatch=micro,split_seed=0,train_episodes=int(ds.train_mask.sum()),val_episodes=int(ds.val_mask.sum()),selection=selection,checkpoint_every=every,lr=args.lr,schedule=args.schedule,warmup=500,gradient_clip=1.,ema=False,parameters=sum(p.numel() for p in model.parameters()))
    (out/'config.json').write_text(json.dumps(config,indent=2))
    rb=ds.replay_buffer;starts=np.r_[0,np.asarray(rb.episode_ends)[:-1]][ds.val_mask];ve=[]
    for ix in starts[:2] if args.smoke else starts:
        env=maze.MazeEnv(spec['layouts'],seed=0,split='train',layout_index=0,goal_split=spec['goal_split']);env.reset();s=np.asarray(rb['agent_pos'][ix]);g=np.asarray(rb['goal_pos'][ix]);env.goal=g.astype(np.int32).copy();ve.append(dict(env=env,start=s,goal=g,nav=env.maze))
    def evaluate(eps,n,rng_seed):
        model.eval()
        with torch.random.fork_rng(devices=[0] if device=='cuda' else []),torch.inference_mode():
            torch.manual_seed(rng_seed)
            if isdt:
                zz=[eval_candidates(model,args.algo,eps[j:j+8],n,H,1.,device) for j in range(0,len(eps),8)]
                T=max(z['traj'].shape[2] for z in zz)
                for z in zz:
                    pad=T-z['traj'].shape[2]
                    if pad:z['traj']=np.concatenate([z['traj'],np.repeat(z['traj'][:,:,-1:],pad,axis=2)],axis=2)
                result={k:np.concatenate([z[k] for z in zz]) for k in ['start','goal','traj','done_step','values','n_steps']};result['nav']=eps[0]['nav'];return result
            tr=[];done=[];acts=[];vals=[]
            for k in range(n):
                roll=BatchedRollout([e['env'] for e in eps],[e['start'] for e in eps],[e['goal'] for e in eps],1.,1);obs=roll.obs_tensor(device);aa=[]
                for j in range(0,len(eps),16):
                    pred=model.predict_action({key:v[j:j+16] for key,v in obs.items()})['action'];assert pred.shape[1:]==(H,2);aa.append(snap_cardinal_torch(pred).cpu().numpy())
                a=np.concatenate(aa);assert np.isfinite(a).all()
                if CL:  # closed loop (--horizon below the setting's): re-plan every step, execute only the first action, CL steps
                    ex=[]
                    for t in range(CL):
                        obs=roll.obs_tensor(device);aa=[model.predict_action({key:v[j:j+16] for key,v in obs.items()})['action'][:,0] for j in range(0,len(eps),16)]
                        ex.append(snap_cardinal_torch(torch.cat(aa)).cpu().numpy());roll.step(ex[-1])
                    a=np.stack(ex,1)
                else:
                    for t in range(H):roll.step(a[:,t])
                tr.append(roll.trajectory(CL or H));done.append(roll.done_step.copy());acts.append(a)
                v=MazeVerifier(env=eps[0]['env'],device='cpu');vals.append(-v.goal_distance(torch.tensor(roll.pos),torch.tensor(roll.goal)).numpy())
            return dict(traj=np.stack(tr,1),done_step=np.stack(done,1),sampled_actions=np.stack(acts,1),values=np.stack(vals,1),start=np.array([e['start'] for e in eps]),goal=np.array([e['goal'] for e in eps]),nav=eps[0]['nav'])
    collate=get_collate_fn() if isdt else None
    # Separate loader generator permits exact replay after preemption without resetting model RNG.
    gen=torch.Generator();gen.manual_seed(args.seed);epoch=0;offset=0
    loader=torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True,collate_fn=collate,generator=gen)
    epoch_rng=gen.get_state();it=iter(loader);rows=[];first=1;resume=out/'resume.pt'
    if resume.exists():
        st=torch.load(resume,map_location='cpu');model.load_state_dict(st['model']);model.to(device);optimizer.load_state_dict(st['optimizer']);first=st['step']+1;rows=st['rows'];epoch=st['epoch'];offset=st['offset'];epoch_rng=st['epoch_rng'];gen.set_state(epoch_rng);it=iter(loader)
        for _ in range(offset):next(it)
        torch.set_rng_state(st['rng']);np.random.set_state(st['numpy_rng']);random.setstate(st['python_rng'])
        if device=='cuda':torch.cuda.set_rng_state(st['cuda_rng'])
    for step in range(first,total+1):
        try:b=next(it);offset+=1
        except StopIteration:
            epoch+=1;offset=1;epoch_rng=gen.get_state();it=iter(loader);b=next(it)
        model.train();lr=args.lr*(step/500 if step<=500 else 1. if args.schedule=='constant' else .5*(1+np.cos(np.pi*(step-500)/(total-500))))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad();loss_value=0.;denom=b['attention_mask'].sum().item() if isdt else batch_size
        # --kv-cache: the K context attempts are rolled out once for the whole batch (cost is per env step), then sliced
        cands=model.context_candidates(dict_apply(b,lambda x:x.to(device))) if args.kv_cache else None
        for j in range(0,batch_size,micro):
            part=dict_apply(b,lambda x:x[j:j+micro].to(device));weight=part['attention_mask'].sum().item()/denom if isdt else len(part['action'])/batch_size
            loss=model.compute_loss(part,cands=model.slice_candidates(cands,slice(j,j+micro))) if cands is not None else model.compute_loss(part);assert torch.isfinite(loss);(loss*weight).backward();loss_value+=loss.item()*weight
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%100==0 or args.smoke:print(json.dumps(dict(step=step,loss=loss_value)),flush=True)
        if step%every==0 or step==total:
            path=out/f'step_{step:07d}.pt';torch.save(dict(model=model.state_dict(),step=step,config=config),out/'saving.pt');os.replace(out/'saving.pt',path)
            rate=float('nan')  # interp30c (continuous maze): no grid validation here; checkpoints are picked by bc_closed.py --select
            if not spec.get('continuous'):result=evaluate(ve,1,20261008);validate_candidates(result,args.variant);rate=float((result['done_step'][:,0]>=0).mean());rows.append(dict(step=step,val_n1_success=rate,episodes=len(ve),checkpoint=str(path)));writecsv(out/'validation.csv',rows);print('VALIDATION',rows[-1],flush=True)
        if step%100==0 or step%every==0 or step==total:
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,rows=rows,epoch=epoch,offset=offset,epoch_rng=epoch_rng,rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None)
            torch.save(state,out/'resume.tmp');os.replace(out/'resume.tmp',resume)
    (out/'TRAINED').write_text(f'updates={total}\nAll optimizer updates and checkpoint validation complete; N64 evaluation tracked separately.\n')
if __name__=='__main__':main()
