import os,sys,json,argparse,csv,fcntl
from pathlib import Path
OUT=Path(__file__).resolve().parent;BASE=OUT.parent
sys.path.insert(0,str(OUT))
from setup_data import maze
import numpy as np,torch,hydra
from l2s.eval_openloop import BatchedRollout,build_episodes,eval_candidates
from l2s.verifier import MazeVerifier,snap_cardinal_torch
sys.path.insert(0,str(BASE/'unet64_20261008'))
from run import writecsv
from evaluation_common import eval_candidates,validate_candidates

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--episodes',type=int,default=100);ap.add_argument('--device',default='cuda');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    path=Path(args.checkpoint);dest=path.parent/'checkpoint_evals_n64'/path.stem
    if args.smoke:dest=path.parent/'smoke_evals'/path.stem
    dest.mkdir(parents=True,exist_ok=True);lock=(dest/'eval.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    if (dest/'COMPLETED').exists():return
    torch.set_num_threads(4);torch.manual_seed(20261009);np.random.seed(20261009)
    payload=torch.load(path,map_location='cpu');config=payload['config'];spec=config['data'];step=int(payload['step']);args.algo=config['algo'];device=args.device;isdt=args.algo.startswith('dt_');H=spec['horizon']
    if isdt:config['policy']['device']=device
    model=hydra.utils.instantiate(config['policy']);model.load_state_dict(payload['model']);model.to(device).eval();del payload
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
                for t in range(H):roll.step(a[:,t])
                tr.append(roll.trajectory(H));done.append(roll.done_step.copy());acts.append(a)
                v=MazeVerifier(env=eps[0]['env'],device='cpu');vals.append(-v.goal_distance(torch.tensor(roll.pos),torch.tensor(roll.goal)).numpy())
            return dict(traj=np.stack(tr,1),done_step=np.stack(done,1),sampled_actions=np.stack(acts,1),values=np.stack(vals,1),start=np.array([e['start'] for e in eps]),goal=np.array([e['goal'] for e in eps]),nav=eps[0]['nav'])
    from diffusion_policy.dataset.sim2real_image_dataset import Sim2RealImageDataset
    shape=config['policy']['shape_meta']
    ds=Sim2RealImageDataset(spec['dataset'],shape,horizon=max(100,spec['expert_max']) if isdt else H,pad_before=0,pad_after=0 if isdt else H-1,n_obs_steps=None if isdt else 1,seed=config['split_seed'],val_ratio=.2,use_cache=False,use_disk=False,return_sequences=isdt,zero_pad_action_after=not isdt,val_split='pair')
    assert int(ds.val_mask.sum())==config['val_episodes']
    rb=ds.replay_buffer;starts=np.r_[0,np.asarray(rb.episode_ends)[:-1]][ds.val_mask];ve=[]
    for ix in starts[:2] if args.smoke else starts:
        env=maze.MazeEnv(spec['layouts'],seed=0,split='train',layout_index=0,goal_split=spec['goal_split']);env.reset()
        st=np.asarray(rb['agent_pos'][ix]);goal=np.asarray(rb['goal_pos'][ix]);env.goal=goal.astype(np.int32).copy()
        ve.append(dict(env=env,start=st,goal=goal,nav=env.maze))
    pairs=maze.load_train_pairs(spec['dataset']);metrics=[]
    ns=[1,2,4,8,16,32,64]
    for split in ['validation','eval','train_heldout']:
        eps=ve if split=='validation' else build_episodes(spec['layouts'],split,0,2 if args.smoke else args.episodes,spec['goal_split'],0,pairs)
        saved=dest/f'{split}_candidates.npz'
        if saved.exists():
            with np.load(saved) as loaded:z={k:loaded[k] for k in loaded.files}
        else:
            z=evaluate(eps,max(ns),20261008 if split=='validation' else 20261009);validate_candidates(z,config.get("variant","clean"))
            with (dest/f'{split}_candidates.tmp').open('wb') as f:np.savez_compressed(f,**z)
            os.replace(dest/f'{split}_candidates.tmp',saved)
        assert z['values'].shape==(len(eps),max(ns))
        validate_candidates(z,config.get("variant","clean"))
        for n in ns:
            for strategy in ['max','last']:
                idx=z['values'][:,:n].argmax(1) if strategy=='max' else np.full(len(eps),n-1)
                metrics.append(dict(setting=config['setting'],algo=args.algo,train_seed=config['train_seed'],checkpoint=step,variant=config.get('variant','clean'),split=split,n=n,strategy=strategy,success_rate=float((z['done_step'][np.arange(len(eps)),idx]>=0).mean()),episodes=len(eps),source_checkpoint=str(path)))
        print(split,step,'evaluated',len(eps),flush=True)
    writecsv(dest/'metrics.csv',metrics);(dest/'COMPLETED').write_text('Validation and both test splits: independent verifier/movement/success checks passed.\n')
if __name__=='__main__':main()
