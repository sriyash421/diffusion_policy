"""Evaluation with unchanged dynamics and explicitly selected feedback/ranking."""
import numpy as np
from l2s.eval_openloop import eval_candidates as bfs_candidates, BatchedRollout, rollout_dt, _pad_steps
from run import validate_candidates as validate_bfs

def eval_candidates(policy, algo, episodes, n_candidates, rollout_steps, success_radius, device):
    if not any(c.__name__=='GoalDeltaPolicy' for c in type(policy).__mro__):  # subclasses too (GPT2GoalDeltaPolicy)
        return bfs_candidates(policy,algo,episodes,n_candidates,rollout_steps,success_radius,device)
    starts=np.stack([e['start'] for e in episodes]);goals=np.stack([e['goal'] for e in episodes])
    roll=BatchedRollout([e['env'] for e in episodes],starts,goals,success_radius,policy.n_obs_steps)
    policy.reset();trajs=[];dones=[];steps=[]
    for _ in range(n_candidates):
        _,tr,ds,ns=rollout_dt(policy,roll,device);trajs.append(tr);dones.append(ds);steps.append(ns)
    length=max(t.shape[1] for t in trajs)
    traj=np.stack(_pad_steps(trajs,length,repeat_last=True),axis=1)
    return dict(start=starts,goal=goals,nav=episodes[0]['nav'],traj=traj,done_step=np.stack(dones,1),n_steps=np.stack(steps,1),values=-np.linalg.norm(traj[:,:,-1]-goals[:,None],axis=-1))

def validate_candidates(z,variant='clean'):
    if variant!='delta':return validate_bfs(z)
    tr=z['traj'];assert np.allclose(tr,np.rint(tr));t=tr.astype(int);nav=z['nav']
    assert (t[...,0]>=0).all() and (t[...,0]<nav.shape[1]).all()
    assert (t[...,1]>=0).all() and (t[...,1]<nav.shape[0]).all()
    assert nav[t[...,1],t[...,0]].all()
    assert (np.abs(np.diff(t,axis=2)).sum(-1)<=1).all()
    assert np.allclose(-np.linalg.norm(tr[:,:,-1]-z['goal'][:,None],axis=-1),z['values'])
    reached=np.linalg.norm(tr[:,:,1:]-z['goal'][:,None,None],axis=-1)<=1
    done=np.where(reached.any(-1),reached.argmax(-1)+1,-1)
    assert np.array_equal(done,z['done_step'])
