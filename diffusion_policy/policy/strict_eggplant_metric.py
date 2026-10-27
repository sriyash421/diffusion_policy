"""Online grasp/carry proxy for strict-filter stages 2 and 3.

No eggplant-to-basket progress reward. Before first grasp reward EE approach;
afterwards penalize separation and reopening until observed basket arrival.
Future contact is not simulated: a currently grasped object follows closed-gripper
translations, and is frozen after opening. This is not contact certification.
State: source[3], target[3], grasp_now, initial_ee_distance, first_grasp_distance,
       basket_arrived, grasp_seen.
"""
import torch
from diffusion_policy.policy.veritas_verifier import _last_obs_step

def components(verifier,obs,action):
    s=_last_obs_step(obs['strict_task_state']).to(action)
    src=s[:,:3];grasp_now=s[:,6]>.5;seen=s[:,10]>.5;arrived=s[:,9]>.5
    a=action[:,:min(action.shape[1],verifier.horizon or action.shape[1])]
    R,scale,_,_=verifier._consts(a)
    delta=(a[:,:,:3]*scale)@R.T
    closed=a[:,:,6]<.5
    carrying=torch.cumprod(closed.to(a.dtype),dim=1)*grasp_now[:,None]
    predicted_src=src[:,None]+torch.cumsum(delta*carrying[:,:,None],dim=1)
    tcp0=_last_obs_step(obs[verifier.ee_key]).to(a)[:,:3]
    tcp=tcp0[:,None]+torch.cumsum(delta,dim=1)
    separation=torch.linalg.vector_norm(tcp-predicted_src,dim=-1)
    current=torch.linalg.vector_norm(tcp0-src,dim=-1)
    approach=((current-separation[:,-1])/s[:,7].clamp(min=.05)).clamp(-1,1)
    max_sep=separation.amax(dim=1)
    carry=((max_sep-.03).clamp(min=0)/.03).clamp(max=1)
    rise=((max_sep-s[:,8]-.015).clamp(min=0)/.015).clamp(max=1)
    reopen=((~closed[:,:-1])&(~closed[:,1:])).any(dim=1).to(a.dtype)
    active=seen&~arrived
    score=torch.where(seen,-(carry+rise+reopen)/3,approach)*~arrived
    return dict(task_score=score,task_approach_progress=approach*~seen,
        task_carry_penalty=carry*active,task_separation_rise_penalty=rise*active,
        task_reopen_penalty=reopen*active,predicted_max_gripper_object_distance=max_sep)
