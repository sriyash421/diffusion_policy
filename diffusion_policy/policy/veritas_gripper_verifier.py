"""Existing Veritas pixel cost plus one additive open/close consistency penalty.

No metric-phase tracker, learned target offsets, or controller gain calibration.
Weight zero preserves the existing VeritasPixelVerifier ranking. Pixel cost is
divided by one default waypoint tolerance times the full lookahead weight sum.
"""
import numpy as np
import torch
from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier, _last_obs_step
from diffusion_policy.env.simpler.veritas.tracker import SimplerEpisodeTracker


class PixelGripperTracker:
    def __init__(self, plan, camera, start_xyz, initial_open=True):
        self.episode=SimplerEpisodeTracker(plan,camera,start_xyz)
        # The existing tracker prepends an initial-position anchor.
        self.open_states=[float(initial_open)]+[float(w.gripper=='open') for w in plan.waypoints]
        self.first=True

    def update(self,tcp,*unused):
        # Match the wrapper: no tracker advancement on reset, once after each step.
        if not self.first:self.episode.update(np.asarray(tcp)[:3])
        self.first=False
        tr=self.episode.tracker;ids=tr.upcoming(3) if not tr.done else []
        prev=self.open_states[max(0,min(tr.i-1,len(self.open_states)-1))]
        grip=[prev]+[self.open_states[i] for i in ids]
        grip += [-1.]*(4-len(grip))
        grip += [self.open_states[-1]]
        self.metrics=dict(phase='waypoint',held=None,wp_active_idx=int(tr.i),
            wp_progress=tr.progress,wp_done=bool(tr.done),version='veritas_pixel_gripper_v1')
        return np.concatenate([self.episode.targets_vector(3),np.asarray(grip,np.float32)])


class VeritasGripperVerifier(VeritasPixelVerifier):
    def __init__(self,gripper_weight=0.1,task_weight=0.,**kwargs):
        super().__init__(**kwargs)
        if not 0<=gripper_weight<=0.1:raise ValueError('gripper weight must be in [0,0.1]')
        self.gripper_weight=float(gripper_weight)
        self.task_weight=float(task_weight)
        if not 0<=self.task_weight<=1:raise ValueError("task weight must be in [0,1]")
        self.pixel_cost_scale=self.default_tol_px*sum(self.lookahead_alpha**i for i in range(self.lookahead_k))
        self.obs_keys=(self.ee_key,self.targets_key,'veritas_gripper_targets')
        if self.task_weight:self.obs_keys+=('strict_task_state',)
        self.version='veritas_pixel_gripper_v1'

    def components(self,obs_dict,action):
        pixel_cost=super().cost(obs_dict,action)
        target=_last_obs_step(obs_dict[self.targets_key]).to(action)
        grip=_last_obs_step(obs_dict['veritas_gripper_targets']).to(action)
        count=target[:,0].long();tol=target[:,1]
        waypoints=target[:,2:8].reshape(-1,3,2)
        a=action[:,:min(action.shape[1],self.horizon or action.shape[1])]
        R,scale,_,_=self._consts(action)
        pos=_last_obs_step(obs_dict[self.ee_key]).to(action)[:,:3]
        active=torch.zeros_like(count)
        desired=torch.where(count>0,grip[:,0],grip[:,4])
        costs=[]
        # This progress is private to scoring a candidate. It never advances the
        # real waypoint tracker and never claims a physical grasp occurred.
        for i in range(a.shape[1]):
            pos=pos+(a[:,i,:3]*scale)@R.T
            px=self.project(pos)
            for _ in range(3):
                idx=active.clamp(max=2)
                goal=waypoints.gather(1,idx[:,None,None].expand(-1,1,2)).squeeze(1)
                hit=(active<count)&(torch.linalg.vector_norm(px-goal,dim=-1)<tol)
                next_open=grip[:,1:4].gather(1,idx[:,None]).squeeze(1)
                desired=torch.where(hit,next_open,desired)
                active=active+hit.long()
            costs.append(((a[:,i,6]>=.5).to(action.dtype)-desired).abs())
        mismatch=torch.stack(costs,dim=1).mean(dim=1)
        normalized_pixel_cost=pixel_cost/self.pixel_cost_scale
        gripper_penalty=self.gripper_weight*mismatch
        score=-normalized_pixel_cost-gripper_penalty
        result=dict(score=score,pixel_cost=pixel_cost,normalized_pixel_cost=normalized_pixel_cost,
                    gripper_cost=mismatch,gripper_penalty=gripper_penalty)
        if self.task_weight:
            from diffusion_policy.policy.strict_eggplant_metric import components
            task=components(self,obs_dict,action)
            result.update(task);result['trajectory_score']=score
            result['score']=score+self.task_weight*task['task_score']
        return result

    def get_value(self,obs_dict,action):
        with torch.no_grad():
            v=self.components(obs_dict,action)['score']
            return (v-self.value_mean)/self.value_std if self.normalize=='fixed' else v
