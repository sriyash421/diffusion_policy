"""Pixel Veritas + bounded relative cube-position score, shared by train/eval.

Kinematic surrogate: a closed gripper within grasp_radius of the source is treated
as carrying it; each commanded displacement moves that source until opening.
No simulated success/contact oracle, stage rewards, or gripper reward. The metric
is alignment, not proof of contact or stable placement.
"""
import torch
from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier, _last_obs_step

class VeritasStackPositionVerifier(VeritasPixelVerifier):
    version='veritas_stack_position_v1'
    def __init__(self,stack_weight=.5,position_sigma=.015,cube_height=.03,grasp_radius=.06,pixel_scale=105.,**kwargs):
        super().__init__(**kwargs)
        if not 0<=stack_weight<=1 or min(position_sigma,cube_height,grasp_radius,pixel_scale)<=0:raise ValueError('Invalid stack metric scales')
        self.stack_weight=float(stack_weight);self.position_sigma=float(position_sigma);self.cube_height=float(cube_height);self.grasp_radius=float(grasp_radius);self.pixel_scale=float(pixel_scale)
        self.obs_keys=(self.ee_key,self.targets_key,'insertive_asset_pose','receptive_asset_pose','last_gripper_action')
    def components(self,obs_dict,action):
        tcp=_last_obs_step(obs_dict[self.ee_key]).to(action)[:,:3]
        source=_last_obs_step(obs_dict['insertive_asset_pose']).to(action)[:,:3]
        target=_last_obs_step(obs_dict['receptive_asset_pose']).to(action)[:,:3]
        opened=_last_obs_step(obs_dict['last_gripper_action']).to(action).reshape(action.shape[0],-1)[:,0]>=.5
        held=(~opened)&(torch.linalg.vector_norm(tcp-source,dim=-1)<self.grasp_radius)
        predicted=source.clone();a=action[:,:min(action.shape[1],self.horizon or action.shape[1])];R,scale,_,_=self._consts(action)
        for t in range(a.shape[1]):
            held=held & (a[:,t,6]<.5)
            displacement=(a[:,t,:3]*scale)@R.T
            predicted=predicted+held[:,None]*displacement
        error=predicted-target;error=error-torch.tensor([0.,0.,self.cube_height],device=action.device,dtype=action.dtype)
        stack=torch.exp(-.5*((error/self.position_sigma)**2).sum(dim=-1))
        pixel=1/(1+self.cost(obs_dict,action).clamp_min(0)/self.pixel_scale)
        score=(1-self.stack_weight)*pixel+self.stack_weight*stack
        return dict(score=score,pixel_similarity=pixel,stack_alignment=stack,relative_error_xyz=error,predicted_source_xyz=predicted,held_proxy=(~opened)&(torch.linalg.vector_norm(tcp-source,dim=-1)<self.grasp_radius))
    def get_value(self,obs_dict,action):
        with torch.no_grad():
            v=self.components(obs_dict,action)['score']
            return (v-self.value_mean)/self.value_std if self.normalize=='fixed' else v
