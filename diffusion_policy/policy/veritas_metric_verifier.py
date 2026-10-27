"""VLM episode plan + observed-state metrics. No simulator success/contact flags.

Vector: [phase, target_tcp_xyz(3), grasp_radius, goal_xy_tol, source_xyz(3),
         held, released_stable]. Offline training must store the identical tracker
snapshots and use this same pure scorer. Candidate rollouts are kinematic estimates.
"""
import numpy as np
import torch
from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier, _last_obs_step
from diffusion_policy.env.simpler.veritas.camera import quat_wxyz_to_R

VERSION='veritas_metric_v1'
DIM=14
PHASES=('approach','grasp','lift','transport','lower','release','retreat')

class MetricTracker:
    def __init__(self, plan, revision='v1', calibration=None):
        self.plan=plan;self.initial_z=None;self.previous=None;self.held=False
        self.stable_steps=0;self.place_steps=0;self.phase=0
        self.revision=revision;self.calibration=calibration;self.initial_xy=None
        if revision not in ('v1','v2'):raise ValueError(revision)
        if revision=='v2' and calibration is None:raise ValueError('v2 requires train-only calibration')

    def update(self, tcp, src, tgt, opened):
        tcp=np.asarray(tcp)[:3];src=np.asarray(src)[:3];tgt_pose=np.asarray(tgt);tgt=tgt_pose[:3]
        p=self.plan
        if self.initial_z is None:self.initial_z=float(src[2])
        near=np.linalg.norm(tcp-src)<p.grasp_radius_m*1.8
        moved=0.;coupled=False
        if self.previous is not None:
            pt,ps=self.previous;moved=float(np.linalg.norm(src-ps))
            coupled=moved>.001 and np.linalg.norm((src-ps)-(tcp-pt))<.012
        if not opened and near and coupled and src[2]-self.initial_z>.012:self.held=True
        if opened or not near:self.held=False
        goal=tgt.copy();goal[2]+=p.goal_height_offset_m
        if self.revision=='v2':
            if len(tgt_pose)!=7:raise ValueError('target xyz+wxyz required for local goal offset')
            goal=tgt+quat_wxyz_to_R(tgt_pose[3:])@np.asarray(self.calibration['goal_offset_target_frame'])
        xy=float(np.linalg.norm(src[:2]-goal[:2]));zerr=abs(float(src[2]-goal[2]))
        if self.initial_xy is None:self.initial_xy=max(xy,p.goal_xy_tolerance_m)
        placed=xy<p.goal_xy_tolerance_m and zerr<.025
        self.place_steps=self.place_steps+1 if placed and moved<.008 else 0
        self.stable_steps=self.stable_steps+1 if placed and opened and moved<.005 else 0
        # Only executed object motion advances possession/placement. No waypoint-only
        # latching and no candidate updates to this tracker.
        if placed and (self.place_steps>=2 or self.stable_steps>0):
            self.phase=6 if opened else 5
            target=tcp.copy()
            if opened:target[2]+=p.lift_clearance_m
        elif self.held:
            if self.revision=='v2':
                # A continuous carrying target avoids lift/transport oscillations
                # and does not discard lateral progress during a diagonal lift.
                obj_target=goal.copy()
                clearance=p.lift_clearance_m*np.clip((xy-p.goal_xy_tolerance_m)/self.initial_xy,0.,1.)
                obj_target[2]+=clearance
                self.phase=4 if xy<=p.goal_xy_tolerance_m else 3
            elif xy>p.goal_xy_tolerance_m:
                obj_target=goal.copy();obj_target[2]=max(goal[2],self.initial_z)+p.lift_clearance_m
                self.phase=2 if src[2]<obj_target[2]-.025 else 3
                if self.phase==2:obj_target[:2]=src[:2]
            else:obj_target=goal;self.phase=4
            target=obj_target+(tcp-src)
        else:
            target=src.copy();target[2]+=p.grasp_offset_z_m
            self.phase=0 if np.linalg.norm(tcp-target)>p.grasp_radius_m else 1
            # A close-and-small-lift probe can establish possession. Requiring held
            # before any lift would deadlock grasp verification.
            if self.phase==1 and not opened:target[2]+=.025
        self.previous=(tcp.copy(),src.copy())
        self.metrics=dict(phase=PHASES[self.phase],held=self.held,xy_error=xy,z_error=zerr,
            source_motion=moved,coupled=bool(coupled),stable_steps=self.stable_steps,place_steps=self.place_steps)
        vector=[self.phase,*target,p.grasp_radius_m,p.goal_xy_tolerance_m,*src,
            float(self.held),float(self.stable_steps>=3),*p.stages[self.phase].uv,
            float(p.stages[self.phase].gripper=='open')]
        if self.revision=='v2':
            vector.extend([*goal,self.initial_z])
            self.metrics.update(goal_xyz=goal.tolist(),version='veritas_metric_v2')
        return np.array(vector,np.float32)

class VeritasMetricVerifier(VeritasPixelVerifier):
    def __init__(self,revision='v1',calibration=None,**kwargs):
        super().__init__(**kwargs)
        self.obs_keys=(self.ee_key,'veritas_metrics')
        if revision not in ('v1','v2'):raise ValueError(revision)
        self.revision=revision;self.calibration=calibration
        self.version='veritas_metric_'+revision

    def components(self,obs_dict,action):
        state=_last_obs_step(obs_dict['veritas_metrics']).to(action)
        tcp=_last_obs_step(obs_dict[self.ee_key]).to(action)[:,:3]
        a=action[:,:min(action.shape[1],self.horizon or action.shape[1])]
        R,scale,_,_=self._consts(action)
        delta=(a[...,:3]*scale)@R.T
        prefix=[]; position=tcp
        for i in range(a.shape[1]):
            position=position+delta[:,i]
            prefix.append(position)
        xyz=torch.stack(prefix,dim=1)
        phase=state[:,0].round().long();target=state[:,1:4];radius=state[:,4].clamp_min(.02)
        distance=torch.linalg.vector_norm(xyz[:,-1]-target,dim=-1)
        opened=(a[...,6]>=.5).to(action.dtype)
        # Approach can close after arriving; transport may never release early.
        grasp_region=torch.linalg.vector_norm(xyz-state[:,None,6:9],dim=-1)<radius[:,None]
        desired_open=torch.where((phase<=1)[:,None],(~grasp_region).to(action.dtype),
                                 state[:,13:14])
        if self.revision=='v2':
            if state.shape[-1]!=18:raise ValueError('v2 metric vector must have 18 entries')
            # Predicted closure is candidate-local, never written to the observed
            # possession tracker. After a plausible close, carrying the object away
            # must not be penalized as "closed outside grasp region".
            closed_near=torch.zeros_like(phase,dtype=torch.bool);desired=[]
            for i in range(a.shape[1]):
                closed_near=closed_near | (grasp_region[:,i] & (opened[:,i]==0))
                desired.append((~(grasp_region[:,i] | closed_near)).to(action.dtype))
            desired_open=torch.where((phase<=1)[:,None],torch.stack(desired,dim=1),state[:,13:14])
            # Score arrival anywhere in the executed prefix, so a chunk that grasps
            # then lifts is not rejected for ending beyond the old grasp waypoint.
            arrival=torch.linalg.vector_norm(xyz-target[:,None],dim=-1).amin(dim=1)
            distance=torch.where(phase<=1,arrival,distance)
        geometry=torch.exp(-distance/radius)
        grip_cost=(opened-desired_open).abs().mean(dim=1)
        dz=xyz[:,-1,2]-tcp[:,2]
        vertical_cost=torch.where((phase==2)|(phase==6),(-dz/.03).clamp(0,1),
                       torch.where(phase==4,(dz/.03).clamp(0,1),torch.zeros_like(dz)))
        pixel=self.project(xyz[:,-1])
        waypoint=torch.exp(-torch.linalg.vector_norm(pixel-state[:,11:13],dim=-1)/60.)
        score=.85*geometry+.15*waypoint-.75*grip_cost-.25*vertical_cost
        score=torch.nan_to_num(score,nan=-2.,posinf=-2.,neginf=-2.)
        return dict(score=score,geometry=geometry,waypoint=waypoint,gripper_cost=grip_cost,
                    vertical_cost=vertical_cost,phase=phase,endpoint_distance=distance)

    def get_value(self,obs_dict,action):
        with torch.no_grad():
            v=self.components(obs_dict,action)['score']
            return (v-self.value_mean)/self.value_std if self.normalize=='fixed' else v
