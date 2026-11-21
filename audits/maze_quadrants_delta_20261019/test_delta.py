"""Regression checks for displacement feedback, collision semantics and BFS independence."""
import sys,json
from pathlib import Path
from setup_data import OUT,maze
import numpy as np,torch,hydra
from diffusion_policy.common.maze_verifier import MazeVerifier,build_images_torch,rollout_step_torch
from goal_delta_policy import GoalDeltaPolicy

def forbidden(*a,**k):raise AssertionError('BFS queried by goal-delta policy')
MazeVerifier._goal_distance_field_torch=forbidden;MazeVerifier._sample_goal_distance_torch=forbidden
p=Path(sys.argv[1]);payload=torch.load(p,map_location='cpu');cfg=payload['config'];cfg['policy']['device']='cpu';model=hydra.utils.instantiate(cfg['policy']);model.load_state_dict(payload['model']);model.eval();torch.set_num_threads(2)
nav=np.load(cfg['data']['layouts'])['nav'][0];free=torch.tensor(nav)[None];coords=np.argwhere(nav)[:,[1,0]]
pos=torch.tensor(coords[0:1],dtype=torch.float32);goal=torch.tensor(coords[-1:],dtype=torch.float32)
assert torch.allclose(model.goal_delta(pos,goal,free),(goal-pos)/10)
assert torch.equal(model.goal_delta(pos,pos,free),torch.zeros_like(pos))
f=torch.tensor([[.2,-.3,.4,-.5]])
assert torch.equal(model._mask_fb(f),torch.tensor([[0.,0.,.4,-.5]]) if model.reward_mode=='traj' else f)
# A wall collision must leave the feedback unchanged.
found=False
for xy in coords:
 for action in [[1,0],[-1,0],[0,1],[0,-1]]:
  p0=torch.tensor(xy,dtype=torch.float32)[None];a=torch.tensor(action,dtype=torch.float32)[None]
  p1=rollout_step_torch(free,p0,a)
  if torch.equal(p0,p1):
   assert torch.equal(model.goal_delta(p0,goal,free),model.goal_delta(p1,goal,free));found=True;break
 if found:break
assert found
with torch.inference_mode():
 z=model.rollout_candidates_torch(free,pos,goal,2)
 assert torch.allclose(z['values'],-(z['traj'][:,:,-1]-goal[:,None]).norm(dim=-1))
 for k,snap in enumerate(z['snaps']):
  delta=(goal-z['traj'][:,k,-1])/10
  assert torch.allclose(snap['last_fb'],torch.cat([delta,delta],-1))
# Teacher forcing, generated contexts and gradients must also work with BFS disabled.
from diffusion_policy.dataset.sim2real_image_dataset import Sim2RealImageDataset
from diffusion_policy.common.sampler import get_collate_fn
spec=cfg['data'];ds=Sim2RealImageDataset(spec['dataset'],cfg['policy']['shape_meta'],horizon=100,pad_before=0,pad_after=0,n_obs_steps=None,seed=0,val_ratio=.2,use_cache=False,use_disk=False,return_sequences=True,zero_pad_action_after=False,val_split='pair')
batch=next(iter(torch.utils.data.DataLoader(ds,batch_size=2,collate_fn=get_collate_fn())))
loss=model.compute_loss(batch);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print('PASS displacement signs/scaling, masks, blocked move, terminal feedback, Euclidean scoring, teacher forcing, backward, no BFS')
