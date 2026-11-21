"""How much of the agent / goal position survives the training-time observation noise (asteroid.noisy_obs) on interp30.
Features = normalizer + FlattenObsEncoder of every step of the train demos; std per dimension as noisy_obs uses it.
For each DDPM t: x' = sqrt(ab) x + sqrt(1-ab) eps std. Decode the agent cell and the goal cell (a) by argmax over the free cells of
their one-hot image channel, (b) from the (x, y) coordinate features (nearest free cell), (c) best of both per sample."""
import os, sys
sys.path[:0] = ['/gscratch/robotics/harine/codex_audits/maze_rounds_20261103']
os.environ.setdefault('MAZE_SETTING', 'interp30')
import numpy as np, torch
import rounds
from asteroid import make_policy

torch.manual_seed(0)
model, pc = make_policy(8, 'cpu')
ds = rounds.load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer())
obs = {k: [] for k in ('image', 'agent_pos', 'goal_pos')}
for i in range(len(ds)):
    o = ds[i]['obs']
    for k in obs:
        obs[k].append(torch.as_tensor(np.asarray(o[k])))
obs = {k: torch.cat(v).float() for k, v in obs.items()}
with torch.no_grad():
    x = model._obs_features(obs)
std = x.std(0).clamp_min(1e-6)
H = W = 11; n_img = 3 * H * W
img = obs['image']                              # (N, 3, 11, 11) raw
free = (img[0, 0] < 0.5).flatten()              # channel 0 = walls
agent = img[:, 2].flatten(1).argmax(1); goal = img[:, 1].flatten(1).argmax(1)
cells = torch.nonzero(free).squeeze(1)
cx, cy = (cells % W).float(), (cells // W).float()
ap_f, gp_f = slice(n_img, n_img + 2), slice(n_img + 2, n_img + 4)
print('N steps', len(x), 'feature dims', x.shape[1], 'free cells', len(cells))
print('feature std: wall px %.3g | goal px median %.3f | agent px median %.3f | agent xy %s | goal xy %s' % (
    std[:H * W].max(), std[H * W:2 * H * W][free].median(), std[2 * H * W:3 * H * W][free].median(), std[ap_f].numpy().round(3), std[gp_f].numpy().round(3)))
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
AB = DDPMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='linear').alphas_cumprod
# coordinate features of each free cell (clean), to decode xy by nearest cell
def coord_feat(off):
    o = {'image': img[:1].repeat(len(cells), 1, 1, 1), 'agent_pos': torch.stack([cx, cy], 1), 'goal_pos': torch.stack([cx, cy], 1)}
    with torch.no_grad():
        return model._obs_features(o)[:, off]
cf_a, cf_g = coord_feat(ap_f), coord_feat(gp_f)
print(f"{'t':>4} {'sqrt(ab)':>8} | agent cell: image  xy  either | goal cell: image  xy  either | xy err cells (agent)")
for t in [0, 200, 300, 400, 500, 600, 700, 800]:
    ab = float(AB[t]) if t else 1.0
    xn = ab ** .5 * x + (1 - ab) ** .5 * torch.randn_like(x) * std
    res = []
    for ch, truth, off, cf in ((2, agent, ap_f, cf_a), (1, goal, gp_f, cf_g)):
        px = xn[:, ch * H * W:(ch + 1) * H * W][:, cells]
        a_img = cells[px.argmax(1)] == truth
        d = ((xn[:, off][:, None, :] / ab ** .5 - cf[None]) ** 2).sum(-1)    # nearest cell to the rescaled coordinates
        a_xy = cells[d.argmin(1)] == truth
        res += [a_img.float().mean(), a_xy.float().mean(), (a_img | a_xy).float().mean()]
        if ch == 2:
            p = cells[d.argmin(1)]; err = ((p % W - truth % W).abs() + (p // W - truth // W).abs()).float().mean()
    print(f'{t:>4} {ab ** .5:8.2f} |        {res[0]:.2f}  {res[1]:.2f}  {res[2]:.2f}  |       {res[3]:.2f}  {res[4]:.2f}  {res[5]:.2f}  | {err:.1f}')
