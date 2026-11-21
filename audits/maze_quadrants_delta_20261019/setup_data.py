"""Fixed map, dataset seed and pair split for the four quadrant experiments."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path('/gscratch/robotics/harine/L2S');OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'diffusion_policy')]
import l2s.maze as maze
_original_goal_pool=maze.goal_pool

INTERP_GOALS=Path('/gscratch/robotics/harine/codex_audits/maze_interp_20261105/goals.json')

def goal_pool(nav,goal_split,role,center=None):
    if goal_split=='random30':  # interpolation split (maze_interp_20261105): 30 random train goal cells, the other 41 held out
        g=json.loads(INTERP_GOALS.read_text());return np.asarray(g['train' if role=='train' else 'test'],np.int32).reshape(-1,2)
    if goal_split not in ['quadrant_tl','quadrant_tr','quadrant_bl']:return _original_goal_pool(nav,goal_split,role,center)
    cy,cx=maze.split_center(nav) if center is None else center
    free=np.argwhere(nav)[:,[1,0]].astype(np.int32);x,y=free.T
    held={'quadrant_tl':(x<cx)&(y<cy),'quadrant_tr':(x>cx)&(y<cy),'quadrant_bl':(x<cx)&(y>cy)}[goal_split]
    return free[held if role=='test' else (x!=cx)&(y!=cy)&~held]
maze.goal_pool=goal_pool

def prepare():
    import zarr
    from l2s.eval_openloop import build_episodes
    specs=json.loads((OUT.parent/'maze_expand_20261008/datasets.json').read_text());checks={}
    for name in ['11tr','11bl']:
        layout=ROOT/'data/procgen_mazes/layouts.npz';split='quadrant_'+name[2:];data=OUT/'datasets'/name/'data.zarr'
        if not data.exists():
            data.parent.mkdir(parents=True,exist_ok=True)
            maze.make_dataset(maze.MazeEnv(str(layout),seed=0,split='train',layout_index=0,goal_split=split),str(data),800,0)
        z=zarr.open(str(data),'r');ends=z['meta/episode_ends'][:];lengths=np.diff(np.r_[0,ends]);nav=np.load(layout)['nav'][0]
        specs[name]=dict(layouts=str(layout),dataset=str(data),goal_split=split,shape=list(nav.shape),horizon=64,expert_max=int(lengths.max()),diameter=32,free_cells=int(nav.sum()),dataset_seed=0,split_seed=0,episodes=800)
        assert len(ends)==800 and lengths.max()<=64
    for name in ['11br','11tl','11tr','11bl']:
        spec=specs[name];z=zarr.open(spec['dataset'],'r');pairs=maze.load_train_pairs(spec['dataset']);nav=np.load(spec['layouts'])['nav'][0]
        starts=z['meta/start_xy'][:];goals=z['meta/goal_xy'][:];ps=np.concatenate([starts,goals],axis=1).astype(int)
        uniq,inverse=np.unique(ps,axis=0,return_inverse=True);rng=np.random.default_rng(0);chosen=rng.choice(len(uniq),size=round(.2*len(uniq)),replace=False);mask=np.isin(inverse,chosen)
        trainpairs=set(map(tuple,ps[~mask]));valpairs=set(map(tuple,ps[mask]));assert not trainpairs&valpairs
        allowed=set(map(tuple,goal_pool(nav,spec['goal_split'],'train')));held=set(map(tuple,goal_pool(nav,spec['goal_split'],'test')))
        assert not allowed&held and set(map(tuple,goals))<=allowed
        row=dict(train_episodes=int((~mask).sum()),validation_episodes=int(mask.sum()),train_pairs=len(trainpairs),validation_pairs=len(valpairs),train_goal_cells=len(allowed),test_goal_cells=len(held))
        for split in ['eval','train_heldout']:
            eps=build_episodes(spec['layouts'],split,0,100,spec['goal_split'],0,pairs)
            ep=[dict(start=e['start'].tolist(),goal=e['goal'].tolist()) for e in eps]
            ep_pairs={tuple(e['start']+e['goal']) for e in ep};assert not ep_pairs&set(map(tuple,ps))
            assert {tuple(e['goal']) for e in ep}<=(held if split=='eval' else allowed)
            (OUT/f'{name}_{split}_episodes.json').write_text(json.dumps(ep));row[split+'_unique_pairs']=len(ep_pairs)
        checks[name]=row
    (OUT/'datasets.json').write_text(json.dumps(specs,indent=2));(OUT/'split_checks.json').write_text(json.dumps(checks,indent=2));(OUT/'DATA_READY').write_text('Goal/pair separation verified\n');print(json.dumps(checks,indent=2),flush=True)
def prepare17():
    """Add 17tl/17tr/17bl (trimmed demos from maze_quadrants_vnoise_20261023) next to 17br; checks on all four 17x17 splits."""
    import zarr
    from l2s.eval_openloop import build_episodes
    specs=json.loads((OUT/'datasets.json').read_text());checks=json.loads((OUT/'split_checks.json').read_text())
    vn=json.loads((OUT.parent/'maze_quadrants_vnoise_20261023/datasets.json').read_text())
    for name in ['17tl','17tr','17bl']:
        v=vn[name];z=zarr.open(v['dataset'],'r');ends=z['meta/episode_ends'][:];lengths=np.diff(np.r_[0,ends])
        specs[name]={**specs['17br'],'dataset':v['dataset'],'goal_split':v['goal_split'],'expert_max':int(lengths.max())}
        assert len(ends)==800 and lengths.max()<specs[name]['horizon']
    for name in ['17br','17tl','17tr','17bl']:
        spec=specs[name];z=zarr.open(spec['dataset'],'r');pairs=maze.load_train_pairs(spec['dataset']);nav=np.load(spec['layouts'])['nav'][0]
        for role in ['train','test']:  # the patched pools are the ones l2s/maze.py built these datasets with
            assert np.array_equal(goal_pool(nav,spec['goal_split'],role),_original_goal_pool(nav,spec['goal_split'],role)),(name,role)
        starts=z['meta/start_xy'][:];goals=z['meta/goal_xy'][:];ps=np.concatenate([starts,goals],axis=1).astype(int)
        uniq,inverse=np.unique(ps,axis=0,return_inverse=True);rng=np.random.default_rng(0);chosen=rng.choice(len(uniq),size=round(.2*len(uniq)),replace=False);mask=np.isin(inverse,chosen)
        trainpairs=set(map(tuple,ps[~mask]));valpairs=set(map(tuple,ps[mask]));assert not trainpairs&valpairs
        allowed=set(map(tuple,goal_pool(nav,spec['goal_split'],'train')));held=set(map(tuple,goal_pool(nav,spec['goal_split'],'test')))
        assert not allowed&held and set(map(tuple,goals))<=allowed
        row=dict(train_episodes=int((~mask).sum()),validation_episodes=int(mask.sum()),train_pairs=len(trainpairs),validation_pairs=len(valpairs),train_goal_cells=len(allowed),test_goal_cells=len(held))
        for split in ['eval','train_heldout']:
            eps=build_episodes(spec['layouts'],split,0,100,spec['goal_split'],0,pairs)
            ep=[dict(start=e['start'].tolist(),goal=e['goal'].tolist()) for e in eps]
            ep_pairs={tuple(e['start']+e['goal']) for e in ep};assert not ep_pairs&set(map(tuple,ps))
            assert {tuple(e['goal']) for e in ep}<=(held if split=='eval' else allowed)
            (OUT/f'{name}_{split}_episodes.json').write_text(json.dumps(ep));row[split+'_unique_pairs']=len(ep_pairs)
        checks[name]=row
    (OUT/'datasets.json').write_text(json.dumps(specs,indent=2));(OUT/'split_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps({k:checks[k] for k in ['17br','17tl','17tr','17bl']},indent=2),flush=True)
if __name__=='__main__':prepare17() if sys.argv[1:]==['17'] else prepare()
