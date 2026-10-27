"""Replay rendered candidate logs to audit short-horizon rankings without pose resets.

Requires chunks_exact + initial_state (render_search_videos.py). Every branch replays
all executed actions from reset, preserving velocity and contact history. Coverage
is evaluated until success or the episode time limit, just like MultiStepWrapper.
This is a short-horizon diagnostic, NOT a long-horizon action-value oracle.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from diffusion_policy.env.pusht.pusht_env import PushTEnv, pymunk_to_shapely
from diffusion_policy.env.pusht.feedback_util import compute_feedback_from_pose, t_goal_distance


def coverage(env):
    goal = pymunk_to_shapely(env._get_goal_pose_body(env.goal_pose), env.block.shapes)
    block = pymunk_to_shapely(env.block, env.block.shapes)
    return float(goal.intersection(block).area / goal.area)


def audit_episode(task):
    label, ep, rows, max_steps = task
    history = []
    output = []
    for row in rows:
        chunks = np.asarray(row['chunks_exact'], dtype=np.float64)
        final, peak, distances = [], [], []
        for chunk in chunks:
            env = PushTEnv(legacy=False, reset_to_state=np.asarray(row['initial_state']))
            env.reset()
            for action in history:
                env.step(action)
            if not final:
                actual = env._get_obs()
                # Existing human-readable state fields have 2 / 4 decimal places.
                np.testing.assert_allclose(actual[:2], row['agent_pos'], atol=.006, rtol=0)
                np.testing.assert_allclose(actual[2:4], row['block_pose'][:2], atol=.00006, rtol=0)
                angle_error = (actual[4] - row['block_pose'][2] + np.pi) % (2*np.pi) - np.pi
                assert abs(angle_error) < .00006, (label, ep, row['decision'], angle_error)
                before = coverage(env)
            covs = []
            for action in chunk[:max_steps-len(history)]:
                _, _, done, _ = env.step(action)
                covs.append(coverage(env))
                if done:
                    break
            final.append(covs[-1]); peak.append(max(covs))
            distances.append(float(t_goal_distance(compute_feedback_from_pose(env._get_obs()[None, 2:]))[0]))
            env.close()
        final, peak = np.asarray(final), np.asarray(peak)
        picked = row['executed']
        value = next(iter(row['scores']))
        scores = np.asarray(row['scores'][value])
        tg_pick = int(np.argmax(row['scores']['t_goal']))
        best = int(np.argmax(peak))
        informative = np.ptp(peak) > .001
        pairs = np.triu(np.ones((len(peak), len(peak)), bool), 1)
        pairs &= np.abs(peak[:, None]-peak[None, :]) > .001
        products = (scores[:,None]-scores[None,:]) * (peak[:,None]-peak[None,:])
        pair_accuracy = float(np.mean((products[pairs] > 0) + .5*(products[pairs] == 0))) if pairs.any() else None
        rec = dict(label=label, episode_idx=ep, decision=row['decision'],
                   env_step=row['env_step'], executed=picked, best_coverage=best,
                   coverage_before=before, candidate_peak=peak.tolist(), candidate_final=final.tolist(),
                   true_t_goal=distances, informative=bool(informative),
                   regret=float(peak[best]-peak[picked]),
                   chosen_is_best=bool(peak[best]-peak[picked] <= .001),
                   tgoal_pick=tg_pick, tgoal_regret=float(peak[best]-peak[tg_pick]),
                   selected_advantage_over_random=float(peak[picked]-peak.mean()),
                   slot_last_minus_first=float(peak[-1]-peak[0]),
                   pair_accuracy=pair_accuracy,
                   endpoint_spread_px=float(np.linalg.norm(chunks[:,-1]-chunks[:,-1].mean(0),axis=1).mean()),
                   tgoal_prediction_mae_px=float(np.mean(np.abs(np.asarray(distances) + np.asarray(row['scores']['t_goal'])))),
                   score_spread=float(np.ptp(scores)))
        output.append(rec)
        history.extend(chunks[picked])
    print(f'{label} episode {ep}: {len(output)} decisions replayed', flush=True)
    return output


def annotate(root, rows):
    import cv2
    import imageio.v2 as imageio
    for label, ep in sorted({(r['label'],r['episode_idx']) for r in rows}):
        group = [r for r in rows if (r['label'],r['episode_idx']) == (label,ep)]
        source = root / f'steps_{label}_step40000_n16.jsonl'
        logs = {r['decision']:r for r in map(json.loads, source.read_text().splitlines()) if r['episode_idx']==ep}
        with imageio.get_writer(str(root/f'audit_{label}_ep{ep}.mp4'),fps=2,codec='libx264',quality=8,macro_block_size=1) as writer:
            for r in group:
                frame = cv2.imread(str(root/f'ep{ep}_{label}_n16_dec{r["decision"]:03d}.png'))
                if frame is None: raise FileNotFoundError((label,ep,r['decision']))
                strip = np.full((148,frame.shape[1],3),24,np.uint8)
                lines = [f"Replay coverage: selected #{r['executed']}={r['candidate_peak'][r['executed']]:.3f} | best #{r['best_coverage']}={max(r['candidate_peak']):.3f}",
                         f"Short-horizon regret={r['regret']:.3f} | t_goal pick #{r['tgoal_pick']} regret={r['tgoal_regret']:.3f}",
                         'CYAN = coverage-best path. Bars: candidate peak coverage, fixed 0..1 scale.']
                for i,line in enumerate(lines): cv2.putText(strip,line,(8,20+23*i),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1,cv2.LINE_AA)
                # This audit uses the renderer defaults: 384px scene, 92+13px value
                # strip, 10px padding, then 1px even-height padding for the codec.
                scene_y = frame.shape[0] - 384 - 105 - 10 - 1
                # Some source renders predate the generic value-axis label fix.
                cv2.rectangle(frame,(0,frame.shape[0]-15),(frame.shape[1],frame.shape[0]),(24,24,24),-1)
                cv2.putText(frame,'Upper bars: negative verifier score (relative scale; lower is better)',
                            (5,frame.shape[0]-4),cv2.FONT_HERSHEY_SIMPLEX,.32,(220,220,220),1,cv2.LINE_AA)
                chunk = np.asarray(logs[r['decision']]['chunks_exact'][r['best_coverage']])
                points = np.rint(chunk * (384/512)).astype(np.int32)
                points[:,1] += scene_y
                cv2.polylines(frame,[points],False,(255,255,0),2,cv2.LINE_AA)
                cv2.circle(frame,tuple(points[-1]),5,(255,255,0),2,cv2.LINE_AA)
                bw = frame.shape[1] // len(r['candidate_peak'])
                for k,cov in enumerate(r['candidate_peak']):
                    col = (255,255,0) if k==r['best_coverage'] else (60,200,255) if k==r['executed'] else (140,140,140)
                    cv2.rectangle(strip,(k*bw+3,126-int(cov*48)),((k+1)*bw-3,126),col,-1)
                    cv2.putText(strip,str(k),(k*bw+5,142),cv2.FONT_HERSHEY_SIMPLEX,.3,col,1)
                writer.append_data(cv2.cvtColor(np.vstack([frame,strip]),cv2.COLOR_BGR2RGB))
    sections = []
    for ep in sorted({r['episode_idx'] for r in rows}):
        panels = []
        for label in ('bc', 'st'):
            group = [r for r in rows if r['episode_idx']==ep and r['label']==label]
            if not group:
                continue
            informative = [r for r in group if r['informative']]
            rate = np.mean([r['chosen_is_best'] for r in informative]) if informative else float('nan')
            regret = np.mean([r['regret'] for r in informative]) if informative else float('nan')
            panels.append(f'<div><h3>{label.upper()}</h3><video controls preload="metadata" '
                          f'src="audit_{label}_ep{ep}.mp4"></video><p>{len(informative)} informative decisions; '
                          f'coverage-best selection {rate:.0%}; mean regret {regret:.3f}.</p></div>')
        sections.append(f'<section><h2>Episode {ep}</h2><div class="pair">'+''.join(panels)+'</div></section>')
    (root/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>PushT candidate ranking audit</title>
<style>body{font:16px system-ui;max-width:1700px;margin:30px auto;padding:0 20px;background:#17191d;color:#eee}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}video{width:100%}p{line-height:1.5}section{border-top:1px solid #555;margin-top:30px}
@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>
<h1>Five episodes: BC and search transformer candidate rankings</h1>
<p>Both models: blq137 split, checkpoint 40k, N=16, wp_v3_la_tgn, seed 42, first five test episodes.
Videos advance by decision, not simulation frame. The two policies visit different states; their decision numbers do not imply matched states.</p>
<p>Amber: verifier selection. Cyan: best sampled action by peak coverage over the next executed chunk.
Magenta: final sampled candidate. Coverage bars use a fixed 0–1 scale. All candidates are replayed from the complete executed history.
Informative means candidate coverage spread &gt; 0.001; best-selection tolerance is 0.001.
Short-horizon coverage does not establish long-horizon action quality, especially before contact.</p>
<p><a href="coverage_summary.json">Summary JSON</a> · <a href="coverage_audit.json">All candidate measurements</a></p>'''+''.join(sections))


def main():
    p=argparse.ArgumentParser(); p.add_argument('directory',type=pathlib.Path)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--max-steps',type=int,default=300)
    args=p.parse_args(); tasks=[]
    for path in sorted(args.directory.glob('steps_*_step40000_n16.jsonl')):
        label=path.name.split('_')[1]; rows=list(map(json.loads,path.read_text().splitlines()))
        for ep in sorted({r['episode_idx'] for r in rows}):
            group=sorted((r for r in rows if r['episode_idx']==ep),key=lambda r:r['decision'])
            assert [r['decision'] for r in group] == list(range(len(group))), 'Need unfiltered complete logs'
            tasks.append((label,ep,group,args.max_steps))
    assert tasks, 'No candidate logs'
    # Reuse complete episode replays when the other model finishes rendering later.
    # Both exact inputs and this implementation enter the key; partial logs cannot
    # accidentally validate a cache generated for a different trajectory.
    rows, pending, cache_info = [], [], []
    implementation = pathlib.Path(__file__).read_bytes()
    for task in tasks:
        key = hashlib.sha256(implementation + json.dumps(task,sort_keys=True).encode()).hexdigest()
        cache = args.directory / f'branch_cache_{task[0]}_ep{task[1]}.json'
        stored = json.loads(cache.read_text()) if cache.exists() else {}
        if stored.get('key') == key:
            rows.extend(stored['rows'])
        else:
            pending.append(task); cache_info.append((cache,key))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result, (cache,key) in zip(pool.map(audit_episode,pending),cache_info):
            cache.write_text(json.dumps({'key':key,'rows':result}))
            rows.extend(result)
    (args.directory/'coverage_audit.json').write_text(json.dumps(rows,indent=2))
    summary={}
    for label in sorted({r['label'] for r in rows}):
        allrows=[r for r in rows if r['label']==label]; info=[r for r in allrows if r['informative']]
        summary[label]={'decisions':len(allrows),'informative':len(info),'episodes':len({r['episode_idx'] for r in allrows})}
        for key in ['regret','chosen_is_best','tgoal_regret','selected_advantage_over_random','slot_last_minus_first','pair_accuracy','endpoint_spread_px','tgoal_prediction_mae_px']:
            summary[label][key]=float(np.mean([r[key] for r in info])) if info else None
    (args.directory/'coverage_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
    annotate(args.directory,rows)

if __name__=='__main__': main()
