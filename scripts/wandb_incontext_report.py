"""Build / refresh the PushT in-context W&B report (l2sml/pusht_reports): the full-trajectory ST policy trained
with its own attempts in context (train_pusht_dt_asteroid arms, 30 gradient steps per batch), then the earlier
recollect-every-iteration run (train_pusht_dt_dagger). Draws the figures (plot_st_candidates, plot_dt_dagger) from
whatever results exist, then publishes; the report keeps its URL (OUT/report_url.txt).

    python scripts/wandb_incontext_report.py
"""
import pathlib
import subprocess
import sys

ENTITY, PROJECT = 'l2sml', 'pusht_reports'
TITLE = 'PushT · full-trajectory ST policy with its own attempts in context (blq50far)'
ROOT = pathlib.Path('/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search')
OUT = ROOT / '_incontext_report'
TAIL = 'ctx-state_fb-step_tf-cov_ver-wp_v3_la_tgn_enc-resnet18_demos-50_split-blq50far_seed-42'
AST = ROOT / 'pusht_image_search_imgonly' / 'dt_asteroid'
ARMS = {'clean, 1 chain/start': AST / f'dt-gpt2_asteroid8x30_s30_b128_cps1_on-none_{TAIL}',
        'clean, 3 chains/start': AST / f'dt-gpt2_asteroid8x30_s30_b128_cps3_on-none_{TAIL}',
        'flat 400, 3 chains/start': AST / f'dt-gpt2_asteroid8x30_s30_b128_cps3_on-flat400_{TAIL}',
        'ramp 600→200, 3 chains/start': AST / f'dt-gpt2_asteroid8x30_s30_b128_cps3_on-ramp600to200_{TAIL}'}
# the same arms with 10k updates per cycle (fresh batch of 64 every update, 500 warmup + cosine)
LONG = {k: AST / v.name.replace('asteroid8x30_s30_b128_', 'asteroid8x10000_s1_b64_cos500_') for k, v in ARMS.items()}
# the same arms with 60 / 100 gradient steps on each batch of 128
STEPS = {n: {k: AST / v.name.replace('_s30_', f'_s{n}_') for k, v in ARMS.items()} for n in (60, 100)}
GROUPS = {'': ARMS, 'long_': LONG, 's60_': STEPS[60], 's100_': STEPS[100]}     # figure-name prefix -> arms
RECOLLECT = (ROOT / 'pusht_image_search_imgonly' / 'dt_dagger'
             / f'dt-gpt2_dagger8x30_rep3_ep10_bs32_ctx-state-mix_fb-step_tf-cov_ver-wp_v3_la_tgn_enc-resnet18'
               '_demos-50_split-blq50far_seed-42')
BC_RUN = (ROOT / 'pusht_image_search_imgonly' / 'unet_bc'
          / 'unetbc-full300_ver-t_goal_enc-resnet18_demos-50_split-blq50far_seed-42')
BC_STEP = 50000
SPLIT_MAPS = ROOT / '_fulltraj_report' / 'split_maps.png'

SETUP = (
    '### Setup\n'
    '- **Input:** the current frame\'s 96×96 RGB image only (no agent or T pose)\n'
    '- **Encoder:** ResNet-18 (ImageNet init, GroupNorm; random 76×76 crop in training, centre crop at eval), trained '
    'end to end → 256-d token, + embeddings of the previous action, feedback, stop flag and step → 4-layer causal GPT-2 '
    '(4 heads) → action\n'
    '- **Context:** the policy\'s own k earlier attempts from the same start (k = 0 … 7), whole trajectories, before the '
    'current attempt in one token stream; per step a 6-d state (agent xy, T xy, sin/cos θ; no image), the previous '
    'action and the feedback\n'
    '- **Feedback:** every step the verifier value −(T-to-goal + Veritas pin cost) at the realized state '
    '(`wp_v3_la_tgn`); first token of the next attempt [verifier value at the final state, max T coverage]')


def figures():
    py, here = sys.executable, pathlib.Path(__file__).resolve().parent
    figs = {}
    for g, arms in GROUPS.items():
        live = [f'--arm={k}={v}' for k, v in arms.items() if v.is_dir()]
        if not live:
            continue
        subprocess.run([py, str(here / 'plot_st_candidates.py'), *live, '--bc-run', str(BC_RUN),
                        '--bc-step', str(BC_STEP), '--out-dir', str(OUT), '--prefix', g], check=True)
        for i, (k, v) in enumerate(arms.items()):
            if v.is_dir():
                subprocess.run([py, str(here / 'plot_st_positions.py'), str(v), '--label', k, '--bc-run',
                                str(BC_RUN), '--bc-step', str(BC_STEP), '--out-prefix', str(OUT / f'{g}arm{i}_')],
                               check=True)
        figs[f'incontext/{g}full_context'] = OUT / f'{g}full_context.png'
        for i in range(len(arms)):
            figs.update({f'incontext/{g}{kd}_{i}': OUT / f'{g}{kd}_{i}.png' for kd in ('perf', 'bon', 'curves')})
            figs[f'incontext/{g}startmap_{i}'] = OUT / f'{g}arm{i}_startmap.png'
            figs.update({f'incontext/{g}fail_{i}_stk{k}': OUT / f'{g}arm{i}_fail_stk{k}.png' for k in (2, 7)})
    if (RECOLLECT / 'rounds_eval' / 'train_own.npz').is_file():
        subprocess.run([py, str(here / 'plot_dt_dagger.py'), str(RECOLLECT), '--bc-run', str(BC_RUN),
                        '--bc-step', str(BC_STEP)], check=True)
    ev = RECOLLECT / 'rounds_eval'
    figs.update({'incontext/split_maps': SPLIT_MAPS, 'recollect/curves': ev / 'dagger_curves.png',
                 'recollect/compare': ev / 'dagger_compare.png', 'recollect/per_start': ev / 'per_start_coverage.png'})
    return {k: p for k, p in figs.items() if p.is_file()}


def main():
    figs = figures()
    stages = {(g, k): len(list(v.glob('stages/pi_k*.ckpt'))) for g, a in GROUPS.items() for k, v in a.items()}
    evald = {(g, k): len(list(v.glob('bon_search/stage_eval/pi_k*/test_candidates.npz')))
             for g, a in GROUPS.items() for k, v in a.items()}
    for g in GROUPS:
        if not any(n for (gg, _), n in evald.items() if gg == g):
            figs.pop(f'incontext/{g}full_context', None)

    import wandb
    import wandb_workspaces.reports.v2 as wr
    rid_file = OUT / 'run_id.txt'
    rid = rid_file.read_text().strip() if rid_file.exists() else wandb.util.generate_id()
    rid_file.write_text(rid)
    run = wandb.init(entity=ENTITY, project=PROJECT, id=rid, resume='allow', dir=str(OUT),
                     name='pusht_incontext_report_figures', settings=wandb.Settings(start_method='fork'))
    run.log({k: wandb.Image(str(p)) for k, p in figs.items()})
    run.finish()

    md = wr.MarkdownBlock

    def grid(key, h, pending='eval queued after training ends; added when it finishes'):
        if key not in figs:
            return md(f'*{pending}.*')
        rs = wr.Runset(entity=ENTITY, project=PROJECT, name='figures')
        return wr.PanelGrid(runsets=[rs], panels=[wr.MediaBrowser(media_keys=[key], num_columns=1,
                                                                  layout=wr.Layout(x=0, y=0, w=24, h=h))])

    def status(g, i):
        k = list(GROUPS[g])[i]
        if not GROUPS[g][k].is_dir():
            return '- status: queued'
        return f'- status: {stages[g, k]}/8 cycles trained, {evald[g, k]}/8 STk evaluated with 8 candidates'

    def arm(i, curves_note, g=''):
        f = f'incontext/{g}'
        return [md(curves_note + '\n' + status(g, i)), grid(f'{f}curves_{i}', 11, 'training queued'),
                md('- perf@n: coverage of candidate n alone, no selection; STk: n attempts in a row, candidate n with '
                   'n − 1 earlier attempts in context; BC: its n-th i.i.d. sample; 50 train / 50 test starts × 1 seed'),
                grid(f'{f}perf_{i}', 11),
                md('- best of n: the verifier value picks one of the first n candidates, that candidate is executed'),
                grid(f'{f}bon_{i}', 11),
                md('- max coverage per start at the T start position; STk on candidate k + 1, BC on candidate 1; '
                   '1 seed; dashed box = held-out quadrant'),
                grid(f'{f}startmap_{i}', 16),
                md('- the test start STk2 and STk7 handle worst (lowest summed coverage of their scored candidate)\n'
                   '- every attempt in a row from the start, attempt j with j − 1 earlier attempts in context; replayed '
                   'from the saved actions: agent path, T at the start, T at its best-coverage step, goal T'),
                grid(f'{f}fail_{i}_stk2', 11), grid(f'{f}fail_{i}_stk7', 11)]

    blocks = [md(SETUP),
              md('### 1. Train and test positions\n- 50 test starts held out (blq50far)'),
              grid('incontext/split_maps', 9, 'split map missing'),
              md('### 2. 30 gradient steps over each batch\n'
                 '- collect-and-train cycle k = 0 … 7: the current weights make k attempts in a row from each training '
                 'start → (those k attempts as context, demo) are added to the dataset → train → STk\n'
                 '- each cycle: fresh AdamW, constant lr 1e-4; 30 batches of 128 × 30 gradient steps each\n'
                 '- 1 chain per start; x = gradient steps within cycle k; eval every 30 steps; 50 train / 50 test starts')]
    blocks += arm(0, '- clean observations, 1 chain per start')
    blocks += [md('- STk at n = k + 1 only (its trained context length), every arm of sections 2–4; BC at the same n'),
               grid('incontext/full_context', 12)]
    blocks += [md('### 3. 3 chains per start\n- as section 2, but each collection makes 3 chains of k attempts per '
                  'training start (150 items per cycle instead of 50)')]
    blocks += arm(1, '- clean observations, 3 chains per start')
    blocks += [md('### 4. Training observation noise\n'
                  '- predicted attempt\'s encoded observation (ResNet features): x → √ᾱ_t·x + √(1−ᾱ_t)·ε·std '
                  '(DDPM, T = 1000, linear β; std = running feature std)\n'
                  '- context, rollouts and evals clean; 3 chains per start')]
    blocks += arm(2, '- flat t = 400')
    blocks += arm(3, '- ramp t = 600 (no context) → 200 (7 attempts in context)')
    blocks += [md('### 5. 10k updates per collect-and-train cycle\n'
                  '- same cycles, policy, context, feedback and arms as sections 2–4; only the training in each cycle '
                  'differs\n\n'
                  '| | 30 steps per batch (sections 2–4) | 10k updates per cycle |\n|---|---|---|\n'
                  '| gradient updates per cycle | 900 (30 batches × 30 steps) | 10,000 |\n'
                  '| batch | 128 items, each batch reused for 30 steps | a fresh random batch of 64 items every update |\n'
                  '| learning rate | constant 1e-4 | 500-update warmup, then cosine decay from 1e-4 |\n'
                  '| total updates by STk7 | 7,200 | 80,000 |\n\n'
                  '- x = updates within cycle k; eval every 500 updates; 50 train / 50 test starts'),
               md('- STk at n = k + 1 only, every 10k arm; BC at the same n'),
               grid('incontext/long_full_context', 12)]
    def arms(sec, g):
        out = []
        for i, (note, title) in enumerate([('- clean observations, 1 chain per start', 'a. clean, 1 chain per start'),
                                           ('- clean observations, 3 chains per start', 'b. clean, 3 chains per start'),
                                           ('- training noise flat t = 400, 3 chains per start', 'c. flat t = 400'),
                                           ('- training noise ramp t = 600 → 200, 3 chains per start',
                                            'd. ramp t = 600 → 200')]):
            out += [md(f'#### {sec}{title}')] + arm(i, note, g)
        return out

    blocks += arms(5, 'long_')
    for sec, n in ((6, 60), (7, 100)):
        blocks += [md(f'### {sec}. {n} gradient steps over each batch\n'
                      f'- as sections 2–4 (30 batches of 128 per cycle, constant lr 1e-4), but {n} gradient steps on '
                      f'each batch instead of 30: {30 * n:,} updates per cycle, {7 * 30 * n:,} by STk7\n'
                      f'- x = gradient steps within cycle k; eval every {n} steps; 50 train / 50 test starts'),
                   md(f'- STk at n = k + 1 only, every {n}-step arm; BC at the same n'),
                   grid(f'incontext/s{n}_full_context', 12)]
        blocks += arms(sec, f's{n}_')
    blocks += [md('### 8. Earlier run: re-collected context every iteration\n'
                  '- weights carried over 8 cycles; every one of 30 iterations per cycle the current weights roll out k '
                  'attempts 3 times from each training start, and only those chains are trained on (10 epochs, batch '
                  '32, context width drawn from 0 … k; AdamW, 3% warmup then cosine per cycle)\n'
                  '- scored: STk\'s attempt k + 1 after its own k attempts'),
               md('- collection over training (train starts; attempt k, so k − 1 own attempts in context)'),
               grid('recollect/curves', 12),
               md('- end of each cycle vs own rollouts collected'), grid('recollect/compare', 11),
               md('- per start (rows: replay / own context)'), grid('recollect/per_start', 16)]

    rep_file = OUT / 'report_url.txt'
    report = None
    if rep_file.exists():
        try:
            report = wr.Report.from_url(rep_file.read_text().strip())
        except Exception as e:   # noqa: BLE001 -- a deleted report just means "make a new one"
            print('could not load existing report:', e)
    if report is None:
        report = wr.Report(entity=ENTITY, project=PROJECT, title=TITLE, blocks=blocks)
    else:
        report.title = TITLE
        report.blocks = blocks
    report.description = 'Stage policies STk0..STk7 with their own attempts in context vs the U-Net BC, best-of-N.'
    report.save()
    rep_file.write_text(report.url)
    print('report:', report.url.replace('https://wandb.ai/', 'https://forge.coreweave.com/wandb/'))


if __name__ == '__main__':
    main()
