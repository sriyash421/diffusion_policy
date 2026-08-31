"""Regenerate success_rates_noised_obs_resnetE2E.md from on-disk eval output.

Four per-slot OBSERVATION-noise schedules against three uncorrupted baselines, all on the
ResNet18-end-to-end backbone. Slot k conditions on the first k scored candidates, so every
ladder puts the most corrupted observation at slot 0 (no context) and the cleanest at slot
15: a slot that cannot see has to explore, a slot with the full context sharpens.

Each trained arm is read FOUR ways -- {argmax, final_pass} x {corrupted rollouts, clean
rollouts} -- because selection and corrupt_obs_eval are readout rules rather than trained
state, so all four come off the same weights. The baselines have no ladder, so the obs fork
is the identity for them and only the clean readout exists.

Reads bon_search_*/success_curves.jsonl and nothing else, so it is safe to re-run mid-sweep.

    python scripts/build_noised_obs_doc.py [-o success_rates_noised_obs_resnetE2E.md]

Nominates no best checkpoint and no best n -- every evaluated cell is printed.
"""
import argparse
import datetime
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_30_100_success_doc import (            # noqa: E402
    ROOT, NS, by_step, provenance, read_rows, table)

# NOT build_30_100's own BASE: that points at `pusht_image_search`, the with-pos task tree
# these arms predate. config/task/pusht_image_search_imgonly.yaml is the only PushT search
# task now, and `task_name` is a path component of hydra.run.dir, so every run here lands
# under `pusht_image_search_imgonly`. Reading the wrong tree finds nothing and reports every
# arm as unevaluated.
BASE = ROOT / 'pusht_search' / 'pusht_image_search_imgonly'

VER = 't_goal'
SUF = f'ver-{VER}'
ENC = 'enc-resnet18'

# label | trainer subdir | run_name | ladder note | which rollout readouts exist
#
# `none` == no ladder, so corrupt_obs_eval is the identity and only the clean curve is
# written; `both` == the fixed ladders; `clean` == random_base, evaluated clean only.
ARMS = [
    ('UNet BC', 'unet_bc', f'unetbc_{SUF}_{ENC}_demos-30_seed-42',
     'A convolutional diffusion UNet with no transformer and no search context at all -- it '
     'isolates the backbone. No ladder.', 'none'),
    ('ST k=1', 'offline', f'value_k1_{SUF}_{ENC}_demos-30_seed-42',
     'The same transformer as the k=16 arms trained at `max_actions: 1`, so its search '
     'context is always empty. No ladder (a one-slot ladder is not a ladder).', 'none'),
    ('ST k=16, uniform', 'outer_inner', f'value_k16_{SUF}_{ENC}_demos-30_seed-42',
     'The control every ladder is read against: identical in every respect except that '
     '`slot_obs_noise` is uniform, so all 16 slots see the same clean observation.', 'none'),
    ('1. linear in t, cap 999', 'outer_inner',
     f'value_k16_{SUF}_son-lint-cap999_{ENC}_demos-30_seed-42',
     '`t_k = (15-k)/15 * 999`. Even in the TIMESTEP index, which is NOT even in corruption: '
     'because alpha_bar is a cumulative product, slots 0-3 all land within 0.03 of each '
     'other in sqrt(alpha_bar) (0.01/0.01/0.02/0.04) while slots 10-15 are spread over 0.44. '
     'Kept as an arm because that skew is exactly the contrast `linear_signal` exists to '
     'fix.', 'both'),
    ('1. linear in t, cap 400', 'outer_inner',
     f'value_k16_{SUF}_son-lint-cap400_{ENC}_demos-30_seed-42',
     'The same shape compressed into [0, 400], so slot 0 sits at sqrt(alpha_bar) = 0.44 '
     'rather than 0.01 -- a degraded observation instead of very nearly pure noise.',
     'both'),
    ('2. linear in a_bar, cap 999', 'outer_inner',
     f'value_k16_{SUF}_son-linsig-cap999_{ENC}_demos-30_seed-42',
     '`t_k` chosen so sqrt(alpha_bar) -- the factor the observation is actually multiplied '
     'by -- falls in equal ~0.066 steps from 0.01 at slot 0 to 1.00 at slot 15. The only '
     'shape that grades evenly from the marginal to the conditional; all 16 levels are '
     'distinct.', 'both'),
    ('2. linear in a_bar, cap 400', 'outer_inner',
     f'value_k16_{SUF}_son-linsig-cap400_{ENC}_demos-30_seed-42',
     'The same even grading compressed into [0, 400]: 0.44 at slot 0 up to 1.00 at slot 15. '
     'Note the compression is even in TIMESTEP, not in signal, so the retained-signal steps '
     'are no longer equal (0.21 from slot 0 to 1, then ~0.02 near the clean end).', 'both'),
    ('3. geometric in t, cap 999', 'outer_inner',
     f'value_k16_{SUF}_son-geo85-cap999_{ENC}_demos-30_seed-42',
     '`t_k = 999 * 0.85^k`. Decay 0.85 rather than the 0.7 that `slot_weights` uses: at '
     'K=16 decay 0.7 leaves 6 of 15 adjacent slots within 0.005 of each other in '
     'sqrt(alpha_bar), so most of that ladder would be the same observation and "geometric '
     'lost" could not be separated from "geometric collapsed". At 0.85 all 16 levels are '
     'distinct. The trade: slot 15 lands at t=87 (sqrt(alpha_bar) 0.958), so this arm\'s '
     'cleanest slot is not fully clean.', 'both'),
    ('3. geometric in t, cap 400', 'outer_inner',
     f'value_k16_{SUF}_son-geo85-cap400_{ENC}_demos-30_seed-42',
     'The same decay compressed into [0, 400]. 2 of 15 adjacent pairs fall within 0.005 '
     'here, so this arm is mildly collapsed at the clean end where the 999 one is not.',
     'both'),
    ('4. random base, decaying in a_bar', 'outer_inner',
     f'value_k16_{SUF}_son-rndlinsig-cap999_{ENC}_demos-30_seed-42',
     'No fixed ladder. Slot 0\'s timestep is drawn PER SAMPLE from [0, 999] and the '
     'linear_signal curve is rescaled into [0, that draw], so slot 0 sits at the drawn level '
     'and slot 15 stays clean -- what varies between samples is the ladder\'s EXTENT, not '
     'its shape. At a draw of 999 it is exactly arm 2; at 0 every slot is clean. The base is '
     'pinned across a decision, so the 16 slots remain one observation seen at graded '
     'levels.', 'clean'),
]

# key | bon_search subdir suffix | column header
ROLLOUTS = [('obs-corrupt', '_obs-corrupt',
             'rollouts noised, same ladder as training'),
            ('obs-clean', '_obs-clean', 'rollouts clean')]
SELECTIONS = [('argmax', 'bon_search_sel-argmax'),
              ('final_pass', 'bon_search_sel-final_pass')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='success_rates_noised_obs_resnetE2E.md')
    args = ap.parse_args()

    L = ['# PushT per-slot observation noise, ResNet18 end-to-end — success rates', '',
         f'_Generated {datetime.date.today().isoformat()} by '
         '`scripts/build_noised_obs_doc.py`. Re-run to refresh._', '',
         '## Config',
         '',
         '| | |',
         '|---|---|',
         '| encoder | ResNet18, `IMAGENET1K_V1`, **trained end to end** (`rnE2E`) |',
         '| | `use_group_norm: True`, `imagenet_norm: False`, `feature_layernorm: False` |',
         '| crop | 76x76 random crop, one offset per sample |',
         '| observation | image only, `To = 2` |',
         '| action | `horizon 16`, `n_action_steps 8` |',
         '| trunk | search transformer 4 layers / 4 heads / 256 emb |',
         '| data | 30 demos, seed 42 |',
         '| optim | batch 32, lr 1e-4, EMA 0.995, 100k gradient steps, checkpoint every 10k |',
         '| sampler | DDIM, 100 train timesteps, **8 inference steps**, epsilon, eta 0 |',
         '| obs noise scheduler | DDPM, **1000** train timesteps, beta 1e-4..0.02 linear |',
         '| verifier | `t_goal` on every arm |',
         '| eval | 50 held-out test episodes, `--skip-val` |',
         '',
         'The obs-noise scheduler is the one the ladder indexes, and it is a different, '
         '**1000-step** schedule from the 100-step DDIM the actions are sampled under. Its '
         'floor is sqrt(alpha_bar) = 0.006, so at cap 999 slot 0 really is very nearly pure '
         'noise; "cap 400" means slot 0 is held at t=400 instead, i.e. sqrt(alpha_bar) 0.44. '
         'The corruption is applied to the encoded observation features, scaled by a running '
         'per-dimension feature std so sqrt(alpha_bar) reads as an SNR rather than an '
         'absolute magnitude. One noise sample per decision, shared across the 16 slots: the '
         'agent has one observation, seen at 16 graded levels.', '',
         '## Reading the tables', '',
         '`n` is the **search width**: n action candidates are generated for the current '
         'observation, each is rolled out in a PushT simulator (the verifier) to a scalar '
         'value, and one is executed. So n is a test-time-compute axis on fixed weights.', '',
         'At 50 episodes a single cell carries a 95% CI of roughly +/-0.13 near 0.5, so '
         '**cell-to-cell differences under ~0.15 are not separable** -- read down a column '
         'or across checkpoints, not one cell.', '',
         '`argmax` executes the verifier\'s pick and sweeps n = 1..64. `final_pass` instead '
         'executes the LAST candidate generated, which under a ladder is the cleanest slot '
         'the search reached; it was asked for at n = 1, 8, 16 only, so its other columns '
         'are blank by design. Blank also means "not yet evaluated". No cell is a nominated '
         'best.', '',
         '**Corrupted vs clean rollouts.** `corrupt_obs_eval` gates only the eval branch of '
         'the corruption and takes no part in the loss, so both rows come off one '
         'checkpoint. Clean rollouts mean the slot -> corruption-level mapping the model '
         'trained under does not hold at rollout: a legitimate readout, but not the '
         'conditional the loss trained. Arm 4 is evaluated clean only, as asked.', '']

    for label, sub, run_name, note, cls in ARMS:
        run = BASE / sub / run_name
        L += [f'## {label}', '', note, '',
              f'<sub>`{sub}/{run_name}`</sub>', '']
        rollouts = [r for r in ROLLOUTS
                    if cls == 'both'
                    or (cls == 'clean' and r[0] == 'obs-clean')
                    or (cls == 'none' and r[0] == 'obs-clean')]
        any_rows = False
        first_rows = None
        for rkey, rsuf, rdesc in rollouts:
            # A ladder-free arm writes its curve without the obs fork at all, so its clean
            # readout lives in the unsuffixed directory.
            suffix = '' if cls == 'none' else rsuf
            for skey, sdir in SELECTIONS:
                rows = read_rows(run, sub=sdir + suffix, verifier=VER)
                if not rows:
                    continue
                any_rows = True
                first_rows = first_rows or rows
                head = f'**{skey}**' if cls == 'none' else f'**{skey}, {rdesc}**'
                L += [f'{head} — test success rate', '',
                      table(by_step(rows, 'success_rate')), '']
        if not any_rows:
            L += ['_no checkpoints evaluated yet_', '']
            continue
        ck, done, partial, facts = provenance(first_rows, run)
        L += [f'<sub>checkpoints on disk: {len(ck)} · complete n-sweeps: {len(done)} · '
              f'partial: {len(partial)} · '
              f'episodes: {sorted(x for x in facts["n_episodes"] if x)} · '
              f'seed: {sorted(x for x in facts["seed"] if x is not None)}</sub>', '']

    L += ['## Caveats', '',
          '**`final_pass` is degenerate on the three baselines.** It executes the '
          'last-generated candidate instead of the verifier\'s pick, so with i.i.d. '
          'candidates and no ladder it reduces to "sample once, ignore the verifier" -- the '
          'n=1 argmax number. Those three tables are that control. On the ladder arms it is '
          'not degenerate, and that is the point: there the last generation is the one that '
          'saw the cleanest observation.', '',
          '**The deployed slot range moves with n.** At n < 16 only slots 0..n-1 are ever '
          'generated -- the noisy end of the ladder -- and past n = 16 a rolling window pins '
          'every further generation at slot 15. So the n axis and the ladder axis are not '
          'independent, and the low-n columns of a ladder arm are read under systematically '
          'more corruption than the high-n ones.', '',
          '**BC is not a like-for-like architecture comparison.** The UNet BC arm optimizes '
          'far more parameters than the ST arms. It shares the encoder, crop and image '
          'pipeline, so the observation is matched; the capacity is not.', '',
          '**Caps 999 and 400 are not a two-point line.** 999 is the full extent of the '
          'obs schedule and 400 is a compression of the same shape into its lower 40%; the '
          'compression is even in timestep, so it does not preserve even spacing in '
          'retained signal for arm 2.', '']

    pathlib.Path(args.out).write_text('\n'.join(L) + '\n')
    print(f'wrote {args.out} ({len(L)} lines)')


if __name__ == '__main__':
    main()
