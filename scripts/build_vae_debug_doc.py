"""Regenerate success_rates_vae_debug.md from on-disk eval output.

WHY THIS EXISTS. Under the frozen SD VAE, BC and ST k=1 collapsed at low n against the
ResNet no-pos runs (ST k=1 n=1: 0.06 -> 0.00; BC n=1: 0.10 -> ~0.02) while HIGH n held up.
The in-training rollout agreed, so it was not an eval artifact, and the ResNet arms were
also image-only, so it was not agent_pos. Three encoders x two arms separates the causes:

    ResNet18   -> reproduces the known-good numbers, testing the speedup revert
    VAE frozen -> ResNet vs this isolates the ENCODER
    VAE trainable -> frozen vs this isolates the FREEZE

All six share everything else: 30 demos, seed 42, t_goal, image-only obs, no ladder
(slot_obs_noise uniform leaves slot_obs_t None, so the corruption is identity), 100k
gradient steps, a checkpoint every 10k, the same 50 test episodes.

Reads bon_search_*/success_curves.jsonl and nothing else, so it is safe to re-run mid-sweep.

    python scripts/build_vae_debug_doc.py [-o success_rates_vae_debug.md]

Nominates no best checkpoint and no best n -- every evaluated cell is printed.
"""
import argparse
import datetime
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_30_100_success_doc import (            # noqa: E402
    ROOT, NS, by_step, provenance, read_rows, table)

VER = 't_goal'
BASE = ROOT / 'pusht_search' / 'pusht_image_search_imgonly'
SUF = f'ver-{VER}'

# (section, note, [(arm label, run dir)])
GROUPS = [
    ('ResNet18, trainable (reference)',
     'ResNet18 IMAGENET1K_V1, `use_group_norm=True`, 76x76 crop, trained end to end. The '
     'target: reproduces `success_rates_no_pos.md` and so tests the speedup revert.',
     [('UNet BC', BASE / 'unet_bc' / f'unetbc_{SUF}_enc-resnet18_demos-30_seed-42'),
      ('ST k=1', BASE / 'offline' / f'value_k1_{SUF}_enc-resnet18_demos-30_seed-42')]),
    ('ResNet18, frozen',
     'The same ResNet18, `training.freeze_encoder=True`, so 11.2M encoder parameters are '
     'held out of the optimizer. Against trainable ResNet this asks whether freezing breaks '
     'ANY encoder; against the frozen VAE it is capacity-matched (both leave the policy the '
     'same trainable parameters), so that contrast reads on the FEATURES alone. Caveat: '
     '`use_group_norm=True` replaced the pretrained BatchNorm with freshly built GroupNorm, '
     'so this freezes never-trained normalization at identity init and discards ImageNet\'s '
     'running statistics -- it is a controlled freeze-vs-trainable contrast, not "frozen '
     'ImageNet features" in the literature sense.',
     [('UNet BC', BASE / f'unet_bc/unetbc_{SUF}_enc-resnet18-frozen_demos-30_seed-42'),
      ('ST k=1', BASE / f'offline/value_k1_{SUF}_enc-resnet18-frozen_demos-30_seed-42')]),
    ('SD VAE, trainable',
     'The same encoder, `trainable=True`, trained end to end. Against the frozen column '
     'this isolates the FREEZE. A debug configuration only: a drifting encoder stops '
     'producing SD latents, which is what the corruption ladder and the latent decoder '
     'both rest on.',
     [('UNet BC', BASE / 'unet_bc' / f'unetbc_{SUF}_enc-vae-ft_demos-30_seed-42'),
      ('ST k=1', BASE / 'offline' / f'value_k1_{SUF}_enc-vae-ft_demos-30_seed-42')]),
    ('SD VAE, frozen',
     'Frozen `sd-vae-ft-mse`, 324-d at the 72x72 crop, 34,163,664 parameters held out of '
     'the optimizer. Against ResNet18 this isolates the ENCODER.',
     [('UNet BC', BASE / 'unet_bc' / f'unetbc_{SUF}_enc-vae_demos-30_seed-42'),
      ('ST k=1', BASE / 'offline' / f'value_k1_{SUF}_enc-vae_demos-30_seed-42')]),
]

READOUTS = [('argmax', 'bon_search_sel-argmax_obs-clean'),
            ('final_pass', 'bon_search_sel-final_pass_obs-clean')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='success_rates_vae_debug.md')
    args = ap.parse_args()

    L = ['# PushT encoder debug — success rates', '',
         f'_Generated {datetime.date.today().isoformat()} by '
         '`scripts/build_vae_debug_doc.py`. Re-run to refresh._', '',
         '30 demos, seed 42, verifier `t_goal`, 50 test episodes, image-only observations, '
         '100k gradient steps, checkpoint every 10k. **No observation corruption on any of '
         'these six** — `slot_obs_noise` is uniform, which leaves `slot_obs_t` None so the '
         'corruption is the identity, and `corrupt_obs` is false.', '',
         'A 2x2 of {ResNet18, SD VAE} x {trainable, frozen}, each on two arms. ResNet18 '
         'trainable tests the revert of the 2026-08-30 speedup pass against '
         '`success_rates_no_pos.md`. Reading the square: down a column asks whether freezing '
         'hurts that encoder; across the frozen row asks whether SD features are worse than '
         'ResNet features when neither can adapt, at matched trainable capacity.', '',
         '`argmax` sweeps n = 1..64; `final_pass` was asked for at n = 1, 8, 16 only, so its '
         'other columns are blank by design. Blank also means "not yet evaluated". No cell '
         'is a nominated best.', '']

    for section, note, arms in GROUPS:
        L += [f'## {section}', '', note, '']
        for label, run in arms:
            L += [f'### {label}', '']
            any_rows = False
            for rule, sub in READOUTS:
                rows = read_rows(run, sub=sub, verifier=VER)
                if not rows:
                    continue
                any_rows = True
                L += [f'**{rule}** — test success rate', '',
                      table(by_step(rows, 'success_rate')), '']
            if not any_rows:
                L += ['_no checkpoints evaluated yet_', '']
                continue
            rows = read_rows(run, sub=READOUTS[0][1], verifier=VER)
            ck, done, partial, facts = provenance(rows, run)
            L += [f'<sub>checkpoints on disk: {len(ck)} · complete n-sweeps: {len(done)} · '
                  f'partial: {len(partial)} · '
                  f'episodes: {sorted(x for x in facts["n_episodes"] if x)} · '
                  f'seed: {sorted(x for x in facts["seed"] if x is not None)}</sub>', '']

    L += ['## Caveats', '',
          '**BC is not a like-for-like architecture comparison.** The UNet BC arm optimizes '
          'far more parameters than ST k=1 (270M vs 5.9M on the VAE). It shares the encoder, '
          'crop and image pipeline, so the observation is matched; the capacity is not.', '',
          '**BC\'s crop changed with this generation.** BC now draws one crop offset per '
          'SAMPLE, shared across the observation window, as ST always did. The older ResNet '
          'runs in `success_rates_no_pos.md` cropped each frame independently, so the BC '
          'column here is not expected to match those exactly. ST k=1 is unaffected and is '
          'the clean reproduction target.', '',
          '**`final_pass` is degenerate without a ladder.** It executes the last-generated '
          'candidate instead of the verifier\'s pick, so with i.i.d. candidates it reduces '
          'to "sample once, ignore the verifier" — the n=1 argmax number. All six arms here '
          'are ladder-free, so every `final_pass` table is that control.', '']

    pathlib.Path(args.out).write_text('\n'.join(L) + '\n')
    print(f'wrote {args.out} ({len(L)} lines)')


if __name__ == '__main__':
    main()
