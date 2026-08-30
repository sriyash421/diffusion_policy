"""Regenerate success_rates_vae_tmrl_aug30.md from on-disk eval output.

THE QUESTION. Does grading the observation by candidate slot buy anything as the search
widens? Slot k's encoded observation is corrupted by a DDPM forward marginal at timestep
t_k, running slot 0 (noisiest) -> slot 15 (clean). n and slot are the same index, so at
n=8 with K=16 only slots 0..7 exist -- the NOISY HALF of the ladder -- and n=16 is the
first width that reaches the clean end. That makes n=1, n=8 and n=16 three different
conditionals under a ladder where under the uniform baseline they are one conditional
sampled more times.

Reads bon_search_*/success_curves.jsonl and nothing else, so it is safe to re-run mid-sweep:
arms that have not reached a checkpoint simply have fewer rows.

    python scripts/build_vae_tmrl_doc.py [-o success_rates_vae_tmrl_aug30.md]

Nominates NO best checkpoint and no best n. Every evaluated cell is printed and the reader
picks; a doc that highlighted a winner would be doing selection on test.
"""
import argparse
import datetime
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_30_100_success_doc import (            # noqa: E402
    ROOT, NS, by_step, provenance, read_rows, table)

VER = 't_goal'
# The imgonly task dir, NOT `pusht_image_search`: `task_name` is a hydra.run.dir component,
# so the image-only era lands under its own path and the ~25 scripts reading the old one
# keep pointing at the pre-change runs they were written for.
BASE = ROOT / 'pusht_search' / 'pusht_image_search_imgonly'
_K16 = BASE / 'outer_inner'
SUF = 'enc-vae_demos-30_seed-42'

# (label, run dir, has_ladder, what it does)
ARMS = [
    ('UNet BC', BASE / 'unet_bc' / f'unetbc_ver-{VER}_{SUF}', False,
     'the diffusion-policy UNet; ranks n i.i.d. draws OUTSIDE the model'),
    ('ST k=1', BASE / 'offline' / f'value_k1_ver-{VER}_{SUF}', False,
     'width 1: no search context'),
    ('ST k=16 — uniform', _K16 / f'value_k16_ver-{VER}_{SUF}', False,
     'uniform slot weights, no obs ladder — the control for every arm below'),
    ('ST k=16 — linear in t, slot0=999', _K16 / f'value_k16_ver-{VER}_son-lint999_{SUF}',
     True, '`slot_obs_noise: linear_t`'),
    ('ST k=16 — linear in t, slot0=400', _K16 / f'value_k16_ver-{VER}_son-lint400_{SUF}',
     True, '`slot_obs_noise: linear_t, max_t 400`'),
    ('ST k=16 — linear in signal', _K16 / f'value_k16_ver-{VER}_son-linsig_{SUF}',
     True, '`slot_obs_noise: linear_signal` — even in sqrt(alpha_bar)'),
    ('ST k=16 — geometric d=0.7', _K16 / f'value_k16_ver-{VER}_son-geo7_{SUF}',
     True, '`slot_obs_noise: geometric, decay 0.7`'),
    ('ST k=16 — random base -> linear in signal',
     _K16 / f'value_k16_ver-{VER}_son-rndlinsig_{SUF}', False,
     '`slot_obs_noise: random_base, shape linear_signal, base_range [0, 999]`'),
]

# obs-clean and obs-corrupt are separate DIRECTORIES because _bon_subdir keys on
# corrupt_obs_eval, so the two conditionals can never merge into one curve.
#
# `has_ladder=False` arms get clean only. For the three uniform baselines the flag is a
# genuine no-op; for random_base the level is redrawn per decision, so "the same noise as
# training" has no single ladder to reproduce and the sweep was never submitted.
RULES = ['argmax', 'final_pass']
OBS = [('clean', 'obs-clean'), ('corrupt', 'obs-corrupt')]


def sub(rule, obs):
    return f'bon_search_sel-{rule}_{obs}'


def ladder_note():
    """The per-slot levels, so the doc does not depend on reading a training log."""
    return """\
All ladders run on TMRL's VLA schedule -- `DDPMScheduler(1000, beta 1e-4 -> 0.02, linear)`,
whose floor is `sqrt(alpha_bar) = 0.0064`, i.e. 0.6 % signal. The legacy T=100 schedule this
replaced bottomed out at 0.589 (59 % signal), so NO ladder shape on it could make slot 0
mostly noise; the ceiling was the schedule, not the shape.

| arm | slot 0 | slot 8 | slot 15 |
|---|---|---|---|
| linear in t, slot0=999 | t=999, 0.006 | t=466, 0.33 | t=0, 1.000 |
| linear in t, slot0=400 | t=400, 0.440 | t=187, 0.83 | t=0, 1.000 |
| linear in signal | t=999, 0.006 | t=348, 0.54 | t=0, 1.000 |
| geometric d=0.7 | t=999, 0.006 | t=58, 0.98 | t=5, 1.000 |
| random base (midpoint draw) | t=499, 0.280 | t=174, 0.85 | t=0, 1.000 |

Cells are `t, sqrt(alpha_bar)`. Decoded panels for each shape are under
`media/obs_latent_*/`, rendered with the measured per-dimension latent sigma
(min 0.026 / median 0.296 / max 0.990), not the constructor's ones.

`linear_signal` is the only shape that grades EVENLY from pure noise to clean.
`geometric d=0.7` is degenerate on a 1000-step schedule -- slots 9-15 all sit between 0.990
and 1.000, so seven of its sixteen slots are effectively the same clean observation. It is
in the matrix as specified; read it as a badly-shaped-ladder control, not a contender.
`linear_t` at 999 is milder but still lopsided, with four low slots near-identical noise."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='success_rates_vae_tmrl_aug30.md')
    args = ap.parse_args()

    L = ['# PushT VAE + TMRL obs-corruption ladder — success rates',
         '',
         f'_Generated {datetime.date.today().isoformat()} by '
         '`scripts/build_vae_tmrl_doc.py`. Re-run to refresh._',
         '',
         '30 demos, seed 42, verifier `t_goal`, 50 test episodes, 100k gradient steps,',
         'checkpoint every 10k. Image-only observations (no `agent_pos`, no `feedback`)',
         'encoded by a FROZEN Stable Diffusion VAE (`sd-vae-ft-mse`, 324-d at the 72x72',
         'crop). Every arm reports 34,163,664 frozen parameters.',
         '',
         '## The ladder', '', ladder_note(), '',
         '## Reading the tables', '',
         '`argmax` sweeps n = 1..64; `final_pass` was asked for at n = 1, 8, 16 only, so its',
         'other columns are blank by design rather than missing. Blank also means "not yet',
         'evaluated" -- the sweep is still running. No cell is a nominated best.', '']

    for label, run, has_ladder, what in ARMS:
        L += [f'## {label}', '', what, '']
        obs_axis = OBS if has_ladder else OBS[:1]
        any_rows = False
        for rule in RULES:
            for obs_label, obs_dir in obs_axis:
                rows = read_rows(run, sub=sub(rule, obs_dir), verifier=VER)
                if not rows:
                    continue
                any_rows = True
                title = f'{rule}, obs {obs_label}'
                L += [f'**{title}** — test success rate', '',
                      table(by_step(rows, 'success_rate')), '']
        if not any_rows:
            L += ['_no checkpoints evaluated yet_', '']
            continue
        # provenance off the arm's primary readout
        rows = read_rows(run, sub=sub('argmax', 'obs-clean'), verifier=VER)
        if rows:
            ck, done, partial, facts = provenance(rows, run)
            L += [f'<sub>checkpoints on disk: {len(ck)} · '
                  f'complete n-sweeps: {len(done)} · partial: {len(partial)} · '
                  f'episodes: {sorted(x for x in facts["n_episodes"] if x)} · '
                  f'seed: {sorted(x for x in facts["seed"] if x is not None)}</sub>', '']

    L += ['## Caveats', '',
          '**BC is not a like-for-like architecture comparison.** The UNet BC arm optimizes',
          '270,370,562 parameters against the ST arms\' 5,896,194 — a 46x gap. It shares the',
          'encoder, crop and image pipeline with the ST arms, so the observation is matched,',
          'but the capacity is not.', '',
          '**`final_pass` is degenerate on the no-ladder arms.** It executes the',
          'last-generated candidate instead of the verifier\'s pick, so without a ladder the',
          'candidates are i.i.d. draws and it reduces to "sample once, ignore the verifier" —',
          'the n=1 argmax number. It only becomes meaningful on a ladder arm, where the last',
          'slot is the CLEANEST observation rather than an arbitrary draw. The baseline',
          '`final_pass` rows are the control those are read against.', '',
          '**Image-only removed the block pose from the OBSERVATION, not from the model\'s',
          'inputs.** Under `search_context: value` slot k>0 still sees k verifier scalars,',
          'each a goal distance computed by the real pymunk simulator from the state that',
          'candidate actually reached. Slot 0 sees only the (corrupted) image. So an ST arm',
          'beating BC is not purely architectural: its inputs include a simulator-derived',
          'signal BC\'s do not.', '',
          '**The ladder is not the textbook DDPM forward process.** It rescales the noise',
          'per dimension by a running std of the encoded features',
          '(`eps = randn_like(x) * obs_feature_std`), because the VAE latent\'s per-dimension',
          'std spans 0.026 to 0.990 and unit noise would obliterate the narrow dimensions',
          'while barely touching the wide ones. That rescale is what makes sqrt(alpha_bar)',
          'an SNR. "We corrupt in the space diffusion models operate in" is a claim about',
          'the SPACE, not the OPERATOR.', '',
          '**The flat `corrupt_obs` arm uses a different operator**, not the same operator on',
          'a different schedule. Do not present the two as a schedule ablation.', '']

    pathlib.Path(args.out).write_text('\n'.join(L) + '\n')
    print(f'wrote {args.out} ({len(L)} lines)')


if __name__ == '__main__':
    main()
