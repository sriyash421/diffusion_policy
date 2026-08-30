"""Regenerate success_rates_no_pos.md from on-disk eval output.

THE ABLATION. Every PushT arm to date conditions on
``shape_meta.obs = {image, agent_pos, feedback}``. ``feedback`` is an exact, invertible
transform of ``block_pos`` (``feedback_util``), so the ground-truth T pose has been handed
to the policy in closed form rather than extracted from pixels -- a stronger observation
than the standard PushT-image setup. The ``nopos`` arms delete both low_dim keys from
``shape_meta.obs`` (``scripts/run_nopos_30demo.sh``), shrinking the obs encoder from 530 to
512, and are otherwise byte-identical to their controls.

The VERIFIER is unchanged and still reads ``agent_pos``/``feedback`` off the obs dict --
it resets a pymunk sim and cannot work otherwise. So this measures what the two keys bought
the POLICY, holding the search's scoring power fixed.

Everything is matched to the controls: 30 demos, seed 42, ``t_goal`` verifier, 4/4/256,
100k gradient steps, a checkpoint every 10k, and the same 50 held-out test episodes swept
over n = 1..64.

    python scripts/build_nopos_doc.py [-o success_rates_no_pos.md]

Safe to re-run mid-sweep: it reads each run's bon_search/success_curves.jsonl and nothing
else, so arms that have not reached a checkpoint render as pending. Like
build_30_100_success_doc.py it deliberately nominates no best checkpoint and no best n --
every evaluated cell is printed and the reader picks.
"""
import argparse
import datetime
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# The row reader, per-step aggregation, table rendering and provenance are identical to the
# 30/100 doc's; importing them keeps the two docs literally the same measurement.
from build_30_100_success_doc import (            # noqa: E402
    BASE, NS, STEPS, by_step, provenance, read_rows, table)

VERIFIER = 't_goal'
SUB = 'bon_search'            # trained AND evaluated under t_goal, so the default subdir

# (label, run dir, what the observation is)
#
# Each nopos arm is listed directly after its control so the pair reads together. The
# control DIRECTORIES carry no `ver-` component: they predate the 2026-08-19 verifier
# rename and were trained under t_goal, which is what the nopos arms use too.
WITHPOS = 'image + agent_pos + feedback (530-d encoder)'
NOPOS = 'image only (512-d encoder)'

PAIRS = [
    ('ST-diffusion k=1 (4/4/256)',
     BASE / 'offline' / 'value_k1_demos-30_seed-42',
     BASE / 'offline' / 'value_k1_ver-t_goal_nopos_corrupt-False_demos-30_seed-42'),
    ('ST-diffusion k=16 (4/4/256)',
     BASE / 'outer_inner' / 'value_k16_corrupt-False_demos-30_seed-42',
     BASE / 'outer_inner' / 'value_k16_ver-t_goal_nopos_corrupt-False_demos-30_seed-42'),
    ('UNet BC (293.4M)',
     BASE / 'unet_bc' / 'unetbc_demos-30_seed-42',
     BASE / 'unet_bc' / 'unetbc_ver-t_goal_nopos_demos-30_seed-42'),
]


def delta_table(agg_ctrl, agg_var):
    """nopos - withpos per (step, n). Blank where either side is missing; never imputed."""
    lines = ['| step | ' + ' | '.join(f'n={n}' for n in NS) + ' |',
             '|---:|' + '---:|' * len(NS)]
    any_row = False
    for step in STEPS:
        if step not in agg_ctrl or step not in agg_var:
            continue
        cells = [f'{agg_var[step][n] - agg_ctrl[step][n]:+.2f}'
                 if n in agg_ctrl[step] and n in agg_var[step] else '–' for n in NS]
        lines.append(f'| {step:,} | ' + ' | '.join(cells) + ' |')
        any_row = True
    return '\n'.join(lines) if any_row else \
        '_not yet comparable — one side has no swept checkpoint_'


def render_arm(A, run, obs_desc):
    """One arm's provenance line and success table. Returns its {step: {n: rate}}."""
    A(f'`{run.relative_to(BASE)}/{SUB}` — {obs_desc}\n')
    rows = read_rows(run, SUB, VERIFIER)
    ck, done, partial, _facts = provenance(rows, run)
    if not ck:
        A('_no checkpoints written yet — training has not reached step 10,000._\n')
        return {}
    A(f'_{len(ck)}/10 checkpoints written, {len(done)} fully swept'
      + (f', {len(partial)} partial' if partial else '') + '._\n')
    agg = by_step(rows, 'success_rate')
    A(table(agg))
    A('')
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='success_rates_no_pos.md')
    args = ap.parse_args()

    L = []
    A = L.append
    A('# PushT image-only ablation — success rates\n')
    A('What did `agent_pos` and `feedback` buy the policy? Each pair below is one arm '
      'trained twice: once on the full observation and once on **pixels alone**, with both '
      'low_dim keys deleted from `shape_meta.obs`. Every arm: 30 demos, seed 42, `t_goal` '
      'verifier, 4/4/256, 100k gradient steps with a checkpoint every 10k, every checkpoint '
      'swept over `n = 1, 2, 4, 8, 16, 32, 64` on the same 50 held-out test episodes.\n')
    A(f'_Generated {datetime.date.today().isoformat()} from the on-disk eval output._\n')
    A('**What changed and what did not.** The ablation is two hydra key deletions, '
      '`~task.shape_meta.obs.agent_pos ~task.shape_meta.obs.feedback`, which shrink '
      '`MultiImageObsEncoder` from 530 to 512 outputs and touch nothing else. The dataset '
      'still emits all three keys and the normalizer still holds params for all three. '
      'Crucially the **verifier is unchanged**: `_verifier_inputs` reads `agent_pos` and '
      '`feedback` straight off the obs dict to reset its sim, so the search scores '
      'candidates exactly as well in both columns. A gap here is the policy losing the '
      'closed-form T pose, not the search losing its ground truth.\n')
    A('`feedback` is an exact, invertible transform of `block_pos`, so the with-pos column '
      'is a privileged-observation setting: it hands the policy the goal-relative block '
      'pose rather than making the ResNet extract it. The image-only column is the one '
      'comparable to published PushT-image baselines.\n')
    A('**Reading these tables.** Success rate is the fraction of the 50 test episodes '
      'reaching the coverage threshold. At 50 episodes a single cell carries a 95% CI of '
      'roughly ±0.13 near 0.5, so **cell-to-cell differences under ~0.15 are not '
      'separable** — read down a column or across several checkpoints, never one cell. '
      'The delta tables are printed in full with no cell singled out; picking the largest '
      'one is selection on test.\n')

    seen = {}
    for label, ctrl, var in PAIRS:
        A(f'## {label}\n')
        A('### With `agent_pos` + `feedback` (control)\n')
        agg_ctrl = render_arm(A, ctrl, WITHPOS)
        A('### Image only\n')
        agg_var = render_arm(A, var, NOPOS)
        seen[label] = (agg_ctrl, agg_var)

    A('## Image-only − with-pos, paired by step and n\n')
    A('Negative = removing the two keys cost success at that checkpoint and that search '
      'width. Blank where either side has no measurement; nothing is imputed.\n')
    for label, _ctrl, _var in PAIRS:
        agg_ctrl, agg_var = seen[label]
        A(f'\n### {label}\n')
        A(delta_table(agg_ctrl, agg_var))
    A('')

    A('## Status\n')
    A('| arm | observation | checkpoints | fully swept | selection | episodes | seed |')
    A('|---|---|---:|---:|---|---:|---:|')
    for label, ctrl, var in PAIRS:
        for run, desc in ((ctrl, WITHPOS), (var, NOPOS)):
            rows = read_rows(run, SUB, VERIFIER)
            ck, done, _p, facts = provenance(rows, run)

            def one(k, facts=facts):
                v = {x for x in facts.get(k, set()) if x is not None}
                return sorted(map(str, v))[0] if len(v) == 1 else \
                    ('—' if not v else '/'.join(sorted(map(str, v))))
            A(f'| {label} | {desc} | {len(ck)}/10 | {len(done)} | {one("selection")} | '
              f'{one("n_episodes")} | {one("seed")} |')
    A('')

    out = pathlib.Path(args.out)
    out.write_text('\n'.join(L) + '\n')
    n_eval = sum(len(c) + len(v) for c, v in seen.values())
    print(f'wrote {out} ({2 * len(PAIRS)} arms, {n_eval} evaluated checkpoints total)')


if __name__ == '__main__':
    main()
