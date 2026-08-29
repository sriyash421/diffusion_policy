"""Does a candidate improve with the context of the candidates before it?

Slot k is decoded conditioned on the first k already-generated-and-scored candidates
(search_procedure.py, staircase mask in diffusion_transformer_search_policy.py). The whole
premise of the architecture is that this makes later candidates better. This script tests
it on the dumps written by scripts/dump_candidate_scores.py, answering four questions:

  Q1  per prediction, what did each slot score -- with slots 0, 7 and 15 called out, and
      which slot took the maximum                        -> per_prediction_<tier>.csv
  Q2  which slot most often wins
  Q3  is slot 15 > slot 7 > slot 0, on average
  Q4  the delta between consecutive slots, and where it stops improving

FOUR THINGS THAT MAKE THIS EASY TO GET WRONG, all handled below:

1. THE VALUE IS TWO DISTANCES. armTn ranks on -(d_t_goal/13.6 + d_arm_t/52.1). A slot can
   out-score another purely by parking the arm nearer the T while making no task progress
   -- the failure mode that retired the raw `armT` value. Every Q3 statistic is therefore
   also computed on the two raw-pixel terms separately. A win that lives entirely in
   d_arm_t is not the result the headline claims.

2. TIES. Under `t_goal` roughly a fifth of control steps are fully degenerate -- the agent
   is nowhere near the block, so every candidate returns an identical value and np.argmax
   returns 0 on all of them, which makes a raw argmax histogram "discover" that slot 0 wins.
   Q2 is therefore scoped to discriminating steps and read under the `unique` convention,
   against a permutation null.
   MEASURED UNDER armTn THAT GUARD DOES NOT BIND: the arm-to-T term always varies across
   candidates, so 0.0% of steps are degenerate on the composite value in all 16 dumps of
   2026-08-27, and `unique` equals `first` everywhere. The scoping is kept because it costs
   nothing and a t_goal dump or a future value can reintroduce the ties -- but the
   per-TERM blind rates are the ones that matter here: d_t_goal alone is flat across
   candidates on 2.6-17.2% of steps while d_arm_t never is, so on those steps the ranking
   is decided by arm reach alone. That is what point 1 is about.

3. STEP DIFFICULTY. Raw per-slot means are dominated by which states happen to be easy,
   not by slot order. Every level statistic is centered within its own control step first.

4. DEPENDENCE. Control steps inside one episode are strongly autocorrelated, so an interval
   that divides by the number of steps is anticonservative. Every interval here is a
   CLUSTER BOOTSTRAP over the 50 episodes, resampling whole episodes.

And one thing that is NOT evidence: the running max over slots rises with k even under a
pure i.i.d. resampler. It is plotted only against prefix_mean -- the gap between the two is
the value of having a selector -- and is never read as a context effect. The arms with
slot_semantics `iid_no_context` (ST k=1) and `iid_unbounded` (UNet BC) are the empirical
null for exactly this reason: they are not baselines OF the statistic, they ARE its null
distribution, measured on real data with the real tie structure.

    python scripts/slot_context_analysis.py \
        --dump-dir analysis/aug27_slot_wts/lin100-l2_step0010000 ... \
        --tier A --out slot_context_aug27.md
"""
import csv
import json
import pathlib
import sys

import click
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
# The tie-aware argmax-slot statistics and the slot-semantics labelling already exist and
# are already correct; importing them is what keeps this script and the dump from drifting
# to two different definitions of "degenerate". They live in common/slot_stats_util rather
# than in dump_candidate_scores precisely so this import stays numpy-only -- reaching them
# through the dump script would pull in torch and pygame for a script that reads an .npz.
from diffusion_policy.common.slot_stats_util import (      # noqa: E402
    BLIND_EPS, TIE_TAU, _argmax_slot_stats, _slot_semantics,
)

SLOTS = (0, 7, 15)
N_BOOT = 4000
# Raw-pixel term axis order, as written by rollout_candidate_scores.
T_GOAL, ARM_T = 0, 1
TERM_LABEL = {T_GOAL: 'd_t_goal', ARM_T: 'd_arm_t'}


# ---------------------------------------------------------------- loading

def load_dump(d):
    """One dump directory -> arrays plus meta, with the live steps flattened per episode.

    `per_ep` is a list of (n_live_steps_e, n) arrays, one per episode, NOT one flat stack:
    the cluster bootstrap resamples episodes, so the episode boundary has to survive.
    """
    d = pathlib.Path(d)
    z = np.load(d / 'candidate_scores.npz')
    meta = json.loads((d / 'candidate_scores_meta.json').read_text())
    alive = z['alive'].astype(bool)                       # (T, B)
    scores = z['scores'].astype(np.float64)               # (T, B, n)
    terms = z['terms'].astype(np.float64) if 'terms' in z.files else None
    chosen = z['chosen']                                  # (T, B)
    idxs = z['episode_idxs']
    B = scores.shape[1]
    per_ep = [scores[alive[:, b], b, :] for b in range(B)]
    per_ep_terms = ([terms[alive[:, b], b, :, :] for b in range(B)]
                    if terms is not None else None)
    return dict(dir=d, meta=meta, scores=scores, terms=terms, alive=alive, chosen=chosen,
                episode_idxs=idxs, per_ep=per_ep, per_ep_terms=per_ep_terms,
                label=meta.get('arm', d.name), step=int(meta.get('step', 0)),
                n=int(meta.get('n', scores.shape[-1])),
                slot_semantics=meta.get('slot_semantics', 'unknown'),
                verifier=meta.get('verifier_value', '?'))


# ---------------------------------------------------------------- cluster bootstrap

class ClusterBoot:
    """Cluster bootstrap over episodes for statistics that are means over control steps.

    Every quantity here is `mean over live steps of f(step)` for some per-step vector f.
    Under a bootstrap that resamples EPISODES, such a mean is
    sum(selected episodes' sums) / sum(selected episodes' counts) -- so the per-episode
    sums and counts are precomputed once and each of the 4000 draws is a single matrix
    product against a multinomial weight vector. That is exact, not an approximation of the
    resample, and it is what makes 4000 draws over a dozen statistics cheap enough to run
    for every arm.

    Episodes with zero live steps (should not happen, but a truncated dump could) carry a
    count of 0 and drop out of both sums, rather than contributing a 0/0.
    """

    def __init__(self, sums, counts, n_boot=N_BOOT, seed=0):
        self.sums = np.asarray(sums, dtype=np.float64)        # (E, D)
        self.counts = np.asarray(counts, dtype=np.float64)    # (E,)
        E = len(self.counts)
        rng = np.random.default_rng(seed)
        # multinomial draw counts == resampling E episodes with replacement
        self.W = rng.multinomial(E, np.full(E, 1.0 / E), size=n_boot).astype(np.float64)

    @property
    def point(self):
        tot = self.counts.sum()
        return self.sums.sum(axis=0) / tot if tot else np.full(self.sums.shape[1], np.nan)

    def draws(self):
        num = self.W @ self.sums                              # (n_boot, D)
        den = (self.W @ self.counts)[:, None]                 # (n_boot, 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(den > 0, num / den, np.nan)

    def ci(self, q=(2.5, 97.5)):
        return np.nanpercentile(self.draws(), q, axis=0)      # (2, D)


def _boot(per_ep_vals, n_boot=N_BOOT, seed=0):
    """per-episode list of (steps_e, D) -> ClusterBoot over the mean of those rows."""
    sums = np.stack([v.sum(axis=0) if len(v) else np.zeros(v.shape[1:])
                     for v in per_ep_vals])
    counts = np.array([len(v) for v in per_ep_vals], dtype=np.float64)
    return ClusterBoot(sums, counts, n_boot=n_boot, seed=seed)


def _sig(point, lo, hi):
    """'+' / '-' / '.' -- whether an interval excludes zero, and on which side."""
    if np.isnan(lo) or np.isnan(hi):
        return '?'
    return '+' if lo > 0 else ('-' if hi < 0 else '.')


# ---------------------------------------------------------------- Q1

def per_prediction_rows(dump):
    """One row per (arm, episode, live control step). Q1, straight from the npz."""
    scores, alive, chosen = dump['scores'], dump['alive'], dump['chosen']
    terms, idxs = dump['terms'], dump['episode_idxs']
    T, B, n = scores.shape
    Ta = int(dump['meta'].get('n_action_steps', 1))
    rows = []
    for b in range(B):
        for t in range(T):
            if not alive[t, b]:
                continue
            v = scores[t, b]
            mx = float(v.max())
            tie = v >= (mx - TIE_TAU)
            ntie = int(tie.sum())
            spread = mx - float(v.min())
            row = {
                'arm': dump['label'], 'step': dump['step'],
                'episode_idx': int(idxs[b]), 't': int(t), 'env_step': int(t * Ta),
                'blind': int(spread <= BLIND_EPS),
                'n_tied': ntie, 'spread': round(spread, 6),
                # argmax_slot_unique is blank when the maximum is tied: on those steps no
                # slot genuinely won, and filling in numpy's first-maximizer would be
                # recording the tiebreak rule as if it were a result.
                'argmax_slot_unique': int(v.argmax()) if ntie == 1 else '',
                'argmax_slot_first': int(v.argmax()),
                'argmax_slot_last': int(n - 1 - tie[::-1].argmax()),
                'executed_slot': int(chosen[t, b]),
            }
            for s in SLOTS:
                row[f'v{s}'] = round(float(v[s]), 6) if s < n else ''
            if terms is not None:
                for s in SLOTS:
                    if s < n:
                        row[f'dTgoal_{s}'] = round(float(terms[t, b, s, T_GOAL]), 4)
                        row[f'dArmT_{s}'] = round(float(terms[t, b, s, ARM_T]), 4)
                tg, at = terms[t, b, :, T_GOAL], terms[t, b, :, ARM_T]
                row['spread_dTgoal'] = round(float(tg.max() - tg.min()), 6)
                row['spread_dArmT'] = round(float(at.max() - at.min()), 6)
            rows.append(row)
    return rows


# ---------------------------------------------------------------- Q2/Q3/Q4

def slot_stats(dump, n_boot=N_BOOT):
    """Q2, Q3 and Q4 for one arm."""
    n = dump['n']
    per_ep = dump['per_ep']
    live_ep = [e[(e.max(axis=1) - e.min(axis=1)) > BLIND_EPS] for e in per_ep]
    flat = np.concatenate([e for e in per_ep if len(e)]) if any(len(e) for e in per_ep) \
        else np.zeros((0, n))
    flat_live = np.concatenate([e for e in live_ep if len(e)]) \
        if any(len(e) for e in live_ep) else np.zeros((0, n))

    out = {
        'n_live_steps': int(sum(len(e) for e in per_ep)),
        'n_discriminating': int(len(flat_live)),
        'frac_blind': float(1 - len(flat_live) / max(sum(len(e) for e in per_ep), 1)),
    }

    # ---- Q2: which slot wins. Reuses the tie-aware statistic and its permutation null.
    rng = np.random.default_rng(0)
    out['argmax'] = _argmax_slot_stats(flat_live, rng, n_perm=200) if len(flat_live) else {}
    if len(flat_live):
        # tie-scoped histogram: only steps with a single maximizer contribute
        mx = flat_live.max(axis=1, keepdims=True)
        uniq = (flat_live >= (mx - TIE_TAU)).sum(axis=1) == 1
        hist = (np.bincount(flat_live[uniq].argmax(axis=1), minlength=n) / max(uniq.sum(), 1)
                if uniq.any() else np.zeros(n))
        out['argmax_hist_unique'] = hist.tolist()
        out['n_unique_steps'] = int(uniq.sum())

    # ---- Q4: the slot profile. Centered within each control step first.
    cen_ep = [e - e.mean(axis=1, keepdims=True) for e in per_ep]
    cb = _boot(cen_ep, n_boot=n_boot, seed=1)
    prof, (plo, phi) = cb.point, cb.ci()
    out['centered_mean_by_index'] = prof.tolist()
    out['centered_ci_by_index'] = [plo.tolist(), phi.tolist()]

    # within-step spread of each slot: E[x^2] - E[x]^2 on the centered values. If later
    # slots have a SMALLER spread the context is narrowing exploration rather than
    # improving quality -- a different finding with the same sign on the mean.
    sq = _boot([e ** 2 for e in cen_ep], n_boot=n_boot, seed=2).point
    out['centered_sd_by_index'] = np.sqrt(np.maximum(sq - prof ** 2, 0)).tolist()

    # consecutive deltas, each with its own interval
    if n >= 2:
        dcb = _boot([np.diff(e, axis=1) for e in cen_ep], n_boot=n_boot, seed=3)
        d, (dlo, dhi) = dcb.point, dcb.ci()
        out['delta_by_index'] = d.tolist()
        out['delta_ci'] = [dlo.tolist(), dhi.tolist()]

    # SATURATION: the largest k for which the remaining gain (slot n-1 minus slot k) is
    # still significantly > 0. Reported as an interval-backed changepoint, not a peak of a
    # noisy curve -- the profile wiggles and its argmax is not stable.
    tail = _boot([e[:, [n - 1]] - e for e in cen_ep], n_boot=n_boot, seed=4)
    t_pt, (tlo, thi) = tail.point, tail.ci()
    out['tail_gain'] = t_pt.tolist()
    out['tail_gain_ci'] = [tlo.tolist(), thi.tolist()]
    sat = [k for k in range(n - 1) if tlo[k] > 0]
    out['saturation_slot'] = int(max(sat) + 1) if sat else None

    # prefix mean vs prefix max, for the guard note only
    if len(flat):
        csum = np.cumsum(flat, axis=1)
        out['prefix_mean'] = (csum / np.arange(1, n + 1)).mean(axis=0).tolist()
        out['prefix_max'] = np.maximum.accumulate(flat, axis=1).mean(axis=0).tolist()

    # ---- Q3: paired slot comparisons, on the value and on each raw term.
    out['pairs'] = _pair_block(per_ep, live_ep, n, n_boot, higher_is_better=True)
    if dump['per_ep_terms'] is not None:
        out['terms'] = {}
        for ax in (T_GOAL, ARM_T):
            tp = [e[:, :, ax] for e in dump['per_ep_terms']]
            tl = [e[(e.max(axis=1) - e.min(axis=1)) > BLIND_EPS] for e in tp]
            # DISTANCES: lower is better, so the sign is flipped before every comparison.
            out['terms'][TERM_LABEL[ax]] = _pair_block(
                tp, tl, n, n_boot, higher_is_better=False)
            out['terms'][TERM_LABEL[ax]]['frac_blind'] = float(
                1 - sum(len(e) for e in tl) / max(sum(len(e) for e in tp), 1))
    return out


def _pair_block(per_ep, live_ep, n, n_boot, higher_is_better):
    """Paired slot-vs-slot comparison, bootstrapped over episodes.

    `higher_is_better` flips the sign for the raw distance terms, so a positive `delta` and
    a `win_rate` above 0.5 always mean "the later slot is BETTER" whichever quantity is
    being read. Getting this backwards would report the arm-reach term inverted.
    """
    sgn = 1.0 if higher_is_better else -1.0
    pairs = [(lo, hi) for lo, hi in ((0, 7), (7, 15), (0, 15)) if lo < n and hi < n]
    res = {}
    for lo, hi in pairs:
        # deltas over ALL live steps; win rates over discriminating steps only, where the
        # comparison is not decided by a tie
        dcb = _boot([sgn * (e[:, [hi]] - e[:, [lo]]) for e in per_ep],
                    n_boot=n_boot, seed=10 + hi)
        d, (dlo, dhi) = float(dcb.point[0]), [float(x[0]) for x in dcb.ci()]
        wcb = _boot([(sgn * (e[:, [hi]] - e[:, [lo]]) > 0).astype(float) for e in live_ep],
                    n_boot=n_boot, seed=20 + hi)
        w, (wlo, whi) = float(wcb.point[0]), [float(x[0]) for x in wcb.ci()]
        ties = np.concatenate([e[:, hi] == e[:, lo] for e in per_ep if len(e)]) \
            if any(len(e) for e in per_ep) else np.zeros(0, bool)
        res[f'{lo}->{hi}'] = {
            'delta': d, 'delta_ci': [dlo, dhi], 'delta_sig': _sig(d, dlo, dhi),
            'win_rate': w, 'win_ci': [wlo, whi],
            'win_sig': _sig(w - 0.5, wlo - 0.5, whi - 0.5),
            'tie_frac': float(ties.mean()) if ties.size else float('nan'),
        }
    return res


# ---------------------------------------------------------------- plots

def plot_profiles(dumps, stats, out_png, title, key='centered_mean_by_index',
                  ci_key='centered_ci_by_index', ylabel='centered verifier value'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for d in dumps:
        s = stats[d['dir'].name]
        y = np.array(s[key], dtype=float)
        x = np.arange(len(y))
        # the i.i.d. arms are the null, so they are drawn as such and never as a result
        iid = d['slot_semantics'] != 'trained_staircase'
        ln, = ax.plot(x, y, marker='o', ms=3, ls='--' if iid else '-',
                      lw=2.2 if iid else 1.5, alpha=0.95 if iid else 0.85,
                      color='0.35' if iid else None,
                      label=f"{d['label']} @{d['step']//1000}k" + (' [i.i.d. null]' if iid else ''))
        if ci_key in s:
            lo, hi = np.array(s[ci_key], dtype=float)
            ax.fill_between(x, lo, hi, alpha=0.12, color=ln.get_color())
    ax.axhline(0, color='k', lw=0.8, alpha=0.5)
    for s in SLOTS:
        ax.axvline(s, color='0.7', lw=0.8, ls=':', zorder=0)
    ax.set_xlabel('candidate slot k  (k = number of scored candidates it was conditioned on)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report

def _fmt_ci(v, ci, w=6, p=3):
    return f'{v:+.{p}f} [{ci[0]:+.{p}f}, {ci[1]:+.{p}f}]'


def render_md(dumps, stats, tier, out_dir, n_boot=N_BOOT):
    L = []
    A = L.append
    A(f'# Slot-context analysis — tier {tier}')
    A('')
    n_ep = len(dumps[0]['episode_idxs'])
    n_cand = dumps[0]['n']
    A(f'Does candidate k improve with the context of candidates 0..k-1? Measured at '
      f'n={n_cand} over the {n_ep} test episodes, with the verifier value each dump was '
      f'scored under.')
    A('')
    tierdesc = {
        'A': 'All arms at the **same** step (10000), so a cross-arm comparison of slot '
             'effects is not confounded with training duration. Chosen by no criterion.',
        'B': "Each arm at **its own** peak `final_pass` success at n=16 (later checkpoint "
             "breaking ties), as of 2026-08-27. Steps DIFFER PER ARM, so this tier is a "
             "within-arm check only — any cross-arm reading here is confounded with "
             "training duration. It is also mildly circular: `final_pass` at n=16 deploys "
             "slot 15, so the checkpoint was selected on a metric downstream of the "
             "quantity being measured. Tier A is the un-selected comparison.",
    }[tier]
    A(tierdesc)
    A('')
    A('## Arms')
    A('')
    A('| arm | step | slot semantics | verifier | live steps | blind | success |')
    A('|---|---:|---|---|---:|---:|---:|')
    for d in dumps:
        s = stats[d['dir'].name]
        A(f"| {d['label']} | {d['step']} | `{d['slot_semantics']}` | {d['verifier']} | "
          f"{s['n_live_steps']} | {s['frac_blind']:.1%} | "
          f"{d['meta'].get('success_rate', float('nan')):.2f} |")
    A('')
    A('`iid_no_context` (ST k=1) and `iid_unbounded` (UNet BC) generate their 16 candidates '
      'independently **by construction**. They are not baselines of the statistic — they '
      'are its null distribution, measured on real data with the real tie structure. If a '
      'k=16 arm does not beat them, the context bought nothing.')
    A('')

    # ---- Q2
    A('## Q2 — which slot most often gets the highest value')
    A('')
    A('Scoped to **discriminating** steps (some candidate differs) and read under the '
      '`unique` convention (a single maximizer), against a **permutation null** that '
      "reshuffles each step's own value multiset and so holds its ties fixed. The `first` "
      'column is what an argmax policy actually executed, which is a fact about the '
      'tiebreak rule as much as about the policy: on a fully degenerate step every slot is '
      'a maximizer and `np.argmax` returns 0.')
    A('')
    A('| arm | mean slot (unique) | perm null | first | tied steps | unique steps |')
    A('|---|---:|---:|---:|---:|---:|')
    for d in dumps:
        a = stats[d['dir'].name].get('argmax', {})
        if not a:
            A(f"| {d['label']} | – | – | – | – | – |")
            continue
        A(f"| {d['label']} | {a.get('argmax_slot_mean_unique', float('nan')):.2f} | "
          f"{a.get('argmax_slot_perm_null', float('nan')):.2f} | "
          f"{a.get('argmax_slot_mean_first', float('nan')):.2f} | "
          f"{a.get('frac_steps_tied', float('nan')):.1%} | "
          f"{stats[d['dir'].name].get('n_unique_steps', 0)} |")
    A('')
    A('A mean slot above the permutation null is a late-slot advantage; at or below it, '
      'slot order carries no information the value can see.')
    A('')

    # ---- Q3
    A('## Q3 — is slot 15 > slot 7 > slot 0?')
    A('')
    A('Paired **within control step**. `delta` is the mean paired difference over all live '
      'steps; `win` is the fraction of **discriminating** steps the later slot wins. '
      f'Intervals are a cluster bootstrap over the {n_ep} episodes '
      f'({n_boot} draws), because control steps within an episode are autocorrelated. '
      '`+` = interval excludes zero (or 0.5) on the better side, `-` = on the worse side, '
      '`.` = indistinguishable.')
    A('')
    A('### On the verifier value')
    A('')
    A('| arm | 0→7 delta | 0→7 win | 7→15 delta | 7→15 win | 0→15 delta | 0→15 win |')
    A('|---|---|---|---|---|---|---|')
    for d in dumps:
        p = stats[d['dir'].name]['pairs']
        cells = []
        for k in ('0->7', '7->15', '0->15'):
            if k not in p:
                cells += ['–', '–']; continue
            e = p[k]
            cells.append(f"{_fmt_ci(e['delta'], e['delta_ci'])} {e['delta_sig']}")
            cells.append(f"{e['win_rate']:.3f} [{e['win_ci'][0]:.2f},{e['win_ci'][1]:.2f}] "
                         f"{e['win_sig']}")
        A(f"| {d['label']} | " + ' | '.join(cells) + ' |')
    A('')

    if any('terms' in stats[d['dir'].name] for d in dumps):
        A('### Decomposed into the two raw-pixel distances')
        A('')
        A('`armTn` = `-(d_t_goal/13.6 + d_arm_t/52.1)`, so the value is a spread-normalized '
          'composite of two different distances. **A gain that lives entirely in '
          '`d_arm_t` is not task progress** — it is the arm parking closer to the T, the '
          'exact behaviour that retired the raw `armT` value. Both terms are reported as '
          'pixels REDUCED, so positive is always better.')
        A('')
        A('| arm | term | blind | 0→15 Δpx | 0→15 win | 7→15 Δpx |')
        A('|---|---|---:|---|---|---|')
        for d in dumps:
            tm = stats[d['dir'].name].get('terms')
            if not tm:
                continue
            for name, blk in tm.items():
                e = blk.get('0->15'); f = blk.get('7->15')
                if not e:
                    continue
                A(f"| {d['label']} | `{name}` | {blk['frac_blind']:.1%} | "
                  f"{_fmt_ci(e['delta'], e['delta_ci'], p=2)} {e['delta_sig']} | "
                  f"{e['win_rate']:.3f} [{e['win_ci'][0]:.2f},{e['win_ci'][1]:.2f}] "
                  f"{e['win_sig']} | "
                  + (f"{_fmt_ci(f['delta'], f['delta_ci'], p=2)} {f['delta_sig']} |"
                     if f else '– |'))
        A('')
        A('The `blind` column is per term and is the point of the split: a step where no '
          'candidate can move the block has zero spread in `d_t_goal` while `d_arm_t` still '
          'varies freely. On those steps the ranking is decided by arm reach alone.')
        A('')

    # ---- Q4
    A('## Q4 — slot-to-slot delta, and where it stops improving')
    A('')
    A('Values centered within each control step, then averaged; cluster-bootstrap CI over '
      'episodes. `saturation` is the largest k whose remaining gain (slot 15 − slot k) '
      'still has an interval excluding zero — i.e. improvement is no longer detectable '
      'past it. `None` means no slot showed a detectable remaining gain.')
    A('')
    A('| arm | slot 0 | slot 7 | slot 15 | saturation | sd@0 | sd@15 |')
    A('|---|---|---|---|---:|---:|---:|')
    for d in dumps:
        s = stats[d['dir'].name]
        prof = np.array(s['centered_mean_by_index'], float)
        lo, hi = np.array(s['centered_ci_by_index'], float)
        sd = np.array(s['centered_sd_by_index'], float)
        cells = [f'{prof[k]:+.3f} [{lo[k]:+.3f}, {hi[k]:+.3f}]' if k < len(prof) else '–'
                 for k in SLOTS]
        A(f"| {d['label']} | " + ' | '.join(cells)
          + f" | {s['saturation_slot'] if s['saturation_slot'] is not None else '–'}"
          + f" | {sd[0]:.3f} | {sd[-1]:.3f} |")
    A('')
    A('**`sd@15` below `sd@0` means the context is narrowing exploration, not improving '
      'quality.** That produces a rising mean too, so the two must be read together: a '
      'later slot that is better *and* less varied has converged onto the region the '
      'earlier candidates already found, which is a different claim from having found '
      'something better.')
    A('')
    A('> A guard on the plots: `prefix_max` (best-of-k) rises with k even under a pure '
      'i.i.d. resampler, so it is never evidence of context. It is plotted only against '
      '`prefix_mean`; the gap between them is the value of having a selector.')
    A('')
    A('## Files')
    A('')
    A(f'- `per_prediction_{tier}.csv` — every prediction, every slot 0/7/15, argmax slot')
    A(f'- `slot_profile_{tier}.png` — centered value vs slot, all arms')
    A(f'- `slot_profile_terms_{tier}.png` — the same split into `d_t_goal` and `d_arm_t`')
    A(f'- `slot_stats_{tier}.json` — every number above, machine-readable')
    return '\n'.join(L) + '\n'


@click.command()
@click.option('--dump-dir', 'dump_dirs', multiple=True, required=True,
              help='a directory written by dump_candidate_scores.py; repeatable.')
@click.option('--tier', type=click.Choice(['A', 'B']), required=True)
@click.option('--out-dir', default='analysis/aug27_slot_wts', show_default=True)
@click.option('--out', default=None, help='markdown path [<out-dir>/slot_context_<tier>.md]')
@click.option('--n-boot', default=N_BOOT, show_default=True)
def main(dump_dirs, tier, out_dir, out, n_boot):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dumps = [load_dump(d) for d in dump_dirs]
    dumps.sort(key=lambda d: (d['slot_semantics'] != 'trained_staircase', d['label']))

    ns = {d['n'] for d in dumps}
    assert len(ns) == 1, f'dumps disagree on search width n: {ns}'
    vs = {d['verifier'] for d in dumps}
    if len(vs) > 1:
        print(f'WARNING: dumps were scored under different verifier values {vs}; '
              f'their values are not on a common scale and must not be compared.')

    stats, rows = {}, []
    for d in dumps:
        print(f"  {d['label']} @{d['step']} ...", flush=True)
        stats[d['dir'].name] = slot_stats(d, n_boot=n_boot)
        rows += per_prediction_rows(d)

    csv_path = out_dir / f'per_prediction_{tier}.csv'
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'  {len(rows)} predictions -> {csv_path}')

    plot_profiles(dumps, stats, out_dir / f'slot_profile_{tier}.png',
                  f'Tier {tier}: centered verifier value by candidate slot (n=16)')
    if any('terms' in s for s in stats.values()):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        for ax, term in zip(axes, (TERM_LABEL[T_GOAL], TERM_LABEL[ARM_T])):
            for d in dumps:
                tp = d['per_ep_terms']
                if tp is None:
                    continue
                axis = T_GOAL if term == TERM_LABEL[T_GOAL] else ARM_T
                cen = [e[:, :, axis] - e[:, :, axis].mean(axis=1, keepdims=True) for e in tp]
                y = _boot(cen, n_boot=n_boot, seed=7).point
                iid = d['slot_semantics'] != 'trained_staircase'
                ax.plot(np.arange(len(y)), y, marker='o', ms=3,
                        ls='--' if iid else '-', color='0.35' if iid else None,
                        lw=2.2 if iid else 1.5,
                        label=f"{d['label']}" + (' [null]' if iid else ''))
            ax.axhline(0, color='k', lw=0.8, alpha=0.5)
            # distances: NEGATIVE is better, so the axis is inverted to keep "up = better"
            ax.invert_yaxis()
            ax.set_title(f'{term} (px, centered) — axis inverted so UP is better')
            ax.set_xlabel('candidate slot k')
        axes[0].set_ylabel('centered distance (px)')
        axes[0].legend(fontsize=7)
        fig.suptitle(f'Tier {tier}: the two distance terms behind the armTn value')
        fig.tight_layout()
        fig.savefig(out_dir / f'slot_profile_terms_{tier}.png', dpi=150)
        plt.close(fig)

    (out_dir / f'slot_stats_{tier}.json').write_text(json.dumps(
        {d['dir'].name: {'meta': {k: d['meta'].get(k) for k in
                                  ('arm', 'step', 'checkpoint', 'n', 'max_actions',
                                   'slot_semantics', 'verifier_value', 'success_rate')},
                         **stats[d['dir'].name]} for d in dumps}, indent=2, default=float))

    md = pathlib.Path(out or (out_dir / f'slot_context_{tier}.md'))
    md.write_text(render_md(dumps, stats, tier, out_dir, n_boot=n_boot))
    print(f'  report -> {md}')


if __name__ == '__main__':
    main()
