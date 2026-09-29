"""Per-candidate-slot statistics over a search dump, in pure numpy.

Split out of scripts/dump_candidate_scores.py so the read-only reporting scripts can use
the SAME definitions without importing the simulator. That script reaches these functions
through eval_search_pusht -> pusht_verifier -> pusht_env -> pygame, a chain that costs ~88 s
and ~440 MB and fails outright in some non-interactive shells; a script that only reads an
.npz and prints a table has no business paying for it. Duplicating the functions instead
would let the dump's notion of "degenerate" and the analysis's drift apart, which is the
one thing every statistic here depends on.

numpy only -- no torch, no gym, no pygame. Keep it that way.

The two constants and the reasoning behind each statistic (the permutation null, the tie
conventions, why the analytic 1/(k+1) and (n-1)/2 baselines are traps) are documented on
the functions themselves.
"""
import numpy as np


def _record_rate(X):
    """P(column k is a new running max over columns 0..k-1), per column."""
    running_max = np.maximum.accumulate(X, axis=1)
    is_record = np.zeros_like(X, dtype=bool)
    is_record[:, 0] = True
    is_record[:, 1:] = X[:, 1:] > running_max[:, :-1]
    return is_record.mean(axis=0)


# Spread at or below which a control step carries no ordering information at all: every
# candidate returned the identical sim value. See _blindness_profile.
BLIND_EPS = 1e-9
# Tolerance on "is a maximizer". Wider than BLIND_EPS because a 1e-7 px margin between two
# candidates is float noise in a contact sim, not a decision the policy made.
TIE_TAU = 1e-6


def _blindness_profile(scores, alive, eps=BLIND_EPS):
    """How often the verifier cannot tell the candidates apart, and WHERE.

    The verifier value is a function of where the T-block ends up after simulating a
    candidate's chunk. When no candidate contacts the block, the block does not move and
    every candidate returns the byte-identical current distance -- the search carries zero
    information and the argmax is picking arbitrarily.

    The intuitive story ("blind during the approach, then works") is wrong by ~5x on
    measured data: blindness spikes at t=0 but only ~17% of blind steps sit in that leading
    run. The rest are mid-episode losses of contact -- the agent backing off, repositioning,
    or circling the T to push from another face. `frac_blind_in_leading_run` is the number
    that settles it, which is why it is reported rather than just the overall rate.

    `scores` (T,B,n), `alive` (T,B). Returns a dict of plain lists/floats.
    """
    T, B, n = scores.shape
    blind = (scores.max(-1) - scores.min(-1)) <= eps          # (T, B)
    by_t, live_t = [], []
    for t in range(T):
        m = alive[t]
        live_t.append(int(m.sum()))
        by_t.append(float(blind[t][m].mean()) if m.any() else float('nan'))
    lead, total, lengths = [], [], []
    for b in range(B):
        d = blind[:, b][alive[:, b]]
        lengths.append(int(len(d)))
        if len(d) == 0:
            lead.append(0); total.append(0); continue
        lead.append(int(np.argmax(~d)) if (~d).any() else int(len(d)))
        total.append(int(d.sum()))
    tot = sum(total)
    live = alive.reshape(-1)
    S = scores.reshape(-1, n)[live]
    spread = S.max(1) - S.min(1)
    disc = spread[spread > eps]
    return {
        'blind_rate_overall': float(blind.reshape(-1)[live].mean()),
        'blind_rate_by_step_index': by_t,
        'live_by_step_index': live_t,
        'blind_leading_run_by_episode': lead,
        'blind_total_by_episode': total,
        'episode_len_by_index': lengths,
        'blind_leading_run_mean': float(np.mean(lead)) if lead else 0.0,
        'blind_total_mean': float(np.mean(total)) if total else 0.0,
        # 1.0 would mean blindness is purely an approach-phase artifact; measured ~0.17
        'frac_blind_in_leading_run': float(sum(lead) / tot) if tot else 0.0,
        'spread_median_discriminating': float(np.median(disc)) if disc.size else 0.0,
        'spread_quantiles_discriminating': (
            [float(q) for q in np.quantile(disc, [.1, .25, .5, .75, .9])]
            if disc.size else [0.0] * 5),
    }


def _argmax_slot_stats(S, rng, tau=TIE_TAU, n_perm=200):
    """Which candidate slot the argmax lands on, under five tiebreak conventions.

    THE PROBLEM. np.argmax returns the FIRST maximizer. On a fully degenerate step every
    slot is a maximizer, so it returns 0 -- 100% of the time. With ~21% of steps degenerate
    and another ~12% partially tied, roughly a third of the samples are decided by numpy's
    tiebreak rule rather than by the policy, and the mean is dragged toward 0. Measured on
    one reference dump the answer ranges 4.32 (first) to 8.92 (last) -- a spread of 4.6
    slots out of 16, wider than any effect worth looking for.

    THE TWO QUESTIONS, which are not the same and must not be conflated:
      first   -- which slot was actually EXECUTED. select_candidate uses
                 scores.argmax(dim=1), also first-maximizer, so this is literally true of
                 the deployed policy. It is also a fact about the tiebreak rule.
      unique  -- which slot genuinely WINS, restricted to steps with a single maximizer,
                 i.e. the steps where the verifier could tell the candidates apart.
    `last` and `random` bracket `first`; `tiemean` is the tie set's centre of mass.

    THE BASELINE is the permutation null -- each step's own value multiset, reshuffled --
    computed under the SAME convention, never the analytic (n-1)/2. Ties are rife here, so
    a shuffled sequence already departs from (n-1)/2; comparing against it would score a
    tie artefact as a real effect. Same reasoning as the record-rate null in analyse().
    """
    n_steps, n = S.shape
    if n_steps == 0:
        return {}
    mx = S.max(1, keepdims=True)
    tie = S >= (mx - tau)                                   # (S, n) is-a-maximizer
    ntie = tie.sum(1)
    idx = np.arange(n)

    def conventions(M):
        t = M >= (M.max(1, keepdims=True) - tau)
        first = M.argmax(1)
        last = n - 1 - t[:, ::-1].argmax(1)
        tiemean = (t * idx).sum(1) / np.maximum(t.sum(1), 1)
        # inverse-CDF pick among the maximizers: unbiased under exchangeability
        cum = np.cumsum(t, axis=1)
        draw = (rng.random(len(M)) * t.sum(1))[:, None]
        rand = (cum <= draw).sum(1).clip(0, n - 1)
        return {'first': first, 'last': last, 'random': rand, 'tiemean': tiemean}

    obs = conventions(S)
    uniq = ntie == 1
    out = {
        'n_argmax_steps': int(n_steps),
        'n_unique_argmax_steps': int(uniq.sum()),
        'frac_steps_tied': float((~uniq).mean()),
        'argmax_tie_count_mean': float(ntie.mean()),
        'argmax_slot_uniform_null': float((n - 1) / 2),
    }
    for k, v in obs.items():
        out[f'argmax_slot_mean_{k}'] = float(np.mean(v))
        out[f'argmax_slot_median_{k}'] = float(np.median(v))
    if uniq.any():
        u = S[uniq].argmax(1)
        out['argmax_slot_mean_unique'] = float(u.mean())
        out['argmax_slot_median_unique'] = float(np.median(u))
    else:
        out['argmax_slot_mean_unique'] = float('nan')
        out['argmax_slot_median_unique'] = float('nan')

    # margin between the best and the runner-up: says whether a "decision" was real
    part = np.partition(S, -2, axis=1) if n >= 2 else None
    if part is not None:
        margin = part[:, -1] - part[:, -2]
        out['argmax_margin_px_mean'] = float(margin.mean())
        out['argmax_margin_px_median'] = float(np.median(margin))

    # permutation null, PER CONVENTION -- holds the tie multiset fixed, destroys only order
    perm = {k: [] for k in obs}
    perm_uniq = []
    for _ in range(n_perm):
        P = rng.permuted(S, axis=1)
        c = conventions(P)
        for k in perm:
            perm[k].append(np.mean(c[k]))
        pu = (P >= (P.max(1, keepdims=True) - tau)).sum(1) == 1
        perm_uniq.append(float(P[pu].argmax(1).mean()) if pu.any() else np.nan)
    for k, v in list(perm.items()) + [('unique', perm_uniq)]:
        key = f'argmax_slot_perm_null_{k}'
        out[key] = float(np.nanmean(v))
        out[key + '_sd'] = float(np.nanstd(v, ddof=1))
    # the headline baseline: the permutation null under the `unique` convention
    out['argmax_slot_perm_null'] = out['argmax_slot_perm_null_unique']
    return out


def _slot_semantics(max_actions, n):
    """Whether the slot INDEX means anything for this (policy, search width).

    UNet BC and ST k=1 draw i.i.d. candidates, so their argmax-slot distribution is not a
    baseline OF the statistic -- it IS the statistic's null distribution, measured on real
    data with the real tie structure. Labelling that in the dump is what stops a reader
    treating a flat histogram as a finding.
    """
    if max_actions >= (1 << 20):
        return ('iid_unbounded',
                'UNet BC: predict_action discards the search context, so the n draws are '
                'i.i.d. and slot order carries no information BY CONSTRUCTION.')
    if max_actions <= 1:
        return ('iid_no_context',
                'ST k=1: max_actions == 1, so any n>1 runs the rolling window with an '
                'empty history. Effectively i.i.d.; slot order carries no information.')
    if n <= max_actions:
        return ('trained_staircase',
                f'slots 0..{n-1} are the trained staircase: slot k attends to exactly the '
                f'first k scored candidates.')
    return ('mixed_rolling',
            f'slots 0..{max_actions-1} are the trained staircase; every candidate beyond '
            f'that is drawn from slot {max_actions-1}\'s conditional via the rolling '
            f'window, so the index is not a slot id past there.')


def analyse(scores, chosen, alive, score_final=None, n_perm=200, seed=0):
    """Per-candidate-index statistics over every live control step.

    `scores` (T,B,n) -> flattened to (S,n) keeping only live steps, since a finished
    episode's padded steps are not states the policy visited.
    """
    S = scores.reshape(-1, scores.shape[-1])[alive.reshape(-1)]        # (S, n)
    ch = chosen.reshape(-1)[alive.reshape(-1)]                         # (S,)
    n_steps, n = S.shape
    idx = np.arange(n)

    # Steps where every candidate scored identically carry no ordering information at all
    # (the agent is nowhere near the block, so nothing any candidate does moves it). They
    # are counted and reported, but excluded from the record test, where they would only
    # dilute both the observed rate and the null by the same factor.
    spread = S.max(axis=1) - S.min(axis=1)
    degenerate = spread <= 1e-9
    live = ~degenerate
    Sl = S[live]

    # per-index level. `centered` removes each control step's own difficulty -- raw means
    # are dominated by which states happen to be easy, not by candidate order.
    centered = S - S.mean(axis=1, keepdims=True)
    sem = S.std(axis=0, ddof=1) / np.sqrt(n_steps)
    csem = centered.std(axis=0, ddof=1) / np.sqrt(n_steps)

    # THE TEST: is candidate k a new running max more often than ORDER ALONE predicts?
    #
    # The baseline is the permutation null -- each step's own candidate values, reshuffled.
    # It is exchangeable by construction, so it isolates exactly the ordering effect while
    # holding the value multiset (and its ties) fixed. The analytic 1/(k+1) is reported too
    # but is NOT the comparison: these scores tie often enough that a shuffled sequence
    # already sits ~30% below it, so measuring against 1/(k+1) would score a tie artefact as
    # a real effect. See the module docstring.
    record_rate = _record_rate(Sl)
    rng = np.random.default_rng(seed)
    perm = np.stack([_record_rate(rng.permuted(Sl, axis=1)) for _ in range(n_perm)])
    perm_rate = perm.mean(axis=0)
    perm_sd = perm.std(axis=0, ddof=1)
    null_rate = 1.0 / (idx + 1.0)
    # binomial SE of the empirical rate, for "is the gap real"
    record_se = np.sqrt(np.clip(record_rate * (1 - record_rate), 0, None) / max(len(Sl), 1))

    # trend of the centered score against index: OLS slope and Spearman rho
    x = np.repeat(idx[None, :], n_steps, axis=0).reshape(-1)
    y = centered.reshape(-1)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    slope_se = float(np.sqrt((resid ** 2).sum() / (len(y) - 2) / ((x - x.mean()) ** 2).sum()))
    rho = float(_spearman(x, y))

    # "does a bigger n raise the AVERAGE candidate, or only the max?" -- prefix statistics
    # over candidates 0..n-1. prefix_mean is the question as asked; prefix_max is the
    # best-of-n curve, and the gap between them is the entire value of having a selector.
    #
    # NOTE a regime change at max_actions: predict_n_actions switches to a rolling window
    # past K, so candidates 0..K-1 come from slots 0..K-1 while every candidate beyond that
    # is drawn from slot K-1's conditional. A rise in prefix_mean across that boundary is
    # partly a change of generator, not a trend within one.
    csum = np.cumsum(S, axis=1)
    denom = np.arange(1, n + 1, dtype=np.float64)[None, :]
    prefix_mean = (csum / denom).mean(axis=0)                 # (n,)
    prefix_max = np.maximum.accumulate(S, axis=1).mean(axis=0)

    # Per-slot version of the deployed-sample analysis: treat EVERY candidate index as if it
    # were the one executed, and ask the same questions. `rank_by_index` is how many of the n
    # candidates beat candidate k (0 = best); `p_is_best_by_index` is how often it is the
    # outright winner. Uniform expectations are (n-1)/2 and 1/n, so departures say whether a
    # slot is systematically better-positioned than its siblings rather than merely different.
    order = (S[:, :, None] < S[:, None, :]).sum(axis=2)      # (S_steps, n) rank of each slot
    rank_by_index = order.mean(axis=0)
    p_best_by_index = (order == 0).mean(axis=0)               # tie-INCLUSIVE: "is a maximizer"

    # The baseline is the MEAN ACROSS SLOTS, not (n-1)/2 and 1/n. Under any exchangeable
    # ordering every slot has the same expected rank, so the across-slot average IS the
    # permutation null -- and unlike the analytic values it is unaffected by ties, which are
    # rife here (many candidates never touch the block and return identical values, so a
    # dozen slots can be jointly "best"). Quoting 1/n would make every slot look ~10x better
    # than chance when the effect is entirely ties, the same trap as the 1/(k+1) record null.
    out_final = {'rank_by_index': rank_by_index.tolist(),
                 'p_is_best_by_index': p_best_by_index.tolist(),
                 'rank_baseline': float(rank_by_index.mean()),
                 'p_is_best_baseline': float(p_best_by_index.mean()),
                 'rank_uniform_naive': float((n - 1) / 2)}
    if score_final is not None:
        F = score_final.reshape(-1)[alive.reshape(-1)]        # (S_all,) deployed sample
        Sa = scores.reshape(-1, scores.shape[-1])[alive.reshape(-1)]
        # rank of the deployed sample among the n candidates it was conditioned on;
        # 0 == better than all of them. Uniform expectation is n/2.
        rank = (Sa > F[:, None]).sum(axis=1)
        # How often does the deployed sample beat the best of the FIRST n candidates, as a
        # function of n? This is the crossover the arm lives or dies on: at n=1 it only has
        # to beat a single draw, at n=64 it has to beat the max of 64. Where this curve
        # crosses 0.5 is the search width past which selection wins.
        pmax = np.maximum.accumulate(Sa, axis=1)             # (S_steps, n)
        beats_prefix = (F[:, None] > pmax).mean(axis=0)      # (n,)
        gap_prefix = (F[:, None] - pmax).mean(axis=0)        # (n,) px
        out_final.update({
            'final_beats_prefix_max': beats_prefix.tolist(),
            'final_minus_prefix_max': gap_prefix.tolist(),
        })
        out_final.update({
            'final_mean': float(F.mean()),
            'final_rank_mean': float(rank.mean()),
            'final_rank_uniform': float(n / 2),
            'final_percentile': float((rank / max(n - 1, 1)).mean()),
            'p_final_is_best': float((rank == 0).mean()),
            'p_final_beats_argmax': float((F > Sa.max(axis=1)).mean()),
            'final_minus_mean': float((F - Sa.mean(axis=1)).mean()),
            'final_minus_best': float((F - Sa.max(axis=1)).mean()),
            'final_minus_first': float((F - Sa[:, 0]).mean()),
        })

    # Argmax-slot statistics, on the SAME `live` mask the record test uses -- one
    # definition of "degenerate", not two that can drift apart. Also computed unscoped, so
    # the effect of the scoping is itself visible rather than asserted.
    rng2 = np.random.default_rng(seed + 1)
    slot_scoped = _argmax_slot_stats(Sl, rng2, n_perm=n_perm)
    slot_all = _argmax_slot_stats(S, np.random.default_rng(seed + 2), n_perm=n_perm)
    blind = _blindness_profile(scores, alive)

    return {
        'n_control_steps': int(n_steps),
        'n_degenerate_steps': int(degenerate.sum()),
        'n_test_steps': int(live.sum()),
        # scoped to discriminating steps -- the headline
        **slot_scoped,
        # same statistics over every live step, degenerate ones included
        **{f'unscoped_{k}': v for k, v in slot_all.items()},
        **blind,
        'prefix_mean': prefix_mean.tolist(),
        'prefix_max': prefix_max.tolist(),
        **out_final,
        'index': idx.tolist(),
        'mean_by_index': S.mean(axis=0).tolist(),
        'sem_by_index': sem.tolist(),
        'centered_mean_by_index': centered.mean(axis=0).tolist(),
        'centered_sem_by_index': csem.tolist(),
        # within-step spread of candidate k: how much room that slot still explores
        'centered_sd_by_index': centered.std(axis=0, ddof=1).tolist(),
        'running_max_by_index': np.maximum.accumulate(S, axis=1).mean(axis=0).tolist(),
        'record_rate': record_rate.tolist(),
        'record_rate_se': record_se.tolist(),
        'perm_null_rate': perm_rate.tolist(),
        'perm_null_sd': perm_sd.tolist(),
        'iid_null_rate': null_rate.tolist(),
        # summed over k>=1: how many more records than SHUFFLING the same values gives.
        # 0 == order carries no information == the search is resampling.
        'excess_records': float((record_rate[1:] - perm_rate[1:]).sum()),
        'excess_records_iid': float((record_rate[1:] - null_rate[1:]).sum()),
        'argmax_hist': (np.bincount(ch, minlength=n) / len(ch)).tolist(),
        'p_argmax_late': float((ch >= n / 2).mean()),
        'ols_slope': float(slope),
        'ols_slope_se': slope_se,
        'spearman_rho': rho,
        'score_first_mean': float(S[:, 0].mean()),
        'score_last_mean': float(S[:, -1].mean()),
        'score_best_mean': float(S.max(axis=1).mean()),
        'spread_mean': float((S.max(axis=1) - S.min(axis=1)).mean()),
    }


def _spearman(x, y):
    """Spearman rho without scipy (not guaranteed in this env)."""
    def rank(a):
        order = a.argsort()
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        # average ties, which x is full of (it is a repeated 0..n-1 grid)
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return 0.0 if denom == 0 else float((rx * ry).sum() / denom)
