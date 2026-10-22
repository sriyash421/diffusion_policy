# Chunked PushT — far50 seeds and held-out quadrants (status 2026-10-21)

**Chunked PushT**: search is per action chunk (16-step horizon, 8 executed, K≤16 candidates
scored at the current state, closed loop). Not the full-trajectory search of maze `dt_st_traj`;
no full-trajectory PushT arm exists.

## What these runs are

50 demos per split (the 50 training episodes farthest from the held-out quadrant), 50 test
episodes (46 for tlq50far), corrected protocol (`fresh_context_v3`, terminal scoring,
`paired_decision_v3`). Five arms × three held-out quadrants × seeds 42/43/44 = 45 runs.

| Campaign (`pusht_search/…`) | Adds |
|---|---|
| `pusht_far50_20261003` | blq50far (bottom-left held out), seed 42 |
| `pusht_far50seeds_20261006` | blq50far seeds 43, 44 |
| `pusht_tlq50far_20261006` | tlq50far (top-left held out), seeds 42–44 |
| `pusht_brq50far_20261006` | brq50far (bottom-right held out), seeds 42–44 |

## Training status — stopped 2026-10-21 22:46 UTC

Training was stopped by request at each run's last saved 10k checkpoint; it will not be resumed.
BC and ST k1 finished all 9 runs at 100k. ST k16 final checkpoints (seed 42 / 43 / 44):

| Split | uniform | flat600 | ramp200to600 |
|---|---|---|---|
| blq50far | 100k / 30k / 40k | 100k / 90k / 70k | 60k / 50k / 40k |
| tlq50far | 70k / 40k / 60k | 30k / 90k / 50k | 70k / 30k / 40k |
| brq50far | 80k / 60k / 80k | 40k / 70k / 50k | 40k / 80k / 30k |

2 of 27 ST k16 runs reached 100k. Most of the shortfall came from a resume bug: `latest.ckpt`
was written only every 10k steps, so a 9 h checkpoint-partition slot (~6–7k steps) usually ended
before the next save and its work was lost; 16 runs made no saved progress for 7–11 days. Fixed
2026-10-21 (`checkpoint.latest_every: 1000`), only hours before the stop.

## Evaluation status

Every 10k checkpoint is evaluated at N = 1…64, argmax and final-candidate, clean and corrupted
observations (corrupted only for the noise arms). **All evaluations are complete** (339 eval
arrays, 2026-10-22 01:35 UTC). The last five — blq50far seed 43 uniform 30k, seed 44 flat600 70k,
seed 44 ramp 20k/30k/40k — were missed while the far50seeds watcher was stopped (Oct 14–21) and
were run after training stopped.

## Results — success rate, argmax selection, clean observations, mean over 3 seeds

N = candidates per decision. "Final" = each run's final checkpoint above (BC/k1: 100k).

| Split | Arm | 10k N=1 | 10k N=16 | final N=1 | final N=16 |
|---|---|---|---|---|---|
| blq50far | BC | 0.02 | 0.34 | 0.07 | 0.12 |
| | ST k1 | 0.00 | 0.14 | 0.07 | 0.06 |
| | ST k16 uniform | 0.01 | 0.12 | 0.01 | 0.14 |
| | ST k16 flat600 | 0.00 | 0.23 | 0.01 | 0.15 |
| | ST k16 ramp200to600 | 0.01 | 0.02 | 0.02 | 0.09 |
| tlq50far | BC | 0.04 | 0.38 | 0.16 | 0.23 |
| | ST k1 | 0.01 | 0.17 | 0.09 | 0.15 |
| | ST k16 uniform | 0.03 | 0.29 | 0.05 | 0.22 |
| | ST k16 flat600 | 0.00 | 0.33 | 0.01 | 0.19 |
| | ST k16 ramp200to600 | 0.04 | 0.37 | 0.06 | 0.25 |
| brq50far | BC | 0.02 | 0.34 | 0.07 | 0.17 |
| | ST k1 | 0.01 | 0.05 | 0.04 | 0.13 |
| | ST k16 uniform | 0.01 | 0.11 | 0.06 | 0.08 |
| | ST k16 flat600 | 0.01 | 0.25 | 0.01 | 0.22 |
| | ST k16 ramp200to600 | 0.03 | 0.31 | 0.02 | 0.11 |

Final-candidate execution (no selection) at N=16 stays at 0.00–0.11 for every ST arm, well below
argmax at the same N: most of the N=16 gain comes from verifier selection, not from later
candidates improving on earlier ones. The best cells anywhere in these sweeps are the nearly untrained 2k
checkpoints at N=64 (0.80–0.93, `selected_best_sweep.json`), which says best-of-N rewards
candidate diversity; those cells are selected on the test set.

## Argmax vs final candidate — per-decision streams (2026-10-22)

Why argmax selection scales with N while executing the final candidate stays at 0.00: per-decision candidate
streams (N = 64, 50 test episodes, seeds 42–44) at each blq50far arm's best sweep checkpoint (2k for every ST arm,
10k for BC). Report: [l2s pushT argmax v final candidate](https://wandb.ai/l2sml/pushT_diffusion_search/reports/l2s-pushT-argmax-v-final-candidate--VmlldzoxODE2MTY2OA==);
markdown copy `docs/reports/pusht_argmax_v_final_candidate_2026-10-22.md`.

- Argmax gains 15–39 score units per decision from N = 1 to 64, ~90% of it by picking a higher-scoring miss; a
  candidate whose chunk finishes the task exists in only 2–5% of decisions. Closed loop those gains compound to
  0.53–0.80 success at N = 64.
- Later candidates are not better: candidate 1 is marginally the best (+0.3 to +5.5 vs the decision mean), every
  later position sits at 0, and records above the permutation null are zero or negative (no steering). Candidate N is
  therefore one draw from a flat tail, and closed-loop final-pass success is 0.00 at every N for every ST arm.
- The argmax lands uniformly over the 64 slots (median position 26–28; the maze's sits at #4–10 because goal-reaching
  candidates tie at the top). The executed final-pass sample beats the best of its 63 context candidates in 1–2% of
  decisions and has mean rank 28–32 of 64.
- Reruns reproduce the sweeps' success rates within a few episodes but are not bit-identical (GPU type / code drift;
  run-to-run on one GPU type is bit-identical). `--store-scores` is now on for future evals, so streams are kept.

## Where things are

- Results: `pusht_search/live_reports/pusht_simpler_corrected_verifier_results.csv`; W&B report
  "PushT 50-demo results across datasets and seeds" (auto-refreshed by the keepalive job).
- Per-checkpoint evals: `<run>/bon_search_sel-*/step_*/success_curve.json`.
- Job registry and event log: `pusht_search/keepalive/{registry.json,events.jsonl}`.
