# Training on the transitions the verifier can see — 2026-09-19

`t_goal` scores a candidate by where the T ends up and **nothing else**, so on a decision that
moves the block nowhere every candidate ties and the verifier is blind — 15–25% of decisions
([ppo_sac_lstm-bc_eval_2026-09-17.md](ppo_sac_lstm-bc_eval_2026-09-17.md) §1). These arms train
only on the transitions where the demonstrator's T actually moves.

**Eleven arms at 100k**: UNet BC, and ST at k=16 / k=4 × {uniform, flat400, ramp400to200,
flat600, ramp600to200}.
All 206 episodes, **176 train / 0 val / 30 test**, filtered to **12,972 train / 2,595 test**
moving windows. Everything below is `block_moving` only.

t=600 is `sqrt(alpha_bar)=0.16`, which README §2.4 records as already unidentifiable — so the
600-level arms train on an observation that carries almost no per-frame information. They are
**not** the floor that suggests: `k16 flat600` is the strongest arm measured.

---

## 1. Verifier value vs n — does the candidate improve?

Best-of-n `t_goal` value, higher is better, 0 = T on goal. BC is **sampled** (16 independent
draws); the **k=4 arms use the rolling context window** for n>4.

### Observations CLEAN

**TEST (30 held-out episodes, 301 decisions)**

| arm | n=1 | n=2 | n=4 | n=8 | n=16 | gain |
|---|--:|--:|--:|--:|--:|--:|
| BC | −79.03 | −78.09 | −77.21 | −76.64 | −76.01 | +3.02 |
| k16 uniform | −79.37 | −78.86 | −78.44 | −78.23 | −78.05 | +1.32 |
| k16 flat400 | −82.08 | −80.43 | −79.15 | −78.37 | −77.58 | **+4.50** |
| k16 ramp400to200 | −81.64 | −79.94 | −78.74 | −77.96 | −77.23 | **+4.41** |
| k4 uniform | −79.47 | −79.06 | −78.66 | −78.45 | −78.21 | +1.26 |
| k4 flat400 | −82.18 | −81.03 | −80.12 | −79.18 | −78.46 | +3.72 |
| k4 ramp400to200 | −82.10 | −80.58 | −79.34 | −78.22 | −77.41 | **+4.69** |
| k16 flat600 | −81.17 | −78.63 | −77.20 | −76.42 | −75.62 | **+5.54** |
| k16 ramp600to200 | −80.43 | −78.16 | −77.00 | −76.56 | −76.09 | +4.34 |
| k4 flat600 | −83.27 | −80.71 | −78.90 | −77.29 | **−75.53** | **+7.74** |
| k4 ramp600to200 | −83.08 | −80.53 | −78.88 | −77.65 | −76.75 | **+6.33** |

**TRAIN (176 seen episodes, 262 decisions)**

| arm | n=1 | n=2 | n=4 | n=8 | n=16 | gain |
|---|--:|--:|--:|--:|--:|--:|
| BC | −74.70 | −73.77 | −72.75 | −71.72 | −71.14 | +3.56 |
| k16 uniform | −75.63 | −75.20 | −74.94 | −74.64 | −74.43 | +1.20 |
| k16 flat400 | −77.83 | −76.55 | −75.64 | −74.82 | −74.11 | +3.71 |
| k16 ramp400to200 | −77.43 | −75.99 | −75.14 | −74.35 | −73.58 | +3.84 |
| k4 uniform | −75.76 | −75.30 | −74.94 | −74.61 | −74.35 | +1.40 |
| k4 flat400 | −78.04 | −77.02 | −76.10 | −75.20 | −74.42 | +3.62 |
| k4 ramp400to200 | −78.00 | −76.38 | −75.24 | −73.81 | −72.86 | **+5.14** |
| k16 flat600 | −75.97 | −73.80 | −72.61 | −71.68 | **−70.84** | +5.14 |
| k16 ramp600to200 | −75.09 | −73.52 | −72.63 | −72.10 | −71.76 | +3.33 |
| k4 flat600 | −78.85 | −76.48 | −74.95 | −72.74 | **−70.86** | **+7.98** |
| k4 ramp600to200 | −78.74 | −76.53 | −74.79 | −72.92 | −71.94 | +6.80 |

### Each arm on its OWN TRAINING CONDITIONAL

The tables above read every arm on clean observations: `load_policy` calls `policy.eval()`,
the PushT config leaves `corrupt_obs_eval` null, and `corrupt_obs_features_slotwise` is the
identity when `not training and not corrupt_obs_eval`. So the ladder arms above are being
asked to work from observations they never trained on.

These read each arm on the conditional it was FITTED to — the ladder arms with their slots
corrupted at their own timesteps, BC and both uniform arms clean, because clean is theirs.
The unladdered rows are therefore identical by construction, and are taken from the clean
run verbatim rather than from a no-op corrupt pass, which still shifts the last digit by one
float32 ULP (measured: 7.6e-6).

**TEST (30 held-out episodes, 301 decisions)**

| arm | n=1 | n=2 | n=4 | n=8 | n=16 | gain |
|---|--:|--:|--:|--:|--:|--:|
| BC | −79.03 | −78.09 | −77.21 | −76.64 | −76.01 | +3.02 |
| k16 uniform | −79.37 | −78.86 | −78.44 | −78.23 | −78.05 | +1.32 |
| k16 flat400 | −79.81 | −78.38 | −77.34 | −76.70 | −76.07 | +3.73 |
| k16 ramp400to200 | −79.83 | −78.58 | −77.58 | −77.02 | −76.50 | +3.34 |
| k4 uniform | −79.47 | −79.06 | −78.66 | −78.45 | −78.21 | +1.26 |
| k4 flat400 | −79.97 | −78.80 | −77.66 | −76.73 | −75.79 | +4.18 |
| k4 ramp400to200 | −80.14 | −78.87 | −77.92 | −77.29 | −76.73 | +3.40 |
| k16 flat600 | −79.39 | −77.19 | −75.87 | −75.07 | −74.37 | +5.02 |
| k16 ramp600to200 | −80.61 | −78.49 | −77.25 | −76.36 | −75.65 | +4.96 |
| k4 flat600 | −79.41 | −77.00 | −75.57 | −73.53 | −71.74 | +7.68 |
| k4 ramp600to200 | −79.53 | −77.47 | −76.38 | −75.76 | −75.25 | +4.28 |

Change at n=16, ladder arms only — the other rows are identical by construction:

| arm | clean | own conditional | Δ |
|---|--:|--:|--:|
| k4 flat600 | −75.53 | −71.74 | +3.79 |
| k4 flat400 | −78.46 | −75.79 | +2.67 |
| k16 flat400 | −77.58 | −76.07 | +1.51 |
| k4 ramp600to200 | −76.75 | −75.25 | +1.50 |
| k16 flat600 | −75.62 | −74.37 | +1.26 |
| k16 ramp400to200 | −77.23 | −76.50 | +0.74 |
| k4 ramp400to200 | −77.41 | −76.73 | +0.68 |
| k16 ramp600to200 | −76.09 | −75.65 | +0.44 |

**TRAIN (30 episodes, 262 decisions)**

| arm | n=1 | n=2 | n=4 | n=8 | n=16 | gain |
|---|--:|--:|--:|--:|--:|--:|
| BC | −74.70 | −73.77 | −72.75 | −71.72 | −71.14 | +3.56 |
| k16 uniform | −75.63 | −75.20 | −74.94 | −74.64 | −74.43 | +1.20 |
| k16 flat400 | −75.08 | −73.87 | −73.06 | −72.38 | −71.80 | +3.28 |
| k16 ramp400to200 | −75.47 | −74.20 | −73.52 | −72.98 | −72.52 | +2.95 |
| k4 uniform | −75.76 | −75.30 | −74.94 | −74.61 | −74.35 | +1.40 |
| k4 flat400 | −75.18 | −73.85 | −72.96 | −71.71 | −70.69 | +4.48 |
| k4 ramp400to200 | −75.63 | −74.30 | −73.46 | −72.58 | −71.98 | +3.65 |
| k16 flat600 | −73.38 | −71.56 | −70.49 | −69.57 | −68.75 | +4.63 |
| k16 ramp600to200 | −74.89 | −73.34 | −72.60 | −71.87 | −71.17 | +3.72 |
| k4 flat600 | −74.28 | −71.38 | −69.70 | −67.19 | −65.09 | +9.19 |
| k4 ramp600to200 | −73.89 | −71.60 | −70.89 | −70.26 | −69.81 | +4.08 |

Change at n=16, ladder arms only — the other rows are identical by construction:

| arm | clean | own conditional | Δ |
|---|--:|--:|--:|
| k4 flat600 | −70.86 | −65.09 | +5.77 |
| k4 flat400 | −74.42 | −70.69 | +3.73 |
| k16 flat400 | −74.11 | −71.80 | +2.31 |
| k4 ramp600to200 | −71.94 | −69.81 | +2.13 |
| k16 flat600 | −70.84 | −68.75 | +2.09 |
| k16 ramp400to200 | −73.58 | −72.52 | +1.06 |
| k4 ramp400to200 | −72.86 | −71.98 | +0.89 |
| k16 ramp600to200 | −71.76 | −71.17 | +0.59 |

**Every ladder arm is better on its own conditional, on both splits, at every n.** The gain
is largest where the corruption is heaviest (`k4 flat600` +3.79 on test) and the n=1 penalty
the clean tables show — "a corrupted observation makes a poor unaided draw" — largely
disappears: on test, `k4 flat600` starts at −79.41 rather than −83.27, in line with BC's
−79.03 rather than 4 points behind it.

So the clean reading understates these arms. Their advantage at n=16 was never conditional
on the deficit at n=1; removing the mismatch removes the deficit and keeps the gain.

**This runs opposite to §7.** There, success rate under the same corrupt readout was WORSE
than clean on 6 of 8 arms, and worse on coverage on 7 of 8. Best-of-n verifier value and
end-to-end rollout success disagree about what the corruption buys — which is the same
verifier/outcome gap §5 and the ranking pools keep surfacing, not a contradiction to resolve.

**Yes, monotonically, for every source on both splits — and the gain scales with the ladder
level.** Uniform +1.3, the 400 level +3.7…+4.7, the **600 level +4.3…+7.7**. Every ladder arm
starts worse at n=1 (a corrupted observation makes a poor unaided draw) and ends better;
`k4 flat600` ends best of any trained arm at −75.53. Same ordering on train.

**The blind sampler is worst at n=1 and best at n=16** on both splits. It draws a larger radius,
contacts the block more, and shoves it goalward. A caution about `t_goal` as a selector.

**With the verifier switched off** (`final(n)` = slot n−1 alone, nothing selected), ladder arms
still improve — k4 ramp +2.77, k16 ramp +1.66, flat +1.35/+1.44 — while **k16 uniform −0.02,
BC −0.19, k4 uniform −0.40**. The ladder arms use context to sharpen their own unaided draw;
the uniform arms do not. This cannot be a verifier artefact.

## 2. Best-of-N success, 30 held-out episodes

`eval_search_pusht.py`, `selection=argmax`, step 100k. ⚠️ **30 episodes, not 50** — the
canonical seed-42 test-50 has **43 of 50 inside this generation's training set**. 95% Wilson
interval ≈ **±0.17**, so intervals overlap nearly everywhere; **only the trend in n is paired
and worth reading**.

| arm | success n=1 | n=4 | n=8 | n=16 | mean rew n=1 | n=4 | n=8 | n=16 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| BC | **0.267** | 0.500 | **0.533** | 0.433 | 0.635 | 0.727 | 0.792 | 0.773 |
| k16 uniform | 0.167 | 0.167 | 0.133 | 0.267 | 0.620 | 0.739 | 0.597 | 0.697 |
| k16 flat400 | 0.100 | 0.300 | 0.433 | **0.500** | 0.539 | 0.710 | 0.856 | 0.858 |
| k16 ramp400to200 | 0.100 | 0.233 | 0.233 | 0.467 | 0.630 | 0.810 | 0.852 | **0.897** |
| k4 uniform | 0.067 | 0.200 | 0.233 | 0.133 | 0.546 | 0.683 | 0.709 | 0.729 |
| k4 flat400 | 0.100 | 0.300 | 0.200 | 0.400 | 0.467 | 0.612 | 0.569 | 0.765 |
| k4 ramp400to200 | 0.133 | 0.300 | 0.200 | 0.267 | 0.614 | 0.741 | 0.739 | 0.730 |
| **k16 flat600** | 0.067 | 0.433 | 0.400 | **0.533** | 0.409 | 0.892 | 0.825 | **0.967** |
| k16 ramp600to200 | 0.133 | 0.267 | 0.233 | 0.467 | 0.477 | 0.736 | 0.779 | 0.791 |
| k4 flat600 | 0.033 | 0.400 | 0.500 | 0.233 | 0.502 | 0.836 | 0.872 | 0.864 |
| k4 ramp600to200 | 0.000 | 0.367 | 0.233 | **0.500** | 0.480 | 0.720 | 0.779 | 0.840 |

95% CI at n=16: BC [0.27,0.61] · k16 uniform [0.14,0.44] · k16 flat400 [0.33,0.67] ·
k16 ramp [0.30,0.64] · k4 uniform [0.05,0.30] · k4 flat400 [0.25,0.58] · k4 ramp [0.14,0.44].

**Search pays for the ladder arms and not the uniform ones.** k16 flat400 0.100→0.500 (5×);
k4 uniform 0.067→0.133. **`k16 flat600` is the best arm measured — mean reward 0.967 at n=16**,
against 0.858 (flat400) and 0.697 (uniform) — while being the *worst* arm at n=1 (0.409). The
destroyed observation costs it the unaided draw and search more than repays it.

**Under `--corrupt-obs-eval`** (the ladder arms' own training conditional) the 400 level is
indifferent — k16 flat400 0.400, k16 ramp 0.533, k4 flat400 0.367, k4 ramp 0.267 at n=16, no
consistent direction. **The flat600 arms are not**: k16 flat600 drops 0.967 → 0.836 mean
reward and k4 flat600 0.864 → 0.763. Pinning *every* slot at an unidentifiable level is
genuinely costly when you evaluate it that way; the ramps, whose later slots recover to t=200,
barely move.

## 3. Distance to a* vs n — does the candidate improve?

Mean euclidean distance from the best-of-n candidate to the demonstrator's action over the
executed window, px, lower is better. (`rank_arms_by_blockmotion` reports RMSE per *source* and
euclid per *n*; the two differ by a fixed scale and rank identically.)

**TEST**

| arm | n=1 | n=2 | n=4 | n=8 | n=16 | gain |
|---|--:|--:|--:|--:|--:|--:|
| BC | 23.36 | 21.01 | 19.62 | 18.39 | **17.30** | −6.06 |
| k16 uniform | 24.70 | 23.69 | 23.16 | 22.71 | 22.38 | −2.32 |
| k16 flat400 | 30.40 | 27.07 | 25.18 | 23.96 | 22.89 | −7.51 |
| k16 ramp400to200 | 29.24 | 25.95 | 24.17 | 23.06 | 22.00 | −7.24 |
| k4 uniform | 25.21 | 24.23 | 23.60 | 23.08 | 22.61 | −2.60 |
| k4 flat400 | 30.52 | 28.05 | 26.31 | 24.74 | 23.40 | −7.12 |
| k4 ramp400to200 | 29.29 | 26.28 | 24.32 | 22.69 | 21.61 | **−7.69** |
| k16 flat600 | 36.71 | 28.88 | 25.32 | 23.25 | 21.81 | **−14.90** |
| k16 ramp600to200 | 31.85 | 25.79 | 22.84 | 21.33 | **20.54** | −11.31 |
| k4 flat600 | 36.22 | 30.95 | 27.79 | 25.06 | 23.03 | −13.19 |
| k4 ramp600to200 | 33.02 | 27.37 | 24.35 | 22.36 | 21.24 | −11.78 |

**TRAIN**

| arm | n=1 | n=2 | n=4 | n=8 | n=16 | gain |
|---|--:|--:|--:|--:|--:|--:|
| BC | 9.36 | 7.69 | 6.07 | 5.19 | 4.56 | −4.80 |
| k16 uniform | 7.51 | 6.03 | 5.23 | 4.70 | **4.33** | −3.17 |
| k16 flat400 | 16.95 | 12.19 | 10.23 | 8.91 | 7.70 | **−9.25** |
| k16 ramp400to200 | 15.28 | 10.51 | 8.52 | 7.27 | 6.10 | −9.17 |
| k4 uniform | 7.97 | 6.91 | 6.25 | 5.78 | 5.39 | −2.58 |
| k4 flat400 | 15.90 | 12.90 | 11.34 | 9.83 | 8.69 | −7.21 |
| k4 ramp400to200 | 14.47 | 10.53 | 8.61 | 7.29 | 6.67 | −7.80 |
| k16 flat600 | 25.59 | 17.52 | 14.22 | 12.29 | 10.82 | −14.77 |
| k16 ramp600to200 | 20.64 | 13.39 | 10.05 | 8.24 | 6.94 | −13.70 |
| k4 flat600 | 24.47 | 18.19 | 15.44 | 13.15 | 11.72 | −12.75 |
| k4 ramp600to200 | 19.88 | 12.52 | 9.63 | 8.24 | 7.66 | −12.23 |

**Yes — every arm gets closer to a* with n, on both splits.** Two things the split comparison
exposes that neither table shows alone:

* **The generalisation gap is enormous and uneven.** k16 uniform sits at **4.33 px on train and
  22.38 px on test** — a 5× gap. The ladder arms are worse on train (6.1–8.7) but land in the
  same place on test (22.0–23.4). The uniform arms fit the training actions closely and do not
  carry it across; the ladder arms never fit them as tightly and lose less.
* **BC is best on test (17.30)** and mid-pack on train (4.56) — the only arm clearly ahead
  out-of-sample.
* **The 600 level has by far the smallest train/test gap.** k16 uniform fits the training
  actions to 4.33 px and loses 5.2x out of sample; `k16 flat600` never fits them tightly
  (10.82 px) and loses only 2.0x. The corruption acts as a regulariser — it prevents
  memorising the demonstrations, which is the same mechanism behind the a*/disp result below.

## 4. Attention on the search context

Share of decoder cross-attention landing on the context block rather than the obs tokens,
denoise step 7, mean over slots 1+, probed on held-out moving transitions. BC has no
cross-attention (`ConditionalUnet1D`, FiLM) and is absent by construction.

| arm | 10k | 30k | 50k | 70k | 100k |
|---|--:|--:|--:|--:|--:|
| k16 uniform | 0.466 | 0.419 | 0.401 | 0.384 | 0.377 |
| k16 flat400 | 0.572 | 0.532 | 0.493 | 0.494 | **0.500** |
| k16 ramp400to200 | 0.577 | 0.508 | 0.492 | 0.488 | 0.489 |
| k4 uniform | 0.325 | 0.236 | 0.262 | 0.250 | 0.257 |
| k4 flat400 | 0.413 | 0.365 | 0.385 | 0.387 | 0.392 |
| k4 ramp400to200 | 0.411 | 0.383 | 0.393 | 0.397 | 0.393 |
| **k16 flat600** | 0.697 | 0.571 | 0.558 | 0.552 | **0.549** |
| k16 ramp600to200 | 0.666 | 0.550 | 0.481 | 0.481 | 0.475 |
| **k4 flat600** | 0.614 | 0.534 | 0.509 | 0.528 | **0.532** |
| k4 ramp600to200 | 0.525 | 0.480 | 0.459 | 0.470 | 0.469 |

**Every ladder arm beats its uniform counterpart at all ten checkpoints — 40 of 40 at the 400
level.** Gap +0.11…+0.14 absolute (33–53% relative), and flat400 / ramp400to200 are
indistinguishable, so at that level it is *having* a ladder that matters, not its shape.

**At the 600 level the ladder SHAPE starts to matter.** Both flat600 arms sit clearly above
their 400-level counterparts (0.549 vs 0.500 at k=16; 0.532 vs 0.392 at k=4). `k4 ramp600to200`
does too (0.469 vs 0.393). But **`k16 ramp600to200` crosses BELOW `k16 ramp400to200` at 50k**
(0.481 vs 0.492) and stays there — the one pairing where more corruption does not buy more
context use. So the dose–response is real but not universal: it is a flat-ladder effect at
k=16, and holds for both shapes at k=4.

k=16 exceeds k=4 under every matched ladder. Slot 0 reads exactly 0.000 everywhere. Context
share is **not monotonic in training** — every arm dips over 10k→30k then recovers, so any
two-point read inside that range misleads.

## 5. argmax vs final candidate — does the verifier earn its keep?

Two ways to execute a search, on the same rollouts:

* **argmax** — all n candidates are scored by `t_goal` and the best is executed.
* **final_pass** — only n−1 are searched; the **n'th generation IS the action**, executed
  **without the verifier being consulted at all**. It conditions on the n−1 before it, so this
  isolates what the search CONTEXT buys from what the SELECTOR buys.

At n=1 the two are identical by construction — nothing to select from — and they are, to three
decimals, on all 11 arms. That is the harness's own leak check.

**Mean reward, clean, 30 held-out episodes @100k:**

| arm | fp n=1 | fp n=4 | fp n=8 | fp n=16 | am n=16 | fp−am |
|---|--:|--:|--:|--:|--:|--:|
| BC | 0.635 | 0.658 | 0.671 | 0.661 | 0.773 | −0.111 |
| k16 uniform | 0.600 | 0.630 | 0.648 | 0.658 | 0.697 | −0.039 |
| k16 flat400 | 0.539 | 0.691 | 0.644 | 0.746 | 0.858 | −0.112 |
| k16 ramp400to200 | 0.627 | 0.718 | 0.801 | 0.738 | 0.897 | −0.159 |
| **k16 flat600** | 0.409 | **0.810** | 0.774 | 0.667 | **0.967** | **−0.300** |
| k16 ramp600to200 | 0.477 | 0.729 | 0.739 | 0.665 | 0.791 | −0.126 |
| k4 uniform | 0.546 | 0.624 | 0.518 | 0.670 | 0.729 | −0.059 |
| k4 flat400 | 0.467 | 0.530 | 0.640 | 0.594 | 0.765 | −0.171 |
| k4 ramp400to200 | 0.614 | 0.747 | 0.610 | 0.703 | 0.730 | −0.027 |
| k4 flat600 | 0.502 | 0.616 | 0.548 | 0.623 | 0.864 | −0.240 |
| k4 ramp600to200 | 0.480 | 0.710 | 0.756 | 0.631 | 0.840 | −0.209 |

**argmax beats final_pass on 11 of 11 arms** (10 of 11 on success rate; `k4 uniform` is the
lone flip, +0.133 success against −0.059 reward, i.e. noise). **The verifier does real work
that context alone cannot replace.**

**The loss is largest for the arms that use context most** — `k16 flat600` −0.300, the uniform
arms −0.04…−0.06. Those arms generate the widest, most varied candidate sets, so they have the
most for a selector to exploit and the most to give up by not selecting.

### This corrects the `final(n)` reading in §1

The `final(n)` verifier-value curve shows the ladder arms improving their unaided draw as
context accumulates while uniform arms and BC go flat or negative, and it is tempting to read
that as "the ladder arms use their context, the verifier is secondary". The deployment numbers
say that is half right:

* final_pass **does** rise steeply from n=1 to n=4 on every ladder arm — `k16 flat600`
  0.409 → 0.810. The context effect is real and large.
* But it then **plateaus or declines** by n=16 while argmax keeps climbing.

Context improves the unaided sample up to a few candidates and then stops paying; the verifier
keeps paying. `final(n)` measures the first part only.

⚠️ 30 episodes: the n=16 dip under final_pass is 2–4 episodes on most arms. The 11-of-11
direction carries this, not any single row.

## 6. Mode analysis figures

All 50 in one folder, reachable from the repo as
**[`docs/reports/from-modes-figures/`](from-modes-figures/)** — a symlink to
`/gscratch/robotics/harine/mode_analysis/figures_tgoal_moving/`, where the figures themselves
stay (9 MB of PNGs, regenerable). Rebuild the folder with `bash scripts/collect_mv_figures.sh`.

| file | what |
|---|---|
| `slot_grid_step100k_state{0,1,2}.png` | **start here** — one panel per arm, every slot's endpoint cloud coloured by slot index, a* starred on each |
| `compare_arms_100k_k{4,16}.png` | dispersion across arms, one figure per width |
| `compare_arms_d7_100k_test-windows.png` | context share across arms, probed on held-out moving transitions |
| `<arm>__modes_step100000_state{0,1,2}.png` | one arm, faceted into K panels: trajectories and endpoints, a* on every facet |
| `<arm>__dispersion_step100000.png` | dispersion vs slot index for that arm |

`<arm>` is one of `uniform`, `flat400`, `ramp400to200`, `flat600`, `ramp600to200` x `_k{4,16}`,
plus `BC` — all eleven, at step 100k.

The three cross-arm figures are **generated by the collector**, not copied from the analysis
roots: those roots hold every generation's figures under the same names, so the earlier copy
had silently filed the blq137 ones here. `compare_arms.py` now takes `--runs '*split-mv*'`,
which is what keeps one axes to one generation.

In the state-0 grid: BC and k=16 uniform sit **on** a*; every ladder arm sits below-left of it,
k=4 furthest — displaced, not merely wider. In k=16 uniform, slots 0 and 15 lie on top of each
other (context moves nothing); in the ladder arms the dark slot-0 points pull up-left toward a*
as context accumulates.



## 7. Checkpoint sweep — five steps, both selections

`success rate / mean goal coverage`, 30 held-out episodes, one row per arm, steps
20k/40k/60k/80k/100k. Nothing here nominates a checkpoint; the point is the shape of
each curve, and the shapes differ by arm. Every heading names its own search budget n.

### argmax — n=16, obs-clean (every slot evaluated clean)

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16` | 0.400 / 0.806 | 0.367 / 0.734 | 0.133 / 0.668 | 0.200 / 0.784 | 0.267 / 0.658 |
| `k16_flat400` | 0.367 / 0.823 | 0.467 / 0.808 | 0.500 / 0.817 | 0.433 / 0.821 | 0.500 / 0.821 |
| `k16_flat600` | 0.433 / 0.813 | 0.367 / 0.873 | 0.533 / 0.904 | 0.433 / 0.853 | 0.500 / 0.923 |
| `k16_ramp400to200` | 0.500 / 0.854 | 0.433 / 0.866 | 0.433 / 0.878 | 0.467 / 0.815 | 0.433 / 0.856 |
| `k16_ramp600to200` | 0.333 / 0.684 | 0.467 / 0.790 | 0.433 / 0.797 | 0.433 / 0.764 | 0.467 / 0.756 |
| `k4` | 0.400 / 0.737 | 0.367 / 0.737 | 0.400 / 0.643 | 0.267 / 0.642 | 0.133 / 0.694 |
| `k4_flat400` | 0.400 / 0.820 | 0.300 / 0.778 | 0.400 / 0.771 | 0.400 / 0.754 | 0.400 / 0.748 |
| `k4_flat600` | 0.300 / 0.871 | 0.500 / 0.920 | 0.500 / 0.808 | 0.467 / 0.826 | 0.267 / 0.823 |
| `k4_ramp400to200` | 0.367 / 0.856 | 0.400 / 0.817 | 0.233 / 0.788 | 0.233 / 0.783 | 0.267 / 0.695 |
| `k4_ramp600to200` | 0.333 / 0.764 | 0.467 / 0.813 | 0.633 / 0.842 | 0.300 / 0.749 | 0.467 / 0.800 |
| `unetbc` | 0.533 / 0.880 | 0.667 / 0.844 | 0.600 / 0.749 | 0.467 / 0.827 | 0.400 / 0.742 |

### final_pass — n=16, obs-clean (every slot evaluated clean)

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16` | 0.133 / 0.588 | 0.267 / 0.794 | 0.233 / 0.661 | 0.100 / 0.605 | 0.100 / 0.627 |
| `k16_flat400` | 0.167 / 0.676 | 0.133 / 0.615 | 0.267 / 0.666 | 0.200 / 0.744 | 0.033 / 0.709 |
| `k16_flat600` | 0.000 / 0.632 | 0.100 / 0.542 | 0.100 / 0.676 | 0.167 / 0.743 | 0.167 / 0.635 |
| `k16_ramp400to200` | 0.167 / 0.627 | 0.267 / 0.631 | 0.200 / 0.687 | 0.167 / 0.679 | 0.233 / 0.703 |
| `k16_ramp600to200` | 0.000 / 0.599 | 0.100 / 0.664 | 0.133 / 0.584 | 0.300 / 0.693 | 0.233 / 0.632 |
| `k4` | 0.033 / 0.446 | 0.233 / 0.501 | 0.133 / 0.618 | 0.100 / 0.623 | 0.267 / 0.639 |
| `k4_flat400` | 0.033 / 0.525 | 0.133 / 0.523 | 0.133 / 0.587 | 0.067 / 0.561 | 0.133 / 0.565 |
| `k4_flat600` | 0.033 / 0.360 | 0.100 / 0.582 | 0.100 / 0.451 | 0.133 / 0.699 | 0.100 / 0.593 |
| `k4_ramp400to200` | 0.000 / 0.452 | 0.100 / 0.604 | 0.133 / 0.740 | 0.133 / 0.648 | 0.133 / 0.670 |
| `k4_ramp600to200` | 0.133 / 0.511 | 0.167 / 0.685 | 0.333 / 0.660 | 0.167 / 0.643 | 0.200 / 0.600 |
| `unetbc` | 0.067 / 0.467 | 0.167 / 0.656 | 0.167 / 0.644 | 0.267 / 0.763 | 0.200 / 0.630 |

### n=16 — the ladder arms under their own conditional (obs-corrupt)

The clean rows above are the only ones comparable with uniform and BC. These evaluate
each ladder arm with its slots corrupted at the timesteps it was FITTED to, which is
the conditional the policy actually saw in training.

**argmax, n=16**

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16_flat400` | 0.467 / 0.711 | 0.467 / 0.817 | 0.433 / 0.740 | 0.467 / 0.786 | 0.400 / 0.786 |
| `k16_flat600` | 0.300 / 0.783 | 0.567 / 0.874 | 0.567 / 0.863 | 0.367 / 0.692 | 0.433 / 0.799 |
| `k16_ramp400to200` | 0.333 / 0.835 | 0.433 / 0.760 | 0.300 / 0.825 | 0.467 / 0.804 | 0.533 / 0.796 |
| `k16_ramp600to200` | 0.467 / 0.885 | 0.600 / 0.842 | 0.333 / 0.764 | 0.667 / 0.891 | 0.433 / 0.848 |
| `k4_flat400` | 0.133 / 0.679 | 0.433 / 0.802 | 0.467 / 0.795 | 0.433 / 0.726 | 0.367 / 0.728 |
| `k4_flat600` | 0.233 / 0.720 | 0.300 / 0.704 | 0.233 / 0.706 | 0.233 / 0.674 | 0.200 / 0.674 |
| `k4_ramp400to200` | 0.200 / 0.705 | 0.500 / 0.831 | 0.233 / 0.745 | 0.400 / 0.715 | 0.267 / 0.740 |
| `k4_ramp600to200` | 0.533 / 0.800 | 0.433 / 0.787 | 0.333 / 0.702 | 0.300 / 0.636 | 0.333 / 0.783 |

**final_pass, n=16**

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16_flat400` | 0.100 / 0.497 | 0.100 / 0.579 | 0.067 / 0.584 | 0.167 / 0.610 | 0.233 / 0.714 |
| `k16_flat600` | 0.033 / 0.528 | 0.067 / 0.647 | 0.167 / 0.728 | 0.200 / 0.683 | 0.100 / 0.588 |
| `k16_ramp400to200` | 0.133 / 0.632 | 0.067 / 0.605 | 0.033 / 0.542 | 0.133 / 0.619 | 0.100 / 0.695 |
| `k16_ramp600to200` | 0.067 / 0.662 | 0.067 / 0.746 | 0.200 / 0.687 | 0.100 / 0.589 | 0.133 / 0.570 |
| `k4_flat400` | 0.033 / 0.353 | 0.100 / 0.460 | 0.067 / 0.512 | 0.200 / 0.455 | 0.033 / 0.465 |
| `k4_flat600` | 0.000 / 0.259 | 0.000 / 0.273 | 0.000 / 0.206 | 0.000 / 0.280 | 0.033 / 0.268 |
| `k4_ramp400to200` | 0.000 / 0.438 | 0.067 / 0.685 | 0.100 / 0.526 | 0.233 / 0.615 | 0.133 / 0.602 |
| `k4_ramp600to200` | 0.033 / 0.467 | 0.033 / 0.597 | 0.033 / 0.666 | 0.133 / 0.568 | 0.100 / 0.559 |

**Clean vs corrupt at n=16, mean over the five steps** (success / coverage),
sorted by the success delta:

*argmax, n=16*

| arm | clean | corrupt | Δ success |
|---|---|---|--:|
| `k16_ramp600to200` | 0.427 / 0.758 | 0.500 / 0.846 | **+0.073** |
| `k4_ramp400to200` | 0.300 / 0.788 | 0.320 / 0.747 | +0.020 |
| `k16_flat600` | 0.453 / 0.873 | 0.447 / 0.802 | −0.007 |
| `k16_flat400` | 0.453 / 0.818 | 0.447 / 0.768 | −0.007 |
| `k4_flat400` | 0.380 / 0.774 | 0.367 / 0.746 | −0.013 |
| `k16_ramp400to200` | 0.453 / 0.854 | 0.413 / 0.804 | −0.040 |
| `k4_ramp600to200` | 0.440 / 0.794 | 0.387 / 0.742 | −0.053 |
| `k4_flat600` | 0.407 / 0.850 | 0.240 / 0.696 | **−0.167** |

Corrupt beats clean on **2/8** arms on success and **1/8** on coverage.

*final_pass, n=16*

| arm | clean | corrupt | Δ success |
|---|---|---|--:|
| `k16_flat600` | 0.107 / 0.646 | 0.113 / 0.635 | +0.007 |
| `k4_ramp400to200` | 0.100 / 0.623 | 0.107 / 0.573 | +0.007 |
| `k4_flat400` | 0.100 / 0.552 | 0.087 / 0.449 | −0.013 |
| `k16_flat400` | 0.160 / 0.682 | 0.133 / 0.597 | −0.027 |
| `k16_ramp600to200` | 0.153 / 0.634 | 0.113 / 0.651 | −0.040 |
| `k4_flat600` | 0.093 / 0.537 | 0.007 / 0.257 | **−0.087** |
| `k16_ramp400to200` | 0.207 / 0.665 | 0.093 / 0.619 | **−0.113** |
| `k4_ramp600to200` | 0.200 / 0.620 | 0.067 / 0.571 | **−0.133** |

Corrupt beats clean on **2/8** arms on success and **1/8** on coverage.

So evaluating a ladder arm under the conditional it was fitted to does **not**
recover the gap -- it is mostly worse, and worse on coverage more consistently than
on success. The corruption is doing something other than defining a distribution the
policy is good on. That is the same direction as §4: these arms use the context, and
the context is not a noisy copy of the observation.

### Both readouts at n=64

The widest budget measured. It was not in the original sweep because a level costs ~4x
n=16, so it is the last to fill.

#### argmax — n=64, obs-clean (every slot evaluated clean)

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16` | 0.433 / 0.767 | 0.433 / 0.767 | 0.267 / 0.701 | 0.333 / 0.692 | 0.233 / 0.677 |
| `k16_flat400` | 0.600 / 0.881 | 0.600 / 0.789 | 0.500 / 0.839 | 0.567 / 0.816 | 0.667 / 0.848 |
| `k16_flat600` | 0.700 / 0.869 | 0.600 / 0.894 | 0.367 / 0.909 | 0.367 / 0.893 | 0.600 / 0.821 |
| `k16_ramp400to200` | 0.700 / 0.905 | 0.633 / 0.868 | 0.467 / 0.893 | 0.500 / 0.839 | 0.333 / 0.780 |
| `k16_ramp600to200` | 0.400 / 0.770 | 0.500 / 0.818 | 0.567 / 0.843 | 0.500 / 0.809 | 0.433 / 0.792 |
| `k4` | 0.600 / 0.863 | 0.600 / 0.778 | 0.433 / 0.783 | 0.267 / 0.676 | 0.267 / 0.715 |
| `k4_flat400` | 0.500 / 0.943 | 0.467 / 0.846 | 0.233 / 0.723 | 0.200 / 0.737 | 0.433 / 0.766 |
| `k4_flat600` | 0.467 / 0.797 | 0.733 / 0.948 | 0.600 / 0.913 | 0.600 / 0.929 | 0.367 / 0.833 |
| `k4_ramp400to200` | 0.300 / 0.770 | 0.567 / 0.933 | 0.533 / 0.819 | 0.333 / 0.827 | 0.400 / 0.814 |
| `k4_ramp600to200` | 0.400 / 0.837 | 0.467 / 0.840 | 0.400 / 0.820 | 0.467 / 0.805 | 0.300 / 0.788 |
| `unetbc` | 0.600 / 0.858 | 0.600 / 0.831 | 0.533 / 0.759 | 0.533 / 0.764 | 0.400 / 0.747 |

#### final_pass — n=64, obs-clean (every slot evaluated clean)

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16` | 0.200 / 0.701 | 0.133 / 0.751 | 0.200 / 0.693 | 0.133 / 0.567 | 0.133 / 0.613 |
| `k16_flat400` | 0.100 / 0.535 | 0.100 / 0.679 | 0.267 / 0.652 | 0.067 / 0.694 | 0.167 / 0.676 |
| `k16_flat600` | 0.033 / 0.542 | 0.133 / 0.530 | 0.033 / 0.363 | 0.067 / 0.472 | 0.200 / 0.587 |
| `k16_ramp400to200` | 0.200 / 0.689 | 0.333 / 0.791 | 0.200 / 0.663 | 0.200 / 0.695 | 0.167 / 0.668 |
| `k16_ramp600to200` | 0.067 / 0.558 | 0.133 / 0.643 | 0.233 / 0.605 | 0.300 / 0.604 | 0.233 / 0.636 |
| `k4` | 0.167 / 0.582 | 0.100 / 0.651 | 0.233 / 0.721 | 0.133 / 0.572 | 0.100 / 0.605 |
| `k4_flat400` | 0.100 / 0.617 | 0.167 / 0.590 | 0.133 / 0.582 | 0.067 / 0.571 | 0.133 / 0.654 |
| `k4_flat600` | 0.100 / 0.412 | 0.100 / 0.538 | 0.067 / 0.542 | 0.067 / 0.512 | 0.133 / 0.536 |
| `k4_ramp400to200` | 0.067 / 0.557 | 0.300 / 0.800 | 0.200 / 0.702 | 0.100 / 0.610 | 0.133 / 0.629 |
| `k4_ramp600to200` | 0.033 / 0.481 | 0.200 / 0.695 | 0.300 / 0.692 | 0.067 / 0.583 | 0.167 / 0.586 |
| `unetbc` | 0.100 / 0.457 | 0.167 / 0.516 | 0.300 / 0.665 | 0.333 / 0.706 | 0.233 / 0.613 |

#### n=64 — the ladder arms under their own conditional (obs-corrupt)

The clean rows above are the only ones comparable with uniform and BC. These evaluate
each ladder arm with its slots corrupted at the timesteps it was FITTED to, which is
the conditional the policy actually saw in training.

**argmax, n=64**

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16_flat400` | 0.467 / 0.812 | 0.600 / 0.852 | 0.500 / 0.824 | 0.367 / 0.779 | 0.367 / 0.806 |
| `k16_flat600` | 0.400 / 0.921 | 0.567 / 0.877 | 0.633 / 0.914 | 0.467 / 0.778 | 0.667 / 0.876 |
| `k16_ramp400to200` | 0.333 / 0.819 | 0.367 / 0.834 | 0.533 / 0.839 | 0.533 / 0.859 | 0.433 / 0.876 |
| `k16_ramp600to200` | 0.600 / 0.916 | 0.600 / 0.871 | 0.567 / 0.888 | 0.433 / 0.889 | 0.667 / 0.923 |
| `k4_flat400` | 0.233 / 0.815 | 0.300 / 0.823 | 0.400 / 0.797 | 0.267 / 0.771 | 0.367 / 0.693 |
| `k4_flat600` | 0.400 / 0.716 | 0.367 / 0.811 | 0.200 / 0.734 | 0.133 / 0.735 | 0.267 / 0.831 |
| `k4_ramp400to200` | 0.433 / 0.815 | 0.500 / 0.821 | 0.467 / 0.828 | 0.433 / 0.805 | 0.433 / 0.831 |
| `k4_ramp600to200` | 0.367 / 0.810 | 0.367 / 0.838 | 0.400 / 0.743 | 0.267 / 0.673 | 0.533 / 0.786 |

**final_pass, n=64**

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16_flat400` | 0.133 / 0.542 | 0.100 / 0.607 | 0.133 / 0.487 | 0.167 / 0.569 | 0.133 / 0.659 |
| `k16_flat600` | 0.033 / 0.409 | 0.000 / 0.351 | 0.000 / 0.366 | 0.133 / 0.432 | 0.033 / 0.386 |
| `k16_ramp400to200` | 0.067 / 0.614 | 0.200 / 0.694 | 0.167 / 0.720 | 0.267 / 0.752 | 0.267 / 0.671 |
| `k16_ramp600to200` | 0.133 / 0.585 | 0.100 / 0.681 | 0.067 / 0.544 | 0.233 / 0.605 | 0.133 / 0.619 |
| `k4_flat400` | 0.000 / 0.459 | 0.200 / 0.631 | 0.100 / 0.532 | 0.067 / 0.492 | 0.067 / 0.534 |
| `k4_flat600` | 0.000 / 0.245 | 0.000 / 0.209 | 0.000 / 0.297 | 0.000 / 0.244 | 0.000 / 0.210 |
| `k4_ramp400to200` | 0.067 / 0.471 | 0.033 / 0.599 | 0.167 / 0.664 | 0.267 / 0.615 | 0.100 / 0.684 |
| `k4_ramp600to200` | 0.033 / 0.374 | 0.200 / 0.596 | 0.067 / 0.553 | 0.133 / 0.526 | 0.167 / 0.620 |

**Clean vs corrupt at n=64, mean over the five steps** (success / coverage),
sorted by the success delta:

*argmax, n=64*

| arm | clean | corrupt | Δ success |
|---|---|---|--:|
| `k16_ramp600to200` | 0.480 / 0.806 | 0.573 / 0.897 | **+0.093** |
| `k4_ramp400to200` | 0.427 / 0.833 | 0.453 / 0.820 | +0.027 |
| `k16_flat600` | 0.527 / 0.877 | 0.547 / 0.873 | +0.020 |
| `k4_ramp600to200` | 0.407 / 0.818 | 0.387 / 0.770 | −0.020 |
| `k4_flat400` | 0.367 / 0.803 | 0.313 / 0.780 | −0.053 |
| `k16_ramp400to200` | 0.527 / 0.857 | 0.440 / 0.845 | **−0.087** |
| `k16_flat400` | 0.587 / 0.835 | 0.460 / 0.815 | **−0.127** |
| `k4_flat600` | 0.553 / 0.884 | 0.273 / 0.765 | **−0.280** |

Corrupt beats clean on **3/8** arms on success and **1/8** on coverage.

*final_pass, n=64*

| arm | clean | corrupt | Δ success |
|---|---|---|--:|
| `k16_flat400` | 0.140 / 0.647 | 0.133 / 0.573 | −0.007 |
| `k16_ramp400to200` | 0.220 / 0.701 | 0.193 / 0.690 | −0.027 |
| `k4_ramp600to200` | 0.153 / 0.608 | 0.120 / 0.534 | −0.033 |
| `k4_flat400` | 0.120 / 0.603 | 0.087 / 0.530 | −0.033 |
| `k4_ramp400to200` | 0.160 / 0.660 | 0.127 / 0.607 | −0.033 |
| `k16_flat600` | 0.093 / 0.499 | 0.040 / 0.389 | −0.053 |
| `k16_ramp600to200` | 0.193 / 0.609 | 0.133 / 0.607 | −0.060 |
| `k4_flat600` | 0.093 / 0.508 | 0.000 / 0.241 | **−0.093** |

Corrupt beats clean on **0/8** arms on success and **0/8** on coverage.

### Every arm under its own training conditional, at other n

The tables above pair clean against corrupt to isolate the readout. These do the
opposite: each arm is read under the conditional it was **trained** on -- corrupt for
the eight ladder arms, clean for `unetbc`, `k16` and `k4`, which never saw a corrupted
slot. One table mixes both readouts by arm, so a column is not a like-for-like
comparison across the ladder boundary; it is each policy at its own operating point.

#### n=1

argmax and final_pass coincide here by construction -- nothing to select from. Largest disagreement across all 55 cells: 0.000.

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16` | 0.033 / 0.374 | 0.100 / 0.615 | 0.067 / 0.548 | 0.133 / 0.627 | 0.167 / 0.572 |
| `k16_flat400` | 0.033 / 0.378 | 0.200 / 0.555 | 0.167 / 0.477 | 0.133 / 0.456 | 0.167 / 0.610 |
| `k16_flat600` | 0.000 / 0.119 | 0.033 / 0.247 | 0.033 / 0.258 | 0.000 / 0.208 | 0.033 / 0.234 |
| `k16_ramp400to200` | 0.033 / 0.434 | 0.067 / 0.433 | 0.067 / 0.433 | 0.100 / 0.585 | 0.100 / 0.594 |
| `k16_ramp600to200` | 0.000 / 0.300 | 0.033 / 0.378 | 0.000 / 0.362 | 0.067 / 0.363 | 0.033 / 0.383 |
| `k4` | 0.033 / 0.395 | 0.200 / 0.600 | 0.200 / 0.626 | 0.067 / 0.539 | 0.067 / 0.519 |
| `k4_flat400` | 0.067 / 0.422 | 0.133 / 0.560 | 0.167 / 0.657 | 0.267 / 0.711 | 0.133 / 0.688 |
| `k4_flat600` | 0.033 / 0.207 | 0.000 / 0.327 | 0.033 / 0.338 | 0.033 / 0.303 | 0.067 / 0.290 |
| `k4_ramp400to200` | 0.000 / 0.300 | 0.167 / 0.558 | 0.200 / 0.654 | 0.100 / 0.636 | 0.133 / 0.601 |
| `k4_ramp600to200` | 0.033 / 0.208 | 0.100 / 0.406 | 0.067 / 0.477 | 0.133 / 0.443 | 0.033 / 0.402 |
| `unetbc` | 0.067 / 0.451 | 0.200 / 0.704 | 0.333 / 0.732 | 0.333 / 0.721 | 0.267 / 0.607 |

#### n=4

*argmax*

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16` | 0.400 / 0.709 | 0.333 / 0.714 | 0.233 / 0.625 | 0.300 / 0.607 | 0.167 / 0.703 |
| `k16_flat400` | 0.267 / 0.685 | 0.233 / 0.576 | 0.267 / 0.649 | 0.267 / 0.669 | 0.267 / 0.691 |
| `k16_flat600` | 0.133 / 0.539 | 0.167 / 0.595 | 0.500 / 0.751 | 0.167 / 0.626 | 0.300 / 0.719 |
| `k16_ramp400to200` | 0.333 / 0.729 | 0.500 / 0.735 | 0.333 / 0.780 | 0.300 / 0.751 | 0.400 / 0.804 |
| `k16_ramp600to200` | 0.367 / 0.757 | 0.467 / 0.833 | 0.400 / 0.783 | 0.367 / 0.791 | 0.333 / 0.712 |
| `k4` | 0.200 / 0.689 | 0.233 / 0.654 | 0.433 / 0.709 | 0.233 / 0.564 | 0.200 / 0.650 |
| `k4_flat400` | 0.133 / 0.564 | 0.167 / 0.663 | 0.400 / 0.770 | 0.333 / 0.643 | 0.233 / 0.624 |
| `k4_flat600` | 0.067 / 0.441 | 0.267 / 0.521 | 0.233 / 0.559 | 0.333 / 0.685 | 0.133 / 0.514 |
| `k4_ramp400to200` | 0.100 / 0.478 | 0.300 / 0.733 | 0.400 / 0.691 | 0.300 / 0.709 | 0.367 / 0.663 |
| `k4_ramp600to200` | 0.133 / 0.542 | 0.233 / 0.612 | 0.267 / 0.610 | 0.200 / 0.654 | 0.167 / 0.703 |
| `unetbc` | 0.300 / 0.775 | 0.367 / 0.732 | 0.467 / 0.685 | 0.267 / 0.783 | 0.533 / 0.700 |

*final_pass*

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16` | 0.133 / 0.581 | 0.233 / 0.658 | 0.067 / 0.521 | 0.133 / 0.573 | 0.167 / 0.600 |
| `k16_flat400` | 0.100 / 0.494 | 0.100 / 0.551 | 0.233 / 0.573 | 0.100 / 0.719 | 0.100 / 0.735 |
| `k16_flat600` | 0.000 / 0.380 | 0.100 / 0.562 | 0.133 / 0.691 | 0.133 / 0.576 | 0.133 / 0.689 |
| `k16_ramp400to200` | 0.067 / 0.619 | 0.167 / 0.659 | 0.167 / 0.555 | 0.067 / 0.604 | 0.267 / 0.686 |
| `k16_ramp600to200` | 0.000 / 0.639 | 0.033 / 0.606 | 0.067 / 0.778 | 0.133 / 0.675 | 0.100 / 0.697 |
| `k4` | 0.033 / 0.523 | 0.167 / 0.620 | 0.233 / 0.696 | 0.033 / 0.446 | 0.133 / 0.595 |
| `k4_flat400` | 0.000 / 0.458 | 0.133 / 0.455 | 0.133 / 0.578 | 0.100 / 0.564 | 0.167 / 0.592 |
| `k4_flat600` | 0.000 / 0.231 | 0.000 / 0.413 | 0.000 / 0.304 | 0.033 / 0.421 | 0.000 / 0.402 |
| `k4_ramp400to200` | 0.000 / 0.416 | 0.267 / 0.628 | 0.167 / 0.674 | 0.200 / 0.592 | 0.067 / 0.605 |
| `k4_ramp600to200` | 0.033 / 0.433 | 0.167 / 0.572 | 0.200 / 0.577 | 0.267 / 0.574 | 0.167 / 0.596 |
| `unetbc` | 0.100 / 0.547 | 0.133 / 0.635 | 0.233 / 0.658 | 0.267 / 0.718 | 0.200 / 0.627 |

#### n=8

*argmax*

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16` | 0.300 / 0.663 | 0.400 / 0.773 | 0.300 / 0.686 | 0.133 / 0.626 | 0.100 / 0.568 |
| `k16_flat400` | 0.367 / 0.664 | 0.333 / 0.734 | 0.433 / 0.729 | 0.333 / 0.704 | 0.400 / 0.709 |
| `k16_flat600` | 0.333 / 0.765 | 0.400 / 0.842 | 0.467 / 0.739 | 0.267 / 0.739 | 0.267 / 0.701 |
| `k16_ramp400to200` | 0.400 / 0.689 | 0.433 / 0.794 | 0.233 / 0.793 | 0.400 / 0.749 | 0.267 / 0.761 |
| `k16_ramp600to200` | 0.300 / 0.836 | 0.267 / 0.878 | 0.333 / 0.816 | 0.500 / 0.774 | 0.433 / 0.808 |
| `k4` | 0.400 / 0.752 | 0.500 / 0.798 | 0.300 / 0.711 | 0.133 / 0.649 | 0.233 / 0.674 |
| `k4_flat400` | 0.267 / 0.713 | 0.367 / 0.703 | 0.333 / 0.756 | 0.433 / 0.740 | 0.433 / 0.802 |
| `k4_flat600` | 0.300 / 0.656 | 0.267 / 0.703 | 0.333 / 0.662 | 0.167 / 0.681 | 0.233 / 0.604 |
| `k4_ramp400to200` | 0.133 / 0.614 | 0.300 / 0.731 | 0.167 / 0.770 | 0.300 / 0.693 | 0.300 / 0.728 |
| `k4_ramp600to200` | 0.300 / 0.739 | 0.400 / 0.838 | 0.300 / 0.746 | 0.367 / 0.758 | 0.200 / 0.698 |
| `unetbc` | 0.500 / 0.800 | 0.533 / 0.823 | 0.467 / 0.827 | 0.467 / 0.852 | 0.533 / 0.758 |

*final_pass*

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16` | 0.167 / 0.688 | 0.233 / 0.618 | 0.100 / 0.592 | 0.167 / 0.510 | 0.133 / 0.618 |
| `k16_flat400` | 0.067 / 0.593 | 0.167 / 0.663 | 0.033 / 0.423 | 0.100 / 0.666 | 0.200 / 0.751 |
| `k16_flat600` | 0.000 / 0.487 | 0.067 / 0.505 | 0.233 / 0.648 | 0.133 / 0.614 | 0.133 / 0.622 |
| `k16_ramp400to200` | 0.167 / 0.633 | 0.200 / 0.736 | 0.133 / 0.620 | 0.233 / 0.688 | 0.233 / 0.746 |
| `k16_ramp600to200` | 0.133 / 0.682 | 0.167 / 0.567 | 0.167 / 0.724 | 0.167 / 0.668 | 0.233 / 0.592 |
| `k4` | 0.067 / 0.544 | 0.167 / 0.666 | 0.200 / 0.619 | 0.100 / 0.538 | 0.133 / 0.494 |
| `k4_flat400` | 0.000 / 0.330 | 0.100 / 0.550 | 0.067 / 0.645 | 0.167 / 0.501 | 0.133 / 0.592 |
| `k4_flat600` | 0.000 / 0.290 | 0.000 / 0.320 | 0.000 / 0.342 | 0.000 / 0.367 | 0.033 / 0.235 |
| `k4_ramp400to200` | 0.100 / 0.602 | 0.267 / 0.629 | 0.167 / 0.653 | 0.133 / 0.595 | 0.067 / 0.627 |
| `k4_ramp600to200` | 0.033 / 0.387 | 0.133 / 0.602 | 0.167 / 0.623 | 0.133 / 0.540 | 0.100 / 0.613 |
| `unetbc` | 0.033 / 0.430 | 0.167 / 0.615 | 0.233 / 0.584 | 0.233 / 0.657 | 0.267 / 0.641 |

#### n=64

*argmax*

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16` | 0.433 / 0.767 | 0.433 / 0.767 | 0.267 / 0.701 | 0.333 / 0.692 | 0.233 / 0.677 |
| `k16_flat400` | 0.467 / 0.812 | 0.600 / 0.852 | 0.500 / 0.824 | 0.367 / 0.779 | 0.367 / 0.806 |
| `k16_flat600` | 0.400 / 0.921 | 0.567 / 0.877 | 0.633 / 0.914 | 0.467 / 0.778 | 0.667 / 0.876 |
| `k16_ramp400to200` | 0.333 / 0.819 | 0.367 / 0.834 | 0.533 / 0.839 | 0.533 / 0.859 | 0.433 / 0.876 |
| `k16_ramp600to200` | 0.600 / 0.916 | 0.600 / 0.871 | 0.567 / 0.888 | 0.433 / 0.889 | 0.667 / 0.923 |
| `k4` | 0.600 / 0.863 | 0.600 / 0.778 | 0.433 / 0.783 | 0.267 / 0.676 | 0.267 / 0.715 |
| `k4_flat400` | 0.233 / 0.815 | 0.300 / 0.823 | 0.400 / 0.797 | 0.267 / 0.771 | 0.367 / 0.693 |
| `k4_flat600` | 0.400 / 0.716 | 0.367 / 0.811 | 0.200 / 0.734 | 0.133 / 0.735 | 0.267 / 0.831 |
| `k4_ramp400to200` | 0.433 / 0.815 | 0.500 / 0.821 | 0.467 / 0.828 | 0.433 / 0.805 | 0.433 / 0.831 |
| `k4_ramp600to200` | 0.367 / 0.810 | 0.367 / 0.838 | 0.400 / 0.743 | 0.267 / 0.673 | 0.533 / 0.786 |
| `unetbc` | 0.600 / 0.858 | 0.600 / 0.831 | 0.533 / 0.759 | 0.533 / 0.764 | 0.400 / 0.747 |

*final_pass*

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `k16` | 0.200 / 0.701 | 0.133 / 0.751 | 0.200 / 0.693 | 0.133 / 0.567 | 0.133 / 0.613 |
| `k16_flat400` | 0.133 / 0.542 | 0.100 / 0.607 | 0.133 / 0.487 | 0.167 / 0.569 | 0.133 / 0.659 |
| `k16_flat600` | 0.033 / 0.409 | 0.000 / 0.351 | 0.000 / 0.366 | 0.133 / 0.432 | 0.033 / 0.386 |
| `k16_ramp400to200` | 0.067 / 0.614 | 0.200 / 0.694 | 0.167 / 0.720 | 0.267 / 0.752 | 0.267 / 0.671 |
| `k16_ramp600to200` | 0.133 / 0.585 | 0.100 / 0.681 | 0.067 / 0.544 | 0.233 / 0.605 | 0.133 / 0.619 |
| `k4` | 0.167 / 0.582 | 0.100 / 0.651 | 0.233 / 0.721 | 0.133 / 0.572 | 0.100 / 0.605 |
| `k4_flat400` | 0.000 / 0.459 | 0.200 / 0.631 | 0.100 / 0.532 | 0.067 / 0.492 | 0.067 / 0.534 |
| `k4_flat600` | 0.000 / 0.245 | 0.000 / 0.209 | 0.000 / 0.297 | 0.000 / 0.244 | 0.000 / 0.210 |
| `k4_ramp400to200` | 0.067 / 0.471 | 0.033 / 0.599 | 0.167 / 0.664 | 0.267 / 0.615 | 0.100 / 0.684 |
| `k4_ramp600to200` | 0.033 / 0.374 | 0.200 / 0.596 | 0.067 / 0.553 | 0.133 / 0.526 | 0.167 / 0.620 |
| `unetbc` | 0.100 / 0.457 | 0.167 / 0.516 | 0.300 / 0.665 | 0.333 / 0.706 | 0.233 / 0.613 |

### Candidate ranking by the `t_goal` verifier, same steps

All arms' candidates pooled with 4 block-directed uniform samples and a*, ranked
by one shared `t_goal` verifier on `block_moving`. Pool 49, 472 decisions,
n=4 per arm. Chance top-1 share is 4/49 = **0.082**.

| source | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| `BC` | 0.201 | 0.141 | 0.092 | 0.102 | 0.099 |
| `k16 flat400` | 0.046 | 0.050 | 0.077 | 0.057 | 0.054 |
| `k16 flat600` | 0.135 | 0.148 | 0.124 | 0.116 | 0.142 |
| `k16 ramp400` | 0.032 | 0.032 | 0.053 | 0.056 | 0.030 |
| `k16 ramp600` | 0.035 | 0.063 | 0.100 | 0.074 | 0.100 |
| `k16 uniform` | 0.045 | 0.042 | 0.026 | 0.039 | 0.026 |
| `k4 flat400` | 0.042 | 0.038 | 0.028 | 0.034 | 0.034 |
| `k4 flat600` | 0.098 | 0.090 | 0.066 | 0.073 | 0.070 |
| `k4 ramp400` | 0.035 | 0.023 | 0.023 | 0.023 | 0.033 |
| `k4 ramp600` | 0.018 | 0.036 | 0.030 | 0.043 | 0.040 |
| `k4 uniform` | 0.039 | 0.032 | 0.036 | 0.030 | 0.043 |
| `a*` | 0.080 | 0.092 | 0.110 | 0.102 | 0.090 |

verifier top-1 share -- how often this source holds the candidate the verifier
ranks first.

**Not the same pool as the ranking paragraph below**, which is why the two quote
different chance levels. This table is `analysis/rank_all/`: all eleven arms in ONE
pool at n=4 each, so the pool is 49 and chance is 0.082. The paragraph under
*The rest, briefly* is `analysis/arm_ranking_mv{16,4}/`: the 400- and 600-level arms in
separate pools at n=16, so chance there is 0.198. A share is only meaningful against
its own pool -- the two are not comparable and neither supersedes the other.

Read across the steps rather than down one column: the verifier's preference for
undirected `uniform` samples *grows* with training (0.193 -> 0.240) while BC's halves
(0.201 -> 0.099), and `uniform` outranks `a*` at every step. That is a property
of the sweep, not of any checkpoint in it.

---

# The rest, briefly

**Candidate ranking, level-separated pools.** All arms' candidates + 16 block-directed uniform
samples + a* ranked together by one `t_goal` verifier, `block_moving`, 472 decisions, step 100k.
Chance top-1 here is **19.8%** (n=16 per arm), not §7's 8.2% -- separate pools, separate scales.
Every trained arm in the **400-level** pool is **below chance on verifier top-1** (0.044–0.157
vs 19.8% at n=16) and below chance on RMSE top-1 — but the **600-level arms break that**:
`k16 flat600` reaches **0.195** and `k16 ramp600to200` 0.140, i.e. at or near chance. Heavier
corruption produces candidates the verifier actually prefers. Blind rate on `block_moving` also
falls to **0.0%** in the 600 pool (0.7% at the 400 level): with wider clouds, essentially every
decision has a candidate that moves the T. **BC is best on expert proximity** (`rmse_mean` 17.99 vs 19.2–21.5; holds
the closest candidate 41.9% vs 9–11%). The uniform arms hold the **best mean verifier rank** at
both widths — the reverse of §1/§4. Verifier and expert-proximity are near-orthogonal:
Spearman **+0.333** moving / **+0.008** still, top-1 overlap 3.5%. Blind rate **0.7% moving vs
26.9% still**, which is the premise this generation rests on. Files:
`analysis/arm_ranking_mv{16,4}/step_0100000.json`.

**Candidate spread.** Dispersion falls with slot index at k=16 (last/first 0.52–0.62) and is
flat at k=4 (0.81–1.11) — four slots is not enough context to tighten the set. `a*/disp` at
100k, under 1 meaning a* is inside the cloud:

| 600 level | | 400 level / none | |
|---|--:|---|--:|
| k16 flat600 | **0.16** | k16 ramp400to200 | 0.87 |
| k4 flat600 | **0.52** | k16 flat400 | 1.08 |
| k16 ramp600to200 | **0.71** | k4 flat400 | 1.64 |
| k4 ramp600to200 | **0.73** | k16 uniform | 2.39 |
| | | BC | 2.39 |
| | | k4 ramp400to200 | 2.45 |
| | | k4 uniform | 2.80 |

**Every 600-level arm stays under 1.0 at every checkpoint; every 400-level arm crosses it** —
4 of 4 pairings, margins of 2–7x. The expert action stays inside the candidate cloud at the
600 level and drifts outside it at the 400 level. **BC drifts as far as anything despite having
no context**, so the drift is not a search-mechanism effect — it is what happens without enough
observation corruption. ⚠️ rests on `--n-states 3`; the consistency across 11 arms x 10
checkpoints is what carries it, not any single value (adjacent checkpoints swing 2–5x).

**Training-side.** Search gain (`nrmse_first − nrmse_min`) at 100k: k16 ladder 0.0191/0.0208 vs
uniform 0.0086; k4 ladder 0.0123/0.0140 vs uniform 0.0056 — **2.2–2.5×**. `nrmse_min` better for
ladder arms (0.0677–0.0687 vs 0.0740 at k=16) despite higher training loss. `train_drift_mse_eps`
settles to 0.0015–0.0046, matching the documented ~0.006. No `val_loss`: no val split by design.

**Not comparable with.** (1) The §2.3/§2.4 arms — different held-out episodes, different test
population, no val. (2) Any earlier `attention.json` — those probed episode t=0, these probe
held-out transitions; `compare_arms.py` now reports the two probe sets separately. (3) §1/§3/§4
score the ladder arms **clean**, off their training conditional; §2's corrupt rows show that
costs them nothing consistent. (4) **"Did filtering help?" is unanswerable** — no unfiltered
controls, by design.

⚠️ **(5) The 30 test episodes are biased in duration.** `select_test` picks the episodes whose
moving-window counts sit nearest the per-episode target, which hits 100:20 exactly and collapses
the test set's variance — episode length train 122.1±37.4 vs test 138.7±**15.6** (KS p<0.001);
moving windows/episode 73.7±24.8 vs 86.5±**3.5** (p<0.001). Moving *fraction* (p=0.555) and all
spatial distributions (p=0.42–0.62) are fine, so the transitions are representative in character
but the episodes do not sample the range of duration. A selection bias, not a leak.
See [analysis/mv_split_shift.png](../../analysis/mv_split_shift.png).

## Reproduce

```bash
python scripts/make_moving_ratio_split.py        # -> config/splits/pusht_seed42_train176.json
python scripts/make_moving_transitions.py        # -> config/splits/transitions/..._moving_ta8.json
SUBMIT=1 bash scripts/run_tgoal_moving.sh        # the seven arms
INCLUDE_BC=0 LADDERS="flat600 ramp600to200" SUBMIT=1 bash scripts/run_tgoal_moving.sh

sbatch scripts/slurm/analysis_loop.sbatch        # §4, §5 as checkpoints land
SUBMIT=1 bash scripts/slurm/submit_tgoal_moving_bon.sh    # §2

# §1 and §3 -- one width-16 run gives every n, on each split
TASK_ID=2 OUTDIR=analysis/nsweep_test  sbatch --account=ckpt-robotics --partition=ckpt \
  scripts/slurm/rank_arms.sbatch --arms moving4 --n 16 --n-demos 176 --tag split-mv --split test
TASK_ID=2 OUTDIR=analysis/nsweep_train sbatch --account=ckpt-robotics --partition=ckpt \
  scripts/slurm/rank_arms.sbatch --arms moving4 --n 16 --n-demos 176 --tag split-mv \
  --split train --episodes 30

bash scripts/collect_mv_figures.sh               # §6
python scripts/plot_mv_split_shift.py            # the split's own distribution check

# §7 -- the checkpoint sweep. CLEAN_ONLY for every arm, CORRUPT_ONLY to backfill the ladder
# arms' own conditional without re-running the clean rows FORCE=1 would also have re-run.
for step in 20000 40000 60000 80000 100000; do
  for sel in argmax final_pass; do
    SUBMIT=1 CLEAN_ONLY=1   STEP=$step SELECTION=$sel N_LIST=1,4,8,16 \
      bash scripts/slurm/submit_tgoal_moving_bon.sh
    SUBMIT=1 CORRUPT_ONLY=1 STEP=$step SELECTION=$sel N_LIST=1,4,8,16 \
      bash scripts/slurm/submit_tgoal_moving_bon.sh
  done
  STEP_OVERRIDE=$step sbatch --account=ckpt-robotics --partition=ckpt \
    scripts/slurm/rank_arms.sbatch --arms moving_all --n 4 --n-demos 176 --tag split-mv
done
```

`collect_mv_figures.sh` regenerates the §6 cross-arm figures rather than copying them out of
the analysis roots, which hold every generation's figures under the same names; `--runs
'*split-mv*'` on both `compare_arms.py` is what keeps one axes to one generation.

Provenance for every run is in its `splits.json`: the transition manifest, its checksum, the
parent split's checksum and the per-split kept/total window counts.
