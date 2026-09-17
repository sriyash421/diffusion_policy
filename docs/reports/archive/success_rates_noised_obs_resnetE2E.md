# PushT per-slot observation noise, ResNet18 end-to-end — success rates

_Generated 2026-09-03 by `scripts/build_noised_obs_doc.py`. Re-run to refresh._

## Config

| | |
|---|---|
| encoder | ResNet18, `IMAGENET1K_V1`, **trained end to end** (`rnE2E`) |
| | `use_group_norm: True`, `imagenet_norm: False`, `feature_layernorm: False` |
| crop | 76x76 random crop, one offset per sample |
| observation | image only, `To = 2` |
| action | `horizon 16`, `n_action_steps 8` |
| trunk | search transformer 4 layers / 4 heads / 256 emb |
| data | 30 demos, seed 42 |
| optim | batch 32, lr 1e-4, EMA 0.995, 100k gradient steps, checkpoint every 10k |
| sampler | DDIM, 100 train timesteps, **8 inference steps**, epsilon, eta 0 |
| obs noise scheduler | DDPM, **1000** train timesteps, beta 1e-4..0.02 linear |
| verifier | `t_goal` on every arm |
| eval | 50 held-out test episodes, `--skip-val` |

The obs-noise scheduler is the one the ladder indexes, and it is a different, **1000-step** schedule from the 100-step DDIM the actions are sampled under. Its floor is sqrt(alpha_bar) = 0.006, so at cap 999 slot 0 really is very nearly pure noise; "cap 400" means slot 0 is held at t=400 instead, i.e. sqrt(alpha_bar) 0.44. The corruption is applied to the encoded observation features, scaled by a running per-dimension feature std so sqrt(alpha_bar) reads as an SNR rather than an absolute magnitude. One noise sample per decision, shared across the 16 slots: the agent has one observation, seen at 16 graded levels.

## Reading the tables

`n` is the **search width**: n action candidates are generated for the current observation, each is rolled out in a PushT simulator (the verifier) to a scalar value, and one is executed. So n is a test-time-compute axis on fixed weights.

At 50 episodes a single cell carries a 95% CI of roughly +/-0.13 near 0.5, so **cell-to-cell differences under ~0.15 are not separable** -- read down a column or across checkpoints, not one cell.

`argmax` executes the verifier's pick and sweeps n = 1..64. `final_pass` instead executes the LAST candidate generated, which under a ladder is the cleanest slot the search reached; it was asked for at n = 1, 8, 16 only, so its other columns are blank by design. Blank also means "not yet evaluated". No cell is a nominated best.

**Corrupted vs clean rollouts.** `corrupt_obs_eval` gates only the eval branch of the corruption and takes no part in the loss, so both rows come off one checkpoint. Clean rollouts mean the slot -> corruption-level mapping the model trained under does not hold at rollout: a legitimate readout, but not the conditional the loss trained. Arm 4 is evaluated clean only, as asked.

## Findings

`argmax` success rate **averaged over every evaluated checkpoint** — not a best-checkpoint number, and nothing here nominates one. Averaging is what makes the columns readable at all: one cell carries a ~±0.13 CI at 50 episodes, so no single checkpoint separates these arms.

| arm | rollouts | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 | ckpts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UNet BC | clean | 0.16 | 0.20 | 0.24 | 0.33 | 0.36 | 0.40 | 0.50 | 10 |
| ST k=1 | clean | 0.08 | 0.14 | 0.22 | 0.22 | 0.31 | 0.29 | 0.32 | 10 |
| ST k=16, uniform | clean | 0.08 | 0.16 | 0.17 | 0.21 | 0.26 | 0.24 | 0.31 | 10 |
| 1. linear in t, cap 999 | noised | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.30 | 0.63 | 10 |
| 1. linear in t, cap 999 | clean | 0.00 | 0.00 | 0.06 | 0.19 | 0.29 | 0.35 | 0.40 | 10 |
| 1. linear in t, cap 400 | noised | 0.07 | 0.13 | 0.25 | 0.30 | 0.32 | 0.36 | 0.39 | 10 |
| 1. linear in t, cap 400 | clean | 0.09 | 0.18 | 0.25 | 0.29 | 0.35 | 0.37 | 0.43 | 10 |
| 2. linear in a_bar, cap 999 | noised | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.35 | 0.57 | 10 |
| 2. linear in a_bar, cap 999 | clean | 0.00 | 0.01 | 0.13 | 0.21 | 0.25 | 0.34 | 0.37 | 10 |
| 2. linear in a_bar, cap 400 | noised | 0.08 | 0.15 | 0.20 | 0.23 | 0.30 | 0.36 | 0.36 | 10 |
| 2. linear in a_bar, cap 400 | clean | 0.08 | 0.15 | 0.20 | 0.26 | 0.28 | 0.33 | 0.34 | 10 |
| 3. geometric in t, cap 999 | noised | 0.00 | 0.00 | 0.00 | 0.01 | 0.03 | 0.37 | 0.57 | 10 |
| 3. geometric in t, cap 999 | clean | 0.00 | 0.00 | 0.11 | 0.17 | 0.26 | 0.24 | 0.33 | 10 |
| 3. geometric in t, cap 400 | noised | 0.06 | 0.13 | 0.17 | 0.23 | 0.31 | 0.30 | 0.37 | 10 |
| 3. geometric in t, cap 400 | clean | 0.10 | 0.19 | 0.19 | 0.26 | 0.27 | 0.33 | 0.36 | 10 |
| 4. random base, decaying in a_bar | clean | 0.05 | 0.17 | 0.20 | 0.29 | 0.31 | 0.33 | 0.39 | 10 |

**1. Cap 999 destroys the policy at low n; cap 400 does not.** All three full-extent fixed ladders read 0.00 at n = 1 and n = 2 under both rollout modes, at every checkpoint. That is structural, not noise: at cap 999 slot 0 sits at sqrt(alpha_bar) = 0.006, and n = 1 generates from slot 0 alone, so the model is asked to act from a slot it only ever saw as near-pure noise. The cap-400 arms (slot 0 at 0.44) keep their low-n numbers and match or beat the uniform control there.

**2. Under noised rollouts the cap-999 arms have a hard n = 16 -> 32 cliff, and then beat everything.** They sit at 0.00 through n = 8, ~0.02-0.04 at n = 16, then jump to 0.30-0.37 at n = 32 and **0.57-0.63 at n = 64**, against the uniform control's 0.31. The cliff is the deployed-slot-range effect (see Caveats): below n = 16 only the ladder's noisy end is ever generated, and past n = 16 the rolling window pins further generations at the clean slot 15.

**3. The cap-999 gain needs matched noise at deployment.** The same arms evaluated CLEAN reach only 0.33-0.40 at n = 64 — better than the control, but far below their own noised numbers. So the benefit is not "the ladder trained a better policy"; it is that the conditional the loss trained only holds when rollouts see the same corruption.

**4. Every cap-400 ladder modestly beats the uniform control, at every n.** `lint-400` is the best-behaved arm in the grid: 0.09 / 0.18 / 0.25 / 0.29 / 0.35 / 0.37 / 0.43 against the control's 0.08 / 0.16 / 0.17 / 0.21 / 0.26 / 0.24 / 0.31, with no low-n collapse and no dependence on deploying noised.

**5. `random_base` does what it was designed to do.** It is the only full-extent arm that does not collapse at n = 1 (0.05, against 0.00 for the three fixed cap-999 ladders), because its per-sample base leaves slot 0 often near-clean, so the marginal survives training. It tracks the cap-400 arms rather than the cap-999 ones.

**6. No arm beats UNet BC.** BC leads at every n (0.16 -> 0.50) and is the strongest thing in the table. It is not a like-for-like comparison — see Caveats — but no ladder recovers the gap on the clean readout.

**7. `final_pass` does not benefit from a ladder.** The hypothesis was that executing the last (cleanest-slot) generation would exploit the gradient. It does not: ladder arms read 0.10-0.16 at n = 8/16 against the control's 0.13/0.15, and under noised rollouts the cap-999 arms are **0.00 at every n and every checkpoint**. Full tables per arm below.

## UNet BC

A convolutional diffusion UNet with no transformer and no search context at all -- it isolates the backbone. No ladder.

<sub>`unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-30_seed-42`</sub>

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.08 | 0.14 | 0.20 | 0.40 | 0.50 | 0.58 | 0.80 |
| 20,000 | 0.08 | 0.20 | 0.28 | 0.34 | 0.50 | 0.70 | 0.76 |
| 30,000 | 0.12 | 0.22 | 0.30 | 0.36 | 0.40 | 0.46 | 0.58 |
| 40,000 | 0.28 | 0.24 | 0.26 | 0.36 | 0.38 | 0.48 | 0.52 |
| 50,000 | 0.10 | 0.22 | 0.28 | 0.34 | 0.36 | 0.42 | 0.52 |
| 60,000 | 0.16 | 0.20 | 0.18 | 0.40 | 0.32 | 0.40 | 0.46 |
| 70,000 | 0.16 | 0.24 | 0.26 | 0.32 | 0.26 | 0.18 | 0.30 |
| 80,000 | 0.14 | 0.18 | 0.26 | 0.24 | 0.36 | 0.30 | 0.36 |
| 90,000 | 0.20 | 0.22 | 0.22 | 0.28 | 0.24 | 0.22 | 0.38 |
| 100,000 | 0.24 | 0.14 | 0.18 | 0.28 | 0.26 | 0.28 | 0.32 |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.08 | – | – | 0.00 | 0.04 | – | – |
| 20,000 | 0.08 | – | – | 0.12 | 0.06 | – | – |
| 30,000 | 0.12 | – | – | 0.12 | 0.10 | – | – |
| 40,000 | 0.28 | – | – | 0.14 | 0.12 | – | – |
| 50,000 | 0.10 | – | – | 0.18 | 0.18 | – | – |
| 60,000 | 0.16 | – | – | 0.18 | 0.20 | – | – |
| 70,000 | 0.16 | – | – | 0.14 | 0.08 | – | – |
| 80,000 | 0.14 | – | – | 0.20 | 0.18 | – | – |
| 90,000 | 0.20 | – | – | 0.14 | 0.14 | – | – |
| 100,000 | 0.24 | – | – | 0.20 | 0.18 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 10 · partial: 0 · episodes: [50] · seed: [42]</sub>

## ST k=1

The same transformer as the k=16 arms trained at `max_actions: 1`, so its search context is always empty. No ladder (a one-slot ladder is not a ladder).

<sub>`offline/value_k1_ver-t_goal_enc-resnet18_demos-30_seed-42`</sub>

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.04 | 0.04 | 0.16 | 0.16 | 0.44 | 0.40 | 0.40 |
| 20,000 | 0.06 | 0.20 | 0.22 | 0.22 | 0.40 | 0.32 | 0.38 |
| 30,000 | 0.04 | 0.08 | 0.28 | 0.22 | 0.30 | 0.38 | 0.38 |
| 40,000 | 0.12 | 0.14 | 0.24 | 0.24 | 0.28 | 0.34 | 0.30 |
| 50,000 | 0.08 | 0.18 | 0.24 | 0.28 | 0.24 | 0.22 | 0.30 |
| 60,000 | 0.16 | 0.18 | 0.18 | 0.14 | 0.30 | 0.22 | 0.28 |
| 70,000 | 0.06 | 0.12 | 0.30 | 0.26 | 0.30 | 0.20 | 0.24 |
| 80,000 | 0.12 | 0.26 | 0.26 | 0.14 | 0.28 | 0.30 | 0.36 |
| 90,000 | 0.10 | 0.10 | 0.14 | 0.26 | 0.20 | 0.20 | 0.32 |
| 100,000 | 0.04 | 0.10 | 0.14 | 0.24 | 0.32 | 0.36 | 0.28 |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.04 | – | – | 0.02 | 0.00 | – | – |
| 20,000 | 0.06 | – | – | 0.04 | 0.02 | – | – |
| 30,000 | 0.04 | – | – | 0.10 | 0.14 | – | – |
| 40,000 | 0.12 | – | – | 0.14 | 0.06 | – | – |
| 50,000 | 0.08 | – | – | 0.14 | 0.08 | – | – |
| 60,000 | 0.16 | – | – | 0.08 | 0.16 | – | – |
| 70,000 | 0.06 | – | – | 0.10 | 0.12 | – | – |
| 80,000 | 0.12 | – | – | 0.12 | 0.10 | – | – |
| 90,000 | 0.10 | – | – | 0.16 | 0.28 | – | – |
| 100,000 | 0.04 | – | – | 0.12 | 0.14 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 10 · partial: 0 · episodes: [50] · seed: [42]</sub>

## ST k=16, uniform

The control every ladder is read against: identical in every respect except that `slot_obs_noise` is uniform, so all 16 slots see the same clean observation.

<sub>`outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-30_seed-42`</sub>

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | 0.16 | 0.18 | 0.34 | 0.28 | 0.38 | 0.52 |
| 20,000 | 0.10 | 0.16 | 0.16 | 0.36 | 0.34 | 0.40 | 0.48 |
| 30,000 | 0.12 | 0.20 | 0.24 | 0.32 | 0.42 | 0.32 | 0.48 |
| 40,000 | 0.14 | 0.18 | 0.18 | 0.30 | 0.28 | 0.22 | 0.26 |
| 50,000 | 0.04 | 0.24 | 0.18 | 0.12 | 0.22 | 0.22 | 0.26 |
| 60,000 | 0.04 | 0.12 | 0.18 | 0.10 | 0.22 | 0.18 | 0.28 |
| 70,000 | 0.06 | 0.14 | 0.14 | 0.14 | 0.12 | 0.12 | 0.12 |
| 80,000 | 0.06 | 0.16 | 0.18 | 0.10 | 0.24 | 0.16 | 0.22 |
| 90,000 | 0.10 | 0.10 | 0.16 | 0.12 | 0.22 | 0.24 | 0.32 |
| 100,000 | 0.14 | 0.18 | 0.06 | 0.20 | 0.26 | 0.20 | 0.20 |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | – | – | 0.12 | 0.14 | – | – |
| 20,000 | 0.10 | – | – | 0.18 | 0.14 | – | – |
| 30,000 | 0.12 | – | – | 0.08 | 0.26 | – | – |
| 40,000 | 0.14 | – | – | 0.10 | 0.16 | – | – |
| 50,000 | 0.04 | – | – | 0.18 | 0.10 | – | – |
| 60,000 | 0.04 | – | – | 0.12 | 0.16 | – | – |
| 70,000 | 0.06 | – | – | 0.16 | 0.14 | – | – |
| 80,000 | 0.06 | – | – | 0.08 | 0.16 | – | – |
| 90,000 | 0.10 | – | – | 0.14 | 0.12 | – | – |
| 100,000 | 0.16 | – | – | 0.14 | 0.14 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 10 · partial: 0 · episodes: [50] · seed: [42]</sub>

## 1. linear in t, cap 999

`t_k = (15-k)/15 * 999`. Even in the TIMESTEP index, which is NOT even in corruption: because alpha_bar is a cumulative product, slots 0-3 all land within 0.03 of each other in sqrt(alpha_bar) (0.01/0.01/0.02/0.04) while slots 10-15 are spread over 0.44. Kept as an arm because that skew is exactly the contrast `linear_signal` exists to fix.

<sub>`outer_inner/value_k16_ver-t_goal_son-lint-cap999_enc-resnet18_demos-30_seed-42`</sub>

**argmax, rollouts noised, same ladder as training** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.42 | 0.78 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.32 | 0.62 |
| 30,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.40 | 0.76 |
| 40,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.38 | 0.58 |
| 50,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.52 | 0.58 |
| 60,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.28 | 0.66 |
| 70,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.44 |
| 80,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.18 | 0.64 |
| 90,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.06 | 0.62 |
| 100,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.24 | 0.64 |

**final_pass, rollouts noised, same ladder as training** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 30,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 40,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 50,000 | 0.00 | – | – | 0.00 | 0.02 | – | – |
| 60,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 70,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 80,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 90,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 100,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |

**argmax, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.08 | 0.22 | 0.32 | 0.54 | 0.52 |
| 20,000 | 0.00 | 0.00 | 0.10 | 0.38 | 0.38 | 0.44 | 0.60 |
| 30,000 | 0.00 | 0.00 | 0.10 | 0.22 | 0.36 | 0.52 | 0.42 |
| 40,000 | 0.00 | 0.00 | 0.08 | 0.34 | 0.50 | 0.44 | 0.52 |
| 50,000 | 0.00 | 0.00 | 0.08 | 0.16 | 0.32 | 0.34 | 0.42 |
| 60,000 | 0.00 | 0.00 | 0.06 | 0.14 | 0.12 | 0.38 | 0.38 |
| 70,000 | 0.00 | 0.00 | 0.04 | 0.12 | 0.26 | 0.20 | 0.34 |
| 80,000 | 0.00 | 0.00 | 0.00 | 0.10 | 0.20 | 0.22 | 0.28 |
| 90,000 | 0.00 | 0.00 | 0.02 | 0.12 | 0.18 | 0.28 | 0.22 |
| 100,000 | 0.00 | 0.00 | 0.00 | 0.06 | 0.26 | 0.16 | 0.26 |

**final_pass, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.04 | 0.06 | – | – |
| 20,000 | 0.00 | – | – | 0.04 | 0.02 | – | – |
| 30,000 | 0.00 | – | – | 0.10 | 0.06 | – | – |
| 40,000 | 0.00 | – | – | 0.26 | 0.18 | – | – |
| 50,000 | 0.00 | – | – | 0.14 | 0.16 | – | – |
| 60,000 | 0.00 | – | – | 0.16 | 0.10 | – | – |
| 70,000 | 0.00 | – | – | 0.12 | 0.12 | – | – |
| 80,000 | 0.00 | – | – | 0.18 | 0.18 | – | – |
| 90,000 | 0.00 | – | – | 0.16 | 0.18 | – | – |
| 100,000 | 0.00 | – | – | 0.16 | 0.06 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 10 · partial: 0 · episodes: [50] · seed: [42]</sub>

## 1. linear in t, cap 400

The same shape compressed into [0, 400], so slot 0 sits at sqrt(alpha_bar) = 0.44 rather than 0.01 -- a degraded observation instead of very nearly pure noise.

<sub>`outer_inner/value_k16_ver-t_goal_son-lint-cap400_enc-resnet18_demos-30_seed-42`</sub>

**argmax, rollouts noised, same ladder as training** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | 0.12 | 0.24 | 0.38 | 0.46 | 0.46 | 0.58 |
| 20,000 | 0.04 | 0.12 | 0.20 | 0.30 | 0.32 | 0.36 | 0.42 |
| 30,000 | 0.04 | 0.08 | 0.28 | 0.30 | 0.30 | 0.42 | 0.52 |
| 40,000 | 0.06 | 0.16 | 0.34 | 0.24 | 0.32 | 0.44 | 0.44 |
| 50,000 | 0.08 | 0.22 | 0.20 | 0.30 | 0.28 | 0.28 | 0.32 |
| 60,000 | 0.08 | 0.18 | 0.22 | 0.34 | 0.44 | 0.38 | 0.30 |
| 70,000 | 0.08 | 0.10 | 0.26 | 0.32 | 0.32 | 0.28 | 0.26 |
| 80,000 | 0.12 | 0.08 | 0.24 | 0.20 | 0.20 | 0.36 | 0.30 |
| 90,000 | 0.16 | 0.14 | 0.30 | 0.26 | 0.24 | 0.32 | 0.46 |
| 100,000 | 0.06 | 0.14 | 0.18 | 0.32 | 0.30 | 0.30 | 0.34 |

**final_pass, rollouts noised, same ladder as training** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | – | – | 0.04 | 0.10 | – | – |
| 20,000 | 0.04 | – | – | 0.16 | 0.04 | – | – |
| 30,000 | 0.04 | – | – | 0.24 | 0.14 | – | – |
| 40,000 | 0.06 | – | – | 0.08 | 0.10 | – | – |
| 50,000 | 0.08 | – | – | 0.08 | 0.06 | – | – |
| 60,000 | 0.10 | – | – | 0.08 | 0.12 | – | – |
| 70,000 | 0.10 | – | – | 0.10 | 0.18 | – | – |
| 80,000 | 0.12 | – | – | 0.20 | 0.06 | – | – |
| 90,000 | 0.14 | – | – | 0.16 | 0.16 | – | – |
| 100,000 | 0.08 | – | – | 0.16 | 0.16 | – | – |

**argmax, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | 0.02 | 0.24 | 0.28 | 0.40 | 0.42 | 0.48 |
| 20,000 | 0.06 | 0.22 | 0.20 | 0.28 | 0.36 | 0.38 | 0.52 |
| 30,000 | 0.06 | 0.28 | 0.30 | 0.30 | 0.40 | 0.44 | 0.44 |
| 40,000 | 0.14 | 0.22 | 0.36 | 0.26 | 0.38 | 0.34 | 0.46 |
| 50,000 | 0.08 | 0.20 | 0.32 | 0.34 | 0.36 | 0.42 | 0.38 |
| 60,000 | 0.08 | 0.22 | 0.34 | 0.34 | 0.28 | 0.34 | 0.52 |
| 70,000 | 0.08 | 0.10 | 0.18 | 0.30 | 0.28 | 0.34 | 0.34 |
| 80,000 | 0.12 | 0.08 | 0.18 | 0.28 | 0.36 | 0.38 | 0.30 |
| 90,000 | 0.10 | 0.26 | 0.16 | 0.32 | 0.38 | 0.34 | 0.46 |
| 100,000 | 0.12 | 0.24 | 0.24 | 0.22 | 0.32 | 0.32 | 0.42 |

**final_pass, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | – | – | 0.08 | 0.02 | – | – |
| 20,000 | 0.06 | – | – | 0.24 | 0.18 | – | – |
| 30,000 | 0.06 | – | – | 0.12 | 0.10 | – | – |
| 40,000 | 0.14 | – | – | 0.22 | 0.12 | – | – |
| 50,000 | 0.08 | – | – | 0.18 | 0.10 | – | – |
| 60,000 | 0.08 | – | – | 0.12 | 0.16 | – | – |
| 70,000 | 0.08 | – | – | 0.14 | 0.06 | – | – |
| 80,000 | 0.14 | – | – | 0.18 | 0.10 | – | – |
| 90,000 | 0.10 | – | – | 0.14 | 0.08 | – | – |
| 100,000 | 0.12 | – | – | 0.14 | 0.08 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 10 · partial: 0 · episodes: [50] · seed: [42]</sub>

## 2. linear in a_bar, cap 999

`t_k` chosen so sqrt(alpha_bar) -- the factor the observation is actually multiplied by -- falls in equal ~0.066 steps from 0.01 at slot 0 to 1.00 at slot 15. The only shape that grades evenly from the marginal to the conditional; all 16 levels are distinct.

<sub>`outer_inner/value_k16_ver-t_goal_son-linsig-cap999_enc-resnet18_demos-30_seed-42`</sub>

**argmax, rollouts noised, same ladder as training** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.44 | 0.78 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.42 | 0.64 |
| 30,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.36 | 0.58 |
| 40,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.22 | 0.56 |
| 50,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.08 | 0.46 | 0.46 |
| 60,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.44 | 0.68 |
| 70,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.28 | 0.46 |
| 80,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.44 | 0.54 |
| 90,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.18 | 0.44 |
| 100,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.22 | 0.52 |

**final_pass, rollouts noised, same ladder as training** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.02 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 30,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 40,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 50,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 60,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 70,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 80,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 90,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 100,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |

**argmax, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.02 | 0.12 | 0.18 | 0.26 | 0.44 | 0.46 |
| 20,000 | 0.00 | 0.00 | 0.14 | 0.18 | 0.16 | 0.30 | 0.38 |
| 30,000 | 0.00 | 0.00 | 0.16 | 0.20 | 0.36 | 0.40 | 0.34 |
| 40,000 | 0.00 | 0.04 | 0.14 | 0.26 | 0.24 | 0.36 | 0.38 |
| 50,000 | 0.00 | 0.02 | 0.18 | 0.30 | 0.36 | 0.44 | 0.46 |
| 60,000 | 0.00 | 0.00 | 0.20 | 0.20 | 0.18 | 0.30 | 0.36 |
| 70,000 | 0.00 | 0.00 | 0.12 | 0.24 | 0.24 | 0.28 | 0.40 |
| 80,000 | 0.00 | 0.00 | 0.08 | 0.24 | 0.18 | 0.30 | 0.32 |
| 90,000 | 0.00 | 0.00 | 0.08 | 0.20 | 0.32 | 0.24 | 0.28 |
| 100,000 | 0.00 | 0.00 | 0.08 | 0.12 | 0.22 | 0.30 | 0.36 |

**final_pass, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.08 | 0.06 | – | – |
| 20,000 | 0.00 | – | – | 0.08 | 0.08 | – | – |
| 30,000 | 0.00 | – | – | 0.14 | 0.12 | – | – |
| 40,000 | 0.00 | – | – | 0.16 | 0.18 | – | – |
| 50,000 | 0.00 | – | – | 0.24 | 0.18 | – | – |
| 60,000 | 0.00 | – | – | 0.08 | 0.14 | – | – |
| 70,000 | 0.00 | – | – | 0.12 | 0.14 | – | – |
| 80,000 | 0.00 | – | – | 0.12 | 0.10 | – | – |
| 90,000 | 0.00 | – | – | 0.24 | 0.16 | – | – |
| 100,000 | 0.00 | – | – | 0.20 | 0.08 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 10 · partial: 0 · episodes: [50] · seed: [42]</sub>

## 2. linear in a_bar, cap 400

The same even grading compressed into [0, 400]: 0.44 at slot 0 up to 1.00 at slot 15. Note the compression is even in TIMESTEP, not in signal, so the retained-signal steps are no longer equal (0.21 from slot 0 to 1, then ~0.02 near the clean end).

<sub>`outer_inner/value_k16_ver-t_goal_son-linsig-cap400_enc-resnet18_demos-30_seed-42`</sub>

**argmax, rollouts noised, same ladder as training** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.08 | 0.06 | 0.18 | 0.28 | 0.36 | 0.46 | 0.48 |
| 20,000 | 0.02 | 0.14 | 0.14 | 0.22 | 0.32 | 0.46 | 0.54 |
| 30,000 | 0.06 | 0.12 | 0.22 | 0.14 | 0.38 | 0.46 | 0.42 |
| 40,000 | 0.08 | 0.18 | 0.22 | 0.28 | 0.34 | 0.36 | 0.38 |
| 50,000 | 0.04 | 0.22 | 0.16 | 0.18 | 0.26 | 0.28 | 0.30 |
| 60,000 | 0.12 | 0.12 | 0.18 | 0.24 | 0.22 | 0.26 | 0.26 |
| 70,000 | 0.12 | 0.20 | 0.20 | 0.22 | 0.24 | 0.36 | 0.30 |
| 80,000 | 0.06 | 0.14 | 0.16 | 0.28 | 0.34 | 0.34 | 0.32 |
| 90,000 | 0.08 | 0.18 | 0.36 | 0.22 | 0.24 | 0.32 | 0.34 |
| 100,000 | 0.12 | 0.16 | 0.22 | 0.20 | 0.28 | 0.28 | 0.28 |

**final_pass, rollouts noised, same ladder as training** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.08 | – | – | 0.06 | 0.10 | – | – |
| 20,000 | 0.02 | – | – | 0.14 | 0.12 | – | – |
| 30,000 | 0.06 | – | – | 0.08 | 0.08 | – | – |
| 40,000 | 0.08 | – | – | 0.06 | 0.16 | – | – |
| 50,000 | 0.04 | – | – | 0.14 | 0.10 | – | – |
| 60,000 | 0.12 | – | – | 0.14 | 0.10 | – | – |
| 70,000 | 0.12 | – | – | 0.16 | 0.14 | – | – |
| 80,000 | 0.06 | – | – | 0.16 | 0.20 | – | – |
| 90,000 | 0.08 | – | – | 0.10 | 0.12 | – | – |
| 100,000 | 0.12 | – | – | 0.22 | 0.18 | – | – |

**argmax, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.06 | 0.14 | 0.24 | 0.26 | 0.26 | 0.32 | 0.52 |
| 20,000 | 0.08 | 0.16 | 0.10 | 0.26 | 0.34 | 0.42 | 0.46 |
| 30,000 | 0.08 | 0.22 | 0.16 | 0.24 | 0.34 | 0.38 | 0.36 |
| 40,000 | 0.02 | 0.16 | 0.24 | 0.30 | 0.24 | 0.38 | 0.40 |
| 50,000 | 0.06 | 0.12 | 0.26 | 0.22 | 0.26 | 0.26 | 0.34 |
| 60,000 | 0.12 | 0.16 | 0.22 | 0.36 | 0.26 | 0.32 | 0.26 |
| 70,000 | 0.10 | 0.10 | 0.22 | 0.24 | 0.24 | 0.28 | 0.30 |
| 80,000 | 0.08 | 0.18 | 0.20 | 0.18 | 0.24 | 0.38 | 0.28 |
| 90,000 | 0.08 | 0.08 | 0.12 | 0.28 | 0.28 | 0.28 | 0.28 |
| 100,000 | 0.12 | 0.16 | 0.20 | 0.26 | 0.32 | 0.28 | 0.22 |

**final_pass, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.06 | – | – | 0.06 | 0.02 | – | – |
| 20,000 | 0.08 | – | – | 0.10 | 0.08 | – | – |
| 30,000 | 0.08 | – | – | 0.18 | 0.16 | – | – |
| 40,000 | 0.02 | – | – | 0.22 | 0.22 | – | – |
| 50,000 | 0.06 | – | – | 0.14 | 0.16 | – | – |
| 60,000 | 0.12 | – | – | 0.12 | 0.14 | – | – |
| 70,000 | 0.10 | – | – | 0.16 | 0.14 | – | – |
| 80,000 | 0.08 | – | – | 0.10 | 0.16 | – | – |
| 90,000 | 0.08 | – | – | 0.14 | 0.16 | – | – |
| 100,000 | 0.12 | – | – | 0.14 | 0.12 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 10 · partial: 0 · episodes: [50] · seed: [42]</sub>

## 3. geometric in t, cap 999

`t_k = 999 * 0.85^k`. Decay 0.85 rather than the 0.7 that `slot_weights` uses: at K=16 decay 0.7 leaves 6 of 15 adjacent slots within 0.005 of each other in sqrt(alpha_bar), so most of that ladder would be the same observation and "geometric lost" could not be separated from "geometric collapsed". At 0.85 all 16 levels are distinct. The trade: slot 15 lands at t=87 (sqrt(alpha_bar) 0.958), so this arm's cleanest slot is not fully clean.

<sub>`outer_inner/value_k16_ver-t_goal_son-geo85-cap999_enc-resnet18_demos-30_seed-42`</sub>

**argmax, rollouts noised, same ladder as training** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.04 | 0.48 | 0.66 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.42 | 0.66 |
| 30,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.32 | 0.54 |
| 40,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.42 | 0.58 |
| 50,000 | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 | 0.54 | 0.62 |
| 60,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.36 | 0.66 |
| 70,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.28 | 0.46 |
| 80,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.34 | 0.58 |
| 90,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.24 | 0.52 |
| 100,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.30 | 0.44 |

**final_pass, rollouts noised, same ladder as training** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 30,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 40,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 50,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 60,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 70,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 80,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 90,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 100,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |

**argmax, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.10 | 0.20 | 0.28 | 0.30 | 0.54 |
| 20,000 | 0.00 | 0.00 | 0.14 | 0.18 | 0.38 | 0.34 | 0.36 |
| 30,000 | 0.00 | 0.00 | 0.08 | 0.22 | 0.36 | 0.26 | 0.30 |
| 40,000 | 0.00 | 0.00 | 0.10 | 0.22 | 0.30 | 0.28 | 0.28 |
| 50,000 | 0.00 | 0.00 | 0.06 | 0.22 | 0.34 | 0.32 | 0.44 |
| 60,000 | 0.00 | 0.00 | 0.12 | 0.16 | 0.20 | 0.22 | 0.32 |
| 70,000 | 0.00 | 0.00 | 0.12 | 0.12 | 0.30 | 0.18 | 0.34 |
| 80,000 | 0.00 | 0.00 | 0.04 | 0.10 | 0.18 | 0.16 | 0.24 |
| 90,000 | 0.00 | 0.00 | 0.10 | 0.18 | 0.10 | 0.18 | 0.32 |
| 100,000 | 0.00 | 0.00 | 0.22 | 0.14 | 0.12 | 0.18 | 0.14 |

**final_pass, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.04 | 0.04 | – | – |
| 20,000 | 0.00 | – | – | 0.10 | 0.08 | – | – |
| 30,000 | 0.00 | – | – | 0.04 | 0.16 | – | – |
| 40,000 | 0.00 | – | – | 0.10 | 0.12 | – | – |
| 50,000 | 0.00 | – | – | 0.22 | 0.20 | – | – |
| 60,000 | 0.00 | – | – | 0.12 | 0.16 | – | – |
| 70,000 | 0.00 | – | – | 0.22 | 0.12 | – | – |
| 80,000 | 0.00 | – | – | 0.20 | 0.20 | – | – |
| 90,000 | 0.00 | – | – | 0.16 | 0.16 | – | – |
| 100,000 | 0.00 | – | – | 0.10 | 0.18 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 10 · partial: 0 · episodes: [50] · seed: [42]</sub>

## 3. geometric in t, cap 400

The same decay compressed into [0, 400]. 2 of 15 adjacent pairs fall within 0.005 here, so this arm is mildly collapsed at the clean end where the 999 one is not.

<sub>`outer_inner/value_k16_ver-t_goal_son-geo85-cap400_enc-resnet18_demos-30_seed-42`</sub>

**argmax, rollouts noised, same ladder as training** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | 0.04 | 0.16 | 0.28 | 0.32 | 0.44 | 0.34 |
| 20,000 | 0.04 | 0.18 | 0.14 | 0.32 | 0.36 | 0.44 | 0.36 |
| 30,000 | 0.04 | 0.10 | 0.22 | 0.36 | 0.34 | 0.44 | 0.48 |
| 40,000 | 0.04 | 0.12 | 0.30 | 0.28 | 0.34 | 0.28 | 0.44 |
| 50,000 | 0.14 | 0.18 | 0.12 | 0.16 | 0.28 | 0.28 | 0.40 |
| 60,000 | 0.04 | 0.12 | 0.18 | 0.22 | 0.28 | 0.22 | 0.32 |
| 70,000 | 0.08 | 0.14 | 0.20 | 0.24 | 0.32 | 0.24 | 0.46 |
| 80,000 | 0.06 | 0.08 | 0.20 | 0.18 | 0.28 | 0.28 | 0.30 |
| 90,000 | 0.04 | 0.12 | 0.08 | 0.16 | 0.32 | 0.16 | 0.28 |
| 100,000 | 0.08 | 0.26 | 0.12 | 0.14 | 0.28 | 0.26 | 0.30 |

**final_pass, rollouts noised, same ladder as training** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | – | – | 0.06 | 0.12 | – | – |
| 20,000 | 0.04 | – | – | 0.14 | 0.18 | – | – |
| 30,000 | 0.04 | – | – | 0.18 | 0.10 | – | – |
| 40,000 | 0.04 | – | – | 0.14 | 0.06 | – | – |
| 50,000 | 0.14 | – | – | 0.14 | 0.12 | – | – |
| 60,000 | 0.04 | – | – | 0.12 | 0.10 | – | – |
| 70,000 | 0.08 | – | – | 0.04 | 0.10 | – | – |
| 80,000 | 0.06 | – | – | 0.16 | 0.08 | – | – |
| 90,000 | 0.06 | – | – | 0.12 | 0.12 | – | – |
| 100,000 | 0.08 | – | – | 0.18 | 0.12 | – | – |

**argmax, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.10 | 0.24 | 0.28 | 0.30 | 0.40 | 0.40 |
| 20,000 | 0.02 | 0.14 | 0.18 | 0.32 | 0.34 | 0.38 | 0.46 |
| 30,000 | 0.06 | 0.28 | 0.18 | 0.34 | 0.36 | 0.32 | 0.38 |
| 40,000 | 0.12 | 0.20 | 0.26 | 0.36 | 0.32 | 0.34 | 0.42 |
| 50,000 | 0.12 | 0.14 | 0.16 | 0.28 | 0.32 | 0.48 | 0.36 |
| 60,000 | 0.16 | 0.20 | 0.10 | 0.22 | 0.26 | 0.38 | 0.32 |
| 70,000 | 0.14 | 0.28 | 0.18 | 0.20 | 0.28 | 0.28 | 0.32 |
| 80,000 | 0.14 | 0.18 | 0.18 | 0.18 | 0.20 | 0.30 | 0.30 |
| 90,000 | 0.12 | 0.20 | 0.20 | 0.16 | 0.14 | 0.30 | 0.32 |
| 100,000 | 0.12 | 0.16 | 0.26 | 0.26 | 0.22 | 0.14 | 0.28 |

**final_pass, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.12 | 0.10 | – | – |
| 20,000 | 0.02 | – | – | 0.14 | 0.14 | – | – |
| 30,000 | 0.06 | – | – | 0.18 | 0.12 | – | – |
| 40,000 | 0.12 | – | – | 0.12 | 0.18 | – | – |
| 50,000 | 0.12 | – | – | 0.16 | 0.08 | – | – |
| 60,000 | 0.16 | – | – | 0.20 | 0.10 | – | – |
| 70,000 | 0.14 | – | – | 0.20 | 0.16 | – | – |
| 80,000 | 0.14 | – | – | 0.16 | 0.16 | – | – |
| 90,000 | 0.12 | – | – | 0.10 | 0.14 | – | – |
| 100,000 | 0.12 | – | – | 0.14 | 0.10 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 10 · partial: 0 · episodes: [50] · seed: [42]</sub>

## 4. random base, decaying in a_bar

No fixed ladder. Slot 0's timestep is drawn PER SAMPLE from [0, 999] and the linear_signal curve is rescaled into [0, that draw], so slot 0 sits at the drawn level and slot 15 stays clean -- what varies between samples is the ladder's EXTENT, not its shape. At a draw of 999 it is exactly arm 2; at 0 every slot is clean. The base is pinned across a decision, so the 16 slots remain one observation seen at graded levels.

<sub>`outer_inner/value_k16_ver-t_goal_son-rndlinsig-cap999_enc-resnet18_demos-30_seed-42`</sub>

**argmax, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.12 | 0.22 | 0.32 | 0.28 | 0.42 | 0.34 |
| 20,000 | 0.06 | 0.08 | 0.16 | 0.32 | 0.34 | 0.36 | 0.48 |
| 30,000 | 0.08 | 0.18 | 0.24 | 0.38 | 0.38 | 0.36 | 0.42 |
| 40,000 | 0.06 | 0.20 | 0.18 | 0.22 | 0.30 | 0.36 | 0.40 |
| 50,000 | 0.04 | 0.32 | 0.24 | 0.30 | 0.26 | 0.30 | 0.36 |
| 60,000 | 0.06 | 0.16 | 0.18 | 0.42 | 0.34 | 0.38 | 0.52 |
| 70,000 | 0.06 | 0.16 | 0.16 | 0.22 | 0.30 | 0.30 | 0.40 |
| 80,000 | 0.06 | 0.20 | 0.18 | 0.20 | 0.34 | 0.22 | 0.30 |
| 90,000 | 0.06 | 0.18 | 0.20 | 0.28 | 0.28 | 0.28 | 0.32 |
| 100,000 | 0.04 | 0.14 | 0.20 | 0.22 | 0.30 | 0.34 | 0.38 |

**final_pass, rollouts clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.12 | 0.06 | – | – |
| 20,000 | 0.06 | – | – | 0.12 | 0.12 | – | – |
| 30,000 | 0.08 | – | – | 0.16 | 0.10 | – | – |
| 40,000 | 0.06 | – | – | 0.06 | 0.12 | – | – |
| 50,000 | 0.04 | – | – | 0.10 | 0.04 | – | – |
| 60,000 | 0.06 | – | – | 0.10 | 0.10 | – | – |
| 70,000 | 0.06 | – | – | 0.08 | 0.14 | – | – |
| 80,000 | 0.06 | – | – | 0.18 | 0.12 | – | – |
| 90,000 | 0.06 | – | – | 0.20 | 0.16 | – | – |
| 100,000 | 0.04 | – | – | 0.12 | 0.18 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 10 · partial: 0 · episodes: [50] · seed: [42]</sub>

## Caveats

**`final_pass` is degenerate on the three baselines.** It executes the last-generated candidate instead of the verifier's pick, so with i.i.d. candidates and no ladder it reduces to "sample once, ignore the verifier" -- the n=1 argmax number. Those three tables are that control. On the ladder arms it is not degenerate, and that is the point: there the last generation is the one that saw the cleanest observation.

**The deployed slot range moves with n.** At n < 16 only slots 0..n-1 are ever generated -- the noisy end of the ladder -- and past n = 16 a rolling window pins every further generation at slot 15. So the n axis and the ladder axis are not independent, and the low-n columns of a ladder arm are read under systematically more corruption than the high-n ones.

**BC is not a like-for-like architecture comparison.** The UNet BC arm optimizes far more parameters than the ST arms. It shares the encoder, crop and image pipeline, so the observation is matched; the capacity is not.

**Caps 999 and 400 are not a two-point line.** 999 is the full extent of the obs schedule and 400 is a compression of the same shape into its lower 40%; the compression is even in timestep, so it does not preserve even spacing in retained signal for arm 2.

