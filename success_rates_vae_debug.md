# PushT encoder debug — success rates

_Generated 2026-08-30 by `scripts/build_vae_debug_doc.py`. Re-run to refresh._

30 demos, seed 42, verifier `t_goal`, 50 test episodes, image-only observations, 100k gradient steps, checkpoint every 10k. **No observation corruption on any of these six** — `slot_obs_noise` is uniform, which leaves `slot_obs_t` None so the corruption is the identity, and `corrupt_obs` is false.

A 2x2 of {ResNet18, SD VAE} x {trainable, frozen}, each on two arms, at TWO training budgets (30 and 126 demos -- 126 is every episode that is neither test nor val). ResNet18 trainable tests the revert of the 2026-08-30 speedup pass against `success_rates_no_pos.md`. Reading the square: down a column asks whether freezing hurts that encoder; across the frozen row asks whether SD features are worse than ResNet features when neither can adapt, at matched trainable capacity.

`argmax` sweeps n = 1..64; `final_pass` was asked for at n = 1, 8, 16 only, so its other columns are blank by design. Blank also means "not yet evaluated". No cell is a nominated best.

## 30 demos - ResNet18, trainable

ResNet18 IMAGENET1K_V1, `use_group_norm=True`, 76x76 crop, trained end to end. The reference: at 30 demos it reproduces `success_rates_no_pos.md`, which is what validates the revert of the 2026-08-30 speedup pass.

### UNet BC

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.08 | 0.14 | 0.20 | 0.40 | 0.50 | 0.58 | 0.80 |
| 20,000 | 0.08 | 0.20 | 0.28 | 0.34 | 0.50 | 0.70 | 0.76 |
| 30,000 | 0.12 | 0.22 | 0.30 | 0.36 | 0.40 | 0.46 | 0.58 |
| 40,000 | 0.28 | 0.24 | 0.26 | 0.36 | 0.38 | 0.48 | 0.52 |
| 50,000 | 0.10 | 0.22 | 0.28 | 0.34 | 0.36 | 0.42 | 0.52 |
| 60,000 | 0.16 | 0.20 | 0.18 | 0.40 | 0.32 | 0.40 | 0.46 |
| 70,000 | 0.16 | 0.24 | 0.26 | 0.32 | 0.26 | – | – |

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
| 80,000 | 0.14 | – | – | 0.18 | 0.18 | – | – |
| 90,000 | 0.20 | – | – | 0.14 | 0.14 | – | – |
| 100,000 | 0.24 | – | – | 0.20 | 0.18 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 6 · partial: 1 · episodes: [50] · seed: [42]</sub>

### ST k=1

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.04 | 0.04 | 0.16 | 0.16 | 0.44 | 0.40 | 0.40 |
| 20,000 | 0.06 | 0.20 | 0.22 | 0.22 | 0.40 | 0.32 | 0.38 |
| 30,000 | 0.04 | 0.08 | 0.28 | 0.22 | 0.28 | 0.34 | 0.40 |
| 40,000 | 0.12 | 0.14 | 0.24 | 0.24 | 0.30 | 0.32 | 0.30 |
| 50,000 | 0.08 | 0.18 | 0.22 | 0.28 | 0.24 | 0.24 | 0.28 |
| 60,000 | 0.16 | 0.18 | 0.18 | 0.14 | 0.30 | 0.22 | 0.26 |
| 70,000 | 0.06 | 0.12 | 0.28 | 0.28 | 0.30 | 0.20 | 0.24 |
| 80,000 | 0.12 | 0.24 | 0.26 | 0.14 | 0.28 | 0.32 | 0.34 |
| 90,000 | 0.10 | 0.10 | 0.14 | 0.24 | 0.22 | – | – |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.04 | – | – | 0.02 | 0.00 | – | – |
| 20,000 | 0.06 | – | – | 0.04 | 0.02 | – | – |
| 30,000 | 0.04 | – | – | 0.10 | 0.14 | – | – |
| 40,000 | 0.12 | – | – | 0.14 | 0.06 | – | – |
| 50,000 | 0.08 | – | – | 0.14 | 0.10 | – | – |
| 60,000 | 0.16 | – | – | 0.08 | 0.16 | – | – |
| 70,000 | 0.06 | – | – | 0.10 | 0.14 | – | – |
| 80,000 | 0.12 | – | – | 0.12 | 0.10 | – | – |
| 90,000 | 0.10 | – | – | 0.14 | 0.28 | – | – |
| 100,000 | 0.04 | – | – | 0.12 | 0.14 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 8 · partial: 1 · episodes: [50] · seed: [42]</sub>

## 30 demos - ResNet18, frozen

`training.freeze_encoder=True`, so 11.2M encoder parameters are held out of the optimizer. Against trainable ResNet this asks whether freezing breaks ANY encoder. Caveat: `use_group_norm` replaced the pretrained BatchNorm with freshly built GroupNorm, so this freezes never-trained normalization at identity init and discards ImageNet's running statistics -- a controlled freeze-vs-trainable contrast, not "frozen ImageNet features" in the literature sense.

### UNet BC

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.12 | 0.22 | 0.48 | 0.74 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.20 | 0.32 | 0.46 | 0.84 |
| 30,000 | 0.00 | 0.00 | 0.04 | 0.16 | 0.28 | – | – |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 30,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 40,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 50,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 60,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 70,000 | 0.00 | – | – | – | – | – | – |

<sub>checkpoints on disk: 7 · complete n-sweeps: 2 · partial: 1 · episodes: [50] · seed: [42]</sub>

### ST k=1

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.16 | 0.24 | 0.22 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.06 | 0.18 | 0.14 |
| 30,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.24 | 0.24 |
| 40,000 | 0.00 | 0.00 | – | – | – | – | – |

**final_pass** — test success rate

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

<sub>checkpoints on disk: 10 · complete n-sweeps: 3 · partial: 1 · episodes: [50] · seed: [42]</sub>

## 30 demos - SD VAE, trainable

`sd-vae-ft-mse` with `trainable=True`, 324-d at the 72x72 crop, trained end to end. Against trainable ResNet this isolates the ENCODER with neither one frozen.

### UNet BC

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.06 | 0.10 | 0.14 | 0.24 | 0.38 | 0.56 | 0.62 |
| 20,000 | 0.16 | 0.06 | 0.26 | 0.36 | 0.38 | 0.36 | 0.56 |
| 30,000 | 0.14 | 0.14 | 0.30 | 0.46 | 0.32 | 0.44 | 0.42 |
| 40,000 | 0.18 | 0.06 | 0.40 | 0.24 | 0.42 | 0.46 | 0.52 |
| 50,000 | 0.16 | 0.20 | 0.20 | 0.36 | – | – | – |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.06 | – | – | 0.06 | 0.02 | – | – |
| 20,000 | 0.16 | – | – | 0.12 | 0.08 | – | – |
| 30,000 | 0.14 | – | – | 0.14 | 0.14 | – | – |
| 40,000 | 0.18 | – | – | 0.14 | 0.12 | – | – |
| 50,000 | 0.16 | – | – | 0.16 | 0.16 | – | – |
| 60,000 | 0.18 | – | – | 0.28 | 0.14 | – | – |

<sub>checkpoints on disk: 6 · complete n-sweeps: 4 · partial: 1 · episodes: [50] · seed: [42]</sub>

### ST k=1

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | 0.06 | 0.14 | 0.26 | 0.34 | 0.32 | 0.32 |
| 20,000 | 0.04 | 0.10 | 0.18 | 0.24 | 0.34 | 0.28 | 0.28 |
| 30,000 | 0.08 | 0.08 | 0.16 | 0.24 | 0.34 | 0.30 | 0.32 |
| 40,000 | 0.08 | 0.12 | 0.20 | 0.20 | 0.18 | 0.30 | 0.28 |
| 50,000 | 0.10 | 0.10 | 0.18 | 0.20 | 0.30 | 0.26 | 0.34 |
| 60,000 | 0.10 | 0.06 | 0.12 | 0.18 | 0.28 | 0.28 | 0.30 |
| 70,000 | 0.08 | 0.14 | 0.14 | 0.12 | 0.14 | 0.10 | 0.20 |
| 80,000 | 0.10 | 0.10 | 0.10 | 0.22 | – | – | – |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | – | – | 0.02 | 0.00 | – | – |
| 20,000 | 0.04 | – | – | 0.08 | 0.08 | – | – |
| 30,000 | 0.08 | – | – | 0.04 | 0.14 | – | – |
| 40,000 | 0.08 | – | – | 0.04 | 0.06 | – | – |
| 50,000 | 0.12 | – | – | 0.06 | 0.04 | – | – |
| 60,000 | 0.10 | – | – | 0.10 | 0.16 | – | – |
| 70,000 | 0.08 | – | – | 0.08 | 0.04 | – | – |
| 80,000 | 0.10 | – | – | 0.04 | 0.10 | – | – |

<sub>checkpoints on disk: 8 · complete n-sweeps: 7 · partial: 1 · episodes: [50] · seed: [42]</sub>

## 30 demos - SD VAE, frozen

The same VAE self-frozen, 34,163,664 parameters held out of the optimizer. Against trainable VAE this isolates the FREEZE; against frozen ResNet it is capacity-matched (5.89M vs 5.94M trainable on ST k=1), so that contrast reads on the FEATURES alone.

### UNet BC

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.02 | 0.04 | 0.34 | 0.54 | 0.76 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.22 | 0.44 | 0.70 | 0.86 |
| 30,000 | 0.00 | 0.02 | 0.06 | 0.22 | 0.40 | 0.62 | 0.78 |
| 40,000 | 0.00 | 0.00 | 0.08 | 0.22 | 0.32 | 0.60 | 0.52 |
| 50,000 | 0.00 | 0.02 | 0.08 | 0.14 | 0.36 | 0.30 | – |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 30,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 40,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 50,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 60,000 | 0.02 | – | – | 0.02 | 0.02 | – | – |
| 70,000 | 0.04 | – | – | 0.00 | 0.00 | – | – |
| 80,000 | 0.04 | – | – | 0.00 | 0.00 | – | – |
| 90,000 | 0.06 | – | – | 0.02 | 0.02 | – | – |
| 100,000 | 0.02 | – | – | 0.04 | 0.02 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 4 · partial: 1 · episodes: [50] · seed: [42]</sub>

### ST k=1

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.04 | 0.02 | 0.12 | 0.24 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.06 | 0.08 | 0.16 |
| 30,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.14 | 0.24 |
| 40,000 | 0.00 | 0.00 | 0.02 | 0.00 | 0.10 | 0.20 | 0.26 |
| 50,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.18 | 0.26 |
| 60,000 | 0.00 | 0.00 | 0.00 | 0.06 | 0.08 | 0.08 | 0.12 |
| 70,000 | 0.00 | 0.02 | 0.04 | 0.04 | 0.00 | 0.02 | 0.22 |
| 80,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.08 | 0.08 |
| 90,000 | 0.00 | 0.02 | 0.00 | 0.04 | 0.08 | 0.06 | 0.06 |
| 100,000 | 0.00 | 0.00 | 0.02 | 0.00 | – | – | – |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 30,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 40,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 50,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 60,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 70,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 80,000 | 0.00 | – | – | 0.00 | 0.02 | – | – |
| 90,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 100,000 | 0.00 | – | – | 0.02 | 0.00 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 9 · partial: 1 · episodes: [50] · seed: [42]</sub>

## 126 demos - ResNet18, trainable

ResNet18 IMAGENET1K_V1, `use_group_norm=True`, 76x76 crop, trained end to end. The reference: at 30 demos it reproduces `success_rates_no_pos.md`, which is what validates the revert of the 2026-08-30 speedup pass.

### UNet BC

_no checkpoints evaluated yet_

### ST k=1

_no checkpoints evaluated yet_

## 126 demos - ResNet18, frozen

`training.freeze_encoder=True`, so 11.2M encoder parameters are held out of the optimizer. Against trainable ResNet this asks whether freezing breaks ANY encoder. Caveat: `use_group_norm` replaced the pretrained BatchNorm with freshly built GroupNorm, so this freezes never-trained normalization at identity init and discards ImageNet's running statistics -- a controlled freeze-vs-trainable contrast, not "frozen ImageNet features" in the literature sense.

### UNet BC

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.02 | 0.06 | 0.38 | 0.48 | – |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 30,000 | 0.00 | – | – | – | – | – | – |

<sub>checkpoints on disk: 3 · complete n-sweeps: 0 · partial: 1 · episodes: [50] · seed: [42]</sub>

### ST k=1

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.12 | 0.26 | – |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 30,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 40,000 | 0.00 | – | – | 0.00 | – | – | – |

<sub>checkpoints on disk: 7 · complete n-sweeps: 0 · partial: 1 · episodes: [50] · seed: [42]</sub>

## 126 demos - SD VAE, trainable

`sd-vae-ft-mse` with `trainable=True`, 324-d at the 72x72 crop, trained end to end. Against trainable ResNet this isolates the ENCODER with neither one frozen.

### UNet BC

_no checkpoints evaluated yet_

### ST k=1

_no checkpoints evaluated yet_

## 126 demos - SD VAE, frozen

The same VAE self-frozen, 34,163,664 parameters held out of the optimizer. Against trainable VAE this isolates the FREEZE; against frozen ResNet it is capacity-matched (5.89M vs 5.94M trainable on ST k=1), so that contrast reads on the FEATURES alone.

### UNet BC

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.08 | 0.36 | 0.66 | – |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |

<sub>checkpoints on disk: 2 · complete n-sweeps: 0 · partial: 1 · episodes: [50] · seed: [42]</sub>

### ST k=1

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.04 | 0.06 | – |

**final_pass** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 30,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |

<sub>checkpoints on disk: 3 · complete n-sweeps: 0 · partial: 1 · episodes: [50] · seed: [42]</sub>

## Caveats

**BC is not a like-for-like architecture comparison.** The UNet BC arm optimizes far more parameters than ST k=1 (270M vs 5.9M on the VAE). It shares the encoder, crop and image pipeline, so the observation is matched; the capacity is not.

**BC's crop changed with this generation.** BC now draws one crop offset per SAMPLE, shared across the observation window, as ST always did. The older ResNet runs in `success_rates_no_pos.md` cropped each frame independently, so the BC column here is not expected to match those exactly. ST k=1 is unaffected and is the clean reproduction target.

**`final_pass` is degenerate without a ladder.** It executes the last-generated candidate instead of the verifier's pick, so with i.i.d. candidates it reduces to "sample once, ignore the verifier" — the n=1 argmax number. All six arms here are ladder-free, so every `final_pass` table is that control.

