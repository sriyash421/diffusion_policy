# PushT VAE + TMRL obs-corruption ladder — success rates

_Generated 2026-08-30 by `scripts/build_vae_tmrl_doc.py`. Re-run to refresh._

30 demos, seed 42, verifier `t_goal`, 50 test episodes, 100k gradient steps,
checkpoint every 10k. Image-only observations (no `agent_pos`, no `feedback`)
encoded by a FROZEN Stable Diffusion VAE (`sd-vae-ft-mse`, 324-d at the 72x72
crop). Every arm reports 34,163,664 frozen parameters.

## The ladder

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
`linear_t` at 999 is milder but still lopsided, with four low slots near-identical noise.

## Reading the tables

`argmax` sweeps n = 1..64; `final_pass` was asked for at n = 1, 8, 16 only, so its
other columns are blank by design rather than missing. Blank also means "not yet
evaluated" -- the sweep is still running. No cell is a nominated best.

## UNet BC

the diffusion-policy UNet; ranks n i.i.d. draws OUTSIDE the model

**argmax, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.06 | 0.30 | 0.56 | 0.76 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.10 | 0.46 | 0.68 | 0.86 |
| 30,000 | 0.00 | 0.00 | 0.04 | 0.20 | 0.30 | 0.52 | 0.68 |
| 40,000 | 0.00 | 0.00 | 0.12 | 0.16 | 0.30 | 0.38 | 0.68 |
| 50,000 | 0.02 | 0.04 | 0.04 | 0.16 | 0.30 | 0.26 | 0.28 |
| 60,000 | 0.02 | 0.10 | 0.10 | 0.16 | 0.20 | 0.30 | 0.18 |
| 70,000 | 0.04 | 0.04 | 0.12 | 0.08 | 0.20 | 0.20 | 0.20 |
| 80,000 | 0.00 | 0.06 | 0.08 | 0.08 | 0.16 | 0.08 | 0.28 |
| 90,000 | 0.02 | 0.10 | 0.04 | 0.14 | – | – | – |

**final_pass, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 30,000 | 0.00 | – | – | 0.02 | 0.00 | – | – |
| 40,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 50,000 | 0.02 | – | – | 0.00 | 0.00 | – | – |
| 60,000 | 0.02 | – | – | 0.02 | 0.04 | – | – |
| 70,000 | 0.04 | – | – | 0.04 | 0.04 | – | – |
| 80,000 | 0.00 | – | – | 0.04 | 0.00 | – | – |
| 90,000 | 0.02 | – | – | 0.04 | 0.04 | – | – |
| 100,000 | 0.04 | – | – | 0.04 | 0.04 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 8 · partial: 1 · episodes: [50] · seed: [42]</sub>

## ST k=1

width 1: no search context

**argmax, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.04 | 0.02 | 0.12 | 0.24 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.06 | 0.08 | 0.16 |
| 30,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.12 | 0.24 |
| 40,000 | 0.00 | 0.00 | 0.02 | 0.00 | 0.10 | 0.20 | 0.22 |
| 50,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.06 | 0.16 | 0.28 |
| 60,000 | 0.00 | 0.00 | 0.00 | 0.04 | 0.08 | 0.10 | 0.14 |
| 70,000 | 0.00 | 0.02 | 0.04 | 0.04 | 0.08 | 0.12 | 0.14 |
| 80,000 | 0.00 | 0.00 | 0.04 | 0.04 | 0.04 | 0.10 | 0.08 |
| 90,000 | 0.00 | 0.02 | 0.02 | 0.02 | 0.02 | 0.04 | 0.12 |
| 100,000 | 0.00 | 0.02 | 0.04 | 0.00 | 0.12 | 0.16 | 0.12 |

**final_pass, obs clean** — test success rate

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
| 100,000 | 0.00 | – | – | 0.00 | 0.02 | – | – |

<sub>checkpoints on disk: 10 · complete n-sweeps: 10 · partial: 0 · episodes: [50] · seed: [42]</sub>

## ST k=16 — uniform

uniform slot weights, no obs ladder — the control for every arm below

**argmax, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.06 | 0.34 | 0.46 | 0.64 |
| 20,000 | 0.00 | 0.00 | 0.12 | 0.26 | 0.24 | 0.40 | 0.48 |

**final_pass, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |

<sub>checkpoints on disk: 2 · complete n-sweeps: 2 · partial: 0 · episodes: [50] · seed: [42]</sub>

## ST k=16 — linear in t, slot0=999

`slot_obs_noise: linear_t`

**argmax, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.34 | 0.62 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.04 | 0.22 | 0.50 | 0.62 |

**argmax, obs corrupt** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.30 | 0.72 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.06 | 0.38 | 0.84 |

**final_pass, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.02 | – | – |

**final_pass, obs corrupt** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |

<sub>checkpoints on disk: 2 · complete n-sweeps: 2 · partial: 0 · episodes: [50] · seed: [42]</sub>

## ST k=16 — linear in t, slot0=400

`slot_obs_noise: linear_t, max_t 400`

**argmax, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.08 | 0.32 | 0.58 | 0.76 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.18 | 0.46 | 0.42 | 0.60 |

**argmax, obs corrupt** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.22 | 0.52 | 0.76 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.02 | 0.24 | 0.64 | 0.74 |

**final_pass, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 | 0.00 | – | – | 0.00 | 0.04 | – | – |

**final_pass, obs corrupt** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |

<sub>checkpoints on disk: 2 · complete n-sweeps: 2 · partial: 0 · episodes: [50] · seed: [42]</sub>

## ST k=16 — linear in signal

`slot_obs_noise: linear_signal` — even in sqrt(alpha_bar)

**argmax, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.20 | 0.50 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.16 | 0.30 | 0.56 |

**argmax, obs corrupt** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.36 | 0.78 |
| 20,000 | 0.00 | 0.00 | 0.02 | 0.04 | 0.04 | 0.46 | 0.70 |

**final_pass, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |

**final_pass, obs corrupt** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |

<sub>checkpoints on disk: 2 · complete n-sweeps: 2 · partial: 0 · episodes: [50] · seed: [42]</sub>

## ST k=16 — geometric d=0.7

`slot_obs_noise: geometric, decay 0.7`

**argmax, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.26 | 0.44 | 0.46 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.12 | 0.20 | 0.44 | 0.46 |
| 30,000 | 0.00 | 0.00 | 0.00 | 0.14 | 0.36 | – | – |

**argmax, obs corrupt** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.00 | 0.04 | 0.30 | 0.54 | 0.76 |
| 20,000 | 0.00 | 0.00 | 0.00 | 0.06 | 0.22 | 0.54 | 0.78 |
| 30,000 | 0.00 | 0.00 | 0.00 | 0.04 | 0.26 | 0.40 | – |

**final_pass, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 | 0.00 | – | – | 0.00 | 0.02 | – | – |
| 30,000 | 0.00 | – | – | 0.00 | 0.06 | – | – |

**final_pass, obs corrupt** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 30,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |

<sub>checkpoints on disk: 3 · complete n-sweeps: 2 · partial: 1 · episodes: [50] · seed: [42]</sub>

## ST k=16 — random base -> linear in signal

`slot_obs_noise: random_base, shape linear_signal, base_range [0, 999]`

**argmax, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.00 | 0.02 | 0.18 | 0.34 | 0.54 | 0.76 |
| 20,000 | 0.00 | 0.00 | 0.04 | 0.28 | 0.24 | 0.32 | 0.36 |

**final_pass, obs clean** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | – | – | 0.00 | 0.00 | – | – |
| 20,000 | 0.00 | – | – | 0.04 | 0.02 | – | – |

<sub>checkpoints on disk: 2 · complete n-sweeps: 2 · partial: 0 · episodes: [50] · seed: [42]</sub>

## Caveats

**BC is not a like-for-like architecture comparison.** The UNet BC arm optimizes
270,370,562 parameters against the ST arms' 5,896,194 — a 46x gap. It shares the
encoder, crop and image pipeline with the ST arms, so the observation is matched,
but the capacity is not.

**`final_pass` is degenerate on the no-ladder arms.** It executes the
last-generated candidate instead of the verifier's pick, so without a ladder the
candidates are i.i.d. draws and it reduces to "sample once, ignore the verifier" —
the n=1 argmax number. It only becomes meaningful on a ladder arm, where the last
slot is the CLEANEST observation rather than an arbitrary draw. The baseline
`final_pass` rows are the control those are read against.

**Image-only removed the block pose from the OBSERVATION, not from the model's
inputs.** Under `search_context: value` slot k>0 still sees k verifier scalars,
each a goal distance computed by the real pymunk simulator from the state that
candidate actually reached. Slot 0 sees only the (corrupted) image. So an ST arm
beating BC is not purely architectural: its inputs include a simulator-derived
signal BC's do not.

**The ladder is not the textbook DDPM forward process.** It rescales the noise
per dimension by a running std of the encoded features
(`eps = randn_like(x) * obs_feature_std`), because the VAE latent's per-dimension
std spans 0.026 to 0.990 and unit noise would obliterate the narrow dimensions
while barely touching the wide ones. That rescale is what makes sqrt(alpha_bar)
an SNR. "We corrupt in the space diffusion models operate in" is a claim about
the SPACE, not the OPERATOR.

**The flat `corrupt_obs` arm uses a different operator**, not the same operator on
a different schedule. Do not present the two as a schedule ablation.

