# PushT encoder debug — success rates

_Generated 2026-08-30 by `scripts/build_vae_debug_doc.py`. Re-run to refresh._

30 demos, seed 42, verifier `t_goal`, 50 test episodes, image-only observations, 100k gradient steps, checkpoint every 10k. **No observation corruption on any of these six** — `slot_obs_noise` is uniform, which leaves `slot_obs_t` None so the corruption is the identity, and `corrupt_obs` is false.

Three encoders x two arms. ResNet18 tests the revert of the 2026-08-30 speedup pass; ResNet vs frozen VAE isolates the encoder; frozen vs trainable VAE isolates the freeze.

`argmax` sweeps n = 1..64; `final_pass` was asked for at n = 1, 8, 16 only, so its other columns are blank by design. Blank also means "not yet evaluated". No cell is a nominated best.

## ResNet18 (reference)

ResNet18 IMAGENET1K_V1, `use_group_norm=True`, 76x76 crop, trained end to end. The target: reproduces `success_rates_no_pos.md` and so tests the speedup revert.

### UNet BC

_no checkpoints evaluated yet_

### ST k=1

_no checkpoints evaluated yet_

## SD VAE, frozen

Frozen `sd-vae-ft-mse`, 324-d at the 72x72 crop, 34,163,664 parameters held out of the optimizer. Against ResNet18 this isolates the ENCODER.

### UNet BC

**argmax** — test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.04 | 0.04 | 0.04 | 0.08 | 0.16 | 0.14 | – |

<sub>checkpoints on disk: 0 · complete n-sweeps: 0 · partial: 1 · episodes: [50] · seed: [42]</sub>

### ST k=1

_no checkpoints evaluated yet_

## SD VAE, trainable

The same encoder, `trainable=True`, trained end to end. Against the frozen column this isolates the FREEZE. A debug configuration only: a drifting encoder stops producing SD latents, which is what the corruption ladder and the latent decoder both rest on.

### UNet BC

_no checkpoints evaluated yet_

### ST k=1

_no checkpoints evaluated yet_

## Caveats

**BC is not a like-for-like architecture comparison.** The UNet BC arm optimizes far more parameters than ST k=1 (270M vs 5.9M on the VAE). It shares the encoder, crop and image pipeline, so the observation is matched; the capacity is not.

**BC's crop changed with this generation.** BC now draws one crop offset per SAMPLE, shared across the observation window, as ST always did. The older ResNet runs in `success_rates_no_pos.md` cropped each frame independently, so the BC column here is not expected to match those exactly. ST k=1 is unaffected and is the clean reproduction target.

**`final_pass` is degenerate without a ladder.** It executes the last-generated candidate instead of the verifier's pick, so with i.i.d. candidates it reduces to "sample once, ignore the verifier" — the n=1 argmax number. All six arms here are ladder-free, so every `final_pass` table is that control.

