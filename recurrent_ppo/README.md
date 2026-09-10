# Recurrent PPO on PushT

A recurrent policy and value function trained with `sb3_contrib.RecurrentPPO`, in two
observation arms and two noise regimes.

```
train.py                       argparse -> vec env -> RecurrentPPO.learn, plus its three callbacks
play.py                        checkpoint -> LSTM-stateful rollout -> success rate, score, video
pusht_gym.py                   gym-0.21 PushT -> gymnasium: obs, actions, rewards, occlusion
corrupt_policy.py              the corrupting features extractor, ST's ResNet, and the Q head
run_io.py                      run-directory bookkeeping, shared by train.py and play.py
scripts/plot_start_states.py   where episodes start, and how far one action moves the agent
scripts/render_noise_levels.py what each corruption level actually destroys, decoded to images
```

`pusht_gym.py` and `corrupt_policy.py` are each imported by both entry points, which is the
whole reason they are not inside one of them: play.py must rebuild bit-for-bit the environment
and policy that train.py built, and a wrapper that drifted between the two would silently change
what an evaluation measures. `run_io.py` is there for the same reason -- it holds the four things
both scripts need to agree on (what `params/args.yaml` contains, how a checkpoint is found, where
the VecNormalize pickle sits, which arguments a resume may not change) and nothing else. The
callbacks, by contrast, have exactly one caller, so they live in train.py.

## Running

Everything runs from the repo root in `robodiff2` (`sb3_contrib` was added to it; it brought
in only `gymnasium`, `stable_baselines3` and `farama-notifications`).

```bash
# phase 1 -- clean observations
python recurrent_ppo/train.py --obs keypoint
python recurrent_ppo/train.py --obs image --lstm-hidden-size 256

# phase 2 -- noised observations
python recurrent_ppo/train.py --obs keypoint --corrupt-obs

# occlusion, an axis independent of the DDPM corruption
python recurrent_ppo/train.py --obs keypoint --keypoint-visible-rate 0.5

# evaluate the newest run of an arm; --arm picks the DIRECTORY, it does not corrupt anything
python recurrent_ppo/play.py --n-episodes 50
python recurrent_ppo/play.py --arm corrupt --corrupt-obs-eval
python recurrent_ppo/play.py --checkpoint logs/.../model.zip --video --render-size 512
```

Runs land in `logs/recurrent_ppo/<obs>_<clean|corrupt>/<timestamp>/`. The noise regime is part
of the directory name because it is part of the run's identity, not a knob: two regimes sharing
a directory would overwrite each other's checkpoints. `--checkpoint` resumes **in the
checkpoint's own directory**, continuing its step counter, and refuses any flag that would
change what the checkpoint is.

play.py reads the run's `params/args.yaml` for how the environment was shaped, so there is
nothing to retype; a CLI flag still overrides, and says so when it does. `--render-size` is the
image obs resolution as well as the video resolution, so for the `image` arm it must match what
the checkpoint trained at — the keypoint arm's observation does not depend on it, so raise it
there for a legible video.

## The reward, and why termination is tied to it

`--reward dense` (default) pays `coverage / 0.95` every step and **never terminates** — the
episode always runs to `--max-episode-steps` and truncates, so it bootstraps. `--reward sparse`
pays `+1` on the first solve and terminates there.

They are not interchangeable settings on one MDP; each termination rule is the one that makes its
reward self-consistent. PushT's reward saturates at 1.0 when coverage hits 0.95, while `done`
requires coverage **strictly above** it — so terminating on success under the dense reward forfeits
a stream worth 1.0/step. Measured at gamma=0.99 over 300 steps, solving at step 20 returned 9.2
while parking at 95% coverage returned 85.2: the old setup paid the agent *not* to finish, and paid
it most when it could finish fastest. Under both modes now, solving earlier scores strictly higher.

`is_success` is sticky in both modes (the task can be solved without the episode ending), and
`max_reward` always reports the env's own dense coverage, so the episode score stays one
comparable quantity across modes — the same one `pusht_image_runner` reports.

## Occlusion: two modes, one mechanism

`--occlusion iid` (default) redraws each keypoint's visibility every step; `--occlusion persistent`
hides it for a stretch of `--occlusion-persistence` steps. Both are the *same* two-state Markov
chain, parameterised so the stationary visible fraction equals `--keypoint-visible-rate` in either
mode — so the arms are matched on **how much** is missing and differ only in **how it is
distributed in time**. Measured at rate 0.5: visible fraction 0.498 vs 0.507, mean hidden run 2.01
vs 19.75 steps.

That contrast is the point. Under `iid` a 4-frame average recovers nearly everything, so memory
buys nothing; under `persistent` the agent must carry a belief and propagate it through its own
actions. An LSTM should beat a memoryless baseline on one and tie it on the other.

The chain lives in `PushTGymEnv`, and the inner `PushTKeypointsEnv` is pinned to full visibility —
that env is shared with the diffusion-policy training and must not change behaviour.

## Conventions worth knowing

**Everything the policy sees is in `[-1, 1]`.** Positions are scaled against the arena's known
bounds, the keypoint visibility mask is mapped to `{-1, +1}` rather than left at `{0, 1}`, and
the image is uint8 `[0, 255]` for the single reason that SB3's `NatureCNN` then does the `/255`
itself. Bounds-based scaling, not `VecNormalize`: the ranges are constants, so there is nothing
to estimate and no running statistics to keep in sync between training and play.

**Corruption is scaled and capped.** The noise is multiplied by a running per-dimension feature
std, so `sqrt(alpha_bar_t)` denotes a fixed SNR instead of riding on the encoder's magnitude —
verified invariant across feature stds spanning 0.03 to 9188. Unscaled it is not a controlled
variable at all: across one ST run's checkpoints the feature scale climbed 30 -> 9188, so a fixed
`t` annealed itself from real corruption to none. Timesteps are drawn from `U[0, 200)`, chosen by
looking: `scripts/render_noise_levels.py` decodes the corruption through the frozen SD VAE, and
the block's *orientation* stops being readable around t=100. The old `U[0, 1000)` put 90% of draws
past that — an absent observation, not a degraded one.

**Evaluation is clean by default, in both arms.** Every reported number — `eval/` during
training and play.py's — is a clean-observation number, so the arms differ only in how they
trained. `play.py --corrupt-obs-eval` opts back in.

**Episodes start where the demonstrations start.** `PushTEnv` draws the block centre from
[100, 400], which excludes a quarter of the block starts in `data/pusht_cchi_v7_replay.zarr`
(span x 66-440, y 116-486). The adapter draws the initial state itself and hands it over as
`reset_to_state`, from [60, 490], and redraws when the T's arms would spawn through a wall
(~74% of draws are accepted, so about 1.4 resets). All 206 demonstrated starts now lie inside
the sampled region. `--agent-start-range` and `--block-start-range` set it, and both are part of
a run's identity.

**Actions are offsets, not destinations.** `--action-mode delta` (the default) makes the action
a bounded displacement of the agent, `--delta-scale` pixels per axis at `|a| = 1`. It defaults to
`auto`, which measures the p99 of the human per-axis target step in
`data/pusht_cchi_v7_replay.zarr` (33 px, against a median of 4) and records the resolved number
in `params/args.yaml`, so play.py reproduces it without re-reading the dataset. There is no
physical scale to derive it from: PushT's PD controller is perfectly linear, covering 29.6% of
whatever move you request with no saturation, so "how far is one action" is a question only the
demonstrations answer. `--delta-percentile` moves the choice; a plain number bypasses it. Under
`--action-mode absolute` the action is a target anywhere in the arena, so a Gaussian at std 1
explores with a standard deviation of 256 px — half the table, ~32x the median human step.
`scripts/plot_start_states.py` draws exactly this comparison.

## What SB3 does not provide

Everything below had to be written; the rest (PPO, the LSTM policy, rollout buffers,
`VecNormalize`, `Monitor`, checkpoint/eval callbacks, tensorboard) is imported.

**1. A gymnasium PushT** (`pusht_gym.py`). `PushTEnv` is gym 0.21 and SB3 2.x is gymnasium: a
4-tuple `step` with one `done` flag, a `reset` that takes no seed, `render(mode)` with no
default, gym rather than gymnasium `spaces`, and no time limit.

**2. Per-episode initial states.** `PushTEnv.reset()` re-derives its state from
`RandomState(self._seed)` and never advances it, so every `reset()` returns *the same* episode.
The adapter draws the state itself instead, which also buys the wider start region above.
Without this the agent trains on one episode and the run looks like it is learning.

**3. terminated vs truncated.** PushT's `done` is coverage > 0.95, i.e. the task is solved — a
real termination. Running out of steps is a truncation and PushT has no time limit to say so, so
the adapter keeps the step count and reports the two separately. Conflating them teaches the
critic that hitting the step budget ends the world.

**4. Occlusion the policy can see.** `PushTKeypointsEnv` reports the **true** keypoints followed
by a visibility mask, and zeroes the occluded ones only in its *rendering* copy — so consuming
the first half alone, as the lowdim policies do, makes `keypoint_visible_rate` a no-op. The
adapter zeroes the occluded entries and keeps the mask, giving a 40-d observation in which
"hidden" is distinguishable from "at the arena centre".

**5. Delta actions.** The action space is an absolute PD-controller target in `[0, 512]`; the
useful target is a waypoint near the agent. See the conventions above.

**6. Episode metrics.** SB3 logs return and length. `is_success` (task solved) and `max_reward`
(the repo's episode score: max normalised coverage, as in `pusht_image_runner`) are added to
`info` so `rollout/success_rate` is logged and play.py can report the score.

**7. The observation corruption** (`corrupt_policy.py`). The noise itself is the repo's —
`ObsCorruptionMixin.corrupt_obs_features`, the flat `corrupt_obs` arm, under the same DDPM
scheduler as `config/train_pusht_diffusion_search.yaml:339`, so a level here means what it means
in ST. What is new is *where* it is injected. It has to act on the encoded vector, and SB3 reads
that vector at four call sites, two of which (`get_distribution`, `predict_values`) deliberately
bypass `ActorCriticPolicy.extract_features`. Overriding that one method would corrupt the
gradient and the rollout while leaving the value bootstrap and the play rollout clean. So the
features extractor itself is wrapped, which covers every path without copying SB3 internals.

**8. Action-clipping diagnostics** (`ActionDiagnostics` in train.py). PPO stores the *unclipped* Gaussian samples
in the rollout buffer and clips only on the way into the env, so the buffer is the one place the
raw exploration distribution is visible. `rollout/action_clip_frac`, `action_corner_frac` and
`action_abs_mean` are logged from it every rollout.

**9. A stateful rollout** (`play.py`). `predict` needs the previous hidden state and an
episode_start mask threaded by hand, and because a vec env auto-resets, `dones` from step t is
the mask for step t+1. Omitting it runs a memoryless policy that still looks like it works.
Episodes are also counted on a **per-env budget**: stopping at the first N episodes across all
envs over-samples the short ones, and success is exactly what ends an episode early, so a global
count reports a success rate biased upward.

## Two things to hold in mind

* **`corrupt_obs_eval` is pinned True in the policy.** SB3 flips `policy.training` between
  rollout collection (False) and the update epochs (True), so the mixin's usual train/eval gate
  would corrupt only the gradient and leave the behaviour policy clean. Whether a given rollout
  is corrupted is decided by toggling `policy.corrupt_obs` instead — which is what
  `CleanEvalCallback` and `play.py --corrupt-obs-eval` do.
* **The noise is redrawn on every forward pass**, so the action sampled during collection and the
  log-prob recomputed during the epochs sit at different draws, and PPO's importance ratio mixes
  the two. This is what ST does, but ST is BC and has no ratio to bias; here it is the
  dropout-in-PPO situation. Making it exact means replaying the per-step draw out of the rollout
  buffer, i.e. a custom `RecurrentRolloutBuffer` — not a knob.

## Not built

* **The per-slot noise ladder** (`slot_obs_noise`). PPO has no candidate slots, so the ladder
  needs an axis chosen for it — episode step (LSTM context length) is the direct analogue of
  "slot k has k context entries". Flat corruption only, for now.
* **A Beta action distribution**, gated on what the diagnostics show. At `--log-std-init 0` about
  31% of action components are clipped and 12% of actions are full corners; at
  `--log-std-init -1` both are **zero**, so the std, not the distribution family, is what drives
  clipping at initialisation. If `action_corner_frac` climbs during training while returns
  plateau, that is the signal to replace the Gaussian.
