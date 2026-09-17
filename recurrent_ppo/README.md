# PPO on PushT, with and without recurrence

A policy, a value function and a Q head trained on PushT, in three observation arms and two
noise regimes, under **two architectures**: an LSTM (`sb3_contrib.RecurrentPPO`) and frame stacking
(`stable_baselines3.PPO` + `VecFrameStack`). sb3-contrib's own RecurrentPPO documentation
recommends trying the second first -- "a simpler, faster and usually competitive alternative" --
so both are here and they share everything except the architecture.

```
train.py                       LSTM entry point: flags -> runner.train(args, LstmArch)
play.py                        LSTM entry point: flags -> runner.play(args, LstmArch)
ppo/train.py                   frame-stacking entry point, plus --n-stack
ppo/play.py                    frame-stacking entry point
arch.py                        LstmArch / StackArch: the FOUR things that differ
runner.py                      the training and evaluation loops, shared
cli.py                         every flag that is not the architecture, shared
callbacks.py                   action diagnostics, Q head, dual eval, rollout video, clean eval
pusht_gym.py                   gym-0.21 PushT -> gymnasium: obs, actions, rewards, occlusion
corrupt_policy.py              the corrupting features extractor, ST's ResNet, and the Q head
config.py                      every default, in one place
run_io.py                      run-directory bookkeeping
scripts/plot_start_states.py   where episodes start, and how far one action moves the agent
scripts/render_noise_levels.py what each corruption level actually destroys, decoded to images
```

The split is driven by one thing: **the two architectures exist to be compared.** Anything
duplicated between them -- the reward, the curricula, the evaluation protocol, the seed offsets,
the run bookkeeping -- is a place where the comparison can quietly stop being like-for-like, so
none of it is duplicated. The entry points are ~20 lines each and differ only in which `arch`
object they pass. `arch.py` lists exactly what an architecture gets to decide: how the vec env is
wrapped, how the agent is built, how it is loaded, and how many frames the video rollout has to
stack by hand.

The same argument is why `pusht_gym.py`, `corrupt_policy.py` and `run_io.py` were already
separate: play.py must rebuild bit-for-bit the environment and policy that train.py built, and a
wrapper that drifted between the two would silently change what an evaluation measures.

### What frame stacking can and cannot do here

It reconstructs state hidden from a single observation -- velocity, in the report sb3-contrib
cites. **Our observation is currently missing none.** Corruption is off by default, occlusion is
off by default, and PushT sets `space.damping = 0`, so the block carries no momentum between
steps. On the clean arm the LSTM, `--n-stack 4` and `--n-stack 1` are all looking at a Markov
observation, and any difference between them is an optimisation effect rather than a
representational one. `--n-stack 1` is therefore the honest plain-PPO baseline, and the
architecture comparison only becomes meaningful once `--keypoint-visible-rate` or
`--corrupt-obs` is on.

On the image arm the stack is encoded frame by frame, not all at once: `VecFrameStack`
concatenates on the CHANNEL axis, so a 4-stack arrives as `(B, 12, H, W)` and resnet18's
pretrained `conv1` takes 3. `STResNetExtractor` reshapes to `(B*4, 3, H, W)`, runs the ONE
shared encoder, and concatenates the 512-d results, exactly as ST handles its `n_obs_steps`.
Widening `conv1` to 12 channels instead would discard the pretrained kernel, which is the whole
reason this encoder matches ST's -- and for the same reason `arch.py` turns SB3's `ortho_init`
off on this arm. The crop offset is drawn once per transition and repeated across the frames,
so one observation replays identically.

## Running

Everything runs from the repo root in the `robodiff` conda env. `sb3_contrib` was added to it
with `pip install sb3_contrib==2.7.1`, which brought in only `stable_baselines3==2.7.1`,
`gymnasium==1.1.1` and `farama-notifications` — nothing already installed was upgraded or
removed, and gymnasium coexists with the `gym==0.21` the rest of the repo uses.

```bash
# phase 1 -- clean observations
python -m recurrent_ppo.train --obs keypoint
python -m recurrent_ppo.train --obs state                    # 6-d: agent xy, block xy, cos/sin
python -m recurrent_ppo.train --obs image --lstm-hidden-size 256

# phase 2 -- noised observations
python -m recurrent_ppo.train --obs keypoint --corrupt-obs

# occlusion, an axis independent of the DDPM corruption
python -m recurrent_ppo.train --obs keypoint --keypoint-visible-rate 0.5

# arm 2's actual answer: the same checkpoint scored clean and corrupted
python -m recurrent_ppo.train --obs keypoint --corrupt-obs --eval-corrupt

# the frame-stacking arm -- same flags, plus --n-stack
python -m recurrent_ppo.ppo.train --obs keypoint --n-stack 4
python -m recurrent_ppo.ppo.train --obs keypoint --n-stack 1     # plain PPO, no stacking

# evaluate the newest run of an arm; --arm picks the DIRECTORY, it does not corrupt anything
python -m recurrent_ppo.play --n-episodes 50
python -m recurrent_ppo.play --arm corrupt --corrupt-obs-eval
python -m recurrent_ppo.play --checkpoint logs/.../model.zip --video --render-size 512
python -m recurrent_ppo.ppo.play --checkpoint logs/ppo/.../model.zip
```

`--eval-freq` defaults to 100k steps, so an `eval/` series exists without asking. `--eval-corrupt`
adds `eval_corrupt/` alongside it — for the `--corrupt-obs` arm that is the number that says
whether the POMDP was solved, and the clean series stays beside it so the two are comparable.
Both re-seed the eval env before every evaluation, from a seed block disjoint from training's:
SB3 hands seeds over at the next reset and then clears them, so without that each eval point
draws different episodes and the curve mixes policy improvement with episode luck — and the two
series would not be scoring the same task.

Runs land in `logs/recurrent_ppo/<obs>_<clean|corrupt>/<timestamp>/` for the LSTM arm and
`logs/ppo/...` for the frame-stacking arm -- the architecture is part of a run's identity, like
the noise regime, so the two cannot collide. The noise regime is part
of the directory name because it is part of the run's identity, not a knob: two regimes sharing
a directory would overwrite each other's checkpoints. `--checkpoint` resumes **in the
checkpoint's own directory**, continuing its step counter, and refuses any flag that would
change what the checkpoint is.

play.py reads the run's `params/args.yaml` for how the environment was shaped, so there is
nothing to retype; a CLI flag still overrides, and says so when it does. `--render-size` is the
image obs resolution as well as the video resolution, so for the `image` arm it must match what
the checkpoint trained at — the keypoint arm's observation does not depend on it, so raise it
there for a legible video.

## The three observation arms

| `--obs` | what the policy sees | dim |
|---|---|---|
| `keypoint` (default) | 9 block-T keypoints + agent xy, then their visibility mask as {-1,+1} | 40 |
| `state` | agent xy, block xy, block angle as (cos, sin) | 6 |
| `image` | `{image (3,96,96) uint8}` -- **image only**, see below | dict |

`state` is `PushTEnv`'s own `_get_obs` with one change: the angle arrives as `block.angle % 2*pi`,
so a scalar encoding breaks at the wrap -- 0.01 and 6.27 rad are the same pose but land at
opposite ends of [-1,1], the furthest apart two values can be. (cos, sin) is continuous
everywhere for one extra dimension. The keypoint arm never had this problem because 9 points
encode rotation continuously by construction; that is plausibly a large part of why the repo's
lowdim task uses them.

Occlusion (`--keypoint-visible-rate`, `--occlusion`) applies to the **keypoint arm only** -- it
is defined per keypoint, and there is no corresponding notion for a 6-d pose. The DDPM
corruption (`--corrupt-obs`) applies to all three, since it acts on the encoded features.

### The image arm is image only

`agent_pos` used to be concatenated onto the encoder's 512 features, making the image arm
514-d. It is gone. `task/pusht_image_search_imgonly.yaml` is image-only *by construction* --
the diffusion-policy arms assert that no low_dim key is declared, because `feedback` is an
invertible transform of the block pose and handing the policy the T's pose in closed form is a
strictly stronger observation than the standard PushT-image setup. Keeping `agent_pos` here
gave the RL arms a privilege the offline arms are forbidden, so the two families' success rates
could not be compared -- which is the only reason `STResNetExtractor` mirrors that encoder at
all. Three places encoded the old width and all three changed together:
`PushTGymEnv.observation_space`, `_convert_obs`, and `aug_for`'s `feature_dim` (`512 *
n_stack`, with nothing concatenated alongside: the stack is encoded frame by frame through the
one shared ResNet). An earlier image generation predates this and was deleted with the
rest of that grid on 2026-09-16.


## The reward, and why termination is tied to it

`--reward delta` (default) pays for CHANGE rather than level: `progress_coef` times the
per-step reduction in the T's distance from the goal pose, plus `success_bonus` once on the
first solve. Summed over an episode it telescopes to total progress, so it cannot be farmed by
loitering — which is why it is the default. Under a level-valued reward a lucky reset paid a
do-nothing policy 92.7, more than any policy earned by acting.

`--reward dense` pays `coverage / 0.95` every step and **never terminates** — the
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
the image is uint8 `[0, 255]` for the single reason that SB3's `preprocess_obs` then does the
`/255` itself, and the rollout buffer is 4x smaller for it. (The image arm's encoder is
`STResNetExtractor`, never `NatureCNN`.) Bounds-based scaling, not `VecNormalize`: the ranges
are constants, so there is nothing to estimate and no running statistics to keep in sync between
training and play.

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
`recurrent_ppo/scripts/plot_start_states.py` draws exactly this comparison.

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

**6. Reading V and Q out.** Both heads are fitted to `VecNormalize`-scaled returns, so their
outputs are in units of `reward / sqrt(ret_rms.var)` and **mean nothing until that factor is put
back** — which is how a working critic looks broken. Any readout must de-normalise through the
saved `model_vecnormalize.pkl`. The critic's LSTM state also has to be threaded by hand:
`predict` returns only the actor's, so one `policy.forward` has to drive the action and `V`
together, and `Q` is asked about the **clipped** action, the one the env actually executed.

There was once a `play.py --report-values` flag doing this inline; it was lost in the arm merge
and is not coming back. V is now needed at *arbitrary* states — the state a candidate chunk
reaches — rather than along a rollout, so it lives in its own scoring module that the best-of-N
verifier imports.

**7. Episode metrics.** SB3 logs return and length. `is_success` (task solved) and `max_reward`
(the repo's episode score: max normalised coverage, as in `pusht_image_runner`) are added to
`info` so `rollout/success_rate` is logged and play.py can report the score.

**8. The observation corruption** (`corrupt_policy.py`). The noise itself is the repo's —
`ObsCorruptionMixin.corrupt_obs_features`, the flat `corrupt_obs` arm, under the same DDPM
scheduler as `config/train_pusht_diffusion_search.yaml:339`, so a level here means what it means
in ST. What is new is *where* it is injected. It has to act on the encoded vector, and SB3 reads
that vector at four call sites, two of which (`get_distribution`, `predict_values`) deliberately
bypass `ActorCriticPolicy.extract_features`. Overriding that one method would corrupt the
gradient and the rollout while leaving the value bootstrap and the play rollout clean. So the
features extractor itself is wrapped, which covers every path without copying SB3 internals.

**8. Action-clipping diagnostics** (`ActionDiagnostics` in callbacks.py). PPO stores the *unclipped* Gaussian samples
in the rollout buffer and clips only on the way into the env, so the buffer is the one place the
raw exploration distribution is visible. `rollout/action_clip_frac`, `action_corner_frac` and
`action_abs_mean` are logged from it every rollout.

**10. A stateful rollout** (`play.py`). `predict` needs the previous hidden state and an
episode_start mask threaded by hand, and because a vec env auto-resets, `dones` from step t is
the mask for step t+1. Omitting it runs a memoryless policy that still looks like it works.
Episodes are also counted on a **per-env budget**: stopping at the first N episodes across all
envs over-samples the short ones, and success is exactly what ends an episode early, so a global
count reports a success rate biased upward.

## The draw rides in the observation, and why

SB3 flips `policy.training` between rollout collection (`False`) and the update epochs (`True`),
and PPO re-encodes every stored transition once per `--n-epochs`. Anything random inside the
features extractor is therefore **a different draw on every epoch**, so
`ratio = exp(log_prob - old_log_prob)` compares `pi(a | draw 2)` against `pi_old(a | draw 1)` and
stops being a ratio of policies at all. Two things were doing this:

* the DDPM corruption noise, and
* the image arm's `CropRandomizer`, which switches on `self.training` — so the **clean** image
  arm was biased too, not only the corrupted one.

`AugmentationDraw` (in `pusht_gym.py`) fixes the noise by drawing the vector and its timestep
**once per environment step** and carrying them in the observation. The observation is the only
per-transition channel SB3 has into a features extractor, so that is what puts the draw in the
rollout buffer, where every epoch reads back the same numbers. `CorruptingExtractor` then reads
the noise instead of drawing it. No SB3 internals are copied and no buffer is subclassed.

**The crop is answered differently, and more cheaply: this arm does not draw one.**
`STResNetExtractor` centre-crops whenever no offset is supplied, and `aug_for` supplies none for
PPO — so the transform is deterministic, identical in both phases, and there is nothing a mode
flip could change. A deterministic crop also makes eval true to training for free, which is what
the earlier design bought with machinery. `self.training` is never consulted on this arm, in
either direction.

The `_forced_offsets` path it used to rely on is still there and still load-bearing, just not
for PPO: **SAC** wraps its env in `VecAugmentationDraw` to keep random-crop augmentation across
its off-policy updates, and `sac.score.obs_for_arm` supplies the centre offset explicitly so the
learned Q is deployed on a centre crop while having trained on random ones.

`test_evaluate_actions_reproduces_the_collected_log_prob` asserts the ratio is exactly 1 on the
first epoch for the corrupted keypoint arm and both image arms;
`test_the_ppo_image_arm_centre_crops_in_both_modes` pins the crop half.

One consequence: the corrupted keypoint arm's observation is a `Dict` rather than a `Box`, so it
uses the MultiInput policy. `policy_for` dispatches on it.

**The feature-std EMA is frozen for the update.** The noise is scaled by a running per-dimension
std, and that std must track the encoder's drift across rollouts while holding still *within* one
update — a std that moved between epochs would re-scale an already-stored transition's noise and
undo the replay. `self.training` is exactly backwards for gating this, so `FeatureStdWindow` opens
the EMA at `on_rollout_start` and closes it at `on_rollout_end`.

## Two things to hold in mind

* **Corruption is switched with `CorruptingExtractor.enabled`**, not with a policy attribute. The
  evaluation shares the training policy object, so it cannot be a construction-time choice;
  `CleanEvalCallback` turns it off around the eval rollout, `CorruptEvalCallback` turns it on, and
  `play.py --corrupt-obs-eval` does the same for a whole rollout.
* **The noise level was chosen in a space neither arm encodes.** `t_max = 200` comes from decoding
  the frozen SD-VAE latent (`scripts/render_noise_levels.py`); what is actually corrupted is the
  40-d keypoint vector or a 514-d end-to-end ResNet feature. The feature-std scaling makes the SNR
  invariant to feature *magnitude* — which is all
  `test_scaling_makes_the_snr_independent_of_feature_magnitude` checks — and magnitude invariance
  is not semantic invariance. Re-deriving the level in the space actually corrupted is open.
  Related: on the keypoint arm the corruption lands on the `{-1, +1}` visibility mask too, and
  only because that mask's batch std is ~0 at full visibility does it escape untouched — below
  `--keypoint-visible-rate 1.0` the flag that says what is hidden is noised as hard as the
  positions are.

## Not built

* **The per-slot noise ladder** (`slot_obs_noise`). PPO has no candidate slots, so the ladder
  needs an axis chosen for it — episode step (LSTM context length) is the direct analogue of
  "slot k has k context entries". Flat corruption only, for now.
* **A Beta action distribution**, gated on what the diagnostics show. At `--log-std-init 0` about
  31% of action components are clipped and 12% of actions are full corners; at
  `--log-std-init -1` both are **zero**, so the std, not the distribution family, is what drives
  clipping at initialisation. If `action_corner_frac` climbs during training while returns
  plateau, that is the signal to replace the Gaussian.
