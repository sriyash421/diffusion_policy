"""The two architectures, and the four things that actually differ between them.

sb3-contrib's own advice on RecurrentPPO is to try frame stacking first -- "a simpler, faster and
usually competitive alternative" (their PPO vs RecurrentPPO report on masked-velocity envs, and
the Procgen paper's appendix Fig 11). This module is what lets both be run without a second copy
of the env, the reward, the callbacks or the bookkeeping: the entry points differ only in which
of these two objects they hand to runner.train.

Only four things differ, and they are exactly the four methods below -- how the vec env is
wrapped, how the agent is constructed, how it is loaded, and how many frames the video rollout
has to stack by hand. Everything else is shared by construction rather than by discipline.

NOTE on what frame stacking can and cannot do here. It reconstructs state that is hidden from a
single observation -- velocity in the report above. Our observation is currently NOT missing any:
corruption is off, occlusion is off, and PushT's space damping is 0 so the block carries no
momentum between steps. So on the clean arm `--n-stack 4`, `--n-stack 1` (plain PPO) and the LSTM
are all seeing a Markov observation, and a difference between them would be an optimisation
effect rather than a representational one. The comparison only bites once
--keypoint-visible-rate or --corrupt-obs is on.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from sb3_contrib import RecurrentPPO

from recurrent_ppo.corrupt_policy import features_extractor_kwargs, policy_for


def _shared_kwargs(cfg, log_dir):
    """The PPO arguments that have nothing to do with the architecture."""
    lr = cfg["learning_rate"]
    return dict(
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        learning_rate=(lambda progress: progress * lr) if cfg["lr_schedule"] == "linear" else lr,
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        target_kl=cfg["target_kl"],
        seed=cfg["seed"],
        device=cfg["device"],
        verbose=1,
        tensorboard_log=log_dir,
    )


def _shared_policy_kwargs(cfg):
    net_arch = [int(x) for x in cfg["net_arch"].split(",") if x]
    kwargs = {
        "net_arch": dict(pi=net_arch, vf=net_arch),
        "log_std_init": cfg["log_std_init"],
        "q_net_arch": tuple(int(x) for x in cfg["q_net_arch"].split(",") if x),
        "q_lr": cfg["q_lr"],
        # corruption arrives as a features extractor, through SB3's own extension point
        **features_extractor_kwargs(cfg["obs"], cfg["corrupt_obs"]),
    }
    if cfg["obs"] == "image":
        # SB3 applies orthogonal init to the WHOLE features extractor, which here is a ResNet18
        # carrying IMAGENET1K_V1 weights. Re-measured against this tree: conv1 does NOT survive
        # it, |w| 0.0762 -> 0.0936, so the arm would silently train from orthogonal noise rather
        # than from the pretrained encoder that is the entire reason it matches ST's. The same
        # call also deadlocks under multi-threaded BLAS -- torch.nn.init.orthogonal_ runs a QR
        # per layer; policy construction takes 0.40s at one thread and had not returned after
        # 120s at the default 24. Left ON for keypoint and state, where there are no pretrained
        # weights to destroy and orthogonal init is SB3's sensible default on a small MLP.
        kwargs["ortho_init"] = False
    return kwargs


class LstmArch:
    """sb3-contrib RecurrentPPO: memory in an LSTM carried across steps."""

    name = "lstm"
    log_root = "recurrent_ppo"
    # extra params/args.yaml keys play.py must read back; the LSTM sizes travel
    # inside the checkpoint itself, so there are none
    resolve_keys = ()

    def wrap(self, venv, cfg):
        return venv

    def video_n_stack(self, cfg):
        return 1

    def build(self, env, cfg, log_dir):
        policy_kwargs = {
            **_shared_policy_kwargs(cfg),
            "lstm_hidden_size": cfg["lstm_hidden_size"],
            "n_lstm_layers": cfg["n_lstm_layers"],
            # a shared LSTM and a critic LSTM are mutually exclusive in sb3-contrib
            "shared_lstm": cfg["shared_lstm"],
            "enable_critic_lstm": not cfg["shared_lstm"],
        }
        return RecurrentPPO(policy_for(cfg["obs"], recurrent=True, corrupt_obs=cfg["corrupt_obs"]), env,
                            policy_kwargs=policy_kwargs, **_shared_kwargs(cfg, log_dir))

    def load(self, path, env, cfg, print_system_info=True):
        return RecurrentPPO.load(path, env, print_system_info=print_system_info, device=cfg["device"])


class StackArch:
    """Stock PPO over the last `n_stack` observations, concatenated by VecFrameStack.

    `--n-stack 1` is plain PPO with no stacking at all, which is the baseline this project never
    had: if it matches the LSTM, the recurrence is not doing anything.
    """

    name = "stack"
    log_root = "ppo"
    # the stack depth shapes the ENV, not the checkpoint, so play.py has to read it
    resolve_keys = ("n_stack",)

    def wrap(self, venv, cfg):
        # applied INNERMOST, before VecNormalize, so the stacked observation is what the policy
        # sees in training, evaluation and play alike. VecNormalize runs with norm_obs=False here,
        # so the order cannot change any observation either way.
        return venv if cfg["n_stack"] <= 1 else VecFrameStack(venv, cfg["n_stack"])

    def video_n_stack(self, cfg):
        return cfg["n_stack"]

    def build(self, env, cfg, log_dir):
        return PPO(policy_for(cfg["obs"], recurrent=False, corrupt_obs=cfg["corrupt_obs"]), env,
                   policy_kwargs=_shared_policy_kwargs(cfg), **_shared_kwargs(cfg, log_dir))

    def load(self, path, env, cfg, print_system_info=True):
        return PPO.load(path, env, print_system_info=print_system_info, device=cfg["device"])


ARCHS = {a.name: a for a in (LstmArch(), StackArch())}
