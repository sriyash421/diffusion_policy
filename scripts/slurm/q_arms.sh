# The arms and the Q checkpoints both Q evaluations share. Sourced, never run.
#
# ONE DEFINITION FOR BOTH EVALUATIONS. rank-expert asks where a* ranks; bon-sweep asks whether
# the ranking solves episodes. They are only two readings of the same experiment if they run on
# the same checkpoints, and a second copy of this list is how they stop being.

ROOT=/mmfs1/home/harine/diffusion_policy_standalone
B=ckpts/pusht_search/pusht_image_search_imgonly
STEP=checkpoints/step_0030000.ckpt

# label | run dir | extra flags for bon-sweep
# All three: split_file pusht_seed42_train176.json (176 train / 0 val / 30 test), filtered to
# the 12,972 moving train windows of transitions/pusht_seed42_train176_moving_ta8.json.
# There is no k1 arm in this generation -- it trained k16 and k4 only.
ARMS="
k16_uniform | $B/outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-176_split-mv_seed-42            |
k16_flat400 | $B/outer_inner/value_k16_ver-t_goal_son-flat400_enc-resnet18_demos-176_split-mv_seed-42 |
unetbc      | $B/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-176_split-mv_seed-42                    | --skip-context-sim
"

# label | checkpoint
# sac206 seeded its buffer from ALL 206 episodes (`demo_episodes: all`), so NO episode either
# evaluation scores is held out from it. sac106 seeded from the 106 train episodes of
# pusht_seed42_train106_val50.json only. That difference is the whole control, and it lives in
# params/args.yaml -- the two runs' command.txt are byte-identical.
# A THIRD FIELD: whether this Q needs --allow-contaminated. `assert_held_out` (sac/score.py)
# refuses to score episodes a Q's buffer was seeded from unless the flag is passed, and records
# the fact in the output. sac206 seeded from all 206, so EVERY episode either evaluation scores
# is in its buffer and the flag is always required; sac106 needs it only where the scored
# episodes overlap its 106 -- which is the mv test-30 (20 of 30), but not the non-106 manifest.
QS="
sac206_8p5M | /gscratch/robotics/harine/value_arms/sac_keypoint/model_8500000_steps.zip                         | always
sac106_3p3M | /gscratch/robotics/harine/value_arms/archive/sac_keypoint_demos106/model_3300000_steps.zip        | mv-only
"

CONDA=/gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
ENVNAME=robodiff

# GPUS pins the GPU MODEL when set (e.g. GPUS=rtx6k:1). Measured 2026-09-19: two jobs with
# identical seed, split and checkpoint return bit-identical curves on the same model and differ
# by 1-2 episodes of 30 across models, because the reduction order changes and a 300-step contact
# sim amplifies it. Pin it for anything that will be compared cell-by-cell to another job.
#
# `unset WANDB_API_KEY`: the shell's key is invalid and SHADOWS the valid ~/.netrc, so a job
# that inherits it dies at wandb.init. ckpt is preemptible and free, so the guaranteed
# robotics/weirdlab GPUs stay on training; --requeue re-runs after preemption.
submit_or_echo() {   # $1 = job name, $2 = command
    if [ "${SUBMIT:-0}" = "1" ]; then
        echo "[SUBMIT $1]"
        sbatch --partition=ckpt --account=robotics --requeue --job-name="$1" \
               --gpus="${GPUS:-1}" --cpus-per-task="${CPUS:-8}" --mem=32G \
               --time="${TIME:-12:00:00}" \
               --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out \
               --wrap "set -u; source $CONDA; conda activate $ENVNAME; cd $ROOT; \
                       unset WANDB_API_KEY; $2"
    else
        echo "[$1] $2"
    fi
}
