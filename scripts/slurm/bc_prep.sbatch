#!/bin/bash
#SBATCH --job-name=bc_prep
#SBATCH --partition=ckpt-all
#SBATCH --account=robotics
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# One-off prep for the BC baselines, run on a compute node because the login node's shared
# ~10G cgroup makes even an import-and-instantiate smoke test time out under contention.
#   1. write the 25-episode split manifest (a PREFIX of the 100-episode train pool)
#   2. prove the search policy degenerates to plain BC at max_actions=1
set -euo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
set +u; conda activate robodiff; set -u

echo "=== 1. per-budget manifests (_train25 / _train100, same naming so configs interpolate) ==="
for n in 25 100; do
    python scripts/dump_pusht_splits.py --n-train-episodes $n \
        -o "diffusion_policy/config/splits/pusht_seed42_train${n}.json"
done

echo
echo "=== 2. nesting check: the 25-episode budget must be a SUBSET of the 100 ==="
python - <<'EOF'
import json
a = json.load(open('diffusion_policy/config/splits/pusht_seed42_train25.json'))
b = json.load(open('diffusion_policy/config/splits/pusht_seed42_train100.json'))
A, B = {int(i) for i in a['train']}, {int(i) for i in b['train']}
# subset, NOT prefix: the manifest stores sorted indices while the derivation truncates
# the *permuted* pool, so the budgets nest as sets and not in list order.
assert A < B, 'the 25-episode budget is not contained in the 100-episode one'
for k in ('val', 'test'):
    assert a[k] == b[k], f'{k} split differs between the two manifests'
assert a['episode_ends_checksum'] == b['episode_ends_checksum']
print(f"OK  train25 subset of train100 ({len(A)} of {len(B)}); "
      f"{a['train_frames']} vs {b['train_frames']} frames")
print(f"OK  val/test identical: {len(a['val'])} val, {len(a['test'])} test")
EOF

echo
echo "=== 3. BC degenerate case (max_actions=1) ==="
python scripts/bc_smoke.py
