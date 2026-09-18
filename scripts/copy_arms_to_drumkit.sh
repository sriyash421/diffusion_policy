#!/usr/bin/env bash
# Copy the five arms (BC UNet, ST k=16 uniform, flat400, flat200, ramp400to200) on BOTH
# geometric datasets to drumkit. ~30GB.
#
#   bash scripts/copy_arms_to_drumkit.sh -n      # dry run: list what would move
#   bash scripts/copy_arms_to_drumkit.sh
#
# RUN THIS FROM YOUR OWN TERMINAL. drumkit accepts publickey or password and rejects every
# key in ~/.ssh here (id_ed25519, id_github, klone), so the transfer needs a password typed
# interactively. `ssh-copy-id -i ~/.ssh/id_ed25519.pub harine@drumkit.cs.washington.edu`
# once would remove that, after which this is unattended.
#
# ONE ssh connection, shared by every rsync below via ControlMaster -- otherwise each of the
# ten runs prompts for the password separately. The master is opened once up front and torn
# down on exit.
#
# CHECKPOINT SCOPE IS ASYMMETRIC, on purpose: all ten steps for each k=16 arm (274MB each,
# 2.6GB per run) but only step_0100000 for BC, whose checkpoints are 4.4GB apiece -- all ten
# BC steps would be 44GB per dataset, 81% of the bytes, to carry a trajectory nothing here
# reads. run.json and splits.json ride along with every run: without splits.json the
# checkpoint cannot be evaluated on the split it was trained against.
set -uo pipefail

DEST_HOST="${DEST_HOST:-harine@drumkit.cs.washington.edu}"
DEST_DIR="${DEST_DIR:-pusht_ckpts}"                  # relative => remote $HOME
SRC=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search_imgonly

DRY=()
[[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]] && DRY=(--dry-run)

MANIFEST=$(mktemp); trap 'rm -f "$MANIFEST"' EXIT
for d in demos-137_split-blq demos-60_split-brd60; do
  bc="unet_bc/unetbc_ver-t_goal_enc-resnet18_${d}_seed-42"
  printf '%s\n' "$bc/run.json" "$bc/splits.json" "$bc/checkpoints/step_0100000.ckpt" >> "$MANIFEST"
  for a in '' _son-flat400 _son-flat200 _son-ramp400to200; do
    r="outer_inner/value_k16_ver-t_goal${a}_enc-resnet18_${d}_seed-42"
    printf '%s\n' "$r/run.json" "$r/splits.json" "$r/checkpoints/" >> "$MANIFEST"
  done
done

# Every path in the manifest must exist locally before anything is sent: a typo in a run
# name is otherwise a silently missing arm on the far side.
missing=0
while read -r p; do [[ -e "$SRC/$p" ]] || { echo "MISSING: $SRC/$p" >&2; missing=1; }; done < "$MANIFEST"
[[ $missing -eq 0 ]] || { echo 'aborting -- fix the manifest' >&2; exit 1; }
echo "sending $(du -shc $(sed "s|^|$SRC/|" "$MANIFEST") 2>/dev/null | tail -1 | cut -f1) to $DEST_HOST:$DEST_DIR"

CTL=~/.ssh/cm-drumkit-$$
SSH="ssh -o ControlMaster=auto -o ControlPath=$CTL -o ControlPersist=4h"
trap 'ssh -O exit -o ControlPath=$CTL "$DEST_HOST" 2>/dev/null; rm -f "$MANIFEST"' EXIT
$SSH "$DEST_HOST" "mkdir -p '$DEST_DIR'" || { echo 'ssh failed -- see the header' >&2; exit 1; }

# --partial so an interrupted 4.4GB BC checkpoint resumes instead of restarting; no -z,
# because these are float tensors that do not compress and the login node is shared.
rsync -rlptDh --partial --info=progress2 "${DRY[@]}" \
  -e "$SSH" --files-from="$MANIFEST" "$SRC/" "$DEST_HOST:$DEST_DIR/"
