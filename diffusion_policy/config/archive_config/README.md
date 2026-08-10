# Archived configs

These five files carried **nothing but overrides**. Each set `search_context`,
`corrupt_obs`, `arm`, `name` and `logging.tags` on top of
`train_pusht_diffusion_search` and changed no other value — which is why Round 8 stopped
using them: `scripts/slurm/launch_round8_29demo.sh` launches the same six arms from one config
plus three CLI overrides, and all six r8 runs came out of that path.

Keeping a file per cell of a 3×2 matrix meant six places to edit whenever the base
changed, and two of them (`_subgoal_corrupt`, `_subgoal_verifier_corrupt`) inherited from
their clean sibling rather than the base, so a change had to be traced through two hops to
know what a run actually resolved to.

They are archived rather than deleted because twelve run directories on gscratch record
these names in `.hydra/hydra.yaml` as the config they trained from. Nothing needs to load
them again — the runs are finished and their fully-resolved configs are saved in their own
`.hydra/config.yaml`, which is what `aug9_analysis.md` reads.

## The replacement

Live configs: **`train_pusht_diffusion_search`** (100 demos) and
**`train_pusht_diffusion_search_29`** (29 demos, r8 parity). Everything else is an
override.

| archived config | equivalent |
|---|---|
| `_corrupt` | `search_context=value arm=value corrupt_obs=True` |
| `_subgoal` | `search_context=subgoal arm=subgoal-chosen4value corrupt_obs=False` |
| `_subgoal_corrupt` | `search_context=subgoal arm=subgoal-chosen4value corrupt_obs=True` |
| `_subgoal_verifier` | `search_context=subgoal_value arm=subgoal-value corrupt_obs=False` |
| `_subgoal_verifier_corrupt` | `search_context=subgoal_value arm=subgoal-value corrupt_obs=True` |

```bash
CONFIG_NAME=train_pusht_diffusion_search \
  sbatch --account=$A --partition=$P --export=ALL,CONFIG_NAME \
  scripts/slurm/train_pusht_search.sbatch \
  search_context=subgoal arm=subgoal-chosen4value corrupt_obs=True
```

`arm` is not decoration and must be passed: `TrainMLPImageWorkspace._check_arm_label`
asserts it against `(search_context, selection)` and refuses to start if they disagree,
because the run directory is named from `arm` and a mismatch would file a whole column of
`SUCCESS_RATES.md` under the wrong ablation. Omitting `arm` entirely skips the check and
resolves the directory to the base's `value` — silently wrong. Pass all three.

The run directory is unaffected by `name`: `run_name` is built from
`arm`/`corrupt_obs`/`n_demos`/`training.seed`. Pass `name=...` only if you want the wandb
tag to read something other than `train_pusht_diffusion_search`.

## What was NOT archived, and why

- **`train_pusht_bc*`** — not an override of the search arms. BC sets
  `policy.max_actions: 1` (empty context, so no candidates and no verifier pool),
  `n_search_actions: 1`, and its own `run_name` template. It is the baseline every search
  number is read against, so it stays first-class.
- **`train_pusht_diffusion_search_subgoal_only*`** (5 files) — these change the
  *selection rule* to `final_pass`, plus `slot_weight_decay` / `context_decay` /
  `n_candidates` and the training budget, and the `cd`/`k` variants use a different
  `run_name` template (`${arm}_k${n_candidates}_cd${context_decay}_...`). Reproducing one
  takes six overrides including a template, which is worse than a file.
- **`train_pusht_search_outer_inner*`** (3 files) — a different trainer entirely
  (`TrainSearchOuterInnerWorkspace`), not a variant of the offline arms. See
  `README_pusht.md` §6.
