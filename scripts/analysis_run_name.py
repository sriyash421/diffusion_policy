"""The analysis-folder naming convention: dataset first, then noise schedule.

    python scripts/analysis_run_name.py <run_name> [...]     # print the canonical folder name

WHY THE ORDER DIFFERS FROM THE CHECKPOINT DIR. A training run is named for its policy first
(`value_k16_ver-t_goal_son-flat400_enc-resnet18_demos-137_split-blq_seed-42`), which is the
right order when you are looking for a checkpoint. But the analysis folders are read as a
GROUP -- every arm side by side -- and there the two axes the experiment varies are the
dataset and the noise schedule. Leading with them makes an `ls` sort into aligned columns:
all of one dataset together, and within it the schedules in order.

    value_k16_ver-t_goal_son-flat400_enc-resnet18_demos-137_split-blq_seed-42
    -> split-blq_demos-137_son-flat400_value_k16_ver-t_goal_enc-resnet18_seed-42

BASELINE ARMS GET AN EXPLICIT `son-none`. An arm with no obs-noise schedule has nothing in
that position, and omitting it would shift every later token left so the baseline row no
longer lines up with the noised rows beside it. The placeholder keeps the columns.

CHECKPOINT DIRECTORIES ARE NOT RENAMED. They hold the eval output that
scripts/build_geometric_splits_doc.py addresses by exact path, so the run name stays the
identity of the training run; this is only how its ANALYSIS output is filed.
"""
import re
import sys

# `ver-t_goal` contains an underscore, so the head is matched non-greedily up to whichever
# of the optional noise token or `_enc-` comes first -- splitting on '_' would break it.
PATTERN = re.compile(
    r'^(?P<head>value_k\d+_ver-.+?)'
    r'(?:_(?P<noise>son-[^_]+|gm-[^_]+))?'
    r'_enc-(?P<enc>[^_]+)'
    r'_demos-(?P<demos>\d+)'
    r'_split-(?P<tag>[^_]+)'
    r'_seed-(?P<seed>\d+)$')

NO_NOISE = 'son-none'


def canonical(run):
    """Canonical analysis-folder name for a run, or None if `run` is not a run name.

    Returning None rather than raising is deliberate: callers walk directories that also
    contain legacy names (`k16_blq/`) and top-level figures, and those must be skipped
    rather than crash the walk.
    """
    m = PATTERN.match(run)
    if not m:
        return None
    g = m.groupdict()
    return (f"split-{g['tag']}_demos-{g['demos']}_{g['noise'] or NO_NOISE}"
            f"_{g['head']}_enc-{g['enc']}_seed-{g['seed']}")


def is_canonical(name):
    return canonical(name) == name or (canonical(name) is None and _looks_canonical(name))


def _looks_canonical(name):
    return name.startswith('split-') and f'_{NO_NOISE}_' in name or (
        name.startswith('split-') and ('_son-' in name or '_gm-' in name))


if __name__ == '__main__':
    for a in sys.argv[1:]:
        print(canonical(a) or a)
