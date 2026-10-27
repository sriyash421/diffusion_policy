"""Keep tracked PushT Slurm jobs alive: resubmit any that left the queue unfinished.

Registry (JSON list) entries: name, kind ('train' | 'watch'), target (run dir or audit dir),
argv (the sbatch command), job (current Slurm id), restarts (UTC timestamps).
A train entry is finished at a step_0100000 checkpoint; a watch entry when its audit has
every 10k..100k checkpoint registered in fresh_evaluations.json (the watcher's own exit rule).
Resubmission reuses argv unchanged: same run_name -> the run resumes from latest.ckpt.

    python scripts/pusht_keepalive.py REGISTRY [--once]
    python scripts/pusht_keepalive.py REGISTRY --move NAME ACCOUNT PARTITION [TIME]
        # resubmit a PENDING run on a dedicated partition (pick_gpu.sh), then cancel the old job;
        # TIME defaults to DEDICATED_TIME (cse caps jobs at 1 day: pass 1-00:00:00)

Every registry read-modify-write holds REGISTRY.lock, so a move never races the loop
(a race could resubmit the cancelled job and put two trainers on one run dir).
"""
import contextlib, datetime, fcntl, getpass, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SQUEUE, SBATCH, SCONTROL, SCANCEL = '/usr/local/bin/squeue', '/usr/bin/sbatch', '/usr/bin/scontrol', '/usr/bin/scancel'
DEDICATED_TIME = '5-00:00:00'  # must end before the next maintenance reservation
PERIOD = 600
# Back off a run that died 3 times within 2 h, so a crash loop cannot flood the scheduler.
MAX_RESTARTS, WINDOW = 3, 7200


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def finished(e):
    target = Path(e['target'])
    if e['kind'] == 'train':
        return (target / 'checkpoints/step_0100000.ckpt').exists()
    runs = json.loads((target / 'retrain_manifest.json').read_text())
    registry = target / 'fresh_evaluations.json'
    done = json.loads(registry.read_text()) if registry.exists() else {}
    return all(f"{r['name']}_{s}" in done for r in runs for s in range(10000, 100001, 10000))


def queue():
    out = subprocess.run([SQUEUE, '-u', getpass.getuser(), '-h', '-o', '%i|%T|%r'], capture_output=True, text=True, check=True).stdout
    return {l.split('|')[0]: l.split('|')[1:] for l in out.splitlines() if l}


def log(path, **event):
    event = dict(at=now().isoformat(timespec='seconds'), **event)
    with open(path, 'a') as fh:
        fh.write(json.dumps(event) + '\n')
    print(json.dumps(event), flush=True)


@contextlib.contextmanager
def locked(registry_path):
    with open(registry_path.with_suffix('.lock'), 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        entries = json.loads(registry_path.read_text())
        yield entries
        tmp = registry_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(entries, indent=1))
        tmp.replace(registry_path)


def move(registry_path, name, account, partition, wall=DEDICATED_TIME):
    with locked(registry_path) as entries:
        e = next(e for e in entries if e['name'] == name)
        state = queue().get(e['job'], [None])[0]
        if state != 'PENDING':
            print(f'{name}: job {e["job"]} is {state}, not moving')
            return
        drop = ('--account=', '--partition=', '--qos=', '--time=')
        argv = [a for a in e['argv'] if not a.startswith(drop)]
        if partition == 'ckpt':  # back to the preemptible pool: 9 h cuts, self-requeue on USR1
            argv[1:1] = [f'--account={account}', '--partition=ckpt', '--qos=ckpt-gpu', '--time=9:00:00']
        else:
            argv[1:1] = [f'--account={account}', f'--partition={partition}', f'--time={wall}']
        p = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
        if p.returncode:
            print(f'{name}: submit failed: {p.stderr.strip()[-300:]}')
            return
        old, e['job'], e['argv'] = e['job'], p.stdout.strip().split(';')[0], argv
        subprocess.run([SCANCEL, old], check=True)
        log(registry_path.with_name('events.jsonl'), event='moved', name=name, old_job=old,
            job=e['job'], partition=f'{account}/{partition}')


def step(registry_path):
    with locked(registry_path) as entries:
        return _step(registry_path, entries)


def _step(registry_path, entries):
    events = registry_path.with_name('events.jsonl')
    q = queue()
    for e in entries:
        if e.get('finished'):
            continue
        if finished(e):
            e['finished'] = now().isoformat(timespec='seconds')
            log(events, event='finished', name=e['name'], job=e['job'])
            continue
        if e['job'] in q:
            state, reason = q[e['job']]
            if state == 'PENDING' and 'requeued held' in reason:
                subprocess.run([SCONTROL, 'release', e['job']], check=False)
                log(events, event='released', name=e['name'], job=e['job'], reason=reason)
            continue
        recent = [t for t in e.get('restarts', [])
                  if (now() - datetime.datetime.fromisoformat(t)).total_seconds() < WINDOW]
        if len(recent) >= MAX_RESTARTS:
            if not e.get('backoff_logged'):
                log(events, event='backoff', name=e['name'], job=e['job'], restarts=recent)
                e['backoff_logged'] = True
            continue
        p = subprocess.run(e['argv'], cwd=REPO, capture_output=True, text=True)
        if p.returncode:
            log(events, event='submit_failed', name=e['name'], stderr=p.stderr.strip()[-300:])
            continue
        old, e['job'] = e['job'], p.stdout.strip().split(';')[0]
        e.setdefault('restarts', []).append(now().isoformat(timespec='seconds'))
        e['backoff_logged'] = False
        log(events, event='resubmitted', name=e['name'], old_job=old, job=e['job'])
    return all(e.get('finished') for e in entries)


if __name__ == '__main__':
    registry = Path(sys.argv[1])
    if '--move' in sys.argv:
        i = sys.argv.index('--move')
        move(registry, *sys.argv[i + 1:i + 5])  # NAME ACCOUNT PARTITION [TIME]
        sys.exit()
    while True:
        try:
            if step(registry):
                print('all tracked jobs finished', flush=True)
                break
        except Exception as exc:  # a transient squeue/GPFS error must not stop the keeper
            print(f'{now().isoformat()} keeper error: {exc}', flush=True)
        if '--once' in sys.argv:
            break
        time.sleep(PERIOD)
