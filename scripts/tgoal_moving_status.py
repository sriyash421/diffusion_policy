"""One compact line per arm: training progress AND whatever eval has landed so far.

    python scripts/tgoal_moving_status.py [--filter son-.*600]

Reads checkpoints on disk, squeue, and the attention/mode jsons. Touches no GPU and no zarr,
so it is safe to poll.
"""
import argparse
import json
import pathlib
import re
import subprocess

R = pathlib.Path('/gscratch/robotics/harine/diffusion_policy_outputs'
                 '/pusht_search/pusht_image_search_imgonly')
A = pathlib.Path('/gscratch/robotics/harine/attention_analysis')
M = pathlib.Path('/gscratch/robotics/harine/mode_analysis')
LOGS = pathlib.Path('/gscratch/robotics/harine/slurm_logs')


def canon(run):
    m = re.match(r'^(value_k\d+_ver-.+?)(?:_(son-[^_]+))?_enc-([^_]+)_demos-(\d+)'
                 r'_split-([^_]+)_seed-(\d+)$', run)
    if not m:
        return run
    head, noise, enc, demos, tag, seed = m.groups()
    return f'split-{tag}_demos-{demos}_{noise or "son-none"}_{head}_enc-{enc}_seed-{seed}'


def label(run):
    m = re.search(r'value_(k\d+)_ver-t_goal(?:_son-([a-z0-9to]+))?', run)
    if not m:
        return 'BC'
    return f'{m.group(1)}-{m.group(2) or "uniform"}'


def latest(d):
    ck = sorted(int(p.name[5:-5]) for p in (d / 'checkpoints').glob('step_*.ckpt'))
    return ck[-1] if ck else 0


def live_step(d):
    """The step the optimiser is actually on, from logs.json.txt.

    Checkpoints land every 10k, so at ~5k steps/hour an arm shows the SAME checkpoint number
    for two hours and a healthy run is indistinguishable from a hung one. This reads the
    training log instead, which is appended continuously.
    """
    f = d / 'logs.json.txt'
    if not f.is_file():
        return None
    best = None
    for ln in f.read_text(errors='ignore').splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            g = json.loads(ln).get('global_step')
        except Exception:
            continue
        if g is not None and (best is None or g > best):
            best = g
    return best


def eval_at(run, step):
    """(context share, dispersion, a*/disp) at the newest analysed step <= step."""
    out = [None, None, None]
    ap, mp = A / canon(run) / 'attention.json', M / canon(run) / 'modes.json'
    if ap.is_file():
        st = json.loads(ap.read_text()).get('steps', {})
        ok = [int(k) for k in st if int(k) <= step]
        if ok:
            dn = st[str(max(ok))]['denoise']
            ps = dn[max(dn, key=int)]['context_mass_per_slot']
            out[0] = sum(ps[1:]) / max(len(ps) - 1, 1)
    if mp.is_file():
        st = json.loads(mp.read_text()).get('steps', {})
        ok = [int(k) for k in st if int(k) <= step]
        if ok:
            r = st[str(max(ok))]
            d = r['endpoint_dispersion_px']
            out[1] = sum(d) / len(d)
            n = r.get('astar_min_dist_norm')
            out[2] = min(n) if n else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--filter', default='', help='regex on the run name')
    args = ap.parse_args()

    q = subprocess.run(['squeue', '-u', 'harine', '-h', '-o', '%T|%j'],
                       capture_output=True, text=True).stdout
    running = {}
    for ln in q.splitlines():
        if '|' in ln:
            s, j = ln.split('|', 1)
            running[j.strip()] = s.strip()

    dirs = sorted(d for d in R.glob('*/*demos-176_split-mv*') if d.is_dir()
                  and re.search(args.filter, d.name))
    for d in dirs:
        run = d.name
        step = latest(d)
        st = running.get(f'tr_{run}') or ('DONE' if step >= 100000 else 'gone')
        ctx, disp, an = eval_at(run, step)
        f = lambda v, w, p: (f'{v:{w}.{p}f}' if v is not None else ' ' * (w - 1) + '-')  # noqa
        live = live_step(d)
        lv = f'{live/1000:5.1f}k' if live is not None else '    -'
        print(f'{label(run):<18} ckpt{step//1000:4d}k live{lv} {st:<8} '
              f'ctx{f(ctx,7,3)} disp{f(disp,7,1)}px a*/disp{f(an,6,2)}')
    bad = [p.name for p in LOGS.glob('tr_*demos-176_split-mv*.out')
           if re.search(args.filter, p.name)
           and re.search(r'Traceback|CUDA error|slurmstepd: error', p.read_text(errors='ignore'))]
    if bad:
        print(f'ERRORS in {len(bad)} log(s): {bad[0]}')


if __name__ == '__main__':
    main()
