"""Splice the n=1…64 tables into docs/reports/q_fn_sac_all_206_demos_2026-09-20.md.

Rewrites only the region between the Q_N64_TABLE markers, so anything written around it by
hand survives. Idempotent -- running it twice produces the same file.

    python scripts/fill_q_n64_doc.py            # write
    python scripts/fill_q_n64_doc.py --dry-run  # print what would be written
"""
import argparse
import pathlib
import subprocess
import sys

DOC = pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'reports' / \
    'q_fn_sac_all_206_demos_2026-09-20.md'
START, END = '<!-- Q_N64_TABLE_START -->', '<!-- Q_N64_TABLE_END -->'
# success first: it is the headline. Coverage second, because it is the one that can separate
# arms among their SOLVED episodes, where the reward saturates.
METRICS = ('success', 'coverage', 'reward_max')


def table(metric):
    r = subprocess.run([sys.executable,
                        str(pathlib.Path(__file__).with_name('q_n64_table.py')),
                        '--metric', metric],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f'*(`--metric {metric}` failed: {r.stderr.strip().splitlines()[-1:]})*\n'
    return r.stdout.strip() + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    body = '\n\n'.join(table(m) for m in METRICS)
    doc = DOC.read_text()
    if START not in doc or END not in doc:
        raise SystemExit(f'markers {START} / {END} not found in {DOC}')
    head, rest = doc.split(START, 1)
    _, tail = rest.split(END, 1)
    out = f'{head}{START}\n\n{body}\n{END}{tail}'

    if args.dry_run:
        print(body)
        return
    DOC.write_text(out)
    print(f'wrote {len(body.splitlines())} lines into {DOC.name}')


if __name__ == '__main__':
    main()
