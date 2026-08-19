"""Exit 0 if <curves.jsonl> already has a <selection> row for <step_XXXXXXX> covering n=32,64.

usage: _curve_has_n.py CURVES STEP [SELECTION]
`selection` defaults to argmax; pass the literal string 'null' for a run recorded without
a selection override (eval_search_pusht writes null there, not the resolved rule).
"""
import json, os, sys
path, step = sys.argv[1], int(sys.argv[2].split('_')[1])
want = sys.argv[3] if len(sys.argv) > 3 else 'argmax'
want = None if want == 'null' else want
if os.path.exists(path):
    for line in open(path):
        c = json.loads(line)
        if (c.get('step') == step and c.get('selection') == want
                and {32, 64} <= set(c.get('n') or [])):
            sys.exit(0)
sys.exit(1)
