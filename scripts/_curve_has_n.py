"""Exit 0 if <curves.jsonl> already has an argmax row for <step_XXXXXXX> covering n=32,64."""
import json, os, sys
path, step = sys.argv[1], int(sys.argv[2].split('_')[1])
if os.path.exists(path):
    for line in open(path):
        c = json.loads(line)
        if (c.get('step') == step and c.get('selection') == 'argmax'
                and {32, 64} <= set(c.get('n') or [])):
            sys.exit(0)
sys.exit(1)
