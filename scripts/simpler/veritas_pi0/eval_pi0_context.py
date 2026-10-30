#!/usr/bin/env python
"""run_pi0_eval.py with the ContextEvalAgent (see diffusion_policy/pi0_context/context_eval_agent.py).

Extra overrides: +ctx_mode=base_bon|ckpt_bon|ctx_seq|ctx_tworound, +ctx_ckpt=<train ckpt .pt>,
+interp_lambda=0, +value_stats=<plans_h4/veritas_targets.json>, +eval_seeds=<json dict of splits>.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from run_pi0_eval import main  # noqa: E402
from diffusion_policy.pi0_context.context_eval_agent import ContextEvalAgent  # noqa: E402

if __name__ == "__main__":
    main(agent_cls=ContextEvalAgent)
