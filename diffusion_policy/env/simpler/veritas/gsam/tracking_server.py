# Vendored verbatim from veritas@8896225 src/agent/tracking_server.py (exact upstream Grounded-SAM-2 object
# hints); only the src.agent import paths are rewritten. Do not edit; re-vendor instead.
"""
Thin CLI wrapper for the Grounded-SAM-2 tracking server.
Core logic lives in src.agent.tracking_server_core to keep this entrypoint small.
"""

import argparse

from diffusion_policy.env.simpler.veritas.gsam.tracking_server_core import run_tracking_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--model_cfg", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt_type", default="box")
    args = parser.parse_args()
    run_tracking_server(args)


if __name__ == "__main__":
    main()

