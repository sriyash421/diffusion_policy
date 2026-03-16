"""No-op env runners for UWLab tasks.

These return empty dicts so training can proceed without a live simulator.
Replace with a real UWLab runner when you want rollout evaluation during
training.
"""

from typing import Dict
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class UwlabDummyImageRunner(BaseImageRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy) -> Dict:
        return {}


class UwlabDummyLowdimRunner(BaseLowdimRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy) -> Dict:
        return {}
