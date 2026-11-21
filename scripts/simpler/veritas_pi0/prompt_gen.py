#!/usr/bin/env python
"""Ask Gemini to rewrite the Veritas planning guidance from successful demonstrations.

One call: the official plan prompt, how the tracker consumes a plan (tol / min_hold /
skippable and the skip rule), and the demo_examples.py examples of every task (first frame
with the end-effector path and its key-moment pixels). Gemini returns a replacement for the
guidance block (everything after the "Instruction:" line), generic across scenes. The
``gemini_written`` variant in plan_prompts.py splices it into the official header.

    python prompt_gen.py [--model gemini-3.8-flash] [--n 5]   (simpler_env python, GEMINI key)
Out: $STEP2/expert_analysis/prompt_gen/{meta_prompt.txt, response.json, guidance.txt}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(HERE))), HERE]
from diffusion_policy.env.simpler.veritas.simpler_client import PROMPT_V1_UPSTREAM  # noqa: E402
from diffusion_policy.veritas.gemini_client import GeminiClient  # noqa: E402
from plan_prompts import GUIDANCE_START, STEP2, TASK_OF  # noqa: E402

TRACKER = """How the plan is consumed. A tracker holds the ACTIVE waypoint. Each frame the
end-effector pixel is compared with it: within tol_px for min_hold consecutive frames
latches it and the next waypoint becomes active. If the NEXT waypoint is skippable the
tracker may also abandon the active one early: when the end-effector has moved more than
40% of the way from the active waypoint toward the next for 3 frames (or stalls near the
next one), the active waypoint is skipped. A verifier then ranks the policy's candidate
action chunks by how close their endpoint gets to the active waypoint (and, with lower
weight, the two after it) and executes the best. The verifier does not see the gripper.
Failure seen with the current prompt: the pre-grasp and lift waypoints sit at the same
pixel between the start pose and the object, the descending arm passes that pixel first,
the tracker skips ahead to lift/carry while the gripper is still open above the object,
and the verifier pulls the arm away before it grasps. Also, final waypoints placed on the
target itself are never reached: the gripper releases 50-75 mm above the target.
"""
ASK = """Write a replacement for the guidance block of the prompt (everything from "Camera is
fixed" to the end). Requirements:
- generic: it will be used in every scene of these four tasks and must not contain any
  pixel coordinates or scene-specific numbers from the examples;
- it must say where to put each waypoint relative to the objects as seen in the
  examples (how far above the object the pre-grasp and lift sit, where the grasp point
  is on each kind of object, where the final waypoint is relative to the target), and
  which tol_px, min_hold and skippable to give each waypoint so the tracker, described
  above, follows a successful trajectory without skipping past the grasp or getting
  stuck on the final waypoint;
- keep the same JSON schema and the same meaning of tol_px, min_hold, skippable;
- at most 25 lines of plain text; start with "Camera is fixed; pixel coords are absolute."
Return JSON: {"guidance": "<the block>", "rationale": "<what in the examples drove each choice>"}
"""


class Rewrite(BaseModel):
    guidance: str
    rationale: str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=f"{STEP2}/expert_analysis/prompt_gen")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    official = PROMPT_V1_UPSTREAM.replace("{instruction}", "<instruction>")
    parts = ["You are improving a prompt that asks a vision-language model to plan pixel waypoints for a "
             "robot end-effector from one camera image.\n\nCURRENT PROMPT (the model also receives the "
             "image, the current end-effector pixel and the object pixels):\n" + official + "\n" + TRACKER
             + "\nSUCCESSFUL DEMONSTRATIONS. Each image is the first frame of a demonstration that "
             "completed the task: orange dots are the end-effector path, numbered rings are the pixels "
             "at the key moments (1 pre-grasp, 2 grasp, 3 lift, 4 carry, 5 place), crosses are the "
             "objects, the white ring is the end-effector at the start.\n"]
    for instr, task in TASK_OF.items():
        d = f"{STEP2}/{task}/pi0_demos/examples"
        for k in range(a.n):
            ex = json.load(open(f"{d}/ex{k}.json"))
            parts.append(Image.open(f"{d}/{ex['png']}").convert("RGB"))
            parts.append(f"Task '{instr}', example {k + 1}. Start end-effector pixel {ex['eef_pixel']}; objects: "
                         + "; ".join(f"{h['name']}({h['role']}) uv={h['uv']}" for h in ex["hints"])
                         + ". Key-moment pixels: " + json.dumps([dict(moment=w["name"], uv=w["uv"]) for w in ex["waypoints"]]))
    parts.append(ASK)
    open(f"{a.out}/meta_prompt.txt", "w").write("\n\n".join(p if isinstance(p, str) else "<image>" for p in parts))
    client = GeminiClient(model_name=a.model)
    cfg = {"response_mime_type": "application/json", "response_schema": Rewrite.model_json_schema()}
    r = client._call_with_retry(parts, cfg, Rewrite, label="PromptGen", max_retries=5)
    assert r is not None, "Gemini returned nothing"
    assert r.guidance.startswith(GUIDANCE_START.strip()), r.guidance[:80]
    json.dump(dict(model=a.model, usage=client.last_usage, **r.model_dump()), open(f"{a.out}/response.json", "w"), indent=1)
    open(f"{a.out}/guidance.txt", "w").write(r.guidance.rstrip() + "\n")
    print(r.guidance, "\n--- rationale ---\n", r.rationale, "\n", client.last_usage)


if __name__ == "__main__":
    main()
