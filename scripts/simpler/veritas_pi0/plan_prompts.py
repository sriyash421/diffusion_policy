"""Named variants of the two Veritas Gemini prompts (waypoint plan, object selection).

A variant rewrites the text the official ``VeritasGeminiClient`` builds
(``$VERITAS_ROOT/src/agent/veritas_verifier/gemini_client.py``) just before it is sent, so
schema, generation config, retry and fallback stay official. ``official`` is the identity.
Selected with ``PLAN_PROMPT=<name>`` (see ``run_pi0_eval.py``); ``plan_simpler_episodes.py
--variant`` uses the same ``apply``. A variant may also prepend ``parts`` (few-shot
examples from ``demo_examples.py``), looked up by the instruction in the plan text.
"""
from __future__ import annotations

import json
import os
import re

PLAN_MARK = "Generate 5-10 absolute pixel waypoints"
OBJECTS_MARK = "Select the minimal set of objects"
STEP2 = os.environ.get("STEP2", "/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2")
TASK_OF = {"put eggplant into yellow basket": "eggplant", "stack the green block on the yellow block": "stack",
           "put carrot on plate": "carrot", "put the spoon on the towel": "spoon"}


def _same(text: str) -> str:
    return text


OFFICIAL_TOL = """use ~60 pixels per waypoint by default. Adjust within ~50-80 only when you
have a clear reason — slightly tighter (~50) for precise contact/placement,
slightly looser (~70-80) for coarse approach transits. Stay close to 60.
"""
TIGHT_TOL = """use ~60 pixels for transit waypoints, and ~30 pixels for the grasp waypoint
and for the final placement waypoint, so the gripper has to reach the object
itself rather than its surroundings.
"""
STRUCTURE = """Structure the plan as: (1) a pre-grasp waypoint directly above the object to
pick up; (2) the grasp waypoint on the middle of that object's graspable part
(e.g. the handle of a utensil, the middle of an elongated object, the center of
a block), where the gripper closes; (3) a lift waypoint directly above the grasp
point; (4) a transit waypoint above the target; (5) the final placement waypoint
on the center of the target. Do not put a waypoint at the current end-effector
position. Use the object pixel positions given above when they are provided.
"""


OFFICIAL_SKIP = """Set skippable=true for every intermediate waypoint so the verifier can skip
ahead when one is approximately satisfied. Set skippable=false only for the
final contact/placement waypoint — that one must latch for success.
"""
# The verifier ignores the gripper and a 60 px circle is met while the gripper is still above
# the object, so the tracker moves on (or skips the grasp waypoint) before the grasp; the
# verifier then pulls the arm toward the next waypoint and it leaves empty-handed.
# grasp_lock does not stop the skip: upstream leaves the ACTIVE waypoint when the NEXT one is
# skippable, and a descending arm passes the lift pixel first (transit_skip). lift_lock and
# noskip lock the waypoint after the grasp instead.
GRASP_LOCK_SKIP = """Set skippable=false for the grasp waypoint and for the final placement
waypoint: the gripper must actually reach the object and close on it before the
plan moves on. Set skippable=true for the other waypoints.
"""

LIFT_LOCK_SKIP = """Set skippable=false for the waypoint right after the grasp waypoint (the lift)
and for the final placement waypoint; set skippable=true for the other waypoints.
The verifier only skips past the active waypoint when the next waypoint is
skippable, so a non-skippable lift makes the gripper actually reach the grasp
waypoint before the plan moves on.
"""
NOSKIP = """Set skippable=false for every waypoint: each one must be reached in order.
"""
GRASP_HOLD = ("4 for the grasp waypoint (time to descend and close), 3 for the final\n"
              "placement waypoint.")
# Successful demos end with the gripper 50-75 mm above the target, 50-60 px above its pixel
# in the image; a final waypoint on the target itself is never reached (the plan never
# completes and the verifier keeps pushing down).
RELEASE_ABOVE = """Put the final placement waypoint where the gripper opens: about 50 pixels
ABOVE the target object's pixel in the image (smaller v), not on the target itself,
because the object is released from above and hangs below the gripper.
"""
FEWSHOT_NOTE = """The images before the last one are successful demonstrations of this task in
other scenes: the orange dots are the end-effector path and the numbered rings are
the pixels it passed through at the key moments (pre-grasp, grasp, lift, carry,
place), listed under each image in the schema. Plan the LAST image the same way:
the same five moments, placed on this scene's objects.
"""


def _replace(text: str, old: str, new: str) -> str:
    assert old in text, "official prompt text changed"
    return text.replace(old, new)


def _structured(text: str) -> str:
    return text + STRUCTURE


def _tight(text: str) -> str:
    text = _replace(text, OFFICIAL_TOL, TIGHT_TOL)
    return _replace(text, "2-3 for contact/placement waypoints that need a brief settle.",
                    "2 for the grasp waypoint, 3 for the final placement waypoint.")


def _grasp_lock(text: str) -> str:
    text = _replace(_tight(text), OFFICIAL_SKIP, GRASP_LOCK_SKIP)
    return _replace(text, "2 for the grasp waypoint, 3 for the final placement waypoint.", GRASP_HOLD)


def _lift_lock(text: str) -> str:
    text = _replace(text, OFFICIAL_SKIP, LIFT_LOCK_SKIP)
    return _replace(text, "2-3 for contact/placement waypoints that need a brief settle.", GRASP_HOLD)


def _noskip(text: str) -> str:
    text = _replace(text, OFFICIAL_SKIP, NOSKIP)
    return _replace(text, "2-3 for contact/placement waypoints that need a brief settle.", GRASP_HOLD)


def _release_above(text: str) -> str:
    return _lift_lock(text) + RELEASE_ABOVE


GUIDANCE_START = "Camera is fixed; pixel coords are absolute.\n"
GEMINI_GUIDANCE = os.environ.get("PROMPT_GEN_FILE", f"{STEP2}/expert_analysis/prompt_gen/guidance.txt")


def _gemini_written(text: str) -> str:
    """Official header (context line, schema, instruction) + the guidance Gemini wrote from
    successful demos (prompt_gen.py); replaces everything from GUIDANCE_START on."""
    assert GUIDANCE_START in text, "official prompt text changed"
    return text[:text.index(GUIDANCE_START)] + open(GEMINI_GUIDANCE).read().rstrip() + "\n"


def _task(text: str) -> str:
    m = re.search(r"Instruction: (.*?)\.\n", text)
    return TASK_OF[m.group(1)]


def _examples(text: str, n: int = 5) -> list:
    """[image, text] * n for the task named in the plan text (demo_examples.py output)."""
    from PIL import Image
    d = f"{STEP2}/{_task(text)}/pi0_demos/examples"
    parts = []
    for k in range(n):
        ex = json.load(open(f"{d}/ex{k}.json"))
        ctx = (f"Example {k + 1}. Image size: {ex['image_size'][0]}x{ex['image_size'][1]}. "
               f"Current end-effector pixel: {ex['eef_pixel']}. Objects (name/role/uv): "
               + "; ".join(f"{h['name']}({h['role']}) uv={h['uv']}" for h in ex["hints"]) + "\n"
               "Waypoints the end-effector passed through: " + json.dumps(
                   [dict(uv=w["uv"], moment=w["name"]) for w in ex["waypoints"]]))
        parts += [Image.open(f"{d}/{ex['png']}").convert("RGB"), ctx]
    return parts


VARIANTS = {
    "official": dict(plan=_same, objects=_same),
    "structured": dict(plan=_structured, objects=_same),
    "tight": dict(plan=_tight, objects=_same),
    "structured_tight": dict(plan=lambda t: _structured(_tight(t)), objects=_same),
    "grasp_lock": dict(plan=_grasp_lock, objects=_same),
    "structured_grasp_lock": dict(plan=lambda t: _structured(_grasp_lock(t)), objects=_same),
    "lift_lock": dict(plan=_lift_lock, objects=_same),
    "noskip": dict(plan=_noskip, objects=_same),
    "release_above": dict(plan=_release_above, objects=_same),
    "gemini_written": dict(plan=_gemini_written, objects=_same),
    "fewshot": dict(plan=lambda t: _release_above(t) + FEWSHOT_NOTE, objects=_same, parts=_examples),
    "fewshot_official": dict(plan=lambda t: t + FEWSHOT_NOTE, objects=_same, parts=_examples),
}


def apply(name: str, parts: list) -> list:
    """Rewrite the text part of a plan or object-selection request (and prepend the variant's
    example parts to a plan request); other calls pass through."""
    v = VARIANTS[name]
    out = []
    for p in parts:
        if isinstance(p, str) and PLAN_MARK in p:
            if "parts" in v:
                out = v["parts"](p) + out
            p = v["plan"](p)
        elif isinstance(p, str) and p.startswith(OBJECTS_MARK):
            p = v["objects"](p)
        out.append(p)
    return out
