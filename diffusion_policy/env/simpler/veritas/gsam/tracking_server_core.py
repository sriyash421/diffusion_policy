# Vendored verbatim from veritas@8896225 src/agent/tracking_server_core.py (exact upstream Grounded-SAM-2 object
# hints); only the src.agent import paths are rewritten. Do not edit; re-vendor instead.
"""
Core logic for the Grounded-SAM-2 tracking server.
Split out of tracking_server.py to keep the CLI wrapper small.
"""

import base64
import json
import logging
import os
import sys
import traceback
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

# Configure logging to stderr so stdout is clean for JSON IPC
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# Import Grounded-SAM-2 components
try:
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    try:
        from utils.track_utils import sample_points_from_masks
    except ImportError:
        # Default to the in-tree submodule (Grounded-SAM-2 lives at the repo root).
        _default_gsam2 = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "Grounded-SAM-2",
        )
        grounded_sam2_path = os.environ.get("GROUNDED_SAM2_PATH", _default_gsam2)
        if os.path.exists(grounded_sam2_path):
            sys.path.insert(0, grounded_sam2_path)
            from utils.track_utils import sample_points_from_masks
except ImportError as e:  # pragma: no cover - env specific
    log.error(f"Failed to import Grounded-SAM-2 dependencies: {e}")
    raise


class TrackingServer:
    """Server that handles tracking requests."""

    def __init__(self, sam2_checkpoint, model_cfg, device="cuda", prompt_type="box"):
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.device = device
        self.prompt_type = prompt_type

        # Default to the in-tree submodule (Grounded-SAM-2 lives at the repo root).
        _default_gsam2 = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "Grounded-SAM-2",
        )
        grounded_sam2_path = os.environ.get("GROUNDED_SAM2_PATH", _default_gsam2)
        sam2_configs_dir = os.path.join(grounded_sam2_path, "sam2", "configs")
        if not os.path.exists(sam2_configs_dir):
            raise FileNotFoundError(
                f"SAM 2 configs directory not found: {sam2_configs_dir}\n"
                f"Please set GROUNDED_SAM2_PATH environment variable correctly."
            )

        # Initialize Hydra config path for SAM2
        try:
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            from hydra import initialize_config_dir

            initialize_config_dir(config_dir=os.path.abspath(sam2_configs_dir), version_base=None)
            log.info(f"Initialized Hydra with config directory: {sam2_configs_dir}")
        except Exception as e:
            log.error(f"Failed to initialize Hydra config directory: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Hydra for SAM 2 configs: {sam2_configs_dir}") from e

        log.info(f"Initializing SAM 2 from {sam2_checkpoint} with config {model_cfg}")
        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        # grounding-dino-tiny fails to detect many simulated objects (e.g., the SimplerEnv
        # eggplant is invisible to tiny even at threshold 0.01). grounding-dino-base is the
        # default; override via GDINO_MODEL_ID env var if you need the smaller/faster tiny.
        model_id = os.environ.get("GDINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
        log.info(f"Loading Grounding DINO: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        self.initialized = False
        self.object_names = []
        self.object_id_map = {}
        self.inference_state = None
        self.temp_dir = None
        self.frame_idx = 0
        self.last_propagated_frame = -1
        self.window_size = 50
        self.frames_in_window = 0
        self.last_frame_masks = {}
        self.last_frame = None

    def initialize(self, first_frame, object_names, box_threshold=0.4, text_threshold=0.3, gdino_prompts=None):
        """Initialize tracking on first frame.

        Args:
            object_names:  canonical names ["eggplant", "yellow basket", ...]
                           used for downstream identification.
            gdino_prompts: optional list parallel to object_names with visually-
                           grounded query strings (e.g. ["small purple object",
                           "yellow plastic basket"]). When provided we send these
                           to Grounding-DINO instead of the bare names — they
                           drastically improve detection on rendered/simulated
                           objects whose silhouettes don't match the noun word.
                           Detected labels are mapped back to object_names by
                           index via the existing greedy bipartite matcher.
        """
        if not object_names:
            return False
        self.object_names = object_names
        # Use gdino_prompts as the GDINO query when provided; fall back to
        # object_names entry-by-entry if a particular gdino_prompt is missing.
        if gdino_prompts and len(gdino_prompts) == len(object_names):
            query_strings = [gp if gp else nm for gp, nm in zip(gdino_prompts, object_names)]
        else:
            query_strings = list(object_names)
        self.frame_idx = 0
        self.frames_in_window = 0
        self.last_frame_masks = {}
        self.last_frame = None
        self._last_error = None

        import tempfile
        self.temp_dir = tempfile.mkdtemp(prefix="tracking_server_")

        try:
            H, W = first_frame.shape[:2]
            log.info(f"Initializing inference state with video dimensions: H={H}, W={W}")

            import contextlib

            @contextlib.contextmanager
            def redirect_stdout_to_stderr():
                old_stdout = sys.stdout
                sys.stdout = sys.stderr
                try:
                    yield
                finally:
                    sys.stdout = old_stdout

            with redirect_stdout_to_stderr():
                self.inference_state = self.video_predictor.init_state(video_path=None)
            self.inference_state["video_height"] = H
            self.inference_state["video_width"] = W

            log.info(f"Detecting objects with Grounding DINO; query={query_strings}")
            text_prompt = ". ".join(query_strings) + "."
            from PIL import Image
            image_pil = Image.fromarray(first_frame)
            inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

            # Retry loop: progressively relax thresholds until every requested object is
            # matched (or until thresholds bottom out). We rerun only post-processing —
            # the (expensive) DINO forward pass happens once above.
            schedule = []
            _bt, _tt = float(box_threshold), float(text_threshold)
            while True:
                schedule.append((_bt, _tt))
                if _bt <= 0.05 and _tt <= 0.05:
                    break
                _bt = max(0.05, _bt * 0.7)
                _tt = max(0.05, _tt * 0.7)
            log.info(f"GDINO retry schedule (box, text): {schedule}")

            def _assign(object_names, labels, boxes, confidences):
                """Greedy bipartite assignment of detections to object_names.

                Each object_i is matched against its bare `name` AND its
                `gdino_prompt` (from the parallel `query_strings` in the enclosing
                scope) — whichever produces the better tier wins for that object.

                Tiers (prefer lower):
                    0 = exact label match against name or gdino_prompt
                    1 = substring match (term in label or label in term)
                    2 = word-overlap (need >=50% of the term's words present)

                Within a tier, higher confidence wins. Each detection used at most
                once, each object name matched at most once.
                """
                obj_l = [n.lower() for n in object_names]
                # Combine each object's bare name + its gdino_prompt query string
                # into a list of "search terms" — match against any term.
                obj_terms = []
                for i, name_l in enumerate(obj_l):
                    terms = {name_l}
                    if i < len(query_strings):
                        qs_l = query_strings[i].lower()
                        if qs_l:
                            terms.add(qs_l)
                    obj_terms.append(list(terms))
                obj_term_words = [[set(t.split()) for t in terms] for terms in obj_terms]
                candidates = []
                for j, lab in enumerate(labels):
                    lab_l = lab.lower()
                    lab_words = set(lab_l.split())
                    for i, terms in enumerate(obj_terms):
                        best_tier = None
                        for t_idx, term in enumerate(terms):
                            if term == lab_l:
                                tier = 0
                            elif term in lab_l or lab_l in term:
                                tier = 1
                            else:
                                tw = obj_term_words[i][t_idx]
                                shared = tw & lab_words
                                tier = 2 if shared and len(shared) / max(len(tw), 1) >= 0.5 else None
                            if tier is not None and (best_tier is None or tier < best_tier):
                                best_tier = tier
                        if best_tier is not None:
                            candidates.append((best_tier, -float(confidences[j]), i, j))
                candidates.sort()
                used_obj, used_lab = set(), set()
                matched = {}
                for tier, neg_conf, i, j in candidates:
                    if i in used_obj or j in used_lab:
                        continue
                    matched[object_names[i]] = boxes[j]
                    used_obj.add(i)
                    used_lab.add(j)
                return matched

            matched_boxes = {}
            best_matched_boxes = {}  # keep the largest match across attempts in case none is complete
            detected_labels = []
            input_boxes = np.empty((0, 4))
            for attempt, (bt, tt) in enumerate(schedule):
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=bt,
                    text_threshold=tt,
                    target_sizes=[image_pil.size[::-1]],
                )
                input_boxes = results[0]["boxes"].cpu().numpy()
                detected_labels = results[0]["labels"]
                confidences = results[0]["scores"].cpu().numpy()
                log.info(
                    f"  attempt {attempt}: box={bt:.3f} text={tt:.3f} -> "
                    f"{len(input_boxes)} dets, labels={detected_labels}"
                )

                matched_boxes = _assign(object_names, detected_labels, input_boxes, confidences)

                if len(matched_boxes) > len(best_matched_boxes):
                    best_matched_boxes = matched_boxes.copy()

                if len(matched_boxes) == len(object_names):
                    log.info(f"  all {len(object_names)} objects matched at attempt {attempt}")
                    break
            else:
                # Loop exited without a full match — fall back to the best partial we saw.
                matched_boxes = best_matched_boxes

            if not matched_boxes:
                if len(input_boxes) == 0:
                    self._last_error = (
                        f"No objects detected after {len(schedule)} threshold sweeps. "
                        f"Expected: {object_names}"
                    )
                    log.warning(self._last_error)
                    return False
                # Last-ditch positional fallback: assign first N boxes by position.
                for i, obj_name in enumerate(object_names[: len(input_boxes)]):
                    matched_boxes[obj_name] = input_boxes[i]
                log.warning(
                    f"  no name matches; positional fallback for {list(matched_boxes.keys())}"
                )

            missing = [n for n in object_names if n not in matched_boxes]
            if missing:
                log.warning(f"  matched {list(matched_boxes.keys())}, still missing {missing}")
            else:
                log.info(f"Matched boxes: {list(matched_boxes.keys())}")
            self.initial_matched_boxes = matched_boxes.copy()

            log.info("Getting masks from SAM 2 image predictor...")
            self.image_predictor.set_image(np.array(first_frame))
            all_boxes = np.array([matched_boxes[name] for name in object_names if name in matched_boxes])
            masks, scores, logits = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=all_boxes,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            self.initial_masks = masks.copy() if len(masks) > 0 else None

            log.info("Adding first frame to inference state...")
            first_frame_np = np.ascontiguousarray(first_frame)
            first_frame_idx = self.video_predictor.add_new_frame(self.inference_state, first_frame_np)
            log.info(f"Added first frame, got frame_idx={first_frame_idx}")

            self.object_id_map = {}
            obj_idx = 0
            for obj_name in object_names:
                if obj_name not in matched_boxes:
                    log.warning(f"Object '{obj_name}' not found in matched_boxes, skipping")
                    continue
                object_id = obj_idx + 1
                self.object_id_map[obj_name] = object_id
                if self.prompt_type == "box":
                    box = matched_boxes[obj_name]
                    log.debug(f"Registering {obj_name} with object_id={object_id}, box={box.tolist()}, frame_idx={first_frame_idx}")
                    _, out_obj_ids, _ = self.video_predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=first_frame_idx,
                        obj_id=object_id,
                        box=box,
                    )
                    log.debug(f"Registered {obj_name} with object_id={object_id}, box={box.tolist()}")
                elif self.prompt_type == "mask":
                    mask_idx = list(matched_boxes.keys()).index(obj_name)
                    if mask_idx < len(masks):
                        mask = masks[mask_idx]
                        _, out_obj_ids, _ = self.video_predictor.add_new_mask(
                            inference_state=self.inference_state,
                            frame_idx=first_frame_idx,
                            obj_id=object_id,
                            mask=mask,
                        )
                        log.debug(f"Registered {obj_name} with object_id={object_id} using mask")
                elif self.prompt_type == "point":
                    mask_idx = list(matched_boxes.keys()).index(obj_name)
                    if mask_idx < len(masks):
                        mask = masks[mask_idx]
                        points = sample_points_from_masks(masks=mask[None], num_points=10)
                        if len(points) > 0:
                            labels = np.ones((points.shape[0]), dtype=np.int32)
                            _, out_obj_ids, _ = self.video_predictor.add_new_points_or_box(
                                inference_state=self.inference_state,
                                frame_idx=first_frame_idx,
                                obj_id=object_id,
                                points=points[0],
                                labels=labels,
                            )
                            log.debug(f"Registered {obj_name} with object_id={object_id} using points")
                obj_idx += 1

            if len(self.object_id_map) == 0:
                self._last_error = f"No objects were registered for tracking! Matched boxes: {matched_boxes}, Object names: {object_names}"
                log.error(self._last_error)
                return False

            log.info("Inferring tracking for the first frame after registration...")
            try:
                self.video_predictor.infer_single_frame(self.inference_state, first_frame_idx)
            except Exception as e:
                log.warning(f"Failed to infer first frame (this is OK for streaming mode): {e}")

            self.initialized = True
            log.info(f"Tracking initialized for {len(self.object_id_map)} objects: {list(self.object_id_map.keys())}")
            log.debug(f"Object ID map: {self.object_id_map}")
            return True

        except Exception as e:
            self._last_error = f"Exception during initialization: {str(e)}"
            log.error(f"Failed to initialize tracking: {e}", exc_info=True)
            return False

    def _reinitialize_from_frame(self, frame, current_masks):
        """Reinitialize tracking from current frame using existing masks (sliding window OOM guard)."""
        import contextlib
        import gc

        log.info(f"Reinitializing tracking at frame {self.frame_idx} (window size: {self.window_size})")

        if torch.cuda.is_available():
            if hasattr(self, "inference_state") and self.inference_state is not None:
                for key, value in list(self.inference_state.items()):
                    if isinstance(value, torch.Tensor):
                        if value.is_cuda:
                            value_cpu = value.cpu()
                            del value
                            del value_cpu
                        else:
                            del value
                        if key in self.inference_state:
                            del self.inference_state[key]
                self.inference_state.clear()
                del self.inference_state
                self.inference_state = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        @contextlib.contextmanager
        def redirect_stdout_to_stderr():
            old_stdout = sys.stdout
            sys.stdout = sys.stderr
            try:
                yield
            finally:
                sys.stdout = old_stdout

        with redirect_stdout_to_stderr():
            self.inference_state = self.video_predictor.init_state(video_path=None)

        H, W = frame.shape[:2]
        self.inference_state["video_height"] = H
        self.inference_state["video_width"] = W

        frame_np = np.ascontiguousarray(frame)
        first_frame_idx = self.video_predictor.add_new_frame(self.inference_state, frame_np)

        self.object_id_map = {}
        obj_idx = 0
        for obj_name in self.object_names:
            if obj_name not in current_masks or current_masks[obj_name] is None:
                log.warning(f"Object '{obj_name}' has no mask for reinitialization, skipping")
                continue

            object_id = obj_idx + 1
            self.object_id_map[obj_name] = object_id

            mask = current_masks[obj_name]
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                log.warning(f"Object '{obj_name}' mask has unexpected shape {mask.shape}, skipping")
                continue

            try:
                self.video_predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=first_frame_idx,
                    obj_id=object_id,
                    mask=mask,
                )
                log.debug(f"Re-registered {obj_name} with object_id={object_id} using mask")
            except Exception as e:
                log.warning(f"Failed to re-register {obj_name} with mask: {e}")
                y_coords, x_coords = np.where(mask)
                if len(x_coords) > 0 and len(y_coords) > 0:
                    x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
                    y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
                    box = np.array([x_min, y_min, x_max, y_max])
                    try:
                        self.video_predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=first_frame_idx,
                            obj_id=object_id,
                            box=box,
                        )
                        log.debug(f"Re-registered {obj_name} with object_id={object_id} using box")
                    except Exception as e2:
                        log.warning(f"Failed to re-register {obj_name} with box: {e2}")
                        continue
            obj_idx += 1

        if len(self.object_id_map) == 0:
            log.error("No objects were re-registered for tracking!")
            return False

        try:
            self.video_predictor.infer_single_frame(self.inference_state, first_frame_idx)
        except Exception as e:
            log.warning(f"Failed to infer reinitialized frame: {e}")

        self.frames_in_window = 0
        self.last_frame = None
        self.last_frame_masks = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        log.info(f"Tracking reinitialized for {len(self.object_id_map)} objects: {list(self.object_id_map.keys())}")
        return True

    def track(self, frame):
        """Track objects in frame using incremental tracking."""
        if not self.initialized:
            return {name: None for name in self.object_names}

        try:
            self.frame_idx += 1
            if self.frames_in_window >= self.window_size:
                if len(self.last_frame_masks) > 0 and self.last_frame is not None:
                    log.info(f"Reinitializing tracking at frame {self.frame_idx} (window size: {self.window_size}) to prevent OOM")
                    success = self._reinitialize_from_frame(self.last_frame, self.last_frame_masks)
                    if not success:
                        log.warning("Failed to reinitialize tracking, resetting counter (may OOM)")
                        self.frames_in_window = 0
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    log.warning("No previous masks/frame available for reinitialization, resetting counter (may OOM)")
                    self.frames_in_window = 0
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            self.frames_in_window += 1
            frame_np = np.ascontiguousarray(frame)
            new_frame_idx = self.video_predictor.add_new_frame(self.inference_state, frame_np)
            if self.frame_idx <= 5:
                log.debug(f"Added frame {self.frame_idx} to inference state, got frame_idx={new_frame_idx}")

            out_frame_idx, out_obj_ids, video_res_masks = self.video_predictor.infer_single_frame(
                self.inference_state, new_frame_idx
            )

            if self.frame_idx <= 5:
                log.debug(
                    f"Tracking frame {self.frame_idx}: inferred frame_idx={out_frame_idx}, "
                    f"obj_ids={out_obj_ids}, masks shape={video_res_masks.shape if video_res_masks is not None else None}"
                )

            result = {}
            if video_res_masks is not None and len(out_obj_ids) > 0:
                masks_np = (video_res_masks > 0.0).cpu().numpy()
                for i, obj_id in enumerate(out_obj_ids):
                    obj_name = None
                    for name, mapped_id in self.object_id_map.items():
                        if mapped_id == obj_id:
                            obj_name = name
                            break
                    if obj_name is None:
                        if self.frame_idx <= 5:
                            log.warning(f"Object ID {obj_id} not found in object_id_map {self.object_id_map}")
                        continue
                    if i < len(masks_np):
                        mask = np.squeeze(masks_np[i])
                        if mask.ndim != 2:
                            log.warning(f"Mask has unexpected shape {mask.shape}, expected 2D")
                            result[obj_name] = None
                            continue
                        y_coords, x_coords = np.where(mask)
                        if len(x_coords) > 0 and len(y_coords) > 0:
                            center_x = float(np.mean(x_coords))
                            center_y = float(np.mean(y_coords))
                            x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
                            y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
                            result[obj_name] = {
                                "center": [center_x, center_y],
                                "bbox": [x_min, y_min, x_max, y_max],
                            }
                            if self.frame_idx <= 5:
                                log.debug(f"✓ Tracked {obj_name} (id={obj_id}) at [{center_x:.1f}, {center_y:.1f}], bbox=[{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]")
                        else:
                            result[obj_name] = None
                            if self.frame_idx <= 5:
                                log.warning(f"✗ Tracked {obj_name} (id={obj_id}) but mask is empty")
                    else:
                        result[obj_name] = None
                        if self.frame_idx <= 5:
                            log.warning(f"✗ Tracked {obj_name} (id={obj_id}) but mask index {i} out of range")

                for obj_name in self.object_names:
                    if obj_name not in result:
                        result[obj_name] = None

                self.last_frame_masks = {}
                if video_res_masks is not None and len(out_obj_ids) > 0:
                    masks_np = (video_res_masks > 0.0).cpu().numpy()
                    for i, obj_id in enumerate(out_obj_ids):
                        obj_name = None
                        for name, mapped_id in self.object_id_map.items():
                            if mapped_id == obj_id:
                                obj_name = name
                                break
                        if obj_name is not None and i < len(masks_np):
                            mask = np.squeeze(masks_np[i])
                            if mask.ndim == 2:
                                self.last_frame_masks[obj_name] = mask.copy()
            else:
                log.warning(f"No masks returned for frame {self.frame_idx}")
                result = {name: None for name in self.object_names}

            self.last_frame = frame.copy()
            if self.frame_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result

        except Exception as e:
            log.warning(f"Tracking failed for frame {self.frame_idx}: {e}", exc_info=True)
            return {name: None for name in self.object_names}

    def cleanup(self):
        """Reset per-episode state so the same subprocess can serve the next episode."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass
            self.temp_dir = None
        if hasattr(self, "inference_state") and self.inference_state is not None:
            try:
                for key, value in list(self.inference_state.items()):
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        del value
                self.inference_state.clear()
            except Exception:
                pass
            self.inference_state = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.initialized = False
        self.object_names = []
        self.object_id_map = {}
        self.frame_idx = 0
        self.frames_in_window = 0
        self.last_frame_masks = {}
        self.last_frame = None
        self.initial_matched_boxes = {}
        self.initial_masks = None
        self._last_error = None


def run_tracking_server(args):
    """Main server loop shared by CLI wrapper."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", message=".*Memory efficient kernel not used.*")
    warnings.filterwarnings("ignore", message=".*Memory Efficient attention has been runtime disabled.*")
    warnings.filterwarnings("ignore", message=".*Flash attention kernel not used.*")
    warnings.filterwarnings("ignore", message=".*Expected query, key and value to all be of dtype.*")
    warnings.filterwarnings("ignore", message=".*cuDNN attention kernel not used.*")
    warnings.filterwarnings("ignore", message=".*Flash Attention kernel failed due to.*")
    warnings.filterwarnings("ignore", message=".*Falling back to all available kernels.*")
    warnings.filterwarnings("ignore", message=".*cannot import name '_C'.*")
    warnings.filterwarnings("ignore", message=".*Skipping the post-processing step.*")

    server = None
    try:
        log.info("Starting TrackingServer initialization...")
        server = TrackingServer(
            args.sam2_checkpoint,
            args.model_cfg,
            args.device,
            args.prompt_type,
        )
        log.info("TrackingServer initialized successfully")
        ready_response = {"status": "ready"}
        print(json.dumps(ready_response))
        sys.stdout.flush()
    except Exception as e:
        error_msg = f"Failed to initialize TrackingServer: {str(e)}\n{traceback.format_exc()}"
        log.error(error_msg)
        sys.stderr.write(error_msg + "\n")
        sys.stderr.flush()
        error_response = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        print(json.dumps(error_response))
        sys.stdout.flush()
        sys.exit(1)

    log.info("Tracking server ready, waiting for commands...")
    for line in sys.stdin:
        try:
            if not line.strip():
                continue
            log.debug(f"Received command line (length: {len(line)})")
            cmd = json.loads(line.strip())
            cmd_type = cmd.get("type")
            log.debug(f"Parsed command type: {cmd_type}")

            if cmd_type == "init":
                log.debug(f"Received init command for objects: {cmd.get('object_names', [])}")
                sys.stderr.flush()
                try:
                    log.debug("Decoding frame...")
                    sys.stderr.flush()
                    frame_data = base64.b64decode(cmd["frame"])
                    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        response = {"success": False, "error": "Failed to decode frame"}
                        log.error("Failed to decode frame")
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        log.debug(f"Frame decoded, shape: {frame.shape}, calling server.initialize()...")
                        sys.stderr.flush()
                        success = server.initialize(
                            frame,
                            cmd["object_names"],
                            box_threshold=float(cmd.get("box_threshold", 0.4)),
                            text_threshold=float(cmd.get("text_threshold", 0.3)),
                            gdino_prompts=cmd.get("gdino_prompts"),
                        )
                        log.debug(f"server.initialize() returned: {success}")
                        sys.stderr.flush()
                        if success:
                            response = {"success": True}
                            log.debug("Initialization successful")
                        else:
                            error_msg = getattr(server, "_last_error", "Initialization failed (unknown reason)")
                            response = {"success": False, "error": error_msg}
                            log.error(f"Initialization failed: {error_msg}")
                except Exception as e:
                    error_tb = traceback.format_exc()
                    response = {"success": False, "error": str(e), "traceback": error_tb}
                    log.error(f"Exception during init: {e}\n{error_tb}")
                    sys.stderr.flush()
                log.debug(f"Sending response: success={response.get('success', False)}")
                sys.stderr.flush()
                print(json.dumps(response))
                sys.stdout.flush()
                continue

            elif cmd_type == "track":
                frame_data = base64.b64decode(cmd["frame"])
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    response = {"error": "Failed to decode frame"}
                    print(json.dumps(response))
                    sys.stdout.flush()
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = server.track(frame)
                    print(json.dumps(result))
                    sys.stdout.flush()

            elif cmd_type == "cleanup":
                server.cleanup()
                response = {"success": True}
                print(json.dumps(response))
                sys.stdout.flush()
            else:
                response = {"error": f"Unknown command type: {cmd_type}"}
                print(json.dumps(response))
                sys.stdout.flush()

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON command: {str(e)}"
            log.error(error_msg)
            sys.stderr.write(error_msg + "\n")
            sys.stderr.flush()
            response = {"error": error_msg}
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            error_msg = f"Error processing command: {str(e)}\n{traceback.format_exc()}"
            log.error(error_msg)
            sys.stderr.write(error_msg + "\n")
            sys.stderr.flush()
            response = {"error": str(e), "traceback": traceback.format_exc()}
            print(json.dumps(response))
            sys.stdout.flush()


