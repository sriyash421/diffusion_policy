# Vendored verbatim from veritas@8896225 src/agent/utils/tracker_adapter.py (exact upstream Grounded-SAM-2 object
# hints); only the src.agent import paths are rewritten. Do not edit; re-vendor instead.
"""
Tracker adapter for window-based tracking with stitching.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from diffusion_policy.env.simpler.veritas.gsam.object_tracker import ObjectTracker

log = logging.getLogger(__name__)


class TrackDet:
    """Tracked detection."""
    
    def __init__(
        self,
        track_id: str,
        object_name: str,
        role: str,
        bbox: np.ndarray,  # [x_min, y_min, x_max, y_max]
        confidence: float = 1.0,
    ):
        self.track_id = track_id
        self.object_name = object_name
        self.role = role
        self.bbox = bbox  # [x_min, y_min, x_max, y_max]
        self.confidence = confidence
    
    @property
    def center(self) -> np.ndarray:
        """Get bbox center [x, y]."""
        x_min, y_min, x_max, y_max = self.bbox
        return np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float64)
    
    @property
    def width(self) -> float:
        """Get bbox width."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Get bbox height."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Get bbox area."""
        return self.width * self.height


class TrackerAdapter:
    """
    Adapter for ObjectTracker that handles window-based tracking with stitching.
    
    Tracks objects in 50-frame windows and stitches results across windows.
    """
    
    def __init__(
        self,
        tracker: ObjectTracker,
        window_size: int = 50,
    ):
        """
        Initialize tracker adapter.
        
        Args:
            tracker: ObjectTracker instance
            window_size: Window size for tracking (default: 50)
        """
        self.tracker = tracker
        self.window_size = window_size
        
        # Track state
        self.object_names: List[str] = []
        self.object_roles: Dict[str, str] = {}  # object_name -> role
        self.track_ids: Dict[str, str] = {}  # object_name -> track_id
        
        # Window state
        self.current_window_start = 0
        self.frame_idx = 0
        self.last_bboxes: Dict[str, Optional[np.ndarray]] = {}  # object_name -> last bbox
        self.initialized: bool = False
    
    def initialize(
        self,
        first_frame: np.ndarray,
        objects: List[Tuple[str, str]],  # (object_name, role)
        gdino_prompts: Optional[List[str]] = None,
    ) -> bool:
        """
        Initialize tracking for objects.

        Args:
            first_frame:   First frame [H, W, 3] uint8 RGB
            objects:       List of (object_name, role) tuples
            gdino_prompts: Optional parallel list of visually-grounded query
                           strings to feed Grounding-DINO instead of the bare
                           object names (e.g. ["small purple object", ...]).

        Returns:
            True if successful
        """
        self.object_names = [name for name, _ in objects]
        self.object_roles = {name: role for name, role in objects}

        # Generate track IDs (slug from object_name or index)
        for i, (name, _) in enumerate(objects):
            # Simple track ID: use object name or index
            track_id = name.lower().replace(" ", "_") if name else f"obj_{i}"
            self.track_ids[name] = track_id

        # Use human-readable prompts for detection (underscores -> spaces)
        prompt_names = [name.replace("_", " ") for name in self.object_names]

        # Initialize tracker
        success = self.tracker.initialize(first_frame, prompt_names, gdino_prompts=gdino_prompts)
        
        if success:
            self.initialized = True
            # Get initial bboxes
            initial_results = self.tracker.track(first_frame)
            for obj_name in self.object_names:
                result = self._lookup_result(initial_results, obj_name)
                if result is not None:
                    if isinstance(result, dict):
                        if "bbox" in result:
                            self.last_bboxes[obj_name] = np.array(result["bbox"], dtype=np.float64)
                        elif "center" in result:
                            cx, cy = result["center"]
                            self.last_bboxes[obj_name] = np.array([cx - 1, cy - 1, cx + 1, cy + 1], dtype=np.float64)
                        else:
                            self.last_bboxes[obj_name] = None
                    else:
                        try:
                            self.last_bboxes[obj_name] = np.array(result, dtype=np.float64)
                        except Exception:
                            self.last_bboxes[obj_name] = None
                else:
                    self.last_bboxes[obj_name] = None
            
            self.current_window_start = 0
            self.frame_idx = 0
            log.debug(f"Tracker adapter initialized for {len(self.object_names)} objects")
        
        return success
    
    def track_frame(self, frame: np.ndarray) -> List[TrackDet]:
        """
        Track objects in current frame (handles windowing internally).
        
        Args:
            frame: Current frame [H, W, 3] uint8 RGB
        
        Returns:
            List of TrackDet for current frame
        """
        # Check if we need to reinitialize window
        frames_in_window = self.frame_idx - self.current_window_start
        if frames_in_window >= self.window_size:
            # Reinitialize tracking with last known bboxes
            log.debug(f"Reinitializing tracking window at frame {self.frame_idx}")
            # The tracker handles reinitialization internally via sliding window
            # We just need to track the frame
            self.current_window_start = self.frame_idx
        
        # Track current frame
        results = self.tracker.track(frame)
        
        # Convert to TrackDet list
        detections = []
        for obj_name in self.object_names:
            track_id = self.track_ids[obj_name]
            role = self.object_roles.get(obj_name, "other")
            
            result = self._lookup_result(results, obj_name)
            if result is not None:
                if isinstance(result, dict):
                    if "bbox" in result:
                        bbox = np.array(result["bbox"], dtype=np.float64)  # [x_min, y_min, x_max, y_max]
                        confidence = 1.0  # Tracker doesn't provide confidence, use 1.0
                        self.last_bboxes[obj_name] = bbox
                    elif "center" in result:
                        # Build a small box around center if bbox missing
                        cx, cy = result["center"]
                        bbox = np.array([cx - 1, cy - 1, cx + 1, cy + 1], dtype=np.float64)
                        confidence = 0.6
                        self.last_bboxes[obj_name] = bbox
                    else:
                        # Fallback: use last known bbox
                        bbox = self.last_bboxes.get(obj_name)
                        confidence = 0.5 if bbox is not None else 0.0
                else:
                    # Fallback: use last known bbox
                    bbox = self.last_bboxes.get(obj_name)
                    confidence = 0.5 if bbox is not None else 0.0
            else:
                # Object not detected: use last known bbox with low confidence
                bbox = self.last_bboxes.get(obj_name)
                confidence = 0.3 if bbox is not None else 0.0
            
            if bbox is not None:
                detections.append(TrackDet(track_id, obj_name, role, bbox, confidence))
        
        self.frame_idx += 1
        return detections

    def _lookup_result(self, results: Dict[str, object], obj_name: str):
        """Find result by matching name with common normalizations (space/underscore, lowercase)."""
        if results is None:
            return None
        if obj_name in results and results[obj_name] is not None:
            return results[obj_name]
        # Build variants
        variants = {
            obj_name,
            obj_name.replace(" ", "_"),
            obj_name.replace("_", " "),
            obj_name.lower(),
            obj_name.lower().replace(" ", "_"),
            obj_name.lower().replace("_", " "),
        }
        for k, v in results.items():
            if v is None:
                continue
            key_variants = {
                k,
                k.replace(" ", "_"),
                k.replace("_", " "),
                k.lower(),
                k.lower().replace(" ", "_"),
                k.lower().replace("_", " "),
            }
            if variants & key_variants:
                return v
        return None
    
    def get_active_objects(self, detections: List[TrackDet]) -> Dict[str, TrackDet]:
        """Get dict of active objects by track_id."""
        return {det.track_id: det for det in detections if det.confidence > 0.0}

