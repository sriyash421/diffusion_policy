# Vendored verbatim from veritas@8896225 src/agent/object_tracker.py (exact upstream Grounded-SAM-2 object
# hints); only the src.agent import paths are rewritten. Do not edit; re-vendor instead.
"""
Object tracking module using Grounded-SAM-2 for verifier-based object detection.

This module provides object tracking capabilities that replace simulation-based
object position queries with vision-based tracking using Grounding DINO and SAM 2.

The tracking runs in a separate subprocess with the Grounded-SAM-2 environment
to avoid dependency conflicts.
"""

import base64
import collections
import json
import logging
import os
import subprocess
import sys
import threading
from typing import Dict, List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Always available since we use subprocess
GROUNDED_SAM2_AVAILABLE = True


class ObjectTracker:
    """
    Tracks objects across video frames using Grounded-SAM-2.
    
    The tracking runs in a separate subprocess with the Grounded-SAM-2 environment.
    
    Usage:
        tracker = ObjectTracker(sam2_checkpoint, model_cfg, device="cuda")
        tracker.initialize(frame, object_names=["cube", "apple"])
        bbox_centers = tracker.track(frame)  # Returns dict: {"cube": [x, y], "apple": [x, y]}
    """
    
    def __init__(
        self,
        sam2_checkpoint: str,
        model_cfg: str,
        device: str = "cuda",
        prompt_type: str = "box",
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        grounded_sam2_python: Optional[str] = None,
    ):
        """
        Initialize the tracker.
        
        Args:
            sam2_checkpoint: Path to SAM 2 checkpoint
            model_cfg: Path to SAM 2 model config (relative to sam2/configs, e.g., "sam2.1/sam2.1_hiera_l")
            device: Device to run on ("cuda" or "cpu")
            prompt_type: Type of prompt for video predictor ("point", "box", or "mask")
            box_threshold: Threshold for Grounding DINO box detection
            text_threshold: Threshold for Grounding DINO text matching
            grounded_sam2_python: Path to Python executable in Grounded-SAM-2 environment
                                 (default: tries to find it automatically)
        """
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.device = device
        self.prompt_type = prompt_type
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Resolve the Python executable for the tracking-server subprocess.
        # Order: explicit ctor arg → GROUNDED_SAM2_PYTHON env var → current interpreter.
        if not grounded_sam2_python or not os.path.exists(str(grounded_sam2_python)):
            grounded_sam2_python = self._find_grounded_sam2_python()
        if not os.path.exists(grounded_sam2_python):
            raise FileNotFoundError(
                f"Grounded-SAM-2 Python not found at: {grounded_sam2_python}. "
                f"Run `bash scripts/setup_env.sh` to install sam2 into the project venv."
            )
        self.grounded_sam2_python = grounded_sam2_python

        # Path to tracking server script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.tracking_server_script = os.path.join(script_dir, "tracking_server.py")
        
        if not os.path.exists(self.tracking_server_script):
            raise FileNotFoundError(
                f"Tracking server script not found: {self.tracking_server_script}"
            )
        
        # Extract relative config name from absolute path if needed
        import os.path as osp
        if osp.isabs(model_cfg):
            # Extract relative part from sam2/configs
            if 'configs' in model_cfg:
                configs_idx = model_cfg.rindex('configs')
                rel_path = model_cfg[configs_idx + len('configs') + 1:]
                config_name = rel_path.replace('.yaml', '')
            else:
                config_name = osp.basename(model_cfg).replace('.yaml', '')
        else:
            config_name = model_cfg.replace('configs/', '').replace('.yaml', '')
        
        self.config_name = config_name
        
        # Subprocess for tracking server
        self.process = None
        self.initialized = False
        self.object_names: List[str] = []
        # Ring buffer of last stderr lines from the subprocess (filled by _start_server).
        self._stderr_tail: collections.deque = collections.deque(maxlen=200)
        self._stderr_thread: Optional[threading.Thread] = None
        # Suppress repeated "Tracker not initialized" warnings — emit once per init failure.
        self._warned_not_initialized = False

    def _format_stderr_tail(self) -> str:
        """Return the captured tail of the subprocess's stderr, joined."""
        if not getattr(self, "_stderr_tail", None):
            return "<no stderr captured>"
        return "\n".join(self._stderr_tail)

    def _find_grounded_sam2_python(self) -> str:
        """Resolve the Python executable for the tracking-server subprocess.

        Order: GROUNDED_SAM2_PYTHON env var → sys.executable (if it can import sam2).
        Single-venv layout (sam2 installed into .venv via uv) is the supported config.
        """
        env_python = os.environ.get("GROUNDED_SAM2_PYTHON", "")
        if env_python and os.path.exists(env_python):
            log.info(f"Using GROUNDED_SAM2_PYTHON: {env_python}")
            return env_python

        try:
            result = subprocess.run(
                [sys.executable, "-c", "import sam2"], capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                log.info(f"Using current interpreter for tracking server: {sys.executable}")
                return sys.executable
        except Exception as e:
            log.debug(f"sys.executable sam2 probe failed: {e}")

        raise FileNotFoundError(
            "Could not locate a Python with sam2 installed. "
            "Run `bash scripts/setup_env.sh` (installs sam2 into .venv) "
            "or set GROUNDED_SAM2_PYTHON to a Python executable that can `import sam2`."
        )
    
    def _start_server(self):
        """Start the tracking server subprocess (or restart if a prior one died)."""
        # If an existing subprocess is still alive, reuse it.
        if self.process is not None and self.process.poll() is None:
            return
        # Otherwise drop any stale handle so we spawn fresh.
        if self.process is not None:
            log.warning(
                f"Previous tracking server died (exit={self.process.returncode}); "
                f"respawning. Last stderr:\n{self._format_stderr_tail()}"
            )
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                try:
                    if stream is not None and not stream.closed:
                        stream.close()
                except Exception:
                    pass
            self.process = None
            self.initialized = False
        # Ring buffer for the subprocess's stderr — replaces the DEBUG-only stream
        # so we still have the last ~200 lines available when the server dies.
        self._stderr_tail = collections.deque(maxlen=200)
        
        # Handle micromamba run command
        if self.grounded_sam2_python.startswith("MICROMAMBA:"):
            # Extract micromamba path and construct command
            micromamba_bin = self.grounded_sam2_python.replace("MICROMAMBA:", "")
            cmd = [
                micromamba_bin,
                "run",
                "-n", "grounded-sam2",
                "python",
                self.tracking_server_script,
                "--sam2_checkpoint", self.sam2_checkpoint,
                "--model_cfg", self.config_name,
                "--device", self.device,
                "--prompt_type", self.prompt_type,
            ]
        else:
            # Validate Python path before using it
            if not os.path.exists(self.grounded_sam2_python):
                raise FileNotFoundError(
                    f"Grounded-SAM-2 Python not found: {self.grounded_sam2_python}\n"
                    f"Please set GROUNDED_SAM2_PYTHON environment variable correctly."
                )
            
            # Add -u flag for unbuffered I/O
            cmd = [
                self.grounded_sam2_python,
                "-u",  # Unbuffered output
                self.tracking_server_script,
                "--sam2_checkpoint", self.sam2_checkpoint,
                "--model_cfg", self.config_name,
                "--device", self.device,
                "--prompt_type", self.prompt_type,
            ]
        
        log.debug(f"Starting tracking server: {' '.join(cmd)}")
        
        # Set environment variables for Grounded-SAM-2
        env = os.environ.copy()
        # Default to the in-tree submodule; the eval script overrides this via GROUNDED_SAM2_PATH.
        _default_gsam2 = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "Grounded-SAM-2",
        )
        grounded_sam2_path = os.environ.get("GROUNDED_SAM2_PATH", _default_gsam2)
        env["GROUNDED_SAM2_PATH"] = grounded_sam2_path
        env["PYTHONPATH"] = f"{grounded_sam2_path}:{env.get('PYTHONPATH', '')}"
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=False,  # Use binary mode for base64
            bufsize=0,  # Unbuffered
        )
        
        # Stream subprocess stderr into a ring buffer so we can include the tail
        # in any error message. Logging it at DEBUG also keeps it visible when
        # the user enables debug logging.
        import errno
        def log_stderr():
            while True:
                try:
                    line = self.process.stderr.readline()
                    if not line:
                        break
                    decoded = line.decode('utf-8', errors='ignore').rstrip()
                    self._stderr_tail.append(decoded)
                    log.debug(f"[tracking-server-stderr] {decoded}")
                except OSError as e:
                    if e.errno == errno.EINTR:
                        continue
                    log.debug(f"Stderr reading thread error: {e}")
                    break
                except Exception as e:
                    log.debug(f"Stderr reading thread exception: {e}")
                    break

        stderr_thread = threading.Thread(target=log_stderr, daemon=True)
        stderr_thread.start()
        self._stderr_thread = stderr_thread
        
        log.debug("Tracking server started, waiting for ready signal...")
        
        # Wait for ready signal from server
        # Note: SAM2's init_state() may print to stdout, so we need to skip non-JSON lines
        import errno
        try:
            while True:
                try:
                    ready_line = self.process.stdout.readline()
                except OSError as e:
                    # Handle EINTR (Interrupted system call) - retry
                    if e.errno == errno.EINTR:
                        continue
                    # For other errors, re-raise
                    raise
                
                if not ready_line:
                    # Check if process died
                    return_code = self.process.poll()
                    if return_code is not None:
                        # Give the stderr thread a moment to drain remaining bytes into the ring buffer.
                        if self._stderr_thread is not None:
                            self._stderr_thread.join(timeout=2.0)
                        raise RuntimeError(
                            f"Tracking server exited with code {return_code} during initialization.\n"
                            f"Stderr tail (last {len(self._stderr_tail)} lines):\n"
                            f"{self._format_stderr_tail()}"
                        )
                    raise RuntimeError(
                        "Tracking server closed connection before ready signal.\n"
                        f"Stderr tail (last {len(self._stderr_tail)} lines):\n"
                        f"{self._format_stderr_tail()}"
                    )
                
                # Try to parse as JSON - if it fails, it's probably a print statement from SAM2
                try:
                    ready_response = json.loads(ready_line.decode('utf-8').strip())
                    # If we successfully parsed JSON, check if it's the ready signal
                    if ready_response.get("status") == "ready":
                        break
                    # If it's not the ready signal, continue reading
                except json.JSONDecodeError:
                    # This is not JSON, probably a print statement from SAM2
                    # Log it and continue reading
                    log.debug(f"Skipping non-JSON line from tracking server: {ready_line.decode('utf-8', errors='ignore').strip()}")
                    continue
            if ready_response.get("status") == "error":
                error_msg = ready_response.get("error", "Unknown error")
                traceback = ready_response.get("traceback", "")
                raise RuntimeError(f"Tracking server initialization failed: {error_msg}\n{traceback}")
            elif ready_response.get("status") != "ready":
                log.warning(f"Unexpected ready response: {ready_response}")
            
            log.debug("Tracking server ready")
        except json.JSONDecodeError as e:
            # Try to read stderr for error info
            stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
            raise RuntimeError(
                f"Failed to parse ready response from tracking server: {e}\n"
                f"Stderr output:\n{stderr_output}"
            )
    
    def _send_command(self, cmd: dict) -> dict:
        """Send command to tracking server and get response."""
        if self.process is None:
            self._start_server()
        
        # Check if process is still alive
        if self.process.poll() is not None:
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=2.0)
            raise RuntimeError(
                f"Tracking server process died (exit code {self.process.returncode}).\n"
                f"Stderr tail (last {len(self._stderr_tail)} lines):\n"
                f"{self._format_stderr_tail()}"
            )
        
        # Send command as JSON line
        cmd_json = json.dumps(cmd) + "\n"
        cmd_type = cmd.get('type', 'unknown')
        log.debug(f"Sending command to tracking server: type={cmd_type}, json_length={len(cmd_json)}")
        try:
            self.process.stdin.write(cmd_json.encode('utf-8'))
            self.process.stdin.flush()
            log.debug("Command sent and flushed to stdin")
        except BrokenPipeError:
            stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
            log.error(f"Broken pipe error. Stderr: {stderr_output}")
            raise RuntimeError(
                f"Tracking server closed stdin pipe.\n"
                f"Stderr output:\n{stderr_output}"
            )
        
        # Read response with timeout handling
        log.debug("Waiting for response from tracking server (reading from stdout)...")
        # readline() should block until a newline is received
        # Note: SAM2 may print non-JSON lines to stdout, so we need to skip them
        import errno
        response_line = None
        while True:
            try:
                line = self.process.stdout.readline()
            except OSError as e:
                # Handle EINTR (Interrupted system call) - retry
                if e.errno == errno.EINTR:
                    continue
                # For other errors, break and check process status
                log.debug(f"Error reading from stdout: {e}")
                break
            
            if not line:
                break
            log.debug(f"Received line (length: {len(line)})")
            # Try to parse as JSON - if it fails, it's probably a print statement from SAM2
            try:
                line_str = line.decode('utf-8').strip()
                # Try to parse as JSON
                json.loads(line_str)
                # If successful, this is our JSON response
                response_line = line
                log.debug(f"Found JSON response: {line_str[:100]}")
                break
            except json.JSONDecodeError:
                # This is not JSON, probably a print statement from SAM2
                # Log it and continue reading
                log.debug(f"Skipping non-JSON line from tracking server: {line_str[:100]}")
                continue
        
        if response_line:
            log.debug(f"Using response line (length: {len(response_line)})")
        else:
            log.warning("No response line received, checking if process is alive...")
            # Check stderr for any error messages
            import select
            if hasattr(select, 'select'):
                # Check if there's data in stderr
                ready, _, _ = select.select([self.process.stderr], [], [], 0.1)
                if ready:
                    stderr_data = self.process.stderr.read(1024).decode('utf-8', errors='ignore')
                    if stderr_data:
                        log.warning(f"Stderr output from tracking server: {stderr_data}")
        if not response_line:
            # Check if process died
            return_code = self.process.poll()
            if return_code is not None:
                stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
                raise RuntimeError(
                    f"Tracking server exited with code {return_code}.\n"
                    f"Stderr output:\n{stderr_output}"
                )
            raise RuntimeError("Tracking server closed connection (no response)")
        
        try:
            log.debug("Decoding response line from bytes to string...")
            response_str = response_line.decode('utf-8')
            log.debug(f"Response string decoded, length: {len(response_str)}, preview: {response_str[:100]}")
            log.debug("Parsing JSON response...")
            response = json.loads(response_str)
            log.debug(f"JSON parsed successfully: {response}")
        except json.JSONDecodeError as e:
            stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
            log.error(f"Failed to parse JSON response: {e}")
            log.error(f"Response line: {response_line.decode('utf-8', errors='ignore')[:200]}")
            raise RuntimeError(
                f"Failed to parse response from tracking server: {e}\n"
                f"Response line: {response_line.decode('utf-8', errors='ignore')[:200]}\n"
                f"Stderr output:\n{stderr_output}"
            )
        
        log.debug("Checking for errors in response...")
        if "error" in response:
            error_msg = response.get("error", "Unknown error")
            traceback = response.get("traceback", "")
            log.error(f"Tracking server returned error: {error_msg}")
            raise RuntimeError(f"Tracking server error: {error_msg}\n{traceback}")
        
        log.debug(f"Response parsed successfully: success={response.get('success', 'N/A')}")
        return response
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64-encoded JPEG."""
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # Encode as base64
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        return frame_b64
    
    def initialize(
        self,
        first_frame: np.ndarray,
        object_names: List[str],
        gdino_prompts: Optional[List[str]] = None,
    ) -> bool:
        """
        Initialize tracking on the first frame.

        Args:
            first_frame:   First frame as [H, W, 3] uint8 RGB image
            object_names:  Canonical names for downstream identification
                           (e.g. ["eggplant", "yellow basket"]).
            gdino_prompts: Optional parallel list of visually-grounded query
                           strings (e.g. ["small purple object", "yellow basket"]).
                           When supplied, these are what we feed to Grounding-DINO
                           instead of the bare object names — significantly
                           improves recall on rendered/simulated scenes.

        Returns:
            True if initialization successful, False otherwise
        """
        if not object_names:
            log.warning("No object names provided for tracking")
            return False

        self.object_names = object_names

        try:
            self._start_server()

            # Send init command
            log.debug(f"Sending initialization command for {len(object_names)} objects: {object_names}")
            frame_b64 = self._frame_to_base64(first_frame)
            log.debug(f"Frame encoded to base64, length: {len(frame_b64)}")
            cmd = {
                "type": "init",
                "frame": frame_b64,
                "object_names": object_names,
                "gdino_prompts": gdino_prompts,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
            }
            
            log.debug("Calling _send_command()...")
            response = self._send_command(cmd)
            log.debug(f"Received response from tracking server: {response}")
            success = response.get("success", False)
            log.debug(f"Response success flag: {success}")
            
            if success:
                self.initialized = True
                self._warned_not_initialized = False  # reset spam guard on fresh init
                log.debug(
                    f"Tracking initialized successfully for {len(object_names)} objects: {object_names}"
                )
            else:
                error_msg = response.get("error", "Unknown error")
                traceback_str = response.get("traceback", "")
                log.error(
                    f"Failed to initialize tracking. Response: {response}, "
                    f"Error: {error_msg}\nTraceback: {traceback_str}"
                )
            
            log.debug(f"Returning initialization result: {success}")
            return success
            
        except Exception as e:
            log.error(f"Failed to initialize tracking: {e}", exc_info=True)
            return False
    
    def track(self, frame: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """
        Track objects in the current frame.
        
        Args:
            frame: Current frame as [H, W, 3] uint8 RGB image
        
        Returns:
            Dictionary mapping object names to bbox centers [x, y] in pixels.
            Returns None for objects not found.
        """
        if not self.initialized:
            if not self._warned_not_initialized:
                log.warning("Tracker not initialized. Call initialize() first.")
                self._warned_not_initialized = True
            return {name: None for name in self.object_names}
        
        try:
            # Send track command
            frame_b64 = self._frame_to_base64(frame)
            cmd = {
                "type": "track",
                "frame": frame_b64,
            }
            
            response = self._send_command(cmd)
            # Response is now the result dict directly, not wrapped in {"result": ...}
            # Format: {"obj_name1": {"center": [x, y], "bbox": [x_min, y_min, x_max, y_max]}, ...}
            result = response if isinstance(response, dict) and "error" not in response else {}
            
            # Convert to numpy arrays, preserving both center and bbox
            result_np = {}
            for obj_name in self.object_names:
                if obj_name in result and result[obj_name] is not None:
                    if isinstance(result[obj_name], dict) and "center" in result[obj_name]:
                        # New format with center and bbox
                        result_np[obj_name] = {
                            "center": np.array(result[obj_name]["center"], dtype=np.float64),
                            "bbox": np.array(result[obj_name]["bbox"], dtype=np.float64)
                        }
                    elif isinstance(result[obj_name], list) and len(result[obj_name]) == 2:
                        # Old format: just center [x, y] (backward compatibility)
                        result_np[obj_name] = {
                            "center": np.array(result[obj_name], dtype=np.float64),
                            "bbox": None
                        }
                    else:
                        result_np[obj_name] = None
                else:
                    result_np[obj_name] = None
            
            return result_np
            
        except Exception as e:
            log.warning(f"Tracking failed for frame: {e}")
            return {name: None for name in self.object_names}
    
    def cleanup(self):
        """Reset per-episode state on the server; keep the subprocess alive."""
        if not hasattr(self, 'process') or self.process is None:
            self.initialized = False
            return
        if self.process.poll() is not None:
            self.process = None
            self.initialized = False
            return
        try:
            self._send_command({"type": "cleanup"})
        except Exception:
            pass
        self.initialized = False

    def shutdown(self):
        """Terminate the subprocess and release its pipe fds. End-of-eval reap."""
        if not hasattr(self, 'process') or self.process is None:
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                self.process.kill()
                self.process.wait(timeout=5)
            except Exception:
                pass
        except Exception:
            pass
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except Exception:
                pass
        self.process = None
        self.initialized = False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
