import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from scipy.spatial.transform import Rotation as R

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer)
from diffusion_policy.real_world.robotiq_gripper import RobotiqGripper


class Command(enum.Enum):
    STOP = 0
    FKCartesianImpedanceControl = 1  # Joint Position Control: Forward kinematics + impedance control
    DeltaCartesianImpedanceControl = 2  # Delta Cartesian Control: Apply delta to current pose


class RTDEInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """

    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 robot_ip,
                 gripper_port=63352,
                 frequency=125,
                 acceleration=2.0,
                 Kp=1.0,
                 Kd=0.0,
                 launch_timeout=3,
                 tcp_offset_pose=None,
                 payload_mass=None,
                 payload_cog=None,
                 joints_init=None,
                 joints_init_speed=1.05,
                 soft_real_time=False,
                 verbose=False,
                 receive_keys=None,
                 get_max_k=128,
                 # Impedance control parameters
                 imp_kp=10000.0,
                 imp_kd=2200.0,
                 ori_kp=100.0,
                 ori_kd=22.0,
                 imp_delta=0.05,
                 forcemode_damping=0.02,
                 ):
        """
        frequency: CB2=125, UR3e=500
        Kp: proportional gain for following target position
        Kd: derivative gain for following target position
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0 < frequency <= 500
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="RTDEPositionalController")
        self.robot_ip = robot_ip
        self.gripper_port = gripper_port
        self.frequency = frequency
        self.acceleration = acceleration
        self.Kp = Kp
        self.Kd = Kd
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        
        # Impedance control parameters
        self.imp_kp = imp_kp
        self.imp_kd = imp_kd
        self.ori_kp = ori_kp
        self.ori_kd = ori_kd
        self.imp_delta = imp_delta
        self.forcemode_damping = forcemode_damping
        self.dt = 1.0 / frequency
        
        # Force mode configuration
        self.fm_task_frame = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.fm_selection_vector = [1, 1, 1, 1, 1, 1]
        self.fm_limits = [0.5, 0.5, 0.1, 1., 1., 1.]

        # build input queue
        example = {
            'cmd': Command.FKCartesianImpedanceControl.value,
            'target_joints': np.zeros((6,), dtype=np.float64),
            'target_pose': np.zeros((6,), dtype=np.float64),  # [x, y, z, rx, ry, rz]
            'delta_pose': np.zeros((6,), dtype=np.float64),   # [dx, dy, dz, drx, dry, drz]
            'delta_scale': 1.0,  # Scale factor for delta poses
            'close_gripper': np.zeros((1,), dtype=np.bool_),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',

                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd'
            ]
        rtde_r = RTDEReceiveInterface(hostname=robot_ip)
        example = dict()
        for key in receive_keys:
            example[key] = np.array(getattr(rtde_r, 'get'+key)())
        example['robot_receive_timestamp'] = time.time()
        # Add new fields for end-effector pose and actions
        example['ee_pose'] = np.zeros(6, dtype=np.float64)  # [x,y,z,rx,ry,rz]
        example['ee_action'] = np.zeros(6, dtype=np.float64)  # [dx,dy,dz,drx,dry,drz]
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process "
                  f"spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': np.array([Command.STOP.value], dtype=np.int32),
            'target_joints': np.zeros((6,), dtype=np.float64),
            'target_pose': np.zeros((6,), dtype=np.float64),
            'delta_pose': np.zeros((6,), dtype=np.float64),
            'delta_scale': 1.0,
            'close_gripper': np.zeros((1,), dtype=np.bool_),
            'duration': 0.0,
            'target_time': 0.0
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def fk_cartesian_control(self, joints, close_gripper, duration=0.1):
        """
        Send joint position command to the robot using forward kinematics + impedance control.
        
        Args:
            joints: Array of 6 joint positions in radians
            duration: desired time to reach joint positions
        """
        assert self.is_alive()
        assert (duration >= (1/self.frequency))
        joints = np.array(joints)
        assert joints.shape == (6,)

        message = {
            'cmd': np.array([Command.FKCartesianImpedanceControl.value], dtype=np.int32),
            'target_joints': joints,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'delta_pose': np.zeros((6,), dtype=np.float64),
            'delta_scale': 1.0,
            'close_gripper': np.array([close_gripper], dtype=np.bool_),
            'duration': float(duration),
            'target_time': 0.0
        }
        self.input_queue.put(message)

    def delta_cartesian_control(self, delta_pose, close_gripper, delta_scale=1.0, duration=0.1):
        """
        Send cartesian delta command to the robot using impedance control.
        
        Args:
            delta_pose: Array of 6 delta values [dx, dy, dz, drx, dry, drz] in axis-angle format
            close_gripper: Boolean for gripper state
            delta_scale: Scale factor for delta poses (0.0 = no change, 1.0 = full delta)
            duration: desired time to reach target pose
        """
        assert self.is_alive()
        assert (duration >= (1/self.frequency))
        delta_pose = np.array(delta_pose)
        assert delta_pose.shape == (6,)

        message = {
            'cmd': np.array([Command.DeltaCartesianImpedanceControl.value], dtype=np.int32),
            'target_joints': np.zeros((6,), dtype=np.float64),
            'target_pose': np.zeros((6,), dtype=np.float64),
            'delta_pose': delta_pose,
            'delta_scale': float(delta_scale),
            'close_gripper': np.array([close_gripper], dtype=np.bool_),
            'duration': float(duration),
            'target_time': 0.0
        }
        self.input_queue.put(message)

    def reset_to_initial_position(self, duration=3.0):
        """
        Reset robot to initial joint position.
        
        Args:
            duration: Time to reach initial position (seconds)
        """
        if self.joints_init is not None:
            print(f"Resetting robot to initial position: {self.joints_init}")
            self.fk_cartesian_control(
                joints=self.joints_init,
                close_gripper=False,  # Open gripper during reset
                duration=duration
            )
            # Wait for movement to complete
            time.sleep(duration + 0.5)
        else:
            print("Warning: No initial joint positions defined for reset")

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= helper methods for impedance control ============
    def _apply_delta_pose(self, source_pose: np.ndarray, delta_pose: np.ndarray, scale: float = 1.0, eps: float = 1.0e-6) -> np.ndarray:
        """
        Apply delta pose transformation on source pose with interpolation scaling.
        
        Args:
            source_pose: Current TCP pose [x, y, z, rx, ry, rz] in axis-angle format
            delta_pose: Position and orientation displacements [dx, dy, dz, drx, dry, drz]
            scale: Interpolation factor (0.0 = no change, 1.0 = full delta)
            eps: Tolerance to consider orientation displacement as zero
            
        Returns:
            Target pose [x, y, z, rx, ry, rz] in axis-angle format
        """
        # Scale the delta pose
        scaled_delta_pose = delta_pose * scale
        
        # Position delta: simply add scaled delta
        target_pos = source_pose[:3] + scaled_delta_pose[:3]
        
        # Rotation delta: compose rotations with scaled delta
        rot_actions = scaled_delta_pose[3:6]
        angle = np.linalg.norm(rot_actions)
        
        # if angle > eps:
        # Convert delta rotation to rotation matrix
        delta_rot = R.from_rotvec(rot_actions)
        
        # Convert current rotation to rotation matrix
        current_rot = R.from_rotvec(source_pose[3:6])
        
        # Compose rotations: target = delta * current
        target_rot = delta_rot * current_rot
        target_rotvec = target_rot.as_rotvec()
        # else:
        #     # No rotation change
        #     target_rotvec = source_pose[3:6]
        
        return np.concatenate([target_pos, target_rotvec])

    def _pose_to_quat(self, pose_xyz_rxryrz: np.ndarray) -> np.ndarray:
        """Convert pose [x,y,z,rx,ry,rz] to [x,y,z,qx,qy,qz,qw]."""
        pos = np.array(pose_xyz_rxryrz[:3], dtype=float)
        rotvec = np.array(pose_xyz_rxryrz[3:], dtype=float)
        quat = R.from_rotvec(rotvec).as_quat()  # Returns [x,y,z,w]
        return np.concatenate([pos, quat])

    def _compute_impedance_wrench(self, target_pose_quat: np.ndarray, 
                                 curr_pose: np.ndarray, curr_vel: np.ndarray) -> np.ndarray:
        """Compute impedance control wrench from target and current pose."""
        curr_quat_pose = self._pose_to_quat(curr_pose)
        
        # Position impedance
        diff_p = target_pose_quat[:3] - curr_quat_pose[:3]
        diff_p = np.clip(diff_p, a_min=-self.imp_delta, a_max=self.imp_delta)
        vel_delta = 2.0 * self.imp_delta / self.dt
        diff_d = np.clip(-curr_vel[:3], a_min=-vel_delta, a_max=vel_delta)
        force = self.imp_kp * diff_p + self.imp_kd * diff_d

        # Orientation impedance using scipy Rotation (robust to discontinuities)
        target_quat = target_pose_quat[3:7]  # Extract quaternion (4 values)
        curr_quat = curr_quat_pose[3:7]      # Extract quaternion (4 values)
        
        # Use scipy Rotation for robust orientation error calculation
        rot_diff = R.from_quat(target_quat) * R.from_quat(curr_quat).inv()
        rotvec_err = rot_diff.as_rotvec()
        ang_vel = curr_vel[3:]
        torque = self.ori_kp * rotvec_err + self.ori_kd * (-ang_vel)

        return np.concatenate([force, torque]).astype(float)

    def _compute_ee_actions(self, target_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
        """Compute relative end-effector actions (delta pose) from current to target pose."""
        # Position delta: target_pos - current_pos
        delta_pos = target_pose[:3] - current_pose[:3]
        
        # Rotation delta: compute rotation that transforms current to target
        current_rot = R.from_rotvec(current_pose[3:])
        target_rot = R.from_rotvec(target_pose[3:])
        
        # delta_rot such that delta_rot * current_rot = target_rot
        # So delta_rot = target_rot * current_rot^(-1)
        delta_rot = target_rot * current_rot.inv()
        delta_rotvec = delta_rot.as_rotvec()
        
        return np.concatenate([delta_pos, delta_rotvec]).astype(float)

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start gripper
        gripper = RobotiqGripper()
        gripper.connect(self.robot_ip, self.gripper_port)
        # start rtde
        robot_ip = self.robot_ip
        rtde_c = RTDEControlInterface(hostname=robot_ip)
        rtde_r = RTDEReceiveInterface(hostname=robot_ip)

        try:
            if self.verbose:
                print(f"[RTDEPositionalController] Connect to robot: "
                      f"{robot_ip}")

            # set parameters
            if self.tcp_offset_pose is not None:
                rtde_c.setTcp(self.tcp_offset_pose)
            if self.payload_mass is not None:
                if self.payload_cog is not None:
                    assert rtde_c.setPayload(self.payload_mass,
                                             self.payload_cog)
                else:
                    assert rtde_c.setPayload(self.payload_mass)
            tcp_offset = rtde_c.getTCPOffset()
            
            # Initialize force mode parameters
            rtde_c.forceModeSetDamping(self.forcemode_damping)
            rtde_c.zeroFtSensor()

            # init pose
            if self.joints_init is not None:
                assert rtde_c.moveJ(self.joints_init,
                                    self.joints_init_speed, 1.4)

            gripper.activate()

            # main loop
            dt = 1. / self.frequency
            curr_joints = rtde_r.getActualQ()

            # Track current control mode
            current_target_joints = curr_joints
            current_delta_pose = None  # Will be set for delta cartesian mode
            current_delta_scale = 1.0  # Scale for delta poses
            current_control_mode = Command.DeltaCartesianImpedanceControl  # Default to joint control
            current_gripper_close = False
            current_gripper_state = 'open'

            iter_idx = 0
            keep_running = True
            while keep_running:
                # start control iteration
                t_start = rtde_c.initPeriod()

                curr_joints = rtde_r.getActualQ()
                
                # Compute target pose based on control mode
                if current_control_mode == Command.FKCartesianImpedanceControl:
                    # Joint control mode: compute target pose from target joints using forward kinematics
                    target_pose_xyz_rxryrz = rtde_c.getForwardKinematics(current_target_joints, tcp_offset)
                elif current_control_mode == Command.DeltaCartesianImpedanceControl and current_delta_pose is not None:
                    # Delta cartesian control mode: apply delta to current pose
                    current_pose = rtde_r.getActualTCPPose()
                    
                    # HACK: Transform delta from policy frame to robot base frame
                    # Policy operates in (1,0,0,0) frame, robot base is (0,0,0,1) - 180 deg rotation
                    transformed_delta_pose = current_delta_pose.copy()
                    # Comment out the next 4 lines to disable coordinate transformation
                    # policy_to_robot_rot = R.from_quat([0, 0, 1, 0])  # 180 deg around Z
                    # delta_pos = transformed_delta_pose[:3]
                    # delta_rot = transformed_delta_pose[3:6]
                    # transformed_delta_pose[:3] = policy_to_robot_rot.apply(delta_pos)
                    # transformed_delta_pose[3:6] = policy_to_robot_rot.apply(delta_rot)
                    target_pose_xyz_rxryrz = self._apply_delta_pose(current_pose, transformed_delta_pose, current_delta_scale)
                else:
                    # Fallback to current pose
                    target_pose_xyz_rxryrz = rtde_r.getActualTCPPose()
                
                # Convert to quaternion format
                target_pose_quat = self._pose_to_quat(target_pose_xyz_rxryrz)
                
                # Get current pose and velocity
                curr_pose = rtde_r.getActualTCPPose()
                curr_vel = np.array(rtde_r.getActualTCPSpeed(), dtype=float)
                
                # Compute impedance control wrench
                wrench = self._compute_impedance_wrench(target_pose_quat, curr_pose, curr_vel)
                
                # Apply force mode control
                ok = rtde_c.forceMode(
                    self.fm_task_frame,
                    self.fm_selection_vector,
                    wrench.tolist(),
                    2,  # type 2 for normal force mode
                    self.fm_limits,
                )
                if not ok:
                    rtde_c.forceModeStop()
                    if self.verbose:
                        print("[RTDEPositionalController] forceMode failed, stopping")

                # update gripper state
                if (current_gripper_close and
                        current_gripper_state == 'open'):
                    gripper.move(gripper.get_closed_position(), 255, 255)
                    current_gripper_state = 'closed'
                elif (not current_gripper_close and
                      current_gripper_state == 'closed'):
                    gripper.move(gripper.get_open_position(), 255, 255)
                    current_gripper_state = 'open'
                else:
                    pass

                # update robot state
                state = dict()
                for key in self.receive_keys:
                    state[key] = np.array(getattr(rtde_r, 'get'+key)())
                state['robot_receive_timestamp'] = time.time()
                
                # Add end-effector pose and actions
                state['ee_pose'] = np.array(curr_pose, dtype=np.float64)
                state['ee_action'] = self._compute_ee_actions(np.array(target_pose_xyz_rxryrz, dtype=np.float64), np.array(curr_pose, dtype=np.float64))
                
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                    commands = None

                # execute commands
                if commands is not None:
                    for i in range(n_cmd):
                        command = dict()
                        for key, value in commands.items():
                            command[key] = value[i]
                        cmd = command['cmd'][0] if isinstance(command['cmd'], np.ndarray) else command['cmd']

                        if cmd == Command.STOP.value:
                            keep_running = False
                            # stop immediately, ignore later commands
                            break
                        elif cmd == Command.FKCartesianImpedanceControl.value:
                            # Update target joint positions
                            current_target_joints = command['target_joints']
                            current_control_mode = Command.FKCartesianImpedanceControl
                            current_gripper_close = command['close_gripper'][0] if isinstance(command['close_gripper'], np.ndarray) else command['close_gripper']
                            duration = float(command['duration'])
                            if self.verbose:
                                print("[RTDEPositionalController] New joint "
                                      f"target:{current_target_joints} "
                                      f"duration:{duration}s")
                        elif cmd == Command.DeltaCartesianImpedanceControl.value:
                            # Update delta cartesian pose
                            current_delta_pose = command['delta_pose']
                            current_delta_scale = float(command['delta_scale'])
                            current_control_mode = Command.DeltaCartesianImpedanceControl
                            current_gripper_close = command['close_gripper'][0] if isinstance(command['close_gripper'], np.ndarray) else command['close_gripper']
                            duration = float(command['duration'])
                            if self.verbose:
                                print("[RTDEPositionalController] New delta cartesian "
                                      f"delta:{current_delta_pose} scale:{current_delta_scale} "
                                      f"duration:{duration}s")
                        else:
                            keep_running = False
                            break

                # regulate frequency
                rtde_c.waitPeriod(t_start)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    freq = 1/(time.perf_counter() - t_start)
                    print(f"[RTDEPositionalController] Actual frequency "
                          f"{freq}")

        finally:
            # mandatory cleanup
            try:
                # Stop force mode first
                rtde_c.forceModeStop()
                
                # Hold current position briefly to prevent drift
                current_joints = rtde_r.getActualQ()
                rtde_c.servoJ(current_joints, 0.5, 0.5, 0.1, 0.1, 300)
                
                # decelerate
                rtde_c.servoStop()
            except Exception as e:
                if self.verbose:
                    print(f"[RTDEPositionalController] Cleanup error: {e}")

            # terminate
            rtde_c.stopScript()
            rtde_c.disconnect()
            rtde_r.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from "
                      f"robot: {robot_ip}")
