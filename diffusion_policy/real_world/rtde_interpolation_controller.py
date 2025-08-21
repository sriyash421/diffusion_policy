import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer)
from diffusion_policy.real_world.pd import PD
from diffusion_policy.real_world.robotiq_gripper import RobotiqGripper


class Command(enum.Enum):
    STOP = 0
    SpeedPdJ = 1  # Joint Position Control: SpeedJ with PD control


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

        # build input queue
        example = {
            'cmd': Command.SpeedPdJ.value,
            'target_joints': np.zeros((6,), dtype=np.float64),
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

        self.pd_controller = PD(
            num_joints=6,
            Kp=self.Kp,
            Kd=self.Kd,
            sample_time=None
        )

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
            'cmd': Command.STOP.value
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
    def speedPdJ(self, joints, close_gripper, duration=0.1):
        """
        Send joint position command to the robot.
        
        Args:
            joints: Array of 6 joint positions in radians
            duration: desired time to reach joint positions
        """
        assert self.is_alive()
        assert (duration >= (1/self.frequency))
        joints = np.array(joints)
        assert joints.shape == (6,)

        message = {
            'cmd': Command.SpeedPdJ.value,
            'target_joints': joints,
            'close_gripper': close_gripper,
            'duration': duration
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

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
            current_gripper_close = False
            current_gripper_state = 'open'

            iter_idx = 0
            keep_running = True
            while keep_running:
                # start control iteration
                t_start = rtde_c.initPeriod()

                # Check forward kinematics safety constraint
                if self.tcp_offset_pose is not None:
                    fk_pose = rtde_c.getForwardKinematics(
                        current_target_joints, self.tcp_offset_pose)
                    fk_pose_current = rtde_c.getForwardKinematics(
                        curr_joints, self.tcp_offset_pose)
                else:
                    fk_pose = rtde_c.getForwardKinematics(
                        current_target_joints)
                    fk_pose_current = rtde_c.getForwardKinematics(
                        curr_joints)

                curr_joints = rtde_r.getActualQ()
                if fk_pose[2] < 0.1 and fk_pose_current[2] < 0.1:
                    # Safety stop
                    self.pd_controller.setpoint = curr_joints
                else:
                    # Execute control
                    self.pd_controller.setpoint = current_target_joints

                speeds = self.pd_controller(curr_joints)

                rtde_c.speedJ(
                    qd=speeds,
                    acceleration=self.acceleration,
                    time=dt,
                )

                # print("fk_pose", fk_pose)
                # print("fk_pose_current", fk_pose_current)
                # print("actual tcp pose", rtde_r.getActualTCPPose())
                # print("target tcp pose", rtde_r.getTargetTCPPose())
                # print()

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
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SpeedPdJ.value:
                        # Update target joint positions
                        current_target_joints = command['target_joints']
                        current_gripper_close = command['close_gripper']
                        duration = float(command['duration'])
                        if self.verbose:
                            print("[RTDEPositionalController] New joint "
                                  f"target:{current_target_joints} "
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
            # manditory cleanup
            # decelerate
            rtde_c.servoStop()

            # terminate
            rtde_c.stopScript()
            rtde_c.disconnect()
            rtde_r.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from "
                      f"robot: {robot_ip}")
