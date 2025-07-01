import sys
import time
import threading
import numpy as np
import zmq
from typing import List, Dict, Any, Type
import sched
from termcolor import colored
from sshkeyboard import listen_keyboard


sys.path.append(".")
# Ensure g1_interface can be imported
sys.path.append("/home/unitree/haoyang_deploy/unitree_sdk2/build/lib")
try:
    import g1_interface  # type: ignore
except ImportError as e:
    raise ImportError("g1_interface module not found. Make sure it is built and added to PYTHONPATH.")

from loguru import logger

# Re-use utilities from the existing sim2real package
from utils.strings import resolve_matching_names_values, unitree_joint_names
from utils.onnx_module import Timer
from rl_policy.observations import Observation, ObsGroup

class G1StateProcessor:
    """Light-weight state processor reading data directly from ``g1_interface``.

    This class provides the minimal subset of attributes/methods expected by the
    existing observation modules so that they remain drop-in compatible.
    """

    def __init__(self, robot: "g1_interface.G1Interface", dest_joint_names: List[str]):
        self.robot = robot

        # Mapping from Isaac order (policy / sim order) to the internal Unitree order
        self.num_dof = len(dest_joint_names)
        self.joint_indices_in_source = [unitree_joint_names.index(name) for name in dest_joint_names]

        # Isaac style qpos / qvel buffers (same layout as original ``StateProcessor``)
        self.qpos = np.zeros(3 + 4 + self.num_dof)
        self.qvel = np.zeros(3 + 3 + self.num_dof)

        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]

        self.root_quat_b = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]

        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]

        self.zmq_context = zmq.Context()
        self.mocap_subscribers: Dict[str, zmq.Socket] = {}
        self.mocap_threads: Dict[str, threading.Thread] = {}
        self.mocap_data: Dict[str, Any] = {}
        self.mocap_data_lock = threading.Lock()

    def register_subscriber(self, object_name: str, port: int):
        if object_name in self.mocap_subscribers:
            return
        from utils.common import MOCAP_IP
        socket = self.zmq_context.socket(zmq.SUB)
        socket.connect(f"tcp://{MOCAP_IP}:{port}")
        socket.setsockopt(zmq.SUBSCRIBE, object_name.encode())
        socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms
        self.mocap_subscribers[object_name] = socket

        def _sub_thread(obj_name: str):
            while True:
                try:
                    msg = socket.recv_multipart(zmq.NOBLOCK)
                    if len(msg) == 2:
                        name = msg[0].decode()
                        data = np.frombuffer(msg[1], dtype=np.float64)
                        if len(data) == 7:
                            pos, quat = data[:3].copy(), data[3:].copy()
                            with self.mocap_data_lock:
                                self.mocap_data[f"{name}_pos"] = pos
                                self.mocap_data[f"{name}_quat"] = quat
                except zmq.Again:
                    time.sleep(0.001)
                except Exception as e:
                    logger.warning(f"{obj_name} subscriber error: {e}")
                    time.sleep(0.01)
        th = threading.Thread(target=_sub_thread, args=(object_name,), daemon=True)
        th.start()
        self.mocap_threads[object_name] = th

    def get_mocap_data(self, key: str):
        with self.mocap_data_lock:
            return self.mocap_data.get(key, None)

    # ------------------------------------------------------------------
    # Low-state preparation (called every control / RL step)
    # ------------------------------------------------------------------
    def _prepare_low_state(self) -> bool:
        try:
            state = self.robot.read_low_state()
        except Exception as e:
            logger.warning(f"Failed to read G1 low state: {e}")
            return False

        if state is None:
            return False

        # IMU
        self.root_quat_b[:] = state.imu.quat  # [w, x, y, z]
        self.root_ang_vel_b[:] = state.imu.omega

        # Joints
        for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
            self.joint_pos[dst_idx] = state.motor.q[src_idx]
            self.joint_vel[dst_idx] = state.motor.dq[src_idx]
        return True


# -------------------------------------------------------------------------------------------------
# Command sender wrapping MotorCommand for G1
# -------------------------------------------------------------------------------------------------
class G1CommandSender:
    """Thin wrapper translating Isaace style command arrays to ``MotorCommand``."""

    def __init__(
        self,
        robot: "g1_interface.G1Interface",
        policy_config: Dict[str, Any],
    ):
        """Create a command sender using policy-specified gains.

        Args:
            robot:          Active ``g1_interface`` instance.
            dest_joint_names:  Joint order used by the policy (Isaac order).
            policy_config:  Dict containing ``joint_kp``, ``joint_kd``, and ``default_joint_pos``.
        """

        self.robot = robot
        self.policy_config = policy_config
        # init robot and kp kd
        self._kp_level = 1.0  # 0.1

        joint_kp_dict = self.policy_config["joint_kp"]
        joint_indices, joint_names, joint_kp = resolve_matching_names_values(
            joint_kp_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.joint_kp_unitree_default = np.zeros(len(unitree_joint_names))
        self.joint_kp_unitree_default[joint_indices] = joint_kp
        self.joint_kp_unitree = self.joint_kp_unitree_default.copy()

        joint_kd_dict = self.policy_config["joint_kd"]
        joint_indices, joint_names, joint_kd = resolve_matching_names_values(
            joint_kd_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.joint_kd_unitree = np.zeros(len(unitree_joint_names))
        self.joint_kd_unitree[joint_indices] = joint_kd

        default_joint_pos_dict = self.policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_joint_pos_unitree = np.zeros(len(unitree_joint_names))
        self.default_joint_pos_unitree[joint_indices] = default_joint_pos

        joint_names_isaac = self.policy_config["isaac_joint_names"]
        self.joint_indices_unitree = [unitree_joint_names.index(name) for name in joint_names_isaac]

    # Expose kp_level so it can be tuned by UI callbacks (same API as original)
    @property
    def kp_level(self) -> float:
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value: float):
        self._kp_level = float(value)

    # --------------------------------------------------------------
    # Core API expected by BasePolicy
    # --------------------------------------------------------------
    def send_command(self, cmd_q: np.ndarray, cmd_dq: np.ndarray, cmd_tau: np.ndarray):
        """Construct a ``MotorCommand`` and dispatch it via the interface."""
        cmd = self.robot.create_zero_command()

        # Apply kp_level scaling (kd remains constant, consistent with original implementation)
        kp_scaled = self.joint_kp_unitree * self._kp_level
        kd_scaled = self.joint_kd_unitree

        q_target = list(cmd.q_target)
        dq_target = list(cmd.dq_target)
        tau_ff = list(cmd.tau_ff)
        kp = list(cmd.kp)
        kd = list(cmd.kd)
        for i_policy, idx_unitree in enumerate(self.joint_indices_unitree):
            q_target[idx_unitree] = float(cmd_q[i_policy])
            dq_target[idx_unitree] = float(cmd_dq[i_policy])
            tau_ff[idx_unitree] = float(cmd_tau[i_policy])
            kp[idx_unitree] = float(kp_scaled[idx_unitree])
            kd[idx_unitree] = float(kd_scaled[idx_unitree])

        cmd.q_target = q_target
        cmd.dq_target = dq_target
        cmd.tau_ff = tau_ff
        cmd.kp = kp
        cmd.kd = kd

        self.robot.write_low_command(cmd)


# -------------------------------------------------------------------------------------------------
# High-level RL policy that plugs into the existing framework
# -------------------------------------------------------------------------------------------------
class G1Policy:
    def __init__(
        self,
        robot_config: Dict[str, Any],
        policy_config: Dict[str, Any],
        model_path: str,
        rl_rate: int = 50,
    ) -> None:
        network_interface = robot_config.get("INTERFACE", None)
        # Create shared G1 interface instance
        self.robot = g1_interface.G1Interface(network_interface)
        # Ensure we are in PR mode (most policies work in PR)
        try:
            self.robot.set_control_mode(g1_interface.ControlMode.PR)
        except Exception:
            pass  # Ignore if firmware already in the correct mode

        # Plug-in our custom state processor & command sender
        self.state_processor = G1StateProcessor(self.robot, policy_config["isaac_joint_names"])
        self.command_sender = G1CommandSender(self.robot, policy_config)

        self.rl_dt = 1.0 / rl_rate

        self.policy_config = policy_config

        self.setup_policy(model_path)
        self.obs_cfg = policy_config["observation"]

        self.isaac_joint_names = policy_config["isaac_joint_names"]
        self.num_dofs = len(self.isaac_joint_names)

        default_joint_pos_dict = policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            self.isaac_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_dof_angles = np.zeros(len(self.isaac_joint_names))
        self.default_dof_angles[joint_indices] = default_joint_pos

        action_scale_cfg = policy_config["action_scale"]
        self.action_scale = np.ones((self.num_dofs))
        if isinstance(action_scale_cfg, float):
            self.action_scale *= action_scale_cfg
        elif isinstance(action_scale_cfg, dict):
            joint_ids, joint_names, action_scales = resolve_matching_names_values(
                action_scale_cfg, self.isaac_joint_names, preserve_order=True
            )
            self.action_scale[joint_ids] = action_scales
        else:
            raise ValueError(f"Invalid action scale type: {type(action_scale_cfg)}")

        self.policy_joint_names = policy_config["policy_joint_names"]
        self.num_actions = len(self.policy_joint_names)
        self.controlled_joint_indices = [
            self.isaac_joint_names.index(name)
            for name in self.policy_joint_names
        ]

        # Keypress control state
        self.use_policy_action = False

        self.first_time_init = True
        self.init_count = 0
        self.get_ready_state = False

        # Joint limits
        joint_indices, joint_names, joint_pos_lower_limit = (
            resolve_matching_names_values(
                robot_config["joint_pos_lower_limit"],
                self.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_lower_limit = np.zeros(self.num_dofs)
        self.joint_pos_lower_limit[joint_indices] = joint_pos_lower_limit

        joint_indices, joint_names, joint_pos_upper_limit = (
            resolve_matching_names_values(
                robot_config["joint_pos_upper_limit"],
                self.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_upper_limit = np.zeros(self.num_dofs)
        self.joint_pos_upper_limit[joint_indices] = joint_pos_upper_limit

        # ------------------------------------------------------
        # Joystick / keyboard setup (mirrors base_policy logic)
        # ------------------------------------------------------
        if robot_config.get("USE_JOYSTICK", False):
            print("Using joystick")
            self.use_joystick = True
            self.wc_msg = None  # type: ignore
            self.last_wc_msg = self.robot.read_wireless_controller()
            print("Wireless Controller Initialized")
        else:
            print("Using keyboard")
            self.use_joystick = False
            self.key_listener_thread = threading.Thread(
                target=self.start_key_listener, daemon=True
            )
            self.key_listener_thread.start()

        # Setup observations after all processors are ready
        self.setup_observations()


    def setup_policy(self, model_path):
        # load onnx policy
        from utils.onnx_module import ONNXModule
        onnx_module = ONNXModule(model_path)

        def policy(input_dict):
            output_dict = onnx_module(input_dict)
            action = output_dict["action"].squeeze(0)
            carry = {k[1]: v for k, v in output_dict.items() if k[0] == "next"}
            return action, carry

        self.policy = policy

    def setup_observations(self):
        """Setup observations for policy inference"""
        self.observations: Dict[str, ObsGroup] = {}
        self.reset_callbacks = []
        self.update_callbacks = []
        
        # Create observation instances based on config
        for obs_group, obs_items in self.obs_cfg.items():
            print(f"obs_group: {obs_group}")
            obs_funcs = {}
            for obs_name, obs_config in obs_items.items():
                obs_class: Type[Observation] = Observation.registry[obs_name]
                obs_func = obs_class(env=self, **obs_config)
                obs_funcs[obs_name] = obs_func
                self.reset_callbacks.append(obs_func.reset)
                self.update_callbacks.append(obs_func.update)
                print(f"\t{obs_name}: {obs_config}")
            self.observations[obs_group] = ObsGroup(obs_group, obs_funcs)

    def reset(self):
        for reset_callback in self.reset_callbacks:
            reset_callback()

    def update(self):
        for update_callback in self.update_callbacks:
            update_callback(self.state_dict)

    def prepare_obs_for_rl(self):
        """Prepare observation for policy inference using observation classes"""
        obs_dict: Dict[str, np.ndarray] = {}
        for obs_group in self.observations.values():
            obs = obs_group.compute()
            obs_dict[obs_group.name] = obs[None, :].astype(np.float32)
        return obs_dict

    def get_init_target(self):
        if self.init_count > 500:
            self.init_count = 500

        # interpolate from current dof_pos to default angles
        dof_pos = self.state_processor.joint_pos
        progress = self.init_count / 500
        q_target = dof_pos + (self.default_dof_angles - dof_pos) * progress
        self.init_count += 1
        return q_target

    def run(self):
        total_inference_cnt = 0
        
        # 初始化状态变量
        state_dict = {}
        state_dict["adapt_hx"] = np.zeros((1, 128), dtype=np.float32)
        state_dict["action"] = np.zeros(self.num_actions)
        self.state_dict = state_dict
        self.total_inference_cnt = total_inference_cnt
        self.perf_dict = {}

        try:
            # 使用scheduler进行精确时间控制
            scheduler = sched.scheduler(time.perf_counter, time.sleep)
            next_run_time = time.perf_counter()
            
            while True:
                # 调度下一次执行
                scheduler.enterabs(next_run_time, 1, self._rl_step_scheduled, ())
                scheduler.run()
                
                next_run_time += self.rl_dt
                self.total_inference_cnt += 1

                if self.total_inference_cnt % 100 == 0:
                    # print(f"total_inference_cnt: {self.total_inference_cnt}")
                    # for key, value in self.perf_dict.items():
                    #     print(f"\t{key}: {value/100*1000:.3f} ms")
                    self.perf_dict = {}
        except KeyboardInterrupt:
            pass

    def _rl_step_scheduled(self):
        loop_start = time.perf_counter()

        with Timer(self.perf_dict, "prepare_low_state"):
            if self.use_joystick:
                self.process_joystick_input()

            if not self.state_processor._prepare_low_state():
                print("low state not ready.")
                return
            
        try:
            with Timer(self.perf_dict, "prepare_obs"):
                # Prepare observations
                self.update()
                obs_dict = self.prepare_obs_for_rl()
                self.state_dict.update(obs_dict)
                self.state_dict["is_init"] = np.zeros(1, dtype=bool)

            with Timer(self.perf_dict, "policy"):   
                # Inference
                action, self.state_dict = self.policy(self.state_dict)
                # Clip policy action
                action = action.clip(-100, 100)
                self.state_dict["action"] = action
        except Exception as e:
            print(f"Error in policy inference: {e}")
            self.state_dict["action"] = np.zeros(self.num_actions)
            return

        with Timer(self.perf_dict, "rule_based_control_flow"):
            # rule based control flow
            if self.get_ready_state:
                q_target = self.get_init_target()
            elif not self.use_policy_action:
                q_target = self.state_processor.joint_pos
            else:
                policy_action = np.zeros((self.num_dofs))
                policy_action[self.controlled_joint_indices] = self.state_dict["action"]
                policy_action = policy_action * self.action_scale
                q_target = policy_action + self.default_dof_angles

            # Clip q target
            q_target = np.clip(
                q_target, self.joint_pos_lower_limit, self.joint_pos_upper_limit
            )

            # Send command
            cmd_q = q_target
            cmd_dq = np.zeros(self.num_dofs)
            cmd_tau = np.zeros(self.num_dofs)
            self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)

        elapsed = time.perf_counter() - loop_start
        if elapsed > self.rl_dt:
            logger.warning(f"RL step took {elapsed:.6f} seconds, expected {self.rl_dt} seconds")

    def process_joystick_input(self):
        """Poll current wireless controller state and translate to high-level key events."""
        try:
            self.wc_msg = self.robot.read_wireless_controller()
        except Exception:
            return

        if self.wc_msg is None:
            return

        # print(f"wc_msg.A: {self.wc_msg.A}")
        if self.wc_msg.A and not self.last_wc_msg.A:
            self.handle_joystick_button("A")
        if self.wc_msg.B and not self.last_wc_msg.B:
            self.handle_joystick_button("B")
        if self.wc_msg.X and not self.last_wc_msg.X:
            self.handle_joystick_button("X")
        if self.wc_msg.Y and not self.last_wc_msg.Y:
            self.handle_joystick_button("Y")
        if self.wc_msg.L1 and not self.last_wc_msg.L1:
            self.handle_joystick_button("L1")
        if self.wc_msg.L2 and not self.last_wc_msg.L2:
            self.handle_joystick_button("L2")
        if self.wc_msg.R1 and not self.last_wc_msg.R1:
            self.handle_joystick_button("R1")
        if self.wc_msg.R2 and not self.last_wc_msg.R2:
            self.handle_joystick_button("R2")
        
        self.last_wc_msg = self.wc_msg

    def handle_joystick_button(self, cur_key):
        if cur_key == "R1":
            self.use_policy_action = True
            self.get_ready_state = False
            logger.info(colored("Using policy actions", "blue"))
            self.phase = 0.0  # type: ignore
        elif cur_key == "R2":
            self.use_policy_action = False
            self.get_ready_state = False
            logger.info(colored("Actions set to zero", "blue"))
        elif cur_key == "A":
            self.get_ready_state = True
            self.init_count = 0
            logger.info(colored("Setting to init state", "blue"))
        # elif cur_key == "Y+left":
        #     self.command_sender.kp_level -= 0.1
        # elif cur_key == "Y+right":
        #     self.command_sender.kp_level += 0.1
        # elif cur_key == "A+left":
        #     self.command_sender.kp_level -= 0.01
        # elif cur_key == "A+right":
        #     self.command_sender.kp_level += 0.01

        # Debug print for kp level tuning
        if cur_key in ["Y+left", "Y+right", "A+left", "A+right"]:
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))

    # ----------------------------- Keyboard handling -----------------------------
    def start_key_listener(self):
        """Start a key listener using sshkeyboard (same as BasePolicy)."""

        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except AttributeError as e:
                logger.warning(f"Keyboard key {keycode}. Error: {e}")

        listener = listen_keyboard(on_press=on_press)
        listener.start()
        listener.join()

    def handle_keyboard_button(self, keycode):
        """Mirrors BasePolicy keyboard shortcuts."""
        if keycode == "]":
            self.use_policy_action = True
            self.get_ready_state = False
            logger.info("Using policy actions")
            self.phase = 0.0  # type: ignore
        elif keycode == "o":
            self.use_policy_action = False
            self.get_ready_state = False
            logger.info("Actions set to zero")
        elif keycode == "i":
            self.use_policy_action = False
            self.get_ready_state = True
            self.init_count = 0
            logger.info("Setting to init state")
        elif keycode == "5":
            self.command_sender.kp_level -= 0.01
        elif keycode == "6":
            self.command_sender.kp_level += 0.01
        elif keycode == "4":
            self.command_sender.kp_level -= 0.1
        elif keycode == "7":
            self.command_sender.kp_level += 0.1
        elif keycode == "0":
            self.command_sender.kp_level = 1.0

        if keycode in ["5", "6", "4", "7", "0"]:
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
