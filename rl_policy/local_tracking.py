import numpy as np
import argparse
import yaml
import sys
sys.path.append(".")

from rl_policy.g1_policy import G1Policy
np.set_printoptions(precision=3, suppress=True, linewidth=1000)


class Policy(G1Policy):
    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == "]":
            self.reset()
    
    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)
        if cur_key == "R1":
            self.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--policy_config", type=str, default="config/policy/deepmimic_29dof.yaml", help="policy config file"
    )
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument(
        "--motion_path", type=str, default="data/motion/hdmi_walk.npy", help="motion path"
    )
    args = parser.parse_args()

    with open(args.policy_config) as file:
        policy_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    model_path = args.policy_config.replace(".yaml", ".onnx")

    motion_obs_names = [
        "ref_joint_pos_future",
        "ref_joint_vel_future",
        "ref_body_pos_future_local",
        "ref_body_lin_vel_future_local",
        "ref_body_ori_future_local",
        "ref_body_ang_vel_future_local",
    ]

    for motion_obs_name in motion_obs_names:
        if motion_obs_name not in policy_config["observation"]["command"]:
            continue
        motion_obs_config = policy_config["observation"]["command"][motion_obs_name]
        motion_obs_config["motion_path"] = args.motion_path
        motion_obs_config["future_steps"] = policy_config["future_steps"]
        motion_obs_config["joint_names"] = policy_config["tracking_joint_names"]
        motion_obs_config["body_names"] = policy_config["tracking_keypoint_names"]
        motion_obs_config["root_body_name"] = "pelvis"

    policy = Policy(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=model_path,
        rl_rate=50,
    )
    policy.run()
