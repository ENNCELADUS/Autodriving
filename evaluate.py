from metadrive.envs import MetaDriveEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from functools import partial
from IPython.display import clear_output
import os
import random
import matplotlib.pyplot as plt
import base64
from PIL import Image
import io
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import json
from datetime import datetime  # Added to get current date and time
import imageio

class CustomRewardEnv(MetaDriveEnv):
    def __init__(self, config):
        super().__init__(config)
        self.termination_status = None  # Initialize termination status
    
    def reset(self, **kwargs):
        self.termination_status = None  # Reset termination status
        return super().reset(**kwargs)


    def reward_function(self, vehicle_id: str):
        """
        自定义奖励函数，增加逆行车道上的惩罚
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()


        # 初始化奖励
        reward = 0.0

        # 初始化 current_road，避免未定义错误
        current_road = None

        # 判断车辆是否在参考车道上
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1  # 正向道路
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road  # 确保此处一定会被赋值
            positive_road = 1 if not current_road.is_negative_road() else -1

        # 检查是否在逆行车道上，并给予惩罚
        if current_road and current_road.is_negative_road() and positive_road == 1:
            reward -= self.config.get("wrong_way_penalty", 50.0)  # 逆行惩罚

        # 计算车道中的位置变化
        long_last, lat_last = current_lane.local_coordinates(vehicle.last_position)
        long_now, lat_now = current_lane.local_coordinates(vehicle.position)

        # print("当前位置{}".format(abs(lat_now)))
        # print("中心线位置{}".format(current_lane.width/2))

        # # 横向奖励，根据车辆离车道中心的距离给予惩罚（绝对值越大惩罚越高）
        # lateral_factor =  (1.0 - (abs(lat_now) - current_lane.width/2) / current_lane.width) * 0.02
        # reward += self.config.get("lateral_reward_weight", 0.5) * lateral_factor
        # # print("居中奖励={}".format(self.config.get("lateral_reward_weight", 0.5) * lateral_factor))

        # 奖励前进距离
        reward += self.config.get("driving_reward", 1.0) * (long_now - long_last)  * positive_road
        # print("前进奖励={}".format(self.config.get("driving_reward", 1.0) * (long_now - long_last)  * positive_road))

        # 奖励速度，鼓励合理速度
        speed_factor = vehicle.speed_km_h / vehicle.max_speed_km_h
        reward += self.config.get("speed_reward", 0.1) * speed_factor * positive_road
        # print("速度奖励={}".format(self.config.get("speed_reward", 0.1) * speed_factor * positive_road))

        # 加速度系数 0.01/0.02/0
        global last_velocity
        acceleration = 0.00 * (vehicle.speed_km_h - last_velocity)
        reward -= abs(acceleration)
        # print("加速度奖励={}".format(acceleration))
        global total_acc
        total_acc += abs(vehicle.speed_km_h - last_velocity)
        # 终止reward
        done, _ = self.done_function(vehicle_id)

        if done :
            last_velocity = 0
            if self._is_arrive_destination(vehicle):
                reward += self.config.get("success_reward", 40.0)
                print("走到终点了")

            elif vehicle.crash_vehicle:
                global crash_case
                crash_case+= 1
                # reward -= self.config.get("crash_vehicle_penalty", 20.0)
                print("撞车了")


                reward -= 20



            else:
                reward -= self.config.get("early_termination_penalty", 20.0)
                print("走错路了")


        # 保存每步的信息
        step_info["step_reward"] = reward
        step_info["route_completion"] = vehicle.navigation.route_completion

        return reward, step_info

def create_custom_env(need_monitor=False):
    # Create a custom environment
    env = CustomRewardEnv(dict(
        map="O",
        discrete_action=True,
        discrete_throttle_dim=10,
        discrete_steering_dim=10,
        horizon=500,
        random_spawn_lane_index=True,
        num_scenarios=1800,
        start_seed=8000,
        traffic_density=0.5,
        accident_prob=0.3,
        use_lateral_reward=True,
        driving_reward=1.0,
        speed_reward=0.1,
        success_reward=40.0,
        out_of_road_penalty=20.0,
        crash_vehicle_penalty=20.0,
        crash_object_penalty=20.0,
        log_level=50
    ))
    if need_monitor:
        env = Monitor(env)
    return env

def save_to_json(result, filename):
    """
    Save evaluation results to a JSON file.

    Parameters:
    - result (dict): The result of the current episode.
    - filename (str): The full path to the JSON file.
    """
    # Check if the file already exists
    if os.path.exists(filename):
        # Load existing results
        with open(filename, "r") as f:
            existing_results = json.load(f)
    else:
        # Initialize an empty list if the file doesn't exist
        existing_results = []

    # Append the new result
    existing_results.append(result)

    # Save the updated results
    with open(filename, "w") as f:
        json.dump(existing_results, f, indent=4)

set_random_seed(0)

if __name__ == '__main__':
    success_case = 0
    all_case = 0
    globals()["success_case"] = success_case
    globals()["all_case"] = all_case

    crash_case = 0
    globals()["crash_case"] = crash_case

    total_acc = 0
    globals()["total_acc"] = total_acc

    last_velocity = 0
    globals()["last_velocity"] = last_velocity

    # Create the training environment
    train_env = create_custom_env()

    print("Starting training")

    # Load the pre-trained model
    # 记得改名字
    model = PPO.load("round_model_000.zip", env=train_env, n_steps=4096, verbose=1, device="cuda")

    print("Training is finished!")

    # Number of evaluation episodes
    num_eval_episodes = 2000
    env = create_custom_env()

    # ------------------ Modifications Start Here ------------------

    # Define the experiment output path
    experiment_output_path = None  # User can specify this path
    if experiment_output_path is None:
        # Generate a default path using the current date and time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_output_path = f'exp_{timestamp}'

    # Create the required directories
    os.makedirs(os.path.join(experiment_output_path, 'json'), exist_ok=True)
    os.makedirs(os.path.join(experiment_output_path, 'gif', 'left'), exist_ok=True)
    os.makedirs(os.path.join(experiment_output_path, 'gif', 'straight'), exist_ok=True)
    os.makedirs(os.path.join(experiment_output_path, 'gif', 'right'), exist_ok=True)

    # ------------------ Modifications End Here ------------------

    for episode in range(num_eval_episodes):
        flag = False
        globals()["flag"] = flag
        if episode % 20 == 0:
            print("Processing evaluation case {}".format(episode))
            print(success_case)
            print(all_case)
        total_reward = 0
        frames = []

        obs, _ = env.reset()

        total_acc = 0
        try:
            # Perform evaluation
            for i in range(1000):
                # Use the model to predict actions
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward

                # Render and save each frame
                frame = env.render(
                    mode="topdown",
                    screen_record=True,
                    window=False,
                    screen_size=(600, 1000),
                    camera_position=(50, 50)
                )
                frames.append(frame)

                if done:
                    print(f"Episode {episode + 1} reward:", total_reward)
                    break

            # ------------------ Modifications Start Here ------------------

            # Get start and end positions to determine the turn type
            start = env.vehicle.navigation.final_lane.start  # Start coordinates
            end = env.vehicle.navigation.final_lane.end  # End coordinates
            delta_x = end[0] - start[0]
            delta_y = end[1] - start[1]
            if delta_x == 0 and delta_y > 0:
                turn = "left"
            elif delta_x == 0 and delta_y < 0:
                turn = "right"
            elif delta_x > 0 and delta_y == 0:
                turn = "straight"
            else:
                turn = "straight"  # Default to straight if unknown

            # Determine the subdirectory based on the turn type
            gif_subdir = os.path.join(experiment_output_path, 'gif', turn)

            # Generate the GIF filename
            gif_filename = f"model_new_episode_{episode + 1}.gif"

            # Full path where the GIF will be saved
            gif_path = os.path.join(gif_subdir, gif_filename)

            # Generate the GIF and move it to the appropriate directory

            # 保存为gif
            imageio.mimsave('demo.gif', frames, fps=30)  # fps可调整，推荐在10-30之间
            os.rename("demo.gif", gif_path)
            print(f"Generated {gif_path}")
            print(f"final_lane.start = {env.vehicle.navigation.final_lane.start}")
            print(f"final_lane.end = {env.vehicle.navigation.final_lane.end}")

            # Calculate the relative GIF path
            relative_gif_path = os.path.relpath(gif_path, experiment_output_path)

            acc_average = total_acc / len(frames)

            # Prepare the episode result dictionary
            episode_result = {
                "trial_id": episode,  # Current evaluation ID
                "termination_status": env.termination_status,  # Placeholder
                "ego_responsibility": None,  # Placeholder
                "frames": len(frames),  # Number of frames in the GIF
                "comfort": acc_average,
                "turning": turn,  # Type of turn
                "start": start.tolist(),  # Start coordinates
                "end": end.tolist(),  # End coordinates
                "gif_path": relative_gif_path, # Relative path of the GIF
                "engine_global_random_seed": env.engine.global_random_seed # Record engine random seed
            }

            # Full path to the JSON file
            json_filename = os.path.join(experiment_output_path, 'json', 'result.json')

            # Save the episode result
            save_to_json(episode_result, filename=json_filename)
            print(f"Saved result for episode {episode}")

            # ------------------ Modifications End Here ------------------

        finally:
            env.close()
    print(success_case)
    print(all_case)
    print("All GIFs generated.")

    print("crash_case_cnt={}".format(crash_case))