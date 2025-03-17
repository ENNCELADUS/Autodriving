from sympy.physics.units import acceleration

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
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from PIL import Image

class CustomRewardEnv(MetaDriveEnv):
    def __init__(self, config):
        super().__init__(config)


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
        acceleration = 0.02 * (vehicle.speed_km_h - last_velocity)
        reward -= abs(acceleration)
        # print("加速度奖励={}".format(acceleration))

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
    # 创建自定义环境
    env = CustomRewardEnv(dict(
        map="O",
        discrete_action=True,
        discrete_throttle_dim=10,
        discrete_steering_dim=10,
        horizon=500,
        random_spawn_lane_index=True,
        num_scenarios=2000,
        start_seed=5, #训练时写的是5
        traffic_density=0.5,
        accident_prob=0.3,
        use_lateral_reward=True,
        driving_reward=1.0,
        speed_reward=0.1,
        success_reward=40.0,
        out_of_road_penalty=20.0,
        crash_vehicle_penalty=20.0,
        crash_object_penalty=20.0,
        log_level=50,
    ))
    if need_monitor:
        env = Monitor(env)
    return env



set_random_seed(0)





if __name__ == '__main__':
    # # 检查设备是否支持 GPU
    # device = torch.device("cpu")
    # print(f"Using device: {device}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    crash_case = 0   #本次训练初始化crash_case = 0
    globals()["crash_case"] = crash_case

    last_velocity = 0
    globals()["last_velocity"] = last_velocity

    # 创建并行环境
    train_env = create_custom_env()  #最开始是4

    # # # 加载或初始化模型

    # 每换一次系数改一下名字
    log_dir = "./ppo_logs_002/"
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # 定义保存模型的回调函数，每10000步保存一次模型0
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/',
                                             name_prefix='ppo_model')

    # 创建PPO模型
    #model = PPO.load("traditional_r.zip", env=train_env,n_steps=4096, verbose=1, device="cpu",tensorboard_log=log_dir)
    model = PPO("MlpPolicy", train_env, n_steps=4096, verbose=1, device="cpu", tensorboard_log=log_dir)
    model.set_logger(new_logger)

    # 使用回调函数训练模型
    model.learn(total_timesteps=1800000, callback=checkpoint_callback)

    # 保存模型（修改名称）
    model.save("round_model_002")
    clear_output()
    print("Training is finished!")



# 居中奖励=0.014857402529035295
# 前进奖励=0.24815845489501953
# 速度奖励=0.012165070015951233