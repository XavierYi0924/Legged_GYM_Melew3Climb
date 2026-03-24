# 测试配置：机器人悬空，测试轮子是否能转动

from legged_gym.envs.melew3_flat.melew3_flat_config import Melew3FlatCfg

class Melew3TestWheelCfg(Melew3FlatCfg):
    class env(Melew3FlatCfg.env):
        num_envs = 1024
    
    class init_state(Melew3FlatCfg.init_state):
        pos = [0.0, 0.0, 5.0]  # 进一步提高高度到5.0m，确保轮子完全离地
        default_joint_angles = {
            # FL leg - 腿部向下垂直伸直
            "HIP_YAW_FL": 0.0,
            "HIP_ROLL_FL": 0.0,
            "HIP_PITCH_FL": -1.57,  # 90度向下
            "KNEE_FL": 0.0,
            "WHEEL_FL": 0.0,
            # FR leg
            "HIP_YAW_FR": 0.0,
            "HIP_ROLL_FR": 0.0,
            "HIP_PITCH_FR": -1.57,
            "KNEE_FR": 0.0,
            "WHEEL_FR": 0.0,
            # BL leg
            "HIP_YAW_BL": 0.0,
            "HIP_ROLL_BL": 0.0,
            "HIP_PITCH_BL": -1.57,
            "KNEE_BL": 0.0,
            "WHEEL_BL": 0.0,
            # BR leg
            "HIP_YAW_BR": 0.0,
            "HIP_ROLL_BR": 0.0,
            "HIP_PITCH_BR": -1.57,
            "KNEE_BR": 0.0,
            "WHEEL_BR": 0.0,
        }
    
    class control(Melew3FlatCfg.control):
        wheel_velocity_control = True
        wheel_target_velocity = 10.0  # 目标速度 10 rad/s
        wheel_velocity_kp = 500.0  # P增益很大
        wheel_velocity_kd = 0.5  # 减小D增益以减少阻尼
        wheel_max_torque = 500.0  # URDF最大值
        # 👇 关键：覆盖轮子的 damping，设为 0 以允许自由旋转
        damping = {
            'HIP_YAW': 1.0, 'HIP_ROLL': 1.0, 'HIP_PITCH': 1.0,
            'KNEE': 1.0, 'WHEEL': 0.0  # 改为 0！
        }
    
    class asset(Melew3FlatCfg.asset):
        fix_base_link = True  # 固定base，让机器人悬空不掉落
