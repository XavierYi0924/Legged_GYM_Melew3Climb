from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Melew3FlatCfg(LeggedRobotCfg):
    """
    Configuration for the Melew3 robot in a flat terrain environment.
    This class defines environment settings, robot assets, and control parameters.
    """
    class env(LeggedRobotCfg.env):
        num_envs = 1024*9  # 降低环境数量以避免 CUDA OOM (原为 1024*9=9216)
        num_observations = 73
        num_actions = 20
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = True
        selected = True
        
        # 我们手动控制课程，所以设为False
        curriculum = False
        
        static_friction = 1.5
        dynamic_friction = 1.5
        
        # ✅ 基于论文实验数据设计的“第一阶段”地形参数
        # terrain_kwargs = {
        #     'type': 'progressive_climb_terrain',  # 你的自定义地形类型名称
        #     'channel_width':0.64,
        #     'wall_height': 2.0,
        #     'wall_thickness': 0.2,    # 减少墙壁厚度到20cm
        #     'u_shape_depth': 0.2,     # 对应论文中的 r 参
        #     'slope_angle': 0,         # 初始阶段不设坡度
        # }
        terrain_kwargs = {
            'type': 'channel_terrain',  # 你的自定义地形类型名称
            'channel_width': 0.78, 
            'wall_height': 2.0,
            'wall_thickness': 0.2,    # 减少墙壁厚度到20cm
        }

        # 其他参数保持不变
        terrain_length = 16.
        terrain_width = 16.
        border_size = 25
        num_rows = 4
        num_cols = 8

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        rot = [0.0, 0.0, 0.819152, 0.573576]  # yaw=110°，补偿策略的顺时针旋转，防止BL悬空
        default_joint_angles = {
            "HIP_YAW_FL": 0.0,
            "HIP_ROLL_FL": 0.0,
            "HIP_PITCH_FL": 0.8,   
            "KNEE_FL": -1.5,      
            "WHEEL_FL": 0.0,

            "HIP_YAW_FR": 0.0,
            "HIP_ROLL_FR": 0.0,
            "HIP_PITCH_FR": 0.8,   
            "KNEE_FR": -1.5,      
            "WHEEL_FR": 0.0,

            "HIP_YAW_BL": 0.0,
            "HIP_ROLL_BL": 0.0,
            "HIP_PITCH_BL": -0.8,  
            "KNEE_BL": 1.5,      
            "WHEEL_BL": 0.0,
            
            "HIP_YAW_BR": 0.0,
            "HIP_ROLL_BR": 0.0,
            "HIP_PITCH_BR": -0.8,  
            "KNEE_BR": 1.5,       
            "WHEEL_BR": 0.0,
        }

    class control(LeggedRobotCfg.control):

        stiffness = {
            'HIP_YAW': 40., 'HIP_ROLL': 40., 'HIP_PITCH': 40.,
            'KNEE': 40., 'WHEEL': 10000
        }
        damping = {
            'HIP_YAW': 1.0, 'HIP_ROLL': 1.0, 'HIP_PITCH': 1.0,
            'KNEE': 1.0, 'WHEEL': 10000  # 🔥 改为0：轮子速度控制不需要阻尼，阻尼会抵消控制力矩！
        }
        action_scale = 0.25
        decimation = 4
        
        # 轮子速度控制参数
        wheel_velocity_control = False  # 是否启用轮子恒定速度控制
        wheel_target_velocity =0.0  # 轮子目标角速度 [rad/s]
        wheel_velocity_kp = 85.0  # 🔥 提高P增益（从10.0到50.0，加快响应速度）
        wheel_velocity_kd = 5  # 速度控制D增益（提高以增加阻尼）
        wheel_max_torque = 500.0  # 🔥 提高最大力矩限制（从50到500，充分驱动轮子）

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/melew3/urdf/MELEW3.urdf"
        name = "melew3"
        foot_name = "WHEEL"
        self_collisions = 1
        penalize_contacts_on = ["HIP_PITCH", "KNEE", "HIP_ROLL", "HIP_YAW", "THIGH", "SHANK"]
        terminate_after_contacts_on = ["base"]  # 恢复base终止条件，摔倒后立即重置
        fix_base_link = False
        default_dof_drive_mode = 3 
        collapse_fixed_joints = True
        replace_cylinder_with_capsule = True

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = False
        added_mass_range = [-5., 5.]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 10
        randomize_friction = True
        friction_range = [0.5, 1.0]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 
        resampling_time = 10. 
        heading_command = True 

        class ranges:
            lin_vel_x = [0.0, 0.0]   # 暂时禁用前进命令，专注爬墙
            lin_vel_y = [0.0, 0.0]   
            ang_vel_yaw = [0.0, 0.0] 
            heading = [-3.14, 3.14]

class Melew3FlatCfgPPO(LeggedRobotCfgPPO):
    """
    Configuration for the PPO algorithm and runner for Melew3.
    """
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        schedule = 'adaptive'          # 使用自适应学习率
        entropy_coef = 0.01            # 🔥 提高熵系数，鼓励更多探索
        learning_rate = 3e-4           # � 进一步降低学习率（微调专用）
        # gamma = 0.99                   # 🔥 降低折扣因子，更关注短期奖励
        # lam = 0.95                     # GAE lambda


    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'ultra_slow_climb_v1'  # 🔥🔥 新方案：超慢速爬升（0.4 m/s限制）
        experiment_name = 'flat_melew3'
        resume = True
        # load_run = "Oct22_12-07-15_capped_exponential_v1"
        # load_run = "Oct01_11-23-46_climbing_rewards_final"
        # load_run = "Oct27_12-25-37_contact_gated_height_v1"  # 最新的
        # load_run = "Oct27_18-46-11_contact_gated_height_v1"  # 6500
        # load_run = "Dec08_21-04-59_contact_gated_height_v1"  # 最新的
        # load_run = "Dec09_18-51-04_contact_gated_height_v1"
        # load_run = "Dec13_12-27-17_contact_gated_height_v1"
        load_run = "Jan18_19-32-52_ultra_slow_climb_v1"  
        # checkpoint = 600
        # checkpoint = 9800
        checkpoint = 6300  # 使用最新checkpoint
        max_iterations = 4000  # 继续训练4000轮

    class rewards(LeggedRobotCfg.rewards):
        # 严格按照论文TABLE II设置奖励权重
        base_height_target = 0.4  # 机器人初始高度为0.4，考虑基座旋转
        soft_dof_pos_limit = 0.8  # 对应表格中的80%限制