from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Melew3FlatCfg(LeggedRobotCfg):
    """
    Configuration for the Melew3 robot in a flat terrain environment.
    This class defines environment settings, robot assets, and control parameters.
    """
    class env(LeggedRobotCfg.env):
        num_envs = 1024*9
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
        terrain_kwargs = {
            'type': 'progressive_climb_terrain',  # 你的自定义地形类型名称
            'channel_width':0.65,
            'wall_height': 2.0,
            'wall_thickness': 0.2,    # 减少墙壁厚度到20cm
            'u_shape_depth': 0.2,     # 对应论文中的 r 参
            'slope_angle': 0,         # 初始阶段不设坡度
        }

        # 其他参数保持不变
        terrain_length = 16.
        terrain_width = 16.
        border_size = 25
        num_rows = 4
        num_cols = 8

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        rot = [0.0, 0.0, 0.707, 0.707]  
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
            'KNEE': 40., 'WHEEL': 40.
        }
        damping = {
            'HIP_YAW': 1.0, 'HIP_ROLL': 1.0, 'HIP_PITCH': 1.0,
            'KNEE': 1.0, 'WHEEL': 1.0
        }
        action_scale = 0.25
        decimation = 4

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
        push_robots = False
        push_interval_s = 7
        max_push_vel_xy = 0.
        randomize_friction = False
        friction_range = [1.0, 1.0]

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
        entropy_coef = 0.01
        learning_rate = 5e-5

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'climbing_rewards_v1'  # 新的实验名称
        experiment_name = 'flat_melew3'
        load_run = -1                    # 加载最新的checkpoint，如果没有则从头开始
        max_iterations = 6000
        
    class rewards(LeggedRobotCfg.rewards):
        # 严格按照论文TABLE II设置奖励权重
        base_height_target = 0.4  # 机器人初始高度为0.4，考虑基座旋转
        soft_dof_pos_limit = 0.8  # 对应表格中的80%限制
        