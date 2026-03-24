import os
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Melew3ClimbCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 0
        num_observations = 43 
        num_actions = 20 
        env_spacing = 3.0  
        episode_length_s = 20
        terrain_file = '/home/mrp/gym/legged_gym/resources/terrains/solid_crevice_80cm.urdf'

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        static_friction = 1.5
        dynamic_friction = 1.5
        # curriculum = False
    
    class sim( LeggedRobotCfg.sim ):
        dt = 0.005  
        substeps = 1
        gravity = [0., 0. ,-9.81]  # m/s^2
        up_axis = 1  # 0 is y, 1 is z
        
        class physx( LeggedRobotCfg.sim.physx ):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8  # 增加迭代次数提高精度
            num_velocity_iterations = 2
            contact_offset = 0.005  # 较小的接触偏移
            rest_offset = 0.0  # 静止偏移
            bounce_threshold_velocity = 0.2
            friction_offset_threshold = 0.01
            friction_correlation_distance = 0.005
            max_depenetration_velocity = 1.0
            use_gpu = True
            num_subscenes = 0  
    
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 
        resampling_time = 10.
        heading_command = True 
        class ranges: 
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-3.14, 3.14]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35]  
        rot = [0.0, 0.0, 0.0, 1.0]
        default_joint_angles = {
            'HIP_YAW_FL': 0.0, 'HIP_ROLL_FL': 0.0, 'HIP_PITCH_FL': 1.5, 'KNEE_FL': -0.7, 'WHEEL_FL': 0.0,
            'HIP_YAW_FR': 0.0, 'HIP_ROLL_FR': 0.0, 'HIP_PITCH_FR': 1.5, 'KNEE_FR': -0.7, 'WHEEL_FR': 0.0,
            'HIP_YAW_BL': 0.0, 'HIP_ROLL_BL': 0.0, 'HIP_PITCH_BL': -1.5, 'KNEE_BL': 0.7, 'WHEEL_BL': 0.0,
            'HIP_YAW_BR': 0.0, 'HIP_ROLL_BR': 0.0, 'HIP_PITCH_BR': -1.5, 'KNEE_BR': 0.7, 'WHEEL_BR': 0.0,
        }

    class asset( LeggedRobotCfg.asset ):

        file = '/home/mrp/gym/legged_gym/resources/robots/melew3/urdf/MELEW3.urdf'

        foot_name = ["WHEEL_FL", "WHEEL_FR", "WHEEL_BL", "WHEEL_BR"] 
        
        penalized_contact_names = ["HIP_YAW_FL", "HIP_YAW_FR", "HIP_YAW_BL", "HIP_YAW_BR"]  
        
        termination_body_name = ""  
        collapse_fixed_joints = False
        self_collisions = 0 
        
        override_inertia = True
        replace_cylinder_with_capsule = False
        
        # DOF 限制 (弧度)
        default_dof_drive_mode = 1  # 使用位置控制
        
        # DOF limits for YAW and ROLL joints - 超严格限制，几乎锁死
        dof_pos_limit_lower = {
            'HIP_YAW_FL': -0.01, 'HIP_ROLL_FL': -0.01, 'HIP_PITCH_FL': -1.57, 'KNEE_FL': -2.09, 'WHEEL_FL': -100.0,
            'HIP_YAW_FR': -0.01, 'HIP_ROLL_FR': -0.01, 'HIP_PITCH_FR': -1.57, 'KNEE_FR': -2.09, 'WHEEL_FR': -100.0,
            'HIP_YAW_BL': -0.01, 'HIP_ROLL_BL': -0.01, 'HIP_PITCH_BL': -1.57, 'KNEE_BL': -2.09, 'WHEEL_BL': -100.0,
            'HIP_YAW_BR': -0.01, 'HIP_ROLL_BR': -0.01, 'HIP_PITCH_BR': -1.57, 'KNEE_BR': -2.09, 'WHEEL_BR': -100.0,
        }
        
        dof_pos_limit_upper = {
            'HIP_YAW_FL': 0.01, 'HIP_ROLL_FL': 0.01, 'HIP_PITCH_FL': 1.57, 'KNEE_FL': 2.09, 'WHEEL_FL': 100.0,
            'HIP_YAW_FR': 0.01, 'HIP_ROLL_FR': 0.01, 'HIP_PITCH_FR': 1.57, 'KNEE_FR': 2.09, 'WHEEL_FR': 100.0,
            'HIP_YAW_BL': 0.01, 'HIP_ROLL_BL': 0.01, 'HIP_PITCH_BL': 1.57, 'KNEE_BL': 2.09, 'WHEEL_BL': 100.0,
            'HIP_YAW_BR': 0.01, 'HIP_ROLL_BR': 0.01, 'HIP_PITCH_BR': 1.57, 'KNEE_BR': 2.09, 'WHEEL_BR': 100.0,
        }

    class control( LeggedRobotCfg.control ):
        stiffness = {
            'HIP_YAW_FL': 40.0, 'HIP_ROLL_FL': 50.0, 'HIP_PITCH_FL': 80.0, 'KNEE_FL': 70.0, 'WHEEL_FL': 0.0,
            'HIP_YAW_FR': 40.0, 'HIP_ROLL_FR': 50.0, 'HIP_PITCH_FR': 80.0, 'KNEE_FR': 70.0, 'WHEEL_FR': 0.0,
            'HIP_YAW_BL': 40.0, 'HIP_ROLL_BL': 50.0, 'HIP_PITCH_BL': 80.0, 'KNEE_BL': 70.0, 'WHEEL_BL': 0.0,
            'HIP_YAW_BR': 40.0, 'HIP_ROLL_BR': 50.0, 'HIP_PITCH_BR': 80.0, 'KNEE_BR': 70.0, 'WHEEL_BR': 0.0,
        }
        damping = {
            'HIP_YAW_FL': 3.0, 'HIP_ROLL_FL': 4.0, 'HIP_PITCH_FL': 6.0, 'KNEE_FL': 5.0, 'WHEEL_FL': 0.1,
            'HIP_YAW_FR': 3.0, 'HIP_ROLL_FR': 4.0, 'HIP_PITCH_FR': 6.0, 'KNEE_FR': 5.0, 'WHEEL_FR': 0.1,
            'HIP_YAW_BL': 3.0, 'HIP_ROLL_BL': 4.0, 'HIP_PITCH_BL': 6.0, 'KNEE_BL': 5.0, 'WHEEL_BL': 0.1,
            'HIP_YAW_BR': 3.0, 'HIP_ROLL_BR': 4.0, 'HIP_PITCH_BR': 6.0, 'KNEE_BR': 5.0, 'WHEEL_BR': 0.1,
        }
        action_scale = 0.25  # 减小动作尺度，让控制更温和 

    class rewards:
        soft_dof_pos_limit = 0.9
        only_positive_rewards = False
        base_height_target = 4.0  # 大幅提高目标高度，鼓励向更高处攀爬！
        
        # 关节角度限制（度为单位，转换为弧度） 
        hip_joint_limit_deg = 7.0   # 温和限制YAW和ROLL，允许7度
        knee_joint_limit_deg = 120.0  # 膝盖需要较大范围用于攀爬
        
        class scales:
            # 适度惩罚消极行为
            termination = -50.0
            torques = -0.00001             # 减少扭矩惩罚，鼓励大动作
            
            # 🚀 攀爬任务核心：极端疯狂的高度奖励！！！
            climbing_progress = 2000.0     # 极端提高！攀爬是绝对目标
            height_reward = 1500.0         # 极端提高！高度就是生命
            height_bonus = 1200.0          # 极端提高：高度额外奖励
            upward_velocity = 800.0        # 极端提高：向上速度超重要
            
            # 严厉打击关节外扭 - 解决腿部和轮子外扩问题
            yaw_stability = 300.0          # 巨幅加强：强制YAW归零
            yaw_roll_violation = -500.0    # 史上最严厉：外扭直接重罚
            wheel_yaw_control = 200.0      # 新增：专门控制轮子YAW
            leg_symmetry = 200.0           # 巨幅加强：腿部对称性
            
            # 卡墙攀爬技术奖励
            wall_press_climbing = 80.0     # 适度奖励卡墙技术
            leg_wall_coordination = 60.0   # 适度奖励腿部协调
            wheel_leg_synergy = 70.0       # 适度奖励轮腿协同
            climbing_posture = 50.0        # 适度奖励攀爬姿态
            
            # 攀爬支持奖励
            wheel_rotation = 40.0          # 奖励轮子转动
            wall_contact = 30.0            # 奖励墙面接触
            
            # 身体姿态奖励 - 不要过度约束攀爬动作
            body_horizontal = 10.0         # 减少，允许攀爬时的身体调整
            upright_bonus = 5.0            # 减少，不阻止攀爬姿态
            
            # 腿部运动奖励
            leg_lifting = 30.0             # 鼓励抬腿
            coordinated_movement = 20.0    # 鼓励协调运动
            active_movement = 15.0         # 鼓励主动移动
            hip_pitch_coordination = 100.0 # 加强：正确的HIP_PITCH运动
            
            # 严厉惩罚阻碍攀爬的行为
            staying_still = -20.0          # 加强惩罚静止
            low_height = -100.0            # 巨幅加强：严厉惩罚低高度
            height_loss = -200.0           # 史上最严厉：下降是绝对禁止的
            downward_velocity = -150.0     # 史上最严厉：向下运动重罚
            wrong_hip_movement = -200.0    # 史上最严厉：错误腿部运动重罚
            lateral_drift = -5.0           # 适度惩罚侧向偏移
            backward_movement = -10.0      # 适度惩罚后退
            wrong_direction = -3.0         # 减少错误方向惩罚
            
            # 严厉控制YAW扭转 - 解决外扭问题
            yaw_violation = -200.0         # 超强惩罚YAW偏移，防止外扭
            yaw_stability = 50.0           # 新增：奖励YAW保持接近0
            roll_violation = -50.0         # 适度惩罚ROLL偏移
            hip_joint_limit = -1.0         # 减少关节限制惩罚
            knee_joint_limit = -0.5        # 进一步减少膝盖限制
            
            # 鼓励性奖励 - 增加这些来鼓励探索
            forward_progress = 15.0        # 提高向前移动奖励
            momentum = 10.0                # 提高动量奖励，鼓励持续动作
            centering = -0.05              # 大幅减少居中要求

    class noise( LeggedRobotCfg.noise ):
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            ang_vel = 0.2
            gravity = 0.05
            lin_vel = 0.0
            height_measurements = 0.0 
    
    class obs_scales:
        dof_pos = 1.0
        dof_vel = 0.05
        ang_vel = 0.25
        lin_vel = 0.1
        height_measurements = 1.0

class Melew3ClimbCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'melew3_climb'