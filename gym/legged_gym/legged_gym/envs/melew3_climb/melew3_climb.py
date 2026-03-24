import os
import torch
import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymtorch

class Melew3Climb(LeggedRobot):

    def _create_envs(self):
        # 加载机器人资源
        robot_asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        robot_asset_root = os.path.dirname(robot_asset_path)
        robot_asset_file = os.path.basename(robot_asset_path)
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.fix_base_link = self.cfg.asset.fix_base_link
        robot_asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        robot_asset_options.flip_visual_attachments = True  # 修复网格方向问题
        robot_asset_options.armature = 0.01  # 添加少量armature提高稳定性
        robot_asset_options.thickness = 0.01  # 设置碰撞厚度
        robot_asset_options.disable_gravity = False  # 确保重力作用于机器人
        robot_asset_options.use_mesh_materials = True  # 使用网格材质
        robot_asset = self.gym.load_asset(self.sim, robot_asset_root, robot_asset_file, robot_asset_options)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)

        # 加载地形资源 (通过solid_crevice.urdf)
        terrain_asset_path = self.cfg.env.terrain_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        terrain_asset_root = os.path.dirname(terrain_asset_path)
        terrain_asset_file = os.path.basename(terrain_asset_path)
        terrain_asset_options = gymapi.AssetOptions()
        terrain_asset_options.fix_base_link = True
        terrain_asset_options.disable_gravity = True
        # 关键修改：优化VHACD设置以提高碰撞检测精度
        terrain_asset_options.vhacd_enabled = True
        terrain_asset_options.vhacd_params = gymapi.VhacdParams()
        terrain_asset_options.vhacd_params.resolution = 100000  # 降低分辨率提高稳定性
        terrain_asset_options.vhacd_params.max_convex_hulls = 16  # 限制凸包数量
        terrain_asset_options.vhacd_params.max_num_vertices_per_ch = 64  # 限制顶点数
        terrain_asset_options.vhacd_params.min_volume_per_ch = 0.0001  # 最小体积阈值
        terrain_asset = self.gym.load_asset(self.sim, terrain_asset_root, terrain_asset_file, terrain_asset_options)

        # 创建环境和actor
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        
        # 计算网格布局
        envs_per_row = int(np.sqrt(self.num_envs))
        spacing = self.cfg.env.env_spacing
        
        for i in range(self.num_envs):
            # 计算每个环境的偏移位置
            row = i // envs_per_row
            col = i % envs_per_row
            env_offset_x = (col - envs_per_row // 2) * spacing
            env_offset_y = (row - envs_per_row // 2) * spacing
            
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            
            # 地形位置保持在原点
            terrain_pose = gymapi.Transform()
            terrain_pose.p = gymapi.Vec3(env_offset_x, env_offset_y, 0.0)
            terrain_handle = self.gym.create_actor(env_handle, terrain_asset, terrain_pose, "terrain", i, 0, 0)
            
            # 关键修改：设置地形的物理属性以防止穿透
            terrain_props = self.gym.get_actor_rigid_shape_properties(env_handle, terrain_handle)
            for prop in terrain_props:
                prop.friction = 1.0  # 高摩擦力
                prop.restitution = 0.0  # 无弹性
                prop.contact_offset = 0.005  # 较小的接触偏移
                prop.rest_offset = 0.0  # 静止偏移为0
            self.gym.set_actor_rigid_shape_properties(env_handle, terrain_handle, terrain_props)
            
            # 机器人位置加上环境偏移
            pos = self.cfg.init_state.pos
            orientation = gymapi.Quat(0.0, 0.0, 0.0, 1.0) 
            robot_start_pose = gymapi.Transform()
            robot_start_pose.p = gymapi.Vec3(pos[0] + env_offset_x, pos[1] + env_offset_y, pos[2])
            robot_start_pose.r = orientation
            
            # 保存环境原点用于后续重置
            self.env_origins[i, 0] = env_offset_x
            self.env_origins[i, 1] = env_offset_y
            self.env_origins[i, 2] = 0.0
            
            actor_handle = self.gym.create_actor(env_handle, robot_asset, robot_start_pose, "melew3", i, self.cfg.asset.self_collisions, 0)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        foot_names = self.cfg.asset.foot_name
        if isinstance(foot_names, str): foot_names = [foot_names]
        self.feet_indices = torch.tensor([self.gym.find_asset_rigid_body_index(robot_asset, name) for name in foot_names], device=self.device)
        self.wheel_indices = self.feet_indices.clone()

        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.torque_limits = torch.tensor(dof_props['effort'], device=self.device)

        foot_names = self.cfg.asset.foot_name
        if isinstance(foot_names, str):
            foot_names = [foot_names]
        self.feet_indices = torch.tensor([self.gym.find_asset_rigid_body_index(robot_asset, name) for name in foot_names], device=self.device)
        
        # 修复终止接触索引 - 排除空字符串
        termination_contact_names = self.cfg.asset.penalized_contact_names
        if self.cfg.asset.termination_body_name and self.cfg.asset.termination_body_name.strip():
            termination_contact_names = termination_contact_names + [self.cfg.asset.termination_body_name]
        
        # 如果没有终止接触体，创建一个空的tensor
        if termination_contact_names:
            self.termination_contact_indices = torch.tensor([self.gym.find_asset_rigid_body_index(robot_asset, name) for name in termination_contact_names], device=self.device)
        else:
            self.termination_contact_indices = torch.tensor([], device=self.device, dtype=torch.long)

    def _init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        robot_indices = torch.arange(1, 2 * self.num_envs, 2, device=self.device, dtype=torch.long)
        self.base_pos = self.root_states[robot_indices, 0:3]
        self.base_quat = self.root_states[robot_indices, 3:7]
        self.base_lin_vel = self.root_states[robot_indices, 7:10]
        self.base_ang_vel = self.root_states[robot_indices, 10:13]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
        self.common_step_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.max_terrain_level = 0
        self.custom_origins = False
        self.base_init_state = self.root_states[robot_indices].clone()
        self.base_init_state[:, :3] = to_torch(self.cfg.init_state.pos, device=self.device)
        self.base_init_state[:, 3:7] = to_torch(self.cfg.init_state.rot, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        # initialize PD gains
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            self.p_gains[i] = self.cfg.control.stiffness.get(name, 0.)
            self.d_gains[i] = self.cfg.control.damping.get(name, 0.)
        
        # 🔧 **激进方案：强制YAW和ROLL关节的PD gains为0**
        yaw_indices = [0, 5, 10, 15]    # YAW关节索引
        roll_indices = [1, 6, 11, 16]   # ROLL关节索引
        hip_pitch_indices = [2, 7, 12, 17]  # HIP_PITCH关节索引
        
        for idx in yaw_indices:
            self.p_gains[idx] = 0.0  # YAW关节P增益为0
            self.d_gains[idx] = 0.0  # YAW关节D增益为0
        for idx in roll_indices:
            self.p_gains[idx] = 0.0  # ROLL关节P增益为0
            self.d_gains[idx] = 0.0  # ROLL关节D增益为0
        
        # **新增：降低HIP_PITCH关节的PD增益，减少过度控制** - 进一步降低到20%
        for idx in hip_pitch_indices:
            self.p_gains[idx] *= 0.2  # 降低HIP_PITCH的P增益到20%
            self.d_gains[idx] *= 0.2  # 降低HIP_PITCH的D增益到20%

    def _reset_root_states(self, env_ids):
        robot_actor_indices = 2 * env_ids + 1
        self.root_states[robot_actor_indices] = self.base_init_state[env_ids]
        # 加上环境偏移，让每个机器人回到自己的分散位置
        self.root_states[robot_actor_indices, :3] += self.env_origins[env_ids]

    def compute_observations(self):
        self.obs_buf = torch.cat((
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 20维：关节位置误差
            self.dof_vel * self.obs_scales.dof_vel,                           # 20维：关节速度
            self.base_ang_vel * self.obs_scales.ang_vel,                      # 3维：机体角速度
        ), dim=-1)

    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew 
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _reward_climbing_progress(self):
        # 奖励Z轴（向上）的速度
        return self.base_lin_vel[:, 2]

    def _reward_outward_pressure(self):
        # 奖励对墙壁的侧向压力
        # 假设左侧轮子在Y轴正方向，右侧在Y轴负方向
        # 我们奖励左轮受到的Y轴负向接触力，和右轮受到的Y轴正向接触力
        left_wheel_indices = self.feet_indices[[0, 2]]  # FL, BL
        right_wheel_indices = self.feet_indices[[1, 3]]  # FR, BR

        left_pressure = torch.sum(torch.clamp(-self.contact_forces[:, left_wheel_indices, 1], min=0, max=50), dim=1)
        right_pressure = torch.sum(torch.clamp(self.contact_forces[:, right_wheel_indices, 1], min=0, max=50), dim=1)
        return (left_pressure + right_pressure) / 50.0  # 归一化到合理范围

    def _reward_body_orientation(self):
        return self.projected_gravity[:, 2]

    def _reward_centering(self):
        return torch.square(self.base_pos[:, 0]) + torch.square(self.base_pos[:, 1])

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1) / 10000.0  # 归一化
    
    def _reward_height_reward(self):
        """奖励机器人的高度增加 - 修复为纯正值奖励"""
        # 基于绝对高度给予奖励，高度越高奖励越大
        current_height = self.base_pos[:, 2]
        # 使用指数奖励，鼓励攀爬得更高
        height_reward = torch.clamp(current_height, min=0.0)  # 确保非负
        return height_reward
    
    def _reward_forward_progress(self):
        """奖励向前（Y轴正方向）的移动，假设裂缝沿Y轴方向"""
        # 使用Y轴速度作为前进奖励
        forward_vel = self.base_lin_vel[:, 1]  # Y轴速度
        return torch.clamp(forward_vel, min=0.0, max=2.0)  # 限制最大奖励
    
    def _reward_lateral_drift(self):
        """惩罚侧向（X轴）偏移，保持在裂缝中央"""
        lateral_pos = torch.abs(self.base_pos[:, 0])  # X轴偏移
        return -lateral_pos  # 负奖励，偏移越大惩罚越大
    
    def _reward_backward_movement(self):
        """惩罚后退运动"""
        backward_vel = torch.clamp(-self.base_lin_vel[:, 1], min=0.0)  # 负Y速度
        return -backward_vel
    
    def _reward_wrong_direction(self):
        """惩罚错误方向移动（例如转向侧面平整地面）"""
        # 简化版本：通过检查机器人速度方向是否偏离Y轴来判断
        vel_magnitude = torch.norm(self.base_lin_vel[:, :2], dim=1)  # XY平面速度大小
        y_vel_ratio = torch.abs(self.base_lin_vel[:, 1]) / (vel_magnitude + 1e-6)  # Y轴速度占比
        wrong_direction_penalty = 1.0 - y_vel_ratio  # Y轴速度占比越小，惩罚越大
        return -wrong_direction_penalty
    
    def _reward_hip_joint_limit(self):
        """惩罚髋关节（HIP_YAW, HIP_ROLL, HIP_PITCH）过度扭曲"""
        hip_limit_rad = np.radians(self.cfg.rewards.hip_joint_limit_deg)
        
        # 髋关节索引：HIP_YAW, HIP_ROLL, HIP_PITCH for each leg
        hip_indices = []
        for leg in ['BL', 'BR', 'FL', 'FR']:
            for joint in ['HIP_YAW', 'HIP_ROLL', 'HIP_PITCH']:
                joint_name = f'{joint}_{leg}'
                if joint_name in self.dof_names:
                    hip_indices.append(self.dof_names.index(joint_name))
        
        if not hip_indices:
            return torch.zeros(self.num_envs, device=self.device)
        
        hip_positions = self.dof_pos[:, hip_indices]
        # 计算超出限制的关节角度
        violations = torch.clamp(torch.abs(hip_positions) - hip_limit_rad, min=0.0)
        total_violation = torch.sum(violations, dim=1)
        return -total_violation  # 负奖励
    
    def _reward_knee_joint_limit(self):
        """惩罚膝关节（KNEE）过度扭曲"""
        knee_limit_rad = np.radians(self.cfg.rewards.knee_joint_limit_deg)
        
        # 膝关节索引
        knee_indices = []
        for leg in ['BL', 'BR', 'FL', 'FR']:
            joint_name = f'KNEE_{leg}'
            if joint_name in self.dof_names:
                knee_indices.append(self.dof_names.index(joint_name))
        
        if not knee_indices:
            return torch.zeros(self.num_envs, device=self.device)
        
        knee_positions = self.dof_pos[:, knee_indices]
        # 计算超出限制的关节角度
        violations = torch.clamp(torch.abs(knee_positions) - knee_limit_rad, min=0.0)
        total_violation = torch.sum(violations, dim=1)
        return -total_violation  # 负奖励
    
    def _reward_base_height(self):
        """奖励保持适当的基础高度"""
        height_diff = torch.abs(self.base_pos[:, 2] - self.cfg.rewards.base_height_target)
        return -height_diff  # 偏离目标高度越多，惩罚越大
    
    def _reward_wheel_rotation(self):
        """奖励轮子主动旋转，避免机器人躺着不动"""
        # 找到轮子关节的索引
        wheel_indices = []
        for i, name in enumerate(self.dof_names):
            if 'WHEEL' in name:
                wheel_indices.append(i)
        
        if not wheel_indices:
            return torch.zeros(self.num_envs, device=self.device)
            
        # 计算轮子角速度的平均值
        wheel_vel = self.dof_vel[:, wheel_indices]
        wheel_rotation_reward = torch.mean(torch.abs(wheel_vel), dim=1)
        
        # 归一化到合理范围 (0-1)
        return torch.clamp(wheel_rotation_reward / 10.0, max=1.0)
    
    def _reward_wall_contact(self):
        """奖励与墙壁的有效接触，确保机器人贴近墙面"""
        # 轮子与墙面的接触
        wheel_forces = self.contact_forces[:, self.feet_indices, :]
        # 计算侧向接触力 (Y轴)，表示与墙壁的压力
        lateral_contact = torch.abs(wheel_forces[:, :, 1])  # Y轴接触力
        total_wall_contact = torch.sum(lateral_contact, dim=1)
        
        # 归一化并限制最大奖励
        return torch.clamp(total_wall_contact / 100.0, max=2.0)
    
    def _reward_active_movement(self):
        """奖励主动移动，避免静止不动"""
        # 计算机器人的总体运动
        total_vel = torch.norm(self.base_lin_vel, dim=1)  # 线速度大小
        angular_vel = torch.norm(self.base_ang_vel, dim=1)  # 角速度大小
        
        # 综合运动奖励
        movement_reward = total_vel + 0.5 * angular_vel
        return torch.clamp(movement_reward / 3.0, max=1.0)
    
    def _reward_staying_still(self):
        """惩罚保持静止状态"""
        # 检测机器人是否几乎静止
        vel_threshold = 0.1  # 速度阈值
        total_vel = torch.norm(self.base_lin_vel, dim=1)
        
        # 如果速度太小，给予惩罚
        is_still = total_vel < vel_threshold
        return -is_still.float()  # 静止时返回-1，运动时返回0
    
    def _reward_low_height(self):
        """严重惩罚低高度（躺在地上）"""
        # 如果机器人高度太低，给予重惩罚
        low_height_threshold = 0.2  # 高度阈值
        is_too_low = self.base_pos[:, 2] < low_height_threshold
        return -is_too_low.float() * 2.0  # 太低时返回-2，正常时返回0
    
    def _reward_upright_bonus(self):
        """奖励保持直立姿态"""
        # 使用projected_gravity来判断机器人是否直立
        # projected_gravity[:, 2] 接近1表示直立
        upright_score = self.projected_gravity[:, 2]
        # 只有当直立度超过阈值时才给奖励
        upright_threshold = 0.7
        upright_bonus = torch.clamp(upright_score - upright_threshold, min=0.0)
        return upright_bonus
    
    def _reward_coordinated_movement(self):
        """奖励协调的腿部和轮子运动"""
        # 检查腿部关节的协调性
        leg_joint_indices = []
        wheel_indices = []
        
        for i, name in enumerate(self.dof_names):
            if 'WHEEL' in name:
                wheel_indices.append(i)
            elif any(joint_type in name for joint_type in ['HIP', 'KNEE']):
                leg_joint_indices.append(i)
        
        if not leg_joint_indices or not wheel_indices:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 计算腿部关节活动度
        leg_activity = torch.mean(torch.abs(self.dof_vel[:, leg_joint_indices]), dim=1)
        # 计算轮子活动度
        wheel_activity = torch.mean(torch.abs(self.dof_vel[:, wheel_indices]), dim=1)
        
        # 协调奖励：当腿部和轮子都在运动时给予奖励
        coordination_score = torch.min(leg_activity, wheel_activity)
        return torch.clamp(coordination_score / 5.0, max=1.0)
    
    def _reward_momentum(self):
        """奖励保持动量，鼓励持续运动"""
        # 计算机器人的总动量（速度的持续性）
        velocity_magnitude = torch.norm(self.base_lin_vel, dim=1)
        
        # 奖励持续的中等速度运动
        optimal_velocity = 0.8  # 理想速度
        momentum_score = 1.0 - torch.abs(velocity_magnitude - optimal_velocity) / optimal_velocity
        
        # 确保动量分数为正
        momentum_score = torch.clamp(momentum_score, min=0.0)
        
        return momentum_score
    
    def _reward_body_horizontal(self):
        """奖励保持身体水平（攀爬时的理想姿态）"""
        # 使用projected_gravity来判断身体倾斜度
        # 当机器人水平时，projected_gravity[:, 0] 和 [:, 1] 应该接近0
        roll_tilt = torch.abs(self.projected_gravity[:, 0])  # roll倾斜
        pitch_tilt = torch.abs(self.projected_gravity[:, 1])  # pitch倾斜
        
        # 总倾斜度（越小越好）
        total_tilt = roll_tilt + pitch_tilt
        
        # 奖励水平姿态（倾斜度小）
        horizontal_score = torch.clamp(1.0 - total_tilt, min=0.0)
        return horizontal_score
    
    def _reward_leg_lifting(self):
        """奖励抬腿动作，鼓励攀爬行为"""
        # 检测腿部关节的运动幅度，特别是HIP_PITCH和KNEE
        leg_lift_score = torch.zeros(self.num_envs, device=self.device)
        
        # 找到HIP_PITCH和KNEE关节
        hip_pitch_indices = []
        knee_indices = []
        
        for i, name in enumerate(self.dof_names):
            if 'HIP_PITCH' in name:
                hip_pitch_indices.append(i)
            elif 'KNEE' in name:
                knee_indices.append(i)
        
        if hip_pitch_indices and knee_indices:
            # 计算HIP_PITCH和KNEE的角速度（抬腿的关键指标）
            hip_pitch_vel = torch.abs(self.dof_vel[:, hip_pitch_indices])
            knee_vel = torch.abs(self.dof_vel[:, knee_indices])
            
            # 抬腿分数：HIP_PITCH和KNEE同时活动时得分更高
            hip_activity = torch.mean(hip_pitch_vel, dim=1)
            knee_activity = torch.mean(knee_vel, dim=1)
            
            # 协调的抬腿动作
            leg_lift_score = (hip_activity + knee_activity) / 2.0
            leg_lift_score = torch.clamp(leg_lift_score / 3.0, max=1.0)
        
        return leg_lift_score
    
    def _reward_yaw_roll_violation(self):
        """温和惩罚YAW和ROLL关节的过度运动"""
        yaw_roll_violation = torch.zeros(self.num_envs, device=self.device)
        
        # 找到YAW和ROLL关节
        yaw_roll_indices = []
        for i, name in enumerate(self.dof_names):
            if 'HIP_YAW' in name or 'HIP_ROLL' in name:
                yaw_roll_indices.append(i)
        
        if yaw_roll_indices:
            # 检查是否超出7度限制（0.122弧度），更宽容的限制
            yaw_roll_pos = torch.abs(self.dof_pos[:, yaw_roll_indices])
            violations = torch.clamp(yaw_roll_pos - 0.122, min=0.0)
            yaw_roll_violation = torch.sum(violations, dim=1)
        
        return -yaw_roll_violation  # 返回负值作为惩罚
    
    def _reward_yaw_violation(self):
        """严厉惩罚YAW关节过度运动（专门解决外扭问题）"""
        yaw_violation = torch.zeros(self.num_envs, device=self.device)
        
        # 找到YAW关节
        yaw_indices = []
        for i, name in enumerate(self.dof_names):
            if 'HIP_YAW' in name:
                yaw_indices.append(i)
        
        if yaw_indices:
            # 严格限制YAW角度，防止外扭
            yaw_pos = torch.abs(self.dof_pos[:, yaw_indices])
            # 更严格的限制：5度（0.087弧度）
            violations = torch.clamp(yaw_pos - 0.087, min=0.0)
            yaw_violation = torch.sum(violations, dim=1)
        
        return -yaw_violation  # 返回负值作为惩罚
    
    def _reward_yaw_stability(self):
        """奖励YAW关节保持接近0，防止外扭"""
        yaw_stability = torch.zeros(self.num_envs, device=self.device)
        
        # 找到YAW关节
        yaw_indices = []
        for i, name in enumerate(self.dof_names):
            if 'HIP_YAW' in name:
                yaw_indices.append(i)
        
        if yaw_indices:
            # 奖励YAW角度接近0
            yaw_pos = torch.abs(self.dof_pos[:, yaw_indices])
            # 使用高斯函数奖励接近0的角度
            stability_score = torch.exp(-50.0 * yaw_pos)  # 强烈奖励接近0
            yaw_stability = torch.mean(stability_score, dim=1)
        
        return yaw_stability
    
    def _reward_roll_violation(self):
        """单独惩罚ROLL关节过度运动"""
        roll_violation = torch.zeros(self.num_envs, device=self.device)
        
        # 找到ROLL关节
        roll_indices = []
        for i, name in enumerate(self.dof_names):
            if 'HIP_ROLL' in name:
                roll_indices.append(i)
        
        if roll_indices:
            # 适度限制ROLL角度
            roll_pos = torch.abs(self.dof_pos[:, roll_indices])
            violations = torch.clamp(roll_pos - 0.122, min=0.0)  # 7度限制
            roll_violation = torch.sum(violations, dim=1)
        
        return -roll_violation  # 返回负值作为惩罚

    def _reward_height_bonus(self):
        """额外高度奖励 - 鼓励机器人尽可能高"""
        # 基于当前高度给予奖励，高度越高奖励越大
        current_height = self.base_pos[:, 2]
        
        # 使用指数增长的高度奖励，鼓励达到更高位置
        height_bonus = torch.clamp(current_height - 0.3, min=0.0)  # 超过0.3m开始奖励
        height_bonus = height_bonus * height_bonus  # 平方增长，高度越高奖励越丰厚
        
        return height_bonus
    
    def _reward_upward_velocity(self):
        """奖励向上速度 - 鼓励积极向上运动"""
        # 直接奖励Z轴正向速度
        upward_vel = torch.clamp(self.base_lin_vel[:, 2], min=0.0)  # 只奖励向上速度
        
        # 对较快的向上速度给予更高奖励
        return upward_vel * 2.0  # 放大向上速度奖励
    
    def _reward_height_loss(self):
        """严厉惩罚高度下降"""
        # 惩罚Z轴负向速度（下降）
        downward_vel = torch.clamp(-self.base_lin_vel[:, 2], min=0.0)  # 只惩罚向下速度
        
        return -downward_vel  # 返回负值作为惩罚
    
    def _reward_downward_velocity(self):
        """严厉惩罚向下运动"""
        # 这是height_loss的补充，确保向下运动得到充分惩罚
        downward_vel = torch.clamp(-self.base_lin_vel[:, 2], min=0.0)
        
        # 对快速下降给予更严厉惩罚
        return -downward_vel * 3.0  # 放大向下速度惩罚
    
    def _reward_leg_symmetry(self):
        """奖励腿部对称性运动"""
        # 左右腿对称性检查
        fl_hip_pitch = self.dof_pos[:, 5]  # FL_HIP_PITCH
        fr_hip_pitch = self.dof_pos[:, 9]  # FR_HIP_PITCH
        rl_hip_pitch = self.dof_pos[:, 13] # RL_HIP_PITCH
        rr_hip_pitch = self.dof_pos[:, 17] # RR_HIP_PITCH
        
        # 左右对称性奖励（前腿之间，后腿之间）
        front_symmetry = torch.exp(-5.0 * torch.abs(fl_hip_pitch - fr_hip_pitch))
        rear_symmetry = torch.exp(-5.0 * torch.abs(rl_hip_pitch - rr_hip_pitch))
        
        # 前后协调性奖励（避免前腿后移、后腿前移）
        front_avg = (fl_hip_pitch + fr_hip_pitch) / 2.0
        rear_avg = (rl_hip_pitch + rr_hip_pitch) / 2.0
        
        # 鼓励前腿稍微向前（正值），后腿稍微向后（负值）
        coordination_reward = torch.exp(-3.0 * torch.abs(front_avg + rear_avg))  # 前后平衡
        
        return (front_symmetry + rear_symmetry + coordination_reward) / 3.0

    def _reward_hip_pitch_coordination(self):
        """奖励正确的HIP_PITCH协调运动模式"""
        fl_hip_pitch = self.dof_pos[:, 5]  # FL_HIP_PITCH
        fr_hip_pitch = self.dof_pos[:, 9]  # FR_HIP_PITCH
        rl_hip_pitch = self.dof_pos[:, 13] # RL_HIP_PITCH
        rr_hip_pitch = self.dof_pos[:, 17] # RR_HIP_PITCH
        
        # 检查是否有错误的运动模式
        # 前腿应该向前（正值）或保持中性，不应该大幅向后
        front_legs_good = torch.all(torch.stack([fl_hip_pitch > -0.3, fr_hip_pitch > -0.3], dim=1), dim=1)
        
        # 后腿应该向后（负值）或保持中性，不应该大幅向前
        rear_legs_good = torch.all(torch.stack([rl_hip_pitch < 0.3, rr_hip_pitch < 0.3], dim=1), dim=1)
        
        # 协调性奖励
        coordination_good = front_legs_good & rear_legs_good
        
        return coordination_good.float()

    def _reward_wrong_hip_movement(self):
        """惩罚错误的HIP_PITCH运动模式"""
        fl_hip_pitch = self.dof_pos[:, 5]  # FL_HIP_PITCH
        fr_hip_pitch = self.dof_pos[:, 9]  # FR_HIP_PITCH
        rl_hip_pitch = self.dof_pos[:, 13] # RL_HIP_PITCH
        rr_hip_pitch = self.dof_pos[:, 17] # RR_HIP_PITCH
        
        # 检测错误模式：前腿大幅向后，后腿大幅向前
        front_legs_backward = torch.sum((torch.stack([fl_hip_pitch, fr_hip_pitch], dim=1) < -0.5).float(), dim=1)
        rear_legs_forward = torch.sum((torch.stack([rl_hip_pitch, rr_hip_pitch], dim=1) > 0.5).float(), dim=1)
        
        # 返回错误模式的数量（将被负权重惩罚）
        wrong_patterns = front_legs_backward + rear_legs_forward
        
        return wrong_patterns
        # 分析轮子对左右墙面的压力
        left_wheel_indices = [self.feet_indices[0], self.feet_indices[2]]  # FL, BL 
        right_wheel_indices = [self.feet_indices[1], self.feet_indices[3]]  # FR, BR
        
        # 计算左右墙面的压力 (Y轴力)
        left_wall_pressure = torch.sum(torch.clamp(-self.contact_forces[:, left_wheel_indices, 1], min=0), dim=1)
        right_wall_pressure = torch.sum(torch.clamp(self.contact_forces[:, right_wheel_indices, 1], min=0), dim=1)
        
        # 要求两侧都要有压力，形成"卡住"效果
        balanced_pressure = torch.min(left_wall_pressure, right_wall_pressure)
        total_pressure = left_wall_pressure + right_wall_pressure
        
        # 平衡压力奖励：两侧压力越平衡越好
        pressure_balance = balanced_pressure / (total_pressure + 1e-6)
        
        # 总压力奖励：总压力要足够大
        pressure_magnitude = torch.clamp(total_pressure / 100.0, max=2.0)
        
        return pressure_balance * pressure_magnitude
    
    def _reward_leg_wall_coordination(self):
        """奖励腿部与墙面的协调接触"""
        # 检查四条腿是否都与墙面有良好接触
        wheel_forces = self.contact_forces[:, self.feet_indices, :]
        
        # 计算每个轮子的侧向接触力 (Y轴)
        lateral_forces = torch.abs(wheel_forces[:, :, 1])  # Y轴接触力
        
        # 要求所有四个轮子都有侧向接触
        min_contact_per_wheel = torch.clamp(lateral_forces - 5.0, min=0.0, max=20.0)  # 最小5N接触力
        
        # 协调性：四个轮子接触力的一致性
        contact_consistency = torch.min(min_contact_per_wheel, dim=1)[0]
        
        return contact_consistency / 20.0  # 归一化
    
    def _reward_wheel_leg_synergy(self):
        """奖励轮腿协同：腿部施压的同时轮子转动"""
        # 获取轮子转动速度
        wheel_indices = []
        for i, name in enumerate(self.dof_names):
            if 'WHEEL' in name:
                wheel_indices.append(i)
        
        if not wheel_indices:
            return torch.zeros(self.num_envs, device=self.device)
        
        wheel_rotation_speed = torch.mean(torch.abs(self.dof_vel[:, wheel_indices]), dim=1)
        
        # 获取墙面压力
        wheel_forces = self.contact_forces[:, self.feet_indices, :]
        total_wall_pressure = torch.sum(torch.abs(wheel_forces[:, :, 1]), dim=1)
        
        # 协同奖励：压力和转动速度同时存在时给高奖励
        pressure_score = torch.clamp(total_wall_pressure / 50.0, max=1.0)
        rotation_score = torch.clamp(wheel_rotation_speed / 5.0, max=1.0)
        
        # 协同效果：两者相乘，都有才能得到高分
        synergy_score = pressure_score * rotation_score
        
        return synergy_score
    
    def _reward_climbing_posture(self):
        """奖励正确的攀爬姿态：身体垂直，腿部外展"""
        # 身体应该保持垂直（Z轴向上）
        body_upright = self.projected_gravity[:, 2]  # 接近1表示直立
        
        # 腿部应该适度外展以贴墙
        leg_spread_score = torch.zeros(self.num_envs, device=self.device)
        
        # 检查HIP_ROLL角度，应该有适度外展
        hip_roll_indices = []
        for i, name in enumerate(self.dof_names):
            if 'HIP_ROLL' in name:
                hip_roll_indices.append(i)
        
        if hip_roll_indices:
            # 左腿应该向左外展(负角度)，右腿应该向右外展(正角度)
            left_legs = [hip_roll_indices[0], hip_roll_indices[2]]  # FL, BL
            right_legs = [hip_roll_indices[1], hip_roll_indices[3]]  # FR, BR
            
            left_spread = torch.mean(-self.dof_pos[:, left_legs], dim=1)  # 负角度表示向左
            right_spread = torch.mean(self.dof_pos[:, right_legs], dim=1)  # 正角度表示向右
            
            # 理想外展角度约5-15度(0.087-0.26弧度)
            left_spread_score = torch.clamp((left_spread - 0.087) / 0.173, min=0.0, max=1.0)
            right_spread_score = torch.clamp((right_spread - 0.087) / 0.173, min=0.0, max=1.0)
            
            leg_spread_score = (left_spread_score + right_spread_score) / 2.0
        
        # 综合姿态评分
        posture_score = (body_upright + leg_spread_score) / 2.0
        return torch.clamp(posture_score, min=0.0, max=1.0)
    
    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

    def post_physics_step(self):
        """
        重写父类的这个函数，用正确的索引来更新机器人的状态。
        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # --- 核心修正：只用机器人的索引来更新状态 ---
        robot_indices = torch.arange(1, 2 * self.num_envs, 2, device=self.device, dtype=torch.long)
        self.base_quat[:] = self.root_states[robot_indices, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[robot_indices, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[robot_indices, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # 🔧 **激进强制方案：每个物理步后强制YAW和ROLL关节位置为0**
        yaw_indices = [0, 5, 10, 15]    # YAW关节索引
        roll_indices = [1, 6, 11, 16]   # ROLL关节索引
        hip_pitch_indices = [2, 7, 12, 17]  # HIP_PITCH关节索引
        
        # 强制将YAW和ROLL关节位置拉回0
        self.dof_pos[:, yaw_indices] = 0.0
        self.dof_pos[:, roll_indices] = 0.0
        
        # **新增：强制限制HIP_PITCH关节角度范围** - 超严格的±5度限制
        self.dof_pos[:, hip_pitch_indices] = torch.clamp(self.dof_pos[:, hip_pitch_indices], -0.087, 0.087)
        
        print(f"DEBUG: YAW位置被强制为0: {self.dof_pos[0, yaw_indices]}")
        print(f"DEBUG: ROLL位置被强制为0: {self.dof_pos[0, roll_indices]}")
        print(f"DEBUG: HIP_PITCH位置被限制到±5度: {self.dof_pos[0, hip_pitch_indices]}")
        
        # 同时强制速度为0，防止惯性
        self.dof_vel[:, yaw_indices] = 0.0
        self.dof_vel[:, roll_indices] = 0.0

        # --- 后续逻辑和父类保持一致 ---
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        
    def check_termination(self):
        """ 检查是否需要重置环境 """
        self.reset_buf = torch.zeros_like(self.base_pos[:, 0], dtype=torch.bool)
        self.time_out_buf = torch.zeros_like(self.base_pos[:, 0], dtype=torch.bool)
        
        # 基于高度的终止条件：如果机器人掉得太低（低于地面-0.5m，给予更多空间）
        fallen_over = self.base_pos[:, 2] < -0.5
        
        # 基于距离的终止条件：如果机器人偏离裂缝太远（X轴偏移超过3.0m，增加允许范围）
        too_far_lateral = torch.abs(self.base_pos[:, 0]) > 3.0
        
        # 基于倾覆的终止条件：如果机器人严重翻转（增加角度容忍度）
        roll_pitch_too_large = torch.abs(self.projected_gravity[:, 0]) > 0.9  # roll > ~64度
        roll_pitch_too_large |= torch.abs(self.projected_gravity[:, 1]) > 0.9  # pitch > ~64度
        
        # 时间超限
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        
        # 综合终止条件 - 主要依赖时间超限，其他条件较宽松
        self.reset_buf = fallen_over | too_far_lateral | roll_pitch_too_large | self.time_out_buf
    

    
    def reset_idx(self, env_ids):
        """重置指定环境"""
        if len(env_ids) == 0:
            return
            
        # 调用父类的重置逻辑
        super().reset_idx(env_ids)
        self.reset_buf |= self.time_out_buf
        
    def _compute_torques(self, actions):
        """ 计算扭矩，对轮子使用力矩控制，对其他关节使用位置控制 """
        actions_scaled = actions * self.cfg.control.action_scale
        
        # 🚫 **彻底阻止YAW和ROLL外扩** - 经过调试确认的正确索引！
        # 真实关节索引（来自调试脚本）：
        # YAW: [0, 5, 10, 15] = [HIP_YAW_BL, HIP_YAW_BR, HIP_YAW_FL, HIP_YAW_FR]
        # ROLL: [1, 6, 11, 16] = [HIP_ROLL_BL, HIP_ROLL_BR, HIP_ROLL_FL, HIP_ROLL_FR]
        # WHEEL: [4, 9, 14, 19] = [WHEEL_BL, WHEEL_BR, WHEEL_FL, WHEEL_FR]
        
        yaw_indices = [0, 5, 10, 15]    # 确认正确
        roll_indices = [1, 6, 11, 16]   # 确认正确
        wheel_indices = [4, 9, 14, 19]  # 确认正确
        hip_pitch_indices = [2, 7, 12, 17]  # HIP_PITCH关节索引
        
        # **史上最严格的动作限制** - 绝对禁止YAW和ROLL移动
        for idx in yaw_indices:
            actions_scaled[:, idx] = 0.0  # YAW完全禁止
        for idx in roll_indices:
            actions_scaled[:, idx] = 0.0  # ROLL完全禁止
        
        # **新增：超严格限制HIP_PITCH角度，防止八字腿** - ±5度限制
        for idx in hip_pitch_indices:
            actions_scaled[:, idx] = torch.clamp(actions_scaled[:, idx], -0.087, 0.087)  # 限制在±5度
        
        # 轮子也严格限制，避免侧向旋转
        for idx in wheel_indices:
            actions_scaled[:, idx] = torch.clamp(actions_scaled[:, idx], -0.1, 1.5)  # 主要正向
        
        # 🔧 **激进方案：直接强制YAW和ROLL关节位置为0** 
        # 在计算扭矩之前，先强制设置关节位置
        self.dof_pos[:, yaw_indices] = 0.0  # 强制YAW位置为0
        self.dof_pos[:, roll_indices] = 0.0  # 强制ROLL位置为0
        
        # print(f"DEBUG: YAW动作被设为0: {actions_scaled[0, yaw_indices]}")  # 调试输出
        # print(f"DEBUG: ROLL动作被设为0: {actions_scaled[0, roll_indices]}")  # 调试输出
        # print(f"DEBUG: YAW位置被强制为0: {self.dof_pos[0, yaw_indices]}")  # 调试输出
        # print(f"DEBUG: ROLL位置被强制为0: {self.dof_pos[0, roll_indices]}")  # 调试输出
        
        # 位置控制的PD扭矩
        torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        
        # **最严格的扭矩级别限制** - 无论PD控制器计算什么，强制YAW和ROLL扭矩为0
        for idx in yaw_indices:
            torques[:, idx] = 0.0  # YAW扭矩完全禁止
        for idx in roll_indices:
            torques[:, idx] = 0.0  # ROLL扭矩完全禁止
        
        # print(f"DEBUG: YAW扭矩被设为0: {torques[0, yaw_indices]}")  # 调试输出
        # print(f"DEBUG: ROLL扭矩被设为0: {torques[0, roll_indices]}")  # 调试输出
        
        # 对轮子使用直接力矩控制
        wheel_indices_tensor = torch.tensor(wheel_indices, device=self.device)
        wheel_torque_scale = 200.0
        torques[:, wheel_indices_tensor] = actions_scaled[:, wheel_indices_tensor] * wheel_torque_scale
            
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # ==================== 新增严格关节控制奖励函数 ====================
    
    def _reward_wall_press_climbing(self):
        """奖励利用压力传感器的卡墙攀爬技术"""
        wall_press_reward = torch.zeros(self.num_envs, device=self.device)
        
        # 简化的墙面压力检测 - 基于腿部关节角度和接触情况
        # 检查膝关节是否处于合适的墙面接触角度
        fl_knee = self.dof_pos[:, 6]   # FL_KNEE
        fr_knee = self.dof_pos[:, 10]  # FR_KNEE
        rl_knee = self.dof_pos[:, 14]  # RL_KNEE
        rr_knee = self.dof_pos[:, 18]  # RR_KNEE
        
        # 奖励膝关节适度弯曲（攀爬姿态）
        knee_angles = torch.stack([fl_knee, fr_knee, rl_knee, rr_knee], dim=1)
        optimal_angles = torch.abs(knee_angles + 1.2)  # 约70度弯曲
        wall_press_reward = torch.exp(-optimal_angles).mean(dim=1)
        
        return wall_press_reward

    def _reward_hip_pitch_limit(self):
        """严厉惩罚HIP_PITCH角度过大，防止八字腿外扩"""
        hip_pitch_indices = [2, 7, 12, 17]  # HIP_PITCH关节索引
        hip_pitch_positions = self.dof_pos[:, hip_pitch_indices]
        
        # 超严格限制HIP_PITCH在±5度（±0.087弧度）内
        angle_limit = 0.087  # 5度
        violations = torch.clamp(torch.abs(hip_pitch_positions) - angle_limit, min=0.0)
        total_violation = torch.sum(violations, dim=1)
        
        # 返回负值作为严厉惩罚
        return -total_violation * 20.0  # 进一步放大惩罚力度

    def _reward_wheel_yaw_control(self):
        """专门控制轮子YAW关节，防止轮子往外扩"""
        # WHEEL关节索引: 2
        wheel_yaw = self.dof_pos[:, 2]  # WHEEL关节
        
        # 强制轮子YAW保持在零位，严厉惩罚任何偏离
        wheel_yaw_stability = torch.exp(-20.0 * torch.abs(wheel_yaw))
        
        return wheel_yaw_stability
