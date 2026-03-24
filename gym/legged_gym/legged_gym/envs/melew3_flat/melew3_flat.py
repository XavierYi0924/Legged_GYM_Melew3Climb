from legged_gym.envs.base.legged_robot import LeggedRobot
from .melew3_flat_config import Melew3FlatCfg
from isaacgym.torch_utils import torch_rand_float
from isaacgym import gymtorch
import torch

class Melew3Flat(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        cfg.rewards.scales.tracking_lin_vel = 0.0
        cfg.rewards.scales.collision = -1.0
        
        # 🔥 新策略：分段奖励（基础线性+高处指数）
        cfg.rewards.scales.climb_high = 3.5  # 提高权重，让高度奖励更明显
        
        cfg.rewards.scales.orientation = -5.5  # 恢复原值，只惩罚roll/pitch
        cfg.rewards.scales.torques = -0.0008  # 适度惩罚
        cfg.rewards.scales.dof_acc = 0.0
        cfg.rewards.scales.dof_vel = 0.0  # 让轮子自由转动
        cfg.rewards.scales.dof_pos_limits = -0.6
        cfg.rewards.scales.dof_vel_limits = -0.004
        cfg.rewards.scales.torque_limits = -0.0008
        # cfg.rewards.scales.center = 0.0001
        cfg.rewards.scales.yaw = 0.0  # 🔥 禁用yaw位置惩罚，改用ang_vel_z
        cfg.rewards.scales.ang_vel_z = -20.0  # 🔥 新增：惩罚绕Z轴的角速度（旋转速度）
        cfg.rewards.scales.base_height = 0.5
        cfg.rewards.scales.leg_symmetry = -2.0
        cfg.rewards.scales.z_velocity_positive = 0.05  # 🔥🔥 大幅降低以强制慢速上升（从0.1降至0.05）
        cfg.rewards.scales.movement_penalty = -100.0  # 降低惩罚（从-100）
        cfg.rewards.scales.horizontal_velocity = -6.0  # 降低惩罚（从-4.0）
        cfg.rewards.scales.static_legs_penalty = -2.0  # 🔥 大幅降低，鼓励跳跃动作（从-3.0降至-1.0）
        cfg.rewards.scales.continuous_contact = -2.0  # 微量惩罚，鼓励四腿同时接触
        cfg.rewards.scales.bl_contact = 1.0  # 🔥 新增：专门奖励BL接触，惩罚BL悬空
        cfg.rewards.scales.jumping_motion = 0.0 # 🔥 新增：鼓励跳跃式腿部动作
        cfg.rewards.scales.termination = -200.0  # 降低惩罚（从-50），避免过于保守

        cfg.rewards.scales.tracking_ang_vel = 0.0
        cfg.rewards.scales.ang_vel_xy = 0.0
        cfg.rewards.scales.feet_stumble = 0.0
        cfg.rewards.scales.stand_still = 0.0
        cfg.rewards.scales.lin_vel_z = 0.0
        cfg.rewards.scales.joint_regularization = 0.0
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.contact_history_length = 15
        self.contact_history = torch.zeros(self.num_envs, len(self.feet_indices), self.contact_history_length, 
                                           dtype=torch.bool, device=self.device)

        # 用于检测轮子角速度异常上升的历史记录
        self.wheel_vel_history = torch.zeros(self.num_envs, 4, 10, device=self.device)  # 记录最近10步的轮子角速度
        
        # 🔥 新增：用于接触力质量检测的历史缓冲区
        self.force_history = torch.zeros(self.num_envs, 4, 10, device=self.device)  # 记录最近10步的接触力
        self.height_short_history = torch.zeros(self.num_envs, 20, device=self.device)  # 记录最近20步的高度
        self.last_wheel_forces = torch.zeros(self.num_envs, 4, device=self.device)  # 上一步的轮子接触力

        self.leg_dof_indices = []
        for name in self.dof_names:
            if 'WHEEL' not in name:
                self.leg_dof_indices.append(self.dof_names.index(name))
        self.leg_dof_indices = torch.tensor(self.leg_dof_indices, device=self.device, dtype=torch.long)
        
        self.front_leg_indices = {'hip_pitch': [], 'knee': []}
        self.back_leg_indices = {'hip_pitch': [], 'knee': []}
        
        for i, name in enumerate(self.dof_names):
            if 'HIP_PITCH_FL' in name or 'HIP_PITCH_FR' in name:  # 前腿HIP_PITCH
                self.front_leg_indices['hip_pitch'].append(i)
            elif 'HIP_PITCH_BL' in name or 'HIP_PITCH_BR' in name:  # 后腿HIP_PITCH
                self.back_leg_indices['hip_pitch'].append(i)
            elif 'KNEE_FL' in name or 'KNEE_FR' in name:  # 前腿KNEE
                self.front_leg_indices['knee'].append(i)
            elif 'KNEE_BL' in name or 'KNEE_BR' in name:  # 后腿KNEE
                self.back_leg_indices['knee'].append(i)
        
        self.front_leg_indices['hip_pitch'] = torch.tensor(self.front_leg_indices['hip_pitch'], device=self.device, dtype=torch.long)
        self.front_leg_indices['knee'] = torch.tensor(self.front_leg_indices['knee'], device=self.device, dtype=torch.long)
        self.back_leg_indices['hip_pitch'] = torch.tensor(self.back_leg_indices['hip_pitch'], device=self.device, dtype=torch.long)
        self.back_leg_indices['knee'] = torch.tensor(self.back_leg_indices['knee'], device=self.device, dtype=torch.long)
        
        # 在初始化时创建单个连续的日志文件
        self.log_counter = 0
        if not headless:
            try:
                import datetime
                import os
                
                # 创建logs目录（如果不存在）
                if not os.path.exists("logs"):
                    os.makedirs("logs")
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = f"logs/wheel_velocities_{timestamp}.log"
                self.log_file = open(log_filename, "w")
                self.log_file.write("time_s,vel_BL,vel_BR,vel_FL,vel_FR\n")
                print(f"连续日志文件 '{log_filename}' 已创建。")
            except IOError:
                print("错误：无法创建日志文件！")
                self.log_file = None
        else:
            self.log_file = None
        
        # 添加轮子关节索引
        self.wheel_indices = []
        for i, name in enumerate(self.dof_names):
            if 'WHEEL' in name:
                self.wheel_indices.append(i)
        self.wheel_indices = torch.tensor(self.wheel_indices, device=self.device, dtype=torch.long)
        print(f"\n=== 轮子索引调试信息 ===")
        print(f"总DOF数: {self.num_dof}")
        print(f"总Actions数: {self.num_actions}")
        print(f"轮子索引: {self.wheel_indices.cpu().numpy()}")
        print(f"对应关节: {[self.dof_names[i] for i in self.wheel_indices.cpu().numpy()]}")
        print(f"所有DOF名称: {self.dof_names}")
        print(f"\n轮子力矩限制:")
        for idx in self.wheel_indices:
            print(f"  {self.dof_names[idx]}: {self.torque_limits[idx].item()} Nm")
        print(f"=========================\n")
    
    def _reset_dofs(self, env_ids):
        """
        覆盖父类方法：重置DOF状态
        - 腿部DOF：重置为默认位置（保持原逻辑）
        - 轮子DOF：完全不重置，保持原来的位置和速度
        
        这样轮子可以独立运行，不受环境重置影响。
        """
        # 只重置腿部DOF，不重置轮子
        # 创建一个临时张量来存储新的DOF位置
        temp_dof_pos = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        
        # 将轮子位置保持不变
        for wheel_idx in self.wheel_indices:
            temp_dof_pos[:, wheel_idx] = self.dof_pos[env_ids, wheel_idx]
        
        # 应用新的位置
        self.dof_pos[env_ids] = temp_dof_pos
        
        # 只重置腿部速度，轮子速度保持不变
        temp_dof_vel = torch.zeros_like(self.dof_vel[env_ids])
        for wheel_idx in self.wheel_indices:
            temp_dof_vel[:, wheel_idx] = self.dof_vel[env_ids, wheel_idx]
        
        self.dof_vel[env_ids] = temp_dof_vel
        
        # 同步到物理引擎
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def _compute_torques(self, actions):
        """
        力矩计算方法
        - 在轮子速度控制模式下，轮子使用独立的速度控制，腿部继续执行强化学习策略
        - 前轮和后轮需要反向转动（双面墙爬行需求）
        - 在正常模式下，使用训练好的策略
        """
        # 首先计算腿部力矩（使用强化学习策略）
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        
        if control_type == "P":
            # 对所有DOF计算PD力矩
            torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
            
            # 🔥 关键改动：对轮子不应用位置控制（因为轮子stiffness=0，p_gain=0本应无效，但为了保险还是清零轮子的位置项）
            # 只保留轮子的阻尼项（如果需要的话）
            for wheel_idx in self.wheel_indices:
                torques[:, wheel_idx] = -self.d_gains[wheel_idx] * self.dof_vel[:, wheel_idx]
        else:
            # 如果不是P控制，调用父类方法
            torques = super()._compute_torques(actions)
        
        # 如果启用了轮子速度控制，则覆盖轮子的力矩
        if hasattr(self.cfg.control, 'wheel_velocity_control') and self.cfg.control.wheel_velocity_control:
            # 从配置读取轮子参数
            target_wheel_vel = self.cfg.control.wheel_target_velocity  # rad/s
            wheel_kp = self.cfg.control.wheel_velocity_kp
            wheel_kd = self.cfg.control.wheel_velocity_kd
            max_wheel_torque = self.cfg.control.wheel_max_torque
            
            # 获取当前轮子速度
            current_wheel_vel = self.dof_vel[:, self.wheel_indices]
            
            # 🔥 双面墙爬行：前轮和后轮需要反向转动
            # 轮子索引顺序: [WHEEL_BL, WHEEL_BR, WHEEL_FL, WHEEL_FR]
            # 前轮(FL=2, FR=3): 正向转动，推动身体上升
            # 后轮(BL=0, BR=1): 反向转动，拉动身体上升
            wheel_directions = torch.ones(len(self.wheel_indices), device=self.device)
            
            # 找到前后轮的位置
            for i, wheel_idx in enumerate(self.wheel_indices):
                wheel_name = self.dof_names[wheel_idx]
                if 'WHEEL_BL' in wheel_name or 'WHEEL_BR' in wheel_name:
                    # 后轮反向转动
                    wheel_directions[i] = -1.0
                elif 'WHEEL_FL' in wheel_name or 'WHEEL_FR' in wheel_name:
                    # 前轮正向转动
                    wheel_directions[i] = 1.0
            
            # 计算目标速度（考虑方向）
            target_wheel_vel_vector = target_wheel_vel * wheel_directions.unsqueeze(0)  # (1, 4) -> (num_envs, 4)
            vel_error = target_wheel_vel_vector - current_wheel_vel
            
            # 🔥 简化的控制器：只根据速度误差调整力矩
            # 如果速度太低，增加力矩；如果速度太高，减少力矩
            # 使用比例控制（P控制）而不是P+D，避免震荡
            wheel_torques = wheel_kp * vel_error
            
            # 限制轮子力矩
            wheel_torques = torch.clamp(wheel_torques, -max_wheel_torque, max_wheel_torque)
            
            # 设置轮子力矩，腿部保持策略输出
            for i, wheel_idx in enumerate(self.wheel_indices):
                torques[:, wheel_idx] = wheel_torques[:, i]
        
        # 应用力矩限制
        torques = torch.clamp(torques, -self.torque_limits, self.torque_limits)
        
        return torques
    
    def _reset_root_states(self, env_ids):
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            
            x_random = torch_rand_float(-7.5, 7.5, (len(env_ids), 1), device=self.device) 
            y_random = torch_rand_float(-0.05, 0.05, (len(env_ids), 1), device=self.device) 

            self.root_states[env_ids, 0:1] += x_random  
            self.root_states[env_ids, 1:2] += y_random  
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 2] = self.cfg.init_state.pos[2]
        # if len(env_ids) > 0:
        #     print(f"发送给物理引擎前的Z高度: {self.root_states[env_ids[0], 2].item()}")
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # if len(env_ids) > 0:
        #     print(f"从物理引擎刷新后的Z高度: {self.root_states[env_ids[0], 2].item()}")

    def post_physics_step(self):
        # 首先，必须调用父类的原始方法，确保所有基础计算都已完成
        super().post_physics_step()

        # # --- 检测并消除轮子角速度的均匀上升趋势（已禁用，让轮子自由转动）---
        # wheel_indices = [4, 9, 14, 19]  # WHEEL_FL, WHEEL_FR, WHEEL_BL, WHEEL_BR
        # current_wheel_vels = self.dof_vel[:, wheel_indices]
        # 
        # # 更新轮子角速度历史
        # self.wheel_vel_history = torch.roll(self.wheel_vel_history, shifts=-1, dims=-1)
        # self.wheel_vel_history[:, :, -1] = current_wheel_vels
        # 
        # # 检测持续单调上升的轮子（连续5步都在增长）
        # for i in range(4):
        #     recent_history = self.wheel_vel_history[:, i, -5:]
        #     if recent_history.shape[-1] >= 5:
        #         diffs = recent_history[:, 1:] - recent_history[:, :-1]
        #         is_monotonic_rising = torch.all(diffs > 0.1, dim=1)
        #         diff_std = torch.std(diffs, dim=1)
        #         is_uniform_slope = diff_std < 0.5
        #         anomaly_envs = is_monotonic_rising & is_uniform_slope
        #         if torch.any(anomaly_envs):
        #             self.dof_vel[anomaly_envs, wheel_indices[i]] = 0.0

        # --- 更新我们的接触历史缓冲区 ---
        # 1. 获取当前这一步的接触状态 (True/False)
        current_contact = (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0)

        # 2. "滚动"历史缓冲区，丢弃最旧的数据
        self.contact_history = torch.roll(self.contact_history, shifts=-1, dims=-1)

        # 3. 将最新的接触状态存入缓冲区
        self.contact_history[..., -1] = current_contact

    def compute_observations(self):
        base_pos_y = self.root_states[:, 1].unsqueeze(1) 
        self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                self.projected_gravity,
                                self.commands[:, :3] * self.commands_scale,
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.actions,
                                base_pos_y 
                                ),dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    
    def _reward_tracking_lin_vel(self):
        """tracking velocity: f((pz - vz^ref)^2 + px^2, 0.01)"""
        vz_ref = 0.0  
        pz_error = (self.base_lin_vel[:, 2] - vz_ref) ** 2
        px_error = self.base_lin_vel[:, 0] ** 2
        total_error = pz_error + px_error
        return torch.exp(-total_error / 0.01)
    
    def _reward_collision(self):
        """collision: ∑j∈P(|fj^contact| > 0.1)"""
        body_contacts = 0
        for i, contact_name in enumerate(self.cfg.asset.penalize_contacts_on):
            contact_forces = torch.norm(self.contact_forces[:, i], dim=-1)
            body_contacts += (contact_forces > 0.1).float()
        return body_contacts
    
    def _reward_climb_high(self):
        """分段奖励：1m以下无奖励，1-1.5m指数增长，1.5m以上保持不变"""
        current_height = self.root_states[:, 2] 
        threshold_low = 1.0   # 低阈值：1m以下无奖励
        threshold_high = 1.5  # 高阈值：1.5m以上奖励不变
        
        # 阶段1: 0-1.0m 无奖励
        # 阶段2: 1.0-1.5m 指数奖励
        height_in_range = torch.clamp(current_height - threshold_low, min=0.0, max=threshold_high - threshold_low)
        exponential_reward = torch.exp(3.0 * height_in_range) - 1.0
        
        # 阶段3: 1.5m以上 保持最大值不变
        # 计算最大奖励值 (使用numpy计算常量，避免每次都计算)
        import math
        max_reward_value = math.exp(3.0 * 0.5) - 1.0  # ≈ 3.48
        reward = torch.where(
            current_height >= threshold_high,
            torch.full_like(exponential_reward, max_reward_value),
            exponential_reward
        )
        
        return reward
    
    def _reward_termination(self):
        """提前终止惩罚:鼓励机器人维持长episode"""
        # 对所有非正常终止(摔倒/停滞)给予惩罚
        return self.reset_buf.float()
    
    def _reward_orientation(self):
        """orientation: 惩罚水平倾斜(roll/pitch)，不惩罚yaw旋转"""
        # 只惩罚水平倾斜（roll和pitch）
        tilt_magnitude = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        tilt_penalty = torch.exp(tilt_magnitude * 5.0) - 1.0
        
        # 移除yaw惩罚，改为单独的ang_vel_z惩罚
        return tilt_penalty
    
    def _reward_torques(self):
        """rated torques: τ/τrate - 排除轮子DOF"""
        max_torque = 40.0
        normalized_torques = torch.abs(self.torques) / max_torque
        # 排除轮子DOF，只计算腿部的力矩惩罚
        normalized_torques[:, self.wheel_indices] = 0.0
        return torch.sum(normalized_torques, dim=1)
    
    def _reward_dof_acc(self):
        """dof acc: ||θ̈||^2 - 排除轮子DOF"""
        dof_acc = (self.last_dof_vel - self.dof_vel) / (self.dt ** 2)
        # 排除轮子DOF，只计算腿部的加速度惩罚
        dof_acc[:, self.wheel_indices] = 0.0
        return torch.sum(torch.square(dof_acc), dim=1)
    
    def _reward_dof_vel(self):
        """dof vel: ||θ̇||^2"""
        leg_joint_vels = self.dof_vel[:, self.leg_dof_indices]
        return torch.sum(torch.square(leg_joint_vels), dim=1)
    
    def _reward_dof_pos_limits(self):
        """dof pos limits: ||θ - clip(θ, 0.8θmin, 0.8θmax)|| - 排除轮子DOF"""
        dof_pos_clamped = torch.clamp(self.dof_pos, 
                                     0.8 * self.dof_pos_limits[:, 0], 
                                     0.8 * self.dof_pos_limits[:, 1])
        out_of_limits = torch.square(self.dof_pos - dof_pos_clamped)
        # 排除轮子DOF，只计算腿部的位置限制惩罚
        out_of_limits[:, self.wheel_indices] = 0.0
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_vel_limits(self):
        """dof vel limits: ||θ̇ - clip(θ̇, 0.6θ̇min, 0.6θ̇max)|| - 排除轮子DOF"""
        dof_vel_limits = 10.0
        dof_vel_clamped = torch.clamp(self.dof_vel, -0.6 * dof_vel_limits, 0.6 * dof_vel_limits)
        out_of_limits = torch.square(self.dof_vel - dof_vel_clamped)
        # 排除轮子DOF，只计算腿部的速度限制惩罚
        out_of_limits[:, self.wheel_indices] = 0.0
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_torque_limits(self):
        """torque limits: ||τ - clip(τ, -0.8τmax, 0.8τmax)|| - 排除轮子DOF"""
        max_torque = 40.0
        torque_clamped = torch.clamp(self.torques, -0.8 * max_torque, 0.8 * max_torque)
        out_of_limits = torch.square(self.torques - torque_clamped)
        # 排除轮子DOF，只计算腿部的力矩限制惩罚
        out_of_limits[:, self.wheel_indices] = 0.0
        return torch.sum(out_of_limits, dim=1)
    
    # def _reward_center(self):
    #     """center: py^2 - 保持在通道中心"""
    #     return torch.square(self.root_states[:, 0])
    
    def _reward_yaw(self):
        """yaw: θ_yaw^2 - 考虑基座90度旋转"""
        qw, qx, qy, qz = self.root_states[:, 3], self.root_states[:, 4], self.root_states[:, 5], self.root_states[:, 6]
        yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        target_yaw = 1.5708  
        yaw_error = torch.square(yaw - target_yaw)
        return yaw_error
    
    def _reward_ang_vel_z(self):
        """惩罚绕Z轴的角速度（旋转速度），防止body旋转"""
        # base_ang_vel[:, 2] 是Z轴角速度
        return torch.square(self.base_ang_vel[:, 2])
    
    # def _reward_collar_angles(self):
    #     """collar angles: ∑i∈D |θi^collar|^2"""
    #     collar_indices = []
    #     for i, name in enumerate(self.dof_names):
    #         if 'HIP_YAW' in name:
    #             collar_indices.append(i)
    #     if collar_indices:
    #         collar_angles = self.dof_pos[:, collar_indices]
    #         return torch.sum(torch.square(collar_angles), dim=1)
    #     else:
    #         return torch.zeros(self.num_envs, device=self.device)
    
    # def _reward_low_foot(self):
    #     """low foot: ∑i∈D log(zi^foot/0.2)"""
    #     base_height = self.root_states[:, 2]
    #     estimated_foot_height = base_height - 0.35
    #     foot_height_threshold = 0.2
    #     safe_heights = torch.clamp(estimated_foot_height, min=0.01)  
    #     log_reward = torch.log(safe_heights / foot_height_threshold)
    #     return log_reward
    
    def _reward_base_height(self):
        """base height: (pz - 0.6) + 9.0 * min(pz - 0.6, 0)"""
        base_height = self.root_states[:, 2]
        target_height = 0.4  
        height_diff = base_height - target_height
        positive_reward = height_diff
        negative_penalty = 9.0 * torch.clamp(height_diff, max=0.0)
        return positive_reward + negative_penalty
    
    def _reward_static_legs_penalty(self):
        """
        巨额惩罚【未能】四轮同时接触墙壁。
        只要不是所有轮子都产生超过阈值的接触力，就触发惩罚。
        """
        # (num_envs, num_feet, 3) -> (num_envs, num_feet)
        wheel_contact_forces = self.contact_forces[:, self.feet_indices, :]
        horizontal_force_magnitude = torch.norm(wheel_contact_forces[:, :, :2], dim=2)
        min_contact_force_per_wheel = 5.0  # 5N
        is_wheel_in_contact = horizontal_force_magnitude > min_contact_force_per_wheel
        all_four_wheels_in_contact = torch.all(is_wheel_in_contact, dim=1)
        penalty_signal = (~all_four_wheels_in_contact).float()

        return penalty_signal


    def _reward_leg_symmetry(self):
        """前后腿姿态平衡奖励：只在极端不对称时才惩罚"""
        front_hip_pitch = self.dof_pos[:, self.front_leg_indices['hip_pitch']]  # (num_envs, 2)
        back_hip_pitch = self.dof_pos[:, self.back_leg_indices['hip_pitch']]    # (num_envs, 2)
        front_knee = self.dof_pos[:, self.front_leg_indices['knee']]            # (num_envs, 2)
        back_knee = self.dof_pos[:, self.back_leg_indices['knee']]              # (num_envs, 2)
        front_hip_avg = torch.mean(front_hip_pitch, dim=1)  # (num_envs,)
        back_hip_avg = torch.mean(back_hip_pitch, dim=1)    # (num_envs,)
        front_knee_avg = torch.mean(front_knee, dim=1)      # (num_envs,)
        back_knee_avg = torch.mean(back_knee, dim=1)        # (num_envs,)
        hip_diff = torch.abs(front_hip_avg - back_hip_avg)
        knee_diff = torch.abs(front_knee_avg - back_knee_avg)
        hip_threshold = 1.0  # 57度
        knee_threshold = 1.2  # 69度
        hip_penalty = torch.clamp(hip_diff - hip_threshold, min=0.0)
        knee_penalty = torch.clamp(knee_diff - knee_threshold, min=0.0)
        return hip_penalty + knee_penalty
    
    # def _reward_leg_length_consistency(self):
    #     """前后腿长度一致性奖励：只在极端不一致时才惩罚"""
    #     link1_length = 0.2
    #     link2_length = 0.2
    #     front_hip_pitch = self.dof_pos[:, self.front_leg_indices['hip_pitch']]
    #     back_hip_pitch = self.dof_pos[:, self.back_leg_indices['hip_pitch']]
    #     front_knee = self.dof_pos[:, self.front_leg_indices['knee']]
    #     back_knee = self.dof_pos[:, self.back_leg_indices['knee']]
    #     front_leg_length = link1_length * torch.cos(front_hip_pitch) + link2_length * torch.cos(front_hip_pitch + front_knee)
    #     back_leg_length = link1_length * torch.cos(back_hip_pitch) + link2_length * torch.cos(back_hip_pitch + back_knee)
    #     all_leg_lengths = torch.cat([front_leg_length, back_leg_length], dim=1)  # (num_envs, 4)
    #     leg_length_std = torch.std(all_leg_lengths, dim=1)  # (num_envs,)
    #     std_threshold = 0.15 
    #     std_penalty = torch.clamp(leg_length_std - std_threshold, min=0.0)
    #     return -std_penalty

    # def _reward_leg_straight_penalty(self):
    #     """严厉惩罚腿部伸直：一条腿伸直就严重惩罚"""
    #     front_hip_pitch = self.dof_pos[:, self.front_leg_indices['hip_pitch']]
    #     back_hip_pitch = self.dof_pos[:, self.back_leg_indices['hip_pitch']]
    #     front_knee = self.dof_pos[:, self.front_leg_indices['knee']]
    #     back_knee = self.dof_pos[:, self.back_leg_indices['knee']]              
    #     front_hip_straight = torch.abs(front_hip_pitch) < 0.2  
    #     back_hip_straight = torch.abs(back_hip_pitch) < 0.2
    #     front_knee_straight = torch.abs(front_knee) < 0.2      
    #     back_knee_straight = torch.abs(back_knee) < 0.2
    #     front_legs_straight = (front_hip_straight & front_knee_straight).float()  
    #     back_legs_straight = (back_hip_straight & back_knee_straight).float()     
    #     total_straight_legs = torch.sum(front_legs_straight, dim=1) + torch.sum(back_legs_straight, dim=1)  # (num_envs,)
    #     penalty = torch.where(total_straight_legs >= 1, 
    #                          torch.tensor(-10.0, device=self.device),  # 任何腿伸直都严重惩罚
    #                          torch.tensor(0.0, device=self.device))     # 无伸直腿：无惩罚
    #     return penalty

    def _reward_z_velocity_positive(self):
        """
        🔥🔥 追踪目标z速度：奖励接近目标速度，惩罚偏离
        目标：稳定慢速爬升，避免快速冲刺或频繁停顿
        """
        z_velocity = self.base_lin_vel[:, 2]  # (num_envs,)
        target_z_velocity = 0.15  # 🎯 目标速度：0.15 m/s（慢速稳定）
        
        # 计算与目标速度的偏差
        velocity_error = torch.abs(z_velocity - target_z_velocity)
        
        # 使用指数函数：越接近目标，奖励越高
        # 当误差=0时，reward=1.0；误差越大，奖励越小
        reward = torch.exp(-velocity_error * 10.0)  # 10.0是敏感度系数
        
        # 额外惩罚：如果速度为负（向下移动），给予严厉惩罚
        downward_penalty = torch.where(z_velocity < 0.0,
                                      z_velocity * -20.0,  # 向下速度越大，惩罚越重
                                      torch.tensor(0.0, device=self.device))
        
        return reward + downward_penalty  

    def _reward_movement_penalty(self):
        """巨额惩罚：当xyz速度都极其小的时候（防止机器人不动）"""
        x_vel = torch.abs(self.base_lin_vel[:, 0])
        y_vel = torch.abs(self.base_lin_vel[:, 1]) 
        z_vel = torch.abs(self.base_lin_vel[:, 2])
        total_velocity = torch.sqrt(x_vel**2 + y_vel**2 + z_vel**2)
        
        min_velocity_threshold = 0.05  # 🔥 从0.02提高到0.05（更严格）
        
        is_too_slow = total_velocity < min_velocity_threshold
        huge_penalty = torch.where(is_too_slow, 
                                torch.tensor(-10.0, device=self.device),  
                                torch.tensor(0.0, device=self.device))  
        return -huge_penalty

    # def _reward_wheel_wall_contact(self):
    #     """激励轮子与墙壁接触：每个轮子独立计算奖励后求和"""
    #     # (num_envs, num_feet, 3) -> (num_envs, num_feet)
    #     wheel_contact_forces = self.contact_forces[:, self.feet_indices, :]
    #     horizontal_force_magnitude = torch.norm(wheel_contact_forces[:, :, :2], dim=2)
    #     min_contact_force_per_wheel = 5.0  # 5N
    #     is_wheel_in_contact = horizontal_force_magnitude > min_contact_force_per_wheel
    #     all_four_wheels_in_contact = torch.all(is_wheel_in_contact, dim=1)
    #     total_horizontal_force = torch.sum(horizontal_force_magnitude, dim=1)
    #     max_force_reward = 200.0
    #     potential_reward = torch.clamp(total_horizontal_force / max_force_reward, max=1.0)
    #     final_reward = potential_reward * all_four_wheels_in_contact.float()
    #     return final_reward
        
    # def _reward_leg_joint_symmetry(self):
    #     """
    #     惩罚【左右腿】关节角度的差异，鼓励形成对称步态。
    #     这个函数会强制“装死”的腿模仿正在工作的腿。
    #     """
    #     back_hip_pitch_pos = self.dof_pos[:, self.back_leg_indices['hip_pitch']] # 形状: (env, 2)
    #     back_knee_pos = self.dof_pos[:, self.back_leg_indices['knee']]           # 形状: (env, 2)
    #     front_hip_pitch_pos = self.dof_pos[:, self.front_leg_indices['hip_pitch']]
    #     front_knee_pos = self.dof_pos[:, self.front_leg_indices['knee']]
    #     back_hip_diff = torch.abs(back_hip_pitch_pos[:, 0] - back_hip_pitch_pos[:, 1])
    #     back_knee_diff = torch.abs(back_knee_pos[:, 0] - back_knee_pos[:, 1])
    #     front_hip_diff = torch.abs(front_hip_pitch_pos[:, 0] - front_hip_pitch_pos[:, 1])
    #     front_knee_diff = torch.abs(front_knee_pos[:, 0] - front_knee_pos[:, 1])
    #     total_symmetry_error = back_hip_diff + back_knee_diff + front_hip_diff + front_knee_diff
    #     return total_symmetry_error
        
    def _reward_horizontal_velocity(self):
        """
        严厉惩罚机器人在X和Y轴上的线速度，鼓励纯粹的Z轴运动。
        """
        horizontal_vel_sq = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        return horizontal_vel_sq

    def _reward_continuous_contact(self):
        # 计算在历史窗口内，每只脚的平均接触时间（0到1之间）
        contact_ratios = torch.mean(self.contact_history.float(), dim=-1)
        
        # 找出接触时间最短的那只脚（木桶短板）
        min_ratio, _ = torch.min(contact_ratios, dim=-1)
        
        # 如果最短接触率低于阈值，给予惩罚
        # 阈值0.3意味着至少30%的时间内所有脚都应该接触墙壁
        contact_threshold = 0.3
        penalty = torch.clamp(contact_threshold - min_ratio, min=0.0)
        return min_ratio

    def _reward_bl_contact(self):
        """
        专门惩罚BL轮子悬空的情况
        轮子索引顺序: [WHEEL_BL, WHEEL_BR, WHEEL_FL, WHEEL_FR]
        BL索引 = 0
        """
        # 获取BL轮子的接触力 (num_envs,)
        bl_contact_force = torch.norm(self.contact_forces[:, self.feet_indices[0], :], dim=-1)
        
        # 接触力阈值（牛顿）
        contact_threshold = 1.0
        
        # BL接触：给正奖励；BL悬空：给负惩罚
        reward = torch.where(bl_contact_force > contact_threshold,
                           torch.tensor(0.5, device=self.device),   # BL接触时给小奖励
                           torch.tensor(-1.0, device=self.device))  # BL悬空时给惩罚
        
        return reward

    def _reward_jumping_motion(self):
        """
        🔥 新增：鼓励跳跃式的腿部动作
        通过奖励腿部关节（hip_pitch和knee）的速度变化来鼓励动态运动
        排除轮子关节，只考虑腿部关节
        """
        # 获取所有非轮子关节的速度 (排除每条腿的第5个DOF，即轮子)
        # 每条腿5个DOF: [HIP_YAW, HIP_ROLL, HIP_PITCH, KNEE, WHEEL]
        # 我们只关注 HIP_PITCH (索引2) 和 KNEE (索引3)
        leg_dof_indices = []
        for leg_start_idx in range(0, 20, 5):  # 0, 5, 10, 15 (4条腿)
            leg_dof_indices.append(leg_start_idx + 2)  # HIP_PITCH
            leg_dof_indices.append(leg_start_idx + 3)  # KNEE
        
        leg_dof_vel = self.dof_vel[:, leg_dof_indices]  # (num_envs, 8)
        
        # 计算腿部关节速度的绝对值之和，作为动态性指标
        leg_motion_magnitude = torch.sum(torch.abs(leg_dof_vel), dim=1)  # (num_envs,)
        
        # 设置一个合理的阈值范围，鼓励适度的动态运动
        # 太小：静止；太大：剧烈抖动
        min_motion = 2.0  # rad/s，低于此值认为太静态
        max_motion = 15.0  # rad/s，高于此值认为太剧烈
        
        # 计算奖励：在合理范围内的动态运动给予奖励
        normalized_motion = torch.clamp((leg_motion_magnitude - min_motion) / (max_motion - min_motion), 0.0, 1.0)
        
        return normalized_motion

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        print(f"最终加载的奖励: {self.reward_names}")
        print("===============================")

    def step(self, actions):
        obs, privileged_obs, rews, dones, infos = super().step(actions)
        return obs, privileged_obs, rews, dones, infos
    
    def check_termination(self):
        super().check_termination()
        too_low = self.root_states[:, 2] < 0.1 
        roll_pitch_too_large = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) > 0.25
        self.reset_buf |= too_low
        self.reset_buf |= roll_pitch_too_large  
        
        # 🔥 优化版：分段高度停滞检测（高处更宽容）
        # 初始化高度追踪（只在第一次调用时执行）
        if not hasattr(self, 'last_height_checkpoint'):
            self.last_height_checkpoint = self.root_states[:, 2].clone()
            self.height_check_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
            self.no_progress_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        
        # 每20秒检查一次高度进度（更宽容）
        self.height_check_counter += 1
        check_interval = int(20.0 / self.dt)  # 🔥 放宽到20秒
        
        # 只在特定步数时检查（避免每步都计算）
        if self.height_check_counter[0] >= check_interval:
            current_height = self.root_states[:, 2]
            height_gain = current_height - self.last_height_checkpoint
            
            # 🔥 分段要求：低处要求高，1.8m以上完全豁免
            # <1.0m: 需要20cm/20秒 (必须努力爬升)
            # 1.0-1.5m: 需要15cm/20秒 (鼓励继续爬)
            # 1.5-1.8m: 需要5cm/20秒 (允许稳定停留)
            # >=1.8m: 完全豁免停滞检测 (胜利区域！)
            
            # 只对1.8m以下的机器人进行停滞检测
            below_safe_zone = current_height < 1.8
            
            threshold_low = torch.where(current_height < 1.0, 
                                       torch.tensor(0.20, device=self.device),
                                       torch.where(current_height < 1.5,
                                                  torch.tensor(0.15, device=self.device),
                                                  torch.tensor(0.05, device=self.device)))
            
            no_progress = (height_gain < threshold_low) & below_safe_zone  # 🔥 1.8m以上不判定
            
            # 更新计数器（使用整数操作更快）
            self.no_progress_counter = torch.where(
                no_progress,
                self.no_progress_counter + 1,
                torch.zeros_like(self.no_progress_counter)
            )
            
            # 连续3次（60秒）无进度才重置（更宽容）
            stuck_too_long = self.no_progress_counter >= 3
            self.reset_buf |= stuck_too_long
            
            # 重置计数器和更新检查点
            self.height_check_counter[:] = 0
            self.last_height_checkpoint = current_height.clone()
        
        if torch.any(too_low) and self.common_step_counter % 100 == 0:
            print(f"机器人高度过低终止: {torch.sum(too_low).item()} 个")
        if torch.any(roll_pitch_too_large) and self.common_step_counter % 100 == 0:
            print(f"机器人倾斜过大终止: {torch.sum(roll_pitch_too_large).item()} 个")
        if hasattr(self, 'no_progress_counter') and self.common_step_counter % 100 == 0:
            stuck = self.no_progress_counter >= 2
            if torch.any(stuck):
                print(f"机器人高度停滞终止: {torch.sum(stuck).item()} 个")

        # 连续日志记录轮子速度
        self.log_counter += 1
        if self.log_counter % 5 == 0 and self.log_file is not None and hasattr(self, 'log_file'):
            time_s = (self.episode_length_buf[0] * self.dt).item()
            wheel_indices = [4, 9, 14, 19]
            wheel_velocities = self.dof_vel[0, wheel_indices].cpu().numpy()
            log_line = f"{time_s:.4f},{wheel_velocities[0]:.4f},{wheel_velocities[1]:.4f},{wheel_velocities[2]:.4f},{wheel_velocities[3]:.4f}\n"
            self.log_file.write(log_line)
            self.log_file.flush()  # 立即写入文件，确保数据不丢失
    
    def __del__(self):
        if hasattr(self, 'log_file') and self.log_file is not None:
            log_filename = getattr(self.log_file, 'name', '轮子速度日志文件')
            self.log_file.close()
            print(f"日志文件 '{log_filename}' 已关闭。")