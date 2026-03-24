# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        # 使用copy来避免修改原始字典
        terrain_kwargs_copy = self.cfg.terrain_kwargs.copy()
        terrain_type = terrain_kwargs_copy.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,  # 使用正确的长度
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)

            eval(terrain_type)(terrain, **terrain_kwargs_copy)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,  # 使用正确的长度
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[7]:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        else:
            # 如果选择概率超出了默认地形，使用通道地形
            channel_terrain(terrain, channel_width=0.8, wall_height=2.0, wall_thickness=0.3)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        
        # 对于通道地形，确保机器人在通道内而不是墙上生成
        if hasattr(terrain, 'is_channel_terrain') and terrain.is_channel_terrain:
            # 通道地形：机器人应该在地面高度生成（通道内）
            env_origin_z = 0.0
        else:
            # 其他地形：使用原来的高度计算方法
            x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
            x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
            y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
            y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def channel_terrain(terrain, channel_width=0.7, wall_height=2.0, wall_thickness=0.7):
    """
    创建通道地形 - 两侧有墙壁的直线通道，沿X方向延伸
    
    Parameters:
        terrain: 地形对象
        channel_width (float): 通道宽度 [m]
        wall_height (float): 墙壁高度 [m] 
        wall_thickness (float): 墙壁厚度 [m]
    """
    # 标记这是通道地形
    terrain.is_channel_terrain = True
    
    # 转换为离散单位（使用round四舍五入以支持更精确的宽度，如0.78米）
    wall_height_discrete = int(wall_height / terrain.vertical_scale)
    channel_width_discrete = round(channel_width / terrain.horizontal_scale)
    wall_thickness_discrete = round(wall_thickness / terrain.horizontal_scale)
    
    # 地形中心Y坐标
    center_y = terrain.width // 2
    
    # 计算通道在Y方向的边界
    channel_half_width = channel_width_discrete // 2
    channel_start_y = center_y - channel_half_width
    channel_end_y = center_y + channel_half_width
    
    # 将所有区域设为地面高度(0)
    terrain.height_field_raw[:, :] = 0
    
    # 创建左侧墙壁 (y < channel_start_y)
    if channel_start_y > wall_thickness_discrete:
        left_wall_start = max(0, channel_start_y - wall_thickness_discrete)
        terrain.height_field_raw[:, left_wall_start:channel_start_y] = wall_height_discrete
    
    # 创建右侧墙壁 (y > channel_end_y)  
    if channel_end_y + wall_thickness_discrete < terrain.width:
        right_wall_end = min(terrain.width, channel_end_y + wall_thickness_discrete)
        terrain.height_field_raw[:, channel_end_y:right_wall_end] = wall_height_discrete
    
    return terrain

def progressive_climb_terrain(terrain, channel_width=0.7, wall_height=2.0, wall_thickness=0.7, u_shape_depth=0.0, slope_angle=0):
    """
    创建渐进式攀爬地形 - 基于论文的课程学习地形，带平滑U型过渡
    
    Parameters:
        terrain: 地形对象
        channel_width (float): 通道宽度 [m]
        wall_height (float): 墙壁高度 [m] 
        wall_thickness (float): 墙壁厚度 [m]
        u_shape_depth (float): U形凹槽深度 [m] - 论文中的r参数
        slope_angle (float): 坡度角度 [度]
    """
    import math
    
    # 标记这是通道地形
    terrain.is_channel_terrain = True
        
    # 转换为离散单位（使用round四舍五入以支持更精确的宽度，如0.78米）
    wall_height_discrete = int(wall_height / terrain.vertical_scale)
    channel_width_discrete = round(channel_width / terrain.horizontal_scale)
    wall_thickness_discrete = round(wall_thickness / terrain.horizontal_scale)
    u_depth_discrete = int(u_shape_depth / terrain.vertical_scale)
    
    # 地形中心Y坐标
    center_y = terrain.width // 2
    
    # 计算通道在Y方向的边界
    channel_half_width = channel_width_discrete // 2
    channel_start_y = center_y - channel_half_width
    channel_end_y = center_y + channel_half_width
    
    # 将所有区域初始化为地面高度(0)
    terrain.height_field_raw[:, :] = 0
    
    # 为整个地形创建平滑的U形到墙面的过渡
    for x in range(terrain.length):
        for y in range(terrain.width):
            # 计算Y方向距离通道中心的距离
            distance_from_center = abs(y - center_y)
            
            if distance_from_center <= channel_half_width:
                # 在通道内部 - 创建U形凹槽，底部为零点
                if u_shape_depth > 0:
                    # 使用二次函数创建平滑的U形，中心为0，边缘高度为u_depth_discrete
                    normalized_distance = distance_from_center / channel_half_width  # 0到1
                    # 二次函数：中心为0（底部），边缘高度为u_shape_depth
                    u_height = u_depth_discrete * (normalized_distance ** 2)
                    terrain.height_field_raw[x, y] = int(u_height)
                else:
                    # 无U形，保持地面水平
                    terrain.height_field_raw[x, y] = 0
                    
            elif distance_from_center <= channel_half_width + wall_thickness_discrete:
                # 在墙壁区域 - 从通道边缘到墙顶的平滑过渡
                wall_distance = distance_from_center - channel_half_width
                wall_progress = wall_distance / wall_thickness_discrete  # 0到1
                
                # 计算起始高度（通道边缘的高度，基于U型坡）
                if u_shape_depth > 0:
                    start_height = u_depth_discrete  # 从U型坡边缘高度开始
                else:
                    start_height = 0  # 从底部开始
                
                if slope_angle >= 90:
                    # 垂直墙面 - 线性插值从地面到墙顶
                    height = start_height + wall_height_discrete * wall_progress
                    terrain.height_field_raw[x, y] = int(height)
                elif slope_angle > 0:
                    # 倾斜墙面
                    slope_rad = math.radians(slope_angle)
                    x_progress = x / terrain.length  # 沿X方向的进度
                    max_wall_height = int(wall_height_discrete * x_progress)
                    
                    # 使用斜率计算高度
                    slope_height = int(wall_distance * math.tan(slope_rad))
                    target_height = min(slope_height, max_wall_height)
                    
                    # 从起始高度平滑过渡到目标高度
                    height = start_height + target_height * wall_progress
                    terrain.height_field_raw[x, y] = int(height)
                else:
                    # 0度 - 垂直墙面
                    height = start_height + wall_height_discrete * wall_progress
                    terrain.height_field_raw[x, y] = int(height)
            else:
                # 墙壁外侧区域保持地面高度
                terrain.height_field_raw[x, y] = 0
    
    return terrain
