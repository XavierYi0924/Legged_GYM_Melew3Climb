#!/usr/bin/env python3
"""
简单的Isaac Gym模型查看器
只加载和显示机器人及地形模型，不进行训练
"""

import os
import sys
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

def main():
    # 解析命令行参数
    args = gymutil.parse_arguments(description="Isaac Gym Model Viewer")
    
    # 创建gym实例
    gym = gymapi.acquire_gym()
    
    # 配置仿真参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0  # 60 FPS
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    
    # 使用GPU物理
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
    
    # 创建仿真实例
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("Failed to create sim")
        quit()
    
    # 添加地面平面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    
    # 加载机器人模型
    robot_asset_root = "/home/mrp/gym/legged_gym/resources/robots/melew3"
    robot_asset_file = "urdf/MELEW3.urdf"  # 修正为正确的文件名
    
    robot_asset_options = gymapi.AssetOptions()
    robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    robot_asset_options.collapse_fixed_joints = True
    robot_asset_options.replace_cylinder_with_capsule = True
    robot_asset_options.flip_visual_attachments = True
    robot_asset_options.fix_base_link = False
    robot_asset_options.density = 0.001
    robot_asset_options.angular_damping = 0.0
    robot_asset_options.linear_damping = 0.0
    robot_asset_options.max_angular_velocity = 1000.0
    robot_asset_options.max_linear_velocity = 1000.0
    robot_asset_options.armature = 0.0
    robot_asset_options.thickness = 0.01
    
    print(f"Loading robot asset from: {robot_asset_root}/{robot_asset_file}")
    robot_asset = gym.load_asset(sim, robot_asset_root, robot_asset_file, robot_asset_options)
    
    # 加载地形模型（可选）
    terrain_asset = None
    try:
        terrain_asset_path = "/home/mrp/gym/legged_gym/resources/terrains/crevice_80cm.urdf"
        terrain_asset_root = os.path.dirname(terrain_asset_path)
        terrain_asset_file = os.path.basename(terrain_asset_path)
        
        terrain_asset_options = gymapi.AssetOptions()
        terrain_asset_options.fix_base_link = True
        terrain_asset_options.disable_gravity = True
        
        print(f"Loading terrain asset from: {terrain_asset_path}")
        terrain_asset = gym.load_asset(sim, terrain_asset_root, terrain_asset_file, terrain_asset_options)
        print("Terrain asset loaded successfully")
    except Exception as e:
        print(f"Failed to load terrain: {e}")
        print("Continuing without terrain...")
    
    # 创建环境
    spacing = 2.0
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    
    env = gym.create_env(sim, lower, upper, 1)
    
    # 添加地形到环境（如果加载成功）
    if terrain_asset is not None:
        terrain_pose = gymapi.Transform()
        terrain_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        gym.create_actor(env, terrain_asset, terrain_pose, "terrain", 0, 0)
        print("Terrain actor created")
    
    # 添加机器人到环境
    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # 稍微抬高一点
    robot_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    robot_actor = gym.create_actor(env, robot_asset, robot_pose, "melew3", 0, 0)
    print("Robot actor created")
    
    # 获取机器人关节信息
    num_dofs = gym.get_actor_dof_count(env, robot_actor)
    print(f"Robot has {num_dofs} DOFs")
    
    # 设置查看器
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("Failed to create viewer")
        quit()
    
    # 设置相机位置
    cam_pos = gymapi.Vec3(3.0, 3.0, 2.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    print("\n=== Isaac Gym Model Viewer ===")
    print("Controls:")
    print("- Mouse: Rotate view")
    print("- WASD: Move camera")
    print("- Q/E: Up/Down")
    print("- ESC: Exit")
    print("- R: Reset camera position")
    print("\nPress any key to continue...")
    
    # 主循环
    while not gym.query_viewer_has_closed(viewer):
        # 刷新仿真状态
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 更新viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        
        # 检查键盘输入
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "reset_camera" and evt.value > 0:
                gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
                print("Camera reset")
    
    print("Closing viewer...")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
