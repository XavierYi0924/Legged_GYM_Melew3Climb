#!/usr/bin/env python3
"""
查看新的实心地形模型
"""

import os
from isaacgym import gymapi
from isaacgym import gymutil

def main():
    # 解析命令行参数
    args = gymutil.parse_arguments(description="Solid Terrain Viewer")
    
    # 创建gym实例
    gym = gymapi.acquire_gym()
    
    # 配置仿真参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    
    # 使用GPU物理
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
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
    
    # 加载地形模型
    terrain_asset_path = "/home/mrp/gym/legged_gym/resources/terrains/solid_crevice_80cm.urdf"
    terrain_asset_root = os.path.dirname(terrain_asset_path)
    terrain_asset_file = os.path.basename(terrain_asset_path)
    
    terrain_asset_options = gymapi.AssetOptions()
    terrain_asset_options.fix_base_link = True
    terrain_asset_options.disable_gravity = True
    
    print(f"Loading terrain asset from: {terrain_asset_path}")
    try:
        terrain_asset = gym.load_asset(sim, terrain_asset_root, terrain_asset_file, terrain_asset_options)
        print("实心地形加载成功")
    except Exception as e:
        print(f"Failed to load terrain asset: {e}")
        return
    
    # 创建环境
    spacing = 2.0
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    
    env = gym.create_env(sim, lower, upper, 1)
    
    # 添加地形到环境
    terrain_pose = gymapi.Transform()
    terrain_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    
    try:
        terrain_actor = gym.create_actor(env, terrain_asset, terrain_pose, "terrain", 0, 0)
        print("实心地形actor创建成功")
    except Exception as e:
        print(f"Failed to create terrain actor: {e}")
        return
    
    # 设置查看器
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("Failed to create viewer")
        quit()
    
    # 设置相机位置 - 俯视缝隙
    cam_pos = gymapi.Vec3(5.0, 0.0, 3.0)
    cam_target = gymapi.Vec3(0.0, 0.0, -1.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    print("\n=== 实心地形查看器 ===")
    print("特点:")
    print("- 墙壁厚度: 0.3m")
    print("- 只有左右两面墙壁（实心）")
    print("- 没有地面，没有前后堵墙")
    print("- 缝隙宽度: 0.8m，墙高: 2.0m")
    print("- 通道长度: 10.0m（前后开放）")
    print("\n控制:")
    print("- 鼠标拖拽: 旋转视角")
    print("- 滚轮: 缩放")
    print("- ESC: 退出")
    
    # 主循环
    while not gym.query_viewer_has_closed(viewer):
        # 刷新仿真状态
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 更新viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
    
    print("Closing viewer...")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
