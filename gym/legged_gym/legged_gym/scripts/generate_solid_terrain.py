#!/usr/bin/env python3
"""
创建改进的缝隙地形模型
- 确保每面墙都是实心的
- 墙壁厚度为0.3m
- 生成STL和OBJ格式的文件
"""

import numpy as np
import trimesh

def create_solid_crevice_terrain():
    """
    创建符合原始设计的缝隙地形
    - 只有两面平行墙壁
    - 没有地面，没有前后堵墙
    - 墙壁厚度0.3m（按要求增厚）
    """
    # 地形参数
    terrain_length = 10.0  # 地形总长度 (m) 
    
    crevice_width = 0.8    # 缝隙宽度 (m)
    wall_thickness = 0.3   # 墙壁厚度 (m) - 按要求增厚到0.3m
    wall_height = 2.0      # 墙壁高度 (m)
    
    # 创建组件列表
    meshes = []
    
    # 1. 创建右侧墙壁（实心）
    # 墙壁内表面在 +crevice_width/2，外表面在 +crevice_width/2 + wall_thickness
    right_wall = trimesh.creation.box(
        extents=[terrain_length, wall_thickness, wall_height]
    )
    right_wall.apply_translation([
        0,  # x方向居中
        crevice_width/2 + wall_thickness/2,  # y方向：缝隙边缘 + 厚度的一半
        wall_height/2  # z方向：从地面开始
    ])
    meshes.append(right_wall)
    
    # 2. 创建左侧墙壁（实心）
    # 墙壁内表面在 -crevice_width/2，外表面在 -crevice_width/2 - wall_thickness
    left_wall = trimesh.creation.box(
        extents=[terrain_length, wall_thickness, wall_height]
    )
    left_wall.apply_translation([
        0,  # x方向居中
        -(crevice_width/2 + wall_thickness/2),  # y方向：缝隙边缘 - 厚度的一半
        wall_height/2  # z方向：从地面开始
    ])
    meshes.append(left_wall)
    
    # 合并所有网格
    combined_mesh = trimesh.util.concatenate(meshes)
    
    # 确保网格是水密的（实心的）
    if not combined_mesh.is_watertight:
        print("警告：网格不是水密的，尝试修复...")
        combined_mesh.fill_holes()
        combined_mesh.remove_degenerate_faces()
        combined_mesh.remove_duplicate_faces()
        combined_mesh.remove_unreferenced_vertices()
    
    print(f"地形网格信息:")
    print(f"- 顶点数: {len(combined_mesh.vertices)}")
    print(f"- 面数: {len(combined_mesh.faces)}")
    print(f"- 是否水密: {combined_mesh.is_watertight}")
    print(f"- 体积: {combined_mesh.volume:.2f} m³")
    print(f"- 缝隙内部宽度: {crevice_width}m")
    print(f"- 墙壁厚度: {wall_thickness}m")
    print(f"- 墙壁高度: {wall_height}m")
    print(f"- 通道长度: {terrain_length}m（前后开放）")
    
    return combined_mesh

def main():
    print("正在生成改进的缝隙地形...")
    
    # 创建地形网格
    terrain_mesh = create_solid_crevice_terrain()
    
    # 保存文件路径
    output_dir = "/home/mrp/gym/legged_gym/resources/terrains"
    
    # 保存为STL格式（更适合物理仿真）
    stl_path = f"{output_dir}/solid_crevice_80cm.stl"
    terrain_mesh.export(stl_path)
    print(f"STL文件已保存: {stl_path}")
    
    # 保存为OBJ格式（兼容性）
    obj_path = f"{output_dir}/solid_crevice_80cm.obj"
    terrain_mesh.export(obj_path)
    print(f"OBJ文件已保存: {obj_path}")
    
    # 创建对应的URDF文件
    urdf_content = f"""<?xml version="1.0"?>
<robot name="solid_crevice_terrain_80cm">
  <link name="terrain_link">
    <inertial>
      <mass value="1000.0"/>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <inertia ixx="1000.0" ixy="0.0" ixz="0.0" iyy="1000.0" iyz="0.0" izz="1000.0"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="solid_crevice_80cm.stl" scale="1 1 1"/>
      </geometry>
      <material name="terrain_material">
        <color rgba="0.6 0.4 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="solid_crevice_80cm.stl" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>"""
    
    urdf_path = f"{output_dir}/solid_crevice_80cm.urdf"
    with open(urdf_path, 'w') as f:
        f.write(urdf_content)
    print(f"URDF文件已保存: {urdf_path}")
    
    print("\n地形生成完成！")
    print("特点：")
    print("- 墙壁厚度: 0.3m")
    print("- 只有左右两面墙壁（实心）")
    print("- 没有地面，没有前后堵墙")
    print("- 缝隙宽度: 0.8m，墙高: 2.0m")
    print("- 通道长度: 10.0m（前后开放）")

if __name__ == "__main__":
    main()
