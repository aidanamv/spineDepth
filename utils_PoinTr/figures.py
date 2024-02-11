import os

import open3d as o3d

dir = "/Volumes/Extreme SSD/PoinTr dataset/segmented_spinedepth_new/Specimen_7/recording_1/cam_0/frame_0"

ply1 = o3d.io.read_point_cloud(os.path.join(dir,"pointcloud_vert1.ply"))
ply2 = o3d.io.read_point_cloud(os.path.join(dir,"pointcloud_vert2.ply"))
ply3 = o3d.io.read_point_cloud(os.path.join(dir,"pointcloud_vert3.ply"))
ply4 = o3d.io.read_point_cloud(os.path.join(dir,"pointcloud_vert4.ply"))
ply5 = o3d.io.read_point_cloud(os.path.join(dir,"pointcloud_vert5.ply"))

mesh1 = o3d.io.read_triangle_mesh(os.path.join(dir,"transformed_vertebra1.stl"))
mesh2 = o3d.io.read_triangle_mesh(os.path.join(dir,"transformed_vertebra2.stl"))
mesh3 = o3d.io.read_triangle_mesh(os.path.join(dir,"transformed_vertebra3.stl"))
mesh4 = o3d.io.read_triangle_mesh(os.path.join(dir,"transformed_vertebra4.stl"))
mesh5 = o3d.io.read_triangle_mesh(os.path.join(dir,"transformed_vertebra5.stl"))

mesh1.compute_vertex_normals()
mesh2.compute_vertex_normals()
mesh3.compute_vertex_normals()
mesh4.compute_vertex_normals()
mesh5.compute_vertex_normals()

ply1.paint_uniform_color([0,0,1])
ply2.paint_uniform_color([0,1,0])
ply3.paint_uniform_color([0,0,1])
ply4.paint_uniform_color([1,1,0])
ply5.paint_uniform_color([1,0,1])

mesh1_pcd = mesh1.sample_points_uniformly(number_of_points=4096)
mesh1_pcd.paint_uniform_color([0,0,1])

ply1.translate([0,0,-100], relative=True)
o3d.visualization.draw_geometries([ply1,mesh1_pcd])
