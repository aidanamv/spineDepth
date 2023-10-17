import open3d as o3d
import os

def crop_pcd(pc_dir_1, pc_dir_2):
    pcd_path_1 = os.path.join(pc_dir_1, 'pointcloud.ply')
    pcd_path_2 = os.path.join(pc_dir_2, 'pointcloud.ply')

    pcd_1 = o3d.io.read_point_cloud(pcd_path_1)
    pcd_2 = o3d.io.read_point_cloud(pcd_path_2)
    o3d.visualization.draw_geometries_with_editing([pcd_1])
    o3d.visualization.draw_geometries_with_editing([pcd_2])
    pcd_path_cropped_1 = os.path.join(pc_dir_1, 'cropped_1.ply')
    pcd_path_cropped_2 = os.path.join(pc_dir_2, 'cropped_1.ply')
    pcd_cropped_1 = o3d.io.read_point_cloud(pcd_path_cropped_1)
    pcd_cropped_2 = o3d.io.read_point_cloud(pcd_path_cropped_2)

    o3d.visualization.draw_geometries([pcd_cropped_1,pcd_cropped_2])
    return pcd_cropped_1, pcd_cropped_2





