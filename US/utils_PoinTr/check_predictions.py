import os
import numpy as np
import open3d as o3d
import random
directory_path_partial = "/Users/aidanamassalimova/Documents/US_Paper/Point_Tr_Dataset/val/partial/10102023"
directory_path_complete = "/Users/aidanamassalimova/Documents/US_Paper/Point_Tr_Dataset/val/complete/10102023"
directory_path_pred = "/Users/aidanamassalimova/Documents/US_Paper/Point_Tr_Dataset/val/predictions"
files = os.listdir(directory_path_partial)
def randomly_downsample_point_cloud(point_cloud, num_points=1024):
    # Check total number of points
    total_points = point_cloud.shape[0]

    if total_points <= num_points:
        return point_cloud  # No need to downsample if already <= num_points

    # Randomly select num_points indices from total_points
    selected_indices = np.random.choice(total_points, num_points, replace=False)

    # Extract the subset of point cloud using the selected indices
    downsampled_point_cloud = point_cloud[selected_indices]

    return downsampled_point_cloud
random.shuffle(files)
for file in files:
    print(file)
    specimen = file[:-2]
    print(specimen)
    colors = [(1,0,0), (0,0,1), (0,1,0), (1,1,0), (1,0,1)]

    for i in range(1,6):
        vis = []

        partial = o3d.io.read_point_cloud(os.path.join(directory_path_partial, specimen+"L{}".format(i), "00.pcd"))
        partial_dpd = randomly_downsample_point_cloud(np.asarray(partial.points))
        partial_dpd_pcd = o3d.geometry.PointCloud()
        partial_dpd_pcd.points =o3d.utility.Vector3dVector(partial_dpd)
        complete = o3d.io.read_point_cloud(os.path.join(directory_path_complete, specimen+"L{}".format(i) + ".pcd"))
        pred_np = np.load(os.path.join(directory_path_pred, specimen+"L{}".format(i) + ".npz"))["arr_0"]
        predictions = o3d.geometry.PointCloud()
        predictions.points= o3d.utility.Vector3dVector(pred_np.squeeze(0))
        complete.paint_uniform_color((0,1,0))
        partial_dpd_pcd.paint_uniform_color([0,0,1])
        predictions.paint_uniform_color([1,0,0])
        vis.append(partial_dpd_pcd)
        vis.append(complete)
        vis.append(predictions)
        o3d.visualization.draw_geometries(vis)
