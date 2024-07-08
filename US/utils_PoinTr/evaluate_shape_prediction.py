import os
import random
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv
from scipy.spatial import KDTree

def calculate_fscore(gt: o3d.geometry.PointCloud, pr: o3d.geometry.PointCloud, percent):
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''

    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)

    bbox=pr.get_oriented_bounding_box()
    bbox.color=[1,0,0]
    coordinate_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=bbox.get_center())
   # o3d.visualization.draw_geometries([coordinate_axes.rotate(bbox.R),pr, bbox])

    th = percent*(np.max(bbox.extent))/100

    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall


def calculate_iou(point_cloud1, point_cloud2):
    # Calculate the bounding boxes for each point cloud
    min1, max1 = np.min(point_cloud1, axis=0), np.max(point_cloud1, axis=0)
    min2, max2 = np.min(point_cloud2, axis=0), np.max(point_cloud2, axis=0)

    # Calculate the intersection
    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)
    intersection_volume = np.maximum(0, intersection_max - intersection_min).prod()

    # Calculate the union
    union_volume = (max1 - min1).prod() + (max2 - min2).prod() - intersection_volume

    # Calculate IoU
    iou = intersection_volume / union_volume if union_volume > 0 else 0

    return iou
def pick_points(pcd):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        return vis.get_picked_points()

def chamfer_distance(arr1, arr2):
    distance_1_to_2 = 0
    distance_2_to_1 = 0

    points1 = np.column_stack((arr1[:,0], arr1[:,1], arr2[:,2]))
    points2 = np.column_stack((arr2[:,0], arr2[:,1], arr2[:,2]))

    # Compute distance from each point in arr1 to arr2
    for p1 in points1:
        distances = np.sqrt(np.sum((points2 - p1) ** 2, axis=1))
        min_distance = np.min(distances)
        distance_1_to_2 += min_distance

    # Compute distance from each point in arr2 to arr1
    for p2 in points2:
        distances = np.sqrt(np.sum((points1 - p2) ** 2, axis=1))
        min_distance = np.min(distances)
        distance_2_to_1 += min_distance

    return (distance_1_to_2 + distance_2_to_1) / (len(arr1) + len(arr2))
directory_path_partial = "/Users/aidanamassalimova/Documents/US_Paper/Point_Tr_Dataset/val/partial/10102023"
directory_path_complete = "/Users/aidanamassalimova/Documents/US_Paper/Point_Tr_Dataset/val/complete/10102023"
directory_path_pred = "/Users/aidanamassalimova/Documents/US_Paper/Point_Tr_Dataset/val/predictions"
files = os.listdir(directory_path_partial)
specimen_list =[]
vertebrae_list = []
rmse_all =[]
CD_all= []
IOU_all =[]
for file in files:
    print(file)
    specimen = file[:-2]
    specimen_list.append(specimen)
    vertebrae = file[-2:]
    vertebrae_list.append(vertebrae)
    partial = o3d.io.read_point_cloud(os.path.join(directory_path_partial, file, "00.pcd"))
    complete = o3d.io.read_point_cloud(os.path.join(directory_path_complete, file + ".pcd"))
    pred_np = np.load(os.path.join(directory_path_pred, file + ".npz"))["arr_0"]
    predictions = o3d.geometry.PointCloud()
    predictions.points= o3d.utility.Vector3dVector(pred_np.squeeze(0))
    CD = chamfer_distance(np.asarray(predictions.points),np.asarray(complete.points))
    IOU = calculate_iou(np.asarray(predictions.points),np.asarray(complete.points))
    reg_p2p = o3d.pipelines.registration.registration_icp(
        predictions, complete, 10, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000000))
    rmse_all.append(reg_p2p.inlier_rmse)
    CD_all.append(CD)
    IOU_all.append(IOU)

data ={
"Specimen": specimen_list,
"Vertebrae": vertebrae_list,
"CD": CD_all,
"IOU": IOU_all,
"RMSE": rmse_all,

}

df = pd.DataFrame(data)
df.to_csv("evaluation_US.csv", index=False)
print(df.describe())


