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
specimen_list = []
camera_list = []
vertebrae_list = []
CD_all = []
IOU_all = []
rmse_all = []
tre_all = []
df_old = pd.read_csv("evaluation.csv")
print(df_old.describe())

for fold in range(1):
    print(fold)
    dir = "/Users/aidanamassalimova/Documents/fold_{}/val".format(fold)
    files = os.listdir(dir + "/predictions")
    print(len(files))
    vertebrae_registrar = []
    CD_list = []
    IOU_list = []
    inlier_rmses = []
    tre_list = []
    for file in files:
        result = file.split('_')
        specimen = result[1]
        specimen_list.append(specimen)
        camera_view =result[5]
        camera_list.append(camera_view)
        vertebrae = result[8][-5]
        vertebrae_list.append(vertebrae)
        predictions = np.load(os.path.join(dir, "predictions", file))["arr_0"]
        pcd_predictions = o3d.geometry.PointCloud()
        pcd_predictions.points = o3d.utility.Vector3dVector(predictions.squeeze(0))
        pcd_complete = o3d.io.read_point_cloud(os.path.join(dir, "complete", "10102023", file[:-4] + ".pcd"))
        pcd_partial = o3d.io.read_point_cloud(os.path.join(dir, "partial", "10102023", file[:-4] + "/00.pcd"))
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_predictions, pcd_complete, 10, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000000))
        pcd_predictions.paint_uniform_color((1, 0, 0))
        pcd_complete.paint_uniform_color((0, 0, 1))
        pcd_predictions.transform(reg_p2p.transformation)
       # o3d.visualization.draw_geometries([pcd_predictions, pcd_complete])
        pcd_predictions.paint_uniform_color([0, 0, 1])
        pcd_complete.paint_uniform_color([0, 1, 0])

        if vertebrae =="1" and "1" not in vertebrae_registrar:
            picked_id_source_L1 = pick_points(pcd_complete)
            vertebrae_registrar.append(vertebrae)
        if vertebrae == "2" and "2" not in vertebrae_registrar:
            picked_id_source_L2 = pick_points(pcd_complete)
            vertebrae_registrar.append(vertebrae)
        if vertebrae == "3" and "3" not in vertebrae_registrar:
            picked_id_source_L3 = pick_points(pcd_complete)
            vertebrae_registrar.append(vertebrae)
        if vertebrae == "4" and "4" not in vertebrae_registrar:
            picked_id_source_L4 = pick_points(pcd_complete)
            vertebrae_registrar.append(vertebrae)
        if vertebrae == "5" and "5" not in vertebrae_registrar:
            picked_id_source_L5 = pick_points(pcd_complete)
            vertebrae_registrar.append(vertebrae)

        if vertebrae =="1":
            picked_id_source = picked_id_source_L1
        if vertebrae == "2":
            picked_id_source = picked_id_source_L2
        if vertebrae == "3":
            picked_id_source = picked_id_source_L3
        if vertebrae == "4":
            picked_id_source = picked_id_source_L4
        if vertebrae == "5":
            picked_id_source = picked_id_source_L5

        tre =[]
        landmarks =[]
        for i in range(3):
            selected_point = pcd_complete.points[picked_id_source[i]]

            target_pcd = pcd_predictions.points
            kdtree = KDTree(target_pcd)

            # Query the KD-tree to find the index of the closest point in the point cloud to the given point
            closest_point_index = kdtree.query(selected_point)[1]

            # Calculate the distance between the closest point and the given point
            closest_distance = np.linalg.norm(target_pcd[closest_point_index] - selected_point)
            tre.append(closest_distance)
            landmarks.append(target_pcd[closest_point_index])

        L1 = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
        L2 = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
        L3 = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)


        L1.paint_uniform_color((1, 0, 0))
        L2.paint_uniform_color((1, 0, 0))
        L3.paint_uniform_color((1, 0, 0))

        L1.translate(landmarks[0], relative =False)
        L2.translate(landmarks[1], relative =False)
        L3.translate(landmarks[2], relative =False)

        #o3d.visualization.draw_geometries([pcd_predictions,pcd_complete,L1, L2, L3])

        CD = chamfer_distance(predictions.squeeze(0), np.asarray(pcd_complete.points))
        IOU = calculate_iou(predictions.squeeze(0), np.asarray(pcd_complete.points))




        CD_list.append(CD)
        IOU_list.append(IOU)
        inlier_rmses.append(reg_p2p.inlier_rmse)
        tre_list.append(np.median(tre))
        CD_all.append(CD)
        IOU_all.append(IOU)
        rmse_all.append(reg_p2p.inlier_rmse)
        tre_all.append(np.median(tre))


    print("fold_{}".format(fold))
    print(np.median(CD_list))
    print(np.median(IOU_list))
    print(np.median(inlier_rmses))
    print(np.median(tre_all))


print("min")
print(np.min(CD_all))
print(np.min(IOU_all))
print(np.min(rmse_all))
print(np.min(tre_all))

print("q1")
print(np.percentile(CD_all, 25))
print(np.percentile(IOU_all, 25))
print(np.percentile(rmse_all, 25))
print(np.percentile(tre_all, 25))
print("median")
print(np.median(CD_all))
print(np.median(IOU_all))
print(np.median(rmse_all))
print(np.median(tre_all))
print("q3")
print(np.percentile(CD_all, 75))
print(np.percentile(IOU_all, 75))
print(np.percentile(rmse_all, 75))
print(np.percentile(tre_all, 75))


print("max")
print(np.max(CD_all))
print(np.max(IOU_all))
print(np.max(rmse_all))
print(np.max(tre_all))



data ={
    "Specimen": specimen_list,
    "Vertebrae": vertebrae_list,
    "Camera": camera_list,
    "CD": CD_all,
    "IOU": IOU_all,
    "RMSE": rmse_all,
    "TRE": tre_all,

}
df = pd.concat([df_old, pd.DataFrame(data)], ignore_index=True)
df.to_csv("evaluation.csv", index= False)


