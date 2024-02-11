from itertools import chain

import pyvista as pv
import os
import numpy as np
import open3d as o3d
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from functools import reduce
from py_cylinder_fitting import BestFitCylinder
from skspatial.objects import Points



def compute_f1_score(gt_pc, pred_pc, threshold=0.1):


    # Perform nearest neighbor search
    search_tree = o3d.geometry.KDTreeFlann(gt_pc)
    distances, _ = search_tree.search_knn_vector_3d(np.asarray(pred_pc.points), 1)

    # Count true positives, false positives, and false negatives
    tp = np.sum(distances < threshold)
    fp = len(distances) - tp
    fn = len(gt_pc.points) - tp

    # Compute precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score


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

def chamfer_distance(arr1, arr2):
    distance_1_to_2 = 0
    distance_2_to_1 = 0

    points1 = np.column_stack((arr1[0], arr1[1], arr2[2]))
    points2 = np.column_stack((arr2[0], arr2[1], arr2[2]))

    # Compute distance from each point in arr1 to arr2
    for p1 in points1:
        distances = np.sqrt(np.sum((points2 - p1)**2, axis=1))
        min_distance = np.min(distances)
        distance_1_to_2 += min_distance

    # Compute distance from each point in arr2 to arr1
    for p2 in points2:
        distances = np.sqrt(np.sum((points1 - p2)**2, axis=1))
        min_distance = np.min(distances)
        distance_2_to_1 += min_distance

    return (distance_1_to_2 + distance_2_to_1) / (len(arr1) + len(arr2))


def get_rigid(src, dst):
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    H = reduce(lambda s, p: s + np.outer(p[0], p[1]), zip(src - src_mean, dst - dst_mean), np.zeros((3,3)))
    u, s, v = np.linalg.svd(H)
    R = v.T.dot(u.T)
    T = - R.dot(src_mean) + dst_mean
    return np.hstack((R, T[:, np.newaxis]))

dir = "/Users/aidanamassalimova/Documents/FinalDataset_2048/fold_0/val"
files = os.listdir(dir+"/predictions")
CD_list =[]
IOU_list =[]
print(len(files))
for file in files:
    print(file)
    heatmaps = np.load(os.path.join(dir, "heatmaps_pred_pointr", file))["arr_0"]
    data = np.load(os.path.join(dir, "predictions", file))["arr_0"]
    pcd_complete = o3d.io.read_point_cloud(os.path.join(dir, "complete","10102023", file[:-4]+".pcd"))
    pcd_partial = o3d.io.read_point_cloud(os.path.join(dir, "partial","10102023", file[:-4]+"/00.pcd"))

    pcd_complete_pv = pv.PolyData(np.asarray(pcd_complete.points))
    pcd_partial = pv.PolyData(np.asarray(pcd_partial.points))
    CD = chamfer_distance(data.squeeze(0), np.asarray(pcd_complete.points))

    CD_list.append(CD)

    planning = np.load(os.path.join(dir, "planning/10102023", file))["arr_0"]

    cloud1 = pv.PolyData(data.squeeze(0))

    cloud2 = pv.PolyData(data.squeeze(0))
    cloud3 = pv.PolyData(data.squeeze(0))
    cloud4 = pv.PolyData(data.squeeze(0))


    cloud1.point_data['Prediction'] = heatmaps[:, 0]
    cloud2.point_data['Prediction'] = heatmaps[:, 1]
    cloud3.point_data['Prediction'] = heatmaps[:, 2]
    cloud4.point_data['Prediction'] = heatmaps[:, 3]


    cmap = 'jet'  # You can choose any colormap you prefer
    colors1 = cloud1.point_data['Prediction']
    colors2 = cloud2.point_data['Prediction']
    colors3 = cloud3.point_data['Prediction']
    colors4 = cloud4.point_data['Prediction']


    # Check if any of the points have a red channel value greater than the threshold
    red_points_mask1 = np.where(colors1 >= 0.75 * np.max(colors1))
    red_points_mask3 = np.where(colors3 >= 0.75 * np.max(colors3))

    red_points_mask2 = np.where(colors2 >= 0.5)
    red_points_mask4 = np.where(colors4 >= 0.5)



    red_cloud1 = cloud1.extract_points(red_points_mask1)
    red_cloud3 = cloud3.extract_points(red_points_mask3)
    red_cloud2 = cloud2.extract_points(red_points_mask2)
    red_cloud4 = cloud4.extract_points(red_points_mask4)
    # Build KD tree for fast nearest neighbor search



    regression_point_num = 4
    shape = data.squeeze(0)
    Heatmap = heatmaps

    Heatmap_sort = np.sort(Heatmap, 0)
    sortIdx = np.argsort(Heatmap, 0)
    ### Select r points with maximum values on each heatmap ###
    shape_sort_select = np.array([shape[sortIdx[-regression_point_num:, ld]] for ld in range(Heatmap.shape[1])])
    Heatmap_sort_select = np.array(
        [Heatmap[sortIdx[-regression_point_num:, ld], ld] for ld in range(Heatmap.shape[1])]).reshape(-1,
                                                                                                      regression_point_num,
                                                                                                      1)

    shape_sort_select_rep = np.expand_dims(shape_sort_select, axis=-1).repeat(regression_point_num, axis=-1)
    shape2_exp_eer = shape_sort_select_rep.transpose(0, 1, 3, 2) - shape_sort_select_rep.transpose(0, 3, 1, 2)
    ### Compute the distance matrix ###
    D_Matrix = np.linalg.norm(shape2_exp_eer, axis=3)
    Heatmap_weight = Heatmap_sort_select.repeat(regression_point_num, axis=-1)
    Distance_matrix = D_Matrix
    ### Apply MDS to D_Matrix to obtain a dimension-degraded version of local shape ###
    mds = MDS(n_components=2, dissimilarity='precomputed')
    shape_MDS = np.array([mds.fit_transform(Distance_matrix[i]) for i in range(Heatmap.shape[1])])
    shape_MDS = np.concatenate((shape_MDS, np.zeros((Heatmap.shape[1], regression_point_num, 1))), axis=2)
    landmark2D = np.sum(Heatmap_sort_select.repeat(3, axis=2) * shape_MDS, axis=1) / Heatmap_sort_select.sum(1)
    N = 4
    neigh = NearestNeighbors(n_neighbors=N)
    IDX = []
    for i in range(Heatmap.shape[1]):
        neigh.fit(shape_MDS[i])
        IDX_ = neigh.kneighbors(landmark2D[i].reshape(1, -1))[1]
        IDX.append(IDX_)
    IDX = np.array(IDX)

    shape_ext = np.array(
        [shape_MDS[i, IDX[i], :].reshape(-1, 3) - landmark2D[i].reshape(1, -1).repeat(N, axis=0) for i in
         range(Heatmap.shape[1])])
    shape_ext_T = np.array([shape_sort_select[i, IDX[i], :] for i in range(Heatmap.shape[1])]).reshape(-1, N, 3)
    ### shape Centralization and Scale uniformization ###
    w1 = shape_ext - np.repeat(shape_ext.mean(1, keepdims=True), N, axis=1)
    w2 = shape_ext_T - np.repeat(shape_ext_T.mean(1, keepdims=True), N, axis=1)
    w1 = np.linalg.norm(w1.reshape(Heatmap.shape[1], -1), axis=1).reshape(-1, 1, 1)
    w2 = np.linalg.norm(w2.reshape(Heatmap.shape[1], -1), axis=1).reshape(-1, 1, 1)
    shape_ext = shape_ext * w2 / w1
    ### Get the 3D landmark coordinates after registration ###
    landmark3D = np.array([get_rigid(shape_ext[i], shape_ext_T[i])[:, 3] for i in range(Heatmap.shape[1])])

    ep1 = landmark3D[0]
    ep2 = landmark3D[2]


    mp1 = [np.average(red_cloud2.points[:, 0], weights=colors2[red_points_mask2]),
           np.average(red_cloud2.points[:, 1], weights=colors2[red_points_mask2]),
           np.average(red_cloud2.points[:, 2], weights=colors2[red_points_mask2])]
    mp2 = [np.average(red_cloud4.points[:, 0], weights=colors4[red_points_mask4]),
           np.average(red_cloud4.points[:, 1], weights=colors4[red_points_mask4]),
           np.average(red_cloud4.points[:, 2], weights=colors4[red_points_mask4])]






    # Calculate the direction vector of the line (from start_point to the center of the point cloud)
    direction_vector1 = mp1 - ep1
    direction_vector2 = mp2 - ep2

    # Normalize the direction vector
    direction_vector1 /= np.linalg.norm(direction_vector1)
    direction_vector2 /= np.linalg.norm(direction_vector2)
    best_fit_cylinder = BestFitCylinder(Points(np.asarray((cloud1+cloud2).points)))

    cylinder1 = pv.Cylinder(center=mp1, radius=1.5, height=200, direction=direction_vector1)
    cylinder2 = pv.Cylinder(center=mp2, radius=1.5, height=200, direction=direction_vector2)

    direction_vector1_gt = planning[1]-planning[0]
    direction_vector2_gt = planning[3]-planning[2]

    direction_vector1_gt /= np.linalg.norm(direction_vector1_gt)
    direction_vector2_gt /= np.linalg.norm(direction_vector2_gt)

    cylinder1_gt = pv.Cylinder(center=planning[1], radius=1.5, height=200, direction=direction_vector1_gt)
    cylinder2_gt = pv.Cylinder(center=planning[3], radius=1.5, height=200, direction=direction_vector2_gt)

    MP1 = pv.Sphere(center=mp1, radius=1.5)
    MP2 = pv.Sphere(center=mp2, radius=1.5)

    clipped = pcd_complete_pv.clip('x', invert=True)
    iou = calculate_iou(np.asarray(pcd_complete.points), np.asarray(cloud1.points))
    IOU_list.append(iou)
    p = pv.Plotter()
   # p.add_mesh(pv.Box(bounds=clipped.bounds), color="blue", opacity=0.3)
   # p.add_mesh(pv.Box(bounds=pcd_partial.bounds), color="red", opacity=0.3)

    p.add_mesh(cloud1, point_size=5, color="blue")
    p.add_mesh(cylinder1,  color="blue")
    p.add_mesh(cylinder2,  color="blue")

   # p.add_mesh(pcd_complete_pv, color="yellow", point_size=5)
    p.add_mesh(pcd_partial, color="red", point_size=5)
   

    p.show()
print(np.mean(CD_list))
print(np.mean(IOU_list))




