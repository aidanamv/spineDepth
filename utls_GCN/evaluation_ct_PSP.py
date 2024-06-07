import os
import open3d as o3d
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from functools import reduce
import numpy as np
import pyvista as pv
import pandas as pd
import trimesh
def estimate_point(start_point, stop_point, t):
    # Ensure t is between 0 and 1
    t = np.clip(t, 0, 1)
    # Calculate the estimated point
    estimated_point = (1 - t) * start_point + t * stop_point
    return estimated_point
def calculate_rotation_matrix(direction_vector):
    z_axis = np.array([0, 0, 1])

    rotation_axis = np.cross(z_axis, direction_vector)
    rotation_angle = np.arccos(np.dot(z_axis, direction_vector))

    # Create the rotation matrix manually using the axis-angle representation
    c = np.cos(rotation_angle)
    s = np.sin(rotation_angle)
    t = 1 - c
    x, y, z = rotation_axis
    rotation_matrix = np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c]
    ])
    return rotation_matrix



def calculate_rotation_angle(vector1, vector2):
    # Ensure the input vectors are numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Normalize the vectors
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)

    # Calculate the dot product
    dot_product = np.dot(vector1, vector2)

    # Use arccosine to find the angle in radians
    angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees
def euclidean_distance(point1, point2):
    # Ensure the input points are numpy arrays
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate the Euclidean distance
    distance = np.linalg.norm(point2 - point1)

    return distance

def get_rigid(src, dst):
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    H = reduce(lambda s, p: s + np.outer(p[0], p[1]), zip(src - src_mean, dst - dst_mean), np.zeros((3,3)))
    u, s, v = np.linalg.svd(H)
    R = v.T.dot(u.T)
    T = - R.dot(src_mean) + dst_mean
    return np.hstack((R, T[:, np.newaxis]))

def landmark_detection(data, heatmaps):
    regression_point_num = 4
    shape = data
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

    return landmark3D

EP_error_left = []
Traj_error_left = []
EP_error_right = []
Traj_error_right = []
categories_left = []
categories_right = []
specimens = []
levels =[]
intersections = []
dir = "/Users/aidanamassalimova/Documents/MICCAI/CT_based/2048/test"
files = np.load(dir+"/vertices.npz")["arr_0"]
labels = np.load(dir+"/labels.npz")["arr_0"]
mesh_dir = "/Users/aidanamassalimova/Documents/MICCAI/registered_stls"

for el,file in enumerate(files):
    heatmaps = np.load(os.path.join(dir, "heatmaps_pred.npz"))["arr_0"]
    data = file
    label = labels[el]

    results = label.split("_")
    vertebrae = results[0] + "_" + results[1]+ "_" +results[-1]
    specimen = results[0] + "_" + results[1]
    level = results[-1][:-4]
    specimens.append(specimen)
    levels.append(level)

    planning = np.load(os.path.join(dir, "landmarks.npz"))["arr_0"][el]

    cloud1 = pv.PolyData(data)
    cloud2 = pv.PolyData(data)
    cloud3 = pv.PolyData(data)
    cloud4 = pv.PolyData(data)

    gt_o3d = o3d.geometry.PointCloud()
    gt_o3d.points =o3d.utility.Vector3dVector(files[el])
    vertebrae_mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, vertebrae))
    vertebrae_mesh_pcd = vertebrae_mesh.sample_points_uniformly(number_of_points=10000)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        vertebrae_mesh_pcd, gt_o3d, 10, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000000))

    vertebrae_mesh.transform(reg_p2p.transformation)
    vertebrae_mesh.compute_vertex_normals()
    vertebrae_mesh_pcd.transform(reg_p2p.transformation)



    cloud1.point_data['Prediction'] = heatmaps[el,:, 0]
    cloud2.point_data['Prediction'] = heatmaps[el,:, 1]
    cloud3.point_data['Prediction'] = heatmaps[el,:, 2]
    cloud4.point_data['Prediction'] = heatmaps[el,:, 3]

    colors1 = cloud1.point_data['Prediction']
    colors2 = cloud2.point_data['Prediction']
    colors3 = cloud3.point_data['Prediction']
    colors4 = cloud4.point_data['Prediction']

    # Check if any of the points have a red channel value greater than the threshold
    red_points_mask1 = np.where(colors1 >= 0.75 * np.max(colors1))
    red_points_mask3 = np.where(colors3 >= 0.75 * np.max(colors3))

    red_points_mask2 = np.where(colors2 >= 0.75 * np.max(colors2))
    red_points_mask4 = np.where(colors4 >= 0.75 * np.max(colors4))

    red_cloud1 = cloud1.extract_points(red_points_mask1)
    red_cloud3 = cloud3.extract_points(red_points_mask3)
    red_cloud2 = cloud2.extract_points(red_points_mask2)
    red_cloud4 = cloud4.extract_points(red_points_mask4)

    ep1 = np.mean(red_cloud1.points, axis=0)
    ep2 = np.mean(red_cloud3.points, axis=0)
    mp1 = np.mean(red_cloud2.points, axis=0)
    mp2 = np.mean(red_cloud4.points, axis=0)

    landmarks3D = landmark_detection(data, heatmaps[el,:, :])
    #
   # ep1 = landmarks3D[0]
   # ep2 = landmarks3D[2]
    # mp1 = [np.average(red_cloud2.points[:, 0], weights=colors2[red_points_mask2]),
    #        np.average(red_cloud2.points[:, 1], weights=colors2[red_points_mask2]),
    #        np.average(red_cloud2.points[:, 2], weights=colors2[red_points_mask2])]
    # mp2 = [np.average(red_cloud4.points[:, 0], weights=colors4[red_points_mask4]),
    #        np.average(red_cloud4.points[:, 1], weights=colors4[red_points_mask4]),
     #      np.average(red_cloud4.points[:, 2], weights=colors4[red_points_mask4])]


    # Calculate the direction vector of the line (from start_point to the center of the point cloud)
    direction_vector1 = mp1 - ep1
    direction_vector2 = mp2 - ep2

    # Normalize the direction vector
    direction_vector1 /= np.linalg.norm(direction_vector1)
    direction_vector2 /= np.linalg.norm(direction_vector2)

    direction_vector1_gt = planning[1]-planning[0]
    direction_vector2_gt = planning[3]-planning[2]

    direction_vector1_gt /= np.linalg.norm(direction_vector1_gt)
    direction_vector2_gt /= np.linalg.norm(direction_vector2_gt)


    EP_left_offset = euclidean_distance(planning[0], ep1)
    EP_right_offset = euclidean_distance(planning[2], ep2)

    EP_error_left.append(EP_left_offset)
    EP_error_right.append(EP_right_offset)


    traj_left_offset = calculate_rotation_angle(direction_vector1, direction_vector1_gt)
    traj_right_offset = calculate_rotation_angle(direction_vector2, direction_vector2_gt)

    Traj_error_left.append(traj_left_offset)
    Traj_error_right.append(traj_right_offset)

    cylinder_mesh1 = o3d.geometry.TriangleMesh.create_cylinder(radius=2.5, height=300)
    cylinder_mesh2 = o3d.geometry.TriangleMesh.create_cylinder(radius=2.5, height=300)
    cylinder_mesh1.translate(mp1, relative =False)
    cylinder_mesh2.translate(mp2, relative =False)
    # Normalize the direction vector
    cylinder_mesh1.rotate(calculate_rotation_matrix(direction_vector1), center=cylinder_mesh1.get_center())
    cylinder_mesh2.rotate(calculate_rotation_matrix(direction_vector2), center=cylinder_mesh2.get_center())
    # Convert Open3D mesh to trimesh objects
    mesh1_trimesh = trimesh.Trimesh(np.asarray(cylinder_mesh1.vertices), np.asarray(cylinder_mesh1.triangles))
    mesh2_trimesh = trimesh.Trimesh(np.asarray(cylinder_mesh2.vertices), np.asarray(cylinder_mesh2.triangles))
    mesh3_trimesh = trimesh.Trimesh(np.asarray(vertebrae_mesh.vertices), np.asarray(vertebrae_mesh.triangles))


    intersection3 = mesh1_trimesh.intersection(mesh2_trimesh)

    vertices3 = np.asarray(intersection3.vertices)
    if len(vertices3) <= 0:
        intersections.append(1)
    else:
        intersection = o3d.geometry.PointCloud()
        intersection.points = o3d.utility.Vector3dVector(vertices3)
        intersection.paint_uniform_color([0,0,1])
        vertebrae_mesh_pcd.paint_uniform_color([0,1,0])
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(vertebrae_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        query_point = o3d.core.Tensor([vertices3[0,:]], dtype=o3d.core.Dtype.Float32)
        signed_distances = scene.compute_distance(query_point).numpy()
        if signed_distances>=0:
            intersections.append(2)
        else:
            intersections.append(0)

    sp1 = ep1 + 40 * direction_vector1
    sp2 = ep2 + 40 * direction_vector2

    height1 = 40
    height2 = 40


    new_mp1 = estimate_point(ep1, sp1, 0.5)
    new_mp2 = estimate_point(ep2, sp2, 0.5)



    cylinder_mesh1_new = o3d.geometry.TriangleMesh.create_cylinder(radius=2.5, height=height1)
    cylinder_mesh2_new = o3d.geometry.TriangleMesh.create_cylinder(radius=2.5, height=height2)
    cylinder_mesh1_new.translate(new_mp1, relative=False)
    cylinder_mesh2_new.translate(new_mp2, relative=False)
    # Normalize the direction vector

    cylinder_mesh1_new.rotate(calculate_rotation_matrix(direction_vector1), center=cylinder_mesh1_new.get_center())
    cylinder_mesh2_new.rotate(calculate_rotation_matrix(direction_vector2), center=cylinder_mesh2_new.get_center())

    mesh1 = o3d.t.geometry.TriangleMesh.from_legacy(cylinder_mesh1_new)
    scene1 = o3d.t.geometry.RaycastingScene()
    _ = scene1.add_triangles(mesh1)
    query_point1 = o3d.core.Tensor([np.asarray(vertebrae_mesh_pcd.points)], dtype=o3d.core.Dtype.Float32)
    signed_distances1 = scene1.compute_signed_distance(query_point1)

    mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(cylinder_mesh2_new)
    scene2 = o3d.t.geometry.RaycastingScene()
    _ = scene2.add_triangles(mesh2)
    query_point2 = o3d.core.Tensor([np.asarray(vertebrae_mesh_pcd.points)], dtype=o3d.core.Dtype.Float32)
    signed_distances2 = scene2.compute_signed_distance(query_point2)
    min_cylinder1 = np.min(signed_distances1.numpy())
    min_cylinder2 = np.min(signed_distances2.numpy())
    ind1 = np.argmin(signed_distances1.numpy())
    ind2 = np.argmin((signed_distances2.numpy()))

    sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=2.5)
    sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=2.5)
    sphere1.translate(np.asarray(vertebrae_mesh_pcd.points)[ind1, :], relative=False)
    sphere2.translate(np.asarray(vertebrae_mesh_pcd.points)[ind2, :], relative=False)
    dist1 = np.linalg.norm(np.asarray(vertebrae_mesh_pcd.points)[ind1, :] - ep1)
    dist2 = np.linalg.norm(np.asarray(vertebrae_mesh_pcd.points)[ind2, :] - ep2)
    print(min_cylinder1, min_cylinder2)
    sphere1.translate(np.asarray(vertebrae_mesh_pcd.points)[ind1, :], relative=False)
    sphere2.translate(np.asarray(vertebrae_mesh_pcd.points)[ind2, :], relative=False)

    if min_cylinder1 >= 0 or dist1 <= 5:
        categories_left.append(4)
    elif min_cylinder1 >= -2 and min_cylinder1 < 0:

        categories_left.append(3)
    elif min_cylinder1 >= -4 and min_cylinder1 < -2:
        categories_left.append(2)

    elif min_cylinder1 >= -6 and min_cylinder1 < -4:
        categories_left.append(1)

    elif min_cylinder1 < -6:
        categories_left.append(0)

    if min_cylinder2 >= 0 or dist1 <= 5:
        categories_right.append(4)
    elif min_cylinder2 >= -2 and min_cylinder2 < 0:
        categories_right.append(3)


    elif min_cylinder2 >= -4 and min_cylinder2 < -2:
        categories_right.append(2)

    elif min_cylinder2 >= -6 and min_cylinder2 < -4:
        categories_right.append(1)

    elif min_cylinder2 < -6:
        categories_right.append(0)


data ={
    "specimen" : specimens,
    "levels": levels,
    "EP_error_left" : EP_error_left
    , "Traj_error_left" : Traj_error_left
    , "EP_error_right" : EP_error_right
    , "Traj_error_right" : Traj_error_right
    , "categories_left" : categories_left
    , "categories_right" : categories_right
    , "intersections": intersections

}

df = pd.DataFrame(data)
df.to_csv("../evaluations/ct_2048.csv", index=False)
print(df.describe())





