import pyvista as pv
import os
import numpy as np
import open3d as o3d
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from functools import reduce


import numpy as np
import pyvista as pv


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
EP_error_all = []
Traj_error_all = []
for fold in range(0,8):
    dir = "/Users/aidanamassalimova/Documents/planning_dataset_from_pointr/2048/fold_{}/val".format(fold)

    files = np.load(dir+"/vertices.npz")["arr_0"]
    #change for pointr dataset
    files_gt = np.load(dir+"/vertices_gt.npz")["arr_0"]
    labels = np.load(dir+"/labels.npz")["arr_0"]
    mesh_dir = "/Users/aidanamassalimova/Documents/Stereo_navigation/GCN/CT_based_dataset/Aidana_planning_based/registered_stls"

    EP_error_list = []
    Traj_error_list = []

    category_a =0
    category_b =0
    category_c =0
    category_d =0
    category_e =0

    levels = ["L1", "L2", "L3", "L4", "L5"]


    for el,file in enumerate(files):
        heatmaps = np.load(os.path.join(dir, "heatmaps_pred.npz"))["arr_0"]
        data = file
        label = labels[el]
        results = label.split("_")
        if results[-1] in levels:
            levels.remove(results[-1])



            vertebrae = results[0] + "_" + results[1]+ "_" +results[-1] +".stl"

            planning = np.load(os.path.join(dir, "landmarks.npz"))["arr_0"][el]

            cloud1 = pv.PolyData(data)
            cloud2 = pv.PolyData(data)
            cloud3 = pv.PolyData(data)
            cloud4 = pv.PolyData(data)

            gt = pv.PolyData(files_gt[el])
            gt_o3d = o3d.geometry.PointCloud()
            gt_o3d.points =o3d.utility.Vector3dVector(files_gt[el])
            vertebrae_mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, vertebrae))
            vertebrae_mesh_pcd = vertebrae_mesh.sample_points_uniformly(number_of_points=2048)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                vertebrae_mesh_pcd, gt_o3d, 10, np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000000))

            vertebrae_mesh.transform(reg_p2p.transformation)
            vertebrae_mesh_pcd.transform(reg_p2p.transformation)



            cloud1.point_data['Prediction'] = heatmaps[el,:, 0]
            cloud2.point_data['Prediction'] = heatmaps[el,:, 1]
            cloud3.point_data['Prediction'] = heatmaps[el,:, 2]
            cloud4.point_data['Prediction'] = heatmaps[el,:, 3]

            cmap = 'jet'  # You can choose any colormap you prefer
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
            # Build KD tree for fast nearest neighbor search

            regression_point_num = 4
            shape = data
            Heatmap = heatmaps[el,:,:]

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

            ep1 = np.mean(red_cloud1.points, axis=0)
            ep2 = np.mean(red_cloud3.points, axis=0)
            mp1 = np.mean(red_cloud2.points, axis=0)
            mp2 = np.mean(red_cloud4.points, axis=0)


            # Calculate the direction vector of the line (from start_point to the center of the point cloud)
            direction_vector1 = mp1 - ep1
            direction_vector2 = mp2 - ep2

            # Normalize the direction vector
            direction_vector1 /= np.linalg.norm(direction_vector1)
            direction_vector2 /= np.linalg.norm(direction_vector2)
           # best_fit_cylinder = BestFitCylinder(Points(np.asarray((cloud1+cloud3).points)))
            height1 = np.linalg.norm(mp1 - ep1)
            height2 = np.linalg.norm(mp2 - ep2)

            cylinder1 = pv.Cylinder(center=mp1, radius=2.5, height=200, direction=direction_vector1)
            cylinder2 = pv.Cylinder(center=mp2, radius=2.5, height=200, direction=direction_vector2)

            direction_vector1_gt = planning[1]-planning[0]
            direction_vector2_gt = planning[3]-planning[2]

            direction_vector1_gt /= np.linalg.norm(direction_vector1_gt)
            direction_vector2_gt /= np.linalg.norm(direction_vector2_gt)

            cylinder1_gt = pv.Cylinder(center=planning[1], radius=2.5, height=200, direction=direction_vector1_gt)
            cylinder2_gt = pv.Cylinder(center=planning[3], radius=2.5, height=200, direction=direction_vector2_gt)



            EP_left_offset = euclidean_distance(planning[0], ep1)
            EP_right_offset = euclidean_distance(planning[2], ep2)
            EP_error_list.append(EP_left_offset)
            EP_error_list.append(EP_right_offset)
            EP_error_all.append(EP_left_offset)
            EP_error_all.append(EP_right_offset)



            traj_left_offset = calculate_rotation_angle(direction_vector1, direction_vector1_gt)
            traj_right_offset = calculate_rotation_angle(direction_vector2, direction_vector2_gt)
            Traj_error_list.append(traj_left_offset)
            Traj_error_list.append(traj_right_offset)
            Traj_error_all.append(traj_left_offset)
            Traj_error_all.append(traj_right_offset)

            cylinder_mesh1 = o3d.geometry.TriangleMesh.create_cylinder(radius=2.5, height=2*height1)
            cylinder_mesh2 = o3d.geometry.TriangleMesh.create_cylinder(radius=2.5, height=2*height2)
            cylinder_mesh1.translate(mp1, relative =False)
            cylinder_mesh2.translate(mp2, relative =False)
            # Normalize the direction vector

            cylinder_mesh1.rotate(calculate_rotation_matrix(direction_vector1), center=cylinder_mesh1.get_center())
            cylinder_mesh2.rotate(calculate_rotation_matrix(direction_vector2), center=cylinder_mesh2.get_center())

            cylinder1_pcd = cylinder_mesh1.sample_points_uniformly(number_of_points=400)
            cylinder2_pcd = cylinder_mesh2.sample_points_uniformly(number_of_points=400)

            p = pv.Plotter()
            p.add_mesh(cylinder1, color='blue')
            p.add_mesh(cylinder2, color="blue")
            p.add_mesh(cloud1, color="green")
            p.show()

            #o3d.visualization.draw_geometries([cylinder1_pcd, vertebrae_mesh])

            for point in cylinder1_pcd.points:
                mesh = o3d.t.geometry.TriangleMesh.from_legacy(vertebrae_mesh)
                # Create a scene and add the triangle mesh
                scene = o3d.t.geometry.RaycastingScene()
                _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
                query_point = o3d.core.Tensor(gt.points, dtype=o3d.core.Dtype.Float32)
                signed_distances1 = scene.compute_signed_distance(query_point)
            for point in cylinder2_pcd.points:
                mesh = o3d.t.geometry.TriangleMesh.from_legacy(vertebrae_mesh)
                # Create a scene and add the triangle mesh
                scene = o3d.t.geometry.RaycastingScene()
                _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
                query_point = o3d.core.Tensor(gt.points, dtype=o3d.core.Dtype.Float32)
                signed_distances2 = scene.compute_signed_distance(query_point)


            min1=np.min(signed_distances1.numpy())
            min2=np.min(signed_distances2.numpy())
            mins =[min1, min2]
            for min in mins:
                if min >= 0:
                    category_a += 1
                elif min >= -2:
                    category_b += 1
                elif min >= -4 and min<=-2:
                    category_c += 1
                elif min >= -6 and min <= -4:
                    category_d += 1
                elif min <= -6:
                    category_e += 1
        else:
            continue
    print("fold_{}".format(fold))
    print(np.median(EP_error_list))
    print(np.median(Traj_error_all))
    print(category_a)
    print(category_b)
    print(category_c)
    print(category_d)
    print(category_e)






print("min")
print(np.min(EP_error_all))
print(np.min(Traj_error_all))


print("q1")
print(np.percentile(EP_error_all, 25))
print(np.percentile(Traj_error_all, 25))

print("median")
print(np.median(EP_error_all))
print(np.median(Traj_error_all))

print("q3")
print(np.percentile(EP_error_all, 75))
print(np.percentile(Traj_error_all, 75))



print("max")
print(np.max(EP_error_all))
print(np.max(Traj_error_all))





