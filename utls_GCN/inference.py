import os
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import open3d as o3d
from functools import reduce
from sklearn.neighbors import NearestNeighbors

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

def Gaussian_Heatmap(Distance, sigma):
    D2 = Distance * Distance
    S2 = 2.0 * sigma * sigma
    Exponent = D2 / S2
    heatmap = np.exp(-Exponent)
    return heatmap

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
iou_list =[]
dir = "/Users/aidanamassalimova/Documents/MICCAI/PoinTr_based/2048/fold_{}/val".format(0)
vertices = np.load(os.path.join(dir, "vertices.npz"))["arr_0"]
vertices_land = np.load(os.path.join("/Users/aidanamassalimova/Documents/MICCAI/CT_based/2048/test", "vertices.npz"))["arr_0"]
vertices_gt = np.load(os.path.join(dir, "vertices_gt.npz"))["arr_0"]

heatmaps = np.load(os.path.join(dir, "heatmaps.npz"))["arr_0"]
landmarks = np.load(os.path.join("/Users/aidanamassalimova/Documents/MICCAI/CT_based/2048/test", "landmarks.npz"))["arr_0"]
labels_land = np.load(os.path.join("/Users/aidanamassalimova/Documents/MICCAI/CT_based/2048/test", "labels.npz"))["arr_0"].tolist()
labels = np.load(os.path.join(dir, "labels.npz"))["arr_0"]

rgbds = np.load(os.path.join(dir, "rgbd.npz"))["arr_0"]
for el,label in enumerate(labels):
    results = labels[el].split("_")
    if el !=25:
        print(results)
        ind = labels_land.index(str(results[0])+"_"+str(results[1])+"_"+str(results[-1])+".stl")
        print(labels_land[ind])
        mesh = pv.PolyData(pv.read("/Users/aidanamassalimova/Documents/MICCAI/PoinTr_based/stls/{}.stl".format(results[-1])))
        landmark = landmarks[ind]

        shape_sample_complete = vertices_gt[el].reshape(vertices_gt[el].shape[0], 1, vertices_gt[el].shape[1]).repeat(landmark.shape[0], axis=1)
        Euclidean_distance_complete = np.linalg.norm((shape_sample_complete - landmark), axis=2)

        Heat_data_complete = Gaussian_Heatmap(Euclidean_distance_complete, 10)
        pcd = pv.PolyData(vertices[el])
        pcd_gt = pv.PolyData(vertices_gt[el])
        pcd_land = pv.PolyData(vertices_land[ind])

        pcd_rgbd = pv.PolyData(rgbds[el])
        pcd.point_data["Predictions"] = heatmaps[el, :, :]
        red_points_mask1 = np.where(heatmaps[el, :, 0] >= 0.75 * np.max(heatmaps[el, :, 0]))
        red_points_mask2 = np.where(heatmaps[el, :, 1] >= 0.75 * np.max(heatmaps[el, :, 1]))
        red_points_mask3 = np.where(heatmaps[el, :, 2] >= 0.75 * np.max(heatmaps[el, :, 2]))
        red_points_mask4 = np.where(heatmaps[el, :, 3]>=0.75 * np.max(heatmaps[el, :, 3]))

        red_cloud1 = pcd.extract_points(red_points_mask1)
        red_cloud2 = pcd.extract_points(red_points_mask2)
        red_cloud3 = pcd.extract_points(red_points_mask3)
        red_cloud4 = pcd.extract_points(red_points_mask4)
        ep1 = np.mean(red_cloud1.points, axis =0)
        mp1 = np.mean(red_cloud2.points, axis =0)
        ep2 = np.mean(red_cloud3.points, axis =0)
        mp2 = np.mean(red_cloud4.points, axis =0)

        pcd_land_o3d = o3d.geometry.PointCloud()
        pcd_gt_o3d = o3d.geometry.PointCloud()

        pcd_land_o3d.points = o3d.utility.Vector3dVector(vertices_land[ind])
        pcd_gt_o3d.points = o3d.utility.Vector3dVector(vertices_gt[el])

        threshold = 10  # 3cm distance threshold
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_land_o3d, pcd_gt_o3d, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
        pcd_land_o3d.transform(reg_p2p.transformation)
        # o3d.visualization.draw_geometries([pcd_land_o3d, pcd_gt_o3d])
        ep1_gt = np.dot(reg_p2p.transformation, np.append(landmark[0],1))[0:3]
        mp1_gt = np.dot(reg_p2p.transformation,np.append(landmark[1],1))[0:3]
        ep2_gt = np.dot(reg_p2p.transformation, np.append(landmark[2],1))[0:3]
        mp2_gt = np.dot(reg_p2p.transformation, np.append(landmark[3],1))[0:3]



        direction_vector1 = mp1 - ep1
        direction_vector2 = mp2 - ep2

        direction_vector1_gt = mp1_gt - ep1_gt
        direction_vector2_gt = mp2_gt - ep2_gt

        # Normalize the direction vector
        direction_vector1 /= np.linalg.norm(direction_vector1)
        direction_vector2 /= np.linalg.norm(direction_vector2)

        direction_vector1_gt /= np.linalg.norm(direction_vector1_gt)
        direction_vector2_gt /= np.linalg.norm(direction_vector2_gt)
        # best_fit_cylinder = BestFitCylinder(Points(np.asarray((cloud1+cloud3).points)))

        new_mp1 = ep1 + direction_vector1*25
        new_mp2 = ep2 + direction_vector2*25

        new_mp1_gt = ep1_gt+direction_vector1_gt*25
        new_mp2_gt = ep2_gt+direction_vector2_gt*25

        EP_left_offset = euclidean_distance(ep1_gt, ep1)
        EP_right_offset = euclidean_distance(ep2_gt, ep2)

        print(EP_left_offset, EP_right_offset)

        traj_left_offset = calculate_rotation_angle(direction_vector1, direction_vector1_gt)
        traj_right_offset = calculate_rotation_angle(direction_vector2, direction_vector2_gt)
        print(traj_left_offset, traj_right_offset)

        iou = calculate_iou(pcd_rgbd.points, pcd_gt.points)
        print(iou)
        iou_list.append(iou)

        cylinder1 = pv.Cylinder(center=new_mp1, radius=2.5, height=60, direction=direction_vector1)
        cylinder2 = pv.Cylinder(center=new_mp2, radius=2.5, height=60, direction=direction_vector2)
        cylinder1_smaller = pv.Cylinder(center=new_mp1, radius=0.5, height=60, direction=direction_vector1)
        cylinder2_smaller = pv.Cylinder(center=new_mp2, radius=0.5, height=60, direction=direction_vector2)
        cylinder1_gt = pv.Cylinder(center=new_mp1_gt, radius=2.5, height=60, direction=direction_vector1_gt)
        cylinder2_gt = pv.Cylinder(center=new_mp2_gt, radius=2.5, height=60, direction=direction_vector2_gt)
        cylinder1_gt_smaller = pv.Cylinder(center=new_mp1_gt, radius=0.5, height=60, direction=direction_vector1_gt)
        cylinder2_gt_smaller = pv.Cylinder(center=new_mp2_gt, radius=0.5, height=60, direction=direction_vector2_gt)

        p = pv.Plotter()

        colors = [
            (64 / 255, 224 / 255, 208 / 255),  # Green-Blue (Turquoise)
            (127 / 255, 255 / 255, 212 / 255),  # Blue-Green (Aquamarine)
            (210 / 255, 105 / 255, 30 / 255),  # Chocolate
            (101 / 255, 67 / 255, 33 / 255)  # Dark Brown
        ]

        turqouise = "#24B0BA"
        navy_blue = "#2E4A70"
        gold ="#CF8A40"
        pastel_red = '#ffb6c1'

        colors = [
            turqouise,  # Green-Blue (Turquoise)
            navy_blue,  # Blue-Green (Aquamarine)
            gold# Dark Brown
        ]

        cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
        sphere1 = pv.Sphere(center=ep1-10*direction_vector1, radius=2.5)
        sphere2 = pv.Sphere(center=ep2-10*direction_vector2, radius=2.5)
        sphere11 = pv.Sphere(center=ep1, radius=2.5)


        p = pv.Plotter()
        p.add_mesh(pcd_rgbd,color = "orange", point_size=3)
        p.add_mesh(pcd, color = "blue", point_size=5)
        #p.add_mesh(mesh, color="white", opacity=0.2)
        #p.add_mesh(cylinder1_gt, color="green")
        #p.add_mesh(cylinder2_gt, color="green")

        p.add_mesh(cylinder1, color="red")
        p.add_mesh(cylinder2, color="red")
        #
        # p.add_mesh(cylinder1_smaller, color ="red")
        # p.add_mesh(cylinder2_smaller, color ="red")
        p.show()

print(np.mean(iou_list))