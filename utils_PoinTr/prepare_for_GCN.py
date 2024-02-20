import numpy as np
import open3d as o3d
import os
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
def Gaussian_Heatmap(Distance, sigma):
    D2 = Distance * Distance
    S2 = 2.0 * sigma * sigma
    Exponent = D2 / S2
    heatmap = np.exp(-Exponent)
    return heatmap

for fold in range(0,9):
    dir ="/Users/aidanamassalimova/Documents/FinalDataset_2048/fold_{}/val".format(fold)
    save_train = "/Users/aidanamassalimova/Documents/planning_dataset_from_pointr/2048/fold_{}/val".format(fold)
    if not os.path.exists(save_train):
        os.makedirs(save_train)
    files= os.listdir(os.path.join(dir,"predictions"))
    heatmaps =[]
    vertices = []
    landmarks = []
    vertices_gt =[]
    labels =[]
    rgbd =[]

    for el,file in enumerate(files):
        file = file[:-4]
        labels.append(file)
        pcd_partial = o3d.io.read_point_cloud(os.path.join(dir,"partial","10102023", file, "00.pcd"))
        # pcd_complete = o3d.io.read_point_cloud(os.path.join(dir,"complete","10102023", file+".pcd"))
        # landmark=np.load(os.path.join(dir, "planning", "10102023",file+".npz"))["arr_0"]
        # prediction_np = np.load(os.path.join(dir, "predictions", file+".npz"))["arr_0"].squeeze(0)
        # complete_np =np.asarray(pcd_complete.points)
        # pcd_prediction = o3d.geometry.PointCloud()
        # pcd_prediction.points = o3d.utility.Vector3dVector(np.asarray(prediction_np))
        #
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     pcd_prediction, pcd_complete, 10, np.identity(4),
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000000))
        #
        # pcd_prediction.transform(reg_p2p.transformation)

        # shape_sample_pred = prediction_np.reshape(prediction_np.shape[0], 1, prediction_np.shape[1]).repeat(landmark.shape[0], axis=1)
        # shape_sample_complete = complete_np.reshape(complete_np.shape[0], 1, complete_np.shape[1]).repeat(landmark.shape[0], axis=1)
        #
        # Euclidean_distance_pred = np.linalg.norm((shape_sample_pred - landmark), axis=2)
        # Euclidean_distance_complete = np.linalg.norm((shape_sample_complete - landmark), axis=2)
        #
        # Heat_data_pred = Gaussian_Heatmap(Euclidean_distance_pred, 10)
        # Heat_data_complete = Gaussian_Heatmap(Euclidean_distance_complete, 10)
        #
        # #
        # # grayscale_array =Heat_data_complete[:, 0].reshape(4096,1)
        # # grayscale_array = grayscale_array.reshape(-1)  # Flatten the array
        # # cmap = plt.get_cmap('jet')
        # # normed_grayscale = (grayscale_array - np.min(grayscale_array)) / (
        # #             np.max(grayscale_array) - np.min(grayscale_array))
        # # rgb_array = cmap(normed_grayscale)[:, :3]  # Exclude alpha channel and keep only RGB values
        # #
        # # pcd_complete.colors = o3d.utility.Vector3dVector(rgb_array)
        #
        #
        # # Check if any of the points have a red channel value greater than the threshold
        # red_points_mask1 = np.where(Heat_data_complete[:, 0] >= 0.5)
        # red_points_mask3 = np.where(Heat_data_complete[:, 2] >= 0.5)
        #
        # red_points_mask2 = np.where( Heat_data_complete[:, 1] >= 0.5)
        # red_points_mask4 = np.where( Heat_data_complete[:, 3] >= 0.5)
        # extracted_points1 = np.asarray(pcd_complete.points)[red_points_mask1]
        # extracted_points2 = np.asarray(pcd_complete.points)[red_points_mask2]
        # extracted_points3 = np.asarray(pcd_complete.points)[red_points_mask3]
        # extracted_points4 = np.asarray(pcd_complete.points)[red_points_mask4]
        #
        #
        #
        # closest_pts_mp_left = []
        # closest_pts_mp_right = []
        # closest_pts_ep_left = []
        # closest_pts_ep_right = []

        # for i in range(len(extracted_points1)):
        #     selected_point = extracted_points1[i,:]
        #     target_pcd = pcd_prediction.points
        #     kdtree = KDTree(target_pcd)
        #     # Query the KD-tree to find the index of the closest point in the point cloud to the given point
        #     closest_point_index = kdtree.query(selected_point)[1]
        #     # Calculate the distance between the closest point and the given point
        #     closest_distance = np.linalg.norm(target_pcd[closest_point_index] - selected_point)
        #     closest_pts_ep_left.append(target_pcd[closest_point_index])
        #
        # for i in range(len(extracted_points2)):
        #     selected_point = extracted_points2[i,:]
        #     target_pcd = pcd_prediction.points
        #     kdtree = KDTree(target_pcd)
        #     # Query the KD-tree to find the index of the closest point in the point cloud to the given point
        #     closest_point_index = kdtree.query(selected_point)[1]
        #     # Calculate the distance between the closest point and the given point
        #     closest_distance = np.linalg.norm(target_pcd[closest_point_index] - selected_point)
        #     closest_pts_mp_left.append(target_pcd[closest_point_index])
        #
        # for i in range(len(extracted_points3)):
        #     selected_point = extracted_points3[i, :]
        #     target_pcd = pcd_prediction.points
        #     kdtree = KDTree(target_pcd)
        #     # Query the KD-tree to find the index of the closest point in the point cloud to the given point
        #     closest_point_index = kdtree.query(selected_point)[1]
        #     # Calculate the distance between the closest point and the given point
        #     closest_distance = np.linalg.norm(target_pcd[closest_point_index] - selected_point)
        #     closest_pts_ep_right.append(target_pcd[closest_point_index])
        #
        # for i in range(len(extracted_points4)):
        #     selected_point = extracted_points4[i,:]
        #     target_pcd = pcd_prediction.points
        #     kdtree = KDTree(target_pcd)
        #     # Query the KD-tree to find the index of the closest point in the point cloud to the given point
        #     closest_point_index = kdtree.query(selected_point)[1]
        #     # Calculate the distance between the closest point and the given point
        #     closest_distance = np.linalg.norm(target_pcd[closest_point_index] - selected_point)
        #     closest_pts_mp_right.append(target_pcd[closest_point_index])
        #
        #
        # mp1 = np.mean(closest_pts_mp_left, axis=0)
        # mp2 = np.mean(closest_pts_mp_right, axis=0)
        # ep1 = np.mean(closest_pts_ep_left, axis=0)
        # ep2 = np.mean(closest_pts_ep_right, axis=0)
        #
        #
        #
        # landmark_new = np.vstack([ep1, mp1, ep2, mp2])
        # shape_sample_pred = prediction_np.reshape(prediction_np.shape[0], 1, prediction_np.shape[1]).repeat(landmark_new.shape[0], axis=1)
        # Euclidean_distance_pred = np.linalg.norm((shape_sample_pred - landmark_new), axis=2)
        # Heat_data_pred = Gaussian_Heatmap(Euclidean_distance_pred, 10)
        #
        # heatmaps.append(Heat_data_pred)
        # vertices.append(prediction_np)
        # landmarks.append(landmark_new)
        # vertices_gt.append(pcd_complete.points)
        rgbd.append(pcd_partial.points)




    # np.savez(os.path.join(save_train,"vertices.npz"), vertices)
    # np.savez(os.path.join(save_train,"heatmaps.npz"), heatmaps)
    # np.savez(os.path.join(save_train,"landmarks.npz"), landmarks)
    # np.savez(os.path.join(save_train,"vertices_gt.npz"), vertices_gt)
    # np.savez(os.path.join(save_train,"labels.npz"), labels)
    np.savez(os.path.join(save_train,"rgbd.npz"), rgbd)
    print("fold {} done".format(fold))

