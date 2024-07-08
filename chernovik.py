import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

dir = "/Volumes/SpineDepth/YOLO/labeled_pcds"
depth_image = cv2.imread('/Volumes/SpineDepth/PoinTr_dataset/segmented_spinedepth_new/Specimen_3/recording_1/cam_0/frame_1/depth.png', cv2.IMREAD_UNCHANGED)

# Check if the image is loaded properly
if depth_image is None:
    raise ValueError("Image not loaded properly. Please check the file path.")

# Normalize the depth image to the range [0, 1]
depth_image_normalized = cv2.normalize(depth_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Optionally convert the normalized image to [0, 255] range for visualization
depth_image_normalized_255 = (depth_image_normalized * 255).astype(np.uint8)

# Save the normalized image (optional)
cv2.imwrite('normalized_depth_image.png', depth_image_normalized_255)

pcd_np = np.load(os.path.join(dir,"Specimen_3_recording_0_cam_0_frame_0.npz"))["arr_0"]
labels = pcd_np[:,6]
red_points_indices1 = np.where(labels == 0)[0]
red_points_indices2 = np.where(labels == 1)[0]
red_points_indices3 = np.where(labels == 2)[0]
red_points_indices4 = np.where(labels == 3)[0]
red_points_indices5 = np.where(labels == 4)[0]
red_points_indices6 = np.where(labels == 5)[0]

colors = np.zeros((len(labels),3))
colors[red_points_indices1,:] = [0,0,0]
colors[red_points_indices2,:] = [1,0,0]
colors[red_points_indices3,:] = [0,1,0]
colors[red_points_indices4,:] = [0,0,1]
colors[red_points_indices5,:] = [1,1,0]
colors[red_points_indices6,:] = [1,0,1]


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_np[:,0:3])
pcd.colors = o3d.utility.Vector3dVector(pcd_np[:,3:6])

print(len(pcd.points))
pcd  = pcd.uniform_down_sample(every_k_points = int(len(labels)/10000))
o3d.visualization.draw_geometries([pcd])