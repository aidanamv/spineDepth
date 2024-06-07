# this script is designed to segment the vertebrae from the spine depth dataset
import pyrender
import trimesh
import os
import pyzed.sl as sl
import numpy as np
import cv2
import open3d as o3d
import pandas as pd
import shutil
import time
import matplotlib.pyplot as plt

from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # Use the Tkinter backend
# df_plan=pd.read_csv("output.csv")
start_time = time.time()
dir = "/media/aidana/aidana/SpineDepth"
calib_dest_dir = "/usr/local/zed/settings/"
labels_dir = "/media/aidana/SpineDepth/YOLO/binary_seg"
specimens = os.listdir(dir)
specimens = ["Specimen_10"]
camera_nums = [0, 1]


def concatenate_point_clouds_with_colors(*point_clouds):
    # Initialize lists to hold the concatenated points, colors, and normals
    all_points = []
    all_colors = []
    all_normals = []

    for pc in point_clouds:
        points = np.asarray(pc.points)
        if pc.has_colors():
            colors = np.asarray(pc.colors)
            all_points.append(points)
            all_colors.append(colors)
            if pc.has_normals():
                all_normals.append(np.asarray(pc.normals))

    # Concatenate the arrays
    concatenated_points = np.vstack(all_points)
    concatenated_colors = np.vstack(all_colors)
    concatenated_normals = np.vstack(all_normals) if all_normals else None

    # Create a new PointCloud object with the concatenated data
    concatenated_point_cloud = o3d.geometry.PointCloud()
    concatenated_point_cloud.points = o3d.utility.Vector3dVector(concatenated_points)
    concatenated_point_cloud.colors = o3d.utility.Vector3dVector(concatenated_colors)
    if concatenated_normals is not None:
        concatenated_point_cloud.normals = o3d.utility.Vector3dVector(concatenated_normals)

    return concatenated_point_cloud




for specimen in specimens:
    recordings = len(os.listdir(os.path.join(dir, specimen, "recordings")))

    for recording in range(recordings):
        for camera_num in camera_nums:
            cur_frame = 0


            if specimen == "Specimen_2" and recording >= 32:
                calib_src_dir = os.path.join(dir, specimen, "Calib_b")
            if specimen == "Specimen_5" and recording >= 8:
                calib_src_dir = os.path.join(dir, specimen, "Calib_b")
            if specimen == "Specimen_7" and recording >= 12:
                calib_src_dir = os.path.join(dir, specimen, "Calib_b")
            if specimen == "Specimen_9" and recording >= 12:
                calib_src_dir = os.path.join(dir, specimen, "Calib_b")
            else:
                calib_src_dir = os.path.join(dir, specimen, "Calib")
            shutil.copytree(calib_src_dir, calib_dest_dir, dirs_exist_ok=True)

            video_dir = os.path.join(dir, specimen, "recordings/recording{}/Video_{}.svo".format(recording, camera_num))


            input_type = sl.InputType()
            input_type.set_from_svo_file(video_dir)
            init_params = sl.InitParameters()
            init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)

            zed = sl.Camera()

            status = zed.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                exit()

            runtime_parameters = sl.RuntimeParameters()

            zed_pose = sl.Pose()
            zed_sensors = sl.SensorsData()
            calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
            nb_frames = zed.get_svo_number_of_frames()

            while cur_frame <= 3:
                print("processing specimen: {} recording {} video: {} frame: {}".format(specimen, recording, camera_num,
                                                                                        cur_frame))

                save_data_dir = os.path.join("/media/aidana/SpineDepth/PoinTr_dataset/segmented_spinedepth_new",
                                             specimen, "recording_{}".format(recording),
                                             "cam_{}".format(camera_num), "frame_{}".format(cur_frame))


                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

                    left_camera_intrinsic = calibration_params.left_cam
                    mask1 = np.asarray(Image.open(os.path.join(save_data_dir, 'mask1.png')))
                    mask2 = np.asarray(Image.open(os.path.join(save_data_dir, 'mask2.png')))
                    mask3 = np.asarray(Image.open(os.path.join(save_data_dir, 'mask3.png')))
                    mask4 = np.asarray(Image.open(os.path.join(save_data_dir, 'mask4.png')))
                    mask5 = np.asarray(Image.open(os.path.join(save_data_dir, 'mask5.png')))

                    mask6= np.asarray(Image.open(os.path.join(labels_dir, '{}_recording_{}_cam_{}_frame_{}.png'.format(specimen,recording,camera_num,cur_frame))))
                    # Find the bounding box of the smaller mask
                    # Perform logical AND operation to keep only the pixels that are present in both masks
                    mask1 = np.logical_and(mask1, mask6)
                    mask2 = np.logical_and(mask2, mask6)
                    mask3 = np.logical_and(mask3, mask6)
                    mask4 = np.logical_and(mask4, mask6)
                    mask5 = np.logical_and(mask5, mask6)



                    # Label the region within the big mask but outside the small mask as 2
                    condition1 = np.logical_and(mask6 == 255, mask1 == False)
                    condition2 = np.logical_and(mask6 == 255, mask2 == False)
                    condition3 = np.logical_and(mask6 == 255, mask3 == False)
                    condition4 = np.logical_and(mask6 == 255, mask4 == False)
                    condition5 = np.logical_and(mask6 == 255, mask5 == False)


                    # Extract the region where the condition is True
                    extracted_region = np.where(condition1, mask6, 0)
                    extracted_region = np.where(condition2, extracted_region, 0)
                    extracted_region = np.where(condition3, extracted_region, 0)
                    extracted_region = np.where(condition4, extracted_region, 0)
                    extracted_region = np.where(condition5, extracted_region, 0)




                    # Create figure and axes
                    fig, ax = plt.subplots()

                    # Plot mask1 in blue
                    ax.imshow(mask1, cmap='Blues', alpha=0.3)
                    ax.imshow(mask2, cmap='ocean', alpha=0.3)
                    ax.imshow(mask3, cmap='Greens', alpha=0.3)
                    ax.imshow(mask4, cmap='Purples', alpha=0.3)
                    ax.imshow(mask5, cmap='autumn', alpha=0.3)
                    ax.axis('off')

                    # Plot mask2 in red
                    ax.imshow(extracted_region, cmap='Reds', alpha=0.5)

                    # Set title and show plot

                    plt.show()



                    color_image = np.asarray(Image.open(os.path.join(save_data_dir, 'image.png')))
                    depth_image = np.asarray(Image.open(os.path.join(save_data_dir, 'depth.png')))
                    rgb_mask1 = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
                    rgb_mask2 = np.zeros((mask2.shape[0], mask2.shape[1], 3), dtype=np.uint8)
                    rgb_mask3 = np.zeros((mask3.shape[0], mask3.shape[1], 3), dtype=np.uint8)
                    rgb_mask4 = np.zeros((mask4.shape[0], mask4.shape[1], 3), dtype=np.uint8)
                    rgb_mask5 = np.zeros((mask5.shape[0], mask5.shape[1], 3), dtype=np.uint8)
                    rgb_mask6 = np.zeros((mask6.shape[0], mask6.shape[1], 3), dtype=np.uint8)
                    # Set the red channel to maximum for white pixels
                    rgb_mask1[mask1 == True, 0] = 255
                    rgb_mask2[mask2 == True, 0] = 255
                    rgb_mask3[mask3 == True, 0] = 255
                    rgb_mask4[mask4 == True, 0] = 255
                    rgb_mask5[mask5 == True, 0] = 255
                    rgb_mask6[extracted_region == 255, 0] = 255

                    height, width = color_image.shape[:2]
                    # Convert depth image to point cloud
                    u, v = np.meshgrid(np.arange(width), np.arange(height))

                    x = (u - left_camera_intrinsic.cx) * depth_image / left_camera_intrinsic.fx
                    y = (v - left_camera_intrinsic.cy) * depth_image / left_camera_intrinsic.fy
                    z = depth_image

                    # Create point cloud
                    point_cloud_orig = np.stack((x, y, z, color_image[:, :, 0] / 255, color_image[:, :, 1] / 255,
                                                 color_image[:, :, 2] / 255), axis=-1).reshape(
                        -1, 6)
                    point_cloud_mask1 = np.stack((x, y, z, rgb_mask1[:, :, 0] / 255, rgb_mask1[:, :, 1] / 255,
                                                  rgb_mask1[:, :, 2] / 255), axis=-1).reshape(
                        -1, 6)
                    point_cloud_mask2 = np.stack((x, y, z, rgb_mask2[:, :, 0] / 255, rgb_mask2[:, :, 1] / 255,
                                                  rgb_mask2[:, :, 2] / 255), axis=-1).reshape(
                        -1, 6)
                    point_cloud_mask3 = np.stack((x, y, z, rgb_mask3[:, :, 0] / 255, rgb_mask3[:, :, 1] / 255,
                                                  rgb_mask3[:, :, 2] / 255), axis=-1).reshape(
                        -1, 6)
                    point_cloud_mask4 = np.stack((x, y, z, rgb_mask4[:, :, 0] / 255, rgb_mask4[:, :, 1] / 255,
                                                  rgb_mask4[:, :, 2] / 255), axis=-1).reshape(
                        -1, 6)
                    point_cloud_mask5 = np.stack((x, y, z, rgb_mask5[:, :, 0] / 255, rgb_mask5[:, :, 1] / 255,
                                                  rgb_mask5[:, :, 2] / 255), axis=-1).reshape(
                        -1, 6)
                    point_cloud_mask6 = np.stack((x, y, z, rgb_mask6[:, :, 0] / 255, rgb_mask6[:, :, 1] / 255,
                                                 rgb_mask6[:, :, 2] / 255), axis=-1).reshape(
                        -1, 6)
                    point_cloud_orig = np.nan_to_num(point_cloud_orig)
                    point_cloud_mask1 = np.nan_to_num(point_cloud_mask1)
                    point_cloud_mask2 = np.nan_to_num(point_cloud_mask2)
                    point_cloud_mask3 = np.nan_to_num(point_cloud_mask3)
                    point_cloud_mask4 = np.nan_to_num(point_cloud_mask4)
                    point_cloud_mask5 = np.nan_to_num(point_cloud_mask5)
                    point_cloud_mask6 = np.nan_to_num(point_cloud_mask6)


                    # Create open3d point cloud
                    pcd_orig = o3d.geometry.PointCloud()
                    pcd_orig.points = o3d.utility.Vector3dVector(point_cloud_orig[:, :3])
                    pcd_orig.colors = o3d.utility.Vector3dVector(point_cloud_orig[:, 3:6])

                    pcd_mask1 = o3d.geometry.PointCloud()
                    pcd_mask1.points = o3d.utility.Vector3dVector(point_cloud_mask1[:, :3])
                    pcd_mask1.colors = o3d.utility.Vector3dVector(point_cloud_mask1[:, 3:6])

                    pcd_mask2 = o3d.geometry.PointCloud()
                    pcd_mask2.points = o3d.utility.Vector3dVector(point_cloud_mask2[:, :3])
                    pcd_mask2.colors = o3d.utility.Vector3dVector(point_cloud_mask2[:, 3:6])

                    pcd_mask3 = o3d.geometry.PointCloud()
                    pcd_mask3.points = o3d.utility.Vector3dVector(point_cloud_mask3[:, :3])
                    pcd_mask3.colors = o3d.utility.Vector3dVector(point_cloud_mask3[:, 3:6])

                    pcd_mask4 = o3d.geometry.PointCloud()
                    pcd_mask4.points = o3d.utility.Vector3dVector(point_cloud_mask4[:, :3])
                    pcd_mask4.colors = o3d.utility.Vector3dVector(point_cloud_mask4[:, 3:6])

                    pcd_mask5 = o3d.geometry.PointCloud()
                    pcd_mask5.points = o3d.utility.Vector3dVector(point_cloud_mask5[:, :3])
                    pcd_mask5.colors = o3d.utility.Vector3dVector(point_cloud_mask5[:, 3:6])

                    pcd_mask6 = o3d.geometry.PointCloud()
                    pcd_mask6.points = o3d.utility.Vector3dVector(point_cloud_mask6[:, :3])
                    pcd_mask6.colors = o3d.utility.Vector3dVector(point_cloud_mask6[:, 3:6])


                    # Convert Open3D point cloud to numpy array
                    points1 = np.asarray(pcd_mask1.points)
                    points2 = np.asarray(pcd_mask2.points)
                    points3 = np.asarray(pcd_mask3.points)
                    points4 = np.asarray(pcd_mask4.points)
                    points5 = np.asarray(pcd_mask5.points)
                    points6 = np.asarray(pcd_mask6.points)



                    # Access the RGB colors of each point
                    colors1 = np.asarray(pcd_mask1.colors)
                    colors2 = np.asarray(pcd_mask2.colors)
                    colors3 = np.asarray(pcd_mask3.colors)
                    colors4 = np.asarray(pcd_mask4.colors)
                    colors5 = np.asarray(pcd_mask5.colors)
                    colors6 = np.asarray(pcd_mask6.colors)


                    # Filter out points with red color
                    red_points_indices1 = np.where((colors1[:, 0] == 1) & (colors1[:, 1] == 0) & (colors1[:, 2] == 0))[0]
                    red_points_indices2 = np.where((colors2[:, 0] == 1) & (colors2[:, 1] == 0) & (colors2[:, 2] == 0))[0]
                    red_points_indices3 = np.where((colors3[:, 0] == 1) & (colors3[:, 1] == 0) & (colors3[:, 2] == 0))[0]
                    red_points_indices4 = np.where((colors4[:, 0] == 1) & (colors4[:, 1] == 0) & (colors4[:, 2] == 0))[0]
                    red_points_indices5 = np.where((colors5[:, 0] == 1) & (colors5[:, 1] == 0) & (colors5[:, 2] == 0))[0]
                    black_points_indices = np.where((colors6[:, 0] == 1) & (colors6[:, 1] == 0) & (colors6[:, 2] == 0))[0]

                    # Extract red-colored points
                    red_points1 = points1[red_points_indices1]
                    red_points2 = points2[red_points_indices2]
                    red_points3 = points3[red_points_indices3]
                    red_points4 = points4[red_points_indices4]
                    red_points5 = points5[red_points_indices5]
                    black_points = points6[black_points_indices]

                    # Create an Open3D point cloud from the extracted red points
                    red_point_cloud1 = o3d.geometry.PointCloud()
                    red_point_cloud2 = o3d.geometry.PointCloud()
                    red_point_cloud3 = o3d.geometry.PointCloud()
                    red_point_cloud4 = o3d.geometry.PointCloud()
                    red_point_cloud5 = o3d.geometry.PointCloud()
                    black_point_cloud = o3d.geometry.PointCloud()


                    red_point_cloud1.points = o3d.utility.Vector3dVector(red_points1)
                    red_point_cloud2.points = o3d.utility.Vector3dVector(red_points2)
                    red_point_cloud3.points = o3d.utility.Vector3dVector(red_points3)
                    red_point_cloud4.points = o3d.utility.Vector3dVector(red_points4)
                    red_point_cloud5.points = o3d.utility.Vector3dVector(red_points5)
                    black_point_cloud.points = o3d.utility.Vector3dVector(black_points)

                    black_point_cloud.paint_uniform_color((0,0,0))
                    red_point_cloud1.paint_uniform_color((1,0,0))
                    red_point_cloud2.paint_uniform_color((0,1,0))
                    red_point_cloud3.paint_uniform_color((0,1,1))
                    red_point_cloud4.paint_uniform_color((0,0,1))
                    red_point_cloud5.paint_uniform_color((1,1,0))

                    # Visualize the red point cloud
                    print(black_point_cloud)
                    print(red_point_cloud1)
                    print(red_point_cloud2)
                    print(red_point_cloud3)
                    print(red_point_cloud4)
                    print(red_point_cloud5)


                    union = concatenate_point_clouds_with_colors(black_point_cloud,red_point_cloud1,red_point_cloud2,
                                                     red_point_cloud3,red_point_cloud4,red_point_cloud5)
                    points = np.asarray(union.points)
                    downsampled_point_cloud = union.uniform_down_sample( every_k_points=int(points.shape[0]/4096))
                    o3d.visualization.draw_geometries([downsampled_point_cloud])
                    cur_frame += 1


print("time taken: ", time.time() - start_time)