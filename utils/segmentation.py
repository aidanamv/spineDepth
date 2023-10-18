import pyrender
import trimesh
import os
import pyzed.sl as sl
import numpy as np
import cv2
import open3d as o3d
import pandas as pd

dir ="/home/aidana/PycharmProjects/spineDepth/data/specimen4_recording0"
save_pcd_dir = os.path.join(dir, "pointcloud")
video_dir = os.path.join(dir, "Video_0.svo")
calibration_file = os.path.join(dir, "Calib/SN10027879.conf")
input_type = sl.InputType()
input_type.set_from_svo_file(video_dir)
init_params = sl.InitParameters()

init_params.optional_settings_path = calibration_file
init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
zed = sl.Camera()
status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    exit()



runtime_parameters = sl.RuntimeParameters()

# initialize images and point cloud
j = 0
image = sl.Mat()
depth = sl.Mat()
pcd = sl.Mat()

# Enable positional tracking with default parameters
py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object

tracking_parameters = sl.PositionalTrackingParameters(py_transform)

err = zed.enable_positional_tracking(tracking_parameters)
if err != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    exit(1)

zed_pose = sl.Pose()

zed_sensors = sl.SensorsData()

calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters

nb_frames = zed.get_svo_number_of_frames()
print("number of frames: {}".format(nb_frames))


# Grab an image, a RuntimeParameters object must be given to grab()
if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    svo_position_1 = zed.get_svo_position()
    print("svo position: {}".format(svo_position_1))
    # A new image is available if grab() returns SUCCESS
    # retrive and write point cloud
   # zed.retrieve_image(image, sl.VIEW.LEFT)
   


    # Retrieve depth map. Depth is aligned on the left image
    #zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

    # image.write(os.path.join(save_pcd_dir, 'image.png'))
    # depth.write(os.path.join(save_pcd_dir, 'depth.png'))
    # # Check if we have reached the end of the video
    # if svo_position_1 <= (nb_frames - 1):  # End of SVO
    #     tracking_file_1 = os.path.join(dir, "Poses_0.txt")
    #
    #     df = pd.read_csv(tracking_file_1, sep=',', header=None,
    #                      names=["R00", "R01", "R02", "T0", "R10", "R11", "R12", "T1", "R20", "R21", "R22", "T2", "R30",
    #                             "R31", "R33", "T3"])
    #
    #
    #
    #     # reading corresponding stl files
    #     vertebra1_cam1 = o3d.io.read_triangle_mesh( os.path.join(dir,"STL/L1.stl"))
    #     vertebra2_cam1 = o3d.io.read_triangle_mesh( os.path.join(dir,"STL/L2.stl"))
    #     vertebra3_cam1 = o3d.io.read_triangle_mesh( os.path.join(dir,"STL/L3.stl"))
    #     vertebra4_cam1 = o3d.io.read_triangle_mesh( os.path.join(dir,"STL/L4.stl"))
    #     vertebra5_cam1 = o3d.io.read_triangle_mesh( os.path.join(dir,"STL/L5.stl"))
    #
    #
    #
    #     vertebra1_cam1 = vertebra1_cam1.compute_vertex_normals()
    #     vertebra2_cam1 = vertebra2_cam1.compute_vertex_normals()
    #     vertebra3_cam1 = vertebra3_cam1.compute_vertex_normals()
    #     vertebra4_cam1 = vertebra4_cam1.compute_vertex_normals()
    #     vertebra5_cam1 = vertebra5_cam1.compute_vertex_normals()
    #
    #
    #     i = 0
    #     vertebra1_pose_cam1 = np.array([[df.iloc[i]["R00"], df.iloc[i]["R01"], df.iloc[i]["R02"], df.iloc[i]["T0"]],
    #                                     [df.iloc[i]["R10"], df.iloc[i]["R11"], df.iloc[i]["R12"], df.iloc[i]["T1"]],
    #                                     [df.iloc[i]["R20"], df.iloc[i]["R21"], df.iloc[i]["R22"], df.iloc[i]["T2"]],
    #                                     [0, 0, 0, 1]])
    #
    #
    #     vertebra2_pose_cam1 = np.array(
    #         [[df.iloc[i + 1]["R00"], df.iloc[i + 1]["R01"], df.iloc[i + 1]["R02"], df.iloc[i + 1]["T0"]],
    #          [df.iloc[i + 1]["R10"], df.iloc[i + 1]["R11"], df.iloc[i + 1]["R12"], df.iloc[i + 1]["T1"]],
    #          [df.iloc[i + 1]["R20"], df.iloc[i + 1]["R21"], df.iloc[i + 1]["R22"], df.iloc[i + 1]["T2"]],
    #          [0, 0, 0, 1]])
    #
    #
    #     vertebra3_pose_cam1 = np.array(
    #         [[df.iloc[i + 2]["R00"], df.iloc[i + 2]["R01"], df.iloc[i + 2]["R02"], df.iloc[i + 2]["T0"]],
    #          [df.iloc[i + 2]["R10"], df.iloc[i + 2]["R11"], df.iloc[i + 2]["R12"], df.iloc[i + 2]["T1"]],
    #          [df.iloc[i + 2]["R20"], df.iloc[i + 2]["R21"], df.iloc[i + 2]["R22"], df.iloc[i + 2]["T2"]],
    #          [0, 0, 0, 1]])
    #
    #     vertebra4_pose_cam1 = np.array(
    #         [[df.iloc[i + 3]["R00"], df.iloc[i + 3]["R01"], df.iloc[i + 3]["R02"], df.iloc[i + 3]["T0"]],
    #          [df.iloc[i + 3]["R10"], df.iloc[i + 3]["R11"], df.iloc[i + 3]["R12"], df.iloc[i + 3]["T1"]],
    #          [df.iloc[i + 3]["R20"], df.iloc[i + 3]["R21"], df.iloc[i + 3]["R22"], df.iloc[i + 3]["T2"]],
    #          [0, 0, 0, 1]])
    #
    #
    #
    #     vertebra5_pose_cam1 = np.array(
    #         [[df.iloc[i + 4]["R00"], df.iloc[i + 4]["R01"], df.iloc[i + 4]["R02"], df.iloc[i + 4]["T0"]],
    #          [df.iloc[i + 4]["R10"], df.iloc[i + 4]["R11"], df.iloc[i + 4]["R12"], df.iloc[i + 4]["T1"]],
    #          [df.iloc[i + 4]["R20"], df.iloc[i + 4]["R21"], df.iloc[i + 4]["R22"], df.iloc[i + 4]["T2"]],
    #          [0, 0, 0, 1]])
    #
    #
    #
    #     vertebra1_cam1.transform(vertebra1_pose_cam1)
    #     vertebra2_cam1.transform(vertebra2_pose_cam1)
    #     vertebra3_cam1.transform(vertebra3_pose_cam1)
    #     vertebra4_cam1.transform(vertebra4_pose_cam1)
    #     vertebra5_cam1.transform(vertebra5_pose_cam1)
    #
    #
    #     spine_cam1 = vertebra1_cam1 + vertebra2_cam1+ vertebra3_cam1 + vertebra4_cam1 + vertebra5_cam1
    #
    #
    #     R = zed_pose.get_rotation_matrix(sl.Rotation()).r.T
    #     t = zed_pose.get_translation(sl.Translation()).get()
    #
    #
    #     # Create the 4x4 extrinsics transformation matrix
    #     extrinsics_matrix = np.identity(4)
    #     extrinsics_matrix[:3, :3] = R
    #     extrinsics_matrix[:3, 3] = t
    #
    #
    #
    #     camera_pose = np.array([
    #         [1, 0, 0, 0],
    #         [0, 1, 0, 0],
    #         [0, 0, 1, 0],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ])
    #     left_camera_intrinsic = calibration_params.left_cam
    #     camera = pyrender.camera.IntrinsicsCamera(left_camera_intrinsic.fx, left_camera_intrinsic.fy, left_camera_intrinsic.cx,left_camera_intrinsic.cy, znear=250, zfar=2000)
    #     o3d.io.write_triangle_mesh("temp.stl", spine_cam1)
    #     fuze_trimesh = trimesh.load('temp.stl')
    #
    #     mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    #     scene = pyrender.Scene()
    #     scene.add(mesh, pose=extrinsics_matrix)
    #
    #     rotate = trimesh.transformations.rotation_matrix(
    #         angle=np.radians(180.0),
    #         direction=[0, 1, 0],
    #         point=[0, 0, 0])
    #     rotate2 = trimesh.transformations.rotation_matrix(
    #         angle=np.radians(180.0),
    #         direction=[0, 0, 1],
    #         point=[0, 0, 0])
    #     scene.add(camera, pose=rotate * rotate2)
    #
    #     r = pyrender.OffscreenRenderer(1920, 1080)
    #
    #     color, depth = r.render(scene)
    #
    #     cv2.imwrite("depth.png", depth)



        # result_color_image = 255 * np.ones_like(color_image, dtype=np.uint8)
        # result_depth_image = 255 * np.ones_like(depth_image, dtype=np.uint8)
        #
        # # Copy the color image to the result image where the mask is non-zero
        # result_color_image[mask_image > 0] = color_image[mask_image > 0]
        # result_depth_image[mask_image > 0] = depth_image[mask_image > 0]
        #
        # # For saving as an image (using OpenCV)
        # cv2.imwrite("result_color_image.png", result_color_image)
        # cv2.imwrite("result_depth_image.png", result_depth_image)
        #
        #
        # # Create Open3D image from depth and color images
        # depth_o3d = o3d.io.read_image("result_depth_image.png")
        # color_o3d = o3d.io.read_image("result_color_image.png")
        #
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     color_o3d, depth_o3d)
        # # Create the point cloud from the RGBD image and camera intrinsic parameters
        # intrinsic = o3d.camera.PinholeCameraIntrinsic()
        # intrinsic.set_intrinsics(width=1080, height=1920,fx=left_camera_intrinsic.fx, fy=left_camera_intrinsic.fy, cx=left_camera_intrinsic.cx, cy=left_camera_intrinsic.cy)

       # pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    mask_image = cv2.imread("depth.png")

    zed.retrieve_measure(pcd, sl.MEASURE.XYZRGBA)
    os.makedirs(save_pcd_dir, exist_ok=True)
    pcd.write(os.path.join(save_pcd_dir, 'pointcloud.ply'))
    pcd = o3d.io.read_point_cloud(os.path.join(save_pcd_dir, 'pointcloud.ply'))
    pcd_array = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)
    # Convert the point cloud to a NumPy array
    mask_all_zero = np.all(mask_image == 0,axis=2)
    # Get indices where mask is True
    indices = np.argwhere(mask_all_zero)
    print("number of indices",len(indices))

    # Loop over the indices
    for i, j in indices:
        status, rgba = pcd.get_value(int(i), int(j))
        if np.isnan(rgba.all()) == False:
            rgba[0] = np.nan
            rgba[1] = np.nan
            rgba[2] = np.nan
            rgba[3] = np.nan
            pcd.set_value(int(i), int(j), rgba)
            del rgba

