import pyrender
import trimesh
import os
import pyzed.sl as sl
import numpy as np
import cv2
import open3d as o3d
import pandas as pd
import sys

dir ="/home/aidana/PycharmProjects/spineDepth/data/specimen4_recording0"
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



# # Enable positional tracking with default parameters
# py_transform = sl.Transform()
# tracking_parameters = sl.PositionalTrackingParameters(py_transform)
# err = zed.enable_positional_tracking(tracking_parameters)
# if err != sl.ERROR_CODE.SUCCESS:
#     print(repr(status))
#     exit(1)

zed_pose = sl.Pose()
zed_sensors = sl.SensorsData()
calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
i =0
nb_frames = zed.get_svo_number_of_frames()
print("number of frames: {}".format(nb_frames))
svo_position=0
try:
    while svo_position < 1:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()
            print("svo position: {}".format(svo_position))
            save_pcd_dir = os.path.join(dir, "pointcloud{}".format(svo_position))
            if not os.path.exists(save_pcd_dir):
                os.makedirs(save_pcd_dir)
            image = sl.Mat()
            depth = sl.Mat()
            pcd1 = sl.Mat()
            pcd2 = sl.Mat()
            pcd3 = sl.Mat()
            pcd4 = sl.Mat()
            pcd5 = sl.Mat()


            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(pcd1, sl.MEASURE.XYZRGBA)
            zed.retrieve_measure(pcd2, sl.MEASURE.XYZRGBA)
            zed.retrieve_measure(pcd3, sl.MEASURE.XYZRGBA)
            zed.retrieve_measure(pcd4, sl.MEASURE.XYZRGBA)
            zed.retrieve_measure(pcd5, sl.MEASURE.XYZRGBA)


            image.write(os.path.join(save_pcd_dir, 'image.png'))
            depth.write(os.path.join(save_pcd_dir, 'depth.png'))


            tracking_file_1 = os.path.join(dir, "Poses_0.txt")

            df = pd.read_csv(tracking_file_1, sep=',', header=None,
                             names=["R00", "R01", "R02", "T0", "R10", "R11", "R12", "T1", "R20", "R21", "R22", "T2", "R30",
                                    "R31", "R33", "T3"])



            # reading corresponding stl files
            vertebra1 = o3d.io.read_triangle_mesh(os.path.join(dir, "STL/L1.stl"))
            vertebra2 = o3d.io.read_triangle_mesh(os.path.join(dir, "STL/L2.stl"))
            vertebra3 = o3d.io.read_triangle_mesh(os.path.join(dir, "STL/L3.stl"))
            vertebra4 = o3d.io.read_triangle_mesh(os.path.join(dir, "STL/L4.stl"))
            vertebra5 = o3d.io.read_triangle_mesh(os.path.join(dir, "STL/L5.stl"))



            vertebra1 = vertebra1.compute_vertex_normals()
            vertebra2 = vertebra2.compute_vertex_normals()
            vertebra3 = vertebra3.compute_vertex_normals()
            vertebra4 = vertebra4.compute_vertex_normals()
            vertebra5 = vertebra5.compute_vertex_normals()

            vertebra1_pose = np.array([[df.iloc[i]["R00"], df.iloc[i]["R01"], df.iloc[i]["R02"], df.iloc[i]["T0"]],
                                       [df.iloc[i]["R10"], df.iloc[i]["R11"], df.iloc[i]["R12"], df.iloc[i]["T1"]],
                                       [df.iloc[i]["R20"], df.iloc[i]["R21"], df.iloc[i]["R22"], df.iloc[i]["T2"]],
                                       [0, 0, 0, 1]])


            vertebra2_pose = np.array(
                [[df.iloc[i + 1]["R00"], df.iloc[i + 1]["R01"], df.iloc[i + 1]["R02"], df.iloc[i + 1]["T0"]],
                 [df.iloc[i + 1]["R10"], df.iloc[i + 1]["R11"], df.iloc[i + 1]["R12"], df.iloc[i + 1]["T1"]],
                 [df.iloc[i + 1]["R20"], df.iloc[i + 1]["R21"], df.iloc[i + 1]["R22"], df.iloc[i + 1]["T2"]],
                 [0, 0, 0, 1]])


            vertebra3_pose = np.array(
                [[df.iloc[i + 2]["R00"], df.iloc[i + 2]["R01"], df.iloc[i + 2]["R02"], df.iloc[i + 2]["T0"]],
                 [df.iloc[i + 2]["R10"], df.iloc[i + 2]["R11"], df.iloc[i + 2]["R12"], df.iloc[i + 2]["T1"]],
                 [df.iloc[i + 2]["R20"], df.iloc[i + 2]["R21"], df.iloc[i + 2]["R22"], df.iloc[i + 2]["T2"]],
                 [0, 0, 0, 1]])

            vertebra4_pose = np.array(
                [[df.iloc[i + 3]["R00"], df.iloc[i + 3]["R01"], df.iloc[i + 3]["R02"], df.iloc[i + 3]["T0"]],
                 [df.iloc[i + 3]["R10"], df.iloc[i + 3]["R11"], df.iloc[i + 3]["R12"], df.iloc[i + 3]["T1"]],
                 [df.iloc[i + 3]["R20"], df.iloc[i + 3]["R21"], df.iloc[i + 3]["R22"], df.iloc[i + 3]["T2"]],
                 [0, 0, 0, 1]])



            vertebra5_pose = np.array(
                [[df.iloc[i + 4]["R00"], df.iloc[i + 4]["R01"], df.iloc[i + 4]["R02"], df.iloc[i + 4]["T0"]],
                 [df.iloc[i + 4]["R10"], df.iloc[i + 4]["R11"], df.iloc[i + 4]["R12"], df.iloc[i + 4]["T1"]],
                 [df.iloc[i + 4]["R20"], df.iloc[i + 4]["R21"], df.iloc[i + 4]["R22"], df.iloc[i + 4]["T2"]],
                 [0, 0, 0, 1]])

            i = i+5



            vertebra1.transform(vertebra1_pose)
            vertebra2.transform(vertebra2_pose)
            vertebra3.transform(vertebra3_pose)
            vertebra4.transform(vertebra4_pose)
            vertebra5.transform(vertebra5_pose)


            spine = vertebra1 + vertebra2 + vertebra3 + vertebra4 + vertebra5

            R = zed_pose.get_rotation_matrix(sl.Rotation()).r.T
            t = zed_pose.get_translation(sl.Translation()).get()


            # Create the 4x4 extrinsics transformation matrix
            extrinsics_matrix = np.identity(4)
            extrinsics_matrix[:3, :3] = R
            extrinsics_matrix[:3, 3] = t



            camera_pose = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0.0, 0.0, 0.0, 1.0],
            ])


            left_camera_intrinsic = calibration_params.left_cam
            camera = pyrender.camera.IntrinsicsCamera(left_camera_intrinsic.fx, left_camera_intrinsic.fy, left_camera_intrinsic.cx,left_camera_intrinsic.cy, znear=250, zfar=2000)
            o3d.io.write_triangle_mesh(os.path.join(save_pcd_dir,"transformed_vertebra1.stl"), vertebra1)
            o3d.io.write_triangle_mesh(os.path.join(save_pcd_dir,"transformed_vertebra2.stl"), vertebra2)
            o3d.io.write_triangle_mesh(os.path.join(save_pcd_dir,"transformed_vertebra3.stl"), vertebra3)
            o3d.io.write_triangle_mesh(os.path.join(save_pcd_dir,"transformed_vertebra4.stl"), vertebra4)
            o3d.io.write_triangle_mesh(os.path.join(save_pcd_dir,"transformed_vertebra5.stl"), vertebra5)


            mask1_stl = trimesh.load(os.path.join(save_pcd_dir,"transformed_vertebra1.stl"))
            mask2_stl = trimesh.load(os.path.join(save_pcd_dir,"transformed_vertebra2.stl"))
            mask3_stl = trimesh.load(os.path.join(save_pcd_dir,"transformed_vertebra3.stl"))
            mask4_stl = trimesh.load(os.path.join(save_pcd_dir,"transformed_vertebra4.stl"))
            mask5_stl = trimesh.load(os.path.join(save_pcd_dir,"transformed_vertebra5.stl"))


            mesh1 = pyrender.Mesh.from_trimesh(mask1_stl)
            mesh2 = pyrender.Mesh.from_trimesh(mask2_stl)
            mesh3 = pyrender.Mesh.from_trimesh(mask3_stl)
            mesh4 = pyrender.Mesh.from_trimesh(mask4_stl)
            mesh5 = pyrender.Mesh.from_trimesh(mask5_stl)


            scene1 = pyrender.Scene()
            scene2 = pyrender.Scene()
            scene3 = pyrender.Scene()
            scene4 = pyrender.Scene()
            scene5 = pyrender.Scene()

            scene1.add(mesh1, pose=extrinsics_matrix)
            scene2.add(mesh2, pose=extrinsics_matrix)
            scene3.add(mesh3, pose=extrinsics_matrix)
            scene4.add(mesh4, pose=extrinsics_matrix)
            scene5.add(mesh5, pose=extrinsics_matrix)


            rotate = trimesh.transformations.rotation_matrix(
                angle=np.radians(180.0),
                direction=[0, 1, 0],
                point=[0, 0, 0])
            rotate2 = trimesh.transformations.rotation_matrix(
                angle=np.radians(180.0),
                direction=[0, 0, 1],
                point=[0, 0, 0])
            scene1.add(camera, pose=rotate * rotate2)
            scene2.add(camera, pose=rotate * rotate2)
            scene3.add(camera, pose=rotate * rotate2)
            scene4.add(camera, pose=rotate * rotate2)
            scene5.add(camera, pose=rotate * rotate2)

            r1 = pyrender.OffscreenRenderer(1920, 1080)
            r2 = pyrender.OffscreenRenderer(1920, 1080)
            r3 = pyrender.OffscreenRenderer(1920, 1080)
            r4 = pyrender.OffscreenRenderer(1920, 1080)
            r5 = pyrender.OffscreenRenderer(1920, 1080)


            _, depth1 = r1.render(scene1)
            _, depth2 = r1.render(scene2)
            _, depth3 = r1.render(scene3)
            _, depth4 = r1.render(scene4)
            _, depth5 = r1.render(scene5)

            cv2.imwrite(os.path.join(save_pcd_dir,"mask1.png"), depth1)
            cv2.imwrite(os.path.join(save_pcd_dir,"mask2.png"), depth2)
            cv2.imwrite(os.path.join(save_pcd_dir,"mask3.png"), depth3)
            cv2.imwrite(os.path.join(save_pcd_dir,"mask4.png"), depth4)
            cv2.imwrite(os.path.join(save_pcd_dir,"mask5.png"), depth5)

            mask_image1 = cv2.imread(os.path.join(save_pcd_dir,"mask1.png"))
            mask_image2 = cv2.imread(os.path.join(save_pcd_dir,"mask2.png"))
            mask_image3 = cv2.imread(os.path.join(save_pcd_dir,"mask3.png"))
            mask_image4 = cv2.imread(os.path.join(save_pcd_dir,"mask4.png"))
            mask_image5 = cv2.imread(os.path.join(save_pcd_dir,"mask5.png"))

            numpy_array1 = pcd1.get_data()
            numpy_array2 = pcd2.get_data()
            numpy_array3 = pcd3.get_data()
            numpy_array4 = pcd4.get_data()
            numpy_array5 = pcd5.get_data()


            mask_all_zero_1 = np.all(mask_image1 == 0,axis=2)
            mask_all_zero_2 = np.all(mask_image2 == 0,axis=2)
            mask_all_zero_3 = np.all(mask_image3 == 0,axis=2)
            mask_all_zero_4 = np.all(mask_image4 == 0,axis=2)
            mask_all_zero_5 = np.all(mask_image5 == 0,axis=2)

            # Get indices where mask is True
            indices1 = np.argwhere(mask_all_zero_1)
            indices2 = np.argwhere(mask_all_zero_2)
            indices3 = np.argwhere(mask_all_zero_3)
            indices4 = np.argwhere(mask_all_zero_4)
            indices5 = np.argwhere(mask_all_zero_5)


            # Loop over the indices
            for j, k in indices1:
                rgba = numpy_array1[j][k][:]
                if np.isnan(rgba.all()) == False:
                    rgba[0] = np.nan
                    rgba[1] = np.nan
                    rgba[2] = np.nan
                    rgba[3] = np.nan
                    numpy_array1[j][k][:] = rgba
            for j, k in indices2:
                rgba = numpy_array2[j][k][:]
                if np.isnan(rgba.all()) == False:
                    rgba[0] = np.nan
                    rgba[1] = np.nan
                    rgba[2] = np.nan
                    rgba[3] = np.nan
                    numpy_array2[j][k][:] = rgba
            for j, k in indices3:
                rgba = numpy_array3[j][k][:]
                if np.isnan(rgba.all()) == False:
                    rgba[0] = np.nan
                    rgba[1] = np.nan
                    rgba[2] = np.nan
                    rgba[3] = np.nan
                    numpy_array3[j][k][:] = rgba

            for j, k in indices4:
                rgba = numpy_array4[j][k][:]
                if np.isnan(rgba.all()) == False:
                    rgba[0] = np.nan
                    rgba[1] = np.nan
                    rgba[2] = np.nan
                    rgba[3] = np.nan
                    numpy_array4[j][k][:] = rgba

            for j, k in indices5:
                rgba = numpy_array5[j][k][:]
                if np.isnan(rgba.all()) == False:
                    rgba[0] = np.nan
                    rgba[1] = np.nan
                    rgba[2] = np.nan
                    rgba[3] = np.nan
                    numpy_array5[j][k][:] = rgba



           # zed.retrieve_measure(pcd, sl.MEASURE.XYZRGBA)
            os.makedirs(save_pcd_dir, exist_ok=True)
            pcd1.write(os.path.join(save_pcd_dir, 'pointcloud_vert1.ply'))
            pcd2.write(os.path.join(save_pcd_dir, 'pointcloud_vert2.ply'))
            pcd3.write(os.path.join(save_pcd_dir, 'pointcloud_vert3.ply'))
            pcd4.write(os.path.join(save_pcd_dir, 'pointcloud_vert4.ply'))
            pcd5.write(os.path.join(save_pcd_dir, 'pointcloud_vert5.ply'))

            pcd_vert1 = o3d.io.read_point_cloud(os.path.join(save_pcd_dir, 'pointcloud_vert1.ply'))
            pcd_vert2 = o3d.io.read_point_cloud(os.path.join(save_pcd_dir, 'pointcloud_vert2.ply'))
            pcd_vert3 = o3d.io.read_point_cloud(os.path.join(save_pcd_dir, 'pointcloud_vert3.ply'))
            pcd_vert4 = o3d.io.read_point_cloud(os.path.join(save_pcd_dir, 'pointcloud_vert4.ply'))
            pcd_vert5 = o3d.io.read_point_cloud(os.path.join(save_pcd_dir, 'pointcloud_vert5.ply'))

            pcd_vert1.paint_uniform_color([1, 0.706, 0])
            pcd_vert2.paint_uniform_color([1, 0, 0])
            pcd_vert3.paint_uniform_color([0, 1, 0])
            pcd_vert4.paint_uniform_color([0, 0, 1])
            pcd_vert5.paint_uniform_color([0, 1, 1])

            print(np.asarray(pcd_vert1.points).shape)
            print(np.asarray(pcd_vert2.points).shape)
            print(np.asarray(pcd_vert3.points).shape)
            print(np.asarray(pcd_vert4.points).shape)
            print(np.asarray(pcd_vert5.points).shape)

            for el,each_vert in enumerate([pcd_vert1, pcd_vert2, pcd_vert3, pcd_vert4, pcd_vert5]):

                sampled_points_array_temp = np.array(each_vert.points)

                # Create an Open3D PointCloud from the sampled points
                pcd_input = o3d.geometry.PointCloud()
                pcd_input.points = o3d.utility.Vector3dVector(sampled_points_array_temp)
                # Define the number of sample points you want
                constant_sample_count = 2048  # Adjust this number as needed

                # Get the current number of points
                current_point_count = np.asarray(pcd_input.points).shape[0]

                # Generate random indices to select a subset of points
                random_indices = np.random.choice(current_point_count, constant_sample_count, replace=False)

                # Select the subset of points based on the random indices
                downsampled_point_cloud = pcd_input.select_by_index(random_indices)

                o3d.io.write_point_cloud(os.path.join(save_pcd_dir, 'pointcloud_vert{}_downsampled.ply'.format(el+1)), downsampled_point_cloud)
except KeyboardInterrupt:
    print("Keyboard interrupt. Exiting...")
