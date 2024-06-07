#this script is designed to segment the vertebrae from the spine depth dataset
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

#df_plan=pd.read_csv("output.csv")
start_time = time.time()
dir ="/media/aidana/aidana/SpineDepth"
calib_dest_dir = "/usr/local/zed/settings/"
path_stls = "/home/aidana/PycharmProjects/PSP_planning/planning data original/stls all vertebrae"
plannings = "/home/aidana/PycharmProjects/PSP_planning/planning data original/planning_points.xlsx"
specimens = os.listdir(dir)
camera_nums = [0,1]

for specimen in specimens[3:]:
    recordings = len(os.listdir(os.path.join(dir, specimen, "recordings")))
    if specimen == "Specimen_4":
        start = 5
    else:
        start = 0

    for recording in range(start,recordings):
        for camera_num in camera_nums:
            cur_frame = 0
            # if os.path.exists(os.path.join("/media/aidana/Extreme SSD/PoinTr dataset/segmented_spinedepth_new",
            #                                  specimen,"recording_{}".format(recording),
            #                                  "cam_{}".format(camera_num),"frame_{}".format(3), "pointcloud_vert1_cropped.ply")):
            #     print("skipped")
            #     continue


            if specimen=="Specimen_2" and recording>=32:
                calib_src_dir = os.path.join(dir, specimen, "Calib_b")
            if specimen=="Specimen_5" and recording>=8:
                calib_src_dir = os.path.join(dir, specimen, "Calib_b")
            if specimen=="Specimen_7" and recording>=12:
                calib_src_dir = os.path.join(dir, specimen, "Calib_b")
            if specimen=="Specimen_9" and recording>=12:
                calib_src_dir = os.path.join(dir, specimen, "Calib_b")
            else:
                calib_src_dir = os.path.join(dir, specimen, "Calib")
            shutil.copytree(calib_src_dir, calib_dest_dir, dirs_exist_ok=True)


            video_dir = os.path.join(dir, specimen, "recordings/recording{}/Video_{}.svo".format(recording,camera_num))

            tracking_file = os.path.join(dir,  specimen, "recordings/recording{}/Poses_{}.txt".format(recording,camera_num))
            df = pd.read_csv(tracking_file, sep=',', header=None,
                             names=["R00", "R01", "R02", "T0", "R10", "R11", "R12", "T1", "R20", "R21", "R22", "T2", "R30",
                                    "R31", "R33", "T3"])


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


            while cur_frame<=3:
                print("processing specimen: {} recording {} video: {} frame: {}".format(specimen, recording, camera_num, cur_frame))

                save_data_dir = os.path.join("/media/aidana/aidana/PoinTr dataset/segmented_spinedepth_new",
                                             specimen,"recording_{}".format(recording),
                                             "cam_{}".format(camera_num),"frame_{}".format(cur_frame))
                if not os.path.exists(save_data_dir):
                    os.makedirs(save_data_dir)

                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    image = sl.Mat()
                    depth = sl.Mat()
                    pcd1 = sl.Mat()




                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                    zed.retrieve_measure(pcd1, sl.MEASURE.XYZRGBA)




                    image.write(os.path.join(save_data_dir, 'image.png'))
                    depth.write(os.path.join(save_data_dir, 'depth.png'))






                    # reading corresponding stl files
                    vertebra1 = o3d.io.read_triangle_mesh(os.path.join(dir, specimen, "STL/L1.stl"))
                    vertebra2 = o3d.io.read_triangle_mesh(os.path.join(dir, specimen,"STL/L2.stl"))
                    vertebra3 = o3d.io.read_triangle_mesh(os.path.join(dir,specimen,"STL/L3.stl"))
                    vertebra4 = o3d.io.read_triangle_mesh(os.path.join(dir,specimen, "STL/L4.stl"))
                    vertebra5 = o3d.io.read_triangle_mesh(os.path.join(dir,specimen, "STL/L5.stl"))

                    # landmarks_L1 = np.fromstring(df_plan[specimen+"_L1"].dropna().values[0].replace("array", "").replace("(","").replace(")","").replace("[","").replace("]",""), sep=',').reshape((4,3))
                    # landmarks_L2 = np.fromstring(df_plan[specimen+"_L2"].dropna().values[0].replace("array", "").replace("(","").replace(")","").replace("[","").replace("]",""), sep=',').reshape((4,3))
                    # landmarks_L3 = np.fromstring(df_plan[specimen+"_L3"].dropna().values[0].replace("array", "").replace("(","").replace(")","").replace("[","").replace("]",""), sep=',').reshape((4,3))
                    # landmarks_L4 = np.fromstring(df_plan[specimen+"_L4"].dropna().values[0].replace("array", "").replace("(","").replace(")","").replace("[","").replace("]",""), sep=',').reshape((4,3))
                    # landmarks_L5 = np.fromstring(df_plan[specimen+"_L5"].dropna().values[0].replace("array", "").replace("(","").replace(")","").replace("[","").replace("]",""), sep=',').reshape((4,3))



                    vertebra1 = vertebra1.compute_vertex_normals()
                    vertebra2 = vertebra2.compute_vertex_normals()
                    vertebra3 = vertebra3.compute_vertex_normals()
                    vertebra4 = vertebra4.compute_vertex_normals()
                    vertebra5 = vertebra5.compute_vertex_normals()
                    i = 5 * cur_frame

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




                    vertebra1.transform(vertebra1_pose)
                    vertebra2.transform(vertebra2_pose)
                    vertebra3.transform(vertebra3_pose)
                    vertebra4.transform(vertebra4_pose)
                    vertebra5.transform(vertebra5_pose)
                    #
                    # transformed_landmarks = np.vstack([np.dot(vertebra1_pose, np.append(landmarks_L1[0],1))[0:3],
                    #                                       np.dot(vertebra1_pose, np.append(landmarks_L1[1], 1))[0:3],
                    #                                       np.dot(vertebra1_pose, np.append(landmarks_L1[2], 1))[0:3],
                    #                                       np.dot(vertebra1_pose, np.append(landmarks_L1[3], 1))[0:3],
                    #                                       np.dot(vertebra2_pose, np.append(landmarks_L2[0], 1))[0:3],
                    #                                       np.dot(vertebra2_pose, np.append(landmarks_L2[1], 1))[0:3],
                    #                                       np.dot(vertebra2_pose, np.append(landmarks_L2[2], 1))[0:3],
                    #                                       np.dot(vertebra2_pose, np.append(landmarks_L2[3], 1))[0:3],
                    #                                       np.dot(vertebra3_pose, np.append(landmarks_L3[0], 1))[0:3],
                    #                                       np.dot(vertebra3_pose, np.append(landmarks_L3[1], 1))[0:3],
                    #                                       np.dot(vertebra3_pose, np.append(landmarks_L3[2], 1))[0:3],
                    #                                       np.dot(vertebra3_pose, np.append(landmarks_L3[3], 1))[0:3],
                    #                                       np.dot(vertebra4_pose, np.append(landmarks_L4[0], 1))[0:3],
                    #                                       np.dot(vertebra4_pose, np.append(landmarks_L4[1], 1))[0:3],
                    #                                       np.dot(vertebra4_pose, np.append(landmarks_L4[2], 1))[0:3],
                    #                                       np.dot(vertebra4_pose, np.append(landmarks_L4[3], 1))[0:3],
                    #                                       np.dot(vertebra5_pose, np.append(landmarks_L5[0], 1))[0:3],
                    #                                       np.dot(vertebra5_pose, np.append(landmarks_L5[1], 1))[0:3],
                    #                                       np.dot(vertebra5_pose, np.append(landmarks_L5[2], 1))[0:3],
                    #                                       np.dot(vertebra5_pose, np.append(landmarks_L5[3], 1))[0:3]])
                    #
                    #
                    #
                    # sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[0])
                    # sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[1])
                    # sphere3 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[2])
                    # sphere4 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[3])
                    #
                    # sphere5 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[4])
                    # sphere6 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[5])
                    # sphere7= o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[6])
                    # sphere8 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[7])
                    #
                    # sphere9 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[8])
                    # sphere10 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[9])
                    # sphere11 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[10])
                    # sphere12 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate( transformed_landmarks[11])
                    #
                    # sphere13 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[12])
                    # sphere14 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[13])
                    # sphere15 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[14])
                    # sphere16 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[15])
                    #
                    # sphere17 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[16])
                    # sphere18 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(transformed_landmarks[17])
                    # sphere19 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate( transformed_landmarks[18])
                    # sphere20 = o3d.geometry.TriangleMesh.create_sphere(radius=1).translate( transformed_landmarks[19])
                    #
                    #
                    # def paint_spheres_yellow(vertebra, *spheres):
                    #     for sphere in spheres:
                    #         sphere.paint_uniform_color([1, 1, 0])  # Yellow
                    #
                    #
                    #
                    # # Call the function for each set of geometries
                    # paint_spheres_yellow(vertebra1, sphere1, sphere2, sphere3, sphere4)
                    # paint_spheres_yellow(vertebra2, sphere5, sphere6, sphere7, sphere8)
                    # paint_spheres_yellow(vertebra3, sphere9, sphere10, sphere11, sphere12)
                    # paint_spheres_yellow(vertebra4, sphere13, sphere14, sphere15, sphere16)
                    # paint_spheres_yellow(vertebra5, sphere17, sphere18, sphere19, sphere20)



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
                    o3d.io.write_triangle_mesh(os.path.join(save_data_dir,"transformed_vertebra1.stl"), vertebra1)
                    o3d.io.write_triangle_mesh(os.path.join(save_data_dir,"transformed_vertebra2.stl"), vertebra2)
                    o3d.io.write_triangle_mesh(os.path.join(save_data_dir,"transformed_vertebra3.stl"), vertebra3)
                    o3d.io.write_triangle_mesh(os.path.join(save_data_dir,"transformed_vertebra4.stl"), vertebra4)
                    o3d.io.write_triangle_mesh(os.path.join(save_data_dir,"transformed_vertebra5.stl"), vertebra5)
                    o3d.io.write_triangle_mesh(os.path.join(save_data_dir,"transformed_spine.stl"), vertebra1+vertebra2+vertebra3+vertebra4+vertebra5)


                    mask1_stl = trimesh.load(os.path.join(save_data_dir,"transformed_vertebra1.stl"))
                    mask2_stl = trimesh.load(os.path.join(save_data_dir,"transformed_vertebra2.stl"))
                    mask3_stl = trimesh.load(os.path.join(save_data_dir,"transformed_vertebra3.stl"))
                    mask4_stl = trimesh.load(os.path.join(save_data_dir,"transformed_vertebra4.stl"))
                    mask5_stl = trimesh.load(os.path.join(save_data_dir,"transformed_vertebra5.stl"))
                    maskspine_stl = trimesh.load(os.path.join(save_data_dir,"transformed_spine.stl"))


                    mesh1 = pyrender.Mesh.from_trimesh(mask1_stl)
                    mesh2 = pyrender.Mesh.from_trimesh(mask2_stl)
                    mesh3 = pyrender.Mesh.from_trimesh(mask3_stl)
                    mesh4 = pyrender.Mesh.from_trimesh(mask4_stl)
                    mesh5 = pyrender.Mesh.from_trimesh(mask5_stl)
                    mesh6 = pyrender.Mesh.from_trimesh(maskspine_stl)



                    scene1 = pyrender.Scene()
                    scene2 = pyrender.Scene()
                    scene3 = pyrender.Scene()
                    scene4 = pyrender.Scene()
                    scene5 = pyrender.Scene()
                    scene6 = pyrender.Scene()



                    scene1.add(mesh1, pose=extrinsics_matrix)
                    scene2.add(mesh2, pose=extrinsics_matrix)
                    scene3.add(mesh3, pose=extrinsics_matrix)
                    scene4.add(mesh4, pose=extrinsics_matrix)
                    scene5.add(mesh5, pose=extrinsics_matrix)
                    scene6.add(mesh6, pose=extrinsics_matrix)



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
                    scene6.add(camera, pose=rotate * rotate2)


                    r1 = pyrender.OffscreenRenderer(1920, 1080)
                    r2 = pyrender.OffscreenRenderer(1920, 1080)
                    r3 = pyrender.OffscreenRenderer(1920, 1080)
                    r4 = pyrender.OffscreenRenderer(1920, 1080)
                    r5 = pyrender.OffscreenRenderer(1920, 1080)
                    r6 = pyrender.OffscreenRenderer(1920, 1080)


                    _, depth1 = r1.render(scene1)
                    _, depth2 = r1.render(scene2)
                    _, depth3 = r1.render(scene3)
                    _, depth4 = r1.render(scene4)
                    _, depth5 = r1.render(scene5)
                    _, depth6 = r1.render(scene6)


                    cv2.imwrite(os.path.join(save_data_dir,"mask1.png"), depth1)
                    cv2.imwrite(os.path.join(save_data_dir,"mask2.png"), depth2)
                    cv2.imwrite(os.path.join(save_data_dir,"mask3.png"), depth3)
                    cv2.imwrite(os.path.join(save_data_dir,"mask4.png"), depth4)
                    cv2.imwrite(os.path.join(save_data_dir,"mask5.png"), depth5)
                    cv2.imwrite(os.path.join(save_data_dir,"mask.png"), depth6)


                    mask_image1 = cv2.imread(os.path.join(save_data_dir,"mask1.png"))
                    mask_image2 = cv2.imread(os.path.join(save_data_dir,"mask2.png"))
                    mask_image3 = cv2.imread(os.path.join(save_data_dir,"mask3.png"))
                    mask_image4 = cv2.imread(os.path.join(save_data_dir,"mask4.png"))
                    mask_image5 = cv2.imread(os.path.join(save_data_dir,"mask5.png"))




                    mask_all_zero_1 = np.all(mask_image1 == 255,axis=2)
                    mask_all_zero_2 = np.all(mask_image2 == 255,axis=2)
                    mask_all_zero_3 = np.all(mask_image3 == 255,axis=2)
                    mask_all_zero_4 = np.all(mask_image4 == 255,axis=2)
                    mask_all_zero_5 = np.all(mask_image5 == 255,axis=2)
                    mask_all_zero_6 = np.all(mask_image6 == 0,axis=2)


                    # Get indices where mask is True
                    indices1 = np.argwhere(mask_all_zero_1)
                    indices2 = np.argwhere(mask_all_zero_2)
                    indices3 = np.argwhere(mask_all_zero_3)
                    indices4 = np.argwhere(mask_all_zero_4)
                    indices5 = np.argwhere(mask_all_zero_5)
                    indices6 = np.argwhere(mask_all_zero_6)

                    numpy_array1 = image.get_data()

                    # Loop over the indices
                    for j, k in indices1:
                        rgba = numpy_array1[j][k][:]
                        if np.isnan(rgba.all()) == False:
                            rgba[0] = 255
                            rgba[1] = 0
                            rgba[2] = 0
                            rgba[3] = 255
                            numpy_array1[j][k][:] = rgba
                    for j, k in indices2:
                        rgba = numpy_array1[j][k][:]
                        if np.isnan(rgba.all()) == False:
                            rgba[0] = 0
                            rgba[1] = 255
                            rgba[2] = 0
                            rgba[3] = 255
                            numpy_array1[j][k][:] = rgba
                    for j, k in indices3:
                        rgba = numpy_array1[j][k][:]
                        if np.isnan(rgba.all()) == False:
                            rgba[0] = 0
                            rgba[1] = 0
                            rgba[2] = 255
                            rgba[3] = 255
                            numpy_array1[j][k][:] = rgba

                    for j, k in indices4:
                        rgba = numpy_array1[j][k][:]
                        if np.isnan(rgba.all()) == False:

                            rgba[0] = 255
                            rgba[1] = 192
                            rgba[2] = 203
                            rgba[3] = 255
                            numpy_array1[j][k][:] = rgba

                    for j, k in indices5:
                        rgba = numpy_array1[j][k][:]
                        if np.isnan(rgba.all()) == False:
                            rgba[0] = 255
                            rgba[1] = 165
                            rgba[2] = 0
                            rgba[3] = 255
                            numpy_array1[j][k][:] = rgba


                    for j, k in indices6:
                        rgba = numpy_array1[j][k][:]
                        if np.isnan(rgba.all()) == False:
                            rgba[0] = 0
                            rgba[1] = 0
                            rgba[2] = 0
                            rgba[3] = 255
                            numpy_array1[j][k][:] = rgba


                    image.write(os.path.join(save_data_dir, 'labelled_image.png'))

                    from PIL import Image
                    color_image = np.asarray(Image.open(os.path.join(save_data_dir, 'image.png')))
                    label_image = np.asarray(numpy_array1)
                    depth_image = np.asarray(depth.get_data())

                    # Get image dimensions
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
                    point_cloud_label = np.stack((x, y, z, label_image[:, :, 0] / 255, label_image[:, :, 1] / 255,
                                                 label_image[:, :, 2] / 255), axis=-1).reshape(
                        -1, 6)

                    point_cloud_orig = np.nan_to_num(point_cloud_orig)
                    point_cloud_label = np.nan_to_num(point_cloud_label)

                    # Create open3d point cloud
                    pcd_orig = o3d.geometry.PointCloud()
                    pcd_orig.points = o3d.utility.Vector3dVector(point_cloud_orig[:, :3])
                    pcd_orig.colors = o3d.utility.Vector3dVector(point_cloud_orig[:, 3:6])

                    pcd_label = o3d.geometry.PointCloud()
                    pcd_label.points = o3d.utility.Vector3dVector(point_cloud_label[:, :3])
                    pcd_label.colors = o3d.utility.Vector3dVector(point_cloud_label[:, 3:6])

                    o3d.io.write_point_cloud(os.path.join(save_data_dir, "labelled_pcd.pcd"), pcd_label)
                    o3d.io.write_point_cloud(os.path.join(save_data_dir, 'full_pcd.pcd'),pcd_orig)

                    cur_frame += 1

                    # pcd1 = o3d.io.read_point_cloud(c)
                    # pcd2 = o3d.io.read_point_cloud(os.path.join(save_data_dir, "pointcloud_vert2.ply"))
                    # pcd3 = o3d.io.read_point_cloud(os.path.join(save_data_dir, "pointcloud_vert3.ply"))
                    # pcd4 = o3d.io.read_point_cloud(os.path.join(save_data_dir, "pointcloud_vert4.ply"))
                    # pcd5 = o3d.io.read_point_cloud(os.path.join(save_data_dir, "pointcloud_vert5.ply"))
                    #
                    # inverse_vertebra1_pose = np.linalg.inv(vertebra1_pose)
                    # inverse_vertebra2_pose = np.linalg.inv(vertebra2_pose)
                    # inverse_vertebra3_pose = np.linalg.inv(vertebra3_pose)
                    # inverse_vertebra4_pose = np.linalg.inv(vertebra4_pose)
                    # inverse_vertebra5_pose = np.linalg.inv(vertebra5_pose)
                    #
                    # vertebra1.transform(inverse_vertebra1_pose)
                    # vertebra2.transform(inverse_vertebra2_pose)
                    # vertebra3.transform(inverse_vertebra3_pose)
                    # vertebra4.transform(inverse_vertebra4_pose)
                    # vertebra5.transform(inverse_vertebra5_pose)
                    #
                    #
                    # pcd1.transform(inverse_vertebra1_pose)
                    # pcd2.transform(inverse_vertebra2_pose)
                    # pcd3.transform(inverse_vertebra3_pose)
                    # pcd4.transform(inverse_vertebra4_pose)
                    # pcd5.transform(inverse_vertebra5_pose)
                    #
                    #
                    #
                    # bbox1 = vertebra1.get_oriented_bounding_box()
                    # bbox2 = vertebra2.get_oriented_bounding_box()
                    # bbox3 = vertebra3.get_oriented_bounding_box()
                    # bbox4 = vertebra4.get_oriented_bounding_box()
                    # bbox5 = vertebra5.get_oriented_bounding_box()
                    #
                    # pcd1 = pcd1.crop(bbox1)
                    # pcd2 = pcd2.crop(bbox2)
                    # pcd3 = pcd3.crop(bbox3)
                    # pcd4 = pcd4.crop(bbox4)
                    # pcd5 = pcd5.crop(bbox5)
                    #
                    # o3d.io.write_point_cloud(os.path.join(save_data_dir, "pointcloud_vert1_cropped.ply"), pcd1)
                    # o3d.io.write_point_cloud(os.path.join(save_data_dir, "pointcloud_vert2_cropped.ply"), pcd2)
                    # o3d.io.write_point_cloud(os.path.join(save_data_dir, "pointcloud_vert3_cropped.ply"), pcd3)
                    # o3d.io.write_point_cloud(os.path.join(save_data_dir, "pointcloud_vert4_cropped.ply"), pcd4)
                    # o3d.io.write_point_cloud(os.path.join(save_data_dir, "pointcloud_vert5_cropped.ply"), pcd5)
                    #
                    #
                    # def inverse_transform_planning(transformation,*spheres):
                    #     for sphere in spheres:
                    #         sphere.transform(transformation)
                    # inverse_transform_planning(inverse_vertebra1_pose,sphere1, sphere2, sphere3, sphere4)
                    # inverse_transform_planning(inverse_vertebra2_pose,sphere5, sphere6, sphere7, sphere8)
                    # inverse_transform_planning(inverse_vertebra3_pose,sphere9, sphere10, sphere11, sphere12)
                    # inverse_transform_planning(inverse_vertebra4_pose,sphere13, sphere14, sphere15, sphere16)
                    # inverse_transform_planning(inverse_vertebra5_pose,sphere17, sphere18, sphere19, sphere20)
                    #
                    #
                    #
                    # transformed_landmarks[0]=sphere1.get_center()
                    # transformed_landmarks[1]=sphere2.get_center()
                    # transformed_landmarks[2]=sphere3.get_center()
                    # transformed_landmarks[3]=sphere4.get_center()
                    # transformed_landmarks[4]=sphere5.get_center()
                    # transformed_landmarks[5]=sphere6.get_center()
                    # transformed_landmarks[6]=sphere7.get_center()
                    # transformed_landmarks[7]=sphere8.get_center()
                    # transformed_landmarks[8]=sphere9.get_center()
                    # transformed_landmarks[9]=sphere10.get_center()
                    # transformed_landmarks[10]=sphere11.get_center()
                    # transformed_landmarks[11]=sphere12.get_center()
                    # transformed_landmarks[12]=sphere13.get_center()
                    # transformed_landmarks[13]=sphere14.get_center()
                    # transformed_landmarks[14]=sphere15.get_center()
                    # transformed_landmarks[15]=sphere16.get_center()
                    # transformed_landmarks[16]=sphere17.get_center()
                    # transformed_landmarks[17]=sphere18.get_center()
                    # transformed_landmarks[18]=sphere19.get_center()
                    # transformed_landmarks[19]=sphere20.get_center()
                    #
                    #
                    #
                    #
                    # np.savetxt(os.path.join(save_data_dir,"planning.txt"), transformed_landmarks)
                    #
                    #
                    #



print("time taken: ", time.time() - start_time)