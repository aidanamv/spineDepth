import os
import pyvista as pv
import numpy as np
import open3d as o3d
import copy

# Load the STL file
dir ="/media/aidana/US study/segmented_spinedepth"
save_dir ="/media/aidana/US study/aligned"
num_points_to_sample = 16384
target = o3d.io.read_triangle_mesh("/media/aidana/US study/target.stl")
target_pcd = target.sample_points_uniformly(num_points_to_sample)
bounding_box_target = target.get_oriented_bounding_box()
obb_matrix_target = bounding_box_target.R
orientation_matrix = obb_matrix_target[:3, :3]
# The principal axes are the columns of the orientation matrix
axis_x = orientation_matrix[:, 0]
axis_y = orientation_matrix[:, 1]
axis_z = orientation_matrix[:, 2]
center_target = target.get_center()
specimens = os.listdir(dir)
cameras =[0,1]
#specimens = ["Specimen_5"] #the principal axis was in negative direction Specimen_5 camera 1
num_frames = 10
for specimen in specimens:
    recordings = os.listdir(os.path.join(dir, specimen))
    for recording in recordings:
        files = os.listdir(os.path.join(dir, specimen, recording))
        for file in files:
            try:
                camera = file.split("_")[2]
                frame = file.split("_")[4]
                print("Processing specimen {} recording {} file {}".format(specimen,recording,file))
                f_path = os.path.join(dir,specimen,recording,file)
                mesh1 = o3d.io.read_triangle_mesh(os.path.join(f_path,"transformed_vertebra1.stl"))
                mesh2 = o3d.io.read_triangle_mesh(os.path.join(f_path,"transformed_vertebra2.stl"))
                mesh3 = o3d.io.read_triangle_mesh(os.path.join(f_path,"transformed_vertebra3.stl"))
                mesh4 = o3d.io.read_triangle_mesh(os.path.join(f_path,"transformed_vertebra4.stl"))
                mesh5 = o3d.io.read_triangle_mesh(os.path.join(f_path,"transformed_vertebra5.stl"))

                pcd1 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert1.ply"))
                pcd2 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert2.ply"))
                pcd3 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert3.ply"))
                pcd4 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert4.ply"))
                pcd5 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert5.ply"))

                mesh1_pcd = mesh1.sample_points_uniformly(num_points_to_sample)
                mesh2_pcd = mesh2.sample_points_uniformly(num_points_to_sample)
                mesh3_pcd = mesh3.sample_points_uniformly(num_points_to_sample)
                mesh4_pcd = mesh4.sample_points_uniformly(num_points_to_sample)
                mesh5_pcd = mesh5.sample_points_uniformly(num_points_to_sample)



                center1 = mesh1_pcd.get_center()
                center2 = mesh2_pcd.get_center()
                center3 = mesh3_pcd.get_center()
                center4 = mesh4_pcd.get_center()
                center5 = mesh5_pcd.get_center()




                mesh1_pcd.translate(center_target, relative=False)
                mesh2_pcd.translate(center_target, relative=False)
                mesh3_pcd.translate(center_target, relative=False)
                mesh4_pcd.translate(center_target, relative=False)
                mesh5_pcd.translate(center_target, relative=False)

                pcd1.translate(center_target-center1, relative=True)
                pcd2.translate(center_target-center2, relative=True)
                pcd3.translate(center_target-center3, relative=True)
                pcd4.translate(center_target-center4, relative=True)
                pcd5.translate(center_target-center5, relative=True)

                # Compute the principal axes (eigenvalues and eigenvectors) of both point clouds
                _, eigvec1 = np.linalg.eigh(np.cov(np.transpose(np.asarray(target_pcd.points))))
                _, eigvec2 = np.linalg.eigh(np.cov(np.transpose(np.asarray(mesh1_pcd.points))))

                # Calculate the rotation matrix to align the principal axes of point_cloud2 with point_cloud1
                if specimen!="Specimen_5" and camera==1:
                    rotation_matrix = np.dot(eigvec1, eigvec2.T)
                else:
                    rotation_matrix = np.dot(eigvec1, - eigvec2.T)

                # Apply the rotation to align point_cloud2 with point_cloud1
                mesh1_pcd.rotate(rotation_matrix, center=mesh1_pcd.get_center())
                mesh2_pcd.rotate(rotation_matrix, center=mesh2_pcd.get_center())
                mesh3_pcd.rotate(rotation_matrix, center=mesh3_pcd.get_center())
                mesh4_pcd.rotate(rotation_matrix, center=mesh4_pcd.get_center())
                mesh5_pcd.rotate(rotation_matrix, center=mesh5_pcd.get_center())


                pcd1.rotate(rotation_matrix, center=mesh1_pcd.get_center())
                pcd2.rotate(rotation_matrix, center=mesh2_pcd.get_center())
                pcd3.rotate(rotation_matrix, center=mesh3_pcd.get_center())
                pcd4.rotate(rotation_matrix, center=mesh4_pcd.get_center())
                pcd5.rotate(rotation_matrix, center=mesh5_pcd.get_center())


                #o3d.visualization.draw_geometries([mesh1_pcd,target_pcd, mesh2_pcd, mesh3_pcd, mesh4_pcd, mesh5_pcd])

                bbox1 = mesh1_pcd.get_oriented_bounding_box()


                bbox2 = mesh2_pcd.get_oriented_bounding_box()

                bbox3 = mesh3_pcd.get_oriented_bounding_box()


                bbox4 = mesh4_pcd.get_oriented_bounding_box()

                bbox5 = mesh5_pcd.get_oriented_bounding_box()


                pcd1=pcd1.crop(bbox1)
                pcd2=pcd2.crop(bbox2)
                pcd3=pcd3.crop(bbox3)
                pcd4=pcd4.crop(bbox4)
                pcd5=pcd5.crop(bbox5)







                current_point_count1 = np.asarray(mesh1_pcd.points).shape[0]
                current_point_count2 = np.asarray(mesh2_pcd.points).shape[0]
                current_point_count3 = np.asarray(mesh3_pcd.points).shape[0]
                current_point_count4 = np.asarray(mesh4_pcd.points).shape[0]
                current_point_count5 = np.asarray(mesh5_pcd.points).shape[0]

                random_indices1 = np.random.choice(current_point_count1, num_points_to_sample-2048, replace=False)
                random_indices2 = np.random.choice(current_point_count2, num_points_to_sample-2048, replace=False)
                random_indices3 = np.random.choice(current_point_count3, num_points_to_sample-2048, replace=False)
                random_indices4 = np.random.choice(current_point_count4, num_points_to_sample-2048, replace=False)
                random_indices5 = np.random.choice(current_point_count5, num_points_to_sample-2048, replace=False)

                # Select the subset of points based on the random indices
                downsampled_point_cloud1 = mesh1_pcd.select_by_index(random_indices1)
                downsampled_point_cloud2 = mesh2_pcd.select_by_index(random_indices2)
                downsampled_point_cloud3 = mesh3_pcd.select_by_index(random_indices3)
                downsampled_point_cloud4 = mesh4_pcd.select_by_index(random_indices4)
                downsampled_point_cloud5 = mesh5_pcd.select_by_index(random_indices5)


                #downsample input pcds

                random_indices11= np.random.choice(np.asarray(pcd1.points).shape[0], 2048, replace=False)
                random_indices22= np.random.choice(np.asarray(pcd2.points).shape[0], 2048, replace=False)
                random_indices33= np.random.choice(np.asarray(pcd3.points).shape[0], 2048, replace=False)
                random_indices44= np.random.choice(np.asarray(pcd4.points).shape[0], 2048, replace=False)
                random_indices55= np.random.choice(np.asarray(pcd5.points).shape[0], 2048, replace=False)

                downsampled_pcd1 = pcd1.select_by_index(random_indices11)
                downsampled_pcd2 = pcd2.select_by_index(random_indices22)
                downsampled_pcd3 = pcd3.select_by_index(random_indices33)
                downsampled_pcd4 = pcd4.select_by_index(random_indices44)
                downsampled_pcd5 = pcd5.select_by_index(random_indices55)

                downsampled_point_cloud1= copy.deepcopy(downsampled_pcd1) + downsampled_point_cloud1
                downsampled_point_cloud2= copy.deepcopy(downsampled_pcd2) + downsampled_point_cloud2
                downsampled_point_cloud3= copy.deepcopy(downsampled_pcd3) + downsampled_point_cloud3
                downsampled_point_cloud4= copy.deepcopy(downsampled_pcd4) + downsampled_point_cloud4
                downsampled_point_cloud5= copy.deepcopy(downsampled_pcd5) + downsampled_point_cloud5


                def check(pcd1, pcd2, window_name):
                    pcd2.paint_uniform_color([0, 0, 1])
                    pcd1.paint_uniform_color([0, 1, 0])
                    o3d.visualization.draw_geometries([pcd1,pcd2], window_name=window_name, width=800, height=600, left=50, top=50)


                # check(downsampled_point_cloud1, downsampled_pcd1, window_name="L1")
                # check(downsampled_point_cloud2, downsampled_pcd2, window_name="L2")
                # check(downsampled_point_cloud3, downsampled_pcd3, window_name="L3")
                # check(downsampled_point_cloud4, downsampled_pcd4, window_name="L4")
                # check(downsampled_point_cloud5, downsampled_pcd5, window_name="L5")
                #

                pcd_inputs =[downsampled_pcd1, downsampled_pcd2, downsampled_pcd3, downsampled_pcd4, downsampled_pcd5]



                for el, pcd in enumerate([downsampled_point_cloud1, downsampled_point_cloud2, downsampled_point_cloud3, downsampled_point_cloud4, downsampled_point_cloud5]):
                    assert np.asarray(pcd.points).shape[0] == num_points_to_sample
                    assert np.asarray(pcd_inputs[el].points).shape[0] == 2048
                    filename = specimen + "recording_{}".format(recording)+"_camera_{}".format(camera)+"_frame_{}".format(frame)+"_L{}".format(el+1)
                    if not os.path.exists(os.path.join(save_dir,"partial",filename)):
                        os.makedirs(os.path.join(save_dir,"partial",filename))
                    o3d.io.write_point_cloud(os.path.join(save_dir, "partial",filename, "00.pcd"), pcd_inputs[el])
                    if not os.path.exists(os.path.join(save_dir, "complete")):
                        os.makedirs(os.path.join(save_dir, "complete"))
                    o3d.io.write_point_cloud(os.path.join(save_dir, "complete",filename+ ".pcd"), pcd)
            except:
                print("error in recording: ", recording, "camera: ", camera, "frame: ", frame)
                continue
