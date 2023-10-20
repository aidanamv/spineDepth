import os
import pyvista as pv
import numpy as np
import open3d as o3d

# Load the STL file
dir =r"J:\segmented_spinedepth"
save_dir =r"J:\segmented_spinedepth\aligned"
num_points_to_sample = 16384
target = o3d.io.read_triangle_mesh(r"F:\shapeCompletion\stls from ct\from Sascha\1.3.6.1.4.1.9328.50.4.0001_L2.stl")
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

for specimen in specimens:
    for camera in cameras:
        print("Processing specimen {} camera {}".format(specimen,camera))
        f_path = os.path.join(dir,specimen,"pointcloud_{}".format(camera))

        mesh1 = o3d.io.read_triangle_mesh(os.path.join(f_path,"transformed_vertebra1.stl"))
        mesh2 = o3d.io.read_triangle_mesh(os.path.join(f_path,"transformed_vertebra2.stl"))
        mesh3 = o3d.io.read_triangle_mesh(os.path.join(f_path,"transformed_vertebra3.stl"))
        mesh4 = o3d.io.read_triangle_mesh(os.path.join(f_path,"transformed_vertebra4.stl"))
        mesh5 = o3d.io.read_triangle_mesh(os.path.join(f_path,"transformed_vertebra5.stl"))

        pcd1 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert1_downsampled.ply"))
        pcd2 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert2_downsampled.ply"))
        pcd3 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert3_downsampled.ply"))
        pcd4 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert4_downsampled.ply"))
        pcd5 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert5_downsampled.ply"))

        mesh1_pcd = mesh1.sample_points_uniformly(num_points_to_sample)
        mesh2_pcd = mesh2.sample_points_uniformly(num_points_to_sample)
        mesh3_pcd = mesh3.sample_points_uniformly(num_points_to_sample)
        mesh4_pcd = mesh4.sample_points_uniformly(num_points_to_sample)
        mesh5_pcd = mesh5.sample_points_uniformly(num_points_to_sample)

        center1 = mesh1.get_center()
        center2 = mesh2.get_center()
        center3 = mesh3.get_center()
        center4 = mesh4.get_center()
        center5 = mesh5.get_center()


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
        rotation_matrix = np.dot(eigvec1, eigvec2.T)

        # Apply the rotation to align point_cloud2 with point_cloud1
        mesh1_pcd.rotate(rotation_matrix)
        mesh2_pcd.rotate(rotation_matrix)
        mesh3_pcd.rotate(rotation_matrix)
        mesh4_pcd.rotate(rotation_matrix)
        mesh5_pcd.rotate(rotation_matrix)


        pcd1.rotate(rotation_matrix)
        pcd2.rotate(rotation_matrix)
        pcd3.rotate(rotation_matrix)
        pcd4.rotate(rotation_matrix)
        pcd5.rotate(rotation_matrix)


        o3d.visualization.draw_geometries([mesh1_pcd,target_pcd, mesh2_pcd, mesh3_pcd, mesh4_pcd, mesh5_pcd])

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


        mesh1_pcd.paint_uniform_color([1, 0.706, 0])
        mesh2_pcd.paint_uniform_color([1, 0.706, 0])
        mesh3_pcd.paint_uniform_color([1, 0.706, 0])
        mesh4_pcd.paint_uniform_color([1, 0.706, 0])
        mesh5_pcd.paint_uniform_color([1, 0.706, 0])

        pcd1.paint_uniform_color([0, 0.651, 0.929])
        pcd2.paint_uniform_color([0, 0.651, 0.929])
        pcd3.paint_uniform_color([0, 0.651, 0.929])
        pcd4.paint_uniform_color([0, 0.651, 0.929])
        pcd5.paint_uniform_color([0, 0.651, 0.929])

        bbox1.color= (1, 0, 0)
        bbox2.color= (1, 0, 0)
        bbox3.color= (1, 0, 0)
        bbox4.color= (1, 0, 0)
        bbox5.color= (1, 0, 0)
        pcd1_pv= pv.PolyData(np.asarray(pcd1.points))
        pcd2_pv= pv.PolyData(np.asarray(pcd2.points))
        pcd3_pv = pv.PolyData(np.asarray(pcd3.points))
        pcd4_pv = pv.PolyData(np.asarray(pcd4.points))
        pcd5_pv = pv.PolyData(np.asarray(pcd5.points))

        mesh1_pv = pv.PolyData(np.asarray(mesh1_pcd.points))
        mesh2_pv = pv.PolyData(np.asarray(mesh2_pcd.points))
        mesh3_pv = pv.PolyData(np.asarray(mesh3_pcd.points))
        mesh4_pv = pv.PolyData(np.asarray(mesh4_pcd.points))
        mesh5_pv = pv.PolyData(np.asarray(mesh5_pcd.points))

        centroid1 =mesh1_pv.center_of_mass()
        centroid2 =mesh2_pv.center_of_mass()
        centroid3 =mesh3_pv.center_of_mass()
        centroid4 =mesh4_pv.center_of_mass()
        centroid5 =mesh5_pv.center_of_mass()

        split_origin1 = centroid1+np.array([0,-12,0])
        split_origin2 = centroid2+np.array([0,-12,0])
        split_origin3 = centroid3+np.array([0,-12,0])
        split_origin4 = centroid4+np.array([0,-12,0])
        split_origin5 = centroid5+np.array([0,-12,0])

        mesh_part1 = pcd1_pv.clip(normal=-axis_x, origin=split_origin1)
        mesh_part2 = pcd2_pv.clip(normal=-axis_x, origin=split_origin2)
        mesh_part3 = pcd3_pv.clip(normal=-axis_x, origin=split_origin3)
        mesh_part4 = pcd4_pv.clip(normal=-axis_x, origin=split_origin4)
        mesh_part5 = pcd5_pv.clip(normal=-axis_x, origin=split_origin5)

        split_plane1 = pv.Plane(center=split_origin1, direction=-axis_x, i_size=150, j_size=150)
        split_plane2 = pv.Plane(center=split_origin2, direction=-axis_x, i_size=150, j_size=150)
        split_plane3 = pv.Plane(center=split_origin3, direction=-axis_x, i_size=150, j_size=150)
        split_plane4 = pv.Plane(center=split_origin4, direction=-axis_x, i_size=150, j_size=150)
        split_plane5 = pv.Plane(center=split_origin5, direction=-axis_x, i_size=150, j_size=150)

        p1 = pv.Plotter()
        p2 = pv.Plotter()
        p3 = pv.Plotter()
        p4 = pv.Plotter()
        p5 = pv.Plotter()


        p1.add_mesh(mesh_part1, color="tan", show_edges=True)
        p2.add_mesh(mesh_part2, color="tan", show_edges=True)
        p3.add_mesh(mesh_part3, color="tan", show_edges=True)
        p4.add_mesh(mesh_part4, color="tan", show_edges=True)
        p5.add_mesh(mesh_part5, color="tan", show_edges=True)

        p1.add_mesh(split_plane1, color="red", opacity=0.5)
        p2.add_mesh(split_plane2, color="red", opacity=0.5)
        p3.add_mesh(split_plane3, color="red", opacity=0.5)
        p4.add_mesh(split_plane4, color="red", opacity=0.5)
        p5.add_mesh(split_plane5, color="red", opacity=0.5)

        p1.add_mesh(mesh1_pv, color="blue", show_edges=True)
        p2.add_mesh(mesh2_pv, color="blue", show_edges=True)
        p3.add_mesh(mesh3_pv, color="blue", show_edges=True)
        p4.add_mesh(mesh4_pv, color="blue", show_edges=True)
        p5.add_mesh(mesh5_pv, color="blue", show_edges=True)
        # p1.show()
        # p2.show()
        # p3.show()
        # p4.show()
        # p5.show()

        pcd1_cropped=o3d.geometry.PointCloud()
        pcd2_cropped=o3d.geometry.PointCloud()
        pcd3_cropped=o3d.geometry.PointCloud()
        pcd4_cropped=o3d.geometry.PointCloud()
        pcd5_cropped=o3d.geometry.PointCloud()

        pcd1_cropped.points = o3d.utility.Vector3dVector(mesh_part1.points)
        pcd2_cropped.points = o3d.utility.Vector3dVector(mesh_part2.points)
        pcd3_cropped.points = o3d.utility.Vector3dVector(mesh_part3.points)
        pcd4_cropped.points = o3d.utility.Vector3dVector(mesh_part4.points)
        pcd5_cropped.points = o3d.utility.Vector3dVector(mesh_part5.points)




        merged_cloud1 = mesh1_pcd + pcd1
        merged_cloud2 = mesh2_pcd + pcd2
        merged_cloud3 = mesh3_pcd + pcd3
        merged_cloud4 = mesh4_pcd + pcd4
        merged_cloud5 = mesh5_pcd + pcd5

        cleaned_point_cloud1 = merged_cloud1.remove_duplicated_points()
        cleaned_point_cloud2 = merged_cloud2.remove_duplicated_points()
        cleaned_point_cloud3 = merged_cloud3.remove_duplicated_points()
        cleaned_point_cloud4 = merged_cloud4.remove_duplicated_points()
        cleaned_point_cloud5 = merged_cloud5.remove_duplicated_points()

        current_point_count1 = np.asarray(cleaned_point_cloud1.points).shape[0]
        current_point_count2 = np.asarray(cleaned_point_cloud2.points).shape[0]
        current_point_count3 = np.asarray(cleaned_point_cloud3.points).shape[0]
        current_point_count4 = np.asarray(cleaned_point_cloud4.points).shape[0]
        current_point_count5 = np.asarray(cleaned_point_cloud5.points).shape[0]

        random_indices1 = np.random.choice(current_point_count1, num_points_to_sample, replace=False)
        random_indices2 = np.random.choice(current_point_count2, num_points_to_sample, replace=False)
        random_indices3 = np.random.choice(current_point_count3, num_points_to_sample, replace=False)
        random_indices4 = np.random.choice(current_point_count4, num_points_to_sample, replace=False)
        random_indices5 = np.random.choice(current_point_count5, num_points_to_sample, replace=False)

        # Select the subset of points based on the random indices
        downsampled_point_cloud1 = merged_cloud1.select_by_index(random_indices1)
        downsampled_point_cloud2 = merged_cloud2.select_by_index(random_indices2)
        downsampled_point_cloud3 = merged_cloud3.select_by_index(random_indices3)
        downsampled_point_cloud4 = merged_cloud4.select_by_index(random_indices4)
        downsampled_point_cloud5 = merged_cloud5.select_by_index(random_indices5)


        pcd_inputs =[pcd1, pcd2, pcd3, pcd4, pcd5]

        for el, pcd in enumerate([downsampled_point_cloud1, downsampled_point_cloud2, downsampled_point_cloud3, downsampled_point_cloud4, downsampled_point_cloud5]):
            filename = specimen + "_camera_{}".format(camera)+"_L{}".format(el+1)
            if not os.path.exists(os.path.join(save_dir,"partial",filename)):
                os.makedirs(os.path.join(save_dir,"partial",filename))
            o3d.io.write_point_cloud(os.path.join(save_dir, "partial",filename, "00.ply"), pcd_inputs[el])
            if not os.path.exists(os.path.join(save_dir, "complete")):
                os.makedirs(os.path.join(save_dir, "complete"))
            o3d.io.write_point_cloud(os.path.join(save_dir, "complete",filename+ ".ply"), pcd)


