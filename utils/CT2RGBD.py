import os

import pyvista as pv
import numpy as np
import open3d as o3d

# Load the STL file
dir =r"F:\shapeCompletion\stls from ct\from Sascha"
num_points_to_sample = 4096  # Adjust this number according to your needs

target = o3d.io.read_triangle_mesh(r"F:\shapeCompletion\stls from ct\from Sascha\1.3.6.1.4.1.9328.50.4.0001_L2.stl")
target_pcd = target.sample_points_uniformly(num_points_to_sample)
bounding_box_target = target.get_minimal_oriented_bounding_box()
obb_matrix_target = bounding_box_target.R
center = target.get_center()
print(center)
files =os.listdir(dir)
for el,file in enumerate(files):

    if el <int(0.8*len(os.listdir(dir))):
        save_dir = r"F:\shapeCompletion\dataset_\train"
    else:
        save_dir = r"F:\dataset_ply\val"
    #save_dir = r"F:\dataset\test"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(file)

    mesh1 = o3d.io.read_triangle_mesh(os.path.join(dir,file))
    mesh_ds = mesh1.sample_points_uniformly(num_points_to_sample)
    bounding_box = mesh1.get_minimal_oriented_bounding_box()
    obb_matrix = bounding_box.R

    transformation = np.matmul(np.linalg.inv(obb_matrix),obb_matrix_target)

    mesh_ds.translate(center, relative=False)
    #R = mesh_ds.get_rotation_matrix_from_xyz((0, 0, np.pi))
    #mesh_ds.rotate(R, center=center)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        mesh_ds, target_pcd, 1, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # Apply the transformation to the target mesh
    mesh_ds.transform(reg_p2p.transformation)
  #  o3d.visualization.draw_geometries([mesh_ds, target_pcd])

    # The orientation matrix is the 3x3 upper-left submatrix of obb_matrix
    orientation_matrix = obb_matrix_target[:3, :3]

    # The principal axes are the columns of the orientation matrix
    axis_x = orientation_matrix[:, 0]
    axis_y = orientation_matrix[:, 1]
    axis_z = orientation_matrix[:, 2]




    # Get the coordinates of the sampled points as a NumPy array
    sampled_point_cloud = pv.PolyData(np.asarray(mesh_ds.points))



    # Compute the centroid of the mesh
    centroid = sampled_point_cloud.center_of_mass()
    split_origin = [centroid[0],-12+centroid[1], centroid[2]]

    split_normal= axis_x
    split_plane = pv.Plane(center=split_origin, direction=split_normal, i_size=150, j_size=150)

    # Use the `clip` function to split the mesh based on the plane
    mesh_part1= sampled_point_cloud.clip(normal=-split_normal, origin=split_origin)
    plotter = pv.Plotter()

    # Add the point cloud to the plotter
    plotter.add_mesh(sampled_point_cloud, color='blue', point_size=3)

    # Add the plane to the plotter
    plotter.add_mesh(split_plane, color='gray', opacity=0.5)

   # plotter.show()

    # Adjust the camera position if needed

    # Show the plotter

    sampled_points_array_temp = np.array(mesh_part1.points)

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

    print(downsampled_point_cloud)
    #o3d.visualization.draw_geometries([downsampled_point_cloud])

   # o3d.visualization.draw_geometries([pcd_output])
    print(os.path.join(save_dir,"partial",file[:-4]))
    if not os.path.exists(os.path.join(save_dir,"partial",file[:-4])):
        os.makedirs(os.path.join(save_dir,"partial",file[:-4]))
    o3d.io.write_point_cloud(os.path.join(save_dir, "partial",file[:-4], "00.ply"), pcd_input)
    if not os.path.exists(os.path.join(save_dir, "complete")):
        os.makedirs(os.path.join(save_dir, "complete"))
    o3d.io.write_point_cloud(os.path.join(save_dir, "complete",file.replace("stl", "ply")), mesh_ds)
