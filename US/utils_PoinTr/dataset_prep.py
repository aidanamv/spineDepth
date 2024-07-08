import os
import open3d as o3d
import numpy as np
import random
import pyvista as pv
from sklearn.decomposition import PCA

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()




def randomly_downsample_point_cloud(point_cloud, num_points=4096):
    # Check total number of points
    total_points = point_cloud.shape[0]

    if total_points <= num_points:
        return point_cloud  # No need to downsample if already <= num_points

    # Randomly select num_points indices from total_points
    selected_indices = np.random.choice(total_points, num_points, replace=False)

    # Extract the subset of point cloud using the selected indices
    downsampled_point_cloud = point_cloud[selected_indices]

    return downsampled_point_cloud
def create_planes_through_obb(obb):
    obb_center = np.array(obb.get_center())
    obb_orientation = np.array(obb.get_rotation_matrix())
    obb_half_lengths = np.array(obb.get_half_lengths())

    # Define plane offsets relative to the OBB center
    plane_offsets = [
        np.array([0.0, 0.0, 0.0]),  # Plane through the OBB center (middle)
        obb_orientation[:, 0] * obb_half_lengths[0],  # Plane along the first axis
        obb_orientation[:, 1] * obb_half_lengths[1],  # Plane along the second axis
        obb_orientation[:, 2] * obb_half_lengths[2],  # Plane along the third axis
    ]

    # Create plane points
    plane_points = []
    for offset in plane_offsets:
        plane_points.append(obb_center + offset)

    return plane_points
dir = "/Users/aidanamassalimova/Documents/fold_3"

input_points = 2048
output_points = 4096

train_files = os.listdir(os.path.join(dir, "train","point_cloud"))
val_files = os.listdir(os.path.join(dir, "val","point_cloud"))

save_dir = "/Users/aidanamassalimova/Documents/pointr_data/fold_3"



color_map = [[115, 21, 19], [155, 224, 138],[181, 10, 236], [75, 75, 251], [246, 103, 178]]
for file in train_files:
    print(file)
      #  if not os.path.exists(os.path.join(save_train_dir_partial,specimen+"L"+str(level),"00.pcd")):
    partial  =pv.read(os.path.join(dir,"train","point_cloud", file))
    points = np.asarray(partial.points)
    colors = np.asarray(partial["colors"])
    mesh1 = o3d.io.read_triangle_mesh(os.path.join(dir,"train","stls", file.replace(".vtp","_L1.stl")))
    mesh2 = o3d.io.read_triangle_mesh(os.path.join(dir,"train","stls", file.replace(".vtp","_L2.stl")))
    mesh3 = o3d.io.read_triangle_mesh(os.path.join(dir,"train","stls", file.replace(".vtp","_L3.stl")))
    mesh4 = o3d.io.read_triangle_mesh(os.path.join(dir,"train","stls", file.replace(".vtp","_L4.stl")))
    mesh5 = o3d.io.read_triangle_mesh(os.path.join(dir,"train","stls", file.replace(".vtp","_L5.stl")))

    mesh = mesh1 + mesh2 + mesh3 + mesh4 + mesh5
    mesh.compute_vertex_normals()
    spine = pv.PolyData(np.asarray(mesh.vertices))
    mesh1_pcd = pv.PolyData(np.asarray(mesh1.vertices))
    mesh2_pcd = pv.PolyData(np.asarray(mesh2.vertices))
    mesh3_pcd = pv.PolyData(np.asarray(mesh3.vertices))
    mesh4_pcd = pv.PolyData(np.asarray(mesh4.vertices))
    mesh5_pcd = pv.PolyData(np.asarray(mesh5.vertices))

    centroid = spine.center_of_mass()
    bounding_box_target = mesh.get_minimal_oriented_bounding_box()
    obb_matrix_target = bounding_box_target.R
    split_origin = [centroid[0], -120 + centroid[1], centroid[2]]
    orientation_matrix = obb_matrix_target[:3, :3]

    # The principal axes are the columns of the orientation matrix
    axis_x = orientation_matrix[:, 0]
    axis_y = orientation_matrix[:, 1]
    axis_z = orientation_matrix[:, 2]

    split_normal = axis_y
    split_plane = pv.Plane(center=split_origin, direction=split_normal, i_size=250, j_size=250)

    # Use the `clip` spine to split the mesh based on the plane
    mesh_cropped = spine.clip(normal=-split_normal, origin=split_origin, invert=False)




    min_bounds = bounding_box_target.get_min_bound()  # Replace with actual method to get min bounds
    max_bounds = bounding_box_target.get_max_bound()  # Replace with actual method to get max bounds


    cropped_cloud = []
    cropped_colors = []
    min_x, min_y, min_z = min_bounds
    max_x, max_y, max_z = max_bounds

    for el,point in enumerate(points):
        x, y, z = point
        if min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z:
            cropped_cloud.append(point)
            cropped_colors.append(colors[el,:]/255)
    cropped_pcd= pv.PolyData(cropped_cloud)
    cropped_pcd["colors"] = cropped_colors




    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cropped_pcd.points)
    pcd.colors = o3d.utility.Vector3dVector(cropped_colors)

    ct = o3d.geometry.PointCloud()
    ct.points = o3d.utility.Vector3dVector(mesh_cropped.points)
    ct.paint_uniform_color([1,0,1])

    o3d.visualization.draw_geometries([pcd, ct])

    reg_p2p = o3d.pipelines.registration.registration_icp(
        ct, pcd, 1, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000000))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    ct.transform(reg_p2p.transformation)
    o3d.visualization.draw_geometries([pcd, ct])



