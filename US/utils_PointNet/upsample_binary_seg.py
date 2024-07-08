import os

import open3d as o3d
import numpy as np
def nearest_neighbor_mapping(original_pc, downsampled_pc):
    """
    Map the labels from the downsampled point cloud to the original point cloud using nearest neighbor search.

    Args:
        original_pc (o3d.geometry.PointCloud): The original point cloud.
        downsampled_pc (o3d.geometry.PointCloud): The downsampled point cloud with labels.

    Returns:
        np.ndarray: Labels for the original point cloud.
    """
    # Extract points from both point clouds
    original_points = np.asarray(original_pc.points)

    # Create a KDTree from the downsampled points
    kdtree = o3d.geometry.KDTreeFlann(original_points.T)


    # Map each point in the original point cloud to the nearest neighbor in the  downsampled point cloud
    for i, point in enumerate(downsampled_pc):
        k, idx, _ = kdtree.search_knn_vector_3d(point, 2000)
        np.asarray(original_pc.colors)[idx[1:], :] = [1, 0, 0]

    return original_pc

dir = "/Users/aidanamassalimova/Documents/binaryRGBDcolored/binary_seg"
orig_dir = "/Volumes/SpineDepth/PoinTr_dataset/segmented_spinedepth_new"

filenames = os.listdir(dir)

for filename in filenames:
    print(filename)
    _, specimen,_, recording,_ ,cam,_, frame = filename.split('_')
    binary_seg = o3d.io.read_point_cloud(os.path.join(dir, filename))

    seg_numpy = np.asarray(binary_seg.points)
    # Compute the axis-aligned bounding box (AABB)
    aabb = binary_seg.get_axis_aligned_bounding_box()

    # Optionally, compute the oriented bounding box (OBB)
    obb = binary_seg.get_oriented_bounding_box()
    aabb.color = (1, 0, 0)  # Paint the AABB red
    full_pcd_labelled = o3d.io.read_point_cloud(os.path.join(orig_dir,"Specimen_"+specimen,"recording_" +recording, "cam_"+cam, "frame_" +frame[0], "labelled_pcd.pcd" ))


    upsampled_pcd = nearest_neighbor_mapping(full_pcd_labelled,seg_numpy)
    o3d.visualization.draw_geometries([upsampled_pcd,aabb])
