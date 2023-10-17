import open3d as o3d
import numpy as np

def segmentation(pcd_combined,vertebra1, vertebra2, vertebra3, vertebra4, vertebra5):
    print("segmentation")
    pcd_L1 = o3d.t.geometry.TriangleMesh.from_legacy(vertebra1)
    pcd_L2 = o3d.t.geometry.TriangleMesh.from_legacy(vertebra2)
    pcd_L3 = o3d.t.geometry.TriangleMesh.from_legacy(vertebra3)
    pcd_L4 = o3d.t.geometry.TriangleMesh.from_legacy(vertebra4)
    pcd_L5 = o3d.t.geometry.TriangleMesh.from_legacy(vertebra5)
    pcd_array = np.asarray(pcd_combined.points)
    pcd_colors = np.array(pcd_combined.colors)
    close_to_L1 = []
    close_to_L2 = []
    close_to_L3 = []
    close_to_L4 = []
    close_to_L5 = []
    colors_L1 = []
    colors_L2 = []
    colors_L3 = []
    colors_L4 = []
    colors_L5 = []
    dist = []
    scene_L1 = o3d.t.geometry.RaycastingScene()
    scene_L2 = o3d.t.geometry.RaycastingScene()
    scene_L3 = o3d.t.geometry.RaycastingScene()
    scene_L4 = o3d.t.geometry.RaycastingScene()
    scene_L5 = o3d.t.geometry.RaycastingScene()

    scene_L1.add_triangles(pcd_L1)
    scene_L2.add_triangles(pcd_L2)
    scene_L3.add_triangles(pcd_L3)
    scene_L4.add_triangles(pcd_L4)
    scene_L5.add_triangles(pcd_L5)

    for el, each_point in enumerate(pcd_array):
        query_point = o3d.core.Tensor([each_point], dtype=o3d.core.Dtype.Float32)
        ans = scene_L1.compute_closest_points(query_point)
        ans = scene_L2.compute_closest_points(query_point)
        ans = scene_L3.compute_closest_points(query_point)
        ans = scene_L4.compute_closest_points(query_point)
        ans = scene_L5.compute_closest_points(query_point)

        signed_distance_L1 = scene_L1.compute_signed_distance(query_point).numpy()[0]
        signed_distance_L2 = scene_L2.compute_signed_distance(query_point).numpy()[0]
        signed_distance_L3 = scene_L3.compute_signed_distance(query_point).numpy()[0]
        signed_distance_L4 = scene_L4.compute_signed_distance(query_point).numpy()[0]
        signed_distance_L5 = scene_L5.compute_signed_distance(query_point).numpy()[0]

        signed_distances = []
        signed_distances.append(signed_distance_L1)
        signed_distances.append(signed_distance_L2)
        signed_distances.append(signed_distance_L3)
        signed_distances.append(signed_distance_L4)
        signed_distances.append(signed_distance_L5)
        min_signed_dist = np.min(signed_distances)
        thresh = 10
        if signed_distance_L1 < thresh:
            if signed_distance_L1 == min_signed_dist:
                close_to_L1.append(each_point)
                colors_L1.append(pcd_colors[el])
        if signed_distance_L2 < thresh:
            if signed_distance_L2 == min_signed_dist:
                close_to_L2.append(each_point)
                colors_L2.append(pcd_colors[el])
        if signed_distance_L3 < thresh:
            if signed_distance_L3 == min_signed_dist:
                close_to_L3.append(each_point)
                colors_L3.append(pcd_colors[el])
        if signed_distance_L4 < thresh:
            if signed_distance_L4 == min_signed_dist:
                close_to_L4.append(each_point)
                colors_L4.append(pcd_colors[el])
        if signed_distance_L5 < thresh:
            if signed_distance_L5 == min_signed_dist:
                close_to_L5.append(each_point)
                colors_L5.append(pcd_colors[el])

    pcd_L1 = o3d.geometry.PointCloud()
    pcd_L2 = o3d.geometry.PointCloud()
    pcd_L3 = o3d.geometry.PointCloud()
    pcd_L4 = o3d.geometry.PointCloud()
    pcd_L5 = o3d.geometry.PointCloud()

    pcd_L1.points = o3d.utility.Vector3dVector(close_to_L1)
    pcd_L2.points = o3d.utility.Vector3dVector(close_to_L2)
    pcd_L3.points = o3d.utility.Vector3dVector(close_to_L3)
    pcd_L4.points = o3d.utility.Vector3dVector(close_to_L4)
    pcd_L5.points = o3d.utility.Vector3dVector(close_to_L5)

    pcd_L1.colors = o3d.utility.Vector3dVector(colors_L1)
    pcd_L2.colors = o3d.utility.Vector3dVector(colors_L2)
    pcd_L3.colors = o3d.utility.Vector3dVector(colors_L3)
    pcd_L4.colors = o3d.utility.Vector3dVector(colors_L4)
    pcd_L5.colors = o3d.utility.Vector3dVector(colors_L5)


    return pcd_L1, pcd_L2, pcd_L3, pcd_L4, pcd_L5

