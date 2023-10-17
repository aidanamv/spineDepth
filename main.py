import copy
import numpy as np
import cv2
import open3d as o3d
from utils.frame_extractor import frame_extract
from utils.crop_pcds import crop_pcd
from utils.visualize_groundtruth import select_target
from utils.registration import register
from utils.segmentation import segmentation
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.spatial.transform import Rotation as R


specimen = "Specimen_9"
recording = "Recording20"
path = "J:/Aidana_data/"
#%%QQ
#lets read SVO file first
camera_1 = path  + specimen +"/Recordings/" + recording +"/Video_0.svo"
camera_2 = path  + specimen +"/Recordings/" + recording +"/Video_1.svo"
dir_path_1 =  path + specimen + "/Recordings/" + recording +"/Video_0/"
dir_path_2 =  path + specimen + "/Recordings/" + recording +"/Video_1/"
# path for images
pc_dir_1 = os.path.join(dir_path_1, "frame_{}/pointcloud".format(0))
pc_dir_2 = os.path.join(dir_path_2, "frame_{}/pointcloud".format(0))
#frame_extract(camera_1, camera_2,pc_dir_1,pc_dir_2)
pcd_cropped_1, pcd_cropped_2 = crop_pcd(pc_dir_1, pc_dir_2)
source, target,source_pcd, target_pcd,vertebra1, vertebra2,vertebra3,vertebra4,vertebra5,num = select_target(path,specimen, recording, pcd_cropped_1, pcd_cropped_2)
vertebra1_bbox = vertebra1.get_oriented_bounding_box()
vertebra2_bbox = vertebra2.get_oriented_bounding_box()
vertebra3_bbox = vertebra3.get_oriented_bounding_box()
vertebra4_bbox = vertebra4.get_oriented_bounding_box()
vertebra5_bbox = vertebra5.get_oriented_bounding_box()

vertebra1_bbox.color = (1, 0, 0)
vertebra2_bbox.color = (1, 0, 0)
vertebra3_bbox.color = (1, 0, 0)
vertebra4_bbox.color = (1, 0, 0)
vertebra5_bbox.color = (1, 0, 0)
spine = vertebra1+vertebra2+vertebra3+vertebra4+vertebra5

spine_bbox = spine.get_oriented_bounding_box()
spine_bbox.color = (0,0,1)
o3d.visualization.draw_geometries([spine_bbox,pcd_cropped_1,vertebra1_bbox,vertebra2_bbox, vertebra3_bbox, vertebra4_bbox, vertebra5_bbox, vertebra1,vertebra2, vertebra3, vertebra4, vertebra5])

source = source.sample_points_uniformly(100000)
target = target.sample_points_uniformly(100000)
source_pcd = register(source,target,source_pcd, target_pcd)
#pcd_L1_1, pcd_L2_1,pcd_L3_1,pcd_L4_1,pcd_L5_1 = segmentation(source_pcd,vertebra1,vertebra2,vertebra3,vertebra4,vertebra5)
#pcd_L1_2, pcd_L2_2,pcd_L3_2,pcd_L4_2,pcd_L5_2 = segmentation(target_pcd,vertebra1,vertebra2,vertebra3,vertebra4,vertebra5)
threshold = 1.5
current_transformation = np.identity(4)
reg_L1 = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, threshold, current_transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
source_pcd.transform(reg_L1.transformation)
o3d.visualization.draw_geometries([source_pcd, target_pcd])
pcd_combined = source_pcd +target_pcd
o3d.io.write_point_cloud(path + specimen+"/Recordings/" + recording +"/"+specimen+"_"+ recording+"_fused.ply", pcd_combined)

reg_L2 = o3d.pipelines.registration.registration_icp(
    pcd_L2_1, pcd_L2_2, threshold, current_transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
pcd_L2_1.transform(reg_L2.transformation)
o3d.visualization.draw_geometries([pcd_L2_1, pcd_L2_2])
reg_L3 = o3d.pipelines.registration.registration_icp(
    pcd_L3_1, pcd_L3_2, threshold, current_transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
pcd_L3_1.transform(reg_L3.transformation)
o3d.visualization.draw_geometries([pcd_L3_1, pcd_L3_2])
reg_L4 = o3d.pipelines.registration.registration_icp(
    pcd_L4_1, pcd_L4_2, threshold, current_transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
pcd_L4_1.transform(reg_L4.transformation)
o3d.visualization.draw_geometries([pcd_L4_1, pcd_L4_2])
reg_L5 = o3d.pipelines.registration.registration_icp(
    pcd_L5_1, pcd_L5_2, threshold, current_transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
pcd_L5_1.transform(reg_L5.transformation)
o3d.visualization.draw_geometries([pcd_L5_1, pcd_L5_2])

pcd_L1 =pcd_L1_1+pcd_L1_2
pcd_L2 =pcd_L2_1+pcd_L2_2
pcd_L3 =pcd_L3_1+pcd_L3_2
pcd_L4 =pcd_L4_1+pcd_L4_2
pcd_L5 =pcd_L5_1+pcd_L5_2

pcd_combined = pcd_L1+pcd_L2+pcd_L3+pcd_L4+pcd_L5

os.makedirs(path + specimen+"/Recordings/" + recording +"/processing/rgbd/segmented/", exist_ok=True)
os.makedirs(path + specimen+"/Recordings/" + recording +"/processing/rgbd/fused/", exist_ok=True)
os.makedirs(path + specimen+"/Recordings/" + recording +"/processing/registered_vertebrae/", exist_ok=True)

o3d.io.write_point_cloud(path + specimen+"/Recordings/" + recording +"/processing/rgbd/segmented/"+specimen+"_"+ recording+"_L1.ply", pcd_L1)
o3d.io.write_point_cloud(path + specimen+"/Recordings/" + recording +"/processing/rgbd/segmented/"+specimen+"_"+ recording+"_L2.ply", pcd_L2)
o3d.io.write_point_cloud(path + specimen+"/Recordings/" + recording +"/processing/rgbd/segmented/"+specimen+"_"+ recording+"_L3.ply", pcd_L3)
o3d.io.write_point_cloud(path + specimen+"/Recordings/" + recording +"/processing/rgbd/segmented/"+specimen+"_"+ recording+"_L4.ply", pcd_L4)
o3d.io.write_point_cloud(path + specimen+"/Recordings/" + recording +"/processing/rgbd/segmented/"+specimen+"_"+ recording+"_L5.ply", pcd_L5)

o3d.io.write_point_cloud(path + specimen+"/Recordings/" + recording +"/processing/rgbd/fused/"+specimen+"_"+ recording+"_fused.ply", pcd_combined)

o3d.io.write_triangle_mesh(path + specimen+"/Recordings/" + recording +"/processing/registered_vertebrae/L1_registered2video_"+num+".stl", vertebra1)
o3d.io.write_triangle_mesh(path + specimen+"/Recordings/" + recording +"/processing/registered_vertebrae/L2_registered2video_"+num+".stl", vertebra2)
o3d.io.write_triangle_mesh(path + specimen+"/Recordings/" + recording +"/processing/registered_vertebrae/L3_registered2video_"+num+".stl", vertebra3)
o3d.io.write_triangle_mesh(path + specimen+"/Recordings/" + recording +"/processing/registered_vertebrae/L4_registered2video_"+num+".stl", vertebra4)
o3d.io.write_triangle_mesh(path + specimen+"/Recordings/" + recording +"/processing/registered_vertebrae/L5_registered2video_"+num+".stl", vertebra5)

o3d.visualization.draw_geometries([vertebra1, vertebra2, vertebra3, vertebra4, vertebra5, pcd_L1, pcd_L2, pcd_L3, pcd_L4,pcd_L5])



#%% 3D surface reconstruction
#pcd_L1 = o3d.io.read_point_cloud("F:\\Specimen_5\\Recordings\\Recording0\\processing\\rgbd\\segmented\\Specimen_5_Recording0_L1.ply")
#pcd_L2 = o3d.io.read_point_cloud("F:\\Specimen_5\\Recordings\\Recording0\\processing\\rgbd\\segmented\\Specimen_5_Recording0_L2.ply")
#pcd_L3 = o3d.io.read_point_cloud("F:\\Specimen_5\\Recordings\\Recording0\\processing\\rgbd\\segmented\\Specimen_5_Recording0_L3.ply")
#pcd_L4 = o3d.io.read_point_cloud("F:\\Specimen_5\\Recordings\\Recording0\\processing\\rgbd\\segmented\\Specimen_5_Recording0_L4.ply")
#pcd_L5 = o3d.io.read_point_cloud("F:\\Specimen_5\\Recordings\\Recording0\\processing\\rgbd\\segmented\\Specimen_5_Recording0_L5.ply")

downpcd_L1 = pcd_L1.uniform_down_sample(every_k_points=5)
downpcd_L2 = pcd_L2.uniform_down_sample(every_k_points=5)
downpcd_L3 = pcd_L3.uniform_down_sample(every_k_points=5)
downpcd_L4 = pcd_L4.uniform_down_sample(every_k_points=5)
downpcd_L5 = pcd_L5.uniform_down_sample(every_k_points=5)

downpcd_L1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
downpcd_L2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
downpcd_L3.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
downpcd_L4.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
downpcd_L5.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

downpcd_L1.orient_normals_towards_camera_location()
downpcd_L2.orient_normals_towards_camera_location()
downpcd_L3.orient_normals_towards_camera_location()
downpcd_L4.orient_normals_towards_camera_location()
downpcd_L5.orient_normals_towards_camera_location()

#o3d.visualization.draw_geometries([downpcd_L1], point_show_normal=True)
#%% 3D surface reconstruction
distances_L1 = downpcd_L1.compute_nearest_neighbor_distance()
distances_L2 = downpcd_L2.compute_nearest_neighbor_distance()
distances_L3 = downpcd_L3.compute_nearest_neighbor_distance()
distances_L4 = downpcd_L4.compute_nearest_neighbor_distance()
distances_L5 = downpcd_L5.compute_nearest_neighbor_distance()

avg_dist_L1 = np.mean(distances_L1)
avg_dist_L2 = np.mean(distances_L2)
avg_dist_L3 = np.mean(distances_L3)
avg_dist_L4 = np.mean(distances_L4)
avg_dist_L5 = np.mean(distances_L5)

factor = 5
radius_L1 = factor * avg_dist_L1
radius_L2 = factor * avg_dist_L2
radius_L3 = factor * avg_dist_L3
radius_L4 = factor * avg_dist_L4
radius_L5 = factor * avg_dist_L5

radi_L1 = o3d.utility.DoubleVector([radius_L1, radius_L1 * 2])
radi_L2 = o3d.utility.DoubleVector([radius_L2, radius_L2 * 2])
radi_L3 = o3d.utility.DoubleVector([radius_L3, radius_L3 * 2])
radi_L4 = o3d.utility.DoubleVector([radius_L4, radius_L4 * 2])
radi_L5 = o3d.utility.DoubleVector([radius_L5, radius_L5 * 2])

mesh_L1 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(downpcd_L1, radi_L1)
mesh_L2 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(downpcd_L2, radi_L2)
mesh_L3 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(downpcd_L3, radi_L3)
mesh_L4 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(downpcd_L4, radi_L4)
mesh_L5 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(downpcd_L5, radi_L5)

mesh_L1 = o3d.t.geometry.TriangleMesh.from_legacy(mesh_L1)
mesh_L2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh_L2)
mesh_L3 = o3d.t.geometry.TriangleMesh.from_legacy(mesh_L3)
mesh_L4 = o3d.t.geometry.TriangleMesh.from_legacy(mesh_L4)
mesh_L5 = o3d.t.geometry.TriangleMesh.from_legacy(mesh_L5)

filled_L1 = mesh_L1.fill_holes()
filled_L2 = mesh_L2.fill_holes()
filled_L3 = mesh_L3.fill_holes()
filled_L4 = mesh_L4.fill_holes()
filled_L5 = mesh_L5.fill_holes()

filled_mesh_L1=filled_L1.to_legacy()
filled_mesh_L2=filled_L2.to_legacy()
filled_mesh_L3=filled_L3.to_legacy()
filled_mesh_L4=filled_L4.to_legacy()
filled_mesh_L5=filled_L5.to_legacy()

filled_mesh_L1.compute_triangle_normals()
filled_mesh_L2.compute_triangle_normals()
filled_mesh_L3.compute_triangle_normals()
filled_mesh_L4.compute_triangle_normals()
filled_mesh_L5.compute_triangle_normals()

o3d.io.write_triangle_mesh(path + specimen+"/Recordings/" + recording +"/processing/rgbd/reconstructed_meshes/L1_reconstructed.stl", filled_mesh_L1)
o3d.io.write_triangle_mesh(path + specimen+"/Recordings/" + recording +"/processing/rgbd/reconstructed_meshes/L2_reconstructed.stl", filled_mesh_L2)
o3d.io.write_triangle_mesh(path + specimen+"/Recordings/" + recording +"/processing/rgbd/reconstructed_meshes/L3_reconstructed.stl", filled_mesh_L3)
o3d.io.write_triangle_mesh(path + specimen+"/Recordings/" + recording +"/processing/rgbd/reconstructed_meshes/L4_reconstructed.stl", filled_mesh_L4)
o3d.io.write_triangle_mesh(path + specimen+"/Recordings/" + recording +"/processing/rgbd/reconstructed_meshes/L5_reconstructed.stl", filled_mesh_L5)
