import copy
import numpy as np
import cv2
import open3d as o3d
from utils.frame_extractor import frame_extract
from utils.crop_pcds import crop_pcd
from utils.visualize_groundtruth import select_target, change_background_to_black
from utils.registration import register
from utils.segmentation import segmentation
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.spatial.transform import Rotation as R
import distutils
from sklearn.decomposition import PCA




specimen = "Specimen_10"
recording = "Recording0"
path = "J:/Aidana_data/"

# path to source directory
src_dir = path  + specimen +"/Calib"

# path to destination directory
dest_dir = 'C:\ProgramData\Stereolabs\settings'

# getting all the files in the source directory
files = os.listdir(src_dir)

distutils.dir_util.copy_tree(src_dir, dest_dir)
#%%QQ
#lets read SVO file first
camera_1 = path  + specimen +"/Recordings/" + recording +"/Video_0.svo"
camera_2 = path  + specimen +"/Recordings/" + recording +"/Video_1.svo"
dir_path_1 =  path + specimen + "/Recordings/" + recording +"/Video_0/"
dir_path_2 =  path + specimen + "/Recordings/" + recording +"/Video_1/"
# path for images
pc_dir_1 = os.path.join(dir_path_1, "frame_{}/pointcloud".format(0))
pc_dir_2 = os.path.join(dir_path_2, "frame_{}/pointcloud".format(0))
frame_extract(camera_1, camera_2,pc_dir_1,pc_dir_2)
pcd_path_1 = os.path.join(pc_dir_1, 'pointcloud.ply')
pcd_path_2 = os.path.join(pc_dir_2, 'pointcloud.ply')

pcd_1 = o3d.io.read_point_cloud(pcd_path_1)
pcd_2 = o3d.io.read_point_cloud(pcd_path_2)
o3d.visualization.draw_geometries([pcd_2,pcd_1])
source, target,source_pcd, target_pcd,oriented_bounding_box_v1, oriented_bounding_box_v2,\
oriented_bounding_box_v3,oriented_bounding_box_v4,oriented_bounding_box_v5, \
vertebra1, vertebra2, vertebra3, vertebra4, vertebra5,\
cylinder_L1_left, cylinder_L1_right, cylinder_L2_left, cylinder_L2_right,\
cylinder_L3_left, cylinder_L3_right, cylinder_L4_left, cylinder_L4_right,cylinder_L5_left, \
cylinder_L5_right    = select_target(path, specimen, recording, pcd_1, pcd_2)




source = source.sample_points_uniformly(100000)
target = target.sample_points_uniformly(100000)
source_pcd = register(source,target,source_pcd, target_pcd)
#pcd_L1_1, pcd_L2_1,pcd_L3_1,pcd_L4_1,pcd_L5_1 = segmentation(source_pcd,vertebra1,vertebra2,vertebra3,vertebra4,vertebra5)
#pcd_L1_2, pcd_L2_2,pcd_L3_2,pcd_L4_2,pcd_L5_2 = segmentation(target_pcd,vertebra1,vertebra2,vertebra3,vertebra4,vertebra5)

pcd_combined = source_pcd +target_pcd


L1 = pcd_combined.crop(oriented_bounding_box_v1)
L2 = pcd_combined.crop(oriented_bounding_box_v2)
L3 = pcd_combined.crop(oriented_bounding_box_v3)
L4 = pcd_combined.crop(oriented_bounding_box_v4)
L5 = pcd_combined.crop(oriented_bounding_box_v5)
key_to_callback = {}
key_to_callback[ord("K")] = change_background_to_black
o3d.visualization.draw_geometries_with_key_callbacks([oriented_bounding_box_v1, oriented_bounding_box_v2,oriented_bounding_box_v3,oriented_bounding_box_v4,
                                   oriented_bounding_box_v5,pcd_combined,cylinder_L1_left, cylinder_L1_right, cylinder_L2_left,
                                   cylinder_L2_right,cylinder_L3_left, cylinder_L3_right, cylinder_L4_left, cylinder_L4_right,
                                   cylinder_L5_left, cylinder_L5_right],key_to_callback)

o3d.visualization.draw_geometries_with_key_callbacks([L1, vertebra1],key_to_callback)
o3d.visualization.draw_geometries_with_key_callbacks([L2, vertebra2],key_to_callback)
o3d.visualization.draw_geometries_with_key_callbacks([L3, vertebra3],key_to_callback)
o3d.visualization.draw_geometries_with_key_callbacks([L4, vertebra4],key_to_callback)
o3d.visualization.draw_geometries_with_key_callbacks([L5, vertebra5],key_to_callback)

os.makedirs(path + specimen+"/RGBD", exist_ok=True)
o3d.io.write_point_cloud(path + specimen+"/RGBD/"  + recording+"_fused.ply", pcd_combined)
o3d.io.write_point_cloud(path + specimen+"/RGBD/"+ recording+"_L1.ply",L1)
o3d.io.write_point_cloud(path + specimen+"/RGBD/"  +recording+"_L2.ply", L2)
o3d.io.write_point_cloud(path + specimen+"/RGBD/" + recording+"_L3.ply", L3)
o3d.io.write_point_cloud(path + specimen+"/RGBD/"  + recording+"_L4.ply", L4)
o3d.io.write_point_cloud(path + specimen+"/RGBD/"+ recording+"_L5.ply", L5)

pca = PCA(n_components=3)
vertices_L1 = np.asarray((vertebra1+vertebra2+vertebra3+vertebra4+vertebra5).vertices)
pca_L1 =pca.fit(vertices_L1)
T = np.eye(4)
#T[:3, :3]  =pca_L1.components_

origin =[0,0,0]
disp_L1 = origin - vertebra1.get_center()
disp_L2 = origin - vertebra2.get_center()
disp_L3 = origin - vertebra3.get_center()
disp_L4 = origin - vertebra4.get_center()
disp_L5 = origin - vertebra5.get_center()
o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+specimen+"_L1_registered.stl",  vertebra1)
o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L2_registered.stl", vertebra2)
o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L3_registered.stl", vertebra3)
o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L4_registered.stl", vertebra4)
o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L5_registered.stl", vertebra5)

#mesh_L1 = copy.deepcopy(vertebra1).transform(T).translate((0, 0, 0), relative=False)
#mesh_L2 = copy.deepcopy(vertebra2).transform(T).translate((0, 0, 0), relative=False)
#mesh_L3 = copy.deepcopy(vertebra3).transform(T).translate((0, 0, 0), relative=False)
#mesh_L4 = copy.deepcopy(vertebra4).transform(T).translate((0, 0, 0), relative=False)
#mesh_L5 = copy.deepcopy(vertebra5).transform(T).translate((0, 0, 0), relative=False)


#plan_L1_left = copy.deepcopy(cylinder_L1_left).transform(T).translate(disp_L1, relative=True)
#plan_L2_left = copy.deepcopy(cylinder_L2_left).transform(T).translate(disp_L2, relative=True)
#plan_L3_left = copy.deepcopy(cylinder_L3_left).transform(T).translate(disp_L3, relative=True)
#plan_L4_left = copy.deepcopy(cylinder_L4_left).transform(T).translate(disp_L4, relative=True)
#plan_L5_left = copy.deepcopy(cylinder_L5_left).transform(T).translate(disp_L5, relative=True)


#plan_L1_right = copy.deepcopy(cylinder_L1_right).transform(T).translate(disp_L1, relative=True)
#plan_L2_right = copy.deepcopy(cylinder_L2_right).transform(T).translate(disp_L2, relative=True)
#plan_L3_right = copy.deepcopy(cylinder_L3_right).transform(T).translate(disp_L3, relative=True)
#plan_L4_right = copy.deepcopy(cylinder_L4_right).transform(T).translate(disp_L4, relative=True)
#plan_L5_right = copy.deepcopy(cylinder_L5_right).transform(T).translate(disp_L5, relative=True)

#o3d.visualization.draw_geometries([plan_L1_left, plan_L1_right, mesh_L1, plan_L2_left, plan_L2_right, mesh_L2, plan_L3_left, plan_L3_right, mesh_L3,
#                                   plan_L3_left, plan_L3_right, mesh_L3, plan_L4_left, plan_L4_right, mesh_L4,
#                                  plan_L5_left, plan_L5_right, mesh_L5])

#o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+specimen+"_L1.stl",  mesh_L1)
#o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L2.stl", mesh_L2)
#o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L3.stl", mesh_L3)
#o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L4.stl", mesh_L4)
#o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L5.stl", mesh_L5)



#o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L1_planned.stl", plan_L1_left+plan_L1_right)
#o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L2_planned.stl", plan_L2_left+plan_L2_right)
#o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L3_planned.stl", plan_L3_left+plan_L3_right)
#o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L4_planned.stl", plan_L4_left + plan_L4_right)
#o3d.io.write_triangle_mesh(path + specimen+"/RGBD/"+ specimen+"_L5_planned.stl", plan_L5_left + plan_L5_right)


