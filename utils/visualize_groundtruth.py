import pandas as pd
import open3d as o3d
import numpy as np
import copy
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import os


def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False
def estimate_rot (point1, point2, height):
    # Find the angle between the two vectors
    direction =  (point2 - point1)/np.linalg.norm(point2-point1)
    z_unit_vector = np.array([0, 0, 1])
    axis = np.cross(z_unit_vector, direction)
    midpoint = point1 - 30 *direction
    angle = np.arccos(np.dot(z_unit_vector, direction))

    # Create the rotation matrix using the Rodrigues' rotation formula
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis

    rotation_matrix = np.array([
        [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
        [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
        [t * x * z - y * s, t * y * z + x * s, t * z * z + c]
    ])
    return rotation_matrix, midpoint

def select_target(path,specimen, recording, pcd_cropped_1, pcd_cropped_2):
    # reading tracking data
    tracking_file_1 = path + specimen + "/Recordings/" + recording + "/Poses_0.txt"
    tracking_file_2 = path + specimen + "/Recordings/" + recording + "/Poses_1.txt"

    df=pd.read_csv(tracking_file_1,sep = ',',header=None, names=["R00", "R01", "R02","T0","R10","R11", "R12","T1","R20","R21","R22","T2","R30","R31","R33","T3"])
    df_1=pd.read_csv(tracking_file_2,sep = ',',header=None, names=["R00", "R01", "R02","T0","R10","R11", "R12","T1","R20","R21","R22","T2","R30","R31","R33","T3"])

    planningFile = open(path+specimen+"/Planning/PlanningPoints.txt", 'r')
    contents = planningFile.readlines()
    point1_L1_left = np.fromstring(contents[0], dtype=float, sep=',')
    point2_L1_left = np.fromstring(contents[1], dtype=float, sep=',')
    point1_L1_right = np.fromstring(contents[2], dtype=float, sep=',')
    point2_L1_right = np.fromstring(contents[3], dtype=float, sep=',')

    point1_L2_left = np.fromstring(contents[4], dtype=float, sep=',')
    point2_L2_left = np.fromstring(contents[5], dtype=float, sep=',')
    point1_L2_right = np.fromstring(contents[6], dtype=float, sep=',')
    point2_L2_right = np.fromstring(contents[7], dtype=float, sep=',')

    point1_L3_left = np.fromstring(contents[8], dtype=float, sep=',')
    point2_L3_left = np.fromstring(contents[9], dtype=float, sep=',')
    point1_L3_right = np.fromstring(contents[10], dtype=float, sep=',')
    point2_L3_right = np.fromstring(contents[11], dtype=float, sep=',')

    point1_L4_left = np.fromstring(contents[12], dtype=float, sep=',')
    point2_L4_left = np.fromstring(contents[13], dtype=float, sep=',')
    point1_L4_right = np.fromstring(contents[14], dtype=float, sep=',')
    point2_L4_right = np.fromstring(contents[15], dtype=float, sep=',')

    point1_L5_left = np.fromstring(contents[16], dtype=float, sep=',')
    point2_L5_left = np.fromstring(contents[17], dtype=float, sep=',')
    point1_L5_right = np.fromstring(contents[18], dtype=float, sep=',')
    point2_L5_right = np.fromstring(contents[19], dtype=float, sep=',')

    # reading corresponding stl files
    vertebra1_cam1 = o3d.io.read_triangle_mesh(path + specimen + "/STL/L1.stl")
    vertebra2_cam1 = o3d.io.read_triangle_mesh(path + specimen + "/STL/L2.stl")
    vertebra3_cam1 = o3d.io.read_triangle_mesh(path + specimen + "/STL/L3.stl")
    vertebra4_cam1 = o3d.io.read_triangle_mesh(path + specimen + "/STL/L4.stl")
    vertebra5_cam1 = o3d.io.read_triangle_mesh(path + specimen + "/STL/L5.stl")


    vertebra1_cam2 = copy.deepcopy(vertebra1_cam1)
    vertebra2_cam2 = copy.deepcopy(vertebra2_cam1)
    vertebra3_cam2 = copy.deepcopy(vertebra3_cam1)
    vertebra4_cam2 = copy.deepcopy(vertebra4_cam1)
    vertebra5_cam2 = copy.deepcopy(vertebra5_cam1)

    vertebra1_cam1 = vertebra1_cam1.compute_vertex_normals()
    vertebra2_cam1 = vertebra2_cam1.compute_vertex_normals()
    vertebra3_cam1 = vertebra3_cam1.compute_vertex_normals()
    vertebra4_cam1 = vertebra4_cam1.compute_vertex_normals()
    vertebra5_cam1 = vertebra5_cam1.compute_vertex_normals()

    vertebra1_cam2 = vertebra1_cam2.compute_vertex_normals()
    vertebra2_cam2 = vertebra2_cam2.compute_vertex_normals()
    vertebra3_cam2 = vertebra3_cam2.compute_vertex_normals()
    vertebra4_cam2 = vertebra4_cam2.compute_vertex_normals()
    vertebra5_cam2 = vertebra5_cam2.compute_vertex_normals()
    i=0
    vertebra1_pose_cam1 = np.array([[df.iloc[i]["R00"],df.iloc[i]["R01"],df.iloc[i]["R02"],df.iloc[i]["T0"]],
                               [df.iloc[i]["R10"],df.iloc[i]["R11"],df.iloc[i]["R12"],df.iloc[i]["T1"]],
                               [df.iloc[i]["R20"],df.iloc[i]["R21"],df.iloc[i]["R22"],df.iloc[i]["T2"]],
                               [0,0,0,1]])


    vertebra1_pose_cam2 = np.array([[df_1.iloc[i]["R00"], df_1.iloc[i]["R01"], df_1.iloc[i]["R02"], df_1.iloc[i]["T0"]],
                                    [df_1.iloc[i]["R10"], df_1.iloc[i]["R11"], df_1.iloc[i]["R12"], df_1.iloc[i]["T1"]],
                                    [df_1.iloc[i]["R20"], df_1.iloc[i]["R21"], df_1.iloc[i]["R22"], df_1.iloc[i]["T2"]],
                                    [0, 0, 0, 1]])


    vertebra2_pose_cam1 = np.array([[ df.iloc[i+1]["R00"],df.iloc[i+1]["R01"],df.iloc[i+1]["R02"],df.iloc[i+1]["T0"]],
                               [df.iloc[i+1]["R10"],df.iloc[i+1]["R11"],df.iloc[i+1]["R12"],df.iloc[i+1]["T1"]],
                               [df.iloc[i+1]["R20"],df.iloc[i+1]["R21"],df.iloc[i+1]["R22"],df.iloc[i+1]["T2"]],
                               [0,0,0,1]])


    vertebra2_pose_cam2 = np.array(
        [[df_1.iloc[i + 1]["R00"], df_1.iloc[i + 1]["R01"], df_1.iloc[i + 1]["R02"], df_1.iloc[i + 1]["T0"]],
         [df_1.iloc[i + 1]["R10"], df_1.iloc[i + 1]["R11"], df_1.iloc[i + 1]["R12"], df_1.iloc[i + 1]["T1"]],
         [df_1.iloc[i + 1]["R20"], df_1.iloc[i + 1]["R21"], df_1.iloc[i + 1]["R22"], df_1.iloc[i + 1]["T2"]],
         [0, 0, 0, 1]])

    vertebra3_pose_cam1 =  np.array([[ df.iloc[i+2]["R00"],df.iloc[i+2]["R01"],df.iloc[i+2]["R02"],df.iloc[i+2]["T0"]],
                               [df.iloc[i+2]["R10"],df.iloc[i+2]["R11"],df.iloc[i+2]["R12"],df.iloc[i+2]["T1"]],
                               [df.iloc[i+2]["R20"],df.iloc[i+2]["R21"],df.iloc[i+2]["R22"],df.iloc[i+2]["T2"]],
                               [0,0,0,1]])
    vertebra3_pose_cam2 = np.array(
        [[df_1.iloc[i + 2]["R00"], df_1.iloc[i + 2]["R01"], df_1.iloc[i + 2]["R02"], df_1.iloc[i + 2]["T0"]],
         [df_1.iloc[i + 2]["R10"], df_1.iloc[i + 2]["R11"], df_1.iloc[i + 2]["R12"], df_1.iloc[i + 2]["T1"]],
         [df_1.iloc[i + 2]["R20"], df_1.iloc[i + 2]["R21"], df_1.iloc[i + 2]["R22"], df_1.iloc[i + 2]["T2"]],
         [0, 0, 0, 1]])

    vertebra4_pose_cam1 =  np.array([[ df.iloc[i+3]["R00"],df.iloc[i+3]["R01"],df.iloc[i+3]["R02"],df.iloc[i+3]["T0"]],
                               [df.iloc[i+3]["R10"],df.iloc[i+3]["R11"],df.iloc[i+3]["R12"],df.iloc[i+3]["T1"]],
                               [df.iloc[i+3]["R20"],df.iloc[i+3]["R21"],df.iloc[i+3]["R22"],df.iloc[i+3]["T2"]],
                               [0,0,0,1]])

    vertebra4_pose_cam2 =  np.array([[df_1.iloc[i+3]["R00"],df_1.iloc[i+3]["R01"],df_1.iloc[i+3]["R02"],df_1.iloc[i+3]["T0"]],
                               [df_1.iloc[i+3]["R10"],df_1.iloc[i+3]["R11"],df_1.iloc[i+3]["R12"],df_1.iloc[i+3]["T1"]],
                               [df_1.iloc[i+3]["R20"],df_1.iloc[i+3]["R21"],df_1.iloc[i+3]["R22"],df_1.iloc[i+3]["T2"]],
                               [0,0,0,1]])

    vertebra5_pose_cam1 = np.array([[ df.iloc[i+4]["R00"],df.iloc[i+4]["R01"],df.iloc[i+4]["R02"],df.iloc[i+4]["T0"]],
                               [df.iloc[i+4]["R10"],df.iloc[i+4]["R11"],df.iloc[i+4]["R12"],df.iloc[i+4]["T1"]],
                               [df.iloc[i+4]["R20"],df.iloc[i+4]["R21"],df.iloc[i+4]["R22"],df.iloc[i+4]["T2"]],
                               [0,0,0,1]])

    vertebra5_pose_cam2 = np.array(
        [[df_1.iloc[i + 4]["R00"], df_1.iloc[i + 4]["R01"], df_1.iloc[i + 4]["R02"], df_1.iloc[i + 4]["T0"]],
         [df_1.iloc[i + 4]["R10"], df_1.iloc[i + 4]["R11"], df_1.iloc[i + 4]["R12"], df_1.iloc[i + 4]["T1"]],
         [df_1.iloc[i + 4]["R20"], df_1.iloc[i + 4]["R21"], df_1.iloc[i + 4]["R22"], df_1.iloc[i + 4]["T2"]],
         [0, 0, 0, 1]])

    vertebra1_cam1.transform(vertebra1_pose_cam1)
    vertebra2_cam1.transform(vertebra2_pose_cam1)
    vertebra3_cam1.transform(vertebra3_pose_cam1)
    vertebra4_cam1.transform(vertebra4_pose_cam1)
    vertebra5_cam1.transform(vertebra5_pose_cam1)



    vertebra1_cam2.transform(vertebra1_pose_cam2)
    vertebra2_cam2.transform(vertebra2_pose_cam2)
    vertebra3_cam2.transform(vertebra3_pose_cam2)
    vertebra4_cam2.transform(vertebra4_pose_cam2)
    vertebra5_cam2.transform(vertebra5_pose_cam2)





    spine_cam1 = vertebra1_cam1 + vertebra2_cam1+ vertebra3_cam1 + vertebra4_cam1 + vertebra5_cam1
    spine_cam2 = vertebra1_cam2 + vertebra2_cam2 + vertebra3_cam2+ vertebra4_cam2 + vertebra5_cam2


    spine_cam1_bbox = spine_cam1.get_oriented_bounding_box()

    spine_cam2_bbox = spine_cam2.get_oriented_bounding_box()
    spine_cam1_bbox.color = (0, 0, 1)
    spine_cam2_bbox.color = (0, 0, 1)

    crop1 = pcd_cropped_1.crop(spine_cam1_bbox)
    crop2 = pcd_cropped_2.crop(spine_cam2_bbox)


    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black

    o3d.visualization.draw_geometries_with_key_callbacks([spine_cam1_bbox,crop1,vertebra1_cam1, vertebra2_cam1, vertebra3_cam1, vertebra4_cam1, vertebra5_cam1],key_to_callback)
    o3d.visualization.draw_geometries_with_key_callbacks([spine_cam2_bbox,crop2,vertebra1_cam2, vertebra2_cam2, vertebra3_cam2, vertebra4_cam2, vertebra5_cam2],key_to_callback)
    txt = input("Select the best view (1 or 2): ")
    if int(txt)==1:
        source = copy.deepcopy(vertebra1_cam2)
        target = copy.deepcopy(vertebra1_cam1)
        spine_bbox=spine_cam1_bbox
        source_pcd = crop2
        target_pcd = crop1
        vertebra1_pose = vertebra1_pose_cam1
        vertebra2_pose = vertebra2_pose_cam1
        vertebra3_pose = vertebra3_pose_cam1
        vertebra4_pose = vertebra4_pose_cam1
        vertebra5_pose = vertebra5_pose_cam1
        vertebra1 = vertebra1_cam1
        vertebra2 = vertebra2_cam1
        vertebra3 = vertebra3_cam1
        vertebra4 = vertebra4_cam1
        vertebra5 = vertebra5_cam1

    else:
        source = copy.deepcopy(vertebra1_cam1)
        target = copy.deepcopy(vertebra1_cam2)
        spine_bbox=spine_cam2_bbox
        source_pcd = crop1
        target_pcd = crop2
        vertebra1 = vertebra1_cam2
        vertebra2 = vertebra2_cam2
        vertebra3 = vertebra3_cam2
        vertebra4 = vertebra4_cam2
        vertebra5 = vertebra5_cam2
        vertebra1_pose = vertebra1_pose_cam2
        vertebra2_pose = vertebra2_pose_cam2
        vertebra3_pose = vertebra3_pose_cam2
        vertebra4_pose = vertebra4_pose_cam2
        vertebra5_pose = vertebra5_pose_cam2

    vertebra1_bbox = vertebra1.get_oriented_bounding_box(robust=True)
    vertebra2_bbox = vertebra2.get_oriented_bounding_box(robust=True)
    vertebra3_bbox = vertebra3.get_oriented_bounding_box(robust=True)
    vertebra4_bbox = vertebra4.get_oriented_bounding_box(robust=True)
    vertebra5_bbox = vertebra5.get_oriented_bounding_box(robust=True)


    # Create a cylinder using Open3D's create_cylinder_geometry function
    radius =1.5
    height=100
    cylinder_L1_left = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    cylinder_L1_right = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    cylinder_L2_left = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    cylinder_L2_right = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    cylinder_L3_left = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    cylinder_L3_right = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    cylinder_L4_left = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    cylinder_L4_right = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    cylinder_L5_left = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    cylinder_L5_right = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)

    cylinder_L1_left.compute_vertex_normals()
    cylinder_L1_right.compute_vertex_normals()
    cylinder_L2_left.compute_vertex_normals()
    cylinder_L2_right.compute_vertex_normals()
    cylinder_L3_left.compute_vertex_normals()
    cylinder_L3_right.compute_vertex_normals()
    cylinder_L4_left.compute_vertex_normals()
    cylinder_L4_right.compute_vertex_normals()
    cylinder_L5_left.compute_vertex_normals()
    cylinder_L5_right.compute_vertex_normals()

    cylinder_L1_left.paint_uniform_color([0.1, 0.9, 0.1])
    cylinder_L1_right.paint_uniform_color([0.1, 0.9, 0.1])
    cylinder_L2_left.paint_uniform_color([0.1, 0.9, 0.1])
    cylinder_L2_right.paint_uniform_color([0.1, 0.9, 0.1])
    cylinder_L3_left.paint_uniform_color([0.1, 0.9, 0.1])
    cylinder_L3_right.paint_uniform_color([0.1, 0.9, 0.1])
    cylinder_L4_left.paint_uniform_color([0.1, 0.9, 0.1])
    cylinder_L4_right.paint_uniform_color([0.1, 0.9, 0.1])
    cylinder_L5_left.paint_uniform_color([0.1, 0.9, 0.1])
    cylinder_L5_right.paint_uniform_color([0.1, 0.9, 0.1])

    rotation_mat_L1_left, midpoint_L1_left = estimate_rot(point1_L1_left, point2_L1_left, height)
    rotation_mat_L2_left, midpoint_L2_left = estimate_rot(point1_L2_left, point2_L2_left, height)
    rotation_mat_L3_left, midpoint_L3_left = estimate_rot(point1_L3_left, point2_L3_left, height)
    rotation_mat_L4_left, midpoint_L4_left = estimate_rot(point1_L4_left, point2_L4_left, height)
    rotation_mat_L5_left, midpoint_L5_left = estimate_rot(point1_L5_left, point2_L5_left, height)

    rotation_mat_L1_right, midpoint_L1_right= estimate_rot(point1_L1_right, point2_L1_right, height)
    rotation_mat_L2_right, midpoint_L2_right= estimate_rot(point1_L2_right, point2_L2_right, height)
    rotation_mat_L3_right, midpoint_L3_right= estimate_rot(point1_L3_right, point2_L3_right, height)
    rotation_mat_L4_right, midpoint_L4_right= estimate_rot(point1_L4_right, point2_L4_right, height)
    rotation_mat_L5_right, midpoint_L5_right= estimate_rot(point1_L5_right, point2_L5_right, height)

    # Rotate the cylinder to align with the direction vector
    cylinder_L1_left.rotate(rotation_mat_L1_left)
    cylinder_L1_right.rotate(rotation_mat_L1_right)
    cylinder_L2_left.rotate(rotation_mat_L2_left)
    cylinder_L2_right.rotate(rotation_mat_L2_right)
    cylinder_L3_left.rotate(rotation_mat_L3_left)
    cylinder_L3_right.rotate(rotation_mat_L3_right)
    cylinder_L4_left.rotate(rotation_mat_L4_left)
    cylinder_L4_right.rotate(rotation_mat_L4_right)
    cylinder_L5_left.rotate(rotation_mat_L5_left)
    cylinder_L5_right.rotate(rotation_mat_L5_right)

    # Translate the cylinder to the midpoint between the two points
    cylinder_L1_left.translate(midpoint_L1_left)
    cylinder_L1_right.translate(midpoint_L1_right)
    cylinder_L2_left.translate(midpoint_L2_left)
    cylinder_L2_right.translate(midpoint_L2_right)
    cylinder_L3_left.translate(midpoint_L3_left)
    cylinder_L3_right.translate(midpoint_L3_right)
    cylinder_L4_left.translate(midpoint_L4_left)
    cylinder_L4_right.translate(midpoint_L4_right)
    cylinder_L5_left.translate(midpoint_L5_left)
    cylinder_L5_right.translate(midpoint_L5_right)

    cylinder_L1_left.transform(vertebra1_pose)
    cylinder_L1_right.transform(vertebra1_pose)
    cylinder_L2_left.transform(vertebra2_pose)
    cylinder_L2_right.transform(vertebra2_pose)
    cylinder_L3_left.transform(vertebra3_pose)
    cylinder_L3_right.transform(vertebra3_pose)
    cylinder_L4_left.transform(vertebra4_pose)
    cylinder_L4_right.transform(vertebra4_pose)
    cylinder_L5_left.transform(vertebra5_pose)
    cylinder_L5_right.transform(vertebra5_pose)






    R_v1 = spine_bbox.R
    trans_v1 = np.array([[R_v1[0,0],R_v1[0,1],R_v1[0,2],spine_bbox.center[0]],
                [R_v1[1][0], R_v1[1][1], R_v1[1,2], spine_bbox.center[1]],
                [R_v1[2,0], R_v1[2,1], R_v1[2,2], spine_bbox.center[2]],
                [0,0,0,1]])
    print(trans_v1)
    rot_mat_2 = np.eye(4)
    rot_mat_2[0:3,0:3] = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    new_trans_v1 = np.dot(trans_v1, rot_mat_2)
    vertebra1_bbox.R = new_trans_v1[0:3, 0:3]
    vertebra1_bbox.extent =vertebra1_bbox.extent+[25,25,0]
    vertebra2_bbox.R = new_trans_v1[0:3, 0:3]
    vertebra2_bbox.extent =vertebra2_bbox.extent+[25,25,0]

    vertebra3_bbox.R = new_trans_v1[0:3, 0:3]
    vertebra3_bbox.extent =vertebra3_bbox.extent+[25,25,0]

    vertebra4_bbox.R = new_trans_v1[0:3, 0:3]
    vertebra4_bbox.extent =vertebra4_bbox.extent+[25,25,0]

    vertebra5_bbox.R = new_trans_v1[0:3, 0:3]
    vertebra5_bbox.extent =vertebra5_bbox.extent+[25,25,0]



    vertebra1_bbox.color = (0, 1, 0)
    vertebra2_bbox.color = (0, 1, 0)
    vertebra3_bbox.color = (0, 1, 0)
    vertebra4_bbox.color = (0, 1, 0)
    vertebra5_bbox.color = (0, 1, 0)




    o3d.visualization.draw_geometries_with_key_callbacks([cylinder_L1_left, cylinder_L1_right, cylinder_L2_left, cylinder_L2_right,
                                       cylinder_L3_left, cylinder_L3_right, cylinder_L4_left, cylinder_L4_right,
                                       cylinder_L5_left, cylinder_L5_right,
                                       vertebra1, vertebra2, vertebra3, vertebra4, vertebra5,
                                      target_pcd], key_to_callback)

#   o3d.visualization.draw_geometries_with_key_callbacks([spine_bbox,vertebra1_bbox, vertebra2_bbox, vertebra3_bbox,vertebra4_bbox, vertebra5_bbox,vertebra1, vertebra2, vertebra3, vertebra4,vertebra5],key_to_callback)
    return source, target,source_pcd, target_pcd, vertebra1_bbox, vertebra2_bbox, vertebra3_bbox, vertebra4_bbox, vertebra5_bbox, vertebra1, vertebra2, vertebra3, vertebra4, vertebra5, cylinder_L1_left, cylinder_L1_right, cylinder_L2_left, cylinder_L2_right,cylinder_L3_left, cylinder_L3_right, cylinder_L4_left, cylinder_L4_right,cylinder_L5_left, cylinder_L5_right




