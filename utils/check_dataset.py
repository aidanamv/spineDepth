import numpy as np
import open3d as o3d
import os
import random
dir ="/media/aidana/Extreme SSD/aligned_new_new/partial"

files= os.listdir(dir)
random.shuffle(files)
vis=[]
for el,file in enumerate(files):
    print(file)
    pcd_partial = o3d.io.read_point_cloud(os.path.join(dir, file, "00.pcd"))
    pcd_complete = o3d.io.read_point_cloud(os.path.join("/home/aidana/Documents/PoinTr dataset/PCN_new_new/fold_0/train/complete/10102023", file+".pcd"))
    # landmarks=np.load(os.path.join("/home/aidana/Documents/PoinTr dataset/PCN_new/fold_0/train/planning/10102023", file[:-2]+"plannings.npz"))["arr_0"]
    # level = file[-2:]
    # EP_left = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    # EP_right = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    # MP_left = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    # MP_right = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    # EP_left.paint_uniform_color((1,0,0))
    # EP_right.paint_uniform_color((1,0,0))
    # MP_left.paint_uniform_color((1,0,0))
    # MP_right.paint_uniform_color((1,0,0))
    #
    # print(level)
    # if level=="L1":
    #     EP_left.translate(landmarks[0], relative=False)
    #     MP_left.translate(landmarks[1], relative=False)
    #     EP_right.translate(landmarks[2], relative=False)
    #     MP_right.translate(landmarks[3], relative=False)
    # if level=="L2":
    #     EP_left.translate(landmarks[4], relative=False)
    #     MP_left.translate(landmarks[5], relative=False)
    #     EP_right.translate(landmarks[6], relative=False)
    #     MP_right.translate(landmarks[7], relative=False)
    # if level=="L3":
    #     EP_left.translate(landmarks[8], relative=False)
    #     MP_left.translate(landmarks[9], relative=False)
    #     EP_right.translate(landmarks[10], relative=False)
    #     MP_right.translate(landmarks[11], relative=False)
    # if level=="L4":
    #     EP_left.translate(landmarks[12], relative=False)
    #     MP_left.translate(landmarks[13], relative=False)
    #     EP_right.translate(landmarks[14], relative=False)
    #     MP_right.translate(landmarks[15], relative=False)
    # if level=="L5":
    #     EP_left.translate(landmarks[16], relative=False)
    #     MP_left.translate(landmarks[17], relative=False)
    #     EP_right.translate(landmarks[18], relative=False)
    #     MP_right.translate(landmarks[19], relative=False)
    #vis.append(pcd_complete)
    vis.append(pcd_partial)

o3d.visualization.draw_geometries(vis)

