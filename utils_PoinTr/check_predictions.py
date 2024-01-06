import numpy as np
import open3d as o3d
import os
import random


dir ="/Users/aidanamassalimova/Documents/FinalDataset/fold_1/train"

files= os.listdir(os.path.join(dir,"predictions"))
vis=[]
for el,file in enumerate(files):
    print(file)
    file = file[:-4]

    pcd_partial = o3d.io.read_point_cloud(os.path.join(dir,"partial","10102023", file, "00.pcd"))
    pcd_complete = o3d.io.read_point_cloud(os.path.join(dir,"complete","10102023", file+".pcd"))
    landmarks=np.load(os.path.join(dir, "planning", "10102023",file+".npz"))["arr_0"]
    predictions = np.load(os.path.join(dir, "predictions", file+".npz"))["arr_0"]

    EP_left = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    EP_right = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    MP_left = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    MP_right = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)

    EP_left.paint_uniform_color((1,0,0))
    EP_right.paint_uniform_color((1,0,0))
    MP_left.paint_uniform_color((1,0,0))
    MP_right.paint_uniform_color((1,0,0))


    EP_left.translate(landmarks[0], relative=False)
    MP_left.translate(landmarks[1], relative=False)
    EP_right.translate(landmarks[2], relative=False)
    MP_right.translate(landmarks[3], relative=False)

    pcd_prediction = o3d.geometry.PointCloud()
    pcd_prediction.points = o3d.utility.Vector3dVector(np.asarray(predictions.squeeze(0)))

    pcd_prediction.paint_uniform_color((0,0,1))
    pcd_complete.paint_uniform_color((0,1,0))
    pcd_partial.paint_uniform_color((1,0,0))







   # o3d.visualization.draw_geometries([pcd_partial, pcd_prediction,pcd_complete])

    o3d.visualization.draw_geometries([pcd_prediction, EP_right, MP_left, MP_right])





