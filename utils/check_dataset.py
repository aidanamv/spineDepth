import numpy as np
import open3d as o3d
import os

dir ="/media/aidana/US study/PCN/train/partial/10102023"

files= os.listdir(dir)
for file in files:
    print(file)
    pcd = o3d.io.read_point_cloud(os.path.join(dir, file, "00.pcd"))
    print(np.asarray(pcd.points).shape[0])
