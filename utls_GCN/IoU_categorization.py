import pyvista as pv
import os
import numpy as np
import open3d as o3d
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from functools import reduce
from sklearn.decomposition import PCA



def calculate_iou(point_cloud1, point_cloud2):
    # Calculate the bounding boxes for each point cloud
    min1, max1 = np.min(point_cloud1, axis=0), np.max(point_cloud1, axis=0)
    min2, max2 = np.min(point_cloud2, axis=0), np.max(point_cloud2, axis=0)

    # Calculate the intersection
    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)
    intersection_volume = np.maximum(0, intersection_max - intersection_min).prod()

    # Calculate the union
    union_volume = (max1 - min1).prod() + (max2 - min2).prod() - intersection_volume

    # Calculate IoU
    iou = intersection_volume / union_volume if union_volume > 0 else 0

    return iou




dir = "/Users/aidanamassalimova/Documents/FinalDataset_4096/fold_0/val"
files = os.listdir(dir+"/complete/10102023")
iou_list =[]
acceptables = []
unacceptables = []

for file in files:
    pcd_complete = o3d.io.read_point_cloud(os.path.join(dir, "complete","10102023", file[:-4]+".pcd"))
    pcd_partial = o3d.io.read_point_cloud(os.path.join(dir, "partial","10102023", file[:-4]+"/00.pcd"))
    pcd_complete_pv = pv.PolyData(np.asarray(pcd_complete.points))
    pcd_partial_pv = pv.PolyData(np.asarray(pcd_partial.points))




    clipped = pcd_complete_pv.clip('x', invert=True)
    iou = calculate_iou(np.asarray(clipped.points), np.asarray(pcd_partial.points))
    iou_list.append(iou)
    if iou>=0.7:
        acceptables.append(iou)

    else:
        unacceptables.append(iou)

print("general stats")
print(np.mean(iou_list))
print(np.std(iou_list))
print("unacceptable stats")
print(len(unacceptables))
print("acceptable stats")
print(len(acceptables))






