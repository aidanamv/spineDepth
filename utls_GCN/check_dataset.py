import pyvista as pv
import os
import numpy as np
dir = "/Users/aidanamassalimova/Documents/planning data original/final dataset/train"

heatmaps = np.load(os.path.join(dir, "heatmaps.npz"))["arr_0"]
dataset =np.load(os.path.join(dir, "vertices.npz"))["arr_0"]
for el,data in enumerate(dataset):
    pcd = pv.PolyData(data)
    pcd["Colors"] = heatmaps[el,:,:]

    p = pv.Plotter()
    p.add_mesh(pcd)
    p.show()




