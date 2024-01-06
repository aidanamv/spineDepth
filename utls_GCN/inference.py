import os
import pyvista as pv
import numpy as np
# Specify the path to your PLY file
ply_file_path = "/Users/aidanamassalimova/Documents/FinalDataset/fold_1/train/heatmaps_pred"

files = os.listdir(ply_file_path)

for file in files:
    print(file)

    # Load the PLY file into a PyVista PolyData object
    data = np.load(os.path.join("/Users/aidanamassalimova/Documents/FinalDataset/fold_1/train/predictions",file))
    colors = np.load(os.path.join(ply_file_path,file))
    cloud =pv.PolyData(data["arr_0"].squeeze(0))
    cloud['colors'] = colors["arr_0"]
    # Create a PyVista plotter
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, point_size=5, cmap="jet")

    # Display the plot
    plotter.show()
