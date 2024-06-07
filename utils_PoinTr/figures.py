import os
import pyvista as pv
import open3d as o3d

dir = "/Volumes/Extreme SSD/PoinTr dataset/segmented_spinedepth_new/Specimen_2/recording_1/cam_1/frame_3"

ply1 = pv.PolyData(pv.read(os.path.join(dir,"pointcloud_vert1.ply")))
ply2 = pv.PolyData(pv.read(os.path.join(dir,"pointcloud_vert2.ply")))
ply3 = pv.PolyData(pv.read(os.path.join(dir,"pointcloud_vert3.ply")))
ply4 = pv.PolyData(pv.read(os.path.join(dir,"pointcloud_vert4.ply")))
ply5 = pv.PolyData(pv.read(os.path.join(dir,"pointcloud_vert5.ply")))




turqouise = "#24B0BA"
navy_blue = "#2E4A70"
gold ="#CF8A40"
pastel_red = "#ffb6c1"
colors = [

    (64 / 255, 224 / 255, 208 / 255),  # Green-Blue (Turquoise)
    (127 / 255, 255 / 255, 212 / 255),  # Blue-Green (Aquamarine)
    (210 / 255, 105 / 255, 30 / 255),  # Chocolate
    (101 / 255, 67 / 255, 33 / 255)  # Dark Brown
]


p = pv.Plotter()
p.add_mesh(ply1, color=(207,185,151), point_size=1)
p.add_mesh(ply2, color=colors[0], point_size=1)
p.add_mesh(ply3, color=colors[1], point_size=1)
p.add_mesh(ply4, color=(0,0,255), point_size=1)
p.add_mesh(ply5, color=colors[3], point_size=1)
p.show()
