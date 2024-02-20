import os
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.manifold import MDS
from functools import reduce
from sklearn.neighbors import NearestNeighbors
def get_rigid(src, dst):
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    H = reduce(lambda s, p: s + np.outer(p[0], p[1]), zip(src - src_mean, dst - dst_mean), np.zeros((3,3)))
    u, s, v = np.linalg.svd(H)
    R = v.T.dot(u.T)
    T = - R.dot(src_mean) + dst_mean
    return np.hstack((R, T[:, np.newaxis]))

dir = "/Users/aidanamassalimova/Documents/MICCAI/PoinTr_based/2048/fold_0/val"

vertices = np.load(os.path.join(dir, "vertices.npz"))["arr_0"]
vertices_gt = np.load(os.path.join(dir, "vertices_gt.npz"))["arr_0"]
heatmaps = np.load(os.path.join(dir, "heatmaps.npz"))["arr_0"]
landmarks = np.load(os.path.join(dir, "landmarks.npz"))["arr_0"]
labels = np.load(os.path.join(dir, "labels.npz"))["arr_0"]
rgbds = np.load(os.path.join(dir, "rgbd.npz"))["arr_0"]
el =1
results = labels[el].split("_")
print(el)

landmark = landmarks[el]
pcd = pv.PolyData(vertices[el])
pcd_gt = pv.PolyData(vertices_gt[el])
pcd_rgbd = pv.PolyData(rgbds[el])
pcd.point_data["Predictions"] =heatmaps[el, :, 1]

red_points_mask1 = np.where(heatmaps[el, :, 0] >= 0.75 * np.max(heatmaps[el, :, 0]))
red_points_mask2 = np.where(heatmaps[el, :, 1] >= 0.75 * np.max(heatmaps[el, :, 1]))
red_points_mask3 = np.where(heatmaps[el, :, 2] >= 0.75 * np.max(heatmaps[el, :, 2]))
red_points_mask4 = np.where(heatmaps[el, :, 3] >=0.75 * np.max(heatmaps[el, :, 3]))

red_cloud1 = pcd.extract_points(red_points_mask1)
red_cloud2 = pcd.extract_points(red_points_mask2)
red_cloud3 = pcd.extract_points(red_points_mask3)
red_cloud4 = pcd.extract_points(red_points_mask4)

mp1 = np.mean(red_cloud2.points, axis =0)
mp2 = np.mean(red_cloud4.points, axis =0)

regression_point_num = 4
shape = vertices[el]
Heatmap = heatmaps[el, :, :]

Heatmap_sort = np.sort(Heatmap, 0)
sortIdx = np.argsort(Heatmap, 0)
### Select r points with maximum values on each heatmap ###
shape_sort_select = np.array([shape[sortIdx[-regression_point_num:, ld]] for ld in range(Heatmap.shape[1])])
Heatmap_sort_select = np.array(
    [Heatmap[sortIdx[-regression_point_num:, ld], ld] for ld in range(Heatmap.shape[1])]).reshape(-1,
                                                                                                  regression_point_num,
                                                                                                  1)

shape_sort_select_rep = np.expand_dims(shape_sort_select, axis=-1).repeat(regression_point_num, axis=-1)
shape2_exp_eer = shape_sort_select_rep.transpose(0, 1, 3, 2) - shape_sort_select_rep.transpose(0, 3, 1, 2)
### Compute the distance matrix ###
D_Matrix = np.linalg.norm(shape2_exp_eer, axis=3)
Heatmap_weight = Heatmap_sort_select.repeat(regression_point_num, axis=-1)
Distance_matrix = D_Matrix
### Apply MDS to D_Matrix to obtain a dimension-degraded version of local shape ###
mds = MDS(n_components=2, dissimilarity='precomputed')
shape_MDS = np.array([mds.fit_transform(Distance_matrix[i]) for i in range(Heatmap.shape[1])])
shape_MDS = np.concatenate((shape_MDS, np.zeros((Heatmap.shape[1], regression_point_num, 1))), axis=2)
landmark2D = np.sum(Heatmap_sort_select.repeat(3, axis=2) * shape_MDS, axis=1) / Heatmap_sort_select.sum(1)
N = 4
neigh = NearestNeighbors(n_neighbors=N)
IDX = []
for i in range(Heatmap.shape[1]):
    neigh.fit(shape_MDS[i])
    IDX_ = neigh.kneighbors(landmark2D[i].reshape(1, -1))[1]
    IDX.append(IDX_)
IDX = np.array(IDX)

shape_ext = np.array(
    [shape_MDS[i, IDX[i], :].reshape(-1, 3) - landmark2D[i].reshape(1, -1).repeat(N, axis=0) for i in
     range(Heatmap.shape[1])])
shape_ext_T = np.array([shape_sort_select[i, IDX[i], :] for i in range(Heatmap.shape[1])]).reshape(-1, N, 3)
### shape Centralization and Scale uniformization ###
w1 = shape_ext - np.repeat(shape_ext.mean(1, keepdims=True), N, axis=1)
w2 = shape_ext_T - np.repeat(shape_ext_T.mean(1, keepdims=True), N, axis=1)
w1 = np.linalg.norm(w1.reshape(Heatmap.shape[1], -1), axis=1).reshape(-1, 1, 1)
w2 = np.linalg.norm(w2.reshape(Heatmap.shape[1], -1), axis=1).reshape(-1, 1, 1)
shape_ext = shape_ext * w2 / w1
### Get the 3D landmark coordinates after registration ###
landmark3D = np.array([get_rigid(shape_ext[i], shape_ext_T[i])[:, 3] for i in range(Heatmap.shape[1])])

ep1 = landmark3D[0]

ep2 = landmark3D[2]
ep1=landmark[0]
ep2=landmark[2]
mp1=landmark[1]
mp2=landmark[3]

direction_vector1 = mp1 - ep1
direction_vector2 = mp2 - ep2

# Normalize the direction vector
direction_vector1 /= np.linalg.norm(direction_vector1)
direction_vector2 /= np.linalg.norm(direction_vector2)
# best_fit_cylinder = BestFitCylinder(Points(np.asarray((cloud1+cloud3).points)))
height1 = np.linalg.norm(mp1 - ep1)+3
height2 = np.linalg.norm(mp2 - ep2)
new_mp1 = (2 * ep1 + 3 * height1 * direction_vector1) / 2
new_mp2 = (2 * ep2 + 3 * height2 * direction_vector2) / 2

cylinder1 = pv.Cylinder(center=new_mp1, radius=2.5, height=3 * height1, direction=direction_vector1)
cylinder2 = pv.Cylinder(center=new_mp2, radius=2.5, height=3 * height2, direction=direction_vector2)

p = pv.Plotter()

colors = [
    (64 / 255, 224 / 255, 208 / 255),  # Green-Blue (Turquoise)
    (127 / 255, 255 / 255, 212 / 255),  # Blue-Green (Aquamarine)
    (210 / 255, 105 / 255, 30 / 255),  # Chocolate
    (101 / 255, 67 / 255, 33 / 255)  # Dark Brown
]

turqouise = "#24B0BA"
navy_blue = "#2E4A70"
gold ="#CF8A40"
pastel_red = '#ffb6c1'

colors = [
    turqouise,  # Green-Blue (Turquoise)
    navy_blue,  # Blue-Green (Aquamarine)
    gold# Dark Brown
]

cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
sphere1 = pv.Sphere(center=ep1, radius=2.5)
sphere2 = pv.Sphere(center=ep2, radius=2.5)

p = pv.Plotter()
p.add_mesh(pcd_rgbd,color = pastel_red, point_size=5)
p.add_mesh(pcd,color = turqouise, point_size=5)

#p.add_mesh(cylinder1, color =navy_blue)
#p.add_mesh(cylinder2, color =navy_blue)
p.show(screenshot="output.png")

from PIL import Image

# Open the image
image = Image.open("output.png")

# Convert the image to RGBA (if not already in that mode)
image = image.convert("RGBA")

# Get the image data
image_data = image.getdata()

# Create a new image data with transparent background
new_image_data = []
for item in image_data:
    # Set the pixel to transparent if it matches the background color (e.g., white)
    if item[:3] == (255, 255, 255):  # Adjust this condition based on your background color
        new_image_data.append((255, 255, 255, 0))  # Transparent pixel
    else:
        new_image_data.append(item)

# Update the image with the new data
image.putdata(new_image_data)

# Save the image with transparent background
image.save("output_image.png", "PNG")
