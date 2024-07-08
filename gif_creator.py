import open3d as o3d
import numpy as np
import imageio.v2 as imageio
import os
import pyvista as pv
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def Gaussian_Heatmap(Distance, sigma):
    D2 = Distance * Distance
    S2 = 2.0 * sigma * sigma
    Exponent = D2 / S2
    heatmap = np.exp(-Exponent)
    return heatmap
file = "/Users/aidanamassalimova/Documents/full_pcd.pcd"
screw_file = "/Users/aidanamassalimova/Documents/screw.stl"
planning_file = "/Users/aidanamassalimova/Documents/MICCAI paper/temp data/FinalDataset_2048/fold_0/train/planning/10102023/Specimen_3_recording_0_cam_0_frame_0_L3.npz"

planning = np.load(planning_file)["arr_0"]
# Load the point cloud
screw_stl = pv.read(screw_file)
point_cloud = o3d.io.read_point_cloud(file)
#point_cloud.paint_uniform_color([1.0, 0, 1])

direction_vector1 = planning[1] - planning[0]
direction_vector2 = planning[3] - planning[2]


# Normalize the direction vector
direction_vector1 /= np.linalg.norm(direction_vector1)
direction_vector2 /= np.linalg.norm(direction_vector2)



vertices = np.asarray(screw_stl.points)

# Center the vertices to perform PCA
centroid = np.mean(vertices, axis=0)
centered_vertices = vertices - centroid

# Compute the covariance matrix
covariance_matrix = np.cov(centered_vertices, rowvar=False)

# Perform eigen decomposition
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Sort eigenvalues and eigenvectors in descending order
sort_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_indices]
eigenvectors = eigenvectors[:, sort_indices]

# Principal axis is the eigenvector corresponding to the largest eigenvalue
principal_axis = eigenvectors[:,0]

cylinder1 = pv.Cylinder(radius=3,height=100, direction=direction_vector1,center=screw_stl.center)
cylinder2 = pv.Cylinder(radius=3,height=100, direction=principal_axis,center=screw_stl.center)
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

rot_mat1 = rotation_matrix_from_vectors(principal_axis,-direction_vector1)
rot_mat2 = rotation_matrix_from_vectors(principal_axis,-direction_vector2)

screw_1 = o3d.io.read_triangle_mesh(screw_file)
screw_2 = o3d.io.read_triangle_mesh(screw_file)
screw_1.rotate(rot_mat1)
screw_2.rotate(rot_mat2)

screw_1.translate(planning[1], relative=False)
screw_2.translate(planning[3], relative=False)

screw_1.compute_vertex_normals()
screw_2.compute_vertex_normals()









# Create a visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# Add the point cloud to the visualizer
vis.add_geometry(point_cloud)
#vis.add_geometry(screw_1)
#vis.add_geometry(screw_2)
# Set background color to white
opt = vis.get_render_option()
#opt.background_color = np.array([1, 1, 1])
opt.point_size = 7
opt.background_color = [11/255, 37/255, 75/255]


# Directory to save frames
frames_dir = "frames"
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# List to store frame filenames
frame_filenames = []


def remove_background(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert image to RGBA
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to smooth image (optional)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Create a mask with white foreground and black background
    mask = cv2.bitwise_not(thresh)

    # Apply the mask to extract the foreground
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Split channels and merge with alpha channel
    b, g, r, a = cv2.split(masked_image)
    rgba_image = cv2.merge((b, g, r, mask))

    # Save the resulting image
    cv2.imwrite(image_path, rgba_image)# Function to capture the screen and save frames
def capture_frame(vis):
    frame_index = len(frame_filenames)
    filename = f"{frames_dir}/frame_{frame_index:03d}.png"
    vis.capture_screen_image(filename)
    frame_filenames.append(filename)
    return False

# Register key callback to save frames
vis.register_key_callback(ord("S"), capture_frame)

# Run the visualizer
vis.run()
vis.destroy_window()
# Remove the white background from frames




# Create a GIF from the frames
gif_filename = 'pointcloud.gif'

from PIL import Image, ImageSequence



# Create a list to store PIL Image objects
images = []

# Open each image and append to the list
for path in frame_filenames:
    with Image.open(path) as img:
        # Convert images to RGBA mode if not already (for transparency)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        images.append(img.copy())



# Save as animated GIF using save_all() method
images[0].save(gif_filename, save_all=True, append_images=images[1:], duration=500, loop=0)

print(f'GIF saved as {gif_filename}')
pcd_arr = np.asarray(point_cloud.points)
shape_sample = pcd_arr.reshape(pcd_arr.shape[0], 1, pcd_arr.shape[1]).repeat(planning.shape[0], axis=1)
Euclidean_distance_i = np.linalg.norm((shape_sample - planning), axis=2)
Heat_data_i = Gaussian_Heatmap(Euclidean_distance_i, 10)
pcd_rgbd = pv.PolyData(pcd_arr)
for i in range(4):
    pcd_rgbd.point_data["Predictions"] = Heat_data_i[:, i]
    plotter = pv.Plotter()

    # Add the point cloud to the plotter
    plotter.add_mesh(pcd_rgbd, color="#B929CD", point_size=5)

    # Set the background color to dark blue (RGB: 0, 0, 139)
    plotter.set_background( [11/255, 37/255, 75/255])

    # Show the plot
    plotter.show()