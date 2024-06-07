import os
import open3d as o3d
import numpy as np
from PIL import Image
dir = "/media/aidana/aidana/PoinTr dataset/segmented_spinedepth_new"


def create_point_cloud(color_image_path, depth_image_path, intrinsics):
    # Load color and depth images
    color_image = np.array(Image.open(color_image_path))
    depth_image = np.array(Image.open(depth_image_path))

    # Get image dimensions
    height, width = color_image.shape[:2]

    # Convert depth image to point cloud
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - intrinsics[0]) * depth_image / intrinsics[2]
    y = (v - intrinsics[1]) * depth_image / intrinsics[3]
    z = depth_image

    # Create point cloud
    point_cloud = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Get colors from color image
    colors = color_image.reshape(-1, 3)

    # Create open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to range [0, 1]

    return pcd


def map_image_to_point_cloud(image, point_cloud):
    """
    Map the colors from the image onto the point cloud vertices.

    :param image: The RGB image.
    :param point_cloud: The point cloud.
    :return: The textured point cloud.
    """
    height, width = len(point_cloud),len(point_cloud)
    textured_point_cloud = np.zeros_like(point_cloud)

    for i in range(height):
        for j in range(width):
            x, y, z, r, g, b, a = point_cloud[i, j]

            # Map the pixel from the image onto the point in the point cloud
            u = int(j * (image.shape[1] / point_cloud.shape[1]))
            v = int(i * (image.shape[0] / point_cloud.shape[0]))
            if image[u,v] == 255:
                textured_point_cloud[i, j] = [x, y, z, 255,255,0]

    return textured_point_cloud
specimens = os.listdir(dir)
save_pcd_dir = "./data/point_cloud"
save_seg_dir = "./data/seg"
if not os.path.exists(save_pcd_dir):
    os.makedirs(save_pcd_dir)
if not os.path.exists(save_seg_dir):
    os.makedirs(save_seg_dir)
for specimen in specimens:
    recordings = os.listdir(os.path.join(dir, specimen))
    for recording in recordings:
        views = os.listdir(os.path.join(dir, specimen, recording))
        for view in views:
            frames = os.listdir(os.path.join(dir, specimen, recording, view))
            for frame in frames:
                if os.path.exists(os.path.join(dir, specimen, recording, view,frame, "full_pcd.ply")):
                    pcd = o3d.io.read_point_cloud(os.path.join(dir, specimen, recording, view,frame, "full_pcd.ply"))
                    l1 = o3d.io.read_point_cloud(os.path.join(dir, specimen, recording, view,frame, "pointcloud_vert1.ply"))
                    # Load image and point cloud (you need to replace these with your own data)
                    color_image_path = os.path.join(dir, specimen, recording, view,frame,"image.png")
                    depth_image_path = os.path.join(dir, specimen, recording, view,frame,"depth.png")


                    # Example camera intrinsics [fx, fy, cx, cy]
                    intrinsics = [525.0, 525.0, 319.5, 239.5]



                    # Create point cloud
                    point_cloud = create_point_cloud(color_image_path, depth_image_path, intrinsics)

                    # Visualize point cloud
                    o3d.visualization.draw_geometries([point_cloud])
