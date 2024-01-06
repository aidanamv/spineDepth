import os
import open3d as o3d

dir = "/media/aidana/Extreme SSD/PoinTr dataset/segmented_spinedepth_new"

specimens=os.listdir(dir)
for specimen in specimens:
    bad_cases = []

    recordings=os.listdir(os.path.join(dir, specimen))
    for recording in recordings:
        cameras = os.listdir(os.path.join(dir, specimen, recording))
        for camera in cameras:
            frames = os.listdir(os.path.join(dir, specimen, recording, camera))
            for frame in frames:
                filepath =os.path.join(dir, specimen, recording, camera, frame)
                print(filepath)



                pcd1 = o3d.io.read_point_cloud(os.path.join(filepath, "pointcloud_vert1.ply"))
                pcd2 = o3d.io.read_point_cloud(os.path.join(filepath, "pointcloud_vert2.ply"))
                pcd3 = o3d.io.read_point_cloud(os.path.join(filepath, "pointcloud_vert3.ply"))
                pcd4 = o3d.io.read_point_cloud(os.path.join(filepath, "pointcloud_vert4.ply"))
                pcd5 = o3d.io.read_point_cloud(os.path.join(filepath, "pointcloud_vert5.ply"))

                mesh1 = o3d.io.read_triangle_mesh(os.path.join(filepath, "transformed_vertebra1.stl"))
                mesh2 = o3d.io.read_triangle_mesh(os.path.join(filepath, "transformed_vertebra2.stl"))
                mesh3 = o3d.io.read_triangle_mesh(os.path.join(filepath, "transformed_vertebra3.stl"))
                mesh4 = o3d.io.read_triangle_mesh(os.path.join(filepath, "transformed_vertebra4.stl"))
                mesh5 = o3d.io.read_triangle_mesh(os.path.join(filepath, "transformed_vertebra5.stl"))

                colors = [
                    [1, 0, 0],  # Red
                    [0, 1, 0],  # Green
                    [0, 0, 1],  # Blue
                    [1, 1, 0],  # Yellow
                    [1, 0, 1],  # Magenta
                ]

                pcd1.paint_uniform_color(colors[0])
                pcd2.paint_uniform_color(colors[1])
                pcd3.paint_uniform_color(colors[2])
                pcd4.paint_uniform_color(colors[3])
                pcd5.paint_uniform_color(colors[4])
                bbox1 = mesh1.get_oriented_bounding_box()
                bbox2 = mesh2.get_oriented_bounding_box()
                bbox3 = mesh3.get_oriented_bounding_box()
                bbox4 = mesh4.get_oriented_bounding_box()
                bbox5 = mesh5.get_oriented_bounding_box()

                pcd1 = pcd1.crop(bbox1)
                pcd2 = pcd2.crop(bbox2)
                pcd3 = pcd3.crop(bbox3)
                pcd4 = pcd4.crop(bbox4)
                pcd5 = pcd5.crop(bbox5)

                o3d.io.write_point_cloud(os.path.join(filepath, "pointcloud_vert1_cropped.ply"), pcd1)
                o3d.io.write_point_cloud(os.path.join(filepath, "pointcloud_vert2_cropped.ply"), pcd2)
                o3d.io.write_point_cloud(os.path.join(filepath, "pointcloud_vert3_cropped.ply"), pcd3)
                o3d.io.write_point_cloud(os.path.join(filepath, "pointcloud_vert4_cropped.ply"), pcd4)
                o3d.io.write_point_cloud(os.path.join(filepath, "pointcloud_vert5_cropped.ply"), pcd5)











