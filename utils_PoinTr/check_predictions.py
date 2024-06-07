import numpy as np
import open3d as o3d
import os
import random
from scipy.spatial import KDTree, ConvexHull
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


dir ="/Users/aidanamassalimova/Documents/fold_0/val"

files= os.listdir(os.path.join(dir,"predictions"))
random.shuffle(files)
print(len(files))
nums=0
for el,file in enumerate(files):
    file = file[:-4]

    pcd_partial = o3d.io.read_point_cloud(os.path.join(dir,"partial","10102023", file, "00.pcd"))
    pcd_complete = o3d.io.read_point_cloud(os.path.join(dir,"complete","10102023", file+".pcd"))
    landmarks=np.load(os.path.join(dir, "planning", "10102023",file+".npz"))["arr_0"]
    predictions = np.load(os.path.join(dir, "predictions", file+".npz"))["arr_0"]
    pcd_prediction = o3d.geometry.PointCloud()
    pcd_prediction.points = o3d.utility.Vector3dVector(predictions.squeeze(0))
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_prediction, pcd_complete, 10, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000000))
    pcd_prediction.transform(reg_p2p.transformation)
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


    pcd_prediction.paint_uniform_color((0,0,1))
    pcd_complete.paint_uniform_color((0,1,0))
    pcd_partial.paint_uniform_color((1,0,0))
    o3d.visualization.draw_geometries([pcd_prediction, pcd_complete])

    iou = calculate_iou(pcd_prediction.points, pcd_complete.points)
    if iou>=0.8:
        # Assuming `point_cloud` is a list or array of 3D points
        point_cloud = predictions.squeeze(0)  # Your point cloud data
        point1 = landmarks[1]  # The point to check
        point2 = landmarks[2]
        convex_hull = ConvexHull(point_cloud)


        # Function to check if a point is inside the convex hull
        def is_inside_convex_hull(point, convex_hull):
            # Implement ray casting algorithm
            num_intersections = 0
            for i in range(len(convex_hull.simplices)):
                simplex = convex_hull.simplices[i]
                v0, v1 = convex_hull.points[simplex[0]], convex_hull.points[simplex[1]]
                edge_vector = v1 - v0
                v0_to_point = point - v0
                cross_product = np.cross(edge_vector, v0_to_point)
                if cross_product[2] > 0:  # Z component of cross product
                    num_intersections += 1
            return num_intersections % 2 == 1


        if is_inside_convex_hull(point1, convex_hull) and is_inside_convex_hull(point2, convex_hull):
            nums+=1
            o3d.visualization.draw_geometries([pcd_prediction,EP_left, MP_left, MP_right, EP_right])

print(nums)




