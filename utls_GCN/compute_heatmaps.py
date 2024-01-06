import open3d as o3d
import os
import numpy as np
import pyvista as pv
import copy


num_points = 16384


def Gaussian_Heatmap(Distance, sigma):
    D2 = Distance * Distance
    S2 = 2.0 * sigma * sigma
    Exponent = D2 / S2
    heatmap = np.exp(-Exponent)
    return heatmap

vertebrae_path = "/Users/aidanamassalimova/Documents/planning data original/registered vertebrae"
planning_path = "/Users/aidanamassalimova/Documents/planning data original/registered plannings"
heat_data_train = []
vertebrae_train = []
landmarks_train = []



heat_data_val = []
vertebrae_val = []
landmarks_val = []

save_train="/Users/aidanamassalimova/Documents/planning data original/final dataset/train"
save_val="/Users/aidanamassalimova/Documents/planning data original/final dataset/val"

if not os.path.exists(save_train):
    os.makedirs(save_train)

if not os.path.exists(save_val):
    os.makedirs(save_val)

files = os.listdir(vertebrae_path)
print(files)
for el,file in enumerate(files):
    print(file)
    if el <= 0.8*(len(files)):
        vert = o3d.io.read_triangle_mesh(os.path.join(vertebrae_path,file))
        vert_left_ep= o3d.geometry.TriangleMesh.create_sphere(radius =3//2)
        vert_right_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3//2)
        vert_left_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3//2)
        vert_right_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3//2)

        landmarks = np.loadtxt(os.path.join(planning_path,file[:-4]+".txt" ))

        vert_left_ep.translate(landmarks[0], relative=False)
        vert_left_mp.translate(landmarks[1], relative=False)
        vert_right_ep.translate(landmarks[2], relative=False)
        vert_right_mp.translate(landmarks[3], relative=False)

        pcd_vert = vert.sample_points_uniformly(number_of_points=num_points)
        pcd_arr = np.asarray(pcd_vert.points)
        landmarks = np.concatenate([vert_left_ep.get_center(), vert_left_mp.get_center(), vert_right_ep.get_center(),vert_right_mp.get_center()], axis=0).reshape((4, 3))

        shape_sample = pcd_arr.reshape(pcd_arr.shape[0], 1, pcd_arr.shape[1]).repeat(landmarks.shape[0], axis=1)
        Euclidean_distance_i = np.linalg.norm((shape_sample - landmarks), axis=2)
        Heat_data_i = Gaussian_Heatmap(Euclidean_distance_i, 10)
        heat_data_train.append(Heat_data_i)
        vertebrae_train.append(pcd_arr)
        landmarks_train.append(landmarks)


        for i in range(10):
            vert_copy = copy.copy(vert)
            pcd_save_copy = copy.copy(pcd_vert)
            vert_left_ep_copy = copy.copy(vert_left_ep)
            vert_left_mp_copy = copy.copy(vert_left_mp)
            vert_right_ep_copy = copy.copy(vert_right_ep)
            vert_right_mp_copy = copy.copy(vert_right_mp)
            operation = np.random.choice(["translate", "rotate"])
            axis = np.random.choice(['x', 'y', 'z'])
            rotation_range = (0, np.pi / 6)
            translation_range = (0, 10)
            if operation == "translate":
                translation = np.random.uniform(translation_range[0], translation_range[1])
                if axis == 'x':
                    vert_copy.translate([translation, 0, 0], relative=True)
                    pcd_save_copy.translate([translation, 0, 0], relative=True)
                    vert_left_ep_copy.translate([translation, 0, 0], relative=True)
                    vert_left_mp_copy.translate([translation, 0, 0], relative=True)
                    vert_right_ep_copy.translate([translation, 0, 0], relative=True)
                    vert_right_mp_copy.translate([translation, 0, 0], relative=True)

                if axis == 'y':
                    vert_copy.translate([0, translation, 0], relative=True)
                    pcd_save_copy.translate([0, translation, 0], relative=True)
                    vert_left_ep_copy.translate([0, translation, 0], relative=True)
                    vert_left_mp_copy.translate([0, translation, 0], relative=True)
                    vert_right_ep_copy.translate([0, translation, 0], relative=True)
                    vert_right_mp_copy.translate([0, translation, 0], relative=True)

                if axis == 'z':
                    vert_copy.translate([0, 0, translation], relative=True)
                    pcd_save_copy.translate([0, 0, translation], relative=True)
                    vert_left_ep_copy.translate([0, 0, translation], relative=True)
                    vert_left_mp_copy.translate([0, 0, translation], relative=True)
                    vert_right_ep_copy.translate([0, 0, translation], relative=True)
                    vert_right_mp_copy.translate([0, 0, translation], relative=True)

            if operation == "rotate":
                angle = np.random.uniform(rotation_range[0], rotation_range[1])

                if axis == 'x':
                    R = vert.get_rotation_matrix_from_xyz((angle, 0, 0))

                if axis == 'y':
                    R = vert.get_rotation_matrix_from_xyz((0, angle, 0))

                if axis == 'z':
                    R = vert.get_rotation_matrix_from_xyz((0, 0, angle))

                vert_copy.rotate(R, center=vert.get_center())
                vert_left_ep_copy.rotate(R, center=vert.get_center())
                vert_left_mp_copy.rotate(R, center=vert.get_center())
                vert_right_ep_copy.rotate(R, center=vert.get_center())
                vert_right_mp_copy.rotate(R, center=vert.get_center())
                pcd_save_copy.rotate(R, center=vert.get_center())
            pcd_arr = np.asarray(pcd_save_copy.points)
            landmarks = np.concatenate(
                [vert_left_ep_copy.get_center(), vert_left_mp_copy.get_center(), vert_right_ep_copy.get_center(),  vert_right_mp_copy.get_center()],axis=0).reshape((4, 3))
            shape_sample = pcd_arr.reshape(pcd_arr.shape[0], 1, pcd_arr.shape[1]).repeat(landmarks.shape[0], axis=1)
            Euclidean_distance_i = np.linalg.norm((shape_sample - landmarks), axis=2)
            Heat_data_i = Gaussian_Heatmap(Euclidean_distance_i, 10)
            heat_data_train.append(Heat_data_i)
            vertebrae_train.append(pcd_arr)
            landmarks_train.append(landmarks)
    else:
        vert = o3d.io.read_triangle_mesh(os.path.join(vertebrae_path, file))
        vert_left_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
        vert_right_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
        vert_left_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
        vert_right_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)

        landmarks = np.loadtxt(os.path.join(planning_path, file[:-4] + ".txt"))

        vert_left_ep.translate(landmarks[0], relative=False)
        vert_left_mp.translate(landmarks[1], relative=False)
        vert_right_ep.translate(landmarks[2], relative=False)
        vert_right_mp.translate(landmarks[3], relative=False)

        pcd_vert = vert.sample_points_uniformly(number_of_points=num_points)
        pcd_arr = np.asarray(pcd_vert.points)
        landmarks = np.concatenate([vert_left_ep.get_center(), vert_left_mp.get_center(), vert_right_ep.get_center(),
                                    vert_right_mp.get_center()], axis=0).reshape((4, 3))


        shape_sample = pcd_arr.reshape(pcd_arr.shape[0], 1, pcd_arr.shape[1]).repeat(landmarks.shape[0], axis=1)
        Euclidean_distance_i = np.linalg.norm((shape_sample - landmarks), axis=2)
        Heat_data_i = Gaussian_Heatmap(Euclidean_distance_i, 10)
        heat_data_val.append(Heat_data_i)
        vertebrae_val.append(pcd_arr)
        landmarks_val.append(landmarks)


np.savez(os.path.join(save_train,"vertices.npz"), vertebrae_train)
np.savez(os.path.join(save_train,"landmarks.npz"), landmarks_train)
np.savez(os.path.join(save_train,"heatmaps.npz"), heat_data_train)

np.savez(os.path.join(save_val,"vertices.npz"), vertebrae_val)
np.savez(os.path.join(save_val,"landmarks.npz"), landmarks_val)
np.savez(os.path.join(save_val,"heatmaps.npz"), heat_data_val)


print(len(heat_data_train))
print(len(vertebrae_train))
print(len(landmarks_train))

print(len(heat_data_val))
print(len(vertebrae_val))
print(len(landmarks_val))