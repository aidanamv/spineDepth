import os
import numpy as np
import open3d as o3d
import copy
import time


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
# Load the STL file
dir ="/Volumes/Extreme SSD/PoinTr dataset/segmented_spinedepth_new"
save_dir ="/Volumes/Extreme SSD/aligned_2048"
num_points_to_sample = '2048'


specimens = os.listdir(dir)


for specimen in specimens:
    transformations=np.loadtxt(os.path.join("/Volumes/Extreme SSD/SpineDepth",specimen,"transformation_matrices.txt"))
    recordings = os.listdir(os.path.join(dir, specimen))
    for recording in recordings:
        cameras = os.listdir(os.path.join(dir, specimen, recording))
        for camera in cameras:
            frames = os.listdir(os.path.join(dir, specimen, recording,camera))
            for frame in frames:

                print("Processing specimen {} recording {} camera {} frame {}".format(specimen,recording,camera,frame))
                f_path = os.path.join(dir,specimen,recording,camera,frame)



                mesh1 = o3d.io.read_triangle_mesh(os.path.join("/Volumes/Extreme SSD/SpineDepth",specimen,"STL/L1.stl"))
                mesh2 = o3d.io.read_triangle_mesh(os.path.join("/Volumes/Extreme SSD/SpineDepth",specimen,"STL/L2.stl"))
                mesh3 = o3d.io.read_triangle_mesh(os.path.join("/Volumes/Extreme SSD/SpineDepth",specimen,"STL/L3.stl"))
                mesh4 = o3d.io.read_triangle_mesh(os.path.join("/Volumes/Extreme SSD/SpineDepth",specimen,"STL/L4.stl"))
                mesh5 = o3d.io.read_triangle_mesh(os.path.join("/Volumes/Extreme SSD/SpineDepth",specimen,"STL/L5.stl"))

                pcd1 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert1_cropped.ply"))
                pcd2 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert2_cropped.ply"))
                pcd3 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert3_cropped.ply"))
                pcd4 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert4_cropped.ply"))
                pcd5 = o3d.io.read_point_cloud(os.path.join(f_path,"pointcloud_vert5_cropped.ply"))

                planning = np.loadtxt(os.path.join(f_path,"planning.txt"))

                EP_left_vert1 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                EP_left_vert2 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                EP_left_vert3 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                EP_left_vert4 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                EP_left_vert5 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

                EP_right_vert1 =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                EP_right_vert2 =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                EP_right_vert3 =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                EP_right_vert4 =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                EP_right_vert5 =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

                MP_left_vert1 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                MP_left_vert2 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                MP_left_vert3 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                MP_left_vert4 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                MP_left_vert5 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

                MP_right_vert1 =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                MP_right_vert2 =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                MP_right_vert3 =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                MP_right_vert4 =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                MP_right_vert5 =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

                mesh1.compute_vertex_normals()
                mesh2.compute_vertex_normals()
                mesh3.compute_vertex_normals()
                mesh4.compute_vertex_normals()
                mesh5.compute_vertex_normals()


                mesh1_pcd = mesh1.sample_points_uniformly(number_of_points=num_points_to_sample)
                mesh2_pcd = mesh2.sample_points_uniformly(num_points_to_sample)
                mesh3_pcd = mesh3.sample_points_uniformly(num_points_to_sample)
                mesh4_pcd = mesh4.sample_points_uniformly(num_points_to_sample)
                mesh5_pcd = mesh5.sample_points_uniformly(num_points_to_sample)






                center_target = mesh1_pcd.get_center()
                center2 = mesh2_pcd.get_center()
                center3 = mesh3_pcd.get_center()
                center4 = mesh4_pcd.get_center()
                center5 = mesh5_pcd.get_center()

                mesh2_pcd.translate(center_target, relative=False).transform(transformations[0].reshape(-1,4))
                mesh3_pcd.translate(center_target, relative=False).transform(transformations[1].reshape(-1,4))
                mesh4_pcd.translate(center_target, relative=False).transform(transformations[2].reshape(-1,4))
                mesh5_pcd.translate(center_target, relative=False).transform(transformations[3].reshape(-1,4))

                pcd2.translate(center_target-center2, relative=True).transform(transformations[0].reshape(-1,4))
                pcd3.translate(center_target-center3, relative=True).transform(transformations[1].reshape(-1,4))
                pcd4.translate(center_target-center4, relative=True).transform(transformations[2].reshape(-1,4))
                pcd5.translate(center_target-center5, relative=True).transform(transformations[3].reshape(-1,4))
                EP_left_vert1.translate(planning[0][:3], relative=False)
                MP_left_vert1.translate(planning[1][:3], relative=False)
                EP_right_vert1.translate(planning[2][:3], relative=False)
                MP_right_vert1.translate(planning[3][:3], relative=False)


                EP_left_vert2.translate(planning[4][:3], relative=False).translate(center_target-center2, relative=True).transform(transformations[0].reshape(-1,4))
                MP_left_vert2.translate(planning[5][:3], relative=False).translate(center_target-center2, relative=True).transform(transformations[0].reshape(-1,4))
                EP_right_vert2.translate(planning[6][:3], relative=False).translate(center_target-center2, relative=True).transform(transformations[0].reshape(-1,4))
                MP_right_vert2.translate(planning[7][:3], relative=False).translate(center_target-center2, relative=True).transform(transformations[0].reshape(-1,4))

                EP_left_vert3.translate(planning[8][:3], relative=False).translate(center_target-center3, relative=True).transform(transformations[1].reshape(-1,4))
                MP_left_vert3.translate(planning[9][:3], relative=False).translate(center_target-center3, relative=True).transform(transformations[1].reshape(-1,4))
                EP_right_vert3.translate(planning[10][:3], relative=False).translate(center_target-center3, relative=True).transform(transformations[1].reshape(-1,4))
                MP_right_vert3.translate(planning[11][:3], relative=False).translate(center_target-center3, relative=True).transform(transformations[1].reshape(-1,4))

                EP_left_vert4.translate(planning[12][:3], relative=False).translate(center_target-center4, relative=True).transform(transformations[2].reshape(-1,4))
                MP_left_vert4.translate(planning[13][:3], relative=False).translate(center_target-center4, relative=True).transform(transformations[2].reshape(-1,4))
                EP_right_vert4.translate(planning[14][:3], relative=False).translate(center_target-center4, relative=True).transform(transformations[2].reshape(-1,4))
                MP_right_vert4.translate(planning[15][:3], relative=False).translate(center_target-center4, relative=True).transform(transformations[2].reshape(-1,4))

                EP_left_vert5.translate(planning[16][:3], relative=False).translate(center_target-center5, relative=True).transform(transformations[3].reshape(-1,4))
                MP_left_vert5.translate(planning[17][:3], relative=False).translate(center_target-center5, relative=True).transform(transformations[3].reshape(-1,4))
                EP_right_vert5.translate(planning[18][:3], relative=False).translate(center_target-center5, relative=True).transform(transformations[3].reshape(-1,4))
                MP_right_vert5.translate(planning[19][:3], relative=False).translate(center_target-center5, relative=True).transform(transformations[3].reshape(-1,4))


                #
                # o3d.visualization.draw_geometries([mesh1_pcd, pcd1, EP_right_vert1, EP_left_vert1, MP_left_vert1, MP_right_vert1])
                # o3d.visualization.draw_geometries([mesh2_pcd, pcd2, EP_right_vert2, EP_left_vert2, MP_left_vert2, MP_right_vert2])
                # o3d.visualization.draw_geometries([mesh3_pcd, pcd3, EP_right_vert3, EP_left_vert3, MP_left_vert3, MP_right_vert3])
                # o3d.visualization.draw_geometries([mesh4_pcd, pcd4, EP_right_vert4, EP_left_vert4, MP_left_vert4, MP_right_vert4])
                # o3d.visualization.draw_geometries([mesh5_pcd, pcd5, EP_right_vert5, EP_left_vert5, MP_left_vert5, MP_right_vert5])

                planning_L1 = [EP_left_vert1.get_center(), MP_left_vert1.get_center(),EP_right_vert1.get_center(),MP_right_vert1.get_center()]
                planning_L2 = [EP_left_vert2.get_center(), MP_left_vert2.get_center(),EP_right_vert2.get_center(),MP_right_vert2.get_center()]
                planning_L3 = [EP_left_vert3.get_center(), MP_left_vert3.get_center(),EP_right_vert3.get_center(),MP_right_vert3.get_center()]
                planning_L4 = [EP_left_vert4.get_center(), MP_left_vert4.get_center(),EP_right_vert4.get_center(),MP_right_vert4.get_center()]
                planning_L5 = [EP_left_vert5.get_center(), MP_left_vert5.get_center(),EP_right_vert5.get_center(),MP_right_vert5.get_center()]




                #downsample input pcds
                if np.asarray(pcd1.points).shape[0]>2048:
                    random_indices11= np.random.choice(np.asarray(pcd1.points).shape[0], 2048, replace=False)
                    pcd1 = pcd1.select_by_index(random_indices11)
                if np.asarray(pcd2.points).shape[0]>2048:
                    random_indices22= np.random.choice(np.asarray(pcd2.points).shape[0], 2048, replace=False)
                    pcd2 = pcd2.select_by_index(random_indices22)
                if np.asarray(pcd3.points).shape[0]>2048:
                    random_indices33= np.random.choice(np.asarray(pcd3.points).shape[0], 2048, replace=False)
                    pcd3 = pcd3.select_by_index(random_indices33)
                if np.asarray(pcd4.points).shape[0]>2048:
                    random_indices44= np.random.choice(np.asarray(pcd4.points).shape[0], 2048, replace=False)
                    pcd4 = pcd4.select_by_index(random_indices44)
                if np.asarray(pcd5.points).shape[0]>2048:
                    random_indices55= np.random.choice(np.asarray(pcd5.points).shape[0], 2048, replace=False)
                    pcd5 = pcd5.select_by_index(random_indices55)




                pcd_inputs =[pcd1, pcd2, pcd3, pcd4, pcd5]
                pcd_outputs = [mesh1_pcd, mesh2_pcd, mesh3_pcd, mesh4_pcd, mesh5_pcd]
                planning = [planning_L1,planning_L2, planning_L3, planning_L4, planning_L5]



                for el, pcd in enumerate(pcd_outputs):
                    assert np.asarray(pcd.points).shape[0] == num_points_to_sample
                    if np.asarray(pcd_inputs[el].points).shape[0] == 2048:
                        filename = specimen + "_{}".format(recording)+"_{}".format(camera)+"_{}".format(frame)+"_L{}".format(el+1)
                        if not os.path.exists(os.path.join(save_dir,"partial",filename)):
                            os.makedirs(os.path.join(save_dir,"partial",filename))
                        o3d.io.write_point_cloud(os.path.join(save_dir, "partial",filename, "00.pcd"), pcd_inputs[el])
                        if not os.path.exists(os.path.join(save_dir, "complete")):
                            os.makedirs(os.path.join(save_dir, "complete"))
                        o3d.io.write_point_cloud(os.path.join(save_dir, "complete",filename+ ".pcd"), pcd)
                        if not os.path.exists(os.path.join(save_dir, "planning")):
                            os.makedirs(os.path.join(save_dir, "planning"))
                        np.savez(os.path.join(save_dir, "planning" , "{}.npz".format(filename)),planning[el])



