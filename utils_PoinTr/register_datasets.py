import os
import numpy as np
import open3d as o3d
import pandas as pd
import time
start_time = time.time()
dir = "/media/aidana/US study/SpineDepth"
calib_dest_dir = "/usr/local/zed/settings/"
path_stls = "/home/aidana/PycharmProjects/PSP_planning/planning data original/stls all vertebrae"
plannings = "/home/aidana/PycharmProjects/PSP_planning/planning data original/planning_points.xlsx"
specimens = os.listdir(dir)
print(specimens)
camera_nums = [0, 1]
plan_df = pd.read_excel(plannings)

save_plannings = []

for specimen in specimens:

    print("processing specimen: {} ".format(specimen))



    # reading corresponding stl files
    vertebra1 = o3d.io.read_triangle_mesh(os.path.join(dir, specimen, "STL/L1.stl"))
    vertebra2 = o3d.io.read_triangle_mesh(os.path.join(dir, specimen, "STL/L2.stl"))
    vertebra3 = o3d.io.read_triangle_mesh(os.path.join(dir, specimen, "STL/L3.stl"))
    vertebra4 = o3d.io.read_triangle_mesh(os.path.join(dir, specimen, "STL/L4.stl"))
    vertebra5 = o3d.io.read_triangle_mesh(os.path.join(dir, specimen, "STL/L5.stl"))



    vertebra1_plan = o3d.io.read_triangle_mesh(os.path.join(path_stls, specimen+"_L1.stl"))
    vertebra2_plan = o3d.io.read_triangle_mesh(os.path.join(path_stls, specimen+"_L2.stl"))
    vertebra3_plan = o3d.io.read_triangle_mesh(os.path.join(path_stls, specimen+"_L3.stl"))
    vertebra4_plan = o3d.io.read_triangle_mesh(os.path.join(path_stls, specimen+"_L4.stl"))
    vertebra5_plan = o3d.io.read_triangle_mesh(os.path.join(path_stls, specimen+"_L5.stl"))

    vertebrae =[vertebra1,vertebra2, vertebra3, vertebra4, vertebra5]

    vertebrae_plan =[vertebra1_plan,vertebra2_plan,vertebra3_plan, vertebra4_plan, vertebra5_plan]
    for el,vertebra_plan in enumerate(vertebrae_plan):
        vert_left_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
        vert_right_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
        vert_left_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
        vert_right_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
        specimen_info = plan_df[plan_df["Specimen"] == specimen]
        vert_info = specimen_info[specimen_info["Level"] == "L{}".format(el+1)].reset_index(drop=True)
        vert_left_ep.translate([float(i.replace("’", "")) for i in vert_info.LeftEntryPoint[0].split(',')],
                               relative=False)
        vert_right_ep.translate([float(i.replace("’", "")) for i in vert_info.RightEntryPoint[0].split(',')],
                                relative=False)
        vert_left_mp.translate([float(i.replace("’", "")) for i in vert_info.LeftMidPoint[0].split(',')],
                               relative=False)
        vert_right_mp.translate([float(i.replace("’", "")) for i in vert_info.RightMidPoint[0].split(',')],
                                relative=False)

        vert_left_ep.paint_uniform_color([1, 0, 0])
        vert_left_mp.paint_uniform_color([1, 0, 0])
        vert_right_ep.paint_uniform_color([1, 0, 0])
        vert_right_mp.paint_uniform_color([1, 0, 0])
        vert_left_ep.translate(vertebrae[el].get_center()-vertebra_plan.get_center(), relative=True)
        vert_left_mp.translate(vertebrae[el].get_center()-vertebra_plan.get_center(), relative=True)
        vert_right_ep.translate(vertebrae[el].get_center()-vertebra_plan.get_center(), relative=True)
        vert_right_mp.translate(vertebrae[el].get_center()-vertebra_plan.get_center(), relative=True)

        vertebra_plan.translate(vertebrae[el].get_center(), relative=False)
        pcd1_plan = vertebra_plan.sample_points_uniformly(number_of_points=10000)
        pcd1 = vertebrae[el].sample_points_uniformly(number_of_points=10000)


        def pick_points(pcd):
            print("")
            print(
                "1) Please pick at least three correspondences using [shift + left click]"
            )
            print("   Press [shift + right click] to undo point picking")
            print("2) After picking points, press 'Q' to close the window")
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window()
            vis.add_geometry(pcd)
            vis.run()  # user picks points
            vis.destroy_window()
            print("")
            return vis.get_picked_points()

        picked_id_source = pick_points(pcd1_plan)
        picked_id_target = pick_points(pcd1)
        assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
        assert (len(picked_id_source) == len(picked_id_target))
        corr = np.zeros((len(picked_id_source), 2))
        corr[:, 0] = picked_id_source
        corr[:, 1] = picked_id_target

        # estimate rough transformation using correspondences
        print("Compute a rough transform using the correspondences given by user")
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(pcd1_plan, pcd1,
                                                o3d.utility.Vector2iVector(corr))
        # point-to-point ICP for refinement
        print("Perform point-to-point ICP refinement")
        threshold = 10  # 3cm distance threshold
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd1_plan, pcd1, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
        vert_left_ep.transform(reg_p2p.transformation)
        vert_left_mp.transform(reg_p2p.transformation)
        vert_right_ep.transform(reg_p2p.transformation)
        vert_right_mp.transform(reg_p2p.transformation)
        pcd1_plan.transform(reg_p2p.transformation)
        o3d.visualization.draw_geometries([pcd1_plan, pcd1, vert_right_ep,vert_left_ep,vert_left_mp,vert_right_mp])
        save_plannings.append({"{}_L{}".format(specimen,el+1):
                            [vert_left_ep.get_center(),
                            vert_left_mp.get_center(),
                            vert_right_ep.get_center(),
                            vert_right_mp.get_center()] })
        print({"{}_L{}".format(specimen,el+1):
                            [vert_left_ep.get_center(),
                            vert_left_mp.get_center(),
                            vert_right_ep.get_center(),
                            vert_right_mp.get_center()]})

        df = pd.DataFrame(save_plannings)
        df.to_csv('output.csv', index=False)




