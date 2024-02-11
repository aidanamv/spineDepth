import os
import numpy as np
import open3d as o3d
import pandas as pd
import time
start_time = time.time()
path_stls = "/Users/aidanamassalimova/Documents/Stereo_navigation/GCN/CT_based_dataset/Aidana_planning_based/stls"
path_planning ="/Users/aidanamassalimova/Documents/Stereo_navigation/GCN/CT_based_dataset/Aidana_planning_based/plannings"
save_stls = "/Users/aidanamassalimova/Documents/Stereo_navigation/GCN/CT_based_dataset/Aidana_planning_based/registered_stls"
save_plannings = "/Users/aidanamassalimova/Documents/Stereo_navigation/GCN/CT_based_dataset/Aidana_planning_based/registered_plannings"
save_transformations = "/Users/aidanamassalimova/Documents/Stereo_navigation/GCN/CT_based_dataset/Aidana_planning_based/transformations"


if not os.path.exists(save_stls):
    os.makedirs(save_stls)
if not os.path.exists(save_plannings):
    os.makedirs(save_plannings)
if not os.path.exists(save_transformations):
    os.makedirs(save_transformations)
target =o3d.io.read_point_cloud("/Users/aidanamassalimova/Documents/Stereo_navigation/PoinTr/FinalDataset/fold_1/train/complete/10102023/Specimen_10_recording_0_cam_0_frame_0_L1.pcd")

specimens = os.listdir(path_stls)


print(specimens)

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


specimens.remove(".DS_Store")

picked_id_target = pick_points(target)


for specimen in specimens:
    print("processing specimen: {} ".format(specimen))

    #this is done to skip the processed files
    if os.path.exists(os.path.join(save_plannings, str(specimen)+"_L4.txt")):
        continue


    plannings = pd.read_excel(os.path.join(path_planning,"{}.xlsx".format(specimen)), header=None)

    vertebra1_plan = o3d.io.read_triangle_mesh(os.path.join(path_stls, specimen,specimen+"_L1.stl"))
    vertebra2_plan = o3d.io.read_triangle_mesh(os.path.join(path_stls, specimen,specimen+"_L2.stl"))
    vertebra3_plan = o3d.io.read_triangle_mesh(os.path.join(path_stls, specimen,specimen+"_L3.stl"))
    vertebra4_plan = o3d.io.read_triangle_mesh(os.path.join(path_stls, specimen,specimen+"_L4.stl"))
    vertebra5_plan = o3d.io.read_triangle_mesh(os.path.join(path_stls, specimen,specimen+"_L5.stl"))


    L1_left_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L1_right_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L1_left_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L1_right_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)

    L2_left_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L2_right_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L2_left_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L2_right_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)

    L3_left_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L3_right_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L3_left_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L3_right_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)

    L4_left_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L4_right_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L4_left_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L4_right_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)

    L5_left_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L5_right_ep = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L5_left_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)
    L5_right_mp = o3d.geometry.TriangleMesh.create_sphere(radius=3 // 2)

    L1_left_ep.translate([float(value) for value in plannings.iloc[0][0].replace('’', "").split(', ')], relative=False)
    L1_left_mp.translate([float(value) for value in plannings.iloc[0][1].replace('’', "").split(', ')], relative=False)
    L1_right_ep.translate([float(value) for value in plannings.iloc[0][2].replace('’', "").split(', ')], relative=False)
    L1_right_mp.translate([float(value) for value in plannings.iloc[0][3].replace('’', "").split(', ')], relative=False)

    L2_left_ep.translate([float(value) for value in plannings.iloc[1][0].replace('’', "").split(', ')], relative=False)
    L2_left_mp.translate([float(value) for value in plannings.iloc[1][1].replace('’', "").split(', ')], relative=False)
    L2_right_ep.translate([float(value) for value in plannings.iloc[1][2].replace('’', "").split(', ')], relative=False)
    L2_right_mp.translate([float(value) for value in plannings.iloc[1][3].replace('’', "").split(', ')], relative=False)

    L3_left_ep.translate([float(value) for value in plannings.iloc[2][0].replace('’', "").split(', ')], relative=False)
    L3_left_mp.translate([float(value) for value in plannings.iloc[2][1].replace('’', "").split(', ')], relative=False)
    L3_right_ep.translate([float(value) for value in plannings.iloc[2][2].replace('’', "").split(', ')], relative=False)
    L3_right_mp.translate([float(value) for value in plannings.iloc[2][3].replace('’', "").split(', ')], relative=False)

    L4_left_ep.translate([float(value) for value in plannings.iloc[3][0].replace('’', "").split(', ')], relative=False)
    L4_left_mp.translate([float(value) for value in plannings.iloc[3][1].replace('’', "").split(', ')], relative=False)
    L4_right_ep.translate([float(value) for value in plannings.iloc[3][2].replace('’', "").split(', ')], relative=False)
    L4_right_mp.translate([float(value) for value in plannings.iloc[3][3].replace('’', "").split(', ')], relative=False)



    if specimen!= "USR43":
        L5_left_ep.translate([float(value) for value in plannings.iloc[4][0].replace('’', "").split(', ')], relative=False)
        L5_left_mp.translate([float(value) for value in plannings.iloc[4][1].replace('’', "").split(', ')], relative=False)
        L5_right_ep.translate([float(value) for value in plannings.iloc[4][2].replace('’', "").split(', ')], relative=False)
        L5_right_mp.translate([float(value) for value in plannings.iloc[4][3].replace('’', "").split(', ')], relative=False)

    L1_left_ep.paint_uniform_color([1, 0, 0])
    L1_left_mp.paint_uniform_color([1, 0, 0])
    L1_right_mp.paint_uniform_color([1, 0, 0])
    L1_right_ep.paint_uniform_color([1, 0, 0])

    L2_left_ep.paint_uniform_color([1, 0, 0])
    L2_left_mp.paint_uniform_color([1, 0, 0])
    L2_right_mp.paint_uniform_color([1, 0, 0])
    L2_right_ep.paint_uniform_color([1, 0, 0])

    L3_left_ep.paint_uniform_color([1, 0, 0])
    L3_left_mp.paint_uniform_color([1, 0, 0])
    L3_right_mp.paint_uniform_color([1, 0, 0])
    L3_right_ep.paint_uniform_color([1, 0, 0])

    L4_left_ep.paint_uniform_color([1, 0, 0])
    L4_left_mp.paint_uniform_color([1, 0, 0])
    L4_right_mp.paint_uniform_color([1, 0, 0])
    L4_right_ep.paint_uniform_color([1, 0, 0])

    L5_left_ep.paint_uniform_color([1, 0, 0])
    L5_left_mp.paint_uniform_color([1, 0, 0])
    L5_right_mp.paint_uniform_color([1, 0, 0])
    L5_right_ep.paint_uniform_color([1, 0, 0])



    pcd1_plan = vertebra1_plan.sample_points_uniformly(number_of_points=10000)



    picked_id_source = pick_points(pcd1_plan)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(pcd1_plan, target,
                                            o3d.utility.Vector2iVector(corr))
    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 10  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1_plan, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
    L1_left_ep.transform(reg_p2p.transformation)
    L1_right_mp.transform(reg_p2p.transformation)
    L1_left_mp.transform(reg_p2p.transformation)
    L1_right_ep.transform(reg_p2p.transformation)

    L2_left_ep.transform(reg_p2p.transformation)
    L2_right_mp.transform(reg_p2p.transformation)
    L2_left_mp.transform(reg_p2p.transformation)
    L2_right_ep.transform(reg_p2p.transformation)

    L3_left_ep.transform(reg_p2p.transformation)
    L3_right_mp.transform(reg_p2p.transformation)
    L3_left_mp.transform(reg_p2p.transformation)
    L3_right_ep.transform(reg_p2p.transformation)

    L4_left_ep.transform(reg_p2p.transformation)
    L4_right_mp.transform(reg_p2p.transformation)
    L4_left_mp.transform(reg_p2p.transformation)
    L4_right_ep.transform(reg_p2p.transformation)

    L5_left_ep.transform(reg_p2p.transformation)
    L5_right_mp.transform(reg_p2p.transformation)
    L5_left_mp.transform(reg_p2p.transformation)
    L5_right_ep.transform(reg_p2p.transformation)

    vertebra1_plan.transform(reg_p2p.transformation)
    vertebra2_plan.transform(reg_p2p.transformation)
    vertebra3_plan.transform(reg_p2p.transformation)
    vertebra4_plan.transform(reg_p2p.transformation)
    vertebra5_plan.transform(reg_p2p.transformation)

    pcd1_plan.transform(reg_p2p.transformation)

    vertebra1_plan.compute_vertex_normals()
    vertebra2_plan.compute_vertex_normals()
    vertebra3_plan.compute_vertex_normals()
    vertebra4_plan.compute_vertex_normals()
    vertebra5_plan.compute_vertex_normals()



    L2_left_ep.translate(vertebra1_plan.get_center()-vertebra2_plan.get_center(), relative=True)
    L3_left_ep.translate(vertebra1_plan.get_center()-vertebra3_plan.get_center(), relative=True)
    L4_left_ep.translate(vertebra1_plan.get_center()-vertebra4_plan.get_center(), relative=True)
    L5_left_ep.translate(vertebra1_plan.get_center()-vertebra5_plan.get_center(), relative=True)

    L2_left_mp.translate(vertebra1_plan.get_center() - vertebra2_plan.get_center(), relative=True)
    L3_left_mp.translate(vertebra1_plan.get_center() - vertebra3_plan.get_center(), relative=True)
    L4_left_mp.translate(vertebra1_plan.get_center() - vertebra4_plan.get_center(), relative=True)
    L5_left_mp.translate(vertebra1_plan.get_center() - vertebra5_plan.get_center(), relative=True)

    L2_right_ep.translate(vertebra1_plan.get_center()-vertebra2_plan.get_center(), relative=True)
    L3_right_ep.translate(vertebra1_plan.get_center()-vertebra3_plan.get_center(), relative=True)
    L4_right_ep.translate(vertebra1_plan.get_center()-vertebra4_plan.get_center(), relative=True)
    L5_right_ep.translate(vertebra1_plan.get_center()-vertebra5_plan.get_center(), relative=True)

    L2_right_mp.translate(vertebra1_plan.get_center() - vertebra2_plan.get_center(), relative=True)
    L3_right_mp.translate(vertebra1_plan.get_center() - vertebra3_plan.get_center(), relative=True)
    L4_right_mp.translate(vertebra1_plan.get_center() - vertebra4_plan.get_center(), relative=True)
    L5_right_mp.translate(vertebra1_plan.get_center() - vertebra5_plan.get_center(), relative=True)

    vertebra2_plan.translate(vertebra1_plan.get_center(), relative=False)
    vertebra3_plan.translate(vertebra1_plan.get_center(), relative=False)
    vertebra4_plan.translate(vertebra1_plan.get_center(), relative=False)
    vertebra5_plan.translate(vertebra1_plan.get_center(), relative=False)
    # o3d.visualization.draw_geometries([vertebra1_plan,L1_left_mp,L1_left_ep,L1_right_mp,L1_right_ep])
    # o3d.visualization.draw_geometries([vertebra2_plan,L2_left_mp,L2_left_ep,L2_right_mp,L2_right_ep])
    # o3d.visualization.draw_geometries([vertebra3_plan,L3_left_mp,L3_left_ep,L3_right_mp,L3_right_ep])
    # o3d.visualization.draw_geometries([vertebra4_plan,L4_left_mp,L4_left_ep,L4_right_mp,L4_right_ep])
    # o3d.visualization.draw_geometries([vertebra5_plan,L5_left_mp,L5_left_ep,L5_right_mp,L5_right_ep])

    o3d.visualization.draw_geometries([L1_left_mp,L1_left_ep,L1_right_mp,L1_right_ep,
                                       L2_left_ep, L2_right_ep,L2_right_mp,L2_left_mp,
                                       L3_left_ep, L3_left_mp,L3_right_ep,L3_right_mp,
                                       L4_left_ep, L4_left_mp,L4_right_ep,L4_right_mp,
                                       L5_left_ep, L5_left_mp,L5_right_mp,L5_right_ep,
                                       vertebra1_plan, vertebra2_plan, vertebra3_plan, vertebra4_plan, vertebra5_plan])


    plan_L1 = np.vstack(
        [L1_left_ep.get_center(), L1_left_mp.get_center(), L1_right_ep.get_center(), L1_right_mp.get_center()])
    plan_L2 = np.vstack(
        [L2_left_ep.get_center(), L2_left_mp.get_center(), L2_right_ep.get_center(), L2_right_mp.get_center()])
    plan_L3 = np.vstack(
        [L3_left_ep.get_center(), L3_left_mp.get_center(), L3_right_ep.get_center(), L3_right_mp.get_center()])
    plan_L4 = np.vstack(
        [L4_left_ep.get_center(), L4_left_mp.get_center(), L4_right_ep.get_center(), L4_right_mp.get_center()])
    plan_L5 = np.vstack(
        [L5_left_ep.get_center(), L5_left_mp.get_center(), L5_right_ep.get_center(), L5_right_mp.get_center()])

    o3d.io.write_triangle_mesh(os.path.join(save_stls, specimen + "_L1.stl"), vertebra1_plan)
    o3d.io.write_triangle_mesh(os.path.join(save_stls, specimen + "_L2.stl"), vertebra2_plan)
    o3d.io.write_triangle_mesh(os.path.join(save_stls, specimen + "_L3.stl"), vertebra3_plan)
    o3d.io.write_triangle_mesh(os.path.join(save_stls, specimen + "_L4.stl"), vertebra4_plan)
    if specimen != "USR43":
        o3d.io.write_triangle_mesh(os.path.join(save_stls, specimen + "_L5.stl"), vertebra5_plan)

    o3d.visualization.draw_geometries([pcd1_plan, target, L1_left_mp, L1_left_ep, L1_right_mp, L1_right_ep])
    np.savetxt(os.path.join(save_plannings, specimen+"_L1.txt"), plan_L1)
    np.savetxt(os.path.join(save_plannings, specimen+"_L2.txt"), plan_L2)
    np.savetxt(os.path.join(save_plannings, specimen+"_L3.txt"), plan_L3)
    np.savetxt(os.path.join(save_plannings, specimen+"_L4.txt"), plan_L4)

    np.savetxt(os.path.join(save_transformations, specimen+".txt"),reg_p2p.transformation)
    if specimen!= "USR43":
        np.savetxt(os.path.join(save_plannings, specimen+"_L5.txt"), plan_L5)

