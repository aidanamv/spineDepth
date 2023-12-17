import open3d as o3d
import os
import numpy as np


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


def registration(target,*sources):
    picked_id_target = pick_points(target)
    transformations=[]
    for source in sources:
        picked_id_source = pick_points(source)
        assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
        assert (len(picked_id_source) == len(picked_id_target))

        corr = np.zeros((len(picked_id_source), 2))
        corr[:, 0] = picked_id_source
        corr[:, 1] = picked_id_target


        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target,
                                                o3d.utility.Vector2iVector(corr))
        # point-to-point ICP for refinement
        threshold = 10  # 3cm distance threshold
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
        transformations.append(reg_p2p.transformation)

    return transformations

dir ="/media/aidana/Extreme SSD/SpineDepth"

specimens = os.listdir(dir)

for specimen in specimens:
    print(specimen)
    save_dir=os.path.join(dir, specimen)

    if os.path.exists(os.path.join(save_dir, "transformation_matrices.txt")):
        print("skipping specimen {}".format(specimen))
        continue
    vertebra1 = o3d.io.read_triangle_mesh(os.path.join(dir, specimen, "STL/L1.stl"))
    vertebra2 = o3d.io.read_triangle_mesh(os.path.join(dir, specimen, "STL/L2.stl"))
    vertebra3 = o3d.io.read_triangle_mesh(os.path.join(dir, specimen, "STL/L3.stl"))
    vertebra4 = o3d.io.read_triangle_mesh(os.path.join(dir, specimen, "STL/L4.stl"))
    vertebra5 = o3d.io.read_triangle_mesh(os.path.join(dir, specimen, "STL/L5.stl"))

    center1 = vertebra1.get_center()


    vertebra2.translate(center1, relative=False)
    vertebra3.translate(center1, relative=False)
    vertebra4.translate(center1, relative=False)
    vertebra5.translate(center1, relative=False)

    vertebra1.compute_vertex_normals()
    vertebra2.compute_vertex_normals()
    vertebra3.compute_vertex_normals()
    vertebra4.compute_vertex_normals()
    vertebra5.compute_vertex_normals()

    pcd1 = vertebra1.sample_points_uniformly(number_of_points=10000)
    pcd2 = vertebra2.sample_points_uniformly(number_of_points=10000)
    pcd3 = vertebra3.sample_points_uniformly(number_of_points=10000)
    pcd4 = vertebra4.sample_points_uniformly(number_of_points=10000)
    pcd5 = vertebra5.sample_points_uniformly(number_of_points=10000)

    transformations= registration(pcd1,pcd2,pcd3, pcd4, pcd5)
    vertebra2.transform(transformations[0])
    vertebra3.transform(transformations[1])
    vertebra4.transform(transformations[2])
    vertebra5.transform(transformations[3])



    o3d.visualization.draw_geometries([vertebra1,vertebra2, vertebra3, vertebra4, vertebra5])

    np.savetxt(os.path.join(save_dir, "transformation_matrices.txt"),     np.array(transformations).reshape(4,-1))





