# DURING REGISTRATION I FORGOT THE DIFFERENCE IN TRANSLATIONS SO L2-L5 ARE NOT TRANSLATED CORRECTLY
import os
import open3d as o3d
import numpy as np
stls_dir = "/Users/aidanamassalimova/Documents/Stereo_navigation/GCN/CT_based_dataset/Aidana_planning_based/stls"
registered_stls_dir = "/Users/aidanamassalimova/Documents/Stereo_navigation/GCN/CT_based_dataset/Aidana_planning_based/registered_stls"

planning_dir ="/Users/aidanamassalimova/Documents/Stereo_navigation/GCN/CT_based_dataset/Aidana_planning_based/registered_plannings"

specimens = os.listdir(stls_dir)
specimens.remove(".DS_Store")

for specimen in specimens:
    print(specimen)
    landmarks_L2 = np.loadtxt(os.path.join(planning_dir, "{}_L2.txt".format(specimen)))
    landmarks_L3 = np.loadtxt(os.path.join(planning_dir, "{}_L3.txt".format(specimen)))
    landmarks_L4 = np.loadtxt(os.path.join(planning_dir, "{}_L4.txt".format(specimen)))
    if specimen!= "USR43":

        landmarks_L5 = np.loadtxt(os.path.join(planning_dir, "{}_L5.txt".format(specimen)))

    mesh1 = o3d.io.read_triangle_mesh(os.path.join(registered_stls_dir, "{}_L1.stl".format(specimen)))
    mesh2 = o3d.io.read_triangle_mesh(os.path.join(registered_stls_dir, "{}_L2.stl".format(specimen)))
    mesh3 = o3d.io.read_triangle_mesh(os.path.join(registered_stls_dir, "{}_L3.stl".format(specimen)))
    mesh4 = o3d.io.read_triangle_mesh(os.path.join(registered_stls_dir, "{}_L4.stl".format(specimen)))
    mesh5 = o3d.io.read_triangle_mesh(os.path.join(registered_stls_dir, "{}_L5.stl".format(specimen)))

    landmarks_L2_new = landmarks_L2+ mesh1.get_center() - mesh2.get_center()
    landmarks_L3_new = landmarks_L3+ mesh1.get_center() - mesh2.get_center()
    landmarks_L4_new = landmarks_L4+ mesh1.get_center() - mesh2.get_center()
    landmarks_L5_new = landmarks_L5+ mesh1.get_center() - mesh2.get_center()




    mesh2.translate(mesh1.get_center(), relative=False)
    mesh3.translate(mesh1.get_center(), relative=False)
    mesh4.translate(mesh1.get_center(), relative=False)
    mesh5.translate(mesh1.get_center(), relative=False)



    Left_ep = o3d.geometry.TriangleMesh.create_sphere(radius=1.5).translate(landmarks_L2_new[0], relative= False)
    Right_ep = o3d.geometry.TriangleMesh.create_sphere(radius=1.5).translate(landmarks_L2_new[2], relative= False)

    Left_ep.paint_uniform_color([0,0,1])
    Right_ep.paint_uniform_color([0,0,1])


    np.savetxt(os.path.join(planning_dir, specimen + "_L2.txt"), landmarks_L2_new)
    np.savetxt(os.path.join(planning_dir, specimen + "_L3.txt"), landmarks_L3_new)
    np.savetxt(os.path.join(planning_dir, specimen + "_L4.txt"), landmarks_L4_new)
    if specimen!= "USR43":
        np.savetxt(os.path.join(planning_dir, specimen + "_L5.txt"), landmarks_L5_new)
        o3d.io.write_triangle_mesh(os.path.join(registered_stls_dir, "{}_L5.stl".format(specimen)), mesh5)


    o3d.io.write_triangle_mesh(os.path.join(registered_stls_dir, "{}_L2.stl".format(specimen)), mesh2)
    o3d.io.write_triangle_mesh(os.path.join(registered_stls_dir, "{}_L3.stl".format(specimen)), mesh3)
    o3d.io.write_triangle_mesh(os.path.join(registered_stls_dir, "{}_L4.stl".format(specimen)), mesh4)
