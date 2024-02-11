#this script is designed to segment the vertebrae from the spine depth dataset
import pyrender
import trimesh
import os
import pyzed.sl as sl
import numpy as np
import cv2
import open3d as o3d
import pandas as pd
import shutil
import time

df_plan=pd.read_csv("output.csv")
start_time = time.time()
dir ="/media/aidana/Extreme SSD/SpineDepth"
calib_dest_dir = "/usr/local/zed/settings/"
specimens = os.listdir(dir)
camera_nums = [0,1]
for specimen in specimens:
    recordings = len(os.listdir(os.path.join(dir, specimen, "recordings")))
    for recording in range(recordings):
        for camera_num in camera_nums:
            cur_frame = 0
            if specimen=="Specimen_2" and recording>=32:
                calib_src_dir = os.path.join(dir, specimen, "Calib_b")
            if specimen=="Specimen_5" and recording>=8:
                calib_src_dir = os.path.join(dir, specimen, "Calib_b")
            if specimen=="Specimen_7" and recording>=12:
                calib_src_dir = os.path.join(dir, specimen, "Calib_b")
            if specimen=="Specimen_9" and recording>=12:
                calib_src_dir = os.path.join(dir, specimen, "Calib_b")
            else:
                calib_src_dir = os.path.join(dir, specimen, "Calib")
            shutil.copytree(calib_src_dir, calib_dest_dir, dirs_exist_ok=True)
            print("processing specimen: {} recording {} video: {}".format(specimen, recording,camera_num))

            save_data_dir = os.path.join("/media/aidana/Extreme SSD/PoinTr dataset/segmented_spinedepth_new", specimen,
                                         "recording{}".format(recording),
                                         "pointcloud_cam_{}_recording_{}".format(camera_num, cur_frame))


            print("saving files to:{}".format(save_data_dir))
            if not os.path.exists(save_data_dir):
                os.makedirs(save_data_dir)

            video_dir = os.path.join(dir, specimen, "recordings/recording{}/Video_{}.svo".format(recording,camera_num))



            input_type = sl.InputType()
            input_type.set_from_svo_file(video_dir)
            init_params = sl.InitParameters()
            init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)

            zed = sl.Camera()


            status = zed.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                exit()

            runtime_parameters = sl.RuntimeParameters()

            zed_pose = sl.Pose()
            zed_sensors = sl.SensorsData()
            calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
            nb_frames = zed.get_svo_number_of_frames()

            while cur_frame<=3:

                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    image = sl.Mat()
                    depth = sl.Mat()
                    pcd = sl.Mat()


                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                    zed.retrieve_measure(pcd, sl.MEASURE.XYZRGBA)



                    image.write(os.path.join(save_data_dir, 'image_{}.png'.format(cur_frame)))
                    depth.write(os.path.join(save_data_dir, 'depth_{}.png'.format(cur_frame)))


                    pcd.write(os.path.join(save_data_dir, 'pointcloud_{}.ply'.format(cur_frame)))

                    pcd_init = o3d.geometry.PointCloud()
                    pcd_init.points = o3d.utility.Vector3dVector(pcd.get_data())
                    o3d.visualization.draw_geometries([pcd_init])


                    cur_frame+=1

            #

print("time taken: ", time.time() - start_time)