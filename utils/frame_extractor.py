import pyzed.sl as sl
import sys
import os

def parse_camera_parameters(zed):
    calibration_params = zed.get_camera_information().calibration_parameters
    settings_dict = {
        'left_fx': calibration_params.left_cam.fx,
        'left_fy': calibration_params.left_cam.fy,
        'left_cx': calibration_params.left_cam.cx,
        'left_cy': calibration_params.left_cam.cy,
        'left_disto': calibration_params.left_cam.disto.tolist(),
        # numpy array Distortion factor : [ k1, k2, p1, p2, k3 ]. Radial (k1,k2,k3) and Tangential (p1,p2) distortion.

        'right_fx': calibration_params.right_cam.fx,
        'right_fy': calibration_params.right_cam.fy,
        'right_cx': calibration_params.right_cam.cx,
        'right_cy': calibration_params.right_cam.cy,
        'right_disto': calibration_params.right_cam.disto.tolist(),

        'translation': calibration_params.T.tolist(),
        'rotation': calibration_params.R.tolist(),
        'stereo_transform': calibration_params.stereo_transform.m.tolist(),

        'resolution_width': zed.get_camera_information().camera_resolution.width,
        'resolution_height': zed.get_camera_information().camera_resolution.height,
        'fps': zed.get_camera_information().camera_fps,
        'num_frames': zed.get_svo_number_of_frames(),
        'timestamp': zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_milliseconds(),
        'depth_confidence_threshold': zed.get_runtime_parameters().confidence_threshold,
        'depth_min_range': zed.get_init_parameters().depth_minimum_distance,
        'depth_max_range': zed.get_init_parameters().depth_maximum_distance,
        'sdk_version': zed.get_sdk_version()

    }
    return settings_dict

def get_pose(zed, zed_pose, zed_sensors):
    # Get the pose of the left eye of the camera with reference to the world frame
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
    # zed_imu = zed_sensors.get_imu_data()

    # Display the translation and timestamp
    py_translation = sl.Translation()
    tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
    ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
    tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
    print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz,
                                                                           zed_pose.timestamp.get_milliseconds()))

    # Display the orientation quaternion
    py_orientation = sl.Orientation()
    ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
    oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
    oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
    ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
    print("Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

    pose_dict = {'Translation': {'Tx': tx, 'Ty': ty, 'Tz': tz},
                 'Orientation': {'Ox': ox, 'Oy': oy, 'Oz': oz, 'Ow': ow}}

    return pose_dict

def frame_extract(camera_1, camera_2,pc_dir_1,pc_dir_2):

    print("Reading SVO file: {0}".format(camera_1))
    input_type_1 = sl.InputType()
    input_type_1.set_from_svo_file(camera_1)
    init1 = sl.InitParameters(input_t=input_type_1, svo_real_time_mode=False)
    zed1 = sl.Camera()
    status1 = zed1.open(init1)
    if status1 != sl.ERROR_CODE.SUCCESS:
        print(repr(status1))
        exit()
    print("Reading SVO file: {0}".format(camera_2))
    input_type_2 = sl.InputType()
    input_type_2.set_from_svo_file(camera_2)
    init2 = sl.InitParameters(input_t=input_type_2, svo_real_time_mode=False)
    zed2 = sl.Camera()
    status2 = zed2.open(init2)
    if status2 != sl.ERROR_CODE.SUCCESS:
        print(repr(status2))
        exit()

    runtime_parameters = sl.RuntimeParameters()

    # initialize images and point cloud
    j_1 = 0
    image_1 = sl.Mat()
    image_r_1 = sl.Mat()
    pointcloud_1 = sl.Mat()

    image_2 = sl.Mat()
    image_r_2 = sl.Mat()
    pointcloud_2 = sl.Mat()

    # Enable positional tracking with default parameters
    py_transform_1 = sl.Transform()  # First create a Transform object for TrackingParameters object
    py_transform_2 = sl.Transform()  # First create a Transform object for TrackingParameters object

    tracking_parameters_1 = sl.PositionalTrackingParameters(py_transform_1)
    tracking_parameters_2 = sl.PositionalTrackingParameters(py_transform_2)

    err1 = zed1.enable_positional_tracking(tracking_parameters_1)
    if err1 != sl.ERROR_CODE.SUCCESS:
        print(repr(status1))
        exit(1)
    err2 = zed2.enable_positional_tracking(tracking_parameters_2)
    if err2 != sl.ERROR_CODE.SUCCESS:
        print(repr(status2))
        exit(1)
    zed_pose_1 = sl.Pose()
    zed_pose_2 = sl.Pose()

    zed_sensors_1 = sl.SensorsData()
    zed_sensors_2 = sl.SensorsData()

    calibration_params1 = zed1.get_camera_information().calibration_parameters
    calibration_params2 = zed2.get_camera_information().calibration_parameters


    nb_frames_1 = zed1.get_svo_number_of_frames()
    nb_frames_2 = zed2.get_svo_number_of_frames()
    print("for the camera 1")
    print("number of frames: {}".format(nb_frames_1))
    print("for the camera 2")
    print("number of frames: {}".format(nb_frames_2))


    # Grab an image, a RuntimeParameters object must be given to grab()
    if zed1.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        svo_position_1 = zed1.get_svo_position()
        # A new image is available if grab() returns SUCCESS
        # retrive and write point cloud
        print("Writing point cloud from camera 1")
        zed1.retrieve_measure(pointcloud_1, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
        os.makedirs(pc_dir_1, exist_ok=True)
        pointcloud_1.write(os.path.join(pc_dir_1, 'pointcloud.ply'))
        # Check if we have reached the end of the video
        if svo_position_1 >= (nb_frames_1 - 1):  # End of SVO
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
    if zed2.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        svo_position_2 = zed2.get_svo_position()
        # retrive and write point cloud
        print("Writing point cloud from camera 2")
        zed2.retrieve_measure(pointcloud_2, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
        os.makedirs(pc_dir_2, exist_ok=True)
        pointcloud_2.write(os.path.join(pc_dir_2, 'pointcloud.ply'))
        # Check if we have reached the end of the video
        if svo_position_2 >= (nb_frames_2 - 1):  # End of SVO
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
    zed1.close()
    zed2.close()

