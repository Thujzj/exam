from math import cos, sin
import time

import numpy as np
from scene_models.tsdf import TSDFVolume

from control.franka_control import OSC_Control,Joint_Control
from sensors.realsense import RealSenseCamera
import cv2
import torch
from utils.perception import depth_2_point,single_point_to_pc,trans_point_to_base
import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys import config_root
import open3d as o3d
import time
from deoxys.utils.transform_utils import quat2mat,pose2mat,mat2pose,pose_in_A_to_pose_in_B,quat2axisangle,axisangle2quat
from utils.perception import trans_pose_to_base,trans_point_to_base,depth_2_point

if __name__ == "__main__":

    camera = RealSenseCamera()
    intrinsic = camera.get_intr()
    
    controller_type = "OSC_POSE"
    joint_controller_type = "JOINT_POSITION"
    controller_config = "charmander.yml"
    robot_interface = FrankaInterface(
        config_root + f"/{controller_config}", use_visualizer=False)
    
    controller = OSC_Control(robot_interface, controller_type)
    joint_controller = Joint_Control(robot_interface, joint_controller_type)
    
    reset_joint_positions = [0.1046836365465484,
                            -1.012918939205638,
                            -0.10583962013775008, 
                            -2.0722461452106633, 
                            -0.04250783858566149, 
                            1.7124511218553715, 
                            0.7828379999204641]
    joint_controller.control(reset_joint_positions, grasp=False)
    
    for k in range(20):
        camera.get_sensor_info()
        
    ret, bgr_color_image, depth_image, point_cloud = camera.get_sensor_info()
    # color_image = bgr_color_image[...,::-1]
    color_image = cv2.cvtColor(bgr_color_image, cv2.COLOR_BGR2RGB)
    # print(depth_image)
    
    cv2.imwrite('./output/depth_0.png',depth_image)
    cv2.imwrite('./output/color_0.png',color_image)
    
    ee_in_base_mat = controller.last_eef_pos
    POSE_CAMERA_IN_EE = [0.0467162 , -0.0388289 , -0.0419308]  
    QUAT_CAMERA_IN_EE = [-0.008896206339478014, 0.001529932861133454, 0.7130445599416992, 0.7010606053371948]
    CAMERA_IN_EE_MAT = pose2mat([POSE_CAMERA_IN_EE, QUAT_CAMERA_IN_EE])
    
    _,points_in_camera = depth_2_point(depth_image,color_image,intrinsic)
    
    camera_in_base_mat = pose_in_A_to_pose_in_B(CAMERA_IN_EE_MAT, ee_in_base_mat)
    points_in_base = points_in_camera.dot(camera_in_base_mat.T)

    # points_in_base = trans_point_to_base(points_in_camera,ee_in_base_mat)
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_in_base[:,:3])
    # o3d.io.write_point_cloud("pointcloud.pcd", pcd)
    # o3d.visualization.draw_geometries([pcd])

    POSE_BASE_IN_TABLE = [0,0,0]
    QUAT_BASE_IN_TABLE = [0,0,0,1]
    
    BASE_IN_TABLE_MAT = pose2mat([POSE_BASE_IN_TABLE, QUAT_BASE_IN_TABLE])
    camera_in_table_mat = pose_in_A_to_pose_in_B(camera_in_base_mat, BASE_IN_TABLE_MAT)
    
    # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #     o3d.geometry.Image(np.empty_like(depth_image)),
    #     o3d.geometry.Image(depth_image),
    #     depth_scale=1.0,
    #     depth_trunc=2.0,
    #     convert_rgb_to_intensity=False,
    # )
    # intr = o3d.camera.PinholeCameraIntrinsic(
    #         width=intrinsic.width,
    #         height=intrinsic.height,
    #         fx=intrinsic.fx,
    #         fy=intrinsic.fy,
    #         cx=intrinsic.cx,
    #         cy=intrinsic.cy,
    #     )
    
    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    # o3d.visualization.draw_geometries([pcd])
    
    tsdf = TSDFVolume(1,40)

    tsdf.integrate(depth_image ,intrinsic,np.linalg.inv(camera_in_table_mat))
    
    grid =tsdf.get_grid()
    print(grid.shape)
    cloud = tsdf.get_cloud()
    # # get how many non-zero in grid
    print(np.count_nonzero(grid))

    print(cloud)
    print(type(cloud))
    o3d.visualization.draw_geometries([cloud])
    o3d.io.write_point_cloud("pointcloud.pcd", cloud)