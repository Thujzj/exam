from models.detection import Detection,DINO
from models.action_model import ActionModel
from control.franka_control import OSC_Control,Joint_Control
from sensors.realsense import RealSenseCamera
import cv2
import torch
from utils.perception import depth_2_point,single_point_to_pc,trans_point_to_base
import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys import config_root
import open3d as o3d

if __name__ == "__main__":
    camera = RealSenseCamera()
    intr = camera.get_intr()
    
    controller_type = "OSC_POSE"
    joint_controller_type = "JOINT_POSITION"
    controller_config = "charmander.yml"
    robot_interface = FrankaInterface(
        config_root + f"/{controller_config}", use_visualizer=False)
    
    controller = OSC_Control(robot_interface, controller_type)
    joint_controller = Joint_Control(robot_interface, joint_controller_type)
    
    
    # reset the robot
    reset_joint_positions = [0.1490829586041601,
                            -0.859884785317538,
                            -0.037086424104695745,
                            -2.3489012747312845,
                            -0.005561611498809523,
                            2.22558007163479,
                            1.0187188786868417]
    joint_controller.control(reset_joint_positions, grasp=False)
    print("finish reset joint pose")
    for k in range(20):
        camera.get_sensor_info()
        
    ret, color_image, depth_image, point_cloud = camera.get_sensor_info()
    print(controller.last_eef_quat_and_pos)
    h,w,_ = color_image.shape
    # map the center to the pointcloud
    ee_in_base_mat = controller.last_eef_pos
    print(depth_image)
    pcd,points = depth_2_point(depth_image,color_image,intr)


    points_in_base = trans_point_to_base(points,ee_in_base_mat)
    pcd.points = o3d.utility.Vector3dVector(points_in_base[:,:3])
    
    o3d.io.write_point_cloud("pointcloud.pcd", pcd)
    o3d.visualization.draw_geometries([pcd])

    

    