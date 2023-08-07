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
import time

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped

pos = None
ori = None


def receive_pose(msg):
    global pos, ori
    x = msg.pose.position.x
    y = msg.pose.position.y
    z = msg.pose.position.z
    # get orientation
    qx = msg.pose.orientation.x
    qy = msg.pose.orientation.y
    qz = msg.pose.orientation.z
    qw = msg.pose.orientation.w
    
    pos = [x, y, z]
    ori = [qx, qy, qz, qw]


if __name__ == "__main__":
    camera = RealSenseCamera()
    intr = camera.get_intr()
    model_path = "models/GroundingDINO_SwinT_OGC.py"
    model_weights = "models/weights/groundingdino_swint_ogc.pth"
    
    controller_type = "OSC_POSE"
    joint_controller_type = "JOINT_POSITION"
    controller_config = "charmander.yml"
    robot_interface = FrankaInterface(
        config_root + f"/{controller_config}", use_visualizer=False)
    
    controller = OSC_Control(robot_interface, controller_type)
    joint_controller = Joint_Control(robot_interface, joint_controller_type)
    
    TEXT_PROMPT = "table. bottle . pot . box ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    detection_model = DINO(model_weights,model_path,BOX_TRESHOLD,TEXT_TRESHOLD)
    
    
    
    
    
    # reset the robot
    reset_joint_positions = [0.1046836365465484,
                            -1.012918939205638,
                            -0.10583962013775008, 
                            -2.0722461452106633, 
                            -0.04250783858566149, 
                            1.7124511218553715, 
                            0.7828379999204641]
    joint_controller.control(reset_joint_positions, grasp=False)
    # reset_top_down_positions = [0.13711555784303542, 
    #                             0.10420473280705903,
    #                             -0.02046572041338548,
    #                             -1.3077011456238596,
    #                             0.06041137177610551,
    #                             1.457737681892183,
    #                             1.0063563008507275]
    
    # joint_controller.control(reset_top_down_positions,grasp= False)
    time.sleep(1)
    print("finish reset joint pose")
    
    # avoid the error depth image
    for k in range(5):
        camera.get_sensor_info()

    ret, color_image, depth_image, point_cloud = camera.get_sensor_info()
    print(color_image.shape)
    boxes, logits, phrases = detection_model.predict(color_image,TEXT_PROMPT)
    annotated_frame = detection_model.annotate(color_image,boxes,logits,phrases)
    cv2.imwrite("annotated_image.jpg", annotated_frame)
    h,w,_ = color_image.shape
    # user_input = "bottle"
    user_input = "box"
    # boxes are cx cy w h
    for idx, phrase in enumerate(phrases):
        if phrase == user_input:
            # print(boxes[idx])
            cx,cy = int(boxes[idx][0] * w), int(boxes[idx][1] * h)
            bx,by,tx,ty = int(boxes[idx][0] * w - boxes[idx][2] * w / 2), int(boxes[idx][1] * h - boxes[idx][3] * h / 2), int(boxes[idx][0] * w + boxes[idx][2] * w / 2), int(boxes[idx][1] * h + boxes[idx][3] * h / 2)
            print(cx, cy)
            print(bx,by,tx,ty)

    # print("the center depth is:", depth_image[cy,cx])
    
    d = depth_image[by:ty,bx:tx] 
    c = color_image[by:ty,bx:tx]

    # map the center to the pointcloud
    ee_in_base_mat = controller.last_eef_pos
    print("curent pose is:", ee_in_base_mat)
    print("d shape is:", d.shape)
    # pcd, points = depth_2_point(d,c,intr)
    pcd, points = depth_2_point(depth_image,color_image,intr)
    points_in_base = trans_point_to_base(points,ee_in_base_mat)
    pcd.points = o3d.utility.Vector3dVector(points_in_base[:,:3])
    
    
    
    # get target point 
    target_x, target_y = cx , int(by/7 * 5 + ty / 7 * 2)
    c_depth = depth_image[target_y,target_x] 
    c_point = single_point_to_pc(c_depth,intr,target_x,target_y)
    c_point_in_base = trans_point_to_base(c_point,ee_in_base_mat)
    print(c_point_in_base)

    
    # change to the top-down grasp pose
    
    reset_top_down_positions = [0.17766582770305764, 
                                -0.4845583619156599,
                                -0.16949341569448773,
                                -2.062134119397962,
                                -0.039161894266981856,
                                1.6642969456195529,
                                0.8888559566537539]
    
    joint_controller.control(reset_top_down_positions,grasp= False)
    
    time.sleep(1)
    ee_top_in_base_mat = controller.last_eef_pos
    
    delta_position = c_point_in_base[:3] - ee_top_in_base_mat[:3,3] + [0,0,-0.01]
    
    # delta_position[0] = delta_position[0]*1.07 + 0.03
    # delta_position[1] = delta_position[1]*1.07 - 0.02
    delta_pos = np.concatenate((delta_position,[0,0,0]), axis=0) 
    print(delta_pos)
    new_noise = np.random.normal(0,0.001,(100,4)) + c_point_in_base 
    p = np.array(pcd.points)
    c = np.array(pcd.colors) 
    p = np.concatenate((p,new_noise[:,:3]),axis=0)
    c = np.concatenate((c,[[1,0,0] for _ in range(100)]),axis=0)
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_point_cloud("pointcloud.pcd", pcd)
    
    controller.control(delta_pos,grasp=False)
    
    target_pose = delta_pos[:3] + ee_top_in_base_mat[:3,3]

    controller.control([0,0,0,0,0,0],grasp=True)

    reset_drop_positions = [-1.4435671856947112, 
                            -1.0097872837401407,
                            -0.13023245431200314, 
                            -2.676600393226845, 
                            0.03919443071544417, 
                            2.4132427271472077, 
                            0.7542254283626874]
    
    joint_controller.control(reset_drop_positions,grasp= True)
    
    time.sleep(0.5)

    br = CvBridge()
    pub = rospy.Publisher("/realsense_camera", Image, queue_size=1)
    rospy.Subscriber('aruco_single/pose', PoseStamped, receive_pose)
    rospy.init_node("realsense_camera")
    
    rate = rospy.Rate(0.5)

    while pos is None:
        ret, color_image, depth_image, point_cloud = camera.get_sensor_info()
        pub.publish(br.cv2_to_imgmsg(color_image, "rgb8"))
        rate.sleep()
        print("the pose and ori of marker is:",pos, ori)
    
    drop_ee_in_base_mat = controller.last_eef_pos
    marker_pos = np.array(pos + [1])
    marker_pos_in_base = trans_point_to_base(marker_pos,drop_ee_in_base_mat)
    
    drop_delta_position = marker_pos_in_base[:3] - drop_ee_in_base_mat[:3,3] + [0,0,0.2]
    delta_pos = np.concatenate((drop_delta_position,[0,0,0]), axis=0) 
    
    controller.control(delta_pos,grasp=True)
    
    controller.control([0,0,0,0,0,0],grasp=False)
    