from socket import *
from models.detection import Detection,DINO
from models.action_model import ActionModel
from control.franka_control import OSC_Control,Joint_Control
from sensors.realsense import RealSenseCamera
import cv2
import torch
from utils.perception import depth_2_point,single_point_to_pc,trans_point_to_base,trans_pose_to_base
import numpy as np
from std_msgs.msg import Int32MultiArray,Float32MultiArray
from deoxys.franka_interface import FrankaInterface
from deoxys import config_root
import open3d as o3d
import time

from deoxys.utils.transform_utils import quat2mat,quat2axisangle,mat2quat
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import pyrealsense2 as rs

TEXT_PROMPT = "table. bottle . pot . box ." 
pos = None
ori = None
QUAT_CAMERA_IN_EE = [-0.008896206339478014, 0.001529932861133454, 0.7130445599416992, 0.7010606053371948]

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


class GraspModel():
    def __init__(self,detection_model, controller, joint_controller):

        self.detection_model = detection_model
        self.controller = controller
        self.joint_controller = joint_controller
        self.reset_joint_positions =  [0.03260338935522349, -0.8274320986944705, 0.22758139832660154, -2.182303152151275, -0.013391416776898955, 1.8901679635128885, 1.0057438689967648]
        self.reset_top_down_positions = self.reset_joint_positions
        # self.reset_top_down_positions = [-0.25230813224093124, -0.2782045577485158, 0.18375603899247503, -1.6432780397816706, -0.07133616156246558, 1.493804914611447, 0.9094413397869002]
        
        self.reset_drop_positions = [-1.4751571585420975, -0.952519619116397, -0.521451397281415, -1.8337223887415657, -0.5761615101366534, 1.5870364188844388, 1.132414373020331]
        
        self.end_pose = [-0.27656061996493425, -0.13961071318524046, 0.6569752552360109, -1.679524105632534, -0.08407770457465895, 2.0636291815730368, 0.9029973374227045]
        self.marker_pos_in_base = None
        
        self.check_pose = [-0.9445412621581764, -1.2132220859101928, -0.4076681861124541, -1.991061465450364, -0.7218532037006483, 2.401600604538564, 0.949921643992265]
        
        import argparse
        self.intr = argparse.Namespace(**{
            'width': 640,
            'height': 480,
            'fx':596.382688,
            'fy':596.701788,
            'ppx':333.837001,
            'ppy':254.401211,
            'model':rs.distortion.modified_brown_conrady,
            "coeffs" : [0.130405, -0.265366, 0.000961, 0.003105, 0.0]
        })
        
        self.flag = False
    def reset_pose(self,pose,grasp):
        self.joint_controller.control(pose, grasp=grasp)

    
    def grasp_object(self,category):

        if pos is None:
            self.reset_pose(self.reset_drop_positions,grasp=False)
            time.sleep(0.5)
            self.init_quat, self.init_pos = self.controller.last_eef_quat_and_pos


        rospy.Subscriber('aruco_single/pose', PoseStamped, receive_pose)
        # rospy.Subscriber("graspnet", Float32MultiArray, receive_grasp_point)
        # detect_pub = rospy.Publisher("detected_mask",Int32MultiArray, queue_size=1 )
        rospy.init_node("grasp")
        
        rate = rospy.Rate(0.5)

        while pos is None:
            rate.sleep()
            print("the pose and ori of marker is:",pos, ori)
        

        drop_ee_in_base_mat = controller.last_eef_pos
        marker_pos = np.array(pos + [1])
        
        if self.flag is False:
            self.marker_pos_in_base = trans_point_to_base(marker_pos,drop_ee_in_base_mat)

        
        self.reset_pose(self.reset_joint_positions,grasp=False)
        

        flag = True
        cx = None
        while flag is True:

            color_image = np.load("output/color_image.npy")
            depth_image = np.load("output/depth_image.npy")
            boxes, logits, phrases = self.detection_model.predict(color_image,TEXT_PROMPT)
            annotated_frame = self.detection_model.annotate(color_image,boxes,logits,phrases)

            h,w,_ = color_image.shape

            print(phrases)
            for idx, phrase in enumerate(phrases):
                if category in phrase:
                    # print(boxes[idx])
                    cx,cy = int(boxes[idx][0] * w), int(boxes[idx][1] * h)
                    bx,by,tx,ty = int(boxes[idx][0] * w - boxes[idx][2] * w / 2), int(boxes[idx][1] * h - boxes[idx][3] * h / 2), int(boxes[idx][0] * w + boxes[idx][2] * w / 2), int(boxes[idx][1] * h + boxes[idx][3] * h / 2)
                    print(cx, cy)
                    print(bx,by,tx,ty)

            print(flag)
            if cx is None:
                continue
            else:
                flag = False
            time.sleep(0.1)


        ee_in_base_mat = self.controller.last_eef_pos

        d = depth_image[by:ty,bx:tx] 
        c = color_image[by:ty,bx:tx]
        # map the center to the pointcloud

        pcd, points = depth_2_point(depth_image,color_image,self.intr)
        points_in_base = trans_point_to_base(points,ee_in_base_mat)
        pcd.points = o3d.utility.Vector3dVector(points_in_base[:,:3])
        target_x, target_y = cx , int(by/7 * 5 + ty / 7 * 2)
        c_depth = depth_image[target_y,target_x] 
        c_point = single_point_to_pc(c_depth,self.intr,target_x,target_y)

        c_point_in_base = trans_point_to_base(c_point,ee_in_base_mat)

        self.reset_pose(self.reset_top_down_positions,grasp=False)


        ee_top_in_base_mat = self.controller.last_eef_pos
        delta_position = c_point_in_base[:3] - ee_top_in_base_mat[:3,3] + [0,0.01,-0.02]
        delta_pos = np.concatenate((delta_position,[0,0,0]), axis=0) 
        # delta_pos = np.concatenate((delta_position,grasp_quat), axis=0) 
        
        # print(delta_pos)
        # new_noise = np.random.normal(0,0.001,(100,4)) + c_point_in_base 
        # p = np.array(pcd.points)
        # c = np.array(pcd.colors) 
        # p = np.concatenate((p,new_noise[:,:3]),axis=0)
        # c = np.concatenate((c,[[1,0,0] for _ in range(100)]),axis=0)
        # pcd.points = o3d.utility.Vector3dVector(p)
        # pcd.colors = o3d.utility.Vector3dVector(c)
        # o3d.io.write_point_cloud("pointcloud.pcd", pcd)
        
        # self.controller.control_quat(delta_pos,grasp=False)
        self.controller.control(delta_pos,grasp=False)
        self.controller.control([0,0,0,0,0,0],grasp=True)
        self.reset_pose(self.reset_top_down_positions,grasp=True)
        self.flag = True
    
    def move_to_check(self):
        self.reset_pose(self.check_pose, grasp= False)
        time.sleep(3)
        self.reset_pose(self.reset_joint_positions, grasp= False)
    
    def move_to_pose(self):
        self.reset_pose(self.reset_drop_positions,grasp=True)
        
        # target_quat = [ 0.58876944, -0.6990674,   0.27874663,  0.29488218]

        drop_ee_in_base_mat = self.controller.last_eef_pos
        print("now the ee pose is:",drop_ee_in_base_mat)
        drop_delta_position = self.marker_pos_in_base[:3] - drop_ee_in_base_mat[:3,3] + [0,0,0.2]

        delta_pos = np.concatenate((drop_delta_position,[0,0,0]), axis=0) 
        self.controller.control(delta_pos,grasp=True)
        self.controller.control([0,0,0,0,0,0],grasp=False)

        self.reset_pose(self.reset_joint_positions,grasp=False)
        
if __name__ == "__main__":
    model_path = "models/GroundingDINO_SwinT_OGC.py"
    model_weights = "models/weights/groundingdino_swint_ogc.pth"

    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    detection_model = DINO(model_weights,model_path,BOX_TRESHOLD,TEXT_TRESHOLD)
    

    controller_type = "OSC_POSE"
    joint_controller_type = "JOINT_POSITION"
    controller_config = "charmander.yml"
    robot_interface = FrankaInterface(
        config_root + f"/{controller_config}", use_visualizer=False)
    
    controller = OSC_Control(robot_interface, controller_type)
    joint_controller = Joint_Control(robot_interface, joint_controller_type)
    
    grasp_model = GraspModel(detection_model, controller, joint_controller)

    socket_server = socket(AF_INET, SOCK_DGRAM)
    host_port = ("10.6.8.62", 8889)
    socket_server.bind(host_port)

    client_socket = socket(AF_INET, SOCK_DGRAM)
    server_host_port = ("10.6.204.19",8080)

    print("server start")
    while True:
        data = socket_server.recvfrom(1024)
        rec_data = data[0].decode("utf-8")
        command, category = rec_data.split(",")
        
        print("the received command is:",command)
        
        if command == "function_1":
            if category == "饼干":
                category = "box"
            if category == "饮料":
                category = "bottle"
            grasp_model.grasp_object(category)
            print("category")
            client_socket.sendto("finish grasp".encode("utf-8"), server_host_port)
        elif command == "function_2":
            grasp_model.move_to_pose()
            client_socket.sendto("finish drop".encode("utf-8"), server_host_port)
        elif command == "function_3":
            grasp_model.move_to_check()
            client_socket.sendto("finish check".encode("utf-8"), server_host_port)
