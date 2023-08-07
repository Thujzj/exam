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

TEXT_PROMPT = "table. bottle . pot . box . " 
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

grasppoint = None

def receive_grasp_point(msg):
    global grasppoint
    grasppoint = msg.data


class GraspModel():
    def __init__(self,detection_model, controller, joint_controller):

        self.detection_model = detection_model
        self.controller = controller
        self.joint_controller = joint_controller
        self.reset_joint_positions =  [0.03260338935522349, -0.8274320986944705, 0.22758139832660154, -2.182303152151275, -0.013391416776898955, 1.8901679635128885, 1.0057438689967648]
        self.reset_top_down_positions = self.reset_joint_positions
        # self.reset_top_down_positions = [-0.25230813224093124, -0.2782045577485158, 0.18375603899247503, -1.6432780397816706, -0.07133616156246558, 1.493804914611447, 0.9094413397869002]
        
        self.reset_drop_positions = [-1.4751571585420975, -0.952519619116397, -0.521451397281415, -1.8337223887415657, -0.5761615101366534, 1.5870364188844388, 1.132414373020331]
        
        self._pose = [-0.1259001674330279, -0.34384132373960397, 0.2861217321290332, -1.822386393793915, 0.0007420955407637618, 1.7685931144937075, 0.9840520885193512]
        
        self.end_pose = [-0.27656061996493425, -0.13961071318524046, 0.6569752552360109, -1.679524105632534, -0.08407770457465895, 2.0636291815730368, 0.9029973374227045]
        self.marker_pos_in_base = None
        
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

    def reset_pose(self,pose,grasp):
        self.joint_controller.control(pose, grasp=grasp)

    
    def grasp_object(self,category):

        self.reset_pose(self.reset_joint_positions,grasp=False)
        
        
        pose_1 = [0.0942213503291897, 0.7048247435156789, 0.4200088372791708, -1.124747327101682, -0.2976054592679779, 1.8301314080546487, 1.3636636688907942]
        self.reset_pose(pose_1,grasp=False)
        self.controller.control([0,0,0,0,0,0],grasp = True)
        
        
        self.reset_pose(self._pose,grasp=True)
        
        pose_2 = [-0.10450796280528234, 0.15455887161848839, 0.4421247289348122, -1.9144783302684025, -0.05362358314461178, 1.9838960771962937, 1.2611222206339048]
        self.reset_pose(pose_2,grasp=True)
        self.controller.control([0,0,0,0,0,0],grasp = False)
        
        self.reset_pose(self._pose,grasp=False)
        
        pose_3 = [-0.12618660876625462, 0.5852904930071072, 0.28918839373923183, -1.4539743668405631, -0.06953005377341281, 2.0128375904030267, 1.0253579094608625]
        self.reset_pose(pose_3,grasp=False)
        self.controller.control([0,0,0,0,0,0],grasp = True)
        
        self.reset_pose(self._pose,grasp=True)
        
        pose_4 = [-0.06915489034694537, 0.8306049094702068, 0.6294478444471996, -1.1693849447035651, -0.43386333656947457, 1.9348026759069064, 1.3438346611042746]
        self.reset_pose(pose_4,grasp=True)
        self.controller.control([0,0,0,0,0,0],grasp = False)
        
        self.reset_pose(self._pose,grasp=False)
        
        pose_5 = [-0.050838743305569, 0.11598608589784747, 0.38247137092038547, -1.978096634907566, -0.030478437991605864, 2.0903877091672687, 1.207892363025573]
        self.reset_pose(pose_5,grasp=False)
        self.controller.control([0,0,0,0,0,0],grasp = True)
        self.reset_pose(self._pose,grasp=True)
        
        pose_6 = [-0.19063879784784818, 0.5119840804317541, 0.3864292414995272, -1.429709260070533, -0.18680307185992076, 1.958973376777437, 1.033012897432264]
        
        self.reset_pose(pose_6,grasp=True)
        self.controller.control([0,0,0,0,0,0],grasp = False)
        
        self.reset_pose(self._pose,grasp=False)

        
    def move_to_pose(self):
        self.reset_pose(self.reset_drop_positions,grasp=True)
        
        # target_quat = [ 0.58876944, -0.6990674,   0.27874663,  0.29488218]

        # drop_ee_in_base_mat = self.controller.last_eef_pos
        # print("now the ee pose is:",drop_ee_in_base_mat)
        # drop_delta_position = self.marker_pos_in_base[:3] - drop_ee_in_base_mat[:3,3] + [0,0,0.2]

        # delta_pos = np.concatenate((drop_delta_position,[0,0,0]), axis=0) 
        # self.controller.control(delta_pos,grasp=True)
        self.controller.control([0,0,0,0,0,0],grasp=False)

        self.reset_pose(self.reset_joint_positions,grasp=False)
        
if __name__ == "__main__":
    model_path = "models/GroundingDINO_SwinT_OGC.py"
    model_weights = "models/weights/groundingdino_swint_ogc.pth"

    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    # detection_model = DINO(model_weights,model_path,BOX_TRESHOLD,TEXT_TRESHOLD)
    

    controller_type = "OSC_POSE"
    joint_controller_type = "JOINT_POSITION"
    controller_config = "charmander.yml"
    robot_interface = FrankaInterface(
        config_root + f"/{controller_config}", use_visualizer=False)
    
    controller = OSC_Control(robot_interface, controller_type,num_steps=10)
    joint_controller = Joint_Control(robot_interface, joint_controller_type)
    
    grasp_model = GraspModel(None, controller, joint_controller)

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

            # client_socket.sendto("finish grasp".encode("utf-8"), server_host_port)
        elif command == "function_2":
            grasp_model.move_to_pose()
            client_socket.sendto("finish drop".encode("utf-8"), server_host_port)

