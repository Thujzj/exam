

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

    for k in range(5):
        camera.get_sensor_info()
    
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
    
    
    rospy.spin()