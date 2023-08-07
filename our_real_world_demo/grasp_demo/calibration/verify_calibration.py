import rospy
from geometry_msgs.msg import PoseStamped
import tf
import sys
sys.path.append("./")
from control.franka_control import OSC_Control,Joint_Control
from deoxys.utils.transform_utils import quat2mat,pose2mat,mat2pose,pose_in_A_to_pose_in_B,quat2axisangle,axisangle2quat
import time
from deoxys.franka_interface import FrankaInterface
from deoxys import config_root

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




def deal_pose(event):
    # print(pos)
    # print(ori)
    obj_in_cam_mat = pose2mat([pos, ori])
    obj_in_base_mat = pose_in_A_to_pose_in_B(obj_in_cam_mat, camera_in_base_mat)
    print("obj_in_base_mat is: ", obj_in_base_mat)

if __name__ == "__main__":
    pos_camera_to_ee = [0.0467162 , -0.0388289 , -0.0419308]  
    
# new quat is [x,y,z,w] 
    new_quat_camera_to_ee = [-0.008896206339478014, 0.001529932861133454, 0.7130445599416992, 0.7010606053371948] 
#     pos_camera_to_ee = [0.0514359,-0.0396951,-0.0430446]  
    
# # new quat is [x,y,z,w] 
#     new_quat_camera_to_ee = [-0.0010600823690949102, -0.0010874011732674438, 0.7088384291052459, 0.7053692474212825]


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
    time.sleep(2)
    while controller.robot_interface.state_buffer_size == 0:
        controller.logger.warn("Robot state not received")
        time.sleep(0.5)

    rospy.init_node('test_calibration')
    rospy.Subscriber('aruco_single/pose', PoseStamped, receive_pose)
    # rospy.Timer(rospy.Duration(0.1), deal_pose)
    
    time.sleep(2)


    ee_in_base_mat = controller.last_eef_pos

    print(ee_in_base_mat)
    camera_in_ee_mat = pose2mat([pos_camera_to_ee, new_quat_camera_to_ee])

    print(camera_in_ee_mat)

    camera_in_base_mat = pose_in_A_to_pose_in_B(camera_in_ee_mat, ee_in_base_mat)

    print("camera_in_base_mat is: ", camera_in_base_mat)


    obj_in_cam_mat = pose2mat([pos, ori])
    obj_in_base_mat = pose_in_A_to_pose_in_B(obj_in_cam_mat, camera_in_base_mat)
    print("obj_in_base_mat is: ", obj_in_base_mat)
    target_in_base_pos, target_in_base_ori = mat2pose(obj_in_base_mat)
    print("target obj in base pos is: ", target_in_base_pos)
    ee_in_base_pos, ee_in_base_ori = mat2pose(ee_in_base_mat)
    print("ee in base pos is:", ee_in_base_pos)
    
    delta_pose = [target_in_base_pos[0] - ee_in_base_pos[0], target_in_base_pos[1] - ee_in_base_pos[1], \
        target_in_base_pos[2] - ee_in_base_pos[2] ,0,0,0]
    
    # delta_pose = [
    #     -0.15,0.05,0.2,0.5,0.1,0
    # ]
    
    print("the delta pose is: ", delta_pose)
    
    
    controller.control(delta_pose, grasp= False)
    ee_end_pos_mat = controller.last_eef_pos
    ee_pos, _ = mat2pose(ee_end_pos_mat)
    print("desired pose is:", target_in_base_pos)
    print("ee pose is:", ee_pos)


    time.sleep(3)
    ee_end_pos_mat = controller.last_eef_pos
    ee_pos, _ = mat2pose(ee_end_pos_mat)
    print("ee pose is:", ee_pos)
    rospy.spin()
    