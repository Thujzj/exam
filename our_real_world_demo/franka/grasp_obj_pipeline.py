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

if __name__ == "__main__":
    camera = RealSenseCamera()
    intr = camera.get_intr()
    model_path = "models/GroundingDINO_SwinT_OGC.py"
    model_weights = "models/weights/groundingdino_swint_ogc.pth"
    TEXT_PROMPT = "table. bottle . pot . box . plane ."
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
    

    # reset the robot
    # reset_joint_positions = [1.4653579384030426, 
    #                         -0.8462685961556017,
    #                         0.11607228163355274, 
    #                         -2.4335851062269005,
    #                         0.028957184208089428, 
    #                         2.30431342366206, 
    #                         0.9905820587126063]
    reset_joint_positions = [-0.4536431461258938, 
                            0.8828348240099455, 
                            -0.2735634329360828,
                            -0.8908873479504179, 
                            0.16154948102765612, 
                            1.996010904291528,
                            1.0560673414274222]
    joint_controller.control(reset_joint_positions, grasp=False)
    # reset_top_down_positions = [0.13711555784303542, 
    #                             0.10420473280705903,
    #                             -0.02046572041338548,
    #                             -1.3077011456238596,
    #                             0.06041137177610551,
    #                             1.457737681892183,
    #                             1.0063563008507275]
    
    # joint_controller.control(reset_top_down_positions,grasp= False)
    time.sleep(0.5)
    print("finish reset joint pose")
    
    # avoid the error depth image
    for k in range(5):
        camera.get_sensor_info()
        
    ret, color_image, depth_image, point_cloud = camera.get_sensor_info()

    boxes, logits, phrases = detection_model.predict(color_image,TEXT_PROMPT)
    annotated_frame = detection_model.annotate(color_image,boxes,logits,phrases)
    cv2.imwrite("annotated_image.jpg", annotated_frame)
    h,w,_ = color_image.shape
    # user_input = "bottle"
    user_input = "plane"
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
    
    # z_max = np.max(points_in_base[:,2])
    # z_index = np.argmax(points_in_base[:,2])

    # new_points_index = np.where(z_max - points_in_base[:,2] < 0.05)
    
    # new_points = points_in_base[new_points_index]
    
    # print("mean is:", np.mean(new_points,axis=0))
    # print("mean x is:", np.min(new_points[:,0]))

    # l = points_in_base[z_index]
    
    # print("l is:", l)
    

    # z_min =  np.min(points_in_base[:,2])
    # z_min_index = np.argmin(points_in_base[:,2])
    
    # print(points_in_base[z_min_index])
    
    # print(index)
    # print(len(index[0]))

    # c = points_in_base[index]

    # c_point_in_base = np.mean(c,axis=0)
    # print("point is:", c_point_in_base)
    

    
    # change to the top-down grasp pose
    
    reset_top_down_positions = [1.4881616662259685, 
                                0.20876019838609192, 
                                0.26745844482738335, 
                                -1.3334160141213147, 
                                -0.032468163198894916, 
                                1.612353567444535, 
                                0.9872343113625214]
    
    joint_controller.control(reset_top_down_positions,grasp= False)
    
    time.sleep(1)
    ee_top_in_base_mat = controller.last_eef_pos
    delta_position = c_point_in_base[:3] - ee_top_in_base_mat[:3,3] + [0,0,-0.02]
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

    tmp_pose = controller.last_eef_pos
    print("delta pose is:", delta_pos)
    print("target pose is:",target_pose)
    print("tmp_pose is:",tmp_pose)
    

    
    # controller.control([0,0,0,0,0,0],grasp=True)

    # reset_top_down_positions = [0.16936004421856102, 
    #                             -0.5019268226334194, 
    #                             -0.020048975945318022, 
    #                             -1.8255751406338265, 
    #                             0.11233614461951784, 
    #                             1.3912431423114335, 
    #                             1.0151653880481089]
    
    # joint_controller.control(reset_top_down_positions,grasp= True)
    
    # time.sleep(1)