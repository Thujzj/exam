
from sensors.realsense import RealSenseCamera
from models.detection import Detection,DINO
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import numpy as np

TEXT_PROMPT = "table. bottle . pot . box ." 

if __name__ == "__main__":
    br = CvBridge()
    pub = rospy.Publisher("/realsense_camera", Image, queue_size=1)
    rospy.init_node("realsense_camera")
    
    
    camera = RealSenseCamera()
    intr = camera.get_intr()
    i = 0 
    model_path = "models/GroundingDINO_SwinT_OGC.py"
    model_weights = "models/weights/groundingdino_swint_ogc.pth"

    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    detection_model = DINO(model_weights,model_path,BOX_TRESHOLD,TEXT_TRESHOLD)
    cv2.namedWindow('raw image', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('detection results', cv2.WINDOW_AUTOSIZE)
    while True:
        ret, color_image, depth_image, point_cloud = camera.get_sensor_info()
        
        boxes, logits, phrases = detection_model.predict(color_image,TEXT_PROMPT)
        for i, text in enumerate(phrases):
            if "box" in text:
                phrases[i] = "cookie box"
            if "bottle" in text:
                phrases[i] = "gruel bottle"
                
        
        annotated_frame = detection_model.annotate(color_image,boxes,logits,phrases)
        
        cv2.imshow('raw image', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        cv2.imshow('detection results', annotated_frame )
        
        np.save("output/depth_image.npy",depth_image)
        np.save("output/color_image.npy",color_image)
        key = cv2.waitKey(1)
        pub.publish(br.cv2_to_imgmsg(color_image, "rgb8"))
