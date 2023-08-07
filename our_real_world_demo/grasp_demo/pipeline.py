from models.detection import Detection
from models.action_model import ActionModel
from control.franka_control import FrankaControl
from sensors.kinect import RGB_Depth_Kinect
from sensors.realsense import RealSenseCamera
import cv2


def main():
    # Initialize camera
    # camera = RGB_Depth_Kinect()
    camera = RealSenseCamera()
    # Initialize detection model
    detection_model = Detection()
    # cv2.namedWindow('Transformed Color Image',cv2.WINDOW_NORMAL)
    while True:
        ret, color_image, depth_image, pointcloud = camera.get_sensor_info()
        print(ret)
        if not ret:
            continue
        color_image = color_image[:,:,:3]
        prediction = detection_model.predict(color_image)
        detection_model.display(color_image,prediction)
        
        # cv2.imshow('Transformed Color Image',color_image)
        # if cv2.waitKey(1) == ord('q'): 
        #     break

    # Initialize action model
    
    # Initialize franka control
    
    # Run pipeline

if __name__ == "__main__":
    main()
