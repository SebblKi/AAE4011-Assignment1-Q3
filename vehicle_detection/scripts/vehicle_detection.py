#!/home/ubuntu/miniconda3/envs/yolo/bin/python

import rospy
from sensor_msgs.msg import CompressedImage, Image
import numpy as np
import cv2
from ultralytics import YOLO


class VehicleDetector:

    def __init__(self):

        rospy.init_node("vehicle_detector")

        # Load YOLO model
        self.model = YOLO("yolov8n.pt")

        # Subscribe to compressed camera topic
        self.image_sub = rospy.Subscriber(
            "/hikcamera/image_2/compressed",
            CompressedImage,
            self.callback,
            queue_size=1
        )

        # Publish annotated detections
        self.image_pub = rospy.Publisher(
            "/vehicle_detection/image",
            Image,
            queue_size=1
        )

        rospy.loginfo("Vehicle detector node started.")


    def callback(self, msg):

        try:
            # Convert ROS compressed image to OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                rospy.logwarn("Failed to decode image")
                return

            # Run YOLO detection
            results = self.model(frame)

            annotated = results[0].plot()

            # Create ROS Image message manually
            ros_img = Image()
            ros_img.header = msg.header
            ros_img.height = annotated.shape[0]
            ros_img.width = annotated.shape[1]
            ros_img.encoding = "bgr8"
            ros_img.is_bigendian = False
            ros_img.step = annotated.shape[1] * 3
            ros_img.data = annotated.tobytes()

            # Publish annotated image
            self.image_pub.publish(ros_img)

        except Exception as e:
            rospy.logerr("Detection callback error: %s", str(e))


if __name__ == "__main__":

    detector = VehicleDetector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down vehicle detector node.")
