#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class SimPublisher(Node):
    def __init__(self, display=True):
        # Intializes ROS Node
        super().__init__("cyberrunner_sim")

        # Registers the publishing endpoint of the node just intialized
        self.publisher = self.create_publisher(Image, "cyberrunner_camera/image", 1)
        self.br = CvBridge()
        
        # Keep track of how quickly we're publishing images
        self.frame_count = 0
        self.previous = time.time()
        self.display = display

    def publish_frame(self, frame):
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.copyMakeBorder(frame, 20, 20, 0, 0, cv2.BORDER_CONSTANT, 0)

        # Publish this image as a ROS message
        msg = self.br.cv2_to_imgmsg(frame)
        self.publisher.publish(msg)

        # Display the image, if desired
        if self.display:
            cv2.imshow("Camera", frame)
            cv2.pollKey()

        self.frame_count += 1

        # Check the image processing speed
        now = time.time()
        dur = now - self.previous
        if dur >= 2.0:     # Calculate the fps every ~2 seconds
            fps = self.frame_count / dur
            if fps < 50.0:
                print(f"WARNING: Slow image generation: {fps:0.2f} fps")

            # Reset our count
            self.frame_count = 0
            self.previous = now

