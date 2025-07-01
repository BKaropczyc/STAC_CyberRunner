#!/usr/bin/env python3
import time
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CamPublisher(Node):
    def __init__(self, device):
        super().__init__("cyberrunner_camera")

        self.publisher = self.create_publisher(Image, "cyberrunner_camera/image", 1)
        self.br = CvBridge()
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print(f"Cannot open device: {device}")
            exit()

        # Setup the video capture settings
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("U", "Y", "V", "Y"))
        self.cap.set(cv2.CAP_PROP_MODE, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 60)  # 60
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 1280
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 720
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

        # Keep track of how quickly we're publishing images
        frame_count = 0
        previous = time.time()

        while True:
            # Read in the next image
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to obtain camera image. Exiting...")
                exit()

            # Resize the image and add a border to the top & bottom
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.copyMakeBorder(frame, 20, 20, 0, 0, cv2.BORDER_CONSTANT, 0)

            # Publish this image as a ROS message
            msg = self.br.cv2_to_imgmsg(frame)
            self.publisher.publish(msg)

            # Display the image, if desired
            debug = True
            if debug:
                cv2.imshow("img", frame)  # cv2.resize(frame, (160, 100)))
                cv2.waitKey(1)

            frame_count += 1

            # Check the image processing speed
            now = time.time()
            dur = now - previous
            if dur >= 2.0:     # Calculate the fps every ~2 seconds
                fps = frame_count / dur
                if fps < 50.0:
                    print(f"WARNING: Slow processing: {fps:0.2f} fps")

                # Reset our count
                frame_count = 0
                previous = now


    def cap(self):
        pass


def main(args=None):
    args = sys.argv[1:]
    device = "/dev/video0" if not args else args[0]
    print("Using device: {}. To use a different device use the command-line argument, e.g.,\n"
          "ros2 run cyberrunner_camera cam_publisher.py /dev/video*".format(device))

    rclpy.init(args=args)
    vid = CamPublisher(device)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
