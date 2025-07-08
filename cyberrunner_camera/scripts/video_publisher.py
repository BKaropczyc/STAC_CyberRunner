#!/usr/bin/env python3
import sys
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class VideoPublisher(Node):
    """
    This node provides functionality for publishing the frames of a video file instead of the USB camera.
    It is useful when developing code without physical access to the robot, and when debugging issues
    with the state-estimation code.
    """

    EXPECTED_VIDEO_WIDTH = 640
    EXPECTED_VIDEO_HEIGHT = 360
    MIN_EXPECTED_VIDEO_FPS = 50
    MAX_EXPECTED_VIDEO_FPS = 60

    def __init__(self, video_file, repeat=True, display=True):
        super().__init__("cyberrunner_camera")

        # Create our publisher and OpenCV bridge
        self.publisher = self.create_publisher(Image, "cyberrunner_camera/image", 1)
        self.br = CvBridge()

        # Open the video file
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_file}")
            sys.exit(1)

        # Verify the video meets our requirements
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (video_width == self.EXPECTED_VIDEO_WIDTH) and (video_height == self.EXPECTED_VIDEO_HEIGHT):
            scale = False
        elif (video_width == 2 * self.EXPECTED_VIDEO_WIDTH) and (video_height == 2 * self.EXPECTED_VIDEO_HEIGHT):
            # We'll need to resize the video
            scale = True
        else:
            print(f"Error: The video file must have a resolution of either {self.EXPECTED_VIDEO_WIDTH}x{self.EXPECTED_VIDEO_HEIGHT} "
                  f"or {2 * self.EXPECTED_VIDEO_WIDTH}x{2 * self.EXPECTED_VIDEO_HEIGHT}.\n"
                  f"The provided video has a resolution of {video_width}x{video_height}.")
            sys.exit(1)

        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if (video_fps < self.MIN_EXPECTED_VIDEO_FPS) or (video_fps > self.MAX_EXPECTED_VIDEO_FPS):
            print("WARNING: For learning, the video signal should have a frame rate of approximately 55 fps.\n"
                  f"The provided video has a frame rate of {video_fps} fps.")

        # Define a publish rate for this node based on the FPS of the video file
        self.publish_rate = self.create_rate(video_fps)

        # Call spin() on this node in a separate thread
        # The node must be constantly processing events for rate.sleep() to work correctly
        thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        thread.start()

        # Publish the provided video frame by frame
        while True:
            # Read the next frame of the video
            ret, frame = cap.read()

            # If we reached the end of the video...
            if not ret:
                # If we should repeat, re-open the file. Otherwise, exit.
                if repeat:
                    cap = cv2.VideoCapture(video_file)
                    ret, frame = cap.read()
                else:
                    break

            # Resize the video if necessary
            if scale:
                frame = cv2.resize(frame, dsize=(self.EXPECTED_VIDEO_WIDTH, self.EXPECTED_VIDEO_HEIGHT))

            # Add the standard border on the top and bottom
            frame = cv2.copyMakeBorder(frame, 20, 20, 0, 0, cv2.BORDER_CONSTANT, 0)

            # Convert the frame to a ROS image message and publish it
            msg = self.br.cv2_to_imgmsg(frame)
            self.publisher.publish(msg)

            # Display the frame
            if display:
                cv2.imshow("Video", frame)
                cv2.waitKey(1)

            # Wait to publish the next frame to maintain our desired rate
            self.publish_rate.sleep()


def main(args=None):
    # Ensure a video file was specified
    if len(sys.argv) < 2:
        print(f"Usage: Supply the name of the video file as a command-line argument, e.g.,\n"
               "ros2 run cyberrunner_camera video_publisher.py <path/to/video/file> [norepeat] [nodisplay]")
        sys.exit(1)

    # Get the name of the video file to use
    video_file = sys.argv[1]
    print(f"Using video file: {video_file}")

    # Check for 'norepeat' and 'nodisplay' flags
    repeat = not("norepeat" in sys.argv[2:])
    display = not("nodisplay" in sys.argv[2:])

    # Start our ROS node
    rclpy.init(args=args)
    _ = VideoPublisher(video_file, repeat, display)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
