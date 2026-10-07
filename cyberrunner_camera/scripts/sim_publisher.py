#!/usr/bin/env python3
import time
import sys
import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
from cyberrunner_interfaces.msg import DynamixelVel


class SimMaster(Node):
    def __init__(self, display=True):
        super().__init__("cyberrunner_sim")
        # Mujoco and other instance variables
        self.model = mujoco.MjModel.from_xml_path("cyberrunner_mujoco/cyberrunner.xml")
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        self.br = CvBridge()
        
        # Keep track of how quickly we're publishing images
        frame_count = 0
        previous = time.time()

        # Publishing location
        self.publisher = self.create_publisher(Image, "cyberrunner_camera/image", 1)

        # Subscribe to the control message topic, call the callback function to get control commands
        self.subscription = self.create_subscription(DynamixelVel, "cyberrunner_dynamixel/cmd", self.control_callback, 1)
        self.command = (0.0, 0.0)

        # Call spin() on this node in a separate thread
        thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        thread.start()
        while True:
            # Apply dreamerV3 requested commands
            self.data.ctrl[:] = self.command

            # Apply above actuator inputs
            mujoco.mj_step(self.model, self.data, nstep=36)

            # Read in the next image after changes are applied
            self.renderer.update_scene(self.data, camera="fov_camera")
            frame = self.renderer.render()

            # Resize the image and add a border to the top & bottom
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.copyMakeBorder(frame, 20, 20, 0, 0, cv2.BORDER_CONSTANT, 0)

            # state estimation expects BGR, renderer.render() gives RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Publish this image as a ROS message
            msg = self.br.cv2_to_imgmsg(frame)
            self.publisher.publish(msg)

            # Display the image, if desired
            if display:
                cv2.imshow("Camera", frame)
                cv2.waitKey(1)
            frame_count += 1

            # Check the image processing speed
            now = time.time()
            dur = now - previous
            if dur >= 2.0:     # Calculate the fps every ~2 seconds
                fps = frame_count / dur
                if fps < 50.0:
                    print(f"WARNING: Slow image generation: {fps:0.2f} fps")

                # Reset our count
                frame_count = 0
                previous = now

    # Callback function to receive commands from the topic for control messages requested by dreamer
    def control_callback(self, msg):
        self.command = (msg.vel_1, msg.vel_2)
    
def main(args=None):
    # Check for 'nodisplay' flag
    args = sys.argv[1:]
    display = not("nodisplay" in args)
    if "nodisplay" in args:
        args.remove("nodisplay")

    # Start our ROS node
    rclpy.init(args=args)
    _ = SimMaster(display)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
