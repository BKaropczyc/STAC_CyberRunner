#!usr/bin/env python3

import os
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cyberrunner_interfaces.msg import StateEstimateSub
from cv_bridge import CvBridge

from cyberrunner_state_estimation.core.measurements import Measurements
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class ImageSubscriber(Node):
    """A ROS Node for listening to images from the camera and publishing messages describing the system's physical state"""

    # DEBUGGING OPTIONS
    # Set PRINT_MEASUREMENTS to True to display the measurements extracted from each frame
    # Set BROADCAST_TRANSFORMS to True to broadcast our frame transforms using the ROS tf system
    PRINT_MEASUREMENTS = False
    BROADCAST_TRANSFORMS = False

    def __init__(self, skip=1):
        super().__init__("cyberrunner_state_estimation")

        # Subscribe to the camera's Image messages
        self.subscription = self.create_subscription(
            Image, topic="cyberrunner_camera/image", callback=self.listener_callback, qos_profile=10
        )
        self.get_logger().info("Image subscriber has been initialized.")

        # We will publish messages of type StateEstimateSub
        self.publisher = self.create_publisher(
            StateEstimateSub, topic="cyberrunner_state_estimation/estimate_subimg", qos_profile=10
        )

        if ImageSubscriber.BROADCAST_TRANSFORMS:
            # Create our coordinate space transform broadcasters
            self.tf_static_broadcaster = StaticTransformBroadcaster(self)
            self.tf_broadcaster = TransformBroadcaster(self)

        self.br = CvBridge()   # For converting ROS Image messages <-> OpenCV images

        # Read in the markers.csv data generated during the "select_markers" calibration step
        share = get_package_share_directory("cyberrunner_state_estimation")
        markers = np.loadtxt(os.path.join(share, "markers.csv"), delimiter=",")

        # Create the measurements object, which does the heavy lifting of state estimation
        self.measurements = Measurements(
            markers=markers,
            show_3d_anim=False,         # Whether to display a 3D visualization of the estimated physical state
            viewpoint="top",            # The view to use for the 3D visualization: 'top', 'side', or 'topandside'
            show_subimage_masks=False   # Whether the Detector object should show the subimage masks
        )

        if ImageSubscriber.PRINT_MEASUREMENTS:
            np.set_printoptions(precision=3, floatmode="fixed", suppress=True, sign=" ", nanstr="  nan ")

        self.skip = skip   # Number of frames to skip when publishing (1 = publish every frame)
        self.count = 0     # Count of frames processed

    def listener_callback(self, data):
        """This is the method that gets called every time we receive an Image message from cyberrunner_camera/image"""

        # Convert the ROS Image message back into an OpenCV image
        frame = self.br.imgmsg_to_cv2(data)

        # If we haven't fully localized the camera yet, use this frame for camera localization
        if not self.measurements.camera_localized:
            self.measurements.camera_localization(frame)
            return

        # Extract state information from the image
        self.measurements.process_frame(frame, get_ball_subimg=True)
        # ...and get the results
        ball_pos = self.measurements.get_ball_position_in_maze()[:2]   # We only care about the X and Y coords
        board_angles = self.measurements.get_plate_pose()
        subimg = self.measurements.get_ball_subimg()

        # Print out the calculated measurements
        if self.PRINT_MEASUREMENTS:
            print(f"ball_pos: {ball_pos} (m)  |  board_angles: {np.rad2deg(board_angles)} (deg)")

        # Only publish every <skip> messages
        if self.count % self.skip == 0:
            # Fill out the message fields
            msg = StateEstimateSub()
            msg.state.x_b = ball_pos[0]
            msg.state.y_b = ball_pos[1]
            msg.state.alpha = board_angles[0]
            msg.state.beta = board_angles[1]
            msg.subimg = self.br.cv2_to_imgmsg(subimg)

            # Publish the message
            self.publisher.publish(msg)

        # Broadcast coordinate space transforms
        if ImageSubscriber.BROADCAST_TRANSFORMS:
            # The Camera -> World transform is assumed to be static,
            # and thus it is only broadcast ONCE (on the first message)
            if self.count == 0:
                t = self.transform_matrix_to_msg(
                    self.measurements.plate_pose.T__W_C,
                    frame_id='world',
                    child_frame_id='camera'
                )
                self.tf_static_broadcaster.sendTransform(t)

            # The other transforms change with each new state,
            # and thus must be continuously broadcast.
            transform_msgs = []

            # Maze -> World transform
            t_maze = self.transform_matrix_to_msg(
                self.measurements.plate_pose.T__W_M,
                frame_id='world',
                child_frame_id='maze'
            )
            transform_msgs.append(t_maze)

            # Maze -> Ball transform
            # Only broadcast this transform if we know where the ball is currently located in the maze frame
            ball_pos = self.measurements.get_ball_position_in_maze()
            if np.all(np.isfinite(ball_pos)):
                # This is a translation-only transform
                T__M_B = np.eye(4)
                T__M_B[:3, -1] = ball_pos

                t_ball = self.transform_matrix_to_msg(
                    T__M_B,
                    frame_id='maze',
                    child_frame_id='ball'
                )
                transform_msgs.append(t_ball)

            # Broadcast the dynamic transforms
            self.tf_broadcaster.sendTransform(transform_msgs)

        # We've processed one more frame
        self.count += 1

    def transform_matrix_to_msg(self, se3, frame_id, child_frame_id):
        """Convert a given coordinate space transform (4x4 matrix) to a TransformStamped ROS message"""
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id

        # Translation
        t.transform.translation.x = se3[0, 3]
        t.transform.translation.y = se3[1, 3]
        t.transform.translation.z = se3[2, 3]

        # Rotation
        q = Rotation.from_matrix(se3[:3, :3]).as_quat()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        return t


def main(args=None):
    # Create and run our Node
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()
