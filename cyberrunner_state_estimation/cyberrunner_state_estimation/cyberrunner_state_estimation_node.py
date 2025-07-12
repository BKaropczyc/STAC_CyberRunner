#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cyberrunner_state_estimation.core.estimation_pipeline import EstimationPipeline
from cyberrunner_interfaces.msg import StateEstimate


class ImageSubscriber(Node):
    """A ROS Node for listening to images from the camera and publishing messages describing the system's physical state"""
    def __init__(self):
        super().__init__("cyberrunner_state_estimation")

        # Subscribe to the camera's Image messages
        self.subscription = self.create_subscription(
            Image, topic="cyberrunner_camera/image", callback=self.listener_callback, qos_profile=10
        )
        self.get_logger().info("Image subscriber has been initialized.")

        # We will publish messages of type StateEstimateSub
        self.publisher = self.create_publisher(
            StateEstimate, topic="stateEstimation", qos_profile=10
        )

        self.br = CvBridge()   # For converting ROS Image messages <-> OpenCV images

        # Create the state estimation pipeline, which does the heavy lifting
        self.estimation_pipeline = EstimationPipeline(
            fps=55.0,
            estimator="FiniteDiff",  #  "FiniteDiff",  "KF", "KFBias"
            FiniteDiff_mean_steps=4,
            print_measurements=True,
            show_3d_anim=False,
            viewpoint="top",  # 'top', 'side', 'topandside'
            show_subimage_masks=False
        )

    def listener_callback(self, data):
        """This is the method that gets called every time we receive an Image message from cyberrunner_camera/image"""

        # Convert the ROS Image message back into an OpenCV image
        frame = self.br.imgmsg_to_cv2(data)

        # Extract state information from the image
        x_hat, _, angles, xb, yb = self.estimation_pipeline.estimate(frame)

        # Fill out the message fields
        msg = StateEstimate()
        msg.x_b = x_hat[0]
        msg.y_b = x_hat[1]
        msg.x_b_dot = x_hat[2]
        msg.y_b_dot = x_hat[3]
        msg.alpha = -angles[1]
        msg.beta = angles[0]

        # Publish the message
        self.publisher.publish(msg)


def main(args=None):
    # Create and run our Node
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()
