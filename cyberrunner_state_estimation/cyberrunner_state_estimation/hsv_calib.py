import numpy as np
import cv2
import os
from cyberrunner_state_estimation.core import masking
from cyberrunner_state_estimation.core.measurements import Measurements
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory


def nothing(_):
    pass

class HsvCalibration(Node):
    DEFAULT_HSV_PARAMS= (
        (43, 140),  # (minHue, maxHue)
        (125, 255),  # (minSat, maxSat)
        (9, 255),  # (minVal, maxVal)
    )

    def __init__(self, hsv_params: list = DEFAULT_HSV_PARAMS):
        super().__init__("hsv_calib")

        self.br = CvBridge()
        self.camera_localized = False

        #Instance of Measurements, used to mask/separate the labrinth board and the background
        share = get_package_share_directory("cyberrunner_state_estimation")# Location of the markers file
        markers = np.loadtxt(os.path.join(share, "markers.csv"), delimiter=",") #Loading in the markers file
        self.mask_m = Measurements(markers=markers, do_anim_3d=False)

        # Create a window for the trackbars
        cv2.namedWindow("HSV Controls")

        # Create trackbars for Hue, Saturation, and Value ranges
        cv2.createTrackbar("Min Hue", "HSV Controls", hsv_params[0][0], 179, nothing)
        cv2.createTrackbar("Max Hue", "HSV Controls", hsv_params[0][1], 179, nothing)
        cv2.createTrackbar("Min Sat", "HSV Controls", hsv_params[1][0], 255, nothing)
        cv2.createTrackbar("Max Sat", "HSV Controls", hsv_params[1][1], 255, nothing)
        cv2.createTrackbar("Min Val", "HSV Controls", hsv_params[2][0], 255, nothing)
        cv2.createTrackbar("Max Val", "HSV Controls", hsv_params[2][1], 255, nothing)

        self.subscription = self.create_subscription(Image, "cyberrunner_camera/image", self.image_callback, 10)

        self.get_logger().info("HSV Calibration node is running.")

    def image_callback(self, data):
        frame = self.br.imgmsg_to_cv2(data)

        if not self.camera_localized:
            self.mask_m.camera_localization(frame)
            self.camera_localized = True

        self.mask_m.create_mask(frame)
        #Apply the background mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=self.mask_m.mask)

        # Get current positions of all trackbars
        min_h = cv2.getTrackbarPos("Min Hue", "HSV Controls")
        max_h = cv2.getTrackbarPos("Max Hue", "HSV Controls")
        min_s = cv2.getTrackbarPos("Min Sat", "HSV Controls")
        max_s = cv2.getTrackbarPos("Max Sat", "HSV Controls")
        min_v = cv2.getTrackbarPos("Min Val", "HSV Controls")
        max_v = cv2.getTrackbarPos("Max Val", "HSV Controls")

        # Create a tuple with the HSV values from the sliders
        current_hsv_params = (
            (min_h, max_h),
            (min_s, max_s),
            (min_v, max_v)
        )

        # Apply the HSV mask
        _, mask = masking.mask_hsv(masked_frame, current_hsv_params)

        # Display the masked image
        cv2.imshow("Masked Image", mask)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    hsv_c = HsvCalibration()
    rclpy.spin(hsv_c)

    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    main()